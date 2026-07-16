/**
 * Main-thread kernel for multicore process management.
 *
 * The Kernel runs on the main JS thread (which also owns the terminal and the Swift bridge).
 * Each emulated Linux process runs in its own Web Worker with JIT-compiled WASM.
 *
 * Responsibilities:
 *   - Spawn/terminate Workers for fork/exec/exit
 *   - Manage the process table in SharedArrayBuffer
 *   - Route stdin from terminal to the foreground process's stdin ring
 *   - Route stdout/stderr from Workers to terminal
 *   - Relay bridge calls (native filesystem) from Workers to Swift
 *   - Deliver signals (Ctrl+C → SIGINT)
 */

import type { KernelMessage, WorkerInitConfig, SparseMemSnapshot } from './kernel-ipc.js';
import {
  KERNEL_SAB_SIZE,
  PROC_TABLE_OFFSET, PROC_SLOT_SIZE, PROC_PID, PROC_PPID, PROC_PGID,
  PROC_STATE, PROC_EXIT_CODE, PROC_WAIT_FUTEX, PROC_SIGNAL, PROC_STDIN_RING,
  PROC_REPARENTED, PROC_TGID, PROC_EMPTY, PROC_RUNNING, PROC_ZOMBIE,
  MAILBOX_OFFSET, MAILBOX_SLOT_SIZE, MBOX_READY, MBOX_VALUE0,
  GLOBAL_STATE_OFFSET, GLOBAL_NEXT_PID, GLOBAL_NEXT_PIPE, GLOBAL_NEXT_STDIN, GLOBAL_FG_PGID,
  STDIN_RING_OFFSET, STDIN_RING_SIZE,
  SIG_INT,
  BRIDGE_RESP_OFFSET, BRIDGE_RESP_SLOT_SIZE,
  PIPE_RING_COUNT, PIPE_RING_SIZE,
  procSlotOffset, mailboxSlotOffset, stdinRingOffset, pipeRingOffset,
} from './kernel-ipc.js';
import { PipeRing } from './pipe-ring.js';
import type { NativeBridge } from './bridge.js';

// WORKER POOL. guest processes are real Workers, and destroying one only frees its memory via GC
// — under a no-idle fork storm GC never catches up and dead Workers pile up into a jetsam OOM.
// so we DON'T destroy them: a fixed pool of long-lived Workers is REUSED across processes. exit
// returns a Worker to the idle pool (memory wiped, in place — wasm can't shrink, so reuse beats
// realloc); fork grabs an idle one (or creates up to POOL_MAX). no terminate on the hot path ->
// nothing for GC to chase. bounds:
const WORKER_CAP = 12;      // max concurrently-LIVE processes (the memory-critical bound). fork
                            // over this -> EAGAIN (pid 1 is exempt so the shell never stalls).
const POOL_MAX = 16;        // hard ceiling on TOTAL Workers (live + idle), incl. pid-1 headroom.
const POOL_IDLE_KEEP = 3;   // idle Workers to keep warm; excess trimmed during calm (GC has time).
const IDLE_TRIM_MS = 3000;  // how often to trim the idle pool back toward POOL_IDLE_KEEP.

// a threaded process's shared memory spans the guest (1.5 GiB at 0x60000000, == single-threaded) plus
// the per-thread register blocks (at 0x60000000) and the shared atomics shadow bitmap (at 0x60400000,
// 3 MiB). 0x6080 pages = 1.508 GiB covers through the shadow. lazily committed, so untouched pages cost
// no RSS; only the `maximum` reserves virtual address space. sequential threaded processes recycle the
// single spare (see spareSharedMem) so the reservation isn't re-taken per launch; ~2-3 threaded
// processes can be live CONCURRENTLY under the platform's shared-memory pool.
const THREAD_REGS_BASE = 0x60000000;
const THREADED_MEM_PAGES = 0x6080; // 24704 wasm pages (~1.508 GiB)

export interface KernelConfig {
  /** The ELF binary to boot (busybox) */
  elfBytes: Uint8Array;
  /** URL of the worker script (worker.js) */
  workerURL: string;
  /** Callback for terminal output */
  onOutput: (data: Uint8Array) => void;
  /** Optional native bridge for Swift filesystem */
  bridge?: NativeBridge;
  /** Optional zig translator wasm — when supplied, all workers use zig with ts fallback */
  zigTranslatorBytes?: ArrayBuffer;
}

export class Kernel {
  private sab: SharedArrayBuffer;
  private procTable: Int32Array;
  private globalState: Int32Array;
  private workers = new Map<number, Worker>();
  // threaded processes: the kernel (main thread) owns each thread-group's shared memory (only the main
  // thread can allocate shared WebAssembly.Memory on WKWebView/JSC). keyed by tgid, freed when the
  // group's last thread exits. `pendingPromotes` bridges the async first-thread handshake
  // (clone_thread -> promote -> thread_go); `threadCounts` tracks live threads per group for cleanup.
  private sharedMems = new Map<number, WebAssembly.Memory>();
  private threadCounts = new Map<number, number>();
  private threadedLeaders = new Set<number>(); // pids that own a thread-group's shared memory
  // one parked shared memory kept for reuse by the next threaded process (avoids alloc/GC churn).
  // `high` is the previous tenant's touched high-water, so the reuser only has to zero the stale span.
  private spareSharedMem: { mem: WebAssembly.Memory; high: number } | null = null;
  private pendingPromotes = new Map<number, { memory: WebAssembly.Memory; tid: number; slot: number; regBaseAddr: number; tgid: number; pgid: number }>();
  private elfBytes: Uint8Array;
  private workerURL: string;
  private onOutput: (data: Uint8Array) => void;
  private bridge?: NativeBridge;
  private zigTranslatorBytes?: ArrayBuffer;
  private zigTranslatorModule?: WebAssembly.Module;

  // idle Workers parked for reuse (see WORKER POOL). live Workers are in `this.workers`.
  private idleWorkers: Worker[] = [];

  // POSIX advisory record locks (fcntl F_SETLK/F_GETLK/F_SETLKW). each process is its own Worker,
  // so the lock authority must be central: this shared table is keyed by resolved file path. a
  // range is [start, end); end === Infinity means "to EOF" (l_len 0). locks are owned by pid and
  // released when the owner exits (handleExit) or explicitly unlocks.
  private recordLocks = new Map<string, Array<{ pid: number; write: boolean; start: number; end: number }>>();
  // processes parked in F_SETLKW waiting for a conflicting lock to clear.
  private lockWaiters: Array<{ pid: number; slot: number; path: string; write: boolean; start: number; end: number }> = [];

  // Stdin ring for the foreground process
  private fgStdinRing: PipeRing | null = null;

  private termCols = 80;
  private termRows = 24;

  constructor(config: KernelConfig) {
    this.elfBytes = config.elfBytes;
    this.workerURL = config.workerURL;
    this.onOutput = config.onOutput;
    this.bridge = config.bridge;
    this.zigTranslatorBytes = config.zigTranslatorBytes;
    // compile translator.wasm ONCE here; every worker gets this shared module (cloned code)
    // instead of re-compiling 1.5MB each — the per-spawn compile spike that made fork storms OOM.
    if (config.zigTranslatorBytes) {
      try {
        this.zigTranslatorModule = new WebAssembly.Module(new Uint8Array(config.zigTranslatorBytes));
      } catch (e) {
        console.error('[kernel] translator.wasm compile failed, falling back to per-worker bytes:', e);
      }
    }

    // Allocate kernel shared memory
    this.sab = new SharedArrayBuffer(KERNEL_SAB_SIZE);
    this.procTable = new Int32Array(this.sab);
    this.globalState = new Int32Array(this.sab, GLOBAL_STATE_OFFSET, 16);

    // Initialize global state
    Atomics.store(this.globalState, GLOBAL_NEXT_PID, 2); // PID 1 = init process
    Atomics.store(this.globalState, GLOBAL_NEXT_STDIN, 0);
    Atomics.store(this.globalState, GLOBAL_NEXT_PIPE, 0);
    Atomics.store(this.globalState, GLOBAL_FG_PGID, 1);

    // trim the idle Worker pool back toward POOL_IDLE_KEEP during calm. under a fork storm the
    // idle pool stays near-empty (workers are reused as fast as they're freed), so this only bites
    // once forking stops — when GC has time and terminating the surplus is cheap. this reclaims
    // the ~POOL_MAX x 32MB the pool would otherwise hold between bursts.
    setInterval(() => this.trimIdle(), IDLE_TRIM_MS);
  }

  private trimIdle(): void {
    while (this.idleWorkers.length > POOL_IDLE_KEEP) {
      this.idleWorkers.pop()!.terminate();
    }
  }

  /** Boot the initial shell process (PID 1). */
  boot(argv: string[] = ['sh'], envp: string[] = ['HOME=/root', 'TERM=xterm-256color', 'PATH=/bin:/usr/bin', 'PWD=/']): void {
    // Allocate stdin ring for the init process
    const stdinIdx = this.allocStdinRing();
    this.fgStdinRing = new PipeRing(this.sab, stdinRingOffset(stdinIdx), STDIN_RING_SIZE);
    this.fgStdinRing.reset(1, 1); // 1 writer (main thread), 1 reader (shell worker)

    this.spawnProcess({
      pid: 1,
      ppid: 0,
      pgid: 1,
      slot: 0,
      stdinRing: stdinIdx,
      elfBytes: this.elfBytes,
      argv,
      envp,
    });
  }

  /** Send keyboard input to the foreground process's stdin ring. */
  pushStdin(data: Uint8Array): void {
    if (this.fgStdinRing) {
      this.fgStdinRing.write(data);
    }
  }

  setTermSize(cols: number, rows: number): void {
    if (cols < 1 || rows < 1) return;
    if (cols === this.termCols && rows === this.termRows) return;
    this.termCols = cols;
    this.termRows = rows;
    for (const w of this.workers.values()) {
      w.postMessage({ type: 'term_size', cols, rows });
    }
  }

  /** Send Ctrl+C (SIGINT) to the foreground process group. */
  sendInterrupt(): void {
    const fgPgid = Atomics.load(this.globalState, GLOBAL_FG_PGID);
    for (let slot = 0; slot < 64; slot++) {
      const slotBase = procSlotOffset(slot) / 4;
      const state = Atomics.load(this.procTable, slotBase + PROC_STATE);
      if (state === PROC_RUNNING) {
        const pgid = Atomics.load(this.procTable, slotBase + PROC_PGID);
        if (pgid === fgPgid) {
          Atomics.or(this.procTable, slotBase + PROC_SIGNAL, SIG_INT);
          // Wake the Worker from any Atomics.wait it might be in
          // 1. Bridge call mailbox
          const mailbox = new Int32Array(this.sab, mailboxSlotOffset(slot), MAILBOX_SLOT_SIZE / 4);
          Atomics.notify(mailbox, MBOX_READY);
          // 2. Stdin ring (blocked read waits on the ring's write pointer)
          const stdinIdx = Atomics.load(this.procTable, slotBase + 7); // PROC_STDIN_RING
          if (stdinIdx >= 0) {
            const ringI32 = new Int32Array(this.sab, stdinRingOffset(stdinIdx), 4);
            Atomics.notify(ringI32, 0);
          }
        }
      }
    }
  }

  // ── Process management ────────────────────────────────────────────────────

  private allocPid(): number {
    return Atomics.add(this.globalState, GLOBAL_NEXT_PID, 1);
  }

  private allocStdinRing(): number {
    return Atomics.add(this.globalState, GLOBAL_NEXT_STDIN, 1);
  }

  private findFreeSlot(): number {
    for (let slot = 0; slot < 64; slot++) {
      const state = Atomics.load(this.procTable, (procSlotOffset(slot) / 4) + PROC_STATE);
      if (state === PROC_EMPTY) return slot;
    }
    return -1; // table full — caller turns this into a fork() EAGAIN, never a crash
  }

  private spawnProcess(opts: {
    pid: number;
    ppid: number;
    pgid: number;
    slot?: number;
    stdinRing?: number;
    elfBytes: Uint8Array;
    argv: string[];
    envp: string[];
    memSnapshot?: SparseMemSnapshot;
    syscallState?: unknown;
    kernelPipesSnapshot?: unknown;
    threadStart?: WorkerInitConfig['threadStart'];
  }): void {
    const slot = opts.slot ?? this.findFreeSlot();
    const slotBase = procSlotOffset(slot) / 4;

    // Initialize process table entry
    Atomics.store(this.procTable, slotBase + PROC_PID, opts.pid);
    Atomics.store(this.procTable, slotBase + PROC_PPID, opts.ppid);
    Atomics.store(this.procTable, slotBase + PROC_PGID, opts.pgid);
    Atomics.store(this.procTable, slotBase + PROC_STATE, PROC_RUNNING);
    Atomics.store(this.procTable, slotBase + PROC_EXIT_CODE, 0);
    Atomics.store(this.procTable, slotBase + PROC_WAIT_FUTEX, 0);
    Atomics.store(this.procTable, slotBase + PROC_SIGNAL, 0);
    Atomics.store(this.procTable, slotBase + PROC_STDIN_RING, opts.stdinRing ?? -1);
    Atomics.store(this.procTable, slotBase + PROC_REPARENTED, 0);
    // tgid == pid for a normal process / thread-group leader; a thread carries its leader's pid.
    Atomics.store(this.procTable, slotBase + PROC_TGID, opts.threadStart?.tgid ?? opts.pid);

    // Take a warm Worker from the idle pool, or create one. reuse means the Worker's translator
    // instance + wasm Memory are kept (its init() wipes+resets); no per-spawn construction, and
    // crucially no terminate on exit -> no GC churn under a fork storm.
    const worker = this.idleWorkers.pop() ?? new Worker(this.workerURL);
    this.workers.set(opts.pid, worker);

    // (Re)bind the message/error handlers to THIS pid+slot (a reused Worker was bound to its
    // previous tenant).
    worker.onmessage = (e) => this.handleWorkerMessage(opts.pid, slot, e.data);
    worker.onerror = (e) => {
      console.error(`[kernel] Worker ${opts.pid} error:`, e.message, e.filename, e.lineno);
      // surface worker crashes on the terminal — otherwise they are invisible in the
      // native app (no console). the message often pinpoints a translator/syscall gap.
      const msg = `\r\n[kernel] worker ${opts.pid} crashed: ${e.message}\r\n`;
      try { this.onOutput(new TextEncoder().encode(msg)); } catch {}
      // a crashed worker must be reaped like an abnormal exit — otherwise its proc slot stays
      // RUNNING forever and its parent's wait4 blocks indefinitely (wedging the shell), and the
      // slot leaks. status 134 = 128 + SIGABRT. guard on (pid,RUNNING) so we don't double-handle
      // a worker that already posted 'exit', or a slot the kernel has since reused.
      const base = procSlotOffset(slot) / 4;
      if (Atomics.load(this.procTable, base + PROC_PID) === opts.pid &&
          Atomics.load(this.procTable, base + PROC_STATE) === PROC_RUNNING) {
        this.handleExit(opts.pid, slot, 134);
      }
    };

    // Build init config
    const config: WorkerInitConfig = {
      pid: opts.pid,
      ppid: opts.ppid,
      pgid: opts.pgid,
      slot,
      kernelSAB: this.sab,
      elfBytes: opts.elfBytes,
      argv: opts.argv,
      envp: opts.envp,
      memSnapshot: opts.memSnapshot,
      syscallState: opts.syscallState,
      kernelPipesSnapshot: opts.kernelPipesSnapshot as any,
      termCols: this.termCols,
      termRows: this.termRows,
      // prefer the shared compiled module; only ship raw bytes if compilation failed above.
      zigTranslatorModule: this.zigTranslatorModule,
      zigTranslatorBytes: this.zigTranslatorModule ? undefined : this.zigTranslatorBytes,
      threadStart: opts.threadStart,
    };

    // Send init — transfer the sparse snapshot's packed pages if present (zero-copy)
    const transferList: Transferable[] = [];
    if (config.memSnapshot) {
      transferList.push(config.memSnapshot.data);
    }
    worker.postMessage({ type: 'init', config }, transferList);
  }

  // release a thread-group's shared memory. instead of dropping it (relying on GC, which lags under
  // back-to-back threaded launches), park it as the single reusable spare so the next threaded process
  // takes it over. if a spare is already parked (a second threaded group ran concurrently), drop this
  // one for GC. `high` is the group's touched high-water so the reuser only zeros the stale span.
  private freeSharedMem(tgid: number, high = 0): void {
    const mem = this.sharedMems.get(tgid);
    this.sharedMems.delete(tgid);
    this.threadedLeaders.delete(tgid);
    if (mem && !this.spareSharedMem) this.spareSharedMem = { mem, high };
  }

  // first pthread_create ('clone_thread'): create this process's shared memory (only the main thread
  // can) and reserve the child's tid + reg-block slot, then hand the memory back to the caller via
  // 'promote'. the caller copies its state into it and replies 'thread_go', at which point the memory
  // holds the process state and we can safely spawn the thread-Worker (handleThreadGo).
  private handleCloneThreadPromote(parentPid: number, parentSlot: number): void {
    const worker = this.workers.get(parentPid);
    if (!worker) return;
    const slot = this.findFreeSlot();
    if (slot < 0 || (this.idleWorkers.length === 0 && this.workers.size >= POOL_MAX)) {
      worker.postMessage({ type: 'clone_fail' });
      return;
    }
    // reuse the single spare shared memory if one is parked (a previous threaded process freed it).
    // reusing avoids allocating+GC-freeing a ~0.5 GiB shared memory per threaded process, which the
    // engine can't reclaim fast enough under back-to-back launches -> the caller zeros the previous
    // tenant's stale region (staleHigh) instead. only when no spare is available do we allocate.
    let memory: WebAssembly.Memory;
    let staleHigh = 0;
    if (this.spareSharedMem) {
      memory = this.spareSharedMem.mem;
      staleHigh = this.spareSharedMem.high;
      this.spareSharedMem = null;
    } else {
      try {
        memory = new WebAssembly.Memory({ initial: THREADED_MEM_PAGES, maximum: THREADED_MEM_PAGES, shared: true });
      } catch {
        worker.postMessage({ type: 'clone_fail' });
        return;
      }
    }
    // reserve the child's slot NOW so a concurrent spawn can't claim it before thread_go arrives.
    const slotBase = procSlotOffset(slot) / 4;
    Atomics.store(this.procTable, slotBase + PROC_STATE, PROC_RUNNING);
    Atomics.store(this.procTable, slotBase + PROC_PID, -1);

    const tid = this.allocPid();
    const parentBase = procSlotOffset(parentSlot) / 4;
    const pgid = Atomics.load(this.procTable, parentBase + PROC_PGID);
    const tgid = Atomics.load(this.procTable, parentBase + PROC_PID); // parent is the group leader here
    const regBaseAddr = THREAD_REGS_BASE + slot * 0x400;

    this.sharedMems.set(tgid, memory);
    this.threadCounts.set(tgid, 1); // the leader (already running) counts as one live thread
    this.threadedLeaders.add(tgid); // its worker holds the shared memory -> must not be pooled on exit
    this.pendingPromotes.set(parentPid, { memory, tid, slot, regBaseAddr, tgid, pgid });
    worker.postMessage({ type: 'promote', sharedMemory: memory, tid, regBaseAddr, staleHigh });
  }

  // the promoted caller has copied its state into the shared memory and told us to go. spawn the
  // thread-Worker on that memory at its reserved slot/reg-block.
  private handleThreadGo(parentPid: number, msg: KernelMessage & { type: 'thread_go' }): void {
    const p = this.pendingPromotes.get(parentPid);
    if (!p) return;
    this.pendingPromotes.delete(parentPid);
    this.threadCounts.set(p.tgid, (this.threadCounts.get(p.tgid) ?? 1) + 1); // the child is now live
    this.spawnProcess({
      pid: p.tid, ppid: parentPid, pgid: p.pgid, slot: p.slot,
      elfBytes: this.elfBytes, argv: [], envp: [],
      threadStart: {
        sharedMemory: p.memory, startPC: msg.startPC, childStack: msg.childStack, tls: msg.tls,
        tid: p.tid, tgid: p.tgid, regBaseAddr: p.regBaseAddr, parentRegBase: msg.parentRegBase,
        childTidPtr: msg.childTidPtr, flags: msg.flags,
      },
    });
  }

  // subsequent pthread_create ('spawn_thread'): the group's shared memory already exists (keyed by
  // tgid), so reuse it. reply the tid (or -1 EAGAIN) to the caller via its mailbox.
  private handleSpawnThread(parentPid: number, parentSlot: number, msg: KernelMessage & { type: 'spawn_thread' }): void {
    const mb = new Int32Array(this.sab, mailboxSlotOffset(parentSlot), MAILBOX_SLOT_SIZE / 4);
    const memory = this.sharedMems.get(msg.tgid);
    const slot = this.findFreeSlot();
    if (!memory || slot < 0 || (this.idleWorkers.length === 0 && this.workers.size >= POOL_MAX)) {
      Atomics.store(mb, MBOX_VALUE0, -1); // EAGAIN
      Atomics.store(mb, MBOX_READY, 1);
      Atomics.notify(mb, MBOX_READY);
      return;
    }
    const tid = this.allocPid();
    const parentBase = procSlotOffset(parentSlot) / 4;
    const pgid = Atomics.load(this.procTable, parentBase + PROC_PGID);
    const regBaseAddr = THREAD_REGS_BASE + slot * 0x400;
    this.threadCounts.set(msg.tgid, (this.threadCounts.get(msg.tgid) ?? 1) + 1);

    this.spawnProcess({
      pid: tid, ppid: parentPid, pgid, slot,
      elfBytes: this.elfBytes, argv: [], envp: [],
      threadStart: {
        sharedMemory: memory, startPC: msg.startPC, childStack: msg.childStack, tls: msg.tls,
        tid, tgid: msg.tgid, regBaseAddr, parentRegBase: msg.parentRegBase, childTidPtr: msg.childTidPtr, flags: msg.flags,
      },
    });

    Atomics.store(mb, MBOX_VALUE0, tid);
    Atomics.store(mb, MBOX_READY, 1);
    Atomics.notify(mb, MBOX_READY);
  }

  // ── Worker message handling ───────────────────────────────────────────────

  private handleWorkerMessage(pid: number, slot: number, msg: KernelMessage): void {
    switch (msg.type) {
      case 'fork':
        this.handleFork(pid, slot, msg);
        break;

      case 'exit':
        this.handleExit(pid, slot, msg.code, true, msg.high ?? 0); // clean self-exit -> Worker reusable
        break;

      case 'kill':
        this.handleKillSignal(msg.pid, msg.sig);
        break;

      case 'write':
        this.onOutput((msg as any).data);
        break;

      case 'pipe_alloc':
        this.handlePipeAlloc(pid, slot);
        break;

      case 'clone_thread':
        this.handleCloneThreadPromote(pid, slot);
        break;

      case 'thread_go':
        this.handleThreadGo(pid, msg);
        break;

      case 'spawn_thread':
        this.handleSpawnThread(pid, slot, msg);
        break;

      case 'lock_req':
        this.handleLockReq(pid, slot, msg);
        break;

      case 'lock_cancel':
        this.cancelLockWaiter(pid);
        break;

      case 'bridge_call':
        this.handleBridgeCall(pid, slot, msg);
        break;

      case 'jit_compiled':
        // TODO: cache compiled modules for sharing with future forks
        break;

      default:
        console.warn(`[kernel] Unknown message from Worker ${pid}:`, msg);
    }
  }

  private replyForkEagain(parentSlot: number): void {
    const mb = new Int32Array(this.sab, mailboxSlotOffset(parentSlot), MAILBOX_SLOT_SIZE / 4);
    Atomics.store(mb, MBOX_VALUE0, -11); // -EAGAIN
    Atomics.store(mb, MBOX_READY, 1);
    Atomics.notify(mb, MBOX_READY);
  }

  private handleFork(parentPid: number, parentSlot: number, msg: KernelMessage & { type: 'fork' }): void {
    // bound concurrency, all surfaced as fork() EAGAIN (never a crash): the hard 64-slot proc
    // table; the live-Worker ceiling WORKER_CAP (pid 1 EXEMPT so the shell never stalls); and the
    // hard total-Worker ceiling POOL_MAX (a fork can only proceed by reusing an idle Worker or, if
    // none is idle, creating one while under POOL_MAX). pid 1's exemption from WORKER_CAP still
    // bottoms out at POOL_MAX, so total memory stays bounded.
    const childSlot = this.findFreeSlot();
    if (childSlot < 0 ||
        (parentPid !== 1 && this.workers.size >= WORKER_CAP) ||
        (this.idleWorkers.length === 0 && this.workers.size >= POOL_MAX)) {
      this.replyForkEagain(parentSlot);
      return;
    }
    const childPid = this.allocPid();
    const parentBase = procSlotOffset(parentSlot) / 4;
    const parentStdin = Atomics.load(this.procTable, parentBase + PROC_STDIN_RING);
    this.spawnProcess({
      pid: childPid,
      ppid: parentPid,
      pgid: Atomics.load(this.procTable, parentBase + PROC_PGID),
      slot: childSlot,
      stdinRing: parentStdin >= 0 ? parentStdin : undefined,
      elfBytes: this.elfBytes,
      argv: [],
      envp: [],
      memSnapshot: msg.memSnapshot,
      syscallState: msg.syscallState,
      kernelPipesSnapshot: (msg as any).kernelPipesSnapshot,
    });
    const mailbox = new Int32Array(this.sab, mailboxSlotOffset(parentSlot), MAILBOX_SLOT_SIZE / 4);
    Atomics.store(mailbox, MBOX_VALUE0, childPid);
    Atomics.store(mailbox, MBOX_READY, 1);
    Atomics.notify(mailbox, MBOX_READY);
  }

  // SIGKILL enforcement. a worker blocked in Atomics.wait (pipe read, bridge mailbox, ...)
  // cannot observe a signal bit and self-terminate, so the kernel force-terminates it here.
  // reuse the normal exit teardown (zombie + reap-wake) with a signalled exit status; the
  // RUNNING guard makes a racing self-exit idempotent.
  private handleKillSignal(pid: number, sig: number): void {
    for (let s = 0; s < 64; s++) {
      const base = procSlotOffset(s) / 4;
      if (Atomics.load(this.procTable, base + PROC_PID) !== pid) continue;
      if (Atomics.load(this.procTable, base + PROC_STATE) !== PROC_RUNNING) return;
      this.handleExit(pid, s, 128 + sig);
      return;
    }
  }

  // `poolable` is true only for a clean self-exit (the worker posted 'exit' and its runLoop
  // returned — it is now idle and reusable). a kill or crash leaves the Worker dead/mid-execution,
  // so those pass false and the Worker is destroyed instead of pooled.
  private handleExit(pid: number, slot: number, code: number, poolable = false, threadedHigh = 0): void {
    const slotBase = procSlotOffset(slot) / 4;

    // drop every advisory record lock this process held so a blocked F_SETLKW waiter (in another
    // process) can be granted, and so a stale lock never wedges a later same-path locker.
    this.releaseLocksForPid(pid);

    const tgid = Atomics.load(this.procTable, slotBase + PROC_TGID) || pid;

    // a non-leader thread exiting is NOT a process exit: no zombie, no SIGCHLD to the parent, no
    // reparenting. pthread_join blocks on the CLONE_CHILD_CLEARTID futex (zeroed + woken by the
    // thread-Worker's postExit), not wait4. decrement the group; if it was the last thread AND the
    // leader is already gone, free the shared memory. then free the proc slot + retire the worker.
    if (tgid !== pid) {
      if (this.threadCounts.has(tgid)) {
        const n = (this.threadCounts.get(tgid) ?? 1) - 1;
        if (n <= 0) { this.threadCounts.delete(tgid); this.freeSharedMem(tgid, threadedHigh); }
        else this.threadCounts.set(tgid, n);
      }
      const w = this.workers.get(pid);
      if (w) { this.workers.delete(pid); w.terminate(); }
      Atomics.store(this.procTable, slotBase + PROC_PID, 0);
      Atomics.store(this.procTable, slotBase + PROC_STATE, PROC_EMPTY);
      return;
    }

    // a thread-group LEADER dying == the whole process dies (a returning/exiting main thread runs
    // exit_group). so kill every remaining thread of the group and free the shared memory now, rather
    // than waiting on a join that will never come. the leader's own Worker holds the shared-memory
    // reference, so it must be terminated (not pooled) or the ~1.5 GiB reservation lingers.
    if (this.sharedMems.has(pid)) {
      for (let s = 0; s < 64; s++) {
        const b = procSlotOffset(s) / 4;
        const tp = Atomics.load(this.procTable, b + PROC_PID);
        if (tp === pid || tp === 0) continue;
        if (Atomics.load(this.procTable, b + PROC_TGID) !== pid) continue;
        const tw = this.workers.get(tp);
        if (tw) { this.workers.delete(tp); tw.terminate(); }
        Atomics.store(this.procTable, b + PROC_PID, 0);
        Atomics.store(this.procTable, b + PROC_STATE, PROC_EMPTY);
      }
      this.threadCounts.delete(pid);
      this.freeSharedMem(pid, threadedHigh);
      poolable = false;
    }

    // Mark as zombie in process table
    Atomics.store(this.procTable, slotBase + PROC_STATE, PROC_ZOMBIE);
    Atomics.store(this.procTable, slotBase + PROC_EXIT_CODE, code);

    // Wake any parent waiting on this child
    Atomics.store(this.procTable, slotBase + PROC_WAIT_FUTEX, 1);
    Atomics.notify(this.procTable, slotBase + PROC_WAIT_FUTEX);

    // Also wake the parent's wait_futex (the parent waits on its OWN slot's futex,
    // checking children's states). We need to notify on the parent's slot too.
    const ppid = Atomics.load(this.procTable, slotBase + PROC_PPID);
    for (let s = 0; s < 64; s++) {
      const sBase = procSlotOffset(s) / 4;
      if (Atomics.load(this.procTable, sBase + PROC_PID) === ppid &&
          Atomics.load(this.procTable, sBase + PROC_STATE) === PROC_RUNNING) {
        // Wake the parent in case it's blocked in wait4
        Atomics.store(this.procTable, sBase + PROC_WAIT_FUTEX, 1);
        Atomics.notify(this.procTable, sBase + PROC_WAIT_FUTEX);
        break;
      }
    }

    // Reparent this process's children to init (pid 1) and flag them orphaned. our init/shell
    // is not a reaper, so a reparented process's eventual zombie would leak its proc slot; the
    // flag makes the kernel auto-reap it. a child already zombied (was waiting on us to reap it)
    // is swept now, since nobody else will. without this, a killed subtree's descendants pile up
    // in the 64-slot table until fork() starves — exactly what wedged the ltp sweep.
    for (let s = 0; s < 64; s++) {
      const b = procSlotOffset(s) / 4;
      if (Atomics.load(this.procTable, b + PROC_PID) === 0) continue;
      if (Atomics.load(this.procTable, b + PROC_PPID) !== pid) continue;
      Atomics.store(this.procTable, b + PROC_PPID, 1);
      Atomics.store(this.procTable, b + PROC_REPARENTED, 1);
      if (Atomics.load(this.procTable, b + PROC_STATE) === PROC_ZOMBIE) this.reapSlot(s);
    }

    // Retire the Worker. a cleanly-exited one is idle and reusable -> park it in the idle pool
    // (its next init() wipes + resets it); the idle-trim timer bounds the pool during calm. a
    // killed/crashed one is destroyed. this is the crux: no terminate on the clean-exit hot path,
    // so a fork storm leaves no dead Workers for GC to chase.
    const worker = this.workers.get(pid);
    if (worker) {
      this.workers.delete(pid);
      if (poolable) this.idleWorkers.push(worker);
      else worker.terminate();
    }

    // If PID 1 (init) exits, all processes are dead
    if (pid === 1) {
      // Terminate all remaining workers, live and idle
      for (const [p, w] of this.workers) w.terminate();
      this.workers.clear();
      for (const w of this.idleWorkers) w.terminate();
      this.idleWorkers.length = 0;
      return;
    }

    // If we were ourselves an orphan (reparented to init, which won't wait4 us), don't leave a
    // zombie slot to leak — the kernel reaps on init's behalf.
    if (Atomics.load(this.procTable, slotBase + PROC_REPARENTED) === 1) this.reapSlot(slot);
  }

  // Free a proc-table slot outright (kernel-side reap of an orphan zombie). terminates any
  // lingering worker for that pid, then clears the slot so fork() can reuse it.
  private reapSlot(slot: number): void {
    const base = procSlotOffset(slot) / 4;
    const pid = Atomics.load(this.procTable, base + PROC_PID);
    const w = this.workers.get(pid);
    if (w) { w.terminate(); this.workers.delete(pid); }
    Atomics.store(this.procTable, base + PROC_STATE, PROC_EMPTY);
    Atomics.store(this.procTable, base + PROC_PID, 0);
    Atomics.store(this.procTable, base + PROC_PPID, 0);
    Atomics.store(this.procTable, base + PROC_REPARENTED, 0);
  }

  private handlePipeAlloc(pid: number, slot: number): void {
    // find a free ring (both refcounts 0 — either never used or fully closed) and claim it by
    // resetting to one reader + one writer. runs only here on the single-threaded main loop, so
    // the scan-then-claim is atomic wrt other pipe_alloc requests (no cross-alloc race). the old
    // monotonic GLOBAL_NEXT_PIPE counter never wrapped or reused, so it walked off the end of the
    // 32-ring region after 32 pipes — a whole LTP run allocates far more than that.
    let pipeSlot = -1;
    for (let i = 0; i < PIPE_RING_COUNT; i++) {
      const ring = new PipeRing(this.sab, pipeRingOffset(i), PIPE_RING_SIZE);
      if (ring.readerClosed && ring.writerClosed) {
        ring.reset(1, 1);
        pipeSlot = i;
        break;
      }
    }
    // pipeSlot === -1 means the ring pool is exhausted; the worker maps that to ENFILE.
    const mailbox = new Int32Array(this.sab, mailboxSlotOffset(slot), MAILBOX_SLOT_SIZE / 4);
    Atomics.store(mailbox, MBOX_VALUE0, pipeSlot);
    Atomics.store(mailbox, MBOX_READY, 1);
    Atomics.notify(mailbox, MBOX_READY);
  }

  // ── fcntl advisory record locking ─────────────────────────────────────────
  // reply words land in the mailbox starting at MBOX_VALUE0. for F_GETLK a conflict is reported
  // as [0, type, pid, start_lo, start_hi, len_lo, len_hi]; for set/unlock it is [0] or [-errno].
  private lockReply(slot: number, words: number[]): void {
    const mb = new Int32Array(this.sab, mailboxSlotOffset(slot), MAILBOX_SLOT_SIZE / 4);
    for (let i = 0; i < words.length; i++) Atomics.store(mb, MBOX_VALUE0 + i, words[i] | 0);
    Atomics.store(mb, MBOX_READY, 1);
    Atomics.notify(mb, MBOX_READY);
  }

  private lockConflict(list: Array<{ pid: number; write: boolean; start: number; end: number }>,
                       pid: number, wantWrite: boolean, start: number, end: number) {
    // two locks conflict when they overlap, belong to different owners, and at least one is a
    // write lock (read locks share). a process never conflicts with its own locks.
    return list.find(l => l.pid !== pid && l.start < end && start < l.end && (wantWrite || l.write));
  }

  private applyUnlock(path: string, pid: number, start: number, end: number): void {
    const list = this.recordLocks.get(path);
    if (!list) return;
    const out: typeof list = [];
    for (const l of list) {
      if (l.pid !== pid || l.end <= start || end <= l.start) { out.push(l); continue; }
      if (l.start < start) out.push({ pid, write: l.write, start: l.start, end: start });
      if (end < l.end) out.push({ pid, write: l.write, start: end, end: l.end });
    }
    if (out.length) this.recordLocks.set(path, out); else this.recordLocks.delete(path);
  }

  private applyLock(path: string, pid: number, write: boolean, start: number, end: number): void {
    this.applyUnlock(path, pid, start, end); // a new lock supersedes this owner's overlap (may retype)
    const list = this.recordLocks.get(path) ?? [];
    list.push({ pid, write, start, end });
    // coalesce this owner's same-type touching ranges so F_GETLK reports merged extents (posix).
    const mine = list.filter(l => l.pid === pid && l.write === write).sort((a, b) => a.start - b.start);
    const rest = list.filter(l => !(l.pid === pid && l.write === write));
    const merged: typeof list = [];
    for (const l of mine) {
      const last = merged[merged.length - 1];
      if (last && l.start <= last.end) last.end = Math.max(last.end, l.end);
      else merged.push({ ...l });
    }
    this.recordLocks.set(path, [...rest, ...merged]);
  }

  private handleLockReq(pid: number, slot: number, msg: KernelMessage & { type: 'lock_req' }): void {
    const { path, cmd, ltype, start, len } = msg;
    const end = len <= 0 ? Infinity : start + len; // l_len 0 means "to end of file"
    const list = this.recordLocks.get(path) ?? [];

    if (cmd === 5 || cmd === 12) { // F_GETLK — report a conflicting lock, if any
      // report the lowest-start conflicting lock so a probe walking the file sees regions in order.
      const conflicts = list.filter(l => l.pid !== pid && l.start < end && start < l.end && (ltype === 1 || l.write));
      const c = conflicts.length ? conflicts.reduce((a, b) => (b.start < a.start ? b : a)) : undefined;
      if (!c) { this.lockReply(slot, [0, 2]); return; } // F_UNLCK: region is free
      const clen = c.end === Infinity ? 0 : c.end - c.start;
      this.lockReply(slot, [0, c.write ? 1 : 0, c.pid,
        c.start >>> 0, Math.floor(c.start / 4294967296),
        clen >>> 0, Math.floor(clen / 4294967296)]);
      return;
    }

    if (ltype === 2) { // F_UNLCK
      this.applyUnlock(path, pid, start, end);
      this.lockReply(slot, [0]);
      this.retryLockWaiters(path);
      return;
    }

    const c = this.lockConflict(list, pid, ltype === 1, start, end);
    if (c) {
      if (cmd === 7 || cmd === 14) { // F_SETLKW — block until the region frees
        // refuse a wait that would close a cycle in the wait-for graph (posix EDEADLK): otherwise
        // both processes park forever and only their watchdog alarm breaks it.
        if (this.wouldDeadlock(pid, path, ltype === 1, start, end)) { this.lockReply(slot, [-35]); return; }
        this.lockWaiters.push({ pid, slot, path, write: ltype === 1, start, end });
        return; // reply is deferred to retryLockWaiters when the conflict clears
      }
      this.lockReply(slot, [-11]); // EAGAIN
      return;
    }
    this.applyLock(path, pid, ltype === 1, start, end);
    this.lockReply(slot, [0]);
  }

  // pids currently holding a lock that conflicts with the given request (its wait-for edges).
  private conflictingOwners(path: string, pid: number, write: boolean, start: number, end: number): number[] {
    const list = this.recordLocks.get(path) ?? [];
    const owners = new Set<number>();
    for (const l of list) if (l.pid !== pid && l.start < end && start < l.end && (write || l.write)) owners.add(l.pid);
    return [...owners];
  }

  // would parking pid on this request close a cycle? walk the wait-for graph (pid waits for the
  // lock owners, each blocked owner waits for its own conflict owners) and look for a path back.
  private wouldDeadlock(pid: number, path: string, write: boolean, start: number, end: number): boolean {
    const seen = new Set<number>();
    const stack = this.conflictingOwners(path, pid, write, start, end);
    while (stack.length) {
      const o = stack.pop()!;
      if (o === pid) return true;
      if (seen.has(o)) continue;
      seen.add(o);
      const w = this.lockWaiters.find(x => x.pid === o);
      if (w) stack.push(...this.conflictingOwners(w.path, w.pid, w.write, w.start, w.end));
    }
    return false;
  }

  private retryLockWaiters(path: string): void {
    for (let i = 0; i < this.lockWaiters.length;) {
      const w = this.lockWaiters[i];
      if (w.path !== path) { i++; continue; }
      const list = this.recordLocks.get(w.path) ?? [];
      if (!this.lockConflict(list, w.pid, w.write, w.start, w.end)) {
        this.applyLock(w.path, w.pid, w.write, w.start, w.end);
        this.lockReply(w.slot, [0]);
        this.lockWaiters.splice(i, 1);
      } else i++;
    }
  }

  private releaseLocksForPid(pid: number): void {
    for (const [path, list] of this.recordLocks) {
      const kept = list.filter(l => l.pid !== pid);
      if (kept.length) this.recordLocks.set(path, kept); else this.recordLocks.delete(path);
    }
    this.lockWaiters = this.lockWaiters.filter(w => w.pid !== pid);
    for (const p of new Set([...this.recordLocks.keys()])) this.retryLockWaiters(p);
  }

  private cancelLockWaiter(pid: number): void {
    // an F_SETLKW that the worker abandoned (signal / give-up) must not later be granted.
    this.lockWaiters = this.lockWaiters.filter(w => w.pid !== pid);
  }

  private handleBridgeCall(pid: number, slot: number, msg: KernelMessage & { type: 'bridge_call' }): void {
    if (!this.bridge) {
      this.writeBridgeResponse(slot, -38, [], null);
      return;
    }

    const method = (msg as any).method as string;
    this.bridge.call(method, (msg as any).args as Record<string, any>).then((resp) => {
      const ret = (resp as any).ret ?? 0;

      // Pack response into SAB: [ret:i32][dataLen:u32][meta:32B][data:bytes]
      // Meta fields are method-specific fixed-layout integers.
      // Data is variable-length (binary for reads, JSON for getdents, UTF-8 for readlink).
      const meta = [0, 0, 0, 0, 0, 0]; // 6 x f64: mode, size, nlink, mtime, uid, gid
      let rawData: Uint8Array | null = null;

      if (method === 'fs_stat' || method === 'fs_fstat') {
        meta[0] = (resp as any).mode ?? 0;
        meta[1] = (resp as any).size ?? 0;
        meta[2] = (resp as any).nlink ?? 0;
        meta[3] = (resp as any).mtime ?? 0;
        meta[4] = (resp as any).uid ?? 0;
        meta[5] = (resp as any).gid ?? 0;
        meta[6] = (resp as any).ino ?? 0; // st_ino — MUST be unique per file: ld.so dedups libs by it
      } else if (method === 'fs_readlink') {
        if ((resp as any).target) {
          rawData = new TextEncoder().encode((resp as any).target);
        }
      } else if (method === 'fs_getdents') {
        if ((resp as any).entries) {
          rawData = new TextEncoder().encode(JSON.stringify((resp as any).entries));
        }
      } else if ((resp as any).data) {
        // Binary data: fs_read, net_read — base64 decode
        rawData = Uint8Array.from(atob((resp as any).data), c => c.charCodeAt(0));
      }

      this.writeBridgeResponse(slot, ret, meta, rawData);
    }).catch((err) => {
      // a rejected bridge call (unknown/unimplemented method, native exception, transient
      // failure) must STILL wake the worker with an error — otherwise it parks on the response
      // mailbox forever. before this, a single unhandled method (e.g. fs_chown) hung the whole
      // process. surface it as -EIO and log which method failed so gaps are visible, not silent.
      console.error(`[kernel] bridge call '${method}' failed:`, err);
      this.writeBridgeResponse(slot, -5, [0, 0, 0, 0], null); // -EIO
    });
  }

  private writeBridgeResponse(slot: number, ret: number, meta: number[], data: Uint8Array | null): void {
    const respOffset = BRIDGE_RESP_OFFSET + slot * BRIDGE_RESP_SLOT_SIZE;
    const respView = new DataView(this.sab, respOffset, BRIDGE_RESP_SLOT_SIZE);

    // Response layout: [ret:i32][metaCount:u16][dataLen:u32][meta: metaCount×f64][data: dataLen bytes]
    // Header: 10 bytes, then variable meta, then variable data.
    const metaCount = meta.length;
    const dataOffset = 10 + metaCount * 8;

    respView.setInt32(0, ret, true);
    respView.setUint16(4, metaCount, true);

    for (let i = 0; i < metaCount; i++) {
      respView.setFloat64(10 + i * 8, meta[i], true);
    }

    if (data && data.length > 0 && data.length <= BRIDGE_RESP_SLOT_SIZE - dataOffset) {
      respView.setUint32(6, data.length, true);
      new Uint8Array(this.sab, respOffset + dataOffset, data.length).set(data);
    } else {
      respView.setUint32(6, 0, true);
    }

    // Wake the Worker
    const mailbox = new Int32Array(this.sab, mailboxSlotOffset(slot), MAILBOX_SLOT_SIZE / 4);
    Atomics.store(mailbox, MBOX_READY, 1);
    Atomics.notify(mailbox, MBOX_READY);
  }
}
