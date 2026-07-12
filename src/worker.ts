/**
 * Web Worker entry point for multicore process execution.
 */

import { HALT_PC, SVC_PC, GUEST_MEM_BASE, ITER_OFFSET, PC_OFFSET, STATE_SIZE } from './types.js';
import type { CompiledRegion } from './types.js';
import { CpuState } from './cpu.js';
import { initZigTranslator, initZigTranslatorModule, translateRegionZig } from './zig-translator.js';
import { SyscallHandler, badGuestPtr } from './syscall.js';
import { InMemoryVFS } from './vfs.js';
import { loadProgram } from './elf.js';
import { PipeRing } from './pipe-ring.js';
import type { WorkerInitConfig, KernelMessage, SparseMemSnapshot } from './kernel-ipc.js';
import {
  MEM_PAGE_SIZE,
  PROC_SLOT_SIZE, PROC_STATE, PROC_EXIT_CODE, PROC_WAIT_FUTEX, PROC_SIGNAL,
  PROC_EMPTY, PROC_ZOMBIE, PROC_RUNNING,
  PROC_PID, PROC_PPID, PROC_PGID,
  MAILBOX_OFFSET, MAILBOX_SLOT_SIZE, MBOX_READY, MBOX_VALUE0,
  PIPE_RING_OFFSET, PIPE_RING_SIZE,
  STDIN_RING_OFFSET, STDIN_RING_SIZE,
  SIG_INT,
  BRIDGE_RESP_OFFSET, BRIDGE_RESP_SLOT_SIZE,
  procSlotOffset, mailboxSlotOffset, pipeRingOffset, stdinRingOffset,
} from './kernel-ipc.js';

let memory: WebAssembly.Memory;
let cpu: CpuState;
let syscalls: SyscallHandler;
let vfs: InMemoryVFS;
let regionCache = new Map<bigint, CompiledRegion>();

// ── threading ────────────────────────────────────────────────────────────────
// a process stays single-threaded until its first pthread_create (clone with CLONE_VM|CLONE_THREAD),
// then PROMOTES: its memory becomes a shared WebAssembly.Memory (so thread-Workers can import it) and
// translation switches to threaded mode (reg_base-relative registers + real atomics + fences). each
// thread-Worker runs the same shared memory with its own register block at `stateBase`/`regBaseGlobal`.
let threaded = false;
let stateBase = 0;                                    // this thread's register-block base (0 = group main)
let regBaseGlobal: WebAssembly.Global | null = null;  // per-thread reg_base handed to the translator
let clearChildTid = 0;                                // CLONE_CHILD_CLEARTID: guest addr to zero+wake on exit
// threaded memory layout (all lazily committed in the shared memory). a threaded process gets the
// SAME 1.5 GiB guest ceiling as single-threaded: the single-spare reuse in the kernel means sequential
// threaded processes recycle one shared memory rather than alloc/free-churning, so the large per-proc
// reservation no longer needs to be reclaimed between launches. the guest occupies [0x10000,
// THREAD_REGS_BASE); the per-thread register blocks (64 x 0x400 at THREAD_REGS_BASE) and the shared
// LDXR/STXR shadow (guest_size/512 = 3 MiB, 4 MiB above the reg blocks) sit ABOVE that ceiling so a
// growing brk/mmap can never reach them. the syscall handler's maxBytes enforces the ceiling.
// (the only cost of 1.5 GiB is a lower CONCURRENT-threaded-process count, ~2-3, bounded by the
// platform's shared-memory pool; sequential runs are unbounded via reuse.)
const THREAD_REGS_BASE = 0x60000000;                  // 1.5 GiB — the threaded guest ceiling (== single-threaded)
const THREADED_SHADOW_BASE = 0x60400000;              // 1.5 GiB + 4 MiB (clear of the 64 reg blocks)
// cap on cached translated blocks per worker. a hot working set is usually a few hundred; this
// bounds the pathological huge-binary case at ~this many compiled wasm blocks (re-JIT on re-hit).
const REGION_CACHE_CAP = 2048;
let elfBytes: Uint8Array;

// Kernel SAB views
let kernelSAB: SharedArrayBuffer;
let procTable: Int32Array;
let mySlot: number;
let myPid: number;
let myMailbox: Int32Array;

// Kernel pipe tracking: local fd → { pipeIndex, mode: 'read'|'write' }
// These are pipes backed by SharedArrayBuffer ring buffers (cross-Worker IPC)
const kernelPipes = new Map<number, { pipeIndex: number; mode: 'read' | 'write' }>();
let nextLocalFd = 100; // kernel pipe fds start high to avoid conflicts with VFS fds

// Stdin ring for this process (if connected to terminal)
let stdinRing: PipeRing | null = null;

// this Worker is pooled and reused across guest processes (see the kernel's worker pool). the
// translator instance and the wasm Memory are created ONCE and reused; everything else is reset
// per process. reusing the Worker (never terminate) is what removes the GC-reclaim lag that a
// fork storm drowned in.
let workerReady = false;
let lastSyscallNr = -1;

const ZIG_SHADOW_BASE = 0x2000000; // 32 MB — guest stays under this

// ── Message handler ─────────────────────────────────────────────────────────

self.onmessage = (e: MessageEvent) => {
  const msg = e.data;

  switch (msg.type) {
    case 'init':
      init(msg.config);
      break;
    case 'fork_result':
      // Wake from Atomics.wait — handled via mailbox, this is a fallback notification
      break;
    case 'stdin_data':
      // Data was written to our stdin ring — we'll pick it up in the read syscall
      break;
    case 'signal': {
      // Signal delivered — we check signal_pending in the run loop
      break;
    }
    case 'term_size':
      if (syscalls) syscalls.setTermSize(msg.cols, msg.rows);
      break;
    case 'promote':
      // the kernel created (or reused) the shared memory for our first thread; finish the yielded clone.
      completePromote(msg.sharedMemory, msg.tid, msg.regBaseAddr, msg.staleHigh);
      break;
    case 'clone_fail':
      completePromoteFail();
      break;
  }
};

// ── Initialization ──────────────────────────────────────────────────────────

function init(config: WorkerInitConfig) {
  myPid = config.pid;
  mySlot = config.slot;
  kernelSAB = config.kernelSAB;
  elfBytes = config.elfBytes;

  if (!workerReady) {
    // ── one-time worker setup: the zig translator (prefer the kernel's shared pre-compiled module,
    // no per-worker 1.5MB compile) and the reusable wasm Memory. NON-shared, single-threaded
    // (shared_monitor=0, no wasm atomics), grows lazily so a big `maximum` is free on RSS. ──
    if (config.zigTranslatorModule) {
      initZigTranslatorModule(config.zigTranslatorModule);
    } else if (config.zigTranslatorBytes) {
      initZigTranslator(config.zigTranslatorBytes);
    } else {
      throw new Error(`[worker ${myPid}] no translator module or bytes in init config`);
    }
    memory = new WebAssembly.Memory({ initial: 512, maximum: 65536 }); // 32MB start, 4GiB ceiling
    workerReady = true;
  } else {
    // ── pool reuse: this Worker just finished another process. wipe the span the outgoing tenant
    // could have dirtied so the incoming process sees a clean slate and no cross-process data
    // leaks. [0, mmapBase] covers it: the ELF, BSS, heap, stack (top 8MB), TLS (9MB) and every
    // mmap (which bump-allocates upward from 33MB) all live below mmapBase. clamp to the committed
    // size. then drop the per-process caches (the new tenant's code differs, possibly at identical
    // PCs). (a finer dirty-tracked wipe is a noted follow-up optimization.)
    // a worker that PROMOTED to shared memory for threads must not hand that shared memory (or the
    // threaded translation mode) to the next pooled tenant — reset to a fresh non-shared process.
    if (threaded) {
      memory = new WebAssembly.Memory({ initial: 512, maximum: 65536 });
      threaded = false; stateBase = 0; regBaseGlobal = null;
    }
    const dirtyEnd = Math.min(Number(syscalls.getMmapBase()) + GUEST_MEM_BASE, memory.buffer.byteLength);
    new Uint8Array(memory.buffer).fill(0, 0, dirtyEnd);
    regionCache.clear();
    kernelPipes.clear();
    nextLocalFd = 100;
    stdinRing = null;
  }

  // Set up SAB views (slot/mailbox change on every reuse)
  procTable = new Int32Array(kernelSAB);
  myMailbox = new Int32Array(kernelSAB, mailboxSlotOffset(mySlot), MAILBOX_SLOT_SIZE / 4);

  // on ipad the bridge owns real files (/bin /etc /tmp ...) — vfs only synthesises /proc and /dev.
  // off-ipad (tests, browsers without bridge) vfs is the whole filesystem
  const bridgeHandlesRealFs = !!config.kernelSAB;
  vfs = new InMemoryVFS({ bridgeHandlesRealFs });
  if (!bridgeHandlesRealFs) {
    vfs.loadBinary(elfBytes);
  }

  // ── Thread-start path: this Worker is a THREAD of an existing process (CLONE_THREAD). it imports
  // the process's SHARED memory (not its own) and runs at its own register-block base. ──
  if (config.threadStart) {
    const ts = config.threadStart;
    memory = ts.sharedMemory;
    threaded = true;
    stateBase = ts.regBaseAddr;
    regBaseGlobal = new WebAssembly.Global({ value: 'i32', mutable: false }, ts.regBaseAddr);
    // start from a copy of the parent thread's registers, then apply clone semantics.
    new Uint8Array(memory.buffer).copyWithin(stateBase, ts.parentRegBase, ts.parentRegBase + STATE_SIZE);
    cpu = new CpuState(memory, stateBase);
    cpu.setReg(0, 0n);                                   // child returns 0 from clone
    cpu.setSP(BigInt(ts.childStack >>> 0));              // thread's own stack
    new DataView(memory.buffer).setBigUint64(stateBase + 0x118, BigInt(ts.tls >>> 0), true); // TPIDR
    if ((ts.flags & 0x01000000) && ts.childTidPtr) {    // CLONE_CHILD_SETTID
      new DataView(memory.buffer).setInt32(ts.childTidPtr + GUEST_MEM_BASE, ts.tid, true);
    }
    if ((ts.flags & 0x00200000) && ts.childTidPtr) clearChildTid = ts.childTidPtr; // CLONE_CHILD_CLEARTID
    // a fresh syscall handler for the thread. NOTE: threads SHARE the fd table / brk with the group;
    // this first cut gives each thread its own handler (fine for compute+futex threads). shared fds
    // are a follow-up (thread the parent's syscallState through spawn_thread).
    syscalls = new SyscallHandler(0n, {
      writeStdout: (d) => postWrite(1, d),
      writeStderr: (d) => postWrite(2, d),
      readStdin: (count) => readStdin(count),
      stdinHasData: () => stdinRing != null && (!stdinRing.empty || stdinRing.writerClosed),
      waitStdin: (timeoutMs) => waitStdin(timeoutMs),
    }, vfs);
    syscalls.bridgeAvailable = true;
    syscalls.setThreadIds(ts.tid, ts.tgid);             // gettid=tid, getpid=tgid
    syscalls.maxBytes = THREAD_REGS_BASE;               // same guest ceiling as the rest of the group
    runToCompletion(BigInt(ts.startPC));
    return;
  }

  let pc: bigint;

  if (config.memSnapshot) {
    // ── Fork path: restore parent's sparse memory snapshot (only touched pages travel) ──
    applySparseSnapshot(memory, config.memSnapshot);
    cpu = new CpuState(memory, stateBase);

    // Child returns 0 from fork
    cpu.setReg(0, 0n);

    // Restore syscall state
    syscalls = new SyscallHandler(0n, {
      writeStdout: (d) => postWrite(1, d),
      writeStderr: (d) => postWrite(2, d),
      readStdin: (count) => readStdin(count),
      stdinHasData: () => stdinRing != null && (!stdinRing.empty || stdinRing.writerClosed),
      waitStdin: (timeoutMs) => waitStdin(timeoutMs),
    }, vfs);
    syscalls.bridgeAvailable = true; // Worker can relay bridge calls to kernel
    if (config.syscallState) {
      syscalls.restoreState(config.syscallState as any);
      // restoreState brought over the parent's pid; adopt our own identity, with
      // the parent's pid (just restored) as our ppid
      syscalls.setIdentity(myPid, syscalls.currentPid);
      // each inherited native fd must be dup'd so the child owns its own swift fd;
      // otherwise parent close() yanks the underlying filehandle out from under us
      syscalls.dupNativeFdsAfterFork((oldFd) => {
        let result = -1;
        handleBridgeCall('fs_dup', { fd: oldFd }, (resp) => {
          result = resp.ret;
        });
        return result;
      });
    }

    // Restore kernel pipes map from parent. the refcounts for these inherited ends were
    // already bumped synchronously by the parent in handleFork (see the note there), so we
    // must NOT bump them again here or every pipe end would be over-counted by one.
    if (config.kernelPipesSnapshot) {
      for (const [fd, info] of config.kernelPipesSnapshot) {
        kernelPipes.set(fd, { ...info });
      }
      for (const fd of kernelPipes.keys()) {
        if (fd >= nextLocalFd) nextLocalFd = fd + 1;
      }
    }

    // Resume at the instruction after the fork syscall
    pc = cpu.getPC() + 4n;
  } else {
    // ── Exec path: fresh binary load (static or dynamic via loadProgram) ──
    const prog = loadProgram(memory, elfBytes, config.argv, config.envp, readGuestFile);
    // loadProgram may have grown (detached) memory; take fresh views
    cpu = new CpuState(memory, stateBase);
    cpu.setSP(prog.sp);
    new DataView(memory.buffer).setBigUint64(stateBase + 0x118, BigInt(0x900000), true); // TPIDR_OFFSET

    syscalls = new SyscallHandler(prog.brkStart, {
      writeStdout: (d) => postWrite(1, d),
      writeStderr: (d) => postWrite(2, d),
      readStdin: (count) => readStdin(count),
      stdinHasData: () => stdinRing != null && (!stdinRing.empty || stdinRing.writerClosed),
      waitStdin: (timeoutMs) => waitStdin(timeoutMs),
    }, vfs);
    syscalls.bridgeAvailable = true;

    pc = prog.entry;
  }

  // Set up stdin ring if allocated
  const stdinRingIdx = Atomics.load(procTable, (procSlotOffset(mySlot) / 4) + 7); // PROC_STDIN_RING
  if (stdinRingIdx >= 0) {
    stdinRing = new PipeRing(kernelSAB, stdinRingOffset(stdinRingIdx), STDIN_RING_SIZE);
  }

  if (config.termCols && config.termRows) {
    syscalls.setTermSize(config.termCols, config.termRows);
  }

  // install the rt_sigreturn trampoline at guest 0x1000. this was only ever done in the node
  // harness (dispatch.ts), never here — so in-vivo every signal handler returned into unmapped
  // low memory and hung. covers both the fork-init and exec-init paths.
  syscalls.installSignalTrampoline(memory);

  runToCompletion(pc);
}

// run the guest to termination with the shared crash + fd-cleanup handling. used by every entry into
// the run loop (exec/fork init, thread start, and the async-promote resume) so a crash or exit is
// surfaced + cleaned up identically regardless of which path started it.
function runToCompletion(pc: bigint): void {
  try {
    runLoop(pc);
  } catch (e: any) {
    if (e.message === 'SIGINT') {
      // Already called postExit(130) — just stop
      return;
    }
    console.error(`[worker ${myPid}] CRASH:`, e.message, e.stack);
    // surface the crash on the terminal — otherwise a translator gap or runtime
    // fault during startup is a silent exit(139): "starting sh" then nothing.
    try {
      postWrite(2, new TextEncoder().encode(`\r\n[worker ${myPid}] crash: ${e.message} (last syscall nr=${lastSyscallNr})\r\n`));
    } catch {}
    postExit(139); // SIGSEGV-like exit
  } finally {
    // the process has ended; close its native (Swift) fds so pooling this Worker for the next
    // tenant doesn't leak file handles (which eventually exhausts the device fd budget and makes
    // NativeFS start failing opens). runs on every exit path (normal, SIGINT, crash).
    // EXCEPT a run loop that returned to YIELD for the async thread-promote (pendingClone set): the
    // process is very much alive, waiting on onmessage('promote') to resume — must not touch its fds.
    if (!pendingClone) {
      try {
        syscalls.closeNativeFds((fd) => handleBridgeCall('fs_close', { fd }, () => {}));
      } catch {}
    }
  }
}

// ── JIT region cache ────────────────────────────────────────────────────────

function getRegion(pc: bigint): CompiledRegion {
  const region = regionCache.get(pc);
  if (region) {
    // bounded-LRU: only reorder (move to newest) once we're at the cap, so a small working set
    // — the common case — pays zero Map churn on the hot dispatch path.
    if (regionCache.size >= REGION_CACHE_CAP) { regionCache.delete(pc); regionCache.set(pc, region); }
    return region;
  }

  const view = new DataView(memory.buffer);
  const shadowBase = threaded ? THREADED_SHADOW_BASE : ZIG_SHADOW_BASE;
  const zigRegion = translateRegionZig(view, pc, memory, shadowBase, false, threaded, regBaseGlobal);
  if (!zigRegion) {
    const insn = view.getUint32(Number(pc) + 0x10000, true);
    throw new Error(`[worker ${myPid}] translator failed at pc=0x${pc.toString(16)} (entry insn 0x${insn.toString(16).padStart(8, '0')})`);
  }
  // evict the least-recently-used block (oldest in Map insertion order) before caching a new one,
  // so a huge/long-running program's translation cache can't grow without bound (a per-worker
  // leak). re-JIT on a later re-hit is the only cost; correctness is unaffected (and it happens
  // to make self-modifying code re-translate correctly).
  if (regionCache.size >= REGION_CACHE_CAP) {
    const oldest = regionCache.keys().next().value;
    if (oldest !== undefined) regionCache.delete(oldest);
  }
  regionCache.set(pc, zigRegion);
  return zigRegion;
}

// ── Main execution loop ─────────────────────────────────────────────────────

// fold the armed interval timer and any cross-worker procTable signal bits into the
// process-local pendingSignals queue. bit position in PROC_SIGNAL == signal number
// (SIG_INT = 1<<2). returns true if a signal is now pending.
function pollAndQueueSignals(): boolean {
  syscalls.checkTimers(); // may push SIGALRM
  const sigSlot = (procSlotOffset(mySlot) / 4) + PROC_SIGNAL;
  const bits = Atomics.load(procTable, sigSlot);
  if (bits) {
    Atomics.and(procTable, sigSlot, ~bits); // consume the bits we observed
    for (let signo = 1; signo < 32; signo++) if (bits & (1 << signo)) syscalls.raiseSignal(signo);
  }
  return syscalls.pendingSignals.length > 0;
}

// a futex waiter for indefinite blocking syscalls (pipe read, ...) that yields on signals.
// parks on the given futex for a bounded slice, then checks for a pending signal (an armed
// timer via checkTimers, or an incoming cross-worker kill/term set in PROC_SIGNAL). returns
// true if a signal is now queued and the syscall should abandon its wait with EINTR. the 1s
// (or timer-deadline) cap bounds how long an incoming signal can go unobserved, since it
// notifies the proc slot's wait futex, not the ring futex this parks on.
function interruptibleWaiter(i32: Int32Array, idx: number, expected: number): boolean {
  const rem = syscalls.timerRemainingMs();
  Atomics.wait(i32, idx, expected, rem > 0 ? Math.min(1000, rem) : 1000);
  return pollAndQueueSignals();
}

// deliver queued signals at a safe point (pc = next instruction). returns the pc to
// continue from, or null if the process terminated (caller must return from runLoop).
function deliverQueuedSignals(pc: bigint): bigint | null {
  if (!syscalls.pendingSignals.length) return pc;
  cpu.setPC(pc);
  while (syscalls.pendingSignals.length > 0) {
    const sig = syscalls.pendingSignals.shift()!;
    const r = syscalls.deliverSignal(cpu, memory, sig);
    if (r.action === 'exit') { postExit(r.exitCode ?? 128 + sig); return null; }
    if (r.delivered && r.action === 'continue') return cpu.getPC(); // handler installed
  }
  return pc;
}

function runLoop(startPC: bigint) {
  let pc = startPC;
  let totalInsns = 0;
  let view = new DataView(memory.buffer);
  let lastSignalCheck = 0;
  let blockCount = 0;

  while (pc !== HALT_PC) {
    blockCount++;
    const region = getRegion(pc);
    pc = region.run();

    // Refresh view in case memory grew
    view = new DataView(memory.buffer);
    const insns = Number(view.getBigUint64(stateBase + ITER_OFFSET, true));
    totalInsns += insns;
    view.setBigUint64(stateBase + ITER_OFFSET, 0n, true);

    // Check timers + cross-worker signals periodically, so an armed alarm or an incoming
    // kill fires even in a syscall-less spin (this is what bounds a runaway ltp test).
    lastSignalCheck += insns;
    if (lastSignalCheck > 10000) {
      lastSignalCheck = 0;
      if (pollAndQueueSignals()) {
        const next = deliverQueuedSignals(pc);
        if (next === null) return; // process terminated
        pc = next;
        continue;
      }
    }

    // Handle syscall
    if (pc === SVC_PC) {
      const svcPC = view.getBigUint64(stateBase + PC_OFFSET, true);
      const nr = Number(cpu.getReg(8));
      lastSyscallNr = nr;

      // Intercept syscalls that need direct kernel SAB access
      if (nr === 260) {
        handleWait4(cpu, view);
        // SA_RESTART: if a signal with SA_RESTART interrupted the wait (EINTR), restart the
        // syscall (return to the SVC after the handler) instead of surfacing EINTR to the guest.
        // tst_test's SIGUSR1 heartbeat relies on this — without it the parent EINTRs out of its
        // wait, never reaps the test child, and hits the 30s watchdog.
        const eintr = BigInt.asIntN(64, cpu.getReg(0)) === -4n;
        const retPC = (eintr && syscalls.pendingSignals.length > 0 && syscalls.anyPendingRestart()) ? svcPC : svcPC + 4n;
        if (syscalls.pendingSignals.length) {
          const next = deliverQueuedSignals(retPC);
          if (next === null) return;
          pc = next;
        } else {
          pc = retPC;
        }
        continue;
      }
      // rt_sigreturn: restore the pre-signal context. the resume PC comes from the signal frame,
      // NOT svcPC+4 — the generic 'ok' path below would clobber it, sending every handler return
      // into garbage after the trampoline. this is why signal handlers never returned correctly.
      if (nr === 139) {
        pc = syscalls.handleSigreturn(cpu, memory);
        continue;
      }
      // kill/tkill/tgkill: cross-worker signal delivery via the procTable
      if (nr === 129 || nr === 130 || nr === 131) {
        handleKill(nr, cpu);
        pc = svcPC + 4n;
        if (syscalls.pendingSignals.length) { // kill to self
          const next = deliverQueuedSignals(pc);
          if (next === null) return;
          pc = next;
        }
        continue;
      }
      // setpgid/getpgid: process-group membership lives in the shared procTable, so these
      // must run here (they read/write another process's PROC_PGID slot) rather than in the
      // per-process syscall handler. correct groups are what make kill(0)/kill(-pgid) and
      // timeout's child-group isolation actually scope, instead of hitting every process.
      if (nr === 154) { handleSetpgid(cpu); pc = svcPC + 4n; continue; }
      if (nr === 155) { handleGetpgid(cpu); pc = svcPC + 4n; continue; }
      // clone(CLONE_THREAD): pthread_create. spawn a real thread-Worker sharing this process's memory
      // (a fork-style clone — SIGCHLD set — still falls through to the per-process fork handler).
      if (nr === 220) {
        const cloneFlags = Number(cpu.getReg(0));
        const CLONE_THREAD = 0x10000;
        if (cloneFlags & CLONE_THREAD) {
          if (handleCloneThread(cpu, svcPC)) return; // yielded for async promote; onmessage resumes
          pc = svcPC + 4n;
          continue;
        }
      }
      // futex: real cross-thread blocking on the shared heap. only a threaded process has shared
      // memory (Atomics.wait needs a SharedArrayBuffer); single-threaded futex stays the syscall stub.
      if (nr === 98 && threaded) {
        const interrupted = handleFutex(cpu);
        pc = svcPC + 4n;
        if (interrupted && syscalls.pendingSignals.length) {
          const next = deliverQueuedSignals(pc);
          if (next === null) return;
          pc = next;
        }
        continue;
      }
      // setsid: become leader of a brand-new session and group (pgid = own pid), detaching from
      // the parent's group so the new session's group signals no longer reach it.
      if (nr === 157) {
        Atomics.store(procTable, (procSlotOffset(mySlot) / 4) + PROC_PGID, myPid);
        cpu.setReg(0, BigInt(myPid));
        pc = svcPC + 4n;
        continue;
      }
      // fcntl advisory record locks (F_GETLK/F_SETLK/F_SETLKW + 64-bit variants) are cross-process,
      // so they round-trip to the kernel lock table rather than running in the per-process handler.
      if (nr === 25) {
        const lc = Number(cpu.getReg(1));
        if (lc === 5 || lc === 6 || lc === 7 || lc === 12 || lc === 13 || lc === 14) {
          handleFcntlLock(cpu, view, lc);
          pc = svcPC + 4n;
          continue;
        }
      }
      // nanosleep(101) / clock_nanosleep(115): actually suspend the process for the
      // requested duration (the shared handler is a no-op, which breaks sleep + busybox
      // timeout's watchdog). the worker blocks on its wait futex; a signal wakes it -> EINTR.
      if (nr === 101 || nr === 115) {
        const abs = nr === 115 && (Number(cpu.getReg(1)) & 1) !== 0; // TIMER_ABSTIME
        const reqAddr = Number(nr === 115 ? cpu.getReg(2) : cpu.getReg(0));
        const remAddr = Number(nr === 115 ? cpu.getReg(3) : cpu.getReg(1));
        handleNanosleep(cpu, view, reqAddr, remAddr, abs, nr === 115 ? Number(cpu.getReg(0)) : 1);
        pc = svcPC + 4n;
        if (syscalls.pendingSignals.length) {
          const next = deliverQueuedSignals(pc);
          if (next === null) return;
          pc = next;
        }
        continue;
      }
      if (nr === 59) {
        handlePipe2(cpu, view);
        pc = svcPC + 4n;
        continue;
      }
      // Intercept dup3 when source is a kernel pipe fd
      if (nr === 24 && kernelPipes.has(Number(cpu.getReg(0)))) {
        handleKernelPipeDup3(cpu);
        pc = svcPC + 4n;
        continue;
      }
      // Intercept read/write/writev/close on kernel pipe fds
      if ((nr === 63 || nr === 64 || nr === 66 || nr === 57) && kernelPipes.has(Number(cpu.getReg(0)))) {
        handleKernelPipeIO(nr, cpu, view);
        // an interruptible pipe read may return EINTR with a queued signal; deliver it (running
        // its handler or terminating). honor SA_RESTART by resuming at the SVC so the read reruns.
        const eintr = BigInt.asIntN(64, cpu.getReg(0)) === -4n;
        const retPC = (eintr && syscalls.pendingSignals.length > 0 && syscalls.anyPendingRestart()) ? svcPC : svcPC + 4n;
        if (syscalls.pendingSignals.length) {
          const next = deliverQueuedSignals(retPC);
          if (next === null) return;
          pc = next;
        } else {
          pc = retPC;
        }
        continue;
      }

      // a syscall that dereferences an out-of-range guest pointer throws a RangeError from the
      // underlying typed-array/DataView op. linux answers a bad user address with EFAULT, not by
      // killing the process, so convert it here instead of letting the worker crash.
      let result;
      try {
        result = syscalls.handle(cpu, memory);
      } catch (e) {
        if (e instanceof RangeError) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); result = { status: 'ok' as const }; }
        else throw e;
      }
      view = new DataView(memory.buffer);

      switch (result.status) {
        case 'ok':
          pc = svcPC + 4n;
          break;

        case 'exit':
          postExit(result.code);
          return;

        case 'fork':
          pc = handleFork(result.resumePC);
          break;

        case 'execve':
          pc = handleExec(result.path, result.argv, result.envp);
          break;

        case 'blocked_read':
          // Block on stdin via Atomics.wait on the stdin ring
          handleBlockedRead();
          // Retry the read syscall
          pc = svcPC;
          break;

        case 'bridge':
          // thread this process's creds into every filesystem bridge call so NativeFS can enforce
          // emulated POSIX ownership/permissions against the caller (fd-only ops ignore it).
          if (result.method.startsWith('fs_')) {
            (result.args as Record<string, unknown>).creds = syscalls.getCreds();
          }
          // Bridge calls go through main thread
          handleBridgeCall(result.method, result.args, result.apply);
          pc = svcPC + 4n;
          break;

        default:
          pc = svcPC + 4n;
          break;
      }
      // a blocking syscall may have queued a signal (e.g. wait4/read interrupted by SIGALRM
      // or an incoming kill). deliver it now, before executing the resumed instruction.
      if (syscalls.pendingSignals.length) {
        const next = deliverQueuedSignals(pc);
        if (next === null) return;
        pc = next;
      }
      continue;
    }
  }

  postExit(0);
}

// ── Syscall helpers ─────────────────────────────────────────────────────────

// pack a guest memory region into a sparse snapshot: only the non-zero 64KB pages travel. a
// fresh child memory is all-zero, so the omitted pages are already correct. the ~33MB contiguous
// fork copy (which forced every page resident and OOMed fork storms) becomes "the few MB the
// process actually touched". the zero-scan is a Uint32 word compare (exact — no float -0/NaN
// footguns) with early-exit, run once per fork (not a hot path). also reused by memory swap.
function buildSparseSnapshot(buffer: ArrayBufferLike, totalLen: number): SparseMemSnapshot {
  const pageCount = Math.ceil(totalLen / MEM_PAGE_SIZE);
  const wordsPerPage = MEM_PAGE_SIZE >>> 2;
  const u32 = new Uint32Array(buffer, 0, (pageCount * MEM_PAGE_SIZE) >>> 2);
  const bytes = new Uint8Array(buffer);
  const idx: number[] = [];
  for (let p = 0; p < pageCount; p++) {
    const wStart = p * wordsPerPage;
    let nz = false;
    for (let w = 0; w < wordsPerPage; w++) { if (u32[wStart + w] !== 0) { nz = true; break; } }
    if (nz) idx.push(p);
  }
  const data = new Uint8Array(idx.length * MEM_PAGE_SIZE);
  for (let i = 0; i < idx.length; i++) {
    const off = idx[i] * MEM_PAGE_SIZE;
    data.set(bytes.subarray(off, off + MEM_PAGE_SIZE), i * MEM_PAGE_SIZE);
  }
  return { pages: Uint32Array.from(idx), data: data.buffer, totalLen: pageCount * MEM_PAGE_SIZE };
}

// scatter a sparse snapshot into a (zero) destination memory, growing it to fit first.
function applySparseSnapshot(mem: WebAssembly.Memory, snap: SparseMemSnapshot): void {
  if (snap.totalLen > mem.buffer.byteLength) {
    const need = Math.ceil(snap.totalLen / MEM_PAGE_SIZE);
    const cur = mem.buffer.byteLength / MEM_PAGE_SIZE;
    if (need > cur) mem.grow(need - cur);
  }
  const dst = new Uint8Array(mem.buffer);
  const data = new Uint8Array(snap.data);
  for (let i = 0; i < snap.pages.length; i++) {
    dst.set(data.subarray(i * MEM_PAGE_SIZE, (i + 1) * MEM_PAGE_SIZE), snap.pages[i] * MEM_PAGE_SIZE);
  }
}

function handleFork(resumePC: bigint): bigint {
  // Snapshot our memory for the child (sparse: only touched pages)
  const snapSize = Math.min(
    Number(syscalls.getMmapBase()) + GUEST_MEM_BASE + 0x10000,
    memory.buffer.byteLength,
  );
  const memSnapshot = buildSparseSnapshot(memory.buffer, snapSize);
  const syscallState = syscalls.getState();

  // Store our PC so the child knows where to resume
  cpu.setPC(resumePC);

  // Serialize kernel pipe map for child inheritance
  const kernelPipesSnapshot = [...kernelPipes.entries()];

  // Bump the refcount of every inherited pipe end NOW, synchronously, on behalf of the child.
  // the child worker adds refs only in its async init, which runs arbitrarily later than the
  // parent resuming — so without this the parent can close both ends and read EOF before the
  // child ever registers, leaving the child's later write staring at zero readers (EPIPE) and
  // truncating the pipe. incrementing here (parent thread, before we return to guest code)
  // makes the child's ends accounted for the instant fork(2) returns.
  for (const [, info] of kernelPipesSnapshot) {
    const ring = new PipeRing(kernelSAB, pipeRingOffset(info.pipeIndex), PIPE_RING_SIZE);
    if (info.mode === 'write') ring.addWriter();
    else ring.addReader();
  }

  // Request fork from kernel
  (self as any).postMessage({
    type: 'fork',
    memSnapshot,
    syscallState,
    kernelPipesSnapshot,
    resumePC: resumePC.toString(),
  }, [memSnapshot.data]); // transfer the packed pages (zero-copy send)

  // Block until kernel responds with child PID
  Atomics.store(myMailbox, MBOX_READY, 0);
  Atomics.wait(myMailbox, MBOX_READY, 0);
  const childPid = Atomics.load(myMailbox, MBOX_VALUE0);

  if (childPid < 0) {
    // fork failed (kernel proc table full -> EAGAIN). the child never spawned, so undo the
    // pipe-end refcounts we optimistically bumped for it; otherwise every failed fork in a
    // storm leaks a reader/writer ref and the ring never reaches EOF.
    for (const [, info] of kernelPipesSnapshot) {
      const ring = new PipeRing(kernelSAB, pipeRingOffset(info.pipeIndex), PIPE_RING_SIZE);
      if (info.mode === 'write') ring.closeWriter();
      else ring.closeReader();
    }
    cpu.setReg(0, BigInt.asUintN(64, BigInt(childPid))); // -EAGAIN
    return resumePC;
  }

  // Parent: set X0 = childPid
  cpu.setReg(0, BigInt(childPid));
  return resumePC;
}

// ── threads (clone with CLONE_THREAD) ────────────────────────────────────────

const CLONE_PARENT_SETTID = 0x00100000;

// a clone(CLONE_THREAD) that's mid-promote: the run loop yielded after posting 'clone_thread' to the
// kernel; onmessage('promote') finishes it once the kernel hands back the (main-thread-created)
// shared memory. saved so the resume knows where to continue + how to set up the child.
let pendingClone: { svcPC: bigint; childStack: number; tls: number; parentTidPtr: number; childTidPtr: number; flags: number } | null = null;

// clone(CLONE_THREAD|CLONE_VM|...): spawn a new thread. WORKERS CANNOT ALLOCATE SHARED MEMORY on
// WKWebView/JSC (it OOMs), so the kernel (main thread) creates it. first thread: post 'clone_thread'
// and YIELD the run loop — onmessage('promote') copies our memory into the shared one, rebinds to
// threaded mode, and resumes. subsequent threads: the process memory is already shared and the kernel
// holds it, so a plain blocking round-trip spawns the thread. returns true if the run loop must yield.
function handleCloneThread(cpu: CpuState, svcPC: bigint): boolean {
  const flags = Number(cpu.getReg(0));
  const childStack = Number(cpu.getReg(1));
  const parentTidPtr = Number(cpu.getReg(2));
  const tls = Number(cpu.getReg(3));
  const childTidPtr = Number(cpu.getReg(4));

  if (!threaded) {
    pendingClone = { svcPC, childStack, tls, parentTidPtr, childTidPtr, flags };
    (self as any).postMessage({ type: 'clone_thread', curPages: memory.buffer.byteLength / MEM_PAGE_SIZE });
    return true; // yield the run loop; resumed by onmessage('promote')
  }

  // already threaded: the kernel owns this process's shared memory (keyed by tgid). blocking round-trip.
  Atomics.store(myMailbox, MBOX_READY, 0);
  (self as any).postMessage({
    type: 'spawn_thread', tgid: syscalls.tgid || myPid,
    startPC: (svcPC + 4n).toString(), childStack, tls, childTidPtr, flags,
    parentRegBase: stateBase,
  });
  Atomics.wait(myMailbox, MBOX_READY, 0);
  const tid = Atomics.load(myMailbox, MBOX_VALUE0);
  if (tid < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-11))); return false; } // EAGAIN
  if ((flags & CLONE_PARENT_SETTID) && parentTidPtr) new DataView(memory.buffer).setInt32(parentTidPtr + GUEST_MEM_BASE, tid, true);
  cpu.setReg(0, BigInt(tid));
  return false;
}

// onmessage('promote'): the kernel created the shared memory + allocated the child tid/reg-block.
// copy our current memory into it, switch to threaded mode, finish the pending clone, tell the kernel
// to spawn the thread-Worker (now that the shared memory holds our state), and resume the run loop.
function completePromote(sharedMemory: WebAssembly.Memory, tid: number, regBaseAddr: number, staleHigh: number): void {
  const pc = pendingClone!;
  const copyLen = memory.buffer.byteLength;
  const shared = new Uint8Array(sharedMemory.buffer);
  shared.set(new Uint8Array(memory.buffer)); // our full image defines [0, copyLen)
  // a REUSED spare still holds the previous tenant's bytes beyond our image: zero the stale guest span
  // [copyLen, staleHigh) plus the whole atomics shadow (stale reservation bits would corrupt LDXR/STXR).
  // beyond staleHigh the shared memory was never touched, so it's already zero; reg blocks are
  // overwritten before use. only the actually-used span is touched, so RSS stays bounded.
  if (staleHigh > copyLen) shared.fill(0, copyLen, staleHigh);
  if (staleHigh > 0) shared.fill(0, THREADED_SHADOW_BASE, THREADED_SHADOW_BASE + (THREAD_REGS_BASE >> 9));
  memory = sharedMemory;
  threaded = true;
  stateBase = 0;
  regBaseGlobal = new WebAssembly.Global({ value: 'i32', mutable: false }, 0);
  cpu = new CpuState(memory, stateBase);
  regionCache.clear();

  syscalls.setThreadIds(myPid, myPid); // the leader's gettid==getpid==tgid==pid; names our thread group
  syscalls.maxBytes = THREAD_REGS_BASE; // threaded guests cap brk/mmap below the register blocks

  // snapshot our register block (the clone-time state) INTO the child's reg block NOW, while we're
  // still paused. once we resume below we race the child-Worker's own read of our block, so the child
  // starts from a copy taken at the clone instant instead (parentRegBase == its own base -> no re-copy).
  new Uint8Array(memory.buffer).copyWithin(regBaseAddr, 0, STATE_SIZE);

  cpu.setReg(0, BigInt(tid)); // clone returns the child tid to the parent
  if ((pc.flags & CLONE_PARENT_SETTID) && pc.parentTidPtr) new DataView(memory.buffer).setInt32(pc.parentTidPtr + GUEST_MEM_BASE, tid, true);

  (self as any).postMessage({
    type: 'thread_go', tid, regBaseAddr,
    startPC: (pc.svcPC + 4n).toString(), childStack: pc.childStack, tls: pc.tls, childTidPtr: pc.childTidPtr,
    flags: pc.flags, parentRegBase: regBaseAddr, // block already holds the clone-time snapshot
  });
  const resume = pc.svcPC + 4n;
  pendingClone = null;
  runToCompletion(resume);
}

// onmessage('clone_fail'): the kernel couldn't create the shared memory or find a free slot. the
// promote never happened, so we're still single-threaded — resume the clone returning EAGAIN.
function completePromoteFail(): void {
  const pc = pendingClone!;
  cpu.setReg(0, BigInt.asUintN(64, BigInt(-11))); // EAGAIN
  const resume = pc.svcPC + 4n;
  pendingClone = null;
  runToCompletion(resume);
}

// futex(uaddr, op, val, timeout, ...): real cross-thread blocking on the SHARED guest memory via
// Atomics.wait/notify (only reachable when threaded, so memory.buffer is a SharedArrayBuffer).
// returns true if a signal interrupted a WAIT (caller delivers it). waits in bounded slices so an
// incoming kill/timer is still observed.
function handleFutex(cpu: CpuState): boolean {
  const uaddr = Number(cpu.getReg(0));
  const op = Number(cpu.getReg(1)) & 0x7f; // strip FUTEX_PRIVATE_FLAG / CLOCK_REALTIME
  const val = Number(cpu.getReg(2)) | 0;
  const i32 = new Int32Array(memory.buffer);
  const idx = (uaddr + GUEST_MEM_BASE) >> 2;

  if (op === 0) { // FUTEX_WAIT: sleep while *uaddr == val
    if (Atomics.load(i32, idx) !== val) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-11))); return false; } // EAGAIN
    const r = Atomics.wait(i32, idx, val, 1000); // 1s slice; spurious wakeups are posix-legal
    if (r === 'not-equal') { cpu.setReg(0, BigInt.asUintN(64, BigInt(-11))); return false; }
    // 'ok' (woken) or 'timed-out' (retry the slice): report success; the guest re-checks + re-waits.
    if (pollAndQueueSignals()) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-4))); return true; } // EINTR
    cpu.setReg(0, 0n);
    return false;
  }
  if (op === 1) { // FUTEX_WAKE: wake up to `val` waiters
    cpu.setReg(0, BigInt(Atomics.notify(i32, idx, val < 0 ? Infinity : val)));
    return false;
  }
  // FUTEX_REQUEUE / CMP_REQUEUE / WAKE_OP / bitset variants: minimal — wake, no requeue.
  if (op === 3 || op === 4 || op === 5 || op === 10 || op === 11) {
    cpu.setReg(0, BigInt(Atomics.notify(i32, idx, val < 0 ? Infinity : val)));
    return false;
  }
  cpu.setReg(0, 0n);
  return false;
}

function handleExec(path: string, argv: string[], envp: string[]): bigint {
  // read the NEW binary from the guest FS (real distros exec different binaries; busybox
  // multicall re-reads /bin/<applet> -> the busybox image). update elfBytes so any
  // later re-exec / interpreter reload sees the current image.
  let bytes = readGuestFile(path);

  // #! script: the kernel re-targets exec to the interpreter, passing the script path as
  // an argument. e.g. exec("/bin/ltp") with "#!/bin/sh" -> exec("/bin/sh", ["/bin/sh",
  // "/bin/ltp", ...args]). one level of indirection (the interpreter must be a real binary).
  if (bytes.length >= 2 && bytes[0] === 0x23 && bytes[1] === 0x21) { // '#!'
    let end = 2;
    while (end < bytes.length && end < 256 && bytes[end] !== 0x0a) end++;
    const m = new TextDecoder().decode(bytes.subarray(2, end)).trim().match(/^(\S+)(?:\s+(.*))?$/);
    if (m) {
      const interp = m[1];
      const optArg = m[2]?.trim();
      const newArgv = [interp];
      if (optArg) newArgv.push(optArg);
      newArgv.push(path, ...argv.slice(1));
      path = interp;
      argv = newArgv;
      bytes = readGuestFile(interp);
    }
  }
  elfBytes = bytes;

  // clear guest memory (best-effort; loadProgram overwrites the segments it needs)
  new Uint8Array(memory.buffer).fill(0, GUEST_MEM_BASE, GUEST_MEM_BASE + 0x4000000);

  const prog = loadProgram(memory, elfBytes, argv, envp, readGuestFile);
  // loadProgram may have grown (detached) memory; rebuild cpu + views afresh
  cpu = new CpuState(memory, stateBase);
  cpu.setSP(prog.sp);
  new DataView(memory.buffer).setBigUint64(stateBase + 0x118, BigInt(0x900000), true); // TLS

  syscalls.resetForExec(prog.brkStart, (fd) => handleBridgeCall('fs_close', { fd }, () => {}));
  regionCache.clear(); // new binary image
  syscalls.installSignalTrampoline(memory); // exec may have wiped low memory; re-install

  return prog.entry;
}

// suspend this process for the requested duration by parking on its wait futex. a signal
// (e.g. SIGTERM from busybox timeout's watchdog) wakes it early -> EINTR + remaining time.
function handleNanosleep(cpu: CpuState, view: DataView, reqAddr: number, remAddr: number, absolute: boolean, clockId: number): void {
  const sec = Number(view.getBigUint64(reqAddr + GUEST_MEM_BASE, true));
  const nsec = Number(view.getBigUint64(reqAddr + GUEST_MEM_BASE + 8, true));
  const targetNs = sec * 1e9 + nsec;
  const ms = absolute ? (targetNs - Number(syscalls.clockNs(clockId))) / 1e6 : targetNs / 1e6;
  if (ms <= 0) { cpu.setReg(0, 0n); return; }

  const slotBase = procSlotOffset(mySlot) / 4;
  const deadline = performance.now() + ms;
  let interrupted = false;
  for (;;) {
    const rem = deadline - performance.now();
    if (rem <= 0) break;
    Atomics.store(procTable, slotBase + PROC_WAIT_FUTEX, 0);
    Atomics.wait(procTable, slotBase + PROC_WAIT_FUTEX, 0, Math.min(rem, 1000));
    if (pollAndQueueSignals()) { interrupted = true; break; }
  }

  if (interrupted) {
    if (remAddr && !absolute) {
      const remMs = Math.max(0, deadline - performance.now());
      view.setBigUint64(remAddr + GUEST_MEM_BASE, BigInt(Math.floor(remMs / 1000)), true);
      view.setBigUint64(remAddr + GUEST_MEM_BASE + 8, BigInt(Math.round((remMs % 1000) * 1e6)), true);
    }
    cpu.setReg(0, BigInt.asUintN(64, -4n)); // -EINTR
  } else {
    if (remAddr && !absolute) {
      view.setBigUint64(remAddr + GUEST_MEM_BASE, 0n, true);
      view.setBigUint64(remAddr + GUEST_MEM_BASE + 8, 0n, true);
    }
    cpu.setReg(0, 0n);
  }
}

function handleWait4(cpu: CpuState, view: DataView): void {
  const targetPid = Number(BigInt.asIntN(32, cpu.getReg(0))); // pid argument (signed: -1 = any)
  const wstatusAddr = Number(cpu.getReg(1));
  const options = Number(cpu.getReg(2));
  const WNOHANG = 1;

  while (true) {
    // Scan process table for matching zombie children
    let foundZombie = -1;
    let hasLivingChild = false;

    for (let slot = 0; slot < 64; slot++) {
      const slotBase = procSlotOffset(slot) / 4;
      const childPid = Atomics.load(procTable, slotBase + PROC_PID);
      if (childPid === 0) continue; // empty slot
      const ppid = Atomics.load(procTable, slotBase + PROC_PPID);
      if (ppid !== myPid) continue; // not our child

      // Check if this child matches the target
      if (targetPid > 0 && childPid !== targetPid) continue;
      // targetPid == -1 means any child, targetPid == 0 means same pgid (treat as any)

      const state = Atomics.load(procTable, slotBase + PROC_STATE);
      if (state === PROC_ZOMBIE) {
        foundZombie = slot;
        break;
      }
      if (state === PROC_RUNNING) {
        hasLivingChild = true;
      }
    }

    if (foundZombie >= 0) {
      // Reap the zombie
      const slotBase = procSlotOffset(foundZombie) / 4;
      const childPid = Atomics.load(procTable, slotBase + PROC_PID);
      const exitCode = Atomics.load(procTable, slotBase + PROC_EXIT_CODE);

      // Write wstatus if requested
      if (wstatusAddr !== 0) {
        view.setUint32(wstatusAddr + GUEST_MEM_BASE, (exitCode & 0xFF) << 8, true);
      }

      // Clear the process table slot (reap)
      Atomics.store(procTable, slotBase + PROC_STATE, PROC_EMPTY);
      Atomics.store(procTable, slotBase + PROC_PID, 0);

      // Return child PID
      cpu.setReg(0, BigInt(childPid));
      return;
    }

    if (!hasLivingChild) {
      // No children at all — ECHILD
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-10))); // -ECHILD
      return;
    }

    if (options & WNOHANG) {
      // No zombie yet but don't block
      cpu.setReg(0, 0n);
      return;
    }

    // Block until a child exits: wait on our own slot's wait_futex
    // wake at the interval-timer deadline if one is armed (busybox timeout waits here),
    // else re-poll at 1s. this makes SIGALRM-based timeouts punctual.
    const rem = syscalls.timerRemainingMs();
    const waitMs = rem > 0 ? Math.min(1000, rem) : 1000;
    const mySlotBase = procSlotOffset(mySlot) / 4;
    Atomics.store(procTable, mySlotBase + PROC_WAIT_FUTEX, 0);
    Atomics.wait(procTable, mySlotBase + PROC_WAIT_FUTEX, 0, waitMs);
    // Loop back and re-check for zombies

    // a fired timer or an incoming signal interrupts the wait with EINTR; the run loop
    // delivers the queued signal (running the handler or terminating) on return.
    if (pollAndQueueSignals()) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-4))); // -EINTR
      return;
    }
  }
}

// kill(pid, sig) / tgkill(tgid, tid, sig): cross-worker signal delivery. sets the target's
// PROC_SIGNAL bit (bit position == signal number) and wakes it if it is blocked on its wait
// futex. handles pid>0 (specific), 0 (own pgroup), -1 (all but init), <-1 (specific pgroup).
function handleKill(nr: number, cpu: CpuState): void {
  const targetPid = Number(BigInt.asIntN(32, cpu.getReg(0)));
  const sig = nr === 131 ? Number(cpu.getReg(2)) : Number(cpu.getReg(1));
  const myPgid = Atomics.load(procTable, (procSlotOffset(mySlot) / 4) + PROC_PGID);
  let delivered = false;
  for (let slot = 0; slot < 64; slot++) {
    const base = procSlotOffset(slot) / 4;
    const pid = Atomics.load(procTable, base + PROC_PID);
    if (pid === 0) continue;
    const pgid = Atomics.load(procTable, base + PROC_PGID);
    let match: boolean;
    if (targetPid > 0) match = pid === targetPid;
    else if (targetPid === 0) match = pgid === myPgid;
    else if (targetPid === -1) match = pid !== 1 && pid !== myPid; // broadcast excludes the sender
    else match = pgid === -targetPid;
    if (!match) continue;
    delivered = true;
    if (sig === 0) continue; // existence check only
    if (pid === myPid) { syscalls.raiseSignal(sig); continue; }
    // SIGKILL is unconditional and un-catchable: a target blocked in Atomics.wait (a pipe
    // read, the bridge mailbox, ...) never runs guest code to observe the signal bit and
    // self-terminate, so setting a bit can't kill it. only the kernel, which owns the Worker
    // objects, can. mirror linux: SIGKILL is enforced by the kernel, the target has no say.
    if (sig === 9) { (self as any).postMessage({ type: 'kill', pid, sig }); continue; }
    Atomics.or(procTable, base + PROC_SIGNAL, 1 << sig);
    // wake the target if it is parked on its wait futex (e.g. blocked in wait4)
    Atomics.store(procTable, base + PROC_WAIT_FUTEX, 1);
    Atomics.notify(procTable, base + PROC_WAIT_FUTEX);
  }
  cpu.setReg(0, delivered ? 0n : BigInt.asUintN(64, BigInt(-3))); // -ESRCH
}

// setpgid(pid, pgid): move a process into a process group. pid 0 means the caller; pgid 0
// means "use pid as the new group id" (become a group leader). the membership is a field in
// the target's shared procTable slot, so we write it directly here.
function handleSetpgid(cpu: CpuState): void {
  let pid = Number(BigInt.asIntN(32, cpu.getReg(0)));
  let pgid = Number(BigInt.asIntN(32, cpu.getReg(1)));
  if (pid === 0) pid = myPid;
  if (pgid === 0) pgid = pid;
  if (pgid < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; } // -EINVAL
  for (let slot = 0; slot < 64; slot++) {
    const base = procSlotOffset(slot) / 4;
    if (Atomics.load(procTable, base + PROC_PID) !== pid) continue;
    Atomics.store(procTable, base + PROC_PGID, pgid);
    cpu.setReg(0, 0n);
    return;
  }
  cpu.setReg(0, BigInt.asUintN(64, BigInt(-3))); // -ESRCH
}

// getpgid(pid): report a process's group. pid 0 means the caller.
function handleGetpgid(cpu: CpuState): void {
  let pid = Number(BigInt.asIntN(32, cpu.getReg(0)));
  if (pid === 0) pid = myPid;
  for (let slot = 0; slot < 64; slot++) {
    const base = procSlotOffset(slot) / 4;
    if (Atomics.load(procTable, base + PROC_PID) !== pid) continue;
    cpu.setReg(0, BigInt(Atomics.load(procTable, base + PROC_PGID)));
    return;
  }
  cpu.setReg(0, BigInt.asUintN(64, BigInt(-3))); // -ESRCH
}

// fcntl(fd, F_GETLK/F_SETLK/F_SETLKW, &flock): round-trip a record-lock request to the kernel's
// shared lock table. the flock struct (aarch64) is { i16 l_type; i16 l_whence; i64 l_start;
// i64 l_len; i32 l_pid; }. we key locks by the fd's resolved path; a non-file fd has no path and
// gets the historical always-grant behaviour (locks on pipes/sockets are meaningless here).
function handleFcntlLock(cpu: CpuState, view: DataView, cmd: number): void {
  const fd = Number(cpu.getReg(0));
  const flockAddr = Number(cpu.getReg(2));
  if (badGuestPtr(new Uint8Array(memory.buffer), flockAddr, 32)) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; // EFAULT: bad flock pointer
  }
  const flockPtr = flockAddr + GUEST_MEM_BASE;
  const ltype = view.getInt16(flockPtr, true);
  const whence = view.getInt16(flockPtr + 2, true);
  const lstart = Number(view.getBigInt64(flockPtr + 8, true));
  const len = Number(view.getBigInt64(flockPtr + 16, true));
  const path = syscalls.fdPath(fd);
  if (path === null) { cpu.setReg(0, 0n); return; } // non-file fd: nothing to lock, succeed

  // resolve l_whence to an absolute start. SEEK_SET is the common case; SEEK_CUR/END add the
  // fd's current offset / file size, which the syscall handler tracks for native fds.
  let start = lstart;
  if (whence === 1) start += syscalls.fdOffset(fd);
  else if (whence === 2) start += syscalls.fdSize(fd);
  if (start < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; } // EINVAL

  Atomics.store(myMailbox, MBOX_READY, 0);
  (self as any).postMessage({ type: 'lock_req', path, cmd, ltype, start, len });
  Atomics.wait(myMailbox, MBOX_READY, 0);
  const r = Atomics.load(myMailbox, MBOX_VALUE0);

  if (cmd === 5 || cmd === 12) { // F_GETLK: write the (conflicting) lock back into the struct
    const rtype = Atomics.load(myMailbox, MBOX_VALUE0 + 1);
    view.setInt16(flockPtr, rtype, true);
    if (rtype !== 2) {
      const rpid = Atomics.load(myMailbox, MBOX_VALUE0 + 2);
      const cstart = (Atomics.load(myMailbox, MBOX_VALUE0 + 3) >>> 0) + Atomics.load(myMailbox, MBOX_VALUE0 + 4) * 4294967296;
      const clen = (Atomics.load(myMailbox, MBOX_VALUE0 + 5) >>> 0) + Atomics.load(myMailbox, MBOX_VALUE0 + 6) * 4294967296;
      view.setInt16(flockPtr + 2, 0, true); // l_whence = SEEK_SET
      view.setBigInt64(flockPtr + 8, BigInt(cstart), true);
      view.setBigInt64(flockPtr + 16, BigInt(clen), true);
      view.setInt32(flockPtr + 24, rpid, true);
    }
    cpu.setReg(0, 0n);
    return;
  }
  cpu.setReg(0, r < 0 ? BigInt.asUintN(64, BigInt(r)) : 0n);
}

// ── Kernel pipe dup/close ────────────────────────────────────────────────────

function handleKernelPipeDup3(cpu: CpuState): void {
  const oldFd = Number(cpu.getReg(0));
  const newFd = Number(cpu.getReg(1));
  const pipeInfo = kernelPipes.get(oldFd)!;

  // If newFd already has a kernel pipe, close it first
  const existing = kernelPipes.get(newFd);
  if (existing) {
    const ring = new PipeRing(kernelSAB, pipeRingOffset(existing.pipeIndex), PIPE_RING_SIZE);
    if (existing.mode === 'write') ring.closeWriter();
    else ring.closeReader();
    kernelPipes.delete(newFd);
  }

  // Duplicate: newFd now points to the same pipe ring, increment refcount
  kernelPipes.set(newFd, { ...pipeInfo });
  const dupRing = new PipeRing(kernelSAB, pipeRingOffset(pipeInfo.pipeIndex), PIPE_RING_SIZE);
  if (pipeInfo.mode === 'write') dupRing.addWriter();
  else dupRing.addReader();

  // Also let the SyscallHandler know (it manages fd table for other ops like fcntl)
  syscalls.registerFd(newFd, {
    type: pipeInfo.mode === 'read' ? 'pipe_read' : 'pipe_write',
    pipe: { data: [], writeClosed: false, readClosed: false },
  });

  cpu.setReg(0, BigInt(newFd));
}

// ── Kernel pipe operations ──────────────────────────────────────────────────

function allocPipeSlot(): number {
  // Ask kernel for a pipe slot via mailbox
  Atomics.store(myMailbox, MBOX_READY, 0);
  (self as any).postMessage({ type: 'pipe_alloc' });
  Atomics.wait(myMailbox, MBOX_READY, 0);
  return Atomics.load(myMailbox, MBOX_VALUE0);
}

function handlePipe2(cpu: CpuState, view: DataView): void {
  const pipefdAddr = Number(cpu.getReg(0));
  const pipeIndex = allocPipeSlot();
  if (pipeIndex < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-23))); return; } // ENFILE: ring pool exhausted

  // Initialize the ring buffer (the kernel already claimed+reset it; this is idempotent)
  const ring = new PipeRing(kernelSAB, pipeRingOffset(pipeIndex), PIPE_RING_SIZE);
  ring.reset();

  // Allocate two local fds
  const readFd = nextLocalFd++;
  const writeFd = nextLocalFd++;

  kernelPipes.set(readFd, { pipeIndex, mode: 'read' });
  kernelPipes.set(writeFd, { pipeIndex, mode: 'write' });

  // Also register in the SyscallHandler's fd table so dup2/close know about them
  // We use a shim pipe buffer that the SyscallHandler won't actually read from
  // (the Worker intercepts all I/O on these fds)
  syscalls.registerFd(readFd, { type: 'pipe_read', pipe: { data: [], writeClosed: false, readClosed: false } });
  syscalls.registerFd(writeFd, { type: 'pipe_write', pipe: { data: [], writeClosed: false, readClosed: false } });

  // Write [readFd, writeFd] to guest memory
  view.setInt32(pipefdAddr + GUEST_MEM_BASE, readFd, true);
  view.setInt32(pipefdAddr + GUEST_MEM_BASE + 4, writeFd, true);

  cpu.setReg(0, 0n);
}

function handleKernelPipeIO(nr: number, cpu: CpuState, view: DataView): void {
  const fd = Number(cpu.getReg(0));
  const pipeInfo = kernelPipes.get(fd)!;
  const ring = new PipeRing(kernelSAB, pipeRingOffset(pipeInfo.pipeIndex), PIPE_RING_SIZE);

  if (nr === 57) {
    // close
    if (pipeInfo.mode === 'write') {
      ring.closeWriter();
    } else {
      ring.closeReader();
    }
    kernelPipes.delete(fd);
    cpu.setReg(0, 0n);
    return;
  }

  if (nr === 63) {
    // read — interruptible: a pending signal (armed timer, incoming kill/term) breaks the
    // block with EINTR instead of parking forever, so a pipe-blocked process stays reapable.
    const buf = Number(cpu.getReg(1));
    const count = Number(cpu.getReg(2));
    if (count > 0 && badGuestPtr(new Uint8Array(memory.buffer), buf, count)) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; // -EFAULT
    }
    const dst = new Uint8Array(count);
    const n = ring.read(dst, interruptibleWaiter);
    if (n === -1) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-4))); return; } // -EINTR
    if (n > 0) {
      new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, n).set(dst.subarray(0, n));
    }
    cpu.setReg(0, BigInt(n));
    return;
  }

  if (nr === 64) {
    // write
    const buf = Number(cpu.getReg(1));
    const count = Number(cpu.getReg(2));
    if (count > 0 && badGuestPtr(new Uint8Array(memory.buffer), buf, count)) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; // -EFAULT
    }
    const src = new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, count);
    if (syscalls.isNonBlockingFd(fd)) {
      const n = ring.write(src);
      if (n === -1) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-32))); return; } // EPIPE
      if (n === 0 && count > 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-11))); return; } // EAGAIN
      cpu.setReg(0, BigInt(n));
      return;
    }
    // blocking fd: park until the whole buffer is written (a slow reader must not lose data).
    const n = ring.writeAll(src, interruptibleWaiter);
    if (n === -1) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-32))); return; }        // EPIPE
    if (n === -2) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-4))); return; }         // EINTR
    cpu.setReg(0, BigInt(n));
    return;
  }

  if (nr === 66) {
    // writev — gather write from iovec array
    const iovAddr = Number(cpu.getReg(1));
    const iovcnt = Number(cpu.getReg(2));
    let totalWritten = 0;
    const bytes = new Uint8Array(memory.buffer);
    if (iovcnt > 0 && badGuestPtr(bytes, iovAddr, iovcnt * 16)) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; // -EFAULT
    }
    for (let i = 0; i < iovcnt; i++) {
      const base = view.getBigUint64(iovAddr + i * 16 + GUEST_MEM_BASE, true);
      const len = view.getBigUint64(iovAddr + i * 16 + 8 + GUEST_MEM_BASE, true);
      if (len === 0n) continue;
      if (badGuestPtr(bytes, Number(base), Number(len))) {
        if (totalWritten === 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; } // -EFAULT
        break;
      }
      const src = bytes.subarray(Number(base) + GUEST_MEM_BASE, Number(base) + GUEST_MEM_BASE + Number(len));
      const n = ring.write(src);
      if (n === -1) {
        if (totalWritten === 0) {
          cpu.setReg(0, BigInt.asUintN(64, BigInt(-32))); // -EPIPE
          return;
        }
        break;
      }
      totalWritten += n;
      if (n < Number(len)) break; // short write
    }
    cpu.setReg(0, BigInt(totalWritten));
    return;
  }
}

function handleBlockedRead(): void {
  if (stdinRing) {
    // Block until data appears in stdin ring
    // The main thread writes data and notifies via Atomics
    const stdinRingIdx = Atomics.load(procTable, (procSlotOffset(mySlot) / 4) + 7);
    const ringI32 = new Int32Array(kernelSAB, stdinRingOffset(stdinRingIdx), 4);
    while (stdinRing.empty && !stdinRing.writerClosed) {
      const wp = Atomics.load(ringI32, 0);
      // bound the wait so an armed timer / incoming kill is observed within <=1s even
      // though it notifies the wait futex, not the stdin ring.
      const rem = syscalls.timerRemainingMs();
      Atomics.wait(ringI32, 0, wp, rem > 0 ? Math.min(1000, rem) : 1000);
      // any pending signal interrupts the read; the run loop delivers it (handler or exit).
      if (pollAndQueueSignals()) return;
    }
  }
}

// read an entire file from the guest FS (bridge) synchronously, in <=slot-sized chunks.
// used to load the dynamic interpreter (ld.so) and execve targets. slow for big files
// (many round-trips) but correct; a bulk-transfer channel is a later optimization.
function readGuestFile(path: string): Uint8Array {
  const fdh = { fd: -1 };
  handleBridgeCall('fs_open', { path, flags: 0, mode: 0 }, (r) => { fdh.fd = r.ret ?? -1; });
  if (fdh.fd < 0) throw new Error(`readGuestFile: cannot open ${path} (ret ${fdh.fd})`);
  const chunks: Uint8Array[] = [];
  let total = 0;
  const CHUNK = 12288; // raw bytes per read; fits base64 in the 16KB response slot
  try {
    for (;;) {
      const h = { data: new Uint8Array(0) };
      handleBridgeCall('fs_read', { fd: fdh.fd, count: CHUNK }, (r) => {
        if (typeof r.data === 'string' && r.data.length) {
          const bin = atob(r.data);
          const u = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
          h.data = u;
        }
      });
      if (h.data.length === 0) break;
      chunks.push(h.data); total += h.data.length;
      if (h.data.length < CHUNK) break;
    }
  } finally {
    handleBridgeCall('fs_close', { fd: fdh.fd }, () => {});
  }
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

function handleBridgeCall(method: string, args: Record<string, unknown>, apply: (resp: any) => void): void {
  // Send bridge request to main thread and block until response
  (self as any).postMessage({ type: 'bridge_call', id: 0, method, args });

  // Block until kernel signals response is ready
  Atomics.store(myMailbox, MBOX_READY, 0);
  while (Atomics.load(myMailbox, MBOX_READY) === 0) {
    Atomics.wait(myMailbox, MBOX_READY, 0);
    // Woke up — check if it's a signal rather than a bridge response
    if (Atomics.load(myMailbox, MBOX_READY) === 0) {
      // Not a bridge response — check for SIGINT
      const sigSlot = (procSlotOffset(mySlot) / 4) + 6; // PROC_SIGNAL offset
      const sig = Atomics.load(procTable, sigSlot);
      if (sig & SIG_INT) {
        Atomics.and(procTable, sigSlot, ~SIG_INT);
        // Check if this process has a custom SIGINT handler (shell does)
        if (syscalls.hasSignalHandler(2)) {
          // Deliver signal through the normal dispatch path — queue it and return
          // The bridge call will get a fake EINTR response
          syscalls.raiseSignal(2);
          // Return EINTR to the caller
          apply({ ret: -4 }); // EINTR
          return;
        }
        // Default action: terminate
        postExit(130); // 128 + SIGINT(2)
        throw new Error('SIGINT');
      }
    }
  }

  // Read response from the bridge response area in kernel SAB
  const respOffset = BRIDGE_RESP_OFFSET + mySlot * BRIDGE_RESP_SLOT_SIZE;
  const respView = new DataView(kernelSAB, respOffset, BRIDGE_RESP_SLOT_SIZE);

  // Response layout: [ret:i32][metaCount:u16][dataLen:u32][meta: metaCount×f64][data: dataLen bytes]
  const ret = respView.getInt32(0, true);
  const metaCount = respView.getUint16(4, true);
  const dataLen = respView.getUint32(6, true);
  const dataOffset = 10 + metaCount * 8;

  const resp: any = { ret };

  // Read meta values
  if (metaCount > 0) {
    resp._meta = new Array(metaCount);
    for (let i = 0; i < metaCount; i++) {
      resp._meta[i] = respView.getFloat64(10 + i * 8, true);
    }
  }

  // Read data
  if (dataLen > 0) {
    const dataBytes = new Uint8Array(kernelSAB, respOffset + dataOffset, dataLen);
    // Store raw bytes — the apply callback decides how to interpret (base64 string, JSON, UTF-8)
    let str = '';
    for (let i = 0; i < dataLen; i++) str += String.fromCharCode(dataBytes[i]);
    resp.data = btoa(str);
  }

  // same guard as the synchronous syscall path: a bridge result copied to a bad guest buffer
  // faults with EFAULT rather than crashing the worker.
  try {
    apply(resp);
  } catch (e) {
    if (e instanceof RangeError) cpu.setReg(0, BigInt.asUintN(64, BigInt(-14)));
    else throw e;
  }
}

// ── I/O helpers ─────────────────────────────────────────────────────────────

function readStdin(count: number): Uint8Array | null {
  if (!stdinRing) return null;
  if (stdinRing.empty && !stdinRing.writerClosed) return null; // signal blocked_read

  const buf = new Uint8Array(count);
  const n = stdinRing.tryRead(buf);
  if (n === 0 && stdinRing.writerClosed) {
    return new Uint8Array(0); // EOF
  }
  if (n === 0) return null; // no data, will block
  return buf.subarray(0, n);
}

function waitStdin(timeoutMs: number): 'data' | 'timeout' {
  if (!stdinRing) return 'timeout';
  if (!stdinRing.empty || stdinRing.writerClosed) return 'data';
  const stdinRingIdx = Atomics.load(procTable, (procSlotOffset(mySlot) / 4) + 7);
  if (stdinRingIdx < 0) return 'timeout';
  const ringI32 = new Int32Array(kernelSAB, stdinRingOffset(stdinRingIdx), 4);
  const wp = Atomics.load(ringI32, 0);
  const result = timeoutMs < 0
    ? Atomics.wait(ringI32, 0, wp)
    : Atomics.wait(ringI32, 0, wp, timeoutMs);
  return result === 'timed-out' ? 'timeout' : 'data';
}

function postWrite(fd: number, data: Uint8Array): void {
  (self as any).postMessage({ type: 'write', fd, data: data.slice() });
}

function postExit(code: number): void {
  // CLONE_CHILD_CLEARTID: a thread exiting must zero its clear_child_tid word and futex-wake it, so
  // a joiner blocked in pthread_join (futex-waiting on that word) is released.
  if (threaded && clearChildTid) {
    try {
      const i32 = new Int32Array(memory.buffer);
      const idx = (clearChildTid + GUEST_MEM_BASE) >> 2;
      Atomics.store(i32, idx, 0);
      Atomics.notify(i32, idx, Infinity);
    } catch {}
    clearChildTid = 0;
  }
  // Close all open kernel pipe fds before exiting
  for (const [fd, info] of kernelPipes) {
    const ring = new PipeRing(kernelSAB, pipeRingOffset(info.pipeIndex), PIPE_RING_SIZE);
    if (info.mode === 'write') ring.closeWriter();
    else ring.closeReader();
  }
  kernelPipes.clear();

  // a threaded process reports its touched high-water so the kernel, when it parks this shared memory
  // as the reusable spare, knows how much of it the next tenant must zero (only the used span is stale).
  const high = threaded ? Number(syscalls.getMmapBase()) + GUEST_MEM_BASE : 0;
  (self as any).postMessage({ type: 'exit', pid: myPid, code, high });
}
