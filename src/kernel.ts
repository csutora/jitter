/**
 * Main-thread kernel for multicore process management.
 *
 * The Kernel runs on the main JS thread (which also owns xterm.js and the Swift bridge).
 * Each emulated Linux process runs in its own Web Worker with JIT-compiled WASM.
 *
 * Responsibilities:
 *   - Spawn/terminate Workers for fork/exec/exit
 *   - Manage the process table in SharedArrayBuffer
 *   - Route stdin from xterm.js to the foreground process's stdin ring
 *   - Route stdout/stderr from Workers to xterm.js
 *   - Relay bridge calls (native filesystem) from Workers to Swift
 *   - Deliver signals (Ctrl+C → SIGINT)
 */

import type { KernelMessage, WorkerInitConfig } from './kernel-ipc.js';
import {
  KERNEL_SAB_SIZE,
  PROC_TABLE_OFFSET, PROC_SLOT_SIZE, PROC_PID, PROC_PPID, PROC_PGID,
  PROC_STATE, PROC_EXIT_CODE, PROC_WAIT_FUTEX, PROC_SIGNAL, PROC_STDIN_RING,
  PROC_EMPTY, PROC_RUNNING, PROC_ZOMBIE,
  MAILBOX_OFFSET, MAILBOX_SLOT_SIZE, MBOX_READY, MBOX_VALUE0,
  GLOBAL_STATE_OFFSET, GLOBAL_NEXT_PID, GLOBAL_NEXT_PIPE, GLOBAL_NEXT_STDIN, GLOBAL_FG_PGID,
  STDIN_RING_OFFSET, STDIN_RING_SIZE,
  SIG_INT,
  BRIDGE_RESP_OFFSET, BRIDGE_RESP_SLOT_SIZE,
  procSlotOffset, mailboxSlotOffset, stdinRingOffset,
} from './kernel-ipc.js';
import { PipeRing } from './pipe-ring.js';
import type { NativeBridge } from './bridge.js';

export interface KernelConfig {
  /** The ELF binary to boot (busybox) */
  elfBytes: Uint8Array;
  /** URL of the worker script (worker.js) */
  workerURL: string;
  /** Callback for terminal output */
  onOutput: (data: Uint8Array) => void;
  /** Optional native bridge for Swift filesystem */
  bridge?: NativeBridge;
}

export class Kernel {
  private sab: SharedArrayBuffer;
  private procTable: Int32Array;
  private globalState: Int32Array;
  private workers = new Map<number, Worker>();
  private elfBytes: Uint8Array;
  private workerURL: string;
  private onOutput: (data: Uint8Array) => void;
  private bridge?: NativeBridge;

  // Stdin ring for the foreground process
  private fgStdinRing: PipeRing | null = null;

  constructor(config: KernelConfig) {
    this.elfBytes = config.elfBytes;
    this.workerURL = config.workerURL;
    this.onOutput = config.onOutput;
    this.bridge = config.bridge;

    // Allocate kernel shared memory
    this.sab = new SharedArrayBuffer(KERNEL_SAB_SIZE);
    this.procTable = new Int32Array(this.sab);
    this.globalState = new Int32Array(this.sab, GLOBAL_STATE_OFFSET, 16);

    // Initialize global state
    Atomics.store(this.globalState, GLOBAL_NEXT_PID, 2); // PID 1 = init process
    Atomics.store(this.globalState, GLOBAL_NEXT_STDIN, 0);
    Atomics.store(this.globalState, GLOBAL_NEXT_PIPE, 0);
    Atomics.store(this.globalState, GLOBAL_FG_PGID, 1);
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

  /** Send Ctrl+C (SIGINT) to the foreground process group. */
  sendInterrupt(): void {
    const fgPgid = Atomics.load(this.globalState, GLOBAL_FG_PGID);
    console.log(`[kernel] sendInterrupt: fgPgid=${fgPgid}`);
    // Signal all processes in the foreground group
    for (let slot = 0; slot < 64; slot++) {
      const slotBase = procSlotOffset(slot) / 4;
      const state = Atomics.load(this.procTable, slotBase + PROC_STATE);
      if (state === PROC_RUNNING) {
        const pgid = Atomics.load(this.procTable, slotBase + PROC_PGID);
        if (pgid === fgPgid) {
          console.log(`[kernel] SIGINT → slot=${slot} pid=${Atomics.load(this.procTable, slotBase + PROC_PID)}`);
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
    throw new Error('Process table full (64 slots)');
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
    memoryBuffer?: ArrayBuffer;
    syscallState?: unknown;
    kernelPipesSnapshot?: unknown;
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

    // Create Worker
    const worker = new Worker(this.workerURL);
    this.workers.set(opts.pid, worker);

    // Set up message handler
    worker.onmessage = (e) => this.handleWorkerMessage(opts.pid, slot, e.data);
    worker.onerror = (e) => {
      console.error(`[kernel] Worker ${opts.pid} error:`, e.message);
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
      memoryBuffer: opts.memoryBuffer,
      syscallState: opts.syscallState,
      kernelPipesSnapshot: opts.kernelPipesSnapshot as any,
    };

    // Send init — transfer memoryBuffer if present (zero-copy)
    const transferList: Transferable[] = [];
    if (config.memoryBuffer) {
      transferList.push(config.memoryBuffer);
    }
    console.log(`[kernel] Sending init to Worker PID ${opts.pid}, elfBytes=${opts.elfBytes.length}, hasMem=${!!config.memoryBuffer}`);
    worker.postMessage({ type: 'init', config }, transferList);
  }

  // ── Worker message handling ───────────────────────────────────────────────

  private handleWorkerMessage(pid: number, slot: number, msg: KernelMessage): void {
    switch (msg.type) {
      case 'fork':
        this.handleFork(pid, slot, msg);
        break;

      case 'exit':
        this.handleExit(pid, slot, msg.code);
        break;

      case 'write':
        this.onOutput((msg as any).data);
        break;

      case 'pipe_alloc':
        this.handlePipeAlloc(pid, slot);
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

  private handleFork(parentPid: number, parentSlot: number, msg: KernelMessage & { type: 'fork' }): void {
    const childPid = this.allocPid();
    const childSlot = this.findFreeSlot();

    // Spawn child with parent's memory snapshot
    this.spawnProcess({
      pid: childPid,
      ppid: parentPid,
      pgid: Atomics.load(this.procTable, (procSlotOffset(parentSlot) / 4) + PROC_PGID),
      slot: childSlot,
      elfBytes: this.elfBytes,
      argv: [], // Not used for fork (child inherits parent's state)
      envp: [],
      memoryBuffer: msg.memoryBuffer,
      syscallState: msg.syscallState,
      kernelPipesSnapshot: (msg as any).kernelPipesSnapshot,
    });

    // Notify parent with child PID via mailbox
    const mailbox = new Int32Array(this.sab, mailboxSlotOffset(parentSlot), MAILBOX_SLOT_SIZE / 4);
    Atomics.store(mailbox, MBOX_VALUE0, childPid);
    Atomics.store(mailbox, MBOX_READY, 1);
    Atomics.notify(mailbox, MBOX_READY);
  }

  private handleExit(pid: number, slot: number, code: number): void {
    const slotBase = procSlotOffset(slot) / 4;

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

    // Terminate the Worker
    const worker = this.workers.get(pid);
    if (worker) {
      worker.terminate();
      this.workers.delete(pid);
    }

    // If PID 1 (init) exits, all processes are dead
    if (pid === 1) {
      // Terminate all remaining workers
      for (const [p, w] of this.workers) {
        w.terminate();
      }
      this.workers.clear();
    }
  }

  private handlePipeAlloc(pid: number, slot: number): void {
    const pipeSlot = Atomics.add(this.globalState, GLOBAL_NEXT_PIPE, 1);
    // Respond via mailbox
    const mailbox = new Int32Array(this.sab, mailboxSlotOffset(slot), MAILBOX_SLOT_SIZE / 4);
    Atomics.store(mailbox, MBOX_VALUE0, pipeSlot);
    Atomics.store(mailbox, MBOX_READY, 1);
    Atomics.notify(mailbox, MBOX_READY);
  }

  private handleBridgeCall(pid: number, slot: number, msg: KernelMessage & { type: 'bridge_call' }): void {
    if (!this.bridge) {
      this.writeBridgeResponse(slot, -38, null);
      return;
    }

    console.log(`[kernel] bridge call: ${(msg as any).method} from PID ${pid}`);
    this.bridge.call((msg as any).method, (msg as any).args as Record<string, any>).then((resp) => {
      console.log(`[kernel] bridge response: ${(msg as any).method} ret=${(resp as any).ret}`);
      // Write response to bridge response area in SAB
      const ret = (resp as any).ret ?? 0;
      const data = (resp as any).data as string | undefined; // base64 string

      let rawData: Uint8Array | null = null;
      if (data) {
        // Decode base64 to raw bytes for SAB storage
        rawData = Uint8Array.from(atob(data), c => c.charCodeAt(0));
      }

      this.writeBridgeResponse(slot, ret, rawData);
    });
  }

  private writeBridgeResponse(slot: number, ret: number, data: Uint8Array | null): void {
    const respOffset = BRIDGE_RESP_OFFSET + slot * BRIDGE_RESP_SLOT_SIZE;
    const respView = new DataView(this.sab, respOffset, BRIDGE_RESP_SLOT_SIZE);

    // Write response: [ret: i32] [dataLen: u32] [data: bytes...]
    respView.setInt32(0, ret, true);
    if (data && data.length > 0 && data.length <= BRIDGE_RESP_SLOT_SIZE - 8) {
      respView.setUint32(4, data.length, true);
      new Uint8Array(this.sab, respOffset + 8, data.length).set(data);
    } else {
      respView.setUint32(4, 0, true);
    }

    // Wake the Worker
    const mailbox = new Int32Array(this.sab, mailboxSlotOffset(slot), MAILBOX_SLOT_SIZE / 4);
    Atomics.store(mailbox, MBOX_READY, 1);
    Atomics.notify(mailbox, MBOX_READY);
  }
}
