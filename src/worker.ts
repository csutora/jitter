/**
 * Web Worker entry point for multicore process execution.
 */
console.log('[worker] Script loaded');

import { HALT_PC, SVC_PC, GUEST_MEM_BASE, ITER_OFFSET, PC_OFFSET, STATE_SIZE } from './types.js';
import type { CompiledRegion, CompiledBlockModule, CompiledRegionModule } from './types.js';
import { CpuState } from './cpu.js';
import { detectRegion, detectBlock } from './block.js';
import { translateBlock, translateRegion, instantiateBlock, instantiateRegion } from './translate.js';
import { SyscallHandler } from './syscall.js';
import { InMemoryVFS } from './vfs.js';
import { loadElf, setupStack } from './elf.js';
import { PipeRing } from './pipe-ring.js';
import type { WorkerInitConfig, KernelMessage } from './kernel-ipc.js';
import {
  PROC_SLOT_SIZE, PROC_STATE, PROC_EXIT_CODE, PROC_WAIT_FUTEX, PROC_SIGNAL,
  PROC_EMPTY, PROC_ZOMBIE, PROC_RUNNING,
  PROC_PID, PROC_PPID,
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
  }
};

// ── Initialization ──────────────────────────────────────────────────────────

function init(config: WorkerInitConfig) {
  myPid = config.pid;
  mySlot = config.slot;
  kernelSAB = config.kernelSAB;
  elfBytes = config.elfBytes;

  // Set up SAB views
  procTable = new Int32Array(kernelSAB);
  myMailbox = new Int32Array(kernelSAB, mailboxSlotOffset(mySlot), MAILBOX_SLOT_SIZE / 4);

  // Create private WASM memory
  const initialPages = 1024; // 64MB — will grow on demand
  memory = new WebAssembly.Memory({ initial: initialPages });

  // VFS (each process gets its own instance for independent cwd / open fd positions)
  vfs = new InMemoryVFS();
  vfs.loadBinary(elfBytes); // populate /bin/busybox + applet symlinks

  let pc: bigint;

  if (config.memoryBuffer) {
    // ── Fork path: restore parent's memory snapshot ──
    const parentBytes = new Uint8Array(config.memoryBuffer);
    const myBytes = new Uint8Array(memory.buffer);

    // Grow memory if parent's snapshot is larger
    if (parentBytes.length > myBytes.length) {
      const pagesNeeded = Math.ceil(parentBytes.length / 65536);
      const currentPages = memory.buffer.byteLength / 65536;
      if (pagesNeeded > currentPages) {
        memory.grow(pagesNeeded - currentPages);
      }
    }

    new Uint8Array(memory.buffer).set(parentBytes);
    cpu = new CpuState(memory);

    // Child returns 0 from fork
    cpu.setReg(0, 0n);

    // Restore syscall state
    syscalls = new SyscallHandler(0n, {
      writeStdout: (d) => postWrite(1, d),
      writeStderr: (d) => postWrite(2, d),
      readStdin: (count) => readStdin(count),
    }, vfs);
    syscalls.bridgeAvailable = true; // Worker can relay bridge calls to kernel
    if (config.syscallState) {
      syscalls.restoreState(config.syscallState as any);
    }

    // Restore kernel pipes map from parent and increment refcounts
    if (config.kernelPipesSnapshot) {
      for (const [fd, info] of config.kernelPipesSnapshot) {
        kernelPipes.set(fd, { ...info });
        // Increment refcount on the shared ring — child is a new holder of this pipe end
        const ring = new PipeRing(kernelSAB, pipeRingOffset(info.pipeIndex), PIPE_RING_SIZE);
        if (info.mode === 'write') ring.addWriter();
        else ring.addReader();
      }
      for (const fd of kernelPipes.keys()) {
        if (fd >= nextLocalFd) nextLocalFd = fd + 1;
      }
    }

    // Resume at the instruction after the fork syscall
    pc = cpu.getPC() + 4n;
  } else {
    // ── Exec path: fresh binary load ──
    cpu = new CpuState(memory);

    const elfInfo = loadElf(elfBytes, memory);
    const sp = setupStack(memory, elfInfo, config.argv, config.envp);
    cpu.setSP(sp);

    // Set TLS pointer
    const view = new DataView(memory.buffer);
    view.setBigUint64(0x118, BigInt(0x900000), true); // TPIDR_OFFSET

    syscalls = new SyscallHandler(elfInfo.brkStart, {
      writeStdout: (d) => postWrite(1, d),
      writeStderr: (d) => postWrite(2, d),
      readStdin: (count) => readStdin(count),
    }, vfs);
    syscalls.bridgeAvailable = true;

    pc = elfInfo.entry;
  }

  // Set up stdin ring if allocated
  const stdinRingIdx = Atomics.load(procTable, (procSlotOffset(mySlot) / 4) + 7); // PROC_STDIN_RING
  if (stdinRingIdx >= 0) {
    stdinRing = new PipeRing(kernelSAB, stdinRingOffset(stdinRingIdx), STDIN_RING_SIZE);
  }

  // ── Run! ──
  console.log(`[worker ${myPid}] Starting at pc=0x${pc.toString(16)}`);
  try {
    runLoop(pc);
  } catch (e: any) {
    if (e.message === 'SIGINT') {
      // Already called postExit(130) — just stop
      return;
    }
    console.error(`[worker ${myPid}] CRASH:`, e.message, e.stack);
    postExit(139); // SIGSEGV-like exit
  }
}

// ── JIT region cache ────────────────────────────────────────────────────────

function getRegion(pc: bigint): CompiledRegion {
  let region = regionCache.get(pc);
  if (region) return region;

  const view = new DataView(memory.buffer);
  // Use single-block mode for now (matching the test harness behavior)
  // Region mode can be enabled later once correctness is confirmed
  const block = detectBlock(view, pc);
  try {
    region = translateBlock(block, memory);
  } catch (e: any) {
    console.error(`[worker ${myPid}] CRASH at pc=0x${pc.toString(16)}:`, e.message);
    // Log the instructions in the block
    for (let i = 0; i < Math.min(block.count, 10); i++) {
      const addr = Number(block.startPC) + i * 4;
      const insn = view.getUint32(addr + 0x10000, true); // GUEST_MEM_BASE
      console.error(`  0x${addr.toString(16)}: 0x${insn.toString(16).padStart(8, '0')}`);
    }
    throw e;
  }

  regionCache.set(pc, region);
  return region;
}

// ── Main execution loop ─────────────────────────────────────────────────────

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
    const insns = Number(view.getBigUint64(ITER_OFFSET, true));
    totalInsns += insns;
    view.setBigUint64(ITER_OFFSET, 0n, true);

    // Check for pending signals periodically (~every 1000 blocks)
    lastSignalCheck += insns;
    if (lastSignalCheck > 10000) {
      lastSignalCheck = 0;
      const sigPending = Atomics.load(procTable, (procSlotOffset(mySlot) / 4) + 6); // PROC_SIGNAL
      if (sigPending & SIG_INT) {
        Atomics.and(procTable, (procSlotOffset(mySlot) / 4) + 6, ~SIG_INT);
        // Default SIGINT action: exit with 128 + 2 = 130
        postExit(130);
        return;
      }
    }

    // Handle syscall
    if (pc === SVC_PC) {
      const svcPC = view.getBigUint64(PC_OFFSET, true);
      const nr = Number(cpu.getReg(8));

      // Intercept syscalls that need direct kernel SAB access
      if (nr === 260) {
        handleWait4(cpu, view);
        pc = svcPC + 4n;
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
        pc = svcPC + 4n;
        continue;
      }

      const result = syscalls.handle(cpu, memory);
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
          pc = handleExec(result.argv, result.envp);
          break;

        case 'blocked_read':
          // Block on stdin via Atomics.wait on the stdin ring
          handleBlockedRead();
          // Retry the read syscall
          pc = svcPC;
          break;

        case 'bridge':
          // Bridge calls go through main thread
          handleBridgeCall(result.method, result.args, result.apply);
          pc = svcPC + 4n;
          break;

        default:
          pc = svcPC + 4n;
          break;
      }
      continue;
    }
  }

  console.log(`[worker] Exited run loop: pc=0x${pc.toString(16)} blocks=${blockCount} insns=${totalInsns}`);
  // Reached HALT_PC — unknown instruction or end of execution
  postExit(0);
}

// ── Syscall helpers ─────────────────────────────────────────────────────────

function handleFork(resumePC: bigint): bigint {
  // Snapshot our memory for the child
  const snapSize = Math.min(
    Number(syscalls.getMmapBase()) + GUEST_MEM_BASE + 0x10000,
    memory.buffer.byteLength,
  );
  const memoryBuffer = memory.buffer.slice(0, snapSize);
  const syscallState = syscalls.getState();

  // Store our PC so the child knows where to resume
  cpu.setPC(resumePC);

  // Serialize kernel pipe map for child inheritance
  const kernelPipesSnapshot = [...kernelPipes.entries()];

  // Request fork from kernel
  (self as any).postMessage({
    type: 'fork',
    memoryBuffer,
    syscallState,
    kernelPipesSnapshot,
    resumePC: resumePC.toString(),
  }, [memoryBuffer]); // Transfer the buffer (zero-copy send, we'll re-read from memory)

  // Block until kernel responds with child PID
  Atomics.store(myMailbox, MBOX_READY, 0);
  Atomics.wait(myMailbox, MBOX_READY, 0);
  const childPid = Atomics.load(myMailbox, MBOX_VALUE0);

  // Parent: set X0 = childPid
  cpu.setReg(0, BigInt(childPid));
  return resumePC;
}

function handleExec(argv: string[], envp: string[]): bigint {
  // Clear guest memory
  const memBytes = new Uint8Array(memory.buffer);
  memBytes.fill(0, GUEST_MEM_BASE, GUEST_MEM_BASE + 0x4000000); // Clear up to 64MB

  // Reload ELF
  const elfInfo = loadElf(elfBytes, memory);
  const sp = setupStack(memory, elfInfo, argv, envp);
  cpu.reset();
  cpu.setSP(sp);

  // Set TLS pointer
  const view = new DataView(memory.buffer);
  view.setBigUint64(0x118, BigInt(0x900000), true);

  // Reset syscall state
  syscalls.resetForExec(elfInfo.brkStart);

  // Clear JIT cache (new binary image)
  regionCache.clear();

  return elfInfo.entry;
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
    const mySlotBase = procSlotOffset(mySlot) / 4;
    Atomics.store(procTable, mySlotBase + PROC_WAIT_FUTEX, 0);
    Atomics.wait(procTable, mySlotBase + PROC_WAIT_FUTEX, 0, 1000);
    // Loop back and re-check for zombies

    // Also check signals while waiting
    const sigPending = Atomics.load(procTable, mySlotBase + PROC_SIGNAL);
    if (sigPending & SIG_INT) {
      Atomics.and(procTable, mySlotBase + PROC_SIGNAL, ~SIG_INT);
      // Interrupted by signal — return EINTR
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-4))); // -EINTR
      return;
    }
  }
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

  // Initialize the ring buffer
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
    // read
    const buf = Number(cpu.getReg(1));
    const count = Number(cpu.getReg(2));
    const dst = new Uint8Array(count);
    const n = ring.read(dst); // blocks via Atomics.wait if empty
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
    const src = new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, count);
    const n = ring.write(src);
    if (n === -1) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-32))); // -EPIPE
      return;
    }
    cpu.setReg(0, BigInt(n));
    return;
  }

  if (nr === 66) {
    // writev — gather write from iovec array
    const iovAddr = Number(cpu.getReg(1));
    const iovcnt = Number(cpu.getReg(2));
    let totalWritten = 0;
    const bytes = new Uint8Array(memory.buffer);
    for (let i = 0; i < iovcnt; i++) {
      const base = view.getBigUint64(iovAddr + i * 16 + GUEST_MEM_BASE, true);
      const len = view.getBigUint64(iovAddr + i * 16 + 8 + GUEST_MEM_BASE, true);
      if (len === 0n) continue;
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
      Atomics.wait(ringI32, 0, wp);

      // Check for signals while waiting
      const sigPending = Atomics.load(procTable, (procSlotOffset(mySlot) / 4) + 6);
      if (sigPending & SIG_INT) {
        Atomics.and(procTable, (procSlotOffset(mySlot) / 4) + 6, ~SIG_INT);
        if (syscalls.hasSignalHandler(2)) {
          // Shell has a custom SIGINT handler — deliver via dispatch, don't exit
          syscalls.raiseSignal(2);
          return; // Return to run loop, which will deliver the signal
        }
        postExit(130);
        throw new Error('SIGINT');
      }
    }
  }
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

  // Response format: [ret: i32] [dataLen: u32] [data: bytes...]
  const ret = respView.getInt32(0, true);
  const dataLen = respView.getUint32(4, true);

  const resp: any = { ret };
  if (dataLen > 0) {
    // Read raw bytes and convert to base64 for the apply callback
    const dataBytes = new Uint8Array(kernelSAB, respOffset + 8, dataLen);
    let b64 = '';
    for (let i = 0; i < dataLen; i++) b64 += String.fromCharCode(dataBytes[i]);
    resp.data = btoa(b64);
  }

  apply(resp);
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

function postWrite(fd: number, data: Uint8Array): void {
  (self as any).postMessage({ type: 'write', fd, data: data.slice() });
}

function postExit(code: number): void {
  // Close all open kernel pipe fds before exiting
  for (const [fd, info] of kernelPipes) {
    const ring = new PipeRing(kernelSAB, pipeRingOffset(info.pipeIndex), PIPE_RING_SIZE);
    if (info.mode === 'write') ring.closeWriter();
    else ring.closeReader();
  }
  kernelPipes.clear();

  (self as any).postMessage({ type: 'exit', pid: myPid, code });
}
