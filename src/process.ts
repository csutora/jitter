/**
 * Process manager: vfork+exec semantics for single-threaded WASM.
 *
 * When fork() is called:
 *   1. Snapshot parent state (memory, registers, syscall handler state)
 *   2. Push onto process stack
 *   3. Continue as child (X0=0)
 *
 * When execve() is called:
 *   1. Clear guest memory
 *   2. Reload ELF, setup new stack with new argv
 *   3. Continue at entry point
 *
 * When exit() is called and process stack is non-empty:
 *   1. Pop parent state
 *   2. Restore memory and registers
 *   3. Set X0 = child PID (parent's fork return value)
 *   4. Store exit code for wait4
 */

import { GUEST_MEM_BASE, STATE_SIZE } from './types.js';
import { CpuState } from './cpu.js';
import { loadElf, setupStack } from './elf.js';

// ── Types ────────────────────────────────────────────────────────────────────

export interface SyscallState {
  brk: bigint;
  mmapBase: bigint;
  cwd: string;
  fdFlags: Map<number, number>;
  fdTable?: Map<number, any>;  // FdEntry map — pipe/dup redirections
  nextFd?: number;
}

interface ProcessState {
  pid: number;
  childPid: number;           // PID assigned to child (parent's fork return value)
  memorySnapshot: ArrayBuffer; // snapshot of used memory region
  cpuState: ArrayBuffer;       // snapshot of STATE_SIZE bytes (registers + SIMD)
  syscallState: SyscallState;  // brk, mmapBase, cwd, fdFlags
  resumePC: bigint;            // PC where parent should resume (SVC + 4)
  childExitCode?: number;
}

// ── ProcessManager ───────────────────────────────────────────────────────────

export class ProcessManager {
  private processStack: ProcessState[] = [];
  private nextPid = 2; // PID 1 is the init shell

  /** How many bytes of guest memory to snapshot (up to mmapBase high-water mark) */
  private snapshotSize(mmapBase: bigint): number {
    // Snapshot from GUEST_MEM_BASE to mmapBase + some padding
    return Number(mmapBase) + GUEST_MEM_BASE + 0x10000; // include a bit extra
  }

  get depth(): number {
    return this.processStack.length;
  }

  getLastChildPid(): number {
    return this.nextPid - 1;
  }

  /**
   * Fork: save parent state, continue as child.
   * Returns the child PID (which will be set as parent's X0 on restore).
   */
  fork(
    memory: WebAssembly.Memory,
    cpu: CpuState,
    syscallState: SyscallState,
    resumePC: bigint,
  ): number {
    const childPid = this.nextPid++;
    const snapSize = this.snapshotSize(syscallState.mmapBase);

    // Clamp to actual memory size
    const actualSize = Math.min(snapSize, memory.buffer.byteLength);

    // Snapshot memory
    const memSnap = memory.buffer.slice(0, actualSize);

    // Snapshot CPU state (first STATE_SIZE bytes)
    const cpuSnap = memory.buffer.slice(0, STATE_SIZE);

    // Save syscall state (getState() already clones Maps, pipe refs stay shared)
    const savedSyscall = syscallState;

    this.processStack.push({
      pid: 1, // parent PID (simplified)
      childPid,
      memorySnapshot: memSnap,
      cpuState: cpuSnap,
      syscallState: savedSyscall,
      resumePC,
    });

    // Child continues with X0 = 0
    cpu.setReg(0, 0n);

    return childPid;
  }

  /**
   * Execve: reload ELF binary with new argv/envp.
   * Returns the new entry PC and brk start.
   */
  execve(
    memory: WebAssembly.Memory,
    cpu: CpuState,
    elfBytes: Uint8Array,
    argv: string[],
    envp: string[],
  ): { entry: bigint; brkStart: bigint } {
    // Zero guest memory region (keep CPU state area intact temporarily)
    const memBytes = new Uint8Array(memory.buffer);
    // Only zero the guest memory area, not the CPU state
    const guestEnd = Math.min(memory.buffer.byteLength, GUEST_MEM_BASE + 0x4000000); // up to 64MB
    memBytes.fill(0, GUEST_MEM_BASE, guestEnd);

    // Reload ELF
    const elfInfo = loadElf(elfBytes, memory);

    // Setup new stack
    const sp = setupStack(memory, elfInfo, argv, envp);

    // Reset CPU
    cpu.reset();
    cpu.setSP(sp);

    // Setup TLS
    const view = new DataView(memory.buffer);
    view.setBigUint64(0x118, BigInt(0x900000), true); // TPIDR_OFFSET

    return { entry: elfInfo.entry, brkStart: elfInfo.brkStart };
  }

  /**
   * Exit from child process: restore parent state.
   * Returns the PC to resume at, or null if this is the top-level process.
   */
  exit(
    memory: WebAssembly.Memory,
    cpu: CpuState,
    code: number,
  ): { resumePC: bigint; childPid: number; parentPid: number; syscallState: SyscallState } | null {
    if (this.processStack.length === 0) return null;

    const parent = this.processStack.pop()!;
    parent.childExitCode = code;

    // Restore memory
    const snapBytes = new Uint8Array(parent.memorySnapshot);
    const memBytes = new Uint8Array(memory.buffer);
    memBytes.set(snapBytes, 0);

    // Restore CPU state (first STATE_SIZE bytes, which includes registers)
    const cpuBytes = new Uint8Array(parent.cpuState);
    memBytes.set(cpuBytes, 0);

    // Parent's fork() returns child PID in X0
    cpu.setReg(0, BigInt(parent.childPid));

    return {
      resumePC: parent.resumePC,
      childPid: parent.childPid,
      parentPid: parent.pid,
      syscallState: parent.syscallState,
    };
  }

  /**
   * Wait4: get exit status of child.
   * In our model, the child has already exited when wait4 is called
   * (parent was restored with exit code available).
   */
  lastChildExitCode = 0;

  /** Store exit code from child exit for subsequent wait4 */
  setLastExitCode(code: number): void {
    this.lastChildExitCode = code;
  }
}
