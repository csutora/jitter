import { HALT_PC, SVC_PC, GUEST_MEM_BASE, ITER_OFFSET, PC_OFFSET } from './types.js';
import type { CompiledRegion } from './types.js';
import { CpuState } from './cpu.js';
import { detectRegion, detectBlock } from './block.js';
import { translateRegion, translateBlock } from './translate.js';
import { ProcessManager } from './process.js';
import type { SyscallResult } from './syscall.js';
import { SyscallHandler as SyscallHandlerClass } from './syscall.js';
import type { NativeBridge } from './bridge.js';

export interface RunResult {
  insns: number;
  elapsedMs: number;
  ips: number;
  exitPC: bigint;
}

export type SyscallHandlerFn = (cpu: CpuState, memory: WebAssembly.Memory) => SyscallResult;

export class Emulator {
  readonly memory: WebAssembly.Memory;
  readonly cpu: CpuState;
  private regionCache = new Map<bigint, CompiledRegion>();
  syscallHandler?: SyscallHandlerFn;
  /** Direct reference to SyscallHandler instance for process management */
  syscallHandlerInstance?: SyscallHandlerClass;
  processManager = new ProcessManager();
  /** ELF bytes for execve (busybox binary) */
  elfBytes?: Uint8Array;
  private _trampolineInstalled = false;
  /** Native bridge for Swift filesystem calls */
  bridge?: NativeBridge;

  constructor(memoryPages = 256) { // 256 pages = 16MB
    this.memory = new WebAssembly.Memory({ initial: memoryPages });
    this.cpu = new CpuState(this.memory);
  }

  loadProgram(code: Uint32Array, guestPC: bigint): void {
    const view = new DataView(this.memory.buffer);
    const baseOffset = Number(guestPC) + GUEST_MEM_BASE;
    for (let i = 0; i < code.length; i++) {
      view.setUint32(baseOffset + i * 4, code[i], true);
    }
  }

  /** Set to true to use single-block mode (for debugging) */
  singleBlockMode = false;

  private getRegion(pc: bigint): CompiledRegion {
    let region = this.regionCache.get(pc);
    if (region) return region;

    const view = new DataView(this.memory.buffer);
    if (this.singleBlockMode) {
      const block = detectBlock(view, pc);
      try {
        region = translateBlock(block, this.memory);
      } catch (e: any) {
        console.error(`Failed to compile block at 0x${pc.toString(16)} (${block.insns.length} insns):`);
        for (const insn of block.insns) console.error(`  ${insn.kind}`);
        throw e;
      }
    } else {
      const detected = detectRegion(view, pc);
      region = translateRegion(detected, this.memory);
    }

    // Cache by entry PC and all block PCs in the region
    this.regionCache.set(pc, region);
    return region;
  }

  run(startPC: bigint, maxInsns?: number): RunResult {
    // Install signal trampoline if not already done
    if (this.syscallHandlerInstance && !this._trampolineInstalled) {
      this.syscallHandlerInstance.installSignalTrampoline(this.memory);
      this._trampolineInstalled = true;
    }
    this.cpu.setPC(startPC);
    let pc = startPC;
    let totalInsns = 0;

    const t0 = performance.now();

    const view = new DataView(this.memory.buffer);

    while (pc !== HALT_PC) {
      const region = this.getRegion(pc);
      pc = region.run();

      // All regions write instruction count to ITER_OFFSET
      const insns = Number(view.getBigUint64(ITER_OFFSET, true));
      totalInsns += insns;
      view.setBigUint64(ITER_OFFSET, 0n, true);

      // Handle SVC (syscall)
      if (pc === SVC_PC) {
        // Read the PC where the SVC occurred
        const svcPC = view.getBigUint64(PC_OFFSET, true);

        if (this.syscallHandler) {
          const result = this.syscallHandler(this.cpu, this.memory);
          if (result.status === 'exit') {
            // Check if this is a child process exit
            const restored = this.processManager.exit(this.memory, this.cpu, result.code);
            if (restored) {
              // Parent restored — set up wait4 state and restore PID
              if (this.syscallHandlerInstance) {
                this.syscallHandlerInstance.lastWaitStatus = result.code;
                this.syscallHandlerInstance.lastWaitPid = restored.childPid;
                this.syscallHandlerInstance.restoreState(restored.syscallState);
                this.syscallHandlerInstance.currentPid = restored.parentPid;
              }
              pc = restored.resumePC;
              continue;
            }
            pc = HALT_PC;
            break;
          }
          if (result.status === 'blocked_read') {
            pc = HALT_PC;
            break;
          }
          if (result.status === 'fork') {
            if (this.syscallHandlerInstance) {
              const parentPid = this.syscallHandlerInstance.currentPid;
              this.processManager.fork(
                this.memory, this.cpu,
                this.syscallHandlerInstance.getState(),
                result.resumePC,
              );
              // Child gets a new PID
              this.syscallHandlerInstance.currentPid = this.processManager.getLastChildPid();
              this.syscallHandlerInstance.currentPpid = parentPid;
            }
            // Child continues at resumePC with X0=0 (set by fork())
            pc = result.resumePC;
            continue;
          }
          if (result.status === 'execve') {
            if (this.elfBytes && this.syscallHandlerInstance) {
              const { entry, brkStart } = this.processManager.execve(
                this.memory, this.cpu,
                this.elfBytes, result.argv, result.envp,
              );
              this.syscallHandlerInstance.resetForExec(brkStart);
              pc = entry;
              continue;
            }
            // No ELF bytes — treat as error
            this.cpu.setReg(0, BigInt.asUintN(64, -2n)); // -ENOENT
            pc = svcPC + 4n;
            continue;
          }
        }
        // Resume at PC + 4
        pc = svcPC + 4n;

        // Check for pending signals and deliver them
        if (this.syscallHandlerInstance?.pendingSignals.length) {
          while (this.syscallHandlerInstance.pendingSignals.length > 0) {
            const sig = this.syscallHandlerInstance.pendingSignals.shift()!;
            const result2 = this.syscallHandlerInstance.deliverSignal(this.cpu, this.memory, sig);
            if (result2.action === 'exit') {
              const restored = this.processManager.exit(this.memory, this.cpu, result2.exitCode ?? 128 + sig);
              if (restored) {
                if (this.syscallHandlerInstance) {
                  this.syscallHandlerInstance.lastWaitStatus = result2.exitCode ?? 128 + sig;
                  this.syscallHandlerInstance.lastWaitPid = restored.childPid;
                  this.syscallHandlerInstance.restoreState(restored.syscallState);
                  this.syscallHandlerInstance.currentPid = restored.parentPid;
                }
                pc = restored.resumePC;
                break;
              }
              pc = HALT_PC;
              break;
            }
            if (result2.delivered && result2.action === 'continue') {
              // Signal handler was set up — PC has been redirected to the handler
              pc = this.cpu.getPC();
              break;
            }
          }
        }
        continue;
      }

      if (maxInsns !== undefined && totalInsns >= maxInsns) break;
    }

    const elapsed = performance.now() - t0;

    return {
      insns: totalInsns,
      elapsedMs: elapsed,
      ips: elapsed > 0 ? (totalInsns / elapsed) * 1000 : 0,
      exitPC: pc,
    };
  }

  /** Optional IOManager reference for interrupt checking */
  io?: IOManager;

  /** Async run loop for shell execution — yields on blocked reads and periodically for UI */
  async runAsync(
    startPC: bigint,
    onBlocked: () => Promise<void>,
    onExit?: (code: number) => void,
  ): Promise<RunResult> {
    this.cpu.setPC(startPC);
    let pc = startPC;
    let totalInsns = 0;

    const t0 = performance.now();
    let lastYield = t0;

    while (pc !== HALT_PC) {
      const view = new DataView(this.memory.buffer);
      const region = this.getRegion(pc);
      pc = region.run();

      const insns = Number(view.getBigUint64(ITER_OFFSET, true));
      totalInsns += insns;
      view.setBigUint64(ITER_OFFSET, 0n, true);

      if (pc === SVC_PC) {
        const svcPC = view.getBigUint64(PC_OFFSET, true);

        if (this.syscallHandler) {
          let result = this.syscallHandler(this.cpu, this.memory);
          if (result.status === 'exit') {
            const restored = this.processManager.exit(this.memory, this.cpu, result.code);
            if (restored) {
              if (this.syscallHandlerInstance) {
                this.syscallHandlerInstance.lastWaitStatus = result.code;
                this.syscallHandlerInstance.lastWaitPid = restored.childPid;
                this.syscallHandlerInstance.restoreState(restored.syscallState);
              }
              pc = restored.resumePC;
              continue;
            }
            onExit?.(result.code);
            pc = HALT_PC;
            break;
          }
          while (result.status === 'blocked_read') {
            await onBlocked();
            result = this.syscallHandler(this.cpu, this.memory);
            if (result.status === 'exit') {
              const restored = this.processManager.exit(this.memory, this.cpu, result.code);
              if (restored) {
                if (this.syscallHandlerInstance) {
                  this.syscallHandlerInstance.lastWaitStatus = result.code;
                  this.syscallHandlerInstance.lastWaitPid = restored.childPid;
                  this.syscallHandlerInstance.restoreState(restored.syscallState);
                }
                pc = restored.resumePC;
                break;
              }
              onExit?.(result.code);
              pc = HALT_PC;
              break;
            }
          }
          if (pc === HALT_PC) break;
          if (result.status === 'fork') {
            if (this.syscallHandlerInstance) {
              this.processManager.fork(
                this.memory, this.cpu,
                this.syscallHandlerInstance.getState(),
                result.resumePC,
              );
            }
            pc = result.resumePC;
            continue;
          }
          if (result.status === 'execve') {
            if (this.elfBytes && this.syscallHandlerInstance) {
              const { entry, brkStart } = this.processManager.execve(
                this.memory, this.cpu,
                this.elfBytes, result.argv, result.envp,
              );
              this.syscallHandlerInstance.resetForExec(brkStart);
              pc = entry;
              continue;
            }
            this.cpu.setReg(0, BigInt.asUintN(64, -2n));
            pc = svcPC + 4n;
            continue;
          }
          if (result.status === 'bridge' && this.bridge) {
            const resp = await this.bridge.call(result.method, result.args);
            result.apply(resp);
            pc = svcPC + 4n;
            continue;
          }
        }
        pc = svcPC + 4n;
        continue;
      }

      // Yield to browser every ~5ms for UI responsiveness
      const now = performance.now();
      if (now - lastYield > 5) {
        await new Promise<void>(r => setTimeout(r, 0));
        lastYield = performance.now();
        // Check for ctrl+c interrupt
        if (this.io?.interrupted) {
          this.io.interrupted = false;
          if (this.processManager.depth > 0) {
            // Force-exit child process with code 130 (128 + SIGINT=2)
            const restored = this.processManager.exit(this.memory, this.cpu, 130);
            if (restored && this.syscallHandlerInstance) {
              this.syscallHandlerInstance.lastWaitStatus = 130;
              this.syscallHandlerInstance.lastWaitPid = restored.childPid;
              this.syscallHandlerInstance.restoreState(restored.syscallState);
              pc = restored.resumePC;
              continue;
            }
          }
        }
      }
    }

    const elapsed = performance.now() - t0;
    return {
      insns: totalInsns,
      elapsedMs: elapsed,
      ips: elapsed > 0 ? (totalInsns / elapsed) * 1000 : 0,
      exitPC: pc,
    };
  }

  reset(): void {
    this.cpu.reset();
    this.regionCache.clear();
  }
}

// ── I/O Manager for async stdin ─────────────────────────────────────────────

export class IOManager {
  private stdinBuffer: Uint8Array[] = [];
  private stdinWaiters: (() => void)[] = [];
  interrupted = false;

  /** Push data from terminal (xterm.js onData) */
  pushStdin(data: Uint8Array): void {
    this.stdinBuffer.push(data);
    // Wake up any waiting readers
    const waiters = this.stdinWaiters.splice(0);
    for (const w of waiters) w();
  }

  /** Attempt to read up to count bytes from stdin. Returns null if no data available. */
  readStdin(count: number): Uint8Array | null {
    if (this.stdinBuffer.length === 0) return null;

    // Gather bytes from buffer
    const chunks: Uint8Array[] = [];
    let remaining = count;

    while (remaining > 0 && this.stdinBuffer.length > 0) {
      const chunk = this.stdinBuffer[0];
      if (chunk.length <= remaining) {
        chunks.push(chunk);
        remaining -= chunk.length;
        this.stdinBuffer.shift();
      } else {
        chunks.push(chunk.subarray(0, remaining));
        this.stdinBuffer[0] = chunk.subarray(remaining);
        remaining = 0;
      }
    }

    const total = chunks.reduce((sum, c) => sum + c.length, 0);
    const result = new Uint8Array(total);
    let offset = 0;
    for (const c of chunks) {
      result.set(c, offset);
      offset += c.length;
    }
    return result;
  }

  /** Wait until stdin has data available */
  waitForStdin(): Promise<void> {
    if (this.stdinBuffer.length > 0) return Promise.resolve();
    return new Promise<void>(resolve => {
      this.stdinWaiters.push(resolve);
    });
  }
}
