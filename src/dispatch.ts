import { HALT_PC, SVC_PC, GUEST_MEM_BASE, ITER_OFFSET, PC_OFFSET } from './types.js';
import type { CompiledRegion } from './types.js';
import { CpuState } from './cpu.js';
import { translateRegionZig } from './zig-translator.js';
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

// thrown by the page-protection guard's mem_fault callback to unwind out of a translated region.
const MEM_FAULT = { memFault: true } as const;
/** true if `e` is the sentinel thrown by a page-protection guard (a guest access to a restricted
 *  page). callers should deliver SIGSEGV to the faulting process. */
export function isMemFault(e: unknown): boolean { return e === MEM_FAULT; }
const MAX_PROCESS_BYTES = 1610612736; // 1.5 GiB — must match syscall.ts (per-process address ceiling)
const SIGSEGV = 11;

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

  /** threaded mode: shared memory + reg_base-relative registers + real atomics (for pthreads). */
  readonly threaded: boolean;
  /** linear-memory base of this context's register block (0 for main/single-threaded). */
  readonly stateBase: number;
  private regBase: WebAssembly.Global | null = null;

  // software page-protection: an on/off flag global the translated guard reads, the per-page
  // permission table's linear-memory offset, and the raised-but-undelivered fault. see setPageProt.
  private protActive: WebAssembly.Global | null = null;
  // permission table's linear-memory offset: placed just above the per-process address ceiling so no
  // guest mapping can ever collide with (and corrupt) it. lazily grown/committed on first use.
  private readonly protBase = MAX_PROCESS_BYTES + GUEST_MEM_BASE; // 1.5 GiB + 64 KiB
  private protReady = false;
  private readonly protTableSize = MAX_PROCESS_BYTES >>> 12; // 1 byte per 4 KiB guest page
  private pendingFault: { addr: number; isWrite: number } | null = null;

  // called by the translated guard (env.mem_fault) when a guest access hits a restricted page.
  // records the fault and throws to unwind out of run() to the dispatch loop's fault handler.
  private memFault = (addr: number, isWrite: number): void => {
    this.pendingFault = { addr, isWrite };
    throw MEM_FAULT;
  };

  /** set the permission (PROT_* bits) of the guest pages covering [addr, addr+len) and turn the
   *  guard on. lazily materializes the permission table (all pages start fully accessible). */
  setPageProt(addr: number, len: number, prot: number): void {
    if (this.threaded) return; // protection is wired for the non-threaded (fork) path only, for now
    const b = prot & 7; // bit0=read bit1=write bit2=exec
    // a fully read+write mapping enforces nothing; if protection was never activated, stay on the
    // fast path (pages default to accessible). once active, we still record it to clear stale bytes.
    if ((b & 3) === 3 && !this.protReady) return;
    if (!this.protReady) {
      const need = this.protBase + this.protTableSize;
      const cur = this.memory.buffer.byteLength;
      if (need > cur) this.memory.grow(Math.ceil((need - cur) / 65536));
      new Uint8Array(this.memory.buffer).fill(7, this.protBase, this.protBase + this.protTableSize);
      this.protReady = true;
    }
    const bytes = new Uint8Array(this.memory.buffer);
    const first = addr >>> 12;
    const last = (addr + len + 4095) >>> 12;
    for (let p = first; p < last; p++) bytes[this.protBase + p] = b;
    if (this.protActive) this.protActive.value = 1;
  }

  constructor(memoryPages = 256, threaded = false, regBaseAddr = 0) {
    this.threaded = threaded;
    this.stateBase = threaded ? regBaseAddr : 0;
    if (threaded) {
      // threaded process: one SHARED memory the thread-Workers import; each thread gets its own
      // register block via reg_base. cap the max modestly to respect the shared pool. a non-zero
      // regBaseAddr places this context's registers there (proves per-thread reg blocks work).
      this.memory = new WebAssembly.Memory({ initial: memoryPages, maximum: 8192, shared: true });
      // immutable to match the module's `addImportGlobal(..., mutable=false)`.
      this.regBase = new WebAssembly.Global({ value: 'i32', mutable: false }, regBaseAddr);
    } else {
      // NON-shared memory: the translator runs single-threaded (shared_monitor=0) so it
      // emits no wasm atomics, which means we don't need a SharedArrayBuffer. non-shared
      // grows LAZILY (only commits touched pages) instead of reserving its max up front,
      // so a 4 GiB ceiling costs nothing until the guest actually uses it. NOTE: grow()
      // detaches memory.buffer, so never cache a view across a potential grow.
      this.memory = new WebAssembly.Memory({ initial: memoryPages, maximum: 65536 });
      // the guard reads this mutable global; it stays 0 (fast path — guard skipped) until the first
      // restricted page is created via setPageProt.
      this.protActive = new WebAssembly.Global({ value: 'i32', mutable: true }, 0);
    }
    this.cpu = new CpuState(this.memory, this.stateBase);
  }

  loadProgram(code: Uint32Array, guestPC: bigint): void {
    const view = new DataView(this.memory.buffer);
    const baseOffset = Number(guestPC) + GUEST_MEM_BASE;
    for (let i = 0; i < code.length; i++) {
      view.setUint32(baseOffset + i * 4, code[i], true);
    }
  }

  /// when true, the translator builds single-block regions (used by the
  /// differential verification tools). production uses multi-block regions.
  zigSingleBlock = false;
  /// linear-memory offset where the exclusive-monitor shadow bitmap lives.
  /// keep above expected guest memory size. 32 MiB is plenty for boot-busybox.
  zigShadowBase = 0x2000000;

  private getRegion(pc: bigint): CompiledRegion {
    let region = this.regionCache.get(pc);
    if (region) return region;

    const view = new DataView(this.memory.buffer);

    const protection = this.protActive
      ? { protBase: this.protBase, protActive: this.protActive, memFault: this.memFault }
      : null;
    const zigRegion = translateRegionZig(view, pc, this.memory, this.zigShadowBase, this.zigSingleBlock, this.threaded, this.regBase, protection);
    if (!zigRegion) {
      const insn = view.getUint32(Number(pc) + GUEST_MEM_BASE, true);
      throw new Error(`translator failed to build a region at pc=0x${pc.toString(16)} (entry insn 0x${insn.toString(16).padStart(8, '0')})`);
    }
    this.regionCache.set(pc, zigRegion);
    return zigRegion;
  }

  run(startPC: bigint, maxInsns?: number): RunResult {
    // Install signal trampoline if not already done
    if (this.syscallHandlerInstance && !this._trampolineInstalled) {
      this.syscallHandlerInstance.installSignalTrampoline(this.memory);
      this.syscallHandlerInstance.isLivePidFn = (pid: number) => this.processManager.isLivePid(pid);
      if (this.protActive) this.syscallHandlerInstance.protHook = (addr, len, prot) => this.setPageProt(addr, len, prot);
      this._trampolineInstalled = true;
    }
    this.cpu.setPC(startPC);
    let pc = startPC;
    let totalInsns = 0;

    const t0 = performance.now();

    let view = new DataView(this.memory.buffer);

    while (pc !== HALT_PC) {
      const region = this.getRegion(pc);
      try {
        pc = region.run();
      } catch (e) {
        if (e !== MEM_FAULT) throw e;
        // guest access to a restricted page: raise SIGSEGV and deliver it. with no handler the
        // default action terminates the process (WIFSIGNALED / WTERMSIG == SIGSEGV for the parent).
        if (this.syscallHandlerInstance) this.syscallHandlerInstance.raiseSignal(SIGSEGV);
        pc = this.deliverPending(this.cpu.getPC());
        view = new DataView(this.memory.buffer); // deliverPending/exit may have restored a parent
        continue;
      }

      // All regions write instruction count to ITER_OFFSET
      const insns = Number(view.getBigUint64(this.stateBase + ITER_OFFSET, true));
      totalInsns += insns;
      view.setBigUint64(this.stateBase + ITER_OFFSET, 0n, true);

      // Handle SVC (syscall)
      if (pc === SVC_PC) {
        // Read the PC where the SVC occurred
        const svcPC = view.getBigUint64(this.stateBase + PC_OFFSET, true);

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
        // Resume at PC + 4, then deliver any pending signals (incl. a just-fired SIGALRM)
        pc = this.deliverPending(svcPC + 4n);
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

      const insns = Number(view.getBigUint64(this.stateBase + ITER_OFFSET, true));
      totalInsns += insns;
      view.setBigUint64(this.stateBase + ITER_OFFSET, 0n, true);

      if (pc === SVC_PC) {
        const svcPC = view.getBigUint64(this.stateBase + PC_OFFSET, true);

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
        // Fire the interval timer even in a syscall-less spin (busybox timeout / alarm).
        if (this.syscallHandlerInstance?.checkTimers()) {
          pc = this.deliverPending(pc);
          continue;
        }
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

  /**
   * Deliver any queued signals, resuming normal execution at nextPC. Sets the CPU PC to
   * nextPC first so a signal handler's saved return address is correct. Returns the PC to
   * continue from: nextPC if nothing changed control flow, the handler entry if a handler
   * was installed, the parent resume PC if the current process was terminated, or HALT_PC.
   */
  private deliverPending(nextPC: bigint): bigint {
    const sh = this.syscallHandlerInstance;
    if (!sh?.pendingSignals.length) return nextPC;
    this.cpu.setPC(nextPC);
    while (sh.pendingSignals.length > 0) {
      const sig = sh.pendingSignals.shift()!;
      const r = sh.deliverSignal(this.cpu, this.memory, sig);
      if (r.action === 'exit') {
        const restored = this.processManager.exit(this.memory, this.cpu, r.exitCode ?? 128 + sig);
        if (restored) {
          sh.lastWaitStatus = r.exitCode ?? 128 + sig;
          sh.lastWaitPid = restored.childPid;
          sh.restoreState(restored.syscallState);
          sh.currentPid = restored.parentPid;
          return restored.resumePC;
        }
        return HALT_PC;
      }
      if (r.delivered && r.action === 'continue') {
        // handler installed (PC redirected) or SIG_IGN (PC unchanged); either way follow it
        return this.cpu.getPC();
      }
    }
    return nextPC;
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

  /** Push data from terminal */
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
