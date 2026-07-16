/**
 * Conformance tests: run static aarch64-linux-musl binaries that exercise
 * individual syscalls and verify correct behavior via TAP output.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';
import { Emulator, IOManager, isMemFault } from '../src/dispatch.js';
import { loadElf, setupStack } from '../src/elf.js';
import { SyscallHandler } from '../src/syscall.js';
import { Scheduler } from '../src/scheduler.js';
import { InMemoryVFS } from '../src/vfs.js';
import { TPIDR_OFFSET, GUEST_MEM_BASE } from '../src/types.js';
import { decode } from '../src/decode.js';

interface ElfResult {
  stdout: string;
  stderr: string;
  totalInsns: number;
  blockCount: number;
  exitCode: number | undefined;
  haltInfo: string | undefined;
}

/**
 * Run an arbitrary static aarch64 ELF binary in the emulator.
 * Returns stdout, stderr, exit code, and instruction stats.
 */
function runElf(elfPath: string, argv: string[], stdinText = ''): ElfResult | null {
  let elfBytes: Uint8Array;
  try {
    elfBytes = new Uint8Array(readFileSync(elfPath));
  } catch {
    return null;
  }

  const emu = new Emulator(1024);
  // zig translator is the only path now
  emu.elfBytes = elfBytes;

  // Minimal VFS — constructor already populates /etc/passwd, /dev/*, /tmp, /proc/self/exe
  // Don't call loadBinary — conformance tests don't need busybox symlinks.
  const vfs = new InMemoryVFS();

  const elfInfo = loadElf(elfBytes, emu.memory);

  const sp = setupStack(emu.memory, elfInfo, argv,
    ['HOME=/root', 'TERM=linux', 'PATH=/bin:/usr/bin', 'PWD=/',
     // the emulator provides a single in-memory tree, so ltp's scratch device is a
     // tmpfs mount (no loop device, no mkfs). LTP_DEV points at a block-device node so
     // `.needs_device` acquisition succeeds; the fs type declares tmpfs truthfully.
     'LTP_DEV=/dev/ltp-scratch', 'LTP_DEV_FS_TYPE=tmpfs']);
  emu.cpu.setSP(sp);

  const tlsView = new DataView(emu.memory.buffer);
  tlsView.setBigUint64(TPIDR_OFFSET, BigInt(0x900000), true);

  const io = new IOManager();
  io.pushStdin(new TextEncoder().encode(stdinText));

  let stdout = '';
  let stderr = '';
  let exitCode: number | undefined;
  const syscalls = new SyscallHandler(elfInfo.brkStart, {
    writeStdout: (data: Uint8Array) => { stdout += new TextDecoder().decode(data); },
    writeStderr: (data: Uint8Array) => { stderr += new TextDecoder().decode(data); },
    readStdin: (count: number) => io.readStdin(count),
  }, vfs);

  // an ltp test binary is a normal userspace process, not container init. give it a
  // realistic non-1 pid: many tests assert pid > 1, and the ltp framework treats a
  // forked child's getppid()==1 as "the main test process died" (reparented to init).
  syscalls.setIdentity(100, 1);

  emu.syscallHandler = (cpu, memory) => syscalls.handle(cpu, memory);
  emu.syscallHandlerInstance = syscalls;
  syscalls.protHook = (a, l, p) => emu.setPageProt(a, l, p);

  // real concurrent processes: an ltp test forks children that synchronize with the
  // parent over pipes / wait / locks, so parent and child must be alive at once. the
  // cooperative scheduler swaps one process image into the live memory at a time and
  // context-switches at blocking points.
  syscalls.coop = true;
  const HALT = 0xDEAD0000n;
  const SVC = 0xDEAD0001n;
  const rootPid = 100;
  const sched = new Scheduler(emu.memory, syscalls, rootPid, 1, elfInfo.entry);
  syscalls.locks = sched.locks; // shared advisory-lock table across processes
  syscalls.killHook = (pid, sig) => sched.signal(pid, sig); // cross-process signal delivery
  syscalls.futexWakeHook = (addr, n) => sched.futexWake(addr, n); // wake FUTEX_WAIT sleepers

  let pc = elfInfo.entry;
  let totalInsns = 0;
  let view = new DataView(emu.memory.buffer);
  let blockCount = 0;
  let prevPC = pc;
  let haltInfo: string | undefined;

  const encodeWait = (code: number) => (code & 0x10000) ? (code & 0x7f) : ((code & 0xFF) << 8);

  // deliver signals queued on the current process. returns an encoded wait status if a
  // default-terminate signal killed it (0x10000 marker => WIFSIGNALED), else null. a
  // caught signal leaves cpu.pc at the handler entry to resume from.
  const deliverSignals = (): number | null => {
    while (syscalls.pendingSignals.length > 0) {
      const sig = syscalls.pendingSignals.shift()!;
      const r = syscalls.deliverSignal(emu.cpu, emu.memory, sig);
      view = new DataView(emu.memory.buffer);
      if (r.action === 'exit') return 0x10000 | sig;
    }
    return null;
  };

  // switch to another runnable process after the current one blocked or exited.
  // returns false if no runnable process remains (whole run is finished / deadlocked).
  const switchAway = (curPC: bigint, deadlockMsg: string): boolean => {
    const nx = sched.pickNext();
    if (!nx) { if (sched.anyLive()) haltInfo = deadlockMsg; return false; }
    pc = sched.switchTo(nx, curPC);
    view = new DataView(emu.memory.buffer);
    return true;
  };

  // record a process exit. returns true if the whole run is done (root exited or no
  // process left to run).
  const onExit = (encoded: number): boolean => {
    if (sched.current.pid === rootPid) { exitCode = encoded; return true; }
    sched.zombify(encoded);
    return !switchAway(sched.current.pc, 'deadlock after exit');
  };

  while (true) {
    // deliver any signals queued on the current process before it runs (e.g. a kill
    // from a sibling, delivered while this process was blocked). a default-terminate
    // signal kills it here; a caught one redirects pc to the handler.
    if (syscalls.pendingSignals.length > 0) {
      emu.cpu.setPC(pc);
      const killed = deliverSignals();
      if (killed !== null) { if (onExit(killed)) break; else continue; }
      pc = emu.cpu.getPC();
    }

    blockCount++;

    const region = (emu as any).getRegion(pc);
    prevPC = pc;
    try {
      pc = region.run();
    } catch (e: any) {
      if (isMemFault(e)) {
        // guest access to a restricted page -> SIGSEGV, delivered to a handler if one is
        // installed (mmap05 catches it and reports TPASS) else the default action kills the
        // process (a forked child dying this way lets the parent's wait() see the signal).
        emu.cpu.setPC(prevPC);
        syscalls.raiseSignal(11); // SIGSEGV
        const killed = deliverSignals();
        view = new DataView(emu.memory.buffer);
        if (killed !== null) { if (onExit(killed)) break; else continue; }
        pc = emu.cpu.getPC(); // resume at the installed handler
        continue;
      }
      console.error(`CRASH at pc=0x${prevPC.toString(16)}, block=${blockCount}, insns=${totalInsns}`);
      throw e;
    }

    view = new DataView(emu.memory.buffer);
    const insns = Number(view.getBigUint64(272, true));
    totalInsns += insns;
    view.setBigUint64(272, 0n, true);

    if (pc === SVC) {
      const svcPC = view.getBigUint64(256, true);
      const result = syscalls.handle(emu.cpu, emu.memory);
      view = new DataView(emu.memory.buffer);

      // pipe writes / closes during this syscall may unblock other processes
      if (syscalls.schedEvent) { sched.wakeAll(); syscalls.schedEvent = false; }

      if (result.status === 'exit') {
        if (onExit(result.code)) break;
        continue;
      }
      if (result.status === 'fork') {
        const childPid = sched.fork(result.resumePC);
        emu.cpu.setReg(0, BigInt(childPid)); // parent's fork() return value
        pc = result.resumePC;
        continue;
      }
      if (result.status === 'coop_block') {
        sched.blockCurrent(result.reason, svcPC, result.addr ?? 0);
        if (!switchAway(svcPC, `deadlock: all processes blocked (${result.reason})`)) break;
        continue;
      }
      if (result.status === 'coop_wait') {
        const zc = sched.findZombieChild(sched.current.pid, result.wantPid);
        if (zc) {
          if (result.kind === 'wait4') {
            if (result.wstatusAddr !== 0) view.setUint32(result.wstatusAddr + GUEST_MEM_BASE, encodeWait(zc.exitCode), true);
            emu.cpu.setReg(0, BigInt(zc.pid));
          } else { // waitid
            if (result.infop !== 0) {
              const base = result.infop + GUEST_MEM_BASE;
              for (let i = 0; i < 128; i += 8) view.setBigUint64(base + i, 0n, true);
              view.setInt32(base, 17, true); // si_signo = SIGCHLD
              view.setInt32(base + 8, (zc.exitCode & 0x10000) ? 2 : 1, true); // CLD_KILLED : CLD_EXITED
              view.setInt32(base + 12, zc.pid, true); // si_pid
              view.setInt32(base + 24, (zc.exitCode & 0x10000) ? (zc.exitCode & 0x7f) : (zc.exitCode & 0xFF), true); // si_status
            }
            emu.cpu.setReg(0, 0n);
          }
          sched.reap(zc);
          pc = svcPC + 4n;
          continue;
        }
        const WNOHANG = 1;
        if (!sched.hasChild(sched.current.pid, result.wantPid)) {
          emu.cpu.setReg(0, BigInt.asUintN(64, -10n)); // ECHILD
          pc = svcPC + 4n;
          continue;
        }
        if (result.options & WNOHANG) {
          emu.cpu.setReg(0, 0n);
          pc = svcPC + 4n;
          continue;
        }
        sched.blockCurrent('wait', svcPC);
        if (!switchAway(svcPC, 'deadlock: wait with no runnable process')) break;
        continue;
      }
      if (result.status === 'blocked_read') {
        // no stdin data will arrive in the harness; park until another process progresses
        sched.blockCurrent('pipe_read', svcPC);
        if (!switchAway(svcPC, 'deadlock: blocked read')) break;
        continue;
      }
      if (result.status === 'execve') {
        const { entry, brkStart } = emu.processManager.execve(emu.memory, emu.cpu, elfBytes, result.argv, result.envp);
        (emu as any).regionCache.clear(); // new image at the same guest addresses -> drop stale regions
        syscalls.resetForExec(brkStart);
        pc = entry;
        continue;
      }

      // 'ok' (and any bridge result, which conformance never produces): advance past the
      // syscall. queued signals are delivered at the top of the next loop iteration.
      pc = svcPC + 4n;
      // a voluntarily-yielding syscall (sleep/sched_yield) hands the cpu to another
      // runnable process so a sibling can make progress (e.g. reach its FUTEX_WAIT).
      if (syscalls.wantYield) {
        syscalls.wantYield = false;
        const nx = sched.pickNext();
        if (nx && nx !== sched.current) { pc = sched.switchTo(nx, pc); view = new DataView(emu.memory.buffer); }
      }
      continue;
    }

    if (pc === HALT) {
      view = new DataView(emu.memory.buffer);
      const raw = view.getUint32(Number(prevPC) + GUEST_MEM_BASE, true);
      const d = decode(raw);
      haltInfo = `HALT at block ${blockCount}: pc=0x${prevPC.toString(16)}, raw=0x${raw.toString(16).padStart(8, '0')}, kind=${d.kind}`;
      break;
    }

    if (totalInsns >= 500_000_000 || blockCount >= 1_000_000) {
      console.warn(`Limit reached: ${totalInsns} insns, ${blockCount} blocks`);
      break;
    }
  }

  return { stdout, stderr, totalInsns, blockCount, exitCode, haltInfo };
}

// ── Auto-discover and run conformance test binaries ──────────────────────────

const binDir = join(__dirname, 'conformance', 'bin');
const ltpDir = join(binDir, 'ltp');

let testBinaries: string[] = [];
try {
  testBinaries = readdirSync(binDir)
    .filter(f => f.startsWith('test_'))
    .sort();
} catch {
  // bin dir doesn't exist — tests will skip
}

let ltpBinaries: string[] = [];
try {
  ltpBinaries = readdirSync(ltpDir).sort();
} catch {
  // ltp dir doesn't exist — skip
}

describe('conformance', () => {
  if (testBinaries.length === 0) {
    it('skipped — no test binaries found', () => {
      console.log('Build test binaries: cd tests/conformance && make');
    });
    return;
  }

  for (const bin of testBinaries) {
    it(bin, () => {
      const result = runElf(join(binDir, bin), [bin]);
      if (!result) {
        console.log(`Binary not found: ${bin}`);
        return;
      }

      // Print TAP output (from stderr — that's where our harness writes)
      const output = result.stderr || result.stdout;
      if (output) console.log(output.trimEnd());

      if (result.haltInfo) {
        console.error(result.haltInfo);
      }

      console.log(`${result.totalInsns.toLocaleString()} insns, ${result.blockCount.toLocaleString()} blocks`);

      // For test_exit, we expect exit code 42
      if (bin === 'test_exit') {
        expect(result.exitCode).toBe(42);
        return;
      }

      // For all other tests: exit code 0 means all TAP assertions passed
      expect(result.exitCode, `${bin} failed with exit code ${result.exitCode}\n${output}`).toBe(0);

      // Also check that no "not ok" lines appear in output
      const notOkLines = output.split('\n').filter(l => l.startsWith('not ok'));
      expect(notOkLines, `Failed assertions:\n${notOkLines.join('\n')}`).toHaveLength(0);
    });
  }
});

// ── LTP tests ───────────────────────────────────────────────────────────────

describe('ltp', () => {
  if (ltpBinaries.length === 0) {
    it('skipped — no LTP binaries found', () => {
      console.log('Cross-compile LTP and place binaries in tests/conformance/bin/ltp/');
    });
    return;
  }

  for (const bin of ltpBinaries) {
    it(bin, () => {
      const result = runElf(join(ltpDir, bin), [bin]);
      if (!result) {
        console.log(`Binary not found: ${bin}`);
        return;
      }

      const output = (result.stderr + '\n' + result.stdout).trim();

      // LTP output format: lines containing TPASS, TFAIL, TBROK, TCONF
      const passed = (output.match(/TPASS/g) || []).length;
      const failed = (output.match(/TFAIL/g) || []).length;
      const broken = (output.match(/TBROK/g) || []).length;
      const skipped = (output.match(/TCONF/g) || []).length;

      console.log(`${bin}: ${passed} pass, ${failed} fail, ${broken} broken, ${skipped} skip | ${result.totalInsns.toLocaleString()} insns`);
      if (failed > 0 || broken > 0) {
        // Print the failing lines
        const failLines = output.split('\n').filter(l => /TFAIL|TBROK/.test(l));
        console.log(failLines.join('\n'));
      }

      if (result.haltInfo) {
        console.error(result.haltInfo);
      }

      // the engine must not have halted on an undecoded/faulting instruction
      expect(result.haltInfo, `${bin} halted the engine`).toBeUndefined();

      // LTP exit codes are a bitmask: TPASS=0, TFAIL=1, TBROK=2, TWARN=4, TINFO=16,
      // TCONF=32. bits 0 and 1 (TFAIL/TBROK) are real failures; everything else is a
      // clean result. a TCONF-only exit (32) means the test isn't applicable to this
      // configuration (e.g. the 16-bit-uid syscalls that don't exist on the aarch64
      // asm-generic ABI) and is a legitimate skip, not a failure.
      const failBits = result.exitCode & 3;
      expect(failBits, `${bin} exited with code ${result.exitCode}`).toBe(0);
      expect(failed, `${bin} had ${failed} TFAIL results`).toBe(0);
      expect(broken, `${bin} had ${broken} TBROK results`).toBe(0);
    });
  }
});
