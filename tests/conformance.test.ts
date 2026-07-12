/**
 * Conformance tests: run static aarch64-linux-musl binaries that exercise
 * individual syscalls and verify correct behavior via TAP output.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';
import { Emulator, IOManager } from '../src/dispatch.js';
import { loadElf, setupStack } from '../src/elf.js';
import { SyscallHandler } from '../src/syscall.js';
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
    ['HOME=/root', 'TERM=linux', 'PATH=/bin:/usr/bin', 'PWD=/']);
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

  let pc = elfInfo.entry;
  let totalInsns = 0;
  let view = new DataView(emu.memory.buffer);
  const HALT = 0xDEAD0000n;
  const SVC = 0xDEAD0001n;
  let blockCount = 0;
  let prevPC = pc;

  while (pc !== HALT) {
    blockCount++;

    const region = (emu as any).getRegion(pc);
    prevPC = pc;
    try {
      pc = region.run();
    } catch (e: any) {
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

      if (result.status === 'exit') {
        // Check for child process exit — restore parent if on process stack
        const restored = emu.processManager.exit(emu.memory, emu.cpu, result.code);
        if (restored) {
          syscalls.lastWaitStatus = result.code;
          syscalls.lastWaitPid = restored.childPid;
          syscalls.restoreState(restored.syscallState);
          pc = restored.resumePC;
          continue;
        }
        exitCode = result.code;
        break;
      }
      if (result.status === 'blocked_read') break;
      if (result.status === 'fork') {
        const parentPid = syscalls.currentPid;
        const childPid = emu.processManager.fork(
          emu.memory, emu.cpu,
          syscalls.getState(),
          result.resumePC,
        );
        // the emulator continues as the child, so adopt the child identity
        syscalls.setIdentity(childPid, parentPid);
        pc = result.resumePC;
        continue;
      }
      if (result.status === 'execve') {
        const { entry, brkStart } = emu.processManager.execve(
          emu.memory, emu.cpu,
          elfBytes, result.argv, result.envp,
        );
        syscalls.resetForExec(brkStart);
        pc = entry;
        continue;
      }
      pc = svcPC + 4n;
      continue;
    }

    // note: 10M was calibrated when self-loop blocks reported iterations rather
    // than executed insns; counts are accurate now so the budget reflects real work
    if (totalInsns >= 500_000_000 || blockCount >= 1_000_000) {
      console.warn(`Limit reached: ${totalInsns} insns, ${blockCount} blocks`);
      break;
    }
  }

  let haltInfo: string | undefined;
  if (pc === HALT) {
    view = new DataView(emu.memory.buffer);
    const raw = view.getUint32(Number(prevPC) + GUEST_MEM_BASE, true);
    const d = decode(raw);
    haltInfo = `HALT at block ${blockCount}: pc=0x${prevPC.toString(16)}, raw=0x${raw.toString(16).padStart(8, '0')}, kind=${d.kind}`;
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
