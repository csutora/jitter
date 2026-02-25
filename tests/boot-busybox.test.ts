/**
 * Integration test: boot busybox ash shell.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';
import { Emulator, IOManager } from '../src/dispatch.js';
import { loadElf, setupStack } from '../src/elf.js';
import { SyscallHandler } from '../src/syscall.js';
import { InMemoryVFS } from '../src/vfs.js';
import { TPIDR_OFFSET, GUEST_MEM_BASE } from '../src/types.js';
import { decode } from '../src/decode.js';

function loadBusybox() {
  const elfPath = join(__dirname, '..', 'assets', 'busybox-aarch64');
  try {
    return new Uint8Array(readFileSync(elfPath));
  } catch {
    return null;
  }
}

function runBusybox(argv: string[], stdinText = '') {
  const elfBytes = loadBusybox();
  if (!elfBytes) return null;

  const emu = new Emulator(1024);
  emu.singleBlockMode = true;
  emu.elfBytes = elfBytes;

  const vfs = new InMemoryVFS();
  vfs.loadBinary(elfBytes);
  const elfInfo = loadElf(elfBytes, emu.memory);

  const sp = setupStack(emu.memory, elfInfo, argv,
    ['HOME=/root', 'TERM=xterm-256color', 'PATH=/bin', 'PWD=/']);
  emu.cpu.setSP(sp);

  const tlsView = new DataView(emu.memory.buffer);
  tlsView.setBigUint64(TPIDR_OFFSET, BigInt(0x900000), true);

  const io = new IOManager();
  io.pushStdin(new TextEncoder().encode(stdinText));

  let stdout = '';
  let exitCode: number | undefined;
  const syscalls = new SyscallHandler(elfInfo.brkStart, {
    writeStdout: (data: Uint8Array) => { stdout += new TextDecoder().decode(data); },
    writeStderr: (data: Uint8Array) => { stdout += new TextDecoder().decode(data); },
    readStdin: (count: number) => io.readStdin(count),
  }, vfs);

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

    // Refresh view in case memory grew
    view = new DataView(emu.memory.buffer);
    const insns = Number(view.getBigUint64(272, true));
    totalInsns += insns;
    view.setBigUint64(272, 0n, true);

    if (pc === SVC) {
      const svcPC = view.getBigUint64(256, true);
      const result = syscalls.handle(emu.cpu, emu.memory);
      view = new DataView(emu.memory.buffer); // refresh after syscall (brk/mmap may grow)

      if (result.status === 'exit') {
        // Check for child process exit
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
        emu.processManager.fork(
          emu.memory, emu.cpu,
          syscalls.getState(),
          result.resumePC,
        );
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

    if (totalInsns >= 500_000_000 || blockCount >= 5_000_000) break;
  }

  let haltInfo: string | undefined;
  if (pc === HALT) {
    view = new DataView(emu.memory.buffer);
    const raw = view.getUint32(Number(prevPC) + GUEST_MEM_BASE, true);
    const d = decode(raw);
    haltInfo = `HALT at block ${blockCount}: pc=0x${prevPC.toString(16)}, raw=0x${raw.toString(16).padStart(8, '0')}, kind=${d.kind}`;
  }

  return { stdout, totalInsns, blockCount, haltInfo, exitCode };
}

function bootShell(stdinText: string) {
  return runBusybox(['sh'], stdinText);
}

describe('busybox boot', () => {
  it('boots to shell prompt', () => {
    const result = bootShell('');
    if (!result) { console.log('busybox not found, skipping'); return; }
    expect(result.stdout).toContain('# ');
    expect(result.totalInsns).toBeGreaterThan(0);
    console.log(`Boot: ${result.totalInsns.toLocaleString()} insns in ${result.blockCount} blocks`);
  });

  it('runs echo command', () => {
    const result = bootShell('echo hello world\n');
    if (!result) { console.log('busybox not found, skipping'); return; }
    expect(result.stdout).toContain('hello world');
    console.log(`Echo: ${result.totalInsns.toLocaleString()} insns`);
  });

  it('runs cd and pwd', () => {
    const result = bootShell('cd /tmp\npwd\n');
    if (!result) { console.log('busybox not found, skipping'); return; }
    expect(result.stdout).toContain('/tmp');
    console.log(`cd+pwd: ${result.totalInsns.toLocaleString()} insns`);
  });

  it('handles cd to nonexistent dir', () => {
    const result = bootShell('cd test\necho ok\n');
    if (!result) { console.log('busybox not found, skipping'); return; }
    console.log(`cd-fail: ${result.totalInsns.toLocaleString()} insns`);
    console.log(`Output: ${JSON.stringify(result.stdout.slice(0, 500))}`);
    expect(result.stdout).toContain('ok');
  });

  it('runs ls / via shell (fork+exec)', () => {
    const result = bootShell('ls /\n');
    if (!result) { console.log('busybox not found, skipping'); return; }
    console.log(`ls via shell: ${result.totalInsns.toLocaleString()} insns`);
    console.log(`Output: ${JSON.stringify(result.stdout.slice(0, 500))}`);
    if (result.haltInfo) console.log(result.haltInfo);
    expect(result.stdout).toContain('bin');
    expect(result.stdout).toContain('dev');
  });

  it('runs ls / directly (no fork needed)', () => {
    const result = runBusybox(['ls', '/']);
    if (!result) { console.log('busybox not found, skipping'); return; }
    console.log(`ls direct: ${result.totalInsns.toLocaleString()} insns, exit=${result.exitCode}`);
    console.log(`Output: ${JSON.stringify(result.stdout)}`);
    if (result.haltInfo) console.log(result.haltInfo);
    // May fail initially — iterating on missing instructions/syscalls
    expect(result.stdout).toContain('bin');
    expect(result.stdout).toContain('dev');
    expect(result.stdout).toContain('etc');
    expect(result.stdout).toContain('tmp');
    expect(result.exitCode).toBe(0);
  });

  it('runs cat /etc/passwd via shell (fork+exec)', () => {
    const result = bootShell('cat /etc/passwd\n');
    if (!result) { console.log('busybox not found, skipping'); return; }
    console.log(`cat via shell: ${result.totalInsns.toLocaleString()} insns`);
    console.log(`Output: ${JSON.stringify(result.stdout.slice(0, 500))}`);
    expect(result.stdout).toContain('root:x:0:0');
  });

  it('runs cat /etc/passwd directly', () => {
    const result = runBusybox(['cat', '/etc/passwd']);
    if (!result) { console.log('busybox not found, skipping'); return; }
    console.log(`cat direct: ${result.totalInsns.toLocaleString()} insns, exit=${result.exitCode}`);
    expect(result.stdout).toContain('root:x:0:0');
    expect(result.exitCode).toBe(0);
  });

  it('runs wc -c on /etc/passwd directly', () => {
    const result = runBusybox(['wc', '-c', '/etc/passwd']);
    if (!result) { console.log('busybox not found, skipping'); return; }
    console.log(`wc direct: ${result.totalInsns.toLocaleString()} insns, exit=${result.exitCode}`);
    console.log(`Output: ${JSON.stringify(result.stdout)}`);
    // /etc/passwd is "root:x:0:0:root:/root:/bin/sh\n" = 30 bytes
    expect(result.stdout).toContain('30');
    expect(result.exitCode).toBe(0);
  });

  it('runs echo hello | cat via shell (pipe)', () => {
    const result = bootShell('echo hello | cat\n');
    if (!result) { console.log('busybox not found, skipping'); return; }
    console.log(`echo|cat: ${result.totalInsns.toLocaleString()} insns`);
    console.log(`Output: ${JSON.stringify(result.stdout.slice(0, 500))}`);
    if (result.haltInfo) console.log(result.haltInfo);
    expect(result.stdout).toContain('hello');
  });

  it('runs ls / | grep bin via shell (pipe)', () => {
    const result = bootShell('ls / | grep -F bin\n');
    if (!result) { console.log('busybox not found, skipping'); return; }
    console.log(`ls|grep: ${result.totalInsns.toLocaleString()} insns`);
    console.log(`Output: ${JSON.stringify(result.stdout.slice(0, 500))}`);
    if (result.haltInfo) console.log(result.haltInfo);
    expect(result.stdout).toContain('bin');
  });

  it('runs grep directly', () => {
    const result = runBusybox(['grep', '-F', 'root', '/etc/passwd']);
    if (!result) { console.log('busybox not found, skipping'); return; }
    console.log(`grep direct: ${result.totalInsns.toLocaleString()} insns, exit=${result.exitCode}`);
    console.log(`Output: ${JSON.stringify(result.stdout.slice(0, 300))}`);
    if (result.haltInfo) console.log(result.haltInfo);
    expect(result.stdout).toContain('root');
    expect(result.exitCode).toBe(0);
  });

  it('runs grep with regex (no -F)', () => {
    const r = runBusybox(['grep', 'root', '/etc/passwd']);
    if (!r) return;
    expect(r.exitCode).toBe(0);
    expect(r.stdout).toContain('root:');
    expect(r.stdout).toContain('/bin/sh');
  });

  it('runs many echo commands', () => {
    const elfBytes = loadBusybox();
    if (!elfBytes) { console.log('busybox not found, skipping'); return; }

    const emu = new Emulator(1024);
    emu.singleBlockMode = true;
    const vfs = new InMemoryVFS();
    vfs.loadBinary(elfBytes);
    const elfInfo = loadElf(elfBytes, emu.memory);
    const sp = setupStack(emu.memory, elfInfo, ['sh'],
      ['HOME=/root', 'TERM=xterm-256color', 'PATH=/bin', 'PWD=/']);
    emu.cpu.setSP(sp);
    const tlsView = new DataView(emu.memory.buffer);
    tlsView.setBigUint64(TPIDR_OFFSET, BigInt(0x900000), true);
    const io = new IOManager();
    // Build 10 echo commands
    const cmds = Array.from({length: 10}, (_, i) => `echo ${i+1}`).join('\n') + '\n';
    io.pushStdin(new TextEncoder().encode(cmds));

    let stdout = '';
    const syscalls = new SyscallHandler(elfInfo.brkStart, {
      writeStdout: (data: Uint8Array) => { stdout += new TextDecoder().decode(data); },
      writeStderr: (data: Uint8Array) => { stdout += new TextDecoder().decode(data); },
      readStdin: (count: number) => io.readStdin(count),
    }, vfs);

    let pc = elfInfo.entry;
    let totalInsns = 0;
    const view = new DataView(emu.memory.buffer);
    const HALT = 0xDEAD0000n;
    const SVC = 0xDEAD0001n;
    let blockCount = 0;
    let prevPC = pc;

    while (pc !== HALT) {
      blockCount++;

      const region = (emu as any).getRegion(pc);
      prevPC = pc;
      pc = region.run();
      const insns = Number(view.getBigUint64(272, true));
      totalInsns += insns;
      view.setBigUint64(272, 0n, true);

      if (pc === SVC) {
        const svcPC = view.getBigUint64(256, true);
        const result = syscalls.handle(emu.cpu, emu.memory);
        if (result.status === 'exit') { console.log(`Exit with code ${result.code}`); break; }
        if (result.status === 'blocked_read') { console.log(`Blocked on read at ${totalInsns} insns`); break; }
        pc = svcPC + 4n;
        continue;
      }
      if (totalInsns >= 500_000_000 || blockCount >= 5_000_000) {
        console.log(`Hit limit: ${totalInsns} insns, ${blockCount} blocks`);
        break;
      }
    }

    console.log(`\nTotal: ${totalInsns.toLocaleString()} insns, ${blockCount} blocks`);
    console.log(`Output: ${JSON.stringify(stdout.slice(0, 500))}`);
    expect(stdout).toContain('1');
  });
});
