// validate DYNAMIC loading of a real musl binary: alpine's dynamically-linked busybox
// (interp /lib/ld-musl-aarch64.so.1). exercises loadProgram (main + interp), the auxv, and ld.so
// relocating + running the program. complements the on-device native-fd mmap/pread path (here the
// files are vfs-backed, so this isolates loader *correctness* for musl, not just glibc).

import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';
import { Emulator, IOManager } from '../src/dispatch.js';
import { loadProgram } from '../src/elf.js';
import { SyscallHandler } from '../src/syscall.js';
import { InMemoryVFS } from '../src/vfs.js';
import { TPIDR_OFFSET } from '../src/types.js';
import { initZigTranslator } from '../src/zig-translator.js';

const ALPINE = join(__dirname, 'dynamic', 'alpine');

function tryRead(p: string): Uint8Array | null {
  try { return new Uint8Array(readFileSync(p)); } catch { return null; }
}

beforeAll(async () => {
  await initZigTranslator(new Uint8Array(readFileSync(join(__dirname, '..', 'engine', 'zig-out', 'bin', 'translator.wasm'))));
});

// run a dynamic program from the alpine fixture with the given argv, return stdout + exit code.
function runDynamic(argv: string[]) {
  const busybox = tryRead(join(ALPINE, 'bin', 'busybox'));
  const ldmusl = tryRead(join(ALPINE, 'lib', 'ld-musl-aarch64.so.1'));
  if (!busybox || !ldmusl) return null; // fixture missing -> skip

  const emu = new Emulator(2048); // 128MB — room for ld.so's mappings above the 33MB mmap arena
  emu.elfBytes = busybox;

  // vfs must serve the interp + libc so ld.so can open()+mmap() them
  const vfs = new InMemoryVFS();
  vfs.writeFileBytes('/lib/ld-musl-aarch64.so.1', ldmusl, 0o755);
  vfs.writeFileBytes('/bin/busybox', busybox, 0o755);

  const prog = loadProgram(emu.memory, busybox, argv,
    ['PATH=/bin', 'HOME=/root', 'TERM=xterm'],
    (path) => {
      const b = tryRead(join(ALPINE, path.replace(/^\//, '')));
      if (!b) throw new Error(`interp not found: ${path}`);
      return b;
    });
  emu.cpu.setSP(prog.sp);
  new DataView(emu.memory.buffer).setBigUint64(emu.stateBase + TPIDR_OFFSET, 0x900000n, true);

  const io = new IOManager();
  let stdout = '';
  let exitCode: number | undefined;
  const syscalls = new SyscallHandler(prog.brkStart, {
    writeStdout: (d: Uint8Array) => { stdout += new TextDecoder().decode(d); },
    writeStderr: (d: Uint8Array) => { stdout += new TextDecoder().decode(d); },
    readStdin: (count: number) => io.readStdin(count),
  }, vfs);

  let pc = prog.entry;
  const HALT = 0xDEAD0000n, SVC = 0xDEAD0001n;
  let blocks = 0;
  const view0 = () => new DataView(emu.memory.buffer);
  while (pc !== HALT) {
    if (++blocks > 5_000_000) break;
    const region = (emu as any).getRegion(pc);
    pc = region.run();
    if (pc === SVC) {
      const view = view0();
      const svcPC = view.getBigUint64(emu.stateBase + 256, true);
      const result = syscalls.handle(emu.cpu, emu.memory);
      if (result.status === 'exit') { exitCode = result.code; break; }
      pc = svcPC + 4n;
    }
  }
  return { stdout, exitCode };
}

// run an arbitrary alpine binary with the whole /lib loaded into the vfs (so ld.so can find + load
// every DT_NEEDED shared object). exercises MULTI-library loading + cross-object symbol resolution.
function runAlpineBin(binGuestPath: string, argv: string[]) {
  const bin = tryRead(join(ALPINE, binGuestPath.replace(/^\//, '')));
  if (!bin) return null;
  const emu = new Emulator(4096); // 256MB — several libs + heap
  emu.elfBytes = bin;
  const vfs = new InMemoryVFS();
  // load every file under alpine/lib into /lib (readFileSync follows symlinks -> real bytes at each name)
  for (const name of readdirSync(join(ALPINE, 'lib'))) {
    const b = tryRead(join(ALPINE, 'lib', name));
    if (b) vfs.writeFileBytes('/lib/' + name, b, 0o755);
  }
  vfs.writeFileBytes(binGuestPath, bin, 0o755);

  const prog = loadProgram(emu.memory, bin, argv, ['PATH=/bin:/sbin', 'HOME=/root'],
    (path) => { const b = tryRead(join(ALPINE, path.replace(/^\//, ''))); if (!b) throw new Error(`interp ${path}`); return b; });
  emu.cpu.setSP(prog.sp);
  new DataView(emu.memory.buffer).setBigUint64(emu.stateBase + TPIDR_OFFSET, 0x900000n, true);

  const io = new IOManager();
  let stdout = '';
  const syscalls = new SyscallHandler(prog.brkStart, {
    writeStdout: (d: Uint8Array) => { stdout += new TextDecoder().decode(d); },
    writeStderr: (d: Uint8Array) => { stdout += new TextDecoder().decode(d); },
    readStdin: (count: number) => io.readStdin(count),
  }, vfs);

  let pc = prog.entry;
  const HALT = 0xDEAD0000n, SVC = 0xDEAD0001n;
  let blocks = 0, exitCode: number | undefined;
  while (pc !== HALT) {
    if (++blocks > 8_000_000) break;
    const region = (emu as any).getRegion(pc);
    pc = region.run();
    if (pc === SVC) {
      const view = new DataView(emu.memory.buffer);
      const svcPC = view.getBigUint64(emu.stateBase + 256, true);
      const result = syscalls.handle(emu.cpu, emu.memory);
      if (result.status === 'exit') { exitCode = result.code; break; }
      pc = svcPC + 4n;
    }
  }
  return { stdout, exitCode };
}

describe('dynamic loading (musl)', () => {
  it('runs alpine apk --version (multi-lib: libapk+libcrypto+libssl+libz)', () => {
    const r = runAlpineBin('/sbin/apk', ['apk', '--version']);
    if (!r) return;
    console.log('apk output:', JSON.stringify(r.stdout.slice(0, 400)), 'exit', r.exitCode);
    expect(r.stdout).not.toContain('symbol not found');
    expect(r.stdout.toLowerCase()).toContain('apk-tools');
  });


  it('runs alpine dynamic busybox: echo', () => {
    const r = runDynamic(['busybox', 'echo', 'hello from musl dynamic']);
    if (!r) return; // fixture missing
    expect(r.stdout).toContain('hello from musl dynamic');
  });

  it('runs alpine dynamic busybox: printf arithmetic via sh', () => {
    const r = runDynamic(['busybox', 'sh', '-c', 'echo $((6 * 7))']);
    if (!r) return;
    expect(r.stdout).toContain('42');
  });
});
