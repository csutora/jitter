// run Alpine's musl-dynamic busybox through the emulator, proving a real distro rootfs
// works end-to-end via the dynamic loader. seeds the whole minirootfs into an in-memory
// vfs, then execs /bin/busybox <applet> and prints output.
// usage: node --import tsx tools/run-alpine.ts [applet] [args...]   (default: echo hello)

import { readFileSync, readdirSync, lstatSync, readlinkSync, realpathSync } from 'fs';
import { join, relative } from 'path';
import { Emulator, IOManager } from '../src/dispatch.js';
import { loadProgram } from '../src/elf.js';
import { SyscallHandler } from '../src/syscall.js';
import { InMemoryVFS } from '../src/vfs.js';
import { TPIDR_OFFSET } from '../src/types.js';
import { initZigTranslator } from '../src/zig-translator.js';

const ROOT = 'tests/dynamic/alpine';
const dec = (d: Uint8Array) => new TextDecoder().decode(d);
initZigTranslator(new Uint8Array(readFileSync('engine/zig-out/bin/translator.wasm')));

const vfs = new InMemoryVFS();
// seed every regular file (resolve symlinks to their target bytes) into the vfs
let seeded = 0;
function walk(dir: string) {
  for (const name of readdirSync(dir)) {
    const abs = join(dir, name);
    let st;
    try { st = lstatSync(abs); } catch { continue; }
    const guestPath = '/' + relative(ROOT, abs);
    if (st.isSymbolicLink()) {
      // seed symlinks-to-files as copies (the harness vfs doesn't resolve relative
      // symlinks; the real app's NativeFS does). guard against links escaping the tree.
      try {
        const real = realpathSync(abs);
        if (real.startsWith(realpathSync(ROOT)) && lstatSync(real).isFile()) {
          vfs.writeFileBytes(guestPath, new Uint8Array(readFileSync(real)), 0o755); seeded++;
        }
      } catch {}
    } else if (st.isDirectory()) {
      walk(abs); // lstat above means we never follow a symlink out of the tree
    } else if (st.isFile()) {
      try { vfs.writeFileBytes(guestPath, new Uint8Array(readFileSync(abs)), st.mode & 0o777); seeded++; } catch {}
    }
  }
}
walk(ROOT);
console.error(`seeded ${seeded} files from ${ROOT}`);

const emu = new Emulator(2048);
// if the first arg is an absolute path, run that binary directly; else busybox <applet>
const first = process.argv[2] ?? 'echo';
let binPath: string, argv: string[];
if (first.startsWith('/')) {
  binPath = first;
  argv = [first, ...process.argv.slice(3)];
} else {
  binPath = '/bin/busybox';
  const rest = process.argv.slice(3);
  argv = [binPath, first, ...(rest.length ? rest : (first === 'echo' ? ['hello from alpine busybox'] : []))];
}
const elfBytes = new Uint8Array(readFileSync(realpathSync(join(ROOT, binPath))));
emu.elfBytes = elfBytes;
const readInterp = (p: string) => {
  // resolve within the alpine tree, following symlinks
  const abs = realpathSync(join(ROOT, p));
  return new Uint8Array(readFileSync(abs));
};

const { entry: startPC, sp, brkStart, interp } = loadProgram(
  emu.memory, elfBytes, argv, ['PATH=/bin:/usr/bin:/sbin', 'HOME=/root', 'TERM=linux', 'PWD=/', ...(process.env.OPENSSL_armcap !== undefined ? [`OPENSSL_armcap=${process.env.OPENSSL_armcap}`] : [])], readInterp,
);
console.error(`load: interp=${interp} startPC=0x${startPC.toString(16)} argv=${JSON.stringify(argv)}`);
emu.cpu.setSP(sp);
new DataView(emu.memory.buffer).setBigUint64(TPIDR_OFFSET, 0x900000n, true);

const io = new IOManager();
let stdout = '', stderr = '';
const syscalls = new SyscallHandler(brkStart, {
  writeStdout: (d: Uint8Array) => { stdout += dec(d); },
  writeStderr: (d: Uint8Array) => { stderr += dec(d); },
  readStdin: (n: number) => io.readStdin(n),
}, vfs);

const HALT = 0xDEAD0000n, SVC = 0xDEAD0001n;
let pc = startPC, blocks = 0, exitCode: number | undefined, svcs = 0, sigills = 0;
const sc: any = syscalls;
function deliverQueued(pcv: bigint): bigint | null {
  if (!sc.pendingSignals?.length) return pcv;
  emu.cpu.setPC(pcv);
  while (sc.pendingSignals.length > 0) {
    const sig = sc.pendingSignals.shift();
    const r: any = sc.deliverSignal(emu.cpu, emu.memory, sig);
    if (r.action === 'exit') { exitCode = r.exitCode ?? 128 + sig; return null; }
    if (r.delivered && r.action === 'continue') return emu.cpu.getPC();
  }
  return pcv;
}
try {
  while (pc !== HALT) {
    blocks++;
    const region = (emu as any).getRegion(pc);
    pc = region.run();
    if (region.lastExitReason === 3 /* unknown insn -> SIGILL, like worker.ts */) {
      sigills++;
      emu.cpu.setPC(pc);
      sc.raiseSignal(4);
      const next = deliverQueued(pc);
      if (next === null) break;
      pc = next;
      continue;
    }
    if (pc === SVC) {
      const num = Number(emu.cpu.getReg(8));
      svcs++;
      const r: any = syscalls.handle(emu.cpu, emu.memory);
      if (r.status === 'exit') { exitCode = r.code; break; }
      if (r.status === 'bridge') { console.error(`unhandled bridge nr=${num} method=${r.method}`); break; }
      if (r.status === 'blocked_read') { break; }
      pc = new DataView(emu.memory.buffer).getBigUint64(256, true) + 4n;
      const next = deliverQueued(pc);
      if (next === null) break;
      pc = next;
      continue;
    }
    if (blocks > 40_000_000) { console.error(`budget exceeded at pc=0x${pc.toString(16)}`); break; }
  }
} catch (e: any) { console.error(`CRASH at pc=0x${pc.toString(16)}: ${e.message}`); }
console.error(`sigills=${sigills}`);

console.error(`--- exit=${exitCode} blocks=${blocks} svcs=${svcs} ---`);
console.error(`STDOUT: ${JSON.stringify(stdout)}`);
if (stderr) console.error(`STDERR: ${JSON.stringify(stderr)}`);
