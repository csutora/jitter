// print the complete applet list of the app's bundled busybox (assets/busybox-aarch64),
// so NativeFS can create an accurate, complete symlink set instead of a hand-maintained one.
import { readFileSync } from 'fs';
import { Emulator, IOManager } from '../src/dispatch.js';
import { loadElf, setupStack } from '../src/elf.js';
import { SyscallHandler } from '../src/syscall.js';
import { InMemoryVFS } from '../src/vfs.js';
import { TPIDR_OFFSET } from '../src/types.js';
import { initZigTranslator } from '../src/zig-translator.js';

initZigTranslator(new Uint8Array(readFileSync('engine/zig-out/bin/translator.wasm')));
const elfBytes = new Uint8Array(readFileSync('assets/busybox-aarch64'));
const emu = new Emulator(1024);
emu.elfBytes = elfBytes;
const vfs = new InMemoryVFS(); vfs.loadBinary(elfBytes);
const elfInfo = loadElf(elfBytes, emu.memory);
emu.cpu.setSP(setupStack(emu.memory, elfInfo, ['busybox', '--list'], ['PATH=/bin', 'PWD=/']));
new DataView(emu.memory.buffer).setBigUint64(TPIDR_OFFSET, 0x900000n, true);

let out = '';
const syscalls = new SyscallHandler(elfInfo.brkStart, {
  writeStdout: (d) => { out += new TextDecoder().decode(d); },
  writeStderr: () => {}, readStdin: () => new Uint8Array(),
}, vfs);

let pc = elfInfo.entry;
const HALT = 0xDEAD0000n, SVC = 0xDEAD0001n;
let blocks = 0;
while (pc !== HALT) {
  blocks++;
  pc = (emu as any).getRegion(pc).run();
  if (pc === SVC) {
    const svcPC = new DataView(emu.memory.buffer).getBigUint64(256, true);
    const r: any = syscalls.handle(emu.cpu, emu.memory);
    if (r.status === 'exit') break;
    pc = svcPC + 4n; continue;
  }
  if (blocks > 2_000_000) break;
}
process.stdout.write(out);
