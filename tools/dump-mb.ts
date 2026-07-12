// dump the multi-block wasm output for a region starting at a specific guest PC.
// loads busybox elf into a fresh emulator, asks the zig translator to translate
// the region in multi-block mode, and writes the resulting wasm bytes to stdout
// as hex. then a separate decode of those bytes can reveal what we emitted.

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { Emulator, IOManager } from '../src/dispatch.js';
import { loadElf, setupStack } from '../src/elf.js';
import { SyscallHandler } from '../src/syscall.js';
import { InMemoryVFS } from '../src/vfs.js';
import { TPIDR_OFFSET } from '../src/types.js';
import { initZigTranslator, translateRegionZig } from '../src/zig-translator.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT = join(__dirname, '..');

const elfBytes = new Uint8Array(readFileSync(join(ROOT, 'assets/busybox-aarch64')));
initZigTranslator(new Uint8Array(readFileSync(join(ROOT, 'engine/zig-out/bin/translator.wasm'))));

const emu = new Emulator(1024);
emu.elfBytes = elfBytes;
const vfs = new InMemoryVFS();
vfs.loadBinary(elfBytes);
const elfInfo = loadElf(elfBytes, emu.memory);
const sp = setupStack(emu.memory, elfInfo, ['sh'], ['HOME=/root', 'PATH=/bin']);
emu.cpu.setSP(sp);
new DataView(emu.memory.buffer).setBigUint64(TPIDR_OFFSET, BigInt(0x900000), true);

const view = new DataView(emu.memory.buffer);
const entryPC = BigInt(process.argv[2] ?? '0x4044e4');
console.log(`translating region at 0x${entryPC.toString(16)} in multi-block mode...`);

// intercept the zig translator's output buffer before it gets compiled
const region = translateRegionZig(view, entryPC, emu.memory, 0x2000000, false);
if (!region) console.log('!! translateRegionZig returned null — translator OR compile OR instantiate failed');

// the region was compiled to a module. dump module bytes — but we lost them
// after compile. easier path: call the translator wasm exports directly.

import { readFileSync as rfs } from 'fs';
const transWasm = new Uint8Array(rfs(join(ROOT, 'engine/zig-out/bin/translator.wasm')));
const transInst = new WebAssembly.Instance(new WebAssembly.Module(transWasm));
const ex = transInst.exports as any;
const transMem = new Uint8Array(ex.memory.buffer);

// copy 4 KiB of guest code starting at entryPC into the translator's input buf
const guestStart = Number(entryPC);
// give it a bidirectional window like zig-translator.ts does for multi-block
const back = 32 * 1024;
const forward = 32 * 1024;
const baseStart = Math.max(0, guestStart - back);
const baseEnd = Math.min((emu.memory.buffer as ArrayBuffer).byteLength, guestStart + 0x10000 + forward);
const codeLen = baseEnd - (baseStart + 0x10000);
const codeBytes = new Uint8Array(emu.memory.buffer, baseStart + 0x10000, codeLen).slice();
const inOff = ex.translator_input_buf_offset();
transMem.set(codeBytes, inOff);

ex.translator_reset();
const lo = Number(BigInt(baseStart) & 0xffffffffn);
const hi = Number((BigInt(baseStart) >> 32n) & 0xffffffffn);
const elo = Number(entryPC & 0xffffffffn);
const ehi = Number((entryPC >> 32n) & 0xffffffffn);
const rc = ex.translator_translate_region(lo, hi, elo, ehi, codeLen, 0x2000000, 0);
console.log('rc=' + rc);
if (rc !== 0) { console.log('translator failed'); process.exit(1); }

const outLen = ex.translator_result_len();
const outOff = ex.translator_result_offset();
const moduleBytes = new Uint8Array(outLen);
moduleBytes.set(new Uint8Array(ex.memory.buffer, outOff, outLen));

console.log(`module size: ${outLen} bytes`);
console.log('--- hex ---');
let hex = '';
for (let i = 0; i < moduleBytes.length; i++) {
  hex += moduleBytes[i].toString(16).padStart(2, '0') + ' ';
  if ((i + 1) % 16 === 0) hex += '\n';
}
console.log(hex);

// also write to file for analysis
import { writeFileSync } from 'fs';
const outPath = '/tmp/mb-region.wasm';
writeFileSync(outPath, moduleBytes);
console.log(`wrote ${outPath}`);
