// dump the per-insn kinds inside the zig block containing a probe pc

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { Emulator } from '../src/dispatch.js';
import { loadElf } from '../src/elf.js';
import { GUEST_MEM_BASE } from '../src/types.js';
import { decode } from '../src/decode.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');

async function main() {
  const elfBytes = new Uint8Array(readFileSync(join(ROOT, 'assets/busybox-aarch64')));
  const emu = new Emulator(1024);
  emu.elfBytes = elfBytes;
  loadElf(elfBytes, emu.memory);
  const view = new DataView(emu.memory.buffer);

  const entryPC = BigInt(process.argv[2] ?? '0x401ee0');
  const probePC = BigInt(process.argv[3] ?? '0x401fcc');

  const transWasm = new Uint8Array(readFileSync(join(ROOT, 'engine/zig-out/bin/translator.wasm')));
  const inst = new WebAssembly.Instance(new WebAssembly.Module(transWasm));
  const ex = inst.exports as any;

  const codeBytes = new Uint8Array(emu.memory.buffer, Number(entryPC) + GUEST_MEM_BASE, 4096).slice();
  new Uint8Array(ex.memory.buffer).set(codeBytes, ex.translator_input_buf_offset());

  const lo = Number(entryPC & 0xffffffffn);
  const hi = Number((entryPC >> 32n) & 0xffffffffn);
  const plo = Number(probePC & 0xffffffffn);
  const phi = Number((probePC >> 32n) & 0xffffffffn);
  const n = ex.translator_debug_block_insns(lo, hi, lo, hi, codeBytes.length, plo, phi);
  if (n === -1 || n === 0xffffffff) { console.log('buildRegion failed'); return; }
  if (n === 0) { console.log('no block contains probe pc'); return; }

  const off = ex.translator_result_offset();
  const tv = new DataView(ex.memory.buffer);
  console.log(`${n} insns in the block containing 0x${probePC.toString(16)}:`);
  for (let i = 0; i < n; i++) {
    const pc = tv.getBigUint64(off + i * 16, true);
    const kind = tv.getBigUint64(off + i * 16 + 8, true);
    const raw = view.getUint32(Number(pc) + GUEST_MEM_BASE, true);
    console.log(`  0x${pc.toString(16)}: raw=0x${raw.toString(16).padStart(8, '0')} zigkind=${kind} ts=${decode(raw).kind}`);
  }
}

main();
