// dump the zig block table for a region (multi-block mode window semantics,
// matching zig-translator.ts: forward 4096-byte window from entry).

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { Emulator } from '../src/dispatch.js';
import { loadElf, setupStack } from '../src/elf.js';
import { TPIDR_OFFSET, GUEST_MEM_BASE } from '../src/types.js';
import { decode } from '../src/decode.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');

const EXIT_KINDS = ['fallthrough_to_block', 'direct_branch', 'cond_branch', 'indirect_branch', 'syscall', 'unknown_insn', 'region_full'];

async function main() {
  const elfBytes = new Uint8Array(readFileSync(join(ROOT, 'assets/busybox-aarch64')));
  const emu = new Emulator(1024);
  emu.elfBytes = elfBytes;
  loadElf(elfBytes, emu.memory);
  const view = new DataView(emu.memory.buffer);

  const entryPC = BigInt(process.argv[2] ?? '0x401ee0');

  const transWasm = new Uint8Array(readFileSync(join(ROOT, 'engine/zig-out/bin/translator.wasm')));
  const inst = new WebAssembly.Instance(new WebAssembly.Module(transWasm));
  const ex = inst.exports as any;

  const codeBytes = new Uint8Array(emu.memory.buffer, Number(entryPC) + GUEST_MEM_BASE, 4096).slice();
  new Uint8Array(ex.memory.buffer).set(codeBytes, ex.translator_input_buf_offset());

  const lo = Number(entryPC & 0xffffffffn);
  const hi = Number((entryPC >> 32n) & 0xffffffffn);
  const n = ex.translator_debug_region(lo, hi, lo, hi, codeBytes.length);
  if (n === 0xffffffff) { console.log('buildRegion failed'); return; }

  const off = ex.translator_result_offset();
  const tv = new DataView(ex.memory.buffer);
  console.log(`${n} blocks for region 0x${entryPC.toString(16)}:`);
  for (let i = 0; i < n; i++) {
    const start = tv.getBigUint64(off + i * 40, true);
    const end = tv.getBigUint64(off + i * 40 + 8, true);
    const cnt = tv.getBigUint64(off + i * 40 + 16, true);
    const kind = Number(tv.getBigUint64(off + i * 40 + 24, true));
    const detail = tv.getBigUint64(off + i * 40 + 32, true);
    let d = '';
    if (kind === 0 || kind === 1) d = `-> 0x${detail.toString(16)}`;
    if (kind === 2) d = `taken=0x${(detail & 0xffffffffn).toString(16)} ft=0x${(detail >> 32n).toString(16)}`;
    const lastInsnPC = Number(end) - 4;
    const raw = view.getUint32(lastInsnPC + GUEST_MEM_BASE, true);
    console.log(`  b${i}: 0x${start.toString(16)}..0x${end.toString(16)} (${cnt} insns) ${EXIT_KINDS[kind]} ${d}  [last: 0x${raw.toString(16).padStart(8, '0')} ${decode(raw).kind}]`);
  }
}

main();
