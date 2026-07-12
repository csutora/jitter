// quick helper: disassemble a few instructions at a given pc in busybox-aarch64.
// usage: tsx tools/disasm-pc.ts 0x4cec94 [count]

import { readFileSync } from 'fs';
import { decode } from '../src/decode.js';

const pcHex = process.argv[2] ?? '0x4cec94';
const count = parseInt(process.argv[3] ?? '12', 10);
const pc = parseInt(pcHex, 16);

const elf = readFileSync('assets/busybox-aarch64');
// loadable PIE base is 0x400000 in the existing setup; insns at virtual pc map
// to elf file offset (pc - 0x400000).
const fileOff = pc - 0x400000;

for (let i = 0; i < count; i++) {
  const off = fileOff + i * 4;
  const insn = elf.readUInt32LE(off);
  const d = decode(insn);
  console.log(`0x${(pc + i * 4).toString(16).padStart(6, '0')}: 0x${insn.toString(16).padStart(8, '0')}  ${d.kind} ${JSON.stringify(d, (_k, v) => typeof v === 'bigint' ? v.toString() : v).slice(0, 160)}`);
}
