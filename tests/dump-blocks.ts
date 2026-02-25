import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { Emulator } from '../src/dispatch.js';
import { loadElf } from '../src/elf.js';
import { GUEST_MEM_BASE } from '../src/types.js';
import { decode } from '../src/decode.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const elfPath = join(__dirname, '..', 'assets', 'busybox-aarch64');
const elfBytes = new Uint8Array(readFileSync(elfPath));

// Check ELF type
const elfDv = new DataView(elfBytes.buffer);
const eType = elfDv.getUint16(16, true);
const rawEntry = elfDv.getBigUint64(24, true);
console.log(`ELF type: ${eType} (2=EXEC, 3=DYN/PIE), entry: 0x${rawEntry.toString(16)}`);

// Read program headers to check segment addrs
const phoff = Number(elfDv.getBigUint64(32, true));
const phentSize = elfDv.getUint16(54, true);
const phnum = elfDv.getUint16(56, true);
for (let i = 0; i < phnum; i++) {
  const base = phoff + i * phentSize;
  const pType = elfDv.getUint32(base, true);
  const pVaddr = elfDv.getBigUint64(base + 16, true);
  const pMemsz = elfDv.getBigUint64(base + 40, true);
  if (pType === 1) { // PT_LOAD
    console.log(`  PT_LOAD: vaddr=0x${pVaddr.toString(16)}, memsz=0x${pMemsz.toString(16)}`);
  }
}

// Check the raw file value at the GOT offset before loading
// Second segment: vaddr=0x11D130. Guest addr = vaddr + loadBase = 0x51D130.
// Target GOT addr: guest 0x51FFC0, offset in segment = 0x51FFC0 - 0x51D130 = 0x2E90
const phoff2 = Number(elfDv.getBigUint64(32, true));
for (let i = 0; i < phnum; i++) {
  const base = phoff2 + i * phentSize;
  const pType = elfDv.getUint32(base, true);
  if (pType !== 1) continue;
  const pOffset = Number(elfDv.getBigUint64(base + 8, true));
  const pVaddr = Number(elfDv.getBigUint64(base + 16, true));
  const pFilesz = Number(elfDv.getBigUint64(base + 32, true));
  // Check if 0x51FFC0 maps into this segment after load at loadBase=0x400000
  const guestStart = pVaddr + 0x400000;
  const guestEnd = guestStart + pFilesz;
  if (0x51FFC0 >= guestStart && 0x51FFC0 < guestEnd) {
    const segOff = 0x51FFC0 - guestStart;
    const fileOff = pOffset + segOff;
    const rawVal = elfDv.getBigUint64(fileOff, true);
    console.log(`Raw file value at GOT offset ${fileOff.toString(16)}: 0x${rawVal.toString(16)}`);
    console.log(`  Expected after relocation: 0x${(rawVal + 0x400000n).toString(16)}`);
  }
}

const emu = new Emulator(1024);
const elfInfo = loadElf(elfBytes, emu.memory);
console.log(`loadBase: 0x${elfInfo.loadBase.toString(16)}, entry: 0x${elfInfo.entry.toString(16)}`);
const view = new DataView(emu.memory.buffer);

// Test specific encodings
const testInsns = [
  0x4e010c20, // DUP V0.16B, V1.B[0]
  0x4e083c01, // UMOV X1, V0.D[0]
  0x3d800000, // STR Q0, [X0]
  0x3d800400, // STR Q0, [X0, #16]
  0x3c9f0080, // STUR Q0, [X4, #-16]
  0xad010000, // STP Q0, Q0, [X0]
  0xad3f0080, // STP Q0, Q0, [X4, #-32]! (pre-index)
  0x4f00041f, // MOVI V31.4S, #0
  0x0f00041f, // MOVI V31.2S, #0 (64-bit)
];
for (const raw of testInsns) {
  const insn = decode(raw);
  console.log(`0x${raw.toString(16).padStart(8, '0')}: ${insn.kind} ${JSON.stringify(insn)}`);
}

// Check init_array data
// ADRP at 0x4b4854: PC_page = 0x4b4000, imm = 0x6B000 → X19 = 0x51F000
// Then ldr X19, [X19, #4032] → reads from 0x51FFC0
const initArrayPtrAddr = 0x51FFC0;
const initArrayEndPtrAddr = 0x51FF70;
console.log(`\n=== Checking .init_array pointers ===`);
console.log(`  *0x${initArrayPtrAddr.toString(16)} = 0x${view.getBigUint64(initArrayPtrAddr + GUEST_MEM_BASE, true).toString(16)}`);
console.log(`  *0x${initArrayEndPtrAddr.toString(16)} = 0x${view.getBigUint64(initArrayEndPtrAddr + GUEST_MEM_BASE, true).toString(16)}`);

// Read the actual init_array entries
const initArrayStart = Number(view.getBigUint64(initArrayPtrAddr + GUEST_MEM_BASE, true));
const initArrayEnd = Number(view.getBigUint64(initArrayEndPtrAddr + GUEST_MEM_BASE, true));
console.log(`  init_array range: 0x${initArrayStart.toString(16)} - 0x${initArrayEnd.toString(16)}`);
if (initArrayStart > 0 && initArrayStart < 0x600000) {
  console.log(`  Entries:`);
  for (let a = initArrayStart; a < Math.min(initArrayEnd, initArrayStart + 80); a += 8) {
    const v = view.getBigUint64(a + GUEST_MEM_BASE, true);
    console.log(`    [0x${a.toString(16)}] = 0x${v.toString(16)}`);
  }
}

// Check memory around 0x51F000 (data segment)
console.log(`\n=== Data around 0x51FFC0 ===`);
for (let off = -16; off < 32; off += 8) {
  const addr = initArrayPtrAddr + off;
  const val = view.getBigUint64(addr + GUEST_MEM_BASE, true);
  console.log(`  0x${addr.toString(16)}: 0x${val.toString(16)}`);
}
