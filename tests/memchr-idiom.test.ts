import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';
import { Emulator } from '../src/dispatch.js';
import { loadElf } from '../src/elf.js';
import { GUEST_MEM_BASE, V_OFFSET } from '../src/types.js';
import { initZigTranslator } from '../src/zig-translator.js';

const PC = 0x600000, RET = 0xd65f03c0, BUF = 0x610000;
let emu: Emulator | null = null;

beforeAll(() => {
  const t = new Uint8Array(readFileSync(join(__dirname, '..', 'engine', 'zig-out', 'bin', 'translator.wasm')));
  initZigTranslator(t);
  emu = new Emulator(1024);
  emu.elfBytes = new Uint8Array(readFileSync(join(__dirname, '..', 'assets', 'busybox-aarch64')));
  loadElf(emu.elfBytes, emu.memory);
});

function runWords(words: number[], gprs: Record<number, bigint>, bufBytes?: number[]): DataView {
  const v = new DataView(emu!.memory.buffer);
  words.forEach((w, i) => v.setUint32(PC + i * 4 + GUEST_MEM_BASE, w, true));
  v.setUint32(PC + words.length * 4 + GUEST_MEM_BASE, RET, true);
  for (let i = 0; i < 32; i++) { v.setBigUint64(V_OFFSET + i * 16, 0n, true); v.setBigUint64(V_OFFSET + i * 16 + 8, 0n, true); }
  for (let i = 0; i < 31; i++) v.setBigUint64(i * 8, 0n, true);
  v.setBigUint64(0x100, BigInt(PC), true);
  v.setBigUint64(0xf8, 0x700000n, true);
  v.setBigUint64(30 * 8, BigInt(PC + 0x800), true);
  for (const [r, val] of Object.entries(gprs)) v.setBigUint64(Number(r) * 8, BigInt.asUintN(64, val), true);
  if (bufBytes) bufBytes.forEach((b, i) => v.setUint8(BUF + GUEST_MEM_BASE + i, b));
  (emu as any).regionCache.clear();
  (emu as any).getRegion(BigInt(PC)).run();
  return new DataView(emu!.memory.buffer);
}

const DUP_V0_W3 = 0x4e010c60, LDR_Q1_X1_X9 = 0x3ce96821, CMEQ = 0x6e208c21,
      UMAXV = 0x6e30a822, FMOV_W8_S2 = 0x1e260048;

describe('the memchr idiom: dup / ldr q / cmeq / umaxv / fmov', () => {
  it('finds a NUL in the first sixteen bytes', () => {
    const buf = new Array(32).fill(0x41); buf[5] = 0;
    const v = runWords([DUP_V0_W3, LDR_Q1_X1_X9, CMEQ, UMAXV, FMOV_W8_S2],
                       { 1: BigInt(BUF), 3: 0n, 9: 0n }, buf);
    expect(Number(v.getBigUint64(8 * 8, true) & 0xffn)).toBe(0xff);
  });
  it('reports nothing when the byte is absent', () => {
    const buf = new Array(32).fill(0x41);
    const v = runWords([DUP_V0_W3, LDR_Q1_X1_X9, CMEQ, UMAXV, FMOV_W8_S2],
                       { 1: BigInt(BUF), 3: 0n, 9: 0n }, buf);
    expect(Number(v.getBigUint64(8 * 8, true) & 0xffffffffn)).toBe(0);
  });
  it('finds a non-zero byte', () => {
    const buf = new Array(32).fill(0x41); buf[9] = 0x7a;
    const v = runWords([DUP_V0_W3, LDR_Q1_X1_X9, CMEQ, UMAXV, FMOV_W8_S2],
                       { 1: BigInt(BUF), 3: 0x7an, 9: 0n }, buf);
    expect(Number(v.getBigUint64(8 * 8, true) & 0xffn)).toBe(0xff);
  });
  it('honours the x9 offset', () => {
    const buf = new Array(48).fill(0x41); buf[20] = 0;
    const v = runWords([DUP_V0_W3, LDR_Q1_X1_X9, CMEQ, UMAXV, FMOV_W8_S2],
                       { 1: BigInt(BUF), 3: 0n, 9: 16n }, buf);
    expect(Number(v.getBigUint64(8 * 8, true) & 0xffn)).toBe(0xff);
  });
});

describe('the memchr idiom as a LOOP (self-loop path)', () => {
  const DUP = 0x4e010c60, MOVZ9 = 0xaa1f03e9, LDR = 0x3ce96821, CMEQ2 = 0x6e208c21,
        UMAXV2 = 0x6e30a822, FMOV = 0x1e260048, TBNZ = 0x37000088,
        ADD16 = 0x91004129, CMP64 = 0xf101013f, BLO = 0x54ffff23, RETI = 0xd65f03c0;
  it('finds a zero on the third pass and stops', () => {
    const buf = new Array(80).fill(0x41); buf[40] = 0;
    const v = runWords([DUP, MOVZ9, LDR, CMEQ2, UMAXV2, FMOV, TBNZ, ADD16, CMP64, BLO, RETI],
                       { 1: BigInt(BUF), 3: 0n }, buf);
    // x9 is the offset of the 16-byte block that matched: 32
    expect(Number(v.getBigUint64(9 * 8, true))).toBe(32);
    expect(Number(v.getBigUint64(8 * 8, true) & 0xffn)).toBe(0xff);
  });
  it('runs the loop out when nothing matches', () => {
    const buf = new Array(80).fill(0x41);
    const v = runWords([DUP, MOVZ9, LDR, CMEQ2, UMAXV2, FMOV, TBNZ, ADD16, CMP64, BLO, RETI],
                       { 1: BigInt(BUF), 3: 0n }, buf);
    expect(Number(v.getBigUint64(9 * 8, true))).toBe(64);
    expect(Number(v.getBigUint64(8 * 8, true) & 0xffffffffn)).toBe(0);
  });
});

describe('the exact memchr loop from the corpus binary', () => {
  // 104bee0 onwards, the >32-byte path: two 16-byte probes per pass
  const DUP = 0x4e010c60, MOVZ9 = 0xaa1f03e9;
  const W = [
    DUP,           // 0:  dup   v0.16b, w3
    MOVZ9,         // 4:  mov   x9, xzr
    0x3ce96821,    // 8:  ldr   q1, [x1, x9]      <- loop
    0x6e208c21,    // 12: cmeq  v1.16b, v1.16b, v0.16b
    0x6e30a822,    // 16: umaxv b2, v1.16b
    0x1e260048,    // 20: fmov  w8, s2
    0x37000148,    // 24: tbnz  w8, #0, +40 (found: the ret at 64)
    0x8b090028,    // 28: add   x8, x1, x9
    0x3dc00501,    // 32: ldr   q1, [x8, #16]
    0x6e208c21,    // 36: cmeq
    0x6e30a822,    // 40: umaxv
    0x1e260048,    // 44: fmov  w8, s2
    0x37000088,    // 48: tbnz  w8, #0, +16 (found)
    0x91008129,    // 52: add   x9, x9, #0x20
    0xf101013f,   // 56: cmp   x9, #64
    0x54fffe63,    // 60: b.cc  loop  (8)
    0xd65f03c0,    // 64: ret
  ];
  it('finds a byte in the SECOND probe of a pass', () => {
    const buf = new Array(96).fill(0x41); buf[20] = 0;   // inside [16,32)
    const v = runWords(W, { 1: BigInt(BUF), 3: 0n }, buf);
    expect(Number(v.getBigUint64(9 * 8, true))).toBe(0);
    expect(Number(v.getBigUint64(8 * 8, true) & 0xffn)).toBe(0xff);
  });
  it('finds a byte in the second pass', () => {
    const buf = new Array(96).fill(0x41); buf[33] = 0;   // inside [32,48)
    const v = runWords(W, { 1: BigInt(BUF), 3: 0n }, buf);
    expect(Number(v.getBigUint64(9 * 8, true))).toBe(32);
  });
  it('walks the whole range when nothing matches', () => {
    const buf = new Array(96).fill(0x41);
    const v = runWords(W, { 1: BigInt(BUF), 3: 0n }, buf);
    expect(Number(v.getBigUint64(9 * 8, true))).toBe(64);
  });
});
