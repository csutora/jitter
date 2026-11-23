// NEON conformance: assemble each vector op at a scratch pc, run it through the
// zig translator on curated input vectors, and check lanes against hand-computed
// ARM-spec results. edge cases (signs, saturation, NaN) included.

import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';
import { Emulator } from '../src/dispatch.js';
import { loadElf } from '../src/elf.js';
import { GUEST_MEM_BASE, V_OFFSET } from '../src/types.js';
import { initZigTranslator } from '../src/zig-translator.js';

const PC = 0x600000, RET = 0xd65f03c0;
let emu: Emulator | null = null;

beforeAll(() => {
  let t: Uint8Array;
  try { t = new Uint8Array(readFileSync(join(__dirname, '..', 'engine', 'zig-out', 'bin', 'translator.wasm'))); }
  catch { return; }
  initZigTranslator(t);
  emu = new Emulator(1024);
  emu.elfBytes = new Uint8Array(readFileSync(join(__dirname, '..', 'assets', 'busybox-aarch64')));
  loadElf(emu.elfBytes, emu.memory);
});

// run one instruction with V registers pre-loaded; return the DataView post-run
function run(word: number, vregs: Record<number, number[]>, asFloat: 'f32' | 'f64' | 'i8' | 'i16' | 'i32' | 'i64' = 'f32', gprs?: Record<number, bigint>): DataView {
  const v = new DataView(emu!.memory.buffer);
  v.setUint32(PC + GUEST_MEM_BASE, word, true);
  v.setUint32(PC + 4 + GUEST_MEM_BASE, RET, true);
  for (let i = 0; i < 32; i++) { v.setBigUint64(V_OFFSET + i * 16, 0n, true); v.setBigUint64(V_OFFSET + i * 16 + 8, 0n, true); }
  if (gprs) for (const [r, val] of Object.entries(gprs)) v.setBigUint64(Number(r) * 8, BigInt.asUintN(64, val), true);
  v.setBigUint64(0x100, BigInt(PC), true);
  v.setBigUint64(0xf8, 0x700000n, true);
  v.setBigUint64(30 * 8, BigInt(PC + 0x800), true);
  for (const [reg, lanes] of Object.entries(vregs)) {
    const base = V_OFFSET + Number(reg) * 16;
    lanes.forEach((val, i) => {
      if (asFloat === 'f32') v.setFloat32(base + i * 4, val, true);
      else if (asFloat === 'f64') v.setFloat64(base + i * 8, val, true);
      else if (asFloat === 'i8') v.setInt8(base + i, val);
      else if (asFloat === 'i16') v.setInt16(base + i * 2, val, true);
      else if (asFloat === 'i32') v.setInt32(base + i * 4, val, true);
      else v.setBigInt64(base + i * 8, BigInt(val), true);
    });
  }
  (emu as any).regionCache.clear();
  (emu as any).getRegion(BigInt(PC)).run();
  return new DataView(emu!.memory.buffer);
}

function f32lanes(v: DataView, reg: number, n: number): number[] {
  const base = V_OFFSET + reg * 16;
  return Array.from({ length: n }, (_, i) => v.getFloat32(base + i * 4, true));
}
function f64lanes(v: DataView, reg: number, n: number): number[] {
  const base = V_OFFSET + reg * 16;
  return Array.from({ length: n }, (_, i) => v.getFloat64(base + i * 8, true));
}

describe('NEON vector FP (native v128)', () => {
  it('FADD .4s', () => {
    if (!emu) return;
    // FADD V0.4s, V1.4s, V2.4s = 0x4e22d420
    const v = run(0x4e22d420, { 1: [1, 2, 3, 4], 2: [10, 20, 30, 40] });
    expect(f32lanes(v, 0, 4)).toEqual([11, 22, 33, 44]);
  });

  it('FSUB .4s', () => {
    if (!emu) return;
    // FSUB V0.4s, V1.4s, V2.4s = 0x4ea2d420
    const v = run(0x4ea2d420, { 1: [10, 20, 30, 40], 2: [1, 2, 3, 4] });
    expect(f32lanes(v, 0, 4)).toEqual([9, 18, 27, 36]);
  });

  it('FMUL .4s', () => {
    if (!emu) return;
    // FMUL V0.4s, V1.4s, V2.4s = 0x6e22dc20
    const v = run(0x6e22dc20, { 1: [2, 3, 4, 5], 2: [10, 10, 10, 10] });
    expect(f32lanes(v, 0, 4)).toEqual([20, 30, 40, 50]);
  });

  it('FDIV .4s', () => {
    if (!emu) return;
    // FDIV V0.4s, V1.4s, V2.4s = 0x6e22fc20
    const v = run(0x6e22fc20, { 1: [20, 30, 40, 50], 2: [10, 10, 10, 10] });
    expect(f32lanes(v, 0, 4)).toEqual([2, 3, 4, 5]);
  });

  it('FADD .2d', () => {
    if (!emu) return;
    // FADD V0.2d, V1.2d, V2.2d = 0x4e62d420
    const v = run(0x4e62d420, { 1: [1.5, 2.5], 2: [10.5, 20.5] }, 'f64');
    expect(f64lanes(v, 0, 2)).toEqual([12, 23]);
  });

  it('FADD .2s clears upper 64 bits', () => {
    if (!emu) return;
    // FADD V0.2s, V1.2s, V2.2s = 0x0e22d420 (Q=0)
    const v = run(0x0e22d420, { 1: [1, 2, 99, 99], 2: [10, 20, 88, 88] });
    expect(f32lanes(v, 0, 4)).toEqual([11, 22, 0, 0]);
  });

  it('FMAX / FMIN .4s', () => {
    if (!emu) return;
    // FMAX V0.4s,V1,V2 = 0x4e22f420 ; FMIN = 0x4ea2f420
    let v = run(0x4e22f420, { 1: [1, 5, 3, 8], 2: [4, 2, 6, 7] });
    expect(f32lanes(v, 0, 4)).toEqual([4, 5, 6, 8]);
    v = run(0x4ea2f420, { 1: [1, 5, 3, 8], 2: [4, 2, 6, 7] });
    expect(f32lanes(v, 0, 4)).toEqual([1, 2, 3, 7]);
  });

  it('FMLA .4s accumulates', () => {
    if (!emu) return;
    // FMLA V0.4s, V1.4s, V2.4s = 0x4e22cc20 ; V0 = V0 + V1*V2
    const v = run(0x4e22cc20, { 0: [100, 100, 100, 100], 1: [2, 3, 4, 5], 2: [10, 10, 10, 10] });
    expect(f32lanes(v, 0, 4)).toEqual([120, 130, 140, 150]);
  });

  it('FNEG / FABS .4s', () => {
    if (!emu) return;
    // FNEG V0.4s, V1.4s = 0x6ea0f820 ; FABS = 0x4ea0f820
    let v = run(0x6ea0f820, { 1: [1, -2, 3, -4] });
    expect(f32lanes(v, 0, 4)).toEqual([-1, 2, -3, 4]);
    v = run(0x4ea0f820, { 1: [1, -2, 3, -4] });
    expect(f32lanes(v, 0, 4)).toEqual([1, 2, 3, 4]);
  });

  it('FSQRT .4s', () => {
    if (!emu) return;
    // FSQRT V0.4s, V1.4s = 0x6ea1f820
    const v = run(0x6ea1f820, { 1: [4, 9, 16, 25] });
    expect(f32lanes(v, 0, 4)).toEqual([2, 3, 4, 5]);
  });

  it('FCMEQ .4s produces all-ones lanes', () => {
    if (!emu) return;
    // FCMEQ V0.4s, V1.4s, V2.4s = 0x4e22e420
    const v = run(0x4e22e420, { 1: [1, 2, 3, 4], 2: [1, 9, 3, 9] });
    const base = V_OFFSET + 0;
    expect(v.getUint32(base, true)).toBe(0xffffffff);      // 1==1
    expect(v.getUint32(base + 4, true)).toBe(0);           // 2!=9
    expect(v.getUint32(base + 8, true)).toBe(0xffffffff);  // 3==3
    expect(v.getUint32(base + 12, true)).toBe(0);          // 4!=9
  });
});

function i32lanes(v: DataView, reg: number, n: number): number[] {
  const base = V_OFFSET + reg * 16;
  return Array.from({ length: n }, (_, i) => v.getInt32(base + i * 4, true));
}
function i8lanes(v: DataView, reg: number, n: number): number[] {
  const base = V_OFFSET + reg * 16;
  return Array.from({ length: n }, (_, i) => v.getInt8(base + i));
}

describe('NEON vector integer (native v128 + helper fallback)', () => {
  it('ADD .4s', () => {
    if (!emu) return;
    // ADD V0.4s, V1.4s, V2.4s = 0x4ea28420
    const v = run(0x4ea28420, { 1: [1, 2, 3, 4], 2: [10, 20, 30, 40] }, 'i32');
    expect(i32lanes(v, 0, 4)).toEqual([11, 22, 33, 44]);
  });

  it('SUB .16b', () => {
    if (!emu) return;
    // SUB V0.16b, V1.16b, V2.16b = 0x6e228420
    const v = run(0x6e228420, { 1: [50, 40, 30, 20], 2: [1, 2, 3, 4] }, 'i8');
    expect(i8lanes(v, 0, 4)).toEqual([49, 38, 27, 16]);
  });

  it('MUL .4s', () => {
    if (!emu) return;
    // MUL V0.4s, V1.4s, V2.4s = 0x4ea29c20
    const v = run(0x4ea29c20, { 1: [2, 3, 4, 5], 2: [10, 10, 10, 10] }, 'i32');
    expect(i32lanes(v, 0, 4)).toEqual([20, 30, 40, 50]);
  });

  it('MUL .16b (i8 via helper)', () => {
    if (!emu) return;
    // MUL V0.16b, V1.16b, V2.16b = 0x0e229c20 (Q=0)
    const v = run(0x0e229c20, { 1: [3, 5, 7, 9], 2: [2, 2, 2, 2] }, 'i8');
    expect(i8lanes(v, 0, 4)).toEqual([6, 10, 14, 18]);
  });

  it('SMAX / SMIN / UMAX / UMIN .4s', () => {
    if (!emu) return;
    // SMAX V0.4s,V1,V2 = 0x4ea26420 ; SMIN = 0x4ea26c20 ; UMAX = 0x6ea26420 ; UMIN = 0x6ea26c20
    let v = run(0x4ea26420, { 1: [1, -5, 3, 8], 2: [4, 2, -6, 7] }, 'i32');
    expect(i32lanes(v, 0, 4)).toEqual([4, 2, 3, 8]);
    v = run(0x4ea26c20, { 1: [1, -5, 3, 8], 2: [4, 2, -6, 7] }, 'i32');
    expect(i32lanes(v, 0, 4)).toEqual([1, -5, -6, 7]);
  });

  it('CMGT / CMGE .4s (signed) all-ones lanes', () => {
    if (!emu) return;
    // CMGT V0.4s,V1,V2 = 0x4ea23420
    const v = run(0x4ea23420, { 1: [5, 2, -3, 8], 2: [1, 9, -3, 7] }, 'i32');
    const base = V_OFFSET + 0;
    expect(v.getUint32(base, true)).toBe(0xffffffff);      // 5>1
    expect(v.getUint32(base + 4, true)).toBe(0);           // 2>9 no
    expect(v.getUint32(base + 8, true)).toBe(0);           // -3>-3 no
    expect(v.getUint32(base + 12, true)).toBe(0xffffffff); // 8>7
  });

  it('SQADD .16b saturates (signed)', () => {
    if (!emu) return;
    // SQADD V0.16b, V1.16b, V2.16b = 0x0e220c20 (Q=0). 100+50 -> 127 (sat)
    const v = run(0x0e220c20, { 1: [100, -100, 10, 0], 2: [50, -50, 5, 0] }, 'i8');
    expect(i8lanes(v, 0, 4)).toEqual([127, -128, 15, 0]);
  });

  it('UQADD .16b saturates (unsigned)', () => {
    if (!emu) return;
    // UQADD V0.16b, V1.16b, V2.16b = 0x2e220c20 (Q=0). 200+100 -> 255 (sat)
    const v = run(0x2e220c20, { 1: [200, 10, 100, 0], 2: [100, 5, 50, 0] }, 'i8');
    const base = V_OFFSET + 0;
    expect(v.getUint8(base)).toBe(255);   // 200+100 sat
    expect(v.getUint8(base + 1)).toBe(15);
    expect(v.getUint8(base + 2)).toBe(150);
  });

  it('NEG / ABS .4s', () => {
    if (!emu) return;
    // NEG V0.4s, V1.4s = 0x6ea0b820 ; ABS = 0x4ea0b820
    let v = run(0x6ea0b820, { 1: [1, -2, 3, -4] }, 'i32');
    expect(i32lanes(v, 0, 4)).toEqual([-1, 2, -3, 4]);
    v = run(0x4ea0b820, { 1: [1, -2, 3, -4] }, 'i32');
    expect(i32lanes(v, 0, 4)).toEqual([1, 2, 3, 4]);
  });
});

describe('NEON widening + reductions', () => {
  it('SMULL .4s = V1.4h * V2.4h', () => {
    if (!emu) return;
    // SMULL V0.4s, V1.4h, V2.4h = 0x0e62c020 (size=01, Q=0 low half)
    const v = run(0x0e62c020, { 1: [3, -4, 5, 6, 0, 0, 0, 0], 2: [10, 10, 10, 10, 0, 0, 0, 0] }, 'i16');
    expect(i32lanes(v, 0, 4)).toEqual([30, -40, 50, 60]);
  });

  it('UMULL .8h = V1.8b * V2.8b', () => {
    if (!emu) return;
    // UMULL V0.8h, V1.8b, V2.8b = 0x2e22c020 (size=00, Q=0 low half)
    const v = run(0x2e22c020, { 1: [3, 4, 5, 6, 0, 0, 0, 0], 2: [10, 10, 10, 10, 0, 0, 0, 0] }, 'i8');
    const base = V_OFFSET + 0;
    expect(v.getUint16(base, true)).toBe(30);
    expect(v.getUint16(base + 2, true)).toBe(40);
    expect(v.getUint16(base + 4, true)).toBe(50);
  });

  it('SADDL .4s = V1.4h + V2.4h widened', () => {
    if (!emu) return;
    // SADDL V0.4s, V1.4h, V2.4h = 0x0e620020
    const v = run(0x0e620020, { 1: [100, -200, 300, 400, 0, 0, 0, 0], 2: [50, 50, 50, 50, 0, 0, 0, 0] }, 'i16');
    expect(i32lanes(v, 0, 4)).toEqual([150, -150, 350, 450]);
  });

  it('ADDV .4s sums all lanes', () => {
    if (!emu) return;
    // ADDV S0, V1.4s = 0x4eb1b820
    const v = run(0x4eb1b820, { 1: [1, 2, 3, 4] }, 'i32');
    expect(v.getInt32(V_OFFSET, true)).toBe(10);
  });

  it('UMAXV / UMINV .4s', () => {
    if (!emu) return;
    // UMAXV S0, V1.4s = 0x6eb0a820 ; UMINV = 0x6eb1a820
    let v = run(0x6eb0a820, { 1: [5, 20, 3, 8] }, 'i32');
    expect(v.getUint32(V_OFFSET, true)).toBe(20);
    v = run(0x6eb1a820, { 1: [5, 20, 3, 8] }, 'i32');
    expect(v.getUint32(V_OFFSET, true)).toBe(3);
  });

  it('UADDLV .8h widens the sum', () => {
    if (!emu) return;
    // UADDLV H0, V1.8b = 0x2e303820 (size=00)
    const v = run(0x2e303820, { 1: [100, 100, 100, 100, 100, 100, 0, 0] }, 'i8');
    expect(v.getUint16(V_OFFSET, true)).toBe(600);
  });
});

// structured load/store need a base register + guest memory; use a helper that
// stages memory bytes and points X0 at them
function runMem(word: number, memBytes: number[], addr = 0x8000, x0 = 0x8000): DataView {
  const v = new DataView(emu!.memory.buffer);
  v.setUint32(PC + GUEST_MEM_BASE, word, true);
  v.setUint32(PC + 4 + GUEST_MEM_BASE, RET, true);
  for (let i = 0; i < 32; i++) { v.setBigUint64(V_OFFSET + i * 16, 0n, true); v.setBigUint64(V_OFFSET + i * 16 + 8, 0n, true); }
  for (let i = 0; i < 31; i++) v.setBigUint64(i * 8, 0n, true);
  v.setBigUint64(0 * 8, BigInt(x0), true);
  v.setBigUint64(0x100, BigInt(PC), true);
  v.setBigUint64(0xf8, 0x700000n, true);
  v.setBigUint64(30 * 8, BigInt(PC + 0x800), true);
  memBytes.forEach((b, i) => v.setUint8(addr + GUEST_MEM_BASE + i, b));
  (emu as any).regionCache.clear();
  (emu as any).getRegion(BigInt(PC)).run();
  return new DataView(emu!.memory.buffer);
}

describe('NEON structured load/store', () => {
  it('LD1 {V0.16b} contiguous', () => {
    if (!emu) return;
    // LD1 {V0.16b}, [X0] = 0x4c407000
    const bytes = Array.from({ length: 16 }, (_, i) => i + 1);
    const v = runMem(0x4c407000, bytes);
    for (let i = 0; i < 16; i++) expect(v.getUint8(V_OFFSET + i)).toBe(i + 1);
  });

  it('LD1 {V0.16b-V3.16b} loads 4 consecutive regs', () => {
    if (!emu) return;
    // LD1 {V0.16b, V1.16b, V2.16b, V3.16b}, [X0] = 0x4c402000
    const bytes = Array.from({ length: 64 }, (_, i) => i & 0xff);
    const v = runMem(0x4c402000, bytes);
    expect(v.getUint8(V_OFFSET + 0)).toBe(0);
    expect(v.getUint8(V_OFFSET + 16)).toBe(16);   // V1[0]
    expect(v.getUint8(V_OFFSET + 32)).toBe(32);   // V2[0]
    expect(v.getUint8(V_OFFSET + 48 + 15)).toBe(63); // V3[15]
  });

  it('ST1 {V0.16b} writes back to memory', () => {
    if (!emu) return;
    // first LD1 into V0, then ST1 V0 to a different addr — simplest: set V0 then ST1
    const v = new DataView(emu!.memory.buffer);
    // ST1 {V0.16b}, [X0] = 0x4c007000
    for (let i = 0; i < 32; i++) { v.setBigUint64(V_OFFSET + i * 16, 0n, true); v.setBigUint64(V_OFFSET + i * 16 + 8, 0n, true); }
    for (let i = 0; i < 16; i++) v.setUint8(V_OFFSET + i, i + 100);
    v.setUint32(PC + GUEST_MEM_BASE, 0x4c007000, true);
    v.setUint32(PC + 4 + GUEST_MEM_BASE, RET, true);
    for (let i = 0; i < 31; i++) v.setBigUint64(i * 8, 0n, true);
    v.setBigUint64(0, 0x9000n, true); // X0
    v.setBigUint64(0x100, BigInt(PC), true);
    v.setBigUint64(0xf8, 0x700000n, true);
    v.setBigUint64(30 * 8, BigInt(PC + 0x800), true);
    (emu as any).regionCache.clear();
    (emu as any).getRegion(BigInt(PC)).run();
    const v2 = new DataView(emu!.memory.buffer);
    for (let i = 0; i < 16; i++) expect(v2.getUint8(0x9000 + GUEST_MEM_BASE + i)).toBe(i + 100);
  });

  it('LD1 post-index advances X0', () => {
    if (!emu) return;
    // LD1 {V0.16b}, [X0], #16 = 0x4cdf7000 (Rm=31 → immediate = 16)
    const bytes = Array.from({ length: 16 }, (_, i) => i + 1);
    const v = runMem(0x4cdf7000, bytes, 0x8000, 0x8000);
    expect(v.getBigUint64(0, true)).toBe(0x8010n); // X0 += 16
  });

  it('LD2 {V0.4s, V1.4s} de-interleaves', () => {
    if (!emu) return;
    // LD2 {V0.4s, V1.4s}, [X0] = 0x4c408800 (size=10)
    // memory: interleaved [a0,b0,a1,b1,a2,b2,a3,b3] as 32-bit words
    const mem: number[] = [];
    const words = [10, 20, 11, 21, 12, 22, 13, 23]; // a0 b0 a1 b1 ...
    for (const w of words) { mem.push(w & 0xff, 0, 0, 0); }
    const v = runMem(0x4c408800, mem);
    // V0 = [10,11,12,13], V1 = [20,21,22,23]
    expect([0, 1, 2, 3].map(i => v.getInt32(V_OFFSET + i * 4, true))).toEqual([10, 11, 12, 13]);
    expect([0, 1, 2, 3].map(i => v.getInt32(V_OFFSET + 16 + i * 4, true))).toEqual([20, 21, 22, 23]);
  });

  it('LD1R replicates one element', () => {
    if (!emu) return;
    // LD1R {V0.4s}, [X0] = 0x4d40c800 (single-structure group, size=10)
    const mem = [0x2a, 0, 0, 0]; // 42
    const v = runMem(0x4d40c800, mem);
    expect([0, 1, 2, 3].map(i => v.getInt32(V_OFFSET + i * 4, true))).toEqual([42, 42, 42, 42]);
  });
});

// scalar advanced-simd + fixed-point converts that musl float formatting relies on.
// these were the chain of decode gaps behind the dd / printf-float spins.
describe('scalar SIMD + fixed-point converts', () => {
  const i64lane0 = (v: DataView, reg: number) => v.getBigInt64(V_OFFSET + reg * 16, true);
  const f64lane0 = (v: DataView, reg: number) => v.getFloat64(V_OFFSET + reg * 16, true);

  it('UCVTF d0, d0 (unsigned int in vreg -> double)', () => {
    if (!emu) return;
    // 0x7e61d800. V0 lane0 holds the integer 42.
    const v = run(0x7e61d800, { 0: [42] }, 'i64');
    expect(f64lane0(v, 0)).toBe(42);
    expect(i64lane0(v, 0)).not.toBe(42n); // it was converted, not left raw
  });

  it('SCVTF d0, d0 (signed int in vreg -> double)', () => {
    if (!emu) return;
    // 0x5e61d800 (U=0).
    const v = run(0x5e61d800, { 0: [-5] }, 'i64');
    expect(f64lane0(v, 0)).toBe(-5);
  });

  it('FCVTZS d0, d0 (double -> signed int, truncating)', () => {
    if (!emu) return;
    // 0x5ee1b800. V0 = 3.9 -> 3.
    const v = run(0x5ee1b800, { 0: [3.9] }, 'f64');
    expect(i64lane0(v, 0)).toBe(3n);
  });

  it('SHL d1, d1, #52 (scalar shift-imm, musl scalbn)', () => {
    if (!emu) return;
    // 0x5f745421. 1 << 52.
    const v = run(0x5f745421, { 1: [1] }, 'i64');
    expect(i64lane0(v, 1)).toBe(1n << 52n);
    expect(i64lane0(v, 1) & ((1n << 64n) - 1n)).toBe(4503599627370496n);
  });

  it('USHR d0, d0, #4 (scalar right shift-imm)', () => {
    if (!emu) return;
    // USHR d0,d0,#4: immh:immb = 128-4 = 124 = 0b1111100, opcode 00000, U=1.
    // 01 1 111110 1111 100 00000 1 Rn Rd = 0x7f7c0400 (Rn=Rd=0).
    const v = run(0x7f7c0400, { 0: [0xff0] }, 'i64');
    expect(i64lane0(v, 0)).toBe(0xffn);
  });

  it('SCVTF d0, w0, #2 (fixed-point: int/4 -> double)', () => {
    if (!emu) return;
    // 0x1e42f800. w0 = 10 -> 10/4 = 2.5.
    const v = run(0x1e42f800, {}, 'f64', { 0: 10n });
    expect(f64lane0(v, 0)).toBe(2.5);
  });

  it('FCVTZS w0, d0, #2 (fixed-point: double*4 -> int)', () => {
    if (!emu) return;
    // FCVTZS Wd, Dn, #fbits: sf=0 type=01 rmode=11 opcode=000 scale=62. = 0x1e58f800.
    // V0 = 2.5 -> 2.5*4 = 10.
    const v = run(0x1e58f800, { 0: [2.5] }, 'f64');
    expect(v.getBigUint64(0, true)).toBe(10n); // result in x0
  });
});

describe('NEON new ops: shift-imm / rev / ins-elem / dup-scalar / rbit', () => {
  const ub = (v: DataView, reg: number, i: number) => v.getUint8(V_OFFSET + reg * 16 + i);
  const i16 = (v: DataView, reg: number, i: number) => v.getInt16(V_OFFSET + reg * 16 + i * 2, true);

  it('USHR v0.16b, v1.16b, #3', () => {
    if (!emu) return;
    const v = run(0x6f0d0420, { 1: [64, 32, 16, 8] }, 'i8');
    expect(i8lanes(v, 0, 8)).toEqual([8, 4, 2, 1, 0, 0, 0, 0]);
  });

  it('SSHR v0.8h, v1.8h, #2 (signed)', () => {
    if (!emu) return;
    const v = run(0x4f1e0420, { 1: [-16, 16, -4, 100] }, 'i16');
    expect([i16(v, 0, 0), i16(v, 0, 1), i16(v, 0, 2), i16(v, 0, 3)]).toEqual([-4, 4, -1, 25]);
  });

  it('SHL v0.4s, v1.4s, #4', () => {
    if (!emu) return;
    const v = run(0x4f245420, { 1: [1, 2, 3, -1] }, 'i32');
    expect(i32lanes(v, 0, 4)).toEqual([16, 32, 48, -16]);
  });

  it('SLI v0.16b, v1.16b, #3', () => {
    if (!emu) return;
    // result[i] = (Vn[i] << 3) | (Vd[i] & 0b111)
    const v = run(0x6f0b5420, { 1: [0x0a, 0x01], 0: [0x05, 0x02] }, 'i8');
    expect([ub(v, 0, 0), ub(v, 0, 1)]).toEqual([0x55, 0x0a]);
  });

  it('SRI v0.16b, v1.16b, #4', () => {
    if (!emu) return;
    // result[i] = (Vn[i] >> 4) | (Vd[i] & 0xf0)
    const v = run(0x6f0c4420, { 1: [0x70, 0x20], 0: [0x30, 0x10] }, 'i8');
    expect([ub(v, 0, 0), ub(v, 0, 1)]).toEqual([0x37, 0x12]);
  });

  it('REV16 v0.16b, v1.16b', () => {
    if (!emu) return;
    const v = run(0x4e201820, { 1: [10, 11, 20, 21, 30, 31, 40, 41] }, 'i8');
    expect(i8lanes(v, 0, 8)).toEqual([11, 10, 21, 20, 31, 30, 41, 40]);
  });

  it('REV32 v0.16b, v1.16b', () => {
    if (!emu) return;
    const v = run(0x6e200820, { 1: [1, 2, 3, 4, 5, 6, 7, 8] }, 'i8');
    expect(i8lanes(v, 0, 8)).toEqual([4, 3, 2, 1, 8, 7, 6, 5]);
  });

  it('REV64 v0.16b, v1.16b', () => {
    if (!emu) return;
    const v = run(0x4e200820, { 1: [1, 2, 3, 4, 5, 6, 7, 8] }, 'i8');
    expect(i8lanes(v, 0, 8)).toEqual([8, 7, 6, 5, 4, 3, 2, 1]);
  });

  it('REV64 v0.4s, v1.4s (32-bit elems)', () => {
    if (!emu) return;
    const v = run(0x4ea00820, { 1: [1, 2, 3, 4] }, 'i32');
    expect(i32lanes(v, 0, 4)).toEqual([2, 1, 4, 3]);
  });

  it('RBIT v0.16b, v1.16b', () => {
    if (!emu) return;
    const v = run(0x6e605820, { 1: [0x01, 0x02, 0x03, 0x80] }, 'i8');
    expect([ub(v, 0, 0), ub(v, 0, 1), ub(v, 0, 2), ub(v, 0, 3)]).toEqual([0x80, 0x40, 0xc0, 0x01]);
  });

  it('INS v0.b[3], v1.b[5] (element move preserves rest)', () => {
    if (!emu) return;
    const v = run(0x6e072c20, { 1: [0, 0, 0, 0, 0, 99], 0: [1, 0, 0, 7] }, 'i8');
    expect(ub(v, 0, 3)).toBe(99);
    expect(ub(v, 0, 0)).toBe(1);
  });

  it('INS v0.s[1], v1.s[3] (32-bit element move)', () => {
    if (!emu) return;
    const v = run(0x6e0c6420, { 1: [0, 0, 0, 287454020], 0: [5, 0] }, 'i32');
    expect(i32lanes(v, 0, 2)).toEqual([5, 287454020]);
  });

  it('DUP d0, v1.d[1] (scalar dup 64-bit)', () => {
    if (!emu) return;
    const v = run(0x5e180420, { 1: [0n, 0xdeadbeefn] } as any, 'i64');
    expect(v.getBigUint64(V_OFFSET + 0, true)).toBe(0xdeadbeefn);
    expect(v.getBigUint64(V_OFFSET + 8, true)).toBe(0n);
  });

  it('DUP s0, v1.s[2] (scalar dup 32-bit, zero-extended)', () => {
    if (!emu) return;
    const v = run(0x5e140420, { 1: [0, 0, 0x1234, 0] }, 'i32');
    expect(v.getBigUint64(V_OFFSET + 0, true)).toBe(0x1234n);
    expect(v.getBigUint64(V_OFFSET + 8, true)).toBe(0n);
  });
});

describe('NEON multiply by indexed element (poly1305 path)', () => {
  const dl = (v: DataView, reg: number, i: number) => v.getBigUint64(V_OFFSET + reg * 16 + i * 8, true);

  it('UMULL v0.2d, v1.2s, v2.s[0]', () => {
    if (!emu) return;
    const v = run(0x2f82a020, { 1: [10, 20], 2: [3] }, 'i32');
    expect([dl(v, 0, 0), dl(v, 0, 1)]).toEqual([30n, 60n]);
  });

  it('UMULL v0.2d, v1.2s, v2.s[2] (index 2)', () => {
    if (!emu) return;
    const v = run(0x2f82a820, { 1: [10, 20], 2: [0, 0, 5, 0] }, 'i32');
    expect([dl(v, 0, 0), dl(v, 0, 1)]).toEqual([50n, 100n]);
  });

  it('UMLAL v0.2d, v1.2s, v2.s[1] (accumulate)', () => {
    if (!emu) return;
    const v = run(0x2fa22020, { 1: [10, 20], 2: [0, 3], 0: [100, 0, 200, 0] }, 'i32');
    // preset v0.2d = [100, 200] via i32 lanes [100,0,200,0]
    expect([dl(v, 0, 0), dl(v, 0, 1)]).toEqual([130n, 260n]);
  });

  it('SMULL v0.4s, v1.4h, v2.h[3] (signed widening)', () => {
    if (!emu) return;
    const v = run(0x0f72a020, { 1: [-2, 3, -4, 5], 2: [0, 0, 0, 7] }, 'i16');
    expect(i32lanes(v, 0, 4)).toEqual([-14, 21, -28, 35]);
  });

  it('MUL v0.4s, v1.4s, v2.s[1] (same-width)', () => {
    if (!emu) return;
    const v = run(0x4fa28020, { 1: [1, 2, 3, 4], 2: [0, 5] }, 'i32');
    expect(i32lanes(v, 0, 4)).toEqual([5, 10, 15, 20]);
  });

  it('MLA v0.4s, v1.4s, v2.s[0] (accumulate same-width)', () => {
    if (!emu) return;
    const v = run(0x6f820020, { 1: [2, 3, 4, 5], 2: [10], 0: [1, 1, 1, 1] }, 'i32');
    expect(i32lanes(v, 0, 4)).toEqual([21, 31, 41, 51]);
  });

  it('UMULL2 v0.2d, v1.4s, v2.s[3] (high half)', () => {
    if (!emu) return;
    const v = run(0x6fa2a820, { 1: [0, 0, 10, 20], 2: [0, 0, 0, 4] }, 'i32');
    expect([dl(v, 0, 0), dl(v, 0, 1)]).toEqual([40n, 80n]);
  });
});

describe('NEON XTN (narrow, poly1305 reduce)', () => {
  const dset = (v: DataView, reg: number, lanes: bigint[]) => lanes.forEach((x, i) => v.setBigUint64(V_OFFSET + reg * 16 + i * 8, x, true));
  it('XTN v0.2s, v1.2d', () => {
    if (!emu) return;
    const v = run(0x0ea12820, {}, 'i64');
    dset(v, 1, [0x1122334455667788n, 0x99aabbccddeeff00n]);
    v.setBigUint64(0x100, BigInt(0x600000), true);
    (emu as any).regionCache.clear();
    (emu as any).getRegion(BigInt(0x600000)).run();
    expect(i32lanes(v, 0, 4)).toEqual([0x55667788 | 0, 0xddeeff00 | 0, 0, 0]);
  });
  it('XTN v0.4h, v1.4s', () => {
    if (!emu) return;
    const v = run(0x0e612820, { 1: [0x11112222, 0x33334444, 0x55556666, 0x77778888] }, 'i32');
    // low 16 bits of each: 0x2222,0x4444,0x6666,0x8888
    expect([v.getUint16(V_OFFSET, true), v.getUint16(V_OFFSET + 2, true), v.getUint16(V_OFFSET + 4, true), v.getUint16(V_OFFSET + 6, true)]).toEqual([0x2222, 0x4444, 0x6666, 0x8888]);
    expect(v.getBigUint64(V_OFFSET + 8, true)).toBe(0n);
  });
});

describe('crypto-ext AES + PMULL', () => {
  it('AESE v0.16b, v1.16b with zero state+key -> all 0x63', () => {
    if (!emu) return;
    // AESE: SubBytes(state^key); state=key=0 -> SubBytes(0)=0x63 for every byte, ShiftRows keeps it.
    const v = run(0x4e284820, {}, 'i8');
    expect(i8lanes(v, 0, 16).map((x) => x & 0xff)).toEqual(new Array(16).fill(0x63));
  });
  it('AESD v0.16b, v1.16b with zero -> all InvSbox[0]=0x52', () => {
    if (!emu) return;
    const v = run(0x4e285820, {}, 'i8');
    expect((v.getUint8(V_OFFSET) )).toBe(0x52);
  });
  it('PMULL v0.1q, v1.1d, v2.1d (2 x 3 carryless = 6)', () => {
    if (!emu) return;
    const v = run(0x0ee2e020, { 1: [2], 2: [3] }, 'i64');
    expect(v.getBigUint64(V_OFFSET, true)).toBe(6n);
    expect(v.getBigUint64(V_OFFSET + 8, true)).toBe(0n);
  });
  it('PMULL carry into high word ((1<<62) x 2 = 1<<63)', () => {
    if (!emu) return;
    const v = run(0x0ee2e020, { 1: [0x4000000000000000], 2: [2] }, 'i64');
    expect(v.getBigUint64(V_OFFSET, true)).toBe(0x8000000000000000n);
    expect(v.getBigUint64(V_OFFSET + 8, true)).toBe(0n);
  });
  it('PMULL2 uses the high 64-bit lanes', () => {
    if (!emu) return;
    // v1.d[1]=2, v2.d[1]=3 -> pmull2 -> v0.d[0]=6
    const v = run(0x4ee2e020, { 1: [0, 2], 2: [0, 3] }, 'i64');
    expect(v.getBigUint64(V_OFFSET, true)).toBe(6n);
  });
});

import { sha1_3reg, sha1h as _sha1h, sha1su0 as _sha1su0, sha1su1 as _sha1su1,
         sha256h as _sha256h, sha256su0 as _sha256su0, sha256su1 as _sha256su1,
         sha512h as _sha512h, sha512h2 as _sha512h2, sha512su0 as _sha512su0, sha512su1 as _sha512su1 }
  from '../src/crypto-ext.js';

describe('crypto-ext SHA1 / SHA256 / SHA512', () => {
  // fill v0..v2 with distinct non-trivial bytes, run the encoded insn, and compare v0 against the
  // (independently NIST-validated) helper applied to the same inputs. this checks decode + dispatch.
  const seed = (n: number, base: number) => Array.from({ length: 16 }, (_, i) => ((n * 37 + i * 53 + base) & 0xff));
  function check(word: number, apply: (buf: Uint8Array, d: number, n: number, m: number) => void) {
    if (!emu) return;
    const V0 = seed(1, 3), V1 = seed(2, 7), V2 = seed(3, 11);
    const v = run(word, { 0: V0, 1: V1, 2: V2 }, 'i8');
    const scratch = new Uint8Array(48);
    scratch.set(V0, 0); scratch.set(V1, 16); scratch.set(V2, 32);
    apply(scratch, 0, 16, 32);
    const got = Array.from({ length: 16 }, (_, i) => v.getUint8(V_OFFSET + i));
    expect(got).toEqual(Array.from(scratch.subarray(0, 16)));
  }
  it('SHA1C', () => check(0x5e020020, (b, d, n, m) => sha1_3reg(b, d, n, m, 0)));
  it('SHA1P', () => check(0x5e021020, (b, d, n, m) => sha1_3reg(b, d, n, m, 1)));
  it('SHA1M', () => check(0x5e022020, (b, d, n, m) => sha1_3reg(b, d, n, m, 2)));
  it('SHA1H', () => check(0x5e280820, (b, d, n) => _sha1h(b, d, n)));
  it('SHA1SU0', () => check(0x5e023020, (b, d, n, m) => _sha1su0(b, d, n, m)));
  it('SHA1SU1', () => check(0x5e281820, (b, d, n) => _sha1su1(b, d, n)));
  it('SHA256H', () => check(0x5e024020, (b, d, n, m) => _sha256h(b, d, n, m, false)));
  it('SHA256H2', () => check(0x5e025020, (b, d, n, m) => _sha256h(b, d, n, m, true)));
  it('SHA256SU0', () => check(0x5e282820, (b, d, n) => _sha256su0(b, d, n)));
  it('SHA256SU1', () => check(0x5e026020, (b, d, n, m) => _sha256su1(b, d, n, m)));
  it('SHA512H', () => check(0xce628020, (b, d, n, m) => _sha512h(b, d, n, m)));
  it('SHA512H2', () => check(0xce628420, (b, d, n, m) => _sha512h2(b, d, n, m)));
  it('SHA512SU0', () => check(0xcec08020, (b, d, n) => _sha512su0(b, d, n)));
  it('SHA512SU1', () => check(0xce628820, (b, d, n, m) => _sha512su1(b, d, n, m)));
});

describe('NEON single-lane LD1/ST1', () => {
  it('LD1 {v1.s}[0], [x0]', () => {
    if (!emu) return;
    const v = runMem(0x0d408001, [0x11, 0x22, 0x33, 0x44, 0, 0, 0, 0]);
    expect(v.getUint32(V_OFFSET + 16, true)).toBe(0x44332211);
    expect(v.getUint32(V_OFFSET + 16 + 4, true)).toBe(0);
  });
  it('LD1 {v0.d}[1], [x0]', () => {
    if (!emu) return;
    const v = runMem(0x4d408400, [1, 2, 3, 4, 5, 6, 7, 8]);
    expect(v.getBigUint64(V_OFFSET + 8, true)).toBe(0x0807060504030201n);
    expect(v.getBigUint64(V_OFFSET, true)).toBe(0n);
  });
  it('LD1 {v0.b}[3], [x0]', () => {
    if (!emu) return;
    const v = runMem(0x0d400c00, [0xab]);
    expect(v.getUint8(V_OFFSET + 3)).toBe(0xab);
  });
});

// the scalar compare-against-zero group. luajit's number->integer path emits `cmge d,d,#0`,
// and hitting it as an undefined instruction is what stopped neovim from starting.
// every encoding below came out of llvm-mc rather than being assembled by hand.
describe('scalar SIMD compare-against-zero', () => {
  const lo = (v: DataView, reg: number) => v.getBigUint64(V_OFFSET + reg * 16, true);
  const hi = (v: DataView, reg: number) => v.getBigUint64(V_OFFSET + reg * 16 + 8, true);
  const ONES = 0xffffffffffffffffn;

  // insn, word, rd, then the expected lane for inputs 7, 0 and -3
  const intCases: Array<[string, number, number, [bigint, bigint, bigint]]> = [
    ['CMGE d0, d1, #0', 0x7ee08820, 0, [ONES, ONES, 0n]],
    ['CMGT d2, d1, #0', 0x5ee08822, 2, [ONES, 0n, 0n]],
    ['CMEQ d3, d1, #0', 0x5ee09823, 3, [0n, ONES, 0n]],
    ['CMLE d4, d1, #0', 0x7ee09824, 4, [0n, ONES, ONES]],
    ['CMLT d5, d1, #0', 0x5ee0a825, 5, [0n, 0n, ONES]],
  ];

  for (const [name, word, rd, [pos, zero, neg]] of intCases) {
    it(name, () => {
      if (!emu) return;
      expect(lo(run(word, { 1: [7] }, 'i64'), rd)).toBe(pos);
      expect(lo(run(word, { 1: [0] }, 'i64'), rd)).toBe(zero);
      expect(lo(run(word, { 1: [-3] }, 'i64'), rd)).toBe(neg);
    });
  }

  it('only touches the low 64 bits', () => {
    if (!emu) return;
    expect(hi(run(0x7ee08820, { 1: [7] }, 'i64'), 0)).toBe(0n);
  });

  it('ABS d6, d1 / NEG d7, d1', () => {
    if (!emu) return;
    expect(lo(run(0x5ee0b826, { 1: [-9] }, 'i64'), 6)).toBe(9n);
    expect(lo(run(0x7ee0b827, { 1: [-9] }, 'i64'), 7)).toBe(9n);
    expect(lo(run(0x7ee0b827, { 1: [9] }, 'i64'), 7)).toBe(BigInt.asUintN(64, -9n));
  });

  const fpCases: Array<[string, number, number, [bigint, bigint, bigint, bigint]]> = [
    // insn, word, rd, results for 2.5, 0.0, -2.5, NaN
    ['FCMGE d8, d1, #0.0', 0x7ee0c828, 8, [ONES, ONES, 0n, 0n]],
    ['FCMGT d9, d1, #0.0', 0x5ee0c829, 9, [ONES, 0n, 0n, 0n]],
    ['FCMEQ d10, d1, #0.0', 0x5ee0d82a, 10, [0n, ONES, 0n, 0n]],
    ['FCMLE d11, d1, #0.0', 0x7ee0d82b, 11, [0n, ONES, ONES, 0n]],
    ['FCMLT d12, d1, #0.0', 0x5ee0e82c, 12, [0n, 0n, ONES, 0n]],
  ];
  for (const [name, word, rd, exp] of fpCases) {
    it(name, () => {
      if (!emu) return;
      const inputs = [2.5, 0, -2.5, NaN];
      inputs.forEach((x, i) => expect(lo(run(word, { 1: [x] }, 'f64'), rd)).toBe(exp[i]));
    });
  }

  it('the single-precision scalar form clears everything above bit 31', () => {
    if (!emu) return;
    // FCMGE s13, s1, #0.0 = 0x7ea0c82d. lane 1 of v1 is deliberately made to compare true,
    // so a q=0 vector emit would leave 0xffffffff sitting in bits 63:32.
    const v = run(0x7ea0c82d, { 1: [2.5, 4.0, 0, 0] }, 'f32');
    expect(lo(v, 13)).toBe(0xffffffffn);
    expect(hi(v, 13)).toBe(0n);
  });
});

describe('NEON vector compare-against-zero (FP)', () => {
  it('FCMGE v15.4s, v1.4s, #0.0 — was silently emitting FABS', () => {
    if (!emu) return;
    const v = run(0x6ea0c82f, { 1: [1.5, -1.5, 0, -0.5] }, 'f32');
    const base = V_OFFSET + 15 * 16;
    expect([0, 1, 2, 3].map((i) => v.getUint32(base + i * 4, true)))
      .toEqual([0xffffffff, 0, 0xffffffff, 0]);
  });

  it('FCMLT v16.4s, v1.4s, #0.0', () => {
    if (!emu) return;
    const v = run(0x4ea0e830, { 1: [1.5, -1.5, 0, 0] }, 'f32');
    const base = V_OFFSET + 16 * 16;
    expect([0, 1].map((i) => v.getUint32(base + i * 4, true))).toEqual([0, 0xffffffff]);
    expect(v.getBigUint64(base + 8, true)).toBe(0n); // q=0
  });

  it('FCMLE v17.2d, v1.2d, #0.0', () => {
    if (!emu) return;
    const v = run(0x6ee0d831, { 1: [-1.5, 1.5] }, 'f64');
    const base = V_OFFSET + 17 * 16;
    expect(v.getBigUint64(base, true)).toBe(0xffffffffffffffffn);
    expect(v.getBigUint64(base + 8, true)).toBe(0n);
  });
});

describe('CMTST (was silently emitting ADD)', () => {
  it('CMTST v0.16b, v1.16b, v2.16b', () => {
    if (!emu) return;
    const v = run(0x4e228c20, { 1: [0x0f, 0x00, 0x33, 0x01], 2: [0xf0, 0xff, 0x30, 0x02] }, 'i8');
    const base = V_OFFSET;
    // lane0: 0x0f & 0xf0 == 0 -> 0. lane1: 0 & anything == 0 -> 0.
    // lane2: 0x33 & 0x30 != 0 -> 0xff. lane3: 1 & 2 == 0 -> 0.
    expect([0, 1, 2, 3].map((i) => v.getUint8(base + i))).toEqual([0, 0, 0xff, 0]);
  });

  it('CMTST v3.2d, v1.2d, v2.2d', () => {
    if (!emu) return;
    const v = run(0x4ee28c23, { 1: [5, 8], 2: [1, 1] }, 'i64');
    const base = V_OFFSET + 3 * 16;
    expect(v.getBigUint64(base, true)).toBe(0xffffffffffffffffn); // 5 & 1 = 1
    expect(v.getBigUint64(base + 8, true)).toBe(0n);              // 8 & 1 = 0
  });
});

// MLA/MLS were not decoded at all, which is a SIGILL the moment any vectorised
// multiply-accumulate runs (musl's malloc path hits it).
describe('MLA / MLS (vector multiply-accumulate)', () => {
  it('MLA .16b keeps the low byte of each product', () => {
    if (!emu) return;
    const v = run(0x4e229420, { 0: [100, 5, 0, 1], 1: [3, 7, 9, 0x40], 2: [4, 6, 0, 4] }, 'i8');
    // 100+3*4=112, 5+7*6=47, 0, 1 + (0x40*4 = 0x100 -> low byte 0) = 1
    expect(i8lanes(v, 0, 4)).toEqual([112, 47, 0, 1]);
  });

  it('MLS .16b subtracts', () => {
    if (!emu) return;
    const v = run(0x6e229423, { 3: [100, 50], 1: [3, 7], 2: [4, 6] }, 'i8');
    expect(i8lanes(v, 3, 2)).toEqual([88, 8]);
  });

  it('MLA / MLS .8h', () => {
    if (!emu) return;
    let v = run(0x4e629424, { 4: [1000, 2000], 1: [3, 7], 2: [4, 6] }, 'i16');
    expect(v.getInt16(V_OFFSET + 4 * 16, true)).toBe(1012);
    expect(v.getInt16(V_OFFSET + 4 * 16 + 2, true)).toBe(2042);
    v = run(0x6e629425, { 5: [1000, 2000], 1: [3, 7], 2: [4, 6] }, 'i16');
    expect(v.getInt16(V_OFFSET + 5 * 16, true)).toBe(988);
    expect(v.getInt16(V_OFFSET + 5 * 16 + 2, true)).toBe(1958);
  });

  it('MLA / MLS .4s', () => {
    if (!emu) return;
    let v = run(0x4ea29426, { 6: [10, 20, 30, 40], 1: [1, 2, 3, 4], 2: [5, 5, 5, 5] }, 'i32');
    expect(i32lanes(v, 6, 4)).toEqual([15, 30, 45, 60]);
    v = run(0x6ea29427, { 7: [10, 20, 30, 40], 1: [1, 2, 3, 4], 2: [5, 5, 5, 5] }, 'i32');
    expect(i32lanes(v, 7, 4)).toEqual([5, 10, 15, 20]);
  });

  it('MLA .2s clears the upper half', () => {
    if (!emu) return;
    const v = run(0x0ea29428, { 8: [1, 2, 99, 99], 1: [1, 1, 1, 1], 2: [7, 7, 7, 7] }, 'i32');
    expect(i32lanes(v, 8, 4)).toEqual([8, 9, 0, 0]);
  });

  it('MLA .16b with negative accumulator lanes', () => {
    if (!emu) return;
    const v = run(0x4e229420, { 0: [-5, -100], 1: [2, 3], 2: [3, 3] }, 'i8');
    expect(i8lanes(v, 0, 2)).toEqual([1, -91]);
  });
});

// scalar FRINT<mode> and the rounding FCVTs. only the round-toward-zero pair used to decode,
// so math.floor (luajit) and lround (musl) were illegal instructions.
describe('scalar FP rounding', () => {
  const d0 = (v: DataView, reg: number) => v.getFloat64(V_OFFSET + reg * 16, true);
  const x = (v: DataView, reg: number) => v.getBigInt64(reg * 8, true);

  const cases: Array<[string, number, number, Record<string, number>]> = [
    // insn name, word, dest reg, input -> expected
    ['FRINTM (floor)', 0x1e654020, 0, { '2.7': 2, '-2.1': -3, '-0.3': -1, '0.7': 0, '2': 2 }],
    ['FRINTP (ceil)', 0x1e64c022, 2, { '2.1': 3, '-2.7': -2, '-0.3': -0, '2': 2 }],
    ['FRINTZ (trunc)', 0x1e65c023, 3, { '2.7': 2, '-2.7': -2, '-0.3': -0 }],
    ['FRINTA (ties away)', 0x1e664024, 4, { '0.5': 1, '-0.5': -1, '2.5': 3, '1.5': 2, '-2.5': -3 }],
    ['FRINTN (ties even)', 0x1e644025, 5, { '0.5': 0, '-0.5': -0, '2.5': 2, '1.5': 2, '3.5': 4 }],
    ['FRINTX (current mode)', 0x1e674026, 6, { '2.5': 2, '0.5': 0 }],
  ];
  for (const [name, word, rd, table] of cases) {
    it(name, () => {
      if (!emu) return;
      for (const [inp, want] of Object.entries(table)) {
        const got = d0(run(word, { 1: [Number(inp)] }, 'f64'), rd);
        expect(`${inp} -> ${Object.is(got, -0) ? '-0' : got}`).toBe(`${inp} -> ${Object.is(want, -0) ? '-0' : want}`);
      }
    });
  }

  it('FRINTM s7, s1 (single precision)', () => {
    if (!emu) return;
    const v = run(0x1e254027, { 1: [-2.5] }, 'f32');
    expect(v.getFloat32(V_OFFSET + 7 * 16, true)).toBe(-3);
  });

  it('FRINT passes NaN and infinity through', () => {
    if (!emu) return;
    expect(d0(run(0x1e654020, { 1: [Infinity] }, 'f64'), 0)).toBe(Infinity);
    expect(d0(run(0x1e664024, { 1: [-Infinity] }, 'f64'), 4)).toBe(-Infinity);
    expect(Number.isNaN(d0(run(0x1e654020, { 1: [NaN] }, 'f64'), 0))).toBe(true);
  });

  it('FCVTAS / FCVTAU x, d (nearest, ties away)', () => {
    if (!emu) return;
    expect(x(run(0x9e640020, { 1: [2.5] }, 'f64'), 0)).toBe(3n);
    expect(x(run(0x9e640020, { 1: [-2.5] }, 'f64'), 0)).toBe(-3n);
    expect(x(run(0x9e650021, { 1: [2.5] }, 'f64'), 1)).toBe(3n);
  });

  it('FCVTMS / FCVTPS / FCVTNS x, d', () => {
    if (!emu) return;
    expect(x(run(0x9e700022, { 1: [2.7] }, 'f64'), 2)).toBe(2n);   // floor
    expect(x(run(0x9e700022, { 1: [-2.1] }, 'f64'), 2)).toBe(-3n);
    expect(x(run(0x9e680023, { 1: [2.1] }, 'f64'), 3)).toBe(3n);   // ceil
    expect(x(run(0x9e600024, { 1: [2.5] }, 'f64'), 4)).toBe(2n);   // ties even
    expect(x(run(0x9e600024, { 1: [3.5] }, 'f64'), 4)).toBe(4n);
  });

  it('FCVTMS w5, d1 (32-bit destination)', () => {
    if (!emu) return;
    expect(x(run(0x1e700025, { 1: [-2.1] }, 'f64'), 5) & 0xffffffffn).toBe(0xfffffffdn);
  });
});

describe('scalar SIMD three-same (d-form)', () => {
  const lo = (v: DataView, reg: number) => v.getBigUint64(V_OFFSET + reg * 16, true);
  const hi = (v: DataView, reg: number) => v.getBigUint64(V_OFFSET + reg * 16 + 8, true);

  it('ADD d9, d1, d2 / SUB d10, d1, d2', () => {
    if (!emu) return;
    let v = run(0x5ee28429, { 1: [7], 2: [5] }, 'i64');
    expect(lo(v, 9)).toBe(12n);
    expect(hi(v, 9)).toBe(0n);
    v = run(0x7ee2842a, { 1: [7], 2: [5] }, 'i64');
    expect(lo(v, 10)).toBe(2n);
  });

  it('CMGE d11, d1, d2', () => {
    if (!emu) return;
    expect(lo(run(0x5ee23c2b, { 1: [7], 2: [5] }, 'i64'), 11)).toBe(0xffffffffffffffffn);
    expect(lo(run(0x5ee23c2b, { 1: [3], 2: [5] }, 'i64'), 11)).toBe(0n);
  });

  it('UQSUB h12, h1, h2 saturates and zeroes everything above the halfword', () => {
    if (!emu) return;
    const v = run(0x7e622c2c, { 1: [3, 0x7fff, 0x7fff, 0x7fff], 2: [10, 1, 1, 1] }, 'i16');
    expect(v.getUint16(V_OFFSET + 12 * 16, true)).toBe(0); // 3 - 10 saturates to 0
    expect(v.getUint16(V_OFFSET + 12 * 16 + 2, true)).toBe(0); // not a second lane
    expect(hi(v, 12)).toBe(0n);
  });
});

describe('FCCMP and scalar fixed-point converts', () => {
  const nzcv = (v: DataView) => v.getUint32(0x108, true) >>> 28;

  it('FCCMP compares when the condition holds', () => {
    if (!emu) return;
    // set NZCV to Z first via a compare of equals, then FCCMP d1,d2,#0,eq (0x1e620420):
    // eq holds, so the real compare runs and 1 vs 2 gives N set (less than)
    const v = new DataView(emu!.memory.buffer);
    v.setUint32(0x108, 0x4 << 28, true); // Z
    const r = run(0x1e620420, { 1: [1], 2: [2] }, 'f64');
    expect(nzcv(r) & 0x8).toBe(0x8); // N: 1 < 2
  });

  it('FCCMP installs the immediate flags when the condition fails', () => {
    if (!emu) return;
    const v = new DataView(emu!.memory.buffer);
    v.setUint32(0x108, 0x4 << 28, true); // Z set, so `ne` fails
    // FCCMP d1, d2, #6, ne = 0x1e621426
    const r = run(0x1e621426, { 1: [1], 2: [2] }, 'f64');
    expect(nzcv(r)).toBe(6);
  });

  it('UCVTF d0, d1, #10 scales by 2^-10', () => {
    if (!emu) return;
    const v = run(0x7f76e420, { 1: [1024] }, 'i64');
    expect(v.getFloat64(V_OFFSET, true)).toBe(1);
    expect(v.getBigUint64(V_OFFSET + 8, true)).toBe(0n);
  });

  it('SCVTF d2, d1, #4 handles negatives', () => {
    if (!emu) return;
    const v = run(0x5f7ce422, { 1: [-24] }, 'i64');
    expect(v.getFloat64(V_OFFSET + 2 * 16, true)).toBe(-1.5);
  });

  it('FCVTZU d3, d1, #8 scales back up', () => {
    if (!emu) return;
    const v = run(0x7f78fc23, { 1: [2.5] }, 'f64');
    expect(v.getBigUint64(V_OFFSET + 3 * 16, true)).toBe(640n); // 2.5 * 256
  });
});

// the families that a scan of the whole upstream ltp corpus found undecodable
// (tools/scan-decode.ts). every word here is one llvm-objdump printed from a real binary.
describe('decode gaps found by scanning the corpus', () => {
  it('ADDP d0, v0.2d sums the two doublewords', () => {
    if (!emu) return;
    const v = run(0x5ef1b800, { 0: [7, 35] }, 'i64');
    expect(v.getBigInt64(V_OFFSET, true)).toBe(42n);
    expect(v.getBigUint64(V_OFFSET + 8, true)).toBe(0n); // everything above the element is zeroed
  });

  it('FADDP d3, v3.2d adds the pair of doubles', () => {
    if (!emu) return;
    const v = run(0x7e70d863, { 3: [1.5, 2.25] }, 'f64');
    expect(v.getFloat64(V_OFFSET + 3 * 16, true)).toBe(3.75);
    expect(v.getBigUint64(V_OFFSET + 3 * 16 + 8, true)).toBe(0n);
  });

  it('FADDP s5, v5.2s adds the pair of singles', () => {
    if (!emu) return;
    const v = run(0x7e30d8a5, { 5: [1.5, 2.25] }, 'f32');
    expect(v.getFloat32(V_OFFSET + 5 * 16, true)).toBe(3.75);
    expect(v.getBigUint64(V_OFFSET + 5 * 16 + 8, true)).toBe(0n);
  });

  it('USHL v1.2s, v1.2s, v2.2s shifts left on a positive count', () => {
    if (!emu) return;
    // v2 lanes hold the SIGNED shift counts in their low bytes
    const v = run(0x2ea24421, { 1: [1, 0x30], 2: [4, 2] }, 'i32');
    expect(v.getUint32(V_OFFSET + 1 * 16, true)).toBe(16);
    expect(v.getUint32(V_OFFSET + 1 * 16 + 4, true)).toBe(0xc0);
  });

  it('USHL shifts RIGHT on a negative count, and clears past the lane width', () => {
    if (!emu) return;
    const v = run(0x2ea24421, { 1: [0x100, 0x40], 2: [-4, -40] }, 'i32');
    expect(v.getUint32(V_OFFSET + 1 * 16, true)).toBe(0x10);
    expect(v.getUint32(V_OFFSET + 1 * 16 + 4, true)).toBe(0); // a count past 32 zeroes the lane
  });

  it('USRA v18.4s, v17.4s, #31 shifts right and ACCUMULATES', () => {
    if (!emu) return;
    const v = new DataView(emu!.memory.buffer);
    // seed the destination: usra adds into what Vd already holds
    const seed = () => { for (let i = 0; i < 4; i++) v.setUint32(V_OFFSET + 18 * 16 + i * 4, 5, true); };
    v.setUint32(PC + GUEST_MEM_BASE, 0x6f211632, true);
    v.setUint32(PC + 4 + GUEST_MEM_BASE, RET, true);
    for (let i = 0; i < 32; i++) { v.setBigUint64(V_OFFSET + i * 16, 0n, true); v.setBigUint64(V_OFFSET + i * 16 + 8, 0n, true); }
    for (let i = 0; i < 4; i++) v.setUint32(V_OFFSET + 17 * 16 + i * 4, 0x80000000, true);
    seed();
    v.setBigUint64(0x100, BigInt(PC), true);
    v.setBigUint64(0xf8, 0x700000n, true);
    v.setBigUint64(30 * 8, BigInt(PC + 0x800), true);
    (emu as any).regionCache.clear();
    (emu as any).getRegion(BigInt(PC)).run();
    // 0x80000000 >>> 31 == 1, plus the 5 already in the lane
    for (let i = 0; i < 4; i++) expect(v.getUint32(V_OFFSET + 18 * 16 + i * 4, true)).toBe(6);
  });

  it('FMUL v4.2d, v4.2d, v2.d[0] multiplies every lane by one element', () => {
    if (!emu) return;
    const v = run(0x4fc29084, { 4: [3, 5], 2: [7, 100] }, 'f64');
    expect(v.getFloat64(V_OFFSET + 4 * 16, true)).toBe(21);
    expect(v.getFloat64(V_OFFSET + 4 * 16 + 8, true)).toBe(35);
  });

  it('FMUL d4, d2, v3.d[1] takes the HIGH lane in the scalar form', () => {
    if (!emu) return;
    const v = run(0x5fc39844, { 2: [3], 3: [0, 11] }, 'f64');
    expect(v.getFloat64(V_OFFSET + 4 * 16, true)).toBe(33);
    expect(v.getBigUint64(V_OFFSET + 4 * 16 + 8, true)).toBe(0n);
  });

  it('FCVT h0, s1 rounds a single to binary16 and back', () => {
    if (!emu) return;
    // s -> h, then h -> s: 1.5 is exact in half precision
    let v = run(0x1e23c020, { 1: [1.5] }, 'f32');
    expect(v.getUint16(V_OFFSET, true)).toBe(0x3e00);
    v = run(0x1ee24001, { 0: [] }, 'f32');
    // rerun with h0 seeded by hand: the harness zeroes registers between runs
    const dv = new DataView(emu!.memory.buffer);
    dv.setUint32(PC + GUEST_MEM_BASE, 0x1ee24001, true); // fcvt s1, h0
    dv.setUint32(PC + 4 + GUEST_MEM_BASE, RET, true);
    for (let i = 0; i < 32; i++) { dv.setBigUint64(V_OFFSET + i * 16, 0n, true); dv.setBigUint64(V_OFFSET + i * 16 + 8, 0n, true); }
    dv.setUint16(V_OFFSET, 0x3e00, true);
    dv.setBigUint64(0x100, BigInt(PC), true);
    dv.setBigUint64(0xf8, 0x700000n, true);
    dv.setBigUint64(30 * 8, BigInt(PC + 0x800), true);
    (emu as any).regionCache.clear();
    (emu as any).getRegion(BigInt(PC)).run();
    expect(dv.getFloat32(V_OFFSET + 1 * 16, true)).toBe(1.5);
  });

  it('FCVT h0, s1 saturates a too-large single to infinity', () => {
    if (!emu) return;
    const v = run(0x1e23c020, { 1: [1e30] }, 'f32');
    expect(v.getUint16(V_OFFSET, true)).toBe(0x7c00); // +inf
  });
});
