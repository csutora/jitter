// direct semantics tests for the zig translator's lse atomics + cas.
// each op is assembled at a scratch pc and run against known state.

import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';
import { Emulator } from '../src/dispatch.js';
import { loadElf } from '../src/elf.js';
import { GUEST_MEM_BASE } from '../src/types.js';
import { initZigTranslator } from '../src/zig-translator.js';

const PC = 0x600000, DATA = 0x700000, RET = 0xd65f03c0, A = DATA + 0x100;

let emu: Emulator | null = null;
let temu: Emulator | null = null; // threaded: exercises the atomic.rmw path

function loadTranslator() {
  try {
    return new Uint8Array(readFileSync(join(__dirname, '..', 'engine', 'zig-out', 'bin', 'translator.wasm')));
  } catch { return null; }
}

beforeAll(() => {
  const t = loadTranslator();
  if (!t) return;
  initZigTranslator(t);
  const bb = new Uint8Array(readFileSync(join(__dirname, '..', 'assets', 'busybox-aarch64')));
  emu = new Emulator(1024);
  emu.elfBytes = bb;
  loadElf(emu.elfBytes, emu.memory);
  temu = new Emulator(1024, true);
  temu.elfBytes = bb;
  loadElf(temu.elfBytes, temu.memory);
});

function runG(e: Emulator, word: number, setup: (v: DataView) => void): DataView {
  const v = new DataView(e.memory.buffer);
  v.setUint32(PC + GUEST_MEM_BASE, word, true);
  v.setUint32(PC + 4 + GUEST_MEM_BASE, RET, true);
  for (let i = 0; i < 31; i++) v.setBigUint64(i * 8, 0n, true);
  v.setBigUint64(0xf8, BigInt(DATA), true);
  v.setBigUint64(0x100, BigInt(PC), true);
  v.setBigUint64(0x110, 0n, true);
  v.setBigUint64(30 * 8, BigInt(PC + 0x800), true);
  setup(v);
  (e as any).regionCache.clear();
  (e as any).getRegion(BigInt(PC)).run();
  return new DataView(e.memory.buffer);
}

function run(word: number, setup: (v: DataView) => void): DataView {
  const v = new DataView(emu!.memory.buffer);
  v.setUint32(PC + GUEST_MEM_BASE, word, true);
  v.setUint32(PC + 4 + GUEST_MEM_BASE, RET, true);
  for (let i = 0; i < 31; i++) v.setBigUint64(i * 8, 0n, true);
  v.setBigUint64(0xf8, BigInt(DATA), true);
  v.setBigUint64(0x100, BigInt(PC), true);
  v.setBigUint64(0x110, 0n, true);
  v.setBigUint64(30 * 8, BigInt(PC + 0x800), true);
  setup(v);
  (emu as any).regionCache.clear();
  (emu as any).getRegion(BigInt(PC)).run();
  return new DataView(emu!.memory.buffer);
}

describe('lse atomics (zig)', () => {
  it('LDADD adds and returns old', () => {
    if (!emu) return;
    const v = run(0xf8210002, (m) => {
      m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 5n, true);
      m.setBigUint64(A + GUEST_MEM_BASE, 100n, true);
    });
    expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(105n);
    expect(v.getBigUint64(16, true)).toBe(100n);
  });

  it('LDSET ors', () => {
    if (!emu) return;
    const v = run(0xf8213002, (m) => {
      m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 0xf0n, true);
      m.setBigUint64(A + GUEST_MEM_BASE, 0x0fn, true);
    });
    expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(0xffn);
  });

  it('LDCLR bit-clears', () => {
    if (!emu) return;
    const v = run(0xf8211002, (m) => {
      m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 0x0fn, true);
      m.setBigUint64(A + GUEST_MEM_BASE, 0xffn, true);
    });
    expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(0xf0n);
  });

  it('LDEOR xors', () => {
    if (!emu) return;
    const v = run(0xf8212002, (m) => {
      m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 0xffn, true);
      m.setBigUint64(A + GUEST_MEM_BASE, 0x0fn, true);
    });
    expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(0xf0n);
  });

  it('SWP swaps', () => {
    if (!emu) return;
    const v = run(0xf8218002, (m) => {
      m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 0xdeadn, true);
      m.setBigUint64(A + GUEST_MEM_BASE, 0xbeefn, true);
    });
    expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(0xdeadn);
    expect(v.getBigUint64(16, true)).toBe(0xbeefn);
  });

  it('CAS stores on match', () => {
    if (!emu) return;
    const v = run(0xc8a17c02, (m) => {
      m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 100n, true); m.setBigUint64(16, 200n, true);
      m.setBigUint64(A + GUEST_MEM_BASE, 100n, true);
    });
    expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(200n);
    expect(v.getBigUint64(8, true)).toBe(100n);
  });

  it('CAS leaves memory on mismatch', () => {
    if (!emu) return;
    const v = run(0xc8a17c02, (m) => {
      m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 100n, true); m.setBigUint64(16, 200n, true);
      m.setBigUint64(A + GUEST_MEM_BASE, 999n, true);
    });
    expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(999n);
    expect(v.getBigUint64(8, true)).toBe(999n);
  });

  it('LDADD 32-bit', () => {
    if (!emu) return;
    const v = run(0xb8210002, (m) => {
      m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 7n, true);
      m.setUint32(A + GUEST_MEM_BASE, 40, true);
    });
    expect(v.getUint32(A + GUEST_MEM_BASE, true)).toBe(47);
  });

  it('LDUMAX/LDUMIN unsigned', () => {
    if (!emu) return;
    let v = run(0xf8216002, (m) => {
      m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 50n, true);
      m.setBigUint64(A + GUEST_MEM_BASE, 30n, true);
    });
    expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(50n);
    v = run(0xf8217002, (m) => {
      m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 50n, true);
      m.setBigUint64(A + GUEST_MEM_BASE, 30n, true);
    });
    expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(30n);
  });

  it('LDSMAX/LDSMIN signed', () => {
    if (!emu) return;
    let v = run(0xf8214002, (m) => {
      m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 3n, true);
      m.setBigUint64(A + GUEST_MEM_BASE, BigInt.asUintN(64, -5n), true);
    });
    expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(3n);
    v = run(0xf8215002, (m) => {
      m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 3n, true);
      m.setBigUint64(A + GUEST_MEM_BASE, BigInt.asUintN(64, -5n), true);
    });
    expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(BigInt.asUintN(64, -5n));
  });

  it('LDSMAX 32-bit signed', () => {
    if (!emu) return;
    const v = run(0xb8214002, (m) => {
      m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 3n, true);
      m.setInt32(A + GUEST_MEM_BASE, -5, true);
    });
    expect(v.getInt32(A + GUEST_MEM_BASE, true)).toBe(3);
  });

  // ── threaded mode: the SAME ops go through real wasm atomic.rmw / cmpxchg (shared_monitor).
  // single thread so results are deterministic and must match the known-good vectors — this
  // validates the atomic-op encodings/sizes. ──
  describe('threaded (real atomic.rmw)', () => {
    const rt = (word: number, setup: (v: DataView) => void) => runG(temu!, word, setup);

    it('LDADD atomic', () => {
      if (!temu) return;
      const v = rt(0xf8210002, (m) => { m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 5n, true); m.setBigUint64(A + GUEST_MEM_BASE, 100n, true); });
      expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(105n);
      expect(v.getBigUint64(16, true)).toBe(100n);
    });
    it('LDSET atomic (or)', () => {
      if (!temu) return;
      const v = rt(0xf8213002, (m) => { m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 0xf0n, true); m.setBigUint64(A + GUEST_MEM_BASE, 0x0fn, true); });
      expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(0xffn);
    });
    it('LDCLR atomic (and ~Rs)', () => {
      if (!temu) return;
      const v = rt(0xf8211002, (m) => { m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 0x0fn, true); m.setBigUint64(A + GUEST_MEM_BASE, 0xffn, true); });
      expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(0xf0n);
    });
    it('LDEOR atomic (xor)', () => {
      if (!temu) return;
      const v = rt(0xf8212002, (m) => { m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 0xffn, true); m.setBigUint64(A + GUEST_MEM_BASE, 0x0fn, true); });
      expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(0xf0n);
    });
    it('SWP atomic (xchg)', () => {
      if (!temu) return;
      const v = rt(0xf8218002, (m) => { m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 0xdeadn, true); m.setBigUint64(A + GUEST_MEM_BASE, 0xbeefn, true); });
      expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(0xdeadn);
      expect(v.getBigUint64(16, true)).toBe(0xbeefn);
    });
    it('CAS atomic match', () => {
      if (!temu) return;
      const v = rt(0xc8a17c02, (m) => { m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 100n, true); m.setBigUint64(16, 200n, true); m.setBigUint64(A + GUEST_MEM_BASE, 100n, true); });
      expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(200n);
      expect(v.getBigUint64(8, true)).toBe(100n);
    });
    it('CAS atomic mismatch', () => {
      if (!temu) return;
      const v = rt(0xc8a17c02, (m) => { m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 100n, true); m.setBigUint64(16, 200n, true); m.setBigUint64(A + GUEST_MEM_BASE, 999n, true); });
      expect(v.getBigUint64(A + GUEST_MEM_BASE, true)).toBe(999n);
      expect(v.getBigUint64(8, true)).toBe(999n);
    });
    it('LDADD atomic 32-bit', () => {
      if (!temu) return;
      const v = rt(0xb8210002, (m) => { m.setBigUint64(0, BigInt(A), true); m.setBigUint64(8, 7n, true); m.setUint32(A + GUEST_MEM_BASE, 40, true); });
      expect(v.getUint32(A + GUEST_MEM_BASE, true)).toBe(47);
    });
  });
});
