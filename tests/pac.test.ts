// pointer authentication (armv8.3 pac) conformance. we do not model the pac key,
// so sign and auth both round-trip to identity, exactly as qemu-user does when no
// key is configured. this is correct for execution because the abi always
// authenticates a pointer before using it, so auth(sign(x)) == x, and no program
// inspects the (absent) signature bits. these tests prove the register-form pac
// ops are no-ops and the pac branch variants behave like their plain counterparts.

import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';
import { Emulator } from '../src/dispatch.js';
import { loadElf } from '../src/elf.js';
import { GUEST_MEM_BASE } from '../src/types.js';
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

// run a sequence of instruction words (terminated by the caller) with GPRs preloaded
function runGpr(words: number[], regs: Record<number, bigint>): DataView {
  const v = new DataView(emu!.memory.buffer);
  words.forEach((w, i) => v.setUint32(PC + i * 4 + GUEST_MEM_BASE, w >>> 0, true));
  for (let i = 0; i < 31; i++) v.setBigUint64(i * 8, 0n, true);
  v.setBigUint64(0x100, BigInt(PC), true);
  v.setBigUint64(0xf8, 0x700000n, true);
  for (const [reg, val] of Object.entries(regs)) v.setBigUint64(Number(reg) * 8, val, true);
  (emu as any).regionCache.clear();
  (emu as any).getRegion(BigInt(PC)).run();
  return new DataView(emu!.memory.buffer);
}

const X = (v: DataView, r: number) => v.getBigUint64(r * 8, true);

describe('PAC as identity (register forms)', () => {
  it('PACIA X0,X1 leaves X0 unchanged', () => {
    if (!emu) return;
    const v = runGpr([0xdac10020, RET], { 0: 0xdeadbeefcafef00dn, 1: 0x1234n });
    expect(X(v, 0)).toBe(0xdeadbeefcafef00dn);
  });

  it('PACIA then AUTIA round-trips the pointer', () => {
    if (!emu) return;
    // PACIA X0,X1 ; AUTIA X0,X1 ; RET
    const v = runGpr([0xdac10020, 0xdac11020, RET], { 0: 0x0000ffff11223344n, 1: 0x99n });
    expect(X(v, 0)).toBe(0x0000ffff11223344n);
  });

  it('XPACI / XPACD strip nothing (never signed)', () => {
    if (!emu) return;
    let v = runGpr([0xdac143e0, RET], { 0: 0x00007fffabcd0000n });
    expect(X(v, 0)).toBe(0x00007fffabcd0000n);
    v = runGpr([0xdac147e0, RET], { 0: 0x00007fffabcd0000n });
    expect(X(v, 0)).toBe(0x00007fffabcd0000n);
  });

  it('PACIASP / AUTIASP (hint space) round-trip the link register', () => {
    if (!emu) return;
    // PACIASP ; AUTIASP ; RET, with X30 as the return address
    const v = runGpr([0xd503233f, 0xd50323bf, RET], { 30: BigInt(PC + 0x40) });
    expect(X(v, 30)).toBe(BigInt(PC + 0x40));
  });
});

describe('PAC branch variants behave as plain branches', () => {
  it('RETAA returns to X30 like RET', () => {
    if (!emu) return;
    // sole terminator RETAA; region should exit with PC = X30
    const target = BigInt(PC + 0x400);
    const v = runGpr([0xd65f0bdf], { 30: target });
    expect(v.getBigUint64(0x100, true)).toBe(target);
  });

  it('BRAAZ X1 branches to X1 like BR', () => {
    if (!emu) return;
    const target = BigInt(PC + 0x200);
    const v = runGpr([0xd61f083f], { 1: target });
    expect(v.getBigUint64(0x100, true)).toBe(target);
  });

  it('BRAA X1,X2 branches to X1 (modifier ignored)', () => {
    if (!emu) return;
    const target = BigInt(PC + 0x280);
    const v = runGpr([0xd71f0822], { 1: target, 2: 0xabcdn });
    expect(v.getBigUint64(0x100, true)).toBe(target);
  });

  it('BLRAA X1,X2 branches to X1 and links (X30 = next pc)', () => {
    if (!emu) return;
    const target = BigInt(PC + 0x300);
    const v = runGpr([0xd73f0822], { 1: target, 2: 0x77n });
    expect(v.getBigUint64(0x100, true)).toBe(target);
    expect(X(v, 30)).toBe(BigInt(PC + 4));
  });
});
