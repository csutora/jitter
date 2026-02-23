import { describe, it, expect } from 'vitest';
import { decode, decodeLogicalImm } from '../src/decode.js';

describe('ARM64 decoder', () => {
  // ── Add/Sub immediate ──────────────────────────────────────────────────────

  it('decodes ADD X1, X2, #42', () => {
    // ADD X1, X2, #42: sf=1, op=0(ADD), S=0, sh=0, imm12=42, Rn=2, Rd=1
    // 1_0_0_10001_0_000000101010_00010_00001
    const insn = 0x91_00_A8_41;
    const d = decode(insn);
    expect(d.kind).toBe('add_imm');
    if (d.kind === 'add_imm') {
      expect(d.sf).toBe(1);
      expect(d.rd).toBe(1);
      expect(d.rn).toBe(2);
      expect(d.imm).toBe(42);
      expect(d.setFlags).toBe(false);
    }
  });

  it('decodes SUB X0, X0, #1', () => {
    // SUB X0, X0, #1: sf=1, op=1(SUB), S=0, sh=0, imm12=1, Rn=0, Rd=0
    const insn = 0xD1000400;
    const d = decode(insn);
    expect(d.kind).toBe('sub_imm');
    if (d.kind === 'sub_imm') {
      expect(d.sf).toBe(1);
      expect(d.rd).toBe(0);
      expect(d.rn).toBe(0);
      expect(d.imm).toBe(1);
      expect(d.setFlags).toBe(false);
    }
  });

  it('decodes ADD X1, X1, #1', () => {
    const insn = 0x91000421;
    const d = decode(insn);
    expect(d.kind).toBe('add_imm');
    if (d.kind === 'add_imm') {
      expect(d.rd).toBe(1);
      expect(d.rn).toBe(1);
      expect(d.imm).toBe(1);
    }
  });

  it('decodes SUBS XZR, X1, X2 (CMP X1, X2)', () => {
    const insn = 0xEB02003F;
    const d = decode(insn);
    expect(d.kind).toBe('sub_reg');
    if (d.kind === 'sub_reg') {
      expect(d.sf).toBe(1);
      expect(d.rd).toBe(31); // XZR
      expect(d.rn).toBe(1);
      expect(d.rm).toBe(2);
      expect(d.setFlags).toBe(true);
    }
  });

  // ── Add/Sub register ───────────────────────────────────────────────────────

  it('decodes ADD X0, X0, X1', () => {
    const insn = 0x8B010000;
    const d = decode(insn);
    expect(d.kind).toBe('add_reg');
    if (d.kind === 'add_reg') {
      expect(d.sf).toBe(1);
      expect(d.rd).toBe(0);
      expect(d.rn).toBe(0);
      expect(d.rm).toBe(1);
      expect(d.shift).toBe('lsl');
      expect(d.amount).toBe(0);
      expect(d.setFlags).toBe(false);
    }
  });

  it('decodes ADD X3, X1, X2', () => {
    const insn = 0x8B020023;
    const d = decode(insn);
    expect(d.kind).toBe('add_reg');
    if (d.kind === 'add_reg') {
      expect(d.rd).toBe(3);
      expect(d.rn).toBe(1);
      expect(d.rm).toBe(2);
    }
  });

  // ── Logical register ──────────────────────────────────────────────────────

  it('decodes ORR X2, XZR, X1 (MOV X2, X1)', () => {
    const insn = 0xAA0103E2;
    const d = decode(insn);
    expect(d.kind).toBe('orr_reg');
    if (d.kind === 'orr_reg') {
      expect(d.sf).toBe(1);
      expect(d.rd).toBe(2);
      expect(d.rn).toBe(31); // XZR
      expect(d.rm).toBe(1);
      expect(d.shift).toBe('lsl');
      expect(d.amount).toBe(0);
      expect(d.invert).toBe(false);
    }
  });

  it('decodes ORR X1, XZR, X3 (MOV X1, X3)', () => {
    const insn = 0xAA0303E1;
    const d = decode(insn);
    expect(d.kind).toBe('orr_reg');
    if (d.kind === 'orr_reg') {
      expect(d.rd).toBe(1);
      expect(d.rn).toBe(31);
      expect(d.rm).toBe(3);
    }
  });

  // ── Move wide ──────────────────────────────────────────────────────────────

  it('decodes MOVZ X0, #5', () => {
    // MOVZ X0, #5: sf=1, opc=10, hw=0, imm16=5, Rd=0
    const insn = 0xD28000A0;
    const d = decode(insn);
    expect(d.kind).toBe('movz');
    if (d.kind === 'movz') {
      expect(d.sf).toBe(1);
      expect(d.rd).toBe(0);
      expect(d.imm16).toBe(5);
      expect(d.hw).toBe(0);
    }
  });

  it('decodes MOVZ X0, #0x1234, LSL #16', () => {
    // hw=1, imm16=0x1234
    const insn = 0xD2A24680;
    const d = decode(insn);
    expect(d.kind).toBe('movz');
    if (d.kind === 'movz') {
      expect(d.hw).toBe(1);
      expect(d.imm16).toBe(0x1234);
    }
  });

  // ── Branches ───────────────────────────────────────────────────────────────

  it('decodes B (backward, offset=-5 insns)', () => {
    const insn = 0x17FFFFFB;
    const d = decode(insn);
    expect(d.kind).toBe('b');
    if (d.kind === 'b') {
      expect(d.offset).toBe(-20); // -5 * 4
    }
  });

  it('decodes B.LE (backward, offset=-3 insns)', () => {
    const insn = 0x54FFFFAD;
    const d = decode(insn);
    expect(d.kind).toBe('b_cond');
    if (d.kind === 'b_cond') {
      expect(d.cond).toBe(13); // LE
      expect(d.offset).toBe(-12); // -3 * 4
    }
  });

  it('decodes CBZ X0, +6 insns', () => {
    const insn = 0xB40000C0;
    const d = decode(insn);
    expect(d.kind).toBe('cbz');
    if (d.kind === 'cbz') {
      expect(d.sf).toBe(1);
      expect(d.rt).toBe(0);
      expect(d.offset).toBe(24); // 6 * 4
    }
  });

  it('decodes RET (BR X30)', () => {
    const insn = 0xD65F03C0;
    const d = decode(insn);
    expect(d.kind).toBe('ret');
    if (d.kind === 'ret') {
      expect(d.rn).toBe(30);
    }
  });

  // ── System ─────────────────────────────────────────────────────────────────

  it('decodes NOP', () => {
    const d = decode(0xD503201F);
    expect(d.kind).toBe('nop');
  });

  it('decodes SVC #0', () => {
    const insn = 0xD4000001;
    const d = decode(insn);
    expect(d.kind).toBe('svc');
    if (d.kind === 'svc') {
      expect(d.imm16).toBe(0);
    }
  });

  // ── Multiply ───────────────────────────────────────────────────────────────

  it('decodes MUL X0, X1, X2 (MADD X0, X1, X2, XZR)', () => {
    // MADD: sf=1, op31=000, Rm=X2, o0=0, Ra=XZR(31), Rn=X1, Rd=X0
    const insn = 0x9B027C20;
    const d = decode(insn);
    expect(d.kind).toBe('madd');
    if (d.kind === 'madd') {
      expect(d.sf).toBe(1);
      expect(d.rd).toBe(0);
      expect(d.rn).toBe(1);
      expect(d.rm).toBe(2);
      expect(d.ra).toBe(31);
    }
  });

  // ── Division ───────────────────────────────────────────────────────────────

  it('decodes SDIV X0, X1, X2', () => {
    const insn = 0x9AC20C20;
    const d = decode(insn);
    expect(d.kind).toBe('sdiv');
    if (d.kind === 'sdiv') {
      expect(d.rd).toBe(0);
      expect(d.rn).toBe(1);
      expect(d.rm).toBe(2);
    }
  });

  // ── Logical immediate ─────────────────────────────────────────────────────

  it('decodes logical immediates', () => {
    // N=1 means 64-bit element (no replication)
    // AND X0, X1, #0xFF → N=1, immr=0, imms=7
    expect(decodeLogicalImm(1, 0, 7, true)).toBe(0xFFn);
    // AND X0, X1, #0xFFFFFFFF → N=1, immr=0, imms=31
    expect(decodeLogicalImm(1, 0, 31, true)).toBe(0xFFFFFFFFn);
    // AND X0, X1, #1 → N=1, immr=0, imms=0
    expect(decodeLogicalImm(1, 0, 0, true)).toBe(1n);

    // N=0 means smaller element, replicated across 64 bits
    // N=0, immr=0, imms=7 → 8 ones in a 32-bit element, replicated 2x
    expect(decodeLogicalImm(0, 0, 7, true)).toBe(0x000000FF_000000FFn);
    // N=0, immr=0, imms=0 → 1 one in a 32-bit element, replicated 2x
    expect(decodeLogicalImm(0, 0, 0, true)).toBe(0x00000001_00000001n);
    // N=0, immr=0, imms=0x3C → 1 one in 2-bit element, replicated = 0x5555...
    expect(decodeLogicalImm(0, 0, 0x3C, true)).toBe(0x5555555555555555n);
  });
});
