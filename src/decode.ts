import type { DecodedInsn, ShiftType } from './types.js';

// ── Bit extraction helpers ───────────────────────────────────────────────────

function bits(insn: number, hi: number, lo: number): number {
  return (insn >>> lo) & ((1 << (hi - lo + 1)) - 1);
}

function bit(insn: number, pos: number): number {
  return (insn >>> pos) & 1;
}

function signExtend(value: number, bitWidth: number): number {
  const shift = 32 - bitWidth;
  return (value << shift) >> shift;
}

const SHIFT_TYPES: ShiftType[] = ['lsl', 'lsr', 'asr', 'ror'];

// ── Logical immediate decoder ────────────────────────────────────────────────
// ARM64's logical immediate encoding is notoriously complex.
// (N, immr, imms) → 64-bit bitmask

export function decodeLogicalImm(N: number, immr: number, imms: number, is64: boolean): bigint | null {
  const len = 31 - Math.clz32((N << 6) | (~imms & 0x3F));
  if (len < 1) return null; // invalid

  const size = 1 << len;
  const levels = size - 1;
  const S = imms & levels;
  const R = immr & levels;

  if (S === levels) return null; // reserved

  const welem = (1n << BigInt(S + 1)) - 1n;
  // Rotate right by R within element of 'size' bits
  const mask = (1n << BigInt(size)) - 1n;
  const rotated = ((welem >> BigInt(R)) | (welem << BigInt(size - R))) & mask;

  // Replicate the element to fill 64 bits
  let result = 0n;
  for (let i = 0; i < 64; i += size) {
    result |= rotated << BigInt(i);
  }

  if (!is64) {
    result &= 0xFFFFFFFFn;
  }

  return result;
}

// ── Main decoder ─────────────────────────────────────────────────────────────

export function decode(insn: number): DecodedInsn {
  const op0 = bits(insn, 28, 25); // Primary dispatch

  switch (op0) {
    // Data processing — immediate
    case 0b1000: case 0b1001:
      return decodeDataProcImm(insn);

    // Branches, exception, system
    case 0b1010: case 0b1011:
      return decodeBranches(insn);

    // Loads and stores
    case 0b0100: case 0b1100: case 0b0110: case 0b1110:
      return decodeLoadStore(insn);

    // Data processing — register
    case 0b0101: case 0b1101:
      return decodeDataProcReg(insn);

    default:
      return { kind: 'unknown', raw: insn };
  }
}

// ── Data processing — immediate ──────────────────────────────────────────────

function decodeDataProcImm(insn: number): DecodedInsn {
  const op0 = bits(insn, 25, 23);

  switch (op0) {
    case 0b010: // Add/subtract (immediate)
      return decodeAddSubImm(insn);
    case 0b100: // Logical (immediate)
      return decodeLogicalImmInsn(insn);
    case 0b101: // Move wide (immediate)
      return decodeMoveWide(insn);
    case 0b110: // Bitfield
      return decodeBitfield(insn);
    case 0b000: case 0b001: // PC-rel addressing
      return decodePCRel(insn);
    default:
      return { kind: 'unknown', raw: insn };
  }
}

function decodeAddSubImm(insn: number): DecodedInsn {
  const sf = bit(insn, 31) as 0 | 1;
  const op = bit(insn, 30);   // 0=ADD, 1=SUB
  const S = bit(insn, 29);    // set flags
  const sh = bit(insn, 22);   // shift: 0=none, 1=LSL #12
  const imm12 = bits(insn, 21, 10);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);

  const imm = sh ? imm12 << 12 : imm12;

  return {
    kind: op ? 'sub_imm' : 'add_imm',
    sf, rd, rn, imm, setFlags: S === 1,
  };
}

function decodeLogicalImmInsn(insn: number): DecodedInsn {
  const sf = bit(insn, 31) as 0 | 1;
  const opc = bits(insn, 30, 29);
  const N = bit(insn, 22);
  const immr = bits(insn, 21, 16);
  const imms = bits(insn, 15, 10);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);

  const imm = decodeLogicalImm(N, immr, imms, sf === 1);
  if (imm === null) return { kind: 'unknown', raw: insn };

  switch (opc) {
    case 0b00: return { kind: 'and_imm', sf, rd, rn, imm, setFlags: false };
    case 0b01: return { kind: 'orr_imm', sf, rd, rn, imm };
    case 0b10: return { kind: 'eor_imm', sf, rd, rn, imm };
    case 0b11: return { kind: 'and_imm', sf, rd, rn, imm, setFlags: true }; // ANDS
    default: return { kind: 'unknown', raw: insn };
  }
}

function decodeMoveWide(insn: number): DecodedInsn {
  const sf = bit(insn, 31) as 0 | 1;
  const opc = bits(insn, 30, 29);
  const hw = bits(insn, 22, 21);
  const imm16 = bits(insn, 20, 5);
  const rd = bits(insn, 4, 0);

  switch (opc) {
    case 0b00: return { kind: 'movn', sf, rd, imm16, hw };
    case 0b10: return { kind: 'movz', sf, rd, imm16, hw };
    case 0b11: return { kind: 'movk', sf, rd, imm16, hw };
    default: return { kind: 'unknown', raw: insn };
  }
}

function decodeBitfield(insn: number): DecodedInsn {
  const sf = bit(insn, 31) as 0 | 1;
  const opc = bits(insn, 30, 29);
  const immr = bits(insn, 21, 16);
  const imms = bits(insn, 15, 10);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);

  switch (opc) {
    case 0b00: return { kind: 'sbfm', sf, rd, rn, immr, imms };
    case 0b01: return { kind: 'bfm', sf, rd, rn, immr, imms };
    case 0b10: return { kind: 'ubfm', sf, rd, rn, immr, imms };
    default: return { kind: 'unknown', raw: insn };
  }
}

function decodePCRel(insn: number): DecodedInsn {
  const op = bit(insn, 31);
  const immlo = bits(insn, 30, 29);
  const immhi = bits(insn, 23, 5);
  const rd = bits(insn, 4, 0);

  const imm = signExtend((immhi << 2) | immlo, 21);

  if (op === 0) {
    return { kind: 'adr', rd, imm };
  } else {
    return { kind: 'adrp', rd, imm: imm << 12 };
  }
}

// ── Branches, exception, system ──────────────────────────────────────────────

function decodeBranches(insn: number): DecodedInsn {
  const op0 = bits(insn, 31, 29);
  const op1 = bits(insn, 25, 22);

  // Unconditional branch (immediate): B, BL
  if (bits(insn, 30, 26) === 0b00101 || bits(insn, 30, 26) === 0b00101) {
    const op = bit(insn, 31);
    const imm26 = bits(insn, 25, 0);
    const offset = signExtend(imm26, 26) * 4;
    return op ? { kind: 'bl', offset } : { kind: 'b', offset };
  }

  // Conditional branch: B.cond
  if (bits(insn, 31, 25) === 0b0101010 && bit(insn, 4) === 0) {
    const imm19 = bits(insn, 23, 5);
    const cond = bits(insn, 3, 0);
    const offset = signExtend(imm19, 19) * 4;
    return { kind: 'b_cond', cond, offset };
  }

  // Compare and branch: CBZ, CBNZ
  if (bits(insn, 30, 25) === 0b011010) {
    const sf = bit(insn, 31) as 0 | 1;
    const op = bit(insn, 24);
    const imm19 = bits(insn, 23, 5);
    const rt = bits(insn, 4, 0);
    const offset = signExtend(imm19, 19) * 4;
    return op
      ? { kind: 'cbnz', sf, rt, offset }
      : { kind: 'cbz', sf, rt, offset };
  }

  // Test and branch: TBZ, TBNZ
  if (bits(insn, 30, 25) === 0b011011) {
    const b5 = bit(insn, 31);
    const op = bit(insn, 24);
    const b40 = bits(insn, 23, 19);
    const imm14 = bits(insn, 18, 5);
    const rt = bits(insn, 4, 0);
    const bitPos = (b5 << 5) | b40;
    const offset = signExtend(imm14, 14) * 4;
    return op
      ? { kind: 'tbnz', rt, bit: bitPos, offset }
      : { kind: 'tbz', rt, bit: bitPos, offset };
  }

  // Unconditional branch (register): BR, BLR, RET
  if (bits(insn, 31, 25) === 0b1101011) {
    const opc = bits(insn, 24, 21);
    const rn = bits(insn, 9, 5);
    switch (opc) {
      case 0b0000: return { kind: 'br', rn };
      case 0b0001: return { kind: 'blr', rn };
      case 0b0010: return { kind: 'ret', rn };
    }
  }

  // Exception generation: SVC
  if (bits(insn, 31, 24) === 0b11010100) {
    const opc = bits(insn, 23, 21);
    const imm16 = bits(insn, 20, 5);
    if (opc === 0b000 && bits(insn, 4, 0) === 0b00001) {
      return { kind: 'svc', imm16 };
    }
  }

  // System: NOP and others
  if (insn === 0xD503201F) return { kind: 'nop' };

  return { kind: 'unknown', raw: insn };
}

// ── Load/Store ───────────────────────────────────────────────────────────────

function decodeLoadStore(insn: number): DecodedInsn {
  const op0 = bits(insn, 31, 28);
  const op1 = bit(insn, 26);
  const op2 = bits(insn, 24, 23);
  const op3 = bits(insn, 21, 16);
  const op4 = bits(insn, 11, 10);

  // Load/store pair (LDP, STP)
  if (bits(insn, 29, 27) === 0b101 && op1 === 0) {
    return decodeLoadStorePair(insn);
  }

  // Load/store register (unsigned offset)
  if (bits(insn, 29, 27) === 0b111 && bit(insn, 24) === 1 && op1 === 0) {
    return decodeLoadStoreUnsignedOff(insn);
  }

  // Load/store register (pre/post index, unscaled)
  if (bits(insn, 29, 27) === 0b111 && bit(insn, 24) === 0 && op1 === 0 && bit(insn, 21) === 0) {
    return decodeLoadStorePrePost(insn);
  }

  // Load/store register (register offset)
  if (bits(insn, 29, 27) === 0b111 && bit(insn, 24) === 0 && op1 === 0 && bit(insn, 21) === 1) {
    return decodeLoadStoreRegOff(insn);
  }

  return { kind: 'unknown', raw: insn };
}

function decodeLoadStoreUnsignedOff(insn: number): DecodedInsn {
  const size = bits(insn, 31, 30);
  const opc = bits(insn, 23, 22);
  const imm12 = bits(insn, 21, 10);
  const rn = bits(insn, 9, 5);
  const rt = bits(insn, 4, 0);

  const scale = size;
  const offset = imm12 << scale;
  const isLoad = (opc & 1) === 1;

  // opc[1] for sign extension
  if (opc >= 2) {
    // Sign-extending loads
    switch (size) {
      case 0b00: // LDRSB
        return { kind: 'ldrsb_imm', sf: (opc === 2 ? 1 : 0) as 0|1, rt, rn, offset: imm12, mode: 'offset' };
      case 0b01: // LDRSH
        return { kind: 'ldrsh_imm', sf: (opc === 2 ? 1 : 0) as 0|1, rt, rn, offset: imm12 << 1, mode: 'offset' };
      case 0b10: // LDRSW
        return { kind: 'ldrsw_imm', rt, rn, offset: imm12 << 2, mode: 'offset' };
    }
  }

  switch (size) {
    case 0b00: // byte
      return isLoad
        ? { kind: 'ldrb_imm', rt, rn, offset, mode: 'offset' }
        : { kind: 'strb_imm', rt, rn, offset, mode: 'offset' };
    case 0b01: // halfword
      return isLoad
        ? { kind: 'ldrh_imm', rt, rn, offset, mode: 'offset' }
        : { kind: 'strh_imm', rt, rn, offset, mode: 'offset' };
    case 0b10: // 32-bit
      return isLoad
        ? { kind: 'ldr_imm', sf: 0, rt, rn, offset, mode: 'offset' }
        : { kind: 'str_imm', sf: 0, rt, rn, offset, mode: 'offset' };
    case 0b11: // 64-bit
      return isLoad
        ? { kind: 'ldr_imm', sf: 1, rt, rn, offset, mode: 'offset' }
        : { kind: 'str_imm', sf: 1, rt, rn, offset, mode: 'offset' };
  }

  return { kind: 'unknown', raw: insn };
}

function decodeLoadStorePrePost(insn: number): DecodedInsn {
  const size = bits(insn, 31, 30);
  const opc = bits(insn, 23, 22);
  const imm9 = bits(insn, 20, 12);
  const mode = bit(insn, 11) ? 'pre' as const : 'post' as const;
  const rn = bits(insn, 9, 5);
  const rt = bits(insn, 4, 0);
  const offset = signExtend(imm9, 9);
  const isLoad = (opc & 1) === 1;

  switch (size) {
    case 0b10:
      return isLoad
        ? { kind: 'ldr_imm', sf: 0, rt, rn, offset, mode }
        : { kind: 'str_imm', sf: 0, rt, rn, offset, mode };
    case 0b11:
      return isLoad
        ? { kind: 'ldr_imm', sf: 1, rt, rn, offset, mode }
        : { kind: 'str_imm', sf: 1, rt, rn, offset, mode };
    case 0b00:
      return isLoad
        ? { kind: 'ldrb_imm', rt, rn, offset, mode }
        : { kind: 'strb_imm', rt, rn, offset, mode };
    case 0b01:
      return isLoad
        ? { kind: 'ldrh_imm', rt, rn, offset, mode }
        : { kind: 'strh_imm', rt, rn, offset, mode };
  }

  return { kind: 'unknown', raw: insn };
}

function decodeLoadStoreRegOff(insn: number): DecodedInsn {
  const size = bits(insn, 31, 30);
  const opc = bits(insn, 23, 22);
  const rm = bits(insn, 20, 16);
  const option = bits(insn, 15, 13);
  const S = bit(insn, 12);
  const rn = bits(insn, 9, 5);
  const rt = bits(insn, 4, 0);
  const isLoad = (opc & 1) === 1;
  const sf = (size >= 2 ? size - 2 : 0) as 0 | 1;
  const shift = S ? size : 0;
  const extend = option === 0b011 ? 'lsl' : option === 0b010 ? 'uxtw' : option === 0b110 ? 'sxtw' : option === 0b111 ? 'sxtx' : 'lsl';

  return isLoad
    ? { kind: 'ldr_reg', sf: size === 3 ? 1 : 0, rt, rn, rm, extend, shift }
    : { kind: 'str_reg', sf: size === 3 ? 1 : 0, rt, rn, rm, extend, shift };
}

function decodeLoadStorePair(insn: number): DecodedInsn {
  const opc = bits(insn, 31, 30);
  const type = bits(insn, 24, 23); // 01=post, 10=offset, 11=pre
  const L = bit(insn, 22);
  const imm7 = bits(insn, 21, 15);
  const rt2 = bits(insn, 14, 10);
  const rn = bits(insn, 9, 5);
  const rt = bits(insn, 4, 0);

  const sf = (opc === 0b10 ? 1 : 0) as 0 | 1;
  const scale = sf ? 3 : 2; // 8 bytes for 64-bit, 4 for 32-bit
  const offset = signExtend(imm7, 7) << scale;
  const mode = type === 0b01 ? 'post' as const : type === 0b11 ? 'pre' as const : 'offset' as const;

  return L
    ? { kind: 'ldp', sf, rt, rt2, rn, offset, mode }
    : { kind: 'stp', sf, rt, rt2, rn, offset, mode };
}

// ── Data processing — register ───────────────────────────────────────────────

function decodeDataProcReg(insn: number): DecodedInsn {
  const op0 = bit(insn, 30);
  const op1 = bit(insn, 28);
  const op2 = bits(insn, 24, 21);

  // Logical (shifted register): bits [28:24] = x1010
  if (bits(insn, 28, 24) === 0b01010) {
    return decodeLogicalReg(insn);
  }

  // Add/sub (shifted register): bits [28:24] = x1011, bit 21 = 0
  if (bits(insn, 28, 24) === 0b01011 && bit(insn, 21) === 0) {
    return decodeAddSubReg(insn);
  }

  // Add/sub (extended register): bits [28:24] = x1011, bit 21 = 1
  if (bits(insn, 28, 24) === 0b01011 && bit(insn, 21) === 1) {
    return decodeAddSubExtReg(insn);
  }

  // Conditional select: bits [28:21] = 11010100
  if (bits(insn, 28, 21) === 0b11010100) {
    return decodeCondSelect(insn);
  }

  // Data processing (2 source): bits [28:21] = 11010110
  if (bits(insn, 28, 21) === 0b11010110) {
    return decodeDataProc2Src(insn);
  }

  // Data processing (3 source): bits [28:24] = 11011
  if (bits(insn, 28, 24) === 0b11011) {
    return decodeDataProc3Src(insn);
  }

  return { kind: 'unknown', raw: insn };
}

function decodeAddSubReg(insn: number): DecodedInsn {
  const sf = bit(insn, 31) as 0 | 1;
  const op = bit(insn, 30);   // 0=ADD, 1=SUB
  const S = bit(insn, 29);
  const shift = SHIFT_TYPES[bits(insn, 23, 22)];
  const rm = bits(insn, 20, 16);
  const amount = bits(insn, 15, 10);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);

  return {
    kind: op ? 'sub_reg' : 'add_reg',
    sf, rd, rn, rm, shift, amount, setFlags: S === 1,
  };
}

function decodeAddSubExtReg(insn: number): DecodedInsn {
  // Treat extended register as shifted register for simplicity
  // The extend type is in bits [15:13], shift amount in bits [12:10]
  const sf = bit(insn, 31) as 0 | 1;
  const op = bit(insn, 30);
  const S = bit(insn, 29);
  const rm = bits(insn, 20, 16);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);
  const amount = bits(insn, 12, 10);

  return {
    kind: op ? 'sub_reg' : 'add_reg',
    sf, rd, rn, rm, shift: 'lsl', amount, setFlags: S === 1,
  };
}

function decodeLogicalReg(insn: number): DecodedInsn {
  const sf = bit(insn, 31) as 0 | 1;
  const opc = bits(insn, 30, 29);
  const shift = SHIFT_TYPES[bits(insn, 23, 22)];
  const N = bit(insn, 21);  // invert Rm
  const rm = bits(insn, 20, 16);
  const amount = bits(insn, 15, 10);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);

  switch (opc) {
    case 0b00: // AND / BIC
      return { kind: 'and_reg', sf, rd, rn, rm, shift, amount, invert: N === 1, setFlags: false };
    case 0b01: // ORR / ORN
      return { kind: 'orr_reg', sf, rd, rn, rm, shift, amount, invert: N === 1 };
    case 0b10: // EOR / EON
      return { kind: 'eor_reg', sf, rd, rn, rm, shift, amount, invert: N === 1 };
    case 0b11: // ANDS / BICS
      return { kind: 'and_reg', sf, rd, rn, rm, shift, amount, invert: N === 1, setFlags: true };
    default:
      return { kind: 'unknown', raw: insn };
  }
}

function decodeCondSelect(insn: number): DecodedInsn {
  const sf = bit(insn, 31) as 0 | 1;
  const op = bit(insn, 30);
  const rm = bits(insn, 20, 16);
  const cond = bits(insn, 15, 12);
  const op2 = bits(insn, 11, 10);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);

  const variants: Record<number, 'csel' | 'csinc' | 'csinv' | 'csneg'> = {
    0b00: op === 0 ? 'csel' : 'csinv',
    0b01: op === 0 ? 'csinc' : 'csneg',
  };

  return {
    kind: 'csel', sf, rd, rn, rm, cond,
    variant: variants[op2] ?? 'csel',
  };
}

function decodeDataProc2Src(insn: number): DecodedInsn {
  const sf = bit(insn, 31) as 0 | 1;
  const rm = bits(insn, 20, 16);
  const opcode = bits(insn, 15, 10);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);

  switch (opcode) {
    case 0b000010: return { kind: 'udiv', sf, rd, rn, rm };
    case 0b000011: return { kind: 'sdiv', sf, rd, rn, rm };
    default: return { kind: 'unknown', raw: insn };
  }
}

function decodeDataProc3Src(insn: number): DecodedInsn {
  const sf = bit(insn, 31) as 0 | 1;
  const op31 = bits(insn, 23, 21);
  const rm = bits(insn, 20, 16);
  const o0 = bit(insn, 15);
  const ra = bits(insn, 14, 10);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);

  if (op31 === 0b000) {
    return o0
      ? { kind: 'msub', sf, rd, rn, rm, ra }
      : { kind: 'madd', sf, rd, rn, rm, ra };
  }

  // SMULH/UMULH (64-bit only)
  if (sf === 1 && op31 === 0b010 && o0 === 0) {
    return { kind: 'smulh', rd, rn, rm };
  }
  if (sf === 1 && op31 === 0b110 && o0 === 0) {
    return { kind: 'umulh', rd, rn, rm };
  }

  return { kind: 'unknown', raw: insn };
}
