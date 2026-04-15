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

    // Data processing — SIMD & FP
    case 0b0111: case 0b1111:
      return decodeSimdFp(insn);

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
    case 0b111: // Extract
      return decodeExtract(insn);
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

  // Exception generation: SVC, BRK
  if (bits(insn, 31, 24) === 0b11010100) {
    const opc = bits(insn, 23, 21);
    const imm16 = bits(insn, 20, 5);
    if (opc === 0b000 && bits(insn, 4, 0) === 0b00001) {
      return { kind: 'svc', imm16 };
    }
    // BRK: opc=001, LL=00
    if (opc === 0b001 && bits(insn, 4, 0) === 0b00000) {
      return { kind: 'brk', imm16 };
    }
  }

  // System instructions: NOP, barriers, MRS, MSR
  if (bits(insn, 31, 22) === 0b1101010100) {
    return decodeSystem(insn);
  }

  return { kind: 'unknown', raw: insn };
}

// ── System instructions ──────────────────────────────────────────────────

function decodeSystem(insn: number): DecodedInsn {
  // NOP
  if (insn === 0xD503201F) return { kind: 'nop' };

  // Barriers: DMB, DSB, ISB
  const crm = bits(insn, 11, 8);
  const op2 = bits(insn, 7, 5);
  const rt = bits(insn, 4, 0);

  // DMB: 1101010100_0_00_011_0011_CRm_1_01_11111
  if (bits(insn, 21, 12) === 0b0000110011 && op2 === 0b101 && rt === 0b11111) {
    return { kind: 'nop' }; // treat as NOP
  }
  // DSB: 1101010100_0_00_011_0011_CRm_1_00_11111
  if (bits(insn, 21, 12) === 0b0000110011 && op2 === 0b100 && rt === 0b11111) {
    return { kind: 'nop' }; // treat as NOP
  }
  // ISB: 1101010100_0_00_011_0011_CRm_1_10_11111
  if (bits(insn, 21, 12) === 0b0000110011 && op2 === 0b110 && rt === 0b11111) {
    return { kind: 'nop' }; // treat as NOP
  }

  // MRS: 1101010100_1_1_o0_op1_CRn_CRm_op2_Rt
  const L = bit(insn, 21); // 1=MRS (read), 0=MSR (write)
  const o0 = bit(insn, 19);
  const op1 = bits(insn, 18, 16);
  const crn = bits(insn, 15, 12);

  // Identify system register
  const sysregId = (o0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2;

  // Known system registers
  const TPIDR_EL0    = (1 << 14) | (3 << 11) | (13 << 7) | (0 << 3) | 2; // S3_3_C13_C0_2
  const NZCV_SYSREG  = (1 << 14) | (3 << 11) | (4 << 7)  | (2 << 3) | 0; // S3_3_C4_C2_0
  const FPCR         = (1 << 14) | (3 << 11) | (4 << 7)  | (4 << 3) | 0; // S3_3_C4_C4_0
  const CNTFRQ_EL0   = (1 << 14) | (3 << 11) | (14 << 7) | (0 << 3) | 0; // S3_3_C14_C0_0
  const CNTVCT_EL0   = (1 << 14) | (3 << 11) | (14 << 7) | (0 << 3) | 2; // S3_3_C14_C0_2

  let sysreg: string;
  switch (sysregId) {
    case TPIDR_EL0:   sysreg = 'tpidr_el0'; break;
    case NZCV_SYSREG: sysreg = 'nzcv'; break;
    case FPCR:        sysreg = 'fpcr'; break;
    case CNTFRQ_EL0:  sysreg = 'cntfrq_el0'; break;
    case CNTVCT_EL0:  sysreg = 'cntvct_el0'; break;
    default:          sysreg = `s${o0}_${op1}_c${crn}_c${crm}_${op2}`; break;
  }

  if (L) {
    return { kind: 'mrs', rt, sysreg };
  } else {
    return { kind: 'msr', rt, sysreg };
  }
}

// ── Load/Store ───────────────────────────────────────────────────────────────

function decodeLoadStore(insn: number): DecodedInsn {
  const op0 = bits(insn, 31, 28);
  const op1 = bit(insn, 26);
  const op2 = bits(insn, 24, 23);
  const op3 = bits(insn, 21, 16);
  const op4 = bits(insn, 11, 10);

  // Load literal (LDR/LDRSW): bits [29:27] = 011, bit 26 = 0
  if (bits(insn, 29, 27) === 0b011 && op1 === 0) {
    return decodeLoadLiteral(insn);
  }

  // Load/store exclusive / ordered: bits [29:28] = 00, bit 26 = 0
  if (bits(insn, 29, 28) === 0b00 && op1 === 0) {
    return decodeLoadStoreExclusive(insn);
  }

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

  // Prefetch (PRFM): size=11, opc=10, unsigned offset
  // Already handled by unsigned offset path returning 'unknown', but let's catch it explicitly
  if (bits(insn, 31, 30) === 0b11 && bits(insn, 29, 27) === 0b111 && bits(insn, 23, 22) === 0b10 && op1 === 0) {
    return { kind: 'nop' }; // PRFM is a hint, treat as NOP
  }

  // ── SIMD/FP load/store (V=1) ──
  if (op1 === 1) {
    // SIMD/FP load/store pair
    if (bits(insn, 29, 27) === 0b101) {
      return decodeSimdLoadStorePair(insn);
    }

    // SIMD/FP unsigned offset
    if (bits(insn, 29, 27) === 0b111 && bit(insn, 24) === 1) {
      return decodeSimdLoadStoreUnsigned(insn);
    }

    // SIMD/FP pre/post/unscaled
    if (bits(insn, 29, 27) === 0b111 && bit(insn, 24) === 0 && bit(insn, 21) === 0) {
      return decodeSimdLoadStorePrePost(insn);
    }
  }

  return { kind: 'unknown', raw: insn };
}

function decodeLoadLiteral(insn: number): DecodedInsn {
  const opc = bits(insn, 31, 30);
  const imm19 = bits(insn, 23, 5);
  const rt = bits(insn, 4, 0);
  const offset = signExtend(imm19, 19) * 4;

  switch (opc) {
    case 0b00: return { kind: 'ldr_lit', sf: 0, rt, offset }; // LDR (32-bit)
    case 0b01: return { kind: 'ldr_lit', sf: 1, rt, offset }; // LDR (64-bit)
    case 0b10: return { kind: 'ldrsw_lit', rt, offset };       // LDRSW
    case 0b11: return { kind: 'nop' };                         // PRFM (literal) → NOP
    default: return { kind: 'unknown', raw: insn };
  }
}

function decodeLoadStoreExclusive(insn: number): DecodedInsn {
  const size = bits(insn, 31, 30);
  const L = bit(insn, 22);
  const rs = bits(insn, 20, 16);
  const rt = bits(insn, 4, 0);
  const rn = bits(insn, 9, 5);
  const sf = (size >= 2 ? (size === 3 ? 1 : 0) : 0) as 0 | 1;

  if (L) {
    // LDXR / LDAR
    return { kind: 'ldxr', sf, rt, rn };
  } else {
    // STXR / STLR
    return { kind: 'stxr', sf, rt, rn, rs };
  }
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
  const idx = bits(insn, 11, 10); // 00=unscaled, 01=post, 10=unprivileged, 11=pre
  const mode = idx === 0b11 ? 'pre' as const : idx === 0b01 ? 'post' as const : 'offset' as const;
  const rn = bits(insn, 9, 5);
  const rt = bits(insn, 4, 0);
  const offset = signExtend(imm9, 9);
  const isLoad = (opc & 1) === 1;

  // Unprivileged (LDTR/STTR) — treat like offset for now
  if (idx === 0b10) {
    // Fall through to normal decoding with mode='offset' (no writeback)
  }

  // Sign-extending loads (pre/post)
  if (opc >= 2) {
    switch (size) {
      case 0b00: return { kind: 'ldrsb_imm', sf: (opc === 2 ? 1 : 0) as 0|1, rt, rn, offset, mode };
      case 0b01: return { kind: 'ldrsh_imm', sf: (opc === 2 ? 1 : 0) as 0|1, rt, rn, offset, mode };
      case 0b10: return { kind: 'ldrsw_imm', rt, rn, offset, mode };
    }
  }

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
  const shift = S ? size : 0;
  const extend = option === 0b011 ? 'lsl' : option === 0b010 ? 'uxtw' : option === 0b110 ? 'sxtw' : option === 0b111 ? 'sxtx' : 'lsl';

  // Sign-extending reg-offset loads
  if (opc >= 2) {
    switch (size) {
      case 0b00: return { kind: 'ldrsb_reg', sf: (opc === 2 ? 1 : 0) as 0|1, rt, rn, rm, extend, shift };
      case 0b01: return { kind: 'ldrsh_reg', sf: (opc === 2 ? 1 : 0) as 0|1, rt, rn, rm, extend, shift };
      case 0b10: return { kind: 'ldrsw_reg', rt, rn, rm, extend, shift };
    }
  }

  switch (size) {
    case 0b00:
      return isLoad
        ? { kind: 'ldrb_reg', rt, rn, rm, extend, shift }
        : { kind: 'strb_reg', rt, rn, rm, extend, shift };
    case 0b01:
      return isLoad
        ? { kind: 'ldrh_reg', rt, rn, rm, extend, shift }
        : { kind: 'strh_reg', rt, rn, rm, extend, shift };
    case 0b10:
      return isLoad
        ? { kind: 'ldr_reg', sf: 0, rt, rn, rm, extend, shift }
        : { kind: 'str_reg', sf: 0, rt, rn, rm, extend, shift };
    case 0b11:
      return isLoad
        ? { kind: 'ldr_reg', sf: 1, rt, rn, rm, extend, shift }
        : { kind: 'str_reg', sf: 1, rt, rn, rm, extend, shift };
  }

  return { kind: 'unknown', raw: insn };
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

  // Add/subtract with carry: bits [28:21] = 11010000
  if (bits(insn, 28, 21) === 0b11010000) {
    const sf = bit(insn, 31) as 0 | 1;
    const op = bit(insn, 30);    // 0=ADC, 1=SBC
    const S = bit(insn, 29);     // 0=no flags, 1=set flags
    const rm = bits(insn, 20, 16);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    return { kind: op ? 'sbc' : 'adc', sf, rd, rn, rm, setFlags: S === 1 };
  }

  // Conditional select: bits [28:21] = 11010100
  if (bits(insn, 28, 21) === 0b11010100) {
    return decodeCondSelect(insn);
  }

  // Data processing (2 source): bits [28:21] = 11010110, bit 30 = 0
  if (bits(insn, 28, 21) === 0b11010110 && bit(insn, 30) === 0) {
    return decodeDataProc2Src(insn);
  }

  // Data processing (1 source): bits [28:21] = 11010110, bit 30 = 1
  if (bits(insn, 28, 21) === 0b11010110 && bit(insn, 30) === 1) {
    return decodeDataProc1Src(insn);
  }

  // Conditional compare (immediate/register)
  if (bits(insn, 28, 21) === 0b11010010) {
    return decodeCondCompare(insn);
  }

  // Extract: bits [28:23] = 100111
  if (bits(insn, 28, 23) === 0b100111) {
    return decodeExtract(insn);
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

const EXTEND_TYPES = ['uxtb', 'uxth', 'uxtw', 'uxtx', 'sxtb', 'sxth', 'sxtw', 'sxtx'];

function decodeAddSubExtReg(insn: number): DecodedInsn {
  const sf = bit(insn, 31) as 0 | 1;
  const op = bit(insn, 30);
  const S = bit(insn, 29);
  const rm = bits(insn, 20, 16);
  const option = bits(insn, 15, 13);
  const amount = bits(insn, 12, 10);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);

  return {
    kind: op ? 'sub_ext' : 'add_ext',
    sf, rd, rn, rm, extend: EXTEND_TYPES[option], amount, setFlags: S === 1,
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

function decodeDataProc1Src(insn: number): DecodedInsn {
  const sf = bit(insn, 31) as 0 | 1;
  const opcode = bits(insn, 15, 10);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);

  switch (opcode) {
    case 0b000000: return { kind: 'rbit', sf, rd, rn };
    case 0b000001: return { kind: 'rev16', sf, rd, rn };
    case 0b000010:
      // REV32 (64-bit) or REV (32-bit)
      if (sf === 1) return { kind: 'rev32', rd, rn };
      return { kind: 'rev', sf: 0, rd, rn };
    case 0b000011:
      // REV (64-bit only)
      if (sf === 1) return { kind: 'rev', sf: 1, rd, rn };
      return { kind: 'unknown', raw: insn };
    case 0b000100: return { kind: 'clz', sf, rd, rn };
    case 0b000101: return { kind: 'cls', sf, rd, rn };
    default: return { kind: 'unknown', raw: insn };
  }
}

function decodeCondCompare(insn: number): DecodedInsn {
  const sf = bit(insn, 31) as 0 | 1;
  const op = bit(insn, 30); // 0=CCMN, 1=CCMP
  const isImm = bit(insn, 11) === 1;
  const rm_imm = bits(insn, 20, 16);
  const cond = bits(insn, 15, 12);
  const rn = bits(insn, 9, 5);
  const nzcv = bits(insn, 3, 0);

  return {
    kind: op ? 'ccmp' : 'ccmn',
    sf, rn, rm_imm, nzcv, cond, isImm,
  };
}

function decodeExtract(insn: number): DecodedInsn {
  const sf = bit(insn, 31) as 0 | 1;
  const rm = bits(insn, 20, 16);
  const imms = bits(insn, 15, 10);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);

  return { kind: 'extr', sf, rd, rn, rm, imms };
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
    case 0b001000: return { kind: 'lslv', sf, rd, rn, rm };
    case 0b001001: return { kind: 'lsrv', sf, rd, rn, rm };
    case 0b001010: return { kind: 'asrv', sf, rd, rn, rm };
    case 0b001011: return { kind: 'rorv', sf, rd, rn, rm };
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

  // SMADDL/SMSUBL (64-bit only, op31=001)
  if (sf === 1 && op31 === 0b001) {
    return o0
      ? { kind: 'smsubl', rd, rn, rm, ra }
      : { kind: 'smaddl', rd, rn, rm, ra };
  }

  // UMADDL/UMSUBL (64-bit only, op31=101)
  if (sf === 1 && op31 === 0b101) {
    return o0
      ? { kind: 'umsubl', rd, rn, rm, ra }
      : { kind: 'umaddl', rd, rn, rm, ra };
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

// ── SIMD/FP load/store decoders ─────────────────────────────────────────────

function simdRegSize(size: number, opc: number): number {
  // Returns register size in bytes: 1=B, 2=H, 4=S, 8=D, 16=Q
  if (opc === 0) return 1 << size; // B=1, H=2, S=4, D=8
  if (opc === 2 && size === 0) return 16; // Q=16
  return 0;
}

function decodeSimdLoadStoreUnsigned(insn: number): DecodedInsn {
  const size = bits(insn, 31, 30);
  const opc = bits(insn, 23, 22);
  const imm12 = bits(insn, 21, 10);
  const rn = bits(insn, 9, 5);
  const rt = bits(insn, 4, 0);
  const isLoad = (opc & 1) === 1;

  const regBytes = simdRegSize(size, opc & 0b10);
  if (regBytes === 0) return { kind: 'unknown', raw: insn };

  const offset = imm12 * regBytes;

  if (isLoad) {
    return { kind: 'ldr_simd', size: regBytes, rt, rn, offset, mode: 'offset' };
  } else {
    return { kind: 'str_simd', size: regBytes, rt, rn, offset, mode: 'offset' };
  }
}

function decodeSimdLoadStorePrePost(insn: number): DecodedInsn {
  const size = bits(insn, 31, 30);
  const opc = bits(insn, 23, 22);
  const imm9 = signExtend(bits(insn, 20, 12), 9);
  const modeCode = bits(insn, 11, 10);
  const rn = bits(insn, 9, 5);
  const rt = bits(insn, 4, 0);
  const isLoad = (opc & 1) === 1;

  const regBytes = simdRegSize(size, opc & 0b10);
  if (regBytes === 0) return { kind: 'unknown', raw: insn };

  const mode: 'offset' | 'post' | 'pre' = modeCode === 0b01 ? 'post' : modeCode === 0b11 ? 'pre' : 'offset';

  if (isLoad) {
    return { kind: 'ldr_simd', size: regBytes, rt, rn, offset: imm9, mode };
  } else {
    return { kind: 'str_simd', size: regBytes, rt, rn, offset: imm9, mode };
  }
}

function decodeSimdLoadStorePair(insn: number): DecodedInsn {
  const opc = bits(insn, 31, 30);
  const typeCode = bits(insn, 25, 23);
  const L = bit(insn, 22);
  const imm7 = signExtend(bits(insn, 21, 15), 7);
  const rt2 = bits(insn, 14, 10);
  const rn = bits(insn, 9, 5);
  const rt = bits(insn, 4, 0);

  // opc: 00=S(4B), 01=D(8B), 10=Q(16B)
  const regBytes = opc === 0 ? 4 : opc === 1 ? 8 : opc === 2 ? 16 : 0;
  if (regBytes === 0) return { kind: 'unknown', raw: insn };

  const offset = imm7 * regBytes;
  const mode: 'offset' | 'post' | 'pre' =
    typeCode === 0b001 ? 'post' :
    typeCode === 0b010 ? 'offset' :
    typeCode === 0b011 ? 'pre' : 'offset';

  if (L) {
    return { kind: 'ldp_simd', size: regBytes, rt, rt2, rn, offset, mode };
  } else {
    return { kind: 'stp_simd', size: regBytes, rt, rt2, rn, offset, mode };
  }
}

// ── SIMD/FP data processing ─────────────────────────────────────────────────

function decodeSimdFp(insn: number): DecodedInsn {
  // Advanced SIMD table lookup (TBL/TBX): 0|Q|001110|op2|0|Rm|0|len|op|0|Rn|Rd
  if (bits(insn, 28, 24) === 0b01110 && bit(insn, 29) === 0 && bit(insn, 21) === 0 &&
      bit(insn, 15) === 0 && bit(insn, 10) === 0) {
    const Q = bit(insn, 30);
    const rm = bits(insn, 20, 16);
    const len = bits(insn, 14, 13); // 0 = 1 register, 1 = 2 registers, etc.
    const op = bit(insn, 12); // 0 = TBL, 1 = TBX
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    return { kind: 'simd_tbl', Q, rd, rn, rm, len, op };
  }

  // Advanced SIMD permute (ZIP1/ZIP2/UZP1/UZP2/TRN1/TRN2): 0|Q|001110|size|0|Rm|0|opcode|10|Rn|Rd
  if (bits(insn, 28, 24) === 0b01110 && bit(insn, 21) === 0 && bits(insn, 11, 10) === 0b10) {
    const Q = bit(insn, 30);
    const size = bits(insn, 23, 22);
    const rm = bits(insn, 20, 16);
    const opcode = bits(insn, 14, 12);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    switch (opcode) {
      case 0b001: return { kind: 'simd_uzp1', Q, rd, rn, rm, size };
      case 0b010: return { kind: 'simd_trn1', Q, rd, rn, rm, size };
      case 0b011: return { kind: 'simd_zip1', Q, rd, rn, rm, size };
      case 0b101: return { kind: 'simd_uzp2', Q, rd, rn, rm, size };
      case 0b110: return { kind: 'simd_trn2', Q, rd, rn, rm, size };
      case 0b111: return { kind: 'simd_zip2', Q, rd, rn, rm, size };
    }
  }

  // Advanced SIMD two-reg misc: 0|Q|U|01110|size|10000|opcode|10|Rn|Rd
  // bit[21]=1, bits[20:17]=0000, bits[11:10]=10
  if (bits(insn, 28, 24) === 0b01110 && bit(insn, 21) === 1 &&
      bits(insn, 20, 17) === 0b0000 && bits(insn, 11, 10) === 0b10) {
    const Q = bit(insn, 30);
    const U = bit(insn, 29);
    const size = bits(insn, 23, 22);
    const opcode = bits(insn, 16, 12);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);

    // NOT (bitwise): U=1, size=00, opcode=00101
    if (U === 1 && opcode === 0b00101) return { kind: 'simd_not', Q, rd, rn };
    // NEG: U=1, opcode=01011
    if (U === 1 && opcode === 0b01011) return { kind: 'simd_neg', Q, rd, rn, size };
    // ABS: U=0, opcode=01011
    if (U === 0 && opcode === 0b01011) return { kind: 'simd_abs', Q, rd, rn, size };
    // CNT: U=0, size=00, opcode=00101
    if (U === 0 && size === 0 && opcode === 0b00101) return { kind: 'simd_cnt', Q, rd, rn };
    // CLS: U=0, opcode=00100
    if (U === 0 && opcode === 0b00100) return { kind: 'simd_cls_vec', Q, rd, rn, size };
    // CLZ: U=1, opcode=00100
    if (U === 1 && opcode === 0b00100) return { kind: 'simd_clz_vec', Q, rd, rn, size };
  }

  // Advanced SIMD three same: 0|Q|U|01110|size|1|Rm|opcode|1|Rn|Rd
  if (bits(insn, 28, 24) === 0b01110 && bit(insn, 21) === 1 && bit(insn, 10) === 1) {
    const Q = bit(insn, 30);
    const U = bit(insn, 29);
    const size = bits(insn, 23, 22);
    const rm = bits(insn, 20, 16);
    const opcode = bits(insn, 15, 11);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);

    // Bitwise: ORR, AND, BIC, EOR (size=00-11 encodes the operation, not element size)
    if (opcode === 0b00011) {
      if (U === 0 && size === 0b10) return { kind: 'simd_orr_reg', Q, rd, rn, rm };
      if (U === 0 && size === 0b00) return { kind: 'simd_and_reg', Q, rd, rn, rm };
      if (U === 0 && size === 0b01) return { kind: 'simd_bic_reg', Q, rd, rn, rm };
      if (U === 1 && size === 0b10) return { kind: 'simd_eor_reg', Q, rd, rn, rm };
    }

    // ADD/SUB (integer vector)
    if (U === 0 && opcode === 0b10000) return { kind: 'simd_add', Q, rd, rn, rm, size };
    if (U === 1 && opcode === 0b10000) return { kind: 'simd_sub', Q, rd, rn, rm, size };

    // CMEQ (register): U=1, opcode=10001
    if (U === 1 && opcode === 0b10001) return { kind: 'simd_cmeq_reg', Q, rd, rn, rm, size };

    // CMGT (signed greater): U=0, opcode=00110
    if (U === 0 && opcode === 0b00110) return { kind: 'simd_cmgt_reg', Q, rd, rn, rm, size };
    // CMHI (unsigned higher): U=1, opcode=00110
    if (U === 1 && opcode === 0b00110) return { kind: 'simd_cmhi_reg', Q, rd, rn, rm, size };
    // CMGE (signed >=): U=0, opcode=00111
    if (U === 0 && opcode === 0b00111) return { kind: 'simd_cmge_reg', Q, rd, rn, rm, size };
    // CMHS (unsigned >=): U=1, opcode=00111
    if (U === 1 && opcode === 0b00111) return { kind: 'simd_cmhs_reg', Q, rd, rn, rm, size };

    // UMAXP (unsigned max pairwise): U=1, size=XX, opcode=10100
    if (U === 1 && opcode === 0b10100) return { kind: 'simd_umaxp', Q, rd, rn, rm, size };
    // ADDP (add pairwise): U=0, opcode=10111
    if (U === 0 && opcode === 0b10111) return { kind: 'simd_addp', Q, rd, rn, rm, size };
  }

  // Advanced SIMD copy: 0|Q|op|01110000|imm5|0|imm4|1|Rn|Rd
  if (bits(insn, 28, 21) === 0b01110000 && bit(insn, 10) === 1 && bit(insn, 15) === 0) {
    return decodeSimdCopy(insn);
  }

  // Advanced SIMD modified immediate: 0|Q|op|0111100000|abc|cmode|o2|1|defgh|Rd
  if (bits(insn, 28, 19) === 0b0111100000 && bit(insn, 10) === 1) {
    return decodeSimdModifiedImm(insn);
  }

  // Advanced SIMD shift by immediate: 0|Q|U|011110|immh|immb|opcode|1|Rn|Rd
  if (bits(insn, 28, 24) === 0b01111 && bit(insn, 23) === 0 && bit(insn, 10) === 1) {
    return decodeSimdShiftImm(insn);
  }

  // Floating-point <-> integer conversions: sf|0|S|11110|ftype|1|rmode|opcode|000000|Rn|Rd
  if (bits(insn, 28, 24) === 0b11110 && bit(insn, 21) === 1 && bits(insn, 15, 10) === 0b000000) {
    return decodeFpIntConv(insn);
  }

  // Floating-point data processing (2-source): sf=0|0|S|11110|ftype|1|Rm|opcode|10|Rn|Rd
  if (bits(insn, 28, 24) === 0b11110 && bit(insn, 21) === 1 && bits(insn, 11, 10) === 0b10) {
    return decodeFpDataProc2(insn);
  }

  // Floating-point data processing (1-source): sf=0|0|S|11110|ftype|1|0000|opcode|10000|Rn|Rd
  if (bits(insn, 28, 24) === 0b11110 && bit(insn, 21) === 1 && bits(insn, 14, 10) === 0b10000) {
    return decodeFpDataProc1(insn);
  }

  // Floating-point immediate: 0|0|0|11110|ftype|1|imm8|100|00000|Rd
  if (bits(insn, 28, 24) === 0b11110 && bit(insn, 21) === 1 &&
      bits(insn, 12, 10) === 0b100 && bits(insn, 9, 5) === 0) {
    return decodeFpImm(insn);
  }

  // Floating-point compare: M|0|S|11110|ftype|1|Rm|op|1000|Rn|opcode2
  if (bits(insn, 28, 24) === 0b11110 && bit(insn, 21) === 1 &&
      bit(insn, 15) === 0 && bits(insn, 13, 10) === 0b1000) {
    return decodeFpCmp(insn);
  }

  // Floating-point data processing (3-source): M|0|S|11111|ftype|o1|Rm|o0|Ra|Rn|Rd
  if (bits(insn, 28, 24) === 0b11111) {
    return decodeFpDataProc3(insn);
  }

  return { kind: 'unknown', raw: insn };
}

// ── Advanced SIMD shift by immediate ────────────────────────────────────────

function decodeSimdShiftImm(insn: number): DecodedInsn {
  const Q = bit(insn, 30);
  const U = bit(insn, 29);
  const immh = bits(insn, 22, 19);
  const immb = bits(insn, 18, 16);
  const opcode = bits(insn, 15, 11);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);

  // Determine element size from immh
  let esize: number;
  if (immh & 0b1000) esize = 8;       // 64-bit
  else if (immh & 0b0100) esize = 4;  // 32-bit
  else if (immh & 0b0010) esize = 2;  // 16-bit
  else if (immh & 0b0001) esize = 1;  // 8-bit
  else return { kind: 'unknown', raw: insn };

  const imm = (immh << 3) | immb;

  // SSHLL/SSHLL2 (U=0, opcode=10100) / USHLL/USHLL2 (U=1, opcode=10100)
  if (opcode === 0b10100) {
    const shift = imm - (esize * 8);
    if (U === 0) {
      return { kind: 'simd_sshll', Q, rd, rn, esize, shift };
    } else {
      return { kind: 'simd_ushll', Q, rd, rn, esize, shift };
    }
  }

  return { kind: 'unknown', raw: insn };
}

// ── Floating-point <-> integer conversions ──────────────────────────────────

function decodeFpIntConv(insn: number): DecodedInsn {
  const sf = bit(insn, 31) as 0 | 1;
  const ftype = bits(insn, 23, 22);
  const rmode = bits(insn, 20, 19);
  const opcode = bits(insn, 18, 16);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);

  // FMOV between GPR and FP register
  if (rmode === 0b00 && opcode === 0b111) {
    // FMOV to FP: Sn/Dn <- Wm/Xm
    // sf=0, ftype=00 → FMOV Sd, Wn  (32-bit)
    // sf=1, ftype=01 → FMOV Dd, Xn  (64-bit)
    return { kind: 'fmov_to_fp', vd: rd, rn, sf };
  }
  if (rmode === 0b00 && opcode === 0b110) {
    // FMOV from FP: Wn/Xn <- Sm/Dm
    // sf=0, ftype=00 → FMOV Wd, Sn  (32-bit)
    // sf=1, ftype=01 → FMOV Xd, Dn  (64-bit)
    return { kind: 'fmov_from_fp', rd, vn: rn, sf };
  }
  // FMOV to top half of 128-bit register: FMOV Vd.D[1], Xn
  if (rmode === 0b01 && opcode === 0b111 && sf === 1 && ftype === 0b10) {
    return { kind: 'fmov_top_to_fp', vd: rd, rn };
  }
  // FMOV from top half: FMOV Xd, Vn.D[1]
  if (rmode === 0b01 && opcode === 0b110 && sf === 1 && ftype === 0b10) {
    return { kind: 'fmov_top_from_fp', rd, vn: rn };
  }

  // SCVTF: signed integer to FP
  if (rmode === 0b00 && opcode === 0b010) {
    return { kind: 'scvtf_int', vd: rd, rn, sf, ftype };
  }
  // UCVTF: unsigned integer to FP
  if (rmode === 0b00 && opcode === 0b011) {
    return { kind: 'ucvtf_int', vd: rd, rn, sf, ftype };
  }
  // FCVTZS: FP to signed integer, round toward zero
  if (rmode === 0b11 && opcode === 0b000) {
    return { kind: 'fcvtzs_int', rd, vn: rn, sf, ftype };
  }
  // FCVTZU: FP to unsigned integer, round toward zero
  if (rmode === 0b11 && opcode === 0b001) {
    return { kind: 'fcvtzu_int', rd, vn: rn, sf, ftype };
  }

  return { kind: 'unknown', raw: insn };
}

// ── Floating-point data processing (2-source) ──────────────────────────────

function decodeFpDataProc2(insn: number): DecodedInsn {
  const ftype = bits(insn, 23, 22);
  const rm = bits(insn, 20, 16);
  const opcode = bits(insn, 15, 12);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);

  switch (opcode) {
    case 0b0000: return { kind: 'fmul', vd: rd, vn: rn, vm: rm, ftype };
    case 0b0001: return { kind: 'fdiv', vd: rd, vn: rn, vm: rm, ftype };
    case 0b0010: return { kind: 'fadd', vd: rd, vn: rn, vm: rm, ftype };
    case 0b0011: return { kind: 'fsub', vd: rd, vn: rn, vm: rm, ftype };
  }

  return { kind: 'unknown', raw: insn };
}

// ── Floating-point data processing (1-source) ──────────────────────────────

function decodeFpDataProc1(insn: number): DecodedInsn {
  const ftype = bits(insn, 23, 22);
  const opcode = bits(insn, 20, 15);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);

  switch (opcode) {
    case 0b000000: return { kind: 'fmov_fp', vd: rd, vn: rn, ftype };
    case 0b000001: return { kind: 'fabs', vd: rd, vn: rn, ftype };
    case 0b000010: return { kind: 'fneg', vd: rd, vn: rn, ftype };
    case 0b000100: // FCVT - convert between FP precisions
      return { kind: 'fcvt', vd: rd, vn: rn, ftype, destFtype: bits(insn, 16, 15) };
  }

  return { kind: 'unknown', raw: insn };
}

// ── Floating-point immediate ────────────────────────────────────────────────

function vfpExpandImm(imm8: number, ftype: number): bigint {
  const sign = (imm8 >> 7) & 1;
  const b = (imm8 >> 6) & 1;
  const bcd = (imm8 >> 4) & 3;
  const efgh = imm8 & 0xf;

  if (ftype === 1) {
    // Double precision: 64-bit
    const exp = ((1 - b) << 10) | (b ? 0x3FC : 0) | bcd; // NOT(b):Replicate(b,8):cd
    const frac = efgh << 48; // efgh:Zeros(48)
    // Build the 64-bit pattern
    return (BigInt(sign) << 63n) | (BigInt(exp) << 52n) | BigInt(frac);
  } else {
    // Single precision: 32-bit
    const exp = ((1 - b) << 7) | (b ? 0x7C : 0) | bcd; // NOT(b):Replicate(b,5):cd
    const frac = efgh << 19; // efgh:Zeros(19)
    return BigInt(((sign << 31) | (exp << 23) | frac) >>> 0);
  }
}

function decodeFpImm(insn: number): DecodedInsn {
  const ftype = bits(insn, 23, 22);
  const imm8 = bits(insn, 20, 13);
  const rd = bits(insn, 4, 0);

  const fpBits = vfpExpandImm(imm8, ftype);
  return { kind: 'fmov_imm', vd: rd, ftype, bits: fpBits };
}

// ── Floating-point compare ──────────────────────────────────────────────────

function decodeFpCmp(insn: number): DecodedInsn {
  const ftype = bits(insn, 23, 22);
  const rm = bits(insn, 20, 16);
  const rn = bits(insn, 9, 5);
  const opcode2 = bits(insn, 4, 0);

  // opcode2[3] distinguishes compare with register vs #0.0
  const withZero = (opcode2 & 0b01000) !== 0;
  // opcode2[4] distinguishes FCMP vs FCMPE (we treat them the same — no FP exceptions)
  return { kind: 'fcmp', vn: rn, vm: rm, ftype, withZero };
}

// ── Floating-point data processing (3-source) ──────────────────────────────

function decodeFpDataProc3(insn: number): DecodedInsn {
  const ftype = bits(insn, 23, 22);
  const o1 = bit(insn, 21);
  const rm = bits(insn, 20, 16);
  const o0 = bit(insn, 15);
  const ra = bits(insn, 14, 10);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);

  // o1=0,o0=0: FMADD; o1=0,o0=1: FMSUB; o1=1,o0=0: FNMADD; o1=1,o0=1: FNMSUB
  const variant = (o1 << 1) | o0;
  switch (variant) {
    case 0: return { kind: 'fmadd',  vd: rd, vn: rn, vm: rm, va: ra, ftype };
    case 1: return { kind: 'fmsub',  vd: rd, vn: rn, vm: rm, va: ra, ftype };
    case 2: return { kind: 'fnmadd', vd: rd, vn: rn, vm: rm, va: ra, ftype };
    case 3: return { kind: 'fnmsub', vd: rd, vn: rn, vm: rm, va: ra, ftype };
  }
  return { kind: 'unknown', raw: insn };
}

function decodeSimdCopy(insn: number): DecodedInsn {
  const Q = bit(insn, 30);
  const op = bit(insn, 29);
  const imm5 = bits(insn, 20, 16);
  const imm4 = bits(insn, 14, 11);
  const rn = bits(insn, 9, 5);
  const rd = bits(insn, 4, 0);

  if (op === 0) {
    if (imm4 === 0b0000) {
      // DUP (element): Vd.T, Vn.Ts[index]
      const { esize, index } = decodeSimdIndex(imm5);
      return { kind: 'simd_dup_elem', Q, rd, rn, esize, index };
    }
    if (imm4 === 0b0001) {
      // DUP (general): Vd.T, Wn/Xn
      const { esize } = decodeSimdIndex(imm5);
      return { kind: 'simd_dup_gpr', Q, rd, rn, esize };
    }
    if (imm4 === 0b0011) {
      // INS (general): Vd.Ts[index], Wn/Xn — insert GPR into vector element
      const { esize, index } = decodeSimdIndex(imm5);
      return { kind: 'simd_ins_gpr', rd, rn, esize, index };
    }
    if (imm4 === 0b0101) {
      // SMOV: Wd/Xd, Vn.Ts[index]
      const { esize, index } = decodeSimdIndex(imm5);
      return { kind: 'simd_umov', Q, rd, rn, esize, index }; // TODO: sign-extend
    }
    if (imm4 === 0b0111) {
      // UMOV: Wd/Xd, Vn.Ts[index]
      const { esize, index } = decodeSimdIndex(imm5);
      return { kind: 'simd_umov', Q, rd, rn, esize, index };
    }
  }
  if (op === 1) {
    if (imm4 === 0b0000) {
      // INS (element): Vd.Ts[index1], Vn.Ts[index2]
      return { kind: 'unknown', raw: insn }; // TODO: implement
    }
  }

  return { kind: 'unknown', raw: insn };
}

function decodeSimdIndex(imm5: number): { esize: number; index: number } {
  if (imm5 & 0b00001) return { esize: 1, index: (imm5 >> 1) & 0xF }; // B
  if (imm5 & 0b00010) return { esize: 2, index: (imm5 >> 2) & 0x7 }; // H
  if (imm5 & 0b00100) return { esize: 4, index: (imm5 >> 3) & 0x3 }; // S
  if (imm5 & 0b01000) return { esize: 8, index: (imm5 >> 4) & 0x1 }; // D
  return { esize: 0, index: 0 };
}

function decodeSimdModifiedImm(insn: number): DecodedInsn {
  const Q = bit(insn, 30);
  const op = bit(insn, 29);
  const a = bit(insn, 18);
  const b = bit(insn, 17);
  const c = bit(insn, 16);
  const cmode = bits(insn, 15, 12);
  const d = bit(insn, 9);
  const e = bit(insn, 8);
  const f = bit(insn, 7);
  const g = bit(insn, 6);
  const h = bit(insn, 5);
  const rd = bits(insn, 4, 0);

  const imm8 = (a << 7) | (b << 6) | (c << 5) | (d << 4) | (e << 3) | (f << 2) | (g << 1) | h;

  return { kind: 'simd_movi', Q, rd, imm8, cmode, op };
}
