"use strict";
(() => {
  // src/types.ts
  function isBranch(insn) {
    switch (insn.kind) {
      case "b":
      case "bl":
      case "b_cond":
      case "cbz":
      case "cbnz":
      case "tbz":
      case "tbnz":
      case "br":
      case "blr":
      case "ret":
      case "svc":
      case "brk":
        return true;
      default:
        return false;
    }
  }
  var REG_OFFSET = 0;
  var SP_OFFSET = 248;
  var PC_OFFSET = 256;
  var NZCV_OFFSET = 264;
  var ITER_OFFSET = 272;
  var TPIDR_OFFSET = 280;
  var V_OFFSET = 512;
  var GUEST_MEM_BASE = 65536;
  function regOffset(r) {
    if (r === 31) return SP_OFFSET;
    return REG_OFFSET + r * 8;
  }
  function vregOffset(r) {
    return V_OFFSET + r * 16;
  }
  var BLOCKTYPE_VOID = 64;
  var HALT_PC = 0xDEAD0000n;
  var SVC_PC = 0xDEAD0001n;

  // src/cpu.ts
  var CpuState = class {
    memory;
    _view;
    _buffer;
    constructor(memory2) {
      this.memory = memory2;
      this._buffer = memory2.buffer;
      this._view = new DataView(this._buffer);
    }
    /** Get a DataView that's always backed by the current memory buffer */
    get view() {
      if (this._buffer !== this.memory.buffer) {
        this._buffer = this.memory.buffer;
        this._view = new DataView(this._buffer);
      }
      return this._view;
    }
    getReg(r) {
      if (r === 31) return 0n;
      return this.view.getBigUint64(regOffset(r), true);
    }
    setReg(r, value) {
      if (r === 31) return;
      this.view.setBigUint64(regOffset(r), value, true);
    }
    getSP() {
      return this.view.getBigUint64(SP_OFFSET, true);
    }
    setSP(value) {
      this.view.setBigUint64(SP_OFFSET, value, true);
    }
    getPC() {
      return this.view.getBigUint64(PC_OFFSET, true);
    }
    setPC(value) {
      this.view.setBigUint64(PC_OFFSET, value, true);
    }
    getNZCV() {
      return this.view.getBigUint64(NZCV_OFFSET, true);
    }
    setNZCV(value) {
      this.view.setBigUint64(NZCV_OFFSET, value, true);
    }
    reset() {
      for (let i = 0; i < 31; i++) this.setReg(i, 0n);
      this.setSP(0n);
      this.setPC(0n);
      this.setNZCV(0n);
      for (let i = 0; i < 32 * 16; i += 8) {
        this.view.setBigUint64(V_OFFSET + i, 0n, true);
      }
    }
  };

  // src/decode.ts
  function bits(insn, hi, lo) {
    return insn >>> lo & (1 << hi - lo + 1) - 1;
  }
  function bit(insn, pos) {
    return insn >>> pos & 1;
  }
  function signExtend(value, bitWidth) {
    const shift = 32 - bitWidth;
    return value << shift >> shift;
  }
  var SHIFT_TYPES = ["lsl", "lsr", "asr", "ror"];
  function decodeLogicalImm(N, immr, imms, is64) {
    const len = 31 - Math.clz32(N << 6 | ~imms & 63);
    if (len < 1) return null;
    const size = 1 << len;
    const levels = size - 1;
    const S = imms & levels;
    const R = immr & levels;
    if (S === levels) return null;
    const welem = (1n << BigInt(S + 1)) - 1n;
    const mask = (1n << BigInt(size)) - 1n;
    const rotated = (welem >> BigInt(R) | welem << BigInt(size - R)) & mask;
    let result = 0n;
    for (let i = 0; i < 64; i += size) {
      result |= rotated << BigInt(i);
    }
    if (!is64) {
      result &= 0xFFFFFFFFn;
    }
    return result;
  }
  function decode(insn) {
    const op0 = bits(insn, 28, 25);
    switch (op0) {
      // Data processing — immediate
      case 8:
      case 9:
        return decodeDataProcImm(insn);
      // Branches, exception, system
      case 10:
      case 11:
        return decodeBranches(insn);
      // Loads and stores
      case 4:
      case 12:
      case 6:
      case 14:
        return decodeLoadStore(insn);
      // Data processing — register
      case 5:
      case 13:
        return decodeDataProcReg(insn);
      // Data processing — SIMD & FP
      case 7:
      case 15:
        return decodeSimdFp(insn);
      default:
        return { kind: "unknown", raw: insn };
    }
  }
  function decodeDataProcImm(insn) {
    const op0 = bits(insn, 25, 23);
    switch (op0) {
      case 2:
        return decodeAddSubImm(insn);
      case 4:
        return decodeLogicalImmInsn(insn);
      case 5:
        return decodeMoveWide(insn);
      case 6:
        return decodeBitfield(insn);
      case 7:
        return decodeExtract(insn);
      case 0:
      case 1:
        return decodePCRel(insn);
      default:
        return { kind: "unknown", raw: insn };
    }
  }
  function decodeAddSubImm(insn) {
    const sf = bit(insn, 31);
    const op = bit(insn, 30);
    const S = bit(insn, 29);
    const sh = bit(insn, 22);
    const imm12 = bits(insn, 21, 10);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    const imm = sh ? imm12 << 12 : imm12;
    return {
      kind: op ? "sub_imm" : "add_imm",
      sf,
      rd,
      rn,
      imm,
      setFlags: S === 1
    };
  }
  function decodeLogicalImmInsn(insn) {
    const sf = bit(insn, 31);
    const opc = bits(insn, 30, 29);
    const N = bit(insn, 22);
    const immr = bits(insn, 21, 16);
    const imms = bits(insn, 15, 10);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    const imm = decodeLogicalImm(N, immr, imms, sf === 1);
    if (imm === null) return { kind: "unknown", raw: insn };
    switch (opc) {
      case 0:
        return { kind: "and_imm", sf, rd, rn, imm, setFlags: false };
      case 1:
        return { kind: "orr_imm", sf, rd, rn, imm };
      case 2:
        return { kind: "eor_imm", sf, rd, rn, imm };
      case 3:
        return { kind: "and_imm", sf, rd, rn, imm, setFlags: true };
      // ANDS
      default:
        return { kind: "unknown", raw: insn };
    }
  }
  function decodeMoveWide(insn) {
    const sf = bit(insn, 31);
    const opc = bits(insn, 30, 29);
    const hw = bits(insn, 22, 21);
    const imm16 = bits(insn, 20, 5);
    const rd = bits(insn, 4, 0);
    switch (opc) {
      case 0:
        return { kind: "movn", sf, rd, imm16, hw };
      case 2:
        return { kind: "movz", sf, rd, imm16, hw };
      case 3:
        return { kind: "movk", sf, rd, imm16, hw };
      default:
        return { kind: "unknown", raw: insn };
    }
  }
  function decodeBitfield(insn) {
    const sf = bit(insn, 31);
    const opc = bits(insn, 30, 29);
    const immr = bits(insn, 21, 16);
    const imms = bits(insn, 15, 10);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    switch (opc) {
      case 0:
        return { kind: "sbfm", sf, rd, rn, immr, imms };
      case 1:
        return { kind: "bfm", sf, rd, rn, immr, imms };
      case 2:
        return { kind: "ubfm", sf, rd, rn, immr, imms };
      default:
        return { kind: "unknown", raw: insn };
    }
  }
  function decodePCRel(insn) {
    const op = bit(insn, 31);
    const immlo = bits(insn, 30, 29);
    const immhi = bits(insn, 23, 5);
    const rd = bits(insn, 4, 0);
    const imm = signExtend(immhi << 2 | immlo, 21);
    if (op === 0) {
      return { kind: "adr", rd, imm };
    } else {
      return { kind: "adrp", rd, imm: imm << 12 };
    }
  }
  function decodeBranches(insn) {
    const op0 = bits(insn, 31, 29);
    const op1 = bits(insn, 25, 22);
    if (bits(insn, 30, 26) === 5 || bits(insn, 30, 26) === 5) {
      const op = bit(insn, 31);
      const imm26 = bits(insn, 25, 0);
      const offset = signExtend(imm26, 26) * 4;
      return op ? { kind: "bl", offset } : { kind: "b", offset };
    }
    if (bits(insn, 31, 25) === 42 && bit(insn, 4) === 0) {
      const imm19 = bits(insn, 23, 5);
      const cond = bits(insn, 3, 0);
      const offset = signExtend(imm19, 19) * 4;
      return { kind: "b_cond", cond, offset };
    }
    if (bits(insn, 30, 25) === 26) {
      const sf = bit(insn, 31);
      const op = bit(insn, 24);
      const imm19 = bits(insn, 23, 5);
      const rt = bits(insn, 4, 0);
      const offset = signExtend(imm19, 19) * 4;
      return op ? { kind: "cbnz", sf, rt, offset } : { kind: "cbz", sf, rt, offset };
    }
    if (bits(insn, 30, 25) === 27) {
      const b5 = bit(insn, 31);
      const op = bit(insn, 24);
      const b40 = bits(insn, 23, 19);
      const imm14 = bits(insn, 18, 5);
      const rt = bits(insn, 4, 0);
      const bitPos = b5 << 5 | b40;
      const offset = signExtend(imm14, 14) * 4;
      return op ? { kind: "tbnz", rt, bit: bitPos, offset } : { kind: "tbz", rt, bit: bitPos, offset };
    }
    if (bits(insn, 31, 25) === 107) {
      const opc = bits(insn, 24, 21);
      const rn = bits(insn, 9, 5);
      switch (opc) {
        case 0:
          return { kind: "br", rn };
        case 1:
          return { kind: "blr", rn };
        case 2:
          return { kind: "ret", rn };
      }
    }
    if (bits(insn, 31, 24) === 212) {
      const opc = bits(insn, 23, 21);
      const imm16 = bits(insn, 20, 5);
      if (opc === 0 && bits(insn, 4, 0) === 1) {
        return { kind: "svc", imm16 };
      }
      if (opc === 1 && bits(insn, 4, 0) === 0) {
        return { kind: "brk", imm16 };
      }
    }
    if (bits(insn, 31, 22) === 852) {
      return decodeSystem(insn);
    }
    return { kind: "unknown", raw: insn };
  }
  function decodeSystem(insn) {
    if (insn === 3573751839) return { kind: "nop" };
    const crm = bits(insn, 11, 8);
    const op2 = bits(insn, 7, 5);
    const rt = bits(insn, 4, 0);
    if (bits(insn, 21, 12) === 51 && op2 === 5 && rt === 31) {
      return { kind: "nop" };
    }
    if (bits(insn, 21, 12) === 51 && op2 === 4 && rt === 31) {
      return { kind: "nop" };
    }
    if (bits(insn, 21, 12) === 51 && op2 === 6 && rt === 31) {
      return { kind: "nop" };
    }
    const L = bit(insn, 21);
    const o0 = bit(insn, 19);
    const op1 = bits(insn, 18, 16);
    const crn = bits(insn, 15, 12);
    const sysregId = o0 << 14 | op1 << 11 | crn << 7 | crm << 3 | op2;
    const TPIDR_EL0 = 1 << 14 | 3 << 11 | 13 << 7 | 0 << 3 | 2;
    const NZCV_SYSREG = 1 << 14 | 3 << 11 | 4 << 7 | 2 << 3 | 0;
    const FPCR = 1 << 14 | 3 << 11 | 4 << 7 | 4 << 3 | 0;
    const CNTFRQ_EL0 = 1 << 14 | 3 << 11 | 14 << 7 | 0 << 3 | 0;
    const CNTVCT_EL0 = 1 << 14 | 3 << 11 | 14 << 7 | 0 << 3 | 2;
    let sysreg;
    switch (sysregId) {
      case TPIDR_EL0:
        sysreg = "tpidr_el0";
        break;
      case NZCV_SYSREG:
        sysreg = "nzcv";
        break;
      case FPCR:
        sysreg = "fpcr";
        break;
      case CNTFRQ_EL0:
        sysreg = "cntfrq_el0";
        break;
      case CNTVCT_EL0:
        sysreg = "cntvct_el0";
        break;
      default:
        sysreg = `s${o0}_${op1}_c${crn}_c${crm}_${op2}`;
        break;
    }
    if (L) {
      return { kind: "mrs", rt, sysreg };
    } else {
      return { kind: "msr", rt, sysreg };
    }
  }
  function decodeLoadStore(insn) {
    const op0 = bits(insn, 31, 28);
    const op1 = bit(insn, 26);
    const op2 = bits(insn, 24, 23);
    const op3 = bits(insn, 21, 16);
    const op4 = bits(insn, 11, 10);
    if (bits(insn, 29, 27) === 3 && op1 === 0) {
      return decodeLoadLiteral(insn);
    }
    if (bits(insn, 29, 28) === 0 && op1 === 0) {
      return decodeLoadStoreExclusive(insn);
    }
    if (bits(insn, 29, 27) === 5 && op1 === 0) {
      return decodeLoadStorePair(insn);
    }
    if (bits(insn, 29, 27) === 7 && bit(insn, 24) === 1 && op1 === 0) {
      return decodeLoadStoreUnsignedOff(insn);
    }
    if (bits(insn, 29, 27) === 7 && bit(insn, 24) === 0 && op1 === 0 && bit(insn, 21) === 0) {
      return decodeLoadStorePrePost(insn);
    }
    if (bits(insn, 29, 27) === 7 && bit(insn, 24) === 0 && op1 === 0 && bit(insn, 21) === 1) {
      return decodeLoadStoreRegOff(insn);
    }
    if (bits(insn, 31, 30) === 3 && bits(insn, 29, 27) === 7 && bits(insn, 23, 22) === 2 && op1 === 0) {
      return { kind: "nop" };
    }
    if (op1 === 1) {
      if (bits(insn, 29, 27) === 5) {
        return decodeSimdLoadStorePair(insn);
      }
      if (bits(insn, 29, 27) === 7 && bit(insn, 24) === 1) {
        return decodeSimdLoadStoreUnsigned(insn);
      }
      if (bits(insn, 29, 27) === 7 && bit(insn, 24) === 0 && bit(insn, 21) === 0) {
        return decodeSimdLoadStorePrePost(insn);
      }
    }
    return { kind: "unknown", raw: insn };
  }
  function decodeLoadLiteral(insn) {
    const opc = bits(insn, 31, 30);
    const imm19 = bits(insn, 23, 5);
    const rt = bits(insn, 4, 0);
    const offset = signExtend(imm19, 19) * 4;
    switch (opc) {
      case 0:
        return { kind: "ldr_lit", sf: 0, rt, offset };
      // LDR (32-bit)
      case 1:
        return { kind: "ldr_lit", sf: 1, rt, offset };
      // LDR (64-bit)
      case 2:
        return { kind: "ldrsw_lit", rt, offset };
      // LDRSW
      case 3:
        return { kind: "nop" };
      // PRFM (literal) → NOP
      default:
        return { kind: "unknown", raw: insn };
    }
  }
  function decodeLoadStoreExclusive(insn) {
    const size = bits(insn, 31, 30);
    const L = bit(insn, 22);
    const rs = bits(insn, 20, 16);
    const rt = bits(insn, 4, 0);
    const rn = bits(insn, 9, 5);
    const sf = size >= 2 ? size === 3 ? 1 : 0 : 0;
    if (L) {
      return { kind: "ldxr", sf, rt, rn };
    } else {
      return { kind: "stxr", sf, rt, rn, rs };
    }
  }
  function decodeLoadStoreUnsignedOff(insn) {
    const size = bits(insn, 31, 30);
    const opc = bits(insn, 23, 22);
    const imm12 = bits(insn, 21, 10);
    const rn = bits(insn, 9, 5);
    const rt = bits(insn, 4, 0);
    const scale = size;
    const offset = imm12 << scale;
    const isLoad = (opc & 1) === 1;
    if (opc >= 2) {
      switch (size) {
        case 0:
          return { kind: "ldrsb_imm", sf: opc === 2 ? 1 : 0, rt, rn, offset: imm12, mode: "offset" };
        case 1:
          return { kind: "ldrsh_imm", sf: opc === 2 ? 1 : 0, rt, rn, offset: imm12 << 1, mode: "offset" };
        case 2:
          return { kind: "ldrsw_imm", rt, rn, offset: imm12 << 2, mode: "offset" };
      }
    }
    switch (size) {
      case 0:
        return isLoad ? { kind: "ldrb_imm", rt, rn, offset, mode: "offset" } : { kind: "strb_imm", rt, rn, offset, mode: "offset" };
      case 1:
        return isLoad ? { kind: "ldrh_imm", rt, rn, offset, mode: "offset" } : { kind: "strh_imm", rt, rn, offset, mode: "offset" };
      case 2:
        return isLoad ? { kind: "ldr_imm", sf: 0, rt, rn, offset, mode: "offset" } : { kind: "str_imm", sf: 0, rt, rn, offset, mode: "offset" };
      case 3:
        return isLoad ? { kind: "ldr_imm", sf: 1, rt, rn, offset, mode: "offset" } : { kind: "str_imm", sf: 1, rt, rn, offset, mode: "offset" };
    }
    return { kind: "unknown", raw: insn };
  }
  function decodeLoadStorePrePost(insn) {
    const size = bits(insn, 31, 30);
    const opc = bits(insn, 23, 22);
    const imm9 = bits(insn, 20, 12);
    const idx = bits(insn, 11, 10);
    const mode = idx === 3 ? "pre" : idx === 1 ? "post" : "offset";
    const rn = bits(insn, 9, 5);
    const rt = bits(insn, 4, 0);
    const offset = signExtend(imm9, 9);
    const isLoad = (opc & 1) === 1;
    if (idx === 2) {
    }
    if (opc >= 2) {
      switch (size) {
        case 0:
          return { kind: "ldrsb_imm", sf: opc === 2 ? 1 : 0, rt, rn, offset, mode };
        case 1:
          return { kind: "ldrsh_imm", sf: opc === 2 ? 1 : 0, rt, rn, offset, mode };
        case 2:
          return { kind: "ldrsw_imm", rt, rn, offset, mode };
      }
    }
    switch (size) {
      case 2:
        return isLoad ? { kind: "ldr_imm", sf: 0, rt, rn, offset, mode } : { kind: "str_imm", sf: 0, rt, rn, offset, mode };
      case 3:
        return isLoad ? { kind: "ldr_imm", sf: 1, rt, rn, offset, mode } : { kind: "str_imm", sf: 1, rt, rn, offset, mode };
      case 0:
        return isLoad ? { kind: "ldrb_imm", rt, rn, offset, mode } : { kind: "strb_imm", rt, rn, offset, mode };
      case 1:
        return isLoad ? { kind: "ldrh_imm", rt, rn, offset, mode } : { kind: "strh_imm", rt, rn, offset, mode };
    }
    return { kind: "unknown", raw: insn };
  }
  function decodeLoadStoreRegOff(insn) {
    const size = bits(insn, 31, 30);
    const opc = bits(insn, 23, 22);
    const rm = bits(insn, 20, 16);
    const option = bits(insn, 15, 13);
    const S = bit(insn, 12);
    const rn = bits(insn, 9, 5);
    const rt = bits(insn, 4, 0);
    const isLoad = (opc & 1) === 1;
    const shift = S ? size : 0;
    const extend = option === 3 ? "lsl" : option === 2 ? "uxtw" : option === 6 ? "sxtw" : option === 7 ? "sxtx" : "lsl";
    if (opc >= 2) {
      switch (size) {
        case 0:
          return { kind: "ldrsb_reg", sf: opc === 2 ? 1 : 0, rt, rn, rm, extend, shift };
        case 1:
          return { kind: "ldrsh_reg", sf: opc === 2 ? 1 : 0, rt, rn, rm, extend, shift };
        case 2:
          return { kind: "ldrsw_reg", rt, rn, rm, extend, shift };
      }
    }
    switch (size) {
      case 0:
        return isLoad ? { kind: "ldrb_reg", rt, rn, rm, extend, shift } : { kind: "strb_reg", rt, rn, rm, extend, shift };
      case 1:
        return isLoad ? { kind: "ldrh_reg", rt, rn, rm, extend, shift } : { kind: "strh_reg", rt, rn, rm, extend, shift };
      case 2:
        return isLoad ? { kind: "ldr_reg", sf: 0, rt, rn, rm, extend, shift } : { kind: "str_reg", sf: 0, rt, rn, rm, extend, shift };
      case 3:
        return isLoad ? { kind: "ldr_reg", sf: 1, rt, rn, rm, extend, shift } : { kind: "str_reg", sf: 1, rt, rn, rm, extend, shift };
    }
    return { kind: "unknown", raw: insn };
  }
  function decodeLoadStorePair(insn) {
    const opc = bits(insn, 31, 30);
    const type = bits(insn, 24, 23);
    const L = bit(insn, 22);
    const imm7 = bits(insn, 21, 15);
    const rt2 = bits(insn, 14, 10);
    const rn = bits(insn, 9, 5);
    const rt = bits(insn, 4, 0);
    const sf = opc === 2 ? 1 : 0;
    const scale = sf ? 3 : 2;
    const offset = signExtend(imm7, 7) << scale;
    const mode = type === 1 ? "post" : type === 3 ? "pre" : "offset";
    return L ? { kind: "ldp", sf, rt, rt2, rn, offset, mode } : { kind: "stp", sf, rt, rt2, rn, offset, mode };
  }
  function decodeDataProcReg(insn) {
    const op0 = bit(insn, 30);
    const op1 = bit(insn, 28);
    const op2 = bits(insn, 24, 21);
    if (bits(insn, 28, 24) === 10) {
      return decodeLogicalReg(insn);
    }
    if (bits(insn, 28, 24) === 11 && bit(insn, 21) === 0) {
      return decodeAddSubReg(insn);
    }
    if (bits(insn, 28, 24) === 11 && bit(insn, 21) === 1) {
      return decodeAddSubExtReg(insn);
    }
    if (bits(insn, 28, 21) === 208) {
      const sf = bit(insn, 31);
      const op = bit(insn, 30);
      const S = bit(insn, 29);
      const rm = bits(insn, 20, 16);
      const rn = bits(insn, 9, 5);
      const rd = bits(insn, 4, 0);
      return { kind: op ? "sbc" : "adc", sf, rd, rn, rm, setFlags: S === 1 };
    }
    if (bits(insn, 28, 21) === 212) {
      return decodeCondSelect(insn);
    }
    if (bits(insn, 28, 21) === 214 && bit(insn, 30) === 0) {
      return decodeDataProc2Src(insn);
    }
    if (bits(insn, 28, 21) === 214 && bit(insn, 30) === 1) {
      return decodeDataProc1Src(insn);
    }
    if (bits(insn, 28, 21) === 210) {
      return decodeCondCompare(insn);
    }
    if (bits(insn, 28, 23) === 39) {
      return decodeExtract(insn);
    }
    if (bits(insn, 28, 24) === 27) {
      return decodeDataProc3Src(insn);
    }
    return { kind: "unknown", raw: insn };
  }
  function decodeAddSubReg(insn) {
    const sf = bit(insn, 31);
    const op = bit(insn, 30);
    const S = bit(insn, 29);
    const shift = SHIFT_TYPES[bits(insn, 23, 22)];
    const rm = bits(insn, 20, 16);
    const amount = bits(insn, 15, 10);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    return {
      kind: op ? "sub_reg" : "add_reg",
      sf,
      rd,
      rn,
      rm,
      shift,
      amount,
      setFlags: S === 1
    };
  }
  var EXTEND_TYPES = ["uxtb", "uxth", "uxtw", "uxtx", "sxtb", "sxth", "sxtw", "sxtx"];
  function decodeAddSubExtReg(insn) {
    const sf = bit(insn, 31);
    const op = bit(insn, 30);
    const S = bit(insn, 29);
    const rm = bits(insn, 20, 16);
    const option = bits(insn, 15, 13);
    const amount = bits(insn, 12, 10);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    return {
      kind: op ? "sub_ext" : "add_ext",
      sf,
      rd,
      rn,
      rm,
      extend: EXTEND_TYPES[option],
      amount,
      setFlags: S === 1
    };
  }
  function decodeLogicalReg(insn) {
    const sf = bit(insn, 31);
    const opc = bits(insn, 30, 29);
    const shift = SHIFT_TYPES[bits(insn, 23, 22)];
    const N = bit(insn, 21);
    const rm = bits(insn, 20, 16);
    const amount = bits(insn, 15, 10);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    switch (opc) {
      case 0:
        return { kind: "and_reg", sf, rd, rn, rm, shift, amount, invert: N === 1, setFlags: false };
      case 1:
        return { kind: "orr_reg", sf, rd, rn, rm, shift, amount, invert: N === 1 };
      case 2:
        return { kind: "eor_reg", sf, rd, rn, rm, shift, amount, invert: N === 1 };
      case 3:
        return { kind: "and_reg", sf, rd, rn, rm, shift, amount, invert: N === 1, setFlags: true };
      default:
        return { kind: "unknown", raw: insn };
    }
  }
  function decodeCondSelect(insn) {
    const sf = bit(insn, 31);
    const op = bit(insn, 30);
    const rm = bits(insn, 20, 16);
    const cond = bits(insn, 15, 12);
    const op2 = bits(insn, 11, 10);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    const variants = {
      0: op === 0 ? "csel" : "csinv",
      1: op === 0 ? "csinc" : "csneg"
    };
    return {
      kind: "csel",
      sf,
      rd,
      rn,
      rm,
      cond,
      variant: variants[op2] ?? "csel"
    };
  }
  function decodeDataProc1Src(insn) {
    const sf = bit(insn, 31);
    const opcode = bits(insn, 15, 10);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    switch (opcode) {
      case 0:
        return { kind: "rbit", sf, rd, rn };
      case 1:
        return { kind: "rev16", sf, rd, rn };
      case 2:
        if (sf === 1) return { kind: "rev32", rd, rn };
        return { kind: "rev", sf: 0, rd, rn };
      case 3:
        if (sf === 1) return { kind: "rev", sf: 1, rd, rn };
        return { kind: "unknown", raw: insn };
      case 4:
        return { kind: "clz", sf, rd, rn };
      case 5:
        return { kind: "cls", sf, rd, rn };
      default:
        return { kind: "unknown", raw: insn };
    }
  }
  function decodeCondCompare(insn) {
    const sf = bit(insn, 31);
    const op = bit(insn, 30);
    const isImm = bit(insn, 11) === 1;
    const rm_imm = bits(insn, 20, 16);
    const cond = bits(insn, 15, 12);
    const rn = bits(insn, 9, 5);
    const nzcv = bits(insn, 3, 0);
    return {
      kind: op ? "ccmp" : "ccmn",
      sf,
      rn,
      rm_imm,
      nzcv,
      cond,
      isImm
    };
  }
  function decodeExtract(insn) {
    const sf = bit(insn, 31);
    const rm = bits(insn, 20, 16);
    const imms = bits(insn, 15, 10);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    return { kind: "extr", sf, rd, rn, rm, imms };
  }
  function decodeDataProc2Src(insn) {
    const sf = bit(insn, 31);
    const rm = bits(insn, 20, 16);
    const opcode = bits(insn, 15, 10);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    switch (opcode) {
      case 2:
        return { kind: "udiv", sf, rd, rn, rm };
      case 3:
        return { kind: "sdiv", sf, rd, rn, rm };
      case 8:
        return { kind: "lslv", sf, rd, rn, rm };
      case 9:
        return { kind: "lsrv", sf, rd, rn, rm };
      case 10:
        return { kind: "asrv", sf, rd, rn, rm };
      case 11:
        return { kind: "rorv", sf, rd, rn, rm };
      default:
        return { kind: "unknown", raw: insn };
    }
  }
  function decodeDataProc3Src(insn) {
    const sf = bit(insn, 31);
    const op31 = bits(insn, 23, 21);
    const rm = bits(insn, 20, 16);
    const o0 = bit(insn, 15);
    const ra = bits(insn, 14, 10);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    if (op31 === 0) {
      return o0 ? { kind: "msub", sf, rd, rn, rm, ra } : { kind: "madd", sf, rd, rn, rm, ra };
    }
    if (sf === 1 && op31 === 1) {
      return o0 ? { kind: "smsubl", rd, rn, rm, ra } : { kind: "smaddl", rd, rn, rm, ra };
    }
    if (sf === 1 && op31 === 5) {
      return o0 ? { kind: "umsubl", rd, rn, rm, ra } : { kind: "umaddl", rd, rn, rm, ra };
    }
    if (sf === 1 && op31 === 2 && o0 === 0) {
      return { kind: "smulh", rd, rn, rm };
    }
    if (sf === 1 && op31 === 6 && o0 === 0) {
      return { kind: "umulh", rd, rn, rm };
    }
    return { kind: "unknown", raw: insn };
  }
  function simdRegSize(size, opc) {
    if (opc === 0) return 1 << size;
    if (opc === 2 && size === 0) return 16;
    return 0;
  }
  function decodeSimdLoadStoreUnsigned(insn) {
    const size = bits(insn, 31, 30);
    const opc = bits(insn, 23, 22);
    const imm12 = bits(insn, 21, 10);
    const rn = bits(insn, 9, 5);
    const rt = bits(insn, 4, 0);
    const isLoad = (opc & 1) === 1;
    const regBytes = simdRegSize(size, opc & 2);
    if (regBytes === 0) return { kind: "unknown", raw: insn };
    const offset = imm12 * regBytes;
    if (isLoad) {
      return { kind: "ldr_simd", size: regBytes, rt, rn, offset, mode: "offset" };
    } else {
      return { kind: "str_simd", size: regBytes, rt, rn, offset, mode: "offset" };
    }
  }
  function decodeSimdLoadStorePrePost(insn) {
    const size = bits(insn, 31, 30);
    const opc = bits(insn, 23, 22);
    const imm9 = signExtend(bits(insn, 20, 12), 9);
    const modeCode = bits(insn, 11, 10);
    const rn = bits(insn, 9, 5);
    const rt = bits(insn, 4, 0);
    const isLoad = (opc & 1) === 1;
    const regBytes = simdRegSize(size, opc & 2);
    if (regBytes === 0) return { kind: "unknown", raw: insn };
    const mode = modeCode === 1 ? "post" : modeCode === 3 ? "pre" : "offset";
    if (isLoad) {
      return { kind: "ldr_simd", size: regBytes, rt, rn, offset: imm9, mode };
    } else {
      return { kind: "str_simd", size: regBytes, rt, rn, offset: imm9, mode };
    }
  }
  function decodeSimdLoadStorePair(insn) {
    const opc = bits(insn, 31, 30);
    const typeCode = bits(insn, 25, 23);
    const L = bit(insn, 22);
    const imm7 = signExtend(bits(insn, 21, 15), 7);
    const rt2 = bits(insn, 14, 10);
    const rn = bits(insn, 9, 5);
    const rt = bits(insn, 4, 0);
    const regBytes = opc === 0 ? 4 : opc === 1 ? 8 : opc === 2 ? 16 : 0;
    if (regBytes === 0) return { kind: "unknown", raw: insn };
    const offset = imm7 * regBytes;
    const mode = typeCode === 1 ? "post" : typeCode === 2 ? "offset" : typeCode === 3 ? "pre" : "offset";
    if (L) {
      return { kind: "ldp_simd", size: regBytes, rt, rt2, rn, offset, mode };
    } else {
      return { kind: "stp_simd", size: regBytes, rt, rt2, rn, offset, mode };
    }
  }
  function decodeSimdFp(insn) {
    if (bits(insn, 28, 24) === 14 && bit(insn, 29) === 0 && bit(insn, 21) === 0 && bit(insn, 15) === 0 && bit(insn, 10) === 0) {
      const Q = bit(insn, 30);
      const rm = bits(insn, 20, 16);
      const len = bits(insn, 14, 13);
      const op = bit(insn, 12);
      const rn = bits(insn, 9, 5);
      const rd = bits(insn, 4, 0);
      return { kind: "simd_tbl", Q, rd, rn, rm, len, op };
    }
    if (bits(insn, 28, 24) === 14 && bit(insn, 21) === 0 && bits(insn, 11, 10) === 2) {
      const Q = bit(insn, 30);
      const size = bits(insn, 23, 22);
      const rm = bits(insn, 20, 16);
      const opcode = bits(insn, 14, 12);
      const rn = bits(insn, 9, 5);
      const rd = bits(insn, 4, 0);
      switch (opcode) {
        case 1:
          return { kind: "simd_uzp1", Q, rd, rn, rm, size };
        case 2:
          return { kind: "simd_trn1", Q, rd, rn, rm, size };
        case 3:
          return { kind: "simd_zip1", Q, rd, rn, rm, size };
        case 5:
          return { kind: "simd_uzp2", Q, rd, rn, rm, size };
        case 6:
          return { kind: "simd_trn2", Q, rd, rn, rm, size };
        case 7:
          return { kind: "simd_zip2", Q, rd, rn, rm, size };
      }
    }
    if (bits(insn, 28, 24) === 14 && bit(insn, 21) === 1 && bits(insn, 20, 17) === 0 && bits(insn, 11, 10) === 2) {
      const Q = bit(insn, 30);
      const U = bit(insn, 29);
      const size = bits(insn, 23, 22);
      const opcode = bits(insn, 16, 12);
      const rn = bits(insn, 9, 5);
      const rd = bits(insn, 4, 0);
      if (U === 1 && opcode === 5) return { kind: "simd_not", Q, rd, rn };
      if (U === 1 && opcode === 11) return { kind: "simd_neg", Q, rd, rn, size };
      if (U === 0 && opcode === 11) return { kind: "simd_abs", Q, rd, rn, size };
      if (U === 0 && size === 0 && opcode === 5) return { kind: "simd_cnt", Q, rd, rn };
      if (U === 0 && opcode === 4) return { kind: "simd_cls_vec", Q, rd, rn, size };
      if (U === 1 && opcode === 4) return { kind: "simd_clz_vec", Q, rd, rn, size };
    }
    if (bits(insn, 28, 24) === 14 && bit(insn, 21) === 1 && bit(insn, 10) === 1) {
      const Q = bit(insn, 30);
      const U = bit(insn, 29);
      const size = bits(insn, 23, 22);
      const rm = bits(insn, 20, 16);
      const opcode = bits(insn, 15, 11);
      const rn = bits(insn, 9, 5);
      const rd = bits(insn, 4, 0);
      if (opcode === 3) {
        if (U === 0 && size === 2) return { kind: "simd_orr_reg", Q, rd, rn, rm };
        if (U === 0 && size === 0) return { kind: "simd_and_reg", Q, rd, rn, rm };
        if (U === 0 && size === 1) return { kind: "simd_bic_reg", Q, rd, rn, rm };
        if (U === 1 && size === 2) return { kind: "simd_eor_reg", Q, rd, rn, rm };
      }
      if (U === 0 && opcode === 16) return { kind: "simd_add", Q, rd, rn, rm, size };
      if (U === 1 && opcode === 16) return { kind: "simd_sub", Q, rd, rn, rm, size };
      if (U === 1 && opcode === 17) return { kind: "simd_cmeq_reg", Q, rd, rn, rm, size };
      if (U === 0 && opcode === 6) return { kind: "simd_cmgt_reg", Q, rd, rn, rm, size };
      if (U === 1 && opcode === 6) return { kind: "simd_cmhi_reg", Q, rd, rn, rm, size };
      if (U === 0 && opcode === 7) return { kind: "simd_cmge_reg", Q, rd, rn, rm, size };
      if (U === 1 && opcode === 7) return { kind: "simd_cmhs_reg", Q, rd, rn, rm, size };
      if (U === 1 && opcode === 20) return { kind: "simd_umaxp", Q, rd, rn, rm, size };
      if (U === 0 && opcode === 23) return { kind: "simd_addp", Q, rd, rn, rm, size };
    }
    if (bits(insn, 28, 21) === 112 && bit(insn, 10) === 1 && bit(insn, 15) === 0) {
      return decodeSimdCopy(insn);
    }
    if (bits(insn, 28, 19) === 480 && bit(insn, 10) === 1) {
      return decodeSimdModifiedImm(insn);
    }
    if (bits(insn, 28, 24) === 15 && bit(insn, 23) === 0 && bit(insn, 10) === 1) {
      return decodeSimdShiftImm(insn);
    }
    if (bits(insn, 28, 24) === 30 && bit(insn, 21) === 1 && bits(insn, 15, 10) === 0) {
      return decodeFpIntConv(insn);
    }
    if (bits(insn, 28, 24) === 30 && bit(insn, 21) === 1 && bits(insn, 11, 10) === 2) {
      return decodeFpDataProc2(insn);
    }
    if (bits(insn, 28, 24) === 30 && bit(insn, 21) === 1 && bits(insn, 14, 10) === 16) {
      return decodeFpDataProc1(insn);
    }
    if (bits(insn, 28, 24) === 30 && bit(insn, 21) === 1 && bits(insn, 12, 10) === 4 && bits(insn, 9, 5) === 0) {
      return decodeFpImm(insn);
    }
    if (bits(insn, 28, 24) === 30 && bit(insn, 21) === 1 && bit(insn, 15) === 0 && bits(insn, 13, 10) === 8) {
      return decodeFpCmp(insn);
    }
    if (bits(insn, 28, 24) === 31) {
      return decodeFpDataProc3(insn);
    }
    return { kind: "unknown", raw: insn };
  }
  function decodeSimdShiftImm(insn) {
    const Q = bit(insn, 30);
    const U = bit(insn, 29);
    const immh = bits(insn, 22, 19);
    const immb = bits(insn, 18, 16);
    const opcode = bits(insn, 15, 11);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    let esize;
    if (immh & 8) esize = 8;
    else if (immh & 4) esize = 4;
    else if (immh & 2) esize = 2;
    else if (immh & 1) esize = 1;
    else return { kind: "unknown", raw: insn };
    const imm = immh << 3 | immb;
    if (opcode === 20) {
      const shift = imm - esize * 8;
      if (U === 0) {
        return { kind: "simd_sshll", Q, rd, rn, esize, shift };
      } else {
        return { kind: "simd_ushll", Q, rd, rn, esize, shift };
      }
    }
    return { kind: "unknown", raw: insn };
  }
  function decodeFpIntConv(insn) {
    const sf = bit(insn, 31);
    const ftype = bits(insn, 23, 22);
    const rmode = bits(insn, 20, 19);
    const opcode = bits(insn, 18, 16);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    if (rmode === 0 && opcode === 7) {
      return { kind: "fmov_to_fp", vd: rd, rn, sf };
    }
    if (rmode === 0 && opcode === 6) {
      return { kind: "fmov_from_fp", rd, vn: rn, sf };
    }
    if (rmode === 1 && opcode === 7 && sf === 1 && ftype === 2) {
      return { kind: "fmov_top_to_fp", vd: rd, rn };
    }
    if (rmode === 1 && opcode === 6 && sf === 1 && ftype === 2) {
      return { kind: "fmov_top_from_fp", rd, vn: rn };
    }
    if (rmode === 0 && opcode === 2) {
      return { kind: "scvtf_int", vd: rd, rn, sf, ftype };
    }
    if (rmode === 0 && opcode === 3) {
      return { kind: "ucvtf_int", vd: rd, rn, sf, ftype };
    }
    if (rmode === 3 && opcode === 0) {
      return { kind: "fcvtzs_int", rd, vn: rn, sf, ftype };
    }
    if (rmode === 3 && opcode === 1) {
      return { kind: "fcvtzu_int", rd, vn: rn, sf, ftype };
    }
    return { kind: "unknown", raw: insn };
  }
  function decodeFpDataProc2(insn) {
    const ftype = bits(insn, 23, 22);
    const rm = bits(insn, 20, 16);
    const opcode = bits(insn, 15, 12);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    switch (opcode) {
      case 0:
        return { kind: "fmul", vd: rd, vn: rn, vm: rm, ftype };
      case 1:
        return { kind: "fdiv", vd: rd, vn: rn, vm: rm, ftype };
      case 2:
        return { kind: "fadd", vd: rd, vn: rn, vm: rm, ftype };
      case 3:
        return { kind: "fsub", vd: rd, vn: rn, vm: rm, ftype };
    }
    return { kind: "unknown", raw: insn };
  }
  function decodeFpDataProc1(insn) {
    const ftype = bits(insn, 23, 22);
    const opcode = bits(insn, 20, 15);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    switch (opcode) {
      case 0:
        return { kind: "fmov_fp", vd: rd, vn: rn, ftype };
      case 1:
        return { kind: "fabs", vd: rd, vn: rn, ftype };
      case 2:
        return { kind: "fneg", vd: rd, vn: rn, ftype };
      case 4:
        return { kind: "fcvt", vd: rd, vn: rn, ftype, destFtype: bits(insn, 16, 15) };
    }
    return { kind: "unknown", raw: insn };
  }
  function vfpExpandImm(imm8, ftype) {
    const sign = imm8 >> 7 & 1;
    const b = imm8 >> 6 & 1;
    const bcd = imm8 >> 4 & 3;
    const efgh = imm8 & 15;
    if (ftype === 1) {
      const exp = 1 - b << 10 | (b ? 1020 : 0) | bcd;
      const frac = efgh << 48;
      return BigInt(sign) << 63n | BigInt(exp) << 52n | BigInt(frac);
    } else {
      const exp = 1 - b << 7 | (b ? 124 : 0) | bcd;
      const frac = efgh << 19;
      return BigInt((sign << 31 | exp << 23 | frac) >>> 0);
    }
  }
  function decodeFpImm(insn) {
    const ftype = bits(insn, 23, 22);
    const imm8 = bits(insn, 20, 13);
    const rd = bits(insn, 4, 0);
    const fpBits = vfpExpandImm(imm8, ftype);
    return { kind: "fmov_imm", vd: rd, ftype, bits: fpBits };
  }
  function decodeFpCmp(insn) {
    const ftype = bits(insn, 23, 22);
    const rm = bits(insn, 20, 16);
    const rn = bits(insn, 9, 5);
    const opcode2 = bits(insn, 4, 0);
    const withZero = (opcode2 & 8) !== 0;
    return { kind: "fcmp", vn: rn, vm: rm, ftype, withZero };
  }
  function decodeFpDataProc3(insn) {
    const ftype = bits(insn, 23, 22);
    const o1 = bit(insn, 21);
    const rm = bits(insn, 20, 16);
    const o0 = bit(insn, 15);
    const ra = bits(insn, 14, 10);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    const variant = o1 << 1 | o0;
    switch (variant) {
      case 0:
        return { kind: "fmadd", vd: rd, vn: rn, vm: rm, va: ra, ftype };
      case 1:
        return { kind: "fmsub", vd: rd, vn: rn, vm: rm, va: ra, ftype };
      case 2:
        return { kind: "fnmadd", vd: rd, vn: rn, vm: rm, va: ra, ftype };
      case 3:
        return { kind: "fnmsub", vd: rd, vn: rn, vm: rm, va: ra, ftype };
    }
    return { kind: "unknown", raw: insn };
  }
  function decodeSimdCopy(insn) {
    const Q = bit(insn, 30);
    const op = bit(insn, 29);
    const imm5 = bits(insn, 20, 16);
    const imm4 = bits(insn, 14, 11);
    const rn = bits(insn, 9, 5);
    const rd = bits(insn, 4, 0);
    if (op === 0) {
      if (imm4 === 0) {
        const { esize, index } = decodeSimdIndex(imm5);
        return { kind: "simd_dup_elem", Q, rd, rn, esize, index };
      }
      if (imm4 === 1) {
        const { esize } = decodeSimdIndex(imm5);
        return { kind: "simd_dup_gpr", Q, rd, rn, esize };
      }
      if (imm4 === 3) {
        const { esize, index } = decodeSimdIndex(imm5);
        return { kind: "simd_ins_gpr", rd, rn, esize, index };
      }
      if (imm4 === 5) {
        const { esize, index } = decodeSimdIndex(imm5);
        return { kind: "simd_umov", Q, rd, rn, esize, index };
      }
      if (imm4 === 7) {
        const { esize, index } = decodeSimdIndex(imm5);
        return { kind: "simd_umov", Q, rd, rn, esize, index };
      }
    }
    if (op === 1) {
      if (imm4 === 0) {
        return { kind: "unknown", raw: insn };
      }
    }
    return { kind: "unknown", raw: insn };
  }
  function decodeSimdIndex(imm5) {
    if (imm5 & 1) return { esize: 1, index: imm5 >> 1 & 15 };
    if (imm5 & 2) return { esize: 2, index: imm5 >> 2 & 7 };
    if (imm5 & 4) return { esize: 4, index: imm5 >> 3 & 3 };
    if (imm5 & 8) return { esize: 8, index: imm5 >> 4 & 1 };
    return { esize: 0, index: 0 };
  }
  function decodeSimdModifiedImm(insn) {
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
    const imm8 = a << 7 | b << 6 | c << 5 | d << 4 | e << 3 | f << 2 | g << 1 | h;
    return { kind: "simd_movi", Q, rd, imm8, cmode, op };
  }

  // src/block.ts
  function detectBlock(memory2, pc) {
    const insns = [];
    const startPC = pc;
    let currentPC = pc;
    for (; ; ) {
      const memOffset = Number(currentPC) + GUEST_MEM_BASE;
      const raw = memory2.getUint32(memOffset, true);
      const insn = decode(raw);
      insns.push(insn);
      currentPC += 4n;
      if (isBranch(insn) || insn.kind === "unknown") {
        break;
      }
      if (insns.length >= 256) break;
    }
    let isSelfLoop = false;
    const lastInsn = insns[insns.length - 1];
    if (lastInsn) {
      switch (lastInsn.kind) {
        case "b": {
          const target = Number(startPC) + insns.length * 4 - 4 + lastInsn.offset;
          isSelfLoop = BigInt(target) === startPC;
          break;
        }
        case "b_cond":
        case "cbz":
        case "cbnz":
        case "tbz":
        case "tbnz": {
          const branchPC = Number(startPC) + (insns.length - 1) * 4;
          const target = branchPC + lastInsn.offset;
          isSelfLoop = BigInt(target) === startPC;
          break;
        }
      }
    }
    return {
      startPC,
      endPC: currentPC,
      insns,
      isSelfLoop
    };
  }

  // src/wasm-emit.ts
  function encodeULEB128(value) {
    const bytes = [];
    do {
      let byte = value & 127;
      value >>>= 7;
      if (value !== 0) byte |= 128;
      bytes.push(byte);
    } while (value !== 0);
    return bytes;
  }
  function encodeSLEB128(value) {
    const bytes = [];
    let more = true;
    while (more) {
      let byte = value & 127;
      value >>= 7;
      if (value === 0 && (byte & 64) === 0 || value === -1 && (byte & 64) !== 0) {
        more = false;
      } else {
        byte |= 128;
      }
      bytes.push(byte);
    }
    return bytes;
  }
  function encodeSLEB128_i64(value) {
    value = BigInt.asIntN(64, value);
    const bytes = [];
    let more = true;
    while (more) {
      let byte = Number(value & 0x7Fn);
      value >>= 7n;
      if (value === 0n && (byte & 64) === 0 || value === -1n && (byte & 64) !== 0) {
        more = false;
      } else {
        byte |= 128;
      }
      bytes.push(byte);
    }
    return bytes;
  }
  function encodeString(s) {
    const utf8 = new TextEncoder().encode(s);
    return [...encodeULEB128(utf8.length), ...utf8];
  }
  function encodeSection(id, content) {
    return [id, ...encodeULEB128(content.length), ...content];
  }
  var WasmModuleBuilder = class {
    // Section data
    types = [];
    // encoded function types
    imports = [];
    // encoded imports
    importCount = 0;
    importedFuncCount = 0;
    // function imports only (for func index space)
    funcTypeIndices = [];
    // type index per function
    exports = [];
    exportCount = 0;
    // Current function being built
    code = [];
    // instruction bytecode
    localGroups = [];
    localCount = 0;
    // total locals (params + declared)
    paramCount = 0;
    funcBodies = [];
    // finalized function bodies
    // ── Types ──────────────────────────────────────────────────────────────────
    addFuncType(params, results) {
      const encoded = [
        96,
        // function type marker
        ...encodeULEB128(params.length),
        ...params,
        ...encodeULEB128(results.length),
        ...results
      ];
      this.types.push(encoded);
      return this.types.length - 1;
    }
    // ── Imports ────────────────────────────────────────────────────────────────
    addMemoryImport(module, name, minPages, maxPages) {
      const importData = [
        ...encodeString(module),
        ...encodeString(name),
        2
        // import kind: memory
      ];
      if (maxPages !== void 0) {
        importData.push(1, ...encodeULEB128(minPages), ...encodeULEB128(maxPages));
      } else {
        importData.push(0, ...encodeULEB128(minPages));
      }
      this.imports.push(...importData);
      this.importCount++;
    }
    addFunctionImport(module, name, typeIndex) {
      const importData = [
        ...encodeString(module),
        ...encodeString(name),
        0,
        // import kind: function
        ...encodeULEB128(typeIndex)
      ];
      this.imports.push(...importData);
      const funcIdx = this.importedFuncCount;
      this.importCount++;
      this.importedFuncCount++;
      return funcIdx;
    }
    // ── Functions ──────────────────────────────────────────────────────────────
    beginFunction(typeIndex) {
      this.funcTypeIndices.push(typeIndex);
      this.code = [];
      this.localGroups = [];
      this.localCount = 0;
      const typeData = this.types[typeIndex];
      this.paramCount = typeData[1];
      this.localCount = this.paramCount;
    }
    addLocals(type, count) {
      const firstIndex = this.localCount;
      this.localGroups.push({ count, type });
      this.localCount += count;
      return firstIndex;
    }
    exportFunction(name, funcIndex) {
      const idx = funcIndex ?? this.importedFuncCount + this.funcTypeIndices.length - 1;
      this.exports.push(
        ...encodeString(name),
        0,
        ...encodeULEB128(idx)
      );
      this.exportCount++;
    }
    endFunction() {
      const localDecls = [
        ...encodeULEB128(this.localGroups.length)
      ];
      for (const g of this.localGroups) {
        localDecls.push(...encodeULEB128(g.count), g.type);
      }
      const body = [...localDecls, ...this.code, 11 /* End */];
      const bodyWithSize = [...encodeULEB128(body.length), ...body];
      this.funcBodies.push(bodyWithSize);
    }
    // ── Instruction emission ───────────────────────────────────────────────────
    // Constants
    i32_const(value) {
      this.code.push(65 /* I32Const */, ...encodeSLEB128(value));
    }
    i64_const(value) {
      this.code.push(66 /* I64Const */, ...encodeSLEB128_i64(value));
    }
    // Variables
    local_get(idx) {
      this.code.push(32 /* LocalGet */, ...encodeULEB128(idx));
    }
    local_set(idx) {
      this.code.push(33 /* LocalSet */, ...encodeULEB128(idx));
    }
    local_tee(idx) {
      this.code.push(34 /* LocalTee */, ...encodeULEB128(idx));
    }
    // Memory
    i32_load(align, offset) {
      this.code.push(40 /* I32Load */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i64_load(align, offset) {
      this.code.push(41 /* I64Load */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i32_store(align, offset) {
      this.code.push(54 /* I32Store */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i64_store(align, offset) {
      this.code.push(55 /* I64Store */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i32_load8_s(align, offset) {
      this.code.push(44 /* I32Load8S */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i32_load8_u(align, offset) {
      this.code.push(45 /* I32Load8U */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i32_load16_s(align, offset) {
      this.code.push(46 /* I32Load16S */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i32_load16_u(align, offset) {
      this.code.push(47 /* I32Load16U */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i64_load8_s(align, offset) {
      this.code.push(48 /* I64Load8S */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i64_load8_u(align, offset) {
      this.code.push(49 /* I64Load8U */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i64_load16_s(align, offset) {
      this.code.push(50 /* I64Load16S */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i64_load16_u(align, offset) {
      this.code.push(51 /* I64Load16U */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i64_load32_s(align, offset) {
      this.code.push(52 /* I64Load32S */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i64_load32_u(align, offset) {
      this.code.push(53 /* I64Load32U */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i64_store8(align, offset) {
      this.code.push(60 /* I64Store8 */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i64_store16(align, offset) {
      this.code.push(61 /* I64Store16 */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i64_store32(align, offset) {
      this.code.push(62 /* I64Store32 */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i32_store8(align, offset) {
      this.code.push(58 /* I32Store8 */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    i32_store16(align, offset) {
      this.code.push(59 /* I32Store16 */, ...encodeULEB128(align), ...encodeULEB128(offset));
    }
    // i32 comparison
    i32_eqz() {
      this.code.push(69 /* I32Eqz */);
    }
    i32_eq() {
      this.code.push(70 /* I32Eq */);
    }
    i32_ne() {
      this.code.push(71 /* I32Ne */);
    }
    // i32 arithmetic
    i32_add() {
      this.code.push(106 /* I32Add */);
    }
    i32_sub() {
      this.code.push(107 /* I32Sub */);
    }
    // i64 comparison
    i64_eqz() {
      this.code.push(80 /* I64Eqz */);
    }
    i64_eq() {
      this.code.push(81 /* I64Eq */);
    }
    i64_ne() {
      this.code.push(82 /* I64Ne */);
    }
    i64_lt_s() {
      this.code.push(83 /* I64LtS */);
    }
    i64_lt_u() {
      this.code.push(84 /* I64LtU */);
    }
    i64_gt_s() {
      this.code.push(85 /* I64GtS */);
    }
    i64_gt_u() {
      this.code.push(86 /* I64GtU */);
    }
    i64_le_s() {
      this.code.push(87 /* I64LeS */);
    }
    i64_le_u() {
      this.code.push(88 /* I64LeU */);
    }
    i64_ge_s() {
      this.code.push(89 /* I64GeS */);
    }
    i64_ge_u() {
      this.code.push(90 /* I64GeU */);
    }
    // i64 unary
    i64_clz() {
      this.code.push(121 /* I64Clz */);
    }
    i64_ctz() {
      this.code.push(122 /* I64Ctz */);
    }
    i64_popcnt() {
      this.code.push(123 /* I64Popcnt */);
    }
    // i64 arithmetic
    i64_add() {
      this.code.push(124 /* I64Add */);
    }
    i64_sub() {
      this.code.push(125 /* I64Sub */);
    }
    i64_mul() {
      this.code.push(126 /* I64Mul */);
    }
    i64_div_s() {
      this.code.push(127 /* I64DivS */);
    }
    i64_div_u() {
      this.code.push(128 /* I64DivU */);
    }
    i64_rem_s() {
      this.code.push(129 /* I64RemS */);
    }
    i64_rem_u() {
      this.code.push(130 /* I64RemU */);
    }
    i64_and() {
      this.code.push(131 /* I64And */);
    }
    i64_or() {
      this.code.push(132 /* I64Or */);
    }
    i64_xor() {
      this.code.push(133 /* I64Xor */);
    }
    i64_shl() {
      this.code.push(134 /* I64Shl */);
    }
    i64_shr_s() {
      this.code.push(135 /* I64ShrS */);
    }
    i64_shr_u() {
      this.code.push(136 /* I64ShrU */);
    }
    i64_rotl() {
      this.code.push(137 /* I64Rotl */);
    }
    i64_rotr() {
      this.code.push(138 /* I64Rotr */);
    }
    // Conversions
    i32_wrap_i64() {
      this.code.push(167 /* I32WrapI64 */);
    }
    i64_extend_i32_s() {
      this.code.push(172 /* I64ExtendI32S */);
    }
    i64_extend_i32_u() {
      this.code.push(173 /* I64ExtendI32U */);
    }
    // Control flow
    block(blocktype) {
      this.code.push(2 /* Block */, blocktype);
    }
    loop(blocktype) {
      this.code.push(3 /* Loop */, blocktype);
    }
    if_(blocktype) {
      this.code.push(4 /* If */, blocktype);
    }
    else_() {
      this.code.push(5 /* Else */);
    }
    end() {
      this.code.push(11 /* End */);
    }
    br(depth) {
      this.code.push(12 /* Br */, ...encodeULEB128(depth));
    }
    br_if(depth) {
      this.code.push(13 /* BrIf */, ...encodeULEB128(depth));
    }
    br_table(labels, defaultLabel) {
      this.code.push(14 /* BrTable */, ...encodeULEB128(labels.length));
      for (const label of labels) {
        this.code.push(...encodeULEB128(label));
      }
      this.code.push(...encodeULEB128(defaultLabel));
    }
    return_() {
      this.code.push(15 /* Return */);
    }
    unreachable() {
      this.code.push(0 /* Unreachable */);
    }
    call(funcIdx) {
      this.code.push(16 /* Call */, ...encodeULEB128(funcIdx));
    }
    drop() {
      this.code.push(26);
    }
    // ── Finalize ───────────────────────────────────────────────────────────────
    finalize() {
      const sections = [];
      sections.push(0, 97, 115, 109);
      sections.push(1, 0, 0, 0);
      if (this.types.length > 0) {
        const content = [
          ...encodeULEB128(this.types.length),
          ...this.types.flat()
        ];
        sections.push(...encodeSection(1, content));
      }
      if (this.importCount > 0) {
        const content = [
          ...encodeULEB128(this.importCount),
          ...this.imports
        ];
        sections.push(...encodeSection(2, content));
      }
      if (this.funcTypeIndices.length > 0) {
        const content = [
          ...encodeULEB128(this.funcTypeIndices.length),
          ...this.funcTypeIndices.flatMap((i) => encodeULEB128(i))
        ];
        sections.push(...encodeSection(3, content));
      }
      if (this.exportCount > 0) {
        const content = [
          ...encodeULEB128(this.exportCount),
          ...this.exports
        ];
        sections.push(...encodeSection(7, content));
      }
      if (this.funcBodies.length > 0) {
        const content = [
          ...encodeULEB128(this.funcBodies.length),
          ...this.funcBodies.flat()
        ];
        sections.push(...encodeSection(10, content));
      }
      return new Uint8Array(sections);
    }
  };

  // src/translate.ts
  var REG_COUNT = 33;
  function regLocal(r) {
    if (r >= 0 && r <= 30) return r;
    if (r === 31) return 31;
    throw new Error(`Invalid register ${r}`);
  }
  var SP_LOCAL = 31;
  var NZCV_LOCAL = 32;
  function analyzeRegUsage(insns) {
    const reads = /* @__PURE__ */ new Set();
    const writes = /* @__PURE__ */ new Set();
    for (const insn of insns) {
      switch (insn.kind) {
        case "add_imm":
        case "sub_imm":
          if (insn.rn !== 31) reads.add(regLocal(insn.rn));
          else reads.add(SP_LOCAL);
          if (insn.rd === 31) {
            if (!insn.setFlags) writes.add(SP_LOCAL);
          } else {
            writes.add(regLocal(insn.rd));
          }
          if (insn.setFlags) writes.add(NZCV_LOCAL);
          break;
        case "add_reg":
        case "sub_reg":
          addRead(reads, insn.rn);
          addRead(reads, insn.rm);
          addWrite(writes, insn.rd);
          if (insn.setFlags) writes.add(NZCV_LOCAL);
          break;
        case "adc":
        case "sbc":
          addRead(reads, insn.rn);
          addRead(reads, insn.rm);
          reads.add(NZCV_LOCAL);
          addWrite(writes, insn.rd);
          if (insn.setFlags) writes.add(NZCV_LOCAL);
          break;
        case "orr_reg":
          addRead(reads, insn.rn);
          addRead(reads, insn.rm);
          addWrite(writes, insn.rd);
          break;
        case "eor_reg":
          addRead(reads, insn.rn);
          addRead(reads, insn.rm);
          addWrite(writes, insn.rd);
          break;
        case "and_reg":
          addRead(reads, insn.rn);
          addRead(reads, insn.rm);
          addWrite(writes, insn.rd);
          if (insn.setFlags) writes.add(NZCV_LOCAL);
          break;
        case "and_imm":
          addRead(reads, insn.rn);
          if (insn.setFlags) {
            addWrite(writes, insn.rd);
            writes.add(NZCV_LOCAL);
          } else {
            if (insn.rd === 31) writes.add(SP_LOCAL);
            else addWrite(writes, insn.rd);
          }
          break;
        case "orr_imm":
        case "eor_imm":
          addRead(reads, insn.rn);
          if (insn.rd === 31) writes.add(SP_LOCAL);
          else addWrite(writes, insn.rd);
          break;
        case "movz":
          addWrite(writes, insn.rd);
          break;
        case "movk":
          addRead(reads, insn.rd);
          addWrite(writes, insn.rd);
          break;
        case "movn":
          addWrite(writes, insn.rd);
          break;
        case "madd":
        case "msub":
        case "smaddl":
        case "smsubl":
        case "umaddl":
        case "umsubl":
          addRead(reads, insn.rn);
          addRead(reads, insn.rm);
          addRead(reads, insn.ra);
          addWrite(writes, insn.rd);
          break;
        case "sdiv":
        case "udiv":
          addRead(reads, insn.rn);
          addRead(reads, insn.rm);
          addWrite(writes, insn.rd);
          break;
        case "csel":
          addRead(reads, insn.rn);
          addRead(reads, insn.rm);
          reads.add(NZCV_LOCAL);
          addWrite(writes, insn.rd);
          break;
        case "sbfm":
        case "ubfm":
          addRead(reads, insn.rn);
          addWrite(writes, insn.rd);
          break;
        case "bfm":
          addRead(reads, insn.rn);
          addRead(reads, insn.rd);
          addWrite(writes, insn.rd);
          break;
        case "b_cond":
          reads.add(NZCV_LOCAL);
          break;
        case "cbz":
        case "cbnz":
          addRead(reads, insn.rt);
          break;
        case "bl":
        case "blr":
          writes.add(regLocal(30));
          if (insn.kind === "blr") addRead(reads, insn.rn);
          break;
        case "br":
          addRead(reads, insn.rn);
          break;
        case "ret":
          addRead(reads, insn.rn);
          break;
        case "ldr_imm":
          if (insn.rn !== 31) addRead(reads, insn.rn);
          else reads.add(SP_LOCAL);
          addWrite(writes, insn.rt);
          if (insn.mode !== "offset") {
            if (insn.rn !== 31) addWrite(writes, insn.rn);
            else writes.add(SP_LOCAL);
          }
          break;
        case "str_imm":
          if (insn.rn !== 31) addRead(reads, insn.rn);
          else reads.add(SP_LOCAL);
          addRead(reads, insn.rt);
          if (insn.mode !== "offset") {
            if (insn.rn !== 31) addWrite(writes, insn.rn);
            else writes.add(SP_LOCAL);
          }
          break;
        case "ldrb_imm":
        case "ldrh_imm":
        case "ldrsw_imm":
          if (insn.rn !== 31) addRead(reads, insn.rn);
          else reads.add(SP_LOCAL);
          addWrite(writes, insn.rt);
          if (insn.mode !== "offset") {
            if (insn.rn !== 31) addWrite(writes, insn.rn);
            else writes.add(SP_LOCAL);
          }
          break;
        case "ldrsb_imm":
        case "ldrsh_imm":
          if (insn.rn !== 31) addRead(reads, insn.rn);
          else reads.add(SP_LOCAL);
          addWrite(writes, insn.rt);
          if (insn.mode !== "offset") {
            if (insn.rn !== 31) addWrite(writes, insn.rn);
            else writes.add(SP_LOCAL);
          }
          break;
        case "strb_imm":
        case "strh_imm":
          if (insn.rn !== 31) addRead(reads, insn.rn);
          else reads.add(SP_LOCAL);
          addRead(reads, insn.rt);
          if (insn.mode !== "offset") {
            if (insn.rn !== 31) addWrite(writes, insn.rn);
            else writes.add(SP_LOCAL);
          }
          break;
        case "ldr_reg":
          if (insn.rn !== 31) addRead(reads, insn.rn);
          else reads.add(SP_LOCAL);
          addRead(reads, insn.rm);
          addWrite(writes, insn.rt);
          break;
        case "str_reg":
          if (insn.rn !== 31) addRead(reads, insn.rn);
          else reads.add(SP_LOCAL);
          addRead(reads, insn.rm);
          addRead(reads, insn.rt);
          break;
        case "ldp":
          if (insn.rn !== 31) addRead(reads, insn.rn);
          else reads.add(SP_LOCAL);
          addWrite(writes, insn.rt);
          addWrite(writes, insn.rt2);
          if (insn.mode !== "offset") {
            if (insn.rn !== 31) addWrite(writes, insn.rn);
            else writes.add(SP_LOCAL);
          }
          break;
        case "stp":
          if (insn.rn !== 31) addRead(reads, insn.rn);
          else reads.add(SP_LOCAL);
          addRead(reads, insn.rt);
          addRead(reads, insn.rt2);
          if (insn.mode !== "offset") {
            if (insn.rn !== 31) addWrite(writes, insn.rn);
            else writes.add(SP_LOCAL);
          }
          break;
        case "adr":
        case "adrp":
          addWrite(writes, insn.rd);
          break;
        case "smulh":
        case "umulh":
          addRead(reads, insn.rn);
          addRead(reads, insn.rm);
          addWrite(writes, insn.rd);
          break;
        case "tbz":
        case "tbnz":
          addRead(reads, insn.rt);
          break;
        case "clz":
        case "cls":
        case "rbit":
        case "rev":
        case "rev16":
          addRead(reads, insn.rn);
          addWrite(writes, insn.rd);
          break;
        case "rev32":
          addRead(reads, insn.rn);
          addWrite(writes, insn.rd);
          break;
        case "extr":
          addRead(reads, insn.rn);
          addRead(reads, insn.rm);
          addWrite(writes, insn.rd);
          break;
        case "ccmp":
        case "ccmn":
          addRead(reads, insn.rn);
          if (!insn.isImm) addRead(reads, insn.rm_imm);
          reads.add(NZCV_LOCAL);
          writes.add(NZCV_LOCAL);
          break;
        case "ldr_lit":
        case "ldrsw_lit":
          addWrite(writes, insn.rt);
          break;
        case "ldrb_reg":
        case "ldrh_reg":
        case "ldrsw_reg":
          if (insn.rn !== 31) addRead(reads, insn.rn);
          else reads.add(SP_LOCAL);
          addRead(reads, insn.rm);
          addWrite(writes, insn.rt);
          break;
        case "ldrsb_reg":
        case "ldrsh_reg":
          if (insn.rn !== 31) addRead(reads, insn.rn);
          else reads.add(SP_LOCAL);
          addRead(reads, insn.rm);
          addWrite(writes, insn.rt);
          break;
        case "strb_reg":
        case "strh_reg":
          if (insn.rn !== 31) addRead(reads, insn.rn);
          else reads.add(SP_LOCAL);
          addRead(reads, insn.rm);
          addRead(reads, insn.rt);
          break;
        case "ldxr":
          if (insn.rn !== 31) addRead(reads, insn.rn);
          else reads.add(SP_LOCAL);
          addWrite(writes, insn.rt);
          break;
        case "stxr":
          if (insn.rn !== 31) addRead(reads, insn.rn);
          else reads.add(SP_LOCAL);
          addRead(reads, insn.rt);
          addWrite(writes, insn.rs);
          break;
        case "mrs":
          addWrite(writes, insn.rt);
          if (insn.sysreg === "nzcv") reads.add(NZCV_LOCAL);
          break;
        case "msr":
          addRead(reads, insn.rt);
          if (insn.sysreg === "nzcv") writes.add(NZCV_LOCAL);
          break;
        case "add_ext":
        case "sub_ext":
          if (insn.rn !== 31) addRead(reads, insn.rn);
          else reads.add(SP_LOCAL);
          addRead(reads, insn.rm);
          if (insn.setFlags) {
            reads.add(NZCV_LOCAL);
            writes.add(NZCV_LOCAL);
          }
          if (insn.rd === 31) {
            if (!insn.setFlags) writes.add(SP_LOCAL);
          } else addWrite(writes, insn.rd);
          break;
        // Variable shifts
        case "lslv":
        case "lsrv":
        case "asrv":
        case "rorv":
          addRead(reads, insn.rn);
          addRead(reads, insn.rm);
          addWrite(writes, insn.rd);
          break;
        // SIMD load/store: reads base register (SP or GPR), may write back
        case "str_simd":
        case "ldr_simd":
          if (insn.rn === 31) reads.add(SP_LOCAL);
          else addRead(reads, insn.rn);
          if (insn.mode !== "offset") {
            if (insn.rn === 31) writes.add(SP_LOCAL);
            else addWrite(writes, insn.rn);
          }
          break;
        case "stp_simd":
        case "ldp_simd":
          if (insn.rn === 31) reads.add(SP_LOCAL);
          else addRead(reads, insn.rn);
          if (insn.mode !== "offset") {
            if (insn.rn === 31) writes.add(SP_LOCAL);
            else addWrite(writes, insn.rn);
          }
          break;
        // SIMD data ops: read/write V regs directly from memory, may read GPRs
        case "simd_dup_gpr":
          addRead(reads, insn.rn);
          break;
        case "simd_umov":
          addWrite(writes, insn.rd);
          break;
        case "simd_ins_gpr":
          addRead(reads, insn.rn);
          break;
        case "simd_dup_elem":
        case "simd_movi":
          break;
        // FMOV GPR → FP: reads GPR
        case "fmov_to_fp":
        case "fmov_top_to_fp":
          addRead(reads, insn.rn);
          break;
        // FMOV FP → GPR: writes GPR
        case "fmov_from_fp":
        case "fmov_top_from_fp":
          addWrite(writes, insn.rd);
          break;
        // FP-only / SIMD-only: no GPR involvement
        case "fmov_fp":
        case "fmul":
        case "fdiv":
        case "fadd":
        case "fsub":
        case "fabs":
        case "fneg":
        case "fcvt":
        case "fmov_imm":
        case "fmadd":
        case "fmsub":
        case "fnmadd":
        case "fnmsub":
        case "simd_orr_reg":
        case "simd_and_reg":
        case "simd_bic_reg":
        case "simd_eor_reg":
        case "simd_add":
        case "simd_sub":
        case "simd_cmeq_reg":
        case "simd_cmgt_reg":
        case "simd_cmhi_reg":
        case "simd_cmge_reg":
        case "simd_cmhs_reg":
        case "simd_umaxp":
        case "simd_addp":
        case "simd_tbl":
        case "simd_zip1":
        case "simd_zip2":
        case "simd_uzp1":
        case "simd_uzp2":
        case "simd_trn1":
        case "simd_trn2":
        case "simd_not":
        case "simd_neg":
        case "simd_abs":
        case "simd_cnt":
        case "simd_cls_vec":
        case "simd_clz_vec":
        case "simd_sshll":
        case "simd_ushll":
        case "simd_scvtf":
        case "simd_ucvtf":
        case "simd_fcvtzs":
        case "simd_fcvtzu":
          break;
        // FP compare: writes NZCV
        case "fcmp":
          writes.add(NZCV_LOCAL);
          break;
        // FP int conversions: read/write GPRs
        case "scvtf_int":
        case "ucvtf_int":
          addRead(reads, insn.rn);
          break;
        case "fcvtzs_int":
        case "fcvtzu_int":
          addWrite(writes, insn.rd);
          break;
        default:
          break;
      }
    }
    return { reads, writes };
  }
  function addRead(set, r) {
    if (r !== 31) set.add(regLocal(r));
  }
  function addWrite(set, r) {
    if (r !== 31) set.add(regLocal(r));
  }
  function setupHelperImports(b) {
    const helperType = b.addFuncType([126 /* I64 */, 126 /* I64 */], [126 /* I64 */]);
    const smulhIdx = b.addFunctionImport("env", "smulh", helperType);
    const umulhIdx = b.addFunctionImport("env", "umulh", helperType);
    const f64_mulIdx = b.addFunctionImport("env", "f64_mul", helperType);
    const f64_divIdx = b.addFunctionImport("env", "f64_div", helperType);
    const f64_addIdx = b.addFunctionImport("env", "f64_add", helperType);
    const f64_subIdx = b.addFunctionImport("env", "f64_sub", helperType);
    const unaryType = b.addFuncType([126 /* I64 */], [126 /* I64 */]);
    const f64_absIdx = b.addFunctionImport("env", "f64_abs", unaryType);
    const f64_negIdx = b.addFunctionImport("env", "f64_neg", unaryType);
    const f64_from_s64Idx = b.addFunctionImport("env", "f64_from_s64", unaryType);
    const f64_from_u64Idx = b.addFunctionImport("env", "f64_from_u64", unaryType);
    const f64_from_s32Idx = b.addFunctionImport("env", "f64_from_s32", unaryType);
    const f64_from_u32Idx = b.addFunctionImport("env", "f64_from_u32", unaryType);
    const f64_to_s64Idx = b.addFunctionImport("env", "f64_to_s64", unaryType);
    const f64_to_u64Idx = b.addFunctionImport("env", "f64_to_u64", unaryType);
    const f64_to_s32Idx = b.addFunctionImport("env", "f64_to_s32", unaryType);
    const f64_to_u32Idx = b.addFunctionImport("env", "f64_to_u32", unaryType);
    const f32_to_f64Idx = b.addFunctionImport("env", "f32_to_f64", unaryType);
    const f64_to_f32Idx = b.addFunctionImport("env", "f64_to_f32", unaryType);
    const f64_cmpIdx = b.addFunctionImport("env", "f64_cmp", helperType);
    const simd_opIdx = b.addFunctionImport("env", "simd_op", helperType);
    return {
      smulhIdx,
      umulhIdx,
      f64_mulIdx,
      f64_divIdx,
      f64_addIdx,
      f64_subIdx,
      f64_absIdx,
      f64_negIdx,
      f64_from_s64Idx,
      f64_from_u64Idx,
      f64_from_s32Idx,
      f64_from_u32Idx,
      f64_to_s64Idx,
      f64_to_u64Idx,
      f64_to_s32Idx,
      f64_to_u32Idx,
      f32_to_f64Idx,
      f64_to_f32Idx,
      f64_cmpIdx,
      simd_opIdx
    };
  }
  var _fpBuf = new ArrayBuffer(8);
  var _f64View = new Float64Array(_fpBuf);
  var _u8View = new Uint8Array(_fpBuf);
  var _f32View = new Float32Array(_fpBuf);
  var _i64View = new BigInt64Array(_fpBuf);
  var _u64View = new BigUint64Array(_fpBuf);
  function f64FromBits(bits2) {
    _u64View[0] = BigInt.asUintN(64, bits2);
    return _f64View[0];
  }
  function f64ToBits(val) {
    _f64View[0] = val;
    return _u64View[0];
  }
  function f32FromBits(bits2) {
    new DataView(_fpBuf).setUint32(0, bits2, true);
    return _f32View[0];
  }
  function f32ToBits(val) {
    _f32View[0] = val;
    return new DataView(_fpBuf).getUint32(0, true);
  }
  var helperFunctions = {
    smulh(a, b) {
      const sa = BigInt.asIntN(64, a);
      const sb = BigInt.asIntN(64, b);
      return BigInt.asUintN(64, sa * sb >> 64n);
    },
    umulh(a, b) {
      const ua = BigInt.asUintN(64, a);
      const ub = BigInt.asUintN(64, b);
      return BigInt.asUintN(64, ua * ub >> 64n);
    },
    // FP 2-source: operate on f64 bit patterns (or f32 widened to f64 bits)
    f64_mul(a, b) {
      return f64ToBits(f64FromBits(a) * f64FromBits(b));
    },
    f64_div(a, b) {
      return f64ToBits(f64FromBits(a) / f64FromBits(b));
    },
    f64_add(a, b) {
      return f64ToBits(f64FromBits(a) + f64FromBits(b));
    },
    f64_sub(a, b) {
      return f64ToBits(f64FromBits(a) - f64FromBits(b));
    },
    // FP 1-source
    f64_abs(a) {
      return f64ToBits(Math.abs(f64FromBits(a)));
    },
    f64_neg(a) {
      return f64ToBits(-f64FromBits(a));
    },
    // Int → FP
    f64_from_s64(a) {
      return f64ToBits(Number(BigInt.asIntN(64, a)));
    },
    f64_from_u64(a) {
      return f64ToBits(Number(BigInt.asUintN(64, a)));
    },
    f64_from_s32(a) {
      return f64ToBits(Number(BigInt.asIntN(32, a)));
    },
    f64_from_u32(a) {
      return f64ToBits(Number(BigInt.asUintN(32, a)));
    },
    // FP → Int (round toward zero)
    f64_to_s64(a) {
      return BigInt.asUintN(64, BigInt(Math.trunc(f64FromBits(a))));
    },
    f64_to_u64(a) {
      const v = Math.trunc(f64FromBits(a));
      return v < 0 ? 0n : BigInt.asUintN(64, BigInt(v));
    },
    f64_to_s32(a) {
      return BigInt.asUintN(64, BigInt(Math.trunc(f64FromBits(a))) & 0xFFFFFFFFn);
    },
    f64_to_u32(a) {
      const v = Math.trunc(f64FromBits(a));
      return v < 0 ? 0n : BigInt(v) & 0xFFFFFFFFn;
    },
    // Precision conversions
    f32_to_f64(a) {
      const f = f32FromBits(Number(a & 0xFFFFFFFFn));
      return f64ToBits(f);
    },
    f64_to_f32(a) {
      const f = f64FromBits(a);
      return BigInt(f32ToBits(Math.fround(f)));
    },
    // FP compare: returns ARM NZCV bits (N=bit3, Z=bit2, C=bit1, V=bit0)
    f64_cmp(a, b) {
      const fa = f64FromBits(a);
      const fb = f64FromBits(b);
      if (isNaN(fa) || isNaN(fb)) return 3n;
      if (fa === fb) return 6n;
      if (fa < fb) return 8n;
      return 2n;
    }
  };
  var SIMD_ADD = 1;
  var SIMD_SUB = 2;
  var SIMD_CMEQ = 10;
  var SIMD_CMGT = 11;
  var SIMD_CMHI = 12;
  var SIMD_CMGE = 13;
  var SIMD_CMHS = 14;
  var SIMD_UMAXP = 20;
  var SIMD_ADDP = 21;
  var SIMD_TBL = 30;
  var SIMD_ZIP1 = 40;
  var SIMD_ZIP2 = 41;
  var SIMD_UZP1 = 42;
  var SIMD_UZP2 = 43;
  var SIMD_TRN1 = 44;
  var SIMD_TRN2 = 45;
  function createSimdOp(memory2) {
    return function simd_op(packed, _extra) {
      const op = Number(packed >> 56n & 0xFFn);
      const Q = Number(packed >> 48n & 0xFFn);
      const size = Number(packed >> 40n & 0xFFn);
      const extra = Number(packed >> 32n & 0xFFn);
      const rd = Number(packed >> 24n & 0xFFn);
      const rn = Number(packed >> 16n & 0xFFn);
      const rm = Number(packed >> 8n & 0xFFn);
      const buf = new Uint8Array(memory2.buffer);
      const view = new DataView(memory2.buffer);
      const vn = V_OFFSET + rn * 16;
      const vm = V_OFFSET + rm * 16;
      const vd = V_OFFSET + rd * 16;
      const bytes = Q ? 16 : 8;
      const esize = 1 << size;
      const a = new Uint8Array(16), b_ = new Uint8Array(16), dst = new Uint8Array(16);
      a.set(buf.subarray(vn, vn + bytes));
      b_.set(buf.subarray(vm, vm + bytes));
      const aView = new DataView(a.buffer);
      const bView = new DataView(b_.buffer);
      const dView = new DataView(dst.buffer);
      const elems = bytes / esize;
      switch (op) {
        case SIMD_ADD:
          for (let i = 0; i < elems; i++) {
            if (esize === 1) dView.setUint8(i, aView.getUint8(i) + bView.getUint8(i) & 255);
            else if (esize === 2) dView.setUint16(i * 2, aView.getUint16(i * 2, true) + bView.getUint16(i * 2, true) & 65535, true);
            else if (esize === 4) dView.setUint32(i * 4, aView.getUint32(i * 4, true) + bView.getUint32(i * 4, true) >>> 0, true);
            else dView.setBigUint64(i * 8, BigInt.asUintN(64, aView.getBigUint64(i * 8, true) + bView.getBigUint64(i * 8, true)), true);
          }
          break;
        case SIMD_SUB:
          for (let i = 0; i < elems; i++) {
            if (esize === 1) dView.setUint8(i, aView.getUint8(i) - bView.getUint8(i) & 255);
            else if (esize === 2) dView.setUint16(i * 2, aView.getUint16(i * 2, true) - bView.getUint16(i * 2, true) & 65535, true);
            else if (esize === 4) dView.setUint32(i * 4, aView.getUint32(i * 4, true) - bView.getUint32(i * 4, true) >>> 0, true);
            else dView.setBigUint64(i * 8, BigInt.asUintN(64, aView.getBigUint64(i * 8, true) - bView.getBigUint64(i * 8, true)), true);
          }
          break;
        case SIMD_CMEQ:
          for (let i = 0; i < elems; i++) {
            let eq;
            if (esize === 1) eq = aView.getUint8(i) === bView.getUint8(i);
            else if (esize === 2) eq = aView.getUint16(i * 2, true) === bView.getUint16(i * 2, true);
            else if (esize === 4) eq = aView.getUint32(i * 4, true) === bView.getUint32(i * 4, true);
            else eq = aView.getBigUint64(i * 8, true) === bView.getBigUint64(i * 8, true);
            if (esize === 1) dView.setUint8(i, eq ? 255 : 0);
            else if (esize === 2) dView.setUint16(i * 2, eq ? 65535 : 0, true);
            else if (esize === 4) dView.setUint32(i * 4, eq ? 4294967295 : 0, true);
            else dView.setBigUint64(i * 8, eq ? 0xFFFFFFFFFFFFFFFFn : 0n, true);
          }
          break;
        case SIMD_CMGT:
          for (let i = 0; i < elems; i++) {
            let gt;
            if (esize === 1) gt = aView.getInt8(i) > bView.getInt8(i);
            else if (esize === 2) gt = aView.getInt16(i * 2, true) > bView.getInt16(i * 2, true);
            else if (esize === 4) gt = aView.getInt32(i * 4, true) > bView.getInt32(i * 4, true);
            else gt = aView.getBigInt64(i * 8, true) > bView.getBigInt64(i * 8, true);
            if (esize <= 4) dView.setUint32(i * esize, gt ? esize === 1 ? 255 : esize === 2 ? 65535 : 4294967295 : 0, true);
            else dView.setBigUint64(i * 8, gt ? 0xFFFFFFFFFFFFFFFFn : 0n, true);
          }
          break;
        case SIMD_CMHI:
          for (let i = 0; i < elems; i++) {
            let hi;
            if (esize === 1) hi = aView.getUint8(i) > bView.getUint8(i);
            else if (esize === 2) hi = aView.getUint16(i * 2, true) > bView.getUint16(i * 2, true);
            else if (esize === 4) hi = aView.getUint32(i * 4, true) > bView.getUint32(i * 4, true);
            else hi = aView.getBigUint64(i * 8, true) > bView.getBigUint64(i * 8, true);
            if (esize <= 4) {
              const mask = esize === 1 ? 255 : esize === 2 ? 65535 : 4294967295;
              if (esize === 1) dView.setUint8(i, hi ? 255 : 0);
              else if (esize === 2) dView.setUint16(i * 2, hi ? 65535 : 0, true);
              else dView.setUint32(i * 4, hi ? 4294967295 : 0, true);
            } else dView.setBigUint64(i * 8, hi ? 0xFFFFFFFFFFFFFFFFn : 0n, true);
          }
          break;
        case SIMD_CMGE:
        // signed >=
        case SIMD_CMHS:
          for (let i = 0; i < elems; i++) {
            let ge;
            if (op === SIMD_CMGE) {
              if (esize === 1) ge = aView.getInt8(i) >= bView.getInt8(i);
              else if (esize === 2) ge = aView.getInt16(i * 2, true) >= bView.getInt16(i * 2, true);
              else if (esize === 4) ge = aView.getInt32(i * 4, true) >= bView.getInt32(i * 4, true);
              else ge = aView.getBigInt64(i * 8, true) >= bView.getBigInt64(i * 8, true);
            } else {
              if (esize === 1) ge = aView.getUint8(i) >= bView.getUint8(i);
              else if (esize === 2) ge = aView.getUint16(i * 2, true) >= bView.getUint16(i * 2, true);
              else if (esize === 4) ge = aView.getUint32(i * 4, true) >= bView.getUint32(i * 4, true);
              else ge = aView.getBigUint64(i * 8, true) >= bView.getBigUint64(i * 8, true);
            }
            if (esize === 1) dView.setUint8(i, ge ? 255 : 0);
            else if (esize === 2) dView.setUint16(i * 2, ge ? 65535 : 0, true);
            else if (esize === 4) dView.setUint32(i * 4, ge ? 4294967295 : 0, true);
            else dView.setBigUint64(i * 8, ge ? 0xFFFFFFFFFFFFFFFFn : 0n, true);
          }
          break;
        case SIMD_UMAXP: {
          const total = elems * 2;
          const all = new Uint8Array(32);
          all.set(a.subarray(0, bytes), 0);
          all.set(b_.subarray(0, bytes), bytes);
          const allView = new DataView(all.buffer);
          for (let i = 0; i < elems; i++) {
            const off1 = i * 2 * esize;
            const off2 = off1 + esize;
            let v1, v2, mx;
            if (esize === 1) {
              v1 = allView.getUint8(off1);
              v2 = allView.getUint8(off2);
              mx = v1 > v2 ? v1 : v2;
              dView.setUint8(i, mx);
            } else if (esize === 2) {
              v1 = allView.getUint16(off1, true);
              v2 = allView.getUint16(off2, true);
              mx = v1 > v2 ? v1 : v2;
              dView.setUint16(i * 2, mx, true);
            } else if (esize === 4) {
              v1 = allView.getUint32(off1, true);
              v2 = allView.getUint32(off2, true);
              mx = v1 > v2 ? v1 : v2;
              dView.setUint32(i * 4, mx, true);
            } else {
              v1 = allView.getBigUint64(off1, true);
              v2 = allView.getBigUint64(off2, true);
              mx = v1 > v2 ? v1 : v2;
              dView.setBigUint64(i * 8, mx, true);
            }
          }
          break;
        }
        case SIMD_ADDP: {
          const all = new Uint8Array(32);
          all.set(a.subarray(0, bytes), 0);
          all.set(b_.subarray(0, bytes), bytes);
          const allView = new DataView(all.buffer);
          for (let i = 0; i < elems; i++) {
            const off1 = i * 2 * esize;
            const off2 = off1 + esize;
            if (esize === 1) dView.setUint8(i, allView.getUint8(off1) + allView.getUint8(off2) & 255);
            else if (esize === 2) dView.setUint16(i * 2, allView.getUint16(off1, true) + allView.getUint16(off2, true) & 65535, true);
            else if (esize === 4) dView.setUint32(i * 4, allView.getUint32(off1, true) + allView.getUint32(off2, true) >>> 0, true);
            else dView.setBigUint64(i * 8, BigInt.asUintN(64, allView.getBigUint64(off1, true) + allView.getBigUint64(off2, true)), true);
          }
          break;
        }
        case SIMD_TBL: {
          const len = extra;
          const tableSize = (len + 1) * 16;
          const table = new Uint8Array(tableSize);
          for (let r = 0; r <= len; r++) {
            const vr = V_OFFSET + (rn + r & 31) * 16;
            table.set(buf.subarray(vr, vr + 16), r * 16);
          }
          for (let i = 0; i < bytes; i++) {
            const idx = b_[i];
            dst[i] = idx < tableSize ? table[idx] : 0;
          }
          break;
        }
        case SIMD_ZIP1:
        case SIMD_ZIP2: {
          const half = op === SIMD_ZIP1 ? 0 : elems / 2;
          for (let i = 0; i < elems / 2; i++) {
            const srcIdx = half + i;
            const dstIdx = i * 2;
            if (esize === 1) {
              dst[dstIdx] = a[srcIdx];
              dst[dstIdx + 1] = b_[srcIdx];
            } else if (esize === 2) {
              dView.setUint16(dstIdx * 2, aView.getUint16(srcIdx * 2, true), true);
              dView.setUint16((dstIdx + 1) * 2, bView.getUint16(srcIdx * 2, true), true);
            } else if (esize === 4) {
              dView.setUint32(dstIdx * 4, aView.getUint32(srcIdx * 4, true), true);
              dView.setUint32((dstIdx + 1) * 4, bView.getUint32(srcIdx * 4, true), true);
            } else {
              dView.setBigUint64(dstIdx * 8, aView.getBigUint64(srcIdx * 8, true), true);
              dView.setBigUint64((dstIdx + 1) * 8, bView.getBigUint64(srcIdx * 8, true), true);
            }
          }
          break;
        }
        case SIMD_UZP1:
        case SIMD_UZP2: {
          const phase = op === SIMD_UZP1 ? 0 : 1;
          for (let i = 0; i < elems / 2; i++) {
            const srcIdx = i * 2 + phase;
            if (esize === 1) {
              dst[i] = a[srcIdx];
              dst[i + elems / 2] = b_[srcIdx];
            } else if (esize === 2) {
              dView.setUint16(i * 2, aView.getUint16(srcIdx * 2, true), true);
              dView.setUint16((i + elems / 2) * 2, bView.getUint16(srcIdx * 2, true), true);
            } else if (esize === 4) {
              dView.setUint32(i * 4, aView.getUint32(srcIdx * 4, true), true);
              dView.setUint32((i + elems / 2) * 4, bView.getUint32(srcIdx * 4, true), true);
            } else {
              dView.setBigUint64(i * 8, aView.getBigUint64(srcIdx * 8, true), true);
              dView.setBigUint64((i + elems / 2) * 8, bView.getBigUint64(srcIdx * 8, true), true);
            }
          }
          break;
        }
        case SIMD_TRN1:
        case SIMD_TRN2: {
          const phase = op === SIMD_TRN1 ? 0 : 1;
          for (let i = 0; i < elems / 2; i++) {
            const evenIdx = i * 2;
            if (esize === 1) {
              dst[evenIdx] = a[evenIdx + phase];
              dst[evenIdx + 1] = b_[evenIdx + phase];
            } else if (esize === 2) {
              dView.setUint16(evenIdx * 2, aView.getUint16((evenIdx + phase) * 2, true), true);
              dView.setUint16((evenIdx + 1) * 2, bView.getUint16((evenIdx + phase) * 2, true), true);
            } else if (esize === 4) {
              dView.setUint32(evenIdx * 4, aView.getUint32((evenIdx + phase) * 4, true), true);
              dView.setUint32((evenIdx + 1) * 4, bView.getUint32((evenIdx + phase) * 4, true), true);
            } else {
              dView.setBigUint64(evenIdx * 8, aView.getBigUint64((evenIdx + phase) * 8, true), true);
              dView.setBigUint64((evenIdx + 1) * 8, bView.getBigUint64((evenIdx + phase) * 8, true), true);
            }
          }
          break;
        }
      }
      buf.set(dst.subarray(0, bytes), vd);
      if (!Q) {
        buf.fill(0, vd + 8, vd + 16);
      }
      return 0n;
    };
  }
  var NEXT_PC_LOCAL = REG_COUNT + 4;
  var ITER_LOCAL = REG_COUNT + 5;
  function translateBlock(block, memory2) {
    const b = new WasmModuleBuilder();
    b.addMemoryImport("env", "memory", 256);
    const helpers = setupHelperImports(b);
    const typeIdx = b.addFuncType([], [126 /* I64 */]);
    b.beginFunction(typeIdx);
    const firstLocal = b.addLocals(126 /* I64 */, REG_COUNT + 6);
    const TMP0 = REG_COUNT;
    const TMP1 = REG_COUNT + 1;
    const TMP2 = REG_COUNT + 2;
    const TMP3 = REG_COUNT + 3;
    const usage = analyzeRegUsage(block.insns);
    const allUsed = /* @__PURE__ */ new Set([...usage.reads, ...usage.writes]);
    emitPrologue(b, allUsed);
    b.i64_const(HALT_PC);
    b.local_set(NEXT_PC_LOCAL);
    if (block.isSelfLoop) {
      emitSelfLoopBlock(b, block, TMP0, TMP1, TMP2, TMP3, helpers);
    } else {
      emitLinearBlock(b, block, TMP0, TMP1, TMP2, TMP3, helpers);
      emitEpilogue(b, usage.writes);
    }
    b.local_get(NEXT_PC_LOCAL);
    b.endFunction();
    b.exportFunction("run");
    const bytes = b.finalize();
    const mod = new WebAssembly.Module(bytes);
    return instantiateBlock(
      { module: mod, startPC: block.startPC, endPC: block.endPC, insnCount: block.insns.length, isSelfLoop: block.isSelfLoop },
      memory2
    );
  }
  function instantiateBlock(compiled, memory2) {
    const env = { memory: memory2, ...helperFunctions, simd_op: createSimdOp(memory2) };
    const instance = new WebAssembly.Instance(compiled.module, { env });
    return {
      run: instance.exports.run,
      startPC: compiled.startPC,
      endPC: compiled.endPC,
      insnCount: compiled.insnCount,
      isSelfLoop: compiled.isSelfLoop
    };
  }
  function emitPrologue(b, allUsed) {
    for (const local of allUsed) {
      if (local === NZCV_LOCAL) {
        b.i32_const(NZCV_OFFSET);
        b.i64_load(3, 0);
        b.local_set(NZCV_LOCAL);
      } else if (local === SP_LOCAL) {
        b.i32_const(SP_OFFSET);
        b.i64_load(3, 0);
        b.local_set(SP_LOCAL);
      } else if (local >= 0 && local <= 30) {
        b.i32_const(local * 8);
        b.i64_load(3, 0);
        b.local_set(local);
      }
    }
  }
  function emitLinearBlock(b, block, TMP0, TMP1, TMP2, TMP3, helpers) {
    const startPC = Number(block.startPC);
    for (let i = 0; i < block.insns.length; i++) {
      const pc = startPC + i * 4;
      emitInsn(b, block.insns[i], pc, TMP0, TMP1, TMP2, TMP3, helpers);
    }
    b.i32_const(ITER_OFFSET);
    b.i64_const(BigInt(block.insns.length));
    b.i64_store(3, 0);
  }
  function emitSelfLoopBlock(b, block, TMP0, TMP1, TMP2, TMP3, helpers) {
    const startPC = Number(block.startPC);
    const insns = block.insns;
    const lastInsn = insns[insns.length - 1];
    const penult = insns.length >= 2 ? insns[insns.length - 2] : null;
    const canFuse = lastInsn.kind === "b_cond" && penult !== null && (penult.kind === "sub_reg" && penult.setFlags && penult.rd === 31 || penult.kind === "sub_imm" && penult.setFlags && penult.rd === 31) && !insnsAccessNZCV(insns.slice(0, insns.length - 2));
    b.block(BLOCKTYPE_VOID);
    b.loop(BLOCKTYPE_VOID);
    b.local_get(ITER_LOCAL);
    b.i64_const(1n);
    b.i64_add();
    b.local_set(ITER_LOCAL);
    const safetyExitPC = startPC + insns.length * 4;
    b.local_get(ITER_LOCAL);
    b.i64_const(BigInt(1e7));
    b.i64_gt_u();
    b.if_(BLOCKTYPE_VOID);
    b.i64_const(BigInt(safetyExitPC));
    b.local_set(NEXT_PC_LOCAL);
    b.br(2);
    b.end();
    const skipLast = canFuse ? 2 : 1;
    for (let i = 0; i < insns.length - skipLast; i++) {
      const pc = startPC + i * 4;
      emitInsn(b, insns[i], pc, TMP0, TMP1, TMP2, TMP3, helpers);
    }
    const lastPC = startPC + (insns.length - 1) * 4;
    if (canFuse && lastInsn.kind === "b_cond") {
      const cmpInsn = penult;
      emitFusedCmpBranch(b, cmpInsn, lastInsn.cond);
    } else if (lastInsn.kind === "b") {
      b.br(0);
    } else if (lastInsn.kind === "b_cond") {
      emitConditionTest(b, lastInsn.cond, NZCV_LOCAL);
      b.br_if(0);
    } else if (lastInsn.kind === "cbz") {
      b.local_get(regLocal(lastInsn.rt));
      b.i64_eqz();
      b.br_if(0);
    } else if (lastInsn.kind === "cbnz") {
      b.local_get(regLocal(lastInsn.rt));
      b.i64_const(0n);
      b.i64_ne();
      b.br_if(0);
    } else if (lastInsn.kind === "tbz") {
      b.local_get(regLocal(lastInsn.rt));
      b.i64_const(BigInt(lastInsn.bit));
      b.i64_shr_u();
      b.i64_const(1n);
      b.i64_and();
      b.i64_eqz();
      b.br_if(0);
    } else if (lastInsn.kind === "tbnz") {
      b.local_get(regLocal(lastInsn.rt));
      b.i64_const(BigInt(lastInsn.bit));
      b.i64_shr_u();
      b.i64_const(1n);
      b.i64_and();
      b.i64_const(0n);
      b.i64_ne();
      b.br_if(0);
    }
    const exitPC = lastPC + 4;
    b.i64_const(BigInt(exitPC));
    b.local_set(NEXT_PC_LOCAL);
    if (canFuse) {
      const cmpInsn = penult;
      if (cmpInsn.kind === "sub_reg") {
        emitGetReg(b, cmpInsn.rn, cmpInsn.sf);
        b.local_tee(TMP0);
        emitGetReg(b, cmpInsn.rm, cmpInsn.sf);
        emitShift(b, cmpInsn.shift, cmpInsn.amount, cmpInsn.sf);
        if (cmpInsn.sf === 0) {
          b.i64_const(0xFFFFFFFFn);
          b.i64_and();
        }
        b.local_tee(TMP1);
        b.i64_sub();
        if (cmpInsn.sf === 0) {
          b.i64_const(0xFFFFFFFFn);
          b.i64_and();
        }
        b.local_set(TMP2);
      } else if (cmpInsn.kind === "sub_imm") {
        emitGetRegSP(b, cmpInsn.rn, cmpInsn.sf);
        b.local_tee(TMP0);
        b.i64_const(BigInt(cmpInsn.imm));
        b.local_tee(TMP1);
        b.i64_sub();
        if (cmpInsn.sf === 0) {
          b.i64_const(0xFFFFFFFFn);
          b.i64_and();
        }
        b.local_set(TMP2);
      }
      emitComputeFlags(b, false, penult.sf, TMP0, TMP1, TMP2);
    }
    b.br(1);
    b.end();
    b.unreachable();
    b.end();
    b.i32_const(ITER_OFFSET);
    b.local_get(ITER_LOCAL);
    b.i64_store(3, 0);
    const usage = analyzeRegUsage(block.insns);
    emitEpilogue(b, usage.writes);
  }
  function insnsAccessNZCV(insns) {
    for (const insn of insns) {
      const usage = analyzeRegUsage([insn]);
      if (usage.reads.has(NZCV_LOCAL) || usage.writes.has(NZCV_LOCAL)) return true;
    }
    return false;
  }
  function emitFusedCmpBranch(b, cmpInsn, cond) {
    if (cmpInsn.kind === "sub_reg") {
      emitGetReg(b, cmpInsn.rn, cmpInsn.sf);
    } else {
      emitGetRegSP(b, cmpInsn.rn, cmpInsn.sf);
    }
    if (cmpInsn.kind === "sub_reg") {
      emitGetReg(b, cmpInsn.rm, cmpInsn.sf);
      emitShift(b, cmpInsn.shift, cmpInsn.amount, cmpInsn.sf);
      if (cmpInsn.sf === 0) {
        b.i64_const(0xFFFFFFFFn);
        b.i64_and();
      }
    } else {
      b.i64_const(BigInt(cmpInsn.imm));
    }
    emitCondCompare(b, cond);
    b.br_if(0);
  }
  function emitCondCompare(b, cond) {
    switch (cond) {
      case 0:
        b.i64_eq();
        break;
      // EQ: Rn == Op2
      case 1:
        b.i64_ne();
        break;
      // NE: Rn != Op2
      case 2:
        b.i64_ge_u();
        break;
      // CS/HS: Rn >=u Op2
      case 3:
        b.i64_lt_u();
        break;
      // CC/LO: Rn <u Op2
      case 8:
        b.i64_gt_u();
        break;
      // HI: Rn >u Op2
      case 9:
        b.i64_le_u();
        break;
      // LS: Rn <=u Op2
      case 10:
        b.i64_ge_s();
        break;
      // GE: Rn >=s Op2
      case 11:
        b.i64_lt_s();
        break;
      // LT: Rn <s Op2
      case 12:
        b.i64_gt_s();
        break;
      // GT: Rn >s Op2
      case 13:
        b.i64_le_s();
        break;
      // LE: Rn <=s Op2
      default:
        b.i64_sub();
        b.drop();
        b.i32_const(0);
        break;
    }
  }
  function emitEpilogue(b, writes) {
    for (const local of writes) {
      if (local === NZCV_LOCAL) {
        b.i32_const(NZCV_OFFSET);
        b.local_get(NZCV_LOCAL);
        b.i64_store(3, 0);
      } else if (local === SP_LOCAL) {
        b.i32_const(SP_OFFSET);
        b.local_get(SP_LOCAL);
        b.i64_store(3, 0);
      } else if (local >= 0 && local <= 30) {
        b.i32_const(local * 8);
        b.local_get(local);
        b.i64_store(3, 0);
      }
    }
  }
  function emitInsn(b, insn, pc, TMP0, TMP1, TMP2, TMP3, helpers) {
    switch (insn.kind) {
      case "add_imm":
        emitAddSubImm(b, insn, true, TMP0, TMP1, TMP2);
        break;
      case "sub_imm":
        emitAddSubImm(b, insn, false, TMP0, TMP1, TMP2);
        break;
      case "add_reg":
        emitAddSubReg(b, insn, true, TMP0, TMP1, TMP2);
        break;
      case "sub_reg":
        emitAddSubReg(b, insn, false, TMP0, TMP1, TMP2);
        break;
      case "adc":
      case "sbc":
        emitAdcSbc(b, insn, TMP0);
        break;
      case "orr_reg":
        emitLogicalReg(b, insn, TMP0);
        break;
      case "eor_reg":
        emitLogicalReg(b, insn, TMP0);
        break;
      case "and_reg":
        emitLogicalReg(b, insn, TMP0);
        break;
      case "and_imm":
        emitLogicalImm(b, insn, TMP0);
        break;
      case "orr_imm":
        emitLogicalImm(b, insn, TMP0);
        break;
      case "eor_imm":
        emitLogicalImm(b, insn, TMP0);
        break;
      case "movz":
        emitMovz(b, insn);
        break;
      case "movk":
        emitMovk(b, insn);
        break;
      case "movn":
        emitMovn(b, insn);
        break;
      case "madd":
        emitMadd(b, insn);
        break;
      case "msub":
        emitMsub(b, insn);
        break;
      case "smaddl":
      case "smsubl":
      case "umaddl":
      case "umsubl":
        emitMulLong(b, insn);
        break;
      case "sdiv":
        emitDiv(b, insn, true);
        break;
      case "udiv":
        emitDiv(b, insn, false);
        break;
      case "smulh": {
        emitGetReg(b, insn.rn, 1);
        emitGetReg(b, insn.rm, 1);
        b.call(helpers.smulhIdx);
        emitSetReg(b, insn.rd, 1);
        break;
      }
      case "umulh": {
        emitGetReg(b, insn.rn, 1);
        emitGetReg(b, insn.rm, 1);
        b.call(helpers.umulhIdx);
        emitSetReg(b, insn.rd, 1);
        break;
      }
      case "sbfm":
        emitBitfield(b, insn);
        break;
      case "ubfm":
        emitBitfield(b, insn);
        break;
      case "bfm":
        emitBitfield(b, insn);
        break;
      case "csel":
        emitCondSelect(b, insn);
        break;
      case "ldr_imm":
        emitLdrImm(b, insn);
        break;
      case "str_imm":
        emitStrImm(b, insn);
        break;
      case "ldrb_imm":
        emitSubwordLoadImm(b, insn, 1, false);
        break;
      case "strb_imm":
        emitSubwordStoreImm(b, insn, 1);
        break;
      case "ldrh_imm":
        emitSubwordLoadImm(b, insn, 2, false);
        break;
      case "strh_imm":
        emitSubwordStoreImm(b, insn, 2);
        break;
      case "ldrsb_imm":
        emitSubwordLoadImm(b, insn, 1, true);
        break;
      case "ldrsh_imm":
        emitSubwordLoadImm(b, insn, 2, true);
        break;
      case "ldrsw_imm":
        emitSubwordLoadImm(b, insn, 4, true);
        break;
      case "ldr_reg":
        emitLdrReg(b, insn);
        break;
      case "str_reg":
        emitStrReg(b, insn);
        break;
      case "ldp":
        emitLdp(b, insn);
        break;
      case "stp":
        emitStp(b, insn);
        break;
      case "adr": {
        b.i64_const(BigInt(pc) + BigInt(insn.imm));
        emitSetReg(b, insn.rd, 1);
        break;
      }
      case "adrp": {
        b.i64_const(BigInt(pc & ~4095) + BigInt(insn.imm));
        emitSetReg(b, insn.rd, 1);
        break;
      }
      // Sub-word register-offset loads/stores
      case "ldrb_reg":
        emitSubwordRegLoad(b, insn, 1, false);
        break;
      case "strb_reg":
        emitSubwordRegStore(b, insn, 1);
        break;
      case "ldrh_reg":
        emitSubwordRegLoad(b, insn, 2, false);
        break;
      case "strh_reg":
        emitSubwordRegStore(b, insn, 2);
        break;
      case "ldrsb_reg":
        emitSubwordRegLoad(b, insn, 1, true);
        break;
      case "ldrsh_reg":
        emitSubwordRegLoad(b, insn, 2, true);
        break;
      case "ldrsw_reg":
        emitSubwordRegLoad(b, insn, 4, true);
        break;
      // Load literal (PC-relative)
      case "ldr_lit": {
        const addr = pc + insn.offset;
        emitGuestAddr(b, BigInt(addr));
        if (insn.sf) {
          b.i64_load(3, 0);
        } else {
          b.i64_load32_u(2, 0);
        }
        emitSetReg(b, insn.rt, 1);
        break;
      }
      case "ldrsw_lit": {
        const addr = pc + insn.offset;
        emitGuestAddr(b, BigInt(addr));
        b.i64_load32_s(2, 0);
        emitSetReg(b, insn.rt, 1);
        break;
      }
      // CLZ: count leading zeros (native WASM)
      case "clz": {
        emitGetReg(b, insn.rn, insn.sf);
        if (insn.sf === 0) {
          b.i64_const(32n);
          b.i64_shl();
          b.i64_clz();
        } else {
          b.i64_clz();
        }
        emitSetReg(b, insn.rd, insn.sf);
        break;
      }
      // CLS: count leading sign bits = clz(x ^ (x >> 1)) - 1
      case "cls": {
        emitGetReg(b, insn.rn, insn.sf);
        if (insn.sf === 0) {
          b.i64_const(32n);
          b.i64_shl();
          b.i64_const(32n);
          b.i64_shr_s();
          b.local_tee(TMP0);
          b.i64_const(1n);
          b.i64_shr_s();
          b.local_get(TMP0);
          b.i64_xor();
          b.i64_const(32n);
          b.i64_shl();
          b.i64_clz();
        } else {
          b.local_tee(TMP0);
          b.i64_const(1n);
          b.i64_shr_s();
          b.local_get(TMP0);
          b.i64_xor();
          b.i64_clz();
        }
        b.i64_const(1n);
        b.i64_sub();
        emitSetReg(b, insn.rd, insn.sf);
        break;
      }
      // RBIT: reverse bits — use JS helper via shift/mask
      case "rbit":
        emitRbit(b, insn, TMP0);
        break;
      // REV: reverse bytes
      case "rev":
        emitRev(b, insn, TMP0, TMP1);
        break;
      // REV16: reverse bytes in each 16-bit halfword
      case "rev16":
        emitRev16(b, insn, TMP0, TMP1);
        break;
      // REV32: reverse bytes in each 32-bit word (64-bit only)
      case "rev32":
        emitRev32(b, insn, TMP0, TMP1);
        break;
      // EXTR: extract from pair of registers
      case "extr":
        emitExtr(b, insn);
        break;
      // CCMP/CCMN: conditional compare
      case "ccmp":
        emitCcmpCcmn(b, insn, false, TMP0, TMP1, TMP2);
        break;
      case "ccmn":
        emitCcmpCcmn(b, insn, true, TMP0, TMP1, TMP2);
        break;
      // Load/store exclusive (treated as plain load/store, single-threaded)
      case "ldxr": {
        emitGetReg(b, insn.rn, 1);
        emitAddGuestBase(b);
        if (insn.sf) {
          b.i64_load(3, 0);
        } else {
          b.i64_load32_u(2, 0);
        }
        emitSetReg(b, insn.rt, 1);
        break;
      }
      case "stxr": {
        emitGetReg(b, insn.rn, 1);
        emitAddGuestBase(b);
        emitGetReg(b, insn.rt, insn.sf);
        if (insn.sf) {
          b.i64_store(3, 0);
        } else {
          b.i64_store32(2, 0);
        }
        b.i64_const(0n);
        emitSetReg(b, insn.rs, 0);
        break;
      }
      // MRS: read system register
      case "mrs":
        emitMrs(b, insn);
        break;
      // MSR: write system register
      case "msr":
        emitMsr(b, insn);
        break;
      // Add/sub extended register
      case "add_ext":
        emitAddSubExt(b, insn, true, TMP0, TMP1, TMP2);
        break;
      case "sub_ext":
        emitAddSubExt(b, insn, false, TMP0, TMP1, TMP2);
        break;
      // Branch instructions (block terminators)
      // Store target PC to NEXT_PC_LOCAL instead of return_()
      // so the epilogue (register writeback) always runs
      case "b": {
        const target = pc + insn.offset;
        b.i64_const(BigInt(target));
        b.local_set(NEXT_PC_LOCAL);
        break;
      }
      case "bl": {
        b.i64_const(BigInt(pc + 4));
        b.local_set(regLocal(30));
        const target = pc + insn.offset;
        b.i64_const(BigInt(target));
        b.local_set(NEXT_PC_LOCAL);
        break;
      }
      case "b_cond": {
        emitConditionTest(b, insn.cond, NZCV_LOCAL);
        b.if_(126 /* I64 */);
        b.i64_const(BigInt(pc + insn.offset));
        b.else_();
        b.i64_const(BigInt(pc + 4));
        b.end();
        b.local_set(NEXT_PC_LOCAL);
        break;
      }
      case "cbz": {
        emitGetReg(b, insn.rt, insn.sf);
        b.i64_eqz();
        b.if_(126 /* I64 */);
        b.i64_const(BigInt(pc + insn.offset));
        b.else_();
        b.i64_const(BigInt(pc + 4));
        b.end();
        b.local_set(NEXT_PC_LOCAL);
        break;
      }
      case "cbnz": {
        emitGetReg(b, insn.rt, insn.sf);
        b.i64_const(0n);
        b.i64_ne();
        b.if_(126 /* I64 */);
        b.i64_const(BigInt(pc + insn.offset));
        b.else_();
        b.i64_const(BigInt(pc + 4));
        b.end();
        b.local_set(NEXT_PC_LOCAL);
        break;
      }
      case "tbz": {
        b.local_get(regLocal(insn.rt));
        b.i64_const(BigInt(insn.bit));
        b.i64_shr_u();
        b.i64_const(1n);
        b.i64_and();
        b.i64_eqz();
        b.if_(126 /* I64 */);
        b.i64_const(BigInt(pc + insn.offset));
        b.else_();
        b.i64_const(BigInt(pc + 4));
        b.end();
        b.local_set(NEXT_PC_LOCAL);
        break;
      }
      case "tbnz": {
        b.local_get(regLocal(insn.rt));
        b.i64_const(BigInt(insn.bit));
        b.i64_shr_u();
        b.i64_const(1n);
        b.i64_and();
        b.i64_const(0n);
        b.i64_ne();
        b.if_(126 /* I64 */);
        b.i64_const(BigInt(pc + insn.offset));
        b.else_();
        b.i64_const(BigInt(pc + 4));
        b.end();
        b.local_set(NEXT_PC_LOCAL);
        break;
      }
      case "br": {
        b.local_get(regLocal(insn.rn));
        b.local_set(NEXT_PC_LOCAL);
        break;
      }
      case "blr": {
        b.i64_const(BigInt(pc + 4));
        b.local_set(regLocal(30));
        b.local_get(regLocal(insn.rn));
        b.local_set(NEXT_PC_LOCAL);
        break;
      }
      case "ret": {
        b.local_get(regLocal(insn.rn));
        b.local_set(NEXT_PC_LOCAL);
        break;
      }
      case "svc": {
        b.i32_const(PC_OFFSET);
        b.i64_const(BigInt(pc));
        b.i64_store(3, 0);
        b.i64_const(SVC_PC);
        b.local_set(NEXT_PC_LOCAL);
        break;
      }
      case "brk": {
        b.i64_const(HALT_PC);
        b.local_set(NEXT_PC_LOCAL);
        break;
      }
      // Variable shifts
      case "lslv":
      case "lsrv":
      case "asrv":
      case "rorv": {
        if (insn.kind === "asrv" && insn.sf === 0) {
          emitGetReg(b, insn.rn, 0);
          b.i64_const(32n);
          b.i64_shl();
          b.i64_const(32n);
          b.i64_shr_s();
          emitGetReg(b, insn.rm, 0);
          b.i64_const(31n);
          b.i64_and();
          b.i64_shr_s();
          b.i64_const(0xFFFFFFFFn);
          b.i64_and();
        } else if (insn.kind === "rorv" && insn.sf === 0) {
          emitGetReg(b, insn.rn, 0);
          b.local_tee(TMP0);
          emitGetReg(b, insn.rm, 0);
          b.i64_const(31n);
          b.i64_and();
          b.local_tee(TMP1);
          b.i64_shr_u();
          b.local_get(TMP0);
          b.i64_const(32n);
          b.local_get(TMP1);
          b.i64_sub();
          b.i64_shl();
          b.i64_or();
          b.i64_const(0xFFFFFFFFn);
          b.i64_and();
        } else {
          emitGetReg(b, insn.rn, insn.sf);
          emitGetReg(b, insn.rm, insn.sf);
          b.i64_const(BigInt(insn.sf ? 63 : 31));
          b.i64_and();
          switch (insn.kind) {
            case "lslv":
              b.i64_shl();
              break;
            case "lsrv":
              b.i64_shr_u();
              break;
            case "asrv":
              b.i64_shr_s();
              break;
            case "rorv":
              b.i64_rotr();
              break;
          }
          if (insn.sf === 0) {
            b.i64_const(0xFFFFFFFFn);
            b.i64_and();
          }
        }
        emitSetReg(b, insn.rd, insn.sf);
        break;
      }
      // SIMD/FP load/store (Q = 128-bit)
      case "str_simd": {
        if (insn.size === 16) {
          emitSimdStore128(b, insn, TMP0);
        } else {
          emitSimdStoreSmall(b, insn, TMP0);
        }
        break;
      }
      case "ldr_simd": {
        if (insn.size === 16) {
          emitSimdLoad128(b, insn, TMP0);
        } else {
          emitSimdLoadSmall(b, insn, TMP0);
        }
        break;
      }
      case "stp_simd": {
        if (insn.size === 16) {
          emitSimdStorePair128(b, insn, TMP0);
        } else {
          emitSimdStorePairSmall(b, insn, TMP0);
        }
        break;
      }
      case "ldp_simd": {
        if (insn.size === 16) {
          emitSimdLoadPair128(b, insn, TMP0);
        } else {
          emitSimdLoadPairSmall(b, insn, TMP0);
        }
        break;
      }
      // SIMD DUP from general register
      case "simd_dup_gpr": {
        emitSimdDupGpr(b, insn);
        break;
      }
      // SIMD DUP from element
      case "simd_dup_elem": {
        emitSimdDupElem(b, insn, TMP0);
        break;
      }
      // SIMD UMOV (extract to GPR)
      case "simd_umov": {
        emitSimdUmov(b, insn);
        break;
      }
      // SIMD MOVI (move immediate to vector)
      case "simd_movi": {
        emitSimdMovi(b, insn);
        break;
      }
      // SIMD INS from GPR
      case "simd_ins_gpr": {
        emitSimdInsGpr(b, insn);
        break;
      }
      // FMOV GPR → FP register
      case "fmov_to_fp": {
        emitFmovToFp(b, insn);
        break;
      }
      // FMOV FP register → GPR
      case "fmov_from_fp": {
        emitFmovFromFp(b, insn);
        break;
      }
      // FMOV GPR → top half of 128-bit FP register
      case "fmov_top_to_fp": {
        const vdOff = vregOffset(insn.vd);
        b.i32_const(vdOff + 8);
        emitGetReg(b, insn.rn, 1);
        b.i64_store(3, 0);
        break;
      }
      // FMOV top half of 128-bit FP register → GPR
      case "fmov_top_from_fp": {
        const vnOff = vregOffset(insn.vn);
        b.i32_const(vnOff + 8);
        b.i64_load(3, 0);
        emitSetReg(b, insn.rd, 1);
        break;
      }
      // FMOV between FP registers
      case "fmov_fp": {
        emitFmovFp(b, insn);
        break;
      }
      // FP <-> integer conversions
      case "scvtf_int":
      case "ucvtf_int":
      case "fcvtzs_int":
      case "fcvtzu_int": {
        emitFpIntConv(b, insn, TMP0, helpers);
        break;
      }
      // FP arithmetic (2-source)
      case "fmul":
      case "fdiv":
      case "fadd":
      case "fsub": {
        emitFpArith2(b, insn, TMP0, helpers);
        break;
      }
      // FP misc (1-source)
      case "fabs":
      case "fneg": {
        emitFpMisc1(b, insn, TMP0, helpers);
        break;
      }
      // FP precision conversion
      case "fcvt": {
        emitFcvt(b, insn, TMP0, helpers);
        break;
      }
      // FP immediate
      case "fmov_imm": {
        emitFmovImm(b, insn);
        break;
      }
      // FP compare
      case "fcmp": {
        emitFpCmp(b, insn, helpers);
        break;
      }
      // FP fused multiply-add/subtract (3-source)
      case "fmadd":
      case "fmsub":
      case "fnmadd":
      case "fnmsub": {
        emitFpMulAdd(b, insn, TMP0, helpers);
        break;
      }
      // SIMD ORR (vector, register) / MOV (vector)
      case "simd_orr_reg": {
        const rnOff = vregOffset(insn.rn);
        const rmOff = vregOffset(insn.rm);
        const rdOff = vregOffset(insn.rd);
        b.i32_const(rdOff);
        b.i32_const(rnOff);
        b.i64_load(3, 0);
        b.i32_const(rmOff);
        b.i64_load(3, 0);
        b.i64_or();
        b.i64_store(3, 0);
        if (insn.Q) {
          b.i32_const(rdOff + 8);
          b.i32_const(rnOff + 8);
          b.i64_load(3, 0);
          b.i32_const(rmOff + 8);
          b.i64_load(3, 0);
          b.i64_or();
          b.i64_store(3, 0);
        } else {
          b.i32_const(rdOff + 8);
          b.i64_const(0n);
          b.i64_store(3, 0);
        }
        break;
      }
      // SIMD NOT (bitwise complement)
      case "simd_not": {
        const rnOff = vregOffset(insn.rn);
        const rdOff = vregOffset(insn.rd);
        b.i32_const(rdOff);
        b.i32_const(rnOff);
        b.i64_load(3, 0);
        b.i64_const(-1n);
        b.i64_xor();
        b.i64_store(3, 0);
        b.i32_const(rdOff + 8);
        if (insn.Q) {
          b.i32_const(rnOff + 8);
          b.i64_load(3, 0);
          b.i64_const(-1n);
          b.i64_xor();
        } else {
          b.i64_const(0n);
        }
        b.i64_store(3, 0);
        break;
      }
      // SIMD NEG, ABS, CNT, CLS, CLZ (two-reg misc) — use JS helper
      case "simd_neg":
      case "simd_abs":
      case "simd_cnt":
      case "simd_cls_vec":
      case "simd_clz_vec": {
        b.i64_const(HALT_PC);
        b.local_set(NEXT_PC_LOCAL);
        break;
      }
      // SIMD bitwise operations (AND, BIC, EOR)
      case "simd_and_reg":
      case "simd_bic_reg":
      case "simd_eor_reg": {
        const rnOff = vregOffset(insn.rn);
        const rmOff = vregOffset(insn.rm);
        const rdOff = vregOffset(insn.rd);
        const op = insn.kind === "simd_and_reg" ? "and" : insn.kind === "simd_bic_reg" ? "bic" : "eor";
        b.i32_const(rdOff);
        b.i32_const(rnOff);
        b.i64_load(3, 0);
        b.i32_const(rmOff);
        b.i64_load(3, 0);
        if (op === "bic") {
          b.i64_const(-1n);
          b.i64_xor();
          b.i64_and();
        } else if (op === "and") {
          b.i64_and();
        } else {
          b.i64_xor();
        }
        b.i64_store(3, 0);
        b.i32_const(rdOff + 8);
        if (insn.Q) {
          b.i32_const(rnOff + 8);
          b.i64_load(3, 0);
          b.i32_const(rmOff + 8);
          b.i64_load(3, 0);
          if (op === "bic") {
            b.i64_const(-1n);
            b.i64_xor();
            b.i64_and();
          } else if (op === "and") {
            b.i64_and();
          } else {
            b.i64_xor();
          }
        } else {
          b.i64_const(0n);
        }
        b.i64_store(3, 0);
        break;
      }
      // SIMD integer ADD/SUB — element-wise via helper function
      case "simd_add":
      case "simd_sub":
      case "simd_cmeq_reg":
      case "simd_cmgt_reg":
      case "simd_cmhi_reg":
      case "simd_cmge_reg":
      case "simd_cmhs_reg":
      case "simd_umaxp":
      case "simd_addp":
      case "simd_tbl":
      case "simd_zip1":
      case "simd_zip2":
      case "simd_uzp1":
      case "simd_uzp2":
      case "simd_trn1":
      case "simd_trn2": {
        emitSimdThreeSameGeneric(b, insn, TMP0, helpers);
        break;
      }
      // SIMD shift by immediate
      case "simd_sshll":
      case "simd_ushll": {
        emitSimdShiftLong(b, insn, TMP0);
        break;
      }
      case "nop":
        break;
      default:
        b.i64_const(HALT_PC);
        b.local_set(NEXT_PC_LOCAL);
        break;
    }
  }
  function emitGetReg(b, r, sf) {
    if (r === 31) {
      b.i64_const(0n);
    } else {
      b.local_get(regLocal(r));
      if (sf === 0) {
        b.i64_const(0xFFFFFFFFn);
        b.i64_and();
      }
    }
  }
  function emitGetRegSP(b, r, sf) {
    if (r === 31) {
      b.local_get(SP_LOCAL);
    } else {
      b.local_get(regLocal(r));
    }
    if (sf === 0) {
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
    }
  }
  function emitSetReg(b, r, sf) {
    if (r === 31) {
      b.drop();
      return;
    }
    if (sf === 0) {
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
    }
    b.local_set(regLocal(r));
  }
  function emitSetRegSP(b, r, sf) {
    if (r === 31) {
      if (sf === 0) {
        b.i64_const(0xFFFFFFFFn);
        b.i64_and();
      }
      b.local_set(SP_LOCAL);
      return;
    }
    emitSetReg(b, r, sf);
  }
  function emitAddSubImm(b, insn, isAdd, TMP0, TMP1, TMP2) {
    if (insn.setFlags) {
      emitGetRegSP(b, insn.rn, insn.sf);
      b.local_tee(TMP0);
      b.i64_const(BigInt(insn.imm));
      b.local_tee(TMP1);
      if (isAdd) b.i64_add();
      else b.i64_sub();
      if (insn.sf === 0) {
        b.i64_const(0xFFFFFFFFn);
        b.i64_and();
      }
      b.local_tee(TMP2);
      emitSetReg(b, insn.rd, insn.sf);
      emitComputeFlags(b, isAdd, insn.sf, TMP0, TMP1, TMP2);
    } else {
      emitGetRegSP(b, insn.rn, insn.sf);
      b.i64_const(BigInt(insn.imm));
      if (isAdd) b.i64_add();
      else b.i64_sub();
      emitSetRegSP(b, insn.rd, insn.sf);
    }
  }
  function emitAddSubReg(b, insn, isAdd, TMP0, TMP1, TMP2) {
    if (insn.setFlags) {
      emitGetReg(b, insn.rn, insn.sf);
      b.local_tee(TMP0);
      emitGetReg(b, insn.rm, insn.sf);
      emitShift(b, insn.shift, insn.amount, insn.sf);
      if (insn.sf === 0) {
        b.i64_const(0xFFFFFFFFn);
        b.i64_and();
      }
      b.local_tee(TMP1);
      if (isAdd) b.i64_add();
      else b.i64_sub();
      if (insn.sf === 0) {
        b.i64_const(0xFFFFFFFFn);
        b.i64_and();
      }
      b.local_tee(TMP2);
      emitSetReg(b, insn.rd, insn.sf);
      emitComputeFlags(b, isAdd, insn.sf, TMP0, TMP1, TMP2);
    } else {
      emitGetReg(b, insn.rn, insn.sf);
      emitGetReg(b, insn.rm, insn.sf);
      emitShift(b, insn.shift, insn.amount, insn.sf);
      if (isAdd) b.i64_add();
      else b.i64_sub();
      emitSetReg(b, insn.rd, insn.sf);
    }
  }
  function emitAdcSbc(b, insn, TMP0) {
    const isSbc = insn.kind === "sbc";
    const mask = insn.sf ? 0xFFFFFFFFFFFFFFFFn : 0xFFFFFFFFn;
    b.local_get(NZCV_LOCAL);
    b.i64_const(1n);
    b.i64_shr_u();
    b.i64_const(1n);
    b.i64_and();
    b.local_set(TMP0);
    emitGetReg(b, insn.rn, insn.sf);
    emitGetReg(b, insn.rm, insn.sf);
    if (isSbc) {
      b.i64_const(-1n);
      b.i64_xor();
      if (insn.sf === 0) {
        b.i64_const(mask);
        b.i64_and();
      }
    }
    b.i64_add();
    b.local_get(TMP0);
    b.i64_add();
    if (insn.sf === 0) {
      b.i64_const(mask);
      b.i64_and();
    }
    emitSetReg(b, insn.rd, insn.sf);
  }
  function emitShift(b, shift, amount, sf = 1) {
    if (amount === 0) return;
    if (sf === 0 && shift === "asr") {
      b.i64_const(32n);
      b.i64_shl();
      b.i64_const(BigInt(32 + amount));
      b.i64_shr_s();
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
      return;
    }
    if (sf === 0 && shift === "ror") {
      const TMP = REG_COUNT + 3;
      b.local_tee(TMP);
      b.i64_const(32n);
      b.i64_shl();
      b.local_get(TMP);
      b.i64_or();
      b.i64_const(BigInt(amount));
      b.i64_rotr();
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
      return;
    }
    b.i64_const(BigInt(amount));
    switch (shift) {
      case "lsl":
        b.i64_shl();
        break;
      case "lsr":
        b.i64_shr_u();
        break;
      case "asr":
        b.i64_shr_s();
        break;
      case "ror":
        b.i64_rotr();
        break;
    }
  }
  function emitLogicalReg(b, insn, TMP0) {
    emitGetReg(b, insn.rn, insn.sf);
    emitGetReg(b, insn.rm, insn.sf);
    emitShift(b, insn.shift, insn.amount, insn.sf);
    if (insn.invert) {
      b.i64_const(-1n);
      b.i64_xor();
    }
    switch (insn.kind) {
      case "orr_reg":
        b.i64_or();
        break;
      case "eor_reg":
        b.i64_xor();
        break;
      case "and_reg":
        b.i64_and();
        break;
    }
    const isAnds = insn.kind === "and_reg" && insn.setFlags;
    if (isAnds) {
      b.local_tee(TMP0);
      emitSetReg(b, insn.rd, insn.sf);
      const bits2 = insn.sf ? 64 : 32;
      b.local_get(TMP0);
      if (insn.sf === 0) {
        b.i64_const(0xFFFFFFFFn);
        b.i64_and();
      }
      b.i64_const(BigInt(bits2 - 1));
      b.i64_shr_u();
      b.i64_const(1n);
      b.i64_and();
      b.i64_const(3n);
      b.i64_shl();
      b.local_get(TMP0);
      if (insn.sf === 0) {
        b.i64_const(0xFFFFFFFFn);
        b.i64_and();
      }
      b.i64_eqz();
      b.i64_extend_i32_u();
      b.i64_const(2n);
      b.i64_shl();
      b.i64_or();
      b.local_set(NZCV_LOCAL);
    } else {
      emitSetReg(b, insn.rd, insn.sf);
    }
  }
  function emitLogicalImm(b, insn, TMP0) {
    emitGetReg(b, insn.rn, insn.sf);
    b.i64_const(insn.imm);
    switch (insn.kind) {
      case "and_imm":
        b.i64_and();
        break;
      case "orr_imm":
        b.i64_or();
        break;
      case "eor_imm":
        b.i64_xor();
        break;
    }
    const isAnds = insn.kind === "and_imm" && insn.setFlags;
    if (isAnds) {
      b.local_tee(TMP0);
      emitSetReg(b, insn.rd, insn.sf);
      const bits2 = insn.sf ? 64 : 32;
      b.local_get(TMP0);
      if (insn.sf === 0) {
        b.i64_const(0xFFFFFFFFn);
        b.i64_and();
      }
      b.i64_const(BigInt(bits2 - 1));
      b.i64_shr_u();
      b.i64_const(1n);
      b.i64_and();
      b.i64_const(3n);
      b.i64_shl();
      b.local_get(TMP0);
      if (insn.sf === 0) {
        b.i64_const(0xFFFFFFFFn);
        b.i64_and();
      }
      b.i64_eqz();
      b.i64_extend_i32_u();
      b.i64_const(2n);
      b.i64_shl();
      b.i64_or();
      b.local_set(NZCV_LOCAL);
    } else {
      emitSetRegSP(b, insn.rd, insn.sf);
    }
  }
  function emitMovz(b, insn) {
    const value = BigInt(insn.imm16) << BigInt(insn.hw * 16);
    b.i64_const(value);
    emitSetReg(b, insn.rd, insn.sf);
  }
  function emitMovk(b, insn) {
    const shift = insn.hw * 16;
    const mask = ~(0xFFFFn << BigInt(shift));
    const newBits = BigInt(insn.imm16) << BigInt(shift);
    b.local_get(regLocal(insn.rd));
    b.i64_const(mask);
    b.i64_and();
    b.i64_const(newBits);
    b.i64_or();
    emitSetReg(b, insn.rd, insn.sf);
  }
  function emitMovn(b, insn) {
    const value = BigInt(insn.imm16) << BigInt(insn.hw * 16);
    b.i64_const(~value);
    if (insn.sf === 0) {
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
    }
    emitSetReg(b, insn.rd, insn.sf);
  }
  function emitMadd(b, insn) {
    emitGetReg(b, insn.ra, insn.sf);
    emitGetReg(b, insn.rn, insn.sf);
    emitGetReg(b, insn.rm, insn.sf);
    b.i64_mul();
    b.i64_add();
    emitSetReg(b, insn.rd, insn.sf);
  }
  function emitMsub(b, insn) {
    emitGetReg(b, insn.ra, insn.sf);
    emitGetReg(b, insn.rn, insn.sf);
    emitGetReg(b, insn.rm, insn.sf);
    b.i64_mul();
    b.i64_sub();
    emitSetReg(b, insn.rd, insn.sf);
  }
  function emitMulLong(b, insn) {
    const isSigned = insn.kind === "smaddl" || insn.kind === "smsubl";
    const isSub = insn.kind === "smsubl" || insn.kind === "umsubl";
    emitGetReg(b, insn.rn, 0);
    if (isSigned) {
      b.i64_const(32n);
      b.i64_shl();
      b.i64_const(32n);
      b.i64_shr_s();
    } else {
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
    }
    emitGetReg(b, insn.rm, 0);
    if (isSigned) {
      b.i64_const(32n);
      b.i64_shl();
      b.i64_const(32n);
      b.i64_shr_s();
    } else {
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
    }
    b.i64_mul();
    if (isSub) {
      const tmpLocal = regLocal(insn.rd);
      b.local_set(tmpLocal);
      emitGetReg(b, insn.ra, 1);
      b.local_get(tmpLocal);
      b.i64_sub();
    } else {
      emitGetReg(b, insn.ra, 1);
      b.i64_add();
    }
    emitSetReg(b, insn.rd, 1);
  }
  function emitDiv(b, insn, isSigned) {
    emitGetReg(b, insn.rm, insn.sf);
    b.i64_eqz();
    b.if_(126 /* I64 */);
    b.i64_const(0n);
    b.else_();
    emitGetReg(b, insn.rn, insn.sf);
    emitGetReg(b, insn.rm, insn.sf);
    if (isSigned) b.i64_div_s();
    else b.i64_div_u();
    b.end();
    emitSetReg(b, insn.rd, insn.sf);
  }
  function emitBitfield(b, insn) {
    const regSize = insn.sf ? 64 : 32;
    if (insn.kind === "ubfm") {
      if (insn.imms + 1 === regSize) {
        emitGetReg(b, insn.rn, insn.sf);
        if (insn.immr > 0) {
          b.i64_const(BigInt(insn.immr));
          b.i64_shr_u();
        }
      } else if (insn.imms < insn.immr) {
        const width = insn.imms + 1;
        const lsb = regSize - insn.immr;
        emitGetReg(b, insn.rn, insn.sf);
        const mask = (1n << BigInt(width)) - 1n;
        b.i64_const(mask);
        b.i64_and();
        b.i64_const(BigInt(lsb));
        b.i64_shl();
      } else {
        emitGetReg(b, insn.rn, insn.sf);
        if (insn.immr > 0) {
          b.i64_const(BigInt(insn.immr));
          b.i64_shr_u();
        }
        const width = insn.imms - insn.immr + 1;
        const mask = (1n << BigInt(width)) - 1n;
        b.i64_const(mask);
        b.i64_and();
      }
    } else if (insn.kind === "sbfm") {
      if (insn.imms + 1 === regSize) {
        emitGetReg(b, insn.rn, insn.sf);
        if (insn.sf === 0) {
          b.i64_const(32n);
          b.i64_shl();
          b.i64_const(32n);
          b.i64_shr_s();
        }
        if (insn.immr > 0) {
          b.i64_const(BigInt(insn.immr));
          b.i64_shr_s();
        }
      } else if (insn.immr === 0) {
        emitGetReg(b, insn.rn, insn.sf);
        const bits2 = insn.imms + 1;
        const shift = BigInt(64 - bits2);
        b.i64_const(shift);
        b.i64_shl();
        b.i64_const(shift);
        b.i64_shr_s();
      } else if (insn.imms < insn.immr) {
        const width = insn.imms + 1;
        const lsb = regSize - insn.immr;
        emitGetReg(b, insn.rn, insn.sf);
        const signExtShift = BigInt(64 - width);
        b.i64_const(signExtShift);
        b.i64_shl();
        b.i64_const(signExtShift);
        b.i64_shr_s();
        if (lsb > 0) {
          b.i64_const(BigInt(lsb));
          b.i64_shl();
        }
      } else {
        emitGetReg(b, insn.rn, insn.sf);
        if (insn.immr > 0) {
          b.i64_const(BigInt(insn.immr));
          b.i64_shr_u();
        }
        const width = insn.imms - insn.immr + 1;
        const shift = BigInt(64 - width);
        b.i64_const(shift);
        b.i64_shl();
        b.i64_const(shift);
        b.i64_shr_s();
      }
    } else {
      if (insn.imms < insn.immr) {
        const width = insn.imms + 1;
        const lsb = regSize - insn.immr;
        const mask = (1n << BigInt(width)) - 1n << BigInt(lsb);
        emitGetReg(b, insn.rd, insn.sf);
        b.i64_const(~mask);
        b.i64_and();
        emitGetReg(b, insn.rn, insn.sf);
        b.i64_const(BigInt(lsb));
        b.i64_shl();
        b.i64_const(mask);
        b.i64_and();
        b.i64_or();
      } else {
        const width = insn.imms - insn.immr + 1;
        const mask = (1n << BigInt(width)) - 1n;
        emitGetReg(b, insn.rd, insn.sf);
        b.i64_const(~mask);
        b.i64_and();
        emitGetReg(b, insn.rn, insn.sf);
        if (insn.immr > 0) {
          b.i64_const(BigInt(insn.immr));
          b.i64_shr_u();
        }
        b.i64_const(mask);
        b.i64_and();
        b.i64_or();
      }
    }
    emitSetReg(b, insn.rd, insn.sf);
  }
  function emitCondSelect(b, insn) {
    emitConditionTest(b, insn.cond, NZCV_LOCAL);
    b.if_(126 /* I64 */);
    emitGetReg(b, insn.rn, insn.sf);
    b.else_();
    switch (insn.variant) {
      case "csel":
        emitGetReg(b, insn.rm, insn.sf);
        break;
      case "csinc":
        emitGetReg(b, insn.rm, insn.sf);
        b.i64_const(1n);
        b.i64_add();
        break;
      case "csinv":
        emitGetReg(b, insn.rm, insn.sf);
        b.i64_const(-1n);
        b.i64_xor();
        break;
      case "csneg":
        b.i64_const(0n);
        emitGetReg(b, insn.rm, insn.sf);
        b.i64_sub();
        break;
    }
    b.end();
    emitSetReg(b, insn.rd, insn.sf);
  }
  function emitLdrImm(b, insn) {
    const guestMemBase2 = 65536;
    if (insn.mode === "pre") {
      emitGetRegSP(b, insn.rn, 1);
      b.i64_const(BigInt(insn.offset));
      b.i64_add();
      b.local_tee(insn.rn === 31 ? SP_LOCAL : regLocal(insn.rn));
      b.i64_const(BigInt(guestMemBase2));
      b.i64_add();
      b.i32_wrap_i64();
    } else if (insn.mode === "post") {
      emitGetRegSP(b, insn.rn, 1);
      b.i64_const(BigInt(guestMemBase2));
      b.i64_add();
      b.i32_wrap_i64();
    } else {
      emitGetRegSP(b, insn.rn, 1);
      b.i64_const(BigInt(insn.offset + guestMemBase2));
      b.i64_add();
      b.i32_wrap_i64();
    }
    if (insn.sf === 1) {
      b.i64_load(3, 0);
    } else {
      b.i64_load32_u(2, 0);
    }
    emitSetReg(b, insn.rt, insn.sf);
    if (insn.mode === "post") {
      emitGetRegSP(b, insn.rn, 1);
      b.i64_const(BigInt(insn.offset));
      b.i64_add();
      if (insn.rn === 31) b.local_set(SP_LOCAL);
      else b.local_set(regLocal(insn.rn));
    }
  }
  function emitStrImm(b, insn) {
    const guestMemBase2 = 65536;
    if (insn.mode === "pre") {
      emitGetRegSP(b, insn.rn, 1);
      b.i64_const(BigInt(insn.offset));
      b.i64_add();
      b.local_tee(insn.rn === 31 ? SP_LOCAL : regLocal(insn.rn));
      b.i64_const(BigInt(guestMemBase2));
      b.i64_add();
      b.i32_wrap_i64();
    } else if (insn.mode === "post") {
      emitGetRegSP(b, insn.rn, 1);
      b.i64_const(BigInt(guestMemBase2));
      b.i64_add();
      b.i32_wrap_i64();
    } else {
      emitGetRegSP(b, insn.rn, 1);
      b.i64_const(BigInt(insn.offset + guestMemBase2));
      b.i64_add();
      b.i32_wrap_i64();
    }
    emitGetReg(b, insn.rt, insn.sf);
    if (insn.sf === 1) {
      b.i64_store(3, 0);
    } else {
      b.i64_store32(2, 0);
    }
    if (insn.mode === "post") {
      emitGetRegSP(b, insn.rn, 1);
      b.i64_const(BigInt(insn.offset));
      b.i64_add();
      if (insn.rn === 31) b.local_set(SP_LOCAL);
      else b.local_set(regLocal(insn.rn));
    }
  }
  function emitSubwordLoadImm(b, insn, byteWidth, signExtend2) {
    const guestMemBase2 = 65536;
    if (insn.mode === "pre") {
      emitGetRegSP(b, insn.rn, 1);
      b.i64_const(BigInt(insn.offset));
      b.i64_add();
      b.local_tee(insn.rn === 31 ? SP_LOCAL : regLocal(insn.rn));
      b.i64_const(BigInt(guestMemBase2));
      b.i64_add();
      b.i32_wrap_i64();
    } else if (insn.mode === "post") {
      emitGetRegSP(b, insn.rn, 1);
      b.i64_const(BigInt(guestMemBase2));
      b.i64_add();
      b.i32_wrap_i64();
    } else {
      emitGetRegSP(b, insn.rn, 1);
      b.i64_const(BigInt(insn.offset + guestMemBase2));
      b.i64_add();
      b.i32_wrap_i64();
    }
    if (byteWidth === 1) {
      if (signExtend2) b.i64_load8_s(0, 0);
      else b.i64_load8_u(0, 0);
    } else if (byteWidth === 2) {
      if (signExtend2) b.i64_load16_s(1, 0);
      else b.i64_load16_u(1, 0);
    } else {
      if (signExtend2) b.i64_load32_s(2, 0);
      else b.i64_load32_u(2, 0);
    }
    const sf = "sf" in insn ? insn.sf : 1;
    if (sf === 0 && !signExtend2) {
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
    }
    const rt = insn.rt;
    if (rt !== 31) b.local_set(regLocal(rt));
    else b.drop();
    if (insn.mode === "post") {
      emitGetRegSP(b, insn.rn, 1);
      b.i64_const(BigInt(insn.offset));
      b.i64_add();
      if (insn.rn === 31) b.local_set(SP_LOCAL);
      else b.local_set(regLocal(insn.rn));
    }
  }
  function emitSubwordStoreImm(b, insn, byteWidth) {
    const guestMemBase2 = 65536;
    if (insn.mode === "pre") {
      emitGetRegSP(b, insn.rn, 1);
      b.i64_const(BigInt(insn.offset));
      b.i64_add();
      b.local_tee(insn.rn === 31 ? SP_LOCAL : regLocal(insn.rn));
      b.i64_const(BigInt(guestMemBase2));
      b.i64_add();
      b.i32_wrap_i64();
    } else if (insn.mode === "post") {
      emitGetRegSP(b, insn.rn, 1);
      b.i64_const(BigInt(guestMemBase2));
      b.i64_add();
      b.i32_wrap_i64();
    } else {
      emitGetRegSP(b, insn.rn, 1);
      b.i64_const(BigInt(insn.offset + guestMemBase2));
      b.i64_add();
      b.i32_wrap_i64();
    }
    if (insn.rt === 31) b.i64_const(0n);
    else b.local_get(regLocal(insn.rt));
    if (byteWidth === 1) b.i64_store8(0, 0);
    else b.i64_store16(1, 0);
    if (insn.mode === "post") {
      emitGetRegSP(b, insn.rn, 1);
      b.i64_const(BigInt(insn.offset));
      b.i64_add();
      if (insn.rn === 31) b.local_set(SP_LOCAL);
      else b.local_set(regLocal(insn.rn));
    }
  }
  function emitExtendReg(b, rm, extend, shift) {
    if (rm === 31) {
      b.i64_const(0n);
    } else {
      b.local_get(regLocal(rm));
    }
    switch (extend) {
      case "uxtw":
        b.i64_const(0xFFFFFFFFn);
        b.i64_and();
        break;
      case "sxtw":
        b.i64_const(32n);
        b.i64_shl();
        b.i64_const(32n);
        b.i64_shr_s();
        break;
      case "sxtx":
        break;
      case "lsl":
      default:
        break;
    }
    if (shift > 0) {
      b.i64_const(BigInt(shift));
      b.i64_shl();
    }
  }
  function emitLdrReg(b, insn) {
    const guestMemBase2 = 65536;
    emitGetRegSP(b, insn.rn, 1);
    emitExtendReg(b, insn.rm, insn.extend, insn.shift);
    b.i64_add();
    b.i64_const(BigInt(guestMemBase2));
    b.i64_add();
    b.i32_wrap_i64();
    if (insn.sf === 1) b.i64_load(3, 0);
    else b.i64_load32_u(2, 0);
    emitSetReg(b, insn.rt, insn.sf);
  }
  function emitStrReg(b, insn) {
    const guestMemBase2 = 65536;
    emitGetRegSP(b, insn.rn, 1);
    emitExtendReg(b, insn.rm, insn.extend, insn.shift);
    b.i64_add();
    b.i64_const(BigInt(guestMemBase2));
    b.i64_add();
    b.i32_wrap_i64();
    emitGetReg(b, insn.rt, insn.sf);
    if (insn.sf === 1) b.i64_store(3, 0);
    else b.i64_store32(2, 0);
  }
  function emitLdp(b, insn) {
    const guestMemBase2 = 65536;
    const regSize = insn.sf === 1 ? 8 : 4;
    const TMP_BASE = REG_COUNT;
    const rnLocal = insn.rn === 31 ? SP_LOCAL : regLocal(insn.rn);
    const hasAlias = insn.rt !== 31 && regLocal(insn.rt) === rnLocal || insn.rt2 !== 31 && regLocal(insn.rt2) === rnLocal;
    if (insn.mode === "pre") {
      emitGetRegSP(b, insn.rn, 1);
      b.i64_const(BigInt(insn.offset));
      b.i64_add();
      if (insn.rn === 31) b.local_set(SP_LOCAL);
      else b.local_set(regLocal(insn.rn));
    }
    if (hasAlias) {
      emitGetRegSP(b, insn.rn, 1);
      b.local_set(TMP_BASE);
    }
    function emitEffAddr(extraOffset) {
      if (hasAlias) {
        b.local_get(TMP_BASE);
      } else {
        emitGetRegSP(b, insn.rn, 1);
      }
      if (insn.mode === "post") {
        b.i64_const(BigInt(guestMemBase2 + extraOffset));
      } else if (insn.mode === "pre") {
        b.i64_const(BigInt(guestMemBase2 + extraOffset));
      } else {
        b.i64_const(BigInt(insn.offset + guestMemBase2 + extraOffset));
      }
      b.i64_add();
      b.i32_wrap_i64();
    }
    emitEffAddr(0);
    if (insn.sf === 1) b.i64_load(3, 0);
    else b.i64_load32_u(2, 0);
    emitSetReg(b, insn.rt, insn.sf);
    emitEffAddr(regSize);
    if (insn.sf === 1) b.i64_load(3, 0);
    else b.i64_load32_u(2, 0);
    emitSetReg(b, insn.rt2, insn.sf);
    if (insn.mode === "post") {
      if (hasAlias) {
        b.local_get(TMP_BASE);
      } else {
        emitGetRegSP(b, insn.rn, 1);
      }
      b.i64_const(BigInt(insn.offset));
      b.i64_add();
      if (insn.rn === 31) b.local_set(SP_LOCAL);
      else b.local_set(regLocal(insn.rn));
    }
  }
  function emitStp(b, insn) {
    const guestMemBase2 = 65536;
    const regSize = insn.sf === 1 ? 8 : 4;
    const TMP_BASE = REG_COUNT;
    emitGetRegSP(b, insn.rn, 1);
    if (insn.mode !== "post") {
      b.i64_const(BigInt(insn.offset));
      b.i64_add();
    }
    b.local_set(TMP_BASE);
    b.local_get(TMP_BASE);
    b.i64_const(BigInt(guestMemBase2));
    b.i64_add();
    b.i32_wrap_i64();
    emitGetReg(b, insn.rt, insn.sf);
    if (insn.sf === 1) b.i64_store(3, 0);
    else b.i64_store32(2, 0);
    b.local_get(TMP_BASE);
    b.i64_const(BigInt(guestMemBase2 + regSize));
    b.i64_add();
    b.i32_wrap_i64();
    emitGetReg(b, insn.rt2, insn.sf);
    if (insn.sf === 1) b.i64_store(3, 0);
    else b.i64_store32(2, 0);
    if (insn.mode === "pre") {
      b.local_get(TMP_BASE);
      if (insn.rn === 31) b.local_set(SP_LOCAL);
      else b.local_set(regLocal(insn.rn));
    } else if (insn.mode === "post") {
      b.local_get(TMP_BASE);
      b.i64_const(BigInt(insn.offset));
      b.i64_add();
      if (insn.rn === 31) b.local_set(SP_LOCAL);
      else b.local_set(regLocal(insn.rn));
    }
  }
  function emitGuestAddr(b, guestAddr) {
    b.i32_const(Number(guestAddr) + GUEST_MEM_BASE);
  }
  function emitAddGuestBase(b) {
    b.i64_const(BigInt(GUEST_MEM_BASE));
    b.i64_add();
    b.i32_wrap_i64();
  }
  function emitSubwordRegLoad(b, insn, byteWidth, signExtend2) {
    emitGetRegSP(b, insn.rn, 1);
    emitExtendReg(b, insn.rm, insn.extend, insn.shift);
    b.i64_add();
    b.i64_const(BigInt(GUEST_MEM_BASE));
    b.i64_add();
    b.i32_wrap_i64();
    switch (byteWidth) {
      case 1:
        signExtend2 ? b.i64_load8_s(0, 0) : b.i64_load8_u(0, 0);
        break;
      case 2:
        signExtend2 ? b.i64_load16_s(1, 0) : b.i64_load16_u(1, 0);
        break;
      case 4:
        b.i64_load32_s(2, 0);
        break;
    }
    const sf = "sf" in insn ? insn.sf : 1;
    emitSetReg(b, insn.rt, sf);
  }
  function emitSubwordRegStore(b, insn, byteWidth) {
    emitGetRegSP(b, insn.rn, 1);
    emitExtendReg(b, insn.rm, insn.extend, insn.shift);
    b.i64_add();
    b.i64_const(BigInt(GUEST_MEM_BASE));
    b.i64_add();
    b.i32_wrap_i64();
    emitGetReg(b, insn.rt, 1);
    switch (byteWidth) {
      case 1:
        b.i64_store8(0, 0);
        break;
      case 2:
        b.i64_store16(1, 0);
        break;
    }
  }
  function emitRbit(b, insn, TMP0) {
    const bits2 = insn.sf ? 64 : 32;
    emitGetReg(b, insn.rn, insn.sf);
    if (insn.sf === 0) {
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
    }
    b.local_tee(TMP0);
    b.i64_const(1n);
    b.i64_shr_u();
    b.i64_const(0x5555555555555555n);
    b.i64_and();
    b.local_get(TMP0);
    b.i64_const(0x5555555555555555n);
    b.i64_and();
    b.i64_const(1n);
    b.i64_shl();
    b.i64_or();
    b.local_tee(TMP0);
    b.i64_const(2n);
    b.i64_shr_u();
    b.i64_const(0x3333333333333333n);
    b.i64_and();
    b.local_get(TMP0);
    b.i64_const(0x3333333333333333n);
    b.i64_and();
    b.i64_const(2n);
    b.i64_shl();
    b.i64_or();
    b.local_tee(TMP0);
    b.i64_const(4n);
    b.i64_shr_u();
    b.i64_const(0x0F0F0F0F0F0F0F0Fn);
    b.i64_and();
    b.local_get(TMP0);
    b.i64_const(0x0F0F0F0F0F0F0F0Fn);
    b.i64_and();
    b.i64_const(4n);
    b.i64_shl();
    b.i64_or();
    b.local_tee(TMP0);
    b.i64_const(8n);
    b.i64_shr_u();
    b.i64_const(0x00FF00FF00FF00FFn);
    b.i64_and();
    b.local_get(TMP0);
    b.i64_const(0x00FF00FF00FF00FFn);
    b.i64_and();
    b.i64_const(8n);
    b.i64_shl();
    b.i64_or();
    b.local_tee(TMP0);
    b.i64_const(16n);
    b.i64_shr_u();
    b.i64_const(0x0000FFFF0000FFFFn);
    b.i64_and();
    b.local_get(TMP0);
    b.i64_const(0x0000FFFF0000FFFFn);
    b.i64_and();
    b.i64_const(16n);
    b.i64_shl();
    b.i64_or();
    if (bits2 === 64) {
      b.local_tee(TMP0);
      b.i64_const(32n);
      b.i64_shr_u();
      b.local_get(TMP0);
      b.i64_const(32n);
      b.i64_shl();
      b.i64_or();
    }
    emitSetReg(b, insn.rd, insn.sf);
  }
  function emitRev(b, insn, TMP0, TMP1) {
    emitGetReg(b, insn.rn, insn.sf);
    if (insn.sf === 0) {
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
    }
    b.local_tee(TMP0);
    b.i64_const(8n);
    b.i64_shr_u();
    b.i64_const(0x00FF00FF00FF00FFn);
    b.i64_and();
    b.local_get(TMP0);
    b.i64_const(0x00FF00FF00FF00FFn);
    b.i64_and();
    b.i64_const(8n);
    b.i64_shl();
    b.i64_or();
    b.local_tee(TMP0);
    b.i64_const(16n);
    b.i64_shr_u();
    b.i64_const(0x0000FFFF0000FFFFn);
    b.i64_and();
    b.local_get(TMP0);
    b.i64_const(0x0000FFFF0000FFFFn);
    b.i64_and();
    b.i64_const(16n);
    b.i64_shl();
    b.i64_or();
    if (insn.sf === 1) {
      b.local_tee(TMP0);
      b.i64_const(32n);
      b.i64_shr_u();
      b.local_get(TMP0);
      b.i64_const(32n);
      b.i64_shl();
      b.i64_or();
    }
    emitSetReg(b, insn.rd, insn.sf);
  }
  function emitRev16(b, insn, TMP0, _TMP1) {
    emitGetReg(b, insn.rn, insn.sf);
    b.local_tee(TMP0);
    b.i64_const(8n);
    b.i64_shr_u();
    b.i64_const(0x00FF00FF00FF00FFn);
    b.i64_and();
    b.local_get(TMP0);
    b.i64_const(0x00FF00FF00FF00FFn);
    b.i64_and();
    b.i64_const(8n);
    b.i64_shl();
    b.i64_or();
    emitSetReg(b, insn.rd, insn.sf);
  }
  function emitRev32(b, insn, TMP0, _TMP1) {
    emitGetReg(b, insn.rn, 1);
    b.local_tee(TMP0);
    b.i64_const(8n);
    b.i64_shr_u();
    b.i64_const(0x00FF00FF00FF00FFn);
    b.i64_and();
    b.local_get(TMP0);
    b.i64_const(0x00FF00FF00FF00FFn);
    b.i64_and();
    b.i64_const(8n);
    b.i64_shl();
    b.i64_or();
    b.local_tee(TMP0);
    b.i64_const(16n);
    b.i64_shr_u();
    b.i64_const(0x0000FFFF0000FFFFn);
    b.i64_and();
    b.local_get(TMP0);
    b.i64_const(0x0000FFFF0000FFFFn);
    b.i64_and();
    b.i64_const(16n);
    b.i64_shl();
    b.i64_or();
    emitSetReg(b, insn.rd, 1);
  }
  function emitExtr(b, insn) {
    if (insn.imms === 0) {
      emitGetReg(b, insn.rm, insn.sf);
    } else if (insn.rn === insn.rm) {
      emitGetReg(b, insn.rn, insn.sf);
      b.i64_const(BigInt(insn.imms));
      b.i64_rotr();
    } else {
      const regSize = insn.sf ? 64 : 32;
      emitGetReg(b, insn.rn, insn.sf);
      b.i64_const(BigInt(regSize - insn.imms));
      b.i64_shl();
      emitGetReg(b, insn.rm, insn.sf);
      b.i64_const(BigInt(insn.imms));
      b.i64_shr_u();
      b.i64_or();
    }
    emitSetReg(b, insn.rd, insn.sf);
  }
  function emitCcmpCcmn(b, insn, isAdd, TMP0, TMP1, TMP2) {
    emitConditionTest(b, insn.cond, NZCV_LOCAL);
    b.if_(BLOCKTYPE_VOID);
    emitGetReg(b, insn.rn, insn.sf);
    b.local_set(TMP0);
    if (insn.isImm) {
      b.i64_const(BigInt(insn.rm_imm));
    } else {
      emitGetReg(b, insn.rm_imm, insn.sf);
    }
    b.local_set(TMP1);
    b.local_get(TMP0);
    b.local_get(TMP1);
    if (isAdd) {
      b.i64_add();
    } else {
      b.i64_sub();
    }
    b.local_set(TMP2);
    emitComputeFlags(b, isAdd, insn.sf, TMP0, TMP1, TMP2);
    b.else_();
    b.i64_const(BigInt(insn.nzcv));
    b.local_set(NZCV_LOCAL);
    b.end();
  }
  function emitMrs(b, insn) {
    switch (insn.sysreg) {
      case "tpidr_el0":
        b.i32_const(TPIDR_OFFSET);
        b.i64_load(3, 0);
        break;
      case "nzcv":
        b.local_get(NZCV_LOCAL);
        b.i64_const(28n);
        b.i64_shl();
        break;
      case "fpcr":
        b.i64_const(0n);
        break;
      case "cntfrq_el0":
        b.i64_const(1000000n);
        break;
      case "cntvct_el0":
        b.i64_const(0n);
        break;
      default:
        b.i64_const(0n);
        break;
    }
    emitSetReg(b, insn.rt, 1);
  }
  function emitMsr(b, insn) {
    switch (insn.sysreg) {
      case "tpidr_el0":
        b.i32_const(TPIDR_OFFSET);
        emitGetReg(b, insn.rt, 1);
        b.i64_store(3, 0);
        break;
      case "nzcv":
        emitGetReg(b, insn.rt, 1);
        b.i64_const(28n);
        b.i64_shr_u();
        b.i64_const(0xFn);
        b.i64_and();
        b.local_set(NZCV_LOCAL);
        break;
      default:
        break;
    }
  }
  function emitAddSubExt(b, insn, isAdd, TMP0, TMP1, TMP2) {
    emitGetRegSP(b, insn.rn, 1);
    if (insn.setFlags) b.local_tee(TMP0);
    emitGetReg(b, insn.rm, 1);
    switch (insn.extend) {
      case "uxtb":
        b.i64_const(0xFFn);
        b.i64_and();
        break;
      case "uxth":
        b.i64_const(0xFFFFn);
        b.i64_and();
        break;
      case "uxtw":
        b.i64_const(0xFFFFFFFFn);
        b.i64_and();
        break;
      case "uxtx":
        break;
      // no-op for 64-bit
      case "sxtb":
        b.i64_const(56n);
        b.i64_shl();
        b.i64_const(56n);
        b.i64_shr_s();
        break;
      case "sxth":
        b.i64_const(48n);
        b.i64_shl();
        b.i64_const(48n);
        b.i64_shr_s();
        break;
      case "sxtw":
        b.i64_const(32n);
        b.i64_shl();
        b.i64_const(32n);
        b.i64_shr_s();
        break;
      case "sxtx":
        break;
    }
    if (insn.amount > 0) {
      b.i64_const(BigInt(insn.amount));
      b.i64_shl();
    }
    if (insn.setFlags) b.local_tee(TMP1);
    if (isAdd) {
      b.i64_add();
    } else {
      b.i64_sub();
    }
    if (insn.setFlags) {
      b.local_tee(TMP2);
      emitComputeFlags(b, isAdd, insn.sf, TMP0, TMP1, TMP2);
    }
    if (insn.sf === 0) {
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
    }
    if (insn.rd === 31) {
      if (insn.setFlags) {
        b.drop();
      } else {
        b.local_set(SP_LOCAL);
      }
    } else {
      b.local_set(regLocal(insn.rd));
    }
  }
  function emitComputeFlags(b, isAdd, sf, TMP_OP1, TMP_OP2, TMP_RESULT) {
    const signBit = sf ? 63 : 31;
    b.local_get(TMP_RESULT);
    b.i64_const(BigInt(signBit));
    b.i64_shr_u();
    b.i64_const(1n);
    b.i64_and();
    b.i64_const(3n);
    b.i64_shl();
    b.local_get(TMP_RESULT);
    if (sf === 0) {
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
    }
    b.i64_eqz();
    b.i64_extend_i32_u();
    b.i64_const(2n);
    b.i64_shl();
    b.i64_or();
    if (isAdd) {
      if (sf === 0) {
        b.local_get(TMP_OP1);
        b.i64_const(0xFFFFFFFFn);
        b.i64_and();
        b.local_get(TMP_OP2);
        b.i64_const(0xFFFFFFFFn);
        b.i64_and();
        b.i64_add();
        b.i64_const(32n);
        b.i64_shr_u();
        b.i64_const(1n);
        b.i64_and();
      } else {
        b.local_get(TMP_RESULT);
        b.local_get(TMP_OP1);
        b.i64_lt_u();
        b.i64_extend_i32_u();
      }
    } else {
      if (sf === 0) {
        b.local_get(TMP_OP1);
        b.i64_const(0xFFFFFFFFn);
        b.i64_and();
        b.local_get(TMP_OP2);
        b.i64_const(0xFFFFFFFFn);
        b.i64_and();
        b.i64_ge_u();
        b.i64_extend_i32_u();
      } else {
        b.local_get(TMP_OP1);
        b.local_get(TMP_OP2);
        b.i64_ge_u();
        b.i64_extend_i32_u();
      }
    }
    b.i64_const(1n);
    b.i64_shl();
    b.i64_or();
    if (isAdd) {
      b.local_get(TMP_OP1);
      b.local_get(TMP_RESULT);
      b.i64_xor();
      b.local_get(TMP_OP2);
      b.local_get(TMP_RESULT);
      b.i64_xor();
      b.i64_and();
    } else {
      b.local_get(TMP_OP1);
      b.local_get(TMP_OP2);
      b.i64_xor();
      b.local_get(TMP_OP1);
      b.local_get(TMP_RESULT);
      b.i64_xor();
      b.i64_and();
    }
    b.i64_const(BigInt(signBit));
    b.i64_shr_u();
    b.i64_const(1n);
    b.i64_and();
    b.i64_or();
    b.local_set(NZCV_LOCAL);
  }
  function emitConditionTest(b, cond, nzcvLocal) {
    switch (cond & 14) {
      // bottom bit inverts
      case 0:
        b.local_get(nzcvLocal);
        b.i64_const(4n);
        b.i64_and();
        b.i64_const(0n);
        b.i64_ne();
        break;
      case 2:
        b.local_get(nzcvLocal);
        b.i64_const(2n);
        b.i64_and();
        b.i64_const(0n);
        b.i64_ne();
        break;
      case 4:
        b.local_get(nzcvLocal);
        b.i64_const(8n);
        b.i64_and();
        b.i64_const(0n);
        b.i64_ne();
        break;
      case 6:
        b.local_get(nzcvLocal);
        b.i64_const(1n);
        b.i64_and();
        b.i64_const(0n);
        b.i64_ne();
        break;
      case 8:
        b.local_get(nzcvLocal);
        b.i64_const(2n);
        b.i64_and();
        b.i64_const(0n);
        b.i64_ne();
        b.local_get(nzcvLocal);
        b.i64_const(4n);
        b.i64_and();
        b.i64_eqz();
        b.i32_add();
        b.i32_const(2);
        b.i32_eq();
        break;
      case 10:
        b.local_get(nzcvLocal);
        b.i64_const(8n);
        b.i64_and();
        b.i64_const(3n);
        b.i64_shr_u();
        b.local_get(nzcvLocal);
        b.i64_const(1n);
        b.i64_and();
        b.i64_eq();
        break;
      case 12:
        b.local_get(nzcvLocal);
        b.i64_const(4n);
        b.i64_and();
        b.i64_eqz();
        b.local_get(nzcvLocal);
        b.i64_const(8n);
        b.i64_and();
        b.i64_const(3n);
        b.i64_shr_u();
        b.local_get(nzcvLocal);
        b.i64_const(1n);
        b.i64_and();
        b.i64_eq();
        b.i32_add();
        b.i32_const(2);
        b.i32_eq();
        break;
      case 14:
        b.i32_const(1);
        break;
    }
    if (cond & 1) {
      b.i32_eqz();
    }
  }
  var BLOCK_ID_LOCAL = REG_COUNT + 6;
  var guestMemBase = GUEST_MEM_BASE;
  function emitSimdEffAddr(b, rn, offset, mode, TMP0) {
    if (mode === "pre") {
      emitGetRegSP(b, rn, 1);
      b.i64_const(BigInt(offset));
      b.i64_add();
      b.local_tee(TMP0);
      emitSetRegSP(b, rn, 1);
      b.local_get(TMP0);
    } else if (mode === "post") {
      emitGetRegSP(b, rn, 1);
      b.local_tee(TMP0);
      b.local_get(TMP0);
      b.i64_const(BigInt(offset));
      b.i64_add();
      emitSetRegSP(b, rn, 1);
      b.local_get(TMP0);
    } else {
      emitGetRegSP(b, rn, 1);
      if (offset !== 0) {
        b.i64_const(BigInt(offset));
        b.i64_add();
      }
    }
    b.i64_const(BigInt(guestMemBase));
    b.i64_add();
  }
  function emitSimdStore128(b, insn, TMP0) {
    const vOff = vregOffset(insn.rt);
    emitSimdEffAddr(b, insn.rn, insn.offset, insn.mode, TMP0);
    b.local_set(TMP0);
    b.local_get(TMP0);
    b.i32_wrap_i64();
    b.i32_const(vOff);
    b.i64_load(3, 0);
    b.i64_store(3, 0);
    b.local_get(TMP0);
    b.i32_wrap_i64();
    b.i32_const(vOff + 8);
    b.i64_load(3, 0);
    b.i64_store(3, 8);
  }
  function emitSimdStoreSmall(b, insn, TMP0) {
    const vOff = vregOffset(insn.rt);
    emitSimdEffAddr(b, insn.rn, insn.offset, insn.mode, TMP0);
    b.local_set(TMP0);
    b.local_get(TMP0);
    b.i32_wrap_i64();
    b.i32_const(vOff);
    b.i64_load(3, 0);
    switch (insn.size) {
      case 1:
        b.i64_store8(0, 0);
        break;
      case 2:
        b.i64_store16(1, 0);
        break;
      case 4:
        b.i64_store32(2, 0);
        break;
      case 8:
        b.i64_store(3, 0);
        break;
    }
  }
  function emitSimdLoadSmall(b, insn, TMP0) {
    const vOff = vregOffset(insn.rt);
    emitSimdEffAddr(b, insn.rn, insn.offset, insn.mode, TMP0);
    b.local_set(TMP0);
    b.i32_const(vOff);
    b.local_get(TMP0);
    b.i32_wrap_i64();
    switch (insn.size) {
      case 1:
        b.i64_load8_u(0, 0);
        break;
      case 2:
        b.i64_load16_u(1, 0);
        break;
      case 4:
        b.i64_load32_u(2, 0);
        break;
      case 8:
        b.i64_load(3, 0);
        break;
    }
    b.i64_store(3, 0);
    b.i32_const(vOff + 8);
    b.i64_const(0n);
    b.i64_store(3, 0);
  }
  function emitSimdLoad128(b, insn, TMP0) {
    const vOff = vregOffset(insn.rt);
    emitSimdEffAddr(b, insn.rn, insn.offset, insn.mode, TMP0);
    b.local_set(TMP0);
    b.i32_const(vOff);
    b.local_get(TMP0);
    b.i32_wrap_i64();
    b.i64_load(3, 0);
    b.i64_store(3, 0);
    b.i32_const(vOff + 8);
    b.local_get(TMP0);
    b.i32_wrap_i64();
    b.i64_load(3, 8);
    b.i64_store(3, 0);
  }
  function emitSimdStorePair128(b, insn, TMP0) {
    const v1Off = vregOffset(insn.rt);
    const v2Off = vregOffset(insn.rt2);
    emitSimdEffAddr(b, insn.rn, insn.offset, insn.mode, TMP0);
    b.local_set(TMP0);
    b.local_get(TMP0);
    b.i32_wrap_i64();
    b.i32_const(v1Off);
    b.i64_load(3, 0);
    b.i64_store(3, 0);
    b.local_get(TMP0);
    b.i32_wrap_i64();
    b.i32_const(v1Off + 8);
    b.i64_load(3, 0);
    b.i64_store(3, 8);
    b.local_get(TMP0);
    b.i32_wrap_i64();
    b.i32_const(v2Off);
    b.i64_load(3, 0);
    b.i64_store(3, 16);
    b.local_get(TMP0);
    b.i32_wrap_i64();
    b.i32_const(v2Off + 8);
    b.i64_load(3, 0);
    b.i64_store(3, 24);
  }
  function emitSimdLoadPair128(b, insn, TMP0) {
    const v1Off = vregOffset(insn.rt);
    const v2Off = vregOffset(insn.rt2);
    emitSimdEffAddr(b, insn.rn, insn.offset, insn.mode, TMP0);
    b.local_set(TMP0);
    b.i32_const(v1Off);
    b.local_get(TMP0);
    b.i32_wrap_i64();
    b.i64_load(3, 0);
    b.i64_store(3, 0);
    b.i32_const(v1Off + 8);
    b.local_get(TMP0);
    b.i32_wrap_i64();
    b.i64_load(3, 8);
    b.i64_store(3, 0);
    b.i32_const(v2Off);
    b.local_get(TMP0);
    b.i32_wrap_i64();
    b.i64_load(3, 16);
    b.i64_store(3, 0);
    b.i32_const(v2Off + 8);
    b.local_get(TMP0);
    b.i32_wrap_i64();
    b.i64_load(3, 24);
    b.i64_store(3, 0);
  }
  function emitSimdStorePairSmall(b, insn, TMP0) {
    const v1Off = vregOffset(insn.rt);
    const v2Off = vregOffset(insn.rt2);
    const size = insn.size;
    emitSimdEffAddr(b, insn.rn, insn.offset, insn.mode, TMP0);
    b.local_set(TMP0);
    b.local_get(TMP0);
    b.i32_wrap_i64();
    b.i32_const(v1Off);
    b.i64_load(3, 0);
    if (size === 4) {
      b.i64_store32(2, 0);
    } else {
      b.i64_store(3, 0);
    }
    b.local_get(TMP0);
    b.i32_wrap_i64();
    b.i32_const(v2Off);
    b.i64_load(3, 0);
    if (size === 4) {
      b.i64_store32(2, size);
    } else {
      b.i64_store(3, size);
    }
  }
  function emitSimdLoadPairSmall(b, insn, TMP0) {
    const v1Off = vregOffset(insn.rt);
    const v2Off = vregOffset(insn.rt2);
    const size = insn.size;
    emitSimdEffAddr(b, insn.rn, insn.offset, insn.mode, TMP0);
    b.local_set(TMP0);
    b.i32_const(v1Off);
    b.local_get(TMP0);
    b.i32_wrap_i64();
    if (size === 4) {
      b.i64_load32_u(2, 0);
    } else {
      b.i64_load(3, 0);
    }
    b.i64_store(3, 0);
    b.i32_const(v1Off + 8);
    b.i64_const(0n);
    b.i64_store(3, 0);
    b.i32_const(v2Off);
    b.local_get(TMP0);
    b.i32_wrap_i64();
    if (size === 4) {
      b.i64_load32_u(2, size);
    } else {
      b.i64_load(3, size);
    }
    b.i64_store(3, 0);
    b.i32_const(v2Off + 8);
    b.i64_const(0n);
    b.i64_store(3, 0);
  }
  function emitSimdDupGpr(b, insn) {
    const vOff = vregOffset(insn.rd);
    const esize = insn.esize;
    const TMP0 = REG_COUNT;
    emitGetReg(b, insn.rn, esize >= 8 ? 1 : 0);
    if (esize === 1) {
      b.i64_const(0xFFn);
      b.i64_and();
      b.i64_const(0x0101010101010101n);
      b.i64_mul();
    } else if (esize === 2) {
      b.i64_const(0xFFFFn);
      b.i64_and();
      b.i64_const(0x0001000100010001n);
      b.i64_mul();
    } else if (esize === 4) {
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
      b.i64_const(0x0000000100000001n);
      b.i64_mul();
    }
    b.local_set(TMP0);
    b.i32_const(vOff);
    b.local_get(TMP0);
    b.i64_store(3, 0);
    b.i32_const(vOff + 8);
    if (insn.Q) {
      b.local_get(TMP0);
    } else {
      b.i64_const(0n);
    }
    b.i64_store(3, 0);
  }
  function emitSimdDupElem(b, insn, TMP0) {
    const srcOff = vregOffset(insn.rn);
    const dstOff = vregOffset(insn.rd);
    const esize = insn.esize;
    const index = insn.index;
    const bytePos = index * esize;
    const qwordIdx = bytePos < 8 ? 0 : 1;
    const bitPos = bytePos % 8 * 8;
    const mask = esize === 8 ? 0xFFFFFFFFFFFFFFFFn : (1n << BigInt(esize * 8)) - 1n;
    b.i32_const(srcOff + qwordIdx * 8);
    b.i64_load(3, 0);
    if (bitPos > 0) {
      b.i64_const(BigInt(bitPos));
      b.i64_shr_u();
    }
    b.i64_const(mask);
    b.i64_and();
    if (esize === 1) {
      b.i64_const(0x0101010101010101n);
      b.i64_mul();
    } else if (esize === 2) {
      b.i64_const(0x0001000100010001n);
      b.i64_mul();
    } else if (esize === 4) {
      b.i64_const(0x0000000100000001n);
      b.i64_mul();
    }
    b.local_set(TMP0);
    b.i32_const(dstOff);
    b.local_get(TMP0);
    b.i64_store(3, 0);
    if (insn.Q) {
      b.i32_const(dstOff + 8);
      b.local_get(TMP0);
      b.i64_store(3, 0);
    } else {
      b.i32_const(dstOff + 8);
      b.i64_const(0n);
      b.i64_store(3, 0);
    }
  }
  function emitSimdUmov(b, insn) {
    const srcOff = vregOffset(insn.rn);
    const esize = insn.esize;
    const index = insn.index;
    const bytePos = index * esize;
    const qwordIdx = bytePos < 8 ? 0 : 1;
    const bitPos = bytePos % 8 * 8;
    b.i32_const(srcOff + qwordIdx * 8);
    b.i64_load(3, 0);
    if (bitPos > 0) {
      b.i64_const(BigInt(bitPos));
      b.i64_shr_u();
    }
    if (esize < 8) {
      const mask = (1n << BigInt(esize * 8)) - 1n;
      b.i64_const(mask);
      b.i64_and();
    }
    emitSetReg(b, insn.rd, esize === 8 ? 1 : 0);
  }
  function emitSimdMovi(b, insn) {
    const vOff = vregOffset(insn.rd);
    const pattern = expandSimdImm(insn.imm8, insn.cmode, insn.op);
    b.i32_const(vOff);
    b.i64_const(pattern);
    b.i64_store(3, 0);
    if (insn.Q) {
      b.i32_const(vOff + 8);
      b.i64_const(pattern);
      b.i64_store(3, 0);
    } else {
      b.i32_const(vOff + 8);
      b.i64_const(0n);
      b.i64_store(3, 0);
    }
  }
  function expandSimdImm(imm8, cmode, op) {
    const cmodeH = cmode >> 1;
    if (op === 0) {
      switch (cmodeH) {
        case 0: {
          const val = BigInt(imm8);
          return val | val << 32n;
        }
        case 1: {
          const val = BigInt(imm8) << 8n;
          return val | val << 32n;
        }
        case 2: {
          const val = BigInt(imm8) << 16n;
          return val | val << 32n;
        }
        case 3: {
          const val = BigInt(imm8) << 24n;
          return val | val << 32n;
        }
        case 4: {
          const val = BigInt(imm8);
          return val | val << 16n | val << 32n | val << 48n;
        }
        case 5: {
          const val = BigInt(imm8) << 8n;
          return val | val << 16n | val << 32n | val << 48n;
        }
        case 7: {
          let val = BigInt(imm8);
          val |= val << 8n;
          val |= val << 16n;
          val |= val << 32n;
          return val;
        }
        default: {
          let val = BigInt(imm8);
          val |= val << 8n;
          val |= val << 16n;
          val |= val << 32n;
          return val;
        }
      }
    } else {
      if (cmodeH <= 3) {
        const base = expandSimdImm(imm8, cmode, 0);
        return ~base & 0xFFFFFFFFFFFFFFFFn;
      }
      return 0n;
    }
  }
  function emitSimdInsGpr(b, insn) {
    const vOff = vregOffset(insn.rd);
    const esize = insn.esize;
    const index = insn.index;
    const bytePos = index * esize;
    const qwordIdx = bytePos < 8 ? 0 : 1;
    const bitPos = bytePos % 8 * 8;
    const mask = esize === 8 ? 0xFFFFFFFFFFFFFFFFn : (1n << BigInt(esize * 8)) - 1n;
    if (esize === 8) {
      b.i32_const(vOff + qwordIdx * 8);
      emitGetReg(b, insn.rn, 1);
      b.i64_store(3, 0);
    } else {
      const addr = vOff + qwordIdx * 8;
      b.i32_const(addr);
      b.i32_const(addr);
      b.i64_load(3, 0);
      b.i64_const(~(mask << BigInt(bitPos)) & 0xFFFFFFFFFFFFFFFFn);
      b.i64_and();
      emitGetReg(b, insn.rn, esize >= 4 ? 1 : 0);
      b.i64_const(mask);
      b.i64_and();
      if (bitPos > 0) {
        b.i64_const(BigInt(bitPos));
        b.i64_shl();
      }
      b.i64_or();
      b.i64_store(3, 0);
    }
  }
  function emitFmovToFp(b, insn) {
    const vOff = vregOffset(insn.vd);
    b.i32_const(vOff);
    emitGetReg(b, insn.rn, insn.sf);
    if (insn.sf === 0) {
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
    }
    b.i64_store(3, 0);
    b.i32_const(vOff + 8);
    b.i64_const(0n);
    b.i64_store(3, 0);
  }
  function emitFmovFromFp(b, insn) {
    const vOff = vregOffset(insn.vn);
    b.i32_const(vOff);
    b.i64_load(3, 0);
    if (insn.sf === 0) {
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
    }
    emitSetReg(b, insn.rd, insn.sf);
  }
  function emitFmovFp(b, insn) {
    const srcOff = vregOffset(insn.vn);
    const dstOff = vregOffset(insn.vd);
    b.i32_const(dstOff);
    b.i32_const(srcOff);
    b.i64_load(3, 0);
    if (insn.ftype === 0) {
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
    }
    b.i64_store(3, 0);
    b.i32_const(dstOff + 8);
    b.i64_const(0n);
    b.i64_store(3, 0);
  }
  function emitFpIntConv(b, insn, TMP0, helpers) {
    if (!helpers) return;
    if (insn.kind === "scvtf_int" || insn.kind === "ucvtf_int") {
      emitGetReg(b, insn.rn, insn.sf);
      if (insn.kind === "scvtf_int") {
        b.call(insn.sf ? helpers.f64_from_s64Idx : helpers.f64_from_s32Idx);
      } else {
        b.call(insn.sf ? helpers.f64_from_u64Idx : helpers.f64_from_u32Idx);
      }
      b.local_set(TMP0);
      const vOff = vregOffset(insn.vd);
      b.i32_const(vOff);
      if (insn.ftype === 0) {
        b.local_get(TMP0);
        b.call(helpers.f64_to_f32Idx);
      } else {
        b.local_get(TMP0);
      }
      b.i64_store(3, 0);
      b.i32_const(vOff + 8);
      b.i64_const(0n);
      b.i64_store(3, 0);
    } else {
      const vOff = vregOffset(insn.vn);
      b.i32_const(vOff);
      b.i64_load(3, 0);
      if (insn.ftype === 0) {
        b.call(helpers.f32_to_f64Idx);
      }
      if (insn.kind === "fcvtzs_int") {
        b.call(insn.sf ? helpers.f64_to_s64Idx : helpers.f64_to_s32Idx);
      } else {
        b.call(insn.sf ? helpers.f64_to_u64Idx : helpers.f64_to_u32Idx);
      }
      emitSetReg(b, insn.rd, insn.sf);
    }
  }
  function emitFpArith2(b, insn, TMP0, helpers) {
    if (!helpers) return;
    const vnOff = vregOffset(insn.vn);
    const vmOff = vregOffset(insn.vm);
    const vdOff = vregOffset(insn.vd);
    b.i32_const(vnOff);
    b.i64_load(3, 0);
    if (insn.ftype === 0) {
      b.call(helpers.f32_to_f64Idx);
    }
    b.i32_const(vmOff);
    b.i64_load(3, 0);
    if (insn.ftype === 0) {
      b.call(helpers.f32_to_f64Idx);
    }
    switch (insn.kind) {
      case "fmul":
        b.call(helpers.f64_mulIdx);
        break;
      case "fdiv":
        b.call(helpers.f64_divIdx);
        break;
      case "fadd":
        b.call(helpers.f64_addIdx);
        break;
      case "fsub":
        b.call(helpers.f64_subIdx);
        break;
    }
    b.local_set(TMP0);
    b.i32_const(vdOff);
    if (insn.ftype === 0) {
      b.local_get(TMP0);
      b.call(helpers.f64_to_f32Idx);
    } else {
      b.local_get(TMP0);
    }
    b.i64_store(3, 0);
    b.i32_const(vdOff + 8);
    b.i64_const(0n);
    b.i64_store(3, 0);
  }
  function emitFpMisc1(b, insn, TMP0, helpers) {
    if (!helpers) return;
    const vnOff = vregOffset(insn.vn);
    const vdOff = vregOffset(insn.vd);
    b.i32_const(vnOff);
    b.i64_load(3, 0);
    if (insn.ftype === 0) {
      b.call(helpers.f32_to_f64Idx);
    }
    switch (insn.kind) {
      case "fabs":
        b.call(helpers.f64_absIdx);
        break;
      case "fneg":
        b.call(helpers.f64_negIdx);
        break;
    }
    b.local_set(TMP0);
    b.i32_const(vdOff);
    if (insn.ftype === 0) {
      b.local_get(TMP0);
      b.call(helpers.f64_to_f32Idx);
    } else {
      b.local_get(TMP0);
    }
    b.i64_store(3, 0);
    b.i32_const(vdOff + 8);
    b.i64_const(0n);
    b.i64_store(3, 0);
  }
  function emitFcvt(b, insn, TMP0, helpers) {
    if (!helpers) return;
    const vnOff = vregOffset(insn.vn);
    const vdOff = vregOffset(insn.vd);
    b.i32_const(vnOff);
    b.i64_load(3, 0);
    if (insn.ftype === 0) {
      b.call(helpers.f32_to_f64Idx);
    }
    b.local_set(TMP0);
    b.i32_const(vdOff);
    if (insn.destFtype === 0) {
      b.local_get(TMP0);
      b.call(helpers.f64_to_f32Idx);
    } else {
      b.local_get(TMP0);
    }
    b.i64_store(3, 0);
    b.i32_const(vdOff + 8);
    b.i64_const(0n);
    b.i64_store(3, 0);
  }
  function emitSimdThreeSameGeneric(b, insn, TMP0, helpers) {
    if (!helpers) return;
    let op = 0, size = 0, extra = 0;
    switch (insn.kind) {
      case "simd_add":
        op = SIMD_ADD;
        break;
      case "simd_sub":
        op = SIMD_SUB;
        break;
      case "simd_cmeq_reg":
        op = SIMD_CMEQ;
        break;
      case "simd_cmgt_reg":
        op = SIMD_CMGT;
        break;
      case "simd_cmhi_reg":
        op = SIMD_CMHI;
        break;
      case "simd_cmge_reg":
        op = SIMD_CMGE;
        break;
      case "simd_cmhs_reg":
        op = SIMD_CMHS;
        break;
      case "simd_umaxp":
        op = SIMD_UMAXP;
        break;
      case "simd_addp":
        op = SIMD_ADDP;
        break;
      case "simd_tbl":
        op = SIMD_TBL;
        extra = insn.len ?? 0;
        break;
      case "simd_zip1":
        op = SIMD_ZIP1;
        break;
      case "simd_zip2":
        op = SIMD_ZIP2;
        break;
      case "simd_uzp1":
        op = SIMD_UZP1;
        break;
      case "simd_uzp2":
        op = SIMD_UZP2;
        break;
      case "simd_trn1":
        op = SIMD_TRN1;
        break;
      case "simd_trn2":
        op = SIMD_TRN2;
        break;
      default:
        return;
    }
    size = insn.size ?? 0;
    const packed = BigInt(op) << 56n | BigInt(insn.Q) << 48n | BigInt(size) << 40n | BigInt(extra) << 32n | BigInt(insn.rd) << 24n | BigInt(insn.rn) << 16n | BigInt(insn.rm) << 8n;
    b.i64_const(packed);
    b.i64_const(0n);
    b.call(helpers.simd_opIdx);
    b.drop();
  }
  function emitFmovImm(b, insn) {
    const vdOff = vregOffset(insn.vd);
    b.i32_const(vdOff);
    b.i64_const(insn.bits);
    b.i64_store(3, 0);
    b.i32_const(vdOff + 8);
    b.i64_const(0n);
    b.i64_store(3, 0);
  }
  function emitFpCmp(b, insn, helpers) {
    if (!helpers) return;
    const vnOff = vregOffset(insn.vn);
    b.i32_const(vnOff);
    b.i64_load(3, 0);
    if (insn.ftype === 0) {
      b.call(helpers.f32_to_f64Idx);
    }
    if (insn.withZero) {
      b.i64_const(0n);
    } else {
      const vmOff = vregOffset(insn.vm);
      b.i32_const(vmOff);
      b.i64_load(3, 0);
      if (insn.ftype === 0) {
        b.call(helpers.f32_to_f64Idx);
      }
    }
    b.call(helpers.f64_cmpIdx);
    b.local_set(NZCV_LOCAL);
  }
  function emitFpMulAdd(b, insn, TMP0, helpers) {
    if (!helpers) return;
    const vnOff = vregOffset(insn.vn);
    const vmOff = vregOffset(insn.vm);
    const vaOff = vregOffset(insn.va);
    const vdOff = vregOffset(insn.vd);
    const isF32 = insn.ftype === 0;
    if (insn.kind === "fnmsub") {
      b.i32_const(vnOff);
      b.i64_load(3, 0);
      if (isF32) {
        b.call(helpers.f32_to_f64Idx);
      }
      b.i32_const(vmOff);
      b.i64_load(3, 0);
      if (isF32) {
        b.call(helpers.f32_to_f64Idx);
      }
      b.call(helpers.f64_mulIdx);
      b.i32_const(vaOff);
      b.i64_load(3, 0);
      if (isF32) {
        b.call(helpers.f32_to_f64Idx);
      }
      b.call(helpers.f64_subIdx);
    } else {
      b.i32_const(vaOff);
      b.i64_load(3, 0);
      if (isF32) {
        b.call(helpers.f32_to_f64Idx);
      }
      b.i32_const(vnOff);
      b.i64_load(3, 0);
      if (isF32) {
        b.call(helpers.f32_to_f64Idx);
      }
      b.i32_const(vmOff);
      b.i64_load(3, 0);
      if (isF32) {
        b.call(helpers.f32_to_f64Idx);
      }
      b.call(helpers.f64_mulIdx);
      if (insn.kind === "fmadd" || insn.kind === "fnmadd") {
        b.call(helpers.f64_addIdx);
      } else {
        b.call(helpers.f64_subIdx);
      }
      if (insn.kind === "fnmadd") {
        b.call(helpers.f64_negIdx);
      }
    }
    b.local_set(TMP0);
    b.i32_const(vdOff);
    if (isF32) {
      b.local_get(TMP0);
      b.call(helpers.f64_to_f32Idx);
    } else {
      b.local_get(TMP0);
    }
    b.i64_store(3, 0);
    b.i32_const(vdOff + 8);
    b.i64_const(0n);
    b.i64_store(3, 0);
  }
  function emitSimdShiftLong(b, insn, TMP0) {
    const srcOff = vregOffset(insn.rn);
    const dstOff = vregOffset(insn.rd);
    const esize = insn.esize;
    const dstEsize = esize * 2;
    const shift = insn.shift;
    const signed = insn.kind === "simd_sshll";
    const srcBits = esize * 8;
    const dstBits = dstEsize * 8;
    const numElems = insn.Q === 0 ? 8 / esize : 8 / esize;
    const srcBase = insn.Q === 0 ? 0 : 8;
    for (let i = numElems - 1; i >= 0; i--) {
      const srcBytePos = srcBase + i * esize;
      const dstBytePos = i * dstEsize;
      const srcQword = srcBytePos < 8 ? 0 : 1;
      const srcBitPos = srcBytePos % 8 * 8;
      const dstQword = dstBytePos < 8 ? 0 : 1;
      const dstBitPos = dstBytePos % 8 * 8;
      b.i32_const(srcOff + srcQword * 8);
      b.i64_load(3, 0);
      if (srcBitPos > 0) {
        b.i64_const(BigInt(srcBitPos));
        b.i64_shr_u();
      }
      if (esize < 8) {
        const mask = (1n << BigInt(srcBits)) - 1n;
        b.i64_const(mask);
        b.i64_and();
      }
      if (signed && esize < 8) {
        const extShift = BigInt(64 - srcBits);
        b.i64_const(extShift);
        b.i64_shl();
        b.i64_const(extShift);
        b.i64_shr_s();
      }
      if (shift > 0) {
        b.i64_const(BigInt(shift));
        b.i64_shl();
      }
      if (dstEsize < 8) {
        const dstMask = (1n << BigInt(dstBits)) - 1n;
        b.i64_const(dstMask);
        b.i64_and();
      }
      b.local_set(TMP0);
      if (dstBitPos === 0 && dstEsize >= 8) {
        b.i32_const(dstOff + dstQword * 8);
        b.local_get(TMP0);
        b.i64_store(3, 0);
      } else {
        const dstMask = (1n << BigInt(dstBits)) - 1n;
        const addr = dstOff + dstQword * 8;
        b.i32_const(addr);
        b.i32_const(addr);
        b.i64_load(3, 0);
        b.i64_const(~(dstMask << BigInt(dstBitPos)) & 0xFFFFFFFFFFFFFFFFn);
        b.i64_and();
        b.local_get(TMP0);
        if (dstBitPos > 0) {
          b.i64_const(BigInt(dstBitPos));
          b.i64_shl();
        }
        b.i64_or();
        b.i64_store(3, 0);
      }
    }
  }

  // src/syscall.ts
  var ENOSYS = 38;
  var EBADF = 9;
  var EINVAL = 22;
  var ENOENT = 2;
  var EPERM = 1;
  var ECHILD = 10;
  var ESPIPE = 29;
  var MAX_FDS = 1024;
  var SIG_DFL = 0n;
  var SIG_IGN = 1n;
  var SIGPIPE = 13;
  var SIGCHLD = 17;
  var SIGNAL_TRAMPOLINE = 4096;
  var SIG_IGNORE_DEFAULT = /* @__PURE__ */ new Set([SIGCHLD, 17, 20]);
  var SIG_STOP_DEFAULT = /* @__PURE__ */ new Set([19, 20, 21, 22]);
  var SIGNAL_FRAME_SIZE = 288;
  var TCGETS = 21505;
  var TCSETS = 21506;
  var TIOCGWINSZ = 21523;
  var TIOCSWINSZ = 21524;
  var F_DUPFD = 0;
  var F_GETFD = 1;
  var F_SETFD = 2;
  var F_GETFL = 3;
  var F_SETFL = 4;
  var F_DUPFD_CLOEXEC = 1030;
  var SyscallHandler = class {
    brk;
    mmapBase = 0x2000000n;
    // mmap region starts at 32MB
    cwd = "/";
    tpidr = 0n;
    umask = 18;
    fdCloexec = /* @__PURE__ */ new Set();
    fdSig = /* @__PURE__ */ new Map();
    pendingSignals = [];
    // queue of signals to deliver
    epollSets;
    signalHandlers;
    signalMask;
    /** True if bridge calls can be handled (either direct bridge or Worker kernel relay) */
    bridgeAvailable = false;
    currentPid = 1;
    currentPpid = 0;
    termCols = 80;
    termRows = 24;
    vfs;
    io;
    bridge;
    fdFlags = /* @__PURE__ */ new Map();
    // fd -> flags
    fdTable = /* @__PURE__ */ new Map();
    // unified fd table
    nativeRefs = /* @__PURE__ */ new Map();
    // nativeFd -> refcount
    nextFd = 3;
    // next available fd (0-2 are stdio)
    lastWaitStatus;
    // exit code from last child (set by dispatch loop)
    lastWaitPid;
    // PID of last exited child
    constructor(brkStart, io, vfs2, bridge) {
      this.brk = brkStart;
      this.io = io;
      this.vfs = vfs2;
      this.bridge = bridge;
    }
    /**
     * Check if a guest path should be routed to the native Swift filesystem.
     * Returns the resolved absolute path if bridging, or null if staying in JS.
     *
     * JS VFS handles: /proc, /dev, /tmp (virtual/ephemeral), /bin, /etc, /usr (busybox + config)
     * Native Swift handles: /root, /home, /var (persistent user storage)
     */
    bridgePath(path) {
      if (!this.bridge?.isNative) return null;
      let resolved = path;
      if (!path.startsWith("/")) {
        const cwd = this.vfs?.getcwd() ?? this.cwd;
        resolved = cwd === "/" ? "/" + path : cwd + "/" + path;
      }
      if (resolved.startsWith("/root") || resolved.startsWith("/home") || resolved.startsWith("/var")) {
        return resolved;
      }
      return null;
    }
    setTermSize(cols, rows) {
      this.termCols = cols;
      this.termRows = rows;
    }
    /** Get the current mmap high-water mark (for fork snapshot sizing). */
    getMmapBase() {
      return this.mmapBase;
    }
    /** Write the signal return trampoline into guest memory. Call once during init. */
    installSignalTrampoline(memory2) {
      const view = new DataView(memory2.buffer);
      const base = SIGNAL_TRAMPOLINE + GUEST_MEM_BASE;
      view.setUint32(base, 3531608424, true);
      view.setUint32(base + 4, 3556769793, true);
    }
    /**
     * Deliver a pending signal to the guest process.
     * Pushes a signal frame onto the guest stack and redirects PC to the handler.
     * Returns true if a signal was delivered (caller should update PC).
     */
    deliverSignal(cpu2, memory2, signum) {
      if (!this.signalHandlers) return { delivered: false };
      const handler = this.signalHandlers.get(signum);
      const handlerAddr = handler?.handler ?? SIG_DFL;
      if (handlerAddr === SIG_IGN) return { delivered: true, action: "continue" };
      if (handlerAddr === SIG_DFL) {
        if (SIG_IGNORE_DEFAULT.has(signum)) return { delivered: true, action: "continue" };
        if (SIG_STOP_DEFAULT.has(signum)) return { delivered: true, action: "continue" };
        return { delivered: true, action: "exit", exitCode: 128 + signum };
      }
      if (this.signalMask && this.signalMask & 1n << BigInt(signum - 1)) {
        return { delivered: false };
      }
      const view = new DataView(memory2.buffer);
      const currentSP = cpu2.getSP();
      const frameSP = currentSP - BigInt(SIGNAL_FRAME_SIZE) & ~15n;
      const frameBase = Number(frameSP) + GUEST_MEM_BASE;
      for (let r = 0; r < 31; r++) {
        view.setBigUint64(frameBase + r * 8, cpu2.getReg(r), true);
      }
      view.setBigUint64(frameBase + 248, currentSP, true);
      view.setBigUint64(frameBase + 256, cpu2.getPC(), true);
      view.setBigUint64(frameBase + 264, cpu2.getNZCV(), true);
      view.setUint32(frameBase + 272, signum, true);
      cpu2.setSP(frameSP);
      cpu2.setPC(handlerAddr);
      cpu2.setReg(0, BigInt(signum));
      cpu2.setReg(30, BigInt(SIGNAL_TRAMPOLINE));
      return { delivered: true, action: "continue" };
    }
    /** Handle rt_sigreturn: restore registers from signal frame. */
    handleSigreturn(cpu2, memory2) {
      const view = new DataView(memory2.buffer);
      const frameSP = cpu2.getSP();
      const frameBase = Number(frameSP) + GUEST_MEM_BASE;
      for (let r = 0; r < 31; r++) {
        cpu2.setReg(r, view.getBigUint64(frameBase + r * 8, true));
      }
      const savedSP = view.getBigUint64(frameBase + 248, true);
      const savedPC = view.getBigUint64(frameBase + 256, true);
      const savedNZCV = view.getBigUint64(frameBase + 264, true);
      cpu2.setSP(savedSP);
      cpu2.setNZCV(savedNZCV);
      return savedPC;
    }
    /** Queue a signal for delivery. */
    raiseSignal(signum) {
      this.pendingSignals.push(signum);
    }
    /** Check if a custom signal handler is installed (not SIG_DFL or SIG_IGN). */
    hasSignalHandler(signum) {
      if (!this.signalHandlers) return false;
      const handler = this.signalHandlers.get(signum);
      if (!handler) return false;
      return handler.handler !== SIG_DFL && handler.handler !== SIG_IGN;
    }
    /** Parse a sockaddr structure from guest memory into JS object. */
    parseSockaddr(bytes, addrPtr, addrLen) {
      const base = addrPtr + GUEST_MEM_BASE;
      const family = bytes[base] | bytes[base + 1] << 8;
      if (family === 2 && addrLen >= 8) {
        const port = bytes[base + 2] << 8 | bytes[base + 3];
        const a = bytes[base + 4], b = bytes[base + 5], c = bytes[base + 6], d = bytes[base + 7];
        return { family, host: `${a}.${b}.${c}.${d}`, port };
      }
      if (family === 10 && addrLen >= 28) {
        const port = bytes[base + 2] << 8 | bytes[base + 3];
        const addr = bytes.slice(base + 8, base + 24);
        const isV4Mapped = addr[10] === 255 && addr[11] === 255 && addr.slice(0, 10).every((b) => b === 0);
        if (isV4Mapped) {
          return { family: 2, host: `${addr[12]}.${addr[13]}.${addr[14]}.${addr[15]}`, port };
        }
        const parts = [];
        for (let i = 0; i < 16; i += 2) {
          parts.push((addr[i] << 8 | addr[i + 1]).toString(16));
        }
        return { family, host: parts.join(":"), port };
      }
      return { family, host: "0.0.0.0", port: 0 };
    }
    /** Write a sockaddr structure to guest memory. Returns bytes written. */
    writeSockaddr(view, addrPtr, addr) {
      const base = addrPtr + GUEST_MEM_BASE;
      if (addr.family === 2) {
        view.setUint16(base, 2, true);
        view.setUint8(base + 2, addr.port >> 8 & 255);
        view.setUint8(base + 3, addr.port & 255);
        const parts = addr.host.split(".").map(Number);
        view.setUint8(base + 4, parts[0]);
        view.setUint8(base + 5, parts[1]);
        view.setUint8(base + 6, parts[2]);
        view.setUint8(base + 7, parts[3]);
        for (let i = 8; i < 16; i++) view.setUint8(base + i, 0);
        return 16;
      }
      view.setUint16(base, addr.family, true);
      return 0;
    }
    /** Register an fd entry directly (used by Worker for kernel pipe fds). */
    registerFd(fd, entry) {
      this.fdTable.set(fd, entry);
      if (fd >= this.nextFd) this.nextFd = fd + 1;
    }
    /** Get state for process snapshot */
    getState() {
      return {
        brk: this.brk,
        mmapBase: this.mmapBase,
        cwd: this.vfs ? this.vfs.getcwd() : this.cwd,
        fdFlags: new Map(this.fdFlags),
        fdTable: new Map(this.fdTable),
        nativeRefs: new Map(this.nativeRefs),
        nextFd: this.nextFd,
        vfsFdSnapshot: this.vfs ? this.vfs.snapshotFds() : void 0
      };
    }
    /** Restore state from process snapshot */
    restoreState(state) {
      this.brk = state.brk;
      this.mmapBase = state.mmapBase;
      if (this.vfs) {
        this.vfs.chdir(state.cwd);
        if (state.vfsFdSnapshot) this.vfs.restoreFds(state.vfsFdSnapshot);
      } else {
        this.cwd = state.cwd;
      }
      this.fdFlags = new Map(state.fdFlags);
      if (state.fdTable) this.fdTable = new Map(state.fdTable);
      if (state.nativeRefs) this.nativeRefs = new Map(state.nativeRefs);
      if (state.nextFd !== void 0) this.nextFd = state.nextFd;
    }
    /** Reset state for execve (new process image) */
    resetForExec(brkStart) {
      this.brk = brkStart;
      this.mmapBase = 0x2000000n;
      const preserved = /* @__PURE__ */ new Map();
      for (const [fd, entry] of this.fdTable) {
        if (fd <= 2) preserved.set(fd, entry);
      }
      this.fdTable = preserved;
      this.nextFd = 3;
      if (this.vfs) this.vfs.closeAllExcept(0, 1, 2);
    }
    /** Allocate next available fd */
    allocFd() {
      return this.nextFd++;
    }
    /** Resolve fd: check fdTable for overrides (dup2'd pipes, etc.), else fall through to default */
    getFdEntry(fd) {
      return this.fdTable.get(fd);
    }
    handle(cpu2, memory2) {
      const nr = Number(cpu2.getReg(8));
      const a0 = cpu2.getReg(0);
      const a1 = cpu2.getReg(1);
      const a2 = cpu2.getReg(2);
      const a3 = cpu2.getReg(3);
      const a4 = cpu2.getReg(4);
      const a5 = cpu2.getReg(5);
      const view = new DataView(memory2.buffer);
      const bytes = new Uint8Array(memory2.buffer);
      let ret;
      switch (nr) {
        // ── Process lifecycle ──
        case 93:
        // exit
        case 94:
          return { status: "exit", code: Number(a0) };
        case 220: {
          const flags = Number(a0);
          const SIGCHLD2 = 17;
          if (flags & SIGCHLD2) {
            const svcPC = view.getBigUint64(PC_OFFSET, true);
            return { status: "fork", resumePC: svcPC + 4n };
          }
          ret = -BigInt(ENOSYS);
          break;
        }
        case 221: {
          const pathname = readGuestString(bytes, Number(a0));
          const argvPtrs = [];
          let argvAddr = Number(a1);
          while (true) {
            const ptr = Number(view.getBigUint64(argvAddr + GUEST_MEM_BASE, true));
            if (ptr === 0) break;
            argvPtrs.push(readGuestString(bytes, ptr));
            argvAddr += 8;
          }
          const envpPtrs = [];
          let envpAddr = Number(a2);
          while (true) {
            const ptr = Number(view.getBigUint64(envpAddr + GUEST_MEM_BASE, true));
            if (ptr === 0) break;
            envpPtrs.push(readGuestString(bytes, ptr));
            envpAddr += 8;
          }
          return { status: "execve", path: pathname, argv: argvPtrs, envp: envpPtrs };
        }
        case 95: {
          const WNOHANG = 1;
          const widOptions = Number(a3);
          if (this.lastWaitStatus !== void 0) {
            const exitCode = this.lastWaitStatus;
            this.lastWaitStatus = void 0;
            const infop = Number(a2);
            if (infop !== 0) {
              const base = infop + GUEST_MEM_BASE;
              for (let i = 0; i < 128; i += 8) view.setBigUint64(base + i, 0n, true);
              view.setInt32(base, 17, true);
              view.setInt32(base + 8, 1, true);
              view.setInt32(base + 12, this.lastWaitPid ?? 1, true);
              view.setInt32(base + 24, exitCode, true);
            }
            ret = 0n;
            this.lastWaitPid = void 0;
          } else if (widOptions & WNOHANG) {
            const infop = Number(a2);
            if (infop !== 0) {
              const base = infop + GUEST_MEM_BASE;
              for (let i = 0; i < 128; i += 8) view.setBigUint64(base + i, 0n, true);
            }
            ret = 0n;
          } else {
            ret = -BigInt(ECHILD);
          }
          break;
        }
        case 260: {
          const wstatus = Number(a1);
          const WNOHANG = 1;
          const options = Number(a2);
          if (this.lastWaitStatus !== void 0) {
            const exitCode = this.lastWaitStatus;
            this.lastWaitStatus = void 0;
            if (wstatus !== 0) {
              view.setUint32(wstatus + GUEST_MEM_BASE, (exitCode & 255) << 8, true);
            }
            ret = BigInt(this.lastWaitPid || 1);
            this.lastWaitPid = void 0;
          } else if (options & WNOHANG) {
            ret = 0n;
          } else {
            ret = -BigInt(ECHILD);
          }
          break;
        }
        // ── Read/Write ──
        case 63: {
          const fd = Number(a0);
          const buf = Number(a1);
          const count = Number(a2);
          const readEntry = this.getFdEntry(fd);
          if (readEntry) {
            if (readEntry.type === "pipe_write") {
              ret = -BigInt(EBADF);
              break;
            }
            if (readEntry.type === "pipe_read") {
              const pipe = readEntry.pipe;
              if (pipe.data.length === 0) {
                if (pipe.writeClosed) {
                  ret = 0n;
                } else {
                  ret = 0n;
                }
              } else {
                const chunk = pipe.data.shift();
                const toRead = Math.min(count, chunk.length);
                bytes.set(chunk.subarray(0, toRead), buf + GUEST_MEM_BASE);
                if (toRead < chunk.length) {
                  pipe.data.unshift(chunk.subarray(toRead));
                }
                ret = BigInt(toRead);
              }
              break;
            }
            if (readEntry.type === "stdio" && readEntry.fd === 0) {
              const data = this.io.readStdin(count);
              if (data === null) return { status: "blocked_read", fd: 0 };
              bytes.set(data, buf + GUEST_MEM_BASE);
              ret = BigInt(data.length);
              break;
            }
            if (readEntry.type === "vfs") {
              ret = this.vfs ? BigInt(this.vfs.read(readEntry.fd, buf, count, view)) : -BigInt(EBADF);
              break;
            }
            if (readEntry.type === "native") {
              return {
                status: "bridge",
                method: "fs_read",
                args: { fd: readEntry.nativeFd, count },
                apply: (resp) => {
                  if (resp.ret < 0) {
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                  } else if (resp.data) {
                    const decoded = Uint8Array.from(atob(resp.data), (c) => c.charCodeAt(0));
                    bytes.set(decoded, buf + GUEST_MEM_BASE);
                    cpu2.setReg(0, BigInt(decoded.length));
                  } else {
                    cpu2.setReg(0, BigInt(resp.ret));
                  }
                }
              };
            }
            if (readEntry.type === "socket") {
              return {
                status: "bridge",
                method: "net_read",
                args: { socketId: readEntry.socketId, count },
                apply: (resp) => {
                  if (resp.ret < 0) {
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                  } else if (resp.data) {
                    const decoded = Uint8Array.from(atob(resp.data), (c) => c.charCodeAt(0));
                    bytes.set(decoded, buf + GUEST_MEM_BASE);
                    cpu2.setReg(0, BigInt(decoded.length));
                  } else {
                    cpu2.setReg(0, BigInt(resp.ret));
                  }
                }
              };
            }
            ret = -BigInt(EBADF);
            break;
          }
          if (fd === 0) {
            const data = this.io.readStdin(count);
            if (data === null) return { status: "blocked_read", fd: 0 };
            bytes.set(data, buf + GUEST_MEM_BASE);
            ret = BigInt(data.length);
          } else if (this.vfs) {
            ret = BigInt(this.vfs.read(fd, buf, count, view));
          } else {
            ret = -BigInt(EBADF);
          }
          break;
        }
        case 64: {
          const fd = Number(a0);
          const buf = Number(a1);
          const count = Number(a2);
          const writeEntry = this.getFdEntry(fd);
          if (writeEntry) {
            if (writeEntry.type === "pipe_read") {
              ret = -BigInt(EBADF);
              break;
            }
            if (writeEntry.type === "pipe_write") {
              if (writeEntry.pipe.readClosed) {
                this.raiseSignal(SIGPIPE);
                ret = -BigInt(32);
                break;
              }
              const data = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
              writeEntry.pipe.data.push(data);
              ret = BigInt(count);
              break;
            }
            if (writeEntry.type === "stdio") {
              const data = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
              if (writeEntry.fd === 1) this.io.writeStdout(data);
              else if (writeEntry.fd === 2) this.io.writeStderr(data);
              ret = BigInt(count);
              break;
            }
            if (writeEntry.type === "vfs") {
              ret = this.vfs ? BigInt(this.vfs.write(writeEntry.fd, buf, count, view)) : -BigInt(EBADF);
              break;
            }
            if (writeEntry.type === "native") {
              const writeData = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
              const b64 = btoa(String.fromCharCode(...writeData));
              return {
                status: "bridge",
                method: "fs_write",
                args: { fd: writeEntry.nativeFd, data: b64 },
                apply: (resp) => {
                  if (resp.ret < 0) {
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                  } else {
                    cpu2.setReg(0, BigInt(resp.ret));
                  }
                }
              };
            }
            if (writeEntry.type === "socket") {
              const sockWriteData = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
              const sockText = new TextDecoder().decode(sockWriteData);
              console.log(`[net] write(fd=${fd}, ${count}B): ${JSON.stringify(sockText)}`);
              const sockB64 = btoa(String.fromCharCode(...sockWriteData));
              return {
                status: "bridge",
                method: "net_write",
                args: { socketId: writeEntry.socketId, data: sockB64 },
                apply: (resp) => {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                }
              };
            }
            ret = -BigInt(EBADF);
            break;
          }
          if (fd === 1) {
            const data = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
            this.io.writeStdout(data);
            ret = BigInt(count);
          } else if (fd === 2) {
            const data = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
            this.io.writeStderr(data);
            ret = BigInt(count);
          } else if (this.vfs) {
            ret = BigInt(this.vfs.write(fd, buf, count, view));
          } else {
            ret = -BigInt(EBADF);
          }
          break;
        }
        case 66: {
          const fd = Number(a0);
          const iovBase = Number(a1);
          const iovcnt = Number(a2);
          let totalWritten = 0;
          const writevEntry = this.getFdEntry(fd);
          if (writevEntry?.type === "socket") {
            let totalLen = 0;
            for (let i = 0; i < iovcnt; i++) {
              totalLen += Number(view.getBigUint64(iovBase + i * 16 + 8 + GUEST_MEM_BASE, true));
            }
            const combined = new Uint8Array(totalLen);
            let off = 0;
            for (let i = 0; i < iovcnt; i++) {
              const base = Number(view.getBigUint64(iovBase + i * 16 + GUEST_MEM_BASE, true));
              const len = Number(view.getBigUint64(iovBase + i * 16 + 8 + GUEST_MEM_BASE, true));
              if (len === 0) continue;
              combined.set(bytes.slice(base + GUEST_MEM_BASE, base + GUEST_MEM_BASE + len), off);
              off += len;
            }
            const b64 = btoa(String.fromCharCode(...combined));
            const text = new TextDecoder().decode(combined);
            const iovDetails = [];
            for (let i = 0; i < iovcnt; i++) {
              const iovBase2 = Number(view.getBigUint64(iovBase + i * 16 + GUEST_MEM_BASE, true));
              const iovLen2 = Number(view.getBigUint64(iovBase + i * 16 + 8 + GUEST_MEM_BASE, true));
              iovDetails.push(`{0x${iovBase2.toString(16)},${iovLen2}}`);
            }
            if (totalLen > 30 && text.includes("User-Agent")) {
              const strAddr = Number(view.getBigUint64(iovBase + 16 + GUEST_MEM_BASE, true));
              const strLen = Number(view.getBigUint64(iovBase + 24 + GUEST_MEM_BASE, true));
              console.log(`[net] DEBUG iov[1] str@0x${strAddr.toString(16)} len=${strLen}`);
              const strBytes = [];
              for (let i = 0; i < Math.min(strLen + 4, 35); i++) {
                strBytes.push(bytes[strAddr + GUEST_MEM_BASE + i].toString(16).padStart(2, "0"));
              }
              console.log(`[net] DEBUG bytes: ${strBytes.join(" ")}`);
              console.log(`[net] DEBUG byte@+4: ${bytes[strAddr + GUEST_MEM_BASE + 4]} (should be 0)`);
            }
            console.log(`[net] writev(fd=${fd}, ${totalLen}B, iovs=[${iovDetails.join(",")}]): ${JSON.stringify(text)}`);
            return {
              status: "bridge",
              method: "net_write",
              args: { socketId: writevEntry.socketId, data: b64 },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          if (writevEntry?.type === "native") {
            let totalLen = 0;
            for (let i = 0; i < iovcnt; i++) {
              totalLen += Number(view.getBigUint64(iovBase + i * 16 + 8 + GUEST_MEM_BASE, true));
            }
            const combined = new Uint8Array(totalLen);
            let off = 0;
            for (let i = 0; i < iovcnt; i++) {
              const base = Number(view.getBigUint64(iovBase + i * 16 + GUEST_MEM_BASE, true));
              const len = Number(view.getBigUint64(iovBase + i * 16 + 8 + GUEST_MEM_BASE, true));
              if (len === 0) continue;
              combined.set(bytes.slice(base + GUEST_MEM_BASE, base + GUEST_MEM_BASE + len), off);
              off += len;
            }
            const b64 = btoa(String.fromCharCode(...combined));
            return {
              status: "bridge",
              method: "fs_write",
              args: { fd: writevEntry.nativeFd, data: b64 },
              apply: (resp) => {
                if (resp.ret < 0) {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                } else {
                  cpu2.setReg(0, BigInt(resp.ret));
                }
              }
            };
          }
          for (let i = 0; i < iovcnt; i++) {
            const base = Number(view.getBigUint64(iovBase + i * 16 + GUEST_MEM_BASE, true));
            const len = Number(view.getBigUint64(iovBase + i * 16 + 8 + GUEST_MEM_BASE, true));
            if (len === 0) continue;
            const data = bytes.slice(base + GUEST_MEM_BASE, base + GUEST_MEM_BASE + len);
            if (writevEntry) {
              if (writevEntry.type === "pipe_write") {
                writevEntry.pipe.data.push(data);
              } else if (writevEntry.type === "stdio") {
                if (writevEntry.fd === 1) this.io.writeStdout(data);
                else if (writevEntry.fd === 2) this.io.writeStderr(data);
              } else if (writevEntry.type === "vfs" && this.vfs) {
                this.vfs.write(writevEntry.fd, base, len, view);
              }
            } else if (fd === 1) {
              this.io.writeStdout(data);
            } else if (fd === 2) {
              this.io.writeStderr(data);
            } else if (this.vfs) {
              this.vfs.write(fd, base, len, view);
            }
            totalWritten += len;
          }
          ret = BigInt(totalWritten);
          break;
        }
        case 65: {
          const rvFd = Number(a0);
          const rvIov = Number(a1);
          const rvCnt = Number(a2);
          let rvTotal = 0;
          for (let i = 0; i < rvCnt; i++) {
            const base = Number(view.getBigUint64(rvIov + i * 16 + GUEST_MEM_BASE, true));
            const len = Number(view.getBigUint64(rvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
            if (len === 0) continue;
            const origA0 = a0, origA1 = a1, origA2 = a2;
            cpu2.setReg(0, BigInt(rvFd));
            cpu2.setReg(1, BigInt(base));
            cpu2.setReg(2, BigInt(len));
            if (this.vfs) {
              const n = this.vfs.read(rvFd, base, len, view);
              if (n < 0) {
                if (rvTotal === 0) {
                  ret = BigInt(n);
                } else {
                  ret = BigInt(rvTotal);
                }
                break;
              }
              rvTotal += n;
              if (n < len) break;
            }
          }
          if (ret === void 0) ret = BigInt(rvTotal);
          break;
        }
        case 67: {
          const prFd = Number(a0);
          const prBuf = Number(a1);
          const prCount = Number(a2);
          const prOffset = BigInt.asIntN(64, a3);
          if (this.vfs) {
            const savedPos = this.vfs.lseek(prFd, 0n, 1);
            if (savedPos < 0n) {
              ret = savedPos;
              break;
            }
            this.vfs.lseek(prFd, prOffset, 0);
            const n = this.vfs.read(prFd, prBuf, prCount, view);
            this.vfs.lseek(prFd, savedPos, 0);
            ret = BigInt(n);
          } else {
            ret = -BigInt(EBADF);
          }
          break;
        }
        case 68: {
          const pwFd = Number(a0);
          const pwBuf = Number(a1);
          const pwCount = Number(a2);
          const pwOffset = BigInt.asIntN(64, a3);
          if (this.vfs) {
            const savedPos = this.vfs.lseek(pwFd, 0n, 1);
            if (savedPos < 0n) {
              ret = savedPos;
              break;
            }
            this.vfs.lseek(pwFd, pwOffset, 0);
            const n = this.vfs.write(pwFd, pwBuf, pwCount, view);
            this.vfs.lseek(pwFd, savedPos, 0);
            ret = BigInt(n);
          } else {
            ret = -BigInt(EBADF);
          }
          break;
        }
        // ── Pipes & fd duplication ──
        case 59: {
          const pipefdAddr = Number(a0);
          const pipe = { data: [], writeClosed: false, readClosed: false };
          const readFd = this.allocFd();
          const writeFd = this.allocFd();
          this.fdTable.set(readFd, { type: "pipe_read", pipe });
          this.fdTable.set(writeFd, { type: "pipe_write", pipe });
          view.setInt32(pipefdAddr + GUEST_MEM_BASE, readFd, true);
          view.setInt32(pipefdAddr + GUEST_MEM_BASE + 4, writeFd, true);
          ret = 0n;
          break;
        }
        case 23: {
          const oldfd = Number(a0);
          const oldEntry = this.fdTable.get(oldfd);
          if (oldEntry) {
            const newfd = this.allocFd();
            if (oldEntry.type === "vfs" && this.vfs) {
              const vfsFd = this.vfs.dup(oldEntry.fd);
              if (vfsFd < 0) {
                ret = BigInt(vfsFd);
                break;
              }
              this.fdTable.set(newfd, { type: "vfs", fd: vfsFd });
            } else if (oldEntry.type === "native") {
              this.fdTable.set(newfd, { ...oldEntry });
              this.nativeRefs.set(oldEntry.nativeFd, (this.nativeRefs.get(oldEntry.nativeFd) ?? 1) + 1);
            } else {
              this.fdTable.set(newfd, { ...oldEntry });
            }
            ret = BigInt(newfd);
          } else if (oldfd <= 2) {
            const newfd = this.allocFd();
            this.fdTable.set(newfd, { type: "stdio", fd: oldfd });
            ret = BigInt(newfd);
          } else if (this.vfs) {
            const vfsFd = this.vfs.dup(oldfd);
            if (vfsFd < 0) {
              ret = BigInt(vfsFd);
              break;
            }
            const newfd = this.allocFd();
            this.fdTable.set(newfd, { type: "vfs", fd: vfsFd });
            ret = BigInt(newfd);
          } else {
            ret = -BigInt(EBADF);
          }
          break;
        }
        case 24: {
          const oldfd = Number(a0);
          const newfd = Number(BigInt.asIntN(32, a1));
          const dup3flags = Number(a2);
          if (oldfd === newfd) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (newfd < 0 || newfd >= MAX_FDS) {
            ret = -BigInt(EBADF);
            break;
          }
          const oldfdValid = oldfd >= 0 && (oldfd <= 2 || this.fdTable.has(oldfd) || this.vfs && this.vfs.hasFd(oldfd));
          if (!oldfdValid) {
            ret = -BigInt(EBADF);
            break;
          }
          const existingEntry = this.fdTable.get(newfd);
          if (existingEntry) {
            if (existingEntry.type === "pipe_write") existingEntry.pipe.writeClosed = true;
            if (existingEntry.type === "pipe_read") existingEntry.pipe.readClosed = true;
            if (existingEntry.type === "vfs" && this.vfs) this.vfs.close(existingEntry.fd);
            if (existingEntry.type === "native") {
              const refs = (this.nativeRefs.get(existingEntry.nativeFd) ?? 1) - 1;
              if (refs <= 0) this.nativeRefs.delete(existingEntry.nativeFd);
              else this.nativeRefs.set(existingEntry.nativeFd, refs);
            }
            this.fdTable.delete(newfd);
          }
          const srcEntry = this.fdTable.get(oldfd);
          if (srcEntry) {
            if (srcEntry.type === "vfs" && this.vfs) {
              const vfsFd = this.vfs.dup(srcEntry.fd);
              if (vfsFd < 0) {
                ret = BigInt(vfsFd);
                break;
              }
              this.fdTable.set(newfd, { type: "vfs", fd: vfsFd });
            } else if (srcEntry.type === "native") {
              this.fdTable.set(newfd, { ...srcEntry });
              this.nativeRefs.set(srcEntry.nativeFd, (this.nativeRefs.get(srcEntry.nativeFd) ?? 1) + 1);
            } else {
              this.fdTable.set(newfd, { ...srcEntry });
            }
            ret = BigInt(newfd);
          } else if (oldfd <= 2) {
            this.fdTable.set(newfd, { type: "stdio", fd: oldfd });
            ret = BigInt(newfd);
          } else if (this.vfs) {
            const vfsFd = this.vfs.dup(oldfd);
            if (vfsFd < 0) {
              ret = BigInt(vfsFd);
              break;
            }
            this.fdTable.set(newfd, { type: "vfs", fd: vfsFd });
            ret = BigInt(newfd);
          } else {
            ret = -BigInt(EBADF);
            break;
          }
          if (dup3flags & 524288) this.fdCloexec.add(newfd);
          else this.fdCloexec.delete(newfd);
          break;
        }
        // ── File operations ──
        case 56: {
          const path = readGuestString(bytes, Number(a1));
          const flags56 = Number(a2);
          const mode56 = Number(a3);
          const bp56 = this.bridgePath(path);
          if (bp56 !== null) {
            const guestFd = this.allocFd();
            return {
              status: "bridge",
              method: "fs_open",
              args: { path: bp56, flags: flags56, mode: mode56 },
              apply: (resp) => {
                if (resp.ret < 0) {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                  this.nextFd--;
                } else {
                  this.fdTable.set(guestFd, { type: "native", nativeFd: resp.ret, path: bp56 });
                  this.nativeRefs.set(resp.ret, 1);
                  cpu2.setReg(0, BigInt(guestFd));
                }
              }
            };
          }
          if (this.vfs) {
            const effectiveMode = mode56 & ~this.umask;
            const openRet = this.vfs.open(path, flags56, effectiveMode);
            if (openRet >= 0) {
              this.fdFlags.set(openRet, flags56 & ~(64 | 128 | 512 | 524288));
              if (flags56 & 524288) this.fdCloexec.add(openRet);
            }
            ret = BigInt(openRet);
          } else {
            ret = -BigInt(ENOENT);
          }
          break;
        }
        case 57: {
          const fd = Number(a0);
          const closeEntry = this.fdTable.get(fd);
          if (closeEntry) {
            if (closeEntry.type === "pipe_write") closeEntry.pipe.writeClosed = true;
            if (closeEntry.type === "pipe_read") closeEntry.pipe.readClosed = true;
            if (closeEntry.type === "vfs" && this.vfs) this.vfs.close(closeEntry.fd);
            if (closeEntry.type === "native") {
              this.fdTable.delete(fd);
              const nfd = closeEntry.nativeFd;
              const refs = (this.nativeRefs.get(nfd) ?? 1) - 1;
              if (refs <= 0) {
                this.nativeRefs.delete(nfd);
                return {
                  status: "bridge",
                  method: "fs_close",
                  args: { fd: nfd },
                  apply: (resp) => {
                    cpu2.setReg(0, BigInt(resp.ret ?? 0));
                  }
                };
              }
              this.nativeRefs.set(nfd, refs);
              ret = 0n;
              break;
            }
            if (closeEntry.type === "socket") {
              this.fdTable.delete(fd);
              return {
                status: "bridge",
                method: "net_close",
                args: { socketId: closeEntry.socketId },
                apply: (resp) => {
                  cpu2.setReg(0, 0n);
                }
              };
            }
            this.fdTable.delete(fd);
            ret = 0n;
          } else if (fd <= 2) {
            ret = 0n;
          } else if (this.vfs) {
            ret = BigInt(this.vfs.close(fd));
          } else {
            ret = -BigInt(EBADF);
          }
          break;
        }
        case 46: {
          const ftFd = Number(a0);
          const ftLen = Number(a1);
          const ftEntry = this.getFdEntry(ftFd);
          if (ftEntry && ftEntry.type === "vfs" && this.vfs) {
            ret = BigInt(this.vfs.ftruncate(ftEntry.fd, ftLen));
          } else if (ftEntry && ftEntry.type === "native") {
            return {
              status: "bridge",
              method: "fs_ftruncate",
              args: { fd: ftEntry.nativeFd, length: ftLen },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          } else if (this.vfs) {
            ret = BigInt(this.vfs.ftruncate(ftFd, ftLen));
          } else {
            ret = -BigInt(EBADF);
          }
          break;
        }
        case 62: {
          const lseekFd = Number(a0);
          const lseekEntry = this.getFdEntry(lseekFd);
          if (lseekFd <= 2 && !lseekEntry) {
            ret = -BigInt(ESPIPE);
            break;
          }
          if (lseekEntry) {
            if (lseekEntry.type === "stdio" || lseekEntry.type === "pipe_read" || lseekEntry.type === "pipe_write") {
              ret = -BigInt(ESPIPE);
              break;
            }
            if (lseekEntry.type === "native") {
              return {
                status: "bridge",
                method: "fs_lseek",
                args: { fd: lseekEntry.nativeFd, offset: Number(a1), whence: Number(a2) },
                apply: (resp) => {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                }
              };
            }
            if (lseekEntry.type === "vfs") {
              ret = this.vfs ? this.vfs.lseek(lseekEntry.fd, BigInt.asIntN(64, a1), Number(a2)) : -BigInt(EBADF);
              break;
            }
          }
          if (this.vfs) {
            ret = this.vfs.lseek(lseekFd, BigInt.asIntN(64, a1), Number(a2));
          } else {
            ret = -BigInt(EBADF);
          }
          break;
        }
        // ── stat/fstat ──
        case 80: {
          const fd = Number(a0);
          const statBuf = Number(a1);
          const fstatEntry = this.getFdEntry(fd);
          if (fstatEntry) {
            if (fstatEntry.type === "stdio" || fstatEntry.type === "pipe_read" || fstatEntry.type === "pipe_write") {
              writeChardevStat(view, statBuf);
              ret = 0n;
            } else if (fstatEntry.type === "native") {
              return {
                status: "bridge",
                method: "fs_stat",
                args: { path: fstatEntry.path },
                apply: (resp) => {
                  if (resp.ret < 0) {
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                  } else {
                    writeBridgeStat(view, statBuf, resp);
                    cpu2.setReg(0, 0n);
                  }
                }
              };
            } else if (fstatEntry.type === "vfs" && this.vfs) {
              ret = BigInt(this.vfs.fstat(fstatEntry.fd, statBuf, view));
            } else {
              ret = -BigInt(EBADF);
            }
          } else if (fd <= 2) {
            writeChardevStat(view, statBuf);
            ret = 0n;
          } else if (this.vfs) {
            ret = BigInt(this.vfs.fstat(fd, statBuf, view));
          } else {
            ret = -BigInt(EBADF);
          }
          break;
        }
        case 79: {
          const path = readGuestString(bytes, Number(a1));
          const statBuf79 = Number(a2);
          const bp79 = this.bridgePath(path);
          if (bp79 !== null) {
            return {
              status: "bridge",
              method: "fs_stat",
              args: { path: bp79 },
              apply: (resp) => {
                if (resp.ret < 0) {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                } else {
                  writeBridgeStat(view, statBuf79, resp);
                  cpu2.setReg(0, 0n);
                }
              }
            };
          }
          if (this.vfs) {
            ret = BigInt(this.vfs.fstatat(Number(a0), path, statBuf79, Number(a3), view));
          } else {
            ret = -BigInt(ENOENT);
          }
          break;
        }
        case 61: {
          const fd = Number(a0);
          const dirp = Number(a1);
          const count = Number(a2);
          const gdEntry = this.getFdEntry(fd);
          if (gdEntry?.type === "native") {
            return {
              status: "bridge",
              method: "fs_getdents",
              args: { fd: gdEntry.nativeFd },
              apply: (resp) => {
                if (resp.ret <= 0) {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                  return;
                }
                const entries = resp.entries ?? [];
                let offset = 0;
                for (const entry of entries) {
                  const nameBytes = new TextEncoder().encode(entry.name);
                  const reclen = 19 + nameBytes.length + 1 + 7 & ~7;
                  if (offset + reclen > count) break;
                  const base = dirp + offset + GUEST_MEM_BASE;
                  view.setBigUint64(base, 1n, true);
                  view.setBigInt64(base + 8, 0n, true);
                  view.setUint16(base + 16, reclen, true);
                  bytes[base + 18] = entry.type;
                  bytes.set(nameBytes, base + 19);
                  bytes[base + 19 + nameBytes.length] = 0;
                  offset += reclen;
                }
                cpu2.setReg(0, BigInt(offset));
              }
            };
          }
          if (this.vfs) {
            ret = BigInt(this.vfs.getdents64(fd, dirp, count, view));
          } else {
            ret = -BigInt(EBADF);
          }
          break;
        }
        case 34: {
          const path = readGuestString(bytes, Number(a1));
          const bp34 = this.bridgePath(path);
          if (bp34 !== null) {
            return {
              status: "bridge",
              method: "fs_mkdir",
              args: { path: bp34, mode: Number(a2) },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          if (this.vfs) {
            ret = BigInt(this.vfs.mkdirat(path, Number(a2) & ~this.umask));
          } else {
            ret = -BigInt(ENOENT);
          }
          break;
        }
        case 35: {
          const path = readGuestString(bytes, Number(a1));
          const bp35 = this.bridgePath(path);
          if (bp35 !== null) {
            return {
              status: "bridge",
              method: "fs_unlink",
              args: { path: bp35 },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          if (this.vfs) {
            ret = BigInt(this.vfs.unlinkat(path, Number(a2)));
          } else {
            ret = -BigInt(ENOENT);
          }
          break;
        }
        case 33: {
          const path = readGuestString(bytes, Number(a1));
          const mode = Number(a2);
          if (this.vfs) {
            const fmt = mode & 61440;
            if (fmt === 4096) {
              ret = BigInt(this.vfs.open(path, 64 | 512, mode & 4095));
              if (ret >= 0n) {
                this.vfs.close(Number(ret));
                ret = 0n;
              }
            } else if (fmt === 32768 || fmt === 0) {
              ret = BigInt(this.vfs.open(path, 64, mode & 4095));
              if (ret >= 0n) {
                this.vfs.close(Number(ret));
                ret = 0n;
              }
            } else {
              ret = -BigInt(EPERM);
            }
          } else {
            ret = -BigInt(ENOSYS);
          }
          break;
        }
        case 38: {
          const oldpath = readGuestString(bytes, Number(a1));
          const newpath = readGuestString(bytes, Number(a3));
          if (this.vfs) {
            ret = BigInt(this.vfs.rename(oldpath, newpath));
          } else {
            ret = -BigInt(ENOSYS);
          }
          break;
        }
        case 36: {
          const target = readGuestString(bytes, Number(a0));
          const linkpath = readGuestString(bytes, Number(a2));
          if (this.vfs) {
            ret = BigInt(this.vfs.symlinkat(target, linkpath));
          } else {
            ret = -BigInt(ENOSYS);
          }
          break;
        }
        case 37: {
          const oldpath = readGuestString(bytes, Number(a1));
          const newpath = readGuestString(bytes, Number(a3));
          if (this.vfs) {
            ret = BigInt(this.vfs.linkat(oldpath, newpath));
          } else {
            ret = -BigInt(ENOSYS);
          }
          break;
        }
        case 88: {
          ret = 0n;
          break;
        }
        case 78: {
          const path = readGuestString(bytes, Number(a1));
          const buf = Number(a2);
          const bufsiz = Number(a3);
          if (this.vfs) {
            const target = this.vfs.readlinkat(path);
            if (target === null) {
              ret = -BigInt(EINVAL);
            } else {
              const encoded = new TextEncoder().encode(target);
              const len = Math.min(encoded.length, bufsiz);
              bytes.set(encoded.subarray(0, len), buf + GUEST_MEM_BASE);
              ret = BigInt(len);
            }
          } else {
            ret = -BigInt(ENOENT);
          }
          break;
        }
        case 48: {
          const path = readGuestString(bytes, Number(a1));
          const bp48 = this.bridgePath(path);
          if (bp48 !== null) {
            return {
              status: "bridge",
              method: "fs_stat",
              args: { path: bp48 },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret < 0 ? resp.ret : 0)));
              }
            };
          }
          if (this.vfs) {
            ret = BigInt(this.vfs.faccessat(Number(a0), path, Number(a2), Number(a3)));
          } else {
            ret = -BigInt(ENOENT);
          }
          break;
        }
        // ── chmod/chown (stubs — we run as root, permissions are a no-op) ──
        case 52: {
          const fchmodFd = Number(a0);
          const fchmodMode = Number(a1);
          if (this.vfs && this.vfs.hasFd(fchmodFd)) {
            this.vfs.fchmod(fchmodFd, fchmodMode);
          }
          ret = 0n;
          break;
        }
        case 53:
        // (unused on aarch64, but keep for safety)
        case 54: {
          ret = 0n;
          break;
        }
        case 55: {
          const fchownFlags = Number(a4);
          if (fchownFlags & 4096) {
            const fchownFd = Number(a0);
            if (this.vfs && this.vfs.hasFd(fchownFd)) {
              this.vfs.fchown(fchownFd, Number(BigInt.asIntN(32, a2)), Number(BigInt.asIntN(32, a3)));
            }
          }
          ret = 0n;
          break;
        }
        // ── statfs/fstatfs — return a plausible tmpfs-like result ──
        case 43:
        // statfs(path, buf)
        case 44: {
          const statfsBuf = Number(nr === 43 ? a1 : a1);
          const sv = new DataView(memory2.buffer);
          const base = statfsBuf + GUEST_MEM_BASE;
          sv.setBigUint64(base + 0, 0x01021994n, true);
          sv.setBigUint64(base + 8, 4096n, true);
          sv.setBigUint64(base + 16, 1048576n, true);
          sv.setBigUint64(base + 24, 524288n, true);
          sv.setBigUint64(base + 32, 524288n, true);
          sv.setBigUint64(base + 40, 1048576n, true);
          sv.setBigUint64(base + 48, 524288n, true);
          sv.setBigUint64(base + 56, 0n, true);
          sv.setBigUint64(base + 64, 255n, true);
          sv.setBigUint64(base + 72, 4096n, true);
          sv.setBigUint64(base + 80, 0n, true);
          ret = 0n;
          break;
        }
        // ── ioctl ──
        case 29: {
          const fd = Number(a0);
          const request = Number(a1);
          const ioctlEntry = this.getFdEntry(fd);
          const isStdio = fd <= 2 || ioctlEntry?.type === "stdio";
          const isPipe = ioctlEntry?.type === "pipe_read" || ioctlEntry?.type === "pipe_write";
          if (isPipe) {
            ret = -BigInt(EINVAL);
            break;
          }
          switch (request) {
            case TCGETS: {
              const buf = Number(a2);
              view.setUint32(buf + GUEST_MEM_BASE, 0, true);
              view.setUint32(buf + GUEST_MEM_BASE + 4, 0, true);
              view.setUint32(buf + GUEST_MEM_BASE + 8, 0, true);
              view.setUint32(buf + GUEST_MEM_BASE + 12, 0, true);
              ret = 0n;
              break;
            }
            case TCSETS: {
              ret = 0n;
              break;
            }
            case TIOCGWINSZ: {
              const buf = Number(a2);
              view.setUint16(buf + GUEST_MEM_BASE, this.termRows, true);
              view.setUint16(buf + GUEST_MEM_BASE + 2, this.termCols, true);
              view.setUint16(buf + GUEST_MEM_BASE + 4, 0, true);
              view.setUint16(buf + GUEST_MEM_BASE + 6, 0, true);
              ret = 0n;
              break;
            }
            case TIOCSWINSZ: {
              ret = 0n;
              break;
            }
            default:
              ret = -BigInt(EINVAL);
          }
          break;
        }
        // ── Memory management ──
        case 214: {
          const addr = a0;
          if (addr === 0n) {
            ret = this.brk;
          } else if (addr >= this.brk) {
            const needed = Number(addr) + GUEST_MEM_BASE;
            if (needed > memory2.buffer.byteLength) {
              const pagesNeeded = Math.ceil(needed / 65536);
              const currentPages = memory2.buffer.byteLength / 65536;
              if (pagesNeeded > currentPages) {
                memory2.grow(pagesNeeded - currentPages);
              }
            }
            this.brk = addr;
            ret = this.brk;
          } else {
            ret = this.brk;
          }
          break;
        }
        case 222: {
          const length = Number(a1);
          const prot = Number(a2);
          const flags = Number(a3);
          const mmapFd = Number(BigInt.asIntN(32, a4));
          const aligned = length + 4095 & ~4095;
          const MAP_ANONYMOUS = 32;
          const MAP_FIXED = 16;
          const MAP_SHARED = 1;
          let addr;
          if (a0 !== 0n) {
            addr = a0;
            const end = a0 + BigInt(aligned);
            if (end > this.mmapBase) this.mmapBase = end;
          } else {
            addr = this.mmapBase;
            this.mmapBase += BigInt(aligned);
          }
          const needed = Number(addr) + aligned + GUEST_MEM_BASE;
          if (needed > memory2.buffer.byteLength) {
            const pagesNeeded = Math.ceil(needed / 65536);
            const currentPages = memory2.buffer.byteLength / 65536;
            if (pagesNeeded > currentPages) memory2.grow(pagesNeeded - currentPages);
          }
          new Uint8Array(memory2.buffer).fill(0, Number(addr) + GUEST_MEM_BASE, Number(addr) + GUEST_MEM_BASE + aligned);
          if (mmapFd >= 0 && !(flags & MAP_ANONYMOUS) && this.vfs) {
            const fileOffset = Number(a5);
            const savedPos = this.vfs.lseek(mmapFd, 0n, 1);
            if (savedPos >= 0n) {
              this.vfs.lseek(mmapFd, BigInt(fileOffset), 0);
              this.vfs.read(mmapFd, Number(addr), aligned, new DataView(memory2.buffer));
              this.vfs.lseek(mmapFd, savedPos, 0);
            }
          }
          ret = addr;
          break;
        }
        case 226:
          ret = 0n;
          break;
        case 215:
          ret = 0n;
          break;
        // ── Thread/process ──
        case 96: {
          ret = 1n;
          break;
        }
        case 99:
          ret = 0n;
          break;
        // gettid handled with getpid above
        case 172:
          ret = BigInt(this.currentPid);
          break;
        case 173:
          ret = BigInt(this.currentPpid);
          break;
        case 178:
          ret = BigInt(this.currentPid);
          break;
        case 174:
        // getuid
        case 175:
        // geteuid
        case 176:
        // getgid
        case 177:
          ret = 0n;
          break;
        // ── Signals ──
        case 134: {
          const signum = Number(a0);
          const actPtr = Number(a1);
          const oldactPtr = Number(a2);
          if (!this.signalHandlers) this.signalHandlers = /* @__PURE__ */ new Map();
          if (oldactPtr !== 0) {
            const base = oldactPtr + GUEST_MEM_BASE;
            const old = this.signalHandlers.get(signum);
            if (old) {
              view.setBigUint64(base, old.handler, true);
              view.setBigUint64(base + 8, old.flags, true);
              view.setBigUint64(base + 16, old.restorer, true);
              view.setBigUint64(base + 24, old.mask, true);
            } else {
              for (let i = 0; i < 32; i += 8) view.setBigUint64(base + i, 0n, true);
            }
          }
          if (actPtr !== 0) {
            const base = actPtr + GUEST_MEM_BASE;
            this.signalHandlers.set(signum, {
              handler: view.getBigUint64(base, true),
              flags: view.getBigUint64(base + 8, true),
              restorer: view.getBigUint64(base + 16, true),
              mask: view.getBigUint64(base + 24, true)
            });
          }
          ret = 0n;
          break;
        }
        case 135: {
          const oldsetPtr = Number(a2);
          if (oldsetPtr !== 0) {
            view.setBigUint64(oldsetPtr + GUEST_MEM_BASE, this.signalMask ?? 0n, true);
          }
          const setPtr = Number(a1);
          if (setPtr !== 0) {
            const how = Number(a0);
            const newSet = view.getBigUint64(setPtr + GUEST_MEM_BASE, true);
            const current = this.signalMask ?? 0n;
            if (how === 0) this.signalMask = current | newSet;
            else if (how === 1) this.signalMask = current & ~newSet;
            else if (how === 2) this.signalMask = newSet;
          }
          ret = 0n;
          break;
        }
        case 132: {
          const oldssPtr = Number(a1);
          if (oldssPtr !== 0) {
            const base = oldssPtr + GUEST_MEM_BASE;
            view.setBigUint64(base, 0n, true);
            view.setInt32(base + 8, 2, true);
            view.setBigUint64(base + 16, 0n, true);
          }
          ret = 0n;
          break;
        }
        case 133: {
          ret = -BigInt(4);
          break;
        }
        case 136: {
          const pendPtr = Number(a0) + GUEST_MEM_BASE;
          view.setBigUint64(pendPtr, 0n, true);
          ret = 0n;
          break;
        }
        // ── Filesystem info ──
        case 17: {
          const buf = Number(a0);
          const size = Number(a1);
          const cwd = this.vfs ? this.vfs.getcwd() : this.cwd;
          const encoded = new TextEncoder().encode(cwd + "\0");
          if (encoded.length > size) {
            ret = -BigInt(34);
          } else {
            const dest = buf + GUEST_MEM_BASE;
            if (dest + encoded.length <= memory2.buffer.byteLength) {
              new Uint8Array(memory2.buffer).set(encoded, dest);
            }
            ret = a0;
          }
          break;
        }
        case 49: {
          const path = readGuestString(bytes, Number(a0));
          if (this.vfs) {
            const res = this.vfs.chdir(path);
            ret = BigInt(res);
          } else {
            this.cwd = path;
            ret = 0n;
          }
          break;
        }
        // ── uname ──
        case 160: {
          const buf = Number(a0) + GUEST_MEM_BASE;
          const fields = ["Linux", "jitter", "6.1.0", "#1", "aarch64", "(none)"];
          for (let i = 0; i < fields.length; i++) {
            const encoded = new TextEncoder().encode(fields[i] + "\0");
            new Uint8Array(memory2.buffer).set(encoded, buf + i * 65);
          }
          ret = 0n;
          break;
        }
        // ── Time ──
        case 113: {
          const tp = Number(a1) + GUEST_MEM_BASE;
          const now = Date.now();
          const sec = BigInt(Math.floor(now / 1e3));
          const nsec = BigInt(now % 1e3 * 1e6);
          view.setBigUint64(tp, sec, true);
          view.setBigUint64(tp + 8, nsec, true);
          ret = 0n;
          break;
        }
        case 169: {
          const tv = Number(a0) + GUEST_MEM_BASE;
          const now = Date.now();
          const sec = BigInt(Math.floor(now / 1e3));
          const usec = BigInt(now % 1e3 * 1e3);
          view.setBigUint64(tv, sec, true);
          view.setBigUint64(tv + 8, usec, true);
          ret = 0n;
          break;
        }
        // ── fcntl ──
        case 25: {
          const fd = Number(a0);
          const cmd = Number(a1);
          switch (cmd) {
            case F_DUPFD:
            case F_DUPFD_CLOEXEC: {
              const minfd = Number(a2);
              let newfd = minfd;
              while (this.fdTable.has(newfd)) newfd++;
              const srcEntry = this.fdTable.get(fd);
              if (srcEntry) {
                if (srcEntry.type === "vfs" && this.vfs) {
                  const vfsFd = this.vfs.dup(srcEntry.fd);
                  if (vfsFd < 0) {
                    ret = BigInt(vfsFd);
                    break;
                  }
                  this.fdTable.set(newfd, { type: "vfs", fd: vfsFd });
                } else if (srcEntry.type === "native") {
                  this.fdTable.set(newfd, { ...srcEntry });
                  this.nativeRefs.set(srcEntry.nativeFd, (this.nativeRefs.get(srcEntry.nativeFd) ?? 1) + 1);
                } else {
                  this.fdTable.set(newfd, { ...srcEntry });
                }
              } else if (fd <= 2) {
                this.fdTable.set(newfd, { type: "stdio", fd });
              } else if (this.vfs) {
                const vfsFd = this.vfs.dup(fd);
                if (vfsFd < 0) {
                  ret = BigInt(vfsFd);
                  break;
                }
                this.fdTable.set(newfd, { type: "vfs", fd: vfsFd });
              } else {
                ret = -BigInt(EBADF);
                break;
              }
              if (newfd >= this.nextFd) this.nextFd = newfd + 1;
              ret = BigInt(newfd);
              break;
            }
            case F_GETFL: {
              const flags = this.fdFlags.get(fd) ?? 0;
              ret = BigInt(flags);
              break;
            }
            case F_SETFL: {
              const modifiable = 1024 | 2048;
              const current = this.fdFlags.get(fd) ?? 0;
              const newFlags = current & ~modifiable | Number(a2) & modifiable;
              this.fdFlags.set(fd, newFlags);
              ret = 0n;
              break;
            }
            case F_GETFD:
              ret = BigInt(this.fdCloexec.has(fd) ? 1 : 0);
              break;
            case F_SETFD:
              if (Number(a2) & 1) this.fdCloexec.add(fd);
              else this.fdCloexec.delete(fd);
              ret = 0n;
              break;
            case 5:
            // F_GETLK — query lock status
            case 12: {
              const flockPtr = Number(a2) + GUEST_MEM_BASE;
              view.setInt16(flockPtr, 2, true);
              ret = 0n;
              break;
            }
            case 6:
            // F_SETLK — set lock (non-blocking)
            case 7:
            // F_SETLKW — set lock (blocking)
            case 13:
            // F_SETLK64
            case 14:
              ret = 0n;
              break;
            case 8:
              ret = 0n;
              break;
            case 9:
              ret = BigInt(this.currentPid);
              break;
            case 10:
              ret = BigInt(this.fdSig.get(fd) ?? 0);
              break;
            case 11:
              this.fdSig.set(fd, Number(a2));
              ret = 0n;
              break;
            case 15:
              ret = 0n;
              break;
            case 16: {
              const ownerPtr = Number(a2) + GUEST_MEM_BASE;
              view.setInt32(ownerPtr, 1, true);
              view.setInt32(ownerPtr + 4, this.currentPid, true);
              ret = 0n;
              break;
            }
            case 1024:
              ret = 0n;
              break;
            case 1025:
              ret = 0n;
              break;
            case 1026:
              ret = 0n;
              break;
            case 1032:
              ret = 1048576n;
              break;
            case 1031: {
              const requestedSize = Number(a2);
              ret = BigInt(Math.max(requestedSize, 4096));
              break;
            }
            default:
              ret = -BigInt(EINVAL);
          }
          break;
        }
        // ── Misc ──
        case 278: {
          const buf = Number(a0) + GUEST_MEM_BASE;
          const len = Number(a1);
          for (let i = 0; i < len; i++) {
            bytes[buf + i] = Math.random() * 256 | 0;
          }
          ret = a1;
          break;
        }
        case 98: {
          const futexAddr = Number(a0) + GUEST_MEM_BASE;
          const futexOp = Number(a1) & 127;
          const futexVal = Number(a2);
          switch (futexOp) {
            case 0: {
              const current = view.getInt32(futexAddr, true);
              if (current !== futexVal) {
                ret = -BigInt(11);
              } else {
                ret = -BigInt(11);
              }
              break;
            }
            case 1:
              ret = 0n;
              break;
            case 2:
              ret = -BigInt(ENOSYS);
              break;
            case 3:
              ret = 0n;
              break;
            case 4:
              ret = 0n;
              break;
            case 5:
              ret = 0n;
              break;
            case 6:
              ret = 0n;
              break;
            case 7:
              ret = 0n;
              break;
            case 8:
              ret = 0n;
              break;
            case 9:
              ret = -BigInt(11);
              break;
            case 10:
              ret = 0n;
              break;
            default:
              ret = -BigInt(ENOSYS);
          }
          break;
        }
        case 131:
          ret = 0n;
          break;
        case 143:
        // setregid
        case 144:
        // setgid
        case 145:
        // setreuid
        case 146:
        // setuid
        case 147:
        // setresuid
        case 149:
        // setresgid
        case 151:
        // setfsuid
        case 152:
        // setfsgid
        case 159:
          ret = 0n;
          break;
        // ── Networking (via Swift bridge) ──
        case 198: {
          const domain = Number(a0);
          const sockType = Number(a1) & 255;
          const protocol = Number(a2);
          console.log(`[net] socket(domain=${domain}, type=${sockType})`);
          if (domain === 1) {
            ret = -BigInt(ENOSYS);
            break;
          }
          if (!this.bridge?.isNative && !this.bridgeAvailable) {
            ret = -BigInt(ENOSYS);
            break;
          }
          const socketFd = this.allocFd();
          return {
            status: "bridge",
            method: "net_socket",
            args: { domain, type: sockType, protocol },
            apply: (resp) => {
              if (resp.ret < 0) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                this.nextFd--;
              } else {
                this.fdTable.set(socketFd, { type: "socket", socketId: resp.ret, domain, sockType });
                if (Number(a1) & 524288) this.fdCloexec.add(socketFd);
                cpu2.setReg(0, BigInt(socketFd));
              }
            }
          };
        }
        case 200: {
          const bindEntry = this.getFdEntry(Number(a0));
          if (!bindEntry || bindEntry.type !== "socket") {
            ret = -BigInt(EBADF);
            break;
          }
          const addrBuf = this.parseSockaddr(bytes, Number(a1), Number(a2));
          return {
            status: "bridge",
            method: "net_bind",
            args: { socketId: bindEntry.socketId, ...addrBuf },
            apply: (resp) => {
              cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            }
          };
        }
        case 201: {
          const listenEntry = this.getFdEntry(Number(a0));
          if (!listenEntry || listenEntry.type !== "socket") {
            ret = -BigInt(EBADF);
            break;
          }
          return {
            status: "bridge",
            method: "net_listen",
            args: { socketId: listenEntry.socketId, backlog: Number(a1) },
            apply: (resp) => {
              cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            }
          };
        }
        case 202: {
          const acceptEntry = this.getFdEntry(Number(a0));
          if (!acceptEntry || acceptEntry.type !== "socket") {
            ret = -BigInt(EBADF);
            break;
          }
          const newFd = this.allocFd();
          return {
            status: "bridge",
            method: "net_accept",
            args: { socketId: acceptEntry.socketId },
            apply: (resp) => {
              if (resp.ret < 0) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                this.nextFd--;
              } else {
                this.fdTable.set(newFd, { type: "socket", socketId: resp.ret, domain: acceptEntry.domain, sockType: acceptEntry.sockType });
                cpu2.setReg(0, BigInt(newFd));
              }
            }
          };
        }
        case 203: {
          const connEntry = this.getFdEntry(Number(a0));
          if (!connEntry || connEntry.type !== "socket") {
            ret = -BigInt(EBADF);
            break;
          }
          const connAddr = this.parseSockaddr(bytes, Number(a1), Number(a2));
          console.log(`[net] connect(${connAddr.host}:${connAddr.port})`);
          connEntry.remoteAddr = connAddr;
          return {
            status: "bridge",
            method: "net_connect",
            args: { socketId: connEntry.socketId, ...connAddr },
            apply: (resp) => {
              console.log(`[net] connect \u2192 ${resp.ret}`);
              cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            }
          };
        }
        case 204: {
          const gsnEntry = this.getFdEntry(Number(a0));
          if (!gsnEntry || gsnEntry.type !== "socket") {
            ret = -BigInt(EBADF);
            break;
          }
          const addrPtr = Number(a1) + GUEST_MEM_BASE;
          const lenPtr = Number(a2) + GUEST_MEM_BASE;
          const family = gsnEntry.domain;
          view.setUint16(addrPtr, family, true);
          if (family === 2) {
            view.setUint32(lenPtr, 16, true);
          } else if (family === 10) {
            view.setUint32(lenPtr, 28, true);
          }
          ret = 0n;
          break;
        }
        case 205: {
          const gpnEntry = this.getFdEntry(Number(a0));
          if (!gpnEntry || gpnEntry.type !== "socket") {
            ret = -BigInt(EBADF);
            break;
          }
          ret = -107n;
          break;
        }
        case 206: {
          const stEntry = this.getFdEntry(Number(a0));
          if (!stEntry || stEntry.type !== "socket") {
            ret = -BigInt(EBADF);
            break;
          }
          const sendBuf = Number(a1);
          const sendLen = Number(a2);
          const sendData = bytes.slice(sendBuf + GUEST_MEM_BASE, sendBuf + GUEST_MEM_BASE + sendLen);
          const b64 = btoa(String.fromCharCode(...sendData));
          const destAddrPtr = Number(a4);
          const destAddrLen = Number(a5);
          if (destAddrPtr !== 0 && destAddrLen > 0) {
            const dest = this.parseSockaddr(bytes, destAddrPtr, destAddrLen);
            if (!stEntry.remoteAddr) stEntry.remoteAddr = dest;
            console.log(`[net] sendto(${dest.host}:${dest.port}, ${sendLen}B)`);
            return {
              status: "bridge",
              method: "net_sendto",
              args: { socketId: stEntry.socketId, data: b64, host: dest.host, port: dest.port },
              apply: (resp) => {
                console.log(`[net] sendto \u2192 ${resp.ret}`);
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          return {
            status: "bridge",
            method: "net_write",
            args: { socketId: stEntry.socketId, data: b64 },
            apply: (resp) => {
              cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            }
          };
        }
        case 207: {
          const rfEntry = this.getFdEntry(Number(a0));
          if (!rfEntry || rfEntry.type !== "socket") {
            ret = -BigInt(EBADF);
            break;
          }
          const recvBuf = Number(a1);
          const recvLen = Number(a2);
          return {
            status: "bridge",
            method: "net_read",
            args: { socketId: rfEntry.socketId, count: recvLen },
            apply: (resp) => {
              if (resp.ret < 0) {
                console.log(`[net] recvfrom \u2192 err ${resp.ret}`);
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              } else if (resp.data) {
                const decoded = Uint8Array.from(atob(resp.data), (c) => c.charCodeAt(0));
                bytes.set(decoded, recvBuf + GUEST_MEM_BASE);
                console.log(`[net] recvfrom \u2192 ${decoded.length}B`);
                cpu2.setReg(0, BigInt(decoded.length));
              } else {
                console.log(`[net] recvfrom \u2192 ${resp.ret} (no data)`);
                cpu2.setReg(0, BigInt(resp.ret));
              }
            }
          };
        }
        case 208:
          ret = 0n;
          break;
        case 209: {
          const optval = Number(a3) + GUEST_MEM_BASE;
          const optlen = Number(a4) + GUEST_MEM_BASE;
          const optname = Number(a2);
          view.setInt32(optval, 0, true);
          view.setUint32(optlen, 4, true);
          ret = 0n;
          break;
        }
        // ── Process identity/groups ──
        case 129: {
          ret = 0n;
          break;
        }
        case 166: {
          const oldMask = this.umask;
          this.umask = Number(a0) & 511;
          ret = BigInt(oldMask);
          break;
        }
        case 154:
          ret = 0n;
          break;
        case 155:
          ret = BigInt(1);
          break;
        case 156:
          ret = BigInt(this.currentPid);
          break;
        case 157:
          ret = BigInt(this.currentPid);
          break;
        case 158: {
          const gSize = Number(a0);
          if (gSize === 0) {
            ret = 1n;
            break;
          }
          if (gSize >= 1) {
            view.setUint32(Number(a1) + GUEST_MEM_BASE, 0, true);
          }
          ret = 1n;
          break;
        }
        case 122:
        // sched_setaffinity — stub
        case 123:
          ret = 0n;
          break;
        case 115: {
          ret = 0n;
          break;
        }
        case 261: {
          const rlOldPtr = Number(a3);
          const RLIM_INFINITY = 0xFFFFFFFFFFFFFFFFn;
          if (rlOldPtr !== 0) {
            const base = rlOldPtr + GUEST_MEM_BASE;
            const resource = Number(a1);
            switch (resource) {
              case 7:
                view.setBigUint64(base, 1024n, true);
                view.setBigUint64(base + 8, 1024n, true);
                break;
              case 3:
                view.setBigUint64(base, 8388608n, true);
                view.setBigUint64(base + 8, RLIM_INFINITY, true);
                break;
              default:
                view.setBigUint64(base, RLIM_INFINITY, true);
                view.setBigUint64(base + 8, RLIM_INFINITY, true);
            }
          }
          ret = 0n;
          break;
        }
        case 233:
          ret = 0n;
          break;
        // ── Missing syscalls (spec-compliant implementations) ──
        case 50: {
          ret = -BigInt(EBADF);
          break;
        }
        case 101: {
          const remAddr = Number(a1);
          if (remAddr !== 0) {
            view.setBigUint64(remAddr + GUEST_MEM_BASE, 0n, true);
            view.setBigUint64(remAddr + GUEST_MEM_BASE + 8, 0n, true);
          }
          ret = 0n;
          break;
        }
        case 124:
          ret = 0n;
          break;
        case 165: {
          const usageBuf = Number(a1) + GUEST_MEM_BASE;
          for (let i = 0; i < 144; i += 8) view.setBigUint64(usageBuf + i, 0n, true);
          ret = 0n;
          break;
        }
        case 167: {
          const prOption = Number(a0);
          switch (prOption) {
            case 15:
              ret = 0n;
              break;
            case 16:
              const nameBuf = Number(a1) + GUEST_MEM_BASE;
              const name = new TextEncoder().encode("jitter\0");
              new Uint8Array(memory2.buffer).set(name, nameBuf);
              ret = 0n;
              break;
            case 38:
              ret = 0n;
              break;
            default:
              ret = 0n;
          }
          break;
        }
        case 179: {
          const siBuf = Number(a0) + GUEST_MEM_BASE;
          for (let i = 0; i < 112; i += 8) view.setBigUint64(siBuf + i, 0n, true);
          const uptime = Math.floor(Date.now() / 1e3) - 17e8;
          view.setBigUint64(siBuf + 0, BigInt(Math.max(uptime, 60)), true);
          view.setBigUint64(siBuf + 32, BigInt(1024 * 1024 * 1024), true);
          view.setBigUint64(siBuf + 40, BigInt(512 * 1024 * 1024), true);
          view.setBigUint64(siBuf + 48, BigInt(512 * 1024 * 1024), true);
          view.setBigUint64(siBuf + 56, BigInt(64 * 1024 * 1024), true);
          view.setBigUint64(siBuf + 80, 1n, true);
          view.setUint32(siBuf + 104, 1, true);
          ret = 0n;
          break;
        }
        case 148: {
          const ruidPtr = Number(a0) + GUEST_MEM_BASE;
          const euidPtr = Number(a1) + GUEST_MEM_BASE;
          const suidPtr = Number(a2) + GUEST_MEM_BASE;
          view.setUint32(ruidPtr, 0, true);
          view.setUint32(euidPtr, 0, true);
          view.setUint32(suidPtr, 0, true);
          ret = 0n;
          break;
        }
        case 150: {
          const rgidPtr = Number(a0) + GUEST_MEM_BASE;
          const egidPtr = Number(a1) + GUEST_MEM_BASE;
          const sgidPtr = Number(a2) + GUEST_MEM_BASE;
          view.setUint32(rgidPtr, 0, true);
          view.setUint32(egidPtr, 0, true);
          view.setUint32(sgidPtr, 0, true);
          ret = 0n;
          break;
        }
        case 153: {
          const tmsBuf = Number(a0) + GUEST_MEM_BASE;
          view.setBigUint64(tmsBuf, 0n, true);
          view.setBigUint64(tmsBuf + 8, 0n, true);
          view.setBigUint64(tmsBuf + 16, 0n, true);
          view.setBigUint64(tmsBuf + 24, 0n, true);
          ret = BigInt(Math.floor(Date.now() / 10));
          break;
        }
        case 92:
          ret = 0n;
          break;
        // ── Timer syscalls ──
        case 102: {
          const itvPtr = Number(a1) + GUEST_MEM_BASE;
          for (let i = 0; i < 32; i += 8) view.setBigUint64(itvPtr + i, 0n, true);
          ret = 0n;
          break;
        }
        case 103: {
          const oldPtr = Number(a2);
          if (oldPtr !== 0) {
            const base = oldPtr + GUEST_MEM_BASE;
            for (let i = 0; i < 32; i += 8) view.setBigUint64(base + i, 0n, true);
          }
          ret = 0n;
          break;
        }
        case 107: {
          const tidPtr = Number(a2) + GUEST_MEM_BASE;
          view.setInt32(tidPtr, 0, true);
          ret = 0n;
          break;
        }
        case 108: {
          const tsPtr = Number(a1) + GUEST_MEM_BASE;
          for (let i = 0; i < 32; i += 8) view.setBigUint64(tsPtr + i, 0n, true);
          ret = 0n;
          break;
        }
        case 109:
          ret = 0n;
          break;
        case 110: {
          const oldValPtr = Number(a3);
          if (oldValPtr !== 0) {
            const base = oldValPtr + GUEST_MEM_BASE;
            for (let i = 0; i < 32; i += 8) view.setBigUint64(base + i, 0n, true);
          }
          ret = 0n;
          break;
        }
        case 111:
          ret = 0n;
          break;
        case 114: {
          const resPtr = Number(a1);
          if (resPtr !== 0) {
            const base = resPtr + GUEST_MEM_BASE;
            view.setBigUint64(base, 0n, true);
            view.setBigUint64(base + 8, 1n, true);
          }
          ret = 0n;
          break;
        }
        case 32:
          ret = 0n;
          break;
        // ── Scheduling (per spec: mostly no-ops in single-core emulator) ──
        case 118:
        // sched_setparam
        case 119:
          ret = 0n;
          break;
        case 120:
          ret = 0n;
          break;
        case 121: {
          const spPtr = Number(a1) + GUEST_MEM_BASE;
          view.setInt32(spPtr, 0, true);
          ret = 0n;
          break;
        }
        case 125: {
          ret = 0n;
          break;
        }
        case 126: {
          ret = 0n;
          break;
        }
        // ── inotify (stub — accept creation, ignore watches) ──
        case 26: {
          const inoFd = this.allocFd();
          this.fdTable.set(inoFd, { type: "stdio", fd: inoFd });
          ret = BigInt(inoFd);
          break;
        }
        case 27:
          ret = 1n;
          break;
        case 28:
          ret = 0n;
          break;
        // ── timerfd (stub — create fd, accept settings) ──
        case 85: {
          const tfFd = this.allocFd();
          this.fdTable.set(tfFd, { type: "stdio", fd: tfFd });
          ret = BigInt(tfFd);
          break;
        }
        case 86: {
          const oldVal = Number(a3);
          if (oldVal !== 0) {
            const b = oldVal + GUEST_MEM_BASE;
            for (let i = 0; i < 32; i += 8) view.setBigUint64(b + i, 0n, true);
          }
          ret = 0n;
          break;
        }
        case 87: {
          const cur = Number(a1) + GUEST_MEM_BASE;
          for (let i = 0; i < 32; i += 8) view.setBigUint64(cur + i, 0n, true);
          ret = 0n;
          break;
        }
        // ── Misc process control ──
        case 97:
          ret = 0n;
          break;
        case 168: {
          const cpuPtr = Number(a0);
          const nodePtr = Number(a1);
          if (cpuPtr !== 0) view.setUint32(cpuPtr + GUEST_MEM_BASE, 0, true);
          if (nodePtr !== 0) view.setUint32(nodePtr + GUEST_MEM_BASE, 0, true);
          ret = 0n;
          break;
        }
        case 140:
          ret = 0n;
          break;
        case 141:
          ret = 20n;
          break;
        // ── Memory control (per spec: accept, no-op for our in-memory VFS) ──
        case 228:
        // mlock
        case 229:
        // munlock
        case 230:
        // mlockall
        case 231:
          ret = 0n;
          break;
        case 227:
          ret = 0n;
          break;
        case 232: {
          const mcLen = Number(a1);
          const mcVec = Number(a2) + GUEST_MEM_BASE;
          const pages = Math.ceil(mcLen / 4096);
          const b = new Uint8Array(memory2.buffer);
          for (let i = 0; i < pages; i++) b[mcVec + i] = 1;
          ret = 0n;
          break;
        }
        case 216: {
          ret = a0;
          break;
        }
        // ── Signal notification via fd ──
        case 74: {
          const sfFd = Number(BigInt.asIntN(32, a0));
          if (sfFd === -1) {
            const newFd = this.allocFd();
            this.fdTable.set(newFd, { type: "stdio", fd: newFd });
            ret = BigInt(newFd);
          } else {
            ret = BigInt(sfFd);
          }
          break;
        }
        // ── Process creation (modern) ──
        case 435: {
          const argsPtr = Number(a0) + GUEST_MEM_BASE;
          const cloneFlags = view.getBigUint64(argsPtr, true);
          const SIGCHLD2 = 17n;
          if (cloneFlags & SIGCHLD2) {
            const svcPC = view.getBigUint64(PC_OFFSET, true);
            return { status: "fork", resumePC: svcPC + 4n };
          }
          ret = -BigInt(ENOSYS);
          break;
        }
        // ── Exec relative to dirfd ──
        case 281: {
          const pathname281 = readGuestString(bytes, Number(a1));
          const argvPtrs281 = [];
          let argvAddr281 = Number(a2);
          while (true) {
            const ptr = Number(view.getBigUint64(argvAddr281 + GUEST_MEM_BASE, true));
            if (ptr === 0) break;
            argvPtrs281.push(readGuestString(bytes, ptr));
            argvAddr281 += 8;
          }
          const envpPtrs281 = [];
          let envpAddr281 = Number(a3);
          while (true) {
            const ptr = Number(view.getBigUint64(envpAddr281 + GUEST_MEM_BASE, true));
            if (ptr === 0) break;
            envpPtrs281.push(readGuestString(bytes, ptr));
            envpAddr281 += 8;
          }
          return { status: "execve", path: pathname281, argv: argvPtrs281, envp: envpPtrs281 };
        }
        case 47: {
          ret = 0n;
          break;
        }
        case 45: {
          const tpath = readGuestString(bytes, Number(a0));
          const tlength = Number(BigInt.asIntN(64, a1));
          if (this.vfs) {
            const fd = this.vfs.open(tpath, 2, 0);
            if (fd < 0) {
              ret = BigInt(fd);
              break;
            }
            ret = BigInt(this.vfs.ftruncate(fd, tlength));
            this.vfs.close(fd);
          } else {
            ret = -BigInt(ENOENT);
          }
          break;
        }
        case 71: {
          ret = -BigInt(ENOSYS);
          break;
        }
        case 276: {
          const oldpath276 = readGuestString(bytes, Number(a1));
          const newpath276 = readGuestString(bytes, Number(a3));
          const renameFlags = Number(a4);
          if (this.vfs) {
            if (renameFlags & 1) {
              ret = BigInt(this.vfs.rename(oldpath276, newpath276));
            } else {
              ret = BigInt(this.vfs.rename(oldpath276, newpath276));
            }
          } else {
            ret = -BigInt(ENOSYS);
          }
          break;
        }
        case 279: {
          if (this.vfs) {
            const mfName = readGuestString(bytes, Number(a0));
            const tmpPath = "/tmp/.memfd_" + mfName + "_" + Date.now();
            const fd = this.vfs.open(tmpPath, 64 | 2, 384);
            ret = BigInt(fd);
          } else {
            ret = -BigInt(ENOSYS);
          }
          break;
        }
        case 291: {
          const sxPath = readGuestString(bytes, Number(a1));
          const sxFlags = Number(a2);
          const sxBufAddr = Number(a4);
          if (this.vfs) {
            const tmpOffset = sxBufAddr + 128;
            const res = this.vfs.fstatat(Number(BigInt.asIntN(32, a0)), sxPath, tmpOffset, sxFlags, view);
            if (res < 0) {
              ret = BigInt(res);
              break;
            }
            const st = tmpOffset + GUEST_MEM_BASE;
            const stDev = view.getBigUint64(st + 0, true);
            const stIno = view.getBigUint64(st + 8, true);
            const stMode = view.getUint32(st + 16, true);
            const stNlink = view.getUint32(st + 20, true);
            const stUid = view.getUint32(st + 24, true);
            const stGid = view.getUint32(st + 28, true);
            const stRdev = view.getBigUint64(st + 32, true);
            const stSize = view.getBigInt64(st + 48, true);
            const stBlksize = view.getInt32(st + 56, true);
            const stBlocks = view.getBigInt64(st + 64, true);
            const stAtime = view.getBigInt64(st + 72, true);
            const stMtime = view.getBigInt64(st + 88, true);
            const stCtime = view.getBigInt64(st + 104, true);
            const b = sxBufAddr + GUEST_MEM_BASE;
            for (let i = 0; i < 256; i += 8) view.setBigUint64(b + i, 0n, true);
            view.setUint32(b + 0, 6143, true);
            view.setUint32(b + 4, stBlksize, true);
            view.setBigUint64(b + 8, 0n, true);
            view.setUint32(b + 16, stNlink, true);
            view.setUint32(b + 20, stUid, true);
            view.setUint32(b + 24, stGid, true);
            view.setUint16(b + 28, stMode & 65535, true);
            view.setBigUint64(b + 32, stIno, true);
            view.setBigUint64(b + 40, BigInt.asUintN(64, stSize), true);
            view.setBigUint64(b + 48, BigInt.asUintN(64, stBlocks), true);
            view.setBigUint64(b + 56, 0n, true);
            view.setBigInt64(b + 64, stAtime, true);
            view.setBigInt64(b + 80, stCtime, true);
            view.setBigInt64(b + 96, stCtime, true);
            view.setBigInt64(b + 112, stMtime, true);
            view.setUint32(b + 128, 0, true);
            view.setUint32(b + 132, Number(stRdev), true);
            view.setUint32(b + 136, 0, true);
            view.setUint32(b + 140, Number(stDev), true);
          }
          ret = 0n;
          break;
        }
        // ── I/O multiplexing (epoll, eventfd) ──
        case 19: {
          if (this.vfs) {
            const buf = new Uint8Array(8);
            const dv = new DataView(buf.buffer);
            dv.setBigUint64(0, BigInt(Number(a0)), true);
            const fd = this.vfs.open("/dev/null", 0, 0);
            ret = BigInt(fd >= 0 ? fd : -EBADF);
          } else {
            ret = -BigInt(ENOSYS);
          }
          break;
        }
        case 20: {
          const epfd = this.allocFd();
          this.fdTable.set(epfd, { type: "stdio", fd: epfd });
          if (!this.epollSets) this.epollSets = /* @__PURE__ */ new Map();
          this.epollSets.set(epfd, /* @__PURE__ */ new Map());
          ret = BigInt(epfd);
          break;
        }
        case 21: {
          const epfd = Number(a0);
          const epOp = Number(a1);
          const targetFd = Number(a2);
          const eventPtr = Number(a3);
          if (!this.epollSets?.has(epfd)) {
            ret = -BigInt(EBADF);
            break;
          }
          const eset = this.epollSets.get(epfd);
          if (epOp === 1 || epOp === 3) {
            const events = view.getUint32(eventPtr + GUEST_MEM_BASE, true);
            const data = view.getBigUint64(eventPtr + GUEST_MEM_BASE + 4, true);
            eset.set(targetFd, { events, data });
            ret = 0n;
          } else if (epOp === 2) {
            eset.delete(targetFd);
            ret = 0n;
          } else {
            ret = -BigInt(EINVAL);
          }
          break;
        }
        case 22: {
          const epfd = Number(a0);
          const eventsPtr = Number(a1);
          const maxEvents = Number(a2);
          if (!this.epollSets?.has(epfd)) {
            ret = -BigInt(EBADF);
            break;
          }
          const eset = this.epollSets.get(epfd);
          let count = 0;
          const EPOLLIN = 1, EPOLLOUT = 4;
          for (const [targetFd, entry] of eset) {
            if (count >= maxEvents) break;
            const readyEvents = entry.events & (EPOLLIN | EPOLLOUT);
            if (readyEvents) {
              const off = eventsPtr + count * 12 + GUEST_MEM_BASE;
              view.setUint32(off, readyEvents, true);
              view.setBigUint64(off + 4, entry.data, true);
              count++;
            }
          }
          ret = BigInt(count);
          break;
        }
        case 72: {
          const nfds = Number(a0);
          ret = BigInt(nfds);
          break;
        }
        case 73: {
          const fdsAddr = Number(a0);
          const nfds = Number(a1);
          const tmoPtr = Number(a2);
          let tmoMs = -1;
          if (tmoPtr !== 0) {
            const tmoSec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE, true));
            const tmoNsec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE + 8, true));
            tmoMs = tmoSec * 1e3 + Math.floor(tmoNsec / 1e6);
          }
          let readyCount = 0;
          let socketFd = -1;
          let socketEntry = null;
          for (let i = 0; i < nfds; i++) {
            const pfd = fdsAddr + i * 8 + GUEST_MEM_BASE;
            const fd = view.getInt32(pfd, true);
            const events = view.getInt16(pfd + 4, true);
            if (fd < 0) {
              view.setInt16(pfd + 6, 0, true);
              continue;
            }
            const entry = this.getFdEntry(fd);
            if (entry?.type === "socket" && events & 1) {
              view.setInt16(pfd + 6, 0, true);
              if (socketFd < 0) {
                socketFd = fd;
                socketEntry = entry;
              }
            } else {
              view.setInt16(pfd + 6, events, true);
              if (events) readyCount++;
            }
          }
          if (readyCount > 0) {
            ret = BigInt(readyCount);
            break;
          }
          if (socketFd >= 0 && socketEntry) {
            const pollTmo = tmoMs < 0 ? 3e4 : tmoMs;
            return {
              status: "bridge",
              method: "net_poll",
              args: { socketId: socketEntry.socketId, timeoutMs: pollTmo },
              apply: (resp) => {
                if (resp.ret > 0) {
                  for (let i = 0; i < nfds; i++) {
                    const pfd = fdsAddr + i * 8 + GUEST_MEM_BASE;
                    const fd = view.getInt32(pfd, true);
                    const events = view.getInt16(pfd + 4, true);
                    if (fd >= 0 && this.getFdEntry(fd)?.type === "socket" && events & 1) {
                      view.setInt16(pfd + 6, events, true);
                    }
                  }
                  cpu2.setReg(0, 1n);
                } else {
                  cpu2.setReg(0, 0n);
                }
              }
            };
          }
          ret = 0n;
          break;
        }
        // ── Remaining syscalls from aarch64 spec ──
        // Mount/unmount (needed for container/init, stub for now)
        case 39:
        // mount
        case 40:
          ret = 0n;
          break;
        case 60:
          ret = 0n;
          break;
        case 41:
          ret = -BigInt(EPERM);
          break;
        // Robust futex list
        case 100: {
          const headPtr = Number(a1) + GUEST_MEM_BASE;
          const lenPtr = Number(a2) + GUEST_MEM_BASE;
          view.setBigUint64(headPtr, 0n, true);
          view.setBigUint64(lenPtr, 24n, true);
          ret = 0n;
          break;
        }
        // Signal: tkill and rt_sigreturn
        case 130:
          ret = 0n;
          break;
        case 139: {
          const restoredPC = this.handleSigreturn(cpu2, memory2);
          cpu2.setPC(restoredPC);
          ret = cpu2.getReg(0);
          break;
        }
        // Networking (remaining)
        case 210:
          ret = -BigInt(EBADF);
          break;
        case 211: {
          const smEntry = this.getFdEntry(Number(a0));
          if (!smEntry || smEntry.type !== "socket") {
            ret = -BigInt(EBADF);
            break;
          }
          const mhdr = Number(a1) + GUEST_MEM_BASE;
          const iovPtr = Number(view.getBigUint64(mhdr + 16, true));
          const iovLen = Number(view.getBigUint64(mhdr + 24, true));
          let totalLen = 0;
          for (let i = 0; i < iovLen; i++) {
            totalLen += Number(view.getBigUint64(iovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
          }
          const combined = new Uint8Array(totalLen);
          let off = 0;
          for (let i = 0; i < iovLen; i++) {
            const base = Number(view.getBigUint64(iovPtr + i * 16 + GUEST_MEM_BASE, true));
            const len = Number(view.getBigUint64(iovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
            combined.set(bytes.subarray(base + GUEST_MEM_BASE, base + GUEST_MEM_BASE + len), off);
            off += len;
          }
          const smB64 = btoa(String.fromCharCode(...combined));
          const namePtr = Number(view.getBigUint64(mhdr, true));
          const nameLen = view.getUint32(mhdr + 8, true);
          if (namePtr !== 0 && nameLen > 0) {
            const dest = this.parseSockaddr(bytes, namePtr, nameLen);
            return {
              status: "bridge",
              method: "net_sendto",
              args: { socketId: smEntry.socketId, data: smB64, host: dest.host, port: dest.port },
              apply: (resp) => {
                cpu2.setReg(0, resp.ret < 0 ? BigInt.asUintN(64, BigInt(resp.ret)) : BigInt(totalLen));
              }
            };
          }
          return {
            status: "bridge",
            method: "net_write",
            args: { socketId: smEntry.socketId, data: smB64 },
            apply: (resp) => {
              cpu2.setReg(0, resp.ret < 0 ? BigInt.asUintN(64, BigInt(resp.ret)) : BigInt(totalLen));
            }
          };
        }
        case 212: {
          const rmEntry = this.getFdEntry(Number(a0));
          if (!rmEntry || rmEntry.type !== "socket") {
            ret = -BigInt(EBADF);
            break;
          }
          const rmhdr = Number(a1) + GUEST_MEM_BASE;
          const rmIovPtr = Number(view.getBigUint64(rmhdr + 16, true));
          const rmIovLen = Number(view.getBigUint64(rmhdr + 24, true));
          let rmCapacity = 0;
          for (let i = 0; i < rmIovLen; i++) {
            rmCapacity += Number(view.getBigUint64(rmIovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
          }
          console.log(`[net] recvmsg(fd=${Number(a0)}, capacity=${rmCapacity})`);
          return {
            status: "bridge",
            method: "net_read",
            args: { socketId: rmEntry.socketId, count: rmCapacity },
            apply: (resp) => {
              if (resp.ret < 0) {
                console.log(`[net] recvmsg \u2192 err ${resp.ret}`);
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                return;
              }
              let recvd;
              if (resp.data) {
                recvd = Uint8Array.from(atob(resp.data), (c) => c.charCodeAt(0));
              } else {
                cpu2.setReg(0, BigInt(resp.ret));
                return;
              }
              let remaining = recvd.length;
              let srcOff = 0;
              for (let i = 0; i < rmIovLen && remaining > 0; i++) {
                const base = Number(view.getBigUint64(rmIovPtr + i * 16 + GUEST_MEM_BASE, true));
                const len = Number(view.getBigUint64(rmIovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
                const copyLen = Math.min(len, remaining);
                bytes.set(recvd.subarray(srcOff, srcOff + copyLen), base + GUEST_MEM_BASE);
                srcOff += copyLen;
                remaining -= copyLen;
              }
              const namePtr = Number(view.getBigUint64(rmhdr, true));
              const nameLenPtr = rmhdr + 8;
              if (namePtr !== 0 && rmEntry.remoteAddr) {
                const written = this.writeSockaddr(view, namePtr, rmEntry.remoteAddr);
                view.setUint32(nameLenPtr, written, true);
              }
              view.setInt32(rmhdr + 48, 0, true);
              view.setBigUint64(rmhdr + 40, 0n, true);
              console.log(`[net] recvmsg \u2192 ${recvd.length}B`);
              cpu2.setReg(0, BigInt(recvd.length));
            }
          };
        }
        case 213:
          ret = -BigInt(ENOSYS);
          break;
        case 242:
          ret = -BigInt(EBADF);
          break;
        case 243:
        // recvmmsg
        case 269:
          ret = -BigInt(EBADF);
          break;
        case 199:
          ret = -BigInt(ENOSYS);
          break;
        // File operations (remaining)
        case 75:
        // splice — copy between fds via pipe
        case 76:
          ret = -BigInt(ENOSYS);
          break;
        case 77:
          ret = 0n;
          break;
        case 81:
          ret = 0n;
          break;
        case 18:
          ret = -BigInt(ENOSYS);
          break;
        // Capabilities
        case 90: {
          const capDataPtr = Number(a1);
          if (capDataPtr !== 0) {
            const base = capDataPtr + GUEST_MEM_BASE;
            view.setUint32(base, 4294967295, true);
            view.setUint32(base + 4, 4294967295, true);
            view.setUint32(base + 8, 4294967295, true);
            view.setUint32(base + 12, 4294967295, true);
            view.setUint32(base + 16, 4294967295, true);
            view.setUint32(base + 20, 4294967295, true);
          }
          const capHdrPtr = Number(a0);
          if (capHdrPtr !== 0) {
            const hBase = capHdrPtr + GUEST_MEM_BASE;
            view.setUint32(hBase, 537396514, true);
            view.setInt32(hBase + 4, this.currentPid, true);
          }
          ret = 0n;
          break;
        }
        case 91:
          ret = 0n;
          break;
        // Kernel log
        case 116:
          ret = 0n;
          break;
        // Process tracing
        case 117:
          ret = -BigInt(EPERM);
          break;
        // Scheduling
        case 127: {
          const rrTp = Number(a1) + GUEST_MEM_BASE;
          view.setBigUint64(rrTp, 0n, true);
          view.setBigUint64(rrTp + 8, 100000000n, true);
          ret = 0n;
          break;
        }
        case 128:
          ret = -BigInt(4);
          break;
        // Time
        case 170:
          ret = -BigInt(EPERM);
          break;
        // Extended attributes
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
          ret = -BigInt(95);
          break;
        // Security
        case 277:
          ret = -BigInt(EINVAL);
          break;
        case 293:
          ret = -BigInt(ENOSYS);
          break;
        // Copy file range
        case 285: {
          const cfrInFd = Number(a0);
          const cfrOffInPtr = Number(a1);
          const cfrOutFd = Number(a2);
          const cfrOffOutPtr = Number(a3);
          const cfrLen = Number(a4);
          if (!this.vfs) {
            ret = -BigInt(EBADF);
            break;
          }
          const inSaved = this.vfs.lseek(cfrInFd, 0n, 1);
          const outSaved = this.vfs.lseek(cfrOutFd, 0n, 1);
          if (inSaved < 0n || outSaved < 0n) {
            ret = -BigInt(EBADF);
            break;
          }
          if (cfrOffInPtr !== 0) {
            const offIn = view.getBigInt64(cfrOffInPtr + GUEST_MEM_BASE, true);
            this.vfs.lseek(cfrInFd, offIn, 0);
          }
          if (cfrOffOutPtr !== 0) {
            const offOut = view.getBigInt64(cfrOffOutPtr + GUEST_MEM_BASE, true);
            this.vfs.lseek(cfrOutFd, offOut, 0);
          }
          const n = this.vfs.copyFileRange?.(cfrInFd, cfrOutFd, cfrLen) ?? -38;
          ret = BigInt(n);
          if (cfrOffInPtr !== 0 && n > 0) {
            const newOff = view.getBigInt64(cfrOffInPtr + GUEST_MEM_BASE, true) + BigInt(n);
            view.setBigInt64(cfrOffInPtr + GUEST_MEM_BASE, newOff, true);
            this.vfs.lseek(cfrInFd, inSaved, 0);
          }
          if (cfrOffOutPtr !== 0 && n > 0) {
            const newOff = view.getBigInt64(cfrOffOutPtr + GUEST_MEM_BASE, true) + BigInt(n);
            view.setBigInt64(cfrOffOutPtr + GUEST_MEM_BASE, newOff, true);
            this.vfs.lseek(cfrOutFd, outSaved, 0);
          }
          break;
        }
        // Process accounting
        case 89:
          ret = -BigInt(ENOSYS);
          break;
        // Namespace
        case 268:
          ret = -BigInt(EPERM);
          break;
        // ── Async I/O (io_setup, io_destroy, io_submit, io_cancel, io_getevents) ──
        case 0:
        case 1:
        case 2:
        case 4:
          ret = -BigInt(ENOSYS);
          break;
        case 3:
          ret = -BigInt(EINVAL);
          break;
        // ── splice/tee/vmsplice ──
        case 30:
        // fanotify_init
        case 31:
          ret = -BigInt(ENOSYS);
          break;
        case 42:
          ret = -BigInt(ENOSYS);
          break;
        case 51: {
          ret = -BigInt(ENOSYS);
          break;
        }
        case 58:
          ret = -BigInt(ENOSYS);
          break;
        case 69: {
          const prvFd = Number(a0);
          const prvIov = Number(a1);
          const prvCnt = Number(a2);
          const prvOff = BigInt.asIntN(64, a3);
          if (!this.vfs) {
            ret = -BigInt(EBADF);
            break;
          }
          const prvSaved = this.vfs.lseek(prvFd, 0n, 1);
          if (prvSaved < 0n) {
            ret = prvSaved;
            break;
          }
          this.vfs.lseek(prvFd, prvOff, 0);
          let prvTotal = 0;
          for (let i = 0; i < prvCnt; i++) {
            const base = Number(view.getBigUint64(prvIov + i * 16 + GUEST_MEM_BASE, true));
            const len = Number(view.getBigUint64(prvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
            if (len === 0) continue;
            const n = this.vfs.read(prvFd, base, len, view);
            if (n < 0) {
              if (prvTotal === 0) prvTotal = n;
              break;
            }
            prvTotal += n;
            if (n < len) break;
          }
          this.vfs.lseek(prvFd, prvSaved, 0);
          ret = BigInt(prvTotal);
          break;
        }
        case 70: {
          const pwvFd = Number(a0);
          const pwvIov = Number(a1);
          const pwvCnt = Number(a2);
          const pwvOff = BigInt.asIntN(64, a3);
          if (!this.vfs) {
            ret = -BigInt(EBADF);
            break;
          }
          const pwvSaved = this.vfs.lseek(pwvFd, 0n, 1);
          if (pwvSaved < 0n) {
            ret = pwvSaved;
            break;
          }
          this.vfs.lseek(pwvFd, pwvOff, 0);
          let pwvTotal = 0;
          for (let i = 0; i < pwvCnt; i++) {
            const base = Number(view.getBigUint64(pwvIov + i * 16 + GUEST_MEM_BASE, true));
            const len = Number(view.getBigUint64(pwvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
            if (len === 0) continue;
            const n = this.vfs.write(pwvFd, base, len, view);
            if (n < 0) {
              if (pwvTotal === 0) pwvTotal = n;
              break;
            }
            pwvTotal += n;
            if (n < len) break;
          }
          this.vfs.lseek(pwvFd, pwvSaved, 0);
          ret = BigInt(pwvTotal);
          break;
        }
        // timerfd (verify correct numbers)
        // Per aarch64 spec: timerfd_create=85, timerfd_settime=86, timerfd_gettime=87
        // 82-84 are unused on aarch64? Let me check — they might be signalfd/timerfd/eventfd
        // Actually: 82=signalfd (old), 83=timerfd_create (old), 84=eventfd (old)
        // These are the OLD non-flagged versions; aarch64 should use 74/85/19 instead
        case 82:
        // signalfd (old, use signalfd4=74)
        case 83:
        // timerfd_create (old, use 85)
        case 84:
          ret = -BigInt(ENOSYS);
          break;
        // setitimer/getitimer duplicates check
        case 104:
          ret = -BigInt(EPERM);
          break;
        case 105:
        // ??? (105 = clock_adjtime_time64 — doesn't exist on aarch64)
        case 106:
          ret = -BigInt(ENOSYS);
          break;
        // Clock
        case 112:
          ret = -BigInt(EPERM);
          break;
        // Signal (remaining)
        case 137:
          ret = -BigInt(4);
          break;
        case 138:
          ret = 0n;
          break;
        // ioctl subcommand handling is in case 29 already
        case 142:
          ret = 0n;
          break;
        // getrlimit/setrlimit (prlimit64 at 261 is the modern replacement)
        case 161:
        case 162:
          if (nr === 161) ret = 0n;
          else ret = 0n;
          break;
        case 163:
        // getrlimit — redirect to prlimit64 logic
        case 164:
          ret = 0n;
          break;
        case 171:
          ret = -BigInt(EPERM);
          break;
        // Key management
        case 180:
        case 181:
        case 182:
        case 183:
        case 184:
        case 185:
          ret = -BigInt(ENOSYS);
          break;
        // SysV IPC
        case 186:
        case 187:
        case 188:
        case 189:
        // msgget, msgctl, msgrcv, msgsnd
        case 190:
        case 191:
        case 192:
        case 193:
        // semget, semctl, semtimedop, semop
        case 194:
        case 195:
        case 196:
        case 197:
          ret = -BigInt(ENOSYS);
          break;
        // getdents64 — we have this at case 61 which is correct on aarch64
        // 217 = getdents (old 32-bit version, not used on aarch64)
        case 217:
          ret = -BigInt(ENOSYS);
          break;
        // Miscellaneous
        case 218:
        // set_mempolicy
        case 219:
          ret = 0n;
          break;
        case 223:
        // mq_open
        case 224:
        // mq_unlink
        case 225:
          ret = -BigInt(ENOSYS);
          break;
        case 234:
          ret = -BigInt(ENOSYS);
          break;
        case 235:
          ret = 0n;
          break;
        case 236:
          ret = 0n;
          break;
        case 237:
          ret = -BigInt(ENOSYS);
          break;
        case 238:
        // preadv2
        case 239:
          ret = -BigInt(ENOSYS);
          break;
        case 240:
          ret = 0n;
          break;
        case 241:
          ret = -BigInt(ENOSYS);
          break;
        case 244:
          ret = -BigInt(ENOSYS);
          break;
        // io_uring (425-427)
        case 425:
        case 426:
        case 427:
          ret = -BigInt(ENOSYS);
          break;
        // Remaining 245-294 (arch specific range + various)
        case 262:
        // fanotify_init
        case 263:
          ret = -BigInt(ENOSYS);
          break;
        case 264:
        // name_to_handle_at
        case 265:
          ret = -BigInt(ENOSYS);
          break;
        case 266:
          ret = -BigInt(EPERM);
          break;
        case 267:
          ret = 0n;
          break;
        case 270:
        // process_vm_readv
        case 271:
          ret = -BigInt(ENOSYS);
          break;
        case 272:
          ret = -BigInt(ENOSYS);
          break;
        case 273:
          ret = -BigInt(EPERM);
          break;
        case 274:
          ret = 0n;
          break;
        case 275: {
          const saPtr = Number(a1) + GUEST_MEM_BASE;
          const saSize = Number(a2);
          for (let i = 0; i < Math.min(saSize, 56); i += 8) view.setBigUint64(saPtr + i, 0n, true);
          view.setUint32(saPtr, Math.min(saSize, 56), true);
          ret = 0n;
          break;
        }
        case 280:
          ret = -BigInt(ENOSYS);
          break;
        case 282:
          ret = -BigInt(ENOSYS);
          break;
        case 283:
          ret = 0n;
          break;
        case 284:
          ret = 0n;
          break;
        case 286:
        // preadv2
        case 287:
          ret = -BigInt(ENOSYS);
          break;
        case 288:
        // pkey_mprotect
        case 289:
        // pkey_alloc
        case 290:
          ret = -BigInt(ENOSYS);
          break;
        case 292:
          ret = -BigInt(ENOSYS);
          break;
        case 294:
          ret = -BigInt(EPERM);
          break;
        // 295-402: unassigned on aarch64 (kernel reserves these)
        // 403-423: time64 compat syscalls (32-bit only, not available on native aarch64)
        // All return ENOSYS.
        // 245-259: arch_specific_syscall range (unused on aarch64)
        case 245:
        case 246:
        case 247:
        case 248:
        case 249:
        case 250:
        case 251:
        case 252:
        case 253:
        case 254:
        case 255:
        case 256:
        case 257:
        case 258:
        case 259:
          ret = -BigInt(ENOSYS);
          break;
        case 424:
          ret = -BigInt(ENOSYS);
          break;
        // io_uring (425-427) already handled above
        case 428:
        // open_tree
        case 429:
        // move_mount
        case 430:
        // fsopen
        case 431:
        // fsconfig
        case 432:
        // fsmount
        case 433:
          ret = -BigInt(ENOSYS);
          break;
        case 434:
          ret = -BigInt(ENOSYS);
          break;
        // clone3 already handled at 435
        case 436:
          if (this.vfs) {
            const first = Number(a0);
            const last = Number(a1);
            for (let fd = first; fd <= Math.min(last, MAX_FDS); fd++) {
              if (this.fdTable.has(fd)) {
                this.fdTable.delete(fd);
              }
              if (this.vfs.hasFd(fd)) {
                this.vfs.close(fd);
              }
            }
          }
          ret = 0n;
          break;
        case 437: {
          const path437 = readGuestString(bytes, Number(a1));
          const howPtr = Number(a2) + GUEST_MEM_BASE;
          const flags437 = Number(view.getBigUint64(howPtr, true));
          const mode437 = Number(view.getBigUint64(howPtr + 8, true));
          if (this.vfs) {
            const effectiveMode = mode437 & ~this.umask;
            const openRet = this.vfs.open(path437, flags437, effectiveMode);
            if (openRet >= 0) {
              this.fdFlags.set(openRet, flags437 & ~(64 | 128 | 512 | 524288));
              if (flags437 & 524288) this.fdCloexec.add(openRet);
            }
            ret = BigInt(openRet);
          } else {
            ret = -BigInt(ENOENT);
          }
          break;
        }
        case 438:
          ret = -BigInt(ENOSYS);
          break;
        case 439: {
          const fa2Path = readGuestString(bytes, Number(a1));
          if (this.vfs) {
            ret = BigInt(this.vfs.faccessat(Number(BigInt.asIntN(32, a0)), fa2Path, Number(a2), Number(a3)));
          } else {
            ret = -BigInt(ENOENT);
          }
          break;
        }
        case 440:
          ret = -BigInt(ENOSYS);
          break;
        case 441: {
          const ep2fd = Number(a0);
          if (!this.epollSets?.has(ep2fd)) {
            ret = -BigInt(EBADF);
            break;
          }
          const ep2set = this.epollSets.get(ep2fd);
          const ep2events = Number(a1);
          const ep2max = Number(a2);
          let ep2count = 0;
          for (const [targetFd, entry] of ep2set) {
            if (ep2count >= ep2max) break;
            const ready = entry.events & 5;
            if (ready) {
              const off = ep2events + ep2count * 12 + GUEST_MEM_BASE;
              view.setUint32(off, ready, true);
              view.setBigUint64(off + 4, entry.data, true);
              ep2count++;
            }
          }
          ret = BigInt(ep2count);
          break;
        }
        case 442:
          ret = 0n;
          break;
        case 443:
          ret = -BigInt(ENOSYS);
          break;
        case 444:
        case 445:
        case 446:
          ret = -BigInt(ENOSYS);
          break;
        case 447:
          ret = -BigInt(ENOSYS);
          break;
        case 448:
          ret = -BigInt(ENOSYS);
          break;
        case 449:
          ret = -BigInt(ENOSYS);
          break;
        case 450:
          ret = 0n;
          break;
        case 451:
          ret = -BigInt(ENOSYS);
          break;
        case 452: {
          ret = 0n;
          break;
        }
        case 453:
          ret = -BigInt(ENOSYS);
          break;
        case 454:
        // futex_wake (new API)
        case 455:
        // futex_wait (new API)
        case 456:
          ret = -BigInt(ENOSYS);
          break;
        case 457:
        // statmount
        case 458:
          ret = -BigInt(ENOSYS);
          break;
        case 459:
        case 460:
        case 461:
          ret = -BigInt(ENOSYS);
          break;
        case 462:
          ret = 0n;
          break;
        case 463:
        case 464:
        case 465:
        case 466:
          ret = -BigInt(95);
          break;
        case 467:
          ret = -BigInt(ENOSYS);
          break;
        case 468:
        case 469:
          ret = -BigInt(ENOSYS);
          break;
        case 470:
          ret = -BigInt(ENOSYS);
          break;
        case 471:
          ret = -BigInt(ENOSYS);
          break;
        default:
          if (nr >= 0 && nr <= 450) {
            ret = -BigInt(ENOSYS);
          } else {
            console.warn(`[jitter] unknown syscall ${nr} (X0=${a0}, X1=${a1}, X2=${a2})`);
            ret = -BigInt(ENOSYS);
          }
      }
      cpu2.setReg(0, BigInt.asUintN(64, ret));
      return { status: "ok" };
    }
  };
  function readGuestString(memory2, guestAddr) {
    const base = guestAddr + GUEST_MEM_BASE;
    let end = base;
    while (end < memory2.length && memory2[end] !== 0) end++;
    return new TextDecoder().decode(memory2.subarray(base, end));
  }
  function writeChardevStat(view, guestAddr) {
    const base = guestAddr + GUEST_MEM_BASE;
    for (let i = 0; i < 128; i += 8) {
      view.setBigUint64(base + i, 0n, true);
    }
    view.setUint32(base + 16, 8630, true);
    view.setUint32(base + 20, 1, true);
  }
  function writeBridgeStat(view, guestAddr, resp) {
    const base = guestAddr + GUEST_MEM_BASE;
    for (let i = 0; i < 128; i += 8) {
      view.setBigUint64(base + i, 0n, true);
    }
    view.setUint32(base + 16, resp.mode ?? 33188, true);
    view.setUint32(base + 20, resp.nlink ?? 1, true);
    view.setBigUint64(base + 48, BigInt(resp.size ?? 0), true);
    view.setBigUint64(base + 88, BigInt(resp.mtime ?? 0), true);
  }

  // src/vfs.ts
  var EPERM2 = 1;
  var ENOENT2 = 2;
  var EBADF2 = 9;
  var EEXIST = 17;
  var ENOTDIR = 20;
  var EISDIR = 21;
  var EINVAL2 = 22;
  var EMFILE = 24;
  var ESPIPE2 = 29;
  var ENAMETOOLONG = 36;
  var ENOTEMPTY = 39;
  var ELOOP = 40;
  var O_ACCMODE = 3;
  var O_RDONLY = 0;
  var O_WRONLY = 1;
  var O_CREAT = 64;
  var O_EXCL = 128;
  var O_TRUNC = 512;
  var O_APPEND = 1024;
  var O_DIRECTORY = 65536;
  var O_NOFOLLOW = 131072;
  var S_IFMT = 61440;
  var S_IFCHR = 8192;
  var S_IFDIR = 16384;
  var S_IFREG = 32768;
  var S_IFLNK = 40960;
  var NAME_MAX = 255;
  var PATH_MAX = 4096;
  var MAX_SYMLINK_DEPTH = 40;
  var MAX_FDS2 = 1024;
  var InMemoryVFS = class {
    root;
    openFds = /* @__PURE__ */ new Map();
    nextFd = 3;
    // 0=stdin, 1=stdout, 2=stderr
    nextIno = 1;
    // inode counter
    cwd = "/";
    cwdInode;
    constructor() {
      this.root = this.makeDir(null, 493);
      this.cwdInode = this.root;
      this.populateDefaults();
    }
    // ── Inode creation helpers ──────────────────────────────────────────────
    now() {
      return Math.floor(Date.now() / 1e3);
    }
    makeFile(mode, data = new Uint8Array(0)) {
      const t = this.now();
      return { type: "file", ino: this.nextIno++, mode: S_IFREG | mode & 4095, nlink: 1, uid: 0, gid: 0, atime: t, mtime: t, ctime: t, data };
    }
    makeDir(parent, mode) {
      const t = this.now();
      const dir = { type: "dir", ino: this.nextIno++, mode: S_IFDIR | mode & 4095, nlink: 2, uid: 0, gid: 0, atime: t, mtime: t, ctime: t, children: /* @__PURE__ */ new Map(), parent: null };
      dir.parent = parent ?? dir;
      return dir;
    }
    makeCharDev(devno, mode) {
      const t = this.now();
      return { type: "chardev", ino: this.nextIno++, mode: S_IFCHR | mode & 4095, nlink: 1, uid: 0, gid: 0, atime: t, mtime: t, ctime: t, devno };
    }
    makeSymlink(target) {
      const t = this.now();
      return { type: "symlink", ino: this.nextIno++, mode: S_IFLNK | 511, nlink: 1, uid: 0, gid: 0, atime: t, mtime: t, ctime: t, target };
    }
    // ── Default filesystem population ───────────────────────────────────────
    populateDefaults() {
      this.mkdirp("/dev");
      this.mkdirp("/etc");
      this.mkdirp("/proc/self/fd");
      this.mkdirp("/proc/sys/kernel");
      this.mkdirp("/proc/sys/fs");
      this.mkdirp("/tmp");
      this.mkdirp("/root");
      this.mkdirp("/bin");
      this.mkdirp("/home");
      this.mkdirp("/var");
      this.addCharDev("/dev/null", 1, 438);
      this.addCharDev("/dev/zero", 2, 438);
      this.addCharDev("/dev/tty", 3, 438);
      this.addCharDev("/dev/urandom", 4, 292);
      this.addFileContent(
        "/etc/passwd",
        "root:x:0:0:root:/root:/bin/sh\ndaemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin\nnobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n"
      );
      this.addFileContent("/etc/group", "root:x:0:\ndaemon:x:1:\nnobody:x:65534:\nnogroup:x:65534:\n");
      this.addFileContent("/etc/hostname", "jitter\n");
      this.addFileContent("/etc/shells", "/bin/sh\n");
      this.addFileContent("/etc/resolv.conf", "nameserver 1.1.1.1\nnameserver 1.0.0.1\n");
      this.addFileContent("/etc/nsswitch.conf", "hosts: files dns\n");
      this.addSymlinkNode("/proc/self/exe", "/bin/busybox");
      this.addFileContent(
        "/proc/meminfo",
        "MemTotal:       1048576 kB\nMemFree:         524288 kB\nMemAvailable:    786432 kB\nBuffers:          65536 kB\nCached:          131072 kB\nSwapTotal:            0 kB\nSwapFree:             0 kB\n"
      );
      this.addFileContent(
        "/proc/cpuinfo",
        "processor	: 0\nBogoMIPS	: 48.00\nFeatures	: fp asimd aes sha1 sha2 crc32\nCPU implementer	: 0x61\nCPU architecture: 8\nCPU variant	: 0x1\nCPU part	: 0x000\n"
      );
      this.addFileContent("/proc/sys/kernel/pid_max", "32768\n");
      this.addFileContent("/proc/sys/fs/pipe-user-pages-soft", "16384\n");
      this.addFileContent("/proc/sys/fs/pipe-max-size", "1048576\n");
      this.addFileContent(
        "/proc/self/status",
        "Name:	jitter\nUmask:	0022\nState:	R (running)\nTgid:	1\nNgid:	0\nPid:	1\nPPid:	0\nTracerPid:	0\nUid:	0	0	0	0\nGid:	0	0	0	0\nFDSize:	64\nGroups:	0\nVmPeak:	    65536 kB\nVmSize:	    65536 kB\nVmRSS:	    32768 kB\nThreads:	1\n"
      );
      this.addFileContent("/proc/mounts", "tmpfs / tmpfs rw,relatime 0 0\nproc /proc proc rw,relatime 0 0\n");
      this.addFileContent("/proc/filesystems", "nodev	tmpfs\n");
    }
    loadBinary(data) {
      this.setFileData("/bin/busybox", data, 493);
      const applets = [
        "sh",
        "ash",
        "ls",
        "cat",
        "echo",
        "grep",
        "find",
        "wc",
        "head",
        "tail",
        "sort",
        "uniq",
        "sed",
        "awk",
        "cut",
        "tr",
        "tee",
        "xargs",
        "cp",
        "mv",
        "rm",
        "mkdir",
        "rmdir",
        "ln",
        "chmod",
        "chown",
        "touch",
        "pwd",
        "env",
        "test",
        "[",
        "true",
        "false",
        "yes",
        "seq",
        "expr",
        "date",
        "sleep",
        "uname",
        "hostname",
        "whoami",
        "id",
        "which",
        "du",
        "df",
        "free",
        "uptime",
        "ps",
        "kill",
        "basename",
        "dirname",
        "readlink",
        "realpath",
        "stat",
        "md5sum",
        "sha256sum",
        "dd",
        "hexdump",
        "od",
        "strings",
        "diff",
        "patch",
        "tar",
        "gzip",
        "gunzip",
        "zcat",
        "bzip2",
        "bunzip2",
        "vi",
        "less",
        "more",
        "clear",
        "wget",
        "nc",
        "ping",
        "nslookup",
        "traceroute"
      ];
      this.mkdirp("/usr/bin");
      for (const name of applets) {
        this.addSymlinkNode(`/bin/${name}`, "/bin/busybox");
        this.addSymlinkNode(`/usr/bin/${name}`, "/bin/busybox");
      }
    }
    // ── Internal tree manipulation ──────────────────────────────────────────
    mkdirp(path) {
      const parts = path.split("/").filter((p) => p.length > 0);
      let current = this.root;
      for (const part of parts) {
        let child = current.children.get(part);
        if (!child) {
          const dir = this.makeDir(current, 493);
          current.children.set(part, dir);
          current.nlink++;
          child = dir;
        }
        if (child.type !== "dir") return;
        current = child;
      }
    }
    addFileContent(path, content) {
      this.setFileData(path, new TextEncoder().encode(content), 420);
    }
    setFileData(path, data, mode) {
      const parts = path.split("/").filter((p) => p.length > 0);
      const name = parts.pop();
      let dir = this.root;
      for (const part of parts) {
        const child = dir.children.get(part);
        if (!child || child.type !== "dir") return;
        dir = child;
      }
      const existing = dir.children.get(name);
      if (existing && existing.type === "file") {
        existing.data = data;
        existing.mode = S_IFREG | mode & 4095;
        existing.mtime = existing.ctime = this.now();
      } else {
        dir.children.set(name, this.makeFile(mode, data));
      }
    }
    addCharDev(path, devno, mode) {
      const parts = path.split("/").filter((p) => p.length > 0);
      const name = parts.pop();
      let dir = this.root;
      for (const part of parts) {
        const child = dir.children.get(part);
        if (!child || child.type !== "dir") return;
        dir = child;
      }
      dir.children.set(name, this.makeCharDev(devno, mode));
    }
    addSymlinkNode(path, target) {
      const parts = path.split("/").filter((p) => p.length > 0);
      const name = parts.pop();
      let dir = this.root;
      for (const part of parts) {
        const child = dir.children.get(part);
        if (!child || child.type !== "dir") return;
        dir = child;
      }
      dir.children.set(name, this.makeSymlink(target));
    }
    // ── Path resolution (follows Linux symlink rules) ───────────────────────
    /**
     * Normalize a path: resolve relative paths against cwd, handle . and ..
     * Returns array of path components (avoids join+split overhead in resolve).
     */
    normalizeParts(path) {
      if (path.charCodeAt(0) !== 47) {
        path = this.cwd + (this.cwd.length > 1 ? "/" : "") + path;
      }
      const result = [];
      let start = 0;
      for (let i = 0; i <= path.length; i++) {
        if (i === path.length || path.charCodeAt(i) === 47) {
          if (i > start) {
            const part = path.substring(start, i);
            if (part === "..") {
              result.pop();
            } else if (part !== ".") {
              result.push(part);
            }
          }
          start = i + 1;
        }
      }
      return result;
    }
    resolvePath(path) {
      const parts = this.normalizeParts(path);
      return parts.length === 0 ? "/" : "/" + parts.join("/");
    }
    /**
     * Resolve a path to an inode, following symlinks per Linux rules.
     * Uses pre-split path components to avoid redundant string splitting.
     */
    resolve(path, followFinal = true, depth = 0) {
      if (depth > MAX_SYMLINK_DEPTH) return null;
      const parts = this.normalizeParts(path);
      return this.resolveFromParts(parts, followFinal, depth);
    }
    resolveFromParts(parts, followFinal, depth) {
      if (parts.length === 0) return this.root;
      let current = this.root;
      for (let i = 0; i < parts.length; i++) {
        if (current.type !== "dir") return null;
        const child = current.children.get(parts[i]);
        if (!child) return null;
        const isLast = i === parts.length - 1;
        if (child.type === "symlink" && (followFinal || !isLast)) {
          const target = child.target;
          const remaining = parts.slice(i + 1);
          const targetParts = this.normalizeParts(target);
          if (remaining.length > 0) targetParts.push(...remaining);
          return this.resolveFromParts(targetParts, followFinal, depth + 1);
        }
        current = child;
      }
      return current;
    }
    /** Resolve parent directory of path. Returns [parentDirInode, childName] or null. */
    resolveParent(path, depth = 0) {
      if (depth > MAX_SYMLINK_DEPTH) return null;
      const resolved = this.resolvePath(path);
      const parts = resolved.split("/").filter((p) => p.length > 0);
      if (parts.length === 0) return null;
      const name = parts.pop();
      let dir = this.root;
      for (const part of parts) {
        const child = dir.children.get(part);
        if (!child) return null;
        if (child.type === "symlink") {
          const target = child.target;
          const remaining = parts.slice(parts.indexOf(part) + 1).join("/");
          const newPath = remaining ? target + "/" + remaining + "/" + name : target + "/" + name;
          return this.resolveParent(newPath, depth + 1);
        }
        if (child.type !== "dir") return null;
        dir = child;
      }
      return [dir, name];
    }
    // ── VFS interface implementation ────────────────────────────────────────
    open(path, flags, mode) {
      if (path.length > PATH_MAX) return -ENAMETOOLONG;
      for (const part of path.split("/")) {
        if (part.length > NAME_MAX) return -ENAMETOOLONG;
      }
      if (this.nextFd >= MAX_FDS2) return -EMFILE;
      const followSymlinks = !(flags & O_NOFOLLOW);
      let inode = this.resolve(path, followSymlinks);
      if (flags & O_NOFOLLOW && !inode) {
        const raw = this.resolve(path, false);
        if (raw && raw.type === "symlink") return -ELOOP;
      }
      if (inode && flags & O_CREAT && flags & O_EXCL) {
        return -EEXIST;
      }
      if (!inode && flags & O_CREAT) {
        const parent = this.resolveParent(path);
        if (!parent) return -ENOENT2;
        const [dir, name] = parent;
        const newFile = this.makeFile(mode & 4095);
        dir.children.set(name, newFile);
        dir.mtime = dir.ctime = this.now();
        inode = newFile;
      }
      if (!inode) return -ENOENT2;
      if (flags & O_DIRECTORY && inode.type !== "dir") return -ENOTDIR;
      if (inode.type === "dir") {
        if ((flags & O_ACCMODE) !== O_RDONLY) return -EISDIR;
        const fd2 = this.nextFd++;
        const entries = [".", "..", ...inode.children.keys()];
        this.openFds.set(fd2, { type: "dir", inode, entries, pos: 0, refcount: 1 });
        return fd2;
      }
      if (inode.type === "chardev") {
        const fd2 = this.nextFd++;
        this.openFds.set(fd2, { type: "chardev", inode, refcount: 1 });
        return fd2;
      }
      if (inode.type !== "file") return -ENOENT2;
      if (flags & O_TRUNC && (flags & O_ACCMODE) !== O_RDONLY) {
        inode.data = new Uint8Array(0);
        inode.mtime = inode.ctime = this.now();
      }
      const fd = this.nextFd++;
      const accessMode = flags & O_ACCMODE;
      const pos = flags & O_APPEND ? inode.data.length : 0;
      this.openFds.set(fd, { type: "file", inode, pos, flags, accessMode, refcount: 1 });
      return fd;
    }
    close(fd) {
      const ofd = this.openFds.get(fd);
      if (!ofd) return -EBADF2;
      this.openFds.delete(fd);
      ofd.refcount--;
      return 0;
    }
    hasFd(fd) {
      return this.openFds.has(fd);
    }
    dup(fd) {
      const ofd = this.openFds.get(fd);
      if (!ofd) return -EBADF2;
      if (this.nextFd >= MAX_FDS2) return -EMFILE;
      const newFd = this.nextFd++;
      ofd.refcount++;
      this.openFds.set(newFd, ofd);
      return newFd;
    }
    read(fd, buf, count, memory2) {
      const ofd = this.openFds.get(fd);
      if (!ofd) return -EBADF2;
      if (ofd.type === "chardev") {
        const bytes2 = new Uint8Array(memory2.buffer);
        switch (ofd.inode.devno) {
          case 1:
            return 0;
          // /dev/null → EOF
          case 2:
            bytes2.fill(0, buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
            return count;
          case 3:
            return 0;
          // /dev/tty → EOF
          case 4:
            for (let i = 0; i < count; i++)
              bytes2[buf + GUEST_MEM_BASE + i] = Math.random() * 256 | 0;
            return count;
          default:
            return 0;
        }
      }
      if (ofd.type === "dir") return -EISDIR;
      if (ofd.type !== "file") return -EBADF2;
      if (ofd.accessMode === O_WRONLY) return -EBADF2;
      const available = ofd.inode.data.length - ofd.pos;
      const toRead = Math.min(count, Math.max(0, available));
      if (toRead <= 0) return 0;
      const bytes = new Uint8Array(memory2.buffer);
      bytes.set(ofd.inode.data.subarray(ofd.pos, ofd.pos + toRead), buf + GUEST_MEM_BASE);
      ofd.pos += toRead;
      ofd.inode.atime = this.now();
      return toRead;
    }
    write(fd, buf, count, memory2) {
      const ofd = this.openFds.get(fd);
      if (!ofd) return -EBADF2;
      if (ofd.type === "chardev") return count;
      if (ofd.type === "dir") return -EBADF2;
      if (ofd.type !== "file") return -EBADF2;
      if (ofd.accessMode === O_RDONLY) return -EBADF2;
      if (count === 0) return 0;
      if (ofd.flags & O_APPEND) {
        ofd.pos = ofd.inode.data.length;
      }
      const src = new Uint8Array(memory2.buffer, buf + GUEST_MEM_BASE, count);
      const endPos = ofd.pos + count;
      if (endPos > ofd.inode.data.length) {
        const newData = new Uint8Array(endPos);
        newData.set(ofd.inode.data);
        ofd.inode.data = newData;
      }
      ofd.inode.data.set(src, ofd.pos);
      ofd.pos = endPos;
      ofd.inode.mtime = ofd.inode.ctime = this.now();
      return count;
    }
    lseek(fd, offset, whence) {
      const ofd = this.openFds.get(fd);
      if (!ofd) return BigInt(-EBADF2);
      if (ofd.type === "chardev") return BigInt(-ESPIPE2);
      if (ofd.type !== "file") return BigInt(-EBADF2);
      let newPos;
      const off = Number(offset);
      switch (whence) {
        case 0:
          newPos = off;
          break;
        // SEEK_SET
        case 1:
          newPos = ofd.pos + off;
          break;
        // SEEK_CUR
        case 2:
          newPos = ofd.inode.data.length + off;
          break;
        // SEEK_END
        default:
          return BigInt(-EINVAL2);
      }
      if (newPos < 0) return BigInt(-EINVAL2);
      ofd.pos = newPos;
      return BigInt(newPos);
    }
    fchmod(fd, mode) {
      const ofd = this.openFds.get(fd);
      if (!ofd) return -EBADF2;
      const inode = ofd.type === "file" ? ofd.inode : ofd.type === "dir" ? ofd.inode : null;
      if (!inode) return -EBADF2;
      inode.mode = inode.mode & S_IFMT | mode & 4095;
      inode.ctime = this.now();
      return 0;
    }
    fchown(fd, uid, gid) {
      const ofd = this.openFds.get(fd);
      if (!ofd) return -EBADF2;
      const inode = ofd.type === "file" ? ofd.inode : ofd.type === "dir" ? ofd.inode : null;
      if (!inode) return -EBADF2;
      if (uid !== -1 && uid !== 4294967295) inode.uid = uid;
      if (gid !== -1 && gid !== 4294967295) inode.gid = gid;
      inode.ctime = this.now();
      return 0;
    }
    ftruncate(fd, length) {
      const ofd = this.openFds.get(fd);
      if (!ofd || ofd.type !== "file") return -EBADF2;
      if (ofd.accessMode === O_RDONLY) return -EBADF2;
      if (length < 0) return -EINVAL2;
      const inode = ofd.inode;
      if (length < inode.data.length) {
        inode.data = inode.data.slice(0, length);
      } else if (length > inode.data.length) {
        const newData = new Uint8Array(length);
        newData.set(inode.data);
        inode.data = newData;
      }
      inode.mtime = inode.ctime = this.now();
      return 0;
    }
    fstat(fd, statBuf, memory2) {
      const ofd = this.openFds.get(fd);
      if (!ofd) return -EBADF2;
      const inode = ofd.type === "file" ? ofd.inode : ofd.type === "dir" ? ofd.inode : ofd.type === "chardev" ? ofd.inode : null;
      if (!inode) return -EBADF2;
      this.writeStat(memory2, statBuf, inode);
      return 0;
    }
    fstatat(dirfd, path, statBuf, flags, memory2) {
      const AT_SYMLINK_NOFOLLOW = 256;
      const AT_EMPTY_PATH = 4096;
      if (flags & AT_EMPTY_PATH && path === "") {
        return this.fstat(dirfd, statBuf, memory2);
      }
      const follow = !(flags & AT_SYMLINK_NOFOLLOW);
      const inode = this.resolve(path, follow);
      if (!inode) return -ENOENT2;
      this.writeStat(memory2, statBuf, inode);
      return 0;
    }
    faccessat(dirfd, path, mode, flags) {
      const inode = this.resolve(path);
      if (!inode) return -ENOENT2;
      return 0;
    }
    getcwd() {
      return this.cwd;
    }
    getdents64(fd, buf, count, memory2) {
      const ofd = this.openFds.get(fd);
      if (!ofd || ofd.type !== "dir") return -EBADF2;
      const bytes = new Uint8Array(memory2.buffer);
      const base = buf + GUEST_MEM_BASE;
      let offset = 0;
      while (ofd.pos < ofd.entries.length) {
        const name = ofd.entries[ofd.pos];
        let d_type = 4;
        let d_ino = ofd.inode.ino;
        if (name === ".") {
          d_ino = ofd.inode.ino;
        } else if (name === "..") {
          d_ino = ofd.inode.parent ? ofd.inode.parent.ino : ofd.inode.ino;
        } else {
          const child = ofd.inode.children.get(name);
          if (!child) {
            ofd.pos++;
            continue;
          }
          d_ino = child.ino;
          if (child.type === "file") d_type = 8;
          else if (child.type === "dir") d_type = 4;
          else if (child.type === "symlink") d_type = 10;
          else if (child.type === "chardev") d_type = 2;
        }
        const nameBytes = new TextEncoder().encode(name);
        const reclen = 19 + nameBytes.length + 1 + 7 & ~7;
        if (offset + reclen > count) break;
        const pos = base + offset;
        memory2.setBigUint64(pos, BigInt(d_ino), true);
        memory2.setBigUint64(pos + 8, BigInt(ofd.pos + 1), true);
        memory2.setUint16(pos + 16, reclen, true);
        bytes[pos + 18] = d_type;
        bytes.set(nameBytes, pos + 19);
        bytes[pos + 19 + nameBytes.length] = 0;
        for (let i = 19 + nameBytes.length + 1; i < reclen; i++) bytes[pos + i] = 0;
        offset += reclen;
        ofd.pos++;
      }
      return offset;
    }
    readlinkat(path) {
      const inode = this.resolve(path, false);
      if (!inode || inode.type !== "symlink") return null;
      return inode.target;
    }
    unlinkat(path, flags) {
      const AT_REMOVEDIR = 512;
      const parent = this.resolveParent(path);
      if (!parent) return -ENOENT2;
      const [dir, name] = parent;
      if (name === "." || name === "..") return -EINVAL2;
      const child = dir.children.get(name);
      if (!child) return -ENOENT2;
      if (flags & AT_REMOVEDIR) {
        if (child.type !== "dir") return -ENOTDIR;
        if (child.children.size > 0) return -ENOTEMPTY;
        dir.nlink--;
      } else {
        if (child.type === "dir") return -EISDIR;
        if (child.type === "file") child.nlink--;
      }
      dir.children.delete(name);
      dir.mtime = dir.ctime = this.now();
      return 0;
    }
    mkdirat(path, mode) {
      const parent = this.resolveParent(path);
      if (!parent) return -ENOENT2;
      const [dir, name] = parent;
      if (dir.children.has(name)) return -EEXIST;
      const newDir = this.makeDir(dir, mode & 4095);
      dir.children.set(name, newDir);
      dir.nlink++;
      dir.mtime = dir.ctime = this.now();
      return 0;
    }
    symlinkat(target, linkpath) {
      const parent = this.resolveParent(linkpath);
      if (!parent) return -ENOENT2;
      const [dir, name] = parent;
      if (dir.children.has(name)) return -EEXIST;
      dir.children.set(name, this.makeSymlink(target));
      dir.mtime = dir.ctime = this.now();
      return 0;
    }
    linkat(oldpath, newpath) {
      const inode = this.resolve(oldpath);
      if (!inode) return -ENOENT2;
      if (inode.type === "dir") return -EPERM2;
      const parent = this.resolveParent(newpath);
      if (!parent) return -ENOENT2;
      const [dir, name] = parent;
      if (dir.children.has(name)) return -EEXIST;
      dir.children.set(name, inode);
      inode.nlink++;
      inode.ctime = this.now();
      dir.mtime = dir.ctime = this.now();
      return 0;
    }
    rename(oldpath, newpath) {
      const oldParent = this.resolveParent(oldpath);
      if (!oldParent) return -ENOENT2;
      const [oldDir, oldName] = oldParent;
      const srcNode = oldDir.children.get(oldName);
      if (!srcNode) return -ENOENT2;
      const newParent = this.resolveParent(newpath);
      if (!newParent) return -ENOENT2;
      const [newDir, newName] = newParent;
      const dstNode = newDir.children.get(newName);
      if (dstNode) {
        if (srcNode.type === "dir" && dstNode.type !== "dir") return -ENOTDIR;
        if (srcNode.type !== "dir" && dstNode.type === "dir") return -EISDIR;
        if (dstNode.type === "dir" && dstNode.children.size > 0) return -ENOTEMPTY;
      }
      oldDir.children.delete(oldName);
      newDir.children.set(newName, srcNode);
      if (srcNode.type === "dir") {
        srcNode.parent = newDir;
        if (oldDir !== newDir) {
          oldDir.nlink--;
          newDir.nlink++;
        }
      }
      oldDir.mtime = oldDir.ctime = this.now();
      newDir.mtime = newDir.ctime = this.now();
      return 0;
    }
    chdir(path) {
      if (path.length > PATH_MAX) return -ENAMETOOLONG;
      for (const part of path.split("/")) {
        if (part.length > NAME_MAX) return -ENAMETOOLONG;
      }
      const inode = this.resolve(path);
      if (!inode) return -ENOENT2;
      if (inode.type !== "dir") return -ENOTDIR;
      this.cwd = this.resolvePath(path);
      this.cwdInode = inode;
      return 0;
    }
    /** Copy data between two open files entirely in JS (no guest memory needed). */
    copyFileRange(fdIn, fdOut, len) {
      const ofdIn = this.openFds.get(fdIn);
      const ofdOut = this.openFds.get(fdOut);
      if (!ofdIn || ofdIn.type !== "file") return -EBADF2;
      if (!ofdOut || ofdOut.type !== "file") return -EBADF2;
      if (ofdIn.accessMode === 1) return -EBADF2;
      if (ofdOut.accessMode === 0) return -EBADF2;
      const available = ofdIn.inode.data.length - ofdIn.pos;
      const toCopy = Math.min(len, Math.max(0, available));
      if (toCopy <= 0) return 0;
      const endPos = ofdOut.pos + toCopy;
      if (endPos > ofdOut.inode.data.length) {
        const newData = new Uint8Array(endPos);
        newData.set(ofdOut.inode.data);
        ofdOut.inode.data = newData;
      }
      ofdOut.inode.data.set(ofdIn.inode.data.subarray(ofdIn.pos, ofdIn.pos + toCopy), ofdOut.pos);
      ofdIn.pos += toCopy;
      ofdOut.pos = endPos;
      ofdOut.inode.mtime = ofdOut.inode.ctime = this.now();
      return toCopy;
    }
    // ── Snapshot/restore for fork ───────────────────────────────────────────
    snapshotFds() {
      const ofdMap = /* @__PURE__ */ new Map();
      const snap = /* @__PURE__ */ new Map();
      for (const [fd, ofd] of this.openFds) {
        if (!ofdMap.has(ofd)) {
          if (ofd.type === "file") {
            ofdMap.set(ofd, { ...ofd });
          } else if (ofd.type === "dir") {
            ofdMap.set(ofd, { ...ofd, entries: [...ofd.entries] });
          } else {
            ofdMap.set(ofd, { ...ofd });
          }
        }
        snap.set(fd, ofdMap.get(ofd));
      }
      return snap;
    }
    restoreFds(snap) {
      this.openFds = new Map(snap);
      this.nextFd = 3;
      for (const fd of this.openFds.keys()) {
        if (fd >= this.nextFd) this.nextFd = fd + 1;
      }
    }
    closeAllExcept(...keep) {
      const keepSet = new Set(keep);
      for (const [fd, ofd] of [...this.openFds.entries()]) {
        if (!keepSet.has(fd)) {
          this.openFds.delete(fd);
          ofd.refcount--;
        }
      }
    }
    // ── Stat writing ────────────────────────────────────────────────────────
    writeStat(memory2, guestAddr, inode) {
      const b = guestAddr + GUEST_MEM_BASE;
      const rdev = inode.type === "chardev" ? inode.devno : 0;
      const size = inode.type === "file" ? inode.data.length : inode.type === "dir" ? 4096 : inode.type === "symlink" ? inode.target.length : 0;
      const blocks = Math.ceil(size / 512);
      const at = BigInt(inode.atime), mt = BigInt(inode.mtime), ct = BigInt(inode.ctime);
      memory2.setBigUint64(b + 0, 1n, true);
      memory2.setBigUint64(b + 8, BigInt(inode.ino), true);
      memory2.setUint32(b + 16, inode.mode, true);
      memory2.setUint32(b + 20, inode.nlink, true);
      memory2.setUint32(b + 24, inode.uid, true);
      memory2.setUint32(b + 28, inode.gid, true);
      memory2.setBigUint64(b + 32, BigInt(rdev), true);
      memory2.setBigUint64(b + 40, 0n, true);
      memory2.setBigInt64(b + 48, BigInt(size), true);
      memory2.setInt32(b + 56, 4096, true);
      memory2.setInt32(b + 60, 0, true);
      memory2.setBigInt64(b + 64, BigInt(blocks), true);
      memory2.setBigInt64(b + 72, at, true);
      memory2.setBigUint64(b + 80, 0n, true);
      memory2.setBigInt64(b + 88, mt, true);
      memory2.setBigUint64(b + 96, 0n, true);
      memory2.setBigInt64(b + 104, ct, true);
      memory2.setBigUint64(b + 112, 0n, true);
      memory2.setUint32(b + 120, 0, true);
      memory2.setUint32(b + 124, 0, true);
    }
  };

  // src/elf.ts
  var ELF_MAGIC = [127, 69, 76, 70];
  var ELFCLASS64 = 2;
  var ELFDATA2LSB = 1;
  var EM_AARCH64 = 183;
  var PT_LOAD = 1;
  var PT_DYNAMIC = 2;
  var PT_PHDR = 6;
  var ET_DYN = 3;
  var PIE_LOAD_BASE = 0x400000n;
  function loadElf(elfBytes2, memory2) {
    const dv = new DataView(elfBytes2.buffer, elfBytes2.byteOffset, elfBytes2.byteLength);
    for (let i = 0; i < 4; i++) {
      if (elfBytes2[i] !== ELF_MAGIC[i]) {
        throw new Error("Not an ELF file");
      }
    }
    if (elfBytes2[4] !== ELFCLASS64) throw new Error("Not a 64-bit ELF");
    if (elfBytes2[5] !== ELFDATA2LSB) throw new Error("Not little-endian");
    const eMachine = dv.getUint16(18, true);
    if (eMachine !== EM_AARCH64) throw new Error(`Not aarch64 (e_machine=0x${eMachine.toString(16)})`);
    const eType = dv.getUint16(16, true);
    const isPie = eType === ET_DYN;
    const rawEntry = dv.getBigUint64(24, true);
    const phoff = Number(dv.getBigUint64(32, true));
    const phentSize = dv.getUint16(54, true);
    const phnum = dv.getUint16(56, true);
    let lowestVaddr = 0xFFFFFFFFFFFFFFFFn;
    for (let i = 0; i < phnum; i++) {
      const phBase = phoff + i * phentSize;
      const pType = dv.getUint32(phBase, true);
      if (pType === PT_LOAD) {
        const pVaddr = dv.getBigUint64(phBase + 16, true);
        if (pVaddr < lowestVaddr) lowestVaddr = pVaddr;
      }
    }
    const loadBase = isPie ? PIE_LOAD_BASE - lowestVaddr : 0n;
    const entry = rawEntry + loadBase;
    let brkStart = 0n;
    let phdrAddr = 0n;
    let dynOffset = 0;
    let dynSize = 0;
    for (let i = 0; i < phnum; i++) {
      const phBase = phoff + i * phentSize;
      const pType = dv.getUint32(phBase, true);
      const pOffset = Number(dv.getBigUint64(phBase + 8, true));
      const pVaddr = dv.getBigUint64(phBase + 16, true) + loadBase;
      const pFilesz = Number(dv.getBigUint64(phBase + 32, true));
      const pMemsz = Number(dv.getBigUint64(phBase + 40, true));
      if (pType === PT_PHDR) {
        phdrAddr = pVaddr;
      }
      if (pType === PT_DYNAMIC) {
        dynOffset = pOffset;
        dynSize = pFilesz;
      }
      if (pType !== PT_LOAD) continue;
      const guestAddr = Number(pVaddr);
      const wasmAddr = guestAddr + GUEST_MEM_BASE;
      const needed = wasmAddr + pMemsz;
      if (needed > memory2.buffer.byteLength) {
        const pagesNeeded = Math.ceil(needed / 65536);
        const currentPages = memory2.buffer.byteLength / 65536;
        if (pagesNeeded > currentPages) {
          memory2.grow(pagesNeeded - currentPages);
        }
      }
      const freshBytes = new Uint8Array(memory2.buffer);
      if (pFilesz > 0) {
        freshBytes.set(elfBytes2.subarray(pOffset, pOffset + pFilesz), wasmAddr);
      }
      if (pMemsz > pFilesz) {
        freshBytes.fill(0, wasmAddr + pFilesz, wasmAddr + pMemsz);
      }
      const segEnd = pVaddr + BigInt(pMemsz);
      if (segEnd > brkStart) {
        brkStart = segEnd;
      }
    }
    if (phdrAddr === 0n) {
      phdrAddr = loadBase + BigInt(phoff);
    }
    brkStart = brkStart + 0xFFFn & ~0xFFFn;
    return {
      entry,
      phdr: phdrAddr,
      phentSize,
      phnum,
      brkStart,
      loadBase
    };
  }
  var STACK_TOP = 8388608;
  var AT_NULL = 0;
  var AT_PHDR = 3;
  var AT_PHENT = 4;
  var AT_PHNUM = 5;
  var AT_PAGESZ = 6;
  var AT_ENTRY = 9;
  var AT_UID = 11;
  var AT_EUID = 12;
  var AT_GID = 13;
  var AT_EGID = 14;
  var AT_RANDOM = 25;
  function setupStack(memory2, elfInfo, argv, envp) {
    const mem = new DataView(memory2.buffer);
    const bytes = new Uint8Array(memory2.buffer);
    let sp = STACK_TOP;
    const stringAddrs = [];
    for (const arg of argv) {
      const encoded = new TextEncoder().encode(arg + "\0");
      sp -= encoded.length;
      bytes.set(encoded, sp + GUEST_MEM_BASE);
      stringAddrs.push(sp);
    }
    const envAddrs = [];
    for (const env of envp) {
      const encoded = new TextEncoder().encode(env + "\0");
      sp -= encoded.length;
      bytes.set(encoded, sp + GUEST_MEM_BASE);
      envAddrs.push(sp);
    }
    sp -= 16;
    const randomAddr = sp;
    for (let i = 0; i < 16; i++) {
      bytes[sp + GUEST_MEM_BASE + i] = Math.random() * 256 | 0;
    }
    sp = sp & ~15;
    const auxv = [
      [AT_PHDR, elfInfo.phdr],
      [AT_PHENT, BigInt(elfInfo.phentSize)],
      [AT_PHNUM, BigInt(elfInfo.phnum)],
      [AT_PAGESZ, 4096n],
      [AT_ENTRY, elfInfo.entry],
      [AT_UID, 0n],
      [AT_EUID, 0n],
      [AT_GID, 0n],
      [AT_EGID, 0n],
      [AT_RANDOM, BigInt(randomAddr)],
      [AT_NULL, 0n]
    ];
    const stackEntries = 8 + // argc
    (argv.length + 1) * 8 + // argv + NULL
    (envp.length + 1) * 8 + // envp + NULL
    auxv.length * 16;
    sp -= stackEntries;
    sp = sp & ~15;
    let pos = sp;
    mem.setBigUint64(pos + GUEST_MEM_BASE, BigInt(argv.length), true);
    pos += 8;
    for (let i = 0; i < argv.length; i++) {
      mem.setBigUint64(pos + GUEST_MEM_BASE, BigInt(stringAddrs[i]), true);
      pos += 8;
    }
    mem.setBigUint64(pos + GUEST_MEM_BASE, 0n, true);
    pos += 8;
    for (let i = 0; i < envp.length; i++) {
      mem.setBigUint64(pos + GUEST_MEM_BASE, BigInt(envAddrs[i]), true);
      pos += 8;
    }
    mem.setBigUint64(pos + GUEST_MEM_BASE, 0n, true);
    pos += 8;
    for (const [type, val] of auxv) {
      mem.setBigUint64(pos + GUEST_MEM_BASE, BigInt(type), true);
      pos += 8;
      mem.setBigUint64(pos + GUEST_MEM_BASE, val, true);
      pos += 8;
    }
    return BigInt(sp);
  }

  // src/kernel-ipc.ts
  var BRIDGE_RESP_OFFSET = 606208;
  var BRIDGE_RESP_SLOT_SIZE = 16384;
  var PROC_TABLE_OFFSET = 0;
  var PROC_SLOT_SIZE = 64;
  var MAILBOX_OFFSET = 4096;
  var MAILBOX_SLOT_SIZE = 128;
  var PIPE_RING_OFFSET = 12288;
  var PIPE_RING_SIZE = 16384;
  var STDIN_RING_OFFSET = 536576;
  var STDIN_RING_SIZE = 8192;
  var PROC_EMPTY = 0;
  var PROC_RUNNING = 1;
  var PROC_ZOMBIE = 2;
  var PROC_PID = 0;
  var PROC_PPID = 1;
  var PROC_STATE = 3;
  var PROC_EXIT_CODE = 4;
  var PROC_WAIT_FUTEX = 5;
  var PROC_SIGNAL = 6;
  var SIG_INT = 1 << 2;
  var SIG_PIPE = 1 << 13;
  var SIG_CHLD = 1 << 17;
  var MBOX_READY = 0;
  var MBOX_VALUE0 = 2;
  var PIPE_WRITE_POS = 0;
  var PIPE_READ_POS = 1;
  var PIPE_WRITER_CLOSED = 2;
  var PIPE_READER_CLOSED = 3;
  var PIPE_HEADER_SIZE = 16;
  var PIPE_DATA_SIZE = PIPE_RING_SIZE - PIPE_HEADER_SIZE;
  var STDIN_DATA_SIZE = STDIN_RING_SIZE - PIPE_HEADER_SIZE;
  function procSlotOffset(slot) {
    return PROC_TABLE_OFFSET + slot * PROC_SLOT_SIZE;
  }
  function mailboxSlotOffset(slot) {
    return MAILBOX_OFFSET + slot * MAILBOX_SLOT_SIZE;
  }
  function pipeRingOffset(pipeIndex) {
    return PIPE_RING_OFFSET + pipeIndex * PIPE_RING_SIZE;
  }
  function stdinRingOffset(stdinIndex) {
    return STDIN_RING_OFFSET + stdinIndex * STDIN_RING_SIZE;
  }

  // src/pipe-ring.ts
  var PipeRing = class {
    i32;
    data;
    dataSize;
    baseOffset;
    /**
     * @param sab   The kernel SharedArrayBuffer
     * @param offset  Byte offset of this ring's header within the SAB
     * @param totalSize  Total size of this ring (header + data)
     */
    constructor(sab, offset, totalSize) {
      this.baseOffset = offset;
      this.dataSize = totalSize - PIPE_HEADER_SIZE;
      this.i32 = new Int32Array(sab, offset, 4);
      this.data = new Uint8Array(sab, offset + PIPE_HEADER_SIZE, this.dataSize);
    }
    /** Reset the ring to empty state with the given number of open reader/writer ends. */
    reset(writers = 1, readers = 1) {
      Atomics.store(this.i32, PIPE_WRITE_POS, 0);
      Atomics.store(this.i32, PIPE_READ_POS, 0);
      Atomics.store(this.i32, PIPE_WRITER_CLOSED, writers);
      Atomics.store(this.i32, PIPE_READER_CLOSED, readers);
    }
    /** Add a writer reference (e.g., after fork inherits the write end). */
    addWriter() {
      Atomics.add(this.i32, PIPE_WRITER_CLOSED, 1);
    }
    /** Add a reader reference (e.g., after fork inherits the read end). */
    addReader() {
      Atomics.add(this.i32, PIPE_READER_CLOSED, 1);
    }
    // ── Writer API ──────────────────────────────────────────────────────────
    /**
     * Write data into the ring. Returns the number of bytes written.
     * Non-blocking: returns 0 if the ring is full.
     * Returns -1 (EPIPE) if the reader has closed.
     */
    write(src) {
      if (Atomics.load(this.i32, PIPE_READER_CLOSED) === 0) return -1;
      const wp = Atomics.load(this.i32, PIPE_WRITE_POS);
      const rp = Atomics.load(this.i32, PIPE_READ_POS);
      const used = wp - rp;
      const available = this.dataSize - 1 - used;
      if (available <= 0) return 0;
      const toWrite = Math.min(src.length, available);
      const ringPos = wp % this.dataSize;
      const firstChunk = Math.min(toWrite, this.dataSize - ringPos);
      this.data.set(src.subarray(0, firstChunk), ringPos);
      if (toWrite > firstChunk) {
        this.data.set(src.subarray(firstChunk, toWrite), 0);
      }
      Atomics.store(this.i32, PIPE_WRITE_POS, wp + toWrite);
      Atomics.notify(this.i32, PIPE_WRITE_POS);
      return toWrite;
    }
    /** Decrement writer refcount. Last close triggers EOF for readers. */
    closeWriter() {
      const remaining = Atomics.sub(this.i32, PIPE_WRITER_CLOSED, 1) - 1;
      if (remaining <= 0) {
        Atomics.store(this.i32, PIPE_WRITER_CLOSED, 0);
        Atomics.notify(this.i32, PIPE_WRITE_POS);
      }
    }
    /** Check if all read ends have been closed (broken pipe). */
    get readerClosed() {
      return Atomics.load(this.i32, PIPE_READER_CLOSED) === 0;
    }
    // ── Reader API ──────────────────────────────────────────────────────────
    /**
     * Read data from the ring into dst. Returns the number of bytes read.
     * Returns 0 for EOF (writer closed and ring empty).
     * Blocks via Atomics.wait() if the ring is empty and writer is still open.
     *
     * IMPORTANT: Only call from a Worker thread (Atomics.wait blocks the thread).
     */
    read(dst) {
      while (true) {
        const wp = Atomics.load(this.i32, PIPE_WRITE_POS);
        const rp = Atomics.load(this.i32, PIPE_READ_POS);
        const available = wp - rp;
        if (available > 0) {
          const toRead = Math.min(dst.length, available);
          const ringPos = rp % this.dataSize;
          const firstChunk = Math.min(toRead, this.dataSize - ringPos);
          dst.set(this.data.subarray(ringPos, ringPos + firstChunk));
          if (toRead > firstChunk) {
            dst.set(this.data.subarray(0, toRead - firstChunk), firstChunk);
          }
          Atomics.store(this.i32, PIPE_READ_POS, rp + toRead);
          Atomics.notify(this.i32, PIPE_READ_POS);
          return toRead;
        }
        if (Atomics.load(this.i32, PIPE_WRITER_CLOSED) === 0) {
          return 0;
        }
        Atomics.wait(this.i32, PIPE_WRITE_POS, wp);
      }
    }
    /**
     * Non-blocking read. Returns 0 if empty (not EOF — caller must check writerClosed).
     * Safe to call from the main thread.
     */
    tryRead(dst) {
      const wp = Atomics.load(this.i32, PIPE_WRITE_POS);
      const rp = Atomics.load(this.i32, PIPE_READ_POS);
      const available = wp - rp;
      if (available <= 0) return 0;
      const toRead = Math.min(dst.length, available);
      const ringPos = rp % this.dataSize;
      const firstChunk = Math.min(toRead, this.dataSize - ringPos);
      dst.set(this.data.subarray(ringPos, ringPos + firstChunk));
      if (toRead > firstChunk) {
        dst.set(this.data.subarray(0, toRead - firstChunk), firstChunk);
      }
      Atomics.store(this.i32, PIPE_READ_POS, rp + toRead);
      Atomics.notify(this.i32, PIPE_READ_POS);
      return toRead;
    }
    /** Decrement reader refcount. Last close signals broken pipe to writers. */
    closeReader() {
      const remaining = Atomics.sub(this.i32, PIPE_READER_CLOSED, 1) - 1;
      if (remaining <= 0) {
        Atomics.store(this.i32, PIPE_READER_CLOSED, 0);
        Atomics.notify(this.i32, PIPE_READ_POS);
      }
    }
    /** Check if all write ends have been closed (EOF for readers). */
    get writerClosed() {
      return Atomics.load(this.i32, PIPE_WRITER_CLOSED) === 0;
    }
    /** Number of bytes currently in the ring. */
    get available() {
      return Atomics.load(this.i32, PIPE_WRITE_POS) - Atomics.load(this.i32, PIPE_READ_POS);
    }
    /** Whether the ring is empty. */
    get empty() {
      return this.available === 0;
    }
  };

  // src/worker.ts
  console.log("[worker] Script loaded");
  var memory;
  var cpu;
  var syscalls;
  var vfs;
  var regionCache = /* @__PURE__ */ new Map();
  var elfBytes;
  var kernelSAB;
  var procTable;
  var mySlot;
  var myPid;
  var myMailbox;
  var kernelPipes = /* @__PURE__ */ new Map();
  var nextLocalFd = 100;
  var stdinRing = null;
  self.onmessage = (e) => {
    const msg = e.data;
    switch (msg.type) {
      case "init":
        init(msg.config);
        break;
      case "fork_result":
        break;
      case "stdin_data":
        break;
      case "signal": {
        break;
      }
    }
  };
  function init(config) {
    myPid = config.pid;
    mySlot = config.slot;
    kernelSAB = config.kernelSAB;
    elfBytes = config.elfBytes;
    procTable = new Int32Array(kernelSAB);
    myMailbox = new Int32Array(kernelSAB, mailboxSlotOffset(mySlot), MAILBOX_SLOT_SIZE / 4);
    const initialPages = 1024;
    memory = new WebAssembly.Memory({ initial: initialPages });
    vfs = new InMemoryVFS();
    vfs.loadBinary(elfBytes);
    let pc;
    if (config.memoryBuffer) {
      const parentBytes = new Uint8Array(config.memoryBuffer);
      const myBytes = new Uint8Array(memory.buffer);
      if (parentBytes.length > myBytes.length) {
        const pagesNeeded = Math.ceil(parentBytes.length / 65536);
        const currentPages = memory.buffer.byteLength / 65536;
        if (pagesNeeded > currentPages) {
          memory.grow(pagesNeeded - currentPages);
        }
      }
      new Uint8Array(memory.buffer).set(parentBytes);
      cpu = new CpuState(memory);
      cpu.setReg(0, 0n);
      syscalls = new SyscallHandler(0n, {
        writeStdout: (d) => postWrite(1, d),
        writeStderr: (d) => postWrite(2, d),
        readStdin: (count) => readStdin(count)
      }, vfs);
      syscalls.bridgeAvailable = true;
      if (config.syscallState) {
        syscalls.restoreState(config.syscallState);
      }
      if (config.kernelPipesSnapshot) {
        for (const [fd, info] of config.kernelPipesSnapshot) {
          kernelPipes.set(fd, { ...info });
          const ring = new PipeRing(kernelSAB, pipeRingOffset(info.pipeIndex), PIPE_RING_SIZE);
          if (info.mode === "write") ring.addWriter();
          else ring.addReader();
        }
        for (const fd of kernelPipes.keys()) {
          if (fd >= nextLocalFd) nextLocalFd = fd + 1;
        }
      }
      pc = cpu.getPC() + 4n;
    } else {
      cpu = new CpuState(memory);
      const elfInfo = loadElf(elfBytes, memory);
      const sp = setupStack(memory, elfInfo, config.argv, config.envp);
      cpu.setSP(sp);
      const view = new DataView(memory.buffer);
      view.setBigUint64(280, BigInt(9437184), true);
      syscalls = new SyscallHandler(elfInfo.brkStart, {
        writeStdout: (d) => postWrite(1, d),
        writeStderr: (d) => postWrite(2, d),
        readStdin: (count) => readStdin(count)
      }, vfs);
      syscalls.bridgeAvailable = true;
      pc = elfInfo.entry;
    }
    const stdinRingIdx = Atomics.load(procTable, procSlotOffset(mySlot) / 4 + 7);
    if (stdinRingIdx >= 0) {
      stdinRing = new PipeRing(kernelSAB, stdinRingOffset(stdinRingIdx), STDIN_RING_SIZE);
    }
    console.log(`[worker ${myPid}] Starting at pc=0x${pc.toString(16)}`);
    try {
      runLoop(pc);
    } catch (e) {
      if (e.message === "SIGINT") {
        return;
      }
      console.error(`[worker ${myPid}] CRASH:`, e.message, e.stack);
      postExit(139);
    }
  }
  function getRegion(pc) {
    let region = regionCache.get(pc);
    if (region) return region;
    const view = new DataView(memory.buffer);
    const block = detectBlock(view, pc);
    try {
      region = translateBlock(block, memory);
    } catch (e) {
      console.error(`[worker ${myPid}] CRASH at pc=0x${pc.toString(16)}:`, e.message);
      for (let i = 0; i < Math.min(block.count, 10); i++) {
        const addr = Number(block.startPC) + i * 4;
        const insn = view.getUint32(addr + 65536, true);
        console.error(`  0x${addr.toString(16)}: 0x${insn.toString(16).padStart(8, "0")}`);
      }
      throw e;
    }
    regionCache.set(pc, region);
    return region;
  }
  function runLoop(startPC) {
    let pc = startPC;
    let totalInsns = 0;
    let view = new DataView(memory.buffer);
    let lastSignalCheck = 0;
    let blockCount = 0;
    while (pc !== HALT_PC) {
      blockCount++;
      const region = getRegion(pc);
      pc = region.run();
      view = new DataView(memory.buffer);
      const insns = Number(view.getBigUint64(ITER_OFFSET, true));
      totalInsns += insns;
      view.setBigUint64(ITER_OFFSET, 0n, true);
      lastSignalCheck += insns;
      if (lastSignalCheck > 1e4) {
        lastSignalCheck = 0;
        const sigPending = Atomics.load(procTable, procSlotOffset(mySlot) / 4 + 6);
        if (sigPending & SIG_INT) {
          Atomics.and(procTable, procSlotOffset(mySlot) / 4 + 6, ~SIG_INT);
          postExit(130);
          return;
        }
      }
      if (pc === SVC_PC) {
        const svcPC = view.getBigUint64(PC_OFFSET, true);
        const nr = Number(cpu.getReg(8));
        if (nr === 260) {
          handleWait4(cpu, view);
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 59) {
          handlePipe2(cpu, view);
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 24 && kernelPipes.has(Number(cpu.getReg(0)))) {
          handleKernelPipeDup3(cpu);
          pc = svcPC + 4n;
          continue;
        }
        if ((nr === 63 || nr === 64 || nr === 66 || nr === 57) && kernelPipes.has(Number(cpu.getReg(0)))) {
          handleKernelPipeIO(nr, cpu, view);
          pc = svcPC + 4n;
          continue;
        }
        const result = syscalls.handle(cpu, memory);
        view = new DataView(memory.buffer);
        switch (result.status) {
          case "ok":
            pc = svcPC + 4n;
            break;
          case "exit":
            postExit(result.code);
            return;
          case "fork":
            pc = handleFork(result.resumePC);
            break;
          case "execve":
            pc = handleExec(result.argv, result.envp);
            break;
          case "blocked_read":
            handleBlockedRead();
            pc = svcPC;
            break;
          case "bridge":
            handleBridgeCall(result.method, result.args, result.apply);
            pc = svcPC + 4n;
            break;
          default:
            pc = svcPC + 4n;
            break;
        }
        continue;
      }
    }
    console.log(`[worker] Exited run loop: pc=0x${pc.toString(16)} blocks=${blockCount} insns=${totalInsns}`);
    postExit(0);
  }
  function handleFork(resumePC) {
    const snapSize = Math.min(
      Number(syscalls.getMmapBase()) + GUEST_MEM_BASE + 65536,
      memory.buffer.byteLength
    );
    const memoryBuffer = memory.buffer.slice(0, snapSize);
    const syscallState = syscalls.getState();
    cpu.setPC(resumePC);
    const kernelPipesSnapshot = [...kernelPipes.entries()];
    self.postMessage({
      type: "fork",
      memoryBuffer,
      syscallState,
      kernelPipesSnapshot,
      resumePC: resumePC.toString()
    }, [memoryBuffer]);
    Atomics.store(myMailbox, MBOX_READY, 0);
    Atomics.wait(myMailbox, MBOX_READY, 0);
    const childPid = Atomics.load(myMailbox, MBOX_VALUE0);
    cpu.setReg(0, BigInt(childPid));
    return resumePC;
  }
  function handleExec(argv, envp) {
    const memBytes = new Uint8Array(memory.buffer);
    memBytes.fill(0, GUEST_MEM_BASE, GUEST_MEM_BASE + 67108864);
    const elfInfo = loadElf(elfBytes, memory);
    const sp = setupStack(memory, elfInfo, argv, envp);
    cpu.reset();
    cpu.setSP(sp);
    const view = new DataView(memory.buffer);
    view.setBigUint64(280, BigInt(9437184), true);
    syscalls.resetForExec(elfInfo.brkStart);
    regionCache.clear();
    return elfInfo.entry;
  }
  function handleWait4(cpu2, view) {
    const targetPid = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    const wstatusAddr = Number(cpu2.getReg(1));
    const options = Number(cpu2.getReg(2));
    const WNOHANG = 1;
    while (true) {
      let foundZombie = -1;
      let hasLivingChild = false;
      for (let slot = 0; slot < 64; slot++) {
        const slotBase = procSlotOffset(slot) / 4;
        const childPid = Atomics.load(procTable, slotBase + PROC_PID);
        if (childPid === 0) continue;
        const ppid = Atomics.load(procTable, slotBase + PROC_PPID);
        if (ppid !== myPid) continue;
        if (targetPid > 0 && childPid !== targetPid) continue;
        const state = Atomics.load(procTable, slotBase + PROC_STATE);
        if (state === PROC_ZOMBIE) {
          foundZombie = slot;
          break;
        }
        if (state === PROC_RUNNING) {
          hasLivingChild = true;
        }
      }
      if (foundZombie >= 0) {
        const slotBase = procSlotOffset(foundZombie) / 4;
        const childPid = Atomics.load(procTable, slotBase + PROC_PID);
        const exitCode = Atomics.load(procTable, slotBase + PROC_EXIT_CODE);
        if (wstatusAddr !== 0) {
          view.setUint32(wstatusAddr + GUEST_MEM_BASE, (exitCode & 255) << 8, true);
        }
        Atomics.store(procTable, slotBase + PROC_STATE, PROC_EMPTY);
        Atomics.store(procTable, slotBase + PROC_PID, 0);
        cpu2.setReg(0, BigInt(childPid));
        return;
      }
      if (!hasLivingChild) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-10)));
        return;
      }
      if (options & WNOHANG) {
        cpu2.setReg(0, 0n);
        return;
      }
      const mySlotBase = procSlotOffset(mySlot) / 4;
      Atomics.store(procTable, mySlotBase + PROC_WAIT_FUTEX, 0);
      Atomics.wait(procTable, mySlotBase + PROC_WAIT_FUTEX, 0, 1e3);
      const sigPending = Atomics.load(procTable, mySlotBase + PROC_SIGNAL);
      if (sigPending & SIG_INT) {
        Atomics.and(procTable, mySlotBase + PROC_SIGNAL, ~SIG_INT);
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-4)));
        return;
      }
    }
  }
  function handleKernelPipeDup3(cpu2) {
    const oldFd = Number(cpu2.getReg(0));
    const newFd = Number(cpu2.getReg(1));
    const pipeInfo = kernelPipes.get(oldFd);
    const existing = kernelPipes.get(newFd);
    if (existing) {
      const ring = new PipeRing(kernelSAB, pipeRingOffset(existing.pipeIndex), PIPE_RING_SIZE);
      if (existing.mode === "write") ring.closeWriter();
      else ring.closeReader();
      kernelPipes.delete(newFd);
    }
    kernelPipes.set(newFd, { ...pipeInfo });
    const dupRing = new PipeRing(kernelSAB, pipeRingOffset(pipeInfo.pipeIndex), PIPE_RING_SIZE);
    if (pipeInfo.mode === "write") dupRing.addWriter();
    else dupRing.addReader();
    syscalls.registerFd(newFd, {
      type: pipeInfo.mode === "read" ? "pipe_read" : "pipe_write",
      pipe: { data: [], writeClosed: false, readClosed: false }
    });
    cpu2.setReg(0, BigInt(newFd));
  }
  function allocPipeSlot() {
    Atomics.store(myMailbox, MBOX_READY, 0);
    self.postMessage({ type: "pipe_alloc" });
    Atomics.wait(myMailbox, MBOX_READY, 0);
    return Atomics.load(myMailbox, MBOX_VALUE0);
  }
  function handlePipe2(cpu2, view) {
    const pipefdAddr = Number(cpu2.getReg(0));
    const pipeIndex = allocPipeSlot();
    const ring = new PipeRing(kernelSAB, pipeRingOffset(pipeIndex), PIPE_RING_SIZE);
    ring.reset();
    const readFd = nextLocalFd++;
    const writeFd = nextLocalFd++;
    kernelPipes.set(readFd, { pipeIndex, mode: "read" });
    kernelPipes.set(writeFd, { pipeIndex, mode: "write" });
    syscalls.registerFd(readFd, { type: "pipe_read", pipe: { data: [], writeClosed: false, readClosed: false } });
    syscalls.registerFd(writeFd, { type: "pipe_write", pipe: { data: [], writeClosed: false, readClosed: false } });
    view.setInt32(pipefdAddr + GUEST_MEM_BASE, readFd, true);
    view.setInt32(pipefdAddr + GUEST_MEM_BASE + 4, writeFd, true);
    cpu2.setReg(0, 0n);
  }
  function handleKernelPipeIO(nr, cpu2, view) {
    const fd = Number(cpu2.getReg(0));
    const pipeInfo = kernelPipes.get(fd);
    const ring = new PipeRing(kernelSAB, pipeRingOffset(pipeInfo.pipeIndex), PIPE_RING_SIZE);
    if (nr === 57) {
      if (pipeInfo.mode === "write") {
        ring.closeWriter();
      } else {
        ring.closeReader();
      }
      kernelPipes.delete(fd);
      cpu2.setReg(0, 0n);
      return;
    }
    if (nr === 63) {
      const buf = Number(cpu2.getReg(1));
      const count = Number(cpu2.getReg(2));
      const dst = new Uint8Array(count);
      const n = ring.read(dst);
      if (n > 0) {
        new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, n).set(dst.subarray(0, n));
      }
      cpu2.setReg(0, BigInt(n));
      return;
    }
    if (nr === 64) {
      const buf = Number(cpu2.getReg(1));
      const count = Number(cpu2.getReg(2));
      const src = new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, count);
      const n = ring.write(src);
      if (n === -1) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-32)));
        return;
      }
      cpu2.setReg(0, BigInt(n));
      return;
    }
    if (nr === 66) {
      const iovAddr = Number(cpu2.getReg(1));
      const iovcnt = Number(cpu2.getReg(2));
      let totalWritten = 0;
      const bytes = new Uint8Array(memory.buffer);
      for (let i = 0; i < iovcnt; i++) {
        const base = view.getBigUint64(iovAddr + i * 16 + GUEST_MEM_BASE, true);
        const len = view.getBigUint64(iovAddr + i * 16 + 8 + GUEST_MEM_BASE, true);
        if (len === 0n) continue;
        const src = bytes.subarray(Number(base) + GUEST_MEM_BASE, Number(base) + GUEST_MEM_BASE + Number(len));
        const n = ring.write(src);
        if (n === -1) {
          if (totalWritten === 0) {
            cpu2.setReg(0, BigInt.asUintN(64, BigInt(-32)));
            return;
          }
          break;
        }
        totalWritten += n;
        if (n < Number(len)) break;
      }
      cpu2.setReg(0, BigInt(totalWritten));
      return;
    }
  }
  function handleBlockedRead() {
    if (stdinRing) {
      const stdinRingIdx = Atomics.load(procTable, procSlotOffset(mySlot) / 4 + 7);
      const ringI32 = new Int32Array(kernelSAB, stdinRingOffset(stdinRingIdx), 4);
      while (stdinRing.empty && !stdinRing.writerClosed) {
        const wp = Atomics.load(ringI32, 0);
        Atomics.wait(ringI32, 0, wp);
        const sigPending = Atomics.load(procTable, procSlotOffset(mySlot) / 4 + 6);
        if (sigPending & SIG_INT) {
          Atomics.and(procTable, procSlotOffset(mySlot) / 4 + 6, ~SIG_INT);
          if (syscalls.hasSignalHandler(2)) {
            syscalls.raiseSignal(2);
            return;
          }
          postExit(130);
          throw new Error("SIGINT");
        }
      }
    }
  }
  function handleBridgeCall(method, args, apply) {
    self.postMessage({ type: "bridge_call", id: 0, method, args });
    Atomics.store(myMailbox, MBOX_READY, 0);
    while (Atomics.load(myMailbox, MBOX_READY) === 0) {
      Atomics.wait(myMailbox, MBOX_READY, 0);
      if (Atomics.load(myMailbox, MBOX_READY) === 0) {
        const sigSlot = procSlotOffset(mySlot) / 4 + 6;
        const sig = Atomics.load(procTable, sigSlot);
        if (sig & SIG_INT) {
          Atomics.and(procTable, sigSlot, ~SIG_INT);
          if (syscalls.hasSignalHandler(2)) {
            syscalls.raiseSignal(2);
            apply({ ret: -4 });
            return;
          }
          postExit(130);
          throw new Error("SIGINT");
        }
      }
    }
    const respOffset = BRIDGE_RESP_OFFSET + mySlot * BRIDGE_RESP_SLOT_SIZE;
    const respView = new DataView(kernelSAB, respOffset, BRIDGE_RESP_SLOT_SIZE);
    const ret = respView.getInt32(0, true);
    const dataLen = respView.getUint32(4, true);
    const resp = { ret };
    if (dataLen > 0) {
      const dataBytes = new Uint8Array(kernelSAB, respOffset + 8, dataLen);
      let b64 = "";
      for (let i = 0; i < dataLen; i++) b64 += String.fromCharCode(dataBytes[i]);
      resp.data = btoa(b64);
    }
    apply(resp);
  }
  function readStdin(count) {
    if (!stdinRing) return null;
    if (stdinRing.empty && !stdinRing.writerClosed) return null;
    const buf = new Uint8Array(count);
    const n = stdinRing.tryRead(buf);
    if (n === 0 && stdinRing.writerClosed) {
      return new Uint8Array(0);
    }
    if (n === 0) return null;
    return buf.subarray(0, n);
  }
  function postWrite(fd, data) {
    self.postMessage({ type: "write", fd, data: data.slice() });
  }
  function postExit(code) {
    for (const [fd, info] of kernelPipes) {
      const ring = new PipeRing(kernelSAB, pipeRingOffset(info.pipeIndex), PIPE_RING_SIZE);
      if (info.mode === "write") ring.closeWriter();
      else ring.closeReader();
    }
    kernelPipes.clear();
    self.postMessage({ type: "exit", pid: myPid, code });
  }
})();
