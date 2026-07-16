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
  var V_OFFSET = 512;
  var STATE_SIZE = 1024;
  var GUEST_MEM_BASE = 65536;
  function regOffset(r) {
    if (r === 31) return SP_OFFSET;
    return REG_OFFSET + r * 8;
  }
  var HALT_PC = 0xDEAD0000n;
  var SVC_PC = 0xDEAD0001n;

  // src/cpu.ts
  var CpuState = class {
    memory;
    /** per-thread register-block base in linear memory. 0 for the main thread / single-threaded;
     * a thread-worker's block sits at its reg_base so threads sharing one memory keep private regs. */
    base;
    _view;
    _buffer;
    constructor(memory2, base = 0) {
      this.memory = memory2;
      this.base = base;
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
      return this.view.getBigUint64(this.base + regOffset(r), true);
    }
    setReg(r, value) {
      if (r === 31) return;
      this.view.setBigUint64(this.base + regOffset(r), value, true);
    }
    getSP() {
      return this.view.getBigUint64(this.base + SP_OFFSET, true);
    }
    setSP(value) {
      this.view.setBigUint64(this.base + SP_OFFSET, value, true);
    }
    getPC() {
      return this.view.getBigUint64(this.base + PC_OFFSET, true);
    }
    setPC(value) {
      this.view.setBigUint64(this.base + PC_OFFSET, value, true);
    }
    getNZCV() {
      return this.view.getBigUint64(this.base + NZCV_OFFSET, true);
    }
    setNZCV(value) {
      this.view.setBigUint64(this.base + NZCV_OFFSET, value, true);
    }
    reset() {
      for (let i = 0; i < 31; i++) this.setReg(i, 0n);
      this.setSP(0n);
      this.setPC(0n);
      this.setNZCV(0n);
      for (let i = 0; i < 32 * 16; i += 8) {
        this.view.setBigUint64(this.base + V_OFFSET + i, 0n, true);
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

  // src/crypto-ext.ts
  var SBOX = new Uint8Array(256);
  var INV_SBOX = new Uint8Array(256);
  var GF_EXP = new Uint8Array(256);
  var GF_LOG = new Uint8Array(256);
  (function buildAesTables() {
    let x = 1;
    for (let i = 0; i < 255; i++) {
      GF_EXP[i] = x;
      GF_LOG[x] = i;
      let hi = x & 128;
      x = x << 1 & 255 ^ (hi ? 27 : 0) ^ x;
    }
    GF_EXP[255] = GF_EXP[0];
    const rotl8 = (b, n) => (b << n | b >>> 8 - n) & 255;
    for (let a = 0; a < 256; a++) {
      const inv = a === 0 ? 0 : GF_EXP[255 - GF_LOG[a]];
      const s = inv ^ rotl8(inv, 1) ^ rotl8(inv, 2) ^ rotl8(inv, 3) ^ rotl8(inv, 4) ^ 99;
      SBOX[a] = s & 255;
      INV_SBOX[s & 255] = a;
    }
    if (SBOX[0] !== 99 || SBOX[1] !== 124 || SBOX[83] !== 237 || SBOX[255] !== 22) {
      throw new Error("AES S-box generation failed self-check");
    }
  })();
  var SHIFT_ROWS = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11];
  var INV_SHIFT_ROWS = new Array(16);
  for (let i = 0; i < 16; i++) INV_SHIFT_ROWS[SHIFT_ROWS[i]] = i;
  function gmul(a, b) {
    if (a === 0 || b === 0) return 0;
    return GF_EXP[(GF_LOG[a] + GF_LOG[b]) % 255];
  }
  function mixColumn(s02, s12, s2, s3, out, off, m) {
    out[off + 0] = gmul(s02, m[0]) ^ gmul(s12, m[1]) ^ gmul(s2, m[2]) ^ gmul(s3, m[3]);
    out[off + 1] = gmul(s02, m[3]) ^ gmul(s12, m[0]) ^ gmul(s2, m[1]) ^ gmul(s3, m[2]);
    out[off + 2] = gmul(s02, m[2]) ^ gmul(s12, m[3]) ^ gmul(s2, m[0]) ^ gmul(s3, m[1]);
    out[off + 3] = gmul(s02, m[1]) ^ gmul(s12, m[2]) ^ gmul(s2, m[3]) ^ gmul(s3, m[0]);
  }
  var MIX = [2, 3, 1, 1];
  var INV_MIX = [14, 11, 13, 9];
  function aese(buf, vdA, vnA) {
    const t = new Uint8Array(16);
    for (let i = 0; i < 16; i++) t[i] = SBOX[buf[vdA + i] ^ buf[vnA + i]];
    for (let i = 0; i < 16; i++) buf[vdA + i] = t[SHIFT_ROWS[i]];
  }
  function aesd(buf, vdA, vnA) {
    const t = new Uint8Array(16);
    for (let i = 0; i < 16; i++) t[i] = INV_SBOX[buf[vdA + i] ^ buf[vnA + i]];
    for (let i = 0; i < 16; i++) buf[vdA + i] = t[INV_SHIFT_ROWS[i]];
  }
  function aesmc(buf, vdA, vnA) {
    const out = new Uint8Array(16);
    for (let c = 0; c < 4; c++) mixColumn(buf[vnA + c * 4], buf[vnA + c * 4 + 1], buf[vnA + c * 4 + 2], buf[vnA + c * 4 + 3], out, c * 4, MIX);
    buf.set(out, vdA);
  }
  function aesimc(buf, vdA, vnA) {
    const out = new Uint8Array(16);
    for (let c = 0; c < 4; c++) mixColumn(buf[vnA + c * 4], buf[vnA + c * 4 + 1], buf[vnA + c * 4 + 2], buf[vnA + c * 4 + 3], out, c * 4, INV_MIX);
    buf.set(out, vdA);
  }
  function readU64(buf, base) {
    let v = 0n;
    for (let k = 0; k < 8; k++) v |= BigInt(buf[base + k]) << BigInt(k * 8);
    return v;
  }
  function pmull(buf, vdA, vnA, vmA, part) {
    const a = readU64(buf, vnA + (part ? 8 : 0));
    const b = readU64(buf, vmA + (part ? 8 : 0));
    let res = 0n;
    for (let i = 0n; i < 64n; i++) {
      if (a >> i & 1n) res ^= b << i;
    }
    for (let k = 0; k < 16; k++) buf[vdA + k] = Number(res >> BigInt(k * 8) & 0xffn);
  }
  function writeU64(buf, base, v) {
    for (let k = 0; k < 8; k++) buf[base + k] = Number(v >> BigInt(k * 8) & 0xffn);
  }
  function w32(buf, base, i) {
    const o = base + 4 * i;
    return (buf[o] | buf[o + 1] << 8 | buf[o + 2] << 16 | buf[o + 3] << 24) >>> 0;
  }
  function setW32(buf, base, i, v) {
    const o = base + 4 * i;
    buf[o] = v & 255;
    buf[o + 1] = v >>> 8 & 255;
    buf[o + 2] = v >>> 16 & 255;
    buf[o + 3] = v >>> 24 & 255;
  }
  var ror32 = (x, n) => (x >>> n | x << 32 - n) >>> 0;
  var rol32 = (x, n) => ror32(x, 32 - n);
  var cho1 = (x, y, z) => (x & (y ^ z) ^ z) >>> 0;
  var par1 = (x, y, z) => (x ^ y ^ z) >>> 0;
  var maj1 = (x, y, z) => (x & y | (x | y) & z) >>> 0;
  function sha1_3reg(buf, vdA, vnA, vmA, variant) {
    const d = [w32(buf, vdA, 0), w32(buf, vdA, 1), w32(buf, vdA, 2), w32(buf, vdA, 3)];
    let n0 = w32(buf, vnA, 0);
    const m = [w32(buf, vmA, 0), w32(buf, vmA, 1), w32(buf, vmA, 2), w32(buf, vmA, 3)];
    for (let i = 0; i < 4; i++) {
      const fn = variant === 0 ? cho1(d[1], d[2], d[3]) : variant === 1 ? par1(d[1], d[2], d[3]) : maj1(d[1], d[2], d[3]);
      const t = fn + rol32(d[0], 5) + n0 + m[i] >>> 0;
      n0 = d[3];
      d[3] = d[2];
      d[2] = ror32(d[1], 2);
      d[1] = d[0];
      d[0] = t;
    }
    for (let i = 0; i < 4; i++) setW32(buf, vdA, i, d[i]);
  }
  function sha1h(buf, vdA, vnA) {
    const v = ror32(w32(buf, vnA, 0), 2);
    setW32(buf, vdA, 0, v);
    setW32(buf, vdA, 1, 0);
    setW32(buf, vdA, 2, 0);
    setW32(buf, vdA, 3, 0);
  }
  function sha1su0(buf, vdA, vnA, vmA) {
    const d0 = readU64(buf, vdA + 8) ^ readU64(buf, vdA) ^ readU64(buf, vmA);
    const d1 = readU64(buf, vnA) ^ readU64(buf, vdA + 8) ^ readU64(buf, vmA + 8);
    writeU64(buf, vdA, d0);
    writeU64(buf, vdA + 8, d1);
  }
  function sha1su1(buf, vdA, vnA) {
    const d = [w32(buf, vdA, 0), w32(buf, vdA, 1), w32(buf, vdA, 2), w32(buf, vdA, 3)];
    const m = [w32(buf, vnA, 0), w32(buf, vnA, 1), w32(buf, vnA, 2), w32(buf, vnA, 3)];
    d[0] = rol32((d[0] ^ m[1]) >>> 0, 1);
    d[1] = rol32((d[1] ^ m[2]) >>> 0, 1);
    d[2] = rol32((d[2] ^ m[3]) >>> 0, 1);
    d[3] = rol32((d[3] ^ d[0]) >>> 0, 1);
    for (let i = 0; i < 4; i++) setW32(buf, vdA, i, d[i]);
  }
  var S0 = (x) => (ror32(x, 2) ^ ror32(x, 13) ^ ror32(x, 22)) >>> 0;
  var S1 = (x) => (ror32(x, 6) ^ ror32(x, 11) ^ ror32(x, 25)) >>> 0;
  var s0 = (x) => (ror32(x, 7) ^ ror32(x, 18) ^ x >>> 3) >>> 0;
  var s1 = (x) => (ror32(x, 17) ^ ror32(x, 19) ^ x >>> 10) >>> 0;
  function sha256h(buf, vdA, vnA, vmA, part2) {
    const d = [w32(buf, vdA, 0), w32(buf, vdA, 1), w32(buf, vdA, 2), w32(buf, vdA, 3)];
    const n = [w32(buf, vnA, 0), w32(buf, vnA, 1), w32(buf, vnA, 2), w32(buf, vnA, 3)];
    const m = [w32(buf, vmA, 0), w32(buf, vmA, 1), w32(buf, vmA, 2), w32(buf, vmA, 3)];
    if (!part2) {
      for (let i = 0; i < 4; i++) {
        let t = cho1(n[0], n[1], n[2]) + n[3] + S1(n[0]) + m[i] >>> 0;
        n[3] = n[2];
        n[2] = n[1];
        n[1] = n[0];
        n[0] = d[3] + t >>> 0;
        t = t + maj1(d[0], d[1], d[2]) + S0(d[0]) >>> 0;
        d[3] = d[2];
        d[2] = d[1];
        d[1] = d[0];
        d[0] = t;
      }
    } else {
      for (let i = 0; i < 4; i++) {
        const t = cho1(d[0], d[1], d[2]) + d[3] + S1(d[0]) + m[i] >>> 0;
        d[3] = d[2];
        d[2] = d[1];
        d[1] = d[0];
        d[0] = n[3 - i] + t >>> 0;
      }
    }
    for (let i = 0; i < 4; i++) setW32(buf, vdA, i, d[i]);
  }
  function sha256su0(buf, vdA, vnA) {
    const d = [w32(buf, vdA, 0), w32(buf, vdA, 1), w32(buf, vdA, 2), w32(buf, vdA, 3)];
    const m0 = w32(buf, vnA, 0);
    setW32(buf, vdA, 0, d[0] + s0(d[1]) >>> 0);
    setW32(buf, vdA, 1, d[1] + s0(d[2]) >>> 0);
    setW32(buf, vdA, 2, d[2] + s0(d[3]) >>> 0);
    setW32(buf, vdA, 3, d[3] + s0(m0) >>> 0);
  }
  function sha256su1(buf, vdA, vnA, vmA) {
    const d = [w32(buf, vdA, 0), w32(buf, vdA, 1), w32(buf, vdA, 2), w32(buf, vdA, 3)];
    const n = [w32(buf, vnA, 0), w32(buf, vnA, 1), w32(buf, vnA, 2), w32(buf, vnA, 3)];
    const m = [w32(buf, vmA, 0), w32(buf, vmA, 1), w32(buf, vmA, 2), w32(buf, vmA, 3)];
    d[0] = d[0] + s1(m[2]) + n[1] >>> 0;
    d[1] = d[1] + s1(m[3]) + n[2] >>> 0;
    d[2] = d[2] + s1(d[0]) + n[3] >>> 0;
    d[3] = d[3] + s1(d[1]) + m[0] >>> 0;
    for (let i = 0; i < 4; i++) setW32(buf, vdA, i, d[i]);
  }
  var M64 = 0xffffffffffffffffn;
  var ror64 = (x, n) => (x >> n | x << 64n - n) & M64;
  var cho512 = (x, y, z) => (x & (y ^ z) ^ z) & M64;
  var maj512 = (x, y, z) => (x & y | (x | y) & z) & M64;
  var S0_512 = (x) => (ror64(x, 28n) ^ ror64(x, 34n) ^ ror64(x, 39n)) & M64;
  var S1_512 = (x) => (ror64(x, 14n) ^ ror64(x, 18n) ^ ror64(x, 41n)) & M64;
  var s0_512 = (x) => (ror64(x, 1n) ^ ror64(x, 8n) ^ x >> 7n) & M64;
  var s1_512 = (x) => (ror64(x, 19n) ^ ror64(x, 61n) ^ x >> 6n) & M64;
  function sha512h(buf, vdA, vnA, vmA) {
    let d0 = readU64(buf, vdA);
    let d1 = readU64(buf, vdA + 8);
    const n0 = readU64(buf, vnA), n1 = readU64(buf, vnA + 8);
    const m0 = readU64(buf, vmA), m1 = readU64(buf, vmA + 8);
    d1 = d1 + S1_512(m1) + cho512(m1, n0, n1) & M64;
    const t = d1 + m0 & M64;
    d0 = d0 + S1_512(t) + cho512(t, m1, n0) & M64;
    writeU64(buf, vdA, d0);
    writeU64(buf, vdA + 8, d1);
  }
  function sha512h2(buf, vdA, vnA, vmA) {
    let d0 = readU64(buf, vdA);
    let d1 = readU64(buf, vdA + 8);
    const n0 = readU64(buf, vnA);
    const m0 = readU64(buf, vmA), m1 = readU64(buf, vmA + 8);
    d1 = d1 + S0_512(m0) + maj512(n0, m1, m0) & M64;
    d0 = d0 + S0_512(d1) + maj512(d1, m0, m1) & M64;
    writeU64(buf, vdA, d0);
    writeU64(buf, vdA + 8, d1);
  }
  function sha512su0(buf, vdA, vnA) {
    const d0 = readU64(buf, vdA), d1 = readU64(buf, vdA + 8);
    const n0 = readU64(buf, vnA);
    writeU64(buf, vdA, d0 + s0_512(d1) & M64);
    writeU64(buf, vdA + 8, d1 + s0_512(n0) & M64);
  }
  function sha512su1(buf, vdA, vnA, vmA) {
    const d0 = readU64(buf, vdA), d1 = readU64(buf, vdA + 8);
    const n0 = readU64(buf, vnA), n1 = readU64(buf, vnA + 8);
    const m0 = readU64(buf, vmA), m1 = readU64(buf, vmA + 8);
    writeU64(buf, vdA, d0 + s1_512(n0) + m0 & M64);
    writeU64(buf, vdA + 8, d1 + s1_512(n1) + m1 & M64);
  }

  // src/helpers.ts
  var _fpBuf = new ArrayBuffer(8);
  var _f64View = new Float64Array(_fpBuf);
  var _f32View = new Float32Array(_fpBuf);
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
    f64_abs(a) {
      return f64ToBits(Math.abs(f64FromBits(a)));
    },
    f64_neg(a) {
      return f64ToBits(-f64FromBits(a));
    },
    f64_sqrt(a) {
      return f64ToBits(Math.sqrt(f64FromBits(a)));
    },
    f64_min(a, b) {
      const fa = f64FromBits(a);
      const fb = f64FromBits(b);
      if (isNaN(fa) || isNaN(fb)) return f64ToBits(NaN);
      return f64ToBits(Math.min(fa, fb));
    },
    f64_max(a, b) {
      const fa = f64FromBits(a);
      const fb = f64FromBits(b);
      if (isNaN(fa) || isNaN(fb)) return f64ToBits(NaN);
      return f64ToBits(Math.max(fa, fb));
    },
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
    f32_to_f64(a) {
      const f = f32FromBits(Number(a & 0xFFFFFFFFn));
      return f64ToBits(f);
    },
    f64_to_f32(a) {
      const f = f64FromBits(a);
      return BigInt(f32ToBits(Math.fround(f)));
    },
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
  var SIMD_TBX = 31;
  var SIMD_ZIP1 = 40;
  var SIMD_ZIP2 = 41;
  var SIMD_UZP1 = 42;
  var SIMD_UZP2 = 43;
  var SIMD_TRN1 = 44;
  var SIMD_TRN2 = 45;
  var SIMD_MUL = 50;
  var SIMD_SMAX = 51;
  var SIMD_SMIN = 52;
  var SIMD_UMIN = 53;
  var SIMD_UMAX = 54;
  var SIMD_SQADD = 55;
  var SIMD_UQADD = 56;
  var SIMD_SQSUB = 57;
  var SIMD_UQSUB = 58;
  var SIMD_ADDV = 60;
  var SIMD_SADDLV = 61;
  var SIMD_UADDLV = 62;
  var SIMD_SMAXV = 63;
  var SIMD_UMAXV = 64;
  var SIMD_SMINV = 65;
  var SIMD_UMINV = 66;
  var SIMD_FMAXV = 67;
  var SIMD_FMINV = 68;
  var SIMD_LDST_STRUCT = 70;
  var SIMD_SHRN = 72;
  var SIMD_ADDHN = 73;
  var SIMD_UMINP = 22;
  var SIMD_EXT = 74;
  var SIMD_REV = 76;
  var SIMD_RBIT = 77;
  var SIMD_IDX_MUL = 78;
  var SIMD_XTN = 79;
  var SIMD_AESE = 80;
  var SIMD_AESD = 81;
  var SIMD_AESMC = 82;
  var SIMD_AESIMC = 83;
  var SIMD_PMULL = 84;
  var SIMD_SHA1C = 85;
  var SIMD_SHA1P = 86;
  var SIMD_SHA1M = 87;
  var SIMD_SHA1H = 88;
  var SIMD_SHA1SU0 = 89;
  var SIMD_SHA1SU1 = 90;
  var SIMD_SHA256H = 91;
  var SIMD_SHA256H2 = 92;
  var SIMD_SHA256SU0 = 93;
  var SIMD_SHA256SU1 = 94;
  var SIMD_SHA512H = 95;
  var SIMD_SHA512H2 = 96;
  var SIMD_SHA512SU0 = 97;
  var SIMD_SHA512SU1 = 98;
  var SIMD_LDST1_SINGLE = 99;
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
      if (op === SIMD_LDST_STRUCT) {
        const regCount = extra & 15;
        const isLoad = extra >> 4 & 1;
        const es = 1 << size;
        const laneBytes = Q ? 16 : 8;
        const lanes = laneBytes / es;
        const gview = new DataView(memory2.buffer);
        const addr = Number(gview.getBigUint64(rn * 8, true)) + GUEST_MEM_BASE;
        const cp = (from, to) => {
          for (let k = 0; k < es; k++) buf[to + k] = buf[from + k];
        };
        if (isLoad) {
          for (let e = 0; e < lanes; e++) {
            for (let r = 0; r < regCount; r++) {
              const memOff = addr + (e * regCount + r) * es;
              const vOff = V_OFFSET + (rd + r & 31) * 16 + e * es;
              cp(memOff, vOff);
            }
          }
          if (!Q) for (let r = 0; r < regCount; r++) buf.fill(0, V_OFFSET + (rd + r & 31) * 16 + 8, V_OFFSET + (rd + r & 31) * 16 + 16);
        } else {
          for (let e = 0; e < lanes; e++) {
            for (let r = 0; r < regCount; r++) {
              const memOff = addr + (e * regCount + r) * es;
              const vOff = V_OFFSET + (rd + r & 31) * 16 + e * es;
              cp(vOff, memOff);
            }
          }
        }
        return 0n;
      }
      if (op === SIMD_SHRN) {
        const destEs = 1 << size;
        const srcEs = destEs * 2;
        const shift = BigInt(extra);
        const n = 8 / destEs;
        const gv = new DataView(memory2.buffer);
        const vnA = V_OFFSET + rn * 16, vdA = V_OFFSET + rd * 16;
        const out = new Uint8Array(8);
        const ov = new DataView(out.buffer);
        for (let i = 0; i < n; i++) {
          const src = srcEs === 2 ? BigInt(gv.getUint16(vnA + i * 2, true)) : srcEs === 4 ? BigInt(gv.getUint32(vnA + i * 4, true)) : gv.getBigUint64(vnA + i * 8, true);
          const t = BigInt.asUintN(destEs * 8, src >> shift);
          if (destEs === 1) ov.setUint8(i, Number(t));
          else if (destEs === 2) ov.setUint16(i * 2, Number(t), true);
          else ov.setUint32(i * 4, Number(t), true);
        }
        const off = Q ? 8 : 0;
        for (let k = 0; k < 8; k++) buf[vdA + off + k] = out[k];
        if (!Q) for (let k = 8; k < 16; k++) buf[vdA + k] = 0;
        return 0n;
      }
      if (op === SIMD_ADDHN) {
        const destEs = 1 << size;
        const srcEs = destEs * 2;
        const srcBits = BigInt(srcEs * 8);
        const srcMask = (1n << srcBits) - 1n;
        const n = 8 / destEs;
        const gv = new DataView(memory2.buffer);
        const vnA = V_OFFSET + rn * 16, vmA = V_OFFSET + rm * 16, vdA = V_OFFSET + rd * 16;
        const rdSrc = (base, i) => srcEs === 2 ? BigInt(gv.getUint16(base + i * 2, true)) : srcEs === 4 ? BigInt(gv.getUint32(base + i * 4, true)) : gv.getBigUint64(base + i * 8, true);
        const out = new Uint8Array(8);
        const ov = new DataView(out.buffer);
        for (let i = 0; i < n; i++) {
          const sum = rdSrc(vnA, i) + rdSrc(vmA, i) & srcMask;
          const t = BigInt.asUintN(destEs * 8, sum >> BigInt(destEs * 8));
          if (destEs === 1) ov.setUint8(i, Number(t));
          else if (destEs === 2) ov.setUint16(i * 2, Number(t), true);
          else ov.setUint32(i * 4, Number(t), true);
        }
        const off = Q ? 8 : 0;
        for (let k = 0; k < 8; k++) buf[vdA + off + k] = out[k];
        if (!Q) for (let k = 8; k < 16; k++) buf[vdA + k] = 0;
        return 0n;
      }
      if (op === SIMD_EXT) {
        const w = Q ? 16 : 8;
        const index = extra;
        const vnA = V_OFFSET + rn * 16, vmA = V_OFFSET + rm * 16, vdA = V_OFFSET + rd * 16;
        const concat = new Uint8Array(2 * w);
        for (let k = 0; k < w; k++) {
          concat[k] = buf[vnA + k];
          concat[w + k] = buf[vmA + k];
        }
        const out = new Uint8Array(16);
        for (let i = 0; i < w; i++) out[i] = concat[index + i];
        for (let k = 0; k < 16; k++) buf[vdA + k] = out[k];
        return 0n;
      }
      if (op === SIMD_REV) {
        const w = Q ? 16 : 8;
        const container = extra;
        const es = 1 << size;
        const nElems = container / es;
        const vnA = V_OFFSET + rn * 16, vdA = V_OFFSET + rd * 16;
        const src = new Uint8Array(16);
        src.set(buf.subarray(vnA, vnA + 16));
        for (let i = 0; i < 16; i++) {
          if (i >= w) {
            buf[vdA + i] = 0;
            continue;
          }
          const base = Math.floor(i / container) * container;
          const pos = i % container;
          const srcPos = base + (nElems - 1 - Math.floor(pos / es)) * es + pos % es;
          buf[vdA + i] = src[srcPos];
        }
        return 0n;
      }
      if (op === SIMD_RBIT) {
        const w = Q ? 16 : 8;
        const vnA = V_OFFSET + rn * 16, vdA = V_OFFSET + rd * 16;
        const src = new Uint8Array(16);
        src.set(buf.subarray(vnA, vnA + 16));
        for (let i = 0; i < 16; i++) {
          if (i >= w) {
            buf[vdA + i] = 0;
            continue;
          }
          let b = src[i], r = 0;
          for (let k = 0; k < 8; k++) {
            r = r << 1 | b & 1;
            b >>= 1;
          }
          buf[vdA + i] = r;
        }
        return 0n;
      }
      if (op === SIMD_XTN) {
        const destEs = 1 << size;
        const srcEs = destEs * 2;
        const n = 8 / destEs;
        const vnA = V_OFFSET + rn * 16, vdA = V_OFFSET + rd * 16;
        const src = new Uint8Array(16);
        src.set(buf.subarray(vnA, vnA + 16));
        const half = new Uint8Array(8);
        for (let j = 0; j < n; j++) {
          for (let k = 0; k < destEs; k++) half[j * destEs + k] = src[j * srcEs + k];
        }
        if (Q) {
          for (let k = 0; k < 8; k++) buf[vdA + 8 + k] = half[k];
        } else {
          for (let k = 0; k < 8; k++) buf[vdA + k] = half[k];
          for (let k = 8; k < 16; k++) buf[vdA + k] = 0;
        }
        return 0n;
      }
      if (op === SIMD_AESE) {
        aese(buf, V_OFFSET + rd * 16, V_OFFSET + rn * 16);
        return 0n;
      }
      if (op === SIMD_AESD) {
        aesd(buf, V_OFFSET + rd * 16, V_OFFSET + rn * 16);
        return 0n;
      }
      if (op === SIMD_AESMC) {
        aesmc(buf, V_OFFSET + rd * 16, V_OFFSET + rn * 16);
        return 0n;
      }
      if (op === SIMD_AESIMC) {
        aesimc(buf, V_OFFSET + rd * 16, V_OFFSET + rn * 16);
        return 0n;
      }
      if (op === SIMD_PMULL) {
        pmull(buf, V_OFFSET + rd * 16, V_OFFSET + rn * 16, V_OFFSET + rm * 16, Q === 1);
        return 0n;
      }
      const D = V_OFFSET + rd * 16, N = V_OFFSET + rn * 16, Mv = V_OFFSET + rm * 16;
      if (op === SIMD_SHA1C) {
        sha1_3reg(buf, D, N, Mv, 0);
        return 0n;
      }
      if (op === SIMD_SHA1P) {
        sha1_3reg(buf, D, N, Mv, 1);
        return 0n;
      }
      if (op === SIMD_SHA1M) {
        sha1_3reg(buf, D, N, Mv, 2);
        return 0n;
      }
      if (op === SIMD_SHA1H) {
        sha1h(buf, D, N);
        return 0n;
      }
      if (op === SIMD_SHA1SU0) {
        sha1su0(buf, D, N, Mv);
        return 0n;
      }
      if (op === SIMD_SHA1SU1) {
        sha1su1(buf, D, N);
        return 0n;
      }
      if (op === SIMD_SHA256H) {
        sha256h(buf, D, N, Mv, false);
        return 0n;
      }
      if (op === SIMD_SHA256H2) {
        sha256h(buf, D, N, Mv, true);
        return 0n;
      }
      if (op === SIMD_SHA256SU0) {
        sha256su0(buf, D, N);
        return 0n;
      }
      if (op === SIMD_SHA256SU1) {
        sha256su1(buf, D, N, Mv);
        return 0n;
      }
      if (op === SIMD_SHA512H) {
        sha512h(buf, D, N, Mv);
        return 0n;
      }
      if (op === SIMD_SHA512H2) {
        sha512h2(buf, D, N, Mv);
        return 0n;
      }
      if (op === SIMD_SHA512SU0) {
        sha512su0(buf, D, N);
        return 0n;
      }
      if (op === SIMD_SHA512SU1) {
        sha512su1(buf, D, N, Mv);
        return 0n;
      }
      if (op === SIMD_LDST1_SINGLE) {
        const esize2 = 1 << size;
        const index = extra & 15;
        const isLoad = extra >> 4 & 1;
        const isPost = extra >> 5 & 1;
        const gv = new DataView(memory2.buffer);
        const addr = Number(gv.getBigUint64(rn * 8, true)) + GUEST_MEM_BASE;
        const laneAddr = V_OFFSET + rd * 16 + index * esize2;
        if (isLoad) for (let k = 0; k < esize2; k++) buf[laneAddr + k] = buf[addr + k];
        else for (let k = 0; k < esize2; k++) buf[addr + k] = buf[laneAddr + k];
        if (isPost) {
          const inc = rm === 31 ? esize2 : Number(gv.getBigUint64(rm * 8, true));
          gv.setBigUint64(rn * 8, gv.getBigUint64(rn * 8, true) + BigInt(inc), true);
        }
        return 0n;
      }
      if (op === SIMD_IDX_MUL) {
        const srcEs = 1 << size;
        const uns = (extra >> 6 & 1) === 1;
        const variant = extra >> 3 & 7;
        const idx = extra & 7;
        const widening = variant >= 3;
        const isSub = variant === 2 || variant === 5;
        const isAcc = variant !== 0 && variant !== 3;
        const destEs = widening ? srcEs * 2 : srcEs;
        const vnA = V_OFFSET + rn * 16, vmA = V_OFFSET + rm * 16, vdA = V_OFFSET + rd * 16;
        const oldVd = new Uint8Array(16);
        oldVd.set(buf.subarray(vdA, vdA + 16));
        const readEl = (b, base, es, i, signed) => {
          let v = 0n;
          for (let k = 0; k < es; k++) v |= BigInt(b[base + i * es + k]) << BigInt(k * 8);
          if (signed) {
            const bits3 = BigInt(es * 8);
            if (v >= 1n << bits3 - 1n) v -= 1n << bits3;
          }
          return v;
        };
        const elem = readEl(buf, vmA, srcEs, idx, !uns);
        const nDest = widening ? 16 / destEs : (Q ? 16 : 8) / srcEs;
        const srcBase = widening && Q ? nDest : 0;
        const out = new Uint8Array(16);
        for (let j = 0; j < nDest; j++) {
          let prod = BigInt.asUintN(destEs * 8, readEl(buf, vnA, srcEs, srcBase + j, !uns) * elem);
          if (isAcc) {
            let acc = 0n;
            for (let k = 0; k < destEs; k++) acc |= BigInt(oldVd[j * destEs + k]) << BigInt(k * 8);
            prod = BigInt.asUintN(destEs * 8, isSub ? acc - prod : acc + prod);
          }
          for (let k = 0; k < destEs; k++) out[j * destEs + k] = Number(prod >> BigInt(k * 8) & 0xffn);
        }
        buf.set(out, vdA);
        return 0n;
      }
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
      const getS = (view, i) => {
        if (esize === 1) return BigInt(view.getInt8(i));
        if (esize === 2) return BigInt(view.getInt16(i * 2, true));
        if (esize === 4) return BigInt(view.getInt32(i * 4, true));
        return view.getBigInt64(i * 8, true);
      };
      const getU = (view, i) => {
        if (esize === 1) return BigInt(view.getUint8(i));
        if (esize === 2) return BigInt(view.getUint16(i * 2, true));
        if (esize === 4) return BigInt(view.getUint32(i * 4, true));
        return view.getBigUint64(i * 8, true);
      };
      const put = (i, val) => {
        const m = BigInt.asUintN(esize * 8, val);
        if (esize === 1) dView.setUint8(i, Number(m));
        else if (esize === 2) dView.setUint16(i * 2, Number(m), true);
        else if (esize === 4) dView.setUint32(i * 4, Number(m), true);
        else dView.setBigUint64(i * 8, m, true);
      };
      const bits2 = BigInt(esize * 8);
      const sMax = (1n << bits2 - 1n) - 1n, sMin = -(1n << bits2 - 1n);
      const uMax = (1n << bits2) - 1n;
      const clampS = (x) => x > sMax ? sMax : x < sMin ? sMin : x;
      const clampU = (x) => x > uMax ? uMax : x < 0n ? 0n : x;
      const putScalar = (val, wbytes) => {
        const m = BigInt.asUintN(wbytes * 8, val);
        if (wbytes === 1) dView.setUint8(0, Number(m));
        else if (wbytes === 2) dView.setUint16(0, Number(m), true);
        else if (wbytes === 4) dView.setUint32(0, Number(m), true);
        else dView.setBigUint64(0, m, true);
      };
      switch (op) {
        case SIMD_ADDV: {
          let s = 0n;
          for (let i = 0; i < elems; i++) s += getU(aView, i);
          putScalar(s, esize);
          break;
        }
        case SIMD_SADDLV: {
          let s = 0n;
          for (let i = 0; i < elems; i++) s += getS(aView, i);
          putScalar(s, esize * 2);
          break;
        }
        case SIMD_UADDLV: {
          let s = 0n;
          for (let i = 0; i < elems; i++) s += getU(aView, i);
          putScalar(s, esize * 2);
          break;
        }
        case SIMD_SMAXV: {
          let s = getS(aView, 0);
          for (let i = 1; i < elems; i++) {
            const x = getS(aView, i);
            if (x > s) s = x;
          }
          putScalar(s, esize);
          break;
        }
        case SIMD_SMINV: {
          let s = getS(aView, 0);
          for (let i = 1; i < elems; i++) {
            const x = getS(aView, i);
            if (x < s) s = x;
          }
          putScalar(s, esize);
          break;
        }
        case SIMD_UMAXV: {
          let s = getU(aView, 0);
          for (let i = 1; i < elems; i++) {
            const x = getU(aView, i);
            if (x > s) s = x;
          }
          putScalar(s, esize);
          break;
        }
        case SIMD_UMINV: {
          let s = getU(aView, 0);
          for (let i = 1; i < elems; i++) {
            const x = getU(aView, i);
            if (x < s) s = x;
          }
          putScalar(s, esize);
          break;
        }
        case SIMD_FMAXV:
        case SIMD_FMINV: {
          const getF = (i) => esize === 4 ? aView.getFloat32(i * 4, true) : aView.getFloat64(i * 8, true);
          let s = getF(0);
          for (let i = 1; i < elems; i++) {
            const x = getF(i);
            s = op === SIMD_FMAXV ? Math.max(s, x) : Math.min(s, x);
          }
          if (esize === 4) dView.setFloat32(0, s, true);
          else dView.setFloat64(0, s, true);
          break;
        }
        case SIMD_MUL:
          for (let i = 0; i < elems; i++) put(i, getU(aView, i) * getU(bView, i));
          break;
        case SIMD_SMAX:
          for (let i = 0; i < elems; i++) {
            const x = getS(aView, i), y = getS(bView, i);
            put(i, x > y ? x : y);
          }
          break;
        case SIMD_SMIN:
          for (let i = 0; i < elems; i++) {
            const x = getS(aView, i), y = getS(bView, i);
            put(i, x < y ? x : y);
          }
          break;
        case SIMD_UMAX:
          for (let i = 0; i < elems; i++) {
            const x = getU(aView, i), y = getU(bView, i);
            put(i, x > y ? x : y);
          }
          break;
        case SIMD_UMIN:
          for (let i = 0; i < elems; i++) {
            const x = getU(aView, i), y = getU(bView, i);
            put(i, x < y ? x : y);
          }
          break;
        case SIMD_SQADD:
          for (let i = 0; i < elems; i++) put(i, clampS(getS(aView, i) + getS(bView, i)));
          break;
        case SIMD_SQSUB:
          for (let i = 0; i < elems; i++) put(i, clampS(getS(aView, i) - getS(bView, i)));
          break;
        case SIMD_UQADD:
          for (let i = 0; i < elems; i++) put(i, clampU(getU(aView, i) + getU(bView, i)));
          break;
        case SIMD_UQSUB:
          for (let i = 0; i < elems; i++) put(i, clampU(getU(aView, i) - getU(bView, i)));
          break;
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
            if (esize === 1) dView.setUint8(i, gt ? 255 : 0);
            else if (esize === 2) dView.setUint16(i * 2, gt ? 65535 : 0, true);
            else if (esize === 4) dView.setUint32(i * 4, gt ? 4294967295 : 0, true);
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
            if (esize === 1) dView.setUint8(i, hi ? 255 : 0);
            else if (esize === 2) dView.setUint16(i * 2, hi ? 65535 : 0, true);
            else if (esize === 4) dView.setUint32(i * 4, hi ? 4294967295 : 0, true);
            else dView.setBigUint64(i * 8, hi ? 0xFFFFFFFFFFFFFFFFn : 0n, true);
          }
          break;
        case SIMD_CMGE:
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
        case SIMD_UMAXP:
        case SIMD_UMINP: {
          const wantMax = op === SIMD_UMAXP;
          const all = new Uint8Array(32);
          all.set(a.subarray(0, bytes), 0);
          all.set(b_.subarray(0, bytes), bytes);
          const allView = new DataView(all.buffer);
          const pick = (v1, v2) => wantMax ? v1 > v2 ? v1 : v2 : v1 < v2 ? v1 : v2;
          for (let i = 0; i < elems; i++) {
            const off1 = i * 2 * esize;
            const off2 = off1 + esize;
            if (esize === 1) dView.setUint8(i, pick(allView.getUint8(off1), allView.getUint8(off2)));
            else if (esize === 2) dView.setUint16(i * 2, pick(allView.getUint16(off1, true), allView.getUint16(off2, true)), true);
            else if (esize === 4) dView.setUint32(i * 4, pick(allView.getUint32(off1, true), allView.getUint32(off2, true)), true);
            else dView.setBigUint64(i * 8, pick(allView.getBigUint64(off1, true), allView.getBigUint64(off2, true)), true);
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
        case SIMD_TBL:
        case SIMD_TBX: {
          const len = extra;
          const tableSize = (len + 1) * 16;
          const table = new Uint8Array(tableSize);
          for (let r = 0; r <= len; r++) {
            const vr = V_OFFSET + (rn + r & 31) * 16;
            table.set(buf.subarray(vr, vr + 16), r * 16);
          }
          if (op === SIMD_TBX) dst.set(buf.subarray(vd, vd + bytes));
          for (let i = 0; i < bytes; i++) {
            const idx = b_[i];
            if (idx < tableSize) dst[i] = table[idx];
            else if (op === SIMD_TBL) dst[i] = 0;
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
      if (!Q) buf.fill(0, vd + 8, vd + 16);
      return 0n;
    };
  }
  function createEnv(memory2, regBase) {
    const env = { memory: memory2, ...helperFunctions, simd_op: createSimdOp(memory2) };
    if (regBase) env.reg_base = regBase;
    return env;
  }

  // src/zig-translator.ts
  var import_meta = {};
  var HAS_PROCESS = typeof process !== "undefined" && !!process.env;
  var JIT_DUMP_PC = HAS_PROCESS ? process.env.JIT_DUMP_PC ?? "" : "";
  var JIT_DEBUG = HAS_PROCESS ? !!process.env.JIT_DEBUG : false;
  var dbgWrite = (s) => {
    if (HAS_PROCESS) process.stderr.write(s);
  };
  var exports_ = null;
  var autoloadTried = false;
  function initZigTranslator(wasmBytes) {
    const bytes = wasmBytes instanceof Uint8Array ? wasmBytes : new Uint8Array(wasmBytes);
    initZigTranslatorModule(new WebAssembly.Module(bytes));
  }
  function initZigTranslatorModule(mod) {
    const instance = new WebAssembly.Instance(mod);
    exports_ = instance.exports;
  }
  function ensureInit() {
    if (exports_) return true;
    if (autoloadTried) return false;
    autoloadTried = true;
    if (typeof process === "undefined" || !process.versions?.node) return false;
    try {
      const fs = process.getBuiltinModule("fs");
      const path = process.getBuiltinModule("path");
      const url = process.getBuiltinModule("url");
      const here = path.dirname(url.fileURLToPath(import_meta.url));
      const bytes = fs.readFileSync(path.join(here, "..", "engine", "zig-out", "bin", "translator.wasm"));
      initZigTranslator(new Uint8Array(bytes));
      return true;
    } catch {
      return false;
    }
  }
  function readGuestBytes(view, pc, maxBytes) {
    const baseOffset = Number(pc) + GUEST_MEM_BASE;
    const memSize = view.byteLength;
    const limit = Math.min(maxBytes, memSize - baseOffset);
    if (limit <= 0) return new Uint8Array(0);
    return new Uint8Array(view.buffer, baseOffset, limit).slice();
  }
  function translateRegionZig(view, entryPC, memory2, shadowBase, singleBlock = false, threaded2 = false, regBase = null) {
    if (!ensureInit()) return null;
    let codeBytes;
    const basePC = entryPC;
    if (singleBlock) {
      const block = detectBlock(view, entryPC);
      codeBytes = readGuestBytes(view, entryPC, block.insns.length * 4);
    } else {
      codeBytes = readGuestBytes(view, entryPC, 4096);
    }
    if (codeBytes.length < 4) return null;
    const ex = exports_;
    const tbuf = new Uint8Array(ex.memory.buffer);
    const inOff = ex.translator_input_buf_offset();
    tbuf.set(codeBytes, inOff);
    ex.translator_reset();
    const basePCLo = Number(basePC & 0xffffffffn);
    const basePCHi = Number(basePC >> 32n & 0xffffffffn);
    const entryLo = Number(entryPC & 0xffffffffn);
    const entryHi = Number(entryPC >> 32n & 0xffffffffn);
    const rc = ex.translator_translate_region(basePCLo, basePCHi, entryLo, entryHi, codeBytes.length, shadowBase, threaded2 ? 1 : 0);
    if (rc !== 0) return null;
    const outLen = ex.translator_result_len();
    const outOff = ex.translator_result_offset();
    const moduleBytes = new Uint8Array(outLen);
    const tbufFresh = new Uint8Array(ex.memory.buffer);
    moduleBytes.set(tbufFresh.subarray(outOff, outOff + outLen));
    let mod;
    try {
      mod = new WebAssembly.Module(moduleBytes);
    } catch {
      return null;
    }
    if (JIT_DUMP_PC && entryPC.toString(16) === JIT_DUMP_PC) {
      dbgWrite(`zig dump for pc=0x${entryPC.toString(16)} module bytes:
`);
      let hex = "";
      for (let i = 0; i < moduleBytes.length; i++) hex += moduleBytes[i].toString(16).padStart(2, "0");
      dbgWrite(hex + "\n");
      dbgWrite("source bytes:\n");
      let src = "";
      for (let i = 0; i < codeBytes.length && i < 64; i++) src += codeBytes[i].toString(16).padStart(2, "0");
      dbgWrite(src + "\n");
    }
    let inst;
    try {
      inst = new WebAssembly.Instance(mod, { env: createEnv(memory2, regBase) });
    } catch {
      return null;
    }
    const runZig = inst.exports.run;
    const pcSlot = (regBase ? regBase.value : 0) + PC_OFFSET;
    let callCount = 0;
    const region = {
      run: () => {
        const packed = runZig();
        const reason = Number(packed >> 32n & 0xffffn);
        region.lastExitReason = reason;
        const view2 = new DataView(memory2.buffer);
        const nextPC = view2.getBigUint64(pcSlot, true);
        callCount++;
        if (JIT_DEBUG && callCount <= 200) {
          dbgWrite(`#${callCount} zig entry=0x${entryPC.toString(16)} reason=${reason} -> 0x${nextPC.toString(16)}
`);
        }
        switch (reason) {
          case 0:
            return HALT_PC;
          case 1:
            return SVC_PC;
          default:
            return nextPC;
        }
      },
      entryPC,
      blockPCs: [entryPC]
      // we don't expose per-block PCs from zig yet; entry only
    };
    return region;
  }

  // src/syscall.ts
  function bytesToBase64(data) {
    const chunk = 32768;
    let s = "";
    for (let i = 0; i < data.length; i += chunk) {
      s += String.fromCharCode.apply(null, data.subarray(i, i + chunk));
    }
    return btoa(s);
  }
  var ENOSYS = 38;
  var EBADF = 9;
  var EINVAL = 22;
  var ENOENT = 2;
  var EPERM = 1;
  var EMFILE = 24;
  var ENOTDIR = 20;
  var ECHILD = 10;
  var ESPIPE = 29;
  var EAGAIN = 11;
  var EFAULT = 14;
  var EACCES = 13;
  var ENOMEM = 12;
  var MAX_FDS = 1024;
  var MAX_PROCESS_BYTES = 1610612736;
  var O_NONBLOCK = 2048;
  var SIG_DFL = 0n;
  var SIG_IGN = 1n;
  var SIGALRM = 14;
  var SIGPIPE = 13;
  var SIGCHLD = 17;
  var SIGNAL_TRAMPOLINE = 4096;
  var SIG_IGNORE_DEFAULT = /* @__PURE__ */ new Set([SIGCHLD, 17, 20]);
  var SIG_STOP_DEFAULT = /* @__PURE__ */ new Set([19, 20, 21, 22]);
  var SIGNAL_FRAME_V_OFFSET = 288;
  var SIGNAL_FRAME_SIZE = SIGNAL_FRAME_V_OFFSET + 512;
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
  var NET_TRACE = false;
  var FD_TRACE = false;
  var SyscallHandler = class {
    nt(s) {
      if (NET_TRACE) this.io.writeStderr(new TextEncoder().encode("[net] " + s + "\n"));
    }
    ft(s) {
      if (FD_TRACE) this.io.writeStderr(new TextEncoder().encode("[fd] " + s + "\n"));
    }
    brk;
    // the exclusive-monitor shadow lives at ZIG_SHADOW_BASE (32MB) and is tiny
    // (1 byte per 512 guest bytes, so <=256KB for a 128MB guest). reserve [32MB,33MB)
    // for it and start the mmap arena above, so mmap'd shared libraries (ld.so mmaps
    // libc.so.6) don't collide with the shadow. mmap grows up toward the shadow_base
    // ceiling of the top-of-memory shadow... here shadow is low, so cap mmap well above.
    mmapBase = 0x2100000n;
    // 33MB — just above the 1MB shadow reservation
    // brk/mmap ceiling. a threaded process uses a lower ceiling (its shared memory places the per-thread
    // register blocks + atomics shadow just above this, and the shared reservation is bounded so several
    // threaded processes can coexist). single-threaded processes keep the full 1.5 GiB.
    maxBytes = MAX_PROCESS_BYTES;
    // MAP_SHARED regions (guest offsets). shared across fork: a child's writes to these
    // must survive the parent-memory restore in the serialized fork model.
    sharedRegions = [];
    // writable MAP_SHARED mappings of real (native) files. the guest writes into the mapping and
    // expects those bytes flushed back to the file on msync/munmap (this is how apk extracts
    // binaries). we track them and pwrite the dirty range through the bridge on flush.
    fileMmaps = [];
    // guest ranges mapped PROT_NONE (inaccessible). a user pointer landing here faults with
    // EFAULT, which is how LTP's tst_get_bad_addr() probes are built.
    protNone = [];
    cwd = "/";
    tpidr = 0n;
    umask = 18;
    fdCloexec = /* @__PURE__ */ new Set();
    fdSig = /* @__PURE__ */ new Map();
    // nanosecond clock. Date.now() is millisecond-resolution; performance.now() gives sub-ms
    // where the host allows it (browsers/WKWebView clamp it to ~1ms for spectre mitigation,
    // which is fine). posix only requires CLOCK_MONOTONIC to be non-decreasing, not strictly
    // increasing, so we allow equal consecutive reads and clamp only to defend against the
    // clock actually running backwards (performance.now() shouldn't, but Date.now() can under
    // ntp/settimeofday). no artificial per-read increment: that would distort measured
    // elapsed time under tight loops, which is the one thing a clock must get right.
    clkHasPerf = typeof performance !== "undefined";
    clkPerfBase = this.clkHasPerf ? performance.now() : Date.now();
    clkRealBaseNs = BigInt(Date.now()) * 1000000n;
    clkLastMonoNs = 0n;
    monoNs() {
      const elapsedMs = (this.clkHasPerf ? performance.now() : Date.now()) - this.clkPerfBase;
      let ns = BigInt(Math.round(elapsedMs * 1e6));
      if (ns < this.clkLastMonoNs) ns = this.clkLastMonoNs;
      else this.clkLastMonoNs = ns;
      return ns;
    }
    clockNs(clkId) {
      const mono = this.monoNs();
      if (clkId === 1 || clkId === 4 || clkId === 7 || clkId === 2 || clkId === 3) return mono;
      return this.clkRealBaseNs + mono;
    }
    // ITIMER_REAL: monotonic ms deadline at which SIGALRM fires, 0 = disarmed. interval is
    // the auto-rearm period (0 = one-shot). used by alarm()/setitimer() and by busybox
    // timeout, which is how the ltp driver bounds each test. checked by the exec loops via
    // checkTimers() so it fires even in a syscall-less spin.
    itimerRealDeadlineMs = 0;
    itimerRealIntervalMs = 0;
    // RLIMIT_NOFILE soft limit: the highest fd number a process may allocate. set via
    // prlimit64/setrlimit; enforced at fd-allocating syscalls (open/dup/...) -> EMFILE.
    nofileLimit = 1024;
    nowMs() {
      return this.monoNs() / 1000000n;
    }
    /** raise SIGALRM if the real interval timer has expired; rearm or disarm. returns true if it fired. */
    checkTimers() {
      if (this.itimerRealDeadlineMs === 0) return false;
      if (Number(this.nowMs()) < this.itimerRealDeadlineMs) return false;
      this.raiseSignal(SIGALRM);
      if (this.itimerRealIntervalMs > 0) this.itimerRealDeadlineMs += this.itimerRealIntervalMs;
      else this.itimerRealDeadlineMs = 0;
      return true;
    }
    /** ms until the real timer fires, or 0 if disarmed (for getitimer / wait timeouts). */
    timerRemainingMs() {
      if (this.itimerRealDeadlineMs === 0) return 0;
      return Math.max(0, this.itimerRealDeadlineMs - Number(this.nowMs()));
    }
    pendingSignals = [];
    // queue of signals to deliver
    epollSets;
    signalHandlers;
    signalMask;
    /** True if bridge calls can be handled (either direct bridge or Worker kernel relay) */
    bridgeAvailable = false;
    currentPid = 1;
    currentPpid = 0;
    /** thread-group id: 0 for a normal process (getpid == currentPid); for a thread it's the
     * process pid so getpid()==tgid while gettid()==currentPid(==tid). */
    tgid = 0;
    /** mark this handler as belonging to a thread: gettid returns `tid`, getpid returns `tgid`. */
    setThreadIds(tid, tgid) {
      this.currentPid = tid;
      this.tgid = tgid;
    }
    // process credentials. real/effective/saved uid+gid. start as root (0).
    ruid = 0;
    euid = 0;
    suid = 0;
    rgid = 0;
    egid = 0;
    sgid = 0;
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
    // Stat cache: populated from getdents, consumed by fs_stat. Avoids N+1 bridge calls for ls -l.
    // Path → {mode, size, nlink, mtime, expiresAt (ms since epoch)}
    statCache = /* @__PURE__ */ new Map();
    STAT_CACHE_TTL_MS = 1e3;
    lastWaitStatus;
    // exit code from last child (set by dispatch loop)
    lastWaitPid;
    // PID of last exited child
    constructor(brkStart, io, vfs2, bridge) {
      this.brk = brkStart;
      this.io = io;
      this.vfs = vfs2;
      this.bridge = bridge;
      this.fdTable.set(0, { type: "stdio", fd: 0 });
      this.fdTable.set(1, { type: "stdio", fd: 1 });
      this.fdTable.set(2, { type: "stdio", fd: 2 });
      if (this.vfs) {
        this.vfs.fdAllocator = () => this.allocFd();
        this.vfs.credsProvider = () => ({ ruid: this.ruid, euid: this.euid, rgid: this.rgid, egid: this.egid });
      }
    }
    /**
     * Check if a guest path should be routed to the native Swift filesystem.
     * Returns the resolved absolute path if bridging, or null if staying in JS VFS.
     *
     * JS VFS handles: /proc, /dev (virtual, per-process)
     * Native Swift handles: everything else (/bin, /etc, /tmp, /root, /home, /var, /usr, ...)
     */
    // resolve an *at()-family path against its dirfd. absolute paths and AT_FDCWD (-100) pass through
    // (bridgePath/vfs resolve them against cwd). otherwise the path is relative to the directory the
    // fd refers to, so join it onto that fd's known path. apk loads trust keys via openat(keys_fd, name),
    // so without this the keys never open and every index signature reads as UNTRUSTED.
    resolveAt(dirfd, path) {
      if (path.startsWith("/")) return path;
      const d = Number(BigInt.asIntN(64, dirfd));
      if (d === -100) return path;
      const base = this.getFdEntry(d)?.path;
      if (typeof base === "string" && base.length) return (base.endsWith("/") ? base : base + "/") + path;
      return path;
    }
    // flush a writable MAP_SHARED native-file mapping back to disk on msync/munmap. finds the mapping
    // overlapping [reqStart, reqStart+reqLen), pwrites its dirty (real-file) bytes through the bridge,
    // and drops the record when unmapping. returns a bridge action, or null if nothing to flush.
    buildMmapFlush(cpu2, memory2, reqStart, reqLen, removeAfter) {
      for (let i = 0; i < this.fileMmaps.length; i++) {
        const m = this.fileMmaps[i];
        if (reqStart >= m.start + m.len || reqStart + reqLen <= m.start) continue;
        if (removeAfter) this.fileMmaps.splice(i, 1);
        const fStart = Math.max(reqStart, m.start);
        const fEnd = Math.min(reqStart + reqLen, m.start + m.fileLen);
        if (fEnd <= fStart) return null;
        const data = new Uint8Array(memory2.buffer).slice(fStart + GUEST_MEM_BASE, fEnd + GUEST_MEM_BASE);
        return {
          status: "bridge",
          method: "fs_pwrite",
          args: { fd: m.nativeFd, offset: m.fileOffset + (fStart - m.start), data: bytesToBase64(data) },
          apply: () => {
            cpu2.setReg(0, 0n);
          }
        };
      }
      return null;
    }
    bridgePath(path) {
      if (!this.bridge?.isNative && !this.bridgeAvailable) return null;
      let resolved = path;
      if (!path.startsWith("/")) {
        const cwd = this.vfs?.getcwd() ?? this.cwd;
        resolved = cwd === "/" ? "/" + path : cwd + "/" + path;
      }
      const parts = [];
      for (const p of resolved.split("/")) {
        if (p === "" || p === ".") continue;
        if (p === "..") {
          parts.pop();
          continue;
        }
        parts.push(p);
      }
      resolved = "/" + parts.join("/");
      if (resolved === "/proc" || resolved.startsWith("/proc/") || resolved === "/dev" || resolved.startsWith("/dev/")) {
        return null;
      }
      return resolved;
    }
    // resolve a (possibly relative) path to a normalized absolute path against the current cwd.
    // used for execve so the worker reads the binary from the right place (a bare "./prog" must
    // become "/cwd/prog"), matching how the kernel resolves the exec pathname.
    absolutizePath(path) {
      let resolved = path;
      if (!path.startsWith("/")) {
        const cwd = this.vfs?.getcwd() ?? this.cwd;
        resolved = cwd === "/" ? "/" + path : cwd + "/" + path;
      }
      const parts = [];
      for (const p of resolved.split("/")) {
        if (p === "" || p === ".") continue;
        if (p === "..") {
          parts.pop();
          continue;
        }
        parts.push(p);
      }
      return "/" + parts.join("/");
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
      view.setBigUint64(frameBase + 280, this.signalMask ?? 0n, true);
      {
        const vBase = cpu2.base + V_OFFSET;
        for (let i = 0; i < 512; i += 8) {
          view.setBigUint64(frameBase + SIGNAL_FRAME_V_OFFSET + i, view.getBigUint64(vBase + i, true), true);
        }
      }
      const flags = handler?.flags ?? 0n;
      let newMask = (this.signalMask ?? 0n) | (handler?.mask ?? 0n);
      if ((flags & 0x40000000n) === 0n) newMask |= 1n << BigInt(signum - 1);
      this.signalMask = newMask;
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
      this.signalMask = view.getBigUint64(frameBase + 280, true);
      {
        const vBase = cpu2.base + V_OFFSET;
        for (let i = 0; i < 512; i += 8) {
          view.setBigUint64(vBase + i, view.getBigUint64(frameBase + SIGNAL_FRAME_V_OFFSET + i, true), true);
        }
      }
      cpu2.setSP(savedSP);
      cpu2.setNZCV(savedNZCV);
      return savedPC;
    }
    /** Queue a signal for delivery. */
    raiseSignal(signum) {
      this.pendingSignals.push(signum);
    }
    /** True if any queued signal has an SA_RESTART handler — the interrupted restartable syscall
     *  (wait4, blocking read) should then be re-executed instead of returning EINTR. */
    anyPendingRestart() {
      if (!this.signalHandlers) return false;
      for (const sig of this.pendingSignals) {
        const h = this.signalHandlers.get(sig);
        if (h && h.handler !== SIG_DFL && h.handler !== SIG_IGN && (h.flags & 0x10000000n) !== 0n) return true;
      }
      return false;
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
    }
    // the resolved path backing a native fd, used as the key for cross-process record locks. a
    // non-native fd (pipe, vfs) returns null: those have no on-disk identity to lock against.
    fdPath(fd) {
      const e = this.fdTable.get(fd);
      return e && e.type === "native" ? e.path : null;
    }
    // fd offset / file size for resolving SEEK_CUR / SEEK_END lock ranges. native fds keep their
    // offset and size on the Swift side, so these are best-effort (0) until a bridge query is added;
    // SEEK_SET locks (the common case) are unaffected.
    fdOffset(_fd) {
      return 0;
    }
    fdSize(_fd) {
      return 0;
    }
    /** Set process identity (used when a fork continues as the child). */
    setIdentity(pid, ppid) {
      this.currentPid = pid;
      this.currentPpid = ppid;
    }
    // the current process credentials, threaded into NativeFS path-op bridge calls so the native
    // side can enforce emulated POSIX ownership/permissions against this process's identity.
    // (supplementary groups aren't tracked yet — setgroups is a no-op — so groups is empty for now.)
    getCreds() {
      return { ruid: this.ruid, euid: this.euid, rgid: this.rgid, egid: this.egid, umask: this.umask, groups: [] };
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
        currentPid: this.currentPid,
        currentPpid: this.currentPpid,
        creds: { ruid: this.ruid, euid: this.euid, suid: this.suid, rgid: this.rgid, egid: this.egid, sgid: this.sgid },
        sharedRegions: this.sharedRegions.map((r) => ({ ...r })),
        protNone: this.protNone.map((r) => ({ ...r })),
        nofileLimit: this.nofileLimit,
        vfsFdSnapshot: this.vfs ? this.vfs.snapshotFds() : void 0
      };
    }
    /** Restore state from process snapshot */
    restoreState(state) {
      this.brk = state.brk;
      this.mmapBase = state.mmapBase;
      this.cwd = state.cwd;
      if (this.vfs) {
        this.vfs.setCwd(state.cwd);
        if (state.vfsFdSnapshot) this.vfs.restoreFds(state.vfsFdSnapshot);
      }
      this.fdFlags = new Map(state.fdFlags);
      if (state.fdTable) this.fdTable = new Map(state.fdTable);
      if (state.nativeRefs) this.nativeRefs = new Map(state.nativeRefs);
      if (state.currentPid !== void 0) this.currentPid = state.currentPid;
      if (state.currentPpid !== void 0) this.currentPpid = state.currentPpid;
      if (state.creds) {
        this.ruid = state.creds.ruid;
        this.euid = state.creds.euid;
        this.suid = state.creds.suid;
        this.rgid = state.creds.rgid;
        this.egid = state.creds.egid;
        this.sgid = state.creds.sgid;
      }
      if (state.sharedRegions !== void 0) this.sharedRegions = state.sharedRegions.map((r) => ({ ...r }));
      if (state.protNone !== void 0) this.protNone = state.protNone.map((r) => ({ ...r }));
      if (state.nofileLimit !== void 0) this.nofileLimit = state.nofileLimit;
    }
    /**
     * After fork, native fds in the inherited fdTable still reference the parent's
     * swift filehandles. dup each one so the child owns independent fds — otherwise
     * parent close() kills the child's view of the file.
     */
    dupNativeFdsAfterFork(dupFn) {
      for (const [guestFd, entry] of this.fdTable) {
        if (entry.type !== "native") continue;
        const newFd = dupFn(entry.nativeFd);
        if (newFd < 0) continue;
        this.fdTable.set(guestFd, { type: "native", nativeFd: newFd, path: entry.path });
      }
    }
    /**
     * Close every native (Swift-backed) fd this process still holds. real kernels close a
     * process's fds when it exits; jitter never did, so a Worker that runs many pooled tenants
     * leaks Swift file handles until the device fd budget is exhausted and NativeFS starts failing
     * opens. call this on process exit / before pool reuse. stdio and non-native fds are ignored.
     */
    closeNativeFds(closeFn) {
      for (const [, entry] of this.fdTable) {
        if (entry.type === "native") closeFn(entry.nativeFd);
      }
    }
    resetForExec(brkStart, closeFn) {
      this.brk = brkStart;
      this.mmapBase = 0x2100000n;
      const preserved = /* @__PURE__ */ new Map();
      for (const [fd, entry] of this.fdTable) {
        if (fd <= 2) preserved.set(fd, entry);
        else if (entry.type === "native" && closeFn) closeFn(entry.nativeFd);
      }
      this.fdTable = preserved;
      if (this.vfs) this.vfs.closeAllExcept(0, 1, 2);
    }
    /** Allocate next available fd */
    // lowest-numbered unused guest fd >= min, per posix. the guest fd namespace is
    // shared: pipe/native/socket/tty fds live in fdTable, vfs-backed fds live in the
    // vfs (where the guest fd is also the vfs internal fd). scan both so the two
    // allocators never hand out the same number.
    lowestFreeFd(min) {
      let fd = min;
      while (this.fdTable.has(fd) || (this.vfs?.hasFd(fd) ?? false)) fd++;
      return fd;
    }
    allocFd() {
      return this.lowestFreeFd(0);
    }
    /** true if a new fd (>= min) can be allocated within RLIMIT_NOFILE. */
    fdAvailable(min = 3) {
      return this.lowestFreeFd(min) < this.nofileLimit;
    }
    /** Resolve fd: check fdTable for overrides (dup2'd pipes, etc.), else fall through to default */
    getFdEntry(fd) {
      return this.fdTable.get(fd);
    }
    isNonBlocking(fd) {
      return ((this.fdFlags.get(fd) ?? 0) & O_NONBLOCK) !== 0;
    }
    /** public view of a fd's O_NONBLOCK flag (the worker needs it for kernel-pipe write blocking). */
    isNonBlockingFd(fd) {
      return this.isNonBlocking(fd);
    }
    // a user pointer faults if it is out of the guest-addressable region (badGuestPtr) or lands
    // in a range that was mapped PROT_NONE.
    ptrFaults(bytes, addr, len = 1) {
      if (badGuestPtr(bytes, addr, len)) return true;
      for (const r of this.protNone) {
        if (addr < r.end && addr + len > r.start) return true;
      }
      return false;
    }
    /** Whether the fd has data available for read without blocking. */
    isFdReadable(fd) {
      const entry = this.fdTable.get(fd);
      if (!entry) return false;
      switch (entry.type) {
        case "stdio":
          return entry.fd === 0 ? this.io.stdinHasData?.() ?? true : true;
        case "tty":
          return this.io.stdinHasData?.() ?? true;
        case "pipe_read":
          return entry.pipe.data.length > 0 || entry.pipe.writeClosed;
        case "pipe_write":
          return false;
        case "vfs":
        case "native":
          return true;
        case "socket":
          return false;
        // handled via bridge poll
        case "eventfd":
          return entry.counter > 0n;
        case "inotify":
        case "timerfd":
          return false;
        // stub: never have events
        case "epoll":
          return false;
      }
    }
    /** Whether the fd can accept a write without blocking. */
    isFdWritable(fd) {
      const entry = this.fdTable.get(fd);
      if (!entry) return false;
      switch (entry.type) {
        case "stdio":
          return entry.fd === 1 || entry.fd === 2;
        case "tty":
          return true;
        case "pipe_write":
          return !entry.pipe.readClosed;
        case "pipe_read":
          return false;
        case "vfs":
        case "native":
          return true;
        case "socket":
          return true;
        // connected TCP sockets accept writes (net_write queues via NWConnection)
        case "eventfd":
          return entry.counter < 0xFFFFFFFFFFFFFFFEn;
        case "inotify":
        case "timerfd":
        case "epoll":
          return false;
      }
    }
    /** Whether a wait on this fd's underlying waker mechanism would resume on data arrival. */
    isFdWakeable(fd) {
      const entry = this.fdTable.get(fd);
      if (!entry) return false;
      return entry.type === "stdio" && entry.fd === 0 || entry.type === "tty";
    }
    /** Read an fd_set bitmap from guest memory into a Set<number>. */
    readFdSet(view, ptr, nfds) {
      const out = /* @__PURE__ */ new Set();
      if (ptr === 0) return out;
      const bytes = new Uint8Array(view.buffer);
      for (let fd = 0; fd < nfds; fd++) {
        const byte = bytes[ptr + GUEST_MEM_BASE + (fd >> 3)];
        if ((byte & 1 << (fd & 7)) !== 0) out.add(fd);
      }
      return out;
    }
    /** Write a Set<number> as an fd_set bitmap to guest memory, zeroing the rest. */
    writeFdSet(view, ptr, nfds, set) {
      if (ptr === 0) return;
      const bytes = new Uint8Array(view.buffer);
      const sz = Math.ceil(nfds / 8);
      for (let i = 0; i < sz; i++) bytes[ptr + GUEST_MEM_BASE + i] = 0;
      for (const fd of set) {
        if (fd < nfds && fd >= 0) {
          bytes[ptr + GUEST_MEM_BASE + (fd >> 3)] |= 1 << (fd & 7);
        }
      }
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
      if (NET_TRACE && this._netActive && nr !== 64 && nr !== 66 && nr !== 63) this.nt(`sys ${nr} a0=${Number(a0 & 0xffffn)}`);
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
          return { status: "execve", path: this.absolutizePath(pathname), argv: argvPtrs, envp: envpPtrs };
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
                } else if (this.isNonBlocking(fd)) {
                  ret = -BigInt(EAGAIN);
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
            if (readEntry.type === "stdio" && readEntry.fd === 0 || readEntry.type === "tty") {
              const data = this.io.readStdin(count);
              if (data === null) {
                if (this.isNonBlocking(fd)) {
                  ret = -BigInt(EAGAIN);
                  break;
                }
                return { status: "blocked_read", fd: 0 };
              }
              bytes.set(data, buf + GUEST_MEM_BASE);
              ret = BigInt(data.length);
              break;
            }
            if (readEntry.type === "eventfd") {
              if (readEntry.counter === 0n) {
                ret = -BigInt(EAGAIN);
                break;
              }
              if (count < 8) {
                ret = -BigInt(EINVAL);
                break;
              }
              const value = readEntry.semaphore ? 1n : readEntry.counter;
              readEntry.counter = readEntry.semaphore ? readEntry.counter - 1n : 0n;
              view.setBigUint64(buf + GUEST_MEM_BASE, value, true);
              ret = 8n;
              break;
            }
            if (readEntry.type === "inotify" || readEntry.type === "timerfd") {
              ret = -BigInt(EAGAIN);
              break;
            }
            if (readEntry.type === "epoll") {
              ret = -BigInt(EINVAL);
              break;
            }
            if (readEntry.type === "vfs") {
              ret = this.vfs ? BigInt(this.vfs.read(readEntry.fd, buf, count, view)) : -BigInt(EBADF);
              break;
            }
            if (readEntry.type === "native") {
              const rcount = Math.min(count, 12288);
              return {
                status: "bridge",
                method: "fs_read",
                args: { fd: readEntry.nativeFd, count: rcount },
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
              const scount = Math.min(count, 12288);
              return {
                status: "bridge",
                method: "net_read",
                args: { socketId: readEntry.socketId, count: scount },
                apply: (resp) => {
                  if (resp.ret < 0) {
                    this.nt(`read fd=${fd} req=${count} -> err ${resp.ret}`);
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                  } else if (resp.data) {
                    const decoded = Uint8Array.from(atob(resp.data), (c) => c.charCodeAt(0));
                    bytes.set(decoded, buf + GUEST_MEM_BASE);
                    cpu2.setReg(0, BigInt(decoded.length));
                  } else {
                    this.nt(`read fd=${fd} req=${count} -> EOF/0 (ret=${resp.ret})`);
                    cpu2.setReg(0, BigInt(resp.ret));
                  }
                }
              };
            }
            ret = -BigInt(EBADF);
            break;
          }
          if (this.vfs && this.vfs.hasFd(fd)) {
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
            if (writeEntry.type === "tty") {
              const data = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
              this.io.writeStdout(data);
              ret = BigInt(count);
              break;
            }
            if (writeEntry.type === "vfs") {
              ret = this.vfs ? BigInt(this.vfs.write(writeEntry.fd, buf, count, view)) : -BigInt(EBADF);
              break;
            }
            if (writeEntry.type === "native") {
              const writeData = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
              const b64 = bytesToBase64(writeData);
              this.statCache.delete(writeEntry.path);
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
              const sockB64 = bytesToBase64(sockWriteData);
              this.nt(`write fd=${fd} len=${count}`);
              return {
                status: "bridge",
                method: "net_write",
                args: { socketId: writeEntry.socketId, data: sockB64 },
                apply: (resp) => {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                }
              };
            }
            if (writeEntry.type === "eventfd") {
              if (count < 8) {
                ret = -BigInt(EINVAL);
                break;
              }
              const value = view.getBigUint64(buf + GUEST_MEM_BASE, true);
              if (value === 0xFFFFFFFFFFFFFFFFn) {
                ret = -BigInt(EINVAL);
                break;
              }
              const newCounter = writeEntry.counter + value;
              if (newCounter > 0xFFFFFFFFFFFFFFFEn) {
                ret = -BigInt(EAGAIN);
                break;
              }
              writeEntry.counter = newCounter;
              ret = 8n;
              break;
            }
            if (writeEntry.type === "inotify" || writeEntry.type === "timerfd" || writeEntry.type === "epoll") {
              ret = -BigInt(EBADF);
              break;
            }
            ret = -BigInt(EBADF);
            break;
          }
          if (this.vfs && this.vfs.hasFd(fd)) {
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
            const b64 = bytesToBase64(combined);
            this.nt(`writev fd=${fd} totalLen=${totalLen}`);
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
            const b64 = bytesToBase64(combined);
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
              } else if (writevEntry.type === "tty") {
                this.io.writeStdout(data);
              } else if (writevEntry.type === "vfs" && this.vfs) {
                this.vfs.write(writevEntry.fd, base, len, view);
              }
            } else if (this.vfs && this.vfs.hasFd(fd)) {
              this.vfs.write(fd, base, len, view);
            } else {
              ret = -BigInt(EBADF);
              break;
            }
            totalWritten += len;
          }
          if (ret !== -BigInt(EBADF)) ret = BigInt(totalWritten);
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
          const prEntry = this.fdTable.get(prFd);
          if (this.vfs && this.vfs.hasFd(prFd)) {
            const savedPos = this.vfs.lseek(prFd, 0n, 1);
            if (savedPos < 0n) {
              ret = savedPos;
              break;
            }
            this.vfs.lseek(prFd, prOffset, 0);
            const n = this.vfs.read(prFd, prBuf, prCount, view);
            this.vfs.lseek(prFd, savedPos, 0);
            ret = BigInt(n);
          } else if (prEntry?.type === "native" && this.io.readFileAt) {
            const data = this.io.readFileAt(prEntry.nativeFd, Number(prOffset), prCount);
            new Uint8Array(memory2.buffer).set(data, prBuf + GUEST_MEM_BASE);
            ret = BigInt(data.length);
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
          this.fdTable.set(readFd, { type: "pipe_read", pipe });
          const writeFd = this.allocFd();
          this.fdTable.set(writeFd, { type: "pipe_write", pipe });
          view.setInt32(pipefdAddr + GUEST_MEM_BASE, readFd, true);
          view.setInt32(pipefdAddr + GUEST_MEM_BASE + 4, writeFd, true);
          ret = 0n;
          break;
        }
        case 23: {
          const oldfd = Number(a0);
          const oldEntry = this.fdTable.get(oldfd);
          if (!oldEntry && !(this.vfs && this.vfs.hasFd(oldfd))) {
            ret = -BigInt(EBADF);
            break;
          }
          if (!this.fdAvailable()) {
            ret = -BigInt(EMFILE);
            break;
          }
          if (oldEntry) {
            const newfd = this.allocFd();
            if (oldEntry.type === "vfs" && this.vfs) {
              if (this.vfs.dupTo(oldEntry.fd, newfd) < 0) {
                ret = -BigInt(EBADF);
                break;
              }
            } else if (oldEntry.type === "native") {
              this.fdTable.set(newfd, { ...oldEntry });
              this.nativeRefs.set(oldEntry.nativeFd, (this.nativeRefs.get(oldEntry.nativeFd) ?? 1) + 1);
            } else {
              this.fdTable.set(newfd, { ...oldEntry });
            }
            ret = BigInt(newfd);
          } else if (this.vfs && this.vfs.hasFd(oldfd)) {
            const newfd = this.allocFd();
            if (this.vfs.dupTo(oldfd, newfd) < 0) {
              ret = -BigInt(EBADF);
              break;
            }
            ret = BigInt(newfd);
          } else {
            ret = -BigInt(EBADF);
          }
          this.ft(`dup ${oldfd} -> ${BigInt.asIntN(64, ret)}`);
          break;
        }
        case 24: {
          const oldfd = Number(a0);
          const newfd = Number(BigInt.asIntN(32, a1));
          const dup3flags = Number(a2);
          this.ft(`dup3 ${oldfd} -> ${newfd}`);
          if (oldfd === newfd) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (newfd < 0 || newfd >= MAX_FDS) {
            ret = -BigInt(EBADF);
            break;
          }
          const oldfdValid = oldfd >= 0 && (this.fdTable.has(oldfd) || (this.vfs?.hasFd(oldfd) ?? false));
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
          } else if (this.vfs && this.vfs.hasFd(newfd)) {
            this.vfs.close(newfd);
          }
          const srcEntry = this.fdTable.get(oldfd);
          if (srcEntry) {
            if (srcEntry.type === "vfs" && this.vfs) {
              if (this.vfs.dupTo(srcEntry.fd, newfd) < 0) {
                ret = -BigInt(EBADF);
                break;
              }
            } else if (srcEntry.type === "native") {
              this.fdTable.set(newfd, { ...srcEntry });
              this.nativeRefs.set(srcEntry.nativeFd, (this.nativeRefs.get(srcEntry.nativeFd) ?? 1) + 1);
            } else {
              this.fdTable.set(newfd, { ...srcEntry });
            }
            ret = BigInt(newfd);
          } else if (this.vfs && this.vfs.hasFd(oldfd)) {
            if (this.vfs.dupTo(oldfd, newfd) < 0) {
              ret = -BigInt(EBADF);
              break;
            }
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
          const path = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
          const flags56 = Number(a2);
          const mode56 = Number(a3);
          if (path === "/dev/tty") {
            const ttyFd = this.allocFd();
            this.fdTable.set(ttyFd, { type: "tty" });
            this.ft(`open ${path} -> ${ttyFd}`);
            ret = BigInt(ttyFd);
            break;
          }
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
                } else {
                  this.fdTable.set(guestFd, { type: "native", nativeFd: resp.ret, path: bp56 });
                  this.nativeRefs.set(resp.ret, 1);
                  this.ft(`open ${bp56} -> ${guestFd} (native ${resp.ret})`);
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
            this.ft(`open ${path} -> ${openRet} (vfs)`);
            ret = BigInt(openRet);
          } else {
            ret = -BigInt(ENOENT);
          }
          break;
        }
        case 57: {
          const fd = Number(a0);
          const closeEntry = this.fdTable.get(fd);
          this.ft(`close ${fd} (type=${closeEntry?.type ?? (this.vfs?.hasFd(fd) ? "vfs" : "none")})`);
          this.fdFlags.delete(fd);
          this.fdCloexec.delete(fd);
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
            if (closeEntry.type === "epoll") this.epollSets?.delete(fd);
            this.fdTable.delete(fd);
            ret = 0n;
          } else if (this.vfs && this.vfs.hasFd(fd)) {
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
            this.statCache.delete(ftEntry.path);
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
          if (lseekEntry) {
            switch (lseekEntry.type) {
              case "stdio":
              case "tty":
              case "pipe_read":
              case "pipe_write":
              case "inotify":
              case "timerfd":
              case "eventfd":
              case "epoll":
              case "socket":
                ret = -BigInt(ESPIPE);
                break;
              case "native":
                return {
                  status: "bridge",
                  method: "fs_lseek",
                  args: { fd: lseekEntry.nativeFd, offset: Number(a1), whence: Number(a2) },
                  apply: (resp) => {
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                  }
                };
              case "vfs":
                ret = this.vfs ? this.vfs.lseek(lseekEntry.fd, BigInt.asIntN(64, a1), Number(a2)) : -BigInt(EBADF);
                break;
            }
            break;
          }
          if (this.vfs && this.vfs.hasFd(lseekFd)) {
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
          if (this.ptrFaults(bytes, statBuf, 128)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const fstatEntry = this.getFdEntry(fd);
          if (fd <= 4 || fstatEntry?.type === "tty" || fstatEntry?.type === "stdio") this.ft(`fstat ${fd} type=${fstatEntry?.type ?? (this.vfs?.hasFd(fd) ? "vfs" : "none")}`);
          if (fstatEntry) {
            if (fstatEntry.type === "tty" && this.vfs) {
              ret = BigInt(this.vfs.fstatat(-100, "/dev/tty", statBuf, 0, view));
            } else if (fstatEntry.type === "stdio" || fstatEntry.type === "tty" || fstatEntry.type === "pipe_read" || fstatEntry.type === "pipe_write" || fstatEntry.type === "inotify" || fstatEntry.type === "timerfd" || fstatEntry.type === "eventfd" || fstatEntry.type === "epoll") {
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
          } else if (this.vfs && this.vfs.hasFd(fd)) {
            ret = BigInt(this.vfs.fstat(fd, statBuf, view));
          } else {
            ret = -BigInt(EBADF);
          }
          break;
        }
        case 79: {
          const path = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
          const statBuf79 = Number(a2);
          if (this.ptrFaults(bytes, statBuf79, 128)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const bp79 = this.bridgePath(path);
          if (bp79 !== null) {
            const cached = this.statCache.get(bp79);
            if (cached && cached.expiresAt > Date.now()) {
              writeBridgeStat(view, statBuf79, { _meta: [cached.mode, cached.size, cached.nlink, cached.mtime, cached.uid, cached.gid, cached.ino] });
              ret = 0n;
              break;
            }
            return {
              status: "bridge",
              method: "fs_stat",
              args: { path: bp79 },
              apply: (resp) => {
                if (resp.ret < 0) {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                } else {
                  writeBridgeStat(view, statBuf79, resp);
                  if (resp._meta) {
                    this.statCache.set(bp79, {
                      mode: resp._meta[0],
                      size: resp._meta[1],
                      nlink: resp._meta[2],
                      mtime: resp._meta[3],
                      uid: resp._meta[4] ?? 0,
                      gid: resp._meta[5] ?? 0,
                      ino: resp._meta[6] ?? 0,
                      expiresAt: Date.now() + this.STAT_CACHE_TTL_MS
                    });
                  }
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
            const dirPath = gdEntry.path;
            return {
              status: "bridge",
              method: "fs_getdents",
              args: { fd: gdEntry.nativeFd },
              apply: (resp) => {
                if (resp.ret <= 0) {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                  return;
                }
                let entries = [];
                if (resp.data) {
                  try {
                    const decoded = atob(resp.data);
                    entries = JSON.parse(decoded);
                  } catch {
                  }
                }
                const cacheExpiry = Date.now() + this.STAT_CACHE_TTL_MS;
                for (const entry of entries) {
                  if (entry.mode !== void 0) {
                    const childPath = dirPath === "/" ? "/" + entry.name : dirPath + "/" + entry.name;
                    this.statCache.set(childPath, {
                      mode: entry.mode,
                      size: entry.size ?? 0,
                      nlink: entry.type === 4 ? 2 : 1,
                      // DT_DIR=4
                      mtime: entry.mtime ?? 0,
                      uid: entry.uid ?? 0,
                      gid: entry.gid ?? 0,
                      ino: entry.ino ?? 0,
                      expiresAt: cacheExpiry
                    });
                  }
                }
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
          const path = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
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
          if (this.ptrFaults(bytes, Number(a1))) {
            ret = -BigInt(EFAULT);
            break;
          }
          const path = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
          if (path === "") {
            ret = -BigInt(ENOENT);
            break;
          }
          const bp35 = this.bridgePath(path);
          if (bp35 !== null) {
            this.statCache.delete(bp35);
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
          const path = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
          const mode = Number(a2);
          if (this.vfs) {
            const fmt = mode & 61440;
            if (fmt === 4096) {
              ret = BigInt(this.vfs.mkfifo(path, mode & 4095));
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
          if (this.ptrFaults(bytes, Number(a1)) || this.ptrFaults(bytes, Number(a3))) {
            ret = -BigInt(EFAULT);
            break;
          }
          const oldpath = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
          const newpath = this.resolveAt(a2, readGuestString(bytes, Number(a3)));
          const bOld = this.bridgePath(oldpath);
          if (bOld) {
            const bNew = this.bridgePath(newpath) ?? newpath;
            this.statCache.delete(bOld);
            this.statCache.delete(bNew);
            return {
              status: "bridge",
              method: "fs_rename",
              args: { oldPath: bOld, newPath: bNew },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          if (this.vfs) {
            ret = BigInt(this.vfs.rename(oldpath, newpath));
          } else {
            ret = -BigInt(ENOSYS);
          }
          break;
        }
        case 36: {
          if (this.ptrFaults(bytes, Number(a0)) || this.ptrFaults(bytes, Number(a2))) {
            ret = -BigInt(EFAULT);
            break;
          }
          const target = readGuestString(bytes, Number(a0));
          const linkpath = this.resolveAt(a1, readGuestString(bytes, Number(a2)));
          const bLink = this.bridgePath(linkpath);
          if (bLink) {
            return {
              status: "bridge",
              method: "fs_symlink",
              args: { target, linkpath: bLink },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          if (this.vfs) {
            ret = BigInt(this.vfs.symlinkat(target, linkpath));
          } else {
            ret = -BigInt(ENOSYS);
          }
          break;
        }
        case 37: {
          if (this.ptrFaults(bytes, Number(a1)) || this.ptrFaults(bytes, Number(a3))) {
            ret = -BigInt(EFAULT);
            break;
          }
          const oldpath = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
          const newpath = this.resolveAt(a2, readGuestString(bytes, Number(a3)));
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
          const path = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
          const buf = Number(a2);
          const bufsiz = Number(a3);
          this.ft(`readlinkat ${path}`);
          const pfd = path.match(/^\/proc\/(?:self|\d+)\/fd\/(\d+)$/);
          if (pfd) {
            const tfd = Number(pfd[1]);
            const te = this.getFdEntry(tfd);
            const link = te?.type === "tty" || te?.type === "stdio" ? "/dev/tty" : te?.path ?? (this.vfs?.fdPath?.(tfd) ?? null);
            this.ft(`readlink /proc/self/fd/${tfd} -> ${link ?? "NULL"} (type=${te?.type ?? "?"})`);
            if (link) {
              const enc = new TextEncoder().encode(link);
              const len = Math.min(enc.length, bufsiz);
              if (badGuestPtr(bytes, buf, len)) {
                ret = -BigInt(14);
                break;
              }
              bytes.set(enc.subarray(0, len), buf + GUEST_MEM_BASE);
              ret = BigInt(len);
              break;
            }
          }
          const bPath = this.bridgePath(path);
          if (bPath) {
            return {
              status: "bridge",
              method: "fs_readlink",
              args: { path: bPath },
              apply: (resp) => {
                if (resp.ret < 0) {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                } else if (resp.data) {
                  const decoded = Uint8Array.from(atob(resp.data), (c) => c.charCodeAt(0));
                  const len = Math.min(decoded.length, bufsiz);
                  if (badGuestPtr(bytes, buf, len)) {
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
                    return;
                  }
                  bytes.set(decoded.subarray(0, len), buf + GUEST_MEM_BASE);
                  cpu2.setReg(0, BigInt(len));
                } else {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(-EINVAL)));
                }
              }
            };
          }
          if (this.vfs) {
            const target = this.vfs.readlinkat(path);
            if (target === null) {
              ret = -BigInt(EINVAL);
            } else {
              const encoded = new TextEncoder().encode(target);
              const len = Math.min(encoded.length, bufsiz);
              if (badGuestPtr(bytes, buf, len)) {
                ret = -BigInt(14);
                break;
              }
              bytes.set(encoded.subarray(0, len), buf + GUEST_MEM_BASE);
              ret = BigInt(len);
            }
          } else {
            ret = -BigInt(ENOENT);
          }
          break;
        }
        case 48: {
          if (this.ptrFaults(bytes, Number(a1))) {
            ret = -BigInt(EFAULT);
            break;
          }
          const path = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
          const bp48 = this.bridgePath(path);
          if (bp48 !== null) {
            return {
              status: "bridge",
              method: "fs_faccessat",
              args: { path: bp48, mode: Number(a2) },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
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
        case 52: {
          const fchmodFd = Number(a0);
          const fchmodMode = Number(a1);
          const fchmodEntry = this.getFdEntry(fchmodFd);
          if (fchmodEntry?.type === "native") {
            this.statCache.delete(fchmodEntry.path);
            return {
              status: "bridge",
              method: "fs_chmod",
              args: { path: fchmodEntry.path, mode: fchmodMode },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          if (this.vfs && this.vfs.hasFd(fchmodFd)) ret = BigInt(this.vfs.fchmod(fchmodFd, fchmodMode));
          else if (this.fdTable.has(fchmodFd)) ret = 0n;
          else ret = -BigInt(EBADF);
          break;
        }
        case 53: {
          const fchmodatPath = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
          const fchmodatMode = Number(a2);
          const bpChmod = this.bridgePath(fchmodatPath);
          if (bpChmod !== null) {
            this.statCache.delete(bpChmod);
            return {
              status: "bridge",
              method: "fs_chmod",
              args: { path: bpChmod, mode: fchmodatMode },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          ret = this.vfs ? BigInt(this.vfs.chmodPath(fchmodatPath, fchmodatMode)) : -BigInt(ENOENT);
          break;
        }
        case 54: {
          const fchownFlags = Number(a4);
          const cOwner = Number(BigInt.asIntN(32, a2));
          const cGroup = Number(BigInt.asIntN(32, a3));
          if (fchownFlags & 4096) {
            const fchownFd = Number(a0);
            if (this.vfs && this.vfs.hasFd(fchownFd)) ret = BigInt(this.vfs.fchown(fchownFd, cOwner, cGroup));
            else if (this.fdTable.has(fchownFd)) ret = 0n;
            else ret = -BigInt(EBADF);
            break;
          }
          const fchownatPath = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
          const bpChown = this.bridgePath(fchownatPath);
          if (bpChown !== null) {
            return {
              status: "bridge",
              method: "fs_chown",
              args: { path: bpChown, uid: cOwner, gid: cGroup },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret ?? 0)));
              }
            };
          }
          ret = this.vfs ? BigInt(this.vfs.chownPath(fchownatPath, cOwner, cGroup)) : -BigInt(ENOENT);
          break;
        }
        case 55: {
          const fchownFd = Number(a0);
          const owner = Number(BigInt.asIntN(32, a1));
          const group = Number(BigInt.asIntN(32, a2));
          const ent = this.getFdEntry(fchownFd);
          if (ent?.type === "native") {
            this.statCache.delete(ent.path);
            return {
              status: "bridge",
              method: "fs_chown",
              args: { path: ent.path, uid: owner, gid: group },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret ?? 0)));
              }
            };
          }
          if (this.vfs && this.vfs.hasFd(fchownFd)) ret = BigInt(this.vfs.fchown(fchownFd, owner, group));
          else if (this.fdTable.has(fchownFd)) ret = 0n;
          else ret = -BigInt(EBADF);
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
          if (fd <= 4) this.ft(`ioctl ${fd} req=0x${request.toString(16)} type=${ioctlEntry?.type ?? (this.vfs?.hasFd(fd) ? "vfs" : "none")}`);
          const isStdio = ioctlEntry?.type === "stdio" || ioctlEntry?.type === "tty";
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
            if (needed > this.maxBytes) {
              ret = this.brk;
              break;
            }
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
          const mapType = flags & 15;
          if (length === 0 || mapType !== 1 && mapType !== 2 && mapType !== 3) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (mmapFd >= 0 && !(flags & MAP_ANONYMOUS) && this.vfs && this.vfs.hasFd(mmapFd)) {
            const am = this.vfs.fdAccessMode(mmapFd);
            const PROT_WRITE = 2;
            if (am === 1 || flags & MAP_SHARED && prot & PROT_WRITE && am === 0) {
              ret = -BigInt(EACCES);
              break;
            }
          }
          let addr;
          if (a0 !== 0n && flags & MAP_FIXED) {
            addr = a0;
            const end = a0 + BigInt(aligned);
            if (end > this.mmapBase) this.mmapBase = end;
          } else {
            addr = this.mmapBase;
            this.mmapBase += BigInt(aligned);
          }
          const needed = Number(addr) + aligned + GUEST_MEM_BASE;
          if (needed > this.maxBytes) {
            ret = -BigInt(ENOMEM);
            break;
          }
          if (needed > memory2.buffer.byteLength) {
            const pagesNeeded = Math.ceil(needed / 65536);
            const currentPages = memory2.buffer.byteLength / 65536;
            if (pagesNeeded > currentPages) memory2.grow(pagesNeeded - currentPages);
          }
          new Uint8Array(memory2.buffer).fill(0, Number(addr) + GUEST_MEM_BASE, Number(addr) + GUEST_MEM_BASE + aligned);
          if (mmapFd >= 0 && !(flags & MAP_ANONYMOUS)) {
            const fileOffset = Number(a5);
            const entry = this.fdTable.get(mmapFd);
            if (this.vfs && this.vfs.hasFd(mmapFd)) {
              const savedPos = this.vfs.lseek(mmapFd, 0n, 1);
              if (savedPos >= 0n) {
                this.vfs.lseek(mmapFd, BigInt(fileOffset), 0);
                this.vfs.read(mmapFd, Number(addr), aligned, new DataView(memory2.buffer));
                this.vfs.lseek(mmapFd, savedPos, 0);
              }
            } else if (entry?.type === "native" && this.io.readFileAt) {
              const data = this.io.readFileAt(entry.nativeFd, fileOffset, aligned);
              if (data.length > 0) new Uint8Array(memory2.buffer).set(data, Number(addr) + GUEST_MEM_BASE);
              if (flags & MAP_SHARED && prot & 2) {
                this.fileMmaps.push({ start: Number(addr), len: aligned, fileLen: length, nativeFd: entry.nativeFd, fileOffset });
              }
            }
          }
          if (flags & MAP_SHARED) this.sharedRegions.push({ addr: Number(addr), len: aligned });
          const mmStart = Number(addr), mmEnd = mmStart + aligned;
          this.protNone = this.protNone.filter((r) => r.end <= mmStart || r.start >= mmEnd);
          if ((prot & 7) === 0) this.protNone.push({ start: mmStart, end: mmEnd });
          ret = addr;
          break;
        }
        case 226: {
          const mpStart = Number(a0), mpEnd = mpStart + (Number(a1) + 4095 & ~4095);
          this.protNone = this.protNone.filter((r) => r.end <= mpStart || r.start >= mpEnd);
          if ((Number(a2) & 7) === 0) this.protNone.push({ start: mpStart, end: mpEnd });
          ret = 0n;
          break;
        }
        case 215: {
          const unmapLen = Number(a1);
          const guestMax = BigInt(memory2.buffer.byteLength - GUEST_MEM_BASE);
          if (unmapLen === 0 || (a0 & 0xFFFn) !== 0n || a0 >= guestMax || a0 + BigInt(unmapLen) > guestMax) {
            ret = -BigInt(EINVAL);
            break;
          }
          const umStart = Number(a0), umEnd = umStart + (unmapLen + 4095 & ~4095);
          this.protNone = this.protNone.filter((r) => r.end <= umStart || r.start >= umEnd);
          const umFlush = this.buildMmapFlush(cpu2, memory2, umStart, unmapLen + 4095 & ~4095, true);
          if (umFlush) return umFlush;
          ret = 0n;
          break;
        }
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
          ret = BigInt(this.tgid || this.currentPid);
          break;
        case 173:
          ret = BigInt(this.currentPpid);
          break;
        case 178:
          ret = BigInt(this.currentPid);
          break;
        case 174:
          ret = BigInt(this.ruid);
          break;
        case 175:
          ret = BigInt(this.euid);
          break;
        case 176:
          ret = BigInt(this.rgid);
          break;
        case 177:
          ret = BigInt(this.egid);
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
          const bPath = this.bridgePath(path);
          if (bPath) {
            return {
              status: "bridge",
              method: "fs_stat",
              args: { path: bPath },
              apply: (resp) => {
                if (resp.ret < 0) {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                  return;
                }
                const mode = resp._meta ? resp._meta[0] : resp.mode;
                if (mode && (mode & 61440) === 16384) {
                  this.cwd = bPath;
                  if (this.vfs) this.vfs.setCwd(bPath);
                  cpu2.setReg(0, 0n);
                } else {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(-ENOTDIR)));
                }
              }
            };
          }
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
          if (badGuestPtr(new Uint8Array(memory2.buffer), Number(a0), 6 * 65)) {
            ret = -BigInt(14);
            break;
          }
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
          const ns = this.clockNs(Number(a0));
          view.setBigUint64(tp, ns / 1000000000n, true);
          view.setBigUint64(tp + 8, ns % 1000000000n, true);
          ret = 0n;
          break;
        }
        case 169: {
          if (badGuestPtr(new Uint8Array(memory2.buffer), Number(a0), 16)) {
            ret = -BigInt(14);
            break;
          }
          const tv = Number(a0) + GUEST_MEM_BASE;
          const ns = this.clockNs(0);
          view.setBigUint64(tv, ns / 1000000000n, true);
          view.setBigUint64(tv + 8, ns % 1000000000n / 1000n, true);
          ret = 0n;
          break;
        }
        // ── fcntl ──
        case 25: {
          const fd = Number(a0);
          const cmd = Number(a1);
          if (fd <= 4) this.ft(`fcntl ${fd} cmd=${cmd}`);
          switch (cmd) {
            case F_DUPFD:
            case F_DUPFD_CLOEXEC: {
              const minfd = Number(a2);
              if (!this.fdAvailable(minfd)) {
                ret = -BigInt(EMFILE);
                break;
              }
              const newfd = this.lowestFreeFd(minfd);
              const srcEntry = this.fdTable.get(fd);
              if (srcEntry) {
                if (srcEntry.type === "vfs" && this.vfs) {
                  if (this.vfs.dupTo(srcEntry.fd, newfd) < 0) {
                    ret = -BigInt(EBADF);
                    break;
                  }
                } else if (srcEntry.type === "native") {
                  this.fdTable.set(newfd, { ...srcEntry });
                  this.nativeRefs.set(srcEntry.nativeFd, (this.nativeRefs.get(srcEntry.nativeFd) ?? 1) + 1);
                } else {
                  this.fdTable.set(newfd, { ...srcEntry });
                }
              } else if (this.vfs && this.vfs.hasFd(fd)) {
                if (this.vfs.dupTo(fd, newfd) < 0) {
                  ret = -BigInt(EBADF);
                  break;
                }
              } else {
                ret = -BigInt(EBADF);
                break;
              }
              if (cmd === F_DUPFD_CLOEXEC) this.fdCloexec.add(newfd);
              else this.fdCloexec.delete(newfd);
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
        case 146: {
          const uid = Number(BigInt.asIntN(32, a0));
          if (this.euid === 0) {
            this.ruid = this.euid = this.suid = uid;
            ret = 0n;
          } else if (uid === this.ruid || uid === this.suid) {
            this.euid = uid;
            ret = 0n;
          } else ret = -BigInt(EPERM);
          break;
        }
        case 144: {
          const gid = Number(BigInt.asIntN(32, a0));
          if (this.euid === 0) {
            this.rgid = this.egid = this.sgid = gid;
            ret = 0n;
          } else if (gid === this.rgid || gid === this.sgid) {
            this.egid = gid;
            ret = 0n;
          } else ret = -BigInt(EPERM);
          break;
        }
        case 145: {
          const r = Number(BigInt.asIntN(32, a0)), e = Number(BigInt.asIntN(32, a1));
          const unchanged = 4294967295;
          if (this.euid !== 0) {
            if (r !== -1 && r >>> 0 !== unchanged && r !== this.ruid && r !== this.euid || e !== -1 && e >>> 0 !== unchanged && e !== this.ruid && e !== this.euid && e !== this.suid) {
              ret = -BigInt(EPERM);
              break;
            }
          }
          if (r !== -1 && r >>> 0 !== unchanged) this.ruid = r;
          if (e !== -1 && e >>> 0 !== unchanged) this.euid = e;
          if (r !== -1 && r >>> 0 !== unchanged) this.suid = this.euid;
          ret = 0n;
          break;
        }
        case 147: {
          const r = Number(BigInt.asIntN(32, a0)), e = Number(BigInt.asIntN(32, a1)), s = Number(BigInt.asIntN(32, a2));
          const set = (v) => v !== -1 && v >>> 0 !== 4294967295;
          if (this.euid !== 0) {
            const allowed = (v) => v === this.ruid || v === this.euid || v === this.suid;
            if (set(r) && !allowed(r) || set(e) && !allowed(e) || set(s) && !allowed(s)) {
              ret = -BigInt(EPERM);
              break;
            }
          }
          if (set(r)) this.ruid = r;
          if (set(e)) this.euid = e;
          if (set(s)) this.suid = s;
          ret = 0n;
          break;
        }
        case 143: {
          const r = Number(BigInt.asIntN(32, a0)), e = Number(BigInt.asIntN(32, a1));
          if (r !== -1 && r >>> 0 !== 4294967295) this.rgid = r;
          if (e !== -1 && e >>> 0 !== 4294967295) this.egid = e;
          ret = 0n;
          break;
        }
        case 149: {
          const set = (v) => v !== -1 && v >>> 0 !== 4294967295;
          const r = Number(BigInt.asIntN(32, a0)), e = Number(BigInt.asIntN(32, a1)), s = Number(BigInt.asIntN(32, a2));
          if (set(r)) this.rgid = r;
          if (set(e)) this.egid = e;
          if (set(s)) this.sgid = s;
          ret = 0n;
          break;
        }
        case 151:
          ret = BigInt(this.euid);
          break;
        case 152:
          ret = BigInt(this.egid);
          break;
        case 159:
          ret = 0n;
          break;
        // ── Networking (via Swift bridge) ──
        case 198: {
          const domain = Number(a0);
          const sockType = Number(a1) & 255;
          const protocol = Number(a2);
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
              } else {
                this.fdTable.set(socketFd, { type: "socket", socketId: resp.ret, domain, sockType });
                if (Number(a1) & 524288) this.fdCloexec.add(socketFd);
                this.nt(`socket dom=${domain} type=${sockType} -> fd=${socketFd}`);
                this._netActive = true;
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
          connEntry.remoteAddr = connAddr;
          return {
            status: "bridge",
            method: "net_connect",
            args: { socketId: connEntry.socketId, ...connAddr },
            apply: (resp) => {
              this.nt(`connect fd=${Number(a0)} ${connAddr.host}:${connAddr.port} -> ${resp.ret}`);
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
          const b64 = bytesToBase64(sendData);
          const destAddrPtr = Number(a4);
          const destAddrLen = Number(a5);
          if (destAddrPtr !== 0 && destAddrLen > 0) {
            const dest = this.parseSockaddr(bytes, destAddrPtr, destAddrLen);
            if (!stEntry.remoteAddr) stEntry.remoteAddr = dest;
            return {
              status: "bridge",
              method: "net_sendto",
              args: { socketId: stEntry.socketId, data: b64, host: dest.host, port: dest.port },
              apply: (resp) => {
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
          const recvLen = Math.min(Number(a2), 12288);
          return {
            status: "bridge",
            method: "net_read",
            args: { socketId: rfEntry.socketId, count: recvLen },
            apply: (resp) => {
              if (resp.ret < 0) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              } else if (resp.data) {
                const decoded = Uint8Array.from(atob(resp.data), (c) => c.charCodeAt(0));
                bytes.set(decoded, recvBuf + GUEST_MEM_BASE);
                cpu2.setReg(0, BigInt(decoded.length));
              } else {
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
          const resource = Number(a1);
          const rlNewPtr = Number(a2);
          const rlOldPtr = Number(a3);
          const RLIM_INFINITY = 0xFFFFFFFFFFFFFFFFn;
          if (rlOldPtr !== 0) {
            const base = rlOldPtr + GUEST_MEM_BASE;
            switch (resource) {
              case 7:
                view.setBigUint64(base, BigInt(this.nofileLimit), true);
                view.setBigUint64(base + 8, 1048576n, true);
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
          if (rlNewPtr !== 0 && resource === 7) {
            const cur = view.getBigUint64(rlNewPtr + GUEST_MEM_BASE, true);
            this.nofileLimit = cur >= 1048576n ? 1048576 : Number(cur);
          }
          ret = 0n;
          break;
        }
        case 233: {
          const advice = Number(a2);
          if (advice === 4 || advice === 8) {
            const start = Number(a0), len = Number(a1);
            const end = start + len;
            if (len > 0 && (start & 4095) === 0 && end + GUEST_MEM_BASE <= memory2.buffer.byteLength) {
              new Uint8Array(memory2.buffer).fill(0, start + GUEST_MEM_BASE, end + GUEST_MEM_BASE);
            }
          }
          ret = 0n;
          break;
        }
        // ── Missing syscalls (spec-compliant implementations) ──
        case 50: {
          const fchEntry = this.getFdEntry(Number(a0));
          const fchPath = fchEntry?.path;
          if (typeof fchPath === "string" && fchPath.length) {
            this.cwd = fchPath;
            if (this.vfs) this.vfs.setCwd(fchPath);
            ret = 0n;
          } else {
            ret = -BigInt(EBADF);
          }
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
              if (badGuestPtr(new Uint8Array(memory2.buffer), Number(a1), 16)) {
                ret = -BigInt(14);
                break;
              }
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
          const which = Number(a0);
          const remMs = which === 0 ? this.timerRemainingMs() : 0;
          const intMs = which === 0 ? this.itimerRealIntervalMs : 0;
          view.setBigUint64(itvPtr, BigInt(Math.floor(intMs / 1e3)), true);
          view.setBigUint64(itvPtr + 8, BigInt(intMs % 1e3 * 1e3), true);
          view.setBigUint64(itvPtr + 16, BigInt(Math.floor(remMs / 1e3)), true);
          view.setBigUint64(itvPtr + 24, BigInt(remMs % 1e3 * 1e3), true);
          ret = 0n;
          break;
        }
        case 103: {
          const which = Number(a0);
          const oldPtr = Number(a2);
          if (oldPtr !== 0) {
            const base = oldPtr + GUEST_MEM_BASE;
            const remMs = which === 0 ? this.timerRemainingMs() : 0;
            const intMs = which === 0 ? this.itimerRealIntervalMs : 0;
            view.setBigUint64(base, BigInt(Math.floor(intMs / 1e3)), true);
            view.setBigUint64(base + 8, BigInt(intMs % 1e3 * 1e3), true);
            view.setBigUint64(base + 16, BigInt(Math.floor(remMs / 1e3)), true);
            view.setBigUint64(base + 24, BigInt(remMs % 1e3 * 1e3), true);
          }
          if (which === 0) {
            const newPtr = Number(a1);
            if (newPtr === 0) {
              this.itimerRealDeadlineMs = 0;
              this.itimerRealIntervalMs = 0;
              ret = 0n;
              break;
            }
            const nb = newPtr + GUEST_MEM_BASE;
            const intMs = Number(view.getBigUint64(nb, true)) * 1e3 + Math.floor(Number(view.getBigUint64(nb + 8, true)) / 1e3);
            const valMs = Number(view.getBigUint64(nb + 16, true)) * 1e3 + Math.floor(Number(view.getBigUint64(nb + 24, true)) / 1e3);
            this.itimerRealIntervalMs = intMs;
            this.itimerRealDeadlineMs = valMs > 0 ? Number(this.nowMs()) + valMs : 0;
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
          this.fdTable.set(inoFd, { type: "inotify" });
          this.fdFlags.set(inoFd, Number(a0) & O_NONBLOCK);
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
          this.fdTable.set(tfFd, { type: "timerfd" });
          this.fdFlags.set(tfFd, Number(a1) & O_NONBLOCK);
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
        case 227: {
          const msFlush = this.buildMmapFlush(cpu2, memory2, Number(a0), Number(a1), false);
          if (msFlush) return msFlush;
          ret = 0n;
          break;
        }
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
          const MREMAP_MAYMOVE = 1;
          const oldAddr = Number(a0), oldSize = Number(a1), newSize = Number(a2);
          const mrFlags = Number(a3);
          const alignedOld = oldSize + 4095 & ~4095;
          const alignedNew = newSize + 4095 & ~4095;
          if (alignedNew <= alignedOld) {
            ret = a0;
            break;
          }
          if (BigInt(oldAddr + alignedOld) === this.mmapBase) {
            const needed2 = oldAddr + alignedNew + GUEST_MEM_BASE;
            if (needed2 > this.maxBytes) {
              ret = -BigInt(ENOMEM);
              break;
            }
            if (needed2 > memory2.buffer.byteLength) {
              const pagesNeeded = Math.ceil(needed2 / 65536), currentPages = memory2.buffer.byteLength / 65536;
              if (pagesNeeded > currentPages) memory2.grow(pagesNeeded - currentPages);
            }
            new Uint8Array(memory2.buffer).fill(0, oldAddr + alignedOld + GUEST_MEM_BASE, oldAddr + alignedNew + GUEST_MEM_BASE);
            this.mmapBase = BigInt(oldAddr + alignedNew);
            ret = a0;
            break;
          }
          if (!(mrFlags & MREMAP_MAYMOVE)) {
            ret = -BigInt(ENOMEM);
            break;
          }
          const newAddr = this.mmapBase;
          this.mmapBase += BigInt(alignedNew);
          const needed = Number(newAddr) + alignedNew + GUEST_MEM_BASE;
          if (needed > this.maxBytes) {
            this.mmapBase -= BigInt(alignedNew);
            ret = -BigInt(ENOMEM);
            break;
          }
          if (needed > memory2.buffer.byteLength) {
            const pagesNeeded = Math.ceil(needed / 65536), currentPages = memory2.buffer.byteLength / 65536;
            if (pagesNeeded > currentPages) memory2.grow(pagesNeeded - currentPages);
          }
          const mrMem = new Uint8Array(memory2.buffer);
          mrMem.copyWithin(Number(newAddr) + GUEST_MEM_BASE, oldAddr + GUEST_MEM_BASE, oldAddr + GUEST_MEM_BASE + oldSize);
          mrMem.fill(0, Number(newAddr) + GUEST_MEM_BASE + oldSize, Number(newAddr) + GUEST_MEM_BASE + alignedNew);
          ret = newAddr;
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
          const pathname281 = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
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
          return { status: "execve", path: this.absolutizePath(pathname281), argv: argvPtrs281, envp: envpPtrs281 };
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
          const oldpath276 = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
          const newpath276 = this.resolveAt(a2, readGuestString(bytes, Number(a3)));
          const bOld276 = this.bridgePath(oldpath276);
          if (bOld276) {
            const bNew276 = this.bridgePath(newpath276) ?? newpath276;
            this.statCache.delete(bOld276);
            this.statCache.delete(bNew276);
            return {
              status: "bridge",
              method: "fs_rename",
              args: { oldPath: bOld276, newPath: bNew276 },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          if (this.vfs) {
            ret = BigInt(this.vfs.rename(oldpath276, newpath276));
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
          const sxPath = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
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
          const initVal = BigInt.asUintN(32, a0);
          const efdFlags = Number(a1);
          const EFD_SEMAPHORE = 1;
          const EFD_NONBLOCK = O_NONBLOCK;
          const efd = this.allocFd();
          this.fdTable.set(efd, {
            type: "eventfd",
            counter: initVal,
            semaphore: (efdFlags & EFD_SEMAPHORE) !== 0
          });
          this.fdFlags.set(efd, efdFlags & EFD_NONBLOCK);
          ret = BigInt(efd);
          break;
        }
        case 20: {
          const epfd = this.allocFd();
          this.fdTable.set(epfd, { type: "epoll" });
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
          const readPtr = Number(a1);
          const writePtr = Number(a2);
          const exceptPtr = Number(a3);
          const tmoPtr = Number(a4);
          let tmoMs = -1;
          if (tmoPtr !== 0) {
            const sec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE, true));
            const nsec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE + 8, true));
            tmoMs = sec * 1e3 + Math.floor(nsec / 1e6);
          }
          const wantRead = this.readFdSet(view, readPtr, nfds);
          const wantWrite = this.readFdSet(view, writePtr, nfds);
          const sockReadFd = [...wantRead].find((fd) => this.getFdEntry(fd)?.type === "socket");
          if (sockReadFd !== void 0 && tmoMs !== 0) {
            let preReady = 0;
            for (const fd of wantRead) if (this.getFdEntry(fd)?.type !== "socket" && this.isFdReadable(fd)) preReady++;
            for (const fd of wantWrite) if (this.isFdWritable(fd)) preReady++;
            if (preReady === 0) {
              const sockEntry = this.getFdEntry(sockReadFd);
              const pollTmo = tmoMs < 0 ? 3e4 : tmoMs;
              return {
                status: "bridge",
                method: "net_poll",
                args: { socketId: sockEntry.socketId, timeoutMs: pollTmo },
                apply: (resp) => {
                  const outR = /* @__PURE__ */ new Set(), outW = /* @__PURE__ */ new Set();
                  for (const fd of wantWrite) if (this.isFdWritable(fd)) outW.add(fd);
                  for (const fd of wantRead) {
                    const t = this.getFdEntry(fd)?.type;
                    if (t === "socket") {
                      if (resp.ret > 0) outR.add(fd);
                    } else if (this.isFdReadable(fd)) outR.add(fd);
                  }
                  this.writeFdSet(view, readPtr, nfds, outR);
                  this.writeFdSet(view, writePtr, nfds, outW);
                  if (exceptPtr !== 0) this.writeFdSet(view, exceptPtr, nfds, /* @__PURE__ */ new Set());
                  this.nt(`pselect.sock -> ret=${resp.ret} r=${outR.size} w=${outW.size}`);
                  cpu2.setReg(0, BigInt(outR.size + outW.size));
                }
              };
            }
          }
          const readyRead = /* @__PURE__ */ new Set();
          const readyWrite = /* @__PURE__ */ new Set();
          let deadline = tmoMs < 0 ? -1 : Date.now() + tmoMs;
          while (true) {
            readyRead.clear();
            readyWrite.clear();
            for (const fd of wantRead) if (this.isFdReadable(fd)) readyRead.add(fd);
            for (const fd of wantWrite) if (this.isFdWritable(fd)) readyWrite.add(fd);
            const total = readyRead.size + readyWrite.size;
            if (total > 0) {
              this.writeFdSet(view, readPtr, nfds, readyRead);
              this.writeFdSet(view, writePtr, nfds, readyWrite);
              if (exceptPtr !== 0) this.writeFdSet(view, exceptPtr, nfds, /* @__PURE__ */ new Set());
              ret = BigInt(total);
              break;
            }
            const remaining = deadline < 0 ? -1 : Math.max(0, deadline - Date.now());
            if (deadline >= 0 && remaining === 0) {
              this.writeFdSet(view, readPtr, nfds, readyRead);
              this.writeFdSet(view, writePtr, nfds, readyWrite);
              if (exceptPtr !== 0) this.writeFdSet(view, exceptPtr, nfds, /* @__PURE__ */ new Set());
              ret = 0n;
              break;
            }
            let wakeable = -1;
            for (const fd of wantRead) if (this.isFdWakeable(fd)) {
              wakeable = fd;
              break;
            }
            if (wakeable < 0 || !this.io.waitStdin) {
              this.writeFdSet(view, readPtr, nfds, readyRead);
              this.writeFdSet(view, writePtr, nfds, readyWrite);
              if (exceptPtr !== 0) this.writeFdSet(view, exceptPtr, nfds, /* @__PURE__ */ new Set());
              ret = 0n;
              break;
            }
            const status = this.io.waitStdin(remaining);
            if (status === "timeout" && deadline >= 0) {
              this.writeFdSet(view, readPtr, nfds, readyRead);
              this.writeFdSet(view, writePtr, nfds, readyWrite);
              if (exceptPtr !== 0) this.writeFdSet(view, exceptPtr, nfds, /* @__PURE__ */ new Set());
              ret = 0n;
              break;
            }
          }
          break;
        }
        case 73: {
          const POLLIN = 1, POLLOUT = 4;
          const fdsAddr = Number(a0);
          const nfds = Number(a1);
          const tmoPtr = Number(a2);
          let tmoMs = -1;
          if (tmoPtr !== 0) {
            const sec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE, true));
            const nsec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE + 8, true));
            tmoMs = sec * 1e3 + Math.floor(nsec / 1e6);
          }
          const pfds = [];
          for (let i = 0; i < nfds; i++) {
            const off = fdsAddr + i * 8 + GUEST_MEM_BASE;
            pfds.push({
              fd: view.getInt32(off, true),
              events: view.getInt16(off + 4, true),
              revents: 0
            });
          }
          const writeBack = () => {
            for (let i = 0; i < nfds; i++) {
              view.setInt16(fdsAddr + i * 8 + 6 + GUEST_MEM_BASE, pfds[i].revents, true);
            }
          };
          let socketEntry = null;
          for (const pfd of pfds) {
            if (pfd.fd < 0) continue;
            const entry = this.getFdEntry(pfd.fd);
            if (entry?.type === "socket" && pfd.events & POLLIN) {
              if (!socketEntry) socketEntry = { fd: pfd.fd, entry };
            }
          }
          const checkReady = () => {
            let count = 0;
            for (const pfd of pfds) {
              pfd.revents = 0;
              if (pfd.fd < 0) continue;
              const entry = this.getFdEntry(pfd.fd);
              if (entry?.type === "socket") continue;
              if (pfd.events & POLLIN && this.isFdReadable(pfd.fd)) pfd.revents |= POLLIN;
              if (pfd.events & POLLOUT && this.isFdWritable(pfd.fd)) pfd.revents |= POLLOUT;
              if (pfd.revents !== 0) count++;
            }
            return count;
          };
          const watchingLocal = pfds.some((p) => p.fd >= 0 && this.getFdEntry(p.fd)?.type !== "socket");
          if (!watchingLocal && socketEntry) {
            const pollTmo = tmoMs < 0 ? 3e4 : tmoMs;
            this.nt(`ppoll.enter sock=${socketEntry.fd} events=${socketEntry.entry ? pfds.find((p) => p.fd === socketEntry.fd)?.events : "?"} tmo=${tmoMs}`);
            return {
              status: "bridge",
              method: "net_poll",
              args: { socketId: socketEntry.entry.socketId, timeoutMs: pollTmo },
              apply: (resp) => {
                if (resp.ret > 0) {
                  for (const pfd of pfds) {
                    if (pfd.fd < 0) continue;
                    const e = this.getFdEntry(pfd.fd);
                    if (e?.type === "socket" && pfd.events & POLLIN) pfd.revents |= POLLIN;
                  }
                }
                writeBack();
                const c = pfds.filter((p) => p.revents !== 0).length;
                this.nt(`ppoll.sock -> ret=${resp.ret} ready=${c}`);
                cpu2.setReg(0, BigInt(c));
              }
            };
          }
          let deadline = tmoMs < 0 ? -1 : Date.now() + tmoMs;
          while (true) {
            const ready = checkReady();
            if (ready > 0) {
              writeBack();
              ret = BigInt(ready);
              break;
            }
            const remaining = deadline < 0 ? -1 : Math.max(0, deadline - Date.now());
            if (deadline >= 0 && remaining === 0) {
              writeBack();
              ret = 0n;
              break;
            }
            const wakeable = pfds.some((p) => p.fd >= 0 && p.events & POLLIN && this.isFdWakeable(p.fd));
            if (!wakeable || !this.io.waitStdin) {
              writeBack();
              ret = 0n;
              break;
            }
            const status = this.io.waitStdin(remaining);
            if (status === "timeout" && deadline >= 0) {
              writeBack();
              ret = 0n;
              break;
            }
          }
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
          const smB64 = bytesToBase64(combined);
          this.nt(`sendmsg fd=${Number(a0)} len=${totalLen}`);
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
          return {
            status: "bridge",
            method: "net_read",
            args: { socketId: rmEntry.socketId, count: Math.min(rmCapacity, 12288) },
            apply: (resp) => {
              this.nt(`recvmsg fd=${Number(a0)} cap=${rmCapacity} -> ret=${resp.ret} dlen=${resp.data ? Math.floor(resp.data.length * 3 / 4) : 0}`);
              if (resp.ret < 0) {
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
          const path437 = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
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
          if (this.ptrFaults(bytes, Number(a1))) {
            ret = -BigInt(EFAULT);
            break;
          }
          const fa2Path = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
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
  function badGuestPtr(memory2, guestAddr, len = 1) {
    if (!Number.isFinite(guestAddr) || guestAddr < 4096) return true;
    const max = memory2.length - GUEST_MEM_BASE;
    return guestAddr >= max || guestAddr + len > max;
  }
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
    const meta = resp._meta;
    const mode = meta ? meta[0] : resp.mode ?? 33188;
    const size = meta ? meta[1] : resp.size ?? 0;
    const nlink = meta ? meta[2] : resp.nlink ?? 1;
    const mtime = meta ? meta[3] : resp.mtime ?? 0;
    const uid = meta ? meta[4] ?? 0 : resp.uid ?? 0;
    const gid = meta ? meta[5] ?? 0 : resp.gid ?? 0;
    const ino = meta ? meta[6] ?? 0 : resp.ino ?? 0;
    view.setBigUint64(base + 0, 1n, true);
    view.setBigUint64(base + 8, BigInt.asUintN(64, BigInt(Math.trunc(ino))), true);
    view.setUint32(base + 16, mode, true);
    view.setUint32(base + 20, nlink, true);
    view.setUint32(base + 24, uid, true);
    view.setUint32(base + 28, gid, true);
    view.setBigUint64(base + 48, BigInt(size), true);
    view.setBigUint64(base + 88, BigInt(mtime), true);
  }

  // src/vfs.ts
  var EPERM2 = 1;
  var ENOENT2 = 2;
  var EBADF2 = 9;
  var EACCES2 = 13;
  var EEXIST = 17;
  var ENOTDIR2 = 20;
  var EISDIR = 21;
  var EINVAL2 = 22;
  var EMFILE2 = 24;
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
  var S_IFIFO = 4096;
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
    // when set (by the syscall handler), fd numbers are drawn from the shared guest
    // namespace so vfs fds never collide with pipe/native/socket fds. left null for
    // standalone use, where we fall back to our own lowest-free scan over openFds.
    fdAllocator = null;
    // credentials of the calling process, injected by the syscall handler for permission
    // checks. null (standalone/tests) means "act as root" (all access granted).
    credsProvider = null;
    // path-resolution search-permission state. resolve() enforces +x (search) on each directory
    // component; searchDenied records when a walk failed on a permission (EACCES) vs a missing
    // entry (ENOENT). searchEff selects effective vs real creds for the walk (access() uses real
    // by default, most syscalls use effective). set per-resolve.
    searchDenied = false;
    searchEff = true;
    // errno for the most recent failed resolve() (ENOENT/ENOTDIR/EACCES/ELOOP/ENAMETOOLONG).
    // callers that get a null inode use resolveErrno() to return the correct negative errno.
    lastResolveErrno = ENOENT2;
    nextIno = 1;
    // inode counter
    cwd = "/";
    cwdInode;
    constructor(opts = {}) {
      this.root = this.makeDir(null, 493);
      this.cwdInode = this.root;
      this.populateVirtual();
      if (!opts.bridgeHandlesRealFs) {
        this.populateNativeFallback();
      }
    }
    // ── Inode creation helpers ──────────────────────────────────────────────
    now() {
      return Math.floor(Date.now() / 1e3);
    }
    // stamp a freshly created inode with its creator's ownership. new files take the
    // creating process's euid; the gid is inherited from a set-group-ID parent directory,
    // otherwise the creator's egid. a new directory under a sgid parent is itself sgid.
    applyCreatorOwnership(inode, parent) {
      const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
      inode.uid = c.euid;
      inode.gid = parent.mode & 1024 ? parent.gid : c.egid;
      if (inode.type === "dir" && parent.mode & 1024) inode.mode |= 1024;
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
    makeFifo(mode) {
      const t = this.now();
      return { type: "fifo", ino: this.nextIno++, mode: S_IFIFO | mode & 4095, nlink: 1, uid: 0, gid: 0, atime: t, mtime: t, ctime: t };
    }
    mkfifo(path, mode) {
      const parent = this.resolveParent(path);
      if (!parent) return this.resolveErrno();
      const [dir, name] = parent;
      if (dir.children.has(name)) return -EEXIST;
      if (!this.canWrite(dir)) return -EACCES2;
      const fifo = this.makeFifo(mode);
      this.applyCreatorOwnership(fifo, dir);
      dir.children.set(name, fifo);
      dir.mtime = dir.ctime = this.now();
      return 0;
    }
    // ── Default filesystem population ───────────────────────────────────────
    populateVirtual() {
      this.mkdirp("/dev");
      this.mkdirp("/proc/self/fd");
      this.mkdirp("/proc/sys/kernel");
      this.mkdirp("/proc/sys/fs");
      this.addCharDev("/dev/null", 1, 438);
      this.addCharDev("/dev/zero", 2, 438);
      this.addCharDev("/dev/tty", 3, 438);
      this.addCharDev("/dev/urandom", 4, 292);
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
    /// in-memory backing for real paths, used by tests and non-bridged hosts.
    /// on ipad these paths live on disk via NativeFS instead
    populateNativeFallback() {
      this.mkdirp("/etc");
      this.mkdirp("/tmp");
      this.mkdirp("/root");
      this.mkdirp("/bin");
      this.mkdirp("/home");
      this.mkdirp("/var");
      this.addFileContent(
        "/etc/passwd",
        "root:x:0:0:root:/root:/bin/sh\ndaemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin\nnobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n"
      );
      this.addFileContent("/etc/group", "root:x:0:\ndaemon:x:1:\nnobody:x:65534:\nnogroup:x:65534:\n");
      this.addFileContent("/etc/hostname", "jitter\n");
      this.addFileContent("/etc/shells", "/bin/sh\n");
      this.addFileContent("/etc/resolv.conf", "nameserver 1.1.1.1\nnameserver 1.0.0.1\n");
      this.addFileContent("/etc/nsswitch.conf", "hosts: files dns\n");
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
    // public: inject a file (creating parent directories) with raw bytes. used to seed
    // the rootfs with binaries like /lib/ld-linux-aarch64.so.1 and /lib/libc.so.6.
    writeFileBytes(path, data, mode = 493) {
      const parts = path.split("/").filter((p) => p.length > 0);
      parts.pop();
      let dirPath = "";
      for (const part of parts) {
        dirPath += "/" + part;
        this.mkdirp(dirPath);
      }
      this.setFileData(path, data, mode);
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
    resolve(path, followFinal = true, depth = 0, searchEff = true) {
      if (depth === 0) {
        this.searchDenied = false;
        this.searchEff = searchEff;
        this.lastResolveErrno = ENOENT2;
      }
      if (depth > MAX_SYMLINK_DEPTH) {
        this.lastResolveErrno = ELOOP;
        return null;
      }
      const parts = this.normalizeParts(path);
      for (const part of parts) {
        if (part.length > NAME_MAX) {
          this.lastResolveErrno = ENAMETOOLONG;
          return null;
        }
      }
      return this.resolveFromParts(parts, followFinal, depth);
    }
    // the errno (negative) for the most recent failed resolve().
    resolveErrno() {
      return -this.lastResolveErrno;
    }
    // execute (search) permission on a directory: needed to traverse into/through it. root can
    // always search. non-root needs the x bit for owner/group/other as applicable.
    canSearch(dir) {
      const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
      const uid = this.searchEff ? c.euid : c.ruid;
      const gid = this.searchEff ? c.egid : c.rgid;
      if (uid === 0) return true;
      let perms;
      if (dir.uid === uid) perms = dir.mode >> 6 & 7;
      else if (dir.gid === gid) perms = dir.mode >> 3 & 7;
      else perms = dir.mode & 7;
      return (perms & 1) !== 0;
    }
    // normalize a raw list of path segments (which may contain '', '.', '..') into clean
    // components, resolving '..' against earlier entries. used when splicing a symlink target
    // into a path mid-resolution, where cwd-prepending must not happen.
    normalizePartsArray(segments) {
      const out = [];
      for (const p of segments) {
        if (p === "" || p === ".") continue;
        if (p === "..") out.pop();
        else out.push(p);
      }
      return out;
    }
    // write (modify) permission on a directory: needed to create, remove, or rename entries in
    // it. root bypasses; others need the +w bit for owner/group/other as applicable (effective).
    canWrite(dir) {
      const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
      if (c.euid === 0) return true;
      let perms;
      if (dir.uid === c.euid) perms = dir.mode >> 6 & 7;
      else if (dir.gid === c.egid) perms = dir.mode >> 3 & 7;
      else perms = dir.mode & 7;
      return (perms & 2) !== 0;
    }
    resolveFromParts(parts, followFinal, depth) {
      if (depth > MAX_SYMLINK_DEPTH) {
        this.lastResolveErrno = ELOOP;
        return null;
      }
      if (parts.length === 0) return this.root;
      let current = this.root;
      for (let i = 0; i < parts.length; i++) {
        if (current.type !== "dir") {
          this.lastResolveErrno = ENOTDIR2;
          return null;
        }
        if (!this.canSearch(current)) {
          this.searchDenied = true;
          this.lastResolveErrno = EACCES2;
          return null;
        }
        const child = current.children.get(parts[i]);
        if (!child) {
          this.lastResolveErrno = ENOENT2;
          return null;
        }
        const isLast = i === parts.length - 1;
        if (child.type === "symlink" && (followFinal || !isLast)) {
          const target = child.target;
          const base = target.startsWith("/") ? [] : parts.slice(0, i);
          const combined = [...base, ...target.split("/"), ...parts.slice(i + 1)];
          return this.resolveFromParts(this.normalizePartsArray(combined), followFinal, depth + 1);
        }
        current = child;
      }
      return current;
    }
    /** Resolve parent directory of path. Returns [parentDirInode, childName] or null; on null,
     *  lastResolveErrno holds the reason (ENOENT/ENOTDIR/ENAMETOOLONG/EACCES/ELOOP). */
    resolveParent(path, depth = 0) {
      if (depth === 0) this.lastResolveErrno = ENOENT2;
      if (depth > MAX_SYMLINK_DEPTH) {
        this.lastResolveErrno = ELOOP;
        return null;
      }
      const resolved = this.resolvePath(path);
      const parts = resolved.split("/").filter((p) => p.length > 0);
      if (parts.length === 0) return null;
      const name = parts.pop();
      if (name.length > NAME_MAX) {
        this.lastResolveErrno = ENAMETOOLONG;
        return null;
      }
      let dir = this.root;
      for (let idx = 0; idx < parts.length; idx++) {
        const part = parts[idx];
        if (part.length > NAME_MAX) {
          this.lastResolveErrno = ENAMETOOLONG;
          return null;
        }
        if (!this.canSearch(dir)) {
          this.searchDenied = true;
          this.lastResolveErrno = EACCES2;
          return null;
        }
        const child = dir.children.get(part);
        if (!child) {
          this.lastResolveErrno = ENOENT2;
          return null;
        }
        if (child.type === "symlink") {
          const target = child.target;
          const remaining = parts.slice(idx + 1).join("/");
          const newPath = remaining ? target + "/" + remaining + "/" + name : target + "/" + name;
          return this.resolveParent(newPath, depth + 1);
        }
        if (child.type !== "dir") {
          this.lastResolveErrno = ENOTDIR2;
          return null;
        }
        dir = child;
      }
      if (!this.canSearch(dir)) {
        this.searchDenied = true;
        this.lastResolveErrno = EACCES2;
        return null;
      }
      return [dir, name];
    }
    // ── VFS interface implementation ────────────────────────────────────────
    takeFd() {
      if (this.fdAllocator) return this.fdAllocator();
      let fd = 3;
      while (this.openFds.has(fd)) fd++;
      return fd;
    }
    open(path, flags, mode) {
      if (path.length > PATH_MAX) return -ENAMETOOLONG;
      for (const part of path.split("/")) {
        if (part.length > NAME_MAX) return -ENAMETOOLONG;
      }
      if (this.openFds.size >= MAX_FDS2) return -EMFILE2;
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
        if (!parent) return this.resolveErrno();
        const [dir, name] = parent;
        if (!this.canWrite(dir)) return -EACCES2;
        const newFile = this.makeFile(mode & 4095);
        this.applyCreatorOwnership(newFile, dir);
        dir.children.set(name, newFile);
        dir.mtime = dir.ctime = this.now();
        inode = newFile;
      }
      if (!inode) return this.resolveErrno();
      if (flags & O_DIRECTORY && inode.type !== "dir") return -ENOTDIR2;
      if (inode.type === "dir") {
        if ((flags & O_ACCMODE) !== O_RDONLY) return -EISDIR;
        const fd2 = this.takeFd();
        const entries = [".", "..", ...inode.children.keys()];
        this.openFds.set(fd2, { type: "dir", inode, entries, pos: 0, refcount: 1 });
        return fd2;
      }
      if (inode.type === "chardev") {
        const fd2 = this.takeFd();
        this.openFds.set(fd2, { type: "chardev", inode, refcount: 1 });
        return fd2;
      }
      if (inode.type === "fifo") {
        const fd2 = this.takeFd();
        this.openFds.set(fd2, { type: "fifo", inode, accessMode: flags & O_ACCMODE, refcount: 1 });
        return fd2;
      }
      if (inode.type !== "file") return -ENOENT2;
      if (flags & O_TRUNC && (flags & O_ACCMODE) !== O_RDONLY) {
        inode.data = new Uint8Array(0);
        inode.mtime = inode.ctime = this.now();
      }
      const fd = this.takeFd();
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
    // the O_RDONLY/O_WRONLY/O_RDWR access mode a file descriptor was opened with, or -1 if
    // the fd is unknown or has no access mode (e.g. a directory handle).
    fdAccessMode(fd) {
      const ofd = this.openFds.get(fd);
      return ofd && typeof ofd.accessMode === "number" ? ofd.accessMode : -1;
    }
    dup(fd) {
      const ofd = this.openFds.get(fd);
      if (!ofd) return -EBADF2;
      if (this.openFds.size >= MAX_FDS2) return -EMFILE2;
      const newFd = this.takeFd();
      ofd.refcount++;
      this.openFds.set(newFd, ofd);
      return newFd;
    }
    // install a shared duplicate of oldFd at a caller-chosen newFd. the caller owns
    // the fd namespace (dup2/dup3/F_DUPFD pick the target), so newFd must already be
    // free. keeps the guest fd == vfs internal fd invariant for dup'd fds.
    dupTo(oldFd, newFd) {
      const ofd = this.openFds.get(oldFd);
      if (!ofd) return -EBADF2;
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
      const guestMax = memory2.buffer.byteLength - GUEST_MEM_BASE;
      if (buf < 0 || buf >= guestMax || buf + count > guestMax) return -14;
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
      if (ofd.type === "chardev" || ofd.type === "fifo") return BigInt(-ESPIPE2);
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
    // only the file's owner (or root) may change its mode; otherwise EPERM.
    chmodInode(inode, mode) {
      const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
      if (c.euid !== 0 && c.euid !== inode.uid) return -EPERM2;
      inode.mode = inode.mode & S_IFMT | mode & 4095;
      inode.ctime = this.now();
      return 0;
    }
    fchmod(fd, mode) {
      const ofd = this.openFds.get(fd);
      if (!ofd) return -EBADF2;
      const inode = ofd.type === "file" ? ofd.inode : ofd.type === "dir" ? ofd.inode : null;
      if (!inode) return -EBADF2;
      return this.chmodInode(inode, mode);
    }
    // shared chown logic, matching the kernel's setattr_prepare. -1 (or 0xffffffff) means
    // "don't change". without CAP_CHOWN (non-root euid): changing the owner requires that the
    // caller already owns the file and the target uid equals the current one (i.e. no real
    // change); changing the group requires ownership and that the target gid is one the caller
    // belongs to (its egid/rgid) or the file's current gid. a successful chown of a non-directory
    // drops the set-user-ID bit, and the set-group-ID bit for group-executable files.
    chownInode(inode, uid, gid) {
      const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
      const noChg = (v) => v === -1 || v === 4294967295;
      const uidPresent = !noChg(uid);
      const gidPresent = !noChg(gid);
      const wantUid = uidPresent ? uid : inode.uid;
      const wantGid = gidPresent ? gid : inode.gid;
      if (c.euid !== 0) {
        if (uidPresent && (inode.uid !== c.euid || wantUid !== inode.uid)) return -EPERM2;
        if (gidPresent && (inode.uid !== c.euid || wantGid !== c.egid && wantGid !== c.rgid && wantGid !== inode.gid)) return -EPERM2;
      }
      inode.uid = wantUid;
      inode.gid = wantGid;
      if ((inode.mode & S_IFMT) !== S_IFDIR) {
        inode.mode &= ~2048;
        if (inode.mode & 8) inode.mode &= ~1024;
      }
      inode.ctime = this.now();
      return 0;
    }
    fchown(fd, uid, gid) {
      const ofd = this.openFds.get(fd);
      if (!ofd) return -EBADF2;
      const inode = ofd.type === "file" ? ofd.inode : ofd.type === "dir" ? ofd.inode : null;
      if (!inode) return -EBADF2;
      return this.chownInode(inode, uid, gid);
    }
    chmodPath(path, mode) {
      const inode = this.resolve(path);
      if (!inode) return this.resolveErrno();
      return this.chmodInode(inode, mode);
    }
    chownPath(path, uid, gid) {
      const inode = this.resolve(path);
      if (!inode) return this.resolveErrno();
      return this.chownInode(inode, uid, gid);
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
      if (!inode) return this.resolveErrno();
      this.writeStat(memory2, statBuf, inode);
      return 0;
    }
    // posix access-mode permission check. mode bits: R_OK=4, W_OK=2, X_OK=1, F_OK=0.
    // returns true if the effective creds may access the inode in the requested mode.
    // useEffective selects effective vs real uid/gid. access()/faccessat default to the
    // REAL ids; only faccessat with AT_EACCESS uses the effective ids.
    permitted(inode, mode, useEffective) {
      if (mode === 0) return true;
      const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
      const uid = useEffective ? c.euid : c.ruid;
      const gid = useEffective ? c.egid : c.rgid;
      if (uid === 0) {
        if (mode & 1) return (inode.mode & 73) !== 0;
        return true;
      }
      let perms;
      if (inode.uid === uid) perms = inode.mode >> 6 & 7;
      else if (inode.gid === gid) perms = inode.mode >> 3 & 7;
      else perms = inode.mode & 7;
      return (perms & mode) === mode;
    }
    faccessat(dirfd, path, mode, flags) {
      const AT_EACCESS = 512;
      if ((mode & ~7) !== 0) return -EINVAL2;
      if (path === "") return -ENOENT2;
      const useEff = (flags & AT_EACCESS) !== 0;
      const inode = this.resolve(path, true, 0, useEff);
      if (!inode) return this.resolveErrno();
      return this.permitted(inode, mode & 7, useEff) ? 0 : -EACCES2;
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
      if (!parent) return this.resolveErrno();
      const [dir, name] = parent;
      if (name === "." || name === "..") return -EINVAL2;
      const child = dir.children.get(name);
      if (!child) return -ENOENT2;
      if (!this.canWrite(dir)) return -EACCES2;
      if (flags & AT_REMOVEDIR) {
        if (child.type !== "dir") return -ENOTDIR2;
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
      if (!parent) return this.resolveErrno();
      const [dir, name] = parent;
      if (dir.children.has(name)) return -EEXIST;
      if (!this.canWrite(dir)) return -EACCES2;
      const newDir = this.makeDir(dir, mode & 4095);
      this.applyCreatorOwnership(newDir, dir);
      dir.children.set(name, newDir);
      dir.nlink++;
      dir.mtime = dir.ctime = this.now();
      return 0;
    }
    symlinkat(target, linkpath) {
      const parent = this.resolveParent(linkpath);
      if (!parent) return this.resolveErrno();
      const [dir, name] = parent;
      if (dir.children.has(name)) return -EEXIST;
      if (!this.canWrite(dir)) return -EACCES2;
      const link = this.makeSymlink(target);
      this.applyCreatorOwnership(link, dir);
      dir.children.set(name, link);
      dir.mtime = dir.ctime = this.now();
      return 0;
    }
    linkat(oldpath, newpath) {
      const inode = this.resolve(oldpath);
      if (!inode) return this.resolveErrno();
      if (inode.type === "dir") return -EPERM2;
      const parent = this.resolveParent(newpath);
      if (!parent) return this.resolveErrno();
      const [dir, name] = parent;
      if (dir.children.has(name)) return -EEXIST;
      if (!this.canWrite(dir)) return -EACCES2;
      dir.children.set(name, inode);
      inode.nlink++;
      inode.ctime = this.now();
      dir.mtime = dir.ctime = this.now();
      return 0;
    }
    rename(oldpath, newpath) {
      const oldParent = this.resolveParent(oldpath);
      if (!oldParent) return this.resolveErrno();
      const [oldDir, oldName] = oldParent;
      const srcNode = oldDir.children.get(oldName);
      if (!srcNode) return -ENOENT2;
      const newParent = this.resolveParent(newpath);
      if (!newParent) return this.resolveErrno();
      const [newDir, newName] = newParent;
      const dstNode = newDir.children.get(newName);
      if (dstNode) {
        if (srcNode.type === "dir" && dstNode.type !== "dir") return -ENOTDIR2;
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
    /** Set cwd directly (used when chdir target is on native FS, not in VFS) */
    setCwd(path) {
      this.cwd = path;
    }
    chdir(path) {
      if (path.length > PATH_MAX) return -ENAMETOOLONG;
      for (const part of path.split("/")) {
        if (part.length > NAME_MAX) return -ENAMETOOLONG;
      }
      const inode = this.resolve(path);
      if (!inode) return -ENOENT2;
      if (inode.type !== "dir") return -ENOTDIR2;
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
  var PT_INTERP = 3;
  var PT_PHDR = 6;
  var ET_DYN = 3;
  var INTERP_LOAD_BASE = 0x1000000n;
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
    let interpPath;
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
      if (pType === PT_INTERP) {
        let end = pOffset;
        while (end < pOffset + pFilesz && elfBytes2[end] !== 0) end++;
        interpPath = new TextDecoder().decode(elfBytes2.subarray(pOffset, end));
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
      loadBase,
      interpPath
    };
  }
  function loadInterp(interpBytes, memory2, base = INTERP_LOAD_BASE) {
    const dv = new DataView(interpBytes.buffer, interpBytes.byteOffset, interpBytes.byteLength);
    for (let i = 0; i < 4; i++) if (interpBytes[i] !== ELF_MAGIC[i]) throw new Error("interp not an ELF");
    const rawEntry = dv.getBigUint64(24, true);
    const phoff = Number(dv.getBigUint64(32, true));
    const phentSize = dv.getUint16(54, true);
    const phnum = dv.getUint16(56, true);
    let lowestVaddr = 0xFFFFFFFFFFFFFFFFn;
    for (let i = 0; i < phnum; i++) {
      const phBase = phoff + i * phentSize;
      if (dv.getUint32(phBase, true) === PT_LOAD) {
        const v = dv.getBigUint64(phBase + 16, true);
        if (v < lowestVaddr) lowestVaddr = v;
      }
    }
    const loadBase = base - lowestVaddr;
    for (let i = 0; i < phnum; i++) {
      const phBase = phoff + i * phentSize;
      if (dv.getUint32(phBase, true) !== PT_LOAD) continue;
      const pOffset = Number(dv.getBigUint64(phBase + 8, true));
      const pVaddr = dv.getBigUint64(phBase + 16, true) + loadBase;
      const pFilesz = Number(dv.getBigUint64(phBase + 32, true));
      const pMemsz = Number(dv.getBigUint64(phBase + 40, true));
      const wasmAddr = Number(pVaddr) + GUEST_MEM_BASE;
      const needed = wasmAddr + pMemsz;
      if (needed > memory2.buffer.byteLength) {
        const pagesNeeded = Math.ceil(needed / 65536);
        const currentPages = memory2.buffer.byteLength / 65536;
        if (pagesNeeded > currentPages) memory2.grow(pagesNeeded - currentPages);
      }
      const freshBytes = new Uint8Array(memory2.buffer);
      if (pFilesz > 0) freshBytes.set(interpBytes.subarray(pOffset, pOffset + pFilesz), wasmAddr);
      if (pMemsz > pFilesz) freshBytes.fill(0, wasmAddr + pFilesz, wasmAddr + pMemsz);
    }
    return { entry: rawEntry + loadBase, base: loadBase };
  }
  function loadProgram(memory2, elfBytes2, argv, envp, readInterp) {
    const elfInfo = loadElf(elfBytes2, memory2);
    let startPC = elfInfo.entry;
    if (elfInfo.interpPath) {
      const interpBytes = readInterp(elfInfo.interpPath);
      const { entry: iEntry, base: iBase } = loadInterp(interpBytes, memory2);
      elfInfo.interpBase = iBase;
      elfInfo.interpEntry = iEntry;
      startPC = iEntry;
    }
    const sp = setupStack(memory2, elfInfo, argv, envp);
    return { entry: startPC, sp, brkStart: elfInfo.brkStart, interp: !!elfInfo.interpPath };
  }
  var STACK_TOP = 8388608;
  var AT_NULL = 0;
  var AT_PHDR = 3;
  var AT_PHENT = 4;
  var AT_PHNUM = 5;
  var AT_PAGESZ = 6;
  var AT_BASE = 7;
  var AT_FLAGS = 8;
  var AT_ENTRY = 9;
  var AT_UID = 11;
  var AT_EUID = 12;
  var AT_GID = 13;
  var AT_EGID = 14;
  var AT_PLATFORM = 15;
  var AT_HWCAP = 16;
  var AT_CLKTCK = 17;
  var AT_SECURE = 23;
  var AT_RANDOM = 25;
  var AT_EXECFN = 31;
  var HWCAP_FP = 1 << 0;
  var HWCAP_ASIMD = 1 << 1;
  var HWCAP_AES = 1 << 3;
  var HWCAP_PMULL = 1 << 4;
  var HWCAP_SHA1 = 1 << 5;
  var HWCAP_SHA2 = 1 << 6;
  var HWCAP_SHA512 = 1 << 21;
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
    const platform = new TextEncoder().encode("aarch64\0");
    sp -= platform.length;
    const platformAddr = sp;
    bytes.set(platform, sp + GUEST_MEM_BASE);
    const execfnAddr = stringAddrs.length > 0 ? stringAddrs[0] : 0;
    sp = sp & ~15;
    const auxv = [
      [AT_PHDR, elfInfo.phdr],
      [AT_PHENT, BigInt(elfInfo.phentSize)],
      [AT_PHNUM, BigInt(elfInfo.phnum)],
      [AT_PAGESZ, 4096n],
      [AT_ENTRY, elfInfo.entry],
      [AT_FLAGS, 0n],
      [AT_HWCAP, BigInt(HWCAP_FP | HWCAP_ASIMD | HWCAP_AES | HWCAP_PMULL | HWCAP_SHA1 | HWCAP_SHA2 | HWCAP_SHA512)],
      [AT_CLKTCK, 100n],
      [AT_SECURE, 0n],
      [AT_UID, 0n],
      [AT_EUID, 0n],
      [AT_GID, 0n],
      [AT_EGID, 0n],
      [AT_PLATFORM, BigInt(platformAddr)],
      [AT_EXECFN, BigInt(execfnAddr)],
      [AT_RANDOM, BigInt(randomAddr)]
    ];
    if (elfInfo.interpBase !== void 0) auxv.push([AT_BASE, elfInfo.interpBase]);
    auxv.push([AT_NULL, 0n]);
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
  var PROC_PGID = 2;
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
  var MEM_PAGE_SIZE = 65536;
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
    /**
     * Blocking write: keep writing `src` until it is fully in the ring, parking on reader progress
     * (PIPE_READ_POS) whenever the ring is full. `waiter` is the interruptible park (same one the
     * blocking read uses): it returns true if a signal interrupted the wait.
     * Returns bytes written (== src.length on success); if nothing could be written it returns
     * -1 (all readers closed -> EPIPE) or -2 (interrupted -> EINTR). A partial write that then hits
     * reader-close/interrupt returns the partial count, per posix.
     */
    writeAll(src, waiter) {
      let written = 0;
      while (written < src.length) {
        const n = this.write(src.subarray(written));
        if (n === -1) return written > 0 ? written : -1;
        written += n;
        if (written >= src.length) break;
        const rp = Atomics.load(this.i32, PIPE_READ_POS);
        const wp = Atomics.load(this.i32, PIPE_WRITE_POS);
        if (this.dataSize - 1 - (wp - rp) <= 0) {
          if (waiter(this.i32, PIPE_READ_POS, rp)) return written > 0 ? written : -2;
        }
      }
      return written;
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
     * Returns -1 if `waiter` reported an interruption (EINTR — a pending signal).
     * Blocks via Atomics.wait() (or `waiter`) if the ring is empty and writer is still open.
     *
     * `waiter`, when given, replaces the bare Atomics.wait: it parks on the passed futex and
     * returns true if the wait should be abandoned because a signal is pending. this keeps the
     * signal machinery in the Worker while letting the ring own the empty/EOF logic. without it
     * the read blocks indefinitely, which is un-interruptible by anything but SIGKILL.
     *
     * IMPORTANT: Only call from a Worker thread (Atomics.wait blocks the thread).
     */
    read(dst, waiter) {
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
        if (waiter) {
          if (waiter(this.i32, PIPE_WRITE_POS, wp)) return -1;
        } else {
          Atomics.wait(this.i32, PIPE_WRITE_POS, wp);
        }
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
  var memory;
  var cpu;
  var syscalls;
  var vfs;
  var regionCache = /* @__PURE__ */ new Map();
  var threaded = false;
  var stateBase = 0;
  var regBaseGlobal = null;
  var clearChildTid = 0;
  var THREAD_REGS_BASE = 1610612736;
  var THREADED_SHADOW_BASE = 1614807040;
  var REGION_CACHE_CAP = 2048;
  var elfBytes;
  var kernelSAB;
  var procTable;
  var mySlot;
  var myPid;
  var myMailbox;
  var kernelPipes = /* @__PURE__ */ new Map();
  var nextLocalFd = 100;
  var stdinRing = null;
  var workerReady = false;
  var lastSyscallNr = -1;
  var ZIG_SHADOW_BASE = 33554432;
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
      case "term_size":
        if (syscalls) syscalls.setTermSize(msg.cols, msg.rows);
        break;
      case "promote":
        completePromote(msg.sharedMemory, msg.tid, msg.regBaseAddr, msg.staleHigh);
        break;
      case "clone_fail":
        completePromoteFail();
        break;
    }
  };
  function init(config) {
    myPid = config.pid;
    mySlot = config.slot;
    kernelSAB = config.kernelSAB;
    elfBytes = config.elfBytes;
    if (!workerReady) {
      if (config.zigTranslatorModule) {
        initZigTranslatorModule(config.zigTranslatorModule);
      } else if (config.zigTranslatorBytes) {
        initZigTranslator(config.zigTranslatorBytes);
      } else {
        throw new Error(`[worker ${myPid}] no translator module or bytes in init config`);
      }
      memory = new WebAssembly.Memory({ initial: 512, maximum: 65536 });
      workerReady = true;
    } else {
      if (threaded) {
        memory = new WebAssembly.Memory({ initial: 512, maximum: 65536 });
        threaded = false;
        stateBase = 0;
        regBaseGlobal = null;
      }
      const dirtyEnd = Math.min(Number(syscalls.getMmapBase()) + GUEST_MEM_BASE, memory.buffer.byteLength);
      new Uint8Array(memory.buffer).fill(0, 0, dirtyEnd);
      regionCache.clear();
      kernelPipes.clear();
      nextLocalFd = 100;
      stdinRing = null;
    }
    procTable = new Int32Array(kernelSAB);
    myMailbox = new Int32Array(kernelSAB, mailboxSlotOffset(mySlot), MAILBOX_SLOT_SIZE / 4);
    const bridgeHandlesRealFs = !!config.kernelSAB;
    vfs = new InMemoryVFS({ bridgeHandlesRealFs });
    if (!bridgeHandlesRealFs) {
      vfs.loadBinary(elfBytes);
    }
    if (config.threadStart) {
      const ts = config.threadStart;
      memory = ts.sharedMemory;
      threaded = true;
      stateBase = ts.regBaseAddr;
      regBaseGlobal = new WebAssembly.Global({ value: "i32", mutable: false }, ts.regBaseAddr);
      new Uint8Array(memory.buffer).copyWithin(stateBase, ts.parentRegBase, ts.parentRegBase + STATE_SIZE);
      cpu = new CpuState(memory, stateBase);
      cpu.setReg(0, 0n);
      cpu.setSP(BigInt(ts.childStack >>> 0));
      new DataView(memory.buffer).setBigUint64(stateBase + 280, BigInt(ts.tls >>> 0), true);
      if (ts.flags & 16777216 && ts.childTidPtr) {
        new DataView(memory.buffer).setInt32(ts.childTidPtr + GUEST_MEM_BASE, ts.tid, true);
      }
      if (ts.flags & 2097152 && ts.childTidPtr) clearChildTid = ts.childTidPtr;
      syscalls = new SyscallHandler(0n, {
        writeStdout: (d) => postWrite(1, d),
        writeStderr: (d) => postWrite(2, d),
        readStdin: (count) => readStdin(count),
        stdinHasData: () => stdinRing != null && (!stdinRing.empty || stdinRing.writerClosed),
        waitStdin: (timeoutMs) => waitStdin(timeoutMs),
        readFileAt: (fd, offset, count) => readNativeFileAt(fd, offset, count)
      }, vfs);
      syscalls.bridgeAvailable = true;
      syscalls.setThreadIds(ts.tid, ts.tgid);
      syscalls.maxBytes = THREAD_REGS_BASE;
      runToCompletion(BigInt(ts.startPC));
      return;
    }
    let pc;
    if (config.memSnapshot) {
      applySparseSnapshot(memory, config.memSnapshot);
      cpu = new CpuState(memory, stateBase);
      cpu.setReg(0, 0n);
      syscalls = new SyscallHandler(0n, {
        writeStdout: (d) => postWrite(1, d),
        writeStderr: (d) => postWrite(2, d),
        readStdin: (count) => readStdin(count),
        stdinHasData: () => stdinRing != null && (!stdinRing.empty || stdinRing.writerClosed),
        waitStdin: (timeoutMs) => waitStdin(timeoutMs),
        readFileAt: (fd, offset, count) => readNativeFileAt(fd, offset, count)
      }, vfs);
      syscalls.bridgeAvailable = true;
      if (config.syscallState) {
        syscalls.restoreState(config.syscallState);
        syscalls.setIdentity(myPid, syscalls.currentPid);
        syscalls.dupNativeFdsAfterFork((oldFd) => {
          let result = -1;
          handleBridgeCall("fs_dup", { fd: oldFd }, (resp) => {
            result = resp.ret;
          });
          return result;
        });
      }
      if (config.kernelPipesSnapshot) {
        for (const [fd, info] of config.kernelPipesSnapshot) {
          kernelPipes.set(fd, { ...info });
        }
        for (const fd of kernelPipes.keys()) {
          if (fd >= nextLocalFd) nextLocalFd = fd + 1;
        }
      }
      pc = cpu.getPC() + 4n;
    } else {
      const prog = loadProgram(memory, elfBytes, config.argv, config.envp, readGuestFile);
      cpu = new CpuState(memory, stateBase);
      cpu.setSP(prog.sp);
      new DataView(memory.buffer).setBigUint64(stateBase + 280, BigInt(9437184), true);
      syscalls = new SyscallHandler(prog.brkStart, {
        writeStdout: (d) => postWrite(1, d),
        writeStderr: (d) => postWrite(2, d),
        readStdin: (count) => readStdin(count),
        stdinHasData: () => stdinRing != null && (!stdinRing.empty || stdinRing.writerClosed),
        waitStdin: (timeoutMs) => waitStdin(timeoutMs),
        readFileAt: (fd, offset, count) => readNativeFileAt(fd, offset, count)
      }, vfs);
      syscalls.bridgeAvailable = true;
      pc = prog.entry;
    }
    const stdinRingIdx = Atomics.load(procTable, procSlotOffset(mySlot) / 4 + 7);
    if (stdinRingIdx >= 0) {
      stdinRing = new PipeRing(kernelSAB, stdinRingOffset(stdinRingIdx), STDIN_RING_SIZE);
    }
    if (config.termCols && config.termRows) {
      syscalls.setTermSize(config.termCols, config.termRows);
    }
    syscalls.installSignalTrampoline(memory);
    runToCompletion(pc);
  }
  function runToCompletion(pc) {
    try {
      runLoop(pc);
    } catch (e) {
      if (e.message === "SIGINT") {
        return;
      }
      console.error(`[worker ${myPid}] CRASH:`, e.message, e.stack);
      try {
        postWrite(2, new TextEncoder().encode(`\r
[worker ${myPid}] crash: ${e.message} (last syscall nr=${lastSyscallNr})\r
`));
      } catch {
      }
      postExit(139);
    } finally {
      if (!pendingClone) {
        try {
          syscalls.closeNativeFds((fd) => handleBridgeCall("fs_close", { fd }, () => {
          }));
        } catch {
        }
      }
    }
  }
  function getRegion(pc) {
    const region = regionCache.get(pc);
    if (region) {
      if (regionCache.size >= REGION_CACHE_CAP) {
        regionCache.delete(pc);
        regionCache.set(pc, region);
      }
      return region;
    }
    const view = new DataView(memory.buffer);
    const shadowBase = threaded ? THREADED_SHADOW_BASE : ZIG_SHADOW_BASE;
    const zigRegion = translateRegionZig(view, pc, memory, shadowBase, false, threaded, regBaseGlobal);
    if (!zigRegion) {
      const insn = view.getUint32(Number(pc) + 65536, true);
      throw new Error(`[worker ${myPid}] translator failed at pc=0x${pc.toString(16)} (entry insn 0x${insn.toString(16).padStart(8, "0")})`);
    }
    if (regionCache.size >= REGION_CACHE_CAP) {
      const oldest = regionCache.keys().next().value;
      if (oldest !== void 0) regionCache.delete(oldest);
    }
    regionCache.set(pc, zigRegion);
    return zigRegion;
  }
  function pollAndQueueSignals() {
    syscalls.checkTimers();
    const sigSlot = procSlotOffset(mySlot) / 4 + PROC_SIGNAL;
    const bits2 = Atomics.load(procTable, sigSlot);
    if (bits2) {
      Atomics.and(procTable, sigSlot, ~bits2);
      for (let signo = 1; signo < 32; signo++) if (bits2 & 1 << signo) syscalls.raiseSignal(signo);
    }
    return syscalls.pendingSignals.length > 0;
  }
  function interruptibleWaiter(i32, idx, expected) {
    const rem = syscalls.timerRemainingMs();
    Atomics.wait(i32, idx, expected, rem > 0 ? Math.min(1e3, rem) : 1e3);
    return pollAndQueueSignals();
  }
  function deliverQueuedSignals(pc) {
    if (!syscalls.pendingSignals.length) return pc;
    cpu.setPC(pc);
    while (syscalls.pendingSignals.length > 0) {
      const sig = syscalls.pendingSignals.shift();
      const r = syscalls.deliverSignal(cpu, memory, sig);
      if (r.action === "exit") {
        postExit(r.exitCode ?? 128 + sig);
        return null;
      }
      if (r.delivered && r.action === "continue") return cpu.getPC();
    }
    return pc;
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
      if (region.lastExitReason === 3) {
        {
          const _i = new DataView(memory.buffer).getUint32(Number(pc) + GUEST_MEM_BASE, true);
          postWrite(2, new TextEncoder().encode(`[SIGILL pc=0x${pc.toString(16)} insn=0x${_i.toString(16).padStart(8, "0")}]\r
`));
        }
        cpu.setPC(pc);
        syscalls.raiseSignal(4);
        const next = deliverQueuedSignals(pc);
        if (next === null) return;
        pc = next;
        continue;
      }
      view = new DataView(memory.buffer);
      const insns = Number(view.getBigUint64(stateBase + ITER_OFFSET, true));
      totalInsns += insns;
      view.setBigUint64(stateBase + ITER_OFFSET, 0n, true);
      lastSignalCheck += insns;
      if (lastSignalCheck > 1e4) {
        lastSignalCheck = 0;
        if (pollAndQueueSignals()) {
          const next = deliverQueuedSignals(pc);
          if (next === null) return;
          pc = next;
          continue;
        }
      }
      if (pc === SVC_PC) {
        const svcPC = view.getBigUint64(stateBase + PC_OFFSET, true);
        const nr = Number(cpu.getReg(8));
        lastSyscallNr = nr;
        if (nr === 260) {
          handleWait4(cpu, view);
          const eintr = BigInt.asIntN(64, cpu.getReg(0)) === -4n;
          const retPC = eintr && syscalls.pendingSignals.length > 0 && syscalls.anyPendingRestart() ? svcPC : svcPC + 4n;
          if (syscalls.pendingSignals.length) {
            const next = deliverQueuedSignals(retPC);
            if (next === null) return;
            pc = next;
          } else {
            pc = retPC;
          }
          continue;
        }
        if (nr === 139) {
          pc = syscalls.handleSigreturn(cpu, memory);
          continue;
        }
        if (nr === 129 || nr === 130 || nr === 131) {
          handleKill(nr, cpu);
          pc = svcPC + 4n;
          if (syscalls.pendingSignals.length) {
            const next = deliverQueuedSignals(pc);
            if (next === null) return;
            pc = next;
          }
          continue;
        }
        if (nr === 154) {
          handleSetpgid(cpu);
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 155) {
          handleGetpgid(cpu);
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 220) {
          const cloneFlags = Number(cpu.getReg(0));
          const CLONE_THREAD = 65536;
          if (cloneFlags & CLONE_THREAD) {
            if (handleCloneThread(cpu, svcPC)) return;
            pc = svcPC + 4n;
            continue;
          }
        }
        if (nr === 98 && threaded) {
          const interrupted = handleFutex(cpu);
          pc = svcPC + 4n;
          if (interrupted && syscalls.pendingSignals.length) {
            const next = deliverQueuedSignals(pc);
            if (next === null) return;
            pc = next;
          }
          continue;
        }
        if (nr === 157) {
          Atomics.store(procTable, procSlotOffset(mySlot) / 4 + PROC_PGID, myPid);
          cpu.setReg(0, BigInt(myPid));
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 25) {
          const lc = Number(cpu.getReg(1));
          if (lc === 5 || lc === 6 || lc === 7 || lc === 12 || lc === 13 || lc === 14) {
            handleFcntlLock(cpu, view, lc);
            pc = svcPC + 4n;
            continue;
          }
        }
        if (nr === 101 || nr === 115) {
          const abs = nr === 115 && (Number(cpu.getReg(1)) & 1) !== 0;
          const reqAddr = Number(nr === 115 ? cpu.getReg(2) : cpu.getReg(0));
          const remAddr = Number(nr === 115 ? cpu.getReg(3) : cpu.getReg(1));
          handleNanosleep(cpu, view, reqAddr, remAddr, abs, nr === 115 ? Number(cpu.getReg(0)) : 1);
          pc = svcPC + 4n;
          if (syscalls.pendingSignals.length) {
            const next = deliverQueuedSignals(pc);
            if (next === null) return;
            pc = next;
          }
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
          const eintr = BigInt.asIntN(64, cpu.getReg(0)) === -4n;
          const retPC = eintr && syscalls.pendingSignals.length > 0 && syscalls.anyPendingRestart() ? svcPC : svcPC + 4n;
          if (syscalls.pendingSignals.length) {
            const next = deliverQueuedSignals(retPC);
            if (next === null) return;
            pc = next;
          } else {
            pc = retPC;
          }
          continue;
        }
        let result;
        try {
          result = syscalls.handle(cpu, memory);
        } catch (e) {
          if (e instanceof RangeError) {
            cpu.setReg(0, BigInt.asUintN(64, BigInt(-14)));
            result = { status: "ok" };
          } else throw e;
        }
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
            pc = handleExec(result.path, result.argv, result.envp);
            break;
          case "blocked_read":
            handleBlockedRead();
            pc = svcPC;
            break;
          case "bridge":
            if (result.method.startsWith("fs_")) {
              result.args.creds = syscalls.getCreds();
            }
            handleBridgeCall(result.method, result.args, result.apply);
            pc = svcPC + 4n;
            break;
          default:
            pc = svcPC + 4n;
            break;
        }
        if (syscalls.pendingSignals.length) {
          const next = deliverQueuedSignals(pc);
          if (next === null) return;
          pc = next;
        }
        continue;
      }
    }
    postExit(0);
  }
  function buildSparseSnapshot(buffer, totalLen) {
    const pageCount = Math.ceil(totalLen / MEM_PAGE_SIZE);
    const wordsPerPage = MEM_PAGE_SIZE >>> 2;
    const u32 = new Uint32Array(buffer, 0, pageCount * MEM_PAGE_SIZE >>> 2);
    const bytes = new Uint8Array(buffer);
    const idx = [];
    for (let p = 0; p < pageCount; p++) {
      const wStart = p * wordsPerPage;
      let nz = false;
      for (let w = 0; w < wordsPerPage; w++) {
        if (u32[wStart + w] !== 0) {
          nz = true;
          break;
        }
      }
      if (nz) idx.push(p);
    }
    const data = new Uint8Array(idx.length * MEM_PAGE_SIZE);
    for (let i = 0; i < idx.length; i++) {
      const off = idx[i] * MEM_PAGE_SIZE;
      data.set(bytes.subarray(off, off + MEM_PAGE_SIZE), i * MEM_PAGE_SIZE);
    }
    return { pages: Uint32Array.from(idx), data: data.buffer, totalLen: pageCount * MEM_PAGE_SIZE };
  }
  function applySparseSnapshot(mem, snap) {
    if (snap.totalLen > mem.buffer.byteLength) {
      const need = Math.ceil(snap.totalLen / MEM_PAGE_SIZE);
      const cur = mem.buffer.byteLength / MEM_PAGE_SIZE;
      if (need > cur) mem.grow(need - cur);
    }
    const dst = new Uint8Array(mem.buffer);
    const data = new Uint8Array(snap.data);
    for (let i = 0; i < snap.pages.length; i++) {
      dst.set(data.subarray(i * MEM_PAGE_SIZE, (i + 1) * MEM_PAGE_SIZE), snap.pages[i] * MEM_PAGE_SIZE);
    }
  }
  function handleFork(resumePC) {
    const snapSize = Math.min(
      Number(syscalls.getMmapBase()) + GUEST_MEM_BASE + 65536,
      memory.buffer.byteLength
    );
    const memSnapshot = buildSparseSnapshot(memory.buffer, snapSize);
    const syscallState = syscalls.getState();
    cpu.setPC(resumePC);
    const kernelPipesSnapshot = [...kernelPipes.entries()];
    for (const [, info] of kernelPipesSnapshot) {
      const ring = new PipeRing(kernelSAB, pipeRingOffset(info.pipeIndex), PIPE_RING_SIZE);
      if (info.mode === "write") ring.addWriter();
      else ring.addReader();
    }
    self.postMessage({
      type: "fork",
      memSnapshot,
      syscallState,
      kernelPipesSnapshot,
      resumePC: resumePC.toString()
    }, [memSnapshot.data]);
    Atomics.store(myMailbox, MBOX_READY, 0);
    Atomics.wait(myMailbox, MBOX_READY, 0);
    const childPid = Atomics.load(myMailbox, MBOX_VALUE0);
    if (childPid < 0) {
      for (const [, info] of kernelPipesSnapshot) {
        const ring = new PipeRing(kernelSAB, pipeRingOffset(info.pipeIndex), PIPE_RING_SIZE);
        if (info.mode === "write") ring.closeWriter();
        else ring.closeReader();
      }
      cpu.setReg(0, BigInt.asUintN(64, BigInt(childPid)));
      return resumePC;
    }
    cpu.setReg(0, BigInt(childPid));
    return resumePC;
  }
  var CLONE_PARENT_SETTID = 1048576;
  var pendingClone = null;
  function handleCloneThread(cpu2, svcPC) {
    const flags = Number(cpu2.getReg(0));
    const childStack = Number(cpu2.getReg(1));
    const parentTidPtr = Number(cpu2.getReg(2));
    const tls = Number(cpu2.getReg(3));
    const childTidPtr = Number(cpu2.getReg(4));
    if (!threaded) {
      pendingClone = { svcPC, childStack, tls, parentTidPtr, childTidPtr, flags };
      self.postMessage({ type: "clone_thread", curPages: memory.buffer.byteLength / MEM_PAGE_SIZE });
      return true;
    }
    Atomics.store(myMailbox, MBOX_READY, 0);
    self.postMessage({
      type: "spawn_thread",
      tgid: syscalls.tgid || myPid,
      startPC: (svcPC + 4n).toString(),
      childStack,
      tls,
      childTidPtr,
      flags,
      parentRegBase: stateBase
    });
    Atomics.wait(myMailbox, MBOX_READY, 0);
    const tid = Atomics.load(myMailbox, MBOX_VALUE0);
    if (tid < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-11)));
      return false;
    }
    if (flags & CLONE_PARENT_SETTID && parentTidPtr) new DataView(memory.buffer).setInt32(parentTidPtr + GUEST_MEM_BASE, tid, true);
    cpu2.setReg(0, BigInt(tid));
    return false;
  }
  function completePromote(sharedMemory, tid, regBaseAddr, staleHigh) {
    const pc = pendingClone;
    const copyLen = memory.buffer.byteLength;
    const shared = new Uint8Array(sharedMemory.buffer);
    shared.set(new Uint8Array(memory.buffer));
    if (staleHigh > copyLen) shared.fill(0, copyLen, staleHigh);
    if (staleHigh > 0) shared.fill(0, THREADED_SHADOW_BASE, THREADED_SHADOW_BASE + (THREAD_REGS_BASE >> 9));
    memory = sharedMemory;
    threaded = true;
    stateBase = 0;
    regBaseGlobal = new WebAssembly.Global({ value: "i32", mutable: false }, 0);
    cpu = new CpuState(memory, stateBase);
    regionCache.clear();
    syscalls.setThreadIds(myPid, myPid);
    syscalls.maxBytes = THREAD_REGS_BASE;
    new Uint8Array(memory.buffer).copyWithin(regBaseAddr, 0, STATE_SIZE);
    cpu.setReg(0, BigInt(tid));
    if (pc.flags & CLONE_PARENT_SETTID && pc.parentTidPtr) new DataView(memory.buffer).setInt32(pc.parentTidPtr + GUEST_MEM_BASE, tid, true);
    self.postMessage({
      type: "thread_go",
      tid,
      regBaseAddr,
      startPC: (pc.svcPC + 4n).toString(),
      childStack: pc.childStack,
      tls: pc.tls,
      childTidPtr: pc.childTidPtr,
      flags: pc.flags,
      parentRegBase: regBaseAddr
      // block already holds the clone-time snapshot
    });
    const resume = pc.svcPC + 4n;
    pendingClone = null;
    runToCompletion(resume);
  }
  function completePromoteFail() {
    const pc = pendingClone;
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-11)));
    const resume = pc.svcPC + 4n;
    pendingClone = null;
    runToCompletion(resume);
  }
  function handleFutex(cpu2) {
    const uaddr = Number(cpu2.getReg(0));
    const op = Number(cpu2.getReg(1)) & 127;
    const val = Number(cpu2.getReg(2)) | 0;
    const i32 = new Int32Array(memory.buffer);
    const idx = uaddr + GUEST_MEM_BASE >> 2;
    if (op === 0) {
      if (Atomics.load(i32, idx) !== val) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-11)));
        return false;
      }
      const r = Atomics.wait(i32, idx, val, 1e3);
      if (r === "not-equal") {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-11)));
        return false;
      }
      if (pollAndQueueSignals()) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-4)));
        return true;
      }
      cpu2.setReg(0, 0n);
      return false;
    }
    if (op === 1) {
      cpu2.setReg(0, BigInt(Atomics.notify(i32, idx, val < 0 ? Infinity : val)));
      return false;
    }
    if (op === 3 || op === 4 || op === 5 || op === 10 || op === 11) {
      cpu2.setReg(0, BigInt(Atomics.notify(i32, idx, val < 0 ? Infinity : val)));
      return false;
    }
    cpu2.setReg(0, 0n);
    return false;
  }
  function handleExec(path, argv, envp) {
    let bytes = readGuestFile(path);
    if (bytes.length >= 2 && bytes[0] === 35 && bytes[1] === 33) {
      let end = 2;
      while (end < bytes.length && end < 256 && bytes[end] !== 10) end++;
      const m = new TextDecoder().decode(bytes.subarray(2, end)).trim().match(/^(\S+)(?:\s+(.*))?$/);
      if (m) {
        const interp = m[1];
        const optArg = m[2]?.trim();
        const newArgv = [interp];
        if (optArg) newArgv.push(optArg);
        newArgv.push(path, ...argv.slice(1));
        path = interp;
        argv = newArgv;
        bytes = readGuestFile(interp);
      }
    }
    elfBytes = bytes;
    new Uint8Array(memory.buffer).fill(0, GUEST_MEM_BASE, GUEST_MEM_BASE + 67108864);
    const prog = loadProgram(memory, elfBytes, argv, envp, readGuestFile);
    cpu = new CpuState(memory, stateBase);
    cpu.setSP(prog.sp);
    new DataView(memory.buffer).setBigUint64(stateBase + 280, BigInt(9437184), true);
    syscalls.resetForExec(prog.brkStart, (fd) => handleBridgeCall("fs_close", { fd }, () => {
    }));
    regionCache.clear();
    syscalls.installSignalTrampoline(memory);
    return prog.entry;
  }
  function handleNanosleep(cpu2, view, reqAddr, remAddr, absolute, clockId) {
    const sec = Number(view.getBigUint64(reqAddr + GUEST_MEM_BASE, true));
    const nsec = Number(view.getBigUint64(reqAddr + GUEST_MEM_BASE + 8, true));
    const targetNs = sec * 1e9 + nsec;
    const ms = absolute ? (targetNs - Number(syscalls.clockNs(clockId))) / 1e6 : targetNs / 1e6;
    if (ms <= 0) {
      cpu2.setReg(0, 0n);
      return;
    }
    const slotBase = procSlotOffset(mySlot) / 4;
    const deadline = performance.now() + ms;
    let interrupted = false;
    for (; ; ) {
      const rem = deadline - performance.now();
      if (rem <= 0) break;
      Atomics.store(procTable, slotBase + PROC_WAIT_FUTEX, 0);
      Atomics.wait(procTable, slotBase + PROC_WAIT_FUTEX, 0, Math.min(rem, 1e3));
      if (pollAndQueueSignals()) {
        interrupted = true;
        break;
      }
    }
    if (interrupted) {
      if (remAddr && !absolute) {
        const remMs = Math.max(0, deadline - performance.now());
        view.setBigUint64(remAddr + GUEST_MEM_BASE, BigInt(Math.floor(remMs / 1e3)), true);
        view.setBigUint64(remAddr + GUEST_MEM_BASE + 8, BigInt(Math.round(remMs % 1e3 * 1e6)), true);
      }
      cpu2.setReg(0, BigInt.asUintN(64, -4n));
    } else {
      if (remAddr && !absolute) {
        view.setBigUint64(remAddr + GUEST_MEM_BASE, 0n, true);
        view.setBigUint64(remAddr + GUEST_MEM_BASE + 8, 0n, true);
      }
      cpu2.setReg(0, 0n);
    }
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
      const rem = syscalls.timerRemainingMs();
      const waitMs = rem > 0 ? Math.min(1e3, rem) : 1e3;
      const mySlotBase = procSlotOffset(mySlot) / 4;
      Atomics.store(procTable, mySlotBase + PROC_WAIT_FUTEX, 0);
      Atomics.wait(procTable, mySlotBase + PROC_WAIT_FUTEX, 0, waitMs);
      if (pollAndQueueSignals()) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-4)));
        return;
      }
    }
  }
  function handleKill(nr, cpu2) {
    const targetPid = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    const sig = nr === 131 ? Number(cpu2.getReg(2)) : Number(cpu2.getReg(1));
    const myPgid = Atomics.load(procTable, procSlotOffset(mySlot) / 4 + PROC_PGID);
    let delivered = false;
    for (let slot = 0; slot < 64; slot++) {
      const base = procSlotOffset(slot) / 4;
      const pid = Atomics.load(procTable, base + PROC_PID);
      if (pid === 0) continue;
      const pgid = Atomics.load(procTable, base + PROC_PGID);
      let match;
      if (targetPid > 0) match = pid === targetPid;
      else if (targetPid === 0) match = pgid === myPgid;
      else if (targetPid === -1) match = pid !== 1 && pid !== myPid;
      else match = pgid === -targetPid;
      if (!match) continue;
      delivered = true;
      if (sig === 0) continue;
      if (pid === myPid) {
        syscalls.raiseSignal(sig);
        continue;
      }
      if (sig === 9) {
        self.postMessage({ type: "kill", pid, sig });
        continue;
      }
      Atomics.or(procTable, base + PROC_SIGNAL, 1 << sig);
      Atomics.store(procTable, base + PROC_WAIT_FUTEX, 1);
      Atomics.notify(procTable, base + PROC_WAIT_FUTEX);
    }
    cpu2.setReg(0, delivered ? 0n : BigInt.asUintN(64, BigInt(-3)));
  }
  function handleSetpgid(cpu2) {
    let pid = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    let pgid = Number(BigInt.asIntN(32, cpu2.getReg(1)));
    if (pid === 0) pid = myPid;
    if (pgid === 0) pgid = pid;
    if (pgid < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    for (let slot = 0; slot < 64; slot++) {
      const base = procSlotOffset(slot) / 4;
      if (Atomics.load(procTable, base + PROC_PID) !== pid) continue;
      Atomics.store(procTable, base + PROC_PGID, pgid);
      cpu2.setReg(0, 0n);
      return;
    }
    cpu2.setReg(0, BigInt.asUintN(64, BigInt(-3)));
  }
  function handleGetpgid(cpu2) {
    let pid = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    if (pid === 0) pid = myPid;
    for (let slot = 0; slot < 64; slot++) {
      const base = procSlotOffset(slot) / 4;
      if (Atomics.load(procTable, base + PROC_PID) !== pid) continue;
      cpu2.setReg(0, BigInt(Atomics.load(procTable, base + PROC_PGID)));
      return;
    }
    cpu2.setReg(0, BigInt.asUintN(64, BigInt(-3)));
  }
  function handleFcntlLock(cpu2, view, cmd) {
    const fd = Number(cpu2.getReg(0));
    const flockAddr = Number(cpu2.getReg(2));
    if (badGuestPtr(new Uint8Array(memory.buffer), flockAddr, 32)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
      return;
    }
    const flockPtr = flockAddr + GUEST_MEM_BASE;
    const ltype = view.getInt16(flockPtr, true);
    const whence = view.getInt16(flockPtr + 2, true);
    const lstart = Number(view.getBigInt64(flockPtr + 8, true));
    const len = Number(view.getBigInt64(flockPtr + 16, true));
    const path = syscalls.fdPath(fd);
    if (path === null) {
      cpu2.setReg(0, 0n);
      return;
    }
    let start = lstart;
    if (whence === 1) start += syscalls.fdOffset(fd);
    else if (whence === 2) start += syscalls.fdSize(fd);
    if (start < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    Atomics.store(myMailbox, MBOX_READY, 0);
    self.postMessage({ type: "lock_req", path, cmd, ltype, start, len });
    Atomics.wait(myMailbox, MBOX_READY, 0);
    const r = Atomics.load(myMailbox, MBOX_VALUE0);
    if (cmd === 5 || cmd === 12) {
      const rtype = Atomics.load(myMailbox, MBOX_VALUE0 + 1);
      view.setInt16(flockPtr, rtype, true);
      if (rtype !== 2) {
        const rpid = Atomics.load(myMailbox, MBOX_VALUE0 + 2);
        const cstart = (Atomics.load(myMailbox, MBOX_VALUE0 + 3) >>> 0) + Atomics.load(myMailbox, MBOX_VALUE0 + 4) * 4294967296;
        const clen = (Atomics.load(myMailbox, MBOX_VALUE0 + 5) >>> 0) + Atomics.load(myMailbox, MBOX_VALUE0 + 6) * 4294967296;
        view.setInt16(flockPtr + 2, 0, true);
        view.setBigInt64(flockPtr + 8, BigInt(cstart), true);
        view.setBigInt64(flockPtr + 16, BigInt(clen), true);
        view.setInt32(flockPtr + 24, rpid, true);
      }
      cpu2.setReg(0, 0n);
      return;
    }
    cpu2.setReg(0, r < 0 ? BigInt.asUintN(64, BigInt(r)) : 0n);
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
    if (pipeIndex < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-23)));
      return;
    }
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
      if (count > 0 && badGuestPtr(new Uint8Array(memory.buffer), buf, count)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
      const dst = new Uint8Array(count);
      const n = ring.read(dst, interruptibleWaiter);
      if (n === -1) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-4)));
        return;
      }
      if (n > 0) {
        new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, n).set(dst.subarray(0, n));
      }
      cpu2.setReg(0, BigInt(n));
      return;
    }
    if (nr === 64) {
      const buf = Number(cpu2.getReg(1));
      const count = Number(cpu2.getReg(2));
      if (count > 0 && badGuestPtr(new Uint8Array(memory.buffer), buf, count)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
      const src = new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, count);
      if (syscalls.isNonBlockingFd(fd)) {
        const n2 = ring.write(src);
        if (n2 === -1) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(-32)));
          return;
        }
        if (n2 === 0 && count > 0) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(-11)));
          return;
        }
        cpu2.setReg(0, BigInt(n2));
        return;
      }
      const n = ring.writeAll(src, interruptibleWaiter);
      if (n === -1) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-32)));
        return;
      }
      if (n === -2) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-4)));
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
      if (iovcnt > 0 && badGuestPtr(bytes, iovAddr, iovcnt * 16)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
      for (let i = 0; i < iovcnt; i++) {
        const base = view.getBigUint64(iovAddr + i * 16 + GUEST_MEM_BASE, true);
        const len = view.getBigUint64(iovAddr + i * 16 + 8 + GUEST_MEM_BASE, true);
        if (len === 0n) continue;
        if (badGuestPtr(bytes, Number(base), Number(len))) {
          if (totalWritten === 0) {
            cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
            return;
          }
          break;
        }
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
        const rem = syscalls.timerRemainingMs();
        Atomics.wait(ringI32, 0, wp, rem > 0 ? Math.min(1e3, rem) : 1e3);
        if (pollAndQueueSignals()) return;
      }
    }
  }
  function readGuestFile(path) {
    const fdh = { fd: -1 };
    handleBridgeCall("fs_open", { path, flags: 0, mode: 0 }, (r) => {
      fdh.fd = r.ret ?? -1;
    });
    if (fdh.fd < 0) throw new Error(`readGuestFile: cannot open ${path} (ret ${fdh.fd})`);
    const chunks = [];
    let total = 0;
    const CHUNK = 12288;
    try {
      for (; ; ) {
        const h = { data: new Uint8Array(0) };
        handleBridgeCall("fs_read", { fd: fdh.fd, count: CHUNK }, (r) => {
          if (typeof r.data === "string" && r.data.length) {
            const bin = atob(r.data);
            const u = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
            h.data = u;
          }
        });
        if (h.data.length === 0) break;
        chunks.push(h.data);
        total += h.data.length;
        if (h.data.length < CHUNK) break;
      }
    } finally {
      handleBridgeCall("fs_close", { fd: fdh.fd }, () => {
      });
    }
    const out = new Uint8Array(total);
    let off = 0;
    for (const c of chunks) {
      out.set(c, off);
      off += c.length;
    }
    return out;
  }
  function readNativeFileAt(fd, offset, count) {
    const CHUNK = 12288;
    const out = new Uint8Array(count);
    let got = 0;
    while (got < count) {
      const want = Math.min(CHUNK, count - got);
      const h = { data: new Uint8Array(0) };
      handleBridgeCall("fs_pread", { fd, offset: offset + got, count: want }, (r) => {
        if (r && typeof r.data === "string" && r.data.length) {
          const bin = atob(r.data);
          const u = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
          h.data = u;
        }
      });
      if (h.data.length === 0) break;
      out.set(h.data, got);
      got += h.data.length;
      if (h.data.length < want) break;
    }
    return got === count ? out : out.subarray(0, got);
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
    const metaCount = respView.getUint16(4, true);
    const dataLen = respView.getUint32(6, true);
    const dataOffset = 10 + metaCount * 8;
    const resp = { ret };
    if (metaCount > 0) {
      resp._meta = new Array(metaCount);
      for (let i = 0; i < metaCount; i++) {
        resp._meta[i] = respView.getFloat64(10 + i * 8, true);
      }
    }
    if (dataLen > 0) {
      const dataBytes = new Uint8Array(kernelSAB, respOffset + dataOffset, dataLen);
      let str = "";
      for (let i = 0; i < dataLen; i++) str += String.fromCharCode(dataBytes[i]);
      resp.data = btoa(str);
    }
    try {
      apply(resp);
    } catch (e) {
      if (e instanceof RangeError) cpu.setReg(0, BigInt.asUintN(64, BigInt(-14)));
      else throw e;
    }
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
  function waitStdin(timeoutMs) {
    if (!stdinRing) return "timeout";
    if (!stdinRing.empty || stdinRing.writerClosed) return "data";
    const stdinRingIdx = Atomics.load(procTable, procSlotOffset(mySlot) / 4 + 7);
    if (stdinRingIdx < 0) return "timeout";
    const ringI32 = new Int32Array(kernelSAB, stdinRingOffset(stdinRingIdx), 4);
    const wp = Atomics.load(ringI32, 0);
    const result = timeoutMs < 0 ? Atomics.wait(ringI32, 0, wp) : Atomics.wait(ringI32, 0, wp, timeoutMs);
    return result === "timed-out" ? "timeout" : "data";
  }
  function postWrite(fd, data) {
    self.postMessage({ type: "write", fd, data: data.slice() });
  }
  function postExit(code) {
    if (threaded && clearChildTid) {
      try {
        const i32 = new Int32Array(memory.buffer);
        const idx = clearChildTid + GUEST_MEM_BASE >> 2;
        Atomics.store(i32, idx, 0);
        Atomics.notify(i32, idx, Infinity);
      } catch {
      }
      clearChildTid = 0;
    }
    for (const [fd, info] of kernelPipes) {
      const ring = new PipeRing(kernelSAB, pipeRingOffset(info.pipeIndex), PIPE_RING_SIZE);
      if (info.mode === "write") ring.closeWriter();
      else ring.closeReader();
    }
    kernelPipes.clear();
    const high = threaded ? Number(syscalls.getMmapBase()) + GUEST_MEM_BASE : 0;
    self.postMessage({ type: "exit", pid: myPid, code, high });
  }
})();
