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
  var STATE_SIZE = 1024;
  var GUEST_MEM_BASE = 524288;
  var PROT_TABLE_BASE = 36864;
  var PROT_PAINTED_FLAG = 32768;
  var VMA_TABLE_FIRST = 8;
  var VMA_TABLE_HEADER = VMA_TABLE_FIRST * 4;
  var VMA_TABLE_ENTRIES = 65536;
  var VMA_TABLE_PAGES = 13;
  var VMA_TABLE_BYTES = VMA_TABLE_PAGES * 65536;
  var SIG_TABLE_FIRST = 8;
  var SIG_TABLE_PAGES = 1;
  var SIG_TABLE_BYTES = SIG_TABLE_PAGES * 65536;
  var THREAD_TIERS = [
    { regsBase: 268435456, shadowBase: 272629760, vmaBase: 273154048, sigBase: 273154048 + VMA_TABLE_BYTES, pages: 4168 + VMA_TABLE_PAGES + SIG_TABLE_PAGES },
    // 256 MiB guest
    { regsBase: 536870912, shadowBase: 541065216, vmaBase: 542113792, sigBase: 542113792 + VMA_TABLE_BYTES, pages: 8272 + VMA_TABLE_PAGES + SIG_TABLE_PAGES },
    // 512 MiB guest
    { regsBase: 1073741824, shadowBase: 1077936128, vmaBase: 1080033280, sigBase: 1080033280 + VMA_TABLE_BYTES, pages: 16480 + VMA_TABLE_PAGES + SIG_TABLE_PAGES },
    // 1 GiB guest
    { regsBase: 1610612736, shadowBase: 1614807040, vmaBase: 1617952768, sigBase: 1617952768 + VMA_TABLE_BYTES, pages: 24688 + VMA_TABLE_PAGES + SIG_TABLE_PAGES }
    // 1.5 GiB guest
  ];
  var DEFAULT_THREAD_LAYOUT = THREAD_TIERS[1];
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
  var SIMD_CLZ = 101;
  var SIMD_CLS = 102;
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
  var SIMD_SSHL = 110;
  var SIMD_USHL = 111;
  var SIMD_SSRA = 112;
  var SIMD_USRA = 113;
  var SIMD_PAIR_SCALAR = 114;
  var SIMD_FMUL_ELEM = 115;
  var SIMD_FCVT_HALF = 116;
  var SIMD_LDST1_SINGLE = 99;
  function createSimdOp(memory2, regBase) {
    return function simd_op(packed, _extra) {
      const REG_BASE = regBase ? regBase.value : 0;
      const V_OFFSET2 = V_OFFSET + REG_BASE;
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
        const addr = Number(gview.getBigUint64(REG_BASE + rn * 8, true)) + GUEST_MEM_BASE;
        const cp = (from, to) => {
          for (let k = 0; k < es; k++) buf[to + k] = buf[from + k];
        };
        if (isLoad) {
          for (let e = 0; e < lanes; e++) {
            for (let r = 0; r < regCount; r++) {
              const memOff = addr + (e * regCount + r) * es;
              const vOff = V_OFFSET2 + (rd + r & 31) * 16 + e * es;
              cp(memOff, vOff);
            }
          }
          if (!Q) for (let r = 0; r < regCount; r++) buf.fill(0, V_OFFSET2 + (rd + r & 31) * 16 + 8, V_OFFSET2 + (rd + r & 31) * 16 + 16);
        } else {
          for (let e = 0; e < lanes; e++) {
            for (let r = 0; r < regCount; r++) {
              const memOff = addr + (e * regCount + r) * es;
              const vOff = V_OFFSET2 + (rd + r & 31) * 16 + e * es;
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
        const vnA = V_OFFSET2 + rn * 16, vdA = V_OFFSET2 + rd * 16;
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
        const vnA = V_OFFSET2 + rn * 16, vmA = V_OFFSET2 + rm * 16, vdA = V_OFFSET2 + rd * 16;
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
        const vnA = V_OFFSET2 + rn * 16, vmA = V_OFFSET2 + rm * 16, vdA = V_OFFSET2 + rd * 16;
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
        const vnA = V_OFFSET2 + rn * 16, vdA = V_OFFSET2 + rd * 16;
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
        const vnA = V_OFFSET2 + rn * 16, vdA = V_OFFSET2 + rd * 16;
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
      if (op === SIMD_CLZ || op === SIMD_CLS) {
        const es = 1 << size;
        const bits3 = es * 8;
        const lanes = (Q ? 16 : 8) / es;
        const view = new DataView(memory2.buffer);
        const vnA = V_OFFSET2 + rn * 16, vdA = V_OFFSET2 + rd * 16;
        const src = [];
        for (let i = 0; i < lanes; i++) {
          let v = 0n;
          for (let k = 0; k < es; k++) v |= BigInt(buf[vnA + i * es + k]) << BigInt(k * 8);
          src.push(v);
        }
        for (let i = 0; i < lanes; i++) {
          let v = src[i];
          const cls = op === SIMD_CLS;
          if (cls && v >> BigInt(bits3 - 1) & 1n) v = ~v & (1n << BigInt(bits3)) - 1n;
          let n = 0;
          for (let b = bits3 - 1; b >= 0; b--) {
            if (v >> BigInt(b) & 1n) break;
            n++;
          }
          if (cls) n -= 1;
          for (let k = 0; k < es; k++) buf[vdA + i * es + k] = Number(BigInt(n) >> BigInt(k * 8) & 0xffn);
        }
        if (!Q) view.setBigUint64(vdA + 8, 0n, true);
        return 0n;
      }
      if (op === SIMD_XTN) {
        const destEs = 1 << size;
        const srcEs = destEs * 2;
        const n = 8 / destEs;
        const vnA = V_OFFSET2 + rn * 16, vdA = V_OFFSET2 + rd * 16;
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
        aese(buf, V_OFFSET2 + rd * 16, V_OFFSET2 + rn * 16);
        return 0n;
      }
      if (op === SIMD_AESD) {
        aesd(buf, V_OFFSET2 + rd * 16, V_OFFSET2 + rn * 16);
        return 0n;
      }
      if (op === SIMD_AESMC) {
        aesmc(buf, V_OFFSET2 + rd * 16, V_OFFSET2 + rn * 16);
        return 0n;
      }
      if (op === SIMD_AESIMC) {
        aesimc(buf, V_OFFSET2 + rd * 16, V_OFFSET2 + rn * 16);
        return 0n;
      }
      if (op === SIMD_PMULL) {
        pmull(buf, V_OFFSET2 + rd * 16, V_OFFSET2 + rn * 16, V_OFFSET2 + rm * 16, Q === 1);
        return 0n;
      }
      const D = V_OFFSET2 + rd * 16, N = V_OFFSET2 + rn * 16, Mv = V_OFFSET2 + rm * 16;
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
        const addr = Number(gv.getBigUint64(REG_BASE + rn * 8, true)) + GUEST_MEM_BASE;
        const laneAddr = V_OFFSET2 + rd * 16 + index * esize2;
        if (isLoad) for (let k = 0; k < esize2; k++) buf[laneAddr + k] = buf[addr + k];
        else for (let k = 0; k < esize2; k++) buf[addr + k] = buf[laneAddr + k];
        if (isPost) {
          const inc = rm === 31 ? esize2 : Number(gv.getBigUint64(REG_BASE + rm * 8, true));
          gv.setBigUint64(REG_BASE + rn * 8, gv.getBigUint64(REG_BASE + rn * 8, true) + BigInt(inc), true);
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
        const vnA = V_OFFSET2 + rn * 16, vmA = V_OFFSET2 + rm * 16, vdA = V_OFFSET2 + rd * 16;
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
      if (op === SIMD_FCVT_HALF) {
        const srcT = extra & 3, dstT = extra >> 2 & 3;
        const vnH = V_OFFSET2 + rn * 16, vdH = V_OFFSET2 + rd * 16;
        const view = new DataView(memory2.buffer);
        const h2f = (h) => {
          const sign = h >> 15 & 1, exp = h >> 10 & 31, frac = h & 1023;
          if (exp === 0) return (sign ? -1 : 1) * frac * Math.pow(2, -24);
          if (exp === 31) return frac ? NaN : sign ? -Infinity : Infinity;
          return (sign ? -1 : 1) * (1 + frac / 1024) * Math.pow(2, exp - 15);
        };
        const f2h = (f) => {
          if (Number.isNaN(f)) return 32256;
          const sign = f < 0 || Object.is(f, -0) ? 32768 : 0;
          const a2 = Math.abs(f);
          if (a2 === Infinity) return sign | 31744;
          if (a2 === 0) return sign;
          if (a2 >= 65520) return sign | 31744;
          if (a2 < Math.pow(2, -24)) return sign;
          if (a2 < Math.pow(2, -14)) return sign | Math.round(a2 / Math.pow(2, -24));
          let e = Math.floor(Math.log2(a2));
          let m = a2 / Math.pow(2, e) - 1;
          let frac = Math.round(m * 1024);
          if (frac === 1024) {
            frac = 0;
            e += 1;
          }
          if (e > 15) return sign | 31744;
          return sign | e + 15 << 10 | frac;
        };
        let value;
        if (srcT === 3) value = h2f(view.getUint16(vnH, true));
        else if (srcT === 0) value = view.getFloat32(vnH, true);
        else value = view.getFloat64(vnH, true);
        buf.fill(0, vdH, vdH + 16);
        if (dstT === 3) view.setUint16(vdH, f2h(value), true);
        else if (dstT === 0) view.setFloat32(vdH, Math.fround(value), true);
        else view.setFloat64(vdH, value, true);
        return 0n;
      }
      if (op === SIMD_PAIR_SCALAR) {
        const vnP = V_OFFSET2 + rn * 16, vdP = V_OFFSET2 + rd * 16;
        const view = new DataView(memory2.buffer);
        const dbl = size === 3;
        if (extra === 0) {
          const sum = BigInt.asUintN(64, view.getBigUint64(vnP, true) + view.getBigUint64(vnP + 8, true));
          buf.fill(0, vdP, vdP + 16);
          view.setBigUint64(vdP, sum, true);
          return 0n;
        }
        const x = dbl ? view.getFloat64(vnP, true) : view.getFloat32(vnP, true);
        const y = dbl ? view.getFloat64(vnP + 8, true) : view.getFloat32(vnP + 4, true);
        let r;
        switch (extra) {
          case 1:
            r = x + y;
            break;
          // fadd
          // FMAX/FMIN propagate NaN; FMAXNM/FMINNM return the other operand instead.
          case 2:
            r = Number.isNaN(x) || Number.isNaN(y) ? NaN : Math.max(x, y);
            break;
          case 3:
            r = Number.isNaN(x) || Number.isNaN(y) ? NaN : Math.min(x, y);
            break;
          case 4:
            r = Number.isNaN(x) ? y : Number.isNaN(y) ? x : Math.max(x, y);
            break;
          default:
            r = Number.isNaN(x) ? y : Number.isNaN(y) ? x : Math.min(x, y);
            break;
        }
        buf.fill(0, vdP, vdP + 16);
        if (dbl) view.setFloat64(vdP, r, true);
        else view.setFloat32(vdP, Math.fround(r), true);
        return 0n;
      }
      if (op === SIMD_FMUL_ELEM) {
        const idxF = extra & 15, variantF = extra >> 4 & 7, scalarF = (extra & 128) !== 0;
        const dbl = size === 3;
        const es = dbl ? 8 : 4;
        const vnF = V_OFFSET2 + rn * 16, vmF = V_OFFSET2 + rm * 16, vdF = V_OFFSET2 + rd * 16;
        const view = new DataView(memory2.buffer);
        const lanes = scalarF ? 1 : (Q ? 16 : 8) / es;
        const elem = dbl ? view.getFloat64(vmF + idxF * 8, true) : view.getFloat32(vmF + idxF * 4, true);
        const out = new Float64Array(lanes);
        for (let i = 0; i < lanes; i++) {
          const src = dbl ? view.getFloat64(vnF + i * 8, true) : view.getFloat32(vnF + i * 4, true);
          let v = src * elem;
          if (variantF !== 0) {
            const acc = dbl ? view.getFloat64(vdF + i * 8, true) : view.getFloat32(vdF + i * 4, true);
            v = variantF === 1 ? acc + v : acc - v;
          }
          out[i] = v;
        }
        for (let i = 0; i < lanes; i++) {
          if (dbl) view.setFloat64(vdF + i * 8, out[i], true);
          else view.setFloat32(vdF + i * 4, Math.fround(out[i]), true);
        }
        if (scalarF) buf.fill(0, vdF + es, vdF + 16);
        else if (!Q) buf.fill(0, vdF + 8, vdF + 16);
        return 0n;
      }
      const vn = V_OFFSET2 + rn * 16;
      const vm = V_OFFSET2 + rm * 16;
      const vd = V_OFFSET2 + rd * 16;
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
            const vr = V_OFFSET2 + (rn + r & 31) * 16;
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
        // SSHL/USHL: shift each lane by the SIGNED amount in the low byte of the corresponding lane
        // of Vm -- left when positive, right when negative, and a count at or past the lane width
        // clears the lane rather than wrapping the way a wasm shift would.
        case SIMD_SSHL:
        case SIMD_USHL: {
          const signed = op === SIMD_SSHL;
          const ebits = esize * 8;
          for (let i = 0; i < elems; i++) {
            const amtByte = b_[i * esize];
            const amt = amtByte > 127 ? amtByte - 256 : amtByte;
            let v = signed ? getS(aView, i) : getU(aView, i);
            if (amt >= 0) {
              v = amt >= ebits ? 0n : v << BigInt(amt);
            } else {
              const r = -amt;
              if (r >= ebits) v = signed && v < 0n ? -1n : 0n;
              else v >>= BigInt(r);
            }
            put(i, v);
          }
          break;
        }
        // SSRA/USRA: shift right by `extra` and ADD into what Vd already holds.
        case SIMD_SSRA:
        case SIMD_USRA: {
          const signed = op === SIMD_SSRA;
          const dOld = new Uint8Array(16);
          dOld.set(buf.subarray(vd, vd + 16));
          const oldView = new DataView(dOld.buffer);
          const ebits = esize * 8;
          for (let i = 0; i < elems; i++) {
            let v = signed ? getS(aView, i) : getU(aView, i);
            if (extra >= ebits) v = signed && v < 0n ? -1n : 0n;
            else v >>= BigInt(extra);
            const acc = signed ? getS(oldView, i) : getU(oldView, i);
            put(i, acc + v);
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
  function createEnv(memory2, regBase, protActive2, memFault2, protBase2) {
    const env = { memory: memory2, ...helperFunctions, simd_op: createSimdOp(memory2, regBase) };
    if (regBase) env.reg_base = regBase;
    if (protActive2) env.prot_active = protActive2;
    if (protBase2) env.prot_base = protBase2;
    if (memFault2) env.mem_fault = memFault2;
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
  function translateRegionZig(view, entryPC, memory2, shadowBase, singleBlock = false, threaded2 = false, regBase = null, protection = null) {
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
    if (!ex) return null;
    const tbuf = new Uint8Array(ex.memory.buffer);
    const inOff = ex.translator_input_buf_offset();
    tbuf.set(codeBytes, inOff);
    ex.translator_reset();
    const basePCLo = Number(basePC & 0xffffffffn);
    const basePCHi = Number(basePC >> 32n & 0xffffffffn);
    const entryLo = Number(entryPC & 0xffffffffn);
    const entryHi = Number(entryPC >> 32n & 0xffffffffn);
    const rc = ex.translator_translate_region(basePCLo, basePCHi, entryLo, entryHi, codeBytes.length, shadowBase, threaded2 ? 1 : 0, protection ? 1 : 0);
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
      inst = new WebAssembly.Instance(mod, { env: createEnv(memory2, regBase, protection?.protActive ?? null, protection?.memFault ?? null, protection?.protBase ?? null) });
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

  // src/kernel-ipc.ts
  var PROC_TABLE_SLOTS = 256;
  var PROC_SLOT_SIZE = 256;
  var MAILBOX_SLOTS = PROC_TABLE_SLOTS;
  var MAILBOX_SLOT_SIZE = 128;
  var BRIDGE_RESP_SLOTS = PROC_TABLE_SLOTS;
  var BRIDGE_RESP_SLOT_SIZE = 65536;
  var BRIDGE_DATA_MAX = BRIDGE_RESP_SLOT_SIZE - 256;
  var SIGQ_ENTRIES = 64;
  var SIGQ_ENTRY_WORDS = 7;
  var SIGQ_SLOT_WORDS = 2 + SIGQ_ENTRIES * SIGQ_ENTRY_WORDS;
  var SIGQ_HEAD = 0;
  var SIGQ_TAIL = 1;
  var PROC_CRED_WORDS = 4;
  var PROC_CMDLINE_SLOT_SIZE = 256;
  var STDIN_RING_SIZE = 8192;
  var STDIN_RING_COUNT = 8;
  var pg = (n) => n + 4095 & ~4095;
  var PROC_TABLE_OFFSET = 0;
  var MAILBOX_OFFSET = PROC_TABLE_OFFSET + pg(PROC_TABLE_SLOTS * PROC_SLOT_SIZE);
  var STDIN_RING_OFFSET = MAILBOX_OFFSET + pg(MAILBOX_SLOTS * MAILBOX_SLOT_SIZE);
  var GLOBAL_STATE_OFFSET = STDIN_RING_OFFSET + pg(STDIN_RING_COUNT * STDIN_RING_SIZE);
  var BRIDGE_RESP_OFFSET = GLOBAL_STATE_OFFSET + pg(4096);
  var PROC_CMDLINE_OFFSET = BRIDGE_RESP_OFFSET + pg(BRIDGE_RESP_SLOTS * BRIDGE_RESP_SLOT_SIZE);
  var SIG_SENDER_OFFSET = PROC_CMDLINE_OFFSET + pg(PROC_TABLE_SLOTS * PROC_CMDLINE_SLOT_SIZE);
  var SIGQ_OFFSET = SIG_SENDER_OFFSET + pg(PROC_TABLE_SLOTS * SIGQ_SLOT_WORDS * 4);
  var PROC_CRED_OFFSET = SIGQ_OFFSET + pg(PROC_TABLE_SLOTS * SIGQ_SLOT_WORDS * 4);
  var FUTEX_TABLE_OFFSET = PROC_CRED_OFFSET + pg(PROC_TABLE_SLOTS * PROC_CRED_WORDS * 4);
  var PTRACE_TABLE_OFFSET = FUTEX_TABLE_OFFSET + pg(16 * 4 + 1024 * 8 * 4);
  var PTRACE_HDR_WORDS = 8;
  var PTRACE_SCRATCH = 1024;
  var PTRACE_SLOT_SIZE = PTRACE_HDR_WORDS * 4 + PTRACE_SCRATCH;
  var KERNEL_SAB_SIZE = PTRACE_TABLE_OFFSET + pg(PROC_TABLE_SLOTS * PTRACE_SLOT_SIZE);
  var PT_REQ = 0;
  var PT_ACK = 1;
  var PT_OP = 2;
  var PT_ADDR = 3;
  var PT_LEN = 4;
  var PT_DATA_LO = 5;
  var PT_DATA_HI = 6;
  var PT_STATUS = 7;
  function ptraceSlotWord(slot) {
    return (PTRACE_TABLE_OFFSET + slot * PTRACE_SLOT_SIZE) / 4;
  }
  function ptraceScratchOffset(slot) {
    return PTRACE_TABLE_OFFSET + slot * PTRACE_SLOT_SIZE + PTRACE_HDR_WORDS * 4;
  }
  var FUTEX_SLOTS = 1024;
  var FUTEX_SLOT_WORDS = 8;
  var FUTEX_HDR_WORDS = 16;
  var FUTEX_STATE = 0;
  var FUTEX_KEYLO = 1;
  var FUTEX_KEYHI = 2;
  var FUTEX_BITS = 3;
  var FUTEX_PID = 4;
  var FUTEX_LEADER = 5;
  var FUTEX_RESULT = 6;
  var FUTEX_GROUPED = 7;
  var FUTEX_FREE = 0;
  var FUTEX_QUEUED = 1;
  var FUTEX_WOKEN = 2;
  var futexLockIdx = FUTEX_TABLE_OFFSET / 4;
  var futexSlotIdx = (slot) => FUTEX_TABLE_OFFSET / 4 + FUTEX_HDR_WORDS + slot * FUTEX_SLOT_WORDS;
  function futexStateIndex(slot) {
    return futexSlotIdx(slot) + FUTEX_STATE;
  }
  function futexLock(i32) {
    const start = performance.now();
    for (; ; ) {
      if (Atomics.compareExchange(i32, futexLockIdx, 0, 1) === 0) return;
      for (let spin = 0; spin < 128 && Atomics.load(i32, futexLockIdx) !== 0; spin++) {
      }
      if (Atomics.load(i32, futexLockIdx) === 0) continue;
      if (performance.now() - start > 2e3) {
        Atomics.store(i32, futexLockIdx, 1);
        return;
      }
      try {
        Atomics.wait(i32, futexLockIdx, 1, 1);
      } catch {
      }
    }
  }
  function futexUnlock(i32) {
    Atomics.store(i32, futexLockIdx, 0);
    Atomics.notify(i32, futexLockIdx, 1);
  }
  function futexEnqueue(i32, keyLo, keyHi, bits2, pid, leader, grouped = false) {
    const hint = Atomics.load(i32, futexLockIdx + 1);
    for (let n = 0; n < FUTEX_SLOTS; n++) {
      const slot = (hint + n) % FUTEX_SLOTS;
      const base = futexSlotIdx(slot);
      if (Atomics.load(i32, base + FUTEX_STATE) !== FUTEX_FREE) continue;
      Atomics.store(i32, base + FUTEX_KEYLO, keyLo | 0);
      Atomics.store(i32, base + FUTEX_KEYHI, keyHi | 0);
      Atomics.store(i32, base + FUTEX_BITS, bits2 | 0);
      Atomics.store(i32, base + FUTEX_PID, pid | 0);
      Atomics.store(i32, base + FUTEX_LEADER, leader < 0 ? slot : leader);
      Atomics.store(i32, base + FUTEX_RESULT, -1);
      Atomics.store(i32, base + FUTEX_GROUPED, grouped ? 1 : 0);
      Atomics.store(i32, base + FUTEX_STATE, FUTEX_QUEUED);
      Atomics.store(i32, futexLockIdx + 1, (slot + 1) % FUTEX_SLOTS);
      if (slot > Atomics.load(i32, futexLockIdx + 2)) Atomics.store(i32, futexLockIdx + 2, slot);
      return slot;
    }
    return -1;
  }
  function futexDequeue(i32, slot) {
    const own = futexSlotIdx(slot);
    const grouped = Atomics.load(i32, own + FUTEX_GROUPED) === 1;
    Atomics.store(i32, own + FUTEX_STATE, FUTEX_FREE);
    if (!grouped) return;
    for (let s = 0; s < FUTEX_SLOTS; s++) {
      const base = futexSlotIdx(s);
      if (Atomics.load(i32, base + FUTEX_LEADER) === slot) Atomics.store(i32, base + FUTEX_STATE, FUTEX_FREE);
    }
  }
  function futexWake(i32, keyLo, keyHi, bits2, max, requeue = 0, reqLo = 0, reqHi = 0) {
    let woke = 0, moved = 0;
    const top = Math.min(FUTEX_SLOTS, Atomics.load(i32, futexLockIdx + 2) + 1);
    for (let s = 0; s < top; s++) {
      if (woke >= max && moved >= requeue) break;
      const base = futexSlotIdx(s);
      if (Atomics.load(i32, base + FUTEX_STATE) !== FUTEX_QUEUED) continue;
      if (Atomics.load(i32, base + FUTEX_KEYLO) !== (keyLo | 0)) continue;
      if (Atomics.load(i32, base + FUTEX_KEYHI) !== (keyHi | 0)) continue;
      if ((Atomics.load(i32, base + FUTEX_BITS) & bits2) === 0) continue;
      if (woke < max) {
        const leader = Atomics.load(i32, base + FUTEX_LEADER);
        const lbase = futexSlotIdx(leader);
        if (Atomics.load(i32, base + FUTEX_GROUPED) === 1) {
          for (let g = 0; g < FUTEX_SLOTS; g++) {
            const gbase = futexSlotIdx(g);
            if (g !== leader && Atomics.load(i32, gbase + FUTEX_LEADER) === leader) {
              Atomics.store(i32, gbase + FUTEX_STATE, FUTEX_FREE);
            }
          }
        }
        Atomics.store(i32, lbase + FUTEX_RESULT, s);
        Atomics.store(i32, lbase + FUTEX_STATE, FUTEX_WOKEN);
        Atomics.notify(i32, lbase + FUTEX_STATE, 1);
        woke++;
        continue;
      }
      if (moved < requeue) {
        Atomics.store(i32, base + FUTEX_KEYLO, reqLo | 0);
        Atomics.store(i32, base + FUTEX_KEYHI, reqHi | 0);
        moved++;
        continue;
      }
    }
    return { woke, moved };
  }
  function procCredOffset(slot) {
    return PROC_CRED_OFFSET + slot * PROC_CRED_WORDS * 4;
  }
  function sigqSlotWord(slot) {
    return SIG_SENDER_OFFSET / 4 + slot * SIGQ_SLOT_WORDS;
  }
  function sigqEntryWord(slot, index) {
    return sigqSlotWord(slot) + 2 + (index % SIGQ_ENTRIES + SIGQ_ENTRIES) % SIGQ_ENTRIES * SIGQ_ENTRY_WORDS;
  }
  function sigqPush(q, slot, info) {
    const base = sigqSlotWord(slot);
    for (; ; ) {
      const head = Atomics.load(q, base + SIGQ_HEAD);
      const tail = Atomics.load(q, base + SIGQ_TAIL);
      if ((head - tail | 0) >= SIGQ_ENTRIES) return false;
      if (Atomics.compareExchange(q, base + SIGQ_HEAD, head, head + 1 | 0) !== head) continue;
      const e = sigqEntryWord(slot, head);
      Atomics.store(q, e + 1, info.signo | 0);
      Atomics.store(q, e + 2, info.code | 0);
      Atomics.store(q, e + 3, info.pid | 0);
      Atomics.store(q, e + 4, info.uid | 0);
      Atomics.store(q, e + 5, Number(info.value & 0xffffffffn) | 0);
      Atomics.store(q, e + 6, Number(info.value >> 32n & 0xffffffffn) | 0);
      Atomics.store(q, e + 0, head + 1 | 0 || 1);
      return true;
    }
  }
  function sigqDepth(q, slot) {
    const d = Atomics.load(q, sigqSlotWord(slot) + SIGQ_HEAD) - Atomics.load(q, sigqSlotWord(slot) + SIGQ_TAIL) | 0;
    return d > 0 ? d : 0;
  }
  function sigqDrain(q, slot) {
    const base = sigqSlotWord(slot);
    const out = [];
    for (; ; ) {
      const tail = Atomics.load(q, base + SIGQ_TAIL);
      if (tail === Atomics.load(q, base + SIGQ_HEAD)) return out;
      const e = sigqEntryWord(slot, tail);
      if (Atomics.load(q, e + 0) !== (tail + 1 | 0 || 1)) return out;
      out.push({
        signo: Atomics.load(q, e + 1),
        code: Atomics.load(q, e + 2),
        pid: Atomics.load(q, e + 3),
        uid: Atomics.load(q, e + 4),
        value: BigInt(Atomics.load(q, e + 6) >>> 0) << 32n | BigInt(Atomics.load(q, e + 5) >>> 0)
      });
      Atomics.store(q, e + 0, 0);
      Atomics.store(q, base + SIGQ_TAIL, tail + 1 | 0);
    }
  }
  function sigqPending(q, slot) {
    const base = sigqSlotWord(slot);
    return Atomics.load(q, base + SIGQ_HEAD) !== Atomics.load(q, base + SIGQ_TAIL);
  }
  var PIPE_WRITE_POS = 0;
  var PIPE_READ_POS = 1;
  var PIPE_WRITER_CLOSED = 2;
  var PIPE_READER_CLOSED = 3;
  var PIPE_CAPACITY_W = 4;
  var PIPE_ASYNC_W = 5;
  var PIPE_ALLOC_W = 6;
  var PIPE_SCM_W = 7;
  var PIPE_HEADER_SIZE = 32;
  var PAGE_SIZE = 4096;
  var PIPE_DEF_PAGES = 16;
  function pipeRingSize(pages) {
    return PIPE_HEADER_SIZE + pages * PAGE_SIZE;
  }
  var PIPE_RING_SIZE = pipeRingSize(PIPE_DEF_PAGES);
  var SHM_ARENA_SIZE = 64 * 1024 * 1024;
  var PIPE_PAGES_SOFT = SHM_ARENA_SIZE / PAGE_SIZE / 2;
  function procCmdlineOffset(slot) {
    return PROC_CMDLINE_OFFSET + slot * PROC_CMDLINE_SLOT_SIZE;
  }
  var PROC_EMPTY = 0;
  var PROC_RUNNING = 1;
  var PROC_ZOMBIE = 2;
  var PROC_STOPPED = 4;
  var PROC_PID = 0;
  var PROC_PPID = 1;
  var PROC_PGID = 2;
  var PROC_STATE = 3;
  var PROC_EXIT_CODE = 4;
  var PROC_WAIT_FUTEX = 5;
  var PROC_TGID = 9;
  var PROC_BLOCKED = 15;
  var PROC_SWAP_REQ = 14;
  var PROC_COMM = 10;
  var PROC_UTIME_MS = 16;
  var PROC_STIME_MS = 17;
  var PROC_NICE = 18;
  var PROC_PDEATHSIG = 19;
  var PROC_POLICY = 20;
  var PROC_RTPRIO = 21;
  var PROC_SCHED_FLAGS = 22;
  var PROC_AFFINITY = 23;
  var PROC_DL_RUNTIME = 25;
  var PROC_DL_DEADLINE = 27;
  var PROC_DL_PERIOD = 29;
  var PROC_STOP_SIG = 31;
  var PROC_STOP_SEEN = 32;
  var PROC_CONT_PENDING = 33;
  var PROC_SYSCALL = 34;
  var PROC_SID = 35;
  var PROC_EXECED = 36;
  var PROC_SUBREAPER = 37;
  var PROC_RLIM_SIGPENDING = 38;
  var PROC_TRACER = 39;
  var PROC_TRACE_CMD = 40;
  var PT_RESUME_NONE = 0;
  var PT_RESUME_CONT = 1;
  var PT_RESUME_DETACH = 2;
  var PROC_VFORK = 41;
  var PROC_MAXRSS_KB = 42;
  var PROC_IOPRIO = 43;
  var PROC_RUID = 44;
  var SIG_INT = 1 << 2;
  var SIG_PIPE = 1 << 13;
  var SIG_CHLD = 1 << 17;
  var MBOX_READY = 0;
  var MBOX_VALUE0 = 2;
  var STDIN_DATA_SIZE = STDIN_RING_SIZE - PIPE_HEADER_SIZE;
  var GLOBAL_PIPE_MAX = 4;
  var GLOBAL_IO_GEN = 5;
  var GLOBAL_INOTIFY_WATCHES = 6;
  var GLOBAL_BOOT_MS_LO = 7;
  var GLOBAL_BOOT_MS_HI = 8;
  var GLOBAL_MQ_MSG_MAX = 9;
  var GLOBAL_MQ_MSGSIZE_MAX = 10;
  var GLOBAL_MQ_QUEUES_MAX = 11;
  var GLOBAL_CLK_SEQ = 12;
  var GLOBAL_CLK_OFF_SEC = 13;
  var GLOBAL_CLK_OFF_NSEC = 14;
  var GLOBAL_MSGMNI = 15;
  var GLOBAL_UTS_LEN = 65;
  var GLOBAL_HOSTNAME_OFFSET = 64;
  var GLOBAL_DOMAINNAME_OFFSET = GLOBAL_HOSTNAME_OFFSET + GLOBAL_UTS_LEN;
  var GLOBAL_MSG_NEXT_ID = 49;
  var GLOBAL_SEM_NEXT_ID = 50;
  var GLOBAL_SHM_NEXT_ID = 51;
  var MEM_PAGE_SIZE = 65536;
  function procSlotOffset(slot) {
    return PROC_TABLE_OFFSET + slot * PROC_SLOT_SIZE;
  }
  function mailboxSlotOffset(slot) {
    return MAILBOX_OFFSET + slot * MAILBOX_SLOT_SIZE;
  }
  function stdinRingOffset(stdinIndex) {
    return STDIN_RING_OFFSET + stdinIndex * STDIN_RING_SIZE;
  }
  var ioGenArr = null;
  function bindIoGeneration(globals) {
    ioGenArr = globals;
  }
  function bumpIoGeneration() {
    if (!ioGenArr) return;
    Atomics.add(ioGenArr, GLOBAL_IO_GEN, 1);
    Atomics.notify(ioGenArr, GLOBAL_IO_GEN);
  }
  function readIoGeneration() {
    return ioGenArr ? Atomics.load(ioGenArr, GLOBAL_IO_GEN) : 0;
  }
  function waitIoGeneration(seen, timeoutMs) {
    if (!ioGenArr) return false;
    const r = timeoutMs < 0 ? Atomics.wait(ioGenArr, GLOBAL_IO_GEN, seen) : Atomics.wait(ioGenArr, GLOBAL_IO_GEN, seen, timeoutMs);
    return r !== "timed-out";
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
  var ELOOP = 40;
  var ENOENT = 2;
  var EPERM = 1;
  var EOPNOTSUPP = 95;
  var RWF_SUPPORTED = 63;
  var AT_SYMLINK_NOFOLLOW = 256;
  var UTIME_NOW = (1 << 30) - 1;
  var UTIME_OMIT = (1 << 30) - 2;
  var EINTR = 4;
  var E2BIG = 7;
  var EXDEV = 18;
  var EFBIG = 27;
  var ENOTSOCK = 88;
  function sockTripleErrno(domain, typeArg, protocol, euid) {
    const SOCK_STREAM = 1, SOCK_DGRAM = 2, SOCK_RAW = 3, SOCK_SEQPACKET = 5;
    const IPPROTO_ICMP = 1, IPPROTO_TCP = 6, IPPROTO_UDP = 17;
    const type = typeArg & 255;
    if (domain !== 1 && domain !== 2 && domain !== 10) return EAFNOSUPPORT;
    if (type !== SOCK_STREAM && type !== SOCK_DGRAM && type !== SOCK_RAW && type !== SOCK_SEQPACKET) return EINVAL;
    if (typeArg & ~(255 | 524288 | 2048)) return EINVAL;
    if (domain === 1) return protocol === 0 ? 0 : EPROTONOSUPPORT;
    const ok = protocol === 0 ? type !== SOCK_RAW : type === SOCK_STREAM ? protocol === IPPROTO_TCP : type === SOCK_DGRAM ? protocol === IPPROTO_UDP : type === SOCK_RAW ? protocol === IPPROTO_ICMP || protocol === IPPROTO_TCP || protocol === IPPROTO_UDP : false;
    if (!ok) return EPROTONOSUPPORT;
    if (type === SOCK_RAW && euid !== 0) return EPERM;
    return 0;
  }
  function sockOptLevel(level) {
    return level === 0 || level === 1 || level === 6 || level === 41 || level === 58 || level === 132;
  }
  function ipv6Groups(host) {
    const out = new Array(8).fill(0);
    const [head, tail] = host.split("::");
    const h = head ? head.split(":").filter((x) => x.length) : [];
    const t = tail !== void 0 ? tail.split(":").filter((x) => x.length) : [];
    for (let i = 0; i < h.length && i < 8; i++) out[i] = parseInt(h[i], 16) & 65535;
    for (let i = 0; i < t.length && i < 8; i++) out[8 - t.length + i] = parseInt(t[i], 16) & 65535;
    return out;
  }
  function ipv6Text(groups) {
    let bestStart = -1, bestLen = 0;
    for (let i = 0; i < 8; i++) {
      if (groups[i] !== 0) continue;
      let j = i;
      while (j < 8 && groups[j] === 0) j++;
      if (j - i > bestLen) {
        bestLen = j - i;
        bestStart = i;
      }
      i = j - 1;
    }
    if (bestLen > 1) {
      const head = groups.slice(0, bestStart).map((g) => g.toString(16)).join(":");
      const tail = groups.slice(bestStart + bestLen).map((g) => g.toString(16)).join(":");
      return `${head}::${tail}`;
    }
    return groups.map((g) => g.toString(16)).join(":");
  }
  function sockOptKnown(level, name) {
    switch (level) {
      case 1:
        return name >= 1 && name <= 75;
      // SOL_SOCKET: SO_DEBUG .. SO_RCVMARK
      case 0:
        return name >= 1 && name <= 25 || name >= 32 && name <= 50;
      // IPPROTO_IP
      case 6:
        return name >= 1 && name <= 38;
      // IPPROTO_TCP: TCP_NODELAY .. TCP_TX_DELAY
      case 41:
        return name >= 1 && name <= 80;
      // IPPROTO_IPV6
      case 58:
        return name >= 1 && name <= 24;
      // IPPROTO_ICMPV6
      case 132:
        return name >= 0 && name <= 33;
      // IPPROTO_SCTP
      default:
        return false;
    }
  }
  var EPROTONOSUPPORT = 93;
  var EAFNOSUPPORT = 97;
  var ENOPROTOOPT = 92;
  var EADDRINUSE = 98;
  var EADDRNOTAVAIL = 99;
  var ENOTCONN = 107;
  var EISCONN = 106;
  var CAP_DAC_READ_SEARCH = 2;
  var NATIVE_FS_MOUNT_ID = 1;
  var EMSGSIZE = 90;
  var EPIPE = 32;
  var ENODEV = 19;
  var EMFILE = 24;
  var ENOTDIR = 20;
  var ECHILD = 10;
  var ESPIPE = 29;
  var EISDIR = 21;
  var EBUSY = 16;
  var MAX_FILE_BYTES = 0x1000000000000n;
  var EAGAIN = 11;
  var EFAULT = 14;
  var EACCES = 13;
  var ENOMEM = 12;
  var EEXIST = 17;
  var ENAMETOOLONG = 36;
  var MADV_VALID = /* @__PURE__ */ new Set([0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 100, 101]);
  var IOV_MAX = 1024;
  var NR_OPEN = 1048576;
  var RLIMIT_STACK = 3;
  var RLIMIT_FSIZE = 1;
  var RLIMIT_NOFILE = 7;
  var RLIMIT_MEMLOCK = 8;
  var RLIMIT_SIGPENDING = 11;
  var RLIMIT_NICE = 13;
  var RLIMIT_RTPRIO = 14;
  var SIGPENDING_DEFAULT = 15e3;
  var RLIM_INFINITY = 0xFFFFFFFFFFFFFFFFn;
  var RLIM_NLIMITS = 16;
  var REBOOT_MAGIC2 = /* @__PURE__ */ new Set([672274793, 85072278, 369367448, 537993216]);
  var IN_ACCESS = 1;
  var IN_MODIFY = 2;
  var IN_ATTRIB = 4;
  var IN_CLOSE_WRITE = 8;
  var IN_CLOSE_NOWRITE = 16;
  var IN_OPEN = 32;
  var IN_MOVED_FROM = 64;
  var IN_MOVED_TO = 128;
  var IN_CREATE = 256;
  var IN_DELETE = 512;
  var IN_DELETE_SELF = 1024;
  var IN_MOVE_SELF = 2048;
  var IN_ISDIR = 1073741824;
  var PRIO_PROCESS = 0;
  var PRIO_PGRP = 1;
  var PRIO_USER = 2;
  var ESRCH = 3;
  var MAX_FDS = 1024;
  var EPOLL_EVENT_SIZE = 16;
  var EPOLL_MAX_DEPTH = 5;
  var EPOLL_DATA_OFFSET = 8;
  var MAX_PROCESS_BYTES = 1856 * 1024 * 1024;
  var O_NONBLOCK = 2048;
  var SIG_DFL = 0n;
  var SIG_IGN = 1n;
  var SIGALRM = 14;
  var SIGVTALRM = 26;
  var SIGPROF = 27;
  var ITIMER_SIGNAL = [SIGALRM, SIGVTALRM, SIGPROF];
  var CLOCK_REALTIME = 0;
  var CLOCK_MONOTONIC = 1;
  var CLOCK_CPU_OWN = 2;
  var CLOCK_THREAD_CPUTIME_ID = 3;
  var CLOCK_BOOTTIME = 7;
  var CLOCK_REALTIME_ALARM = 8;
  var CLOCK_BOOTTIME_ALARM = 9;
  var SIGEV_SIGNAL = 0;
  var SIGEV_NONE = 1;
  var SIGEV_THREAD = 2;
  var SIGEV_THREAD_ID = 4;
  var SIGRTMAX = 64;
  var O_TMPFILE_BIT = 0o20000000n;
  var O_DIRECTORY_B = 0o40000n;
  var O_NOFOLLOW_B = 0o100000n;
  var O_PATH_B = 0o10000000n;
  var O_CLOEXEC_B = 0o2000000n;
  var VALID_OPEN_FLAGS = 3n | 0o100n | 0o200n | 0o400n | 0o1000n | 0o2000n | 0o4000n | 0o10000n | 0o20000n | O_DIRECTORY_B | O_NOFOLLOW_B | 0o200000n | 0o400000n | 0o1000000n | O_CLOEXEC_B | 0o4000000n | O_PATH_B | O_TMPFILE_BIT;
  var O_PATH_FLAGS = O_DIRECTORY_B | O_NOFOLLOW_B | O_PATH_B | O_CLOEXEC_B;
  var RESOLVE_NO_XDEV = 0x01n;
  var RESOLVE_NO_MAGICLINKS = 0x02n;
  var RESOLVE_NO_SYMLINKS = 0x04n;
  var RESOLVE_BENEATH = 0x08n;
  var RESOLVE_IN_ROOT = 0x10n;
  var RESOLVE_CACHED = 0x20n;
  var VALID_RESOLVE = RESOLVE_NO_XDEV | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_SYMLINKS | RESOLVE_BENEATH | RESOLVE_IN_ROOT | RESOLVE_CACHED;
  var TFD_CLOEXEC = 524288;
  var TFD_NONBLOCK = 2048;
  var TIMER_ABSTIME = 1;
  function clampOverrun(n) {
    return n > 2147483647 ? 2147483647 : n;
  }
  var OVERRUN_CEIL = 2147483648;
  var TFD_TIMER_ABSTIME = 1;
  var TFD_TIMER_CANCEL_ON_SET = 2;
  var SIGPIPE = 13;
  var SIGCHLD = 17;
  var SIGNAL_TRAMPOLINE = 4096;
  var SIG_IGNORE_DEFAULT = /* @__PURE__ */ new Set([SIGCHLD, 23, 28]);
  var SIG_STOP_DEFAULT = /* @__PURE__ */ new Set([19, 20, 21, 22]);
  var SIG_CORE_DEFAULT = /* @__PURE__ */ new Set([
    3,
    // SIGQUIT
    4,
    // SIGILL
    5,
    // SIGTRAP
    6,
    // SIGABRT
    7,
    // SIGBUS
    8,
    // SIGFPE
    11,
    // SIGSEGV
    24,
    // SIGXCPU
    25,
    // SIGXFSZ
    31
    // SIGSYS
  ]);
  var SI_CODE = 8;
  var SI_ADDR = 16;
  var SI_PID = 16;
  var SI_UID = 20;
  var SI_VALUE = 24;
  var UC_OFF = 128;
  var UC_SIGMASK = UC_OFF + 40;
  var UC_MCONTEXT = UC_OFF + 176;
  var MC_REGS = UC_MCONTEXT + 8;
  var MC_SP = UC_MCONTEXT + 256;
  var MC_PC = UC_MCONTEXT + 264;
  var MC_PSTATE = UC_MCONTEXT + 272;
  var MC_VREGS = UC_MCONTEXT + 296;
  var SIGNAL_FRAME_SIZE = 1216;
  var SIG_WITH_ADDR = /* @__PURE__ */ new Set([4, 7, 8, 11]);
  var ENOTTY = 25;
  var SIOCATMARK = 35077;
  var SIOCGIFNAME = 35088;
  var SIOCGIFCONF = 35090;
  var SIOCGIFFLAGS = 35091;
  var SIOCSIFFLAGS = 35092;
  var SIOCGIFADDR = 35093;
  var SIOCGIFDSTADDR = 35095;
  var SIOCGIFBRDADDR = 35097;
  var SIOCGIFNETMASK = 35099;
  var SIOCGIFMTU = 35105;
  var SIOCGIFHWADDR = 35111;
  var SIOCGIFINDEX = 35123;
  var TCGETS = 21505;
  var TCSETS = 21506;
  var TIOCGWINSZ = 21523;
  var TIOCSWINSZ = 21524;
  var F_DUPFD = 0;
  var CAP_SYS_PTRACE = 19;
  var CAP_SYS_CHROOT = 18;
  var CAP_SYS_NICE = 23;
  var CAP_IPC_LOCK = 14;
  var CAP_SETPCAP = 8;
  var CAP_SETGID = 6;
  var CAP_SYS_TIME = 25;
  var CAP_SYS_BOOT = 22;
  var CAP_SYS_ADMIN = 21;
  var CAP_SYS_RESOURCE = 24;
  var CAP_WAKE_ALARM = 35;
  var ALL_CAPS = (1n << 41n) - 1n;
  var PAGE_NO_BACKING = 8;
  var F_OWNER_TID = 0;
  var F_OWNER_PID = 1;
  var F_OWNER_PGRP = 2;
  var F_GETFD = 1;
  var F_SETFD = 2;
  var F_GETFL = 3;
  var F_SETFL = 4;
  var F_DUPFD_CLOEXEC = 1030;
  var BRIDGE_READ_MAX = BRIDGE_DATA_MAX;
  var DEVICES = /* @__PURE__ */ new Map([
    ["1:3", "null"],
    ["1:5", "zero"],
    ["1:7", "full"],
    ["1:8", "random"],
    ["1:9", "random"],
    ["5:0", "tty"],
    ["5:1", "tty"]
  ]);
  function devKind(major, minor) {
    return DEVICES.get(`${major}:${minor}`);
  }
  var NET_TRACE = false;
  var FD_TRACE = false;
  var SyscallHandler = class _SyscallHandler {
    nt(s) {
      if (NET_TRACE) this.io.writeStderr(new TextEncoder().encode("[net] " + s + "\n"));
    }
    ft(s) {
      if (FD_TRACE) this.io.writeStderr(new TextEncoder().encode("[fd] " + s + "\n"));
    }
    brk;
    // where the heap started. brk(2) may give memory back, but never below the address the program
    // was loaded with -- below that lies the image itself.
    brkFloor;
    /** which membarrier registrations this process has made: bit 0 private expedited, bit 1 sync core. */
    membarrierRegistered = 0;
    /** where set_robust_list left this thread's list of futexes to release if it dies holding them. */
    robustListHead = 0n;
    /**
     * file seals, by open-file id. a seal is a promise about a file that cannot be taken back --
     * that it will never grow, never shrink, or never be written again -- and it is what makes a
     * memfd safe to hand to something you do not trust.
     */
    fileSeals = /* @__PURE__ */ new Map();
    sealsOf(e) {
      return e && e.type === "native" ? this.fileSeals.get(e.ofdId) ?? 0 : 0;
    }
    // the exclusive-monitor shadow lives at ZIG_SHADOW_BASE (32MB) and is tiny
    // (1 byte per 512 guest bytes, so <=256KB for a 128MB guest). reserve [32MB,33MB)
    // for it and start the mmap arena above, so mmap'd shared libraries (ld.so mmaps
    // libc.so.6) don't collide with the shadow. mmap grows up toward the shadow_base
    // ceiling of the top-of-memory shadow... here shadow is low, so cap mmap well above.
    mmapBase = 0x2100000n;
    // 33MB — just above the 1MB shadow reservation
    /** the bottom of the mmap arena. everything above it was handed out by mmap and is described
     *  by a vma, so the gaps between those vmas are genuinely free and may be handed out again. */
    mmapArenaStart = 34603008;
    /** where to begin looking for free address space. see allocMmapRange. */
    mmapHint = 34603008;
    /**
     * grow the guest's wasm memory to cover `needed` bytes. false if the engine would not.
     *
     * a refused grow throws a RangeError, and RangeError is what a bad guest pointer throws too, so
     * it was being reported as EFAULT -- an errno mmap can never return, and one that tells a program
     * its arguments were wrong when what actually happened is that the machine ran out of memory.
     */
    /** report WHY a grow was refused. "out of memory" is what the guest sees either way, but the
     *  numbers say whether the platform said no or we did. */
    growFailed = null;
    growTo(memory2, needed, chunkPages = 1) {
      if (needed <= memory2.buffer.byteLength) return true;
      const currentPages = memory2.buffer.byteLength / 65536;
      const minPages = Math.ceil(needed / 65536);
      const capPages = Math.floor(this.maxBytes / 65536);
      const chunked = Math.ceil(minPages / chunkPages) * chunkPages;
      const geometric = chunkPages > 1 ? Math.ceil(currentPages * 1.5) : 0;
      const want = Math.min(Math.max(chunked, geometric, minPages), capPages);
      if (want <= currentPages) return needed <= memory2.buffer.byteLength;
      try {
        memory2.grow(want - currentPages);
      } catch (e) {
        if (want !== minPages) {
          try {
            memory2.grow(minPages - currentPages);
          } catch (e2) {
            this.growFailed?.(currentPages, minPages, want, capPages, String(e2?.name));
            return false;
          }
        } else {
          this.growFailed?.(currentPages, minPages, want, capPages, String(e?.name));
          return false;
        }
      }
      return needed <= memory2.buffer.byteLength;
    }
    /** vm.max_map_count: how many distinct mappings one process may hold. linux's default. */
    maxMapCount = 65530;
    getMaxMapCount() {
      return this.maxMapCount;
    }
    setMaxMapCount(n) {
      if (!Number.isFinite(n) || n < 1) return -EINVAL;
      this.maxMapCount = Math.min(n, VMA_TABLE_ENTRIES);
      return 0;
    }
    /** how many mappings the process holds, for /proc and for the limit check. */
    vmaCount() {
      return this.vmas.length;
    }
    /**
     * What this process actually COSTS, as opposed to how far its memory has been grown.
     *
     * wasm growth is lazy: measured on device, 158 memories grown to 32 MiB coexist happily, exactly
     * like 156 that merely DECLARED 32 MiB -- pages become real when written, not when grown. so
     * charging a process for buffer.byteLength (which is what the kernel's budget used to do) is not
     * conservative, it is wrong: it refused forks that would have fit comfortably. what a process
     * genuinely holds is its image and stack, its heap up to the break, and its mappings.
     */
    footprintBytes(stackTop) {
      let n = Math.max(Number(this.brk), stackTop) + GUEST_MEM_BASE;
      for (const v of this.vmas) n += v.end - v.start;
      return n;
    }
    /**
     * find `len` bytes of free guest address space for a mapping.
     *
     * a bump allocator over the arena never reuses anything, so a program that maps and unmaps in a
     * loop -- which is what a general-purpose allocator does with any large request -- walks its
     * address space upward forever, and since wasm memory can only grow, every lap COMMITS more real
     * memory. mapping and unmapping one megabyte two hundred times cost two hundred megabytes and
     * 12 ms a lap. so: first fit through the holes, and only extend the arena when none is big enough.
     */
    allocMmapRange(len) {
      const found = this.scanForGap(Math.max(this.mmapHint, this.mmapArenaStart), len);
      if (found >= 0) {
        this.mmapHint = found + len;
        return BigInt(found);
      }
      if (this.mmapHint > this.mmapArenaStart) {
        const retry = this.scanForGap(this.mmapArenaStart, len);
        if (retry >= 0) {
          this.mmapHint = retry + len;
          return BigInt(retry);
        }
      }
      const addr = this.mmapBase;
      this.mmapBase += BigInt(len);
      this.mmapHint = Number(this.mmapBase);
      this.vmaDirty = true;
      return addr;
    }
    /** lowest free address >= `from` with `len` bytes behind it, or -1 if the arena has no such gap. */
    scanForGap(from, len) {
      const top = Number(this.mmapBase);
      let cur = from;
      for (let k = this.vmaLower(cur); k < this.vmas.length; k++) {
        const v = this.vmas[k];
        if (v.start >= top) break;
        if (v.start - cur >= len) return cur;
        if (v.end > cur) cur = v.end;
      }
      return top - cur >= len ? cur : -1;
    }
    // brk/mmap ceiling. a threaded process uses a lower ceiling (its shared memory places the per-thread
    // register blocks + atomics shadow just above this, and the shared reservation is bounded so several
    // threaded processes can coexist). single-threaded processes keep the full 1.5 GiB.
    maxBytes = MAX_PROCESS_BYTES;
    // MAP_SHARED regions (guest offsets). shared across fork: a child's writes to these
    // must survive the parent-memory restore in the serialized fork model.
    sharedRegions = [];
    // installed by the worker: a new MAP_SHARED region needs a block in the shared arena, and an
    // unmapped one has to give it back. see shared-map.ts for why the arena exists at all.
    sharedMapHook = null;
    /** capacity of the ring behind a pipe fd, or null if the fd is not a pipe. pipes are not all the
     *  same size -- the kernel shrinks new ones when many are open -- and F_GETPIPE_SZ is a promise
     *  a program sizes its writes by, so it has to come from the actual ring. */
    pipeCapacityHook = null;
    /** called when a descriptor's async-notification state changes (owner, signal, or the O_ASYNC
     *  flag). the worker uses it to tell the kernel who to signal when the pipe becomes ready. */
    ownerChanged = null;
    /** called when the signal mask changes so a newly-deliverable pending signal is not left waiting
     *  for the next periodic check. */
    signalsUnblocked = null;
    /** reports the first time each (syscall, errno) refusal happens. a call we answer ENOSYS or
     *  EINVAL to is often invisible: libc tries something, we refuse, it falls back and succeeds, and
     *  the only trace left is a stale errno that surfaces somewhere unrelated. */
    refusalHook = null;
    seenRefusals = /* @__PURE__ */ new Set();
    /**
     * Record that a syscall answered with an error, once per (syscall, errno) pair.
     *
     * A failing syscall is usually invisible: the caller recovers, carries on, and the only evidence
     * is a wrong result somewhere else entirely. EAGAIN and EINTR are left out because they are
     * ordinary flow control rather than answers -- everything else is worth knowing about.
     */
    noteResult(nr, ret) {
      if (ret >= 0n || ret < -4095n) return;
      const errno = Number(-ret);
      if (errno === 11 || errno === 4) return;
      const key = nr * 4096 + errno;
      if (this.seenRefusals.has(key)) return;
      this.seenRefusals.add(key);
      this.refusalHook?.(nr, errno);
    }
    /** fs.pipe-max-size, shared across every process. the ceiling on any one pipe's capacity, and
     *  what F_SETPIPE_SZ measures an unprivileged request against. */
    pipeMaxHook = null;
    /** read from a bridge-backed file, gathering as many transport-sized pieces as the caller asked
     *  for. returns the bytes read (possibly short only at end of file) or a positive errno. */
    readNativeHook = null;
    /** resize a pipe's ring to `bytes`, returning what it actually got (or -1). */
    pipeResizeHook = null;
    /** wire the /proc/sys files that name real settings to those settings. called once the vfs exists. */
    bindSysctls = null;
    /** PR_SET_CHILD_SUBREAPER lives in the shared process table: the kernel reads it when it
     *  reparents orphans, and it survives an exec, which this handler does not. */
    subreaperHook = null;
    /**
     * read and validate a capability header, as capget and capset both must.
     *
     * an unknown version is EINVAL, and the kernel writes back the one it prefers so a caller can
     * discover it by asking with a bad one -- which is exactly what libcap does at startup.
     */
    capHeader(bytes, view, ptr) {
      if (ptr === 0 || this.ptrWriteFaults(bytes, ptr, 8)) return { err: EFAULT };
      const base = ptr + GUEST_MEM_BASE;
      const version = view.getUint32(base, true);
      if (version !== 429392688 && version !== 537333798 && version !== 537396514) {
        view.setUint32(base, 537396514, true);
        return { err: EINVAL };
      }
      return { version, pid: view.getInt32(base + 4, true) };
    }
    /** whether a pid names a live process, for the calls that answer ESRCH about one. */
    pidExists(pid) {
      if (pid === this.currentPid) return true;
      const list = this.prioHook?.list();
      return list ? list.some((p) => p.pid === pid) : false;
    }
    /** every socket this process holds, so a forked child can take its own reference to each. */
    inheritedSocketIds() {
      const ids = /* @__PURE__ */ new Set();
      for (const e of this.fdTable.values()) if (e.type === "socket") ids.add(e.socketId);
      return [...ids];
    }
    /** set by a bridge apply that could not finish: the worker runs the syscall again. */
    retryAfterBridge = false;
    takeRetry() {
      const r = this.retryAfterBridge;
      this.retryAfterBridge = false;
      return r;
    }
    /** the CURRENT parent, read from the shared process table rather than remembered. */
    ppidHook = null;
    /** hand a host descriptor back, for closes that cannot wait for a syscall reply. */
    closeNativeHook = null;
    closeSocketHook = null;
    /** how to give up a unix socket: its rings and its name live in the worker, not in this table. */
    closeUnixHook = null;
    /** the timer slack this process is using, for /proc/self/timerslack_ns. */
    getTimerSlack() {
      return this.timerSlackNs;
    }
    /** set it from the /proc file. writing zero restores the inherited default, as prctl does. */
    setTimerSlack(v) {
      if (!Number.isFinite(v) || v < 0) return -EINVAL;
      this.timerSlackNs = v === 0 ? this.defaultTimerSlackNs : v;
      return 0;
    }
    /** who is on the other end of a unix-domain socket, for SO_PEERCRED. only the worker knows:
     *  the connection is made through the kernel's name table, not through this fd table. */
    peerCredHook = null;
    sharedUnmapHook = null;
    // writable MAP_SHARED mappings of real (native) files. the guest writes into the mapping and
    // expects those bytes flushed back to the file on msync/munmap (this is how apk extracts
    // binaries). we track them and pwrite the dirty range through the bridge on flush.
    fileMmaps = [];
    // guest ranges mapped PROT_NONE (inaccessible). a user pointer landing here faults with
    // EFAULT, which is how LTP's tst_get_bad_addr() probes are built.
    protNone = [];
    // page-aligned, non-overlapping locked-memory intervals (mmap MAP_LOCKED / mlock). their
    // total is reported as VmLck in /proc/self/status.
    // MLOCK_ONFAULT locks a range without populating it: the pages join the lock as they are
    // touched. that is the difference between "these pages are here now" and "these pages will stay
    // once they arrive", and mincore has to tell them apart.
    lockedRegions = [];
    // set by mlockall(MCL_FUTURE): every mapping made from now on is locked as it is made.
    lockFuture = false;
    // virtual memory areas (guest offsets == guest-visible addresses), sorted, non-overlapping.
    // drives MAP_FIXED_NOREPLACE overlap detection and the synthetic /proc/self/maps. prot holds
    // the PROT_* bits; shared distinguishes p/s in the maps perms column.
    // `maxProt` is the most this mapping may EVER be granted. mmap already refuses to hand out access
    // the descriptor lacks, but mprotect can be asked for it later, and a mapping is not allowed to
    // acquire by a second call what it was refused by the first.
    vmas = [];
    /**
     * The regions of the address space a fork actually has to copy.
     *
     * Everything else is untouched, and sweeping it is not free: a snapshot used to scan from zero to
     * the top of the mmap arena looking for non-zero pages, which for an ordinary shell means reading
     * 34 MiB to find the two that matter -- 7 to 20 ms on EVERY fork, most of a process's whole
     * creation cost. It also missed anything mapped MAP_FIXED above the bump pointer, which a fork
     * silently failed to copy.
     *
     * [0, brk) covers the loaded image and the heap; STACK_TOP bounds the stack, which grows down and
     * has no vma of its own; the rest is whatever is mapped.
     */
    snapshotRanges(stackTop) {
      return this.readAddressSpace(() => this.snapshotRangesLocked(stackTop));
    }
    snapshotRangesLocked(stackTop) {
      const out = [
        { start: 0, end: Math.max(Number(this.brk), stackTop) }
      ];
      for (const v of this.vmas) out.push({ start: v.start, end: v.end });
      out.sort((a, b) => a.start - b.start);
      const merged = [];
      for (const r of out) {
        const last = merged[merged.length - 1];
        if (last && r.start <= last.end) last.end = Math.max(last.end, r.end);
        else merged.push({ ...r });
      }
      return merged;
    }
    cwd = "/";
    // the filesystem identity. linux keeps it equal to the effective one except when setfsuid or
    // setfsgid has deliberately moved it, and every file permission check uses THIS rather than the
    // effective id -- which is the entire reason the pair exists.
    fsuid = 0;
    fsgid = 0;
    tpidr = 0n;
    umask = 18;
    fdCloexec = /* @__PURE__ */ new Set();
    // who receives the I/O-ready signal for a descriptor. linux calls this the descriptor's OWNER,
    // and it is a real piece of state: F_SETOWN records it and F_GETOWN reads it back. answering
    // F_GETOWN with the caller's own pid regardless of what was set is a plausible-looking lie that
    // breaks the moment a program hands a descriptor's notifications to somebody else.
    fdOwner = /* @__PURE__ */ new Map();
    // linux privilege is not "am I uid 0", it is a set of CAPABILITIES. a process can be uid 0 and
    // still lack the one permission it is asking for, and a process can hold a capability without
    // being root at all -- which is how containers give a program exactly the power it needs. these
    // used to be faked: capget always answered "all of them" and capset accepted anything and did
    // nothing, so dropping a capability appeared to work and changed nothing.
    capEff = ALL_CAPS;
    capPerm = ALL_CAPS;
    capInh = 0n;
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
    // how far settimeofday/clock_settime has moved the wall clock from the host's. the host clock is
    // not ours to change, so the guest's realtime is the host's plus this.
    clkRealOffsetNs = 0n;
    /** where the system-wide clock offset lives, once the worker has somewhere to put it. */
    clockOffsetHook = null;
    get clockOffsetNs() {
      return this.clockOffsetHook ? this.clockOffsetHook.get() : this.clkRealOffsetNs;
    }
    set clockOffsetNs(ns) {
      if (this.clockOffsetHook) this.clockOffsetHook.set(ns);
      else this.clkRealOffsetNs = ns;
      this.clkLastRealNs = 0n;
    }
    /** nanoseconds since this handler was constructed. the base every other clock is built on. */
    elapsedNs() {
      const elapsedMs = (this.clkHasPerf ? performance.now() : Date.now()) - this.clkPerfBase;
      let ns = BigInt(Math.round(elapsedMs * 1e6));
      if (ns < this.clkLastMonoNs) ns = this.clkLastMonoNs;
      else this.clkLastMonoNs = ns;
      return ns;
    }
    // how old the system already was when this process started. CLOCK_MONOTONIC measures the
    // machine's uptime, so a process that starts an hour in must not read it as zero.
    clkBootOffsetNs = 0n;
    setSystemBootMs(bootMs) {
      if (!Number.isFinite(bootMs) || bootMs <= 0) return;
      const age = Date.now() - bootMs;
      this.clkBootOffsetNs = age > 0 ? BigInt(Math.round(age * 1e6)) : 0n;
    }
    monoNs() {
      return this.clkBootOffsetNs + this.elapsedNs();
    }
    /**
     * the wall clock, at nanosecond resolution, without drifting away from the machine's.
     *
     * the fine-grained answer is an epoch captured at startup plus monotonic elapsed time, because
     * Date.now() only counts milliseconds. but the two run at slightly different rates, and the
     * kernel timestamps its own objects (an ipc segment's ctime) from Date.now() -- so after a while
     * the process and the kernel disagreed by a whole second about what time it was, and a test that
     * checks a timestamp against its own clock reads a creation time from before it asked.
     */
    realNs() {
      const wall = BigInt(Date.now()) * 1000000n;
      const est = this.clkRealBaseNs + this.elapsedNs();
      const drift = est - wall;
      if (drift < -2000000n) this.clkRealBaseNs -= drift;
      let v = this.clkRealBaseNs + this.elapsedNs() + this.clockOffsetNs;
      if (drift > 2000000n) v -= drift - 2000000n;
      if (v < this.clkLastRealNs) v = this.clkLastRealNs;
      else this.clkLastRealNs = v;
      return v;
    }
    clkLastRealNs = 0n;
    /**
     * the clock a timeout is measured against, in milliseconds.
     *
     * it has to be the SAME clock the guest reads through clock_gettime, or a wait can end before
     * the guest's own clock says the time has passed -- and every test that measures a sleep against
     * CLOCK_MONOTONIC calls that "woken up early". Date.now() has millisecond granularity, so a
     * deadline built from it was up to a millisecond short of what was asked for.
     */
    nowMs() {
      return Number(this.monoNs() / 1000n) / 1e3;
    }
    clockNs(clkId) {
      if (clkId === CLOCK_CPU_OWN || clkId === CLOCK_THREAD_CPUTIME_ID) {
        return BigInt(Math.round(this.ownCpuMs() * 1e6));
      }
      const isMono = clkId === 1 || clkId === 4 || clkId === 6 || clkId === 7 || clkId === 9;
      const ns = isMono ? this.monoNs() : this.realNs();
      const res = this.clockResNs(clkId);
      return res > 1n ? ns / res * res : ns;
    }
    /**
     * whether a clock id names a clock at all.
     *
     * a NEGATIVE id is linux's encoding for a dynamic clock backed by a file descriptor
     * (CLOCK_INVALID and the /dev/ptp family); we have none, so those are EINVAL too. answering
     * successfully for every id, as this used to, meant clock_gettime could not fail.
     */
    knownClock(clkId) {
      return clkId >= 0 && clkId <= 11 && clkId !== 10;
    }
    /**
     * a clock's resolution, in nanoseconds.
     *
     * the COARSE clocks are the ones read straight off the last timer tick without a hardware read,
     * so their resolution is the tick -- linux reports 1/CONFIG_HZ. everything else is nanosecond.
     */
    clockResNs(clkId) {
      return clkId === 5 || clkId === 6 ? 1000000n : 1n;
    }
    // every timer this process owns -- the three interval timers, the posix timers of
    // timer_create, and the descriptors of timerfd_create -- is a deadline on some clock plus a
    // period. nothing preempts a running guest, so expiry is EVALUATED rather than scheduled:
    // wherever the process could observe a timer (the exec loop's periodic signal poll, a blocking
    // wait, a read of a timerfd) every armed timer is compared against its clock and the whole run
    // of expirations since the last look is accounted for at once. the COUNT matters as much as the
    // fact, because that count is exactly what overrun and a timerfd read report.
    itimers = [
      { clockid: CLOCK_MONOTONIC, nextNs: 0n, intervalNs: 0n },
      // ITIMER_REAL    -> SIGALRM
      { clockid: CLOCK_CPU_OWN, nextNs: 0n, intervalNs: 0n },
      // ITIMER_VIRTUAL -> SIGVTALRM
      { clockid: CLOCK_CPU_OWN, nextNs: 0n, intervalNs: 0n }
      // ITIMER_PROF    -> SIGPROF
    ];
    posixTimers = /* @__PURE__ */ new Map();
    nextPosixTimerId = 0;
    timerFds = /* @__PURE__ */ new Map();
    nextTimerFdId = 0;
    // prctl's per-process settings. these are real state: a program sets one and reads it back, and
    // answering a constant made every get/set pair disagree.
    pdeathSig = 0;
    /** publish the parent-death signal into the shared process table. */
    pdeathHook = null;
    dumpable = 1;
    keepCaps = 0;
    noNewPrivs = 0;
    thpDisable = 0;
    childSubreaper = 0;
    secureBits = 0;
    timerSlackNs = 5e4;
    // PR_SET_TIMERSLACK(0) restores the value this process STARTED with, which is whatever its
    // parent was using when it forked -- not a constant. keeping only the current value made the
    // reset land on 50 us for a child whose parent had chosen something else.
    defaultTimerSlackNs = 5e4;
    comm = "jitter";
    /** publish a new process name so /proc and ps see it too. */
    commHook = null;
    /** exec renames the process after the binary it loaded, and resets the dump flag with it. */
    setComm(name) {
      this.comm = name.slice(0, 15);
      this.dumpable = 1;
      this.posixTimers.clear();
    }
    // RLIMIT_MEMLOCK's soft limit, in bytes. linux's default is 8 MiB on most distributions, and
    // ltp's mlock tests lower it and expect the lower value to bite.
    memlockLimit = 8 * 1024 * 1024;
    // the execution domain, as personality(2) sets and reads it. it survives exec and fork, so it
    // travels with the rest of the process state.
    personality = 0;
    // the supplementary groups this process belongs to. they survive fork and exec, and they decide
    // group access to every file the process opens.
    groups = [];
    // RLIMIT_NOFILE soft limit: the highest fd number a process may allocate. set via
    // prlimit64/setrlimit; enforced at fd-allocating syscalls (open/dup/...) -> EMFILE.
    nofileLimit = 1024;
    // every resource limit a program has explicitly set, by resource number. what is not here takes
    // the system default from rlimitOf.
    rlimits = [];
    /** cpu time this process has burned, in nanoseconds -- wall time minus every moment parked. */
    cpuNs() {
      return BigInt(Math.round(this.ownCpuMs() * 1e6));
    }
    /** the current value of the clock a timer is armed against. */
    timerClockNs(clockid) {
      return clockid === CLOCK_CPU_OWN ? this.cpuNs() : this.clockNs(clockid);
    }
    /**
     * how many times a deadline+period pair has expired by `now`, advancing it past all of them.
     *
     * a one-shot disarms. a periodic timer moves forward by whole periods, so a process that was
     * away for ten periods sees one expiry with an overrun of nine rather than ten separate ones --
     * which is what linux reports and what a timerfd read returns as its count.
     */
    static expire(t, now) {
      if (t.nextNs === 0n || now < t.nextNs) return 0;
      if (t.intervalNs <= 0n) {
        t.nextNs = 0n;
        return 1;
      }
      const n = 1n + (now - t.nextNs) / t.intervalNs;
      t.nextNs += n * t.intervalNs;
      return Number(n);
    }
    /** nothing armed anywhere: the common case, and worth not walking three collections for. */
    noTimers() {
      return this.posixTimers.size === 0 && this.timerFds.size === 0 && this.itimers[0].nextNs === 0n && this.itimers[1].nextNs === 0n && this.itimers[2].nextNs === 0n;
    }
    /**
     * settle every armed timer against its clock: queue the signals that are due, and credit a
     * timerfd with the expirations it can now report. returns true if a signal became pending.
     */
    checkTimers() {
      if (this.noTimers()) return false;
      let fired = false;
      for (let w = 0; w < 3; w++) {
        const t = this.itimers[w];
        if (t.nextNs === 0n) continue;
        if (_SyscallHandler.expire(t, this.timerClockNs(t.clockid)) > 0) {
          this.raiseSignal(ITIMER_SIGNAL[w]);
          fired = true;
        }
      }
      for (const t of this.posixTimers.values()) {
        if (t.nextNs === 0n) continue;
        const n = _SyscallHandler.expire(t, this.timerClockNs(t.clockid));
        if (n === 0) continue;
        t.accrued = Math.min(t.accrued + n, OVERRUN_CEIL);
        if (t.notify === SIGEV_NONE) continue;
        if (this.pendingSignals.includes(t.signo)) continue;
        t.overrun = clampOverrun(t.accrued - 1);
        t.accrued = 0;
        this.raiseSignal(t.signo, { pid: t.id, uid: t.overrun, code: -2, value: t.sigval });
        fired = true;
      }
      for (const t of this.timerFds.values()) {
        if (t.nextNs === 0n) continue;
        const n = _SyscallHandler.expire(t, this.timerClockNs(t.clockid));
        if (n > 0) t.ticks += BigInt(n);
      }
      return fired;
    }
    /**
     * ms until the next timer that will deliver a SIGNAL fires, or 0 if there is none -- the bound a
     * blocking wait must not sleep past.
     *
     * timers on a cpu clock are deliberately absent: cpu time does not advance while the process is
     * parked, so such a timer cannot fire during the wait however long it lasts.
     */
    timerRemainingMs() {
      if (this.noTimers()) return 0;
      let best = -1;
      const consider = (t) => {
        if (t.nextNs === 0n || t.clockid === CLOCK_CPU_OWN) return;
        const ms = Math.max(1, Number((t.nextNs - this.timerClockNs(t.clockid)) / 1000000n));
        if (best < 0 || ms < best) best = ms;
      };
      for (const t of this.itimers) consider(t);
      for (const t of this.posixTimers.values()) if (t.notify !== SIGEV_NONE) consider(t);
      return best < 0 ? 0 : best;
    }
    /**
     * read an itimerval (frac = 1e6, microseconds) or itimerspec (frac = 1e9, nanoseconds): two
     * {seconds, fraction} pairs, the period first and the initial expiry second. null when either
     * is not a valid time -- a negative second count, or a fraction that should have been carried.
     */
    readItimer(view, base, frac) {
      const perUnit = 1000000000n / frac;
      const out = [];
      for (const off of [0, 16]) {
        const sec = view.getBigInt64(base + off, true);
        const sub = view.getBigInt64(base + off + 8, true);
        if (sec < 0n || sub < 0n || sub >= frac) return null;
        out.push(sec * 1000000000n + sub * perUnit);
      }
      return { intervalNs: out[0], valueNs: out[1] };
    }
    writeItimer(view, base, intervalNs, valueNs, frac) {
      const perUnit = 1000000000n / frac;
      view.setBigInt64(base, intervalNs / 1000000000n, true);
      view.setBigInt64(base + 8, intervalNs % 1000000000n / perUnit, true);
      view.setBigInt64(base + 16, valueNs / 1000000000n, true);
      view.setBigInt64(base + 24, valueNs % 1000000000n / perUnit, true);
    }
    /** the timer behind a descriptor, or the errno that descriptor deserves. */
    timerFdOf(fd) {
      const e = this.getFdEntry(fd);
      if (!e) return this.fdIsOpen(fd) ? -EINVAL : -EBADF;
      if (e.type !== "timerfd") return -EINVAL;
      return this.timerFds.get(e.id) ?? -EINVAL;
    }
    /** the itimerval/itimerspec a getter should report for an armed timer: remaining, then period. */
    timerRemaining(t) {
      if (t.nextNs === 0n) return { valueNs: 0n, intervalNs: t.intervalNs };
      const rem = t.nextNs - this.timerClockNs(t.clockid);
      return { valueNs: rem > 0n ? rem : 1n, intervalNs: t.intervalNs };
    }
    pendingSignals = [];
    // queue of signals to deliver
    epollSets;
    signalHandlers;
    /** the thread group's shared dispositions, once this process has more than one thread. */
    sigTable;
    sigTableSeen = -1;
    signalMask;
    /** faulting address for the next fault-signal delivery (SIGSEGV/SIGBUS); set by the run loop's
     *  page-protection fault handler so the handler's siginfo.si_addr is accurate. */
    faultAddr = 0n;
    /** the stack pointer of the last rt_sigreturn we refused, for diagnosing where it came from. */
    badSigreturnSP = 0n;
    /** True if bridge calls can be handled (either direct bridge or Worker kernel relay) */
    bridgeAvailable = false;
    currentPid = 1;
    currentPpid = 0;
    // returns true if `pid` is a live process (a suspended ancestor in the serialized fork model).
    // wired by the dispatch/worker loop to the process manager so kill() can raise ESRCH correctly.
    isLivePidFn = null;
    // wired by the dispatch loop to the page-protection table: records the PROT_* permission of the
    // guest pages covering [addr, addr+len) so guest accesses to restricted pages fault (SIGSEGV).
    protHook = null;
    // publish this process's user ids where other processes can see them. kill(2) checks the
    // TARGET's ids, and those live in the target's own worker, so they have to be shared.
    credsHook = null;
    /**
     * the system's UTS names, read and written where every process can see them.
     *
     * a hostname belongs to the machine, not to whoever last asked: keeping a copy per process would
     * let two shells disagree about what host they are on. the worker points this at the shared
     * global page, so there is no round trip either.
     */
    utsHook = null;
    /**
     * the kernel's eventfd counters. an eventfd is one counter with descriptors onto it, and fork
     * gives parent and child the SAME counter -- so it cannot live in either of them.
     */
    eventfdHook = null;
    efdCount(id) {
      const r = this.eventfdHook?.({ op: "peek", id });
      if (!r || r[0] < 0) return 0n;
      return BigInt(r[2] >>> 0) << 32n | BigInt(r[1] >>> 0);
    }
    /** give up every eventfd this process holds. the process is ending or being replaced. */
    releaseEventfds() {
      for (const id of this.inheritedEventfdIds()) {
        this.eventfdHook?.({ op: "close", id, pid: this.currentPid });
      }
    }
    /** every eventfd this process holds, so fork can take a reference for the child. */
    inheritedEventfdIds() {
      const ids = /* @__PURE__ */ new Set();
      for (const [, e] of this.fdTable) if (e.type === "eventfd") ids.add(e.id);
      return [...ids];
    }
    /** file leases live in the kernel, where every process's opens are visible. */
    leaseHook = null;
    /** the queued-signal budget a sender must respect, in the shared table. */
    sigpendingHook = null;
    publishSigpending() {
      const cur = this.rlimitOf(RLIMIT_SIGPENDING).cur;
      this.sigpendingHook?.(cur >= 0x7fffffffn ? 2147483647 : Number(cur));
    }
    /** call after any change to the credentials, and once at startup. */
    publishCreds() {
      this.fsuid = this.euid;
      this.fsgid = this.egid;
      this.credsHook?.(this.ruid, this.euid, this.suid, this.rgid);
    }
    // spans of a file mapping that the file does not reach. touching one is SIGBUS rather than
    // SIGSEGV -- the mapping is legal, the memory behind it simply does not exist -- and keeping the
    // reason here rather than inferring it from the permission table means the fault handler asks a
    // question with a definite answer.
    noBacking = [];
    /** the recorded no-backing spans, for diagnostics. */
    noBackingRanges() {
      return this.noBacking.map((r) => `${r.start}-${r.end}`).join(",");
    }
    /** is this address inside a mapping the file does not reach? */
    isNoBacking(addr) {
      for (const r of this.noBacking) if (addr >= r.start && addr < r.end) return true;
      return false;
    }
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
    // the whole meta array is kept, not a chosen few fields: a cache that remembers less than the
    // bridge told us answers a later stat with zeros where the timestamps should be.
    statCache = /* @__PURE__ */ new Map();
    STAT_CACHE_TTL_MS = 1e3;
    /** entries live at most a second, but nothing was ever pruning the expired ones: a process that
     *  walked a large tree (one getdents fills an entry PER FILE) grew this map for as long as it
     *  ran. a cache that cannot forget is not a cache, it is a leak with a hit rate. */
    STAT_CACHE_MAX = 4096;
    cacheStat(key, value) {
      if (this.statCache.size >= this.STAT_CACHE_MAX) {
        const now = Date.now();
        for (const [k, v] of this.statCache) if (v.expiresAt <= now) this.statCache.delete(k);
        while (this.statCache.size >= this.STAT_CACHE_MAX) {
          const oldest = this.statCache.keys().next().value;
          if (oldest === void 0) break;
          this.statCache.delete(oldest);
        }
      }
      this.statCache.set(key, value);
    }
    /** forget what we knew about a path. a file has TWO cache entries -- the follow view and the
     *  nofollow one lstat reads -- and dropping only one of them leaves the other to answer with
     *  the state before the change. */
    /**
     * forget the cached stat of the file behind a descriptor. copy_file_range writes through the
     * bridge rather than through write(2), so nothing on the normal write path knows the file has
     * changed -- and the next fstat would report the size and mtime it had before the copy.
     */
    invalidateStatFd(fd) {
      const e = this.getFdEntry(fd);
      if (e?.type === "native") this.invalidateStat(e.path);
    }
    invalidateStat(path) {
      this.statCache.delete(path);
      this.statCache.delete("L:" + path);
    }
    lastWaitStatus;
    // exit code from last child (set by dispatch loop)
    lastWaitPid;
    // PID of last exited child
    constructor(brkStart, io, vfs2, bridge) {
      this.brk = brkStart;
      this.brkFloor = brkStart;
      this.io = io;
      this.vfs = vfs2;
      this.bridge = bridge;
      this.fdTable.set(0, { type: "stdio", fd: 0 });
      this.fdTable.set(1, { type: "stdio", fd: 1 });
      this.fdTable.set(2, { type: "stdio", fd: 2 });
      this.fdFlags.set(0, 0);
      this.fdFlags.set(1, 1);
      this.fdFlags.set(2, 1);
      if (this.vfs) {
        this.vfs.fdAllocator = () => this.allocFd();
        this.vfs.credsProvider = () => ({ ruid: this.ruid, euid: this.euid, rgid: this.rgid, egid: this.egid });
        this.vfs.selfCpuProvider = () => ({ utimeMs: this.ownUserMs(), stimeMs: this.ownSysMs() });
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
    /**
     * what an *at() call should answer about its DIRECTORY descriptor, before it looks at the name:
     * EBADF for one that is not open, ENOTDIR for one that is not a directory. a relative name is the
     * only case where the descriptor means anything -- an absolute one ignores it entirely.
     */
    atDirfdError(dirfd, path) {
      if (path.startsWith("/")) return 0;
      const d = Number(BigInt.asIntN(32, dirfd));
      if (d === -100) return 0;
      const e = this.getFdEntry(d);
      if (!e) return this.vfs?.hasFd(d) ? 0 : EBADF;
      if (e.type === "native") return e.isDir === false ? ENOTDIR : 0;
      return e.type === "vfs" ? 0 : ENOTDIR;
    }
    resolveAt(dirfd, path) {
      if (path.startsWith("/")) return path;
      const d = Number(BigInt.asIntN(32, dirfd));
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
    /**
     * re-read a shared file mapping from the file, for msync(MS_INVALIDATE).
     *
     * the point of MS_INVALIDATE is to throw away what the mapping has cached so the next read sees
     * the FILE. our mapping is a copy taken when it was made, so a write through the descriptor is
     * invisible to it until this happens -- which is exactly what a program does when it uses the
     * two interchangeably.
     */
    buildMmapReload(cpu2, memory2, reqStart, reqLen) {
      for (const m of this.fileMmaps) {
        if (reqStart >= m.start + m.len || reqStart + reqLen <= m.start) continue;
        const fStart = Math.max(reqStart, m.start);
        const fEnd = Math.min(reqStart + reqLen, m.start + m.fileLen);
        if (fEnd <= fStart) return null;
        const want = Math.min(fEnd - fStart, BRIDGE_READ_MAX);
        return {
          status: "bridge",
          method: "fs_pread",
          args: { fd: m.nativeFd, offset: m.fileOffset + (fStart - m.start), count: want },
          apply: (resp) => {
            if (resp.ret >= 0 && resp.bytes) {
              new Uint8Array(memory2.buffer).set(resp.bytes, fStart + GUEST_MEM_BASE);
            }
            cpu2.setReg(0, 0n);
          }
        };
      }
      return null;
    }
    // mark [start,end) (page-aligned) as locked or unlocked in the VmLck accounting, keeping the
    // interval set normalized (non-overlapping, merged), then republish the total to the vfs.
    /** whether any part of [start, end) is pinned in memory. */
    rangeLocked(start, end) {
      return this.lockedRegions.some((r) => r.start < end && r.end > start);
    }
    setLocked(start, end, locked, onFault = false) {
      if (end <= start) return;
      const next = [];
      for (const r of this.lockedRegions) {
        if (r.end <= start || r.start >= end) {
          next.push(r);
          continue;
        }
        if (r.start < start) next.push({ start: r.start, end: start });
        if (r.end > end) next.push({ start: end, end: r.end });
      }
      if (locked) next.push({ start, end, onFault });
      next.sort((a, b) => a.start - b.start);
      const merged = [];
      for (const r of next) {
        const last = merged[merged.length - 1];
        if (last && r.start <= last.end && !!last.onFault === !!r.onFault) {
          last.end = Math.max(last.end, r.end);
        } else {
          merged.push({ ...r });
        }
      }
      this.lockedRegions = merged;
      let total = 0;
      for (const r of merged) total += r.end - r.start;
      if (this.vfs) this.vfs.vmLckKb = Math.floor(total / 1024);
    }
    // does [start,end) overlap any tracked vma? used for MAP_FIXED_NOREPLACE.
    vmaOverlaps(start, end) {
      const v = this.vmas[this.vmaLower(start)];
      return !!v && v.start < end;
    }
    // register the lazy /proc/self/maps renderer once the vma set is live.
    publishVmas() {
      if (this.vfs && !this.vfs.procMapsGen) this.vfs.procMapsGen = () => this.renderProcMaps();
      if (this.vfs && !this.vfs.procSmapsGen) this.vfs.procSmapsGen = () => this.renderProcSmaps();
    }
    // vmas is kept sorted and non-overlapping, so all edits are a single O(n) pass with inline
    // coalescing of adjacent identical vmas: no re-sort, which keeps mapping-heavy loops linear.
    vmaEmit(out, v) {
      const last = out[out.length - 1];
      if (last && last.end === v.start && last.prot === v.prot && last.shared === v.shared && last.maxProt === v.maxProt && !!last.grow === !!v.grow) last.end = v.end;
      else out.push(v);
    }
    // carve [start,end) out of the vma set (splitting boundary vmas), then optionally insert a fresh
    // vma with the given prot/shared, all in sorted order.
    /** set x0 to -errno and finish the syscall. */
    retErr(cpu2, errno) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-errno)));
      return { status: "ok" };
    }
    // ── the thread group's shared address space ───────────────────────────────
    //
    // threads share one address space, so they must share one map of it. each thread worker used to
    // get a COPY of the vma list at clone and never resync, which meant an mmap in one thread was
    // invisible to its siblings: two threads could be handed the SAME address, /proc/self/maps was
    // wrong in every thread but the one that made the mapping, and a fork from a non-leader thread
    // lost whatever its siblings had mapped.
    //
    // the list therefore lives in the shared memory, and each thread keeps its js copy in step with
    // it through a generation counter: pull when the shared copy has moved on, push after an edit.
    // the js array stays the working representation, so the sorted-splice logic is single-sourced
    // and costs the same as before -- a splice was already O(n), and so is a serialize.
    vmaTable = null;
    vmaGen = 0;
    vmaDepth = 0;
    /** set by any edit, so a section that only read publishes nothing and siblings keep their view. */
    vmaDirty = false;
    /** point this handler at its thread group's shared vma table. */
    attachVmaTable(memory2, base) {
      this.vmaTable = new Int32Array(memory2.buffer, base, VMA_TABLE_BYTES >> 2);
    }
    /**
     * the thread group's shared signal dispositions.
     *
     * sigaction(2) is a PROCESS-wide setting: a handler installed by any thread is the handler every
     * thread of the group runs. keeping it per-thread meant a handler installed by a worker thread
     * was invisible to the main one, which then took the DEFAULT action for a signal the program had
     * plainly asked to catch.
     *
     * the layout is a header -- lock, generation, and which signals have been set at all -- then one
     * 8-word record per signal. readers keep a local copy and refresh it when the generation moves,
     * which is checked at every syscall boundary; that is the only place a signal can be delivered.
     */
    attachSigTable(memory2, base) {
      this.sigTable = new Int32Array(memory2.buffer, base, SIG_TABLE_BYTES >> 2);
      this.sigTableSeen = -1;
    }
    /** publish what this thread has as the group's initial state (called once, at promote). */
    publishSigTable() {
      const t = this.sigTable;
      if (!t) return;
      this.sigAcquire();
      try {
        Atomics.store(t, 2, 0);
        Atomics.store(t, 3, 0);
        for (const [sig, h] of this.signalHandlers ?? []) this.sigWriteLocked(sig, h);
        this.sigTableSeen = Atomics.add(t, 1, 1) + 1;
      } finally {
        this.sigRelease();
      }
    }
    /** refresh the local copy if another thread has changed a disposition since we last looked. */
    sigSync() {
      const t = this.sigTable;
      if (!t) return;
      const gen = Atomics.load(t, 1);
      if (gen === this.sigTableSeen) return;
      this.sigAcquire();
      try {
        const present = BigInt(Atomics.load(t, 3) >>> 0) << 32n | BigInt(Atomics.load(t, 2) >>> 0);
        const next = /* @__PURE__ */ new Map();
        for (let sig = 1; sig <= 64; sig++) {
          if ((present & 1n << BigInt(sig - 1)) === 0n) continue;
          const w = SIG_TABLE_FIRST + (sig - 1) * 8;
          const rd = (i) => BigInt(Atomics.load(t, w + i + 1) >>> 0) << 32n | BigInt(Atomics.load(t, w + i) >>> 0);
          next.set(sig, { handler: rd(0), flags: rd(2), restorer: rd(4), mask: rd(6) });
        }
        this.signalHandlers = next;
        this.sigTableSeen = Atomics.load(t, 1);
      } finally {
        this.sigRelease();
      }
    }
    /** record one disposition for the whole group. the caller holds the lock. */
    sigWriteLocked(sig, h) {
      const t = this.sigTable;
      const w = SIG_TABLE_FIRST + (sig - 1) * 8;
      const wr = (i, v) => {
        Atomics.store(t, w + i, Number(v & 0xffffffffn) | 0);
        Atomics.store(t, w + i + 1, Number(v >> 32n & 0xffffffffn) | 0);
      };
      wr(0, h.handler);
      wr(2, h.flags);
      wr(4, h.restorer);
      wr(6, h.mask);
      const bit2 = sig - 1;
      if (bit2 < 32) Atomics.or(t, 2, 1 << bit2);
      else Atomics.or(t, 3, 1 << bit2 - 32);
    }
    /** publish one changed disposition to the group. */
    sigPublish(sig, h) {
      const t = this.sigTable;
      if (!t) return;
      this.sigAcquire();
      try {
        if (h) this.sigWriteLocked(sig, h);
        else {
          const bit2 = sig - 1;
          if (bit2 < 32) Atomics.and(t, 2, ~(1 << bit2));
          else Atomics.and(t, 3, ~(1 << bit2 - 32));
        }
        this.sigTableSeen = Atomics.add(t, 1, 1) + 1;
      } finally {
        this.sigRelease();
      }
    }
    sigAcquire() {
      const t = this.sigTable;
      if (!t) return;
      for (let spin = 0; spin < 64; spin++) {
        if (Atomics.compareExchange(t, 0, 0, 1) === 0) return;
      }
      while (Atomics.compareExchange(t, 0, 0, 1) !== 0) Atomics.wait(t, 0, 1, 5);
    }
    sigRelease() {
      const t = this.sigTable;
      if (!t) return;
      Atomics.store(t, 0, 0);
      Atomics.notify(t, 0, 1);
    }
    /** publish the current list as the group's initial shared state (called once, at promote). */
    publishVmaTable() {
      if (!this.vmaTable) return;
      this.vmaAcquire();
      try {
        this.vmaPush();
      } finally {
        this.vmaRelease();
      }
    }
    vmaAcquire() {
      const t = this.vmaTable;
      if (!t) return;
      for (let spin = 0; spin < 64; spin++) {
        if (Atomics.compareExchange(t, 0, 0, 1) === 0) return;
      }
      while (Atomics.compareExchange(t, 0, 0, 1) !== 0) Atomics.wait(t, 0, 1, 5);
    }
    vmaRelease() {
      const t = this.vmaTable;
      if (!t) return;
      Atomics.store(t, 0, 0);
      Atomics.notify(t, 0, 1);
    }
    /** adopt the shared list if another thread has changed it since we last looked. */
    vmaPull() {
      const t = this.vmaTable;
      if (!t) return;
      const gen = Atomics.load(t, 1);
      if (gen === this.vmaGen) return;
      this.brk = BigInt(Atomics.load(t, 3) >>> 0);
      this.mmapBase = BigInt(Atomics.load(t, 4) >>> 0);
      const n = Atomics.load(t, 2);
      const out = new Array(n);
      for (let i = 0; i < n; i++) {
        const b = VMA_TABLE_FIRST + i * 3;
        const packed = t[b + 2];
        out[i] = {
          start: t[b] >>> 0,
          end: t[b + 1] >>> 0,
          prot: packed & 7,
          shared: (packed & 16) !== 0,
          maxProt: packed >>> 8 & 7,
          grow: (packed & 32) !== 0
        };
      }
      this.vmas = out;
      this.vmaGen = gen;
    }
    /** hand our list to the group and mark it as the newest. */
    vmaPush() {
      const t = this.vmaTable;
      if (!t) return;
      const n = Math.min(this.vmas.length, VMA_TABLE_ENTRIES);
      for (let i = 0; i < n; i++) {
        const v = this.vmas[i], b = VMA_TABLE_FIRST + i * 3;
        t[b] = v.start;
        t[b + 1] = v.end;
        t[b + 2] = v.prot & 7 | (v.shared ? 16 : 0) | (v.grow ? 32 : 0) | (v.maxProt & 7) << 8;
      }
      Atomics.store(t, 2, n);
      Atomics.store(t, 3, Number(this.brk) | 0);
      Atomics.store(t, 4, Number(this.mmapBase) | 0);
      this.vmaGen = Atomics.add(t, 1, 1) + 1;
    }
    /** read the group's address space without changing it. takes the lock: a plain read could
     *  otherwise land in the middle of a sibling's write and see half of two different lists. */
    readAddressSpace(fn) {
      if (!this.vmaTable || this.vmaDepth > 0) return fn();
      this.vmaAcquire();
      this.vmaDepth = 1;
      this.vmaDirty = false;
      try {
        this.vmaPull();
        const r = fn();
        if (this.vmaDirty) this.vmaPush();
        return r;
      } finally {
        this.vmaDepth = 0;
        this.vmaDirty = false;
        this.vmaRelease();
      }
    }
    /**
     * Run `fn` with the thread group's address space held still.
     *
     * choosing an address and claiming it have to be one indivisible step, or two threads mmapping
     * at the same moment both first-fit into the same hole and are handed the same memory. reentrant,
     * so the individual vma helpers can use it without caring whether a caller already holds it.
     */
    withAddressSpace(fn) {
      if (!this.vmaTable || this.vmaDepth > 0) {
        this.vmaDepth++;
        try {
          return fn();
        } finally {
          this.vmaDepth--;
        }
      }
      this.vmaAcquire();
      this.vmaDepth = 1;
      this.vmaDirty = false;
      try {
        this.vmaPull();
        const r = fn();
        if (this.vmaDirty) this.vmaPush();
        return r;
      } finally {
        this.vmaDepth = 0;
        this.vmaDirty = false;
        this.vmaRelease();
      }
    }
    /** index of the first vma that reaches past `addr`. the list is sorted, so this is a search
     *  rather than a walk -- which is what keeps a process with tens of thousands of mappings from
     *  paying a full pass per mmap. */
    vmaLower(addr) {
      let lo = 0, hi = this.vmas.length;
      while (lo < hi) {
        const m = lo + hi >> 1;
        if (this.vmas[m].end <= addr) lo = m + 1;
        else hi = m;
      }
      return lo;
    }
    /** the half-open slice of vmas that [start,end) touches. */
    vmaSlice(start, end) {
      const i = this.vmaLower(start);
      let j = i;
      while (j < this.vmas.length && this.vmas[j].start < end) j++;
      return { i, j };
    }
    /** what the slice [i,j) becomes when [start,end) is carved out and `insert` put in its place. */
    vmaPieces(start, end, i, j, insert) {
      const pieces = [];
      if (j > i) {
        const first = this.vmas[i];
        if (first.start < start) pieces.push({ start: first.start, end: start, prot: first.prot, shared: first.shared, maxProt: first.maxProt, grow: first.grow });
      }
      if (insert) pieces.push({ start, end, prot: insert.prot, shared: insert.shared, maxProt: insert.maxProt, grow: insert.grow });
      if (j > i) {
        const last = this.vmas[j - 1];
        if (last.end > end) pieces.push({ start: end, end: last.end, prot: last.prot, shared: last.shared, maxProt: last.maxProt, grow: last.grow });
      }
      return pieces;
    }
    /** replace vmas[i,i+del) with `pieces`, coalescing only at the two seams -- everything further
     *  out was already maximally merged, so there is nothing else that could have become mergeable. */
    vmaSplice(i, del, pieces) {
      this.vmaDirty = true;
      const merged = [];
      for (const p of pieces) this.vmaEmit(merged, p);
      if (del === 0 && merged.length === 1 && i === this.vmas.length) {
        this.vmas.push(merged[0]);
      } else if (del === 1 && merged.length === 1) {
        this.vmas[i] = merged[0];
      } else {
        this.vmas.splice(i, del, ...merged);
      }
      const lo = Math.max(0, i - 1);
      const hi = Math.min(this.vmas.length - 1, i + merged.length);
      for (let k = hi; k > lo; k--) {
        const a = this.vmas[k - 1], b = this.vmas[k];
        if (a.end === b.start && a.prot === b.prot && a.shared === b.shared && a.maxProt === b.maxProt && !!a.grow === !!b.grow) {
          a.end = b.end;
          this.vmas.splice(k, 1);
        }
      }
      this.publishVmas();
    }
    /**
     * would this edit push the process past vm.max_map_count?
     *
     * splitting a mapping costs a vma, which is why unmapping the MIDDLE of one can fail with ENOMEM
     * on a process that is already at the limit even though it is giving memory back.
     */
    vmaWouldExceed(start, end, insert) {
      if (end <= start) return false;
      const { i, j } = this.vmaSlice(start, end);
      const delta = this.vmaPieces(start, end, i, j, insert).length - (j - i);
      return delta > 0 && this.vmas.length + delta > this.maxMapCount;
    }
    vmaSet(start, end, insert) {
      if (end <= start) return;
      if (!insert && start < this.mmapHint) this.mmapHint = start;
      this.withAddressSpace(() => {
        const { i, j } = this.vmaSlice(start, end);
        this.vmaSplice(i, j - i, this.vmaPieces(start, end, i, j, insert));
      });
    }
    // set the protection of [start,end), splitting vmas at the boundaries (mprotect).
    vmaProtect(start, end, prot) {
      if (end <= start) return;
      this.withAddressSpace(() => this.vmaProtectLocked(start, end, prot));
    }
    vmaProtectLocked(start, end, prot) {
      const { i, j } = this.vmaSlice(start, end);
      const pieces = [];
      for (let k = i; k < j; k++) {
        const v = this.vmas[k];
        if (v.start < start) pieces.push({ start: v.start, end: start, prot: v.prot, shared: v.shared, maxProt: v.maxProt, grow: v.grow });
        pieces.push({ start: Math.max(v.start, start), end: Math.min(v.end, end), prot, shared: v.shared, maxProt: v.maxProt, grow: v.grow });
        if (v.end > end) pieces.push({ start: end, end: v.end, prot: v.prot, shared: v.shared, maxProt: v.maxProt, grow: v.grow });
      }
      this.vmaSplice(i, j - i, pieces);
    }
    /**
     * paint the initial page-protection table.
     *
     * it used to start as "everything accessible" and only mark what was explicitly unmapped, which
     * meant a wild pointer into address space that was never mapped at all read and wrote real bytes
     * instead of faulting. start from the opposite end: nothing is reachable unless something put it
     * there. the program image, its heap and its stack all live below the mmap arena in one run, and
     * everything above is described by a vma.
     */
    /** record the program image's loaded segments as mappings. the loader puts them there, so
     *  nothing else knows about them, and the vma set is what fork, /proc/self/maps and the page
     *  guard all read. */
    adoptImage(segs) {
      for (const g of segs) this.vmaSet(g.start, g.end, { prot: g.prot, shared: false, maxProt: 7 });
    }
    paintProtTable(stackTop, set) {
      this.readAddressSpace(() => this.paintProtTableLocked(stackTop, set));
    }
    paintProtTableLocked(stackTop, set) {
      set(0, Math.max(Number(this.brk), stackTop, this.mmapArenaStart), 7);
      for (const v of this.vmas) set(v.start, v.end, v.prot);
      for (const r of this.noBacking) set(r.start, r.end, PAGE_NO_BACKING);
    }
    /** the guard gap the kernel keeps between a growable stack and the next mapping below it. */
    static STACK_GUARD_GAP = 256 * 4096;
    /**
     * a fault just below a MAP_GROWSDOWN mapping is the stack asking for more room, not an error.
     * extend the mapping down to cover `addr` -- unless doing so would come within the guard gap of
     * whatever is mapped below, which is the whole point of the gap: a stack must never be able to
     * grow silently into its neighbour. returns the newly covered range, or null to let it fault.
     */
    growStackFor(addr) {
      return this.withAddressSpace(() => this.growStackForLocked(addr));
    }
    growStackForLocked(addr) {
      const page = addr & ~4095;
      const i = this.vmaLower(page);
      const v = this.vmas[i];
      if (!v || !v.grow || v.start <= page) return null;
      if (v.start - page > _SyscallHandler.STACK_GUARD_GAP) return null;
      const below = i > 0 ? this.vmas[i - 1].end : 0;
      if (page - below < _SyscallHandler.STACK_GUARD_GAP) return null;
      const grown = { start: page, end: v.start, prot: v.prot };
      this.vmaSet(page, v.start, { prot: v.prot, shared: v.shared, maxProt: v.maxProt, grow: true });
      return grown;
    }
    /**
     * whether [start, end) is covered by mappings with no hole. madvise, mincore, mlock and msync
     * all answer ENOMEM for a range that is not, and a hole in the MIDDLE counts -- which is why
     * this walks rather than just checking the ends.
     */
    rangeMapped(start, end) {
      return this.readAddressSpace(() => {
        let at = start;
        let i = this.vmaLower(at);
        while (at < end) {
          const v = this.vmas[i];
          if (!v || v.start > at) return false;
          at = v.end;
          i++;
        }
        return true;
      });
    }
    /**
     * what kind of memory [start, end) is, as far as madvise cares.
     *
     * madvise's advice values are not universally applicable: MADV_FREE and MADV_WIPEONFORK are
     * defined only on private anonymous memory, MADV_REMOVE only on shared memory that has no file
     * behind it, and MADV_DONTNEED refuses a locked range. accepting them everywhere made madvise a
     * call that could not fail.
     */
    rangeKind(start, end) {
      const shared = this.readAddressSpace(() => {
        let at = start, i = this.vmaLower(at);
        while (at < end) {
          const v = this.vmas[i];
          if (!v || v.start > at) break;
          if (v.shared) return true;
          at = v.end;
          i++;
        }
        return false;
      });
      const fileBacked = this.fileMmaps.some((m) => m.start < end && m.start + m.len > start);
      const locked = this.lockedRegions.some((r) => r.start < end && r.end > start);
      return { shared, fileBacked, locked };
    }
    /** the vma covering `addr`, or null. */
    vmaAt(addr) {
      const i = this.vmaLower(addr);
      const v = this.vmas[i];
      return v && v.start <= addr ? v : null;
    }
    /**
     * cpu time this process has used, in milliseconds. supplied by the worker, which is the only
     * layer that can see the difference between running and being parked in a blocking wait. zero
     * when there is no worker (the host-side unit tests), which is honest: nothing ran.
     */
    cpuMsHook = null;
    ownCpuMs() {
      return this.cpuMsHook ? this.cpuMsHook() : 0;
    }
    /** the part of that cpu time spent inside the kernel, and the rest of it. */
    sysMsHook = null;
    ownSysMs() {
      return Math.min(this.sysMsHook ? this.sysMsHook() : 0, this.ownCpuMs());
    }
    ownUserMs() {
      return Math.max(0, this.ownCpuMs() - this.ownSysMs());
    }
    /** resident size in KiB. the worker owns the stack top, so it is the one that can ask. */
    rssKbHook = null;
    // what reaped children have cost, which is what getrusage(RUSAGE_CHILDREN) and times()' cutime
    // report. linux accumulates these across every child a process ever waits for, so they only grow.
    childUtimeMs = 0;
    childStimeMs = 0;
    childMaxRssKb = 0;
    noteChildCpu(utimeMs, stimeMs, maxRssKb = 0) {
      this.childUtimeMs += utimeMs;
      this.childStimeMs += stimeMs;
      if (maxRssKb > this.childMaxRssKb) this.childMaxRssKb = maxRssKb;
    }
    /**
     * fill a struct rusage (144 bytes on aarch64) at a guest address.
     *
     * the fields we do not measure are written as zero, which is the point: wait4(2) used to leave
     * the caller's buffer untouched entirely, so a program read back whatever its own stack happened
     * to hold. `time` reported 2329 hours of system time for a 50ms command that way.
     */
    writeRusage(view, addr, utimeMs, stimeMs, maxRssKb) {
      const base = addr + GUEST_MEM_BASE;
      for (let i = 0; i < 144; i += 8) view.setBigUint64(base + i, 0n, true);
      view.setBigUint64(base + 0, BigInt(Math.floor(utimeMs / 1e3)), true);
      view.setBigUint64(base + 8, BigInt(Math.round(utimeMs % 1e3 * 1e3)), true);
      view.setBigUint64(base + 16, BigInt(Math.floor(stimeMs / 1e3)), true);
      view.setBigUint64(base + 24, BigInt(Math.round(stimeMs % 1e3 * 1e3)), true);
      view.setBigUint64(base + 32, BigInt(Math.max(0, Math.floor(maxRssKb))), true);
    }
    /**
     * one line describing what the address space says about `addr`: the vma covering it, or the
     * gap it falls in with its neighbours named. an undecodable instruction that reads as
     * 0x00000000 is almost never a translator gap -- it is memory that is not there -- and the two
     * are indistinguishable from the opcode alone, so the fault report has to say which.
     */
    describeAddr(addr) {
      return this.readAddressSpace(() => {
        const v = this.vmaAt(addr);
        if (v) return `in vma [0x${v.start.toString(16)},0x${v.end.toString(16)}) prot=${v.prot}${v.shared ? " shared" : ""}`;
        const i = this.vmaLower(addr);
        const before = this.vmas[i];
        const after = this.vmas[i] && this.vmas[i].start > addr ? this.vmas[i] : this.vmas[i + 1];
        const b = before && before.end <= addr ? `0x${before.end.toString(16)}` : "none";
        const a = after ? `0x${after.start.toString(16)}` : "none";
        return `UNMAPPED (brk=0x${this.brk.toString(16)}, prev vma ends ${b}, next begins ${a})`;
      });
    }
    renderProcMaps() {
      return this.readAddressSpace(() => this.renderProcMapsLocked());
    }
    renderProcMapsLocked() {
      let out = "";
      for (const v of this.vmas) {
        const perms = (v.prot & 1 ? "r" : "-") + (v.prot & 2 ? "w" : "-") + (v.prot & 4 ? "x" : "-") + (v.shared ? "s" : "p");
        const s = v.start.toString(16);
        const e = v.end.toString(16);
        out += `${s}-${e} ${perms} 00000000 00:00 0 
`;
      }
      return out;
    }
    /**
     * /proc/self/smaps: every mapping, with what it costs.
     *
     * the numbers are this system's truth rather than a copy of linux's shape. a mapping here is
     * backed by wasm memory that is committed when the mapping is made, so every page of it IS
     * resident -- there is no second-level paging to make Rss smaller than Size. what mlock changes
     * is whether the range may be handed back, and that is recorded exactly.
     */
    renderProcSmaps() {
      return this.readAddressSpace(() => {
        let out = "";
        for (const v of this.vmas) {
          const perms = (v.prot & 1 ? "r" : "-") + (v.prot & 2 ? "w" : "-") + (v.prot & 4 ? "x" : "-") + (v.shared ? "s" : "p");
          const s = v.start.toString(16);
          const e = v.end.toString(16);
          const kb = Math.ceil((v.end - v.start) / 1024);
          const lockedKb = Math.ceil(this.lockedBytesIn(v.start, v.end) / 1024);
          const anon = v.shared ? 0 : kb;
          out += `${s}-${e} ${perms} 00000000 00:00 0 
`;
          out += `Size:           ${String(kb).padStart(8)} kB
`;
          out += `KernelPageSize:        4 kB
`;
          out += `MMUPageSize:           4 kB
`;
          out += `Rss:            ${String(kb).padStart(8)} kB
`;
          out += `Pss:            ${String(kb).padStart(8)} kB
`;
          out += `Shared_Clean:          0 kB
`;
          out += `Shared_Dirty:   ${String(v.shared ? kb : 0).padStart(8)} kB
`;
          out += `Private_Clean:         0 kB
`;
          out += `Private_Dirty:  ${String(anon).padStart(8)} kB
`;
          out += `Referenced:     ${String(kb).padStart(8)} kB
`;
          out += `Anonymous:      ${String(anon).padStart(8)} kB
`;
          out += `LazyFree:              0 kB
`;
          out += `AnonHugePages:         0 kB
`;
          out += `ShmemPmdMapped:        0 kB
`;
          out += `FilePmdMapped:         0 kB
`;
          out += `Shared_Hugetlb:        0 kB
`;
          out += `Private_Hugetlb:       0 kB
`;
          out += `Swap:                  0 kB
`;
          out += `SwapPss:               0 kB
`;
          out += `Locked:         ${String(lockedKb).padStart(8)} kB
`;
          out += `THPeligible:    0
`;
          out += `VmFlags:${v.prot & 1 ? " rd" : ""}${v.prot & 2 ? " wr" : ""}${v.prot & 4 ? " ex" : ""}${v.shared ? " sh" : " mr mw me"}${lockedKb ? " lo" : ""} ac
`;
        }
        return out;
      });
    }
    /** where file handles are remembered: the kernel holds the map, so a handle taken in one process
     *  can be opened in another. */
    handleHook = null;
    /** a fresh open-file-description id, for the paths outside open() that make one. */
    newOfdIdPublic() {
      return this.newOfdId();
    }
    /** the path a descriptor was opened by, for the *at() calls that accept an empty name. */
    pathOfFd(fd) {
      const e = this.getFdEntry(fd);
      if (e && e.path) return e.path;
      return this.vfs?.fdPath?.(fd) ?? null;
    }
    /** what the address space says about an address -- which mapping it is in, or that it is in none.
     *  the fault trace prints it, because "SIGSEGV at 0x..." alone cannot tell a guard page from a
     *  mapping this kernel forgot it made. */
    addrInfo(addr) {
      return this.readAddressSpace(() => {
        const v = this.vmas.find((x) => addr >= x.start && addr < x.end);
        if (v) return `vma[0x${v.start.toString(16)},0x${v.end.toString(16)}) prot=${v.prot}${v.shared ? " shared" : ""}`;
        const pn = this.protNone.find((r) => addr >= r.start && addr < r.end);
        if (pn) return `hole[0x${pn.start.toString(16)},0x${pn.end.toString(16)})`;
        let below, above;
        for (const x of this.vmas) {
          if (x.end <= addr && (!below || x.end > below.end)) below = x;
          if (x.start > addr && (!above || x.start < above.start)) above = x;
        }
        const near = (v2) => v2 ? `0x${v2.start.toString(16)}-0x${v2.end.toString(16)}` : "none";
        return `no mapping (below=${near(below)} above=${near(above)} mmapBase=0x${this.mmapBase.toString(16)})`;
      });
    }
    /**
     * the file-size ceiling to hand the filesystem with a write, or -1 for none.
     *
     * RLIMIT_FSIZE is enforced where the offset is known, which is on the other side of the bridge:
     * a write that starts past the limit is EFBIG (and SIGXFSZ), one that crosses it is short.
     */
    fsizeLimitArg() {
      const lim = this.rlimitOf(RLIMIT_FSIZE);
      if (lim.cur === RLIM_INFINITY || lim.cur > BigInt(Number.MAX_SAFE_INTEGER)) return -1;
      return Number(lim.cur);
    }
    /**
     * is [start,end) still a SHARED mapping, as the thread group's live vma table sees it?
     *
     * a worker's own list of shared mappings is a snapshot: a sibling thread that unmapped one leaves
     * every other thread believing it is still there, and if the address is later handed out again the
     * two threads disagree about what the memory IS. that disagreement is invisible until a futex on
     * it keys differently in each thread and the wake reaches nobody.
     */
    rangeIsShared(start, end) {
      return this.rangeKind(start, end).shared;
    }
    /** how many bytes of [start,end) are mlocked. */
    lockedBytesIn(start, end) {
      let n = 0;
      for (const r of this.lockedRegions) {
        const a = Math.max(r.start, start), b = Math.min(r.end, end);
        if (b > a) n += b - a;
      }
      return n;
    }
    /** how many bytes of [start,end) are locked AND populated -- an MLOCK_ONFAULT range is neither
     *  resident nor absent by the lock alone, so it does not count here. */
    populatedBytesIn(start, end) {
      let n = 0;
      for (const r of this.lockedRegions) {
        if (r.onFault) continue;
        const a = Math.max(r.start, start), b = Math.min(r.end, end);
        if (b > a) n += b - a;
      }
      return n;
    }
    /**
     * the process's root directory, as a host path. chroot(2) moves it; everything below resolves
     * against it, and nothing can climb out with `..` (linux clamps the same way -- a chrooted
     * process cannot escape by walking up).
     *
     * one deliberate divergence: /proc and /dev stay synthesized after a chroot rather than
     * disappearing. on linux a chroot without those mounted has neither, but ours are the only
     * source of process information in the system and losing them costs more than the fidelity is
     * worth. every other path goes through the new root.
     */
    rootDir = "/";
    /**
     * which of the two filesystems a path lives on. /proc/mounts advertises exactly two -- the root
     * and /proc -- and that is not a simplification: /proc really is synthesised per process while
     * everything else is one real filesystem underneath.
     */
    mountOf(path) {
      return path === "/proc" || path.startsWith("/proc/") ? "/proc" : "/";
    }
    /**
     * the /proc links that name an OBJECT rather than a path -- exe, cwd, root and the fd/ entries.
     * following one lands somewhere no amount of reading the path could have predicted, which is
     * why openat2 lets a caller refuse to be led by one.
     */
    isMagicLink(path) {
      const m = /^\/proc\/(self|thread-self|[0-9]+)\/(.*)$/.exec(path);
      if (!m) return false;
      return m[2] === "exe" || m[2] === "cwd" || m[2] === "root" || /^fd\/[0-9]+$/.test(m[2]);
    }
    /** the working directory as the guest names it -- the vfs owns it, so it is the one to ask. */
    currentCwd() {
      return this.hostToGuest(this.vfs ? this.vfs.getcwd() : this.cwd);
    }
    /**
     * where a path lands once openat2's scoping flags have had their say, or the errno they refuse
     * it with.
     *
     * RESOLVE_BENEATH and RESOLVE_IN_ROOT both say the walk may not leave the directory the
     * descriptor names. they differ in what an absolute path or a climbing ".." means: BENEATH
     * refuses one outright, IN_ROOT reinterprets it against that directory, which is what a chroot
     * would do. NO_XDEV refuses a walk that ends on a different filesystem from the one it started
     * on, and NO_MAGICLINKS refuses to be taken through a /proc link that names an object.
     * RESOLVE_NO_SYMLINKS is the one this cannot answer, because only the filesystem can see the
     * links -- it travels down to the open itself.
     */
    scopeOpenPath(dirfd, raw, resolve) {
      const checked = resolve & (RESOLVE_BENEATH | RESOLVE_IN_ROOT | RESOLVE_NO_XDEV | RESOLVE_NO_MAGICLINKS);
      if (!checked) return this.resolveAt(dirfd, raw);
      const d = Number(BigInt.asIntN(32, dirfd));
      const base = this.normalizePath(d === -100 ? this.currentCwd() : this.getFdEntry(d)?.path ?? "");
      if (!base.startsWith("/")) return EBADF;
      let full;
      if (resolve & RESOLVE_BENEATH) {
        if (raw.startsWith("/")) return EXDEV;
        full = this.normalizePath(base === "/" ? "/" + raw : base + "/" + raw);
        const prefix = base === "/" ? "/" : base + "/";
        if (full !== base && !full.startsWith(prefix)) return EXDEV;
      } else if (resolve & RESOLVE_IN_ROOT) {
        const rel = this.normalizePath(raw.startsWith("/") ? raw : "/" + raw);
        full = base === "/" ? rel : rel === "/" ? base : base + rel;
      } else {
        const at = this.resolveAt(dirfd, raw);
        full = at.startsWith("/") ? this.normalizePath(at) : this.normalizePath(base === "/" ? "/" + at : base + "/" + at);
      }
      if (resolve & RESOLVE_NO_XDEV && this.mountOf(full) !== this.mountOf(base)) return EXDEV;
      if (resolve & RESOLVE_NO_MAGICLINKS) {
        const parts = full.split("/").filter((c) => c.length > 0);
        let walked = "";
        for (const c of parts) {
          walked += "/" + c;
          if (this.isMagicLink(walked)) return ELOOP;
        }
      }
      return full;
    }
    /** collapse `.` and `..` and duplicate separators. the result always starts with `/`. */
    normalizePath(p) {
      const parts = [];
      for (const seg of p.split("/")) {
        if (seg === "" || seg === ".") continue;
        if (seg === "..") {
          parts.pop();
          continue;
        }
        parts.push(seg);
      }
      return "/" + parts.join("/");
    }
    /** a guest-visible absolute path, as the host sees it. */
    guestToHost(guestAbs) {
      if (this.rootDir === "/") return guestAbs;
      return guestAbs === "/" ? this.rootDir : this.rootDir + guestAbs;
    }
    /** a host path, as the guest sees it. paths outside the root report as the root itself. */
    hostToGuest(hostAbs) {
      if (this.rootDir === "/") return hostAbs;
      if (hostAbs === this.rootDir) return "/";
      if (hostAbs.startsWith(this.rootDir + "/")) return hostAbs.slice(this.rootDir.length);
      return "/";
    }
    /**
     * the on-disk path a unix-socket name binds to, or null if it does not live on the real
     * filesystem. bind(AF_UNIX) creates a file there, so it needs the same resolution every other
     * path-taking call gets -- relative to the cwd, pushed under the root.
     */
    socketBindPath(path) {
      return this.bridgePath(this.resolveAt(-100n, path));
    }
    /**
     * an empty pathname names nothing.
     *
     * every path syscall without AT_EMPTY_PATH answers ENOENT for "" -- resolving it lands on the
     * cwd instead, which then reports whatever is true of the DIRECTORY (EEXIST, EISDIR) rather
     * than the fact that the caller named no file at all.
     */
    isEmptyPath(raw) {
      return raw.length === 0;
    }
    /**
     * what is wrong with a pathname before anyone tries to resolve it.
     *
     * an empty name is ENOENT, a name longer than PATH_MAX is ENAMETOOLONG, and so is a single
     * COMPONENT longer than NAME_MAX -- the last of which used to be reported as whatever the
     * walk happened to hit first (usually ENOTDIR).
     */
    pathCheck(raw) {
      if (raw.length === 0) return ENOENT;
      if (raw.length > 4095) return ENAMETOOLONG;
      for (const part of raw.split("/")) if (part.length > 255) return ENAMETOOLONG;
      return 0;
    }
    /**
     * the host path a name resolves to, or null when it names something only this process has (the
     * synthesised /proc and /dev). exec asks for this so it can check the file's permissions before
     * it replaces the address space with it.
     */
    absolutePath(path) {
      return this.bridgePath(path);
    }
    bridgePath(path) {
      if (!this.bridge?.isNative && !this.bridgeAvailable) return null;
      let host;
      if (path.startsWith("/")) {
        host = this.guestToHost(this.normalizePath(path));
      } else {
        const cwd = this.vfs?.getcwd() ?? this.cwd;
        host = this.normalizePath(cwd === "/" ? "/" + path : cwd + "/" + path);
      }
      if (this.rootDir !== "/" && host !== this.rootDir && !host.startsWith(this.rootDir + "/")) {
        host = this.rootDir;
      }
      const guest = this.hostToGuest(host);
      if (guest === "/dev/shm" || guest.startsWith("/dev/shm/")) return host;
      if (guest === "/proc" || guest.startsWith("/proc/") || guest === "/dev" || guest.startsWith("/dev/")) {
        return null;
      }
      return host;
    }
    // resolve a (possibly relative) path to a normalized absolute path against the current cwd.
    // used for execve so the worker reads the binary from the right place (a bare "./prog" must
    // become "/cwd/prog"), matching how the kernel resolves the exec pathname.
    absolutizePath(path) {
      if (path.startsWith("/")) return this.guestToHost(this.normalizePath(path));
      const cwd = this.vfs?.getcwd() ?? this.cwd;
      return this.normalizePath(cwd === "/" ? "/" + path : cwd + "/" + path);
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
      const handler = this.signalHandlers?.get(signum);
      const handlerAddr = signum === 9 || signum === 19 ? SIG_DFL : handler?.handler ?? SIG_DFL;
      if (signum !== 9 && signum !== 19 && this.signalMask && this.signalMask & 1n << BigInt(signum - 1)) {
        return { delivered: false };
      }
      if (handlerAddr === SIG_IGN) return { delivered: true, action: "continue" };
      if (handlerAddr === SIG_DFL) {
        if (SIG_IGNORE_DEFAULT.has(signum)) return { delivered: true, action: "continue" };
        if (SIG_STOP_DEFAULT.has(signum)) return { delivered: true, action: "stop", exitCode: signum };
        if (signum === 18) return { delivered: true, action: "continue" };
        return { delivered: true, action: "exit", exitCode: 65536 | (SIG_CORE_DEFAULT.has(signum) ? 131072 : 0) | signum };
      }
      const view = new DataView(memory2.buffer);
      const currentSP = cpu2.getSP();
      const frameSP = currentSP - BigInt(SIGNAL_FRAME_SIZE) & ~15n;
      const frameBase = Number(frameSP) + GUEST_MEM_BASE;
      const withAddr = SIG_WITH_ADDR.has(signum);
      new Uint8Array(memory2.buffer).fill(0, frameBase, frameBase + UC_MCONTEXT);
      view.setInt32(frameBase, signum, true);
      const sender = this.sigSender.get(signum);
      this.sigSender.delete(signum);
      view.setInt32(
        frameBase + SI_CODE,
        withAddr ? 2 : sender ? sender.code : 128,
        true
      );
      if (withAddr) view.setBigUint64(frameBase + SI_ADDR, this.faultAddr, true);
      else if (sender) {
        view.setInt32(frameBase + SI_PID, sender.pid, true);
        view.setInt32(frameBase + SI_UID, sender.uid, true);
        view.setBigUint64(frameBase + SI_VALUE, sender.value, true);
      }
      if (withAddr) view.setBigUint64(frameBase + UC_MCONTEXT, this.faultAddr, true);
      for (let r = 0; r < 31; r++) view.setBigUint64(frameBase + MC_REGS + r * 8, cpu2.getReg(r), true);
      view.setBigUint64(frameBase + MC_SP, currentSP, true);
      view.setBigUint64(frameBase + MC_PC, cpu2.getPC(), true);
      view.setBigUint64(frameBase + MC_PSTATE, cpu2.getNZCV(), true);
      view.setBigUint64(frameBase + UC_SIGMASK, this.suspendRestoreMask ?? this.signalMask ?? 0n, true);
      this.suspendRestoreMask = null;
      {
        const vBase = cpu2.base + V_OFFSET;
        for (let i = 0; i < 512; i += 8) view.setBigUint64(frameBase + MC_VREGS + i, view.getBigUint64(vBase + i, true), true);
      }
      const flags = handler?.flags ?? 0n;
      let newMask = (this.signalMask ?? 0n) | (handler?.mask ?? 0n);
      if ((flags & 0x40000000n) === 0n) newMask |= 1n << BigInt(signum - 1);
      this.signalMask = newMask;
      cpu2.setSP(frameSP);
      cpu2.setPC(handlerAddr);
      cpu2.setReg(0, BigInt(signum));
      cpu2.setReg(1, frameSP);
      cpu2.setReg(2, frameSP + BigInt(UC_OFF));
      cpu2.setReg(30, BigInt(SIGNAL_TRAMPOLINE));
      this.faultAddr = 0n;
      return { delivered: true, action: "continue" };
    }
    /** Handle rt_sigreturn: restore registers from signal frame. */
    handleSigreturn(cpu2, memory2) {
      const view = new DataView(memory2.buffer);
      const frameSP = cpu2.getSP();
      const frameBase = Number(frameSP) + GUEST_MEM_BASE;
      if (frameSP < 0n || frameBase < GUEST_MEM_BASE || frameBase + SIGNAL_FRAME_SIZE > memory2.buffer.byteLength) {
        this.badSigreturnSP = frameSP;
        this.raiseSignal(11);
        return cpu2.getPC();
      }
      for (let r = 0; r < 31; r++) cpu2.setReg(r, view.getBigUint64(frameBase + MC_REGS + r * 8, true));
      const savedSP = view.getBigUint64(frameBase + MC_SP, true);
      const savedPC = view.getBigUint64(frameBase + MC_PC, true);
      const savedNZCV = view.getBigUint64(frameBase + MC_PSTATE, true);
      this.signalMask = view.getBigUint64(frameBase + UC_SIGMASK, true);
      {
        const vBase = cpu2.base + V_OFFSET;
        for (let i = 0; i < 512; i += 8) view.setBigUint64(vBase + i, view.getBigUint64(frameBase + MC_VREGS + i, true), true);
      }
      cpu2.setSP(savedSP);
      cpu2.setNZCV(savedNZCV);
      return savedPC;
    }
    // who raised each pending signal, when it was a kill(2) rather than the kernel. a handler
    // installed with SA_SIGINFO reads si_pid to find out, and 0 means "the kernel" -- which for a
    // signal a process explicitly sent is simply false, and a program that replies to si_pid ends up
    // signalling process group 0 instead of the sender.
    sigSender = /* @__PURE__ */ new Map();
    /**
     * Take a pending signal that is in `mask` (bit sig-1), WITHOUT running its handler.
     *
     * This is what sigwait/sigtimedwait do: the signal is accepted by the caller directly rather than
     * being delivered, which is why programs block the signals they intend to wait for. Returns the
     * sender too, so the caller can fill in siginfo.
     */
    takePendingIn(mask) {
      for (let i = 0; i < this.pendingSignals.length; i++) {
        const sig = this.pendingSignals[i];
        if (sig < 1 || sig > 64) continue;
        if (!(mask & 1n << BigInt(sig - 1))) continue;
        this.pendingSignals.splice(i, 1);
        const from = this.sigSender.get(sig);
        this.sigSender.delete(sig);
        return { sig, pid: from?.pid ?? 0, uid: from?.uid ?? 0 };
      }
      return null;
    }
    /** Queue a signal for delivery. `from` names the sender for a kill(2)-generated signal. */
    /**
     * would delivering this signal DO anything? a signal set to ignore -- explicitly with SIG_IGN, or
     * by its own default like SIGCHLD and SIGWINCH -- is consumed without a trace.
     */
    signalIsNoop(signum) {
      const h = this.signalHandlers?.get(signum);
      const addr = signum === 9 || signum === 19 ? SIG_DFL : h?.handler ?? SIG_DFL;
      if (addr === SIG_IGN) return true;
      return addr === SIG_DFL && SIG_IGNORE_DEFAULT.has(signum);
    }
    /** is this process being traced? a tracer sees signals its tracee would otherwise discard. */
    tracedHook = null;
    raiseSignal(signum, from) {
      const blocked = this.signalMask ? (this.signalMask & 1n << BigInt(signum - 1)) !== 0n : false;
      if (!blocked && !this.tracedHook?.() && this.signalIsNoop(signum)) return;
      if (signum < 32 && this.pendingSignals.includes(signum)) return;
      this.pendingSignals.push(signum);
      if (from) this.sigSender.set(signum, from);
      else this.sigSender.delete(signum);
    }
    /**
     * raise a signal caused by the instruction that is about to re-run: SIGILL, SIGSEGV, SIGBUS,
     * SIGFPE from a fault.
     *
     * such a signal cannot be blocked or ignored into oblivion. if it were, the faulting instruction
     * would simply run again, fault again, and the process would spin at 100% forever instead of
     * dying -- which is exactly what it did. linux forces the disposition back to the default and
     * unblocks it (force_sig_info), turning an unfixable loop into the death it already is.
     */
    raiseFaultSignal(signum) {
      const h = this.signalHandlers?.get(signum);
      if (h && h.handler === SIG_IGN) this.signalHandlers.delete(signum);
      if (this.signalMask) this.signalMask &= ~(1n << BigInt(signum - 1));
      this.raiseSignal(signum);
    }
    /** this process's blocked-signal mask, for diagnostics. */
    currentMask() {
      return this.signalMask ?? 0n;
    }
    /** the caller's effective uid, for stamping si_uid on a signal it sends. */
    currentEuid() {
      return this.euid;
    }
    /** signal mask accessors — used by the worker's blocking rt_sigsuspend, which temporarily swaps in
     *  the caller's mask while parked and restores the original on return. */
    getSignalMask() {
      return this.signalMask ?? 0n;
    }
    setSignalMask(m) {
      this.signalMask = m;
    }
    /**
     * the mask sigsuspend must put back, once the handler it waited for has run.
     *
     * sigsuspend replaces the mask, waits, and lets the signal be TAKEN under the replacement --
     * that is the whole point, and unblocking a signal only to re-block it before the handler could
     * run is the same as not unblocking it at all. so the frame the handler returns through carries
     * the ORIGINAL mask, and rt_sigreturn is what restores it.
     */
    suspendRestoreMask = null;
    setSuspendMask(temp, saved) {
      this.signalMask = temp;
      this.suspendRestoreMask = saved;
    }
    /** sigsuspend ended without a handler running, so the mask goes back here instead. */
    clearSuspendMask() {
      if (this.suspendRestoreMask !== null) {
        this.signalMask = this.suspendRestoreMask;
        this.suspendRestoreMask = null;
      }
    }
    /**
     * Is there a signal we could actually take RIGHT NOW?
     *
     * Not the same question as "is anything queued". A blocked signal stays queued until it is
     * unblocked -- that is what blocking means -- and treating it as an interrupt would abandon every
     * blocking call with EINTR forever while never running a handler, because the thing that would
     * clear the queue is the delivery that the mask is preventing. libc blocks signals constantly
     * (musl's raise() blocks ALL of them), so one signal arriving at the wrong moment would leave the
     * process spinning out of every wait it ever entered.
     */
    hasPendingDeliverable() {
      return this.hasDeliverableSignal(this.signalMask ?? 0n);
    }
    /** true if a queued signal is deliverable under `mask` (unblocked) — the wake condition for suspend. */
    hasDeliverableSignal(mask) {
      return this.pendingSignals.some((s) => (mask & 1n << BigInt(s - 1)) === 0n);
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
    /**
     * resolve a descriptor that must name a socket.
     *
     * EBADF means "there is no such descriptor"; ENOTSOCK means "there is, and it is not a socket".
     * they are not interchangeable: a program handed the wrong fd needs to know which mistake it
     * made, and ltp checks the distinction on bind, listen, setsockopt, getsockopt and shutdown.
     */
    sockFd(fd) {
      if (this.fdIsPath(fd)) return { err: EBADF };
      const entry = this.getFdEntry(fd);
      if (entry) return entry.type === "socket" || entry.type === "unixsock" ? { e: entry } : { err: ENOTSOCK };
      return { err: this.fdIsOpen(fd) ? ENOTSOCK : EBADF };
    }
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
        const groups = [];
        for (let i = 0; i < 16; i += 2) groups.push(addr[i] << 8 | addr[i + 1]);
        let bestStart = -1, bestLen = 0;
        for (let i = 0; i < 8; i++) {
          if (groups[i] !== 0) continue;
          let j = i;
          while (j < 8 && groups[j] === 0) j++;
          if (j - i > bestLen) {
            bestLen = j - i;
            bestStart = i;
          }
          i = j - 1;
        }
        let host;
        if (bestLen > 1) {
          const head = groups.slice(0, bestStart).map((g) => g.toString(16)).join(":");
          const tail = groups.slice(bestStart + bestLen).map((g) => g.toString(16)).join(":");
          host = `${head}::${tail}`;
        } else {
          host = groups.map((g) => g.toString(16)).join(":");
        }
        return { family, host, port };
      }
      return { family, host: "0.0.0.0", port: 0 };
    }
    /** Write a sockaddr structure to guest memory. Returns bytes written. */
    /** the size a struct sockaddr of this family occupies, which is what getsockname reports. */
    sockaddrLen(family) {
      return family === 10 ? 28 : family === 1 ? 2 : 16;
    }
    /**
     * copy an address out to a caller's (addr, addrlen) pair, the way the kernel does: at most
     * *addrlen bytes are written, and *addrlen is set to the address's FULL size so the caller can
     * tell it was truncated. accept, recvfrom and recvmsg all answer with one of these.
     */
    putSockaddr(bytes, view, addrPtr, lenPtr, addr) {
      if (lenPtr === 0 || this.ptrWriteFaults(bytes, lenPtr, 4)) return -EFAULT;
      const want = view.getInt32(lenPtr + GUEST_MEM_BASE, true);
      if (want < 0) return -EINVAL;
      const full = this.sockaddrLen(addr.family);
      const copy = Math.min(want, full);
      if (addrPtr === 0 || copy > 0 && this.ptrWriteFaults(bytes, addrPtr, copy)) return -EFAULT;
      const buf = new Uint8Array(full);
      const bv = new DataView(buf.buffer);
      bv.setUint16(0, addr.family, true);
      if (addr.family === 2) {
        bv.setUint16(2, addr.port, false);
        const oct = addr.host.split(".").map((n) => Number(n) & 255);
        for (let i = 0; i < 4; i++) buf[4 + i] = oct[i] ?? 0;
      } else if (addr.family === 10) {
        bv.setUint16(2, addr.port, false);
        let g;
        if (addr.host.includes(":")) {
          g = ipv6Groups(addr.host);
        } else {
          const o = addr.host.split(".").map((n) => Number(n) & 255);
          g = [0, 0, 0, 0, 0, 65535, (o[0] ?? 0) << 8 | (o[1] ?? 0), (o[2] ?? 0) << 8 | (o[3] ?? 0)];
        }
        for (let i = 0; i < 8; i++) bv.setUint16(8 + i * 2, g[i], false);
      }
      bytes.set(buf.subarray(0, copy), addrPtr + GUEST_MEM_BASE);
      view.setUint32(lenPtr + GUEST_MEM_BASE, full, true);
      return 0;
    }
    /** the interface list a net_ifaddrs reply carries. */
    ifaceList(resp) {
      if (!resp?.bytes) return [];
      try {
        return JSON.parse(new TextDecoder().decode(resp.bytes));
      } catch {
        return [];
      }
    }
    /** an interface name in its fixed 16-byte field, nul-padded. */
    putIfName(bytes, addr, name) {
      const enc = new TextEncoder().encode(name).subarray(0, 15);
      bytes.fill(0, addr + GUEST_MEM_BASE, addr + GUEST_MEM_BASE + 16);
      bytes.set(enc, addr + GUEST_MEM_BASE);
    }
    readIfName(bytes, addr) {
      let end = 0;
      while (end < 16 && bytes[addr + GUEST_MEM_BASE + end] !== 0) end++;
      return new TextDecoder().decode(bytes.subarray(addr + GUEST_MEM_BASE, addr + GUEST_MEM_BASE + end));
    }
    /** a sockaddr written straight into a struct that has room for one, with no length to report. */
    putSockaddrRaw(view, addr, a) {
      const base = addr + GUEST_MEM_BASE;
      view.setUint16(base, a.family, true);
      view.setUint16(base + 2, a.port, false);
      if (a.family === 2) {
        const oct = a.host.split(".").map((n) => Number(n) & 255);
        for (let i = 0; i < 4; i++) view.setUint8(base + 4 + i, oct[i] ?? 0);
      } else if (a.family === 10) {
        const g = ipv6Groups(a.host);
        for (let i = 0; i < 8; i++) view.setUint16(base + 8 + i * 2, g[i], false);
      }
    }
    /** the peer a bridge reply describes: the host side packs it as family, port and eight groups. */
    peerFromMeta(resp) {
      const m = resp?._meta;
      if (!m || m.length < 10) return null;
      const family = m[1] | 0;
      if (family !== 2 && family !== 10) return null;
      const g = [];
      for (let i = 0; i < 8; i++) g.push(m[2 + i] | 0);
      const host = family === 2 ? `${g[0] >> 8 & 255}.${g[0] & 255}.${g[1] >> 8 & 255}.${g[1] & 255}` : ipv6Text(g);
      return { family, host, port: m[0] | 0 };
    }
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
    /**
     * Reserve address space for a SysV attach, and give it back on detach.
     *
     * shmat is mmap's anonymous-shared case wearing a different name: the segment's bytes live in the
     * same arena, and what it needs from the address space is identical. Sharing the mechanism keeps
     * one story about where mappings come from and what the vma list says about them -- which matters,
     * because fork's memory snapshot now reads that list.
     */
    attachShared(memory2, len, prot, key, want = 0) {
      const aligned = len + 4095 & ~4095;
      if (want !== 0) {
        if (want <= 0 || (want & 4095) !== 0) return -EINVAL;
        if (want + aligned + GUEST_MEM_BASE > this.maxBytes) return -EINVAL;
        if (this.vmaOverlaps(want, want + aligned)) return -EINVAL;
        if (!this.growTo(memory2, want + aligned + GUEST_MEM_BASE)) return -ENOMEM;
        new Uint8Array(memory2.buffer).fill(0, want + GUEST_MEM_BASE, want + GUEST_MEM_BASE + aligned);
        this.vmaSet(want, want + aligned, { prot: prot & 7, shared: true, maxProt: 7 });
        try {
          this.protHook?.(want, aligned, prot & 7);
        } catch {
        }
        this.sharedRegions.push({ addr: want, len: aligned });
        this.sharedMapHook?.(want, aligned, key);
        return want;
      }
      const addr = Number(this.mmapBase);
      const needed = addr + aligned + GUEST_MEM_BASE;
      if (needed > this.maxBytes) return -ENOMEM;
      this.mmapBase += BigInt(aligned);
      if (!this.growTo(memory2, needed)) return -ENOMEM;
      new Uint8Array(memory2.buffer).fill(0, addr + GUEST_MEM_BASE, addr + GUEST_MEM_BASE + aligned);
      this.vmaSet(addr, addr + aligned, { prot: prot & 7, shared: true, maxProt: 7 });
      try {
        this.protHook?.(addr, aligned, prot & 7);
      } catch {
      }
      this.sharedRegions.push({ addr, len: aligned });
      this.sharedMapHook?.(addr, aligned, key);
      return addr;
    }
    /** the segment attached at this address, if any -- shmdt names an address, not an id. */
    sharedRegionAt(addr) {
      return this.sharedRegions.find((r) => r.addr === addr);
    }
    detachShared(addr, len) {
      this.vmaSet(addr, addr + len, null);
      this.sharedRegions = this.sharedRegions.filter((r) => r.addr !== addr);
      try {
        this.protHook?.(addr, len, 0);
      } catch {
      }
    }
    /** Register an fd entry directly (used by Worker for kernel pipe fds). */
    /**
     * Register a descriptor created outside this handler (kernel pipes, unix sockets).
     *
     * `status` is the O_* STATUS flags of the new open file description, and passing it is not
     * optional bookkeeping: descriptor numbers are reused lowest-free-first, so a number that is
     * silently left with its previous holder's flags starts life non-blocking (or appending) for no
     * reason anyone can see. a fresh description starts with exactly the flags it was opened with.
     */
    registerFd(fd, entry, status = 0) {
      this.fdTable.set(fd, entry);
      this.fdFlags.set(fd, status);
    }
    /** the O_* status flags of a descriptor, for dup: a duplicate SHARES the description, flags and all. */
    statusFlags(fd) {
      return this.fdFlags.get(fd) ?? 0;
    }
    /**
     * the process's resource limits, in one place.
     *
     * getrlimit, setrlimit and prlimit64 are three spellings of the same state, and each used to
     * carry its own copy of the table. they drifted the moment one of them learned about a limit the
     * others did not: prlimit64 reported RLIMIT_MEMLOCK as 8 MiB while getrlimit still said
     * infinity, and ltp's getrlimit03 exists precisely to catch that disagreement.
     */
    rlimitOf(resource) {
      const stored = this.rlimits[resource];
      if (stored) return stored;
      switch (resource) {
        case RLIMIT_STACK:
          return { cur: 8388608n, max: RLIM_INFINITY };
        case RLIMIT_NOFILE:
          return { cur: BigInt(this.nofileLimit), max: 1048576n };
        case RLIMIT_MEMLOCK:
          return { cur: BigInt(this.memlockLimit), max: RLIM_INFINITY };
        // an ordinary user gets no nice budget and no real-time budget at all: linux starts both at
        // zero, and that zero is exactly what makes an unprivileged sched_setscheduler EPERM.
        case RLIMIT_SIGPENDING:
          return { cur: BigInt(SIGPENDING_DEFAULT), max: BigInt(SIGPENDING_DEFAULT) };
        case RLIMIT_NICE:
          return { cur: 0n, max: 0n };
        case RLIMIT_RTPRIO:
          return { cur: 0n, max: 0n };
        default:
          return { cur: RLIM_INFINITY, max: RLIM_INFINITY };
      }
    }
    /**
     * record a new limit, and apply it where this system enforces one.
     *
     * every resource is REMEMBERED even when nothing here acts on it: a limit a program sets and
     * reads back has to be the limit it set. storing only the two we enforce meant setrlimit
     * appeared to succeed and getrlimit then reported the default.
     */
    setRlimit(resource, cur, max) {
      this.rlimits[resource] = { cur, max };
      if (resource === RLIMIT_NOFILE) this.nofileLimit = cur >= 1048576n ? 1048576 : Number(cur);
      else if (resource === RLIMIT_MEMLOCK) this.memlockLimit = cur >= 0x7fffffffn ? 2147483647 : Number(cur);
      else if (resource === RLIMIT_SIGPENDING) this.publishSigpending();
    }
    /**
     * report a filesystem change to whoever is watching it.
     *
     * supplied by the worker, which owns the channel to the kernel's watch registry. it costs one
     * atomic load when nobody is watching, which is the normal case.
     */
    inotifyHook = null;
    /** pairs the two halves of a rename in the event stream. */
    renameCookie = 0;
    // whether ctrl-alt-del reboots immediately, per reboot(LINUX_REBOOT_CMD_CAD_ON/OFF). there is no
    // ctrl-alt-del here, but the setting is real state a program can toggle.
    cadEnabled = false;
    /** stop the guest system. supplied by the worker; the argument is true for a restart. */
    haltHook = null;
    /**
     * report a filesystem change to whoever is watching.
     *
     * `isDir` decides IN_ISDIR, which watchers use to tell a directory apart from a file of the same
     * name. leave it undefined and the reporter stats the object to find out; pass it explicitly when
     * the object is already gone (a delete, the departing half of a rename) or when the answer is
     * free. `opts.probe` names a different path to stat for that answer, for the same reason.
     *
     * `opts.selfOnly` keeps the event off the parent directory's watch. an unlink drops the link
     * count, which is an attribute change of the file itself and no business of the directory.
     */
    notify(path, mask, cookie = 0, isDir, opts) {
      if (path) {
        this.inotifyHook?.(
          path,
          mask,
          cookie,
          isDir,
          opts?.selfOnly ?? false,
          opts?.probe ?? path,
          opts?.unlinked ?? false
        );
      }
    }
    /** report an event about whatever file a descriptor refers to. */
    notifyFd(fd, mask) {
      if (!this.inotifyHook) return;
      const e = this.fdTable.get(fd);
      if (e && e.type === "native" && e.path) {
        this.inotifyHook(e.path, mask, 0, e.isDir === true, false, e.path, e.unlinked === true);
      }
    }
    /**
     * a guest path as the host filesystem sees it, for a caller outside this handler.
     *
     * inotify watches name a path rather than a descriptor, and the watch registry lives in the
     * kernel -- which has no idea what this process's cwd or root directory are. null for /proc
     * and /dev, which are synthesised per-process and cannot be watched from outside.
     */
    resolveForWatch(path) {
      return this.bridgePath(path);
    }
    /** the kernel-side inotify instance a descriptor names, or -1. */
    inotifyIdOf(fd) {
      const e = this.fdTable.get(fd);
      return e && e.type === "inotify" ? e.id : -1;
    }
    /** the kernel-side message queue a descriptor names, or -1. */
    mqIdOf(fd) {
      const e = this.fdTable.get(fd);
      return e && e.type === "mqueue" ? e.id : -1;
    }
    /** does any OTHER descriptor still name this queue? a close is only the last one's business. */
    mqHeldElsewhere(fd, id) {
      for (const [other, e] of this.fdTable) {
        if (other !== fd && e.type === "mqueue" && e.id === id) return true;
      }
      return false;
    }
    /** every inotify instance this process holds, so a fork can take a reference on each. */
    inotifyInstanceIds() {
      const out = /* @__PURE__ */ new Set();
      for (const e of this.fdTable.values()) if (e.type === "inotify") out.add(e.id);
      return [...out];
    }
    /** mark a descriptor close-on-exec, for the creators that live outside this handler. */
    markCloexec(fd) {
      this.fdCloexec.add(fd);
    }
    clearCloexec(fd) {
      this.fdCloexec.delete(fd);
    }
    /** undo a registerFd -- a rollback, or a close handled outside the close(2) case. per-fd metadata
     *  never outlives the fd, or the next holder of the number inherits it. */
    unregisterFd(fd) {
      this.fdTable.delete(fd);
      this.fdFlags.delete(fd);
      this.fdCloexec.delete(fd);
    }
    // the resolved path backing a native fd, used as the key for cross-process record locks. a
    // non-native fd (pipe, vfs) returns null: those have no on-disk identity to lock against.
    fdPath(fd) {
      const e = this.fdTable.get(fd);
      return e && e.type === "native" ? e.path : null;
    }
    // the open file description behind an fd, which is what owns an F_OFD_* lock and an flock. dup
    // and fork copy it (they share the description, exactly as linux does); a fresh open gets a new
    // one. ids are unique across processes because the pid is folded into them, so the kernel can
    // compare descriptions from different workers without coordinating a counter.
    fdDescriptionId(fd) {
      const e = this.fdTable.get(fd);
      return e && e.type === "native" ? e.ofdId : 0;
    }
    // how many of this process's descriptors still point at one open file description.
    countFdsForDescription(ofdId) {
      let n = 0;
      for (const e of this.fdTable.values()) if (e.type === "native" && e.ofdId === ofdId) n++;
      return n;
    }
    ofdSeq = 0;
    newOfdId() {
      return this.currentPid * 1048576 + ++this.ofdSeq;
    }
    /** the swift-side descriptor behind a native fd, or -1. */
    fdNative(fd) {
      const e = this.fdTable.get(fd);
      return e && e.type === "native" ? e.nativeFd : -1;
    }
    /** Set process identity (used when a fork continues as the child). */
    setIdentity(pid, ppid) {
      this.currentPid = pid;
      this.currentPpid = ppid;
    }
    // the current process credentials, threaded into NativeFS path-op bridge calls so the native
    // side can enforce emulated POSIX ownership/permissions against this process's identity.
    // (supplementary groups aren't tracked yet — setgroups is a no-op — so groups is empty for now.)
    // may this process traverse a directory with these owner/mode bits? root always may, provided
    // some execute bit is set anywhere -- matching the kernel, and InMemoryVFS.permitted.
    mayExec(mode, uid, gid) {
      if (this.euid === 0) return (mode & 73) !== 0;
      if (this.euid === uid) return (mode & 64) !== 0;
      if (this.egid === gid) return (mode & 8) !== 0;
      return (mode & 1) !== 0;
    }
    getCreds() {
      return {
        ruid: this.ruid,
        euid: this.fsuid,
        rgid: this.rgid,
        egid: this.fsgid,
        umask: this.umask,
        groups: [...this.groups]
      };
    }
    /** Get state for process snapshot */
    getState() {
      return {
        brk: this.brk,
        mmapBase: this.mmapBase,
        cwd: this.vfs ? this.vfs.getcwd() : this.cwd,
        // a child inherits its parent's root directory, and exec keeps it: that is what makes a
        // chroot stick to everything a process goes on to run.
        rootDir: this.rootDir,
        // resource limits survive both fork and exec, which is how `ulimit -n 64; ./prog` works.
        rlimits: this.rlimits.map((r) => r ? { ...r } : void 0),
        fdFlags: new Map(this.fdFlags),
        // WHICH descriptors are close-on-exec is per-process state that fork copies, exactly like the
        // fd table itself. leaving it out meant a forked child began with an empty set, so the exec
        // that followed closed nothing and every FD_CLOEXEC descriptor survived into the new program.
        // that is the one thing FD_CLOEXEC exists to prevent: it is how a process keeps its private
        // descriptors out of a child it execs, and a leaked one holds files open, keeps pipes from
        // ever reaching EOF, and hands the new program access it was never meant to have.
        fdCloexec: new Set(this.fdCloexec),
        fdTable: new Map(this.fdTable),
        nativeRefs: new Map(this.nativeRefs),
        currentPid: this.currentPid,
        currentPpid: this.currentPpid,
        creds: {
          ruid: this.ruid,
          euid: this.euid,
          suid: this.suid,
          rgid: this.rgid,
          egid: this.egid,
          sgid: this.sgid,
          fsuid: this.fsuid,
          fsgid: this.fsgid
        },
        caps: { eff: this.capEff, perm: this.capPerm, inh: this.capInh },
        sharedRegions: this.sharedRegions.map((r) => ({ ...r })),
        // a fork inherits the parent's ADDRESS SPACE, so it inherits the mappings that describe it.
        // leaving them behind gave every child an empty vma list: /proc/self/maps showed nothing,
        // munmap and mprotect had no record to edit, and mmap could hand back an address that was
        // already in use. it also made a fork's own memory snapshot incomplete once that snapshot
        // started asking which regions are mapped rather than sweeping the whole address space.
        vmas: this.vmas.map((v) => ({ ...v })),
        // an epoll descriptor survives fork, and so does what it is watching: linux copies the
        // interest list into the child. without it every epoll_wait in a child was EINVAL, since the
        // descriptor was there and the set behind it was not.
        epollSets: this.epollSets ? [...this.epollSets].map(([fd, set]) => [fd, [...set].map(([t, e]) => [t, { ...e }])]) : void 0,
        protNone: this.protNone.map((r) => ({ ...r })),
        // a forked child inherits its parent's mappings, so it inherits which parts of them the
        // file does not reach. without this the child faulted on the same page and was told it was
        // a bad pointer rather than a short file.
        noBacking: this.noBacking.map((r) => ({ ...r })),
        nofileLimit: this.nofileLimit,
        // the execution domain and what the process has pinned in memory both survive a fork.
        personality: this.personality,
        groups: [...this.groups],
        // a child inherits the timer slack its parent was using, and that value becomes the child's
        // own default -- what PR_SET_TIMERSLACK(0) puts back.
        timerSlackNs: this.timerSlackNs,
        lockedRegions: this.lockedRegions.map((r) => ({ ...r })),
        lockFuture: this.lockFuture,
        signalHandlers: this.signalHandlers ? new Map(this.signalHandlers) : void 0,
        vfsFdSnapshot: this.vfs ? this.vfs.snapshotFds() : void 0
      };
    }
    /** Restore state from process snapshot */
    restoreState(state) {
      this.brk = state.brk;
      this.mmapBase = state.mmapBase;
      if (state.rootDir) this.rootDir = state.rootDir;
      if (state.rlimits) this.rlimits = state.rlimits.map((r) => r ? { ...r } : void 0);
      this.cwd = state.cwd;
      if (state.fdCloexec) this.fdCloexec = new Set(state.fdCloexec);
      if (state.caps) {
        this.capEff = state.caps.eff;
        this.capPerm = state.caps.perm;
        this.capInh = state.caps.inh;
      }
      if (state.noBacking) this.noBacking = state.noBacking.map((r) => ({ ...r }));
      if (state.personality !== void 0) this.personality = state.personality;
      if (state.groups) this.groups = [...state.groups];
      if (state.timerSlackNs !== void 0) {
        this.timerSlackNs = state.timerSlackNs;
        this.defaultTimerSlackNs = state.timerSlackNs;
      }
      if (state.lockedRegions) {
        this.lockedRegions = state.lockedRegions.map((r) => ({ ...r }));
        let lkTotal = 0;
        for (const r of this.lockedRegions) lkTotal += r.end - r.start;
        if (this.vfs) this.vfs.vmLckKb = Math.floor(lkTotal / 1024);
      }
      if (state.lockFuture !== void 0) this.lockFuture = state.lockFuture;
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
        this.fsuid = state.creds.fsuid ?? state.creds.euid;
        this.fsgid = state.creds.fsgid ?? state.creds.egid;
      }
      if (state.sharedRegions !== void 0) this.sharedRegions = state.sharedRegions.map((r) => ({ ...r }));
      if (state.vmas !== void 0) this.vmas = state.vmas.map((v) => ({ ...v }));
      if (state.epollSets !== void 0) {
        this.epollSets = new Map(state.epollSets.map(([fd, set]) => [fd, new Map(set.map(([t, e]) => [t, { ...e }]))]));
      }
      if (state.protNone !== void 0) this.protNone = state.protNone.map((r) => ({ ...r }));
      if (state.nofileLimit !== void 0) this.nofileLimit = state.nofileLimit;
      if (state.signalHandlers !== void 0) this.signalHandlers = new Map(state.signalHandlers);
      this.publishVmas();
    }
    /**
     * Close every native (Swift-backed) fd this process still holds. real kernels close a
     * process's fds when it exits; jitter never did, so a Worker that runs many pooled tenants
     * leaks Swift file handles until the device fd budget is exhausted and NativeFS starts failing
     * opens. call this on process exit / before pool reuse. stdio and non-native fds are ignored.
     */
    /** close one descriptor, for the callers that cannot return a bridge result -- close_range
     *  closes a whole span in a single syscall, and there is only ever one reply to give. */
    closeOne(fd) {
      const e = this.fdTable.get(fd);
      this.fdFlags.delete(fd);
      this.fdCloexec.delete(fd);
      if (!e) {
        if (this.vfs?.hasFd(fd)) this.vfs.close(fd);
        return;
      }
      this.fdTable.delete(fd);
      switch (e.type) {
        case "pipe_write":
          e.pipe.writeClosed = true;
          break;
        case "pipe_read":
          e.pipe.readClosed = true;
          break;
        case "vfs":
          this.vfs?.close(e.fd);
          break;
        case "epoll":
          this.epollSets?.delete(fd);
          break;
        case "native": {
          const refs = (this.nativeRefs.get(e.nativeFd) ?? 1) - 1;
          if (refs <= 0) {
            this.nativeRefs.delete(e.nativeFd);
            this.closeNativeHook?.(e.nativeFd);
          } else this.nativeRefs.set(e.nativeFd, refs);
          break;
        }
        case "socket":
          this.closeSocketHook?.(e.socketId);
          break;
        case "unixsock":
          this.closeUnixHook?.(fd);
          break;
        default:
          break;
      }
    }
    closeNativeFds(closeFn) {
      const seen = /* @__PURE__ */ new Set();
      for (const [guestFd, entry] of [...this.fdTable]) {
        if (entry.type !== "native") continue;
        if (!seen.has(entry.nativeFd)) {
          seen.add(entry.nativeFd);
          closeFn(entry.nativeFd);
        }
        this.fdTable.delete(guestFd);
      }
    }
    /** descriptors marked FD_CLOEXEC, for owners of fd kinds this handler does not itself track. */
    cloexecFds() {
      return [...this.fdCloexec];
    }
    resetForExec(brkStart, closeFn) {
      this.brk = brkStart;
      this.brkFloor = brkStart;
      this.mmapBase = 0x2100000n;
      this.mmapHint = 34603008;
      this.protNone = [];
      this.vmas = [];
      this.vmaTable = null;
      this.vmaGen = 0;
      this.vmaDepth = 0;
      this.sharedRegions = [];
      this.fileMmaps = [];
      this.noBacking = [];
      this.lockedRegions = [];
      if (this.signalHandlers) {
        for (const [sig, h] of [...this.signalHandlers]) {
          if (h.handler !== SIG_IGN) this.signalHandlers.delete(sig);
        }
      }
      this.sigTable = void 0;
      this.sigTableSeen = -1;
      this.pendingSignals = [];
      for (const fd of [...this.fdCloexec]) {
        const entry = this.fdTable.get(fd);
        if (entry) {
          if (entry.type === "native" && closeFn) closeFn(entry.nativeFd);
          if (entry.type === "eventfd") this.eventfdHook?.({ op: "close", id: entry.id, pid: this.currentPid });
          else if (entry.type === "vfs" && this.vfs) this.vfs.close(entry.fd);
          this.fdTable.delete(fd);
        } else if (this.vfs?.hasFd(fd)) {
          this.vfs.close(fd);
        }
        this.fdFlags.delete(fd);
      }
      this.fdCloexec.clear();
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
    /** the lowest free descriptor at or above `min`, or -1 when RLIMIT_NOFILE leaves none. */
    allocLowestFd(min = 0) {
      const fd = this.lowestFreeFd(min);
      return fd < this.nofileLimit ? fd : -1;
    }
    /** can `n` more descriptors be allocated within RLIMIT_NOFILE? pipe() needs two at once. */
    fdRoomFor(n) {
      let free = 0;
      for (let fd = 0; fd < this.nofileLimit && free < n; fd++) {
        if (!this.fdTable.has(fd) && !(this.vfs?.hasFd(fd) ?? false)) free++;
      }
      return free >= n;
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
    /** a descriptor's open status flags, for the worker-side handlers that need them. */
    fdStatusFlags(fd) {
      return this.fdFlags.get(fd) ?? 0;
    }
    /** public view of a fd's O_NONBLOCK flag (the worker needs it for kernel-pipe write blocking). */
    isNonBlockingFd(fd) {
      return this.isNonBlocking(fd);
    }
    /** turn O_NONBLOCK on or off for a descriptor, for the callers that live outside this handler. */
    setFdNonBlocking(fd, on) {
      const cur = this.fdFlags.get(fd) ?? 0;
      this.fdFlags.set(fd, on ? cur | O_NONBLOCK : cur & ~O_NONBLOCK);
    }
    /**
     * would touching this guest range fault? bounds AND the process's PROT_NONE mappings, which is
     * how a caller distinguishes "you handed me an unmapped address" (EFAULT) from every other
     * complaint. worker-side syscall handlers need it too: bounds alone miss a PROT_NONE page,
     * which is exactly what a test reaches for when it wants a guaranteed-bad pointer.
     */
    ptrWouldFault(bytes, addr, len = 1) {
      return this.ptrFaults(bytes, addr, len);
    }
    /**
     * O_PATH opens a descriptor that NAMES a file without granting access to its contents: it is
     * for *at() calls and fstat, and reading or writing through it is EBADF. it survives dup, so
     * the duplicate is just as inert.
     */
    fdIsPath(fd) {
      return ((this.fdFlags.get(fd) ?? 0) & 2097152) !== 0;
    }
    /**
     * the process a descriptor names, for the calls that take a pidfd: one made by pidfd_open, or an
     * open /proc/<pid> directory, which linux accepts wherever a pidfd is wanted. -1 for anything
     * else, which is EBADF to the caller.
     */
    pidfdTarget(fd) {
      const e = this.getFdEntry(fd);
      if (e?.type === "pidfd") return e.pid;
      if (e !== void 0 && e.type !== "vfs") return -1;
      const path = this.vfs?.fdPath?.(fd) ?? null;
      const m = typeof path === "string" ? path.match(/^\/proc\/(self|\d+)\/?$/) : null;
      if (!m) return -1;
      return m[1] === "self" ? this.tgid || this.currentPid : Number(m[1]);
    }
    /** is a signal this descriptor reports pending right now? */
    signalfdReady(fd) {
      const e = this.getFdEntry(fd);
      if (e?.type !== "signalfd") return false;
      return this.pendingSignals.some((sig) => (e.mask & 1n << BigInt(sig - 1)) !== 0n);
    }
    /**
     * read(2) on a signalfd: hand back one struct signalfd_siginfo per signal in the descriptor's
     * mask that is pending, CONSUMING it -- a signal read this way is not delivered again.
     *
     * returns the byte count, -EINVAL for a buffer too small for one record, or -EAGAIN when
     * nothing matches (the caller decides whether to wait).
     */
    signalfdRead(fd, buf, count, view, bytes) {
      const e = this.getFdEntry(fd);
      if (e?.type !== "signalfd") return -EBADF;
      const SIGINFO_SIZE = 128;
      if (count < SIGINFO_SIZE) return -EINVAL;
      if (this.ptrWriteFaults(bytes, buf, Math.min(count, SIGINFO_SIZE))) return -EFAULT;
      let written = 0;
      while (written + SIGINFO_SIZE <= count) {
        const idx = this.pendingSignals.findIndex((sig2) => (e.mask & 1n << BigInt(sig2 - 1)) !== 0n);
        if (idx < 0) break;
        const sig = this.pendingSignals[idx];
        this.pendingSignals.splice(idx, 1);
        const from = this.sigSender.get(sig);
        this.sigSender.delete(sig);
        const off = buf + written + GUEST_MEM_BASE;
        bytes.fill(0, off, off + SIGINFO_SIZE);
        view.setUint32(off + 0, sig, true);
        view.setInt32(off + 4, 0, true);
        view.setInt32(off + 8, from ? 0 : 128, true);
        view.setUint32(off + 12, from?.pid ?? 0, true);
        view.setUint32(off + 16, from?.uid ?? 0, true);
        written += SIGINFO_SIZE;
      }
      return written === 0 ? -EAGAIN : written;
    }
    /** does this descriptor name a directory? a relative *at() path can only resolve against one. */
    fdIsDir(fd) {
      const entry = this.getFdEntry(fd);
      if (entry) return entry.type === "native" && entry.isDir === true;
      return this.vfs?.fdIsDir(fd) ?? false;
    }
    /** is this descriptor open at all, by any of the kinds this handler tracks? */
    fdIsOpen(fd) {
      return this.fdTable.has(fd) || (this.vfs?.hasFd(fd) ?? false);
    }
    /**
     * what /proc/self/fd should show: every open descriptor and what it names. the targets are the
     * strings linux uses -- a real path for a file, "pipe:[n]" / "socket:[n]" / "anon_inode:[kind]"
     * for the things that have no path -- because programs read them and match on exactly those.
     */
    /**
     * write bytes to a descriptor that this handler owns end to end, for a caller outside write(2).
     * the descriptors the WORKER owns (native files, pipes, unix sockets) are not here: it writes to
     * those itself. returns bytes written, or a negative errno.
     */
    writeOutOfBand(fd, data) {
      const e = this.fdTable.get(fd);
      if (!e) return -EBADF;
      switch (e.type) {
        case "stdio":
          if (e.fd === 1) this.io.writeStdout(data);
          else if (e.fd === 2) this.io.writeStderr(data);
          else return -EBADF;
          return data.length;
        case "tty":
          this.io.writeStdout(data);
          return data.length;
        // /proc and /dev are synthesised rather than stored, so there is nothing to append bytes to.
        case "vfs":
          return -EINVAL;
        case "devnode": {
          const kind = devKind(e.major, e.minor);
          return kind === "null" || kind === "zero" ? data.length : -EINVAL;
        }
        default:
          return -EINVAL;
      }
    }
    /** the inotify watch lines for a descriptor's fdinfo; supplied by the worker, which can reach
     *  the kernel's watch registry. */
    inotifyFdinfo = null;
    /** where a native descriptor currently sits; the offset lives on the host side of the bridge. */
    fdPosition = null;
    openFdList() {
      const out = [];
      for (const [fd, e] of this.fdTable) {
        let target;
        switch (e.type) {
          case "native":
            target = e.path;
            break;
          case "vfs":
            target = this.vfs?.fdPath(e.fd) ?? `anon_inode:[vfs]`;
            break;
          case "stdio":
            target = "/dev/console";
            break;
          case "tty":
            target = "/dev/tty";
            break;
          case "jitterctl":
            target = "/dev/jitter";
            break;
          case "pipe_read":
          case "pipe_write":
            target = `pipe:[${fd}]`;
            break;
          case "socket":
            target = `socket:[${e.socketId}]`;
            break;
          case "unixsock":
            target = `socket:[${fd}]`;
            break;
          case "inotify":
            target = "anon_inode:inotify";
            break;
          case "timerfd":
            target = "anon_inode:[timerfd]";
            break;
          case "eventfd":
            target = "anon_inode:[eventfd]";
            break;
          case "epoll":
            target = "anon_inode:[eventpoll]";
            break;
          case "signalfd":
            target = "anon_inode:[signalfd]";
            break;
          case "mqueue":
            target = `/dev/mqueue/${e.name}`;
            break;
          case "devnode":
            target = `/dev/${e.major}:${e.minor}`;
            break;
          case "pidfd":
            target = "anon_inode:[pidfd]";
            break;
        }
        let info = `pos:	${e.type === "native" ? this.fdPosition?.(fd) ?? 0 : 0}
flags:	0${((this.fdFlags.get(fd) ?? 0) >>> 0).toString(8)}
mnt_id:	1
`;
        if (e.type === "inotify") info += this.inotifyFdinfo?.(e.id) ?? "";
        if (e.type === "pidfd") info += `Pid:	${e.pid}
NSpid:	${e.pid}
`;
        out.push({ fd, target, info });
      }
      if (this.vfs) {
        for (const fd of this.vfs.openFdNumbers()) {
          if (!this.fdTable.has(fd)) {
            out.push({ fd, target: this.vfs.fdPath(fd) ?? "anon_inode:[vfs]", info: "pos:	0\nflags:	0100000\nmnt_id:	1\n" });
          }
        }
      }
      out.sort((a, b) => a.fd - b.fd);
      return out;
    }
    /**
     * whether the descriptor was opened for reading / writing. record locks are gated on this:
     * posix says a shared lock needs a descriptor open for reading and an exclusive lock one open
     * for writing, and answers EBADF otherwise -- which is what makes fcntl(1, F_SETLK, F_RDLCK)
     * fail on a write-only stdout.
     */
    /** the descriptor's async-notification state: who to signal, with what, and whether O_ASYNC is
     *  armed at all. O_ASYNC is 0o20000 on aarch64 (asm-generic FASYNC). */
    asyncState(fd) {
      const o = this.fdOwner.get(fd);
      return {
        pid: o?.pid ?? 0,
        sig: this.fdSig.get(fd) ?? 0,
        armed: ((this.fdFlags.get(fd) ?? 0) & 8192) !== 0 && (o?.pid ?? 0) !== 0
      };
    }
    /** is this process running as root? */
    isRoot() {
      return this.euid === 0;
    }
    /** does this process hold a capability in its EFFECTIVE set? this, not the uid, is what a
     *  privileged operation should ask. */
    hasCap(cap) {
      return (this.capEff & 1n << BigInt(cap)) !== 0n;
    }
    /**
     * recompute the capability sets after a uid change, the way linux's cap_emulate_setxuid does.
     *
     * capabilities and uid 0 are not the same thing, but they are tied: a process that gives up root
     * gives up the powers root came with, and one that returns to root gets back only what it still
     * has PERMITTED. only setuid(2) did any of this, so a process that dropped privilege with
     * setresuid or setreuid kept every capability it had -- and then chroot, mknod and the rest all
     * succeeded for a caller that was no longer root.
     */
    recomputeCapsAfterUid(prevRuid, prevEuid, prevSuid) {
      const wasRootAnywhere = prevRuid === 0 || prevEuid === 0 || prevSuid === 0;
      const isRootAnywhere = this.ruid === 0 || this.euid === 0 || this.suid === 0;
      if (wasRootAnywhere && !isRootAnywhere) {
        this.capPerm = 0n;
        this.capEff = 0n;
        return;
      }
      if (prevEuid === 0 && this.euid !== 0) this.capEff = 0n;
      else if (prevEuid !== 0 && this.euid === 0) this.capEff = this.capPerm;
    }
    fdAccess(fd) {
      const entry = this.fdTable.get(fd);
      switch (entry?.type) {
        // bidirectional by construction: nothing about them is one-way.
        case "tty":
        case "jitterctl":
        case "socket":
        case "unixsock":
        case "eventfd":
          return { readable: true, writable: true };
        // a pipe end goes exactly one way, whatever flags happen to be recorded for it.
        case "pipe_read":
          return { readable: true, writable: false };
        case "pipe_write":
          return { readable: false, writable: true };
        // event queues you can only consume from.
        case "signalfd":
        case "inotify":
        case "timerfd":
        case "epoll":
          return { readable: true, writable: false };
        default:
          break;
      }
      const mode = (this.fdFlags.get(fd) ?? 0) & 3;
      return { readable: mode === 0 || mode === 2, writable: mode === 1 || mode === 2 };
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
    /**
     * The drivers behind the device nodes we back. These are linux's semantics, not approximations:
     * a program that reads /dev/zero for a megabyte of zeros, or writes a log to /dev/null, or fills
     * a disk-full test against /dev/full, is relying on exactly this.
     */
    deviceRead(major, minor, bytes, buf, count) {
      switch (devKind(major, minor)) {
        case "null":
          return 0;
        // always at end of file
        case "zero":
        case "full":
          bytes.fill(0, buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
          return count;
        case "random": {
          const dst = bytes.subarray(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
          crypto.getRandomValues(dst);
          return count;
        }
        case "tty": {
          const d = this.io.readStdin(count);
          if (!d || d.length === 0) return 0;
          bytes.set(d, buf + GUEST_MEM_BASE);
          return d.length;
        }
        default:
          return -6;
      }
    }
    deviceWrite(major, minor, bytes, buf, count) {
      switch (devKind(major, minor)) {
        case "null":
          return count;
        // swallowed
        case "zero":
          return count;
        // linux accepts and discards
        case "full":
          return -28;
        // ENOSPC -- that is what /dev/full is FOR
        case "random":
          return count;
        // writing seeds the pool; accepting is enough
        case "tty":
          this.io.writeStdout(bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count));
          return count;
        default:
          return -6;
      }
    }
    /**
     * the checks pread/pwrite/preadv/pwritev all share, in linux's order.
     *
     * Returns a negative errno, or 0 to proceed. The positional calls are defined only on seekable
     * descriptors: a pipe, socket or fifo is ESPIPE, not a short read. And the OFFSET is checked
     * before the buffer, so a negative offset is EINVAL even when the buffer is also bad.
     */
    positionalIoCheck(fd, offset, forWrite) {
      if (this.io.pipeReady?.(fd)) return -ESPIPE;
      if (!this.fdExists(fd)) return -EBADF;
      const acc = this.fdAccess(fd);
      if (forWrite ? !acc.writable : !acc.readable) return -EBADF;
      const entry = this.fdTable.get(fd);
      switch (entry?.type) {
        case "pipe_read":
        case "pipe_write":
        case "socket":
        case "unixsock":
        case "tty":
        case "stdio":
        case "eventfd":
        case "epoll":
        case "inotify":
        case "timerfd":
        case "jitterctl":
        case "signalfd":
        case "mqueue":
          return -ESPIPE;
        default:
          break;
      }
      if (this.fdIsPath(fd)) return -EBADF;
      if (entry?.type === "native" && entry.isDir) return -EISDIR;
      if (offset < 0n) return -EINVAL;
      return 0;
    }
    /**
     * total length of an iovec array, or a negative errno.
     *
     * iovcnt is bounded by IOV_MAX (1024 on linux) and each length is a size_t that the kernel reads
     * as signed: a negative one is EINVAL, not an enormous transfer. the sum overflowing is EINVAL too.
     */
    iovTotal(view, iov, iovcnt) {
      if (iovcnt < 0 || iovcnt > IOV_MAX) return -EINVAL;
      let total = 0;
      for (let i = 0; i < iovcnt; i++) {
        const len = Number(view.getBigInt64(iov + i * 16 + 8 + GUEST_MEM_BASE, true));
        if (len < 0) return -EINVAL;
        total += len;
        if (total > 2147483647) return -EINVAL;
      }
      return total;
    }
    /**
     * whether this descriptor number refers to anything at all.
     *
     * the guest fd namespace is split: pipes/sockets/native files live in fdTable, vfs-backed files
     * live in the vfs. a check against only one of them calls half the open descriptors in the
     * process closed.
     */
    fdExists(fd) {
      return this.fdTable.has(fd) || (this.vfs?.hasFd(fd) ?? false) || !!this.io.pipeReady?.(fd);
    }
    /**
     * whether epoll (and poll, and select) can meaningfully wait on this descriptor.
     *
     * linux answers EPERM for the ones it cannot: a regular file and a directory are always ready,
     * so there is nothing to wait for, and letting a caller add one would turn its event loop into
     * a spin. everything with a readiness state -- pipes, sockets, ttys, eventfds, timerfds, other
     * epoll instances -- is fair game.
     */
    fdPollable(fd) {
      if (this.io.pipeReady?.(fd)) return true;
      const entry = this.fdTable.get(fd);
      if (!entry) return false;
      switch (entry.type) {
        case "native":
        case "vfs":
          return false;
        default:
          return true;
      }
    }
    /** Whether the fd has data available for read without blocking. */
    isFdReadable(fd) {
      const pr = this.io.pipeReady?.(fd);
      if (pr) return pr.readable;
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
          return this.efdCount(entry.id) > 0n;
        // an inotify descriptor is readable exactly when an event is waiting for it, and the queue
        // lives in the kernel -- so poll, select and epoll have to ask.
        case "inotify":
          return (this.io.inotifyPending?.(fd) ?? 0) > 0;
        // a timerfd is readable exactly when it has expirations nobody has collected, and that
        // is a question about the clock rather than about any event we were told of.
        case "timerfd": {
          this.checkTimers();
          return (this.timerFds.get(entry.id)?.ticks ?? 0n) > 0n;
        }
        case "epoll":
          return false;
        // a pidfd reports readable exactly once its process has exited, which is how poll waits for
        // a process without being its parent.
        case "pidfd":
          return this.io.pidExited?.(entry.pid) ?? false;
        case "jitterctl":
          return entry.resp !== null;
        // readable once the host has answered
        // a listening socket is readable when a connection is queued; a connected one when its rx
        // ring has bytes or the peer is gone. answering "always" instead is a lie with teeth: the
        // caller believes there is something there and issues a blocking accept or read that never
        // returns. libuv asserts and aborts on it, which is how neovim died.
        case "unixsock":
          return this.io.unixReady?.(fd).readable ?? false;
        case "signalfd":
          return this.signalfdReady(fd);
        case "mqueue":
          return this.io.mqReady?.(fd)?.readable ?? false;
        // /dev/null, /dev/zero and friends are always ready; a tty node follows the terminal.
        case "devnode":
          return devKind(entry.major, entry.minor) === "tty" ? this.io.stdinHasData?.() ?? true : true;
      }
    }
    /** Whether the fd can accept a write without blocking. */
    isFdWritable(fd) {
      const pw = this.io.pipeReady?.(fd);
      if (pw) return pw.writable;
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
          return this.efdCount(entry.id) < 0xFFFFFFFFFFFFFFFEn;
        case "inotify":
        case "timerfd":
        case "epoll":
        case "signalfd":
        case "pidfd":
          return false;
        case "jitterctl":
          return true;
        // the request side always accepts a write
        case "unixsock":
          return this.io.unixReady?.(fd).writable ?? false;
        case "mqueue":
          return this.io.mqReady?.(fd)?.writable ?? false;
        case "devnode":
          return true;
      }
    }
    /**
     * epoll_pwait and epoll_pwait2, which differ only in how the timeout is spelled.
     *
     * struct epoll_event is `{ uint32_t events; uint64_t data; }`, and it is packed ONLY on x86_64
     * (__EPOLL_PACKED is defined for that arch and nothing else). On aarch64 it takes its natural
     * layout: 16 bytes, with data at offset 8. Writing the x86 layout put every data pointer four
     * bytes early and made every event the caller read back garbage -- which is most of what the
     * epoll tests were failing on.
     *
     * Returns the number of events, or a negative errno.
     */
    /**
     * may the guest WRITE this range? a mapping it can only read is as much a fault as none at all,
     * and a syscall that fills a buffer has to say so before filling it.
     */
    ptrWriteFaults(bytes, addr, len) {
      if (this.ptrFaults(bytes, addr, len)) return true;
      for (const v of this.vmas) {
        if (addr < v.end && addr + len > v.start && !(v.prot & 2)) return true;
      }
      return false;
    }
    /** how many epoll instances deep a chain rooted here goes; a non-epoll descriptor is zero. */
    epollDepth(fd, seen = /* @__PURE__ */ new Set()) {
      const set = this.epollSets?.get(fd);
      if (!set || seen.has(fd)) return 0;
      seen.add(fd);
      let d = 1;
      for (const child of set.keys()) d = Math.max(d, 1 + this.epollDepth(child, seen));
      seen.delete(fd);
      return d;
    }
    /** can `target` be reached from `from` by following epoll-watches-epoll edges? */
    epollReaches(from, target, seen = /* @__PURE__ */ new Set()) {
      if (from === target) return true;
      const set = this.epollSets?.get(from);
      if (!set || seen.has(from)) return false;
      seen.add(from);
      for (const child of set.keys()) if (this.epollReaches(child, target, seen)) return true;
      return false;
    }
    epollWait(view, bytes, epfd, eventsPtr, maxEvents, timeoutMs) {
      if (!this.fdExists(epfd)) return -EBADF;
      if (!this.epollSets?.has(epfd)) return -EINVAL;
      if (maxEvents <= 0) return -EINVAL;
      if (this.ptrWriteFaults(bytes, eventsPtr, maxEvents * EPOLL_EVENT_SIZE)) return -EFAULT;
      const eset = this.epollSets.get(epfd);
      const EPOLLIN = 1, EPOLLOUT = 4, EPOLLHUP = 16, EPOLLRDHUP = 8192;
      const EPOLLONESHOT = 1073741824, EPOLLET = 2147483648;
      const gather = () => {
        let n = 0;
        for (const [targetFd, entry] of eset) {
          if (n >= maxEvents) break;
          let readyEvents = 0;
          if (entry.events & EPOLLIN && this.isFdReadable(targetFd)) readyEvents |= EPOLLIN;
          if (entry.events & EPOLLOUT && this.isFdWritable(targetFd)) readyEvents |= EPOLLOUT;
          const fdEntry = this.fdTable.get(targetFd);
          if (fdEntry?.type === "pipe_read" && fdEntry.pipe.writeClosed) readyEvents |= EPOLLHUP;
          if (fdEntry?.type === "pipe_write" && fdEntry.pipe.readClosed) readyEvents |= EPOLLHUP;
          if (fdEntry?.type === "socket") {
            if (entry.events & EPOLLRDHUP && fdEntry.shutRd) readyEvents |= EPOLLRDHUP;
            if (fdEntry.shutRd && fdEntry.shutWr) readyEvents |= EPOLLHUP;
          }
          const report = entry.events & EPOLLET ? readyEvents & ~entry.lastReady : readyEvents;
          entry.lastReady = readyEvents;
          if (!report) continue;
          const off = eventsPtr + n * EPOLL_EVENT_SIZE + GUEST_MEM_BASE;
          view.setUint32(off, report, true);
          view.setBigUint64(off + EPOLL_DATA_OFFSET, entry.data, true);
          if (entry.events & EPOLLONESHOT) entry.events &= ~(EPOLLIN | EPOLLOUT);
          n++;
        }
        return n;
      };
      const deadline = timeoutMs < 0 ? -1 : this.nowMs() + timeoutMs;
      let seen = this.ioGeneration();
      let count = gather();
      while (count === 0 && timeoutMs !== 0) {
        const remaining = deadline < 0 ? -1 : Math.max(0, deadline - this.nowMs());
        if (deadline >= 0 && remaining === 0) break;
        if (!this.parkForIo(seen, remaining)) break;
        seen = this.ioGeneration();
        count = gather();
        if (this.hasPendingDeliverable()) {
          if (count === 0) return -EINTR;
          break;
        }
      }
      return count;
    }
    /**
     * who setpriority/getpriority are talking about. the worker supplies the process table, because
     * the answer is about OTHER processes and nothing in this handler can see them.
     */
    prioHook = null;
    priorityTargets(which, who) {
      const all = this.prioHook?.list() ?? [];
      switch (which) {
        case PRIO_PROCESS: {
          const want = who === 0 ? this.currentPid : who;
          return all.filter((p) => p.pid === want);
        }
        case PRIO_PGRP: {
          const self2 = all.find((p) => p.pid === this.currentPid);
          const want = who === 0 ? self2?.pgid ?? this.currentPid : who;
          return all.filter((p) => p.pgid === want);
        }
        default: {
          const want = who === 0 ? this.ruid : who;
          return all.filter((p) => p.ruid === want);
        }
      }
    }
    /** the current io readiness generation; 0 when nothing here can produce events. */
    ioGeneration() {
      return this.io.ioGeneration?.() ?? 0;
    }
    /**
     * park until something may have become ready, or `remainingMs` elapses (negative = no deadline
     * of our own). `seen` must be the generation read BEFORE the readiness check that came up
     * empty; otherwise an event landing between the check and the park is slept through.
     *
     * returns false when there is no way to block at all, which is the case off-device where poll
     * degrades to a non-blocking check. the park itself is bounded internally so a pending signal
     * is still noticed.
     */
    parkForIo(seen, remainingMs) {
      if (this.io.waitIo) {
        this.io.waitIo(seen, remainingMs);
        return true;
      }
      if (this.io.waitStdin) {
        this.io.waitStdin(remainingMs);
        return true;
      }
      return false;
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
      let a2 = cpu2.getReg(2);
      let a3 = cpu2.getReg(3);
      const a4 = cpu2.getReg(4);
      const a5 = cpu2.getReg(5);
      let openat2Path = null;
      let openat2NoSymlinks = false;
      const view = new DataView(memory2.buffer);
      const bytes = new Uint8Array(memory2.buffer);
      let ret = -BigInt(ENOSYS);
      this.sigSync();
      const pathArgs = PATH_ARG_REGS[nr];
      if (pathArgs !== void 0) {
        const args = [a0, a1, a2, a3, a4, a5];
        for (const i of pathArgs) {
          if (this.ptrFaults(bytes, Number(args[i]))) return this.retErr(cpu2, EFAULT);
          if (readGuestString(bytes, Number(args[i])).length > 4095) {
            return this.retErr(cpu2, ENAMETOOLONG);
          }
        }
      }
      const flagSpec = AT_FLAG_ARGS[nr];
      if (flagSpec !== void 0) {
        const args = [a0, a1, a2, a3, a4, a5];
        const given = Number(BigInt.asIntN(32, args[flagSpec[0]]));
        if ((given & ~flagSpec[1]) !== 0) return this.retErr(cpu2, EINVAL);
      }
      const dirfdArgs = DIRFD_ARG_REGS[nr];
      if (dirfdArgs !== void 0) {
        const args = [a0, a1, a2, a3, a4, a5];
        for (const [dfdReg, pathReg] of dirfdArgs) {
          const raw = readGuestString(bytes, Number(args[pathReg]));
          if (raw.length === 0 || raw.startsWith("/")) continue;
          const dfd = Number(BigInt.asIntN(32, args[dfdReg]));
          if (dfd === -100) continue;
          if (!this.fdIsOpen(dfd)) return this.retErr(cpu2, EBADF);
          if (!this.fdIsDir(dfd)) return this.retErr(cpu2, ENOTDIR);
        }
      }
      if (NET_TRACE && this._netActive && nr !== 64 && nr !== 66 && nr !== 63) this.nt(`sys ${nr} a0=${Number(a0 & 0xffffn)}`);
      switch (nr) {
        // ── Process lifecycle ──
        case 93:
          return { status: "exit", code: Number(a0) & 255 };
        case 94:
          return { status: "exit", code: Number(a0) & 255, group: true };
        case 220: {
          const flags = Number(a0);
          const SIGCHLD2 = 17;
          if (flags & SIGCHLD2) {
            const svcPC = view.getBigUint64(PC_OFFSET, true);
            return { status: "fork", resumePC: svcPC + 4n, flags, childStack: a1, ptid: a2, tls: a3, ctid: a4 };
          }
          ret = -BigInt(ENOSYS);
          break;
        }
        case 221: {
          if (this.ptrFaults(bytes, Number(a0))) {
            ret = -BigInt(EFAULT);
            break;
          }
          const pathname = readGuestString(bytes, Number(a0));
          {
            const pe221 = this.pathCheck(pathname);
            if (pe221) {
              ret = -BigInt(pe221);
              break;
            }
          }
          const argvPtrs = [];
          let argvAddr = Number(a1);
          while (true) {
            const ptr = Number(view.getBigUint64(argvAddr + GUEST_MEM_BASE, true));
            if (ptr === 0) break;
            argvPtrs.push(readGuestString(bytes, ptr));
            argvAddr += 8;
          }
          if (argvPtrs.length === 0) argvPtrs.push("");
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
              const encoded = exitCode & 65536 ? exitCode & 127 | (exitCode & 131072 ? 128 : 0) : (exitCode & 255) << 8;
              view.setUint32(wstatus + GUEST_MEM_BASE, encoded, true);
            }
            if (Number(a3) !== 0) this.writeRusage(view, Number(a3), 0, 0, 0);
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
          if (readEntry !== void 0 && (!this.fdAccess(fd).readable || this.fdIsPath(fd))) {
            ret = -BigInt(EBADF);
            break;
          }
          if ((readEntry !== void 0 || (this.vfs?.hasFd(fd) ?? false)) && count > 0 && this.ptrFaults(bytes, buf, count)) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (readEntry) {
            if (readEntry.type === "devnode") {
              ret = BigInt(this.deviceRead(readEntry.major, readEntry.minor, bytes, buf, count));
              break;
            }
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
              if (count < 8) {
                ret = -BigInt(EINVAL);
                break;
              }
              let efdR = this.eventfdHook?.({ op: "read", id: readEntry.id }) ?? [-EBADF];
              if (efdR[0] === -11) {
                if (this.isNonBlocking(fd)) {
                  ret = -BigInt(EAGAIN);
                  break;
                }
                let seen = this.ioGeneration();
                let efdIntr = false;
                while (efdR[0] === -11) {
                  if (!this.parkForIo(seen, -1)) break;
                  seen = this.ioGeneration();
                  if (this.hasPendingDeliverable()) {
                    efdIntr = true;
                    break;
                  }
                  efdR = this.eventfdHook?.({ op: "read", id: readEntry.id }) ?? [-EBADF];
                }
                if (efdR[0] === -11) {
                  ret = -BigInt(efdIntr ? EINTR : EAGAIN);
                  break;
                }
              }
              if (efdR[0] < 0) {
                ret = BigInt(efdR[0]);
                break;
              }
              view.setBigUint64(
                buf + GUEST_MEM_BASE,
                BigInt(efdR[2] >>> 0) << 32n | BigInt(efdR[1] >>> 0),
                true
              );
              ret = 8n;
              break;
            }
            if (readEntry.type === "inotify") {
              ret = -BigInt(EAGAIN);
              break;
            }
            if (readEntry.type === "timerfd") {
              if (count < 8) {
                ret = -BigInt(EINVAL);
                break;
              }
              const tfr = this.timerFds.get(readEntry.id);
              if (!tfr) {
                ret = -BigInt(EINVAL);
                break;
              }
              this.checkTimers();
              let tfErr = 0;
              while (tfr.ticks === 0n) {
                if (tfr.nextNs === 0n || this.isNonBlocking(fd)) {
                  tfErr = EAGAIN;
                  break;
                }
                const untilMs = Number((tfr.nextNs - this.timerClockNs(tfr.clockid)) / 1000000n);
                if (!this.parkForIo(this.ioGeneration(), Math.max(1, untilMs))) {
                  tfErr = EAGAIN;
                  break;
                }
                this.checkTimers();
                if (tfr.ticks === 0n && this.pendingSignals.length) {
                  tfErr = EINTR;
                  break;
                }
              }
              if (tfErr !== 0) {
                ret = -BigInt(tfErr);
                break;
              }
              view.setBigUint64(buf + GUEST_MEM_BASE, tfr.ticks, true);
              tfr.ticks = 0n;
              ret = 8n;
              break;
            }
            if (readEntry.type === "epoll" || readEntry.type === "mqueue") {
              ret = -BigInt(EINVAL);
              break;
            }
            if (readEntry.type === "jitterctl") {
              if (readEntry.resp === null) {
                const e = readEntry;
                return {
                  status: "bridge",
                  method: "jitter_ctl",
                  args: { cmd: e.req },
                  apply: (resp) => {
                    const body = resp?.bytes ?? new Uint8Array(0);
                    const out = new Uint8Array(4 + body.length);
                    new DataView(out.buffer).setInt32(0, Number(resp?.ret ?? 1), true);
                    out.set(body, 4);
                    e.resp = out;
                    e.pos = 0;
                    const n2 = Math.min(count, out.length);
                    bytes.set(out.subarray(0, n2), buf + GUEST_MEM_BASE);
                    e.pos = n2;
                    cpu2.setReg(0, BigInt(n2));
                  }
                };
              }
              const remaining = readEntry.resp.length - readEntry.pos;
              const n = Math.min(count, remaining);
              if (n > 0) {
                bytes.set(readEntry.resp.subarray(readEntry.pos, readEntry.pos + n), buf + GUEST_MEM_BASE);
                readEntry.pos += n;
              }
              ret = BigInt(n);
              break;
            }
            if (readEntry.type === "vfs") {
              ret = this.vfs ? BigInt(this.vfs.read(readEntry.fd, buf, count, view)) : -BigInt(EBADF);
              break;
            }
            if (readEntry.type === "native") {
              if (this.readNativeHook) {
                const got = this.readNativeHook(readEntry.nativeFd, count);
                if (typeof got === "number") {
                  ret = -BigInt(got);
                  break;
                }
                if (got.length) {
                  bytes.set(got, buf + GUEST_MEM_BASE);
                  this.notify(readEntry.path, IN_ACCESS, 0, false, { unlinked: readEntry.unlinked === true });
                }
                ret = BigInt(got.length);
                break;
              }
              const rcount = Math.min(count, BRIDGE_READ_MAX);
              return {
                status: "bridge",
                method: "fs_read",
                args: { fd: readEntry.nativeFd, count: rcount },
                apply: (resp) => {
                  if (resp.ret < 0) {
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                  } else if (resp.bytes) {
                    this.notify(readEntry.path, IN_ACCESS, 0, false, { unlinked: readEntry.unlinked === true });
                    bytes.set(resp.bytes, buf + GUEST_MEM_BASE);
                    cpu2.setReg(0, BigInt(resp.bytes.length));
                  } else {
                    cpu2.setReg(0, BigInt(resp.ret));
                  }
                }
              };
            }
            if (readEntry.type === "socket") {
              const scount = Math.min(count, BRIDGE_READ_MAX);
              return {
                status: "bridge",
                method: "net_read",
                args: {
                  socketId: readEntry.socketId,
                  count: scount,
                  nonblock: this.isNonBlockingFd(fd) ? 1 : 0
                },
                apply: (resp) => {
                  if (resp.ret < 0) {
                    this.nt(`read fd=${fd} req=${count} -> err ${resp.ret}`);
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                  } else if (resp.bytes) {
                    bytes.set(resp.bytes, buf + GUEST_MEM_BASE);
                    cpu2.setReg(0, BigInt(resp.bytes.length));
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
          if (writeEntry !== void 0 && (!this.fdAccess(fd).writable || this.fdIsPath(fd))) {
            ret = -BigInt(EBADF);
            break;
          }
          if ((writeEntry !== void 0 || (this.vfs?.hasFd(fd) ?? false)) && count > 0 && this.ptrFaults(bytes, buf, count)) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (writeEntry) {
            if (writeEntry.type === "devnode") {
              ret = BigInt(this.deviceWrite(writeEntry.major, writeEntry.minor, bytes, buf, count));
              break;
            }
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
            if (writeEntry.type === "jitterctl") {
              writeEntry.req += new TextDecoder().decode(bytes.subarray(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count));
              ret = BigInt(count);
              break;
            }
            if (writeEntry.type === "vfs") {
              ret = this.vfs ? BigInt(this.vfs.write(writeEntry.fd, buf, count, view)) : -BigInt(EBADF);
              break;
            }
            if (writeEntry.type === "native") {
              const wSeals = this.sealsOf(writeEntry);
              if (wSeals & 8) {
                ret = -BigInt(EPERM);
                break;
              }
              const writeData = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
              const b64 = bytesToBase64(writeData);
              this.invalidateStat(writeEntry.path);
              return {
                status: "bridge",
                method: "fs_write",
                args: {
                  fd: writeEntry.nativeFd,
                  data: b64,
                  limit: this.fsizeLimitArg(),
                  // F_SEAL_GROW: the file may not get longer, so a write past the end is
                  // refused rather than shortened. only the filesystem knows where the end is.
                  noGrow: wSeals & 4 ? 1 : 0
                },
                apply: (resp) => {
                  if (resp.ret < 0) {
                    if (resp.ret === -EFBIG) this.raiseSignal(25);
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                  } else {
                    this.notify(writeEntry.path, IN_MODIFY, 0, false, { unlinked: writeEntry.unlinked === true });
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
              const efdWrite = () => this.eventfdHook?.({ op: "write", id: writeEntry.id, value: value.toString() }) ?? [-EBADF];
              let efdW = efdWrite();
              if (efdW[0] === -11) {
                if (this.isNonBlocking(fd)) {
                  ret = -BigInt(EAGAIN);
                  break;
                }
                let seen = this.ioGeneration();
                let efdWIntr = false;
                while (efdW[0] === -11) {
                  if (!this.parkForIo(seen, -1)) break;
                  seen = this.ioGeneration();
                  if (this.hasPendingDeliverable()) {
                    efdWIntr = true;
                    break;
                  }
                  efdW = efdWrite();
                }
                if (efdW[0] === -11) {
                  ret = -BigInt(efdWIntr ? EINTR : EAGAIN);
                  break;
                }
              }
              if (efdW[0] < 0) {
                ret = BigInt(efdW[0]);
                break;
              }
              ret = 8n;
              break;
            }
            if (writeEntry.type === "inotify" || writeEntry.type === "timerfd" || writeEntry.type === "epoll") {
              ret = -BigInt(EBADF);
              break;
            }
            if (writeEntry.type === "mqueue") {
              ret = -BigInt(EINVAL);
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
          if (!this.fdExists(fd) || !this.fdAccess(fd).writable || this.fdIsPath(fd)) {
            ret = -BigInt(EBADF);
            break;
          }
          if (iovcnt < 0 || iovcnt > IOV_MAX) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (iovcnt > 0 && this.ptrFaults(bytes, iovBase, iovcnt * 16)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const wvLen = this.iovTotal(view, iovBase, iovcnt);
          if (wvLen < 0) {
            ret = BigInt(wvLen);
            break;
          }
          let wvFault = false;
          for (let i = 0; i < iovcnt; i++) {
            const bPtr = Number(view.getBigUint64(iovBase + i * 16 + GUEST_MEM_BASE, true));
            const bLen = Number(view.getBigUint64(iovBase + i * 16 + 8 + GUEST_MEM_BASE, true));
            if (bLen > 0 && this.ptrFaults(bytes, bPtr, bLen)) {
              wvFault = true;
              break;
            }
          }
          if (wvFault) {
            ret = -BigInt(EFAULT);
            break;
          }
          let totalWritten = 0;
          let writevBad = false;
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
            this.invalidateStat(writevEntry.path);
            return {
              status: "bridge",
              method: "fs_write",
              args: { fd: writevEntry.nativeFd, data: b64, limit: this.fsizeLimitArg() },
              apply: (resp) => {
                if (resp.ret < 0) {
                  if (resp.ret === -EFBIG) this.raiseSignal(25);
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                } else {
                  this.notifyFd(fd, IN_MODIFY);
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
              writevBad = true;
              break;
            }
            totalWritten += len;
          }
          ret = writevBad ? -BigInt(EBADF) : BigInt(totalWritten);
          break;
        }
        case 65: {
          const rvFd = Number(a0);
          const rvIov = Number(a1);
          const rvCnt = Number(a2);
          if (!this.fdExists(rvFd) || !this.fdAccess(rvFd).readable || this.fdIsPath(rvFd)) {
            ret = -BigInt(EBADF);
            break;
          }
          if (rvCnt < 0 || rvCnt > IOV_MAX) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (rvCnt > 0 && this.ptrFaults(bytes, rvIov, rvCnt * 16)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const rvLen = this.iovTotal(view, rvIov, rvCnt);
          if (rvLen < 0) {
            ret = BigInt(rvLen);
            break;
          }
          const rvEntry = this.getFdEntry(rvFd);
          if (rvEntry?.type === "native" && rvEntry.isDir) {
            ret = -BigInt(EISDIR);
            break;
          }
          let rvFault = false;
          for (let i = 0; i < rvCnt; i++) {
            const len = Number(view.getBigUint64(rvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
            if (len === 0) continue;
            const base = Number(view.getBigUint64(rvIov + i * 16 + GUEST_MEM_BASE, true));
            if (this.ptrWriteFaults(bytes, base, len)) {
              rvFault = true;
              break;
            }
          }
          if (rvFault) {
            ret = -BigInt(EFAULT);
            break;
          }
          const rvScatter = (d) => {
            let off = 0;
            for (let i = 0; i < rvCnt && off < d.length; i++) {
              const len = Number(view.getBigUint64(rvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
              if (len === 0) continue;
              const base = Number(view.getBigUint64(rvIov + i * 16 + GUEST_MEM_BASE, true));
              const n = Math.min(len, d.length - off);
              bytes.set(d.subarray(off, off + n), base + GUEST_MEM_BASE);
              off += n;
            }
          };
          const rvWant = Math.min(rvLen, BRIDGE_READ_MAX);
          const applyBridgeRead = (resp) => {
            if (resp.ret < 0) {
              this.nt(`readv fd=${rvFd} req=${rvWant} -> err ${resp.ret}`);
              cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            } else if (resp.bytes) {
              const d = resp.bytes;
              rvScatter(d);
              this.nt(`readv fd=${rvFd} req=${rvWant} -> ${d.length} (ret=${resp.ret})`);
              cpu2.setReg(0, BigInt(d.length));
            } else {
              this.nt(`readv fd=${rvFd} req=${rvWant} -> EOF ret=${resp.ret}`);
              cpu2.setReg(0, BigInt(resp.ret));
            }
          };
          if (rvEntry?.type === "socket") {
            if (rvWant === 0) {
              ret = 0n;
              break;
            }
            return { status: "bridge", method: "net_read", args: { socketId: rvEntry.socketId, count: rvWant }, apply: applyBridgeRead };
          }
          if (rvEntry?.type === "native") {
            if (rvWant === 0) {
              ret = 0n;
              break;
            }
            return { status: "bridge", method: "fs_read", args: { fd: rvEntry.nativeFd, count: rvWant }, apply: applyBridgeRead };
          }
          let rvTotal = 0;
          let rvErr = null;
          for (let i = 0; i < rvCnt; i++) {
            const base = Number(view.getBigUint64(rvIov + i * 16 + GUEST_MEM_BASE, true));
            const len = Number(view.getBigUint64(rvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
            if (len === 0) continue;
            if (this.vfs) {
              const n = this.vfs.read(rvFd, base, len, view);
              if (n < 0) {
                rvErr = rvTotal === 0 ? BigInt(n) : BigInt(rvTotal);
                break;
              }
              rvTotal += n;
              if (n < len) break;
            }
          }
          ret = rvErr ?? BigInt(rvTotal);
          break;
        }
        case 67: {
          const prFd = Number(a0), prBuf = Number(a1), prCount = Number(a2);
          const prOffset = BigInt.asIntN(64, a3);
          const prErr = this.positionalIoCheck(prFd, prOffset, false);
          if (prErr !== 0) {
            ret = BigInt(prErr);
            break;
          }
          if (prCount > 0 && this.ptrFaults(bytes, prBuf, prCount)) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (prCount === 0) {
            ret = 0n;
            break;
          }
          const prEntry = this.fdTable.get(prFd);
          if (prEntry?.type === "native") {
            return {
              status: "bridge",
              method: "fs_pread",
              args: { fd: prEntry.nativeFd, offset: Number(prOffset), count: Math.min(prCount, BRIDGE_READ_MAX) },
              apply: (resp) => {
                if (resp.ret < 0) {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                  return;
                }
                const d = resp.bytes ?? new Uint8Array(0);
                new Uint8Array(memory2.buffer).set(d, prBuf + GUEST_MEM_BASE);
                cpu2.setReg(0, BigInt(d.length));
              }
            };
          }
          if (this.vfs?.hasFd(prFd)) {
            const savedPos = this.vfs.lseek(prFd, 0n, 1);
            if (savedPos < 0n) {
              ret = savedPos;
              break;
            }
            this.vfs.lseek(prFd, prOffset, 0);
            const n = this.vfs.read(prFd, prBuf, prCount, view);
            this.vfs.lseek(prFd, savedPos, 0);
            ret = BigInt(n);
            break;
          }
          ret = -BigInt(EBADF);
          break;
        }
        case 68: {
          const pwFd = Number(a0), pwBuf = Number(a1), pwCount = Number(a2);
          const pwOffset = BigInt.asIntN(64, a3);
          const pwErr = this.positionalIoCheck(pwFd, pwOffset, true);
          if (pwErr !== 0) {
            ret = BigInt(pwErr);
            break;
          }
          if (pwCount > 0 && this.ptrFaults(bytes, pwBuf, pwCount)) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (pwCount === 0) {
            ret = 0n;
            break;
          }
          const pwEntry = this.fdTable.get(pwFd);
          if (this.sealsOf(pwEntry) & 8) {
            ret = -BigInt(EPERM);
            break;
          }
          if (pwEntry?.type === "native") {
            const payload = bytes.slice(pwBuf + GUEST_MEM_BASE, pwBuf + GUEST_MEM_BASE + pwCount);
            this.invalidateStat(pwEntry.path);
            return {
              status: "bridge",
              method: "fs_pwrite",
              args: {
                fd: pwEntry.nativeFd,
                offset: Number(pwOffset),
                data: bytesToBase64(payload),
                noGrow: this.sealsOf(pwEntry) & 4 ? 1 : 0
              },
              apply: (resp) => {
                if (resp.ret >= 0) this.notifyFd(pwFd, IN_MODIFY);
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          if (this.vfs?.hasFd(pwFd)) {
            const savedPos = this.vfs.lseek(pwFd, 0n, 1);
            if (savedPos < 0n) {
              ret = savedPos;
              break;
            }
            this.vfs.lseek(pwFd, pwOffset, 0);
            const n = this.vfs.write(pwFd, pwBuf, pwCount, view);
            this.vfs.lseek(pwFd, savedPos, 0);
            ret = BigInt(n);
            break;
          }
          ret = -BigInt(EBADF);
          break;
        }
        // ── Pipes & fd duplication ──
        case 59: {
          const pipefdAddr = Number(a0);
          if (!this.fdAvailable()) {
            ret = -BigInt(EMFILE);
            break;
          }
          const pipe = { data: [], writeClosed: false, readClosed: false };
          const readFd = this.allocFd();
          this.fdTable.set(readFd, { type: "pipe_read", pipe });
          if (!this.fdAvailable()) {
            this.fdTable.delete(readFd);
            ret = -BigInt(EMFILE);
            break;
          }
          const writeFd = this.allocFd();
          this.fdTable.set(writeFd, { type: "pipe_write", pipe });
          if (this.ptrFaults(bytes, pipefdAddr, 8)) {
            this.fdTable.delete(readFd);
            this.fdTable.delete(writeFd);
            ret = -BigInt(EFAULT);
            break;
          }
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
          if (ret >= 0n) this.fdFlags.set(Number(ret), this.fdFlags.get(oldfd) ?? 0);
          this.ft(`dup ${oldfd} -> ${BigInt.asIntN(64, ret)}`);
          break;
        }
        case 24: {
          const oldfd = Number(a0);
          const newfd = Number(BigInt.asIntN(32, a1));
          const dup3flags = Number(BigInt.asIntN(32, a2));
          this.ft(`dup3 ${oldfd} -> ${newfd}`);
          if (dup3flags & ~524288) {
            ret = -BigInt(EINVAL);
            break;
          }
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
          if (ret >= 0n) this.fdFlags.set(newfd, this.fdFlags.get(oldfd) ?? 0);
          if (dup3flags & 524288) this.fdCloexec.add(newfd);
          else this.fdCloexec.delete(newfd);
          break;
        }
        // ── File operations ──
        /**
         * openat2(dirfd, pathname, how, size).
         *
         * openat with its flags and mode moved into a struct that can grow, plus a `resolve` word
         * of restrictions on how the path may be walked. the extensible struct has real rules: a
         * short one is EINVAL, a long one is only allowed if everything past what we understand is
         * zero, and anything else is E2BIG -- that is how a program discovers what this kernel
         * supports. this used to accept anything, ignore `resolve` entirely, and then look only in
         * the per-process /proc and /dev tree rather than at the real filesystem.
         */
        case 437: {
          const OPEN_HOW_SIZE = 24;
          const howPtr = Number(a2);
          const howSize = Number(BigInt.asUintN(64, a3));
          if (howSize < OPEN_HOW_SIZE) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (this.ptrFaults(bytes, howPtr, howSize)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const howBase = howPtr + GUEST_MEM_BASE;
          let tooBig = false;
          for (let off = OPEN_HOW_SIZE; off < howSize; off++) if (bytes[howBase + off] !== 0) {
            tooBig = true;
            break;
          }
          if (tooBig) {
            ret = -BigInt(E2BIG);
            break;
          }
          const howFlags = view.getBigUint64(howBase, true);
          const howMode = view.getBigUint64(howBase + 8, true);
          const howResolve = view.getBigUint64(howBase + 16, true);
          if ((howFlags & ~VALID_OPEN_FLAGS) !== 0n) {
            ret = -BigInt(EINVAL);
            break;
          }
          if ((howResolve & ~VALID_RESOLVE) !== 0n) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (howResolve & RESOLVE_BENEATH && howResolve & RESOLVE_IN_ROOT) {
            ret = -BigInt(EINVAL);
            break;
          }
          const willCreate = (howFlags & (0o100n | O_TMPFILE_BIT)) !== 0n;
          if (willCreate ? (howMode & ~0o7777n) !== 0n : howMode !== 0n) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (howFlags & O_TMPFILE_BIT) {
            if ((howFlags & (O_TMPFILE_BIT | O_DIRECTORY_B | 0o100n)) !== (O_TMPFILE_BIT | O_DIRECTORY_B)) {
              ret = -BigInt(EINVAL);
              break;
            }
            if ((howFlags & 3n) !== 1n && (howFlags & 3n) !== 2n) {
              ret = -BigInt(EINVAL);
              break;
            }
          }
          if (howFlags & O_PATH_B && (howFlags & ~O_PATH_FLAGS) !== 0n) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (this.ptrFaults(bytes, Number(a1))) {
            ret = -BigInt(EFAULT);
            break;
          }
          const scoped = this.scopeOpenPath(a0, readGuestString(bytes, Number(a1)), howResolve);
          if (typeof scoped === "number") {
            ret = -BigInt(scoped);
            break;
          }
          openat2Path = scoped;
          openat2NoSymlinks = (howResolve & RESOLVE_NO_SYMLINKS) !== 0n;
          a2 = howFlags;
          a3 = howMode;
        }
        // eslint-disable-next-line no-fallthrough
        case 56: {
          if (openat2Path === null && this.ptrFaults(bytes, Number(a1))) {
            ret = -BigInt(EFAULT);
            break;
          }
          const raw56 = openat2Path === null ? readGuestString(bytes, Number(a1)) : null;
          if (raw56 !== null && this.isEmptyPath(raw56)) {
            ret = -BigInt(ENOENT);
            break;
          }
          const path = openat2Path ?? this.resolveAt(a0, raw56);
          const flags56 = Number(a2);
          const mode56 = Number(a3);
          if (path === "/dev/jitter") {
            const jfd = this.allocFd();
            this.fdTable.set(jfd, { type: "jitterctl", req: "", resp: null, pos: 0 });
            ret = BigInt(jfd);
            break;
          }
          if (path === "/dev/tty") {
            const ttyFd = this.allocFd();
            this.fdTable.set(ttyFd, { type: "tty" });
            this.ft(`open ${path} -> ${ttyFd}`);
            ret = BigInt(ttyFd);
            break;
          }
          if (!this.fdAvailable(0)) {
            ret = -BigInt(EMFILE);
            break;
          }
          const magic56 = path.match(/^\/proc\/(?:self|\d+)\/fd\/(\d+)$/);
          if (magic56) {
            const srcFd = Number(magic56[1]);
            const srcEnt = this.getFdEntry(srcFd);
            if (srcEnt?.type === "native") {
              const reFd = this.allocFd();
              const reFlags = flags56;
              return {
                status: "bridge",
                method: "fs_reopen",
                // O_TRUNC through the link is a shrink of the object, and a file sealed against
                // shrinking refuses it.
                args: {
                  fd: srcEnt.nativeFd,
                  flags: reFlags,
                  noShrink: this.sealsOf(srcEnt) & 2 ? 1 : 0
                },
                apply: (resp) => {
                  if (resp.ret < 0) {
                    this.unregisterFd(reFd);
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                    return;
                  }
                  this.fdTable.set(reFd, {
                    type: "native",
                    nativeFd: resp.ret,
                    path: srcEnt.path,
                    ofdId: this.newOfdIdPublic(),
                    isDir: srcEnt.isDir,
                    ftype: srcEnt.ftype,
                    unlinked: srcEnt.unlinked,
                    sealable: srcEnt.sealable
                  });
                  this.nativeRefs.set(resp.ret, 1);
                  this.fdFlags.set(reFd, reFlags & ~(64 | 512 | 524288));
                  if (reFlags & 524288) this.fdCloexec.add(reFd);
                  cpu2.setReg(0, BigInt(reFd));
                }
              };
            }
          }
          const bp56 = this.bridgePath(path);
          if (bp56 !== null) {
            const guestFd = this.allocFd();
            return {
              status: "bridge",
              method: "fs_open",
              args: { path: bp56, flags: flags56, mode: mode56, noSymlinks: openat2NoSymlinks ? 1 : 0 },
              apply: (resp) => {
                if (resp.ret === -19 && resp._meta) {
                  const dmode = resp._meta[0] ?? 0;
                  const drdev = resp._meta[1] ?? 0;
                  const major = drdev >> 8 & 4095;
                  const minor = drdev & 255 | drdev >> 12 & 1048320;
                  if (!devKind(major, minor)) {
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(-6)));
                    return;
                  }
                  this.fdTable.set(guestFd, { type: "devnode", major, minor });
                  this.fdFlags.set(guestFd, flags56 & ~(64 | 128 | 512 | 524288));
                  if (flags56 & 524288) this.fdCloexec.add(guestFd);
                  this.ft(`open ${bp56} -> ${guestFd} (dev ${major}:${minor} mode ${dmode.toString(8)})`);
                  cpu2.setReg(0, BigInt(guestFd));
                  return;
                }
                if (resp.ret < 0) {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                } else {
                  const openedDir = ((resp._meta?.[0] ?? 0) & 61440) === 16384;
                  this.fdTable.set(guestFd, {
                    type: "native",
                    nativeFd: resp.ret,
                    path: bp56,
                    ofdId: this.newOfdId(),
                    isDir: openedDir,
                    ftype: (resp._meta?.[0] ?? 0) & 61440
                  });
                  this.nativeRefs.set(resp.ret, 1);
                  if (resp._meta?.[2]) this.notify(bp56, IN_CREATE, 0, false);
                  this.notify(bp56, IN_OPEN, 0, openedDir);
                  if (flags56 & 512) this.notify(bp56, IN_MODIFY, 0, false);
                  this.fdFlags.set(guestFd, flags56 & ~(64 | 128 | 512 | 524288));
                  if (flags56 & 524288) this.fdCloexec.add(guestFd);
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
          const closedWritable = this.fdAccess(fd).writable;
          this.fdFlags.delete(fd);
          this.fdCloexec.delete(fd);
          if (closeEntry) {
            if (closeEntry.type === "pipe_write") closeEntry.pipe.writeClosed = true;
            if (closeEntry.type === "pipe_read") closeEntry.pipe.readClosed = true;
            if (closeEntry.type === "vfs" && this.vfs) this.vfs.close(closeEntry.fd);
            if (closeEntry.type === "native") {
              const closedPath = closeEntry.path;
              const closedDir = closeEntry.isDir === true, closedGone = closeEntry.unlinked === true;
              this.fdTable.delete(fd);
              this.notify(
                closedPath,
                closedWritable ? IN_CLOSE_WRITE : IN_CLOSE_NOWRITE,
                0,
                closedDir,
                { unlinked: closedGone }
              );
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
            if (closeEntry.type === "eventfd") {
              this.fdTable.delete(fd);
              let stillHeld = false;
              for (const [, e] of this.fdTable) if (e.type === "eventfd" && e.id === closeEntry.id) {
                stillHeld = true;
                break;
              }
              if (!stillHeld) this.eventfdHook?.({ op: "close", id: closeEntry.id, pid: this.currentPid });
              ret = 0n;
              break;
            }
            if (closeEntry.type === "epoll") this.epollSets?.delete(fd);
            this.fdTable.delete(fd);
            if (closeEntry.type === "timerfd") {
              let stillOpen = false;
              for (const other of this.fdTable.values()) {
                if (other.type === "timerfd" && other.id === closeEntry.id) {
                  stillOpen = true;
                  break;
                }
              }
              if (!stillOpen) this.timerFds.delete(closeEntry.id);
            }
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
          if (ftEntry !== void 0 && ftEntry.type !== "vfs" && ftEntry.type !== "native") {
            ret = -BigInt(EINVAL);
            break;
          }
          if (ftLen < 0) {
            ret = -BigInt(EINVAL);
            break;
          }
          const ftLimit = this.rlimitOf(RLIMIT_FSIZE);
          if (ftLimit.cur !== RLIM_INFINITY && BigInt(ftLen) > ftLimit.cur) {
            this.raiseSignal(25);
            ret = -BigInt(EFBIG);
            break;
          }
          if (this.fdIsOpen(ftFd) && (!this.fdAccess(ftFd).writable || this.fdIsPath(ftFd))) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (ftEntry && ftEntry.type === "vfs" && this.vfs) {
            ret = BigInt(this.vfs.ftruncate(ftEntry.fd, ftLen));
          } else if (ftEntry && ftEntry.type === "native") {
            this.invalidateStat(ftEntry.path);
            const ftSeals = this.sealsOf(ftEntry);
            return {
              status: "bridge",
              method: "fs_ftruncate",
              args: {
                fd: ftEntry.nativeFd,
                length: ftLen,
                noShrink: ftSeals & 2 ? 1 : 0,
                noGrow: ftSeals & 4 ? 1 : 0
              },
              apply: (resp) => {
                if (resp.ret >= 0) this.notify(ftEntry.path, IN_MODIFY, 0, false);
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
              // a memory device has no position to lose: linux lets you seek one and hands back the
              // offset you asked for, which is how `dd skip=` over /dev/zero works.
              case "devnode":
                ret = devKind(lseekEntry.major, lseekEntry.minor) === "tty" ? -BigInt(ESPIPE) : BigInt.asUintN(64, BigInt.asIntN(64, a1));
                break;
              case "native":
                return {
                  status: "bridge",
                  method: "fs_lseek",
                  args: { fd: lseekEntry.nativeFd, offset: Number(BigInt.asIntN(64, a1)), whence: Number(a2) },
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
            } else if (fstatEntry.type === "devnode") {
              writeChardevStat(view, statBuf, fstatEntry.major << 8 | fstatEntry.minor & 255 | (fstatEntry.minor & ~255) << 12);
              ret = 0n;
            } else if (fstatEntry.type === "stdio" || fstatEntry.type === "tty" || fstatEntry.type === "pipe_read" || fstatEntry.type === "pipe_write" || fstatEntry.type === "inotify" || fstatEntry.type === "timerfd" || fstatEntry.type === "eventfd" || fstatEntry.type === "epoll") {
              writeChardevStat(view, statBuf);
              ret = 0n;
            } else if (fstatEntry.type === "native") {
              return {
                status: "bridge",
                method: "fs_fstat",
                args: { fd: fstatEntry.nativeFd },
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
          const raw79 = readGuestString(bytes, Number(a1));
          if (this.isEmptyPath(raw79)) {
            if (!(Number(a3) & 4096)) {
              ret = -BigInt(ENOENT);
              break;
            }
          } else {
            const pe79 = this.pathCheck(raw79);
            if (pe79) {
              ret = -BigInt(pe79);
              break;
            }
          }
          const path = this.resolveAt(a0, raw79);
          const statBuf79 = Number(a2);
          if (this.ptrFaults(bytes, statBuf79, 128)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const bp79 = this.bridgePath(path);
          if (bp79 !== null) {
            const nofollow79 = (Number(a3) & 256) !== 0;
            const key79 = nofollow79 ? "L:" + bp79 : bp79;
            const cached = this.statCache.get(key79);
            if (cached && cached.expiresAt > Date.now()) {
              writeBridgeStat(view, statBuf79, { _meta: cached.meta });
              ret = 0n;
              break;
            }
            return {
              status: "bridge",
              method: nofollow79 ? "fs_lstat" : "fs_stat",
              args: { path: bp79 },
              apply: (resp) => {
                if (resp.ret < 0) {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                } else {
                  writeBridgeStat(view, statBuf79, resp);
                  if (resp._meta) {
                    this.cacheStat(key79, {
                      meta: resp._meta.slice(),
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
                    this.cacheStat("L:" + childPath, {
                      meta: [
                        entry.mode,
                        entry.size ?? 0,
                        entry.nlink ?? 1,
                        entry.mtime ?? 0,
                        entry.uid ?? 0,
                        entry.gid ?? 0,
                        entry.ino ?? 0,
                        entry.rdev ?? 0,
                        entry.atime ?? 0,
                        entry.ctime ?? 0,
                        entry.mtimeNsec ?? 0,
                        entry.atimeNsec ?? 0,
                        entry.ctimeNsec ?? 0,
                        entry.blocks ?? 0,
                        entry.blksize ?? 4096
                      ],
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
          if (this.ptrFaults(bytes, Number(a1))) {
            ret = -BigInt(EFAULT);
            break;
          }
          const raw34 = readGuestString(bytes, Number(a1));
          {
            const pe = this.pathCheck(raw34);
            if (pe) {
              ret = -BigInt(pe);
              break;
            }
          }
          const path = this.resolveAt(a0, raw34);
          const bp34 = this.bridgePath(path);
          if (bp34 !== null) {
            this.invalidateStat(bp34);
            return {
              status: "bridge",
              method: "fs_mkdir",
              args: { path: bp34, mode: Number(a2) },
              apply: (resp) => {
                if (resp.ret >= 0) this.notify(bp34, IN_CREATE | IN_ISDIR);
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
            this.invalidateStat(bp35);
            return {
              status: "bridge",
              method: "fs_unlink",
              // AT_REMOVEDIR is what tells unlink from rmdir. dropping it made both the same call,
              // so unlink(directory) succeeded (and recursively) instead of answering EISDIR.
              args: { path: bp35, removeDir: (Number(a2) & 512) !== 0 },
              apply: (resp) => {
                if (resp.ret >= 0) {
                  const removedDir = (Number(a2) & 512) !== 0;
                  for (const e of this.fdTable.values()) {
                    if (e.type === "native" && e.path === bp35) e.unlinked = true;
                  }
                  if (!removedDir) this.notify(bp35, IN_ATTRIB, 0, false, { selfOnly: true });
                  this.notify(bp35, IN_DELETE_SELF, 0, removedDir);
                  this.notify(bp35, IN_DELETE, 0, removedDir);
                }
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
          const rawMknod = readGuestString(bytes, Number(a1));
          {
            const pe = this.pathCheck(rawMknod);
            if (pe) {
              ret = -BigInt(pe);
              break;
            }
          }
          const path = this.resolveAt(a0, rawMknod);
          const mode = Number(a2);
          const bMknod = this.bridgePath(path);
          if (bMknod) {
            this.invalidateStat(bMknod);
            return {
              status: "bridge",
              method: "fs_mknod",
              // the device number matters: a char/block node IS its major:minor, and dropping it left
              // every device the guest made pointing at nothing.
              args: { path: bMknod, mode, dev: Number(a3) },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
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
          const rawOld38 = readGuestString(bytes, Number(a1));
          const rawNew38 = readGuestString(bytes, Number(a3));
          {
            const pe = this.pathCheck(rawOld38) || this.pathCheck(rawNew38);
            if (pe) {
              ret = -BigInt(pe);
              break;
            }
          }
          const oldpath = this.resolveAt(a0, rawOld38);
          const newpath = this.resolveAt(a2, rawNew38);
          const bOld = this.bridgePath(oldpath);
          if (bOld) {
            const bNew = this.bridgePath(newpath) ?? newpath;
            this.invalidateStat(bOld);
            this.invalidateStat(bNew);
            return {
              status: "bridge",
              method: "fs_rename",
              args: { oldPath: bOld, newPath: bNew },
              apply: (resp) => {
                if (resp.ret >= 0) {
                  const ck = ++this.renameCookie;
                  this.notify(bOld, IN_MOVED_FROM, ck, void 0, { probe: bNew });
                  this.notify(bNew, IN_MOVED_TO, ck);
                  this.notify(bOld, IN_MOVE_SELF, 0, void 0, { probe: bNew });
                }
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
          const rawLink = readGuestString(bytes, Number(a2));
          if (rawLink === "") {
            ret = -BigInt(ENOENT);
            break;
          }
          const linkpath = this.resolveAt(a1, rawLink);
          const bLink = this.bridgePath(linkpath);
          if (bLink) {
            this.invalidateStat(bLink);
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
          const oldStr = readGuestString(bytes, Number(a1));
          const newStr = readGuestString(bytes, Number(a3));
          if (oldStr === "" || newStr === "") {
            ret = -BigInt(ENOENT);
            break;
          }
          const oldpath = this.resolveAt(a0, oldStr);
          const newpath = this.resolveAt(a2, newStr);
          const selfFd = /^\/proc\/(self|\d+)\/fd\/(\d+)$/.exec(oldpath);
          if (selfFd) {
            const srcFd = Number(selfFd[2]);
            const nativeSrc = this.fdNative(srcFd);
            if (nativeSrc >= 0) {
              const bLinkNew2 = this.bridgePath(newpath) ?? newpath;
              this.invalidateStat(bLinkNew2);
              return {
                status: "bridge",
                method: "fs_link_fd",
                args: { fd: nativeSrc, newPath: bLinkNew2, creds: this.getCreds() },
                apply: (resp) => {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                }
              };
            }
          }
          const bLinkOld = this.bridgePath(oldpath);
          if (bLinkOld) {
            const bLinkNew = this.bridgePath(newpath) ?? newpath;
            this.invalidateStat(bLinkNew);
            return {
              status: "bridge",
              method: "fs_link",
              args: { oldPath: bLinkOld, newPath: bLinkNew },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          if (this.vfs) {
            ret = BigInt(this.vfs.linkat(oldpath, newpath));
          } else {
            ret = -BigInt(ENOSYS);
          }
          break;
        }
        case 88: {
          const utFlags = Number(BigInt.asIntN(32, a3));
          if (utFlags & ~AT_SYMLINK_NOFOLLOW) {
            ret = -BigInt(EINVAL);
            break;
          }
          const utTimes = Number(a2);
          const ts = [0, UTIME_NOW, 0, UTIME_NOW];
          if (utTimes !== 0) {
            if (this.ptrFaults(bytes, utTimes, 32)) {
              ret = -BigInt(EFAULT);
              break;
            }
            let bad = false;
            for (let i = 0; i < 2; i++) {
              const sec = view.getBigInt64(utTimes + i * 16 + GUEST_MEM_BASE, true);
              const nsec = Number(view.getBigInt64(utTimes + i * 16 + 8 + GUEST_MEM_BASE, true));
              if (nsec !== UTIME_NOW && nsec !== UTIME_OMIT && (nsec < 0 || nsec >= 1e9)) bad = true;
              ts[i * 2] = Number(sec);
              ts[i * 2 + 1] = nsec;
            }
            if (bad) {
              ret = -BigInt(EINVAL);
              break;
            }
            if (ts[1] === UTIME_OMIT && ts[3] === UTIME_OMIT) {
              ret = 0n;
              break;
            }
          }
          const utRaw = Number(a1) === 0 ? null : readGuestString(bytes, Number(a1));
          if (utRaw === null) {
            const utEnt = this.getFdEntry(Number(BigInt.asIntN(32, a0)));
            if (!utEnt || utEnt.type !== "native") {
              ret = this.fdExists(Number(BigInt.asIntN(32, a0))) ? 0n : -BigInt(EBADF);
              break;
            }
            this.invalidateStat(utEnt.path);
            return {
              status: "bridge",
              method: "fs_utimens",
              args: { path: utEnt.path, times: ts, nofollow: false },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret ?? 0)));
              }
            };
          }
          {
            const peUt = this.pathCheck(utRaw);
            if (peUt) {
              ret = -BigInt(peUt);
              break;
            }
          }
          const utPath = this.resolveAt(a0, utRaw);
          const bUt = this.bridgePath(utPath);
          if (bUt !== null) {
            this.invalidateStat(bUt);
            return {
              status: "bridge",
              method: "fs_utimens",
              args: { path: bUt, times: ts, nofollow: (utFlags & AT_SYMLINK_NOFOLLOW) !== 0 },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret ?? 0)));
              }
            };
          }
          ret = 0n;
          break;
        }
        case 78: {
          const rawPath = readGuestString(bytes, Number(a1));
          const path = this.resolveAt(a0, rawPath);
          const buf = Number(a2);
          const bufsiz = Number(a3);
          if (rawPath === "") {
            const rlFd = Number(BigInt.asIntN(32, a0));
            const rlEnt = this.getFdEntry(rlFd);
            if (rlEnt?.type === "native" && this.fdIsPath(rlFd) && bufsiz > 0) {
              return {
                status: "bridge",
                method: "fs_readlink",
                args: { path: rlEnt.path },
                apply: (resp) => {
                  if (resp.ret < 0 || !resp.data) {
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret < 0 ? resp.ret : -EINVAL)));
                    return;
                  }
                  const target = Uint8Array.from(atob(resp.data), (c) => c.charCodeAt(0));
                  const n = Math.min(target.length, bufsiz);
                  if (badGuestPtr(bytes, buf, n)) {
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
                    return;
                  }
                  bytes.set(target.subarray(0, n), buf + GUEST_MEM_BASE);
                  cpu2.setReg(0, BigInt(n));
                }
              };
            }
            ret = -BigInt(ENOENT);
            break;
          }
          if (bufsiz <= 0) {
            ret = -BigInt(EINVAL);
            break;
          }
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
              ret = BigInt(this.vfs.resolveErrno?.() ?? -EINVAL);
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
          if (Number(BigInt.asUintN(32, a2)) & ~7) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (this.ptrFaults(bytes, Number(a1))) {
            ret = -BigInt(EFAULT);
            break;
          }
          const raw48 = readGuestString(bytes, Number(a1));
          {
            const pe = this.pathCheck(raw48);
            if (pe) {
              ret = -BigInt(pe);
              break;
            }
          }
          const path = this.resolveAt(a0, raw48);
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
          if (this.fdIsPath(fchmodFd) || this.vfs?.isOPath(fchmodFd)) {
            ret = -BigInt(EBADF);
            break;
          }
          const fchmodMode = Number(a1);
          const fchmodEntry = this.getFdEntry(fchmodFd);
          if (fchmodEntry?.type === "native") {
            this.invalidateStat(fchmodEntry.path);
            return {
              status: "bridge",
              method: "fs_chmod",
              args: { path: fchmodEntry.path, mode: fchmodMode },
              apply: (resp) => {
                if (resp.ret >= 0) this.notify(
                  fchmodEntry.path,
                  IN_ATTRIB,
                  0,
                  fchmodEntry.isDir === true,
                  { unlinked: fchmodEntry.unlinked === true }
                );
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
          if (this.ptrFaults(bytes, Number(a1), 1)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const raw53 = readGuestString(bytes, Number(a1));
          {
            const pe = this.pathCheck(raw53);
            if (pe) {
              ret = -BigInt(pe);
              break;
            }
          }
          {
            const de = this.atDirfdError(a0, raw53);
            if (de) {
              ret = -BigInt(de);
              break;
            }
          }
          const fchmodatPath = this.resolveAt(a0, raw53);
          const fchmodatMode = Number(a2);
          const bpChmod = this.bridgePath(fchmodatPath);
          if (bpChmod !== null) {
            this.invalidateStat(bpChmod);
            return {
              status: "bridge",
              method: "fs_chmod",
              args: { path: bpChmod, mode: fchmodatMode },
              apply: (resp) => {
                if (resp.ret >= 0) this.notify(bpChmod, IN_ATTRIB);
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
          const raw54 = readGuestString(bytes, Number(a1));
          {
            const pe = this.pathCheck(raw54);
            if (pe) {
              ret = -BigInt(pe);
              break;
            }
          }
          const fchownatPath = this.resolveAt(a0, raw54);
          const bpChown = this.bridgePath(fchownatPath);
          if (bpChown !== null) {
            this.invalidateStat(bpChown);
            return {
              status: "bridge",
              method: "fs_chown",
              args: { path: bpChown, uid: cOwner, gid: cGroup, nofollow: (fchownFlags & 256) !== 0 },
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
          if (this.fdIsPath(fchownFd) || this.vfs?.isOPath(fchownFd)) {
            ret = -BigInt(EBADF);
            break;
          }
          const owner = Number(BigInt.asIntN(32, a1));
          const group = Number(BigInt.asIntN(32, a2));
          const ent = this.getFdEntry(fchownFd);
          if (ent?.type === "native") {
            this.invalidateStat(ent.path);
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
          const statfsBuf = Number(a1);
          if (this.ptrWriteFaults(bytes, statfsBuf, 120)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const fillStatfs = () => {
            const sv = new DataView(memory2.buffer);
            const base = statfsBuf + GUEST_MEM_BASE;
            new Uint8Array(memory2.buffer).fill(0, base, base + 120);
            sv.setBigUint64(base + 0, 0x01021994n, true);
            sv.setBigUint64(base + 8, 4096n, true);
            sv.setBigUint64(base + 16, 1048576n, true);
            sv.setBigUint64(base + 24, 524288n, true);
            sv.setBigUint64(base + 32, 524288n, true);
            sv.setBigUint64(base + 40, 1048576n, true);
            sv.setBigUint64(base + 48, 524288n, true);
            sv.setBigUint64(base + 64, 255n, true);
            sv.setBigUint64(base + 72, 4096n, true);
          };
          if (nr === 44) {
            if (!this.fdExists(Number(a0))) {
              ret = -BigInt(EBADF);
              break;
            }
            fillStatfs();
            ret = 0n;
            break;
          }
          if (this.ptrFaults(bytes, Number(a0), 1)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const sfPath = readGuestString(bytes, Number(a0));
          if (sfPath.length > 4095) {
            ret = -BigInt(ENAMETOOLONG);
            break;
          }
          const sfBridge = this.bridgePath(sfPath);
          if (sfBridge === null) {
            if (!this.vfs || this.vfs.faccessat(-100, this.absolutizePath(sfPath), 0, 0) < 0) {
              ret = BigInt(this.vfs?.faccessat(-100, this.absolutizePath(sfPath), 0, 0) ?? -ENOENT);
              break;
            }
            fillStatfs();
            ret = 0n;
            break;
          }
          return {
            status: "bridge",
            method: "fs_stat",
            args: { path: sfBridge, creds: this.getCreds() },
            apply: (resp) => {
              if (resp.ret < 0) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                return;
              }
              fillStatfs();
              cpu2.setReg(0, 0n);
            }
          };
        }
        // ── ioctl ──
        case 29: {
          const fd = Number(a0);
          if (this.fdIsPath(fd) || this.vfs?.isOPath(fd)) {
            ret = -BigInt(EBADF);
            break;
          }
          if (!this.fdIsOpen(fd)) {
            ret = -BigInt(EBADF);
            break;
          }
          const request = Number(BigInt.asUintN(32, a1));
          const ioctlEntry = this.getFdEntry(fd);
          if (fd <= 4) this.ft(`ioctl ${fd} req=0x${request.toString(16)} type=${ioctlEntry?.type ?? (this.vfs?.hasFd(fd) ? "vfs" : "none")}`);
          const isStdio = ioctlEntry?.type === "stdio" || ioctlEntry?.type === "tty";
          if (!isStdio && (request === TCGETS || request === TCSETS || request === TIOCGWINSZ || request === TIOCSWINSZ)) {
            ret = -BigInt(ENOTTY);
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
            // RNDGETENTCNT: how much entropy the random pool holds. ours is the host's csprng, which
            // is never short of it, so the answer is the full pool -- the same number
            // /proc/sys/kernel/random/entropy_avail reports, which is what a caller compares it to.
            case 2147766784: {
              const rndDev = ioctlEntry?.type === "devnode" ? { major: ioctlEntry.major, minor: ioctlEntry.minor } : this.vfs?.fdDevice(fd) ?? null;
              if (!rndDev || devKind(rndDev.major, rndDev.minor) !== "random") {
                ret = -BigInt(ENOTTY);
                break;
              }
              const rndBuf = Number(a2);
              if (this.ptrFaults(bytes, rndBuf, 4)) {
                ret = -BigInt(14);
                break;
              }
              view.setInt32(rndBuf + GUEST_MEM_BASE, 4096, true);
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
            // ── the socket ioctls ────────────────────────────────────────────────────
            // these are how a program asks what the machine is attached to. every one of them used
            // to be ENOTTY, so `ifconfig` and `ip` could not list an interface, let alone bring one
            // up, and a program looking for its own address found none.
            case SIOCATMARK: {
              if (ioctlEntry?.type !== "socket") {
                ret = -BigInt(ENOTTY);
                break;
              }
              if (ioctlEntry.sockType !== 1) {
                ret = -BigInt(ENOTTY);
                break;
              }
              if (Number(a2) === 0 || this.ptrWriteFaults(bytes, Number(a2), 4)) {
                ret = -BigInt(EFAULT);
                break;
              }
              view.setInt32(Number(a2) + GUEST_MEM_BASE, 0, true);
              ret = 0n;
              break;
            }
            case SIOCGIFCONF: {
              if (ioctlEntry?.type !== "socket") {
                ret = -BigInt(ENOTTY);
                break;
              }
              const ifcPtr = Number(a2);
              if (ifcPtr === 0 || this.ptrWriteFaults(bytes, ifcPtr, 16)) {
                ret = -BigInt(EFAULT);
                break;
              }
              return {
                status: "bridge",
                method: "net_ifaddrs",
                args: {},
                apply: (resp) => {
                  const list = this.ifaceList(resp);
                  const v4 = list.filter((i) => i.family === 2);
                  const want = view.getInt32(ifcPtr + GUEST_MEM_BASE, true);
                  const bufPtr = Number(view.getBigUint64(ifcPtr + 8 + GUEST_MEM_BASE, true));
                  if (bufPtr === 0) {
                    view.setInt32(ifcPtr + GUEST_MEM_BASE, v4.length * 40, true);
                    cpu2.setReg(0, 0n);
                    return;
                  }
                  const n = Math.max(0, Math.min(v4.length, Math.floor(want / 40)));
                  if (n > 0 && this.ptrWriteFaults(bytes, bufPtr, n * 40)) {
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(-EFAULT)));
                    return;
                  }
                  for (let i = 0; i < n; i++) {
                    const base = bufPtr + i * 40;
                    bytes.fill(0, base + GUEST_MEM_BASE, base + GUEST_MEM_BASE + 40);
                    this.putIfName(bytes, base, v4[i].name);
                    this.putSockaddrRaw(view, base + 16, { family: 2, host: v4[i].addr, port: 0 });
                  }
                  view.setInt32(ifcPtr + GUEST_MEM_BASE, n * 40, true);
                  cpu2.setReg(0, 0n);
                }
              };
            }
            case SIOCGIFNAME:
            case SIOCGIFFLAGS:
            case SIOCSIFFLAGS:
            case SIOCGIFADDR:
            case SIOCGIFDSTADDR:
            case SIOCGIFBRDADDR:
            case SIOCGIFNETMASK:
            case SIOCGIFMTU:
            case SIOCGIFHWADDR:
            case SIOCGIFINDEX: {
              if (ioctlEntry?.type !== "socket") {
                ret = -BigInt(ENOTTY);
                break;
              }
              const ifrPtr = Number(a2);
              if (ifrPtr === 0 || this.ptrWriteFaults(bytes, ifrPtr, 40)) {
                ret = -BigInt(EFAULT);
                break;
              }
              if (request === SIOCSIFFLAGS && !this.hasCap(12)) {
                ret = -BigInt(EPERM);
                break;
              }
              const wantName = this.readIfName(bytes, ifrPtr);
              const wantIndex = view.getInt32(ifrPtr + 16 + GUEST_MEM_BASE, true);
              return {
                status: "bridge",
                method: "net_ifaddrs",
                args: {},
                apply: (resp) => {
                  const list = this.ifaceList(resp);
                  const iface = request === SIOCGIFNAME ? list.find((i) => i.index === wantIndex) : list.find((i) => i.name === wantName);
                  if (!iface) {
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(-19)));
                    return;
                  }
                  switch (request) {
                    case SIOCGIFNAME:
                      this.putIfName(bytes, ifrPtr, iface.name);
                      break;
                    case SIOCSIFFLAGS:
                      break;
                    case SIOCGIFFLAGS:
                      view.setUint16(ifrPtr + 16 + GUEST_MEM_BASE, iface.flags & 65535, true);
                      break;
                    case SIOCGIFADDR:
                      this.putSockaddrRaw(view, ifrPtr + 16, { family: iface.family, host: iface.addr, port: 0 });
                      break;
                    case SIOCGIFDSTADDR:
                    case SIOCGIFBRDADDR:
                      this.putSockaddrRaw(
                        view,
                        ifrPtr + 16,
                        { family: iface.family, host: iface.broadcast || "0.0.0.0", port: 0 }
                      );
                      break;
                    case SIOCGIFNETMASK:
                      this.putSockaddrRaw(
                        view,
                        ifrPtr + 16,
                        { family: iface.family, host: iface.netmask || "0.0.0.0", port: 0 }
                      );
                      break;
                    case SIOCGIFMTU:
                      view.setInt32(ifrPtr + 16 + GUEST_MEM_BASE, iface.mtu, true);
                      break;
                    case SIOCGIFINDEX:
                      view.setInt32(ifrPtr + 16 + GUEST_MEM_BASE, iface.index, true);
                      break;
                    case SIOCGIFHWADDR:
                      view.setUint16(ifrPtr + 16 + GUEST_MEM_BASE, iface.flags & 8 ? 772 : 1, true);
                      bytes.fill(0, ifrPtr + 18 + GUEST_MEM_BASE, ifrPtr + 24 + GUEST_MEM_BASE);
                      break;
                  }
                  cpu2.setReg(0, 0n);
                }
              };
            }
            default:
              ret = -BigInt(ENOTTY);
          }
          break;
        }
        // ── Memory management ──
        case 214: {
          const addr = a0;
          if (addr === 0n) {
            ret = this.brk;
          } else if (addr >= this.brk) {
            const grownFrom = this.withAddressSpace(() => {
              if (addr < this.brk) return -1;
              const needed = Number(addr) + GUEST_MEM_BASE;
              if (needed > this.maxBytes || !this.growTo(memory2, needed)) return -1;
              const from = Number(this.brk);
              this.brk = addr;
              this.vmaDirty = true;
              return from;
            });
            if (grownFrom >= 0) {
              const lo = grownFrom & ~4095, hi = Number(addr) + 4095 & ~4095;
              this.protNone = this.protNone.filter((r) => r.end <= lo || r.start >= hi);
              this.noBacking = this.noBacking.filter((r) => r.end <= lo || r.start >= hi);
              try {
                this.protHook?.(grownFrom, Number(addr) - grownFrom, 3);
              } catch {
              }
            }
            ret = this.brk;
          } else if (addr >= this.brkFloor) {
            const shrunkFrom = this.withAddressSpace(() => {
              if (addr >= this.brk) return -1;
              const from = Number(this.brk);
              this.brk = addr;
              this.vmaDirty = true;
              return from;
            });
            if (shrunkFrom >= 0) {
              const lo = Number(addr) + 4095 & ~4095;
              const hi = shrunkFrom + 4095 & ~4095;
              if (hi > lo) {
                this.protNone = this.protNone.filter((r) => r.end <= lo || r.start >= hi);
                this.protNone.push({ start: lo, end: hi });
                this.setLocked(lo, hi, false);
                try {
                  this.protHook?.(lo, hi - lo, 0);
                } catch {
                }
              }
            }
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
          if (!(flags & MAP_ANONYMOUS)) {
            const validFd = (this.vfs?.hasFd(mmapFd) ?? false) || this.fdTable.has(mmapFd);
            if (mmapFd < 0 || !validFd || this.fdIsPath(mmapFd) || (this.vfs?.isOPath(mmapFd) ?? false)) {
              ret = -BigInt(EBADF);
              break;
            }
          }
          const mapType = flags & 15;
          if (length === 0 || mapType !== 1 && mapType !== 2 && mapType !== 3) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (!(flags & MAP_ANONYMOUS)) {
            const macc = this.fdAccess(mmapFd);
            if (!macc.readable) {
              ret = -BigInt(EACCES);
              break;
            }
            if (flags & MAP_SHARED && prot & 2 && !macc.writable) {
              ret = -BigInt(EACCES);
              break;
            }
            if (flags & MAP_SHARED && prot & 2 && this.sealsOf(this.getFdEntry(mmapFd)) & 8) {
              ret = -BigInt(EPERM);
              break;
            }
          }
          if (mapType === 3) {
            const VALID_MMAP = 31 | 256 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 | 262144 | 524288 | 1048576 | 4227858432;
            if ((flags & ~VALID_MMAP) !== 0) {
              ret = -BigInt(95);
              break;
            }
          }
          if (mmapFd >= 0 && !(flags & MAP_ANONYMOUS) && this.vfs && this.vfs.hasFd(mmapFd)) {
            const am = this.vfs.fdAccessMode(mmapFd);
            const PROT_WRITE = 2;
            if (am === 1 || flags & MAP_SHARED && prot & PROT_WRITE && am === 0) {
              ret = -BigInt(EACCES);
              break;
            }
          }
          const MAP_FIXED_NOREPLACE = 1048576;
          const mmSealedRO = !(flags & MAP_ANONYMOUS) && flags & MAP_SHARED && ((this.sealsOf(this.getFdEntry(mmapFd)) & 8) !== 0 || !this.fdAccess(mmapFd).writable);
          const mmMaxProt = mmSealedRO ? 5 : 7;
          const mmVma = {
            prot: prot & 7,
            shared: (flags & MAP_SHARED) !== 0,
            maxProt: mmMaxProt,
            grow: (flags & 256) !== 0
            /* MAP_GROWSDOWN */
          };
          let mmErr = 0;
          const addr = this.withAddressSpace(() => {
            if (a0 !== 0n && flags & (MAP_FIXED | MAP_FIXED_NOREPLACE)) {
              if (flags & MAP_FIXED_NOREPLACE && this.vmaOverlaps(Number(a0), Number(a0) + aligned)) {
                mmErr = EEXIST;
                return 0n;
              }
              const end = a0 + BigInt(aligned);
              if (end > this.mmapBase) {
                this.mmapBase = end;
                this.vmaDirty = true;
              }
              if (this.vmaWouldExceed(Number(a0), Number(a0) + aligned, mmVma)) {
                mmErr = ENOMEM;
                return 0n;
              }
              this.vmaSet(Number(a0), Number(a0) + aligned, mmVma);
              return a0;
            }
            if (a0 !== 0n) {
              const hint = Number(a0 & ~0xFFFn);
              if (hint >= this.mmapArenaStart && hint + aligned <= Number(this.mmapBase) && !this.vmaOverlaps(hint, hint + aligned)) {
                if (this.vmaWouldExceed(hint, hint + aligned, mmVma)) {
                  mmErr = ENOMEM;
                  return 0n;
                }
                this.vmaSet(hint, hint + aligned, mmVma);
                return BigInt(hint);
              }
            }
            const picked = this.allocMmapRange(aligned);
            if (this.vmaWouldExceed(Number(picked), Number(picked) + aligned, mmVma)) {
              mmErr = ENOMEM;
              return 0n;
            }
            this.vmaSet(Number(picked), Number(picked) + aligned, mmVma);
            return picked;
          });
          if (mmErr) {
            ret = -BigInt(mmErr);
            break;
          }
          const needed = Number(addr) + aligned + GUEST_MEM_BASE;
          if (needed > this.maxBytes || !this.growTo(memory2, needed, 16)) {
            this.vmaSet(Number(addr), Number(addr) + aligned, null);
            ret = -BigInt(ENOMEM);
            break;
          }
          new Uint8Array(memory2.buffer).fill(0, Number(addr) + GUEST_MEM_BASE, Number(addr) + GUEST_MEM_BASE + aligned);
          let sharedKey;
          let fileBackedBytes = -1;
          if (mmapFd >= 0 && !(flags & MAP_ANONYMOUS)) {
            const fileOffset = Number(a5);
            const entry = this.fdTable.get(mmapFd);
            if (entry?.type === "native") sharedKey = `${entry.path}:${fileOffset}:${aligned}`;
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
              fileBackedBytes = data.length;
              if (flags & MAP_SHARED && prot & 2) {
                const backed = Math.min(length, data.length);
                this.fileMmaps.push({ start: Number(addr), len: aligned, fileLen: backed, nativeFd: entry.nativeFd, fileOffset });
              }
            }
          }
          if (flags & MAP_SHARED) {
            this.sharedRegions.push({ addr: Number(addr), len: aligned });
            this.sharedMapHook?.(Number(addr), aligned, sharedKey);
          }
          const mmStart = Number(addr), mmEnd = mmStart + aligned;
          this.protNone = this.protNone.filter((r) => r.end <= mmStart || r.start >= mmEnd);
          if ((prot & 7) === 0) this.protNone.push({ start: mmStart, end: mmEnd });
          if (flags & 8192 || this.lockFuture) this.setLocked(mmStart, mmEnd, true);
          try {
            this.protHook?.(mmStart, aligned, prot & 7);
          } catch {
          }
          this.noBacking = this.noBacking.filter((r) => r.end <= mmStart || r.start >= mmEnd);
          if (fileBackedBytes >= 0) {
            const firstDead = mmStart + Math.ceil(fileBackedBytes / 4096) * 4096;
            if (firstDead < mmEnd) {
              this.noBacking.push({ start: firstDead, end: mmEnd });
              try {
                this.protHook?.(firstDead, mmEnd - firstDead, PAGE_NO_BACKING);
              } catch {
              }
            }
          }
          ret = addr;
          break;
        }
        case 226: {
          if ((a0 & 0xFFFn) !== 0n) {
            ret = -BigInt(EINVAL);
            break;
          }
          const mpStart = Number(a0), mpEnd = mpStart + (Number(a1) + 4095 & ~4095);
          if (mpStart < 4096) {
            ret = -BigInt(ENOMEM);
            break;
          }
          const mpWant = Number(a2) & 7;
          const mpDenied = this.withAddressSpace(() => {
            const { i, j } = this.vmaSlice(mpStart, mpEnd);
            for (let k = i; k < j; k++) if (mpWant & ~this.vmas[k].maxProt) return EACCES;
            let extra = 0;
            for (let k = i; k < j; k++) {
              const v = this.vmas[k];
              if (v.prot === mpWant) continue;
              if (v.start < mpStart) extra++;
              if (v.end > mpEnd) extra++;
            }
            if (extra > 0 && this.vmas.length + extra > this.maxMapCount) return ENOMEM;
            this.vmaProtectLocked(mpStart, mpEnd, mpWant);
            return 0;
          });
          if (mpDenied) return this.retErr(cpu2, mpDenied);
          this.protNone = this.protNone.filter((r) => r.end <= mpStart || r.start >= mpEnd);
          if ((Number(a2) & 7) === 0) this.protNone.push({ start: mpStart, end: mpEnd });
          try {
            this.protHook?.(mpStart, mpEnd - mpStart, Number(a2) & 7);
          } catch {
          }
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
          if (this.withAddressSpace(() => this.vmaWouldExceed(umStart, umEnd, null))) {
            ret = -BigInt(ENOMEM);
            break;
          }
          this.sharedRegions = this.sharedRegions.filter((r) => r.addr + r.len <= umStart || r.addr >= umEnd);
          this.sharedUnmapHook?.(umStart, umEnd);
          this.protNone = this.protNone.filter((r) => r.end <= umStart || r.start >= umEnd);
          this.protNone.push({ start: umStart, end: umEnd });
          this.noBacking = this.noBacking.filter((r) => r.end <= umStart || r.start >= umEnd);
          try {
            this.protHook?.(umStart, umEnd - umStart, 0);
          } catch {
          }
          this.setLocked(umStart, umEnd, false);
          this.vmaSet(umStart, umEnd, null);
          const umFlush = this.buildMmapFlush(cpu2, memory2, umStart, unmapLen + 4095 & ~4095, true);
          if (umFlush) return umFlush;
          ret = 0n;
          break;
        }
        // ── Thread/process ──
        case 96: {
          ret = BigInt(this.currentPid);
          break;
        }
        case 99:
          if (Number(BigInt.asUintN(64, a1)) !== 24) {
            ret = -BigInt(EINVAL);
            break;
          }
          this.robustListHead = BigInt.asUintN(64, a0);
          ret = 0n;
          break;
        // gettid handled with getpid above
        case 172:
          ret = BigInt(this.tgid || this.currentPid);
          break;
        case 173:
          ret = BigInt(this.ppidHook ? this.ppidHook() : this.currentPpid);
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
          if (Number(BigInt.asIntN(64, a3)) !== 8) {
            ret = -BigInt(EINVAL);
            break;
          }
          const signum = Number(BigInt.asIntN(32, a0));
          const actPtr = Number(a1);
          const oldactPtr = Number(a2);
          if (actPtr !== 0 && this.ptrFaults(bytes, actPtr, 32)) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (signum < 1 || signum > 64 || actPtr !== 0 && (signum === 9 || signum === 19)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (!this.signalHandlers) this.signalHandlers = /* @__PURE__ */ new Map();
          const oldSa = this.signalHandlers.get(signum);
          if (actPtr !== 0) {
            const base = actPtr + GUEST_MEM_BASE;
            const entry = {
              handler: view.getBigUint64(base, true),
              flags: view.getBigUint64(base + 8, true),
              restorer: view.getBigUint64(base + 16, true),
              mask: view.getBigUint64(base + 24, true)
            };
            this.signalHandlers.set(signum, entry);
            this.sigPublish(signum, entry);
          }
          if (oldactPtr !== 0) {
            if (this.ptrWriteFaults(bytes, oldactPtr, 32)) {
              ret = -BigInt(EFAULT);
              break;
            }
            const base = oldactPtr + GUEST_MEM_BASE;
            view.setBigUint64(base, oldSa?.handler ?? 0n, true);
            view.setBigUint64(base + 8, oldSa?.flags ?? 0n, true);
            view.setBigUint64(base + 16, oldSa?.restorer ?? 0n, true);
            view.setBigUint64(base + 24, oldSa?.mask ?? 0n, true);
          }
          ret = 0n;
          break;
        }
        case 135: {
          if (Number(BigInt.asIntN(64, a3)) !== 8) {
            ret = -BigInt(EINVAL);
            break;
          }
          const setPtr = Number(a1), oldsetPtr = Number(a2);
          const previous = this.signalMask ?? 0n;
          if (setPtr !== 0) {
            if (this.ptrFaults(bytes, setPtr, 8)) {
              ret = -BigInt(EFAULT);
              break;
            }
            const how = Number(BigInt.asIntN(32, a0));
            if (how !== 0 && how !== 1 && how !== 2) {
              ret = -BigInt(EINVAL);
              break;
            }
            const newSet = view.getBigUint64(setPtr + GUEST_MEM_BASE, true) & ~(1n << 8n | 1n << 18n);
            if (how === 0) this.signalMask = previous | newSet;
            else if (how === 1) this.signalMask = previous & ~newSet;
            else this.signalMask = newSet;
            if (this.signalMask !== previous) this.signalsUnblocked?.();
          }
          if (oldsetPtr !== 0) {
            if (this.ptrWriteFaults(bytes, oldsetPtr, 8)) {
              ret = -BigInt(EFAULT);
              break;
            }
            view.setBigUint64(oldsetPtr + GUEST_MEM_BASE, previous, true);
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
          const pendSize = Number(BigInt.asIntN(64, a1));
          if (pendSize > 8) {
            ret = -BigInt(EINVAL);
            break;
          }
          const pendPtr = Number(a0);
          if (pendSize > 0 && this.ptrWriteFaults(bytes, pendPtr, pendSize)) {
            ret = -BigInt(EFAULT);
            break;
          }
          let pendMask = 0n;
          for (const sig of this.pendingSignals) if (sig >= 1 && sig <= 64) pendMask |= 1n << BigInt(sig - 1);
          const outBytes = new Uint8Array(8);
          new DataView(outBytes.buffer).setBigUint64(0, pendMask, true);
          bytes.set(outBytes.subarray(0, pendSize), pendPtr + GUEST_MEM_BASE);
          ret = 0n;
          break;
        }
        // ── Filesystem info ──
        case 17: {
          const buf = Number(a0);
          const size = Number(a1);
          const hostCwd = this.vfs ? this.vfs.getcwd() : this.cwd;
          if (this.rootDir !== "/" && hostCwd !== this.rootDir && !hostCwd.startsWith(this.rootDir + "/")) {
            ret = -BigInt(ENOENT);
            break;
          }
          const cwd = this.hostToGuest(hostCwd);
          const encoded = new TextEncoder().encode(cwd + "\0");
          if (encoded.length > size) {
            ret = -BigInt(34);
            break;
          }
          if (this.ptrFaults(new Uint8Array(memory2.buffer), buf, encoded.length)) {
            ret = -BigInt(EFAULT);
            break;
          }
          new Uint8Array(memory2.buffer).set(encoded, buf + GUEST_MEM_BASE);
          ret = a0;
          break;
        }
        case 49: {
          if (this.ptrFaults(bytes, Number(a0))) {
            ret = -BigInt(EFAULT);
            break;
          }
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
                if (!mode || (mode & 61440) !== 16384) {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(-ENOTDIR)));
                  return;
                }
                const dUid = resp._meta ? resp._meta[4] : 0;
                const dGid = resp._meta ? resp._meta[5] : 0;
                if (!this.mayExec(mode, dUid, dGid)) {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(-EACCES)));
                  return;
                }
                let realCwd = bPath;
                if (resp.data) {
                  try {
                    const d = atob(resp.data);
                    if (d.length > 0) realCwd = d;
                  } catch {
                  }
                }
                this.cwd = realCwd;
                if (this.vfs) this.vfs.setCwd(realCwd);
                cpu2.setReg(0, 0n);
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
          if (this.ptrFaults(new Uint8Array(memory2.buffer), Number(a0), 6 * 65)) {
            ret = -BigInt(14);
            break;
          }
          const buf = Number(a0) + GUEST_MEM_BASE;
          const fields = [
            "Linux",
            this.utsHook?.get("host") || "jitter",
            "6.1.0",
            "#1",
            "aarch64",
            this.utsHook?.get("domain") || "(none)"
          ];
          for (let i = 0; i < fields.length; i++) {
            const encoded = new TextEncoder().encode(fields[i] + "\0");
            new Uint8Array(memory2.buffer).set(encoded, buf + i * 65);
          }
          ret = 0n;
          break;
        }
        // ── Time ──
        case 113: {
          const gtClk = Number(BigInt.asIntN(32, a0));
          if (!this.knownClock(gtClk)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (this.ptrFaults(bytes, Number(a1), 16)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const tp = Number(a1) + GUEST_MEM_BASE;
          const ns = this.clockNs(gtClk);
          view.setBigUint64(tp, ns / 1000000000n, true);
          view.setBigUint64(tp + 8, ns % 1000000000n, true);
          ret = 0n;
          break;
        }
        case 112: {
          const stClk = Number(BigInt.asIntN(32, a0));
          if (!this.knownClock(stClk)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (stClk !== 0 && stClk !== 8 && stClk !== 11) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (this.ptrFaults(bytes, Number(a1), 16)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const stSec = view.getBigInt64(Number(a1) + GUEST_MEM_BASE, true);
          const stNsec = view.getBigInt64(Number(a1) + GUEST_MEM_BASE + 8, true);
          if (stNsec < 0n || stNsec >= 1000000000n || stSec < 0n) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (!this.hasCap(CAP_SYS_TIME)) {
            ret = -BigInt(EPERM);
            break;
          }
          const want = stSec * 1000000000n + stNsec;
          this.clockOffsetNs = want - (this.clkRealBaseNs + this.elapsedNs());
          ret = 0n;
          break;
        }
        case 169: {
          if (this.ptrFaults(new Uint8Array(memory2.buffer), Number(a0), 16)) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (a1 !== 0n && this.ptrFaults(new Uint8Array(memory2.buffer), Number(a1), 8)) {
            ret = -BigInt(EFAULT);
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
          if (!this.fdTable.has(fd) && !(this.vfs?.hasFd(fd) ?? false)) {
            ret = -BigInt(EBADF);
            break;
          }
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
              this.fdFlags.set(newfd, this.fdFlags.get(fd) ?? 0);
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
              const modifiable = 1024 | 2048 | 8192;
              const current = this.fdFlags.get(fd) ?? 0;
              const newFlags = current & ~modifiable | Number(a2) & modifiable;
              this.fdFlags.set(fd, newFlags);
              if ((current ^ newFlags) & 8192) this.ownerChanged?.(fd);
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
            // F_SETLK
            case 7:
            // F_SETLKW
            case 13:
            // F_SETLK64
            case 14:
              ret = 0n;
              break;
            case 8: {
              const who = Number(BigInt.asIntN(32, a2));
              if (who === 0) this.fdOwner.delete(fd);
              else this.fdOwner.set(fd, who > 0 ? { type: F_OWNER_PID, pid: who } : { type: F_OWNER_PGRP, pid: -who });
              this.ownerChanged?.(fd);
              ret = 0n;
              break;
            }
            case 9: {
              const o = this.fdOwner.get(fd);
              ret = BigInt(!o ? 0 : o.type === F_OWNER_PGRP ? -o.pid : o.pid);
              break;
            }
            case 10: {
              this.fdSig.set(fd, Number(a2));
              this.ownerChanged?.(fd);
              const se = this.getFdEntry(fd);
              if (se?.type === "native") this.leaseHook?.("sig", se.nativeFd, se.path, 0, Number(a2), false, true);
              ret = 0n;
              break;
            }
            case 11:
              ret = BigInt(this.fdSig.get(fd) ?? 0);
              break;
            case 15: {
              const exPtr = Number(a2);
              if (this.ptrFaults(bytes, exPtr, 8)) {
                ret = -BigInt(EFAULT);
                break;
              }
              const otype = view.getInt32(exPtr + GUEST_MEM_BASE, true);
              const opid = view.getInt32(exPtr + GUEST_MEM_BASE + 4, true);
              if (otype !== F_OWNER_TID && otype !== F_OWNER_PID && otype !== F_OWNER_PGRP) {
                ret = -BigInt(EINVAL);
                break;
              }
              if (opid === 0) this.fdOwner.delete(fd);
              else this.fdOwner.set(fd, { type: otype, pid: opid });
              this.ownerChanged?.(fd);
              ret = 0n;
              break;
            }
            case 16: {
              const exPtr = Number(a2);
              if (this.ptrFaults(bytes, exPtr, 8)) {
                ret = -BigInt(EFAULT);
                break;
              }
              const o = this.fdOwner.get(fd);
              view.setInt32(exPtr + GUEST_MEM_BASE, o?.type ?? F_OWNER_PID, true);
              view.setInt32(exPtr + GUEST_MEM_BASE + 4, o?.pid ?? 0, true);
              ret = 0n;
              break;
            }
            case 1024: {
              const le = this.getFdEntry(fd);
              if (!le || le.type !== "native") {
                ret = -BigInt(EINVAL);
                break;
              }
              const larg = Number(a2);
              if (larg !== 0 && larg !== 1 && larg !== 2) {
                ret = -BigInt(EINVAL);
                break;
              }
              const lfd = le.nativeFd, lpath = le.path;
              const lsig = this.fdSig.get(fd) || 29;
              const leuid = this.euid, lhook = this.leaseHook;
              return {
                status: "bridge",
                method: "fs_stat",
                args: { path: lpath },
                apply: (resp) => {
                  if (resp.ret < 0) {
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                    return;
                  }
                  const mode = (resp._meta ? resp._meta[0] : resp.mode) ?? 0;
                  const uid = (resp._meta ? resp._meta[4] : resp.uid) ?? 0;
                  const regular = (mode & 61440) === 32768;
                  const owner = leuid === 0 || uid === leuid;
                  const r = lhook ? lhook("set", lfd, lpath, larg, lsig, owner, regular) : -EINVAL;
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(r)));
                }
              };
            }
            case 1025: {
              const ge = this.getFdEntry(fd);
              if (!ge || ge.type !== "native") {
                ret = 2n;
                break;
              }
              ret = BigInt(this.leaseHook?.("get", ge.nativeFd, ge.path, 0, 0, false, true) ?? 2);
              break;
            }
            case 1026:
              ret = 0n;
              break;
            // a pipe's capacity is a promise about how much can be buffered before a writer
            // blocks, so it has to be the truth: our rings are a fixed slot in the kernel's shared
            // buffer, and claiming a megabyte would strand any program that sized its writes by the
            // answer. linux rounds a request up to a power of two and clamps it, so report what we
            // can actually hold and clamp requests to it.
            case 1033: {
              const sealEntry = this.getFdEntry(fd);
              const F_SEAL_ALL = 1 | 2 | 4 | 8 | 16;
              const want = Number(BigInt.asUintN(32, a2));
              if (want & ~F_SEAL_ALL) {
                ret = -BigInt(EINVAL);
                break;
              }
              if (sealEntry?.type !== "native" || !sealEntry.sealable) {
                ret = -BigInt(EPERM);
                break;
              }
              if (!this.fdAccess(fd).writable) {
                ret = -BigInt(EPERM);
                break;
              }
              const have = this.fileSeals.get(sealEntry.ofdId) ?? 0;
              if (have & 1) {
                ret = -BigInt(EPERM);
                break;
              }
              this.fileSeals.set(sealEntry.ofdId, have | want);
              ret = 0n;
              break;
            }
            case 1034: {
              const getSealEntry = this.getFdEntry(fd);
              if (getSealEntry?.type !== "native" || !getSealEntry.sealable) {
                ret = -BigInt(EINVAL);
                break;
              }
              ret = BigInt(this.fileSeals.get(getSealEntry.ofdId) ?? 0);
              break;
            }
            case 1032: {
              const cap = this.pipeCapacityHook?.(fd) ?? null;
              if (cap === null) {
                ret = -BigInt(EINVAL);
                break;
              }
              ret = BigInt(cap);
              break;
            }
            case 1031: {
              const cap = this.pipeCapacityHook?.(fd) ?? null;
              if (cap === null) {
                ret = -BigInt(EINVAL);
                break;
              }
              const want = Number(BigInt.asUintN(32, a2));
              if (want > 2147483648) {
                ret = -BigInt(EINVAL);
                break;
              }
              let rounded = 4096;
              while (rounded < want) rounded *= 2;
              const max = this.pipeMaxHook?.get() ?? cap;
              if (rounded > max && !this.hasCap(CAP_SYS_RESOURCE)) {
                ret = -BigInt(EPERM);
                break;
              }
              const granted = this.pipeResizeHook?.(fd, Math.min(rounded, max));
              if (granted === void 0) {
                ret = -BigInt(EINVAL);
                break;
              }
              ret = granted < 0 ? -BigInt(-granted) : BigInt(granted);
              break;
            }
            default:
              ret = -BigInt(EINVAL);
          }
          break;
        }
        // ── Misc ──
        case 278: {
          const GRND_NONBLOCK = 1, GRND_RANDOM = 2, GRND_INSECURE = 4;
          if (Number(BigInt.asIntN(32, a2)) & ~(GRND_NONBLOCK | GRND_RANDOM | GRND_INSECURE)) {
            ret = -BigInt(EINVAL);
            break;
          }
          const len = Number(a1);
          if (len < 0) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (len === 0) {
            ret = 0n;
            break;
          }
          if (this.ptrWriteFaults(bytes, Number(a0), len)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const buf = Number(a0) + GUEST_MEM_BASE;
          for (let i = 0; i < len; i++) {
            bytes[buf + i] = Math.random() * 256 | 0;
          }
          ret = BigInt(len);
          break;
        }
        case 98: {
          const futexAddr = Number(a0) + GUEST_MEM_BASE;
          const futexOp = Number(a1) & 127;
          const futexVal = Number(a2);
          switch (futexOp) {
            case 0:
            // FUTEX_WAIT
            case 9:
              ret = -BigInt(11);
              break;
            case 1:
            // FUTEX_WAKE
            case 10:
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
          const pR = this.ruid, pE = this.euid, pS = this.suid;
          if (this.euid === 0) {
            this.ruid = this.euid = this.suid = uid;
            this.publishCreds();
            this.recomputeCapsAfterUid(pR, pE, pS);
            ret = 0n;
          } else if (uid === this.ruid || uid === this.suid) {
            this.euid = uid;
            this.publishCreds();
            this.recomputeCapsAfterUid(pR, pE, pS);
            ret = 0n;
          } else ret = -BigInt(EPERM);
          break;
        }
        case 144: {
          const gid = Number(BigInt.asIntN(32, a0));
          if (this.euid === 0) {
            this.rgid = this.egid = this.sgid = gid;
          } else if (gid === this.rgid || gid === this.sgid) {
            this.egid = gid;
          } else {
            ret = -BigInt(EPERM);
            break;
          }
          this.publishCreds();
          ret = 0n;
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
          const preR = this.ruid, preE = this.euid, preS = this.suid;
          if (r !== -1 && r >>> 0 !== unchanged) this.ruid = r;
          if (e !== -1 && e >>> 0 !== unchanged) this.euid = e;
          if (r !== -1 && r >>> 0 !== unchanged || e !== -1 && e >>> 0 !== unchanged && e !== preR) this.suid = this.euid;
          this.publishCreds();
          this.recomputeCapsAfterUid(preR, preE, preS);
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
          const p3R = this.ruid, p3E = this.euid, p3S = this.suid;
          if (set(r)) this.ruid = r;
          if (set(e)) this.euid = e;
          if (set(s)) this.suid = s;
          this.publishCreds();
          this.recomputeCapsAfterUid(p3R, p3E, p3S);
          ret = 0n;
          break;
        }
        case 143: {
          const set143 = (v) => v !== -1 && v >>> 0 !== 4294967295;
          const r = Number(BigInt.asIntN(32, a0)), e = Number(BigInt.asIntN(32, a1));
          if (this.euid !== 0) {
            if (set143(r) && r !== this.rgid && r !== this.egid) {
              ret = -BigInt(EPERM);
              break;
            }
            if (set143(e) && e !== this.rgid && e !== this.egid && e !== this.sgid) {
              ret = -BigInt(EPERM);
              break;
            }
          }
          const preRgid = this.rgid;
          if (set143(r)) this.rgid = r;
          if (set143(e)) this.egid = e;
          if (set143(r) || set143(e) && e !== preRgid) this.sgid = this.egid;
          this.publishCreds();
          ret = 0n;
          break;
        }
        case 149: {
          const set = (v) => v !== -1 && v >>> 0 !== 4294967295;
          const r = Number(BigInt.asIntN(32, a0)), e = Number(BigInt.asIntN(32, a1)), s = Number(BigInt.asIntN(32, a2));
          if (this.euid !== 0) {
            const allowed = (v) => v === this.rgid || v === this.egid || v === this.sgid;
            if (set(r) && !allowed(r) || set(e) && !allowed(e) || set(s) && !allowed(s)) {
              ret = -BigInt(EPERM);
              break;
            }
          }
          if (set(r)) this.rgid = r;
          if (set(e)) this.egid = e;
          if (set(s)) this.sgid = s;
          this.publishCreds();
          ret = 0n;
          break;
        }
        // setfsuid / setfsgid: the identity the kernel uses for FILESYSTEM access, separate from the
        // effective one so a server can act as a client for file checks without becoming that client
        // for signals. these used to answer with the euid and change nothing, so setfsuid(-1) -- the
        // query form -- reported the wrong identity and setting one had no effect at all.
        //
        // neither ever reports an error: an unpermitted change is silently ignored, and the caller
        // learns what happened by calling again with -1 and comparing.
        case 151: {
          const prevFsuid = this.fsuid;
          const wantUid = Number(BigInt.asUintN(32, a0));
          if (wantUid !== 4294967295 && (this.euid === 0 || wantUid === this.ruid || wantUid === this.euid || wantUid === this.suid || wantUid === this.fsuid)) {
            this.fsuid = wantUid;
          }
          ret = BigInt(prevFsuid);
          break;
        }
        case 152: {
          const prevFsgid = this.fsgid;
          const wantGid = Number(BigInt.asUintN(32, a0));
          if (wantGid !== 4294967295 && (this.euid === 0 || wantGid === this.rgid || wantGid === this.egid || wantGid === this.sgid || wantGid === this.fsgid)) {
            this.fsgid = wantGid;
          }
          ret = BigInt(prevFsgid);
          break;
        }
        case 159: {
          const sgSize = Number(BigInt.asIntN(32, a0));
          if (sgSize < 0 || sgSize > 65536) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (!this.hasCap(CAP_SETGID)) {
            ret = -BigInt(EPERM);
            break;
          }
          if (sgSize > 0 && this.ptrFaults(bytes, Number(a1), sgSize * 4)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const next = [];
          for (let i = 0; i < sgSize; i++) {
            next.push(view.getUint32(Number(a1) + GUEST_MEM_BASE + i * 4, true));
          }
          this.groups = next;
          ret = 0n;
          break;
        }
        // ── Networking (via Swift bridge) ──
        case 198: {
          const domain = Number(a0);
          const sockType = Number(a1) & 255;
          const protocol = Number(a2);
          const sockErr = sockTripleErrno(domain, Number(BigInt.asIntN(32, a1)), protocol, this.euid);
          if (sockErr !== 0) {
            ret = -BigInt(sockErr);
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
                this.fdTable.set(socketFd, { type: "socket", socketId: resp.ret, domain, sockType, protocol });
                if (Number(a1) & 524288) this.fdCloexec.add(socketFd);
                if (Number(a1) & 2048) this.fdFlags.set(socketFd, (this.fdFlags.get(socketFd) ?? 0) | O_NONBLOCK);
                this.nt(`socket dom=${domain} type=${sockType} -> fd=${socketFd}`);
                this._netActive = true;
                cpu2.setReg(0, BigInt(socketFd));
              }
            }
          };
        }
        case 200: {
          const bindSock = this.sockFd(Number(a0));
          if ("err" in bindSock) {
            ret = -BigInt(bindSock.err);
            break;
          }
          if (bindSock.e.type !== "socket") {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          const bindEntry = bindSock.e;
          const bindLen = Number(BigInt.asIntN(32, a2));
          if (bindLen < 2 || bindLen > 128) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (bindEntry.domain === 2 && bindLen < 16) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (bindEntry.domain === 10 && bindLen < 28) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (this.ptrFaults(bytes, Number(a1), bindLen)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const addrBuf = this.parseSockaddr(bytes, Number(a1), bindLen);
          if (addrBuf.family !== bindEntry.domain && !(bindEntry.domain === 10 && addrBuf.family === 2)) {
            ret = -BigInt(EAFNOSUPPORT);
            break;
          }
          if (addrBuf.port > 0 && addrBuf.port < 1024 && !this.hasCap(10)) {
            ret = -BigInt(EACCES);
            break;
          }
          return {
            status: "bridge",
            method: "net_bind",
            args: { socketId: bindEntry.socketId, ...addrBuf },
            apply: (resp) => {
              if (resp.ret >= 0) {
                const bound = resp._meta ? resp._meta[0] | 0 : 0;
                bindEntry.localAddr = { ...addrBuf, port: bound || addrBuf.port };
              }
              cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            }
          };
        }
        case 201: {
          const listenSock = this.sockFd(Number(a0));
          if ("err" in listenSock) {
            ret = -BigInt(listenSock.err);
            break;
          }
          if (listenSock.e.type !== "socket") {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          const listenEntry = listenSock.e;
          if (listenEntry.sockType !== 1 && listenEntry.sockType !== 5) {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          return {
            status: "bridge",
            method: "net_listen",
            args: { socketId: listenEntry.socketId, backlog: Number(a1) },
            apply: (resp) => {
              if (resp.ret >= 0) listenEntry.listening = true;
              cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            }
          };
        }
        case 202:
        // accept(sockfd, addr, addrlen)
        case 242: {
          const acceptEntrySock = this.sockFd(Number(a0));
          if ("err" in acceptEntrySock) {
            ret = -BigInt(acceptEntrySock.err);
            break;
          }
          if (acceptEntrySock.e.type !== "socket") {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          const acceptEntry = acceptEntrySock.e;
          const acFlags = nr === 242 ? Number(BigInt.asIntN(32, a3)) : 0;
          if (acFlags & ~(2048 | 524288)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (acceptEntry.sockType !== 1 && acceptEntry.sockType !== 5) {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          if (!acceptEntry.listening) {
            ret = -BigInt(EINVAL);
            break;
          }
          const acAddrPtr = Number(a1);
          const acLenPtr = Number(a2);
          const newFd = this.allocFd();
          return {
            status: "bridge",
            method: "net_accept",
            args: { socketId: acceptEntry.socketId },
            apply: (resp) => {
              if (resp.ret === -11 && !this.isNonBlockingFd(Number(a0))) {
                this.retryAfterBridge = true;
                return;
              }
              if (resp.ret < 0) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                return;
              }
              const peer = this.peerFromMeta(resp);
              this.fdTable.set(newFd, {
                type: "socket",
                socketId: resp.ret,
                domain: acceptEntry.domain,
                sockType: acceptEntry.sockType,
                remoteAddr: peer ?? void 0,
                localAddr: acceptEntry.localAddr
              });
              if (acFlags & 524288) this.fdCloexec.add(newFd);
              if (acFlags & 2048) this.fdFlags.set(newFd, (this.fdFlags.get(newFd) ?? 0) | O_NONBLOCK);
              if (acAddrPtr !== 0 && peer) {
                const e = this.putSockaddr(bytes, view, acAddrPtr, acLenPtr, peer);
                if (e < 0) {
                  this.closeSocketHook?.(resp.ret);
                  this.fdTable.delete(newFd);
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(e)));
                  return;
                }
              }
              cpu2.setReg(0, BigInt(newFd));
            }
          };
        }
        case 203: {
          const connEntrySock = this.sockFd(Number(a0));
          if ("err" in connEntrySock) {
            ret = -BigInt(connEntrySock.err);
            break;
          }
          if (connEntrySock.e.type !== "socket") {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          const connEntry = connEntrySock.e;
          const connLen = Number(BigInt.asIntN(32, a2));
          if (connLen < 2 || connLen > 128) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (Number(a1) === 0 || this.ptrFaults(bytes, Number(a1), connLen)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const connFamily = bytes[Number(a1) + GUEST_MEM_BASE] | bytes[Number(a1) + GUEST_MEM_BASE + 1] << 8;
          if (connFamily === 0) {
            connEntry.remoteAddr = void 0;
            ret = 0n;
            break;
          }
          if (connFamily !== connEntry.domain && !(connEntry.domain === 10 && connFamily === 2)) {
            ret = -BigInt(EAFNOSUPPORT);
            break;
          }
          if (connFamily === 2 && connLen < 16) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (connFamily === 10 && connLen < 28) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (connEntry.sockType === 1 && connEntry.remoteAddr) {
            ret = -BigInt(EISCONN);
            break;
          }
          const connAddr = this.parseSockaddr(bytes, Number(a1), Number(a2));
          if (connAddr.host === "0.0.0.0") connAddr.host = "127.0.0.1";
          if (connAddr.host === "::") connAddr.host = "::1";
          return {
            status: "bridge",
            method: "net_connect",
            args: { socketId: connEntry.socketId, ...connAddr },
            apply: (resp) => {
              this.nt(`connect fd=${Number(a0)} ${connAddr.host}:${connAddr.port} -> ${resp.ret}`);
              if (resp.ret >= 0) connEntry.remoteAddr = connAddr;
              cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            }
          };
        }
        case 204: {
          const gsnEntrySock = this.sockFd(Number(a0));
          if ("err" in gsnEntrySock) {
            ret = -BigInt(gsnEntrySock.err);
            break;
          }
          const gsnEntry = gsnEntrySock.e;
          const gsnLenPtr = Number(a2);
          if (gsnLenPtr === 0 || this.ptrWriteFaults(bytes, gsnLenPtr, 4)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const gsnWant = view.getInt32(gsnLenPtr + GUEST_MEM_BASE, true);
          if (gsnWant < 0) {
            ret = -BigInt(EINVAL);
            break;
          }
          const family = gsnEntry.type === "socket" ? gsnEntry.domain : 1;
          const gsnAddr = gsnEntry.type === "socket" ? gsnEntry.localAddr ?? { family, host: family === 10 ? "::" : "0.0.0.0", port: 0 } : { family: 1, host: "", port: 0 };
          const gsnErr = this.putSockaddr(
            bytes,
            view,
            Number(a1),
            gsnLenPtr,
            { ...gsnAddr, family }
          );
          if (gsnErr < 0) {
            ret = -BigInt(-gsnErr);
            break;
          }
          ret = 0n;
          break;
        }
        case 205: {
          const gpnSock = this.sockFd(Number(a0));
          if ("err" in gpnSock) {
            ret = -BigInt(gpnSock.err);
            break;
          }
          const gpnLenPtr = Number(a2);
          if (gpnLenPtr === 0 || this.ptrWriteFaults(bytes, gpnLenPtr, 4)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const gpnWant = view.getInt32(gpnLenPtr + GUEST_MEM_BASE, true);
          if (gpnWant < 0) {
            ret = -BigInt(EINVAL);
            break;
          }
          const gpnUnix = gpnSock.e.type === "unixsock";
          const gpnPeer = gpnUnix ? this.peerCredHook?.(Number(a0)) ? { family: 1, host: "", port: 0 } : void 0 : gpnSock.e.type === "socket" ? gpnSock.e.remoteAddr : void 0;
          if (!gpnPeer) {
            ret = -107n;
            break;
          }
          const gpnErr = this.putSockaddr(bytes, view, Number(a1), gpnLenPtr, gpnPeer);
          if (gpnErr < 0) {
            ret = -BigInt(-gpnErr);
            break;
          }
          ret = 0n;
          break;
        }
        case 206: {
          const stEntrySock = this.sockFd(Number(a0));
          if ("err" in stEntrySock) {
            ret = -BigInt(stEntrySock.err);
            break;
          }
          if (stEntrySock.e.type !== "socket") {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          const stEntry = stEntrySock.e;
          const sendBuf = Number(a1);
          const sendLen = Number(BigInt.asUintN(64, a2));
          const sendFlags = Number(BigInt.asIntN(32, a3));
          if (sendFlags & 1 && stEntry.sockType !== 1) {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          if (sendLen > 0 && this.ptrFaults(bytes, sendBuf, sendLen)) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (stEntry.sockType === 2 && sendLen > (stEntry.domain === 10 ? 65527 : 65507)) {
            ret = -BigInt(EMSGSIZE);
            break;
          }
          if (stEntry.sockType === 1 && (!stEntry.remoteAddr || stEntry.shutWr)) {
            if (!(sendFlags & 16384)) this.raiseSignal(13);
            ret = -BigInt(EPIPE);
            break;
          }
          const sendData = bytes.slice(sendBuf + GUEST_MEM_BASE, sendBuf + GUEST_MEM_BASE + sendLen);
          const b64 = bytesToBase64(sendData);
          const destAddrPtr = Number(a4);
          const destAddrLen = Number(BigInt.asIntN(32, a5));
          const stConnectedStream = stEntry.sockType === 1 && !!stEntry.remoteAddr;
          if (!stConnectedStream && (destAddrPtr !== 0 || destAddrLen !== 0)) {
            if (destAddrLen < 2 || destAddrLen > 128) {
              ret = -BigInt(EINVAL);
              break;
            }
            if (destAddrPtr === 0 || this.ptrFaults(bytes, destAddrPtr, destAddrLen)) {
              ret = -BigInt(EFAULT);
              break;
            }
            const destFamily = bytes[destAddrPtr + GUEST_MEM_BASE] | bytes[destAddrPtr + GUEST_MEM_BASE + 1] << 8;
            if (destFamily === 2 && destAddrLen < 16) {
              ret = -BigInt(EINVAL);
              break;
            }
            if (destFamily === 10 && destAddrLen < 28) {
              ret = -BigInt(EINVAL);
              break;
            }
            if (destFamily !== stEntry.domain && !(stEntry.domain === 10 && destFamily === 2)) {
              ret = -BigInt(EAFNOSUPPORT);
              break;
            }
          }
          if (!stConnectedStream && destAddrPtr !== 0 && destAddrLen > 0) {
            const dest = this.parseSockaddr(bytes, destAddrPtr, destAddrLen);
            if (!stEntry.remoteAddr) stEntry.remoteAddr = dest;
            return {
              status: "bridge",
              method: "net_sendto",
              args: {
                socketId: stEntry.socketId,
                data: b64,
                host: dest.host,
                port: dest.port,
                more: sendFlags & 32768 ? 1 : 0
              },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          return {
            status: "bridge",
            method: "net_write",
            args: { socketId: stEntry.socketId, data: b64, more: sendFlags & 32768 ? 1 : 0 },
            apply: (resp) => {
              cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            }
          };
        }
        case 207: {
          const rfEntrySock = this.sockFd(Number(a0));
          if ("err" in rfEntrySock) {
            ret = -BigInt(rfEntrySock.err);
            break;
          }
          if (rfEntrySock.e.type !== "socket") {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          const rfEntry = rfEntrySock.e;
          const recvBuf = Number(a1);
          const recvLen = Math.min(Number(a2), BRIDGE_READ_MAX);
          const rfFlags = Number(BigInt.asIntN(32, a3));
          if (rfFlags & 1) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (rfFlags & 8192) {
            ret = -BigInt(EAGAIN);
            break;
          }
          if (recvLen > 0 && this.ptrWriteFaults(bytes, recvBuf, recvLen)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const rfAddrPtr = Number(a4);
          const rfLenPtr = Number(a5);
          if (rfLenPtr !== 0) {
            if (this.ptrWriteFaults(bytes, rfLenPtr, 4)) {
              ret = -BigInt(EFAULT);
              break;
            }
            if (view.getInt32(rfLenPtr + GUEST_MEM_BASE, true) < 0) {
              ret = -BigInt(EINVAL);
              break;
            }
          }
          if (rfEntry.shutRd) {
            ret = 0n;
            break;
          }
          const rfWantFrom = rfEntry.sockType === 2;
          const rfNonblock = this.isNonBlockingFd(Number(a0)) || (rfFlags & 64) !== 0;
          return {
            status: "bridge",
            method: "net_read",
            args: {
              socketId: rfEntry.socketId,
              count: recvLen,
              nonblock: rfNonblock ? 1 : 0,
              peek: rfFlags & 2 ? 1 : 0
            },
            apply: (resp) => {
              if (resp.ret === -11 && !rfNonblock) {
                this.retryAfterBridge = true;
                return;
              }
              if (resp.ret < 0) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                return;
              }
              const from = this.peerFromMeta(resp) ?? rfEntry.remoteAddr;
              if (rfLenPtr !== 0) {
                if (rfWantFrom && rfAddrPtr !== 0 && from) {
                  const e = this.putSockaddr(bytes, view, rfAddrPtr, rfLenPtr, from);
                  if (e < 0) {
                    cpu2.setReg(0, BigInt.asUintN(64, BigInt(e)));
                    return;
                  }
                } else {
                  view.setUint32(rfLenPtr + GUEST_MEM_BASE, 0, true);
                }
              }
              if (resp.bytes) {
                bytes.set(resp.bytes, recvBuf + GUEST_MEM_BASE);
                cpu2.setReg(0, BigInt(resp.bytes.length));
              } else {
                cpu2.setReg(0, BigInt(resp.ret));
              }
            }
          };
        }
        // setsockopt / getsockopt. these used to answer 0 to everything and write a zero back,
        // which meant a program could not tell a supported option from one this kernel has never
        // heard of -- and every wrong call looked like it worked.
        case 208: {
          const soSock = this.sockFd(Number(a0));
          if ("err" in soSock) {
            ret = -BigInt(soSock.err);
            break;
          }
          const soLevel = Number(BigInt.asIntN(32, a1));
          const soName = Number(BigInt.asIntN(32, a2));
          const soLen = Number(BigInt.asIntN(32, a4));
          if (!sockOptKnown(soLevel, soName)) {
            ret = -BigInt(ENOPROTOOPT);
            break;
          }
          if (soLen < 4) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (Number(a3) === 0 || this.ptrFaults(bytes, Number(a3), soLen)) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (soLevel === 1 && (soName === 3 || soName === 30 || soName === 38 || soName === 39)) {
            ret = -BigInt(ENOPROTOOPT);
            break;
          }
          const IP_ADD_MEMBERSHIP = 35, IP_DROP_MEMBERSHIP = 36;
          if (soLevel === 0 && (soName === IP_ADD_MEMBERSHIP || soName === IP_DROP_MEMBERSHIP)) {
            if (soLen < 8) {
              ret = -BigInt(EINVAL);
              break;
            }
            const b = Number(a3) + GUEST_MEM_BASE;
            const group = view.getUint32(b, false);
            const iface = view.getUint32(b + 4, false);
            const key = `${group}:${iface}`;
            if (!soSock.e.mcast) soSock.e.mcast = /* @__PURE__ */ new Set();
            if (soName === IP_ADD_MEMBERSHIP) {
              if (soSock.e.mcast.has(key)) {
                ret = -BigInt(EADDRINUSE);
                break;
              }
              soSock.e.mcast.add(key);
            } else {
              if (!soSock.e.mcast.delete(key)) {
                ret = -BigInt(EADDRNOTAVAIL);
                break;
              }
            }
            ret = 0n;
            break;
          }
          if (!soSock.e.opts) soSock.e.opts = /* @__PURE__ */ new Map();
          soSock.e.opts.set(soLevel * 65536 + soName, view.getInt32(Number(a3) + GUEST_MEM_BASE, true));
          ret = 0n;
          break;
        }
        case 209: {
          const goSock = this.sockFd(Number(a0));
          if ("err" in goSock) {
            ret = -BigInt(goSock.err);
            break;
          }
          const goLevel = Number(BigInt.asIntN(32, a1));
          const goName = Number(BigInt.asIntN(32, a2));
          if (!sockOptLevel(goLevel)) {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          if (!sockOptKnown(goLevel, goName)) {
            ret = -BigInt(ENOPROTOOPT);
            break;
          }
          if (Number(a4) === 0 || this.ptrWriteFaults(bytes, Number(a4), 4)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const goLen = view.getInt32(Number(a4) + GUEST_MEM_BASE, true);
          if (goLen < 0) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (Number(a3) === 0 || this.ptrWriteFaults(bytes, Number(a3), Math.min(Math.max(goLen, 1), 4))) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (goLevel === 1 && goName === 17) {
            if (goLen < 12) {
              ret = -BigInt(EINVAL);
              break;
            }
            if (this.ptrWriteFaults(bytes, Number(a3), 12)) {
              ret = -BigInt(EFAULT);
              break;
            }
            const peer = this.peerCredHook?.(Number(a0)) ?? null;
            view.setInt32(Number(a3) + GUEST_MEM_BASE, peer?.pid ?? 0, true);
            view.setUint32(Number(a3) + GUEST_MEM_BASE + 4, peer?.uid ?? 0, true);
            view.setUint32(Number(a3) + GUEST_MEM_BASE + 8, peer?.gid ?? 0, true);
            view.setInt32(Number(a4) + GUEST_MEM_BASE, 12, true);
            ret = 0n;
            break;
          }
          const e = goSock.e;
          let value;
          if (goLevel === 1 && goName === 3) value = e.sockType ?? 1;
          else if (goLevel === 1 && goName === 4) value = 0;
          else if (goLevel === 1 && goName === 30) value = e.listening ? 1 : 0;
          else if (goLevel === 1 && goName === 38) value = (e.type === "socket" ? e.protocol : 0) ?? 0;
          else if (goLevel === 1 && goName === 39) value = e.type === "socket" ? e.domain : 1;
          else value = e.opts?.get(goLevel * 65536 + goName) ?? 0;
          view.setInt32(Number(a3) + GUEST_MEM_BASE, value, true);
          view.setInt32(Number(a4) + GUEST_MEM_BASE, 4, true);
          ret = 0n;
          break;
        }
        // ── Process identity/groups ──
        case 129: {
          const killPid = Number(BigInt.asIntN(32, a0));
          const killSig = Number(a1);
          if (killSig < 0 || killSig > 64) {
            ret = -BigInt(EINVAL);
            break;
          }
          const pidExists = (p) => p === this.currentPid || p === 1 || p === this.currentPpid || (this.isLivePidFn?.(p) ?? false);
          if (killPid > 0) {
            if (!pidExists(killPid)) {
              ret = -BigInt(ESRCH);
              break;
            }
            if (killSig !== 0 && killPid === this.currentPid) this.raiseSignal(killSig);
            ret = 0n;
          } else if (killPid === 0 || killPid === -1) {
            if (killSig !== 0) this.raiseSignal(killSig);
            ret = 0n;
          } else {
            ret = -BigInt(ESRCH);
          }
          break;
        }
        case 166: {
          const oldMask = this.umask;
          this.umask = Number(a0) & 511;
          ret = BigInt(oldMask);
          break;
        }
        case 158: {
          const gSize = Number(BigInt.asIntN(32, a0));
          if (gSize < 0) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (gSize === 0) {
            ret = BigInt(this.groups.length);
            break;
          }
          if (gSize < this.groups.length) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (this.ptrWriteFaults(bytes, Number(a1), this.groups.length * 4)) {
            ret = -BigInt(EFAULT);
            break;
          }
          for (let i = 0; i < this.groups.length; i++) {
            view.setUint32(Number(a1) + GUEST_MEM_BASE + i * 4, this.groups[i] >>> 0, true);
          }
          ret = BigInt(this.groups.length);
          break;
        }
        case 115: {
          ret = 0n;
          break;
        }
        case 261: {
          const resource = Number(BigInt.asIntN(32, a1));
          if (resource < 0 || resource >= RLIM_NLIMITS) {
            ret = -BigInt(EINVAL);
            break;
          }
          const rlPid = Number(BigInt.asIntN(32, a0));
          if (rlPid !== 0 && rlPid !== this.currentPid) {
            ret = -BigInt(ESRCH);
            break;
          }
          const rlNewPtr = Number(a2);
          const rlOldPtr = Number(a3);
          if (rlOldPtr !== 0 && this.ptrFaults(bytes, rlOldPtr, 16)) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (rlNewPtr !== 0 && this.ptrFaults(bytes, rlNewPtr, 16)) {
            ret = -BigInt(EFAULT);
            break;
          }
          let rlNewCur = 0n;
          if (rlNewPtr !== 0) {
            rlNewCur = view.getBigUint64(rlNewPtr + GUEST_MEM_BASE, true);
            const newMax = view.getBigUint64(rlNewPtr + GUEST_MEM_BASE + 8, true);
            if (rlNewCur > newMax) {
              ret = -BigInt(EINVAL);
              break;
            }
            if (resource === RLIMIT_NOFILE && newMax > BigInt(NR_OPEN)) {
              ret = -BigInt(EPERM);
              break;
            }
            if (newMax > this.rlimitOf(resource).max && !this.hasCap(CAP_SYS_RESOURCE)) {
              ret = -BigInt(EPERM);
              break;
            }
          }
          if (rlOldPtr !== 0) {
            const lim = this.rlimitOf(resource);
            view.setBigUint64(rlOldPtr + GUEST_MEM_BASE, lim.cur, true);
            view.setBigUint64(rlOldPtr + GUEST_MEM_BASE + 8, lim.max, true);
          }
          if (rlNewPtr !== 0) this.setRlimit(resource, rlNewCur, view.getBigUint64(rlNewPtr + GUEST_MEM_BASE + 8, true));
          ret = 0n;
          break;
        }
        case 233: {
          const advice = Number(BigInt.asIntN(32, a2));
          const mvStart = Number(a0);
          const mvLen = Number(BigInt.asIntN(64, a1));
          if (mvStart & 4095) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (mvLen < 0) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (!MADV_VALID.has(advice)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if ((advice === 100 || advice === 101) && !this.hasCap(CAP_SYS_ADMIN)) {
            ret = -BigInt(EPERM);
            break;
          }
          if (mvLen === 0) {
            ret = 0n;
            break;
          }
          const mvEnd = mvStart + mvLen + 4095 & ~4095;
          if (mvEnd <= mvStart) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (!this.rangeMapped(mvStart, mvEnd)) {
            ret = -BigInt(ENOMEM);
            break;
          }
          const mvKind = this.rangeKind(mvStart, mvEnd);
          const privAnon = !mvKind.shared && !mvKind.fileBacked;
          let mvBad = false;
          switch (advice) {
            // MADV_DONTNEED refuses a locked range: dropping the pages would unlock it.
            case 4:
              mvBad = mvKind.locked;
              break;
            // MADV_FREE and MADV_WIPEONFORK are defined only on private anonymous memory.
            case 8:
            case 18:
              mvBad = !privAnon;
              break;
            // MADV_REMOVE punches a hole, which only shared memory with no file behind it has.
            case 9:
              mvBad = !mvKind.shared || mvKind.fileBacked;
              break;
            // KSM merges anonymous private pages and nothing else.
            case 12:
            case 13:
              mvBad = !privAnon;
              break;
            default:
              break;
          }
          if (mvBad) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (advice === 4 || advice === 8) {
            if (mvEnd + GUEST_MEM_BASE <= memory2.buffer.byteLength) {
              new Uint8Array(memory2.buffer).fill(0, mvStart + GUEST_MEM_BASE, mvEnd + GUEST_MEM_BASE);
            }
          }
          ret = 0n;
          break;
        }
        // ── Missing syscalls (spec-compliant implementations) ──
        case 50: {
          const fchEntry = this.getFdEntry(Number(a0));
          const fchPath = fchEntry?.path;
          if (typeof fchPath !== "string" || !fchPath.length) {
            ret = -BigInt(EBADF);
            break;
          }
          if (fchEntry?.type === "native" && fchEntry.isDir === false) {
            ret = -BigInt(ENOTDIR);
            break;
          }
          const fchBridge = this.bridgePath(fchPath);
          if (fchBridge === null) {
            this.cwd = fchPath;
            if (this.vfs) this.vfs.setCwd(fchPath);
            ret = 0n;
            break;
          }
          return {
            status: "bridge",
            method: "fs_faccessat",
            args: { path: fchBridge, mode: 1, eaccess: true },
            apply: (resp) => {
              if (resp.ret < 0) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                return;
              }
              this.cwd = fchPath;
              if (this.vfs) this.vfs.setCwd(fchPath);
              cpu2.setReg(0, 0n);
            }
          };
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
          const RUSAGE_SELF = 0, RUSAGE_CHILDREN = -1, RUSAGE_THREAD = 1;
          const who = Number(BigInt.asIntN(32, a0));
          if (who !== RUSAGE_SELF && who !== RUSAGE_CHILDREN && who !== RUSAGE_THREAD) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (this.ptrFaults(bytes, Number(a1), 144)) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (who === RUSAGE_CHILDREN) {
            this.writeRusage(view, Number(a1), this.childUtimeMs, this.childStimeMs, this.childMaxRssKb);
          } else {
            this.writeRusage(view, Number(a1), this.ownUserMs(), this.ownSysMs(), this.rssKbHook ? this.rssKbHook() : 0);
          }
          ret = 0n;
          break;
        }
        case 167: {
          const prOption = Number(BigInt.asIntN(32, a0));
          const prArg2 = Number(BigInt.asIntN(64, a1));
          switch (prOption) {
            case 1:
              if (prArg2 < 0 || prArg2 > 64) {
                ret = -BigInt(EINVAL);
                break;
              }
              this.pdeathSig = prArg2;
              this.pdeathHook?.(prArg2);
              ret = 0n;
              break;
            case 2:
              if (this.ptrFaults(bytes, Number(a1), 4)) {
                ret = -BigInt(EFAULT);
                break;
              }
              view.setInt32(Number(a1) + GUEST_MEM_BASE, this.pdeathSig, true);
              ret = 0n;
              break;
            case 3:
              ret = BigInt(this.dumpable);
              break;
            case 4:
              if (prArg2 !== 0 && prArg2 !== 1) {
                ret = -BigInt(EINVAL);
                break;
              }
              this.dumpable = prArg2;
              ret = 0n;
              break;
            case 7:
              ret = BigInt(this.keepCaps);
              break;
            case 8:
              if (prArg2 !== 0 && prArg2 !== 1) {
                ret = -BigInt(EINVAL);
                break;
              }
              this.keepCaps = prArg2;
              ret = 0n;
              break;
            case 15: {
              if (this.ptrFaults(bytes, Number(a1), 1)) {
                ret = -BigInt(EFAULT);
                break;
              }
              const raw = readGuestString(bytes, Number(a1));
              this.comm = raw.slice(0, 15);
              this.commHook?.(this.comm);
              ret = 0n;
              break;
            }
            case 16: {
              if (this.ptrFaults(bytes, Number(a1), 16)) {
                ret = -BigInt(EFAULT);
                break;
              }
              const out = new Uint8Array(16);
              const enc = new TextEncoder().encode(this.comm.slice(0, 15));
              out.set(enc.subarray(0, 15));
              new Uint8Array(memory2.buffer).set(out, Number(a1) + GUEST_MEM_BASE);
              ret = 0n;
              break;
            }
            case 21:
              ret = 0n;
              break;
            case 22:
              ret = -BigInt(EINVAL);
              break;
            case 23:
              if (prArg2 < 0 || prArg2 > 40) {
                ret = -BigInt(EINVAL);
                break;
              }
              ret = this.hasCap(prArg2) ? 1n : 0n;
              break;
            case 24:
              if (prArg2 < 0 || prArg2 > 40) {
                ret = -BigInt(EINVAL);
                break;
              }
              if (!this.hasCap(CAP_SETPCAP)) {
                ret = -BigInt(EPERM);
                break;
              }
              this.capPerm &= ~(1n << BigInt(prArg2));
              this.capEff &= ~(1n << BigInt(prArg2));
              ret = 0n;
              break;
            case 29:
              this.timerSlackNs = prArg2 === 0 ? this.defaultTimerSlackNs : prArg2;
              ret = 0n;
              break;
            case 30:
              ret = BigInt(this.timerSlackNs);
              break;
            case 36:
              this.childSubreaper = prArg2 !== 0 ? 1 : 0;
              this.subreaperHook?.set(this.childSubreaper);
              ret = 0n;
              break;
            case 37:
              if (this.ptrFaults(bytes, Number(a1), 4)) {
                ret = -BigInt(EFAULT);
                break;
              }
              view.setInt32(
                Number(a1) + GUEST_MEM_BASE,
                this.subreaperHook ? this.subreaperHook.get() : this.childSubreaper,
                true
              );
              ret = 0n;
              break;
            case 38:
              if (prArg2 !== 1) {
                ret = -BigInt(EINVAL);
                break;
              }
              if (Number(a2) !== 0 || Number(a3) !== 0 || Number(a4) !== 0) {
                ret = -BigInt(EINVAL);
                break;
              }
              this.noNewPrivs = 1;
              ret = 0n;
              break;
            case 39:
              if (prArg2 !== 0 || Number(a2) !== 0 || Number(a3) !== 0 || Number(a4) !== 0) {
                ret = -BigInt(EINVAL);
                break;
              }
              ret = BigInt(this.noNewPrivs);
              break;
            case 41:
              if (Number(a2) !== 0 || Number(a3) !== 0 || Number(a4) !== 0) {
                ret = -BigInt(EINVAL);
                break;
              }
              this.thpDisable = prArg2 !== 0 ? 1 : 0;
              ret = 0n;
              break;
            case 42:
              if (prArg2 !== 0 || Number(a2) !== 0 || Number(a3) !== 0 || Number(a4) !== 0) {
                ret = -BigInt(EINVAL);
                break;
              }
              ret = BigInt(this.thpDisable);
              break;
            // options that exist but describe hardware or kernel facilities this system does not
            // have. linux answers EINVAL for these on an arch that lacks them, and so do we.
            case 5:
            case 6:
            // PR_GET/SET_UNALIGN
            case 9:
            case 10:
            // PR_GET/SET_FPEMU
            case 11:
            case 12:
            // PR_GET/SET_FPEXC
            case 19:
            case 20:
            // PR_GET/SET_ENDIAN
            case 25:
            case 26:
            // PR_GET/SET_TSC
            case 43:
            case 44:
            // PR_MPX_ENABLE/DISABLE_MANAGEMENT
            case 45:
            case 46:
            // PR_SET/GET_FP_MODE
            case 50:
            case 51:
            // PR_SVE_SET/GET_VL
            case 63:
            case 64:
              ret = -BigInt(EINVAL);
              break;
            case 13:
              ret = 0n;
              break;
            case 14:
              ret = prArg2 === 0 ? 0n : -BigInt(EINVAL);
              break;
            case 27:
              ret = BigInt(this.secureBits);
              break;
            case 28:
              if (!this.hasCap(CAP_SETPCAP)) {
                ret = -BigInt(EPERM);
                break;
              }
              this.secureBits = prArg2;
              ret = 0n;
              break;
            case 31:
            case 32:
            // PR_TASK_PERF_EVENTS_DISABLE/ENABLE
            case 33:
            case 34:
            // PR_MCE_KILL / PR_MCE_KILL_GET
            case 35:
            // PR_SET_MM
            case 47:
            // PR_CAP_AMBIENT
            case 52:
            case 53:
            // PR_GET/SET_SPECULATION_CTRL
            case 54:
            // PR_PAC_RESET_KEYS
            case 55:
            case 56:
            // PR_SET/GET_TAGGED_ADDR_CTRL
            case 57:
            case 58:
            // PR_SET/GET_IO_FLUSHER
            case 59:
            // PR_SET_SYSCALL_USER_DISPATCH
            case 60:
            case 61:
            // PR_PAC_SET/GET_ENABLED_KEYS
            case 62:
            // PR_SCHED_CORE
            case 65:
            case 66:
            // PR_SET/GET_MDWE
            case 67:
            case 68:
              ret = -BigInt(EINVAL);
              break;
            case 40:
              if (this.ptrFaults(bytes, Number(a1), 8)) {
                ret = -BigInt(EFAULT);
                break;
              }
              view.setBigUint64(Number(a1) + GUEST_MEM_BASE, 0n, true);
              ret = 0n;
              break;
            case 1398164801:
              ret = 0n;
              break;
            case 1499557217:
              ret = -BigInt(EINVAL);
              break;
            default:
              ret = -BigInt(EINVAL);
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
        // these answered root, root, root no matter what the process had done. a program that drops
        // privilege and then checks that it worked -- which is the whole point of dropping it -- was
        // told it had not.
        case 148: {
          if (this.ptrFaults(bytes, Number(a0), 4) || this.ptrFaults(bytes, Number(a1), 4) || this.ptrFaults(bytes, Number(a2), 4)) {
            ret = -BigInt(EFAULT);
            break;
          }
          view.setUint32(Number(a0) + GUEST_MEM_BASE, this.ruid, true);
          view.setUint32(Number(a1) + GUEST_MEM_BASE, this.euid, true);
          view.setUint32(Number(a2) + GUEST_MEM_BASE, this.suid, true);
          ret = 0n;
          break;
        }
        case 150: {
          if (this.ptrFaults(bytes, Number(a0), 4) || this.ptrFaults(bytes, Number(a1), 4) || this.ptrFaults(bytes, Number(a2), 4)) {
            ret = -BigInt(EFAULT);
            break;
          }
          view.setUint32(Number(a0) + GUEST_MEM_BASE, this.rgid, true);
          view.setUint32(Number(a1) + GUEST_MEM_BASE, this.egid, true);
          view.setUint32(Number(a2) + GUEST_MEM_BASE, this.sgid, true);
          ret = 0n;
          break;
        }
        case 153: {
          if (Number(a0) !== 0) {
            if (this.ptrFaults(bytes, Number(a0), 32)) {
              ret = -BigInt(EFAULT);
              break;
            }
            const tmsBuf = Number(a0) + GUEST_MEM_BASE;
            view.setBigUint64(tmsBuf, BigInt(Math.floor(this.ownUserMs() / 10)), true);
            view.setBigUint64(tmsBuf + 8, BigInt(Math.floor(this.ownSysMs() / 10)), true);
            view.setBigUint64(tmsBuf + 16, BigInt(Math.floor(this.childUtimeMs / 10)), true);
            view.setBigUint64(tmsBuf + 24, BigInt(Math.floor(this.childStimeMs / 10)), true);
          }
          ret = BigInt(Math.floor(Number(this.monoNs() / 1000000n) / 10));
          break;
        }
        case 92: {
          const persNew = Number(BigInt.asUintN(32, a0));
          const persOld = this.personality;
          if (persNew !== 4294967295) this.personality = persNew;
          ret = BigInt(persOld);
          break;
        }
        // ── Timer syscalls ──
        case 102: {
          const which = Number(BigInt.asIntN(32, a0));
          if (which < 0 || which > 2) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (this.ptrWriteFaults(bytes, Number(a1), 32)) {
            ret = -BigInt(EFAULT);
            break;
          }
          this.checkTimers();
          const cur = this.timerRemaining(this.itimers[which]);
          this.writeItimer(view, Number(a1) + GUEST_MEM_BASE, cur.intervalNs, cur.valueNs, 1000000n);
          ret = 0n;
          break;
        }
        case 103: {
          const which = Number(BigInt.asIntN(32, a0));
          const newPtr = Number(a1), oldPtr = Number(a2);
          let want = { intervalNs: 0n, valueNs: 0n };
          if (newPtr !== 0) {
            if (this.ptrFaults(bytes, newPtr, 32)) {
              ret = -BigInt(EFAULT);
              break;
            }
            const got = this.readItimer(view, newPtr + GUEST_MEM_BASE, 1000000n);
            if (!got) {
              ret = -BigInt(EINVAL);
              break;
            }
            want = got;
          }
          if (which < 0 || which > 2) {
            ret = -BigInt(EINVAL);
            break;
          }
          this.checkTimers();
          const t = this.itimers[which];
          const prev = this.timerRemaining(t);
          t.intervalNs = which === 0 && want.valueNs === 0n ? 0n : want.intervalNs;
          t.nextNs = want.valueNs > 0n ? this.timerClockNs(t.clockid) + want.valueNs : 0n;
          if (oldPtr !== 0) {
            if (this.ptrWriteFaults(bytes, oldPtr, 32)) {
              ret = -BigInt(EFAULT);
              break;
            }
            this.writeItimer(view, oldPtr + GUEST_MEM_BASE, prev.intervalNs, prev.valueNs, 1000000n);
          }
          ret = 0n;
          break;
        }
        case 107: {
          const tcClk = Number(BigInt.asIntN(32, a0));
          if (!this.knownClock(tcClk)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (tcClk === 4 || tcClk === 5 || tcClk === 6) {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          let tcNotify = SIGEV_SIGNAL, tcSigno = SIGALRM, tcVal = BigInt(this.nextPosixTimerId);
          const tcEvp = Number(a1);
          if (tcEvp !== 0) {
            if (this.ptrFaults(bytes, tcEvp, 64)) {
              ret = -BigInt(EFAULT);
              break;
            }
            const eb = tcEvp + GUEST_MEM_BASE;
            tcVal = view.getBigUint64(eb, true);
            tcSigno = view.getInt32(eb + 8, true);
            tcNotify = view.getInt32(eb + 12, true);
            if (tcNotify === SIGEV_SIGNAL || tcNotify === SIGEV_THREAD || tcNotify === SIGEV_THREAD_ID) {
              if (tcSigno <= 0 || tcSigno > SIGRTMAX) {
                ret = -BigInt(EINVAL);
                break;
              }
            } else if (tcNotify !== SIGEV_NONE) {
              ret = -BigInt(EINVAL);
              break;
            }
          }
          if (this.ptrWriteFaults(bytes, Number(a2), 4)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const tcId = this.nextPosixTimerId++;
          this.posixTimers.set(tcId, {
            id: tcId,
            clockid: tcClk,
            nextNs: 0n,
            intervalNs: 0n,
            notify: tcNotify,
            signo: tcSigno,
            sigval: tcVal,
            overrun: 0,
            accrued: 0
          });
          view.setInt32(Number(a2) + GUEST_MEM_BASE, tcId, true);
          ret = 0n;
          break;
        }
        case 108: {
          const tgT = this.posixTimers.get(Number(BigInt.asIntN(32, a0)));
          if (!tgT) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (this.ptrWriteFaults(bytes, Number(a1), 32)) {
            ret = -BigInt(EFAULT);
            break;
          }
          this.checkTimers();
          const tgCur = this.timerRemaining(tgT);
          this.writeItimer(view, Number(a1) + GUEST_MEM_BASE, tgCur.intervalNs, tgCur.valueNs, 1000000000n);
          ret = 0n;
          break;
        }
        case 109: {
          const toT = this.posixTimers.get(Number(BigInt.asIntN(32, a0)));
          if (!toT) {
            ret = -BigInt(EINVAL);
            break;
          }
          ret = BigInt(toT.overrun);
          break;
        }
        case 110: {
          const tsNew = Number(a2), tsOld = Number(a3);
          if (tsNew === 0) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (this.ptrFaults(bytes, tsNew, 32)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const tsWant = this.readItimer(view, tsNew + GUEST_MEM_BASE, 1000000000n);
          if (!tsWant) {
            ret = -BigInt(EINVAL);
            break;
          }
          const tsFlags = Number(BigInt.asIntN(32, a1));
          if (tsFlags & ~TIMER_ABSTIME) {
            ret = -BigInt(EINVAL);
            break;
          }
          const tsT = this.posixTimers.get(Number(BigInt.asIntN(32, a0)));
          if (!tsT) {
            ret = -BigInt(EINVAL);
            break;
          }
          this.checkTimers();
          const tsPrev = this.timerRemaining(tsT);
          tsT.intervalNs = tsWant.intervalNs;
          tsT.nextNs = tsWant.valueNs === 0n ? 0n : tsFlags & TIMER_ABSTIME ? tsWant.valueNs : this.timerClockNs(tsT.clockid) + tsWant.valueNs;
          tsT.overrun = 0;
          tsT.accrued = 0;
          if (tsOld !== 0) {
            if (this.ptrWriteFaults(bytes, tsOld, 32)) {
              ret = -BigInt(EFAULT);
              break;
            }
            this.writeItimer(view, tsOld + GUEST_MEM_BASE, tsPrev.intervalNs, tsPrev.valueNs, 1000000000n);
          }
          ret = 0n;
          break;
        }
        case 111:
          ret = this.posixTimers.delete(Number(BigInt.asIntN(32, a0))) ? 0n : -BigInt(EINVAL);
          break;
        case 114: {
          const grClk = Number(BigInt.asIntN(32, a0));
          if (!this.knownClock(grClk)) {
            ret = -BigInt(EINVAL);
            break;
          }
          const resPtr = Number(a1);
          if (resPtr !== 0) {
            if (this.ptrFaults(bytes, resPtr, 16)) {
              ret = -BigInt(EFAULT);
              break;
            }
            const base = resPtr + GUEST_MEM_BASE;
            view.setBigUint64(base, 0n, true);
            view.setBigUint64(base + 8, this.clockResNs(grClk), true);
          }
          ret = 0n;
          break;
        }
        case 32:
          ret = 0n;
          break;
        // ── Scheduling ──
        // the sched_* family (118..123, 125..127, 274, 275) is handled by the worker, which can reach
        // the shared process table these calls read and write -- they name a pid rather than the
        // caller. they only arrive here on the host-side test path, where there is no table.
        case 118:
        case 119:
        case 120:
        case 121:
        case 122:
        case 123:
        case 125:
        case 126:
          ret = -BigInt(ENOSYS);
          break;
        // inotify (26/27/28) is handled by the worker, which can reach the kernel's watch registry.
        // it only arrives here on the host-side test path, where there is no kernel to ask.
        case 26:
        case 27:
        case 28:
          ret = -BigInt(ENOSYS);
          break;
        // ── timerfd ──
        // the same timer as timer_create, announced by making a descriptor readable instead of by
        // raising a signal. a read yields the number of expirations since the last read and clears
        // it, which is why the count has to be carried rather than a mere "it fired" flag.
        case 85: {
          const tfClk = Number(BigInt.asIntN(32, a0));
          const tfFlags = Number(BigInt.asIntN(32, a1));
          if (tfFlags & ~(TFD_CLOEXEC | TFD_NONBLOCK)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (tfClk !== CLOCK_REALTIME && tfClk !== CLOCK_MONOTONIC && tfClk !== CLOCK_BOOTTIME && tfClk !== CLOCK_REALTIME_ALARM && tfClk !== CLOCK_BOOTTIME_ALARM) {
            ret = -BigInt(EINVAL);
            break;
          }
          if ((tfClk === CLOCK_REALTIME_ALARM || tfClk === CLOCK_BOOTTIME_ALARM) && !this.hasCap(CAP_WAKE_ALARM)) {
            ret = -BigInt(EPERM);
            break;
          }
          const tfFd = this.allocFd();
          if (tfFd < 0) {
            ret = -BigInt(EMFILE);
            break;
          }
          const tfId = this.nextTimerFdId++;
          this.timerFds.set(tfId, { clockid: tfClk, nextNs: 0n, intervalNs: 0n, ticks: 0n });
          this.fdTable.set(tfFd, { type: "timerfd", id: tfId });
          if (tfFlags & TFD_CLOEXEC) this.fdCloexec.add(tfFd);
          this.fdFlags.set(tfFd, tfFlags & TFD_NONBLOCK);
          ret = BigInt(tfFd);
          break;
        }
        case 86: {
          const tsfNew = Number(a2), tsfOld = Number(a3);
          if (this.ptrFaults(bytes, tsfNew, 32)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const tsfWant = this.readItimer(view, tsfNew + GUEST_MEM_BASE, 1000000000n);
          if (!tsfWant) {
            ret = -BigInt(EINVAL);
            break;
          }
          const tsfFlags = Number(BigInt.asIntN(32, a1));
          if ((tsfFlags & ~(TFD_TIMER_ABSTIME | TFD_TIMER_CANCEL_ON_SET)) !== 0 || tsfFlags & TFD_TIMER_CANCEL_ON_SET && !(tsfFlags & TFD_TIMER_ABSTIME)) {
            ret = -BigInt(EINVAL);
            break;
          }
          const tsfT = this.timerFdOf(Number(BigInt.asIntN(32, a0)));
          if (typeof tsfT === "number") {
            ret = BigInt(tsfT);
            break;
          }
          this.checkTimers();
          const tsfPrev = this.timerRemaining(tsfT);
          tsfT.intervalNs = tsfWant.intervalNs;
          tsfT.nextNs = tsfWant.valueNs === 0n ? 0n : tsfFlags & TFD_TIMER_ABSTIME ? tsfWant.valueNs : this.timerClockNs(tsfT.clockid) + tsfWant.valueNs;
          tsfT.ticks = 0n;
          if (tsfOld !== 0) {
            if (this.ptrWriteFaults(bytes, tsfOld, 32)) {
              ret = -BigInt(EFAULT);
              break;
            }
            this.writeItimer(view, tsfOld + GUEST_MEM_BASE, tsfPrev.intervalNs, tsfPrev.valueNs, 1000000000n);
          }
          ret = 0n;
          break;
        }
        case 87: {
          const tgfT = this.timerFdOf(Number(BigInt.asIntN(32, a0)));
          if (typeof tgfT === "number") {
            ret = BigInt(tgfT);
            break;
          }
          if (this.ptrWriteFaults(bytes, Number(a1), 32)) {
            ret = -BigInt(EFAULT);
            break;
          }
          this.checkTimers();
          const tgfCur = this.timerRemaining(tgfT);
          this.writeItimer(view, Number(a1) + GUEST_MEM_BASE, tgfCur.intervalNs, tgfCur.valueNs, 1000000000n);
          ret = 0n;
          break;
        }
        // ── Misc process control ──
        case 97: {
          const CLONE_FILES = 1024, CLONE_FS = 512, CLONE_SYSVSEM = 262144;
          const CLONE_NEWNS = 131072, CLONE_NEWCGROUP = 33554432, CLONE_NEWUTS = 67108864;
          const CLONE_NEWIPC = 134217728, CLONE_NEWUSER = 268435456, CLONE_NEWPID = 536870912;
          const CLONE_NEWNET = 1073741824, CLONE_NEWTIME = 128;
          const NAMESPACES = CLONE_NEWNS | CLONE_NEWCGROUP | CLONE_NEWUTS | CLONE_NEWIPC | CLONE_NEWUSER | CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWTIME;
          const unFlags = Number(BigInt.asUintN(32, a0));
          if (unFlags & ~(CLONE_FILES | CLONE_FS | CLONE_SYSVSEM | NAMESPACES)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (unFlags & NAMESPACES && !this.hasCap(CAP_SYS_ADMIN)) {
            ret = -BigInt(EPERM);
            break;
          }
          if (unFlags & NAMESPACES) {
            ret = -BigInt(EINVAL);
            break;
          }
          ret = 0n;
          break;
        }
        case 168: {
          const cpuPtr = Number(a0);
          const nodePtr = Number(a1);
          if (cpuPtr !== 0 && this.ptrWriteFaults(bytes, cpuPtr, 4)) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (nodePtr !== 0 && this.ptrWriteFaults(bytes, nodePtr, 4)) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (cpuPtr !== 0) view.setUint32(cpuPtr + GUEST_MEM_BASE, 0, true);
          if (nodePtr !== 0) view.setUint32(nodePtr + GUEST_MEM_BASE, 0, true);
          ret = 0n;
          break;
        }
        // ── scheduling priority ─────────────────────────────────────────────────
        // these used to accept anything and always answer "nice 0". that is not a simplification:
        // setpriority reported success for a `which` that does not exist and for processes that do
        // not exist, and nothing a caller set was ever visible again.
        case 140: {
          const spWhich = Number(BigInt.asIntN(32, a0));
          const spWho = Number(BigInt.asIntN(32, a1));
          let spPrio = Number(BigInt.asIntN(32, a2));
          if (spWhich < PRIO_PROCESS || spWhich > PRIO_USER) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (spPrio < -20) spPrio = -20;
          if (spPrio > 19) spPrio = 19;
          const spTargets = this.priorityTargets(spWhich, spWho);
          if (spTargets.length === 0) {
            ret = -BigInt(ESRCH);
            break;
          }
          let spErr = 0;
          for (const t of spTargets) {
            if (this.euid !== 0 && t.ruid !== this.ruid && t.ruid !== this.euid) {
              spErr = EPERM;
              break;
            }
            if (spPrio < t.nice && !this.hasCap(CAP_SYS_NICE)) {
              spErr = EACCES;
              break;
            }
          }
          if (spErr !== 0) {
            ret = -BigInt(spErr);
            break;
          }
          for (const t of spTargets) this.prioHook?.setNice(t.pid, spPrio);
          ret = 0n;
          break;
        }
        case 141: {
          const gpWhich = Number(BigInt.asIntN(32, a0));
          const gpWho = Number(BigInt.asIntN(32, a1));
          if (gpWhich < PRIO_PROCESS || gpWhich > PRIO_USER) {
            ret = -BigInt(EINVAL);
            break;
          }
          const gpTargets = this.priorityTargets(gpWhich, gpWho);
          if (gpTargets.length === 0) {
            ret = -BigInt(ESRCH);
            break;
          }
          let best = 19;
          for (const t of gpTargets) if (t.nice < best) best = t.nice;
          ret = BigInt(20 - best);
          break;
        }
        // ── Memory control (per spec: accept, no-op for our in-memory VFS) ──
        case 228:
        // mlock(addr, len)
        case 284:
        // mlock2(addr, len, flags) -- mlock with MLOCK_ONFAULT
        case 229: {
          const MLOCK_ONFAULT = 1;
          if (nr === 284 && Number(BigInt.asIntN(32, a2)) & ~MLOCK_ONFAULT) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (nr !== 229 && this.memlockLimit === 0 && !this.hasCap(CAP_IPC_LOCK)) {
            ret = -BigInt(EPERM);
            break;
          }
          const lkLen = Number(BigInt.asIntN(64, a1));
          if (lkLen < 0) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (lkLen === 0) {
            ret = 0n;
            break;
          }
          const lkStart = Number(a0) & ~4095;
          const lkEnd = Number(a0) + lkLen + 4095 & ~4095;
          if (lkEnd <= lkStart) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (!this.rangeMapped(lkStart, lkEnd)) {
            ret = -BigInt(ENOMEM);
            break;
          }
          if (nr !== 229 && !this.hasCap(CAP_IPC_LOCK) && lkEnd - lkStart > this.memlockLimit) {
            ret = -BigInt(ENOMEM);
            break;
          }
          const lkOnFault = nr === 284 && (Number(BigInt.asIntN(32, a2)) & 1) !== 0;
          this.setLocked(lkStart, lkEnd, nr !== 229, lkOnFault);
          ret = 0n;
          break;
        }
        case 230: {
          const MCL_CURRENT = 1, MCL_FUTURE = 2, MCL_ONFAULT = 4;
          const mlaFlags = Number(BigInt.asIntN(32, a0));
          if (mlaFlags === 0 || mlaFlags & ~(MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (mlaFlags & MCL_ONFAULT && !(mlaFlags & MCL_CURRENT)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (!this.hasCap(CAP_IPC_LOCK) && this.memlockLimit === 0) {
            ret = -BigInt(EPERM);
            break;
          }
          if (mlaFlags & MCL_CURRENT && !this.hasCap(CAP_IPC_LOCK)) {
            const mapped = this.readAddressSpace(() => this.vmas.reduce((n, v) => n + (v.end - v.start), 0));
            if (mapped > this.memlockLimit) {
              ret = -BigInt(ENOMEM);
              break;
            }
          }
          if (mlaFlags & MCL_CURRENT) {
            for (const v of this.readAddressSpace(() => this.vmas.map((x) => ({ start: x.start, end: x.end })))) {
              this.setLocked(v.start, v.end, true);
            }
          }
          this.lockFuture = (mlaFlags & MCL_FUTURE) !== 0;
          ret = 0n;
          break;
        }
        case 231:
          this.lockedRegions = [];
          this.lockFuture = false;
          if (this.vfs) this.vfs.vmLckKb = 0;
          ret = 0n;
          break;
        case 227: {
          const MS_ASYNC = 1, MS_INVALIDATE = 2, MS_SYNC = 4;
          const msFlags = Number(BigInt.asIntN(32, a2));
          if (msFlags & ~(MS_ASYNC | MS_INVALIDATE | MS_SYNC)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (msFlags & MS_ASYNC && msFlags & MS_SYNC) {
            ret = -BigInt(EINVAL);
            break;
          }
          const msAddrB = BigInt.asUintN(64, a0);
          if ((msAddrB & 0xFFFn) !== 0n) {
            ret = -BigInt(EINVAL);
            break;
          }
          const msAddr = Number(msAddrB);
          const msLenU = BigInt.asUintN(64, a1);
          if (BigInt(msAddr) + msLenU > BigInt(MAX_PROCESS_BYTES)) {
            ret = -BigInt(ENOMEM);
            break;
          }
          const msEnd = msAddr + (Number(msLenU) + 4095 & ~4095);
          if (msEnd === msAddr) {
            ret = 0n;
            break;
          }
          if (!this.rangeMapped(msAddr, msEnd)) {
            ret = -BigInt(ENOMEM);
            break;
          }
          if (msFlags & MS_INVALIDATE && this.rangeLocked(msAddr, msEnd)) {
            ret = -BigInt(EBUSY);
            break;
          }
          if (msFlags & MS_INVALIDATE && !(msFlags & (MS_SYNC | MS_ASYNC))) {
            const msReload = this.buildMmapReload(cpu2, memory2, msAddr, Number(a1));
            if (msReload) return msReload;
            ret = 0n;
            break;
          }
          const msFlush = this.buildMmapFlush(cpu2, memory2, msAddr, Number(a1), false);
          if (msFlush) return msFlush;
          ret = 0n;
          break;
        }
        case 232: {
          const mcAddr = Number(a0);
          const mcLenU = BigInt.asUintN(64, a1);
          if (mcAddr & 4095) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (BigInt(mcAddr) + mcLenU > BigInt(MAX_PROCESS_BYTES)) {
            ret = -BigInt(ENOMEM);
            break;
          }
          const mcLen = Number(mcLenU);
          const mcPages = Math.ceil(mcLen / 4096);
          if (mcPages > 0 && !this.rangeMapped(mcAddr, mcAddr + mcPages * 4096)) {
            ret = -BigInt(ENOMEM);
            break;
          }
          if (this.ptrFaults(bytes, Number(a2), Math.max(1, mcPages))) {
            ret = -BigInt(EFAULT);
            break;
          }
          const mcVec = Number(a2) + GUEST_MEM_BASE;
          const mcB = new Uint8Array(memory2.buffer);
          const mcKind = this.rangeKind(mcAddr, mcAddr + mcPages * 4096);
          const mcBacked = mcKind.shared || mcKind.fileBacked;
          for (let i = 0; i < mcPages; i++) {
            const base = mcAddr + i * 4096 + GUEST_MEM_BASE;
            let present = this.populatedBytesIn(mcAddr + i * 4096, mcAddr + (i + 1) * 4096) > 0 ? 1 : 0;
            if (!present && mcBacked) {
              const k = this.rangeKind(mcAddr + i * 4096, mcAddr + (i + 1) * 4096);
              if (k.shared || k.fileBacked) present = 1;
            }
            if (!present && base + 4096 <= mcB.length) {
              for (let off = 0; off < 4096; off += 8) {
                if (mcB[base + off] || mcB[base + off + 1] || mcB[base + off + 2] || mcB[base + off + 3] || mcB[base + off + 4] || mcB[base + off + 5] || mcB[base + off + 6] || mcB[base + off + 7]) {
                  present = 1;
                  break;
                }
              }
            }
            mcB[mcVec + i] = present;
          }
          ret = 0n;
          break;
        }
        case 216: {
          const MREMAP_MAYMOVE = 1, MREMAP_FIXED = 2, MREMAP_DONTUNMAP = 4;
          const oldAddr = Number(a0), oldSize = Number(a1), newSize = Number(a2);
          const mrFlags = Number(BigInt.asIntN(32, a3));
          const alignedOld = oldSize + 4095 & ~4095;
          const alignedNew = newSize + 4095 & ~4095;
          if (mrFlags & ~(MREMAP_MAYMOVE | MREMAP_FIXED | MREMAP_DONTUNMAP)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (oldAddr & 4095) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (mrFlags & MREMAP_FIXED && !(mrFlags & MREMAP_MAYMOVE)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (mrFlags & MREMAP_DONTUNMAP && (!(mrFlags & MREMAP_MAYMOVE) || mrFlags & MREMAP_FIXED)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (alignedNew === 0) {
            ret = -BigInt(EINVAL);
            break;
          }
          const mrVma = this.readAddressSpace(() => {
            const v = this.vmaAt(oldAddr);
            return v ? { ...v } : null;
          });
          if (!mrVma) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (alignedOld > mrVma.end - oldAddr) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (mrFlags & MREMAP_FIXED) {
            const fixedAddr = Number(a4);
            if (fixedAddr & 4095) {
              ret = -BigInt(EINVAL);
              break;
            }
            if (oldAddr + alignedOld > fixedAddr && fixedAddr + alignedNew > oldAddr) {
              ret = -BigInt(EINVAL);
              break;
            }
            const fixedEnd = fixedAddr + alignedNew;
            if (fixedEnd + GUEST_MEM_BASE > this.maxBytes) {
              ret = -BigInt(ENOMEM);
              break;
            }
            if (!this.growTo(memory2, fixedEnd + GUEST_MEM_BASE)) {
              ret = -BigInt(ENOMEM);
              break;
            }
            const fixedMem = new Uint8Array(memory2.buffer);
            const keep = Math.min(alignedOld, alignedNew);
            fixedMem.copyWithin(
              fixedAddr + GUEST_MEM_BASE,
              oldAddr + GUEST_MEM_BASE,
              oldAddr + GUEST_MEM_BASE + keep
            );
            fixedMem.fill(0, fixedAddr + GUEST_MEM_BASE + keep, fixedEnd + GUEST_MEM_BASE);
            this.withAddressSpace(() => {
              this.vmaSet(
                fixedAddr,
                fixedEnd,
                { prot: mrVma.prot, shared: mrVma.shared, maxProt: mrVma.maxProt }
              );
              this.vmaSet(oldAddr, oldAddr + alignedOld, null);
              return 0n;
            });
            try {
              this.protHook?.(fixedAddr, alignedNew, mrVma.prot);
            } catch {
            }
            try {
              this.protHook?.(oldAddr, alignedOld, 0);
            } catch {
            }
            ret = BigInt(fixedAddr);
            break;
          }
          if (alignedNew <= alignedOld) {
            if (alignedNew < alignedOld) {
              this.vmaSet(oldAddr + alignedNew, oldAddr + alignedOld, null);
              try {
                this.protHook?.(oldAddr + alignedNew, alignedOld - alignedNew, 0);
              } catch {
              }
            }
            ret = a0;
            break;
          }
          if (BigInt(oldAddr + alignedOld) === this.mmapBase) {
            const needed = oldAddr + alignedNew + GUEST_MEM_BASE;
            if (needed > this.maxBytes) {
              ret = -BigInt(ENOMEM);
              break;
            }
            if (!this.growTo(memory2, needed)) {
              ret = -BigInt(ENOMEM);
              break;
            }
            new Uint8Array(memory2.buffer).fill(0, oldAddr + alignedOld + GUEST_MEM_BASE, oldAddr + alignedNew + GUEST_MEM_BASE);
            this.mmapBase = BigInt(oldAddr + alignedNew);
            this.vmaSet(oldAddr, oldAddr + alignedNew, {
              prot: mrVma.prot,
              shared: mrVma.shared,
              maxProt: mrVma.maxProt
            });
            try {
              this.protHook?.(oldAddr + alignedOld, alignedNew - alignedOld, mrVma.prot);
            } catch {
            }
            ret = a0;
            break;
          }
          if (!(mrFlags & MREMAP_MAYMOVE)) {
            ret = -BigInt(ENOMEM);
            break;
          }
          const newAddr = this.withAddressSpace(() => {
            const picked = this.allocMmapRange(alignedNew);
            const need = Number(picked) + alignedNew + GUEST_MEM_BASE;
            if (need > this.maxBytes || !this.growTo(memory2, need)) return -1n;
            this.vmaSet(Number(picked), Number(picked) + alignedNew, {
              prot: mrVma.prot,
              shared: mrVma.shared,
              maxProt: mrVma.maxProt
            });
            this.vmaSet(oldAddr, oldAddr + alignedOld, null);
            return picked;
          });
          if (newAddr < 0n) {
            ret = -BigInt(ENOMEM);
            break;
          }
          const mrMem = new Uint8Array(memory2.buffer);
          mrMem.copyWithin(Number(newAddr) + GUEST_MEM_BASE, oldAddr + GUEST_MEM_BASE, oldAddr + GUEST_MEM_BASE + oldSize);
          mrMem.fill(0, Number(newAddr) + GUEST_MEM_BASE + oldSize, Number(newAddr) + GUEST_MEM_BASE + alignedNew);
          try {
            this.protHook?.(Number(newAddr), alignedNew, mrVma.prot);
          } catch {
          }
          try {
            this.protHook?.(oldAddr, alignedOld, 0);
          } catch {
          }
          ret = newAddr;
          break;
        }
        // ── Signal notification via fd ──
        case 74: {
          const sfFd = Number(BigInt.asIntN(32, a0));
          const sfMaskPtr = Number(a1);
          const sfSize = Number(BigInt.asIntN(64, a2));
          const sfFlags = Number(BigInt.asIntN(32, a3));
          const SFD_NONBLOCK = 2048, SFD_CLOEXEC = 524288;
          if (sfSize !== 8) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (sfFlags & ~(SFD_NONBLOCK | SFD_CLOEXEC)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (this.ptrFaults(bytes, sfMaskPtr, 8)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const sfMask = view.getBigUint64(sfMaskPtr + GUEST_MEM_BASE, true) & ~(1n << 8n | 1n << 18n);
          if (sfFd === -1) {
            const newFd = this.allocFd();
            if (newFd < 0) {
              ret = -BigInt(EMFILE);
              break;
            }
            this.fdTable.set(newFd, { type: "signalfd", mask: sfMask });
            if (sfFlags & SFD_CLOEXEC) this.fdCloexec.add(newFd);
            if (sfFlags & SFD_NONBLOCK) this.fdFlags.set(newFd, (this.fdFlags.get(newFd) ?? 0) | O_NONBLOCK);
            ret = BigInt(newFd);
            break;
          }
          const sfEntry = this.getFdEntry(sfFd);
          if (!sfEntry) {
            ret = -BigInt(EBADF);
            break;
          }
          if (sfEntry.type !== "signalfd") {
            ret = -BigInt(EINVAL);
            break;
          }
          sfEntry.mask = sfMask;
          ret = BigInt(sfFd);
          break;
        }
        // ── Process creation (modern) ──
        case 435: {
          const argsPtr = Number(a0) + GUEST_MEM_BASE;
          const cloneFlags = view.getBigUint64(argsPtr, true);
          const SIGCHLD2 = 17n;
          if (cloneFlags & SIGCHLD2) {
            const svcPC = view.getBigUint64(PC_OFFSET, true);
            const c3Stack = view.getBigUint64(argsPtr + 40, true);
            const c3Size = view.getBigUint64(argsPtr + 48, true);
            return {
              status: "fork",
              resumePC: svcPC + 4n,
              flags: Number(cloneFlags & 0xffffffffn),
              childStack: c3Stack === 0n ? 0n : c3Stack + c3Size,
              ptid: view.getBigUint64(argsPtr + 24, true),
              tls: view.getBigUint64(argsPtr + 56, true),
              ctid: view.getBigUint64(argsPtr + 16, true)
            };
          }
          ret = -BigInt(ENOSYS);
          break;
        }
        // ── Exec relative to dirfd ──
        case 281: {
          const raw281 = readGuestString(bytes, Number(a1));
          const flags281 = Number(BigInt.asIntN(32, a4));
          const AT_EMPTY_PATH = 4096, AT_SYMLINK_NOFOLLOW2 = 256;
          if (flags281 & ~(AT_EMPTY_PATH | AT_SYMLINK_NOFOLLOW2)) {
            ret = -BigInt(EINVAL);
            break;
          }
          const dirfd281 = Number(BigInt.asIntN(32, a0));
          if ((this.isEmptyPath(raw281) || !raw281.startsWith("/")) && dirfd281 !== -100) {
            if (!this.fdIsOpen(dirfd281)) {
              ret = -BigInt(EBADF);
              break;
            }
          }
          if (this.isEmptyPath(raw281) && !(flags281 & AT_EMPTY_PATH)) {
            ret = -BigInt(ENOENT);
            break;
          }
          const pathname281 = this.resolveAt(a0, raw281);
          const argvPtrs281 = [];
          let argvAddr281 = Number(a2);
          while (true) {
            const ptr = Number(view.getBigUint64(argvAddr281 + GUEST_MEM_BASE, true));
            if (ptr === 0) break;
            argvPtrs281.push(readGuestString(bytes, ptr));
            argvAddr281 += 8;
          }
          if (argvPtrs281.length === 0) argvPtrs281.push("");
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
          const faEntry = this.getFdEntry(Number(a0));
          const faMode = Number(a1);
          const faOff = Number(BigInt.asIntN(64, a2));
          const faLen = Number(BigInt.asIntN(64, a3));
          if (!faEntry) {
            ret = this.vfs?.hasFd(Number(a0)) ? 0n : -BigInt(EBADF);
            break;
          }
          if (this.fdIsPath(Number(a0)) || !this.fdAccess(Number(a0)).writable) {
            ret = -BigInt(EBADF);
            break;
          }
          if (faLen <= 0 || faOff < 0) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (faEntry.type !== "native") {
            ret = faEntry.type === "pipe_read" || faEntry.type === "pipe_write" ? -BigInt(ESPIPE) : -BigInt(ENODEV);
            break;
          }
          const FALLOC_KEEP_SIZE = 1, FALLOC_PUNCH_HOLE = 2;
          if (faMode & ~(FALLOC_KEEP_SIZE | FALLOC_PUNCH_HOLE)) {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          if (faMode & FALLOC_PUNCH_HOLE && !(faMode & FALLOC_KEEP_SIZE)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (BigInt(faOff) + BigInt(faLen) > MAX_FILE_BYTES) {
            ret = -BigInt(EFBIG);
            break;
          }
          if (faMode & FALLOC_PUNCH_HOLE) {
            if (this.sealsOf(faEntry) & 8) {
              ret = -BigInt(EPERM);
              break;
            }
            this.invalidateStat(faEntry.path);
            return {
              status: "bridge",
              method: "fs_punch_hole",
              args: { fd: faEntry.nativeFd, offset: faOff, length: faLen },
              apply: (resp) => {
                if (resp.ret >= 0) this.notify(faEntry.path, IN_MODIFY, 0, false);
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          if (faMode & FALLOC_KEEP_SIZE) {
            ret = 0n;
            break;
          }
          this.invalidateStat(faEntry.path);
          return {
            status: "bridge",
            method: "fs_fallocate",
            args: {
              fd: faEntry.nativeFd,
              size: faOff + faLen,
              noGrow: this.sealsOf(faEntry) & 4 ? 1 : 0
            },
            apply: (resp) => {
              cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            }
          };
        }
        case 45: {
          const traw = readGuestString(bytes, Number(a0));
          {
            const pe = this.pathCheck(traw);
            if (pe) {
              ret = -BigInt(pe);
              break;
            }
          }
          const tpath = this.resolveAt(-100n, traw);
          const tlength = Number(BigInt.asIntN(64, a1));
          if (tlength < 0) {
            ret = -BigInt(EINVAL);
            break;
          }
          const fsize = this.rlimitOf(RLIMIT_FSIZE);
          if (fsize.cur !== RLIM_INFINITY && BigInt(tlength) > fsize.cur) {
            this.raiseSignal(25);
            ret = -BigInt(EFBIG);
            break;
          }
          const bTrunc = this.bridgePath(tpath);
          if (bTrunc) {
            this.invalidateStat(bTrunc);
            return {
              status: "bridge",
              method: "fs_truncate",
              args: { path: bTrunc, length: tlength },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
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
          const RENAME_NOREPLACE = 1, RENAME_EXCHANGE = 2, RENAME_WHITEOUT = 4;
          const r2Flags = Number(BigInt.asUintN(32, a4));
          if (r2Flags & ~(RENAME_NOREPLACE | RENAME_EXCHANGE | RENAME_WHITEOUT)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (r2Flags & RENAME_EXCHANGE && r2Flags & (RENAME_NOREPLACE | RENAME_WHITEOUT)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (r2Flags & RENAME_WHITEOUT) {
            ret = -BigInt(EINVAL);
            break;
          }
          const rawOld276 = readGuestString(bytes, Number(a1));
          const rawNew276 = readGuestString(bytes, Number(a3));
          {
            const pe = this.pathCheck(rawOld276) || this.pathCheck(rawNew276);
            if (pe) {
              ret = -BigInt(pe);
              break;
            }
          }
          const oldpath276 = this.resolveAt(a0, rawOld276);
          const newpath276 = this.resolveAt(a2, rawNew276);
          const bOld276 = this.bridgePath(oldpath276);
          if (bOld276) {
            const bNew276 = this.bridgePath(newpath276) ?? newpath276;
            this.invalidateStat(bOld276);
            this.invalidateStat(bNew276);
            return {
              status: "bridge",
              method: "fs_rename",
              args: { oldPath: bOld276, newPath: bNew276, flags: r2Flags },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          if (this.vfs) {
            if (r2Flags !== 0) {
              ret = -BigInt(EINVAL);
              break;
            }
            ret = BigInt(this.vfs.rename(oldpath276, newpath276));
          } else {
            ret = -BigInt(ENOSYS);
          }
          break;
        }
        case 279: {
          const MFD_CLOEXEC = 1, MFD_ALLOW_SEALING = 2, MFD_HUGETLB = 4, MFD_NOEXEC_SEAL = 8, MFD_EXEC = 16;
          const mfFlags = Number(BigInt.asUintN(32, a1));
          if (mfFlags & ~(MFD_CLOEXEC | MFD_ALLOW_SEALING | MFD_HUGETLB | MFD_NOEXEC_SEAL | MFD_EXEC)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (mfFlags & MFD_NOEXEC_SEAL && mfFlags & MFD_EXEC) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (this.ptrFaults(bytes, Number(a0), 1)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const mfName = readGuestString(bytes, Number(a0));
          if (mfName.length > 249) {
            ret = -BigInt(EINVAL);
            break;
          }
          const mfDir = this.bridgePath("/tmp");
          if (mfDir === null) {
            ret = -BigInt(ENOSYS);
            break;
          }
          if (!this.fdAvailable(0)) {
            ret = -BigInt(EMFILE);
            break;
          }
          const mfFd = this.allocFd();
          return {
            status: "bridge",
            method: "fs_open",
            args: { path: mfDir, flags: Number(O_TMPFILE_BIT | O_DIRECTORY_B | 2n), mode: 384 },
            apply: (resp) => {
              if (resp.ret < 0) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                return;
              }
              const mfOfd = this.newOfdIdPublic();
              this.fdTable.set(mfFd, {
                type: "native",
                nativeFd: resp.ret,
                path: "/memfd:" + mfName,
                ofdId: mfOfd,
                isDir: false,
                ftype: 32768,
                unlinked: true,
                sealable: true
              });
              if (!(mfFlags & MFD_ALLOW_SEALING)) this.fileSeals.set(mfOfd, 1);
              this.nativeRefs.set(resp.ret, 1);
              this.fdFlags.set(mfFd, 2);
              if (mfFlags & MFD_CLOEXEC) this.fdCloexec.add(mfFd);
              cpu2.setReg(0, BigInt(mfFd));
            }
          };
        }
        case 291: {
          const sxFlags = Number(BigInt.asIntN(32, a2));
          const AT_STATX_FLAGS = 256 | 2048 | 4096 | 8192 | 16384 | 24576;
          const STATX_RESERVED = 2147483648;
          if (sxFlags & ~AT_STATX_FLAGS) {
            ret = -BigInt(EINVAL);
            break;
          }
          if ((sxFlags & 24576) === 24576) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (Number(BigInt.asUintN(32, a3)) & STATX_RESERVED) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (!(sxFlags & 4096) && this.ptrFaults(bytes, Number(a1), 1)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const rawSx = readGuestString(bytes, Number(a1));
          if (this.isEmptyPath(rawSx)) {
            if (!(sxFlags & 4096)) {
              ret = -BigInt(ENOENT);
              break;
            }
          } else {
            const peSx = this.pathCheck(rawSx);
            if (peSx) {
              ret = -BigInt(peSx);
              break;
            }
          }
          const sxPath = this.resolveAt(a0, rawSx);
          const sxBufAddr = Number(a4);
          if (this.ptrFaults(bytes, sxBufAddr, 256)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const widenToStatx = () => {
            const tmpOffset = sxBufAddr + 128;
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
            const stAtimeNs = view.getBigInt64(st + 80, true);
            const stMtime = view.getBigInt64(st + 88, true);
            const stMtimeNs = view.getBigInt64(st + 96, true);
            const stCtime = view.getBigInt64(st + 104, true);
            const stCtimeNs = view.getBigInt64(st + 112, true);
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
            view.setUint32(b + 72, Number(stAtimeNs), true);
            view.setBigInt64(b + 80, stCtime, true);
            view.setUint32(b + 88, Number(stCtimeNs), true);
            view.setBigInt64(b + 96, stCtime, true);
            view.setUint32(b + 104, Number(stCtimeNs), true);
            view.setBigInt64(b + 112, stMtime, true);
            view.setUint32(b + 120, Number(stMtimeNs), true);
            const devMajor = (d) => Number(d >> 8n & 0xfffn);
            const devMinor = (d) => Number(d & 0xffn | d >> 12n & 0xfff00n);
            view.setUint32(b + 128, devMajor(BigInt.asUintN(64, stRdev)), true);
            view.setUint32(b + 132, devMinor(BigInt.asUintN(64, stRdev)), true);
            view.setUint32(b + 136, devMajor(BigInt.asUintN(64, stDev)), true);
            view.setUint32(b + 140, devMinor(BigInt.asUintN(64, stDev)), true);
            view.setBigUint64(b + 144, 1n, true);
          };
          const sxTmp = sxBufAddr + 128;
          const sxEntry = sxFlags & 4096 ? this.getFdEntry(Number(BigInt.asIntN(32, a0))) : void 0;
          if (sxEntry?.type === "native") {
            return {
              status: "bridge",
              method: "fs_stat",
              args: { path: sxEntry.path },
              apply: (resp) => {
                if (resp.ret < 0) {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                  return;
                }
                writeBridgeStat(view, sxTmp, resp);
                widenToStatx();
                cpu2.setReg(0, 0n);
              }
            };
          }
          if (sxEntry && sxEntry.type !== "vfs") {
            writeChardevStat(view, sxTmp);
            widenToStatx();
            ret = 0n;
            break;
          }
          const bp291 = sxFlags & 4096 ? null : this.bridgePath(sxPath);
          if (bp291 !== null) {
            return {
              status: "bridge",
              method: sxFlags & 256 ? "fs_lstat" : "fs_stat",
              args: { path: bp291 },
              apply: (resp) => {
                if (resp.ret < 0) {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                  return;
                }
                writeBridgeStat(view, sxTmp, resp);
                widenToStatx();
                cpu2.setReg(0, 0n);
              }
            };
          }
          if (this.vfs) {
            const res = this.vfs.fstatat(Number(BigInt.asIntN(32, a0)), sxPath, sxTmp, sxFlags, view);
            if (res < 0) {
              ret = BigInt(res);
              break;
            }
            widenToStatx();
          } else {
            ret = -BigInt(ENOENT);
            break;
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
          const EFD_CLOEXEC = 524288;
          if (efdFlags & ~(EFD_SEMAPHORE | EFD_NONBLOCK | EFD_CLOEXEC)) {
            ret = -BigInt(EINVAL);
            break;
          }
          const efd = this.allocFd();
          if (efd < 0) {
            ret = -BigInt(EMFILE);
            break;
          }
          const efdId = this.eventfdHook?.({
            op: "create",
            initval: Number(initVal),
            semaphore: (efdFlags & EFD_SEMAPHORE) !== 0
          })?.[0] ?? -1;
          if (efdId < 0) {
            ret = -BigInt(ENOMEM);
            break;
          }
          this.fdTable.set(efd, {
            type: "eventfd",
            id: efdId,
            semaphore: (efdFlags & EFD_SEMAPHORE) !== 0
          });
          this.fdFlags.set(efd, efdFlags & EFD_NONBLOCK);
          if (efdFlags & EFD_CLOEXEC) this.fdCloexec.add(efd);
          ret = BigInt(efd);
          break;
        }
        case 20: {
          const EPOLL_CLOEXEC = 524288;
          const epFlags = Number(a0);
          if (epFlags & ~EPOLL_CLOEXEC) {
            ret = -BigInt(EINVAL);
            break;
          }
          const epfd = this.allocFd();
          if (epfd < 0) {
            ret = -BigInt(EMFILE);
            break;
          }
          this.fdTable.set(epfd, { type: "epoll" });
          if (epFlags & EPOLL_CLOEXEC) this.fdCloexec.add(epfd);
          if (!this.epollSets) this.epollSets = /* @__PURE__ */ new Map();
          this.epollSets.set(epfd, /* @__PURE__ */ new Map());
          ret = BigInt(epfd);
          break;
        }
        case 21: {
          const epfd = Number(BigInt.asIntN(32, a0));
          const epOp = Number(BigInt.asIntN(32, a1));
          const targetFd = Number(BigInt.asIntN(32, a2));
          const eventPtr = Number(a3);
          const EPOLL_CTL_ADD = 1, EPOLL_CTL_DEL = 2, EPOLL_CTL_MOD = 3;
          if (!this.fdExists(epfd)) {
            ret = -BigInt(EBADF);
            break;
          }
          if (!this.epollSets?.has(epfd)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (!this.fdExists(targetFd)) {
            ret = -BigInt(EBADF);
            break;
          }
          if (epfd === targetFd) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (epOp !== EPOLL_CTL_ADD && epOp !== EPOLL_CTL_DEL && epOp !== EPOLL_CTL_MOD) {
            ret = -BigInt(EINVAL);
            break;
          }
          const eset = this.epollSets.get(epfd);
          if (epOp === EPOLL_CTL_DEL) {
            if (!eset.has(targetFd)) {
              ret = -BigInt(ENOENT);
              break;
            }
            eset.delete(targetFd);
            ret = 0n;
            break;
          }
          if (this.ptrFaults(bytes, eventPtr, EPOLL_EVENT_SIZE)) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (!this.fdPollable(targetFd)) {
            ret = -BigInt(EPERM);
            break;
          }
          if (epOp === EPOLL_CTL_ADD && eset.has(targetFd)) {
            ret = -BigInt(EEXIST);
            break;
          }
          if (epOp === EPOLL_CTL_MOD && !eset.has(targetFd)) {
            ret = -BigInt(ENOENT);
            break;
          }
          if (epOp === EPOLL_CTL_ADD && this.epollSets.has(targetFd)) {
            if (this.epollReaches(targetFd, epfd)) {
              ret = -BigInt(ELOOP);
              break;
            }
            if (Math.max(this.epollDepth(epfd), 1 + this.epollDepth(targetFd)) > EPOLL_MAX_DEPTH) {
              ret = -BigInt(EINVAL);
              break;
            }
          }
          const events = view.getUint32(eventPtr + GUEST_MEM_BASE, true);
          const data = view.getBigUint64(eventPtr + GUEST_MEM_BASE + EPOLL_DATA_OFFSET, true);
          eset.set(targetFd, { events, data, lastReady: 0 });
          ret = 0n;
          break;
        }
        case 22: {
          const epSigmask = Number(a4);
          if (epSigmask !== 0 && this.ptrFaults(bytes, epSigmask, 8)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const epSaved = epSigmask !== 0 ? this.signalMask ?? 0n : null;
          if (epSigmask !== 0) this.setSignalMask(view.getBigUint64(epSigmask + GUEST_MEM_BASE, true));
          ret = BigInt(this.epollWait(
            view,
            bytes,
            Number(a0),
            Number(a1),
            Number(BigInt.asIntN(32, a2)),
            Number(BigInt.asIntN(32, a3))
          ));
          if (epSaved !== null) this.setSignalMask(epSaved);
          break;
        }
        case 72: {
          const nfdsArg = Number(BigInt.asIntN(32, a0));
          if (nfdsArg < 0) {
            ret = -BigInt(EINVAL);
            break;
          }
          const nfds = Math.min(nfdsArg, this.nofileLimit);
          const readPtr = Number(a1);
          const writePtr = Number(a2);
          const exceptPtr = Number(a3);
          const tmoPtr = Number(a4);
          const setBytes = nfds + 7 >> 3;
          let selFault = false;
          for (const p of [readPtr, writePtr, exceptPtr]) {
            if (p !== 0 && setBytes > 0 && this.ptrWriteFaults(bytes, p, setBytes)) {
              selFault = true;
              break;
            }
          }
          if (selFault) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (tmoPtr !== 0 && this.ptrFaults(bytes, tmoPtr, 16)) {
            ret = -BigInt(EFAULT);
            break;
          }
          let tmoMs = -1;
          if (tmoPtr !== 0) {
            const sec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE, true));
            const nsec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE + 8, true));
            if (sec < 0 || nsec < 0 || nsec >= 1e9) {
              ret = -BigInt(EINVAL);
              break;
            }
            tmoMs = sec * 1e3 + Math.floor(nsec / 1e6);
          }
          const wantRead = this.readFdSet(view, readPtr, nfds);
          const wantWrite = this.readFdSet(view, writePtr, nfds);
          const wantExcept = this.readFdSet(view, exceptPtr, nfds);
          let selBad = false;
          for (const set of [wantRead, wantWrite, wantExcept]) {
            for (const fd of set) if (!this.fdIsOpen(fd)) {
              selBad = true;
              break;
            }
            if (selBad) break;
          }
          if (selBad) {
            ret = -BigInt(EBADF);
            break;
          }
          let selSaved = null;
          const selSigPtr = Number(a5);
          if (selSigPtr !== 0) {
            if (this.ptrFaults(bytes, selSigPtr, 16)) {
              ret = -BigInt(EFAULT);
              break;
            }
            const ssPtr = Number(view.getBigUint64(selSigPtr + GUEST_MEM_BASE, true));
            const ssLen = Number(view.getBigUint64(selSigPtr + GUEST_MEM_BASE + 8, true));
            if (ssPtr !== 0) {
              if (ssLen !== 8) {
                ret = -BigInt(EINVAL);
                break;
              }
              if (this.ptrFaults(bytes, ssPtr, 8)) {
                ret = -BigInt(EFAULT);
                break;
              }
              selSaved = this.signalMask ?? 0n;
              this.setSignalMask(view.getBigUint64(ssPtr + GUEST_MEM_BASE, true));
            }
          }
          const sockReadFd = [...wantRead].find((fd) => this.getFdEntry(fd)?.type === "socket");
          if (sockReadFd !== void 0 && tmoMs !== 0) {
            let preReady = 0;
            for (const fd of wantRead) if (this.getFdEntry(fd)?.type !== "socket" && this.isFdReadable(fd)) preReady++;
            for (const fd of wantWrite) if (this.isFdWritable(fd)) preReady++;
            if (preReady === 0) {
              const sockEntry = this.getFdEntry(sockReadFd);
              if (sockEntry?.type !== "socket") {
                if (selSaved !== null) this.setSignalMask(selSaved);
                ret = -BigInt(EBADF);
                break;
              }
              const pollTmo = tmoMs < 0 ? 3e4 : tmoMs;
              return {
                status: "bridge",
                method: "net_poll",
                args: { socketId: sockEntry.socketId, timeoutMs: pollTmo },
                apply: (resp) => {
                  if (selSaved !== null) this.setSignalMask(selSaved);
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
          let deadline = tmoMs < 0 ? -1 : this.nowMs() + tmoMs;
          let selSeen = this.ioGeneration();
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
            const remaining = deadline < 0 ? -1 : Math.max(0, deadline - this.nowMs());
            if (deadline >= 0 && remaining === 0) {
              this.writeFdSet(view, readPtr, nfds, readyRead);
              this.writeFdSet(view, writePtr, nfds, readyWrite);
              if (exceptPtr !== 0) this.writeFdSet(view, exceptPtr, nfds, /* @__PURE__ */ new Set());
              ret = 0n;
              break;
            }
            if (!this.parkForIo(selSeen, remaining)) {
              this.writeFdSet(view, readPtr, nfds, readyRead);
              this.writeFdSet(view, writePtr, nfds, readyWrite);
              if (exceptPtr !== 0) this.writeFdSet(view, exceptPtr, nfds, /* @__PURE__ */ new Set());
              ret = 0n;
              break;
            }
            if (this.hasPendingDeliverable()) {
              ret = -BigInt(EINTR);
              break;
            }
            selSeen = this.ioGeneration();
            if (deadline >= 0 && this.nowMs() >= deadline) {
              this.writeFdSet(view, readPtr, nfds, readyRead);
              this.writeFdSet(view, writePtr, nfds, readyWrite);
              if (exceptPtr !== 0) this.writeFdSet(view, exceptPtr, nfds, /* @__PURE__ */ new Set());
              ret = 0n;
              break;
            }
          }
          if (selSaved !== null) this.setSignalMask(selSaved);
          break;
        }
        case 73: {
          const POLLIN = 1, POLLOUT = 4;
          const fdsAddr = Number(a0);
          const nfds = Number(BigInt.asUintN(64, a1));
          if (nfds > this.nofileLimit) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (nfds > 0 && (fdsAddr === 0 || this.ptrWriteFaults(bytes, fdsAddr, nfds * 8))) {
            ret = -BigInt(EFAULT);
            break;
          }
          const tmoPtr = Number(a2);
          if (tmoPtr !== 0 && this.ptrFaults(bytes, tmoPtr, 16)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const ppSigmask = Number(a3);
          if (ppSigmask !== 0 && this.ptrFaults(bytes, ppSigmask, 8)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const ppSaved = ppSigmask !== 0 ? this.signalMask ?? 0n : null;
          if (ppSigmask !== 0) this.setSignalMask(view.getBigUint64(ppSigmask + GUEST_MEM_BASE, true));
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
              if (!this.fdIsOpen(pfd.fd)) {
                pfd.revents = 32;
                count++;
                continue;
              }
              const entry = this.getFdEntry(pfd.fd);
              if (entry?.type === "socket") continue;
              if (pfd.events & POLLIN && this.isFdReadable(pfd.fd)) pfd.revents |= POLLIN;
              if (pfd.events & POLLOUT && this.isFdWritable(pfd.fd)) pfd.revents |= POLLOUT;
              const pe = this.fdTable.get(pfd.fd);
              if (pe?.type === "socket") {
                if (pfd.events & 8192 && pe.shutRd) pfd.revents |= 8192;
                if (pe.shutRd && pe.shutWr) pfd.revents |= 16;
              }
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
          let deadline = tmoMs < 0 ? -1 : this.nowMs() + tmoMs;
          let pollSeen = this.ioGeneration();
          while (true) {
            const ready = checkReady();
            if (ready > 0) {
              writeBack();
              ret = BigInt(ready);
              break;
            }
            const remaining = deadline < 0 ? -1 : Math.max(0, deadline - this.nowMs());
            if (deadline >= 0 && remaining === 0) {
              writeBack();
              ret = 0n;
              break;
            }
            if (!this.parkForIo(pollSeen, remaining)) {
              writeBack();
              ret = 0n;
              break;
            }
            if (this.hasPendingDeliverable()) {
              ret = -BigInt(EINTR);
              break;
            }
            pollSeen = this.ioGeneration();
            if (deadline >= 0 && this.nowMs() >= deadline) {
              writeBack();
              ret = 0n;
              break;
            }
          }
          if (ppSaved !== null) this.setSignalMask(ppSaved);
          break;
        }
        // ── Remaining syscalls from aarch64 spec ──
        // mount(source, target, filesystemtype, mountflags, data). the emulator has a
        // single in-memory tree, so a mount records the mount point in the namespace
        // (visible via /proc/self/mounts and mountinfo) rather than grafting a real fs.
        // the ltp scratch-device tests select tmpfs (LTP_DEV_FS_TYPE=tmpfs), which needs
        // no backing device or mkfs; the mount point directory already exists.
        case 40:
          ret = -BigInt(19);
          break;
        case 39:
        // umount2
        case 60:
        // (legacy no-op slot)
        case 442:
          ret = 0n;
          break;
        case 41:
          ret = -BigInt(EPERM);
          break;
        // Robust futex list
        case 100: {
          const grlPid = Number(BigInt.asIntN(32, a0));
          const grlHead = Number(a1), grlLen = Number(a2);
          if (grlHead === 0 || grlLen === 0) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (this.ptrFaults(bytes, grlHead, 8) || this.ptrFaults(bytes, grlLen, 8)) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (grlPid !== 0 && grlPid !== this.currentPid && !this.pidExists(grlPid)) {
            ret = -BigInt(ESRCH);
            break;
          }
          if (grlPid !== 0 && grlPid !== this.currentPid && !this.hasCap(CAP_SYS_PTRACE)) {
            ret = -BigInt(EPERM);
            break;
          }
          view.setBigUint64(grlHead + GUEST_MEM_BASE, this.robustListHead, true);
          view.setBigUint64(grlLen + GUEST_MEM_BASE, 24n, true);
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
        case 210: {
          const shSock = this.sockFd(Number(a0));
          if ("err" in shSock) {
            ret = -BigInt(shSock.err);
            break;
          }
          const how = Number(BigInt.asIntN(32, a1));
          if (how < 0 || how > 2) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (shSock.e.type === "socket" && !shSock.e.remoteAddr && !shSock.e.listening) {
            ret = -BigInt(ENOTCONN);
            break;
          }
          if (shSock.e.type === "socket") {
            if (how === 0 || how === 2) shSock.e.shutRd = true;
            if (how === 1 || how === 2) shSock.e.shutWr = true;
          }
          ret = 0n;
          break;
        }
        case 211: {
          const smSock = this.sockFd(Number(a0));
          if ("err" in smSock) {
            ret = -BigInt(smSock.err);
            break;
          }
          if (smSock.e.type !== "socket") {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          const smEntry = smSock.e;
          if (Number(a1) === 0 || this.ptrFaults(bytes, Number(a1), 56)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const mhdr = Number(a1) + GUEST_MEM_BASE;
          const iovPtr = Number(view.getBigUint64(mhdr + 16, true));
          const iovLen = Number(view.getBigUint64(mhdr + 24, true));
          if (iovLen > 1024) {
            ret = -BigInt(EMSGSIZE);
            break;
          }
          if (iovLen > 0 && (iovPtr === 0 || this.ptrFaults(bytes, iovPtr, iovLen * 16))) {
            ret = -BigInt(EFAULT);
            break;
          }
          const smFlags = Number(BigInt.asIntN(32, a2));
          if (smFlags & 1 && smEntry.sockType !== 1) {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          let totalLen = 0;
          let smFault = false;
          for (let i = 0; i < iovLen; i++) {
            const base = Number(view.getBigUint64(iovPtr + i * 16 + GUEST_MEM_BASE, true));
            const len = Number(view.getBigUint64(iovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
            if (len > 0 && this.ptrFaults(bytes, base, len)) {
              smFault = true;
              break;
            }
            totalLen += len;
          }
          if (smFault) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (smEntry.sockType === 2 && totalLen > (smEntry.domain === 10 ? 65527 : 65507)) {
            ret = -BigInt(EMSGSIZE);
            break;
          }
          const smNamePtr0 = Number(view.getBigUint64(mhdr, true));
          if (smEntry.sockType === 1 && (!smEntry.remoteAddr || smEntry.shutWr) && smNamePtr0 === 0) {
            if (!(smFlags & 16384)) this.raiseSignal(13);
            ret = -BigInt(EPIPE);
            break;
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
              args: {
                socketId: smEntry.socketId,
                data: smB64,
                host: dest.host,
                port: dest.port,
                more: Number(BigInt.asIntN(32, a2)) & 32768 ? 1 : 0
              },
              apply: (resp) => {
                cpu2.setReg(0, resp.ret < 0 ? BigInt.asUintN(64, BigInt(resp.ret)) : BigInt(totalLen));
              }
            };
          }
          return {
            status: "bridge",
            method: "net_write",
            args: {
              socketId: smEntry.socketId,
              data: smB64,
              more: Number(BigInt.asIntN(32, a2)) & 32768 ? 1 : 0
            },
            apply: (resp) => {
              cpu2.setReg(0, resp.ret < 0 ? BigInt.asUintN(64, BigInt(resp.ret)) : BigInt(totalLen));
            }
          };
        }
        case 212: {
          const rmSock = this.sockFd(Number(a0));
          if ("err" in rmSock) {
            ret = -BigInt(rmSock.err);
            break;
          }
          if (rmSock.e.type !== "socket") {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          const rmEntry = rmSock.e;
          if (Number(a1) === 0 || this.ptrWriteFaults(bytes, Number(a1), 56)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const rmhdr = Number(a1) + GUEST_MEM_BASE;
          const rmIovPtr = Number(view.getBigUint64(rmhdr + 16, true));
          const rmIovLen = Number(view.getBigUint64(rmhdr + 24, true));
          if (rmIovLen > 1024) {
            ret = -BigInt(EMSGSIZE);
            break;
          }
          if (rmIovLen > 0 && (rmIovPtr === 0 || this.ptrFaults(bytes, rmIovPtr, rmIovLen * 16))) {
            ret = -BigInt(EFAULT);
            break;
          }
          let rmCapacity = 0;
          for (let i = 0; i < rmIovLen; i++) {
            rmCapacity += Number(view.getBigUint64(rmIovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
          }
          const rmFlags = Number(BigInt.asIntN(32, a2));
          if (rmFlags & 1) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (rmFlags & 8192) {
            ret = -BigInt(EAGAIN);
            break;
          }
          const rmNonblock = this.isNonBlockingFd(Number(a0)) || (rmFlags & 64) !== 0;
          return {
            status: "bridge",
            method: "net_read",
            args: {
              socketId: rmEntry.socketId,
              count: Math.min(rmCapacity, BRIDGE_READ_MAX),
              nonblock: rmNonblock ? 1 : 0,
              peek: rmFlags & 2 ? 1 : 0
            },
            apply: (resp) => {
              this.nt(`recvmsg fd=${Number(a0)} cap=${rmCapacity} -> ret=${resp.ret} dlen=${resp.data ? Math.floor(resp.data.length * 3 / 4) : 0}`);
              if (resp.ret === -11 && !rmNonblock) {
                this.retryAfterBridge = true;
                return;
              }
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
              const rmFrom = this.peerFromMeta(resp) ?? rmEntry.remoteAddr;
              if (namePtr !== 0 && rmEntry.sockType === 2 && rmFrom) {
                const e = this.putSockaddr(bytes, view, namePtr, nameLenPtr - GUEST_MEM_BASE, rmFrom);
                if (e < 0) {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(e)));
                  return;
                }
              } else {
                view.setUint32(nameLenPtr, 0, true);
              }
              view.setInt32(rmhdr + 48, 0, true);
              view.setBigUint64(rmhdr + 40, 0n, true);
              cpu2.setReg(0, BigInt(recvd.length));
            }
          };
        }
        case 213: {
          const raFd = Number(BigInt.asIntN(32, a0));
          if (!this.fdIsOpen(raFd) || this.fdIsPath(raFd) || !this.fdAccess(raFd).readable) {
            ret = -BigInt(EBADF);
            break;
          }
          const raEnt = this.getFdEntry(raFd);
          if (raEnt?.type === "pipe_read" || raEnt?.type === "pipe_write") {
            ret = -BigInt(29);
            break;
          }
          if (raEnt?.type === "native") {
            if (raEnt.isDir === true) {
              ret = -BigInt(EINVAL);
              break;
            }
          } else if ((this.vfs?.fdOfdType(raFd) ?? null) !== "file") {
            ret = -BigInt(EINVAL);
            break;
          }
          ret = 0n;
          break;
        }
        case 269: {
          const smmSock = this.sockFd(Number(a0));
          if ("err" in smmSock) {
            ret = -BigInt(smmSock.err);
            break;
          }
          if (smmSock.e.type !== "socket") {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          const smmEntry = smmSock.e;
          const smmVec = Number(a1);
          let smmLen = Number(BigInt.asUintN(32, a2));
          if (smmLen === 0) {
            ret = 0n;
            break;
          }
          if (smmLen > 1024) smmLen = 1024;
          if (this.ptrWriteFaults(bytes, smmVec, smmLen * 64)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const smmMsgs = [];
          const smmSizes = [];
          let smmErr = 0;
          for (let m = 0; m < smmLen; m++) {
            const hdr = smmVec + m * 64 + GUEST_MEM_BASE;
            const iovPtr = Number(view.getBigUint64(hdr + 16, true));
            const iovCount = Number(view.getBigUint64(hdr + 24, true));
            if (iovCount > 1024) {
              smmErr = EMSGSIZE;
              break;
            }
            let total = 0;
            for (let i = 0; i < iovCount; i++) {
              total += Number(view.getBigUint64(iovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
            }
            if (smmEntry.sockType === 2 && total > (smmEntry.domain === 10 ? 65527 : 65507)) {
              smmErr = EMSGSIZE;
              break;
            }
            const payload = new Uint8Array(total);
            let off = 0;
            for (let i = 0; i < iovCount; i++) {
              const base = Number(view.getBigUint64(iovPtr + i * 16 + GUEST_MEM_BASE, true));
              const len = Number(view.getBigUint64(iovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
              if (len > 0 && this.ptrFaults(bytes, base, len)) {
                smmErr = EFAULT;
                break;
              }
              payload.set(bytes.subarray(base + GUEST_MEM_BASE, base + GUEST_MEM_BASE + len), off);
              off += len;
            }
            if (smmErr) break;
            const namePtr = Number(view.getBigUint64(hdr, true));
            const nameLen = view.getUint32(hdr + 8, true);
            const dest = namePtr !== 0 && nameLen > 0 ? this.parseSockaddr(bytes, namePtr, nameLen) : null;
            smmMsgs.push({ data: bytesToBase64(payload), host: dest ? dest.host : "", port: dest ? dest.port : 0 });
            smmSizes.push(total);
          }
          if (smmErr && smmMsgs.length === 0) {
            ret = -BigInt(smmErr);
            break;
          }
          if (smmEntry.sockType === 1 && (!smmEntry.remoteAddr || smmEntry.shutWr) && !smmMsgs.some((m) => m.host)) {
            this.raiseSignal(13);
            ret = -BigInt(EPIPE);
            break;
          }
          return {
            status: "bridge",
            method: "net_sendmmsg",
            args: { socketId: smmEntry.socketId, msgs: smmMsgs },
            apply: (resp) => {
              if (resp.ret < 0) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                return;
              }
              for (let m = 0; m < resp.ret && m < smmSizes.length; m++) {
                view.setUint32(smmVec + m * 64 + 56 + GUEST_MEM_BASE, smmSizes[m], true);
              }
              cpu2.setReg(0, BigInt(resp.ret));
            }
          };
        }
        case 243: {
          const rmmSock = this.sockFd(Number(a0));
          if ("err" in rmmSock) {
            ret = -BigInt(rmmSock.err);
            break;
          }
          if (rmmSock.e.type !== "socket") {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          const rmmEntry = rmmSock.e;
          const rmmVec = Number(a1);
          let rmmLen = Number(BigInt.asUintN(32, a2));
          if (rmmLen === 0) {
            ret = 0n;
            break;
          }
          if (rmmLen > 1024) rmmLen = 1024;
          if (this.ptrWriteFaults(bytes, rmmVec, rmmLen * 64)) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (rmmEntry.shutRd) {
            ret = 0n;
            break;
          }
          const rmmCaps = [];
          for (let m = 0; m < rmmLen; m++) {
            const hdr = rmmVec + m * 64 + GUEST_MEM_BASE;
            const iovPtr = Number(view.getBigUint64(hdr + 16, true));
            const iovCount = Number(view.getBigUint64(hdr + 24, true));
            let cap = 0;
            for (let i = 0; i < iovCount; i++) {
              cap += Number(view.getBigUint64(iovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
            }
            rmmCaps.push(cap);
          }
          const rmmFlags = Number(BigInt.asIntN(32, a3));
          let rmmTimeout = -1;
          if (Number(a4) !== 0) {
            if (this.ptrFaults(bytes, Number(a4), 16)) {
              ret = -BigInt(EFAULT);
              break;
            }
            const tsec = Number(view.getBigInt64(Number(a4) + GUEST_MEM_BASE, true));
            const tnsec = Number(view.getBigInt64(Number(a4) + 8 + GUEST_MEM_BASE, true));
            if (tsec < 0 || tnsec < 0 || tnsec > 999999999) {
              ret = -BigInt(EINVAL);
              break;
            }
            rmmTimeout = tsec * 1e3 + Math.ceil(tnsec / 1e6);
          }
          if (rmmFlags & 64) rmmTimeout = 0;
          if (this.isNonBlockingFd(Number(a0))) rmmTimeout = 0;
          if (rmmTimeout < 0) rmmTimeout = 3e4;
          const rmmMax = Math.min(Math.max(...rmmCaps, 0), BRIDGE_READ_MAX);
          return {
            status: "bridge",
            method: "net_recvmmsg",
            args: { socketId: rmmEntry.socketId, vlen: rmmLen, maxLen: rmmMax, timeoutMs: rmmTimeout },
            apply: (resp) => {
              if (resp.ret < 0) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                return;
              }
              const msgs = resp.bytes ? JSON.parse(new TextDecoder().decode(resp.bytes)) : [];
              for (let m = 0; m < msgs.length; m++) {
                const hdr = rmmVec + m * 64 + GUEST_MEM_BASE;
                const payload = Uint8Array.from(atob(msgs[m].data), (c) => c.charCodeAt(0));
                const iovPtr = Number(view.getBigUint64(hdr + 16, true));
                const iovCount = Number(view.getBigUint64(hdr + 24, true));
                let remaining = Math.min(payload.length, rmmCaps[m]);
                const written = remaining;
                let src = 0;
                for (let i = 0; i < iovCount && remaining > 0; i++) {
                  const base = Number(view.getBigUint64(iovPtr + i * 16 + GUEST_MEM_BASE, true));
                  const len = Number(view.getBigUint64(iovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
                  const take = Math.min(len, remaining);
                  bytes.set(payload.subarray(src, src + take), base + GUEST_MEM_BASE);
                  src += take;
                  remaining -= take;
                }
                const namePtr = Number(view.getBigUint64(hdr, true));
                if (namePtr !== 0 && msgs[m].host) {
                  const wrote = this.writeSockaddr(
                    view,
                    namePtr,
                    { family: msgs[m].host.includes(":") ? 10 : 2, host: msgs[m].host, port: msgs[m].port }
                  );
                  view.setUint32(hdr + 8, wrote, true);
                }
                view.setInt32(hdr + 48, payload.length > rmmCaps[m] ? 32 : 0, true);
                view.setBigUint64(hdr + 40, 0n, true);
                view.setUint32(hdr + 56, written, true);
              }
              cpu2.setReg(0, BigInt(msgs.length));
            }
          };
        }
        // socketpair is handled by the worker (case 199), which owns the unix-socket rings. it can
        // only reach here on the host-side test path, where there is no kernel to allocate them.
        case 199: {
          const spErr = sockTripleErrno(
            Number(BigInt.asIntN(32, a0)),
            Number(BigInt.asIntN(32, a1)),
            Number(BigInt.asIntN(32, a2)),
            this.euid
          );
          ret = -BigInt(spErr !== 0 ? spErr : EOPNOTSUPP);
          break;
        }
        // File operations (remaining)
        case 75:
        // vmsplice
        case 76:
        // splice
        case 77:
          ret = -BigInt(ENOSYS);
          break;
        case 81:
          ret = 0n;
          break;
        case 84: {
          const SYNC_FILE_RANGE_ALL = 7;
          if (Number(BigInt.asUintN(32, a3)) & ~SYNC_FILE_RANGE_ALL) {
            ret = -BigInt(EINVAL);
            break;
          }
          const sfrOff = BigInt.asIntN(64, a1);
          const sfrLen = BigInt.asIntN(64, a2);
          if (sfrOff < 0n || sfrLen < 0n) {
            ret = -BigInt(EINVAL);
            break;
          }
          const sfrFd = Number(BigInt.asIntN(32, a0));
          if (!this.fdIsOpen(sfrFd)) {
            ret = -BigInt(EBADF);
            break;
          }
          const sfrEntry = this.getFdEntry(sfrFd);
          const sfrType = sfrEntry?.type === "native" ? sfrEntry.ftype ?? 0 : 0;
          const sfrPipe = sfrEntry?.type === "pipe_read" || sfrEntry?.type === "pipe_write" || sfrEntry?.type === "socket" || sfrEntry?.type === "unixsock" || sfrType === 4096 || sfrType === 49152 || sfrType === 8192 || !sfrEntry && !!this.vfs?.isCharDev?.(sfrFd);
          if (sfrPipe) {
            ret = -BigInt(ESPIPE);
            break;
          }
          ret = 0n;
          break;
        }
        case 18:
          ret = -BigInt(ENOSYS);
          break;
        // Capabilities
        case 90: {
          const capHdr = this.capHeader(bytes, view, Number(a0));
          if ("err" in capHdr && capHdr.err !== EINVAL) {
            ret = -BigInt(capHdr.err);
            break;
          }
          const capDataPtr = Number(a1);
          if (capDataPtr === 0) {
            ret = 0n;
            break;
          }
          if ("err" in capHdr) {
            ret = -BigInt(capHdr.err);
            break;
          }
          if (capHdr.pid < 0) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (capHdr.pid !== 0 && capHdr.pid !== this.currentPid && !this.pidExists(capHdr.pid)) {
            ret = -BigInt(ESRCH);
            break;
          }
          const capWords = capHdr.version === 429392688 ? 12 : 24;
          if (this.ptrWriteFaults(bytes, capDataPtr, capWords)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const base = capDataPtr + GUEST_MEM_BASE;
          const lo = (v) => Number(v & 0xFFFFFFFFn) >>> 0;
          const hi = (v) => Number(v >> 32n & 0xFFFFFFFFn) >>> 0;
          view.setUint32(base, lo(this.capEff), true);
          view.setUint32(base + 4, lo(this.capPerm), true);
          view.setUint32(base + 8, lo(this.capInh), true);
          if (capWords === 24) {
            view.setUint32(base + 12, hi(this.capEff), true);
            view.setUint32(base + 16, hi(this.capPerm), true);
            view.setUint32(base + 20, hi(this.capInh), true);
          }
          ret = 0n;
          break;
        }
        case 91: {
          const setHdr = this.capHeader(bytes, view, Number(a0));
          if ("err" in setHdr) {
            ret = -BigInt(setHdr.err);
            break;
          }
          if (setHdr.pid !== 0 && setHdr.pid !== this.currentPid) {
            ret = -BigInt(EPERM);
            break;
          }
          const dp = Number(a1);
          if (dp === 0) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (this.ptrFaults(bytes, dp, setHdr.version === 429392688 ? 12 : 24)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const b = dp + GUEST_MEM_BASE;
          const join = (loOff, hiOff) => BigInt(view.getUint32(b + loOff, true) >>> 0) | BigInt(view.getUint32(b + hiOff, true) >>> 0) << 32n;
          const wantEff = join(0, 12), wantPerm = join(4, 16), wantInh = join(8, 20);
          if ((wantPerm & ~this.capPerm) !== 0n) {
            ret = -BigInt(EPERM);
            break;
          }
          if ((wantEff & ~wantPerm) !== 0n) {
            ret = -BigInt(EPERM);
            break;
          }
          if ((wantInh & ~(this.capInh | this.capPerm)) !== 0n) {
            ret = -BigInt(EPERM);
            break;
          }
          this.capPerm = wantPerm;
          this.capEff = wantEff;
          this.capInh = wantInh;
          ret = 0n;
          break;
        }
        // Kernel log
        case 116: {
          const slType = Number(BigInt.asIntN(32, a0));
          const slBuf = Number(a1);
          const slLen = Number(BigInt.asIntN(32, a2));
          if (slType < 0 || slType > 10) {
            ret = -BigInt(EINVAL);
            break;
          }
          const slPriv = slType !== 3 && slType !== 9 && slType !== 10;
          if (slPriv && !this.hasCap(CAP_SYS_ADMIN)) {
            ret = -BigInt(EPERM);
            break;
          }
          if (slType === 2 || slType === 3 || slType === 4) {
            if (slBuf === 0) {
              ret = -BigInt(EINVAL);
              break;
            }
            if (slLen < 0) {
              ret = -BigInt(EINVAL);
              break;
            }
            if (slLen > 0 && this.ptrWriteFaults(bytes, slBuf, slLen)) {
              ret = -BigInt(EFAULT);
              break;
            }
            ret = 0n;
            break;
          }
          if (slType === 8) {
            if (slLen < 1 || slLen > 8) {
              ret = -BigInt(EINVAL);
              break;
            }
          }
          ret = 0n;
          break;
        }
        // Process tracing
        case 117:
          ret = -BigInt(EPERM);
          break;
        case 127:
          ret = -BigInt(ENOSYS);
          break;
        case 128:
          ret = -BigInt(4);
          break;
        case 170: {
          if (Number(a0) === 0) {
            ret = 0n;
            break;
          }
          if (this.ptrFaults(bytes, Number(a0), 16)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const stvSec = view.getBigInt64(Number(a0) + GUEST_MEM_BASE, true);
          const stvUsec = view.getBigInt64(Number(a0) + GUEST_MEM_BASE + 8, true);
          if (stvUsec < 0n || stvUsec >= 1000000n || stvSec < 0n) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (!this.hasCap(CAP_SYS_TIME)) {
            ret = -BigInt(EPERM);
            break;
          }
          this.clockOffsetNs = stvSec * 1000000000n + stvUsec * 1000n - (this.clkRealBaseNs + this.elapsedNs());
          ret = 0n;
          break;
        }
        // Extended attributes
        //
        // twelve syscalls, three operations x (path, link, descriptor). the differences between the
        // flavours are the target and whether a final symlink is followed; everything after that is
        // the same call into the filesystem, which owns the namespace rules ("user.*" only on regular
        // files and directories, "trusted.*" privileged) and the storage.
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
        case 16: {
          const byFd = nr === 7 || nr === 10 || nr === 13 || nr === 16;
          const nofollow = nr === 6 || nr === 9 || nr === 12 || nr === 15;
          const isList = nr >= 11 && nr <= 13;
          let target;
          if (byFd) {
            const xfd = Number(BigInt.asIntN(32, a0));
            if (!this.fdIsOpen(xfd) || this.fdIsPath(xfd) || this.vfs?.isOPath(xfd)) {
              ret = -BigInt(EBADF);
              break;
            }
            const xe = this.getFdEntry(xfd);
            if (xe?.type !== "native") {
              ret = -BigInt(EOPNOTSUPP);
              break;
            }
            target = xe.path;
          } else {
            if (this.ptrFaults(bytes, Number(a0), 1)) {
              ret = -BigInt(14);
              break;
            }
            const rawX = readGuestString(bytes, Number(a0));
            const pe = this.pathCheck(rawX);
            if (pe) {
              ret = -BigInt(pe);
              break;
            }
            const bp = this.bridgePath(rawX);
            if (bp === null) {
              ret = -BigInt(EOPNOTSUPP);
              break;
            }
            target = bp;
          }
          let xname = "";
          if (!isList) {
            if (this.ptrFaults(bytes, Number(a1), 1)) {
              ret = -BigInt(14);
              break;
            }
            xname = readGuestString(bytes, Number(a1));
          }
          const creds = this.getCreds();
          if (nr <= 7) {
            const xsize = Number(a3);
            if (xsize < 0) {
              ret = -BigInt(EINVAL);
              break;
            }
            if (xsize > 0 && this.ptrFaults(bytes, Number(a2), xsize)) {
              ret = -BigInt(14);
              break;
            }
            const val = xsize > 0 ? bytes.subarray(Number(a2) + GUEST_MEM_BASE, Number(a2) + GUEST_MEM_BASE + xsize) : new Uint8Array(0);
            return {
              status: "bridge",
              method: "fs_setxattr",
              args: {
                path: target,
                name: xname,
                value: bytesToBase64(val),
                flags: Number(BigInt.asIntN(32, a4)),
                nofollow,
                creds
              },
              apply: (resp) => cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)))
            };
          }
          if (nr >= 14) {
            return {
              status: "bridge",
              method: "fs_removexattr",
              args: { path: target, name: xname, nofollow, creds },
              apply: (resp) => cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)))
            };
          }
          const bufPtr = Number(isList ? a1 : a2);
          const bufLen = Number(isList ? a2 : a3);
          if (bufLen < 0) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (bufLen > 0 && this.ptrFaults(bytes, bufPtr, bufLen)) {
            ret = -BigInt(14);
            break;
          }
          return {
            status: "bridge",
            method: isList ? "fs_listxattr" : "fs_getxattr",
            args: isList ? { path: target, nofollow, creds } : { path: target, name: xname, nofollow, creds },
            apply: (resp) => {
              if (resp.ret < 0) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                return;
              }
              const out = resp.data ? Uint8Array.from(atob(resp.data), (c) => c.charCodeAt(0)) : new Uint8Array(0);
              if (bufLen === 0) {
                cpu2.setReg(0, BigInt(out.length));
                return;
              }
              if (out.length > bufLen) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(-34)));
                return;
              }
              bytes.set(out, bufPtr + GUEST_MEM_BASE);
              cpu2.setReg(0, BigInt(out.length));
            }
          };
        }
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
        // i/o scheduling class and priority (fanotify is 262/263). handled by the worker, which can
        // reach the shared process table these name a pid in; they only arrive here on the host-side
        // test path, where there is no table.
        case 30:
        // ioprio_set
        case 31:
          ret = -BigInt(ENOSYS);
          break;
        case 42:
          ret = -BigInt(ENOSYS);
          break;
        case 51: {
          if (this.ptrFaults(bytes, Number(a0))) {
            ret = -BigInt(EFAULT);
            break;
          }
          const crPath = readGuestString(bytes, Number(a0));
          if (crPath.length === 0) {
            ret = -BigInt(ENOENT);
            break;
          }
          if (crPath.length > 4095) {
            ret = -BigInt(ENAMETOOLONG);
            break;
          }
          const crBridge = this.bridgePath(crPath);
          if (!crBridge) {
            ret = -BigInt(EPERM);
            break;
          }
          return {
            status: "bridge",
            method: "fs_stat",
            args: { path: crBridge },
            apply: (resp) => {
              if (resp.ret < 0) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                return;
              }
              const mode = resp._meta ? resp._meta[0] : resp.mode;
              if (!mode || (mode & 61440) !== 16384) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(-ENOTDIR)));
                return;
              }
              const rUid = resp._meta ? resp._meta[4] : 0;
              const rGid = resp._meta ? resp._meta[5] : 0;
              if (!this.mayExec(mode, rUid, rGid)) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(-EACCES)));
                return;
              }
              if (!this.hasCap(CAP_SYS_CHROOT)) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(-EPERM)));
                return;
              }
              let newRoot = crBridge;
              if (resp.data) {
                try {
                  const d = atob(resp.data);
                  if (d.length > 0) newRoot = d;
                } catch {
                }
              }
              this.rootDir = newRoot === "/" ? "/" : newRoot.replace(/\/+$/, "");
              cpu2.setReg(0, 0n);
            }
          };
        }
        case 58:
          ret = -BigInt(ENOSYS);
          break;
        case 69:
        // preadv(fd, iov, iovcnt, offset)
        case 286: {
          const prvFd = Number(a0), prvIov = Number(a1), prvCnt = Number(a2);
          let prvOff = BigInt.asIntN(64, a3);
          const prvCurrent = nr === 286 && prvOff === -1n;
          if (prvCurrent) prvOff = 0n;
          const prvErr = this.positionalIoCheck(prvFd, prvCurrent ? 0n : prvOff, false);
          if (prvErr !== 0) {
            ret = BigInt(prvErr);
            break;
          }
          if (nr === 286 && (Number(BigInt.asIntN(32, a5)) & ~RWF_SUPPORTED) !== 0) {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          const prvLen = this.iovTotal(view, prvIov, prvCnt);
          if (prvLen < 0) {
            ret = BigInt(prvLen);
            break;
          }
          if (prvLen === 0) {
            ret = 0n;
            break;
          }
          let prvFault = false;
          for (let i = 0; i < prvCnt; i++) {
            const len = Number(view.getBigUint64(prvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
            if (len === 0) continue;
            const base = Number(view.getBigUint64(prvIov + i * 16 + GUEST_MEM_BASE, true));
            if (this.ptrWriteFaults(bytes, base, len)) {
              prvFault = true;
              break;
            }
          }
          if (prvFault) {
            ret = -BigInt(EFAULT);
            break;
          }
          const prvScatter = (d) => {
            let off = 0;
            for (let i = 0; i < prvCnt && off < d.length; i++) {
              const len = Number(view.getBigUint64(prvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
              if (len === 0) continue;
              const base = Number(view.getBigUint64(prvIov + i * 16 + GUEST_MEM_BASE, true));
              const n = Math.min(len, d.length - off);
              bytes.set(d.subarray(off, off + n), base + GUEST_MEM_BASE);
              off += n;
            }
          };
          const prvWant = Math.min(prvLen, BRIDGE_READ_MAX);
          const prvEntry = this.fdTable.get(prvFd);
          if (prvEntry?.type === "native") {
            const prvAt = prvCurrent ? -1 : Number(prvOff);
            return {
              status: "bridge",
              method: prvCurrent ? "fs_read" : "fs_pread",
              args: prvCurrent ? { fd: prvEntry.nativeFd, count: prvWant } : { fd: prvEntry.nativeFd, offset: prvAt, count: prvWant },
              apply: (resp) => {
                if (resp.ret < 0) {
                  cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                  return;
                }
                const d = resp.bytes ?? new Uint8Array(0);
                prvScatter(d);
                cpu2.setReg(0, BigInt(d.length));
              }
            };
          }
          if (this.vfs?.hasFd(prvFd)) {
            const prvSaved = this.vfs.lseek(prvFd, 0n, 1);
            if (prvSaved < 0n) {
              ret = prvSaved;
              break;
            }
            if (!prvCurrent) this.vfs.lseek(prvFd, prvOff, 0);
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
            if (!prvCurrent) this.vfs.lseek(prvFd, prvSaved, 0);
            ret = BigInt(prvTotal);
            break;
          }
          ret = -BigInt(EBADF);
          break;
        }
        case 70:
        // pwritev(fd, iov, iovcnt, offset)
        case 287: {
          const pwvFd = Number(a0), pwvIov = Number(a1), pwvCnt = Number(a2);
          let pwvOff = BigInt.asIntN(64, a3);
          const pwvCurrent = nr === 287 && pwvOff === -1n;
          if (pwvCurrent) pwvOff = 0n;
          const pwvErr = this.positionalIoCheck(pwvFd, pwvCurrent ? 0n : pwvOff, true);
          if (pwvErr !== 0) {
            ret = BigInt(pwvErr);
            break;
          }
          if (nr === 287 && (Number(BigInt.asIntN(32, a5)) & ~RWF_SUPPORTED) !== 0) {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          const pwvLen = this.iovTotal(view, pwvIov, pwvCnt);
          if (pwvLen < 0) {
            ret = BigInt(pwvLen);
            break;
          }
          if (pwvLen === 0) {
            ret = 0n;
            break;
          }
          const pwvEntry = this.fdTable.get(pwvFd);
          if (this.sealsOf(pwvEntry) & 8) {
            ret = -BigInt(EPERM);
            break;
          }
          if (pwvEntry?.type === "native") {
            const combined = new Uint8Array(pwvLen);
            let off = 0;
            let faulted = false;
            for (let i = 0; i < pwvCnt; i++) {
              const base = Number(view.getBigUint64(pwvIov + i * 16 + GUEST_MEM_BASE, true));
              const len = Number(view.getBigUint64(pwvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
              if (len === 0) continue;
              if (this.ptrFaults(bytes, base, len)) {
                faulted = true;
                break;
              }
              combined.set(bytes.subarray(base + GUEST_MEM_BASE, base + GUEST_MEM_BASE + len), off);
              off += len;
            }
            if (faulted) {
              ret = -BigInt(EFAULT);
              break;
            }
            return {
              status: "bridge",
              method: pwvCurrent ? "fs_write" : "fs_pwrite",
              args: pwvCurrent ? { fd: pwvEntry.nativeFd, data: bytesToBase64(combined) } : { fd: pwvEntry.nativeFd, offset: Number(pwvOff), data: bytesToBase64(combined) },
              apply: (resp) => {
                if (resp.ret >= 0) this.notifyFd(pwvFd, IN_MODIFY);
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          if (this.vfs?.hasFd(pwvFd)) {
            const pwvSaved = this.vfs.lseek(pwvFd, 0n, 1);
            if (pwvSaved < 0n) {
              ret = pwvSaved;
              break;
            }
            if (!pwvCurrent) this.vfs.lseek(pwvFd, pwvOff, 0);
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
            if (!pwvCurrent) this.vfs.lseek(pwvFd, pwvSaved, 0);
            ret = BigInt(pwvTotal);
            break;
          }
          ret = -BigInt(EBADF);
          break;
        }
        // timerfd (verify correct numbers)
        // Per aarch64 spec: timerfd_create=85, timerfd_settime=86, timerfd_gettime=87
        // 82-84 are unused on aarch64? Let me check — they might be signalfd/timerfd/eventfd
        // Actually: 82=signalfd (old), 83=timerfd_create (old), 84=eventfd (old)
        // These are the OLD non-flagged versions; aarch64 should use 74/85/19 instead
        // fsync/fdatasync. these were numbered as if this were x86 -- as old signalfd/timerfd/eventfd
        // -- and answered ENOSYS, so the one call a program makes to be SURE its data is on disk was
        // the one call that failed. apk builds every file it installs through an output stream whose
        // commit step is "fsync, then rename into place"; the fsync failed, so it treated the whole
        // stream as failed and left an empty file behind. a durability primitive that reports failure
        // is worse than a slow one: everything that cares stops, and everything that does not care
        // never asked.
        case 82:
        // fsync
        case 83: {
          const syncEntry = this.getFdEntry(Number(a0));
          if (!syncEntry) {
            if (!this.vfs?.hasFd(Number(a0))) {
              ret = -BigInt(EBADF);
              break;
            }
            ret = this.vfs?.isCharDev?.(Number(a0)) ? -BigInt(EINVAL) : 0n;
            break;
          }
          const syncType = syncEntry.type === "native" ? syncEntry.ftype ?? 0 : 0;
          if (syncType === 4096 || syncType === 49152 || syncType === 8192) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (syncEntry.type === "native") {
            return {
              status: "bridge",
              method: "fs_fsync",
              args: { fd: syncEntry.nativeFd, data: nr === 83 },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          const syncNoData = syncEntry.type === "pipe_read" || syncEntry.type === "pipe_write" || syncEntry.type === "socket" || syncEntry.type === "unixsock" || syncEntry.type === "vfs" && !!this.vfs?.isCharDev?.(syncEntry.fd);
          ret = syncNoData ? -BigInt(EINVAL) : 0n;
          break;
        }
        // setitimer/getitimer duplicates check
        case 104:
          ret = -BigInt(EPERM);
          break;
        case 105:
        // ??? (105 = clock_adjtime_time64 — doesn't exist on aarch64)
        case 106:
          ret = -BigInt(ENOSYS);
          break;
          ret = -BigInt(EPERM);
          break;
        // Signal (remaining)
        // 137 rt_sigtimedwait is handled in the worker (it has to BLOCK, and only the worker can park
        // on the proc-table futex). it reaches here only in the host-side dispatch used by tests.
        case 137:
          ret = -BigInt(4);
          break;
        case 138:
          ret = 0n;
          break;
        // ioctl subcommand handling is in case 29 already
        case 142: {
          if (!this.hasCap(CAP_SYS_BOOT)) {
            ret = -BigInt(EPERM);
            break;
          }
          const rbMagic1 = Number(BigInt.asIntN(32, a0)) >>> 0;
          const rbMagic2 = Number(BigInt.asIntN(32, a1)) >>> 0;
          if (rbMagic1 !== 4276215469 || !REBOOT_MAGIC2.has(rbMagic2)) {
            ret = -BigInt(EINVAL);
            break;
          }
          const rbCmd = Number(BigInt.asIntN(32, a2)) >>> 0;
          switch (rbCmd) {
            // CAD_ON/CAD_OFF only choose what ctrl-alt-del does; they reboot nothing.
            case 2309737967:
              this.cadEnabled = true;
              ret = 0n;
              break;
            case 0:
              this.cadEnabled = false;
              ret = 0n;
              break;
            // the ones that actually stop the machine. our machine is the guest system, and stopping
            // it is what `halt` and `poweroff` are for -- so they do exactly that rather than
            // pretending. reboot(2) does not return when it succeeds.
            case 19088743:
            // RESTART
            case 3454992675:
            // HALT
            case 1126301404:
            // POWER_OFF
            case 2712847316:
              this.haltHook?.(rbCmd === 19088743 || rbCmd === 2712847316);
              return { status: "exit", code: 0 };
            case 3489725666:
            // SW_SUSPEND: there is no hibernation image to write
            case 1163412803:
              ret = -BigInt(EINVAL);
              break;
            default:
              ret = -BigInt(EINVAL);
          }
          break;
        }
        // getrlimit/setrlimit (prlimit64 at 261 is the modern replacement)
        case 161:
        // sethostname(name, len)
        case 162: {
          const utsLen = Number(BigInt.asIntN(64, a1));
          if (utsLen < 0 || utsLen > 64) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (this.ptrFaults(bytes, Number(a0), Math.max(utsLen, 1))) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (!this.hasCap(CAP_SYS_ADMIN)) {
            ret = -BigInt(EPERM);
            break;
          }
          const base = Number(a0) + GUEST_MEM_BASE;
          const name = new TextDecoder().decode(bytes.subarray(base, base + utsLen));
          this.utsHook?.set(nr === 161 ? "host" : "domain", name);
          ret = 0n;
          break;
        }
        // getrlimit/setrlimit are the older two-argument spellings of prlimit64, with the same
        // struct rlimit { u64 cur; u64 max; } on a 64-bit target. they used to return success
        // without doing anything: setrlimit silently ignored the new limit, so a process that
        // lowered its own ceiling was never held to it, and getrlimit left the caller's struct
        // untouched while claiming to have filled it.
        case 163: {
          const res = Number(BigInt.asIntN(32, a0));
          if (res < 0 || res >= RLIM_NLIMITS) {
            ret = -BigInt(EINVAL);
            break;
          }
          const ptr = Number(a1);
          if (ptr === 0 || this.ptrFaults(bytes, ptr, 16)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const lim = this.rlimitOf(res);
          view.setBigUint64(ptr + GUEST_MEM_BASE, lim.cur, true);
          view.setBigUint64(ptr + GUEST_MEM_BASE + 8, lim.max, true);
          ret = 0n;
          break;
        }
        case 164: {
          const res = Number(BigInt.asIntN(32, a0));
          if (res < 0 || res >= RLIM_NLIMITS) {
            ret = -BigInt(EINVAL);
            break;
          }
          const ptr = Number(a1);
          if (ptr === 0 || this.ptrFaults(bytes, ptr, 16)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const cur = view.getBigUint64(ptr + GUEST_MEM_BASE, true);
          const max = view.getBigUint64(ptr + GUEST_MEM_BASE + 8, true);
          if (cur > max) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (res === RLIMIT_NOFILE && max > BigInt(NR_OPEN)) {
            ret = -BigInt(EPERM);
            break;
          }
          if (max > this.rlimitOf(res).max && !this.hasCap(CAP_SYS_RESOURCE)) {
            ret = -BigInt(EPERM);
            break;
          }
          this.setRlimit(res, cur, max);
          ret = 0n;
          break;
        }
        case 171:
        // adjtimex(buf)
        case 266: {
          const axBuf = nr === 171 ? Number(a0) : Number(a1);
          if (nr === 266 && !this.knownClock(Number(BigInt.asIntN(32, a0)))) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (this.ptrFaults(bytes, axBuf, 208)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const axBase = axBuf + GUEST_MEM_BASE;
          const modes = view.getInt32(axBase, true);
          const ADJ_OFFSET = 1, ADJ_FREQUENCY = 2, ADJ_SETOFFSET = 256;
          const ADJ_NANO = 8192, ADJ_TICK = 16384, ADJ_ADJTIME = 32768;
          if (modes & ADJ_ADJTIME) {
            if (!(modes & ADJ_OFFSET)) {
              ret = -BigInt(EINVAL);
              break;
            }
            if (!this.hasCap(CAP_SYS_TIME)) {
              ret = -BigInt(EPERM);
              break;
            }
          } else if (modes !== 0) {
            if (!this.hasCap(CAP_SYS_TIME)) {
              ret = -BigInt(EPERM);
              break;
            }
            if (modes & ADJ_TICK) {
              const tick = view.getBigInt64(axBase + 88, true);
              if (tick < 9000n || tick > 11000n) {
                ret = -BigInt(EINVAL);
                break;
              }
            }
          }
          if (modes & ADJ_SETOFFSET) {
            const secOff = view.getBigInt64(axBase + 72, true);
            const fracOff = view.getBigInt64(axBase + 80, true);
            const fracMax = modes & ADJ_NANO ? 1000000000n : 1000000n;
            if (fracOff < 0n || fracOff >= fracMax) {
              ret = -BigInt(EINVAL);
              break;
            }
            this.clockOffsetNs = this.clockOffsetNs + secOff * 1000000000n + fracOff * (modes & ADJ_NANO ? 1n : 1000n);
          }
          if (modes & ADJ_FREQUENCY) {
            const freq = view.getBigInt64(axBase + 16, true);
            if (freq < -32768000n * 65536n || freq > 32768000n * 65536n) {
              ret = -BigInt(EINVAL);
              break;
            }
          }
          const nowNs = this.clockNs(0);
          view.setBigInt64(axBase + 8, 0n, true);
          view.setBigInt64(axBase + 16, 0n, true);
          view.setBigInt64(axBase + 24, 500000n, true);
          view.setBigInt64(axBase + 32, 0n, true);
          view.setInt32(axBase + 40, 64, true);
          view.setBigInt64(axBase + 48, 0n, true);
          view.setBigInt64(axBase + 56, 0n, true);
          view.setBigInt64(axBase + 64, 500000n, true);
          view.setBigInt64(axBase + 72, nowNs / 1000000000n, true);
          view.setBigInt64(axBase + 80, nowNs % 1000000000n / 1000n, true);
          view.setBigInt64(axBase + 88, 10000n, true);
          ret = 5n;
          break;
        }
        // posix message queues (mq_open, mq_unlink, mq_timedsend, mq_timedreceive, mq_notify,
        // mq_getsetattr) are handled by the worker: the queues are named system-wide, so they need
        // the kernel round trip only the worker can make. reaching here means there is no kernel.
        case 180:
        case 181:
        case 182:
        case 183:
        case 184:
        case 185:
          ret = -BigInt(ENOSYS);
          break;
        // SysV IPC
        // 186-197 (msg*, sem*, shm*) are all handled in the worker: every one of these registries is
        // system-wide, so they need a round trip to the kernel that only the worker can make. these
        // cases are unreachable from a real guest and exist so a stray number still gets an answer.
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
        // 218 and 219 are request_key and keyctl, not the NUMA policy calls they were labelled as
        // (those are 235..237). answering 0 meant a program asking the kernel keyring for a key was
        // told it had succeeded and then read back nothing.
        case 218:
        // request_key
        case 219:
          ret = -BigInt(ENOSYS);
          break;
        case 223: {
          const faFd = Number(a0);
          const faAdvice = Number(BigInt.asIntN(32, a3));
          if (!this.fdExists(faFd)) {
            ret = -BigInt(EBADF);
            break;
          }
          const faEntry = this.fdTable.get(faFd);
          if (this.io.pipeReady?.(faFd) || faEntry?.type === "pipe_read" || faEntry?.type === "pipe_write" || faEntry?.type === "socket" || faEntry?.type === "unixsock" || faEntry?.type === "tty" || faEntry?.type === "stdio" || faEntry?.type === "eventfd" || faEntry?.type === "epoll" || faEntry?.type === "inotify" || faEntry?.type === "timerfd" || faEntry?.type === "jitterctl") {
            ret = -BigInt(ESPIPE);
            break;
          }
          if (faAdvice < 0 || faAdvice > 5) {
            ret = -BigInt(EINVAL);
            break;
          }
          ret = 0n;
          break;
        }
        case 224:
        // swapon
        case 225:
          ret = -BigInt(EPERM);
          break;
        case 234:
          ret = -BigInt(ENOSYS);
          break;
        // 235..239 are the NUMA calls: mbind, get_mempolicy, set_mempolicy, migrate_pages,
        // move_pages. they were labelled membarrier/mlock2/copy_file_range/preadv2/pwritev2, which
        // are 283..287 -- and 236 in particular did not merely answer the wrong question, it CALLED
        // setLocked() and mlocked a range of the caller's address space on a get_mempolicy.
        case 235:
        // mbind - a placement policy for a range. one node, so nothing to place.
        case 237:
          ret = 0n;
          break;
        case 236: {
          if (Number(a0) !== 0) {
            if (this.ptrFaults(bytes, Number(a0), 4)) {
              ret = -BigInt(EFAULT);
              break;
            }
            view.setInt32(Number(a0) + GUEST_MEM_BASE, 0, true);
          }
          if (Number(a1) !== 0 && Number(a2) > 0) {
            if (this.ptrFaults(bytes, Number(a1), 8)) {
              ret = -BigInt(EFAULT);
              break;
            }
            view.setBigUint64(Number(a1) + GUEST_MEM_BASE, 1n, true);
          }
          ret = 0n;
          break;
        }
        case 238:
        // migrate_pages
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
        case 264: {
          const nhFlags = Number(BigInt.asIntN(32, a4));
          const AT_EMPTY_PATH_NH = 4096, AT_SYMLINK_FOLLOW_NH = 1024, AT_HANDLE_FID = 512, AT_HANDLE_MNT_ID_UNIQUE = 1;
          if (nhFlags & ~(AT_EMPTY_PATH_NH | AT_SYMLINK_FOLLOW_NH | AT_HANDLE_FID | AT_HANDLE_MNT_ID_UNIQUE)) {
            ret = -BigInt(EINVAL);
            break;
          }
          const nhHandle = Number(a2);
          if (nhHandle === 0 || this.ptrWriteFaults(bytes, nhHandle, 8)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const nhRoom = view.getUint32(nhHandle + GUEST_MEM_BASE, true);
          const NH_SIZE = 8;
          const MAX_HANDLE_SZ = 128;
          if (nhRoom > MAX_HANDLE_SZ) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (Number(a3) !== 0 && this.ptrWriteFaults(bytes, Number(a3), 4)) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (this.ptrFaults(bytes, Number(a1))) {
            ret = -BigInt(EFAULT);
            break;
          }
          const nhRaw = readGuestString(bytes, Number(a1));
          if (this.isEmptyPath(nhRaw) && !(nhFlags & AT_EMPTY_PATH_NH)) {
            ret = -BigInt(ENOENT);
            break;
          }
          if (!this.isEmptyPath(nhRaw)) {
            const pe = this.pathCheck(nhRaw);
            if (pe) {
              ret = -BigInt(pe);
              break;
            }
          }
          if (!this.isEmptyPath(nhRaw)) {
            const de = this.atDirfdError(a0, nhRaw);
            if (de) {
              ret = -BigInt(de);
              break;
            }
          }
          const nhDfd = Number(BigInt.asIntN(32, a0));
          const nhPath = !this.isEmptyPath(nhRaw) ? this.resolveAt(a0, nhRaw) : nhDfd === -100 ? this.vfs?.getcwd() ?? this.cwd : this.pathOfFd(nhDfd);
          if (nhPath === null) {
            ret = -BigInt(EBADF);
            break;
          }
          const nhBridge = this.bridgePath(nhPath);
          if (nhBridge === null) {
            ret = -BigInt(EOPNOTSUPP);
            break;
          }
          return {
            status: "bridge",
            method: "fs_stat",
            args: { path: nhBridge, nofollow: (nhFlags & AT_SYMLINK_FOLLOW_NH) === 0 },
            apply: (resp) => {
              if (resp.ret < 0) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                return;
              }
              if (nhRoom < NH_SIZE) {
                view.setUint32(nhHandle + GUEST_MEM_BASE, NH_SIZE, true);
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(-75)));
                return;
              }
              const ino = resp._meta ? Math.round(resp._meta[6]) : 0;
              view.setUint32(nhHandle + GUEST_MEM_BASE, NH_SIZE, true);
              view.setInt32(nhHandle + 4 + GUEST_MEM_BASE, 1, true);
              view.setBigUint64(nhHandle + 8 + GUEST_MEM_BASE, BigInt(ino), true);
              if (Number(a3) !== 0) view.setInt32(Number(a3) + GUEST_MEM_BASE, NATIVE_FS_MOUNT_ID, true);
              this.handleHook?.put(ino, nhBridge);
              cpu2.setReg(0, 0n);
            }
          };
        }
        case 265: {
          const ohMountFd = Number(BigInt.asIntN(32, a0));
          if (ohMountFd !== -100 && !this.fdIsOpen(ohMountFd)) {
            ret = -BigInt(EBADF);
            break;
          }
          const ohHandle = Number(a1);
          if (ohHandle === 0 || this.ptrFaults(bytes, ohHandle, 8)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const ohBytes = view.getUint32(ohHandle + GUEST_MEM_BASE, true);
          const ohType = view.getInt32(ohHandle + 4 + GUEST_MEM_BASE, true);
          if (ohBytes !== 8 || ohType !== 1) {
            ret = -BigInt(22);
            break;
          }
          if (this.ptrFaults(bytes, ohHandle, 16)) {
            ret = -BigInt(EFAULT);
            break;
          }
          if (!this.hasCap(CAP_DAC_READ_SEARCH)) {
            ret = -BigInt(EPERM);
            break;
          }
          if (ohMountFd !== -100) {
            const mEnt = this.getFdEntry(ohMountFd);
            if (mEnt?.type !== "native") {
              ret = -BigInt(116);
              break;
            }
          }
          const ohIno = Number(view.getBigUint64(ohHandle + 8 + GUEST_MEM_BASE, true));
          const ohPath = this.handleHook?.get(ohIno) ?? null;
          if (ohPath === null) {
            ret = -BigInt(116);
            break;
          }
          const ohFd = this.allocFd();
          const ohFlags = Number(BigInt.asIntN(32, a2));
          return {
            status: "bridge",
            method: "fs_open",
            // a handle names an inode, and a symlink's inode is the LINK: there is no following it
            // from here, so opening one is ELOOP -- which is exactly what O_NOFOLLOW answers.
            args: { path: ohPath, flags: ohFlags | 32768, mode: 0 },
            apply: (resp) => {
              if (resp.ret < 0) {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                return;
              }
              this.fdTable.set(ohFd, {
                type: "native",
                nativeFd: resp.ret,
                path: ohPath,
                ofdId: this.newOfdIdPublic()
              });
              this.nativeRefs.set(resp.ret, 1);
              cpu2.setReg(0, BigInt(ohFd));
            }
          };
        }
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
        case 275:
          ret = -BigInt(ENOSYS);
          break;
        case 280:
          ret = -BigInt(ENOSYS);
          break;
        case 282:
          ret = -BigInt(ENOSYS);
          break;
        case 283: {
          const MB_QUERY = 0, MB_GLOBAL = 1, MB_GLOBAL_EXPEDITED = 2, MB_REG_GLOBAL_EXPEDITED = 4, MB_PRIVATE_EXPEDITED = 8, MB_REG_PRIVATE_EXPEDITED = 16, MB_PRIVATE_EXPEDITED_SYNC_CORE = 32, MB_REG_PRIVATE_EXPEDITED_SYNC_CORE = 64;
          const MB_SUPPORTED = MB_GLOBAL | MB_GLOBAL_EXPEDITED | MB_REG_GLOBAL_EXPEDITED | MB_PRIVATE_EXPEDITED | MB_REG_PRIVATE_EXPEDITED | MB_PRIVATE_EXPEDITED_SYNC_CORE | MB_REG_PRIVATE_EXPEDITED_SYNC_CORE;
          const mbCmd = Number(BigInt.asIntN(32, a0));
          if (Number(BigInt.asIntN(32, a1)) !== 0) {
            ret = -BigInt(EINVAL);
            break;
          }
          switch (mbCmd) {
            case MB_QUERY:
              ret = BigInt(MB_SUPPORTED);
              break;
            case MB_GLOBAL:
            case MB_GLOBAL_EXPEDITED:
            case MB_REG_GLOBAL_EXPEDITED:
              ret = 0n;
              break;
            case MB_REG_PRIVATE_EXPEDITED:
              this.membarrierRegistered |= 1;
              ret = 0n;
              break;
            case MB_REG_PRIVATE_EXPEDITED_SYNC_CORE:
              this.membarrierRegistered |= 2;
              ret = 0n;
              break;
            // an expedited private barrier is only for a process that said it wanted one.
            case MB_PRIVATE_EXPEDITED:
              ret = this.membarrierRegistered & 1 ? 0n : -BigInt(EPERM);
              break;
            case MB_PRIVATE_EXPEDITED_SYNC_CORE:
              ret = this.membarrierRegistered & 2 ? 0n : -BigInt(EPERM);
              break;
            default:
              ret = -BigInt(EINVAL);
              break;
          }
          break;
        }
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
        case 436: {
          const CLOSE_RANGE_UNSHARE = 2, CLOSE_RANGE_CLOEXEC = 4;
          const crFlags = Number(BigInt.asUintN(32, a2));
          if (crFlags & ~(CLOSE_RANGE_UNSHARE | CLOSE_RANGE_CLOEXEC)) {
            ret = -BigInt(EINVAL);
            break;
          }
          const crFirst = Number(BigInt.asUintN(32, a0));
          const crLast = Number(BigInt.asUintN(32, a1));
          if (crFirst > crLast) {
            ret = -BigInt(EINVAL);
            break;
          }
          const crEnd = Math.min(crLast, MAX_FDS);
          for (let fd = crFirst; fd <= crEnd; fd++) {
            if (!this.fdIsOpen(fd)) continue;
            if (crFlags & CLOSE_RANGE_CLOEXEC) {
              this.fdCloexec.add(fd);
              continue;
            }
            this.closeOne(fd);
          }
          ret = 0n;
          break;
        }
        case 438:
          ret = -BigInt(ENOSYS);
          break;
        case 439: {
          const AT_EACCESS = 512, AT_EMPTY_PATH = 4096;
          const fa2Flags = Number(BigInt.asIntN(32, a3));
          if (fa2Flags & ~(AT_EACCESS | AT_SYMLINK_NOFOLLOW | AT_EMPTY_PATH)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (Number(BigInt.asUintN(32, a2)) & ~7) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (this.ptrFaults(bytes, Number(a1))) {
            ret = -BigInt(EFAULT);
            break;
          }
          const rawFa2 = readGuestString(bytes, Number(a1));
          if (this.isEmptyPath(rawFa2)) {
            if (!(fa2Flags & AT_EMPTY_PATH)) {
              ret = -BigInt(ENOENT);
              break;
            }
          } else {
            const peFa2 = this.pathCheck(rawFa2);
            if (peFa2) {
              ret = -BigInt(peFa2);
              break;
            }
          }
          const fa2Path = this.resolveAt(a0, rawFa2);
          const bpFa2 = this.bridgePath(fa2Path);
          if (bpFa2 !== null) {
            return {
              status: "bridge",
              method: "fs_faccessat",
              // AT_EACCESS asks the question of the EFFECTIVE ids rather than the real ones, which
              // is the difference between "may this program reach it" and "may the user it runs for".
              args: { path: bpFa2, mode: Number(a2), eaccess: (fa2Flags & AT_EACCESS) !== 0 },
              apply: (resp) => {
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          if (this.vfs) {
            ret = BigInt(this.vfs.faccessat(Number(BigInt.asIntN(32, a0)), fa2Path, Number(a2), fa2Flags));
          } else {
            ret = -BigInt(ENOENT);
          }
          break;
        }
        case 440:
          ret = -BigInt(ENOSYS);
          break;
        case 441: {
          let ep2Timeout = -1;
          if (Number(a3) !== 0) {
            if (this.ptrFaults(bytes, Number(a3), 16)) {
              ret = -BigInt(EFAULT);
              break;
            }
            const tsSec = Number(view.getBigInt64(Number(a3) + GUEST_MEM_BASE, true));
            const tsNsec = Number(view.getBigInt64(Number(a3) + GUEST_MEM_BASE + 8, true));
            if (tsSec < 0 || tsNsec < 0 || tsNsec >= 1e9) {
              ret = -BigInt(EINVAL);
              break;
            }
            ep2Timeout = tsSec * 1e3 + Math.ceil(tsNsec / 1e6);
          }
          const ep2Sigmask = Number(a4);
          if (ep2Sigmask !== 0 && this.ptrFaults(bytes, ep2Sigmask, 8)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const ep2Saved = ep2Sigmask !== 0 ? this.signalMask ?? 0n : null;
          if (ep2Sigmask !== 0) this.setSignalMask(view.getBigUint64(ep2Sigmask + GUEST_MEM_BASE, true));
          ret = BigInt(this.epollWait(
            view,
            bytes,
            Number(a0),
            Number(a1),
            Number(BigInt.asIntN(32, a2)),
            ep2Timeout
          ));
          if (ep2Saved !== null) this.setSignalMask(ep2Saved);
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
          const AT_SYMLINK_NOFOLLOW_452 = 256, AT_EMPTY_PATH_452 = 4096;
          const flags452 = Number(BigInt.asIntN(32, a3));
          if (flags452 & ~(AT_SYMLINK_NOFOLLOW_452 | AT_EMPTY_PATH_452)) {
            ret = -BigInt(EINVAL);
            break;
          }
          if (this.ptrFaults(bytes, Number(a1), 1)) {
            ret = -BigInt(EFAULT);
            break;
          }
          const raw452 = readGuestString(bytes, Number(a1));
          {
            const pe = this.pathCheck(raw452);
            if (pe) {
              ret = -BigInt(pe);
              break;
            }
          }
          {
            const de = this.atDirfdError(a0, raw452);
            if (de) {
              ret = -BigInt(de);
              break;
            }
          }
          const path452 = this.resolveAt(a0, raw452);
          const mode452 = Number(BigInt.asUintN(32, a2)) & 4095;
          const bp452 = this.bridgePath(path452);
          if (bp452 !== null) {
            this.invalidateStat(bp452);
            return {
              status: "bridge",
              method: "fs_chmod",
              args: { path: bp452, mode: mode452, nofollow: flags452 & AT_SYMLINK_NOFOLLOW_452 ? 1 : 0 },
              apply: (resp) => {
                if (resp.ret >= 0) this.notify(bp452, IN_ATTRIB);
                cpu2.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              }
            };
          }
          ret = this.vfs ? BigInt(this.vfs.chmodPath(path452, mode452)) : -BigInt(ENOENT);
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
          ret = -BigInt(ENOSYS);
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
      this.noteResult(nr, ret);
      cpu2.setReg(0, BigInt.asUintN(64, ret));
      return { status: "ok" };
    }
  };
  var AT_FLAG_ARGS = {
    35: [2, 512],
    // unlinkat: AT_REMOVEDIR
    37: [4, 1024 | 4096],
    // linkat: AT_SYMLINK_FOLLOW | AT_EMPTY_PATH
    54: [4, 256 | 4096],
    // fchownat: AT_SYMLINK_NOFOLLOW | AT_EMPTY_PATH
    79: [3, 256 | 2048 | 4096],
    // fstatat
    88: [3, 256],
    // utimensat: AT_SYMLINK_NOFOLLOW
    291: [2, 256 | 2048 | 4096 | 24576],
    // statx
    439: [3, 256 | 512 | 4096]
    // faccessat2: NOFOLLOW | EACCESS | EMPTY_PATH
  };
  var DIRFD_ARG_REGS = {
    33: [[0, 1]],
    // mknodat
    34: [[0, 1]],
    // mkdirat
    35: [[0, 1]],
    // unlinkat
    36: [[1, 2]],
    // symlinkat(target, newdirfd, linkpath)
    37: [[0, 1], [2, 3]],
    // linkat
    38: [[0, 1], [2, 3]],
    // renameat
    48: [[0, 1]],
    // faccessat
    53: [[0, 1]],
    // fchmodat
    54: [[0, 1]],
    // fchownat
    56: [[0, 1]],
    // openat
    78: [[0, 1]],
    // readlinkat
    79: [[0, 1]],
    // fstatat
    88: [[0, 1]],
    // utimensat
    276: [[0, 1], [2, 3]],
    // renameat2
    281: [[0, 1]],
    // execveat
    291: [[0, 1]],
    // statx
    437: [[0, 1]],
    // openat2
    439: [[0, 1]]
    // faccessat2
  };
  var PATH_ARG_REGS = {
    5: [0],
    6: [0],
    // setxattr, lsetxattr
    8: [0],
    9: [0],
    // getxattr, lgetxattr
    11: [0],
    12: [0],
    // listxattr, llistxattr
    14: [0],
    15: [0],
    // removexattr, lremovexattr
    33: [1],
    // mknodat
    34: [1],
    // mkdirat
    35: [1],
    // unlinkat
    36: [0, 2],
    // symlinkat(target, newdirfd, linkpath)
    37: [1, 3],
    // linkat
    38: [1, 3],
    // renameat
    45: [0],
    // truncate
    48: [1],
    // faccessat
    49: [0],
    // chdir
    53: [1],
    // fchmodat
    54: [1],
    // fchownat
    56: [1],
    // openat
    78: [1],
    // readlinkat
    79: [1],
    // fstatat
    221: [0],
    // execve
    276: [1, 3],
    // renameat2
    279: [0],
    // memfd_create (a name, not a path, but read the same way)
    281: [1],
    // execveat
    291: [1],
    // statx
    437: [1],
    // openat2
    439: [1]
    // faccessat2
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
  function writeChardevStat(view, guestAddr, rdev = 0) {
    const base = guestAddr + GUEST_MEM_BASE;
    for (let i = 0; i < 128; i += 8) {
      view.setBigUint64(base + i, 0n, true);
    }
    view.setUint32(base + 16, 8630, true);
    view.setUint32(base + 20, 1, true);
    if (rdev !== 0) view.setBigUint64(base + 32, BigInt(rdev), true);
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
    const rdev = meta ? meta[7] ?? 0 : resp.rdev ?? 0;
    const atime = meta ? meta[8] ?? 0 : resp.atime ?? 0;
    const ctime = meta ? meta[9] ?? 0 : resp.ctime ?? 0;
    const mtimeNsec = meta ? meta[10] ?? 0 : 0;
    const atimeNsec = meta ? meta[11] ?? 0 : 0;
    const ctimeNsec = meta ? meta[12] ?? 0 : 0;
    const blocks = meta ? meta[13] ?? 0 : 0;
    const blksize = meta ? meta[14] ?? 4096 : 4096;
    view.setBigUint64(base + 0, 1n, true);
    view.setBigUint64(base + 8, BigInt.asUintN(64, BigInt(Math.trunc(ino))), true);
    view.setUint32(base + 16, mode, true);
    view.setUint32(base + 20, nlink, true);
    view.setUint32(base + 24, uid, true);
    view.setUint32(base + 28, gid, true);
    view.setBigUint64(base + 32, BigInt.asUintN(64, BigInt(Math.trunc(rdev))), true);
    view.setBigUint64(base + 48, BigInt(size), true);
    view.setUint32(base + 56, blksize || 4096, true);
    view.setBigUint64(base + 64, BigInt(Math.trunc(blocks)), true);
    view.setBigInt64(base + 72, BigInt(Math.trunc(atime)), true);
    view.setBigInt64(base + 80, BigInt(Math.trunc(atimeNsec)), true);
    view.setBigUint64(base + 88, BigInt(mtime), true);
    view.setBigInt64(base + 96, BigInt(Math.trunc(mtimeNsec)), true);
    view.setBigInt64(base + 104, BigInt(Math.trunc(ctime)), true);
    view.setBigInt64(base + 112, BigInt(Math.trunc(ctimeNsec)), true);
  }

  // src/binfmt.ts
  var BINPRM_BUF_SIZE = 256;
  var MAX_INTERP_DEPTH = 4;
  function isElf(bytes) {
    return bytes.length >= 4 && bytes[0] === 127 && bytes[1] === 69 && bytes[2] === 76 && bytes[3] === 70;
  }
  function parseShebang(bytes) {
    if (bytes.length < 2 || bytes[0] !== 35 || bytes[1] !== 33) return null;
    const limit = Math.min(bytes.length, BINPRM_BUF_SIZE);
    let end = 2;
    while (end < limit && bytes[end] !== 10) end++;
    const line = new TextDecoder().decode(bytes.subarray(2, end)).trim();
    if (!line) return null;
    const sp = line.search(/\s/);
    if (sp < 0) return { interp: line };
    const arg = line.slice(sp).trim();
    return arg ? { interp: line.slice(0, sp), arg } : { interp: line.slice(0, sp) };
  }
  var ExecError = class extends Error {
    constructor(message, errno) {
      super(message);
      this.errno = errno;
    }
  };
  function resolveExec(path, argv, read) {
    let bytes = read(path);
    if (!bytes) throw new ExecError(`exec: cannot read ${path}`, 2);
    for (let depth = 0; ; depth++) {
      const sb = parseShebang(bytes);
      if (!sb) break;
      if (depth >= MAX_INTERP_DEPTH) throw new ExecError(`exec: interpreter loop at ${path}`, 40);
      const next = [sb.interp];
      if (sb.arg) next.push(sb.arg);
      next.push(path, ...argv.slice(1));
      path = sb.interp;
      argv = next;
      bytes = read(path);
      if (!bytes) throw new ExecError(`exec: interpreter ${path} not found`, 2);
    }
    if (!isElf(bytes)) throw new ExecError(`exec: not an ELF file: ${path}`, 8);
    return { path, argv, bytes };
  }

  // src/vfs.ts
  var EPERM2 = 1;
  var ENOENT2 = 2;
  var ENXIO = 6;
  var EBADF2 = 9;
  var EAGAIN2 = 11;
  var EACCES2 = 13;
  var EEXIST2 = 17;
  var ENOTDIR2 = 20;
  var EISDIR2 = 21;
  var EINVAL2 = 22;
  var EMFILE2 = 24;
  var ESPIPE2 = 29;
  var ENAMETOOLONG2 = 36;
  var ENOTEMPTY = 39;
  var ELOOP2 = 40;
  var O_ACCMODE = 3;
  var O_RDONLY = 0;
  var O_WRONLY = 1;
  var O_RDWR = 2;
  var O_CREAT = 64;
  var O_EXCL = 128;
  var O_TRUNC = 512;
  var O_APPEND = 1024;
  var O_NONBLOCK2 = 2048;
  var O_DIRECTORY = 16384;
  var O_NOFOLLOW = 32768;
  var O_PATH = 2097152;
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
  function gzipStored(data) {
    const blocks = Math.max(1, Math.ceil(data.length / 65535));
    const out = new Uint8Array(10 + blocks * 5 + data.length + 8);
    out.set([31, 139, 8, 0, 0, 0, 0, 0, 0, 3], 0);
    let o = 10;
    for (let i = 0; i < blocks; i++) {
      const start = i * 65535;
      const len = Math.min(65535, data.length - start);
      out[o++] = i === blocks - 1 ? 1 : 0;
      out[o++] = len & 255;
      out[o++] = len >> 8 & 255;
      out[o++] = ~len & 255;
      out[o++] = ~len >> 8 & 255;
      out.set(data.subarray(start, start + len), o);
      o += len;
    }
    let crc = 4294967295;
    for (let i = 0; i < data.length; i++) {
      crc ^= data[i];
      for (let b = 0; b < 8; b++) crc = crc >>> 1 ^ 3988292384 & -(crc & 1);
    }
    crc = (crc ^ 4294967295) >>> 0;
    const dv = new DataView(out.buffer);
    dv.setUint32(o, crc, true);
    dv.setUint32(o + 4, data.length >>> 0, true);
    return out;
  }
  function guestCpus() {
    const n = globalThis.navigator?.hardwareConcurrency;
    return Math.max(1, Math.min(64, typeof n === "number" && n > 0 ? Math.floor(n) : 1));
  }
  var PAGEMAP_PAGES = 1610612736 / 4096;
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
    // this process's own cpu accounting, live. the shared table is only refreshed periodically, and
    // a program that spins on its own utime waiting for it to become nonzero needs the exact number
    // rather than the last one published.
    selfCpuProvider = null;
    // path-resolution search-permission state. resolve() enforces +x (search) on each directory
    // component; searchDenied records when a walk failed on a permission (EACCES) vs a missing
    // entry (ENOENT). searchEff selects effective vs real creds for the walk (access() uses real
    // by default, most syscalls use effective). set per-resolve.
    searchDenied = false;
    searchEff = true;
    // errno for the most recent failed resolve() (ENOENT/ENOTDIR/EACCES/ELOOP/ENAMETOOLONG).
    // callers that get a null inode use resolveErrno() to return the correct negative errno.
    lastResolveErrno = ENOENT2;
    // locked-memory accounting (kb) reported via /proc/self/status VmLck. updated by the syscall
    // handler on mmap(MAP_LOCKED)/mlock/munlock/munmap of locked regions.
    vmLckKb = 0;
    // lazy /proc/self/smaps, alongside maps: the same walk with what each mapping costs.
    procSmapsGen = null;
    // lazy /proc/self/maps: the syscall handler registers a renderer that walks its live vma set
    // only when the file is actually read, so mapping-heavy workloads don't pay per-mmap rendering.
    procMapsGen = null;
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
    makeCharDev(major, minor, mode) {
      const t = this.now();
      return { type: "chardev", ino: this.nextIno++, mode: S_IFCHR | mode & 4095, nlink: 1, uid: 0, gid: 0, atime: t, mtime: t, ctime: t, major, minor };
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
      if (dir.children.has(name)) return -EEXIST2;
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
      this.mkdirp("/proc/sys/vm");
      this.addCharDev("/dev/null", 1, 3, 438);
      this.addCharDev("/dev/zero", 1, 5, 438);
      this.addCharDev("/dev/full", 1, 7, 438);
      this.addCharDev("/dev/random", 1, 8, 292);
      this.addCharDev("/dev/urandom", 1, 9, 292);
      this.addCharDev("/dev/tty", 5, 0, 438);
      this.addCharDev("/dev/jitter", 240, 0, 384);
      this.addSymlinkNode("/proc/self/exe", "/bin/busybox");
      this.addFileContent(
        "/proc/meminfo",
        "MemTotal:       1048576 kB\nMemFree:         524288 kB\nMemAvailable:    786432 kB\nBuffers:          65536 kB\nCached:          131072 kB\nSwapTotal:            0 kB\nSwapFree:             0 kB\n"
      );
      this.addFileContent(
        "/proc/cpuinfo",
        Array.from({ length: guestCpus() }, (_, i) => `processor	: ${i}
BogoMIPS	: 48.00
Features	: fp asimd aes sha1 sha2 crc32
CPU implementer	: 0x61
CPU architecture: 8
CPU variant	: 0x1
CPU part	: 0x000
`).join("\n")
      );
      this.addFileContent("/proc/sys/kernel/random/entropy_avail", "4096\n");
      this.addFileContent("/proc/sys/kernel/random/poolsize", "4096\n");
      this.addFileContent("/proc/sys/kernel/pid_max", "32768\n");
      this.addFileContent("/proc/sys/kernel/sched_rr_timeslice_ms", "100\n");
      this.addFileContent("/proc/sys/kernel/msg_next_id", "-1\n");
      this.addFileContent("/proc/sys/kernel/sem_next_id", "-1\n");
      this.addFileContent("/proc/sys/kernel/shm_next_id", "-1\n");
      this.addFileContent("/proc/sys/kernel/sem", "32000	1024000000	500	32000\n");
      this.addFileContent("/proc/self/timerslack_ns", "50000\n");
      this.addFileContent("/proc/sys/kernel/threads-max", "248\n");
      this.addFileContent("/proc/self/coredump_filter", "00000033\n");
      this.addFileContent("/proc/sys/kernel/msgmni", "32000\n");
      this.addFileContent("/proc/sys/kernel/msgmax", "8192\n");
      this.addFileContent("/proc/sys/kernel/msgmnb", "16384\n");
      this.addFileContent("/proc/sys/kernel/shmmax", "18446744073692774399\n");
      this.addFileContent("/proc/sys/kernel/shmall", "18446744073692774399\n");
      this.addFileContent("/proc/sys/kernel/shmmni", "4096\n");
      this.addFileContent("/proc/sys/kernel/core_pattern", "core\n");
      this.addFileContent("/proc/sys/kernel/tainted", "0\n");
      this.mkdirp("/proc/sys/fs/mqueue");
      this.addFileContent("/proc/sys/fs/mqueue/msg_max", "10\n");
      this.addFileContent("/proc/sys/fs/mqueue/msgsize_max", "8192\n");
      this.addFileContent("/proc/sys/fs/mqueue/queues_max", "256\n");
      this.addFileContent("/proc/sys/fs/pipe-user-pages-soft", `${PIPE_PAGES_SOFT}
`);
      this.mkdirp("/proc/sys/fs/inotify");
      this.addFileContent("/proc/sys/fs/inotify/max_queued_events", "16384\n");
      this.addFileContent("/proc/sys/fs/inotify/max_user_instances", "128\n");
      this.addFileContent("/proc/sys/fs/inotify/max_user_watches", "65536\n");
      this.addFileContent("/proc/sys/fs/pipe-max-size", "65536\n");
      this.addFileContent("/proc/sys/kernel/printk", "7	4	1	7\n");
      this.addFileContent("/proc/sys/vm/max_map_count", "65530\n");
      this.addFileContent("/proc/self/pagemap", "");
      this.setRangedFile("/proc/self/pagemap", PAGEMAP_PAGES * 8, (offset, len) => {
        const out = new Uint8Array(len);
        const dv = new DataView(out.buffer);
        const first = Math.floor(offset / 8);
        const limit = this.memSizeProvider ? Math.floor(this.memSizeProvider() / 4096) : 0;
        for (let i = 0; i * 8 < len; i++) {
          const page = first + i;
          if (page >= limit) continue;
          dv.setBigUint64(i * 8, 1n << 63n | BigInt(page), true);
        }
        return out;
      });
      this.addFileContent("/proc/self/status", "");
      this.setDynamicFile("/proc/self/status", () => new TextEncoder().encode(this.procStatus(this.selfPid)));
      this.addFileContent("/proc/self/syscall", "");
      this.setDynamicFile("/proc/self/syscall", () => new TextEncoder().encode(this.procSyscall(this.selfPid)));
      this.addFileContent("/proc/self/stat", "");
      this.setDynamicFile("/proc/self/stat", () => new TextEncoder().encode(this.procStat(this.selfPid)));
      this.addFileContent("/proc/self/cmdline", "");
      this.setDynamicFile("/proc/self/cmdline", () => new TextEncoder().encode(this.procOf(this.selfPid)?.cmdline ?? ""));
      this.addFileContent("/proc/self/comm", "");
      this.setDynamicFile("/proc/self/comm", () => new TextEncoder().encode((this.procOf(this.selfPid)?.comm ?? "") + "\n"));
      this.addFileContent("/proc/self/maps", "");
      this.setDynamicFile("/proc/self/maps", () => new TextEncoder().encode(this.procMapsGen ? this.procMapsGen() : ""));
      this.addFileContent("/proc/self/smaps", "");
      this.setDynamicFile("/proc/self/smaps", () => new TextEncoder().encode(this.procSmapsGen ? this.procSmapsGen() : ""));
      const mountTable = "tmpfs / tmpfs rw,relatime 0 0\nproc /proc proc rw,relatime 0 0\n";
      this.addFileContent("/proc/mounts", mountTable);
      this.addFileContent("/proc/self/mounts", mountTable);
      this.addFileContent("/proc/self/mountinfo", "1 0 0:1 / / rw,relatime - tmpfs tmpfs rw\n");
      this.addFileContent("/proc/filesystems", "nodev	tmpfs\n");
      this.addFileContent("/proc/version", "Linux version 6.1.0 (jitter@localhost) #1 SMP aarch64\n");
      this.addFileContent("/proc/cmdline", "console=hvc0\n");
      this.addFileContent("/proc/uptime", "0.00 0.00\n");
      this.mkdirp("/proc/sysvipc");
      for (const which of ["sem", "msg", "shm"]) {
        this.addFileContent(`/proc/sysvipc/${which}`, "");
        this.setDynamicFile(
          `/proc/sysvipc/${which}`,
          () => new TextEncoder().encode(this.sysvipcProvider ? this.sysvipcProvider(which) : "")
        );
      }
      this.setFileData("/proc/config.gz", gzipStored(new TextEncoder().encode(
        "#\n# jitter emulated kernel (arm64) config\n#\n" + [
          "CONFIG_ARM64",
          "CONFIG_64BIT",
          "CONFIG_MMU",
          "CONFIG_FUTEX",
          "CONFIG_EPOLL",
          "CONFIG_EVENTFD",
          "CONFIG_TIMERFD",
          "CONFIG_INOTIFY_USER",
          "CONFIG_SIGNALFD",
          "CONFIG_POSIX_TIMERS",
          "CONFIG_POSIX_MQUEUE",
          "CONFIG_PROC_FS",
          "CONFIG_TMPFS",
          "CONFIG_DEVTMPFS",
          "CONFIG_MULTIUSER",
          "CONFIG_SYSVIPC",
          "CONFIG_FILE_LOCKING",
          "CONFIG_SHMEM",
          "CONFIG_ADVISE_SYSCALLS",
          "CONFIG_MEMBARRIER",
          // msgrcv(MSG_COPY) is the part of checkpoint/restore this kernel has.
          "CONFIG_CHECKPOINT_RESTORE"
        ].map((k) => k + "=y\n").join("")
      )), 292);
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
    /** wire a /proc/sys file to the setting it actually names: reads report the live value, writes
     *  change it (or report a negative errno). */
    setSysctl(path, get, set) {
      this.setDynamicFile(path, () => new TextEncoder().encode(get()));
      const parts = path.split("/").filter((p) => p.length > 0);
      const name = parts.pop();
      let dir = this.root;
      for (const part of parts) {
        const child = dir.children.get(part);
        if (!child || child.type !== "dir") return;
        dir = child;
      }
      const inode = dir.children.get(name);
      if (inode && inode.type === "file") inode.sysctlWrite = set;
    }
    /** how much guest memory exists, for the files that describe the address space. */
    memSizeProvider = null;
    setRangedFile(path, size, read) {
      const parts = path.split("/").filter((p) => p.length > 0);
      const name = parts.pop();
      let dir = this.root;
      for (const part of parts) {
        const child = dir.children.get(part);
        if (!child || child.type !== "dir") return;
        dir = child;
      }
      const inode = dir.children.get(name);
      if (inode && inode.type === "file") inode.ranged = { size, read };
    }
    setDynamicFile(path, gen) {
      const parts = path.split("/").filter((p) => p.length > 0);
      const name = parts.pop();
      let dir = this.root;
      for (const part of parts) {
        const child = dir.children.get(part);
        if (!child || child.type !== "dir") return;
        dir = child;
      }
      const inode = dir.children.get(name);
      if (inode && inode.type === "file") inode.dynamic = gen;
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
      this.mkdirp("/" + parts.join("/"));
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
    addCharDev(path, major, minor, mode) {
      const parts = path.split("/").filter((p) => p.length > 0);
      const name = parts.pop();
      let dir = this.root;
      for (const part of parts) {
        const child = dir.children.get(part);
        if (!child || child.type !== "dir") return;
        dir = child;
      }
      dir.children.set(name, this.makeCharDev(major, minor, mode));
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
    /** live process list, supplied by the worker (which owns the shared process table). */
    procListProvider = null;
    /** the text of /proc/sysvipc/<which>, straight from the kernel's registries. */
    sysvipcProvider = null;
    // what /proc/self/fd should contain right now. the fd table lives in the syscall layer, so the
    // directory can only be honest if that layer hands it over. it was an empty directory, which is a
    // lie with consequences: anything that counts its own descriptors by listing it (ltp's pipe07,
    // every "close all fds above 3" loop, lsof-shaped code) concluded it had none open.
    fdListProvider = null;
    /** which pid /proc/self refers to; the worker keeps it current across fork and exec. */
    selfPid = 1;
    /**
     * bring /proc's per-process directories in line with who is actually alive. the tree is a plain
     * inode structure, so the pid directories are materialised here rather than being faked at read
     * time -- that way stat, open, getdents and readlink all see the same thing with no special
     * cases. called on any walk into /proc, which is the only place it can be observed.
     */
    syncProcPids() {
      if (!this.procListProvider) return;
      const procDir = this.root.children.get("proc");
      if (!procDir || procDir.type !== "dir") return;
      const live = this.procListProvider();
      const want = new Set(live.map((p) => String(p.pid)));
      for (const name of [...procDir.children.keys()]) {
        if (/^\d+$/.test(name) && !want.has(name)) procDir.children.delete(name);
      }
      for (const p of live) {
        const dir = String(p.pid);
        if (procDir.children.has(dir)) continue;
        this.mkdirp(`/proc/${dir}`);
        this.addFileContent(`/proc/${dir}/stat`, "");
        this.setDynamicFile(`/proc/${dir}/stat`, () => new TextEncoder().encode(this.procStat(p.pid)));
        this.addFileContent(`/proc/${dir}/syscall`, "");
        this.setDynamicFile(`/proc/${dir}/syscall`, () => new TextEncoder().encode(this.procSyscall(p.pid)));
        this.addFileContent(`/proc/${dir}/status`, "");
        this.setDynamicFile(`/proc/${dir}/status`, () => new TextEncoder().encode(this.procStatus(p.pid)));
        this.addFileContent(`/proc/${dir}/cmdline`, "");
        this.setDynamicFile(`/proc/${dir}/cmdline`, () => new TextEncoder().encode(this.procOf(p.pid)?.cmdline ?? ""));
        this.addFileContent(`/proc/${dir}/comm`, "");
        this.setDynamicFile(`/proc/${dir}/comm`, () => new TextEncoder().encode((this.procOf(p.pid)?.comm ?? "") + "\n"));
      }
      for (const p of live) {
        const taskPath = `/proc/${p.pid}/task`;
        this.mkdirp(taskPath);
        const members = live.filter((t) => t.tgid === p.pid);
        const taskDir = this.lookupDir(taskPath);
        if (taskDir) {
          const wantTids = new Set(members.map((t) => String(t.pid)));
          for (const name of [...taskDir.children.keys()]) if (!wantTids.has(name)) taskDir.children.delete(name);
        }
        for (const t of members) {
          const tp = `${taskPath}/${t.pid}`;
          if (taskDir?.children.has(String(t.pid))) continue;
          this.addTaskDir(tp, t.pid);
        }
      }
      const selfProc = this.procOf(this.selfPid);
      const selfGroup = selfProc ? live.filter((t) => t.tgid === selfProc.tgid) : [];
      this.mkdirp("/proc/self/task");
      const selfTaskDir = this.lookupDir("/proc/self/task");
      if (selfTaskDir) {
        const want2 = new Set(selfGroup.map((t) => String(t.pid)));
        for (const name of [...selfTaskDir.children.keys()]) if (!want2.has(name)) selfTaskDir.children.delete(name);
        for (const t of selfGroup) {
          if (selfTaskDir.children.has(String(t.pid))) continue;
          this.addTaskDir(`/proc/self/task/${t.pid}`, t.pid);
        }
      }
    }
    /** the per-thread files under a /proc/<pid>/task/<tid> directory. */
    addTaskDir(path, tid) {
      this.mkdirp(path);
      this.addFileContent(`${path}/comm`, "");
      this.setDynamicFile(`${path}/comm`, () => new TextEncoder().encode((this.procOf(tid)?.comm ?? "") + "\n"));
      this.addFileContent(`${path}/stat`, "");
      this.setDynamicFile(`${path}/stat`, () => new TextEncoder().encode(this.procStat(tid)));
      this.addFileContent(`${path}/status`, "");
      this.setDynamicFile(`${path}/status`, () => new TextEncoder().encode(this.procStatus(tid)));
    }
    /** the directory inode at an absolute path, or null. */
    lookupDir(path) {
      let node = this.root;
      for (const part of path.split("/")) {
        if (!part) continue;
        if (!node || node.type !== "dir") return null;
        node = node.children.get(part);
      }
      return node && node.type === "dir" ? node : null;
    }
    syncSelfFds() {
      if (!this.fdListProvider) return;
      const live = this.fdListProvider();
      this.fillFdDirs("/proc/self", live);
      this.fillFdDirs(`/proc/${this.selfPid}`, live);
    }
    /** the fd and fdinfo directories under one /proc process directory. */
    fillFdDirs(base, live) {
      this.mkdirp(`${base}/fd`);
      this.mkdirp(`${base}/fdinfo`);
      const dir = this.lookupDir(`${base}/fd`);
      const infoDir = this.lookupDir(`${base}/fdinfo`);
      if (!dir || !infoDir) return;
      dir.children.clear();
      infoDir.children.clear();
      for (const e of live) {
        dir.children.set(String(e.fd), this.makeSymlink(e.target));
        this.addFileContent(`${base}/fdinfo/${e.fd}`, e.info);
      }
    }
    procOf(pid) {
      return this.procListProvider?.().find((p) => p.pid === pid) ?? null;
    }
    stateLetter(p) {
      if (p.state === 2) return "Z";
      if (p.state === 4) return "T";
      return p.blocked ? "S" : "R";
    }
    /**
     * /proc/<pid>/stat. the fields we can answer honestly are filled in and the rest are zero, which
     * is what a reader gets for a process with no accounting anyway. the layout matters more than the
     * values: ps and friends index fields positionally, and comm is parenthesised because it may
     * contain spaces (every parser finds it by the LAST ')').
     */
    procStat(pid) {
      const p = this.procOf(pid) ?? { pid, ppid: 0, pgid: pid, tgid: pid, state: 1, blocked: false, comm: "", cmdline: "" };
      const own = pid === this.selfPid ? this.selfCpuProvider?.() : void 0;
      const ticks = (ms) => Math.floor(ms / 10);
      const nice = p.nice ?? 0;
      const f = [p.pid, `(${p.comm})`, this.stateLetter(p), p.ppid, p.pgid, p.sid ?? p.pgid, 0, -1, 0];
      while (f.length < 52) f.push(0);
      f[13] = ticks(own?.utimeMs ?? p.utimeMs ?? 0);
      f[14] = ticks(own?.stimeMs ?? p.stimeMs ?? 0);
      f[17] = 20 + nice;
      f[18] = nice;
      f[19] = 1;
      return f.join(" ") + "\n";
    }
    procSyscall(pid) {
      const n = this.procOf(pid)?.syscall ?? -1;
      return n < 0 ? "running\n" : `${n} 0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x0
`;
    }
    procStatus(pid) {
      const p = this.procOf(pid) ?? { pid, ppid: 0, pgid: pid, tgid: pid, state: 1, blocked: false, comm: "", cmdline: "" };
      const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
      const L = this.stateLetter(p);
      const state = L === "Z" ? "Z (zombie)" : L === "S" ? "S (sleeping)" : "R (running)";
      return `Name:	${p.comm}
Umask:	0022
State:	${state}
Tgid:	${p.tgid}
Ngid:	0
Pid:	${p.pid}
PPid:	${p.ppid}
TracerPid:	0
Uid:	${c.ruid}	${c.euid}	${c.euid}	${c.euid}
Gid:	${c.rgid}	${c.egid}	${c.egid}	${c.egid}
FDSize:	64
Groups:	0
VmPeak:	    65536 kB
VmSize:	    65536 kB
VmRSS:	    32768 kB
VmLck:	${String(this.vmLckKb).padStart(8)} kB
VmSwap:	       0 kB
Threads:	1
`;
    }
    resolve(path, followFinal = true, depth = 0, searchEff = true) {
      if (depth === 0) {
        this.searchDenied = false;
        this.searchEff = searchEff;
        this.lastResolveErrno = ENOENT2;
        if (path.startsWith("/proc") || this.cwd.startsWith("/proc")) {
          this.syncProcPids();
          this.syncSelfFds();
        }
      }
      if (depth > MAX_SYMLINK_DEPTH) {
        this.lastResolveErrno = ELOOP2;
        return null;
      }
      const parts = this.normalizeParts(path);
      for (const part of parts) {
        if (part.length > NAME_MAX) {
          this.lastResolveErrno = ENAMETOOLONG2;
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
      return this.canAccess(dir, 2);
    }
    // DAC check: does the effective uid/gid have all of `need` (4=read|2=write|1=exec) on the inode?
    // root (euid 0) bypasses. mirrors the kernel's owner/group/other mode-bit selection.
    canAccess(inode, need) {
      const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
      if (c.euid === 0) return true;
      let perms;
      if (inode.uid === c.euid) perms = inode.mode >> 6 & 7;
      else if (inode.gid === c.egid) perms = inode.mode >> 3 & 7;
      else perms = inode.mode & 7;
      return (perms & need) === need;
    }
    resolveFromParts(parts, followFinal, depth) {
      if (depth > MAX_SYMLINK_DEPTH) {
        this.lastResolveErrno = ELOOP2;
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
      if (depth === 0) {
        this.lastResolveErrno = ENOENT2;
        if (path.startsWith("/proc") || this.cwd.startsWith("/proc")) {
          this.syncProcPids();
          this.syncSelfFds();
        }
      }
      if (depth > MAX_SYMLINK_DEPTH) {
        this.lastResolveErrno = ELOOP2;
        return null;
      }
      const resolved = this.resolvePath(path);
      const parts = resolved.split("/").filter((p) => p.length > 0);
      if (parts.length === 0) return null;
      const name = parts.pop();
      if (name.length > NAME_MAX) {
        this.lastResolveErrno = ENAMETOOLONG2;
        return null;
      }
      let dir = this.root;
      for (let idx = 0; idx < parts.length; idx++) {
        const part = parts[idx];
        if (part.length > NAME_MAX) {
          this.lastResolveErrno = ENAMETOOLONG2;
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
          const base = target.startsWith("/") ? "" : "/" + parts.slice(0, idx).join("/");
          const remaining = parts.slice(idx + 1).join("/");
          const newPath = [base, target, remaining, name].filter((s) => s.length > 0).join("/").replace(/\/+/g, "/");
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
      const resolved = this.resolvePath(path);
      if (path.length > PATH_MAX) return -ENAMETOOLONG2;
      for (const part of path.split("/")) {
        if (part.length > NAME_MAX) return -ENAMETOOLONG2;
      }
      if (this.openFds.size >= MAX_FDS2) return -EMFILE2;
      const followSymlinks = !(flags & O_NOFOLLOW);
      let inode = this.resolve(path, followSymlinks);
      let created = false;
      if (flags & O_NOFOLLOW && !(flags & O_CREAT)) {
        if (inode && inode.type === "symlink") return -ELOOP2;
        if (!inode) {
          const raw = this.resolve(path, false);
          if (raw && raw.type === "symlink") return -ELOOP2;
        }
      }
      if (inode && flags & O_CREAT && flags & O_EXCL) {
        return -EEXIST2;
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
        created = true;
      }
      if (!inode) return this.resolveErrno();
      if (!created && inode.type !== "dir" && !(flags & O_PATH)) {
        const acc = flags & O_ACCMODE;
        let need = 0;
        if (acc === O_RDONLY || acc === O_RDWR) need |= 4;
        if (acc === O_WRONLY || acc === O_RDWR) need |= 2;
        if (need && !this.canAccess(inode, need)) return -EACCES2;
        if (flags & 262144) {
          const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
          if (c.euid !== 0 && inode.uid !== c.euid) return -EPERM2;
        }
      }
      if (flags & O_DIRECTORY && inode.type !== "dir") return -ENOTDIR2;
      if (inode.type === "dir") {
        if ((flags & O_ACCMODE) !== O_RDONLY) return -EISDIR2;
        const fd2 = this.takeFd();
        const entries = [".", "..", ...inode.children.keys()];
        this.openFds.set(fd2, { type: "dir", path: resolved, inode, entries, pos: 0, refcount: 1 });
        return fd2;
      }
      if (inode.type === "chardev") {
        const fd2 = this.takeFd();
        this.openFds.set(fd2, { type: "chardev", path: resolved, inode, refcount: 1 });
        return fd2;
      }
      if (inode.type === "fifo") {
        const acc = flags & O_ACCMODE;
        if (acc === O_WRONLY && flags & O_NONBLOCK2 && (inode.readers ?? 0) === 0) return -ENXIO;
        const fd2 = this.takeFd();
        if (acc === O_RDONLY || acc === O_RDWR) inode.readers = (inode.readers ?? 0) + 1;
        this.openFds.set(fd2, { type: "fifo", path: resolved, inode, accessMode: acc, refcount: 1 });
        return fd2;
      }
      if (inode.type !== "file") return -ENOENT2;
      if (inode.dynamic) inode.data = inode.dynamic();
      if (flags & O_TRUNC && (flags & O_ACCMODE) !== O_RDONLY) {
        inode.data = new Uint8Array(0);
        inode.mtime = inode.ctime = this.now();
      }
      const fd = this.takeFd();
      const accessMode = flags & O_ACCMODE;
      const pos = flags & O_APPEND ? inode.data.length : 0;
      this.openFds.set(fd, { type: "file", path: resolved, inode, pos, flags, accessMode, refcount: 1, oPath: (flags & O_PATH) !== 0 });
      return fd;
    }
    /** true if the fd was opened O_PATH: it references a location only, so read/write/fchmod/fchown/
     *  ioctl/mmap on it must fail EBADF. */
    /** the path an open descriptor was opened by, for /proc/self/fd. */
    fdPath(fd) {
      return this.openFds.get(fd)?.path ?? null;
    }
    /** true when the descriptor names a character device. fsync has nothing to flush for one, which
     *  is why linux answers EINVAL rather than pretending it wrote something back. */
    isCharDev(fd) {
      return this.openFds.get(fd)?.type === "chardev";
    }
    /** does this descriptor name a directory? *at() syscalls need to know before they resolve. */
    fdIsDir(fd) {
      return this.openFds.get(fd)?.type === "dir";
    }
    /** every descriptor this vfs holds open. */
    openFdNumbers() {
      return [...this.openFds.keys()];
    }
    isOPath(fd) {
      return this.openFds.get(fd)?.oPath === true;
    }
    close(fd) {
      const ofd = this.openFds.get(fd);
      if (!ofd) return -EBADF2;
      this.openFds.delete(fd);
      ofd.refcount--;
      if (ofd.refcount <= 0 && ofd.type === "fifo" && (ofd.accessMode === O_RDONLY || ofd.accessMode === O_RDWR)) {
        ofd.inode.readers = Math.max(0, (ofd.inode.readers ?? 0) - 1);
      }
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
      if (ofd.type === "file" && ofd.oPath) return -EBADF2;
      if (ofd.type === "chardev") {
        const bytes2 = new Uint8Array(memory2.buffer);
        switch (`${ofd.inode.major}:${ofd.inode.minor}`) {
          case "1:3":
            return 0;
          // /dev/null -- always at end of file
          case "1:5":
          case "1:7":
            bytes2.fill(0, buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
            return count;
          case "1:8":
          case "1:9": {
            crypto.getRandomValues(bytes2.subarray(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count));
            return count;
          }
          default:
            return 0;
        }
      }
      if (ofd.type === "dir") return -EISDIR2;
      if (ofd.type === "fifo") return -EAGAIN2;
      if (ofd.type !== "file") return -EBADF2;
      if (ofd.accessMode === O_WRONLY) return -EBADF2;
      const bytes = new Uint8Array(memory2.buffer);
      const ranged = ofd.inode.ranged;
      if (ranged) {
        const avail = ranged.size - ofd.pos;
        const n = Math.min(count, Math.max(0, avail));
        if (n <= 0) return 0;
        bytes.set(ranged.read(ofd.pos, n), buf + GUEST_MEM_BASE);
        ofd.pos += n;
        return n;
      }
      const available = ofd.inode.data.length - ofd.pos;
      const toRead = Math.min(count, Math.max(0, available));
      if (toRead <= 0) return 0;
      bytes.set(ofd.inode.data.subarray(ofd.pos, ofd.pos + toRead), buf + GUEST_MEM_BASE);
      ofd.pos += toRead;
      ofd.inode.atime = this.now();
      return toRead;
    }
    write(fd, buf, count, memory2) {
      const ofd = this.openFds.get(fd);
      if (!ofd) return -EBADF2;
      if (ofd.type === "file" && ofd.oPath) return -EBADF2;
      if (ofd.type === "chardev") return ofd.inode.major === 1 && ofd.inode.minor === 7 ? -28 : count;
      if (ofd.type === "dir") return -EBADF2;
      if (ofd.type !== "file") return -EBADF2;
      if (ofd.accessMode === O_RDONLY) return -EBADF2;
      if (count === 0) return 0;
      const guestMax = memory2.buffer.byteLength - GUEST_MEM_BASE;
      if (buf < 0 || buf >= guestMax || buf + count > guestMax) return -14;
      if (ofd.inode.sysctlWrite) {
        const bytes = new Uint8Array(memory2.buffer, buf + GUEST_MEM_BASE, count);
        const rc = ofd.inode.sysctlWrite(new TextDecoder().decode(bytes).trim());
        return rc < 0 ? rc : count;
      }
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
          newPos = (ofd.inode.ranged ? ofd.inode.ranged.size : ofd.inode.data.length) + off;
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
      const AT_SYMLINK_NOFOLLOW2 = 256;
      const AT_EMPTY_PATH = 4096;
      if (flags & AT_EMPTY_PATH && path === "") {
        return this.fstat(dirfd, statBuf, memory2);
      }
      const follow = !(flags & AT_SYMLINK_NOFOLLOW2);
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
      if (!inode) return null;
      if (inode.type !== "symlink") {
        this.lastResolveErrno = EINVAL2;
        return null;
      }
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
        if (child.type === "dir") return -EISDIR2;
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
      if (dir.children.has(name)) return -EEXIST2;
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
      if (dir.children.has(name)) return -EEXIST2;
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
      if (dir.children.has(name)) return -EEXIST2;
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
        if (srcNode.type !== "dir" && dstNode.type === "dir") return -EISDIR2;
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
      if (path.length > PATH_MAX) return -ENAMETOOLONG2;
      for (const part of path.split("/")) {
        if (part.length > NAME_MAX) return -ENAMETOOLONG2;
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
    /**
     * the open descriptions, as plain data.
     *
     * this crosses a postMessage (fork, and the promotion of a process to its first thread), so it
     * must contain nothing that structured clone refuses. an OFD points at its inode, and a
     * synthesised /proc inode carries the FUNCTION that computes its contents -- copying the object
     * graph therefore threw DataCloneError and killed the worker, which is what any program that
     * held a /proc descriptor across pthread_create saw.
     *
     * what the other side needs is the NAME. its own tree is built the same way, so it re-resolves.
     * dup'd descriptors share one description, and that sharing survives as an index.
     */
    snapshotFds() {
      const idx = /* @__PURE__ */ new Map();
      const ofds = [];
      const fds = [];
      for (const [fd, ofd] of this.openFds) {
        if (!idx.has(ofd)) {
          idx.set(ofd, ofds.length);
          const d = { type: ofd.type, path: ofd.path, refcount: ofd.refcount };
          if (ofd.type === "file") {
            d.pos = ofd.pos;
            d.flags = ofd.flags;
            d.accessMode = ofd.accessMode;
            d.oPath = ofd.oPath;
          } else if (ofd.type === "dir") {
            d.entries = [...ofd.entries];
            d.pos = ofd.pos;
          } else if (ofd.type === "fifo") {
            d.accessMode = ofd.accessMode;
          }
          ofds.push(d);
        }
        fds.push([fd, idx.get(ofd)]);
      }
      return { ofds, fds };
    }
    restoreFds(snap) {
      this.openFds = /* @__PURE__ */ new Map();
      if (!snap || !Array.isArray(snap.ofds)) return;
      let synced = false;
      const rebuilt = snap.ofds.map((d) => {
        let node = this.lookupNode(d.path);
        if (!node && !synced) {
          synced = true;
          this.syncSelfFds();
          node = this.lookupNode(d.path);
        }
        if (!node) return null;
        if (d.type === "file" && node.type === "file") {
          return {
            type: "file",
            path: d.path,
            inode: node,
            pos: d.pos ?? 0,
            flags: d.flags ?? 0,
            accessMode: d.accessMode ?? 0,
            refcount: d.refcount ?? 1,
            oPath: d.oPath
          };
        }
        if (d.type === "dir" && node.type === "dir") {
          return {
            type: "dir",
            path: d.path,
            inode: node,
            entries: [...d.entries ?? []],
            pos: d.pos ?? 0,
            refcount: d.refcount ?? 1
          };
        }
        if (d.type === "chardev" && node.type === "chardev") {
          return { type: "chardev", path: d.path, inode: node, refcount: d.refcount ?? 1 };
        }
        if (d.type === "fifo" && node.type === "fifo") {
          return {
            type: "fifo",
            path: d.path,
            inode: node,
            accessMode: d.accessMode ?? 0,
            refcount: d.refcount ?? 1
          };
        }
        return null;
      });
      for (const [fd, i] of snap.fds) {
        const ofd = rebuilt[i];
        if (ofd) this.openFds.set(fd, ofd);
      }
    }
    /** the device numbers behind a descriptor open on one of our char devices, else null. */
    fdDevice(fd) {
      const ofd = this.openFds.get(fd);
      if (!ofd || ofd.type !== "chardev") return null;
      return { major: ofd.inode.major, minor: ofd.inode.minor };
    }
    /** what kind of thing a descriptor of ours is open on, or null when it is not one of ours. */
    fdOfdType(fd) {
      return this.openFds.get(fd)?.type ?? null;
    }
    /** the inode a path names in this tree, of whatever type, or null. */
    lookupNode(path) {
      let node = this.root;
      for (const part of path.split("/")) {
        if (!part) continue;
        if (!node || node.type !== "dir") return null;
        node = node.children.get(part);
      }
      return node ?? null;
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
      const rdev = inode.type === "chardev" ? inode.major << 8 | inode.minor & 255 | (inode.minor & ~255) << 12 : 0;
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
    const segs = [];
    let phdrAddr = 0n;
    let phoffAddr = 0n;
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
      const pFlags = dv.getUint32(phBase + 4, true);
      if (pType === PT_PHDR) {
        phdrAddr = pVaddr;
      }
      if (pType === PT_LOAD && phoffAddr === 0n && phoff >= pOffset && phoff < pOffset + pFilesz) {
        phoffAddr = pVaddr + BigInt(phoff - pOffset);
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
      segs.push({
        start: guestAddr & ~4095,
        end: guestAddr + pMemsz + 4095 & ~4095,
        prot: (pFlags & 4 ? 1 : 0) | (pFlags & 2 ? 2 : 0) | (pFlags & 1 ? 4 : 0)
      });
      const segEnd = pVaddr + BigInt(pMemsz);
      if (segEnd > brkStart) {
        brkStart = segEnd;
      }
    }
    if (phdrAddr === 0n) {
      phdrAddr = phoffAddr !== 0n ? phoffAddr : loadBase + BigInt(phoff);
    }
    brkStart = brkStart + 0xFFFn & ~0xFFFn;
    return {
      entry,
      phdr: phdrAddr,
      phentSize,
      phnum,
      brkStart,
      loadBase,
      interpPath,
      segs
    };
  }
  function loadInterp(interpBytes, memory2, base = INTERP_LOAD_BASE) {
    const dv = new DataView(interpBytes.buffer, interpBytes.byteOffset, interpBytes.byteLength);
    for (let i = 0; i < 4; i++) if (interpBytes[i] !== ELF_MAGIC[i]) throw new Error("interp not an ELF");
    const rawEntry = dv.getBigUint64(24, true);
    const phoff = Number(dv.getBigUint64(32, true));
    const phentSize = dv.getUint16(54, true);
    const phnum = dv.getUint16(56, true);
    const segs = [];
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
      const pFlags = dv.getUint32(phBase + 4, true);
      segs.push({
        start: Number(pVaddr) & ~4095,
        end: Number(pVaddr) + pMemsz + 4095 & ~4095,
        prot: (pFlags & 4 ? 1 : 0) | (pFlags & 2 ? 2 : 0) | (pFlags & 1 ? 4 : 0)
      });
    }
    return { entry: rawEntry + loadBase, base: loadBase, segs };
  }
  function loadProgram(memory2, elfBytes2, argv, envp, readInterp) {
    const elfInfo = loadElf(elfBytes2, memory2);
    let startPC = elfInfo.entry;
    if (elfInfo.interpPath) {
      const interpBytes = readInterp(elfInfo.interpPath);
      const { entry: iEntry, base: iBase, segs: iSegs } = loadInterp(interpBytes, memory2);
      elfInfo.segs.push(...iSegs);
      elfInfo.interpBase = iBase;
      elfInfo.interpEntry = iEntry;
      startPC = iEntry;
    }
    const sp = setupStack(memory2, elfInfo, argv, envp);
    return { entry: startPC, sp, brkStart: elfInfo.brkStart, interp: !!elfInfo.interpPath, segs: elfInfo.segs };
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

  // src/pipe-ring.ts
  var PipeRing = class {
    i32;
    data;
    dataSize;
    baseOffset;
    /**
     * @param sab   The kernel SharedArrayBuffer (or the shared arena)
     * @param offset  Byte offset of this ring's header within it
     * @param totalSize  Total size of this ring (header + data). Pass it when CREATING the ring;
     *   omit it to adopt an existing one, whose size is recorded in its own header. Pipes are not all
     *   the same size any more, so a caller that guessed from a build constant would run off the end
     *   of a small ring -- omitting the argument is both shorter and the safe default.
     */
    constructor(sab, offset, totalSize) {
      this.baseOffset = offset;
      const hdr = new Int32Array(sab, offset, PIPE_HEADER_SIZE / 4);
      this.dataSize = totalSize !== void 0 ? totalSize - PIPE_HEADER_SIZE : Atomics.load(hdr, PIPE_CAPACITY_W);
      if (!(this.dataSize > 0)) {
        throw new RangeError(`PipeRing at ${offset}: ring has no capacity (adopted before reset?)`);
      }
      this.i32 = hdr;
      this.data = new Uint8Array(sab, offset + PIPE_HEADER_SIZE, this.dataSize);
    }
    /** usable data bytes -- what F_GETPIPE_SZ reports for this particular pipe. */
    get capacity() {
      return this.dataSize;
    }
    /** the monotonic stream positions. ancillary data (SCM_RIGHTS) is delivered with the byte it was
     *  sent alongside, so both ends need to name a place in the stream, not just a count. */
    get writePos() {
      return Atomics.load(this.i32, PIPE_WRITE_POS);
    }
    get readPos() {
      return Atomics.load(this.i32, PIPE_READ_POS);
    }
    /** how many descriptors are queued in flight over this ring, and how to change that count. */
    get scmPending() {
      return Atomics.load(this.i32, PIPE_SCM_W);
    }
    addScm(n) {
      Atomics.add(this.i32, PIPE_SCM_W, n);
    }
    /** how much the underlying block can hold; capacity may be set anywhere up to this. */
    get allocated() {
      return Atomics.load(this.i32, PIPE_ALLOC_W) || this.dataSize;
    }
    /**
     * Change this pipe's capacity (F_SETPIPE_SZ). Returns the size actually set, or a negative errno.
     *
     * The ring's indexing is modular in the capacity, so every buffered byte's position depends on it:
     * changing the number while data is in flight would move the data out from under whoever is
     * reading it. Linux resizes by copying under the pipe mutex; having no such lock, this refuses a
     * resize of a pipe with anything buffered -- EBUSY, which is the same answer linux gives when the
     * request cannot be satisfied without losing data. A program that sizes its pipe does it on a
     * fresh one, before any bytes flow, which is exactly the case that works.
     */
    resize(bytes) {
      const room = this.allocated;
      const want = Math.min(bytes, room);
      if (want < 4096) return -22;
      if (this.available !== 0) return -16;
      Atomics.store(this.i32, PIPE_WRITE_POS, 0);
      Atomics.store(this.i32, PIPE_READ_POS, 0);
      Atomics.store(this.i32, PIPE_CAPACITY_W, want);
      return want;
    }
    /** Reset the ring to empty state with the given number of open reader/writer ends. */
    reset(writers = 1, readers = 1) {
      Atomics.store(this.i32, PIPE_WRITE_POS, 0);
      Atomics.store(this.i32, PIPE_READ_POS, 0);
      Atomics.store(this.i32, PIPE_WRITER_CLOSED, writers);
      Atomics.store(this.i32, PIPE_READER_CLOSED, readers);
      Atomics.store(this.i32, PIPE_CAPACITY_W, this.dataSize);
      Atomics.store(this.i32, PIPE_ALLOC_W, this.dataSize);
      Atomics.store(this.i32, PIPE_SCM_W, 0);
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
      const available = this.dataSize - used;
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
      bumpIoGeneration();
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
        if (this.dataSize - (wp - rp) <= 0) {
          if (waiter(this.i32, PIPE_READ_POS, rp)) return written > 0 ? written : -2;
        }
      }
      return written;
    }
    /** room left in the ring: how much a write could take without blocking. */
    get freeBytes() {
      return this.dataSize - (Atomics.load(this.i32, PIPE_WRITE_POS) - Atomics.load(this.i32, PIPE_READ_POS));
    }
    /** bytes sitting in the ring right now, which is what FIONREAD reports. */
    get bytesAvailable() {
      return Atomics.load(this.i32, PIPE_WRITE_POS) - Atomics.load(this.i32, PIPE_READ_POS);
    }
    /** Decrement writer refcount. Last close triggers EOF for readers. */
    closeWriter() {
      const remaining = Atomics.sub(this.i32, PIPE_WRITER_CLOSED, 1) - 1;
      if (remaining <= 0) {
        Atomics.store(this.i32, PIPE_WRITER_CLOSED, 0);
        Atomics.notify(this.i32, PIPE_WRITE_POS);
        bumpIoGeneration();
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
          bumpIoGeneration();
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
    /** does this pipe still have a writer? distinguishes "empty for now" (EAGAIN) from EOF. */
    writersOpen() {
      return Atomics.load(this.i32, PIPE_WRITER_CLOSED) > 0;
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
      bumpIoGeneration();
      return toRead;
    }
    /**
     * copy bytes out WITHOUT consuming them.
     *
     * tee(2) duplicates a pipe's contents into another pipe and leaves the original readable, so it
     * needs a read that does not move the read pointer. splice needs it too, for a different reason:
     * it must not take bytes out of the pipe until it knows the destination accepted them, or a
     * short write silently loses whatever did not fit.
     */
    peek(dst) {
      const wp = Atomics.load(this.i32, PIPE_WRITE_POS);
      const rp = Atomics.load(this.i32, PIPE_READ_POS);
      const available = wp - rp;
      if (available <= 0) return 0;
      const toRead = Math.min(dst.length, available);
      const ringPos = rp % this.dataSize;
      const firstChunk = Math.min(toRead, this.dataSize - ringPos);
      dst.set(this.data.subarray(ringPos, ringPos + firstChunk));
      if (toRead > firstChunk) dst.set(this.data.subarray(0, toRead - firstChunk), firstChunk);
      return toRead;
    }
    /** consume `n` bytes that a previous peek already copied out. */
    discard(n) {
      const wp = Atomics.load(this.i32, PIPE_WRITE_POS);
      const rp = Atomics.load(this.i32, PIPE_READ_POS);
      const dropped = Math.min(n, Math.max(0, wp - rp));
      if (dropped <= 0) return 0;
      Atomics.store(this.i32, PIPE_READ_POS, rp + dropped);
      Atomics.notify(this.i32, PIPE_READ_POS);
      return dropped;
    }
    /** Decrement reader refcount. Last close signals broken pipe to writers. */
    closeReader() {
      const remaining = Atomics.sub(this.i32, PIPE_READER_CLOSED, 1) - 1;
      if (remaining <= 0) {
        Atomics.store(this.i32, PIPE_READER_CLOSED, 0);
        Atomics.notify(this.i32, PIPE_READ_POS);
        bumpIoGeneration();
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

  // src/shared-map.ts
  function reconcile(mine, arena, shadow) {
    const n = mine.length;
    const words = n >>> 2;
    const m32 = new Uint32Array(mine.buffer, mine.byteOffset, words);
    const a32 = new Uint32Array(arena.buffer, arena.byteOffset, words);
    const s32 = new Uint32Array(shadow.buffer, shadow.byteOffset, words);
    for (let i = 0; i < words; i++) {
      const mv = m32[i];
      const sv = s32[i];
      if (mv !== sv) {
        a32[i] = mv;
        s32[i] = mv;
      } else {
        const av = a32[i];
        if (av !== sv) {
          m32[i] = av;
          s32[i] = av;
        }
      }
    }
    for (let i = words << 2; i < n; i++) {
      const mv = mine[i];
      const sv = shadow[i];
      if (mv !== sv) {
        arena[i] = mv;
        shadow[i] = mv;
      } else if (arena[i] !== sv) {
        mine[i] = arena[i];
        shadow[i] = arena[i];
      }
    }
  }

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
  var threadLayout = DEFAULT_THREAD_LAYOUT;
  var REGION_CACHE_CAP = 2048;
  var elfBytes;
  var kernelSAB;
  var procTable;
  var sigQueue;
  var procCreds;
  var globalWords;
  var mySlot;
  var myPid;
  var myMailbox;
  var kernelPipes = /* @__PURE__ */ new Map();
  var nextLocalFd = 100;
  var stdinRing = null;
  var workerReady = false;
  var lastSyscallNr = -1;
  var tracing = false;
  function readCString(addr) {
    try {
      const b = new Uint8Array(memory.buffer);
      let end = addr + GUEST_MEM_BASE;
      const lim = end + 512;
      while (end < lim && b[end] !== 0) end++;
      return new TextDecoder().decode(b.subarray(addr + GUEST_MEM_BASE, end));
    } catch {
      return null;
    }
  }
  var tA3 = 0n;
  var tA4 = 0n;
  function traceCall(nr, a0, a1, a2, path) {
    if (!tracing) return;
    const r = BigInt.asIntN(64, cpu.getReg(0));
    self.postMessage({
      type: "trace",
      line: `pid${myPid} ${SYSNAME[nr] ?? nr}(${a0}, ${a1}, ${a2}, ${tA3}, ${tA4})${path ? ' "' + path + '"' : ""} = ${r}`
    });
  }
  var SYSNAME = {
    56: "openat",
    57: "close",
    62: "lseek",
    63: "read",
    64: "write",
    66: "writev",
    68: "pwrite",
    37: "linkat",
    38: "renameat",
    35: "unlinkat",
    36: "mkdirat",
    45: "truncate",
    46: "ftruncate",
    82: "fsync",
    83: "fdatasync",
    222: "mmap",
    215: "munmap",
    88: "utimensat",
    79: "fstatat",
    80: "fstat",
    78: "readlinkat",
    439: "faccessat2"
  };
  var ZIG_SHADOW_BASE = 33554432;
  var parkedMs = 0;
  var parkDepth = 0;
  var parkStartMs = 0;
  var procStartMs = performance.now();
  var GUEST_STACK_TOP = 8388608;
  var MEM_FAULT = { memFault: true };
  var MAX_PROCESS_BYTES2 = 1856 * 1024 * 1024;
  var PROT_TABLE_SIZE = MAX_PROCESS_BYTES2 >>> 12;
  var protActive = null;
  var protBase = null;
  var protReady = false;
  var pendingFault = null;
  var memFault = (addr, isWrite) => {
    pendingFault = { addr, isWrite };
    throw MEM_FAULT;
  };
  function resetPageProt() {
    if (protActive) protActive.value = 0;
    protReady = false;
    try {
      new Int32Array(memory.buffer, PROT_PAINTED_FLAG, 1)[0] = 0;
    } catch {
    }
  }
  var PAGE_NO_BACKING2 = 8;
  function protPainted() {
    return protReady || threaded && new Int32Array(memory.buffer, PROT_PAINTED_FLAG, 1)[0] === 1;
  }
  function claimPaint() {
    if (!threaded) {
      if (protReady) return false;
      protReady = true;
      return true;
    }
    const flag = new Int32Array(memory.buffer, PROT_PAINTED_FLAG, 1);
    const won = Atomics.compareExchange(flag, 0, 0, 1) === 0;
    protReady = true;
    return won;
  }
  function paintProtTable() {
    const base = PROT_TABLE_BASE;
    const bytes = new Uint8Array(memory.buffer);
    bytes.fill(0, base, base + PROT_TABLE_SIZE);
    syscalls.paintProtTable(GUEST_STACK_TOP, (s, e, p) => {
      const f = s >>> 12, l = e + 4095 >>> 12;
      const pb = p & 15;
      for (let q = f; q < l; q++) bytes[base + q] = pb;
    });
  }
  function setPageProt(addr, len, prot) {
    if (!protActive || !protBase) return;
    const b = prot & 15;
    if ((b & 3) === 3 && !protPainted()) return;
    const base = PROT_TABLE_BASE;
    const bytes = new Uint8Array(memory.buffer);
    if (!protPainted() && claimPaint()) paintProtTable();
    const first = addr >>> 12;
    const last = addr + len + 4095 >>> 12;
    for (let p = first; p < last; p++) bytes[base + p] = b;
    protActive.value = 1;
  }
  self.onmessage = (e) => {
    try {
      dispatchMessage(e.data);
    } catch (err) {
      self.postMessage({
        type: "worker_error",
        detail: `${err?.name ?? "Error"}: ${err?.message ?? err}
${err?.stack ?? ""}`
      });
      throw err;
    }
  };
  var dispatchMessage = (msg) => {
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
      case "set_trace":
        tracing = !!msg.on;
        break;
      case "promote":
        completePromote(msg.sharedMemory, msg.tid, msg.regBaseAddr, msg.staleHigh, msg.layout);
        break;
      case "clone_fail":
        completePromoteFail();
        break;
    }
  };
  var tStage = {};
  function stamp(name) {
    tStage[name] = performance.now();
  }
  function reportStages() {
    if (!tracing) return;
    const t0 = tStage.init_enter ?? 0;
    const parts = Object.entries(tStage).map(([k, v]) => `${k}=${(v - t0).toFixed(1)}`).join(" ");
    self.postMessage({ type: "trace", line: `pid${myPid} STAGES ${parts}` });
  }
  function init(config) {
    for (const k of Object.keys(tStage)) delete tStage[k];
    stamp("init_enter");
    myPid = config.pid;
    mySlot = config.slot;
    procTable = new Int32Array(config.kernelSAB);
    publishSyscall(-2);
    shmArena = config.shmArena;
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
      protActive = new WebAssembly.Global({ value: "i32", mutable: true }, 0);
      protBase = new WebAssembly.Global({ value: "i32", mutable: true }, PROT_TABLE_BASE);
      protReady = false;
      workerReady = true;
    } else {
      if (threaded) {
        memory = new WebAssembly.Memory({ initial: 512, maximum: 65536 });
        threaded = false;
        stateBase = 0;
        regBaseGlobal = null;
        protActive = new WebAssembly.Global({ value: "i32", mutable: true }, 0);
        protBase = new WebAssembly.Global({ value: "i32", mutable: true }, PROT_TABLE_BASE);
        protReady = false;
      } else if (protActive) {
        protActive.value = 0;
        protReady = false;
      }
      const dirtyEnd = Math.min(Number(syscalls.getMmapBase()) + GUEST_MEM_BASE, memory.buffer.byteLength);
      new Uint8Array(memory.buffer).fill(0, 0, dirtyEnd);
      regionCache.clear();
      kernelPipes.clear();
      unixSocks.clear();
      nextLocalFd = 100;
      stdinRing = null;
    }
    procTable = new Int32Array(kernelSAB);
    sigQueue = new Int32Array(kernelSAB);
    procCreds = new Int32Array(kernelSAB);
    globalWords = new Int32Array(kernelSAB, GLOBAL_STATE_OFFSET, 64);
    bindIoGeneration(globalWords);
    myMailbox = new Int32Array(kernelSAB, mailboxSlotOffset(mySlot), MAILBOX_SLOT_SIZE / 4);
    const bridgeHandlesRealFs = !!config.kernelSAB;
    vfs = new InMemoryVFS({ bridgeHandlesRealFs });
    vfs.procListProvider = procSnapshot;
    vfs.sysvipcProvider = sysvipcListing;
    stamp("vfs_ready");
    vfs.fdListProvider = () => syscalls?.openFdList() ?? [];
    vfs.memSizeProvider = () => memory.buffer.byteLength - GUEST_MEM_BASE;
    if (!bridgeHandlesRealFs) {
      vfs.loadBinary(elfBytes);
    }
    if (config.threadStart) {
      const ts = config.threadStart;
      if (ts.layout) threadLayout = ts.layout;
      memory = ts.sharedMemory;
      threaded = true;
      protActive = new WebAssembly.Global({ value: "i32", mutable: true }, 0);
      protBase = new WebAssembly.Global({ value: "i32", mutable: true }, PROT_TABLE_BASE);
      if (new Int32Array(memory.buffer, PROT_PAINTED_FLAG, 1)[0] === 1) {
        protReady = true;
        protActive.value = 1;
      }
      sharedMaps = (ts.shmMaps ?? []).map((m) => {
        const start = m.addr + GUEST_MEM_BASE;
        const inherited = start + m.len <= memory.buffer.byteLength ? new Uint8Array(new Uint8Array(memory.buffer, start, m.len)) : new Uint8Array(new Uint8Array(shmArena, m.off, m.len));
        return { addr: m.addr, len: m.len, arenaOff: m.off, shadow: inherited };
      });
      sysvAttached = new Map((ts.shmMaps ?? []).filter((m) => m.sysv !== void 0).map((m) => [m.addr, m.sysv]));
      if (ts.unixSocksSnapshot) {
        for (const [fd, sock] of ts.unixSocksSnapshot) unixSocks.set(fd, { ...sock });
      }
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
      syscalls = newSyscallHandler(0n, {
        writeStdout: (d) => postWrite(1, d),
        writeStderr: (d) => postWrite(2, d),
        readStdin: (count) => readStdin(count),
        stdinHasData: () => stdinRing != null && (!stdinRing.empty || stdinRing.writerClosed),
        waitStdin: (timeoutMs) => waitStdin(timeoutMs),
        ioGeneration: () => readIoGeneration(),
        waitIo: (seen, timeoutMs) => waitIo(seen, timeoutMs),
        unixReady: (fd) => unixReady(fd),
        pipeReady: (fd) => pipeReady(fd),
        inotifyPending: (fd) => {
          const id = syscalls.inotifyIdOf(fd);
          return id < 0 ? 0 : inotifyCall({ op: "pending", id })[1];
        },
        mqReady: (fd) => mqReady(fd),
        pidExited: (pid) => pidExited(pid),
        readFileAt: (fd, offset, count) => readNativeFileAt(fd, offset, count)
      }, vfs);
      syscalls.bridgeAvailable = true;
      syscalls.eventfdHook = eventfdCall;
      if (ts.syscallState) syscalls.restoreState(ts.syscallState);
      syscalls.attachVmaTable(memory, threadLayout.vmaBase);
      syscalls.attachSigTable(memory, threadLayout.sigBase);
      syscalls.setThreadIds(ts.tid, ts.tgid);
      syscalls.maxBytes = threadLayout.regsBase;
      runToCompletion(BigInt(ts.startPC));
      return;
    }
    publishSyscall(-5);
    let pc;
    if (config.memSnapshot) {
      stamp("snapshot_begin");
      publishSyscall(-3);
      applySparseSnapshot(memory, config.memSnapshot);
      publishSyscall(-4);
      if (config.memSnapshot.stageOff !== void 0) self.postMessage({ type: "fork_ready" });
      stamp("snapshot_done");
      cpu = new CpuState(memory, stateBase);
      if (!config.resume) {
        cpu.setReg(0, 0n);
        const cStack = BigInt(config.childStack ?? "0");
        if (cStack !== 0n) cpu.setSP(cStack);
        const cFlags = Number(config.cloneFlags ?? 0);
        if (cFlags & 524288) {
          new DataView(memory.buffer).setBigUint64(stateBase + TPIDR_OFFSET, BigInt(config.childTls ?? "0"), true);
        }
        const cCtid = Number(BigInt(config.childCtid ?? "0"));
        if (cFlags & (16777216 | 2097152) && cCtid !== 0) {
          if (cFlags & 16777216) new DataView(memory.buffer).setInt32(cCtid + GUEST_MEM_BASE, myPid, true);
        }
      }
      syscalls = newSyscallHandler(0n, {
        writeStdout: (d) => postWrite(1, d),
        writeStderr: (d) => postWrite(2, d),
        readStdin: (count) => readStdin(count),
        stdinHasData: () => stdinRing != null && (!stdinRing.empty || stdinRing.writerClosed),
        waitStdin: (timeoutMs) => waitStdin(timeoutMs),
        ioGeneration: () => readIoGeneration(),
        waitIo: (seen, timeoutMs) => waitIo(seen, timeoutMs),
        unixReady: (fd) => unixReady(fd),
        pipeReady: (fd) => pipeReady(fd),
        inotifyPending: (fd) => {
          const id = syscalls.inotifyIdOf(fd);
          return id < 0 ? 0 : inotifyCall({ op: "pending", id })[1];
        },
        mqReady: (fd) => mqReady(fd),
        pidExited: (pid) => pidExited(pid),
        readFileAt: (fd, offset, count) => readNativeFileAt(fd, offset, count)
      }, vfs);
      syscalls.bridgeAvailable = true;
      syscalls.eventfdHook = eventfdCall;
      if (config.syscallState) {
        syscalls.restoreState(config.syscallState);
        if (!config.resume) {
          syscalls.setIdentity(myPid, syscalls.currentPid);
        }
      }
      if (config.unixSocksSnapshot) {
        for (const [fd, sock] of config.unixSocksSnapshot) {
          unixSocks.set(fd, { ...sock });
        }
      }
      if (config.kernelPipesSnapshot) {
        for (const [fd, info] of config.kernelPipesSnapshot) {
          kernelPipes.set(fd, { ...info });
        }
        for (const fd of kernelPipes.keys()) {
          if (fd >= nextLocalFd) nextLocalFd = fd + 1;
        }
      }
      if (config.protArmed && protActive) {
        protReady = true;
        protActive.value = 1;
      }
      sharedMaps = (config.shmMaps ?? []).map((m) => {
        const start = m.addr + GUEST_MEM_BASE;
        const inherited = start + m.len <= memory.buffer.byteLength ? new Uint8Array(new Uint8Array(memory.buffer, start, m.len)) : new Uint8Array(new Uint8Array(shmArena, m.off, m.len));
        return { addr: m.addr, len: m.len, arenaOff: m.off, shadow: inherited };
      });
      sysvAttached = new Map((config.shmMaps ?? []).filter((m) => m.sysv !== void 0).map((m) => [m.addr, m.sysv]));
      publishIdentity(config.argv?.length ? config.argv : currentArgv);
      pc = config.resume ? cpu.getPC() : cpu.getPC() + 4n;
    } else {
      publishIdentity(config.argv);
      stamp("elf_begin");
      const prog = loadProgram(memory, elfBytes, config.argv, config.envp, readGuestFile);
      stamp("elf_done");
      cpu = new CpuState(memory, stateBase);
      cpu.setSP(prog.sp);
      new DataView(memory.buffer).setBigUint64(stateBase + 280, BigInt(9437184), true);
      syscalls = newSyscallHandler(prog.brkStart, {
        writeStdout: (d) => postWrite(1, d),
        writeStderr: (d) => postWrite(2, d),
        readStdin: (count) => readStdin(count),
        stdinHasData: () => stdinRing != null && (!stdinRing.empty || stdinRing.writerClosed),
        waitStdin: (timeoutMs) => waitStdin(timeoutMs),
        ioGeneration: () => readIoGeneration(),
        waitIo: (seen, timeoutMs) => waitIo(seen, timeoutMs),
        unixReady: (fd) => unixReady(fd),
        pipeReady: (fd) => pipeReady(fd),
        inotifyPending: (fd) => {
          const id = syscalls.inotifyIdOf(fd);
          return id < 0 ? 0 : inotifyCall({ op: "pending", id })[1];
        },
        mqReady: (fd) => mqReady(fd),
        pidExited: (pid) => pidExited(pid),
        readFileAt: (fd, offset, count) => readNativeFileAt(fd, offset, count)
      }, vfs);
      syscalls.bridgeAvailable = true;
      syscalls.eventfdHook = eventfdCall;
      syscalls.adoptImage(prog.segs);
      syscalls.setIdentity(config.pid, config.ppid);
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
    stamp("run_begin");
    procStartMs = performance.now();
    parkedMs = 0;
    parkDepth = 0;
    reportStages();
    reportMemSize();
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
      if (!pendingClone && !isNonLeaderThread()) {
        try {
          syscalls.closeNativeFds((fd) => handleBridgeCall("fs_close", { fd }, () => {
          }));
          syscalls.releaseEventfds();
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
    const shadowBase = threaded ? threadLayout.shadowBase : ZIG_SHADOW_BASE;
    const protection = protActive && protBase ? { protBase, protActive, memFault } : null;
    const zigRegion = translateRegionZig(view, pc, memory, shadowBase, false, threaded, regBaseGlobal, protection);
    if (!zigRegion) {
      const insn = view.getUint32(Number(pc) + GUEST_MEM_BASE, true);
      throw new Error(`[worker ${myPid}] translator failed at pc=0x${pc.toString(16)} (entry insn 0x${insn.toString(16).padStart(8, "0")})`);
    }
    if (regionCache.size >= REGION_CACHE_CAP) {
      const oldest = regionCache.keys().next().value;
      if (oldest !== void 0) regionCache.delete(oldest);
    }
    regionCache.set(pc, zigRegion);
    return zigRegion;
  }
  function drainSignalQueue() {
    for (const q of sigqDrain(sigQueue, mySlot)) {
      syscalls.raiseSignal(q.signo, { pid: q.pid, uid: q.uid, code: q.code, value: q.value });
    }
    return syscalls.hasPendingDeliverable();
  }
  function pollAndQueueSignals() {
    syscalls.checkTimers();
    if (performance.now() - lastCpuPublishMs > 50) {
      lastCpuPublishMs = performance.now();
      publishCpuTime();
    }
    return drainSignalQueue();
  }
  var swapPending = false;
  var lastReportedMem = 0;
  var growRefusedNoted = false;
  function reportMemSize() {
    lastReportedMem = memory.buffer.byteLength;
    self.postMessage({
      type: "mem_size",
      bytes: syscalls.footprintBytes(GUEST_STACK_TOP),
      grown: lastReportedMem
    });
  }
  function swapRequested() {
    return Atomics.load(procTable, procSlotOffset(mySlot) / 4 + PROC_SWAP_REQ) !== 0;
  }
  function postSwapImage(resumePC) {
    cpu.setPC(resumePC);
    const snapSize = Math.min(
      Number(syscalls.getMmapBase()) + GUEST_MEM_BASE + 65536,
      memory.buffer.byteLength
    );
    self.postMessage({
      type: "swap_image",
      memSnapshot: buildSparseSnapshot(memory.buffer, snapSize),
      syscallState: syscalls.getState(),
      kernelPipesSnapshot: [...kernelPipes.entries()],
      shmMaps: sharedMaps.map((m) => ({
        addr: m.addr,
        len: m.len,
        off: m.arenaOff,
        sysv: sysvAttached.get(m.addr)
      })),
      protArmed: protReady
    });
  }
  function setSleeping(on) {
    if (on) {
      if (parkDepth++ === 0) parkStartMs = performance.now();
    } else if (parkDepth > 0 && --parkDepth === 0) {
      parkedMs += performance.now() - parkStartMs;
    }
    Atomics.store(procTable, procSlotOffset(mySlot) / 4 + PROC_BLOCKED, on ? 1 : 0);
  }
  function cpuMs() {
    return Math.max(0, performance.now() - procStartMs - parkedTotalMs());
  }
  function parkedTotalMs() {
    return parkedMs + (parkDepth > 0 ? performance.now() - parkStartMs : 0);
  }
  var sysAccumMs = 0;
  var sysEnterMs = -1;
  var sysEnterParkedMs = 0;
  function syscallEnter() {
    sysEnterMs = performance.now();
    sysEnterParkedMs = parkedTotalMs();
  }
  function publishSyscall(nr) {
    Atomics.store(procTable, procSlotOffset(mySlot) / 4 + PROC_SYSCALL, nr | 0);
  }
  function sysMs() {
    return Math.min(sysAccumMs, cpuMs());
  }
  var lastCpuPublishMs = 0;
  function publishCpuTime() {
    const b = procSlotOffset(mySlot) / 4;
    const sys = sysMs();
    Atomics.store(procTable, b + PROC_UTIME_MS, Math.round(cpuMs() - sys) | 0);
    Atomics.store(procTable, b + PROC_STIME_MS, Math.round(sys) | 0);
    const kb = syscalls.rssKbHook ? syscalls.rssKbHook() : 0;
    if (kb > Atomics.load(procTable, b + PROC_MAXRSS_KB)) {
      Atomics.store(procTable, b + PROC_MAXRSS_KB, kb | 0);
    }
    Atomics.store(procTable, b + PROC_RUID, syscalls.getCreds().ruid | 0);
  }
  function sleepingWait(i32, idx, expected, timeout) {
    setSleeping(true);
    try {
      return timeout === void 0 ? Atomics.wait(i32, idx, expected) : Atomics.wait(i32, idx, expected, timeout);
    } finally {
      setSleeping(false);
    }
  }
  function awaitMailbox(what) {
    const start = performance.now();
    let warned = false;
    setSleeping(true);
    try {
      while (Atomics.load(myMailbox, MBOX_READY) === 0) {
        Atomics.wait(myMailbox, MBOX_READY, 0, 100);
        if (!warned && performance.now() - start > 2e4) {
          warned = true;
          self.postMessage({
            type: "worker_error",
            detail: `pid ${myPid} has waited 20s for a '${what}' reply from the kernel`
          });
        }
      }
    } finally {
      setSleeping(false);
    }
  }
  function interruptibleWaiter(i32, idx, expected) {
    const rem = syscalls.timerRemainingMs();
    sleepingWait(i32, idx, expected, rem > 0 ? Math.min(1e3, rem) : 1e3);
    if (swapRequested()) {
      swapPending = true;
      return true;
    }
    return pollAndQueueSignals();
  }
  function takeSwapAtSyscall(svcPC) {
    if (!swapPending) return false;
    swapPending = false;
    if (BigInt.asIntN(64, cpu.getReg(0)) !== -4n) return false;
    postSwapImage(svcPC);
    return true;
  }
  function stopSelf(sig) {
    const base = procSlotOffset(mySlot) / 4;
    Atomics.store(procTable, base + PROC_STOP_SIG, sig);
    Atomics.store(procTable, base + PROC_STOP_SEEN, 0);
    Atomics.store(procTable, base + PROC_STATE, PROC_STOPPED);
    wakeParent();
    while (Atomics.load(procTable, base + PROC_STATE) === PROC_STOPPED) {
      Atomics.store(procTable, base + PROC_WAIT_FUTEX, 0);
      if (Atomics.load(procTable, base + PROC_STATE) !== PROC_STOPPED) break;
      sleepingWait(procTable, base + PROC_WAIT_FUTEX, 0, 1e3);
      if (Atomics.load(procTable, base + PROC_PID) <= 0) break;
    }
    Atomics.compareExchange(procTable, base + PROC_STATE, PROC_STOPPED, PROC_RUNNING);
  }
  function wakeParent() {
    wakeParentOf(myPid);
  }
  function wakeParentOf(pid) {
    let ppid = -1;
    for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
      const base = procSlotOffset(slot) / 4;
      if (Atomics.load(procTable, base + PROC_PID) === pid) {
        ppid = Atomics.load(procTable, base + PROC_PPID);
        break;
      }
    }
    if (ppid <= 0) return;
    wakeWatcher(ppid);
  }
  function wakeWatcher(pid) {
    if (pid <= 0) return;
    for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
      const base = procSlotOffset(slot) / 4;
      if (Atomics.load(procTable, base + PROC_PID) !== pid) continue;
      sigqPush(sigQueue, slot, { signo: 17, code: 128, pid: 0, uid: 0, value: 0n });
      Atomics.store(procTable, base + PROC_WAIT_FUTEX, 1);
      Atomics.notify(procTable, base + PROC_WAIT_FUTEX);
      return;
    }
  }
  function deliverQueuedSignals(pc) {
    if (!syscalls.pendingSignals.length) return pc;
    syscalls.sigSync();
    cpu.setPC(pc);
    const blocked = [];
    let result = pc;
    const traceBase = procSlotOffset(mySlot) / 4;
    while (syscalls.pendingSignals.length > 0) {
      let sig = syscalls.pendingSignals.shift();
      if (sig !== 9 && Atomics.load(procTable, traceBase + PROC_TRACER) !== 0) {
        sig = ptraceStop(sig);
        if (sig === 0) continue;
        if (Atomics.load(procTable, traceBase + PROC_PID) <= 0) return null;
      }
      const r = syscalls.deliverSignal(cpu, memory, sig);
      if (r.action === "exit") {
        postExit(r.exitCode ?? 128 + sig);
        return null;
      }
      if (r.action === "stop") {
        stopSelf(r.exitCode ?? sig);
        continue;
      }
      if (!r.delivered) {
        blocked.push(sig);
        continue;
      }
      if (r.action === "continue") {
        result = cpu.getPC();
        break;
      }
    }
    if (blocked.length) syscalls.pendingSignals.unshift(...blocked);
    return result;
  }
  function runLoop(startPC) {
    let pc = startPC;
    let totalInsns = 0;
    let view = new DataView(memory.buffer);
    let lastSignalCheck = 0;
    let blockCount = 0;
    syscalls.protHook = setPageProt;
    syscalls.growFailed = (cur, min, want, cap, err) => {
      if (growRefusedNoted) return;
      growRefusedNoted = true;
      self.postMessage({
        type: "grow_refused",
        detail: `pid ${myPid} grow refused: have ${cur} pages, need ${min}, asked ${want}, cap ${cap} (${err})`
      });
    };
    syscalls.leaseHook = (op, nativeFd, path, arg, breakSig, isOwner, isRegular) => {
      Atomics.store(myMailbox, MBOX_READY, 0);
      self.postMessage({ type: "lease_req", op, path, nativeFd, arg, breakSig, isOwner, isRegular });
      awaitMailbox("lease");
      return Atomics.load(myMailbox, MBOX_VALUE0);
    };
    syscalls.credsHook = (ruid, euid, suid, rgid) => {
      const b = procCredOffset(mySlot) / 4;
      Atomics.store(procCreds, b, ruid | 0);
      Atomics.store(procCreds, b + 1, euid | 0);
      Atomics.store(procCreds, b + 2, suid | 0);
      Atomics.store(procCreds, b + 3, rgid | 0);
    };
    syscalls.tracedHook = () => Atomics.load(procTable, procSlotOffset(mySlot) / 4 + PROC_TRACER) !== 0;
    syscalls.sigpendingHook = (limit) => {
      Atomics.store(procTable, procSlotOffset(mySlot) / 4 + PROC_RLIM_SIGPENDING, limit | 0);
    };
    syscalls.cpuMsHook = cpuMs;
    syscalls.sysMsHook = sysMs;
    syscalls.clockOffsetHook = {
      get() {
        for (; ; ) {
          const s12 = Atomics.load(globalWords, GLOBAL_CLK_SEQ);
          if (s12 & 1) continue;
          const sec = Atomics.load(globalWords, GLOBAL_CLK_OFF_SEC);
          const nsec = Atomics.load(globalWords, GLOBAL_CLK_OFF_NSEC);
          if (Atomics.load(globalWords, GLOBAL_CLK_SEQ) === s12) return BigInt(sec) * 1000000000n + BigInt(nsec);
        }
      },
      set(ns) {
        Atomics.add(globalWords, GLOBAL_CLK_SEQ, 1);
        Atomics.store(globalWords, GLOBAL_CLK_OFF_SEC, Number(ns / 1000000000n) | 0);
        Atomics.store(globalWords, GLOBAL_CLK_OFF_NSEC, Number(ns % 1000000000n) | 0);
        Atomics.add(globalWords, GLOBAL_CLK_SEQ, 1);
      }
    };
    syscalls.setSystemBootMs(
      Atomics.load(globalWords, GLOBAL_BOOT_MS_HI) * 4294967296 + (Atomics.load(globalWords, GLOBAL_BOOT_MS_LO) >>> 0)
    );
    syscalls.haltHook = (restart) => self.postMessage({ type: "halt", restart });
    const utsBytes = new Uint8Array(kernelSAB, GLOBAL_STATE_OFFSET, 4096);
    syscalls.utsHook = {
      get(which) {
        const off = which === "host" ? GLOBAL_HOSTNAME_OFFSET : GLOBAL_DOMAINNAME_OFFSET;
        let end = off;
        while (end < off + GLOBAL_UTS_LEN && utsBytes[end] !== 0) end++;
        return new TextDecoder().decode(utsBytes.subarray(off, end));
      },
      set(which, name) {
        const off = which === "host" ? GLOBAL_HOSTNAME_OFFSET : GLOBAL_DOMAINNAME_OFFSET;
        const enc = new TextEncoder().encode(name);
        utsBytes.fill(0, off, off + GLOBAL_UTS_LEN);
        utsBytes.set(enc.subarray(0, GLOBAL_UTS_LEN - 1), off);
      }
    };
    syscalls.inotifyHook = reportInotify;
    syscalls.inotifyFdinfo = inotifyFdinfo;
    syscalls.fdPosition = nativeFdOffset;
    syscalls.pdeathHook = (sig) => Atomics.store(procTable, procSlotOffset(mySlot) / 4 + PROC_PDEATHSIG, sig | 0);
    syscalls.commHook = (name) => {
      const commBytes = new Uint8Array(16);
      commBytes.set(new TextEncoder().encode(name).subarray(0, 15));
      new Uint8Array(kernelSAB, procSlotOffset(mySlot) + PROC_COMM * 4, 16).set(commBytes);
    };
    syscalls.prioHook = {
      list: () => {
        const out = [];
        for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
          const b = procSlotOffset(slot) / 4;
          const pid = Atomics.load(procTable, b + PROC_PID);
          if (pid === 0) continue;
          const listState = Atomics.load(procTable, b + PROC_STATE);
          if (listState !== PROC_RUNNING && listState !== PROC_STOPPED) continue;
          out.push({
            pid,
            pgid: Atomics.load(procTable, b + PROC_PGID),
            ruid: Atomics.load(procCreds, procCredOffset(slot) / 4),
            nice: Atomics.load(procTable, b + PROC_NICE)
          });
        }
        return out;
      },
      setNice: (pid, nice) => {
        for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
          const b = procSlotOffset(slot) / 4;
          if (Atomics.load(procTable, b + PROC_PID) !== pid) continue;
          Atomics.store(procTable, b + PROC_NICE, nice | 0);
          return;
        }
      }
    };
    syscalls.rssKbHook = () => Math.floor(syscalls.footprintBytes(GUEST_STACK_TOP) / 1024);
    syscalls.publishCreds();
    syscalls.publishSigpending();
    while (pc !== HALT_PC) {
      blockCount++;
      const region = getRegion(pc);
      try {
        pc = region.run();
      } catch (e) {
        if (e !== MEM_FAULT) throw e;
        const rawFault = pendingFault?.addr ?? 0;
        let faultCtx = "";
        if (tracing) {
          const fpc = Number(view.getBigUint64(stateBase + PC_OFFSET, true));
          if (fpc + GUEST_MEM_BASE + 4 <= memory.buffer.byteLength) {
            const words = [];
            for (let k = -6; k <= 3; k++) {
              const a = fpc + k * 4 + GUEST_MEM_BASE;
              if (a >= 0 && a + 4 <= memory.buffer.byteLength) {
                words.push((k === 0 ? "*" : "") + view.getUint32(a, true).toString(16).padStart(8, "0"));
              }
            }
            faultCtx += " text=" + words.join(",");
          }
        }
        if (tracing && rawFault >= 32) {
          const fb = new Uint8Array(memory.buffer, rawFault - 32, 32);
          faultCtx += " before=" + Array.from(fb).map((b) => b.toString(16).padStart(2, "0")).join("");
        }
        if (tracing) self.postMessage({ type: "trace", line: `pid${myPid} FAULT raw=0x${(rawFault >>> 0).toString(16)} guest=0x${(rawFault - GUEST_MEM_BASE >>> 0).toString(16)} in=${syscalls.addrInfo(rawFault - GUEST_MEM_BASE >>> 0)} write=${pendingFault?.isWrite ?? 0} pc=0x${pc.toString(16)} sp=0x${cpu.getSP().toString(16)} tp=0x${new DataView(memory.buffer).getBigUint64(stateBase + TPIDR_OFFSET, true).toString(16)} x=${[0, 1, 2, 3, 8, 9].map((r) => cpu.getReg(r).toString(16)).join(",")} lr=0x${cpu.getReg(30).toString(16)}${faultCtx}` });
        const fAddr = rawFault >= GUEST_MEM_BASE ? rawFault - GUEST_MEM_BASE : rawFault;
        const faultPC = view.getBigUint64(stateBase + PC_OFFSET, true);
        const grown = syscalls.growStackFor(fAddr);
        if (grown) {
          setPageProt(grown.start, grown.end - grown.start, grown.prot);
          pc = faultPC;
          view = new DataView(memory.buffer);
          continue;
        }
        cpu.setPC(faultPC);
        syscalls.faultAddr = BigInt(fAddr);
        const faultPage = protPainted() ? new Uint8Array(memory.buffer)[PROT_TABLE_BASE + (fAddr >>> 12)] : -1;
        const noBacking = faultPage >= 0 ? faultPage === PAGE_NO_BACKING2 : syscalls.isNoBacking(fAddr);
        self.postMessage({
          type: "klog",
          line: `fault ${noBacking ? "SIGBUS" : "SIGSEGV"} addr=0x${fAddr.toString(16)} pc=0x${faultPC.toString(16)} ${syscalls.describeAddr(fAddr)}`
        });
        syscalls.raiseFaultSignal(
          noBacking ? 7 : 11
          /* SIGSEGV */
        );
        const next = deliverQueuedSignals(cpu.getPC());
        if (next === null) return;
        pc = next;
        view = new DataView(memory.buffer);
        continue;
      }
      if (region.lastExitReason === 3) {
        if (tracing) {
          let word = 0;
          try {
            word = new DataView(memory.buffer).getUint32(Number(pc) + GUEST_MEM_BASE, true);
          } catch {
          }
          self.postMessage({ type: "trace", line: `pid${myPid} SIGILL pc=0x${pc.toString(16)} insn=0x${(word >>> 0).toString(16).padStart(8, "0")} ${syscalls.describeAddr(Number(pc))}` });
        }
        cpu.setPC(pc);
        syscalls.faultAddr = pc;
        syscalls.raiseFaultSignal(4);
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
        const realPC = pc === SVC_PC ? view.getBigUint64(stateBase + PC_OFFSET, true) : pc;
        if (swapRequested()) {
          postSwapImage(realPC);
          return;
        }
        if (Math.abs(memory.buffer.byteLength - lastReportedMem) >= 8 * 1024 * 1024) reportMemSize();
        if (pollAndQueueSignals()) {
          const next = deliverQueuedSignals(realPC);
          if (next === null) return;
          if (next !== realPC) {
            pc = next;
            continue;
          }
        }
      }
      if (pc === SVC_PC) {
        const svcPC = view.getBigUint64(stateBase + PC_OFFSET, true);
        syscallEnter();
        publishSyscall(Number(cpu.getReg(8)));
        const nr = Number(cpu.getReg(8));
        const origX0 = cpu.getReg(0);
        lastSyscallNr = nr;
        const tA0 = tracing ? cpu.getReg(0) : 0n;
        const tA1 = tracing ? cpu.getReg(1) : 0n;
        const tA2 = tracing ? cpu.getReg(2) : 0n;
        tA3 = tracing ? cpu.getReg(3) : 0n;
        tA4 = tracing ? cpu.getReg(4) : 0n;
        if (tracing) self.postMessage({ type: "trace", line: `pid${myPid} -> ${SYSNAME[nr] ?? nr}(${cpu.getReg(0)}, ${cpu.getReg(1)}, ${cpu.getReg(2)})` });
        const tPath = tracing && (nr === 56 || nr === 79 || nr === 35 || nr === 36 || nr === 38 || nr === 37) ? readCString(Number(cpu.getReg(1))) : null;
        if (sharedMaps.length) reconcileShared();
        if (nr === 260) {
          handleWait4(cpu, view);
          const eintr = BigInt.asIntN(64, cpu.getReg(0)) === -4n;
          syscallDone();
          if (takeSwapAtSyscall(svcPC)) return;
          const restart = eintr && syscalls.hasPendingDeliverable() && syscalls.anyPendingRestart();
          const retPC = restart ? svcPC : svcPC + 4n;
          if (restart) cpu.setReg(0, origX0);
          if (syscalls.hasPendingDeliverable()) {
            const next = deliverQueuedSignals(retPC);
            if (next === null) return;
            pc = next;
          } else {
            pc = retPC;
          }
          continue;
        }
        if (nr === 139) {
          syscallDone();
          pc = syscalls.handleSigreturn(cpu, memory);
          const next = deliverQueuedSignals(pc);
          if (next === null) return;
          pc = next;
          continue;
        }
        if (nr === 129 || nr === 130 || nr === 131 || nr === 138 || nr === 240) {
          handleKill(nr, cpu, view);
          syscallDone();
          pc = svcPC + 4n;
          if (drainSignalQueue()) {
            const next = deliverQueuedSignals(pc);
            if (next === null) return;
            pc = next;
          }
          continue;
        }
        if (nr === 154) {
          handleSetpgid(cpu);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 155) {
          handleGetpgid(cpu);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 156) {
          handleGetsid(cpu);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 157) {
          handleSetsid(cpu);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 133) {
          handleSigsuspend(cpu);
          syscallDone();
          pc = svcPC + 4n;
          const next = deliverQueuedSignals(pc);
          if (next === null) return;
          pc = next;
          continue;
        }
        if (nr === 73 && Number(cpu.getReg(1)) === 0) {
          handlePpollWait(cpu);
          syscallDone();
          pc = svcPC + 4n;
          const next = deliverQueuedSignals(pc);
          if (next === null) return;
          pc = next;
          continue;
        }
        if (nr === 220) {
          const cloneFlags = Number(cpu.getReg(0));
          const CLONE_THREAD = 65536;
          if (cloneFlags & CLONE_THREAD) {
            if (handleCloneThread(cpu, svcPC)) return;
            syscallDone();
            pc = svcPC + 4n;
            continue;
          }
        }
        if (nr === 26 || nr === 27 || nr === 28 || (nr === 63 || nr === 57) && syscalls.inotifyIdOf(Number(cpu.getReg(0))) >= 0) {
          handleInotify(nr, cpu, view);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 95) {
          handleWaitid(cpu, view);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 71) {
          handleSendfile(cpu, view);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 63 && syscalls.getFdEntry(Number(cpu.getReg(0)))?.type === "signalfd") {
          const sfd = Number(cpu.getReg(0));
          const sbuf = Number(cpu.getReg(1));
          const scount = Number(cpu.getReg(2));
          for (; ; ) {
            const r = syscalls.signalfdRead(sfd, sbuf, scount, view, new Uint8Array(memory.buffer));
            if (r !== -11) {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(r)));
              break;
            }
            if (syscalls.isNonBlockingFd(sfd)) {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(-11)));
              break;
            }
            if (pollAndQueueSignals()) {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(-4)));
              break;
            }
            waitIo(readIoGeneration(), 20);
          }
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 75 || nr === 76 || nr === 77) {
          handleSplice(nr, cpu, view);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr >= 118 && nr <= 123 || nr === 125 || nr === 126 || nr === 127 || nr === 274 || nr === 275 || nr === 30 || nr === 31 || nr === 168) {
          handleSched(nr, cpu, view);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if ((nr === 113 || nr === 114) && Number(BigInt.asIntN(32, cpu.getReg(0))) < 0) {
          handleCpuClock(nr, cpu, view);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 424 || nr === 434) {
          handlePidfd(nr, cpu, view);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr >= 190 && nr <= 193) {
          handleSysvSem(nr, cpu, view);
          syscallDone();
          pc = svcPC + 4n;
          if (drainSignalQueue()) {
            const next = deliverQueuedSignals(pc);
            if (next === null) return;
            pc = next;
          }
          continue;
        }
        if (nr >= 180 && nr <= 185 || nr === 57 && syscalls.mqIdOf(Number(cpu.getReg(0))) >= 0) {
          handlePosixMq(nr, cpu, view);
          syscallDone();
          pc = svcPC + 4n;
          if (drainSignalQueue()) {
            const next = deliverQueuedSignals(pc);
            if (next === null) return;
            pc = next;
          }
          continue;
        }
        if (nr === 186 || nr === 187 || nr === 188 || nr === 189) {
          handleSysvMsg(nr, cpu, view);
          syscallDone();
          pc = svcPC + 4n;
          if (drainSignalQueue()) {
            const next = deliverQueuedSignals(pc);
            if (next === null) return;
            pc = next;
          }
          continue;
        }
        if (nr === 194 || nr === 195 || nr === 196 || nr === 197) {
          handleSysvShm(nr, cpu, view);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 285 && syscalls.fdNative(Number(BigInt.asIntN(32, cpu.getReg(0)))) >= 0 && syscalls.fdNative(Number(BigInt.asIntN(32, cpu.getReg(2)))) >= 0) {
          handleCopyFileRange(cpu, view);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 137) {
          handleSigtimedwait(cpu, view);
          syscallDone();
          pc = svcPC + 4n;
          if (drainSignalQueue()) {
            const next = deliverQueuedSignals(pc);
            if (next === null) return;
            pc = next;
          }
          continue;
        }
        if (nr === 117) {
          handlePtrace(cpu, view);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 98 || nr === 422 || nr === 449) {
          if (nr === 449) handleFutexWaitv(cpu);
          else handleFutex(cpu);
          syscallDone();
          pc = svcPC + 4n;
          if (drainSignalQueue()) {
            const next = deliverQueuedSignals(pc);
            if (next === null) return;
            pc = next;
          }
          continue;
        }
        if (nr === 157) {
          Atomics.store(procTable, procSlotOffset(mySlot) / 4 + PROC_PGID, myPid);
          cpu.setReg(0, BigInt(myPid));
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 25) {
          const lc = Number(cpu.getReg(1));
          if (lc === 5 || lc === 6 || lc === 7 || lc === 12 || lc === 13 || lc === 14) {
            handleFcntlLock(cpu, view, lc, "posix");
            syscallDone();
            pc = svcPC + 4n;
            continue;
          }
          if (lc === 36 || lc === 37 || lc === 38) {
            handleFcntlLock(cpu, view, lc, "ofd");
            syscallDone();
            pc = svcPC + 4n;
            continue;
          }
        }
        if (nr === 199 && Number(cpu.getReg(0)) === 1) {
          handleUnixSocketpair(cpu, view, Number(cpu.getReg(1)));
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 198 && Number(cpu.getReg(0)) === 1) {
          handleUnixSocket(cpu, Number(cpu.getReg(1)));
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 198 && Number(cpu.getReg(0)) === 16) {
          handleNetlinkSocket(cpu, Number(cpu.getReg(1)));
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (unixSocks.has(Number(cpu.getReg(0))) && (nr === 200 || nr === 201 || nr === 202 || nr === 242 || nr === 203 || nr === 63 || nr === 64 || nr === 57 || nr === 206 || nr === 207 || nr === 211 || nr === 212 || nr === 204 || nr === 205)) {
          if (nr === 204 || nr === 205) handleUnixGetName(cpu, view, nr === 205);
          else if (nr === 200) handleUnixBind(cpu, view);
          else if (nr === 201) handleUnixListen(cpu);
          else if (nr === 202 || nr === 242) handleUnixAccept(cpu, view);
          else if (nr === 203) handleUnixConnect(cpu, view);
          else if (nr === 211) handleUnixSendmsg(cpu, view);
          else if (nr === 212) handleUnixRecvmsg(cpu, view);
          else if (nr === 206) handleUnixSendto(cpu, view);
          else if (nr === 207) handleUnixRecvfrom(cpu, view);
          else handleUnixIO(nr, cpu);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 32) {
          handleFlock(cpu);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 57 && usedLocks) {
          releaseLocksForClosingFd(Number(cpu.getReg(0)));
        }
        if (nr === 101 || nr === 115) {
          const abs = nr === 115 && (Number(cpu.getReg(1)) & 1) !== 0;
          const reqAddr = Number(nr === 115 ? cpu.getReg(2) : cpu.getReg(0));
          const remAddr = Number(nr === 115 ? cpu.getReg(3) : cpu.getReg(1));
          handleNanosleep(cpu, view, reqAddr, remAddr, abs, nr === 115 ? Number(cpu.getReg(0)) : 1);
          syscallDone();
          pc = svcPC + 4n;
          if (syscalls.hasPendingDeliverable()) {
            const next = deliverQueuedSignals(pc);
            if (next === null) return;
            pc = next;
          }
          continue;
        }
        if (nr === 59) {
          handlePipe2(cpu, view);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 24 && Number(cpu.getReg(0)) !== Number(cpu.getReg(1))) {
          const dupTo = Number(cpu.getReg(1));
          const shadow = kernelPipes.get(dupTo);
          if (shadow && !kernelPipes.has(Number(cpu.getReg(0)))) {
            pipeRef(shadow.pipeOff, shadow.mode, -1);
            kernelPipes.delete(dupTo);
          }
          if (unixSocks.has(dupTo) && !unixSocks.has(Number(cpu.getReg(0)))) {
            closeUnixSock(dupTo);
          }
        }
        if (nr === 24 && kernelPipes.has(Number(cpu.getReg(0)))) {
          handleKernelPipeDup3(cpu);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if ((nr === 23 || nr === 25 && (Number(cpu.getReg(1)) === 0 || Number(cpu.getReg(1)) === 1030)) && kernelPipes.has(Number(cpu.getReg(0)))) {
          handleKernelPipeDup(cpu, nr === 25 ? Number(cpu.getReg(2)) : 0, nr === 25 && Number(cpu.getReg(1)) === 1030);
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if (nr === 29 && Number(cpu.getReg(1)) === 21531 && kernelPipes.has(Number(cpu.getReg(0)))) {
          const info = kernelPipes.get(Number(cpu.getReg(0)));
          const ring = new PipeRing(shmArena, info.pipeOff);
          const outPtr = Number(cpu.getReg(2));
          if (badGuestPtr(new Uint8Array(memory.buffer), outPtr, 4)) {
            cpu.setReg(0, BigInt.asUintN(64, BigInt(-14)));
          } else {
            view.setInt32(outPtr + GUEST_MEM_BASE, ring.bytesAvailable, true);
            cpu.setReg(0, 0n);
          }
          syscallDone();
          pc = svcPC + 4n;
          continue;
        }
        if ((nr === 63 || nr === 64 || nr === 65 || nr === 66 || nr === 57) && kernelPipes.has(Number(cpu.getReg(0)))) {
          handleKernelPipeIO(nr, cpu, view);
          const eintr = BigInt.asIntN(64, cpu.getReg(0)) === -4n;
          syscallDone();
          const restart = eintr && syscalls.hasPendingDeliverable() && syscalls.anyPendingRestart();
          const retPC = restart ? svcPC : svcPC + 4n;
          if (restart) cpu.setReg(0, origX0);
          if (syscalls.hasPendingDeliverable()) {
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
        syscallDone();
        if (takeSwapAtSyscall(svcPC)) return;
        view = new DataView(memory.buffer);
        switch (result.status) {
          case "ok":
            pc = svcPC + 4n;
            break;
          case "exit":
            postExit(result.code, result.group === true);
            return;
          case "fork":
            pc = handleFork(result);
            break;
          case "execve":
            try {
              pc = handleExec(result.path, result.argv, result.envp);
              Atomics.store(procTable, procSlotOffset(mySlot) / 4 + PROC_EXECED, 1);
            } catch (e) {
              const errno = e?.errno;
              if (typeof errno === "number") {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(-errno)));
                pc = svcPC + 4n;
              } else throw e;
            }
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
            if (syscalls.takeRetry()) {
              if (pollAndQueueSignals()) {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(-4)));
                pc = svcPC + 4n;
              } else {
                waitIo(readIoGeneration(), 20);
                pc = svcPC;
              }
              break;
            }
            syscalls.noteResult(nr, BigInt.asIntN(64, cpu.getReg(0)));
            pc = svcPC + 4n;
            break;
          default:
            pc = svcPC + 4n;
            break;
        }
        traceCall(nr, tA0, tA1, tA2, tPath);
        if (drainSignalQueue()) {
          const next = deliverQueuedSignals(pc);
          if (next === null) return;
          pc = next;
        }
        continue;
      }
    }
    postExit(0);
  }
  function takeForkStage(bytes) {
    Atomics.store(myMailbox, MBOX_READY, 0);
    self.postMessage({ type: "fork_stage", bytes });
    awaitMailbox("fork_stage");
    return Atomics.load(myMailbox, MBOX_VALUE0);
  }
  function buildSparseSnapshot(buffer, totalLen, ranges, stage = false) {
    const pageCount = Math.ceil(totalLen / MEM_PAGE_SIZE);
    const wordsPerPage = MEM_PAGE_SIZE >>> 2;
    const u32 = new Uint32Array(buffer, 0, pageCount * MEM_PAGE_SIZE >>> 2);
    const bytes = new Uint8Array(buffer);
    const idx = [];
    const seen = new Uint8Array(pageCount);
    const scan = ranges ?? [{ start: 0, end: pageCount * MEM_PAGE_SIZE }];
    for (const r of scan) {
      const first = Math.max(0, Math.floor(r.start / MEM_PAGE_SIZE));
      const last = Math.min(pageCount, Math.ceil(r.end / MEM_PAGE_SIZE));
      for (let p = first; p < last; p++) {
        if (seen[p]) continue;
        seen[p] = 1;
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
    }
    idx.sort((a, b) => a - b);
    const need = idx.length * MEM_PAGE_SIZE;
    const packed = (dst) => {
      for (let i = 0; i < idx.length; i++) {
        const off = idx[i] * MEM_PAGE_SIZE;
        dst.set(bytes.subarray(off, off + MEM_PAGE_SIZE), i * MEM_PAGE_SIZE);
      }
    };
    if (stage && need > 0) {
      const off = takeForkStage(need);
      if (off >= 0) {
        packed(new Uint8Array(shmArena, off, need));
        return {
          pages: Uint32Array.from(idx),
          stageOff: off,
          byteLen: need,
          totalLen: pageCount * MEM_PAGE_SIZE
        };
      }
    }
    const data = new Uint8Array(need);
    packed(data);
    return { pages: Uint32Array.from(idx), data: data.buffer, totalLen: pageCount * MEM_PAGE_SIZE };
  }
  function applySparseSnapshot(mem, snap) {
    if (snap.totalLen > mem.buffer.byteLength) {
      const need = Math.ceil(snap.totalLen / MEM_PAGE_SIZE);
      const cur = mem.buffer.byteLength / MEM_PAGE_SIZE;
      if (need > cur) mem.grow(need - cur);
    }
    const dst = new Uint8Array(mem.buffer);
    const data = snap.data ? new Uint8Array(snap.data) : new Uint8Array(shmArena, snap.stageOff, snap.byteLen);
    for (let i = 0; i < snap.pages.length; i++) {
      dst.set(data.subarray(i * MEM_PAGE_SIZE, (i + 1) * MEM_PAGE_SIZE), snap.pages[i] * MEM_PAGE_SIZE);
    }
  }
  function handleFork(req) {
    const resumePC = req.resumePC;
    const tFork0 = performance.now();
    const snapSize = Math.min(
      Number(syscalls.getMmapBase()) + GUEST_MEM_BASE + 65536,
      memory.buffer.byteLength
    );
    const guestRanges = [
      ...syscalls.snapshotRanges(GUEST_STACK_TOP),
      // MAP_SHARED regions are shadowed into guest memory and reconciled at every syscall boundary,
      // so a child that starts with zeros there does not merely miss data -- it PUBLISHES the zeros
      // over the arena and destroys what the parent wrote. they are in the vma list too, but this is
      // the one category where being wrong is silent and destructive, so it is stated outright.
      ...sharedMaps.map((m) => ({ start: m.addr, end: m.addr + m.len }))
    ].map((r) => ({
      start: r.start + GUEST_MEM_BASE,
      end: Math.min(r.end + GUEST_MEM_BASE, snapSize)
    }));
    let memSnapshot;
    try {
      memSnapshot = buildSparseSnapshot(
        memory.buffer,
        snapSize,
        [{ start: 0, end: GUEST_MEM_BASE }, ...guestRanges],
        true
      );
    } catch (e) {
      self.postMessage({
        type: "worker_error",
        detail: `pid ${myPid} fork snapshot failed (snapSize=${snapSize}, ranges=${guestRanges.length}): ${e}`
      });
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-12)));
      return resumePC;
    }
    if (tracing) self.postMessage({ type: "trace", line: `pid${myPid} FORKRANGES snapSize=0x${snapSize.toString(16)} ` + guestRanges.map((r) => `[0x${(r.start - GUEST_MEM_BASE).toString(16)},0x${(r.end - GUEST_MEM_BASE).toString(16)})`).join(" ") });
    const syscallState = syscalls.getState();
    const forkNativeDups = /* @__PURE__ */ new Map();
    for (const [guestFd, entry] of syscallState.fdTable) {
      if (entry.type !== "native") continue;
      let newFd = forkNativeDups.get(entry.nativeFd);
      if (newFd === void 0) {
        newFd = -1;
        handleBridgeCall("fs_dup", { fd: entry.nativeFd }, (resp) => {
          newFd = resp.ret;
        });
        forkNativeDups.set(entry.nativeFd, newFd);
      }
      if (newFd < 0) continue;
      syscallState.fdTable.set(guestFd, { ...entry, nativeFd: newFd });
    }
    const childNativeRefs = /* @__PURE__ */ new Map();
    for (const e of syscallState.fdTable.values()) {
      if (e.type === "native") childNativeRefs.set(e.nativeFd, (childNativeRefs.get(e.nativeFd) ?? 0) + 1);
    }
    syscallState.nativeRefs = childNativeRefs;
    const forkSocketIds = syscalls.inheritedSocketIds();
    for (const sockId of forkSocketIds) handleBridgeCall("net_ref", { socketId: sockId }, () => {
    });
    const unixSocksSnapshot = refUnixSocksForChild();
    cpu.setPC(resumePC);
    const kernelPipesSnapshot = [...kernelPipes.entries()];
    for (const [, info] of kernelPipesSnapshot) {
      const ring = new PipeRing(shmArena, info.pipeOff);
      if (info.mode === "write") ring.addWriter();
      else ring.addReader();
    }
    self.postMessage({
      type: "fork",
      memSnapshot,
      syscallState,
      kernelPipesSnapshot,
      unixSocksSnapshot,
      inotifyIds: syscalls.inotifyInstanceIds(),
      shmBlocks: sharedMaps.map((m) => m.arenaOff),
      // which segment is attached where travels with the mapping: shmdt names an ADDRESS, and a
      // child that inherited an attach has to be able to detach it.
      shmMaps: sharedMaps.map((m) => ({
        addr: m.addr,
        len: m.len,
        off: m.arenaOff,
        sysv: sysvAttached.get(m.addr)
      })),
      protArmed: protReady,
      resumePC: resumePC.toString(),
      // clone(2)'s child-context arguments. these are not thread-only: vfork, posix_spawn and
      // system() are all fork-style clones WITH a stack, and the child has to start on it.
      cloneFlags: req.flags,
      childStack: req.childStack.toString(),
      childTls: req.tls.toString(),
      childCtid: req.ctid.toString()
    });
    const tPost = performance.now();
    Atomics.store(myMailbox, MBOX_READY, 0);
    awaitMailbox("fork");
    const childPid = Atomics.load(myMailbox, MBOX_VALUE0);
    if (tracing) self.postMessage({ type: "trace", line: `pid${myPid} FORK snapshot=${(tPost - tFork0).toFixed(1)}ms reply=${(performance.now() - tPost).toFixed(1)}ms` });
    if (childPid < 0) {
      for (const [, info] of kernelPipesSnapshot) {
        const ring = new PipeRing(shmArena, info.pipeOff);
        if (info.mode === "write") ring.closeWriter();
        else ring.closeReader();
      }
      for (const newFd of forkNativeDups.values()) {
        if (newFd >= 0) handleBridgeCall("fs_close", { fd: newFd }, () => {
        });
      }
      for (const sockId of forkSocketIds) handleBridgeCall("net_close", { socketId: sockId }, () => {
      });
      releaseUnixSocksSnapshot(unixSocksSnapshot);
      cpu.setReg(0, BigInt.asUintN(64, BigInt(childPid)));
      return resumePC;
    }
    if (req.flags & CLONE_PARENT_SETTID && req.ptid !== 0n) {
      const p = Number(req.ptid);
      if (!syscalls.ptrFaults(new Uint8Array(memory.buffer), p, 4)) {
        new DataView(memory.buffer).setInt32(p + GUEST_MEM_BASE, childPid, true);
      }
    }
    for (const efd of syscalls.inheritedEventfdIds()) eventfdCall({ op: "ref", id: efd, pid: childPid });
    if (req.flags & CLONE_VFORK) {
      const cslot = slotOfPid(childPid);
      if (cslot >= 0) {
        const cbase = cslot;
        while (Atomics.load(procTable, cbase + PROC_VFORK) === 1 && Atomics.load(procTable, cbase + PROC_PID) === childPid && (Atomics.load(procTable, cbase + PROC_STATE) === PROC_RUNNING || Atomics.load(procTable, cbase + PROC_STATE) === PROC_STOPPED)) {
          setSleeping(true);
          try {
            sleepingWait(procTable, cbase + PROC_VFORK, 1, 50);
          } finally {
            setSleeping(false);
          }
          if (pollAndQueueSignals()) break;
        }
      }
    }
    cpu.setReg(0, BigInt(childPid));
    return resumePC;
  }
  var CLONE_PARENT_SETTID = 1048576;
  var CLONE_VFORK = 16384;
  function refUnixSocksForChild() {
    const snap = [...unixSocks.entries()].map(([fd, s]) => [fd, { ...s }]);
    for (const [, sock] of snap) {
      if (sock.rx !== void 0) new PipeRing(shmArena, sock.rx).addReader();
      if (sock.tx !== void 0) new PipeRing(shmArena, sock.tx).addWriter();
    }
    return snap;
  }
  function releaseUnixSocksSnapshot(snap) {
    for (const [, sock] of snap) {
      if (sock.rx !== void 0) new PipeRing(shmArena, sock.rx).closeReader();
      if (sock.tx !== void 0) new PipeRing(shmArena, sock.tx).closeWriter();
    }
  }
  var pendingClone = null;
  function handleCloneThread(cpu2, svcPC) {
    const flags = Number(cpu2.getReg(0));
    const childStack = Number(cpu2.getReg(1));
    const parentTidPtr = Number(cpu2.getReg(2));
    const tls = Number(cpu2.getReg(3));
    const childTidPtr = Number(cpu2.getReg(4));
    if (!threaded) {
      if (Number(syscalls.getMmapBase()) > threadLayout.regsBase) {
        cpu2.setReg(0, BigInt.asUintN(64, -11n));
        return false;
      }
      pendingClone = { svcPC, childStack, tls, parentTidPtr, childTidPtr, flags };
      self.postMessage({ type: "clone_thread", curPages: memory.buffer.byteLength / MEM_PAGE_SIZE });
      return true;
    }
    Atomics.store(myMailbox, MBOX_READY, 0);
    const threadSocks = refUnixSocksForChild();
    self.postMessage({
      type: "spawn_thread",
      tgid: syscalls.tgid || myPid,
      startPC: (svcPC + 4n).toString(),
      childStack,
      tls,
      childTidPtr,
      flags,
      parentRegBase: stateBase,
      syscallState: syscalls.getState(),
      unixSocksSnapshot: threadSocks,
      shmMaps: sharedMaps.map((m) => ({
        addr: m.addr,
        len: m.len,
        off: m.arenaOff,
        sysv: sysvAttached.get(m.addr)
      }))
    });
    awaitMailbox("clone_thread");
    const tid = Atomics.load(myMailbox, MBOX_VALUE0);
    if (tid < 0) {
      releaseUnixSocksSnapshot(threadSocks);
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-11)));
      return false;
    }
    if (flags & CLONE_PARENT_SETTID && parentTidPtr) new DataView(memory.buffer).setInt32(parentTidPtr + GUEST_MEM_BASE, tid, true);
    cpu2.setReg(0, BigInt(tid));
    return false;
  }
  function completePromote(sharedMemory, tid, regBaseAddr, staleHigh, layout) {
    if (layout) threadLayout = layout;
    const pc = pendingClone;
    const copyLen = Math.min(memory.buffer.byteLength, threadLayout.regsBase);
    const shared = new Uint8Array(sharedMemory.buffer);
    shared.set(new Uint8Array(memory.buffer, 0, copyLen));
    if (staleHigh > copyLen) shared.fill(0, copyLen, staleHigh);
    if (staleHigh > 0) shared.fill(0, threadLayout.shadowBase, threadLayout.shadowBase + (threadLayout.regsBase >> 9));
    memory = sharedMemory;
    threaded = true;
    syscalls.attachVmaTable(memory, threadLayout.vmaBase);
    syscalls.publishVmaTable();
    syscalls.attachSigTable(memory, threadLayout.sigBase);
    syscalls.publishSigTable();
    if (protActive && protBase) {
      if (!protReady) {
        paintProtTable();
        protReady = true;
      }
      new Int32Array(memory.buffer, PROT_PAINTED_FLAG, 1)[0] = 1;
      protActive.value = 1;
    }
    stateBase = 0;
    regBaseGlobal = new WebAssembly.Global({ value: "i32", mutable: false }, 0);
    cpu = new CpuState(memory, stateBase);
    regionCache.clear();
    syscalls.setThreadIds(myPid, myPid);
    syscalls.maxBytes = threadLayout.regsBase;
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
      parentRegBase: regBaseAddr,
      // block already holds the clone-time snapshot
      syscallState: syscalls.getState(),
      // thread inherits the group's fds/cwd/creds
      unixSocksSnapshot: refUnixSocksForChild(),
      shmMaps: sharedMaps.map((m) => ({
        addr: m.addr,
        len: m.len,
        off: m.arenaOff,
        sysv: sysvAttached.get(m.addr)
      }))
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
  function sharedFutexSlot(uaddr) {
    for (const m of sharedMaps) {
      if (uaddr < m.addr || uaddr + 4 > m.addr + m.len) continue;
      if (!syscalls.rangeIsShared(uaddr & ~4095, (uaddr & ~4095) + 4096)) continue;
      return m.arenaOff + (uaddr - m.addr) >> 2;
    }
    return -1;
  }
  var FUTEX_WAIT = 0;
  var FUTEX_WAKE = 1;
  var FUTEX_REQUEUE = 3;
  var FUTEX_CMP_REQUEUE = 4;
  var FUTEX_WAKE_OP = 5;
  var FUTEX_WAIT_BITSET = 9;
  var FUTEX_WAKE_BITSET = 10;
  var FUTEX_CLOCK_REALTIME = 256;
  var FX_EINTR = 4;
  var FX_EFAULT = 14;
  var FX_EINVAL = 22;
  var FX_EAGAIN = 11;
  var FX_ENOMEM = 12;
  var FX_ENOSYS = 38;
  var FX_ETIMEDOUT = 110;
  function futexRef(uaddr) {
    const shared = sharedFutexSlot(uaddr);
    if (shared >= 0) return { i32: new Int32Array(shmArena), idx: shared, keyLo: shared, keyHi: 0 };
    return {
      i32: new Int32Array(memory.buffer),
      idx: uaddr + GUEST_MEM_BASE >> 2,
      keyLo: uaddr | 0,
      keyHi: (syscalls.tgid || myPid) | 0
    };
  }
  function futexDeadline(view, tsPtr, absolute, clk) {
    if (tsPtr === 0) return Infinity;
    if (syscalls.ptrFaults(new Uint8Array(memory.buffer), tsPtr, 16)) return "fault";
    const sec = view.getBigInt64(tsPtr + GUEST_MEM_BASE, true);
    const nsec = view.getBigInt64(tsPtr + GUEST_MEM_BASE + 8, true);
    if (nsec < 0n || nsec >= 1000000000n) return "invalid";
    const want = sec * 1000000000n + nsec;
    if (!absolute) {
      if (want < 0n) return "invalid";
      return performance.now() + Number(want) / 1e6;
    }
    const left = want - syscalls.clockNs(clk);
    return performance.now() + Math.max(0, Number(left) / 1e6);
  }
  function futexPark(slot, deadline) {
    const stateIdx = futexStateIndex(slot);
    let ret = 0, signalled = false;
    for (; ; ) {
      if (Atomics.load(procTable, stateIdx) === FUTEX_WOKEN) break;
      const left = deadline - performance.now();
      if (left <= 0) {
        ret = -FX_ETIMEDOUT;
        break;
      }
      sleepingWait(procTable, stateIdx, FUTEX_QUEUED, Math.min(left, 50));
      if (Atomics.load(procTable, stateIdx) === FUTEX_WOKEN) break;
      if (pollAndQueueSignals()) {
        ret = -FX_EINTR;
        signalled = true;
        break;
      }
    }
    futexLock(procTable);
    if (Atomics.load(procTable, stateIdx) === FUTEX_WOKEN) ret = 0;
    const woken = Atomics.load(procTable, FUTEX_TABLE_OFFSET / 4 + FUTEX_HDR_WORDS + slot * FUTEX_SLOT_WORDS + FUTEX_RESULT);
    futexDequeue(procTable, slot);
    futexUnlock(procTable);
    return { ret: ret === 0 ? Math.max(0, woken) : ret, signalled };
  }
  function handleFutex(cpu2) {
    const bytes = new Uint8Array(memory.buffer);
    const view = new DataView(memory.buffer);
    const uaddr = Number(BigInt.asUintN(32, cpu2.getReg(0)));
    const rawOp = Number(BigInt.asIntN(32, cpu2.getReg(1)));
    const op = rawOp & 127;
    const realtime = (rawOp & FUTEX_CLOCK_REALTIME) !== 0;
    const val = Number(BigInt.asIntN(32, cpu2.getReg(2)));
    const arg3 = cpu2.getReg(3);
    const uaddr2 = Number(BigInt.asUintN(32, cpu2.getReg(4)));
    const val3 = Number(BigInt.asIntN(32, cpu2.getReg(5)));
    const err = (e) => {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(e)));
    };
    if (uaddr & 3) {
      err(-FX_EINVAL);
      return false;
    }
    if (op === FUTEX_WAIT || op === FUTEX_WAIT_BITSET) {
      const bits2 = op === FUTEX_WAIT_BITSET ? val3 >>> 0 : 4294967295;
      if (bits2 === 0) {
        err(-FX_EINVAL);
        return false;
      }
      if (syscalls.ptrFaults(bytes, uaddr, 4)) {
        err(-FX_EFAULT);
        return false;
      }
      const dl = futexDeadline(
        view,
        Number(BigInt.asUintN(32, arg3)),
        op === FUTEX_WAIT_BITSET,
        realtime ? 0 : 1
      );
      if (dl === "fault") {
        err(-FX_EFAULT);
        return false;
      }
      if (dl === "invalid") {
        err(-FX_EINVAL);
        return false;
      }
      const t = futexRef(uaddr);
      futexLock(procTable);
      if (Atomics.load(t.i32, t.idx) !== (val | 0)) {
        futexUnlock(procTable);
        err(-FX_EAGAIN);
        return false;
      }
      const slot = futexEnqueue(procTable, t.keyLo, t.keyHi, bits2, myPid, -1);
      futexUnlock(procTable);
      if (slot < 0) {
        err(-FX_ENOMEM);
        return false;
      }
      const r = futexPark(slot, dl);
      err(r.ret > 0 ? 0 : r.ret);
      return r.signalled;
    }
    if (op === FUTEX_WAKE || op === FUTEX_WAKE_BITSET) {
      const bits2 = op === FUTEX_WAKE_BITSET ? val3 >>> 0 : 4294967295;
      if (bits2 === 0) {
        err(-FX_EINVAL);
        return false;
      }
      if (syscalls.ptrFaults(bytes, uaddr, 4)) {
        err(-FX_EFAULT);
        return false;
      }
      const t = futexRef(uaddr);
      futexLock(procTable);
      const r = futexWake(procTable, t.keyLo, t.keyHi, bits2, Math.max(0, val));
      futexUnlock(procTable);
      cpu2.setReg(0, BigInt(r.woke));
      return false;
    }
    if (op === FUTEX_REQUEUE || op === FUTEX_CMP_REQUEUE) {
      if (uaddr2 & 3) {
        err(-FX_EINVAL);
        return false;
      }
      if (syscalls.ptrFaults(bytes, uaddr, 4) || syscalls.ptrFaults(bytes, uaddr2, 4)) {
        err(-FX_EFAULT);
        return false;
      }
      const nrRequeue = Number(BigInt.asIntN(32, arg3));
      const t = futexRef(uaddr), t2 = futexRef(uaddr2);
      futexLock(procTable);
      if (op === FUTEX_CMP_REQUEUE && Atomics.load(t.i32, t.idx) !== (val3 | 0)) {
        futexUnlock(procTable);
        err(-FX_EAGAIN);
        return false;
      }
      const r = futexWake(
        procTable,
        t.keyLo,
        t.keyHi,
        4294967295,
        Math.max(0, val),
        Math.max(0, nrRequeue),
        t2.keyLo,
        t2.keyHi
      );
      futexUnlock(procTable);
      cpu2.setReg(0, BigInt(r.woke + r.moved));
      return false;
    }
    if (op === FUTEX_WAKE_OP) {
      if (uaddr2 & 3) {
        err(-FX_EINVAL);
        return false;
      }
      if (syscalls.ptrFaults(bytes, uaddr, 4) || syscalls.ptrFaults(bytes, uaddr2, 4)) {
        err(-FX_EFAULT);
        return false;
      }
      const enc = val3 >>> 0;
      const opcode = enc >> 28 & 7;
      const shifted = (enc >> 28 & 8) !== 0;
      const cmp = enc >> 24 & 15;
      let oparg = enc << 8 >> 20;
      const cmparg = enc << 20 >> 20;
      if (shifted) {
        if (oparg < 0 || oparg > 31) {
          err(-FX_EINVAL);
          return false;
        }
        oparg = 1 << oparg;
      }
      const t = futexRef(uaddr), t2 = futexRef(uaddr2);
      let old;
      switch (opcode) {
        case 0:
          old = Atomics.exchange(t2.i32, t2.idx, oparg);
          break;
        case 1:
          old = Atomics.add(t2.i32, t2.idx, oparg);
          break;
        case 2:
          old = Atomics.or(t2.i32, t2.idx, oparg);
          break;
        case 3:
          old = Atomics.and(t2.i32, t2.idx, ~oparg);
          break;
        case 4:
          old = Atomics.xor(t2.i32, t2.idx, oparg);
          break;
        default:
          err(-FX_ENOSYS);
          return false;
      }
      let second;
      switch (cmp) {
        case 0:
          second = old === cmparg;
          break;
        case 1:
          second = old !== cmparg;
          break;
        case 2:
          second = old < cmparg;
          break;
        case 3:
          second = old <= cmparg;
          break;
        case 4:
          second = old > cmparg;
          break;
        case 5:
          second = old >= cmparg;
          break;
        default:
          err(-FX_ENOSYS);
          return false;
      }
      const nrWake2 = Number(BigInt.asIntN(32, arg3));
      futexLock(procTable);
      let total = futexWake(procTable, t.keyLo, t.keyHi, 4294967295, Math.max(0, val)).woke;
      if (second) total += futexWake(procTable, t2.keyLo, t2.keyHi, 4294967295, Math.max(0, nrWake2)).woke;
      futexUnlock(procTable);
      cpu2.setReg(0, BigInt(total));
      return false;
    }
    err(-FX_ENOSYS);
    return false;
  }
  function handleFutexWaitv(cpu2) {
    const bytes = new Uint8Array(memory.buffer);
    const view = new DataView(memory.buffer);
    const listPtr = Number(BigInt.asUintN(32, cpu2.getReg(0)));
    const nr = Number(BigInt.asUintN(32, cpu2.getReg(1)));
    const flags = Number(BigInt.asUintN(32, cpu2.getReg(2)));
    const tsPtr = Number(BigInt.asUintN(32, cpu2.getReg(3)));
    const clockid = Number(BigInt.asIntN(32, cpu2.getReg(4)));
    const err = (e) => {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(e)));
    };
    const FUTEX2_SIZE_MASK = 3, FUTEX2_32 = 2, FUTEX2_KNOWN = 131;
    if (flags !== 0) {
      err(-FX_EINVAL);
      return false;
    }
    if (nr === 0 || nr > 128) {
      err(-FX_EINVAL);
      return false;
    }
    if (listPtr === 0) {
      err(-FX_EINVAL);
      return false;
    }
    if (clockid !== 0 && clockid !== 1) {
      err(-FX_EINVAL);
      return false;
    }
    if (syscalls.ptrFaults(bytes, listPtr, nr * 24)) {
      err(-FX_EFAULT);
      return false;
    }
    const want = [];
    for (let i = 0; i < nr; i++) {
      const rec = listPtr + GUEST_MEM_BASE + i * 24;
      const wval = view.getBigUint64(rec, true);
      const waddr = Number(BigInt.asUintN(32, view.getBigUint64(rec + 8, true)));
      const wflags = view.getUint32(rec + 16, true);
      const reserved = view.getUint32(rec + 20, true);
      if (reserved !== 0) {
        err(-FX_EINVAL);
        return false;
      }
      if (wflags & ~FUTEX2_KNOWN) {
        err(-FX_EINVAL);
        return false;
      }
      if ((wflags & FUTEX2_SIZE_MASK) !== FUTEX2_32) {
        err(-FX_EINVAL);
        return false;
      }
      if (waddr & 3) {
        err(-FX_EINVAL);
        return false;
      }
      if (syscalls.ptrFaults(bytes, waddr, 4)) {
        err(-FX_EFAULT);
        return false;
      }
      want.push({ uaddr: waddr, val: Number(BigInt.asIntN(32, BigInt.asUintN(32, wval))) });
    }
    const dl = futexDeadline(view, tsPtr, true, clockid);
    if (dl === "fault") {
      err(-FX_EFAULT);
      return false;
    }
    if (dl === "invalid") {
      err(-FX_EINVAL);
      return false;
    }
    futexLock(procTable);
    const slots = [];
    let leader = -1, bad = 0;
    for (let i = 0; i < want.length; i++) {
      const t = futexRef(want[i].uaddr);
      if (Atomics.load(t.i32, t.idx) !== (want[i].val | 0)) {
        bad = -FX_EAGAIN;
        break;
      }
      const slot = futexEnqueue(procTable, t.keyLo, t.keyHi, 4294967295, myPid, leader, true);
      if (slot < 0) {
        bad = -FX_ENOMEM;
        break;
      }
      if (leader < 0) leader = slot;
      slots.push(slot);
    }
    if (bad !== 0) {
      if (leader >= 0) futexDequeue(procTable, leader);
      futexUnlock(procTable);
      err(bad);
      return false;
    }
    futexUnlock(procTable);
    const r = futexPark(leader, dl);
    if (r.ret < 0) {
      err(r.ret);
      return r.signalled;
    }
    const woken = slots.indexOf(r.ret);
    cpu2.setReg(0, BigInt(woken < 0 ? 0 : woken));
    return r.signalled;
  }
  function handleExec(path, argv, envp) {
    const execAbs = syscalls.absolutePath(path);
    if (execAbs !== null) {
      let execAcc = 0;
      handleBridgeCall(
        "fs_faccessat",
        { path: execAbs, mode: 1, eaccess: true, creds: syscalls.getCreds() },
        (resp) => {
          execAcc = resp?.ret ?? 0;
        }
      );
      if (execAcc < 0) throw { errno: -execAcc };
    }
    const target = resolveExec(path, argv, (p) => {
      try {
        return readGuestFile(p);
      } catch {
        return null;
      }
    });
    path = target.path;
    argv = target.argv;
    const bytes = target.bytes;
    elfBytes = bytes;
    new Uint8Array(memory.buffer).fill(0, GUEST_MEM_BASE, GUEST_MEM_BASE + 67108864);
    resetPageProt();
    const prog = loadProgram(memory, elfBytes, argv, envp, readGuestFile);
    cpu = new CpuState(memory, stateBase);
    cpu.setSP(prog.sp);
    new DataView(memory.buffer).setBigUint64(stateBase + 280, BigInt(9437184), true);
    releaseSharedMaps(0, Number.MAX_SAFE_INTEGER);
    for (const fd of syscalls.cloexecFds()) {
      const info = kernelPipes.get(fd);
      if (info) {
        pipeRef(info.pipeOff, info.mode, -1);
        kernelPipes.delete(fd);
      }
      if (unixSocks.has(fd)) closeUnixSock(fd);
    }
    publishIdentity(argv);
    {
      const img = syscalls.absolutePath(path);
      if (img) self.postMessage({ type: "exec_image", path: img });
    }
    syscalls.setComm((argv[0] ?? "").split("/").pop() ?? "");
    syscalls.resetForExec(prog.brkStart, (fd) => handleBridgeCall("fs_close", { fd }, () => {
    }));
    syscalls.adoptImage(prog.segs);
    regionCache.clear();
    syscalls.installSignalTrampoline(memory);
    releaseVforkParent();
    if (Atomics.load(procTable, procSlotOffset(mySlot) / 4 + PROC_TRACER) !== 0) syscalls.raiseSignal(5);
    return prog.entry;
  }
  function handleNanosleep(cpu2, view, reqAddr, remAddr, absolute, clockId) {
    const bytes = new Uint8Array(memory.buffer);
    if (reqAddr === 0 || syscalls.ptrFaults(bytes, reqAddr, 16)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
      return;
    }
    if (clockId < 0 || clockId > 11 || clockId === 3) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    const sec = Number(view.getBigInt64(reqAddr + GUEST_MEM_BASE, true));
    const nsec = Number(view.getBigInt64(reqAddr + GUEST_MEM_BASE + 8, true));
    if (sec < 0 || nsec < 0 || nsec >= 1e9) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    if (remAddr !== 0 && !absolute && syscalls.ptrWriteFaults(bytes, remAddr, 16)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
      return;
    }
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
      if (rem <= 1.2) {
        while (performance.now() < deadline) {
          if (pollAndQueueSignals()) {
            interrupted = true;
            break;
          }
        }
        break;
      }
      Atomics.store(procTable, slotBase + PROC_WAIT_FUTEX, 0);
      sleepingWait(procTable, slotBase + PROC_WAIT_FUTEX, 0, Math.max(1.2, Math.min(rem * 0.7, 1e3)));
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
  function handleSigsuspend(cpu2) {
    const maskPtr = Number(cpu2.getReg(0));
    if (maskPtr === 0 || syscalls.ptrFaults(new Uint8Array(memory.buffer), maskPtr, 8)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
      return;
    }
    const tempMask = new DataView(memory.buffer).getBigUint64(maskPtr + GUEST_MEM_BASE, true);
    const savedMask = syscalls.getSignalMask();
    syscalls.setSuspendMask(tempMask, savedMask);
    const slotBase = procSlotOffset(mySlot) / 4;
    for (; ; ) {
      pollAndQueueSignals();
      if (syscalls.hasDeliverableSignal(tempMask)) break;
      Atomics.store(procTable, slotBase + PROC_WAIT_FUTEX, 0);
      const rem = syscalls.timerRemainingMs();
      sleepingWait(procTable, slotBase + PROC_WAIT_FUTEX, 0, rem > 0 ? Math.min(1e3, rem) : 1e3);
    }
    cpu2.setReg(0, BigInt.asUintN(64, -4n));
  }
  function handlePpollWait(cpu2) {
    const view = new DataView(memory.buffer);
    const tmoPtr = Number(cpu2.getReg(2));
    const sigmaskPtr = Number(cpu2.getReg(3));
    let deadline = -1;
    if (tmoPtr !== 0) {
      const sec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE, true));
      const nsec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE + 8, true));
      deadline = performance.now() + sec * 1e3 + nsec / 1e6;
    }
    const savedMask = syscalls.getSignalMask();
    const mask = sigmaskPtr ? view.getBigUint64(sigmaskPtr + GUEST_MEM_BASE, true) : savedMask;
    if (sigmaskPtr) syscalls.setSignalMask(mask);
    const slotBase = procSlotOffset(mySlot) / 4;
    let ret = 0n;
    for (; ; ) {
      pollAndQueueSignals();
      if (syscalls.hasDeliverableSignal(mask)) {
        ret = BigInt.asUintN(64, -4n);
        break;
      }
      if (deadline >= 0 && performance.now() >= deadline) {
        ret = 0n;
        break;
      }
      Atomics.store(procTable, slotBase + PROC_WAIT_FUTEX, 0);
      const timerRem = syscalls.timerRemainingMs();
      let wait = deadline >= 0 ? Math.max(0, deadline - performance.now()) : timerRem > 0 ? timerRem : 1e3;
      wait = Math.min(1e3, wait || 1e3);
      sleepingWait(procTable, slotBase + PROC_WAIT_FUTEX, 0, wait);
    }
    if (sigmaskPtr) syscalls.setSignalMask(savedMask);
    cpu2.setReg(0, ret);
  }
  function handleWaitid(cpu2, view) {
    const err = (e) => cpu2.setReg(0, BigInt.asUintN(64, BigInt(-e)));
    const P_ALL = 0, P_PID = 1, P_PGID = 2, P_PIDFD = 3;
    const WNOHANG = 1, WSTOPPED = 2, WEXITED = 4, WCONTINUED = 8;
    const WNOWAIT = 16777216, WNOTHREAD = 536870912, WALL = 1073741824, WCLONE = 2147483648;
    const KNOWN = WNOHANG | WSTOPPED | WEXITED | WCONTINUED | WNOWAIT | WNOTHREAD | WALL | WCLONE;
    let idtype = Number(cpu2.getReg(0));
    let id = Number(BigInt.asIntN(32, cpu2.getReg(1)));
    const infop = Number(cpu2.getReg(2));
    const options = Number(BigInt.asUintN(32, cpu2.getReg(3)));
    let pidfdNonblock = false;
    if (idtype === P_PIDFD) {
      const pid = syscalls.pidfdTarget(id);
      if (pid < 0) {
        err(9);
        return;
      }
      pidfdNonblock = syscalls.isNonBlockingFd(id);
      idtype = P_PID;
      id = pid;
    }
    if (options & ~KNOWN) {
      err(22);
      return;
    }
    if ((options & (WEXITED | WSTOPPED | WCONTINUED)) === 0) {
      err(22);
      return;
    }
    if (idtype === P_PID && id <= 0) {
      err(22);
      return;
    }
    if (idtype === P_PGID && id < 0) {
      err(22);
      return;
    }
    if (idtype !== P_ALL && idtype !== P_PID && idtype !== P_PGID) {
      err(22);
      return;
    }
    if (infop !== 0 && !guestAccessible(infop, 128, true)) {
      err(14);
      return;
    }
    const bytes = new Uint8Array(memory.buffer);
    const zeroInfo = () => {
      if (infop !== 0) bytes.fill(0, infop + GUEST_MEM_BASE, infop + GUEST_MEM_BASE + 128);
    };
    for (; ; ) {
      let zombie = -1;
      let stopped = -1;
      let cont = -1;
      let living = false;
      for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
        const base = procSlotOffset(slot) / 4;
        const childPid = Atomics.load(procTable, base + PROC_PID);
        if (childPid === 0) continue;
        if (Atomics.load(procTable, base + PROC_PPID) !== myPid && Atomics.load(procTable, base + PROC_TRACER) !== myPid) continue;
        const baseTgid = Atomics.load(procTable, base + PROC_TGID);
        if (baseTgid !== 0 && baseTgid !== childPid) continue;
        if (idtype === P_PID && childPid !== id) continue;
        if (idtype === P_PGID) {
          const want = id === 0 ? Atomics.load(procTable, procSlotOffset(mySlot) / 4 + PROC_PGID) : id;
          if (Atomics.load(procTable, base + PROC_PGID) !== want) continue;
        }
        const state = Atomics.load(procTable, base + PROC_STATE);
        if (state === PROC_ZOMBIE) {
          if (zombie < 0) zombie = slot;
        } else if (state === PROC_RUNNING || state === PROC_STOPPED) {
          living = true;
          if (stopped < 0 && ((options & WSTOPPED) !== 0 || Atomics.load(procTable, base + PROC_TRACER) === myPid) && state === PROC_STOPPED && Atomics.load(procTable, base + PROC_STOP_SEEN) === 0) stopped = slot;
          if (cont < 0 && options & WCONTINUED && Atomics.load(procTable, base + PROC_CONT_PENDING) === 1) cont = slot;
        }
      }
      if (zombie < 0 && (stopped >= 0 || cont >= 0)) {
        const isStop = stopped >= 0;
        const base = procSlotOffset(isStop ? stopped : cont) / 4;
        const childPid = Atomics.load(procTable, base + PROC_PID);
        const uid = Atomics.load(procCreds, procCredOffset(isStop ? stopped : cont) / 4);
        const CLD_STOPPED = 5, CLD_CONTINUED = 6;
        const status = isStop ? Atomics.load(procTable, base + PROC_STOP_SIG) : 18;
        if (infop !== 0) {
          zeroInfo();
          const b = infop + GUEST_MEM_BASE;
          view.setInt32(b, 17, true);
          view.setInt32(b + 8, isStop ? CLD_STOPPED : CLD_CONTINUED, true);
          view.setInt32(b + 16, childPid, true);
          view.setUint32(b + 20, uid, true);
          view.setInt32(b + 24, status, true);
        }
        if (!(options & WNOWAIT)) {
          if (isStop) Atomics.store(procTable, base + PROC_STOP_SEEN, 1);
          else Atomics.store(procTable, base + PROC_CONT_PENDING, 0);
        }
        cpu2.setReg(0, 0n);
        return;
      }
      if (zombie >= 0 && options & WEXITED) {
        const base = procSlotOffset(zombie) / 4;
        const childPid = Atomics.load(procTable, base + PROC_PID);
        const exitCode = Atomics.load(procTable, base + PROC_EXIT_CODE);
        const uid = Atomics.load(procCreds, procCredOffset(zombie) / 4);
        const killed = (exitCode & 65536) !== 0;
        const CLD_EXITED = 1, CLD_KILLED = 2, CLD_DUMPED = 3;
        const code = killed ? exitCode & 131072 ? CLD_DUMPED : CLD_KILLED : CLD_EXITED;
        const status = killed ? exitCode & 127 : exitCode & 255;
        const cUtime = Atomics.load(procTable, base + PROC_UTIME_MS);
        const cStime = Atomics.load(procTable, base + PROC_STIME_MS);
        if (infop !== 0) {
          zeroInfo();
          const b = infop + GUEST_MEM_BASE;
          view.setInt32(b, 17, true);
          view.setInt32(b + 8, code, true);
          view.setInt32(b + 16, childPid, true);
          view.setUint32(b + 20, uid, true);
          view.setInt32(b + 24, status, true);
          view.setBigInt64(b + 32, BigInt(Math.floor(cUtime / 10)), true);
          view.setBigInt64(b + 40, BigInt(Math.floor(cStime / 10)), true);
        }
        const rusageAddr = Number(cpu2.getReg(4));
        if (!(options & WNOWAIT)) {
          syscalls.noteChildCpu(cUtime, cStime);
          if (rusageAddr !== 0) syscalls.writeRusage(view, rusageAddr, cUtime, cStime, 0);
          Atomics.store(procTable, base + PROC_STATE, PROC_EMPTY);
          Atomics.store(procTable, base + PROC_PID, 0);
        }
        cpu2.setReg(0, 0n);
        return;
      }
      if (!living && zombie < 0) {
        err(10);
        return;
      }
      if (options & WNOHANG) {
        zeroInfo();
        cpu2.setReg(0, 0n);
        return;
      }
      if (pidfdNonblock) {
        err(11);
        return;
      }
      const mySlotBase = procSlotOffset(mySlot) / 4;
      const rem = syscalls.timerRemainingMs();
      Atomics.store(procTable, mySlotBase + PROC_WAIT_FUTEX, 0);
      sleepingWait(procTable, mySlotBase + PROC_WAIT_FUTEX, 0, rem > 0 ? Math.min(1e3, rem) : 1e3);
      if (pollAndQueueSignals()) {
        err(4);
        return;
      }
    }
  }
  function handleWait4(cpu2, view) {
    const targetPid = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    const wstatusAddr = Number(cpu2.getReg(1));
    const options = Number(BigInt.asUintN(32, cpu2.getReg(2)));
    const rusageAddr = Number(cpu2.getReg(3));
    const WNOHANG = 1, WUNTRACED = 2, WCONTINUED = 8;
    const WNOTHREAD = 536870912, WALL = 1073741824, WCLONE = 2147483648;
    if (options & ~(WNOHANG | WUNTRACED | WCONTINUED | WNOTHREAD | WALL | WCLONE)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    let wantPgid = -1;
    if (targetPid === 0) {
      wantPgid = Atomics.load(procTable, procSlotOffset(mySlot) / 4 + PROC_PGID);
    } else if (targetPid < -1) {
      wantPgid = -targetPid;
      let idExists = wantPgid > 0;
      if (idExists) {
        idExists = false;
        for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
          const base = procSlotOffset(slot) / 4;
          if (Atomics.load(procTable, base + PROC_PID) === wantPgid) {
            idExists = true;
            break;
          }
        }
      }
      if (!idExists) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-3)));
        return;
      }
    }
    while (true) {
      let foundZombie = -1;
      let foundStopped = -1;
      let foundCont = -1;
      let hasLivingChild = false;
      for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
        const slotBase = procSlotOffset(slot) / 4;
        const childPid = Atomics.load(procTable, slotBase + PROC_PID);
        if (childPid === 0) continue;
        const ppid = Atomics.load(procTable, slotBase + PROC_PPID);
        if (ppid !== myPid && Atomics.load(procTable, slotBase + PROC_TRACER) !== myPid) continue;
        const slotTgid = Atomics.load(procTable, slotBase + PROC_TGID);
        if (slotTgid !== 0 && slotTgid !== childPid) continue;
        if (targetPid > 0 && childPid !== targetPid) continue;
        if (wantPgid >= 0 && Atomics.load(procTable, slotBase + PROC_PGID) !== wantPgid) continue;
        const state = Atomics.load(procTable, slotBase + PROC_STATE);
        if (state === PROC_ZOMBIE) {
          foundZombie = slot;
          break;
        }
        if (state === PROC_RUNNING || state === PROC_STOPPED) {
          hasLivingChild = true;
          if (foundStopped < 0 && state === PROC_STOPPED && ((options & WUNTRACED) !== 0 || Atomics.load(procTable, slotBase + PROC_TRACER) === myPid) && Atomics.load(procTable, slotBase + PROC_STOP_SEEN) === 0) foundStopped = slot;
          if (foundCont < 0 && options & WCONTINUED && Atomics.load(procTable, slotBase + PROC_CONT_PENDING) === 1) foundCont = slot;
        }
      }
      if (foundZombie < 0 && (foundStopped >= 0 || foundCont >= 0)) {
        const stopped = foundStopped >= 0;
        const slotBase = procSlotOffset(stopped ? foundStopped : foundCont) / 4;
        const childPid = Atomics.load(procTable, slotBase + PROC_PID);
        if (stopped) Atomics.store(procTable, slotBase + PROC_STOP_SEEN, 1);
        else Atomics.store(procTable, slotBase + PROC_CONT_PENDING, 0);
        if (wstatusAddr !== 0) {
          const stopSig = Atomics.load(procTable, slotBase + PROC_STOP_SIG);
          view.setUint32(wstatusAddr + GUEST_MEM_BASE, stopped ? stopSig << 8 | 127 : 65535, true);
        }
        cpu2.setReg(0, BigInt(childPid));
        return;
      }
      if (foundZombie >= 0) {
        const slotBase = procSlotOffset(foundZombie) / 4;
        const childPid = Atomics.load(procTable, slotBase + PROC_PID);
        const exitCode = Atomics.load(procTable, slotBase + PROC_EXIT_CODE);
        if (wstatusAddr !== 0) {
          const wstatus = exitCode & 65536 ? exitCode & 127 | (exitCode & 131072 ? 128 : 0) : (exitCode & 255) << 8;
          view.setUint32(wstatusAddr + GUEST_MEM_BASE, wstatus, true);
        }
        const cUtime = Atomics.load(procTable, slotBase + PROC_UTIME_MS);
        const cStime = Atomics.load(procTable, slotBase + PROC_STIME_MS);
        const cMaxRss = Atomics.load(procTable, slotBase + PROC_MAXRSS_KB);
        syscalls.noteChildCpu(cUtime, cStime, cMaxRss);
        if (rusageAddr !== 0) syscalls.writeRusage(view, rusageAddr, cUtime, cStime, cMaxRss);
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
      sleepingWait(procTable, mySlotBase + PROC_WAIT_FUTEX, 0, waitMs);
      if (pollAndQueueSignals()) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-4)));
        return;
      }
    }
  }
  function sigpendingAllows(slot) {
    const limit = Atomics.load(procTable, procSlotOffset(slot) / 4 + PROC_RLIM_SIGPENDING);
    if (limit <= 0) return false;
    const ruid = Atomics.load(procCreds, procCredOffset(slot) / 4);
    let pending = 0;
    for (let s = 0; s < PROC_TABLE_SLOTS; s++) {
      if (Atomics.load(procTable, procSlotOffset(s) / 4 + PROC_PID) === 0) continue;
      if (Atomics.load(procCreds, procCredOffset(s) / 4) !== ruid) continue;
      pending += sigqDepth(sigQueue, s);
      if (pending >= limit) return false;
    }
    return true;
  }
  function handleCpuClock(nr, cpu2, view) {
    const err = (e) => cpu2.setReg(0, BigInt.asUintN(64, BigInt(-e)));
    const clk = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    const ptr = Number(cpu2.getReg(1));
    const id = ~(clk >> 3);
    const perThread = (clk & 4) !== 0;
    const which = clk & 3;
    if (which === 3 || id <= 0) {
      err(22);
      return;
    }
    if (ptr !== 0 && syscalls.ptrFaults(new Uint8Array(memory.buffer), ptr, 16)) {
      err(14);
      return;
    }
    if (nr === 114) {
      if (ptr !== 0) {
        view.setBigInt64(ptr + GUEST_MEM_BASE, 0n, true);
        view.setBigInt64(ptr + GUEST_MEM_BASE + 8, 1000000n, true);
      }
      cpu2.setReg(0, 0n);
      return;
    }
    publishCpuTime();
    const readMs = (slot) => {
      const b = procSlotOffset(slot) / 4;
      const u = Atomics.load(procTable, b + PROC_UTIME_MS);
      const sy = Atomics.load(procTable, b + PROC_STIME_MS);
      return which === 1 ? u : u + sy;
    };
    let ms = 0;
    if (perThread) {
      const slot = schedSlotOf(id);
      if (slot < 0) {
        err(22);
        return;
      }
      ms = readMs(slot);
    } else {
      let found = false;
      for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
        const b = procSlotOffset(slot) / 4;
        const pid = Atomics.load(procTable, b + PROC_PID);
        if (pid === 0 || Atomics.load(procTable, b + PROC_STATE) === PROC_EMPTY) continue;
        if ((Atomics.load(procTable, b + PROC_TGID) || pid) !== id) continue;
        found = true;
        ms += readMs(slot);
      }
      if (!found) {
        err(22);
        return;
      }
    }
    if (ptr !== 0) {
      view.setBigInt64(ptr + GUEST_MEM_BASE, BigInt(Math.floor(ms / 1e3)), true);
      view.setBigInt64(ptr + GUEST_MEM_BASE + 8, BigInt(ms % 1e3 * 1e6), true);
    }
    cpu2.setReg(0, 0n);
  }
  function pidExited(pid) {
    const slot = schedSlotOf(pid);
    if (slot < 0) return true;
    return Atomics.load(procTable, procSlotOffset(slot) / 4 + PROC_STATE) === PROC_ZOMBIE;
  }
  function handlePidfd(nr, cpu2, view) {
    const err = (e) => cpu2.setReg(0, BigInt.asUintN(64, BigInt(-e)));
    const PIDFD_NONBLOCK = 2048;
    if (nr === 434) {
      const pid = Number(BigInt.asIntN(32, cpu2.getReg(0)));
      const flags2 = Number(BigInt.asUintN(32, cpu2.getReg(1)));
      if (pid <= 0) {
        err(22);
        return;
      }
      if (flags2 & ~PIDFD_NONBLOCK) {
        err(22);
        return;
      }
      const slot = schedSlotOf(pid);
      if (slot < 0) {
        err(3);
        return;
      }
      const base = procSlotOffset(slot) / 4;
      if ((Atomics.load(procTable, base + PROC_TGID) || pid) !== pid) {
        err(22);
        return;
      }
      const fd = syscalls.allocLowestFd();
      if (fd < 0) {
        err(24);
        return;
      }
      syscalls.registerFd(fd, { type: "pidfd", pid }, flags2 & PIDFD_NONBLOCK);
      syscalls.markCloexec(fd);
      cpu2.setReg(0, BigInt(fd));
      return;
    }
    const target = syscalls.pidfdTarget(Number(BigInt.asIntN(32, cpu2.getReg(0))));
    const sig = Number(BigInt.asIntN(32, cpu2.getReg(1)));
    const infoPtr = Number(cpu2.getReg(2));
    const flags = Number(BigInt.asUintN(32, cpu2.getReg(3)));
    if (flags !== 0) {
      err(22);
      return;
    }
    if (target < 0) {
      err(9);
      return;
    }
    if (sig < 0 || sig > 64) {
      err(22);
      return;
    }
    if (pidExited(target)) {
      err(3);
      return;
    }
    if (infoPtr !== 0) {
      if (syscalls.ptrFaults(new Uint8Array(memory.buffer), infoPtr, 128)) {
        err(14);
        return;
      }
      if (view.getInt32(infoPtr + GUEST_MEM_BASE, true) !== sig) {
        err(22);
        return;
      }
    }
    handleKill(infoPtr !== 0 ? 138 : 129, cpu2, view, { targetPid: target, sig, infoPtr });
  }
  function handleKill(nr, cpu2, view, from) {
    const targetPid = from ? from.targetPid : Number(BigInt.asIntN(32, cpu2.getReg(0)));
    const sig = from ? from.sig : nr === 131 || nr === 240 ? Number(BigInt.asIntN(32, cpu2.getReg(2))) : Number(BigInt.asIntN(32, cpu2.getReg(1)));
    if (sig < 0 || sig > 64) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    const namedTid = from ? 0 : nr === 131 || nr === 240 ? Number(BigInt.asIntN(32, cpu2.getReg(1))) : 0;
    if (nr === 130 && targetPid <= 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    if (nr === 138 && targetPid <= 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-3)));
      return;
    }
    if ((nr === 131 || nr === 240) && (targetPid <= 0 || namedTid <= 0)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    const myPgid = Atomics.load(procTable, procSlotOffset(mySlot) / 4 + PROC_PGID);
    const myTgid = Atomics.load(procTable, procSlotOffset(mySlot) / 4 + PROC_TGID) || myPid;
    const myCredBase = procCredOffset(mySlot) / 4;
    const myRuid = Atomics.load(procCreds, myCredBase);
    const myEuid = Atomics.load(procCreds, myCredBase + 1);
    let info = null;
    if (nr === 138 || nr === 240) {
      const uinfo = from ? from.infoPtr : Number(cpu2.getReg(nr === 138 ? 2 : 3));
      if (syscalls.ptrFaults(new Uint8Array(memory.buffer), uinfo, 128)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
      const ib = uinfo + GUEST_MEM_BASE;
      const code = view.getInt32(ib + 8, true);
      const SI_TKILL2 = -6;
      if ((code >= 0 || code === SI_TKILL2) && targetPid !== myTgid) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-1)));
        return;
      }
      info = {
        signo: sig,
        code,
        pid: view.getInt32(ib + 16, true),
        uid: view.getInt32(ib + 20, true),
        value: view.getBigUint64(ib + 24, true)
      };
    }
    const SI_TKILL = -6;
    const sentCode = info ? info.code : nr === 130 || nr === 131 ? SI_TKILL : 0;
    let delivered = false;
    let denied = false;
    let overLimit = false;
    for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
      const base = procSlotOffset(slot) / 4;
      const pid = Atomics.load(procTable, base + PROC_PID);
      if (pid === 0) continue;
      const pgid = Atomics.load(procTable, base + PROC_PGID);
      let match;
      if (namedTid > 0) match = pid === namedTid && Atomics.load(procTable, base + PROC_TGID) === targetPid;
      else if (targetPid > 0) match = pid === targetPid;
      else if (targetPid === 0) match = pgid === myPgid;
      else if (targetPid === -1) match = pid !== 1 && pid !== myPid;
      else match = pgid === -targetPid;
      if (!match) continue;
      if (myEuid !== 0 && pid !== myPid) {
        const tb = procCredOffset(slot) / 4;
        const tRuid = Atomics.load(procCreds, tb);
        const tSuid = Atomics.load(procCreds, tb + 2);
        if (myRuid !== tRuid && myRuid !== tSuid && myEuid !== tRuid && myEuid !== tSuid) {
          denied = true;
          continue;
        }
      }
      delivered = true;
      if (sig === 0) continue;
      if (sig === 18 && Atomics.load(procTable, base + PROC_STATE) === PROC_STOPPED) {
        Atomics.store(procTable, base + PROC_CONT_PENDING, 1);
        Atomics.store(procTable, base + PROC_STATE, PROC_RUNNING);
        Atomics.store(procTable, base + PROC_WAIT_FUTEX, 1);
        Atomics.notify(procTable, base + PROC_WAIT_FUTEX);
        wakeParentOf(pid);
      }
      if (sig !== 9 && sentCode < 0 && !sigpendingAllows(slot)) {
        if (sig >= 32) {
          overLimit = true;
          delivered = false;
          continue;
        }
      }
      if (pid === myPid) {
        syscalls.raiseSignal(sig, info ?? { pid: myPid, uid: syscalls.currentEuid(), code: sentCode, value: 0n });
        continue;
      }
      if (sig === 9) {
        self.postMessage({ type: "kill", pid, sig });
        continue;
      }
      sigqPush(sigQueue, slot, info ?? { signo: sig, code: sentCode, pid: myPid, uid: syscalls.currentEuid(), value: 0n });
      Atomics.store(procTable, base + PROC_WAIT_FUTEX, 1);
      Atomics.notify(procTable, base + PROC_WAIT_FUTEX);
      bumpIoGeneration();
    }
    const err = delivered ? 0 : overLimit ? -11 : denied ? -1 : -3;
    cpu2.setReg(0, BigInt.asUintN(64, BigInt(err)));
  }
  function slotOfPid(pid) {
    for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
      const base = procSlotOffset(slot) / 4;
      if (Atomics.load(procTable, base + PROC_PID) === pid && Atomics.load(procTable, base + PROC_STATE) !== PROC_EMPTY) return base;
    }
    return -1;
  }
  function sessionOf(base) {
    return Atomics.load(procTable, base + PROC_SID);
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
    const target = slotOfPid(pid);
    if (target < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-3)));
      return;
    }
    const me = procSlotOffset(mySlot) / 4;
    if (pid !== myPid) {
      if (Atomics.load(procTable, target + PROC_PPID) !== myPid) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-3)));
        return;
      }
      if (sessionOf(target) !== sessionOf(me)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-1)));
        return;
      }
      if (Atomics.load(procTable, target + PROC_EXECED)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-13)));
        return;
      }
    }
    if (sessionOf(target) === pid) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-1)));
      return;
    }
    if (pgid !== pid) {
      let found = false;
      for (let slot = 0; slot < PROC_TABLE_SLOTS && !found; slot++) {
        const base = procSlotOffset(slot) / 4;
        if (Atomics.load(procTable, base + PROC_STATE) === PROC_EMPTY) continue;
        if (Atomics.load(procTable, base + PROC_PGID) !== pgid) continue;
        found = sessionOf(base) === sessionOf(target);
      }
      if (!found) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-1)));
        return;
      }
    }
    Atomics.store(procTable, target + PROC_PGID, pgid);
    cpu2.setReg(0, 0n);
  }
  function handleSetsid(cpu2) {
    const me = procSlotOffset(mySlot) / 4;
    for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
      const base = procSlotOffset(slot) / 4;
      if (Atomics.load(procTable, base + PROC_STATE) === PROC_EMPTY) continue;
      if (Atomics.load(procTable, base + PROC_PGID) === myPid) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-1)));
        return;
      }
    }
    Atomics.store(procTable, me + PROC_SID, myPid);
    Atomics.store(procTable, me + PROC_PGID, myPid);
    cpu2.setReg(0, BigInt(myPid));
  }
  function handleGetsid(cpu2) {
    let pid = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    if (pid === 0) pid = myPid;
    if (pid < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-3)));
      return;
    }
    const base = slotOfPid(pid);
    if (base < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-3)));
      return;
    }
    cpu2.setReg(0, BigInt(sessionOf(base)));
  }
  function handleGetpgid(cpu2) {
    let pid = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    if (pid === 0) pid = myPid;
    if (pid < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-3)));
      return;
    }
    const base = slotOfPid(pid);
    if (base < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-3)));
      return;
    }
    cpu2.setReg(0, BigInt(Atomics.load(procTable, base + PROC_PGID)));
  }
  function handleFcntlLock(cpu2, view, cmd, scope) {
    const fd = Number(cpu2.getReg(0));
    const flockAddr = Number(cpu2.getReg(2));
    if (syscalls.ptrWouldFault(new Uint8Array(memory.buffer), flockAddr, 32)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
      return;
    }
    const flockPtr = flockAddr + GUEST_MEM_BASE;
    const ltype = view.getInt16(flockPtr, true);
    const whence = view.getInt16(flockPtr + 2, true);
    const lstart = Number(view.getBigInt64(flockPtr + 8, true));
    const len = Number(view.getBigInt64(flockPtr + 16, true));
    if (!syscalls.fdIsOpen(fd) && !kernelPipes.has(fd)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-9)));
      return;
    }
    if (ltype !== 0 && ltype !== 1 && ltype !== 2) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    if (whence !== 0 && whence !== 1 && whence !== 2) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    if (cmd !== 5 && cmd !== 12 && cmd !== 36) {
      const acc = syscalls.fdAccess(fd);
      if (ltype === 0 && !acc.readable) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-9)));
        return;
      }
      if (ltype === 1 && !acc.writable) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-9)));
        return;
      }
    }
    const path = syscalls.fdPath(fd);
    if (path === null) {
      cpu2.setReg(0, 0n);
      return;
    }
    let start = lstart;
    if (whence === 1) start += nativeFdOffset(fd);
    else if (whence === 2) start += nativeFdSize(fd, path);
    if (start < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    Atomics.store(myMailbox, MBOX_READY, 0);
    if (scope === "ofd" && view.getInt32(flockPtr + 24, true) !== 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    const ownerId = scope === "ofd" ? syscalls.fdDescriptionId(fd) : 0;
    usedLocks = true;
    self.postMessage({ type: "lock_req", space: "record", scope, ownerId, path, cmd, ltype, start, len });
    const r = waitLockReply();
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
  function newSyscallHandler(brk, io, fs) {
    const h = new SyscallHandler(brk, io, fs);
    h.sharedMapHook = registerSharedMap;
    h.bindSysctls = (vfs2) => {
      vfs2.setSysctl(
        "/proc/sys/fs/pipe-max-size",
        () => `${Atomics.load(globalWords, GLOBAL_PIPE_MAX)}
`,
        (text) => {
          const v = parseInt(text, 10);
          if (!Number.isFinite(v) || v < 4096) return -22;
          if (!h.hasCap(21)) return -1;
          Atomics.store(globalWords, GLOBAL_PIPE_MAX, v);
          return 0;
        }
      );
      vfs2.setSysctl(
        "/proc/sys/kernel/msgmni",
        () => `${Atomics.load(globalWords, GLOBAL_MSGMNI)}
`,
        (text) => {
          const v = parseInt(text, 10);
          if (!Number.isFinite(v) || v < 0 || v > 32768) return -22;
          if (!h.hasCap(21)) return -1;
          Atomics.store(globalWords, GLOBAL_MSGMNI, v);
          return 0;
        }
      );
      for (const [path, slot] of [
        ["msg", GLOBAL_MSG_NEXT_ID],
        ["sem", GLOBAL_SEM_NEXT_ID],
        ["shm", GLOBAL_SHM_NEXT_ID]
      ]) {
        vfs2.setSysctl(
          `/proc/sys/kernel/${path}_next_id`,
          () => `${Atomics.load(globalWords, slot)}
`,
          (text) => {
            const v = parseInt(text, 10);
            if (!Number.isFinite(v) || v < -1 || v > 2147483647) return -22;
            if (!h.hasCap(21)) return -1;
            Atomics.store(globalWords, slot, v);
            return 0;
          }
        );
      }
      vfs2.setSysctl(
        "/proc/sys/vm/max_map_count",
        () => `${h.getMaxMapCount()}
`,
        (text) => {
          const v = parseInt(text, 10);
          if (!Number.isFinite(v) || v < 1) return -22;
          if (!h.hasCap(21)) return -1;
          return h.setMaxMapCount(v);
        }
      );
      const mqLimit = (path, word, min, max) => {
        vfs2.setSysctl(
          path,
          () => `${Atomics.load(globalWords, word)}
`,
          (text) => {
            const v = parseInt(text, 10);
            if (!Number.isFinite(v) || v < min || v > max) return -22;
            if (!h.hasCap(21)) return -1;
            Atomics.store(globalWords, word, v);
            return 0;
          }
        );
      };
      mqLimit("/proc/sys/fs/mqueue/msg_max", GLOBAL_MQ_MSG_MAX, 1, 65536);
      mqLimit("/proc/sys/fs/mqueue/msgsize_max", GLOBAL_MQ_MSGSIZE_MAX, 128, 16 * 1024 * 1024);
      mqLimit("/proc/sys/fs/mqueue/queues_max", GLOBAL_MQ_QUEUES_MAX, 0, 1024);
      const shmLimit = (path, which, idx) => {
        vfs2.setSysctl(
          path,
          () => {
            const r = sysvShmCall({ op: "limits" });
            const v = BigInt(r[idx + 1] >>> 0) << 32n | BigInt(r[idx] >>> 0);
            return `${v}
`;
          },
          (text) => {
            let v;
            try {
              v = BigInt(text.trim());
            } catch {
              return -22;
            }
            if (v < 0n || v > 0xffffffffffffffffn) return -22;
            if (!h.hasCap(21)) return -1;
            sysvShmCall({
              op: "setlimit",
              which,
              lo: Number(v & 0xffffffffn) | 0,
              hi: Number(v >> 32n & 0xffffffffn) | 0
            });
            return 0;
          }
        );
      };
      vfs2.setSysctl(
        "/proc/self/timerslack_ns",
        () => `${h.getTimerSlack()}
`,
        (text) => {
          const v = parseInt(text, 10);
          if (!Number.isFinite(v)) return -22;
          if (!h.hasCap(23) && !h.hasCap(21)) return -1;
          return h.setTimerSlack(v);
        }
      );
      shmLimit("/proc/sys/kernel/shmmax", 0, 1);
      shmLimit("/proc/sys/kernel/shmall", 1, 3);
      vfs2.setDynamicFile("/proc/uptime", () => new TextEncoder().encode(
        `${(Number(h.clockNs(1) / 1000000n) / 1e3).toFixed(2)} 0.00
`
      ));
    };
    h.pipeMaxHook = {
      get: () => Atomics.load(globalWords, GLOBAL_PIPE_MAX),
      // changing a system-wide limit is root's business, exactly as writing /proc/sys is on linux
      set: (v) => {
        if (!h.hasCap(21)) return -1;
        Atomics.store(globalWords, GLOBAL_PIPE_MAX, v);
        return 0;
      }
      // CAP_SYS_ADMIN
    };
    h.refusalHook = (nr, errno) => {
      self.postMessage({ type: "refusal", nr, errno });
    };
    h.readNativeHook = (nativeFd, count) => {
      if (count <= 0) return new Uint8Array(0);
      const CHUNK = 12288;
      const out = new Uint8Array(count);
      let got = 0;
      while (got < count) {
        const want = Math.min(CHUNK, count - got);
        const h2 = { data: new Uint8Array(0), err: 0 };
        handleBridgeCall("fs_read", { fd: nativeFd, count: want }, (r) => {
          if (r && typeof r.ret === "number" && r.ret < 0) {
            h2.err = -r.ret;
            return;
          }
          if (r && typeof r.data === "string" && r.data.length) {
            const bin = atob(r.data);
            const u = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
            h2.data = u;
          }
        });
        if (h2.err) return got > 0 ? out.subarray(0, got) : h2.err;
        if (h2.data.length === 0) break;
        out.set(h2.data, got);
        got += h2.data.length;
        if (h2.data.length < want) break;
      }
      return out.subarray(0, got);
    };
    h.pipeResizeHook = (fd, bytes) => {
      const info = kernelPipes.get(fd);
      if (!info) return -1;
      return new PipeRing(shmArena, info.pipeOff).resize(bytes);
    };
    h.ownerChanged = (fd) => {
      const info = kernelPipes.get(fd);
      if (!info) return;
      const a = h.asyncState(fd);
      self.postMessage({ type: "pipe_async", off: info.pipeOff, mode: info.mode, sig: a.sig, on: a.armed });
    };
    h.pipeCapacityHook = (fd) => {
      const info = kernelPipes.get(fd);
      return info ? new PipeRing(shmArena, info.pipeOff).capacity : null;
    };
    h.ppidHook = () => {
      const myBase = procSlotOffset(mySlot) / 4;
      const tgid = Atomics.load(procTable, myBase + PROC_TGID);
      if (tgid !== 0 && tgid !== myPid) {
        const lead = slotOfPid(tgid);
        if (lead >= 0) return Atomics.load(procTable, lead + PROC_PPID);
      }
      return Atomics.load(procTable, myBase + PROC_PPID);
    };
    h.closeNativeHook = (nativeFd) => {
      handleBridgeCall("fs_close", { fd: nativeFd }, () => {
      });
    };
    h.closeSocketHook = (socketId) => {
      handleBridgeCall("net_close", { socketId }, () => {
      });
    };
    h.closeUnixHook = (fd) => {
      closeUnixSock(fd);
    };
    h.handleHook = {
      put: (ino, path) => {
        handleBridgeCall("handle_put", { ino, path }, () => {
        });
      },
      get: (ino) => {
        let out = null;
        handleBridgeCall("handle_get", { ino }, (resp) => {
          if (resp.ret > 0 && resp.bytes) out = new TextDecoder().decode(resp.bytes);
        });
        return out;
      }
    };
    h.peerCredHook = (fd) => unixSocks.get(fd)?.peer ?? null;
    h.subreaperHook = {
      get: () => Atomics.load(procTable, procSlotOffset(mySlot) / 4 + PROC_SUBREAPER),
      set: (on) => {
        Atomics.store(procTable, procSlotOffset(mySlot) / 4 + PROC_SUBREAPER, on);
      }
    };
    h.sharedUnmapHook = releaseSharedMaps;
    if (globalWords && vfs) h.bindSysctls?.(vfs);
    return h;
  }
  var shmArena;
  var sharedMaps = [];
  function waitLockReply() {
    let asked = false;
    while (Atomics.load(myMailbox, MBOX_READY) === 0) {
      Atomics.wait(myMailbox, MBOX_READY, 0, 50);
      if (Atomics.load(myMailbox, MBOX_READY) !== 0) break;
      if (asked) continue;
      if (sigqPending(procTable, mySlot) || syscalls.hasPendingDeliverable()) {
        asked = true;
        self.postMessage({ type: "lock_cancel" });
      }
    }
    return Atomics.load(myMailbox, MBOX_VALUE0);
  }
  var MSG_DONTWAIT = 64;
  var unixSocks = /* @__PURE__ */ new Map();
  function readSunAddr(view, addr, addrLen) {
    if (addr === 0 || addrLen < 2 || addrLen > 110) return null;
    const base = addr + GUEST_MEM_BASE;
    if (view.getUint16(base, true) !== 1) return null;
    const bytes = new Uint8Array(memory.buffer, base + 2, Math.min(108, addrLen - 2));
    if (bytes.length === 0) return { raw: "", abstract: false };
    if (bytes[0] === 0) return { raw: "\0" + new TextDecoder().decode(bytes.subarray(1)), abstract: true };
    let n = bytes.indexOf(0);
    if (n < 0) n = bytes.length;
    return { raw: new TextDecoder().decode(bytes.subarray(0, n)), abstract: false };
  }
  function unixKey(a) {
    return a.abstract ? a.raw : syscalls.socketBindPath(a.raw) ?? a.raw;
  }
  function putSunAddr(view, addrPtr, lenPtr, name) {
    const bytes = new Uint8Array(memory.buffer);
    if (lenPtr === 0 || syscalls.ptrFaults(bytes, lenPtr, 4)) return -14;
    const want = view.getInt32(lenPtr + GUEST_MEM_BASE, true);
    if (want < 0) return -22;
    const enc = new TextEncoder();
    let body;
    if (!name) body = new Uint8Array(0);
    else if (name.charCodeAt(0) === 0) {
      const nm = enc.encode(name.slice(1));
      body = new Uint8Array(1 + nm.length);
      body.set(nm, 1);
    } else {
      const nm = enc.encode(name);
      body = new Uint8Array(nm.length + 1);
      body.set(nm, 0);
    }
    const full = 2 + body.length;
    const copy = Math.min(Math.max(0, want), full);
    if (copy > 0 && (addrPtr === 0 || syscalls.ptrFaults(bytes, addrPtr, copy))) return -14;
    const buf = new Uint8Array(full);
    buf[0] = 1;
    buf[1] = 0;
    buf.set(body, 2);
    if (copy > 0) bytes.set(buf.subarray(0, copy), addrPtr + GUEST_MEM_BASE);
    view.setUint32(lenPtr + GUEST_MEM_BASE, full, true);
    return 0;
  }
  var autobindSeq = 0;
  function unixCall(type, path, dgram = false) {
    Atomics.store(myMailbox, MBOX_READY, 0);
    const uc = syscalls.getCreds();
    self.postMessage({ type, path, dgram, uid: uc.euid, gid: uc.egid });
    awaitMailbox("unix");
    return [0, 1, 2, 3, 4, 5].map((i) => Atomics.load(myMailbox, MBOX_VALUE0 + i));
  }
  function handleUnixSocket(cpu2, sockType) {
    const kind = sockType & 255;
    if (kind !== 1 && kind !== 2 && kind !== 5) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-94)));
      return;
    }
    if (Number(cpu2.getReg(2)) !== 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-93)));
      return;
    }
    const fd = syscalls.allocLowestFd();
    if (fd < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-24)));
      return;
    }
    syscalls.registerFd(fd, { type: "unixsock", sockType: kind });
    unixSocks.set(fd, { listening: false, dgram: kind === 2, framed: kind !== 1 });
    cpu2.setReg(0, BigInt(fd));
  }
  function handleNetlinkSocket(cpu2, sockType) {
    const kind = sockType & 15;
    if (kind !== 3 && kind !== 2) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-94)));
      return;
    }
    const fd = syscalls.allocLowestFd();
    if (fd < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-24)));
      return;
    }
    Atomics.store(myMailbox, MBOX_READY, 0);
    self.postMessage({ type: "nl_socket", fd });
    awaitMailbox("netlink");
    const r = Atomics.load(myMailbox, MBOX_VALUE0);
    if (r !== 0) {
      syscalls.unregisterFd(fd);
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(r)));
      return;
    }
    const ring = Atomics.load(myMailbox, MBOX_VALUE0 + 1);
    syscalls.registerFd(fd, { type: "unixsock" });
    unixSocks.set(fd, { listening: false, rx: ring });
    if (sockType & 524288) syscalls.markCloexec(fd);
    if (sockType & 2048) syscalls.registerFd(fd, { type: "unixsock" }, 2048);
    cpu2.setReg(0, BigInt(fd));
  }
  function handleUnixSocketpair(cpu2, view, sockType) {
    const SOCK_STREAM = 1, SOCK_DGRAM = 2, SOCK_SEQPACKET = 5;
    const kind = sockType & 15;
    if (Number(cpu2.getReg(2)) !== 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-93)));
      return;
    }
    if (kind !== SOCK_STREAM && kind !== SOCK_DGRAM && kind !== SOCK_SEQPACKET) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    const SOCK_NONBLOCK_F = 2048, SOCK_CLOEXEC_F = 524288;
    if (sockType & ~(15 | SOCK_NONBLOCK_F | SOCK_CLOEXEC_F)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    const svPtr = Number(cpu2.getReg(3));
    if (svPtr === 0 || badGuestPtr(new Uint8Array(memory.buffer), svPtr, 8)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
      return;
    }
    const fd0 = syscalls.allocLowestFd();
    if (fd0 < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-24)));
      return;
    }
    syscalls.registerFd(fd0, { type: "unixsock" });
    const fd1 = syscalls.allocLowestFd();
    if (fd1 < 0) {
      syscalls.unregisterFd(fd0);
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-24)));
      return;
    }
    syscalls.registerFd(fd1, { type: "unixsock" });
    Atomics.store(myMailbox, MBOX_READY, 0);
    self.postMessage({ type: "unix_pair" });
    awaitMailbox("unix");
    const r = Atomics.load(myMailbox, MBOX_VALUE0);
    if (r !== 0) {
      syscalls.unregisterFd(fd0);
      syscalls.unregisterFd(fd1);
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(r)));
      return;
    }
    const a = Atomics.load(myMailbox, MBOX_VALUE0 + 1);
    const b = Atomics.load(myMailbox, MBOX_VALUE0 + 2);
    const dgram = kind !== SOCK_STREAM;
    const spCreds = syscalls.getCreds();
    const spPeer = { pid: myPid, uid: spCreds.euid, gid: spCreds.egid };
    unixSocks.set(fd0, { listening: false, tx: a, rx: b, framed: dgram, peer: spPeer });
    unixSocks.set(fd1, { listening: false, tx: b, rx: a, framed: dgram, peer: spPeer });
    const SOCK_NONBLOCK = SOCK_NONBLOCK_F, SOCK_CLOEXEC = SOCK_CLOEXEC_F;
    const O_NONBLOCK_FLAG = 2048;
    if (sockType & SOCK_NONBLOCK) {
      syscalls.registerFd(fd0, { type: "unixsock" }, O_NONBLOCK_FLAG);
      syscalls.registerFd(fd1, { type: "unixsock" }, O_NONBLOCK_FLAG);
    }
    if (sockType & SOCK_CLOEXEC) {
      syscalls.markCloexec(fd0);
      syscalls.markCloexec(fd1);
    }
    view.setInt32(svPtr + GUEST_MEM_BASE, fd0, true);
    view.setInt32(svPtr + GUEST_MEM_BASE + 4, fd1, true);
    cpu2.setReg(0, 0n);
  }
  function handleUnixBind(cpu2, view) {
    const fd = Number(cpu2.getReg(0));
    const sock = unixSocks.get(fd);
    const a = readSunAddr(view, Number(cpu2.getReg(1)), Number(cpu2.getReg(2)));
    if (a === null) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    if (sock.path) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    if (a.raw === "") {
      for (let i = 0; i < 1048576; i++) {
        const nm = "\0" + (autobindSeq++ & 1048575).toString(16).padStart(5, "0");
        const [rc, inbox2] = unixCall("unix_bind", nm, !!sock.dgram);
        if (rc === 0) {
          sock.path = nm;
          sock.name = nm;
          if (sock.dgram && inbox2 >= 0) sock.rx = inbox2;
          cpu2.setReg(0, 0n);
          return;
        }
        if (rc !== -98) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(rc)));
          return;
        }
      }
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-98)));
      return;
    }
    const path = unixKey(a);
    const disk = a.abstract ? null : syscalls.socketBindPath(a.raw);
    if (disk !== null) {
      let r2 = -5;
      handleBridgeCall(
        "fs_mksock",
        { path: disk, mode: 511 },
        (resp) => {
          r2 = typeof resp?.ret === "number" ? resp.ret : -5;
        }
      );
      if (r2 < 0) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(r2)));
        return;
      }
    }
    const [r, inbox] = unixCall("unix_bind", path, !!sock.dgram);
    if (r === 0) {
      sock.path = path;
      sock.name = a.raw;
      if (sock.dgram && inbox >= 0) sock.rx = inbox;
    }
    cpu2.setReg(0, r === 0 ? 0n : BigInt.asUintN(64, BigInt(r)));
  }
  function handleUnixGetName(cpu2, view, peer) {
    const sock = unixSocks.get(Number(cpu2.getReg(0)));
    if (peer && sock.rx === void 0 && sock.tx === void 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-107)));
      return;
    }
    const r = putSunAddr(
      view,
      Number(cpu2.getReg(1)),
      Number(cpu2.getReg(2)),
      peer ? sock.peerName : sock.name
    );
    cpu2.setReg(0, r === 0 ? 0n : BigInt.asUintN(64, BigInt(r)));
  }
  function dgramRingOf(name) {
    const [r, inbox] = unixCall("unix_lookup", unixKey({ raw: name, abstract: name.charCodeAt(0) === 0 }));
    return r === 0 ? inbox : r;
  }
  function dgramDefaultRing(sock) {
    if (sock.dgram && sock.peerName !== void 0) return dgramRingOf(sock.peerName);
    if (sock.tx !== void 0) return sock.tx;
    return -107;
  }
  function autobindIfUnnamed(sock) {
    if (sock.path) return 0;
    for (let i = 0; i < 1048576; i++) {
      const nm = "\0" + (autobindSeq++ & 1048575).toString(16).padStart(5, "0");
      const [rc, inbox] = unixCall("unix_bind", nm, true);
      if (rc === 0) {
        sock.path = nm;
        sock.name = nm;
        if (inbox >= 0) sock.rx = inbox;
        return 0;
      }
      if (rc !== -98) return rc;
    }
    return -98;
  }
  function handleUnixListen(cpu2) {
    const sock = unixSocks.get(Number(cpu2.getReg(0)));
    if (!sock.path) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    const r = unixCall("unix_listen", sock.path)[0];
    if (r === 0) {
      sock.listening = true;
      const e = syscalls.getFdEntry(Number(cpu2.getReg(0)));
      if (e && e.type === "unixsock") e.listening = true;
    }
    cpu2.setReg(0, r === 0 ? 0n : BigInt.asUintN(64, BigInt(r)));
  }
  function handleUnixConnect(cpu2, view) {
    const fd = Number(cpu2.getReg(0));
    const sock = unixSocks.get(fd);
    const addr = readSunAddr(view, Number(cpu2.getReg(1)), Number(cpu2.getReg(2)));
    if (addr === null || addr.raw === "") {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    if (sock.dgram) {
      const ab = autobindIfUnnamed(sock);
      if (ab < 0) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(ab)));
        return;
      }
      const [dr, , dpid, duid, dgid] = unixCall("unix_lookup", unixKey(addr));
      if (dr !== 0) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(dr)));
        return;
      }
      sock.peerName = addr.raw;
      sock.peer = { pid: dpid, uid: duid, gid: dgid };
      cpu2.setReg(0, 0n);
      return;
    }
    const [r, a, b, ppid, puid, pgid] = unixCall("unix_connect", unixKey(addr));
    if (r !== 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(r)));
      return;
    }
    sock.tx = a;
    sock.rx = b;
    sock.peerName = addr.raw;
    sock.peer = { pid: ppid, uid: puid, gid: pgid };
    cpu2.setReg(0, 0n);
  }
  function handleUnixAccept(cpu2, view) {
    const sock = unixSocks.get(Number(cpu2.getReg(0)));
    if (!sock.path || !sock.listening) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    const nonblock = syscalls.isNonBlockingFd(Number(cpu2.getReg(0)));
    for (; ; ) {
      const [r, rx, tx, ppid, puid, pgid] = unixCall("unix_accept", sock.path);
      if (r === 0) {
        const nfd = syscalls.allocLowestFd();
        if (nfd < 0) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(-24)));
          return;
        }
        syscalls.registerFd(nfd, { type: "unixsock" });
        unixSocks.set(nfd, {
          listening: false,
          rx,
          tx,
          name: sock.name,
          framed: sock.framed,
          peer: { pid: ppid, uid: puid, gid: pgid }
        });
        const addrPtr = Number(cpu2.getReg(1));
        if (addrPtr !== 0) putSunAddr(view, addrPtr, Number(cpu2.getReg(2)), void 0);
        cpu2.setReg(0, BigInt(nfd));
        return;
      }
      if (r !== -11 || nonblock) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(r)));
        return;
      }
      if (pollAndQueueSignals()) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-4)));
        return;
      }
      Atomics.wait(myMailbox, MBOX_READY, Atomics.load(myMailbox, MBOX_READY), 20);
    }
  }
  function unixReady(fd) {
    const sock = unixSocks.get(fd);
    if (!sock) return { readable: false, writable: false };
    if (sock.listening && sock.path) {
      return { readable: unixCall("unix_pending", sock.path)[1] > 0, writable: false };
    }
    if (sock.dgram && (sock.peerName !== void 0 || sock.path !== void 0)) {
      const inbox = sock.rx !== void 0 ? new PipeRing(shmArena, sock.rx) : null;
      return { readable: inbox ? inbox.available > 0 : false, writable: true };
    }
    if (sock.rx === void 0 && sock.tx === void 0) return { readable: false, writable: false };
    if (sock.rx === void 0 || sock.tx === void 0) {
      const one = sock.rx !== void 0 ? new PipeRing(shmArena, sock.rx) : null;
      const outRing = sock.tx !== void 0 ? new PipeRing(shmArena, sock.tx) : null;
      return {
        readable: one ? one.available > 0 || one.writerClosed : false,
        writable: outRing ? !outRing.readerClosed && outRing.available < outRing.capacity : false
      };
    }
    const rx = new PipeRing(shmArena, sock.rx);
    const tx = new PipeRing(shmArena, sock.tx);
    return {
      readable: rx.available > 0 || rx.writerClosed,
      writable: !tx.readerClosed && tx.available < tx.capacity
    };
  }
  function waitIo(seen, timeoutMs) {
    const rem = syscalls.timerRemainingMs();
    const cap = rem > 0 ? Math.min(1e3, rem) : 1e3;
    const want = timeoutMs < 0 ? cap : Math.min(cap, timeoutMs);
    if (timeoutMs >= 0 && timeoutMs <= 1.2) return waitIoGeneration(seen, 0);
    setSleeping(true);
    try {
      const r = waitIoGeneration(seen, Math.max(1, want * 0.7));
      drainSignalQueue();
      return r;
    } finally {
      setSleeping(false);
    }
  }
  function mqReady(fd) {
    const id = syscalls.mqIdOf(fd);
    if (id < 0) return null;
    const r = posixMqCall({ op: "getattr", id });
    if (r[0] < 0) return null;
    const access = syscalls.fdAccess(fd);
    return { readable: access.readable && r[3] > 0, writable: access.writable && r[3] < r[1] };
  }
  function pipeReady(fd) {
    const info = kernelPipes.get(fd);
    if (!info) return null;
    const ring = new PipeRing(shmArena, info.pipeOff);
    if (info.mode === "read") {
      return { readable: ring.bytesAvailable > 0 || ring.writerClosed, writable: false };
    }
    const PIPE_BUF = 4096;
    return {
      readable: false,
      writable: ring.freeBytes >= Math.min(PIPE_BUF, ring.capacity) || ring.readerClosed
    };
  }
  function closeUnixSock(fd) {
    const sock = unixSocks.get(fd);
    if (!sock) return;
    if (sock.rx !== void 0) pipeRef(sock.rx, "read", -1);
    if (sock.tx !== void 0) pipeRef(sock.tx, "write", -1);
    if (sock.path) unixCall("unix_unbind", sock.path);
    unixSocks.delete(fd);
    syscalls.unregisterFd(fd);
  }
  function handleUnixSendto(cpu2, view) {
    const fd = Number(cpu2.getReg(0));
    const sock = unixSocks.get(fd);
    const addrPtr = Number(cpu2.getReg(4));
    const dontwait = (Number(BigInt.asIntN(32, cpu2.getReg(3))) & MSG_DONTWAIT) !== 0;
    if (!sock.dgram || addrPtr === 0) {
      handleUnixIO(64, cpu2, dontwait);
      return;
    }
    const a = readSunAddr(view, addrPtr, Number(cpu2.getReg(5)));
    if (a === null || a.raw === "") {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    const ab = autobindIfUnnamed(sock);
    if (ab < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(ab)));
      return;
    }
    const ring = dgramRingOf(a.raw);
    if (ring < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(ring)));
      return;
    }
    const buf = Number(cpu2.getReg(1));
    const count = Number(cpu2.getReg(2));
    const n = unixDgramWrite(fd, ring, new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, count), sock.name, dontwait);
    cpu2.setReg(0, n < 0 ? BigInt.asUintN(64, BigInt(n)) : BigInt(n));
  }
  function handleUnixRecvfrom(cpu2, view) {
    const fd = Number(cpu2.getReg(0));
    const sock = unixSocks.get(fd);
    const addrPtr = Number(cpu2.getReg(4));
    const dontwait = (Number(BigInt.asIntN(32, cpu2.getReg(3))) & MSG_DONTWAIT) !== 0;
    if (!sock.dgram || addrPtr === 0) {
      handleUnixIO(63, cpu2, dontwait);
      return;
    }
    if (sock.rx === void 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-107)));
      return;
    }
    const buf = Number(cpu2.getReg(1));
    const count = Number(cpu2.getReg(2));
    const got = unixDgramRead(fd, sock.rx, count, dontwait);
    if (typeof got === "number") {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(got)));
      return;
    }
    if (got.data.length > 0) {
      new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, got.data.length).set(got.data);
    }
    putSunAddr(view, addrPtr, Number(cpu2.getReg(5)), got.from || void 0);
    cpu2.setReg(0, BigInt(got.data.length));
  }
  function handleUnixIO(nr, cpu2, dontwait = false) {
    const fd = Number(cpu2.getReg(0));
    const sock = unixSocks.get(fd);
    if (nr === 57) {
      closeUnixSock(fd);
      cpu2.setReg(0, 0n);
      return;
    }
    const canSend = sock.tx !== void 0 || sock.dgram === true && sock.peerName !== void 0;
    if (nr === 63 && sock.rx === void 0 || nr !== 63 && !canSend) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-107)));
      return;
    }
    const buf = Number(cpu2.getReg(1));
    const count = Number(cpu2.getReg(2));
    if (sock.framed) {
      handleUnixDgramIO(nr, cpu2, sock, fd, buf, count, dontwait);
      return;
    }
    if (nr === 63) {
      const got = unixStreamRead(fd, sock.rx, count, dontwait);
      if (typeof got === "number") {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(got)));
        return;
      }
      if (got.data.length > 0) {
        new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, got.data.length).set(got.data);
      }
      cpu2.setReg(0, BigInt(got.data.length));
      return;
    }
    const n = unixStreamWrite(fd, sock.tx, new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, count), dontwait);
    cpu2.setReg(0, n < 0 ? BigInt.asUintN(64, BigInt(n)) : BigInt(n));
  }
  function unixStreamWrite(fd, txOff, src, dontwait = false) {
    const ring = new PipeRing(shmArena, txOff);
    if (dontwait || syscalls.isNonBlockingFd(fd)) {
      const n2 = ring.write(src);
      if (n2 === -1) {
        syscalls.raiseSignal(13);
        return -32;
      }
      return n2 === 0 && src.length > 0 ? -11 : n2;
    }
    const n = ring.writeAll(src, interruptibleWaiter);
    if (n === -1) {
      syscalls.raiseSignal(13);
      return -32;
    }
    return n === -2 ? -4 : n;
  }
  function unixStreamRead(fd, rxOff, cap, dontwait = false) {
    const ring = new PipeRing(shmArena, rxOff);
    const dst = new Uint8Array(cap);
    if (dontwait || syscalls.isNonBlockingFd(fd)) {
      const n2 = ring.tryRead(dst);
      if (n2 === 0 && ring.writerClosed) return { data: dst.subarray(0, 0), truncated: false };
      if (n2 === 0 && cap > 0) return -11;
      return { data: dst.subarray(0, n2), truncated: false };
    }
    const n = ring.read(dst, interruptibleWaiter);
    if (n < 0) return -4;
    return { data: dst.subarray(0, n), truncated: false };
  }
  function handleUnixDgramIO(nr, cpu2, sock, fd, buf, count, dontwait = false) {
    if (nr === 63) {
      const got = unixDgramRead(fd, sock.rx, count, dontwait);
      if (typeof got === "number") {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(got)));
        return;
      }
      if (got.data.length > 0) {
        new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, got.data.length).set(got.data);
      }
      cpu2.setReg(0, BigInt(got.data.length));
      return;
    }
    const ring = dgramDefaultRing(sock);
    if (ring < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(ring)));
      return;
    }
    const n = unixDgramWrite(fd, ring, new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, count), sock.name, dontwait);
    cpu2.setReg(0, n < 0 ? BigInt.asUintN(64, BigInt(n)) : BigInt(n));
  }
  function unixDgramWrite(fd, txOff, src, fromName, dontwait = false) {
    const ring = new PipeRing(shmArena, txOff);
    const count = src.length;
    const nameBytes = fromName ? new TextEncoder().encode(fromName) : new Uint8Array(0);
    const framed = new Uint8Array(8 + nameBytes.length + count);
    const fv = new DataView(framed.buffer);
    fv.setUint32(0, count, true);
    fv.setUint32(4, nameBytes.length, true);
    framed.set(nameBytes, 8);
    if (count > 0) framed.set(src, 8 + nameBytes.length);
    if (dontwait || syscalls.isNonBlockingFd(fd)) {
      if (ring.freeBytes < framed.length) {
        if (ring.readerClosed) {
          syscalls.raiseSignal(13);
          return -32;
        }
        return -11;
      }
      const w = ring.write(framed);
      if (w === -1) {
        syscalls.raiseSignal(13);
        return -32;
      }
      return count;
    }
    const n = ring.writeAll(framed, interruptibleWaiter);
    if (n === -1) {
      syscalls.raiseSignal(13);
      return -32;
    }
    if (n === -2) return -4;
    return count;
  }
  function unixDgramRead(fd, rxOff, cap, dontwait = false) {
    const ring = new PipeRing(shmArena, rxOff);
    const hdr = new Uint8Array(8);
    const empty = { data: hdr.subarray(0, 0), truncated: false, from: "" };
    if ((dontwait || syscalls.isNonBlockingFd(fd)) && ring.bytesAvailable < 8) {
      if (ring.writerClosed) return empty;
      return -11;
    }
    const fill = (dst) => {
      let n = 0;
      while (n < dst.length) {
        const r = ring.read(dst.subarray(n), interruptibleWaiter);
        if (r < 0) return -4;
        if (r === 0) break;
        n += r;
      }
      return n;
    };
    const got = fill(hdr);
    if (got < 0) return -4;
    if (got < 8) return empty;
    const hv = new DataView(hdr.buffer);
    const len = hv.getUint32(0, true);
    const nameLen = hv.getUint32(4, true);
    let from = "";
    if (nameLen > 0) {
      const nb = new Uint8Array(nameLen);
      if (fill(nb) < 0) return -4;
      from = new TextDecoder().decode(nb);
    }
    const body = new Uint8Array(len);
    const have = fill(body);
    if (have < 0) return -4;
    const copied = Math.min(have, cap);
    return { data: body.subarray(0, copied), truncated: have > cap, from };
  }
  function describeFdForScm(fd) {
    const e = syscalls.getFdEntry(fd);
    if (e?.type === "native") {
      let nfd = -1;
      handleBridgeCall("fs_dup", { fd: e.nativeFd }, (resp) => {
        nfd = resp.ret;
      });
      if (nfd < 0) return null;
      return { k: "native", nfd, path: e.path, ofdId: e.ofdId, isDir: e.isDir, unlinked: e.unlinked };
    }
    if (e?.type === "socket") {
      handleBridgeCall("net_ref", { socketId: e.socketId }, () => {
      });
      return { k: "socket", socketId: e.socketId, domain: e.domain, sockType: e.sockType };
    }
    if (e?.type === "unixsock") {
      const s = unixSocks.get(fd);
      if (!s || s.rx === void 0 || s.tx === void 0) return null;
      new PipeRing(shmArena, s.rx).addReader();
      new PipeRing(shmArena, s.tx).addWriter();
      return {
        k: "unix",
        rx: s.rx,
        tx: s.tx,
        dgram: s.dgram,
        framed: s.framed,
        peer: s.peer,
        ends: [[s.rx, "r"], [s.tx, "w"]]
      };
    }
    const kp = kernelPipes.get(fd);
    if (kp) {
      const ring = new PipeRing(shmArena, kp.pipeOff);
      if (kp.mode === "read") ring.addReader();
      else ring.addWriter();
      return {
        k: "pipe",
        off: kp.pipeOff,
        mode: kp.mode,
        ends: [[kp.pipeOff, kp.mode === "read" ? "r" : "w"]]
      };
    }
    const vp = vfs?.fdPath(fd);
    if (vp) return { k: "vfs", path: vp };
    return null;
  }
  function materializeScmFd(desc) {
    const fd = syscalls.allocLowestFd();
    if (fd < 0) return -1;
    switch (desc?.k) {
      case "native":
        syscalls.registerFd(fd, {
          type: "native",
          nativeFd: desc.nfd,
          path: desc.path,
          ofdId: desc.ofdId ?? 0,
          isDir: desc.isDir,
          unlinked: desc.unlinked
        });
        return fd;
      case "socket":
        syscalls.registerFd(fd, {
          type: "socket",
          socketId: desc.socketId,
          domain: desc.domain,
          sockType: desc.sockType
        });
        return fd;
      case "unix":
        syscalls.registerFd(fd, { type: "unixsock" });
        unixSocks.set(fd, {
          listening: false,
          rx: desc.rx,
          tx: desc.tx,
          dgram: desc.dgram,
          framed: desc.framed,
          peer: desc.peer
        });
        return fd;
      case "pipe":
        syscalls.registerFd(fd, {
          type: desc.mode === "read" ? "pipe_read" : "pipe_write",
          pipe: { data: [], writeClosed: false, readClosed: false }
        });
        kernelPipes.set(fd, { pipeOff: desc.off, mode: desc.mode });
        if (fd >= nextLocalFd) nextLocalFd = fd + 1;
        return fd;
      case "vfs": {
        syscalls.unregisterFd(fd);
        const r = vfs ? vfs.open(desc.path, 0, 0) : -1;
        return r;
      }
      default:
        syscalls.unregisterFd(fd);
        return -1;
    }
  }
  function readMsghdr(view, p) {
    const b = p + GUEST_MEM_BASE;
    return {
      name: Number(view.getBigUint64(b, true)),
      nameLen: view.getUint32(b + 8, true),
      iov: Number(view.getBigUint64(b + 16, true)),
      iovLen: Number(view.getBigUint64(b + 24, true)),
      control: Number(view.getBigUint64(b + 32, true)),
      controlLen: Number(view.getBigUint64(b + 40, true)),
      flags: view.getInt32(b + 48, true)
    };
  }
  function gatherIov(view, iov, n) {
    let total = 0;
    for (let i = 0; i < n; i++) total += Number(view.getBigUint64(iov + i * 16 + 8 + GUEST_MEM_BASE, true));
    const out = new Uint8Array(total);
    let off = 0;
    for (let i = 0; i < n; i++) {
      const base = Number(view.getBigUint64(iov + i * 16 + GUEST_MEM_BASE, true));
      const len = Number(view.getBigUint64(iov + i * 16 + 8 + GUEST_MEM_BASE, true));
      out.set(new Uint8Array(memory.buffer, base + GUEST_MEM_BASE, len), off);
      off += len;
    }
    return out;
  }
  function scatterIov(view, iov, n, data) {
    let left = data.length, src = 0;
    for (let i = 0; i < n && left > 0; i++) {
      const base = Number(view.getBigUint64(iov + i * 16 + GUEST_MEM_BASE, true));
      const len = Number(view.getBigUint64(iov + i * 16 + 8 + GUEST_MEM_BASE, true));
      const take = Math.min(len, left);
      new Uint8Array(memory.buffer, base + GUEST_MEM_BASE, take).set(data.subarray(src, src + take));
      src += take;
      left -= take;
    }
    return data.length - left;
  }
  function collectScmRights(view, control, controlLen) {
    const out = [];
    let off = 0;
    while (off + 16 <= controlLen) {
      const b = control + off + GUEST_MEM_BASE;
      const len = Number(view.getBigUint64(b, true));
      const level = view.getInt32(b + 8, true);
      const type = view.getInt32(b + 12, true);
      if (len < 16 || off + len > controlLen) return { fds: out, err: 22 };
      if (level === 1 && type === 1) {
        const count = Math.floor((len - 16) / 4);
        for (let i = 0; i < count; i++) {
          const gfd = view.getInt32(b + 16 + i * 4, true);
          if (!syscalls.fdIsOpen(gfd)) return { fds: out, err: 9 };
          const d = describeFdForScm(gfd);
          if (!d) return { fds: out, err: 9 };
          out.push(d);
        }
      } else if (level !== 1) {
        return { fds: out, err: 92 };
      }
      off += len + 7 & ~7;
    }
    return { fds: out, err: 0 };
  }
  function handleUnixSendmsg(cpu2, view) {
    const fd = Number(cpu2.getReg(0));
    const sock = unixSocks.get(fd);
    const hp = Number(cpu2.getReg(1));
    const flags = Number(BigInt.asIntN(32, cpu2.getReg(2)));
    const bytes = new Uint8Array(memory.buffer);
    if (hp === 0 || syscalls.ptrFaults(bytes, hp, 56)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
      return;
    }
    const h = readMsghdr(view, hp);
    if (h.iovLen > 1024) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-90)));
      return;
    }
    if (h.iovLen > 0 && (h.iov === 0 || syscalls.ptrFaults(bytes, h.iov, h.iovLen * 16))) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
      return;
    }
    for (let i = 0; i < h.iovLen; i++) {
      const base = Number(view.getBigUint64(h.iov + i * 16 + GUEST_MEM_BASE, true));
      const len = Number(view.getBigUint64(h.iov + i * 16 + 8 + GUEST_MEM_BASE, true));
      if (len > 0 && syscalls.ptrFaults(bytes, base, len)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
    }
    if (h.controlLen > 0 && (h.control === 0 || syscalls.ptrFaults(bytes, h.control, h.controlLen))) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
      return;
    }
    if (flags & ~(16384 | 64 | 32768 | 128 | 4)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-95)));
      return;
    }
    const msgDontwait = (flags & MSG_DONTWAIT) !== 0;
    let dest;
    if (sock.dgram && h.name !== 0) {
      const a = readSunAddr(view, h.name, h.nameLen);
      if (a === null || a.raw === "") {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
        return;
      }
      const ab = autobindIfUnnamed(sock);
      if (ab < 0) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(ab)));
        return;
      }
      dest = dgramRingOf(a.raw);
    } else {
      dest = dgramDefaultRing(sock);
    }
    if (dest < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(dest)));
      return;
    }
    const scm = h.controlLen > 0 ? collectScmRights(view, h.control, h.controlLen) : { fds: [], err: 0 };
    if (scm.err) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-scm.err)));
      return;
    }
    const payload = gatherIov(view, h.iov, h.iovLen);
    const tx = new PipeRing(shmArena, dest);
    const pos = tx.writePos;
    if (scm.fds.length) {
      handleBridgeCall("scm_send", { ring: dest, pos, fds: scm.fds }, () => {
      });
      tx.addScm(scm.fds.length);
    }
    const n = sock.framed ? unixDgramWrite(fd, dest, payload, sock.name, msgDontwait) : unixStreamWrite(fd, dest, payload, msgDontwait);
    if (n < 0) {
      if (scm.fds.length) {
        handleBridgeCall("scm_unsend", { ring: dest, pos, n: scm.fds.length }, () => {
        });
        tx.addScm(-scm.fds.length);
      }
      for (const d of scm.fds) releaseScmDescriptor(d);
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(n)));
      return;
    }
    cpu2.setReg(0, BigInt(n));
  }
  function releaseScmDescriptor(d) {
    if (d?.k === "native") handleBridgeCall("fs_close", { fd: d.nfd }, () => {
    });
    else if (d?.k === "socket") handleBridgeCall("net_close", { socketId: d.socketId }, () => {
    });
    else if (d?.ends) {
      for (const [off, mode] of d.ends) {
        const r = new PipeRing(shmArena, off);
        if (mode === "w") r.closeWriter();
        else r.closeReader();
      }
    }
  }
  function handleUnixRecvmsg(cpu2, view) {
    const fd = Number(cpu2.getReg(0));
    const sock = unixSocks.get(fd);
    const hp = Number(cpu2.getReg(1));
    const flags = Number(BigInt.asIntN(32, cpu2.getReg(2)));
    const bytes = new Uint8Array(memory.buffer);
    if (hp === 0 || syscalls.ptrFaults(bytes, hp, 56)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
      return;
    }
    const h = readMsghdr(view, hp);
    if (h.iovLen > 1024) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-90)));
      return;
    }
    if (h.iovLen > 0 && (h.iov === 0 || syscalls.ptrFaults(bytes, h.iov, h.iovLen * 16))) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
      return;
    }
    if (sock.rx === void 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-107)));
      return;
    }
    let cap = 0;
    for (let i = 0; i < h.iovLen; i++) cap += Number(view.getBigUint64(h.iov + i * 16 + 8 + GUEST_MEM_BASE, true));
    const rx = new PipeRing(shmArena, sock.rx);
    const msgDontwait = (flags & MSG_DONTWAIT) !== 0;
    const got = sock.framed ? unixDgramRead(fd, sock.rx, cap, msgDontwait) : unixStreamRead(fd, sock.rx, cap, msgDontwait);
    if (typeof got === "number") {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(got)));
      return;
    }
    const wrote = scatterIov(view, h.iov, h.iovLen, got.data);
    let msgFlags = got.truncated ? 32 : 0;
    let controlUsed = 0;
    const after = rx.readPos;
    let records = [];
    const claimTo = got.data.length > 0 ? after - 1 | 0 : after | 0;
    if (rx.scmPending > 0) {
      handleBridgeCall("scm_take", { ring: sock.rx, to: claimTo }, (resp) => {
        if (resp.ret > 0 && resp.bytes) {
          try {
            records = JSON.parse(new TextDecoder().decode(resp.bytes));
          } catch {
            records = [];
          }
        }
      });
      if (records.length) rx.addScm(-records.length);
    }
    if (records.length) {
      const room = h.controlLen >= 16 ? Math.floor((h.controlLen - 16) / 4) : 0;
      const give = Math.min(room, records.length);
      if (give < records.length) msgFlags |= 8;
      const newFds = [];
      for (let i = 0; i < give; i++) {
        const nfd = materializeScmFd(records[i]);
        if (nfd >= 0) {
          newFds.push(nfd);
          if (flags & 1073741824) syscalls.markCloexec(nfd);
        }
      }
      for (let i = give; i < records.length; i++) releaseScmDescriptor(records[i]);
      if (newFds.length) {
        const cb = h.control + GUEST_MEM_BASE;
        const len = 16 + newFds.length * 4;
        view.setBigUint64(cb, BigInt(len), true);
        view.setInt32(cb + 8, 1, true);
        view.setInt32(cb + 12, 1, true);
        for (let i = 0; i < newFds.length; i++) view.setInt32(cb + 16 + i * 4, newFds[i], true);
        controlUsed = len + 7 & ~7;
      }
    }
    let nameLen = 0;
    if (h.name !== 0 && sock.framed && got.from) {
      const buf = new Uint8Array(memory.buffer);
      const nm = got.from;
      const enc = new TextEncoder().encode(nm.charCodeAt(0) === 0 ? nm.slice(1) : nm);
      const body = new Uint8Array(nm.charCodeAt(0) === 0 ? 1 + enc.length : enc.length + 1);
      body.set(enc, nm.charCodeAt(0) === 0 ? 1 : 0);
      const full = 2 + body.length;
      const copy = Math.min(h.nameLen, full);
      if (copy > 0 && !syscalls.ptrFaults(buf, h.name, copy)) {
        const out = new Uint8Array(full);
        out[0] = 1;
        out[1] = 0;
        out.set(body, 2);
        buf.set(out.subarray(0, copy), h.name + GUEST_MEM_BASE);
        nameLen = full;
      }
    }
    view.setUint32(hp + 8 + GUEST_MEM_BASE, nameLen, true);
    view.setBigUint64(hp + 40 + GUEST_MEM_BASE, BigInt(controlUsed), true);
    view.setInt32(hp + 48 + GUEST_MEM_BASE, msgFlags, true);
    cpu2.setReg(0, BigInt(wrote));
  }
  function handleCopyFileRange(cpu2, view) {
    const err = (e) => cpu2.setReg(0, BigInt.asUintN(64, BigInt(-e)));
    const bytes = new Uint8Array(memory.buffer);
    const inFd = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    const offInPtr = Number(cpu2.getReg(1));
    const outFd = Number(BigInt.asIntN(32, cpu2.getReg(2)));
    const offOutPtr = Number(cpu2.getReg(3));
    const len = Number(BigInt.asIntN(64, cpu2.getReg(4)));
    const flags = Number(BigInt.asUintN(32, cpu2.getReg(5)));
    if (flags !== 0) {
      err(22);
      return;
    }
    if (len < 0) {
      err(22);
      return;
    }
    if (!syscalls.fdAccess(inFd).readable || !syscalls.fdAccess(outFd).writable) {
      err(9);
      return;
    }
    if (offInPtr !== 0 && syscalls.ptrWriteFaults(bytes, offInPtr, 8)) {
      err(14);
      return;
    }
    if (offOutPtr !== 0 && syscalls.ptrWriteFaults(bytes, offOutPtr, 8)) {
      err(14);
      return;
    }
    const inNative = syscalls.fdNative(inFd);
    const outNative = syscalls.fdNative(outFd);
    let inOff = offInPtr !== 0 ? Number(view.getBigInt64(offInPtr + GUEST_MEM_BASE, true)) : nativeFdOffset(inFd);
    let outOff = offOutPtr !== 0 ? Number(view.getBigInt64(offOutPtr + GUEST_MEM_BASE, true)) : nativeFdOffset(outFd);
    if (inOff < 0 || outOff < 0) {
      err(22);
      return;
    }
    if (inNative === outNative && inOff < outOff + len && outOff < inOff + len) {
      err(22);
      return;
    }
    let copied = 0;
    while (copied < len) {
      const chunk = readNativeFileAt(inNative, inOff + copied, Math.min(65536, len - copied));
      if (chunk.length === 0) break;
      let wrote = 0;
      handleBridgeCall("fs_pwrite", {
        fd: outNative,
        offset: outOff + copied,
        data: bytesToBase64Fast(chunk)
      }, (r) => {
        if (r && typeof r.ret === "number") wrote = r.ret;
      });
      if (wrote < 0) {
        if (copied === 0) {
          err(-wrote);
          return;
        }
        break;
      }
      copied += wrote;
      if (wrote < chunk.length) break;
    }
    if (offInPtr !== 0) view.setBigInt64(offInPtr + GUEST_MEM_BASE, BigInt(inOff + copied), true);
    else handleBridgeCall("fs_lseek", { fd: inNative, offset: inOff + copied, whence: 0 }, () => {
    });
    if (offOutPtr !== 0) view.setBigInt64(offOutPtr + GUEST_MEM_BASE, BigInt(outOff + copied), true);
    else handleBridgeCall("fs_lseek", { fd: outNative, offset: outOff + copied, whence: 0 }, () => {
    });
    if (copied > 0) syscalls.invalidateStatFd(outFd);
    cpu2.setReg(0, BigInt(copied));
  }
  function nativeFdOffset(fd) {
    const nfd = syscalls.fdNative(fd);
    if (nfd < 0) return 0;
    let off = 0;
    handleBridgeCall("fs_lseek", { fd: nfd, offset: 0, whence: 1 }, (r) => {
      if (r && typeof r.ret === "number" && r.ret >= 0) off = r.ret;
    });
    return off;
  }
  function nativeFdSize(fd, path) {
    let size = 0;
    handleBridgeCall("fs_stat", { path, creds: syscalls.getCreds() }, (r) => {
      if (r && r._meta && typeof r._meta[1] === "number") size = r._meta[1];
    });
    return size;
  }
  function syscallDone() {
    publishSyscall(-1);
    if (sysEnterMs >= 0) {
      sysAccumMs += Math.max(0, performance.now() - sysEnterMs - (parkedTotalMs() - sysEnterParkedMs));
      sysEnterMs = -1;
    }
    if (sharedMaps.length) reconcileShared();
  }
  function reconcileShared() {
    const guest = new Uint8Array(memory.buffer);
    const arena = new Uint8Array(shmArena);
    for (const m of sharedMaps) {
      const start = m.addr + GUEST_MEM_BASE;
      if (start + m.len > guest.length) continue;
      reconcile(guest.subarray(start, start + m.len), arena.subarray(m.arenaOff, m.arenaOff + m.len), m.shadow);
    }
  }
  function registerSharedMap(addr, len, key) {
    Atomics.store(myMailbox, MBOX_READY, 0);
    self.postMessage({ type: "shm_alloc", len, key });
    awaitMailbox("shm_map");
    const off = Atomics.load(myMailbox, MBOX_VALUE0);
    if (off < 0) return;
    const created = Atomics.load(myMailbox, MBOX_VALUE0 + 1) === 1;
    const guest = new Uint8Array(memory.buffer, addr + GUEST_MEM_BASE, len);
    const block = new Uint8Array(shmArena, off, len);
    if (created) {
      block.set(guest);
    } else {
      guest.set(block);
    }
    sharedMaps.push({ addr, len, arenaOff: off, shadow: new Uint8Array(block) });
  }
  function releaseSharedMaps(start, end) {
    const kept = [];
    for (const m of sharedMaps) {
      if (m.addr + m.len <= start || m.addr >= end) {
        kept.push(m);
        continue;
      }
      Atomics.store(myMailbox, MBOX_READY, 0);
      self.postMessage({ type: "shm_free", off: m.arenaOff });
      awaitMailbox("shm_release");
    }
    sharedMaps = kept;
  }
  var sysvAttached = /* @__PURE__ */ new Map();
  var currentArgv = [];
  function publishIdentity(argv) {
    currentArgv = argv;
    if (vfs) vfs.selfPid = myPid;
    const name = (argv[0] ?? "").split("/").pop() ?? "";
    const comm = new TextEncoder().encode(name).subarray(0, 15);
    const commBytes = new Uint8Array(16);
    commBytes.set(comm);
    new Uint8Array(kernelSAB, procSlotOffset(mySlot) + PROC_COMM * 4, 16).set(commBytes);
    const joined = argv.join("\0") + "\0";
    const raw = new TextEncoder().encode(joined).subarray(0, PROC_CMDLINE_SLOT_SIZE - 1);
    const slot = new Uint8Array(kernelSAB, procCmdlineOffset(mySlot), PROC_CMDLINE_SLOT_SIZE);
    slot.fill(0);
    slot.set(raw);
  }
  function procSnapshot() {
    const out = [];
    for (let s = 0; s < PROC_TABLE_SLOTS; s++) {
      const base = procSlotOffset(s) / 4;
      const state = Atomics.load(procTable, base + PROC_STATE);
      if (state === PROC_EMPTY) continue;
      const pid = Atomics.load(procTable, base + PROC_PID);
      if (pid <= 0) continue;
      const commRaw = new Uint8Array(kernelSAB, procSlotOffset(s) + PROC_COMM * 4, 16);
      let n = commRaw.indexOf(0);
      if (n < 0) n = 16;
      const cmdRaw = new Uint8Array(kernelSAB, procCmdlineOffset(s), PROC_CMDLINE_SLOT_SIZE);
      let cn = cmdRaw.length;
      while (cn > 0 && cmdRaw[cn - 1] === 0) cn--;
      out.push({
        pid,
        ppid: Atomics.load(procTable, base + PROC_PPID),
        pgid: Atomics.load(procTable, base + PROC_PGID),
        sid: Atomics.load(procTable, base + PROC_SID),
        tgid: Atomics.load(procTable, base + PROC_TGID),
        state,
        blocked: Atomics.load(procTable, base + PROC_BLOCKED) !== 0,
        utimeMs: Atomics.load(procTable, base + PROC_UTIME_MS),
        stimeMs: Atomics.load(procTable, base + PROC_STIME_MS),
        nice: Atomics.load(procTable, base + PROC_NICE),
        syscall: Atomics.load(procTable, base + PROC_SYSCALL),
        comm: new TextDecoder().decode(commRaw.subarray(0, n)),
        cmdline: new TextDecoder().decode(cmdRaw.subarray(0, cn))
      });
    }
    return out;
  }
  var usedLocks = false;
  function releaseLocksForClosingFd(fd) {
    const path = syscalls.fdPath(fd);
    if (path === null) return;
    const ofdId = syscalls.fdDescriptionId(fd);
    Atomics.store(myMailbox, MBOX_READY, 0);
    self.postMessage({
      type: "lock_close",
      path,
      ofdId,
      lastRef: syscalls.countFdsForDescription(ofdId) <= 1
    });
    awaitMailbox("lock_close");
  }
  function handleFlock(cpu2) {
    const fd = Number(cpu2.getReg(0));
    const op = Number(cpu2.getReg(1)) | 0;
    const path = syscalls.fdPath(fd);
    if (path === null) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-9)));
      return;
    }
    const nonblock = (op & 4) !== 0;
    const base = op & ~4;
    let ltype;
    if (base === 1) ltype = 0;
    else if (base === 2) ltype = 1;
    else if (base === 8) ltype = 2;
    else {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    const cmd = ltype === 2 ? 6 : nonblock ? 6 : 7;
    usedLocks = true;
    Atomics.store(myMailbox, MBOX_READY, 0);
    self.postMessage({
      type: "lock_req",
      space: "flock",
      scope: "ofd",
      ownerId: syscalls.fdDescriptionId(fd),
      path,
      cmd,
      ltype,
      start: 0,
      len: 0
    });
    const r = waitLockReply();
    cpu2.setReg(0, r < 0 ? BigInt.asUintN(64, BigInt(r)) : 0n);
  }
  function handleKernelPipeDup(cpu2, minFd, cloexec) {
    const oldFd = Number(cpu2.getReg(0));
    const info = kernelPipes.get(oldFd);
    const newFd = syscalls.allocLowestFd(minFd);
    if (newFd < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-24)));
      return;
    }
    kernelPipes.set(newFd, { ...info });
    pipeRef(info.pipeOff, info.mode, 1);
    syscalls.registerFd(newFd, {
      type: info.mode === "read" ? "pipe_read" : "pipe_write",
      pipe: { data: [], writeClosed: false, readClosed: false }
    }, syscalls.statusFlags(oldFd));
    if (cloexec) syscalls.markCloexec(newFd);
    cpu2.setReg(0, BigInt(newFd));
  }
  function handleKernelPipeDup3(cpu2) {
    const oldFd = Number(cpu2.getReg(0));
    const newFd = Number(cpu2.getReg(1));
    if (Number(BigInt.asIntN(32, cpu2.getReg(2))) & ~524288) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    if (oldFd === newFd) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    const pipeInfo = kernelPipes.get(oldFd);
    const existing = kernelPipes.get(newFd);
    if (existing) {
      pipeRef(existing.pipeOff, existing.mode, -1);
      kernelPipes.delete(newFd);
    }
    kernelPipes.set(newFd, { ...pipeInfo });
    pipeRef(pipeInfo.pipeOff, pipeInfo.mode, 1);
    syscalls.registerFd(newFd, {
      type: pipeInfo.mode === "read" ? "pipe_read" : "pipe_write",
      pipe: { data: [], writeClosed: false, readClosed: false }
    }, syscalls.statusFlags(oldFd));
    if (Number(BigInt.asIntN(32, cpu2.getReg(2))) & 524288) syscalls.markCloexec(newFd);
    else syscalls.clearCloexec(newFd);
    cpu2.setReg(0, BigInt(newFd));
  }
  function allocPipeSlot() {
    Atomics.store(myMailbox, MBOX_READY, 0);
    self.postMessage({ type: "pipe_alloc", priv: syscalls.hasCap(24) });
    awaitMailbox("pipe_alloc");
    return Atomics.load(myMailbox, MBOX_VALUE0);
  }
  function handlePipe2(cpu2, view) {
    const pipefdAddr = Number(cpu2.getReg(0));
    if (syscalls.ptrFaults(new Uint8Array(memory.buffer), pipefdAddr, 8)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
      return;
    }
    const O_DIRECT = 65536, O_NONBLOCK3 = 2048, O_CLOEXEC = 524288;
    const p2Flags = Number(BigInt.asIntN(32, cpu2.getReg(1)));
    if (p2Flags & ~(O_DIRECT | O_NONBLOCK3 | O_CLOEXEC)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    if (!syscalls.fdRoomFor(2)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-24)));
      return;
    }
    const pipeOff = allocPipeSlot();
    if (pipeOff < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-23)));
      return;
    }
    const ring = new PipeRing(shmArena, pipeOff);
    ring.reset();
    const readFd = syscalls.allocLowestFd();
    if (readFd < 0) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-24)));
      return;
    }
    const pipeStatus = p2Flags & (O_NONBLOCK3 | O_DIRECT);
    syscalls.registerFd(readFd, { type: "pipe_read", pipe: { data: [], writeClosed: false, readClosed: false } }, pipeStatus);
    const writeFd = syscalls.allocLowestFd();
    if (writeFd < 0) {
      syscalls.unregisterFd(readFd);
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-24)));
      return;
    }
    syscalls.registerFd(writeFd, { type: "pipe_write", pipe: { data: [], writeClosed: false, readClosed: false } }, pipeStatus);
    if (p2Flags & O_CLOEXEC) {
      syscalls.markCloexec(readFd);
      syscalls.markCloexec(writeFd);
    }
    kernelPipes.set(readFd, { pipeOff, mode: "read" });
    kernelPipes.set(writeFd, { pipeOff, mode: "write" });
    view.setInt32(pipefdAddr + GUEST_MEM_BASE, readFd, true);
    view.setInt32(pipefdAddr + GUEST_MEM_BASE + 4, writeFd, true);
    cpu2.setReg(0, 0n);
  }
  function notePipeReadable(off) {
    const hdr = new Int32Array(shmArena, off, PIPE_HEADER_SIZE / 4);
    if (Atomics.load(hdr, PIPE_ASYNC_W) === 0) return;
    self.postMessage({ type: "pipe_io", off, mode: "read" });
  }
  function pipeRef(off, mode, delta) {
    const ring = new PipeRing(shmArena, off);
    if (delta > 0) {
      if (mode === "write") ring.addWriter();
      else ring.addReader();
    } else {
      if (mode === "write") ring.closeWriter();
      else ring.closeReader();
    }
    self.postMessage({ type: "pipe_ref", off, mode, delta });
  }
  var PTRACE_TRACEME = 0;
  var PTRACE_PEEKTEXT = 1;
  var PTRACE_PEEKDATA = 2;
  var PTRACE_PEEKUSER = 3;
  var PTRACE_POKETEXT = 4;
  var PTRACE_POKEDATA = 5;
  var PTRACE_POKEUSER = 6;
  var PTRACE_CONT = 7;
  var PTRACE_KILL = 8;
  var PTRACE_ATTACH = 16;
  var PTRACE_DETACH = 17;
  var PTRACE_SETOPTIONS = 16896;
  var PTRACE_GETREGSET = 16900;
  var PTRACE_SETREGSET = 16901;
  var PT_OP_PEEK = 1;
  var PT_OP_POKE = 2;
  var PT_OP_GETREGS = 3;
  var PT_OP_SETREGS = 4;
  var PTRACE_REGS_BYTES = 34 * 8;
  function traceSlotOfPid(pid) {
    for (let s = 0; s < PROC_TABLE_SLOTS; s++) {
      const b = procSlotOffset(s) / 4;
      if (Atomics.load(procTable, b + PROC_PID) === pid && Atomics.load(procTable, b + PROC_STATE) !== PROC_EMPTY) return s;
    }
    return -1;
  }
  function ptraceAsk(slot, op, addr, len, data) {
    const w = ptraceSlotWord(slot);
    const seq = Atomics.load(procTable, w + PT_REQ) + 1 | 0;
    Atomics.store(procTable, w + PT_OP, op);
    Atomics.store(procTable, w + PT_ADDR, addr | 0);
    Atomics.store(procTable, w + PT_LEN, len | 0);
    Atomics.store(procTable, w + PT_DATA_LO, Number(data & 0xffffffffn) | 0);
    Atomics.store(procTable, w + PT_DATA_HI, Number(data >> 32n & 0xffffffffn) | 0);
    Atomics.store(procTable, w + PT_STATUS, 0);
    Atomics.store(procTable, w + PT_REQ, seq);
    const tb = procSlotOffset(slot) / 4;
    Atomics.store(procTable, tb + PROC_WAIT_FUTEX, 1);
    Atomics.notify(procTable, tb + PROC_WAIT_FUTEX);
    const deadline = performance.now() + 2e3;
    while (Atomics.load(procTable, w + PT_ACK) !== seq) {
      if (performance.now() > deadline) return { status: -3, data: 0n };
      if (Atomics.load(procTable, tb + PROC_PID) <= 0) return { status: -3, data: 0n };
      sleepingWait(procTable, w + PT_ACK, Atomics.load(procTable, w + PT_ACK), 5);
    }
    const lo = BigInt(Atomics.load(procTable, w + PT_DATA_LO) >>> 0);
    const hi = BigInt(Atomics.load(procTable, w + PT_DATA_HI) >>> 0);
    return { status: Atomics.load(procTable, w + PT_STATUS), data: hi << 32n | lo };
  }
  function ptraceServe() {
    const w = ptraceSlotWord(mySlot);
    const seq = Atomics.load(procTable, w + PT_REQ);
    if (seq === Atomics.load(procTable, w + PT_ACK)) return false;
    const op = Atomics.load(procTable, w + PT_OP);
    const addr = Atomics.load(procTable, w + PT_ADDR) >>> 0;
    const len = Atomics.load(procTable, w + PT_LEN);
    const bytes = new Uint8Array(memory.buffer);
    const view = new DataView(memory.buffer);
    let status = 0;
    let out = 0n;
    try {
      if (op === PT_OP_PEEK) {
        if (syscalls.ptrFaults(bytes, addr, 8)) status = -5;
        else out = view.getBigUint64(addr + GUEST_MEM_BASE, true);
      } else if (op === PT_OP_POKE) {
        const lo = BigInt(Atomics.load(procTable, w + PT_DATA_LO) >>> 0);
        const hi = BigInt(Atomics.load(procTable, w + PT_DATA_HI) >>> 0);
        if (syscalls.ptrWriteFaults(bytes, addr, 8)) status = -5;
        else view.setBigUint64(addr + GUEST_MEM_BASE, hi << 32n | lo, true);
      } else if (op === PT_OP_GETREGS) {
        const sc = ptraceScratchOffset(mySlot);
        const sab = new DataView(procTable.buffer);
        for (let i = 0; i < 31; i++) sab.setBigUint64(sc + i * 8, cpu.getReg(i), true);
        sab.setBigUint64(sc + 31 * 8, cpu.getSP(), true);
        sab.setBigUint64(sc + 32 * 8, cpu.getPC(), true);
        sab.setBigUint64(sc + 33 * 8, cpu.getNZCV(), true);
        out = BigInt(PTRACE_REGS_BYTES);
      } else if (op === PT_OP_SETREGS) {
        const sc = ptraceScratchOffset(mySlot);
        const sab = new DataView(procTable.buffer);
        const n = Math.min(len, PTRACE_REGS_BYTES);
        for (let i = 0; i < 31 && (i + 1) * 8 <= n; i++) cpu.setReg(i, sab.getBigUint64(sc + i * 8, true));
        if (n >= 32 * 8) cpu.setSP(sab.getBigUint64(sc + 31 * 8, true));
        if (n >= 33 * 8) cpu.setPC(sab.getBigUint64(sc + 32 * 8, true));
        out = BigInt(n);
      } else {
        status = -22;
      }
    } catch {
      status = -5;
    }
    Atomics.store(procTable, w + PT_DATA_LO, Number(out & 0xffffffffn) | 0);
    Atomics.store(procTable, w + PT_DATA_HI, Number(out >> 32n & 0xffffffffn) | 0);
    Atomics.store(procTable, w + PT_STATUS, status);
    Atomics.store(procTable, w + PT_ACK, seq);
    Atomics.notify(procTable, w + PT_ACK);
    return true;
  }
  function ptraceStop(sig) {
    const base = procSlotOffset(mySlot) / 4;
    Atomics.store(procTable, base + PROC_TRACE_CMD, PT_RESUME_NONE);
    Atomics.store(procTable, base + PROC_STOP_SIG, sig);
    Atomics.store(procTable, base + PROC_STOP_SEEN, 0);
    Atomics.store(procTable, base + PROC_STATE, PROC_STOPPED);
    wakeWatcher(Atomics.load(procTable, base + PROC_TRACER));
    for (; ; ) {
      while (ptraceServe()) {
      }
      const cmd = Atomics.load(procTable, base + PROC_TRACE_CMD);
      if (cmd !== PT_RESUME_NONE) break;
      if (Atomics.load(procTable, base + PROC_PID) <= 0) return 0;
      Atomics.store(procTable, base + PROC_WAIT_FUTEX, 0);
      if (Atomics.load(procTable, base + PROC_TRACE_CMD) !== PT_RESUME_NONE) break;
      sleepingWait(procTable, base + PROC_WAIT_FUTEX, 0, 200);
    }
    Atomics.store(procTable, base + PROC_TRACE_CMD, PT_RESUME_NONE);
    Atomics.store(procTable, base + PROC_STATE, PROC_RUNNING);
    const inject = Atomics.load(procTable, base + PROC_STOP_SIG);
    Atomics.store(procTable, base + PROC_STOP_SIG, 0);
    return inject;
  }
  function handlePtrace(cpu2, view) {
    const req = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    const pid = Number(BigInt.asIntN(32, cpu2.getReg(1)));
    const addr = Number(BigInt.asUintN(64, cpu2.getReg(2)));
    const data = cpu2.getReg(3);
    const err = (e) => cpu2.setReg(0, BigInt.asUintN(64, BigInt(e)));
    const myBase = procSlotOffset(mySlot) / 4;
    if (req === PTRACE_TRACEME) {
      if (Atomics.load(procTable, myBase + PROC_TRACER) !== 0) {
        err(-1);
        return;
      }
      Atomics.store(procTable, myBase + PROC_TRACER, Atomics.load(procTable, myBase + PROC_PPID));
      cpu2.setReg(0, 0n);
      return;
    }
    const slot = traceSlotOfPid(pid);
    if (slot < 0) {
      err(-3);
      return;
    }
    const base = procSlotOffset(slot) / 4;
    if (req === PTRACE_ATTACH) {
      if (pid === myPid) {
        err(-1);
        return;
      }
      if (Atomics.load(procTable, base + PROC_TRACER) !== 0) {
        err(-1);
        return;
      }
      const meCred = procCredOffset(mySlot) / 4;
      const myEuid2 = Atomics.load(procCreds, meCred + 1);
      if (myEuid2 !== 0) {
        const tb = procCredOffset(slot) / 4;
        const myRuid2 = Atomics.load(procCreds, meCred);
        const tRuid = Atomics.load(procCreds, tb);
        const tEuid = Atomics.load(procCreds, tb + 1);
        const tSuid = Atomics.load(procCreds, tb + 2);
        if (myRuid2 !== tRuid || myRuid2 !== tEuid || myRuid2 !== tSuid) {
          err(-1);
          return;
        }
      }
      Atomics.store(procTable, base + PROC_TRACER, myPid);
      sigqPush(sigQueue, slot, { signo: 19, code: 0, pid: myPid, uid: syscalls.currentEuid(), value: 0n });
      Atomics.store(procTable, base + PROC_WAIT_FUTEX, 1);
      Atomics.notify(procTable, base + PROC_WAIT_FUTEX);
      bumpIoGeneration();
      cpu2.setReg(0, 0n);
      return;
    }
    if (Atomics.load(procTable, base + PROC_TRACER) !== myPid) {
      err(-3);
      return;
    }
    if (req === PTRACE_KILL) {
      self.postMessage({ type: "kill", pid, sig: 9 });
      cpu2.setReg(0, 0n);
      return;
    }
    if (req === PTRACE_SETOPTIONS) {
      cpu2.setReg(0, 0n);
      return;
    }
    if (Atomics.load(procTable, base + PROC_STATE) !== PROC_STOPPED) {
      err(-3);
      return;
    }
    switch (req) {
      case PTRACE_CONT:
      case PTRACE_DETACH: {
        const sig = Number(BigInt.asIntN(32, data));
        if (sig < 0 || sig > 64) {
          err(-22);
          return;
        }
        if (req === PTRACE_DETACH) Atomics.store(procTable, base + PROC_TRACER, 0);
        Atomics.store(procTable, base + PROC_STOP_SIG, sig);
        Atomics.store(
          procTable,
          base + PROC_TRACE_CMD,
          req === PTRACE_DETACH ? PT_RESUME_DETACH : PT_RESUME_CONT
        );
        Atomics.store(procTable, base + PROC_WAIT_FUTEX, 1);
        Atomics.notify(procTable, base + PROC_WAIT_FUTEX);
        cpu2.setReg(0, 0n);
        return;
      }
      case PTRACE_PEEKTEXT:
      case PTRACE_PEEKDATA: {
        const r = ptraceAsk(slot, PT_OP_PEEK, addr, 8, 0n);
        if (r.status < 0) {
          err(r.status);
          return;
        }
        const out = Number(data);
        if (out === 0 || syscalls.ptrFaults(new Uint8Array(memory.buffer), out, 8)) {
          err(-14);
          return;
        }
        view.setBigUint64(out + GUEST_MEM_BASE, r.data, true);
        cpu2.setReg(0, 0n);
        return;
      }
      case PTRACE_POKETEXT:
      case PTRACE_POKEDATA: {
        const r = ptraceAsk(slot, PT_OP_POKE, addr, 8, BigInt.asUintN(64, data));
        if (r.status < 0) {
          err(r.status);
          return;
        }
        cpu2.setReg(0, 0n);
        return;
      }
      case PTRACE_PEEKUSER:
      case PTRACE_POKEUSER: {
        if (addr % 8 !== 0 || addr >= PTRACE_REGS_BYTES) {
          err(-5);
          return;
        }
        const g = ptraceAsk(slot, PT_OP_GETREGS, 0, PTRACE_REGS_BYTES, 0n);
        if (g.status < 0) {
          err(g.status);
          return;
        }
        const sc = ptraceScratchOffset(slot);
        const sab = new DataView(procTable.buffer);
        if (req === PTRACE_PEEKUSER) {
          const word = sab.getBigUint64(sc + addr, true);
          const out = Number(data);
          if (out === 0 || syscalls.ptrFaults(new Uint8Array(memory.buffer), out, 8)) {
            err(-14);
            return;
          }
          view.setBigUint64(out + GUEST_MEM_BASE, word, true);
        } else {
          sab.setBigUint64(sc + addr, BigInt.asUintN(64, data), true);
          const r = ptraceAsk(slot, PT_OP_SETREGS, 0, PTRACE_REGS_BYTES, 0n);
          if (r.status < 0) {
            err(r.status);
            return;
          }
        }
        cpu2.setReg(0, 0n);
        return;
      }
      case PTRACE_GETREGSET:
      case PTRACE_SETREGSET: {
        const NT_PRSTATUS = 1;
        if (addr !== NT_PRSTATUS) {
          err(-22);
          return;
        }
        const iov = Number(data);
        const bytes = new Uint8Array(memory.buffer);
        if (iov === 0 || syscalls.ptrFaults(bytes, iov, 16)) {
          err(-14);
          return;
        }
        const bufPtr = Number(view.getBigUint64(iov + GUEST_MEM_BASE, true));
        const bufLen = Number(view.getBigUint64(iov + GUEST_MEM_BASE + 8, true));
        const n = Math.min(bufLen, PTRACE_REGS_BYTES);
        if (n <= 0) {
          err(-22);
          return;
        }
        const sc = ptraceScratchOffset(slot);
        const sabBytes = new Uint8Array(procTable.buffer);
        if (req === PTRACE_GETREGSET) {
          const g = ptraceAsk(slot, PT_OP_GETREGS, 0, PTRACE_REGS_BYTES, 0n);
          if (g.status < 0) {
            err(g.status);
            return;
          }
          if (syscalls.ptrWriteFaults(bytes, bufPtr, n)) {
            err(-14);
            return;
          }
          bytes.set(sabBytes.subarray(sc, sc + n), bufPtr + GUEST_MEM_BASE);
        } else {
          if (syscalls.ptrFaults(bytes, bufPtr, n)) {
            err(-14);
            return;
          }
          sabBytes.set(bytes.subarray(bufPtr + GUEST_MEM_BASE, bufPtr + GUEST_MEM_BASE + n), sc);
          const r = ptraceAsk(slot, PT_OP_SETREGS, 0, n, 0n);
          if (r.status < 0) {
            err(r.status);
            return;
          }
        }
        view.setBigUint64(iov + GUEST_MEM_BASE + 8, BigInt(n), true);
        cpu2.setReg(0, 0n);
        return;
      }
      default:
        err(-5);
    }
  }
  function handleSigtimedwait(cpu2, view) {
    const setPtr = Number(cpu2.getReg(0));
    const infoPtr = Number(cpu2.getReg(1));
    const tsPtr = Number(cpu2.getReg(2));
    const bytes = new Uint8Array(memory.buffer);
    if (Number(BigInt.asIntN(64, cpu2.getReg(3))) !== 8) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
      return;
    }
    if (setPtr === 0 || syscalls.ptrFaults(bytes, setPtr, 8)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
      return;
    }
    const want = view.getBigUint64(setPtr + GUEST_MEM_BASE, true) & ~(1n << 8n | 1n << 18n);
    let deadline = Infinity;
    if (tsPtr !== 0) {
      if (syscalls.ptrFaults(bytes, tsPtr, 16)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
      const sec = Number(view.getBigInt64(tsPtr + GUEST_MEM_BASE, true));
      const nsec = Number(view.getBigInt64(tsPtr + GUEST_MEM_BASE + 8, true));
      if (sec < 0 || nsec < 0 || nsec >= 1e9) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
        return;
      }
      deadline = performance.now() + sec * 1e3 + nsec / 1e6;
    }
    const slotBase = procSlotOffset(mySlot) / 4;
    for (; ; ) {
      pollAndQueueSignals();
      const got = syscalls.takePendingIn(want);
      if (got) {
        if (infoPtr !== 0 && syscalls.ptrWriteFaults(bytes, infoPtr, 128)) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
          return;
        }
        if (infoPtr !== 0) {
          const base = infoPtr + GUEST_MEM_BASE;
          bytes.fill(0, base, base + 128);
          view.setInt32(base + 0, got.sig, true);
          view.setInt32(base + 8, got.pid > 0 ? 0 : 128, true);
          if (got.pid > 0) {
            view.setInt32(base + 16, got.pid, true);
            view.setInt32(base + 20, got.uid, true);
          }
        }
        cpu2.setReg(0, BigInt(got.sig));
        return;
      }
      if (syscalls.hasPendingDeliverable()) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-4)));
        return;
      }
      const rem = deadline - performance.now();
      if (rem <= 0) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-11)));
        return;
      }
      Atomics.store(procTable, slotBase + PROC_WAIT_FUTEX, 0);
      sleepingWait(procTable, slotBase + PROC_WAIT_FUTEX, 0, Math.min(rem, 50));
    }
  }
  var SCHED_OTHER = 0;
  var SCHED_FIFO = 1;
  var SCHED_RR = 2;
  var SCHED_BATCH = 3;
  var SCHED_IDLE = 5;
  var SCHED_DEADLINE = 6;
  var SCHED_RESET_ON_FORK = 1073741824;
  var SCHED_ATTR_SIZE_VER0 = 48;
  var GUEST_CPUS = guestCpus();
  var CPUMASK_BYTES = Math.ceil(GUEST_CPUS / 64) * 8;
  var RR_TIMESLICE_MS = 100;
  var FAIR_SLICE_MS = 3;
  function isRtPolicy(p) {
    return p === SCHED_FIFO || p === SCHED_RR;
  }
  function validPolicy(p) {
    return p === SCHED_OTHER || p === SCHED_FIFO || p === SCHED_RR || p === SCHED_BATCH || p === SCHED_IDLE || p === SCHED_DEADLINE;
  }
  function ioprioTargets(which, who) {
    if (which === 1) {
      const s = schedSlotOf(who);
      return s < 0 ? [] : [s];
    }
    const meBase = procSlotOffset(mySlot) / 4;
    const want = who !== 0 ? who : Atomics.load(procTable, meBase + (which === 2 ? PROC_PGID : PROC_RUID));
    const out = [];
    for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
      const base = procSlotOffset(slot) / 4;
      if (Atomics.load(procTable, base + PROC_STATE) === PROC_EMPTY) continue;
      if (Atomics.load(procTable, base + (which === 2 ? PROC_PGID : PROC_RUID)) !== want) continue;
      out.push(slot);
    }
    return out;
  }
  function schedSlotOf(pid) {
    if (pid === 0) return mySlot;
    for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
      const base = procSlotOffset(slot) / 4;
      if (Atomics.load(procTable, base + PROC_PID) !== pid) continue;
      if (Atomics.load(procTable, base + PROC_STATE) === PROC_EMPTY) continue;
      return slot;
    }
    return -1;
  }
  function schedSameOwner(slot) {
    const myBase = procCredOffset(mySlot) / 4;
    const myEuid = Atomics.load(procCreds, myBase + 1);
    if (myEuid === 0) return true;
    const tb = procCredOffset(slot) / 4;
    return myEuid === Atomics.load(procCreds, tb + 1) || myEuid === Atomics.load(procCreds, tb);
  }
  function schedPrivileged() {
    return Atomics.load(procCreds, procCredOffset(mySlot) / 4 + 1) === 0;
  }
  function schedRead(slot, field) {
    return Atomics.load(procTable, procSlotOffset(slot) / 4 + field);
  }
  function schedWrite(slot, field, value) {
    Atomics.store(procTable, procSlotOffset(slot) / 4 + field, value | 0);
  }
  function schedRead64(slot, field) {
    const base = procSlotOffset(slot) / 4;
    const lo = BigInt(Atomics.load(procTable, base + field) >>> 0);
    const hi = BigInt(Atomics.load(procTable, base + field + 1) >>> 0);
    return hi << 32n | lo;
  }
  function schedWrite64(slot, field, v) {
    const base = procSlotOffset(slot) / 4;
    Atomics.store(procTable, base + field, Number(v & 0xffffffffn) | 0);
    Atomics.store(procTable, base + field + 1, Number(v >> 32n & 0xffffffffn) | 0);
  }
  function schedAffinity(slot) {
    const base = procSlotOffset(slot) / 4;
    const lo = BigInt(Atomics.load(procTable, base + PROC_AFFINITY) >>> 0);
    const hi = BigInt(Atomics.load(procTable, base + PROC_AFFINITY + 1) >>> 0);
    const m = hi << 32n | lo;
    return m === 0n ? GUEST_CPUS >= 64 ? ~0n & (1n << 64n) - 1n : (1n << BigInt(GUEST_CPUS)) - 1n : m;
  }
  function handleSched(nr, cpu2, view) {
    const bytes = new Uint8Array(memory.buffer);
    const err = (e) => cpu2.setReg(0, BigInt.asUintN(64, BigInt(-e)));
    const EPERM3 = 1, ESRCH2 = 3, EFAULT2 = 14, EINVAL3 = 22, E2BIG2 = 7;
    if (nr === 168) {
      const cpuPtr = Number(cpu2.getReg(0));
      const nodePtr = Number(cpu2.getReg(1));
      const bytes168 = new Uint8Array(memory.buffer);
      if (cpuPtr !== 0 && syscalls.ptrWriteFaults(bytes168, cpuPtr, 4)) {
        err(EFAULT2);
        return;
      }
      if (nodePtr !== 0 && syscalls.ptrWriteFaults(bytes168, nodePtr, 4)) {
        err(EFAULT2);
        return;
      }
      const mask = schedAffinity(mySlot);
      let which = 0;
      while (which < 64 && (mask & 1n << BigInt(which)) === 0n) which++;
      if (which >= 64) which = 0;
      if (cpuPtr !== 0) view.setUint32(cpuPtr + GUEST_MEM_BASE, which, true);
      if (nodePtr !== 0) view.setUint32(nodePtr + GUEST_MEM_BASE, 0, true);
      cpu2.setReg(0, 0n);
      return;
    }
    if (nr === 30 || nr === 31) {
      const which = Number(BigInt.asIntN(32, cpu2.getReg(0)));
      const who = Number(BigInt.asIntN(32, cpu2.getReg(1)));
      if (which < 1 || which > 3) {
        err(EINVAL3);
        return;
      }
      const slots = ioprioTargets(which, who);
      if (slots.length === 0) {
        err(ESRCH2);
        return;
      }
      if (nr === 31) {
        let best = -1;
        for (const s of slots) {
          const v = schedRead(s, PROC_IOPRIO);
          if (best < 0 || v < best) best = v;
        }
        cpu2.setReg(0, BigInt(best < 0 ? 0 : best));
        return;
      }
      const value = Number(BigInt.asIntN(32, cpu2.getReg(2)));
      const cls = value >> 13 & 7;
      const level = value & 8191;
      if (cls === 1 && !syscalls.hasCap(
        21
        /* CAP_SYS_ADMIN */
      )) {
        err(EPERM3);
        return;
      }
      if (cls === 1 || cls === 2) {
        if (level >= 8) {
          err(EINVAL3);
          return;
        }
      } else if (cls === 0) {
        if (level !== 0) {
          err(EINVAL3);
          return;
        }
      } else if (cls !== 3) {
        err(EINVAL3);
        return;
      }
      for (const s of slots) schedWrite(s, PROC_IOPRIO, value);
      cpu2.setReg(0, 0n);
      return;
    }
    if (nr === 125 || nr === 126) {
      const policy = Number(BigInt.asIntN(32, cpu2.getReg(0)));
      if (!validPolicy(policy)) {
        err(EINVAL3);
        return;
      }
      if (!isRtPolicy(policy)) {
        cpu2.setReg(0, 0n);
        return;
      }
      cpu2.setReg(0, nr === 125 ? 99n : 1n);
      return;
    }
    if (nr === 120) {
      const pid2 = Number(BigInt.asIntN(32, cpu2.getReg(0)));
      if (pid2 < 0) {
        err(EINVAL3);
        return;
      }
      const slot2 = schedSlotOf(pid2);
      if (slot2 < 0) {
        err(ESRCH2);
        return;
      }
      const reset = schedRead(slot2, PROC_SCHED_FLAGS) & 1 ? SCHED_RESET_ON_FORK : 0;
      cpu2.setReg(0, BigInt(schedRead(slot2, PROC_POLICY) | reset));
      return;
    }
    if (nr === 121) {
      const pid2 = Number(BigInt.asIntN(32, cpu2.getReg(0)));
      const ptr2 = Number(cpu2.getReg(1));
      if (pid2 < 0 || ptr2 === 0) {
        err(EINVAL3);
        return;
      }
      const slot2 = schedSlotOf(pid2);
      if (slot2 < 0) {
        err(ESRCH2);
        return;
      }
      if (syscalls.ptrFaults(bytes, ptr2, 4)) {
        err(EFAULT2);
        return;
      }
      view.setInt32(ptr2 + GUEST_MEM_BASE, schedRead(slot2, PROC_RTPRIO), true);
      cpu2.setReg(0, 0n);
      return;
    }
    if (nr === 118 || nr === 119) {
      const pid2 = Number(BigInt.asIntN(32, cpu2.getReg(0)));
      const ptr2 = Number(nr === 118 ? cpu2.getReg(1) : cpu2.getReg(2));
      if (pid2 < 0 || ptr2 === 0) {
        err(EINVAL3);
        return;
      }
      if (syscalls.ptrFaults(bytes, ptr2, 4)) {
        err(EFAULT2);
        return;
      }
      const prio = view.getInt32(ptr2 + GUEST_MEM_BASE, true);
      const slot2 = schedSlotOf(pid2);
      if (slot2 < 0) {
        err(ESRCH2);
        return;
      }
      let policy, resetOnFork;
      if (nr === 119) {
        const raw = Number(BigInt.asIntN(32, cpu2.getReg(1)));
        resetOnFork = raw & SCHED_RESET_ON_FORK ? 1 : 0;
        policy = raw & ~SCHED_RESET_ON_FORK;
        if (!validPolicy(policy)) {
          err(EINVAL3);
          return;
        }
        if (policy === SCHED_DEADLINE) {
          err(EINVAL3);
          return;
        }
      } else {
        policy = schedRead(slot2, PROC_POLICY);
        resetOnFork = schedRead(slot2, PROC_SCHED_FLAGS) & 1;
        if (policy === SCHED_DEADLINE) {
          err(EINVAL3);
          return;
        }
      }
      if (isRtPolicy(policy) ? prio < 1 || prio > 99 : prio !== 0) {
        err(EINVAL3);
        return;
      }
      if (!schedSameOwner(slot2)) {
        err(EPERM3);
        return;
      }
      if (isRtPolicy(policy) && !schedPrivileged()) {
        err(EPERM3);
        return;
      }
      schedWrite(slot2, PROC_POLICY, policy);
      schedWrite(slot2, PROC_RTPRIO, prio);
      schedWrite(slot2, PROC_SCHED_FLAGS, resetOnFork);
      if (policy !== SCHED_DEADLINE) {
        schedWrite64(slot2, PROC_DL_RUNTIME, 0n);
        schedWrite64(slot2, PROC_DL_DEADLINE, 0n);
        schedWrite64(slot2, PROC_DL_PERIOD, 0n);
      }
      cpu2.setReg(0, 0n);
      return;
    }
    if (nr === 122 || nr === 123) {
      const pid2 = Number(BigInt.asIntN(32, cpu2.getReg(0)));
      const len = Number(cpu2.getReg(1));
      const ptr2 = Number(cpu2.getReg(2));
      if (pid2 < 0) {
        err(EINVAL3);
        return;
      }
      if (len % 8 !== 0) {
        err(EINVAL3);
        return;
      }
      if (nr === 123 && len * 8 < GUEST_CPUS) {
        err(EINVAL3);
        return;
      }
      if (nr === 123 && len === 0) {
        err(EINVAL3);
        return;
      }
      if (nr === 122) {
        const take = Math.min(len, CPUMASK_BYTES);
        if (syscalls.ptrFaults(bytes, ptr2, take)) {
          err(EFAULT2);
          return;
        }
        let mask2 = 0n;
        for (let i = 0; i < take; i++) mask2 |= BigInt(bytes[ptr2 + GUEST_MEM_BASE + i]) << BigInt(i * 8);
        const slot3 = schedSlotOf(pid2);
        if (slot3 < 0) {
          err(ESRCH2);
          return;
        }
        if (!schedSameOwner(slot3)) {
          err(EPERM3);
          return;
        }
        const allowed = GUEST_CPUS >= 64 ? (1n << 64n) - 1n : (1n << BigInt(GUEST_CPUS)) - 1n;
        const eff = mask2 & allowed;
        if (eff === 0n) {
          err(EINVAL3);
          return;
        }
        schedWrite(slot3, PROC_AFFINITY, Number(eff & 0xffffffffn));
        schedWrite(slot3, PROC_AFFINITY + 1, Number(eff >> 32n & 0xffffffffn));
        cpu2.setReg(0, 0n);
        return;
      }
      const slot2 = schedSlotOf(pid2);
      if (slot2 < 0) {
        err(ESRCH2);
        return;
      }
      const out2 = Math.min(len, CPUMASK_BYTES);
      if (syscalls.ptrFaults(bytes, ptr2, out2)) {
        err(EFAULT2);
        return;
      }
      const mask = schedAffinity(slot2);
      for (let i = 0; i < out2; i++) {
        bytes[ptr2 + GUEST_MEM_BASE + i] = Number(mask >> BigInt(i * 8) & 0xffn);
      }
      cpu2.setReg(0, BigInt(out2));
      return;
    }
    if (nr === 127) {
      const pid2 = Number(BigInt.asIntN(32, cpu2.getReg(0)));
      const ptr2 = Number(cpu2.getReg(1));
      if (pid2 < 0) {
        err(EINVAL3);
        return;
      }
      const slot2 = schedSlotOf(pid2);
      if (slot2 < 0) {
        err(ESRCH2);
        return;
      }
      if (ptr2 === 0 || syscalls.ptrFaults(bytes, ptr2, 16)) {
        err(EFAULT2);
        return;
      }
      const policy = schedRead(slot2, PROC_POLICY);
      const ms = policy === SCHED_RR ? RR_TIMESLICE_MS : policy === SCHED_FIFO || policy === SCHED_DEADLINE ? 0 : FAIR_SLICE_MS;
      view.setBigInt64(ptr2 + GUEST_MEM_BASE, BigInt(Math.floor(ms / 1e3)), true);
      view.setBigInt64(ptr2 + GUEST_MEM_BASE + 8, BigInt(ms % 1e3 * 1e6), true);
      cpu2.setReg(0, 0n);
      return;
    }
    if (nr === 274) {
      const pid2 = Number(BigInt.asIntN(32, cpu2.getReg(0)));
      const ptr2 = Number(cpu2.getReg(1));
      const flags2 = Number(cpu2.getReg(2));
      if (flags2 !== 0) {
        err(EINVAL3);
        return;
      }
      if (ptr2 === 0 || pid2 < 0) {
        err(EINVAL3);
        return;
      }
      if (syscalls.ptrFaults(bytes, ptr2, 4)) {
        err(EFAULT2);
        return;
      }
      const size2 = view.getUint32(ptr2 + GUEST_MEM_BASE, true);
      if (size2 < SCHED_ATTR_SIZE_VER0 || size2 > 4096) {
        if (!syscalls.ptrFaults(bytes, ptr2, 4)) view.setUint32(ptr2 + GUEST_MEM_BASE, SCHED_ATTR_SIZE_VER0, true);
        err(E2BIG2);
        return;
      }
      if (syscalls.ptrFaults(bytes, ptr2, SCHED_ATTR_SIZE_VER0)) {
        err(EFAULT2);
        return;
      }
      const base2 = ptr2 + GUEST_MEM_BASE;
      const rawPolicy = view.getUint32(base2 + 4, true);
      const attrFlags = view.getBigUint64(base2 + 8, true);
      const nice = view.getInt32(base2 + 16, true);
      const prio = view.getUint32(base2 + 20, true);
      const runtime = view.getBigUint64(base2 + 24, true);
      const deadline = view.getBigUint64(base2 + 32, true);
      const period = view.getBigUint64(base2 + 40, true);
      const resetOnFork = attrFlags & 1n ? 1 : 0;
      const policy = rawPolicy & ~SCHED_RESET_ON_FORK;
      if (!validPolicy(policy)) {
        err(EINVAL3);
        return;
      }
      const slot2 = schedSlotOf(pid2);
      if (slot2 < 0) {
        err(ESRCH2);
        return;
      }
      if (isRtPolicy(policy) ? prio < 1 || prio > 99 : prio !== 0) {
        err(EINVAL3);
        return;
      }
      if (policy === SCHED_DEADLINE) {
        const eff = period === 0n ? deadline : period;
        if (runtime === 0n || deadline === 0n || runtime > deadline || deadline > eff) {
          err(EINVAL3);
          return;
        }
      }
      if (!schedSameOwner(slot2)) {
        err(EPERM3);
        return;
      }
      if ((isRtPolicy(policy) || policy === SCHED_DEADLINE) && !schedPrivileged()) {
        err(EPERM3);
        return;
      }
      schedWrite(slot2, PROC_POLICY, policy);
      schedWrite(slot2, PROC_RTPRIO, prio);
      schedWrite(slot2, PROC_SCHED_FLAGS, resetOnFork);
      if (policy === SCHED_OTHER || policy === SCHED_BATCH) {
        schedWrite(slot2, PROC_NICE, Math.max(-20, Math.min(19, nice)));
      }
      schedWrite64(slot2, PROC_DL_RUNTIME, policy === SCHED_DEADLINE ? runtime : 0n);
      schedWrite64(slot2, PROC_DL_DEADLINE, policy === SCHED_DEADLINE ? deadline : 0n);
      schedWrite64(slot2, PROC_DL_PERIOD, policy === SCHED_DEADLINE ? period === 0n ? deadline : period : 0n);
      cpu2.setReg(0, 0n);
      return;
    }
    const pid = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    const ptr = Number(cpu2.getReg(1));
    const size = Number(cpu2.getReg(2));
    const flags = Number(cpu2.getReg(3));
    if (ptr === 0 || size < SCHED_ATTR_SIZE_VER0 || size > 4096 || flags !== 0 || pid < 0) {
      err(EINVAL3);
      return;
    }
    const slot = schedSlotOf(pid);
    if (slot < 0) {
      err(ESRCH2);
      return;
    }
    const out = Math.min(size, SCHED_ATTR_SIZE_VER0);
    if (syscalls.ptrFaults(bytes, ptr, out)) {
      err(EFAULT2);
      return;
    }
    const base = ptr + GUEST_MEM_BASE;
    view.setUint32(base, out, true);
    view.setUint32(base + 4, schedRead(slot, PROC_POLICY), true);
    view.setBigUint64(base + 8, BigInt(schedRead(slot, PROC_SCHED_FLAGS) & 1), true);
    view.setInt32(base + 16, schedRead(slot, PROC_NICE), true);
    view.setUint32(base + 20, schedRead(slot, PROC_RTPRIO), true);
    view.setBigUint64(base + 24, schedRead64(slot, PROC_DL_RUNTIME), true);
    view.setBigUint64(base + 32, schedRead64(slot, PROC_DL_DEADLINE), true);
    view.setBigUint64(base + 40, schedRead64(slot, PROC_DL_PERIOD), true);
    cpu2.setReg(0, 0n);
  }
  function guestAccessible(addr, len, wantWrite) {
    if (syscalls.ptrFaults(new Uint8Array(memory.buffer), addr, len)) return false;
    if (!protPainted()) return true;
    const bytes = new Uint8Array(memory.buffer);
    const need = wantWrite ? 2 : 1;
    for (let p = addr >>> 12, last = addr + len - 1 >>> 12; p <= last; p++) {
      if ((bytes[PROT_TABLE_BASE + p] & need) === 0) return false;
    }
    return true;
  }
  var SPLICE_F_NONBLOCK = 2;
  var SPLICE_F_ALL = 1 | 2 | 4 | 8;
  var SPLICE_CHUNK = 65536;
  function pipeEnd(fd) {
    const info = kernelPipes.get(fd);
    if (!info) return null;
    return { ring: new PipeRing(shmArena, info.pipeOff), off: info.pipeOff, mode: info.mode };
  }
  function unixStreamRx(fd) {
    const s = unixSocks.get(fd);
    if (!s || s.framed || s.rx === void 0) return null;
    return new PipeRing(shmArena, s.rx);
  }
  function unixStreamTx(fd) {
    const s = unixSocks.get(fd);
    if (!s || s.framed || s.tx === void 0) return null;
    return new PipeRing(shmArena, s.tx);
  }
  function spliceReadFile(nativeFd, count, at) {
    let out = null;
    let e = 0;
    const want = Math.min(count, BRIDGE_DATA_MAX);
    const method = at === null ? "fs_read" : "fs_pread";
    const args = at === null ? { fd: nativeFd, count: want } : { fd: nativeFd, offset: at, count: want };
    handleBridgeCall(method, args, (resp) => {
      if (typeof resp?.ret === "number" && resp.ret < 0) e = resp.ret;
      else out = resp?.bytes ?? new Uint8Array(0);
    });
    return e !== 0 ? e : out ?? new Uint8Array(0);
  }
  function spliceWriteAt(nativeFd, data, at) {
    let r = -5;
    handleBridgeCall(
      "fs_pwrite",
      { fd: nativeFd, offset: at, data: bytesToBase64Fast(data) },
      (resp) => {
        r = typeof resp?.ret === "number" ? resp.ret : -5;
      }
    );
    return r;
  }
  function handleSplice(nr, cpu2, view) {
    const err = (e) => cpu2.setReg(0, BigInt.asUintN(64, BigInt(-e)));
    if (nr === 75) {
      handleVmsplice(cpu2, view);
      return;
    }
    if (nr === 77) {
      handleTee(cpu2);
      return;
    }
    const inFd = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    const offInPtr = Number(cpu2.getReg(1));
    const outFd = Number(BigInt.asIntN(32, cpu2.getReg(2)));
    const offOutPtr = Number(cpu2.getReg(3));
    const len = Number(BigInt.asIntN(64, cpu2.getReg(4)));
    const flags = Number(cpu2.getReg(5));
    if (flags & ~SPLICE_F_ALL) {
      err(22);
      return;
    }
    if (len < 0) {
      err(22);
      return;
    }
    const inEntry = syscalls.getFdEntry(inFd);
    const outEntry = syscalls.getFdEntry(outFd);
    if (!inEntry || !syscalls.fdAccess(inFd).readable) {
      err(9);
      return;
    }
    if (!outEntry || !syscalls.fdAccess(outFd).writable) {
      err(9);
      return;
    }
    if (syscalls.fdIsPath(inFd) || syscalls.fdIsPath(outFd)) {
      err(9);
      return;
    }
    if (inEntry.type === "native" && inEntry.isDir) {
      err(22);
      return;
    }
    if (outEntry.type === "native" && outEntry.isDir) {
      err(22);
      return;
    }
    const src = pipeEnd(inFd), dst = pipeEnd(outFd);
    const inIsPipe = src?.mode === "read", outIsPipe = dst?.mode === "write";
    if (!inIsPipe && !outIsPipe) {
      err(22);
      return;
    }
    if (inIsPipe && offInPtr !== 0) {
      err(29);
      return;
    }
    if (outIsPipe && offOutPtr !== 0) {
      err(29);
      return;
    }
    if (inIsPipe && outIsPipe && src.off === dst.off) {
      err(22);
      return;
    }
    if (len === 0) {
      cpu2.setReg(0, 0n);
      return;
    }
    const nonblock = (flags & SPLICE_F_NONBLOCK) !== 0;
    if (!outIsPipe && syscalls.fdStatusFlags(outFd) & 1024) {
      err(22);
      return;
    }
    const srcRing = inIsPipe ? src.ring : unixStreamRx(inFd);
    if (srcRing) {
      if (!outIsPipe && outEntry.type !== "native" && !unixSocks.has(outFd)) {
        err(22);
        return;
      }
      let outAt = null;
      if (offOutPtr !== 0) {
        if (!guestAccessible(offOutPtr, 8, true)) {
          err(14);
          return;
        }
        outAt = view.getBigInt64(offOutPtr + GUEST_MEM_BASE, true);
        if (outAt < 0n) {
          err(22);
          return;
        }
      }
      const dstRing = outIsPipe ? dst.ring : unixStreamTx(outFd);
      const buf = new Uint8Array(Math.min(len, SPLICE_CHUNK));
      let got2 = 0;
      for (; ; ) {
        if (dstRing?.readerClosed) {
          syscalls.raiseSignal(13);
          err(32);
          return;
        }
        got2 = srcRing.peek(buf);
        if (dstRing) got2 = Math.min(got2, dstRing.freeBytes);
        if (got2 > 0) break;
        if (srcRing.bytesAvailable === 0 && !srcRing.writersOpen()) {
          cpu2.setReg(0, 0n);
          return;
        }
        if (nonblock || syscalls.isNonBlockingFd(inFd) || syscalls.isNonBlockingFd(outFd)) {
          err(11);
          return;
        }
        if (pollAndQueueSignals()) {
          err(4);
          return;
        }
        waitIo(readIoGeneration(), 20);
      }
      const data = buf.subarray(0, got2);
      const w = dstRing ? dstRing.write(data) : outAt !== null ? spliceWriteAt(outEntry.nativeFd, data, Number(outAt)) : sendfileWrite(outFd, data);
      if (w < 0) {
        err(-w);
        return;
      }
      srcRing.discard(w);
      bumpIoGeneration();
      if (outAt !== null) view.setBigInt64(offOutPtr + GUEST_MEM_BASE, outAt + BigInt(w), true);
      cpu2.setReg(0, BigInt(w));
      return;
    }
    if (inEntry.type !== "native") {
      err(22);
      return;
    }
    let inAt = null;
    if (offInPtr !== 0) {
      if (!guestAccessible(offInPtr, 8, true)) {
        err(14);
        return;
      }
      inAt = view.getBigInt64(offInPtr + GUEST_MEM_BASE, true);
      if (inAt < 0n) {
        err(22);
        return;
      }
    }
    let room = 0;
    for (; ; ) {
      if (dst.ring.readerClosed) {
        syscalls.raiseSignal(13);
        err(32);
        return;
      }
      room = Math.min(len, SPLICE_CHUNK, dst.ring.freeBytes);
      if (room > 0) break;
      if (nonblock || syscalls.isNonBlockingFd(outFd)) {
        err(11);
        return;
      }
      if (pollAndQueueSignals()) {
        err(4);
        return;
      }
      waitIo(readIoGeneration(), 20);
    }
    const got = spliceReadFile(inEntry.nativeFd, room, inAt === null ? null : Number(inAt));
    if (typeof got === "number") {
      err(-got);
      return;
    }
    if (got.length === 0) {
      cpu2.setReg(0, 0n);
      return;
    }
    const n = dst.ring.write(got);
    bumpIoGeneration();
    if (inAt !== null) view.setBigInt64(offInPtr + GUEST_MEM_BASE, inAt + BigInt(n), true);
    cpu2.setReg(0, BigInt(n));
  }
  function handleTee(cpu2) {
    const err = (e) => cpu2.setReg(0, BigInt.asUintN(64, BigInt(-e)));
    const inFd = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    const outFd = Number(BigInt.asIntN(32, cpu2.getReg(1)));
    const len = Number(BigInt.asIntN(64, cpu2.getReg(2)));
    const flags = Number(cpu2.getReg(3));
    if (flags & ~SPLICE_F_ALL) {
      err(22);
      return;
    }
    if (len < 0) {
      err(22);
      return;
    }
    if (!syscalls.getFdEntry(inFd) || !syscalls.getFdEntry(outFd)) {
      err(9);
      return;
    }
    const src = pipeEnd(inFd), dst = pipeEnd(outFd);
    if (src?.mode !== "read" || dst?.mode !== "write") {
      err(22);
      return;
    }
    if (src.off === dst.off) {
      err(22);
      return;
    }
    if (len === 0) {
      cpu2.setReg(0, 0n);
      return;
    }
    const nonblock = (flags & SPLICE_F_NONBLOCK) !== 0;
    const buf = new Uint8Array(Math.min(len, SPLICE_CHUNK));
    let got = 0;
    for (; ; ) {
      if (dst.ring.readerClosed) {
        syscalls.raiseSignal(13);
        err(32);
        return;
      }
      got = Math.min(src.ring.peek(buf), dst.ring.freeBytes);
      if (got > 0) break;
      if (src.ring.bytesAvailable === 0 && !src.ring.writersOpen()) {
        cpu2.setReg(0, 0n);
        return;
      }
      if (nonblock || syscalls.isNonBlockingFd(inFd)) {
        err(11);
        return;
      }
      if (pollAndQueueSignals()) {
        err(4);
        return;
      }
      waitIo(readIoGeneration(), 20);
    }
    const n = dst.ring.write(buf.subarray(0, got));
    bumpIoGeneration();
    cpu2.setReg(0, BigInt(n));
  }
  function handleVmsplice(cpu2, view) {
    const err = (e) => cpu2.setReg(0, BigInt.asUintN(64, BigInt(-e)));
    const fd = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    const iovPtr = Number(cpu2.getReg(1));
    const nrSegs = Number(cpu2.getReg(2));
    const flags = Number(cpu2.getReg(3));
    if (flags & ~SPLICE_F_ALL) {
      err(22);
      return;
    }
    if (!syscalls.getFdEntry(fd)) {
      err(9);
      return;
    }
    const end = pipeEnd(fd);
    if (!end) {
      err(9);
      return;
    }
    if (nrSegs > 1024) {
      err(22);
      return;
    }
    if (nrSegs === 0) {
      cpu2.setReg(0, 0n);
      return;
    }
    if (!guestAccessible(iovPtr, nrSegs * 16, false)) {
      err(14);
      return;
    }
    const segs = [];
    let total = 0;
    for (let i = 0; i < nrSegs; i++) {
      const base = Number(view.getBigUint64(iovPtr + GUEST_MEM_BASE + i * 16, true));
      const l = Number(view.getBigUint64(iovPtr + GUEST_MEM_BASE + i * 16 + 8, true));
      if (l > 0 && !guestAccessible(base, l, end.mode === "read")) {
        err(14);
        return;
      }
      segs.push({ base, len: l });
      total += l;
    }
    if (total === 0) {
      cpu2.setReg(0, 0n);
      return;
    }
    const bytes = new Uint8Array(memory.buffer);
    const nonblock = (flags & SPLICE_F_NONBLOCK) !== 0;
    if (end.mode === "write") {
      const data = new Uint8Array(Math.min(total, SPLICE_CHUNK));
      let filled = 0;
      for (const s of segs) {
        if (filled >= data.length) break;
        const take = Math.min(s.len, data.length - filled);
        data.set(bytes.subarray(s.base + GUEST_MEM_BASE, s.base + GUEST_MEM_BASE + take), filled);
        filled += take;
      }
      for (; ; ) {
        if (end.ring.readerClosed) {
          syscalls.raiseSignal(13);
          err(32);
          return;
        }
        if (end.ring.freeBytes > 0) break;
        if (nonblock || syscalls.isNonBlockingFd(fd)) {
          err(11);
          return;
        }
        if (pollAndQueueSignals()) {
          err(4);
          return;
        }
        waitIo(readIoGeneration(), 20);
      }
      const n = end.ring.write(data.subarray(0, filled));
      bumpIoGeneration();
      cpu2.setReg(0, BigInt(n));
      return;
    }
    const buf = new Uint8Array(Math.min(total, SPLICE_CHUNK));
    let got = 0;
    for (; ; ) {
      got = end.ring.peek(buf);
      if (got > 0) break;
      if (!end.ring.writersOpen()) {
        cpu2.setReg(0, 0n);
        return;
      }
      if (nonblock || syscalls.isNonBlockingFd(fd)) {
        err(11);
        return;
      }
      if (pollAndQueueSignals()) {
        err(4);
        return;
      }
      waitIo(readIoGeneration(), 20);
    }
    let placed = 0;
    for (const s of segs) {
      if (placed >= got) break;
      const take = Math.min(s.len, got - placed);
      bytes.set(buf.subarray(placed, placed + take), s.base + GUEST_MEM_BASE);
      placed += take;
    }
    end.ring.discard(placed);
    bumpIoGeneration();
    cpu2.setReg(0, BigInt(placed));
  }
  function sendfileWrite(fd, data) {
    const e = syscalls.getFdEntry(fd);
    if (!e) return -9;
    if (e.type === "native") {
      let r = -5;
      handleBridgeCall("fs_write", { fd: e.nativeFd, data: bytesToBase64Fast(data) }, (resp) => {
        r = typeof resp?.ret === "number" ? resp.ret : -5;
      });
      return r;
    }
    const sock = unixSocks.get(fd);
    if (sock) {
      if (sock.framed) {
        const dest = dgramDefaultRing(sock);
        if (dest < 0) return dest;
        return unixDgramWrite(fd, dest, data, sock.name);
      }
      if (sock.tx === void 0) return -107;
      return unixStreamWrite(fd, sock.tx, data);
    }
    const pipe = kernelPipes.get(fd);
    if (pipe?.mode === "write") {
      const ring = new PipeRing(shmArena, pipe.pipeOff);
      if (syscalls.isNonBlockingFd(fd)) {
        const n2 = ring.write(data);
        if (n2 === -1) {
          syscalls.raiseSignal(13);
          return -32;
        }
        if (n2 === 0) return -11;
        return n2;
      }
      const n = ring.writeAll(data, interruptibleWaiter);
      if (n === -1) {
        syscalls.raiseSignal(13);
        return -32;
      }
      if (n === -2) return -4;
      return n;
    }
    return syscalls.writeOutOfBand(fd, data);
  }
  function handleSendfile(cpu2, view) {
    const err = (e) => cpu2.setReg(0, BigInt.asUintN(64, BigInt(-e)));
    const outFd = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    const inFd = Number(BigInt.asIntN(32, cpu2.getReg(1)));
    const offPtr = Number(cpu2.getReg(2));
    let count = Number(BigInt.asIntN(64, cpu2.getReg(3)));
    const inEntry = syscalls.getFdEntry(inFd);
    if (!inEntry || !syscalls.fdAccess(inFd).readable) {
      err(9);
      return;
    }
    if (inEntry.type !== "native") {
      err(22);
      return;
    }
    let pos = -1n;
    if (offPtr !== 0) {
      if (!guestAccessible(offPtr, 8, true)) {
        err(14);
        return;
      }
      pos = view.getBigInt64(offPtr + GUEST_MEM_BASE, true);
      if (pos < 0n) {
        err(22);
        return;
      }
    }
    const outEntry = syscalls.getFdEntry(outFd);
    if (!outEntry || !syscalls.fdAccess(outFd).writable) {
      err(9);
      return;
    }
    if (count < 0) {
      err(22);
      return;
    }
    let moved = 0;
    while (count > 0) {
      const chunk = Math.min(count, BRIDGE_DATA_MAX);
      let data = null;
      let readErr = 0;
      const args = offPtr !== 0 ? { fd: inEntry.nativeFd, offset: Number(pos) + moved, count: chunk } : { fd: inEntry.nativeFd, count: chunk };
      handleBridgeCall(offPtr !== 0 ? "fs_pread" : "fs_read", args, (resp) => {
        if (typeof resp?.ret === "number" && resp.ret < 0) readErr = resp.ret;
        else data = resp?.bytes ?? new Uint8Array(0);
      });
      if (readErr !== 0) {
        if (moved === 0) {
          err(-readErr);
          return;
        }
        break;
      }
      const got = data ?? new Uint8Array(0);
      if (got.length === 0) break;
      const w = sendfileWrite(outFd, got);
      if (w < 0) {
        if (moved === 0) {
          err(-w);
          return;
        }
        break;
      }
      moved += w;
      count -= w;
      if (w < got.length) break;
    }
    if (offPtr !== 0) view.setBigInt64(offPtr + GUEST_MEM_BASE, pos + BigInt(moved), true);
    cpu2.setReg(0, BigInt(moved));
  }
  var NATIVE_FS_DEV = 1;
  function inotifyCall(msg, words = 3) {
    Atomics.store(myMailbox, MBOX_READY, 0);
    self.postMessage({ type: "inotify", ...msg });
    awaitMailbox("inotify");
    const out = [];
    for (let i = 0; i < words; i++) out.push(Atomics.load(myMailbox, MBOX_VALUE0 + i));
    return out;
  }
  function inotifyFdinfo(id) {
    let out = "", start = 0;
    for (; ; ) {
      const r = inotifyCall({ op: "watches", id, start }, 30);
      if (r[0] !== 0) return out;
      const total = r[1], n = r[2];
      for (let i = 0; i < n; i++) {
        const wd = r[3 + i * 3], mask = r[4 + i * 3] >>> 0, ino = r[5 + i * 3] >>> 0;
        out += `inotify wd:${wd} ino:${ino.toString(16)} sdev:${NATIVE_FS_DEV.toString(16)} mask:${mask.toString(16)} ignored_mask:0 fhandle-bytes:8 fhandle-type:1 f_handle:${ino.toString(16).padStart(16, "0")}
`;
      }
      start += n;
      if (start >= total || n === 0) return out;
    }
  }
  function inotifyActive() {
    return Atomics.load(globalWords, GLOBAL_INOTIFY_WATCHES) > 0;
  }
  function reportInotify(path, mask, cookie = 0, isDir, selfOnly = false, probe = path, unlinked = false) {
    if (!inotifyActive()) return;
    if (isDir === void 0) {
      handleBridgeCall("fs_stat", { path: probe }, (r) => {
        if (r && r.ret === 0 && r._meta) isDir = ((r._meta[0] | 0) & 61440) === 16384;
      });
    }
    self.postMessage({
      type: "inotify_event",
      path,
      mask,
      cookie,
      isdir: isDir === true,
      selfonly: selfOnly,
      moved: probe,
      unlinked
    });
  }
  function handleInotify(nr, cpu2, view) {
    const bytes = new Uint8Array(memory.buffer);
    if (nr === 26) {
      const IN_NONBLOCK = 2048, IN_CLOEXEC = 524288;
      const flags = Number(cpu2.getReg(0));
      if (flags & ~(IN_NONBLOCK | IN_CLOEXEC)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
        return;
      }
      const fd2 = syscalls.allocLowestFd();
      if (fd2 < 0) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-24)));
        return;
      }
      const id = inotifyCall({ op: "init", id: 0 })[1];
      syscalls.registerFd(fd2, { type: "inotify", id }, flags & IN_NONBLOCK);
      if (flags & IN_CLOEXEC) syscalls.markCloexec(fd2);
      cpu2.setReg(0, BigInt(fd2));
      return;
    }
    if (nr === 27) {
      const id = syscalls.inotifyIdOf(Number(cpu2.getReg(0)));
      if (id < 0) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-9)));
        return;
      }
      const pathPtr = Number(cpu2.getReg(1));
      if (badGuestPtr(bytes, pathPtr, 1)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
      const mask = Number(cpu2.getReg(2)) >>> 0;
      const IN_ALL_EVENTS = 4095;
      if ((mask & IN_ALL_EVENTS) === 0) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
        return;
      }
      const raw = readCString(pathPtr);
      if (raw === null || raw.length === 0) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-2)));
        return;
      }
      const path = syscalls.resolveForWatch(raw);
      if (path === null) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
        return;
      }
      let exists = false, ino = 0;
      handleBridgeCall("fs_stat", { path }, (r2) => {
        exists = (r2?.ret ?? -1) >= 0;
        if (exists && r2._meta) ino = r2._meta[6] | 0;
      });
      if (!exists) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-2)));
        return;
      }
      const r = inotifyCall({ op: "add", id, path, mask, ino });
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
      return;
    }
    if (nr === 28) {
      const id = syscalls.inotifyIdOf(Number(cpu2.getReg(0)));
      if (id < 0) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-9)));
        return;
      }
      const r = inotifyCall({ op: "rm", id, wd: Number(BigInt.asIntN(32, cpu2.getReg(1))) });
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
      return;
    }
    if (nr === 57) {
      inotifyCall({ op: "close", id: syscalls.inotifyIdOf(Number(cpu2.getReg(0))) });
      syscalls.unregisterFd(Number(cpu2.getReg(0)));
      cpu2.setReg(0, 0n);
      return;
    }
    const fd = Number(cpu2.getReg(0));
    const rid = syscalls.inotifyIdOf(fd);
    const buf = Number(cpu2.getReg(1));
    const count = Number(cpu2.getReg(2));
    if (count > 0 && badGuestPtr(bytes, buf, count)) {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
      return;
    }
    for (; ; ) {
      const r = inotifyCall({ op: "read", id: rid, maxlen: count });
      if (r[0] === 0) {
        const off = r[1], len = r[2];
        new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, len).set(new Uint8Array(shmArena, off, len));
        inotifyCall({ op: "free", id: rid, off, len });
        cpu2.setReg(0, BigInt(len));
        return;
      }
      if (r[0] !== -11) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
        return;
      }
      if (syscalls.isNonBlockingFd(fd)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-11)));
        return;
      }
      if (pollAndQueueSignals()) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-4)));
        return;
      }
      waitIo(readIoGeneration(), 20);
    }
  }
  function eventfdCall(msg) {
    Atomics.store(myMailbox, MBOX_READY, 0);
    self.postMessage({ type: "eventfd", ...msg });
    awaitMailbox("eventfd");
    return [
      Atomics.load(myMailbox, MBOX_VALUE0),
      Atomics.load(myMailbox, MBOX_VALUE0 + 1),
      Atomics.load(myMailbox, MBOX_VALUE0 + 2)
    ];
  }
  function posixMqCall(msg) {
    Atomics.store(myMailbox, MBOX_READY, 0);
    self.postMessage({ type: "posix_mq", ...msg });
    awaitMailbox("posix_mq");
    const out = [];
    for (let i = 0; i < 6; i++) out.push(Atomics.load(myMailbox, MBOX_VALUE0 + i));
    return out;
  }
  function mqName(ptr) {
    const bytes = new Uint8Array(memory.buffer);
    if (syscalls.ptrFaults(bytes, ptr, 1)) return -14;
    const raw = readCString(ptr);
    if (raw === null) return -14;
    const name = raw.startsWith("/") ? raw.slice(1) : raw;
    if (name.length === 0 || name === "." || name === ".." || name.includes("/")) return -22;
    if (name.length > 255) return -36;
    return name;
  }
  function mqDeadline(ptr, view) {
    if (ptr === 0) return { leftMs: null };
    const bytes = new Uint8Array(memory.buffer);
    if (syscalls.ptrFaults(bytes, ptr, 16)) return { err: -14 };
    const sec = view.getBigInt64(ptr + GUEST_MEM_BASE, true);
    const nsec = view.getBigInt64(ptr + GUEST_MEM_BASE + 8, true);
    if (nsec < 0n || nsec >= 1000000000n || sec < 0n) return { err: -22 };
    const targetMs = Number(sec) * 1e3 + Number(nsec) / 1e6;
    return { leftMs: targetMs - Number(syscalls.clockNs(0) / 1000000n) };
  }
  function handlePosixMq(nr, cpu2, view) {
    const bytes = new Uint8Array(memory.buffer);
    const creds = syscalls.getCreds();
    const err = (e) => cpu2.setReg(0, BigInt.asUintN(64, BigInt(e)));
    const O_NONBLOCK3 = 2048, O_CLOEXEC = 524288, O_CREAT2 = 64;
    const MQ_PRIO_MAX = 32768;
    if (nr === 57) {
      const fd2 = Number(cpu2.getReg(0));
      const id2 = syscalls.mqIdOf(fd2);
      if (!syscalls.mqHeldElsewhere(fd2, id2)) posixMqCall({ op: "close", id: id2, pid: myPid });
      syscalls.unregisterFd(fd2);
      cpu2.setReg(0, 0n);
      return;
    }
    if (nr === 180) {
      const name = mqName(Number(cpu2.getReg(0)));
      if (typeof name === "number") {
        err(name);
        return;
      }
      const oflag = Number(BigInt.asIntN(32, cpu2.getReg(1)));
      const accmode = oflag & 3;
      if (accmode === 3) {
        err(-22);
        return;
      }
      const mode = Number(cpu2.getReg(2));
      const attrPtr = Number(cpu2.getReg(3));
      let hasAttr = false, maxmsg = 0, msgsize = 0;
      if (oflag & O_CREAT2 && attrPtr !== 0) {
        if (syscalls.ptrFaults(bytes, attrPtr, 64)) {
          err(-14);
          return;
        }
        maxmsg = Number(view.getBigInt64(attrPtr + GUEST_MEM_BASE + 8, true));
        msgsize = Number(view.getBigInt64(attrPtr + GUEST_MEM_BASE + 16, true));
        hasAttr = true;
      }
      const fd2 = syscalls.allocLowestFd(0);
      if (fd2 < 0) {
        err(-24);
        return;
      }
      const r = posixMqCall({
        op: "open",
        name,
        oflag,
        accmode,
        mode,
        hasAttr,
        maxmsg,
        msgsize,
        uid: creds.euid,
        gid: creds.egid,
        pid: myPid
      });
      if (r[0] < 0) {
        err(r[0]);
        return;
      }
      syscalls.registerFd(fd2, { type: "mqueue", id: r[0], name }, oflag & (3 | O_NONBLOCK3));
      if (oflag & O_CLOEXEC) syscalls.markCloexec(fd2);
      cpu2.setReg(0, BigInt(fd2));
      return;
    }
    if (nr === 181) {
      const name = mqName(Number(cpu2.getReg(0)));
      if (typeof name === "number") {
        err(name);
        return;
      }
      err(posixMqCall({ op: "unlink", name, uid: creds.euid, gid: creds.egid })[0]);
      return;
    }
    const SIGEV_SIGNAL2 = 0, SIGEV_NONE2 = 1, SIGEV_THREAD2 = 2;
    const NOTIFY_COOKIE_LEN = 32;
    let notifySigno = -1;
    let notifyRing = -1;
    let notifyFd = -1;
    let notifyCookie;
    if (nr === 184) {
      const sevp = Number(cpu2.getReg(1));
      if (sevp !== 0) {
        if (syscalls.ptrFaults(bytes, sevp, 64)) {
          err(-14);
          return;
        }
        const signo = view.getInt32(sevp + GUEST_MEM_BASE + 8, true);
        const kind = view.getInt32(sevp + GUEST_MEM_BASE + 12, true);
        if (kind !== SIGEV_NONE2 && kind !== SIGEV_SIGNAL2 && kind !== SIGEV_THREAD2) {
          err(-22);
          return;
        }
        if (kind === SIGEV_SIGNAL2 && (signo <= 0 || signo > 64)) {
          err(-22);
          return;
        }
        if (kind === SIGEV_THREAD2) {
          const nlSock = unixSocks.get(signo);
          notifyRing = nlSock?.rx ?? -1;
          notifyFd = signo;
          const cookiePtr = Number(view.getBigUint64(sevp + GUEST_MEM_BASE, true) & 0xffffffffn);
          if (syscalls.ptrFaults(bytes, cookiePtr, NOTIFY_COOKIE_LEN)) {
            err(-14);
            return;
          }
          notifyCookie = bytes.slice(cookiePtr + GUEST_MEM_BASE, cookiePtr + GUEST_MEM_BASE + NOTIFY_COOKIE_LEN);
        }
        notifySigno = kind === SIGEV_NONE2 ? 0 : signo;
      }
    }
    const fd = Number(cpu2.getReg(0));
    const id = syscalls.mqIdOf(fd);
    if (id < 0) {
      err(-9);
      return;
    }
    const access = syscalls.fdAccess(fd);
    const nonblock = syscalls.isNonBlockingFd(fd);
    if (nr === 182) {
      if (!access.writable) {
        err(-9);
        return;
      }
      const ptr = Number(cpu2.getReg(1));
      const len = Number(BigInt.asIntN(64, cpu2.getReg(2)));
      const prio = Number(BigInt.asIntN(32, cpu2.getReg(3)));
      if (prio < 0 || prio >= MQ_PRIO_MAX) {
        err(-22);
        return;
      }
      if (len < 0) {
        err(-22);
        return;
      }
      if (len > 0 && syscalls.ptrFaults(bytes, ptr, len)) {
        err(-14);
        return;
      }
      const dl = mqDeadline(Number(cpu2.getReg(4)), view);
      if ("err" in dl) {
        err(dl.err);
        return;
      }
      const leftMs = dl.leftMs;
      const data = len > 0 ? bytes.slice(ptr + GUEST_MEM_BASE, ptr + GUEST_MEM_BASE + len) : new Uint8Array(0);
      const startMs = Number(syscalls.clockNs(0) / 1000000n);
      for (; ; ) {
        const r = posixMqCall({ op: "send", id, prio, len, data });
        if (r[0] !== -11) {
          err(r[0]);
          return;
        }
        if (nonblock) {
          err(-11);
          return;
        }
        if (leftMs !== null && Number(syscalls.clockNs(0) / 1000000n) - startMs >= leftMs) {
          err(-110);
          return;
        }
        if (pollAndQueueSignals()) {
          err(-4);
          return;
        }
        waitIo(readIoGeneration(), 20);
      }
    }
    if (nr === 183) {
      if (!access.readable) {
        err(-9);
        return;
      }
      const ptr = Number(cpu2.getReg(1));
      const maxlen = Number(BigInt.asIntN(64, cpu2.getReg(2)));
      const prioPtr = Number(cpu2.getReg(3));
      if (maxlen < 0) {
        err(-22);
        return;
      }
      if (maxlen > 0 && syscalls.ptrWriteFaults(bytes, ptr, maxlen)) {
        err(-14);
        return;
      }
      if (prioPtr !== 0 && syscalls.ptrWriteFaults(bytes, prioPtr, 4)) {
        err(-14);
        return;
      }
      const dl = mqDeadline(Number(cpu2.getReg(4)), view);
      if ("err" in dl) {
        err(dl.err);
        return;
      }
      const leftMs = dl.leftMs;
      const startMs = Number(syscalls.clockNs(0) / 1000000n);
      for (; ; ) {
        const r = posixMqCall({ op: "recv", id, maxlen });
        if (r[0] === 0) {
          const off = r[1], len = r[2];
          if (len > 0) bytes.set(new Uint8Array(shmArena, off, len), ptr + GUEST_MEM_BASE);
          if (prioPtr !== 0) view.setUint32(prioPtr + GUEST_MEM_BASE, r[3] >>> 0, true);
          posixMqCall({ op: "free", off, len });
          cpu2.setReg(0, BigInt(len));
          return;
        }
        if (r[0] !== -11) {
          err(r[0]);
          return;
        }
        if (nonblock) {
          err(-11);
          return;
        }
        if (leftMs !== null && Number(syscalls.clockNs(0) / 1000000n) - startMs >= leftMs) {
          err(-110);
          return;
        }
        if (pollAndQueueSignals()) {
          err(-4);
          return;
        }
        waitIo(readIoGeneration(), 20);
      }
    }
    if (nr === 184) {
      err(posixMqCall({ op: "notify", id, pid: myPid, signo: notifySigno, ring: notifyRing, nlFd: notifyFd, cookie: notifyCookie })[0]);
      return;
    }
    const newPtr = Number(cpu2.getReg(1));
    const oldPtr = Number(cpu2.getReg(2));
    if (newPtr !== 0 && syscalls.ptrFaults(bytes, newPtr, 64)) {
      err(-14);
      return;
    }
    if (oldPtr !== 0 && syscalls.ptrWriteFaults(bytes, oldPtr, 64)) {
      err(-14);
      return;
    }
    const attr = posixMqCall({ op: "getattr", id });
    if (attr[0] < 0) {
      err(attr[0]);
      return;
    }
    if (oldPtr !== 0) {
      const b = oldPtr + GUEST_MEM_BASE;
      bytes.fill(0, b, b + 64);
      view.setBigInt64(b, BigInt(nonblock ? O_NONBLOCK3 : 0), true);
      view.setBigInt64(b + 8, BigInt(attr[1]), true);
      view.setBigInt64(b + 16, BigInt(attr[2]), true);
      view.setBigInt64(b + 24, BigInt(attr[3]), true);
    }
    if (newPtr !== 0) {
      const want = Number(view.getBigInt64(newPtr + GUEST_MEM_BASE, true));
      syscalls.setFdNonBlocking(fd, (want & O_NONBLOCK3) !== 0);
    }
    cpu2.setReg(0, 0n);
  }
  function sysvMsgCall(msg) {
    Atomics.store(myMailbox, MBOX_READY, 0);
    self.postMessage({ type: "sysv_msg", ...msg });
    awaitMailbox("sysv_msg");
    const out = [];
    for (let i = 0; i < 14; i++) out.push(Atomics.load(myMailbox, MBOX_VALUE0 + i));
    return out;
  }
  function handleSysvMsg(nr, cpu2, view) {
    const creds = syscalls.getCreds();
    const bytes = new Uint8Array(memory.buffer);
    if (nr === 186) {
      const r = sysvMsgCall({
        op: "get",
        key: Number(BigInt.asIntN(32, cpu2.getReg(0))),
        flg: Number(cpu2.getReg(1)),
        uid: creds.euid,
        gid: creds.egid
      });
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
      return;
    }
    if (nr === 189) {
      const id2 = Number(BigInt.asIntN(32, cpu2.getReg(0)));
      const ptr = Number(cpu2.getReg(1));
      const sz = Number(BigInt.asIntN(64, cpu2.getReg(2)));
      const flg = Number(cpu2.getReg(3));
      const IPC_NOWAIT = 2048;
      if (sz < 0 || sz > MSGMAX) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
        return;
      }
      if (badGuestPtr(bytes, ptr, 8 + Math.max(0, sz))) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
      const mtype = view.getBigInt64(ptr + GUEST_MEM_BASE, true);
      if (mtype < 1n) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
        return;
      }
      const data = bytes.slice(ptr + GUEST_MEM_BASE + 8, ptr + GUEST_MEM_BASE + 8 + sz);
      let slept = false;
      for (; ; ) {
        const r = sysvMsgCall({
          op: "snd",
          id: id2,
          mtype: mtype.toString(),
          len: sz,
          data,
          uid: creds.euid,
          gid: creds.egid
        });
        if (r[0] === -22 && slept) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(-43)));
          return;
        }
        if (r[0] !== -11) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
          return;
        }
        if (flg & IPC_NOWAIT) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(-11)));
          return;
        }
        if (pollAndQueueSignals()) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(-4)));
          return;
        }
        slept = true;
        waitIo(readIoGeneration(), 20);
      }
    }
    if (nr === 188) {
      const id2 = Number(BigInt.asIntN(32, cpu2.getReg(0)));
      const ptr = Number(cpu2.getReg(1));
      const sz = Number(BigInt.asIntN(64, cpu2.getReg(2)));
      const mtyp = BigInt.asIntN(64, cpu2.getReg(3));
      const flg = Number(cpu2.getReg(4));
      const IPC_NOWAIT = 2048, MSG_COPY = 16384, MSG_EXCEPT = 8192;
      if (sz < 0) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
        return;
      }
      if (badGuestPtr(bytes, ptr, 8 + Math.max(0, sz))) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
      if (flg & MSG_COPY) {
        if (!(flg & IPC_NOWAIT) || flg & MSG_EXCEPT) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
          return;
        }
        const rc = sysvMsgCall({
          op: "copy",
          id: id2,
          mtype: mtyp.toString(),
          maxlen: sz,
          flg,
          uid: creds.euid,
          gid: creds.egid
        });
        if (rc[0] !== 0) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(rc[0])));
          return;
        }
        const ctype = BigInt(rc[4] >>> 0) << 32n | BigInt(rc[3] >>> 0);
        const ccopied = Math.min(rc[2], sz);
        view.setBigInt64(ptr + GUEST_MEM_BASE, BigInt.asIntN(64, ctype), true);
        if (ccopied > 0) bytes.set(new Uint8Array(shmArena, rc[1], ccopied), ptr + GUEST_MEM_BASE + 8);
        cpu2.setReg(0, BigInt(ccopied));
        return;
      }
      let slept = false;
      for (; ; ) {
        const r = sysvMsgCall({
          op: "rcv",
          id: id2,
          mtype: mtyp.toString(),
          maxlen: sz,
          flg,
          uid: creds.euid,
          gid: creds.egid
        });
        if (r[0] === -22 && slept) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(-43)));
          return;
        }
        if (r[0] === 0) {
          const off = r[1], len = r[2];
          const type = BigInt(r[4] >>> 0) << 32n | BigInt(r[3] >>> 0);
          const copied = Math.min(len, sz);
          view.setBigInt64(ptr + GUEST_MEM_BASE, BigInt.asIntN(64, type), true);
          if (copied > 0) bytes.set(new Uint8Array(shmArena, off, copied), ptr + GUEST_MEM_BASE + 8);
          sysvMsgCall({ op: "free", off, len });
          cpu2.setReg(0, BigInt(copied));
          return;
        }
        if (r[0] !== -11) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
          return;
        }
        if (flg & IPC_NOWAIT) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(-42)));
          return;
        }
        if (pollAndQueueSignals()) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(-4)));
          return;
        }
        slept = true;
        waitIo(readIoGeneration(), 20);
      }
    }
    const id = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    const cmd = Number(cpu2.getReg(1));
    const buf = Number(cpu2.getReg(2));
    const IPC_RMID = 0, IPC_SET = 1, IPC_STAT = 2, IPC_INFO = 3;
    const MSG_STAT = 11, MSG_INFO = 12, MSG_STAT_ANY = 13;
    if (cmd === IPC_INFO || cmd === MSG_INFO) {
      if (badGuestPtr(bytes, buf, 32)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
      const r = sysvMsgCall({ op: "info" });
      const base = buf + GUEST_MEM_BASE;
      bytes.fill(0, base, base + 32);
      const usage = cmd === MSG_INFO;
      view.setInt32(base + 0, usage ? r[3] : MSGPOOL, true);
      view.setInt32(base + 4, usage ? r[2] : 0, true);
      view.setInt32(base + 8, MSGMAX, true);
      view.setInt32(base + 12, r[4], true);
      view.setInt32(base + 16, MSGMNI, true);
      view.setInt32(base + 20, 16, true);
      view.setInt32(base + 24, usage ? r[2] : MSGTQL, true);
      view.setUint16(base + 28, 65535, true);
      cpu2.setReg(0, BigInt(Math.max(0, r[1] - 1)));
      return;
    }
    if (cmd === MSG_STAT || cmd === MSG_STAT_ANY) {
      if (badGuestPtr(bytes, buf, MSQID_DS_SIZE)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
      const r = sysvMsgCall({ op: "statidx", id, uid: creds.euid, gid: creds.egid });
      if (r[0] < 0) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
        return;
      }
      writeMsqidDs(view, buf, r);
      cpu2.setReg(0, BigInt(r[13]));
      return;
    }
    if (cmd === IPC_RMID) {
      const r = sysvMsgCall({ op: "rm", id, uid: creds.euid, gid: creds.egid });
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
      return;
    }
    if (cmd === IPC_STAT) {
      if (badGuestPtr(bytes, buf, MSQID_DS_SIZE)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
      const r = sysvMsgCall({ op: "stat", id, uid: creds.euid, gid: creds.egid });
      if (r[0] < 0) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
        return;
      }
      writeMsqidDs(view, buf, r);
      cpu2.setReg(0, 0n);
      return;
    }
    if (cmd === IPC_SET) {
      if (badGuestPtr(bytes, buf, MSQID_DS_SIZE)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
      const base = buf + GUEST_MEM_BASE;
      const r = sysvMsgCall({
        op: "set",
        id,
        uid: creds.euid,
        gid: creds.egid,
        newUid: view.getUint32(base + 4, true),
        newGid: view.getUint32(base + 8, true),
        mode: view.getUint32(base + 20, true),
        qbytes: Number(view.getBigUint64(base + 88, true))
      });
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
      return;
    }
    cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
  }
  var MSQID_DS_SIZE = 120;
  function writeMsqidDs(view, buf, r) {
    const base = buf + GUEST_MEM_BASE;
    new Uint8Array(memory.buffer).fill(0, base, base + MSQID_DS_SIZE);
    view.setInt32(base + 0, r[1], true);
    view.setUint32(base + 4, r[2], true);
    view.setUint32(base + 8, r[3], true);
    view.setUint32(base + 12, r[2], true);
    view.setUint32(base + 16, r[3], true);
    view.setUint32(base + 20, r[4], true);
    view.setBigUint64(base + 48, BigInt(r[10]), true);
    view.setBigUint64(base + 56, BigInt(r[11]), true);
    view.setBigUint64(base + 64, BigInt(r[12]), true);
    view.setBigUint64(base + 72, BigInt(r[5]), true);
    view.setBigUint64(base + 80, BigInt(r[6]), true);
    view.setBigUint64(base + 88, BigInt(r[7]), true);
    view.setInt32(base + 96, r[8], true);
    view.setInt32(base + 100, r[9], true);
  }
  var MSGMAX = 8192;
  var MSGMNI = 32e3;
  var MSGPOOL = 1024;
  var MSGTQL = 32e3;
  function sysvSemCall(msg) {
    Atomics.store(myMailbox, MBOX_READY, 0);
    self.postMessage({ type: "sysv_sem", ...msg });
    awaitMailbox("sysv_sem");
    const out = [];
    for (let i = 0; i < 11; i++) out.push(Atomics.load(myMailbox, MBOX_VALUE0 + i));
    return out;
  }
  function sysvipcListing(which) {
    const r = sysvSemCall({ op: "proclist", which });
    if (r[0] !== 0) return "";
    const off = r[1], len = r[2];
    const text = new TextDecoder().decode(new Uint8Array(shmArena, off, len).slice());
    sysvSemCall({ op: "free", off, len });
    return text;
  }
  var SEMID_DS_SIZE = 88;
  var SEMBUF_SIZE = 6;
  var SEMMSL = 32e3;
  var SEMMNS = 1024e6;
  var SEMOPM = 500;
  var SEMMNI = 32e3;
  var SEMVMX = 32767;
  var SEMAEM = SEMVMX;
  var SEMUME = SEMOPM;
  var SEMUSZ = 20;
  function writeSemidDs(view, buf, r) {
    const base = buf + GUEST_MEM_BASE;
    new Uint8Array(memory.buffer).fill(0, base, base + SEMID_DS_SIZE);
    view.setInt32(base + 0, r[1], true);
    view.setUint32(base + 4, r[2], true);
    view.setUint32(base + 8, r[3], true);
    view.setUint32(base + 12, r[9], true);
    view.setUint32(base + 16, r[10], true);
    view.setUint32(base + 20, r[4], true);
    view.setBigUint64(base + 48, BigInt(r[6] >>> 0), true);
    view.setBigUint64(base + 56, BigInt(r[7] >>> 0), true);
    view.setBigUint64(base + 64, BigInt(r[5] >>> 0), true);
  }
  function handleSysvSem(nr, cpu2, view) {
    const creds = syscalls.getCreds();
    const uid = creds.euid, gid = creds.egid;
    const bytes = new Uint8Array(memory.buffer);
    const fail = (e) => cpu2.setReg(0, BigInt.asUintN(64, BigInt(e)));
    if (nr === 190) {
      const r = sysvSemCall({
        op: "get",
        key: Number(BigInt.asIntN(32, cpu2.getReg(0))),
        nsems: Number(BigInt.asIntN(32, cpu2.getReg(1))),
        flg: Number(cpu2.getReg(2)),
        uid,
        gid
      });
      fail(r[0]);
      return;
    }
    if (nr === 192 || nr === 193) {
      const id2 = Number(BigInt.asIntN(32, cpu2.getReg(0)));
      const ptr = Number(cpu2.getReg(1));
      const nsops = Number(cpu2.getReg(2));
      if (nsops === 0) {
        fail(-22);
        return;
      }
      if (nsops > SEMOPM) {
        fail(-7);
        return;
      }
      if (!guestAccessible(ptr, nsops * SEMBUF_SIZE, false)) {
        fail(-14);
        return;
      }
      let deadline = Infinity;
      if (nr === 192) {
        const tsp = Number(cpu2.getReg(3));
        if (tsp !== 0) {
          if (!guestAccessible(tsp, 16, false)) {
            fail(-14);
            return;
          }
          const sec = view.getBigInt64(tsp + GUEST_MEM_BASE, true);
          const nsec = view.getBigInt64(tsp + GUEST_MEM_BASE + 8, true);
          if (sec < 0n || nsec < 0n || nsec >= 1000000000n) {
            fail(-22);
            return;
          }
          deadline = Date.now() + Number(sec) * 1e3 + Number(nsec) / 1e6;
        }
      }
      const ops = [];
      for (let i = 0; i < nsops; i++) {
        const o = ptr + GUEST_MEM_BASE + i * SEMBUF_SIZE;
        ops.push(view.getUint16(o, true), view.getInt16(o + 2, true), view.getInt16(o + 4, true));
      }
      let waiting = 0;
      for (; ; ) {
        const r = sysvSemCall({ op: "op", id: id2, ops, wait: waiting, uid, gid });
        if (r[0] === 0) {
          cpu2.setReg(0, 0n);
          return;
        }
        if (r[0] === -22 && waiting) {
          fail(-43);
          return;
        }
        if (r[0] !== -11) {
          fail(r[0]);
          return;
        }
        if (r[1] === 1) {
          fail(-11);
          return;
        }
        if (Date.now() >= deadline) {
          sysvSemCall({ op: "unwait", id: id2 });
          fail(-11);
          return;
        }
        if (pollAndQueueSignals()) {
          sysvSemCall({ op: "unwait", id: id2 });
          fail(-4);
          return;
        }
        if (!waiting) {
          waiting = 1;
          continue;
        }
        waitIo(readIoGeneration(), 20);
      }
    }
    const id = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    const num = Number(BigInt.asIntN(32, cpu2.getReg(1)));
    const cmd = Number(BigInt.asIntN(32, cpu2.getReg(2))) & ~256;
    const arg = cpu2.getReg(3);
    const IPC_RMID = 0, IPC_SET = 1, IPC_STAT = 2, IPC_INFO = 3;
    const GETPID = 11, GETVAL = 12, GETALL = 13, GETNCNT = 14, GETZCNT = 15, SETVAL = 16, SETALL = 17;
    const SEM_STAT = 18, SEM_INFO = 19, SEM_STAT_ANY = 20;
    if (cmd === IPC_INFO || cmd === SEM_INFO) {
      const buf = Number(arg);
      if (!guestAccessible(buf, 40, true)) {
        fail(-14);
        return;
      }
      const r = sysvSemCall({ op: "info" });
      const base = buf + GUEST_MEM_BASE;
      const usage = cmd === SEM_INFO;
      view.setInt32(base + 0, usage ? r[1] : SEMMNS, true);
      view.setInt32(base + 4, SEMMNI, true);
      view.setInt32(base + 8, usage ? r[2] : SEMMNS, true);
      view.setInt32(base + 12, usage ? r[1] : SEMMNS, true);
      view.setInt32(base + 16, SEMMSL, true);
      view.setInt32(base + 20, SEMOPM, true);
      view.setInt32(base + 24, SEMUME, true);
      view.setInt32(base + 28, usage ? r[1] : SEMUSZ, true);
      view.setInt32(base + 32, SEMVMX, true);
      view.setInt32(base + 36, usage ? r[2] : SEMAEM, true);
      cpu2.setReg(0, BigInt(Math.max(0, r[1] - 1)));
      return;
    }
    if (cmd === IPC_STAT || cmd === SEM_STAT || cmd === SEM_STAT_ANY) {
      const buf = Number(arg);
      if (!guestAccessible(buf, SEMID_DS_SIZE, true)) {
        fail(-14);
        return;
      }
      const byIndex = cmd !== IPC_STAT;
      const r = sysvSemCall({ op: byIndex ? "statidx" : "stat", id, uid, gid });
      if (r[0] < 0) {
        fail(r[0]);
        return;
      }
      writeSemidDs(view, buf, r);
      cpu2.setReg(0, byIndex ? BigInt(r[8]) : 0n);
      return;
    }
    if (cmd === IPC_SET) {
      const buf = Number(arg);
      if (!guestAccessible(buf, SEMID_DS_SIZE, false)) {
        fail(-14);
        return;
      }
      const base = buf + GUEST_MEM_BASE;
      const r = sysvSemCall({
        op: "set",
        id,
        uid,
        gid,
        nuid: view.getUint32(base + 4, true),
        ngid: view.getUint32(base + 8, true),
        mode: view.getUint32(base + 20, true)
      });
      fail(r[0]);
      return;
    }
    if (cmd === IPC_RMID) {
      fail(sysvSemCall({ op: "rm", id, uid, gid })[0]);
      return;
    }
    if (cmd === GETALL) {
      const buf = Number(arg);
      const r = sysvSemCall({ op: "getall", id, uid, gid });
      if (r[0] < 0) {
        fail(r[0]);
        return;
      }
      const off = r[1], len = r[2];
      if (!guestAccessible(buf, len, true)) {
        sysvSemCall({ op: "free", off, len });
        fail(-14);
        return;
      }
      bytes.set(new Uint8Array(shmArena, off, len), buf + GUEST_MEM_BASE);
      sysvSemCall({ op: "free", off, len });
      cpu2.setReg(0, 0n);
      return;
    }
    if (cmd === SETALL) {
      const buf = Number(arg);
      const st = sysvSemCall({ op: "nsems", id });
      if (st[0] < 0) {
        fail(st[0]);
        return;
      }
      const nsems = st[0];
      if (!guestAccessible(buf, nsems * 2, false)) {
        fail(-14);
        return;
      }
      const vals = [];
      for (let i = 0; i < nsems; i++) vals.push(view.getUint16(buf + GUEST_MEM_BASE + i * 2, true));
      fail(sysvSemCall({ op: "setall", id, vals, uid, gid })[0]);
      return;
    }
    if (cmd === SETVAL) {
      fail(sysvSemCall({ op: "setval", id, num, val: Number(BigInt.asIntN(32, arg)), uid, gid })[0]);
      return;
    }
    if (cmd === GETVAL || cmd === GETPID || cmd === GETNCNT || cmd === GETZCNT) {
      const what = cmd === GETVAL ? "val" : cmd === GETPID ? "pid" : cmd === GETNCNT ? "ncnt" : "zcnt";
      fail(sysvSemCall({ op: "getone", id, num, what, uid, gid })[0]);
      return;
    }
    fail(-22);
  }
  function sysvShmCall(msg) {
    Atomics.store(myMailbox, MBOX_READY, 0);
    self.postMessage({ type: "sysv_shm", ...msg });
    awaitMailbox("sysv_shm");
    const out = [];
    for (let i = 0; i < 13; i++) out.push(Atomics.load(myMailbox, MBOX_VALUE0 + i));
    return out;
  }
  function handleSysvShm(nr, cpu2, view) {
    const creds = syscalls.getCreds();
    if (nr === 194) {
      const SHM_HUGETLB = 2048;
      if (Number(BigInt.asIntN(32, cpu2.getReg(2))) & SHM_HUGETLB) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
        return;
      }
      const r = sysvShmCall({
        op: "get",
        key: Number(BigInt.asIntN(32, cpu2.getReg(0))),
        size: Number(cpu2.getReg(1)),
        flg: Number(cpu2.getReg(2)),
        uid: creds.euid,
        gid: creds.egid
      });
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
      return;
    }
    if (nr === 196) {
      const id2 = Number(BigInt.asIntN(32, cpu2.getReg(0)));
      const SHM_RDONLY = 4096, SHM_RND = 8192, SHMLBA = 4096;
      const flg = Number(BigInt.asIntN(32, cpu2.getReg(2)));
      let want = Number(cpu2.getReg(1));
      if (want !== 0) {
        if (flg & SHM_RND) want -= want % SHMLBA;
        else if (want % SHMLBA !== 0) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
          return;
        }
        if (want === 0) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
          return;
        }
      }
      const r = sysvShmCall({
        op: "attach",
        id: id2,
        rdonly: flg & SHM_RDONLY ? 1 : 0,
        uid: creds.euid,
        gid: creds.egid
      });
      if (r[0] < 0) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
        return;
      }
      const size = r[0];
      const prot = flg & SHM_RDONLY ? 1 : 3;
      const addr = syscalls.attachShared(memory, size, prot, `sysvshm:${id2}`, want);
      if (addr < 0) {
        sysvShmCall({ op: "detach", id: id2 });
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(addr)));
        return;
      }
      sysvAttached.set(addr, id2);
      cpu2.setReg(0, BigInt(addr));
      return;
    }
    if (nr === 197) {
      const addr = Number(cpu2.getReg(0));
      const id2 = sysvAttached.get(addr);
      const region = syscalls.sharedRegionAt(addr);
      if (id2 === void 0 || !region) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
        return;
      }
      releaseSharedMaps(addr, addr + region.len);
      syscalls.detachShared(addr, region.len);
      sysvAttached.delete(addr);
      sysvShmCall({ op: "detach", id: id2 });
      cpu2.setReg(0, 0n);
      return;
    }
    const id = Number(BigInt.asIntN(32, cpu2.getReg(0)));
    const cmd = Number(BigInt.asIntN(32, cpu2.getReg(1))) & ~256;
    const buf = Number(cpu2.getReg(2));
    const shmCreds = syscalls.getCreds();
    const IPC_RMID = 0, IPC_SET = 1, IPC_STAT = 2, SHM_LOCK = 11, SHM_UNLOCK = 12;
    if (cmd === IPC_RMID) {
      const r = sysvShmCall({ op: "rm", id, uid: shmCreds.euid, gid: shmCreds.egid });
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
      return;
    }
    const IPC_INFO = 3, SHM_INFO = 14;
    if (cmd === IPC_INFO || cmd === SHM_INFO) {
      if (buf === 0 || syscalls.ptrFaults(new Uint8Array(memory.buffer), buf, 48)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
      const r = sysvShmCall({ op: "info" });
      const b = buf + GUEST_MEM_BASE;
      new Uint8Array(memory.buffer).fill(0, b, b + 48);
      if (cmd === IPC_INFO) {
        const w64 = (lo, hi) => BigInt(hi >>> 0) << 32n | BigInt(lo >>> 0);
        view.setBigUint64(b + 0, w64(r[4], r[5]), true);
        view.setBigUint64(b + 8, 1n, true);
        view.setBigUint64(b + 16, BigInt(r[3]), true);
        view.setBigUint64(b + 24, BigInt(r[3]), true);
        view.setBigUint64(b + 32, w64(r[6], r[7]), true);
      } else {
        view.setInt32(b + 0, r[1], true);
        view.setBigUint64(b + 8, BigInt(r[2]), true);
        view.setBigUint64(b + 16, BigInt(r[2]), true);
        view.setBigUint64(b + 24, 0n, true);
        view.setBigUint64(b + 32, 0n, true);
        view.setBigUint64(b + 40, 0n, true);
      }
      cpu2.setReg(0, BigInt(Math.max(0, r[1] - 1)));
      return;
    }
    if (cmd === SHM_LOCK || cmd === SHM_UNLOCK) {
      const r = sysvShmCall({
        op: "lock",
        id,
        on: cmd === SHM_LOCK ? 1 : 0,
        uid: shmCreds.euid,
        gid: shmCreds.egid
      });
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
      return;
    }
    const SHM_STAT = 13, SHM_STAT_ANY = 15;
    if (cmd === IPC_STAT || cmd === SHM_STAT || cmd === SHM_STAT_ANY) {
      if (buf === 0 || syscalls.ptrFaults(new Uint8Array(memory.buffer), buf, 112)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
      const byIdx = cmd !== IPC_STAT;
      const r = sysvShmCall({ op: byIdx ? "statidx" : "stat", id, uid: shmCreds.euid, gid: shmCreds.egid });
      if (r[0] < 0) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
        return;
      }
      const [, key, uid, gid, mode, size, cpid, lpid, nattch, atime, dtime, ctime] = r;
      const b = buf + GUEST_MEM_BASE;
      for (let i = 0; i < 112; i += 8) view.setBigUint64(b + i, 0n, true);
      view.setInt32(b + 0, key, true);
      view.setUint32(b + 4, uid, true);
      view.setUint32(b + 8, gid, true);
      view.setUint32(b + 12, uid, true);
      view.setUint32(b + 16, gid, true);
      view.setUint32(b + 20, mode, true);
      view.setBigUint64(b + 48, BigInt(size), true);
      view.setBigInt64(b + 56, BigInt(atime), true);
      view.setBigInt64(b + 64, BigInt(dtime), true);
      view.setBigInt64(b + 72, BigInt(ctime), true);
      view.setInt32(b + 80, cpid, true);
      view.setInt32(b + 84, lpid, true);
      view.setBigUint64(b + 88, BigInt(nattch), true);
      cpu2.setReg(0, byIdx ? BigInt(r[12]) : 0n);
      return;
    }
    if (cmd === IPC_SET) {
      if (buf === 0 || syscalls.ptrFaults(new Uint8Array(memory.buffer), buf, 112)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
      const b = buf + GUEST_MEM_BASE;
      const r = sysvShmCall({
        op: "set",
        id,
        uid: shmCreds.euid,
        gid: shmCreds.egid,
        mode: view.getUint32(b + 20, true),
        newUid: view.getUint32(b + 4, true),
        newGid: view.getUint32(b + 8, true)
      });
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
      return;
    }
    cpu2.setReg(0, BigInt.asUintN(64, BigInt(-22)));
  }
  function handleKernelPipeIO(nr, cpu2, view) {
    const fd = Number(cpu2.getReg(0));
    const pipeInfo = kernelPipes.get(fd);
    const ring = new PipeRing(shmArena, pipeInfo.pipeOff);
    if ((nr === 63 || nr === 65) && pipeInfo.mode === "write") {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-9)));
      return;
    }
    if ((nr === 64 || nr === 66) && pipeInfo.mode === "read") {
      cpu2.setReg(0, BigInt.asUintN(64, BigInt(-9)));
      return;
    }
    if (nr === 57) {
      pipeRef(pipeInfo.pipeOff, pipeInfo.mode, -1);
      kernelPipes.delete(fd);
      syscalls.unregisterFd(fd);
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
      if (syscalls.isNonBlockingFd(fd)) {
        const n2 = ring.tryRead(dst);
        if (n2 === 0 && ring.writersOpen()) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(-11)));
          return;
        }
        if (n2 > 0) new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, n2).set(dst.subarray(0, n2));
        cpu2.setReg(0, BigInt(n2));
        return;
      }
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
        if (n2 > 0) notePipeReadable(pipeInfo.pipeOff);
        if (n2 === -1) {
          syscalls.raiseSignal(13);
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
      if (n > 0) notePipeReadable(pipeInfo.pipeOff);
      if (n === -1) {
        syscalls.raiseSignal(13);
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
    if (nr === 65) {
      const iovAddr = Number(cpu2.getReg(1));
      const iovcnt = Number(cpu2.getReg(2));
      const bytes = new Uint8Array(memory.buffer);
      if (iovcnt > 0 && badGuestPtr(bytes, iovAddr, iovcnt * 16)) {
        cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
      let total = 0;
      const segs = [];
      for (let i = 0; i < iovcnt; i++) {
        const base = Number(view.getBigUint64(iovAddr + i * 16 + GUEST_MEM_BASE, true));
        const len = Number(view.getBigUint64(iovAddr + i * 16 + 8 + GUEST_MEM_BASE, true));
        if (len === 0) continue;
        if (badGuestPtr(bytes, base, len)) {
          if (total === 0) {
            cpu2.setReg(0, BigInt.asUintN(64, BigInt(-14)));
            return;
          }
          break;
        }
        segs.push({ addr: base, len });
        total += len;
      }
      if (total === 0) {
        cpu2.setReg(0, 0n);
        return;
      }
      const dst = new Uint8Array(total);
      let n;
      if (syscalls.isNonBlockingFd(fd)) {
        n = ring.tryRead(dst);
        if (n === 0 && ring.writersOpen()) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(-11)));
          return;
        }
      } else {
        n = ring.read(dst, interruptibleWaiter);
        if (n === -1) {
          cpu2.setReg(0, BigInt.asUintN(64, BigInt(-4)));
          return;
        }
      }
      let off = 0;
      for (const sg of segs) {
        if (off >= n) break;
        const take = Math.min(sg.len, n - off);
        bytes.set(dst.subarray(off, off + take), sg.addr + GUEST_MEM_BASE);
        off += take;
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
        if (n > 0) notePipeReadable(pipeInfo.pipeOff);
        if (n === -1) {
          if (totalWritten === 0) {
            syscalls.raiseSignal(13);
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
        sleepingWait(ringI32, 0, wp, rem > 0 ? Math.min(1e3, rem) : 1e3);
        if (pollAndQueueSignals()) return;
      }
    }
  }
  function bytesToBase64Fast(u) {
    let out = "";
    const BLOCK = 32768;
    for (let i = 0; i < u.length; i += BLOCK) {
      out += String.fromCharCode.apply(null, u.subarray(i, i + BLOCK));
    }
    return btoa(out);
  }
  function readGuestFile(path) {
    const fdh = { fd: -1 };
    handleBridgeCall("fs_open", { path, flags: 0, mode: 0 }, (r) => {
      fdh.fd = r.ret ?? -1;
    });
    if (fdh.fd < 0) {
      const e = new Error(`readGuestFile: cannot open ${path} (ret ${fdh.fd})`);
      e.errno = -fdh.fd || 2;
      throw e;
    }
    const chunks = [];
    let total = 0;
    const CHUNK = BRIDGE_DATA_MAX;
    try {
      for (; ; ) {
        const h = { data: new Uint8Array(0) };
        handleBridgeCall("fs_read", { fd: fdh.fd, count: CHUNK }, (r) => {
          if (r.bytes && r.bytes.length) h.data = r.bytes;
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
    Atomics.store(myMailbox, MBOX_READY, 0);
    self.postMessage({ type: "bridge_call", id: 0, method, args });
    const bridgeStart = performance.now();
    let bridgeWarned = false;
    while (Atomics.load(myMailbox, MBOX_READY) === 0) {
      Atomics.wait(myMailbox, MBOX_READY, 0, 100);
      if (Atomics.load(myMailbox, MBOX_READY) === 0) {
        if (!bridgeWarned && performance.now() - bridgeStart > 2e4) {
          bridgeWarned = true;
          self.postMessage({
            type: "worker_error",
            detail: `pid ${myPid} has waited 20s for bridge '${method}'`
          });
        }
        const queued = sigqDrain(sigQueue, mySlot);
        for (const q of queued) syscalls.raiseSignal(q.signo, { pid: q.pid, uid: q.uid, code: q.code, value: q.value });
        if (queued.some((q) => q.signo === 2)) {
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
      const raw = new Uint8Array(dataLen);
      raw.set(new Uint8Array(kernelSAB, respOffset + dataOffset, dataLen));
      resp.bytes = raw;
      Object.defineProperty(resp, "data", {
        configurable: true,
        enumerable: true,
        get: () => bytesToBase64Fast(raw)
      });
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
    const result = timeoutMs < 0 ? sleepingWait(ringI32, 0, wp) : sleepingWait(ringI32, 0, wp, timeoutMs);
    return result === "timed-out" ? "timeout" : "data";
  }
  function postWrite(fd, data) {
    self.postMessage({ type: "write", fd, data: data.slice() });
  }
  function isNonLeaderThread() {
    return threaded && syscalls.tgid !== 0 && syscalls.tgid !== syscalls.currentPid;
  }
  function releaseVforkParent() {
    const base = procSlotOffset(mySlot) / 4;
    if (Atomics.exchange(procTable, base + PROC_VFORK, 0) === 1) {
      Atomics.notify(procTable, base + PROC_VFORK);
    }
  }
  function postExit(code, group = false) {
    releaseVforkParent();
    publishSyscall(-6);
    stamp("exit");
    reportStages();
    if (!pendingClone && !isNonLeaderThread()) {
      try {
        syscalls.closeNativeFds((fd) => handleBridgeCall("fs_close", { fd }, () => {
        }));
        syscalls.releaseEventfds();
      } catch {
      }
    }
    if (threaded && clearChildTid) {
      try {
        const t = futexRef(clearChildTid);
        Atomics.store(t.i32, t.idx, 0);
        futexLock(procTable);
        futexWake(procTable, t.keyLo, t.keyHi, 4294967295, 2147483647);
        futexUnlock(procTable);
      } catch {
      }
      clearChildTid = 0;
    }
    publishSyscall(-7);
    for (const [, info] of kernelPipes) pipeRef(info.pipeOff, info.mode, -1);
    kernelPipes.clear();
    for (const fd of [...unixSocks.keys()]) closeUnixSock(fd);
    publishCpuTime();
    publishSyscall(-8);
    const high = threaded ? Number(syscalls.getMmapBase()) + GUEST_MEM_BASE : 0;
    self.postMessage({
      type: "exit",
      pid: myPid,
      code,
      high,
      group,
      memBytes: memory.buffer.byteLength
    });
  }
})();
