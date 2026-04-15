import { WasmModuleBuilder } from './wasm-emit.js';
import {
  ValType, BLOCKTYPE_VOID, HALT_PC, SVC_PC,
  regOffset, vregOffset, SP_OFFSET, NZCV_OFFSET, ITER_OFFSET, TPIDR_OFFSET, PC_OFFSET, GUEST_MEM_BASE, V_OFFSET,
} from './types.js';
import type { DecodedInsn, ShiftType, CompiledBlock, CompiledRegion, CompiledBlockModule, CompiledRegionModule } from './types.js';
import type { BasicBlock, Region } from './block.js';

// ── Register local mapping ──────────────────────────────────────────────────
// WASM locals: 0-30 = X0-X30, 31 = SP, 32 = NZCV, 33+ = temporaries

const REG_COUNT = 33; // X0-X30 (31) + SP (1) + NZCV (1)

function regLocal(r: number): number {
  if (r >= 0 && r <= 30) return r;
  if (r === 31) return 31; // SP local
  throw new Error(`Invalid register ${r}`);
}
const SP_LOCAL = 31;
const NZCV_LOCAL = 32;

// ── Analyze which registers a block reads/writes ─────────────────────────────

interface RegUsage {
  reads: Set<number>;  // local indices read
  writes: Set<number>; // local indices written
}

function analyzeRegUsage(insns: DecodedInsn[]): RegUsage {
  const reads = new Set<number>();
  const writes = new Set<number>();

  for (const insn of insns) {
    switch (insn.kind) {
      case 'add_imm': case 'sub_imm':
        if (insn.rn !== 31) reads.add(regLocal(insn.rn));
        else reads.add(SP_LOCAL); // ADD/SUB imm uses SP for Rn=31
        if (insn.rd === 31) {
          if (!insn.setFlags) writes.add(SP_LOCAL); // Rd=31 without flags → SP
          // Rd=31 with flags → XZR (CMP/CMN), no write needed
        } else {
          writes.add(regLocal(insn.rd));
        }
        if (insn.setFlags) writes.add(NZCV_LOCAL);
        break;
      case 'add_reg': case 'sub_reg':
        addRead(reads, insn.rn); addRead(reads, insn.rm);
        addWrite(writes, insn.rd);
        if (insn.setFlags) writes.add(NZCV_LOCAL);
        break;
      case 'adc': case 'sbc':
        addRead(reads, insn.rn); addRead(reads, insn.rm);
        reads.add(NZCV_LOCAL);  // reads carry flag
        addWrite(writes, insn.rd);
        if (insn.setFlags) writes.add(NZCV_LOCAL);
        break;
      case 'orr_reg':
        addRead(reads, insn.rn); addRead(reads, insn.rm);
        addWrite(writes, insn.rd);
        break;
      case 'eor_reg':
        addRead(reads, insn.rn); addRead(reads, insn.rm);
        addWrite(writes, insn.rd);
        break;
      case 'and_reg':
        addRead(reads, insn.rn); addRead(reads, insn.rm);
        addWrite(writes, insn.rd);
        if (insn.setFlags) writes.add(NZCV_LOCAL);
        break;
      case 'and_imm':
        // For logical immediates, Rn=31 is XZR (not SP)
        addRead(reads, insn.rn);
        if (insn.setFlags) {
          // ANDS: Rd=31 is XZR
          addWrite(writes, insn.rd);
          writes.add(NZCV_LOCAL);
        } else {
          // AND non-flag: Rd=31 is SP
          if (insn.rd === 31) writes.add(SP_LOCAL); else addWrite(writes, insn.rd);
        }
        break;
      case 'orr_imm': case 'eor_imm':
        // Rn=31 is XZR, Rd=31 is SP
        addRead(reads, insn.rn);
        if (insn.rd === 31) writes.add(SP_LOCAL); else addWrite(writes, insn.rd);
        break;
      case 'movz':
        addWrite(writes, insn.rd);
        break;
      case 'movk':
        addRead(reads, insn.rd); // MOVK reads then modifies
        addWrite(writes, insn.rd);
        break;
      case 'movn':
        addWrite(writes, insn.rd);
        break;
      case 'madd': case 'msub':
      case 'smaddl': case 'smsubl': case 'umaddl': case 'umsubl':
        addRead(reads, insn.rn); addRead(reads, insn.rm); addRead(reads, insn.ra);
        addWrite(writes, insn.rd);
        break;
      case 'sdiv': case 'udiv':
        addRead(reads, insn.rn); addRead(reads, insn.rm);
        addWrite(writes, insn.rd);
        break;
      case 'csel':
        addRead(reads, insn.rn); addRead(reads, insn.rm);
        reads.add(NZCV_LOCAL);
        addWrite(writes, insn.rd);
        break;
      case 'sbfm': case 'ubfm':
        addRead(reads, insn.rn);
        addWrite(writes, insn.rd);
        break;
      case 'bfm':
        addRead(reads, insn.rn);
        addRead(reads, insn.rd); // BFM merges into Rd
        addWrite(writes, insn.rd);
        break;
      case 'b_cond':
        reads.add(NZCV_LOCAL);
        break;
      case 'cbz': case 'cbnz':
        addRead(reads, insn.rt);
        break;
      case 'bl': case 'blr':
        writes.add(regLocal(30)); // X30 = link register
        if (insn.kind === 'blr') addRead(reads, insn.rn);
        break;
      case 'br':
        addRead(reads, insn.rn);
        break;
      case 'ret':
        addRead(reads, insn.rn);
        break;
      case 'ldr_imm':
        if (insn.rn !== 31) addRead(reads, insn.rn); else reads.add(SP_LOCAL);
        addWrite(writes, insn.rt);
        if (insn.mode !== 'offset') {
          if (insn.rn !== 31) addWrite(writes, insn.rn); else writes.add(SP_LOCAL);
        }
        break;
      case 'str_imm':
        if (insn.rn !== 31) addRead(reads, insn.rn); else reads.add(SP_LOCAL);
        addRead(reads, insn.rt);
        if (insn.mode !== 'offset') {
          if (insn.rn !== 31) addWrite(writes, insn.rn); else writes.add(SP_LOCAL);
        }
        break;
      case 'ldrb_imm': case 'ldrh_imm': case 'ldrsw_imm':
        if (insn.rn !== 31) addRead(reads, insn.rn); else reads.add(SP_LOCAL);
        addWrite(writes, insn.rt);
        if (insn.mode !== 'offset') {
          if (insn.rn !== 31) addWrite(writes, insn.rn); else writes.add(SP_LOCAL);
        }
        break;
      case 'ldrsb_imm': case 'ldrsh_imm':
        if (insn.rn !== 31) addRead(reads, insn.rn); else reads.add(SP_LOCAL);
        addWrite(writes, insn.rt);
        if (insn.mode !== 'offset') {
          if (insn.rn !== 31) addWrite(writes, insn.rn); else writes.add(SP_LOCAL);
        }
        break;
      case 'strb_imm': case 'strh_imm':
        if (insn.rn !== 31) addRead(reads, insn.rn); else reads.add(SP_LOCAL);
        addRead(reads, insn.rt);
        if (insn.mode !== 'offset') {
          if (insn.rn !== 31) addWrite(writes, insn.rn); else writes.add(SP_LOCAL);
        }
        break;
      case 'ldr_reg':
        if (insn.rn !== 31) addRead(reads, insn.rn); else reads.add(SP_LOCAL);
        addRead(reads, insn.rm);
        addWrite(writes, insn.rt);
        break;
      case 'str_reg':
        if (insn.rn !== 31) addRead(reads, insn.rn); else reads.add(SP_LOCAL);
        addRead(reads, insn.rm);
        addRead(reads, insn.rt);
        break;
      case 'ldp':
        if (insn.rn !== 31) addRead(reads, insn.rn); else reads.add(SP_LOCAL);
        addWrite(writes, insn.rt);
        addWrite(writes, insn.rt2);
        if (insn.mode !== 'offset') {
          if (insn.rn !== 31) addWrite(writes, insn.rn); else writes.add(SP_LOCAL);
        }
        break;
      case 'stp':
        if (insn.rn !== 31) addRead(reads, insn.rn); else reads.add(SP_LOCAL);
        addRead(reads, insn.rt);
        addRead(reads, insn.rt2);
        if (insn.mode !== 'offset') {
          if (insn.rn !== 31) addWrite(writes, insn.rn); else writes.add(SP_LOCAL);
        }
        break;
      case 'adr': case 'adrp':
        addWrite(writes, insn.rd);
        break;
      case 'smulh': case 'umulh':
        addRead(reads, insn.rn); addRead(reads, insn.rm);
        addWrite(writes, insn.rd);
        break;
      case 'tbz': case 'tbnz':
        addRead(reads, insn.rt);
        break;
      case 'clz': case 'cls': case 'rbit': case 'rev': case 'rev16':
        addRead(reads, insn.rn);
        addWrite(writes, insn.rd);
        break;
      case 'rev32':
        addRead(reads, insn.rn);
        addWrite(writes, insn.rd);
        break;
      case 'extr':
        addRead(reads, insn.rn); addRead(reads, insn.rm);
        addWrite(writes, insn.rd);
        break;
      case 'ccmp': case 'ccmn':
        addRead(reads, insn.rn);
        if (!insn.isImm) addRead(reads, insn.rm_imm);
        reads.add(NZCV_LOCAL);
        writes.add(NZCV_LOCAL);
        break;
      case 'ldr_lit': case 'ldrsw_lit':
        addWrite(writes, insn.rt);
        break;
      case 'ldrb_reg': case 'ldrh_reg': case 'ldrsw_reg':
        if (insn.rn !== 31) addRead(reads, insn.rn); else reads.add(SP_LOCAL);
        addRead(reads, insn.rm);
        addWrite(writes, insn.rt);
        break;
      case 'ldrsb_reg': case 'ldrsh_reg':
        if (insn.rn !== 31) addRead(reads, insn.rn); else reads.add(SP_LOCAL);
        addRead(reads, insn.rm);
        addWrite(writes, insn.rt);
        break;
      case 'strb_reg': case 'strh_reg':
        if (insn.rn !== 31) addRead(reads, insn.rn); else reads.add(SP_LOCAL);
        addRead(reads, insn.rm);
        addRead(reads, insn.rt);
        break;
      case 'ldxr':
        if (insn.rn !== 31) addRead(reads, insn.rn); else reads.add(SP_LOCAL);
        addWrite(writes, insn.rt);
        break;
      case 'stxr':
        if (insn.rn !== 31) addRead(reads, insn.rn); else reads.add(SP_LOCAL);
        addRead(reads, insn.rt);
        addWrite(writes, insn.rs);
        break;
      case 'mrs':
        addWrite(writes, insn.rt);
        if (insn.sysreg === 'nzcv') reads.add(NZCV_LOCAL);
        break;
      case 'msr':
        addRead(reads, insn.rt);
        if (insn.sysreg === 'nzcv') writes.add(NZCV_LOCAL);
        break;
      case 'add_ext': case 'sub_ext':
        if (insn.rn !== 31) addRead(reads, insn.rn); else reads.add(SP_LOCAL);
        addRead(reads, insn.rm);
        if (insn.setFlags) { reads.add(NZCV_LOCAL); writes.add(NZCV_LOCAL); }
        if (insn.rd === 31) { if (!insn.setFlags) writes.add(SP_LOCAL); }
        else addWrite(writes, insn.rd);
        break;
      // Variable shifts
      case 'lslv': case 'lsrv': case 'asrv': case 'rorv':
        addRead(reads, insn.rn);
        addRead(reads, insn.rm);
        addWrite(writes, insn.rd);
        break;
      // SIMD load/store: reads base register (SP or GPR), may write back
      case 'str_simd': case 'ldr_simd':
        if (insn.rn === 31) reads.add(SP_LOCAL); else addRead(reads, insn.rn);
        if (insn.mode !== 'offset') {
          if (insn.rn === 31) writes.add(SP_LOCAL); else addWrite(writes, insn.rn);
        }
        break;
      case 'stp_simd': case 'ldp_simd':
        if (insn.rn === 31) reads.add(SP_LOCAL); else addRead(reads, insn.rn);
        if (insn.mode !== 'offset') {
          if (insn.rn === 31) writes.add(SP_LOCAL); else addWrite(writes, insn.rn);
        }
        break;
      // SIMD data ops: read/write V regs directly from memory, may read GPRs
      case 'simd_dup_gpr':
        addRead(reads, insn.rn);
        break;
      case 'simd_umov':
        addWrite(writes, insn.rd);
        break;
      case 'simd_ins_gpr':
        addRead(reads, insn.rn);
        break;
      case 'simd_dup_elem': case 'simd_movi':
        // V registers are in linear memory, no GPR involvement
        break;
      // FMOV GPR → FP: reads GPR
      case 'fmov_to_fp': case 'fmov_top_to_fp':
        addRead(reads, insn.rn);
        break;
      // FMOV FP → GPR: writes GPR
      case 'fmov_from_fp': case 'fmov_top_from_fp':
        addWrite(writes, insn.rd);
        break;
      // FP-only / SIMD-only: no GPR involvement
      case 'fmov_fp': case 'fmul': case 'fdiv': case 'fadd': case 'fsub':
      case 'fabs': case 'fneg': case 'fcvt': case 'fmov_imm':
      case 'fmadd': case 'fmsub': case 'fnmadd': case 'fnmsub':
      case 'simd_orr_reg': case 'simd_and_reg': case 'simd_bic_reg': case 'simd_eor_reg':
      case 'simd_add': case 'simd_sub':
      case 'simd_cmeq_reg': case 'simd_cmgt_reg': case 'simd_cmhi_reg':
      case 'simd_cmge_reg': case 'simd_cmhs_reg':
      case 'simd_umaxp': case 'simd_addp':
      case 'simd_tbl':
      case 'simd_zip1': case 'simd_zip2': case 'simd_uzp1': case 'simd_uzp2':
      case 'simd_trn1': case 'simd_trn2':
      case 'simd_not': case 'simd_neg': case 'simd_abs':
      case 'simd_cnt': case 'simd_cls_vec': case 'simd_clz_vec':
      case 'simd_sshll': case 'simd_ushll':
      case 'simd_scvtf': case 'simd_ucvtf':
      case 'simd_fcvtzs': case 'simd_fcvtzu':
        break;
      // FP compare: writes NZCV
      case 'fcmp':
        writes.add(NZCV_LOCAL);
        break;
      // FP int conversions: read/write GPRs
      case 'scvtf_int': case 'ucvtf_int':
        addRead(reads, insn.rn);
        break;
      case 'fcvtzs_int': case 'fcvtzu_int':
        addWrite(writes, insn.rd);
        break;
      default:
        break;
    }
  }

  return { reads, writes };
}

function addRead(set: Set<number>, r: number): void {
  if (r !== 31) set.add(regLocal(r)); // XZR reads don't need a local
}

function addWrite(set: Set<number>, r: number): void {
  if (r !== 31) set.add(regLocal(r)); // XZR writes are discarded
}

// ── Helper function imports for instructions that need JS helpers ────────────

interface HelperImports {
  smulhIdx: number;
  umulhIdx: number;
  // FP helpers (i64 bit-pattern ↔ f64 arithmetic)
  f64_mulIdx: number;
  f64_divIdx: number;
  f64_addIdx: number;
  f64_subIdx: number;
  f64_absIdx: number;
  f64_negIdx: number;
  f64_from_s64Idx: number;
  f64_from_u64Idx: number;
  f64_from_s32Idx: number;
  f64_from_u32Idx: number;
  f64_to_s64Idx: number;
  f64_to_u64Idx: number;
  f64_to_s32Idx: number;
  f64_to_u32Idx: number;
  f32_to_f64Idx: number;
  f64_to_f32Idx: number;
  f64_cmpIdx: number;
  simd_opIdx: number;
}

function setupHelperImports(b: WasmModuleBuilder): HelperImports {
  const helperType = b.addFuncType([ValType.I64, ValType.I64], [ValType.I64]);
  const smulhIdx = b.addFunctionImport('env', 'smulh', helperType);
  const umulhIdx = b.addFunctionImport('env', 'umulh', helperType);

  // FP 2-operand helpers: (i64, i64) -> i64
  const f64_mulIdx = b.addFunctionImport('env', 'f64_mul', helperType);
  const f64_divIdx = b.addFunctionImport('env', 'f64_div', helperType);
  const f64_addIdx = b.addFunctionImport('env', 'f64_add', helperType);
  const f64_subIdx = b.addFunctionImport('env', 'f64_sub', helperType);

  // FP 1-operand helpers: (i64, i64) -> i64 (second arg unused, kept for uniform type)
  const unaryType = b.addFuncType([ValType.I64], [ValType.I64]);
  const f64_absIdx = b.addFunctionImport('env', 'f64_abs', unaryType);
  const f64_negIdx = b.addFunctionImport('env', 'f64_neg', unaryType);

  // Int→FP conversions: (i64) -> i64  (input: int bits, output: FP bits)
  const f64_from_s64Idx = b.addFunctionImport('env', 'f64_from_s64', unaryType);
  const f64_from_u64Idx = b.addFunctionImport('env', 'f64_from_u64', unaryType);
  const f64_from_s32Idx = b.addFunctionImport('env', 'f64_from_s32', unaryType);
  const f64_from_u32Idx = b.addFunctionImport('env', 'f64_from_u32', unaryType);

  // FP→Int conversions: (i64) -> i64  (input: FP bits, output: int bits)
  const f64_to_s64Idx = b.addFunctionImport('env', 'f64_to_s64', unaryType);
  const f64_to_u64Idx = b.addFunctionImport('env', 'f64_to_u64', unaryType);
  const f64_to_s32Idx = b.addFunctionImport('env', 'f64_to_s32', unaryType);
  const f64_to_u32Idx = b.addFunctionImport('env', 'f64_to_u32', unaryType);

  // Precision conversions: (i64) -> i64  (f32 bits → f64 bits and vice versa)
  const f32_to_f64Idx = b.addFunctionImport('env', 'f32_to_f64', unaryType);
  const f64_to_f32Idx = b.addFunctionImport('env', 'f64_to_f32', unaryType);

  // FP compare: (i64, i64) -> i64 (returns NZCV bits)
  const f64_cmpIdx = b.addFunctionImport('env', 'f64_cmp', helperType);

  // SIMD generic operation: (opcode_packed, extra_packed) -> 0
  // Operates on V registers in linear memory directly
  const simd_opIdx = b.addFunctionImport('env', 'simd_op', helperType);

  return {
    smulhIdx, umulhIdx,
    f64_mulIdx, f64_divIdx, f64_addIdx, f64_subIdx,
    f64_absIdx, f64_negIdx,
    f64_from_s64Idx, f64_from_u64Idx, f64_from_s32Idx, f64_from_u32Idx,
    f64_to_s64Idx, f64_to_u64Idx, f64_to_s32Idx, f64_to_u32Idx,
    f32_to_f64Idx, f64_to_f32Idx,
    f64_cmpIdx,
    simd_opIdx,
  };
}

// Shared Float64Array + Uint8Array for bit reinterpretation
const _fpBuf = new ArrayBuffer(8);
const _f64View = new Float64Array(_fpBuf);
const _u8View = new Uint8Array(_fpBuf);
const _f32View = new Float32Array(_fpBuf);
const _i64View = new BigInt64Array(_fpBuf);
const _u64View = new BigUint64Array(_fpBuf);

function f64FromBits(bits: bigint): number {
  _u64View[0] = BigInt.asUintN(64, bits);
  return _f64View[0];
}
function f64ToBits(val: number): bigint {
  _f64View[0] = val;
  return _u64View[0];
}
function f32FromBits(bits: number): number {
  new DataView(_fpBuf).setUint32(0, bits, true);
  return _f32View[0];
}
function f32ToBits(val: number): number {
  _f32View[0] = val;
  return new DataView(_fpBuf).getUint32(0, true);
}

// JS implementations for 128-bit multiply high halves
const helperFunctions: Record<string, Function> = {
  smulh(a: bigint, b: bigint): bigint {
    const sa = BigInt.asIntN(64, a);
    const sb = BigInt.asIntN(64, b);
    return BigInt.asUintN(64, (sa * sb) >> 64n);
  },
  umulh(a: bigint, b: bigint): bigint {
    const ua = BigInt.asUintN(64, a);
    const ub = BigInt.asUintN(64, b);
    return BigInt.asUintN(64, (ua * ub) >> 64n);
  },
  // FP 2-source: operate on f64 bit patterns (or f32 widened to f64 bits)
  f64_mul(a: bigint, b: bigint): bigint { return f64ToBits(f64FromBits(a) * f64FromBits(b)); },
  f64_div(a: bigint, b: bigint): bigint { return f64ToBits(f64FromBits(a) / f64FromBits(b)); },
  f64_add(a: bigint, b: bigint): bigint { return f64ToBits(f64FromBits(a) + f64FromBits(b)); },
  f64_sub(a: bigint, b: bigint): bigint { return f64ToBits(f64FromBits(a) - f64FromBits(b)); },
  // FP 1-source
  f64_abs(a: bigint): bigint { return f64ToBits(Math.abs(f64FromBits(a))); },
  f64_neg(a: bigint): bigint { return f64ToBits(-f64FromBits(a)); },
  // Int → FP
  f64_from_s64(a: bigint): bigint { return f64ToBits(Number(BigInt.asIntN(64, a))); },
  f64_from_u64(a: bigint): bigint { return f64ToBits(Number(BigInt.asUintN(64, a))); },
  f64_from_s32(a: bigint): bigint { return f64ToBits(Number(BigInt.asIntN(32, a))); },
  f64_from_u32(a: bigint): bigint { return f64ToBits(Number(BigInt.asUintN(32, a))); },
  // FP → Int (round toward zero)
  f64_to_s64(a: bigint): bigint { return BigInt.asUintN(64, BigInt(Math.trunc(f64FromBits(a)))); },
  f64_to_u64(a: bigint): bigint { const v = Math.trunc(f64FromBits(a)); return v < 0 ? 0n : BigInt.asUintN(64, BigInt(v)); },
  f64_to_s32(a: bigint): bigint { return BigInt.asUintN(64, BigInt(Math.trunc(f64FromBits(a))) & 0xFFFFFFFFn); },
  f64_to_u32(a: bigint): bigint { const v = Math.trunc(f64FromBits(a)); return v < 0 ? 0n : BigInt(v) & 0xFFFFFFFFn; },
  // Precision conversions
  f32_to_f64(a: bigint): bigint {
    // a has f32 bits in low 32; widen to f64
    const f = f32FromBits(Number(a & 0xFFFFFFFFn));
    return f64ToBits(f);
  },
  f64_to_f32(a: bigint): bigint {
    // a has f64 bits; narrow to f32 bits
    const f = f64FromBits(a);
    return BigInt(f32ToBits(Math.fround(f)));
  },
  // FP compare: returns ARM NZCV bits (N=bit3, Z=bit2, C=bit1, V=bit0)
  f64_cmp(a: bigint, b: bigint): bigint {
    const fa = f64FromBits(a);
    const fb = f64FromBits(b);
    if (isNaN(fa) || isNaN(fb)) return 3n;  // Unordered: C=1, V=1
    if (fa === fb) return 6n;                // Equal:     Z=1, C=1
    if (fa < fb) return 8n;                  // Less:      N=1
    return 2n;                               // Greater:   C=1
  },
};

// ── SIMD operation helper (per-memory closure) ──────────────────────────────

// Operation codes for simd_op
const SIMD_ADD = 1, SIMD_SUB = 2;
const SIMD_CMEQ = 10, SIMD_CMGT = 11, SIMD_CMHI = 12, SIMD_CMGE = 13, SIMD_CMHS = 14;
const SIMD_UMAXP = 20, SIMD_ADDP = 21;
const SIMD_TBL = 30;
const SIMD_ZIP1 = 40, SIMD_ZIP2 = 41, SIMD_UZP1 = 42, SIMD_UZP2 = 43;
const SIMD_TRN1 = 44, SIMD_TRN2 = 45;

function createSimdOp(memory: WebAssembly.Memory) {
  return function simd_op(packed: bigint, _extra: bigint): bigint {
    const op = Number((packed >> 56n) & 0xFFn);
    const Q = Number((packed >> 48n) & 0xFFn);
    const size = Number((packed >> 40n) & 0xFFn);
    const extra = Number((packed >> 32n) & 0xFFn);
    const rd = Number((packed >> 24n) & 0xFFn);
    const rn = Number((packed >> 16n) & 0xFFn);
    const rm = Number((packed >> 8n) & 0xFFn);

    const buf = new Uint8Array(memory.buffer);
    const view = new DataView(memory.buffer);
    const vn = V_OFFSET + rn * 16;
    const vm = V_OFFSET + rm * 16;
    const vd = V_OFFSET + rd * 16;
    const bytes = Q ? 16 : 8;
    const esize = 1 << size; // element size in bytes

    // Read source vectors
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
          if (esize === 1) dView.setUint8(i, (aView.getUint8(i) + bView.getUint8(i)) & 0xFF);
          else if (esize === 2) dView.setUint16(i*2, (aView.getUint16(i*2, true) + bView.getUint16(i*2, true)) & 0xFFFF, true);
          else if (esize === 4) dView.setUint32(i*4, (aView.getUint32(i*4, true) + bView.getUint32(i*4, true)) >>> 0, true);
          else dView.setBigUint64(i*8, BigInt.asUintN(64, aView.getBigUint64(i*8, true) + bView.getBigUint64(i*8, true)), true);
        }
        break;
      case SIMD_SUB:
        for (let i = 0; i < elems; i++) {
          if (esize === 1) dView.setUint8(i, (aView.getUint8(i) - bView.getUint8(i)) & 0xFF);
          else if (esize === 2) dView.setUint16(i*2, (aView.getUint16(i*2, true) - bView.getUint16(i*2, true)) & 0xFFFF, true);
          else if (esize === 4) dView.setUint32(i*4, (aView.getUint32(i*4, true) - bView.getUint32(i*4, true)) >>> 0, true);
          else dView.setBigUint64(i*8, BigInt.asUintN(64, aView.getBigUint64(i*8, true) - bView.getBigUint64(i*8, true)), true);
        }
        break;
      case SIMD_CMEQ:
        for (let i = 0; i < elems; i++) {
          let eq: boolean;
          if (esize === 1) eq = aView.getUint8(i) === bView.getUint8(i);
          else if (esize === 2) eq = aView.getUint16(i*2, true) === bView.getUint16(i*2, true);
          else if (esize === 4) eq = aView.getUint32(i*4, true) === bView.getUint32(i*4, true);
          else eq = aView.getBigUint64(i*8, true) === bView.getBigUint64(i*8, true);
          // Result is all-ones if equal, all-zeros if not
          if (esize === 1) dView.setUint8(i, eq ? 0xFF : 0);
          else if (esize === 2) dView.setUint16(i*2, eq ? 0xFFFF : 0, true);
          else if (esize === 4) dView.setUint32(i*4, eq ? 0xFFFFFFFF : 0, true);
          else dView.setBigUint64(i*8, eq ? 0xFFFFFFFFFFFFFFFFn : 0n, true);
        }
        break;
      case SIMD_CMGT: // signed greater than
        for (let i = 0; i < elems; i++) {
          let gt: boolean;
          if (esize === 1) gt = aView.getInt8(i) > bView.getInt8(i);
          else if (esize === 2) gt = aView.getInt16(i*2, true) > bView.getInt16(i*2, true);
          else if (esize === 4) gt = aView.getInt32(i*4, true) > bView.getInt32(i*4, true);
          else gt = aView.getBigInt64(i*8, true) > bView.getBigInt64(i*8, true);
          if (esize <= 4) dView.setUint32(i * esize, gt ? (esize === 1 ? 0xFF : esize === 2 ? 0xFFFF : 0xFFFFFFFF) : 0, true);
          else dView.setBigUint64(i*8, gt ? 0xFFFFFFFFFFFFFFFFn : 0n, true);
        }
        break;
      case SIMD_CMHI: // unsigned higher
        for (let i = 0; i < elems; i++) {
          let hi: boolean;
          if (esize === 1) hi = aView.getUint8(i) > bView.getUint8(i);
          else if (esize === 2) hi = aView.getUint16(i*2, true) > bView.getUint16(i*2, true);
          else if (esize === 4) hi = aView.getUint32(i*4, true) > bView.getUint32(i*4, true);
          else hi = aView.getBigUint64(i*8, true) > bView.getBigUint64(i*8, true);
          if (esize <= 4) { const mask = esize === 1 ? 0xFF : esize === 2 ? 0xFFFF : 0xFFFFFFFF; if (esize === 1) dView.setUint8(i, hi ? 0xFF : 0); else if (esize === 2) dView.setUint16(i*2, hi ? 0xFFFF : 0, true); else dView.setUint32(i*4, hi ? 0xFFFFFFFF : 0, true); }
          else dView.setBigUint64(i*8, hi ? 0xFFFFFFFFFFFFFFFFn : 0n, true);
        }
        break;
      case SIMD_CMGE: // signed >=
      case SIMD_CMHS: // unsigned >=
        for (let i = 0; i < elems; i++) {
          let ge: boolean;
          if (op === SIMD_CMGE) {
            if (esize === 1) ge = aView.getInt8(i) >= bView.getInt8(i);
            else if (esize === 2) ge = aView.getInt16(i*2, true) >= bView.getInt16(i*2, true);
            else if (esize === 4) ge = aView.getInt32(i*4, true) >= bView.getInt32(i*4, true);
            else ge = aView.getBigInt64(i*8, true) >= bView.getBigInt64(i*8, true);
          } else {
            if (esize === 1) ge = aView.getUint8(i) >= bView.getUint8(i);
            else if (esize === 2) ge = aView.getUint16(i*2, true) >= bView.getUint16(i*2, true);
            else if (esize === 4) ge = aView.getUint32(i*4, true) >= bView.getUint32(i*4, true);
            else ge = aView.getBigUint64(i*8, true) >= bView.getBigUint64(i*8, true);
          }
          if (esize === 1) dView.setUint8(i, ge ? 0xFF : 0);
          else if (esize === 2) dView.setUint16(i*2, ge ? 0xFFFF : 0, true);
          else if (esize === 4) dView.setUint32(i*4, ge ? 0xFFFFFFFF : 0, true);
          else dView.setBigUint64(i*8, ge ? 0xFFFFFFFFFFFFFFFFn : 0n, true);
        }
        break;
      case SIMD_UMAXP: { // unsigned max pairwise
        const total = elems * 2; // concat Vn:Vm
        const all = new Uint8Array(32);
        all.set(a.subarray(0, bytes), 0);
        all.set(b_.subarray(0, bytes), bytes);
        const allView = new DataView(all.buffer);
        for (let i = 0; i < elems; i++) {
          const off1 = i * 2 * esize;
          const off2 = off1 + esize;
          let v1: number | bigint, v2: number | bigint, mx: number | bigint;
          if (esize === 1) { v1 = allView.getUint8(off1); v2 = allView.getUint8(off2); mx = v1 > v2 ? v1 : v2; dView.setUint8(i, mx); }
          else if (esize === 2) { v1 = allView.getUint16(off1, true); v2 = allView.getUint16(off2, true); mx = v1 > v2 ? v1 : v2; dView.setUint16(i*2, mx, true); }
          else if (esize === 4) { v1 = allView.getUint32(off1, true); v2 = allView.getUint32(off2, true); mx = v1 > v2 ? v1 : v2; dView.setUint32(i*4, mx, true); }
          else { v1 = allView.getBigUint64(off1, true); v2 = allView.getBigUint64(off2, true); mx = v1 > v2 ? v1 : v2; dView.setBigUint64(i*8, mx, true); }
        }
        break;
      }
      case SIMD_ADDP: { // add pairwise
        const all = new Uint8Array(32);
        all.set(a.subarray(0, bytes), 0);
        all.set(b_.subarray(0, bytes), bytes);
        const allView = new DataView(all.buffer);
        for (let i = 0; i < elems; i++) {
          const off1 = i * 2 * esize;
          const off2 = off1 + esize;
          if (esize === 1) dView.setUint8(i, (allView.getUint8(off1) + allView.getUint8(off2)) & 0xFF);
          else if (esize === 2) dView.setUint16(i*2, (allView.getUint16(off1, true) + allView.getUint16(off2, true)) & 0xFFFF, true);
          else if (esize === 4) dView.setUint32(i*4, (allView.getUint32(off1, true) + allView.getUint32(off2, true)) >>> 0, true);
          else dView.setBigUint64(i*8, BigInt.asUintN(64, allView.getBigUint64(off1, true) + allView.getBigUint64(off2, true)), true);
        }
        break;
      }
      case SIMD_TBL: { // table lookup
        const len = extra; // number of consecutive table registers (0-3 = 1-4 regs)
        const tableSize = (len + 1) * 16;
        const table = new Uint8Array(tableSize);
        for (let r = 0; r <= len; r++) {
          const vr = V_OFFSET + ((rn + r) & 31) * 16;
          table.set(buf.subarray(vr, vr + 16), r * 16);
        }
        for (let i = 0; i < bytes; i++) {
          const idx = b_[i];
          dst[i] = idx < tableSize ? table[idx] : 0;
        }
        break;
      }
      case SIMD_ZIP1: case SIMD_ZIP2: { // interleave
        const half = op === SIMD_ZIP1 ? 0 : elems / 2;
        for (let i = 0; i < elems / 2; i++) {
          const srcIdx = half + i;
          const dstIdx = i * 2;
          if (esize === 1) { dst[dstIdx] = a[srcIdx]; dst[dstIdx + 1] = b_[srcIdx]; }
          else if (esize === 2) { dView.setUint16(dstIdx*2, aView.getUint16(srcIdx*2, true), true); dView.setUint16((dstIdx+1)*2, bView.getUint16(srcIdx*2, true), true); }
          else if (esize === 4) { dView.setUint32(dstIdx*4, aView.getUint32(srcIdx*4, true), true); dView.setUint32((dstIdx+1)*4, bView.getUint32(srcIdx*4, true), true); }
          else { dView.setBigUint64(dstIdx*8, aView.getBigUint64(srcIdx*8, true), true); dView.setBigUint64((dstIdx+1)*8, bView.getBigUint64(srcIdx*8, true), true); }
        }
        break;
      }
      case SIMD_UZP1: case SIMD_UZP2: { // de-interleave
        const phase = op === SIMD_UZP1 ? 0 : 1;
        for (let i = 0; i < elems / 2; i++) {
          const srcIdx = i * 2 + phase;
          if (esize === 1) { dst[i] = a[srcIdx]; dst[i + elems/2] = b_[srcIdx]; }
          else if (esize === 2) { dView.setUint16(i*2, aView.getUint16(srcIdx*2, true), true); dView.setUint16((i+elems/2)*2, bView.getUint16(srcIdx*2, true), true); }
          else if (esize === 4) { dView.setUint32(i*4, aView.getUint32(srcIdx*4, true), true); dView.setUint32((i+elems/2)*4, bView.getUint32(srcIdx*4, true), true); }
          else { dView.setBigUint64(i*8, aView.getBigUint64(srcIdx*8, true), true); dView.setBigUint64((i+elems/2)*8, bView.getBigUint64(srcIdx*8, true), true); }
        }
        break;
      }
      case SIMD_TRN1: case SIMD_TRN2: { // transpose
        const phase = op === SIMD_TRN1 ? 0 : 1;
        for (let i = 0; i < elems / 2; i++) {
          const evenIdx = i * 2;
          if (esize === 1) { dst[evenIdx] = a[evenIdx + phase]; dst[evenIdx + 1] = b_[evenIdx + phase]; }
          else if (esize === 2) { dView.setUint16(evenIdx*2, aView.getUint16((evenIdx+phase)*2, true), true); dView.setUint16((evenIdx+1)*2, bView.getUint16((evenIdx+phase)*2, true), true); }
          else if (esize === 4) { dView.setUint32(evenIdx*4, aView.getUint32((evenIdx+phase)*4, true), true); dView.setUint32((evenIdx+1)*4, bView.getUint32((evenIdx+phase)*4, true), true); }
          else { dView.setBigUint64(evenIdx*8, aView.getBigUint64((evenIdx+phase)*8, true), true); dView.setBigUint64((evenIdx+1)*8, bView.getBigUint64((evenIdx+phase)*8, true), true); }
        }
        break;
      }
    }

    // Write result
    buf.set(dst.subarray(0, bytes), vd);
    if (!Q) { // Zero upper half for 64-bit vector ops
      buf.fill(0, vd + 8, vd + 16);
    }
    return 0n;
  };
}

// ── Translate a basic block to a compiled WASM module ────────────────────────

// The NEXT_PC local stores the return value. Branch instructions write to it
// instead of using return_(), so the epilogue (which stores registers back to
// linear memory) always runs.
const NEXT_PC_LOCAL = REG_COUNT + 4;
const ITER_LOCAL = REG_COUNT + 5;  // self-loop iteration counter

export function translateBlock(
  block: BasicBlock,
  memory: WebAssembly.Memory,
): CompiledBlock {
  const b = new WasmModuleBuilder();
  b.addMemoryImport('env', 'memory', 256);
  const helpers = setupHelperImports(b);
  const typeIdx = b.addFuncType([], [ValType.I64]); // () -> i64 (next PC)
  b.beginFunction(typeIdx);

  // Allocate locals: 0-30=X0-X30, 31=SP, 32=NZCV, 33-36=TMP, 37=NEXT_PC, 38=ITER
  const firstLocal = b.addLocals(ValType.I64, REG_COUNT + 6);
  const TMP0 = REG_COUNT;
  const TMP1 = REG_COUNT + 1;
  const TMP2 = REG_COUNT + 2;
  const TMP3 = REG_COUNT + 3;

  const usage = analyzeRegUsage(block.insns);
  const allUsed = new Set([...usage.reads, ...usage.writes]);

  // ── Prologue: load registers from linear memory ──
  emitPrologue(b, allUsed);

  // Initialize NEXT_PC to HALT (safety default)
  b.i64_const(HALT_PC);
  b.local_set(NEXT_PC_LOCAL);

  if (block.isSelfLoop) {
    emitSelfLoopBlock(b, block, TMP0, TMP1, TMP2, TMP3, helpers);
  } else {
    emitLinearBlock(b, block, TMP0, TMP1, TMP2, TMP3, helpers);
    // ── Epilogue: store modified registers back ──
    emitEpilogue(b, usage.writes);
  }

  // Return NEXT_PC
  b.local_get(NEXT_PC_LOCAL);

  b.endFunction();
  b.exportFunction('run');

  const bytes = b.finalize();
  const mod = new WebAssembly.Module(bytes);

  return instantiateBlock(
    { module: mod, startPC: block.startPC, endPC: block.endPC, insnCount: block.insns.length, isSelfLoop: block.isSelfLoop },
    memory,
  );
}

/** Compile a basic block to a reusable WebAssembly.Module (expensive, cacheable). */
export function compileBlock(
  block: BasicBlock,
  memory: WebAssembly.Memory,
): CompiledBlockModule {
  const b = new WasmModuleBuilder();
  b.addMemoryImport('env', 'memory', 256);
  const helpers = setupHelperImports(b);
  const typeIdx = b.addFuncType([], [ValType.I64]);
  b.beginFunction(typeIdx);
  b.addLocals(ValType.I64, REG_COUNT + 6);
  const TMP0 = REG_COUNT, TMP1 = REG_COUNT + 1, TMP2 = REG_COUNT + 2, TMP3 = REG_COUNT + 3;
  const usage = analyzeRegUsage(block.insns);
  const allUsed = new Set([...usage.reads, ...usage.writes]);
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
  b.exportFunction('run');
  const bytes = b.finalize();
  return {
    module: new WebAssembly.Module(bytes),
    startPC: block.startPC,
    endPC: block.endPC,
    insnCount: block.insns.length,
    isSelfLoop: block.isSelfLoop,
  };
}

/** Instantiate a compiled block module with a specific Memory (cheap, per-Worker). */
export function instantiateBlock(
  compiled: CompiledBlockModule,
  memory: WebAssembly.Memory,
): CompiledBlock {
  const env = { memory, ...helperFunctions, simd_op: createSimdOp(memory) };
  const instance = new WebAssembly.Instance(compiled.module, { env });
  return {
    run: instance.exports.run as () => bigint,
    startPC: compiled.startPC,
    endPC: compiled.endPC,
    insnCount: compiled.insnCount,
    isSelfLoop: compiled.isSelfLoop,
  };
}

function emitPrologue(b: WasmModuleBuilder, allUsed: Set<number>): void {
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

// ── Emit a linear (non-looping) block ────────────────────────────────────────

function emitLinearBlock(
  b: WasmModuleBuilder,
  block: BasicBlock,
  TMP0: number, TMP1: number, TMP2: number, TMP3: number,
  helpers?: HelperImports,
): void {
  const startPC = Number(block.startPC);
  for (let i = 0; i < block.insns.length; i++) {
    const pc = startPC + i * 4;
    emitInsn(b, block.insns[i], pc, TMP0, TMP1, TMP2, TMP3, helpers);
  }
  // Write instruction count to memory for the dispatch loop
  b.i32_const(ITER_OFFSET);
  b.i64_const(BigInt(block.insns.length));
  b.i64_store(3, 0);
}

// ── Emit a self-looping block ────────────────────────────────────────────────

function emitSelfLoopBlock(
  b: WasmModuleBuilder,
  block: BasicBlock,
  TMP0: number, TMP1: number, TMP2: number, TMP3: number,
  helpers?: HelperImports,
): void {
  const startPC = Number(block.startPC);
  const insns = block.insns;
  const lastInsn = insns[insns.length - 1];

  // Detect CMP+B.cond fusion opportunity:
  // If last insn is B.cond, penultimate is SUBS to XZR (CMP),
  // and no other insns in the block access NZCV, we can fuse the
  // comparison into a direct i64 compare — saving ~40 WASM ops per iteration.
  const penult = insns.length >= 2 ? insns[insns.length - 2] : null;
  const canFuse = lastInsn.kind === 'b_cond' && penult !== null &&
    ((penult.kind === 'sub_reg' && penult.setFlags && penult.rd === 31) ||
     (penult.kind === 'sub_imm' && penult.setFlags && penult.rd === 31)) &&
    !insnsAccessNZCV(insns.slice(0, insns.length - 2));

  // block $exit (void) — we use NEXT_PC_LOCAL instead of stack result
  b.block(BLOCKTYPE_VOID);
  // loop $loop (void)
  b.loop(BLOCKTYPE_VOID);

  // Increment iteration counter
  b.local_get(ITER_LOCAL);
  b.i64_const(1n);
  b.i64_add();
  b.local_set(ITER_LOCAL);

  // Safety cap: prevent infinite self-loops from causing OOB memory access
  const safetyExitPC = startPC + insns.length * 4;
  b.local_get(ITER_LOCAL);
  b.i64_const(BigInt(10_000_000));
  b.i64_gt_u();
  b.if_(BLOCKTYPE_VOID);
  b.i64_const(BigInt(safetyExitPC));
  b.local_set(NEXT_PC_LOCAL);
  b.br(2); // break to $exit block
  b.end();

  // Emit all instructions except the last (and penultimate if fusing)
  const skipLast = canFuse ? 2 : 1;
  for (let i = 0; i < insns.length - skipLast; i++) {
    const pc = startPC + i * 4;
    emitInsn(b, insns[i], pc, TMP0, TMP1, TMP2, TMP3, helpers);
  }

  // Handle the backward branch
  const lastPC = startPC + (insns.length - 1) * 4;

  if (canFuse && lastInsn.kind === 'b_cond') {
    // Fused CMP+B.cond: emit direct comparison
    const cmpInsn = penult!;
    emitFusedCmpBranch(b, cmpInsn as any, lastInsn.cond);
  } else if (lastInsn.kind === 'b') {
    b.br(0); // back to $loop
  } else if (lastInsn.kind === 'b_cond') {
    emitConditionTest(b, lastInsn.cond, NZCV_LOCAL);
    b.br_if(0); // if true, back to $loop
  } else if (lastInsn.kind === 'cbz') {
    b.local_get(regLocal(lastInsn.rt));
    b.i64_eqz();
    b.br_if(0);
  } else if (lastInsn.kind === 'cbnz') {
    b.local_get(regLocal(lastInsn.rt));
    b.i64_const(0n);
    b.i64_ne();
    b.br_if(0);
  } else if (lastInsn.kind === 'tbz') {
    b.local_get(regLocal(lastInsn.rt));
    b.i64_const(BigInt(lastInsn.bit));
    b.i64_shr_u();
    b.i64_const(1n);
    b.i64_and();
    b.i64_eqz(); // tbz: loop back if bit is zero
    b.br_if(0);
  } else if (lastInsn.kind === 'tbnz') {
    b.local_get(regLocal(lastInsn.rt));
    b.i64_const(BigInt(lastInsn.bit));
    b.i64_shr_u();
    b.i64_const(1n);
    b.i64_and();
    b.i64_const(0n);
    b.i64_ne(); // tbnz: loop back if bit is non-zero
    b.br_if(0);
  }

  // Fall through: exit the loop
  const exitPC = lastPC + 4;
  b.i64_const(BigInt(exitPC));
  b.local_set(NEXT_PC_LOCAL);

  if (canFuse) {
    // Recompute NZCV on exit for correctness (runs only once)
    const cmpInsn = penult!;
    if (cmpInsn.kind === 'sub_reg') {
      emitGetReg(b, cmpInsn.rn, cmpInsn.sf);
      b.local_tee(TMP0);
      emitGetReg(b, cmpInsn.rm, cmpInsn.sf);
      emitShift(b, cmpInsn.shift, cmpInsn.amount, cmpInsn.sf);
      if (cmpInsn.sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
      b.local_tee(TMP1);
      b.i64_sub();
      if (cmpInsn.sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
      b.local_set(TMP2);
    } else if (cmpInsn.kind === 'sub_imm') {
      emitGetRegSP(b, cmpInsn.rn, cmpInsn.sf);
      b.local_tee(TMP0);
      b.i64_const(BigInt(cmpInsn.imm));
      b.local_tee(TMP1);
      b.i64_sub();
      if (cmpInsn.sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
      b.local_set(TMP2);
    }
    emitComputeFlags(b, false, (penult as any).sf, TMP0, TMP1, TMP2);
  }

  b.br(1); // to $exit

  b.end(); // end loop
  b.unreachable(); // needed for block result type validation
  b.end(); // end block

  // Write iteration count to memory for the dispatch loop
  b.i32_const(ITER_OFFSET);
  b.local_get(ITER_LOCAL);
  b.i64_store(3, 0);

  // Epilogue: store registers after loop exit
  const usage = analyzeRegUsage(block.insns);
  emitEpilogue(b, usage.writes);
}

// ── Check if any instructions access NZCV (for fusion safety check) ──────

function insnsAccessNZCV(insns: DecodedInsn[]): boolean {
  for (const insn of insns) {
    const usage = analyzeRegUsage([insn]);
    if (usage.reads.has(NZCV_LOCAL) || usage.writes.has(NZCV_LOCAL)) return true;
  }
  return false;
}

// ── Fused CMP+B.cond: direct i64 comparison ─────────────────────────────

function emitFusedCmpBranch(
  b: WasmModuleBuilder,
  cmpInsn: DecodedInsn & { kind: 'sub_reg' | 'sub_imm' },
  cond: number,
): void {
  // Push operand1 (Rn)
  if (cmpInsn.kind === 'sub_reg') {
    emitGetReg(b, cmpInsn.rn, cmpInsn.sf);
  } else {
    emitGetRegSP(b, cmpInsn.rn, cmpInsn.sf);
  }

  // Push operand2 (Rm or #imm)
  if (cmpInsn.kind === 'sub_reg') {
    emitGetReg(b, cmpInsn.rm, cmpInsn.sf);
    emitShift(b, cmpInsn.shift, cmpInsn.amount, cmpInsn.sf);
    if (cmpInsn.sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
  } else {
    b.i64_const(BigInt(cmpInsn.imm));
  }

  // Emit the comparison based on condition code
  // CMP = SUBS XZR, Rn, Op2  →  flags from (Rn - Op2)
  // B.cond branches when condition on (Rn - Op2) is true
  emitCondCompare(b, cond);
  b.br_if(0); // back to $loop
}

function emitCondCompare(b: WasmModuleBuilder, cond: number): void {
  // Stack has [op1, op2]. Emit the comparison that matches the condition.
  // For CMP (Rn - Op2): the condition tests are equivalent to relational ops on Rn, Op2.
  switch (cond) {
    case 0:  b.i64_eq(); break;        // EQ: Rn == Op2
    case 1:  b.i64_ne(); break;        // NE: Rn != Op2
    case 2:  b.i64_ge_u(); break;      // CS/HS: Rn >=u Op2
    case 3:  b.i64_lt_u(); break;      // CC/LO: Rn <u Op2
    case 8:  b.i64_gt_u(); break;      // HI: Rn >u Op2
    case 9:  b.i64_le_u(); break;      // LS: Rn <=u Op2
    case 10: b.i64_ge_s(); break;      // GE: Rn >=s Op2
    case 11: b.i64_lt_s(); break;      // LT: Rn <s Op2
    case 12: b.i64_gt_s(); break;      // GT: Rn >s Op2
    case 13: b.i64_le_s(); break;      // LE: Rn <=s Op2
    default:
      // MI, PL, VS, VC, AL — can't easily fuse, fall back to full path
      // (shouldn't reach here due to canFuse check, but safety fallback)
      b.i64_sub();
      b.drop();
      b.i32_const(0);
      break;
  }
}

// ── Emit epilogue: store modified registers ──────────────────────────────────

function emitEpilogue(b: WasmModuleBuilder, writes: Set<number>): void {
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

// ── Emit a single instruction ────────────────────────────────────────────────

function emitInsn(
  b: WasmModuleBuilder,
  insn: DecodedInsn,
  pc: number,
  TMP0: number, TMP1: number, TMP2: number, TMP3: number,
  helpers?: HelperImports,
): void {
  switch (insn.kind) {
    case 'add_imm': emitAddSubImm(b, insn, true, TMP0, TMP1, TMP2); break;
    case 'sub_imm': emitAddSubImm(b, insn, false, TMP0, TMP1, TMP2); break;
    case 'add_reg': emitAddSubReg(b, insn, true, TMP0, TMP1, TMP2); break;
    case 'sub_reg': emitAddSubReg(b, insn, false, TMP0, TMP1, TMP2); break;
    case 'adc': case 'sbc': emitAdcSbc(b, insn, TMP0); break;
    case 'orr_reg': emitLogicalReg(b, insn, TMP0); break;
    case 'eor_reg': emitLogicalReg(b, insn, TMP0); break;
    case 'and_reg': emitLogicalReg(b, insn, TMP0); break;
    case 'and_imm': emitLogicalImm(b, insn, TMP0); break;
    case 'orr_imm': emitLogicalImm(b, insn, TMP0); break;
    case 'eor_imm': emitLogicalImm(b, insn, TMP0); break;
    case 'movz': emitMovz(b, insn); break;
    case 'movk': emitMovk(b, insn); break;
    case 'movn': emitMovn(b, insn); break;
    case 'madd': emitMadd(b, insn); break;
    case 'msub': emitMsub(b, insn); break;
    case 'smaddl': case 'smsubl': case 'umaddl': case 'umsubl':
      emitMulLong(b, insn);
      break;
    case 'sdiv': emitDiv(b, insn, true); break;
    case 'udiv': emitDiv(b, insn, false); break;
    case 'smulh': {
      emitGetReg(b, insn.rn, 1);
      emitGetReg(b, insn.rm, 1);
      b.call(helpers!.smulhIdx);
      emitSetReg(b, insn.rd, 1);
      break;
    }
    case 'umulh': {
      emitGetReg(b, insn.rn, 1);
      emitGetReg(b, insn.rm, 1);
      b.call(helpers!.umulhIdx);
      emitSetReg(b, insn.rd, 1);
      break;
    }
    case 'sbfm': emitBitfield(b, insn); break;
    case 'ubfm': emitBitfield(b, insn); break;
    case 'bfm': emitBitfield(b, insn); break;
    case 'csel': emitCondSelect(b, insn); break;
    case 'ldr_imm': emitLdrImm(b, insn); break;
    case 'str_imm': emitStrImm(b, insn); break;
    case 'ldrb_imm': emitSubwordLoadImm(b, insn, 1, false); break;
    case 'strb_imm': emitSubwordStoreImm(b, insn, 1); break;
    case 'ldrh_imm': emitSubwordLoadImm(b, insn, 2, false); break;
    case 'strh_imm': emitSubwordStoreImm(b, insn, 2); break;
    case 'ldrsb_imm': emitSubwordLoadImm(b, insn, 1, true); break;
    case 'ldrsh_imm': emitSubwordLoadImm(b, insn, 2, true); break;
    case 'ldrsw_imm': emitSubwordLoadImm(b, insn, 4, true); break;
    case 'ldr_reg': emitLdrReg(b, insn); break;
    case 'str_reg': emitStrReg(b, insn); break;
    case 'ldp': emitLdp(b, insn); break;
    case 'stp': emitStp(b, insn); break;
    case 'adr': {
      b.i64_const(BigInt(pc) + BigInt(insn.imm));
      emitSetReg(b, insn.rd, 1);
      break;
    }
    case 'adrp': {
      b.i64_const(BigInt(pc & ~0xFFF) + BigInt(insn.imm));
      emitSetReg(b, insn.rd, 1);
      break;
    }

    // Sub-word register-offset loads/stores
    case 'ldrb_reg': emitSubwordRegLoad(b, insn, 1, false); break;
    case 'strb_reg': emitSubwordRegStore(b, insn, 1); break;
    case 'ldrh_reg': emitSubwordRegLoad(b, insn, 2, false); break;
    case 'strh_reg': emitSubwordRegStore(b, insn, 2); break;
    case 'ldrsb_reg': emitSubwordRegLoad(b, insn, 1, true); break;
    case 'ldrsh_reg': emitSubwordRegLoad(b, insn, 2, true); break;
    case 'ldrsw_reg': emitSubwordRegLoad(b, insn, 4, true); break;

    // Load literal (PC-relative)
    case 'ldr_lit': {
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
    case 'ldrsw_lit': {
      const addr = pc + insn.offset;
      emitGuestAddr(b, BigInt(addr));
      b.i64_load32_s(2, 0);
      emitSetReg(b, insn.rt, 1);
      break;
    }

    // CLZ: count leading zeros (native WASM)
    case 'clz': {
      emitGetReg(b, insn.rn, insn.sf);
      if (insn.sf === 0) {
        // 32-bit: clz of low 32 bits, but need to handle the 64-bit local
        // Shift left 32, then clz gives 32-bit result
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
    case 'cls': {
      emitGetReg(b, insn.rn, insn.sf);
      if (insn.sf === 0) {
        // sign-extend 32→64, then xor with arithmetic right shift
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
      // CLS = CLZ(x ^ (x >> 1)) - 1, but CLZ already subtracted the sign bit count
      // Actually CLS is just CLZ of the xored value minus 1
      b.i64_const(1n);
      b.i64_sub();
      emitSetReg(b, insn.rd, insn.sf);
      break;
    }

    // RBIT: reverse bits — use JS helper via shift/mask
    case 'rbit': emitRbit(b, insn, TMP0); break;

    // REV: reverse bytes
    case 'rev': emitRev(b, insn, TMP0, TMP1); break;

    // REV16: reverse bytes in each 16-bit halfword
    case 'rev16': emitRev16(b, insn, TMP0, TMP1); break;

    // REV32: reverse bytes in each 32-bit word (64-bit only)
    case 'rev32': emitRev32(b, insn, TMP0, TMP1); break;

    // EXTR: extract from pair of registers
    case 'extr': emitExtr(b, insn); break;

    // CCMP/CCMN: conditional compare
    case 'ccmp': emitCcmpCcmn(b, insn, false, TMP0, TMP1, TMP2); break;
    case 'ccmn': emitCcmpCcmn(b, insn, true, TMP0, TMP1, TMP2); break;

    // Load/store exclusive (treated as plain load/store, single-threaded)
    case 'ldxr': {
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
    case 'stxr': {
      emitGetReg(b, insn.rn, 1);
      emitAddGuestBase(b);
      emitGetReg(b, insn.rt, insn.sf);
      if (insn.sf) {
        b.i64_store(3, 0);
      } else {
        b.i64_store32(2, 0);
      }
      // Rs = 0 (always succeed)
      b.i64_const(0n);
      emitSetReg(b, insn.rs, 0);
      break;
    }

    // MRS: read system register
    case 'mrs': emitMrs(b, insn); break;
    // MSR: write system register
    case 'msr': emitMsr(b, insn); break;

    // Add/sub extended register
    case 'add_ext': emitAddSubExt(b, insn, true, TMP0, TMP1, TMP2); break;
    case 'sub_ext': emitAddSubExt(b, insn, false, TMP0, TMP1, TMP2); break;

    // Branch instructions (block terminators)
    // Store target PC to NEXT_PC_LOCAL instead of return_()
    // so the epilogue (register writeback) always runs
    case 'b': {
      const target = pc + insn.offset;
      b.i64_const(BigInt(target));
      b.local_set(NEXT_PC_LOCAL);
      break;
    }
    case 'bl': {
      b.i64_const(BigInt(pc + 4));
      b.local_set(regLocal(30));
      const target = pc + insn.offset;
      b.i64_const(BigInt(target));
      b.local_set(NEXT_PC_LOCAL);
      break;
    }
    case 'b_cond': {
      emitConditionTest(b, insn.cond, NZCV_LOCAL);
      b.if_(ValType.I64);
      b.i64_const(BigInt(pc + insn.offset));
      b.else_();
      b.i64_const(BigInt(pc + 4));
      b.end();
      b.local_set(NEXT_PC_LOCAL);
      break;
    }
    case 'cbz': {
      emitGetReg(b, insn.rt, insn.sf);
      b.i64_eqz();
      b.if_(ValType.I64);
      b.i64_const(BigInt(pc + insn.offset));
      b.else_();
      b.i64_const(BigInt(pc + 4));
      b.end();
      b.local_set(NEXT_PC_LOCAL);
      break;
    }
    case 'cbnz': {
      emitGetReg(b, insn.rt, insn.sf);
      b.i64_const(0n);
      b.i64_ne();
      b.if_(ValType.I64);
      b.i64_const(BigInt(pc + insn.offset));
      b.else_();
      b.i64_const(BigInt(pc + 4));
      b.end();
      b.local_set(NEXT_PC_LOCAL);
      break;
    }
    case 'tbz': {
      b.local_get(regLocal(insn.rt));
      b.i64_const(BigInt(insn.bit));
      b.i64_shr_u();
      b.i64_const(1n);
      b.i64_and();
      b.i64_eqz(); // tbz: branch if bit is zero
      b.if_(ValType.I64);
      b.i64_const(BigInt(pc + insn.offset));
      b.else_();
      b.i64_const(BigInt(pc + 4));
      b.end();
      b.local_set(NEXT_PC_LOCAL);
      break;
    }
    case 'tbnz': {
      b.local_get(regLocal(insn.rt));
      b.i64_const(BigInt(insn.bit));
      b.i64_shr_u();
      b.i64_const(1n);
      b.i64_and();
      b.i64_const(0n);
      b.i64_ne(); // tbnz: branch if bit is non-zero
      b.if_(ValType.I64);
      b.i64_const(BigInt(pc + insn.offset));
      b.else_();
      b.i64_const(BigInt(pc + 4));
      b.end();
      b.local_set(NEXT_PC_LOCAL);
      break;
    }
    case 'br': {
      b.local_get(regLocal(insn.rn));
      b.local_set(NEXT_PC_LOCAL);
      break;
    }
    case 'blr': {
      b.i64_const(BigInt(pc + 4));
      b.local_set(regLocal(30));
      b.local_get(regLocal(insn.rn));
      b.local_set(NEXT_PC_LOCAL);
      break;
    }
    case 'ret': {
      b.local_get(regLocal(insn.rn));
      b.local_set(NEXT_PC_LOCAL);
      break;
    }
    case 'svc': {
      // Write current PC to PC_OFFSET so dispatcher knows where we stopped
      b.i32_const(PC_OFFSET);
      b.i64_const(BigInt(pc));
      b.i64_store(3, 0);
      b.i64_const(SVC_PC);
      b.local_set(NEXT_PC_LOCAL);
      break;
    }
    case 'brk': {
      // BRK = debug breakpoint / abort — halt execution
      b.i64_const(HALT_PC);
      b.local_set(NEXT_PC_LOCAL);
      break;
    }
    // Variable shifts
    case 'lslv': case 'lsrv': case 'asrv': case 'rorv': {
      if (insn.kind === 'asrv' && insn.sf === 0) {
        // 32-bit ASR: sign-extend value to 64-bit first, then shift
        emitGetReg(b, insn.rn, 0);
        b.i64_const(32n);
        b.i64_shl();
        b.i64_const(32n);
        b.i64_shr_s(); // sign-extended
        emitGetReg(b, insn.rm, 0);
        b.i64_const(31n);
        b.i64_and();
        b.i64_shr_s();
        b.i64_const(0xFFFFFFFFn);
        b.i64_and();
      } else if (insn.kind === 'rorv' && insn.sf === 0) {
        // 32-bit ROR: need to handle within 32-bit space
        // ror32(x, n) = (x >> n) | (x << (32 - n)) masked to 32 bits
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
          case 'lslv': b.i64_shl(); break;
          case 'lsrv': b.i64_shr_u(); break;
          case 'asrv': b.i64_shr_s(); break;
          case 'rorv': b.i64_rotr(); break;
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
    case 'str_simd': {
      if (insn.size === 16) {
        emitSimdStore128(b, insn, TMP0);
      } else {
        emitSimdStoreSmall(b, insn, TMP0);
      }
      break;
    }
    case 'ldr_simd': {
      if (insn.size === 16) {
        emitSimdLoad128(b, insn, TMP0);
      } else {
        emitSimdLoadSmall(b, insn, TMP0);
      }
      break;
    }
    case 'stp_simd': {
      if (insn.size === 16) {
        emitSimdStorePair128(b, insn, TMP0);
      } else {
        emitSimdStorePairSmall(b, insn, TMP0);
      }
      break;
    }
    case 'ldp_simd': {
      if (insn.size === 16) {
        emitSimdLoadPair128(b, insn, TMP0);
      } else {
        emitSimdLoadPairSmall(b, insn, TMP0);
      }
      break;
    }
    // SIMD DUP from general register
    case 'simd_dup_gpr': {
      emitSimdDupGpr(b, insn);
      break;
    }
    // SIMD DUP from element
    case 'simd_dup_elem': {
      emitSimdDupElem(b, insn, TMP0);
      break;
    }
    // SIMD UMOV (extract to GPR)
    case 'simd_umov': {
      emitSimdUmov(b, insn);
      break;
    }
    // SIMD MOVI (move immediate to vector)
    case 'simd_movi': {
      emitSimdMovi(b, insn);
      break;
    }
    // SIMD INS from GPR
    case 'simd_ins_gpr': {
      emitSimdInsGpr(b, insn);
      break;
    }
    // FMOV GPR → FP register
    case 'fmov_to_fp': {
      emitFmovToFp(b, insn);
      break;
    }
    // FMOV FP register → GPR
    case 'fmov_from_fp': {
      emitFmovFromFp(b, insn);
      break;
    }
    // FMOV GPR → top half of 128-bit FP register
    case 'fmov_top_to_fp': {
      const vdOff = vregOffset(insn.vd);
      b.i32_const(vdOff + 8);  // upper qword
      emitGetReg(b, insn.rn, 1);
      b.i64_store(3, 0);
      break;
    }
    // FMOV top half of 128-bit FP register → GPR
    case 'fmov_top_from_fp': {
      const vnOff = vregOffset(insn.vn);
      b.i32_const(vnOff + 8);  // upper qword
      b.i64_load(3, 0);
      emitSetReg(b, insn.rd, 1);
      break;
    }
    // FMOV between FP registers
    case 'fmov_fp': {
      emitFmovFp(b, insn);
      break;
    }
    // FP <-> integer conversions
    case 'scvtf_int': case 'ucvtf_int':
    case 'fcvtzs_int': case 'fcvtzu_int': {
      emitFpIntConv(b, insn, TMP0, helpers);
      break;
    }
    // FP arithmetic (2-source)
    case 'fmul': case 'fdiv': case 'fadd': case 'fsub': {
      emitFpArith2(b, insn, TMP0, helpers);
      break;
    }
    // FP misc (1-source)
    case 'fabs': case 'fneg': {
      emitFpMisc1(b, insn, TMP0, helpers);
      break;
    }
    // FP precision conversion
    case 'fcvt': {
      emitFcvt(b, insn, TMP0, helpers);
      break;
    }
    // FP immediate
    case 'fmov_imm': {
      emitFmovImm(b, insn);
      break;
    }
    // FP compare
    case 'fcmp': {
      emitFpCmp(b, insn, helpers);
      break;
    }
    // FP fused multiply-add/subtract (3-source)
    case 'fmadd': case 'fmsub': case 'fnmadd': case 'fnmsub': {
      emitFpMulAdd(b, insn, TMP0, helpers);
      break;
    }
    // SIMD ORR (vector, register) / MOV (vector)
    case 'simd_orr_reg': {
      const rnOff = vregOffset(insn.rn);
      const rmOff = vregOffset(insn.rm);
      const rdOff = vregOffset(insn.rd);
      // Low qword: Vn | Vm
      b.i32_const(rdOff);
      b.i32_const(rnOff); b.i64_load(3, 0);
      b.i32_const(rmOff); b.i64_load(3, 0);
      b.i64_or();
      b.i64_store(3, 0);
      if (insn.Q) {
        // High qword
        b.i32_const(rdOff + 8);
        b.i32_const(rnOff + 8); b.i64_load(3, 0);
        b.i32_const(rmOff + 8); b.i64_load(3, 0);
        b.i64_or();
        b.i64_store(3, 0);
      } else {
        // Q=0: clear high qword
        b.i32_const(rdOff + 8);
        b.i64_const(0n);
        b.i64_store(3, 0);
      }
      break;
    }
    // SIMD NOT (bitwise complement)
    case 'simd_not': {
      const rnOff = vregOffset(insn.rn);
      const rdOff = vregOffset(insn.rd);
      b.i32_const(rdOff);
      b.i32_const(rnOff); b.i64_load(3, 0);
      b.i64_const(-1n); b.i64_xor();
      b.i64_store(3, 0);
      b.i32_const(rdOff + 8);
      if (insn.Q) {
        b.i32_const(rnOff + 8); b.i64_load(3, 0);
        b.i64_const(-1n); b.i64_xor();
      } else {
        b.i64_const(0n);
      }
      b.i64_store(3, 0);
      break;
    }

    // SIMD NEG, ABS, CNT, CLS, CLZ (two-reg misc) — use JS helper
    case 'simd_neg': case 'simd_abs':
    case 'simd_cnt': case 'simd_cls_vec': case 'simd_clz_vec': {
      // TODO: implement properly. For now halt (these are less common)
      b.i64_const(HALT_PC);
      b.local_set(NEXT_PC_LOCAL);
      break;
    }

    // SIMD bitwise operations (AND, BIC, EOR)
    case 'simd_and_reg': case 'simd_bic_reg': case 'simd_eor_reg': {
      const rnOff = vregOffset(insn.rn);
      const rmOff = vregOffset(insn.rm);
      const rdOff = vregOffset(insn.rd);
      const op = insn.kind === 'simd_and_reg' ? 'and' : insn.kind === 'simd_bic_reg' ? 'bic' : 'eor';
      // Low qword
      b.i32_const(rdOff);
      b.i32_const(rnOff); b.i64_load(3, 0);
      b.i32_const(rmOff); b.i64_load(3, 0);
      if (op === 'bic') { b.i64_const(-1n); b.i64_xor(); b.i64_and(); }
      else if (op === 'and') { b.i64_and(); }
      else { b.i64_xor(); }
      b.i64_store(3, 0);
      // High qword
      b.i32_const(rdOff + 8);
      if (insn.Q) {
        b.i32_const(rnOff + 8); b.i64_load(3, 0);
        b.i32_const(rmOff + 8); b.i64_load(3, 0);
        if (op === 'bic') { b.i64_const(-1n); b.i64_xor(); b.i64_and(); }
        else if (op === 'and') { b.i64_and(); }
        else { b.i64_xor(); }
      } else {
        b.i64_const(0n);
      }
      b.i64_store(3, 0);
      break;
    }

    // SIMD integer ADD/SUB — element-wise via helper function
    case 'simd_add': case 'simd_sub':
    case 'simd_cmeq_reg': case 'simd_cmgt_reg': case 'simd_cmhi_reg':
    case 'simd_cmge_reg': case 'simd_cmhs_reg':
    case 'simd_umaxp': case 'simd_addp':
    case 'simd_tbl':
    case 'simd_zip1': case 'simd_zip2': case 'simd_uzp1': case 'simd_uzp2':
    case 'simd_trn1': case 'simd_trn2': {
      // These all need element-wise processing — emit JS helper calls
      // For now, use a WASM→JS helper approach: call out to JS for the operation
      emitSimdThreeSameGeneric(b, insn, TMP0, helpers);
      break;
    }

    // SIMD shift by immediate
    case 'simd_sshll': case 'simd_ushll': {
      emitSimdShiftLong(b, insn, TMP0);
      break;
    }
    case 'nop':
      break;
    default:
      // Unknown instruction: halt
      b.i64_const(HALT_PC);
      b.local_set(NEXT_PC_LOCAL);
      break;
  }
}

// ── Get register value (handles XZR and 32-bit truncation) ───────────────────

function emitGetReg(b: WasmModuleBuilder, r: number, sf: 0 | 1): void {
  if (r === 31) {
    b.i64_const(0n); // XZR
  } else {
    b.local_get(regLocal(r));
    if (sf === 0) {
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
    }
  }
}

function emitGetRegSP(b: WasmModuleBuilder, r: number, sf: 0 | 1): void {
  // For instructions where reg 31 = SP (not XZR)
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

function emitSetReg(b: WasmModuleBuilder, r: number, sf: 0 | 1): void {
  // Value is on stack. Mask to 32-bit if needed, then store.
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

function emitSetRegSP(b: WasmModuleBuilder, r: number, sf: 0 | 1): void {
  if (r === 31) {
    if (sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
    b.local_set(SP_LOCAL);
    return;
  }
  emitSetReg(b, r, sf);
}

// ── ADD/SUB immediate ────────────────────────────────────────────────────────

function emitAddSubImm(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'add_imm' | 'sub_imm' },
  isAdd: boolean,
  TMP0: number, TMP1: number, TMP2: number,
): void {
  if (insn.setFlags) {
    // Save operands for flag computation
    // For ADDS/SUBS, Rn=31 is still SP, but Rd=31 is XZR (CMP/CMN pattern)
    emitGetRegSP(b, insn.rn, insn.sf);
    b.local_tee(TMP0); // op1
    b.i64_const(BigInt(insn.imm));
    b.local_tee(TMP1); // op2
    if (isAdd) b.i64_add(); else b.i64_sub();
    if (insn.sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
    b.local_tee(TMP2); // result
    emitSetReg(b, insn.rd, insn.sf); // Rd=31 → XZR for ADDS/SUBS
    emitComputeFlags(b, isAdd, insn.sf, TMP0, TMP1, TMP2);
  } else {
    emitGetRegSP(b, insn.rn, insn.sf);
    b.i64_const(BigInt(insn.imm));
    if (isAdd) b.i64_add(); else b.i64_sub();
    emitSetRegSP(b, insn.rd, insn.sf);
  }
}

// ── ADD/SUB register ─────────────────────────────────────────────────────────

function emitAddSubReg(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'add_reg' | 'sub_reg' },
  isAdd: boolean,
  TMP0: number, TMP1: number, TMP2: number,
): void {
  if (insn.setFlags) {
    emitGetReg(b, insn.rn, insn.sf);
    b.local_tee(TMP0);
    emitGetReg(b, insn.rm, insn.sf);
    emitShift(b, insn.shift, insn.amount, insn.sf);
    if (insn.sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
    b.local_tee(TMP1);
    if (isAdd) b.i64_add(); else b.i64_sub();
    if (insn.sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
    b.local_tee(TMP2);
    emitSetReg(b, insn.rd, insn.sf);
    emitComputeFlags(b, isAdd, insn.sf, TMP0, TMP1, TMP2);
  } else {
    emitGetReg(b, insn.rn, insn.sf);
    emitGetReg(b, insn.rm, insn.sf);
    emitShift(b, insn.shift, insn.amount, insn.sf);
    if (isAdd) b.i64_add(); else b.i64_sub();
    emitSetReg(b, insn.rd, insn.sf);
  }
}

// ── Add/subtract with carry ──────────────────────────────────────────────────

function emitAdcSbc(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'adc' | 'sbc' },
  TMP0: number,
): void {
  const isSbc = insn.kind === 'sbc';
  const mask = insn.sf ? 0xFFFFFFFFFFFFFFFFn : 0xFFFFFFFFn;

  // Extract carry: C = (NZCV >> 1) & 1
  b.local_get(NZCV_LOCAL);
  b.i64_const(1n);
  b.i64_shr_u();
  b.i64_const(1n);
  b.i64_and();
  b.local_set(TMP0); // TMP0 = carry

  // ADC: Rd = Rn + Rm + C
  // SBC: Rd = Rn + NOT(Rm) + C
  emitGetReg(b, insn.rn, insn.sf);
  emitGetReg(b, insn.rm, insn.sf);
  if (isSbc) {
    b.i64_const(-1n);
    b.i64_xor(); // NOT(Rm)
    if (insn.sf === 0) { b.i64_const(mask); b.i64_and(); }
  }
  b.i64_add();
  b.local_get(TMP0); // + carry
  b.i64_add();
  if (insn.sf === 0) { b.i64_const(mask); b.i64_and(); }
  emitSetReg(b, insn.rd, insn.sf);

  // ADCS/SBCS: set flags — not needed for seq (only ADC/SBC without S)
  // If setFlags were needed, we'd compute NZCV here
}

// ── Shift operand ────────────────────────────────────────────────────────────

function emitShift(b: WasmModuleBuilder, shift: ShiftType, amount: number, sf: 0|1 = 1): void {
  if (amount === 0) return;
  if (sf === 0 && shift === 'asr') {
    // 32-bit ASR: value on stack is zero-extended to 64 bits, so bit 63 is always 0.
    // i64_shr_s would sign-extend from bit 63 (wrong). We need sign extension from bit 31.
    // Fix: shift left 32 to put 32-bit value in upper half, then arithmetic shift right
    // by (32 + amount). This sign-extends from the original bit 31 and shifts correctly.
    b.i64_const(32n);
    b.i64_shl();
    b.i64_const(BigInt(32 + amount));
    b.i64_shr_s();
    b.i64_const(0xFFFFFFFFn);
    b.i64_and();
    return;
  }
  if (sf === 0 && shift === 'ror') {
    // 32-bit ROR: i64_rotr rotates within 64 bits, but we need 32-bit rotation.
    // For a zero-extended 32-bit value, rotating 64-bit brings in zeros from upper 32.
    // Fix: duplicate the 32-bit value into both halves, then rotr64 gives correct lower 32.
    // stack: x (zero-extended 32-bit)
    const TMP = REG_COUNT + 3; // Use TMP3 to avoid conflict with callers using TMP0
    b.local_tee(TMP);
    b.i64_const(32n);
    b.i64_shl();
    b.local_get(TMP);
    b.i64_or();  // x | (x << 32) — both halves are the 32-bit value
    b.i64_const(BigInt(amount));
    b.i64_rotr();
    b.i64_const(0xFFFFFFFFn);
    b.i64_and();
    return;
  }
  b.i64_const(BigInt(amount));
  switch (shift) {
    case 'lsl': b.i64_shl(); break;
    case 'lsr': b.i64_shr_u(); break;
    case 'asr': b.i64_shr_s(); break;
    case 'ror': b.i64_rotr(); break;
  }
}

// ── Logical register (ORR, EOR, AND) ─────────────────────────────────────────

function emitLogicalReg(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'orr_reg' | 'eor_reg' | 'and_reg' },
  TMP0: number,
): void {
  emitGetReg(b, insn.rn, insn.sf);
  emitGetReg(b, insn.rm, insn.sf);
  emitShift(b, insn.shift, insn.amount, insn.sf);
  if (insn.invert) { b.i64_const(-1n); b.i64_xor(); }

  switch (insn.kind) {
    case 'orr_reg': b.i64_or(); break;
    case 'eor_reg': b.i64_xor(); break;
    case 'and_reg': b.i64_and(); break;
  }

  const isAnds = insn.kind === 'and_reg' && (insn as any).setFlags;
  if (isAnds) {
    // ANDS/BICS: compute flags (N, Z, C=0, V=0), Rd=31 is XZR
    b.local_tee(TMP0);
    emitSetReg(b, insn.rd, insn.sf);

    const bits = insn.sf ? 64 : 32;
    // N flag = sign bit of result
    b.local_get(TMP0);
    if (insn.sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
    b.i64_const(BigInt(bits - 1));
    b.i64_shr_u();
    b.i64_const(1n);
    b.i64_and();
    b.i64_const(3n);
    b.i64_shl(); // N << 3

    // Z flag = result is zero
    b.local_get(TMP0);
    if (insn.sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
    b.i64_eqz();
    b.i64_extend_i32_u();
    b.i64_const(2n);
    b.i64_shl(); // Z << 2

    b.i64_or(); // (N << 3) | (Z << 2), C=0, V=0
    b.local_set(NZCV_LOCAL);
  } else {
    emitSetReg(b, insn.rd, insn.sf);
  }
}

// ── Logical immediate ────────────────────────────────────────────────────────

function emitLogicalImm(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'and_imm' | 'orr_imm' | 'eor_imm' },
  TMP0: number,
): void {
  // For logical immediates, Rn=31 is XZR (not SP)
  emitGetReg(b, insn.rn, insn.sf);
  b.i64_const(insn.imm);
  switch (insn.kind) {
    case 'and_imm': b.i64_and(); break;
    case 'orr_imm': b.i64_or(); break;
    case 'eor_imm': b.i64_xor(); break;
  }

  const isAnds = insn.kind === 'and_imm' && (insn as any).setFlags;

  if (isAnds) {
    // ANDS: compute flags (N, Z, C=0, V=0), Rd=31 is XZR
    b.local_tee(TMP0);
    emitSetReg(b, insn.rd, insn.sf);

    const bits = insn.sf ? 64 : 32;
    // nzcv = (N << 3) | (Z << 2), C=0, V=0
    b.local_get(TMP0);
    if (insn.sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
    b.i64_const(BigInt(bits - 1));
    b.i64_shr_u();
    b.i64_const(1n);
    b.i64_and();
    b.i64_const(3n);
    b.i64_shl(); // N << 3

    b.local_get(TMP0);
    if (insn.sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
    b.i64_eqz();
    b.i64_extend_i32_u();
    b.i64_const(2n);
    b.i64_shl(); // Z << 2

    b.i64_or(); // (N << 3) | (Z << 2)
    b.local_set(NZCV_LOCAL);
  } else {
    // AND/ORR/EOR without flags: Rd=31 is SP
    emitSetRegSP(b, insn.rd, insn.sf);
  }
}

// ── Move wide ────────────────────────────────────────────────────────────────

function emitMovz(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'movz' }): void {
  const value = BigInt(insn.imm16) << BigInt(insn.hw * 16);
  b.i64_const(value);
  emitSetReg(b, insn.rd, insn.sf);
}

function emitMovk(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'movk' }): void {
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

function emitMovn(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'movn' }): void {
  const value = BigInt(insn.imm16) << BigInt(insn.hw * 16);
  b.i64_const(~value);
  if (insn.sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
  emitSetReg(b, insn.rd, insn.sf);
}

// ── Multiply ─────────────────────────────────────────────────────────────────

function emitMadd(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'madd' }): void {
  emitGetReg(b, insn.ra, insn.sf);
  emitGetReg(b, insn.rn, insn.sf);
  emitGetReg(b, insn.rm, insn.sf);
  b.i64_mul();
  b.i64_add();
  emitSetReg(b, insn.rd, insn.sf);
}

function emitMsub(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'msub' }): void {
  emitGetReg(b, insn.ra, insn.sf);
  emitGetReg(b, insn.rn, insn.sf);
  emitGetReg(b, insn.rm, insn.sf);
  b.i64_mul();
  b.i64_sub();
  emitSetReg(b, insn.rd, insn.sf);
}

// ── Multiply-Long (SMADDL, SMSUBL, UMADDL, UMSUBL) ────────────────────────

function emitMulLong(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'smaddl' | 'smsubl' | 'umaddl' | 'umsubl' },
): void {
  const isSigned = insn.kind === 'smaddl' || insn.kind === 'smsubl';
  const isSub = insn.kind === 'smsubl' || insn.kind === 'umsubl';

  // Get Wn extended to 64-bit
  emitGetReg(b, insn.rn, 0); // 32-bit read (zero-extended to i64)
  if (isSigned) {
    // Sign-extend from 32-bit: (val << 32) >> 32
    b.i64_const(32n); b.i64_shl();
    b.i64_const(32n); b.i64_shr_s();
  } else {
    b.i64_const(0xFFFFFFFFn); b.i64_and();
  }

  // Get Wm extended to 64-bit
  emitGetReg(b, insn.rm, 0);
  if (isSigned) {
    b.i64_const(32n); b.i64_shl();
    b.i64_const(32n); b.i64_shr_s();
  } else {
    b.i64_const(0xFFFFFFFFn); b.i64_and();
  }

  b.i64_mul();

  // Add or subtract Ra
  if (isSub) {
    // Stack: [product]. Need: Ra - product
    // Store product to temp, push Ra, push product, subtract
    const tmpLocal = regLocal(insn.rd); // safe to use as temp since we'll overwrite rd
    b.local_set(tmpLocal);
    emitGetReg(b, insn.ra, 1);
    b.local_get(tmpLocal);
    b.i64_sub();
  } else {
    // Xd = product + Ra
    emitGetReg(b, insn.ra, 1);
    b.i64_add();
  }

  emitSetReg(b, insn.rd, 1);
}

// ── Division (with ARM64 div-by-zero = 0 semantics) ──────────────────────────

function emitDiv(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'sdiv' | 'udiv' },
  isSigned: boolean,
): void {
  emitGetReg(b, insn.rm, insn.sf);
  b.i64_eqz();
  b.if_(ValType.I64);
  b.i64_const(0n); // div by zero returns 0 in ARM64
  b.else_();
  emitGetReg(b, insn.rn, insn.sf);
  emitGetReg(b, insn.rm, insn.sf);
  if (isSigned) b.i64_div_s(); else b.i64_div_u();
  b.end();
  emitSetReg(b, insn.rd, insn.sf);
}

// ── Bitfield ─────────────────────────────────────────────────────────────────

function emitBitfield(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'sbfm' | 'ubfm' | 'bfm' },
): void {
  const regSize = insn.sf ? 64 : 32;

  if (insn.kind === 'ubfm') {
    // Common aliases: LSL, LSR, UXTB, UXTH, UXTW, UBFX
    if (insn.imms + 1 === regSize) {
      // LSR: Rd = Rn >> immr
      emitGetReg(b, insn.rn, insn.sf);
      if (insn.immr > 0) {
        b.i64_const(BigInt(insn.immr));
        b.i64_shr_u();
      }
    } else if (insn.imms < insn.immr) {
      // UBFIZ / LSL: extract low (imms+1) bits, shift left by (regSize - immr)
      const width = insn.imms + 1;
      const lsb = regSize - insn.immr;
      emitGetReg(b, insn.rn, insn.sf);
      // Mask low 'width' bits (for LSL alias where width=immr this is a no-op effectively)
      const mask = (1n << BigInt(width)) - 1n;
      b.i64_const(mask);
      b.i64_and();
      b.i64_const(BigInt(lsb));
      b.i64_shl();
    } else {
      // UBFX: extract bitfield
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
  } else if (insn.kind === 'sbfm') {
    if (insn.imms + 1 === regSize) {
      // ASR: Rd = Rn >> immr (arithmetic)
      emitGetReg(b, insn.rn, insn.sf);
      if (insn.sf === 0) {
        // Sign extend to 64-bit for arithmetic shift
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
      // SXTB (imms=7), SXTH (imms=15), SXTW (imms=31)
      emitGetReg(b, insn.rn, insn.sf);
      const bits = insn.imms + 1;
      const shift = BigInt(64 - bits);
      b.i64_const(shift);
      b.i64_shl();
      b.i64_const(shift);
      b.i64_shr_s();
    } else if (insn.imms < insn.immr) {
      // SBFIZ: sign-extend low (imms+1) bits, shift left by (regSize - immr)
      const width = insn.imms + 1;
      const lsb = regSize - insn.immr;
      emitGetReg(b, insn.rn, insn.sf);
      // Sign-extend low 'width' bits: shift left to bit 63, then arithmetic shift right
      const signExtShift = BigInt(64 - width);
      b.i64_const(signExtShift);
      b.i64_shl();
      b.i64_const(signExtShift);
      b.i64_shr_s();
      // Shift left by lsb
      if (lsb > 0) {
        b.i64_const(BigInt(lsb));
        b.i64_shl();
      }
    } else {
      // General SBFX: extract (width) bits from Rn at position immr, sign-extend
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
    // BFM: bitfield move — inserts a field from Rn into Rd
    if (insn.imms < insn.immr) {
      // BFI: insert low (imms+1) bits of Rn into Rd at position (regSize - immr)
      const width = insn.imms + 1;
      const lsb = regSize - insn.immr;
      const mask = ((1n << BigInt(width)) - 1n) << BigInt(lsb);
      // Rd = (Rd & ~mask) | ((Rn << lsb) & mask)
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
      // BFXIL: extract (width) bits from Rn at position immr, insert at bit 0 of Rd
      const width = insn.imms - insn.immr + 1;
      const mask = (1n << BigInt(width)) - 1n;
      // Rd = (Rd & ~mask) | ((Rn >> immr) & mask)
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

// ── Conditional select ───────────────────────────────────────────────────────

function emitCondSelect(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'csel' },
): void {
  emitConditionTest(b, insn.cond, NZCV_LOCAL);
  b.if_(ValType.I64);
  emitGetReg(b, insn.rn, insn.sf);
  b.else_();
  switch (insn.variant) {
    case 'csel':
      emitGetReg(b, insn.rm, insn.sf);
      break;
    case 'csinc':
      emitGetReg(b, insn.rm, insn.sf);
      b.i64_const(1n);
      b.i64_add();
      break;
    case 'csinv':
      emitGetReg(b, insn.rm, insn.sf);
      b.i64_const(-1n);
      b.i64_xor();
      break;
    case 'csneg':
      b.i64_const(0n);
      emitGetReg(b, insn.rm, insn.sf);
      b.i64_sub();
      break;
  }
  b.end();
  emitSetReg(b, insn.rd, insn.sf);
}

// ── Load/Store ───────────────────────────────────────────────────────────────

function emitLdrImm(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'ldr_imm' }): void {
  const guestMemBase = 0x10000;

  // Pre-index: update base first
  if (insn.mode === 'pre') {
    emitGetRegSP(b, insn.rn, 1);
    b.i64_const(BigInt(insn.offset));
    b.i64_add();
    b.local_tee(insn.rn === 31 ? SP_LOCAL : regLocal(insn.rn));
    b.i64_const(BigInt(guestMemBase));
    b.i64_add();
    b.i32_wrap_i64();
  } else if (insn.mode === 'post') {
    // Post-index: load from base + GUEST_MEM_BASE (no offset for the load)
    emitGetRegSP(b, insn.rn, 1);
    b.i64_const(BigInt(guestMemBase));
    b.i64_add();
    b.i32_wrap_i64();
  } else {
    // Unsigned offset: base + offset + GUEST_MEM_BASE
    emitGetRegSP(b, insn.rn, 1);
    b.i64_const(BigInt(insn.offset + guestMemBase));
    b.i64_add();
    b.i32_wrap_i64();
  }

  // Load
  if (insn.sf === 1) {
    b.i64_load(3, 0);
  } else {
    b.i64_load32_u(2, 0);
  }
  emitSetReg(b, insn.rt, insn.sf);

  // Post-index: update base after
  if (insn.mode === 'post') {
    emitGetRegSP(b, insn.rn, 1);
    b.i64_const(BigInt(insn.offset));
    b.i64_add();
    if (insn.rn === 31) b.local_set(SP_LOCAL);
    else b.local_set(regLocal(insn.rn));
  }
}

function emitStrImm(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'str_imm' }): void {
  const guestMemBase = 0x10000;

  // Pre-index: update base first
  if (insn.mode === 'pre') {
    emitGetRegSP(b, insn.rn, 1);
    b.i64_const(BigInt(insn.offset));
    b.i64_add();
    b.local_tee(insn.rn === 31 ? SP_LOCAL : regLocal(insn.rn));
    b.i64_const(BigInt(guestMemBase));
    b.i64_add();
    b.i32_wrap_i64();
  } else if (insn.mode === 'post') {
    emitGetRegSP(b, insn.rn, 1);
    b.i64_const(BigInt(guestMemBase));
    b.i64_add();
    b.i32_wrap_i64();
  } else {
    emitGetRegSP(b, insn.rn, 1);
    b.i64_const(BigInt(insn.offset + guestMemBase));
    b.i64_add();
    b.i32_wrap_i64();
  }

  // Value to store
  emitGetReg(b, insn.rt, insn.sf);

  // Store
  if (insn.sf === 1) {
    b.i64_store(3, 0);
  } else {
    b.i64_store32(2, 0);
  }

  // Post-index
  if (insn.mode === 'post') {
    emitGetRegSP(b, insn.rn, 1);
    b.i64_const(BigInt(insn.offset));
    b.i64_add();
    if (insn.rn === 31) b.local_set(SP_LOCAL);
    else b.local_set(regLocal(insn.rn));
  }
}

// ── Sub-word load immediate ──────────────────────────────────────────────────

function emitSubwordLoadImm(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' },
  byteWidth: 1 | 2 | 4,
  signExtend: boolean,
): void {
  const guestMemBase = 0x10000;

  // Pre-index: update base first
  if (insn.mode === 'pre') {
    emitGetRegSP(b, insn.rn, 1);
    b.i64_const(BigInt(insn.offset));
    b.i64_add();
    b.local_tee(insn.rn === 31 ? SP_LOCAL : regLocal(insn.rn));
    b.i64_const(BigInt(guestMemBase));
    b.i64_add();
    b.i32_wrap_i64();
  } else if (insn.mode === 'post') {
    emitGetRegSP(b, insn.rn, 1);
    b.i64_const(BigInt(guestMemBase));
    b.i64_add();
    b.i32_wrap_i64();
  } else {
    emitGetRegSP(b, insn.rn, 1);
    b.i64_const(BigInt(insn.offset + guestMemBase));
    b.i64_add();
    b.i32_wrap_i64();
  }

  // Load with appropriate width
  if (byteWidth === 1) {
    if (signExtend) b.i64_load8_s(0, 0); else b.i64_load8_u(0, 0);
  } else if (byteWidth === 2) {
    if (signExtend) b.i64_load16_s(1, 0); else b.i64_load16_u(1, 0);
  } else {
    if (signExtend) b.i64_load32_s(2, 0); else b.i64_load32_u(2, 0);
  }

  // For signed loads that target W registers (sf=0), truncate to 32 bits
  const sf = 'sf' in insn ? (insn as any).sf : 1;
  if (sf === 0 && !signExtend) {
    b.i64_const(0xFFFFFFFFn);
    b.i64_and();
  }
  const rt = insn.rt;
  if (rt !== 31) b.local_set(regLocal(rt)); else b.drop();

  // Post-index
  if (insn.mode === 'post') {
    emitGetRegSP(b, insn.rn, 1);
    b.i64_const(BigInt(insn.offset));
    b.i64_add();
    if (insn.rn === 31) b.local_set(SP_LOCAL);
    else b.local_set(regLocal(insn.rn));
  }
}

// ── Sub-word store immediate ─────────────────────────────────────────────────

function emitSubwordStoreImm(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' },
  byteWidth: 1 | 2,
): void {
  const guestMemBase = 0x10000;

  if (insn.mode === 'pre') {
    emitGetRegSP(b, insn.rn, 1);
    b.i64_const(BigInt(insn.offset));
    b.i64_add();
    b.local_tee(insn.rn === 31 ? SP_LOCAL : regLocal(insn.rn));
    b.i64_const(BigInt(guestMemBase));
    b.i64_add();
    b.i32_wrap_i64();
  } else if (insn.mode === 'post') {
    emitGetRegSP(b, insn.rn, 1);
    b.i64_const(BigInt(guestMemBase));
    b.i64_add();
    b.i32_wrap_i64();
  } else {
    emitGetRegSP(b, insn.rn, 1);
    b.i64_const(BigInt(insn.offset + guestMemBase));
    b.i64_add();
    b.i32_wrap_i64();
  }

  // Value to store
  if (insn.rt === 31) b.i64_const(0n);
  else b.local_get(regLocal(insn.rt));

  if (byteWidth === 1) b.i64_store8(0, 0);
  else b.i64_store16(1, 0);

  if (insn.mode === 'post') {
    emitGetRegSP(b, insn.rn, 1);
    b.i64_const(BigInt(insn.offset));
    b.i64_add();
    if (insn.rn === 31) b.local_set(SP_LOCAL);
    else b.local_set(regLocal(insn.rn));
  }
}

// ── Load/Store register offset ──────────────────────────────────────────────

function emitExtendReg(b: WasmModuleBuilder, rm: number, extend: string, shift: number): void {
  // Push the extended+shifted index register value onto the stack
  if (rm === 31) {
    b.i64_const(0n);
  } else {
    b.local_get(regLocal(rm));
  }

  switch (extend) {
    case 'uxtw':
      b.i64_const(0xFFFFFFFFn);
      b.i64_and();
      break;
    case 'sxtw':
      b.i64_const(32n);
      b.i64_shl();
      b.i64_const(32n);
      b.i64_shr_s();
      break;
    case 'sxtx':
      // Already 64-bit signed, nothing to do
      break;
    case 'lsl':
    default:
      // No extension needed
      break;
  }

  if (shift > 0) {
    b.i64_const(BigInt(shift));
    b.i64_shl();
  }
}

function emitLdrReg(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'ldr_reg' }): void {
  const guestMemBase = 0x10000;
  // Address = Rn + extend(Rm) << shift + GUEST_MEM_BASE
  emitGetRegSP(b, insn.rn, 1);
  emitExtendReg(b, insn.rm, insn.extend, insn.shift);
  b.i64_add();
  b.i64_const(BigInt(guestMemBase));
  b.i64_add();
  b.i32_wrap_i64();

  if (insn.sf === 1) b.i64_load(3, 0);
  else b.i64_load32_u(2, 0);
  emitSetReg(b, insn.rt, insn.sf);
}

function emitStrReg(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'str_reg' }): void {
  const guestMemBase = 0x10000;
  emitGetRegSP(b, insn.rn, 1);
  emitExtendReg(b, insn.rm, insn.extend, insn.shift);
  b.i64_add();
  b.i64_const(BigInt(guestMemBase));
  b.i64_add();
  b.i32_wrap_i64();

  emitGetReg(b, insn.rt, insn.sf);
  if (insn.sf === 1) b.i64_store(3, 0);
  else b.i64_store32(2, 0);
}

// ── Load/Store pair ─────────────────────────────────────────────────────────

function emitLdp(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'ldp' }): void {
  const guestMemBase = 0x10000;
  const regSize = insn.sf === 1 ? 8 : 4;
  const TMP_BASE = REG_COUNT; // Borrow TMP0 for base address

  // Check if Rt or Rt2 aliases the base register Rn (aliasing hazard)
  const rnLocal = insn.rn === 31 ? SP_LOCAL : regLocal(insn.rn);
  const hasAlias = (insn.rt !== 31 && regLocal(insn.rt) === rnLocal) ||
                   (insn.rt2 !== 31 && regLocal(insn.rt2) === rnLocal);

  // Pre-index: update base first
  if (insn.mode === 'pre') {
    emitGetRegSP(b, insn.rn, 1);
    b.i64_const(BigInt(insn.offset));
    b.i64_add();
    if (insn.rn === 31) b.local_set(SP_LOCAL);
    else b.local_set(regLocal(insn.rn));
  }

  // Save the base address to a temp if there's an aliasing hazard
  if (hasAlias) {
    emitGetRegSP(b, insn.rn, 1);
    b.local_set(TMP_BASE);
  }

  // Compute effective address for loads
  function emitEffAddr(extraOffset: number): void {
    if (hasAlias) {
      b.local_get(TMP_BASE);
    } else {
      emitGetRegSP(b, insn.rn, 1);
    }
    if (insn.mode === 'post') {
      b.i64_const(BigInt(guestMemBase + extraOffset));
    } else if (insn.mode === 'pre') {
      b.i64_const(BigInt(guestMemBase + extraOffset));
    } else {
      b.i64_const(BigInt(insn.offset + guestMemBase + extraOffset));
    }
    b.i64_add();
    b.i32_wrap_i64();
  }

  // Load first register
  emitEffAddr(0);
  if (insn.sf === 1) b.i64_load(3, 0);
  else b.i64_load32_u(2, 0);
  emitSetReg(b, insn.rt, insn.sf);

  // Load second register
  emitEffAddr(regSize);
  if (insn.sf === 1) b.i64_load(3, 0);
  else b.i64_load32_u(2, 0);
  emitSetReg(b, insn.rt2, insn.sf);

  // Post-index: update base
  if (insn.mode === 'post') {
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

function emitStp(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'stp' }): void {
  const guestMemBase = 0x10000;
  const regSize = insn.sf === 1 ? 8 : 4;
  const TMP_BASE = REG_COUNT; // Borrow TMP0 for base address

  // Compute effective store address into TMP_BASE (avoids aliasing with Rt/Rt2)
  emitGetRegSP(b, insn.rn, 1);
  if (insn.mode !== 'post') {
    // Pre-index and offset: address = Rn + offset
    b.i64_const(BigInt(insn.offset));
    b.i64_add();
  }
  b.local_set(TMP_BASE);

  // Store first register (reads Rt BEFORE any writeback)
  b.local_get(TMP_BASE);
  b.i64_const(BigInt(guestMemBase));
  b.i64_add();
  b.i32_wrap_i64();
  emitGetReg(b, insn.rt, insn.sf);
  if (insn.sf === 1) b.i64_store(3, 0);
  else b.i64_store32(2, 0);

  // Store second register
  b.local_get(TMP_BASE);
  b.i64_const(BigInt(guestMemBase + regSize));
  b.i64_add();
  b.i32_wrap_i64();
  emitGetReg(b, insn.rt2, insn.sf);
  if (insn.sf === 1) b.i64_store(3, 0);
  else b.i64_store32(2, 0);

  // Writeback
  if (insn.mode === 'pre') {
    // TMP_BASE already holds Rn + offset
    b.local_get(TMP_BASE);
    if (insn.rn === 31) b.local_set(SP_LOCAL);
    else b.local_set(regLocal(insn.rn));
  } else if (insn.mode === 'post') {
    // TMP_BASE holds original Rn; update to Rn + offset
    b.local_get(TMP_BASE);
    b.i64_const(BigInt(insn.offset));
    b.i64_add();
    if (insn.rn === 31) b.local_set(SP_LOCAL);
    else b.local_set(regLocal(insn.rn));
  }
}

// ── Address helpers ──────────────────────────────────────────────────────────

/** Push guest address as i32 WASM address: i32(guestAddr + GUEST_MEM_BASE) */
function emitGuestAddr(b: WasmModuleBuilder, guestAddr: bigint): void {
  b.i32_const(Number(guestAddr) + GUEST_MEM_BASE);
}

/** Stack top has a 64-bit guest address; convert to WASM linear memory i32 address */
function emitAddGuestBase(b: WasmModuleBuilder): void {
  b.i64_const(BigInt(GUEST_MEM_BASE));
  b.i64_add();
  b.i32_wrap_i64();
}

// ── Sub-word register-offset loads/stores ────────────────────────────────────

function emitSubwordRegLoad(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { rt: number; rn: number; rm: number; extend: string; shift: number },
  byteWidth: number,
  signExtend: boolean,
): void {
  // Address = Rn + extend(Rm) << shift + GUEST_MEM_BASE
  emitGetRegSP(b, insn.rn, 1);
  emitExtendReg(b, insn.rm, insn.extend, insn.shift);
  b.i64_add();
  b.i64_const(BigInt(GUEST_MEM_BASE));
  b.i64_add();
  b.i32_wrap_i64();

  switch (byteWidth) {
    case 1: signExtend ? b.i64_load8_s(0, 0) : b.i64_load8_u(0, 0); break;
    case 2: signExtend ? b.i64_load16_s(1, 0) : b.i64_load16_u(1, 0); break;
    case 4: b.i64_load32_s(2, 0); break; // ldrsw_reg always sign-extends
  }

  const sf = 'sf' in insn ? insn.sf : 1;
  emitSetReg(b, insn.rt, sf as 0|1);
}

function emitSubwordRegStore(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { rt: number; rn: number; rm: number; extend: string; shift: number },
  byteWidth: number,
): void {
  // Address = Rn + extend(Rm) << shift + GUEST_MEM_BASE
  emitGetRegSP(b, insn.rn, 1);
  emitExtendReg(b, insn.rm, insn.extend, insn.shift);
  b.i64_add();
  b.i64_const(BigInt(GUEST_MEM_BASE));
  b.i64_add();
  b.i32_wrap_i64();

  emitGetReg(b, insn.rt, 1);
  switch (byteWidth) {
    case 1: b.i64_store8(0, 0); break;
    case 2: b.i64_store16(1, 0); break;
  }
}

// ── RBIT: reverse bits ──────────────────────────────────────────────────────

function emitRbit(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'rbit'; sf: 0|1; rd: number; rn: number }, TMP0: number): void {
  // RBIT implemented via shift/mask sequence
  // result = 0; for each bit: result = (result << 1) | (val & 1); val >>= 1
  // But that's 64 iterations. Instead use the parallel bit-swap approach:
  // Swap adjacent bits, then pairs, then nibbles, then bytes, then words
  const bits = insn.sf ? 64 : 32;
  emitGetReg(b, insn.rn, insn.sf);

  if (insn.sf === 0) {
    // Zero-extend to 64 bits for uniform processing
    b.i64_const(0xFFFFFFFFn);
    b.i64_and();
  }

  // Swap adjacent bits: ((x >> 1) & 0x5555...) | ((x & 0x5555...) << 1)
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

  // Swap adjacent pairs
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

  // Swap nibbles
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

  // Now we have reversed within each byte. Reverse bytes using REV logic.
  // Swap bytes within halfwords
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

  // Swap halfwords within words
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

  if (bits === 64) {
    // Swap 32-bit halves
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

// ── REV: reverse bytes ──────────────────────────────────────────────────────

function emitRev(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'rev'; sf: 0|1; rd: number; rn: number }, TMP0: number, TMP1: number): void {
  emitGetReg(b, insn.rn, insn.sf);

  if (insn.sf === 0) {
    b.i64_const(0xFFFFFFFFn);
    b.i64_and();
  }

  // Swap bytes within halfwords
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

  // Swap halfwords within words
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
    // Swap 32-bit halves
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

// ── REV16: reverse bytes in each 16-bit halfword ────────────────────────────

function emitRev16(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'rev16'; sf: 0|1; rd: number; rn: number }, TMP0: number, _TMP1: number): void {
  emitGetReg(b, insn.rn, insn.sf);

  // Swap bytes within each 16-bit halfword
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

// ── REV32: reverse bytes in each 32-bit word (64-bit only) ──────────────────

function emitRev32(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'rev32'; rd: number; rn: number }, TMP0: number, _TMP1: number): void {
  emitGetReg(b, insn.rn, 1);

  // Swap bytes within halfwords
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

  // Swap halfwords within words
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

// ── EXTR: extract from pair of registers ────────────────────────────────────

function emitExtr(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'extr'; sf: 0|1; rd: number; rn: number; rm: number; imms: number }): void {
  if (insn.imms === 0) {
    // EXTR with imms=0 is just Rm
    emitGetReg(b, insn.rm, insn.sf);
  } else if (insn.rn === insn.rm) {
    // ROR: rotate right by imms
    emitGetReg(b, insn.rn, insn.sf);
    b.i64_const(BigInt(insn.imms));
    b.i64_rotr();
  } else {
    const regSize = insn.sf ? 64 : 32;
    // Rd = (Rn << (regSize - imms)) | (Rm >> imms)
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

// ── CCMP/CCMN: conditional compare ──────────────────────────────────────────

function emitCcmpCcmn(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'ccmp' | 'ccmn'; sf: 0|1; rn: number; rm_imm: number; nzcv: number; cond: number; isImm: boolean },
  isAdd: boolean,
  TMP0: number, TMP1: number, TMP2: number,
): void {
  // if (cond) { flags = compare(Rn, Op2) } else { flags = nzcv }
  emitConditionTest(b, insn.cond, NZCV_LOCAL);

  b.if_(BLOCKTYPE_VOID);
  // Condition true: perform comparison
  emitGetReg(b, insn.rn, insn.sf);
  b.local_set(TMP0);

  if (insn.isImm) {
    b.i64_const(BigInt(insn.rm_imm));
  } else {
    emitGetReg(b, insn.rm_imm, insn.sf);
  }
  b.local_set(TMP1);

  // Result = Rn +/- Op2
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
  // Condition false: set flags to immediate nzcv
  b.i64_const(BigInt(insn.nzcv));
  b.local_set(NZCV_LOCAL);

  b.end();
}

// ── MRS/MSR: system register access ────────────────────────────────────────

function emitMrs(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'mrs'; rt: number; sysreg: string }): void {
  switch (insn.sysreg) {
    case 'tpidr_el0':
      b.i32_const(TPIDR_OFFSET);
      b.i64_load(3, 0);
      break;
    case 'nzcv':
      b.local_get(NZCV_LOCAL);
      // NZCV is stored as 4 bits (N=3,Z=2,C=1,V=0), need to shift to bits [31:28]
      b.i64_const(28n);
      b.i64_shl();
      break;
    case 'fpcr':
      b.i64_const(0n); // Not emulating FP control
      break;
    case 'cntfrq_el0':
      b.i64_const(1000000n); // 1 MHz virtual counter
      break;
    case 'cntvct_el0':
      // Return 0 for now; would need Date.now() import for real timing
      b.i64_const(0n);
      break;
    default:
      b.i64_const(0n); // Unknown sysreg: return 0
      break;
  }
  emitSetReg(b, insn.rt, 1);
}

function emitMsr(b: WasmModuleBuilder, insn: DecodedInsn & { kind: 'msr'; rt: number; sysreg: string }): void {
  switch (insn.sysreg) {
    case 'tpidr_el0':
      b.i32_const(TPIDR_OFFSET);
      emitGetReg(b, insn.rt, 1);
      b.i64_store(3, 0);
      break;
    case 'nzcv':
      emitGetReg(b, insn.rt, 1);
      b.i64_const(28n);
      b.i64_shr_u();
      b.i64_const(0xFn);
      b.i64_and();
      b.local_set(NZCV_LOCAL);
      break;
    default:
      // Ignore writes to unknown/unimplemented system registers
      break;
  }
}

// ── ADD/SUB extended register ───────────────────────────────────────────────

function emitAddSubExt(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'add_ext' | 'sub_ext'; sf: 0|1; rd: number; rn: number; rm: number; extend: string; amount: number; setFlags: boolean },
  isAdd: boolean,
  TMP0: number, TMP1: number, TMP2: number,
): void {
  emitGetRegSP(b, insn.rn, 1);
  if (insn.setFlags) b.local_tee(TMP0);

  // Extend Rm
  emitGetReg(b, insn.rm, 1);
  switch (insn.extend) {
    case 'uxtb': b.i64_const(0xFFn); b.i64_and(); break;
    case 'uxth': b.i64_const(0xFFFFn); b.i64_and(); break;
    case 'uxtw': b.i64_const(0xFFFFFFFFn); b.i64_and(); break;
    case 'uxtx': break; // no-op for 64-bit
    case 'sxtb': b.i64_const(56n); b.i64_shl(); b.i64_const(56n); b.i64_shr_s(); break;
    case 'sxth': b.i64_const(48n); b.i64_shl(); b.i64_const(48n); b.i64_shr_s(); break;
    case 'sxtw': b.i64_const(32n); b.i64_shl(); b.i64_const(32n); b.i64_shr_s(); break;
    case 'sxtx': break; // no-op for 64-bit
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
      b.drop(); // ADDS/SUBS to SP → discard result, keep flags
    } else {
      b.local_set(SP_LOCAL);
    }
  } else {
    b.local_set(regLocal(insn.rd));
  }
}

// ── NZCV flag computation ────────────────────────────────────────────────────

function emitComputeFlags(
  b: WasmModuleBuilder,
  isAdd: boolean,
  sf: 0 | 1,
  TMP_OP1: number, TMP_OP2: number, TMP_RESULT: number,
): void {
  const signBit = sf ? 63 : 31;

  // N flag (bit 3): sign bit of result
  b.local_get(TMP_RESULT);
  b.i64_const(BigInt(signBit));
  b.i64_shr_u();
  b.i64_const(1n);
  b.i64_and();
  b.i64_const(3n);
  b.i64_shl();
  // Stack: N_value (0 or 8)

  // Z flag (bit 2): result == 0
  b.local_get(TMP_RESULT);
  if (sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
  b.i64_eqz();
  b.i64_extend_i32_u();
  b.i64_const(2n);
  b.i64_shl();
  b.i64_or();
  // Stack: N | Z

  // C flag (bit 1)
  if (isAdd) {
    // Carry for add: result < op1 (unsigned) when sf=1
    // For sf=0: check if bit 32 of the 64-bit add result is set
    if (sf === 0) {
      // Recompute 64-bit addition to check carry
      b.local_get(TMP_OP1);
      b.i64_const(0xFFFFFFFFn); b.i64_and();
      b.local_get(TMP_OP2);
      b.i64_const(0xFFFFFFFFn); b.i64_and();
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
    // Carry for sub: op1 >= op2 (unsigned) — "no borrow"
    if (sf === 0) {
      b.local_get(TMP_OP1);
      b.i64_const(0xFFFFFFFFn); b.i64_and();
      b.local_get(TMP_OP2);
      b.i64_const(0xFFFFFFFFn); b.i64_and();
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
  // Stack: N | Z | C

  // V flag (bit 0): signed overflow
  if (isAdd) {
    // V = ((op1 ^ result) & (op2 ^ result)) >> signBit
    b.local_get(TMP_OP1);
    b.local_get(TMP_RESULT);
    b.i64_xor();
    b.local_get(TMP_OP2);
    b.local_get(TMP_RESULT);
    b.i64_xor();
    b.i64_and();
  } else {
    // V = ((op1 ^ op2) & (op1 ^ result)) >> signBit
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
  // Stack: N | Z | C | V

  b.local_set(NZCV_LOCAL);
}

// ── Condition code evaluation ────────────────────────────────────────────────

function emitConditionTest(b: WasmModuleBuilder, cond: number, nzcvLocal: number): void {
  // Pushes an i32 (0 or 1) indicating whether the condition is true
  switch (cond & 0xE) { // bottom bit inverts
    case 0: // EQ/NE: Z set
      b.local_get(nzcvLocal);
      b.i64_const(4n); // Z is bit 2
      b.i64_and();
      b.i64_const(0n);
      b.i64_ne();
      break;
    case 2: // CS/CC: C set
      b.local_get(nzcvLocal);
      b.i64_const(2n); // C is bit 1
      b.i64_and();
      b.i64_const(0n);
      b.i64_ne();
      break;
    case 4: // MI/PL: N set
      b.local_get(nzcvLocal);
      b.i64_const(8n); // N is bit 3
      b.i64_and();
      b.i64_const(0n);
      b.i64_ne();
      break;
    case 6: // VS/VC: V set
      b.local_get(nzcvLocal);
      b.i64_const(1n); // V is bit 0
      b.i64_and();
      b.i64_const(0n);
      b.i64_ne();
      break;
    case 8: // HI/LS: C set AND Z clear
      b.local_get(nzcvLocal);
      b.i64_const(2n);
      b.i64_and();
      b.i64_const(0n);
      b.i64_ne(); // C set → i32
      b.local_get(nzcvLocal);
      b.i64_const(4n);
      b.i64_and();
      b.i64_eqz(); // Z clear → i32
      b.i32_add(); // both true = 2
      b.i32_const(2);
      b.i32_eq(); // exactly both true
      break;
    case 10: // GE/LT: N == V
      b.local_get(nzcvLocal);
      b.i64_const(8n);
      b.i64_and();
      b.i64_const(3n);
      b.i64_shr_u(); // N bit → position 0
      b.local_get(nzcvLocal);
      b.i64_const(1n);
      b.i64_and(); // V bit at position 0
      b.i64_eq(); // N == V → i32
      break;
    case 12: // GT/LE: Z==0 AND N==V
      // Z clear
      b.local_get(nzcvLocal);
      b.i64_const(4n);
      b.i64_and();
      b.i64_eqz(); // Z clear → i32
      // N == V
      b.local_get(nzcvLocal);
      b.i64_const(8n);
      b.i64_and();
      b.i64_const(3n);
      b.i64_shr_u();
      b.local_get(nzcvLocal);
      b.i64_const(1n);
      b.i64_and();
      b.i64_eq(); // N == V → i32
      b.i32_add();
      b.i32_const(2);
      b.i32_eq(); // both true
      break;
    case 14: // AL
      b.i32_const(1);
      break;
  }

  // Invert for odd condition codes (NE, CC, PL, VC, LS, LT, LE)
  if (cond & 1) {
    b.i32_eqz();
  }
}

// ── Region translation ──────────────────────────────────────────────────────

const BLOCK_ID_LOCAL = REG_COUNT + 6;  // i32 local for br_table dispatch

export function translateRegion(
  region: Region,
  memory: WebAssembly.Memory,
): CompiledRegion {
  const { blocks, blockMap } = region;
  const N = blocks.length;

  // Analyze registers across ALL blocks in the region
  const allReads = new Set<number>();
  const allWrites = new Set<number>();
  for (const block of blocks) {
    const usage = analyzeRegUsage(block.insns);
    for (const r of usage.reads) allReads.add(r);
    for (const r of usage.writes) allWrites.add(r);
  }
  const allUsed = new Set([...allReads, ...allWrites]);

  const b = new WasmModuleBuilder();
  b.addMemoryImport('env', 'memory', 256);
  const helpers = setupHelperImports(b);
  const typeIdx = b.addFuncType([], [ValType.I64]); // () -> i64 (next PC)
  b.beginFunction(typeIdx);

  // Allocate locals: 0-32=regs, 33-36=TMP, 37=NEXT_PC, 38=ITER, 39=BLOCK_ID(i32)
  b.addLocals(ValType.I64, REG_COUNT + 6); // locals 0..38
  const blockIdLocal = b.addLocals(ValType.I32, 1); // local 39 (i32)
  const TMP0 = REG_COUNT;
  const TMP1 = REG_COUNT + 1;
  const TMP2 = REG_COUNT + 2;
  const TMP3 = REG_COUNT + 3;

  // ── Prologue: load registers from linear memory ──
  emitPrologue(b, allUsed);

  // Initialize NEXT_PC to HALT (safety default)
  b.i64_const(HALT_PC);
  b.local_set(NEXT_PC_LOCAL);

  // ── Dispatch structure ──
  // block $exit {
  //   loop $dispatch {
  //     block $bN-1 { ... block $b0 {
  //       br_table $b0 $b1 ... $bN-1 $exit
  //     } Block 0 code } ... Block N-1 code
  //   }
  // }

  b.block(BLOCKTYPE_VOID); // $exit
  b.loop(BLOCKTYPE_VOID);  // $dispatch

  // Open N nested blocks for br_table targets
  for (let i = 0; i < N; i++) {
    b.block(BLOCKTYPE_VOID);
  }

  // br_table: dispatch on block_id
  b.local_get(blockIdLocal);
  const labels = Array.from({ length: N }, (_, i) => i); // $b0=0, $b1=1, ...
  b.br_table(labels, N + 1); // default → $exit (depth N+1 from innermost block)

  // Emit code for each block
  for (let i = 0; i < N; i++) {
    b.end(); // close $bi block — code after this is block i's body

    const block = blocks[i];
    const startPC = Number(block.startPC);
    const insns = block.insns;
    const dispatchDepth = N - 1 - i;  // depth to $dispatch (loop)
    const exitDepth = N - i;          // depth to $exit (block)

    // Increment instruction counter
    b.local_get(ITER_LOCAL);
    b.i64_const(BigInt(insns.length));
    b.i64_add();
    b.local_set(ITER_LOCAL);

    if (block.isSelfLoop) {
      emitRegionSelfLoop(b, block, i, blockMap, dispatchDepth, exitDepth,
        blockIdLocal, TMP0, TMP1, TMP2, TMP3, helpers);
    } else {
      emitRegionLinearBlock(b, block, i, blockMap, dispatchDepth, exitDepth,
        blockIdLocal, TMP0, TMP1, TMP2, TMP3, helpers);
    }
  }

  b.end(); // end $dispatch loop
  b.end(); // end $exit block

  // ── Epilogue: store registers + write insn count ──
  b.i32_const(ITER_OFFSET);
  b.local_get(ITER_LOCAL);
  b.i64_store(3, 0);

  emitEpilogue(b, allWrites);

  // Return NEXT_PC
  b.local_get(NEXT_PC_LOCAL);

  b.endFunction();
  b.exportFunction('run');

  const bytes = b.finalize();
  const mod = new WebAssembly.Module(bytes);

  return instantiateRegion(
    { module: mod, entryPC: blocks[0].startPC, blockPCs: blocks.map(bl => bl.startPC) },
    memory,
  );
}

/** Instantiate a compiled region module with a specific Memory (cheap, per-Worker). */
export function instantiateRegion(
  compiled: CompiledRegionModule,
  memory: WebAssembly.Memory,
): CompiledRegion {
  const env = { memory, ...helperFunctions, simd_op: createSimdOp(memory) };
  const instance = new WebAssembly.Instance(compiled.module, { env });
  return {
    run: instance.exports.run as () => bigint,
    entryPC: compiled.entryPC,
    blockPCs: compiled.blockPCs,
  };
}

// ── SIMD/FP emit helpers ────────────────────────────────────────────────────

const guestMemBase = GUEST_MEM_BASE;

/** Compute effective address for a SIMD load/store with pre/post/offset mode */
function emitSimdEffAddr(
  b: WasmModuleBuilder,
  rn: number, offset: number, mode: string,
  TMP0: number,
): void {
  if (mode === 'pre') {
    // Pre: compute addr = base + offset, update base
    emitGetRegSP(b, rn, 1);
    b.i64_const(BigInt(offset));
    b.i64_add();
    b.local_tee(TMP0);
    emitSetRegSP(b, rn, 1);
    b.local_get(TMP0);
  } else if (mode === 'post') {
    // Post: addr = base, then update base
    emitGetRegSP(b, rn, 1);
    b.local_tee(TMP0);
    // Update base for writeback
    b.local_get(TMP0);
    b.i64_const(BigInt(offset));
    b.i64_add();
    emitSetRegSP(b, rn, 1);
    b.local_get(TMP0);
  } else {
    // Offset: addr = base + offset
    emitGetRegSP(b, rn, 1);
    if (offset !== 0) {
      b.i64_const(BigInt(offset));
      b.i64_add();
    }
  }
  // Add GUEST_MEM_BASE (keep as i64 — caller wraps to i32 at use sites)
  b.i64_const(BigInt(guestMemBase));
  b.i64_add();
}

/** STR Q: store 128-bit from V register to guest memory */
function emitSimdStore128(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'str_simd' },
  TMP0: number,
): void {
  const vOff = vregOffset(insn.rt);
  // Compute address → TMP0 (i64)
  emitSimdEffAddr(b, insn.rn, insn.offset, insn.mode, TMP0);
  b.local_set(TMP0);

  // Store low 8 bytes
  b.local_get(TMP0); b.i32_wrap_i64();
  b.i32_const(vOff);
  b.i64_load(3, 0);
  b.i64_store(3, 0);

  // Store high 8 bytes
  b.local_get(TMP0); b.i32_wrap_i64();
  b.i32_const(vOff + 8);
  b.i64_load(3, 0);
  b.i64_store(3, 8);
}

/** STR B/H/S/D: store 8/16/32/64-bit from V register to guest memory */
function emitSimdStoreSmall(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'str_simd' },
  TMP0: number,
): void {
  const vOff = vregOffset(insn.rt);
  // Compute address → TMP0 (i64)
  emitSimdEffAddr(b, insn.rn, insn.offset, insn.mode, TMP0);
  b.local_set(TMP0);

  // Load value from V register (low bytes)
  b.local_get(TMP0); b.i32_wrap_i64();
  b.i32_const(vOff);
  b.i64_load(3, 0);

  // Store appropriate size
  switch (insn.size) {
    case 1: // B (8-bit)
      b.i64_store8(0, 0);
      break;
    case 2: // H (16-bit)
      b.i64_store16(1, 0);
      break;
    case 4: // S (32-bit)
      b.i64_store32(2, 0);
      break;
    case 8: // D (64-bit)
      b.i64_store(3, 0);
      break;
  }
}

/** LDR B/H/S/D: load 8/16/32/64-bit from guest memory to V register */
function emitSimdLoadSmall(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'ldr_simd' },
  TMP0: number,
): void {
  const vOff = vregOffset(insn.rt);
  // Compute address → TMP0 (i64)
  emitSimdEffAddr(b, insn.rn, insn.offset, insn.mode, TMP0);
  b.local_set(TMP0);

  // Load from memory into low qword of V register
  b.i32_const(vOff);
  b.local_get(TMP0); b.i32_wrap_i64();
  switch (insn.size) {
    case 1: // B (8-bit)
      b.i64_load8_u(0, 0);
      break;
    case 2: // H (16-bit)
      b.i64_load16_u(1, 0);
      break;
    case 4: // S (32-bit)
      b.i64_load32_u(2, 0);
      break;
    case 8: // D (64-bit)
      b.i64_load(3, 0);
      break;
  }
  b.i64_store(3, 0);

  // Zero the high qword
  b.i32_const(vOff + 8);
  b.i64_const(0n);
  b.i64_store(3, 0);
}

/** LDR Q: load 128-bit from guest memory to V register */
function emitSimdLoad128(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'ldr_simd' },
  TMP0: number,
): void {
  const vOff = vregOffset(insn.rt);
  // Compute address → TMP0 (i64)
  emitSimdEffAddr(b, insn.rn, insn.offset, insn.mode, TMP0);
  b.local_set(TMP0);

  // Load low 8 bytes
  b.i32_const(vOff);
  b.local_get(TMP0); b.i32_wrap_i64();
  b.i64_load(3, 0);
  b.i64_store(3, 0);

  // Load high 8 bytes
  b.i32_const(vOff + 8);
  b.local_get(TMP0); b.i32_wrap_i64();
  b.i64_load(3, 8);
  b.i64_store(3, 0);
}

/** STP Q: store pair of 128-bit V registers */
function emitSimdStorePair128(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'stp_simd' },
  TMP0: number,
): void {
  const v1Off = vregOffset(insn.rt);
  const v2Off = vregOffset(insn.rt2);

  // Compute base address → TMP0 (i64)
  emitSimdEffAddr(b, insn.rn, insn.offset, insn.mode, TMP0);
  b.local_set(TMP0);

  // Store first Q register (16 bytes)
  b.local_get(TMP0); b.i32_wrap_i64();
  b.i32_const(v1Off);
  b.i64_load(3, 0);
  b.i64_store(3, 0);

  b.local_get(TMP0); b.i32_wrap_i64();
  b.i32_const(v1Off + 8);
  b.i64_load(3, 0);
  b.i64_store(3, 8);

  // Store second Q register (16 bytes at offset +16)
  b.local_get(TMP0); b.i32_wrap_i64();
  b.i32_const(v2Off);
  b.i64_load(3, 0);
  b.i64_store(3, 16);

  b.local_get(TMP0); b.i32_wrap_i64();
  b.i32_const(v2Off + 8);
  b.i64_load(3, 0);
  b.i64_store(3, 24);
}

/** LDP Q: load pair of 128-bit V registers */
function emitSimdLoadPair128(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'ldp_simd' },
  TMP0: number,
): void {
  const v1Off = vregOffset(insn.rt);
  const v2Off = vregOffset(insn.rt2);

  // Compute base address → TMP0 (i64)
  emitSimdEffAddr(b, insn.rn, insn.offset, insn.mode, TMP0);
  b.local_set(TMP0);

  // Load first Q register
  b.i32_const(v1Off);
  b.local_get(TMP0); b.i32_wrap_i64();
  b.i64_load(3, 0);
  b.i64_store(3, 0);

  b.i32_const(v1Off + 8);
  b.local_get(TMP0); b.i32_wrap_i64();
  b.i64_load(3, 8);
  b.i64_store(3, 0);

  // Load second Q register
  b.i32_const(v2Off);
  b.local_get(TMP0); b.i32_wrap_i64();
  b.i64_load(3, 16);
  b.i64_store(3, 0);

  b.i32_const(v2Off + 8);
  b.local_get(TMP0); b.i32_wrap_i64();
  b.i64_load(3, 24);
  b.i64_store(3, 0);
}

/** STP S/D: store pair of 32/64-bit V registers */
function emitSimdStorePairSmall(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'stp_simd' },
  TMP0: number,
): void {
  const v1Off = vregOffset(insn.rt);
  const v2Off = vregOffset(insn.rt2);
  const size = insn.size; // 4 for S, 8 for D

  emitSimdEffAddr(b, insn.rn, insn.offset, insn.mode, TMP0);
  b.local_set(TMP0);

  // Store first register
  b.local_get(TMP0); b.i32_wrap_i64();
  b.i32_const(v1Off);
  b.i64_load(3, 0);
  if (size === 4) {
    b.i64_store32(2, 0);
  } else {
    b.i64_store(3, 0);
  }

  // Store second register at offset +size
  b.local_get(TMP0); b.i32_wrap_i64();
  b.i32_const(v2Off);
  b.i64_load(3, 0);
  if (size === 4) {
    b.i64_store32(2, size);
  } else {
    b.i64_store(3, size);
  }
}

/** LDP S/D: load pair of 32/64-bit V registers */
function emitSimdLoadPairSmall(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'ldp_simd' },
  TMP0: number,
): void {
  const v1Off = vregOffset(insn.rt);
  const v2Off = vregOffset(insn.rt2);
  const size = insn.size; // 4 for S, 8 for D

  emitSimdEffAddr(b, insn.rn, insn.offset, insn.mode, TMP0);
  b.local_set(TMP0);

  // Load first register (zero-extended to 64-bit)
  b.i32_const(v1Off);
  b.local_get(TMP0); b.i32_wrap_i64();
  if (size === 4) {
    b.i64_load32_u(2, 0);
  } else {
    b.i64_load(3, 0);
  }
  b.i64_store(3, 0);
  // Zero high qword
  b.i32_const(v1Off + 8);
  b.i64_const(0n);
  b.i64_store(3, 0);

  // Load second register at offset +size
  b.i32_const(v2Off);
  b.local_get(TMP0); b.i32_wrap_i64();
  if (size === 4) {
    b.i64_load32_u(2, size);
  } else {
    b.i64_load(3, size);
  }
  b.i64_store(3, 0);
  // Zero high qword
  b.i32_const(v2Off + 8);
  b.i64_const(0n);
  b.i64_store(3, 0);
}

/** DUP Vd.T, Wn/Xn — duplicate GPR to all lanes of vector */
function emitSimdDupGpr(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'simd_dup_gpr' },
): void {
  const vOff = vregOffset(insn.rd);
  const esize = insn.esize;
  const TMP0 = REG_COUNT; // borrow TMP0

  // Build a 64-bit qword with the value replicated across lanes
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
  // esize === 8: value is already a full 64-bit qword

  b.local_set(TMP0);

  // Store to low half of V register
  b.i32_const(vOff);
  b.local_get(TMP0);
  b.i64_store(3, 0);

  // Store to high half (or zero if 64-bit vector)
  b.i32_const(vOff + 8);
  if (insn.Q) {
    b.local_get(TMP0);
  } else {
    b.i64_const(0n);
  }
  b.i64_store(3, 0);
}

/** DUP Vd.T, Vn.Ts[index] — duplicate element to all lanes */
function emitSimdDupElem(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'simd_dup_elem' },
  TMP0: number,
): void {
  const srcOff = vregOffset(insn.rn);
  const dstOff = vregOffset(insn.rd);
  const esize = insn.esize;
  const index = insn.index;

  // Read the source element from the V register in memory
  const bytePos = index * esize;
  const qwordIdx = bytePos < 8 ? 0 : 1;
  const bitPos = (bytePos % 8) * 8;
  const mask = esize === 8 ? 0xFFFFFFFFFFFFFFFFn : (1n << BigInt(esize * 8)) - 1n;

  // Load the source qword
  b.i32_const(srcOff + qwordIdx * 8);
  b.i64_load(3, 0);
  if (bitPos > 0) {
    b.i64_const(BigInt(bitPos));
    b.i64_shr_u();
  }
  b.i64_const(mask);
  b.i64_and();

  // Now replicate element to fill 64-bit qword
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
  // esize === 8: already a full qword

  b.local_set(TMP0);

  // Store to both low and high halves of destination V register
  b.i32_const(dstOff);
  b.local_get(TMP0);
  b.i64_store(3, 0);

  if (insn.Q) {
    b.i32_const(dstOff + 8);
    b.local_get(TMP0);
    b.i64_store(3, 0);
  } else {
    // 64-bit vector: zero the high half
    b.i32_const(dstOff + 8);
    b.i64_const(0n);
    b.i64_store(3, 0);
  }
}

/** UMOV Xd/Wd, Vn.Ts[index] — extract element to GPR */
function emitSimdUmov(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'simd_umov' },
): void {
  const srcOff = vregOffset(insn.rn);
  const esize = insn.esize;
  const index = insn.index;
  const bytePos = index * esize;
  const qwordIdx = bytePos < 8 ? 0 : 1;
  const bitPos = (bytePos % 8) * 8;

  // Load the source qword
  b.i32_const(srcOff + qwordIdx * 8);
  b.i64_load(3, 0);

  if (bitPos > 0) {
    b.i64_const(BigInt(bitPos));
    b.i64_shr_u();
  }

  // Mask to element size
  if (esize < 8) {
    const mask = (1n << BigInt(esize * 8)) - 1n;
    b.i64_const(mask);
    b.i64_and();
  }

  // Store to GPR (esize determines if we use 32-bit or 64-bit target)
  emitSetReg(b, insn.rd, esize === 8 ? 1 : 0);
}

/** MOVI Vd.T, #imm — move immediate to vector */
function emitSimdMovi(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'simd_movi' },
): void {
  const vOff = vregOffset(insn.rd);

  // For cmode=0, op=0: MOVI Vd.xS, #imm8 (replicate byte to 32-bit lanes)
  // For cmode=1110, op=0: MOVI Vd.8B/16B, #imm8 (byte to all lanes)
  // For cmode=1110, op=1: MOVI Dd/FMOV (64-bit)
  // Simplified: expand imm8 based on cmode to a 64-bit pattern
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

function expandSimdImm(imm8: number, cmode: number, op: number): bigint {
  // Simplified expansion for common cases
  const cmodeH = cmode >> 1;
  if (op === 0) {
    switch (cmodeH) {
      case 0: { // MOVI Vd.xS, #imm8 (no shift)
        const val = BigInt(imm8);
        return val | (val << 32n);
      }
      case 1: { // MOVI Vd.xS, #imm8, LSL #8
        const val = BigInt(imm8) << 8n;
        return val | (val << 32n);
      }
      case 2: { // MOVI Vd.xS, #imm8, LSL #16
        const val = BigInt(imm8) << 16n;
        return val | (val << 32n);
      }
      case 3: { // MOVI Vd.xS, #imm8, LSL #24
        const val = BigInt(imm8) << 24n;
        return val | (val << 32n);
      }
      case 4: { // MOVI Vd.xH, #imm8
        const val = BigInt(imm8);
        return val | (val << 16n) | (val << 32n) | (val << 48n);
      }
      case 5: { // MOVI Vd.xH, #imm8, LSL #8
        const val = BigInt(imm8) << 8n;
        return val | (val << 16n) | (val << 32n) | (val << 48n);
      }
      case 7: { // MOVI Vd.8B/16B, #imm8 (byte broadcast)
        let val = BigInt(imm8);
        val |= val << 8n;
        val |= val << 16n;
        val |= val << 32n;
        return val;
      }
      default: {
        // MSL variants and others — expand as byte broadcast for now
        let val = BigInt(imm8);
        val |= val << 8n;
        val |= val << 16n;
        val |= val << 32n;
        return val;
      }
    }
  } else {
    // op=1: MVNI or FMOV variants
    // For MVNI, invert the result
    if (cmodeH <= 3) {
      const base = expandSimdImm(imm8, cmode, 0);
      return ~base & 0xFFFFFFFFFFFFFFFFn;
    }
    // FMOV — treat as zero for now
    return 0n;
  }
}

/** INS Vd.Ts[index], Wn/Xn — insert GPR into vector element */
function emitSimdInsGpr(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'simd_ins_gpr' },
): void {
  const vOff = vregOffset(insn.rd);
  const esize = insn.esize;
  const index = insn.index;
  const bytePos = index * esize;
  const qwordIdx = bytePos < 8 ? 0 : 1;
  const bitPos = (bytePos % 8) * 8;
  const mask = esize === 8 ? 0xFFFFFFFFFFFFFFFFn : (1n << BigInt(esize * 8)) - 1n;

  if (esize === 8) {
    // Simple: just write the GPR value to the correct qword
    b.i32_const(vOff + qwordIdx * 8);
    emitGetReg(b, insn.rn, 1);
    b.i64_store(3, 0);
  } else {
    // Read-modify-write: clear the element bits, OR in the new value
    const addr = vOff + qwordIdx * 8;
    b.i32_const(addr);

    // Load existing qword
    b.i32_const(addr);
    b.i64_load(3, 0);
    // Clear the element bits
    b.i64_const(~(mask << BigInt(bitPos)) & 0xFFFFFFFFFFFFFFFFn);
    b.i64_and();
    // OR in the new value
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

// ── FMOV: GPR → FP register ─────────────────────────────────────────────────

function emitFmovToFp(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'fmov_to_fp' },
): void {
  const vOff = vregOffset(insn.vd);
  // Write GPR value to low qword of V register
  b.i32_const(vOff);
  emitGetReg(b, insn.rn, insn.sf);
  if (insn.sf === 0) {
    // 32-bit: mask to 32 bits
    b.i64_const(0xFFFFFFFFn);
    b.i64_and();
  }
  b.i64_store(3, 0);
  // Clear upper qword
  b.i32_const(vOff + 8);
  b.i64_const(0n);
  b.i64_store(3, 0);
}

// ── FMOV: FP register → GPR ─────────────────────────────────────────────────

function emitFmovFromFp(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'fmov_from_fp' },
): void {
  const vOff = vregOffset(insn.vn);
  // Read low qword of V register → GPR
  b.i32_const(vOff);
  b.i64_load(3, 0);
  if (insn.sf === 0) {
    // 32-bit: mask to 32 bits
    b.i64_const(0xFFFFFFFFn);
    b.i64_and();
  }
  emitSetReg(b, insn.rd, insn.sf);
}

// ── FMOV: FP register → FP register ────────────────────────────────────────

function emitFmovFp(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'fmov_fp' },
): void {
  const srcOff = vregOffset(insn.vn);
  const dstOff = vregOffset(insn.vd);
  // Copy low qword
  b.i32_const(dstOff);
  b.i32_const(srcOff);
  b.i64_load(3, 0);
  if (insn.ftype === 0) {
    // Single precision: only copy 32 bits, zero the rest
    b.i64_const(0xFFFFFFFFn);
    b.i64_and();
  }
  b.i64_store(3, 0);
  // Clear upper qword
  b.i32_const(dstOff + 8);
  b.i64_const(0n);
  b.i64_store(3, 0);
}

// ── FP <-> integer conversions ──────────────────────────────────────────────

function emitFpIntConv(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'scvtf_int' | 'ucvtf_int' | 'fcvtzs_int' | 'fcvtzu_int' },
  TMP0: number,
  helpers?: HelperImports,
): void {
  if (!helpers) return;

  if (insn.kind === 'scvtf_int' || insn.kind === 'ucvtf_int') {
    // Integer → FP: read GPR, convert, store to V register
    emitGetReg(b, insn.rn, insn.sf);

    // Call the appropriate conversion helper
    if (insn.kind === 'scvtf_int') {
      b.call(insn.sf ? helpers.f64_from_s64Idx : helpers.f64_from_s32Idx);
    } else {
      b.call(insn.sf ? helpers.f64_from_u64Idx : helpers.f64_from_u32Idx);
    }
    b.local_set(TMP0);

    // If target is single precision, narrow f64 bits to f32 bits
    const vOff = vregOffset(insn.vd);
    b.i32_const(vOff);
    if (insn.ftype === 0) {
      // Single precision: convert f64 bits to f32 bits
      b.local_get(TMP0);
      b.call(helpers.f64_to_f32Idx);
    } else {
      b.local_get(TMP0);
    }
    b.i64_store(3, 0);
    // Clear upper
    b.i32_const(vOff + 8);
    b.i64_const(0n);
    b.i64_store(3, 0);
  } else {
    // FP → Integer: read V register, convert, store to GPR
    const vOff = vregOffset(insn.vn);
    b.i32_const(vOff);
    b.i64_load(3, 0);

    // If source is single precision, widen f32 bits to f64 bits
    if (insn.ftype === 0) {
      b.call(helpers.f32_to_f64Idx);
    }

    // Call the appropriate conversion helper
    if (insn.kind === 'fcvtzs_int') {
      b.call(insn.sf ? helpers.f64_to_s64Idx : helpers.f64_to_s32Idx);
    } else {
      b.call(insn.sf ? helpers.f64_to_u64Idx : helpers.f64_to_u32Idx);
    }
    emitSetReg(b, insn.rd, insn.sf);
  }
}

// ── FP arithmetic (2-source) ────────────────────────────────────────────────

function emitFpArith2(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'fmul' | 'fdiv' | 'fadd' | 'fsub' },
  TMP0: number,
  helpers?: HelperImports,
): void {
  if (!helpers) return;

  const vnOff = vregOffset(insn.vn);
  const vmOff = vregOffset(insn.vm);
  const vdOff = vregOffset(insn.vd);

  // Load operands from V registers
  b.i32_const(vnOff);
  b.i64_load(3, 0);
  if (insn.ftype === 0) { b.call(helpers.f32_to_f64Idx); }

  b.i32_const(vmOff);
  b.i64_load(3, 0);
  if (insn.ftype === 0) { b.call(helpers.f32_to_f64Idx); }

  // Perform operation
  switch (insn.kind) {
    case 'fmul': b.call(helpers.f64_mulIdx); break;
    case 'fdiv': b.call(helpers.f64_divIdx); break;
    case 'fadd': b.call(helpers.f64_addIdx); break;
    case 'fsub': b.call(helpers.f64_subIdx); break;
  }
  b.local_set(TMP0);

  // Store result
  b.i32_const(vdOff);
  if (insn.ftype === 0) {
    b.local_get(TMP0);
    b.call(helpers.f64_to_f32Idx);
  } else {
    b.local_get(TMP0);
  }
  b.i64_store(3, 0);
  // Clear upper
  b.i32_const(vdOff + 8);
  b.i64_const(0n);
  b.i64_store(3, 0);
}

// ── FP misc (1-source) ──────────────────────────────────────────────────────

function emitFpMisc1(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'fabs' | 'fneg' },
  TMP0: number,
  helpers?: HelperImports,
): void {
  if (!helpers) return;

  const vnOff = vregOffset(insn.vn);
  const vdOff = vregOffset(insn.vd);

  b.i32_const(vnOff);
  b.i64_load(3, 0);
  if (insn.ftype === 0) { b.call(helpers.f32_to_f64Idx); }

  switch (insn.kind) {
    case 'fabs': b.call(helpers.f64_absIdx); break;
    case 'fneg': b.call(helpers.f64_negIdx); break;
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

// ── FP precision conversion ─────────────────────────────────────────────────

function emitFcvt(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'fcvt' },
  TMP0: number,
  helpers?: HelperImports,
): void {
  if (!helpers) return;

  const vnOff = vregOffset(insn.vn);
  const vdOff = vregOffset(insn.vd);

  b.i32_const(vnOff);
  b.i64_load(3, 0);

  // Source ftype → f64 bits
  if (insn.ftype === 0) {
    b.call(helpers.f32_to_f64Idx);
  }
  b.local_set(TMP0);

  // f64 bits → dest ftype
  b.i32_const(vdOff);
  if (insn.destFtype === 0) {
    // Dest is f32
    b.local_get(TMP0);
    b.call(helpers.f64_to_f32Idx);
  } else {
    // Dest is f64
    b.local_get(TMP0);
  }
  b.i64_store(3, 0);
  b.i32_const(vdOff + 8);
  b.i64_const(0n);
  b.i64_store(3, 0);
}

// ── SIMD three-same generic (via JS helper) ─────────────────────────────────

function emitSimdThreeSameGeneric(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: string; Q: number; rd: number; rn: number; rm: number },
  TMP0: number,
  helpers?: HelperImports,
): void {
  if (!helpers) return;

  // Map instruction kind to operation code
  let op = 0, size = 0, extra = 0;
  switch (insn.kind) {
    case 'simd_add': op = SIMD_ADD; break;
    case 'simd_sub': op = SIMD_SUB; break;
    case 'simd_cmeq_reg': op = SIMD_CMEQ; break;
    case 'simd_cmgt_reg': op = SIMD_CMGT; break;
    case 'simd_cmhi_reg': op = SIMD_CMHI; break;
    case 'simd_cmge_reg': op = SIMD_CMGE; break;
    case 'simd_cmhs_reg': op = SIMD_CMHS; break;
    case 'simd_umaxp': op = SIMD_UMAXP; break;
    case 'simd_addp': op = SIMD_ADDP; break;
    case 'simd_tbl': op = SIMD_TBL; extra = (insn as any).len ?? 0; break;
    case 'simd_zip1': op = SIMD_ZIP1; break;
    case 'simd_zip2': op = SIMD_ZIP2; break;
    case 'simd_uzp1': op = SIMD_UZP1; break;
    case 'simd_uzp2': op = SIMD_UZP2; break;
    case 'simd_trn1': op = SIMD_TRN1; break;
    case 'simd_trn2': op = SIMD_TRN2; break;
    default: return; // unknown, emit nothing (will HALT)
  }
  size = (insn as any).size ?? 0;

  // Pack: op|Q|size|extra|rd|rn|rm|0 into a single i64
  const packed = (BigInt(op) << 56n) | (BigInt(insn.Q) << 48n) | (BigInt(size) << 40n) |
    (BigInt(extra) << 32n) | (BigInt(insn.rd) << 24n) | (BigInt(insn.rn) << 16n) |
    (BigInt(insn.rm) << 8n);

  b.i64_const(packed);
  b.i64_const(0n); // unused second arg
  b.call(helpers.simd_opIdx);
  b.drop(); // discard return value
}

// ── FMOV immediate ──────────────────────────────────────────────────────────

function emitFmovImm(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'fmov_imm' },
): void {
  const vdOff = vregOffset(insn.vd);
  b.i32_const(vdOff);
  b.i64_const(insn.bits);
  b.i64_store(3, 0);
  // Clear upper qword
  b.i32_const(vdOff + 8);
  b.i64_const(0n);
  b.i64_store(3, 0);
}

// ── FP compare ──────────────────────────────────────────────────────────────

function emitFpCmp(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'fcmp' },
  helpers?: HelperImports,
): void {
  if (!helpers) return;

  const vnOff = vregOffset(insn.vn);

  // Load Vn
  b.i32_const(vnOff);
  b.i64_load(3, 0);
  if (insn.ftype === 0) { b.call(helpers.f32_to_f64Idx); }

  if (insn.withZero) {
    // Compare with #0.0 (0n is +0.0 in f64 bits)
    b.i64_const(0n);
  } else {
    // Load Vm
    const vmOff = vregOffset(insn.vm);
    b.i32_const(vmOff);
    b.i64_load(3, 0);
    if (insn.ftype === 0) { b.call(helpers.f32_to_f64Idx); }
  }

  b.call(helpers.f64_cmpIdx);
  b.local_set(NZCV_LOCAL);
}

// ── FP fused multiply-add/subtract (3-source) ──────────────────────────────

function emitFpMulAdd(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'fmadd' | 'fmsub' | 'fnmadd' | 'fnmsub' },
  TMP0: number,
  helpers?: HelperImports,
): void {
  if (!helpers) return;

  const vnOff = vregOffset(insn.vn);
  const vmOff = vregOffset(insn.vm);
  const vaOff = vregOffset(insn.va);
  const vdOff = vregOffset(insn.vd);
  const isF32 = insn.ftype === 0;

  // FMADD:  Rd = Ra + Rn * Rm
  // FMSUB:  Rd = Ra - Rn * Rm
  // FNMADD: Rd = -(Ra + Rn * Rm)
  // FNMSUB: Rd = Rn * Rm - Ra

  if (insn.kind === 'fnmsub') {
    // Rd = product - Ra: push Rn, Rm, mul, push Ra, sub
    b.i32_const(vnOff); b.i64_load(3, 0);
    if (isF32) { b.call(helpers.f32_to_f64Idx); }
    b.i32_const(vmOff); b.i64_load(3, 0);
    if (isF32) { b.call(helpers.f32_to_f64Idx); }
    b.call(helpers.f64_mulIdx);
    b.i32_const(vaOff); b.i64_load(3, 0);
    if (isF32) { b.call(helpers.f32_to_f64Idx); }
    b.call(helpers.f64_subIdx);
  } else {
    // All others start with Ra, then compute Rn*Rm
    b.i32_const(vaOff); b.i64_load(3, 0);
    if (isF32) { b.call(helpers.f32_to_f64Idx); }
    b.i32_const(vnOff); b.i64_load(3, 0);
    if (isF32) { b.call(helpers.f32_to_f64Idx); }
    b.i32_const(vmOff); b.i64_load(3, 0);
    if (isF32) { b.call(helpers.f32_to_f64Idx); }
    b.call(helpers.f64_mulIdx);

    if (insn.kind === 'fmadd' || insn.kind === 'fnmadd') {
      b.call(helpers.f64_addIdx);    // Ra + product
    } else {
      b.call(helpers.f64_subIdx);    // Ra - product (FMSUB)
    }

    if (insn.kind === 'fnmadd') {
      b.call(helpers.f64_negIdx);    // negate result
    }
  }

  b.local_set(TMP0);

  // Store result
  b.i32_const(vdOff);
  if (isF32) {
    b.local_get(TMP0);
    b.call(helpers.f64_to_f32Idx);
  } else {
    b.local_get(TMP0);
  }
  b.i64_store(3, 0);
  // Clear upper
  b.i32_const(vdOff + 8);
  b.i64_const(0n);
  b.i64_store(3, 0);
}

// ── SIMD shift long (SSHLL/USHLL) ───────────────────────────────────────────

function emitSimdShiftLong(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'simd_sshll' | 'simd_ushll' },
  TMP0: number,
): void {
  const srcOff = vregOffset(insn.rn);
  const dstOff = vregOffset(insn.rd);
  const esize = insn.esize;       // Source element size in bytes
  const dstEsize = esize * 2;     // Destination element size (doubled)
  const shift = insn.shift;
  const signed = insn.kind === 'simd_sshll';
  const srcBits = esize * 8;
  const dstBits = dstEsize * 8;

  // Q=0: process lower half (elements 0..n-1), Q=1: process upper half (elements n..2n-1)
  const numElems = insn.Q === 0 ? (8 / esize) : (8 / esize);  // Always process 8 bytes of source
  const srcBase = insn.Q === 0 ? 0 : 8;  // Source starts at byte 0 or 8

  // Process elements. For widening, we go from smaller elements to larger.
  // We need to be careful about overlapping src/dst — process in reverse if needed.
  // Since dest is always 128-bit and source is 64-bit half, they may overlap if src==dst.
  // Process from highest element to lowest to avoid clobbering.

  for (let i = numElems - 1; i >= 0; i--) {
    const srcBytePos = srcBase + i * esize;
    const dstBytePos = i * dstEsize;
    const srcQword = srcBytePos < 8 ? 0 : 1;
    const srcBitPos = (srcBytePos % 8) * 8;
    const dstQword = dstBytePos < 8 ? 0 : 1;
    const dstBitPos = (dstBytePos % 8) * 8;

    // Load source element
    b.i32_const(srcOff + srcQword * 8);
    b.i64_load(3, 0);
    if (srcBitPos > 0) {
      b.i64_const(BigInt(srcBitPos));
      b.i64_shr_u();
    }

    // Mask to source element size
    if (esize < 8) {
      const mask = (1n << BigInt(srcBits)) - 1n;
      b.i64_const(mask);
      b.i64_and();
    }

    // Sign-extend if SSHLL
    if (signed && esize < 8) {
      // Sign-extend: shift left then arithmetic shift right
      const extShift = BigInt(64 - srcBits);
      b.i64_const(extShift);
      b.i64_shl();
      b.i64_const(extShift);
      b.i64_shr_s();
    }

    // Apply shift
    if (shift > 0) {
      b.i64_const(BigInt(shift));
      b.i64_shl();
    }

    // Mask to destination element size
    if (dstEsize < 8) {
      const dstMask = (1n << BigInt(dstBits)) - 1n;
      b.i64_const(dstMask);
      b.i64_and();
    }

    b.local_set(TMP0);

    // Write to destination
    if (dstBitPos === 0 && dstEsize >= 8) {
      // Simple case: write full qword
      b.i32_const(dstOff + dstQword * 8);
      b.local_get(TMP0);
      b.i64_store(3, 0);
    } else {
      // Read-modify-write the destination qword
      const dstMask = (1n << BigInt(dstBits)) - 1n;
      const addr = dstOff + dstQword * 8;
      b.i32_const(addr);
      // Load existing
      b.i32_const(addr);
      b.i64_load(3, 0);
      // Clear element bits
      b.i64_const(~(dstMask << BigInt(dstBitPos)) & 0xFFFFFFFFFFFFFFFFn);
      b.i64_and();
      // OR in new value
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

// ── Emit a linear block within a region ─────────────────────────────────────

function emitRegionLinearBlock(
  b: WasmModuleBuilder,
  block: BasicBlock,
  _blockIndex: number,
  blockMap: Map<bigint, number>,
  dispatchDepth: number,
  exitDepth: number,
  blockIdLocal: number,
  TMP0: number, TMP1: number, TMP2: number, TMP3: number,
  helpers?: HelperImports,
): void {
  const startPC = Number(block.startPC);
  const insns = block.insns;

  // Emit all non-branch instructions
  for (let i = 0; i < insns.length - 1; i++) {
    const pc = startPC + i * 4;
    emitInsn(b, insns[i], pc, TMP0, TMP1, TMP2, TMP3, helpers);
  }

  // Handle the terminating instruction with region-aware branching
  const lastInsn = insns[insns.length - 1];
  const lastPC = startPC + (insns.length - 1) * 4;
  emitRegionBranch(b, lastInsn, lastPC, block, blockMap,
    dispatchDepth, exitDepth, blockIdLocal);
}

// ── Emit a self-loop block within a region ──────────────────────────────────

function emitRegionSelfLoop(
  b: WasmModuleBuilder,
  block: BasicBlock,
  _blockIndex: number,
  blockMap: Map<bigint, number>,
  dispatchDepth: number,
  exitDepth: number,
  blockIdLocal: number,
  TMP0: number, TMP1: number, TMP2: number, TMP3: number,
  helpers?: HelperImports,
): void {
  const startPC = Number(block.startPC);
  const insns = block.insns;
  const lastInsn = insns[insns.length - 1];

  // Detect CMP+B.cond fusion
  const penult = insns.length >= 2 ? insns[insns.length - 2] : null;
  const canFuse = lastInsn.kind === 'b_cond' && penult !== null &&
    ((penult.kind === 'sub_reg' && penult.setFlags && penult.rd === 31) ||
     (penult.kind === 'sub_imm' && penult.setFlags && penult.rd === 31)) &&
    !insnsAccessNZCV(insns.slice(0, insns.length - 2));

  // block $selfExit
  b.block(BLOCKTYPE_VOID);
  // loop $inner
  b.loop(BLOCKTYPE_VOID);

  // Increment iteration counter per iteration
  b.local_get(ITER_LOCAL);
  b.i64_const(BigInt(insns.length));
  b.i64_add();
  b.local_set(ITER_LOCAL);

  // Emit body (all insns except last, and penultimate if fusing)
  const skipLast = canFuse ? 2 : 1;
  for (let i = 0; i < insns.length - skipLast; i++) {
    const pc = startPC + i * 4;
    emitInsn(b, insns[i], pc, TMP0, TMP1, TMP2, TMP3, helpers);
  }

  // Handle the backward branch
  if (canFuse && lastInsn.kind === 'b_cond') {
    const cmpInsn = penult!;
    emitFusedCmpBranch(b, cmpInsn as any, lastInsn.cond);
    // br_if(0) is emitted by emitFusedCmpBranch → back to $inner
  } else if (lastInsn.kind === 'b') {
    b.br(0); // back to $inner
  } else if (lastInsn.kind === 'b_cond') {
    emitConditionTest(b, lastInsn.cond, NZCV_LOCAL);
    b.br_if(0);
  } else if (lastInsn.kind === 'cbz') {
    b.local_get(regLocal(lastInsn.rt));
    b.i64_eqz();
    b.br_if(0);
  } else if (lastInsn.kind === 'cbnz') {
    b.local_get(regLocal(lastInsn.rt));
    b.i64_const(0n);
    b.i64_ne();
    b.br_if(0);
  } else if (lastInsn.kind === 'tbz') {
    b.local_get(regLocal(lastInsn.rt));
    b.i64_const(BigInt(lastInsn.bit));
    b.i64_shr_u();
    b.i64_const(1n);
    b.i64_and();
    b.i64_eqz(); // tbz: loop back if bit is zero
    b.br_if(0);
  } else if (lastInsn.kind === 'tbnz') {
    b.local_get(regLocal(lastInsn.rt));
    b.i64_const(BigInt(lastInsn.bit));
    b.i64_shr_u();
    b.i64_const(1n);
    b.i64_and();
    b.i64_const(0n);
    b.i64_ne(); // tbnz: loop back if bit is non-zero
    b.br_if(0);
  }

  // Fall through self-loop: determine where the fallthrough goes
  if (canFuse) {
    // Recompute NZCV on exit
    const cmpInsn = penult!;
    if (cmpInsn.kind === 'sub_reg') {
      emitGetReg(b, cmpInsn.rn, cmpInsn.sf);
      b.local_tee(TMP0);
      emitGetReg(b, cmpInsn.rm, cmpInsn.sf);
      emitShift(b, cmpInsn.shift, cmpInsn.amount, cmpInsn.sf);
      if (cmpInsn.sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
      b.local_tee(TMP1);
      b.i64_sub();
      if (cmpInsn.sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
      b.local_set(TMP2);
    } else if (cmpInsn.kind === 'sub_imm') {
      emitGetRegSP(b, cmpInsn.rn, cmpInsn.sf);
      b.local_tee(TMP0);
      b.i64_const(BigInt(cmpInsn.imm));
      b.local_tee(TMP1);
      b.i64_sub();
      if (cmpInsn.sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
      b.local_set(TMP2);
    }
    emitComputeFlags(b, false, (penult as any).sf, TMP0, TMP1, TMP2);
  }

  // Fallthrough PC from self-loop exit
  const lastPC = startPC + (insns.length - 1) * 4;
  const fallthroughPC = BigInt(lastPC + 4);
  const fallthroughIdx = blockMap.get(fallthroughPC);

  if (fallthroughIdx !== undefined) {
    // Internal: dispatch to fallthrough block
    b.i32_const(fallthroughIdx);
    b.local_set(blockIdLocal);
    b.br(dispatchDepth + 2); // +2 because we're inside $selfExit block + $inner loop
  } else {
    // External: exit region
    b.i64_const(fallthroughPC);
    b.local_set(NEXT_PC_LOCAL);
    b.br(exitDepth + 2); // +2 for the extra nesting
  }

  b.end(); // end $inner loop
  b.unreachable();
  b.end(); // end $selfExit block
}

// ── Emit region-aware branch for a block's terminating instruction ───────────

function emitRegionBranch(
  b: WasmModuleBuilder,
  insn: DecodedInsn,
  pc: number,
  block: BasicBlock,
  blockMap: Map<bigint, number>,
  dispatchDepth: number,
  exitDepth: number,
  blockIdLocal: number,
): void {
  switch (insn.kind) {
    case 'b': {
      const target = BigInt(pc + insn.offset);
      const idx = blockMap.get(target);
      if (idx !== undefined) {
        b.i32_const(idx);
        b.local_set(blockIdLocal);
        b.br(dispatchDepth);
      } else {
        b.i64_const(target);
        b.local_set(NEXT_PC_LOCAL);
        b.br(exitDepth);
      }
      break;
    }
    case 'bl': {
      // Set link register
      b.i64_const(BigInt(pc + 4));
      b.local_set(regLocal(30));
      // BL target is external (function call), fallthrough is pc+4
      const target = BigInt(pc + insn.offset);
      b.i64_const(target);
      b.local_set(NEXT_PC_LOCAL);
      b.br(exitDepth);
      break;
    }
    case 'b_cond': {
      const takenPC = BigInt(pc + insn.offset);
      const fallPC = BigInt(pc + 4);
      emitRegionCondBranch(b, blockMap, dispatchDepth, exitDepth, blockIdLocal,
        takenPC, fallPC, () => emitConditionTest(b, insn.cond, NZCV_LOCAL));
      break;
    }
    case 'cbz': {
      const takenPC = BigInt(pc + insn.offset);
      const fallPC = BigInt(pc + 4);
      emitRegionCondBranch(b, blockMap, dispatchDepth, exitDepth, blockIdLocal,
        takenPC, fallPC, () => {
          emitGetReg(b, insn.rt, insn.sf);
          b.i64_eqz();
        });
      break;
    }
    case 'cbnz': {
      const takenPC = BigInt(pc + insn.offset);
      const fallPC = BigInt(pc + 4);
      emitRegionCondBranch(b, blockMap, dispatchDepth, exitDepth, blockIdLocal,
        takenPC, fallPC, () => {
          emitGetReg(b, insn.rt, insn.sf);
          b.i64_const(0n);
          b.i64_ne();
        });
      break;
    }
    case 'tbz': {
      const takenPC = BigInt(pc + insn.offset);
      const fallPC = BigInt(pc + 4);
      emitRegionCondBranch(b, blockMap, dispatchDepth, exitDepth, blockIdLocal,
        takenPC, fallPC, () => {
          b.local_get(regLocal(insn.rt));
          b.i64_const(BigInt(insn.bit));
          b.i64_shr_u();
          b.i64_const(1n);
          b.i64_and();
          b.i64_eqz(); // tbz: branch if bit is zero
          // i64_eqz returns i32
        });
      break;
    }
    case 'tbnz': {
      const takenPC = BigInt(pc + insn.offset);
      const fallPC = BigInt(pc + 4);
      emitRegionCondBranch(b, blockMap, dispatchDepth, exitDepth, blockIdLocal,
        takenPC, fallPC, () => {
          b.local_get(regLocal(insn.rt));
          b.i64_const(BigInt(insn.bit));
          b.i64_shr_u();
          b.i64_const(1n);
          b.i64_and();
          b.i64_const(0n);
          b.i64_ne(); // tbnz: branch if bit is non-zero
        });
      break;
    }
    case 'br': {
      b.local_get(regLocal(insn.rn));
      b.local_set(NEXT_PC_LOCAL);
      b.br(exitDepth);
      break;
    }
    case 'blr': {
      b.i64_const(BigInt(pc + 4));
      b.local_set(regLocal(30));
      b.local_get(regLocal(insn.rn));
      b.local_set(NEXT_PC_LOCAL);
      b.br(exitDepth);
      break;
    }
    case 'ret': {
      b.local_get(regLocal(insn.rn));
      b.local_set(NEXT_PC_LOCAL);
      b.br(exitDepth);
      break;
    }
    case 'svc': {
      b.i64_const(HALT_PC);
      b.local_set(NEXT_PC_LOCAL);
      b.br(exitDepth);
      break;
    }
    default: {
      // Non-branch last instruction (e.g., nop, unknown): fall through
      b.i64_const(block.endPC);
      b.local_set(NEXT_PC_LOCAL);
      b.br(exitDepth);
      break;
    }
  }
}

// ── Emit a conditional branch with region-aware targets ─────────────────────

function emitRegionCondBranch(
  b: WasmModuleBuilder,
  blockMap: Map<bigint, number>,
  dispatchDepth: number,
  exitDepth: number,
  blockIdLocal: number,
  takenPC: bigint,
  fallPC: bigint,
  emitTest: () => void,
): void {
  const takenIdx = blockMap.get(takenPC);
  const fallIdx = blockMap.get(fallPC);

  emitTest(); // pushes i32 condition

  b.if_(BLOCKTYPE_VOID);
  // Taken path
  if (takenIdx !== undefined) {
    b.i32_const(takenIdx);
    b.local_set(blockIdLocal);
    b.br(dispatchDepth + 1); // +1 for the if block
  } else {
    b.i64_const(takenPC);
    b.local_set(NEXT_PC_LOCAL);
    b.br(exitDepth + 1);
  }
  b.else_();
  // Fallthrough path
  if (fallIdx !== undefined) {
    b.i32_const(fallIdx);
    b.local_set(blockIdLocal);
    b.br(dispatchDepth + 1);
  } else {
    b.i64_const(fallPC);
    b.local_set(NEXT_PC_LOCAL);
    b.br(exitDepth + 1);
  }
  b.end();
}
