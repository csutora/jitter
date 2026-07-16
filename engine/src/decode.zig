//! arm64 instruction decoder. takes a 32-bit instruction word, returns
//! a tagged-union op. reference: arm architecture reference manual,
//! section c4 (instruction encoding).

const std = @import("std");


pub const Kind = enum(u32) {
    unknown = 0,
    nop,

    // data-processing immediate
    add_imm,
    sub_imm,
    and_imm,
    orr_imm,
    eor_imm,
    movn,
    movz,
    movk,
    adr,
    adrp,
    bfm,
    sbfm,
    ubfm,
    extr,

    // data-processing register
    add_reg,
    sub_reg,
    add_ext_reg,
    sub_ext_reg,
    and_reg,
    orr_reg,
    eor_reg,
    bic_reg,
    orn_reg,
    eon_reg,
    lsl_reg,
    lsr_reg,
    asr_reg,
    ror_reg,
    udiv,
    sdiv,
    madd,
    msub,
    smulh,
    umulh,
    smaddl,
    smsubl,
    umaddl,
    umsubl,
    csel,
    csinc,
    csinv,
    csneg,
    ccmp_reg,
    ccmp_imm,
    adc,
    sbc,
    rev,
    clz,
    cls,
    rbit,

    // branches
    b,
    bl,
    br,
    blr,
    ret,
    b_cond,
    cbz,
    cbnz,
    tbz,
    tbnz,
    svc,
    brk,
    hint,

    // load/store
    ldr_imm,
    ldrb_imm,
    ldrh_imm,
    ldrsb_imm,
    ldrsh_imm,
    ldrsw_imm,
    str_imm,
    strb_imm,
    strh_imm,
    ldr_reg,
    ldrb_reg,
    ldrh_reg,
    ldrsb_reg,
    ldrsh_reg,
    ldrsw_reg,
    str_reg,
    strb_reg,
    strh_reg,
    ldr_lit,
    ldr_lit_fp,
    ldp,
    stp,
    ldxr,
    stxr,

    // system
    mrs,
    msr,
    barrier,

    // float / simd
    fcmp,
    fcvt,
    fcsel,
    fmov_imm,
    fmov_fp,
    fmov_to_fp,
    fmov_from_fp,
    fmov_top_to_fp,
    fmov_top_from_fp,
    fabs,
    fneg,
    fsqrt,
    fadd,
    fsub,
    fmul,
    fdiv,
    fmadd,
    fmsub,
    fnmadd,
    fnmsub,
    fminnm,
    fmaxnm,
    scvtf_int,
    ucvtf_int,
    fcvtzs_int,
    fcvtzu_int,
    fp_fixed_cvt,
    simd_scalar_cvt,
    simd_scalar_shift,
    simd_movi,
    simd_dup_gpr,
    simd_dup_elem,
    simd_umov,
    simd_ins_gpr,
    simd_load,
    simd_store,
    simd_load_pair,
    simd_store_pair,
    simd_load_reg,
    simd_store_reg,
    simd_add,
    simd_sub,
    simd_cmeq_reg,
    simd_cmgt_reg,
    simd_cmhi_reg,
    simd_cmge_reg,
    simd_cmhs_reg,
    simd_umaxp,
    simd_uminp,
    simd_addhn,
    simd_ext,
    simd_addp,
    simd_orr_reg,
    simd_and_reg,
    simd_bic_reg,
    simd_orn_reg,
    simd_eor_reg,
    simd_bsl,
    simd_bit,
    simd_bif,
    simd_not,
    simd_neg,
    simd_abs,
    simd_cnt,
    simd_cls_vec,
    simd_clz_vec,
    simd_sshll,
    simd_ushll,
    simd_shrn,
    simd_shift_imm,
    simd_rev,
    simd_ins_elem,
    simd_dup_scalar,
    simd_rbit,
    simd_idx_mul,
    simd_xtn,
    crypto_aes,
    crypto_pmull,
    crypto_sha,
    simd_tbl,
    simd_zip1,
    simd_zip2,
    simd_uzp1,
    simd_uzp2,
    simd_trn1,
    simd_trn2,

    // vector neon (native wasm v128)
    vec_fp3, // fp three-same: fadd/fsub/fmul/fdiv/fmax/fmin/fmla/fcmeq...
    vec_fp2, // fp two-reg-misc: fneg/fabs/fsqrt/frint/fcvt vector
    vec_int3, // int three-same: add/sub/mul/cmp/min/max/saturating
    vec_int2, // int two-reg-misc: neg/abs/shl/shr...
    vec_across, // across-lanes reductions: addv/uaddlv/umaxv/uminv...
    vec_widen, // three-different widening: smull/umull/saddl/ssubl/smlal...
    simd_ldst_multi, // LD1/ST1 .. LD4/ST4 multiple structures
    simd_ld1r, // LD1R replicate
    simd_ldst1_single, // LD1/ST1 single-lane

    // lse atomics (armv8.1)
    cas,
    lse,
};

pub const Op = union(Kind) {
    unknown: u32,
    nop: void,

    add_imm: ArithImm,
    sub_imm: ArithImm,
    and_imm: LogicalImm,
    orr_imm: LogicalImm,
    eor_imm: LogicalImm,
    movn: MovWide,
    movz: MovWide,
    movk: MovWide,
    adr: PcRel,
    adrp: PcRel,
    bfm: BitfieldImm,
    sbfm: BitfieldImm,
    ubfm: BitfieldImm,
    extr: Extr,

    add_reg: ArithReg,
    sub_reg: ArithReg,
    add_ext_reg: ArithExtReg,
    sub_ext_reg: ArithExtReg,
    and_reg: LogicalReg,
    orr_reg: LogicalReg,
    eor_reg: LogicalReg,
    bic_reg: LogicalReg,
    orn_reg: LogicalReg,
    eon_reg: LogicalReg,
    lsl_reg: ShiftReg,
    lsr_reg: ShiftReg,
    asr_reg: ShiftReg,
    ror_reg: ShiftReg,
    udiv: DivReg,
    sdiv: DivReg,
    madd: MulAdd,
    msub: MulAdd,
    smulh: MulHigh,
    umulh: MulHigh,
    smaddl: MulAdd,
    smsubl: MulAdd,
    umaddl: MulAdd,
    umsubl: MulAdd,
    csel: CondSelect,
    csinc: CondSelect,
    csinv: CondSelect,
    csneg: CondSelect,
    ccmp_reg: CondCmpReg,
    ccmp_imm: CondCmpImm,
    adc: AdcSbc,
    sbc: AdcSbc,
    rev: RevOp,
    clz: ClzRbit,
    cls: ClzRbit,
    rbit: ClzRbit,

    b: BrUncond,
    bl: BrUncond,
    br: BrReg,
    blr: BrReg,
    ret: BrReg,
    b_cond: BrCond,
    cbz: BrCmpZ,
    cbnz: BrCmpZ,
    tbz: BrTestBit,
    tbnz: BrTestBit,
    svc: Imm16,
    brk: Imm16,
    hint: u8,

    ldr_imm: LoadStoreImm,
    ldrb_imm: LoadStoreImm,
    ldrh_imm: LoadStoreImm,
    ldrsb_imm: LoadStoreImm,
    ldrsh_imm: LoadStoreImm,
    ldrsw_imm: LoadStoreImm,
    str_imm: LoadStoreImm,
    strb_imm: LoadStoreImm,
    strh_imm: LoadStoreImm,
    ldr_reg: LoadStoreReg,
    ldrb_reg: LoadStoreReg,
    ldrh_reg: LoadStoreReg,
    ldrsb_reg: LoadStoreReg,
    ldrsh_reg: LoadStoreReg,
    ldrsw_reg: LoadStoreReg,
    str_reg: LoadStoreReg,
    strb_reg: LoadStoreReg,
    strh_reg: LoadStoreReg,
    ldr_lit: LoadLit,
    ldr_lit_fp: LoadLitFp,
    ldp: LoadStorePair,
    stp: LoadStorePair,
    ldxr: LoadStoreEx,
    stxr: LoadStoreEx,

    mrs: SysReg,
    msr: SysReg,
    barrier: BarrierOp,

    fcmp: Fcmp,
    fcvt: Fcvt,
    fcsel: Fcsel,
    fmov_imm: FmovImm,
    fmov_fp: FpArith1, // FMOV Vd, Vn — register copy
    fmov_to_fp: FmovGprFp, // FMOV Sd/Dd from Wn/Xn
    fmov_from_fp: FmovGprFp, // FMOV Wd/Xd from Sn/Dn
    fmov_top_to_fp: FmovTop, // FMOV Vd.D[1], Xn
    fmov_top_from_fp: FmovTop, // FMOV Xd, Vn.D[1]
    fabs: FpArith1,
    fneg: FpArith1,
    fsqrt: FpArith1,
    fadd: FpArith2,
    fsub: FpArith2,
    fmul: FpArith2,
    fdiv: FpArith2,
    fmadd: FpArith3,
    fmsub: FpArith3,
    fnmadd: FpArith3,
    fnmsub: FpArith3,
    fminnm: FpArith2,
    fmaxnm: FpArith2,
    scvtf_int: FpIntCvt,
    ucvtf_int: FpIntCvt,
    fcvtzs_int: FpIntCvt,
    fcvtzu_int: FpIntCvt,
    fp_fixed_cvt: FpFixedCvt,
    simd_scalar_cvt: SimdScalarCvt,
    simd_scalar_shift: SimdScalarShift,
    simd_movi: SimdMovi,
    simd_dup_gpr: SimdDupGpr,
    simd_dup_elem: SimdDupElem,
    simd_umov: SimdElem,
    simd_ins_gpr: SimdElem,
    simd_load: SimdMem,
    simd_store: SimdMem,
    simd_load_pair: SimdMemPair,
    simd_store_pair: SimdMemPair,
    simd_load_reg: SimdMemReg,
    simd_store_reg: SimdMemReg,
    simd_add: SimdThreeSame,
    simd_sub: SimdThreeSame,
    simd_cmeq_reg: SimdThreeSame,
    simd_cmgt_reg: SimdThreeSame,
    simd_cmhi_reg: SimdThreeSame,
    simd_cmge_reg: SimdThreeSame,
    simd_cmhs_reg: SimdThreeSame,
    simd_umaxp: SimdThreeSame,
    simd_uminp: SimdThreeSame,
    simd_addhn: SimdThreeSame,
    simd_ext: SimdExt,
    simd_addp: SimdThreeSame,
    simd_orr_reg: SimdThreeSame,
    simd_and_reg: SimdThreeSame,
    simd_bic_reg: SimdThreeSame,
    simd_orn_reg: SimdThreeSame,
    simd_eor_reg: SimdThreeSame,
    simd_bsl: SimdThreeSame,
    simd_bit: SimdThreeSame,
    simd_bif: SimdThreeSame,
    simd_not: SimdTwoRegNoSize,
    simd_neg: SimdTwoReg,
    simd_abs: SimdTwoReg,
    simd_cnt: SimdTwoRegNoSize,
    simd_cls_vec: SimdTwoReg,
    simd_clz_vec: SimdTwoReg,
    simd_sshll: SimdShiftLong,
    simd_ushll: SimdShiftLong,
    simd_shrn: SimdShrn,
    simd_shift_imm: SimdShiftImm,
    simd_rev: SimdRev,
    simd_ins_elem: SimdInsElem,
    simd_dup_scalar: SimdDupScalar,
    simd_rbit: SimdTwoRegNoSize,
    simd_idx_mul: SimdIdxMul,
    simd_xtn: SimdTwoReg,
    crypto_aes: CryptoAes,
    crypto_pmull: CryptoPmull,
    crypto_sha: CryptoSha,
    simd_tbl: SimdTbl,
    simd_zip1: SimdThreeSame,
    simd_zip2: SimdThreeSame,
    simd_uzp1: SimdThreeSame,
    simd_uzp2: SimdThreeSame,
    simd_trn1: SimdThreeSame,
    simd_trn2: SimdThreeSame,

    vec_fp3: VecFp3,
    vec_fp2: VecFp2,
    vec_int3: VecInt3,
    vec_int2: VecInt2,
    vec_across: VecAcross,
    vec_widen: VecWiden,
    simd_ldst_multi: SimdLdstMulti,
    simd_ld1r: SimdLd1r,
    simd_ldst1_single: SimdLdSt1Single,

    cas: Cas,
    lse: Lse,
};

pub const SimdLdstMulti = struct { q: bool, is_load: bool, rt: u5, rn: u5, rm: u5, post: bool, reg_count: u3, deinterleave: bool, size: u2 };
pub const SimdLd1r = struct { q: bool, rt: u5, rn: u5, rm: u5, post: bool, size: u2 };
pub const SimdLdSt1Single = struct { load: bool, rt: u5, rn: u5, rm: u5, post: bool, esize: u8, index: u8 };

pub const VecAcrossOp = enum { addv, saddlv, uaddlv, smaxv, umaxv, sminv, uminv, fmaxv, fminv };
pub const VecAcross = struct { q: bool, size: u2, rd: u5, rn: u5, op: VecAcrossOp };

pub const VecWidenOp = enum { smull, umull, saddl, uaddl, ssubl, usubl, smlal, umlal, smlsl, umlsl, saddw, uaddw, ssubw, usubw };
// size: element size of the NARROW source operands (0=8b,1=16b,2=32b); result is 2x
pub const VecWiden = struct { q: bool, size: u2, rd: u5, rn: u5, rm: u5, op: VecWidenOp };

pub const VecFpOp = enum {
    add, sub, mul, div, max, min, maxnm, minnm, mla, mls,
    cmeq, cmge, cmgt, fabd, addp, maxp, minp, maxnmp, minnmp,
};
// sz: 0 = f32 lanes (.2s/.4s), 1 = f64 lanes (.2d)
pub const VecFp3 = struct { q: bool, sz: u1, rd: u5, rn: u5, rm: u5, op: VecFpOp };

pub const VecFpMiscOp = enum {
    neg, abs, sqrt, frintn, frintm, frintp, frintz, frinta, frinti, frintx,
    fcvtzs, fcvtzu, scvtf, ucvtf, fcmeq0, fcmge0, fcmgt0, fcmle0, fcmlt0,
};
pub const VecFp2 = struct { q: bool, sz: u1, rd: u5, rn: u5, op: VecFpMiscOp };

pub const VecIntOp = enum {
    add, sub, mul, cmeq, cmgt, cmge, cmhi, cmhs, cmtst,
    smax, smin, umax, umin, sqadd, uqadd, sqsub, uqsub,
    sshl, ushl, add_pairwise, smax_pairwise, umax_pairwise, smin_pairwise, umin_pairwise,
    @"and", @"or", xor, bic, orn,
};
// size: 0=8b,1=16b,2=32b,3=64b lanes
pub const VecInt3 = struct { q: bool, size: u2, rd: u5, rn: u5, rm: u5, op: VecIntOp };

pub const VecIntMiscOp = enum { neg, abs, cmeq0, cmgt0, cmge0, cmlt0, cmle0 };
pub const VecInt2 = struct { q: bool, size: u2, rd: u5, rn: u5, op: VecIntMiscOp };

pub const Cas = struct { size: u8, rs: u5, rt: u5, rn: u5 };
pub const LseOp = enum { add, clr, eor, set, smax, smin, umax, umin, swp };
pub const Lse = struct { size: u8, op: LseOp, rs: u5, rt: u5, rn: u5 };

// ── operand structs ──

pub const ArithImm = struct { sf: bool, rd: u5, rn: u5, imm: u32, set_flags: bool };
pub const ArithReg = struct {
    sf: bool,
    rd: u5,
    rn: u5,
    rm: u5,
    shift: ShiftKind,
    amount: u6,
    set_flags: bool,
};
pub const ShiftKind = enum(u2) { lsl = 0, lsr = 1, asr = 2, ror = 3 };

pub const LogicalImm = struct { sf: bool, rd: u5, rn: u5, imm: u64, set_flags: bool };
pub const LogicalReg = struct {
    sf: bool,
    rd: u5,
    rn: u5,
    rm: u5,
    shift: ShiftKind,
    amount: u6,
    invert_rm: bool,
    set_flags: bool,
};

pub const ShiftReg = struct { sf: bool, rd: u5, rn: u5, rm: u5 };
pub const ArithExtReg = struct {
    sf: bool,
    rd: u5,
    rn: u5,
    rm: u5,
    option: u3, // 000=UXTB, 001=UXTH, 010=UXTW, 011=UXTX, 100=SXTB, 101=SXTH, 110=SXTW, 111=SXTX
    amount: u3, // 0..4 shift after extend
    set_flags: bool,
};
pub const DivReg = struct { sf: bool, rd: u5, rn: u5, rm: u5 };
pub const MulAdd = struct { sf: bool, rd: u5, rn: u5, rm: u5, ra: u5 };
pub const MulHigh = struct { rd: u5, rn: u5, rm: u5 };

pub const MovWide = struct { sf: bool, rd: u5, imm16: u16, hw: u2 }; // shift = hw*16
pub const PcRel = struct { rd: u5, offset: i32 };
pub const BitfieldImm = struct { sf: bool, rd: u5, rn: u5, immr: u6, imms: u6 };
pub const Extr = struct { sf: bool, rd: u5, rn: u5, rm: u5, imms: u6 };

pub const CondSelect = struct { sf: bool, rd: u5, rn: u5, rm: u5, cond: u4 };
pub const CondCmpReg = struct { sf: bool, rn: u5, rm: u5, cond: u4, nzcv: u4, is_ccmn: bool };
pub const CondCmpImm = struct { sf: bool, rn: u5, imm5: u5, cond: u4, nzcv: u4, is_ccmn: bool };

pub const AdcSbc = struct { sf: bool, rd: u5, rn: u5, rm: u5, set_flags: bool };
pub const RevOp = struct { sf: bool, rd: u5, rn: u5, opc: u2 };
pub const ClzRbit = struct { sf: bool, rd: u5, rn: u5 };

pub const BrUncond = struct { offset: i32 }; // bytes from current pc
pub const BrReg = struct { rn: u5 };
pub const BrCond = struct { cond: u4, offset: i32 };
pub const BrCmpZ = struct { sf: bool, rt: u5, offset: i32 };
pub const BrTestBit = struct { rt: u5, bit: u6, offset: i32 };
pub const Imm16 = struct { imm: u16 };

pub const LoadStoreImm = struct {
    sf: bool, // 64-bit access
    rt: u5,
    rn: u5,
    offset: i32,
    mode: AddrMode,
};
pub const AddrMode = enum(u2) { offset_unsigned = 0, post_index = 1, offset_signed = 2, pre_index = 3 };
pub const LoadStoreReg = struct {
    sf: bool,
    rt: u5,
    rn: u5,
    rm: u5,
    option: u3, // extend kind
    shift: bool,
};
pub const LoadLit = struct { sf: bool, rt: u5, offset: i32, signed: bool = false };
pub const LoadLitFp = struct { rt: u5, offset: i32, bytes: u8 };
pub const LoadStorePair = struct {
    sf: bool,
    rt: u5,
    rt2: u5,
    rn: u5,
    offset: i32,
    mode: AddrMode,
};
pub const LoadStoreEx = struct { size: u8, rt: u5, rt2: u5, rn: u5, rs: u5 };

pub const SysReg = struct { rt: u5, op0: u2, op1: u3, crn: u4, crm: u4, op2: u3 };
pub const BarrierOp = struct { crm: u4, op2: u3 };

pub const Fcmp = struct { ftype: u2, rn: u5, rm: u5, with_zero: bool };
pub const Fcvt = struct { rd: u5, rn: u5, ftype: u2, dest_ftype: u2 };
pub const Fcsel = struct { rd: u5, rn: u5, rm: u5, ftype: u2, cond: u4 };
pub const FmovImm = struct { ftype: u2, rd: u5, bits64: u64 };
pub const FmovGprFp = struct { rd: u5, rn: u5, sf: bool };
pub const FmovTop = struct { rd: u5, rn: u5 };
pub const FpArith2 = struct { ftype: u2, rd: u5, rn: u5, rm: u5 };
pub const FpArith1 = struct { ftype: u2, rd: u5, rn: u5 };
pub const FpArith3 = struct { ftype: u2, rd: u5, rn: u5, rm: u5, ra: u5 };
pub const FpIntCvt = struct { sf: bool, rd: u5, rn: u5, ftype: u2 };
// fp <-> fixed-point convert (scvtf d0,w0,#2 etc): like the integer convert but the
// value is scaled by 2^fbits. reduces to an int<->fp convert plus a multiply by a
// power-of-two constant. musl float formatting uses scvtf with fbits.
pub const FpFixedCvt = struct { sf: bool, rd: u5, rn: u5, ftype: u2, fbits: u8, signed: bool, to_int: bool };
// scalar advanced-simd int<->fp convert (e.g. ucvtf d0,d0 / fcvtzs s0,s0). operates on
// the low lane only; the integer source/dest lives in the fp register, not a gpr.
pub const SimdScalarCvt = struct { rd: u5, rn: u5, dbl: bool, signed: bool, to_int: bool };
// scalar advanced-simd shift by immediate (only the 64-bit D-reg forms exist, e.g.
// shl d1,d1,#52 in musl scalbn). all reduce to a single i64 op on lane 0.
pub const ScalarShiftOp = enum { shl, sshr, ushr, ssra, usra, sli, sri };
pub const SimdScalarShift = struct { rd: u5, rn: u5, shift: u8, op: ScalarShiftOp };
pub const SimdMovi = struct { rd: u5, imm8: u8, cmode: u4, op: u1, q: bool };
pub const SimdDupGpr = struct { rd: u5, rn: u5, esize: u8, q: bool };
pub const SimdDupElem = struct { rd: u5, rn: u5, esize: u8, index: u5, q: bool };
pub const SimdElem = struct { rd: u5, rn: u5, esize: u8, index: u5, q: bool };
pub const SimdThreeSame = struct { q: bool, rd: u5, rn: u5, rm: u5, size: u2 };
pub const SimdTwoReg = struct { q: bool, rd: u5, rn: u5, size: u2 };
pub const SimdTwoRegNoSize = struct { q: bool, rd: u5, rn: u5 };
pub const IdxMulOp = enum { mul, mla, mls, mull, mlal, mlsl };
pub const SimdIdxMul = struct { q: bool, u: bool, rd: u5, rn: u5, rm: u5, size: u2, index: u8, op: IdxMulOp };
pub const CryptoAesOp = enum { aese, aesd, aesmc, aesimc };
pub const CryptoAes = struct { op: CryptoAesOp, rd: u5, rn: u5 };
pub const CryptoPmull = struct { part: bool, rd: u5, rn: u5, rm: u5 };
pub const CryptoShaOp = enum { sha1c, sha1p, sha1m, sha1h, sha1su0, sha1su1, sha256h, sha256h2, sha256su0, sha256su1, sha512h, sha512h2, sha512su0, sha512su1 };
pub const CryptoSha = struct { op: CryptoShaOp, rd: u5, rn: u5, rm: u5 };
pub const SimdShiftLong = struct { q: bool, rd: u5, rn: u5, esize: u8, shift: u8 };
// same-width vector shift by immediate (USHR/SSHR/SHL over .8b/.16b/.4h/.8h/.2s/.4s/.2d).
pub const ShiftImmOp = enum { ushr, sshr, shl, sli, sri };
pub const SimdShiftImm = struct { q: bool, rd: u5, rn: u5, esize: u8, shift: u8, op: ShiftImmOp };
// REV16/REV32/REV64 (vector): reverse the `size`-encoded elements within each `container`-byte group.
pub const SimdRev = struct { q: bool, rd: u5, rn: u5, size: u2, container: u8 };
// INS (element): Vd.T[dst] = Vn.T[src] (leaves other lanes of Vd).
pub const SimdInsElem = struct { rd: u5, rn: u5, esize: u8, dst_index: u5, src_index: u5 };
// DUP (element -> scalar): Dd = Vn.T[index] (writes lane 0, zeros the rest of Vd).
pub const SimdDupScalar = struct { rd: u5, rn: u5, esize: u8, index: u5 };
pub const SimdShrn = struct { q: bool, rd: u5, rn: u5, size: u2, shift: u8 };
pub const SimdExt = struct { q: bool, rd: u5, rn: u5, rm: u5, index: u4 };
pub const SimdTbl = struct { q: bool, rd: u5, rn: u5, rm: u5, len: u2, is_tbx: bool };
pub const SimdMem = struct { size: u8, rt: u5, rn: u5, offset: i32, mode: AddrMode };
pub const SimdMemPair = struct { size: u8, rt: u5, rt2: u5, rn: u5, offset: i32, mode: AddrMode };
pub const SimdMemReg = struct { size: u8, rt: u5, rn: u5, rm: u5, option: u3, shift: bool };

// ── decoder ──

pub fn decode(insn: u32) Op {
    const op0 = (insn >> 25) & 0b1111;
    return switch (op0) {
        0b0000 => Op{ .unknown = insn }, // reserved / sve etc.
        0b0001 => Op{ .unknown = insn },
        0b0010 => Op{ .unknown = insn }, // sve
        0b0011 => Op{ .unknown = insn },
        0b1000, 0b1001 => decodeDataImm(insn),
        0b1010, 0b1011 => decodeBranchOrSys(insn),
        0b0100, 0b0110, 0b1100, 0b1110 => decodeLoadStore(insn),
        0b0101, 0b1101 => decodeDataReg(insn),
        0b0111, 0b1111 => decodeFpSimd(insn),
        else => Op{ .unknown = insn },
    };
}

// ── data-processing immediate (op0 = 100x) ──

fn decodeDataImm(insn: u32) Op {
    const op0 = (insn >> 23) & 0b111;
    return switch (op0) {
        0b000, 0b001 => decodeAdr(insn),
        0b010, 0b011 => decodeArithImm(insn),
        0b100 => decodeLogicalImm(insn),
        0b101 => decodeMovWide(insn),
        0b110 => decodeBitfield(insn),
        0b111 => decodeExtract(insn),
        else => Op{ .unknown = insn },
    };
}

fn decodeAdr(insn: u32) Op {
    const op: u1 = @truncate(insn >> 31);
    const immlo: u32 = (insn >> 29) & 0b11;
    const immhi: u32 = (insn >> 5) & 0x7_FFFF;
    const rd: u5 = @truncate(insn & 0x1f);
    var offset: i32 = @bitCast((immhi << 2) | immlo);
    // sign-extend 21 bits
    if (offset & (1 << 20) != 0) offset |= @as(i32, @bitCast(@as(u32, 0xFFE0_0000)));
    if (op == 0) {
        return Op{ .adr = .{ .rd = rd, .offset = offset } };
    } else {
        return Op{ .adrp = .{ .rd = rd, .offset = offset * 4096 } };
    }
}

fn decodeArithImm(insn: u32) Op {
    const sf: bool = (insn >> 31) != 0;
    const op_bit: u1 = @truncate(insn >> 30);
    const s_bit: u1 = @truncate(insn >> 29);
    const sh: u1 = @truncate(insn >> 22);
    const imm12: u32 = (insn >> 10) & 0xfff;
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    const imm: u32 = if (sh == 1) imm12 << 12 else imm12;
    const set_flags = s_bit == 1;
    const arith = ArithImm{ .sf = sf, .rd = rd, .rn = rn, .imm = imm, .set_flags = set_flags };
    return if (op_bit == 0) Op{ .add_imm = arith } else Op{ .sub_imm = arith };
}

fn decodeLogicalImm(insn: u32) Op {
    const sf: bool = (insn >> 31) != 0;
    const opc: u2 = @truncate(insn >> 29);
    const n_bit: u1 = @truncate(insn >> 22);
    const immr: u6 = @truncate((insn >> 16) & 0x3f);
    const imms: u6 = @truncate((insn >> 10) & 0x3f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    const imm = decodeBitmaskImm(n_bit, immr, imms, sf) catch return Op{ .unknown = insn };
    const log = LogicalImm{ .sf = sf, .rd = rd, .rn = rn, .imm = imm, .set_flags = opc == 0b11 };
    return switch (opc) {
        0b00 => Op{ .and_imm = log },
        0b01 => Op{ .orr_imm = log },
        0b10 => Op{ .eor_imm = log },
        0b11 => Op{ .and_imm = log },
    };
}

/// expand the n/immr/imms encoding into the 32- or 64-bit bitmask value.
/// reference: armarm shared/functions/system/decodebitmasks.
fn decodeBitmaskImm(n: u1, immr: u6, imms: u6, sf: bool) !u64 {
    // top: N. below: NOT(imms). find highest set bit position.
    const combined: u32 = (@as(u32, n) << 6) | (~@as(u32, imms) & 0x3f);
    if (combined == 0) return error.InvalidBitmask;
    const len: u32 = 31 - @clz(combined);
    if (len < 1) return error.InvalidBitmask;

    const size: u32 = @as(u32, 1) << @intCast(len);
    const levels: u32 = size - 1;
    const s: u32 = @as(u32, imms) & levels;
    const r: u32 = @as(u32, immr) & levels;
    if (s == levels) return error.InvalidBitmask;

    const welem: u64 = if (s + 1 >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(s + 1)) - 1;
    const mask: u64 = if (size >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(size)) - 1;
    const rotated: u64 = if (r == 0) welem else ((welem >> @intCast(r)) | (welem << @intCast(size - r))) & mask;

    var result: u64 = 0;
    var i: u32 = 0;
    while (i < 64) : (i += size) {
        result |= rotated << @intCast(i);
    }
    if (!sf) result &= 0xffff_ffff;
    return result;
}

fn decodeMovWide(insn: u32) Op {
    const sf: bool = (insn >> 31) != 0;
    const opc: u2 = @truncate(insn >> 29);
    const hw: u2 = @truncate((insn >> 21) & 0x3);
    const imm16: u16 = @truncate((insn >> 5) & 0xffff);
    const rd: u5 = @truncate(insn & 0x1f);
    const mw = MovWide{ .sf = sf, .rd = rd, .imm16 = imm16, .hw = hw };
    return switch (opc) {
        0b00 => Op{ .movn = mw },
        0b10 => Op{ .movz = mw },
        0b11 => Op{ .movk = mw },
        else => Op{ .unknown = insn },
    };
}

fn decodeBitfield(insn: u32) Op {
    const sf: bool = (insn >> 31) != 0;
    const opc: u2 = @truncate(insn >> 29);
    const immr: u6 = @truncate((insn >> 16) & 0x3f);
    const imms: u6 = @truncate((insn >> 10) & 0x3f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    const bf = BitfieldImm{ .sf = sf, .rd = rd, .rn = rn, .immr = immr, .imms = imms };
    return switch (opc) {
        0b00 => Op{ .sbfm = bf },
        0b01 => Op{ .bfm = bf },
        0b10 => Op{ .ubfm = bf },
        else => Op{ .unknown = insn },
    };
}

fn decodeExtract(insn: u32) Op {
    const sf: bool = (insn >> 31) != 0;
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const imms: u6 = @truncate((insn >> 10) & 0x3f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    return Op{ .extr = .{ .sf = sf, .rd = rd, .rn = rn, .rm = rm, .imms = imms } };
}

// ── branches & system ──

fn decodeBranchOrSys(insn: u32) Op {
    // unconditional branch (immediate): bits 30-26 = 00101
    if (((insn >> 26) & 0x1f) == 0b00101) {
        const op_bit: u1 = @truncate(insn >> 31);
        const imm26: u32 = insn & 0x3ff_ffff;
        const offset = signExtend(imm26, 26) << 2;
        return if (op_bit == 0)
            Op{ .b = .{ .offset = offset } }
        else
            Op{ .bl = .{ .offset = offset } };
    }

    // conditional branch (B.cond): bits 31-25 = 0101010, bit 4 = 0
    if (((insn >> 25) & 0x7f) == 0b0101010 and ((insn >> 4) & 1) == 0) {
        const imm19: u32 = (insn >> 5) & 0x7_ffff;
        const cond: u4 = @truncate(insn & 0xf);
        return Op{ .b_cond = .{ .cond = cond, .offset = signExtend(imm19, 19) << 2 } };
    }

    // compare-and-branch (CBZ/CBNZ): bits 30-25 = 011010
    if (((insn >> 25) & 0x3f) == 0b011010) {
        const sf: bool = (insn >> 31) != 0;
        const op_bit: u1 = @truncate(insn >> 24);
        const imm19: u32 = (insn >> 5) & 0x7_ffff;
        const rt: u5 = @truncate(insn & 0x1f);
        const data = BrCmpZ{ .sf = sf, .rt = rt, .offset = signExtend(imm19, 19) << 2 };
        return if (op_bit == 0) Op{ .cbz = data } else Op{ .cbnz = data };
    }

    // test-and-branch (TBZ/TBNZ): bits 30-25 = 011011
    if (((insn >> 25) & 0x3f) == 0b011011) {
        const b5: u1 = @truncate(insn >> 31);
        const op_bit: u1 = @truncate(insn >> 24);
        const b40: u5 = @truncate((insn >> 19) & 0x1f);
        const imm14: u32 = (insn >> 5) & 0x3fff;
        const rt: u5 = @truncate(insn & 0x1f);
        const bit_pos: u6 = (@as(u6, b5) << 5) | @as(u6, b40);
        const data = BrTestBit{ .rt = rt, .bit = bit_pos, .offset = signExtend(imm14, 14) << 2 };
        return if (op_bit == 0) Op{ .tbz = data } else Op{ .tbnz = data };
    }

    // unconditional branch (register): bits 31-25 = 1101011
    if (((insn >> 25) & 0x7f) == 0b1101011) {
        const opc: u4 = @truncate((insn >> 21) & 0xf);
        const rn: u5 = @truncate((insn >> 5) & 0x1f);
        return switch (opc) {
            0b0000 => Op{ .br = .{ .rn = rn } },
            0b0001 => Op{ .blr = .{ .rn = rn } },
            0b0010 => Op{ .ret = .{ .rn = rn } },
            // pac branch variants with explicit modifier (braa/brab, blraa/blrab).
            // auth is identity here, so the target is just rn, as with plain br/blr.
            // the braaz/blraaz/retaa/retab zero-modifier forms share opc 0000/0001/0010
            // and already fall through to br/blr/ret above.
            0b1000 => Op{ .br = .{ .rn = rn } },
            0b1001 => Op{ .blr = .{ .rn = rn } },
            else => Op{ .unknown = insn },
        };
    }

    // exception generation: bits 31-24 = 11010100
    if (((insn >> 24) & 0xff) == 0b11010100) {
        const opc: u3 = @truncate((insn >> 21) & 0x7);
        const ll: u5 = @truncate(insn & 0x1f);
        const imm16: u16 = @truncate((insn >> 5) & 0xffff);
        if (opc == 0b000 and ll == 0b00001) return Op{ .svc = .{ .imm = imm16 } };
        if (opc == 0b001 and ll == 0b00000) return Op{ .brk = .{ .imm = imm16 } };
        return Op{ .unknown = insn };
    }

    // system instructions: bits 31-22 = 1101010100
    if (((insn >> 22) & 0x3ff) == 0b1101010100) {
        return decodeSystem(insn);
    }

    return Op{ .unknown = insn };
}

fn decodeSystem(insn: u32) Op {
    const l: u1 = @truncate(insn >> 21);
    // bit 20 is hardcoded to 1 for sysreg access; bit 19 is o0 (selects op0=2 vs 3)
    const o0: u1 = @truncate(insn >> 19);
    const op1: u3 = @truncate((insn >> 16) & 0x7);
    const crn: u4 = @truncate((insn >> 12) & 0xf);
    const crm: u4 = @truncate((insn >> 8) & 0xf);
    const op2: u3 = @truncate((insn >> 5) & 0x7);
    const rt: u5 = @truncate(insn & 0x1f);

    // NOP / HINT: rt=11111, crn=0010, l=0, op1=011, o0 from CRm distinguishes things
    // hints live in the "MSR (immediate)" / hint group with bit 20=0. for simplicity
    // treat the common form: op1=011, crn=0010
    if (rt == 0x1f and crn == 0b0010 and l == 0 and op1 == 0b011) {
        if (crm == 0 and op2 == 0) return Op{ .nop = {} };
        const hint_imm: u8 = (@as(u8, crm) << 3) | @as(u8, op2);
        return Op{ .hint = hint_imm };
    }

    if (rt == 0x1f and crn == 0b0011 and l == 0) {
        return Op{ .barrier = .{ .crm = crm, .op2 = op2 } };
    }

    const sr = SysReg{
        .rt = rt,
        .op0 = @intCast(2 + @as(u8, o0)), // 2 for S2_x_x_x_x, 3 for S3_x_x_x_x
        .op1 = op1,
        .crn = crn,
        .crm = crm,
        .op2 = op2,
    };
    return if (l == 1) Op{ .mrs = sr } else Op{ .msr = sr };
}

inline fn signExtend(value: u32, bits: u5) i32 {
    const shift: u5 = @intCast(@as(u32, 32) - @as(u32, bits));
    return @bitCast(@as(u32, @bitCast(@as(i32, @bitCast(value << shift)) >> shift)));
}

// ── load/store ──
// pattern-match by bit ranges, mirroring the ts decoder.

fn decodeLoadStore(insn: u32) Op {
    const bits_29_27: u3 = @truncate((insn >> 27) & 0x7);
    const bits_29_28: u2 = @truncate((insn >> 28) & 0x3);
    const v: u1 = @truncate(insn >> 26);

    if (bits_29_27 == 0b011 and v == 0 and (insn >> 24) & 0x3 == 0) return decodeLoadLit(insn);
    if (bits_29_27 == 0b011 and v == 1 and (insn >> 24) & 0x3 == 0) return decodeLoadLitFp(insn);
    if (bits_29_28 == 0b00 and v == 0) return decodeLoadStoreEx(insn);
    if (bits_29_27 == 0b101 and v == 0) return decodeLoadStorePair(insn);

    if (bits_29_27 == 0b111 and v == 0) {
        const bit24: u1 = @truncate(insn >> 24);
        const bit21: u1 = @truncate(insn >> 21);
        if (bit24 == 1) return decodeLoadStoreImmUnsigned(insn);
        if (bit24 == 0 and bit21 == 0) return decodeLoadStoreImmIxd(insn);
        if (bit24 == 0 and bit21 == 1) {
            // bits 11-10 split: 00 = atomic memory op (lse), 10 = register offset
            const lo2: u2 = @truncate((insn >> 10) & 0x3);
            if (lo2 == 0b10) return decodeLoadStoreReg(insn);
            if (lo2 == 0b00) return decodeLseAtomic(insn);
            return Op{ .unknown = insn };
        }
    }

    // advanced simd load/store multiple structures: bits 29-23 = 0011000 (no
    // offset) / 0011001 (post-index)
    const b29_23: u7 = @truncate((insn >> 23) & 0x7f);
    if (b29_23 == 0b0011000 or b29_23 == 0b0011001) return decodeSimdLoadMulti(insn);
    // single structure (incl. LD1R replicate): 0011010 / 0011011
    if (b29_23 == 0b0011010 or b29_23 == 0b0011011) return decodeSimdLoadSingle(insn);

    // simd/fp load/store: V=1
    if (v == 1) {
        if (bits_29_27 == 0b101) return decodeSimdLoadStorePair(insn);
        if (bits_29_27 == 0b111) {
            const bit24: u1 = @truncate(insn >> 24);
            const bit21: u1 = @truncate(insn >> 21);
            if (bit24 == 1) return decodeSimdLoadStoreImmUnsigned(insn);
            if (bit24 == 0 and bit21 == 0) return decodeSimdLoadStoreImmIxd(insn);
            if (bit24 == 0 and bit21 == 1) return decodeSimdLoadStoreReg(insn);
        }
    }

    return Op{ .unknown = insn };
}

/// advanced simd load/store multiple structures (LD1/ST1 .. LD4/ST4)
fn decodeSimdLoadMulti(insn: u32) Op {
    const q: bool = ((insn >> 30) & 1) == 1;
    const post: bool = ((insn >> 23) & 1) == 1;
    const l: u1 = @truncate(insn >> 22);
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const opcode: u4 = @truncate((insn >> 12) & 0xf);
    const size: u2 = @truncate((insn >> 10) & 0x3);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rt: u5 = @truncate(insn & 0x1f);

    // opcode → (reg count, de-interleaving)
    const info: ?struct { n: u3, deint: bool } = switch (opcode) {
        0b0000 => .{ .n = 4, .deint = true }, // LD4/ST4
        0b0010 => .{ .n = 4, .deint = false }, // LD1/ST1 x4
        0b0100 => .{ .n = 3, .deint = true }, // LD3/ST3
        0b0110 => .{ .n = 3, .deint = false }, // LD1/ST1 x3
        0b0111 => .{ .n = 1, .deint = false }, // LD1/ST1 x1
        0b1000 => .{ .n = 2, .deint = true }, // LD2/ST2
        0b1010 => .{ .n = 2, .deint = false }, // LD1/ST1 x2
        else => null,
    };
    if (info) |i| return Op{ .simd_ldst_multi = .{
        .q = q,
        .is_load = l == 1,
        .rt = rt,
        .rn = rn,
        .rm = rm,
        .post = post,
        .reg_count = i.n,
        .deinterleave = i.deint,
        .size = size,
    } };
    return Op{ .unknown = insn };
}

/// advanced simd load/store single structure. currently only LD1R (replicate)
fn decodeSimdLoadSingle(insn: u32) Op {
    const q: bool = ((insn >> 30) & 1) == 1;
    const post: bool = ((insn >> 23) & 1) == 1;
    const l: u1 = @truncate(insn >> 22);
    const r: u1 = @truncate(insn >> 21);
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const opcode: u3 = @truncate((insn >> 13) & 0x7);
    const size: u2 = @truncate((insn >> 10) & 0x3);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rt: u5 = @truncate(insn & 0x1f);

    // LD1R: L=1, R=0, opcode=110 — replicate one element across all lanes
    if (l == 1 and r == 0 and opcode == 0b110) {
        return Op{ .simd_ld1r = .{ .q = q, .rt = rt, .rn = rn, .rm = rm, .post = post, .size = size } };
    }
    // single-lane LD1/ST1 (1 register, R=0): element size + lane index from opcode/S/size/Q.
    if (r == 0 and (opcode == 0b000 or opcode == 0b010 or opcode == 0b100)) {
        const s: u1 = @truncate((insn >> 12) & 1);
        const qn: u8 = if (q) 1 else 0;
        var esize: u8 = 0;
        var index: u8 = 0;
        switch (opcode) {
            0b000 => { esize = 1; index = (qn << 3) | (@as(u8, s) << 2) | @as(u8, size); },
            0b010 => if ((size & 1) == 0) { esize = 2; index = (qn << 2) | (@as(u8, s) << 1) | @as(u8, size >> 1); },
            0b100 => {
                if (size == 0b00) { esize = 4; index = (qn << 1) | @as(u8, s); } else if (size == 0b01 and s == 0) { esize = 8; index = qn; }
            },
            else => {},
        }
        if (esize != 0) return Op{ .simd_ldst1_single = .{ .load = l == 1, .rt = rt, .rn = rn, .rm = rm, .post = post, .esize = esize, .index = index } };
    }
    return Op{ .unknown = insn };
}

/// access size in bytes from (size, opc[1]). returns 0 if encoding is reserved
fn simdRegBytes(size: u2, opc_hi: u1) u8 {
    if (opc_hi == 0) return @as(u8, 1) << size; // B/H/S/D
    if (opc_hi == 1 and size == 0) return 16; // Q
    return 0;
}

fn decodeSimdLoadStoreImmUnsigned(insn: u32) Op {
    const size: u2 = @truncate(insn >> 30);
    const opc: u2 = @truncate((insn >> 22) & 0x3);
    const imm12: u32 = (insn >> 10) & 0xfff;
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rt: u5 = @truncate(insn & 0x1f);
    const opc_hi: u1 = @truncate(opc >> 1);
    const reg_bytes = simdRegBytes(size, opc_hi);
    if (reg_bytes == 0) return Op{ .unknown = insn };
    const is_load = (opc & 1) == 1;
    const offset: i32 = @bitCast(imm12 * @as(u32, reg_bytes));
    const sm = SimdMem{ .size = reg_bytes, .rt = rt, .rn = rn, .offset = offset, .mode = .offset_unsigned };
    return if (is_load) Op{ .simd_load = sm } else Op{ .simd_store = sm };
}

fn decodeSimdLoadStoreImmIxd(insn: u32) Op {
    const size: u2 = @truncate(insn >> 30);
    const opc: u2 = @truncate((insn >> 22) & 0x3);
    const imm9_raw: u32 = (insn >> 12) & 0x1ff;
    const mode_bits: u2 = @truncate((insn >> 10) & 0x3);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rt: u5 = @truncate(insn & 0x1f);

    const opc_hi: u1 = @truncate(opc >> 1);
    const reg_bytes = simdRegBytes(size, opc_hi);
    if (reg_bytes == 0) return Op{ .unknown = insn };

    const is_load = (opc & 1) == 1;
    const offset: i32 = signExtend(imm9_raw, 9);
    const mode: AddrMode = switch (mode_bits) {
        0b00 => .offset_signed,
        0b01 => .post_index,
        0b10 => .offset_signed,
        0b11 => .pre_index,
    };

    const sm = SimdMem{ .size = reg_bytes, .rt = rt, .rn = rn, .offset = offset, .mode = mode };
    return if (is_load) Op{ .simd_load = sm } else Op{ .simd_store = sm };
}

fn decodeSimdLoadStoreReg(insn: u32) Op {
    const size: u2 = @truncate(insn >> 30);
    const opc: u2 = @truncate((insn >> 22) & 0x3);
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const option: u3 = @truncate((insn >> 13) & 0x7);
    const s_bit: u1 = @truncate(insn >> 12);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rt: u5 = @truncate(insn & 0x1f);

    const opc_hi: u1 = @truncate(opc >> 1);
    const reg_bytes = simdRegBytes(size, opc_hi);
    if (reg_bytes == 0) return Op{ .unknown = insn };

    const is_load = (opc & 1) == 1;
    const sm = SimdMemReg{ .size = reg_bytes, .rt = rt, .rn = rn, .rm = rm, .option = option, .shift = s_bit == 1 };
    return if (is_load) Op{ .simd_load_reg = sm } else Op{ .simd_store_reg = sm };
}

fn decodeSimdLoadStorePair(insn: u32) Op {
    // opc field for simd pair: 00=S(4), 01=D(8), 10=Q(16). bit 22 = L
    const opc: u2 = @truncate(insn >> 30);
    const type_bits: u3 = @truncate((insn >> 23) & 0x7); // 25-23: 100=post, 101=offset, 111=pre
    const l: u1 = @truncate(insn >> 22);
    const imm7_raw: u32 = (insn >> 15) & 0x7f;
    const rt2: u5 = @truncate((insn >> 10) & 0x1f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rt: u5 = @truncate(insn & 0x1f);

    const reg_bytes: u8 = switch (opc) {
        0b00 => 4, // Sd pair
        0b01 => 8, // Dd pair
        0b10 => 16, // Qd pair
        else => return Op{ .unknown = insn },
    };

    const mode: AddrMode = switch (type_bits) {
        // 000 = ldnp/stnp (non-temporal): identical to signed-offset ldp/stp,
        // the hint has no architectural effect
        0b000 => .offset_signed,
        0b001 => .post_index,
        0b010 => .offset_signed,
        0b011 => .pre_index,
        else => return Op{ .unknown = insn },
    };

    const imm7_signed: i32 = signExtend(imm7_raw, 7);
    const offset: i32 = imm7_signed * @as(i32, reg_bytes);

    const sp = SimdMemPair{ .size = reg_bytes, .rt = rt, .rt2 = rt2, .rn = rn, .offset = offset, .mode = mode };
    return if (l == 1) Op{ .simd_load_pair = sp } else Op{ .simd_store_pair = sp };
}

fn decodeLoadStoreImmUnsigned(insn: u32) Op {
    const size: u2 = @truncate(insn >> 30);
    const opc: u2 = @truncate((insn >> 22) & 0x3);
    const imm12: u32 = (insn >> 10) & 0xfff;
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rt: u5 = @truncate(insn & 0x1f);
    const scale: u32 = @as(u32, 1) << @intCast(@as(u32, size));
    const offset: i32 = @bitCast(imm12 * scale);
    return classifyLoadStore(size, opc, rt, rn, offset, .offset_unsigned);
}

fn decodeLoadStoreEx(insn: u32) Op {
    // size|001000|o2|L|o1|Rs|o0|Rt2|Rn|Rt
    const size_bits: u2 = @truncate(insn >> 30);
    const byte_size: u8 = switch (size_bits) {
        0 => 1,
        1 => 2,
        2 => 4,
        3 => 8,
    };
    const o2: u1 = @truncate(insn >> 23);
    const l: u1 = @truncate(insn >> 22);
    const o1: u1 = @truncate(insn >> 21);
    const rs: u5 = @truncate((insn >> 16) & 0x1f);
    const rt2: u5 = @truncate((insn >> 10) & 0x1f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rt: u5 = @truncate(insn & 0x1f);

    if (o2 == 0 and o1 == 0) {
        // ldxr/ldaxr / stxr/stlxr — acquire/release ordering is a no-op here
        const data = LoadStoreEx{ .size = byte_size, .rt = rt, .rt2 = rt2, .rn = rn, .rs = rs };
        return if (l == 1) Op{ .ldxr = data } else Op{ .stxr = data };
    }
    if (o2 == 1 and o1 == 0) {
        // ldar/ldlar / stlr/stllr — plain load/store under sequential execution
        const mem = LoadStoreImm{ .sf = size_bits == 0b11, .rt = rt, .rn = rn, .offset = 0, .mode = .offset_unsigned };
        if (l == 1) {
            return switch (size_bits) {
                0b00 => Op{ .ldrb_imm = mem },
                0b01 => Op{ .ldrh_imm = mem },
                else => Op{ .ldr_imm = mem },
            };
        }
        return switch (size_bits) {
            0b00 => Op{ .strb_imm = mem },
            0b01 => Op{ .strh_imm = mem },
            else => Op{ .str_imm = mem },
        };
    }
    if (o2 == 1 and o1 == 1 and rt2 == 0b11111) {
        // cas/casa/casl/casal — l/o0 are ordering-only
        return Op{ .cas = .{ .size = byte_size, .rs = rs, .rt = rt, .rn = rn } };
    }
    // ldxp/stxp (pair-exclusive) not implemented
    return Op{ .unknown = insn };
}

// LDR (literal, SIMD&FP): PC-relative load of 4/8/16 bytes into a V register.
// software crypto (vpaes/gcm) loads its constant tables this way, so it is on the apk path.
fn decodeLoadLitFp(insn: u32) Op {
    const opc: u2 = @truncate(insn >> 30);
    if (opc == 0b11) return Op{ .unknown = insn }; // reserved
    const imm19: u32 = (insn >> 5) & 0x7ffff;
    var offset: i32 = @bitCast(imm19 << 2);
    if (offset & (1 << 20) != 0) offset |= @as(i32, @bitCast(@as(u32, 0xFFE0_0000)));
    const bytes: u8 = switch (opc) { 0b00 => 4, 0b01 => 8, else => 16 };
    return Op{ .ldr_lit_fp = .{ .rt = @truncate(insn & 0x1f), .offset = offset, .bytes = bytes } };
}

fn decodeLoadLit(insn: u32) Op {
    const opc: u2 = @truncate(insn >> 30);
    if (opc == 0b11) return Op{ .nop = {} }; // PRFM literal: prefetch hint, no architectural effect
    const imm19: u32 = (insn >> 5) & 0x7ffff;
    var offset: i32 = @bitCast(imm19 << 2);
    if (offset & (1 << 20) != 0) offset |= @as(i32, @bitCast(@as(u32, 0xFFE0_0000)));
    const rt: u5 = @truncate(insn & 0x1f);
    // opc: 00 = 32-bit ldr (zero-extend), 01 = 64-bit ldr, 10 = ldrsw (sign-extend)
    return Op{ .ldr_lit = .{
        .sf = opc == 0b01,
        .rt = rt,
        .offset = offset,
        .signed = opc == 0b10,
    } };
}

fn decodeLoadStorePair(insn: u32) Op {
    const opc: u2 = @truncate(insn >> 30);
    const l: u1 = @truncate(insn >> 22);
    const imm7: u32 = (insn >> 15) & 0x7f;
    const rt2: u5 = @truncate((insn >> 10) & 0x1f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rt: u5 = @truncate(insn & 0x1f);
    const sf = opc == 0b10;
    var offset: i32 = @bitCast(imm7);
    if (offset & 0x40 != 0) offset |= @as(i32, @bitCast(@as(u32, 0xFFFF_FF80)));
    offset *= if (sf) 8 else 4;

    const mode_bits: u2 = @truncate((insn >> 23) & 0x3);
    const mode: AddrMode = switch (mode_bits) {
        0b01 => .post_index,
        0b10 => .offset_unsigned,
        0b11 => .pre_index,
        else => .offset_unsigned,
    };

    const data = LoadStorePair{ .sf = sf, .rt = rt, .rt2 = rt2, .rn = rn, .offset = offset, .mode = mode };
    return if (l == 1) Op{ .ldp = data } else Op{ .stp = data };
}

fn decodeLoadStoreImmIxd(insn: u32) Op {
    const size: u2 = @truncate(insn >> 30);
    const opc: u2 = @truncate((insn >> 22) & 0x3);
    const imm9: u32 = (insn >> 12) & 0x1ff;
    const mode_disc: u2 = @truncate((insn >> 10) & 0x3);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rt: u5 = @truncate(insn & 0x1f);
    var offset: i32 = @bitCast(imm9);
    if (offset & 0x100 != 0) offset |= @as(i32, @bitCast(@as(u32, 0xFFFF_FE00)));
    const mode: AddrMode = switch (mode_disc) {
        0b00 => .offset_signed, // unscaled imm (LDUR/STUR)
        0b01 => .post_index,
        0b10 => .offset_signed, // LDTR/STTR unprivileged
        0b11 => .pre_index,
    };
    return classifyLoadStore(size, opc, rt, rn, offset, mode);
}

/// atomic memory operations (armv8.1 lse):
/// size|111000|A|R|1|Rs|o3|opc|00|Rn|Rt
fn decodeLseAtomic(insn: u32) Op {
    const size_bits: u2 = @truncate(insn >> 30);
    const byte_size: u8 = @as(u8, 1) << size_bits;
    const rs: u5 = @truncate((insn >> 16) & 0x1f);
    const o3: u1 = @truncate(insn >> 15);
    const opc: u3 = @truncate((insn >> 12) & 0x7);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rt: u5 = @truncate(insn & 0x1f);

    if (o3 == 0) {
        const op: LseOp = switch (opc) {
            0b000 => .add,
            0b001 => .clr,
            0b010 => .eor,
            0b011 => .set,
            0b100 => .smax,
            0b101 => .smin,
            0b110 => .umax,
            0b111 => .umin,
        };
        return Op{ .lse = .{ .size = byte_size, .op = op, .rs = rs, .rt = rt, .rn = rn } };
    }
    if (opc == 0b000) {
        return Op{ .lse = .{ .size = byte_size, .op = .swp, .rs = rs, .rt = rt, .rn = rn } };
    }
    // ldaprb/ldaprh/ldapr (o3=1, opc=100, rs=11111): plain load-acquire
    if (opc == 0b100 and rs == 0b11111) {
        const mem = LoadStoreImm{ .sf = size_bits == 0b11, .rt = rt, .rn = rn, .offset = 0, .mode = .offset_unsigned };
        return switch (size_bits) {
            0b00 => Op{ .ldrb_imm = mem },
            0b01 => Op{ .ldrh_imm = mem },
            else => Op{ .ldr_imm = mem },
        };
    }
    return Op{ .unknown = insn };
}

fn decodeLoadStoreReg(insn: u32) Op {
    const size: u2 = @truncate(insn >> 30);
    const opc: u2 = @truncate(insn >> 22);
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const option: u3 = @truncate((insn >> 13) & 0x7);
    const s_bit: u1 = @truncate(insn >> 12);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rt: u5 = @truncate(insn & 0x1f);

    // for signed loads opc=10 -> X destination, opc=11 -> W destination
    const signed_x_dest: bool = (opc == 0b10);
    var ls = LoadStoreReg{
        .sf = size == 0b11,
        .rt = rt,
        .rn = rn,
        .rm = rm,
        .option = option,
        .shift = s_bit == 1,
    };

    return switch (size) {
        0b00 => blk: {
            if (opc == 0b00) break :blk Op{ .strb_reg = ls };
            if (opc == 0b01) break :blk Op{ .ldrb_reg = ls };
            ls.sf = signed_x_dest;
            break :blk Op{ .ldrsb_reg = ls };
        },
        0b01 => blk: {
            if (opc == 0b00) break :blk Op{ .strh_reg = ls };
            if (opc == 0b01) break :blk Op{ .ldrh_reg = ls };
            ls.sf = signed_x_dest;
            break :blk Op{ .ldrsh_reg = ls };
        },
        0b10 => blk: {
            if (opc == 0b00) break :blk Op{ .str_reg = ls };
            if (opc == 0b01) break :blk Op{ .ldr_reg = ls };
            if (opc == 0b10) {
                ls.sf = true; // ldrsw always X destination
                break :blk Op{ .ldrsw_reg = ls };
            }
            break :blk Op{ .unknown = insn };
        },
        0b11 => blk: {
            if (opc == 0b00) break :blk Op{ .str_reg = ls };
            if (opc == 0b01) break :blk Op{ .ldr_reg = ls };
            if (opc == 0b10) break :blk Op{ .nop = {} }; // PRFM (register): prefetch hint
            break :blk Op{ .unknown = insn };
        },
    };
}

fn classifyLoadStore(size: u2, opc: u2, rt: u5, rn: u5, offset: i32, mode: AddrMode) Op {
    var ls = LoadStoreImm{ .sf = size == 0b11, .rt = rt, .rn = rn, .offset = offset, .mode = mode };
    // for signed loads opc=10 -> X destination (sf=true), opc=11 -> W destination (sf=false)
    const signed_x_dest: bool = (opc == 0b10);
    return switch (size) {
        0b00 => switch (opc) {
            0b00 => Op{ .strb_imm = ls },
            0b01 => Op{ .ldrb_imm = ls },
            0b10, 0b11 => blk: {
                ls.sf = signed_x_dest;
                break :blk Op{ .ldrsb_imm = ls };
            },
        },
        0b01 => switch (opc) {
            0b00 => Op{ .strh_imm = ls },
            0b01 => Op{ .ldrh_imm = ls },
            0b10, 0b11 => blk: {
                ls.sf = signed_x_dest;
                break :blk Op{ .ldrsh_imm = ls };
            },
        },
        0b10 => switch (opc) {
            0b00 => Op{ .str_imm = ls },
            0b01 => Op{ .ldr_imm = ls },
            0b10 => blk: {
                ls.sf = true;
                break :blk Op{ .ldrsw_imm = ls };
            },
            0b11 => Op{ .unknown = 0 },
        },
        0b11 => switch (opc) {
            0b00 => Op{ .str_imm = ls },
            0b01 => Op{ .ldr_imm = ls },
            0b10 => Op{ .nop = {} }, // PRFM: prefetch hint, no architectural effect
            0b11 => Op{ .unknown = 0 },
        },
    };
}

// ── data-processing register (op0 = x101) ──

fn decodeDataReg(insn: u32) Op {
    const op0: u1 = @truncate(insn >> 30);
    const op1: u1 = @truncate(insn >> 28);
    const op2: u4 = @truncate((insn >> 21) & 0xf);

    if (op1 == 0) {
        // bit 28 = op1 = 0 covers logical (shifted), arith (shifted), and arith (extended register).
        // op2 bit 3 (= insn bit 24) selects logical vs arith. op2 bit 0 (= insn bit 21) selects
        // shifted (0) vs extended (1) for arith. logical doesn't have an extended form.
        if (op2 & 0b1000 == 0) return decodeLogicalShifted(insn);
        if (op2 & 1 == 1) return decodeArithExtended(insn);
        return decodeArithShifted(insn);
    }

    // op1 == 1: data-processing 1/2/3 source, conditional select, etc.
    _ = op0;
    return switch (op2) {
        0b0000 => decodeAdcSbc(insn),
        0b0010 => decodeCondCmp(insn),
        0b0100 => decodeCondSelect(insn),
        0b0110 => decodeDataProc12(insn),
        0b1000, 0b1001, 0b1010, 0b1011, 0b1100, 0b1101, 0b1110, 0b1111 => decodeMul(insn),
        else => Op{ .unknown = insn },
    };
}

fn decodeLogicalShifted(insn: u32) Op {
    const sf: bool = (insn >> 31) != 0;
    const opc: u2 = @truncate(insn >> 29);
    const shift: u2 = @truncate(insn >> 22);
    const n: u1 = @truncate(insn >> 21);
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const imm6: u6 = @truncate((insn >> 10) & 0x3f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    const lr = LogicalReg{
        .sf = sf,
        .rd = rd,
        .rn = rn,
        .rm = rm,
        .shift = @enumFromInt(shift),
        .amount = imm6,
        .invert_rm = n == 1,
        .set_flags = opc == 0b11,
    };
    return switch (opc) {
        0b00 => if (n == 0) Op{ .and_reg = lr } else Op{ .bic_reg = lr },
        0b01 => if (n == 0) Op{ .orr_reg = lr } else Op{ .orn_reg = lr },
        0b10 => if (n == 0) Op{ .eor_reg = lr } else Op{ .eon_reg = lr },
        0b11 => if (n == 0) Op{ .and_reg = lr } else Op{ .bic_reg = lr },
    };
}

fn decodeArithExtended(insn: u32) Op {
    const sf: bool = (insn >> 31) != 0;
    const op_bit: u1 = @truncate(insn >> 30);
    const s_bit: u1 = @truncate(insn >> 29);
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const option: u3 = @truncate((insn >> 13) & 0x7);
    const imm3: u3 = @truncate((insn >> 10) & 0x7);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    if (imm3 > 4) return Op{ .unknown = insn };
    const ae = ArithExtReg{
        .sf = sf,
        .rd = rd,
        .rn = rn,
        .rm = rm,
        .option = option,
        .amount = imm3,
        .set_flags = s_bit == 1,
    };
    return if (op_bit == 0) Op{ .add_ext_reg = ae } else Op{ .sub_ext_reg = ae };
}

fn decodeArithShifted(insn: u32) Op {
    const sf: bool = (insn >> 31) != 0;
    const op_bit: u1 = @truncate(insn >> 30);
    const s_bit: u1 = @truncate(insn >> 29);
    const shift: u2 = @truncate(insn >> 22);
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const imm6: u6 = @truncate((insn >> 10) & 0x3f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    const ar = ArithReg{
        .sf = sf,
        .rd = rd,
        .rn = rn,
        .rm = rm,
        .shift = @enumFromInt(shift),
        .amount = imm6,
        .set_flags = s_bit == 1,
    };
    return if (op_bit == 0) Op{ .add_reg = ar } else Op{ .sub_reg = ar };
}

fn decodeAdcSbc(insn: u32) Op {
    const sf: bool = (insn >> 31) != 0;
    const op_bit: u1 = @truncate(insn >> 30);
    const s_bit: u1 = @truncate(insn >> 29);
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    const a = AdcSbc{ .sf = sf, .rd = rd, .rn = rn, .rm = rm, .set_flags = s_bit == 1 };
    return if (op_bit == 0) Op{ .adc = a } else Op{ .sbc = a };
}

fn decodeCondCmp(insn: u32) Op {
    const sf: bool = (insn >> 31) != 0;
    const op_bit: u1 = @truncate(insn >> 30);
    // bit 11 selects the operand form: 1 = immediate (imm5), 0 = register (rm)
    const is_imm: u1 = @truncate(insn >> 11);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const nzcv: u4 = @truncate(insn & 0xf);
    const cond: u4 = @truncate((insn >> 12) & 0xf);

    if (is_imm == 1) {
        const imm5: u5 = @truncate((insn >> 16) & 0x1f);
        return Op{ .ccmp_imm = .{
            .sf = sf,
            .rn = rn,
            .imm5 = imm5,
            .cond = cond,
            .nzcv = nzcv,
            .is_ccmn = op_bit == 0,
        } };
    }
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    return Op{ .ccmp_reg = .{
        .sf = sf,
        .rn = rn,
        .rm = rm,
        .cond = cond,
        .nzcv = nzcv,
        .is_ccmn = op_bit == 0,
    } };
}

fn decodeCondSelect(insn: u32) Op {
    const sf: bool = (insn >> 31) != 0;
    const op_bit: u1 = @truncate(insn >> 30);
    const op2: u2 = @truncate((insn >> 10) & 0x3);
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const cond: u4 = @truncate((insn >> 12) & 0xf);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    const cs = CondSelect{ .sf = sf, .rd = rd, .rn = rn, .rm = rm, .cond = cond };
    return switch ((@as(u3, op_bit) << 2) | op2) {
        0b000 => Op{ .csel = cs },
        0b001 => Op{ .csinc = cs },
        0b100 => Op{ .csinv = cs },
        0b101 => Op{ .csneg = cs },
        else => Op{ .unknown = insn },
    };
}

fn decodeDataProc12(insn: u32) Op {
    // bit 30 distinguishes 1-source (1) from 2-source (0). S is always 0
    const bit30: u1 = @truncate(insn >> 30);
    if (bit30 == 1) return decodeDataProc1Source(insn);
    const op_id: u6 = @truncate((insn >> 10) & 0x3f);
    const sf: bool = (insn >> 31) != 0;
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    const dr = DivReg{ .sf = sf, .rd = rd, .rn = rn, .rm = rm };
    const sr = ShiftReg{ .sf = sf, .rd = rd, .rn = rn, .rm = rm };
    return switch (op_id) {
        0b000010 => Op{ .udiv = dr },
        0b000011 => Op{ .sdiv = dr },
        0b001000 => Op{ .lsl_reg = sr },
        0b001001 => Op{ .lsr_reg = sr },
        0b001010 => Op{ .asr_reg = sr },
        0b001011 => Op{ .ror_reg = sr },
        else => Op{ .unknown = insn },
    };
}

fn decodeDataProc1Source(insn: u32) Op {
    const sf: bool = (insn >> 31) != 0;
    const opcode2: u5 = @truncate((insn >> 16) & 0x1f);
    // armv8.3 pointer authentication (register form): pacia/pacib/pacda/pacdb,
    // autia/autib/autda/autdb, the *za/*zb zero-modifier variants, and xpaci/xpacd.
    // we do not model the pac key, so sign and auth both round-trip to identity;
    // the pointer register is in/out and left unchanged. correct for execution
    // because the abi always authenticates before use, so auth(sign(x)) == x.
    if (opcode2 == 0b00001 and sf) {
        const opcode: u6 = @truncate((insn >> 10) & 0x3f);
        if (opcode <= 0b010001) return Op{ .nop = {} };
    }
    if (opcode2 != 0) return Op{ .unknown = insn };
    const opcode: u6 = @truncate((insn >> 10) & 0x3f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    const cr = ClzRbit{ .sf = sf, .rd = rd, .rn = rn };
    return switch (opcode) {
        0b000000 => Op{ .rbit = cr },
        0b000001 => Op{ .rev = .{ .sf = sf, .rd = rd, .rn = rn, .opc = 1 } }, // REV16
        0b000010 => Op{ .rev = .{ .sf = sf, .rd = rd, .rn = rn, .opc = 2 } }, // REV/REV32
        0b000011 => Op{ .rev = .{ .sf = sf, .rd = rd, .rn = rn, .opc = 3 } }, // REV (64-bit)
        0b000100 => Op{ .clz = cr },
        0b000101 => Op{ .cls = cr },
        else => Op{ .unknown = insn },
    };
}

fn decodeMul(insn: u32) Op {
    const sf: bool = (insn >> 31) != 0;
    const op_id: u3 = @truncate((insn >> 21) & 0x7);
    const o0: u1 = @truncate(insn >> 15);
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const ra: u5 = @truncate((insn >> 10) & 0x1f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    const ma = MulAdd{ .sf = sf, .rd = rd, .rn = rn, .rm = rm, .ra = ra };
    return switch (op_id) {
        0b000 => if (o0 == 0) Op{ .madd = ma } else Op{ .msub = ma },
        0b001 => if (o0 == 0) Op{ .smaddl = ma } else Op{ .smsubl = ma },
        0b010 => Op{ .smulh = .{ .rd = rd, .rn = rn, .rm = rm } },
        0b101 => if (o0 == 0) Op{ .umaddl = ma } else Op{ .umsubl = ma },
        0b110 => Op{ .umulh = .{ .rd = rd, .rn = rn, .rm = rm } },
        else => Op{ .unknown = insn },
    };
}

// ── fp/simd (op0 = x111) ──

fn decodeFpSimd(insn: u32) Op {
    const bits_28_24: u5 = @truncate((insn >> 24) & 0x1f);
    const bit_31: u1 = @truncate(insn >> 31);
    const bit_30: u1 = @truncate(insn >> 30);
    const bit_29: u1 = @truncate(insn >> 29);
    const bit_23: u1 = @truncate(insn >> 23);
    const bit_21: u1 = @truncate(insn >> 21);
    const bit_15: u1 = @truncate(insn >> 15);
    const bit_10: u1 = @truncate(insn >> 10);

    // cryptographic SHA512 (FEAT_SHA512): 0xCE group. three-register (bits 23:21 == 011, 15:12 == 1000)
    // covers SHA512H/H2/SU1 (11 = RAX1, a SHA3 op we don't provide); two-register SHA512SU0 is fixed.
    if ((insn >> 24) & 0xff == 0xce) {
        if ((insn >> 21) & 0x7 == 0b011 and (insn >> 12) & 0xf == 0b1000) {
            const so: ?CryptoShaOp = switch (@as(u2, @truncate((insn >> 10) & 0x3))) {
                0b00 => .sha512h, 0b01 => .sha512h2, 0b10 => .sha512su1, else => null,
            };
            if (so) |o| return Op{ .crypto_sha = .{ .op = o, .rd = @truncate(insn & 0x1f), .rn = @truncate((insn >> 5) & 0x1f), .rm = @truncate((insn >> 16) & 0x1f) } };
        }
        if ((insn & 0xfffffc00) == 0xcec08000) {
            return Op{ .crypto_sha = .{ .op = .sha512su0, .rd = @truncate(insn & 0x1f), .rn = @truncate((insn >> 5) & 0x1f), .rm = 0 } };
        }
    }
    // cryptographic SHA1/SHA256 (0x5E group, bit28==1 so it is NOT in the 0x4E AES block below).
    if ((insn >> 24) & 0xff == 0x5e) {
        // three-register: 0101 1110 000 Rm 0 opcode(3) 00 Rn Rd.
        if ((insn >> 21) & 0x7 == 0 and (insn >> 15) & 1 == 0 and (insn >> 10) & 0x3 == 0) {
            const so: ?CryptoShaOp = switch (@as(u3, @truncate((insn >> 12) & 0x7))) {
                0b000 => .sha1c, 0b001 => .sha1p, 0b010 => .sha1m, 0b011 => .sha1su0,
                0b100 => .sha256h, 0b101 => .sha256h2, 0b110 => .sha256su1, else => null,
            };
            if (so) |o| return Op{ .crypto_sha = .{ .op = o, .rd = @truncate(insn & 0x1f), .rn = @truncate((insn >> 5) & 0x1f), .rm = @truncate((insn >> 16) & 0x1f) } };
        }
        // two-register: 0101 1110 0010 1000 0 opcode(5) 10 Rn Rd.
        if ((insn >> 17) & 0x1f == 0b10100 and (insn >> 10) & 0x3 == 0b10) {
            const so: ?CryptoShaOp = switch (@as(u5, @truncate((insn >> 12) & 0x1f))) {
                0b00000 => .sha1h, 0b00001 => .sha1su1, 0b00010 => .sha256su0, else => null,
            };
            if (so) |o| return Op{ .crypto_sha = .{ .op = o, .rd = @truncate(insn & 0x1f), .rn = @truncate((insn >> 5) & 0x1f), .rm = 0 } };
        }
    }

    // advanced simd vector group: 0|Q|U|01110|... — bit 31 must be 0.
    // scalar simd (01|U|11110) has different bits 28-24 and is handled (rejected)
    // in the fp space below via the m/s gates
    if (bit_31 == 0 and bits_28_24 == 0b01110) {
        // cryptographic AES: 0100 1110 00 10100 opcode(5) 10 Rn Rd (fixed Q=1, U=0, size=00).
        if ((insn >> 24) & 0x7f == 0x4e and (insn >> 17) & 0x1f == 0b10100 and (insn >> 10) & 0x3 == 0b10) {
            const aes_op: ?CryptoAesOp = switch (@as(u5, @truncate((insn >> 12) & 0x1f))) {
                0b00100 => .aese,
                0b00101 => .aesd,
                0b00110 => .aesmc,
                0b00111 => .aesimc,
                else => null,
            };
            if (aes_op) |o| return Op{ .crypto_aes = .{ .op = o, .rd = @truncate(insn & 0x1f), .rn = @truncate((insn >> 5) & 0x1f) } };
        }
        // tbl/tbx and permute share bit_29==0,bit_21==0,bit_15==0 and are distinguished
        // by bits 11:10 -- TBL/TBX have 00, ZIP/UZP/TRN permutes have 10. the old check
        // keyed only on bit_10==0, which also matched permutes and wrongly rejected them.
        if (bit_29 == 0 and bit_21 == 0 and bit_15 == 0) {
            const lo2: u2 = @truncate((insn >> 10) & 0x3);
            if (lo2 == 0b00) {
                // tbl/tbx: 0|Q|001110|op2|0|Rm|0|len|op|00|Rn|Rd, op2(23:22) must be 0
                const op2: u2 = @truncate((insn >> 22) & 0x3);
                if (op2 == 0) return decodeSimdTbl(insn);
                return Op{ .unknown = insn };
            }
            // permute: 0|Q|001110|size|0|Rm|0|opcode|10|Rn|Rd
            if (lo2 == 0b10) return decodeSimdPermute(insn);
        }
        // EXT: 0|Q|101110|00|0|Rm|0|imm4|0|Rn|Rd (bit_29==1, op2==00, bit_15==0, bit_10==0)
        if (bit_29 == 1 and bit_21 == 0 and bit_15 == 0 and bit_10 == 0 and ((insn >> 22) & 0x3) == 0) {
            return Op{ .simd_ext = .{
                .q = ((insn >> 30) & 1) == 1,
                .rd = @truncate(insn & 0x1f),
                .rn = @truncate((insn >> 5) & 0x1f),
                .rm = @truncate((insn >> 16) & 0x1f),
                .index = @truncate((insn >> 11) & 0xf),
            } };
        }
        if (bit_21 == 1) {
            const b21_17: u5 = @truncate((insn >> 17) & 0x1f);
            const mid: u4 = @truncate((insn >> 17) & 0xf);
            const lo2: u2 = @truncate((insn >> 10) & 0x3);
            // across lanes: 0|Q|U|01110|size|11000|opcode|10|Rn|Rd (fixed field bits 21-17)
            if (b21_17 == 0b11000 and lo2 == 0b10) {
                if (decodeVecAcross(insn)) |op| return op;
                return Op{ .unknown = insn };
            }
            // two-reg misc: 0|Q|U|01110|size|10000|opcode|10|Rn|Rd
            if (mid == 0 and lo2 == 0b10) {
                if (decodeSimdTwoRegMisc(insn)) |op| return op;
                return Op{ .unknown = insn };
            }
            // three-same: 0|Q|U|01110|size|1|Rm|opcode|1|Rn|Rd
            if (bit_10 == 1) {
                if (decodeSimdThreeSame(insn)) |op| return op;
                return Op{ .unknown = insn };
            }
            // three-different (widening): 0|Q|U|01110|size|1|Rm|opcode|00|Rn|Rd
            if (lo2 == 0b00) {
                if (decodeVecWiden(insn)) |op| return op;
                return Op{ .unknown = insn };
            }
        }
        // copy: 0|Q|op|01110000|imm5|0|imm4|1|Rn|Rd
        if (bit_21 == 0 and bit_15 == 0 and bit_10 == 1) {
            const b23: u1 = @truncate(insn >> 23);
            const b22: u1 = @truncate(insn >> 22);
            if (b23 == 0 and b22 == 0) return decodeSimdCopy(insn);
        }
        return Op{ .unknown = insn };
    }

    // advanced simd modified-imm / shift-imm: 0|Q|U|011110|...
    if (bit_31 == 0 and bits_28_24 == 0b01111 and bit_23 == 0 and bit_10 == 1) {
        // modified imm requires bits 28-19 == 0111100000 (immh == 0000)
        const immh: u4 = @truncate((insn >> 19) & 0xf);
        if (immh == 0) return decodeSimdModifiedImm(insn);
        if (decodeSimdShiftImm(insn)) |op| return op;
        return Op{ .unknown = insn };
    }

    // advanced simd vector x indexed element: 0|Q|U|01111|size|L|M|Rm|opcode|H|0|Rn|Rd.
    // integer mul/mla/mls (same-width) and smull/umull/smlal/umlal/smlsl/umlsl (widening) by element.
    // openssl's NEON poly1305 uses umull/umlal-by-element, so this is on the TLS path.
    if (bit_31 == 0 and bits_28_24 == 0b01111 and bit_10 == 0) {
        return decodeSimdIndexedElem(insn);
    }

    // scalar advanced-simd copy: 01|0|11110000|imm5|0|0000|1|Rn|Rd = DUP (element -> scalar), e.g.
    // `mov d0, v1.d[1]` (extract a vector lane into a D register). only op=0, imm4=0000 exists.
    if (bit_31 == 0 and bit_30 == 1 and bits_28_24 == 0b11110 and bit_23 == 0 and ((insn >> 22) & 1) == 0 and
        bit_21 == 0 and bit_15 == 0 and bit_10 == 1 and ((insn >> 29) & 1) == 0 and ((insn >> 11) & 0xf) == 0)
    {
        const imm5c: u5 = @truncate((insn >> 16) & 0x1f);
        const idxc = decodeSimdElemIndex(imm5c);
        if (idxc.esize != 0) return Op{ .simd_dup_scalar = .{ .rd = @truncate(insn & 0x1f), .rn = @truncate((insn >> 5) & 0x1f), .esize = idxc.esize, .index = idxc.index } };
    }

    // scalar advanced-simd two-reg-misc: 01|U|11110|size|10000|opcode|10|Rn|Rd.
    // we handle the fp<->int converts (scvtf/ucvtf/fcvtzs/fcvtzu) that operate on a
    // single lane in an fp register -- wasm simd has no i64x2<->f64x2 lane convert, so
    // these must go through scalar wasm ops (see emitSimdScalarCvt). glibc/busybox
    // number formatting (dd's transfer rate) uses ucvtf d,d + fcvtzu here.
    if (bit_31 == 0 and bit_30 == 1 and bits_28_24 == 0b11110 and bit_21 == 1 and bit_15 == 1 and bit_10 == 0) {
        const u_bit: u1 = @truncate(insn >> 29);
        const b16_12: u5 = @truncate((insn >> 12) & 0x1f);
        const b21_17: u5 = @truncate((insn >> 17) & 0x1f);
        const b11: u1 = @truncate(insn >> 11);
        // scvtf/ucvtf: opcode 11101, size high (bit23) == 0. fcvtzs/fcvtzu: opcode
        // 11011, size high == 1. bit22 selects single(0)/double(1) in both.
        const is_cvtf = b16_12 == 0b11101 and bit_23 == 0;
        const is_fcvtz = b16_12 == 0b11011 and bit_23 == 1;
        if (b21_17 == 0b10000 and b11 == 1 and (is_cvtf or is_fcvtz)) {
            return Op{ .simd_scalar_cvt = .{
                .rd = @truncate(insn & 0x1f),
                .rn = @truncate((insn >> 5) & 0x1f),
                .dbl = ((insn >> 22) & 1) == 1,
                .signed = u_bit == 0,
                .to_int = is_fcvtz,
            } };
        }
    }

    // scalar advanced-simd shift by immediate: 01|U|111110|immh|immb|opcode|1|Rn|Rd.
    // only the 64-bit (immh=1xxx) D-reg forms are architecturally defined here. reduces
    // to one i64 op on lane 0. musl scalbn / float formatting uses shl d,d,#n.
    if (bit_31 == 0 and bit_30 == 1 and bits_28_24 == 0b11111 and bit_23 == 0 and bit_10 == 1) {
        const u_bit: u1 = @truncate(insn >> 29);
        const immh: u4 = @truncate((insn >> 19) & 0xf);
        const immb: u3 = @truncate((insn >> 16) & 0x7);
        const opc: u5 = @truncate((insn >> 11) & 0x1f);
        if ((immh & 0b1000) != 0) {
            const imm: u8 = (@as(u8, immh) << 3) | @as(u8, immb);
            const left: u8 = imm - 64; // 0..63 for shl/sli
            const right: u8 = 128 - imm; // 1..64 for the right shifts
            const so: ?ScalarShiftOp = switch (opc) {
                0b00000 => if (u_bit == 0) @as(ScalarShiftOp, .sshr) else .ushr,
                0b00010 => if (u_bit == 0) @as(ScalarShiftOp, .ssra) else .usra,
                0b01010 => if (u_bit == 0) @as(ScalarShiftOp, .shl) else .sli,
                0b01000 => if (u_bit == 1) @as(ScalarShiftOp, .sri) else null,
                else => null,
            };
            if (so) |o| {
                const is_left = o == .shl or o == .sli;
                return Op{ .simd_scalar_shift = .{
                    .rd = @truncate(insn & 0x1f),
                    .rn = @truncate((insn >> 5) & 0x1f),
                    .shift = if (is_left) left else right,
                    .op = o,
                } };
            }
        }
    }

    // fp <-> fixed-point convert: sf|0|S|11110|type|0|rmode|opcode|scale|Rn|Rd. bit21==0
    // (integer converts have bit21==1). fbits = 64 - scale. scvtf/ucvtf rmode=00 opcode
    // 010/011; fcvtzs/fcvtzu rmode=11 opcode 000/001.
    if (bit_30 == 0 and bit_29 == 0 and bits_28_24 == 0b11110 and bit_23 == 0 and bit_21 == 0) {
        const rmode: u2 = @truncate((insn >> 19) & 0x3);
        const opcode: u3 = @truncate((insn >> 16) & 0x7);
        const scale: u6 = @truncate((insn >> 10) & 0x3f);
        const ftype: u2 = @truncate((insn >> 22) & 0x3);
        const to_int = rmode == 0b11 and (opcode == 0b000 or opcode == 0b001);
        const to_fp = rmode == 0b00 and (opcode == 0b010 or opcode == 0b011);
        if (to_int or to_fp) {
            return Op{ .fp_fixed_cvt = .{
                .sf = bit_31 == 1,
                .rd = @truncate(insn & 0x1f),
                .rn = @truncate((insn >> 5) & 0x1f),
                .ftype = ftype,
                .fbits = 64 - @as(u8, scale),
                .signed = if (to_int) opcode == 0b000 else opcode == 0b010,
                .to_int = to_int,
            } };
        }
    }

    // fp scalar space: M|0|S|11110|ftype|1|... — require bit 30 == 0 and S == 0.
    // this gate is what excludes scalar simd (01|U|11110/11111, bit 30 == 1)
    if (bit_30 == 0 and bit_29 == 0 and bits_28_24 == 0b11110 and bit_21 == 1) {
        // fp <-> int conv (bit 31 = sf, allowed to be 1): bits 15-10 == 000000
        const lo6: u6 = @truncate((insn >> 10) & 0x3f);
        if (lo6 == 0) return decodeFpIntConv(insn);
        // everything below has M (bit 31) == 0
        if (bit_31 != 0) return Op{ .unknown = insn };
        // fp 1-source: bits 14-10 == 10000
        const lo5: u5 = @truncate((insn >> 10) & 0x1f);
        if (lo5 == 0b10000) return decodeFpDataProc1(insn);
        // fp 2-source: bits 11-10 == 10
        const lo2: u2 = @truncate((insn >> 10) & 0x3);
        if (lo2 == 0b10) return decodeFpDataProc2(insn);
        // fp conditional select: bits 11-10 == 11
        if (lo2 == 0b11) return decodeFpCsel(insn);
        // fp immediate: bits 12-10 == 100, bits 9-5 == 0
        const lo3: u3 = @truncate((insn >> 10) & 0x7);
        const rn_field: u5 = @truncate((insn >> 5) & 0x1f);
        if (lo3 == 0b100 and rn_field == 0) return decodeFpImm(insn);
        // fp compare: bit 15 == 0, bits 13-10 == 1000
        const lo4: u4 = @truncate((insn >> 10) & 0xf);
        if (bit_15 == 0 and lo4 == 0b1000) return decodeFpCmp(insn);
        return Op{ .unknown = insn };
    }

    // fp 3-source: M|0|S|11111|... — same M/bit30/S gates. scalar simd shift-imm
    // (01|U|111110, bit 30 == 1) must NOT land here
    if (bit_31 == 0 and bit_30 == 0 and bit_29 == 0 and bits_28_24 == 0b11111) {
        return decodeFpDataProc3(insn);
    }

    return Op{ .unknown = insn };
}

fn decodeFpIntConv(insn: u32) Op {
    const sf: bool = (insn >> 31) != 0;
    const ftype: u2 = @truncate((insn >> 22) & 0x3);
    const rmode: u2 = @truncate((insn >> 19) & 0x3);
    const opcode: u3 = @truncate((insn >> 16) & 0x7);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);

    if (rmode == 0 and opcode == 0b111) return Op{ .fmov_to_fp = .{ .rd = rd, .rn = rn, .sf = sf } };
    if (rmode == 0 and opcode == 0b110) return Op{ .fmov_from_fp = .{ .rd = rd, .rn = rn, .sf = sf } };
    if (rmode == 1 and opcode == 0b111 and sf and ftype == 0b10) return Op{ .fmov_top_to_fp = .{ .rd = rd, .rn = rn } };
    if (rmode == 1 and opcode == 0b110 and sf and ftype == 0b10) return Op{ .fmov_top_from_fp = .{ .rd = rd, .rn = rn } };

    const conv = FpIntCvt{ .sf = sf, .rd = rd, .rn = rn, .ftype = ftype };
    if (rmode == 0 and opcode == 0b010) return Op{ .scvtf_int = conv };
    if (rmode == 0 and opcode == 0b011) return Op{ .ucvtf_int = conv };
    if (rmode == 0b11 and opcode == 0b000) return Op{ .fcvtzs_int = conv };
    if (rmode == 0b11 and opcode == 0b001) return Op{ .fcvtzu_int = conv };
    return Op{ .unknown = insn };
}

fn decodeFpDataProc2(insn: u32) Op {
    const ftype: u2 = @truncate((insn >> 22) & 0x3);
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const opcode: u4 = @truncate((insn >> 12) & 0xf);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    const arith = FpArith2{ .ftype = ftype, .rd = rd, .rn = rn, .rm = rm };
    return switch (opcode) {
        0b0000 => Op{ .fmul = arith },
        0b0001 => Op{ .fdiv = arith },
        0b0010 => Op{ .fadd = arith },
        0b0011 => Op{ .fsub = arith },
        0b0110 => Op{ .fmaxnm = arith },
        0b0111 => Op{ .fminnm = arith },
        else => Op{ .unknown = insn },
    };
}

fn decodeFpCsel(insn: u32) Op {
    const ftype: u2 = @truncate((insn >> 22) & 0x3);
    if (ftype == 0b10) return Op{ .unknown = insn }; // unallocated type
    return Op{ .fcsel = .{
        .rd = @truncate(insn & 0x1f),
        .rn = @truncate((insn >> 5) & 0x1f),
        .rm = @truncate((insn >> 16) & 0x1f),
        .ftype = ftype,
        .cond = @truncate((insn >> 12) & 0xf),
    } };
}

fn decodeFpDataProc1(insn: u32) Op {
    const ftype: u2 = @truncate((insn >> 22) & 0x3);
    const opcode: u6 = @truncate((insn >> 15) & 0x3f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    const arith = FpArith1{ .ftype = ftype, .rd = rd, .rn = rn };
    return switch (opcode) {
        0b000000 => Op{ .fmov_fp = arith },
        0b000001 => Op{ .fabs = arith },
        0b000010 => Op{ .fneg = arith },
        0b000011 => Op{ .fsqrt = arith },
        0b000100, 0b000101 => blk: {
            // FCVT between precisions. opcode low 2 bits = dest ftype. we support single<->double
            // only; half needs real f16 conversion so leave it to SIGILL rather than mis-emit.
            const dst: u2 = @truncate((insn >> 15) & 0x3);
            if (ftype > 1 or dst > 1) break :blk Op{ .unknown = insn };
            break :blk Op{ .fcvt = .{ .rd = rd, .rn = rn, .ftype = ftype, .dest_ftype = dst } };
        },
        else => Op{ .unknown = insn },
    };
}

fn decodeFpImm(insn: u32) Op {
    const ftype: u2 = @truncate((insn >> 22) & 0x3);
    const imm8: u8 = @truncate((insn >> 13) & 0xff);
    const rd: u5 = @truncate(insn & 0x1f);
    return Op{ .fmov_imm = .{ .ftype = ftype, .rd = rd, .bits64 = vfpExpandImm(imm8, ftype) } };
}

fn vfpExpandImm(imm8: u8, ftype: u2) u64 {
    const sign: u64 = (imm8 >> 7) & 1;
    const b: u64 = (imm8 >> 6) & 1;
    const bcd: u64 = (imm8 >> 4) & 3;
    const efgh: u64 = imm8 & 0xf;

    if (ftype == 1) {
        const exp: u64 = ((1 - b) << 10) | (if (b != 0) @as(u64, 0x3FC) else 0) | bcd;
        const frac: u64 = efgh << 48;
        return (sign << 63) | (exp << 52) | frac;
    } else {
        const exp: u64 = ((1 - b) << 7) | (if (b != 0) @as(u64, 0x7C) else 0) | bcd;
        const frac: u64 = efgh << 19;
        return (sign << 31) | (exp << 23) | frac;
    }
}

fn decodeFpCmp(insn: u32) Op {
    const ftype: u2 = @truncate((insn >> 22) & 0x3);
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const opcode2: u5 = @truncate(insn & 0x1f);
    const with_zero = (opcode2 & 0b01000) != 0;
    return Op{ .fcmp = .{ .ftype = ftype, .rn = rn, .rm = rm, .with_zero = with_zero } };
}

fn decodeFpDataProc3(insn: u32) Op {
    const ftype: u2 = @truncate((insn >> 22) & 0x3);
    const o1: u1 = @truncate(insn >> 21);
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const o0: u1 = @truncate(insn >> 15);
    const ra: u5 = @truncate((insn >> 10) & 0x1f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    const arith = FpArith3{ .ftype = ftype, .rd = rd, .rn = rn, .rm = rm, .ra = ra };
    const variant: u2 = (@as(u2, o1) << 1) | @as(u2, o0);
    return switch (variant) {
        0 => Op{ .fmadd = arith },
        1 => Op{ .fmsub = arith },
        2 => Op{ .fnmadd = arith },
        3 => Op{ .fnmsub = arith },
    };
}

/// advanced simd three-same, fp subclass (opcode bits 15-11 in 0x18..0x1f).
/// size[1] (bit 23) is the 'a' op-selector, size[0] (bit 22) is sz (0=f32,1=f64)
fn decodeVecFp3(insn: u32) ?Op {
    const q: bool = ((insn >> 30) & 1) == 1;
    const u: u1 = @truncate(insn >> 29);
    const a: u1 = @truncate(insn >> 23);
    const sz: u1 = @truncate(insn >> 22);
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const opcode: u5 = @truncate((insn >> 11) & 0x1f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);

    const op: ?VecFpOp = switch (opcode) {
        0b11000 => if (u == 0) (if (a == 0) .maxnm else .minnm) else (if (a == 0) .maxnmp else .minnmp),
        0b11001 => if (u == 0) (if (a == 0) @as(VecFpOp, .mla) else .mls) else null,
        0b11010 => if (u == 0) (if (a == 0) @as(VecFpOp, .add) else .sub) else (if (a == 0) @as(VecFpOp, .addp) else .fabd),
        0b11011 => if (u == 1 and a == 0) @as(VecFpOp, .mul) else null, // fmulx (u=0) not yet
        0b11100 => if (u == 0 and a == 0) @as(VecFpOp, .cmeq) else if (u == 1) (if (a == 0) @as(VecFpOp, .cmge) else .cmgt) else null,
        0b11110 => if (u == 0) (if (a == 0) @as(VecFpOp, .max) else .min) else (if (a == 0) @as(VecFpOp, .maxp) else .minp),
        0b11111 => if (u == 1 and a == 0) @as(VecFpOp, .div) else null, // frecps/frsqrts not yet
        else => null,
    };
    if (op) |o| return Op{ .vec_fp3 = .{ .q = q, .sz = sz, .rd = rd, .rn = rn, .rm = rm, .op = o } };
    return null;
}

/// advanced simd across-lanes (reductions): opcode bits 16-12
fn decodeVecAcross(insn: u32) ?Op {
    const q: bool = ((insn >> 30) & 1) == 1;
    const u: u1 = @truncate(insn >> 29);
    const size: u2 = @truncate((insn >> 22) & 0x3);
    const opcode: u5 = @truncate((insn >> 12) & 0x1f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    const op: ?VecAcrossOp = switch (opcode) {
        0b11011 => if (u == 0) @as(VecAcrossOp, .addv) else null,
        0b00011 => if (u == 0) @as(VecAcrossOp, .saddlv) else .uaddlv,
        0b01010 => if (u == 0) @as(VecAcrossOp, .smaxv) else .umaxv,
        0b11010 => if (u == 0) @as(VecAcrossOp, .sminv) else .uminv,
        0b01111 => if (u == 1) @as(VecAcrossOp, .fmaxv) else null, // fp uses sz in size
        0b01100 => if (u == 1) @as(VecAcrossOp, .fminv) else null,
        else => null,
    };
    if (op) |o| return Op{ .vec_across = .{ .q = q, .size = size, .rd = rd, .rn = rn, .op = o } };
    return null;
}

/// advanced simd three-different (widening): opcode bits 15-12
fn decodeVecWiden(insn: u32) ?Op {
    const q: bool = ((insn >> 30) & 1) == 1;
    const u: u1 = @truncate(insn >> 29);
    const size: u2 = @truncate((insn >> 22) & 0x3);
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const opcode: u4 = @truncate((insn >> 12) & 0xf);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    // ADDHN (opcode 0100, u=0): add + take high half, narrowing. size = dest elem log2.
    if (opcode == 0b0100 and u == 0) {
        return Op{ .simd_addhn = .{ .q = q, .rd = rd, .rn = rn, .rm = rm, .size = size } };
    }
    // PMULL/PMULL2 (opcode 1110, u=0): 64x64->128 carryless multiply (size=11) for GHASH. q selects
    // the high (PMULL2) vs low (PMULL) 64-bit source lanes. (the 8x8->16 size=00 form is not emitted here.)
    if (opcode == 0b1110 and u == 0 and size == 0b11) {
        return Op{ .crypto_pmull = .{ .part = q, .rd = rd, .rn = rn, .rm = rm } };
    }
    const op: ?VecWidenOp = switch (opcode) {
        0b1100 => if (u == 0) @as(VecWidenOp, .smull) else .umull, // SMULL/UMULL
        0b1000 => if (u == 0) @as(VecWidenOp, .smlal) else .umlal, // SMLAL/UMLAL
        0b1010 => if (u == 0) @as(VecWidenOp, .smlsl) else .umlsl, // SMLSL/UMLSL
        0b0000 => if (u == 0) @as(VecWidenOp, .saddl) else .uaddl, // SADDL/UADDL
        0b0010 => if (u == 0) @as(VecWidenOp, .ssubl) else .usubl, // SSUBL/USUBL
        0b0001 => if (u == 0) @as(VecWidenOp, .saddw) else .uaddw, // SADDW/UADDW
        0b0011 => if (u == 0) @as(VecWidenOp, .ssubw) else .usubw, // SSUBW/USUBW
        else => null,
    };
    if (op) |o| return Op{ .vec_widen = .{ .q = q, .size = size, .rd = rd, .rn = rn, .rm = rm, .op = o } };
    return null;
}

fn decodeSimdThreeSame(insn: u32) ?Op {
    const q: bool = ((insn >> 30) & 1) == 1;
    const u: u1 = @truncate(insn >> 29);
    const size: u2 = @truncate((insn >> 22) & 0x3);
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const opcode: u5 = @truncate((insn >> 11) & 0x1f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);

    // fp three-same ops live at opcodes 0x18..0x1f
    if (opcode >= 0b11000) return decodeVecFp3(insn);

    const three = SimdThreeSame{ .q = q, .rd = rd, .rn = rn, .rm = rm, .size = size };

    if (opcode == 0b00011) {
        // bitwise: the size field selects the logical op (not element size).
        // u=0: 00 AND, 01 BIC, 10 ORR, 11 ORN. u=1: 00 EOR, 01 BSL, 10 BIT, 11 BIF.
        if (u == 0) return switch (size) {
            0b00 => Op{ .simd_and_reg = three },
            0b01 => Op{ .simd_bic_reg = three },
            0b10 => Op{ .simd_orr_reg = three },
            0b11 => Op{ .simd_orn_reg = three },
        };
        return switch (size) {
            0b00 => Op{ .simd_eor_reg = three },
            0b01 => Op{ .simd_bsl = three },
            0b10 => Op{ .simd_bit = three },
            0b11 => Op{ .simd_bif = three },
        };
    }
    // native-v128 integer ops route through vec_int3
    const vop: ?VecIntOp = switch (opcode) {
        0b10000 => if (u == 0) @as(VecIntOp, .add) else .sub,
        0b10001 => if (u == 1) @as(VecIntOp, .cmeq) else .cmtst,
        0b10011 => if (u == 0) @as(VecIntOp, .mul) else null, // mul (u=0); pmul (u=1) via helper
        0b00110 => if (u == 0) @as(VecIntOp, .cmgt) else .cmhi,
        0b00111 => if (u == 0) @as(VecIntOp, .cmge) else .cmhs,
        0b01100 => if (u == 0) @as(VecIntOp, .smax) else .umax,
        0b01101 => if (u == 0) @as(VecIntOp, .smin) else .umin,
        0b00001 => if (u == 0) @as(VecIntOp, .sqadd) else .uqadd,
        0b00101 => if (u == 0) @as(VecIntOp, .sqsub) else .uqsub,
        else => null,
    };
    if (vop) |o| return Op{ .vec_int3 = .{ .q = q, .size = size, .rd = rd, .rn = rn, .rm = rm, .op = o } };

    // remaining ops still go through the js simd_op helper (pairwise etc.)
    if (u == 1 and opcode == 0b10100) return Op{ .simd_umaxp = three };
    if (u == 1 and opcode == 0b10101) return Op{ .simd_uminp = three };
    if (u == 0 and opcode == 0b10111) return Op{ .simd_addp = three };
    return null;
}

fn decodeSimdTwoRegMisc(insn: u32) ?Op {
    const q: bool = ((insn >> 30) & 1) == 1;
    const u: u1 = @truncate(insn >> 29);
    const size: u2 = @truncate((insn >> 22) & 0x3);
    const opcode: u5 = @truncate((insn >> 12) & 0x1f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);

    // fp two-reg-misc: size high bit (23) marks the fp subgroup with sz=bit22
    const sz: u1 = @truncate(insn >> 22);
    const hi: u1 = @truncate(insn >> 23);
    if (hi == 1) {
        const fop: ?VecFpMiscOp = switch (opcode) {
            0b01111 => if (u == 0) @as(VecFpMiscOp, .abs) else .neg, // fabs / fneg
            0b11111 => if (u == 1) @as(VecFpMiscOp, .sqrt) else null, // fsqrt
            0b11010 => if (u == 0) @as(VecFpMiscOp, .fcvtzs) else .fcvtzu,
            0b11101 => if (u == 0) @as(VecFpMiscOp, .scvtf) else .ucvtf,
            0b01100 => if (u == 0) @as(VecFpMiscOp, .fcmgt0) else .fcmge0,
            0b01101 => if (u == 0) @as(VecFpMiscOp, .fcmeq0) else .fcmle0,
            0b11000 => if (u == 0) @as(VecFpMiscOp, .frintn) else .frinta,
            0b11001 => if (u == 0) @as(VecFpMiscOp, .frintm) else .frintx,
            else => null,
        };
        if (fop) |o| return Op{ .vec_fp2 = .{ .q = q, .sz = sz, .rd = rd, .rn = rn, .op = o } };
    }

    const no_sz = SimdTwoRegNoSize{ .q = q, .rd = rd, .rn = rn };
    const sized = SimdTwoReg{ .q = q, .rd = rd, .rn = rn, .size = size };
    if (u == 1 and size == 0 and opcode == 0b00101) return Op{ .simd_not = no_sz };
    if (u == 1 and size == 1 and opcode == 0b00101) return Op{ .simd_rbit = no_sz }; // RBIT (vector): reverse bits in each byte
    // integer neg/abs + compare-against-zero via native v128. glibc string/mem
    // routines lean heavily on cmeq #0 (strlen/memchr/strcmp).
    const iop: ?VecIntMiscOp = switch (opcode) {
        0b01011 => if (u == 1) @as(VecIntMiscOp, .neg) else .abs,
        0b01000 => if (u == 0) @as(VecIntMiscOp, .cmgt0) else .cmge0,
        0b01001 => if (u == 0) @as(VecIntMiscOp, .cmeq0) else .cmle0,
        0b01010 => if (u == 0) @as(VecIntMiscOp, .cmlt0) else null,
        else => null,
    };
    if (iop) |o| return Op{ .vec_int2 = .{ .q = q, .size = size, .rd = rd, .rn = rn, .op = o } };
    if (u == 0 and size == 0 and opcode == 0b00101) return Op{ .simd_cnt = no_sz };
    if (u == 0 and opcode == 0b00100) return Op{ .simd_cls_vec = sized };
    if (u == 1 and opcode == 0b00100) return Op{ .simd_clz_vec = sized };
    // REV64 (u=0, op 00000), REV32 (u=1, op 00000), REV16 (u=0, op 00001). `size` gives the
    // sub-element byte width being reversed within the container. used by NEON SHA/byteswap code.
    if (opcode == 0b00000 and u == 0) return Op{ .simd_rev = .{ .q = q, .rd = rd, .rn = rn, .size = size, .container = 8 } };
    if (opcode == 0b00000 and u == 1) return Op{ .simd_rev = .{ .q = q, .rd = rd, .rn = rn, .size = size, .container = 4 } };
    if (opcode == 0b00001 and u == 0) return Op{ .simd_rev = .{ .q = q, .rd = rd, .rn = rn, .size = size, .container = 2 } };
    // XTN/XTN2 (u=0, opcode 10010): narrow each element to its low half. `size` is the dest element
    // size (0=.8b<-.8h, 1=.4h<-.4s, 2=.2s<-.2d); q selects XTN2 (write the high 64 bits). poly1305 uses this.
    if (u == 0 and opcode == 0b10010) return Op{ .simd_xtn = .{ .q = q, .rd = rd, .rn = rn, .size = size } };
    return null;
}

fn decodeSimdIndexedElem(insn: u32) Op {
    const q: bool = ((insn >> 30) & 1) == 1;
    const u: bool = ((insn >> 29) & 1) == 1;
    const size: u2 = @truncate((insn >> 22) & 0x3);
    const l: u1 = @truncate((insn >> 21) & 1);
    const m: u1 = @truncate((insn >> 20) & 1);
    const h: u1 = @truncate((insn >> 11) & 1);
    const opcode: u4 = @truncate((insn >> 12) & 0xf);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    var rm: u5 = @truncate((insn >> 16) & 0x1f);
    var index: u8 = 0;
    // element size 16 uses index=H:L:M with a 3-bit Rm; size 32 uses index=H:L with a 4-bit Rm.
    if (size == 0b01) {
        index = (@as(u8, h) << 2) | (@as(u8, l) << 1) | @as(u8, m);
        rm = @truncate((insn >> 16) & 0x7);
    } else if (size == 0b10) {
        index = (@as(u8, h) << 1) | @as(u8, l);
    } else return Op{ .unknown = insn };
    // mul is U=0; mla/mls are U=0 in the encoding table? no — MLA/MLS use U=1, MUL uses U=0. the
    // widening s/u variants use U as the signedness flag. same-width mul/mla/mls are modular so
    // signedness is irrelevant to the low bits.
    const op: IdxMulOp = switch (opcode) {
        0b1000 => if (u) return Op{ .unknown = insn } else .mul, // MUL (U=0)
        0b0000 => if (u) .mla else return Op{ .unknown = insn }, // MLA (U=1)
        0b0100 => if (u) .mls else return Op{ .unknown = insn }, // MLS (U=1)
        0b1010 => .mull, // S/UMULL
        0b0010 => .mlal, // S/UMLAL
        0b0110 => .mlsl, // S/UMLSL
        else => return Op{ .unknown = insn }, // fp-indexed / saturating handled elsewhere / not yet
    };
    return Op{ .simd_idx_mul = .{ .q = q, .u = u, .rd = rd, .rn = rn, .rm = rm, .size = size, .index = index, .op = op } };
}

fn decodeSimdShiftImm(insn: u32) ?Op {
    const q: bool = ((insn >> 30) & 1) == 1;
    const u: u1 = @truncate(insn >> 29);
    const immh: u4 = @truncate((insn >> 19) & 0xf);
    const immb: u3 = @truncate((insn >> 16) & 0x7);
    const opcode: u5 = @truncate((insn >> 11) & 0x1f);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);

    const esize: u8 = blk: {
        if ((immh & 0b1000) != 0) break :blk 8;
        if ((immh & 0b0100) != 0) break :blk 4;
        if ((immh & 0b0010) != 0) break :blk 2;
        if ((immh & 0b0001) != 0) break :blk 1;
        return null;
    };
    const imm: u8 = (@as(u8, immh) << 3) | @as(u8, immb);

    if (opcode == 0b10100) {
        // sshll/ushll only exist for b/h/s elements (immh = 0001..0111)
        if (esize == 8) return null;
        const shift: u8 = imm - (esize * 8);
        const sl = SimdShiftLong{ .q = q, .rd = rd, .rn = rn, .esize = esize, .shift = shift };
        return if (u == 0) Op{ .simd_sshll = sl } else Op{ .simd_ushll = sl };
    }
    if (opcode == 0b10000 and u == 0) {
        // SHRN/SHRN2: narrow to a half-width dest. esize here is the DEST element
        // width (1/2/4 bytes; immh=1xxx / esize==8 is reserved). the source element
        // is twice as wide, so shift = src_bits - imm = esize*16 - imm.
        const size2: u2 = switch (esize) { 1 => 0, 2 => 1, 4 => 2, else => return null };
        const shift: u8 = @intCast(@as(u16, esize) * 16 - imm);
        return Op{ .simd_shrn = .{ .q = q, .rd = rd, .rn = rn, .size = size2, .shift = shift } };
    }
    if (opcode == 0b00000) {
        // SSHR (u=0) / USHR (u=1): same-width right shift. shift = 2*esize_bits - imm.
        const shift: u8 = @intCast(@as(u16, esize) * 16 - imm);
        return Op{ .simd_shift_imm = .{ .q = q, .rd = rd, .rn = rn, .esize = esize, .shift = shift, .op = if (u == 0) .sshr else .ushr } };
    }
    if (opcode == 0b01010 and u == 0) {
        // SHL: same-width left shift. shift = imm - esize_bits.
        const shift: u8 = @intCast(imm - @as(u16, esize) * 8);
        return Op{ .simd_shift_imm = .{ .q = q, .rd = rd, .rn = rn, .esize = esize, .shift = shift, .op = .shl } };
    }
    if (opcode == 0b01010 and u == 1) {
        // SLI: shift-left-insert. shift = imm - esize_bits. (reads Vd for the preserved low bits)
        const shift: u8 = @intCast(imm - @as(u16, esize) * 8);
        return Op{ .simd_shift_imm = .{ .q = q, .rd = rd, .rn = rn, .esize = esize, .shift = shift, .op = .sli } };
    }
    if (opcode == 0b01000 and u == 1) {
        // SRI: shift-right-insert. shift = 2*esize_bits - imm. (reads Vd for the preserved high bits)
        const shift: u8 = @intCast(@as(u16, esize) * 16 - imm);
        return Op{ .simd_shift_imm = .{ .q = q, .rd = rd, .rn = rn, .esize = esize, .shift = shift, .op = .sri } };
    }
    return null;
}

fn decodeSimdTbl(insn: u32) Op {
    const q: bool = ((insn >> 30) & 1) == 1;
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const len: u2 = @truncate((insn >> 13) & 0x3);
    const op_bit: u1 = @truncate(insn >> 12);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    return Op{ .simd_tbl = .{ .q = q, .rd = rd, .rn = rn, .rm = rm, .len = len, .is_tbx = op_bit == 1 } };
}

fn decodeSimdPermute(insn: u32) Op {
    const q: bool = ((insn >> 30) & 1) == 1;
    const size: u2 = @truncate((insn >> 22) & 0x3);
    const rm: u5 = @truncate((insn >> 16) & 0x1f);
    const opcode: u3 = @truncate((insn >> 12) & 0x7);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);
    const three = SimdThreeSame{ .q = q, .rd = rd, .rn = rn, .rm = rm, .size = size };
    return switch (opcode) {
        0b001 => Op{ .simd_uzp1 = three },
        0b010 => Op{ .simd_trn1 = three },
        0b011 => Op{ .simd_zip1 = three },
        0b101 => Op{ .simd_uzp2 = three },
        0b110 => Op{ .simd_trn2 = three },
        0b111 => Op{ .simd_zip2 = three },
        else => Op{ .unknown = insn },
    };
}

fn decodeSimdCopy(insn: u32) Op {
    const q: bool = (insn >> 30) & 1 == 1;
    const op: u1 = @truncate(insn >> 29);
    const imm5: u5 = @truncate((insn >> 16) & 0x1f);
    const imm4: u4 = @truncate((insn >> 11) & 0xf);
    const rn: u5 = @truncate((insn >> 5) & 0x1f);
    const rd: u5 = @truncate(insn & 0x1f);

    const idx = decodeSimdElemIndex(imm5);
    if (idx.esize == 0) return Op{ .unknown = insn };

    if (op == 0) {
        switch (imm4) {
            0b0000 => return Op{ .simd_dup_elem = .{ .rd = rd, .rn = rn, .esize = idx.esize, .index = idx.index, .q = q } },
            0b0001 => return Op{ .simd_dup_gpr = .{ .rd = rd, .rn = rn, .esize = idx.esize, .q = q } },
            0b0011 => return Op{ .simd_ins_gpr = .{ .rd = rd, .rn = rn, .esize = idx.esize, .index = idx.index, .q = q } },
            0b0101 => return Op{ .simd_umov = .{ .rd = rd, .rn = rn, .esize = idx.esize, .index = idx.index, .q = q } },
            0b0111 => return Op{ .simd_umov = .{ .rd = rd, .rn = rn, .esize = idx.esize, .index = idx.index, .q = q } },
            else => return Op{ .unknown = insn },
        }
    }
    if (op == 1) {
        // INS (element): Vd.T[dst] = Vn.T[src]. dst index from imm5, src index from imm4 >> log2(esize).
        const size_log2: u3 = switch (idx.esize) { 1 => 0, 2 => 1, 4 => 2, 8 => 3, else => return Op{ .unknown = insn } };
        const src_index: u5 = @truncate(@as(u32, imm4) >> size_log2);
        return Op{ .simd_ins_elem = .{ .rd = rd, .rn = rn, .esize = idx.esize, .dst_index = idx.index, .src_index = src_index } };
    }
    return Op{ .unknown = insn };
}

const SimdIdx = struct { esize: u8, index: u5 };

fn decodeSimdElemIndex(imm5: u5) SimdIdx {
    if ((imm5 & 0b00001) != 0) return .{ .esize = 1, .index = @truncate((imm5 >> 1) & 0xf) }; // B
    if ((imm5 & 0b00010) != 0) return .{ .esize = 2, .index = @truncate((imm5 >> 2) & 0x7) }; // H
    if ((imm5 & 0b00100) != 0) return .{ .esize = 4, .index = @truncate((imm5 >> 3) & 0x3) }; // S
    if ((imm5 & 0b01000) != 0) return .{ .esize = 8, .index = @truncate((imm5 >> 4) & 0x1) }; // D
    return .{ .esize = 0, .index = 0 };
}

fn decodeSimdModifiedImm(insn: u32) Op {
    const q: bool = (insn >> 30) & 1 == 1;
    const op: u1 = @truncate(insn >> 29);
    const a: u1 = @truncate(insn >> 18);
    const b: u1 = @truncate(insn >> 17);
    const c: u1 = @truncate(insn >> 16);
    const cmode: u4 = @truncate((insn >> 12) & 0xf);
    const d: u1 = @truncate(insn >> 9);
    const e: u1 = @truncate(insn >> 8);
    const f: u1 = @truncate(insn >> 7);
    const g: u1 = @truncate(insn >> 6);
    const h: u1 = @truncate(insn >> 5);
    const rd: u5 = @truncate(insn & 0x1f);

    const imm8: u8 = (@as(u8, a) << 7) | (@as(u8, b) << 6) | (@as(u8, c) << 5) |
        (@as(u8, d) << 4) | (@as(u8, e) << 3) | (@as(u8, f) << 2) |
        (@as(u8, g) << 1) | @as(u8, h);
    return Op{ .simd_movi = .{ .rd = rd, .imm8 = imm8, .cmode = cmode, .op = op, .q = q } };
}
