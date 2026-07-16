// shared js helpers imported by translated wasm modules.
// these implement arm semantics that don't map naturally to wasm: ieee 754
// rounding, simd vector ops over linear memory, 128-bit multiply highs.

import { V_OFFSET, GUEST_MEM_BASE } from './types.js';
import { aese, aesd, aesmc, aesimc, pmull,
         sha1_3reg, sha1h, sha1su0, sha1su1,
         sha256h, sha256su0, sha256su1,
         sha512h, sha512h2, sha512su0, sha512su1 } from './crypto-ext.js';

const _fpBuf = new ArrayBuffer(8);
const _f64View = new Float64Array(_fpBuf);
const _f32View = new Float32Array(_fpBuf);
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

export const helperFunctions: Record<string, Function> = {
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
  f64_mul(a: bigint, b: bigint): bigint { return f64ToBits(f64FromBits(a) * f64FromBits(b)); },
  f64_div(a: bigint, b: bigint): bigint { return f64ToBits(f64FromBits(a) / f64FromBits(b)); },
  f64_add(a: bigint, b: bigint): bigint { return f64ToBits(f64FromBits(a) + f64FromBits(b)); },
  f64_sub(a: bigint, b: bigint): bigint { return f64ToBits(f64FromBits(a) - f64FromBits(b)); },
  f64_abs(a: bigint): bigint { return f64ToBits(Math.abs(f64FromBits(a))); },
  f64_neg(a: bigint): bigint { return f64ToBits(-f64FromBits(a)); },
  f64_sqrt(a: bigint): bigint { return f64ToBits(Math.sqrt(f64FromBits(a))); },
  f64_min(a: bigint, b: bigint): bigint {
    const fa = f64FromBits(a); const fb = f64FromBits(b);
    if (isNaN(fa) || isNaN(fb)) return f64ToBits(NaN);
    return f64ToBits(Math.min(fa, fb));
  },
  f64_max(a: bigint, b: bigint): bigint {
    const fa = f64FromBits(a); const fb = f64FromBits(b);
    if (isNaN(fa) || isNaN(fb)) return f64ToBits(NaN);
    return f64ToBits(Math.max(fa, fb));
  },
  f64_from_s64(a: bigint): bigint { return f64ToBits(Number(BigInt.asIntN(64, a))); },
  f64_from_u64(a: bigint): bigint { return f64ToBits(Number(BigInt.asUintN(64, a))); },
  f64_from_s32(a: bigint): bigint { return f64ToBits(Number(BigInt.asIntN(32, a))); },
  f64_from_u32(a: bigint): bigint { return f64ToBits(Number(BigInt.asUintN(32, a))); },
  f64_to_s64(a: bigint): bigint { return BigInt.asUintN(64, BigInt(Math.trunc(f64FromBits(a)))); },
  f64_to_u64(a: bigint): bigint { const v = Math.trunc(f64FromBits(a)); return v < 0 ? 0n : BigInt.asUintN(64, BigInt(v)); },
  f64_to_s32(a: bigint): bigint { return BigInt.asUintN(64, BigInt(Math.trunc(f64FromBits(a))) & 0xFFFFFFFFn); },
  f64_to_u32(a: bigint): bigint { const v = Math.trunc(f64FromBits(a)); return v < 0 ? 0n : BigInt(v) & 0xFFFFFFFFn; },
  f32_to_f64(a: bigint): bigint {
    const f = f32FromBits(Number(a & 0xFFFFFFFFn));
    return f64ToBits(f);
  },
  f64_to_f32(a: bigint): bigint {
    const f = f64FromBits(a);
    return BigInt(f32ToBits(Math.fround(f)));
  },
  f64_cmp(a: bigint, b: bigint): bigint {
    const fa = f64FromBits(a);
    const fb = f64FromBits(b);
    if (isNaN(fa) || isNaN(fb)) return 3n;
    if (fa === fb) return 6n;
    if (fa < fb) return 8n;
    return 2n;
  },
};

// simd_op packed encoding: byte 7 = opcode, 6 = Q, 5 = size, 4 = extra,
// 3 = rd, 2 = rn, 1 = rm. all ops read Vn and Vm (or just Vn for two-reg
// misc), write Vd, optionally zero the upper qword when Q=0
export const SIMD_ADD = 1, SIMD_SUB = 2;
export const SIMD_CMEQ = 10, SIMD_CMGT = 11, SIMD_CMHI = 12, SIMD_CMGE = 13, SIMD_CMHS = 14;
export const SIMD_UMAXP = 20, SIMD_ADDP = 21;
export const SIMD_TBL = 30, SIMD_TBX = 31;
export const SIMD_ZIP1 = 40, SIMD_ZIP2 = 41, SIMD_UZP1 = 42, SIMD_UZP2 = 43;
export const SIMD_TRN1 = 44, SIMD_TRN2 = 45;
// element-wise integer ops used only for (op,size) combos wasm SIMD lacks
// natively: i8 mul, i64 min/max/unsigned-cmp, 32/64-bit saturating add/sub
export const SIMD_MUL = 50, SIMD_SMAX = 51, SIMD_SMIN = 52, SIMD_UMIN = 53, SIMD_UMAX = 54;
export const SIMD_SQADD = 55, SIMD_UQADD = 56, SIMD_SQSUB = 57, SIMD_UQSUB = 58;
// across-lanes reductions (result scalar in Vd lane 0, rest zeroed)
export const SIMD_ADDV = 60, SIMD_SADDLV = 61, SIMD_UADDLV = 62;
export const SIMD_SMAXV = 63, SIMD_UMAXV = 64, SIMD_SMINV = 65, SIMD_UMINV = 66;
export const SIMD_FMAXV = 67, SIMD_FMINV = 68;
export const SIMD_LDST_STRUCT = 70;
export const SIMD_SHRN = 72;
export const SIMD_ADDHN = 73;
export const SIMD_UMINP = 22;
export const SIMD_EXT = 74;
export const SIMD_REV = 76; // REV16/REV32/REV64: reverse `size`-byte elements within `extra`-byte containers
export const SIMD_RBIT = 77; // RBIT (vector): reverse the bits within each byte
export const SIMD_IDX_MUL = 78; // MUL/MLA/MLS + widening S/UMULL,S/UMLAL,S/UMLSL by indexed element
export const SIMD_XTN = 79; // XTN/XTN2: narrow each element to its low half (size = dest element log2)
export const SIMD_AESE = 80, SIMD_AESD = 81, SIMD_AESMC = 82, SIMD_AESIMC = 83, SIMD_PMULL = 84; // crypto ext
// SHA crypto ext: SHA1 (85-90), SHA256 (91-94), SHA512 (95-98). sha1c/p/m share sha1_3reg via `size`.
export const SIMD_SHA1C = 85, SIMD_SHA1P = 86, SIMD_SHA1M = 87, SIMD_SHA1H = 88, SIMD_SHA1SU0 = 89, SIMD_SHA1SU1 = 90;
export const SIMD_SHA256H = 91, SIMD_SHA256H2 = 92, SIMD_SHA256SU0 = 93, SIMD_SHA256SU1 = 94;
export const SIMD_SHA512H = 95, SIMD_SHA512H2 = 96, SIMD_SHA512SU0 = 97, SIMD_SHA512SU1 = 98;
export const SIMD_LDST1_SINGLE = 99; // LD1/ST1 single-lane: extra = index | load<<4 | post<<5, rn=base gpr, rm=post gpr

export function createSimdOp(memory: WebAssembly.Memory) {
  return function simd_op(packed: bigint, _extra: bigint): bigint {
    const op = Number((packed >> 56n) & 0xFFn);
    const Q = Number((packed >> 48n) & 0xFFn);
    const size = Number((packed >> 40n) & 0xFFn);
    const extra = Number((packed >> 32n) & 0xFFn);
    const rd = Number((packed >> 24n) & 0xFFn);
    const rn = Number((packed >> 16n) & 0xFFn);
    const rm = Number((packed >> 8n) & 0xFFn);

    const buf = new Uint8Array(memory.buffer);

    // LD2/LD3/LD4 (or ST): de/interleaving structured load/store. rd=first V reg,
    // rn=base GPR index, extra = regCount | (isLoad<<4)
    if (op === SIMD_LDST_STRUCT) {
        const regCount = extra & 0xf;
        const isLoad = (extra >> 4) & 1;
        const es = 1 << size;
        const laneBytes = Q ? 16 : 8;
        const lanes = laneBytes / es;
        const gview = new DataView(memory.buffer);
        const addr = Number(gview.getBigUint64(rn * 8, true)) + GUEST_MEM_BASE;
        const cp = (from: number, to: number) => { for (let k = 0; k < es; k++) buf[to + k] = buf[from + k]; };
        if (isLoad) {
            for (let e = 0; e < lanes; e++) {
                for (let r = 0; r < regCount; r++) {
                    const memOff = addr + (e * regCount + r) * es;
                    const vOff = V_OFFSET + ((rd + r) & 31) * 16 + e * es;
                    cp(memOff, vOff);
                }
            }
            // zero upper 64 bits of each dest reg when Q=0
            if (!Q) for (let r = 0; r < regCount; r++) buf.fill(0, V_OFFSET + ((rd + r) & 31) * 16 + 8, V_OFFSET + ((rd + r) & 31) * 16 + 16);
        } else {
            for (let e = 0; e < lanes; e++) {
                for (let r = 0; r < regCount; r++) {
                    const memOff = addr + (e * regCount + r) * es;
                    const vOff = V_OFFSET + ((rd + r) & 31) * 16 + e * es;
                    cp(vOff, memOff);
                }
            }
        }
        return 0n;
    }

    // SHRN/SHRN2: shift each source element right (unsigned) and truncate to the
    // half-width dest element. source is always the full 128-bit reg; `size` is the
    // DEST element log2, `extra` is the shift. Q=0 writes the low half (SHRN), Q=1
    // writes the high half preserving the low (SHRN2).
    if (op === SIMD_SHRN) {
      const destEs = 1 << size;
      const srcEs = destEs * 2;
      const shift = BigInt(extra);
      const n = 8 / destEs;
      const gv = new DataView(memory.buffer);
      const vnA = V_OFFSET + rn * 16, vdA = V_OFFSET + rd * 16;
      const out = new Uint8Array(8);
      const ov = new DataView(out.buffer);
      for (let i = 0; i < n; i++) {
        const src = srcEs === 2 ? BigInt(gv.getUint16(vnA + i * 2, true))
          : srcEs === 4 ? BigInt(gv.getUint32(vnA + i * 4, true))
          : gv.getBigUint64(vnA + i * 8, true);
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

    // ADDHN/ADDHN2: add two full-width source elements and take the HIGH half,
    // narrowing to a half-width dest. `size` is the dest element log2. Q selects
    // low half (ADDHN) vs high half preserving low (ADDHN2).
    if (op === SIMD_ADDHN) {
      const destEs = 1 << size;
      const srcEs = destEs * 2;
      const srcBits = BigInt(srcEs * 8);
      const srcMask = (1n << srcBits) - 1n;
      const n = 8 / destEs;
      const gv = new DataView(memory.buffer);
      const vnA = V_OFFSET + rn * 16, vmA = V_OFFSET + rm * 16, vdA = V_OFFSET + rd * 16;
      const rdSrc = (base: number, i: number): bigint =>
        srcEs === 2 ? BigInt(gv.getUint16(base + i * 2, true))
          : srcEs === 4 ? BigInt(gv.getUint32(base + i * 4, true))
          : gv.getBigUint64(base + i * 8, true);
      const out = new Uint8Array(8);
      const ov = new DataView(out.buffer);
      for (let i = 0; i < n; i++) {
        const sum = (rdSrc(vnA, i) + rdSrc(vmA, i)) & srcMask;
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

    // EXT: extract `bytes` bytes from the concatenation Vn:Vm starting at byte `extra`.
    if (op === SIMD_EXT) {
      const w = Q ? 16 : 8;
      const index = extra;
      const vnA = V_OFFSET + rn * 16, vmA = V_OFFSET + rm * 16, vdA = V_OFFSET + rd * 16;
      const concat = new Uint8Array(2 * w);
      for (let k = 0; k < w; k++) { concat[k] = buf[vnA + k]; concat[w + k] = buf[vmA + k]; }
      const out = new Uint8Array(16);
      for (let i = 0; i < w; i++) out[i] = concat[index + i];
      for (let k = 0; k < 16; k++) buf[vdA + k] = out[k];
      return 0n;
    }

    if (op === SIMD_REV) {
      // reverse the order of `es`-byte elements within each `container`-byte group (REV16/32/64).
      const w = Q ? 16 : 8;
      const container = extra; // 2, 4, or 8 bytes
      const es = 1 << size;    // element (sub-unit) byte width being reversed
      const nElems = container / es;
      const vnA = V_OFFSET + rn * 16, vdA = V_OFFSET + rd * 16;
      const src = new Uint8Array(16); src.set(buf.subarray(vnA, vnA + 16));
      for (let i = 0; i < 16; i++) {
        if (i >= w) { buf[vdA + i] = 0; continue; }
        const base = Math.floor(i / container) * container;
        const pos = i % container;
        const srcPos = base + (nElems - 1 - Math.floor(pos / es)) * es + (pos % es);
        buf[vdA + i] = src[srcPos];
      }
      return 0n;
    }

    if (op === SIMD_RBIT) {
      // reverse the bit order within each byte (operates on .8b / .16b).
      const w = Q ? 16 : 8;
      const vnA = V_OFFSET + rn * 16, vdA = V_OFFSET + rd * 16;
      const src = new Uint8Array(16); src.set(buf.subarray(vnA, vnA + 16));
      for (let i = 0; i < 16; i++) {
        if (i >= w) { buf[vdA + i] = 0; continue; }
        let b = src[i], r = 0;
        for (let k = 0; k < 8; k++) { r = (r << 1) | (b & 1); b >>= 1; }
        buf[vdA + i] = r;
      }
      return 0n;
    }

    if (op === SIMD_XTN) {
      // narrow each source element to its low `destEs` bytes. XTN (Q=0) writes the low 64 bits of Vd
      // and zeroes the high half; XTN2 (Q=1) writes the high 64 bits and preserves the low half.
      const destEs = 1 << size;            // 1, 2, or 4 bytes
      const srcEs = destEs * 2;
      const n = 8 / destEs;                // number of narrowed elements (fills 64 bits)
      const vnA = V_OFFSET + rn * 16, vdA = V_OFFSET + rd * 16;
      const src = new Uint8Array(16); src.set(buf.subarray(vnA, vnA + 16));
      const half = new Uint8Array(8);
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < destEs; k++) half[j * destEs + k] = src[j * srcEs + k];
      }
      if (Q) { for (let k = 0; k < 8; k++) buf[vdA + 8 + k] = half[k]; }
      else { for (let k = 0; k < 8; k++) buf[vdA + k] = half[k]; for (let k = 8; k < 16; k++) buf[vdA + k] = 0; }
      return 0n;
    }

    if (op === SIMD_AESE) { aese(buf, V_OFFSET + rd * 16, V_OFFSET + rn * 16); return 0n; }
    if (op === SIMD_AESD) { aesd(buf, V_OFFSET + rd * 16, V_OFFSET + rn * 16); return 0n; }
    if (op === SIMD_AESMC) { aesmc(buf, V_OFFSET + rd * 16, V_OFFSET + rn * 16); return 0n; }
    if (op === SIMD_AESIMC) { aesimc(buf, V_OFFSET + rd * 16, V_OFFSET + rn * 16); return 0n; }
    if (op === SIMD_PMULL) { pmull(buf, V_OFFSET + rd * 16, V_OFFSET + rn * 16, V_OFFSET + rm * 16, Q === 1); return 0n; }
    const D = V_OFFSET + rd * 16, N = V_OFFSET + rn * 16, Mv = V_OFFSET + rm * 16;
    if (op === SIMD_SHA1C) { sha1_3reg(buf, D, N, Mv, 0); return 0n; }
    if (op === SIMD_SHA1P) { sha1_3reg(buf, D, N, Mv, 1); return 0n; }
    if (op === SIMD_SHA1M) { sha1_3reg(buf, D, N, Mv, 2); return 0n; }
    if (op === SIMD_SHA1H) { sha1h(buf, D, N); return 0n; }
    if (op === SIMD_SHA1SU0) { sha1su0(buf, D, N, Mv); return 0n; }
    if (op === SIMD_SHA1SU1) { sha1su1(buf, D, N); return 0n; }
    if (op === SIMD_SHA256H) { sha256h(buf, D, N, Mv, false); return 0n; }
    if (op === SIMD_SHA256H2) { sha256h(buf, D, N, Mv, true); return 0n; }
    if (op === SIMD_SHA256SU0) { sha256su0(buf, D, N); return 0n; }
    if (op === SIMD_SHA256SU1) { sha256su1(buf, D, N, Mv); return 0n; }
    if (op === SIMD_SHA512H) { sha512h(buf, D, N, Mv); return 0n; }
    if (op === SIMD_SHA512H2) { sha512h2(buf, D, N, Mv); return 0n; }
    if (op === SIMD_SHA512SU0) { sha512su0(buf, D, N); return 0n; }
    if (op === SIMD_SHA512SU1) { sha512su1(buf, D, N, Mv); return 0n; }
    if (op === SIMD_LDST1_SINGLE) {
      const esize = 1 << size;
      const index = extra & 0xf;
      const isLoad = (extra >> 4) & 1;
      const isPost = (extra >> 5) & 1;
      const gv = new DataView(memory.buffer);
      const addr = Number(gv.getBigUint64(rn * 8, true)) + GUEST_MEM_BASE;
      const laneAddr = V_OFFSET + rd * 16 + index * esize;
      if (isLoad) for (let k = 0; k < esize; k++) buf[laneAddr + k] = buf[addr + k];
      else for (let k = 0; k < esize; k++) buf[addr + k] = buf[laneAddr + k];
      if (isPost) {
        const inc = rm === 31 ? esize : Number(gv.getBigUint64(rm * 8, true));
        gv.setBigUint64(rn * 8, gv.getBigUint64(rn * 8, true) + BigInt(inc), true);
      }
      return 0n;
    }

    if (op === SIMD_IDX_MUL) {
      // multiply (mul/mla/mls, same-width) or widening multiply (s/umull, s/umlal, s/umlsl) by a
      // single indexed element of vm. `size` is log2 of the source element bytes (1=16b, 2=32b).
      const srcEs = 1 << size;                    // 2 or 4 bytes
      const uns = ((extra >> 6) & 1) === 1;
      const variant = (extra >> 3) & 7;           // 0mul 1mla 2mls 3mull 4mlal 5mlsl
      const idx = extra & 7;
      const widening = variant >= 3;
      const isSub = variant === 2 || variant === 5;
      const isAcc = variant !== 0 && variant !== 3;
      const destEs = widening ? srcEs * 2 : srcEs;
      const vnA = V_OFFSET + rn * 16, vmA = V_OFFSET + rm * 16, vdA = V_OFFSET + rd * 16;
      const oldVd = new Uint8Array(16); oldVd.set(buf.subarray(vdA, vdA + 16));
      const readEl = (b: Uint8Array, base: number, es: number, i: number, signed: boolean): bigint => {
        let v = 0n;
        for (let k = 0; k < es; k++) v |= BigInt(b[base + i * es + k]) << BigInt(k * 8);
        if (signed) { const bits = BigInt(es * 8); if (v >= (1n << (bits - 1n))) v -= (1n << bits); }
        return v;
      };
      const elem = readEl(buf, vmA, srcEs, idx, !uns);
      const nDest = widening ? (16 / destEs) : ((Q ? 16 : 8) / srcEs);
      const srcBase = widening && Q ? nDest : 0; // umull2/umlal2 use the high half of vn
      const out = new Uint8Array(16);
      for (let j = 0; j < nDest; j++) {
        let prod = BigInt.asUintN(destEs * 8, readEl(buf, vnA, srcEs, srcBase + j, !uns) * elem);
        if (isAcc) {
          let acc = 0n;
          for (let k = 0; k < destEs; k++) acc |= BigInt(oldVd[j * destEs + k]) << BigInt(k * 8);
          prod = BigInt.asUintN(destEs * 8, isSub ? acc - prod : acc + prod);
        }
        for (let k = 0; k < destEs; k++) out[j * destEs + k] = Number((prod >> BigInt(k * 8)) & 0xffn);
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

    // generic signed/unsigned lane accessors + writer for the element-wise ops
    const getS = (view: DataView, i: number): bigint => {
      if (esize === 1) return BigInt(view.getInt8(i));
      if (esize === 2) return BigInt(view.getInt16(i * 2, true));
      if (esize === 4) return BigInt(view.getInt32(i * 4, true));
      return view.getBigInt64(i * 8, true);
    };
    const getU = (view: DataView, i: number): bigint => {
      if (esize === 1) return BigInt(view.getUint8(i));
      if (esize === 2) return BigInt(view.getUint16(i * 2, true));
      if (esize === 4) return BigInt(view.getUint32(i * 4, true));
      return view.getBigUint64(i * 8, true);
    };
    const put = (i: number, val: bigint) => {
      const m = BigInt.asUintN(esize * 8, val);
      if (esize === 1) dView.setUint8(i, Number(m));
      else if (esize === 2) dView.setUint16(i * 2, Number(m), true);
      else if (esize === 4) dView.setUint32(i * 4, Number(m), true);
      else dView.setBigUint64(i * 8, m, true);
    };
    const bits = BigInt(esize * 8);
    const sMax = (1n << (bits - 1n)) - 1n, sMin = -(1n << (bits - 1n));
    const uMax = (1n << bits) - 1n;
    const clampS = (x: bigint) => x > sMax ? sMax : x < sMin ? sMin : x;
    const clampU = (x: bigint) => x > uMax ? uMax : x < 0n ? 0n : x;

    // reduction writer: place a scalar of `wbytes` width into dst lane 0
    const putScalar = (val: bigint, wbytes: number) => {
      const m = BigInt.asUintN(wbytes * 8, val);
      if (wbytes === 1) dView.setUint8(0, Number(m));
      else if (wbytes === 2) dView.setUint16(0, Number(m), true);
      else if (wbytes === 4) dView.setUint32(0, Number(m), true);
      else dView.setBigUint64(0, m, true);
    };

    switch (op) {
      case SIMD_ADDV: { let s = 0n; for (let i = 0; i < elems; i++) s += getU(aView, i); putScalar(s, esize); break; }
      case SIMD_SADDLV: { let s = 0n; for (let i = 0; i < elems; i++) s += getS(aView, i); putScalar(s, esize * 2); break; }
      case SIMD_UADDLV: { let s = 0n; for (let i = 0; i < elems; i++) s += getU(aView, i); putScalar(s, esize * 2); break; }
      case SIMD_SMAXV: { let s = getS(aView, 0); for (let i = 1; i < elems; i++) { const x = getS(aView, i); if (x > s) s = x; } putScalar(s, esize); break; }
      case SIMD_SMINV: { let s = getS(aView, 0); for (let i = 1; i < elems; i++) { const x = getS(aView, i); if (x < s) s = x; } putScalar(s, esize); break; }
      case SIMD_UMAXV: { let s = getU(aView, 0); for (let i = 1; i < elems; i++) { const x = getU(aView, i); if (x > s) s = x; } putScalar(s, esize); break; }
      case SIMD_UMINV: { let s = getU(aView, 0); for (let i = 1; i < elems; i++) { const x = getU(aView, i); if (x < s) s = x; } putScalar(s, esize); break; }
      case SIMD_FMAXV: case SIMD_FMINV: {
        const getF = (i: number) => esize === 4 ? aView.getFloat32(i * 4, true) : aView.getFloat64(i * 8, true);
        let s = getF(0);
        for (let i = 1; i < elems; i++) { const x = getF(i); s = op === SIMD_FMAXV ? Math.max(s, x) : Math.min(s, x); }
        if (esize === 4) dView.setFloat32(0, s, true); else dView.setFloat64(0, s, true);
        break;
      }
      case SIMD_MUL: for (let i = 0; i < elems; i++) put(i, getU(aView, i) * getU(bView, i)); break;
      case SIMD_SMAX: for (let i = 0; i < elems; i++) { const x = getS(aView, i), y = getS(bView, i); put(i, x > y ? x : y); } break;
      case SIMD_SMIN: for (let i = 0; i < elems; i++) { const x = getS(aView, i), y = getS(bView, i); put(i, x < y ? x : y); } break;
      case SIMD_UMAX: for (let i = 0; i < elems; i++) { const x = getU(aView, i), y = getU(bView, i); put(i, x > y ? x : y); } break;
      case SIMD_UMIN: for (let i = 0; i < elems; i++) { const x = getU(aView, i), y = getU(bView, i); put(i, x < y ? x : y); } break;
      case SIMD_SQADD: for (let i = 0; i < elems; i++) put(i, clampS(getS(aView, i) + getS(bView, i))); break;
      case SIMD_SQSUB: for (let i = 0; i < elems; i++) put(i, clampS(getS(aView, i) - getS(bView, i))); break;
      case SIMD_UQADD: for (let i = 0; i < elems; i++) put(i, clampU(getU(aView, i) + getU(bView, i))); break;
      case SIMD_UQSUB: for (let i = 0; i < elems; i++) put(i, clampU(getU(aView, i) - getU(bView, i))); break;
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
          if (esize === 1) dView.setUint8(i, eq ? 0xFF : 0);
          else if (esize === 2) dView.setUint16(i*2, eq ? 0xFFFF : 0, true);
          else if (esize === 4) dView.setUint32(i*4, eq ? 0xFFFFFFFF : 0, true);
          else dView.setBigUint64(i*8, eq ? 0xFFFFFFFFFFFFFFFFn : 0n, true);
        }
        break;
      case SIMD_CMGT:
        for (let i = 0; i < elems; i++) {
          let gt: boolean;
          if (esize === 1) gt = aView.getInt8(i) > bView.getInt8(i);
          else if (esize === 2) gt = aView.getInt16(i*2, true) > bView.getInt16(i*2, true);
          else if (esize === 4) gt = aView.getInt32(i*4, true) > bView.getInt32(i*4, true);
          else gt = aView.getBigInt64(i*8, true) > bView.getBigInt64(i*8, true);
          if (esize === 1) dView.setUint8(i, gt ? 0xFF : 0);
          else if (esize === 2) dView.setUint16(i*2, gt ? 0xFFFF : 0, true);
          else if (esize === 4) dView.setUint32(i*4, gt ? 0xFFFFFFFF : 0, true);
          else dView.setBigUint64(i*8, gt ? 0xFFFFFFFFFFFFFFFFn : 0n, true);
        }
        break;
      case SIMD_CMHI:
        for (let i = 0; i < elems; i++) {
          let hi: boolean;
          if (esize === 1) hi = aView.getUint8(i) > bView.getUint8(i);
          else if (esize === 2) hi = aView.getUint16(i*2, true) > bView.getUint16(i*2, true);
          else if (esize === 4) hi = aView.getUint32(i*4, true) > bView.getUint32(i*4, true);
          else hi = aView.getBigUint64(i*8, true) > bView.getBigUint64(i*8, true);
          if (esize === 1) dView.setUint8(i, hi ? 0xFF : 0);
          else if (esize === 2) dView.setUint16(i*2, hi ? 0xFFFF : 0, true);
          else if (esize === 4) dView.setUint32(i*4, hi ? 0xFFFFFFFF : 0, true);
          else dView.setBigUint64(i*8, hi ? 0xFFFFFFFFFFFFFFFFn : 0n, true);
        }
        break;
      case SIMD_CMGE: case SIMD_CMHS:
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
      case SIMD_UMAXP: case SIMD_UMINP: {
        const wantMax = op === SIMD_UMAXP;
        const all = new Uint8Array(32);
        all.set(a.subarray(0, bytes), 0);
        all.set(b_.subarray(0, bytes), bytes);
        const allView = new DataView(all.buffer);
        const pick = (v1: number | bigint, v2: number | bigint) => (wantMax ? (v1 > v2 ? v1 : v2) : (v1 < v2 ? v1 : v2));
        for (let i = 0; i < elems; i++) {
          const off1 = i * 2 * esize;
          const off2 = off1 + esize;
          if (esize === 1) dView.setUint8(i, pick(allView.getUint8(off1), allView.getUint8(off2)) as number);
          else if (esize === 2) dView.setUint16(i*2, pick(allView.getUint16(off1, true), allView.getUint16(off2, true)) as number, true);
          else if (esize === 4) dView.setUint32(i*4, pick(allView.getUint32(off1, true), allView.getUint32(off2, true)) as number, true);
          else dView.setBigUint64(i*8, pick(allView.getBigUint64(off1, true), allView.getBigUint64(off2, true)) as bigint, true);
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
          if (esize === 1) dView.setUint8(i, (allView.getUint8(off1) + allView.getUint8(off2)) & 0xFF);
          else if (esize === 2) dView.setUint16(i*2, (allView.getUint16(off1, true) + allView.getUint16(off2, true)) & 0xFFFF, true);
          else if (esize === 4) dView.setUint32(i*4, (allView.getUint32(off1, true) + allView.getUint32(off2, true)) >>> 0, true);
          else dView.setBigUint64(i*8, BigInt.asUintN(64, allView.getBigUint64(off1, true) + allView.getBigUint64(off2, true)), true);
        }
        break;
      }
      case SIMD_TBL: case SIMD_TBX: {
        const len = extra;
        const tableSize = (len + 1) * 16;
        const table = new Uint8Array(tableSize);
        for (let r = 0; r <= len; r++) {
          const vr = V_OFFSET + ((rn + r) & 31) * 16;
          table.set(buf.subarray(vr, vr + 16), r * 16);
        }
        // tbx preserves the existing destination byte for out-of-range indices
        if (op === SIMD_TBX) dst.set(buf.subarray(vd, vd + bytes));
        for (let i = 0; i < bytes; i++) {
          const idx = b_[i];
          if (idx < tableSize) dst[i] = table[idx];
          else if (op === SIMD_TBL) dst[i] = 0;
        }
        break;
      }
      case SIMD_ZIP1: case SIMD_ZIP2: {
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
      case SIMD_UZP1: case SIMD_UZP2: {
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
      case SIMD_TRN1: case SIMD_TRN2: {
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

    buf.set(dst.subarray(0, bytes), vd);
    if (!Q) buf.fill(0, vd + 8, vd + 16);
    return 0n;
  };
}

export function createEnv(
  memory: WebAssembly.Memory,
  regBase?: WebAssembly.Global | null,
  protActive?: WebAssembly.Global | null,
  memFault?: ((addr: number, isWrite: number) => void) | null,
): Record<string, unknown> {
  const env: Record<string, unknown> = { memory, ...helperFunctions, simd_op: createSimdOp(memory) };
  // threaded modules import `env.reg_base` (the per-thread register-block base). single-threaded
  // modules don't import it, so it's only added when provided.
  if (regBase) env.reg_base = regBase;
  // page-protection modules import `env.prot_active` (the guard on/off flag) and `env.mem_fault`
  // (raises SIGSEGV/SIGBUS). only added when protection is enabled.
  if (protActive) env.prot_active = protActive;
  if (memFault) env.mem_fault = memFault;
  return env;
}
