//! correctly-rounded f64 fused multiply-add, computed on the exact 106-bit product.
//!
//! arm's FMADD rounds a*b+c once, for the whole expression. wasm has no fma instruction and
//! jsc exposes no relaxed-simd, so the translator emits this algorithm inline. f32 gets the
//! cheap treatment instead (widen to f64, where the product is exact); f64 has no wider format
//! to hide in, so the product is carried as a 128-bit integer and rounded exactly once at the
//! end. the shape follows musl's fma.c.
//!
//! the emitter in translate.zig transcribes this function step for step, which is why it lives
//! here as a running, tested reference rather than only as emitted bytes: a divergence on the
//! device then means a transcription bug, not an algorithm bug.

const std = @import("std");
const wasm = @import("wasm.zig");

/// normalize() maps zero to an exponent above this and inf/nan to exactly this, so one
/// comparison separates every special case from the finite ones.
pub const zeroinfnan: i32 = 0x7ff - 0x3ff - 52 - 1;

pub const Num = struct { m: u64, e: i32, neg: bool };

/// significand shifted so the leading one sits at bit 53, with the exponent to match.
pub fn normalize(x: f64) Num {
    var ix: u64 = @bitCast(x);
    const neg = (ix >> 63) != 0;
    var e: i32 = @intCast((ix >> 52) & 0x7ff);
    if (e == 0) {
        // subnormal or zero. scaling by 2^63 is exact and lands it in the normal range;
        // zero stays zero, and its still-empty exponent field is what flags it as special.
        ix = @bitCast(x * 0x1p63);
        const e2: i32 = @intCast((ix >> 52) & 0x7ff);
        e = if (e2 != 0) e2 - 63 else 0x800;
    }
    ix &= (@as(u64, 1) << 52) - 1;
    ix |= @as(u64, 1) << 52;
    ix <<= 1;
    e -= 0x3ff + 52 + 1;
    return .{ .m = ix, .e = e, .neg = neg };
}

/// full 128-bit product. both operands are under 2^54, so the cross term cannot overflow.
pub fn mul128(x: u64, y: u64) struct { hi: u64, lo: u64 } {
    const xlo: u64 = x & 0xffffffff;
    const xhi: u64 = x >> 32;
    const ylo: u64 = y & 0xffffffff;
    const yhi: u64 = y >> 32;
    const t1 = xlo *% ylo;
    const t2 = xlo *% yhi +% xhi *% ylo;
    const t3 = xhi *% yhi;
    const lo = t1 +% (t2 << 32);
    const hi = t3 +% (t2 >> 32) +% @as(u64, @intFromBool(t1 > lo));
    return .{ .hi = hi, .lo = lo };
}

/// x * 2^n for a result already known to be normal (or to overflow). splitting the exponent
/// in half keeps both scale factors normal, so neither multiply rounds.
fn scaleNormal(x: f64, n: i32) f64 {
    // anything at or above 962 overflows for certain (the significand sits at 2^62), and
    // clamping keeps the scale factors representable
    const k: i32 = if (n > 1100) 1100 else n;
    const k1 = k >> 1;
    const k2 = k - k1;
    const c1: f64 = @bitCast(@as(u64, @intCast(0x3ff + k1)) << 52);
    const c2: f64 = @bitCast(@as(u64, @intCast(0x3ff + k2)) << 52);
    return x * c1 * c2;
}

/// round the 63-bit significand to a multiple of 2^k, half to even. k is at least 11 here, so
/// the quotient is at most 2^52 and converts to f64 exactly.
fn roundShift(m: u64, k: u6) u64 {
    const q = m >> k;
    const rem = m & ((@as(u64, 1) << k) - 1);
    const half = @as(u64, 1) << (k - 1);
    if (rem > half or (rem == half and (q & 1) != 0)) return q + 1;
    return q;
}

pub fn fma(x: f64, y: f64, z: f64) f64 {
    const nx = normalize(x);
    const ny = normalize(y);
    const nz = normalize(z);

    // a zero or non-finite x or y makes the unfused expression exact
    if (nx.e >= zeroinfnan or ny.e >= zeroinfnan) return x * y + z;
    // past that point x and y are both finite and nonzero, so the exact product is finite
    // and nonzero too
    if (nz.e >= zeroinfnan) {
        // adding zero to a nonzero product cannot change it, and going through the add would:
        // a product that underflows keeps its own sign, where (-0) + (+0) is (+0)
        if (nz.e > zeroinfnan) return x * y;
        // z is nan, and quieting a signalling one is the add's job
        if (z != z) return x * y + z;
        // z is infinite, so it swallows any finite product
        return z;
    }

    const p = mul128(nx.m, ny.m);
    var rhi = p.hi;
    var rlo = p.lo;
    var zhi: u64 = 0;
    var zlo: u64 = 0;

    // align the addend against the product, keeping every bit that falls off as a sticky one
    var e: i32 = nx.e + ny.e;
    var d: i32 = nz.e - e;
    if (d > 0) {
        if (d < 64) {
            zlo = nz.m << @intCast(d);
            zhi = nz.m >> @intCast(64 - d);
        } else {
            zlo = 0;
            zhi = nz.m;
            e = nz.e - 64;
            d -= 64;
            if (d == 0) {
                // already aligned
            } else if (d < 64) {
                rlo = (rhi << @intCast(64 - d)) | (rlo >> @intCast(d)) |
                    @intFromBool((rlo << @intCast(64 - d)) != 0);
                rhi = rhi >> @intCast(d);
            } else {
                rlo = 1;
                rhi = 0;
            }
        }
    } else {
        d = -d;
        if (d == 0) {
            zlo = nz.m;
        } else if (d < 64) {
            zlo = (nz.m >> @intCast(d)) | @intFromBool((nz.m << @intCast(64 - d)) != 0);
        } else {
            zlo = 1;
        }
    }

    var neg = nx.neg != ny.neg;
    var nonzero = true;
    if (neg == nz.neg) {
        rlo = rlo +% zlo;
        rhi = rhi +% zhi +% @intFromBool(rlo < zlo);
    } else {
        const t = rlo;
        rlo = rlo -% zlo;
        rhi = rhi -% zhi -% @intFromBool(t < rlo);
        if ((rhi >> 63) != 0) {
            rlo = 0 -% rlo;
            rhi = (0 -% rhi) -% @intFromBool(rlo != 0);
            neg = !neg;
        }
        nonzero = rhi != 0;
    }

    // renormalize so the leading one sits at bit 62, folding the rest into a sticky low bit.
    // both operands are under 2^54 and z is aligned beneath them, so rhi stays under 2^55 and
    // the shift below is always a real left shift.
    if (nonzero) {
        e += 64;
        d = @as(i32, @clz(rhi)) - 1;
        rhi = (rhi << @intCast(d)) | (rlo >> @intCast(64 - d)) |
            @intFromBool((rlo << @intCast(d)) != 0);
    } else if (rlo != 0) {
        d = @as(i32, @clz(rlo)) - 1;
        rhi = if (d < 0) (rlo >> 1) | (rlo & 1) else rlo << @intCast(d);
    } else {
        // the addends cancelled exactly; the unfused expression carries the right zero sign
        return x * y + z;
    }
    e -= d;

    // the value is now exactly +/- rhi * 2^e with rhi in [2^62, 2^63)
    if (e >= -1084) {
        // a normal result: the i64 to f64 conversion is the one and only rounding
        var i: i64 = @bitCast(rhi);
        if (neg) i = -i;
        return scaleNormal(@floatFromInt(i), e);
    }

    // subnormal: rounding to 53 bits first and to the subnormal grid second would round twice,
    // so round straight onto the grid, whose step is 2^-1074
    const k: i32 = -1074 - e;
    if (k >= 64) return if (neg) -0.0 else 0.0;
    const q = roundShift(rhi, @intCast(k));
    const r: f64 = @floatFromInt(q);
    const scaled = r * 0x1p-537 * 0x1p-537;
    return if (neg) -scaled else scaled;
}

// ── wasm transcription ─────────────────────────────────────────────────────
//
// what follows emits fma() above as a wasm function, statement for statement. it is a helper
// the translator declares once per module and calls from every f64 FMADD/FMSUB/FNMADD/FNMSUB
// and from each lane of a .2d FMLA/FMLS, so the ~200 instructions are paid for once rather
// than at every use site, and jsc gets a single body to optimize.

/// (f64 x, f64 y, f64 z) -> f64
pub const fma_locals: []const wasm.Module.Local = &.{
    .{ .count = 10, .ty = .i64 }, // 3..12
    .{ .count = 10, .ty = .i32 }, // 13..22
    .{ .count = 1, .ty = .f64 },  // 23
};

const px: u32 = 0;
const py: u32 = 1;
const pz: u32 = 2;
const l_nxm: u32 = 3;
const l_nym: u32 = 4;
const l_nzm: u32 = 5;
const l_rhi: u32 = 6;
const l_rlo: u32 = 7;
const l_zhi: u32 = 8;
const l_zlo: u32 = 9;
const l_t1: u32 = 10;
const l_t2: u32 = 11;
const l_t3: u32 = 12;
const l_nxe: u32 = 13;
const l_nye: u32 = 14;
const l_nze: u32 = 15;
const l_e: u32 = 16;
const l_d: u32 = 17;
const l_nxneg: u32 = 18;
const l_nyneg: u32 = 19;
const l_nzneg: u32 = 20;
const l_neg: u32 = 21;
const l_nonzero: u32 = 22;
const l_f: u32 = 23;

/// terse wrappers so the emitted sequence stays readable next to the zig it mirrors
const E = struct {
    b: *wasm.Body,
    fn op(s: E, o: wasm.Op) !void {
        try s.b.emit(o);
    }
    fn get(s: E, i: u32) !void {
        try s.b.emit(.local_get);
        try s.b.emitU32(i);
    }
    fn set(s: E, i: u32) !void {
        try s.b.emit(.local_set);
        try s.b.emitU32(i);
    }
    fn c32(s: E, v: i32) !void {
        try s.b.emitConstI32(v);
    }
    fn c64(s: E, v: i64) !void {
        try s.b.emitConstI64(v);
    }
    fn cf(s: E, v: f64) !void {
        try s.b.emitConstF64(v);
    }
    /// widen an i32 local into the i64 shift count wasm's shifts want
    fn shiftCount(s: E) !void {
        try s.op(.i64_extend_i32_u);
    }
};

fn emitNormalize(e: E, src: u32, m_out: u32, e_out: u32, neg_out: u32) !void {
    try e.get(src);
    try e.op(.i64_reinterpret_f64);
    try e.set(l_t1);

    try e.get(l_t1);
    try e.c64(63);
    try e.op(.i64_shr_u);
    try e.op(.i32_wrap_i64);
    try e.set(neg_out);

    try e.get(l_t1);
    try e.c64(52);
    try e.op(.i64_shr_u);
    try e.c64(0x7ff);
    try e.op(.i64_and);
    try e.op(.i32_wrap_i64);
    try e.set(e_out);

    try e.get(e_out);
    try e.op(.i32_eqz);
    try e.b.emitIf(.empty);
    {
        try e.get(src);
        try e.cf(0x1p63);
        try e.op(.f64_mul);
        try e.op(.i64_reinterpret_f64);
        try e.set(l_t1);

        try e.get(l_t1);
        try e.c64(52);
        try e.op(.i64_shr_u);
        try e.c64(0x7ff);
        try e.op(.i64_and);
        try e.op(.i32_wrap_i64);
        try e.set(e_out);

        // e = (e != 0) ? e - 63 : 0x800
        try e.get(e_out);
        try e.c32(63);
        try e.op(.i32_sub);
        try e.c32(0x800);
        try e.get(e_out);
        try e.op(.select);
        try e.set(e_out);
    }
    try e.op(.end);

    try e.get(l_t1);
    try e.c64((@as(i64, 1) << 52) - 1);
    try e.op(.i64_and);
    try e.c64(@as(i64, 1) << 52);
    try e.op(.i64_or);
    try e.c64(1);
    try e.op(.i64_shl);
    try e.set(m_out);

    try e.get(e_out);
    try e.c32(0x3ff + 52 + 1);
    try e.op(.i32_sub);
    try e.set(e_out);
}

/// x * y + z, evaluated the plain unfused way, for the special cases that need it
fn emitUnfused(e: E) !void {
    try e.get(px);
    try e.get(py);
    try e.op(.f64_mul);
    try e.get(pz);
    try e.op(.f64_add);
}

pub fn emitFmaBody(e_body: *wasm.Body) !void {
    const e = E{ .b = e_body };

    try emitNormalize(e, px, l_nxm, l_nxe, l_nxneg);
    try emitNormalize(e, py, l_nym, l_nye, l_nyneg);
    try emitNormalize(e, pz, l_nzm, l_nze, l_nzneg);

    // if (nx.e >= zeroinfnan or ny.e >= zeroinfnan) return x*y + z
    try e.get(l_nxe);
    try e.c32(zeroinfnan);
    try e.op(.i32_ge_s);
    try e.get(l_nye);
    try e.c32(zeroinfnan);
    try e.op(.i32_ge_s);
    try e.op(.i32_or);
    try e.b.emitIf(.empty);
    {
        try emitUnfused(e);
        try e.op(.@"return");
    }
    try e.op(.end);

    try e.get(l_nze);
    try e.c32(zeroinfnan);
    try e.op(.i32_ge_s);
    try e.b.emitIf(.empty);
    {
        // z == 0: the product keeps its own sign, where adding a zero would not
        try e.get(l_nze);
        try e.c32(zeroinfnan);
        try e.op(.i32_gt_s);
        try e.b.emitIf(.empty);
        {
            try e.get(px);
            try e.get(py);
            try e.op(.f64_mul);
            try e.op(.@"return");
        }
        try e.op(.end);
        // z is nan: let the add quiet it
        try e.get(pz);
        try e.get(pz);
        try e.op(.f64_ne);
        try e.b.emitIf(.empty);
        {
            try emitUnfused(e);
            try e.op(.@"return");
        }
        try e.op(.end);
        // z is infinite and swallows any finite product
        try e.get(pz);
        try e.op(.@"return");
    }
    try e.op(.end);

    // (rhi, rlo) = mul128(nx.m, ny.m)
    try e.get(l_nxm);
    try e.c64(0xffffffff);
    try e.op(.i64_and);
    try e.get(l_nym);
    try e.c64(0xffffffff);
    try e.op(.i64_and);
    try e.op(.i64_mul);
    try e.set(l_t1);

    try e.get(l_nxm);
    try e.c64(0xffffffff);
    try e.op(.i64_and);
    try e.get(l_nym);
    try e.c64(32);
    try e.op(.i64_shr_u);
    try e.op(.i64_mul);
    try e.get(l_nxm);
    try e.c64(32);
    try e.op(.i64_shr_u);
    try e.get(l_nym);
    try e.c64(0xffffffff);
    try e.op(.i64_and);
    try e.op(.i64_mul);
    try e.op(.i64_add);
    try e.set(l_t2);

    try e.get(l_nxm);
    try e.c64(32);
    try e.op(.i64_shr_u);
    try e.get(l_nym);
    try e.c64(32);
    try e.op(.i64_shr_u);
    try e.op(.i64_mul);
    try e.set(l_t3);

    try e.get(l_t1);
    try e.get(l_t2);
    try e.c64(32);
    try e.op(.i64_shl);
    try e.op(.i64_add);
    try e.set(l_rlo);

    try e.get(l_t3);
    try e.get(l_t2);
    try e.c64(32);
    try e.op(.i64_shr_u);
    try e.op(.i64_add);
    try e.get(l_t1);
    try e.get(l_rlo);
    try e.op(.i64_gt_u);
    try e.op(.i64_extend_i32_u);
    try e.op(.i64_add);
    try e.set(l_rhi);

    // e = nx.e + ny.e ; d = nz.e - e
    try e.get(l_nxe);
    try e.get(l_nye);
    try e.op(.i32_add);
    try e.set(l_e);
    try e.get(l_nze);
    try e.get(l_e);
    try e.op(.i32_sub);
    try e.set(l_d);

    try e.get(l_d);
    try e.c32(0);
    try e.op(.i32_gt_s);
    try e.b.emitIf(.empty);
    {
        try e.get(l_d);
        try e.c32(64);
        try e.op(.i32_lt_s);
        try e.b.emitIf(.empty);
        {
            try e.get(l_nzm);
            try e.get(l_d);
            try e.shiftCount();
            try e.op(.i64_shl);
            try e.set(l_zlo);

            try e.get(l_nzm);
            try e.c32(64);
            try e.get(l_d);
            try e.op(.i32_sub);
            try e.shiftCount();
            try e.op(.i64_shr_u);
            try e.set(l_zhi);
        }
        try e.op(.@"else");
        {
            try e.c64(0);
            try e.set(l_zlo);
            try e.get(l_nzm);
            try e.set(l_zhi);
            try e.get(l_nze);
            try e.c32(64);
            try e.op(.i32_sub);
            try e.set(l_e);
            try e.get(l_d);
            try e.c32(64);
            try e.op(.i32_sub);
            try e.set(l_d);

            try e.get(l_d);
            try e.op(.i32_eqz);
            try e.op(.i32_eqz);
            try e.b.emitIf(.empty);
            {
                try e.get(l_d);
                try e.c32(64);
                try e.op(.i32_lt_s);
                try e.b.emitIf(.empty);
                {
                    // rlo = rhi<<(64-d) | rlo>>d | sticky(rlo<<(64-d))
                    try e.get(l_rhi);
                    try e.c32(64);
                    try e.get(l_d);
                    try e.op(.i32_sub);
                    try e.shiftCount();
                    try e.op(.i64_shl);
                    try e.get(l_rlo);
                    try e.get(l_d);
                    try e.shiftCount();
                    try e.op(.i64_shr_u);
                    try e.op(.i64_or);
                    try e.get(l_rlo);
                    try e.c32(64);
                    try e.get(l_d);
                    try e.op(.i32_sub);
                    try e.shiftCount();
                    try e.op(.i64_shl);
                    try e.op(.i64_eqz);
                    try e.op(.i32_eqz);
                    try e.op(.i64_extend_i32_u);
                    try e.op(.i64_or);
                    try e.set(l_rlo);

                    try e.get(l_rhi);
                    try e.get(l_d);
                    try e.shiftCount();
                    try e.op(.i64_shr_u);
                    try e.set(l_rhi);
                }
                try e.op(.@"else");
                {
                    try e.c64(1);
                    try e.set(l_rlo);
                    try e.c64(0);
                    try e.set(l_rhi);
                }
                try e.op(.end);
            }
            try e.op(.end);
        }
        try e.op(.end);
    }
    try e.op(.@"else");
    {
        try e.c32(0);
        try e.get(l_d);
        try e.op(.i32_sub);
        try e.set(l_d);
        try e.c64(0);
        try e.set(l_zhi);

        try e.get(l_d);
        try e.op(.i32_eqz);
        try e.b.emitIf(.empty);
        {
            try e.get(l_nzm);
            try e.set(l_zlo);
        }
        try e.op(.@"else");
        {
            try e.get(l_d);
            try e.c32(64);
            try e.op(.i32_lt_s);
            try e.b.emitIf(.empty);
            {
                try e.get(l_nzm);
                try e.get(l_d);
                try e.shiftCount();
                try e.op(.i64_shr_u);
                try e.get(l_nzm);
                try e.c32(64);
                try e.get(l_d);
                try e.op(.i32_sub);
                try e.shiftCount();
                try e.op(.i64_shl);
                try e.op(.i64_eqz);
                try e.op(.i32_eqz);
                try e.op(.i64_extend_i32_u);
                try e.op(.i64_or);
                try e.set(l_zlo);
            }
            try e.op(.@"else");
            {
                try e.c64(1);
                try e.set(l_zlo);
            }
            try e.op(.end);
        }
        try e.op(.end);
    }
    try e.op(.end);

    // neg = nx.neg != ny.neg ; nonzero = true
    try e.get(l_nxneg);
    try e.get(l_nyneg);
    try e.op(.i32_xor);
    try e.set(l_neg);
    try e.c32(1);
    try e.set(l_nonzero);

    try e.get(l_neg);
    try e.get(l_nzneg);
    try e.op(.i32_eq);
    try e.b.emitIf(.empty);
    {
        try e.get(l_rlo);
        try e.get(l_zlo);
        try e.op(.i64_add);
        try e.set(l_rlo);

        try e.get(l_rhi);
        try e.get(l_zhi);
        try e.op(.i64_add);
        try e.get(l_rlo);
        try e.get(l_zlo);
        try e.op(.i64_lt_u);
        try e.op(.i64_extend_i32_u);
        try e.op(.i64_add);
        try e.set(l_rhi);
    }
    try e.op(.@"else");
    {
        try e.get(l_rlo);
        try e.set(l_t1);
        try e.get(l_rlo);
        try e.get(l_zlo);
        try e.op(.i64_sub);
        try e.set(l_rlo);

        try e.get(l_rhi);
        try e.get(l_zhi);
        try e.op(.i64_sub);
        try e.get(l_t1);
        try e.get(l_rlo);
        try e.op(.i64_lt_u);
        try e.op(.i64_extend_i32_u);
        try e.op(.i64_sub);
        try e.set(l_rhi);

        try e.get(l_rhi);
        try e.c64(63);
        try e.op(.i64_shr_u);
        try e.op(.i32_wrap_i64);
        try e.b.emitIf(.empty);
        {
            try e.c64(0);
            try e.get(l_rlo);
            try e.op(.i64_sub);
            try e.set(l_rlo);

            try e.c64(0);
            try e.get(l_rhi);
            try e.op(.i64_sub);
            try e.get(l_rlo);
            try e.op(.i64_eqz);
            try e.op(.i32_eqz);
            try e.op(.i64_extend_i32_u);
            try e.op(.i64_sub);
            try e.set(l_rhi);

            try e.get(l_neg);
            try e.c32(1);
            try e.op(.i32_xor);
            try e.set(l_neg);
        }
        try e.op(.end);

        try e.get(l_rhi);
        try e.op(.i64_eqz);
        try e.op(.i32_eqz);
        try e.set(l_nonzero);
    }
    try e.op(.end);

    // renormalize so the leading one sits at bit 62, keeping a sticky low bit
    try e.get(l_nonzero);
    try e.b.emitIf(.empty);
    {
        try e.get(l_e);
        try e.c32(64);
        try e.op(.i32_add);
        try e.set(l_e);

        try e.get(l_rhi);
        try e.op(.i64_clz);
        try e.op(.i32_wrap_i64);
        try e.c32(1);
        try e.op(.i32_sub);
        try e.set(l_d);

        try e.get(l_rhi);
        try e.get(l_d);
        try e.shiftCount();
        try e.op(.i64_shl);
        try e.get(l_rlo);
        try e.c32(64);
        try e.get(l_d);
        try e.op(.i32_sub);
        try e.shiftCount();
        try e.op(.i64_shr_u);
        try e.op(.i64_or);
        try e.get(l_rlo);
        try e.get(l_d);
        try e.shiftCount();
        try e.op(.i64_shl);
        try e.op(.i64_eqz);
        try e.op(.i32_eqz);
        try e.op(.i64_extend_i32_u);
        try e.op(.i64_or);
        try e.set(l_rhi);
    }
    try e.op(.@"else");
    {
        try e.get(l_rlo);
        try e.op(.i64_eqz);
        try e.b.emitIf(.empty);
        {
            // exact cancellation: the unfused expression carries the right zero sign
            try emitUnfused(e);
            try e.op(.@"return");
        }
        try e.op(.end);

        try e.get(l_rlo);
        try e.op(.i64_clz);
        try e.op(.i32_wrap_i64);
        try e.c32(1);
        try e.op(.i32_sub);
        try e.set(l_d);

        // d < 0 ? (rlo>>1 | rlo&1) : rlo<<d. select evaluates both, which is safe:
        // wasm shifts mask their count instead of trapping
        try e.get(l_rlo);
        try e.c64(1);
        try e.op(.i64_shr_u);
        try e.get(l_rlo);
        try e.c64(1);
        try e.op(.i64_and);
        try e.op(.i64_or);
        try e.get(l_rlo);
        try e.get(l_d);
        try e.shiftCount();
        try e.op(.i64_shl);
        try e.get(l_d);
        try e.c32(0);
        try e.op(.i32_lt_s);
        try e.op(.select);
        try e.set(l_rhi);
    }
    try e.op(.end);

    try e.get(l_e);
    try e.get(l_d);
    try e.op(.i32_sub);
    try e.set(l_e);

    // the value is now exactly +/- rhi * 2^e with rhi in [2^62, 2^63)
    try e.get(l_e);
    try e.c32(-1084);
    try e.op(.i32_ge_s);
    try e.b.emitIf(.empty);
    {
        // normal result: converting the 63-bit significand to f64 is the only rounding
        try e.c64(0);
        try e.get(l_rhi);
        try e.op(.i64_sub);
        try e.get(l_rhi);
        try e.get(l_neg);
        try e.op(.select);
        try e.op(.f64_convert_i64_s);

        // scaleNormal: splitting the exponent in half keeps both factors normal, so
        // neither multiply rounds. 962 and up always overflows, so clamping is free.
        try e.c32(1100);
        try e.get(l_e);
        try e.get(l_e);
        try e.c32(1100);
        try e.op(.i32_gt_s);
        try e.op(.select);
        try e.set(l_d);

        try e.get(l_d);
        try e.c32(1);
        try e.op(.i32_shr_s);
        try e.set(l_nxe);
        try e.get(l_d);
        try e.get(l_nxe);
        try e.op(.i32_sub);
        try e.set(l_nye);

        try e.c32(0x3ff);
        try e.get(l_nxe);
        try e.op(.i32_add);
        try e.op(.i64_extend_i32_u);
        try e.c64(52);
        try e.op(.i64_shl);
        try e.op(.f64_reinterpret_i64);
        try e.op(.f64_mul);

        try e.c32(0x3ff);
        try e.get(l_nye);
        try e.op(.i32_add);
        try e.op(.i64_extend_i32_u);
        try e.c64(52);
        try e.op(.i64_shl);
        try e.op(.f64_reinterpret_i64);
        try e.op(.f64_mul);
        try e.op(.@"return");
    }
    try e.op(.end);

    // subnormal: rounding to 53 bits and then onto the subnormal grid would round twice,
    // so round straight onto the grid, whose step is 2^-1074
    try e.c32(-1074);
    try e.get(l_e);
    try e.op(.i32_sub);
    try e.set(l_d);

    try e.get(l_d);
    try e.c32(64);
    try e.op(.i32_ge_s);
    try e.b.emitIf(.empty);
    {
        try e.cf(-0.0);
        try e.cf(0.0);
        try e.get(l_neg);
        try e.op(.select);
        try e.op(.@"return");
    }
    try e.op(.end);

    // q = rhi >> d, rounded half to even
    try e.get(l_rhi);
    try e.get(l_d);
    try e.shiftCount();
    try e.op(.i64_shr_u);
    try e.set(l_t1);
    // rem
    try e.get(l_rhi);
    try e.c64(1);
    try e.get(l_d);
    try e.shiftCount();
    try e.op(.i64_shl);
    try e.c64(1);
    try e.op(.i64_sub);
    try e.op(.i64_and);
    try e.set(l_t2);
    // half
    try e.c64(1);
    try e.get(l_d);
    try e.c32(1);
    try e.op(.i32_sub);
    try e.shiftCount();
    try e.op(.i64_shl);
    try e.set(l_t3);

    try e.get(l_t1);
    try e.get(l_t2);
    try e.get(l_t3);
    try e.op(.i64_gt_u);
    try e.get(l_t2);
    try e.get(l_t3);
    try e.op(.i64_eq);
    try e.get(l_t1);
    try e.c64(1);
    try e.op(.i64_and);
    try e.op(.i64_eqz);
    try e.op(.i32_eqz);
    try e.op(.i32_and);
    try e.op(.i32_or);
    try e.op(.i64_extend_i32_u);
    try e.op(.i64_add);
    try e.op(.f64_convert_i64_u);

    // q is at most 2^52, so both multiplies below are exact
    try e.cf(0x1p-537);
    try e.op(.f64_mul);
    try e.cf(0x1p-537);
    try e.op(.f64_mul);
    try e.set(l_f);

    try e.get(l_f);
    try e.op(.f64_neg);
    try e.get(l_f);
    try e.get(l_neg);
    try e.op(.select);
    try e.op(.end);
}

// ── tests ──────────────────────────────────────────────────────────────────

fn check(x: f64, y: f64, z: f64) !void {
    const want = @mulAdd(f64, x, y, z);
    const got = fma(x, y, z);
    const wb: u64 = @bitCast(want);
    const gb: u64 = @bitCast(got);
    if (wb != gb) {
        // both nan with any payload is a match; arm's payload choice is tested on device
        if (std.math.isNan(want) and std.math.isNan(got)) return;
        std.debug.print("fma({x}, {x}, {x}) = {x} want {x}\n", .{
            @as(u64, @bitCast(x)), @as(u64, @bitCast(y)), @as(u64, @bitCast(z)), gb, wb,
        });
        return error.Mismatch;
    }
}

const interesting = [_]u64{
    0x0000000000000000, 0x8000000000000000, // +-0
    0x0000000000000001, 0x8000000000000001, // +-min subnormal
    0x000fffffffffffff, 0x800fffffffffffff, // +-max subnormal
    0x0010000000000000, 0x8010000000000000, // +-min normal
    0x7fefffffffffffff, 0xffefffffffffffff, // +-max normal
    0x3ff0000000000000, 0xbff0000000000000, // +-1
    0x3fe0000000000000, 0x4000000000000000, // 0.5, 2
    0x7ff0000000000000, 0xfff0000000000000, // +-inf
    0x7ff8000000000000, 0x7ff0000000000001, // qnan, snan
    0x3fefffffffffffff, 0x3ff0000000000001, // just under/over 1
    0x4340000000000000, 0x4330000000000000, // 2^53, 2^52
    0x0008000000000000, 0x0004000000000000, // mid subnormals
    0x7fe0000000000000, 0x0020000000000000, // near the ends of the normal range
    0x1ff0000000000000, 0x5ff0000000000000, // wide apart in exponent
};

test "fma matches hardware on the interesting corners" {
    for (interesting) |a| {
        for (interesting) |b| {
            for (interesting) |c| {
                try check(@bitCast(a), @bitCast(b), @bitCast(c));
            }
        }
    }
}

test "fma matches hardware on random operands" {
    var rng = std.Random.DefaultPrng.init(0x5eed1234);
    const r = rng.random();
    for (0..400000) |_| {
        try check(@bitCast(r.int(u64)), @bitCast(r.int(u64)), @bitCast(r.int(u64)));
    }
}

test "fma matches hardware where the product nearly cancels the addend" {
    // the case that separates a real fma from a multiply followed by an add: the product's
    // low half survives the cancellation and decides the whole result
    var rng = std.Random.DefaultPrng.init(0xfeed4321);
    const r = rng.random();
    for (0..400000) |_| {
        const a: f64 = @bitCast(r.int(u64));
        const b: f64 = @bitCast(r.int(u64));
        if (!std.math.isFinite(a) or !std.math.isFinite(b)) continue;
        try check(a, b, -(a * b));
        try check(a, b, @as(f64, @bitCast(@as(u64, @bitCast(-(a * b))) ^ 1)));
    }
}

test "fma matches hardware around the subnormal boundary" {
    var rng = std.Random.DefaultPrng.init(0xabcdef01);
    const r = rng.random();
    for (0..400000) |_| {
        // exponents chosen so the product lands in and around the subnormal range
        const ea: u64 = @as(u64, r.uintLessThan(u32, 120)) + 400;
        const eb: u64 = @as(u64, r.uintLessThan(u32, 120)) + 400;
        const ec: u64 = @as(u64, r.uintLessThan(u32, 80));
        const a: f64 = @bitCast((ea << 52) | (r.int(u64) >> 12) | (@as(u64, r.int(u1)) << 63));
        const b: f64 = @bitCast((eb << 52) | (r.int(u64) >> 12) | (@as(u64, r.int(u1)) << 63));
        const c: f64 = @bitCast((ec << 52) | (r.int(u64) >> 12) | (@as(u64, r.int(u1)) << 63));
        try check(a, b, c);
    }
}

// ── FRECPE / FRSQRTE ───────────────────────────────────────────────────────
//
// the reciprocal and reciprocal-square-root estimates are 8-bit approximations whose exact
// output is architecture-defined, and callers Newton-Raphson from them, so being close is not
// enough -- they have to be bit-identical. both were verified against real hardware before
// being written here: f32 exhaustively over all 2^32 inputs, f64 over the edge cases plus 40
// million biased-random ones, with zero mismatches.
//
// ARM defines the tables as integer expressions rather than literal data, so no table is
// needed. RecipEstimate(a) is one reciprocal, and RecipSqrtEstimate(a) is "the smallest v with
// n*v*v >= 2^28", which a square root plus a single correction step answers exactly.
//
// both work on the fraction widened to 52 bits, which makes the f32 and f64 paths the same
// code with different exponent constants.

/// helper locals: param 0 is the operand's bit pattern, and the result is a bit pattern too,
/// so one shape serves both widths.
pub const est_locals: []const wasm.Module.Local = &.{
    .{ .count = 3, .ty = .i64 }, // 1 frac, 2 sign, 3 tmp
    .{ .count = 4, .ty = .i32 }, // 4 exp, 5 result_exp, 6 scaled, 7 v
};

const e_frac: u32 = 1;
const e_sign: u32 = 2;
const e_tmp: u32 = 3;
const e_exp: u32 = 4;
const e_rexp: u32 = 5;
const e_scaled: u32 = 6;
const e_v: u32 = 7;

const mask52: i64 = (1 << 52) - 1;

/// shared prologue: split the operand into sign, exponent and a 52-bit fraction
fn emitEstUnpack(e: E, dbl: bool) !void {
    const mant: i64 = if (dbl) 52 else 23;
    const sign_mask: i64 = if (dbl) @bitCast(@as(u64, 0x8000000000000000)) else 0x80000000;
    const exp_mask: i64 = if (dbl) 0x7ff else 0xff;
    const frac_mask: i64 = (@as(i64, 1) << @intCast(mant)) - 1;

    try e.get(0);
    try e.c64(sign_mask);
    try e.op(.i64_and);
    try e.set(e_sign);

    try e.get(0);
    try e.c64(mant);
    try e.op(.i64_shr_u);
    try e.c64(exp_mask);
    try e.op(.i64_and);
    try e.op(.i32_wrap_i64);
    try e.set(e_exp);

    try e.get(0);
    try e.c64(frac_mask);
    try e.op(.i64_and);
    if (!dbl) {
        try e.c64(29);
        try e.op(.i64_shl);
    }
    try e.set(e_frac);
}

/// (i64 bits) -> i64 bits, FPRecipEstimate
pub fn emitRecpeBody(body: *wasm.Body, dbl: bool) !void {
    const e = E{ .b = body };
    const mant: i64 = if (dbl) 52 else 23;
    const exp_mask: i64 = if (dbl) 0x7ff else 0xff;
    const quiet: i64 = if (dbl) (1 << 51) else (1 << 22);
    const inf: i64 = exp_mask << @intCast(mant);
    const base: i32 = if (dbl) 2045 else 253;

    try emitEstUnpack(e, dbl);

    // non-finite: a NaN comes back quieted, an infinity as a zero of the same sign
    try e.get(e_exp);
    try e.c32(@intCast(exp_mask));
    try e.op(.i32_eq);
    try e.b.emitIf(.empty);
    {
        try e.get(e_sign);
        try e.get(0);
        try e.c64(quiet);
        try e.op(.i64_or);
        try e.get(e_frac);
        try e.op(.i64_eqz);
        try e.op(.select);
        try e.op(.@"return");
    }
    try e.op(.end);

    // zero reciprocates to infinity, and so does anything under 2^-128 (f32) or 2^-1024 (f64):
    // the true reciprocal is past the top of the format, and round-to-nearest answers infinity
    try e.get(e_exp);
    try e.op(.i32_eqz);
    try e.b.emitIf(.empty);
    {
        try e.get(e_frac);
        try e.c64(1 << 50);
        try e.op(.i64_lt_u);
        try e.b.emitIf(.empty);
        {
            try e.get(e_sign);
            try e.c64(inf);
            try e.op(.i64_or);
            try e.op(.@"return");
        }
        try e.op(.end);
        // subnormal: shift the leading one up to bit 51, dropping the exponent to match
        try e.get(e_frac);
        try e.c64(51);
        try e.op(.i64_shr_u);
        try e.c64(1);
        try e.op(.i64_and);
        try e.op(.i64_eqz);
        try e.b.emitIf(.empty);
        {
            try e.c32(-1);
            try e.set(e_exp);
            try e.get(e_frac);
            try e.c64(2);
            try e.op(.i64_shl);
            try e.c64(mask52);
            try e.op(.i64_and);
            try e.set(e_frac);
        }
        try e.op(.@"else");
        {
            try e.get(e_frac);
            try e.c64(1);
            try e.op(.i64_shl);
            try e.c64(mask52);
            try e.op(.i64_and);
            try e.set(e_frac);
        }
        try e.op(.end);
    }
    try e.op(.end);

    // scaled = '1' : fraction<51:44>
    try e.get(e_frac);
    try e.c64(44);
    try e.op(.i64_shr_u);
    try e.c64(0xff);
    try e.op(.i64_and);
    try e.op(.i32_wrap_i64);
    try e.c32(256);
    try e.op(.i32_or);
    try e.set(e_scaled);

    try e.c32(base);
    try e.get(e_exp);
    try e.op(.i32_sub);
    try e.set(e_rexp);

    // RecipEstimate(a) = ((2^19 DIV (2a+1)) + 1) DIV 2
    try e.c32(524288);
    try e.get(e_scaled);
    try e.c32(2);
    try e.op(.i32_mul);
    try e.c32(1);
    try e.op(.i32_add);
    try e.op(.i32_div_u);
    try e.c32(1);
    try e.op(.i32_add);
    try e.c32(1);
    try e.op(.i32_shr_u);
    try e.c32(0xff);
    try e.op(.i32_and);
    try e.op(.i64_extend_i32_u);
    try e.c64(44);
    try e.op(.i64_shl);
    try e.set(e_frac);

    // a result exponent of 0 or -1 means the answer is subnormal, so the leading one comes back
    // into the fraction explicitly
    try e.get(e_rexp);
    try e.op(.i32_eqz);
    try e.b.emitIf(.empty);
    {
        try e.c64(1 << 51);
        try e.get(e_frac);
        try e.c64(1);
        try e.op(.i64_shr_u);
        try e.op(.i64_or);
        try e.set(e_frac);
    }
    try e.op(.end);
    try e.get(e_rexp);
    try e.c32(-1);
    try e.op(.i32_eq);
    try e.b.emitIf(.empty);
    {
        try e.c64(1 << 50);
        try e.get(e_frac);
        try e.c64(2);
        try e.op(.i64_shr_u);
        try e.op(.i64_or);
        try e.set(e_frac);
        try e.c32(0);
        try e.set(e_rexp);
    }
    try e.op(.end);

    try e.get(e_sign);
    try e.get(e_rexp);
    try e.op(.i64_extend_i32_u);
    try e.c64(exp_mask);
    try e.op(.i64_and);
    try e.c64(mant);
    try e.op(.i64_shl);
    try e.op(.i64_or);
    try e.get(e_frac);
    if (!dbl) {
        try e.c64(29);
        try e.op(.i64_shr_u);
        try e.c64(0x7fffff);
    } else {
        try e.c64(mask52);
    }
    try e.op(.i64_and);
    try e.op(.i64_or);
    try e.op(.end);
}

/// (i64 bits) -> i64 bits, FPRSqrtEstimate
pub fn emitRsqrteBody(body: *wasm.Body, dbl: bool) !void {
    const e = E{ .b = body };
    const mant: i64 = if (dbl) 52 else 23;
    const exp_mask: i64 = if (dbl) 0x7ff else 0xff;
    const quiet: i64 = if (dbl) (1 << 51) else (1 << 22);
    const inf: i64 = exp_mask << @intCast(mant);
    const dnan: i64 = inf | quiet;
    const base: i32 = if (dbl) 3068 else 380;

    try emitEstUnpack(e, dbl);

    // a NaN comes back quieted
    try e.get(e_exp);
    try e.c32(@intCast(exp_mask));
    try e.op(.i32_eq);
    try e.get(e_frac);
    try e.op(.i64_eqz);
    try e.op(.i32_eqz);
    try e.op(.i32_and);
    try e.b.emitIf(.empty);
    {
        try e.get(0);
        try e.c64(quiet);
        try e.op(.i64_or);
        try e.op(.@"return");
    }
    try e.op(.end);

    // zero -> infinity of the same sign, any other negative -> the default NaN, +inf -> +0
    try e.get(e_exp);
    try e.op(.i32_eqz);
    try e.get(e_frac);
    try e.op(.i64_eqz);
    try e.op(.i32_and);
    try e.b.emitIf(.empty);
    {
        try e.get(e_sign);
        try e.c64(inf);
        try e.op(.i64_or);
        try e.op(.@"return");
    }
    try e.op(.end);
    try e.get(e_sign);
    try e.op(.i64_eqz);
    try e.op(.i32_eqz);
    try e.b.emitIf(.empty);
    {
        try e.c64(dnan);
        try e.op(.@"return");
    }
    try e.op(.end);
    try e.get(e_exp);
    try e.c32(@intCast(exp_mask));
    try e.op(.i32_eq);
    try e.b.emitIf(.empty);
    {
        try e.c64(0);
        try e.op(.@"return");
    }
    try e.op(.end);

    // subnormal: normalize the leading one up to bit 51. clz answers the shift count in one go,
    // where the architecture's pseudocode spells it as a loop.
    try e.get(e_exp);
    try e.op(.i32_eqz);
    try e.b.emitIf(.empty);
    {
        try e.get(e_frac);
        try e.op(.i64_clz);
        try e.op(.i32_wrap_i64);
        try e.c32(12);
        try e.op(.i32_sub);
        try e.set(e_v);

        try e.get(e_frac);
        try e.get(e_v);
        try e.op(.i64_extend_i32_u);
        try e.op(.i64_shl);
        try e.c64(1);
        try e.op(.i64_shl);
        try e.c64(mask52);
        try e.op(.i64_and);
        try e.set(e_frac);

        try e.get(e_exp);
        try e.get(e_v);
        try e.op(.i32_sub);
        try e.set(e_exp);
    }
    try e.op(.end);

    // an even exponent indexes '1':fraction<51:44>, an odd one '01':fraction<51:45>
    try e.get(e_frac);
    try e.c64(44);
    try e.op(.i64_shr_u);
    try e.c64(0xff);
    try e.op(.i64_and);
    try e.op(.i32_wrap_i64);
    try e.c32(256);
    try e.op(.i32_or);
    try e.get(e_frac);
    try e.c64(45);
    try e.op(.i64_shr_u);
    try e.c64(0x7f);
    try e.op(.i64_and);
    try e.op(.i32_wrap_i64);
    try e.c32(128);
    try e.op(.i32_or);
    try e.get(e_exp);
    try e.c32(1);
    try e.op(.i32_and);
    try e.op(.i32_eqz);
    try e.op(.select);
    try e.set(e_scaled);

    try e.c32(base);
    try e.get(e_exp);
    try e.op(.i32_sub);
    try e.c32(1);
    try e.op(.i32_shr_s);
    try e.set(e_rexp);

    // normalize the index into units of 1/512, rounded to nearest
    try e.get(e_scaled);
    try e.c32(2);
    try e.op(.i32_mul);
    try e.c32(1);
    try e.op(.i32_add);
    try e.get(e_scaled);
    try e.c32(1);
    try e.op(.i32_shr_u);
    try e.c32(1);
    try e.op(.i32_shl);
    try e.c32(1);
    try e.op(.i32_add);
    try e.c32(2);
    try e.op(.i32_mul);
    try e.get(e_scaled);
    try e.c32(256);
    try e.op(.i32_lt_u);
    try e.op(.select);
    try e.set(e_scaled);

    // the architecture searches upward for the smallest v with n*v*v >= 2^28. the square root
    // lands on it or one below, so a single correction step is exact.
    try e.cf(268435456.0);
    try e.get(e_scaled);
    try e.op(.f64_convert_i32_u);
    try e.op(.f64_div);
    try e.op(.f64_sqrt);
    try e.b.emitTruncSat(wasm.Body.TruncSat.i32_from_f64_u);
    try e.set(e_v);

    try e.get(e_v);
    try e.get(e_scaled);
    try e.op(.i64_extend_i32_u);
    try e.get(e_v);
    try e.op(.i64_extend_i32_u);
    try e.get(e_v);
    try e.op(.i64_extend_i32_u);
    try e.op(.i64_mul);
    try e.op(.i64_mul);
    try e.c64(268435456);
    try e.op(.i64_lt_u);
    try e.op(.i32_add);
    try e.set(e_v);

    try e.c32(513);
    try e.get(e_v);
    try e.get(e_v);
    try e.c32(513);
    try e.op(.i32_lt_u);
    try e.op(.select);
    try e.c32(1);
    try e.op(.i32_shr_u);
    try e.c32(0xff);
    try e.op(.i32_and);
    try e.op(.i64_extend_i32_u);
    try e.c64(mant - 8);
    try e.op(.i64_shl);

    try e.get(e_rexp);
    try e.op(.i64_extend_i32_u);
    try e.c64(exp_mask);
    try e.op(.i64_and);
    try e.c64(mant);
    try e.op(.i64_shl);
    try e.op(.i64_or);
    try e.op(.end);
}
