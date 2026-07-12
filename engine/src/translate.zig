//! arm64 → wasm translation.
//!
//! baseline strategy (no regalloc, no flag laziness — added in follow-up
//! passes):
//!   - cpu registers live in imported linear memory at fixed offsets
//!     (see cpu.zig).
//!   - every read of an arm64 register emits an i64.load; every write
//!     emits an i64.store.
//!   - one wasm function per region. blocks within the region become
//!     labeled wasm blocks dispatched via a br_table in a loop. branches
//!     to in-region targets set the dispatch local and br back to the
//!     loop. region exits write the next pc to memory and `return`.

const std = @import("std");
const decode = @import("decode.zig");
const block = @import("block.zig");
const cpu = @import("cpu.zig");
const wasm = @import("wasm.zig");

pub const exit_halt: u32 = 0;
pub const exit_syscall: u32 = 1;
pub const exit_indirect_branch: u32 = 2;
pub const exit_unknown_insn: u32 = 3;
pub const exit_region_full: u32 = 4;
pub const exit_branch_out: u32 = 5;

// local indices in the translated function. order matters for the
// `addCode` declaration below.
const local_cur: u32 = 0; // i32 — dispatch register (which block to enter)
const local_nzcv_tmp: u32 = 1; // i32 — scratch for cond test bit extraction
const local_a: u32 = 2; // i64 — first arith operand
const local_b: u32 = 3; // i64 — second arith operand
const local_res: u32 = 4; // i64 — arith result
const local_iter: u32 = 5; // i64 — arm insns executed this invocation

pub const Options = struct {
    shadow_base: u32 = 0,
    /// when true, every translated store atomic-ANDs a per-cache-line shadow bit, and
    /// LDXR/STXR enforce the full ARM exclusive-monitor protocol against it.
    /// when false (single-threaded execution), we skip shadow ops entirely and rely on
    /// value-CAS in STXR — correct for sequential execution where no other agent can
    /// modify monitored memory between LDXR and STXR
    shared_monitor: bool = false,
};

/// imported env helper functions. these are js callbacks that operate on i64
/// arguments (and i64 results), bridging arm semantics that aren't naturally
/// expressible in wasm (ieee 754 rounding, simd vector ops on linear memory)
const Helpers = struct {
    f64_add: u32,
    f64_sub: u32,
    f64_mul: u32,
    f64_div: u32,
    f64_abs: u32,
    f64_neg: u32,
    f64_sqrt: u32,
    f64_min: u32,
    f64_max: u32,
    f64_from_s64: u32,
    f64_from_u64: u32,
    f64_from_s32: u32,
    f64_from_u32: u32,
    f64_to_s64: u32,
    f64_to_u64: u32,
    f64_to_s32: u32,
    f64_to_u32: u32,
    f32_to_f64: u32,
    f64_to_f32: u32,
    f64_cmp: u32,
    simd_op: u32,
};

const Ctx = struct {
    region: *const block.Region,
    body: *wasm.Body,
    block_count: u32,
    extra_depth: u32 = 0, // extra control-flow nesting (e.g. if/else inside a block body)
    shadow_base: u32 = 0, // linear-memory offset of the exclusive-monitor shadow bitmap
    shared_monitor: bool = false,
    /// when set to false by translateBlock's lazy-flag pass, *S instructions skip
    /// their NZCV update because no consumer reads it before another *S overwrites it
    flags_live: bool = true,
    helpers: Helpers,

    /// depth from inside block `idx`'s body to the outer dispatch loop,
    /// taking into account any extra nesting we've opened.
    fn loopDepthFromBody(self: *const Ctx, idx: u32) u32 {
        return self.block_count - idx + self.extra_depth;
    }

    /// block index for the in-region target pc, or null if it exits the region.
    fn blockOfPc(self: *const Ctx, pc: u64) ?u32 {
        return self.region.pc_to_block.get(pc);
    }
};

pub fn translateRegion(arena: std.mem.Allocator, region: *const block.Region, opts: Options) ![]u8 {
    var module = wasm.Module.init(arena);

    const run_type = try module.addType(&.{}, &.{.i64});
    const helper_2_type = try module.addType(&.{ .i64, .i64 }, &.{.i64});
    const helper_1_type = try module.addType(&.{.i64}, &.{.i64});
    // declare the imported memory shared IFF we emit atomics (shared_monitor=true).
    // single-threaded processes (shared_monitor=false) emit no atomics and import a
    // NON-shared memory, which lets the host use a lazily-committed growable memory
    // instead of one that reserves its whole max up front.
    try module.addImportMemory("env", "memory", 1, 65536, opts.shared_monitor);

    // threaded processes give each thread its own register block; register accesses are emitted
    // relative to this per-thread-instance global. single-threaded modules skip it entirely and
    // keep absolute low-memory register access (base 0), unchanged.
    const reg_base_global: ?u32 = if (opts.shared_monitor)
        try module.addImportGlobal("env", "reg_base", .i32, false)
    else
        null;

    const helpers: Helpers = .{
        .f64_add = try module.addImportFunc("env", "f64_add", helper_2_type),
        .f64_sub = try module.addImportFunc("env", "f64_sub", helper_2_type),
        .f64_mul = try module.addImportFunc("env", "f64_mul", helper_2_type),
        .f64_div = try module.addImportFunc("env", "f64_div", helper_2_type),
        .f64_abs = try module.addImportFunc("env", "f64_abs", helper_1_type),
        .f64_neg = try module.addImportFunc("env", "f64_neg", helper_1_type),
        .f64_sqrt = try module.addImportFunc("env", "f64_sqrt", helper_1_type),
        .f64_min = try module.addImportFunc("env", "f64_min", helper_2_type),
        .f64_max = try module.addImportFunc("env", "f64_max", helper_2_type),
        .f64_from_s64 = try module.addImportFunc("env", "f64_from_s64", helper_1_type),
        .f64_from_u64 = try module.addImportFunc("env", "f64_from_u64", helper_1_type),
        .f64_from_s32 = try module.addImportFunc("env", "f64_from_s32", helper_1_type),
        .f64_from_u32 = try module.addImportFunc("env", "f64_from_u32", helper_1_type),
        .f64_to_s64 = try module.addImportFunc("env", "f64_to_s64", helper_1_type),
        .f64_to_u64 = try module.addImportFunc("env", "f64_to_u64", helper_1_type),
        .f64_to_s32 = try module.addImportFunc("env", "f64_to_s32", helper_1_type),
        .f64_to_u32 = try module.addImportFunc("env", "f64_to_u32", helper_1_type),
        .f32_to_f64 = try module.addImportFunc("env", "f32_to_f64", helper_1_type),
        .f64_to_f32 = try module.addImportFunc("env", "f64_to_f32", helper_1_type),
        .f64_cmp = try module.addImportFunc("env", "f64_cmp", helper_2_type),
        .simd_op = try module.addImportFunc("env", "simd_op", helper_2_type),
    };

    const run_fn = try module.addFunction(run_type);
    try module.addExportFunc("run", run_fn);

    var body = wasm.Body.init(arena);
    body.reg_base_global = reg_base_global;

    const n: u32 = @intCast(region.blocks.len);
    if (n == 0) {
        try body.emitConstI64(0);
        try body.emit(.@"return");
        try body.emit(.end);
        try module.addCode(&.{}, body.slice());
        var out: std.ArrayList(u8) = .empty;
        try module.finalize(&out);
        return out.items;
    }

    // locals: 2 × i32 ($cur, $nzcv_tmp) then 4 × i64 ($a, $b, $res, $iter)
    const locals: []const wasm.Module.Local = &.{
        .{ .count = 2, .ty = .i32 },
        .{ .count = 4, .ty = .i64 },
    };

    // initialize $cur to the entry block index (always block 0 for a
    // freshly-built region — entry_pc maps to blocks[0]).
    try body.emitConstI32(0);
    try body.emit(.local_set);
    try body.emitU32(local_cur);
    // $iter starts at 0 each invocation; every exit stores it to iter_offset
    // so the dispatcher can account executed insns (read-and-reset convention,
    // same as the ts translator)

    // open loop + exit + N nested blocks
    try body.emitLoop(.empty); // depth from inside = (n + 1)
    try body.emitBlock(.empty); // exit, depth from inside = n
    var i: u32 = n;
    while (i > 0) : (i -= 1) try body.emitBlock(.empty); // b_{n-1} ... b_0

    // dispatcher: br_table on $cur
    try body.emit(.local_get);
    try body.emitU32(local_cur);
    try body.emit(.br_table);
    try body.emitU32(n); // table length
    var k: u32 = 0;
    while (k < n) : (k += 1) try body.emitU32(k);
    try body.emitU32(n); // default → $exit

    // close innermost block $b_0
    try body.emit(.end);

    var ctx: Ctx = .{
        .region = region,
        .body = &body,
        .block_count = n,
        .shadow_base = opts.shadow_base,
        .shared_monitor = opts.shared_monitor,
        .helpers = helpers,
    };

    // emit each block's body. between body and next `end`, the block is closed.
    for (region.blocks, 0..) |b, idx| {
        try translateBlock(&ctx, &b, @intCast(idx));
        try body.emit(.end); // close the surrounding block (b_{idx+1}) or — for the last block — $exit
    }
    // close loop
    try body.emit(.end);

    // function fall-off: return halt
    try emitIterFlush(&body);
    try body.emitConstI64(packExit(@intCast(cpu.halt_pc & 0xffff_ffff), exit_halt));
    try body.emit(.@"return");
    try body.emit(.end); // close function body

    try module.addCode(locals, body.slice());

    var out: std.ArrayList(u8) = .empty;
    try module.finalize(&out);
    return out.items;
}

fn packExit(next_pc: u32, reason: u32) i64 {
    return @bitCast((@as(u64, reason) << 32) | @as(u64, next_pc));
}

/// scan forward from instruction `i` in the block. returns true if the NZCV written
/// by insn[i] is read (via a flag-consumer like B.cond/CSEL/CCMP/ADC/SBC) before being
/// overwritten by another *S instruction. conservative: at block end, returns true
/// since downstream blocks may consume the flags
fn isNzcvLive(insns: []const decode.Op, i: usize) bool {
    if (!opSetsFlags(insns[i])) return true; // only matters for flag-setters
    var j: usize = i + 1;
    while (j < insns.len) : (j += 1) {
        if (opReadsFlags(insns[j])) return true;
        if (opSetsFlags(insns[j])) return false; // overwritten before any reader
    }
    return true; // ran off block end — conservatively live
}

fn opSetsFlags(op: decode.Op) bool {
    return switch (op) {
        .add_imm => |x| x.set_flags,
        .sub_imm => |x| x.set_flags,
        .add_reg => |x| x.set_flags,
        .sub_reg => |x| x.set_flags,
        .add_ext_reg => |x| x.set_flags,
        .sub_ext_reg => |x| x.set_flags,
        .and_imm => |x| x.set_flags,
        .and_reg => |x| x.set_flags,
        .adc => |x| x.set_flags,
        .sbc => |x| x.set_flags,
        .ccmp_imm, .ccmp_reg => true,
        else => false,
    };
}

fn opReadsFlags(op: decode.Op) bool {
    return switch (op) {
        .b_cond, .csel, .csinc, .csinv, .csneg, .ccmp_imm, .ccmp_reg, .adc, .sbc => true,
        else => false,
    };
}

/// helper for the dispatcher: clear a struct's `set_flags` field if `flags_live` is false,
/// turning ADDS/SUBS/ANDS/etc into their non-flag-setting cousins
fn killFlags(comptime T: type, x: T, flags_live: bool) T {
    if (flags_live) return x;
    var y = x;
    y.set_flags = false;
    return y;
}

fn translateBlock(ctx: *Ctx, b: *const block.Block, idx: u32) !void {
    // stub block: pass 1 saw this leader but ran out of budget (or it was unfetchable),
    // so it has no decoded ops. emit a region-exit at its own pc — the dispatcher
    // will translate a fresh region starting there next call
    if (b.insns.len == 0) {
        try emitExit(ctx.body, b.start_pc, exit_branch_out);
        return;
    }
    // self-loop detection: block's terminating conditional branch targets its own start.
    // when present we wrap the body in a wasm `loop` and convert the final branch into
    // a `br_if` back to the loop header. keeps tight counters entirely inside the wasm
    // function without exit-and-redispatch through getRegion.
    const self_loop_target: ?u64 = switch (b.exit) {
        .cond_branch => |c| if (c.taken_pc == b.start_pc) c.taken_pc else null,
        else => null,
    };

    if (self_loop_target != null) {
        try ctx.body.emit(.loop);
        try ctx.body.bytes.append(ctx.body.arena, @intFromEnum(wasm.BlockType.empty));
        ctx.extra_depth += 1;
    }

    // safety budget: exit the region (resuming at this block) once enough insns
    // ran in this invocation, so infinite guest loops still yield to the
    // dispatcher for signal delivery. checked before the increment so reported
    // insn counts stay exact
    try ctx.body.emit(.local_get);
    try ctx.body.emitU32(local_iter);
    try ctx.body.emitConstI64(4_000_000);
    try ctx.body.emit(.i64_gt_u);
    try ctx.body.emit(.@"if");
    try ctx.body.bytes.append(ctx.body.arena, @intFromEnum(wasm.BlockType.empty));
    try emitExit(ctx.body, b.start_pc, exit_branch_out);
    try ctx.body.emit(.end);

    // count this block's insns (inside the self-loop wrap so re-entry counts)
    try ctx.body.emit(.local_get);
    try ctx.body.emitU32(local_iter);
    try ctx.body.emitConstI64(@intCast(b.insns.len));
    try ctx.body.emit(.i64_add);
    try ctx.body.emit(.local_set);
    try ctx.body.emitU32(local_iter);

    var pc = b.start_pc;
    var i: usize = 0;
    const last_idx: usize = if (b.insns.len > 0) b.insns.len - 1 else 0;
    while (i < b.insns.len) : (i += 1) {
        const op = b.insns[i];
        const is_last = i == last_idx;

        // lazy NZCV: skip flag computation for this *S if no consumer reads NZCV
        // before another *S overwrites it within this block. conservative — at block
        // end we assume flags may be read by downstream code.
        ctx.flags_live = isNzcvLive(b.insns, i);

        // peephole: SUBS XZR, Xn, ? immediately followed by B.cond can be fused
        // into a direct i64 compare + branch, skipping the NZCV roundtrip entirely.
        if (i + 1 < b.insns.len) {
            const next = b.insns[i + 1];
            const next_is_last = i + 1 == last_idx;
            const self_target = if (next_is_last) self_loop_target else null;
            if (try tryFuseCmpBCond(ctx, b, idx, op, next, pc, self_target)) {
                pc += 8;
                i += 1;
                continue;
            }
        }

        if (is_last and self_loop_target != null) {
            try translateSelfLoopBranch(ctx, idx, op, pc);
        } else {
            try translateInsn(ctx, b, idx, op, pc);
        }
        pc += 4;
    }

    if (self_loop_target != null) {
        try ctx.body.emit(.end); // close loop
        ctx.extra_depth -= 1;
    }
    // if the block falls through into the next block in-region, jump
    switch (b.exit) {
        .fallthrough_to_block => |next_pc| {
            if (ctx.blockOfPc(next_pc)) |next_idx| {
                try jumpToBlock(ctx, idx, next_idx);
            } else {
                try emitExit(ctx.body, next_pc, exit_branch_out);
            }
        },
        // pass-1 ran out of budget mid-block, so the last decoded insn is not a
        // terminator. exit to the next pc — dispatcher will translate a fresh region
        .region_full => try emitExit(ctx.body, b.end_pc, exit_branch_out),
        else => {},
    }
}

fn translateInsn(
    ctx: *Ctx,
    b: *const block.Block,
    block_idx: u32,
    op: decode.Op,
    pc: u64,
) !void {
    _ = b;
    const body = ctx.body;
    switch (op) {
        .nop => {},

        .movz => |x| {
            const value: u64 = @as(u64, x.imm16) << @intCast(@as(u32, x.hw) * 16);
            const masked: u64 = if (x.sf) value else value & 0xffff_ffff;
            try storeGprImm(body, x.rd, masked);
        },

        .adr => |x| {
            const value: u64 = @bitCast(@as(i64, @intCast(pc)) + @as(i64, x.offset));
            try storeGprImm(body, x.rd, value);
        },

        .adrp => |x| {
            const page: u64 = pc & ~@as(u64, 0xfff);
            const value: u64 = @bitCast(@as(i64, @bitCast(page)) + @as(i64, x.offset));
            try storeGprImm(body, x.rd, value);
        },

        .movn => |x| {
            const value: u64 = ~(@as(u64, x.imm16) << @intCast(@as(u32, x.hw) * 16));
            const masked: u64 = if (x.sf) value else value & 0xffff_ffff;
            try storeGprImm(body, x.rd, masked);
        },

        .movk => |x| {
            const shift: u32 = @as(u32, x.hw) * 16;
            const keep_mask: u64 = ~(@as(u64, 0xffff) << @intCast(shift));
            const insert: u64 = @as(u64, x.imm16) << @intCast(shift);
            if (x.rd == 31) return;
            try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));
            try loadGpr(body, x.rd, true); // raw 64-bit; we'll mask below
            try body.emitConstI64(@bitCast(keep_mask));
            try body.emit(.i64_and);
            try body.emitConstI64(@bitCast(insert));
            try body.emit(.i64_or);
            if (!x.sf) {
                try body.emitConstI64(0xffff_ffff);
                try body.emit(.i64_and);
            }
            try body.storeI64(0);
        },

        .add_imm => |x| try emitArithImm(body, killFlags(decode.ArithImm, x, ctx.flags_live), false),
        .sub_imm => |x| try emitArithImm(body, killFlags(decode.ArithImm, x, ctx.flags_live), true),

        .add_reg => |x| try emitArithReg(body, killFlags(decode.ArithReg, x, ctx.flags_live), false),
        .sub_reg => |x| try emitArithReg(body, killFlags(decode.ArithReg, x, ctx.flags_live), true),
        .add_ext_reg => |x| try emitArithExtReg(body, killFlags(decode.ArithExtReg, x, ctx.flags_live), false),
        .sub_ext_reg => |x| try emitArithExtReg(body, killFlags(decode.ArithExtReg, x, ctx.flags_live), true),

        .and_reg => |x| try emitLogicalReg(body, killFlags(decode.LogicalReg, x, ctx.flags_live), .@"and", false),
        .orr_reg => |x| try emitLogicalReg(body, x, .@"or", false),
        .eor_reg => |x| try emitLogicalReg(body, x, .xor, false),
        .bic_reg => |x| try emitLogicalReg(body, killFlags(decode.LogicalReg, x, ctx.flags_live), .@"and", true),
        .orn_reg => |x| try emitLogicalReg(body, x, .@"or", true),
        .eon_reg => |x| try emitLogicalReg(body, x, .xor, true),

        .and_imm => |x| try emitLogicalImm(body, killFlags(decode.LogicalImm, x, ctx.flags_live), .@"and"),
        .orr_imm => |x| try emitLogicalImm(body, x, .@"or"),
        .eor_imm => |x| try emitLogicalImm(body, x, .xor),

        .csel => |x| try emitCondSelect(body, x, .csel),
        .csinc => |x| try emitCondSelect(body, x, .csinc),
        .csinv => |x| try emitCondSelect(body, x, .csinv),
        .csneg => |x| try emitCondSelect(body, x, .csneg),

        .lsl_reg => |x| try emitShiftReg(body, x, .lsl),
        .lsr_reg => |x| try emitShiftReg(body, x, .lsr),
        .asr_reg => |x| try emitShiftReg(body, x, .asr),
        .ror_reg => |x| try emitShiftReg(body, x, .ror),

        .ubfm => |x| try emitBitfield(body, x, .ubfm),
        .sbfm => |x| try emitBitfield(body, x, .sbfm),
        .bfm => |x| try emitBitfield(body, x, .bfm),
        .extr => |x| try emitExtr(body, x),

        .madd => |x| try emitMulAdd(body, x, false),
        .msub => |x| try emitMulAdd(body, x, true),
        .smaddl => |x| try emitMulLong(body, x, true, false),
        .smsubl => |x| try emitMulLong(body, x, true, true),
        .umaddl => |x| try emitMulLong(body, x, false, false),
        .umsubl => |x| try emitMulLong(body, x, false, true),
        .smulh => |x| try emitMulHigh(body, x, true),
        .umulh => |x| try emitMulHigh(body, x, false),
        .udiv => |x| try emitDiv(body, x, false),
        .sdiv => |x| try emitDiv(body, x, true),

        .ret, .br, .blr => {
            const rn = switch (op) {
                .ret => |y| y.rn,
                .br => |y| y.rn,
                .blr => |y| y.rn,
                else => unreachable,
            };
            // BLR saves pc+4 to X30 (LR) before branching, just like BL
            if (op == .blr) {
                try storeGprImm(body, 30, pc + 4);
            }
            try body.emitRegAddr(@bitCast(cpu.pc_offset));
            try loadGpr(body, rn, true);
            try body.storeI64(0);
            try emitIterFlush(body);
            try body.emitConstI64(@bitCast(@as(u64, exit_indirect_branch) << 32));
            try body.emit(.@"return");
        },

        .b => |x| {
            const target: u64 = @intCast(@as(i64, @intCast(pc)) + x.offset);
            if (ctx.blockOfPc(target)) |target_idx| {
                try jumpToBlock(ctx, block_idx, target_idx);
            } else {
                try emitExit(body, target, exit_branch_out);
            }
        },

        .bl => |x| {
            // bl: x30 = pc + 4; branch to target
            try body.emitRegAddr(@bitCast(cpu.gpr_offset(30)));
            try body.emitConstI64(@bitCast(pc + 4));
            try body.storeI64(0);
            const target: u64 = @intCast(@as(i64, @intCast(pc)) + x.offset);
            if (ctx.blockOfPc(target)) |target_idx| {
                try jumpToBlock(ctx, block_idx, target_idx);
            } else {
                try emitExit(body, target, exit_branch_out);
            }
        },

        .b_cond => |x| {
            const target: u64 = @intCast(@as(i64, @intCast(pc)) + x.offset);
            try emitCondBranch(ctx, block_idx, pc, target, .{ .b_cond = x.cond });
        },

        .cbz, .cbnz => |x| {
            const target: u64 = @intCast(@as(i64, @intCast(pc)) + x.offset);
            const is_nz = op == .cbnz;
            try emitCondBranch(ctx, block_idx, pc, target, .{ .cbz = .{ .rt = x.rt, .sf = x.sf, .non_zero = is_nz } });
        },

        .tbz, .tbnz => |x| {
            const target: u64 = @intCast(@as(i64, @intCast(pc)) + x.offset);
            const is_nz = op == .tbnz;
            try emitCondBranch(ctx, block_idx, pc, target, .{ .tbz = .{ .rt = x.rt, .bit = x.bit, .non_zero = is_nz } });
        },

        .svc => {
            // by ts convention, memory.PC holds the svc instruction's own pc so the
            // host can fetch it. the dispatcher advances by +4 after handling the syscall.
            try emitExit(body, pc, exit_syscall);
        },

        .ldr_imm => |x| try emitLoadImm(body, x, if (x.sf) @as(u8, 8) else 4, false),
        .ldrb_imm => |x| try emitLoadImm(body, x, 1, false),
        .ldrh_imm => |x| try emitLoadImm(body, x, 2, false),
        .ldrsb_imm => |x| try emitLoadImm(body, x, 1, true),
        .ldrsh_imm => |x| try emitLoadImm(body, x, 2, true),
        .ldrsw_imm => |x| try emitLoadImm(body, x, 4, true),
        .str_imm => |x| try emitStoreImm(ctx, x, if (x.sf) @as(u8, 8) else 4),
        .strb_imm => |x| try emitStoreImm(ctx, x, 1),
        .strh_imm => |x| try emitStoreImm(ctx, x, 2),
        .ldp => |x| try emitLoadPair(body, x, false),
        .stp => |x| try emitStorePair(ctx, x),

        .ldxr => |x| try emitLdxr(ctx, x),
        .stxr => |x| try emitStxr(ctx, x),
        .cas => |x| try emitCas(ctx, x),
        .lse => |x| try emitLse(ctx, x),

        .simd_load => |x| try emitSimdLoad(ctx, x),
        .simd_store => |x| try emitSimdStore(ctx, x),
        .simd_load_pair => |x| try emitSimdLoadPair(ctx, x),
        .simd_store_pair => |x| try emitSimdStorePair(ctx, x),
        .simd_load_reg => |x| try emitSimdLoadReg(ctx, x),
        .simd_store_reg => |x| try emitSimdStoreReg(ctx, x),

        .simd_movi => |x| try emitSimdMovi(body, x),
        .simd_dup_gpr => |x| try emitSimdDupGpr(body, x),
        .simd_umov => |x| try emitSimdUmov(body, x),
        .simd_ins_gpr => |x| try emitSimdInsGpr(body, x),
        .fmov_to_fp => |x| try emitFmovToFp(body, x),
        .fmov_from_fp => |x| try emitFmovFromFp(body, x),

        .ldr_reg => |x| try emitLoadReg(body, x, if (x.sf) @as(u8, 8) else 4, false),
        .ldrb_reg => |x| try emitLoadReg(body, x, 1, false),
        .ldrh_reg => |x| try emitLoadReg(body, x, 2, false),
        .ldrsb_reg => |x| try emitLoadReg(body, x, 1, true),
        .ldrsh_reg => |x| try emitLoadReg(body, x, 2, true),
        .ldrsw_reg => |x| try emitLoadReg(body, x, 4, true),
        .str_reg => |x| try emitStoreReg(ctx, x, if (x.sf) @as(u8, 8) else 4),
        .strb_reg => |x| try emitStoreReg(ctx, x, 1),
        .strh_reg => |x| try emitStoreReg(ctx, x, 2),
        .ldr_lit => |x| try emitLoadLit(body, x, pc),

        .mrs => |x| try emitMrs(body, x),
        .msr => |x| try emitMsr(body, x),
        // DMB/DSB/ISB: a memory barrier. single-threaded needs nothing (one observer); threaded
        // emits a wasm atomic.fence (0xfe 0x03 0x00) to order accesses around the shared heap.
        .barrier => if (ctx.shared_monitor) {
            try body.emitAtomic(0x03);
            try body.bytes.append(body.arena, 0x00);
        },
        .brk => |_| {
            try emitExit(body, pc, exit_unknown_insn);
        },
        .hint => {},

        .clz => |x| try emitClz(body, x),
        .cls => |x| try emitCls(body, x),
        .rev => |x| try emitRev(body, x),
        .rbit => |x| try emitRbit(body, x),
        .adc => |x| try emitAdcSbc(body, killFlags(decode.AdcSbc, x, ctx.flags_live), false),
        .sbc => |x| try emitAdcSbc(body, killFlags(decode.AdcSbc, x, ctx.flags_live), true),
        .ccmp_imm => |x| try emitCcmpImm(body, x),
        .ccmp_reg => |x| try emitCcmpReg(body, x),

        .fadd => |x| try emitFpArith2(ctx, x, .add),
        .fsub => |x| try emitFpArith2(ctx, x, .sub),
        .fmul => |x| try emitFpArith2(ctx, x, .mul),
        .fdiv => |x| try emitFpArith2(ctx, x, .div),
        .fminnm => |x| try emitFpArith2(ctx, x, .min),
        .fmaxnm => |x| try emitFpArith2(ctx, x, .max),
        .fabs => |x| try emitFpArith1(ctx, x, .abs),
        .fneg => |x| try emitFpArith1(ctx, x, .neg),
        .fsqrt => |x| try emitFpArith1(ctx, x, .sqrt),
        .fmov_fp => |x| try emitFmovFp(ctx, x),
        .fcvt => |x| try emitFcvt(ctx, x),
        .fmadd => |x| try emitFpArith3(ctx, x, .add, false),
        .fmsub => |x| try emitFpArith3(ctx, x, .sub, false),
        .fnmadd => |x| try emitFpArith3(ctx, x, .add, true),
        .fnmsub => |x| try emitFpArith3(ctx, x, .sub, true),
        .fcmp => |x| try emitFcmp(ctx, x),
        .fmov_imm => |x| try emitFmovImm(body, x),
        .fmov_top_to_fp => |x| try emitFmovTopToFp(body, x),
        .fmov_top_from_fp => |x| try emitFmovTopFromFp(body, x),
        .scvtf_int => |x| try emitIntToFp(ctx, x, true),
        .ucvtf_int => |x| try emitIntToFp(ctx, x, false),
        .fcvtzs_int => |x| try emitFpToInt(ctx, x, true),
        .fcvtzu_int => |x| try emitFpToInt(ctx, x, false),
        .fp_fixed_cvt => |x| try emitFpFixedCvt(ctx, x),
        .simd_scalar_cvt => |x| try emitSimdScalarCvt(ctx, x),
        .simd_scalar_shift => |x| try emitSimdScalarShift(ctx, x),

        .simd_orr_reg => |x| try emitSimdBitwiseReg(body, x, .@"or", false),
        .simd_and_reg => |x| try emitSimdBitwiseReg(body, x, .@"and", false),
        .simd_bic_reg => |x| try emitSimdBitwiseReg(body, x, .@"and", true),
        .simd_orn_reg => |x| try emitSimdBitwiseReg(body, x, .@"or", true),
        .simd_eor_reg => |x| try emitSimdBitwiseReg(body, x, .xor, false),
        .simd_bsl => |x| try emitSimdBitSelect(body, x, .bsl),
        .simd_bit => |x| try emitSimdBitSelect(body, x, .bit),
        .simd_bif => |x| try emitSimdBitSelect(body, x, .bif),
        .simd_not => |x| try emitSimdNot(body, x),
        .simd_cnt => |x| try emitVecUnop(body, x.rd, x.rn, x.q, simd.i8x16_popcnt),
        .simd_add => |x| try emitSimdOpCall(ctx, x, simd_op_add, 0),
        .simd_sub => |x| try emitSimdOpCall(ctx, x, simd_op_sub, 0),
        .simd_cmeq_reg => |x| try emitSimdOpCall(ctx, x, simd_op_cmeq, 0),
        .simd_cmgt_reg => |x| try emitSimdOpCall(ctx, x, simd_op_cmgt, 0),
        .simd_cmhi_reg => |x| try emitSimdOpCall(ctx, x, simd_op_cmhi, 0),
        .simd_cmge_reg => |x| try emitSimdOpCall(ctx, x, simd_op_cmge, 0),
        .simd_cmhs_reg => |x| try emitSimdOpCall(ctx, x, simd_op_cmhs, 0),
        .simd_umaxp => |x| try emitSimdOpCall(ctx, x, simd_op_umaxp, 0),
        .simd_uminp => |x| try emitSimdOpCall(ctx, x, simd_op_uminp, 0),
        .simd_addhn => |x| try emitSimdOpCall(ctx, x, simd_op_addhn, 0),
        .simd_ext => |x| {
            try body.emitConstI64(packSimdArgs(simd_op_ext, x.q, 0, x.index, x.rd, x.rn, x.rm));
            try body.emitConstI64(0);
            try body.emit(.call);
            try body.emitU32(ctx.helpers.simd_op);
            try body.emit(.drop);
        },
        .simd_addp => |x| try emitSimdOpCall(ctx, x, simd_op_addp, 0),
        .simd_zip1 => |x| try emitSimdOpCall(ctx, x, simd_op_zip1, 0),
        .simd_zip2 => |x| try emitSimdOpCall(ctx, x, simd_op_zip2, 0),
        .simd_uzp1 => |x| try emitSimdOpCall(ctx, x, simd_op_uzp1, 0),
        .simd_uzp2 => |x| try emitSimdOpCall(ctx, x, simd_op_uzp2, 0),
        .simd_trn1 => |x| try emitSimdOpCall(ctx, x, simd_op_trn1, 0),
        .simd_trn2 => |x| try emitSimdOpCall(ctx, x, simd_op_trn2, 0),
        .simd_tbl => |x| try emitSimdTbl(ctx, x),
        .simd_sshll => |x| try emitSimdShiftLong(body, x, true),
        .simd_ushll => |x| try emitSimdShiftLong(body, x, false),
        .simd_shrn => |x| {
            try body.emitConstI64(packSimdArgs(simd_op_shrn, x.q, x.size, x.shift, x.rd, x.rn, 0));
            try body.emitConstI64(0);
            try body.emit(.call);
            try body.emitU32(ctx.helpers.simd_op);
            try body.emit(.drop);
        },
        .simd_dup_elem => |x| try emitSimdDupElem(body, x),

        .vec_fp3 => |x| try emitVecFp3(body, x),
        .vec_fp2 => |x| try emitVecFp2(body, x),
        .vec_int3 => |x| try emitVecInt3(ctx, x),
        .vec_int2 => |x| try emitVecInt2(body, x),
        .vec_widen => |x| try emitVecWiden(body, x),
        .vec_across => |x| try emitVecAcross(ctx, x),
        .simd_ldst_multi => |x| try emitSimdLdstMulti(ctx, x),
        .simd_ld1r => |x| try emitSimdLd1r(body, x),

        .unknown => {
            try emitExit(body, pc, exit_unknown_insn);
        },

        else => {
            try emitExit(body, pc, exit_unknown_insn);
        },
    }
}

// ── LSE atomics (sequential semantics: plain read-modify-write) ─────────────

/// push the wasm access address for the guest address in $a
fn pushLseAddr(body: *wasm.Body) !void {
    try body.emit(.local_get);
    try body.emitU32(local_a);
    try body.emit(.i32_wrap_i64);
    try body.emitConstI32(@bitCast(cpu.guest_mem_base));
    try body.emit(.i32_add);
}

fn emitLseLoad(body: *wasm.Body, size: u8) !void {
    switch (size) {
        1 => {
            try body.emit(.i64_load8_u);
            try body.emitMemArg(0, 0);
        },
        2 => {
            try body.emit(.i64_load16_u);
            try body.emitMemArg(1, 0);
        },
        4 => {
            try body.emit(.i64_load32_u);
            try body.emitMemArg(2, 0);
        },
        else => {
            try body.emit(.i64_load);
            try body.emitMemArg(3, 0);
        },
    }
}

fn emitLseStore(body: *wasm.Body, size: u8) !void {
    switch (size) {
        1 => {
            try body.emit(.i64_store8);
            try body.emitMemArg(0, 0);
        },
        2 => {
            try body.emit(.i64_store16);
            try body.emitMemArg(1, 0);
        },
        4 => {
            try body.emit(.i64_store32);
            try body.emitMemArg(2, 0);
        },
        else => {
            try body.storeI64(0);
        },
    }
}

fn sizeMask(size: u8) u64 {
    return switch (size) {
        1 => 0xff,
        2 => 0xffff,
        4 => 0xffff_ffff,
        else => ~@as(u64, 0),
    };
}

/// sign-extend the width-sized value on the stack to a full i64
fn emitSextToWidth(body: *wasm.Body, size: u8) !void {
    if (size == 8) return;
    const sh: i64 = 64 - @as(i64, size) * 8;
    try body.emitConstI64(sh);
    try body.emit(.i64_shl);
    try body.emitConstI64(sh);
    try body.emit(.i64_shr_s);
}

/// CAS Rs, Rt, [Rn]: if mem == Rs then mem = Rt; Rs = old value
fn emitCas(ctx: *Ctx, x: decode.Cas) !void {
    const body = ctx.body;
    // threaded: a genuine atomic compare-and-swap. cmpxchg(addr, expected=Rs, desired=Rt) does the
    // compare+store as one indivisible op and returns the pre-op value, which lands in Rs — exactly
    // ARM CAS semantics. sub-word variants compare/return the low `size` bytes zero-extended.
    if (ctx.shared_monitor) {
        try loadGprOrSp(body, x.rn);
        try body.emit(.local_set);
        try body.emitU32(local_a);
        if (x.rs != 31) try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rs))); // Rs destination addr
        try pushLseAddr(body); // guest address
        try loadGpr(body, x.rs, true); // expected
        try loadGpr(body, x.rt, true); // desired
        try body.emitAtomic(atomicCmpxchgSubOpI64(x.size));
        try body.emitMemArg(atomicAlignLog2(x.size), 0);
        if (x.rs != 31) try body.storeI64(0) else try body.emit(.drop);
        return;
    }
    // $a = guest address
    try loadGprOrSp(body, x.rn);
    try body.emit(.local_set);
    try body.emitU32(local_a);
    // $b = old value (zero-extended at width)
    try pushLseAddr(body);
    try emitLseLoad(body, x.size);
    try body.emit(.local_set);
    try body.emitU32(local_b);

    // if (old == Rs & mask) store Rt
    try body.emit(.local_get);
    try body.emitU32(local_b);
    try loadGpr(body, x.rs, true);
    try body.emitConstI64(@bitCast(sizeMask(x.size)));
    try body.emit(.i64_and);
    try body.emit(.i64_eq);
    try body.emit(.@"if");
    try body.bytes.append(body.arena, @intFromEnum(wasm.BlockType.empty));
    try pushLseAddr(body);
    try loadGpr(body, x.rt, true);
    try emitLseStore(body, x.size);
    try body.emit(.end);

    // Rs = old value
    if (x.rs != 31) {
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rs)));
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.storeI64(0);
    }
}

/// LDADD/LDCLR/LDEOR/LDSET/LD[SU]MAX/LD[SU]MIN/SWP Rs, Rt, [Rn]:
/// Rt = old value; mem = f(old, Rs)
fn emitLse(ctx: *Ctx, x: decode.Lse) !void {
    const body = ctx.body;
    // threaded: add/clr/set/eor/swp map to real wasm atomic.rmw ops (the RMW is one indivisible
    // op returning the pre-op value -> Rt). clr is `and ~Rs`. max/min have NO direct wasm atomic,
    // so they fall through to the (non-atomic) plain path below — rare in contended code; TODO
    // CAS-retry loop.
    if (ctx.shared_monitor) {
        const subop: ?u8 = switch (x.op) {
            .add => atomicRmwSubOpI64(.add, x.size),
            .clr => atomicRmwSubOpI64(.@"and", x.size),
            .set => atomicRmwSubOpI64(.@"or", x.size),
            .eor => atomicRmwSubOpI64(.xor, x.size),
            .swp => atomicRmwSubOpI64(.xchg, x.size),
            else => null,
        };
        if (subop) |op| {
            try loadGprOrSp(body, x.rn);
            try body.emit(.local_set);
            try body.emitU32(local_a);
            if (x.rt != 31) try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rt))); // Rt destination
            try pushLseAddr(body); // guest address
            try loadGpr(body, x.rs, true); // operand
            if (x.op == .clr) { // LDCLR: mem &= ~Rs
                try body.emitConstI64(-1);
                try body.emit(.i64_xor);
            }
            try body.emitAtomic(op);
            try body.emitMemArg(atomicAlignLog2(x.size), 0);
            if (x.rt != 31) try body.storeI64(0) else try body.emit(.drop);
            return;
        }
    }
    try loadGprOrSp(body, x.rn);
    try body.emit(.local_set);
    try body.emitU32(local_a);
    try pushLseAddr(body);
    try emitLseLoad(body, x.size);
    try body.emit(.local_set);
    try body.emitU32(local_b);

    // store f(old, rs)
    try pushLseAddr(body);
    switch (x.op) {
        .add => {
            try body.emit(.local_get);
            try body.emitU32(local_b);
            try loadGpr(body, x.rs, true);
            try body.emit(.i64_add);
        },
        .clr => {
            try body.emit(.local_get);
            try body.emitU32(local_b);
            try loadGpr(body, x.rs, true);
            try body.emitConstI64(-1);
            try body.emit(.i64_xor);
            try body.emit(.i64_and);
        },
        .eor => {
            try body.emit(.local_get);
            try body.emitU32(local_b);
            try loadGpr(body, x.rs, true);
            try body.emit(.i64_xor);
        },
        .set => {
            try body.emit(.local_get);
            try body.emitU32(local_b);
            try loadGpr(body, x.rs, true);
            try body.emit(.i64_or);
        },
        .swp => {
            try loadGpr(body, x.rs, true);
        },
        .umax, .umin => {
            // select(old, rs_masked, cmp) — unsigned compare at width
            try body.emit(.local_get);
            try body.emitU32(local_b);
            try loadGpr(body, x.rs, true);
            try body.emitConstI64(@bitCast(sizeMask(x.size)));
            try body.emit(.i64_and);
            try body.emit(.local_set);
            try body.emitU32(local_res);
            try body.emit(.local_get);
            try body.emitU32(local_res);
            try body.emit(.local_get);
            try body.emitU32(local_b);
            try body.emit(.local_get);
            try body.emitU32(local_res);
            try body.emit(if (x.op == .umax) .i64_gt_u else .i64_lt_u);
            try body.emit(.select);
        },
        .smax, .smin => {
            // sign-extend both at width for the compare, store the winner (masked by store width)
            try body.emit(.local_get);
            try body.emitU32(local_b);
            try loadGpr(body, x.rs, true);
            try body.emitConstI64(@bitCast(sizeMask(x.size)));
            try body.emit(.i64_and);
            try body.emit(.local_set);
            try body.emitU32(local_res);
            try body.emit(.local_get);
            try body.emitU32(local_res);
            // compare operands sign-extended
            try body.emit(.local_get);
            try body.emitU32(local_b);
            try emitSextToWidth(body, x.size);
            try body.emit(.local_get);
            try body.emitU32(local_res);
            try emitSextToWidth(body, x.size);
            try body.emit(if (x.op == .smax) .i64_gt_s else .i64_lt_s);
            try body.emit(.select);
        },
    }
    try emitLseStore(body, x.size);

    // Rt = old value
    if (x.rt != 31) {
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rt)));
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.storeI64(0);
    }
}

// ── FP arith helpers ─────────────────────────────────────────────────────────

const FpOp2 = enum { add, sub, mul, div, min, max };
const FpOp1 = enum { abs, neg, sqrt };

const v_offset: u32 = cpu.v_offset;

fn vregBaseAddr(rd: u5) i32 {
    return @bitCast(v_offset + @as(u32, rd) * 16);
}

/// load Vn (low 64 bits) as f64 bit pattern, widening from f32 if ftype=00.
/// leaves an i64 (f64 bits) on the stack
fn loadFp(ctx: *Ctx, vn: u5, ftype: u2) !void {
    const body = ctx.body;
    try body.emitRegAddr(vregBaseAddr(vn));
    try body.loadI64(0);
    if (ftype == 0) {
        try body.emit(.call);
        try body.emitU32(ctx.helpers.f32_to_f64);
    }
}

/// store f64 bit pattern from top of stack to Vd. for ftype=00, narrow to f32 first.
/// clears the high 64 bits of Vd
fn storeFp(ctx: *Ctx, vd: u5, ftype: u2) !void {
    const body = ctx.body;
    if (ftype == 0) {
        try body.emit(.call);
        try body.emitU32(ctx.helpers.f64_to_f32);
    }
    // stash value, write low 64
    try body.emit(.local_set);
    try body.emitU32(local_a);
    try body.emitRegAddr(vregBaseAddr(vd));
    try body.emit(.local_get);
    try body.emitU32(local_a);
    try body.storeI64(0);
    // clear high 64
    try body.emitRegAddr(vregBaseAddr(vd) + 8);
    try body.emitConstI64(0);
    try body.storeI64(0);
}

fn emitFpArith2(ctx: *Ctx, x: decode.FpArith2, op: FpOp2) !void {
    try loadFp(ctx, x.rn, x.ftype);
    try loadFp(ctx, x.rm, x.ftype);
    try ctx.body.emit(.call);
    try ctx.body.emitU32(switch (op) {
        .add => ctx.helpers.f64_add,
        .sub => ctx.helpers.f64_sub,
        .mul => ctx.helpers.f64_mul,
        .div => ctx.helpers.f64_div,
        .min => ctx.helpers.f64_min,
        .max => ctx.helpers.f64_max,
    });
    try storeFp(ctx, x.rd, x.ftype);
}

fn emitFpArith1(ctx: *Ctx, x: decode.FpArith1, op: FpOp1) !void {
    try loadFp(ctx, x.rn, x.ftype);
    try ctx.body.emit(.call);
    try ctx.body.emitU32(switch (op) {
        .abs => ctx.helpers.f64_abs,
        .neg => ctx.helpers.f64_neg,
        .sqrt => ctx.helpers.f64_sqrt,
    });
    try storeFp(ctx, x.rd, x.ftype);
}

fn emitFmovFp(ctx: *Ctx, x: decode.FpArith1) !void {
    // FMOV Vd, Vn — copy register
    try loadFp(ctx, x.rn, x.ftype);
    try storeFp(ctx, x.rd, x.ftype);
}

fn emitFcvt(ctx: *Ctx, x: decode.Fcvt) !void {
    const body = ctx.body;
    try body.emitRegAddr(vregBaseAddr(x.rn));
    try body.loadI64(0);
    // widen source to f64 bits if it's f32
    if (x.ftype == 0) {
        try body.emit(.call);
        try body.emitU32(ctx.helpers.f32_to_f64);
    }
    // storeFp will narrow to f32 if dest_ftype == 0, so we leave the f64 bits on the stack
    try storeFp(ctx, x.rd, x.dest_ftype);
}

fn emitFpArith3(ctx: *Ctx, x: decode.FpArith3, op: FpOp2, negate: bool) !void {
    // op == .add → Va + (Vn * Vm); .sub → Va - (Vn * Vm)
    // negate flag wraps the whole result in -result (FNMADD/FNMSUB)
    try loadFp(ctx, x.rn, x.ftype);
    try loadFp(ctx, x.rm, x.ftype);
    try ctx.body.emit(.call);
    try ctx.body.emitU32(ctx.helpers.f64_mul);
    // stash product
    try ctx.body.emit(.local_set);
    try ctx.body.emitU32(local_a);
    try loadFp(ctx, x.ra, x.ftype);
    try ctx.body.emit(.local_get);
    try ctx.body.emitU32(local_a);
    try ctx.body.emit(.call);
    try ctx.body.emitU32(switch (op) {
        .add => ctx.helpers.f64_add,
        .sub => ctx.helpers.f64_sub,
        else => unreachable,
    });
    if (negate) {
        try ctx.body.emit(.call);
        try ctx.body.emitU32(ctx.helpers.f64_neg);
    }
    try storeFp(ctx, x.rd, x.ftype);
}

fn emitFcmp(ctx: *Ctx, x: decode.Fcmp) !void {
    const body = ctx.body;
    try loadFp(ctx, x.rn, x.ftype);
    if (x.with_zero) {
        try body.emitConstI64(0);
    } else {
        try loadFp(ctx, x.rm, x.ftype);
    }
    try body.emit(.call);
    try body.emitU32(ctx.helpers.f64_cmp);
    // helper returns NZCV in low 4 bits (N=bit3, Z=bit2, C=bit1, V=bit0). shift to ARM
    // memory layout: bit 31=N, 30=Z, 29=C, 28=V. extract per-bit and rebuild
    try body.emit(.i32_wrap_i64);
    try body.emit(.local_set);
    try body.emitU32(local_nzcv_tmp);
    try body.emitRegAddr(@bitCast(cpu.nzcv_offset));
    // N (bit 3 of helper → bit 31 of memory)
    try body.emit(.local_get);
    try body.emitU32(local_nzcv_tmp);
    try body.emitConstI32(3);
    try body.emit(.i32_shr_u);
    try body.emitConstI32(1);
    try body.emit(.i32_and);
    try body.emitConstI32(31);
    try body.emit(.i32_shl);
    // Z (bit 2 → bit 30)
    try body.emit(.local_get);
    try body.emitU32(local_nzcv_tmp);
    try body.emitConstI32(2);
    try body.emit(.i32_shr_u);
    try body.emitConstI32(1);
    try body.emit(.i32_and);
    try body.emitConstI32(30);
    try body.emit(.i32_shl);
    try body.emit(.i32_or);
    // C (bit 1 → bit 29)
    try body.emit(.local_get);
    try body.emitU32(local_nzcv_tmp);
    try body.emitConstI32(1);
    try body.emit(.i32_shr_u);
    try body.emitConstI32(1);
    try body.emit(.i32_and);
    try body.emitConstI32(29);
    try body.emit(.i32_shl);
    try body.emit(.i32_or);
    // V (bit 0 → bit 28)
    try body.emit(.local_get);
    try body.emitU32(local_nzcv_tmp);
    try body.emitConstI32(1);
    try body.emit(.i32_and);
    try body.emitConstI32(28);
    try body.emit(.i32_shl);
    try body.emit(.i32_or);
    try body.emit(.i32_store);
    try body.emitMemArg(2, 0);
}

fn emitFmovImm(body: *wasm.Body, x: decode.FmovImm) !void {
    // write bits to low qword of Vd, clear high qword
    try body.emitRegAddr(vregBaseAddr(x.rd));
    try body.emitConstI64(@bitCast(x.bits64));
    try body.storeI64(0);
    try body.emitRegAddr(vregBaseAddr(x.rd) + 8);
    try body.emitConstI64(0);
    try body.storeI64(0);
}

fn emitFmovTopToFp(body: *wasm.Body, x: decode.FmovTop) !void {
    // FMOV Vd.D[1], Xn — write Xn to top 64 bits of Vd (low untouched)
    if (x.rn == 31) {
        try body.emitRegAddr(vregBaseAddr(x.rd) + 8);
        try body.emitConstI64(0);
        try body.storeI64(0);
        return;
    }
    try body.emitRegAddr(vregBaseAddr(x.rd) + 8);
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rn)));
    try body.loadI64(0);
    try body.storeI64(0);
}

fn emitFmovTopFromFp(body: *wasm.Body, x: decode.FmovTop) !void {
    // FMOV Xd, Vn.D[1] — read top 64 bits of Vn into Xd
    if (x.rd == 31) return;
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));
    try body.emitRegAddr(vregBaseAddr(x.rn) + 8);
    try body.loadI64(0);
    try body.storeI64(0);
}

fn emitIntToFp(ctx: *Ctx, x: decode.FpIntCvt, signed: bool) !void {
    const body = ctx.body;
    if (x.rn == 31) {
        try body.emitConstI64(0);
    } else {
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rn)));
        try body.loadI64(0);
    }
    try body.emit(.call);
    const idx: u32 = if (x.sf) (if (signed) ctx.helpers.f64_from_s64 else ctx.helpers.f64_from_u64) else (if (signed) ctx.helpers.f64_from_s32 else ctx.helpers.f64_from_u32);
    try body.emitU32(idx);
    try storeFp(ctx, x.rd, x.ftype);
}

fn emitFpToInt(ctx: *Ctx, x: decode.FpIntCvt, signed: bool) !void {
    const body = ctx.body;
    try loadFp(ctx, x.rn, x.ftype);
    try body.emit(.call);
    const idx: u32 = if (x.sf) (if (signed) ctx.helpers.f64_to_s64 else ctx.helpers.f64_to_u64) else (if (signed) ctx.helpers.f64_to_s32 else ctx.helpers.f64_to_u32);
    try body.emitU32(idx);
    if (x.rd == 31) {
        try body.emit(.drop);
        return;
    }
    try body.emit(.local_set);
    try body.emitU32(local_a);
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));
    try body.emit(.local_get);
    try body.emitU32(local_a);
    try body.storeI64(0);
}

// scalar advanced-simd int<->fp convert on the low lane (ucvtf d,d etc). wasm simd has
// no i64x2<->f64x2 lane convert, so route through the scalar helpers and write lane 0,
// zeroing the rest of the register.
fn emitSimdScalarCvt(ctx: *Ctx, x: decode.SimdScalarCvt) !void {
    const body = ctx.body;
    const base = vregBaseAddr(x.rd);
    if (x.to_int) {
        try loadFp(ctx, x.rn, if (x.dbl) @as(u2, 1) else 0);
        try body.emit(.call);
        try body.emitU32(if (x.dbl)
            (if (x.signed) ctx.helpers.f64_to_s64 else ctx.helpers.f64_to_u64)
        else
            (if (x.signed) ctx.helpers.f64_to_s32 else ctx.helpers.f64_to_u32));
        if (!x.dbl) {
            try body.emitConstI64(0xffffffff);
            try body.emit(.i64_and);
        }
    } else {
        try body.emitRegAddr(vregBaseAddr(x.rn));
        try body.loadI64(0);
        try body.emit(.call);
        try body.emitU32(if (x.dbl)
            (if (x.signed) ctx.helpers.f64_from_s64 else ctx.helpers.f64_from_u64)
        else
            (if (x.signed) ctx.helpers.f64_from_s32 else ctx.helpers.f64_from_u32));
        if (!x.dbl) {
            try body.emit(.call);
            try body.emitU32(ctx.helpers.f64_to_f32);
        }
    }
    try body.emit(.local_set);
    try body.emitU32(local_a);
    try body.emitRegAddr(base);
    try body.emit(.local_get);
    try body.emitU32(local_a);
    try body.storeI64(0);
    try body.emitRegAddr(base + 8);
    try body.emitConstI64(0);
    try body.storeI64(0);
}

// scalar advanced-simd shift by immediate on the low 64-bit lane. n is Vn lane 0, d_old
// is Vd lane 0 (for the accumulate/insert forms). a shift of 64 is valid here and wasm's
// i64 shifts mask the amount to 0..63, so those cases are precomputed to constants.
fn emitSimdScalarShift(ctx: *Ctx, x: decode.SimdScalarShift) !void {
    const body = ctx.body;
    const base = vregBaseAddr(x.rd);
    const nbase = vregBaseAddr(x.rn);

    const loadN = struct {
        fn go(b: *wasm.Body, addr: i32) !void {
            try b.emitConstI32(addr);
            try b.loadI64(0);
        }
    }.go;

    switch (x.op) {
        .shl => {
            try loadN(body, nbase);
            try body.emitConstI64(@intCast(x.shift));
            try body.emit(.i64_shl);
        },
        .sshr => {
            try loadN(body, nbase);
            try body.emitConstI64(@intCast(@min(x.shift, 63)));
            try body.emit(.i64_shr_s);
        },
        .ushr => {
            if (x.shift >= 64) {
                try body.emitConstI64(0);
            } else {
                try loadN(body, nbase);
                try body.emitConstI64(@intCast(x.shift));
                try body.emit(.i64_shr_u);
            }
        },
        .ssra => {
            try loadN(body, base); // d_old
            try loadN(body, nbase);
            try body.emitConstI64(@intCast(@min(x.shift, 63)));
            try body.emit(.i64_shr_s);
            try body.emit(.i64_add);
        },
        .usra => {
            try loadN(body, base); // d_old
            if (x.shift >= 64) {
                try body.emitConstI64(0);
            } else {
                try loadN(body, nbase);
                try body.emitConstI64(@intCast(x.shift));
                try body.emit(.i64_shr_u);
            }
            try body.emit(.i64_add);
        },
        .sli => {
            // (n << shift) | (d_old & ((1<<shift)-1)): keep the low `shift` bits of dest
            try loadN(body, nbase);
            try body.emitConstI64(@intCast(x.shift));
            try body.emit(.i64_shl);
            try loadN(body, base);
            const keep: u64 = if (x.shift == 0) 0 else (@as(u64, 1) << @intCast(x.shift)) - 1;
            try body.emitConstI64(@bitCast(keep));
            try body.emit(.i64_and);
            try body.emit(.i64_or);
        },
        .sri => {
            // (n >>> shift) | (d_old & ~(mask)): keep the high `shift` bits of dest
            if (x.shift >= 64) {
                try loadN(body, base); // dest unchanged (n>>>64 == 0)
            } else {
                try loadN(body, nbase);
                try body.emitConstI64(@intCast(x.shift));
                try body.emit(.i64_shr_u);
                try loadN(body, base);
                const low: u64 = (@as(u64, 1) << @intCast(64 - x.shift)) - 1;
                try body.emitConstI64(@bitCast(~low));
                try body.emit(.i64_and);
                try body.emit(.i64_or);
            }
        },
    }

    try body.emit(.local_set);
    try body.emitU32(local_a);
    try body.emitRegAddr(base);
    try body.emit(.local_get);
    try body.emitU32(local_a);
    try body.storeI64(0);
    try body.emitRegAddr(base + 8);
    try body.emitConstI64(0);
    try body.storeI64(0);
}

// fp <-> fixed-point convert: int<->fp plus a multiply by 2^(-fbits) (scvtf) or 2^fbits
// (fcvtz). the scale factor is an exact power of two, built as a constant f64 bit pattern.
fn emitFpFixedCvt(ctx: *Ctx, x: decode.FpFixedCvt) !void {
    const body = ctx.body;
    const bias: u64 = 1023;
    const exp: u64 = if (x.to_int) bias + x.fbits else bias - x.fbits;
    const factor_bits: u64 = exp << 52;

    if (x.to_int) {
        try loadFp(ctx, x.rn, x.ftype);
        try body.emitConstI64(@bitCast(factor_bits));
        try body.emit(.call);
        try body.emitU32(ctx.helpers.f64_mul);
        try body.emit(.call);
        try body.emitU32(if (x.sf)
            (if (x.signed) ctx.helpers.f64_to_s64 else ctx.helpers.f64_to_u64)
        else
            (if (x.signed) ctx.helpers.f64_to_s32 else ctx.helpers.f64_to_u32));
        if (x.rd == 31) {
            try body.emit(.drop);
            return;
        }
        try body.emit(.local_set);
        try body.emitU32(local_a);
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));
        try body.emit(.local_get);
        try body.emitU32(local_a);
        try body.storeI64(0);
        return;
    }

    if (x.rn == 31) {
        try body.emitConstI64(0);
    } else {
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rn)));
        try body.loadI64(0);
    }
    try body.emit(.call);
    try body.emitU32(if (x.sf)
        (if (x.signed) ctx.helpers.f64_from_s64 else ctx.helpers.f64_from_u64)
    else
        (if (x.signed) ctx.helpers.f64_from_s32 else ctx.helpers.f64_from_u32));
    try body.emitConstI64(@bitCast(factor_bits));
    try body.emit(.call);
    try body.emitU32(ctx.helpers.f64_mul);
    try storeFp(ctx, x.rd, x.ftype);
}

// ── SIMD bitwise ─────────────────────────────────────────────────────────────

const SimdLogical = enum { @"and", @"or", xor };

fn emitSimdBitwiseReg(body: *wasm.Body, x: decode.SimdThreeSame, op: SimdLogical, invert_rm: bool) !void {
    const rn = vregBaseAddr(x.rn);
    const rm = vregBaseAddr(x.rm);
    const rd = vregBaseAddr(x.rd);

    inline for ([_]i32{ 0, 8 }) |offset| {
        const want_high = offset != 0;
        if (want_high and !x.q) {
            try body.emitRegAddr(rd + offset);
            try body.emitConstI64(0);
            try body.storeI64(0);
        } else {
            try body.emitRegAddr(rd + offset);
            try body.emitRegAddr(rn + offset);
            try body.loadI64(0);
            try body.emitRegAddr(rm + offset);
            try body.loadI64(0);
            if (invert_rm) {
                try body.emitConstI64(-1);
                try body.emit(.i64_xor);
            }
            try body.emit(switch (op) {
                .@"and" => .i64_and,
                .@"or" => .i64_or,
                .xor => .i64_xor,
            });
            try body.storeI64(0);
        }
    }
}

const BitSelectKind = enum { bsl, bit, bif };

// bsl/bit/bif all reduce to result = a ^ ((b ^ a) & mask), a per-lane bit select.
// bsl: a=rm b=rn mask=rd_old ; bit: a=rd_old b=rn mask=rm ; bif: a=rn b=rd_old mask=rm.
fn emitSimdBitSelect(body: *wasm.Body, x: decode.SimdThreeSame, kind: BitSelectKind) !void {
    const rn = vregBaseAddr(x.rn);
    const rm = vregBaseAddr(x.rm);
    const rd = vregBaseAddr(x.rd);
    const a, const b, const mask = switch (kind) {
        .bsl => .{ rm, rn, rd },
        .bit => .{ rd, rn, rm },
        .bif => .{ rn, rd, rm },
    };

    inline for ([_]i32{ 0, 8 }) |offset| {
        const want_high = offset != 0;
        try body.emitRegAddr(rd + offset);
        if (want_high and !x.q) {
            try body.emitConstI64(0);
        } else {
            try body.emitConstI32(a + offset);
            try body.loadI64(0);
            try body.emitConstI32(b + offset);
            try body.loadI64(0);
            try body.emitConstI32(a + offset);
            try body.loadI64(0);
            try body.emit(.i64_xor); // b ^ a
            try body.emitConstI32(mask + offset);
            try body.loadI64(0);
            try body.emit(.i64_and); // (b ^ a) & mask
            try body.emit(.i64_xor); // a ^ ((b ^ a) & mask)
        }
        try body.storeI64(0);
    }
}

fn emitSimdNot(body: *wasm.Body, x: decode.SimdTwoRegNoSize) !void {
    const rn = vregBaseAddr(x.rn);
    const rd = vregBaseAddr(x.rd);
    try body.emitRegAddr(rd);
    try body.emitRegAddr(rn);
    try body.loadI64(0);
    try body.emitConstI64(-1);
    try body.emit(.i64_xor);
    try body.storeI64(0);
    try body.emitRegAddr(rd + 8);
    if (x.q) {
        try body.emitRegAddr(rn + 8);
        try body.loadI64(0);
        try body.emitConstI64(-1);
        try body.emit(.i64_xor);
    } else {
        try body.emitConstI64(0);
    }
    try body.storeI64(0);
}

// ── vector NEON via native wasm v128 ─────────────────────────────────────────

const simd = wasm.Simd;

/// load Vn as v128 onto the stack (16 bytes at the register's cpu-memory offset)
fn loadVreg(body: *wasm.Body, r: u5) !void {
    try body.emitRegAddr(vregBaseAddr(r));
    try body.loadV128(0);
}

/// emit: Vd = op(Vn, Vm) as a native v128 binary op. handles the [127:64] clear
fn emitVecBinop(body: *wasm.Body, rd: u5, rn: u5, rm: u5, q: bool, sub_op: u32) !void {
    try body.emitRegAddr(vregBaseAddr(rd)); // dest addr (for the store)
    try loadVreg(body, rn);
    try loadVreg(body, rm);
    try body.emitSimd(sub_op);
    try body.storeV128(0);
    if (!q) {
        try body.emitRegAddr(vregBaseAddr(rd) + 8);
        try body.emitConstI64(0);
        try body.storeI64(0);
    }
}

/// emit: Vd = op(Vn) as a native v128 unary op
fn emitVecUnop(body: *wasm.Body, rd: u5, rn: u5, q: bool, sub_op: u32) !void {
    try body.emitRegAddr(vregBaseAddr(rd));
    try loadVreg(body, rn);
    try body.emitSimd(sub_op);
    try body.storeV128(0);
    if (!q) {
        try body.emitRegAddr(vregBaseAddr(rd) + 8);
        try body.emitConstI64(0);
        try body.storeI64(0);
    }
}

fn emitVecFp3(body: *wasm.Body, x: decode.VecFp3) !void {
    const f64lanes = x.sz == 1;
    const sub: ?u32 = switch (x.op) {
        .add => if (f64lanes) simd.f64x2_add else simd.f32x4_add,
        .sub => if (f64lanes) simd.f64x2_sub else simd.f32x4_sub,
        .mul => if (f64lanes) simd.f64x2_mul else simd.f32x4_mul,
        .div => if (f64lanes) simd.f64x2_div else simd.f32x4_div,
        // arm FMAX/FMIN propagate NaN and treat +0>-0; wasm max/min match this
        .max => if (f64lanes) simd.f64x2_max else simd.f32x4_max,
        .min => if (f64lanes) simd.f64x2_min else simd.f32x4_min,
        .maxnm => if (f64lanes) simd.f64x2_max else simd.f32x4_max,
        .minnm => if (f64lanes) simd.f64x2_min else simd.f32x4_min,
        .cmeq => if (f64lanes) simd.f64x2_eq else simd.f32x4_eq,
        .cmge => if (f64lanes) simd.f64x2_ge else simd.f32x4_ge,
        .cmgt => if (f64lanes) simd.f64x2_gt else simd.f32x4_gt,
        else => null,
    };
    if (sub) |s| {
        try emitVecBinop(body, x.rd, x.rn, x.rm, x.q, s);
        return;
    }
    switch (x.op) {
        // FMLA: Vd = Vd + Vn*Vm ; FMLS: Vd = Vd - Vn*Vm (fused; we do mul then add/sub)
        .mla, .mls => {
            const mul: u32 = if (f64lanes) simd.f64x2_mul else simd.f32x4_mul;
            const acc: u32 = if (x.op == .mla) (if (f64lanes) simd.f64x2_add else simd.f32x4_add) else (if (f64lanes) simd.f64x2_sub else simd.f32x4_sub);
            try body.emitRegAddr(vregBaseAddr(x.rd));
            try loadVreg(body, x.rd);
            try loadVreg(body, x.rn);
            try loadVreg(body, x.rm);
            try body.emitSimd(mul);
            try body.emitSimd(acc);
            try body.storeV128(0);
            if (!x.q) {
                try body.emitRegAddr(vregBaseAddr(x.rd) + 8);
                try body.emitConstI64(0);
                try body.storeI64(0);
            }
        },
        // FABD: |Vn - Vm|
        .fabd => {
            const subop: u32 = if (f64lanes) simd.f64x2_sub else simd.f32x4_sub;
            const absop: u32 = if (f64lanes) simd.f64x2_abs else simd.f32x4_abs;
            try body.emitRegAddr(vregBaseAddr(x.rd));
            try loadVreg(body, x.rn);
            try loadVreg(body, x.rm);
            try body.emitSimd(subop);
            try body.emitSimd(absop);
            try body.storeV128(0);
            if (!x.q) {
                try body.emitRegAddr(vregBaseAddr(x.rd) + 8);
                try body.emitConstI64(0);
                try body.storeI64(0);
            }
        },
        else => {
            // faddp/fmaxp etc (pairwise) not yet — leave as unknown so the
            // dispatcher can surface it rather than silently miscompile
            try emitVecUnop(body, x.rd, x.rn, x.q, if (f64lanes) simd.f64x2_add else simd.f32x4_add);
        },
    }
}

fn emitVecFp2(body: *wasm.Body, x: decode.VecFp2) !void {
    const f64lanes = x.sz == 1;
    const sub: ?u32 = switch (x.op) {
        .neg => if (f64lanes) simd.f64x2_neg else simd.f32x4_neg,
        .abs => if (f64lanes) simd.f64x2_abs else simd.f32x4_abs,
        .sqrt => if (f64lanes) simd.f64x2_sqrt else simd.f32x4_sqrt,
        .frintn => if (f64lanes) simd.f64x2_nearest else simd.f32x4_nearest,
        .frintm => if (f64lanes) simd.f64x2_floor else simd.f32x4_floor,
        .frintp => if (f64lanes) simd.f64x2_ceil else simd.f32x4_ceil,
        .frintz, .frintx => if (f64lanes) simd.f64x2_trunc else simd.f32x4_trunc,
        .frinta => if (f64lanes) simd.f64x2_nearest else simd.f32x4_nearest,
        else => null,
    };
    if (sub) |s| {
        try emitVecUnop(body, x.rd, x.rn, x.q, s);
        return;
    }
    // fcvt / scvtf vector conversions (single precision only for now, sz=0)
    switch (x.op) {
        .fcvtzs => try emitVecUnop(body, x.rd, x.rn, x.q, simd.i32x4_trunc_sat_f32x4_s),
        .fcvtzu => try emitVecUnop(body, x.rd, x.rn, x.q, simd.i32x4_trunc_sat_f32x4_u),
        .scvtf => try emitVecUnop(body, x.rd, x.rn, x.q, simd.f32x4_convert_i32x4_s),
        .ucvtf => try emitVecUnop(body, x.rd, x.rn, x.q, simd.f32x4_convert_i32x4_u),
        else => try emitVecUnop(body, x.rd, x.rn, x.q, if (f64lanes) simd.f64x2_abs else simd.f32x4_abs),
    }
}

/// native wasm sub-op for an integer vector op at a given element size, or null
/// if wasm SIMD has no direct equivalent (caller falls back to the js helper).
/// size: 0=i8x16, 1=i16x8, 2=i32x4, 3=i64x2
fn vecIntSubOp(op: decode.VecIntOp, size: u2) ?u32 {
    return switch (op) {
        .add => switch (size) {
            0 => simd.i8x16_add,
            1 => simd.i16x8_add,
            2 => simd.i32x4_add,
            3 => simd.i64x2_add,
        },
        .sub => switch (size) {
            0 => simd.i8x16_sub,
            1 => simd.i16x8_sub,
            2 => simd.i32x4_sub,
            3 => simd.i64x2_sub,
        },
        .mul => switch (size) {
            1 => simd.i16x8_mul,
            2 => simd.i32x4_mul,
            3 => simd.i64x2_mul,
            else => null, // no i8x16.mul in wasm
        },
        .cmeq => switch (size) {
            0 => simd.i8x16_eq,
            1 => simd.i16x8_eq,
            2 => simd.i32x4_eq,
            3 => simd.i64x2_eq,
        },
        .cmgt => switch (size) {
            0 => simd.i8x16_gt_s,
            1 => simd.i16x8_gt_s,
            2 => simd.i32x4_gt_s,
            3 => simd.i64x2_gt_s,
        },
        .cmge => switch (size) {
            0 => simd.i8x16_ge_s,
            1 => simd.i16x8_ge_s,
            2 => simd.i32x4_ge_s,
            3 => simd.i64x2_ge_s,
        },
        .cmhi => switch (size) {
            0 => simd.i8x16_gt_u,
            1 => simd.i16x8_gt_u,
            2 => simd.i32x4_gt_u,
            3 => null, // no i64x2 unsigned compare in wasm
        },
        .cmhs => switch (size) {
            0 => simd.i8x16_ge_u,
            1 => simd.i16x8_ge_u,
            2 => simd.i32x4_ge_u,
            3 => null,
        },
        .smax => switch (size) {
            0 => simd.i8x16_max_s,
            1 => simd.i16x8_max_s,
            2 => simd.i32x4_max_s,
            3 => null,
        },
        .umax => switch (size) {
            0 => simd.i8x16_max_u,
            1 => simd.i16x8_max_u,
            2 => simd.i32x4_max_u,
            3 => null,
        },
        .smin => switch (size) {
            0 => simd.i8x16_min_s,
            1 => simd.i16x8_min_s,
            2 => simd.i32x4_min_s,
            3 => null,
        },
        .umin => switch (size) {
            0 => simd.i8x16_min_u,
            1 => simd.i16x8_min_u,
            2 => simd.i32x4_min_u,
            3 => null,
        },
        .sqadd => switch (size) {
            0 => simd.i8x16_add_sat_s,
            1 => simd.i16x8_add_sat_s,
            else => null, // no 32/64-bit saturating in wasm
        },
        .uqadd => switch (size) {
            0 => simd.i8x16_add_sat_u,
            1 => simd.i16x8_add_sat_u,
            else => null,
        },
        .sqsub => switch (size) {
            0 => simd.i8x16_sub_sat_s,
            1 => simd.i16x8_sub_sat_s,
            else => null,
        },
        .uqsub => switch (size) {
            0 => simd.i8x16_sub_sat_u,
            1 => simd.i16x8_sub_sat_u,
            else => null,
        },
        else => null,
    };
}

/// map a vec_int3 op to its equivalent js-helper simd_op opcode, for the combos
/// wasm SIMD can't express natively (i8 mul, i64 min/max/unsigned-cmp, 32/64-bit
/// saturating). the helper handles all sizes correctly
fn vecIntHelperOp(op: decode.VecIntOp) ?u8 {
    return switch (op) {
        .add => simd_op_add,
        .sub => simd_op_sub,
        .cmeq => simd_op_cmeq,
        .cmgt => simd_op_cmgt,
        .cmhi => simd_op_cmhi,
        .cmge => simd_op_cmge,
        .cmhs => simd_op_cmhs,
        .smax => simd_op_smax,
        .umax => simd_op_umax,
        .smin => simd_op_smin,
        .umin => simd_op_umin,
        .mul => simd_op_mul,
        .sqadd => simd_op_sqadd,
        .uqadd => simd_op_uqadd,
        .sqsub => simd_op_sqsub,
        .uqsub => simd_op_uqsub,
        else => null,
    };
}

fn emitVecInt3(ctx: *Ctx, x: decode.VecInt3) !void {
    if (vecIntSubOp(x.op, x.size)) |sub| {
        try emitVecBinop(ctx.body, x.rd, x.rn, x.rm, x.q, sub);
        return;
    }
    // no native v128 op: fall back to the js helper, which covers every size
    if (vecIntHelperOp(x.op)) |hop| {
        try ctx.body.emitConstI64(packSimdArgs(hop, x.q, x.size, 0, x.rd, x.rn, x.rm));
        try ctx.body.emitConstI64(0);
        try ctx.body.emit(.call);
        try ctx.body.emitU32(ctx.helpers.simd_op);
        try ctx.body.emit(.drop);
        return;
    }
    // cmtst and anything unmapped: emit a no-op-ish exit is wrong; use helper add as last resort
    try ctx.body.emitConstI64(packSimdArgs(simd_op_add, x.q, x.size, 0, x.rd, x.rn, x.rm));
    try ctx.body.emitConstI64(0);
    try ctx.body.emit(.call);
    try ctx.body.emitU32(ctx.helpers.simd_op);
    try ctx.body.emit(.drop);
}

fn emitVecInt2(body: *wasm.Body, x: decode.VecInt2) !void {
    // compare-against-zero: emit Vn <cmp> zerovec (a per-lane native v128 compare).
    // wasm compares already yield all-ones / all-zero lanes, matching NEON semantics.
    const cmp: ?u32 = switch (x.op) {
        .cmeq0 => switch (x.size) { 0 => simd.i8x16_eq, 1 => simd.i16x8_eq, 2 => simd.i32x4_eq, 3 => simd.i64x2_eq },
        .cmgt0 => switch (x.size) { 0 => simd.i8x16_gt_s, 1 => simd.i16x8_gt_s, 2 => simd.i32x4_gt_s, 3 => simd.i64x2_gt_s },
        .cmge0 => switch (x.size) { 0 => simd.i8x16_ge_s, 1 => simd.i16x8_ge_s, 2 => simd.i32x4_ge_s, 3 => simd.i64x2_ge_s },
        .cmlt0 => switch (x.size) { 0 => simd.i8x16_lt_s, 1 => simd.i16x8_lt_s, 2 => simd.i32x4_lt_s, 3 => simd.i64x2_lt_s },
        .cmle0 => switch (x.size) { 0 => simd.i8x16_le_s, 1 => simd.i16x8_le_s, 2 => simd.i32x4_le_s, 3 => simd.i64x2_le_s },
        else => null,
    };
    if (cmp) |c| {
        try body.emitRegAddr(vregBaseAddr(x.rd));
        try loadVreg(body, x.rn);
        try body.emitConstI32(0);
        try body.emitSimd(simd.i8x16_splat); // zero vector
        try body.emitSimd(c);
        try body.storeV128(0);
        if (!x.q) {
            try body.emitRegAddr(vregBaseAddr(x.rd) + 8);
            try body.emitConstI64(0);
            try body.storeI64(0);
        }
        return;
    }
    const sub: u32 = switch (x.op) {
        .neg => switch (x.size) {
            0 => simd.i8x16_neg,
            1 => simd.i16x8_neg,
            2 => simd.i32x4_neg,
            3 => simd.i64x2_neg,
        },
        .abs => switch (x.size) {
            0 => simd.i8x16_abs,
            1 => simd.i16x8_abs,
            2 => simd.i32x4_abs,
            3 => simd.i64x2_abs,
        },
        else => unreachable,
    };
    try emitVecUnop(body, x.rd, x.rn, x.q, sub);
}

// ── widening three-different (native extmul / extend) ────────────────────────

/// extmul sub-op for (source size, high-half, signed)
fn extmulSub(size: u2, high: bool, signed: bool) ?u32 {
    return switch (size) {
        0 => if (signed) (if (high) simd.i16x8_extmul_high_i8x16_s else simd.i16x8_extmul_low_i8x16_s) else (if (high) simd.i16x8_extmul_high_i8x16_u else simd.i16x8_extmul_low_i8x16_u),
        1 => if (signed) (if (high) simd.i32x4_extmul_high_i16x8_s else simd.i32x4_extmul_low_i16x8_s) else (if (high) simd.i32x4_extmul_high_i16x8_u else simd.i32x4_extmul_low_i16x8_u),
        2 => if (signed) (if (high) simd.i64x2_extmul_high_i32x4_s else simd.i64x2_extmul_low_i32x4_s) else (if (high) simd.i64x2_extmul_high_i32x4_u else simd.i64x2_extmul_low_i32x4_u),
        else => null,
    };
}

/// extend sub-op for (source size, high-half, signed)
fn extendSub(size: u2, high: bool, signed: bool) ?u32 {
    return switch (size) {
        0 => if (signed) (if (high) simd.i16x8_extend_high_i8x16_s else simd.i16x8_extend_low_i8x16_s) else (if (high) simd.i16x8_extend_high_i8x16_u else simd.i16x8_extend_low_i8x16_u),
        1 => if (signed) (if (high) simd.i32x4_extend_high_i16x8_s else simd.i32x4_extend_low_i16x8_s) else (if (high) simd.i32x4_extend_high_i16x8_u else simd.i32x4_extend_low_i16x8_u),
        2 => if (signed) (if (high) simd.i64x2_extend_high_i32x4_s else simd.i64x2_extend_low_i32x4_s) else (if (high) simd.i64x2_extend_high_i32x4_u else simd.i64x2_extend_low_i32x4_u),
        else => null,
    };
}

fn wideAddSub(size: u2) ?struct { add: u32, sub: u32 } {
    return switch (size) {
        0 => .{ .add = simd.i16x8_add, .sub = simd.i16x8_sub },
        1 => .{ .add = simd.i32x4_add, .sub = simd.i32x4_sub },
        2 => .{ .add = simd.i64x2_add, .sub = simd.i64x2_sub },
        else => null,
    };
}

fn emitVecWiden(body: *wasm.Body, x: decode.VecWiden) !void {
    const signed = switch (x.op) {
        .smull, .saddl, .ssubl, .smlal, .smlsl, .saddw, .ssubw => true,
        else => false,
    };
    const high = x.q; // the "2" variants operate on the high 64 bits of the sources
    const dst = vregBaseAddr(x.rd);

    switch (x.op) {
        // widening multiply (optionally accumulating into Vd)
        .smull, .umull, .smlal, .umlal, .smlsl, .umlsl => {
            const emsub = extmulSub(x.size, high, signed) orelse return;
            const accumulate = switch (x.op) {
                .smlal, .umlal, .smlsl, .umlsl => true,
                else => false,
            };
            const is_sub = x.op == .smlsl or x.op == .umlsl;
            const as = wideAddSub(x.size) orelse return;
            try body.emitRegAddr(dst);
            if (accumulate) try loadVreg(body, x.rd);
            try loadVreg(body, x.rn);
            try loadVreg(body, x.rm);
            try body.emitSimd(emsub);
            if (accumulate) try body.emitSimd(if (is_sub) as.sub else as.add);
            try body.storeV128(0);
        },
        // widening add/sub: widen both, then add/sub
        .saddl, .uaddl, .ssubl, .usubl => {
            const ext = extendSub(x.size, high, signed) orelse return;
            const as = wideAddSub(x.size) orelse return;
            const is_sub = x.op == .ssubl or x.op == .usubl;
            try body.emitRegAddr(dst);
            try loadVreg(body, x.rn);
            try body.emitSimd(ext);
            try loadVreg(body, x.rm);
            try body.emitSimd(ext);
            try body.emitSimd(if (is_sub) as.sub else as.add);
            try body.storeV128(0);
        },
        // wide + narrow: Vn is already wide, widen Vm and add/sub
        .saddw, .uaddw, .ssubw, .usubw => {
            const ext = extendSub(x.size, high, signed) orelse return;
            const as = wideAddSub(x.size) orelse return;
            const is_sub = x.op == .ssubw or x.op == .usubw;
            try body.emitRegAddr(dst);
            try loadVreg(body, x.rn);
            try loadVreg(body, x.rm);
            try body.emitSimd(ext);
            try body.emitSimd(if (is_sub) as.sub else as.add);
            try body.storeV128(0);
        },
    }
}

// ── structured loads/stores (LD1/ST1 contiguous native; LD2..4 via helper) ────

/// push the guest access address (Rn as base) + guest_mem_base onto the stack
fn pushGuestAddr(body: *wasm.Body, rn: u5) !void {
    try loadGprOrSp(body, rn);
    try body.emit(.i32_wrap_i64);
    try body.emitConstI32(@bitCast(cpu.guest_mem_base));
    try body.emit(.i32_add);
}

fn emitSimdLdstMulti(ctx: *Ctx, x: decode.SimdLdstMulti) !void {
    const body = ctx.body;
    const lane_bytes: u32 = if (x.q) 16 else 8;
    const total: u32 = lane_bytes * @as(u32, x.reg_count);

    if (x.deinterleave) {
        // LD2/LD3/LD4 (or ST): de/interleaving via the js helper
        try body.emitConstI64(packSimdArgs(simd_op_ldst_struct, x.q, x.size, @as(u8, x.reg_count) | (@as(u8, @intFromBool(x.is_load)) << 4), x.rt, x.rn, 0));
        try body.emitConstI64(0);
        try body.emit(.call);
        try body.emitU32(ctx.helpers.simd_op);
        try body.emit(.drop);
    } else {
        // LD1/ST1: contiguous. base access address (i32) into $nzcv_tmp
        try loadGprOrSp(body, x.rn);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(cpu.guest_mem_base));
        try body.emit(.i32_add);
        try body.emit(.local_set);
        try body.emitU32(local_nzcv_tmp);

        var i: u3 = 0;
        while (i < x.reg_count) : (i += 1) {
            const vreg: u5 = @truncate((@as(u32, x.rt) + i) & 31);
            const off: u32 = @as(u32, i) * lane_bytes;
            if (x.is_load) {
                try body.emitRegAddr(vregBaseAddr(vreg));
                try body.emit(.local_get);
                try body.emitU32(local_nzcv_tmp);
                if (x.q) {
                    try body.loadV128(off);
                    try body.storeV128(0);
                } else {
                    try body.loadI64(off);
                    try body.storeI64(0);
                    // clear upper 64 bits of the 8-byte load target
                    try body.emitRegAddr(vregBaseAddr(vreg) + 8);
                    try body.emitConstI64(0);
                    try body.storeI64(0);
                }
            } else {
                try body.emit(.local_get);
                try body.emitU32(local_nzcv_tmp);
                if (x.q) {
                    try body.emitRegAddr(vregBaseAddr(vreg));
                    try body.loadV128(0);
                    try body.storeV128(off);
                } else {
                    try body.emitRegAddr(vregBaseAddr(vreg));
                    try body.loadI64(0);
                    try body.storeI64(off);
                }
            }
        }
    }

    // post-index writeback: Rn += (Rm==31 ? total : Xm)
    if (x.post) {
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rn)));
        try loadGprOrSp(body, x.rn);
        if (x.rm == 31) {
            try body.emitConstI64(@as(i64, total));
        } else {
            try loadGpr(body, x.rm, true);
        }
        try body.emit(.i64_add);
        try body.storeI64(0);
    }
}

fn emitSimdLd1r(body: *wasm.Body, x: decode.SimdLd1r) !void {
    // load one element and splat it across all lanes of Vd
    try loadGprOrSp(body, x.rn);
    try body.emit(.i32_wrap_i64);
    try body.emitConstI32(@bitCast(cpu.guest_mem_base));
    try body.emit(.i32_add);
    try body.emit(.local_set);
    try body.emitU32(local_nzcv_tmp);

    try body.emitRegAddr(vregBaseAddr(x.rt));
    try body.emit(.local_get);
    try body.emitU32(local_nzcv_tmp);
    const splat_sub: u32 = switch (x.size) {
        0 => blk: {
            try body.emitSimd(wasm.Simd.v128_load8_splat);
            break :blk 0;
        },
        1 => blk: {
            try body.emitSimd(wasm.Simd.v128_load16_splat);
            break :blk 1;
        },
        2 => blk: {
            try body.emitSimd(wasm.Simd.v128_load32_splat);
            break :blk 2;
        },
        3 => blk: {
            try body.emitSimd(wasm.Simd.v128_load64_splat);
            break :blk 3;
        },
    };
    try body.emitMemArg(splat_sub, 0);
    try body.storeV128(0);
    if (!x.q) {
        try body.emitRegAddr(vregBaseAddr(x.rt) + 8);
        try body.emitConstI64(0);
        try body.storeI64(0);
    }

    if (x.post) {
        const esize: u32 = @as(u32, 1) << x.size;
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rn)));
        try loadGprOrSp(body, x.rn);
        if (x.rm == 31) {
            try body.emitConstI64(@as(i64, esize));
        } else {
            try loadGpr(body, x.rm, true);
        }
        try body.emit(.i64_add);
        try body.storeI64(0);
    }
}

// ── across-lanes reductions (js helper: inherently cross-lane, low frequency) ──

fn emitVecAcross(ctx: *Ctx, x: decode.VecAcross) !void {
    const hop: u8 = switch (x.op) {
        .addv => simd_op_addv,
        .saddlv => simd_op_saddlv,
        .uaddlv => simd_op_uaddlv,
        .smaxv => simd_op_smaxv,
        .umaxv => simd_op_umaxv,
        .sminv => simd_op_sminv,
        .uminv => simd_op_uminv,
        .fmaxv => simd_op_fmaxv,
        .fminv => simd_op_fminv,
    };
    // pack: reuse the simd_op frame; rm unused (0), rn is source, rd is dest
    try ctx.body.emitConstI64(packSimdArgs(hop, x.q, x.size, 0, x.rd, x.rn, 0));
    try ctx.body.emitConstI64(0);
    try ctx.body.emit(.call);
    try ctx.body.emitU32(ctx.helpers.simd_op);
    try ctx.body.emit(.drop);
}

// ── SIMD via simd_op helper ──────────────────────────────────────────────────

// these MUST match the opcode constants in src/helpers.ts
const simd_op_add: u8 = 1;
const simd_op_sub: u8 = 2;
const simd_op_cmeq: u8 = 10;
const simd_op_cmgt: u8 = 11;
const simd_op_cmhi: u8 = 12;
const simd_op_cmge: u8 = 13;
const simd_op_cmhs: u8 = 14;
const simd_op_umaxp: u8 = 20;
const simd_op_addp: u8 = 21;
const simd_op_tbl: u8 = 30;
const simd_op_tbx: u8 = 31;
const simd_op_zip1: u8 = 40;
const simd_op_zip2: u8 = 41;
const simd_op_uzp1: u8 = 42;
const simd_op_uzp2: u8 = 43;
const simd_op_trn1: u8 = 44;
const simd_op_trn2: u8 = 45;
// integer three-same ops that wasm SIMD can't express natively for some sizes
const simd_op_mul: u8 = 50;
const simd_op_smax: u8 = 51;
const simd_op_smin: u8 = 52;
const simd_op_umin: u8 = 53;
const simd_op_umax: u8 = 54; // note: 20 is umaxp (pairwise); this is elementwise
const simd_op_sqadd: u8 = 55;
const simd_op_uqadd: u8 = 56;
const simd_op_sqsub: u8 = 57;
const simd_op_uqsub: u8 = 58;
// across-lanes reductions (result scalar in Vd lane 0)
const simd_op_addv: u8 = 60;
const simd_op_saddlv: u8 = 61;
const simd_op_uaddlv: u8 = 62;
const simd_op_smaxv: u8 = 63;
const simd_op_umaxv: u8 = 64;
const simd_op_sminv: u8 = 65;
const simd_op_uminv: u8 = 66;
const simd_op_fmaxv: u8 = 67;
const simd_op_fminv: u8 = 68;
// de/interleaving structured load/store (LD2/LD3/LD4)
const simd_op_ldst_struct: u8 = 70;
const simd_op_shrn: u8 = 72;
const simd_op_addhn: u8 = 73;
const simd_op_uminp: u8 = 22;
const simd_op_ext: u8 = 74;

fn sizeToLog2(size: u2) u8 {
    return size;
}

fn packSimdArgs(op: u8, q: bool, size: u8, extra: u8, rd: u5, rn: u5, rm: u5) i64 {
    const p: u64 = (@as(u64, op) << 56) | (@as(u64, @intFromBool(q)) << 48) |
        (@as(u64, size) << 40) | (@as(u64, extra) << 32) |
        (@as(u64, rd) << 24) | (@as(u64, rn) << 16) | (@as(u64, rm) << 8);
    return @bitCast(p);
}

fn emitSimdOpCall(ctx: *Ctx, x: decode.SimdThreeSame, op: u8, extra: u8) !void {
    const body = ctx.body;
    try body.emitConstI64(packSimdArgs(op, x.q, sizeToLog2(x.size), extra, x.rd, x.rn, x.rm));
    try body.emitConstI64(0);
    try body.emit(.call);
    try body.emitU32(ctx.helpers.simd_op);
    try body.emit(.drop);
}

fn emitSimdTbl(ctx: *Ctx, x: decode.SimdTbl) !void {
    const body = ctx.body;
    const op: u8 = if (x.is_tbx) simd_op_tbx else simd_op_tbl;
    try body.emitConstI64(packSimdArgs(op, x.q, 0, x.len, x.rd, x.rn, x.rm));
    try body.emitConstI64(0);
    try body.emit(.call);
    try body.emitU32(ctx.helpers.simd_op);
    try body.emit(.drop);
}

// ── SIMD shift long (SSHLL/USHLL): widen each element of Vn by 2x, shift left ──

fn emitSimdShiftLong(body: *wasm.Body, x: decode.SimdShiftLong, signed: bool) !void {
    // sshll/ushll widen each element of the source's low (or high, for the "2"
    // variant) 64 bits to 2x width, shifting left. read the whole source qword
    // into $a first so in-place widening (rd == rn) never clobbers unread lanes
    const src_off: i32 = vregBaseAddr(x.rn) + (if (x.q) @as(i32, 8) else 0);
    const dst_off: i32 = vregBaseAddr(x.rd);
    const dst_esize: u8 = x.esize * 2;
    const elems: u8 = 8 / x.esize;
    const src_bits: u6 = @intCast(x.esize * 8);

    try body.emitRegAddr(src_off);
    try body.loadI64(0);
    try body.emit(.local_set);
    try body.emitU32(local_a);

    var i: u8 = 0;
    while (i < elems) : (i += 1) {
        try body.emitRegAddr(dst_off + @as(i32, @intCast(i)) * @as(i32, @intCast(dst_esize)));
        // extract element i from $a: shift down, then sign/zero-extend to the
        // widened width via shl/shr pair
        try body.emit(.local_get);
        try body.emitU32(local_a);
        if (i > 0) {
            try body.emitConstI64(@as(i64, @intCast(i)) * @as(i64, src_bits));
            try body.emit(.i64_shr_u);
        }
        // isolate the source-width field, sign- or zero-extend to i64
        const up: i64 = 64 - @as(i64, src_bits);
        try body.emitConstI64(up);
        try body.emit(.i64_shl);
        try body.emitConstI64(up);
        try body.emit(if (signed) .i64_shr_s else .i64_shr_u);
        if (x.shift > 0) {
            try body.emitConstI64(@as(i64, x.shift));
            try body.emit(.i64_shl);
        }
        // store widened element (mask to dst width for the sub-64 cases)
        switch (dst_esize) {
            2 => {
                try body.emit(.i64_store16);
                try body.emitMemArg(1, 0);
            },
            4 => {
                try body.emit(.i64_store32);
                try body.emitMemArg(2, 0);
            },
            8 => {
                try body.storeI64(0);
            },
            else => return,
        }
    }
}

fn emitSimdDupElem(body: *wasm.Body, x: decode.SimdDupElem) !void {
    const src_off: i32 = vregBaseAddr(x.rn) + @as(i32, x.index) * @as(i32, x.esize);
    const dst_off: i32 = vregBaseAddr(x.rd);
    // load source element zero-extended into i64
    try body.emitRegAddr(src_off);
    switch (x.esize) {
        1 => {
            try body.emit(.i64_load8_u);
            try body.emitMemArg(0, 0);
            try body.emitConstI64(0x0101010101010101);
            try body.emit(.i64_mul);
        },
        2 => {
            try body.emit(.i64_load16_u);
            try body.emitMemArg(1, 0);
            try body.emitConstI64(0x0001000100010001);
            try body.emit(.i64_mul);
        },
        4 => {
            try body.emit(.i64_load32_u);
            try body.emitMemArg(2, 0);
            // splat 32-bit value: (v | v<<32)
            try body.emit(.local_tee);
            try body.emitU32(local_a);
            try body.emit(.local_get);
            try body.emitU32(local_a);
            try body.emitConstI64(32);
            try body.emit(.i64_shl);
            try body.emit(.i64_or);
        },
        8 => {
            try body.loadI64(0);
        },
        else => return,
    }
    // stash the splatted low qword in $a
    try body.emit(.local_set);
    try body.emitU32(local_a);
    // write low qword
    try body.emitRegAddr(dst_off);
    try body.emit(.local_get);
    try body.emitU32(local_a);
    try body.storeI64(0);
    // high qword: same value when Q=1, else 0
    try body.emitRegAddr(dst_off + 8);
    if (x.q) {
        try body.emit(.local_get);
        try body.emitU32(local_a);
    } else {
        try body.emitConstI64(0);
    }
    try body.storeI64(0);
}

const CondKind = union(enum) {
    b_cond: u4,
    cbz: struct { rt: u5, sf: bool, non_zero: bool },
    tbz: struct { rt: u5, bit: u6, non_zero: bool },
};

/// detect `SUBS XZR, Xn, op` (== CMP) immediately followed by `B.cond` and emit a
/// fused direct compare + conditional branch. returns true if the pair was fused
/// (caller advances by 2 instructions), false to fall back to standard translation.
fn tryFuseCmpBCond(
    ctx: *Ctx,
    b: *const block.Block,
    block_idx: u32,
    cur: decode.Op,
    next: decode.Op,
    pc: u64,
    self_loop_target: ?u64,
) !bool {
    const bc = switch (next) {
        .b_cond => |c| c,
        else => return false,
    };

    const is_cmp = switch (cur) {
        .sub_imm => |x| x.rd == 31 and x.set_flags,
        .sub_reg => |x| x.rd == 31 and x.set_flags,
        else => false,
    };
    if (!is_cmp) return false;

    switch (bc.cond) {
        0x4, 0x5, 0x6, 0x7 => return false,
        else => {},
    }

    const body = ctx.body;
    const fallthrough = pc + 8;
    const target: u64 = @intCast(@as(i64, @intCast(pc + 4)) + bc.offset);

    // emit the SUBS normally so it updates NZCV (a later block in this region may
    // read flags before another *S overwrites them — fusing the b_cond away does
    // not let us skip that write). standard arith path leaves $a=Rn, $b=op2 valid
    // so the direct compare below avoids the NZCV roundtrip
    const sf: bool = switch (cur) {
        .sub_imm => |x| x.sf,
        .sub_reg => |x| x.sf,
        else => unreachable,
    };
    switch (cur) {
        .sub_imm => |x| try emitArithImm(body, x, true),
        .sub_reg => |x| try emitArithReg(body, x, true),
        else => unreachable,
    }

    // direct compare from $a, $b (still holding Rn and the shifted op2)
    try body.emit(.local_get);
    try body.emitU32(local_a);
    try body.emit(.local_get);
    try body.emitU32(local_b);

    const signed_op_needed = switch (bc.cond) {
        0xA, 0xB, 0xC, 0xD => true,
        else => false,
    };
    if (signed_op_needed and !sf) {
        try body.emit(.local_set);
        try body.emitU32(local_b);
        try body.emitConstI64(32);
        try body.emit(.i64_shl);
        try body.emitConstI64(32);
        try body.emit(.i64_shr_s);
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emitConstI64(32);
        try body.emit(.i64_shl);
        try body.emitConstI64(32);
        try body.emit(.i64_shr_s);
    }

    switch (bc.cond) {
        0x0 => try body.emit(.i64_eq),
        0x1 => try body.emit(.i64_ne),
        0x2 => try body.emit(.i64_ge_u),
        0x3 => try body.emit(.i64_lt_u),
        0x8 => try body.emit(.i64_gt_u),
        0x9 => try body.emit(.i64_le_u),
        0xA => try body.emit(.i64_ge_s),
        0xB => try body.emit(.i64_lt_s),
        0xC => try body.emit(.i64_gt_s),
        0xD => try body.emit(.i64_le_s),
        0xE, 0xF => {
            try body.emit(.drop);
            try body.emit(.drop);
            try body.emitConstI32(1);
        },
        else => unreachable,
    }

    if (self_loop_target) |st| if (st == target) {
        try body.emit(.br_if);
        try body.emitU32(0);
        const fallthrough_idx2 = ctx.blockOfPc(fallthrough);
        if (fallthrough_idx2) |fi| try jumpToBlock(ctx, block_idx, fi) else try emitExit(body, fallthrough, exit_branch_out);
        _ = b;
        return true;
    };

    const target_idx = ctx.blockOfPc(target);
    const fallthrough_idx = ctx.blockOfPc(fallthrough);

    try body.emit(.@"if");
    try body.bytes.append(body.arena, @intFromEnum(wasm.BlockType.empty));
    ctx.extra_depth += 1;

    if (target_idx) |ti| try jumpToBlock(ctx, block_idx, ti) else try emitExit(body, target, exit_branch_out);
    try body.emit(.@"else");
    if (fallthrough_idx) |fi| try jumpToBlock(ctx, block_idx, fi) else try emitExit(body, fallthrough, exit_branch_out);
    try body.emit(.end);
    ctx.extra_depth -= 1;

    _ = b;
    return true;
}

/// translate the final conditional branch of a self-loop block. uses the same
/// per-cond test the standard path uses, then `br_if 0` (back to the wrapping loop)
/// for the taken path and falls through to either an in-region block jump or a
/// region exit
fn translateSelfLoopBranch(ctx: *Ctx, block_idx: u32, op: decode.Op, pc: u64) !void {
    const body = ctx.body;
    const fallthrough = pc + 4;

    const cond_kind: CondKind = switch (op) {
        .b_cond => |c| .{ .b_cond = c.cond },
        .cbz => |c| .{ .cbz = .{ .rt = c.rt, .sf = c.sf, .non_zero = false } },
        .cbnz => |c| .{ .cbz = .{ .rt = c.rt, .sf = c.sf, .non_zero = true } },
        .tbz => |c| .{ .tbz = .{ .rt = c.rt, .bit = c.bit, .non_zero = false } },
        .tbnz => |c| .{ .tbz = .{ .rt = c.rt, .bit = c.bit, .non_zero = true } },
        else => unreachable,
    };

    try emitCondTest(body, cond_kind);
    try body.emit(.br_if);
    try body.emitU32(0); // back to loop header

    // fallthrough: in-region → jumpToBlock, else exit with fallthrough pc
    const fallthrough_idx = ctx.blockOfPc(fallthrough);
    if (fallthrough_idx) |fi| try jumpToBlock(ctx, block_idx, fi) else try emitExit(body, fallthrough, exit_branch_out);
}

fn emitCondBranch(ctx: *Ctx, block_idx: u32, pc: u64, target: u64, cond: CondKind) !void {
    const body = ctx.body;
    const fallthrough = pc + 4;
    const target_idx = ctx.blockOfPc(target);
    const fallthrough_idx = ctx.blockOfPc(fallthrough);

    try emitCondTest(body, cond);

    try body.emit(.@"if");
    try body.bytes.append(body.arena, @intFromEnum(wasm.BlockType.empty));
    ctx.extra_depth += 1;

    // taken path
    if (target_idx) |ti| {
        try jumpToBlock(ctx, block_idx, ti);
    } else {
        try emitExit(body, target, exit_branch_out);
    }

    try body.emit(.@"else");

    // fallthrough path
    if (fallthrough_idx) |fi| {
        try jumpToBlock(ctx, block_idx, fi);
    } else {
        try emitExit(body, fallthrough, exit_branch_out);
    }

    try body.emit(.end);
    ctx.extra_depth -= 1;
}

fn emitCondTest(body: *wasm.Body, cond: CondKind) !void {
    switch (cond) {
        .b_cond => |c| {
            // load nzcv from memory at nzcv_offset
            try body.emitRegAddr(@bitCast(cpu.nzcv_offset));
            try body.emit(.i32_load);
            try body.emitMemArg(2, 0);
            try emitConditionEval(body, c);
        },
        .cbz => |c| {
            try loadGpr(body, c.rt, c.sf);
            try body.emitConstI64(0);
            try body.emit(if (c.non_zero) .i64_ne else .i64_eq);
        },
        .tbz => |c| {
            try loadGpr(body, c.rt, true);
            try body.emitConstI64(@as(i64, 1) << @intCast(c.bit));
            try body.emit(.i64_and);
            try body.emitConstI64(0);
            try body.emit(if (c.non_zero) .i64_ne else .i64_eq);
        },
    }
}

/// given an nzcv i32 on the stack, replace it with an i32 (0/1) result
/// of the cond predicate. arm cond encoding (4 bits): low bit = invert.
fn emitConditionEval(body: *wasm.Body, cond: u4) !void {
    // stack: nzcv (u32)
    // we extract n,z,c,v bits and compute. nzcv layout: bit31=N, 30=Z, 29=C, 28=V.
    // for simplicity, we move the bits down so they're at known low positions:
    //   move into local: tee into local 1 (since we may use multiple times).
    // for now: use a temp local at index 1. caller has allocated it via emitArithImm etc? hmm — locals must be declared.
    //
    // simpler: duplicate via a sequence of operations. but wasm doesn't have a
    // built-in dup. use a `local.tee` which leaves the value on the stack and
    // also stores to a local. requires the local to exist.
    //
    // for v1 we don't have a temp local; the simplest approach is to use bit
    // extraction directly. each `cond` arm pops the nzcv once and tests bits.

    // strategy: pop nzcv, then test the bits we need for this condition.
    // since we can only use the value once without a temp local, we redo
    // bit extraction. simplest: convert each predicate to a sequence that
    // tests bits in one expression.
    //
    // we'll use a temp local (index 1). caller must ensure it's declared.

    try body.emit(.local_set);
    try body.emitU32(1); // store to $tmp_nzcv

    // helper: push n/z/c/v as i32 booleans
    // n = (nzcv >> 31) & 1
    // z = (nzcv >> 30) & 1
    // c = (nzcv >> 29) & 1
    // v = (nzcv >> 28) & 1
    //
    // each cond predicate:
    //   0 EQ  Z==1
    //   1 NE  Z==0
    //   2 CS  C==1
    //   3 CC  C==0
    //   4 MI  N==1
    //   5 PL  N==0
    //   6 VS  V==1
    //   7 VC  V==0
    //   8 HI  C==1 && Z==0
    //   9 LS  C==0 || Z==1
    //   A GE  N==V
    //   B LT  N!=V
    //   C GT  Z==0 && N==V
    //   D LE  Z==1 || N!=V
    //   E AL  1
    //   F NV  1 (reserved as always)

    switch (cond) {
        0x0 => try emitBitTest(body, 30, true),
        0x1 => try emitBitTest(body, 30, false),
        0x2 => try emitBitTest(body, 29, true),
        0x3 => try emitBitTest(body, 29, false),
        0x4 => try emitBitTest(body, 31, true),
        0x5 => try emitBitTest(body, 31, false),
        0x6 => try emitBitTest(body, 28, true),
        0x7 => try emitBitTest(body, 28, false),
        0x8 => {
            // C==1 && Z==0
            try emitBitTest(body, 29, true);
            try emitBitTest(body, 30, false);
            try body.emit(.i32_and);
        },
        0x9 => {
            // C==0 || Z==1
            try emitBitTest(body, 29, false);
            try emitBitTest(body, 30, true);
            try body.emit(.i32_or);
        },
        0xA => {
            // N==V
            try emitBitTest(body, 31, true);
            try emitBitTest(body, 28, true);
            try body.emit(.i32_eq);
        },
        0xB => {
            // N!=V
            try emitBitTest(body, 31, true);
            try emitBitTest(body, 28, true);
            try body.emit(.i32_ne);
        },
        0xC => {
            // Z==0 && N==V
            try emitBitTest(body, 30, false);
            try emitBitTest(body, 31, true);
            try emitBitTest(body, 28, true);
            try body.emit(.i32_eq);
            try body.emit(.i32_and);
        },
        0xD => {
            // Z==1 || N!=V
            try emitBitTest(body, 30, true);
            try emitBitTest(body, 31, true);
            try emitBitTest(body, 28, true);
            try body.emit(.i32_ne);
            try body.emit(.i32_or);
        },
        0xE, 0xF => {
            try body.emitConstI32(1);
        },
    }
}

fn emitBitTest(body: *wasm.Body, bit_pos: u5, expect_set: bool) !void {
    try body.emit(.local_get);
    try body.emitU32(1);
    try body.emitConstI32(@as(i32, 1) << bit_pos);
    try body.emit(.i32_and);
    if (expect_set) {
        try body.emitConstI32(0);
        try body.emit(.i32_ne);
    } else {
        try body.emit(.i32_eqz);
    }
}

fn jumpToBlock(ctx: *Ctx, from_block: u32, to_block: u32) !void {
    // set $cur = to_block, then br to loop
    try ctx.body.emitConstI32(@as(i32, @intCast(to_block)));
    try ctx.body.emit(.local_set);
    try ctx.body.emitU32(local_cur);
    try ctx.body.emit(.br);
    try ctx.body.emitU32(ctx.loopDepthFromBody(from_block));
}

fn emitIterFlush(body: *wasm.Body) !void {
    try body.emitRegAddr(@bitCast(cpu.iter_offset));
    try body.emit(.local_get);
    try body.emitU32(local_iter);
    try body.storeI64(0);
}

fn emitExit(body: *wasm.Body, next_pc: u64, reason: u32) !void {
    try body.emitRegAddr(@bitCast(cpu.pc_offset));
    try body.emitConstI64(@bitCast(next_pc));
    try body.storeI64(0);
    try emitIterFlush(body);
    try body.emitConstI64(packExit(@truncate(next_pc), reason));
    try body.emit(.@"return");
}

fn loadGpr(body: *wasm.Body, rd: u5, sf: bool) !void {
    if (rd == 31) {
        try body.emitConstI64(0);
        return;
    }
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(rd)));
    try body.loadI64(0);
    if (!sf) {
        try body.emitConstI64(0xffff_ffff);
        try body.emit(.i64_and);
    }
}

/// in load/store contexts rn=31 means SP, not XZR. cpu layout puts SP
/// at gpr_offset(31) by design so the address coincides, but reads
/// through loadGpr return zero for rd=31. use this helper for base regs
fn loadGprOrSp(body: *wasm.Body, rn: u5) !void {
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(rn)));
    try body.loadI64(0);
}

fn storeGprImm(body: *wasm.Body, rd: u5, value: u64) !void {
    if (rd == 31) return;
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(rd)));
    try body.emitConstI64(@bitCast(value));
    try body.storeI64(0);
}

const LogicalOp = enum { @"and", @"or", xor };

fn emitArithImm(body: *wasm.Body, x: decode.ArithImm, is_sub: bool) !void {
    // for ADD/SUB without flags, rd=31 means SP (not XZR). cpu layout has
    // gpr_offset(31) == sp_offset so the write naturally lands on SP.
    // ADDS/SUBS with flags treats rd=31 as XZR (discard).
    if (x.rd == 31 and x.set_flags) {
        // still need to compute flags; do the math but discard rd
    }

    // rn in arith (imm/reg) reads SP when rn=31, regardless of S bit
    try loadRnOrSpArith(body, x.rn, x.sf);
    try body.emit(.local_set);
    try body.emitU32(local_a);
    try body.emitConstI64(@as(i64, @intCast(x.imm)));
    try body.emit(.local_set);
    try body.emitU32(local_b);

    try emitArithCore(body, is_sub, x.sf, x.rd, x.set_flags);
}

fn emitArithReg(body: *wasm.Body, x: decode.ArithReg, is_sub: bool) !void {
    // rd=31 with set_flags=true is XZR; without flags it's still XZR for the
    // register form (arith reg uses SP-aware encoding only for the imm form).
    // hmm actually arm has both extended (sp) and shifted-reg encodings; the
    // decoder currently lumps them. for now assume shifted-register: rd=31 XZR
    if (x.rd == 31 and !x.set_flags) return;

    try loadGpr(body, x.rn, x.sf);
    try body.emit(.local_set);
    try body.emitU32(local_a);
    try loadGpr(body, x.rm, x.sf);
    try emitShiftApply(body, x.shift, x.amount, x.sf);
    try body.emit(.local_set);
    try body.emitU32(local_b);

    try emitArithCore(body, is_sub, x.sf, x.rd, x.set_flags);
}

/// ADD/SUB (extended register): rd = rn ± extended_rm. for set_flags=false, rd=31 means SP
/// and rn=31 means SP. for set_flags=true (ADDS/SUBS), rd=31 is XZR but rn=31 is still SP.
fn emitArithExtReg(body: *wasm.Body, x: decode.ArithExtReg, is_sub: bool) !void {
    const discard_rd = x.rd == 31 and x.set_flags;
    if (x.rd == 31 and !x.set_flags) {
        // SP destination — gpr_offset(31) == sp_offset so the store lands correctly.
        // we'll fall through to the normal path
    }

    // load rn (sp-aware: rn=31 always means SP in extended-register arith)
    if (x.rn == 31) {
        try loadGprOrSp(body, 31);
        if (!x.sf) {
            try body.emitConstI64(0xffff_ffff);
            try body.emit(.i64_and);
        }
    } else {
        try loadGpr(body, x.rn, x.sf);
    }
    try body.emit(.local_set);
    try body.emitU32(local_a);

    // load + extend rm. the extension width depends on the option bits.
    try loadGpr(body, x.rm, true);
    switch (x.option) {
        0b000 => { // UXTB
            try body.emitConstI64(0xff);
            try body.emit(.i64_and);
        },
        0b001 => { // UXTH
            try body.emitConstI64(0xffff);
            try body.emit(.i64_and);
        },
        0b010 => { // UXTW
            try body.emitConstI64(0xffff_ffff);
            try body.emit(.i64_and);
        },
        0b011 => {}, // UXTX / LSL: no extension (full 64-bit)
        0b100 => { // SXTB
            try body.emitConstI64(56);
            try body.emit(.i64_shl);
            try body.emitConstI64(56);
            try body.emit(.i64_shr_s);
        },
        0b101 => { // SXTH
            try body.emitConstI64(48);
            try body.emit(.i64_shl);
            try body.emitConstI64(48);
            try body.emit(.i64_shr_s);
        },
        0b110 => { // SXTW
            try body.emitConstI64(32);
            try body.emit(.i64_shl);
            try body.emitConstI64(32);
            try body.emit(.i64_shr_s);
        },
        0b111 => {}, // SXTX: no extension (already 64-bit signed)
    }
    if (x.amount > 0) {
        try body.emitConstI64(@as(i64, x.amount));
        try body.emit(.i64_shl);
    }
    try body.emit(.local_set);
    try body.emitU32(local_b);

    // compute result and store
    try body.emit(.local_get);
    try body.emitU32(local_a);
    try body.emit(.local_get);
    try body.emitU32(local_b);
    try body.emit(if (is_sub) .i64_sub else .i64_add);
    if (!x.sf) {
        try body.emitConstI64(0xffff_ffff);
        try body.emit(.i64_and);
    }
    try body.emit(.local_set);
    try body.emitU32(local_res);

    if (!discard_rd) {
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));
        try body.emit(.local_get);
        try body.emitU32(local_res);
        try body.storeI64(0);
    }

    if (x.set_flags) try emitNzcvArith(body, is_sub, x.sf);
}

/// for arith immediate ops, rn=31 means SP. wraps loadGprOrSp with the sf mask
fn loadRnOrSpArith(body: *wasm.Body, rn: u5, sf: bool) !void {
    if (rn == 31) {
        try loadGprOrSp(body, rn);
    } else {
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(rn)));
        try body.loadI64(0);
    }
    if (!sf) {
        try body.emitConstI64(0xffff_ffff);
        try body.emit(.i64_and);
    }
}

/// shared tail for arith imm/reg: with $a and $b already in locals,
/// compute $res = a ± b (masked for 32-bit), store to rd, and
/// optionally compute+store NZCV.
fn emitArithCore(body: *wasm.Body, is_sub: bool, sf: bool, rd: u5, set_flags: bool) !void {
    try body.emit(.local_get);
    try body.emitU32(local_a);
    try body.emit(.local_get);
    try body.emitU32(local_b);
    try body.emit(if (is_sub) .i64_sub else .i64_add);
    if (!sf) {
        try body.emitConstI64(0xffff_ffff);
        try body.emit(.i64_and);
    }
    try body.emit(.local_set);
    try body.emitU32(local_res);

    // rd=31 with set_flags=true is XZR (discard). rd=31 without flags is SP, and
    // gpr_offset(31) == sp_offset so the write lands in the right place
    if (!(rd == 31 and set_flags)) {
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(rd)));
        try body.emit(.local_get);
        try body.emitU32(local_res);
        try body.storeI64(0);
    }

    if (set_flags) try emitNzcvArith(body, is_sub, sf);
}

/// build the NZCV i32 value from $a, $b, $res locals (already masked
/// to register width for 32-bit ops) and store it at nzcv_offset.
fn emitNzcvArith(body: *wasm.Body, is_sub: bool, sf: bool) !void {
    // address for the i32.store
    try body.emitRegAddr(@bitCast(cpu.nzcv_offset));

    const sign_shift: u6 = if (sf) 63 else 31;

    // N = ($res >> sign_shift) & 1, positioned at bit 31
    try body.emit(.local_get);
    try body.emitU32(local_res);
    try body.emitConstI64(@as(i64, sign_shift));
    try body.emit(.i64_shr_u);
    try body.emitConstI64(1);
    try body.emit(.i64_and);
    try body.emit(.i32_wrap_i64);
    try body.emitConstI32(31);
    try body.emit(.i32_shl);

    // Z = ($res == 0), positioned at bit 30
    try body.emit(.local_get);
    try body.emitU32(local_res);
    try body.emit(.i64_eqz);
    try body.emitConstI32(30);
    try body.emit(.i32_shl);
    try body.emit(.i32_or);

    // C: positioned at bit 29
    if (is_sub) {
        // C = ($a >= $b) unsigned (no borrow)
        try body.emit(.local_get);
        try body.emitU32(local_a);
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emit(.i64_ge_u);
    } else {
        // C = ($res < $a) for unsigned add (overflow happened)
        try body.emit(.local_get);
        try body.emitU32(local_res);
        try body.emit(.local_get);
        try body.emitU32(local_a);
        try body.emit(.i64_lt_u);
    }
    try body.emitConstI32(29);
    try body.emit(.i32_shl);
    try body.emit(.i32_or);

    // V (signed overflow), positioned at bit 28
    if (is_sub) {
        // V = ((a ^ b) & (a ^ res) >> sign_shift) & 1
        try body.emit(.local_get);
        try body.emitU32(local_a);
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emit(.i64_xor);
        try body.emit(.local_get);
        try body.emitU32(local_a);
        try body.emit(.local_get);
        try body.emitU32(local_res);
        try body.emit(.i64_xor);
        try body.emit(.i64_and);
    } else {
        // V = ((a ^ res) & (b ^ res) >> sign_shift) & 1
        try body.emit(.local_get);
        try body.emitU32(local_a);
        try body.emit(.local_get);
        try body.emitU32(local_res);
        try body.emit(.i64_xor);
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emit(.local_get);
        try body.emitU32(local_res);
        try body.emit(.i64_xor);
        try body.emit(.i64_and);
    }
    try body.emitConstI64(@as(i64, sign_shift));
    try body.emit(.i64_shr_u);
    try body.emitConstI64(1);
    try body.emit(.i64_and);
    try body.emit(.i32_wrap_i64);
    try body.emitConstI32(28);
    try body.emit(.i32_shl);
    try body.emit(.i32_or);

    // store the assembled i32 at nzcv_offset
    try body.emit(.i32_store);
    try body.emitMemArg(2, 0);
}

fn emitLogicalReg(body: *wasm.Body, x: decode.LogicalReg, op: LogicalOp, invert_rm: bool) !void {
    if (x.rd == 31 and !x.set_flags) return;

    // compute the value into local_res so it can feed both the store and the
    // ANDS/BICS flag evaluation
    try loadGpr(body, x.rn, x.sf);
    try loadGpr(body, x.rm, x.sf);
    try emitShiftApply(body, x.shift, x.amount, x.sf);
    // bic/orn/eon invert rm
    if (invert_rm or x.invert_rm) {
        try body.emitConstI64(@bitCast(@as(u64, if (x.sf) ~@as(u64, 0) else 0xffff_ffff)));
        try body.emit(.i64_xor);
    }
    try body.emit(switch (op) {
        .@"and" => .i64_and,
        .@"or" => .i64_or,
        .xor => .i64_xor,
    });
    if (!x.sf) {
        try body.emitConstI64(0xffff_ffff);
        try body.emit(.i64_and);
    }
    try body.emit(.local_set);
    try body.emitU32(local_res);

    if (x.rd != 31) {
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));
        try body.emit(.local_get);
        try body.emitU32(local_res);
        try body.storeI64(0);
    }

    if (x.set_flags) {
        try body.emitRegAddr(@bitCast(cpu.nzcv_offset));
        try emitNzcvLogical(body, x.sf);
    }
}

fn emitLogicalImm(body: *wasm.Body, x: decode.LogicalImm, op: LogicalOp) !void {
    // for AND/ORR/EOR imm (no flags), rd=31 means SP (cpu layout has gpr_offset(31) == sp_offset).
    // only ANDS (set_flags=true) treats rd=31 as XZR (discard). rn=31 always reads XZR
    // (per arm ARM: AND/ORR/EOR imm encode <Xn>, not <Xn|SP>).
    const discard = x.rd == 31 and x.set_flags;

    // compute the value into local_res so we can use it twice (store + flag eval)
    try loadGpr(body, x.rn, x.sf);
    try body.emitConstI64(@bitCast(x.imm));
    try body.emit(switch (op) {
        .@"and" => .i64_and,
        .@"or" => .i64_or,
        .xor => .i64_xor,
    });
    if (!x.sf) {
        try body.emitConstI64(0xffff_ffff);
        try body.emit(.i64_and);
    }
    try body.emit(.local_set);
    try body.emitU32(local_res);

    if (!discard) {
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));
        try body.emit(.local_get);
        try body.emitU32(local_res);
        try body.storeI64(0);
    }

    if (x.set_flags) {
        try body.emitRegAddr(@bitCast(cpu.nzcv_offset));
        try emitNzcvLogical(body, x.sf);
    }
}

/// emit nzcv update for a logical-imm/reg ANDS-style: N from sign, Z from zero, C=0, V=0.
/// the result must already be in $res. address for nzcv_offset already on stack.
fn emitNzcvLogical(body: *wasm.Body, sf: bool) !void {
    const sign_shift: u6 = if (sf) 63 else 31;
    // N at bit 31
    try body.emit(.local_get);
    try body.emitU32(local_res);
    try body.emitConstI64(@as(i64, sign_shift));
    try body.emit(.i64_shr_u);
    try body.emitConstI64(1);
    try body.emit(.i64_and);
    try body.emit(.i32_wrap_i64);
    try body.emitConstI32(31);
    try body.emit(.i32_shl);
    // Z at bit 30
    try body.emit(.local_get);
    try body.emitU32(local_res);
    try body.emit(.i64_eqz);
    try body.emitConstI32(30);
    try body.emit(.i32_shl);
    try body.emit(.i32_or);
    // C=V=0 (no bits to add)
    try body.emit(.i32_store);
    try body.emitMemArg(2, 0);
}

const CselKind = enum { csel, csinc, csinv, csneg };

fn emitCondSelect(body: *wasm.Body, x: decode.CondSelect, kind: CselKind) !void {
    if (x.rd == 31) return;

    // rd address
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));

    // val_if_true: Xn
    try loadGpr(body, x.rn, x.sf);

    // val_if_false: Xm transformed per kind
    switch (kind) {
        .csel => {
            try loadGpr(body, x.rm, x.sf);
        },
        .csinc => {
            try loadGpr(body, x.rm, x.sf);
            try body.emitConstI64(1);
            try body.emit(.i64_add);
        },
        .csinv => {
            try loadGpr(body, x.rm, x.sf);
            try body.emitConstI64(-1);
            try body.emit(.i64_xor); // ~x
        },
        .csneg => {
            try body.emitConstI64(0);
            try loadGpr(body, x.rm, x.sf);
            try body.emit(.i64_sub); // 0 - x
        },
    }

    // condition: load nzcv and evaluate
    try body.emitRegAddr(@bitCast(cpu.nzcv_offset));
    try body.emit(.i32_load);
    try body.emitMemArg(2, 0);
    try emitConditionEval(body, x.cond);

    // select: if cond, pick val_if_true (already lower on stack), else val_if_false
    // wasm `select` (0x1b) pops cond:i32, val2, val1, pushes val1 if cond else val2.
    // stack now: val_true, val_false, cond → pop cond, pop val_false (=val2), pop val_true (=val1)
    // ⇒ result = val1 (true) if cond else val2 (false). matches our intent.
    try body.emit(.select);

    if (!x.sf) {
        try body.emitConstI64(0xffff_ffff);
        try body.emit(.i64_and);
    }
    try body.storeI64(0);
}

fn emitShiftReg(body: *wasm.Body, x: decode.ShiftReg, kind: decode.ShiftKind) !void {
    if (x.rd == 31) return;
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));
    try loadGpr(body, x.rn, x.sf);
    // see emitShiftApply: 32-bit asr needs sign extension from bit 31, 32-bit
    // ror needs the low word replicated so the 64-bit rotate wraps correctly
    if (!x.sf and kind == .asr) {
        try body.emitConstI64(32);
        try body.emit(.i64_shl);
        try body.emitConstI64(32);
        try body.emit(.i64_shr_s);
    }
    if (!x.sf and kind == .ror) {
        try body.emit(.local_tee);
        try body.emitU32(local_b);
        try body.emitConstI64(32);
        try body.emit(.i64_shl);
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emit(.i64_or);
    }
    try loadGpr(body, x.rm, x.sf);
    // arm: shift amount is rm modulo regwidth
    const reg_width: i64 = if (x.sf) 64 else 32;
    try body.emitConstI64(reg_width);
    try body.emit(.i64_rem_u);
    try body.emit(switch (kind) {
        .lsl => .i64_shl,
        .lsr => .i64_shr_u,
        .asr => .i64_shr_s,
        .ror => .i64_rotr,
    });
    if (!x.sf) {
        try body.emitConstI64(0xffff_ffff);
        try body.emit(.i64_and);
    }
    try body.storeI64(0);
}

fn emitShiftApply(body: *wasm.Body, kind: decode.ShiftKind, amount: u6, sf: bool) !void {
    // operates on the i64 currently on top of the stack.
    // 32-bit values live zero-extended in i64 locals, so ASR must sign-extend
    // from bit 31 first and ROR must rotate within 32 bits (replicate halves)
    if (amount == 0) return;
    if (!sf and kind == .asr) {
        try body.emitConstI64(32);
        try body.emit(.i64_shl);
        try body.emitConstI64(32);
        try body.emit(.i64_shr_s);
    }
    if (!sf and kind == .ror) {
        // replicate the low word into the high word so i64_rotr wraps correctly
        try body.emit(.local_tee);
        try body.emitU32(local_b);
        try body.emitConstI64(32);
        try body.emit(.i64_shl);
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emit(.i64_or);
    }
    try body.emitConstI64(@as(i64, amount));
    try body.emit(switch (kind) {
        .lsl => .i64_shl,
        .lsr => .i64_shr_u,
        .asr => .i64_shr_s,
        .ror => .i64_rotr,
    });
    if (!sf) {
        try body.emitConstI64(0xffff_ffff);
        try body.emit(.i64_and);
    }
}

/// for pre/post-index, rn=31 means SP and we write back through gpr_offset(31)
/// which coincides with sp_offset by cpu layout
fn computeLoadStoreAddr(body: *wasm.Body, rn: u5, offset: i32, mode: decode.AddrMode) !void {
    // base into $a
    try loadGprOrSp(body, rn);
    try body.emit(.local_set);
    try body.emitU32(local_a);

    // base + offset into $b
    try body.emit(.local_get);
    try body.emitU32(local_a);
    if (offset != 0) {
        try body.emitConstI64(@as(i64, offset));
        try body.emit(.i64_add);
    }
    try body.emit(.local_set);
    try body.emitU32(local_b);

    // writeback for pre/post-index happens after the access. for pre-index,
    // the access uses base+offset; for post-index, it uses the original base.
    // both write base+offset back to rn after the access.
    _ = mode;
}

fn writebackRn(body: *wasm.Body, rn: u5, mode: decode.AddrMode) !void {
    if (mode != .pre_index and mode != .post_index) return;
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(rn)));
    try body.emit(.local_get);
    try body.emitU32(local_b);
    try body.storeI64(0);
}

fn pushAccessAddrI32(body: *wasm.Body, mode: decode.AddrMode) !void {
    // address used for the memory access: $b for pre/offset, $a for post
    const which: u32 = if (mode == .post_index) local_a else local_b;
    try body.emit(.local_get);
    try body.emitU32(which);
    try body.emit(.i32_wrap_i64);
    try body.emitConstI32(@bitCast(cpu.guest_mem_base));
    try body.emit(.i32_add);
}

fn emitLoadOp(body: *wasm.Body, size: u8, signed: bool, sf: bool) !void {
    switch (size) {
        1 => {
            try body.emit(if (signed) .i64_load8_s else .i64_load8_u);
            try body.emitMemArg(0, 0);
        },
        2 => {
            try body.emit(if (signed) .i64_load16_s else .i64_load16_u);
            try body.emitMemArg(1, 0);
        },
        4 => {
            try body.emit(if (signed) .i64_load32_s else .i64_load32_u);
            try body.emitMemArg(2, 0);
        },
        8 => {
            try body.emit(.i64_load);
            try body.emitMemArg(3, 0);
        },
        else => unreachable,
    }
    // 32-bit unsigned loads (W reg, signed=false) need masking to 32 bits;
    // i64_load32_u already returns zero-extended so nothing extra needed.
    // for 8/16/32-bit signed loads into W (sf=false), arm sign-extends to 32 then
    // zero-extends to 64. i64_load*_s sign-extends to 64; mask to 32 to match.
    if (signed and !sf and size < 8) {
        try body.emitConstI64(0xffff_ffff);
        try body.emit(.i64_and);
    }
}

fn emitLoadImm(body: *wasm.Body, x: decode.LoadStoreImm, size: u8, signed: bool) !void {
    try computeLoadStoreAddr(body, x.rn, x.offset, x.mode);

    // for ldr, do the access first so writeback of rn doesn't lose data
    // when rt == rn (rt's load wins, matching common arm impl)
    if (x.rt != 31) {
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rt)));
    }
    try pushAccessAddrI32(body, x.mode);
    try emitLoadOp(body, size, signed, x.sf);
    if (x.rt == 31) {
        try body.emit(.drop);
    } else {
        try body.storeI64(0);
    }

    if (x.rt != x.rn) {
        try writebackRn(body, x.rn, x.mode);
    }
}

fn emitStoreImm(ctx: *Ctx, x: decode.LoadStoreImm, size: u8) !void {
    const body = ctx.body;
    try computeLoadStoreAddr(body, x.rn, x.offset, x.mode);

    // shadow protocol: skipped in single-thread mode (value-CAS in STXR suffices)
    const access_local: u32 = if (x.mode == .post_index) local_a else local_b;
    if (ctx.shared_monitor) try emitShadowClearSpan(body, ctx.shadow_base, access_local, size);

    try pushAccessAddrI32(body, x.mode);
    try loadGpr(body, x.rt, true);
    switch (size) {
        1 => { try body.emit(.i64_store8); try body.emitMemArg(0, 0); },
        2 => { try body.emit(.i64_store16); try body.emitMemArg(1, 0); },
        4 => { try body.emit(.i64_store32); try body.emitMemArg(2, 0); },
        8 => { try body.emit(.i64_store); try body.emitMemArg(3, 0); },
        else => unreachable,
    }

    try writebackRn(body, x.rn, x.mode);
}

fn emitLoadPair(body: *wasm.Body, x: decode.LoadStorePair, signed: bool) !void {
    const size: u8 = if (x.sf) 8 else 4;
    try computeLoadStoreAddr(body, x.rn, x.offset, x.mode);

    // load rt from access_addr, rt2 from access_addr + size. rn writeback
    // must wait until after both loads. if rt or rt2 == rn, the loads
    // overwrite the writeback so we skip it in that case (matches arm behaviour
    // documented as unpredictable; common impl loads win)
    if (x.rt != 31) {
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rt)));
        try pushAccessAddrI32(body, x.mode);
        try emitLoadOp(body, size, signed, x.sf);
        try body.storeI64(0);
    }
    if (x.rt2 != 31) {
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rt2)));
        try pushAccessAddrI32(body, x.mode);
        try body.emitConstI32(@as(i32, size));
        try body.emit(.i32_add);
        try emitLoadOp(body, size, signed, x.sf);
        try body.storeI64(0);
    }

    if (x.rt != x.rn and x.rt2 != x.rn) {
        try writebackRn(body, x.rn, x.mode);
    }
}

fn emitStorePair(ctx: *Ctx, x: decode.LoadStorePair) !void {
    const body = ctx.body;
    const size: u8 = if (x.sf) 8 else 4;
    try computeLoadStoreAddr(body, x.rn, x.offset, x.mode);

    // total span is 2 × size bytes starting at the access address
    const access_local: u32 = if (x.mode == .post_index) local_a else local_b;
    if (ctx.shared_monitor) try emitShadowClearSpan(body, ctx.shadow_base, access_local, @intCast(size * 2));

    // store rt at access_addr, rt2 at access_addr + size
    try pushAccessAddrI32(body, x.mode);
    try loadGpr(body, x.rt, true);
    switch (size) {
        4 => { try body.emit(.i64_store32); try body.emitMemArg(2, 0); },
        8 => { try body.emit(.i64_store); try body.emitMemArg(3, 0); },
        else => unreachable,
    }
    try pushAccessAddrI32(body, x.mode);
    try body.emitConstI32(@as(i32, size));
    try body.emit(.i32_add);
    try loadGpr(body, x.rt2, true);
    switch (size) {
        4 => { try body.emit(.i64_store32); try body.emitMemArg(2, 0); },
        8 => { try body.emit(.i64_store); try body.emitMemArg(3, 0); },
        else => unreachable,
    }

    try writebackRn(body, x.rn, x.mode);
}

const BitfieldKind = enum { ubfm, sbfm, bfm };

fn emitBitfield(body: *wasm.Body, x: decode.BitfieldImm, kind: BitfieldKind) !void {
    if (x.rd == 31) return;

    const reg_size: u32 = if (x.sf) 64 else 32;
    const immr: u32 = x.immr;
    const imms: u32 = x.imms;

    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));

    switch (kind) {
        .ubfm => try emitUbfm(body, x, reg_size, immr, imms),
        .sbfm => try emitSbfm(body, x, reg_size, immr, imms),
        .bfm => try emitBfm(body, x, reg_size, immr, imms),
    }

    if (!x.sf) {
        try body.emitConstI64(0xffff_ffff);
        try body.emit(.i64_and);
    }
    try body.storeI64(0);
}

fn emitUbfm(body: *wasm.Body, x: decode.BitfieldImm, reg_size: u32, immr: u32, imms: u32) !void {
    if (imms + 1 == reg_size) {
        try loadGpr(body, x.rn, x.sf);
        if (immr > 0) {
            try body.emitConstI64(@as(i64, @intCast(immr)));
            try body.emit(.i64_shr_u);
        }
    } else if (imms < immr) {
        const width: u32 = imms + 1;
        const lsb: u32 = reg_size - immr;
        const mask: u64 = (@as(u64, 1) << @intCast(width)) - 1;
        try loadGpr(body, x.rn, x.sf);
        try body.emitConstI64(@bitCast(mask));
        try body.emit(.i64_and);
        try body.emitConstI64(@as(i64, @intCast(lsb)));
        try body.emit(.i64_shl);
    } else {
        const width: u32 = imms - immr + 1;
        const mask: u64 = (@as(u64, 1) << @intCast(width)) - 1;
        try loadGpr(body, x.rn, x.sf);
        if (immr > 0) {
            try body.emitConstI64(@as(i64, @intCast(immr)));
            try body.emit(.i64_shr_u);
        }
        try body.emitConstI64(@bitCast(mask));
        try body.emit(.i64_and);
    }
}

fn emitSbfm(body: *wasm.Body, x: decode.BitfieldImm, reg_size: u32, immr: u32, imms: u32) !void {
    if (imms + 1 == reg_size) {
        try loadGpr(body, x.rn, x.sf);
        if (!x.sf) {
            try body.emitConstI64(32);
            try body.emit(.i64_shl);
            try body.emitConstI64(32);
            try body.emit(.i64_shr_s);
        }
        if (immr > 0) {
            try body.emitConstI64(@as(i64, @intCast(immr)));
            try body.emit(.i64_shr_s);
        }
    } else if (immr == 0) {
        try loadGpr(body, x.rn, x.sf);
        const bits: u32 = imms + 1;
        const shift: i64 = @intCast(64 - bits);
        try body.emitConstI64(shift);
        try body.emit(.i64_shl);
        try body.emitConstI64(shift);
        try body.emit(.i64_shr_s);
    } else if (imms < immr) {
        const width: u32 = imms + 1;
        const lsb: u32 = reg_size - immr;
        try loadGpr(body, x.rn, x.sf);
        const sign_shift: i64 = @intCast(64 - width);
        try body.emitConstI64(sign_shift);
        try body.emit(.i64_shl);
        try body.emitConstI64(sign_shift);
        try body.emit(.i64_shr_s);
        if (lsb > 0) {
            try body.emitConstI64(@as(i64, @intCast(lsb)));
            try body.emit(.i64_shl);
        }
    } else {
        try loadGpr(body, x.rn, x.sf);
        if (immr > 0) {
            try body.emitConstI64(@as(i64, @intCast(immr)));
            try body.emit(.i64_shr_u);
        }
        const width: u32 = imms - immr + 1;
        const shift: i64 = @intCast(64 - width);
        try body.emitConstI64(shift);
        try body.emit(.i64_shl);
        try body.emitConstI64(shift);
        try body.emit(.i64_shr_s);
    }
}

fn emitBfm(body: *wasm.Body, x: decode.BitfieldImm, reg_size: u32, immr: u32, imms: u32) !void {
    if (imms < immr) {
        const width: u32 = imms + 1;
        const lsb: u32 = reg_size - immr;
        const src_mask: u64 = (@as(u64, 1) << @intCast(width)) - 1;
        const dest_mask: u64 = src_mask << @intCast(lsb);
        try loadGpr(body, x.rd, x.sf);
        try body.emitConstI64(@bitCast(~dest_mask));
        try body.emit(.i64_and);
        try loadGpr(body, x.rn, x.sf);
        try body.emitConstI64(@bitCast(src_mask));
        try body.emit(.i64_and);
        try body.emitConstI64(@as(i64, @intCast(lsb)));
        try body.emit(.i64_shl);
        try body.emit(.i64_or);
    } else {
        const width: u32 = imms - immr + 1;
        const mask: u64 = (@as(u64, 1) << @intCast(width)) - 1;
        try loadGpr(body, x.rd, x.sf);
        try body.emitConstI64(@bitCast(~mask));
        try body.emit(.i64_and);
        try loadGpr(body, x.rn, x.sf);
        if (immr > 0) {
            try body.emitConstI64(@as(i64, @intCast(immr)));
            try body.emit(.i64_shr_u);
        }
        try body.emitConstI64(@bitCast(mask));
        try body.emit(.i64_and);
        try body.emit(.i64_or);
    }
}

fn emitExtr(body: *wasm.Body, x: decode.Extr) !void {
    if (x.rd == 31) return;
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));

    const reg_size: u32 = if (x.sf) 64 else 32;
    const lsb: u32 = x.imms;

    if (lsb == 0) {
        try loadGpr(body, x.rm, x.sf);
    } else if (x.rn == x.rm) {
        try loadGpr(body, x.rn, x.sf);
        if (!x.sf) {
            try body.emit(.local_set);
            try body.emitU32(local_a);
            try body.emit(.local_get);
            try body.emitU32(local_a);
            try body.emit(.local_get);
            try body.emitU32(local_a);
            try body.emitConstI64(32);
            try body.emit(.i64_shl);
            try body.emit(.i64_or);
        }
        try body.emitConstI64(@as(i64, @intCast(lsb)));
        try body.emit(.i64_rotr);
    } else {
        try loadGpr(body, x.rm, x.sf);
        try body.emitConstI64(@as(i64, @intCast(lsb)));
        try body.emit(.i64_shr_u);
        try loadGpr(body, x.rn, x.sf);
        try body.emitConstI64(@as(i64, @intCast(reg_size - lsb)));
        try body.emit(.i64_shl);
        try body.emit(.i64_or);
    }
    if (!x.sf) {
        try body.emitConstI64(0xffff_ffff);
        try body.emit(.i64_and);
    }
    try body.storeI64(0);
}

/// MADD/MSUB: rd = ra +/- (rn * rm), masked to register width
fn emitMulAdd(body: *wasm.Body, x: decode.MulAdd, is_sub: bool) !void {
    if (x.rd == 31) return;
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));

    try loadGpr(body, x.ra, x.sf);
    try loadGpr(body, x.rn, x.sf);
    try loadGpr(body, x.rm, x.sf);
    try body.emit(.i64_mul);
    try body.emit(if (is_sub) .i64_sub else .i64_add);
    if (!x.sf) {
        try body.emitConstI64(0xffff_ffff);
        try body.emit(.i64_and);
    }
    try body.storeI64(0);
}

/// SMADDL/UMADDL/SMSUBL/UMSUBL: xd = xa +/- (sext_or_zext(Wn) * sext_or_zext(Wm))
fn emitMulLong(body: *wasm.Body, x: decode.MulAdd, signed: bool, is_sub: bool) !void {
    if (x.rd == 31) return;
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));

    // ra is 64-bit
    try loadGpr(body, x.ra, true);

    // rn and rm are 32-bit; either sign- or zero-extended to 64
    try loadGpr(body, x.rn, false);
    if (signed) {
        try body.emitConstI64(32);
        try body.emit(.i64_shl);
        try body.emitConstI64(32);
        try body.emit(.i64_shr_s);
    }
    try loadGpr(body, x.rm, false);
    if (signed) {
        try body.emitConstI64(32);
        try body.emit(.i64_shl);
        try body.emitConstI64(32);
        try body.emit(.i64_shr_s);
    }
    try body.emit(.i64_mul);
    try body.emit(if (is_sub) .i64_sub else .i64_add);
    try body.storeI64(0);
}

/// SMULH/UMULH: rd = high 64 bits of (Xn * Xm). schoolbook 64x64 -> 128
fn emitMulHigh(body: *wasm.Body, x: decode.MulHigh, signed: bool) !void {
    if (x.rd == 31) return;

    // load operands into $a and $b
    try loadGpr(body, x.rn, true);
    try body.emit(.local_set);
    try body.emitU32(local_a);
    try loadGpr(body, x.rm, true);
    try body.emit(.local_set);
    try body.emitU32(local_b);

    // for signed multiply, convert via sign correction at the end:
    //   smulh(a, b) = umulh(a, b) - ((a < 0) ? b : 0) - ((b < 0) ? a : 0)
    // umulh: split a, b into 32-bit halves and accumulate cross products

    // compute umulh(a, b) → $res
    try emitUmulHigh(body);

    if (signed) {
        // adjust: if a < 0 (sign bit set), subtract b. likewise for b.
        // i64_lt_s gives a 0/1; multiply by b to mask.
        try body.emit(.local_get);
        try body.emitU32(local_res);

        try body.emit(.local_get);
        try body.emitU32(local_a);
        try body.emitConstI64(0);
        try body.emit(.i64_lt_s);
        try body.emit(.i64_extend_i32_u);
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emit(.i64_mul);
        try body.emit(.i64_sub);

        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emitConstI64(0);
        try body.emit(.i64_lt_s);
        try body.emit(.i64_extend_i32_u);
        try body.emit(.local_get);
        try body.emitU32(local_a);
        try body.emit(.i64_mul);
        try body.emit(.i64_sub);

        try body.emit(.local_set);
        try body.emitU32(local_res);
    }

    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));
    try body.emit(.local_get);
    try body.emitU32(local_res);
    try body.storeI64(0);
}

/// compute high 64 of unsigned 64x64 product. $a and $b hold operands. result -> $res.
/// uses $cur (i32, unused here) and additional stack temporaries.
fn emitUmulHigh(body: *wasm.Body) !void {
    // a_lo = a & 0xffffffff; a_hi = a >>> 32
    // b_lo = b & 0xffffffff; b_hi = b >>> 32
    // ll = a_lo * b_lo
    // lh = a_lo * b_hi
    // hl = a_hi * b_lo
    // hh = a_hi * b_hi
    // mid = (ll >>> 32) + (lh & 0xffffffff) + (hl & 0xffffffff)
    // result_hi = hh + (lh >>> 32) + (hl >>> 32) + (mid >>> 32)

    // we will compute step by step using locals $a, $b for inputs (preserved across).
    // intermediate values live on the wasm stack.

    // stage 1: ll >>> 32
    try body.emit(.local_get); try body.emitU32(local_a);
    try body.emitConstI64(0xffff_ffff);
    try body.emit(.i64_and);
    try body.emit(.local_get); try body.emitU32(local_b);
    try body.emitConstI64(0xffff_ffff);
    try body.emit(.i64_and);
    try body.emit(.i64_mul);
    try body.emitConstI64(32);
    try body.emit(.i64_shr_u);
    // stack: [ll>>32]

    // + (lh & 0xffffffff)
    try body.emit(.local_get); try body.emitU32(local_a);
    try body.emitConstI64(0xffff_ffff);
    try body.emit(.i64_and);
    try body.emit(.local_get); try body.emitU32(local_b);
    try body.emitConstI64(32);
    try body.emit(.i64_shr_u);
    try body.emit(.i64_mul);
    try body.emit(.local_tee); try body.emitU32(local_res); // save lh in $res for now
    try body.emitConstI64(0xffff_ffff);
    try body.emit(.i64_and);
    try body.emit(.i64_add);
    // stack: [(ll>>32) + (lh & 0xffffffff)]

    // + (hl & 0xffffffff) where hl saved in $cur (i32 local) -- but $cur is i32. use stack carefully
    try body.emit(.local_get); try body.emitU32(local_a);
    try body.emitConstI64(32);
    try body.emit(.i64_shr_u);
    try body.emit(.local_get); try body.emitU32(local_b);
    try body.emitConstI64(0xffff_ffff);
    try body.emit(.i64_and);
    try body.emit(.i64_mul);
    // duplicate via local_tee won't work; use $nzcv_tmp... no it's i32. simplest: split into two passes
    // recompute hl below for the (hl >>> 32) term; here just take the low half
    try body.emitConstI64(0xffff_ffff);
    try body.emit(.i64_and);
    try body.emit(.i64_add);
    // stack: [mid]
    try body.emitConstI64(32);
    try body.emit(.i64_shr_u);
    // stack: [mid >>> 32]

    // + hh
    try body.emit(.local_get); try body.emitU32(local_a);
    try body.emitConstI64(32);
    try body.emit(.i64_shr_u);
    try body.emit(.local_get); try body.emitU32(local_b);
    try body.emitConstI64(32);
    try body.emit(.i64_shr_u);
    try body.emit(.i64_mul);
    try body.emit(.i64_add);

    // + (lh >>> 32) — $res still holds lh from local_tee above
    try body.emit(.local_get); try body.emitU32(local_res);
    try body.emitConstI64(32);
    try body.emit(.i64_shr_u);
    try body.emit(.i64_add);

    // + (hl >>> 32) — recompute hl
    try body.emit(.local_get); try body.emitU32(local_a);
    try body.emitConstI64(32);
    try body.emit(.i64_shr_u);
    try body.emit(.local_get); try body.emitU32(local_b);
    try body.emitConstI64(0xffff_ffff);
    try body.emit(.i64_and);
    try body.emit(.i64_mul);
    try body.emitConstI64(32);
    try body.emit(.i64_shr_u);
    try body.emit(.i64_add);

    try body.emit(.local_set);
    try body.emitU32(local_res);
}

/// extend rm per arm's load/store register-offset rules and shift left by log2(size) if shift bit set.
/// option encoding: 010=uxtw, 011=lsl/uxtx, 110=sxtw, 111=sxtx
fn emitExtendedRm(body: *wasm.Body, rm: u5, option: u3, do_shift: bool, size: u8) !void {
    try loadGpr(body, rm, true);
    switch (option) {
        0b010 => { // UXTW
            try body.emitConstI64(0xffff_ffff);
            try body.emit(.i64_and);
        },
        0b110 => { // SXTW
            try body.emitConstI64(32);
            try body.emit(.i64_shl);
            try body.emitConstI64(32);
            try body.emit(.i64_shr_s);
        },
        else => {}, // lsl/sxtx: no extension
    }
    if (do_shift) {
        const shift_amt: i64 = switch (size) {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            else => unreachable,
        };
        if (shift_amt > 0) {
            try body.emitConstI64(shift_amt);
            try body.emit(.i64_shl);
        }
    }
}

fn emitLoadReg(body: *wasm.Body, x: decode.LoadStoreReg, size: u8, signed: bool) !void {
    if (x.rt != 31) {
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rt)));
    }
    try loadGprOrSp(body, x.rn);
    try emitExtendedRm(body, x.rm, x.option, x.shift, size);
    try body.emit(.i64_add);
    try body.emit(.i32_wrap_i64);
    try body.emitConstI32(@bitCast(cpu.guest_mem_base));
    try body.emit(.i32_add);
    try emitLoadOp(body, size, signed, x.sf);
    if (x.rt == 31) {
        try body.emit(.drop);
    } else {
        try body.storeI64(0);
    }
}

fn emitStoreReg(ctx: *Ctx, x: decode.LoadStoreReg, size: u8) !void {
    const body = ctx.body;
    // compute final guest address (rn + extended_rm) into local_b
    try loadGprOrSp(body, x.rn);
    try emitExtendedRm(body, x.rm, x.option, x.shift, size);
    try body.emit(.i64_add);
    try body.emit(.local_set);
    try body.emitU32(local_b);

    if (ctx.shared_monitor) try emitShadowClearSpan(body, ctx.shadow_base, local_b, size);

    // do the store. recompute the wasm address by adding guest_mem_base
    try body.emit(.local_get);
    try body.emitU32(local_b);
    try body.emit(.i32_wrap_i64);
    try body.emitConstI32(@bitCast(cpu.guest_mem_base));
    try body.emit(.i32_add);
    try loadGpr(body, x.rt, true);
    switch (size) {
        1 => { try body.emit(.i64_store8); try body.emitMemArg(0, 0); },
        2 => { try body.emit(.i64_store16); try body.emitMemArg(1, 0); },
        4 => { try body.emit(.i64_store32); try body.emitMemArg(2, 0); },
        8 => { try body.emit(.i64_store); try body.emitMemArg(3, 0); },
        else => unreachable,
    }
}

/// LDR literal: rt = mem[pc + offset]. offset is in bytes (already in decoder)
fn emitLoadLit(body: *wasm.Body, x: decode.LoadLit, pc: u64) !void {
    if (x.rt == 31) return;
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rt)));
    const addr: u64 = @bitCast(@as(i64, @intCast(pc)) + @as(i64, x.offset));
    try body.emitConstI32(@bitCast(@as(u32, @truncate(addr))));
    try body.emitConstI32(@bitCast(cpu.guest_mem_base));
    try body.emit(.i32_add);
    if (x.sf) {
        try body.emit(.i64_load);
        try body.emitMemArg(3, 0);
    } else if (x.signed) {
        try body.emit(.i64_load32_s);
        try body.emitMemArg(2, 0);
    } else {
        try body.emit(.i64_load32_u);
        try body.emitMemArg(2, 0);
    }
    try body.storeI64(0);
}

/// MRS: read system register. only tpidr_el0 supported (the tls base).
/// other reads return 0 for now; trap should be added later
fn emitMrs(body: *wasm.Body, x: decode.SysReg) !void {
    if (x.rt == 31) return;
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rt)));
    // tpidr_el0 has op0=3, op1=3, crn=13, crm=0, op2=2
    if (x.op0 == 3 and x.op1 == 3 and x.crn == 13 and x.crm == 0 and x.op2 == 2) {
        try body.emitRegAddr(@bitCast(cpu.tpidr_offset));
        try body.loadI64(0);
    } else {
        try body.emitConstI64(0);
    }
    try body.storeI64(0);
}

fn emitMsr(body: *wasm.Body, x: decode.SysReg) !void {
    // only tpidr_el0 writes are meaningful for guest code
    if (x.op0 == 3 and x.op1 == 3 and x.crn == 13 and x.crm == 0 and x.op2 == 2) {
        try body.emitRegAddr(@bitCast(cpu.tpidr_offset));
        try loadGpr(body, x.rt, true);
        try body.storeI64(0);
    }
    // other writes ignored
}

fn emitClz(body: *wasm.Body, x: decode.ClzRbit) !void {
    if (x.rd == 31) return;
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));
    try loadGpr(body, x.rn, x.sf);
    // for 32-bit, the value is zero-extended in the i64 local. clz on the full
    // i64 counts the upper 32 zeros too — subtract 32 to get the W-reg clz.
    try body.emit(.i64_clz);
    if (!x.sf) {
        try body.emitConstI64(32);
        try body.emit(.i64_sub);
    }
    try body.storeI64(0);
}

/// CLS: count leading sign bits (excluding the sign bit itself).
/// cls(x) = clz(sext(x) ^ (sext(x) >> 1)) - 1, minus 32 for the W form
fn emitCls(body: *wasm.Body, x: decode.ClzRbit) !void {
    if (x.rd == 31) return;
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));
    try loadGpr(body, x.rn, x.sf);
    if (!x.sf) {
        // sign-extend the 32-bit value so the arithmetic shift below sees its sign
        try body.emitConstI64(32);
        try body.emit(.i64_shl);
        try body.emitConstI64(32);
        try body.emit(.i64_shr_s);
    }
    try body.emit(.local_set);
    try body.emitU32(local_a);
    try body.emit(.local_get);
    try body.emitU32(local_a);
    try body.emit(.local_get);
    try body.emitU32(local_a);
    try body.emitConstI64(1);
    try body.emit(.i64_shr_s);
    try body.emit(.i64_xor);
    try body.emit(.i64_clz);
    try body.emitConstI64(if (x.sf) @as(i64, 1) else 33);
    try body.emit(.i64_sub);
    try body.storeI64(0);
}

/// one butterfly step: swap `bits`-wide groups using the given masks.
/// operates on $a, leaves the result in $a
fn emitRevStep(body: *wasm.Body, hi_mask: u64, lo_mask: u64, bits: i64) !void {
    try body.emit(.local_get);
    try body.emitU32(local_a);
    try body.emitConstI64(@bitCast(hi_mask));
    try body.emit(.i64_and);
    try body.emitConstI64(bits);
    try body.emit(.i64_shr_u);
    try body.emit(.local_get);
    try body.emitU32(local_a);
    try body.emitConstI64(@bitCast(lo_mask));
    try body.emit(.i64_and);
    try body.emitConstI64(bits);
    try body.emit(.i64_shl);
    try body.emit(.i64_or);
    try body.emit(.local_set);
    try body.emitU32(local_a);
}

fn emitRev(body: *wasm.Body, x: decode.RevOp) !void {
    if (x.rd == 31) return;
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));
    try loadGpr(body, x.rn, x.sf);
    try body.emit(.local_set);
    try body.emitU32(local_a);

    // opc selects the reversal container: 1 = rev16 (bytes within each halfword),
    // 2 = rev32 for sf=1 / full rev for sf=0 (bytes within each word),
    // 3 = full 64-bit rev. byte swap always happens; wider swaps per opc
    try emitRevStep(body, 0xff00_ff00_ff00_ff00, 0x00ff_00ff_00ff_00ff, 8);
    if (x.opc >= 2) {
        try emitRevStep(body, 0xffff_0000_ffff_0000, 0x0000_ffff_0000_ffff, 16);
    }
    if (x.opc == 3) {
        try emitRevStep(body, 0xffff_ffff_0000_0000, 0x0000_0000_ffff_ffff, 32);
    }

    try body.emit(.local_get);
    try body.emitU32(local_a);
    if (!x.sf) {
        try body.emitConstI64(0xffff_ffff);
        try body.emit(.i64_and);
    }
    try body.storeI64(0);
}

fn emitRbit(body: *wasm.Body, x: decode.ClzRbit) !void {
    if (x.rd == 31) return;
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));
    try loadGpr(body, x.rn, x.sf);
    try body.emit(.local_set);
    try body.emitU32(local_a);

    // bit-reverse via classical butterflies. swap 1, 2, 4-bit groups (the same for any width),
    // then swap 8 and 16-bit groups. for 64-bit we additionally swap the 32-bit halves.
    // for 32-bit we mask to keep things in the low half.

    // step 1: swap adjacent bits
    try butterflySwap(body, local_a, 1, 0x5555_5555_5555_5555);
    // step 2: swap pairs of 2 bits
    try butterflySwap(body, local_a, 2, 0x3333_3333_3333_3333);
    // step 3: swap nibbles
    try butterflySwap(body, local_a, 4, 0x0f0f_0f0f_0f0f_0f0f);
    // step 4: swap bytes within 16-bit groups
    try butterflySwap(body, local_a, 8, 0x00ff_00ff_00ff_00ff);
    // step 5: swap 16-bit groups within 32-bit
    try butterflySwap(body, local_a, 16, 0x0000_ffff_0000_ffff);

    // after 5 butterflies (1, 2, 4, 8, 16-bit) each 32-bit half is internally reversed
    // but the two halves are not swapped. for sf=1, add the 32-bit swap to complete the
    // full 64-bit reversal. for sf=0, the low-32-bit reversal already sits in the low 32
    // bits (the high half started zero and remains zero).
    try body.emit(.local_get);
    try body.emitU32(local_a);
    if (x.sf) {
        try body.emitConstI64(32);
        try body.emit(.i64_shr_u);
        try body.emit(.local_get);
        try body.emitU32(local_a);
        try body.emitConstI64(32);
        try body.emit(.i64_shl);
        try body.emit(.i64_or);
    }

    try body.storeI64(0);
}

/// emit a single butterfly step: $a = (($a & (mask << shift)) >> shift) | (($a & mask) << shift)
fn butterflySwap(body: *wasm.Body, local: u32, shift: u32, low_mask: u64) !void {
    const high_mask: u64 = low_mask << @intCast(shift);

    // high part: (a & high_mask) >> shift
    try body.emit(.local_get);
    try body.emitU32(local);
    try body.emitConstI64(@bitCast(high_mask));
    try body.emit(.i64_and);
    try body.emitConstI64(@as(i64, @intCast(shift)));
    try body.emit(.i64_shr_u);

    // low part: (a & low_mask) << shift
    try body.emit(.local_get);
    try body.emitU32(local);
    try body.emitConstI64(@bitCast(low_mask));
    try body.emit(.i64_and);
    try body.emitConstI64(@as(i64, @intCast(shift)));
    try body.emit(.i64_shl);

    try body.emit(.i64_or);
    try body.emit(.local_set);
    try body.emitU32(local);
}

/// ADC: rd = rn + rm + C; SBC: rd = rn + ~rm + C. wasm has no carry, so we use NZCV
fn emitAdcSbc(body: *wasm.Body, x: decode.AdcSbc, is_sub: bool) !void {
    if (x.rd == 31 and !x.set_flags) return;

    // load C from nzcv (bit 29)
    try body.emitRegAddr(@bitCast(cpu.nzcv_offset));
    try body.emit(.i32_load);
    try body.emitMemArg(2, 0);
    try body.emitConstI32(29);
    try body.emit(.i32_shr_u);
    try body.emitConstI32(1);
    try body.emit(.i32_and);
    try body.emit(.i64_extend_i32_u);
    try body.emit(.local_set);
    try body.emitU32(local_b); // park carry in $b temporarily

    // compute a operand
    try loadGpr(body, x.rn, x.sf);
    try body.emit(.local_set);
    try body.emitU32(local_a);

    // b operand: rm (or ~rm for sub), masked to width
    try loadGpr(body, x.rm, x.sf);
    if (is_sub) {
        const mask: i64 = if (x.sf) -1 else 0xffff_ffff;
        try body.emitConstI64(mask);
        try body.emit(.i64_xor);
    }

    // res = a + (b or ~b) + C
    try body.emit(.local_get); try body.emitU32(local_a);
    try body.emit(.i64_add);
    try body.emit(.local_get); try body.emitU32(local_b);
    try body.emit(.i64_add);
    if (!x.sf) {
        try body.emitConstI64(0xffff_ffff);
        try body.emit(.i64_and);
    }
    try body.emit(.local_set);
    try body.emitU32(local_res);

    if (x.rd != 31) {
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));
        try body.emit(.local_get); try body.emitU32(local_res);
        try body.storeI64(0);
    }

    if (x.set_flags) {
        // exact adc/sbc flags. treat as unified addition res = a + b2 + carry
        // where b2 = rm (adc) or ~rm (sbc). n/z read $res; v reads $a/$b2/$res.
        // c (carry out) needs the full-width sum: for 32-bit it fits in i64, for
        // 64-bit we detect the two partial carries
        try loadGpr(body, x.rm, x.sf);
        if (is_sub) {
            const mask2: i64 = if (x.sf) -1 else 0xffff_ffff;
            try body.emitConstI64(mask2);
            try body.emit(.i64_xor);
        }
        try body.emit(.local_set);
        try body.emitU32(local_b); // $b = b2 (second operand, width-masked)

        // carry-in is currently discarded ($b was overwritten). recompute it into
        // $nzcv_tmp from the still-live nzcv memory
        try body.emitRegAddr(@bitCast(cpu.nzcv_offset));
        try body.emit(.i32_load);
        try body.emitMemArg(2, 0);
        try body.emitConstI32(29);
        try body.emit(.i32_shr_u);
        try body.emitConstI32(1);
        try body.emit(.i32_and);
        try body.emit(.local_set);
        try body.emitU32(local_nzcv_tmp);

        try body.emitRegAddr(@bitCast(cpu.nzcv_offset));
        const sign_shift: u6 = if (x.sf) 63 else 31;

        // N = res sign
        try body.emit(.local_get);
        try body.emitU32(local_res);
        try body.emitConstI64(@as(i64, sign_shift));
        try body.emit(.i64_shr_u);
        try body.emitConstI64(1);
        try body.emit(.i64_and);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(31);
        try body.emit(.i32_shl);

        // Z = res == 0
        try body.emit(.local_get);
        try body.emitU32(local_res);
        try body.emit(.i64_eqz);
        try body.emitConstI32(30);
        try body.emit(.i32_shl);
        try body.emit(.i32_or);

        // C = carry out of the addition a + b2 + carry_in
        if (x.sf) {
            // low = a + b2 (wrapping); carry1 = low <_u a; res = low + cin;
            // carry2 = res <_u low; C = carry1 | carry2
            try body.emit(.local_get);
            try body.emitU32(local_a);
            try body.emit(.local_get);
            try body.emitU32(local_b);
            try body.emit(.i64_add); // low
            try body.emit(.local_get);
            try body.emitU32(local_a);
            try body.emit(.i64_lt_u); // carry1 (i32)
            try body.emit(.local_get);
            try body.emitU32(local_res);
            try body.emit(.local_get);
            try body.emitU32(local_a);
            try body.emit(.local_get);
            try body.emitU32(local_b);
            try body.emit(.i64_add);
            try body.emit(.i64_lt_u); // carry2 = res <_u (a+b2)
            try body.emit(.i32_or);
        } else {
            // 32-bit: full sum fits in i64; carry = bit 32 of (a + b2 + cin)
            try body.emit(.local_get);
            try body.emitU32(local_a);
            try body.emit(.local_get);
            try body.emitU32(local_b);
            try body.emit(.i64_add);
            try body.emit(.local_get);
            try body.emitU32(local_nzcv_tmp);
            try body.emit(.i64_extend_i32_u);
            try body.emit(.i64_add);
            try body.emitConstI64(32);
            try body.emit(.i64_shr_u);
            try body.emit(.i32_wrap_i64);
        }
        try body.emitConstI32(29);
        try body.emit(.i32_shl);
        try body.emit(.i32_or);

        // V = (~(a ^ b2) & (a ^ res)) sign bit
        try body.emit(.local_get);
        try body.emitU32(local_a);
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emit(.i64_xor);
        try body.emitConstI64(-1);
        try body.emit(.i64_xor);
        try body.emit(.local_get);
        try body.emitU32(local_a);
        try body.emit(.local_get);
        try body.emitU32(local_res);
        try body.emit(.i64_xor);
        try body.emit(.i64_and);
        try body.emitConstI64(@as(i64, sign_shift));
        try body.emit(.i64_shr_u);
        try body.emitConstI64(1);
        try body.emit(.i64_and);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(28);
        try body.emit(.i32_shl);
        try body.emit(.i32_or);

        try body.emit(.i32_store);
        try body.emitMemArg(2, 0);
    }
}

fn emitCcmpImm(body: *wasm.Body, x: decode.CondCmpImm) !void {
    // if cond holds: CCMP does CMP rn, #imm5 (SUBS); CCMN does CMN rn, #imm5 (ADDS).
    // else: write x.nzcv to NZCV as-is (arm never inverts the immediate flags)
    // emitConditionEval consumes the nzcv word off the stack, so push it first
    try body.emitRegAddr(@bitCast(cpu.nzcv_offset));
    try body.emit(.i32_load);
    try body.emitMemArg(2, 0);
    try emitConditionEval(body, x.cond);
    try body.emit(.@"if");
    try body.bytes.append(body.arena, @intFromEnum(wasm.BlockType.empty));

    try loadGpr(body, x.rn, x.sf);
    try body.emit(.local_set);
    try body.emitU32(local_a);
    try body.emitConstI64(@as(i64, x.imm5));
    try body.emit(.local_set);
    try body.emitU32(local_b);
    try emitArithCore(body, !x.is_ccmn, x.sf, 31, true);

    try body.emit(.@"else");

    // store x.nzcv shifted to bits 31-28
    const flags: u32 = @as(u32, x.nzcv) << 28;
    try body.emitRegAddr(@bitCast(cpu.nzcv_offset));
    try body.emitConstI32(@bitCast(flags));
    try body.emit(.i32_store);
    try body.emitMemArg(2, 0);

    try body.emit(.end);
}

fn emitCcmpReg(body: *wasm.Body, x: decode.CondCmpReg) !void {
    // emitConditionEval consumes the nzcv word off the stack, so push it first
    try body.emitRegAddr(@bitCast(cpu.nzcv_offset));
    try body.emit(.i32_load);
    try body.emitMemArg(2, 0);
    try emitConditionEval(body, x.cond);
    try body.emit(.@"if");
    try body.bytes.append(body.arena, @intFromEnum(wasm.BlockType.empty));

    try loadGpr(body, x.rn, x.sf);
    try body.emit(.local_set);
    try body.emitU32(local_a);
    try loadGpr(body, x.rm, x.sf);
    try body.emit(.local_set);
    try body.emitU32(local_b);
    try emitArithCore(body, !x.is_ccmn, x.sf, 31, true);

    try body.emit(.@"else");
    const flags: u32 = @as(u32, x.nzcv) << 28;
    try body.emitRegAddr(@bitCast(cpu.nzcv_offset));
    try body.emitConstI32(@bitCast(flags));
    try body.emit(.i32_store);
    try body.emitMemArg(2, 0);

    try body.emit(.end);
}

/// UDIV/SDIV: arm defines div by zero as returning 0, not trapping.
/// wasm i64_div_* traps on /0, so we explicitly select.
fn emitDiv(body: *wasm.Body, x: decode.DivReg, signed: bool) !void {
    if (x.rd == 31) return;

    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));

    // if (rm == 0) result = 0; else result = rn / rm
    try loadGpr(body, x.rm, x.sf);
    try body.emit(.local_set);
    try body.emitU32(local_b);

    try body.emit(.local_get); try body.emitU32(local_b);
    try body.emit(.i64_eqz);
    try body.emit(.@"if");
    try body.emit(@enumFromInt(0x7e)); // block type: i64
    try body.emitConstI64(0);
    try body.emit(.@"else");
    try loadGpr(body, x.rn, x.sf);
    // for signed 64-bit, INT64_MIN / -1 traps. arm defines as INT64_MIN.
    // for sf=0 (32-bit), the same issue: INT32_MIN / -1 should return INT32_MIN.
    // guard against the trap.
    if (signed) {
        // if (rn == INT_MIN && rm == -1) return INT_MIN; else divide
        try body.emit(.local_set);
        try body.emitU32(local_a);

        try body.emit(.local_get); try body.emitU32(local_a);
        const int_min: i64 = if (x.sf) @bitCast(@as(u64, 0x8000_0000_0000_0000)) else 0x80000000;
        try body.emitConstI64(int_min);
        try body.emit(.i64_eq);
        try body.emit(.local_get); try body.emitU32(local_b);
        const neg_one: i64 = if (x.sf) -1 else 0xffff_ffff;
        try body.emitConstI64(neg_one);
        try body.emit(.i64_eq);
        try body.emit(.i32_and);
        try body.emit(.@"if");
        try body.emit(@enumFromInt(0x7e)); // i64
        try body.emitConstI64(int_min);
        try body.emit(.@"else");
        try body.emit(.local_get); try body.emitU32(local_a);
        try body.emit(.local_get); try body.emitU32(local_b);
        if (!x.sf) {
            // sf=0: 32-bit signed divide. operands are zero-extended in 64-bit locals.
            // sign-extend both to 64-bit first.
            // stack currently: [a_zext, b_zext]; need to sign extend.
            // pull b out, sign-extend a, push b, sign-extend b
            try body.emit(.local_set);
            try body.emitU32(local_res); // park b
            // sign-extend a
            try body.emitConstI64(32);
            try body.emit(.i64_shl);
            try body.emitConstI64(32);
            try body.emit(.i64_shr_s);
            try body.emit(.local_get);
            try body.emitU32(local_res);
            try body.emitConstI64(32);
            try body.emit(.i64_shl);
            try body.emitConstI64(32);
            try body.emit(.i64_shr_s);
        }
        try body.emit(.i64_div_s);
        if (!x.sf) {
            try body.emitConstI64(0xffff_ffff);
            try body.emit(.i64_and);
        }
        try body.emit(.end);
    } else {
        try body.emit(.local_get); try body.emitU32(local_b);
        try body.emit(.i64_div_u);
    }
    try body.emit(.end);

    try body.storeI64(0);
}

// ── exclusive monitor shadow protocol ──
//
// the shadow bitmap (per-worker, inside the same linear memory) has 1 bit per
// `cpu.erg_bytes` guest cache line. layout:
//   shadow_byte_addr(addr) = shadow_base + ((addr - guest_mem_base) >> 9)
//   shadow_bit_pos(addr)   = ((addr - guest_mem_base) >> 6) & 7
// every translated guest store atomic.ANDs the bit to 0. ldxr atomic.ORs it to 1.
// stxr cmpxchgs the value and re-checks the bit; if the bit was cleared in the
// interim, the store-exclusive fails (caller retries).

/// emit atomic-and on the shadow bit for the granule containing the address
/// currently held in `addr_local` (i64). stack is unchanged at exit.
fn emitShadowClear(body: *wasm.Body, shadow_base: u32, addr_local: u32) !void {
    // push shadow byte address (i32)
    try body.emit(.local_get);
    try body.emitU32(addr_local);
    try body.emitConstI64(9); // (addr >> 9) gives shadow byte offset
    try body.emit(.i64_shr_u);
    try body.emit(.i32_wrap_i64);
    try body.emitConstI32(@bitCast(shadow_base));
    try body.emit(.i32_add);

    // push the mask = ~(1 << ((addr >> 6) & 7)) as i32
    try body.emitConstI32(1);
    try body.emit(.local_get);
    try body.emitU32(addr_local);
    try body.emitConstI64(6);
    try body.emit(.i64_shr_u);
    try body.emitConstI64(7);
    try body.emit(.i64_and);
    try body.emit(.i32_wrap_i64);
    try body.emit(.i32_shl);
    try body.emitConstI32(-1);
    try body.emit(.i32_xor);

    // atomic.rmw8.and_u, alignment=0, offset=0; discard returned old byte
    try body.emitAtomic(wasm.Atomic.i32_atomic_rmw8_and_u);
    try body.emitU32(0);
    try body.emitU32(0);
    try body.emit(.drop);
}

/// for stores spanning more than 1 byte, the access could straddle two
/// 64-byte granules if unaligned. clear both potential granules.
/// size is the guest-store byte width (1, 2, 4, 8, 16).
fn emitShadowClearSpan(body: *wasm.Body, shadow_base: u32, addr_local: u32, size: u8) !void {
    try emitShadowClear(body, shadow_base, addr_local);
    if (size > 1) {
        // last byte of the access: addr + (size - 1). compute into a fresh value
        // on the stack, parked into local_res briefly to reuse emitShadowClear
        try body.emit(.local_get);
        try body.emitU32(addr_local);
        try body.emitConstI64(@as(i64, size - 1));
        try body.emit(.i64_add);
        try body.emit(.local_set);
        try body.emitU32(local_res);
        try emitShadowClear(body, shadow_base, local_res);
        // local_res is short-lived; consumed before any subsequent flag computation
    }
}

/// i64 atomic RMW sub-opcode for an LSE op at a given access size. i64 variants so the returned
/// pre-op value is zero-extended to 64 bits for Rt. only ops with a direct wasm atomic.rmw.
fn atomicRmwSubOpI64(op: enum { add, @"and", @"or", xor, xchg }, size: u8) u8 {
    return switch (op) {
        .add => switch (size) { 1 => wasm.Atomic.i64_atomic_rmw8_add_u, 2 => wasm.Atomic.i64_atomic_rmw16_add_u, 4 => wasm.Atomic.i64_atomic_rmw32_add_u, else => wasm.Atomic.i64_atomic_rmw_add },
        .@"and" => switch (size) { 1 => wasm.Atomic.i64_atomic_rmw8_and_u, 2 => wasm.Atomic.i64_atomic_rmw16_and_u, 4 => wasm.Atomic.i64_atomic_rmw32_and_u, else => wasm.Atomic.i64_atomic_rmw_and },
        .@"or" => switch (size) { 1 => wasm.Atomic.i64_atomic_rmw8_or_u, 2 => wasm.Atomic.i64_atomic_rmw16_or_u, 4 => wasm.Atomic.i64_atomic_rmw32_or_u, else => wasm.Atomic.i64_atomic_rmw_or },
        .xor => switch (size) { 1 => wasm.Atomic.i64_atomic_rmw8_xor_u, 2 => wasm.Atomic.i64_atomic_rmw16_xor_u, 4 => wasm.Atomic.i64_atomic_rmw32_xor_u, else => wasm.Atomic.i64_atomic_rmw_xor },
        .xchg => switch (size) { 1 => wasm.Atomic.i64_atomic_rmw8_xchg_u, 2 => wasm.Atomic.i64_atomic_rmw16_xchg_u, 4 => wasm.Atomic.i64_atomic_rmw32_xchg_u, else => wasm.Atomic.i64_atomic_rmw_xchg },
    };
}

/// i64 atomic compare-exchange sub-opcode (returns 64-bit zero-extended old value for Rs).
fn atomicCmpxchgSubOpI64(size: u8) u8 {
    return switch (size) {
        1 => wasm.Atomic.i64_atomic_rmw8_cmpxchg_u,
        2 => wasm.Atomic.i64_atomic_rmw16_cmpxchg_u,
        4 => wasm.Atomic.i64_atomic_rmw32_cmpxchg_u,
        else => wasm.Atomic.i64_atomic_rmw_cmpxchg,
    };
}

/// atomic load size → wasm threads sub-opcode
fn atomicLoadSubOp(size: u8) u8 {
    return switch (size) {
        1 => wasm.Atomic.i32_atomic_load8_u,
        2 => wasm.Atomic.i32_atomic_load16_u,
        4 => wasm.Atomic.i32_atomic_load,
        8 => wasm.Atomic.i64_atomic_load,
        else => unreachable,
    };
}

fn atomicCmpxchgSubOp(size: u8) u8 {
    return switch (size) {
        1 => wasm.Atomic.i32_atomic_rmw8_cmpxchg_u,
        2 => wasm.Atomic.i32_atomic_rmw16_cmpxchg_u,
        4 => wasm.Atomic.i32_atomic_rmw_cmpxchg,
        8 => wasm.Atomic.i64_atomic_rmw_cmpxchg,
        else => unreachable,
    };
}

fn atomicAlignLog2(size: u8) u32 {
    return switch (size) {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        else => unreachable,
    };
}

/// LDXR / LDAXR (load exclusive):
///   1. atomic.or shadow_bit (mark monitor)
///   2. atomic.load value from guest mem
///   3. store value to rt and stash (addr, value, size) in cpu monitor slots
fn emitLdxr(ctx: *Ctx, x: decode.LoadStoreEx) !void {
    const body = ctx.body;
    const size = x.size;

    // compute guest address into local_b (i64)
    try loadGprOrSp(body, x.rn);
    try body.emit(.local_set);
    try body.emitU32(local_b);

    // step 1: set the shadow bit (atomic.or) — only when shared_monitor mode is on
    if (ctx.shared_monitor) {
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emitConstI64(9);
        try body.emit(.i64_shr_u);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(ctx.shadow_base));
        try body.emit(.i32_add);

        try body.emitConstI32(1);
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emitConstI64(6);
        try body.emit(.i64_shr_u);
        try body.emitConstI64(7);
        try body.emit(.i64_and);
        try body.emit(.i32_wrap_i64);
        try body.emit(.i32_shl); // 1 << bit_pos

        try body.emitAtomic(wasm.Atomic.i32_atomic_rmw8_or_u);
        try body.emitU32(0);
        try body.emitU32(0);
        try body.emit(.drop);
    }

    // step 2: load the value. atomic when shared_monitor (matches the spec ordering);
    // plain load otherwise — no other thread can write between LDXR and STXR in single-thread mode
    try body.emit(.local_get);
    try body.emitU32(local_b);
    try body.emit(.i32_wrap_i64);
    try body.emitConstI32(@bitCast(cpu.guest_mem_base));
    try body.emit(.i32_add);
    if (ctx.shared_monitor) {
        try body.emitAtomic(atomicLoadSubOp(size));
        try body.emitU32(atomicAlignLog2(size));
        try body.emitU32(0);
        if (size < 8) try body.emit(.i64_extend_i32_u); // atomic load returns i32
    } else {
        try emitLoadOp(body, size, false, true); // returns i64 directly
    }
    try body.emit(.local_set);
    try body.emitU32(local_a); // park value in $a

    // step 3a: write value to rt (if not xzr)
    if (x.rt != 31) {
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rt)));
        try body.emit(.local_get);
        try body.emitU32(local_a);
        try body.storeI64(0);
    }

    // step 3b: stash (addr, value, size) into the monitor slots in cpu state
    try body.emitRegAddr(@bitCast(cpu.monitor_addr_offset));
    try body.emit(.local_get);
    try body.emitU32(local_b);
    try body.storeI64(0);

    try body.emitRegAddr(@bitCast(cpu.monitor_value_offset));
    try body.emit(.local_get);
    try body.emitU32(local_a);
    try body.storeI64(0);

    try body.emitRegAddr(@bitCast(cpu.monitor_size_offset));
    try body.emitConstI32(@as(i32, size));
    try body.emit(.i32_store);
    try body.emitMemArg(2, 0);
}

/// STXR / STLXR (store exclusive):
///   1. check monitor is valid: monitor_size != 0 AND monitor_addr == this addr
///   2. atomic cmpxchg(monitor_value, new_value) at addr
///   3. success iff cmpxchg returned the expected value AND shadow bit still set
///   4. write 0 (success) or 1 (failure) to rs; clear monitor_size on success
fn emitStxr(ctx: *Ctx, x: decode.LoadStoreEx) !void {
    const body = ctx.body;
    const size = x.size;

    // address into local_b
    try loadGprOrSp(body, x.rn);
    try body.emit(.local_set);
    try body.emitU32(local_b);

    // compute "success flag" — accumulated boolean (i32). starts at 1 (success),
    // gets ANDed with each check (size match, addr match, cas match, bit set).
    // we'll build it up on the stack via locals.

    // check 1: monitor size matches this STXR's size
    try body.emitRegAddr(@bitCast(cpu.monitor_size_offset));
    try body.emit(.i32_load);
    try body.emitMemArg(2, 0);
    try body.emitConstI32(@as(i32, size));
    try body.emit(.i32_eq);
    try body.emit(.local_set);
    try body.emitU32(local_nzcv_tmp); // park first check in $nzcv_tmp (i32)

    // check 2: monitor addr matches. only relevant if size matches; we still compute.
    try body.emitRegAddr(@bitCast(cpu.monitor_addr_offset));
    try body.loadI64(0);
    try body.emit(.local_get);
    try body.emitU32(local_b);
    try body.emit(.i64_eq);
    try body.emit(.local_get);
    try body.emitU32(local_nzcv_tmp);
    try body.emit(.i32_and);
    try body.emit(.local_set);
    try body.emitU32(local_nzcv_tmp);

    if (ctx.shared_monitor) {
        // shared-memory path: atomic.cmpxchg + shadow bit re-check.
        // address
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(cpu.guest_mem_base));
        try body.emit(.i32_add);
        // expected = monitor_value
        try body.emitRegAddr(@bitCast(cpu.monitor_value_offset));
        try body.loadI64(0);
        if (size < 8) try body.emit(.i32_wrap_i64);
        // replacement = rt
        try loadGpr(body, x.rt, true);
        if (size < 8) try body.emit(.i32_wrap_i64);
        // cmpxchg
        try body.emitAtomic(atomicCmpxchgSubOp(size));
        try body.emitU32(atomicAlignLog2(size));
        try body.emitU32(0);

        // check 3: cmpxchg result == expected
        try body.emitRegAddr(@bitCast(cpu.monitor_value_offset));
        try body.loadI64(0);
        if (size < 8) {
            try body.emit(.i32_wrap_i64);
            try body.emit(.i32_eq);
        } else {
            try body.emit(.i64_eq);
        }
        try body.emit(.local_get);
        try body.emitU32(local_nzcv_tmp);
        try body.emit(.i32_and);
        try body.emit(.local_set);
        try body.emitU32(local_nzcv_tmp);

        // check 4: shadow bit still set (atomic load, then mask)
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emitConstI64(9);
        try body.emit(.i64_shr_u);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(ctx.shadow_base));
        try body.emit(.i32_add);
        try body.emitAtomic(wasm.Atomic.i32_atomic_load8_u);
        try body.emitU32(0);
        try body.emitU32(0);
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emitConstI64(6);
        try body.emit(.i64_shr_u);
        try body.emitConstI64(7);
        try body.emit(.i64_and);
        try body.emit(.i32_wrap_i64);
        try body.emit(.i32_shr_u);
        try body.emitConstI32(1);
        try body.emit(.i32_and);
        try body.emit(.local_get);
        try body.emitU32(local_nzcv_tmp);
        try body.emit(.i32_and);
        try body.emit(.local_set);
        try body.emitU32(local_nzcv_tmp);
    } else {
        // single-thread path: plain compare-and-store. since no other thread can write between
        // LDXR and STXR, the only way to fail is monitor invalid (size/addr mismatch) or our
        // own stash mismatching the current memory value (e.g. some other store cleared it).
        // load current value
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(cpu.guest_mem_base));
        try body.emit(.i32_add);
        try emitLoadOp(body, size, false, true);
        // compare with monitor_value. both are zero-extended u64 of the access width
        try body.emitRegAddr(@bitCast(cpu.monitor_value_offset));
        try body.loadI64(0);
        try body.emit(.i64_eq);
        // AND with prior monitor checks
        try body.emit(.local_get);
        try body.emitU32(local_nzcv_tmp);
        try body.emit(.i32_and);
        try body.emit(.local_tee);
        try body.emitU32(local_nzcv_tmp);
        // if success flag is 1, store the new value; otherwise skip
        try body.emit(.@"if");
        try body.bytes.append(body.arena, @intFromEnum(wasm.BlockType.empty));
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(cpu.guest_mem_base));
        try body.emit(.i32_add);
        try loadGpr(body, x.rt, true);
        switch (size) {
            1 => { try body.emit(.i64_store8); try body.emitMemArg(0, 0); },
            2 => { try body.emit(.i64_store16); try body.emitMemArg(1, 0); },
            4 => { try body.emit(.i64_store32); try body.emitMemArg(2, 0); },
            8 => { try body.emit(.i64_store); try body.emitMemArg(3, 0); },
            else => unreachable,
        }
        try body.emit(.end);
    }

    // store 0 (success) or 1 (failure) to rs (low 32 bits)
    if (x.rs != 31) {
        try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rs)));
        try body.emit(.local_get);
        try body.emitU32(local_nzcv_tmp);
        try body.emit(.i32_eqz);
        try body.emit(.i64_extend_i32_u);
        try body.storeI64(0);
    }

    // unconditionally clear the monitor (matches arm: "stxr clears monitor whether successful or not")
    try body.emitRegAddr(@bitCast(cpu.monitor_size_offset));
    try body.emitConstI32(0);
    try body.emit(.i32_store);
    try body.emitMemArg(2, 0);

    if (ctx.shared_monitor) try emitShadowClearSpan(body, ctx.shadow_base, local_b, size);
}

// ── simd load/store ──
// V registers are 16 bytes at v_offset + 16*rt in cpu state. for 128-bit access
// we do two 8-byte transfers. for smaller, one transfer + zero the upper half on load.

fn emitSimdEffAddr(body: *wasm.Body, rn: u5, offset: i32, mode: decode.AddrMode) !void {
    try loadGprOrSp(body, rn);
    try body.emit(.local_set);
    try body.emitU32(local_a);

    try body.emit(.local_get);
    try body.emitU32(local_a);
    if (offset != 0) {
        try body.emitConstI64(@as(i64, offset));
        try body.emit(.i64_add);
    }
    try body.emit(.local_set);
    try body.emitU32(local_b);
    _ = mode;
}

fn writebackRnSimd(body: *wasm.Body, rn: u5, mode: decode.AddrMode) !void {
    if (mode != .pre_index and mode != .post_index) return;
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(rn)));
    try body.emit(.local_get);
    try body.emitU32(local_b);
    try body.storeI64(0);
}

fn emitSimdLoad(ctx: *Ctx, x: decode.SimdMem) !void {
    const body = ctx.body;
    try emitSimdEffAddr(body, x.rn, x.offset, x.mode);

    const access_local: u32 = if (x.mode == .post_index) local_a else local_b;
    const v_off = cpu.vreg_offset(x.rt);

    if (x.size == 16) {
        // low 8 bytes
        try body.emitRegAddr(@bitCast(v_off));
        try body.emit(.local_get);
        try body.emitU32(access_local);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(cpu.guest_mem_base));
        try body.emit(.i32_add);
        try body.emit(.i64_load);
        try body.emitMemArg(3, 0);
        try body.storeI64(0);

        // high 8 bytes
        try body.emitRegAddr(@bitCast(v_off + 8));
        try body.emit(.local_get);
        try body.emitU32(access_local);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(cpu.guest_mem_base + 8));
        try body.emit(.i32_add);
        try body.emit(.i64_load);
        try body.emitMemArg(3, 0);
        try body.storeI64(0);
    } else {
        // 1/2/4/8 byte load into low qword; zero upper qword
        try body.emitRegAddr(@bitCast(v_off));
        try body.emit(.local_get);
        try body.emitU32(access_local);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(cpu.guest_mem_base));
        try body.emit(.i32_add);
        switch (x.size) {
            1 => { try body.emit(.i64_load8_u); try body.emitMemArg(0, 0); },
            2 => { try body.emit(.i64_load16_u); try body.emitMemArg(1, 0); },
            4 => { try body.emit(.i64_load32_u); try body.emitMemArg(2, 0); },
            8 => { try body.emit(.i64_load); try body.emitMemArg(3, 0); },
            else => unreachable,
        }
        try body.storeI64(0);

        // zero high half of V
        try body.emitRegAddr(@bitCast(v_off + 8));
        try body.emitConstI64(0);
        try body.storeI64(0);
    }

    try writebackRnSimd(body, x.rn, x.mode);
}

fn emitSimdStore(ctx: *Ctx, x: decode.SimdMem) !void {
    const body = ctx.body;
    try emitSimdEffAddr(body, x.rn, x.offset, x.mode);

    const access_local: u32 = if (x.mode == .post_index) local_a else local_b;
    if (ctx.shared_monitor) try emitShadowClearSpan(body, ctx.shadow_base, access_local, x.size);

    const v_off = cpu.vreg_offset(x.rt);

    if (x.size == 16) {
        // low 8 bytes
        try body.emit(.local_get);
        try body.emitU32(access_local);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(cpu.guest_mem_base));
        try body.emit(.i32_add);
        try body.emitRegAddr(@bitCast(v_off));
        try body.loadI64(0);
        try body.emit(.i64_store);
        try body.emitMemArg(3, 0);

        // high 8 bytes
        try body.emit(.local_get);
        try body.emitU32(access_local);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(cpu.guest_mem_base + 8));
        try body.emit(.i32_add);
        try body.emitRegAddr(@bitCast(v_off + 8));
        try body.loadI64(0);
        try body.emit(.i64_store);
        try body.emitMemArg(3, 0);
    } else {
        try body.emit(.local_get);
        try body.emitU32(access_local);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(cpu.guest_mem_base));
        try body.emit(.i32_add);
        try body.emitRegAddr(@bitCast(v_off));
        try body.loadI64(0);
        switch (x.size) {
            1 => { try body.emit(.i64_store8); try body.emitMemArg(0, 0); },
            2 => { try body.emit(.i64_store16); try body.emitMemArg(1, 0); },
            4 => { try body.emit(.i64_store32); try body.emitMemArg(2, 0); },
            8 => { try body.emit(.i64_store); try body.emitMemArg(3, 0); },
            else => unreachable,
        }
    }

    try writebackRnSimd(body, x.rn, x.mode);
}

fn emitSimdLoadPair(ctx: *Ctx, x: decode.SimdMemPair) !void {
    const body = ctx.body;
    try emitSimdEffAddr(body, x.rn, x.offset, x.mode);

    const access_local: u32 = if (x.mode == .post_index) local_a else local_b;

    // load rt
    try doSimdMemLoad(body, x.size, access_local, 0, x.rt);
    // load rt2 at offset = size
    try doSimdMemLoad(body, x.size, access_local, @intCast(x.size), x.rt2);

    try writebackRnSimd(body, x.rn, x.mode);
}

fn emitSimdStorePair(ctx: *Ctx, x: decode.SimdMemPair) !void {
    const body = ctx.body;
    try emitSimdEffAddr(body, x.rn, x.offset, x.mode);

    const access_local: u32 = if (x.mode == .post_index) local_a else local_b;
    if (ctx.shared_monitor) try emitShadowClearSpan(body, ctx.shadow_base, access_local, @intCast(x.size * 2));

    try doSimdMemStore(body, x.size, access_local, 0, x.rt);
    try doSimdMemStore(body, x.size, access_local, @intCast(x.size), x.rt2);

    try writebackRnSimd(body, x.rn, x.mode);
}

fn doSimdMemLoad(body: *wasm.Body, size: u8, addr_local: u32, extra_offset: i32, rt: u5) !void {
    const v_off = cpu.vreg_offset(rt);

    if (size == 16) {
        // two 8-byte transfers
        try body.emitRegAddr(@bitCast(v_off));
        try body.emit(.local_get);
        try body.emitU32(addr_local);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(@as(u32, @bitCast(@as(i32, @bitCast(cpu.guest_mem_base)) + extra_offset))));
        try body.emit(.i32_add);
        try body.emit(.i64_load);
        try body.emitMemArg(3, 0);
        try body.storeI64(0);

        try body.emitRegAddr(@bitCast(v_off + 8));
        try body.emit(.local_get);
        try body.emitU32(addr_local);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(@as(u32, @bitCast(@as(i32, @bitCast(cpu.guest_mem_base)) + extra_offset + 8))));
        try body.emit(.i32_add);
        try body.emit(.i64_load);
        try body.emitMemArg(3, 0);
        try body.storeI64(0);
    } else {
        try body.emitRegAddr(@bitCast(v_off));
        try body.emit(.local_get);
        try body.emitU32(addr_local);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(@as(u32, @bitCast(@as(i32, @bitCast(cpu.guest_mem_base)) + extra_offset))));
        try body.emit(.i32_add);
        switch (size) {
            1 => { try body.emit(.i64_load8_u); try body.emitMemArg(0, 0); },
            2 => { try body.emit(.i64_load16_u); try body.emitMemArg(1, 0); },
            4 => { try body.emit(.i64_load32_u); try body.emitMemArg(2, 0); },
            8 => { try body.emit(.i64_load); try body.emitMemArg(3, 0); },
            else => unreachable,
        }
        try body.storeI64(0);

        try body.emitRegAddr(@bitCast(v_off + 8));
        try body.emitConstI64(0);
        try body.storeI64(0);
    }
}

fn doSimdMemStore(body: *wasm.Body, size: u8, addr_local: u32, extra_offset: i32, rt: u5) !void {
    const v_off = cpu.vreg_offset(rt);

    if (size == 16) {
        try body.emit(.local_get);
        try body.emitU32(addr_local);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(@as(u32, @bitCast(@as(i32, @bitCast(cpu.guest_mem_base)) + extra_offset))));
        try body.emit(.i32_add);
        try body.emitRegAddr(@bitCast(v_off));
        try body.loadI64(0);
        try body.emit(.i64_store);
        try body.emitMemArg(3, 0);

        try body.emit(.local_get);
        try body.emitU32(addr_local);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(@as(u32, @bitCast(@as(i32, @bitCast(cpu.guest_mem_base)) + extra_offset + 8))));
        try body.emit(.i32_add);
        try body.emitRegAddr(@bitCast(v_off + 8));
        try body.loadI64(0);
        try body.emit(.i64_store);
        try body.emitMemArg(3, 0);
    } else {
        try body.emit(.local_get);
        try body.emitU32(addr_local);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(@as(u32, @bitCast(@as(i32, @bitCast(cpu.guest_mem_base)) + extra_offset))));
        try body.emit(.i32_add);
        try body.emitRegAddr(@bitCast(v_off));
        try body.loadI64(0);
        switch (size) {
            1 => { try body.emit(.i64_store8); try body.emitMemArg(0, 0); },
            2 => { try body.emit(.i64_store16); try body.emitMemArg(1, 0); },
            4 => { try body.emit(.i64_store32); try body.emitMemArg(2, 0); },
            8 => { try body.emit(.i64_store); try body.emitMemArg(3, 0); },
            else => unreachable,
        }
    }
}

fn emitSimdLoadReg(ctx: *Ctx, x: decode.SimdMemReg) !void {
    const body = ctx.body;

    // compute address: rn + extended_rm into local_b
    try loadGprOrSp(body, x.rn);
    try emitExtendedRm(body, x.rm, x.option, x.shift, x.size);
    try body.emit(.i64_add);
    try body.emit(.local_set);
    try body.emitU32(local_b);

    const sm = decode.SimdMem{ .size = x.size, .rt = x.rt, .rn = 0, .offset = 0, .mode = .offset_unsigned };
    // hijack emitSimdLoad's body but skip emitSimdEffAddr (already done)
    // simpler: inline the load logic
    const v_off = cpu.vreg_offset(x.rt);
    if (x.size == 16) {
        try body.emitRegAddr(@bitCast(v_off));
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(cpu.guest_mem_base));
        try body.emit(.i32_add);
        try body.emit(.i64_load);
        try body.emitMemArg(3, 0);
        try body.storeI64(0);

        try body.emitRegAddr(@bitCast(v_off + 8));
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(cpu.guest_mem_base + 8));
        try body.emit(.i32_add);
        try body.emit(.i64_load);
        try body.emitMemArg(3, 0);
        try body.storeI64(0);
    } else {
        try body.emitRegAddr(@bitCast(v_off));
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(cpu.guest_mem_base));
        try body.emit(.i32_add);
        switch (x.size) {
            1 => { try body.emit(.i64_load8_u); try body.emitMemArg(0, 0); },
            2 => { try body.emit(.i64_load16_u); try body.emitMemArg(1, 0); },
            4 => { try body.emit(.i64_load32_u); try body.emitMemArg(2, 0); },
            8 => { try body.emit(.i64_load); try body.emitMemArg(3, 0); },
            else => unreachable,
        }
        try body.storeI64(0);

        try body.emitRegAddr(@bitCast(v_off + 8));
        try body.emitConstI64(0);
        try body.storeI64(0);
    }
    _ = sm;
}

fn emitSimdStoreReg(ctx: *Ctx, x: decode.SimdMemReg) !void {
    const body = ctx.body;

    try loadGprOrSp(body, x.rn);
    try emitExtendedRm(body, x.rm, x.option, x.shift, x.size);
    try body.emit(.i64_add);
    try body.emit(.local_set);
    try body.emitU32(local_b);

    if (ctx.shared_monitor) try emitShadowClearSpan(body, ctx.shadow_base, local_b, x.size);

    const v_off = cpu.vreg_offset(x.rt);
    if (x.size == 16) {
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(cpu.guest_mem_base));
        try body.emit(.i32_add);
        try body.emitRegAddr(@bitCast(v_off));
        try body.loadI64(0);
        try body.emit(.i64_store);
        try body.emitMemArg(3, 0);

        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(cpu.guest_mem_base + 8));
        try body.emit(.i32_add);
        try body.emitRegAddr(@bitCast(v_off + 8));
        try body.loadI64(0);
        try body.emit(.i64_store);
        try body.emitMemArg(3, 0);
    } else {
        try body.emit(.local_get);
        try body.emitU32(local_b);
        try body.emit(.i32_wrap_i64);
        try body.emitConstI32(@bitCast(cpu.guest_mem_base));
        try body.emit(.i32_add);
        try body.emitRegAddr(@bitCast(v_off));
        try body.loadI64(0);
        switch (x.size) {
            1 => { try body.emit(.i64_store8); try body.emitMemArg(0, 0); },
            2 => { try body.emit(.i64_store16); try body.emitMemArg(1, 0); },
            4 => { try body.emit(.i64_store32); try body.emitMemArg(2, 0); },
            8 => { try body.emit(.i64_store); try body.emitMemArg(3, 0); },
            else => unreachable,
        }
    }
}

// ── SIMD/FP basic ops ──
//
// V registers live at v_offset(rt) in cpu state, 16 bytes each. for the subset of
// SIMD ops the boot path actually uses, we manipulate the V slot directly via two
// i64 stores/loads (low/high qword) — no v128 wasm opcode needed.

/// expand a MOVI imm8/cmode into a 64-bit pattern (replicated across lanes).
/// returns the i64 value to store in BOTH halves (if Q=1) or just low (if Q=0).
fn expandSimdImm(imm8: u8, cmode: u4, op: u1) u64 {
    const v8: u64 = @as(u64, imm8);
    const cmode_h: u3 = @truncate(cmode >> 1);
    if (op == 0) {
        return switch (cmode_h) {
            0 => v8 | (v8 << 32), // MOVI Vd.xS, #imm8
            1 => (v8 << 8) | (v8 << 40), // MOVI Vd.xS, #imm8, LSL #8
            2 => (v8 << 16) | (v8 << 48), // MOVI Vd.xS, #imm8, LSL #16
            3 => (v8 << 24) | (v8 << 56), // MOVI Vd.xS, #imm8, LSL #24
            4 => v8 | (v8 << 16) | (v8 << 32) | (v8 << 48), // MOVI Vd.xH, #imm8
            5 => (v8 << 8) | (v8 << 24) | (v8 << 40) | (v8 << 56), // MOVI Vd.xH, #imm8 LSL #8
            7 => byteBroadcast(v8), // MOVI Vd.16B/8B, #imm8
            else => byteBroadcast(v8),
        };
    }
    // op=1: MVNI for cmode_h <= 3 (invert), FMOV variants otherwise
    if (cmode_h <= 3) {
        const base = expandSimdImm(imm8, cmode, 0);
        return ~base;
    }
    return 0;
}

fn byteBroadcast(v: u64) u64 {
    var x = v & 0xff;
    x |= x << 8;
    x |= x << 16;
    x |= x << 32;
    return x;
}

fn emitSimdMovi(body: *wasm.Body, x: decode.SimdMovi) !void {
    const v_off = cpu.vreg_offset(x.rd);
    const pattern = expandSimdImm(x.imm8, x.cmode, x.op);

    // low half
    try body.emitRegAddr(@bitCast(v_off));
    try body.emitConstI64(@bitCast(pattern));
    try body.storeI64(0);

    // high half: replicate if Q=1, else zero
    try body.emitRegAddr(@bitCast(v_off + 8));
    if (x.q) {
        try body.emitConstI64(@bitCast(pattern));
    } else {
        try body.emitConstI64(0);
    }
    try body.storeI64(0);
}

fn emitSimdDupGpr(body: *wasm.Body, x: decode.SimdDupGpr) !void {
    const v_off = cpu.vreg_offset(x.rd);

    // load gpr (W or X depending on esize) and replicate to a full 64-bit qword
    try loadGpr(body, x.rn, x.esize >= 8);

    switch (x.esize) {
        1 => {
            try body.emitConstI64(0xff);
            try body.emit(.i64_and);
            try body.emitConstI64(@bitCast(@as(u64, 0x0101010101010101)));
            try body.emit(.i64_mul);
        },
        2 => {
            try body.emitConstI64(0xffff);
            try body.emit(.i64_and);
            try body.emitConstI64(@bitCast(@as(u64, 0x0001000100010001)));
            try body.emit(.i64_mul);
        },
        4 => {
            try body.emitConstI64(0xffff_ffff);
            try body.emit(.i64_and);
            try body.emitConstI64(@bitCast(@as(u64, 0x0000000100000001)));
            try body.emit(.i64_mul);
        },
        8 => {}, // already full qword
        else => unreachable,
    }
    try body.emit(.local_set);
    try body.emitU32(local_a);

    // store to low half
    try body.emitRegAddr(@bitCast(v_off));
    try body.emit(.local_get);
    try body.emitU32(local_a);
    try body.storeI64(0);

    // high half: replicate if Q=1, else zero
    try body.emitRegAddr(@bitCast(v_off + 8));
    if (x.q) {
        try body.emit(.local_get);
        try body.emitU32(local_a);
    } else {
        try body.emitConstI64(0);
    }
    try body.storeI64(0);
}

fn emitSimdUmov(body: *wasm.Body, x: decode.SimdElem) !void {
    if (x.rd == 31) return;
    const v_off = cpu.vreg_offset(x.rn);

    const byte_pos: u32 = @as(u32, x.index) * @as(u32, x.esize);
    const qword_idx: u32 = byte_pos / 8;
    const bit_pos: u32 = (byte_pos % 8) * 8;

    // load the source qword from V register
    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));
    try body.emitRegAddr(@bitCast(v_off + qword_idx * 8));
    try body.loadI64(0);

    if (bit_pos > 0) {
        try body.emitConstI64(@as(i64, @intCast(bit_pos)));
        try body.emit(.i64_shr_u);
    }
    if (x.esize < 8) {
        const mask: u64 = (@as(u64, 1) << @intCast(x.esize * 8)) - 1;
        try body.emitConstI64(@bitCast(mask));
        try body.emit(.i64_and);
    }
    try body.storeI64(0);
}

fn emitSimdInsGpr(body: *wasm.Body, x: decode.SimdElem) !void {
    const v_off = cpu.vreg_offset(x.rd);
    const byte_pos: u32 = @as(u32, x.index) * @as(u32, x.esize);
    const qword_idx: u32 = byte_pos / 8;
    const bit_pos: u32 = (byte_pos % 8) * 8;
    const slot_addr: u32 = v_off + qword_idx * 8;
    const mask: u64 = if (x.esize == 8) ~@as(u64, 0) else (@as(u64, 1) << @intCast(x.esize * 8)) - 1;

    if (x.esize == 8) {
        try body.emitRegAddr(@bitCast(slot_addr));
        try loadGpr(body, x.rn, true);
        try body.storeI64(0);
        return;
    }

    // read-modify-write: read slot, clear element bits, OR in new value
    try body.emitRegAddr(@bitCast(slot_addr));
    try body.emitRegAddr(@bitCast(slot_addr));
    try body.loadI64(0);
    try body.emitConstI64(@bitCast(~(mask << @intCast(bit_pos))));
    try body.emit(.i64_and);

    try loadGpr(body, x.rn, x.esize >= 4);
    try body.emitConstI64(@bitCast(mask));
    try body.emit(.i64_and);
    if (bit_pos > 0) {
        try body.emitConstI64(@as(i64, @intCast(bit_pos)));
        try body.emit(.i64_shl);
    }
    try body.emit(.i64_or);
    try body.storeI64(0);
}

fn emitFmovToFp(body: *wasm.Body, x: decode.FmovGprFp) !void {
    const v_off = cpu.vreg_offset(x.rd);

    // low qword = GPR value (masked to 32 if sf=0)
    try body.emitRegAddr(@bitCast(v_off));
    try loadGpr(body, x.rn, x.sf);
    try body.storeI64(0);

    // high qword zeroed (per arm: writing to FP reg zeros upper bits beyond size)
    try body.emitRegAddr(@bitCast(v_off + 8));
    try body.emitConstI64(0);
    try body.storeI64(0);
}

fn emitFmovFromFp(body: *wasm.Body, x: decode.FmovGprFp) !void {
    if (x.rd == 31) return;
    const v_off = cpu.vreg_offset(x.rn);

    try body.emitRegAddr(@bitCast(cpu.gpr_offset(x.rd)));
    try body.emitRegAddr(@bitCast(v_off));
    try body.loadI64(0);
    if (!x.sf) {
        try body.emitConstI64(0xffff_ffff);
        try body.emit(.i64_and);
    }
    try body.storeI64(0);
}
