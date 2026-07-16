//! translator.wasm entry points.
//!
//! interface model: the wasm linear memory is shared with the js host.
//! the host writes arm64 guest bytes into the input buffer at a known
//! offset, calls translate, and reads the produced wasm module bytes
//! from the output buffer at another known offset.

const std = @import("std");
const decode = @import("decode.zig");
const block = @import("block.zig");
const translate = @import("translate.zig");

// fixed buffers inside our wasm linear memory.
// the host imports `memory` from us (exported below) and can directly
// read/write at these offsets.
const input_buf_offset: u32 = 0x10000; // 64 kib in
const input_buf_size: u32 = 0x10000; // 64 kib of guest code
const output_buf_offset: u32 = input_buf_offset + input_buf_size;
const output_buf_size: u32 = 0x100000; // 1 mib of output
const arena_offset: u32 = output_buf_offset + output_buf_size;
const arena_size: u32 = 0x200000; // 2 mib scratch
const total_size: u32 = arena_offset + arena_size;

// raw arena backed by a static array placed in linear memory at link
// time. start size is enough; can `memory.grow` if needed (but we won't
// for this iteration).
var arena_buf: [arena_size]u8 = undefined;
var arena_fba: std.heap.FixedBufferAllocator = .init(&arena_buf);

var last_result_offset: u32 = 0;
var last_result_len: u32 = 0;

export fn translator_input_buf_offset() u32 {
    return input_buf_offset;
}
export fn translator_input_buf_size() u32 {
    return input_buf_size;
}
export fn translator_output_buf_offset() u32 {
    return output_buf_offset;
}

/// reset the arena before a new translation.
export fn translator_reset() void {
    arena_fba.reset();
    last_result_offset = 0;
    last_result_len = 0;
}

/// translate a region starting at entry_pc. the guest code for that
/// region must already be written into the input buffer, with the byte
/// at offset 0 corresponding to guest address `base_pc`.
///
/// `shadow_base` is the offset within the output module's linear memory where
/// the per-worker exclusive-monitor shadow bitmap lives (1 bit per `erg_bytes`
/// of guest memory). only consulted when `shared_monitor == 1`.
///
/// `shared_monitor` (0 or 1) enables the full ARM exclusive-monitor shadow
/// protocol. set to 1 when memory is actually shared across threads; set to 0
/// for single-threaded workers (the current jitter architecture). when 0, LDXR
/// stashes (addr,value,size) in cpu state and STXR succeeds iff the value at
/// addr still matches — value-CAS is correct under sequential execution.
///
/// returns 0 on success, nonzero on failure.
export fn translator_translate_region(
    base_pc_lo: u32,
    base_pc_hi: u32,
    entry_pc_lo: u32,
    entry_pc_hi: u32,
    code_len: u32,
    shadow_base: u32,
    shared_monitor: u32,
    protection: u32,
    prot_base: u32,
) u32 {
    if (code_len > input_buf_size) return 1;
    arena_fba.reset();

    const base_pc: u64 = (@as(u64, base_pc_hi) << 32) | @as(u64, base_pc_lo);
    const entry_pc: u64 = (@as(u64, entry_pc_hi) << 32) | @as(u64, entry_pc_lo);

    const guest_mem_ptr: [*]u8 = @ptrFromInt(input_buf_offset);
    const guest_mem: []const u8 = guest_mem_ptr[0..code_len];

    var region = block.buildRegion(arena_fba.allocator(), guest_mem, base_pc, entry_pc) catch return 2;
    defer block.freeRegion(arena_fba.allocator(), &region);

    const opts: translate.Options = .{
        .shadow_base = shadow_base,
        .shared_monitor = shared_monitor != 0,
        .protection = protection != 0,
        .prot_base = prot_base,
    };
    const wasm_bytes = translate.translateRegion(arena_fba.allocator(), &region, opts) catch return 3;

    if (wasm_bytes.len > output_buf_size) return 4;
    const output_ptr: [*]u8 = @ptrFromInt(output_buf_offset);
    @memcpy(output_ptr[0..wasm_bytes.len], wasm_bytes);

    last_result_offset = output_buf_offset;
    last_result_len = @intCast(wasm_bytes.len);
    return 0;
}

export fn translator_result_offset() u32 {
    return last_result_offset;
}

export fn translator_result_len() u32 {
    return last_result_len;
}

/// debug / parity-testing entry point. decode a single instruction and
/// return its kind tag.
export fn translator_decode_kind(insn: u32) u32 {
    const op = decode.decode(insn);
    return @intFromEnum(op);
}

/// debug: build the region and write its block table into the output buffer.
/// each row is 5 u64 little-endian: start_pc, end_pc, insn_count, exit_kind,
/// exit_detail (branch target / fallthrough pcs packed as two u32s where
/// applicable). returns the number of blocks, or 0xffffffff on failure.
export fn translator_debug_region(
    base_pc_lo: u32,
    base_pc_hi: u32,
    entry_pc_lo: u32,
    entry_pc_hi: u32,
    code_len: u32,
) u32 {
    if (code_len > input_buf_size) return 0xffffffff;
    arena_fba.reset();

    const base_pc: u64 = (@as(u64, base_pc_hi) << 32) | @as(u64, base_pc_lo);
    const entry_pc: u64 = (@as(u64, entry_pc_hi) << 32) | @as(u64, entry_pc_lo);
    const guest_mem_ptr: [*]u8 = @ptrFromInt(input_buf_offset);
    const guest_mem: []const u8 = guest_mem_ptr[0..code_len];

    var region = block.buildRegion(arena_fba.allocator(), guest_mem, base_pc, entry_pc) catch return 0xffffffff;
    defer block.freeRegion(arena_fba.allocator(), &region);

    const out: [*]u64 = @ptrFromInt(output_buf_offset);
    for (region.blocks, 0..) |b, i| {
        out[i * 5 + 0] = b.start_pc;
        out[i * 5 + 1] = b.end_pc;
        out[i * 5 + 2] = b.insns.len;
        out[i * 5 + 3] = @intFromEnum(b.exit);
        out[i * 5 + 4] = switch (b.exit) {
            .fallthrough_to_block => |pc| pc,
            .direct_branch => |pc| pc,
            .cond_branch => |c| (c.taken_pc & 0xffff_ffff) | ((c.fallthrough_pc & 0xffff_ffff) << 32),
            else => 0,
        };
    }
    last_result_offset = output_buf_offset;
    last_result_len = @intCast(region.blocks.len * 5 * 8);
    return @intCast(region.blocks.len);
}

/// debug: dump the per-insn kinds a region's blocks captured. rows of
/// (pc, kind) u64 pairs for the block containing `probe_pc`.
export fn translator_debug_block_insns(
    base_pc_lo: u32,
    base_pc_hi: u32,
    entry_pc_lo: u32,
    entry_pc_hi: u32,
    code_len: u32,
    probe_pc_lo: u32,
    probe_pc_hi: u32,
) u32 {
    if (code_len > input_buf_size) return 0xffffffff;
    arena_fba.reset();
    const base_pc: u64 = (@as(u64, base_pc_hi) << 32) | @as(u64, base_pc_lo);
    const entry_pc: u64 = (@as(u64, entry_pc_hi) << 32) | @as(u64, entry_pc_lo);
    const probe_pc: u64 = (@as(u64, probe_pc_hi) << 32) | @as(u64, probe_pc_lo);
    const guest_mem_ptr: [*]u8 = @ptrFromInt(input_buf_offset);
    const guest_mem: []const u8 = guest_mem_ptr[0..code_len];

    var region = block.buildRegion(arena_fba.allocator(), guest_mem, base_pc, entry_pc) catch return 0xffffffff;
    defer block.freeRegion(arena_fba.allocator(), &region);

    const out: [*]u64 = @ptrFromInt(output_buf_offset);
    last_result_offset = output_buf_offset;
    for (region.blocks) |b| {
        if (b.start_pc <= probe_pc and probe_pc < b.end_pc) {
            for (b.insns, 0..) |op, i| {
                out[i * 2 + 0] = b.start_pc + i * 4;
                out[i * 2 + 1] = @intFromEnum(op);
            }
            last_result_len = @intCast(b.insns.len * 16);
            return @intCast(b.insns.len);
        }
    }
    return 0;
}
