//! basic block + region detection.
//!
//! a region is a connected control-flow subgraph rooted at an entry pc.
//! blocks are split at control-flow boundaries (branches, syscalls,
//! unknown instructions). branches whose targets fall inside the region
//! become wasm `br` between labeled blocks. branches that exit (indirect,
//! out of budget, etc.) become region exits.

const std = @import("std");
const decode = @import("decode.zig");

pub const region_size_budget: u32 = 256; // arm64 instructions per region
pub const max_blocks_per_region: u32 = 64;

pub const ExitKind = enum {
    fallthrough_to_block, // continue to next block in region (no branch insn)
    direct_branch, // unconditional B to in-region target
    cond_branch, // B.cond / CBZ / CBNZ / TBZ / TBNZ — two-way
    indirect_branch, // BR/BLR/RET — exits region
    syscall, // SVC — exits region
    unknown_insn, // decoder returned .unknown — exits region
    region_full, // ran out of budget — exits region
};

pub const Block = struct {
    start_pc: u64,
    end_pc: u64, // pc one past the last insn (start_pc + 4 * insn_count)
    insns: []const decode.Op,
    exit: Exit,
};

pub const Exit = union(ExitKind) {
    fallthrough_to_block: u64, // pc of next block
    direct_branch: u64, // target pc (always in-region)
    cond_branch: struct { taken_pc: u64, fallthrough_pc: u64, in_region_taken: bool, in_region_fallthrough: bool },
    indirect_branch: void, // target unknown
    syscall: void,
    unknown_insn: void,
    region_full: void,
};

pub const Region = struct {
    entry_pc: u64,
    blocks: []const Block,
    // pc → block index, for resolving in-region branch targets
    pc_to_block: std.AutoHashMapUnmanaged(u64, u32),
};

pub const Error = error{
    InvalidPc,
    OutOfMemory,
    BadGuestMemory,
};

/// fetch a 4-byte arm64 instruction at the given guest pc from the
/// caller-supplied guest memory view. caller must ensure the memory
/// range is mapped.
pub fn fetchInsn(guest_mem: []const u8, base_pc: u64, pc: u64) ?u32 {
    if (pc < base_pc) return null;
    const off: usize = @intCast(pc - base_pc);
    if (off + 4 > guest_mem.len) return null;
    return std.mem.readInt(u32, guest_mem[off..][0..4], .little);
}


/// pass 1: walk from leaders, decoding through, find all "leader" pcs
/// (entry points to basic blocks). a pc is a leader if:
///   - it is the region entry
///   - it is the target of a branch
///   - it is the instruction immediately after a branch (fall-through)
///
/// also returns a map from pc → decoded op for all visited pcs.
fn discoverLeaders(
    allocator: std.mem.Allocator,
    guest_mem: []const u8,
    base_pc: u64,
    entry_pc: u64,
) Error!struct {
    leaders: std.AutoHashMapUnmanaged(u64, void),
    decoded: std.AutoHashMapUnmanaged(u64, decode.Op),
    budget_remaining: u32,
} {
    var leaders: std.AutoHashMapUnmanaged(u64, void) = .empty;
    var decoded: std.AutoHashMapUnmanaged(u64, decode.Op) = .empty;
    var queue: std.ArrayList(u64) = .empty;
    defer queue.deinit(allocator);

    try leaders.put(allocator, entry_pc, {});
    try queue.append(allocator, entry_pc);
    var budget: u32 = region_size_budget;

    while (queue.items.len > 0) {
        const start = queue.orderedRemove(0);
        var pc = start;
        while (budget > 0) {
            if (decoded.contains(pc)) break; // already explored from another path
            // a leader outside the code window (backward branch target, bl below
            // base) simply stays undecoded: pass 2 forms a stub block that exits
            // the region at that pc, and the dispatcher translates a fresh region
            // there with its own window
            const word = fetchInsn(guest_mem, base_pc, pc) orelse break;
            const op = decode.decode(word);
            try decoded.put(allocator, pc, op);
            budget -= 1;
            const next_pc = pc + 4;

            switch (op) {
                .b => |x| {
                    const target: u64 = @intCast(@as(i64, @intCast(pc)) + x.offset);
                    if (!leaders.contains(target)) {
                        try leaders.put(allocator, target, {});
                        try queue.append(allocator, target);
                    }
                    break;
                },
                .bl => |x| {
                    const target: u64 = @intCast(@as(i64, @intCast(pc)) + x.offset);
                    if (!leaders.contains(target)) {
                        try leaders.put(allocator, target, {});
                        try queue.append(allocator, target);
                    }
                    break;
                },
                .b_cond, .cbz, .cbnz, .tbz, .tbnz => {
                    const offset: i32 = switch (op) {
                        .b_cond => |y| y.offset,
                        .cbz, .cbnz => |y| y.offset,
                        .tbz, .tbnz => |y| y.offset,
                        else => unreachable,
                    };
                    const taken: u64 = @intCast(@as(i64, @intCast(pc)) + offset);
                    if (!leaders.contains(taken)) {
                        try leaders.put(allocator, taken, {});
                        try queue.append(allocator, taken);
                    }
                    if (!leaders.contains(next_pc)) {
                        try leaders.put(allocator, next_pc, {});
                        try queue.append(allocator, next_pc);
                    }
                    break;
                },
                .br, .blr, .ret, .svc, .unknown => break,
                else => {},
            }
            pc = next_pc;
        }
    }

    return .{ .leaders = leaders, .decoded = decoded, .budget_remaining = budget };
}

/// build a region by first discovering all leader pcs (branch targets +
/// fall-throughs + entry), then forming a block for each leader by
/// scanning insns until the next leader or a control-flow terminator.
pub fn buildRegion(
    allocator: std.mem.Allocator,
    guest_mem: []const u8,
    base_pc: u64,
    entry_pc: u64,
) Error!Region {
    var discovery = try discoverLeaders(allocator, guest_mem, base_pc, entry_pc);
    defer discovery.leaders.deinit(allocator);
    defer discovery.decoded.deinit(allocator);

    var blocks: std.ArrayList(Block) = .empty;
    errdefer {
        for (blocks.items) |b| allocator.free(b.insns);
        blocks.deinit(allocator);
    }
    var pc_to_block: std.AutoHashMapUnmanaged(u64, u32) = .empty;
    errdefer pc_to_block.deinit(allocator);

    // sort leaders so block ordering is deterministic
    var leaders: std.ArrayList(u64) = .empty;
    defer leaders.deinit(allocator);
    {
        var it = discovery.leaders.keyIterator();
        while (it.next()) |k| try leaders.append(allocator, k.*);
    }
    std.mem.sort(u64, leaders.items, {}, std.sort.asc(u64));

    // entry block must be index 0
    if (leaders.items[0] != entry_pc) {
        // move entry_pc to front
        for (leaders.items, 0..) |pc, i| {
            if (pc == entry_pc) {
                std.mem.swap(u64, &leaders.items[0], &leaders.items[i]);
                break;
            }
        }
    }

    // form a block for each leader
    for (leaders.items) |leader| {
        var ops: std.ArrayList(decode.Op) = .empty;
        errdefer ops.deinit(allocator);

        var pc = leader;
        var exit: Exit = .{ .region_full = {} };

        while (true) {
            const op = discovery.decoded.get(pc) orelse break;
            try ops.append(allocator, op);
            const next_pc = pc + 4;

            switch (op) {
                .b => |x| {
                    exit = .{ .direct_branch = @intCast(@as(i64, @intCast(pc)) + x.offset) };
                    pc = next_pc;
                    break;
                },
                .bl => |x| {
                    exit = .{ .direct_branch = @intCast(@as(i64, @intCast(pc)) + x.offset) };
                    pc = next_pc;
                    break;
                },
                .b_cond, .cbz, .cbnz, .tbz, .tbnz => {
                    const offset: i32 = switch (op) {
                        .b_cond => |y| y.offset,
                        .cbz, .cbnz => |y| y.offset,
                        .tbz, .tbnz => |y| y.offset,
                        else => unreachable,
                    };
                    exit = .{ .cond_branch = .{
                        .taken_pc = @intCast(@as(i64, @intCast(pc)) + offset),
                        .fallthrough_pc = next_pc,
                        .in_region_taken = false,
                        .in_region_fallthrough = false,
                    } };
                    pc = next_pc;
                    break;
                },
                .br, .blr, .ret => {
                    exit = .{ .indirect_branch = {} };
                    pc = next_pc;
                    break;
                },
                .svc => {
                    exit = .{ .syscall = {} };
                    pc = next_pc;
                    break;
                },
                .unknown => {
                    exit = .{ .unknown_insn = {} };
                    pc = next_pc;
                    break;
                },
                else => {},
            }
            pc = next_pc;
            // stop if the next pc is itself a leader (block boundary)
            if (discovery.leaders.contains(pc)) {
                exit = .{ .fallthrough_to_block = pc };
                break;
            }
        }

        const idx: u32 = @intCast(blocks.items.len);
        try blocks.append(allocator, .{
            .start_pc = leader,
            .end_pc = pc,
            .insns = try ops.toOwnedSlice(allocator),
            .exit = exit,
        });
        try pc_to_block.put(allocator, leader, idx);
    }

    // the entry block must contain at least one insn — a stub entry would make
    // the region exit at its own entry pc and spin the dispatcher forever
    if (blocks.items.len == 0 or blocks.items[0].insns.len == 0) return Error.BadGuestMemory;

    // fill in cond_branch in-region flags
    for (blocks.items) |*b| {
        switch (b.exit) {
            .cond_branch => |*c| {
                c.in_region_taken = pc_to_block.contains(c.taken_pc);
                c.in_region_fallthrough = pc_to_block.contains(c.fallthrough_pc);
            },
            else => {},
        }
    }

    return Region{
        .entry_pc = entry_pc,
        .blocks = try blocks.toOwnedSlice(allocator),
        .pc_to_block = pc_to_block,
    };
}

pub fn freeRegion(allocator: std.mem.Allocator, region: *Region) void {
    for (region.blocks) |b| allocator.free(b.insns);
    allocator.free(region.blocks);
    region.pc_to_block.deinit(allocator);
}
