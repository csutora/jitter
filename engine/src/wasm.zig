//! webassembly bytecode emission. minimal but correct: emits modules
//! consumable by jsc / v8 with imported shared memory, exported run
//! function, and a labeled control-flow body.

const std = @import("std");

// ── value types ──

pub const ValType = enum(u8) {
    i32 = 0x7f,
    i64 = 0x7e,
    f32 = 0x7d,
    f64 = 0x7c,
    v128 = 0x7b,
    funcref = 0x70,
    externref = 0x6f,
};

pub const BlockType = enum(u8) {
    empty = 0x40,
    i32 = 0x7f,
    i64 = 0x7e,
    f32 = 0x7d,
    f64 = 0x7c,
};

// ── core opcodes (subset we actually emit) ──

pub const Op = enum(u8) {
    @"unreachable" = 0x00,
    nop = 0x01,
    block = 0x02,
    loop = 0x03,
    @"if" = 0x04,
    @"else" = 0x05,
    end = 0x0b,
    br = 0x0c,
    br_if = 0x0d,
    br_table = 0x0e,
    @"return" = 0x0f,
    call = 0x10,
    call_indirect = 0x11,

    drop = 0x1a,
    select = 0x1b,

    local_get = 0x20,
    local_set = 0x21,
    local_tee = 0x22,
    global_get = 0x23,
    global_set = 0x24,

    i32_load = 0x28,
    i64_load = 0x29,
    i32_load8_s = 0x2c,
    i32_load8_u = 0x2d,
    i32_load16_s = 0x2e,
    i32_load16_u = 0x2f,
    i64_load8_s = 0x30,
    i64_load8_u = 0x31,
    i64_load16_s = 0x32,
    i64_load16_u = 0x33,
    i64_load32_s = 0x34,
    i64_load32_u = 0x35,
    i32_store = 0x36,
    i64_store = 0x37,
    i32_store8 = 0x3a,
    i32_store16 = 0x3b,
    i64_store8 = 0x3c,
    i64_store16 = 0x3d,
    i64_store32 = 0x3e,

    i32_const = 0x41,
    i64_const = 0x42,

    i32_eqz = 0x45,
    i32_eq = 0x46,
    i32_ne = 0x47,
    i32_lt_s = 0x48,
    i32_lt_u = 0x49,
    i32_gt_s = 0x4a,
    i32_gt_u = 0x4b,
    i32_le_s = 0x4c,
    i32_le_u = 0x4d,
    i32_ge_s = 0x4e,
    i32_ge_u = 0x4f,

    i64_eqz = 0x50,
    i64_eq = 0x51,
    i64_ne = 0x52,
    i64_lt_s = 0x53,
    i64_lt_u = 0x54,
    i64_gt_s = 0x55,
    i64_gt_u = 0x56,
    i64_le_s = 0x57,
    i64_le_u = 0x58,
    i64_ge_s = 0x59,
    i64_ge_u = 0x5a,

    i32_add = 0x6a,
    i32_sub = 0x6b,
    i32_mul = 0x6c,
    i32_and = 0x71,
    i32_or = 0x72,
    i32_xor = 0x73,
    i32_shl = 0x74,
    i32_shr_s = 0x75,
    i32_shr_u = 0x76,
    i32_rotl = 0x77,
    i32_rotr = 0x78,

    i64_add = 0x7c,
    i64_sub = 0x7d,
    i64_mul = 0x7e,
    i64_div_s = 0x7f,
    i64_div_u = 0x80,
    i64_rem_s = 0x81,
    i64_rem_u = 0x82,
    i64_and = 0x83,
    i64_or = 0x84,
    i64_xor = 0x85,
    i64_shl = 0x86,
    i64_shr_s = 0x87,
    i64_shr_u = 0x88,
    i64_rotl = 0x89,
    i64_rotr = 0x8a,

    i32_wrap_i64 = 0xa7,
    i64_extend_i32_s = 0xac,
    i64_extend_i32_u = 0xad,

    i32_clz = 0x67,
    i32_ctz = 0x68,
    i32_popcnt = 0x69,
    i64_clz = 0x79,
    i64_ctz = 0x7a,
    i64_popcnt = 0x7b,
};

pub const section_type: u8 = 1;
pub const section_import: u8 = 2;
pub const section_function: u8 = 3;
pub const section_memory: u8 = 5;
pub const section_export: u8 = 7;
pub const section_code: u8 = 10;

pub const export_kind_func: u8 = 0;
pub const export_kind_memory: u8 = 2;
pub const import_kind_func: u8 = 0;
pub const import_kind_global: u8 = 3;
pub const import_kind_memory: u8 = 2;

// ── leb128 ──

pub fn writeULEB128(out: *std.ArrayList(u8), allocator: std.mem.Allocator, value: u64) !void {
    var v = value;
    while (true) {
        var byte: u8 = @intCast(v & 0x7f);
        v >>= 7;
        if (v != 0) byte |= 0x80;
        try out.append(allocator, byte);
        if (v == 0) break;
    }
}

pub fn writeSLEB128(out: *std.ArrayList(u8), allocator: std.mem.Allocator, value: i64) !void {
    var v = value;
    while (true) {
        const byte: u8 = @intCast(@as(i64, @bitCast(@as(u64, @bitCast(v)))) & 0x7f);
        const sign_bit = byte & 0x40;
        v >>= 7;
        const done = (v == 0 and sign_bit == 0) or (v == -1 and sign_bit != 0);
        try out.append(allocator, byte | (if (done) @as(u8, 0) else 0x80));
        if (done) break;
    }
}

// ── builder ──

pub const Module = struct {
    arena: std.mem.Allocator,
    types: std.ArrayList(FuncType),
    imports: std.ArrayList(Import),
    functions: std.ArrayList(u32), // type index per function
    exports: std.ArrayList(Export),
    code: std.ArrayList(Function),

    pub const FuncType = struct {
        params: []const ValType,
        results: []const ValType,
    };

    pub const Import = union(enum) {
        memory: struct { module: []const u8, name: []const u8, min: u32, max: ?u32, shared: bool },
        func: struct { module: []const u8, name: []const u8, type_idx: u32 },
        global: struct { module: []const u8, name: []const u8, valtype: ValType, mutable: bool },
    };

    pub const Export = struct {
        name: []const u8,
        kind: u8,
        index: u32,
    };

    pub const Local = struct { count: u32, ty: ValType };

    pub const Function = struct {
        locals: []const Local,
        body: []const u8,
    };

    pub fn init(arena: std.mem.Allocator) Module {
        return .{
            .arena = arena,
            .types = .empty,
            .imports = .empty,
            .functions = .empty,
            .exports = .empty,
            .code = .empty,
        };
    }

    pub fn addType(self: *Module, params: []const ValType, results: []const ValType) !u32 {
        const idx: u32 = @intCast(self.types.items.len);
        try self.types.append(self.arena, .{ .params = params, .results = results });
        return idx;
    }

    pub fn addImportMemory(self: *Module, module: []const u8, name: []const u8, min: u32, max: ?u32, shared: bool) !void {
        try self.imports.append(self.arena, .{ .memory = .{ .module = module, .name = name, .min = min, .max = max, .shared = shared } });
    }

    /// import a mutable/immutable global; returns its index in the global index space (imported
    /// globals precede any defined globals, so this is just the running count of global imports).
    pub fn addImportGlobal(self: *Module, module: []const u8, name: []const u8, valtype: ValType, mutable: bool) !u32 {
        var idx: u32 = 0;
        for (self.imports.items) |imp| switch (imp) {
            .global => idx += 1,
            else => {},
        };
        try self.imports.append(self.arena, .{ .global = .{ .module = module, .name = name, .valtype = valtype, .mutable = mutable } });
        return idx;
    }

    fn countFuncImports(self: *const Module) u32 {
        var n: u32 = 0;
        for (self.imports.items) |imp| switch (imp) {
            .func => n += 1,
            else => {},
        };
        return n;
    }

    pub fn addImportFunc(self: *Module, module: []const u8, name: []const u8, type_idx: u32) !u32 {
        const idx = self.countFuncImports();
        try self.imports.append(self.arena, .{ .func = .{ .module = module, .name = name, .type_idx = type_idx } });
        return idx;
    }

    pub fn addFunction(self: *Module, type_idx: u32) !u32 {
        const idx: u32 = self.countFuncImports() + @as(u32, @intCast(self.functions.items.len));
        try self.functions.append(self.arena, type_idx);
        return idx;
    }

    pub fn addExportFunc(self: *Module, name: []const u8, fn_idx: u32) !void {
        try self.exports.append(self.arena, .{ .name = name, .kind = export_kind_func, .index = fn_idx });
    }

    pub fn addCode(self: *Module, locals: []const Local, body: []const u8) !void {
        try self.code.append(self.arena, .{ .locals = locals, .body = body });
    }

    /// serialize the module to webassembly binary format.
    pub fn finalize(self: *Module, out: *std.ArrayList(u8)) !void {
        // magic + version
        try out.appendSlice(self.arena, &[_]u8{ 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00 });

        if (self.types.items.len > 0) try writeTypeSection(self, out);
        if (self.imports.items.len > 0) try writeImportSection(self, out);
        if (self.functions.items.len > 0) try writeFunctionSection(self, out);
        if (self.exports.items.len > 0) try writeExportSection(self, out);
        if (self.code.items.len > 0) try writeCodeSection(self, out);
    }
};

fn writeSectionHeader(out: *std.ArrayList(u8), arena: std.mem.Allocator, id: u8, payload_len: u64) !void {
    try out.append(arena, id);
    try writeULEB128(out, arena, payload_len);
}

fn writeTypeSection(self: *Module, out: *std.ArrayList(u8)) !void {
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(self.arena);
    try writeULEB128(&body, self.arena, self.types.items.len);
    for (self.types.items) |ft| {
        try body.append(self.arena, 0x60); // func type
        try writeULEB128(&body, self.arena, ft.params.len);
        for (ft.params) |p| try body.append(self.arena, @intFromEnum(p));
        try writeULEB128(&body, self.arena, ft.results.len);
        for (ft.results) |r| try body.append(self.arena, @intFromEnum(r));
    }
    try writeSectionHeader(out, self.arena, section_type, body.items.len);
    try out.appendSlice(self.arena, body.items);
}

fn writeImportSection(self: *Module, out: *std.ArrayList(u8)) !void {
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(self.arena);
    try writeULEB128(&body, self.arena, self.imports.items.len);
    for (self.imports.items) |imp| {
        switch (imp) {
            .memory => |m| {
                try writeULEB128(&body, self.arena, m.module.len);
                try body.appendSlice(self.arena, m.module);
                try writeULEB128(&body, self.arena, m.name.len);
                try body.appendSlice(self.arena, m.name);
                try body.append(self.arena, import_kind_memory);
                // limits: flags byte
                var flags: u8 = 0;
                if (m.max != null) flags |= 0x01;
                if (m.shared) flags |= 0x02;
                try body.append(self.arena, flags);
                try writeULEB128(&body, self.arena, m.min);
                if (m.max) |mx| try writeULEB128(&body, self.arena, mx);
            },
            .func => |f| {
                try writeULEB128(&body, self.arena, f.module.len);
                try body.appendSlice(self.arena, f.module);
                try writeULEB128(&body, self.arena, f.name.len);
                try body.appendSlice(self.arena, f.name);
                try body.append(self.arena, import_kind_func);
                try writeULEB128(&body, self.arena, f.type_idx);
            },
            .global => |g| {
                try writeULEB128(&body, self.arena, g.module.len);
                try body.appendSlice(self.arena, g.module);
                try writeULEB128(&body, self.arena, g.name.len);
                try body.appendSlice(self.arena, g.name);
                try body.append(self.arena, import_kind_global);
                try body.append(self.arena, @intFromEnum(g.valtype));
                try body.append(self.arena, if (g.mutable) @as(u8, 1) else 0);
            },
        }
    }
    try writeSectionHeader(out, self.arena, section_import, body.items.len);
    try out.appendSlice(self.arena, body.items);
}

fn writeFunctionSection(self: *Module, out: *std.ArrayList(u8)) !void {
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(self.arena);
    try writeULEB128(&body, self.arena, self.functions.items.len);
    for (self.functions.items) |type_idx| try writeULEB128(&body, self.arena, type_idx);
    try writeSectionHeader(out, self.arena, section_function, body.items.len);
    try out.appendSlice(self.arena, body.items);
}

fn writeExportSection(self: *Module, out: *std.ArrayList(u8)) !void {
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(self.arena);
    try writeULEB128(&body, self.arena, self.exports.items.len);
    for (self.exports.items) |e| {
        try writeULEB128(&body, self.arena, e.name.len);
        try body.appendSlice(self.arena, e.name);
        try body.append(self.arena, e.kind);
        try writeULEB128(&body, self.arena, e.index);
    }
    try writeSectionHeader(out, self.arena, section_export, body.items.len);
    try out.appendSlice(self.arena, body.items);
}

fn writeCodeSection(self: *Module, out: *std.ArrayList(u8)) !void {
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(self.arena);
    try writeULEB128(&body, self.arena, self.code.items.len);
    for (self.code.items) |fn_body| {
        var fb: std.ArrayList(u8) = .empty;
        defer fb.deinit(self.arena);
        try writeULEB128(&fb, self.arena, fn_body.locals.len);
        for (fn_body.locals) |l| {
            try writeULEB128(&fb, self.arena, l.count);
            try fb.append(self.arena, @intFromEnum(l.ty));
        }
        try fb.appendSlice(self.arena, fn_body.body);
        try writeULEB128(&body, self.arena, fb.items.len);
        try body.appendSlice(self.arena, fb.items);
    }
    try writeSectionHeader(out, self.arena, section_code, body.items.len);
    try out.appendSlice(self.arena, body.items);
}

// ── code-body emission helpers ──

pub const Body = struct {
    bytes: std.ArrayList(u8),
    arena: std.mem.Allocator,
    /// when set, register-file accesses are emitted relative to this imported i32 global (the
    /// per-thread register-block base) instead of absolute low memory. null == single-threaded
    /// (base 0), which emits a plain constant — byte-identical to the pre-threads translator.
    reg_base_global: ?u32 = null,

    /// push the linear-memory address of a register-file slot at byte offset `off`. threaded
    /// modules resolve it against the per-thread reg_base global; single-threaded ones use the
    /// absolute offset (base 0). ALL register/SP/PC/NZCV/TPIDR/monitor/vreg accesses go through
    /// this so each thread gets private registers while sharing guest data.
    pub fn emitRegAddr(self: *Body, off: i32) !void {
        if (self.reg_base_global) |idx| {
            try self.emit(.global_get);
            try self.emitU32(idx);
            try self.emitConstI32(off);
            try self.emit(.i32_add);
        } else {
            try self.emitConstI32(off);
        }
    }

    pub fn init(arena: std.mem.Allocator) Body {
        return .{ .bytes = .empty, .arena = arena };
    }

    pub fn emit(self: *Body, op: Op) !void {
        try self.bytes.append(self.arena, @intFromEnum(op));
    }

    pub fn emitU32(self: *Body, value: u32) !void {
        try writeULEB128(&self.bytes, self.arena, value);
    }

    pub fn emitI32(self: *Body, value: i32) !void {
        try writeSLEB128(&self.bytes, self.arena, value);
    }

    pub fn emitI64(self: *Body, value: i64) !void {
        try writeSLEB128(&self.bytes, self.arena, value);
    }

    pub fn emitMemArg(self: *Body, alignment: u32, offset: u32) !void {
        try writeULEB128(&self.bytes, self.arena, alignment);
        try writeULEB128(&self.bytes, self.arena, offset);
    }

    pub fn emitBlock(self: *Body, bt: BlockType) !void {
        try self.bytes.append(self.arena, @intFromEnum(Op.block));
        try self.bytes.append(self.arena, @intFromEnum(bt));
    }

    pub fn emitLoop(self: *Body, bt: BlockType) !void {
        try self.bytes.append(self.arena, @intFromEnum(Op.loop));
        try self.bytes.append(self.arena, @intFromEnum(bt));
    }

    pub fn emitConstI64(self: *Body, v: i64) !void {
        try self.emit(.i64_const);
        try self.emitI64(v);
    }

    pub fn emitConstI32(self: *Body, v: i32) !void {
        try self.emit(.i32_const);
        try self.emitI32(v);
    }

    /// store an i64 value (top of stack) at linear memory offset `addr`.
    /// emits: i32.const addr; (value already on stack); i64.store offset=0 align=3
    /// caller is responsible for putting the address and value on the stack.
    pub fn storeI64(self: *Body, addr: u32) !void {
        try self.emit(.i64_store);
        try self.emitMemArg(3, addr);
    }

    pub fn loadI64(self: *Body, addr: u32) !void {
        try self.emit(.i64_load);
        try self.emitMemArg(3, addr);
    }

    /// emit a threads-proposal atomic op. memarg must follow.
    /// sub_op values from the wasm threads spec (e.g. 0x10=i32.atomic.load, 0x48=i32.atomic.rmw.cmpxchg)
    pub fn emitAtomic(self: *Body, sub_op: u8) !void {
        try self.bytes.append(self.arena, 0xfe);
        try self.bytes.append(self.arena, sub_op);
    }

    /// emit a fixed-length SIMD (v128) op: the 0xfd prefix followed by the LEB128
    /// sub-opcode. any immediates (memarg, lane index, i8x16 shuffle mask) follow.
    pub fn emitSimd(self: *Body, sub_op: u32) !void {
        try self.bytes.append(self.arena, 0xfd);
        try writeULEB128(&self.bytes, self.arena, sub_op);
    }

    /// v128.load from linear memory (address already on stack). alignment is the
    /// log2 byte alignment; V registers are 16-byte aligned so align=4
    pub fn loadV128(self: *Body, addr: u32) !void {
        try self.emitSimd(Simd.v128_load);
        try self.emitMemArg(4, addr);
    }

    /// v128.store to linear memory (address + value on stack)
    pub fn storeV128(self: *Body, addr: u32) !void {
        try self.emitSimd(Simd.v128_store);
        try self.emitMemArg(4, addr);
    }

    pub fn slice(self: *const Body) []const u8 {
        return self.bytes.items;
    }
};

/// wasm fixed-width SIMD sub-opcodes (after the 0xfd prefix). names mirror the
/// spec text. only the ops we emit are listed
pub const Simd = struct {
    pub const v128_load: u32 = 0;
    pub const v128_load8_splat: u32 = 7;
    pub const v128_load16_splat: u32 = 8;
    pub const v128_load32_splat: u32 = 9;
    pub const v128_load64_splat: u32 = 10;
    pub const v128_store: u32 = 11;
    pub const v128_const: u32 = 12;
    pub const i8x16_shuffle: u32 = 13;
    pub const i8x16_splat: u32 = 15;
    pub const i16x8_splat: u32 = 16;
    pub const i32x4_splat: u32 = 17;
    pub const i64x2_splat: u32 = 18;
    pub const f32x4_splat: u32 = 19;
    pub const f64x2_splat: u32 = 20;

    pub const i8x16_extract_lane_s: u32 = 21;
    pub const i8x16_extract_lane_u: u32 = 22;
    pub const i8x16_replace_lane: u32 = 23;
    pub const i16x8_extract_lane_s: u32 = 24;
    pub const i16x8_extract_lane_u: u32 = 25;
    pub const i16x8_replace_lane: u32 = 26;
    pub const i32x4_extract_lane: u32 = 27;
    pub const i32x4_replace_lane: u32 = 28;
    pub const i64x2_extract_lane: u32 = 29;
    pub const i64x2_replace_lane: u32 = 30;
    pub const f32x4_extract_lane: u32 = 31;
    pub const f32x4_replace_lane: u32 = 32;
    pub const f64x2_extract_lane: u32 = 33;
    pub const f64x2_replace_lane: u32 = 34;

    // integer compare (result lanes all-ones / all-zeros, matching NEON)
    pub const i8x16_eq: u32 = 35;
    pub const i8x16_ne: u32 = 36;
    pub const i8x16_lt_s: u32 = 37;
    pub const i8x16_lt_u: u32 = 38;
    pub const i8x16_gt_s: u32 = 39;
    pub const i8x16_gt_u: u32 = 40;
    pub const i8x16_le_s: u32 = 41;
    pub const i8x16_le_u: u32 = 42;
    pub const i8x16_ge_s: u32 = 43;
    pub const i8x16_ge_u: u32 = 44;
    pub const i16x8_eq: u32 = 45;
    pub const i16x8_ne: u32 = 46;
    pub const i16x8_lt_s: u32 = 47;
    pub const i16x8_lt_u: u32 = 48;
    pub const i16x8_gt_s: u32 = 49;
    pub const i16x8_gt_u: u32 = 50;
    pub const i16x8_le_s: u32 = 51;
    pub const i16x8_le_u: u32 = 52;
    pub const i16x8_ge_s: u32 = 53;
    pub const i16x8_ge_u: u32 = 54;
    pub const i32x4_eq: u32 = 55;
    pub const i32x4_ne: u32 = 56;
    pub const i32x4_lt_s: u32 = 57;
    pub const i32x4_lt_u: u32 = 58;
    pub const i32x4_gt_s: u32 = 59;
    pub const i32x4_gt_u: u32 = 60;
    pub const i32x4_le_s: u32 = 61;
    pub const i32x4_le_u: u32 = 62;
    pub const i32x4_ge_s: u32 = 63;
    pub const i32x4_ge_u: u32 = 64;

    // float compare
    pub const f32x4_eq: u32 = 65;
    pub const f32x4_ne: u32 = 66;
    pub const f32x4_lt: u32 = 67;
    pub const f32x4_gt: u32 = 68;
    pub const f32x4_le: u32 = 69;
    pub const f32x4_ge: u32 = 70;
    pub const f64x2_eq: u32 = 71;
    pub const f64x2_ne: u32 = 72;
    pub const f64x2_lt: u32 = 73;
    pub const f64x2_gt: u32 = 74;
    pub const f64x2_le: u32 = 75;
    pub const f64x2_ge: u32 = 76;

    pub const v128_not: u32 = 77;
    pub const v128_and: u32 = 78;
    pub const v128_andnot: u32 = 79;
    pub const v128_or: u32 = 80;
    pub const v128_xor: u32 = 81;
    pub const v128_bitselect: u32 = 82;
    pub const v128_any_true: u32 = 83;

    // i8x16 lane ops
    pub const i8x16_abs: u32 = 96;
    pub const i8x16_neg: u32 = 97;
    pub const i8x16_popcnt: u32 = 98;
    pub const i8x16_all_true: u32 = 99;
    pub const i8x16_bitmask: u32 = 100;
    pub const i8x16_narrow_i16x8_s: u32 = 101;
    pub const i8x16_narrow_i16x8_u: u32 = 102;
    pub const i8x16_shl: u32 = 107;
    pub const i8x16_shr_s: u32 = 108;
    pub const i8x16_shr_u: u32 = 109;
    pub const i8x16_add: u32 = 110;
    pub const i8x16_add_sat_s: u32 = 111;
    pub const i8x16_add_sat_u: u32 = 112;
    pub const i8x16_sub: u32 = 113;
    pub const i8x16_sub_sat_s: u32 = 114;
    pub const i8x16_sub_sat_u: u32 = 115;
    pub const i8x16_min_s: u32 = 118;
    pub const i8x16_min_u: u32 = 119;
    pub const i8x16_max_s: u32 = 120;
    pub const i8x16_max_u: u32 = 121;

    // i16x8 lane ops
    pub const i16x8_extadd_pairwise_i8x16_s: u32 = 124;
    pub const i16x8_extadd_pairwise_i8x16_u: u32 = 125;
    pub const i16x8_abs: u32 = 128;
    pub const i16x8_neg: u32 = 129;
    pub const i16x8_all_true: u32 = 131;
    pub const i16x8_bitmask: u32 = 132;
    pub const i16x8_narrow_i32x4_s: u32 = 133;
    pub const i16x8_narrow_i32x4_u: u32 = 134;
    pub const i16x8_extend_low_i8x16_s: u32 = 135;
    pub const i16x8_extend_high_i8x16_s: u32 = 136;
    pub const i16x8_extend_low_i8x16_u: u32 = 137;
    pub const i16x8_extend_high_i8x16_u: u32 = 138;
    pub const i16x8_shl: u32 = 139;
    pub const i16x8_shr_s: u32 = 140;
    pub const i16x8_shr_u: u32 = 141;
    pub const i16x8_add: u32 = 142;
    pub const i16x8_add_sat_s: u32 = 143;
    pub const i16x8_add_sat_u: u32 = 144;
    pub const i16x8_sub: u32 = 145;
    pub const i16x8_sub_sat_s: u32 = 146;
    pub const i16x8_sub_sat_u: u32 = 147;
    pub const i16x8_mul: u32 = 149;
    pub const i16x8_min_s: u32 = 150;
    pub const i16x8_min_u: u32 = 151;
    pub const i16x8_max_s: u32 = 152;
    pub const i16x8_max_u: u32 = 153;
    pub const i16x8_extmul_low_i8x16_s: u32 = 156;
    pub const i16x8_extmul_high_i8x16_s: u32 = 157;
    pub const i16x8_extmul_low_i8x16_u: u32 = 158;
    pub const i16x8_extmul_high_i8x16_u: u32 = 159;

    // i32x4 lane ops
    pub const i32x4_extadd_pairwise_i16x8_s: u32 = 126;
    pub const i32x4_extadd_pairwise_i16x8_u: u32 = 127;
    pub const i32x4_abs: u32 = 160;
    pub const i32x4_neg: u32 = 161;
    pub const i32x4_all_true: u32 = 163;
    pub const i32x4_bitmask: u32 = 164;
    pub const i32x4_extend_low_i16x8_s: u32 = 167;
    pub const i32x4_extend_high_i16x8_s: u32 = 168;
    pub const i32x4_extend_low_i16x8_u: u32 = 169;
    pub const i32x4_extend_high_i16x8_u: u32 = 170;
    pub const i32x4_shl: u32 = 171;
    pub const i32x4_shr_s: u32 = 172;
    pub const i32x4_shr_u: u32 = 173;
    pub const i32x4_add: u32 = 174;
    pub const i32x4_sub: u32 = 177;
    pub const i32x4_mul: u32 = 181;
    pub const i32x4_min_s: u32 = 182;
    pub const i32x4_min_u: u32 = 183;
    pub const i32x4_max_s: u32 = 184;
    pub const i32x4_max_u: u32 = 185;
    pub const i32x4_dot_i16x8_s: u32 = 186;
    pub const i32x4_extmul_low_i16x8_s: u32 = 188;
    pub const i32x4_extmul_high_i16x8_s: u32 = 189;
    pub const i32x4_extmul_low_i16x8_u: u32 = 190;
    pub const i32x4_extmul_high_i16x8_u: u32 = 191;

    // i64x2 lane ops
    pub const i64x2_abs: u32 = 192;
    pub const i64x2_neg: u32 = 193;
    pub const i64x2_all_true: u32 = 195;
    pub const i64x2_bitmask: u32 = 196;
    pub const i64x2_extend_low_i32x4_s: u32 = 199;
    pub const i64x2_extend_high_i32x4_s: u32 = 200;
    pub const i64x2_extend_low_i32x4_u: u32 = 201;
    pub const i64x2_extend_high_i32x4_u: u32 = 202;
    pub const i64x2_shl: u32 = 203;
    pub const i64x2_shr_s: u32 = 204;
    pub const i64x2_shr_u: u32 = 205;
    pub const i64x2_add: u32 = 206;
    pub const i64x2_sub: u32 = 209;
    pub const i64x2_mul: u32 = 213;
    pub const i64x2_eq: u32 = 214;
    pub const i64x2_ne: u32 = 215;
    pub const i64x2_lt_s: u32 = 216;
    pub const i64x2_gt_s: u32 = 217;
    pub const i64x2_le_s: u32 = 218;
    pub const i64x2_ge_s: u32 = 219;
    pub const i64x2_extmul_low_i32x4_s: u32 = 220;
    pub const i64x2_extmul_high_i32x4_s: u32 = 221;
    pub const i64x2_extmul_low_i32x4_u: u32 = 222;
    pub const i64x2_extmul_high_i32x4_u: u32 = 223;

    // f32x4 lane ops
    pub const f32x4_ceil: u32 = 103;
    pub const f32x4_floor: u32 = 104;
    pub const f32x4_trunc: u32 = 105;
    pub const f32x4_nearest: u32 = 106;
    pub const f32x4_abs: u32 = 224;
    pub const f32x4_neg: u32 = 225;
    pub const f32x4_sqrt: u32 = 227;
    pub const f32x4_add: u32 = 228;
    pub const f32x4_sub: u32 = 229;
    pub const f32x4_mul: u32 = 230;
    pub const f32x4_div: u32 = 231;
    pub const f32x4_min: u32 = 232;
    pub const f32x4_max: u32 = 233;
    pub const f32x4_pmin: u32 = 234;
    pub const f32x4_pmax: u32 = 235;

    // f64x2 lane ops
    pub const f64x2_ceil: u32 = 116;
    pub const f64x2_floor: u32 = 117;
    pub const f64x2_trunc: u32 = 122;
    pub const f64x2_nearest: u32 = 148;
    pub const f64x2_abs: u32 = 236;
    pub const f64x2_neg: u32 = 237;
    pub const f64x2_sqrt: u32 = 239;
    pub const f64x2_add: u32 = 240;
    pub const f64x2_sub: u32 = 241;
    pub const f64x2_mul: u32 = 242;
    pub const f64x2_div: u32 = 243;
    pub const f64x2_min: u32 = 244;
    pub const f64x2_max: u32 = 245;
    pub const f64x2_pmin: u32 = 246;
    pub const f64x2_pmax: u32 = 247;

    // conversions
    pub const f32x4_convert_i32x4_s: u32 = 250;
    pub const f32x4_convert_i32x4_u: u32 = 251;
    pub const i32x4_trunc_sat_f32x4_s: u32 = 248;
    pub const i32x4_trunc_sat_f32x4_u: u32 = 249;
    pub const f64x2_convert_low_i32x4_s: u32 = 254;
    pub const f64x2_convert_low_i32x4_u: u32 = 255;
    pub const f32x4_demote_f64x2_zero: u32 = 94;
    pub const f64x2_promote_low_f32x4: u32 = 95;
};

/// wasm threads-proposal atomic sub-opcodes (after the 0xfe prefix)
pub const Atomic = struct {
    pub const memory_atomic_notify: u8 = 0x00;
    pub const memory_atomic_wait32: u8 = 0x01;
    pub const memory_atomic_wait64: u8 = 0x02;
    pub const atomic_fence: u8 = 0x03;

    pub const i32_atomic_load: u8 = 0x10;
    pub const i64_atomic_load: u8 = 0x11;
    pub const i32_atomic_load8_u: u8 = 0x12;
    pub const i32_atomic_load16_u: u8 = 0x13;
    pub const i64_atomic_load8_u: u8 = 0x14;
    pub const i64_atomic_load16_u: u8 = 0x15;
    pub const i64_atomic_load32_u: u8 = 0x16;
    pub const i32_atomic_store: u8 = 0x17;
    pub const i64_atomic_store: u8 = 0x18;
    pub const i32_atomic_store8: u8 = 0x19;
    pub const i32_atomic_store16: u8 = 0x1a;
    pub const i64_atomic_store8: u8 = 0x1b;
    pub const i64_atomic_store16: u8 = 0x1c;
    pub const i64_atomic_store32: u8 = 0x1d;

    pub const i32_atomic_rmw_add: u8 = 0x1e;
    pub const i64_atomic_rmw_add: u8 = 0x1f;
    pub const i32_atomic_rmw8_add_u: u8 = 0x20;
    pub const i32_atomic_rmw16_add_u: u8 = 0x21;
    pub const i64_atomic_rmw8_add_u: u8 = 0x22;
    pub const i64_atomic_rmw16_add_u: u8 = 0x23;
    pub const i64_atomic_rmw32_add_u: u8 = 0x24;

    pub const i32_atomic_rmw_sub: u8 = 0x25;
    pub const i64_atomic_rmw_sub: u8 = 0x26;
    pub const i32_atomic_rmw8_sub_u: u8 = 0x27;
    pub const i32_atomic_rmw16_sub_u: u8 = 0x28;
    pub const i64_atomic_rmw8_sub_u: u8 = 0x29;
    pub const i64_atomic_rmw16_sub_u: u8 = 0x2a;
    pub const i64_atomic_rmw32_sub_u: u8 = 0x2b;

    pub const i32_atomic_rmw_and: u8 = 0x2c;
    pub const i64_atomic_rmw_and: u8 = 0x2d;
    pub const i32_atomic_rmw8_and_u: u8 = 0x2e;
    pub const i32_atomic_rmw16_and_u: u8 = 0x2f;
    pub const i64_atomic_rmw8_and_u: u8 = 0x30;
    pub const i64_atomic_rmw16_and_u: u8 = 0x31;
    pub const i64_atomic_rmw32_and_u: u8 = 0x32;

    pub const i32_atomic_rmw_or: u8 = 0x33;
    pub const i64_atomic_rmw_or: u8 = 0x34;
    pub const i32_atomic_rmw8_or_u: u8 = 0x35;
    pub const i32_atomic_rmw16_or_u: u8 = 0x36;
    pub const i64_atomic_rmw8_or_u: u8 = 0x37;
    pub const i64_atomic_rmw16_or_u: u8 = 0x38;
    pub const i64_atomic_rmw32_or_u: u8 = 0x39;

    pub const i32_atomic_rmw_xor: u8 = 0x3a;
    pub const i64_atomic_rmw_xor: u8 = 0x3b;
    pub const i32_atomic_rmw8_xor_u: u8 = 0x3c;
    pub const i32_atomic_rmw16_xor_u: u8 = 0x3d;
    pub const i64_atomic_rmw8_xor_u: u8 = 0x3e;
    pub const i64_atomic_rmw16_xor_u: u8 = 0x3f;
    pub const i64_atomic_rmw32_xor_u: u8 = 0x40;

    pub const i32_atomic_rmw_xchg: u8 = 0x41;
    pub const i64_atomic_rmw_xchg: u8 = 0x42;
    pub const i32_atomic_rmw8_xchg_u: u8 = 0x43;
    pub const i32_atomic_rmw16_xchg_u: u8 = 0x44;
    pub const i64_atomic_rmw8_xchg_u: u8 = 0x45;
    pub const i64_atomic_rmw16_xchg_u: u8 = 0x46;
    pub const i64_atomic_rmw32_xchg_u: u8 = 0x47;

    pub const i32_atomic_rmw_cmpxchg: u8 = 0x48;
    pub const i64_atomic_rmw_cmpxchg: u8 = 0x49;
    pub const i32_atomic_rmw8_cmpxchg_u: u8 = 0x4a;
    pub const i32_atomic_rmw16_cmpxchg_u: u8 = 0x4b;
    pub const i64_atomic_rmw8_cmpxchg_u: u8 = 0x4c;
    pub const i64_atomic_rmw16_cmpxchg_u: u8 = 0x4d;
    pub const i64_atomic_rmw32_cmpxchg_u: u8 = 0x4e;
};
