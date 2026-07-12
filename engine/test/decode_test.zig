const std = @import("std");
const testing = std.testing;
const decode = @import("decode");

// helper: tag check
fn expectTag(insn: u32, expected: decode.Kind) !void {
    const op = decode.decode(insn);
    try testing.expectEqual(expected, std.meta.activeTag(op));
}

test "add immediate" {
    // ADD X1, X2, #1 → 91 00 04 41
    const insn: u32 = 0x91000441;
    const op = decode.decode(insn);
    try testing.expect(op == .add_imm);
    try testing.expectEqual(@as(u5, 1), op.add_imm.rd);
    try testing.expectEqual(@as(u5, 2), op.add_imm.rn);
    try testing.expectEqual(@as(u32, 1), op.add_imm.imm);
    try testing.expect(op.add_imm.sf);
    try testing.expect(!op.add_imm.set_flags);
}

test "subs immediate (CMP)" {
    // SUBS XZR, X3, #4 → f1 00 10 7f
    const insn: u32 = 0xf100107f;
    const op = decode.decode(insn);
    try testing.expect(op == .sub_imm);
    try testing.expect(op.sub_imm.set_flags);
    try testing.expectEqual(@as(u5, 31), op.sub_imm.rd); // zr
}

test "movz x0, #0x1234" {
    // MOVZ X0, #0x1234 → d2 82 46 80
    const insn: u32 = 0xd2824680;
    const op = decode.decode(insn);
    try testing.expect(op == .movz);
    try testing.expectEqual(@as(u16, 0x1234), op.movz.imm16);
    try testing.expectEqual(@as(u2, 0), op.movz.hw);
}

test "b #+8" {
    // B .+8 → 14 00 00 02
    const insn: u32 = 0x14000002;
    const op = decode.decode(insn);
    try testing.expect(op == .b);
    try testing.expectEqual(@as(i32, 8), op.b.offset);
}

test "b.eq #+12" {
    // B.EQ .+12 → 54 00 00 60
    const insn: u32 = 0x54000060;
    const op = decode.decode(insn);
    try testing.expect(op == .b_cond);
    try testing.expectEqual(@as(u4, 0), op.b_cond.cond);
    try testing.expectEqual(@as(i32, 12), op.b_cond.offset);
}

test "ret" {
    // RET → d6 5f 03 c0
    const insn: u32 = 0xd65f03c0;
    try expectTag(insn, .ret);
}

test "svc #0" {
    // SVC #0 → d4 00 00 01
    const insn: u32 = 0xd4000001;
    const op = decode.decode(insn);
    try testing.expect(op == .svc);
    try testing.expectEqual(@as(u16, 0), op.svc.imm);
}

test "nop" {
    // NOP → d5 03 20 1f (HINT 0)
    const insn: u32 = 0xd503201f;
    const op = decode.decode(insn);
    // hint with op2=0 is nop; for now decoded as hint
    _ = op;
}

test "ldr x0, [sp, #16]" {
    // LDR X0, [SP, #16] → f9 40 0b e0
    const insn: u32 = 0xf9400be0;
    const op = decode.decode(insn);
    try testing.expect(op == .ldr_imm);
    try testing.expectEqual(@as(u5, 0), op.ldr_imm.rt);
    try testing.expectEqual(@as(u5, 31), op.ldr_imm.rn); // sp
    try testing.expectEqual(@as(i32, 16), op.ldr_imm.offset);
    try testing.expectEqual(decode.AddrMode.offset_unsigned, op.ldr_imm.mode);
}

test "str x0, [sp, #-16]!" {
    // STR X0, [SP, #-16]! → f8 1f 0f e0
    const insn: u32 = 0xf81f0fe0;
    const op = decode.decode(insn);
    try testing.expect(op == .str_imm);
    try testing.expectEqual(@as(i32, -16), op.str_imm.offset);
    try testing.expectEqual(decode.AddrMode.pre_index, op.str_imm.mode);
}

test "stp x29, x30, [sp, #-16]!" {
    // STP X29, X30, [SP, #-16]! → a9 bf 7b fd
    const insn: u32 = 0xa9bf7bfd;
    const op = decode.decode(insn);
    try testing.expect(op == .stp);
    try testing.expectEqual(@as(u5, 29), op.stp.rt);
    try testing.expectEqual(@as(u5, 30), op.stp.rt2);
    try testing.expectEqual(@as(i32, -16), op.stp.offset);
    try testing.expectEqual(decode.AddrMode.pre_index, op.stp.mode);
}

test "add x0, x1, x2" {
    // ADD X0, X1, X2 → 8b 02 00 20
    const insn: u32 = 0x8b020020;
    const op = decode.decode(insn);
    try testing.expect(op == .add_reg);
    try testing.expectEqual(@as(u5, 0), op.add_reg.rd);
    try testing.expectEqual(@as(u5, 1), op.add_reg.rn);
    try testing.expectEqual(@as(u5, 2), op.add_reg.rm);
    try testing.expect(!op.add_reg.set_flags);
}

test "and x0, x1, #0xff" {
    // AND X0, X1, #0xff → 92 40 1c 20
    const insn: u32 = 0x92401c20;
    const op = decode.decode(insn);
    try testing.expect(op == .and_imm);
    try testing.expectEqual(@as(u64, 0xff), op.and_imm.imm);
}

test "csel x0, x1, x2, eq" {
    // CSEL X0, X1, X2, EQ → 9a 82 00 20
    const insn: u32 = 0x9a820020;
    const op = decode.decode(insn);
    try testing.expect(op == .csel);
    try testing.expectEqual(@as(u4, 0), op.csel.cond);
}

test "cbz w0, #+4" {
    // CBZ W0, .+4 → 34 00 00 20
    const insn: u32 = 0x34000020;
    const op = decode.decode(insn);
    try testing.expect(op == .cbz);
    try testing.expectEqual(@as(u5, 0), op.cbz.rt);
    try testing.expect(!op.cbz.sf);
}

test "ldr x0, =literal at .+8" {
    // LDR X0, .+8 → 58 00 00 40
    const insn: u32 = 0x58000040;
    const op = decode.decode(insn);
    try testing.expect(op == .ldr_lit);
    try testing.expect(op.ldr_lit.sf);
    try testing.expectEqual(@as(i32, 8), op.ldr_lit.offset);
}

test "unknown returns insn payload" {
    const insn: u32 = 0x00000000;
    const op = decode.decode(insn);
    try testing.expect(op == .unknown);
    try testing.expectEqual(insn, op.unknown);
}
