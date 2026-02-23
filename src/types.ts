// ── Shift types ──────────────────────────────────────────────────────────────
export type ShiftType = 'lsl' | 'lsr' | 'asr' | 'ror';

// ── Condition codes ──────────────────────────────────────────────────────────
export const enum Cond {
  EQ = 0,  NE = 1,  CS = 2,  CC = 3,
  MI = 4,  PL = 5,  VS = 6,  VC = 7,
  HI = 8,  LS = 9,  GE = 10, LT = 11,
  GT = 12, LE = 13, AL = 14, NV = 15,
}

// ── Decoded instruction (discriminated union) ────────────────────────────────
export type DecodedInsn =
  | { kind: 'add_imm';   sf: 0|1; rd: number; rn: number; imm: number; setFlags: boolean }
  | { kind: 'sub_imm';   sf: 0|1; rd: number; rn: number; imm: number; setFlags: boolean }
  | { kind: 'add_reg';   sf: 0|1; rd: number; rn: number; rm: number; shift: ShiftType; amount: number; setFlags: boolean }
  | { kind: 'sub_reg';   sf: 0|1; rd: number; rn: number; rm: number; shift: ShiftType; amount: number; setFlags: boolean }
  | { kind: 'and_imm';   sf: 0|1; rd: number; rn: number; imm: bigint; setFlags: boolean }
  | { kind: 'orr_imm';   sf: 0|1; rd: number; rn: number; imm: bigint }
  | { kind: 'eor_imm';   sf: 0|1; rd: number; rn: number; imm: bigint }
  | { kind: 'and_reg';   sf: 0|1; rd: number; rn: number; rm: number; shift: ShiftType; amount: number; invert: boolean; setFlags: boolean }
  | { kind: 'orr_reg';   sf: 0|1; rd: number; rn: number; rm: number; shift: ShiftType; amount: number; invert: boolean }
  | { kind: 'eor_reg';   sf: 0|1; rd: number; rn: number; rm: number; shift: ShiftType; amount: number; invert: boolean }
  | { kind: 'movz';      sf: 0|1; rd: number; imm16: number; hw: number }
  | { kind: 'movk';      sf: 0|1; rd: number; imm16: number; hw: number }
  | { kind: 'movn';      sf: 0|1; rd: number; imm16: number; hw: number }
  | { kind: 'sbfm';      sf: 0|1; rd: number; rn: number; immr: number; imms: number }
  | { kind: 'ubfm';      sf: 0|1; rd: number; rn: number; immr: number; imms: number }
  | { kind: 'bfm';       sf: 0|1; rd: number; rn: number; immr: number; imms: number }
  | { kind: 'madd';      sf: 0|1; rd: number; rn: number; rm: number; ra: number }
  | { kind: 'msub';      sf: 0|1; rd: number; rn: number; rm: number; ra: number }
  | { kind: 'sdiv';      sf: 0|1; rd: number; rn: number; rm: number }
  | { kind: 'udiv';      sf: 0|1; rd: number; rn: number; rm: number }
  | { kind: 'smulh';     rd: number; rn: number; rm: number }
  | { kind: 'umulh';     rd: number; rn: number; rm: number }
  | { kind: 'csel';      sf: 0|1; rd: number; rn: number; rm: number; cond: number; variant: 'csel'|'csinc'|'csinv'|'csneg' }
  | { kind: 'ldr_imm';   sf: 0|1; rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'str_imm';   sf: 0|1; rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'ldr_reg';   sf: 0|1; rt: number; rn: number; rm: number; extend: string; shift: number }
  | { kind: 'str_reg';   sf: 0|1; rt: number; rn: number; rm: number; extend: string; shift: number }
  | { kind: 'ldrb_imm';  rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'strb_imm';  rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'ldrh_imm';  rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'strh_imm';  rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'ldrsw_imm'; rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'ldrsh_imm'; sf: 0|1; rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'ldrsb_imm'; sf: 0|1; rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'ldp';       sf: 0|1; rt: number; rt2: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'stp';       sf: 0|1; rt: number; rt2: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'adr';       rd: number; imm: number }
  | { kind: 'adrp';      rd: number; imm: number }
  | { kind: 'b';         offset: number }
  | { kind: 'bl';        offset: number }
  | { kind: 'b_cond';    cond: number; offset: number }
  | { kind: 'cbz';       sf: 0|1; rt: number; offset: number }
  | { kind: 'cbnz';      sf: 0|1; rt: number; offset: number }
  | { kind: 'tbz';       rt: number; bit: number; offset: number }
  | { kind: 'tbnz';      rt: number; bit: number; offset: number }
  | { kind: 'br';        rn: number }
  | { kind: 'blr';       rn: number }
  | { kind: 'ret';       rn: number }
  | { kind: 'svc';       imm16: number }
  | { kind: 'nop' }
  | { kind: 'unknown';   raw: number };

// ── Instructions that terminate a basic block ────────────────────────────────
export function isBranch(insn: DecodedInsn): boolean {
  switch (insn.kind) {
    case 'b': case 'bl': case 'b_cond': case 'cbz': case 'cbnz':
    case 'tbz': case 'tbnz': case 'br': case 'blr': case 'ret':
    case 'svc':
      return true;
    default:
      return false;
  }
}

// ── CPU state memory layout (offsets in bytes) ───────────────────────────────
export const REG_OFFSET = 0;          // X0 at 0, X1 at 8, ..., X30 at 240
export const SP_OFFSET = 0xF8;        // 248
export const PC_OFFSET = 0x100;       // 256
export const NZCV_OFFSET = 0x108;     // 264
export const ITER_OFFSET = 0x110;     // 272 — self-loop iteration count
export const STATE_SIZE = 0x200;      // 512 bytes reserved for CPU state
export const GUEST_MEM_BASE = 0x10000; // guest memory starts at 64KB

export function regOffset(r: number): number {
  if (r === 31) return SP_OFFSET;
  return REG_OFFSET + r * 8;
}

// ── WASM value types ─────────────────────────────────────────────────────────
export const enum ValType {
  I32 = 0x7F,
  I64 = 0x7E,
  F32 = 0x7D,
  F64 = 0x7C,
}

// ── WASM opcodes ─────────────────────────────────────────────────────────────
export const enum Op {
  // Control
  Unreachable   = 0x00,
  Nop           = 0x01,
  Block         = 0x02,
  Loop          = 0x03,
  If            = 0x04,
  Else          = 0x05,
  End           = 0x0B,
  Br            = 0x0C,
  BrIf          = 0x0D,
  BrTable       = 0x0E,
  Return        = 0x0F,
  Call          = 0x10,
  CallIndirect  = 0x11,

  // Variables
  LocalGet      = 0x20,
  LocalSet      = 0x21,
  LocalTee      = 0x22,

  // Memory (i32)
  I32Load       = 0x28,
  I64Load       = 0x29,
  I32Load8S     = 0x2C,
  I32Load8U     = 0x2D,
  I32Load16S    = 0x2E,
  I32Load16U    = 0x2F,
  I64Load8S     = 0x30,
  I64Load8U     = 0x31,
  I64Load16S    = 0x32,
  I64Load16U    = 0x33,
  I64Load32S    = 0x34,
  I64Load32U    = 0x35,
  I32Store      = 0x36,
  I64Store      = 0x37,
  I32Store8     = 0x3A,
  I32Store16    = 0x3B,
  I64Store8     = 0x3C,
  I64Store16    = 0x3D,
  I64Store32    = 0x3E,

  // Constants
  I32Const      = 0x41,
  I64Const      = 0x42,

  // i32 comparison
  I32Eqz        = 0x45,
  I32Eq         = 0x46,
  I32Ne         = 0x47,

  // i64 comparison
  I64Eqz        = 0x50,
  I64Eq         = 0x51,
  I64Ne         = 0x52,
  I64LtS        = 0x53,
  I64LtU        = 0x54,
  I64GtS        = 0x55,
  I64GtU        = 0x56,
  I64LeS        = 0x57,
  I64LeU        = 0x58,
  I64GeS        = 0x59,
  I64GeU        = 0x5A,

  // i32 arithmetic
  I32Add        = 0x6A,
  I32Sub        = 0x6B,

  // i64 unary
  I64Clz        = 0x79,
  I64Ctz        = 0x7A,
  I64Popcnt     = 0x7B,

  // i64 arithmetic
  I64Add        = 0x7C,
  I64Sub        = 0x7D,
  I64Mul        = 0x7E,
  I64DivS       = 0x7F,
  I64DivU       = 0x80,
  I64RemS       = 0x81,
  I64RemU       = 0x82,
  I64And        = 0x83,
  I64Or         = 0x84,
  I64Xor        = 0x85,
  I64Shl        = 0x86,
  I64ShrS       = 0x87,
  I64ShrU       = 0x88,
  I64Rotl       = 0x89,
  I64Rotr       = 0x8A,

  // Conversions
  I32WrapI64    = 0xA7,
  I64ExtendI32S = 0xAC,
  I64ExtendI32U = 0xAD,
}

// Block types for structured control flow
export const BLOCKTYPE_VOID = 0x40;
export const BLOCKTYPE_I32 = ValType.I32;
export const BLOCKTYPE_I64 = ValType.I64;

// ── Halt sentinel ────────────────────────────────────────────────────────────
export const HALT_PC = 0xDEAD_0000n;

// ── Compiled block ───────────────────────────────────────────────────────────
export interface CompiledBlock {
  run: () => bigint;
  startPC: bigint;
  endPC: bigint;
  insnCount: number;
  isSelfLoop: boolean;
}

// ── Compiled region ─────────────────────────────────────────────────────────
export interface CompiledRegion {
  run: () => bigint;       // returns next PC (outside region)
  entryPC: bigint;
  blockPCs: bigint[];      // all block start PCs in this region
}
