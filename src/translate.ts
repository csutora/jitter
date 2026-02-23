import { WasmModuleBuilder } from './wasm-emit.js';
import {
  ValType, BLOCKTYPE_VOID, HALT_PC,
  regOffset, SP_OFFSET, NZCV_OFFSET, ITER_OFFSET,
} from './types.js';
import type { DecodedInsn, ShiftType, CompiledBlock, CompiledRegion } from './types.js';
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
        else reads.add(SP_LOCAL); // ADD/SUB imm uses SP for reg 31
        if (insn.rd !== 31) writes.add(regLocal(insn.rd));
        else writes.add(SP_LOCAL);
        if (insn.setFlags) { reads.add(NZCV_LOCAL); writes.add(NZCV_LOCAL); }
        break;
      case 'add_reg': case 'sub_reg':
        addRead(reads, insn.rn); addRead(reads, insn.rm);
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
        addRead(reads, insn.rn);
        addWrite(writes, insn.rd);
        if (insn.setFlags) writes.add(NZCV_LOCAL);
        break;
      case 'orr_imm': case 'eor_imm':
        addRead(reads, insn.rn);
        addWrite(writes, insn.rd);
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
      case 'sbfm': case 'ubfm': case 'bfm':
        addRead(reads, insn.rn);
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
    emitSelfLoopBlock(b, block, TMP0, TMP1, TMP2, TMP3);
  } else {
    emitLinearBlock(b, block, TMP0, TMP1, TMP2, TMP3);
    // ── Epilogue: store modified registers back ──
    emitEpilogue(b, usage.writes);
  }

  // Return NEXT_PC
  b.local_get(NEXT_PC_LOCAL);

  b.endFunction();
  b.exportFunction('run');

  const bytes = b.finalize();
  const mod = new WebAssembly.Module(bytes);
  const instance = new WebAssembly.Instance(mod, { env: { memory } });
  const run = instance.exports.run as () => bigint;

  return {
    run,
    startPC: block.startPC,
    endPC: block.endPC,
    insnCount: block.insns.length,
    isSelfLoop: block.isSelfLoop,
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
): void {
  const startPC = Number(block.startPC);
  for (let i = 0; i < block.insns.length; i++) {
    const pc = startPC + i * 4;
    emitInsn(b, block.insns[i], pc, TMP0, TMP1, TMP2, TMP3);
  }
}

// ── Emit a self-looping block ────────────────────────────────────────────────

function emitSelfLoopBlock(
  b: WasmModuleBuilder,
  block: BasicBlock,
  TMP0: number, TMP1: number, TMP2: number, TMP3: number,
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

  // Emit all instructions except the last (and penultimate if fusing)
  const skipLast = canFuse ? 2 : 1;
  for (let i = 0; i < insns.length - skipLast; i++) {
    const pc = startPC + i * 4;
    emitInsn(b, insns[i], pc, TMP0, TMP1, TMP2, TMP3);
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
      emitShift(b, cmpInsn.shift, cmpInsn.amount);
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
    emitShift(b, cmpInsn.shift, cmpInsn.amount);
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
): void {
  switch (insn.kind) {
    case 'add_imm': emitAddSubImm(b, insn, true, TMP0, TMP1, TMP2); break;
    case 'sub_imm': emitAddSubImm(b, insn, false, TMP0, TMP1, TMP2); break;
    case 'add_reg': emitAddSubReg(b, insn, true, TMP0, TMP1, TMP2); break;
    case 'sub_reg': emitAddSubReg(b, insn, false, TMP0, TMP1, TMP2); break;
    case 'orr_reg': emitLogicalReg(b, insn); break;
    case 'eor_reg': emitLogicalReg(b, insn); break;
    case 'and_reg': emitLogicalReg(b, insn); break;
    case 'and_imm': emitLogicalImm(b, insn); break;
    case 'orr_imm': emitLogicalImm(b, insn); break;
    case 'eor_imm': emitLogicalImm(b, insn); break;
    case 'movz': emitMovz(b, insn); break;
    case 'movk': emitMovk(b, insn); break;
    case 'movn': emitMovn(b, insn); break;
    case 'madd': emitMadd(b, insn); break;
    case 'msub': emitMsub(b, insn); break;
    case 'sdiv': emitDiv(b, insn, true); break;
    case 'udiv': emitDiv(b, insn, false); break;
    case 'sbfm': emitBitfield(b, insn); break;
    case 'ubfm': emitBitfield(b, insn); break;
    case 'csel': emitCondSelect(b, insn); break;
    case 'ldr_imm': emitLdrImm(b, insn); break;
    case 'str_imm': emitStrImm(b, insn); break;

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
      b.i64_const(HALT_PC);
      b.local_set(NEXT_PC_LOCAL);
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
    emitGetRegSP(b, insn.rn, insn.sf);
    b.local_tee(TMP0); // op1
    b.i64_const(BigInt(insn.imm));
    b.local_tee(TMP1); // op2
    if (isAdd) b.i64_add(); else b.i64_sub();
    if (insn.sf === 0) { b.i64_const(0xFFFFFFFFn); b.i64_and(); }
    b.local_tee(TMP2); // result
    emitSetRegSP(b, insn.rd, insn.sf);
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
    emitShift(b, insn.shift, insn.amount);
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
    emitShift(b, insn.shift, insn.amount);
    if (isAdd) b.i64_add(); else b.i64_sub();
    emitSetReg(b, insn.rd, insn.sf);
  }
}

// ── Shift operand ────────────────────────────────────────────────────────────

function emitShift(b: WasmModuleBuilder, shift: ShiftType, amount: number): void {
  if (amount === 0) return;
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
): void {
  emitGetReg(b, insn.rn, insn.sf);
  emitGetReg(b, insn.rm, insn.sf);
  emitShift(b, insn.shift, insn.amount);
  if (insn.invert) { b.i64_const(-1n); b.i64_xor(); }

  switch (insn.kind) {
    case 'orr_reg': b.i64_or(); break;
    case 'eor_reg': b.i64_xor(); break;
    case 'and_reg': b.i64_and(); break;
  }
  emitSetReg(b, insn.rd, insn.sf);
}

// ── Logical immediate ────────────────────────────────────────────────────────

function emitLogicalImm(
  b: WasmModuleBuilder,
  insn: DecodedInsn & { kind: 'and_imm' | 'orr_imm' | 'eor_imm' },
): void {
  emitGetReg(b, insn.rn, insn.sf);
  b.i64_const(insn.imm);
  switch (insn.kind) {
    case 'and_imm': b.i64_and(); break;
    case 'orr_imm': b.i64_or(); break;
    case 'eor_imm': b.i64_xor(); break;
  }
  emitSetReg(b, insn.rd, insn.sf);
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
      // LSL: shift = regSize - immr, width = imms + 1
      const shift = regSize - insn.immr;
      emitGetReg(b, insn.rn, insn.sf);
      b.i64_const(BigInt(shift));
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
    } else {
      // General SBFX
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
    // BFM: general bitfield move — complex, just zero for now
    b.i64_const(0n);
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
  } else {
    // Compute address: base + offset + GUEST_MEM_BASE
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
        blockIdLocal, TMP0, TMP1, TMP2, TMP3);
    } else {
      emitRegionLinearBlock(b, block, i, blockMap, dispatchDepth, exitDepth,
        blockIdLocal, TMP0, TMP1, TMP2, TMP3);
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
  const instance = new WebAssembly.Instance(mod, { env: { memory } });
  const run = instance.exports.run as () => bigint;

  return {
    run,
    entryPC: blocks[0].startPC,
    blockPCs: blocks.map(bl => bl.startPC),
  };
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
): void {
  const startPC = Number(block.startPC);
  const insns = block.insns;

  // Emit all non-branch instructions
  for (let i = 0; i < insns.length - 1; i++) {
    const pc = startPC + i * 4;
    emitInsn(b, insns[i], pc, TMP0, TMP1, TMP2, TMP3);
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
    emitInsn(b, insns[i], pc, TMP0, TMP1, TMP2, TMP3);
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
  }

  // Fall through self-loop: determine where the fallthrough goes
  if (canFuse) {
    // Recompute NZCV on exit
    const cmpInsn = penult!;
    if (cmpInsn.kind === 'sub_reg') {
      emitGetReg(b, cmpInsn.rn, cmpInsn.sf);
      b.local_tee(TMP0);
      emitGetReg(b, cmpInsn.rm, cmpInsn.sf);
      emitShift(b, cmpInsn.shift, cmpInsn.amount);
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
