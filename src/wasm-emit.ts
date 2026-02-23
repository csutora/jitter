import { ValType, Op, BLOCKTYPE_VOID } from './types.js';

// ── LEB128 encoders ──────────────────────────────────────────────────────────

export function encodeULEB128(value: number): number[] {
  const bytes: number[] = [];
  do {
    let byte = value & 0x7F;
    value >>>= 7;
    if (value !== 0) byte |= 0x80;
    bytes.push(byte);
  } while (value !== 0);
  return bytes;
}

export function encodeSLEB128(value: number): number[] {
  const bytes: number[] = [];
  let more = true;
  while (more) {
    let byte = value & 0x7F;
    value >>= 7;
    if ((value === 0 && (byte & 0x40) === 0) || (value === -1 && (byte & 0x40) !== 0)) {
      more = false;
    } else {
      byte |= 0x80;
    }
    bytes.push(byte);
  }
  return bytes;
}

export function encodeSLEB128_i64(value: bigint): number[] {
  value = BigInt.asIntN(64, value); // ensure signed 64-bit representation
  const bytes: number[] = [];
  let more = true;
  while (more) {
    let byte = Number(value & 0x7Fn);
    value >>= 7n;
    if ((value === 0n && (byte & 0x40) === 0) || (value === -1n && (byte & 0x40) !== 0)) {
      more = false;
    } else {
      byte |= 0x80;
    }
    bytes.push(byte);
  }
  return bytes;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function encodeString(s: string): number[] {
  const utf8 = new TextEncoder().encode(s);
  return [...encodeULEB128(utf8.length), ...utf8];
}

function encodeSection(id: number, content: number[]): number[] {
  return [id, ...encodeULEB128(content.length), ...content];
}

// ── WASM Module Builder ──────────────────────────────────────────────────────

export class WasmModuleBuilder {
  // Section data
  private types: number[][] = [];         // encoded function types
  private imports: number[] = [];         // encoded imports
  private importCount = 0;
  private funcTypeIndices: number[] = [];  // type index per function
  private exports: number[] = [];
  private exportCount = 0;

  // Current function being built
  private code: number[] = [];            // instruction bytecode
  private localGroups: Array<{ count: number; type: number }> = [];
  private localCount = 0;                 // total locals (params + declared)
  private paramCount = 0;
  private funcBodies: number[][] = [];    // finalized function bodies

  // ── Types ──────────────────────────────────────────────────────────────────

  addFuncType(params: ValType[], results: ValType[]): number {
    const encoded = [
      0x60, // function type marker
      ...encodeULEB128(params.length), ...params,
      ...encodeULEB128(results.length), ...results,
    ];
    this.types.push(encoded);
    return this.types.length - 1;
  }

  // ── Imports ────────────────────────────────────────────────────────────────

  addMemoryImport(module: string, name: string, minPages: number, maxPages?: number): void {
    const importData = [
      ...encodeString(module),
      ...encodeString(name),
      0x02, // import kind: memory
    ];
    if (maxPages !== undefined) {
      importData.push(0x01, ...encodeULEB128(minPages), ...encodeULEB128(maxPages));
    } else {
      importData.push(0x00, ...encodeULEB128(minPages));
    }
    this.imports.push(...importData);
    this.importCount++;
  }

  addFunctionImport(module: string, name: string, typeIndex: number): number {
    const importData = [
      ...encodeString(module),
      ...encodeString(name),
      0x00, // import kind: function
      ...encodeULEB128(typeIndex),
    ];
    this.imports.push(...importData);
    const funcIdx = this.importCount;
    this.importCount++;
    return funcIdx;
  }

  // ── Functions ──────────────────────────────────────────────────────────────

  beginFunction(typeIndex: number): void {
    this.funcTypeIndices.push(typeIndex);
    this.code = [];
    this.localGroups = [];
    this.localCount = 0;
    // Count params from the type
    const typeData = this.types[typeIndex];
    // typeData: [0x60, paramCount(uleb), ...paramTypes, resultCount(uleb), ...resultTypes]
    // Simple parse: skip 0x60, read param count
    this.paramCount = typeData[1]; // works for param count < 128
    this.localCount = this.paramCount;
  }

  addLocals(type: ValType, count: number): number {
    const firstIndex = this.localCount;
    this.localGroups.push({ count, type });
    this.localCount += count;
    return firstIndex;
  }

  exportFunction(name: string, funcIndex?: number): void {
    const idx = funcIndex ?? this.funcTypeIndices.length - 1;
    this.exports.push(
      ...encodeString(name),
      0x00, // export kind: function
      ...encodeULEB128(idx),
    );
    this.exportCount++;
  }

  endFunction(): void {
    // Encode local declarations
    const localDecls: number[] = [
      ...encodeULEB128(this.localGroups.length),
    ];
    for (const g of this.localGroups) {
      localDecls.push(...encodeULEB128(g.count), g.type);
    }
    // Function body = locals + code + end
    const body = [...localDecls, ...this.code, Op.End];
    const bodyWithSize = [...encodeULEB128(body.length), ...body];
    this.funcBodies.push(bodyWithSize);
  }

  // ── Instruction emission ───────────────────────────────────────────────────

  // Constants
  i32_const(value: number): void { this.code.push(Op.I32Const, ...encodeSLEB128(value)); }
  i64_const(value: bigint): void { this.code.push(Op.I64Const, ...encodeSLEB128_i64(value)); }

  // Variables
  local_get(idx: number): void { this.code.push(Op.LocalGet, ...encodeULEB128(idx)); }
  local_set(idx: number): void { this.code.push(Op.LocalSet, ...encodeULEB128(idx)); }
  local_tee(idx: number): void { this.code.push(Op.LocalTee, ...encodeULEB128(idx)); }

  // Memory
  i32_load(align: number, offset: number): void {
    this.code.push(Op.I32Load, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i64_load(align: number, offset: number): void {
    this.code.push(Op.I64Load, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i32_store(align: number, offset: number): void {
    this.code.push(Op.I32Store, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i64_store(align: number, offset: number): void {
    this.code.push(Op.I64Store, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i32_load8_s(align: number, offset: number): void {
    this.code.push(Op.I32Load8S, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i32_load8_u(align: number, offset: number): void {
    this.code.push(Op.I32Load8U, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i32_load16_s(align: number, offset: number): void {
    this.code.push(Op.I32Load16S, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i32_load16_u(align: number, offset: number): void {
    this.code.push(Op.I32Load16U, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i64_load8_s(align: number, offset: number): void {
    this.code.push(Op.I64Load8S, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i64_load8_u(align: number, offset: number): void {
    this.code.push(Op.I64Load8U, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i64_load16_s(align: number, offset: number): void {
    this.code.push(Op.I64Load16S, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i64_load16_u(align: number, offset: number): void {
    this.code.push(Op.I64Load16U, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i64_load32_s(align: number, offset: number): void {
    this.code.push(Op.I64Load32S, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i64_load32_u(align: number, offset: number): void {
    this.code.push(Op.I64Load32U, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i64_store8(align: number, offset: number): void {
    this.code.push(Op.I64Store8, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i64_store16(align: number, offset: number): void {
    this.code.push(Op.I64Store16, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i64_store32(align: number, offset: number): void {
    this.code.push(Op.I64Store32, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i32_store8(align: number, offset: number): void {
    this.code.push(Op.I32Store8, ...encodeULEB128(align), ...encodeULEB128(offset));
  }
  i32_store16(align: number, offset: number): void {
    this.code.push(Op.I32Store16, ...encodeULEB128(align), ...encodeULEB128(offset));
  }

  // i32 comparison
  i32_eqz(): void { this.code.push(Op.I32Eqz); }
  i32_eq(): void { this.code.push(Op.I32Eq); }
  i32_ne(): void { this.code.push(Op.I32Ne); }

  // i32 arithmetic
  i32_add(): void { this.code.push(Op.I32Add); }
  i32_sub(): void { this.code.push(Op.I32Sub); }

  // i64 comparison
  i64_eqz(): void { this.code.push(Op.I64Eqz); }
  i64_eq(): void { this.code.push(Op.I64Eq); }
  i64_ne(): void { this.code.push(Op.I64Ne); }
  i64_lt_s(): void { this.code.push(Op.I64LtS); }
  i64_lt_u(): void { this.code.push(Op.I64LtU); }
  i64_gt_s(): void { this.code.push(Op.I64GtS); }
  i64_gt_u(): void { this.code.push(Op.I64GtU); }
  i64_le_s(): void { this.code.push(Op.I64LeS); }
  i64_le_u(): void { this.code.push(Op.I64LeU); }
  i64_ge_s(): void { this.code.push(Op.I64GeS); }
  i64_ge_u(): void { this.code.push(Op.I64GeU); }

  // i64 unary
  i64_clz(): void { this.code.push(Op.I64Clz); }
  i64_ctz(): void { this.code.push(Op.I64Ctz); }
  i64_popcnt(): void { this.code.push(Op.I64Popcnt); }

  // i64 arithmetic
  i64_add(): void { this.code.push(Op.I64Add); }
  i64_sub(): void { this.code.push(Op.I64Sub); }
  i64_mul(): void { this.code.push(Op.I64Mul); }
  i64_div_s(): void { this.code.push(Op.I64DivS); }
  i64_div_u(): void { this.code.push(Op.I64DivU); }
  i64_rem_s(): void { this.code.push(Op.I64RemS); }
  i64_rem_u(): void { this.code.push(Op.I64RemU); }
  i64_and(): void { this.code.push(Op.I64And); }
  i64_or(): void { this.code.push(Op.I64Or); }
  i64_xor(): void { this.code.push(Op.I64Xor); }
  i64_shl(): void { this.code.push(Op.I64Shl); }
  i64_shr_s(): void { this.code.push(Op.I64ShrS); }
  i64_shr_u(): void { this.code.push(Op.I64ShrU); }
  i64_rotl(): void { this.code.push(Op.I64Rotl); }
  i64_rotr(): void { this.code.push(Op.I64Rotr); }

  // Conversions
  i32_wrap_i64(): void { this.code.push(Op.I32WrapI64); }
  i64_extend_i32_s(): void { this.code.push(Op.I64ExtendI32S); }
  i64_extend_i32_u(): void { this.code.push(Op.I64ExtendI32U); }

  // Control flow
  block(blocktype: number): void { this.code.push(Op.Block, blocktype); }
  loop(blocktype: number): void { this.code.push(Op.Loop, blocktype); }
  if_(blocktype: number): void { this.code.push(Op.If, blocktype); }
  else_(): void { this.code.push(Op.Else); }
  end(): void { this.code.push(Op.End); }
  br(depth: number): void { this.code.push(Op.Br, ...encodeULEB128(depth)); }
  br_if(depth: number): void { this.code.push(Op.BrIf, ...encodeULEB128(depth)); }
  br_table(labels: number[], defaultLabel: number): void {
    this.code.push(Op.BrTable, ...encodeULEB128(labels.length));
    for (const label of labels) {
      this.code.push(...encodeULEB128(label));
    }
    this.code.push(...encodeULEB128(defaultLabel));
  }
  return_(): void { this.code.push(Op.Return); }
  unreachable(): void { this.code.push(Op.Unreachable); }
  call(funcIdx: number): void { this.code.push(Op.Call, ...encodeULEB128(funcIdx)); }
  drop(): void { this.code.push(0x1A); }

  // ── Finalize ───────────────────────────────────────────────────────────────

  finalize(): Uint8Array {
    const sections: number[] = [];

    // Magic + version
    sections.push(0x00, 0x61, 0x73, 0x6D); // \0asm
    sections.push(0x01, 0x00, 0x00, 0x00);  // version 1

    // Type section
    if (this.types.length > 0) {
      const content = [
        ...encodeULEB128(this.types.length),
        ...this.types.flat(),
      ];
      sections.push(...encodeSection(1, content));
    }

    // Import section
    if (this.importCount > 0) {
      const content = [
        ...encodeULEB128(this.importCount),
        ...this.imports,
      ];
      sections.push(...encodeSection(2, content));
    }

    // Function section
    if (this.funcTypeIndices.length > 0) {
      const content = [
        ...encodeULEB128(this.funcTypeIndices.length),
        ...this.funcTypeIndices.flatMap(i => encodeULEB128(i)),
      ];
      sections.push(...encodeSection(3, content));
    }

    // Export section
    if (this.exportCount > 0) {
      const content = [
        ...encodeULEB128(this.exportCount),
        ...this.exports,
      ];
      sections.push(...encodeSection(7, content));
    }

    // Code section
    if (this.funcBodies.length > 0) {
      const content = [
        ...encodeULEB128(this.funcBodies.length),
        ...this.funcBodies.flat(),
      ];
      sections.push(...encodeSection(10, content));
    }

    return new Uint8Array(sections);
  }
}
