import { describe, it, expect } from 'vitest';
import { WasmModuleBuilder, encodeULEB128, encodeSLEB128, encodeSLEB128_i64 } from '../src/wasm-emit.js';
import { ValType, BLOCKTYPE_I64, BLOCKTYPE_VOID } from '../src/types.js';

describe('LEB128', () => {
  it('encodes unsigned values', () => {
    expect(encodeULEB128(0)).toEqual([0]);
    expect(encodeULEB128(1)).toEqual([1]);
    expect(encodeULEB128(127)).toEqual([127]);
    expect(encodeULEB128(128)).toEqual([0x80, 0x01]);
    expect(encodeULEB128(624485)).toEqual([0xE5, 0x8E, 0x26]);
  });

  it('encodes signed i32 values', () => {
    expect(encodeSLEB128(0)).toEqual([0]);
    expect(encodeSLEB128(1)).toEqual([1]);
    expect(encodeSLEB128(-1)).toEqual([0x7F]);
    expect(encodeSLEB128(63)).toEqual([63]);
    expect(encodeSLEB128(64)).toEqual([0xC0, 0x00]);
    expect(encodeSLEB128(-64)).toEqual([0x40]);
    expect(encodeSLEB128(-65)).toEqual([0xBF, 0x7F]);
  });

  it('encodes signed i64 values', () => {
    expect(encodeSLEB128_i64(0n)).toEqual([0]);
    expect(encodeSLEB128_i64(1n)).toEqual([1]);
    expect(encodeSLEB128_i64(-1n)).toEqual([0x7F]);
    expect(encodeSLEB128_i64(0x7FFFFFFFFFFFFFFFn)).toEqual([
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00
    ]);
  });
});

describe('WasmModuleBuilder', () => {
  it('produces a valid module that returns a constant', () => {
    const b = new WasmModuleBuilder();
    const typeIdx = b.addFuncType([], [ValType.I64]);
    b.beginFunction(typeIdx);
    b.i64_const(42n);
    b.endFunction();
    b.exportFunction('run');

    const bytes = b.finalize();
    expect(WebAssembly.validate(bytes)).toBe(true);

    const mod = new WebAssembly.Module(bytes);
    const inst = new WebAssembly.Instance(mod);
    const run = inst.exports.run as () => bigint;
    expect(run()).toBe(42n);
  });

  it('produces a module with memory import that reads/writes', () => {
    const mem = new WebAssembly.Memory({ initial: 1 });
    const view = new DataView(mem.buffer);
    view.setBigUint64(0, 123456789n, true);

    const b = new WasmModuleBuilder();
    b.addMemoryImport('env', 'memory', 1);
    const typeIdx = b.addFuncType([], [ValType.I64]);
    b.beginFunction(typeIdx);
    // Read i64 from offset 0
    b.i32_const(0);
    b.i64_load(3, 0);
    b.endFunction();
    b.exportFunction('run');

    const bytes = b.finalize();
    expect(WebAssembly.validate(bytes)).toBe(true);

    const mod = new WebAssembly.Module(bytes);
    const inst = new WebAssembly.Instance(mod, { env: { memory: mem } });
    const run = inst.exports.run as () => bigint;
    expect(run()).toBe(123456789n);
  });

  it('supports locals and arithmetic', () => {
    const b = new WasmModuleBuilder();
    const typeIdx = b.addFuncType([], [ValType.I64]);
    b.beginFunction(typeIdx);
    const local0 = b.addLocals(ValType.I64, 2); // local0, local1
    // local0 = 10
    b.i64_const(10n);
    b.local_set(local0);
    // local1 = 20
    b.i64_const(20n);
    b.local_set(local0 + 1);
    // return local0 + local1
    b.local_get(local0);
    b.local_get(local0 + 1);
    b.i64_add();
    b.endFunction();
    b.exportFunction('run');

    const bytes = b.finalize();
    expect(WebAssembly.validate(bytes)).toBe(true);

    const mod = new WebAssembly.Module(bytes);
    const inst = new WebAssembly.Instance(mod);
    const run = inst.exports.run as () => bigint;
    expect(run()).toBe(30n);
  });

  it('supports if/else control flow', () => {
    const b = new WasmModuleBuilder();
    const typeIdx = b.addFuncType([], [ValType.I64]);
    b.beginFunction(typeIdx);
    // if (1) { return 100 } else { return 200 }
    b.i32_const(1);
    b.if_(ValType.I64);
    b.i64_const(100n);
    b.else_();
    b.i64_const(200n);
    b.end();
    b.endFunction();
    b.exportFunction('run');

    const bytes = b.finalize();
    expect(WebAssembly.validate(bytes)).toBe(true);

    const mod = new WebAssembly.Module(bytes);
    const inst = new WebAssembly.Instance(mod);
    const run = inst.exports.run as () => bigint;
    expect(run()).toBe(100n);
  });

  it('supports loop construct', () => {
    const b = new WasmModuleBuilder();
    const typeIdx = b.addFuncType([], [ValType.I64]);
    b.beginFunction(typeIdx);
    const locals = b.addLocals(ValType.I64, 2); // counter, sum
    const counter = locals;
    const sum = locals + 1;

    // counter = 10
    b.i64_const(10n);
    b.local_set(counter);
    // sum = 0
    b.i64_const(0n);
    b.local_set(sum);

    // block $exit
    b.block(BLOCKTYPE_VOID);
    // loop $loop
    b.loop(BLOCKTYPE_VOID);

    // sum += counter
    b.local_get(sum);
    b.local_get(counter);
    b.i64_add();
    b.local_set(sum);

    // counter--
    b.local_get(counter);
    b.i64_const(1n);
    b.i64_sub();
    b.local_set(counter);

    // if counter > 0, branch back to loop
    b.local_get(counter);
    b.i64_const(0n);
    b.i64_gt_s();
    b.br_if(0); // br_if $loop (depth 0 = innermost = loop)

    b.end(); // end loop
    b.end(); // end block

    // return sum
    b.local_get(sum);
    b.endFunction();
    b.exportFunction('run');

    const bytes = b.finalize();
    expect(WebAssembly.validate(bytes)).toBe(true);

    const mod = new WebAssembly.Module(bytes);
    const inst = new WebAssembly.Instance(mod);
    const run = inst.exports.run as () => bigint;
    expect(run()).toBe(55n); // sum 1..10
  });

  it('handles i64 store and load round-trip', () => {
    const mem = new WebAssembly.Memory({ initial: 1 });

    const b = new WasmModuleBuilder();
    b.addMemoryImport('env', 'memory', 1);
    const typeIdx = b.addFuncType([], [ValType.I64]);
    b.beginFunction(typeIdx);
    // Store 0xDEADBEEFCAFEBABE at offset 0
    b.i32_const(0);
    b.i64_const(0xDEADBEEFCAFEBABEn);
    b.i64_store(3, 0);
    // Load it back
    b.i32_const(0);
    b.i64_load(3, 0);
    b.endFunction();
    b.exportFunction('run');

    const bytes = b.finalize();
    expect(WebAssembly.validate(bytes)).toBe(true);

    const mod = new WebAssembly.Module(bytes);
    const inst = new WebAssembly.Instance(mod, { env: { memory: mem } });
    const run = inst.exports.run as () => bigint;

    // The constant is negative when interpreted as signed i64
    const expected = BigInt.asIntN(64, 0xDEADBEEFCAFEBABEn);
    expect(run()).toBe(expected);
  });

  it('handles negative i64 constants', () => {
    const b = new WasmModuleBuilder();
    const typeIdx = b.addFuncType([], [ValType.I64]);
    b.beginFunction(typeIdx);
    b.i64_const(-1n);
    b.endFunction();
    b.exportFunction('run');

    const bytes = b.finalize();
    expect(WebAssembly.validate(bytes)).toBe(true);

    const mod = new WebAssembly.Module(bytes);
    const inst = new WebAssembly.Instance(mod);
    const run = inst.exports.run as () => bigint;
    expect(run()).toBe(-1n);
  });

  it('supports block/loop self-loop pattern (the critical pattern for jitter)', () => {
    // This tests the exact WASM pattern we'll use for self-looping ARM64 blocks:
    // block (result i64) / loop (void) / br_if back to loop / br out with result
    const b = new WasmModuleBuilder();
    const typeIdx = b.addFuncType([], [ValType.I64]);
    b.beginFunction(typeIdx);
    const locals = b.addLocals(ValType.I64, 2);
    const counter = locals;
    const sum = locals + 1;

    // counter = 5, sum = 0
    b.i64_const(5n);
    b.local_set(counter);
    b.i64_const(0n);
    b.local_set(sum);

    // block $exit (result i64)
    b.block(ValType.I64);
    // loop $loop (void)
    b.loop(BLOCKTYPE_VOID);

    // sum += counter
    b.local_get(sum);
    b.local_get(counter);
    b.i64_add();
    b.local_set(sum);

    // counter--
    b.local_get(counter);
    b.i64_const(1n);
    b.i64_sub();
    b.local_set(counter);

    // if counter > 0, loop back
    b.local_get(counter);
    b.i64_const(0n);
    b.i64_gt_s();
    b.br_if(0); // back to $loop

    // fall through: push sum as block result and exit
    b.local_get(sum);
    b.br(1); // break to $exit with sum on stack

    b.end(); // end loop
    b.unreachable(); // satisfies block result type for fallthrough path
    b.end(); // end block — i64 result on stack (this IS the return value)

    b.endFunction();
    b.exportFunction('run');

    const bytes = b.finalize();
    expect(WebAssembly.validate(bytes)).toBe(true);

    const mod = new WebAssembly.Module(bytes);
    const inst = new WebAssembly.Instance(mod);
    const run = inst.exports.run as () => bigint;
    expect(run()).toBe(15n); // 5+4+3+2+1 = 15
  });
});
