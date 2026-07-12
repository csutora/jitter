// end-to-end smoke: load translator.wasm, feed it a tiny arm64 program,
// instantiate the produced wasm module, run it, verify cpu state changes.
//
// program: MOVZ X0, #42; RET
//   movz x0, #42    -> d2 80 05 40
//   ret             -> d6 5f 03 c0

const wasmPath = '/Users/numi/workspace/jitter/engine/zig-out/bin/translator.wasm';

const wasm =
  typeof readFile === 'function'
    ? readFile(wasmPath, 'binary')
    : require('fs').readFileSync(wasmPath);

const out = typeof print === 'function' ? print : console.log;

WebAssembly.instantiate(wasm).then(({ instance }) => {
  const ex = instance.exports;
  const mem = ex.memory;
  const buf = new Uint8Array(mem.buffer);

  // arm64 program at "guest" pc 0x400000
  const insns = [0xd2800540, 0xd65f03c0];
  const inputOffset = ex.translator_input_buf_offset();
  out('input buf at offset: ' + inputOffset.toString(16));

  for (let i = 0; i < insns.length; i++) {
    const v = insns[i];
    buf[inputOffset + i * 4 + 0] = v & 0xff;
    buf[inputOffset + i * 4 + 1] = (v >>> 8) & 0xff;
    buf[inputOffset + i * 4 + 2] = (v >>> 16) & 0xff;
    buf[inputOffset + i * 4 + 3] = (v >>> 24) & 0xff;
  }

  ex.translator_reset();
  // base_pc = 0x400000 (lo, hi), entry_pc = same, code_len = 8
  const rc = ex.translator_translate_region(0x400000, 0, 0x400000, 0, 8, 0, 0);
  out('translate rc=' + rc);
  if (rc !== 0) throw new Error('translation failed rc=' + rc);

  const outOffset = ex.translator_result_offset();
  const outLen = ex.translator_result_len();
  out('output: offset=' + outOffset.toString(16) + ' len=' + outLen);

  // copy out bytes
  const moduleBytes = new Uint8Array(outLen);
  for (let i = 0; i < outLen; i++) moduleBytes[i] = buf[outOffset + i];

  // dump first 32 bytes as hex for inspection
  let hex = '';
  for (let i = 0; i < Math.min(outLen, 32); i++) {
    hex += moduleBytes[i].toString(16).padStart(2, '0') + ' ';
  }
  out('module head: ' + hex);

  // validate header
  const magic = moduleBytes[0] === 0x00 && moduleBytes[1] === 0x61 && moduleBytes[2] === 0x73 && moduleBytes[3] === 0x6d;
  out('magic ok: ' + magic);
  if (!magic) throw new Error('bad wasm magic');

  // instantiate the produced module with a shared memory
  const guestMem = new WebAssembly.Memory({ initial: Math.max(1, 3), maximum: 65536, shared: true });
  return WebAssembly.instantiate(moduleBytes, { env: { memory: guestMem } }).then(({ instance: inst2 }) => {
    out('guest module exports: ' + Object.keys(inst2.exports).sort().join(', '));
    const result = inst2.exports.run();
    // result is i64; jsc returns it as bigint
    out('run returned: ' + result.toString(16));
    // read X0 from guest memory (cpu state at offset 0)
    const guestBuf = new BigUint64Array(guestMem.buffer);
    out('X0 after run: ' + guestBuf[0]);
    out('PC after run: 0x' + guestBuf[32].toString(16)); // pc_offset = 0x100 = 256 = 32 * 8

    if (guestBuf[0] !== 42n) throw new Error('X0 should be 42, got ' + guestBuf[0]);
    out('PASS');
  });
}).catch(e => {
  out('FAIL: ' + e + '\n' + (e.stack || ''));
  if (typeof process !== 'undefined') process.exit(1);
});
