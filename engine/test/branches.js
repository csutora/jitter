// test inter-block branches.
//
// program: B over a MOVZ to a different MOVZ + RET.
//   B .+8           -> 14 00 00 02   block 0
//   MOVZ X0, #100   -> d2 80 0c 80   (dead — should not execute)
//   MOVZ X0, #42    -> d2 80 05 40   block 1 entry
//   RET             -> d6 5f 03 c0

const wasmPath = '/Users/numi/workspace/jitter/engine/zig-out/bin/translator.wasm';
const wasm =
  typeof readFile === 'function' ? readFile(wasmPath, 'binary') : require('fs').readFileSync(wasmPath);
const out = typeof print === 'function' ? print : console.log;

WebAssembly.instantiate(wasm).then(({ instance }) => {
  const ex = instance.exports;
  const buf = new Uint8Array(ex.memory.buffer);

  const insns = [0x14000002, 0xd2800c80, 0xd2800540, 0xd65f03c0];
  const off = ex.translator_input_buf_offset();
  for (let i = 0; i < insns.length; i++) {
    const v = insns[i];
    buf[off + i * 4 + 0] = v & 0xff;
    buf[off + i * 4 + 1] = (v >>> 8) & 0xff;
    buf[off + i * 4 + 2] = (v >>> 16) & 0xff;
    buf[off + i * 4 + 3] = (v >>> 24) & 0xff;
  }

  ex.translator_reset();
  const rc = ex.translator_translate_region(0x400000, 0, 0x400000, 0, 16, 0, 0);
  if (rc !== 0) throw new Error('translate rc=' + rc);

  const outOff = ex.translator_result_offset();
  const outLen = ex.translator_result_len();
  out('module size: ' + outLen + ' bytes');

  const moduleBytes = new Uint8Array(outLen);
  for (let i = 0; i < outLen; i++) moduleBytes[i] = buf[outOff + i];

  const guestMem = new WebAssembly.Memory({ initial: 40, maximum: 40 });
  return WebAssembly.instantiate(moduleBytes, { env: { memory: guestMem, reg_base: 0, simd_op: () => { throw new Error('simd_op'); }, mem_fault: () => { throw new Error('mem_fault: guest access out of bounds'); } } }).then(({ instance: inst2 }) => {
    const result = inst2.exports.run();
    const guestBuf = new BigUint64Array(guestMem.buffer);
    out('X0 after run: ' + guestBuf[0] + ' (want 42)');
    if (guestBuf[0] !== 42n) throw new Error('X0 should be 42, got ' + guestBuf[0]);
    out('exit reason: ' + (result >> 32n));
    out('PASS — inter-block branch works');
  });
}).catch(e => {
  out('FAIL: ' + e + '\n' + (e.stack || ''));
  if (typeof process !== 'undefined') process.exit(1);
});
