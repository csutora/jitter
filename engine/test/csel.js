// test: CMP + CSEL = min(X0, X1)
//
//   MOVZ X0, #5          -> d2 80 00 a0
//   MOVZ X1, #10         -> d2 80 01 41
//   CMP X0, X1           -> eb 01 00 1f   (= SUBS XZR, X0, X1)
//   CSEL X2, X0, X1, LT  -> 9a 81 b0 02   (X2 = (X0<X1) ? X0 : X1)
//   RET                  -> d6 5f 03 c0

const wasmPath = '/Users/numi/workspace/jitter/engine/zig-out/bin/translator.wasm';
const wasm =
  typeof readFile === 'function' ? readFile(wasmPath, 'binary') : require('fs').readFileSync(wasmPath);
const out = typeof print === 'function' ? print : console.log;

WebAssembly.instantiate(wasm).then(({ instance }) => {
  const ex = instance.exports;
  const buf = new Uint8Array(ex.memory.buffer);

  const insns = [
    0xd28000a0, // MOVZ X0, #5
    0xd2800141, // MOVZ X1, #10
    0xeb01001f, // CMP X0, X1
    0x9a81b002, // CSEL X2, X0, X1, LT
    0xd65f03c0, // RET
  ];
  const off = ex.translator_input_buf_offset();
  for (let i = 0; i < insns.length; i++) {
    const v = insns[i];
    buf[off + i * 4 + 0] = v & 0xff;
    buf[off + i * 4 + 1] = (v >>> 8) & 0xff;
    buf[off + i * 4 + 2] = (v >>> 16) & 0xff;
    buf[off + i * 4 + 3] = (v >>> 24) & 0xff;
  }

  ex.translator_reset();
  const rc = ex.translator_translate_region(0x400000, 0, 0x400000, 0, insns.length * 4, 0x20000, 0);
  if (rc !== 0) throw new Error('translate rc=' + rc);

  const outLen = ex.translator_result_len();
  const outOff = ex.translator_result_offset();
  out('module size: ' + outLen);

  const moduleBytes = new Uint8Array(outLen);
  for (let i = 0; i < outLen; i++) moduleBytes[i] = buf[outOff + i];

  const guestMem = new WebAssembly.Memory({ initial: 40, maximum: 40 });
  return WebAssembly.instantiate(moduleBytes, { env: { memory: guestMem, reg_base: 0, simd_op: () => { throw new Error('simd_op'); }, mem_fault: () => { throw new Error('mem_fault: guest access out of bounds'); } } }).then(({ instance: inst2 }) => {
    inst2.exports.run();
    const gb = new BigUint64Array(guestMem.buffer);
    out('X0=' + gb[0] + ' X1=' + gb[1] + ' X2=' + gb[2] + ' (want X2=5)');
    if (gb[2] !== 5n) throw new Error('X2 should be 5 (min), got ' + gb[2]);
    out('PASS — CMP+CSEL works');
  });
}).catch(e => {
  out('FAIL: ' + e + '\n' + (e.stack || ''));
  if (typeof process !== 'undefined') process.exit(1);
});
