// fibonacci. classic JIT benchmark.
//
//   PC=0:  MOVZ X0, #10        n = 10
//   PC=4:  MOVZ X1, #0         prev = 0
//   PC=8:  MOVZ X2, #1         curr = 1
//   PC=12: ADD X3, X1, X2      tmp = prev + curr           <-- loop
//   PC=16: ORR X1, XZR, X2     prev = curr                 (MOV alias)
//   PC=20: ORR X2, XZR, X3     curr = tmp
//   PC=24: SUBS X0, X0, #1     n--; set flags
//   PC=28: B.NE -16            branch back to loop if n != 0
//   PC=32: ORR X0, XZR, X1     result = prev (which holds fib(10))
//   PC=36: RET

const wasmPath = '/Users/numi/workspace/jitter/engine/zig-out/bin/translator.wasm';
const wasm =
  typeof readFile === 'function' ? readFile(wasmPath, 'binary') : require('fs').readFileSync(wasmPath);
const out = typeof print === 'function' ? print : console.log;

WebAssembly.instantiate(wasm).then(({ instance }) => {
  const ex = instance.exports;
  const buf = new Uint8Array(ex.memory.buffer);

  const insns = [
    0xd2800140, // MOVZ X0, #10
    0xd2800001, // MOVZ X1, #0
    0xd2800022, // MOVZ X2, #1
    0x8b020023, // ADD X3, X1, X2
    0xaa0203e1, // ORR X1, XZR, X2
    0xaa0303e2, // ORR X2, XZR, X3
    0xf1000400, // SUBS X0, X0, #1
    0x54ffff81, // B.NE -16
    0xaa0103e0, // ORR X0, XZR, X1
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
  const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const rc = ex.translator_translate_region(0x400000, 0, 0x400000, 0, insns.length * 4, 0x20000, 0);
  const t1 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  if (rc !== 0) throw new Error('translate rc=' + rc);

  const outLen = ex.translator_result_len();
  const outOff = ex.translator_result_offset();
  out('translation: ' + (t1 - t0).toFixed(3) + 'ms, output module=' + outLen + ' bytes');

  const moduleBytes = new Uint8Array(outLen);
  for (let i = 0; i < outLen; i++) moduleBytes[i] = buf[outOff + i];

  const guestMem = new WebAssembly.Memory({ initial: Math.max(1, 3), maximum: 65536, shared: true });
  return WebAssembly.instantiate(moduleBytes, { env: { memory: guestMem } }).then(({ instance: inst2 }) => {
    const t2 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    inst2.exports.run();
    const t3 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const gb = new BigUint64Array(guestMem.buffer);
    out('execution: ' + (t3 - t2).toFixed(3) + 'ms');
    out('fib(10) = ' + gb[0] + ' (want 55)');
    if (gb[0] !== 55n) throw new Error('fib(10) should be 55, got ' + gb[0]);
    out('PASS — fibonacci through Zig translator');
  });
}).catch(e => {
  out('FAIL: ' + e + '\n' + (e.stack || ''));
  if (typeof process !== 'undefined') process.exit(1);
});
