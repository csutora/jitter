// minimal multi-block region: B taken to a later in-region block.
//
// pc=0:   MOVZ X0, #1
// pc=4:   B    .+8     ; → pc=12 (skip pc=8)
// pc=8:   MOVZ X0, #99 ; (skipped)
// pc=12:  MOVZ X1, #42
// pc=16:  RET
//
// expected after run: X0=1 (not 99), X1=42

const wasmPath = '/Users/numi/workspace/jitter/engine/zig-out/bin/translator.wasm';
const wasm = typeof readFile === 'function' ? readFile(wasmPath, 'binary') : require('fs').readFileSync(wasmPath);
const out = typeof print === 'function' ? print : console.log;

WebAssembly.instantiate(wasm).then(({ instance }) => {
  const ex = instance.exports;
  const buf = new Uint8Array(ex.memory.buffer);

  const insns = [
    0xd2800020, // MOVZ X0, #1
    0x14000002, // B .+8
    0xd2800c60, // MOVZ X0, #99
    0xd2800541, // MOVZ X1, #42
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
  const rc = ex.translator_translate_region(0, 0, 0, 0, insns.length * 4, 0x20000, 0);
  if (rc !== 0) throw new Error('translate rc=' + rc);

  const outLen = ex.translator_result_len();
  const outOff = ex.translator_result_offset();
  out('module size: ' + outLen);

  const moduleBytes = new Uint8Array(outLen);
  for (let i = 0; i < outLen; i++) moduleBytes[i] = buf[outOff + i];

  const guestMem = new WebAssembly.Memory({ initial: 3, maximum: 65536, shared: true });
  return WebAssembly.instantiate(moduleBytes, { env: { memory: guestMem } }).then(({ instance: inst2 }) => {
    inst2.exports.run();
    const gb = new BigUint64Array(guestMem.buffer);
    out('X0=' + gb[0] + ' (want 1, NOT 99)');
    out('X1=' + gb[1] + ' (want 42)');
    if (gb[0] !== 1n) throw new Error('X0 wrong — B target jump broken');
    if (gb[1] !== 42n) throw new Error('X1 wrong');
    out('PASS — multi-block B works');
  });
}).catch(e => {
  out('FAIL: ' + e + '\n' + (e.stack || ''));
  if (typeof process !== 'undefined') process.exit(1);
});
