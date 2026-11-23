// test: CMP + B.cond using NZCV flags.
//
//   PC=0  MOVZ X0, #5     -> d2 80 00 a0
//   PC=4  CMP X0, #5      -> f1 00 14 1f   (= SUBS XZR, X0, #5; sets Z=1)
//   PC=8  B.EQ +12        -> 54 00 00 60   (taken → PC=20)
//   PC=12 MOVZ X1, #999   -> d2 80 7c e1   (skipped if branch taken)
//   PC=16 RET             -> d6 5f 03 c0
//   PC=20 MOVZ X1, #42    -> d2 80 05 41   (branch lands here)
//   PC=24 RET             -> d6 5f 03 c0

const wasmPath = '/Users/numi/workspace/jitter/engine/zig-out/bin/translator.wasm';
const wasm =
  typeof readFile === 'function' ? readFile(wasmPath, 'binary') : require('fs').readFileSync(wasmPath);
const out = typeof print === 'function' ? print : console.log;

WebAssembly.instantiate(wasm).then(({ instance }) => {
  const ex = instance.exports;
  const buf = new Uint8Array(ex.memory.buffer);

  const insns = [
    0xd28000a0, // MOVZ X0, #5
    0xf100141f, // CMP X0, #5
    0x54000060, // B.EQ +12
    0xd2807ce1, // MOVZ X1, #999
    0xd65f03c0, // RET
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
  const rc = ex.translator_translate_region(0x400000, 0, 0x400000, 0, insns.length * 4, 0x20000, 0);
  if (rc !== 0) throw new Error('translate rc=' + rc);

  const outOff = ex.translator_result_offset();
  const outLen = ex.translator_result_len();
  out('module size: ' + outLen + ' bytes');

  const moduleBytes = new Uint8Array(outLen);
  for (let i = 0; i < outLen; i++) moduleBytes[i] = buf[outOff + i];

  const guestMem = new WebAssembly.Memory({ initial: 40, maximum: 40 });
  return WebAssembly.instantiate(moduleBytes, { env: { memory: guestMem, reg_base: 0, simd_op: () => { throw new Error('simd_op'); }, mem_fault: () => { throw new Error('mem_fault: guest access out of bounds'); } } }).then(({ instance: inst2 }) => {
    const result = inst2.exports.run();
    const gb = new BigUint64Array(guestMem.buffer);
    const nzcv = new Uint32Array(guestMem.buffer)[0x108 / 4];
    out('X0=' + gb[0] + ' (want 5)');
    out('X1=' + gb[1] + ' (want 42)');
    out('NZCV=0x' + nzcv.toString(16));
    if (gb[0] !== 5n) throw new Error('X0 should be 5');
    if (gb[1] !== 42n) throw new Error('X1 should be 42, got ' + gb[1]);
    out('exit reason: ' + (result >> 32n));
    out('PASS — CMP + B.EQ works');
  });
}).catch(e => {
  out('FAIL: ' + e + '\n' + (e.stack || ''));
  if (typeof process !== 'undefined') process.exit(1);
});
