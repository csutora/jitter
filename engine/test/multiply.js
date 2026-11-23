// multiply + divide ops: madd/msub/mul/mneg, sdiv/udiv, smaddl/umaddl, smulh/umulh
//
// pc=0:  MOVZ X0, #7
// pc=4:  MOVZ X1, #6
// pc=8:  MUL  X2, X0, X1            x2 = 42 = 0x2a  (MADD with Ra=XZR)
// pc=12: MOVZ X3, #100
// pc=16: MADD X4, X0, X1, X3        x4 = 100 + 42 = 142 = 0x8e
// pc=20: MSUB X5, X0, X1, X3        x5 = 100 - 42 = 58 = 0x3a
// pc=24: MOVZ X6, #20
// pc=28: UDIV X7, X6, X0            x7 = 20 / 7 = 2
// pc=32: SDIV X8, X6, X0            x8 = 20 / 7 = 2 (positive)
// pc=36: MOVZ X9, #0xffff, lsl #48  x9 = 0xffff_0000_0000_0000 (very negative as i64)
// pc=40: SMULH X10, X9, X9          x10 = high 64 of (negative * negative) = positive small
// pc=44: UMULH X11, X9, X9          x11 = high 64 of huge unsigned * huge unsigned
// pc=48: RET

const wasmPath = '/Users/numi/workspace/jitter/engine/zig-out/bin/translator.wasm';
const wasm =
  typeof readFile === 'function' ? readFile(wasmPath, 'binary') : require('fs').readFileSync(wasmPath);
const out = typeof print === 'function' ? print : console.log;

WebAssembly.instantiate(wasm).then(({ instance }) => {
  const ex = instance.exports;
  const buf = new Uint8Array(ex.memory.buffer);

  const insns = [
    0xd28000e0, // MOVZ X0, #7
    0xd28000c1, // MOVZ X1, #6
    0x9b017c02, // MUL X2, X0, X1   (MADD X2, X0, X1, XZR)
    0xd2800c83, // MOVZ X3, #100
    0x9b010c04, // MADD X4, X0, X1, X3
    0x9b018c05, // MSUB X5, X0, X1, X3
    0xd2800286, // MOVZ X6, #20
    0x9ac008c7, // UDIV X7, X6, X0
    0x9ac00cc8, // SDIV X8, X6, X0
    0xd2ffffe9, // MOVZ X9, #0xffff, lsl #48
    0x9b497d2a, // SMULH X10, X9, X9
    0x9bc97d2b, // UMULH X11, X9, X9
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

  const guestMem = new WebAssembly.Memory({ initial: 40, maximum: 40 });
  return WebAssembly.instantiate(moduleBytes, { env: { memory: guestMem, reg_base: 0, simd_op: () => { throw new Error('simd_op'); }, mem_fault: () => { throw new Error('mem_fault: guest access out of bounds'); } } }).then(({ instance: inst2 }) => {
    inst2.exports.run();
    const gb = new BigUint64Array(guestMem.buffer);

    // expected SMULH X9,X9 where x9 = 0xffff000000000000:
    //   as signed: -0x0001000000000000
    //   squared:   0x00000001_00000000_00000000_00000000 (positive, 113 bits set high)
    //   high 64:   0x0000000100000000... hmm let me compute
    //   (-0x0001_0000_0000_0000)^2 = +0x0000_0001_0000_0000_0000_0000_0000_0000
    //   high 64 bits = 0x00000001_00000000

    // UMULH X9,X9: x9 = 0xffff000000000000 unsigned
    //   = 18446462598732840960 ≈ 2^64 - 2^48
    //   squared ≈ (2^64)^2 - 2*2^48*2^64 + 2^96 ≈ 2^128 - 2^113 + 2^96
    //   high 64 ≈ 2^64 - 2^49 + ... actually let's just compute:
    //   0xfffe0000_00000001_00000000_00000000 hmm
    //   exact: 0xfffe_0001_0000_0000_0000_0000_0000_0000
    //   high 64: 0xfffe_0001_0000_0000

    out('X2 = ' + gb[2] + ' (want 42)');
    out('X4 = ' + gb[4] + ' (want 142)');
    out('X5 = ' + gb[5] + ' (want 58)');
    out('X7 = ' + gb[7] + ' (want 2)');
    out('X8 = ' + gb[8] + ' (want 2)');
    out('X9 = 0x' + gb[9].toString(16));
    out('X10 = 0x' + gb[10].toString(16) + ' (want 0x100000000)');
    out('X11 = 0x' + gb[11].toString(16) + ' (want 0xfffe000100000000)');

    let ok = true;
    if (gb[2] !== 42n) { out('X2 mismatch'); ok = false; }
    if (gb[4] !== 142n) { out('X4 mismatch'); ok = false; }
    if (gb[5] !== 58n) { out('X5 mismatch'); ok = false; }
    if (gb[7] !== 2n) { out('X7 mismatch'); ok = false; }
    if (gb[8] !== 2n) { out('X8 mismatch'); ok = false; }
    if (gb[10] !== 0x100000000n) { out('X10 mismatch'); ok = false; }
    if (gb[11] !== 0xfffe000100000000n) { out('X11 mismatch'); ok = false; }
    if (!ok) throw new Error('mismatches');
    out('PASS — multiply + divide ops work');
  });
}).catch(e => {
  out('FAIL: ' + e + '\n' + (e.stack || ''));
  if (typeof process !== 'undefined') process.exit(1);
});
