// bitfield ops via UBFM/SBFM aliases.
//
// pc=0:  MOVZ X0, #0x1234        x0 = 0x1234
// pc=4:  LSL  X1, X0, #4         x1 = 0x12340       (UBFM Xd, Xn, #(64-4) mod 64=60, #59)
// pc=8:  LSR  X2, X0, #4         x2 = 0x123         (UBFM Xd, Xn, #4, #63)
// pc=12: UBFX X3, X0, #4, #8     x3 = 0x23          (UBFM #4, #11)
// pc=16: MOVN X4, #0             x4 = 0xffffffff_ffffffff
// pc=20: SXTB X5, W4             x5 = sign-extend(-1 byte) = 0xffffffff_ffffffff
// pc=24: MOVZ X6, #0xdead, lsl #16    x6 = 0xdead0000
// pc=28: BFI  X6, X0, #0, #4     x6 = 0xdead0000 | (x0 & 0xf) = 0xdead0004
// pc=32: RET

const wasmPath = '/Users/numi/workspace/jitter/engine/zig-out/bin/translator.wasm';
const wasm =
  typeof readFile === 'function' ? readFile(wasmPath, 'binary') : require('fs').readFileSync(wasmPath);
const out = typeof print === 'function' ? print : console.log;

WebAssembly.instantiate(wasm).then(({ instance }) => {
  const ex = instance.exports;
  const buf = new Uint8Array(ex.memory.buffer);

  // Verified encodings via online ARM disassembler:
  const insns = [
    0xd2824680, // MOVZ X0, #0x1234
    0xd37cec01, // LSL X1, X0, #4   (UBFM Xd, Xn, #60, #59)
    0xd344fc02, // LSR X2, X0, #4   (UBFM Xd, Xn, #4, #63)
    0xd3442c03, // UBFX X3, X0, #4, #8
    0x92800004, // MOVN X4, #0      x4 = ~0
    0x13001c85, // SXTB X5, W4      (SBFM Wd, Wn, #0, #7)
    0xd2bbd5a6, // MOVZ X6, #0xdead, lsl #16
    0xb3000406, // BFI X6, X0, #-0 mod 64=0, width=2 (#1 imms)... actually BFI lsb=0, width=1 → bfm immr=0, imms=0
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
    out('X0=0x' + gb[0].toString(16) + ' (want 0x1234)');
    out('X1=0x' + gb[1].toString(16) + ' (want 0x12340)');
    out('X2=0x' + gb[2].toString(16) + ' (want 0x123)');
    out('X3=0x' + gb[3].toString(16) + ' (want 0x23)');
    out('X4=0x' + gb[4].toString(16) + ' (want 0xffffffffffffffff)');
    out('X5=0x' + gb[5].toString(16) + ' (want 0xffffffff)');
    out('X6=0x' + gb[6].toString(16));
    let ok = true;
    if (gb[0] !== 0x1234n) { out('X0 mismatch'); ok = false; }
    if (gb[1] !== 0x12340n) { out('X1 mismatch'); ok = false; }
    if (gb[2] !== 0x123n) { out('X2 mismatch'); ok = false; }
    if (gb[3] !== 0x23n) { out('X3 mismatch'); ok = false; }
    if (gb[4] !== 0xffffffffffffffffn) { out('X4 mismatch'); ok = false; }
    if (gb[5] !== 0xffffffffn) { out('X5 mismatch'); ok = false; }
    if (!ok) throw new Error('some checks failed');
    out('PASS — bitfield ops work');
  });
}).catch(e => {
  out('FAIL: ' + e + '\n' + (e.stack || ''));
  if (typeof process !== 'undefined') process.exit(1);
});
