// stp + ldp through pre-/post-index, plus adrp+adr.
//
// PC=0:  MOVZ X0, #0x1000      x0 = 0x1000
// PC=4:  ADD  SP, X0, #0x100   sp = 0x1100 (use as stack base)
// PC=8:  MOVZ X1, #0x1234      x1 = 0x1234
// PC=12: MOVZ X2, #0x5678      x2 = 0x5678
// PC=16: STP X1, X2, [SP, #-16]!     sp = 0x10f0; mem[0x10f0]=x1, mem[0x10f8]=x2
// PC=20: MOVZ X1, #0           clobber
// PC=24: MOVZ X2, #0           clobber
// PC=28: LDP X1, X2, [SP], #16        x1=mem[0x10f0], x2=mem[0x10f8]; sp += 16 -> 0x1100
// PC=32: ADR X3, #16           x3 = 32 + 16 = 48 = 0x30
// PC=36: ADRP X4, #0x1000      x4 = (36 & ~0xfff) + 0x1000 = 0x1000
// PC=40: RET

const wasmPath = '/Users/numi/workspace/jitter/engine/zig-out/bin/translator.wasm';
const wasm =
  typeof readFile === 'function' ? readFile(wasmPath, 'binary') : require('fs').readFileSync(wasmPath);
const out = typeof print === 'function' ? print : console.log;

WebAssembly.instantiate(wasm).then(({ instance }) => {
  const ex = instance.exports;
  const buf = new Uint8Array(ex.memory.buffer);

  const insns = [
    0xd2820000, // MOVZ X0, #0x1000
    0x9104001f, // ADD SP, X0, #0x100   (sf=1, op=0, S=0, sh=0, imm=0x100, rn=0, rd=31=SP)
    0xd2824681, // MOVZ X1, #0x1234
    0xd28acf02, // MOVZ X2, #0x5678
    0xa9bf0be1, // STP X1, X2, [SP, #-16]!   (signed imm = -2 << 3 = -16, mode=pre-index)
    0xd2800001, // MOVZ X1, #0
    0xd2800002, // MOVZ X2, #0
    0xa8c10be1, // LDP X1, X2, [SP], #16     (signed imm = +2 << 3 = 16, mode=post-index)
    0x10000083, // ADR X3, #16
    0xb0000004, // ADRP X4, #0x1000  (immlo=01, immhi=0 → offset_pages=1)
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
    // x0=0, x1=8, x2=16, x3=24, x4=32, sp=0xf8/8 = index 31
    out('X0=0x' + gb[0].toString(16) + ' (want 0x1000)');
    out('X1=0x' + gb[1].toString(16) + ' (want 0x1234)');
    out('X2=0x' + gb[2].toString(16) + ' (want 0x5678)');
    out('X3=0x' + gb[3].toString(16) + ' (want 0x30)');
    out('X4=0x' + gb[4].toString(16) + ' (want 0x1000)');
    out('SP=0x' + gb[31].toString(16) + ' (want 0x1100)');

    // also peek at the bytes that should have been stored
    const stackBytes = new BigUint64Array(guestMem.buffer, 0x80000 + 0x10f0, 2);
    out('mem[0x10f0]=0x' + stackBytes[0].toString(16) + ' (want 0x1234)');
    out('mem[0x10f8]=0x' + stackBytes[1].toString(16) + ' (want 0x5678)');

    if (gb[1] !== 0x1234n) throw new Error('X1 mismatch');
    if (gb[2] !== 0x5678n) throw new Error('X2 mismatch');
    if (gb[3] !== 0x30n) throw new Error('X3 mismatch');
    if (gb[4] !== 0x1000n) throw new Error('X4 mismatch');
    if (gb[31] !== 0x1100n) throw new Error('SP mismatch');
    if (stackBytes[0] !== 0x1234n) throw new Error('mem[0x10f0] mismatch');
    if (stackBytes[1] !== 0x5678n) throw new Error('mem[0x10f8] mismatch');
    out('PASS — adr/adrp + stp pre-index + ldp post-index work');
  });
}).catch(e => {
  out('FAIL: ' + e + '\n' + (e.stack || ''));
  if (typeof process !== 'undefined') process.exit(1);
});
