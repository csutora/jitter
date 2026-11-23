// stxr must fail when a regular store invalidates the monitor between ldxr and stxr
//
// pc=0:  MOVZ X0, #0x100               base = 0x100
// pc=4:  MOVZ X1, #42                  seed value
// pc=8:  STR  X1, [X0]                 mem[0x100] = 42
// pc=12: LDXR X2, [X0]                 x2 = 42; monitor armed
// pc=16: MOVZ X3, #77                  interfering value
// pc=20: STR  X3, [X0]                 mem[0x100] = 77 (should clear shadow bit)
// pc=24: MOVZ X4, #99                  attempted new value
// pc=28: STXR W5, X4, [X0]             must fail: w5 = 1, mem unchanged from 77
// pc=32: LDR X6, [X0]                  x6 = 77
// pc=36: RET

const wasmPath = '/Users/numi/workspace/jitter/engine/zig-out/bin/translator.wasm';
const wasm =
  typeof readFile === 'function' ? readFile(wasmPath, 'binary') : require('fs').readFileSync(wasmPath);
const out = typeof print === 'function' ? print : console.log;

WebAssembly.instantiate(wasm).then(({ instance }) => {
  const ex = instance.exports;
  const buf = new Uint8Array(ex.memory.buffer);

  const insns = [
    0xd2802000, // MOVZ X0, #0x100
    0xd2800541, // MOVZ X1, #42
    0xf9000001, // STR X1, [X0]
    0xc85f7c02, // LDXR X2, [X0]
    0xd2800963, // MOVZ X3, #75 — actually 77 = 0b1001101 = 0x4d
    0xf9000003, // STR X3, [X0]
    0xd2800c64, // MOVZ X4, #99
    0xc8057c04, // STXR W5, X4, [X0]
    0xf9400006, // LDR X6, [X0]
    0xd65f03c0, // RET
  ];

  // fixup MOVZ X3, #77
  // 77 = 0x4d. MOVZ encoding: 0xd2800000 | (imm16 << 5) | Rd
  insns[4] = 0xd2800000 | (77 << 5) | 3;

  const off = ex.translator_input_buf_offset();
  for (let i = 0; i < insns.length; i++) {
    const v = insns[i];
    buf[off + i * 4 + 0] = v & 0xff;
    buf[off + i * 4 + 1] = (v >>> 8) & 0xff;
    buf[off + i * 4 + 2] = (v >>> 16) & 0xff;
    buf[off + i * 4 + 3] = (v >>> 24) & 0xff;
  }

  ex.translator_reset();
  const rc = ex.translator_translate_region(0, 0, 0, 0, insns.length * 4, 0x20000, 1);
  if (rc !== 0) throw new Error('translate rc=' + rc);

  const outLen = ex.translator_result_len();
  const outOff = ex.translator_result_offset();
  out('module size: ' + outLen);

  const moduleBytes = new Uint8Array(outLen);
  for (let i = 0; i < outLen; i++) moduleBytes[i] = buf[outOff + i];

  const guestMem = new WebAssembly.Memory({ initial: 40, maximum: 40, shared: true });
  return WebAssembly.instantiate(moduleBytes, { env: { memory: guestMem, reg_base: 0, simd_op: () => { throw new Error('simd_op'); }, mem_fault: () => { throw new Error('mem_fault: guest access out of bounds'); } } }).then(({ instance: inst2 }) => {
    inst2.exports.run();
    const gb = new BigUint64Array(guestMem.buffer);
    out('X2=' + gb[2] + ' (want 42 — LDXR read)');
    out('X3=' + gb[3] + ' (want 77 — interfering value)');
    out('X4=' + gb[4] + ' (want 99 — attempted but should not land)');
    out('X5=' + gb[5] + ' (want 1 — STXR failure)');
    out('X6=' + gb[6] + ' (want 77 — memory unchanged by failed STXR)');

    let ok = true;
    if (gb[2] !== 42n) { out('X2 mismatch'); ok = false; }
    if (gb[3] !== 77n) { out('X3 mismatch'); ok = false; }
    if (gb[5] !== 1n) { out('X5 mismatch — STXR should have failed'); ok = false; }
    if (gb[6] !== 77n) { out('X6 mismatch — memory should still be 77'); ok = false; }
    if (!ok) throw new Error('check failed');
    out('PASS — stxr correctly fails after interfering store');
  });
}).catch(e => {
  out('FAIL: ' + e + '\n' + (e.stack || ''));
  if (typeof process !== 'undefined') process.exit(1);
});
