// loop test: MOVZ X0, #5; loop { SUB X0,X0,#1; CBNZ X0, loop }; MOVZ X1, #99; RET.
// expected: X0=0, X1=99 after run.
//
//   MOVZ X0, #5      -> d2 80 00 a0
//   SUB  X0, X0, #1  -> d1 00 04 00   <- loop start
//   CBNZ X0, .-4     -> b5 ff ff e0   imm19 = -1 (signed)
//   MOVZ X1, #99     -> d2 80 0c 61
//   RET              -> d6 5f 03 c0

const wasmPath = '/Users/numi/workspace/jitter/engine/zig-out/bin/translator.wasm';
const wasm =
  typeof readFile === 'function' ? readFile(wasmPath, 'binary') : require('fs').readFileSync(wasmPath);
const out = typeof print === 'function' ? print : console.log;

WebAssembly.instantiate(wasm).then(({ instance }) => {
  const ex = instance.exports;
  const buf = new Uint8Array(ex.memory.buffer);

  const insns = [
    0xd28000a0, // MOVZ X0, #5
    0xd1000400, // SUB X0, X0, #1
    0xb5ffffe0, // CBNZ X0, .-4
    0xd2800c61, // MOVZ X1, #99
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
  const rc = ex.translator_translate_region(0x400000, 0, 0x400000, 0, 20, 0, 0);
  if (rc !== 0) throw new Error('translate rc=' + rc);

  const outOff = ex.translator_result_offset();
  const outLen = ex.translator_result_len();
  out('module size: ' + outLen + ' bytes');

  const moduleBytes = new Uint8Array(outLen);
  for (let i = 0; i < outLen; i++) moduleBytes[i] = buf[outOff + i];

  const guestMem = new WebAssembly.Memory({ initial: Math.max(1, 3), maximum: 65536, shared: true });
  return WebAssembly.instantiate(moduleBytes, { env: { memory: guestMem } }).then(({ instance: inst2 }) => {
    const result = inst2.exports.run();
    const gb = new BigUint64Array(guestMem.buffer);
    out('X0=' + gb[0] + ' (want 0)');
    out('X1=' + gb[1] + ' (want 99)');
    if (gb[0] !== 0n) throw new Error('X0 should be 0, got ' + gb[0]);
    if (gb[1] !== 99n) throw new Error('X1 should be 99, got ' + gb[1]);
    out('exit reason: ' + (result >> 32n));
    out('PASS — loop with backward branch works');
  });
}).catch(e => {
  out('FAIL: ' + e + '\n' + (e.stack || ''));
  if (typeof process !== 'undefined') process.exit(1);
});
