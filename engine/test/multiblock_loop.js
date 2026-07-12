// multi-block region with an inner self-loop. tests the structure:
//   block 0: counter setup, fallthrough to block 1
//   block 1: SUB X0, X0, #1; CBNZ X0, .-4  (self-loop)
//   block 2: MOVZ X1, #99; RET (after loop)
//
// pc=0:  MOVZ X0, #5
// pc=4:  SUB  X0, X0, #1   <- block 1 leader (cbnz target)
// pc=8:  CBNZ X0, .-4
// pc=12: MOVZ X1, #99       <- block 2 leader (cbnz fallthrough)
// pc=16: RET
//
// expected: X0=0, X1=99

const wasmPath = '/Users/numi/workspace/jitter/engine/zig-out/bin/translator.wasm';
const wasm = typeof readFile === 'function' ? readFile(wasmPath, 'binary') : require('fs').readFileSync(wasmPath);
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
    out('X0=' + gb[0] + ' (want 0)');
    out('X1=' + gb[1] + ' (want 99)');
    if (gb[0] !== 0n) throw new Error('X0 wrong — counter loop broken');
    if (gb[1] !== 99n) throw new Error('X1 wrong — post-loop block not reached');
    out('PASS — multi-block with self-loop works');
  });
}).catch(e => {
  out('FAIL: ' + e + '\n' + (e.stack || ''));
  if (typeof process !== 'undefined') process.exit(1);
});
