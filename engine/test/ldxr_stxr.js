// ldxr/stxr exclusive monitor — single-threaded happy path
//
// pc=0:  MOVZ X0, #0x100               base = 0x100
// pc=4:  MOVZ X1, #42                  value to store via cas loop
// pc=8:  STR  X1, [X0]                 mem[0x100] = 42 (seed value)
//
// CAS attempt: old=42, new=99
// pc=12: LDXR X2, [X0]                 x2 = mem[0x100] = 42; monitor armed
// pc=16: MOVZ X3, #99                  new value
// pc=20: STXR W4, X3, [X0]             cas(42 -> 99); w4 = 0 (success), mem[0x100]=99
//
// Without intervening writes, STXR should succeed: W4 == 0.
// Also verify mem[0x100] is 99 afterwards.
//
// pc=24: LDR X5, [X0]                  x5 = mem[0x100] = 99
// pc=28: RET

const SHADOW_BASE = 0x20000;
const MIN_PAGES = 3;
const MAX_PAGES = 65536;

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
    0xd2800c63, // MOVZ X3, #99
    0xc8047c03, // STXR W4, X3, [X0]
    0xf9400005, // LDR X5, [X0]
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
  const rc = ex.translator_translate_region(0, 0, 0, 0, insns.length * 4, SHADOW_BASE, 1);
  if (rc !== 0) throw new Error('translate rc=' + rc);

  const outLen = ex.translator_result_len();
  const outOff = ex.translator_result_offset();
  out('module size: ' + outLen);

  const moduleBytes = new Uint8Array(outLen);
  for (let i = 0; i < outLen; i++) moduleBytes[i] = buf[outOff + i];

  const guestMem = new WebAssembly.Memory({ initial: MIN_PAGES, maximum: MAX_PAGES, shared: true });
  return WebAssembly.instantiate(moduleBytes, { env: { memory: guestMem } }).then(({ instance: inst2 }) => {
    inst2.exports.run();
    const gb = new BigUint64Array(guestMem.buffer);
    out('X1=' + gb[1] + ' (want 42)');
    out('X2=' + gb[2] + ' (want 42, loaded via LDXR)');
    out('X3=' + gb[3] + ' (want 99)');
    out('X4=' + gb[4] + ' (want 0 — STXR success)');
    out('X5=' + gb[5] + ' (want 99 — value after CAS)');

    const memCheck = new BigUint64Array(guestMem.buffer, 0x10000 + 0x100, 1);
    out('mem[0x100]=' + memCheck[0] + ' (want 99)');

    let ok = true;
    if (gb[2] !== 42n) { out('X2 mismatch'); ok = false; }
    if (gb[4] !== 0n) { out('X4 mismatch — STXR should have succeeded'); ok = false; }
    if (gb[5] !== 99n) { out('X5 mismatch'); ok = false; }
    if (memCheck[0] !== 99n) { out('memory mismatch'); ok = false; }
    if (!ok) throw new Error('check failed');
    out('PASS — ldxr/stxr CAS loop works');
  });
}).catch(e => {
  out('FAIL: ' + e + '\n' + (e.stack || ''));
  if (typeof process !== 'undefined') process.exit(1);
});
