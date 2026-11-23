// runs the engine wasm through whatever js engine invokes this script.
// usage:
//   jsc engine/test/smoke.js      (system jsc = same engine as ios safari)
//   node --experimental-vm-modules engine/test/smoke.js
//
// uses globalThis.readFile if available (jsc), else node fs.

const wasmPath = '/Users/numi/workspace/jitter/engine/zig-out/bin/translator.wasm';

const wasm =
  typeof readFile === 'function'
    ? readFile(wasmPath, 'binary')
    : require('fs').readFileSync(wasmPath);

const out = typeof print === 'function' ? print : console.log;

WebAssembly.instantiate(wasm).then(({ instance }) => {
  const ex = instance.exports;
  out('exports: ' + Object.keys(ex).sort().join(', '));

  const cases = [
    [0x91000441, 'ADD X1, X2, #1'],
    [0xf100107f, 'SUBS XZR, X3, #4 (CMP)'],
    [0xd2824680, 'MOVZ X0, #0x1234'],
    [0x14000002, 'B .+8'],
    [0x54000060, 'B.EQ .+12'],
    [0xd65f03c0, 'RET'],
    [0xd4000001, 'SVC #0'],
    [0xf9400be0, 'LDR X0, [SP, #16]'],
    [0xf81f0fe0, 'STR X0, [SP, #-16]!'],
    [0xa9bf7bfd, 'STP X29, X30, [SP, #-16]!'],
    [0x8b020020, 'ADD X0, X1, X2'],
    [0x92401c20, 'AND X0, X1, #0xff'],
    [0x9a820020, 'CSEL X0, X1, X2, EQ'],
    [0x34000020, 'CBZ W0, .+4'],
    [0x58000040, 'LDR X0, .+8 (literal)'],
  ];

  // kind 0 is `unknown`: an instruction the decoder does not recognise. printing the kinds is
  // useful, but the thing worth failing over is any of these landing there.
  let ok = true;
  for (const [insn, label] of cases) {
    const kind = ex.translator_decode_kind(insn);
    out('  0x' + insn.toString(16).padStart(8, '0') + '  kind=' + kind + '  ' + label);
    if (kind === 0) { out('    ^ decoded as unknown'); ok = false; }
  }
  if (!ok) { out('FAIL: some instructions decode as unknown'); return; }
  out('PASS — every sample instruction decodes');
});
