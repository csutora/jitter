// SIMD load/store: LDR Q, STR Q, LDP Q, plus a smaller LDR D / STR D
//
// pc=0:  MOVZ X0, #0x200                  base = 0x200
// pc=4:  MOVZ X1, #0xabcd                 low qword payload
// pc=8:  MOVZ X2, #0x1234                 high qword payload
// pc=12: STP  X1, X2, [X0]                mem[0x200..0x210] = (0xabcd, 0x1234)
// pc=16: LDR  Q0, [X0]                    V0 = mem[0x200..0x210]
// pc=20: MOVZ X3, #0x300
// pc=24: STR  Q0, [X3]                    mem[0x300..0x310] = V0 (copy via SIMD)
//
// pc=28: MOVZ X4, #0xbeef
// pc=32: STR  X4, [X0, #16]               mem[0x210] = 0xbeef
// pc=36: LDR  D1, [X0, #16]               V1.D[0] = 0xbeef, V1.D[1] = 0
// pc=40: STR  D1, [X3, #16]               mem[0x310] = 0xbeef
// pc=44: RET

const wasmPath = '/Users/numi/workspace/jitter/engine/zig-out/bin/translator.wasm';
const wasm =
  typeof readFile === 'function' ? readFile(wasmPath, 'binary') : require('fs').readFileSync(wasmPath);
const out = typeof print === 'function' ? print : console.log;

WebAssembly.instantiate(wasm).then(({ instance }) => {
  const ex = instance.exports;
  const buf = new Uint8Array(ex.memory.buffer);

  const insns = [
    0xd2804000, // MOVZ X0, #0x200
    0xd29579a1, // MOVZ X1, #0xabcd
    0xd2824682, // MOVZ X2, #0x1234
    0xa9000801, // STP X1, X2, [X0, #0]   (offset_signed +0)
    0x3dc00000, // LDR Q0, [X0]           (size=00 opc=11, V=1)
    0xd2806003, // MOVZ X3, #0x300
    0x3d800060, // STR Q0, [X3]           (size=00 opc=10, V=1)
    0xd297dde4, // MOVZ X4, #0xbeef
    0xf9000804, // STR X4, [X0, #16]
    0xfd400801, // LDR D1, [X0, #16]      (size=11 opc=01, V=1)
    0xfd000861, // STR D1, [X3, #16]
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

    // V0 lives at v_offset = 0x200 in cpu state, indices 0x200/8=64 and 65
    out('X1=0x' + gb[1].toString(16) + ' (want 0xabcd)');
    out('X2=0x' + gb[2].toString(16) + ' (want 0x1234)');
    out('V0.lo=0x' + gb[64].toString(16) + ' (want 0xabcd)');
    out('V0.hi=0x' + gb[65].toString(16) + ' (want 0x1234)');

    const src = new BigUint64Array(guestMem.buffer, 0x80000 + 0x200, 2);
    const dst = new BigUint64Array(guestMem.buffer, 0x80000 + 0x300, 2);
    out('src qword pair = (0x' + src[0].toString(16) + ', 0x' + src[1].toString(16) + ')');
    out('dst qword pair = (0x' + dst[0].toString(16) + ', 0x' + dst[1].toString(16) + ')');

    // V1 at v_offset 0x210 = bytes 0x210-0x21f
    out('V1.lo=0x' + gb[66].toString(16) + ' (want 0xbeef)');
    out('V1.hi=0x' + gb[67].toString(16) + ' (want 0)');

    const srcD = new BigUint64Array(guestMem.buffer, 0x80000 + 0x210, 1);
    const dstD = new BigUint64Array(guestMem.buffer, 0x80000 + 0x310, 1);
    out('src D = 0x' + srcD[0].toString(16));
    out('dst D = 0x' + dstD[0].toString(16));

    let ok = true;
    if (gb[64] !== 0xabcdn) { out('V0.lo mismatch'); ok = false; }
    if (gb[65] !== 0x1234n) { out('V0.hi mismatch'); ok = false; }
    if (dst[0] !== 0xabcdn) { out('dst[0] mismatch — Q store failed'); ok = false; }
    if (dst[1] !== 0x1234n) { out('dst[1] mismatch — Q store failed'); ok = false; }
    if (gb[66] !== 0xbeefn) { out('V1.lo mismatch'); ok = false; }
    if (gb[67] !== 0n) { out('V1.hi should be zeroed on LDR D'); ok = false; }
    if (dstD[0] !== 0xbeefn) { out('dst D mismatch'); ok = false; }
    if (!ok) throw new Error('check failed');
    out('PASS — SIMD ldp/str q + ldr d / str d work');
  });
}).catch(e => {
  out('FAIL: ' + e + '\n' + (e.stack || ''));
  if (typeof process !== 'undefined') process.exit(1);
});
