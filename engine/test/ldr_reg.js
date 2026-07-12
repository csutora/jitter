// register-offset load/store + ldr literal + mrs/msr tpidr_el0 + clz + rev
//
// pc=0:  MOVZ X0, #0x100               base = 0x100
// pc=4:  MOVZ X1, #0xdead
// pc=8:  MOVZ X2, #5                   index = 5
// pc=12: STR  X1, [X0, X2, LSL #3]     mem[0x100 + 5*8] = 0xdead = mem[0x128] = 0xdead
// pc=16: LDR  X3, [X0, X2, LSL #3]     x3 = 0xdead
// pc=20: MOVZ X4, #0
// pc=24: MOVZ X5, #0x80                // mask 1 byte
// pc=28: STRB W5, [X0, X4]             mem[0x100] = 0x80
// pc=32: LDRSB X6, [X0, X4]            x6 = sign-extend(0x80) = 0xffff_ffff_ffff_ff80
// pc=36: LDRB  X7, [X0, X4]            x7 = 0x80
// pc=40: MOVZ X8, #0
// pc=44: MSR  TPIDR_EL0, X1            tpidr = 0xdead
// pc=48: MRS  X8, TPIDR_EL0            x8 = 0xdead
// pc=52: MOVZ X9, #0x100, lsl #16      x9 = 0x1000000
// pc=56: CLZ  X10, X9                  x10 = 39 (24 + 32 - 1... let me think. 0x100_0000 has bit 24 set → 64-25 = 39)
// pc=60: REV  X11, X9                  x11 = byte-reverse of 0x1000000_00000000... = 0x0000_0000_0000_0001 ... wait that's not right either
//                                       x9 = 0x0000_0000_0100_0000; reverse bytes = 0x0000_0001_0000_0000_0000_0000_0000_0000? as i64 = 0x0000_0000_0000_0001 reversed at byte level...
//                                       0x9 in bytes (le): 00 00 00 01 00 00 00 00; reversed: 00 00 00 00 01 00 00 00 = 0x0000_0000_0100_0000... that's the same
//                                       wait, x9 = 0x0000_0000_0100_0000 = bytes (msb first): 00 00 00 00 01 00 00 00
//                                       byte-reversed = 00 00 00 01 00 00 00 00 = 0x0000_0001_0000_0000
// pc=64: RET

const wasmPath = '/Users/numi/workspace/jitter/engine/zig-out/bin/translator.wasm';
const wasm =
  typeof readFile === 'function' ? readFile(wasmPath, 'binary') : require('fs').readFileSync(wasmPath);
const out = typeof print === 'function' ? print : console.log;

WebAssembly.instantiate(wasm).then(({ instance }) => {
  const ex = instance.exports;
  const buf = new Uint8Array(ex.memory.buffer);

  // Re-encoded by hand from ARM ARM
  const insns = [
    0xd2802000, // MOVZ X0, #0x100
    0xd29bd5a1, // MOVZ X1, #0xdead
    0xd28000a2, // MOVZ X2, #5
    0xf8227801, // STR X1, [X0, X2, LSL #3]
    0xf8627803, // LDR X3, [X0, X2, LSL #3]
    0xd2800004, // MOVZ X4, #0
    0xd2801005, // MOVZ X5, #0x80
    0x38246805, // STRB W5, [X0, X4]      (note: extended-reg form uses option=011 LSL, S=0)
    0x38a46806, // LDRSB X6, [X0, X4]
    0x38646807, // LDRB X7, [X0, X4]
    0xd2800008, // MOVZ X8, #0
    0xd51bd041, // MSR TPIDR_EL0, X1
    0xd53bd048, // MRS X8, TPIDR_EL0
    0xd2a02009, // MOVZ X9, #0x100, lsl #16  → x9 = 0x100_0000
    0xdac0112a, // CLZ X10, X9
    0xdac00d2b, // REV X11, X9
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

  const guestMem = new WebAssembly.Memory({ initial: Math.max(2, 3), maximum: 65536, shared: true });
  return WebAssembly.instantiate(moduleBytes, { env: { memory: guestMem } }).then(({ instance: inst2 }) => {
    inst2.exports.run();
    const gb = new BigUint64Array(guestMem.buffer);
    out('X1=0x' + gb[1].toString(16) + ' (want 0xdead)');
    out('X3=0x' + gb[3].toString(16) + ' (want 0xdead — register-offset load)');
    out('X6=0x' + gb[6].toString(16) + ' (want 0xffffffffffffff80 — ldrsb)');
    out('X7=0x' + gb[7].toString(16) + ' (want 0x80 — ldrb)');
    out('X8=0x' + gb[8].toString(16) + ' (want 0xdead — mrs tpidr after msr)');
    out('X9=0x' + gb[9].toString(16) + ' (want 0x1000000)');
    out('X10=' + gb[10] + ' (want 39 — CLZ)');
    out('X11=0x' + gb[11].toString(16) + ' (want 0x100000000 — REV)');

    let ok = true;
    if (gb[3] !== 0xdeadn) { out('X3 mismatch'); ok = false; }
    if (gb[6] !== 0xffffffffffffff80n) { out('X6 mismatch'); ok = false; }
    if (gb[7] !== 0x80n) { out('X7 mismatch'); ok = false; }
    if (gb[8] !== 0xdeadn) { out('X8 mismatch (tpidr)'); ok = false; }
    if (gb[10] !== 39n) { out('X10 mismatch (CLZ)'); ok = false; }
    if (gb[11] !== 0x100000000n) { out('X11 mismatch (REV)'); ok = false; }
    if (!ok) throw new Error('mismatches');
    out('PASS — register-offset ld/st + mrs/msr + clz + rev work');
  });
}).catch(e => {
  out('FAIL: ' + e + '\n' + (e.stack || ''));
  if (typeof process !== 'undefined') process.exit(1);
});
