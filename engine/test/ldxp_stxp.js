// ldxp / stxp / casp — the 128-bit exclusive pair.
//
// a pair is ONE reservation covering both halves: stxp has to see both unchanged, and a stxp on a
// reservation that was already consumed must fail without touching memory. casp is the atomic form
// of the same idea and reports the value it found whether or not it swapped.
//
// mem[0x1000] = (0x1111, 0x2222)
//   ldxp x3,x4,[x0]            -> x3=0x1111 x4=0x2222, pair reserved
//   stxp w7,x5,x6,[x0]         -> w7=0 (success), mem = (0x3333, 0x4444)
//   stxp w11,x10,x10,[x0]      -> w11=1 (the reservation is gone), mem unchanged
//   casp x14,x15,x16,x17,[x0]  -> expected matches: mem = (0x6666, 0x7777), x14/x15 = old
//   casp x20,x21,x22,x23,[x0]  -> expected does not match: mem unchanged, x20/x21 = what is there

const SHADOW_BASE = 0x20000;
const GUEST_BASE = 0x80000;
const PAGES = 32;

const wasmPath = '/Users/numi/workspace/jitter/engine/zig-out/bin/translator.wasm';
const wasm =
  typeof readFile === 'function' ? readFile(wasmPath, 'binary') : require('fs').readFileSync(wasmPath);
const out = typeof print === 'function' ? print : console.log;

const insns = [
  0xd2820000, // mov  x0, #0x1000
  0xd2822221, // mov  x1, #0x1111
  0xd2844442, // mov  x2, #0x2222
  0xa9000801, // stp  x1, x2, [x0]
  0xc87f1003, // ldxp x3, x4, [x0]
  0xd2866665, // mov  x5, #0x3333
  0xd2888886, // mov  x6, #0x4444
  0xc8271805, // stxp w7, x5, x6, [x0]
  0xa9402408, // ldp  x8, x9, [x0]
  0xd28aaaaa, // mov  x10, #0x5555
  0xc82b280a, // stxp w11, x10, x10, [x0]
  0xa940340c, // ldp  x12, x13, [x0]
  0xd286666e, // mov  x14, #0x3333
  0xd288888f, // mov  x15, #0x4444
  0xd28cccd0, // mov  x16, #0x6666
  0xd28eeef1, // mov  x17, #0x7777
  0x482e7c10, // casp x14, x15, x16, x17, [x0]
  0xa9404c12, // ldp  x18, x19, [x0]
  0xd2824694, // mov  x20, #0x1234
  0xd2824695, // mov  x21, #0x1234
  0xd2911116, // mov  x22, #0x8888
  0xd2933337, // mov  x23, #0x9999
  0x48347c16, // casp x20, x21, x22, x23, [x0]
  0xa9406418, // ldp  x24, x25, [x0]
  0xd65f03c0, // ret
];

// the reservation covers BOTH halves. a plain store to the high word between ldxp and stxp has
// to make the stxp fail, which is the whole reason the monitor remembers two values.
const insnsInterfere = [
  0xd2820000, // mov  x0, #0x1000
  0xd2822221, // mov  x1, #0x1111
  0xd2844442, // mov  x2, #0x2222
  0xa9000801, // stp  x1, x2, [x0]
  0xc87f1003, // ldxp x3, x4, [x0]
  0xd2933325, // mov  x5, #0x9999
  0xf9000405, // str  x5, [x0, #8]      <- only the HIGH half changes
  0xd2866666, // mov  x6, #0x3333
  0xd2888887, // mov  x7, #0x4444
  0xc8281c06, // stxp w8, x6, x7, [x0]
  0xa9402809, // ldp  x9, x10, [x0]
  0xd65f03c0, // ret
];

function build(program, shared) {
  return WebAssembly.instantiate(wasm).then(({ instance }) => {
    const ex = instance.exports;
    const buf = new Uint8Array(ex.memory.buffer);
    const off = ex.translator_input_buf_offset();
    for (let i = 0; i < program.length; i++) {
      const v = program[i];
      buf[off + i * 4 + 0] = v & 0xff;
      buf[off + i * 4 + 1] = (v >>> 8) & 0xff;
      buf[off + i * 4 + 2] = (v >>> 16) & 0xff;
      buf[off + i * 4 + 3] = (v >>> 24) & 0xff;
    }
    ex.translator_reset();
    const rc = ex.translator_translate_region(0, 0, 0, 0, program.length * 4, SHADOW_BASE, shared ? 1 : 0, 0);
    if (rc !== 0) throw new Error('translate rc=' + rc);

    const outLen = ex.translator_result_len();
    const outOff = ex.translator_result_offset();
    const moduleBytes = new Uint8Array(outLen);
    for (let i = 0; i < outLen; i++) moduleBytes[i] = buf[outOff + i];

    const guestMem = shared
      ? new WebAssembly.Memory({ initial: PAGES, maximum: PAGES, shared: true })
      : new WebAssembly.Memory({ initial: PAGES, maximum: PAGES });
    // the translator always declares its helper imports; this program uses neither.
    const env = {
      memory: guestMem,
      simd_op: () => { throw new Error('simd_op should not be reached'); },
      mem_fault: () => { throw new Error('mem_fault: guest access out of bounds'); },
    };
    if (shared) env.reg_base = 0;
    return WebAssembly.instantiate(moduleBytes, { env }).then(({ instance: inst2 }) => {
      inst2.exports.run();
      return {
        x: new BigUint64Array(guestMem.buffer, 0, 32),
        mem: new BigUint64Array(guestMem.buffer, GUEST_BASE + 0x1000, 2),
      };
    });
  });
}

function check(label, want) {
  let ok = true;
  for (const [name, got, exp] of want) {
    if (got !== exp) { out(`${label}: ${name} = ${got} want ${exp}`); ok = false; }
  }
  if (!ok) throw new Error(label + ' check failed');
  out(`PASS (${label})`);
}

function run(shared) {
  const label = shared ? 'shared' : 'single';
  return build(insns, shared).then(({ x, mem }) => {
      check(label + ' ldxp/stxp/casp', [
        ['x3  ldxp lo', x[3], 0x1111n],
        ['x4  ldxp hi', x[4], 0x2222n],
        ['x7  stxp ok', x[7], 0n],
        ['x8  after lo', x[8], 0x3333n],
        ['x9  after hi', x[9], 0x4444n],
        ['x11 stxp again fails', x[11], 1n],
        ['x12 unchanged lo', x[12], 0x3333n],
        ['x13 unchanged hi', x[13], 0x4444n],
        ['x14 casp old lo', x[14], 0x3333n],
        ['x15 casp old hi', x[15], 0x4444n],
        ['x18 casp swapped lo', x[18], 0x6666n],
        ['x19 casp swapped hi', x[19], 0x7777n],
        ['x20 casp miss reports lo', x[20], 0x6666n],
        ['x21 casp miss reports hi', x[21], 0x7777n],
        ['x24 casp miss left lo', x[24], 0x6666n],
        ['x25 casp miss left hi', x[25], 0x7777n],
        ['mem lo', mem[0], 0x6666n],
        ['mem hi', mem[1], 0x7777n],
      ]);
    })
    .then(() => build(insnsInterfere, shared))
    .then(({ x, mem }) => {
      check(label + ' interfering store', [
        ['x8  stxp must fail', x[8], 1n],
        ['x9  memory kept lo', x[9], 0x1111n],
        ['x10 memory kept hi', x[10], 0x9999n],
        ['mem lo', mem[0], 0x1111n],
        ['mem hi', mem[1], 0x9999n],
      ]);
    });
}

run(false).then(() => run(true)).catch((e) => {
  out('FAIL: ' + e + '\n' + (e.stack || ''));
  if (typeof process !== 'undefined') process.exit(1);
});
