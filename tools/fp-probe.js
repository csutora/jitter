// what does this host's wasm engine actually do with floating point?
//
// wasm is deliberately weaker than aarch64 in two places, and both are things a translator has to
// decide about rather than assume:
//   1. there is no fused multiply-add in the wasm MVP. relaxed-simd adds one, but an engine MAY
//      lower it fused or unfused, so it is only usable if you measure what this engine did.
//   2. NaN payload propagation is unspecified. an engine may return the operand's payload (which
//      is what aarch64 does with FPCR.DN clear) or a canonical NaN.
//
// so: measure. the answers pick the emission strategy.
//
// run against the device:  curl -s -X POST --data-binary @body http://maxwell:9080/ctl/js

function leb(n) { const out = []; do { let b = n & 0x7f; n >>>= 7; if (n) b |= 0x80; out.push(b); } while (n); return out; }

function mod(retType, body) {
  const sect = (id, payload) => [id, ...leb(payload.length), ...payload];
  const code = [0x00, ...body, 0x0b];
  return Uint8Array.from([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ...sect(1, [0x01, 0x60, 0x00, 0x01, retType]),
    ...sect(3, [0x01, 0x00]),
    ...sect(7, [0x01, 0x01, 0x66, 0x00, 0x00]),
    ...sect(10, [0x01, ...leb(code.length), ...code]),
  ]);
}
function run(retType, body) {
  try { return new WebAssembly.Instance(new WebAssembly.Module(mod(retType, body))).exports.f(); }
  catch (e) { return 'UNSUPPORTED: ' + e.message.slice(0, 70); }
}

// little-endian bytes of a value, for v128.const / f64.const operands
function f64b(v) { const u = new Uint8Array(8); new DataView(u.buffer).setFloat64(0, v, true); return [...u]; }
function f32b(v) { const u = new Uint8Array(4); new DataView(u.buffer).setFloat32(0, v, true); return [...u]; }
function i64b(hi, lo) { const u = new Uint8Array(8), d = new DataView(u.buffer); d.setUint32(0, lo, true); d.setUint32(4, hi, true); return [...u]; }
function sleb64(bytes) { // i64.const takes a signed LEB; emit the raw pattern via a 10-byte form
  let v = 0n; for (let i = 7; i >= 0; i--) v = (v << 8n) | BigInt(bytes[i]);
  const out = []; let more = true;
  while (more) { let b = Number(v & 0x7fn); v >>= 7n; const sign = b & 0x40;
    if ((v === 0n && !sign) || (v === -1n && sign)) more = false; else b |= 0x80; out.push(b); }
  return out;
}

function probe() {
  const out = {};

  // ── 1. fused multiply-add ──────────────────────────────────────────────
  // a*b lands just under the midpoint between two representable numbers, so rounding it away
  // (unfused) cancels exactly against c and gives zero, while keeping it (fused) leaves the error.
  const V = [0xfd, 0x0c];
  const dup = (bytes) => bytes.length === 8 ? [...bytes, ...bytes] : [...bytes, ...bytes, ...bytes, ...bytes];
  const fmaCase = (dbl) => {
    const a = dbl ? 1 + 2 ** -52 : Math.fround(1 + 2 ** -23);
    const b = dbl ? 1 - 2 ** -53 : Math.fround(1 - 2 ** -24);
    const lane = dbl ? f64b : f32b;
    const EX = dbl ? [0xfd, 0x21, 0x00] : [0xfd, 0x1f, 0x00];
    const MADD = [0xfd, ...leb(dbl ? 0x107 : 0x105)];
    const MUL = [0xfd, ...leb(dbl ? 0xf2 : 0xe6)], ADD = [0xfd, ...leb(dbl ? 0xf0 : 0xe4)];
    const ret = dbl ? 0x7c : 0x7d;
    return {
      unfused: run(ret, [...V, ...dup(lane(a)), ...V, ...dup(lane(b)), ...MUL, ...V, ...dup(lane(-1)), ...ADD, ...EX]),
      relaxed: run(ret, [...V, ...dup(lane(a)), ...V, ...dup(lane(b)), ...V, ...dup(lane(-1)), ...MADD, ...EX]),
    };
  };
  for (const dbl of [true, false]) {
    const k = dbl ? 'f64' : 'f32', r = fmaCase(dbl);
    out[k + '.fma.unfused'] = r.unfused;
    out[k + '.fma.relaxed_madd'] = r.relaxed;
    out[k + '.fma.available'] = typeof r.relaxed === 'number';
    out[k + '.fma.fuses'] = typeof r.relaxed === 'number' && r.relaxed !== 0;
  }

  // ── 2. NaN payload propagation ─────────────────────────────────────────
  // aarch64 with FPCR.DN clear returns the operand NaN, quieted. wasm leaves this open.
  // everything stays inside wasm so the JS number boundary cannot canonicalize it for us.
  const NAN_HI = 0x7ff80000, NAN_LO = 0x00abcdef;
  const payload = i64b(NAN_HI, NAN_LO);
  const asHex = (v) => typeof v === 'bigint' ? '0x' + v.toString(16).padStart(16, '0') : v;
  out['f64.nan.in'] = '0x7ff80000' + '00abcdef';
  // f64.reinterpret_i64 (0xbf), f64.add (0xa0), i64.reinterpret_f64 (0xbd)
  out['f64.nan.add1'] = asHex(run(0x7e, [0x42, ...sleb64(payload), 0xbf, 0x44, ...f64b(1), 0xa0, 0xbd]));
  out['f64.nan.mul1'] = asHex(run(0x7e, [0x42, ...sleb64(payload), 0xbf, 0x44, ...f64b(1), 0xa2, 0xbd]));
  // second operand carrying the payload instead of the first
  out['f64.nan.rhs'] = asHex(run(0x7e, [0x44, ...f64b(1), 0x42, ...sleb64(payload), 0xbf, 0xa0, 0xbd]));
  out['f64.nan.preserved'] = String(out['f64.nan.add1']).endsWith('00abcdef');

  return out;
}

if (typeof process !== 'undefined' && process.versions && process.versions.node) console.log(probe());
