// ARMv8 cryptographic-extension instructions (FEAT_AES, FEAT_PMULL, FEAT_SHA256) implemented as
// host helpers. advertising these in AT_HWCAP makes OpenSSL take its hardware-crypto code path
// (a handful of ops per block) instead of the software vpaes/NEON-poly1305 path, which is both
// faster through the jit and sidesteps the large software-crypto NEON decode surface.
//
// each function operates directly on the guest linear-memory Uint8Array at 16-byte V-register
// offsets (little-endian byte order, matching how AES/GHASH state is laid out in a V register).

// ── AES tables (generated once, with a self-check against known S-box entries) ──

const SBOX = new Uint8Array(256);
const INV_SBOX = new Uint8Array(256);
// GF(2^8) log/antilog with the AES reduction polynomial 0x11b and generator 0x03.
const GF_EXP = new Uint8Array(256);
const GF_LOG = new Uint8Array(256);

(function buildAesTables() {
  let x = 1;
  for (let i = 0; i < 255; i++) {
    GF_EXP[i] = x;
    GF_LOG[x] = i;
    // multiply by generator 3 = x*(1^2) : x ^ (x<<1), reduced by 0x11b
    let hi = x & 0x80;
    x = ((x << 1) & 0xff) ^ (hi ? 0x1b : 0) ^ x;
  }
  GF_EXP[255] = GF_EXP[0];
  const rotl8 = (b: number, n: number) => ((b << n) | (b >>> (8 - n))) & 0xff;
  for (let a = 0; a < 256; a++) {
    const inv = a === 0 ? 0 : GF_EXP[255 - GF_LOG[a]];
    const s = inv ^ rotl8(inv, 1) ^ rotl8(inv, 2) ^ rotl8(inv, 3) ^ rotl8(inv, 4) ^ 0x63;
    SBOX[a] = s & 0xff;
    INV_SBOX[s & 0xff] = a;
  }
  // self-check: well-known S-box values. wrong tables would silently break all AES.
  if (SBOX[0] !== 0x63 || SBOX[1] !== 0x7c || SBOX[0x53] !== 0xed || SBOX[0xff] !== 0x16) {
    throw new Error('AES S-box generation failed self-check');
  }
})();

// ShiftRows byte permutation for the column-major AES state (byte c*4+r). output[i] = input[SHIFT_ROWS[i]].
const SHIFT_ROWS = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11];
const INV_SHIFT_ROWS = new Array(16);
for (let i = 0; i < 16; i++) INV_SHIFT_ROWS[SHIFT_ROWS[i]] = i;

// GF(2^8) multiply via log tables.
function gmul(a: number, b: number): number {
  if (a === 0 || b === 0) return 0;
  return GF_EXP[(GF_LOG[a] + GF_LOG[b]) % 255];
}

function mixColumn(s0: number, s1: number, s2: number, s3: number, out: Uint8Array, off: number, m: number[]) {
  out[off + 0] = gmul(s0, m[0]) ^ gmul(s1, m[1]) ^ gmul(s2, m[2]) ^ gmul(s3, m[3]);
  out[off + 1] = gmul(s0, m[3]) ^ gmul(s1, m[0]) ^ gmul(s2, m[1]) ^ gmul(s3, m[2]);
  out[off + 2] = gmul(s0, m[2]) ^ gmul(s1, m[3]) ^ gmul(s2, m[0]) ^ gmul(s3, m[1]);
  out[off + 3] = gmul(s0, m[1]) ^ gmul(s1, m[2]) ^ gmul(s2, m[3]) ^ gmul(s3, m[0]);
}
const MIX = [2, 3, 1, 1];
const INV_MIX = [14, 11, 13, 9];

// AESE Vd, Vn: Vd = ShiftRows(SubBytes(Vd ^ Vn)). (SubBytes and ShiftRows commute.)
export function aese(buf: Uint8Array, vdA: number, vnA: number): void {
  const t = new Uint8Array(16);
  for (let i = 0; i < 16; i++) t[i] = SBOX[buf[vdA + i] ^ buf[vnA + i]];
  for (let i = 0; i < 16; i++) buf[vdA + i] = t[SHIFT_ROWS[i]];
}

// AESD Vd, Vn: Vd = InvShiftRows(InvSubBytes(Vd ^ Vn)).
export function aesd(buf: Uint8Array, vdA: number, vnA: number): void {
  const t = new Uint8Array(16);
  for (let i = 0; i < 16; i++) t[i] = INV_SBOX[buf[vdA + i] ^ buf[vnA + i]];
  for (let i = 0; i < 16; i++) buf[vdA + i] = t[INV_SHIFT_ROWS[i]];
}

// AESMC Vd, Vn: Vd = MixColumns(Vn).
export function aesmc(buf: Uint8Array, vdA: number, vnA: number): void {
  const out = new Uint8Array(16);
  for (let c = 0; c < 4; c++) mixColumn(buf[vnA + c * 4], buf[vnA + c * 4 + 1], buf[vnA + c * 4 + 2], buf[vnA + c * 4 + 3], out, c * 4, MIX);
  buf.set(out, vdA);
}

// AESIMC Vd, Vn: Vd = InvMixColumns(Vn).
export function aesimc(buf: Uint8Array, vdA: number, vnA: number): void {
  const out = new Uint8Array(16);
  for (let c = 0; c < 4; c++) mixColumn(buf[vnA + c * 4], buf[vnA + c * 4 + 1], buf[vnA + c * 4 + 2], buf[vnA + c * 4 + 3], out, c * 4, INV_MIX);
  buf.set(out, vdA);
}

// ── PMULL: 64x64 -> 128 carryless (polynomial) multiply in GF(2). ──
// part=false uses the low 64 bits of each source (PMULL), part=true uses the high 64 (PMULL2).
function readU64(buf: Uint8Array, base: number): bigint {
  let v = 0n;
  for (let k = 0; k < 8; k++) v |= BigInt(buf[base + k]) << BigInt(k * 8);
  return v;
}
export function pmull(buf: Uint8Array, vdA: number, vnA: number, vmA: number, part: boolean): void {
  const a = readU64(buf, vnA + (part ? 8 : 0));
  const b = readU64(buf, vmA + (part ? 8 : 0));
  let res = 0n;
  for (let i = 0n; i < 64n; i++) {
    if ((a >> i) & 1n) res ^= b << i;
  }
  for (let k = 0; k < 16; k++) buf[vdA + k] = Number((res >> BigInt(k * 8)) & 0xffn);
}

// ── SHA-1 / SHA-256 / SHA-512 (FEAT_SHA1, FEAT_SHA256, FEAT_SHA512) ──
// ported from the ARM ARM pseudocode (via qemu's crypto_helper.c). a V register holds four 32-bit
// words (SHA1/256) or two 64-bit words (SHA512) in little-endian order, so word i sits at byte
// offset 4*i (or 8*i) and round-trips with the same endianness as the rest of guest memory.

function writeU64(buf: Uint8Array, base: number, v: bigint): void {
  for (let k = 0; k < 8; k++) buf[base + k] = Number((v >> BigInt(k * 8)) & 0xffn);
}
function w32(buf: Uint8Array, base: number, i: number): number {
  const o = base + 4 * i;
  return (buf[o] | (buf[o + 1] << 8) | (buf[o + 2] << 16) | (buf[o + 3] << 24)) >>> 0;
}
function setW32(buf: Uint8Array, base: number, i: number, v: number): void {
  const o = base + 4 * i;
  buf[o] = v & 0xff; buf[o + 1] = (v >>> 8) & 0xff; buf[o + 2] = (v >>> 16) & 0xff; buf[o + 3] = (v >>> 24) & 0xff;
}
const ror32 = (x: number, n: number): number => ((x >>> n) | (x << (32 - n))) >>> 0;
const rol32 = (x: number, n: number): number => ror32(x, 32 - n);

const cho1 = (x: number, y: number, z: number) => ((x & (y ^ z)) ^ z) >>> 0;
const par1 = (x: number, y: number, z: number) => (x ^ y ^ z) >>> 0;
const maj1 = (x: number, y: number, z: number) => ((x & y) | ((x | y) & z)) >>> 0;

// SHA1 hash-update: variant 0=SHA1C (choose), 1=SHA1P (parity), 2=SHA1M (majority).
export function sha1_3reg(buf: Uint8Array, vdA: number, vnA: number, vmA: number, variant: number): void {
  const d = [w32(buf, vdA, 0), w32(buf, vdA, 1), w32(buf, vdA, 2), w32(buf, vdA, 3)];
  let n0 = w32(buf, vnA, 0);
  const m = [w32(buf, vmA, 0), w32(buf, vmA, 1), w32(buf, vmA, 2), w32(buf, vmA, 3)];
  for (let i = 0; i < 4; i++) {
    const fn = variant === 0 ? cho1(d[1], d[2], d[3]) : variant === 1 ? par1(d[1], d[2], d[3]) : maj1(d[1], d[2], d[3]);
    const t = (fn + rol32(d[0], 5) + n0 + m[i]) >>> 0;
    n0 = d[3]; d[3] = d[2]; d[2] = ror32(d[1], 2); d[1] = d[0]; d[0] = t;
  }
  for (let i = 0; i < 4; i++) setW32(buf, vdA, i, d[i]);
}
export function sha1h(buf: Uint8Array, vdA: number, vnA: number): void {
  const v = ror32(w32(buf, vnA, 0), 2);
  setW32(buf, vdA, 0, v); setW32(buf, vdA, 1, 0); setW32(buf, vdA, 2, 0); setW32(buf, vdA, 3, 0);
}
export function sha1su0(buf: Uint8Array, vdA: number, vnA: number, vmA: number): void {
  const d0 = readU64(buf, vdA + 8) ^ readU64(buf, vdA) ^ readU64(buf, vmA);
  const d1 = readU64(buf, vnA) ^ readU64(buf, vdA + 8) ^ readU64(buf, vmA + 8);
  writeU64(buf, vdA, d0); writeU64(buf, vdA + 8, d1);
}
export function sha1su1(buf: Uint8Array, vdA: number, vnA: number): void {
  const d = [w32(buf, vdA, 0), w32(buf, vdA, 1), w32(buf, vdA, 2), w32(buf, vdA, 3)];
  const m = [w32(buf, vnA, 0), w32(buf, vnA, 1), w32(buf, vnA, 2), w32(buf, vnA, 3)];
  d[0] = rol32((d[0] ^ m[1]) >>> 0, 1);
  d[1] = rol32((d[1] ^ m[2]) >>> 0, 1);
  d[2] = rol32((d[2] ^ m[3]) >>> 0, 1);
  d[3] = rol32((d[3] ^ d[0]) >>> 0, 1);
  for (let i = 0; i < 4; i++) setW32(buf, vdA, i, d[i]);
}

const S0 = (x: number) => (ror32(x, 2) ^ ror32(x, 13) ^ ror32(x, 22)) >>> 0;
const S1 = (x: number) => (ror32(x, 6) ^ ror32(x, 11) ^ ror32(x, 25)) >>> 0;
const s0 = (x: number) => (ror32(x, 7) ^ ror32(x, 18) ^ (x >>> 3)) >>> 0;
const s1 = (x: number) => (ror32(x, 17) ^ ror32(x, 19) ^ (x >>> 10)) >>> 0;

export function sha256h(buf: Uint8Array, vdA: number, vnA: number, vmA: number, part2: boolean): void {
  const d = [w32(buf, vdA, 0), w32(buf, vdA, 1), w32(buf, vdA, 2), w32(buf, vdA, 3)];
  const n = [w32(buf, vnA, 0), w32(buf, vnA, 1), w32(buf, vnA, 2), w32(buf, vnA, 3)];
  const m = [w32(buf, vmA, 0), w32(buf, vmA, 1), w32(buf, vmA, 2), w32(buf, vmA, 3)];
  if (!part2) {
    for (let i = 0; i < 4; i++) {
      let t = (cho1(n[0], n[1], n[2]) + n[3] + S1(n[0]) + m[i]) >>> 0;
      n[3] = n[2]; n[2] = n[1]; n[1] = n[0]; n[0] = (d[3] + t) >>> 0;
      t = (t + maj1(d[0], d[1], d[2]) + S0(d[0])) >>> 0;
      d[3] = d[2]; d[2] = d[1]; d[1] = d[0]; d[0] = t;
    }
  } else {
    for (let i = 0; i < 4; i++) {
      const t = (cho1(d[0], d[1], d[2]) + d[3] + S1(d[0]) + m[i]) >>> 0;
      d[3] = d[2]; d[2] = d[1]; d[1] = d[0]; d[0] = (n[3 - i] + t) >>> 0;
    }
  }
  for (let i = 0; i < 4; i++) setW32(buf, vdA, i, d[i]);
}
export function sha256su0(buf: Uint8Array, vdA: number, vnA: number): void {
  const d = [w32(buf, vdA, 0), w32(buf, vdA, 1), w32(buf, vdA, 2), w32(buf, vdA, 3)];
  const m0 = w32(buf, vnA, 0);
  setW32(buf, vdA, 0, (d[0] + s0(d[1])) >>> 0);
  setW32(buf, vdA, 1, (d[1] + s0(d[2])) >>> 0);
  setW32(buf, vdA, 2, (d[2] + s0(d[3])) >>> 0);
  setW32(buf, vdA, 3, (d[3] + s0(m0)) >>> 0);
}
export function sha256su1(buf: Uint8Array, vdA: number, vnA: number, vmA: number): void {
  const d = [w32(buf, vdA, 0), w32(buf, vdA, 1), w32(buf, vdA, 2), w32(buf, vdA, 3)];
  const n = [w32(buf, vnA, 0), w32(buf, vnA, 1), w32(buf, vnA, 2), w32(buf, vnA, 3)];
  const m = [w32(buf, vmA, 0), w32(buf, vmA, 1), w32(buf, vmA, 2), w32(buf, vmA, 3)];
  d[0] = (d[0] + s1(m[2]) + n[1]) >>> 0;
  d[1] = (d[1] + s1(m[3]) + n[2]) >>> 0;
  d[2] = (d[2] + s1(d[0]) + n[3]) >>> 0;
  d[3] = (d[3] + s1(d[1]) + m[0]) >>> 0;
  for (let i = 0; i < 4; i++) setW32(buf, vdA, i, d[i]);
}

const M64 = 0xffffffffffffffffn;
const ror64 = (x: bigint, n: bigint) => ((x >> n) | (x << (64n - n))) & M64;
const cho512 = (x: bigint, y: bigint, z: bigint) => ((x & (y ^ z)) ^ z) & M64;
const maj512 = (x: bigint, y: bigint, z: bigint) => ((x & y) | ((x | y) & z)) & M64;
const S0_512 = (x: bigint) => (ror64(x, 28n) ^ ror64(x, 34n) ^ ror64(x, 39n)) & M64;
const S1_512 = (x: bigint) => (ror64(x, 14n) ^ ror64(x, 18n) ^ ror64(x, 41n)) & M64;
const s0_512 = (x: bigint) => (ror64(x, 1n) ^ ror64(x, 8n) ^ (x >> 7n)) & M64;
const s1_512 = (x: bigint) => (ror64(x, 19n) ^ ror64(x, 61n) ^ (x >> 6n)) & M64;

export function sha512h(buf: Uint8Array, vdA: number, vnA: number, vmA: number): void {
  let d0 = readU64(buf, vdA); let d1 = readU64(buf, vdA + 8);
  const n0 = readU64(buf, vnA), n1 = readU64(buf, vnA + 8);
  const m0 = readU64(buf, vmA), m1 = readU64(buf, vmA + 8);
  d1 = (d1 + S1_512(m1) + cho512(m1, n0, n1)) & M64;
  const t = (d1 + m0) & M64;
  d0 = (d0 + S1_512(t) + cho512(t, m1, n0)) & M64;
  writeU64(buf, vdA, d0); writeU64(buf, vdA + 8, d1);
}
export function sha512h2(buf: Uint8Array, vdA: number, vnA: number, vmA: number): void {
  let d0 = readU64(buf, vdA); let d1 = readU64(buf, vdA + 8);
  const n0 = readU64(buf, vnA);
  const m0 = readU64(buf, vmA), m1 = readU64(buf, vmA + 8);
  d1 = (d1 + S0_512(m0) + maj512(n0, m1, m0)) & M64;
  d0 = (d0 + S0_512(d1) + maj512(d1, m0, m1)) & M64;
  writeU64(buf, vdA, d0); writeU64(buf, vdA + 8, d1);
}
export function sha512su0(buf: Uint8Array, vdA: number, vnA: number): void {
  const d0 = readU64(buf, vdA), d1 = readU64(buf, vdA + 8);
  const n0 = readU64(buf, vnA);
  writeU64(buf, vdA, (d0 + s0_512(d1)) & M64);
  writeU64(buf, vdA + 8, (d1 + s0_512(n0)) & M64);
}
export function sha512su1(buf: Uint8Array, vdA: number, vnA: number, vmA: number): void {
  const d0 = readU64(buf, vdA), d1 = readU64(buf, vdA + 8);
  const n0 = readU64(buf, vnA), n1 = readU64(buf, vnA + 8);
  const m0 = readU64(buf, vmA), m1 = readU64(buf, vmA + 8);
  writeU64(buf, vdA, (d0 + s1_512(n0) + m0) & M64);
  writeU64(buf, vdA + 8, (d1 + s1_512(n1) + m1) & M64);
}
