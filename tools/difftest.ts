// differential test: run the same instruction on this Mac's own arm64 core and through the zig
// translator, and report every register the two disagree about.
//
// the reference is the hardware, so there is nothing to be wrong about. a translated instruction
// that computes a plausible-but-different answer (the class of bug that costs the most to find,
// because it surfaces as a mysterious failure three layers up) shows up here as a named register
// with two values.
//
// usage:
//   npx tsx tools/difftest.ts <group> [iterations]
//   npx tsx tools/difftest.ts list

import { readFileSync } from 'fs';
import { join } from 'path';
import { spawn } from 'child_process';
import { Emulator } from '../src/dispatch.js';
import { loadElf } from '../src/elf.js';
import { GUEST_MEM_BASE, V_OFFSET } from '../src/types.js';
import { initZigTranslator, zigDecodeKind } from '../src/zig-translator.js';

const NREG = 28, NVEC = 28, SCRATCH = 512;
const STATE_BYTES = NREG * 8 + 16 + NVEC * 16 + SCRATCH;
const SCRATCH_OFF = NREG * 8 + 16 + NVEC * 16;
// where the guest sees the scratch buffer. the reference's copy lives at whatever address the
// host gave it, so both sides report the base register as a delta from their own base and the
// comparison stays meaningful.
const GUEST_SCRATCH = 0x710000;
const REC_IN = 8 + STATE_BYTES;
const REC_OUT = 8 + STATE_BYTES;

const PC = 0x600000, RET = 0xd65f03c0;
const NZCV_OFFSET = 0x108;

interface CpuState { x: bigint[]; nzcv: bigint; v: Uint8Array[]; scratch: Uint8Array }

// ── the reference: a child process that executes the instruction on real hardware ──

class Reference {
  private proc;
  private pending: Array<(r: { trapped: boolean; state: CpuState }) => void> = [];
  private buf = Buffer.alloc(0);

  constructor(bin: string) {
    this.proc = spawn(bin, [], { stdio: ['pipe', 'pipe', 'inherit'] });
    this.proc.stdout.on('data', (chunk: Buffer) => {
      this.buf = Buffer.concat([this.buf, chunk]);
      while (this.buf.length >= REC_OUT) {
        const rec = this.buf.subarray(0, REC_OUT);
        this.buf = this.buf.subarray(REC_OUT);
        const trapped = rec.readUInt32LE(0) !== 0;
        this.pending.shift()!({ trapped, state: unpack(rec, 8) });
      }
    });
  }

  run(insn: number, st: CpuState, baseReg: number): Promise<{ trapped: boolean; state: CpuState }> {
    const rec = Buffer.alloc(REC_IN);
    rec.writeUInt32LE(insn >>> 0, 0);
    rec.writeInt32LE(baseReg, 4);
    pack(rec, 8, st);
    return new Promise((resolve) => {
      this.pending.push(resolve);
      this.proc.stdin.write(rec);
    });
  }

  close(): void { this.proc.stdin.end(); }
}

function pack(b: Buffer, off: number, st: CpuState): void {
  for (let i = 0; i < NREG; i++) b.writeBigUInt64LE(BigInt.asUintN(64, st.x[i]), off + i * 8);
  b.writeBigUInt64LE(BigInt.asUintN(64, st.nzcv), off + NREG * 8);
  for (let i = 0; i < NVEC; i++) b.set(st.v[i], off + NREG * 8 + 16 + i * 16);
  b.set(st.scratch, off + SCRATCH_OFF);
}

function unpack(b: Buffer, off: number): CpuState {
  const x: bigint[] = [], v: Uint8Array[] = [];
  for (let i = 0; i < NREG; i++) x.push(b.readBigUInt64LE(off + i * 8));
  const nzcv = b.readBigUInt64LE(off + NREG * 8);
  for (let i = 0; i < NVEC; i++) v.push(new Uint8Array(b.subarray(off + NREG * 8 + 16 + i * 16, off + NREG * 8 + 16 + (i + 1) * 16)));
  return { x, nzcv, v, scratch: new Uint8Array(b.subarray(off + SCRATCH_OFF, off + SCRATCH_OFF + SCRATCH)) };
}

// ── the subject: the same instruction through the translator ──
//
// by default this runs ON THE DEVICE, because the translator emits wasm and the wasm is executed
// by the host engine. running it here instead would compare the reference against "translator +
// this laptop's engine", which is not the thing that ships: wasm leaves NaN payload propagation
// to the implementation, and engines disagree about which proposals they accept at all. --local
// runs it in-process as a fast pre-check, and says so in the output.

let emu: Emulator;
let lastExitPc: bigint = 0n;

class DeviceSubject {
  constructor(private url: string) {}

  /** run a batch on the device; returns one {refused, state} per input record.
   *  the bridge cannot hand back an arbitrarily large string, and where the limit falls is not
   *  documented, so a batch that comes back malformed is split and retried rather than tuned to
   *  a magic size that happens to work today. */
  async run(recs: Buffer[]): Promise<Array<{ refused: boolean; state: CpuState }>> {
    try {
      return await this.once(recs);
    } catch (e) {
      if (recs.length === 1) throw e;
      const mid = recs.length >> 1;
      return [...await this.run(recs.slice(0, mid)), ...await this.run(recs.slice(mid))];
    }
  }

  private async once(recs: Buffer[]): Promise<Array<{ refused: boolean; state: CpuState }>> {
    const body = Buffer.concat(recs).toString('base64');
    const resp = await fetch(this.url, { method: 'POST', body: `return __jitterDiff(${JSON.stringify(body)});` });
    const text = (await resp.text()).trim();
    if (!text || text.startsWith('[ctl/js]') || text === 'nil') throw new Error(`device said: ${text.slice(0, 120)}`);
    const out = Buffer.from(text, 'base64');
    const stride = STATE_BYTES + 8;
    if (out.length !== recs.length * stride) throw new Error(`device returned ${out.length} bytes, expected ${recs.length * stride}`);
    return recs.map((_, i) => ({
      refused: out.readUInt32LE(i * stride) !== 0,
      state: unpack(out, i * stride + 8),
    }));
  }
}

function runTranslated(insn: number, st: CpuState, baseReg: number): CpuState {
  const mem = new DataView(emu.memory.buffer);
  mem.setUint32(PC + GUEST_MEM_BASE, insn >>> 0, true);
  mem.setUint32(PC + 4 + GUEST_MEM_BASE, RET, true);
  for (let i = 0; i < 32; i++) mem.setBigUint64(i * 8, 0n, true);
  for (let i = 0; i < NREG; i++) mem.setBigUint64(i * 8, BigInt.asUintN(64, st.x[i]), true);
  for (let i = 0; i < 32; i++) {
    mem.setBigUint64(V_OFFSET + i * 16, 0n, true);
    mem.setBigUint64(V_OFFSET + i * 16 + 8, 0n, true);
  }
  for (let i = 0; i < NVEC; i++) new Uint8Array(emu.memory.buffer).set(st.v[i], V_OFFSET + i * 16);
  mem.setUint32(NZCV_OFFSET, Number(BigInt.asUintN(32, st.nzcv)), true);
  mem.setBigUint64(0x100, BigInt(PC), true);
  mem.setBigUint64(0xf8, 0x700000n, true);         // stack pointer, somewhere mapped
  mem.setBigUint64(30 * 8, BigInt(PC + 0x800), true); // link register: return out of the region
  const bytes = new Uint8Array(emu.memory.buffer);
  bytes.set(st.scratch, GUEST_SCRATCH + GUEST_MEM_BASE);
  if (baseReg >= 0) mem.setBigUint64(baseReg * 8, BigInt(GUEST_SCRATCH + SCRATCH / 2), true);

  (emu as any).regionCache.clear();
  // a region that refuses an instruction exits AT that instruction's pc instead of running on,
  // which is how "the translator does not implement this" is told apart from "it computed the
  // wrong answer". the two deserve very different attention.
  lastExitPc = (emu as any).getRegion(BigInt(PC)).run();

  const out = new DataView(emu.memory.buffer);
  const x: bigint[] = [], v: Uint8Array[] = [];
  for (let i = 0; i < NREG; i++) x.push(out.getBigUint64(i * 8, true));
  for (let i = 0; i < NVEC; i++) v.push(new Uint8Array(emu.memory.buffer.slice(V_OFFSET + i * 16, V_OFFSET + (i + 1) * 16)));
  if (baseReg >= 0) x[baseReg] = BigInt.asUintN(64, x[baseReg] - BigInt(GUEST_SCRATCH + SCRATCH / 2));
  return {
    x, nzcv: BigInt(out.getUint32(NZCV_OFFSET, true)) & 0xffffffffn, v,
    scratch: new Uint8Array(emu.memory.buffer.slice(GUEST_SCRATCH + GUEST_MEM_BASE, GUEST_SCRATCH + GUEST_MEM_BASE + SCRATCH)),
  };
}

// ── instruction generation ──
//
// random 32-bit words would be almost entirely undefined encodings, so each group names a fixed
// opcode skeleton and which bit-fields are free. that keeps every generated word a real
// instruction of the kind we care about.

interface Field { hi: number; lo: number; max?: number; reg?: boolean }
interface Group {
  name: string; base: number; fields: Field[]; vecs?: boolean; mem?: boolean; index?: boolean;
  // where a multiply-accumulate keeps its addend, so an FMA triple can be planted into the
  // registers this instruction actually names. bit 22 selects double in both shapes.
  fma?: 'ra' | 'rd';
}

const R = (hi: number, lo: number, max?: number): Field => ({ hi, lo, max });
// capped at 24, not 27: the multi-register forms (TBL/TBX with a four-register table, LD1..LD4)
// name CONSECUTIVE registers starting at Rn, and anything past v27 is harness state the
// reference cannot carry. leaving three spare keeps every such group inside the compared set.
const REG = (hi: number, lo: number): Field => ({ hi, lo, max: 24, reg: true });
// every register field is capped at 27: x28..x30 and v28..v31 carry the harness's own state, so
// an instruction naming them would diverge for a reason that has nothing to do with the
// translator. `Ra` (bits 14:10 of the 3-source forms) is a register too and needs the same cap.
const RD = REG(4, 0), RN = REG(9, 5), RM = REG(20, 16);

const GROUPS: Group[] = [
  // data processing, register forms. sf, shift type and amount all vary.
  { name: 'addsub-reg', base: 0x0b000000, fields: [RD, RN, RM, R(31, 31), R(30, 29), R(23, 22), R(15, 10)] },
  { name: 'logical-reg', base: 0x0a000000, fields: [RD, RN, RM, R(31, 31), R(30, 29), R(23, 21), R(15, 10)] },
  { name: 'addsub-imm', base: 0x11000000, fields: [RD, RN, R(31, 31), R(30, 29), R(23, 22, 1), R(21, 10)] },
  { name: 'logical-imm', base: 0x12000000, fields: [RD, RN, R(31, 31), R(30, 29), R(22, 10)] },
  { name: 'bitfield', base: 0x13000000, fields: [RD, RN, R(31, 31), R(30, 29), R(22, 22), R(21, 10)] },
  { name: 'shift-var', base: 0x1ac02000, fields: [RD, RN, RM, R(31, 31), R(11, 10)] },
  { name: 'dp1', base: 0x5ac00000, fields: [RD, RN, R(31, 31), R(15, 10, 0x13)] },
  { name: 'dp3', base: 0x9b000000, fields: [RD, RN, RM, R(31, 31), R(23, 21), R(15, 15), REG(14, 10)] },
  { name: 'cond-select', base: 0x1a800000, fields: [RD, RN, RM, R(31, 31), R(30, 29), R(15, 12), R(11, 10)] },
  { name: 'cond-cmp-reg', base: 0x3a400000, fields: [RN, RM, R(31, 31), R(30, 30), R(15, 12), R(3, 0)] },
  { name: 'cond-cmp-imm', base: 0x3a400800, fields: [RN, R(31, 31), R(30, 30), R(20, 16), R(15, 12), R(3, 0)] },
  { name: 'extract', base: 0x13800000, fields: [RD, RN, RM, R(31, 31), R(22, 22), R(15, 10)] },
  { name: 'addsub-extreg', base: 0x0b200000, fields: [RD, RN, RM, R(31, 31), R(30, 29), R(15, 13), R(12, 10, 4)] },
  { name: 'addsub-carry', base: 0x1a000000, fields: [RD, RN, RM, R(31, 31), R(30, 29)] },

  // NEON three-same and two-reg-misc, over every size and Q
  { name: 'neon-3same', base: 0x0e200400, fields: [RD, RN, RM, R(30, 29), R(23, 22), R(15, 11)], vecs: true, fma: 'rd' },
  { name: 'neon-2misc', base: 0x0e200800, fields: [RD, RN, R(30, 29), R(23, 22), R(16, 12)], vecs: true },
  { name: 'neon-3diff', base: 0x0e200000, fields: [RD, RN, RM, R(30, 29), R(23, 22), R(15, 12)], vecs: true },
  { name: 'neon-shift-imm', base: 0x0f000400, fields: [RD, RN, R(30, 29), R(22, 16), R(15, 11)], vecs: true },
  { name: 'neon-permute', base: 0x0e000000, fields: [RD, RN, RM, R(30, 30), R(23, 22), R(14, 12)], vecs: true },
  { name: 'neon-copy', base: 0x0e000400, fields: [RD, RN, R(30, 29), R(20, 16), R(14, 11)], vecs: true },
  { name: 'neon-scalar-3same', base: 0x5e200400, fields: [RD, RN, RM, R(29, 29), R(23, 22), R(15, 11)], vecs: true },
  { name: 'neon-scalar-2misc', base: 0x5e200800, fields: [RD, RN, R(29, 29), R(23, 22), R(16, 12)], vecs: true },
  { name: 'neon-scalar-shift-imm', base: 0x5f000400, fields: [RD, RN, R(29, 29), R(22, 16), R(15, 11)], vecs: true },

  // scalar floating point
  { name: 'fp-2source', base: 0x1e200800, fields: [RD, RN, RM, R(23, 22, 1), R(15, 12)], vecs: true },
  { name: 'fp-1source', base: 0x1e204000, fields: [RD, RN, R(23, 22, 1), R(20, 15)], vecs: true },
  { name: 'fp-compare', base: 0x1e202000, fields: [RN, RM, R(23, 22, 1), R(4, 3)], vecs: true },
  { name: 'fp-condsel', base: 0x1e200c00, fields: [RD, RN, RM, R(23, 22, 1), R(15, 12)], vecs: true },
  { name: 'fp-condcmp', base: 0x1e200400, fields: [RN, RM, R(23, 22, 1), R(15, 12), R(3, 0)], vecs: true },
  { name: 'fp-int-cvt', base: 0x1e200000, fields: [RD, RN, R(31, 31), R(23, 22, 1), R(20, 16)], vecs: true },
  { name: 'fp-fixed-cvt', base: 0x1e000000, fields: [RD, RN, R(31, 31), R(23, 22, 1), R(20, 16), R(15, 10)], vecs: true },
  { name: 'fp-3source', base: 0x1f000000, fields: [RD, RN, RM, R(23, 22, 1), R(21, 21), R(15, 15), REG(14, 10)], vecs: true, fma: 'ra' },

  // load/store. Rn is the addressing base and gets pointed at the scratch buffer; the offsets
  // are kept small enough to stay inside it, which is why the immediate fields are capped.
  // bit 26 is V: with it varying, the same address forms cover the SIMD loads and stores too.
  { name: 'ldst-uimm', base: 0x39000000, fields: [RD, RN, R(31, 30), R(26, 26), R(23, 22), R(21, 10, 15)], mem: true, vecs: true },
  { name: 'ldst-simm', base: 0x38000000, fields: [RD, RN, R(31, 30), R(26, 26), R(23, 22), R(20, 12, 0x1f), R(11, 10)], mem: true, vecs: true },
  { name: 'ldst-reg', base: 0x38200800, fields: [RD, RN, R(31, 30), R(26, 26), R(23, 22), REG(20, 16), R(15, 13), R(12, 12)], mem: true, vecs: true, index: true },
  // LDP/STP: signed-offset, pre-index and post-index all live in bits 24:23
  { name: 'ldst-pair', base: 0x29000000, fields: [RD, RN, REG(14, 10), R(31, 30, 2), R(26, 26), R(24, 23, 2), R(22, 22), R(21, 15, 7)], mem: true, vecs: true },
];

// which field of an encoding names the addressing base. every load/store form in the table above
// puts it in the same place.
const BASE_REG_FIELD = { hi: 9, lo: 5 };

let seed = Number(process.argv.find((a) => a.startsWith('--seed='))?.slice(7) ?? 0x9e3779b9) >>> 0;
// xorshift32. the obvious `seed = seed * A + C` congruential generator is useless here: its
// low-order bits have periods of 2, 4, 8..., and every field of an instruction is drawn from
// exactly those bits, so the "random" words repeat after a handful of draws.
function rnd(): number {
  seed ^= seed << 13; seed >>>= 0;
  seed ^= seed >>> 17;
  seed ^= seed << 5; seed >>>= 0;
  return seed;
}
function rndBig(): bigint {
  // boundaries are where the bugs live: the edges of every field width, the signs, and the
  // values one step either side of them.
  const pick = rnd() % 16;
  switch (pick) {
    case 0: return 0n;
    case 1: return 1n;
    case 2: return 0xffffffffffffffffn;                 // -1 / UINT64_MAX
    case 3: return 0xffffffffn;                         // UINT32_MAX
    case 4: return 0x8000000000000000n;                 // INT64_MIN
    case 5: return 0x7fffffffffffffffn;                 // INT64_MAX
    case 6: return 0x80000000n;                         // INT32_MIN as a 32-bit pattern
    case 7: return 0x7fffffffn;                         // INT32_MAX
    case 8: return BigInt(rnd() % 256);
    case 9: return 1n << BigInt(rnd() % 64);            // a single bit
    case 10: return (1n << BigInt(rnd() % 64)) - 1n;    // a run of low bits
    case 11: return 0xffffffffffffffffn ^ (1n << BigInt(rnd() % 64)); // one bit clear
    default: return (BigInt(rnd()) << 32n) | BigInt(rnd());
  }
}

// ── floating point operands ───────────────────────────────────────────────
//
// uniformly random bits are the wrong distribution for floating point. what an fp instruction
// does interestingly happens at zero, at the subnormal boundary, at infinity, inside NaN
// payloads, and above all in whether a rounding decision goes one way or the other. a random
// 64-bit pattern lands on none of those. that is exactly how a fused-multiply-add bug survived
// thousands of "clean" cases: fusion only shows when the product's discarded bits decide the
// sum, and that has to be built, not stumbled upon.

const bits = new DataView(new ArrayBuffer(8));
const f64FromBits = (b: bigint) => { bits.setBigUint64(0, b & 0xffffffffffffffffn); return bits.getFloat64(0); };
const f32FromBits = (b: number) => { bits.setUint32(0, b >>> 0); return bits.getFloat32(0); };

/** a value just off 1.0 with random low mantissa bits. the product of two of these needs far more
 *  significant bits than the format holds, so rounding it is a real decision rather than a no-op. */
function nearOne(dbl: boolean): number {
  return dbl
    ? f64FromBits(0x3ff0000000000000n | (BigInt(rnd()) & 0xffffffffn) | ((BigInt(rnd()) & 0xfffffn) << 32n))
    : f32FromBits(0x3f800000 | (rnd() & 0x7fffff));
}

function interestingFp(dbl: boolean): number {
  const neg = (rnd() & 1) ? -1 : 1;
  switch (rnd() % 16) {
    case 0: return neg * 0;                                                    // +0 and -0
    case 1: return dbl ? f64FromBits(1n) : f32FromBits(1);                     // smallest subnormal
    case 2: return dbl ? f64FromBits(BigInt(rnd()) & 0xfffffffffffffn)         // somewhere in the subnormals
                       : f32FromBits(rnd() & 0x7fffff);
    case 3: return dbl ? f64FromBits(0x0010000000000000n) : f32FromBits(0x00800000); // smallest normal
    case 4: return dbl ? f64FromBits(0x7fefffffffffffffn) : f32FromBits(0x7f7fffff); // largest finite
    case 5: return neg * Infinity;
    case 6: return dbl ? f64FromBits(0x7ff8000000000000n | (BigInt(rnd()) & 0x7ffffffffffffn))
                       : f32FromBits(0x7fc00000 | (rnd() & 0x3fffff));         // quiet NaN with a payload
    case 7: return dbl ? f64FromBits(0x7ff0000000000000n | (BigInt(rnd() | 1) & 0x7ffffffffffffn))
                       : f32FromBits(0x7f800000 | ((rnd() | 1) & 0x3fffff));   // signalling NaN
    case 8: case 9: case 10: case 11: return neg * nearOne(dbl);
    case 12: return neg * 2 ** ((rnd() % 40) - 20);                            // exact powers of two
    default: return dbl ? f64FromBits((BigInt(rnd()) << 32n) | BigInt(rnd())) : f32FromBits(rnd());
  }
}

/** a, b and the exact negation of their ROUNDED product. the add then cancels to zero unless the
 *  bits the multiply threw away survive into it, which is the whole difference between a fused
 *  multiply-add and a separate multiply and add. */
function fmaTriple(dbl: boolean): [number, number, number] {
  const round = dbl ? (x: number) => x : Math.fround;
  const a = round(((rnd() & 1) ? -1 : 1) * nearOne(dbl));
  const b = round(((rnd() & 1) ? -1 : 1) * nearOne(dbl));
  return [a, b, -round(a * b)];
}

function setLanes(reg: Uint8Array, value: number, dbl: boolean): void {
  const d = new DataView(reg.buffer, reg.byteOffset, 16);
  if (dbl) { d.setFloat64(0, value, true); d.setFloat64(8, value, true); }
  else for (let i = 0; i < 4; i++) d.setFloat32(i * 4, value, true);
}

function generate(g: Group): number {
  let w = g.base >>> 0;
  for (const f of g.fields) {
    const width = f.hi - f.lo + 1;
    const mask = width === 32 ? 0xffffffff : (1 << width) - 1;
    let val = rnd() & mask;
    if (f.max !== undefined) val = val % (f.max + 1);
    // x18 is the reserved platform register on apple silicon: the OS may clobber it at any
    // point, so the reference cannot report what an instruction did to it. never name it.
    if (f.reg && val === 18) val = 17;
    w = (w & ~(mask << f.lo)) >>> 0;
    w = (w | (val << f.lo)) >>> 0;
  }
  return w >>> 0;
}

function randomState(withVecs: boolean): CpuState {
  const x: bigint[] = [], v: Uint8Array[] = [];
  const scratch = new Uint8Array(SCRATCH);
  for (let i = 0; i < SCRATCH; i++) scratch[i] = rnd() & 0xff;
  for (let i = 0; i < NREG; i++) x.push(rndBig());
  for (let i = 0; i < NVEC; i++) {
    const b = new Uint8Array(16);
    if (withVecs) {
      const d = new DataView(b.buffer);
      switch (rnd() % 6) {
        case 0: break;                                    // all zero
        case 1: b.fill(0xff); break;                      // all ones
        case 2: for (let k = 0; k < 16; k++) b[k] = rnd() & 0xff; break;
        // read as doubles, and as singles: the same register is both depending on the
        // instruction, so both interpretations need to be interesting
        case 3: for (let l = 0; l < 2; l++) d.setFloat64(l * 8, interestingFp(true), true); break;
        case 4: for (let l = 0; l < 4; l++) d.setFloat32(l * 4, interestingFp(false), true); break;
        // lanes that differ from each other, so a per-lane bug cannot hide behind uniformity
        default: for (let l = 0; l < 4; l++) d.setFloat32(l * 4, l % 2 ? interestingFp(false) : nearOne(false), true);
      }
    }
    v.push(b);
  }
  return { x, nzcv: BigInt((rnd() & 0xf) << 28), v, scratch };
}

// ── main ──

async function main() {
  const args = process.argv.slice(2);
  if (args[0] === 'list') { for (const g of GROUPS) console.log(g.name); return; }

  initZigTranslator(new Uint8Array(readFileSync(
    join(import.meta.dirname, '..', 'engine', 'zig-out', 'bin', 'translator.wasm'))));
  emu = new Emulator(1024);
  emu.elfBytes = new Uint8Array(readFileSync(join(import.meta.dirname, '..', 'assets', 'busybox-aarch64')));
  loadElf(emu.elfBytes, emu.memory);

  const refBin = args.find((a) => a.startsWith('--ref='))?.slice(6)
    ?? join(import.meta.dirname, '..', 'build', 'armref');
  const ref = new Reference(refBin);

  // the subject runs on the device unless told otherwise, because that is the engine that will
  // execute the emitted wasm in production. --local is a fast pre-check, not a verification.
  const local = args.includes('--local');
  const deviceUrl = args.find((a) => a.startsWith('--device='))?.slice(9) ?? 'http://maxwell:9080/ctl/js';
  const device = local ? null : new DeviceSubject(deviceUrl);
  console.log(local ? 'subject: LOCAL engine (pre-check only)' : `subject: device at ${deviceUrl}`);

  const want = args[0] && !args[0].startsWith('--') && !/^\d+$/.test(args[0]) ? args[0] : null;
  const iters = Number(args.find((a) => /^\d+$/.test(a)) ?? 2000);
  const groups = want ? GROUPS.filter((g) => g.name === want) : GROUPS;
  if (!groups.length) { console.error(`no group named ${want}`); process.exit(2); }

  let checked = 0, skipped = 0, nanPayload = 0;
  const bad = new Map<string, { insn: number; detail: string; count: number }>();
  // an instruction the decoder refuses is a gap, not a wrong answer: it raises SIGILL in the
  // guest, which is loud and honest. worth listing, but separately from a silent miscomputation.
  const missing = new Map<string, { insn: number; count: number }>();

  // cases are collected and handed to the subject in batches: on the device that turns a sweep
  // into a few dozen round trips instead of one per instruction.
  interface Case { g: Group; insn: number; st: CpuState; baseReg: number; ref: CpuState }
  let pending: Case[] = [];
  const BATCH = device ? 128 : 1;

  for (const g of groups) {
    for (let i = 0; i < iters; i++) {
      const insn = generate(g);
      const st = randomState(!!g.vecs);
      // plant an FMA triple into the registers this instruction names. without it the operands
      // are independent and the multiply's rounding never reaches the sum, so a fused
      // multiply-add and a separate multiply-then-add agree on essentially every random case.
      if (g.fma && rnd() % 2) {
        const dbl = ((insn >>> 22) & 1) === 1;
        const [a, b, c] = fmaTriple(dbl);
        const rn = (insn >>> 5) & 0x1f, rm = (insn >>> 16) & 0x1f;
        const acc = g.fma === 'ra' ? (insn >>> 10) & 0x1f : insn & 0x1f;
        if (rn < NVEC && rm < NVEC && acc < NVEC && rn !== rm && rn !== acc && rm !== acc) {
          setLanes(st.v[rn], a, dbl);
          setLanes(st.v[rm], b, dbl);
          setLanes(st.v[acc], c, dbl);
        }
      }
      const baseReg = g.mem ? (insn >>> BASE_REG_FIELD.lo) & 0x1f : -1;
      if (g.mem) {
        // the two sides put the scratch buffer at different addresses, so any register that
        // ends up holding the base itself is incomparable by construction: a store writes the
        // address, a load overwrites the base, write-back updates it. skip those encodings
        // rather than report the harness's own arrangement as a translator bug.
        const rt = insn & 0x1f, rt2 = (insn >>> 10) & 0x1f, rm = (insn >>> 16) & 0x1f;
        if (rt === baseReg) { skipped++; continue; }
        if (g.name === 'ldst-pair' && rt2 === baseReg) { skipped++; continue; }
        if (g.index) {
          if (rm === baseReg) { skipped++; continue; }
          // the index can be scaled by up to 16 (the 128-bit form), so keep it small enough
          // that base + (index << 4) stays inside the scratch buffer
          if (rm < NREG) st.x[rm] = BigInt(rnd() % 8);
        }
      }
      const r = await ref.run(insn, st, baseReg);
      if (r.trapped) { skipped++; continue; }
      if (zigDecodeKind(insn) === 0) {
        const k = `${g.name}:${(insn >>> 10) & 0x3f}:${(insn >>> 22) & 0x3ff}`;
        const m = missing.get(k);
        if (m) m.count++; else missing.set(k, { insn, count: 1 });
        continue;
      }

      pending.push({ g, insn, st, baseReg, ref: r.state });
      if (pending.length >= BATCH) await drain();
    }
  }
  await drain();

  async function drain(): Promise<void> {
    if (!pending.length) return;
    const batch = pending; pending = [];
    let results: Array<{ refused: boolean; state: CpuState }>;
    if (device) {
      const recs = batch.map((c) => { const b = Buffer.alloc(REC_IN); b.writeUInt32LE(c.insn >>> 0, 0); b.writeInt32LE(c.baseReg, 4); pack(b, 8, c.st); return b; });
      results = await device.run(recs);
    } else {
      results = batch.map((c) => {
        try { const state = runTranslated(c.insn, c.st, c.baseReg); return { refused: lastExitPc === BigInt(PC), state }; }
        catch { return { refused: true, state: c.ref }; }
      });
    }
    for (let bi = 0; bi < batch.length; bi++) {
      const { g, insn, baseReg } = batch[bi];
      const r = { state: batch[bi].ref };
      const mine = results[bi].state;
      void baseReg;
      if (results[bi].refused) {
        const k = `${g.name}:emit:${(insn >>> 10) & 0x3f}:${(insn >>> 22) & 0x3ff}`;
        const m = missing.get(k);
        if (m) m.count++; else missing.set(k, { insn, count: 1 });
        continue;
      }

      const diffs: string[] = [];
      for (let k = 0; k < NREG; k++) {
        // x18 is the platform register on apple silicon: the OS may write it at any point, so
        // whatever it holds afterwards says nothing about the instruction under test.
        if (k === 18) continue;
        if (r.state.x[k] !== mine.x[k]) diffs.push(`x${k}: ref 0x${r.state.x[k].toString(16)} got 0x${mine.x[k].toString(16)}`);
      }
      const refN = (r.state.nzcv >> 28n) & 0xfn, myN = (mine.nzcv >> 28n) & 0xfn;
      if (refN !== myN) diffs.push(`nzcv: ref 0x${refN.toString(16)} got 0x${myN.toString(16)}`);
      for (let k = 0; k < NVEC; k++) {
        const a = r.state.v[k], b = mine.v[k];
        if (a.some((byte, j) => byte !== b[j])) diffs.push(`v${k}: ref ${hex(a)} got ${hex(b)}`);
      }
      for (let k = 0; k < SCRATCH; k++) {
        if (r.state.scratch[k] !== mine.scratch[k]) {
          diffs.push(`mem[${k - SCRATCH / 2}]: ref 0x${r.state.scratch[k].toString(16)} got 0x${mine.scratch[k].toString(16)}`);
          break;
        }
      }
      checked++;
      if (!diffs.length) continue;

      // known deviation: with FPCR.DN clear (the reset state) ARM propagates the INPUT NaN,
      // quieted, through an fp operation. scalar fp here is carried as f64 bits and computed in
      // javascript, where every NaN collapses to the canonical one, so the payload is lost. no
      // value is wrong, only the bit pattern of a NaN. counted separately so it cannot hide a
      // real divergence behind it.
      if (diffs.length && diffs.every(isNanOnly)) {
        nanPayload++;
        continue;
      }


      const key = `${g.name}:${diffs[0].split(':')[0]}:${(insn >>> 21).toString(16)}`;
      const e = bad.get(key);
      if (e) e.count++;
      else bad.set(key, { insn, detail: diffs.slice(0, 3).join('; '), count: 1 });
    }
  }
  ref.close();

  console.log(`\nchecked ${checked}, skipped ${skipped} (encodings the hardware rejects), ` +
              `${[...missing.values()].reduce((a, b) => a + b.count, 0)} undecoded in ${missing.size} shapes, ` +
              `${nanPayload} nan-payload-only`);
  for (const [k, m] of [...missing.entries()].sort((a, b) => b[1].count - a[1].count))
    console.log(`  undecoded x${m.count}  0x${m.insn.toString(16).padStart(8, '0')}  ${k}`);
  if (!bad.size) { console.log('no divergences'); return; }
  const rows = [...bad.values()].sort((a, b) => b.count - a.count);
  console.log(`${rows.length} distinct divergences:\n`);
  for (const e of rows) console.log(`  x${e.count}  0x${e.insn.toString(16).padStart(8, '0')}  ${e.detail}`);
  process.exitCode = 1;
}

/** whether a difference is only in the bit pattern of a NaN, with both sides agreeing it is one. */
function isNanOnly(d: string): boolean {
  const m = /^v\d+: ref ([0-9a-f]+) got ([0-9a-f]+)$/.exec(d);
  if (!m) return false;
  const le = (h: string, n: number) => Buffer.from(h.slice(0, n * 2), 'hex');
  for (const width of [4, 8]) {
    const a = le(m[1], width), b = le(m[2], width);
    const fa = width === 4 ? a.readFloatLE(0) : a.readDoubleLE(0);
    const fb = width === 4 ? b.readFloatLE(0) : b.readDoubleLE(0);
    if (Number.isNaN(fa) && Number.isNaN(fb)) return true;
  }
  return false;
}

const hex = (a: Uint8Array) => [...a].map((b) => b.toString(16).padStart(2, '0')).join('');

main();
