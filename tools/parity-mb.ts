// lockstep parity: zig single-block (reference, known good) vs zig multi-block
// (candidate). after each mb region run, the sb emulator single-steps until its
// pc catches up, then registers are compared. divergence pinpoints the mb bug.
//
// drives a busybox shell with stdin so the fp-heavy seq path gets exercised.

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { Emulator, IOManager } from '../src/dispatch.js';
import { loadElf, setupStack } from '../src/elf.js';
import { SyscallHandler } from '../src/syscall.js';
import { InMemoryVFS } from '../src/vfs.js';
import { TPIDR_OFFSET, GUEST_MEM_BASE } from '../src/types.js';
import { initZigTranslator } from '../src/zig-translator.js';
import { decode } from '../src/decode.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');

// constant Math.random so getrandom/AT_RANDOM/urandom produce identical bytes
// in both emulators regardless of interleaving (the sides call it at the same
// guest points but in different host order, so even a shared prng would diverge)
Math.random = () => 0.4321;

const HALT = 0xDEAD0000n;
const SVC = 0xDEAD0001n;
const STDIN = process.argv[2] ?? 'seq 1 5\n';

interface Side {
  emu: Emulator;
  syscalls: SyscallHandler;
  io: IOManager;
  stdout: string;
  pc: bigint;
}

function setup(singleBlock: boolean, elfBytes: Uint8Array): Side {
  const emu = new Emulator(1024);
  emu.elfBytes = elfBytes;
  emu.zigSingleBlock = singleBlock;

  const vfs = new InMemoryVFS();
  vfs.loadBinary(elfBytes);
  const elfInfo = loadElf(elfBytes, emu.memory);
  const sp = setupStack(emu.memory, elfInfo, ['sh'],
    ['HOME=/root', 'TERM=xterm-256color', 'PATH=/bin', 'PWD=/']);
  emu.cpu.setSP(sp);
  new DataView(emu.memory.buffer).setBigUint64(TPIDR_OFFSET, BigInt(0x900000), true);

  const io = new IOManager();
  io.pushStdin(new TextEncoder().encode(STDIN));

  const side: Side = { emu, syscalls: null as any, io, stdout: '', pc: elfInfo.entry };
  const syscalls = new SyscallHandler(elfInfo.brkStart, {
    writeStdout: (d: Uint8Array) => { side.stdout += new TextDecoder().decode(d); },
    writeStderr: (d: Uint8Array) => { side.stdout += new TextDecoder().decode(d); },
    readStdin: (n: number) => io.readStdin(n),
  }, vfs);
  emu.syscallHandler = (cpu, mem) => syscalls.handle(cpu, mem);
  emu.syscallHandlerInstance = syscalls;
  side.syscalls = syscalls;
  return side;
}

function regs(memory: WebAssembly.Memory): bigint[] {
  const view = new DataView(memory.buffer);
  const out: bigint[] = [];
  for (let i = 0; i < 31; i++) out.push(view.getBigUint64(i * 8, true));
  out.push(view.getBigUint64(0xf8, true)); // sp
  return out;
}

function vregs(memory: WebAssembly.Memory): bigint[] {
  const view = new DataView(memory.buffer);
  const out: bigint[] = [];
  for (let i = 0; i < 64; i++) out.push(view.getBigUint64(0x200 + i * 8, true));
  return out;
}

function diffRegs(a: bigint[], b: bigint[]): string[] {
  const names = [...Array.from({ length: 31 }, (_, i) => `X${i}`), 'SP'];
  const out: string[] = [];
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) out.push(`${names[i]}: sb=0x${a[i].toString(16)} mb=0x${b[i].toString(16)}`);
  }
  return out;
}

function diffVregs(a: bigint[], b: bigint[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < a.length; i += 2) {
    if (a[i] !== b[i] || a[i + 1] !== b[i + 1]) {
      out.push(`V${i / 2}: sb=0x${a[i + 1].toString(16)}:${a[i].toString(16)} mb=0x${b[i + 1].toString(16)}:${b[i].toString(16)}`);
    }
  }
  return out;
}

// advance one region. if the region exits with a syscall, PARK at the SVC
// (do not handle it) and return 'svc'. syscalls are handled by the main loop
// only when both sides are parked at the same svc pc.
type StepResult = 'ok' | 'svc' | 'halt';

function stepPark(s: Side): StepResult {
  const region = (s.emu as any).getRegion(s.pc);
  const next = region.run();
  if (next === SVC) return 'svc'; // s.pc stays at region entry; svcPC in memory
  s.pc = next;
  return next === HALT ? 'halt' : 'ok';
}

function svcPCOf(s: Side): bigint {
  return new DataView(s.emu.memory.buffer).getBigUint64(256, true);
}

// handle a parked syscall. both sides call this back-to-back.
function handleSvc(s: Side, elfBytes: Uint8Array): boolean {
  const svcPC = svcPCOf(s);
  const result = s.syscalls.handle(s.emu.cpu, s.emu.memory);
  if (result.status === 'exit') {
    const restored = s.emu.processManager.exit(s.emu.memory, s.emu.cpu, result.code);
    if (restored) {
      s.syscalls.lastWaitStatus = result.code;
      s.syscalls.lastWaitPid = restored.childPid;
      s.syscalls.restoreState(restored.syscallState);
      s.pc = restored.resumePC;
      return true;
    }
    s.pc = HALT;
    return false;
  }
  if (result.status === 'blocked_read') { s.pc = HALT; return false; }
  if (result.status === 'fork') {
    s.emu.processManager.fork(s.emu.memory, s.emu.cpu, s.syscalls.getState(), result.resumePC);
    s.pc = result.resumePC;
    return true;
  }
  if (result.status === 'execve') {
    const { entry, brkStart } = s.emu.processManager.execve(
      s.emu.memory, s.emu.cpu, elfBytes, result.argv, result.envp);
    s.syscalls.resetForExec(brkStart);
    s.pc = entry;
    return true;
  }
  s.pc = svcPC + 4n;
  return true;
}

async function main() {
  const elfBytes = new Uint8Array(readFileSync(join(ROOT, 'assets/busybox-aarch64')));
  await initZigTranslator(new Uint8Array(readFileSync(join(ROOT, 'engine/zig-out/bin/translator.wasm'))));

  const sb = setup(true, elfBytes);
  const mb = setup(false, elfBytes);

  // align initial memory (argv strings can land differently)
  new Uint8Array(mb.emu.memory.buffer).set(new Uint8Array(sb.emu.memory.buffer));
  mb.emu.cpu.setSP(sb.emu.cpu.getSP());
  mb.pc = sb.pc;

  // sync points are syscalls: both sides run freely until parked at an SVC,
  // then svc pcs must match and full state is compared. this is robust to the
  // two sides having different region boundaries.
  let svcs = 0;
  const LIMIT = parseInt(process.env.PARITY_LIMIT ?? '1000000', 10);
  const trailMb: bigint[] = [];
  const trailSb: bigint[] = [];

  while (svcs < LIMIT) {
    // run mb to its next svc (or halt)
    let mbState: StepResult = 'ok';
    let guard = 0;
    trailMb.length = 0;
    while (mbState === 'ok' && guard++ < 3_000_000) {
      mbState = stepPark(mb);
      if (trailMb.length < 50000) trailMb.push(mb.pc);
    }
    // run sb to its next svc (or halt)
    let sbState: StepResult = 'ok';
    guard = 0;
    trailSb.length = 0;
    while (sbState === 'ok' && guard++ < 6_000_000) {
      sbState = stepPark(sb);
      if (trailSb.length < 200000) trailSb.push(sb.pc);
    }

    if (mbState !== sbState || (mbState === 'svc' && svcPCOf(mb) !== svcPCOf(sb))) {
      console.log(`\nDIVERGENCE before svc #${svcs}:`);
      console.log(`  sb: ${sbState}${sbState === 'svc' ? ' at 0x' + svcPCOf(sb).toString(16) : ''}`);
      console.log(`  mb: ${mbState}${mbState === 'svc' ? ' at 0x' + svcPCOf(mb).toString(16) : ''}`);
      reportTrailDivergence(trailSb, trailMb, sb);
      return;
    }
    if (mbState === 'halt') break;

    svcs++;
    // compare state while parked at the same svc
    const rd = diffRegs(regs(sb.emu.memory), regs(mb.emu.memory));
    const vd = diffVregs(vregs(sb.emu.memory), vregs(mb.emu.memory));
    let memDiff: string | null = null;
    {
      const bufA = Buffer.from(sb.emu.memory.buffer as ArrayBuffer, GUEST_MEM_BASE);
      const bufB = Buffer.from(mb.emu.memory.buffer as ArrayBuffer, GUEST_MEM_BASE);
      if (!bufA.equals(bufB)) {
        let off = 0;
        while (off < bufA.length && bufA[off] === bufB[off]) off++;
        memDiff = `first mem diff at guest 0x${off.toString(16)}: sb=0x${bufA.readBigUInt64LE(off & ~7).toString(16)} mb=0x${bufB.readBigUInt64LE(off & ~7).toString(16)}`;
      }
    }
    if (rd.length + vd.length > 0 || memDiff) {
      console.log(`\nSTATE DIVERGENCE at svc #${svcs}, svc pc=0x${svcPCOf(sb).toString(16)}`);
      for (const d of rd.slice(0, 12)) console.log('  ' + d);
      for (const d of vd.slice(0, 12)) console.log('  ' + d);
      if (memDiff) console.log('  ' + memDiff);
      reportTrailDivergence(trailSb, trailMb, sb);
      return;
    }

    // handle the syscall on both sides
    const aliveA = handleSvc(sb, elfBytes);
    const aliveB = handleSvc(mb, elfBytes);
    if (!aliveA || !aliveB) break;
    if (svcs % 500 === 0) console.log(`... svc #${svcs}, pc=0x${sb.pc.toString(16)}`);
  }

  console.log(`\nparity ok through ${svcs} syscalls`);
  console.log(`sb stdout: ${JSON.stringify(sb.stdout.slice(-200))}`);
  console.log(`mb stdout: ${JSON.stringify(mb.stdout.slice(-200))}`);
}

// find where the block trails part ways: last common pc in visit order
function reportTrailDivergence(trailSb: bigint[], trailMb: bigint[], sb: Side) {
  const sbSet = new Set(trailSb.map(String));
  let lastCommon: bigint | null = null;
  let firstMbOnly: bigint | null = null;
  for (const p of trailMb) {
    if (sbSet.has(String(p))) lastCommon = p;
    else if (firstMbOnly === null) firstMbOnly = p;
  }
  console.log(`  last mb pc also visited by sb: ${lastCommon !== null ? '0x' + lastCommon.toString(16) : 'none'}`);
  console.log(`  first mb-only pc: ${firstMbOnly !== null ? '0x' + firstMbOnly.toString(16) : 'none'}`);
  console.log(`  mb trail tail: ${trailMb.slice(-12).map(p => '0x' + p.toString(16)).join(' ')}`);
  console.log(`  sb trail tail: ${trailSb.slice(-12).map(p => '0x' + p.toString(16)).join(' ')}`);
  const uniq = [...new Set(trailMb.map(p => '0x' + p.toString(16)))];
  console.log(`  unique mb region entries in span (${uniq.length}): ${uniq.join(',')}`);
  if (firstMbOnly !== null && firstMbOnly !== HALT) dumpContext(sb, sb, firstMbOnly);
}

function dumpContext(sb: Side, mb: Side, entry: bigint) {
  const view = new DataView(sb.emu.memory.buffer);
  console.log('  insns from region entry:');
  for (let i = 0; i < 24; i++) {
    const ipc = Number(entry) + i * 4;
    const raw = view.getUint32(ipc + GUEST_MEM_BASE, true);
    const d = decode(raw);
    console.log(`    0x${ipc.toString(16)}: 0x${raw.toString(16).padStart(8, '0')} ${d.kind}`);
  }
}

main();
