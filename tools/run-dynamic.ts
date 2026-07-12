// dynamic-loader smoke runner. loads a glibc dynamic executable + its interpreter
// (ld-linux-aarch64.so.1), seeds the vfs with the runtime libs, and runs it through
// the emulator starting at the interpreter entry. prints stdout/stderr and where it
// stops. single-process path (no shared-monitor shadow writes), so it exercises the
// loader logic without the multi-worker memory concerns.

import { readFileSync } from 'fs';
import { Emulator, IOManager } from '../src/dispatch.js';
import { loadProgram } from '../src/elf.js';
import { SyscallHandler } from '../src/syscall.js';
import { InMemoryVFS } from '../src/vfs.js';
import { TPIDR_OFFSET, GUEST_MEM_BASE } from '../src/types.js';
import { initZigTranslator } from '../src/zig-translator.js';

const ROOT = 'tests/dynamic/rootfs';
const dec = (d: Uint8Array) => new TextDecoder().decode(d);

initZigTranslator(new Uint8Array(readFileSync('engine/zig-out/bin/translator.wasm')));

const elfBytes = new Uint8Array(readFileSync(`${ROOT}/hello`));
const emu = new Emulator(2048); // 128MB
emu.elfBytes = elfBytes;

const vfs = new InMemoryVFS();
vfs.writeFileBytes('/lib/ld-linux-aarch64.so.1', new Uint8Array(readFileSync(`${ROOT}/lib/ld-linux-aarch64.so.1`)));
vfs.writeFileBytes('/lib/libc.so.6', new Uint8Array(readFileSync(`${ROOT}/lib/libc.so.6`)));
vfs.writeFileBytes('/hello', elfBytes);

const { entry: startPC, sp, brkStart, interp } = loadProgram(
  emu.memory, elfBytes,
  ['/hello'], ['PATH=/bin:/usr/bin', 'HOME=/root', 'TERM=linux', 'PWD=/'],
  (p) => new Uint8Array(readFileSync(`${ROOT}${p}`)),
);
console.error(`load: interp=${interp} startPC=0x${startPC.toString(16)}`);
emu.cpu.setSP(sp);
new DataView(emu.memory.buffer).setBigUint64(TPIDR_OFFSET, 0x900000n, true);

const io = new IOManager();
let stdout = '', stderr = '';
const syscalls = new SyscallHandler(brkStart, {
  writeStdout: (d: Uint8Array) => { stdout += dec(d); },
  writeStderr: (d: Uint8Array) => { stderr += dec(d); },
  readStdin: (n: number) => io.readStdin(n),
}, vfs);

const HALT = 0xDEAD0000n, SVC = 0xDEAD0001n;
let pc = startPC;
let blocks = 0, totalInsns = 0, exitCode: number | undefined, svcCount = 0;
{
  const v = new DataView(emu.memory.buffer);
  const w = v.getUint32(Number(startPC) + GUEST_MEM_BASE, true);
  console.error(`entry insn @0x${startPC.toString(16)} = 0x${w.toString(16).padStart(8, '0')}`);
}
let minPC = pc, maxPC = pc, selfRepeat = 0;
const BUDGET = 2_000_000;
const ring: bigint[] = [];

try {
  while (pc !== HALT) {
    blocks++;
    const region = (emu as any).getRegion(pc);
    const prev = pc;
    pc = region.run();
    const v = new DataView(emu.memory.buffer);
    totalInsns += Number(v.getBigUint64(272, true));
    v.setBigUint64(272, 0n, true);

    if (pc === SVC) {
      const svcPC = v.getBigUint64(256, true);
      const num = Number(emu.cpu.getReg(8));
      svcCount++;
      if (svcCount <= 40) console.error(`svc #${svcCount} nr=${num} @0x${svcPC.toString(16)}`);
      const result: any = syscalls.handle(emu.cpu, emu.memory);
      if (result.status === 'exit') { exitCode = result.code; break; }
      if (result.status === 'bridge') { console.error(`unhandled bridge syscall ${num} (${result.method}) at pc=0x${(svcPC).toString(16)}`); break; }
      if (result.status === 'blocked_read') { console.error('blocked on stdin read'); break; }
      pc = svcPC + 4n;
      continue;
    }
    if (pc < minPC) minPC = pc;
    if (pc > maxPC) maxPC = pc;
    // detect self-loop onset: same region >5000x in a row; dump lead-up + regs
    if (ring.length > 0 && ring[ring.length - 1] === pc) {
      selfRepeat++;
      if (selfRepeat === 5000) {
        const w = new DataView(emu.memory.buffer).getUint32(Number(pc) + GUEST_MEM_BASE, true);
        console.error(`SELF-LOOP at 0x${pc.toString(16)} insn=0x${w.toString(16).padStart(8,'0')} x30=0x${emu.cpu.getReg(30).toString(16)} x0=0x${emu.cpu.getReg(0).toString(16)} x1=0x${emu.cpu.getReg(1).toString(16)}`);
        console.error('lead-up: ' + ring.slice(-25).map(p => '0x' + p.toString(16)).join(' -> '));
        break;
      }
    } else { selfRepeat = 0; }
    ring.push(pc);
    if (ring.length > 60) ring.shift();
    if (blocks > BUDGET) {
      console.error(`block budget exceeded. pcRange=[0x${minPC.toString(16)},0x${maxPC.toString(16)}] regions=${(emu as any).regionCache?.size} svcs=${svcCount}`);
      console.error('last region-entry PCs (loop cycle):');
      console.error(ring.map(p => '0x' + p.toString(16)).join(' -> '));
      break;
    }
  }
} catch (e: any) {
  console.error(`CRASH at pc=0x${pc.toString(16)} block=${blocks} insns=${totalInsns}: ${e.message}`);
}

// dump the hot region entry PCs + first instructions to localize a spin
try {
  const { decode } = await import('../src/decode.js');
  const cache = (emu as any).regionCache as Map<bigint, any>;
  const v = new DataView(emu.memory.buffer);
  console.error(`--- ${cache.size} hot regions ---`);
  for (const key of cache.keys()) {
    let s = `region @0x${key.toString(16)}: `;
    for (let i = 0; i < 6; i++) {
      const w = v.getUint32(Number(key) + i * 4 + GUEST_MEM_BASE, true);
      s += `${decode(w).kind} `;
    }
    console.error(s);
  }
} catch (e: any) { console.error('disasm failed:', e.message); }

console.error(`--- done: exit=${exitCode} blocks=${blocks} insns=${totalInsns} ---`);
console.error(`STDOUT: ${JSON.stringify(stdout)}`);
console.error(`STDERR: ${JSON.stringify(stderr)}`);
