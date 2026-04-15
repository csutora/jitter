import { Emulator } from '../src/dispatch.js';
import { loadElf, setupStack } from '../src/elf.js';
import { SyscallHandler } from '../src/syscall.js';
import { InMemoryVFS } from '../src/vfs.js';
import { GUEST_MEM_BASE } from '../src/types.js';
import { decode } from '../src/decode.js';
import { readFileSync } from 'fs';

const path = process.argv[2] || 'tests/conformance/bin/ltp/getcwd_getcwd01';
const elfBytes = new Uint8Array(readFileSync(path).buffer);
const emu = new Emulator(1024);
emu.singleBlockMode = true;
emu.elfBytes = elfBytes;
const vfs = new InMemoryVFS();
const elfInfo = loadElf(elfBytes, emu.memory);
const sp = setupStack(emu.memory, elfInfo, ['test'], ['PATH=/bin', 'PWD=/', 'TMPDIR=/tmp']);
emu.cpu.setSP(sp);
new DataView(emu.memory.buffer).setBigUint64(0x118, BigInt(0x900000), true);
const syscalls = new SyscallHandler(elfInfo.brkStart, {
  readStdin: () => null,
  writeStdout: () => {},
  writeStderr: () => {},
}, vfs);
emu.syscallHandler = (cpu, memory) => syscalls.handle(cpu, memory);
emu.syscallHandlerInstance = syscalls;

const HALT = 0xDEAD0000n;
const SVC = 0xDEAD0001n;
let pc = elfInfo.entry;
let insns = 0;
let prevPC = pc;
const unknowns = new Map<string, { raw: number; count: number; pc: bigint }>();

for (let i = 0; i < 500000; i++) {
  const region = (emu as any).getRegion(pc);
  prevPC = pc;
  try { pc = region.run(); } catch { break; }
  const view = new DataView(emu.memory.buffer);
  insns += Number(view.getBigUint64(272, true));
  view.setBigUint64(272, 0n, true);

  if (pc === HALT) {
    // Scan from prevPC to find the unknown instruction
    for (let off = 0; off < 200; off += 4) {
      const addr = Number(prevPC) + off;
      const raw = view.getUint32(addr + GUEST_MEM_BASE, true);
      const d = decode(raw);
      if (d.kind === 'unknown') {
        const hex = '0x' + raw.toString(16).padStart(8, '0');
        if (!unknowns.has(hex)) {
          unknowns.set(hex, { raw, count: 0, pc: BigInt(addr) });
        }
        unknowns.get(hex)!.count++;
        break;
      }
    }
    // Skip by advancing past the block
    pc = prevPC + 4n;
  }

  if (pc === SVC) {
    const svcPC = new DataView(emu.memory.buffer).getBigUint64(256, true);
    const result = syscalls.handle(emu.cpu, emu.memory);
    if (result.status === 'exit') { break; }
    if (result.status === 'fork') {
      emu.processManager.fork(emu.memory, emu.cpu, syscalls.getState(), result.resumePC);
      pc = result.resumePC;
      continue;
    }
    if (result.status === 'execve') {
      const { entry, brkStart } = emu.processManager.execve(emu.memory, emu.cpu, elfBytes, result.argv, result.envp);
      syscalls.resetForExec(brkStart);
      pc = entry;
      continue;
    }
    pc = svcPC + 4n;
  }
}

console.log(`\nRan ${insns} insns. Unknown instructions found: ${unknowns.size}`);
for (const [hex, info] of unknowns) {
  console.log(`  ${hex} at pc=0x${info.pc.toString(16)} (hit ${info.count}x)`);
}
