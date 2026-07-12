// verify one mb region against sb-clone execution. runs the seq scenario in
// mb mode; each time the target region entry is reached, snapshots memory,
// runs the mb region once, then replays the same span on an sb clone from the
// snapshot and compares full state at mb's exit pc.

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

Math.random = () => 0.4321;

const HALT = 0xDEAD0000n;
const SVC = 0xDEAD0001n;
const TARGET = BigInt(process.argv[2] ?? '0'); // 0 = verify every region

async function main() {
  const elfBytes = new Uint8Array(readFileSync(join(ROOT, 'assets/busybox-aarch64')));
  await initZigTranslator(new Uint8Array(readFileSync(join(ROOT, 'engine/zig-out/bin/translator.wasm'))));

  const emu = new Emulator(1024);
  emu.elfBytes = elfBytes;
  emu.zigSingleBlock = false;

  const vfs = new InMemoryVFS();
  vfs.loadBinary(elfBytes);
  const elfInfo = loadElf(elfBytes, emu.memory);
  const sp = setupStack(emu.memory, elfInfo, ['sh'],
    ['HOME=/root', 'TERM=xterm-256color', 'PATH=/bin', 'PWD=/']);
  emu.cpu.setSP(sp);
  new DataView(emu.memory.buffer).setBigUint64(TPIDR_OFFSET, BigInt(0x900000), true);

  const io = new IOManager();
  io.pushStdin(new TextEncoder().encode(process.env.STDIN ?? 'seq 1 5\n'));
  let stdout = '';
  const syscalls = new SyscallHandler(elfInfo.brkStart, {
    writeStdout: (d: Uint8Array) => { stdout += new TextDecoder().decode(d); },
    writeStderr: (d: Uint8Array) => { stdout += new TextDecoder().decode(d); },
    readStdin: (n: number) => io.readStdin(n),
  }, vfs);
  emu.syscallHandler = (cpu, mem) => syscalls.handle(cpu, mem);
  emu.syscallHandlerInstance = syscalls;

  // reference clone runs zig in single-block mode; the main emu runs multi-block.
  // diffing sb vs mb catches region-boundary/dispatch bugs specific to multi-block
  const clone = new Emulator(1024);
  clone.elfBytes = elfBytes;
  clone.zigSingleBlock = true;

  const memSize = (emu.memory.buffer as ArrayBuffer).byteLength;
  const snapshot = new Uint8Array(memSize);

  let pc = elfInfo.entry;
  let blocks = 0;
  let checks = 0;

  while (pc !== HALT && blocks < 5_000_000) {
    blocks++;

    if (TARGET === 0n || pc === TARGET) {
      checks++;
      const entry = pc;
      snapshot.set(new Uint8Array(emu.memory.buffer));
      if (process.env.SNAPSHOT_OUT && pc === TARGET) {
        const { writeFileSync } = await import('fs');
        writeFileSync(process.env.SNAPSHOT_OUT, snapshot);
        console.log(`snapshot written to ${process.env.SNAPSHOT_OUT} at entry 0x${entry.toString(16)}`);
        return;
      }

      const takeIter = (m: WebAssembly.Memory): bigint => {
        const v = new DataView(m.buffer);
        const n = v.getBigUint64(0x110, true);
        v.setBigUint64(0x110, 0n, true);
        return n;
      };

      takeIter(emu.memory);
      let q: bigint;
      try {
        q = (emu as any).getRegion(pc).run();
      } catch (e: any) {
        console.log(`\nCRASH running region 0x${pc.toString(16)} (check #${checks}): ${e.message ?? e}`);
        const sv = new DataView(snapshot.buffer);
        for (let i = 0; i < 8; i++) console.log(`  X${i} = 0x${sv.getBigUint64(i * 8, true).toString(16)}`);
        console.log(`  SP = 0x${sv.getBigUint64(0xf8, true).toString(16)} nzcv=0x${sv.getUint32(0x108, true).toString(16)}`);
        const { writeFileSync } = await import('fs');
        writeFileSync('/tmp/crash-entry.bin', snapshot);
        console.log('  entry snapshot written to /tmp/crash-entry.bin');
        return;
      }
      let mbCount = takeIter(emu.memory);

      // replay on clone from snapshot, aligning on executed-insn counts.
      // when the clone overshoots (its block spans past mb's budget-cut exit),
      // extend mb by one more region until counts can meet
      new Uint8Array(clone.memory.buffer).set(snapshot);
      takeIter(clone.memory);
      let cpc = pc;
      let cloneCount = 0n;
      const trail: bigint[] = [];
      let reached = false;
      let extensions = 0;
      let misaligned: string | null = null;

      while (extensions < 200) {
        if (cloneCount === mbCount) {
          if (cpc === q) { reached = true; break; }
          misaligned = `same insn count ${cloneCount} but clone pc=0x${cpc.toString(16)} mb pc=0x${q.toString(16)}`;
          break;
        }
        if (cloneCount < mbCount) {
          if (cpc === HALT || cpc === SVC) {
            misaligned = `clone stopped (${cpc === SVC ? 'svc' : 'halt'}) at count ${cloneCount}, mb count ${mbCount}`;
            break;
          }
          const creg = (clone as any).getRegion(cpc);
          cpc = creg.run();
          cloneCount += takeIter(clone.memory);
          trail.push(cpc);
        } else {
          if (q === HALT || q === SVC) {
            misaligned = `mb stopped (${q === SVC ? 'svc' : 'halt'}) at count ${mbCount}, clone count ${cloneCount}`;
            break;
          }
          extensions++;
          q = (emu as any).getRegion(q).run();
          mbCount += takeIter(emu.memory);
        }
      }

      if (!reached) {
        console.log(`\nCHECK #${checks} FAILED (region 0x${entry.toString(16)}, ${extensions} extensions)`);
        console.log(`  ${misaligned ?? 'no alignment within extension budget'}`);
        console.log(`  clone trail head: ${trail.slice(0, 12).map(p => '0x' + p.toString(16)).join(' ')}`);
        console.log(`  clone trail tail: ${trail.slice(-10).map(p => '0x' + p.toString(16)).join(' ')}`);
        const sv = new DataView(snapshot.buffer);
        console.log('  entry regs:');
        for (let i = 0; i < 11; i++) console.log(`    X${i} = 0x${sv.getBigUint64(i * 8, true).toString(16)}`);
        console.log(`    SP = 0x${sv.getBigUint64(0xf8, true).toString(16)}`);
        for (let i = 0; i < 4; i++) {
          console.log(`    V${i} = 0x${sv.getBigUint64(0x200 + i * 16 + 8, true).toString(16)}:${sv.getBigUint64(0x200 + i * 16, true).toString(16)}`);
        }
        return;
      }

      // compare registers + full memory
      const va = new DataView(emu.memory.buffer);
      const vb = new DataView(clone.memory.buffer);
      const diffs: string[] = [];
      for (let i = 0; i < 31; i++) {
        const a = va.getBigUint64(i * 8, true);
        const b = vb.getBigUint64(i * 8, true);
        if (a !== b) diffs.push(`X${i}: mb=0x${a.toString(16)} sb=0x${b.toString(16)}`);
      }
      {
        const a = va.getBigUint64(0xf8, true);
        const b = vb.getBigUint64(0xf8, true);
        if (a !== b) diffs.push(`SP: mb=0x${a.toString(16)} sb=0x${b.toString(16)}`);
      }
      for (let i = 0; i < 32; i++) {
        const alo = va.getBigUint64(0x200 + i * 16, true);
        const ahi = va.getBigUint64(0x200 + i * 16 + 8, true);
        const blo = vb.getBigUint64(0x200 + i * 16, true);
        const bhi = vb.getBigUint64(0x200 + i * 16 + 8, true);
        if (alo !== blo || ahi !== bhi) diffs.push(`V${i}: mb=0x${ahi.toString(16)}:${alo.toString(16)} sb=0x${bhi.toString(16)}:${blo.toString(16)}`);
      }
      const bufA = Buffer.from(emu.memory.buffer as ArrayBuffer, GUEST_MEM_BASE);
      const bufB = Buffer.from(clone.memory.buffer as ArrayBuffer, GUEST_MEM_BASE);
      if (!bufA.equals(bufB)) {
        let off = 0;
        while (off < bufA.length && bufA[off] === bufB[off]) off++;
        diffs.push(`mem[0x${off.toString(16)}]: mb=0x${bufA.readBigUInt64LE(off & ~7).toString(16)} sb=0x${bufB.readBigUInt64LE(off & ~7).toString(16)}`);
      }

      if (diffs.length > 0) {
        const { writeFileSync } = await import('fs');
        writeFileSync('/tmp/diverge-entry.bin', snapshot);
        console.log(`\nDIVERGENT EXECUTION #${checks} of region 0x${entry.toString(16)}, exit 0x${q.toString(16)}`);
        console.log('  entry snapshot written to /tmp/diverge-entry.bin');
        console.log(`  clone path: ${trail.slice(0, 40).map(p => '0x' + p.toString(16)).join(' ')}`);
        for (const d of diffs.slice(0, 15)) console.log('  ' + d);
        // dump entry state snapshot registers for reproduction
        const sv = new DataView(snapshot.buffer);
        console.log('  entry regs:');
        for (let i = 0; i < 8; i++) console.log(`    X${i} = 0x${sv.getBigUint64(i * 8, true).toString(16)}`);
        for (let i = 0; i < 4; i++) {
          console.log(`    V${i} = 0x${sv.getBigUint64(0x200 + i * 16 + 8, true).toString(16)}:${sv.getBigUint64(0x200 + i * 16, true).toString(16)}`);
        }
        return;
      }

      pc = q;
    } else {
      const region = (emu as any).getRegion(pc);
      pc = region.run();
    }

    if (pc === SVC) {
      const view = new DataView(emu.memory.buffer);
      const svcPC = view.getBigUint64(256, true);
      const result = syscalls.handle(emu.cpu, emu.memory);
      if (result.status === 'exit') {
        const restored = emu.processManager.exit(emu.memory, emu.cpu, result.code);
        if (restored) {
          syscalls.lastWaitStatus = result.code;
          syscalls.lastWaitPid = restored.childPid;
          syscalls.restoreState(restored.syscallState);
          pc = restored.resumePC;
          continue;
        }
        break;
      }
      if (result.status === 'blocked_read') break;
      if (result.status === 'fork') {
        emu.processManager.fork(emu.memory, emu.cpu, syscalls.getState(), result.resumePC);
        pc = result.resumePC;
        continue;
      }
      if (result.status === 'execve') {
        const { entry, brkStart } = emu.processManager.execve(
          emu.memory, emu.cpu, elfBytes, result.argv, result.envp);
        syscalls.resetForExec(brkStart);
        pc = entry;
        continue;
      }
      pc = svcPC + 4n;
    }
  }

  console.log(`done: ${checks} executions of the region verified, ${blocks} total blocks`);
  console.log(`stdout: ${JSON.stringify(stdout.slice(-120))}`);
}

main();
