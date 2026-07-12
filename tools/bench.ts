// benchmark the TS vs zig translator on the boot-busybox path + a few workloads.
// reports wall time, translation cost, execution cost, module size, and ips
// (instructions per second). run with: tsx tools/bench.ts

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
import { Emulator, IOManager } from '../src/dispatch.js';
import { loadElf, setupStack } from '../src/elf.js';
import { SyscallHandler } from '../src/syscall.js';
import { InMemoryVFS } from '../src/vfs.js';
import { TPIDR_OFFSET, GUEST_MEM_BASE } from '../src/types.js';
import { initZigTranslator } from '../src/zig-translator.js';

interface Workload {
  name: string;
  argv: string[];
  stdin: string;
}

interface BenchResult {
  label: string;
  totalMs: number;
  translateMs: number;
  executeMs: number;
  insns: number;
  blocks: number;
  regions: number;
  totalModuleBytes: number;
}

const ROOT = join(__dirname, '..');

function loadBusybox() {
  return new Uint8Array(readFileSync(join(ROOT, 'assets/busybox-aarch64')));
}

function loadTranslator() {
  return new Uint8Array(readFileSync(join(ROOT, 'engine/zig-out/bin/translator.wasm')));
}

type Mode = 'zig-sb' | 'zig-mb';

function runWorkload(mode: Mode, w: Workload, elfBytes: Uint8Array): BenchResult {
  const emu = new Emulator(1024);
  emu.elfBytes = elfBytes;
  emu.zigSingleBlock = mode === 'zig-sb';

  const vfs = new InMemoryVFS();
  vfs.loadBinary(elfBytes);
  const elfInfo = loadElf(elfBytes, emu.memory);
  const sp = setupStack(emu.memory, elfInfo, w.argv,
    ['HOME=/root', 'TERM=xterm-256color', 'PATH=/bin', 'PWD=/']);
  emu.cpu.setSP(sp);
  new DataView(emu.memory.buffer).setBigUint64(TPIDR_OFFSET, BigInt(0x900000), true);

  const io = new IOManager();
  io.pushStdin(new TextEncoder().encode(w.stdin));

  let stdout = '';
  const syscalls = new SyscallHandler(elfInfo.brkStart, {
    writeStdout: (d: Uint8Array) => { stdout += new TextDecoder().decode(d); },
    writeStderr: (d: Uint8Array) => { stdout += new TextDecoder().decode(d); },
    readStdin: (n: number) => io.readStdin(n),
  }, vfs);
  emu.syscallHandler = (cpu, mem) => syscalls.handle(cpu, mem);
  emu.syscallHandlerInstance = syscalls;

  // hook getRegion to time translation and accumulate module bytes via the region cache.
  // we also wrap each region.run with timing.
  let translateMs = 0;
  let executeMs = 0;
  let regions = 0;
  let totalModuleBytes = 0;

  const origGetRegion = (emu as any).getRegion.bind(emu);
  (emu as any).getRegion = function (pc: bigint) {
    if ((this as any).regionCache.has(pc)) {
      const r = origGetRegion(pc);
      return wrapRun(r);
    }
    const t0 = performance.now();
    const r = origGetRegion(pc);
    const t1 = performance.now();
    translateMs += (t1 - t0);
    regions++;
    return wrapRun(r);
  };

  function wrapRun(r: any) {
    if (r.__bench_wrapped) return r;
    const inner = r.run;
    r.run = function () {
      const t0 = performance.now();
      const ret = inner();
      const t1 = performance.now();
      executeMs += (t1 - t0);
      return ret;
    };
    r.__bench_wrapped = true;
    return r;
  }

  let pc = elfInfo.entry;
  let totalInsns = 0;
  let view = new DataView(emu.memory.buffer);
  const HALT = 0xDEAD0000n;
  const SVC = 0xDEAD0001n;
  let blockCount = 0;

  const tStart = performance.now();
  while (pc !== HALT) {
    blockCount++;
    const region = (emu as any).getRegion(pc);
    pc = region.run();

    view = new DataView(emu.memory.buffer);
    const insns = Number(view.getBigUint64(272, true));
    totalInsns += insns;
    view.setBigUint64(272, 0n, true);

    if (pc === SVC) {
      const svcPC = view.getBigUint64(256, true);
      const result = syscalls.handle(emu.cpu, emu.memory);
      view = new DataView(emu.memory.buffer);
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
      continue;
    }

    if (totalInsns >= 500_000_000 || blockCount >= 5_000_000) break;
  }
  const tEnd = performance.now();

  return {
    label: mode + ': ' + w.name,
    totalMs: tEnd - tStart,
    translateMs,
    executeMs,
    insns: totalInsns,
    blocks: blockCount,
    regions,
    totalModuleBytes,
  };
}

function fmtMs(ms: number) { return ms.toFixed(2).padStart(8); }
function fmtNum(n: number) { return n.toLocaleString().padStart(10); }

async function main() {
  const elfBytes = loadBusybox();
  await initZigTranslator(loadTranslator());

  const workloads: Workload[] = [
    { name: 'boot to prompt', argv: ['sh'], stdin: '' },
    { name: 'echo hello', argv: ['sh'], stdin: 'echo hello\n' },
    { name: 'echo 10x via shell', argv: ['sh'], stdin: '1 2 3 4 5 6 7 8 9 10\n'.split('').join('').replace(/\d+/g, m => 'echo ' + m) + '\n' },
    { name: 'ls / via shell', argv: ['sh'], stdin: 'ls /\n' },
    { name: 'cat /etc/passwd', argv: ['sh'], stdin: 'cat /etc/passwd\n' },
    { name: 'echo | cat (pipe)', argv: ['sh'], stdin: 'echo hello | cat\n' },
    { name: 'shell redirection', argv: ['sh'], stdin: 'echo hi > /tmp/x\ncat /tmp/x\n' },
    { name: 'seq 1 100', argv: ['sh'], stdin: 'seq 1 100\n' },
  ];

  const ITER = 7;

  function bestOf(results: BenchResult[]): BenchResult {
    results.sort((a, b) => a.totalMs - b.totalMs);
    return results[Math.floor(ITER / 2)];
  }

  function runAll(mode: Mode, w: Workload): BenchResult {
    // 1 warmup + ITER measured, median reported
    runWorkload(mode, w, elfBytes);
    const xs: BenchResult[] = [];
    for (let i = 0; i < ITER; i++) xs.push(runWorkload(mode, w, elfBytes));
    return bestOf(xs);
  }

  console.log(`config: 1 warmup + ${ITER} measured iters per cell, median reported\n`);
  console.log('workload                        |  z-sb ms | z-mb ms | mb speedup');
  console.log('--------------------------------+----------+---------+-----------');

  for (const w of workloads) {
    const sb = runAll('zig-sb', w);
    const mb = runAll('zig-mb', w);
    console.log(
      w.name.padEnd(31) + ' |' +
      fmtMs(sb.totalMs) + '  |' +
      fmtMs(mb.totalMs) + ' |   ' +
      (sb.totalMs / mb.totalMs).toFixed(2).padStart(4) + 'x'
    );
  }

  console.log('\n--- boot to prompt details (median of ' + ITER + ') ---');
  const sbDetail = runAll('zig-sb', workloads[0]);
  const mbDetail = runAll('zig-mb', workloads[0]);
  for (const r of [sbDetail, mbDetail]) {
    console.log(`${r.label}`);
    console.log(`  total       ${fmtMs(r.totalMs)} ms`);
    console.log(`  translate   ${fmtMs(r.translateMs)} ms (${(100 * r.translateMs / r.totalMs).toFixed(1)}%)`);
    console.log(`  execute     ${fmtMs(r.executeMs)} ms (${(100 * r.executeMs / r.totalMs).toFixed(1)}%)`);
    console.log(`  overhead    ${fmtMs(r.totalMs - r.translateMs - r.executeMs)} ms (dispatcher + syscalls)`);
    console.log(`  insns       ${fmtNum(r.insns)}`);
    console.log(`  blocks      ${fmtNum(r.blocks)}`);
    console.log(`  regions     ${fmtNum(r.regions)} (unique translated PCs)`);
    console.log(`  ips         ${fmtNum(Math.round(r.insns / (r.totalMs / 1000)))} insns/sec`);
  }

}

main().catch(e => { console.error(e); process.exit(1); });
