import { Emulator } from './dispatch.js';
import {
  FIB_CODE, FIB_START_PC, setupFib,
  SUM_CODE, SUM_START_PC, setupSum,
  SUMSQ_CODE, SUMSQ_START_PC, setupSumSq,
} from './benchmark.js';
import { detectRegion } from './block.js';
import { GUEST_MEM_BASE } from './types.js';

const output = document.getElementById('output')!;

function log(msg: string) {
  output.textContent += msg + '\n';
  output.scrollTop = output.scrollHeight;
}

function clear() {
  output.textContent = '';
}

function formatIPS(ips: number): string {
  if (ips >= 1e9) return (ips / 1e9).toFixed(2) + 'G';
  if (ips >= 1e6) return (ips / 1e6).toFixed(1) + 'M';
  if (ips >= 1e3) return (ips / 1e3).toFixed(1) + 'K';
  return ips.toFixed(0);
}

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length & 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Yield to the browser so it can repaint
const yieldUI = () => new Promise<void>(r => setTimeout(r, 0));

interface BenchResult {
  insns: number;
  medianMs: number;
  ips: number;
  batchSize: number;
  runs: number[];
}

function setProgress(btn: HTMLButtonElement, pct: number) {
  if (pct <= 0) {
    btn.style.background = '';
  } else {
    btn.style.background =
      `linear-gradient(to right, #00ff41 ${pct}%, #1a1a1a ${pct}%)`;
    btn.style.color = pct > 50 ? '#0a0a0a' : '#00ff41';
  }
}

async function bench(
  setup: (emu: Emulator) => void,
  code: Uint32Array,
  startPC: bigint,
  btn: HTMLButtonElement,
  samples = 11,
): Promise<BenchResult> {
  const emu = new Emulator();
  emu.loadProgram(code, startPC);

  // Warmup: 2 runs to JIT compile + optimize
  for (let i = 0; i < 2; i++) {
    setup(emu);
    emu.run(startPC);
  }

  // Calibrate: ramp up batch size until we get a measurable time.
  // Safari clamps performance.now() to 1ms, so a single fast run reads as 0.
  let batchSize = 1;
  let insns = 0;
  while (batchSize <= 100_000) {
    const t0 = performance.now();
    for (let i = 0; i < batchSize; i++) {
      setup(emu);
      emu.run(startPC);
    }
    const calMs = performance.now() - t0;
    setup(emu);
    insns = emu.run(startPC).insns;
    if (calMs >= 50) {
      batchSize = Math.ceil(batchSize * (200 / calMs));
      break;
    }
    batchSize *= 4;
  }

  // Timed samples — yield between each so the UI can update
  const times: number[] = [];
  for (let s = 0; s < samples; s++) {
    setProgress(btn, ((s + 1) / samples) * 100);
    await yieldUI();

    const start = performance.now();
    for (let b = 0; b < batchSize; b++) {
      setup(emu);
      emu.run(startPC);
    }
    times.push((performance.now() - start) / batchSize);
  }

  setProgress(btn, 0);
  btn.style.color = '';

  const med = median(times);
  return {
    insns,
    medianMs: med,
    ips: med > 0 ? (insns / med) * 1000 : 0,
    batchSize,
    runs: times,
  };
}

function disassembleRegion(emu: Emulator, pc: bigint) {
  const view = new DataView(emu.memory.buffer);
  const region = detectRegion(view, pc);
  const totalInsns = region.blocks.reduce((sum, b) => sum + b.insns.length, 0);
  log(`Region @ 0x${pc.toString(16)} (${region.blocks.length} blocks, ${totalInsns} insns):`);
  for (const block of region.blocks) {
    const selfLoop = block.isSelfLoop ? ' [self-loop]' : '';
    log(`  Block 0x${block.startPC.toString(16)} (${block.insns.length} insns)${selfLoop}:`);
    for (let i = 0; i < block.insns.length; i++) {
      const addr = Number(block.startPC) + i * 4;
      const raw = view.getUint32(addr + GUEST_MEM_BASE, true);
      log(`    0x${addr.toString(16).padStart(5, '0')}: ${raw.toString(16).padStart(8, '0')}  ${block.insns[i].kind}`);
    }
  }
}

function setButtons(enabled: boolean) {
  for (const id of ['btn-fib', 'btn-fib-large', 'btn-sum', 'btn-sumsq']) {
    (document.getElementById(id) as HTMLButtonElement).disabled = !enabled;
  }
}

(window as any).runFib = async () => {
  const btn = document.getElementById('btn-fib') as HTMLButtonElement;
  setButtons(false);
  clear();
  log('=== Fibonacci(40) ===');
  log('Expected: 102,334,155');
  log('');

  const emu = new Emulator();
  emu.loadProgram(FIB_CODE, FIB_START_PC);
  disassembleRegion(emu, FIB_START_PC);
  log('');

  setupFib(emu.cpu, 40);
  emu.run(FIB_START_PC);
  const x0 = emu.cpu.getReg(0);
  log(`Result: X0 = ${x0}`);
  log(`Correct: ${x0 === 102334155n ? 'YES' : 'NO'}`);
  log('');

  const result = await bench(
    (e) => setupFib(e.cpu, 40),
    FIB_CODE, FIB_START_PC, btn, 51,
  );
  logBenchResult(result);
  setButtons(true);
};

(window as any).runFibLarge = async () => {
  const btn = document.getElementById('btn-fib-large') as HTMLButtonElement;
  setButtons(false);
  clear();
  const N = 10_000_000;
  log(`=== Fibonacci(${N.toLocaleString()}) ===`);
  log('Tests region compilation with multi-block inner loop');
  log('');

  const emu = new Emulator();
  emu.loadProgram(FIB_CODE, FIB_START_PC);
  disassembleRegion(emu, FIB_START_PC);
  log('');

  setupFib(emu.cpu, N);
  const check = emu.run(FIB_START_PC);
  log(`Result: X0 = ${emu.cpu.getReg(0)} (${check.insns.toLocaleString()} insns)`);
  log('');

  const result = await bench(
    (e) => setupFib(e.cpu, N),
    FIB_CODE, FIB_START_PC, btn,
  );
  logBenchResult(result);
  setButtons(true);
};

(window as any).runSum = async () => {
  const btn = document.getElementById('btn-sum') as HTMLButtonElement;
  setButtons(false);
  clear();
  log('=== Sum 1..100,000,000 ===');
  log('Expected: 5,000,000,050,000,000');
  log('');

  const emu = new Emulator();
  emu.loadProgram(SUM_CODE, SUM_START_PC);
  disassembleRegion(emu, SUM_START_PC);
  log('');

  setupSum(emu.cpu, 100_000_000);
  emu.run(SUM_START_PC);
  const x0 = emu.cpu.getReg(0);
  log(`Result: X0 = ${x0}`);
  log(`Correct: ${x0 === 5_000_000_050_000_000n ? 'YES' : 'NO'}`);
  log('');

  const result = await bench(
    (e) => setupSum(e.cpu, 100_000_000),
    SUM_CODE, SUM_START_PC, btn,
  );
  logBenchResult(result);
  setButtons(true);
};

(window as any).runSumSq = async () => {
  const btn = document.getElementById('btn-sumsq') as HTMLButtonElement;
  setButtons(false);
  clear();
  log('=== Sum of Squares 1..10,000,000 ===');
  log('Expected: 1,291,990,006,563,070,912  (wraps i64)');
  log('  uses MADD (multiply-accumulate) per iteration');
  log('');

  const emu = new Emulator();
  emu.loadProgram(SUMSQ_CODE, SUMSQ_START_PC);
  disassembleRegion(emu, SUMSQ_START_PC);
  log('');

  setupSumSq(emu.cpu, 10_000_000);
  emu.run(SUMSQ_START_PC);
  const x0 = emu.cpu.getReg(0);
  log(`Result: X0 = ${x0}`);
  log(`Correct: ${x0 === 1_291_990_006_563_070_912n ? 'YES' : 'NO'}`);
  log('');

  const result = await bench(
    (e) => setupSumSq(e.cpu, 10_000_000),
    SUMSQ_CODE, SUMSQ_START_PC, btn,
  );
  logBenchResult(result);
  setButtons(true);
};

function logBenchResult(result: BenchResult) {
  log(`Instructions: ${result.insns.toLocaleString()}`);
  log(`Median time:  ${result.medianMs.toFixed(3)} ms  (${result.runs.length} samples${result.batchSize > 1 ? `, ${result.batchSize} runs/sample` : ''}, 2 warmup)`);
  log(`Throughput:   ${formatIPS(result.ips)} IPS`);
  log('');
  log(`Per-sample ms: ${result.runs.map(t => t.toFixed(3)).join(', ')}`);
}

log('jitter v0.1.0 — ARM64 dynamic recompiler');
log('Translates ARM64 basic blocks to WASM, JIT-compiled via WebAssembly.Module()');
log('');
log('Click a benchmark to run.');
