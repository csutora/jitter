import { describe, it, expect } from 'vitest';
import { Emulator } from '../src/dispatch.js';
import { detectRegion } from '../src/block.js';
import { GUEST_MEM_BASE } from '../src/types.js';
import {
  FIB_CODE, FIB_START_PC, setupFib,
  SUM_CODE, SUM_START_PC, setupSum,
  SUMSQ_CODE, SUMSQ_START_PC, setupSumSq,
} from '../src/benchmark.js';

describe('integration', () => {
  it('detects fib region with 3 blocks', () => {
    const memory = new WebAssembly.Memory({ initial: 256 });
    const view = new DataView(memory.buffer);
    const base = Number(FIB_START_PC) + GUEST_MEM_BASE;
    for (let i = 0; i < FIB_CODE.length; i++) {
      view.setUint32(base + i * 4, FIB_CODE[i], true);
    }
    const region = detectRegion(view, FIB_START_PC);
    console.log(`Fib region: ${region.blocks.length} blocks, PCs: ${region.blocks.map(b => '0x' + b.startPC.toString(16)).join(', ')}`);
    expect(region.blocks.length).toBe(3);
    expect(region.blocks[0].startPC).toBe(0x10000n);
    // BFS order: entry block targets are queued in order (CBZ taken=0x1001c, fallthrough=0x10008)
    expect(region.blocks[1].startPC).toBe(0x1001cn);
    expect(region.blocks[2].startPC).toBe(0x10008n);
  });

  it('computes fibonacci(10) = 55', () => {
    const emu = new Emulator();
    emu.loadProgram(FIB_CODE, FIB_START_PC);
    setupFib(emu.cpu, 10);
    const result = emu.run(FIB_START_PC);
    expect(emu.cpu.getReg(0)).toBe(55n);
  });

  it('computes fibonacci(20) = 6765', () => {
    const emu = new Emulator();
    emu.loadProgram(FIB_CODE, FIB_START_PC);
    setupFib(emu.cpu, 20);
    const result = emu.run(FIB_START_PC);
    expect(emu.cpu.getReg(0)).toBe(6765n);
  });

  it('computes fibonacci(40) = 102334155', () => {
    const emu = new Emulator();
    emu.loadProgram(FIB_CODE, FIB_START_PC);
    setupFib(emu.cpu, 40);
    const result = emu.run(FIB_START_PC);
    expect(emu.cpu.getReg(0)).toBe(102334155n);
    console.log(`Fib(40): ${result.insns} insns in ${result.elapsedMs.toFixed(2)}ms = ${(result.ips/1e6).toFixed(1)}M IPS`);
  });

  it('computes sumsq(100) = 338350', () => {
    const emu = new Emulator();
    emu.loadProgram(SUMSQ_CODE, SUMSQ_START_PC);
    setupSumSq(emu.cpu, 100);
    const result = emu.run(SUMSQ_START_PC);
    expect(emu.cpu.getReg(0)).toBe(338350n);
  });

  it('computes sumsq(1_000_000) = 333_333_833_333_500_000', () => {
    const emu = new Emulator();
    emu.loadProgram(SUMSQ_CODE, SUMSQ_START_PC);
    setupSumSq(emu.cpu, 1_000_000);
    const result = emu.run(SUMSQ_START_PC);
    expect(emu.cpu.getReg(0)).toBe(333_333_833_333_500_000n);
  });

  it('computes sumsq(10_000_000) with i64 wrapping', () => {
    const emu = new Emulator();
    emu.loadProgram(SUMSQ_CODE, SUMSQ_START_PC);
    setupSumSq(emu.cpu, 10_000_000);
    const result = emu.run(SUMSQ_START_PC);
    // Overflows i64, wraps to this value
    expect(emu.cpu.getReg(0)).toBe(1_291_990_006_563_070_912n);
  });

  it('computes sum(1..100) = 5050', () => {
    const emu = new Emulator();
    emu.loadProgram(SUM_CODE, SUM_START_PC);
    setupSum(emu.cpu, 100);
    const result = emu.run(SUM_START_PC);
    expect(emu.cpu.getReg(0)).toBe(5050n);
  });

  it('computes sum(1..1000) = 500500', () => {
    const emu = new Emulator();
    emu.loadProgram(SUM_CODE, SUM_START_PC);
    setupSum(emu.cpu, 1000);
    const result = emu.run(SUM_START_PC);
    expect(emu.cpu.getReg(0)).toBe(500500n);
  });

  it('computes sum(1..10_000_000) = 50_000_005_000_000', () => {
    const emu = new Emulator();
    emu.loadProgram(SUM_CODE, SUM_START_PC);
    setupSum(emu.cpu, 10_000_000);
    const result = emu.run(SUM_START_PC);
    expect(emu.cpu.getReg(0)).toBe(50_000_005_000_000n);
    console.log(`Sum benchmark: ${result.insns.toLocaleString()} insns in ${result.elapsedMs.toFixed(1)}ms = ${(result.ips / 1e6).toFixed(1)}M IPS`);
  });
});
