import { HALT_PC, GUEST_MEM_BASE, ITER_OFFSET } from './types.js';
import type { CompiledRegion } from './types.js';
import { CpuState } from './cpu.js';
import { detectRegion } from './block.js';
import { translateRegion } from './translate.js';

export interface RunResult {
  insns: number;
  elapsedMs: number;
  ips: number;
  exitPC: bigint;
}

export class Emulator {
  readonly memory: WebAssembly.Memory;
  readonly cpu: CpuState;
  private regionCache = new Map<bigint, CompiledRegion>();

  constructor(memoryPages = 256) { // 256 pages = 16MB
    this.memory = new WebAssembly.Memory({ initial: memoryPages });
    this.cpu = new CpuState(this.memory);
  }

  loadProgram(code: Uint32Array, guestPC: bigint): void {
    const view = new DataView(this.memory.buffer);
    const baseOffset = Number(guestPC) + GUEST_MEM_BASE;
    for (let i = 0; i < code.length; i++) {
      view.setUint32(baseOffset + i * 4, code[i], true);
    }
  }

  private getRegion(pc: bigint): CompiledRegion {
    let region = this.regionCache.get(pc);
    if (region) return region;

    const view = new DataView(this.memory.buffer);
    const detected = detectRegion(view, pc);
    region = translateRegion(detected, this.memory);

    // Cache by entry PC and all block PCs in the region
    this.regionCache.set(pc, region);
    return region;
  }

  run(startPC: bigint, maxInsns?: number): RunResult {
    this.cpu.setPC(startPC);
    let pc = startPC;
    let totalInsns = 0;

    const t0 = performance.now();

    const view = new DataView(this.memory.buffer);

    while (pc !== HALT_PC) {
      const region = this.getRegion(pc);
      pc = region.run();

      // All regions write instruction count to ITER_OFFSET
      const insns = Number(view.getBigUint64(ITER_OFFSET, true));
      totalInsns += insns;
      view.setBigUint64(ITER_OFFSET, 0n, true);

      if (maxInsns !== undefined && totalInsns >= maxInsns) break;
    }

    const elapsed = performance.now() - t0;

    return {
      insns: totalInsns,
      elapsedMs: elapsed,
      ips: elapsed > 0 ? (totalInsns / elapsed) * 1000 : 0,
      exitPC: pc,
    };
  }

  reset(): void {
    this.cpu.reset();
    this.regionCache.clear();
  }
}
