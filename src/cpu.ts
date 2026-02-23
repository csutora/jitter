import { regOffset, SP_OFFSET, PC_OFFSET, NZCV_OFFSET } from './types.js';

export class CpuState {
  readonly memory: WebAssembly.Memory;
  readonly view: DataView;

  constructor(memory: WebAssembly.Memory) {
    this.memory = memory;
    this.view = new DataView(memory.buffer);
  }

  getReg(r: number): bigint {
    if (r === 31) return 0n; // XZR reads as zero
    return this.view.getBigUint64(regOffset(r), true);
  }

  setReg(r: number, value: bigint): void {
    if (r === 31) return; // XZR writes are discarded
    this.view.setBigUint64(regOffset(r), value, true);
  }

  getSP(): bigint {
    return this.view.getBigUint64(SP_OFFSET, true);
  }

  setSP(value: bigint): void {
    this.view.setBigUint64(SP_OFFSET, value, true);
  }

  getPC(): bigint {
    return this.view.getBigUint64(PC_OFFSET, true);
  }

  setPC(value: bigint): void {
    this.view.setBigUint64(PC_OFFSET, value, true);
  }

  getNZCV(): bigint {
    return this.view.getBigUint64(NZCV_OFFSET, true);
  }

  setNZCV(value: bigint): void {
    this.view.setBigUint64(NZCV_OFFSET, value, true);
  }

  reset(): void {
    // Zero all registers
    for (let i = 0; i < 31; i++) this.setReg(i, 0n);
    this.setSP(0n);
    this.setPC(0n);
    this.setNZCV(0n);
  }
}
