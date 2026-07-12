// real interval timers: setitimer(ITIMER_REAL) must arm a deadline and raise SIGALRM when
// it expires (default action = terminate). this backs busybox `timeout`, which is how the
// in-vivo ltp driver bounds each test.
import { describe, it, expect } from 'vitest';
import { Emulator } from '../src/dispatch.js';
import { SyscallHandler } from '../src/syscall.js';
import { GUEST_MEM_BASE } from '../src/types.js';

const SIGALRM = 14;
const SCRATCH = 0x30000; // guest offset for the itimerval struct

function mk() {
  const emu = new Emulator(64);
  const sys = new SyscallHandler(0x200000n, {
    writeStdout: () => {}, writeStderr: () => {}, readStdin: () => new Uint8Array(),
  });
  return { emu, sys };
}

// setitimer(which, new_value, old_value): itimerval = {interval:{sec,usec}, value:{sec,usec}}
function setitimer(emu: Emulator, sys: SyscallHandler, which: number, intervalMs: number, valueMs: number) {
  const v = new DataView(emu.memory.buffer);
  const b = SCRATCH + GUEST_MEM_BASE;
  v.setBigUint64(b, BigInt(Math.floor(intervalMs / 1000)), true);
  v.setBigUint64(b + 8, BigInt((intervalMs % 1000) * 1000), true);
  v.setBigUint64(b + 16, BigInt(Math.floor(valueMs / 1000)), true);
  v.setBigUint64(b + 24, BigInt((valueMs % 1000) * 1000), true);
  emu.cpu.setReg(8, 103n);
  emu.cpu.setReg(0, BigInt(which));
  emu.cpu.setReg(1, BigInt(SCRATCH));
  emu.cpu.setReg(2, 0n);
  return sys.handle(emu.cpu, emu.memory);
}

describe('interval timers (setitimer / SIGALRM)', () => {
  it('arms ITIMER_REAL and getitimer reports a remaining value', () => {
    const { emu, sys } = mk();
    const r = setitimer(emu, sys, 0, 0, 1000); // one-shot, 1s
    expect((r as any).status).toBe('ok');
    // getitimer(which=0, curr=SCRATCH+64)
    const outPtr = SCRATCH + 64;
    emu.cpu.setReg(8, 102n); emu.cpu.setReg(0, 0n); emu.cpu.setReg(1, BigInt(outPtr));
    sys.handle(emu.cpu, emu.memory);
    const v = new DataView(emu.memory.buffer);
    const remSec = Number(v.getBigUint64(outPtr + 16 + GUEST_MEM_BASE, true));
    const remUsec = Number(v.getBigUint64(outPtr + 24 + GUEST_MEM_BASE, true));
    expect(remSec * 1000 + remUsec / 1000).toBeGreaterThan(0);
    expect(remSec * 1000 + remUsec / 1000).toBeLessThanOrEqual(1000);
  });

  it('does not fire before expiry', () => {
    const { emu, sys } = mk();
    setitimer(emu, sys, 0, 0, 10000); // 10s
    expect(sys.checkTimers()).toBe(false);
    expect(sys.pendingSignals).not.toContain(SIGALRM);
  });

  it('fires SIGALRM once after expiry (one-shot disarms)', async () => {
    const { emu, sys } = mk();
    setitimer(emu, sys, 0, 0, 2); // 2ms one-shot
    const start = Date.now();
    while (!sys.checkTimers() && Date.now() - start < 500) { /* spin until it expires */ }
    expect(sys.pendingSignals).toContain(SIGALRM);
    // one-shot: after firing it is disarmed and never fires again
    sys.pendingSignals.length = 0;
    expect(sys.checkTimers()).toBe(false);
  });

  it('disarms on setitimer with a zero value', () => {
    const { emu, sys } = mk();
    setitimer(emu, sys, 0, 0, 5000);
    setitimer(emu, sys, 0, 0, 0); // disarm
    const start = Date.now();
    while (Date.now() - start < 20) { /* let wall time pass */ }
    expect(sys.checkTimers()).toBe(false);
  });
});
