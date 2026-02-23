// ARM64 benchmark programs as machine code arrays
// Each program starts at guest address 0x10000

// ── Fibonacci(N) ─────────────────────────────────────────────────────────────
// Setup: X0 = N, X1 = 1, X2 = 0
// Result: X0 = fib(N)
//
// 0x10000: SUB  X0, X0, #1      D1000400
// 0x10004: CBZ  X0, +0x18       B40000C0  → 0x1001C
// 0x10008: ADD  X3, X1, X2      8B020023
// 0x1000C: MOV  X2, X1          AA0103E2  (ORR X2, XZR, X1)
// 0x10010: MOV  X1, X3          AA0303E1  (ORR X1, XZR, X3)
// 0x10014: B    -0x14           17FFFFFB  → 0x10000
// 0x10018: NOP                  D503201F
// 0x1001C: MOV  X0, X1          AA0103E0  (ORR X0, XZR, X1)
// 0x10020: SVC  #0              D4000001

export const FIB_CODE = new Uint32Array([
  0xD1000400, // SUB X0, X0, #1
  0xB40000C0, // CBZ X0, +6 insns (to 0x1001C)
  0x8B020023, // ADD X3, X1, X2
  0xAA0103E2, // ORR X2, XZR, X1 (MOV X2, X1)
  0xAA0303E1, // ORR X1, XZR, X3 (MOV X1, X3)
  0x17FFFFFB, // B -5 insns (to 0x10000)
  0xD503201F, // NOP
  0xAA0103E0, // ORR X0, XZR, X1 (MOV X0, X1)
  0xD4000001, // SVC #0 (halt)
]);

export const FIB_START_PC = 0x10000n;

export function setupFib(cpu: import('./cpu.js').CpuState, n: number): void {
  cpu.reset();
  cpu.setReg(0, BigInt(n));  // N
  cpu.setReg(1, 1n);          // a = 1
  cpu.setReg(2, 0n);          // b = 0
}

// Expected: fib(40) = 102334155

// ── Sum 1..N ─────────────────────────────────────────────────────────────────
// Setup: X0 = 0 (sum), X1 = 1 (counter), X2 = N (limit)
// Result: X0 = sum(1..N)
//
// 0x10000: ADD  X0, X0, X1      8B010000
// 0x10004: ADD  X1, X1, #1      91000421
// 0x10008: CMP  X1, X2          EB02003F  (SUBS XZR, X1, X2)
// 0x1000C: B.LE -0xC            54FFFFAD  → 0x10000
// 0x10010: SVC  #0              D4000001

export const SUM_CODE = new Uint32Array([
  0x8B010000, // ADD X0, X0, X1
  0x91000421, // ADD X1, X1, #1
  0xEB02003F, // SUBS XZR, X1, X2 (CMP X1, X2)
  0x54FFFFAD, // B.LE -3 insns (to 0x10000)
  0xD4000001, // SVC #0 (halt)
]);

export const SUM_START_PC = 0x10000n;

export function setupSum(cpu: import('./cpu.js').CpuState, n: number): void {
  cpu.reset();
  cpu.setReg(0, 0n);          // sum = 0
  cpu.setReg(1, 1n);          // counter = 1
  cpu.setReg(2, BigInt(n));   // limit = N
}

// Expected: sum(1..10_000_000) = 50_000_005_000_000

// ── Sum of squares 1..N ────────────────────────────────────────────────────
// Setup: X0 = 0 (sum), X1 = 1 (counter), X2 = N (limit)
// Result: X0 = sum(i*i for i in 1..N)
//
// 0x10000: MADD X0, X1, X1, X0   9B010020  (X0 += X1 * X1)
// 0x10004: ADD  X1, X1, #1       91000421
// 0x10008: CMP  X1, X2           EB02003F  (SUBS XZR, X1, X2)
// 0x1000C: B.LE -0xC             54FFFFAD  → 0x10000
// 0x10010: SVC  #0               D4000001

export const SUMSQ_CODE = new Uint32Array([
  0x9B010020, // MADD X0, X1, X1, X0 (sum += i*i)
  0x91000421, // ADD X1, X1, #1
  0xEB02003F, // SUBS XZR, X1, X2 (CMP X1, X2)
  0x54FFFFAD, // B.LE -3 insns (to 0x10000)
  0xD4000001, // SVC #0 (halt)
]);

export const SUMSQ_START_PC = 0x10000n;

export function setupSumSq(cpu: import('./cpu.js').CpuState, n: number): void {
  cpu.reset();
  cpu.setReg(0, 0n);          // sum = 0
  cpu.setReg(1, 1n);          // counter = 1
  cpu.setReg(2, BigInt(n));   // limit = N
}

// Expected: sumsq(1_000_000) = 333_333_833_333_500_000
