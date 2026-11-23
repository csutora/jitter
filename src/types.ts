// ── Shift types ──────────────────────────────────────────────────────────────
export type ShiftType = 'lsl' | 'lsr' | 'asr' | 'ror';

// ── Condition codes ──────────────────────────────────────────────────────────
export const enum Cond {
  EQ = 0,  NE = 1,  CS = 2,  CC = 3,
  MI = 4,  PL = 5,  VS = 6,  VC = 7,
  HI = 8,  LS = 9,  GE = 10, LT = 11,
  GT = 12, LE = 13, AL = 14, NV = 15,
}

// ── Decoded instruction (discriminated union) ────────────────────────────────
export type DecodedInsn =
  | { kind: 'add_imm';   sf: 0|1; rd: number; rn: number; imm: number; setFlags: boolean }
  | { kind: 'sub_imm';   sf: 0|1; rd: number; rn: number; imm: number; setFlags: boolean }
  | { kind: 'add_reg';   sf: 0|1; rd: number; rn: number; rm: number; shift: ShiftType; amount: number; setFlags: boolean }
  | { kind: 'sub_reg';   sf: 0|1; rd: number; rn: number; rm: number; shift: ShiftType; amount: number; setFlags: boolean }
  | { kind: 'and_imm';   sf: 0|1; rd: number; rn: number; imm: bigint; setFlags: boolean }
  | { kind: 'orr_imm';   sf: 0|1; rd: number; rn: number; imm: bigint }
  | { kind: 'eor_imm';   sf: 0|1; rd: number; rn: number; imm: bigint }
  | { kind: 'and_reg';   sf: 0|1; rd: number; rn: number; rm: number; shift: ShiftType; amount: number; invert: boolean; setFlags: boolean }
  | { kind: 'orr_reg';   sf: 0|1; rd: number; rn: number; rm: number; shift: ShiftType; amount: number; invert: boolean }
  | { kind: 'eor_reg';   sf: 0|1; rd: number; rn: number; rm: number; shift: ShiftType; amount: number; invert: boolean }
  | { kind: 'movz';      sf: 0|1; rd: number; imm16: number; hw: number }
  | { kind: 'movk';      sf: 0|1; rd: number; imm16: number; hw: number }
  | { kind: 'movn';      sf: 0|1; rd: number; imm16: number; hw: number }
  | { kind: 'sbfm';      sf: 0|1; rd: number; rn: number; immr: number; imms: number }
  | { kind: 'ubfm';      sf: 0|1; rd: number; rn: number; immr: number; imms: number }
  | { kind: 'bfm';       sf: 0|1; rd: number; rn: number; immr: number; imms: number }
  | { kind: 'madd';      sf: 0|1; rd: number; rn: number; rm: number; ra: number }
  | { kind: 'msub';      sf: 0|1; rd: number; rn: number; rm: number; ra: number }
  | { kind: 'smaddl';    rd: number; rn: number; rm: number; ra: number }
  | { kind: 'smsubl';    rd: number; rn: number; rm: number; ra: number }
  | { kind: 'umaddl';    rd: number; rn: number; rm: number; ra: number }
  | { kind: 'umsubl';    rd: number; rn: number; rm: number; ra: number }
  | { kind: 'adc';       sf: 0|1; rd: number; rn: number; rm: number; setFlags: boolean }
  | { kind: 'sbc';       sf: 0|1; rd: number; rn: number; rm: number; setFlags: boolean }
  | { kind: 'sdiv';      sf: 0|1; rd: number; rn: number; rm: number }
  | { kind: 'udiv';      sf: 0|1; rd: number; rn: number; rm: number }
  | { kind: 'smulh';     rd: number; rn: number; rm: number }
  | { kind: 'umulh';     rd: number; rn: number; rm: number }
  | { kind: 'csel';      sf: 0|1; rd: number; rn: number; rm: number; cond: number; variant: 'csel'|'csinc'|'csinv'|'csneg' }
  | { kind: 'ldr_imm';   sf: 0|1; rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'str_imm';   sf: 0|1; rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'ldr_reg';   sf: 0|1; rt: number; rn: number; rm: number; extend: string; shift: number }
  | { kind: 'str_reg';   sf: 0|1; rt: number; rn: number; rm: number; extend: string; shift: number }
  | { kind: 'ldrb_imm';  rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'strb_imm';  rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'ldrh_imm';  rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'strh_imm';  rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'ldrsw_imm'; rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'ldrsh_imm'; sf: 0|1; rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'ldrsb_imm'; sf: 0|1; rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'ldp';       sf: 0|1; rt: number; rt2: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'stp';       sf: 0|1; rt: number; rt2: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'adr';       rd: number; imm: number }
  | { kind: 'adrp';      rd: number; imm: number }
  | { kind: 'b';         offset: number }
  | { kind: 'bl';        offset: number }
  | { kind: 'b_cond';    cond: number; offset: number }
  | { kind: 'cbz';       sf: 0|1; rt: number; offset: number }
  | { kind: 'cbnz';      sf: 0|1; rt: number; offset: number }
  | { kind: 'tbz';       rt: number; bit: number; offset: number }
  | { kind: 'tbnz';      rt: number; bit: number; offset: number }
  | { kind: 'br';        rn: number }
  | { kind: 'blr';       rn: number }
  | { kind: 'ret';       rn: number }
  | { kind: 'svc';       imm16: number }
  | { kind: 'brk';       imm16: number }
  | { kind: 'mrs';       rt: number; sysreg: string }
  | { kind: 'msr';       rt: number; sysreg: string }
  | { kind: 'clz';       sf: 0|1; rd: number; rn: number }
  | { kind: 'cls';       sf: 0|1; rd: number; rn: number }
  | { kind: 'rbit';      sf: 0|1; rd: number; rn: number }
  | { kind: 'rev';       sf: 0|1; rd: number; rn: number }
  | { kind: 'rev16';     sf: 0|1; rd: number; rn: number }
  | { kind: 'rev32';     rd: number; rn: number }
  | { kind: 'extr';      sf: 0|1; rd: number; rn: number; rm: number; imms: number }
  | { kind: 'ccmp';      sf: 0|1; rn: number; rm_imm: number; nzcv: number; cond: number; isImm: boolean }
  | { kind: 'ccmn';      sf: 0|1; rn: number; rm_imm: number; nzcv: number; cond: number; isImm: boolean }
  | { kind: 'ldr_lit';   sf: 0|1; rt: number; offset: number }
  | { kind: 'ldrsw_lit'; rt: number; offset: number }
  | { kind: 'ldrb_reg';  rt: number; rn: number; rm: number; extend: string; shift: number }
  | { kind: 'strb_reg';  rt: number; rn: number; rm: number; extend: string; shift: number }
  | { kind: 'ldrh_reg';  rt: number; rn: number; rm: number; extend: string; shift: number }
  | { kind: 'strh_reg';  rt: number; rn: number; rm: number; extend: string; shift: number }
  | { kind: 'ldrsb_reg'; sf: 0|1; rt: number; rn: number; rm: number; extend: string; shift: number }
  | { kind: 'ldrsh_reg'; sf: 0|1; rt: number; rn: number; rm: number; extend: string; shift: number }
  | { kind: 'ldrsw_reg'; rt: number; rn: number; rm: number; extend: string; shift: number }
  | { kind: 'ldxr';      sf: 0|1; rt: number; rn: number }
  | { kind: 'stxr';      sf: 0|1; rt: number; rn: number; rs: number }
  | { kind: 'add_ext';   sf: 0|1; rd: number; rn: number; rm: number; extend: string; amount: number; setFlags: boolean }
  | { kind: 'sub_ext';   sf: 0|1; rd: number; rn: number; rm: number; extend: string; amount: number; setFlags: boolean }
  // Variable shifts
  | { kind: 'lslv';      sf: 0|1; rd: number; rn: number; rm: number }
  | { kind: 'lsrv';      sf: 0|1; rd: number; rn: number; rm: number }
  | { kind: 'asrv';      sf: 0|1; rd: number; rn: number; rm: number }
  | { kind: 'rorv';      sf: 0|1; rd: number; rn: number; rm: number }
  // SIMD/FP load/store
  | { kind: 'str_simd';  size: number; rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'ldr_simd';  size: number; rt: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'stp_simd';  size: number; rt: number; rt2: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  | { kind: 'ldp_simd';  size: number; rt: number; rt2: number; rn: number; offset: number; mode: 'offset'|'pre'|'post' }
  // SIMD data processing
  | { kind: 'simd_dup_elem'; Q: number; rd: number; rn: number; esize: number; index: number }
  | { kind: 'simd_dup_gpr';  Q: number; rd: number; rn: number; esize: number }
  | { kind: 'simd_umov';     Q: number; rd: number; rn: number; esize: number; index: number }
  | { kind: 'simd_movi';     Q: number; rd: number; imm8: number; cmode: number; op: number }
  | { kind: 'simd_ins_gpr';  rd: number; rn: number; esize: number; index: number }
  | { kind: 'simd_orr_reg';  Q: number; rd: number; rn: number; rm: number }  // ORR/MOV (vector)
  | { kind: 'simd_and_reg';  Q: number; rd: number; rn: number; rm: number }
  | { kind: 'simd_bic_reg';  Q: number; rd: number; rn: number; rm: number }
  | { kind: 'simd_eor_reg';  Q: number; rd: number; rn: number; rm: number }
  | { kind: 'simd_add';      Q: number; rd: number; rn: number; rm: number; size: number }
  | { kind: 'simd_sub';      Q: number; rd: number; rn: number; rm: number; size: number }
  | { kind: 'simd_cmeq_reg'; Q: number; rd: number; rn: number; rm: number; size: number }
  | { kind: 'simd_cmgt_reg'; Q: number; rd: number; rn: number; rm: number; size: number }
  | { kind: 'simd_cmhi_reg'; Q: number; rd: number; rn: number; rm: number; size: number }
  | { kind: 'simd_cmge_reg'; Q: number; rd: number; rn: number; rm: number; size: number }
  | { kind: 'simd_cmhs_reg'; Q: number; rd: number; rn: number; rm: number; size: number }
  | { kind: 'simd_umaxp';    Q: number; rd: number; rn: number; rm: number; size: number }
  | { kind: 'simd_addp';     Q: number; rd: number; rn: number; rm: number; size: number }
  | { kind: 'simd_tbl';      Q: number; rd: number; rn: number; rm: number; len: number; op: number }
  | { kind: 'simd_zip1';     Q: number; rd: number; rn: number; rm: number; size: number }
  | { kind: 'simd_zip2';     Q: number; rd: number; rn: number; rm: number; size: number }
  | { kind: 'simd_uzp1';     Q: number; rd: number; rn: number; rm: number; size: number }
  | { kind: 'simd_uzp2';     Q: number; rd: number; rn: number; rm: number; size: number }
  | { kind: 'simd_trn1';     Q: number; rd: number; rn: number; rm: number; size: number }
  | { kind: 'simd_trn2';     Q: number; rd: number; rn: number; rm: number; size: number }
  | { kind: 'simd_not';      Q: number; rd: number; rn: number }
  | { kind: 'simd_neg';      Q: number; rd: number; rn: number; size: number }
  | { kind: 'simd_abs';      Q: number; rd: number; rn: number; size: number }
  | { kind: 'simd_cnt';      Q: number; rd: number; rn: number }
  | { kind: 'simd_cls_vec';  Q: number; rd: number; rn: number; size: number }
  | { kind: 'simd_clz_vec';  Q: number; rd: number; rn: number; size: number }
  // FMOV between GPR and FP registers
  | { kind: 'fmov_to_fp';    vd: number; rn: number; sf: 0|1 }  // FMOV Sn/Dn, Wm/Xm
  | { kind: 'fmov_from_fp';  rd: number; vn: number; sf: 0|1 }  // FMOV Wn/Xn, Sm/Dm
  | { kind: 'fmov_top_to_fp';   vd: number; rn: number }  // FMOV Vd.D[1], Xn
  | { kind: 'fmov_top_from_fp'; rd: number; vn: number }  // FMOV Xd, Vn.D[1]
  | { kind: 'fmov_fp';       vd: number; vn: number; ftype: number }  // FMOV between FP regs
  // FP <-> integer conversions
  | { kind: 'scvtf_int';     vd: number; rn: number; sf: 0|1; ftype: number }
  | { kind: 'ucvtf_int';     vd: number; rn: number; sf: 0|1; ftype: number }
  | { kind: 'fcvtzs_int';    rd: number; vn: number; sf: 0|1; ftype: number }
  | { kind: 'fcvtzu_int';    rd: number; vn: number; sf: 0|1; ftype: number }
  // FP arithmetic
  | { kind: 'fmul';   vd: number; vn: number; vm: number; ftype: number }
  | { kind: 'fdiv';   vd: number; vn: number; vm: number; ftype: number }
  | { kind: 'fadd';   vd: number; vn: number; vm: number; ftype: number }
  | { kind: 'fsub';   vd: number; vn: number; vm: number; ftype: number }
  // FP misc
  | { kind: 'fabs';   vd: number; vn: number; ftype: number }
  | { kind: 'fneg';   vd: number; vn: number; ftype: number }
  | { kind: 'fcvt';   vd: number; vn: number; ftype: number; destFtype: number }
  // FP immediate
  | { kind: 'fmov_imm';  vd: number; ftype: number; bits: bigint }  // FMOV Dd, #imm
  // FP compare
  | { kind: 'fcmp';   vn: number; vm: number; ftype: number; withZero: boolean }  // FCMP/FCMPE
  // FP fused multiply-add/subtract (3-source)
  | { kind: 'fmadd';  vd: number; vn: number; vm: number; va: number; ftype: number }
  | { kind: 'fmsub';  vd: number; vn: number; vm: number; va: number; ftype: number }
  | { kind: 'fnmadd'; vd: number; vn: number; vm: number; va: number; ftype: number }
  | { kind: 'fnmsub'; vd: number; vn: number; vm: number; va: number; ftype: number }
  // SIMD shift by immediate
  | { kind: 'simd_sshll';  Q: number; rd: number; rn: number; esize: number; shift: number }  // SSHLL/SXTL
  | { kind: 'simd_ushll';  Q: number; rd: number; rn: number; esize: number; shift: number }  // USHLL/UXTL
  // SIMD scalar/vector FP conversions
  | { kind: 'simd_scvtf';  Q: number; rd: number; rn: number; esize: number }  // SCVTF (vector)
  | { kind: 'simd_ucvtf';  Q: number; rd: number; rn: number; esize: number }  // UCVTF (vector)
  | { kind: 'simd_fcvtzs';  Q: number; rd: number; rn: number; esize: number }  // FCVTZS (vector)
  | { kind: 'simd_fcvtzu';  Q: number; rd: number; rn: number; esize: number }  // FCVTZU (vector)
  | { kind: 'nop' }
  | { kind: 'unknown';   raw: number };

// ── Instructions that terminate a basic block ────────────────────────────────
export function isBranch(insn: DecodedInsn): boolean {
  switch (insn.kind) {
    case 'b': case 'bl': case 'b_cond': case 'cbz': case 'cbnz':
    case 'tbz': case 'tbnz': case 'br': case 'blr': case 'ret':
    case 'svc': case 'brk':
      return true;
    default:
      return false;
  }
}

// ── CPU state memory layout (offsets in bytes) ───────────────────────────────
export const REG_OFFSET = 0;          // X0 at 0, X1 at 8, ..., X30 at 240
export const SP_OFFSET = 0xF8;        // 248
export const PC_OFFSET = 0x100;       // 256
export const NZCV_OFFSET = 0x108;     // 264
export const ITER_OFFSET = 0x110;     // 272 — self-loop iteration count
export const TPIDR_OFFSET = 0x118;    // 280 — TLS base pointer (TPIDR_EL0)
export const V_OFFSET = 0x200;        // V0-V31 SIMD registers (32 × 16 bytes = 512 bytes)
export const STATE_SIZE = 0x400;      // 1024 bytes reserved for CPU state (GPR + SIMD)
// guest linear memory starts here; everything BELOW is host-reserved. the low page keeps a guest null
// dereference away from real data, and [PROT_TABLE_BASE, +PROT_TABLE_SIZE) holds the page-protection
// permission table. keeping the table below the guest (rather than above its ceiling) means it is
// always inside the initially-committed memory: it never needs a grow, never moves, and can never be
// clobbered by the guest growing or by mmap's zero-fill. must match cpu.guest_mem_base in the engine.
export const GUEST_MEM_BASE = 0x80000;   // 512 KiB
// 36 KiB, just above the painted flag. it sits as low as it can because everything between here
// and the guest is the permission table, and the table's size is what caps how much address space
// a process may have: one byte per 4 KiB page.
export const PROT_TABLE_BASE = 0x9000;
// has the permission table been painted yet? it has to live in the memory itself rather than in a
// worker variable, because the threads of one process share the memory but not their variables:
// a second thread with its own idea of "not painted yet" would repaint from its own view and wipe
// what the first one recorded.
export const PROT_PAINTED_FLAG = 0x8000; // 32 KiB — host-reserved, below the table
// the lock word the engine uses for 128-bit pair atomics (stxp/casp) in a threaded process. wasm
// has no 128-bit atomic, so the pair takes this for the few instructions it spans. it is here
// rather than in the cpu state block because the state block is per-thread and the point is to
// exclude the other threads of the same process. must match cpu.pair_lock_offset in the engine.
export const PAIR_LOCK_OFFSET = 0x8008;

// ── threaded-process shared-memory layout ───────────────────────────────────
// a threaded (pthread-using) process needs a SHARED WebAssembly.Memory, and a shared memory reserves
// its declared `maximum` as address space the moment it is created — measured on device: only ~4.5 GiB
// of such reservation exists for the whole WebContent process, regardless of how little is committed
// (3 x 1.5 GiB with initial:1 page already exhausts it). at the old 1.5 GiB per threaded process, two
// live groups plus the parked spare sat exactly at that edge, so a third allocation either threw
// "Out of memory" (pthread_create -> EAGAIN) or got the whole content process jetsam-killed, which
// looks like the app spontaneously rebooting. sizing the reservation to 512 MiB buys ~8 concurrent
// threaded processes with real headroom. these constants must agree between kernel and worker, hence
// living here. layout: guest [GUEST_MEM_BASE, THREAD_REGS_BASE), the 64 per-thread register blocks at
// THREAD_REGS_BASE, and the atomics shadow at THREADED_SHADOW_BASE (guest_size/512).
export interface ThreadLayout {
  /** guest address ceiling; the 64 per-thread register blocks start here */
  regsBase: number;
  /** atomics shadow, 4 MiB above the register blocks; sized guest_size/512 */
  shadowBase: number;
  /** the thread group's shared vma table, just past the shadow. the address space belongs to the
   *  PROCESS, so its map of mappings has to live where every thread of it can see the same one. */
  vmaBase: number;
  /** the thread group's shared signal dispositions, just past the vma table. sigaction is a
   *  PROCESS-wide setting -- a handler installed by one thread is the handler every thread of the
   *  group runs -- so it belongs beside the address space rather than in each thread's own copy. */
  sigBase: number;
  /** total wasm pages, through the end of the shadow — this is the memory's `maximum`, i.e. the
   *  address-space reservation, so it is what limits how many threaded processes can coexist */
  pages: number;
}

// tiers, smallest first. the kernel picks one per threaded process from its footprint at promote time
// and from how much of the reservation budget is left (pickThreadTier), so the common small process
// costs 256 MiB (~14 concurrent) while a process that is already large still gets room. a shared
// memory's `maximum` is fixed at creation and cannot be raised, so this choice is one-shot: growing
// past the chosen tier means ENOMEM to the guest (what a real kernel does at RLIMIT_AS), not a resize.
// the shared vma table: a 16-byte header (lock, generation, count) then one 12-byte record per
// mapping, sized for vm.max_map_count. rounded up to whole wasm pages so it can simply be appended
// to a tier's reservation.
export const VMA_ENTRY_BYTES = 12;
// header words: lock, generation, count, brk, mmap cursor. the break and the arena cursor belong
// to the address space just as much as the mappings do, so they travel with it.
export const VMA_TABLE_FIRST = 8;   // first entry word, leaving room for the header
export const VMA_TABLE_HEADER = VMA_TABLE_FIRST * 4;
export const VMA_TABLE_ENTRIES = 65536;
export const VMA_TABLE_PAGES = 13; // ceil((16 + 65536*12) / 65536)
export const VMA_TABLE_BYTES = VMA_TABLE_PAGES * 65536;

// the shared signal dispositions: a header (lock, generation, and which signals have been set at
// all) then one 8-word record per signal -- handler, flags, restorer and mask, each 64 bits. two
// kilobytes, rounded up to the one wasm page it is appended to a tier's reservation as.
export const SIG_TABLE_FIRST = 8;   // first entry word, leaving room for the header
export const SIG_TABLE_PAGES = 1;
export const SIG_TABLE_BYTES = SIG_TABLE_PAGES * 65536;

export const THREAD_TIERS: ThreadLayout[] = [
  { regsBase: 0x10000000, shadowBase: 0x10400000, vmaBase: 0x10480000, sigBase: 0x10480000 + VMA_TABLE_BYTES, pages: 0x1048 + VMA_TABLE_PAGES + SIG_TABLE_PAGES }, // 256 MiB guest
  { regsBase: 0x20000000, shadowBase: 0x20400000, vmaBase: 0x20500000, sigBase: 0x20500000 + VMA_TABLE_BYTES, pages: 0x2050 + VMA_TABLE_PAGES + SIG_TABLE_PAGES }, // 512 MiB guest
  { regsBase: 0x40000000, shadowBase: 0x40400000, vmaBase: 0x40600000, sigBase: 0x40600000 + VMA_TABLE_BYTES, pages: 0x4060 + VMA_TABLE_PAGES + SIG_TABLE_PAGES }, // 1 GiB guest
  { regsBase: 0x60000000, shadowBase: 0x60400000, vmaBase: 0x60700000, sigBase: 0x60700000 + VMA_TABLE_BYTES, pages: 0x6070 + VMA_TABLE_PAGES + SIG_TABLE_PAGES }, // 1.5 GiB guest
];
export const DEFAULT_THREAD_LAYOUT = THREAD_TIERS[1];

export function regOffset(r: number): number {
  if (r === 31) return SP_OFFSET;
  return REG_OFFSET + r * 8;
}

export function vregOffset(r: number): number {
  return V_OFFSET + r * 16;
}

// ── WASM value types ─────────────────────────────────────────────────────────
export const enum ValType {
  I32 = 0x7F,
  I64 = 0x7E,
  F32 = 0x7D,
  F64 = 0x7C,
}

// ── WASM opcodes ─────────────────────────────────────────────────────────────
export const enum Op {
  // Control
  Unreachable   = 0x00,
  Nop           = 0x01,
  Block         = 0x02,
  Loop          = 0x03,
  If            = 0x04,
  Else          = 0x05,
  End           = 0x0B,
  Br            = 0x0C,
  BrIf          = 0x0D,
  BrTable       = 0x0E,
  Return        = 0x0F,
  Call          = 0x10,
  CallIndirect  = 0x11,

  // Variables
  LocalGet      = 0x20,
  LocalSet      = 0x21,
  LocalTee      = 0x22,

  // Memory (i32)
  I32Load       = 0x28,
  I64Load       = 0x29,
  I32Load8S     = 0x2C,
  I32Load8U     = 0x2D,
  I32Load16S    = 0x2E,
  I32Load16U    = 0x2F,
  I64Load8S     = 0x30,
  I64Load8U     = 0x31,
  I64Load16S    = 0x32,
  I64Load16U    = 0x33,
  I64Load32S    = 0x34,
  I64Load32U    = 0x35,
  I32Store      = 0x36,
  I64Store      = 0x37,
  I32Store8     = 0x3A,
  I32Store16    = 0x3B,
  I64Store8     = 0x3C,
  I64Store16    = 0x3D,
  I64Store32    = 0x3E,

  // Constants
  I32Const      = 0x41,
  I64Const      = 0x42,

  // i32 comparison
  I32Eqz        = 0x45,
  I32Eq         = 0x46,
  I32Ne         = 0x47,

  // i64 comparison
  I64Eqz        = 0x50,
  I64Eq         = 0x51,
  I64Ne         = 0x52,
  I64LtS        = 0x53,
  I64LtU        = 0x54,
  I64GtS        = 0x55,
  I64GtU        = 0x56,
  I64LeS        = 0x57,
  I64LeU        = 0x58,
  I64GeS        = 0x59,
  I64GeU        = 0x5A,

  // i32 arithmetic
  I32Add        = 0x6A,
  I32Sub        = 0x6B,

  // i64 unary
  I64Clz        = 0x79,
  I64Ctz        = 0x7A,
  I64Popcnt     = 0x7B,

  // i64 arithmetic
  I64Add        = 0x7C,
  I64Sub        = 0x7D,
  I64Mul        = 0x7E,
  I64DivS       = 0x7F,
  I64DivU       = 0x80,
  I64RemS       = 0x81,
  I64RemU       = 0x82,
  I64And        = 0x83,
  I64Or         = 0x84,
  I64Xor        = 0x85,
  I64Shl        = 0x86,
  I64ShrS       = 0x87,
  I64ShrU       = 0x88,
  I64Rotl       = 0x89,
  I64Rotr       = 0x8A,

  // Conversions
  I32WrapI64    = 0xA7,
  I64ExtendI32S = 0xAC,
  I64ExtendI32U = 0xAD,
}

// Block types for structured control flow
export const BLOCKTYPE_VOID = 0x40;
export const BLOCKTYPE_I32 = ValType.I32;
export const BLOCKTYPE_I64 = ValType.I64;

// ── Halt sentinel ────────────────────────────────────────────────────────────
export const HALT_PC = 0xDEAD_0000n;
export const SVC_PC  = 0xDEAD_0001n;

// ── Compiled block ───────────────────────────────────────────────────────────
export interface CompiledBlock {
  run: () => bigint;
  startPC: bigint;
  endPC: bigint;
  insnCount: number;
  isSelfLoop: boolean;
}

// ── Compiled region ─────────────────────────────────────────────────────────
export interface CompiledRegion {
  run: () => bigint;       // returns next PC (outside region)
  entryPC: bigint;
  blockPCs: bigint[];      // all block start PCs in this region
}

// ── Compiled modules (for JIT cache sharing across Workers) ─────────────────
// A compiled module holds the WASM bytecode (Module), which can be instantiated
// multiple times with different Memory imports. This is the split between the
// expensive compile step and the cheap instantiate step.

export interface CompiledBlockModule {
  module: WebAssembly.Module;
  startPC: bigint;
  endPC: bigint;
  insnCount: number;
  isSelfLoop: boolean;
}

export interface CompiledRegionModule {
  module: WebAssembly.Module;
  entryPC: bigint;
  blockPCs: bigint[];
}
