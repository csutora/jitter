/**
 * Shared constants and protocol definitions for the multicore kernel.
 *
 * The kernel uses a SharedArrayBuffer ("kernel bus") for all cross-Worker IPC:
 * process table, syscall mailboxes, pipe ring buffers, and stdin buffers.
 *
 * Layout:
 *   0x00000  Process Table     (64 slots × 64 bytes = 4KB)
 *   0x01000  Syscall Mailboxes (64 slots × 128 bytes = 8KB)
 *   0x03000  Pipe Ring Buffers (32 pipes × 16KB = 512KB)
 *   0x83000  Stdin Ring Buffers (8 × 8KB = 64KB)
 *   0x93000  Global State      (4KB)
 *   Total: ~600KB
 */

// ── Kernel SAB Layout ───────────────────────────────────────────────────────

export const KERNEL_SAB_SIZE = 0x194000; // ~1.6MB (added bridge response area)

// Bridge response area: 64 slots × 16KB = 1MB
// Used for passing bridge response data (e.g., network read data) to Workers
export const BRIDGE_RESP_OFFSET = 0x94000;
export const BRIDGE_RESP_SLOT_SIZE = 16384; // 16KB per worker
export const BRIDGE_RESP_SLOTS = 64;

export const PROC_TABLE_OFFSET = 0x00000;
export const PROC_TABLE_SLOTS = 64;
export const PROC_SLOT_SIZE = 64; // bytes per process entry

export const MAILBOX_OFFSET = 0x01000;
export const MAILBOX_SLOTS = 64;
export const MAILBOX_SLOT_SIZE = 128; // bytes per mailbox

export const PIPE_RING_OFFSET = 0x03000;
export const PIPE_RING_COUNT = 32;
export const PIPE_RING_SIZE = 16384; // 16KB per pipe

export const STDIN_RING_OFFSET = 0x83000;
export const STDIN_RING_COUNT = 8;
export const STDIN_RING_SIZE = 8192; // 8KB per stdin buffer

export const GLOBAL_STATE_OFFSET = 0x93000;

// ── Process States ──────────────────────────────────────────────────────────

export const PROC_EMPTY = 0;
export const PROC_RUNNING = 1;
export const PROC_ZOMBIE = 2;

// ── Process Table Entry Layout (64 bytes, Int32Array indexed) ───────────────
// Each field is a 32-bit int. Index = byte offset / 4.

export const PROC_PID = 0;        // +0
export const PROC_PPID = 1;       // +4
export const PROC_PGID = 2;       // +8
export const PROC_STATE = 3;      // +12
export const PROC_EXIT_CODE = 4;  // +16
export const PROC_WAIT_FUTEX = 5; // +20  — Atomics.wait target for parent waiting on child
export const PROC_SIGNAL = 6;     // +24  — pending signal bitmask
export const PROC_STDIN_RING = 7; // +28  — stdin ring buffer index (-1 = none)
// +32..+63 reserved

// ── Signal Bits ─────────────────────────────────────────────────────────────

export const SIG_INT = 1 << 2;  // SIGINT (Ctrl+C)
export const SIG_PIPE = 1 << 13; // SIGPIPE (broken pipe)
export const SIG_CHLD = 1 << 17; // SIGCHLD (child exited)

// ── Syscall Mailbox Layout (128 bytes, Int32Array indexed) ──────────────────
// Used for synchronous Worker↔Main thread communication.
// Worker writes request, posts message to main, then Atomics.wait on MBOX_READY.
// Main thread processes, writes response, Atomics.store+notify on MBOX_READY.

export const MBOX_READY = 0;     // +0  — 0=pending, 1=response ready (Atomics.wait target)
export const MBOX_TYPE = 1;      // +4  — request/response type
export const MBOX_VALUE0 = 2;    // +8  — generic value fields
export const MBOX_VALUE1 = 3;    // +12
export const MBOX_VALUE2 = 4;    // +16
export const MBOX_VALUE3 = 5;    // +20
// +24..+127 — payload area for larger data (e.g., serialized state)

export const MBOX_TYPE_NONE = 0;
export const MBOX_TYPE_FORK_RESULT = 1;
export const MBOX_TYPE_PIPE_ALLOC = 2;
export const MBOX_TYPE_BRIDGE_RESP = 3;

// ── Pipe Ring Buffer Layout ─────────────────────────────────────────────────
// Header: 4 × Int32 (16 bytes), then data ring.

export const PIPE_WRITE_POS = 0;     // +0  (Atomics)
export const PIPE_READ_POS = 1;      // +4  (Atomics)
export const PIPE_WRITER_CLOSED = 2; // +8  (Atomics)
export const PIPE_READER_CLOSED = 3; // +12 (Atomics)
export const PIPE_HEADER_SIZE = 16;
export const PIPE_DATA_SIZE = PIPE_RING_SIZE - PIPE_HEADER_SIZE; // 16368 bytes

// ── Stdin Ring Buffer — same layout as pipe ring ────────────────────────────

export const STDIN_DATA_SIZE = STDIN_RING_SIZE - PIPE_HEADER_SIZE; // 8176 bytes

// ── Global State Layout ─────────────────────────────────────────────────────

export const GLOBAL_NEXT_PID = 0;       // +0  next PID to allocate (Atomics)
export const GLOBAL_NEXT_PIPE = 1;      // +4  next pipe slot to allocate
export const GLOBAL_NEXT_STDIN = 2;     // +8  next stdin ring to allocate
export const GLOBAL_FG_PGID = 3;        // +12 foreground process group ID

// ── Worker Message Types (postMessage protocol) ─────────────────────────────

export type KernelMessage =
  // Worker → Main
  | { type: 'fork'; memoryBuffer: ArrayBuffer; syscallState: any; resumePC: string }
  | { type: 'exit'; pid: number; code: number }
  | { type: 'write'; fd: number; data: Uint8Array }
  | { type: 'pipe_alloc' }
  | { type: 'bridge_call'; id: number; method: string; args: Record<string, unknown> }
  | { type: 'jit_compiled'; pc: string; wasmBytes: Uint8Array }
  // Main → Worker
  | { type: 'init'; config: WorkerInitConfig }
  | { type: 'fork_result'; childPid: number }
  | { type: 'pipe_result'; pipeSlot: number }
  | { type: 'bridge_response'; id: number; resp: unknown }
  | { type: 'stdin_data' } // notification only; data is in the stdin ring buffer
  | { type: 'signal'; sig: number };

export interface WorkerInitConfig {
  pid: number;
  ppid: number;
  pgid: number;
  slot: number;
  kernelSAB: SharedArrayBuffer;
  elfBytes: Uint8Array;
  argv: string[];
  envp: string[];
  // For fork: parent's memory snapshot to restore
  memoryBuffer?: ArrayBuffer;
  // Serialized syscall state (brk, mmapBase, cwd, fdTable descriptors)
  syscallState?: unknown;
  // For fork: kernel pipe map inherited from parent
  kernelPipesSnapshot?: Array<[number, { pipeIndex: number; mode: 'read' | 'write' }]>;
  // Cached JIT modules from parent (compile once, instantiate per-worker)
  jitModules?: Array<{ pc: string; wasmBytes: ArrayBuffer }>;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/** Get the byte offset of a process table slot in the kernel SAB. */
export function procSlotOffset(slot: number): number {
  return PROC_TABLE_OFFSET + slot * PROC_SLOT_SIZE;
}

/** Get the byte offset of a mailbox slot in the kernel SAB. */
export function mailboxSlotOffset(slot: number): number {
  return MAILBOX_OFFSET + slot * MAILBOX_SLOT_SIZE;
}

/** Get the byte offset of a pipe ring buffer in the kernel SAB. */
export function pipeRingOffset(pipeIndex: number): number {
  return PIPE_RING_OFFSET + pipeIndex * PIPE_RING_SIZE;
}

/** Get the byte offset of a stdin ring buffer in the kernel SAB. */
export function stdinRingOffset(stdinIndex: number): number {
  return STDIN_RING_OFFSET + stdinIndex * STDIN_RING_SIZE;
}
