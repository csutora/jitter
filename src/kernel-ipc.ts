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
export const PROC_REPARENTED = 8; // +32  — 1 if orphaned + reparented to init (kernel auto-reaps)
export const PROC_TGID = 9;       // +36  — thread-group id (== pid for a normal process/leader)
// +40..+63 reserved

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

// A sparse snapshot of a guest linear memory: only the non-zero 64KB pages travel, keyed by
// page index, plus the logical length so the child grows to match. a fresh child memory is all
// zero, so the omitted (zero) pages are already correct — this turns a ~33MB contiguous fork
// copy into "the few MB the process actually touched".
export interface SparseMemSnapshot {
  pages: Uint32Array;   // page indices present in `data`, in order
  data: ArrayBuffer;    // packed page bodies, pages.length * MEM_PAGE_SIZE bytes
  totalLen: number;     // logical byte length of the source memory region
}

export const MEM_PAGE_SIZE = 65536; // wasm page; also the sparse-snapshot granularity

export type KernelMessage =
  // Worker → Main
  | { type: 'fork'; memSnapshot: SparseMemSnapshot; syscallState: any; resumePC: string }
  | { type: 'exit'; pid: number; code: number; high?: number } // high = threaded touched high-water
  | { type: 'kill'; pid: number; sig: number }
  | { type: 'write'; fd: number; data: Uint8Array }
  | { type: 'pipe_alloc' }
  // first pthread_create: workers can't allocate shared memory on WKWebView/JSC, so the worker asks
  // the kernel (main thread) to create it. 'clone_thread' -> kernel makes the shared memory + reserves
  // the child tid/reg-block and replies 'promote'; the worker copies its state in then 'thread_go'
  // tells the kernel to actually spawn the thread-Worker on that (now-populated) shared memory.
  | { type: 'clone_thread'; curPages: number }
  | { type: 'thread_go'; tid: number; regBaseAddr: number; startPC: string; childStack: number; tls: number; childTidPtr: number; flags: number; parentRegBase: number }
  // subsequent threads: the kernel already holds the process's shared memory (keyed by tgid), so no
  // memory travels; the caller names its thread-group and the kernel reuses that shared memory.
  | { type: 'spawn_thread'; tgid: number; startPC: string; childStack: number; tls: number; childTidPtr: number; flags: number; parentRegBase: number }
  | { type: 'lock_req'; path: string; cmd: number; ltype: number; start: number; len: number }
  | { type: 'lock_cancel' }
  | { type: 'bridge_call'; id: number; method: string; args: Record<string, unknown> }
  | { type: 'jit_compiled'; pc: string; wasmBytes: Uint8Array }
  // Main → Worker
  | { type: 'init'; config: WorkerInitConfig }
  | { type: 'fork_result'; childPid: number }
  | { type: 'pipe_result'; pipeSlot: number }
  | { type: 'bridge_response'; id: number; resp: unknown }
  | { type: 'promote'; sharedMemory: WebAssembly.Memory; tid: number; regBaseAddr: number; staleHigh: number }
  | { type: 'clone_fail' } // the kernel couldn't create the shared memory / slot -> clone returns EAGAIN
  | { type: 'stdin_data' } // notification only; data is in the stdin ring buffer
  | { type: 'signal'; sig: number }
  | { type: 'term_size'; cols: number; rows: number };

export interface WorkerInitConfig {
  pid: number;
  ppid: number;
  pgid: number;
  slot: number;
  kernelSAB: SharedArrayBuffer;
  elfBytes: Uint8Array;
  argv: string[];
  envp: string[];
  // For fork: parent's sparse memory snapshot to restore
  memSnapshot?: SparseMemSnapshot;
  // Serialized syscall state (brk, mmapBase, cwd, fdTable descriptors)
  syscallState?: unknown;
  // For fork: kernel pipe map inherited from parent
  kernelPipesSnapshot?: Array<[number, { pipeIndex: number; mode: 'read' | 'write' }]>;
  // Cached JIT modules from parent (compile once, instantiate per-worker)
  jitModules?: Array<{ pc: string; wasmBytes: ArrayBuffer }>;
  termCols?: number;
  termRows?: number;
  // translator.wasm bytes — when present, the worker uses the zig translator with ts fallback
  zigTranslatorBytes?: ArrayBuffer;
  // pre-compiled translator module — the kernel compiles translator.wasm once and shares it with
  // every worker (structured-cloneable, shared code) so forks skip the per-worker 1.5MB compile.
  zigTranslatorModule?: WebAssembly.Module;
  // For CLONE_THREAD: this worker is a THREAD of an existing process. it imports the shared memory
  // instead of creating its own, runs at its own register-block base, and starts mid-program.
  threadStart?: {
    sharedMemory: WebAssembly.Memory;
    startPC: string;       // where the thread begins (just after the parent's clone)
    childStack: number;    // guest SP for the thread
    tls: number;           // guest TPIDR_EL0 for the thread
    tid: number;           // this thread's tid
    tgid: number;          // thread-group id (== the process pid)
    regBaseAddr: number;   // linear-memory base of this thread's register block
    parentRegBase: number; // copy the parent thread's registers as the starting state
    childTidPtr: number;   // CLONE_CHILD_SETTID / CLONE_CHILD_CLEARTID address (0 if unset)
    flags: number;
  };
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
