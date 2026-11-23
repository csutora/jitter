/**
 * Main-thread kernel for multicore process management.
 *
 * The Kernel runs on the main JS thread (which also owns the terminal and the Swift bridge).
 * Each emulated Linux process runs in its own Web Worker with JIT-compiled WASM.
 *
 * Responsibilities:
 *   - Spawn/terminate Workers for fork/exec/exit
 *   - Manage the process table in SharedArrayBuffer
 *   - Route stdin from terminal to the foreground process's stdin ring
 *   - Route stdout/stderr from Workers to terminal
 *   - Relay bridge calls (native filesystem) from Workers to Swift
 *   - Deliver signals (Ctrl+C → SIGINT)
 */

import type { KernelMessage, WorkerInitConfig, SparseMemSnapshot } from './kernel-ipc.js';
import { parseShebang, isElf, MAX_INTERP_DEPTH } from './binfmt.js';
import { LockTable, LeaseTable, EOF_END, type LockSpace, type LockScope, type LockKind } from './locks';
import { ArenaAllocator } from './shared-map';
import {
  KERNEL_SAB_SIZE,
  PROC_TABLE_OFFSET, PROC_SLOT_SIZE, PROC_PID, PROC_PPID, PROC_PGID, PROC_SID, PROC_EXECED,
  PROC_SUBREAPER,
  procCredOffset, PROC_CRED_WORDS,
  PROC_UTIME_MS, PROC_STIME_MS, PROC_NICE, PROC_PDEATHSIG,
  PROC_POLICY, PROC_RTPRIO, PROC_SCHED_FLAGS, PROC_AFFINITY, PROC_DL_RUNTIME, PROC_IOPRIO, PROC_RUID,
  PROC_STATE, PROC_EXIT_CODE, PROC_WAIT_FUTEX, PROC_STDIN_RING, sigqPush, sigqReset, type QueuedSignal,
  PROC_SYSCALL,
  PROC_REPARENTED, PROC_TGID, PROC_TRACER, PROC_TRACE_CMD, PT_RESUME_DETACH, PROC_STOP_SIG, PROC_VFORK,
  PROC_MAXRSS_KB, PROC_EMPTY, PROC_RUNNING, PROC_ZOMBIE, PROC_STOPPED,
  PROC_SWAPPED, PROC_SWAP_REQ, PROC_TABLE_SLOTS,
  MAILBOX_OFFSET, MAILBOX_SLOT_SIZE, MBOX_READY, MBOX_VALUE0,
  GLOBAL_STATE_OFFSET, GLOBAL_NEXT_PID, GLOBAL_NEXT_PIPE, GLOBAL_NEXT_STDIN, GLOBAL_FG_PGID,
  GLOBAL_CLK_SEQ, GLOBAL_CLK_OFF_SEC, GLOBAL_CLK_OFF_NSEC,
  GLOBAL_BOOT_MS_LO, GLOBAL_BOOT_MS_HI, GLOBAL_MQ_MSG_MAX, GLOBAL_MQ_MSGSIZE_MAX, GLOBAL_MQ_QUEUES_MAX,
  GLOBAL_INOTIFY_WATCHES,
  bindIoGeneration, bumpIoGeneration, GLOBAL_MSG_NEXT_ID, GLOBAL_SEM_NEXT_ID, GLOBAL_SHM_NEXT_ID,
  STDIN_RING_OFFSET, STDIN_RING_SIZE, STDIN_RING_COUNT,
  PIPE_HEADER_SIZE, PAGE_SIZE, PIPE_DEF_PAGES, PIPE_MIN_PAGES, pipeRingSize,
  PIPE_WRITER_CLOSED, PIPE_READER_CLOSED, PIPE_ASYNC_W,
  SHM_ARENA_SIZE, PIPE_PAGES_SOFT, GLOBAL_PIPE_MAX, GLOBAL_MSGMNI,
  BRIDGE_RESP_OFFSET, BRIDGE_RESP_SLOT_SIZE,
  PIPE_RING_COUNT,
  procSlotOffset, mailboxSlotOffset, stdinRingOffset, PROC_BLOCKED,
  futexReleasePid, bindFutexInspector,
} from './kernel-ipc.js';
import { PipeRing } from './pipe-ring.js';
import { THREAD_TIERS, DEFAULT_THREAD_LAYOUT, GUEST_MEM_BASE } from './types.js';
import type { ThreadLayout } from './types.js';
import type { NativeBridge } from './bridge.js';

/** one SysV shared-memory segment, as shmctl(IPC_STAT) describes it. */
// MSGMNB: linux's default per-queue byte capacity for a SysV message queue, and what
// /proc/sys/kernel/msgmnb reports. a queue holding this many bytes makes the next sender wait.
const MSGMNB = 16384;
// MSGMNI: how many message queues may exist at once, and what /proc/sys/kernel/msgmni reports.
const MSGMNI = 32000;
// the fixed-size cookie a SIGEV_THREAD registration leaves with the kernel, and the two reasons
// it comes back: a message arrived, or the queue went away under it.
const NOTIFY_COOKIE_LEN = 32, NOTIFY_WOKENUP = 1, NOTIFY_REMOVED = 2;

// linux's default /proc/sys/fs/inotify/max_queued_events
const INOTIFY_MAX_QUEUED = 16384;

/**
 * whether a caller may use a SysV object with the access bits it asked for.
 *
 * the low nine bits of msgflg/shmflg are a mode, and the kernel checks them the way it checks a
 * file's: owner bits if the uid matches, group bits if the gid does, other bits otherwise. root
 * bypasses the check entirely.
 */
function ipcPermitted(mode: number, ownerUid: number, ownerGid: number,
                      uid: number, gid: number, want: number): boolean {
  if (uid === 0) return true;
  let bits: number;
  if (uid === ownerUid) bits = (mode >> 6) & 7;
  else if (gid === ownerGid) bits = (mode >> 3) & 7;
  else bits = mode & 7;
  // the request is spelled in owner-position bits, so compare it there
  return ((want >> 6) & 7 & ~bits) === 0;
}

// ── POSIX message queues ────────────────────────────────────────────────────
// named queues in a namespace of their own, addressed by a descriptor once opened. two things
// separate them from the SysV queues: ordering is by PRIORITY (highest first, oldest first within
// a priority), and a queue can announce the arrival of its first message to one registered
// process, which is what mq_notify buys over polling.
const MQ_QUEUES_MAX = 256;              // fs.mqueue.queues_max
const MQ_MSG_MAX = 10;                  // fs.mqueue.msg_max: the default when mq_open gets no attr
const MQ_MSGSIZE_MAX = 8192;            // fs.mqueue.msgsize_max
const MQ_HARD_MSGMAX = 65536;
const MQ_HARD_MSGSIZEMAX = 16 * 1024 * 1024;

interface PosixMq {
  id: number;
  name: string;
  mode: number; uid: number; gid: number;
  maxmsg: number; msgsize: number;
  msgs: Array<{ prio: number; seq: number; off: number; len: number }>;
  seq: number;                          // insertion counter: keeps equal priorities in fifo order
  unlinked: boolean;
  // which processes hold the queue open. an unlinked queue lives until this empties, exactly as
  // an unlinked file lives until its last descriptor closes. membership rather than a count,
  // because the worker only reports the open that first gave a process a hold and the close that
  // took its last one away -- which is what makes dup and a second mq_open of the same queue
  // come out right without the kernel having to model descriptors it cannot see.
  holders: Set<number>;
  notify: { pid: number; tgid: number; signo: number; ring?: number; cookie?: Uint8Array } | null;
}

/** how many attachments a segment has in total: nattch is not stored, it is counted. */
function shmNattch(seg: SysvSeg): number {
  let n = 0;
  for (const c of seg.attachers.values()) n += c;
  return n;
}

interface SysvSeg {
  id: number; key: number; size: number; mode: number; uid: number; gid: number;
  cpid: number; lpid: number; removed: boolean; locked?: boolean;
  // who has it attached, and how many times each. nattch is the sum: keeping it per process is
  // what lets a fork inherit its parent's attachments and a death give them all back.
  attachers: Map<number, number>;
  atime: number; dtime: number; ctime: number;
}

/**
 * a SysV message queue. the messages themselves live in the shared arena, one block each, so a
 * receiver in a different process reads the bytes straight out of shared memory rather than having
 * them copied through the kernel's own address space.
 */
interface SysvMsgQueue {
  id: number; key: number; mode: number; uid: number; gid: number;
  cpid: number; lspid: number; lrpid: number; removed: boolean;
  stime: number; rtime: number; ctime: number;
  qbytes: number;                                    // the queue's byte capacity (msg_qbytes)
  msgs: Array<{ mtype: bigint; off: number; len: number }>;
}

/**
 * a SysV semaphore set. the values live here rather than in the shared arena because every
 * operation on them has to be atomic across the whole array -- semop either applies all of its
 * operations or none of them -- and the kernel is the only place with a serial view of that.
 */
interface SysvSemSet {
  id: number; key: number; mode: number; uid: number; gid: number; cuid: number; cgid: number;
  nsems: number;
  vals: Int32Array;   // semval
  pids: Int32Array;   // sempid: the last process to operate on each semaphore
  otime: number; ctime: number;
  // who is parked on this set and what for, which is what GETNCNT and GETZCNT report. a blocked
  // process re-registers on every poll and clears itself when it stops waiting, so a process that
  // dies mid-wait leaves no phantom behind (its exit clears the whole pid anyway).
  waiting: Map<number, { dec: Set<number>; zero: Set<number> }>;
}

// linux's semaphore limits, as /proc/sys/kernel/sem reports them.
// linux's shared-memory limits, as /proc/sys/kernel reports them.
// an ipc id is a sequence number times this plus a slot index, which is how linux makes a stale
// or invented id fail to name anything. counting 1, 2, 3 instead made every small integer a live
// id, and every test that passes a made-up one succeeded where it should have been refused.
const IPCMNI = 32768;
const SHMMAX = 0x7fffffffffff;  // largest segment, in bytes
const SHMMNI = 4096;            // how many segments may exist
const SHMALL = 0x7ffffffffff;   // total shared pages allowed

const SEMMSL = 32000;   // semaphores per set
const SEMMNS = 1024000000; // semaphores system-wide
const SEMOPM = 500;     // operations per semop call
const SEMMNI = 32000;   // sets system-wide
const SEMVMX = 32767;   // the largest value a semaphore may hold
const SEMAEM = SEMVMX;  // the largest adjustment an undo entry may carry

// WORKER POOL. guest processes are real Workers, and destroying one only frees its memory via GC
// — under a no-idle fork storm GC never catches up and dead Workers pile up into a jetsam OOM.
// so we DON'T destroy them: a fixed pool of long-lived Workers is REUSED across processes. exit
// returns a Worker to the idle pool (memory wiped, in place — wasm can't shrink, so reuse beats
// realloc); fork grabs an idle one (or creates up to POOL_MAX). no terminate on the hot path ->
// nothing for GC to chase. bounds:
// how many processes may live at once. a FIXED small number is the wrong shape for this: what a
// process actually costs is its WebAssembly.Memory, and that varies by two orders of magnitude
// between a sleeping `true` and a threaded build. a flat cap of 12 refused a tenth concurrent
// sleeper on a machine with gigabytes free (ltp's pipe13 wants a hundred), while still being no
// protection at all against three fat ones. so: a floor that is always allowed, a hard ceiling
// below the proc table's size, and in between an admission test against the real budget.
const WORKER_FLOOR = 12;    // always allowed, whatever the accounting says (a shell must be able to fork)
// concurrency ceilings, kept a little under the proc table so a fork fails with EAGAIN rather than
// running the table dry. these were 56/60 against a 64-slot table; the table is bigger now, and
// they follow it, because a program that wants a hundred children (ltp's pipe13 does) should get
// them. what actually bounds memory is the reservation budget below, not this count -- a process
// that is merely declared costs nothing until it writes.
const WORKER_CAP = PROC_TABLE_SLOTS - 8;   // hard ceiling on concurrently-live processes
const POOL_MAX = PROC_TABLE_SLOTS - 4;     // hard ceiling on TOTAL Workers (live + idle)
// what a fork is assumed to cost when the parent's committed size is not yet known. a worker now
// reports its size before running a single guest instruction, so this is only the window between
// spawning and that first message.
const FORK_COST_FLOOR = 8 * 1024 * 1024;
// how much wasm memory we allow ourselves to hold. measured on device (see the memory-ceiling task):
// what bounds the WebContent process is TOTAL COMMITTED BYTES, about 3.3-3.8 GiB, and neither the
// per-memory size nor the declared maximum matters -- 135 guest-sized memories coexist happily as
// long as they are not written to. so this budget is about what processes actually touch. keep it
// under the measured wall with headroom for the main thread, so we refuse a fork with EAGAIN (what
// a real kernel does) rather than being jetsam-killed.
const RESERVATION_BUDGET = 3 * 1024 * 1024 * 1024;

// a Worker that grew a big memory keeps it forever (wasm memory never shrinks), so pooling it would
// hold that reservation for the app's lifetime. above this, retire the Worker instead of pooling it.
const POOL_MAX_MEM_BYTES = 64 * 1024 * 1024;

// SHM_ARENA_SIZE / PIPE_PAGES_SOFT live in kernel-ipc.ts: the kernel enforces the budget and /proc
// reports it, and the two have to be the same number. a mapping that does not fit reports ENOMEM,
// which is an honest answer and better than silently losing coherence.

const SIGIO = 29; // aarch64 SIGIO/SIGPOLL: the default I/O-ready signal when F_SETSIG is unset
const POOL_IDLE_KEEP = 3;   // idle Workers to keep warm; excess trimmed during calm (GC has time).
// ...and a ceiling on what those warm Workers may cost between them. a count alone is not a bound:
// three Workers whose last tenants each grew to 60 MiB hold 180 MiB of the process's whole wasm
// allowance while doing nothing. warmth is a latency optimization and must never outbid a process
// that actually wants to run.
const POOL_IDLE_BYTES = 96 * 1024 * 1024;
// how long the pool waits after the last park before trimming by cost. long enough that a fork
// burst reuses its workers instead of rebuilding them; short enough that idle memory comes back.
const IDLE_TRIM_DELAY_MS = 500;
// how long a RUNNING process may say nothing, and not be marked sleeping, before we name it.
// generous: a compute-bound guest is legitimately silent for a long time.
const WEDGE_REPORT_MS = 60_000;
const IDLE_TRIM_MS = 3000;  // how often to trim the idle pool back toward POOL_IDLE_KEEP.

// join a detached run's captured stdout/stderr chunks into one string (see Kernel.runDetached).
function decodeChunks(chunks: Uint8Array[]): string {
  let total = 0;
  for (const c of chunks) total += c.length;
  const all = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) { all.set(c, off); off += c.length; }
  return new TextDecoder().decode(all);
}

// the threaded-process shared-memory layout (and why it is sized the way it is) lives in types.ts,
// shared with the worker so the two can't drift.

export interface KernelConfig {
  /** The ELF binary to boot (busybox) */
  elfBytes: Uint8Array;
  /** URL of the worker script (worker.js) */
  workerURL: string;
  /** Callback for terminal output */
  onOutput: (data: Uint8Array) => void;
  /** Optional native bridge for Swift filesystem */
  bridge?: NativeBridge;
  /** Optional zig translator wasm — when supplied, all workers use zig with ts fallback */
  zigTranslatorBytes?: ArrayBuffer;
}

// a bounded ring of kernel diagnostics, readable from the page as __jitterKlog(). the kernel makes
// decisions no guest can observe -- which tier a threaded process got, why a spawn was refused -- and
// without somewhere to say so, every one of those is a silent guess from the outside.
const KLOG_MAX = 256;
const klogRing: string[] = [];
export function klog(line: string): void {
  klogRing.push(line);
  if (klogRing.length > KLOG_MAX) klogRing.shift();
}
// the process table, as the kernel sees it: who exists, what state they are in, and which syscall
// they were last in. a process stuck in a blocking call says nothing from the guest side, and this
// is the only view that can tell "waiting for something that will come" from "waiting forever".
export function bindProcInspector(procTable: Int32Array): void {
  if (typeof globalThis === 'undefined') return;
  (globalThis as any).__jitterProcs = () => {
    const out: Array<Record<string, number>> = [];
    for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
      const b = procSlotOffset(slot) / 4;
      const state = Atomics.load(procTable, b + PROC_STATE);
      if (state === PROC_EMPTY) continue;
      out.push({
        slot, pid: Atomics.load(procTable, b + PROC_PID), ppid: Atomics.load(procTable, b + PROC_PPID),
        tgid: Atomics.load(procTable, b + PROC_TGID), state,
        blocked: Atomics.load(procTable, b + PROC_BLOCKED),
        syscall: Atomics.load(procTable, b + PROC_SYSCALL),
      });
    }
    return out;
  };
}

if (typeof globalThis !== 'undefined') {
  (globalThis as any).__jitterKlog = (clear?: boolean) => {
    const out = klogRing.slice();
    if (clear) klogRing.length = 0;
    return out;
  };
}

export class Kernel {
  private sab: SharedArrayBuffer;
  private procTable: Int32Array;
  private globalState: Int32Array;
  private workers = new Map<number, Worker>();
  // threaded processes: the kernel (main thread) owns each thread-group's shared memory (only the main
  // thread can allocate shared WebAssembly.Memory on WKWebView/JSC). keyed by tgid, freed when the
  // group's last thread exits. `pendingPromotes` bridges the async first-thread handshake
  // (clone_thread -> promote -> thread_go); `threadCounts` tracks live threads per group for cleanup.
  private sharedMems = new Map<number, WebAssembly.Memory>();
  private threadCounts = new Map<number, number>();
  private threadedLeaders = new Set<number>(); // pids that own a thread-group's shared memory
  // one parked shared memory kept for reuse by the next threaded process (avoids alloc/GC churn).
  // `high` is the previous tenant's touched high-water, so the reuser only has to zero the stale span.
  private spareSharedMem: { mem: WebAssembly.Memory; high: number; layout: ThreadLayout } | null = null;
  // each thread group's chosen layout (tier), so its thread spawns place register blocks identically.
  private threadLayouts = new Map<number, ThreadLayout>();
  private pendingPromotes = new Map<number, { memory: WebAssembly.Memory; tid: number; slot: number; regBaseAddr: number; tgid: number; pgid: number }>();
  private elfBytes: Uint8Array;
  private workerURL: string;
  private onOutput: (data: Uint8Array) => void;
  private bridge?: NativeBridge;
  private zigTranslatorBytes?: ArrayBuffer;
  private zigTranslatorModule?: WebAssembly.Module;

  // idle Workers parked for reuse (see WORKER POOL). live Workers are in `this.workers`.
  /** frozen process images, keyed by pid. these live in the content process for now, which
   *  already buys the thing that binds today: jsc caps TOTAL wasm memory at 4 GiB, and an image
   *  held as an ArrayBuffer is not wasm memory. moving them across the bridge into the app
   *  process (16 GiB with the increased-memory-limit entitlement) is the next tier. */
  /** last time each pid said anything to the kernel. a process parked in a blocking syscall sends
   *  nothing, so an old timestamp is a decent stand-in for "idle" and costs one map write per
   *  message. used only to choose a swap victim. */
  private lastActive = new Map<number, number>();

  /** committed wasm bytes per live process, as each reports its own growth. tracked separately
   *  from reservedBytes(), which counts only the SHARED reservations threaded processes make.
   *  jsc counts both against one 4 GiB total, so the shared figure alone under-counts what is
   *  actually spent -- sixteen single-threaded processes at 256 MiB would hit the wall with the
   *  reservation accounting reading zero. */
  private committed = new Map<number, number>();

  /** automatic swapping under memory pressure. off by default: the manual drivers are enough to
   *  exercise the machinery, and turning reclaim on changes behaviour under exactly the load the
   *  conformance suite applies, so it gets its own verification before it becomes the default. */
  private swapPolicy = false;

  private swapped = new Map<number, {
    ppid: number; pgid: number; sid: number; slot: number; stdinRing: number;
    /** present while the bulk still sits in the content process; undefined once it has been
     *  handed to the app process, where `pages`/`totalLen` are what is left to rebuild it */
    memSnapshot?: SparseMemSnapshot;
    pages?: Uint32Array; totalLen?: number; native?: boolean;
    /** bytes actually stored, which is the sparse payload -- NOT totalLen, which is the
     *  address span the pages are scattered across and is far larger */
    storedBytes?: number;
    syscallState: unknown; kernelPipesSnapshot: unknown;
    shmMaps?: Array<{ addr: number; len: number; off: number; sysv?: number }>; protArmed?: boolean;
  }>();
  private idleWorkers: Worker[] = [];
  private idleTrimTimer: number | null = null;
  // how many Workers this kernel has built, reused and thrown away. a fork storm that outruns gc
  // shows up here as created climbing with reused flat, which is the shape that used to crash.
  workersCreated = 0;
  workersReused = 0;
  workersParked = 0;
  workersRetired = 0;
  /** what each parked Worker's memory costs. a wasm memory never shrinks, so an idle Worker is not
   *  free -- it holds every byte its last tenant grew to, and the budget has to know that. */
  private idleBytes = new Map<Worker, number>();
  /** how far each live process's wasm memory has been GROWN, as opposed to what it holds. only the
   *  pooling decision cares: a Worker that ballooned keeps that buffer for the app's lifetime. */
  private grownBytes = new Map<number, number>();
  /** pids already named as wedged, so one stuck process does not spam every tick. */
  private wedgeReported = new Set<number>();

  // detached runs (`runDetached`, driven by the /ctl/run control route): a process spawned OUTSIDE
  // the shell whose stdout/stderr are captured here instead of going to the terminal. this is what
  // makes a root with no working shell debuggable, and it never touches the tty input path (typing
  // into a busy shell is what drops characters), so it is also the reliable way to drive the guest.
  private detached = new Map<number, { chunks: Uint8Array[]; done: (r: { output: string; code: number }) => void }>();
  // every pid belonging to a detached run's process TREE -> that run's owner pid. children inherit it
  // on fork, so a test that forks (ltp does) has its children's output captured too instead of leaking
  // to the terminal.
  private detachedOwner = new Map<number, number>();
  // one shared, permanently writer-closed stdin ring for all detached runs (see runDetached).
  private detachedStdinRing: number | null = null;
  private detachedRingReady = false;

  // advisory file locking (fcntl record locks + flock). each process is its own Worker, so the
  // authority has to be central; the table itself lives in locks.ts with its own unit tests.
  private locks = new LockTable();
  private leases = new LeaseTable((pid, sig) => this.postSignal(pid, sig));

  // MAP_SHARED between processes. wasm cannot alias one memory's page into another, so a shared
  // mapping's authoritative bytes live in this arena and every process reconciles against it at
  // syscall boundaries (shared-map.ts explains the model). blocks are refcounted: a fork makes the
  // child a co-owner, and the block only goes back when the last holder unmaps, execs or dies.
  private shmArena = new SharedArrayBuffer(SHM_ARENA_SIZE);
  private shmAlloc = new ArenaAllocator(SHM_ARENA_SIZE);

  private shmRefs = new Map<number, { len: number; refs: number; key?: string }>();
  private shmByKey = new Map<string, number>();
  // pid -> block -> how many live mappings that process has of it. a COUNT, not a set: a process
  // can map the same block more than once (two mmaps of one file, or a fork carrying several
  // mappings across), and each of those is a separate reference. holding it as a set made joining
  // an already-held block skip the reference entirely, so unmapping one of two mappings released
  // the block while the other was still being written to -- and the arena handed it straight out
  // as somebody's pipe buffer.
  private shmHeld = new Map<number, Map<number, number>>();
  // arena blocks currently handed out as pipe buffers, mapped to their size in bytes. the size has
  // to be recorded because pipes are no longer all the same size and a block must be released to
  // the arena with exactly the length it was taken with.
  private pipeBlocks = new Map<number, number>();
  // data pages currently charged to pipes. linux keeps this count per user and stops giving out
  // full-size pipes once it passes pipe-user-pages-soft; without such a budget a process that
  // opens pipes until it fails (ltp's pipe06) takes the whole arena and starves MAP_SHARED,
  // which does not fail loudly -- it just makes unrelated things stop working.
  private pipePages = 0;
  // which processes were ever handed an end of each pipe block. a process killed outright never
  // runs its close path, so its ends stay counted and the block is never reusable -- the arena
  // fills with pipes belonging to nobody. this lets the allocator reclaim a block once every
  // process that could possibly hold an end of it is gone. the set is only ever added to, so it
  // can make reclamation LATER than necessary but never earlier, which is the safe direction.
  private pipeHolders = new Map<number, Map<number, { r: number; w: number }>>();
  // O_ASYNC: who asked to be signalled when a pipe end becomes ready, and with which signal.
  // keyed by ring offset, then by the process that asked. this lives kernel-side because the
  // process that makes the pipe READABLE is not the one that wants to hear about it.
  private pipeAsync = new Map<number, Map<number, { sig: number; mode: 'read' | 'write' }>>();
  // syscalls we have refused, as "<nr>/<errno>" -> how many processes hit it. what libc asked for
  // and did not get, which is otherwise invisible when it falls back and succeeds anyway.
  private refusals = new Map<string, number>();
  // SysV shared-memory segments, by id and by key. see the 'sysv_shm' handler.
  private sysvShm = new Map<number, SysvSeg>();
  private sysvShmByKey = new Map<number, number>();
  /**
   * inotify: which paths each watcher is interested in, and the events waiting for it.
   *
   * the registry lives here because a watch is about what OTHER processes do to the filesystem,
   * and this is the only place that sees all of them. a process reports its own mutations as it
   * makes them (see the 'inotify_event' message), and only when the global watch count says
   * somebody is listening -- so a system with no watches pays one atomic load per mutation.
   */
  private inoWatchers = new Map<number, {
    watches: Map<number, { path: string; mask: number; ino: number }>;
    queue: Array<{ wd: number; mask: number; cookie: number; name: string }>;
    nextWd: number;
    overflowed: boolean;
    refs: number;
    // the arena block handed to the last read. the reader frees it when it has copied the events
    // out, but a reader that dies in between would leak it -- so the next read (or the close)
    // reclaims whatever is still outstanding.
    lastBlock?: { off: number; len: number };
  }>();
  private inoNextId = 1;

  // SysV message queues, by id and by key. see the 'sysv_msg' handler.
  private sysvMsg = new Map<number, SysvMsgQueue>();
  // POSIX message queues, by id and by name. see the 'posix_mq' handler.
  // the fork staging block each process holds, by pid. see the 'fork_stage' handler.
  private forkStages = new Map<number, { off: number; len: number }>();

  // fs.shmmax / kernel.shmall: settings, not constants. a program lowers them to find out what
  // happens when a segment does not fit, and reads them back to check the lowering took.
  // linux's own defaults on 64-bit: ULONG_MAX rounded down off a page, for both. effectively "no
  // limit", which is the honest answer when the real bound is the arena and not a number here.
  private shmmax = 0xFFFFFFFFFEFFFFFFn;   // ULONG_MAX - (1 << 24), as linux defines both
  private shmall = 0xFFFFFFFFFEFFFFFFn;
  private posixMq = new Map<number, PosixMq>();
  private posixMqByName = new Map<string, number>();
  /**
   * the wall clock AS THE GUEST SEES IT, in whole seconds.
   *
   * settimeofday and clock_settime move the guest's clock by writing an offset into the shared
   * globals; the host's own clock does not move with it. stamping an IPC object with Date.now()
   * therefore recorded a time the guest could not have produced -- ltp reads back a message
   * queue's stime and finds it hours away from the time() it just called.
   */
  private nowSec(): number {
    for (let i = 0; i < 4; i++) {
      const s1 = Atomics.load(this.globalState, GLOBAL_CLK_SEQ);
      if (s1 & 1) continue;                       // a writer is mid-update; read again
      const sec = Atomics.load(this.globalState, GLOBAL_CLK_OFF_SEC);
      const nsec = Atomics.load(this.globalState, GLOBAL_CLK_OFF_NSEC);
      if (Atomics.load(this.globalState, GLOBAL_CLK_SEQ) === s1) {
        // one floor, at the end. flooring the wall clock and the offset separately threw away the
        // carry between them, so with any sub-second offset in play every timestamp the kernel
        // stamped was a second behind the one the guest reads back with time(2).
        return Math.floor(Date.now() / 1000 + sec + nsec / 1_000_000_000);
      }
    }
    return Math.floor(Date.now() / 1000);
  }

  private nextMqId = 1;
  private sysvMsgByKey = new Map<number, number>();
  private nextMsgId = 0;   // ids advance by IPCMNI; see allocIpcId

  // SysV semaphore sets, by id and by key. see the 'sysv_sem' handler.
  private sysvSem = new Map<number, SysvSemSet>();
  private sysvSemByKey = new Map<number, number>();
  private nextSemId = 0;
  // SEM_UNDO bookkeeping: pid -> semid -> per-semaphore adjustment. an operation tagged SEM_UNDO
  // promises that whatever it did to the semaphore is taken back when the process dies, which is
  // what stops a crashed process from holding a lock forever.
  private semUndo = new Map<number, Map<number, Int32Array>>();
  private nextShmId = 0;
  /** recent traced syscalls, when tracing is on. */
  private traceLog: string[] = [];
  private tracing = false;

  /** turn syscall tracing on or off across every process, present and future. */
  setTrace(on: boolean): void {
    this.tracing = on;
    if (on) this.traceLog.length = 0;
    for (const w of this.workers.values()) w.postMessage({ type: 'set_trace', on });
  }
  getTrace(): string[] { return this.traceLog; }
  isTracing(): boolean { return this.tracing; }

  // AF_UNIX: a bound name maps to the pid listening on it and the connections waiting to be
  // accepted. a connection is a PAIR of arena rings -- one for each direction -- so the two ends
  // are symmetric and the existing ring code carries the bytes.
  // a bound name, who owns it, and the connections waiting on it. each queued connection carries
  // the credentials of the process that made it: SO_PEERCRED asks who is on the other end, and
  // once the rings are handed over there is nothing left to ask.
  // descriptors in flight over a unix socket, keyed by the ring they were sent on.
  private scmQueues = new Map<number, Array<{ pos: number; fd: any }>>();
  // which file each handed-out file handle names. a handle is only valid while the system is up,
  // which is exactly what ESTALE is for.
  private fileHandles = new Map<number, string>();
  // eventfd counters, shared by every process holding a descriptor onto one.
  private eventfds = new Map<number, { count: bigint; semaphore: boolean; holders: Set<number> }>();
  private nextEventfdId = 1;

  // a SIGEV_THREAD notification is the registrant's own cookie handed back on its netlink socket.
  private sendMqCookie(target: { ring?: number; cookie?: Uint8Array }, why: number): void {
    if (target.ring === undefined) return;
    const buf = new Uint8Array(NOTIFY_COOKIE_LEN);
    if (target.cookie) buf.set(target.cookie.subarray(0, NOTIFY_COOKIE_LEN));
    buf[NOTIFY_COOKIE_LEN - 1] = why;
    try { new PipeRing(this.shmArena, target.ring).write(buf); } catch {}
  }

  // the image each process is running, so a write to one can be refused with ETXTBSY. keyed by the
  // host path the filesystem knows, valued by the pids running it.
  private execImages = new Map<string, Set<number>>();
  private setExecImage(pid: number, path: string | null): void {
    for (const [p, pids] of this.execImages) {
      pids.delete(pid);
      if (pids.size === 0) this.execImages.delete(p);
    }
    if (!path) return;
    const set = this.execImages.get(path) ?? new Set<number>();
    set.add(pid);
    this.execImages.set(path, set);
  }

  // netlink sockets, by "<tgid>:<fd>": the ring the kernel writes a notification into.
  private nlRings = new Map<string, number>();

  private unixNames = new Map<string, {
    pid: number; uid: number; gid: number; listening: boolean;
    // a DATAGRAM socket has no connection to accept: it owns a receive ring, and anyone who knows
    // its name writes messages straight into that. `dgram` also tells connect what kind of socket
    // it found, since a datagram may not be sent to a stream's name.
    dgram: boolean; inbox?: number;
    backlog: Array<{ a: number; b: number; pid: number; uid: number; gid: number }>;
    // a name belongs to the SOCKET, and fork gives the child a reference to the same socket. the
    // name goes away when the last holder closes it, not when the first one does.
    holders: Set<number>;
  }>();
  // slots parked in a blocking lock request, so a signal or exit can cut the wait short.
  private lockParked = new Map<number, { pid: number; slot: number }>();
  private lockParkSeq = 0;

  // Stdin ring for the foreground process
  private fgStdinRing: PipeRing | null = null;

  private termCols = 80;
  private termRows = 24;

  constructor(config: KernelConfig) {
    this.elfBytes = config.elfBytes;
    this.workerURL = config.workerURL;
    this.onOutput = config.onOutput;
    this.bridge = config.bridge;
    this.zigTranslatorBytes = config.zigTranslatorBytes;
    // compile translator.wasm ONCE here; every worker gets this shared module (cloned code)
    // instead of re-compiling 1.5MB each — the per-spawn compile spike that made fork storms OOM.
    if (config.zigTranslatorBytes) {
      try {
        this.zigTranslatorModule = new WebAssembly.Module(new Uint8Array(config.zigTranslatorBytes));
      } catch (e) {
        console.error('[kernel] translator.wasm compile failed, falling back to per-worker bytes:', e);
      }
    }

    // Allocate kernel shared memory
    this.sab = new SharedArrayBuffer(KERNEL_SAB_SIZE);
    this.procTable = new Int32Array(this.sab);
    bindProcInspector(this.procTable);
    bindFutexInspector(this.procTable);
    // read a threaded process's guest memory from the page. a deadlock between threads is a
    // question about VALUES -- what does the lock word actually say -- and the kernel holds the
    // group's memory, so it is the one place that can answer it.
    if (typeof globalThis !== 'undefined') {
      (globalThis as any).__jitterPeek = (tgid: number, guestAddr: number, words = 4) => {
        const mem = this.sharedMems.get(tgid);
        if (!mem) return `no shared memory for tgid ${tgid}`;
        const i32 = new Int32Array(mem.buffer);
        const base = (guestAddr + GUEST_MEM_BASE) >> 2;
        const out: number[] = [];
        for (let i = 0; i < words; i++) out.push(Atomics.load(i32, base + i));
        return out;
      };
    }
    this.globalState = new Int32Array(this.sab, GLOBAL_STATE_OFFSET, 64);
    bindIoGeneration(this.globalState);
    // fs.pipe-max-size starts at the default pipe capacity: the largest we can actually deliver
    Atomics.store(this.globalState, GLOBAL_PIPE_MAX, PIPE_DEF_PAGES * PAGE_SIZE);
    Atomics.store(this.globalState, GLOBAL_MSGMNI, MSGMNI);
    // -1 is "no id has been asked for"; checkpoint/restore writes one through /proc/sys/kernel.
    Atomics.store(this.globalState, GLOBAL_MSG_NEXT_ID, -1);
    Atomics.store(this.globalState, GLOBAL_SEM_NEXT_ID, -1);
    Atomics.store(this.globalState, GLOBAL_SHM_NEXT_ID, -1);

    // Initialize global state
    Atomics.store(this.globalState, GLOBAL_NEXT_PID, 2); // PID 1 = init process
    Atomics.store(this.globalState, GLOBAL_NEXT_STDIN, 0);
    Atomics.store(this.globalState, GLOBAL_NEXT_PIPE, 0);
    Atomics.store(this.globalState, GLOBAL_FG_PGID, 1);
    const bootMs = Date.now();
    Atomics.store(this.globalState, GLOBAL_BOOT_MS_LO, bootMs % 0x100000000);
    Atomics.store(this.globalState, GLOBAL_BOOT_MS_HI, Math.floor(bootMs / 0x100000000));
    Atomics.store(this.globalState, GLOBAL_MQ_MSG_MAX, MQ_MSG_MAX);
    Atomics.store(this.globalState, GLOBAL_MQ_MSGSIZE_MAX, MQ_MSGSIZE_MAX);
    Atomics.store(this.globalState, GLOBAL_MQ_QUEUES_MAX, MQ_QUEUES_MAX);

    // trim the idle Worker pool back toward POOL_IDLE_KEEP during calm. under a fork storm the
    // idle pool stays near-empty (workers are reused as fast as they're freed), so this only bites
    // once forking stops — when GC has time and terminating the surplus is cheap. this reclaims
    // the ~POOL_MAX x 32MB the pool would otherwise hold between bursts.
    setInterval(() => { this.trimIdle(); this.warmPool(); this.watchForWedges(); }, IDLE_TRIM_MS);
  }

  /**
   * Notice a process that has stopped talking to us.
   *
   * A wedged process looks exactly like a busy one from outside -- still RUNNING, still holding a
   * Worker -- so a whole ltp sweep once sat behind one for ten minutes with nothing to say why. A
   * process legitimately blocked in a syscall sends nothing either, so silence alone is not a bug;
   * what this can do is make the silence VISIBLE, once, with the pid and how long it has been
   * quiet, so the next wedge names itself instead of having to be bisected.
   */
  private watchForWedges(): void {
    const now = performance.now();
    for (const [pid, t] of this.lastActive) {
      if (!this.workers.has(pid)) continue;
      const quiet = now - t;
      if (quiet < WEDGE_REPORT_MS || this.wedgeReported.has(pid)) continue;
      // a process parked in a blocking call marks itself sleeping; that is normal and not a wedge.
      const slot = this.slotOfPid(pid);
      if (slot >= 0 && Atomics.load(this.procTable, procSlotOffset(slot) / 4 + PROC_BLOCKED) === 1) continue;
      this.wedgeReported.add(pid);
      // on the terminal, not just the console: there is no console in the native app, and a
      // wedge that only names itself where nobody can see it might as well not.
      this.onOutput(new TextEncoder().encode(
        `\r\n[kernel] pid ${pid} has been silent and not sleeping for ${Math.round(quiet / 1000)}s\r\n`));
    }
    for (const pid of [...this.wedgeReported]) if (!this.workers.has(pid)) this.wedgeReported.delete(pid);
  }

  /** park a Worker for reuse, remembering what its memory costs while it waits. */
  private park(w: Worker, bytes: number): void {
    // A WORKER MAY ONLY BE IN THE POOL ONCE, AND ONLY WHEN NOBODY IS RUNNING ON IT.
    //
    // parked twice, it is handed out once and left in the pool -- and the next cost trim
    // terminates it while a process is running on it. that process stops mid-instruction: still
    // RUNNING in the table, still holding its slot, its parent waiting in wait4 for a child that
    // will never speak again. it is a rare shape (a fork storm found it at about one in four
    // thousand) and an unfindable one from the outside, so it is refused here rather than trusted
    // not to happen.
    if (this.idleWorkers.includes(w)) return;
    for (const [livePid, live] of this.workers) if (live === w) { this.workers.delete(livePid); break; }
    this.idleWorkers.push(w);
    this.idleBytes.set(w, bytes);
    this.workersParked++;
    // the COUNT cap is enforced now; the COST cap is not.
    //
    // trimming by cost here is what a fork storm turns into a crash. a child of a 70 MiB parent is
    // a 70 MiB worker, so parking one immediately exceeds the idle byte budget and the trim
    // terminates it -- and the next fork, microseconds later, builds a fresh Worker that grows to
    // the same 70 MiB. terminate() only releases memory once gc gets to it, so the storm allocates
    // at full speed while nothing is actually freed: ltp's epoll-ltp forks about 1500 times in two
    // seconds and takes the whole webcontent process down with it. reusing the parked worker costs
    // exactly the same memory as building its replacement, and costs no allocation at all, so the
    // cost trim belongs after the burst rather than inside it.
    while (this.idleWorkers.length > POOL_IDLE_KEEP) this.dropParked();
    this.scheduleIdleTrim();
  }

  /**
   * trim the idle pool by cost once the system stops forking.
   *
   * every park pushes the deadline out, so a burst is trimmed once at its end rather than once per
   * process. `spare` capacity is what this reclaims, and nothing waits on it.
   */
  private scheduleIdleTrim(): void {
    if (this.idleTrimTimer !== null) clearTimeout(this.idleTrimTimer);
    this.idleTrimTimer = setTimeout(() => {
      this.idleTrimTimer = null;
      this.trimIdle();
    }, IDLE_TRIM_DELAY_MS) as unknown as number;
  }

  /** take a parked Worker, and stop charging the pool for it. */
  private takeParked(): Worker | undefined {
    const w = this.idleWorkers.pop();
    if (w) this.idleBytes.delete(w);
    return w;
  }

  private trimIdle(): void {
    while (this.idleWorkers.length > POOL_IDLE_KEEP) this.dropParked();

    // trim by cost too, fattest first: a warm Worker is worth keeping only while it is cheap.
    let held = 0;
    for (const b of this.idleBytes.values()) held += b;
    while (held > POOL_IDLE_BYTES && this.idleWorkers.length > 0) {
      let fattest = this.idleWorkers[0];
      for (const w of this.idleWorkers) {
        if ((this.idleBytes.get(w) ?? 0) > (this.idleBytes.get(fattest) ?? 0)) fattest = w;
      }
      held -= this.idleBytes.get(fattest) ?? 0;
      this.idleWorkers.splice(this.idleWorkers.indexOf(fattest), 1);
      this.idleBytes.delete(fattest);
      fattest.terminate();
    }
  }

  private dropParked(): void {
    const w = this.idleWorkers.pop();
    if (!w) return;
    this.idleBytes.delete(w);
    this.workersRetired++;
    w.terminate();
  }

  /** keep one Worker parked so the next process launch doesn't pay Worker startup + script parse on
   *  the user's critical path. only ever pre-creates ONE, and only while there is room, so warmth
   *  never competes with a real process for the Worker cap. */
  private warmPool(): void {
    if (this.idleWorkers.length > 0) return;
    if (this.workers.size >= POOL_MAX) return;
    try { this.park(new Worker(this.workerURL), 0); } catch {}
  }

  /** Boot the initial shell process (PID 1). */
  boot(argv: string[] = ['sh'], envp: string[] = ['HOME=/root', 'TERM=xterm-256color', 'PATH=/bin:/usr/bin', 'PWD=/']): void {
    // Allocate stdin ring for the init process
    const stdinIdx = this.allocStdinRing();
    this.fgStdinRing = new PipeRing(this.sab, stdinRingOffset(stdinIdx), STDIN_RING_SIZE);
    this.fgStdinRing.reset(1, 1); // 1 writer (main thread), 1 reader (shell worker)

    this.spawnProcess({
      pid: 1,
      ppid: 0,
      pgid: 1,
      slot: 0,
      stdinRing: stdinIdx,
      elfBytes: this.elfBytes,
      argv,
      envp,
    });
  }

  /** Run one binary as its own process with its output captured, independent of the shell.
   *  resolves with everything it wrote plus its exit code (or a timeout note). used by /ctl/run to
   *  drive + debug a guest that has no working shell (e.g. a distro root whose init won't load). */
  runDetached(path: string, argv: string[], envp: string[], timeoutMs: number): Promise<{ output: string; code: number }> {
    return new Promise((resolve) => {
      const finish = (r: { output: string; code: number }) => { clearTimeout(timer); resolve(r); };
      const pid = this.allocPid();
      const timer = setTimeout(() => {
        const st = this.detached.get(pid);
        if (!st) return;
        this.detached.delete(pid);
        // tear down the whole run: the owner plus anything it forked, so a hung run leaves no
        // orphaned workers or occupied proc slots behind.
        // release the group's shared memory first (same reason as the exit path: a leaked wasm memory
        // holds its address-space reservation for the app's lifetime).
        this.tearDownThreadGroup(pid, 0);
        const doomed = [pid];
        for (const [p, o] of this.detachedOwner) if (o === pid && p !== pid) doomed.push(p);
        for (const p of doomed) {
          this.forgetProcess(p);
          const w = this.workers.get(p);
          if (w) { this.workers.delete(p); w.terminate(); }
          for (let s = 0; s < PROC_TABLE_SLOTS; s++) {
            const b = procSlotOffset(s) / 4;
            if (Atomics.load(this.procTable, b + PROC_PID) === p) {
              Atomics.store(this.procTable, b + PROC_PID, 0);
              Atomics.store(this.procTable, b + PROC_STATE, PROC_EMPTY);
            }
          }
        }
        resolve({ output: decodeChunks(st.chunks) + `\n[timeout after ${timeoutMs}ms]\n`, code: -1 });
      }, timeoutMs);

      // load the target through the same bridge the guest uses, so it resolves against the ACTIVE
      // root; a missing/unreadable binary is reported rather than crashing the spawn.
      // resolve the image the way execve would -- follow `#!` to the interpreter, refuse a non-ELF
      // -- BEFORE spawning, so "that file isn't a program" is reported as such rather than as a
      // worker that threw during load and got reaped, which reads like a jitter bug.
      this.resolveSpawn(path, [path, ...argv]).then((target) => {
        if ('error' in target) { finish({ output: `[ctl/run] ${target.error}\n`, code: target.code }); return; }
        const { bytes: elf } = target;
        const slot = this.findFreeSlot();
        if (slot < 0) { finish({ output: '[ctl/run] no free process slot\n', code: -1 }); return; }
        this.detached.set(pid, { chunks: [], done: finish });
        this.detachedOwner.set(pid, pid);
        // give it a real (already-closed) stdin so fd 0 is a valid descriptor that reads EOF, like
        // running with `< /dev/null`. ONE ring is shared by every detached run — it is permanently
        // writer-closed and never written, so there is nothing to keep separate, and allocating one per
        // run exhausted the 8-slot ring table and then wrote over the kernel's global state.
        const stdinIdx = this.detachedStdinRing ?? (this.detachedStdinRing = this.allocStdinRing());
        if (stdinIdx >= 0 && !this.detachedRingReady) {
          const ring = new PipeRing(this.sab, stdinRingOffset(stdinIdx), STDIN_RING_SIZE);
          ring.reset(1, 1);
          ring.closeWriter();
          this.detachedRingReady = true;
        }
        this.spawnProcess({ pid, ppid: 1, pgid: pid, slot, stdinRing: stdinIdx >= 0 ? stdinIdx : undefined, elfBytes: elf, argv: target.argv, envp });
      }).catch((e) => finish({ output: `[ctl/run] load failed: ${e?.message ?? e}\n`, code: -1 }));
    });
  }

  /** read a guest-visible file via the native bridge (resolves against the ACTIVE root, like the
   *  guest's own opens and like boot's init load). */
  /**
   * The spawn-side half of binfmt: read `path`, follow any `#!` chain, and hand back the real ELF
   * plus the argv the interpreter should see. A process spawned from outside (the boot shell, a
   * detached run) has to go through the same rules as one that called execve -- linux draws no
   * distinction, and when only execve knew about `#!`, a script was runnable from a shell but
   * "not an executable" when started directly.
   */
  private async resolveSpawn(path: string, argv: string[]): Promise<{ bytes: Uint8Array; argv: string[] } | { error: string; code: number }> {
    const cache = new Map<string, Uint8Array | null>();
    for (const p of [path]) cache.set(p, await this.readGuestBinary(p));
    for (let depth = 0; depth <= MAX_INTERP_DEPTH; depth++) {
      const bytes = cache.get(path);
      if (!bytes) return { error: `cannot read ${path}`, code: -1 };
      const sb = parseShebang(bytes);
      if (!sb) {
        if (!isElf(bytes)) return { error: `${path}: not an executable (ENOEXEC)`, code: 126 };
        return { bytes, argv };
      }
      if (depth === MAX_INTERP_DEPTH) return { error: `${path}: interpreter loop (ELOOP)`, code: 126 };
      const next: string[] = [sb.interp];
      if (sb.arg) next.push(sb.arg);
      next.push(path, ...argv.slice(1));
      path = sb.interp;
      argv = next;
      cache.set(path, await this.readGuestBinary(path));
    }
    return { error: `${path}: interpreter loop (ELOOP)`, code: 126 };
  }

  private async readGuestBinary(path: string): Promise<Uint8Array | null> {
    if (!this.bridge) return null;
    const r: any = await this.bridge.call('load_root_file', { path });
    if (!r || r.error || typeof r.data !== 'string') return null;
    return Uint8Array.from(atob(r.data), (c) => c.charCodeAt(0));
  }

  /** wasm-memory accounting, for diagnosing the address-space reservation budget. a SHARED memory
   *  reserves its declared `maximum` up front and a wasm memory never shrinks, so what we hold here
   *  (live threaded groups + the parked spare) is what a new threaded process has to fit alongside. */
  memStats(): Record<string, number> {
    let sharedBytes = 0;
    for (const m of this.sharedMems.values()) sharedBytes += m.buffer.byteLength;
    const spareBytes = this.spareSharedMem ? this.spareSharedMem.mem.buffer.byteLength : 0;
    const b0 = procSlotOffset(0) / 4;
    return {
      slot0Pid: Atomics.load(this.procTable, b0 + PROC_PID),
      slot0State: Atomics.load(this.procTable, b0 + PROC_STATE),
      slot0Exit: Atomics.load(this.procTable, b0 + PROC_EXIT_CODE),
      liveWorkers: this.workers.size,
      idleWorkers: this.idleWorkers.length,
      // what jsc's single 4 GiB wasm allowance is actually carrying: shared reservations plus
      // committed non-shared. sharedMiB below is only the first half of that.
      wasmMiB: Math.round(this.wasmBytesInUse() / 1048576),
      sharedMems: this.sharedMems.size,
      sharedMiB: Math.round(sharedBytes / 1048576),
      spareMiB: Math.round(spareBytes / 1048576),
      tiersInUse: this.threadLayouts.size,
      reservedMiB: Math.round((sharedBytes + spareBytes) / 1048576),
    };
  }

  /**
   * Everything finite the kernel hands out, plus the whole process table.
   *
   * The recurring failure mode in this system is not a wrong answer, it is a table quietly filling
   * up or an entry going missing -- proc slots, arena pages, pipe blocks, stdin rings. From the
   * outside every one of those looks identical: the guest stops. This makes the difference legible
   * without a rebuild, so a wedge can name the resource that ran out instead of being narrowed down
   * by bisection.
   */
  census(): Record<string, unknown> {
    const procs: Array<Record<string, number>> = [];
    for (let s = 0; s < PROC_TABLE_SLOTS; s++) {
      const b = procSlotOffset(s) / 4;
      const pid = Atomics.load(this.procTable, b + PROC_PID);
      if (pid === 0) continue;
      procs.push({
        slot: s,
        pid,
        ppid: Atomics.load(this.procTable, b + PROC_PPID),
        tgid: Atomics.load(this.procTable, b + PROC_TGID),
        state: Atomics.load(this.procTable, b + PROC_STATE),
        exit: Atomics.load(this.procTable, b + PROC_EXIT_CODE),
        reparented: Atomics.load(this.procTable, b + PROC_REPARENTED),
        hasWorker: this.workers.has(pid) ? 1 : 0,
        // which call it is inside, and whether it is parked in one. "stuck" is not a state; being
        // stuck in a named syscall is, and that is the difference between a report and a guess.
        syscall: Atomics.load(this.procTable, b + PROC_SYSCALL),
        blocked: Atomics.load(this.procTable, b + PROC_BLOCKED),
      });
    }
    let shmBytes = 0;
    for (const e of this.shmRefs.values()) shmBytes += e.len;
    return {
      procs,
      slotsUsed: procs.length,
      slotsTotal: PROC_TABLE_SLOTS,
      workers: this.workers.size,
      idleWorkers: this.idleWorkers.length,
      idleMiB: Math.round([...this.idleBytes.values()].reduce((a, b) => a + b, 0) / 1048576),
      // a worker with no proc-table slot, or a slot with no worker: either is a leak in progress
      orphanWorkers: [...this.workers.keys()].filter((p) => !procs.some((q) => q.pid === p)),
      arenaUsedKiB: Math.round(this.shmAlloc.used() / 1024),
      // anything but zero means a shared block was handed back twice, which is a refcount bug
      // somewhere upstream. the arena refuses the second release, so this is the only trace of it.
      arenaDoubleReleases: this.shmAlloc.doubleReleases,
      workersCreated: this.workersCreated,
      workersReused: this.workersReused,
      workersParked: this.workersParked,
      workersRetired: this.workersRetired,
      arenaTotalKiB: Math.round(SHM_ARENA_SIZE / 1024),
      shmMappings: this.shmRefs.size,
      shmBytes,
      // a stuck reader is almost always a ring whose writer count never reached zero, so report the
      // counts themselves: "readers 1 writers 1 but every holder is dead" names the bug outright.
      pipes: [...this.pipeBlocks.keys()].map((off) => {
        const r = new PipeRing(this.shmArena, off);
        const holders = [...(this.pipeHolders.get(off)?.keys() ?? [])];
        return { off, bytes: r.capacity, buffered: r.available,
                 writers: Atomics.load(new Int32Array(this.shmArena, off, 4), PIPE_WRITER_CLOSED),
                 readers: Atomics.load(new Int32Array(this.shmArena, off, 4), PIPE_READER_CLOSED),
                 holders, holdersAlive: holders.filter((p) => this.pidIsLive(p)) };
      }),
      refusals: Object.fromEntries(this.refusals),
      pipeBlocks: this.pipeBlocks.size,
      pipePages: this.pipePages,
      pipePagesSoft: PIPE_PAGES_SOFT,
      pipeHolders: this.pipeHolders.size,
      unixNames: this.unixNames.size,
      detachedRuns: this.detached.size,
      locks: this.locks.count?.() ?? -1,
      // every map keyed by pid is a place a dead process can be left behind. a census that only
      // counted the obvious ones (workers, slots, arena) could report a clean system while the
      // bookkeeping underneath it grew without bound -- which is what sustained load actually
      // hits. count them all, and count how many entries name a pid that no longer exists.
      bookkeeping: {
        sharedMems: this.sharedMems.size,
        threadCounts: this.threadCounts.size,
        threadedLeaders: this.threadedLeaders.size,
        threadLayouts: this.threadLayouts.size,
        pendingPromotes: this.pendingPromotes.size,
        lastActive: this.lastActive.size,
        committed: this.committed.size,
        swapped: this.swapped.size,
        detachedOwner: this.detachedOwner.size,
        shmHeld: this.shmHeld.size,
        shmByKey: this.shmByKey.size,
        pipeAsync: this.pipeAsync.size,
        sysvShm: this.sysvShm.size,
        sysvShmByKey: this.sysvShmByKey.size,
        sysvMsg: this.sysvMsg.size,
        sysvMsgByKey: this.sysvMsgByKey.size,
        lockParked: this.lockParked.size,
        refusals: this.refusals.size,
      },
      stale: this.staleBookkeeping(procs.map((p) => p.pid)),
      ...this.memStats(),
    };
  }

  /** how many entries in each pid-keyed map name a process that no longer exists. anything but
   *  zero here is a leak, and it is the kind that only shows up after thousands of processes. */
  private staleBookkeeping(live: number[]): Record<string, number> {
    const alive = new Set(live);
    const count = (keys: Iterable<number>) => {
      let n = 0;
      for (const k of keys) if (!alive.has(k)) n++;
      return n;
    };
    return {
      sharedMems: count(this.sharedMems.keys()),
      threadCounts: count(this.threadCounts.keys()),
      threadedLeaders: count(this.threadedLeaders),
      threadLayouts: count(this.threadLayouts.keys()),
      pendingPromotes: count(this.pendingPromotes.keys()),
      lastActive: count(this.lastActive.keys()),
      committed: count(this.committed.keys()),
      detachedOwner: count(this.detachedOwner.keys()),
      shmHeld: count(this.shmHeld.keys()),
      pipeAsync: count(this.pipeAsync.keys()),
      workers: count(this.workers.keys()),
    };
  }

  /** Send keyboard input to the foreground process's stdin ring. */
  /**
   * Feed keyboard (or control-channel) input to the foreground process.
   *
   * the return value used to be discarded, so a ring that was momentarily full simply ATE the
   * characters that did not fit -- silently, and from the middle of whatever the user was typing.
   * a ring only stays full for as long as the reader takes to drain it, so retrying is the whole
   * fix; what is not acceptable is losing input and saying nothing.
   */
  pushStdin(data: Uint8Array): void {
    if (!this.fgStdinRing) return;
    let off = 0;
    for (let attempt = 0; off < data.length && attempt < 64; attempt++) {
      const n = this.fgStdinRing.write(data.subarray(off));
      if (n < 0) return;          // every reader is gone; there is nobody to type at
      off += n;
    }
    if (off < data.length) {
      console.warn(`[kernel] dropped ${data.length - off} input bytes: stdin ring stayed full`);
    }
  }

  setTermSize(cols: number, rows: number): void {
    if (cols < 1 || rows < 1) return;
    if (cols === this.termCols && rows === this.termRows) return;
    this.termCols = cols;
    this.termRows = rows;
    for (const w of this.workers.values()) {
      w.postMessage({ type: 'term_size', cols, rows });
    }
  }

  /** Send Ctrl+C (SIGINT) to the foreground process group. */
  sendInterrupt(): void {
    const fgPgid = Atomics.load(this.globalState, GLOBAL_FG_PGID);
    for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
      const slotBase = procSlotOffset(slot) / 4;
      const state = Atomics.load(this.procTable, slotBase + PROC_STATE);
      if (state === PROC_RUNNING || state === PROC_STOPPED) {
        const pgid = Atomics.load(this.procTable, slotBase + PROC_PGID);
        if (pgid === fgPgid) {
          // an interrupt from the terminal comes from the terminal, not from a process.
          sigqPush(this.procTable, slot, { signo: 2, code: 0x80 /* SI_KERNEL */, pid: 0, uid: 0, value: 0n });
          // Wake the Worker from any Atomics.wait it might be in
          // 1. Bridge call mailbox
          const mailbox = new Int32Array(this.sab, mailboxSlotOffset(slot), MAILBOX_SLOT_SIZE / 4);
          Atomics.notify(mailbox, MBOX_READY);
          // 2. Stdin ring (blocked read waits on the ring's write pointer)
          const stdinIdx = Atomics.load(this.procTable, slotBase + 7); // PROC_STDIN_RING
          if (stdinIdx >= 0) {
            const ringI32 = new Int32Array(this.sab, stdinRingOffset(stdinIdx), 4);
            Atomics.notify(ringI32, 0);
          }
        }
      }
    }
  }

  // ── memory swap ───────────────────────────────────────────────────────────
  // a wasm memory never shrinks and there is no discard, so the only way to give a process's
  // bytes back is to destroy the whole Memory -- which means destroying its Worker. freezing is
  // therefore per-process: ask it to hand over an image, take the image, terminate it. thawing
  // spawns a fresh Worker and replays the image, which is the fork-restore path told that the
  // snapshot is the process's own rather than a parent's.

  /** wake a Worker out of any Atomics.wait it may be parked in, the same way a signal does */
  private wakeSlot(slot: number): void {
    const mailbox = new Int32Array(this.sab, mailboxSlotOffset(slot), MAILBOX_SLOT_SIZE / 4);
    Atomics.notify(mailbox, MBOX_READY);
    const stdinIdx = Atomics.load(this.procTable, procSlotOffset(slot) / 4 + PROC_STDIN_RING);
    if (stdinIdx >= 0) {
      Atomics.notify(new Int32Array(this.sab, stdinRingOffset(stdinIdx), 4), 0);
    }
    bumpIoGeneration();
  }

  /** ask a running process to freeze. it answers with a swap_image at its next safe point. */
  swapOut(pid: number): boolean {
    if (pid <= 1) return false;
    const slot = this.slotOf(pid);
    if (slot < 0) return false;
    const base = procSlotOffset(slot) / 4;
    if (Atomics.load(this.procTable, base + PROC_STATE) !== PROC_RUNNING) return false;
    if (!this.workers.has(pid)) return false;
    Atomics.store(this.procTable, base + PROC_SWAP_REQ, 1);
    this.wakeSlot(slot);
    return true;
  }

  private handleSwapImage(pid: number, slot: number, msg: any): void {
    const base = procSlotOffset(slot) / 4;
    Atomics.store(this.procTable, base + PROC_SWAP_REQ, 0);
    this.swapped.set(pid, {
      ppid: Atomics.load(this.procTable, base + PROC_PPID),
      pgid: Atomics.load(this.procTable, base + PROC_PGID),
      sid: Atomics.load(this.procTable, base + PROC_SID),
      slot,
      stdinRing: Atomics.load(this.procTable, base + PROC_STDIN_RING),
      memSnapshot: msg.memSnapshot,
      syscallState: msg.syscallState,
      kernelPipesSnapshot: msg.kernelPipesSnapshot,
      shmMaps: msg.shmMaps,
      protArmed: msg.protArmed,
    });
    Atomics.store(this.procTable, base + PROC_STATE, PROC_SWAPPED);
    this.lastActive.delete(pid);
    this.committed.delete(pid); // its memory is gone with the worker
    // terminate rather than pool: pooling would keep the very Memory we are trying to release
    const w = this.workers.get(pid);
    if (w) { this.workers.delete(pid); w.terminate(); }
    // then push the bulk out of the content process entirely
    void this.offloadImage(pid);
  }

  /** chunked, because String.fromCharCode(...wholeArray) overflows the argument list on anything
   *  bigger than a few hundred KB, and these images are megabytes. */
  private static toBase64(buf: ArrayBuffer): string {
    const b = new Uint8Array(buf);
    let s = '';
    for (let i = 0; i < b.length; i += 0x8000) {
      s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000) as unknown as number[]);
    }
    return btoa(s);
  }

  private static fromBase64(s: string): ArrayBuffer {
    const bin = atob(s);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out.buffer;
  }

  /** hand the memory image to the app process. the page index and the small bookkeeping stay
   *  here: they are kilobytes, and keeping them means a thaw knows the layout without a round
   *  trip. this is the tier that matters -- an image parked here still occupies the content
   *  process, whose ceiling is 5 GiB, where the app process has 16. */
  private async offloadImage(pid: number): Promise<void> {
    const img = this.swapped.get(pid);
    if (!img || !this.bridge || img.native || !img.memSnapshot) return;
    const snap = img.memSnapshot;
    // a swap image is a one-off that may be far bigger than the arena, so it always travels as
    // its own buffer -- unlike a fork snapshot, which is staged in shared memory.
    if (!snap.data) return;
    try {
      const r = await this.bridge.call('swap_store', { pid, data: Kernel.toBase64(snap.data) });
      if (typeof r.ret === 'number' && r.ret >= 0 && this.swapped.get(pid) === img) {
        img.pages = snap.pages;
        img.totalLen = snap.totalLen;
        img.storedBytes = snap.data!.byteLength;
        img.native = true;
        img.memSnapshot = undefined; // release the content-process copy
      }
    } catch {
      // stays in the content process; still correct, just not in the big tier
    }
  }

  /** bring a frozen process back. the slot it left is reused, so its pid and place are unchanged. */
  async thaw(pid: number): Promise<boolean> {
    const img = this.swapped.get(pid);
    if (!img) return false;
    let snap = img.memSnapshot;
    if (!snap) {
      // the image lives in the app process; fetch it back before rebuilding
      if (!this.bridge || !img.pages || img.totalLen === undefined) return false;
      const r = await this.bridge.call('swap_load', { pid });
      if (typeof r.data !== 'string') return false;
      snap = { pages: img.pages, data: Kernel.fromBase64(r.data), totalLen: img.totalLen };
      void this.bridge.call('swap_drop', { pid });
    }
    this.swapped.delete(pid);
    this.spawnProcess({
      pid,
      ppid: img.ppid,
      pgid: img.pgid,
      sid: img.sid,
      slot: img.slot,
      stdinRing: img.stdinRing >= 0 ? img.stdinRing : undefined,
      elfBytes: new Uint8Array(0), // the snapshot IS the image; no binary is loaded on a thaw
      argv: [],
      envp: [],
      memSnapshot: snap,
      syscallState: img.syscallState,
      kernelPipesSnapshot: img.kernelPipesSnapshot,
      shmMaps: img.shmMaps,
      protArmed: img.protArmed,
      resume: true,
    });
    return true;
  }

  /** pids currently frozen, with the size of each image */
  swappedProcesses(): Array<{ pid: number; bytes: number; where: string }> {
    return [...this.swapped.entries()].map(([pid, img]) => ({
      pid,
      bytes: img.memSnapshot?.data?.byteLength ?? img.storedBytes ?? 0,
      where: img.native ? 'app-process' : 'content-process',
    }));
  }

  /** the idlest swappable process: running, has a worker, and is neither init nor whatever the
   *  user is looking at. returns -1 when there is nothing safe to take. */
  private pickSwapVictim(): number {
    const fg = Atomics.load(this.globalState, GLOBAL_FG_PGID);
    let best = -1;
    let bestSeen = Infinity;
    for (let s = 0; s < PROC_TABLE_SLOTS; s++) {
      const base = procSlotOffset(s) / 4;
      if (Atomics.load(this.procTable, base + PROC_STATE) !== PROC_RUNNING) continue;
      const pid = Atomics.load(this.procTable, base + PROC_PID);
      if (pid <= 1 || !this.workers.has(pid)) continue;
      if (Atomics.load(this.procTable, base + PROC_PGID) === fg) continue;
      const seen = this.lastActive.get(pid) ?? 0;
      if (seen < bestSeen) { bestSeen = seen; best = pid; }
    }
    return best;
  }

  /** turn automatic reclaim on or off, and report what it would pick right now */
  setSwapPolicy(on: boolean): { enabled: boolean; wouldPick: number } {
    this.swapPolicy = on;
    return { enabled: this.swapPolicy, wouldPick: this.pickSwapVictim() };
  }

  private slotOf(pid: number): number {
    for (let s = 0; s < PROC_TABLE_SLOTS; s++) {
      if (Atomics.load(this.procTable, procSlotOffset(s) / 4 + PROC_PID) === pid) return s;
    }
    return -1;
  }

  // ── Process management ────────────────────────────────────────────────────

  private allocPid(): number {
    return Atomics.add(this.globalState, GLOBAL_NEXT_PID, 1);
  }

  // stdin rings are a FIXED table of STDIN_RING_COUNT slots; stdinRingOffset() does no bounds check, so
  // handing out an index past the end writes straight over the global state that follows it (next-pid,
  // fg-pgid, this very counter). return -1 when exhausted — callers treat that as "no stdin ring", which
  // PROC_STDIN_RING already encodes — instead of silently corrupting the kernel's shared state.
  private allocStdinRing(): number {
    const idx = Atomics.add(this.globalState, GLOBAL_NEXT_STDIN, 1);
    return idx < STDIN_RING_COUNT ? idx : -1;
  }

  /** which process-table slot currently holds `pid`, or -1. */
  private slotOfPid(pid: number): number {
    for (let i = 0; i < PROC_TABLE_SLOTS; i++) {
      if (Atomics.load(this.procTable, procSlotOffset(i) / 4 + PROC_PID) === pid) return i;
    }
    return -1;
  }

  private findFreeSlot(): number {
    for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
      const state = Atomics.load(this.procTable, (procSlotOffset(slot) / 4) + PROC_STATE);
      if (state === PROC_EMPTY) return slot;
    }
    return -1; // table full — caller turns this into a fork() EAGAIN, never a crash
  }

  /** the scheduling attributes of the slot at `base`, for a child about to inherit them. */
  private schedStateOf(base: number): { policy: number; rtprio: number; flags: number; affLo: number; affHi: number; dl: number[]; ioprio: number; ruid: number } {
    const dl: number[] = [];
    for (let i = 0; i < 6; i++) dl.push(Atomics.load(this.procTable, base + PROC_DL_RUNTIME + i));
    return {
      policy: Atomics.load(this.procTable, base + PROC_POLICY),
      rtprio: Atomics.load(this.procTable, base + PROC_RTPRIO),
      flags: Atomics.load(this.procTable, base + PROC_SCHED_FLAGS),
      affLo: Atomics.load(this.procTable, base + PROC_AFFINITY),
      affHi: Atomics.load(this.procTable, base + PROC_AFFINITY + 1),
      dl,
      // i/o priority and the owning user are inherited across fork, like the rest of this block.
      ioprio: Atomics.load(this.procTable, base + PROC_IOPRIO),
      ruid: Atomics.load(this.procTable, base + PROC_RUID),
    };
  }

  /** the pid that should adopt this process's orphans: the nearest subreaper above it, else init. */
  private nearestSubreaper(pid: number): number {
    let cur = pid;
    for (let hops = 0; hops < PROC_TABLE_SLOTS; hops++) {
      const slot = this.slotOfPid(cur);
      if (slot < 0) return 1;
      const base = procSlotOffset(slot) / 4;
      const parent = Atomics.load(this.procTable, base + PROC_PPID);
      if (parent <= 1) return 1;
      const pslot = this.slotOfPid(parent);
      if (pslot < 0) return 1;
      if (Atomics.load(this.procTable, procSlotOffset(pslot) / 4 + PROC_SUBREAPER)) return parent;
      cur = parent;
    }
    return 1;
  }

  private spawnProcess(opts: {
    pid: number;
    ppid: number;
    pgid: number;
    sid?: number;
    nice?: number;
    sched?: { policy: number; rtprio: number; flags: number; affLo: number; affHi: number; dl: number[]; ioprio: number; ruid: number };
    slot?: number;
    stdinRing?: number;
    elfBytes: Uint8Array;
    argv: string[];
    envp: string[];
    memSnapshot?: SparseMemSnapshot;
    syscallState?: unknown;
    shmMaps?: Array<{ addr: number; len: number; off: number; sysv?: number }>;
    protArmed?: boolean;
    kernelPipesSnapshot?: unknown;
    unixSocksSnapshot?: unknown;
    cloneFlags?: number;
    childStack?: string;
    childTls?: string;
    childCtid?: string;
    threadStart?: WorkerInitConfig['threadStart'];
    resume?: boolean;
  }): void {
    const slot = opts.slot ?? this.findFreeSlot();
    const slotBase = procSlotOffset(slot) / 4;

    // a fresh slot may still hold the credentials of whoever had it last, and kill(2) reads those
    // to decide who may signal whom. seed them from the parent -- which is what the child will
    // publish for itself a moment later anyway -- so the window in between says the right thing.
    if (opts.slot === undefined) {
      const parentSlot = this.slotOfPid(opts.ppid);
      const dst = procCredOffset(slot) / 4;
      const src = parentSlot >= 0 ? procCredOffset(parentSlot) / 4 : -1;
      for (let i = 0; i < PROC_CRED_WORDS; i++) {
        Atomics.store(this.procTable, dst + i, src >= 0 ? Atomics.load(this.procTable, src + i) : 0);
      }
    }

    // Initialize process table entry
    Atomics.store(this.procTable, slotBase + PROC_PID, opts.pid);
    Atomics.store(this.procTable, slotBase + PROC_PPID, opts.ppid);
    Atomics.store(this.procTable, slotBase + PROC_PGID, opts.pgid);
    // a session is inherited, like a process group: only setsid(2) starts a new one. a process
    // with no parent to inherit from (init, and a resumed image whose sid travels with it) leads
    // its own.
    Atomics.store(this.procTable, slotBase + PROC_SID, opts.sid ?? opts.pgid);
    if (!opts.resume) {
      Atomics.store(this.procTable, slotBase + PROC_EXECED, 0);
      // being a subreaper is NOT inherited: fork(2) hands the child a clean slate.
      Atomics.store(this.procTable, slotBase + PROC_SUBREAPER, 0);
    }
    Atomics.store(this.procTable, slotBase + PROC_STATE, PROC_RUNNING);
    Atomics.store(this.procTable, slotBase + PROC_EXIT_CODE, 0);
    Atomics.store(this.procTable, slotBase + PROC_WAIT_FUTEX, 0);
    // a reused slot must not hand its new occupant signals meant for the last one.
    sigqReset(this.procTable, slot);
    Atomics.store(this.procTable, slotBase + PROC_STDIN_RING, opts.stdinRing ?? -1);
    Atomics.store(this.procTable, slotBase + PROC_REPARENTED, 0);
    Atomics.store(this.procTable, slotBase + PROC_SWAP_REQ, 0);
    Atomics.store(this.procTable, slotBase + PROC_BLOCKED, 0);
    // cpu accounting starts at zero, and the scheduling priority is inherited: fork(2) gives the
    // child its parent's nice value, and a slot reused by a later process must not keep the old
    // tenant's.
    Atomics.store(this.procTable, slotBase + PROC_UTIME_MS, 0);
    Atomics.store(this.procTable, slotBase + PROC_STIME_MS, 0);
    Atomics.store(this.procTable, slotBase + PROC_MAXRSS_KB, 0);
    Atomics.store(this.procTable, slotBase + PROC_NICE, opts.nice ?? 0);
    // scheduling attributes are inherited too, unless the parent asked for SCHED_RESET_ON_FORK --
    // which exists so a real-time process can fork a child that cannot inherit its priority.
    const sched = opts.sched;
    const resetOnFork = (sched?.flags ?? 0) & 1;
    Atomics.store(this.procTable, slotBase + PROC_POLICY, resetOnFork ? 0 : (sched?.policy ?? 0));
    Atomics.store(this.procTable, slotBase + PROC_RTPRIO, resetOnFork ? 0 : (sched?.rtprio ?? 0));
    // the reset itself is not inherited: linux applies SCHED_RESET_ON_FORK to the child and then
    // clears it, so the grandchild is an ordinary one.
    Atomics.store(this.procTable, slotBase + PROC_SCHED_FLAGS, resetOnFork ? 0 : (sched?.flags ?? 0));
    Atomics.store(this.procTable, slotBase + PROC_AFFINITY, sched?.affLo ?? 0);
    Atomics.store(this.procTable, slotBase + PROC_AFFINITY + 1, sched?.affHi ?? 0);
    // i/o priority is inherited and SCHED_RESET_ON_FORK does not touch it; the owning user is
    // republished by the process itself at its first syscall, but a reused slot must not show the
    // last tenant's until then.
    Atomics.store(this.procTable, slotBase + PROC_IOPRIO, sched?.ioprio ?? 0);
    Atomics.store(this.procTable, slotBase + PROC_RUID, sched?.ruid ?? 0);
    for (let i = 0; i < 6; i++) {
      Atomics.store(this.procTable, slotBase + PROC_DL_RUNTIME + i, resetOnFork ? 0 : (sched?.dl?.[i] ?? 0));
    }
    // being traced is not inherited either, and a REUSED slot must not hand its new occupant the
    // last tenant's tracer -- that process would stop for a debugger that was never watching it.
    Atomics.store(this.procTable, slotBase + PROC_TRACER, 0);
    Atomics.store(this.procTable, slotBase + PROC_TRACE_CMD, 0);
    // a vforked child owes its parent an exec or an exit before that parent may run again.
    Atomics.store(this.procTable, slotBase + PROC_VFORK,
                  ((opts.cloneFlags ?? 0) & 0x00004000) ? 1 : 0);
    // the parent-death signal is explicitly NOT inherited: linux clears it in the child of a fork,
    // because the setting names a relationship the child does not have yet.
    Atomics.store(this.procTable, slotBase + PROC_PDEATHSIG, 0);
    // tgid == pid for a normal process / thread-group leader; a thread carries its leader's pid.
    Atomics.store(this.procTable, slotBase + PROC_TGID, opts.threadStart?.tgid ?? opts.pid);

    // Take a warm Worker from the idle pool, or create one. reuse means the Worker's translator
    // instance + wasm Memory are kept (its init() wipes+resets); no per-spawn construction, and
    // crucially no terminate on exit -> no GC churn under a fork storm.
    const parked = this.takeParked();
    if (parked) this.workersReused++; else this.workersCreated++;
    const worker = parked ?? new Worker(this.workerURL);
    this.workers.set(opts.pid, worker);

    // (Re)bind the message/error handlers to THIS pid+slot (a reused Worker was bound to its
    // previous tenant).
    // AN EXCEPTION HERE IS NOT SOMETHING TO LOSE. a message handled halfway leaves the system in
    // whatever half-state it reached -- an exit that never marks the zombie leaves a process
    // RUNNING forever and its parent in wait4 forever -- and with the throw swallowed by the event
    // loop there is nothing to say what happened. so: name it, on the terminal, with the message
    // that caused it.
    worker.onmessage = (e) => {
      try {
        this.handleWorkerMessage(opts.pid, slot, e.data);
      } catch (err) {
        this.onOutput(new TextEncoder().encode(`[kernel] pid ${opts.pid} '${(e.data as any)?.type}' handler threw: ${err}\r\n`));
      }
    };
    if (this.tracing) worker.postMessage({ type: 'set_trace', on: true });
    worker.onerror = (e) => {
      console.error(`[kernel] Worker ${opts.pid} error:`, e.message, e.filename, e.lineno);
      // surface worker crashes on the terminal — otherwise they are invisible in the
      // native app (no console). the message often pinpoints a translator/syscall gap.
      const msg = `\r\n[kernel] worker ${opts.pid} crashed: ${e.message}\r\n`;
      // a detached run's crash belongs in its captured output, not on the terminal it never used.
      try {
        const own = this.detachedOwner.get(opts.pid);
        const cap = own !== undefined ? this.detached.get(own) : undefined;
        if (cap) cap.chunks.push(new TextEncoder().encode(msg));
        else this.onOutput(new TextEncoder().encode(msg));
      } catch {}
      // a crashed worker must be reaped like an abnormal exit — otherwise its proc slot stays
      // RUNNING forever and its parent's wait4 blocks indefinitely (wedging the shell), and the
      // slot leaks. status 134 = 128 + SIGABRT. guard on (pid,RUNNING) so we don't double-handle
      // a worker that already posted 'exit', or a slot the kernel has since reused.
      const base = procSlotOffset(slot) / 4;
      if (Atomics.load(this.procTable, base + PROC_PID) === opts.pid &&
          Atomics.load(this.procTable, base + PROC_STATE) === PROC_RUNNING) {
        this.handleExit(opts.pid, slot, 134);
      }
    };

    // a name in the unix-domain namespace belongs to the SOCKET, and the child now holds the same
    // socket. without counting it, the parent's exit takes the name away while the child is still
    // listening on it -- and the next bind to that path succeeds against a live server.
    for (const entry of (opts.unixSocksSnapshot as Array<[number, { path?: string }]> | undefined) ?? []) {
      const nm = entry?.[1]?.path;
      if (nm) this.unixNames.get(nm)?.holders.add(opts.pid);
    }

    // Build init config
    const config: WorkerInitConfig = {
      pid: opts.pid,
      ppid: opts.ppid,
      pgid: opts.pgid,
      slot,
      kernelSAB: this.sab,
      shmArena: this.shmArena,
      shmMaps: opts.shmMaps,
      protArmed: opts.protArmed,
      elfBytes: opts.elfBytes,
      argv: opts.argv,
      envp: opts.envp,
      memSnapshot: opts.memSnapshot,
      syscallState: opts.syscallState,
      kernelPipesSnapshot: opts.kernelPipesSnapshot as any,
      unixSocksSnapshot: opts.unixSocksSnapshot as any,
      cloneFlags: opts.cloneFlags,
      childStack: opts.childStack,
      childTls: opts.childTls,
      childCtid: opts.childCtid,
      resume: opts.resume,
      termCols: this.termCols,
      termRows: this.termRows,
      // prefer the shared compiled module; only ship raw bytes if compilation failed above.
      zigTranslatorModule: this.zigTranslatorModule,
      zigTranslatorBytes: this.zigTranslatorModule ? undefined : this.zigTranslatorBytes,
      threadStart: opts.threadStart,
    };

    // Send init — transfer the sparse snapshot's packed pages if present (zero-copy)
    // a snapshot staged in the shared arena has nothing to transfer: the bytes are already
    // somewhere both sides can see, which is the whole point of staging them there.
    const transferList: Transferable[] = [];
    if (config.memSnapshot?.data) transferList.push(config.memSnapshot.data);
    worker.postMessage({ type: 'init', config }, transferList);
  }

  // release a thread-group's shared memory. instead of dropping it (relying on GC, which lags under
  // back-to-back threaded launches), park it as the single reusable spare so the next threaded process
  // takes it over. if a spare is already parked (a second threaded group ran concurrently), drop this
  // one for GC. `high` is the group's touched high-water so the reuser only zeros the stale span.
  private freeSharedMem(tgid: number, high = 0): void {
    const mem = this.sharedMems.get(tgid);
    this.sharedMems.delete(tgid);
    this.threadedLeaders.delete(tgid);
    const layout = this.threadLayouts.get(tgid) ?? DEFAULT_THREAD_LAYOUT;
    this.threadLayouts.delete(tgid);
    if (mem && !this.spareSharedMem) this.spareSharedMem = { mem, high, layout };
  }

  // first pthread_create ('clone_thread'): create this process's shared memory (only the main thread
  // can) and reserve the child's tid + reg-block slot, then hand the memory back to the caller via
  // 'promote'. the caller copies its state into it and replies 'thread_go', at which point the memory
  // holds the process state and we can safely spawn the thread-Worker (handleThreadGo).
  private handleCloneThreadPromote(parentPid: number, parentSlot: number, msg: KernelMessage & { type: 'clone_thread' }): void {
    const worker = this.workers.get(parentPid);
    if (!worker) return;
    const slot = this.findFreeSlot();
    if (slot < 0 || (this.idleWorkers.length === 0 && this.workers.size >= POOL_MAX)) {
      klog(`promote EAGAIN pid=${parentPid} slot=${slot} idle=${this.idleWorkers.length} workers=${this.workers.size}`);
      worker.postMessage({ type: 'clone_fail' });
      return;
    }
    // reuse the single spare shared memory if one is parked (a previous threaded process freed it).
    // reusing avoids allocating+GC-freeing a ~0.5 GiB shared memory per threaded process, which the
    // engine can't reclaim fast enough under back-to-back launches -> the caller zeros the previous
    // tenant's stale region (staleHigh) instead. only when no spare is available do we allocate.
    // size the process's address space from its current footprint + remaining budget. the spare is only
    // reusable when it was cut to the SAME tier — a different tier puts the register blocks and shadow
    // at different offsets, so handing it over would scatter thread state across the wrong addresses.
    const layout = this.pickThreadTier(msg.curPages ?? 0);
    klog(`promote pid=${parentPid} curPages=${msg.curPages} -> tier ${layout.pages * 65536} (reserved=${this.reservedBytes()})`);
    let memory: WebAssembly.Memory;
    let staleHigh = 0;
    if (this.spareSharedMem && this.spareSharedMem.layout.pages === layout.pages) {
      memory = this.spareSharedMem.mem;
      staleHigh = this.spareSharedMem.high;
      this.spareSharedMem = null;
    } else {
      const mem = this.allocSharedMem(layout);
      if (!mem) {
        klog(`promote EAGAIN pid=${parentPid}: no shared memory for tier ${layout.pages * 65536} ` +
             `(reserved=${this.reservedBytes()} spare=${this.spareSharedMem ? this.spareSharedMem.mem.buffer.byteLength : 0})`);
        worker.postMessage({ type: 'clone_fail' });
        return; // -> pthread_create EAGAIN
      }
      memory = mem;
    }
    // reserve the child's slot NOW so a concurrent spawn can't claim it before thread_go arrives.
    const slotBase = procSlotOffset(slot) / 4;
    Atomics.store(this.procTable, slotBase + PROC_STATE, PROC_RUNNING);
    Atomics.store(this.procTable, slotBase + PROC_PID, -1);

    const tid = this.allocPid();
    const parentBase = procSlotOffset(parentSlot) / 4;
    const pgid = Atomics.load(this.procTable, parentBase + PROC_PGID);
    const tgid = Atomics.load(this.procTable, parentBase + PROC_PID); // parent is the group leader here
    const regBaseAddr = layout.regsBase + slot * 0x400;

    this.threadLayouts.set(tgid, layout);
    this.sharedMems.set(tgid, memory);
    this.threadCounts.set(tgid, 1); // the leader (already running) counts as one live thread
    this.threadedLeaders.add(tgid); // its worker holds the shared memory -> must not be pooled on exit
    this.pendingPromotes.set(parentPid, { memory, tid, slot, regBaseAddr, tgid, pgid });
    worker.postMessage({ type: 'promote', sharedMemory: memory, tid, regBaseAddr, staleHigh, layout });
  }

  // the promoted caller has copied its state into the shared memory and told us to go. spawn the
  // thread-Worker on that memory at its reserved slot/reg-block.
  private handleThreadGo(parentPid: number, msg: KernelMessage & { type: 'thread_go' }): void {
    const p = this.pendingPromotes.get(parentPid);
    if (!p) return;
    this.pendingPromotes.delete(parentPid);
    this.threadCounts.set(p.tgid, (this.threadCounts.get(p.tgid) ?? 1) + 1); // the child is now live
    const promoterBase = procSlotOffset(this.slotOfPid(parentPid)) / 4;
    this.spawnProcess({
      pid: p.tid, ppid: parentPid, pgid: p.pgid, slot: p.slot,
      elfBytes: this.elfBytes, argv: [], envp: [],
      nice: Atomics.load(this.procTable, promoterBase + PROC_NICE),
      sched: this.schedStateOf(promoterBase),
      threadStart: {
        sharedMemory: p.memory, startPC: msg.startPC, childStack: msg.childStack, tls: msg.tls,
        tid: p.tid, tgid: p.tgid, regBaseAddr: p.regBaseAddr, parentRegBase: msg.parentRegBase,
        childTidPtr: msg.childTidPtr, flags: msg.flags, syscallState: (msg as any).syscallState,
        unixSocksSnapshot: (msg as any).unixSocksSnapshot,
        shmMaps: (msg as any).shmMaps,
        layout: this.threadLayouts.get(p.tgid) ?? DEFAULT_THREAD_LAYOUT,
      },
    });
  }

  // subsequent pthread_create ('spawn_thread'): the group's shared memory already exists (keyed by
  // tgid), so reuse it. reply the tid (or -1 EAGAIN) to the caller via its mailbox.
  private handleSpawnThread(parentPid: number, parentSlot: number, msg: KernelMessage & { type: 'spawn_thread' }): void {
    const mb = new Int32Array(this.sab, mailboxSlotOffset(parentSlot), MAILBOX_SLOT_SIZE / 4);
    const memory = this.sharedMems.get(msg.tgid);
    const slot = this.findFreeSlot();
    if (!memory || slot < 0 || (this.idleWorkers.length === 0 && this.workers.size >= POOL_MAX)) {
      klog(`spawn_thread EAGAIN tgid=${msg.tgid} mem=${!!memory} slot=${slot} idle=${this.idleWorkers.length} workers=${this.workers.size}`);
      Atomics.store(mb, MBOX_VALUE0, -1); // EAGAIN
      Atomics.store(mb, MBOX_READY, 1);
      Atomics.notify(mb, MBOX_READY);
      return;
    }
    const tid = this.allocPid();
    const parentBase = procSlotOffset(parentSlot) / 4;
    const pgid = Atomics.load(this.procTable, parentBase + PROC_PGID);
    const regBaseAddr = (this.threadLayouts.get(msg.tgid) ?? DEFAULT_THREAD_LAYOUT).regsBase + slot * 0x400;
    this.threadCounts.set(msg.tgid, (this.threadCounts.get(msg.tgid) ?? 1) + 1);

    this.spawnProcess({
      pid: tid, ppid: parentPid, pgid, slot,
      elfBytes: this.elfBytes, argv: [], envp: [],
      nice: Atomics.load(this.procTable, parentBase + PROC_NICE),
      sched: this.schedStateOf(parentBase),
      threadStart: {
        sharedMemory: memory, startPC: msg.startPC, childStack: msg.childStack, tls: msg.tls,
        tid, tgid: msg.tgid, regBaseAddr, parentRegBase: msg.parentRegBase, childTidPtr: msg.childTidPtr, flags: msg.flags,
        syscallState: (msg as any).syscallState,
        unixSocksSnapshot: (msg as any).unixSocksSnapshot,
        shmMaps: (msg as any).shmMaps,
        layout: this.threadLayouts.get(msg.tgid) ?? DEFAULT_THREAD_LAYOUT,
      },
    });

    Atomics.store(mb, MBOX_VALUE0, tid);
    Atomics.store(mb, MBOX_READY, 1);
    Atomics.notify(mb, MBOX_READY);
  }

  // ── Worker message handling ───────────────────────────────────────────────

  private handleWorkerMessage(pid: number, slot: number, msg: KernelMessage): void {
    // only track a process we still own a Worker for. this used to record EVERY message from
    // EVERY pid, so a message that arrived after the process was gone -- including the exit
    // message itself, which is dispatched after this line -- put the dead pid back. under load
    // that grew by tens of entries a second and never shrank: measured at 5820 dead pids after
    // two minutes of ltp. bounding it by the live worker set makes it structurally incapable of
    // growing past the number of processes that exist.
    if (this.workers.has(pid)) this.lastActive.set(pid, performance.now());
    switch (msg.type) {
      case 'fork':
        this.handleFork(pid, slot, msg);
        break;

      // the child has copied the pages out of its parent's staging block, so the parent may have
      // fork(2) back and reuse it. this is why the reply is deferred rather than sent at spawn:
      // the block is shared memory, and the parent's next fork would overwrite it underneath a
      // child that had not finished reading.
      case 'fork_ready':
        this.releaseForkStage(pid);
        break;

      // reboot(2) asked for the system to stop. ending init ends everything: the exit path already
      // treats the death of pid 1 as the end of the world.
      case 'halt': {
        const initSlot = this.slotOfPid(1);
        if (initSlot >= 0) this.handleExit(1, initSlot, 0);
        break;
      }

      case 'exit': {
        // exit_group from ANY thread ends the whole group, and the status belongs to the leader --
        // that is the one the parent waits for. exiting only the calling thread left the process
        // alive with its leader still parked wherever it was.
        const tgid = Atomics.load(this.procTable, procSlotOffset(slot) / 4 + PROC_TGID) || pid;
        if ((msg as any).group === true && tgid !== pid) {
          const leaderSlot = this.slotOfPid(tgid);
          if (leaderSlot >= 0) {
            this.handleExit(tgid, leaderSlot, msg.code, false, 0, 0);
            break;
          }
        }
        // clean self-exit -> the Worker is reusable, but only pool it if its memory is still small: a
        // wasm memory never shrinks, so pooling a Worker that ballooned would hold that reservation
        // for the app's lifetime and starve threaded processes.
        this.handleExit(pid, slot, msg.code, (msg.memBytes ?? 0) <= POOL_MAX_MEM_BYTES, msg.high ?? 0,
                        msg.memBytes ?? 0);
        break;
      }

      case 'mem_size':
        if (this.workers.has(pid)) {
          this.committed.set(pid, (msg as any).bytes);
          this.grownBytes.set(pid, (msg as any).grown ?? (msg as any).bytes);
        }
        break;

      case 'swap_image':
        this.handleSwapImage(pid, slot, msg);
        break;

      case 'kill':
        this.handleKillSignal(msg.pid, msg.sig);
        break;

      case 'write': {
        // a detached run's output (from any process in its tree) belongs to its caller, not the
        // terminal (see runDetached).
        const owner = this.detachedOwner.get(pid);
        const cap = owner !== undefined ? this.detached.get(owner) : undefined;
        if (cap) cap.chunks.push((msg as any).data);
        else this.onOutput((msg as any).data);
        break;
      }

      case 'pipe_alloc':
        this.handlePipeAlloc(pid, slot, msg.priv === true);
        break;

      case 'clone_thread':
        this.handleCloneThreadPromote(pid, slot, msg as KernelMessage & { type: 'clone_thread' });
        break;

      case 'thread_go':
        this.handleThreadGo(pid, msg);
        break;

      case 'spawn_thread':
        this.handleSpawnThread(pid, slot, msg);
        break;

      case 'lock_req':
        this.handleLockReq(pid, slot, msg);
        break;

      // the worker caught its own exception and sent the stack; an ErrorEvent would have lost it.
      // a process that could not grow its memory. ordinary, survivable, and NOT the guest's
      // business to read about: it already got ENOMEM from the syscall that asked.
      case 'grow_refused':
        console.warn(`[kernel] ${(msg as any).detail}`);
        break;

      case 'worker_error':
        console.error(`[kernel] worker ${pid}:`, (msg as any).detail);
        this.onOutput(new TextEncoder().encode(`\r\n[kernel] worker ${pid} threw: ${(msg as any).detail}\r\n`));
        break;

      case 'lease_req':
        this.handleLeaseReq(pid, slot, msg);
        break;

      // ── SysV shared memory ──
      //
      // The ids and the IPC_CREAT/IPC_EXCL rules are system-wide, so the registry has to live in
      // the one place that sees every process. The SEGMENT itself is not new machinery: it is an
      // arena block keyed "sysvshm:<id>", which is exactly what a MAP_SHARED mapping already is,
      // so attaching one is attaching a shared mapping and coherence comes for free.
      case 'sysv_shm': {
        const m = msg as any;
        // shmctl has permissions, and it used to have none: reading a segment's stat needs read
        // access to it, and removing one or changing its mode is the owner's to do. without these
        // any process could read and destroy any other's shared memory.
        const shmMay = (seg: SysvSeg, want: number) =>
          ipcPermitted(seg.mode, seg.uid, seg.gid, m.uid ?? 0, m.gid ?? 0, want);
        const shmOwns = (seg: SysvSeg) => (m.uid ?? 0) === 0 || (m.uid ?? 0) === seg.uid;
        switch (m.op) {
          case 'get': {
            const IPC_CREAT = 0o1000, IPC_EXCL = 0o2000, IPC_PRIVATE = 0;
            const key = m.key | 0;
            if (key !== IPC_PRIVATE) {
              const existing = this.sysvShmByKey.get(key);
              if (existing !== undefined) {
                const seg = this.sysvShm.get(existing)!;
                // IPC_EXCL only means anything alongside IPC_CREAT: it says "create it, and fail
                // if it is already there". on its own it is not a request to create at all, and
                // answering EEXIST to it turned a size mismatch into the wrong error.
                if ((m.flg & IPC_CREAT) && (m.flg & IPC_EXCL)) { this.lockReply(slot, [-17]); break; }
                // a size larger than the segment has cannot be satisfied by joining it
                if (m.size > seg.size) { this.lockReply(slot, [-22]); break; } // EINVAL
                // joining is subject to the segment's permissions, exactly as opening a file is.
                if (!shmMay(seg, m.flg & 0o777)) { this.lockReply(slot, [-13]); break; } // EACCES
                this.lockReply(slot, [seg.id]);
                break;
              }
              if (!(m.flg & IPC_CREAT)) { this.lockReply(slot, [-2]); break; }  // ENOENT
            }
            if (m.size <= 0 || BigInt(m.size) > this.shmmax) { this.lockReply(slot, [-22]); break; } // EINVAL
            if (this.sysvShm.size >= SHMMNI) { this.lockReply(slot, [-28]); break; }     // ENOSPC
            // checkpoint/restore may have asked for a specific id: kernel.shm_next_id names the
            // id the next segment must get, and is consumed by this create.
            const wantShm = Atomics.exchange(this.globalState, GLOBAL_SHM_NEXT_ID, -1);
            let id: number;
            if (wantShm >= 0 && !this.sysvShm.has(wantShm)) id = wantShm;
            else { id = this.nextShmId; this.nextShmId += IPCMNI; }
            const now = this.nowSec();
            this.sysvShm.set(id, {
              // ONLY the permission bits. IPC_CREAT and IPC_EXCL are commands, not permissions,
              // and storing them made every segment report mode 01666 where 0666 was asked for.
              id, key, size: m.size, mode: m.flg & 0o777, uid: m.uid, gid: m.gid,
              cpid: pid, lpid: 0, removed: false, attachers: new Map(), atime: 0, dtime: 0, ctime: now,
            });
            if (key !== IPC_PRIVATE) this.sysvShmByKey.set(key, id);
            this.lockReply(slot, [id]);
            break;
          }
          case 'attach': {
            const seg = this.sysvShm.get(m.id | 0);
            if (!seg) { this.lockReply(slot, [-22]); break; }                   // EINVAL
            // attaching is reading it, and attaching writable is writing it. a segment you have no
            // permission for is EACCES here rather than a mapping you were never entitled to.
            if (!shmMay(seg, m.rdonly ? 0o400 : 0o600)) { this.lockReply(slot, [-13]); break; }
            seg.attachers.set(pid, (seg.attachers.get(pid) ?? 0) + 1);
            seg.lpid = pid;
            seg.atime = this.nowSec();
            this.lockReply(slot, [seg.size]);
            break;
          }
          case 'detach': {
            const seg = this.sysvShm.get(m.id | 0);
            if (!seg) { this.lockReply(slot, [-22]); break; }
            const held = (seg.attachers.get(pid) ?? 0) - 1;
            if (held > 0) seg.attachers.set(pid, held); else seg.attachers.delete(pid);
            seg.lpid = pid;
            seg.dtime = this.nowSec();
            this.dropSysvIfDone(seg);
            this.lockReply(slot, [0]);
            break;
          }
          case 'stat':
          case 'statidx': {
            // SHM_STAT reads by INDEX into the kernel's segment array rather than by id, and hands
            // the id back as its return value. the array is this map's insertion order.
            const seg = m.op === 'stat'
              ? this.sysvShm.get(m.id | 0)
              : [...this.sysvShm.values()][m.id | 0];
            if (!seg) { this.lockReply(slot, [-22]); break; }
            if (!shmMay(seg, 0o400)) { this.lockReply(slot, [-13]); break; }    // EACCES
            // SHM_DEST and SHM_LOCKED are the kernel's own bits in shm_perm.mode: "marked for
            // destruction" and "locked in memory". a caller reads them back to find out what it
            // asked for actually took.
            const statMode = seg.mode | (seg.removed ? 0o1000 : 0) | (seg.locked ? 0o2000 : 0);
            this.lockReply(slot, [0, seg.key, seg.uid, seg.gid, statMode, seg.size,
                                  seg.cpid, seg.lpid, shmNattch(seg), seg.atime, seg.dtime, seg.ctime,
                                  seg.id]);
            break;
          }
          case 'set': {
            const seg = this.sysvShm.get(m.id | 0);
            if (!seg) { this.lockReply(slot, [-22]); break; }
            if (!shmOwns(seg)) { this.lockReply(slot, [-1]); break; }           // EPERM
            seg.mode = m.mode & 0o777; seg.uid = m.newUid | 0; seg.gid = m.newGid | 0;
            seg.ctime = this.nowSec();
            this.lockReply(slot, [0]);
            break;
          }
          // SHM_LOCK / SHM_UNLOCK: pin a segment in memory. everything here is already resident
          // in the shared arena, so the state is the only observable part -- but the PERMISSION is
          // observable too, and refusing a non-owner is the whole of what ltp checks.
          case 'lock': {
            const seg = this.sysvShm.get(m.id | 0);
            if (!seg) { this.lockReply(slot, [-22]); break; }
            if (!shmOwns(seg)) { this.lockReply(slot, [-1]); break; }           // EPERM
            seg.locked = m.on ? true : false;
            this.lockReply(slot, [0]);
            break;
          }
          // IPC_INFO reports the system LIMITS, SHM_INFO the current USAGE. programs sizing a
          // segment ask for these before they ask for memory, and answering EINVAL made them fall
          // back on guesses.
          case 'info': {
            let segs = 0, pages = 0;
            for (const seg of this.sysvShm.values()) { segs++; pages += Math.ceil(seg.size / 4096); }
            // the two limits are 64-bit and travel as halves; sending only the low word made
            // IPC_INFO report 4 GiB minus a page for a limit of 16 exabytes.
            this.lockReply(slot, [0, segs, pages, SHMMNI,
              Number(this.shmmax & 0xffffffffn) | 0, Number((this.shmmax >> 32n) & 0xffffffffn) | 0,
              Number(this.shmall & 0xffffffffn) | 0, Number((this.shmall >> 32n) & 0xffffffffn) | 0]);
            break;
          }
          // the two system limits, as /proc/sys/kernel/{shmmax,shmall} reads and writes them.
          // they are 64-bit, so they travel as halves.
          case 'limits':
            this.lockReply(slot, [0,
              Number(this.shmmax & 0xffffffffn) | 0, Number((this.shmmax >> 32n) & 0xffffffffn) | 0,
              Number(this.shmall & 0xffffffffn) | 0, Number((this.shmall >> 32n) & 0xffffffffn) | 0]);
            break;
          case 'setlimit': {
            const v = (BigInt(m.hi >>> 0) << 32n) | BigInt(m.lo >>> 0);
            if (m.which === 0) this.shmmax = v; else this.shmall = v;
            this.lockReply(slot, [0]);
            break;
          }
          case 'rm': {
            const seg = this.sysvShm.get(m.id | 0);
            if (!seg) { this.lockReply(slot, [-22]); break; }
            if (!shmOwns(seg)) { this.lockReply(slot, [-1]); break; }           // EPERM
            // linux marks a removed segment DESTROYED and keeps it alive for whoever is still
            // attached; it disappears when the last of them detaches. a key that has been removed
            // is free for reuse immediately, though.
            seg.removed = true;
            if (seg.key !== 0) this.sysvShmByKey.delete(seg.key);
            this.dropSysvIfDone(seg);
            this.lockReply(slot, [0]);
            break;
          }
          default: this.lockReply(slot, [-22]);
        }
        break;
      }

      // inotify. the watch registry is system-wide, and so is the notification: a process that
      // changes a file tells the kernel, and the kernel hands the event to everyone watching.
      case 'inotify': {
        const m = msg as any;
        switch (m.op) {
          case 'init': {
            // an instance, not a descriptor: fork copies the descriptor and both halves then read
            // from the SAME queue, which is what linux's shared open file description gives you.
            const id = this.inoNextId++;
            this.inoWatchers.set(id, { watches: new Map(), queue: [], nextWd: 1, overflowed: false, refs: 1 });
            this.lockReply(slot, [0, id]);
            break;
          }
          case 'add': {
            const w = this.inoWatchers.get(m.id);
            if (!w) { this.lockReply(slot, [-9]); break; }              // EBADF
            // adding a watch for a path already watched UPDATES it and keeps the same descriptor,
            // which is how a program refreshes a mask without leaking watch numbers.
            const IN_MASK_ADD = 0x20000000, IN_MASK_CREATE = 0x10000000;
            let done = false;
            for (const [wd, ex] of w.watches) {
              if (ex.path !== m.path) continue;
              if (m.mask & IN_MASK_CREATE) { this.lockReply(slot, [-17]); done = true; break; } // EEXIST
              ex.mask = (m.mask & IN_MASK_ADD) ? (ex.mask | m.mask) : m.mask;
              this.lockReply(slot, [wd]);
              done = true;
              break;
            }
            if (done) break;
            const wd = w.nextWd++;
            w.watches.set(wd, { path: m.path, mask: m.mask, ino: m.ino | 0 });
            this.bumpInotifyCount(+1);
            this.lockReply(slot, [wd]);
            break;
          }
          case 'watches': {
            // one page of the watch list, for /proc/<pid>/fdinfo/<fd>. the mailbox is a fixed number
            // of words, so a caller with many watches asks for the rest by moving `start` along.
            const w = this.inoWatchers.get(m.id);
            if (!w) { this.lockReply(slot, [-9]); break; }
            const all = [...w.watches.entries()];
            const start = m.start | 0;
            const n = Math.min(all.length - start, 9);
            const words = [0, all.length, n];
            for (let i = 0; i < n; i++) {
              const [wd, watch] = all[start + i];
              words.push(wd, watch.mask | 0, watch.ino | 0);
            }
            this.lockReply(slot, words);
            break;
          }
          case 'rm': {
            const w = this.inoWatchers.get(m.id);
            if (!w || !w.watches.has(m.wd)) { this.lockReply(slot, [-22]); break; } // EINVAL
            w.watches.delete(m.wd);
            this.bumpInotifyCount(-1);
            // linux delivers IN_IGNORED when a watch goes away, so a reader knows the descriptor
            // it is holding is dead rather than merely quiet.
            w.queue.push({ wd: m.wd, mask: 0x8000 /* IN_IGNORED */, cookie: 0, name: '' });
            bumpIoGeneration();
            this.lockReply(slot, [0]);
            break;
          }
          case 'close': {
            const w = this.inoWatchers.get(m.id);
            if (w && --w.refs <= 0) {
              this.releaseInoBlock(w);
              this.bumpInotifyCount(-w.watches.size);
              this.inoWatchers.delete(m.id);
            }
            this.lockReply(slot, [0]);
            break;
          }
          case 'read': {
            const w = this.inoWatchers.get(m.id);
            if (!w) { this.lockReply(slot, [-9]); break; }
            this.releaseInoBlock(w);
            if (w.queue.length === 0) { this.lockReply(slot, [-11]); break; } // EAGAIN: nothing queued
            const packed = this.packInotifyEvents(w.queue, m.maxlen);
            if (packed === null) { this.lockReply(slot, [-22]); break; }      // EINVAL: buffer too small
            w.lastBlock = packed;
            this.lockReply(slot, [0, packed.off, packed.len]);
            break;
          }
          case 'pending': {
            const w = this.inoWatchers.get(m.id);
            this.lockReply(slot, [0, w ? w.queue.length : 0]);
            break;
          }
          case 'free': {
            const w = this.inoWatchers.get(m.id);
            if (w) this.releaseInoBlock(w);
            else if (m.len > 0) this.shmAlloc.release(m.off, m.len);
            this.lockReply(slot, [0]);
            break;
          }
          default: this.lockReply(slot, [-22]);
        }
        break;
      }

      // a process reporting a change it just made to the filesystem, so watchers hear about it.
      case 'inotify_event':
        this.deliverInotify((msg as any).path, (msg as any).mask, (msg as any).cookie ?? 0,
                            (msg as any).isdir === true, (msg as any).selfonly === true,
                            (msg as any).moved ?? (msg as any).path, (msg as any).unlinked === true);
        break;

      // SysV message queues. the registry is system-wide, so every operation is a kernel round
      // trip; the message BODIES are arena blocks, which is what lets a receiver in another
      // process read them without a copy through here.
      // a process's fork staging block: where it puts the pages it hands a child. it keeps the
      // block between forks, so the arena sees one allocation per forking process rather than one
      // per fork -- which is the difference between a fork storm being survivable and not.
      case 'fork_stage': {
        const want = Math.max(0, (msg as any).bytes | 0);
        const off = want > 0 ? this.shmAlloc.alloc(want) : -1;
        if (off < 0) { this.lockReply(slot, [-1]); break; }
        // held against the PARENT until the fork message names the child it is for; if the fork
        // never comes (the parent died mid-syscall) the parent's own exit gives it back.
        this.forkStages.set(pid, { off, len: want });
        this.lockReply(slot, [off]);
        break;
      }

      // an eventfd is a counter with a descriptor on it, and fork(2) gives BOTH processes the same
      // counter -- so it cannot live in the process that made it. it lives here, and the descriptor
      // in each process names it.
      // the worker says which image it is now running; a write to that file is ETXTBSY until it
      // execs something else or exits.
      // a worker asking to say something in the kernel's diagnostic ring.
      case 'klog':
        klog(`pid${pid}: ${(msg as any).line}`);
        break;

      case 'exec_image':
        this.setExecImage(pid, ((msg as any).path as string) || null);
        break;

      case 'eventfd': {
        const m = msg as any;
        const ev = this.eventfds.get(m.id | 0);
        switch (m.op) {
          case 'create': {
            const id = this.nextEventfdId++;
            this.eventfds.set(id, {
              count: BigInt(m.initval ?? 0), semaphore: !!m.semaphore, holders: new Set([pid]),
            });
            this.lockReply(slot, [id]);
            break;
          }
          case 'ref':
            if (ev) ev.holders.add(m.pid | 0);
            this.lockReply(slot, [0]);
            break;
          case 'close':
            if (ev) {
              ev.holders.delete(m.pid | 0);
              if (ev.holders.size === 0) this.eventfds.delete(m.id | 0);
            }
            this.lockReply(slot, [0]);
            break;
          case 'peek':
            if (!ev) { this.lockReply(slot, [-9]); break; }
            this.lockReply(slot, [0, Number(ev.count & 0xffffffffn) | 0, Number(ev.count >> 32n) | 0]);
            break;
          case 'read': {
            if (!ev) { this.lockReply(slot, [-9]); break; }
            if (ev.count === 0n) { this.lockReply(slot, [-11]); break; }   // EAGAIN
            // EFD_SEMAPHORE reads one at a time; an ordinary eventfd read takes the whole count.
            const got = ev.semaphore ? 1n : ev.count;
            ev.count -= got;
            bumpIoGeneration();
            this.lockReply(slot, [0, Number(got & 0xffffffffn) | 0, Number(got >> 32n) | 0]);
            break;
          }
          case 'write': {
            if (!ev) { this.lockReply(slot, [-9]); break; }
            const add = BigInt(m.value);
            if (ev.count + add > 0xFFFFFFFFFFFFFFFEn) { this.lockReply(slot, [-11]); break; } // EAGAIN
            ev.count += add;
            bumpIoGeneration();
            this.lockReply(slot, [0]);
            break;
          }
          default:
            this.lockReply(slot, [-22]);
        }
        break;
      }

      case 'posix_mq': {
        const m = msg as any;
        // every reply is an int array: a negative first element is an errno, as everywhere else.
        const mqOf = (id: number): PosixMq | undefined => this.posixMq.get(id | 0);
        // a queue is freed once it has no name AND nobody holding it, which is the only moment
        // its message bodies can be given back to the arena.
        const reap = (q: PosixMq): void => {
          if (!q.unlinked || q.holders.size > 0) return;
          if (q.notify) { this.sendMqCookie(q.notify, NOTIFY_REMOVED); q.notify = null; }
          for (const e of q.msgs) if (e.len > 0) this.shmAlloc.release(e.off, e.len);
          q.msgs.length = 0;
          this.posixMq.delete(q.id);
        };
        switch (m.op) {
          case 'open': {
            const O_CREAT = 0o100, O_EXCL = 0o200;
            const name: string = m.name;
            const existingId = this.posixMqByName.get(name);
            if (existingId !== undefined) {
              if (m.oflag & O_EXCL) { this.lockReply(slot, [-17]); break; }       // EEXIST
              const q = this.posixMq.get(existingId)!;
              // opening asks for read, write or both, and the queue's mode has to grant it.
              const want = ((m.accmode === 1 ? 0 : 4) | (m.accmode === 0 ? 0 : 2)) << 6;
              if (!ipcPermitted(q.mode, q.uid, q.gid, m.uid, m.gid, want)) {
                this.lockReply(slot, [-13]); break;                               // EACCES
              }
              q.holders.add(m.pid);
              this.lockReply(slot, [q.id, q.maxmsg, q.msgsize]);
              break;
            }
            if (!(m.oflag & O_CREAT)) { this.lockReply(slot, [-2]); break; }      // ENOENT
            // the three limits are settings, not constants: /proc/sys/fs/mqueue holds them and a
            // test that lowers one expects the next mq_open to be refused by it.
            const softMsg = Atomics.load(this.globalState, GLOBAL_MQ_MSG_MAX);
            const softSize = Atomics.load(this.globalState, GLOBAL_MQ_MSGSIZE_MAX);
            const maxQueues = Atomics.load(this.globalState, GLOBAL_MQ_QUEUES_MAX);
            if (this.posixMqByName.size >= maxQueues) { this.lockReply(slot, [-28]); break; } // ENOSPC
            const maxmsg = m.hasAttr ? m.maxmsg : softMsg;
            const msgsize = m.hasAttr ? m.msgsize : softSize;
            if (maxmsg <= 0 || msgsize <= 0) { this.lockReply(slot, [-22]); break; } // EINVAL
            // above the soft limits only a privileged process may go, and never above the hard ones.
            if (maxmsg > MQ_HARD_MSGMAX || msgsize > MQ_HARD_MSGSIZEMAX) { this.lockReply(slot, [-22]); break; }
            if ((maxmsg > softMsg || msgsize > softSize) && m.uid !== 0) {
              this.lockReply(slot, [-1]); break;                                  // EPERM
            }
            const id = this.nextMqId++;
            const q: PosixMq = {
              id, name, mode: m.mode & 0o7777, uid: m.uid, gid: m.gid,
              maxmsg, msgsize, msgs: [], seq: 0, unlinked: false,
              holders: new Set([m.pid]), notify: null,
            };
            this.posixMq.set(id, q);
            this.posixMqByName.set(name, id);
            this.lockReply(slot, [id, maxmsg, msgsize]);
            break;
          }
          case 'ref': {
            const q = mqOf(m.id);
            if (q) q.holders.add(m.pid);
            this.lockReply(slot, [0]);
            break;
          }
          case 'close': {
            // sent when the process's LAST descriptor onto this queue goes away, so it also ends
            // any notification it had registered: there is nothing left to notify it through.
            const q = mqOf(m.id);
            if (q) {
              q.holders.delete(m.pid);
              if (q.notify && (q.notify.pid === m.pid || q.notify.tgid === m.pid)) {
                this.sendMqCookie(q.notify, NOTIFY_REMOVED); q.notify = null;
              }
              reap(q);
            }
            this.lockReply(slot, [0]);
            break;
          }
          case 'unlink': {
            const id = this.posixMqByName.get(m.name);
            if (id === undefined) { this.lockReply(slot, [-2]); break; }          // ENOENT
            const q = this.posixMq.get(id)!;
            // removing a name needs write permission on what it names, as unlink does everywhere.
            if (!ipcPermitted(q.mode, q.uid, q.gid, m.uid, m.gid, 0o200)) {
              this.lockReply(slot, [-13]); break;                                 // EACCES
            }
            this.posixMqByName.delete(m.name);
            q.unlinked = true;
            reap(q);
            this.lockReply(slot, [0]);
            break;
          }
          case 'send': {
            const q = mqOf(m.id);
            if (!q) { this.lockReply(slot, [-9]); break; }                        // EBADF
            if (m.len > q.msgsize) { this.lockReply(slot, [-90]); break; }        // EMSGSIZE
            if (q.msgs.length >= q.maxmsg) { this.lockReply(slot, [-11]); break; } // EAGAIN: full
            const off = m.len > 0 ? this.shmAlloc.alloc(m.len) : 0;
            if (m.len > 0 && off < 0) { this.lockReply(slot, [-12]); break; }     // ENOMEM
            if (m.len > 0) new Uint8Array(this.shmArena, off, m.len).set(m.data);
            const wasEmpty = q.msgs.length === 0;
            q.msgs.push({ prio: m.prio | 0, seq: q.seq++, off, len: m.len });
            // the point of mq_notify: the process that registered is told when the queue goes from
            // empty to not, and the registration is spent in the telling.
            if (wasEmpty && q.notify) {
              const target = q.notify;
              q.notify = null;
              if (target.ring !== undefined) {
                // SIGEV_THREAD: the caller's own 32-byte cookie goes back to it on the netlink
                // socket libc is reading, with the last byte saying which of the two things
                // happened. libc runs the notify function only for NOTIFY_WOKENUP.
                this.sendMqCookie(target, NOTIFY_WOKENUP);
                bumpIoGeneration();
              } else if (target.signo > 0) {
                this.postSignal(target.pid, target.signo);
              }
            }
            bumpIoGeneration();
            this.lockReply(slot, [0]);
            break;
          }
          case 'recv': {
            const q = mqOf(m.id);
            if (!q) { this.lockReply(slot, [-9]); break; }
            // the buffer must be able to hold ANY message the queue could carry, whatever is
            // actually in it -- that check comes before looking at whether one is.
            if (m.maxlen < q.msgsize) { this.lockReply(slot, [-90]); break; }     // EMSGSIZE
            if (q.msgs.length === 0) { this.lockReply(slot, [-11]); break; }      // EAGAIN
            let best = 0;
            for (let i = 1; i < q.msgs.length; i++) {
              const a = q.msgs[i], b = q.msgs[best];
              if (a.prio > b.prio || (a.prio === b.prio && a.seq < b.seq)) best = i;
            }
            const e = q.msgs.splice(best, 1)[0];
            bumpIoGeneration();  // a blocked sender may now have room
            this.lockReply(slot, [0, e.off, e.len, e.prio]);
            break;
          }
          case 'free':
            if (m.len > 0) this.shmAlloc.release(m.off, m.len);
            this.lockReply(slot, [0]);
            break;
          case 'getattr': {
            const q = mqOf(m.id);
            if (!q) { this.lockReply(slot, [-9]); break; }
            this.lockReply(slot, [0, q.maxmsg, q.msgsize, q.msgs.length]);
            break;
          }
          case 'notify': {
            const q = mqOf(m.id);
            if (!q) { this.lockReply(slot, [-9]); break; }
            if (m.signo < 0) {                       // deregistering: only the registrant may
              // "the registrant" is the PROCESS, not the thread: musl registers a SIGEV_THREAD
              // notification from its helper thread and unregisters from the one that asked, and
              // comparing thread ids meant the queue stayed registered and the next attempt was
              // EBUSY forever.
              const g = Atomics.load(this.procTable, procSlotOffset(slot) / 4 + PROC_TGID) || pid;
              if (q.notify && (q.notify.tgid === g || q.notify.pid === m.pid)) {
                // the registration was made by a helper thread that is still sitting on its socket
                // waiting to hear. it hears that it was removed, and goes away.
                this.sendMqCookie(q.notify, NOTIFY_REMOVED);
                bumpIoGeneration();
                q.notify = null;
              }
              this.lockReply(slot, [0]);
              break;
            }
            if (q.notify) { this.lockReply(slot, [-16]); break; }                 // EBUSY
            // a SIGEV_THREAD registration names a netlink socket by descriptor. the caller may be
            // a thread that never saw it opened, so the ring is resolved here rather than there.
            if (typeof m.nlFd === 'number' && m.nlFd >= 0 && !(m.ring >= 0)) {
              const g = Atomics.load(this.procTable, procSlotOffset(slot) / 4 + PROC_TGID) || pid;
              const found = this.nlRings.get(`${g}:${m.nlFd | 0}`);
              if (found === undefined) { this.lockReply(slot, [-22]); break; }  // EINVAL
              m.ring = found;
            }
            q.notify = {
              pid: m.pid, tgid: Atomics.load(this.procTable, procSlotOffset(slot) / 4 + PROC_TGID) || pid,
              signo: m.signo | 0,
              ring: typeof m.ring === 'number' && m.ring >= 0 ? (m.ring | 0) : undefined,
              cookie: m.cookie ? new Uint8Array(m.cookie) : undefined,
            };
            this.lockReply(slot, [0]);
            break;
          }
          default:
            this.lockReply(slot, [-22]);
        }
        break;
      }

      case 'sysv_msg': {
        const m = msg as any;
        const now = () => this.nowSec();
        // a queue is protected the way a file is: sending needs write, receiving and reading its
        // state need read, and changing or removing it is the owner's alone.
        const msgMay = (q: SysvMsgQueue, want: number) =>
          ipcPermitted(q.mode, q.uid, q.gid, m.uid ?? 0, m.gid ?? 0, want);
        const msgOwns = (q: SysvMsgQueue) => (m.uid ?? 0) === 0 || (m.uid ?? 0) === q.uid;
        switch (m.op) {
          case 'get': {
            const IPC_CREAT = 0o1000, IPC_EXCL = 0o2000, IPC_PRIVATE = 0;
            const key = m.key | 0;
            if (key !== IPC_PRIVATE) {
              const existing = this.sysvMsgByKey.get(key);
              if (existing !== undefined) {
                if (m.flg & IPC_EXCL) { this.lockReply(slot, [-17]); break; }   // EEXIST
                // joining an existing queue is subject to its permissions, exactly as opening a
                // file is: the low bits of msgflg are the access the caller is asking for.
                const q = this.sysvMsg.get(existing)!;
                if (!ipcPermitted(q.mode, q.uid, q.gid, m.uid, m.gid, m.flg & 0o777)) {
                  this.lockReply(slot, [-13]); break;                            // EACCES
                }
                this.lockReply(slot, [existing]);
                break;
              }
              if (!(m.flg & IPC_CREAT)) { this.lockReply(slot, [-2]); break; }  // ENOENT
            }
            // there is a limit on how many queues may exist, /proc/sys/kernel/msgmni names it, and
            // programs that make queues in a loop stop when the kernel says ENOSPC. saying so in
            // the file and never enforcing it would make the number a decoration.
            const msgmni = Atomics.load(this.globalState, GLOBAL_MSGMNI) || MSGMNI;
            if (this.sysvMsg.size >= msgmni) { this.lockReply(slot, [-28]); break; } // ENOSPC
            // ...and kernel.msg_next_id names the id the next queue must get.
            const wantMsg = Atomics.exchange(this.globalState, GLOBAL_MSG_NEXT_ID, -1);
            let id: number;
            if (wantMsg >= 0 && !this.sysvMsg.has(wantMsg)) id = wantMsg;
            else { id = this.nextMsgId; this.nextMsgId += IPCMNI; }
            this.sysvMsg.set(id, {
              id, key, mode: m.flg & 0o777, uid: m.uid, gid: m.gid, cpid: pid,
              lspid: 0, lrpid: 0, removed: false, stime: 0, rtime: 0, ctime: now(),
              qbytes: MSGMNB, msgs: [],
            });
            if (key !== IPC_PRIVATE) this.sysvMsgByKey.set(key, id);
            this.lockReply(slot, [id]);
            break;
          }
          case 'snd': {
            const q = this.sysvMsg.get(m.id | 0);
            if (!q) { this.lockReply(slot, [-22]); break; } // EINVAL
            if (!msgMay(q, 0o200)) { this.lockReply(slot, [-13]); break; }      // EACCES
            const used = q.msgs.reduce((n, x) => n + x.len, 0);
            // a full queue blocks the sender, or answers EAGAIN when it asked not to wait. the
            // worker owns the retry, so the answer here is only ever the state right now.
            if (used + m.len > q.qbytes) { this.lockReply(slot, [-11]); break; } // EAGAIN
            const off = m.len > 0 ? this.shmAlloc.alloc(m.len) : 0;
            if (m.len > 0 && off < 0) { this.lockReply(slot, [-105]); break; }   // ENOBUFS
            new Uint8Array(this.shmArena, off, m.len).set(m.data);
            q.msgs.push({ mtype: BigInt(m.mtype), off, len: m.len });
            q.lspid = pid;
            q.stime = now();
            // a receiver parked waiting for a message has to be told one arrived.
            bumpIoGeneration();
            this.lockReply(slot, [0]);
            break;
          }
          case 'rcv': {
            const q = this.sysvMsg.get(m.id | 0);
            if (!q) { this.lockReply(slot, [-22]); break; }
            if (!msgMay(q, 0o400)) { this.lockReply(slot, [-13]); break; }      // EACCES
            const want = BigInt(m.mtype);
            const MSG_EXCEPT = 0o20000;
            // msgtyp selects: 0 takes the first message; positive takes the first of that type;
            // negative takes the lowest type that is <= |msgtyp|; MSG_EXCEPT inverts a positive.
            let idx = -1;
            if (want === 0n) idx = q.msgs.length ? 0 : -1;
            else if (want > 0n && (m.flg & MSG_EXCEPT)) idx = q.msgs.findIndex((x) => x.mtype !== want);
            else if (want > 0n) idx = q.msgs.findIndex((x) => x.mtype === want);
            else {
              const limit = -want;
              let best = -1;
              for (let i = 0; i < q.msgs.length; i++) {
                if (q.msgs[i].mtype > limit) continue;
                if (best < 0 || q.msgs[i].mtype < q.msgs[best].mtype) best = i;
              }
              idx = best;
            }
            if (idx < 0) { this.lockReply(slot, [-11]); break; }                 // EAGAIN: nothing matches
            const msgEntry = q.msgs[idx];
            const MSG_NOERROR = 0o10000;
            if (msgEntry.len > m.maxlen && !(m.flg & MSG_NOERROR)) { this.lockReply(slot, [-7]); break; } // E2BIG
            q.msgs.splice(idx, 1);
            q.lrpid = pid;
            q.rtime = now();
            bumpIoGeneration(); // a blocked sender may now have room
            // hand back where the bytes are; the worker copies them and frees the block.
            this.lockReply(slot, [0, msgEntry.off, msgEntry.len,
                                  Number(msgEntry.mtype & 0xffffffffn),
                                  Number((msgEntry.mtype >> 32n) & 0xffffffffn)]);
            break;
          }
          // MSG_COPY: read the message at an INDEX without taking it out of the queue. it exists
          // for checkpoint/restore, which dumps a queue and has to put it back exactly as it was --
          // so the message stays where it is and the bytes are handed over in place.
          case 'copy': {
            const cq = this.sysvMsg.get(m.id | 0);
            if (!cq) { this.lockReply(slot, [-22]); break; }
            if (!msgMay(cq, 0o400)) { this.lockReply(slot, [-13]); break; }     // EACCES
            const at = Number(m.mtype);
            if (at < 0 || at >= cq.msgs.length) { this.lockReply(slot, [-42]); break; } // ENOMSG
            const copyOf = cq.msgs[at];
            const MSG_NOERROR_C = 0o10000;
            if (copyOf.len > m.maxlen && !(m.flg & MSG_NOERROR_C)) { this.lockReply(slot, [-7]); break; } // E2BIG
            this.lockReply(slot, [0, copyOf.off, copyOf.len,
                                  Number(copyOf.mtype & 0xffffffffn),
                                  Number((copyOf.mtype >> 32n) & 0xffffffffn)]);
            break;
          }
          case 'free':
            if (m.len > 0) this.shmAlloc.release(m.off, m.len);
            this.lockReply(slot, [0]);
            break;
          case 'stat':
          case 'statidx': {
            // MSG_STAT reads by INDEX into the kernel's queue array rather than by id, and hands
            // the id back as its return value. the array is this map's insertion order.
            const q = m.op === 'stat'
              ? this.sysvMsg.get(m.id | 0)
              : [...this.sysvMsg.values()][m.id | 0];
            if (!q) { this.lockReply(slot, [-22]); break; }
            if (!msgMay(q, 0o400)) { this.lockReply(slot, [-13]); break; }      // EACCES
            const used = q.msgs.reduce((n, x) => n + x.len, 0);
            this.lockReply(slot, [0, q.key, q.uid, q.gid, q.mode, used, q.msgs.length,
                                  q.qbytes, q.lspid, q.lrpid, q.stime, q.rtime, q.ctime, q.id]);
            break;
          }
          case 'info': {
            // how many queues exist and how much they hold, for IPC_INFO / MSG_INFO.
            let msgs = 0, bytes = 0;
            for (const q of this.sysvMsg.values()) {
              msgs += q.msgs.length;
              bytes += q.msgs.reduce((n, x) => n + x.len, 0);
            }
            this.lockReply(slot, [0, this.sysvMsg.size, msgs, bytes, MSGMNB]);
            break;
          }
          case 'set': {
            const q = this.sysvMsg.get(m.id | 0);
            if (!q) { this.lockReply(slot, [-22]); break; }
            if (!msgOwns(q)) { this.lockReply(slot, [-1]); break; }             // EPERM
            q.mode = m.mode & 0o7777; q.uid = m.newUid | 0; q.gid = m.newGid | 0;
            if (m.qbytes > 0) q.qbytes = m.qbytes;
            q.ctime = now();
            this.lockReply(slot, [0]);
            break;
          }
          case 'rm': {
            const q = this.sysvMsg.get(m.id | 0);
            if (!q) { this.lockReply(slot, [-22]); break; }
            if (!msgOwns(q)) { this.lockReply(slot, [-1]); break; }             // EPERM
            // unlike a shared segment, a removed queue is gone AT ONCE: anyone blocked on it gets
            // EIDRM, and the messages still in it are dropped.
            q.removed = true;
            for (const x of q.msgs) if (x.len > 0) this.shmAlloc.release(x.off, x.len);
            q.msgs.length = 0;
            if (q.key !== 0) this.sysvMsgByKey.delete(q.key);
            this.sysvMsg.delete(q.id);
            bumpIoGeneration(); // wake anyone parked on it so they can see it is gone
            this.lockReply(slot, [0]);
            break;
          }
          default: this.lockReply(slot, [-22]);
        }
        break;
      }

      // SysV semaphores. the set's values live in the kernel because semop is atomic across the
      // whole array: either every operation in the call applies or none does, and only a serial
      // view can decide that. blocking is the worker's job -- it retries until the kernel says the
      // operations fit -- so every answer here describes the state right now and nothing else.
      case 'sysv_sem': {
        const m = msg as any;
        const now = () => this.nowSec();
        const R = 0o400, W = 0o200;
        const EINVAL = -22, EACCES = -13, EIDRM = -43, EPERM = -1, ERANGE = -34, EAGAIN = -11;
        // an id that names nothing is EINVAL. EIDRM is for the caller who was ASLEEP on the set
        // when it was removed, and only the worker (which owns the sleeping) can tell that apart.
        const lookup = (id: number): SysvSemSet | number => this.sysvSem.get(id | 0) ?? EINVAL;
        const may = (st: SysvSemSet, want: number) =>
          ipcPermitted(st.mode, st.uid, st.gid, m.uid, m.gid, want);
        // SEM_UNDO belongs to the thread GROUP, not the thread: pthread_create passes
        // CLONE_SYSVSEM, so every thread of a process shares one undo list and it is taken back
        // when the LAST of them exits. keying it by thread made a posting thread's adjustment
        // undo itself the moment that thread returned, which is exactly the deadlock semop05
        // exists to catch.
        const group = Atomics.load(this.procTable, procSlotOffset(slot) / 4 + PROC_TGID) || pid;
        // IPC_SET and IPC_RMID are the owner's to make, not merely a writer's.
        const owns = (st: SysvSemSet) => m.uid === 0 || m.uid === st.uid || m.uid === st.cuid;

        switch (m.op) {
          case 'get': {
            const IPC_CREAT = 0o1000, IPC_EXCL = 0o2000, IPC_PRIVATE = 0;
            const key = m.key | 0, nsems = m.nsems | 0;
            if (nsems < 0 || nsems > SEMMSL) { this.lockReply(slot, [EINVAL]); break; }
            if (key !== IPC_PRIVATE) {
              const existing = this.sysvSemByKey.get(key);
              if (existing !== undefined) {
                if (m.flg & IPC_EXCL) { this.lockReply(slot, [-17]); break; }   // EEXIST
                const st = this.sysvSem.get(existing)!;
                // asking for more semaphores than the set has is EINVAL, and asking for none is
                // "however many it already has".
                if (nsems > st.nsems) { this.lockReply(slot, [EINVAL]); break; }
                if (!may(st, (m.flg & 0o777))) { this.lockReply(slot, [EACCES]); break; }
                this.lockReply(slot, [existing]);
                break;
              }
              if (!(m.flg & IPC_CREAT)) { this.lockReply(slot, [-2]); break; }  // ENOENT
            }
            // creating a set with no semaphores in it is meaningless, and linux refuses it.
            if (nsems === 0) { this.lockReply(slot, [EINVAL]); break; }
            if (this.sysvSem.size >= SEMMNI) { this.lockReply(slot, [-28]); break; } // ENOSPC
            // ...and kernel.sem_next_id names the id the next set must get.
            const wantSem = Atomics.exchange(this.globalState, GLOBAL_SEM_NEXT_ID, -1);
            let id: number;
            if (wantSem >= 0 && !this.sysvSem.has(wantSem)) id = wantSem;
            else { id = this.nextSemId; this.nextSemId += IPCMNI; }
            this.sysvSem.set(id, {
              id, key, mode: m.flg & 0o777, uid: m.uid, gid: m.gid, cuid: m.uid, cgid: m.gid,
              nsems, vals: new Int32Array(nsems), pids: new Int32Array(nsems),
              otime: 0, ctime: now(), waiting: new Map(),
            });
            if (key !== IPC_PRIVATE) this.sysvSemByKey.set(key, id);
            this.lockReply(slot, [id]);
            break;
          }

          case 'op': {
            const st = lookup(m.id);
            if (typeof st === 'number') { this.clearSemWait(pid, m.id | 0); this.lockReply(slot, [st]); break; }
            const ops: number[] = m.ops;                   // flat triples: sem_num, sem_op, sem_flg
            const n = ops.length / 3;
            // a call that only reads (every sem_op zero) needs read permission; anything that
            // changes a value needs write.
            let alters = false;
            for (let i = 0; i < n; i++) if (ops[i * 3 + 1] !== 0) { alters = true; break; }
            if (!may(st, alters ? W : R)) { this.clearSemWait(pid, st.id); this.lockReply(slot, [EACCES]); break; }
            let outOfRange = false;
            for (let i = 0; i < n; i++) {
              const num = ops[i * 3];
              if (num < 0 || num >= st.nsems) { outOfRange = true; break; }
            }
            if (outOfRange) { this.clearSemWait(pid, st.id); this.lockReply(slot, [-27]); break; } // EFBIG

            const SEM_UNDO = 0o10000;
            const undo = this.semUndo.get(group)?.get(st.id);
            // trial run against copies: the whole call applies or none of it does, so nothing is
            // written until every operation is known to fit.
            const next = Int32Array.from(st.vals);
            const nextAdj = undo ? Int32Array.from(undo) : null;
            let blocked = -1, err = 0;
            for (let i = 0; i < n && err === 0 && blocked < 0; i++) {
              const num = ops[i * 3], op = ops[i * 3 + 1], flg = ops[i * 3 + 2];
              if (op === 0) { if (next[num] !== 0) blocked = i; continue; }
              const r = next[num] + op;
              if (r < 0) { blocked = i; continue; }
              if (r > SEMVMX) { err = ERANGE; continue; }
              if (flg & SEM_UNDO) {
                const adj = (nextAdj ? nextAdj[num] : 0) - op;
                if (adj > SEMAEM || adj < -SEMAEM - 1) { err = ERANGE; continue; }
              }
              next[num] = r;
            }
            if (err !== 0) { this.clearSemWait(pid, st.id); this.lockReply(slot, [err]); break; }
            if (blocked >= 0) {
              // cannot proceed. IPC_NOWAIT is per-OPERATION, and the one that matters is the one
              // that blocked, so the decision belongs here where that is known.
              const IPC_NOWAIT = 0o4000;
              if (ops[blocked * 3 + 2] & IPC_NOWAIT) {
                this.clearSemWait(pid, st.id);
                this.lockReply(slot, [EAGAIN, 1]);
                break;
              }
              // the caller may wait. record what it is waiting FOR, since that is precisely what
              // GETNCNT and GETZCNT are asked to report.
              if (m.wait) {
                let w = st.waiting.get(pid);
                if (!w) { w = { dec: new Set(), zero: new Set() }; st.waiting.set(pid, w); }
                w.dec.clear(); w.zero.clear();
                const num = ops[blocked * 3];
                if (ops[blocked * 3 + 1] === 0) w.zero.add(num); else w.dec.add(num);
              }
              this.lockReply(slot, [EAGAIN, 0]);
              break;
            }
            // it fits: commit the values, the undo adjustments and sempid together.
            this.clearSemWait(pid, st.id);
            st.vals.set(next);
            if (nextAdj) {
              for (let i = 0; i < n; i++) {
                if (!(ops[i * 3 + 2] & SEM_UNDO)) continue;
                nextAdj[ops[i * 3]] -= ops[i * 3 + 1];
              }
              undo!.set(nextAdj);
            } else {
              let created: Int32Array | null = null;
              for (let i = 0; i < n; i++) {
                if (!(ops[i * 3 + 2] & SEM_UNDO)) continue;
                if (!created) {
                  created = new Int32Array(st.nsems);
                  let per = this.semUndo.get(group);
                  if (!per) { per = new Map(); this.semUndo.set(group, per); }
                  per.set(st.id, created);
                }
                created[ops[i * 3]] -= ops[i * 3 + 1];
              }
            }
            for (let i = 0; i < n; i++) st.pids[ops[i * 3]] = pid;
            st.otime = now();
            // someone parked waiting for one of these values may now be able to proceed.
            bumpIoGeneration();
            this.lockReply(slot, [0]);
            break;
          }

          case 'unwait':
            this.clearSemWait(pid, m.id | 0);
            this.lockReply(slot, [0]);
            break;

          case 'getone': {
            const st = lookup(m.id);
            if (typeof st === 'number') { this.lockReply(slot, [st]); break; }
            if (!may(st, R)) { this.lockReply(slot, [EACCES]); break; }
            const num = m.num | 0;
            if (num < 0 || num >= st.nsems) { this.lockReply(slot, [EINVAL]); break; }
            let v = 0;
            if (m.what === 'val') v = st.vals[num];
            else if (m.what === 'pid') v = st.pids[num];
            else {
              let c = 0;
              for (const w of st.waiting.values()) if ((m.what === 'ncnt' ? w.dec : w.zero).has(num)) c++;
              v = c;
            }
            this.lockReply(slot, [v]);
            break;
          }

          case 'setval': {
            const st = lookup(m.id);
            if (typeof st === 'number') { this.lockReply(slot, [st]); break; }
            if (!may(st, W)) { this.lockReply(slot, [EACCES]); break; }
            const num = m.num | 0, val = m.val | 0;
            if (num < 0 || num >= st.nsems) { this.lockReply(slot, [EINVAL]); break; }
            if (val < 0 || val > SEMVMX) { this.lockReply(slot, [ERANGE]); break; }
            st.vals[num] = val;
            st.pids[num] = pid;
            st.ctime = now();
            // SETVAL discards every pending undo for that semaphore: the adjustments were raised
            // against a value that no longer exists.
            for (const per of this.semUndo.values()) { const a = per.get(st.id); if (a) a[num] = 0; }
            bumpIoGeneration();
            this.lockReply(slot, [0]);
            break;
          }

          case 'getall': {
            const st = lookup(m.id);
            if (typeof st === 'number') { this.lockReply(slot, [st]); break; }
            if (!may(st, R)) { this.lockReply(slot, [EACCES]); break; }
            // the values go through the shared arena rather than the mailbox: a set may hold tens
            // of thousands of them and a mailbox holds thirty.
            const len = st.nsems * 2;
            const off = this.shmAlloc.alloc(len);
            if (off < 0) { this.lockReply(slot, [-105]); break; }   // ENOBUFS
            const out = new Uint16Array(this.shmArena, off, st.nsems);
            for (let i = 0; i < st.nsems; i++) out[i] = st.vals[i];
            this.lockReply(slot, [0, off, len]);
            break;
          }

          case 'setall': {
            const st = lookup(m.id);
            if (typeof st === 'number') { this.lockReply(slot, [st]); break; }
            if (!may(st, W)) { this.lockReply(slot, [EACCES]); break; }
            const vals: number[] = m.vals;
            if (vals.length !== st.nsems) { this.lockReply(slot, [EINVAL]); break; }
            let outOfRange = false;
            for (const v of vals) if (v < 0 || v > SEMVMX) { outOfRange = true; break; }
            if (outOfRange) { this.lockReply(slot, [ERANGE]); break; }
            for (let i = 0; i < st.nsems; i++) { st.vals[i] = vals[i]; st.pids[i] = pid; }
            st.ctime = now();
            for (const per of this.semUndo.values()) { const a = per.get(st.id); if (a) a.fill(0); }
            bumpIoGeneration();
            this.lockReply(slot, [0]);
            break;
          }

          case 'free':
            if (m.len > 0) this.shmAlloc.release(m.off, m.len);
            this.lockReply(slot, [0]);
            break;

          case 'stat':
          case 'statidx': {
            // SEM_STAT reads by INDEX into the kernel's array of sets rather than by id, and hands
            // the id back as its return value. the array is this map's insertion order.
            const st = m.op === 'stat' ? lookup(m.id) : ([...this.sysvSem.values()][m.id | 0] ?? EINVAL);
            if (typeof st === 'number') { this.lockReply(slot, [st]); break; }
            if (!may(st, R)) { this.lockReply(slot, [EACCES]); break; }
            this.lockReply(slot, [0, st.key, st.uid, st.gid, st.mode, st.nsems,
                                  st.otime, st.ctime, st.id, st.cuid, st.cgid]);
            break;
          }

          // how many semaphores the set holds. SETALL needs this to know how much of the
          // caller's array to read, and it is not a permission question: linux reads the size
          // from its own record long before it decides whether the caller may write.
          case 'nsems': {
            const st = lookup(m.id);
            this.lockReply(slot, [typeof st === 'number' ? st : st.nsems]);
            break;
          }

          // /proc/sysvipc/{sem,msg,shm}. these are the only way a program can ask what ipc
          // objects exist without owning them -- ipcs(1) reads them, and so does anything that
          // needs to know how close the system is to its limits.
          case 'proclist': {
            const pad = (v: string | number, w: number) => String(v).padStart(w);
            let text: string;
            if (m.which === 'sem') {
              text = '       key      semid perms      nsems   uid   gid  cuid  cgid      otime      ctime\n';
              for (const st of this.sysvSem.values()) {
                text += pad(st.key, 10) + ' ' + pad(st.id, 10) + ' ' + pad(st.mode.toString(8), 5) + ' ' +
                        pad(st.nsems, 10) + ' ' + pad(st.uid, 5) + ' ' + pad(st.gid, 5) + ' ' +
                        pad(st.cuid, 5) + ' ' + pad(st.cgid, 5) + ' ' +
                        pad(st.otime, 10) + ' ' + pad(st.ctime, 10) + '\n';
              }
            } else if (m.which === 'msg') {
              text = '       key      msqid perms      cbytes       qnum lspid lrpid   uid   gid  cuid  cgid      stime      rtime      ctime\n';
              for (const q of this.sysvMsg.values()) {
                const used = q.msgs.reduce((n, x) => n + x.len, 0);
                text += pad(q.key, 10) + ' ' + pad(q.id, 10) + ' ' + pad(q.mode.toString(8), 5) + ' ' +
                        pad(used, 11) + ' ' + pad(q.msgs.length, 10) + ' ' + pad(q.lspid, 5) + ' ' +
                        pad(q.lrpid, 5) + ' ' + pad(q.uid, 5) + ' ' + pad(q.gid, 5) + ' ' +
                        pad(q.uid, 5) + ' ' + pad(q.gid, 5) + ' ' + pad(q.stime, 10) + ' ' +
                        pad(q.rtime, 10) + ' ' + pad(q.ctime, 10) + '\n';
              }
            } else {
              text = '       key      shmid perms       size  cpid  lpid nattch   uid   gid  cuid  cgid      atime      dtime      ctime        rss       swap\n';
              for (const seg of this.sysvShm.values()) {
                text += pad(seg.key, 10) + ' ' + pad(seg.id, 10) + ' ' + pad(seg.mode.toString(8), 5) + ' ' +
                        pad(seg.size, 10) + ' ' + pad(seg.cpid, 5) + ' ' + pad(seg.lpid, 5) + ' ' +
                        pad(shmNattch(seg), 6) + ' ' + pad(seg.uid, 5) + ' ' + pad(seg.gid, 5) + ' ' +
                        pad(seg.uid, 5) + ' ' + pad(seg.gid, 5) + ' ' + pad(seg.atime, 10) + ' ' +
                        pad(seg.dtime, 10) + ' ' + pad(seg.ctime, 10) + ' ' + pad(seg.size, 10) + ' 0\n';
              }
            }
            const buf = new TextEncoder().encode(text);
            const off = this.shmAlloc.alloc(buf.length);
            if (off < 0) { this.lockReply(slot, [-105]); break; }   // ENOBUFS
            new Uint8Array(this.shmArena, off, buf.length).set(buf);
            this.lockReply(slot, [0, off, buf.length]);
            break;
          }

          case 'info': {
            let sems = 0;
            for (const st of this.sysvSem.values()) sems += st.nsems;
            this.lockReply(slot, [0, this.sysvSem.size, sems]);
            break;
          }

          case 'set': {
            const st = lookup(m.id);
            if (typeof st === 'number') { this.lockReply(slot, [st]); break; }
            if (!owns(st)) { this.lockReply(slot, [EPERM]); break; }
            st.uid = m.nuid; st.gid = m.ngid; st.mode = m.mode & 0o777;
            st.ctime = now();
            this.lockReply(slot, [0]);
            break;
          }

          case 'rm': {
            const st = lookup(m.id);
            if (typeof st === 'number') { this.lockReply(slot, [st]); break; }
            if (!owns(st)) { this.lockReply(slot, [EPERM]); break; }
            // a removed set is gone at once: anyone parked on it gets EIDRM on their next poll.
            if (st.key !== 0) this.sysvSemByKey.delete(st.key);
            this.sysvSem.delete(st.id);
            for (const per of this.semUndo.values()) per.delete(st.id);
            bumpIoGeneration();
            this.lockReply(slot, [0]);
            break;
          }

          default: this.lockReply(slot, [EINVAL]);
        }
        break;
      }

      case 'shm_alloc': {
        // a file-backed mapping carries a key (its path, offset and length), so two processes that
        // map the same file independently -- never having forked from one another -- land on the
        // same block and genuinely share it. an anonymous mapping has no such identity: it is only
        // ever shared by inheritance, so each one gets a block of its own.
        const key = msg.key;
        const existing = key !== undefined ? this.shmByKey.get(key) : undefined;
        if (existing !== undefined && this.shmRefs.has(existing)) {
          this.holdShm(pid, existing);
          this.lockReply(slot, [existing, 0]); // 0: joined, do not re-seed from our copy
          break;
        }
        const off = this.shmAlloc.alloc(msg.len);
        if (off >= 0) {
          this.shmRefs.set(off, { len: msg.len, refs: 0, key });
          this.holdShm(pid, off);   // takes the first reference
          if (key !== undefined) this.shmByKey.set(key, off);
        }
        this.lockReply(slot, [off, 1]); // 1: freshly created, seed it
        break;
      }

      case 'pipe_ref':
        this.notePipeRef(pid, msg.off, msg.mode, msg.delta);
        break;

      case 'pipe_async':
        this.setPipeAsync(pid, msg.off, msg.mode, msg.sig, msg.on);
        break;

      case 'pipe_io':
        this.deliverPipeIo(msg.off, msg.mode);
        break;

      case 'trace':
        this.traceLog.push(msg.line);
        if (this.traceLog.length > 60000) this.traceLog.shift();
        break;

      case 'refusal': {
        const key = `${msg.nr}/${msg.errno}`;
        this.refusals.set(key, (this.refusals.get(key) ?? 0) + 1);
        break;
      }

      case 'shm_free':
        this.dropShm(pid, msg.off);
        this.lockReply(slot, [0]);
        break;

      case 'unix_bind': {
        if (this.unixNames.has(msg.path)) { this.lockReply(slot, [-98]); break; } // EADDRINUSE
        const wantsInbox = !!(msg as any).dgram;
        let inbox = -1;
        if (wantsInbox) {
          this.reclaimDeadPipes();
          inbox = this.allocPipeRing(this.pipePagesForNew(false), pid);
          if (inbox < 0) { this.lockReply(slot, [-23]); break; } // ENFILE: no arena left
        }
        this.unixNames.set(msg.path, { pid, uid: msg.uid | 0, gid: msg.gid | 0, listening: false,
                                      dgram: wantsInbox, inbox: inbox >= 0 ? inbox : undefined,
                                      backlog: [], holders: new Set([pid]) });
        this.lockReply(slot, [0, inbox]);
        break;
      }

      case 'unix_unbind': {
        const named = this.unixNames.get(msg.path);
        if (named) {
          named.holders.delete(pid);
          if (named.holders.size === 0) {
            if (named.inbox !== undefined) this.releasePipeRing(named.inbox);
            this.unixNames.delete(msg.path);
          }
        }
        this.lockReply(slot, [0]);
        break;
      }

      // where to send a datagram addressed by name. nothing is created and nothing is queued: the
      // sender writes into the ring the receiver bound, which is what makes a datagram socket
      // reachable by anyone who knows its name rather than only by whoever connected to it.
      case 'unix_lookup': {
        const target = this.unixNames.get(msg.path);
        if (!target || target.inbox === undefined) { this.lockReply(slot, [-111]); break; } // ECONNREFUSED
        if (!target.dgram) { this.lockReply(slot, [-91]); break; }                          // EPROTOTYPE
        this.lockReply(slot, [0, target.inbox, target.pid, target.uid, target.gid]);
        break;
      }

      case 'unix_listen': {
        const l = this.unixNames.get(msg.path);
        if (!l) { this.lockReply(slot, [-22]); break; } // EINVAL: listen before bind
        l.listening = true;
        this.lockReply(slot, [0]);
        break;
      }

      case 'unix_connect': {
        const target = this.unixNames.get(msg.path);
        // nothing bound there, or bound but not listening: linux says ECONNREFUSED either way.
        if (!target || !target.listening) { this.lockReply(slot, [-111]); break; }
        this.reclaimDeadPipes();
        const pages = this.pipePagesForNew(false);
        const a = this.allocPipeRing(pages, pid);
        const b = a >= 0 ? this.allocPipeRing(pages, pid) : -1;
        if (a < 0 || b < 0) {
          if (a >= 0) this.releasePipeRing(a);
          this.lockReply(slot, [-23]); // ENFILE: no arena left for the streams
          break;
        }
        // who holds which end, so that killing either side closes exactly the ends it owned. the
        // connecting side writes `a` and reads `b`; the far end is the mirror, and until somebody
        // accepts it that end belongs to the process that is listening.
        this.setPipeHolder(a, pid, 0, 1);
        this.setPipeHolder(b, pid, 1, 0);
        this.setPipeHolder(a, target.pid, 1, 0);
        this.setPipeHolder(b, target.pid, 0, 1);
        target.backlog.push({ a, b, pid, uid: msg.uid | 0, gid: msg.gid | 0 });
        // the listener's descriptor is now readable, which is how poll/accept report a waiting
        // connection. a listener parked in poll has to be told, or it sleeps through its client.
        bumpIoGeneration();
        // the connecting side writes into `a` and reads from `b`; the accepting side is the mirror.
        this.lockReply(slot, [0, a, b, target.pid, target.uid, target.gid]);
        break;
      }

      // socketpair(2): two connected endpoints with no name in the filesystem at all. it needs
      // exactly what connect+accept produce -- a pair of rings, crossed -- so it allocates them
      // directly and hands both back to the one caller.
      // a netlink socket, which for us exists so that mq_notify(SIGEV_THREAD) can work: libc opens
      // one, parks a thread reading it, and the kernel writes a cookie into it when the queue it
      // was told about becomes readable. only that one direction is ever used.
      case 'nl_socket': {
        this.reclaimDeadPipes();
        const off = this.allocPipeRing(1, pid);
        if (off < 0) { this.lockReply(slot, [-23]); break; }   // ENFILE
        // which descriptor of which THREAD GROUP this ring belongs to. mq_notify may be called
        // from a different thread than the one that opened the socket -- musl's does exactly that
        // -- and a thread's own view of its descriptors lives in its worker, not here.
        const nlGroup = Atomics.load(this.procTable, procSlotOffset(slot) / 4 + PROC_TGID) || pid;
        this.nlRings.set(`${nlGroup}:${(msg as any).fd | 0}`, off);
        this.lockReply(slot, [0, off]);
        break;
      }

      case 'unix_pair': {
        this.reclaimDeadPipes();
        const pages = this.pipePagesForNew(false);
        const a = this.allocPipeRing(pages, pid);
        const b = a >= 0 ? this.allocPipeRing(pages, pid) : -1;
        if (a < 0 || b < 0) {
          if (a >= 0) this.releasePipeRing(a);
          this.lockReply(slot, [-23]); // ENFILE: no arena left for the streams
          break;
        }
        this.lockReply(slot, [0, a, b]);
        break;
      }

      // how many connections are queued, WITHOUT taking one. poll on a listening socket has to
      // answer "is a connection waiting" without consuming it; accept-and-put-back would race.
      case 'unix_pending': {
        const l = this.unixNames.get(msg.path);
        this.lockReply(slot, [0, l ? l.backlog.length : 0]);
        break;
      }

      case 'unix_accept': {
        const listener = this.unixNames.get(msg.path);
        if (!listener) { this.lockReply(slot, [-22]); break; } // EINVAL: not bound
        const conn = listener.backlog.shift();
        if (!conn) { this.lockReply(slot, [-11]); break; } // EAGAIN: nobody waiting
        // the accepted end passes from whoever was listening to whoever accepted -- the same
        // process most of the time, a forked child of it otherwise.
        if (pid !== listener.pid) {
          this.setPipeHolder(conn.a, listener.pid, 0, 0);
          this.setPipeHolder(conn.b, listener.pid, 0, 0);
        }
        this.setPipeHolder(conn.a, pid, 1, 0);
        this.setPipeHolder(conn.b, pid, 0, 1);
        // the connecting side was told (tx=a, rx=b); the accepting side is the SAME pair read the
        // other way round -- it reads `a` (what the client writes) and writes `b`. handing back
        // (rx=b, tx=a) instead crosses the streams and both ends block reading their own writes.
        this.lockReply(slot, [0, conn.a, conn.b, conn.pid, conn.uid, conn.gid]);
        break;
      }

      case 'lock_close':
        this.releaseLocksOnClose(pid, msg.path, msg.ofdId, msg.lastRef);
        this.lockReply(slot, [0]);
        break;

      case 'lock_cancel':
        this.cancelLockWaiter(pid);
        break;

      case 'bridge_call':
        this.handleBridgeCall(pid, slot, msg);
        break;

      case 'jit_compiled':
        // TODO: cache compiled modules for sharing with future forks
        break;

      default:
        console.warn(`[kernel] Unknown message from Worker ${pid}:`, msg);
    }
  }

  /**
   * Can the system afford one more process?
   *
   * Below WORKER_FLOOR the answer is always yes: the accounting is coarse (workers report their
   * memory in 8 MiB steps) and a shell that cannot fork is worse than a slightly optimistic
   * estimate. Above it, the child is charged what its parent has actually committed -- a fork
   * copies the parent's touched pages, so that is the honest prediction -- and the fork is refused
   * if that would put the total past what jsc will give us. EAGAIN is what linux answers when a
   * fork cannot get memory, so the guest already knows what to do with it.
   */
  private forkFitsBudget(parentPid: number): boolean {
    if (this.workers.size < WORKER_FLOOR) return true;
    const cost = Math.max(this.committed.get(parentPid) ?? 0, FORK_COST_FLOOR);
    return this.wasmBytesInUse() + cost <= RESERVATION_BUDGET;
  }

  /** a removed segment lives until the last attach goes; then its arena block goes with it. */
  private dropSysvIfDone(seg: SysvSeg): void {
    if (!seg.removed || shmNattch(seg) > 0) return;
    this.sysvShm.delete(seg.id);
    // the arena block is refcounted by shm_alloc/shm_free like any shared mapping; with nobody
    // attached there is nothing left holding it, so only the registry entry has to go here.
  }

  private replyForkEagain(parentSlot: number): void {
    const mb = new Int32Array(this.sab, mailboxSlotOffset(parentSlot), MAILBOX_SLOT_SIZE / 4);
    Atomics.store(mb, MBOX_VALUE0, -11); // -EAGAIN
    Atomics.store(mb, MBOX_READY, 1);
    Atomics.notify(mb, MBOX_READY);
  }

  private handleFork(parentPid: number, parentSlot: number, msg: KernelMessage & { type: 'fork' }): void {
    // bound concurrency, all surfaced as fork() EAGAIN (never a crash): the hard 64-slot proc
    // table; the live-Worker ceiling WORKER_CAP (pid 1 EXEMPT so the shell never stalls); and the
    // hard total-Worker ceiling POOL_MAX (a fork can only proceed by reusing an idle Worker or, if
    // none is idle, creating one while under POOL_MAX). pid 1's exemption from WORKER_CAP still
    // bottoms out at POOL_MAX, so total memory stays bounded.
    const childSlot = this.findFreeSlot();
    if (childSlot < 0 ||
        (parentPid !== 1 && this.workers.size >= WORKER_CAP) ||
        (parentPid !== 1 && !this.forkFitsBudget(parentPid)) ||
        (this.idleWorkers.length === 0 && this.workers.size >= POOL_MAX)) {
      this.replyForkEagain(parentSlot);
      return;
    }
    const childPid = this.allocPid();
    const parentBase = procSlotOffset(parentSlot) / 4;
    const parentStdin = Atomics.load(this.procTable, parentBase + PROC_STDIN_RING);
    // a child of a detached run belongs to that run: its output is captured with the rest.
    const owner = this.detachedOwner.get(parentPid);
    if (owner !== undefined) this.detachedOwner.set(childPid, owner);
    // the child maps everything the parent had mapped, shared regions included
    this.inheritShm(childPid, (msg as any).shmBlocks);
    // the child's fd table is a copy of the parent's, so it holds exactly what the parent held.
    // the parent already moved the RINGS' refcounts on the child's behalf (synchronously, before
    // fork() returned); this records who those new references belong to, so they can be dropped if
    // the child dies without closing them.
    this.inheritPipeRefs(parentPid, childPid);
    // a unix socket the parent bound is now held by the child too: the name survives until both
    // are gone, exactly as the socket does.
    for (const entry of this.unixNames.values()) {
      if (entry.holders.has(parentPid)) entry.holders.add(childPid);
    }
    this.inheritMqForFork(parentPid, childPid);
    this.inheritSysvShm(parentPid, childPid);
    // an inotify descriptor survives fork, and both halves read the same queue -- so the instance
    // gains a holder and must not disappear when only one of them closes.
    for (const id of ((msg as any).inotifyIds ?? []) as number[]) {
      const w = this.inoWatchers.get(id);
      if (w) w.refs++;
    }
    this.spawnProcess({
      pid: childPid,
      ppid: parentPid,
      pgid: Atomics.load(this.procTable, parentBase + PROC_PGID),
      sid: Atomics.load(this.procTable, parentBase + PROC_SID),
      nice: Atomics.load(this.procTable, parentBase + PROC_NICE),
      sched: this.schedStateOf(parentBase),
      slot: childSlot,
      stdinRing: parentStdin >= 0 ? parentStdin : undefined,
      elfBytes: this.elfBytes,
      argv: [],
      envp: [],
      memSnapshot: msg.memSnapshot,
      shmMaps: (msg as any).shmMaps,
      protArmed: (msg as any).protArmed,
      syscallState: msg.syscallState,
      kernelPipesSnapshot: (msg as any).kernelPipesSnapshot,
      unixSocksSnapshot: (msg as any).unixSocksSnapshot,
      cloneFlags: (msg as any).cloneFlags,
      childStack: (msg as any).childStack,
      childTls: (msg as any).childTls,
      childCtid: (msg as any).childCtid,
    });
    // the staging block passes to the child with the pages in it: the child frees it when it has
    // copied them out, and its exit frees it if it never gets that far. the parent is not made to
    // wait on either -- a parent blocked on its own child is a deadlock waiting for a reason.
    const staged = this.forkStages.get(parentPid);
    if (staged && msg.memSnapshot?.stageOff === staged.off) {
      this.forkStages.delete(parentPid);
      this.forkStages.set(childPid, staged);
    }
    const mailbox = new Int32Array(this.sab, mailboxSlotOffset(parentSlot), MAILBOX_SLOT_SIZE / 4);
    Atomics.store(mailbox, MBOX_VALUE0, childPid);
    Atomics.store(mailbox, MBOX_READY, 1);
    Atomics.notify(mailbox, MBOX_READY);
  }

  // SIGKILL enforcement. a worker blocked in Atomics.wait (pipe read, bridge mailbox, ...)
  // cannot observe a signal bit and self-terminate, so the kernel force-terminates it here.
  // reuse the normal exit teardown (zombie + reap-wake) with a signalled exit status; the
  // RUNNING guard makes a racing self-exit idempotent.
  /**
   * Post a signal for a process to handle ITSELF.
   *
   * Not the same thing as handleKillSignal, which ends the process then and there. That one exists
   * for SIGKILL, which is uncatchable by definition and has to be enforced by the kernel because a
   * process parked in a futex will never run guest code to notice it. Every OTHER signal belongs to
   * the target: it may have a handler, it may have the signal blocked, it may be ignoring it. Ending
   * the process on its behalf takes all of that away -- which is exactly what happened to SIGIO,
   * delivered to a process that had installed a perfectly good handler and was killed anyway.
   */
  private postSignal(pid: number, sig: number, info?: QueuedSignal): void {
    if (sig === 9) { this.handleKillSignal(pid, sig); return; }
    for (let s = 0; s < PROC_TABLE_SLOTS; s++) {
      const base = procSlotOffset(s) / 4;
      if (Atomics.load(this.procTable, base + PROC_PID) !== pid) continue;
      // STOPPED counts as alive: a stopped process still has a slot, a worker and a future, and
      // the whole point of SIGCONT is that it reaches one.
      const st = Atomics.load(this.procTable, base + PROC_STATE);
      if (st !== PROC_RUNNING && st !== PROC_STOPPED) return;
      // every signal number, with the siginfo the caller meant. a real-time signal used to have
      // no bit to set in the old word and was dropped on the floor.
      if (sig > 0 && sig <= 64) sigqPush(this.procTable, s, info ?? { signo: sig, code: 0x80, pid: 0, uid: 0, value: 0n });
      // wake it if it is parked, so a blocked process observes the signal instead of sleeping past it
      Atomics.store(this.procTable, base + PROC_WAIT_FUTEX, 1);
      Atomics.notify(this.procTable, base + PROC_WAIT_FUTEX);
      return;
    }
  }

  private handleKillSignal(pid: number, sig: number): void {
    for (let s = 0; s < PROC_TABLE_SLOTS; s++) {
      const base = procSlotOffset(s) / 4;
      if (Atomics.load(this.procTable, base + PROC_PID) !== pid) continue;
      const killState = Atomics.load(this.procTable, base + PROC_STATE);
      if (killState !== PROC_RUNNING && killState !== PROC_STOPPED) return;
      this.handleExit(pid, s, 0x10000 | sig); // signalled death -> WIFSIGNALED for the parent
      return;
    }
  }

  /** bytes of address space we are currently holding in shared memories (live groups + parked spare).
   *  a shared memory reserves its `maximum` at creation and is only released when it is dropped AND
   *  collected, so this is the number that decides whether another threaded process can start. */
  /** everything drawing on jsc's single wasm allowance: the shared reservations plus what the
   *  single-threaded processes have actually committed. this is the honest figure; the tier
   *  chooser below still budgets on reservedBytes() alone, which is deliberately left alone for
   *  now because tightening it would refuse allocations that currently succeed, and that belongs
   *  in its own change with its own conformance run. */
  /**
   * What the process is really carrying, in bytes.
   *
   * `committed` holds each live process's FOOTPRINT rather than its grown buffer size, because
   * wasm growth is lazy and the buffer size overstates the cost several-fold -- charging it
   * refused a hundred-child fork that the platform would have accepted without noticing.
   */
  wasmBytesInUse(): number {
    let n = this.reservedBytes();
    for (const [pid, bytes] of this.committed) if (this.workers.has(pid)) n += bytes;
    // a parked Worker still owns its memory. leaving the idle pool out of the accounting meant the
    // budget was systematically optimistic by however much the pool happened to be holding.
    for (const b of this.idleBytes.values()) n += b;
    return n;
  }

  private reservedBytes(): number {
    let n = this.spareSharedMem ? this.spareSharedMem.mem.buffer.byteLength : 0;
    for (const m of this.sharedMems.values()) n += m.buffer.byteLength;
    return n;
  }

  /** allocate a threaded process's shared memory within the platform's reservation budget, reclaiming
   *  first rather than discovering the ceiling by being jetsam-killed. returns null -> EAGAIN, which is
   *  what a real kernel returns when a thread can't be created. */
  /** choose the smallest tier that holds the process with headroom AND fits the remaining budget.
   *  a shared memory's maximum can't be raised later, so this is the process's permanent ceiling. */
  // how much address space a threaded process gets, for the rest of its life.
  //
  // the temptation is to be generous: a thread costs its STACK in address space (8 MiB by default),
  // so 256 MiB runs out at about twenty-seven of them and pthread_create starts failing with EAGAIN.
  // but a shared wasm memory is COMMITTED at its maximum rather than grown into, so a bigger tier is
  // paid for in real memory the moment the first thread is created -- and at 512 MiB, a threaded test
  // that forks left the app with no room to build the child's memory at all. so: the smallest tier
  // that holds the image with room to grow, and a thread-heavy process is bounded by that.
  private pickThreadTier(curPages: number): ThreadLayout {
    const footprint = curPages * 65536;
    const free = RESERVATION_BUDGET - this.reservedBytes();
    for (const t of THREAD_TIERS) {
      if (t.regsBase >= footprint * 2 && t.pages * 65536 <= free) return t; // room to grow, affordable
    }
    for (const t of THREAD_TIERS) {
      if (t.regsBase > footprint && t.pages * 65536 <= free) return t;      // at least holds the image
    }
    return THREAD_TIERS[0]; // nothing fits; allocSharedMem's budget check turns this into EAGAIN
  }

  private allocSharedMem(layout: ThreadLayout): WebAssembly.Memory | null {
    const need = layout.pages * 65536;
    if (this.reservedBytes() + need > RESERVATION_BUDGET) {
      // reclaim what we can: idle pooled Workers still own their (never-shrinking) memories. keep one
      // parked for launch latency unless we are still short, in which case pressure wins over warmth.
      while (this.idleWorkers.length > 1) { const w = this.idleWorkers.pop()!; this.idleBytes.delete(w); w.terminate(); }
      if (this.reservedBytes() + need > RESERVATION_BUDGET) {
        while (this.idleWorkers.length) { const w = this.idleWorkers.pop()!; this.idleBytes.delete(w); w.terminate(); }
        if (this.reservedBytes() + need > RESERVATION_BUDGET) {
          // nothing left to reclaim from the pool: ask the idlest process to freeze. it answers at
          // its next safe point rather than now, so this attempt still fails -- which is what a
          // real kernel does under reclaim pressure, and the retry finds the room.
          if (this.swapPolicy) this.swapOut(this.pickSwapVictim());
          return null;
        }
      }
    }
    try {
      return new WebAssembly.Memory({ initial: layout.pages, maximum: layout.pages, shared: true });
    } catch {
      return null; // the platform said no even though our accounting allowed it — stay honest
    }
  }

  // a thread-group LEADER dying == the whole process dies, so kill every remaining thread of the group
  // and release the group's shared memory (a wasm memory reserves address space until it is dropped;
  // leaking one costs a whole threaded-process slot for the app's lifetime). returns true if this pid
  // owned a thread group, in which case its Worker must be destroyed rather than pooled — the Worker
  // holds the shared-memory reference. MUST run on every leader exit path, including detached runs.
  private tearDownThreadGroup(pid: number, threadedHigh: number): boolean {
    if (!this.sharedMems.has(pid)) return false;
    for (let s = 0; s < PROC_TABLE_SLOTS; s++) {
      const b = procSlotOffset(s) / 4;
      const tp = Atomics.load(this.procTable, b + PROC_PID);
      if (tp === pid || tp === 0) continue;
      if (Atomics.load(this.procTable, b + PROC_TGID) !== pid) continue;
      this.forgetProcess(tp);
      const tw = this.workers.get(tp);
      if (tw) { this.workers.delete(tp); tw.terminate(); }
      Atomics.store(this.procTable, b + PROC_PID, 0);
      Atomics.store(this.procTable, b + PROC_STATE, PROC_EMPTY);
    }
    this.threadCounts.delete(pid);
    this.freeSharedMem(pid, threadedHigh);
    return true;
  }

  // `poolable` is true only for a clean self-exit (the worker posted 'exit' and its runLoop
  // returned — it is now idle and reusable). a kill or crash leaves the Worker dead/mid-execution,
  // so those pass false and the Worker is destroyed instead of pooled.
  private handleExit(pid: number, slot: number, code: number, poolable = false, threadedHigh = 0,
                     memBytes = 0): void {
    this.lastActive.delete(pid);
    this.committed.delete(pid);
    this.reclaimNativeFds(pid);
    const slotBase = procSlotOffset(slot) / 4;

    // a detached run (/ctl/run) has no parent waiting on it: hand its captured output + status to the
    // caller and free the slot here rather than leaving a zombie nobody will ever reap. only the run's
    // OWNER finishes it; its children take the normal path below (their parent reaps them via wait4).
    this.detachedOwner.delete(pid); // exited: nothing more to attribute to the run
    const cap = this.detached.get(pid);
    if (cap) {
      this.detached.delete(pid);
      this.releaseLocksForPid(pid);
      this.releaseMqForPid(pid);
      this.releaseForkStage(pid);
      this.releaseSysvShm(pid);
      this.clearSemWaiters(pid);
      this.releaseSemUndo(pid);
      // a detached run can be threaded too: without this its shared memory leaked on every run and
      // threading died after a handful of them.
      if (this.tearDownThreadGroup(pid, threadedHigh)) poolable = false;
      const w = this.workers.get(pid);
      if (w) { this.workers.delete(pid); if (poolable) this.park(w, memBytes); else w.terminate(); }
      Atomics.store(this.procTable, slotBase + PROC_PID, 0);
      Atomics.store(this.procTable, slotBase + PROC_STATE, PROC_EMPTY);
      cap.done({ output: decodeChunks(cap.chunks), code: (code & 0x10000) ? 128 + (code & 0x7f) : code });
      return;
    }

    // drop every advisory record lock this process held so a blocked F_SETLKW waiter (in another
    // process) can be granted, and so a stale lock never wedges a later same-path locker.
    this.releaseLocksForPid(pid);
    this.releaseMqForPid(pid);
    this.releaseForkStage(pid);
    this.releaseSysvShm(pid);
    this.clearSemWaiters(pid);

    const tgid = Atomics.load(this.procTable, slotBase + PROC_TGID) || pid;
    // the undo list is the thread group's. a leader that still has live threads leaves it for
    // whichever of them goes last (the branch below), which is what linux does.
    if (tgid === pid && !this.threadCounts.has(pid)) this.releaseSemUndo(pid);

    // a non-leader thread exiting is NOT a process exit: no zombie, no SIGCHLD to the parent, no
    // reparenting. pthread_join blocks on the CLONE_CHILD_CLEARTID futex (zeroed + woken by the
    // thread-Worker's postExit), not wait4. decrement the group; if it was the last thread AND the
    // leader is already gone, free the shared memory. then free the proc slot + retire the worker.
    if (tgid !== pid) {
      // ...unless it died of a fatal SIGNAL. a signal that is not handled terminates the whole
      // THREAD GROUP, not the thread that took it: a segfault anywhere is the process's death.
      // treating it as an ordinary thread exit made a faulting thread vanish silently, leaving
      // the leader to join a thread that had crashed and then carry on as if nothing happened.
      if (code & 0x10000) {
        const leaderSlot = this.slotOfPid(tgid);
        if (leaderSlot >= 0) {
          this.handleExit(tgid, leaderSlot, code, false, threadedHigh);
        }
        const tw = this.workers.get(pid);
        if (tw) { this.workers.delete(pid); tw.terminate(); }
        Atomics.store(this.procTable, slotBase + PROC_PID, 0);
        Atomics.store(this.procTable, slotBase + PROC_STATE, PROC_EMPTY);
        return;
      }
      if (this.threadCounts.has(tgid)) {
        const n = (this.threadCounts.get(tgid) ?? 1) - 1;
        if (n <= 0) {
          this.threadCounts.delete(tgid);
          this.freeSharedMem(tgid, threadedHigh);
          this.releaseSemUndo(tgid);
        }
        else this.threadCounts.set(tgid, n);
      }
      const w = this.workers.get(pid);
      if (w) { this.workers.delete(pid); w.terminate(); }
      Atomics.store(this.procTable, slotBase + PROC_PID, 0);
      Atomics.store(this.procTable, slotBase + PROC_STATE, PROC_EMPTY);
      return;
    }

    // a thread-group LEADER dying == the whole process dies (a returning/exiting main thread runs
    // exit_group). so kill every remaining thread of the group and free the shared memory now, rather
    // than waiting on a join that will never come. the leader's own Worker holds the shared-memory
    // reference, so it must be terminated (not pooled) or the ~1.5 GiB reservation lingers.
    if (this.tearDownThreadGroup(pid, threadedHigh)) poolable = false;

    // a vforked parent is waiting on this child; whatever ended it, it is not coming back.
    if (Atomics.exchange(this.procTable, slotBase + PROC_VFORK, 0) === 1) {
      Atomics.notify(this.procTable, slotBase + PROC_VFORK);
    }
    // Mark as zombie in process table
    Atomics.store(this.procTable, slotBase + PROC_STATE, PROC_ZOMBIE);
    Atomics.store(this.procTable, slotBase + PROC_EXIT_CODE, code);
    // a process ending is an i/o event: a pidfd held on it becomes readable, and whoever is parked
    // in poll waiting for exactly that has to be woken to see it.
    bumpIoGeneration();

    // Wake any parent waiting on this child
    Atomics.store(this.procTable, slotBase + PROC_WAIT_FUTEX, 1);
    Atomics.notify(this.procTable, slotBase + PROC_WAIT_FUTEX);

    // Also wake the parent's wait_futex (the parent waits on its OWN slot's futex,
    // checking children's states). We need to notify on the parent's slot too.
    const ppid = Atomics.load(this.procTable, slotBase + PROC_PPID);
    // a non-leader thread ending is not a child death: it raises no SIGCHLD and is not waitable.
    const exitTgid = Atomics.load(this.procTable, slotBase + PROC_TGID);
    const isThread = exitTgid !== 0 && exitTgid !== pid;
    for (let s = 0; s < PROC_TABLE_SLOTS; s++) {
      const sBase = procSlotOffset(s) / 4;
      const parentState = Atomics.load(this.procTable, sBase + PROC_STATE);
      if (Atomics.load(this.procTable, sBase + PROC_PID) === ppid &&
          (parentState === PROC_RUNNING || parentState === PROC_STOPPED)) {
        // A DEAD CHILD RAISES SIGCHLD IN ITS PARENT. this was simply missing: the parent was
        // woken, which is enough for a wait4 that polls, and nothing else. a shell's `wait`
        // builtin does not poll -- it sigsuspends until SIGCHLD arrives -- so `cmd & wait` hung
        // forever, and so would anything else that reaps from a handler.
        //
        // siginfo for SIGCHLD is _sifields._sigchld: pid, uid, then the exit STATUS where a
        // kill's sigval sits. si_code says which kind of end it was.
        if (isThread) {
          Atomics.store(this.procTable, sBase + PROC_WAIT_FUTEX, 1);
          Atomics.notify(this.procTable, sBase + PROC_WAIT_FUTEX);
          break;
        }
        const exitedBySignal = (code & 0x10000) !== 0;
        const cldCode = !exitedBySignal ? 1 /* CLD_EXITED */
                      : ((code & 0x20000) ? 3 /* CLD_DUMPED */ : 2 /* CLD_KILLED */);
        const status = exitedBySignal ? (code & 0x7f) : (code & 0xff);
        sigqPush(this.procTable, s, {
          signo: 17, code: cldCode, pid,
          uid: Atomics.load(this.procTable, procCredOffset(slot) / 4 + 1),
          value: BigInt(status),
        });
        // Wake the parent in case it's blocked in wait4
        Atomics.store(this.procTable, sBase + PROC_WAIT_FUTEX, 1);
        Atomics.notify(this.procTable, sBase + PROC_WAIT_FUTEX);
        break;
      }
    }

    // a tracer that dies lets its tracees go: the ptrace attachment is undone and anything stopped
    // for it is put back on its feet, or it would stay stopped for a process that no longer exists.
    for (let s2 = 0; s2 < PROC_TABLE_SLOTS; s2++) {
      const b = procSlotOffset(s2) / 4;
      if (Atomics.load(this.procTable, b + PROC_TRACER) !== pid) continue;
      Atomics.store(this.procTable, b + PROC_TRACER, 0);
      if (Atomics.load(this.procTable, b + PROC_STATE) === PROC_STOPPED) {
        Atomics.store(this.procTable, b + PROC_STOP_SIG, 0);
        Atomics.store(this.procTable, b + PROC_TRACE_CMD, PT_RESUME_DETACH);
        Atomics.store(this.procTable, b + PROC_WAIT_FUTEX, 1);
        Atomics.notify(this.procTable, b + PROC_WAIT_FUTEX);
      }
    }

    // Reparent this process's children to init (pid 1) and flag them orphaned. our init/shell
    // is not a reaper, so a reparented process's eventual zombie would leak its proc slot; the
    // flag makes the kernel auto-reap it. a child already zombied (was waiting on us to reap it)
    // is swept now, since nobody else will. without this, a killed subtree's descendants pile up
    // in the 64-slot table until fork() starves — exactly what wedged the ltp sweep.
    const orphanSignals: Array<{ pid: number; sig: number }> = [];
    // orphans go to the nearest ancestor that asked for them (PR_SET_CHILD_SUBREAPER), and only
    // to init when there is none. that is the whole point of a subreaper: a service manager keeps
    // its descendants' descendants rather than losing them to pid 1.
    const reaper = this.nearestSubreaper(pid);
    for (let s = 0; s < PROC_TABLE_SLOTS; s++) {
      const b = procSlotOffset(s) / 4;
      if (Atomics.load(this.procTable, b + PROC_PID) === 0) continue;
      if (Atomics.load(this.procTable, b + PROC_PPID) !== pid) continue;
      // a child that asked to hear about this gets its parent-death signal, before it is told it
      // now belongs to init. collected first and delivered after the loop, because delivery can
      // end the target and rewrite the table underneath us.
      const death = Atomics.load(this.procTable, b + PROC_PDEATHSIG);
      const orphanState = Atomics.load(this.procTable, b + PROC_STATE);
      if (death > 0 && (orphanState === PROC_RUNNING || orphanState === PROC_STOPPED)) {
        orphanSignals.push({ pid: Atomics.load(this.procTable, b + PROC_PID), sig: death });
      }
      Atomics.store(this.procTable, b + PROC_PPID, reaper);
      // only init auto-reaps; a real subreaper is expected to wait(2) for what it adopted.
      Atomics.store(this.procTable, b + PROC_REPARENTED, reaper === 1 ? 1 : 0);
      if (Atomics.load(this.procTable, b + PROC_STATE) === PROC_ZOMBIE) this.reapSlot(s);
    }
    for (const o of orphanSignals) this.postSignal(o.pid, o.sig);

    // Retire the Worker. a cleanly-exited one is idle and reusable -> park it in the idle pool
    // (its next init() wipes + resets it); the idle-trim timer bounds the pool during calm. a
    // killed/crashed one is destroyed. this is the crux: no terminate on the clean-exit hot path,
    // so a fork storm leaves no dead Workers for GC to chase.
    const worker = this.workers.get(pid);
    if (worker) {
      this.workers.delete(pid);
      if (poolable) this.park(worker, memBytes);
      else worker.terminate();
    }

    // If PID 1 (init) exits, all processes are dead
    if (pid === 1) {
      // Terminate all remaining workers, live and idle
      for (const [p, w] of this.workers) w.terminate();
      this.workers.clear();
      for (const w of this.idleWorkers) w.terminate();
      this.idleWorkers.length = 0;
      return;
    }

    // If we were ourselves an orphan (reparented to init, which won't wait4 us), don't leave a
    // zombie slot to leak — the kernel reaps on init's behalf.
    if (Atomics.load(this.procTable, slotBase + PROC_REPARENTED) === 1) this.reapSlot(slot);
  }

  // Free a proc-table slot outright (kernel-side reap of an orphan zombie). terminates any
  // lingering worker for that pid, then clears the slot so fork() can reuse it.
  private reapSlot(slot: number): void {
    const base = procSlotOffset(slot) / 4;
    const pid = Atomics.load(this.procTable, base + PROC_PID);
    if (pid === 0) return;
    this.forgetProcess(pid);
    const w = this.workers.get(pid);
    if (w) { w.terminate(); this.workers.delete(pid); }
    Atomics.store(this.procTable, base + PROC_STATE, PROC_EMPTY);
    Atomics.store(this.procTable, base + PROC_PID, 0);
    Atomics.store(this.procTable, base + PROC_PPID, 0);
    Atomics.store(this.procTable, base + PROC_REPARENTED, 0);
  }

  private handlePipeAlloc(pid: number, slot: number, privileged: boolean): void {
    // reclaim first, then allocate. pipe rings used to be recycled in place from a pool that only
    // ever grew: the pool ended up owning the high-water mark of concurrently-open pipes for the
    // rest of the session, and since the arena is shared with MAP_SHARED, a burst of pipes
    // permanently took memory away from everything else. handing blocks back to the arena means a
    // burst costs only while it lasts. runs on the single-threaded main loop, so
    // reclaim-then-allocate is atomic with respect to other pipe requests.
    this.reclaimDeadPipes();
    let pipeSlot = this.allocPipeRing(this.pipePagesForNew(privileged), pid);
    // no room for the size we wanted: a smaller pipe is far better than a failed one, and it is
    // what linux falls back to as well.
    if (pipeSlot < 0 && this.pipePagesForNew(privileged) !== PIPE_MIN_PAGES) pipeSlot = this.allocPipeRing(PIPE_MIN_PAGES, pid);
    // pipeSlot === -1 means the arena is genuinely exhausted; the worker maps that to ENFILE.
    const mailbox = new Int32Array(this.sab, mailboxSlotOffset(slot), MAILBOX_SLOT_SIZE / 4);
    Atomics.store(mailbox, MBOX_VALUE0, pipeSlot);
    Atomics.store(mailbox, MBOX_READY, 1);
    Atomics.notify(mailbox, MBOX_READY);
  }

  // ── fcntl advisory record locking ─────────────────────────────────────────
  // reply words land in the mailbox starting at MBOX_VALUE0. for F_GETLK a conflict is reported
  // as [0, type, pid, start_lo, start_hi, len_lo, len_hi]; for set/unlock it is [0] or [-errno].
  /** give back the arena block a read handed out, if it has not been reclaimed already. */
  private releaseInoBlock(w: { lastBlock?: { off: number; len: number } }): void {
    if (!w.lastBlock) return;
    this.shmAlloc.release(w.lastBlock.off, w.lastBlock.len);
    w.lastBlock = undefined;
  }

  /** publish how many watches exist, so a mutating syscall can skip the report when none do. */
  private bumpInotifyCount(delta: number): void {
    if (delta !== 0) Atomics.add(this.globalState, GLOBAL_INOTIFY_WATCHES, delta);
  }

  /**
   * hand an event to every watcher that asked for it.
   *
   * a watch on a DIRECTORY hears about its immediate children, and the event carries the child's
   * name; a watch on the file itself hears about the file, with no name. linux distinguishes the
   * two with the *_SELF masks, and so does this.
   */
  private deliverInotify(path: string, mask: number, cookie: number, isDir = false,
                         selfOnly = false, moved = path, unlinked = false): void {
    const slash = path.lastIndexOf('/');
    const parent = slash <= 0 ? '/' : path.slice(0, slash);
    const name = path.slice(slash + 1);
    const IN_ISDIR = 0x40000000;
    // a watch follows the object, not the name it was reached by. when the watched thing is renamed
    // out from under it, the registry has to move with it or every later event misses.
    if ((mask & 0x800) && moved !== path) this.retargetWatches(path, moved);
    // which watch an event can reach depends on what the event is ABOUT. a directory's watch hears
    // about its children being created, deleted or moved; the object's own watch hears about the
    // object being deleted or moved. everything else (access, modify, attrib, open, close) reaches
    // both -- named from the directory, unnamed from the object.
    const CHILD_ONLY = 0x100 /* CREATE */ | 0x200 /* DELETE */ | 0x040 /* MOVED_FROM */ | 0x080 /* MOVED_TO */;
    const SELF_ONLY = 0x400 /* DELETE_SELF */ | 0x800 /* MOVE_SELF */;
    let woke = false;
    for (const w of this.inoWatchers.values()) {
      for (const [wd, watch] of w.watches) {
        let evMask = 0, evName = '';
        // IN_ISDIR is not something a watcher can ask for or filter out: it describes the object the
        // event is about, so it survives the mask and is added back after it.
        if (watch.path === path) evMask = mask & watch.mask & ~CHILD_ONLY;
        // IN_EXCL_UNLINK: a directory watch set with it stops hearing about children whose name has
        // been removed, even though a descriptor somewhere still holds the object open.
        else if (watch.path === parent && !selfOnly && !(unlinked && (watch.mask & 0x04000000))) {
          evMask = mask & watch.mask & ~SELF_ONLY; evName = name;
        }
        if (evMask === 0) continue;
        // IN_DELETE_SELF and IN_MOVE_SELF never carry it: the object is already identified by the
        // watch descriptor, so linux says nothing more about it.
        if (isDir && !(evMask & SELF_ONLY)) evMask |= IN_ISDIR;
        // the queue is bounded: linux drops events and reports IN_Q_OVERFLOW once rather than
        // growing without limit behind a reader that never reads.
        if (w.queue.length >= INOTIFY_MAX_QUEUED) {
          if (!w.overflowed) { w.overflowed = true; w.queue.push({ wd: -1, mask: 0x4000, cookie: 0, name: '' }); }
          continue;
        }
        w.overflowed = false;
        // an event identical to the one already at the tail is folded into it. a watcher wants to
        // know a thing happened, and reporting the same unread thing twice is noise, not detail.
        const tail = w.queue[w.queue.length - 1];
        if (tail && tail.wd === wd && tail.mask === evMask && tail.cookie === cookie && tail.name === evName) continue;
        w.queue.push({ wd, mask: evMask, cookie, name: evName });
        // a watch on an object that has just gone away is over: linux retires it and says so with
        // IN_IGNORED, which is how a watcher learns its descriptor is dead rather than merely quiet.
        if (evMask & 0x400) {
          w.watches.delete(wd);
          this.bumpInotifyCount(-1);
          w.queue.push({ wd, mask: 0x8000 /* IN_IGNORED */, cookie: 0, name: '' });
        } else if (watch.mask & 0x80000000) {
          // IN_ONESHOT retires the watch as soon as it has fired once.
          w.watches.delete(wd);
          this.bumpInotifyCount(-1);
        }
        woke = true;
      }
    }
    if (woke) bumpIoGeneration();
  }

  /** follow a renamed object: every watch on it, or on anything inside it, moves to the new name. */
  private retargetWatches(from: string, to: string): void {
    for (const w of this.inoWatchers.values()) {
      for (const watch of w.watches.values()) {
        if (watch.path === from) watch.path = to;
        else if (watch.path.startsWith(from + '/')) watch.path = to + watch.path.slice(from.length);
      }
    }
  }

  /**
   * pack queued events into an arena block for the reader to copy out.
   *
   * struct inotify_event is { s32 wd; u32 mask; u32 cookie; u32 len; char name[len]; } and the
   * name is NUL-padded to a multiple of 16 so the next event stays aligned. a read whose buffer
   * cannot hold even the first event is EINVAL, not a short read.
   */
  private packInotifyEvents(queue: Array<{ wd: number; mask: number; cookie: number; name: string }>,
                            maxlen: number): { off: number; len: number } | null {
    const enc = new TextEncoder();
    const parts: Uint8Array[] = [];
    let total = 0;
    while (queue.length > 0) {
      const e = queue[0];
      const nameBytes = e.name ? enc.encode(e.name) : new Uint8Array(0);
      const nameLen = nameBytes.length === 0 ? 0 : (nameBytes.length + 1 + 15) & ~15;
      const size = 16 + nameLen;
      if (total + size > maxlen) break;
      const rec = new Uint8Array(size);
      const dv = new DataView(rec.buffer);
      dv.setInt32(0, e.wd, true);
      dv.setUint32(4, e.mask >>> 0, true);
      dv.setUint32(8, e.cookie >>> 0, true);
      dv.setUint32(12, nameLen, true);
      if (nameLen > 0) rec.set(nameBytes, 16);
      parts.push(rec);
      total += size;
      queue.shift();
    }
    if (total === 0) return null; // the buffer cannot hold even one event
    const off = this.shmAlloc.alloc(total);
    if (off < 0) return null;
    const out = new Uint8Array(this.shmArena, off, total);
    let at = 0;
    for (const part of parts) { out.set(part, at); at += part.length; }
    return { off, len: total };
  }

  private lockReply(slot: number, words: number[]): void {
    const mb = new Int32Array(this.sab, mailboxSlotOffset(slot), MAILBOX_SLOT_SIZE / 4);
    for (let i = 0; i < words.length; i++) Atomics.store(mb, MBOX_VALUE0 + i, words[i] | 0);
    Atomics.store(mb, MBOX_READY, 1);
    Atomics.notify(mb, MBOX_READY);
  }

  /**
   * fcntl record locks and flock, arbitrated in the one place that sees every process. the request
   * carries its own ownership scope: a classic posix lock is owned by the process, an F_OFD_* lock
   * and an flock are owned by the open file description, and the two scopes conflict with each
   * other even inside a single process.
   */
  private handleLeaseReq(pid: number, slot: number, msg: KernelMessage & { type: 'lease_req' }): void {
    if (msg.op === 'get') { this.lockReply(slot, [this.leases.get(pid, msg.nativeFd)]); return; }
    if (msg.op === 'sig') { this.leases.setBreakSignal(pid, msg.nativeFd, msg.breakSig); this.lockReply(slot, [0]); return; }
    this.lockReply(slot, [this.leases.set(pid, msg.nativeFd, msg.path, msg.arg, msg.breakSig, msg.isOwner, msg.isRegular)]);
  }

  private handleLockReq(pid: number, slot: number, msg: KernelMessage & { type: 'lock_req' }): void {
    const { path, cmd, ltype, start, len } = msg;
    const space: LockSpace = (msg as any).space === 'flock' ? 'flock' : 'record';
    const scope: LockScope = (msg as any).scope === 'ofd' ? 'ofd' : 'posix';
    const owner = { scope, id: scope === 'ofd' ? ((msg as any).ownerId ?? pid) : pid, pid };
    const kind: LockKind = ltype === 2 ? 'unlock' : ltype === 1 ? 'write' : 'read';
    const req = { space, file: path, owner, kind, start, end: len <= 0 ? EOF_END : start + len };

    if (cmd === 5 || cmd === 12 || cmd === 36) { // F_GETLK / F_OFD_GETLK: report, take nothing
      const c = this.locks.test(req);
      if (!c) { this.lockReply(slot, [0, 2]); return; } // F_UNLCK: the region is free
      const clen = c.end === EOF_END ? 0 : c.end - c.start;
      // an ofd lock has no meaningful holder pid; linux reports -1 for one.
      this.lockReply(slot, [0, c.kind === 'write' ? 1 : 0, c.scope === 'ofd' ? -1 : c.pid,
        c.start >>> 0, Math.floor(c.start / 4294967296),
        clen >>> 0, Math.floor(clen / 4294967296)]);
      return;
    }

    if (kind === 'unlock') { this.locks.apply(req); this.lockReply(slot, [0]); return; }

    const blocking = cmd === 7 || cmd === 14 || cmd === 38; // F_SETLKW / F_OFD_SETLKW
    if (!blocking) { this.lockReply(slot, [this.locks.trySet(req)]); return; }

    const token = ++this.lockParkSeq;
    this.lockParked.set(token, { pid, slot });
    this.locks.setBlocking(req, (errno) => {
      if (!this.lockParked.delete(token)) return; // already cut short
      this.lockReply(slot, [errno]);
    });
  }

  /** how many data pages a NEW pipe gets. linux hands out PIPE_DEF_BUFFERS until the user is past
   *  pipe-user-pages-soft and a single page after that, so opening very many pipes degrades their
   *  capacity instead of failing or eating everything. F_GETPIPE_SZ then honestly reports the
   *  smaller size, which is exactly what linux does too. */
  private pipePagesForNew(privileged: boolean): number {
    const pages = this.pipePages >= PIPE_PAGES_SOFT ? PIPE_MIN_PAGES : PIPE_DEF_PAGES;
    // fs.pipe-max-size caps any single pipe -- lowering it is how an administrator makes every new
    // pipe smaller. a PRIVILEGED process is exempt, as it is on linux: the limit exists to bound
    // what ordinary programs can take, not to constrain the administrator who set it.
    if (privileged) return pages;
    const maxPages = Math.max(1, Math.floor(Atomics.load(this.globalState, GLOBAL_PIPE_MAX) / PAGE_SIZE));
    return Math.min(pages, maxPages);
  }

  /** give back every pipe block nobody can reach any more.
   *
   *  a block is unreachable once both ends are closed (no fd anywhere refers to it) or once every
   *  process that was ever handed an end has died (killed processes never run their close path, so
   *  without the second test their rings would be held forever). releasing to the arena rather than
   *  parking the block in a private pool is what lets the memory go back to MAP_SHARED as well --
   *  the old pool kept its high-water mark for the life of the session. */
  private reclaimDeadPipes(): void {
    // the ring's own refcounts decide this, and nothing else. reclamation used to also release a
    // ring whose recorded holders were all dead, which sounds equivalent and is not: liveness is
    // read from the process table, which races with fork -- a child that exists but is not yet
    // registered makes its ring look abandoned, and the block is handed to somebody else while it
    // is still in use. now that a dying process's ends are released for it (releasePipeRefs), a
    // ring that nobody holds reaches zero on its own and this test is both simpler and safe.
    for (const [off] of [...this.pipeBlocks]) {
      const ring = new PipeRing(this.shmArena, off);
      if (ring.readerClosed && ring.writerClosed) this.releasePipeRing(off);
    }
  }

  /** hand one ring back to the arena and uncharge its pages. */
  private releasePipeRing(off: number): void {
    const bytes = this.pipeBlocks.get(off);
    if (bytes === undefined) return;
    // a ring that is going away takes any descriptors still in flight over it with it. nobody can
    // collect them any more, and each one holds a real swift fd or another end of another ring.
    const stranded = this.scmQueues.get(off);
    if (stranded) {
      this.scmQueues.delete(off);
      for (const rec of stranded) this.releaseScmFd(rec.fd);
    }
    this.pipeBlocks.delete(off);
    this.pipeHolders.delete(off);
    this.shmAlloc.release(off, bytes);
    this.pipePages -= (bytes - PIPE_HEADER_SIZE) / PAGE_SIZE;
  }

  /** take a ring of `pages` data pages from the arena for `pid`, or -1 if it cannot be had.
   *
   *  Registering the holder is not bookkeeping, it is what keeps the ring ALIVE: reclamation asks
   *  whether any process that was handed an end still exists, and a block with no holder recorded
   *  answers "nobody" and is released the moment anyone asks for the next pipe -- straight back out
   *  of the arena and into the next caller's hands, so two unrelated pipes end up sharing one ring
   *  and each reads what the other wrote. */
  private allocPipeRing(pages: number, pid: number): number {
    const bytes = pipeRingSize(pages);
    const off = this.shmAlloc.alloc(bytes);
    if (off < 0) return -1;
    this.pipeBlocks.set(off, bytes);
    this.pipeHolders.set(off, new Map([[pid, { r: 1, w: 1 }]]));
    this.pipePages += pages;
    new PipeRing(this.shmArena, off, bytes).reset(1, 1);
    return off;
  }

  /** arm or disarm O_ASYNC notification for `pid` on one end of a ring.
   *
   *  The armed/not-armed answer is mirrored into the RING HEADER so a writer can decide whether to
   *  say anything with a single load. Nobody uses O_ASYNC most of the time, and the cost of
   *  supporting it should be paid by the programs that do. */
  private setPipeAsync(pid: number, off: number, mode: 'read' | 'write', sig: number, on: boolean): void {
    let per = this.pipeAsync.get(off);
    if (on) {
      if (!per) { per = new Map(); this.pipeAsync.set(off, per); }
      // the OTHER end's readiness is what this process wants to hear about: a reader is woken by
      // somebody writing, so it registers against the write that will make it readable.
      per.set(pid, { sig, mode });
    } else if (per) {
      per.delete(pid);
      if (per.size === 0) this.pipeAsync.delete(off);
    }
    if (this.pipeBlocks.has(off)) {
      const hdr = new Int32Array(this.shmArena, off, PIPE_HEADER_SIZE / 4);
      Atomics.store(hdr, PIPE_ASYNC_W, this.pipeAsync.get(off)?.size ? 1 : 0);
    }
  }

  /** a ring became ready; signal everyone who asked to hear about it.
   *  SIGIO is the default, and F_SETSIG chooses another -- that choice is the whole point of
   *  F_SETSIG, since SIGIO carries no information about WHICH descriptor woke up. */
  private deliverPipeIo(off: number, mode: 'read' | 'write'): void {
    const per = this.pipeAsync.get(off);
    if (!per) return;
    for (const [pid, reg] of per) {
      if (reg.mode !== mode) continue;
      this.postSignal(pid, reg.sig || SIGIO);
    }
  }

  /** adjust what `pid` holds of the ring at `off`. the ring's own refcount is moved by the worker;
   *  this is the kernel's record of WHO holds it, which is what lets it clean up after a process
   *  that never got to close anything. */
  /** say exactly which ends of `off` belong to `pid`, replacing whatever was recorded. */
  private setPipeHolder(off: number, pid: number, r: number, w: number): void {
    let per = this.pipeHolders.get(off);
    if (!per) { per = new Map(); this.pipeHolders.set(off, per); }
    if (r <= 0 && w <= 0) per.delete(pid); else per.set(pid, { r, w });
  }

  private notePipeRef(pid: number, off: number, mode: 'read' | 'write', delta: number): void {
    let per = this.pipeHolders.get(off);
    if (!per) { per = new Map(); this.pipeHolders.set(off, per); }
    const cur = per.get(pid) ?? { r: 0, w: 0 };
    if (mode === 'write') cur.w += delta; else cur.r += delta;
    if (cur.r <= 0 && cur.w <= 0) per.delete(pid);
    else per.set(pid, cur);
  }

  /** a forked child inherits every descriptor its parent had open, pipe ends included. */
  private inheritPipeRefs(parentPid: number, childPid: number): void {
    for (const per of this.pipeHolders.values()) {
      const p = per.get(parentPid);
      if (p) per.set(childPid, { r: p.r, w: p.w });
    }
  }

  /** drop every pipe end `pid` still held.
   *
   *  This is the whole point of tracking holders. A process that exits normally closes its own ends
   *  and there is nothing left here to do. A process that is KILLED never runs that code, and its
   *  ends would stay counted forever -- so a reader waits on an EOF that cannot come, because the
   *  ring still believes in a writer that no longer exists. That is a shell that never returns to
   *  its prompt, and it is what the kernel closing fds on exit exists to prevent. */
  private releasePipeRefs(pid: number): void {
    for (const [off, per] of this.pipeAsync) {
      if (!per.delete(pid)) continue;
      if (per.size === 0) this.pipeAsync.delete(off);
      if (this.pipeBlocks.has(off)) {
        Atomics.store(new Int32Array(this.shmArena, off, PIPE_HEADER_SIZE / 4), PIPE_ASYNC_W, per.size ? 1 : 0);
      }
    }
    for (const [off, per] of this.pipeHolders) {
      const held = per.get(pid);
      if (!held) continue;
      per.delete(pid);
      if (!this.pipeBlocks.has(off)) continue;
      const ring = new PipeRing(this.shmArena, off);
      for (let i = 0; i < held.w; i++) ring.closeWriter();
      for (let i = 0; i < held.r; i++) ring.closeReader();
    }
  }

  /** does this pid still occupy a proc-table slot? */
  private pidIsLive(pid: number): boolean {
    for (let s = 0; s < PROC_TABLE_SLOTS; s++) {
      const base = procSlotOffset(s) / 4;
      if (Atomics.load(this.procTable, base + PROC_PID) !== pid) continue;
      if (Atomics.load(this.procTable, base + PROC_STATE) !== PROC_EMPTY) return true;
    }
    return false;
  }


  /** take one reference to a shared block on behalf of `pid`. the block's own count moves here so
   *  the two records cannot drift apart. */
  private holdShm(pid: number, off: number): void {
    const e = this.shmRefs.get(off);
    if (!e) return;
    e.refs++;
    let per = this.shmHeld.get(pid);
    if (!per) { per = new Map(); this.shmHeld.set(pid, per); }
    per.set(off, (per.get(off) ?? 0) + 1);
  }

  /** give one reference back. the block goes to the arena only when the last one is gone. */
  private dropShm(pid: number, off: number): void {
    const per = this.shmHeld.get(pid);
    const held = per?.get(off) ?? 0;
    if (held <= 0) return;
    if (held === 1) per!.delete(off); else per!.set(off, held - 1);
    const e = this.shmRefs.get(off);
    if (!e) return;
    if (--e.refs > 0) return;
    this.shmRefs.delete(off);
    if (e.key !== undefined) this.shmByKey.delete(e.key);
    this.shmAlloc.release(off, e.len);
  }

  /** a forked child co-owns every shared block its parent held. */
  /** a child of fork(2) has its parent's segments attached at the same addresses, and counts. */
  private inheritSysvShm(parentPid: number, childPid: number): void {
    for (const seg of this.sysvShm.values()) {
      const held = seg.attachers.get(parentPid);
      if (held) seg.attachers.set(childPid, held);
    }
  }

  /** a dead process detaches from everything, which may be what finally destroys a segment. */
  private releaseSysvShm(pid: number): void {
    for (const seg of [...this.sysvShm.values()]) {
      if (!seg.attachers.delete(pid)) continue;
      this.dropSysvIfDone(seg);
    }
  }

  private inheritShm(childPid: number, blocks: number[] | undefined): void {
    // one entry per MAPPING, so a parent holding the same block twice hands the child two
    // references -- which is exactly what the child has.
    for (const off of blocks ?? []) this.holdShm(childPid, off);
  }

  private releaseShmForPid(pid: number): void {
    const per = this.shmHeld.get(pid);
    if (!per) return;
    for (const [off, held] of [...per]) {
      for (let i = 0; i < held; i++) this.dropShm(pid, off);
    }
    this.shmHeld.delete(pid);
  }

  /**
   * Everything a process owns, released in one place.
   *
   * A process can leave in four different ways -- it exits, its worker crashes, it is swept up as an
   * orphan zombie, or it is a thread of a group whose leader died -- and each of those used to do
   * its own cleanup. They did not agree. Three of them terminated the worker and freed the proc slot
   * but never dropped the process's RECORD LOCKS, so a process killed rather than exited kept its
   * locks for the rest of the session, and the next process to want that range blocked forever on a
   * holder that no longer existed. That is a wedge with no error and no clue: every table looks
   * healthy, the guest just stops.
   *
   * Any new path that makes a process disappear must come through here.
   */
  /**
   * Everything the kernel remembers about a process, released in ONE place.
   *
   * this used to be split: handleExit dropped some maps, forgetProcess dropped others, and the two
   * did not agree -- so a process that took the reap path instead of the exit path left entries
   * behind. each one is only a few bytes, but they are per-pid and a long run makes millions of
   * pids, and the same split is how a real resource (a native fd, a pipe ring) gets stranded.
   * every per-pid map belongs here, and anything added later belongs here too.
   */
  private forgetProcess(pid: number): void {
    this.reclaimNativeFds(pid);
    // a process killed while blocked in futex(2) never gets to take its entry out of the wait
    // queue, and a queue full of ghosts is a queue no live waiter can enter.
    futexReleasePid(this.procTable, pid);
    this.setExecImage(pid, null);
    for (const [id, ev] of this.eventfds) {
      ev.holders.delete(pid);
      if (ev.holders.size === 0) this.eventfds.delete(id);
    }
    // a name in the unix-domain namespace belongs to the socket that bound it, and a process that
    // died still holding one would keep that path unbindable for the rest of the run -- so the
    // second program to use a well-known socket path got EADDRINUSE from a ghost.
    for (const [path, entry] of this.unixNames) {
      entry.holders.delete(pid);
      if (entry.holders.size === 0) {
        if (entry.inbox !== undefined) this.releasePipeRing(entry.inbox);
        this.unixNames.delete(path);
      }
    }
    this.locks.releaseProcess(pid);
    this.forgetParked(pid);
    this.releaseShmForPid(pid);
    this.detachedOwner.delete(pid);
    // the pipe holder sets are what tell the allocator a ring is unreachable; dropping the dead pid
    // lets a ring be reclaimed on the next pipe() instead of waiting out a slot-reuse coincidence.
    this.releasePipeRefs(pid);
    this.lastActive.delete(pid);
    this.committed.delete(pid);
    this.pipeAsync.delete(pid);
    this.grownBytes.delete(pid);
    this.threadLayouts.delete(pid);
    this.threadCounts.delete(pid);
    this.threadedLeaders.delete(pid);
    this.pendingPromotes.delete(pid);
  }

  /** stop counting `pid` as a waiter on `semid`, so GETNCNT/GETZCNT stay honest. */
  private clearSemWait(pid: number, semid: number): void {
    this.sysvSem.get(semid)?.waiting.delete(pid);
  }

  /** a thread that will never come back cannot still be waiting on anything. */
  private clearSemWaiters(pid: number): void {
    for (const st of this.sysvSem.values()) st.waiting.delete(pid);
  }

  /**
   * apply a dead process's SEM_UNDO adjustments. this is the whole point of SEM_UNDO: a process
   * that took a semaphore and then died would otherwise hold it forever, and every other user of
   * that semaphore would wait on a process that no longer exists. the adjustment is clamped at
   * zero rather than allowed to go negative, exactly as linux does.
   */
  private releaseSemUndo(tgid: number): void {
    const per = this.semUndo.get(tgid);
    if (!per) return;
    this.semUndo.delete(tgid);
    let changed = false;
    for (const [semid, adj] of per) {
      const st = this.sysvSem.get(semid);
      if (!st) continue;
      for (let i = 0; i < st.nsems && i < adj.length; i++) {
        if (adj[i] === 0) continue;
        let v = st.vals[i] + adj[i];
        if (v < 0) v = 0;
        if (v > SEMVMX) v = SEMVMX;
        st.vals[i] = v;
        st.pids[i] = tgid;
        changed = true;
      }
    }
    if (changed) bumpIoGeneration();
  }

  /**
   * a dead process holds nothing. drop its share of every message queue, and with it any
   * notification it had registered -- otherwise an unlinked queue would never be freed, and a
   * queue whose registrant died could never be registered on again.
   */
  private releaseMqForPid(pid: number): void {
    for (const q of [...this.posixMq.values()]) {
      if (q.notify?.pid === pid) q.notify = null;
      if (!q.holders.delete(pid)) continue;
      if (!q.unlinked || q.holders.size > 0) continue;
      for (const e of q.msgs) if (e.len > 0) this.shmAlloc.release(e.off, e.len);
      q.msgs.length = 0;
      this.posixMq.delete(q.id);
    }
  }

  /** a forked child inherits every descriptor, and so every queue its parent was holding. */
  private inheritMqForFork(parentPid: number, childPid: number): void {
    for (const q of this.posixMq.values()) if (q.holders.has(parentPid)) q.holders.add(childPid);
  }

  /** give back the block a process staged its forks in. */
  private releaseForkStage(pid: number): void {
    const held = this.forkStages.get(pid);
    if (!held) return;
    this.forkStages.delete(pid);
    this.shmAlloc.release(held.off, held.len);
  }

  private releaseLocksForPid(pid: number): void {
    this.forgetProcess(pid);
  }

  /** hand back every native descriptor a process still holds. a process that reaches its own exit
   *  path closes them itself and this finds nothing; one the kernel force-terminated never runs
   *  another line of javascript, and without this its Swift FileHandles live until the app dies.
   *  the ltp runner sends `kill -9 -1` between every test, so on that workload it is the rule
   *  rather than the exception. same reasoning as closing a dead process's pipe ends. */
  private reclaimNativeFds(pid: number): void {
    if (!this.bridge) return;
    void this.bridge.call('fs_close_owner', { pid }).catch(() => {});
    this.leases.dropProcess(pid);
  }

  private cancelLockWaiter(pid: number): void {
    // a blocking request the worker abandoned (signal / give-up) must not later be granted.
    this.locks.interrupt(pid);
    this.forgetParked(pid);
  }

  private forgetParked(pid: number): void {
    for (const [token, p] of [...this.lockParked]) if (p.pid === pid) this.lockParked.delete(token);
  }

  /**
   * closing a descriptor is a lock event: posix locks are dropped for the whole FILE as soon as the
   * process closes any descriptor for it, while an ofd lock survives until the last descriptor of
   * that description is gone.
   */
  private releaseLocksOnClose(pid: number, path: string, ofdId: number, lastRef: boolean): void {
    this.locks.releaseProcessFile(pid, path);
    if (lastRef && ofdId) this.locks.releaseDescription(ofdId);
  }

  private handleBridgeCall(pid: number, slot: number, msg: KernelMessage & { type: 'bridge_call' }): void {
    if (!this.bridge) {
      this.writeBridgeResponse(slot, -38, [], null);
      return;
    }

    const method = (msg as any).method as string;
    // ancillary descriptors travelling over a unix socket are the kernel's business, not swift's:
    // they are queued here, against the stream position they were sent at, and handed to whoever
    // reads past that position. answering through the bridge's own response slot is what lets the
    // record carry a whole descriptor -- a path, a ring offset, a socket id -- rather than the six
    // integers a mailbox reply has room for.
    if (method === 'scm_send' || method === 'scm_take' || method === 'scm_drop' ||
        method === 'scm_unsend') {
      this.handleScm(slot, method, (msg as any).args as Record<string, any>);
      return;
    }
    // a file handle names a file by identity rather than by path, and open_by_handle_at may be
    // called by a DIFFERENT process than the one that took the handle -- so the map of which inode
    // is which file belongs to the kernel, not to any one process.
    if (method === 'handle_put') {
      const a = (msg as any).args as Record<string, any>;
      this.fileHandles.set(Number(a.ino), String(a.path));
      this.writeBridgeResponse(slot, 0, [], null);
      return;
    }
    if (method === 'handle_get') {
      const a = (msg as any).args as Record<string, any>;
      const path = this.fileHandles.get(Number(a.ino));
      this.writeBridgeResponse(slot, path ? 1 : 0, [],
        path ? new TextEncoder().encode(path) : null);
      return;
    }
    // stamp the caller so a descriptor it opens can be reclaimed if it is later force-terminated
    // and never gets to close anything itself
    const args: Record<string, any> = { ...((msg as any).args as Record<string, any>), _owner: pid };
    // a program that is RUNNING may not be written to. every open and truncate already passes
    // through here on its way to the filesystem, and this is the only place that knows which
    // images are executing, so it is where ETXTBSY belongs.
    if ((method === 'fs_open' || method === 'fs_truncate') && typeof args.path === 'string') {
      const wantsWrite = method === 'fs_truncate' ||
        ((Number(args.flags) & 3) !== 0) || ((Number(args.flags) & 0o1000) !== 0); // O_WRONLY/O_RDWR/O_TRUNC
      if (wantsWrite && this.execImages.has(args.path)) {
        this.writeBridgeResponse(slot, -26, [], null);   // ETXTBSY
        return;
      }
    }
    // leases sit exactly here: every native open and close already passes through on its way to
    // swift, so the open-file bookkeeping a lease decision needs rides along for free, and a
    // conflicting open can be held back until the holder has been told and gotten out of the way.
    if (this.leaseRelevant(method)) { void this.bridgeWithLeases(pid, slot, method, args); return; }
    this.dispatchBridge(slot, method, args);
  }

  private leaseRelevant(method: string): boolean {
    return this.leases.active && (method === 'fs_open' || method === 'fs_truncate');
  }

  /** hold a conflicting open back until the lease on that path has been broken. */
  private async bridgeWithLeases(pid: number, slot: number, method: string, args: Record<string, any>): Promise<void> {
    const path = String(args.path ?? '');
    const flags = Number(args.flags ?? 0);
    const forWrite = method === 'fs_truncate' || (flags & 3) !== 0 || (flags & 0x200) !== 0 /* O_TRUNC */;
    const r = await this.leases.breakOn(path, forWrite, (flags & 0x800) !== 0 /* O_NONBLOCK */);
    if (r < 0) { this.writeBridgeResponse(slot, r, [0, 0, 0, 0], null); return; }
    this.dispatchBridge(slot, method, args);
  }

  /**
   * SCM_RIGHTS: descriptors in flight over a unix-domain socket.
   *
   * queued against the SENDING ring's stream position, because that is what both ends can name --
   * the sender knows where it wrote, the receiver knows how far it has read, and linux delivers
   * ancillary data with the byte it accompanied. the descriptor itself was already duplicated by
   * the sender (a dup'd swift fd, another reference on a socket, another end on a ring), so a
   * record that is never collected owns a real resource: dropping the queue releases them.
   */
  private handleScm(slot: number, method: string, args: Record<string, any>): void {
    const ring = Number(args.ring);
    if (method === 'scm_send') {
      const q = this.scmQueues.get(ring) ?? [];
      for (const fd of (args.fds as any[])) q.push({ pos: Number(args.pos) | 0, fd });
      this.scmQueues.set(ring, q);
      this.writeBridgeResponse(slot, (args.fds as any[]).length, [], null);
      return;
    }
    if (method === 'scm_unsend') {
      // the send those records belonged to did not happen. take them back out, in place, so the
      // caller can give the descriptors back rather than leaking them into a stream nobody reads.
      const q = this.scmQueues.get(ring) ?? [];
      const pos = Number(args.pos) | 0;
      let left = Number(args.n) | 0;
      for (let i = q.length - 1; i >= 0 && left > 0; i--) {
        if ((q[i].pos | 0) === pos) { q.splice(i, 1); left--; }
      }
      if (q.length === 0) this.scmQueues.delete(ring); else this.scmQueues.set(ring, q);
      this.writeBridgeResponse(slot, 0, [], null);
      return;
    }
    if (method === 'scm_drop') {
      const q = this.scmQueues.get(ring);
      this.scmQueues.delete(ring);
      for (const rec of q ?? []) this.releaseScmFd(rec.fd);
      this.writeBridgeResponse(slot, 0, [], null);
      return;
    }
    // scm_take: everything queued at or before the last byte this reader consumed. positions are
    // monotonic int32s and wrap, so the comparison is done on the difference.
    const to = Number(args.to) | 0;
    const q = this.scmQueues.get(ring) ?? [];
    const taken: any[] = [];
    while (q.length && ((to - (q[0].pos | 0)) | 0) >= 0) taken.push(q.shift()!.fd);
    if (q.length === 0) this.scmQueues.delete(ring); else this.scmQueues.set(ring, q);
    this.writeBridgeResponse(slot, taken.length, [],
      taken.length ? new TextEncoder().encode(JSON.stringify(taken)) : null);
  }

  /** give back a descriptor nobody collected. */
  private releaseScmFd(fd: any): void {
    if (!fd) return;
    if (fd.k === 'native' && this.bridge) void this.bridge.call('fs_close', { fd: fd.nfd });
    else if (fd.k === 'socket' && this.bridge) void this.bridge.call('net_close', { socketId: fd.socketId });
    else if (fd.k === 'pipe' || fd.k === 'unix') {
      for (const [off, mode] of fd.ends as Array<[number, string]>) {
        const r = new PipeRing(this.shmArena, off);
        if (mode === 'w') r.closeWriter(); else r.closeReader();
      }
    }
  }

  private dispatchBridge(slot: number, method: string, args: Record<string, any>): void {
    const pid = Number(args._owner);
    this.bridge!.call(method, args).then((resp) => {
      const ret = (resp as any).ret ?? 0;

      // Pack response into SAB: [ret:i32][dataLen:u32][meta:32B][data:bytes]
      // Meta fields are method-specific fixed-layout integers.
      // Data is variable-length (binary for reads, JSON for getdents, UTF-8 for readlink).
      const meta = [0, 0, 0, 0, 0, 0]; // 6 x f64: mode, size, nlink, mtime, uid, gid
      let rawData: Uint8Array | null = null;

      if (method === 'fs_open') {
        // only meaningful when ret is -ENODEV: what device node the path names.
        meta[0] = (resp as any).mode ?? 0;
        meta[1] = (resp as any).rdev ?? 0;
        // whether this open CREATED the file, which the caller cannot tell afterwards and inotify
        // needs to distinguish IN_CREATE from IN_OPEN.
        meta[2] = (resp as any).created ?? 0;
        if (ret >= 0) {
          const fl = Number(args.flags ?? 0);
          this.leases.noteOpen(pid, ret, String(args.path ?? ''), (fl & 3) !== 0);
        }
      } else if (method === 'fs_close' && ret >= 0) {
        this.leases.noteClose(pid, Number(args.fd));
      } else if (method === 'fs_dup' && ret >= 0) {
        this.leases.noteDupFrom(pid, Number(args.fd), ret);
      } else if (method === 'net_accept' || method === 'net_read') {
        // who is at the other end. accept(2) and recvfrom(2) both answer it, and it cannot ride the
        // data channel because that is the payload -- so the address travels as the eight 16-bit
        // groups of a struct sockaddr, which is exact for both families.
        meta[0] = (resp as any).peerPort ?? 0;
        meta[1] = (resp as any).peerFamily ?? 0;
        const peerAddr = (resp as any).peerAddr as number[] | undefined;
        for (let i = 0; i < 8; i++) meta[2 + i] = peerAddr?.[i] ?? 0;
        if ((resp as any).data) {
          rawData = Uint8Array.from(atob((resp as any).data), (c) => c.charCodeAt(0));
        }
      } else if (method === 'net_ifaddrs') {
        if ((resp as any).ifs) rawData = new TextEncoder().encode(JSON.stringify((resp as any).ifs));
      } else if (method === 'net_recvmmsg') {
        // a whole batch of datagrams, each with its own sender. json on the data channel is the
        // only shape that carries n payloads and n addresses in one reply.
        if ((resp as any).msgs) rawData = new TextEncoder().encode(JSON.stringify((resp as any).msgs));
      } else if (method === 'net_bind') {
        // bind(port 0) lets the system choose; the caller learns which one through getsockname,
        // so the choice has to come back with the answer.
        meta[0] = (resp as any).port ?? 0;
      } else if (method === 'fs_stat' || method === 'fs_fstat' || method === 'fs_lstat') {
        meta[0] = (resp as any).mode ?? 0;
        meta[1] = (resp as any).size ?? 0;
        meta[2] = (resp as any).nlink ?? 0;
        meta[3] = (resp as any).mtime ?? 0;
        meta[4] = (resp as any).uid ?? 0;
        meta[5] = (resp as any).gid ?? 0;
        meta[6] = (resp as any).ino ?? 0; // st_ino — MUST be unique per file: ld.so dedups libs by it
        meta[7] = (resp as any).rdev ?? 0; // st_rdev — the major:minor of a device node
        // the rest of struct stat: the two other timestamps (with the nanoseconds all three
        // carry), and the block accounting du and tar read. these used to arrive as zero, so
        // every file claimed to have been last read at the epoch and to occupy no blocks.
        meta[8] = (resp as any).atime ?? 0;
        meta[9] = (resp as any).ctime ?? 0;
        meta[10] = (resp as any).mtimeNsec ?? 0;
        meta[11] = (resp as any).atimeNsec ?? 0;
        meta[12] = (resp as any).ctimeNsec ?? 0;
        meta[13] = (resp as any).blocks ?? 0;
        meta[14] = (resp as any).blksize ?? 4096;
        // the fully resolved path rides the data channel, which stat otherwise leaves empty. chdir
        // needs it: the working directory is the directory, not the name it was reached by.
        if ((resp as any).real) rawData = new TextEncoder().encode((resp as any).real);
      } else if (method === 'fs_readlink') {
        if ((resp as any).target) {
          rawData = new TextEncoder().encode((resp as any).target);
        }
      } else if (method === 'fs_getdents') {
        if ((resp as any).entries) {
          rawData = new TextEncoder().encode(JSON.stringify((resp as any).entries));
        }
      } else if ((resp as any).data) {
        // Binary data: fs_read, net_read — base64 decode
        rawData = Uint8Array.from(atob((resp as any).data), c => c.charCodeAt(0));
      }

      this.writeBridgeResponse(slot, ret, meta, rawData);
    }).catch((err) => {
      // a rejected bridge call (unknown/unimplemented method, native exception, transient
      // failure) must STILL wake the worker with an error — otherwise it parks on the response
      // mailbox forever. before this, a single unhandled method (e.g. fs_chown) hung the whole
      // process. surface it as -EIO and log which method failed so gaps are visible, not silent.
      console.error(`[kernel] bridge call '${method}' failed:`, err);
      this.writeBridgeResponse(slot, -5, [0, 0, 0, 0], null); // -EIO
    });
  }

  private writeBridgeResponse(slot: number, ret: number, meta: number[], data: Uint8Array | null): void {
    const respOffset = BRIDGE_RESP_OFFSET + slot * BRIDGE_RESP_SLOT_SIZE;
    const respView = new DataView(this.sab, respOffset, BRIDGE_RESP_SLOT_SIZE);

    // Response layout: [ret:i32][metaCount:u16][dataLen:u32][meta: metaCount×f64][data: dataLen bytes]
    // Header: 10 bytes, then variable meta, then variable data.
    const metaCount = meta.length;
    const dataOffset = 10 + metaCount * 8;

    respView.setInt32(0, ret, true);
    respView.setUint16(4, metaCount, true);

    for (let i = 0; i < metaCount; i++) {
      respView.setFloat64(10 + i * 8, meta[i], true);
    }

    if (data && data.length > 0 && data.length <= BRIDGE_RESP_SLOT_SIZE - dataOffset) {
      respView.setUint32(6, data.length, true);
      new Uint8Array(this.sab, respOffset + dataOffset, data.length).set(data);
    } else {
      respView.setUint32(6, 0, true);
      // a payload that does not fit used to be dropped while the COUNT was kept, so a read of
      // 65536 bytes reported success and delivered nothing -- which is how `cp` produced empty
      // files. every caller caps its request now; if one still does not, say so and fail the call
      // rather than hand back a number that is not true.
      if (data && data.length > 0) {
        console.error(`[kernel] bridge reply of ${data.length} bytes does not fit the response slot`);
        respView.setInt32(0, -5, true); // EIO
      }
    }

    // Wake the Worker
    const mailbox = new Int32Array(this.sab, mailboxSlotOffset(slot), MAILBOX_SLOT_SIZE / 4);
    Atomics.store(mailbox, MBOX_READY, 1);
    Atomics.notify(mailbox, MBOX_READY);
  }
}
