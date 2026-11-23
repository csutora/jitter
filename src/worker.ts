/**
 * Web Worker entry point for multicore process execution.
 */

import { HALT_PC, SVC_PC, GUEST_MEM_BASE, ITER_OFFSET, PC_OFFSET, STATE_SIZE,
         DEFAULT_THREAD_LAYOUT, PROT_TABLE_BASE, PROT_PAINTED_FLAG, TPIDR_OFFSET } from './types.js';
import type { ThreadLayout } from './types.js';
import type { CompiledRegion } from './types.js';
import { CpuState } from './cpu.js';
import { initZigTranslator, initZigTranslatorModule, translateRegionZig } from './zig-translator.js';
import { SyscallHandler, badGuestPtr } from './syscall.js';
import { resolveExec } from './binfmt.js';
import { InMemoryVFS, guestCpus } from './vfs.js';
import { loadProgram } from './elf.js';
import { PipeRing } from './pipe-ring.js';
import { reconcile, type SharedMapping } from './shared-map.js';
import type { WorkerInitConfig, KernelMessage, SparseMemSnapshot } from './kernel-ipc.js';
import {
  MEM_PAGE_SIZE,
  futexLock, futexUnlock, futexEnqueue, futexDequeue, futexWake, futexStateIndex,
  FUTEX_QUEUED, FUTEX_WOKEN, FUTEX_RESULT, FUTEX_SLOT_WORDS, FUTEX_HDR_WORDS, FUTEX_TABLE_OFFSET,
  PROC_SLOT_SIZE, PROC_STATE, PROC_EXIT_CODE, PROC_WAIT_FUTEX, PROC_SWAP_REQ, PROC_BLOCKED, sigqPending, PROC_SYSCALL,
  type QueuedSignal,
  PROC_UTIME_MS, PROC_STIME_MS, PROC_NICE, PROC_PDEATHSIG,
  PROC_POLICY, PROC_RTPRIO, PROC_SCHED_FLAGS, PROC_AFFINITY, PROC_IOPRIO, PROC_RUID,
  PROC_DL_RUNTIME, PROC_DL_DEADLINE, PROC_DL_PERIOD,
  PROC_EMPTY, PROC_ZOMBIE, PROC_RUNNING, PROC_STOPPED,
  PROC_STOP_SIG, PROC_STOP_SEEN, PROC_CONT_PENDING,
  PROC_PID, PROC_PPID, PROC_PGID, PROC_SID, PROC_EXECED, PROC_SUBREAPER, PROC_TGID,
  PROC_RLIM_SIGPENDING, PROC_TRACER, PROC_TRACE_CMD, PROC_VFORK, PROC_MAXRSS_KB, PT_RESUME_NONE, PT_RESUME_CONT, PT_RESUME_DETACH,
  ptraceSlotWord, ptraceScratchOffset, PT_REQ, PT_ACK, PT_OP, PT_ADDR, PT_LEN, PT_DATA_LO, PT_DATA_HI, PT_STATUS,
  MAILBOX_OFFSET, MAILBOX_SLOT_SIZE, MBOX_READY, MBOX_VALUE0,
  PIPE_HEADER_SIZE, PIPE_ASYNC_W, GLOBAL_STATE_OFFSET, GLOBAL_PIPE_MAX, GLOBAL_INOTIFY_WATCHES, GLOBAL_MSGMNI,
  GLOBAL_MSG_NEXT_ID, GLOBAL_SEM_NEXT_ID, GLOBAL_SHM_NEXT_ID,
  GLOBAL_BOOT_MS_LO, GLOBAL_BOOT_MS_HI, GLOBAL_MQ_MSG_MAX, GLOBAL_MQ_MSGSIZE_MAX, GLOBAL_MQ_QUEUES_MAX,
  GLOBAL_CLK_SEQ, GLOBAL_CLK_OFF_SEC, GLOBAL_CLK_OFF_NSEC,
  GLOBAL_UTS_LEN, GLOBAL_HOSTNAME_OFFSET, GLOBAL_DOMAINNAME_OFFSET,
  bindIoGeneration, bumpIoGeneration, readIoGeneration, waitIoGeneration,
  STDIN_RING_OFFSET, STDIN_RING_SIZE,
  BRIDGE_RESP_OFFSET, BRIDGE_RESP_SLOT_SIZE, BRIDGE_DATA_MAX,
  procSlotOffset, mailboxSlotOffset, stdinRingOffset, procCmdlineOffset,
  PROC_COMM, PROC_CMDLINE_SLOT_SIZE, PROC_TABLE_SLOTS, sigqDrain, sigqPush, sigqDepth, procCredOffset,
} from './kernel-ipc.js';

let memory: WebAssembly.Memory;
let cpu: CpuState;
let syscalls: SyscallHandler;
let vfs: InMemoryVFS;
let regionCache = new Map<bigint, CompiledRegion>();

// ── threading ────────────────────────────────────────────────────────────────
// a process stays single-threaded until its first pthread_create (clone with CLONE_VM|CLONE_THREAD),
// then PROMOTES: its memory becomes a shared WebAssembly.Memory (so thread-Workers can import it) and
// translation switches to threaded mode (reg_base-relative registers + real atomics + fences). each
// thread-Worker runs the same shared memory with its own register block at `stateBase`/`regBaseGlobal`.
let threaded = false;
let stateBase = 0;                                    // this thread's register-block base (0 = group main)
let regBaseGlobal: WebAssembly.Global | null = null;  // per-thread reg_base handed to the translator
let clearChildTid = 0;                                // CLONE_CHILD_CLEARTID: guest addr to zero+wake on exit
// this process's threaded address-space tier. the kernel picks it per process (see pickThreadTier) and
// sends it with the shared memory, so register blocks and the atomics shadow land where the kernel
// placed them — it is NOT a global constant, since different processes get different ceilings.
let threadLayout: ThreadLayout = DEFAULT_THREAD_LAYOUT;
// threaded memory layout: the guest occupies [GUEST_MEM_BASE, regsBase); the per-thread
// register blocks (64 x 0x400 at regsBase) and the shared LDXR/STXR shadow (guest_size/512,
// 4 MiB above the reg blocks) sit ABOVE that ceiling so a growing brk/mmap can never reach them, and
// the syscall handler's maxBytes enforces it. a threaded guest's ceiling is therefore smaller than a
// single-threaded one's — the sizing rationale (shared-memory address-space reservation) is in types.ts,
// which both the kernel and this file import so the two can't drift.
// cap on cached translated blocks per worker. a hot working set is usually a few hundred; this
// bounds the pathological huge-binary case at ~this many compiled wasm blocks (re-JIT on re-hit).
const REGION_CACHE_CAP = 2048;
let elfBytes: Uint8Array;

// Kernel SAB views
let kernelSAB: SharedArrayBuffer;
let procTable: Int32Array;
// the per-(process, signal) sender table; same buffer, its own view for clarity at the use sites
let sigQueue: Int32Array;
let procCreds: Int32Array;
let globalWords: Int32Array;
let mySlot: number;
let myPid: number;
let myMailbox: Int32Array;

// Kernel pipe tracking: local fd -> { pipeOff, mode }. pipeOff is a byte offset into the shared
// arena: pipe buffers are allocated there on demand rather than coming from a fixed array of
// preallocated rings, so the system is not capped at a handful of pipes and idle capacity is not
// reserved up front.
// These are pipes backed by SharedArrayBuffer ring buffers (cross-Worker IPC)
const kernelPipes = new Map<number, { pipeOff: number; mode: 'read' | 'write' }>();
let nextLocalFd = 100; // kernel pipe fds start high to avoid conflicts with VFS fds

// Stdin ring for this process (if connected to terminal)
let stdinRing: PipeRing | null = null;

// this Worker is pooled and reused across guest processes (see the kernel's worker pool). the
// translator instance and the wasm Memory are created ONCE and reused; everything else is reset
// per process. reusing the Worker (never terminate) is what removes the GC-reclaim lag that a
// fork storm drowned in.
let workerReady = false;
let lastSyscallNr = -1;
// syscall tracing, off by default and switched on from the host. only the calls that touch files
// and descriptors are recorded: enough to answer "what did this program do to the filesystem",
// which is the question black-box probing cannot settle when every call succeeds and the outcome
// is still wrong.
let tracing = false;
const TRACED = new Set([56, 57, 62, 63, 64, 66, 68, 37, 38, 35, 36, 45, 46, 82, 83, 79, 80, 78, 439]);
/** a NUL-terminated guest string, for tracing only. */
function readCString(addr: number): string | null {
  try {
    const b = new Uint8Array(memory.buffer);
    let end = addr + GUEST_MEM_BASE;
    const lim = end + 512;
    while (end < lim && b[end] !== 0) end++;
    return new TextDecoder().decode(b.subarray(addr + GUEST_MEM_BASE, end));
  } catch { return null; }
}

let tA3 = 0n, tA4 = 0n;
function traceCall(nr: number, a0: bigint, a1: bigint, a2: bigint, path: string | null): void {
  if (!tracing) return;
  const r = BigInt.asIntN(64, cpu.getReg(0));
  (self as any).postMessage({
    type: 'trace',
    line: `pid${myPid} ${SYSNAME[nr] ?? nr}(${a0}, ${a1}, ${a2}, ${tA3}, ${tA4})${path ? ' "' + path + '"' : ''} = ${r}`,
  });
}
const SYSNAME: Record<number, string> = {
  56: 'openat', 57: 'close', 62: 'lseek', 63: 'read', 64: 'write', 66: 'writev', 68: 'pwrite',
  37: 'linkat', 38: 'renameat', 35: 'unlinkat', 36: 'mkdirat', 45: 'truncate', 46: 'ftruncate',
  82: 'fsync', 83: 'fdatasync', 222: 'mmap', 215: 'munmap', 88: 'utimensat',
  79: 'fstatat', 80: 'fstat', 78: 'readlinkat', 439: 'faccessat2',
};

const ZIG_SHADOW_BASE = 0x2000000; // 32 MB — guest stays under this

// software page-protection guard (non-threaded/fork path only, matching CpuDispatch). the translated
// guard reads `protActive` (0 = fast path, skipped) and on a restricted access calls `memFault`, which
// cpu accounting. procStartMs is reset when the process begins running guest code; parkedMs
// accumulates every millisecond spent in a blocking wait (see setSleeping).
let parkedMs = 0;
let parkDepth = 0;
let parkStartMs = 0;
let procStartMs = performance.now();

// records the fault and throws MEM_FAULT to unwind out of region.run() to the fault handler in runLoop.
// software page-protection guard (non-threaded/fork path only). the translated guard reads `protActive`
// (0 = fast path, skipped) and on a restricted access calls `memFault`, which records the fault and
// throws MEM_FAULT to unwind out of region.run() to the fault handler in runLoop.
// the permission table lives BELOW the guest region (PROT_TABLE_BASE, see types.ts): it is inside the
// initially-committed memory, so arming the guard never grows memory — the old design put it above the
// 1.5 GiB process ceiling, which cost a third of the platform's ~4 GiB address-space budget per worker
// and usually just failed, and a version that placed it above the current end got zeroed by mmap's
// zero-fill of the region the guest grew into. its base is still an imported global so the host owns
// the placement without the translator baking it in.
// the guest stack is placed at 8 MiB and grows DOWN, with no vma of its own, so a snapshot has to
// treat everything below it as potentially live. must match STACK_TOP in elf.ts.
const GUEST_STACK_TOP = 0x800000;

const MEM_FAULT = { memFault: true } as const;
// how much guest address space one process may have. 1.5 GiB was chosen when we believed the
// platform ceiling was about 1.25 GiB; it is not -- a single wasm memory reaches 4 GiB, written,
// and what actually bounds us is total committed bytes across the app (~3.7 GiB measured). the
// limit now is the permission table, which must fit between PROT_TABLE_BASE and GUEST_MEM_BASE at
// one byte per 4 KiB page. 1856 MiB needs 464 KiB of table and leaves room to spare.
const MAX_PROCESS_BYTES = 1856 * 1024 * 1024;
const PROT_TABLE_SIZE = MAX_PROCESS_BYTES >>> 12;       // 1 byte per 4 KiB guest page (384 KiB)
let protActive: WebAssembly.Global | null = null;      // created with the non-threaded memory
let protBase: WebAssembly.Global | null = null;        // fixed at PROT_TABLE_BASE once created
let protReady = false;                                  // permission table initialized?
let pendingFault: { addr: number; isWrite: number } | null = null;
const memFault = (addr: number, isWrite: number): void => { pendingFault = { addr, isWrite }; throw MEM_FAULT; };

function resetPageProt(): void {
  if (protActive) protActive.value = 0;
  protReady = false;
  try { new Int32Array(memory.buffer, PROT_PAINTED_FLAG, 1)[0] = 0; } catch {}
}

// a page of a file mapping that lies past the end of the file has no backing store. touching one
// is not a permission problem, so it is not SIGSEGV: the mapping is perfectly legal and the memory
// simply is not there, which linux reports as SIGBUS. programs use exactly this to notice that a
// file they mapped has been truncated underneath them.
const PAGE_NO_BACKING = 8;

/** has this process's permission table been painted? shared, so threads agree. */
function protPainted(): boolean {
  return protReady || (threaded && new Int32Array(memory.buffer, PROT_PAINTED_FLAG, 1)[0] === 1);
}

/** claim the right to paint. exactly one thread wins; the losers just use what it painted. */
function claimPaint(): boolean {
  if (!threaded) { if (protReady) return false; protReady = true; return true; }
  const flag = new Int32Array(memory.buffer, PROT_PAINTED_FLAG, 1);
  const won = Atomics.compareExchange(flag, 0, 0, 1) === 0;
  protReady = true;
  return won;
}

/**
 * Fill the permission table in from what the process actually has.
 *
 * nothing is reachable until something maps it. the table used to start out all-accessible and
 * only record explicit holes, so a pointer into address space that was never mapped at all quietly
 * read and wrote real memory instead of faulting. start from nothing and write down what exists:
 * everything below the mmap arena is the process's fixed layout (image, stack, tls, the dynamic
 * loader), and above it the vma set is the whole truth.
 */
function paintProtTable(): void {
  const base = PROT_TABLE_BASE;
  const bytes = new Uint8Array(memory.buffer);
  bytes.fill(0, base, base + PROT_TABLE_SIZE);
  syscalls.paintProtTable(GUEST_STACK_TOP, (s, e, p) => {
    const f = s >>> 12, l = (e + 4095) >>> 12;
    const pb = p & 15;
    for (let q = f; q < l; q++) bytes[base + q] = pb;
  });
}

function setPageProt(addr: number, len: number, prot: number): void {
  if (!protActive || !protBase) return;
  const b = prot & 15;
  if ((b & 3) === 3 && !protPainted()) return;
  const base = PROT_TABLE_BASE;
  const bytes = new Uint8Array(memory.buffer);
  if (!protPainted() && claimPaint()) paintProtTable();
  const first = addr >>> 12;
  const last = (addr + len + 4095) >>> 12;
  for (let p = first; p < last; p++) bytes[base + p] = b;
  protActive.value = 1;
}

// ── Message handler ─────────────────────────────────────────────────────────

// a worker that throws surfaces to the kernel as an ErrorEvent, and an ErrorEvent from a module
// worker carries no message -- every crash read "crashed: undefined", which says only that
// something went wrong somewhere. catch it where the stack still exists and send that instead.
self.onmessage = (e: MessageEvent) => {
  try {
    dispatchMessage(e.data);
  } catch (err: any) {
    (self as any).postMessage({ type: 'worker_error',
      detail: `${err?.name ?? 'Error'}: ${err?.message ?? err}\n${err?.stack ?? ''}` });
    throw err;
  }
};

const dispatchMessage = (msg: any) => {
  switch (msg.type) {
    case 'init':
      init(msg.config);
      break;
    case 'fork_result':
      // Wake from Atomics.wait — handled via mailbox, this is a fallback notification
      break;
    case 'stdin_data':
      // Data was written to our stdin ring — we'll pick it up in the read syscall
      break;
    case 'signal': {
      // Signal delivered — we check signal_pending in the run loop
      break;
    }
    case 'term_size':
      if (syscalls) syscalls.setTermSize(msg.cols, msg.rows);
      break;
    case 'set_trace':
      tracing = !!msg.on;
      break;
    case 'promote':
      // the kernel created (or reused) the shared memory for our first thread; finish the yielded clone.
      completePromote(msg.sharedMemory, msg.tid, msg.regBaseAddr, msg.staleHigh, msg.layout);
      break;
    case 'clone_fail':
      completePromoteFail();
      break;
  }
};

// ── Initialization ──────────────────────────────────────────────────────────

// where a process's startup time actually goes. one line per process, only when tracing is on.
// this exists because "process creation costs 25 ms" is not an actionable statement: a
// one-instruction guest program pays the same 25 ms as busybox, so half of it is fixed cost
// somewhere in fork/init/exec and the only way to know where is to timestamp it.
const tStage: Record<string, number> = {};
function stamp(name: string): void { tStage[name] = performance.now(); }
function reportStages(): void {
  if (!tracing) return;
  const t0 = tStage.init_enter ?? 0;
  const parts = Object.entries(tStage).map(([k, v]) => `${k}=${(v - t0).toFixed(1)}`).join(' ');
  (self as any).postMessage({ type: 'trace', line: `pid${myPid} STAGES ${parts}` });
}

function init(config: WorkerInitConfig) {
  for (const k of Object.keys(tStage)) delete tStage[k];
  stamp('init_enter');
  myPid = config.pid;
  mySlot = config.slot;
  procTable = new Int32Array(config.kernelSAB);
  publishSyscall(-2);   // init: entered

  shmArena = config.shmArena;
  kernelSAB = config.kernelSAB;
  elfBytes = config.elfBytes;

  if (!workerReady) {
    // ── one-time worker setup: the zig translator (prefer the kernel's shared pre-compiled module,
    // no per-worker 1.5MB compile) and the reusable wasm Memory. NON-shared, single-threaded
    // (shared_monitor=0, no wasm atomics), grows lazily so a big `maximum` is free on RSS. ──
    if (config.zigTranslatorModule) {
      initZigTranslatorModule(config.zigTranslatorModule);
    } else if (config.zigTranslatorBytes) {
      initZigTranslator(config.zigTranslatorBytes);
    } else {
      throw new Error(`[worker ${myPid}] no translator module or bytes in init config`);
    }
    memory = new WebAssembly.Memory({ initial: 512, maximum: 65536 }); // 32MB start, 4GiB ceiling
    protActive = new WebAssembly.Global({ value: 'i32', mutable: true }, 0);
    protBase = new WebAssembly.Global({ value: 'i32', mutable: true }, PROT_TABLE_BASE);
    protReady = false;
    workerReady = true;
  } else {
    // ── pool reuse: this Worker just finished another process. wipe the span the outgoing tenant
    // could have dirtied so the incoming process sees a clean slate and no cross-process data
    // leaks. [0, mmapBase] covers it: the ELF, BSS, heap, stack (top 8MB), TLS (9MB) and every
    // mmap (which bump-allocates upward from 33MB) all live below mmapBase. clamp to the committed
    // size. then drop the per-process caches (the new tenant's code differs, possibly at identical
    // PCs). (a finer dirty-tracked wipe is a noted follow-up optimization.)
    // a worker that PROMOTED to shared memory for threads must not hand that shared memory (or the
    // threaded translation mode) to the next pooled tenant — reset to a fresh non-shared process.
    if (threaded) {
      memory = new WebAssembly.Memory({ initial: 512, maximum: 65536 });
      threaded = false; stateBase = 0; regBaseGlobal = null;
      protActive = new WebAssembly.Global({ value: 'i32', mutable: true }, 0);
      protBase = new WebAssembly.Global({ value: 'i32', mutable: true }, PROT_TABLE_BASE);
      protReady = false;
    } else if (protActive) {
      // reused non-shared memory: disarm the guard and drop the permission table for the new tenant
      // (the table lives above mmapBase so the dirty-span wipe below doesn't clear it).
      protActive.value = 0;
      protReady = false;
    }
    const dirtyEnd = Math.min(Number(syscalls.getMmapBase()) + GUEST_MEM_BASE, memory.buffer.byteLength);
    new Uint8Array(memory.buffer).fill(0, 0, dirtyEnd);
    regionCache.clear();
    kernelPipes.clear();
    unixSocks.clear();
    nextLocalFd = 100;
    stdinRing = null;
  }

  // Set up SAB views (slot/mailbox change on every reuse)
  procTable = new Int32Array(kernelSAB);
  sigQueue = new Int32Array(kernelSAB);
  procCreds = new Int32Array(kernelSAB);
  globalWords = new Int32Array(kernelSAB, GLOBAL_STATE_OFFSET, 64);
  bindIoGeneration(globalWords);
  myMailbox = new Int32Array(kernelSAB, mailboxSlotOffset(mySlot), MAILBOX_SLOT_SIZE / 4);

  // on ipad the bridge owns real files (/bin /etc /tmp ...) — vfs only synthesises /proc and /dev.
  // off-ipad (tests, browsers without bridge) vfs is the whole filesystem
  const bridgeHandlesRealFs = !!config.kernelSAB;
  vfs = new InMemoryVFS({ bridgeHandlesRealFs });
  // /proc/<pid> is built from the shared process table, which only the worker can read
  vfs.procListProvider = procSnapshot;
  vfs.sysvipcProvider = sysvipcListing;
  // /proc/self/fd is built from the syscall layer's fd table, which is where descriptors actually
  // live. without this the directory exists and is empty, and anything that counts its own open
  // files by listing it believes it has none.
  stamp('vfs_ready');
  vfs.fdListProvider = () => syscalls?.openFdList() ?? [];
  vfs.memSizeProvider = () => memory.buffer.byteLength - GUEST_MEM_BASE;
  if (!bridgeHandlesRealFs) {
    vfs.loadBinary(elfBytes);
  }

  // ── Thread-start path: this Worker is a THREAD of an existing process (CLONE_THREAD). it imports
  // the process's SHARED memory (not its own) and runs at its own register-block base. ──
  if (config.threadStart) {
    const ts = config.threadStart;
    if (ts.layout) threadLayout = ts.layout; // this thread must use its group's tier
    memory = ts.sharedMemory;
    threaded = true;
    // the permission table lives in the memory the thread group shares, so a thread inherits it
    // for free; what it needs of its own is the pair of globals its instance reads. arm it right
    // away if the group has already painted -- a thread must honour the mappings it was born into.
    protActive = new WebAssembly.Global({ value: 'i32', mutable: true }, 0);
    protBase = new WebAssembly.Global({ value: 'i32', mutable: true }, PROT_TABLE_BASE);
    if (new Int32Array(memory.buffer, PROT_PAINTED_FLAG, 1)[0] === 1) {
      protReady = true;
      protActive.value = 1;
    }
    // the group's MAP_SHARED mappings. a thread shares the guest memory they live in, but not this
    // worker's idea of WHERE they are -- and without that a thread's writes never reached the arena,
    // so a result written from a thread was invisible to the process that forked it, and a futex on
    // a shared page keyed differently in each thread and woke nobody.
    sharedMaps = (((ts as any).shmMaps ?? []) as any[]).map((m: any) => {
      const start = m.addr + GUEST_MEM_BASE;
      const inherited = start + m.len <= memory.buffer.byteLength
        ? new Uint8Array(new Uint8Array(memory.buffer, start, m.len))
        : new Uint8Array(new Uint8Array(shmArena, m.off, m.len));
      return { addr: m.addr, len: m.len, arenaOff: m.off, shadow: inherited };
    });
    sysvAttached = new Map(((ts as any).shmMaps ?? [])
      .filter((m: any) => m.sysv !== undefined)
      .map((m: any) => [m.addr, m.sysv!] as [number, number]));
    if ((ts as any).unixSocksSnapshot) {
      for (const [fd, sock] of (ts as any).unixSocksSnapshot as Array<[number, any]>) unixSocks.set(fd, { ...sock });
    }
    stateBase = ts.regBaseAddr;
    regBaseGlobal = new WebAssembly.Global({ value: 'i32', mutable: false }, ts.regBaseAddr);
    // start from a copy of the parent thread's registers, then apply clone semantics.
    new Uint8Array(memory.buffer).copyWithin(stateBase, ts.parentRegBase, ts.parentRegBase + STATE_SIZE);
    cpu = new CpuState(memory, stateBase);
    cpu.setReg(0, 0n);                                   // child returns 0 from clone
    cpu.setSP(BigInt(ts.childStack >>> 0));              // thread's own stack
    new DataView(memory.buffer).setBigUint64(stateBase + 0x118, BigInt(ts.tls >>> 0), true); // TPIDR
    if ((ts.flags & 0x01000000) && ts.childTidPtr) {    // CLONE_CHILD_SETTID
      new DataView(memory.buffer).setInt32(ts.childTidPtr + GUEST_MEM_BASE, ts.tid, true);
    }
    if ((ts.flags & 0x00200000) && ts.childTidPtr) clearChildTid = ts.childTidPtr; // CLONE_CHILD_CLEARTID
    // the thread shares the group's open files, cwd, creds and heap/mmap layout (CLONE_FILES|
    // CLONE_FS|CLONE_VM). we inherit a copy of the parent's syscall state captured at clone time,
    // so an fd opened before pthread_create is usable in the thread. NOTE: this is initial sharing;
    // fd/mmap changes made AFTER spawn don't yet propagate across threads (a live shared table is
    // the follow-up). native fds are SHARED (not dup'd, unlike fork) — same underlying handle.
    syscalls = newSyscallHandler(0n, {
      writeStdout: (d) => postWrite(1, d),
      writeStderr: (d) => postWrite(2, d),
      readStdin: (count) => readStdin(count),
      stdinHasData: () => stdinRing != null && (!stdinRing.empty || stdinRing.writerClosed),
      waitStdin: (timeoutMs) => waitStdin(timeoutMs),
      ioGeneration: () => readIoGeneration(),
      waitIo: (seen, timeoutMs) => waitIo(seen, timeoutMs),
      unixReady: (fd) => unixReady(fd),
      pipeReady: (fd) => pipeReady(fd),
      inotifyPending: (fd) => { const id = syscalls.inotifyIdOf(fd); return id < 0 ? 0 : inotifyCall({ op: 'pending', id })[1]; },
      mqReady: (fd) => mqReady(fd),
      pidExited: (pid) => pidExited(pid),
      readFileAt: (fd, offset, count) => readNativeFileAt(fd, offset, count),
    }, vfs);
    syscalls.bridgeAvailable = true;
    syscalls.eventfdHook = eventfdCall;
    if (ts.syscallState) syscalls.restoreState(ts.syscallState as any); // inherit fds/cwd/creds/brk
    // ...except the address space, which is not inherited but SHARED: the copy that came with the
    // state is a snapshot, and it goes stale the moment a sibling maps anything.
    syscalls.attachVmaTable(memory, threadLayout.vmaBase);
    syscalls.attachSigTable(memory, threadLayout.sigBase);
    syscalls.setThreadIds(ts.tid, ts.tgid);             // gettid=tid, getpid=tgid (after restore)
    syscalls.maxBytes = threadLayout.regsBase;          // same guest ceiling as the rest of the group
    runToCompletion(BigInt(ts.startPC));
    return;
  }

  publishSyscall(-5);   // init: state restored, about to work out where to start
  let pc: bigint;

  if (config.memSnapshot) {
    // ── Fork path: restore parent's sparse memory snapshot (only touched pages travel) ──
    stamp('snapshot_begin');
    publishSyscall(-3);   // init: applying the fork snapshot
    applySparseSnapshot(memory, config.memSnapshot);
    publishSyscall(-4);   // init: snapshot in place
    // the block those pages came out of is ours now, and we are done with it.
    if (config.memSnapshot.stageOff !== undefined) (self as any).postMessage({ type: 'fork_ready' });
    stamp('snapshot_done');
    cpu = new CpuState(memory, stateBase);

    // fork's child returns 0; a thawed process is resuming its own life and returns nothing
    if (!config.resume) {
      cpu.setReg(0, 0n);
      // clone(2)'s child context: a stack of its own, its own TLS, and its tid written where it
      // asked for it. a libc's __clone stub stores the entry function at the top of the stack it
      // passed and reloads it from sp in the child, so a child that keeps the parent's sp branches
      // into whatever happened to be there.
      const cStack = BigInt(config.childStack ?? '0');
      if (cStack !== 0n) cpu.setSP(cStack);
      const cFlags = Number(config.cloneFlags ?? 0);
      if (cFlags & 0x00080000 /* CLONE_SETTLS */) {
        new DataView(memory.buffer).setBigUint64(stateBase + TPIDR_OFFSET, BigInt(config.childTls ?? '0'), true);
      }
      const cCtid = Number(BigInt(config.childCtid ?? '0'));
      if ((cFlags & (0x01000000 /* CHILD_SETTID */ | 0x00200000 /* CHILD_CLEARTID */)) && cCtid !== 0) {
        if (cFlags & 0x01000000) new DataView(memory.buffer).setInt32(cCtid + GUEST_MEM_BASE, myPid, true);
      }
    }

    // Restore syscall state
    syscalls = newSyscallHandler(0n, {
      writeStdout: (d) => postWrite(1, d),
      writeStderr: (d) => postWrite(2, d),
      readStdin: (count) => readStdin(count),
      stdinHasData: () => stdinRing != null && (!stdinRing.empty || stdinRing.writerClosed),
      waitStdin: (timeoutMs) => waitStdin(timeoutMs),
      ioGeneration: () => readIoGeneration(),
      waitIo: (seen, timeoutMs) => waitIo(seen, timeoutMs),
      unixReady: (fd) => unixReady(fd),
      pipeReady: (fd) => pipeReady(fd),
      inotifyPending: (fd) => { const id = syscalls.inotifyIdOf(fd); return id < 0 ? 0 : inotifyCall({ op: 'pending', id })[1]; },
      mqReady: (fd) => mqReady(fd),
      pidExited: (pid) => pidExited(pid),
      readFileAt: (fd, offset, count) => readNativeFileAt(fd, offset, count),
    }, vfs);
    syscalls.bridgeAvailable = true; // Worker can relay bridge calls to kernel
    syscalls.eventfdHook = eventfdCall;
    if (config.syscallState) {
      syscalls.restoreState(config.syscallState as any);
      // a thawed process restored its OWN state, so its identity and its descriptors are already
      // its own -- re-adopting an identity or re-dup'ing would give it a second copy of every fd
      if (!config.resume) {
      // restoreState brought over the parent's pid; adopt our own identity, with
      // the parent's pid (just restored) as our ppid
      syscalls.setIdentity(myPid, syscalls.currentPid);
      // the host-side references for our inherited descriptors were taken by the parent, in
      // handleFork, before fork(2) returned -- see the note there for why they cannot be taken here.
      }
    }

    // Restore kernel pipes map from parent. the refcounts for these inherited ends were
    // already bumped synchronously by the parent in handleFork (see the note there), so we
    // must NOT bump them again here or every pipe end would be over-counted by one.
    // the unix sockets came across with the fd table, and their rings were already counted for us
    // by the parent -- so this only has to remember which rings each descriptor names.
    if ((config as any).unixSocksSnapshot) {
      for (const [fd, sock] of (config as any).unixSocksSnapshot as Array<[number, any]>) {
        unixSocks.set(fd, { ...sock });
      }
    }
    if (config.kernelPipesSnapshot) {
      for (const [fd, info] of config.kernelPipesSnapshot) {
        kernelPipes.set(fd, { ...info });
      }
      for (const fd of kernelPipes.keys()) {
        if (fd >= nextLocalFd) nextLocalFd = fd + 1;
      }
    }

    // the page-protection table came across with the copied memory; turn the guard back on so the
    // child actually honours the mappings it inherited (PROT_NONE ranges, unmapped holes).
    if (config.protArmed && protActive) { protReady = true; protActive.value = 1; }

    // the shadow is "what this process and the arena last agreed on", and for a fresh child that
    // is its INHERITED memory, not whatever the arena holds by the time the worker starts. seeding
    // it from the arena made every byte the parent wrote between the fork and the child's first
    // syscall look like a write BY THE CHILD of the older value -- so the child published the stale
    // byte back and the parent's write vanished. (a parent filling an array of child pids loses the
    // last entry every time, which is exactly how this was found.)
    sharedMaps = (config.shmMaps ?? []).map((m) => {
      const start = m.addr + GUEST_MEM_BASE;
      const inherited = start + m.len <= memory.buffer.byteLength
        ? new Uint8Array(new Uint8Array(memory.buffer, start, m.len))
        : new Uint8Array(new Uint8Array(shmArena, m.off, m.len));
      return { addr: m.addr, len: m.len, arenaOff: m.off, shadow: inherited };
    });
    sysvAttached = new Map((config.shmMaps ?? [])
      .filter((m) => m.sysv !== undefined)
      .map((m) => [m.addr, m.sysv!] as [number, number]));

    // the child occupies a different slot, so the name and argv it inherited have to be written
    // into it -- until it execs, a fork really is running the same command line.
    publishIdentity(config.argv?.length ? config.argv : currentArgv);

    // fork resumes after its SVC; a thaw resumes exactly where the freeze caught it
    pc = config.resume ? cpu.getPC() : cpu.getPC() + 4n;
  } else {
    // ── Exec path: fresh binary load (static or dynamic via loadProgram) ──
    publishIdentity(config.argv);
    stamp('elf_begin');
    const prog = loadProgram(memory, elfBytes, config.argv, config.envp, readGuestFile);
    stamp('elf_done');
    // loadProgram may have grown (detached) memory; take fresh views
    cpu = new CpuState(memory, stateBase);
    cpu.setSP(prog.sp);
    new DataView(memory.buffer).setBigUint64(stateBase + 0x118, BigInt(0x900000), true); // TPIDR_OFFSET

    syscalls = newSyscallHandler(prog.brkStart, {
      writeStdout: (d) => postWrite(1, d),
      writeStderr: (d) => postWrite(2, d),
      readStdin: (count) => readStdin(count),
      stdinHasData: () => stdinRing != null && (!stdinRing.empty || stdinRing.writerClosed),
      waitStdin: (timeoutMs) => waitStdin(timeoutMs),
      ioGeneration: () => readIoGeneration(),
      waitIo: (seen, timeoutMs) => waitIo(seen, timeoutMs),
      unixReady: (fd) => unixReady(fd),
      pipeReady: (fd) => pipeReady(fd),
      inotifyPending: (fd) => { const id = syscalls.inotifyIdOf(fd); return id < 0 ? 0 : inotifyCall({ op: 'pending', id })[1]; },
      mqReady: (fd) => mqReady(fd),
      pidExited: (pid) => pidExited(pid),
      readFileAt: (fd, offset, count) => readNativeFileAt(fd, offset, count),
    }, vfs);
    syscalls.bridgeAvailable = true;
    syscalls.eventfdHook = eventfdCall;
    syscalls.adoptImage(prog.segs); // the loader's segments are mappings too
    // adopt the pid/ppid the kernel assigned. without this the handler kept its default (pid 1), so
    // any exec-spawned process other than the boot shell reported getpid()==1 — and a child it forked
    // then saw a getppid() that didn't match its parent (ltp's "Main test process might have exit!").
    syscalls.setIdentity(config.pid, config.ppid);

    pc = prog.entry;
  }

  // Set up stdin ring if allocated
  const stdinRingIdx = Atomics.load(procTable, (procSlotOffset(mySlot) / 4) + 7); // PROC_STDIN_RING
  if (stdinRingIdx >= 0) {
    stdinRing = new PipeRing(kernelSAB, stdinRingOffset(stdinRingIdx), STDIN_RING_SIZE);
  }

  if (config.termCols && config.termRows) {
    syscalls.setTermSize(config.termCols, config.termRows);
  }

  // install the rt_sigreturn trampoline at guest 0x1000. this was only ever done in the node
  // harness (dispatch.ts), never here — so in-vivo every signal handler returned into unmapped
  // low memory and hung. covers both the fork-init and exec-init paths.
  syscalls.installSignalTrampoline(memory);

  stamp('run_begin');
  procStartMs = performance.now();
  parkedMs = 0;
  parkDepth = 0;
  reportStages();
  // say what we cost BEFORE running a single guest instruction. the kernel's memory budget is only
  // as good as its worst-informed entry, and a worker it thinks is free is one it will happily
  // over-commit against.
  reportMemSize();
  runToCompletion(pc);
}

// run the guest to termination with the shared crash + fd-cleanup handling. used by every entry into
// the run loop (exec/fork init, thread start, and the async-promote resume) so a crash or exit is
// surfaced + cleaned up identically regardless of which path started it.
function runToCompletion(pc: bigint): void {
  try {
    runLoop(pc);
  } catch (e: any) {
    if (e.message === 'SIGINT') {
      // Already called postExit(130) — just stop
      return;
    }
    console.error(`[worker ${myPid}] CRASH:`, e.message, e.stack);
    // surface the crash on the terminal — otherwise a translator gap or runtime
    // fault during startup is a silent exit(139): "starting sh" then nothing.
    try {
      postWrite(2, new TextEncoder().encode(`\r\n[worker ${myPid}] crash: ${e.message} (last syscall nr=${lastSyscallNr})\r\n`));
    } catch {}
    postExit(139); // SIGSEGV-like exit
  } finally {
    // the native descriptors are released in postExit, BEFORE the exit is announced -- see the
    // comment there. a run loop that ends without reaching postExit (an exception on the way out)
    // still has to release them, and closeNativeFds is idempotent, so this stays as the backstop.
    // EXCEPT a run loop that returned to YIELD for the async thread-promote (pendingClone set): the
    // process is very much alive, waiting on onmessage('promote') to resume — must not touch its fds,
    // and likewise a thread whose siblings still own them.
    if (!pendingClone && !isNonLeaderThread()) {
      try {
        syscalls.closeNativeFds((fd) => handleBridgeCall('fs_close', { fd }, () => {}));
        syscalls.releaseEventfds();
      } catch {}
    }
  }
}

// ── JIT region cache ────────────────────────────────────────────────────────

function getRegion(pc: bigint): CompiledRegion {
  const region = regionCache.get(pc);
  if (region) {
    // bounded-LRU: only reorder (move to newest) once we're at the cap, so a small working set
    // — the common case — pays zero Map churn on the hot dispatch path.
    if (regionCache.size >= REGION_CACHE_CAP) { regionCache.delete(pc); regionCache.set(pc, region); }
    return region;
  }

  const view = new DataView(memory.buffer);
  const shadowBase = threaded ? threadLayout.shadowBase : ZIG_SHADOW_BASE;
  // non-threaded: always compile the guard in (it reads protActive at runtime — near-zero cost while
  // the flag is 0), so no cache invalidation is needed when a later mprotect/PROT_NONE arms it.
  const protection = (protActive && protBase) ? { protBase, protActive, memFault } : null;
  const zigRegion = translateRegionZig(view, pc, memory, shadowBase, false, threaded, regBaseGlobal, protection);
  if (!zigRegion) {
    const insn = view.getUint32(Number(pc) + GUEST_MEM_BASE, true);
    throw new Error(`[worker ${myPid}] translator failed at pc=0x${pc.toString(16)} (entry insn 0x${insn.toString(16).padStart(8, '0')})`);
  }
  // evict the least-recently-used block (oldest in Map insertion order) before caching a new one,
  // so a huge/long-running program's translation cache can't grow without bound (a per-worker
  // leak). re-JIT on a later re-hit is the only cost; correctness is unaffected (and it happens
  // to make self-modifying code re-translate correctly).
  if (regionCache.size >= REGION_CACHE_CAP) {
    const oldest = regionCache.keys().next().value;
    if (oldest !== undefined) regionCache.delete(oldest);
  }
  regionCache.set(pc, zigRegion);
  return zigRegion;
}

// ── Main execution loop ─────────────────────────────────────────────────────

// fold the armed interval timer and any cross-worker procTable signal bits into the
// process-local pendingSignals queue, with the siginfo each one arrived with. returns true if a
// signal is now pending.
/**
 * take every signal queued for this process and make it pending.
 *
 * cheap enough to run on every syscall return, which is where it belongs: a signal that arrives
 * while a process is parked has to be seen at the boundary, not at the next periodic poll. a futex
 * wake racing an incoming signal used to leave the signal sitting in the queue -- and a thread that
 * returned from that wake straight into its own exit never came back to collect it.
 */
function drainSignalQueue(): boolean {
  for (const q of sigqDrain(sigQueue, mySlot)) {
    syscalls.raiseSignal(q.signo, { pid: q.pid, uid: q.uid, code: q.code, value: q.value });
  }
  return syscalls.hasPendingDeliverable();
}

function pollAndQueueSignals(): boolean {
  syscalls.checkTimers(); // may push SIGALRM
  // keep this process's cpu accounting visible to everyone else. this is the periodic tick, and
  // ps reading a stale number is how a busy process looks idle -- but the tick also runs inside
  // tight poll loops, and nobody needs the number refreshed more often than they can read it.
  if (performance.now() - lastCpuPublishMs > 50) {
    lastCpuPublishMs = performance.now();
    publishCpuTime();
  }
  // every signal anyone sent this process, oldest first, with the siginfo its sender meant.
  return drainSignalQueue();
}

// ── freeze (memory swap) ────────────────────────────────────────────────────
// the kernel asks a process to freeze by setting PROC_SWAP_REQ and waking it exactly the way a
// signal does. the answer is an image, and once the kernel holds it this Worker is destroyed --
// destroying the WebAssembly.Memory with it is the only thing that actually returns the bytes,
// since wasm memory never shrinks and there is no discard.
//
// the register file rides along for free: a non-threaded process keeps its cpu state at offset 0
// of its own memory, so the sparse snapshot already contains it, PC included.
// set when a park was abandoned because the kernel wants this process frozen. only
// interruptibleWaiter sets it, which is what bounds the blast radius: every other blocking site
// (nanosleep, wait4, the bridge mailbox) parks on a bare Atomics.wait and is therefore never a
// freeze point. that is deliberate -- restarting a nanosleep would sleep the full duration again,
// and a bridge call has a request in flight whose response would be stranded.
let swapPending = false;

// the kernel budgets against JSC's 4 GiB total wasm allowance, and a single-threaded process's
// committed non-shared memory draws on exactly the same pool as a threaded one's shared
// reservation. reporting growth from the run loop's housekeeping point catches every source of it
// -- brk, mmap, program load -- without hooking each grow() call site across three files.
// what we last told the kernel this worker's memory is. it is module state, so a POOLED worker
// carries the previous tenant's figure -- and since a wasm memory never shrinks, the new tenant
// would report nothing at all until it grew 8 MiB past whatever the last one reached. the kernel
// then believed that worker cost ZERO, over-committed against it, and discovered the real ceiling
// as a RangeError instead of an honest ENOMEM. reset it whenever a process starts.
let lastReportedMem = 0;
// whether this process has already been noted as unable to grow. a program that allocates until it
// fails -- which several ltp tests do ON PURPOSE, and any allocator probing its ceiling does too --
// refuses once per attempt, and a line each would bury everything else the guest printed.
let growRefusedNoted = false;

/** tell the kernel what this worker costs. `bytes` is the FOOTPRINT -- what the guest actually
 *  holds -- because wasm growth is lazy and the grown size wildly overstates it. `grown` is the
 *  buffer size, which is what decides whether this Worker is cheap enough to park for reuse. */
function reportMemSize(): void {
  lastReportedMem = memory.buffer.byteLength;
  (self as any).postMessage({
    type: 'mem_size',
    bytes: syscalls.footprintBytes(GUEST_STACK_TOP),
    grown: lastReportedMem,
  });
}

function swapRequested(): boolean {
  return Atomics.load(procTable, (procSlotOffset(mySlot) / 4) + PROC_SWAP_REQ) !== 0;
}

function postSwapImage(resumePC: bigint): void {
  cpu.setPC(resumePC);
  const snapSize = Math.min(
    Number(syscalls.getMmapBase()) + GUEST_MEM_BASE + 0x10000,
    memory.buffer.byteLength,
  );
  (self as any).postMessage({
    type: 'swap_image',
    memSnapshot: buildSparseSnapshot(memory.buffer, snapSize),
    syscallState: syscalls.getState(),
    kernelPipesSnapshot: [...kernelPipes.entries()],
    shmMaps: sharedMaps.map((m) => ({ addr: m.addr, len: m.len, off: m.arenaOff,
                                     sysv: sysvAttached.get(m.addr) })),
    protArmed: protReady,
  });
}

// a futex waiter for indefinite blocking syscalls (pipe read, ...) that yields on signals.
// parks on the given futex for a bounded slice, then checks for a pending signal (an armed
// timer via checkTimers, or an incoming cross-worker kill/term left in the signal queue). returns
// true if a signal is now queued and the syscall should abandon its wait with EINTR. the 1s
// (or timer-deadline) cap bounds how long an incoming signal can go unobserved, since it
// notifies the proc slot's wait futex, not the ring futex this parks on.
/**
 * Mark this process as parked in a blocking syscall, or awake again.
 *
 * /proc/<pid>/stat's third field is not decoration. ltp's TST_PROCESS_STATE_WAIT (and anything
 * else that waits for a child to settle before poking it) spins until the child reads 'S', so a
 * kernel that always says 'R' turns every such wait into a hang -- which is what pipe13 was,
 * stuck at TWO children rather than the hundred it eventually asks for.
 */
function setSleeping(on: boolean): void {
  // every blocking wait in the worker funnels through here, on both edges, which makes it the one
  // place that can tell "this process was running" from "this process was parked". cpu time is the
  // difference: a process asleep in wait4 for a minute has burned no cpu, and rusage/times must not
  // claim it did. the depth counter is because these nest (a bridge call made while parked).
  if (on) {
    if (parkDepth++ === 0) parkStartMs = performance.now();
  } else if (parkDepth > 0 && --parkDepth === 0) {
    parkedMs += performance.now() - parkStartMs;
  }
  Atomics.store(procTable, (procSlotOffset(mySlot) / 4) + PROC_BLOCKED, on ? 1 : 0);
}

/**
 * cpu time this process has consumed, in milliseconds: wall time since it started running minus
 * every millisecond it spent parked.
 *
 * this is one number, not the two (user, system) that rusage wants, because the host cannot give
 * us two: WKWebView clamps performance.now() to about a millisecond, and a syscall costs tens of
 * microseconds, so timing the syscall layer separately would measure the clamp rather than the
 * work. attributing it all to user time is the honest split -- it is where nearly all of it goes,
 * and it never claims time the process did not spend.
 */
function cpuMs(): number {
  return Math.max(0, (performance.now() - procStartMs) - parkedTotalMs());
}

/** every millisecond this process has spent parked, including the park it may be in right now. */
function parkedTotalMs(): number {
  return parkedMs + (parkDepth > 0 ? performance.now() - parkStartMs : 0);
}

// system time: the part of the process's cpu time spent inside the kernel, which here means
// inside syscall handling. it is measured rather than guessed because a program can see the
// difference -- times(2) reports the two separately, and a shell's `time` prints both.
let sysAccumMs = 0;
let sysEnterMs = -1;
let sysEnterParkedMs = 0;
function syscallEnter(): void {
  sysEnterMs = performance.now();
  sysEnterParkedMs = parkedTotalMs();
}

/** publish which syscall this process is in, for /proc/<pid>/syscall. -1 means "its own code". */
function publishSyscall(nr: number): void {
  Atomics.store(procTable, (procSlotOffset(mySlot) / 4) + PROC_SYSCALL, nr | 0);
}
function sysMs(): number { return Math.min(sysAccumMs, cpuMs()); }

let lastCpuPublishMs = 0;

/** publish this process's cpu accounting where its reaper can find it after the worker is gone. */
function publishCpuTime(): void {
  const b = procSlotOffset(mySlot) / 4;
  const sys = sysMs();
  Atomics.store(procTable, b + PROC_UTIME_MS, Math.round(cpuMs() - sys) | 0);
  Atomics.store(procTable, b + PROC_STIME_MS, Math.round(sys) | 0);
  // ...and the high-water resident size, for whoever reaps this process: wait4 hands the parent a
  // struct rusage FOR THE CHILD, and by then the child's memory is gone.
  const kb = syscalls.rssKbHook ? syscalls.rssKbHook() : 0;
  if (kb > Atomics.load(procTable, b + PROC_MAXRSS_KB)) {
    Atomics.store(procTable, b + PROC_MAXRSS_KB, kb | 0);
  }
  // ...and who this process belongs to, for the calls that name a user rather than a pid.
  Atomics.store(procTable, b + PROC_RUID, syscalls.getCreds().ruid | 0);
}

/** park on a futex with the process marked sleeping for as long as it is parked. */
function sleepingWait(i32: Int32Array, idx: number, expected: number, timeout?: number): 'ok' | 'not-equal' | 'timed-out' {
  setSleeping(true);
  try {
    return timeout === undefined ? Atomics.wait(i32, idx, expected) : Atomics.wait(i32, idx, expected, timeout);
  } finally {
    setSleeping(false);
  }
}

/**
 * Wait for the kernel's reply to land in this worker's mailbox.
 *
 * Every one of these used to be an UNBOUNDED Atomics.wait. That makes a lost wakeup fatal and
 * silent: the process stops forever, still marked running, and nothing in the system can notice or
 * recover -- a whole ltp sweep sat on one for ten minutes with three other processes waiting behind
 * it. A reply that is already in the mailbox is not lost data, only a lost NOTIFY, so waiting in
 * bounded slices and re-reading the flag turns "hangs forever" into "costs a few milliseconds".
 *
 * The process is marked sleeping while parked, so /proc tells the truth about it, and a wait that
 * runs implausibly long is reported rather than endured in silence.
 */
function awaitMailbox(what: string): void {
  const start = performance.now();
  let warned = false;
  setSleeping(true);
  try {
    while (Atomics.load(myMailbox, MBOX_READY) === 0) {
      Atomics.wait(myMailbox, MBOX_READY, 0, 100);
      if (!warned && performance.now() - start > 20000) {
        warned = true;
        (self as any).postMessage({ type: 'worker_error',
          detail: `pid ${myPid} has waited 20s for a '${what}' reply from the kernel` });
      }
    }
  } finally {
    setSleeping(false);
  }
}

function interruptibleWaiter(i32: Int32Array, idx: number, expected: number): boolean {
  const rem = syscalls.timerRemainingMs();
  sleepingWait(i32, idx, expected, rem > 0 ? Math.min(1000, rem) : 1000);
  if (swapRequested()) { swapPending = true; return true; } // abandon the wait, freeze at the svc
  return pollAndQueueSignals();
}

/// a freeze requested while the process was parked in a syscall can only be taken where the
/// syscall did no observable work, which is exactly the case it reports as -EINTR: PipeRing.read
/// only parks on an empty ring, and writeAll answers -2 (mapped to -EINTR) only when it has
/// written nothing. a partial write returns its positive count instead, and that must not be
/// re-executed -- the same reason linux restricts ERESTARTSYS to the nothing-transferred case.
///
/// when it is safe, the pc rewinds to the SVC, so on thaw the syscall simply runs again and the
/// guest never observes an EINTR it did not earn.
function takeSwapAtSyscall(svcPC: bigint): boolean {
  if (!swapPending) return false;
  swapPending = false;
  if (BigInt.asIntN(64, cpu.getReg(0)) !== -4n) return false; // work was done; the run loop will
  postSwapImage(svcPC);
  return true;
}

// deliver queued signals at a safe point (pc = next instruction). returns the pc to
// continue from, or null if the process terminated (caller must return from runLoop).
/**
 * stop this process until somebody continues it.
 *
 * a stopped process is not gone: it keeps its slot, its memory and its descriptors, and picks up at
 * the exact instruction it was about to run. its parent has to be able to see the stop, so the state
 * goes in the shared table and the parent is woken to look.
 */
function stopSelf(sig: number): void {
  const base = procSlotOffset(mySlot) / 4;
  Atomics.store(procTable, base + PROC_STOP_SIG, sig);
  Atomics.store(procTable, base + PROC_STOP_SEEN, 0);
  Atomics.store(procTable, base + PROC_STATE, PROC_STOPPED);
  wakeParent();
  // park until somebody puts the state back. the wait futex is the same one the parent's wait uses,
  // so a SIGCONT and a reaping parent both reach us through one word.
  while (Atomics.load(procTable, base + PROC_STATE) === PROC_STOPPED) {
    Atomics.store(procTable, base + PROC_WAIT_FUTEX, 0);
    if (Atomics.load(procTable, base + PROC_STATE) !== PROC_STOPPED) break;
    sleepingWait(procTable, base + PROC_WAIT_FUTEX, 0, 1000);
    // SIGKILL is the one thing that gets a stopped process moving without continuing it. the
    // kernel enforces it by ending the process, which clears the slot -- so an emptied slot is
    // the signal here.
    if (Atomics.load(procTable, base + PROC_PID) <= 0) break;
  }
  Atomics.compareExchange(procTable, base + PROC_STATE, PROC_STOPPED, PROC_RUNNING);
}

/** nudge whoever is waiting on us, so a parent parked in wait() looks again. */
function wakeParent(): void { wakeParentOf(myPid); }

/**
 * tell the parent of `pid` that something happened to its child: raise SIGCHLD and wake it.
 *
 * a stop and a continue are child state changes exactly as an exit is, and a parent learns about
 * them the same two ways -- the signal, and coming back from wait().
 */
function wakeParentOf(pid: number): void {
  let ppid = -1;
  for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
    const base = procSlotOffset(slot) / 4;
    if (Atomics.load(procTable, base + PROC_PID) === pid) { ppid = Atomics.load(procTable, base + PROC_PPID); break; }
  }
  if (ppid <= 0) return;
  wakeWatcher(ppid);
}

/** raise SIGCHLD in `pid` and wake it: something it is waiting for has changed state. */
function wakeWatcher(pid: number): void {
  if (pid <= 0) return;
  for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
    const base = procSlotOffset(slot) / 4;
    if (Atomics.load(procTable, base + PROC_PID) !== pid) continue;
    // SIGCHLD, from the kernel's own account: a child changed state, and no user sent it.
    sigqPush(sigQueue, slot, { signo: 17, code: 0x80 /* SI_KERNEL */, pid: 0, uid: 0, value: 0n });
    Atomics.store(procTable, base + PROC_WAIT_FUTEX, 1);
    Atomics.notify(procTable, base + PROC_WAIT_FUTEX);
    return;
  }
}

function deliverQueuedSignals(pc: bigint): bigint | null {
  if (!syscalls.pendingSignals.length) return pc;
  // the disposition belongs to the thread GROUP: a sibling may have installed a handler since our
  // last look, and this is the moment it decides what happens. syscalls that never reach the
  // generic handler (pidfd_send_signal, the sched family) would otherwise deliver against a stale
  // copy -- which is a process killed by a signal it had plainly asked to catch.
  syscalls.sigSync();
  cpu.setPC(pc);
  // a BLOCKED signal stays pending. it was being taken off the queue and then dropped on the floor,
  // which is the one thing a signal mask must never do: blocking says "not yet", not "never". the
  // classic victim is raise(), which musl implements as block-everything, tkill, restore -- so the
  // signal arrived while blocked, was discarded, and the handler never ran. that makes raise() a
  // silent no-op, and abort() with it, since abort() is raise(SIGABRT).
  const blocked: number[] = [];
  let result: bigint | null = pc;
  const traceBase = procSlotOffset(mySlot) / 4;
  while (syscalls.pendingSignals.length > 0) {
    let sig = syscalls.pendingSignals.shift()!;
    // a traced process stops here instead of acting on the signal, and its tracer decides what
    // happens next: let it through, swap it for another, or swallow it. SIGKILL is nobody's to
    // hold up.
    if (sig !== 9 && Atomics.load(procTable, traceBase + PROC_TRACER) !== 0) {
      sig = ptraceStop(sig);
      if (sig === 0) continue;
      if (Atomics.load(procTable, traceBase + PROC_PID) <= 0) return null;
    }
    const r = syscalls.deliverSignal(cpu, memory, sig);
    if (r.action === 'exit') { postExit(r.exitCode ?? 128 + sig); return null; }
    if (r.action === 'stop') { stopSelf(r.exitCode ?? sig); continue; }
    if (!r.delivered) { blocked.push(sig); continue; }
    if (r.action === 'continue') { result = cpu.getPC(); break; } // handler installed
  }
  if (blocked.length) syscalls.pendingSignals.unshift(...blocked);
  return result;
}

function runLoop(startPC: bigint) {
  let pc = startPC;
  let totalInsns = 0;
  let view = new DataView(memory.buffer);
  let lastSignalCheck = 0;
  let blockCount = 0;
  syscalls.protHook = setPageProt; // arm mmap/mprotect -> page-protection table (non-threaded no-ops)
  syscalls.growFailed = (cur, min, want, cap, err) => {
    // running out of memory is an ordinary ENOMEM, not a fault: the guest gets the errno and
    // decides what to do. report the shape of the refusal once, to the log rather than to the
    // user's terminal, and let the refusal counter carry the rest.
    if (growRefusedNoted) return;
    growRefusedNoted = true;
    (self as any).postMessage({ type: 'grow_refused',
      detail: `pid ${myPid} grow refused: have ${cur} pages, need ${min}, asked ${want}, cap ${cap} (${err})` });
  };
  syscalls.leaseHook = (op, nativeFd, path, arg, breakSig, isOwner, isRegular) => {
    Atomics.store(myMailbox, MBOX_READY, 0);
    (self as any).postMessage({ type: 'lease_req', op, path, nativeFd, arg, breakSig, isOwner, isRegular });
    awaitMailbox('lease');
    return Atomics.load(myMailbox, MBOX_VALUE0);
  };
  syscalls.credsHook = (ruid, euid, suid, rgid) => {
    const b = procCredOffset(mySlot) / 4;
    Atomics.store(procCreds, b, ruid | 0);
    Atomics.store(procCreds, b + 1, euid | 0);
    Atomics.store(procCreds, b + 2, suid | 0);
    Atomics.store(procCreds, b + 3, rgid | 0);
  };
  syscalls.tracedHook = () =>
    Atomics.load(procTable, (procSlotOffset(mySlot) / 4) + PROC_TRACER) !== 0;
  syscalls.sigpendingHook = (limit) => {
    Atomics.store(procTable, (procSlotOffset(mySlot) / 4) + PROC_RLIM_SIGPENDING, limit | 0);
  };
  syscalls.cpuMsHook = cpuMs;
  syscalls.sysMsHook = sysMs;
  // CLOCK_MONOTONIC counts from the system's boot, not from this worker's construction.
  // the wall-clock offset is system-wide: one process setting the time is every process's clock
  // moving, and the seqlock is what keeps a reader from catching a half-written value.
  syscalls.clockOffsetHook = {
    get() {
      for (;;) {
        const s1 = Atomics.load(globalWords, GLOBAL_CLK_SEQ);
        if (s1 & 1) continue;
        const sec = Atomics.load(globalWords, GLOBAL_CLK_OFF_SEC);
        const nsec = Atomics.load(globalWords, GLOBAL_CLK_OFF_NSEC);
        if (Atomics.load(globalWords, GLOBAL_CLK_SEQ) === s1) return BigInt(sec) * 1_000_000_000n + BigInt(nsec);
      }
    },
    set(ns: bigint) {
      Atomics.add(globalWords, GLOBAL_CLK_SEQ, 1);
      Atomics.store(globalWords, GLOBAL_CLK_OFF_SEC, Number(ns / 1_000_000_000n) | 0);
      Atomics.store(globalWords, GLOBAL_CLK_OFF_NSEC, Number(ns % 1_000_000_000n) | 0);
      Atomics.add(globalWords, GLOBAL_CLK_SEQ, 1);
    },
  };
  syscalls.setSystemBootMs(
    (Atomics.load(globalWords, GLOBAL_BOOT_MS_HI) * 0x100000000) +
    (Atomics.load(globalWords, GLOBAL_BOOT_MS_LO) >>> 0));
  // reboot(HALT/POWER_OFF/RESTART) stops the guest system: the kernel already treats the death of
  // pid 1 as the end of everything, so telling it to end init is the whole shutdown.
  syscalls.haltHook = (restart: boolean) => (self as any).postMessage({ type: 'halt', restart });
  // the system's UTS names live in the shared global page, so uname() reads what the last
  // sethostname() wrote no matter which process made the call -- and with no round trip.
  const utsBytes = new Uint8Array(kernelSAB, GLOBAL_STATE_OFFSET, 4096);
  syscalls.utsHook = {
    get(which) {
      const off = which === 'host' ? GLOBAL_HOSTNAME_OFFSET : GLOBAL_DOMAINNAME_OFFSET;
      let end = off;
      while (end < off + GLOBAL_UTS_LEN && utsBytes[end] !== 0) end++;
      return new TextDecoder().decode(utsBytes.subarray(off, end));
    },
    set(which, name) {
      const off = which === 'host' ? GLOBAL_HOSTNAME_OFFSET : GLOBAL_DOMAINNAME_OFFSET;
      const enc = new TextEncoder().encode(name);
      utsBytes.fill(0, off, off + GLOBAL_UTS_LEN);
      utsBytes.set(enc.subarray(0, GLOBAL_UTS_LEN - 1), off);
    },
  };
  syscalls.inotifyHook = reportInotify;
  syscalls.inotifyFdinfo = inotifyFdinfo;
  syscalls.fdPosition = nativeFdOffset;
  syscalls.pdeathHook = (sig: number) =>
    Atomics.store(procTable, (procSlotOffset(mySlot) / 4) + PROC_PDEATHSIG, sig | 0);
  // PR_SET_NAME renames the thread, and /proc and ps read that name out of the shared table.
  syscalls.commHook = (name: string) => {
    const commBytes = new Uint8Array(16);
    commBytes.set(new TextEncoder().encode(name).subarray(0, 15));
    new Uint8Array(kernelSAB, procSlotOffset(mySlot) + PROC_COMM * 4, 16).set(commBytes);
  };
  // the process table is the only place a process can see other processes, so setpriority and
  // getpriority -- which are defined across processes and process groups -- have to read it here.
  syscalls.prioHook = {
    list: () => {
      const out: Array<{ pid: number; pgid: number; ruid: number; nice: number }> = [];
      for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
        const b = procSlotOffset(slot) / 4;
        const pid = Atomics.load(procTable, b + PROC_PID);
        if (pid === 0) continue;
        const listState = Atomics.load(procTable, b + PROC_STATE);
        if (listState !== PROC_RUNNING && listState !== PROC_STOPPED) continue;
        out.push({
          pid,
          pgid: Atomics.load(procTable, b + PROC_PGID),
          ruid: Atomics.load(procCreds, procCredOffset(slot) / 4),
          nice: Atomics.load(procTable, b + PROC_NICE),
        });
      }
      return out;
    },
    setNice: (pid, nice) => {
      for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
        const b = procSlotOffset(slot) / 4;
        if (Atomics.load(procTable, b + PROC_PID) !== pid) continue;
        Atomics.store(procTable, b + PROC_NICE, nice | 0);
        return;
      }
    },
  };
  syscalls.rssKbHook = () => Math.floor(syscalls.footprintBytes(GUEST_STACK_TOP) / 1024);
  syscalls.publishCreds();
  syscalls.publishSigpending();

  while (pc !== HALT_PC) {
    blockCount++;
    const region = getRegion(pc);
    try {
      pc = region.run();
    } catch (e) {
      if (e !== MEM_FAULT) throw e;
      // which signal depends on WHY the page is unreachable. no permission is SIGSEGV; no backing
      // store behind a legal mapping is SIGBUS. reporting the second as the first tells a program
      // its pointer is wrong when its FILE is short.
      // the engine reports the address it was actually indexing, which includes GUEST_MEM_BASE.
      // everything above the syscall boundary -- si_addr, the mapping tables, the guest's own idea
      // of where it was pointing -- speaks in GUEST addresses, so convert once, here. reporting the
      // raw one put si_addr half a megabyte away from the pointer the program used, and made every
      // lookup against a mapping range miss.
      const rawFault = pendingFault?.addr ?? 0;
      // the 32 bytes leading up to the faulting address, so a trace can say whether the string the
      // guest was walking was terminated or whether it ran on into the guard.
      let faultCtx = '';
      if (tracing) {
        // the instruction that faulted, so the trace says WHAT the guest was doing rather than
        // only where it landed.
        const fpc = Number(view.getBigUint64(stateBase + PC_OFFSET, true));
        if (fpc + GUEST_MEM_BASE + 4 <= memory.buffer.byteLength) {
          const words: string[] = [];
          for (let k = -6; k <= 3; k++) {
            const a = fpc + k * 4 + GUEST_MEM_BASE;
            if (a >= 0 && a + 4 <= memory.buffer.byteLength) {
              words.push((k === 0 ? '*' : '') + view.getUint32(a, true).toString(16).padStart(8, '0'));
            }
          }
          faultCtx += ' text=' + words.join(',');
        }
      }
      if (tracing && rawFault >= 32) {
        const fb = new Uint8Array(memory.buffer, rawFault - 32, 32);
        faultCtx += ' before=' + Array.from(fb).map((b) => b.toString(16).padStart(2, '0')).join('');
      }
      if (tracing) (self as any).postMessage({ type: 'trace', line:
        `pid${myPid} FAULT raw=0x${(rawFault >>> 0).toString(16)} guest=0x${((rawFault - GUEST_MEM_BASE) >>> 0).toString(16)} in=${syscalls.addrInfo((rawFault - GUEST_MEM_BASE) >>> 0)} write=${pendingFault?.isWrite ?? 0} pc=0x${pc.toString(16)} sp=0x${cpu.getSP().toString(16)} tp=0x${new DataView(memory.buffer).getBigUint64(stateBase + TPIDR_OFFSET, true).toString(16)} x=${[0,1,2,3,8,9].map((r) => cpu.getReg(r).toString(16)).join(',')} lr=0x${cpu.getReg(30).toString(16)}${faultCtx}` });
      const fAddr = rawFault >= GUEST_MEM_BASE ? rawFault - GUEST_MEM_BASE : rawFault;
      // a stack mapped MAP_GROWSDOWN grows on demand: the fault below it is the request, not the
      // error. extend it and re-run the instruction, which is what the guest expects to happen.
      // the guard wrote the faulting instruction's address into the state block before trapping.
      // `pc` here is still the BLOCK's entry -- region.run() threw, so its return was never
      // assigned -- and resuming from that would re-run everything between the block start and the
      // fault.
      const faultPC = view.getBigUint64(stateBase + PC_OFFSET, true);
      const grown = syscalls.growStackFor(fAddr);
      if (grown) {
        setPageProt(grown.start, grown.end - grown.start, grown.prot);
        pc = faultPC;      // re-run the instruction that asked for the page
        view = new DataView(memory.buffer);
        continue;
      }
      cpu.setPC(faultPC);  // the handler's frame must name the instruction that faulted
      syscalls.faultAddr = BigInt(fAddr); // siginfo.si_addr for the handler
      // which signal depends on WHY the page is unreachable, and the permission table is the one
      // record of that every thread of the group shares -- a sibling's short file mapping is in the
      // table but not in this worker's own noBacking list.
      const faultPage = protPainted()
        ? new Uint8Array(memory.buffer)[PROT_TABLE_BASE + (fAddr >>> 12)]
        : -1;
      const noBacking = faultPage >= 0 ? faultPage === PAGE_NO_BACKING : syscalls.isNoBacking(fAddr);
      // say where it happened. a fault that kills a process is the single most useful thing to
      // have in the log afterwards, and reconstructing it from an exit status is guesswork.
      (self as any).postMessage({ type: 'klog',
        line: `fault ${noBacking ? 'SIGBUS' : 'SIGSEGV'} addr=0x${fAddr.toString(16)} pc=0x${faultPC.toString(16)} ${syscalls.describeAddr(fAddr)}` });
      syscalls.raiseFaultSignal(noBacking ? 7 /* SIGBUS */ : 11 /* SIGSEGV */);
      const next = deliverQueuedSignals(cpu.getPC());
      if (next === null) return;
      pc = next;
      view = new DataView(memory.buffer);
      continue;
    }

    // Undecodable / unsupported instruction (reason 3): raise SIGILL at the faulting PC instead of
    // looping. this is real CPU behaviour and is what lets feature-probing code (e.g. OpenSSL testing
    // for the AES/PMULL/SHA crypto extensions we don't implement) catch the trap and fall back to its
    // generic path. without it the block re-exits at the same PC forever (a hang).
    if ((region as any).lastExitReason === 3 /* exit_unknown_insn */) {
      // name the instruction we could not decode. an undecodable instruction is a gap in the
      // translator, and the 32 bits at the faulting pc are the whole bug report -- without them the
      // only evidence is "Illegal instruction" and a program that stops.
      if (tracing) {
        let word = 0;
        try { word = new DataView(memory.buffer).getUint32(Number(pc) + GUEST_MEM_BASE, true); } catch {}
        (self as any).postMessage({ type: 'trace', line: `pid${myPid} SIGILL pc=0x${pc.toString(16)} insn=0x${(word >>> 0).toString(16).padStart(8, '0')} ${syscalls.describeAddr(Number(pc))}` });
      }
      cpu.setPC(pc);
      syscalls.faultAddr = pc; // siginfo.si_addr = faulting instruction address
      syscalls.raiseFaultSignal(4); // SIGILL
      const next = deliverQueuedSignals(pc);
      if (next === null) return; // default action terminated the process
      pc = next;
      continue;
    }

    // Refresh view in case memory grew
    view = new DataView(memory.buffer);
    const insns = Number(view.getBigUint64(stateBase + ITER_OFFSET, true));
    totalInsns += insns;
    view.setBigUint64(stateBase + ITER_OFFSET, 0n, true);

    // Check timers + cross-worker signals periodically, so an armed alarm or an incoming
    // kill fires even in a syscall-less spin (this is what bounds a runaway ltp test).
    lastSignalCheck += insns;
    if (lastSignalCheck > 10000) {
      lastSignalCheck = 0;
      // a block that ends in a syscall exits with the SVC_PC SENTINEL, not a guest address -- the
      // syscall branch below is what turns it back into one. so anything here that RECORDS a resume
      // pc (a signal frame's mc_pc, a freeze image) has to use the real address, or the guest comes
      // back from the handler to 0xdead0001 and traps out of bounds. re-entering at the svc itself
      // is the correct resume point: the instruction has not executed yet, so this is just "the
      // signal arrived immediately before the syscall".
      const realPC = pc === SVC_PC ? view.getBigUint64(stateBase + PC_OFFSET, true) : pc;
      // freezing is checked before signals: a process on its way out to swap should not first
      // run a handler whose effects would then be captured mid-flight
      if (swapRequested()) { postSwapImage(realPC); return; }
      if (Math.abs(memory.buffer.byteLength - lastReportedMem) >= 8 * 1024 * 1024) reportMemSize();
      if (pollAndQueueSignals()) {
        const next = deliverQueuedSignals(realPC);
        if (next === null) return; // process terminated
        // nothing was actually installed (everything queued turned out blocked): leave pc alone so
        // a pending syscall still takes the branch below instead of re-translating at the svc.
        if (next !== realPC) { pc = next; continue; }
      }
    }

    // Handle syscall
    if (pc === SVC_PC) {
      const svcPC = view.getBigUint64(stateBase + PC_OFFSET, true);
      syscallEnter();
      publishSyscall(Number(cpu.getReg(8)));
      const nr = Number(cpu.getReg(8));
      // aarch64 keeps the syscall's first argument in orig_x0 precisely because x0 is also where
      // the RESULT goes: a syscall restarted after a signal has to be handed its original argument
      // back, or it re-runs against the errno it just produced. that is how a restarted
      // wait4(pid) became wait4(-EINTR) -- a wait on process group 4.
      const origX0 = cpu.getReg(0);
      lastSyscallNr = nr;
      // the arguments have to be taken BEFORE the call: x0 is both the first argument and the
      // result register, so reading it afterwards reports the answer as if it were the question.
      const tA0 = tracing ? cpu.getReg(0) : 0n;
      const tA1 = tracing ? cpu.getReg(1) : 0n;
      const tA2 = tracing ? cpu.getReg(2) : 0n;
      tA3 = tracing ? cpu.getReg(3) : 0n;
      tA4 = tracing ? cpu.getReg(4) : 0n;
      // record the ENTRY too: a syscall that never returns is exactly the one worth naming, and a
      // tracer that only records completions is blind to it.
      if (tracing) (self as any).postMessage({ type: 'trace', line: `pid${myPid} -> ${SYSNAME[nr] ?? nr}(${cpu.getReg(0)}, ${cpu.getReg(1)}, ${cpu.getReg(2)})` });
      const tPath = (tracing && (nr === 56 || nr === 79 || nr === 35 || nr === 36 || nr === 38 || nr === 37))
        ? readCString(Number(cpu.getReg(1)))
        : null;

      // a syscall is the synchronisation point for MAP_SHARED memory (shared-map.ts): publish
      // what we wrote and observe what everyone else did. this has to sit ABOVE the intercepts --
      // a process blocked in an intercepted read() is exactly the one waiting to see a write.
      if (sharedMaps.length) reconcileShared();

      // Intercept syscalls that need direct kernel SAB access
      if (nr === 260) {
        handleWait4(cpu, view);
        // SA_RESTART: if a signal with SA_RESTART interrupted the wait (EINTR), restart the
        // syscall (return to the SVC after the handler) instead of surfacing EINTR to the guest.
        // tst_test's SIGUSR1 heartbeat relies on this — without it the parent EINTRs out of its
        // wait, never reaps the test child, and hits the 30s watchdog.
        const eintr = BigInt.asIntN(64, cpu.getReg(0)) === -4n;
        syscallDone();
        if (takeSwapAtSyscall(svcPC)) return;
        const restart = eintr && syscalls.hasPendingDeliverable() && syscalls.anyPendingRestart();
        const retPC = restart ? svcPC : svcPC + 4n;
        if (restart) cpu.setReg(0, origX0);
        if (syscalls.hasPendingDeliverable()) {
          const next = deliverQueuedSignals(retPC);
          if (next === null) return;
          pc = next;
        } else {
          pc = retPC;
        }
        continue;
      }
      // rt_sigreturn: restore the pre-signal context. the resume PC comes from the signal frame,
      // NOT svcPC+4 — the generic 'ok' path below would clobber it, sending every handler return
      // into garbage after the trampoline. this is why signal handlers never returned correctly.
      if (nr === 139) {
        syscallDone(); pc = syscalls.handleSigreturn(cpu, memory);
        // returning from a handler IS a return to userspace, and that is where linux looks for the
        // next deliverable signal. without this, two signals unblocked together ran one handler now
        // and the other at whatever syscall came next -- so a program that unblocks and immediately
        // counts its handler calls sees one.
        const next = deliverQueuedSignals(pc);
        if (next === null) return;
        pc = next;
        continue;
      }
      // kill/tkill/tgkill, and the two that carry a whole siginfo of their own:
      // rt_sigqueueinfo and rt_tgsigqueueinfo. all five put a signal in another process's queue.
      if (nr === 129 || nr === 130 || nr === 131 || nr === 138 || nr === 240) {
        handleKill(nr, cpu, view);
        syscallDone(); pc = svcPC + 4n;
        if (drainSignalQueue()) { // kill to self, or a signal that arrived while we sent one
          const next = deliverQueuedSignals(pc);
          if (next === null) return;
          pc = next;
        }
        continue;
      }
      // setpgid/getpgid: process-group membership lives in the shared procTable, so these
      // must run here (they read/write another process's PROC_PGID slot) rather than in the
      // per-process syscall handler. correct groups are what make kill(0)/kill(-pgid) and
      // timeout's child-group isolation actually scope, instead of hitting every process.
      if (nr === 154) { handleSetpgid(cpu); syscallDone(); pc = svcPC + 4n; continue; }
      if (nr === 155) { handleGetpgid(cpu); syscallDone(); pc = svcPC + 4n; continue; }
      // sessions live in the same table for the same reason: setsid has to see every process
      // group in the system to know whether the caller leads one, and getsid names a pid.
      if (nr === 156) { handleGetsid(cpu); syscallDone(); pc = svcPC + 4n; continue; }
      if (nr === 157) { handleSetsid(cpu); syscallDone(); pc = svcPC + 4n; continue; }
      // rt_sigsuspend: really block until a deliverable signal arrives (was a stubbed immediate EINTR,
      // which made pause()-based sync — e.g. kill02's children waiting for the parent's signal — spin
      // or hang). parks on the proc wait futex that handleKill notifies on a cross-worker signal.
      if (nr === 133) {
        handleSigsuspend(cpu);
        syscallDone(); pc = svcPC + 4n;
        const next = deliverQueuedSignals(pc);
        if (next === null) return;
        pc = next;
        continue;
      }
      // ppoll with no fds is how musl implements pause()/timed sigwait on aarch64 (no SYS_pause). the
      // generic handler returned 0 immediately (fds present = poll); with zero fds it must block until a
      // signal or the timeout, else pause() busy-spins and cross-process signal sync (kill02) hangs.
      if (nr === 73 && Number(cpu.getReg(1)) === 0) {
        handlePpollWait(cpu);
        syscallDone(); pc = svcPC + 4n;
        const next = deliverQueuedSignals(pc);
        if (next === null) return;
        pc = next;
        continue;
      }
      // clone(CLONE_THREAD): pthread_create. spawn a real thread-Worker sharing this process's memory
      // (a fork-style clone — SIGCHLD set — still falls through to the per-process fork handler).
      if (nr === 220) {
        const cloneFlags = Number(cpu.getReg(0));
        const CLONE_THREAD = 0x10000;
        if (cloneFlags & CLONE_THREAD) {
          if (handleCloneThread(cpu, svcPC)) return; // yielded for async promote; onmessage resumes
          syscallDone(); pc = svcPC + 4n;
          continue;
        }
      }
      // futex: real cross-thread blocking on the shared heap. only a threaded process has shared
      // memory (Atomics.wait needs a SharedArrayBuffer); single-threaded futex stays the syscall stub.
      // futex on a MAP_SHARED word is how two PROCESSES rendezvous (ltp's checkpoints, and any
      // pthread primitive placed in shared memory), and that works even unthreaded because the
      // arena backing shared mappings is itself a SharedArrayBuffer.
      // SysV shared memory. the registry is system-wide so every one of these is a kernel round
      // trip, which only the worker can make.
      // inotify: init/add/rm, plus read and close on the resulting descriptor. the watch registry
      // is kernel-owned because a watch is about what OTHER processes do to the filesystem.
      if (nr === 26 || nr === 27 || nr === 28 ||
          ((nr === 63 || nr === 57) && syscalls.inotifyIdOf(Number(cpu.getReg(0))) >= 0)) {
        handleInotify(nr, cpu, view);
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      // sendfile: the guest gives no buffer, so the bytes never touch its address space -- they go
      // from one descriptor to the other through the worker.
      if (nr === 95) {
        handleWaitid(cpu, view);
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      if (nr === 71) {
        handleSendfile(cpu, view);
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      // read(2) on a signalfd. it lives here rather than in the generic read path for the same
      // reason the inotify read does: with nothing pending it has to WAIT, and waiting is this
      // side's job.
      if (nr === 63 && syscalls.getFdEntry(Number(cpu.getReg(0)))?.type === 'signalfd') {
        const sfd = Number(cpu.getReg(0));
        const sbuf = Number(cpu.getReg(1));
        const scount = Number(cpu.getReg(2));
        for (;;) {
          const r = syscalls.signalfdRead(sfd, sbuf, scount, view, new Uint8Array(memory.buffer));
          if (r !== -11) { cpu.setReg(0, BigInt.asUintN(64, BigInt(r))); break; }
          if (syscalls.isNonBlockingFd(sfd)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-11))); break; }
          // a signal that is NOT in this descriptor's mask can still interrupt the wait.
          if (pollAndQueueSignals()) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-4))); break; }
          waitIo(readIoGeneration(), 20);
        }
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      // splice, tee and vmsplice: a pipe is an arena ring, so only this side can reach one.
      if (nr === 75 || nr === 76 || nr === 77) {
        handleSplice(nr, cpu, view);
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      // the sched_* family: policy, priority, affinity and deadline attributes. they name a pid
      // rather than the caller, so they read and write the shared process table.
      if ((nr >= 118 && nr <= 123) || nr === 125 || nr === 126 || nr === 127 || nr === 274 || nr === 275 ||
          nr === 30 || nr === 31 || nr === 168) {
        handleSched(nr, cpu, view);
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      // a NEGATIVE clockid is a task's cpu clock -- the task is encoded in the id itself, and only
      // the shared table can say how much cpu that task has had.
      if ((nr === 113 || nr === 114) && Number(BigInt.asIntN(32, cpu.getReg(0))) < 0) {
        handleCpuClock(nr, cpu, view);
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      // pidfd_open / pidfd_send_signal: a descriptor that names a process rather than a pid, so
      // they read the shared process table.
      if (nr === 424 || nr === 434) {
        handlePidfd(nr, cpu, view);
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      // SysV semaphores: semget/semctl/semtimedop/semop. system-wide registry, kernel-owned.
      if (nr >= 190 && nr <= 193) {
        handleSysvSem(nr, cpu, view);
        syscallDone(); pc = svcPC + 4n;
        if (drainSignalQueue()) {
          const next = deliverQueuedSignals(pc);
          if (next === null) return;
          pc = next;
        }
        continue;
      }
      // POSIX message queues: mq_open/unlink/timedsend/timedreceive/notify/getsetattr, plus the
      // close of a descriptor onto one. the queues are named system-wide, so the kernel owns them.
      if ((nr >= 180 && nr <= 185) || (nr === 57 && syscalls.mqIdOf(Number(cpu.getReg(0))) >= 0)) {
        handlePosixMq(nr, cpu, view);
        syscallDone(); pc = svcPC + 4n;
        if (drainSignalQueue()) {
          const next = deliverQueuedSignals(pc);
          if (next === null) return;
          pc = next;
        }
        continue;
      }
      // SysV message queues: msgget/msgctl/msgrcv/msgsnd. system-wide registry, kernel-owned.
      if (nr === 186 || nr === 187 || nr === 188 || nr === 189) {
        handleSysvMsg(nr, cpu, view);
        syscallDone(); pc = svcPC + 4n;
        if (drainSignalQueue()) {
          const next = deliverQueuedSignals(pc);
          if (next === null) return;
          pc = next;
        }
        continue;
      }
      if (nr === 194 || nr === 195 || nr === 196 || nr === 197) {
        handleSysvShm(nr, cpu, view);
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      // copy_file_range between two REAL files: the bytes move through the bridge rather than
      // through the guest, which is the guarantee that matters and the reason the call exists. the
      // generic handler only knows the synthesised filesystem, so it answered EBADF for every file
      // actually on disk.
      if (nr === 285 && syscalls.fdNative(Number(BigInt.asIntN(32, cpu.getReg(0)))) >= 0
                     && syscalls.fdNative(Number(BigInt.asIntN(32, cpu.getReg(2)))) >= 0) {
        handleCopyFileRange(cpu, view);
        syscallDone(); pc = svcPC + 4n;
        continue;
      }

      // rt_sigtimedwait and the ipc waits all return EINTR when a signal they are NOT waiting for
      // arrives, and libc retries the call. the signal has to be DELIVERED between the two, or the
      // retry finds it pending again and the pair becomes a spin -- sigwaitinfo01 called
      // rt_sigtimedwait sixty thousand times in twenty seconds without the handler ever running.
      // rt_sigtimedwait: accept one of a set of signals DIRECTLY, without running its handler.
      // that is what sigwait() is, and it has to block here rather than in the per-process handler,
      // which cannot park on the proc-table futex. it was a stub returning EINTR immediately, so
      // every sigwait loop became a spin -- ltp's fcntl31 span it thirty thousand times in half a
      // minute and never noticed the signal it was waiting for.
      if (nr === 137) {
        handleSigtimedwait(cpu, view);
        syscallDone(); pc = svcPC + 4n;
        if (drainSignalQueue()) {
          const next = deliverQueuedSignals(pc);
          if (next === null) return;
          pc = next;
        }
        continue;
      }
      // ptrace: a tracer names another process, so it has to run where the process table is.
      if (nr === 117) {
        handlePtrace(cpu, view);
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      // futex, futex_time64 (identical here: a 64-bit timespec is the only timespec we have) and
      // futex_waitv. handled for EVERY process, threaded or not: a single-threaded process still has
      // to see its own timeout expire, and may share a futex word with another process.
      if (nr === 98 || nr === 422 || nr === 449) {
        if (nr === 449) handleFutexWaitv(cpu); else handleFutex(cpu);
        syscallDone(); pc = svcPC + 4n;
        if (drainSignalQueue()) {
          const next = deliverQueuedSignals(pc);
          if (next === null) return;
          pc = next;
        }
        continue;
      }
      // setsid: become leader of a brand-new session and group (pgid = own pid), detaching from
      // the parent's group so the new session's group signals no longer reach it.
      if (nr === 157) {
        Atomics.store(procTable, (procSlotOffset(mySlot) / 4) + PROC_PGID, myPid);
        cpu.setReg(0, BigInt(myPid));
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      // fcntl advisory record locks (F_GETLK/F_SETLK/F_SETLKW + 64-bit variants) are cross-process,
      // so they round-trip to the kernel lock table rather than running in the per-process handler.
      if (nr === 25) {
        const lc = Number(cpu.getReg(1));
        // 5/6/7 (+ the 12/13/14 64-bit aliases) are the process-owned posix locks; 36/37/38 are
        // the open-file-description locks, which differ only in who owns the range.
        if (lc === 5 || lc === 6 || lc === 7 || lc === 12 || lc === 13 || lc === 14) {
          handleFcntlLock(cpu, view, lc, 'posix');
          syscallDone(); pc = svcPC + 4n;
          continue;
        }
        if (lc === 36 || lc === 37 || lc === 38) {
          handleFcntlLock(cpu, view, lc, 'ofd');
          syscallDone(); pc = svcPC + 4n;
          continue;
        }
      }
      // AF_UNIX sockets are implemented here, not over the swift network bridge: they never touch
      // the network, and their byte streams are arena rings just like a pipe's.
      // socketpair(2) only exists in the unix domain: there is no such thing as a connected pair
      // of internet sockets, and linux says EAFNOSUPPORT (not ENOSYS, which is what fell out of
      // the generic handler and told callers the syscall itself was missing).
      // socketpair(AF_UNIX, ...): the kernel owns the ring arena, so only this side can make one.
      // every other family falls through to the generic handler, which validates the triple and
      // then answers EOPNOTSUPP -- socketpair exists only in the unix domain.
      if (nr === 199 && Number(cpu.getReg(0)) === 1) {
        handleUnixSocketpair(cpu, view, Number(cpu.getReg(1)));
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      if (nr === 198 && Number(cpu.getReg(0)) === 1) {
        handleUnixSocket(cpu, Number(cpu.getReg(1)));
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      // AF_NETLINK. the kernel is the only thing that ever writes to one of these, and the only
      // thing it writes is an mq_notify cookie -- which is what libc's SIGEV_THREAD needs to exist.
      if (nr === 198 && Number(cpu.getReg(0)) === 16) {
        handleNetlinkSocket(cpu, Number(cpu.getReg(1)));
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      if (unixSocks.has(Number(cpu.getReg(0))) &&
          (nr === 200 || nr === 201 || nr === 202 || nr === 242 || nr === 203 ||
           nr === 63 || nr === 64 || nr === 57 || nr === 206 || nr === 207 ||
           nr === 211 || nr === 212 || nr === 204 || nr === 205)) {
        if (nr === 204 || nr === 205) handleUnixGetName(cpu, view, nr === 205);
        else if (nr === 200) handleUnixBind(cpu, view);
        else if (nr === 201) handleUnixListen(cpu);
        else if (nr === 202 || nr === 242) handleUnixAccept(cpu, view);
        else if (nr === 203) handleUnixConnect(cpu, view);
        // sendmsg/recvmsg carry the same bytes as write/read, and can carry descriptors with them.
        else if (nr === 211) handleUnixSendmsg(cpu, view);
        else if (nr === 212) handleUnixRecvmsg(cpu, view);
        // sendto/recvfrom on a connected unix socket are write/read with a flags word. the
        // address arguments are meaningless on a connected socket and linux ignores them.
        else if (nr === 206) handleUnixSendto(cpu, view);
        else if (nr === 207) handleUnixRecvfrom(cpu, view);
        else handleUnixIO(nr, cpu);
        syscallDone(); pc = svcPC + 4n;
        continue;
      }

      // flock(fd, op): whole-file advisory locks, owned by the open file description and living in
      // a namespace of their own -- an flock and a record lock on one file never see each other.
      if (nr === 32) {
        handleFlock(cpu);
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      // closing a descriptor drops this process's posix locks on the whole file, and the last
      // descriptor of an open description takes its ofd locks with it. told to the kernel BEFORE
      // the close itself, which is indistinguishable to everyone else (the lock is gone either
      // way) and keeps the ordering against another process's request. only a process that has
      // actually taken a lock pays for this, so the common close costs nothing.
      if (nr === 57 && usedLocks) {
        releaseLocksForClosingFd(Number(cpu.getReg(0)));
        // deliberately no `continue`: the real close still has to run
      }
      // nanosleep(101) / clock_nanosleep(115): actually suspend the process for the
      // requested duration (the shared handler is a no-op, which breaks sleep + busybox
      // timeout's watchdog). the worker blocks on its wait futex; a signal wakes it -> EINTR.
      if (nr === 101 || nr === 115) {
        const abs = nr === 115 && (Number(cpu.getReg(1)) & 1) !== 0; // TIMER_ABSTIME
        const reqAddr = Number(nr === 115 ? cpu.getReg(2) : cpu.getReg(0));
        const remAddr = Number(nr === 115 ? cpu.getReg(3) : cpu.getReg(1));
        handleNanosleep(cpu, view, reqAddr, remAddr, abs, nr === 115 ? Number(cpu.getReg(0)) : 1);
        syscallDone(); pc = svcPC + 4n;
        if (syscalls.hasPendingDeliverable()) {
          const next = deliverQueuedSignals(pc);
          if (next === null) return;
          pc = next;
        }
        continue;
      }
      if (nr === 59) {
        handlePipe2(cpu, view);
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      // dup3 ONTO a descriptor that currently holds a kernel pipe end (or a unix socket) has to
      // close that end first, exactly as close(2) would. these live in maps keyed by fd on THIS
      // side, and the generic dup3 in the syscall handler cannot see them -- so the old entry
      // survived and went on shadowing the descriptor. every later write(1, ...) was intercepted
      // and sent to the pipe, which is why a script's `> file` redirect silently did nothing
      // whenever the script's own stdout was a pipe: `./script | cat` wrote to the pipe and left
      // the file empty.
      if (nr === 24 && Number(cpu.getReg(0)) !== Number(cpu.getReg(1))) {
        const dupTo = Number(cpu.getReg(1));
        const shadow = kernelPipes.get(dupTo);
        if (shadow && !kernelPipes.has(Number(cpu.getReg(0)))) {
          pipeRef(shadow.pipeOff, shadow.mode, -1);
          kernelPipes.delete(dupTo);
        }
        if (unixSocks.has(dupTo) && !unixSocks.has(Number(cpu.getReg(0)))) {
          closeUnixSock(dupTo);
        }
      }
      // Intercept dup3 when source is a kernel pipe fd
      if (nr === 24 && kernelPipes.has(Number(cpu.getReg(0)))) {
        handleKernelPipeDup3(cpu);
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      // dup(2) and fcntl(F_DUPFD) copy a descriptor to a NEW number. for a kernel pipe end that
      // has to register the new number in this side's map and take a reference on the ring --
      // neither of which the generic handler can do, because the ring is not in its fd table.
      // without it a shell that saves its stdout with dup(1) before a redirect got a descriptor
      // that pointed at nothing, and restoring from it left every later write going nowhere.
      if ((nr === 23 || (nr === 25 && (Number(cpu.getReg(1)) === 0 || Number(cpu.getReg(1)) === 1030))) &&
          kernelPipes.has(Number(cpu.getReg(0)))) {
        handleKernelPipeDup(cpu, nr === 25 ? Number(cpu.getReg(2)) : 0, nr === 25 && Number(cpu.getReg(1)) === 1030);
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      // FIONREAD on a pipe: how many bytes are sitting in the ring. only this side can answer,
      // because the ring lives in shared memory and the generic handler knows nothing about it.
      if (nr === 29 && Number(cpu.getReg(1)) === 0x541B && kernelPipes.has(Number(cpu.getReg(0)))) {
        const info = kernelPipes.get(Number(cpu.getReg(0)))!;
        const ring = new PipeRing(shmArena, info.pipeOff);
        const outPtr = Number(cpu.getReg(2));
        if (badGuestPtr(new Uint8Array(memory.buffer), outPtr, 4)) {
          cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); // EFAULT
        } else {
          view.setInt32(outPtr + GUEST_MEM_BASE, ring.bytesAvailable, true);
          cpu.setReg(0, 0n);
        }
        syscallDone(); pc = svcPC + 4n;
        continue;
      }
      // Intercept read/write/writev/close on kernel pipe fds
      if ((nr === 63 || nr === 64 || nr === 65 || nr === 66 || nr === 57) && kernelPipes.has(Number(cpu.getReg(0)))) {
        handleKernelPipeIO(nr, cpu, view);
        // an interruptible pipe read may return EINTR with a queued signal; deliver it (running
        // its handler or terminating). honor SA_RESTART by resuming at the SVC so the read reruns.
        const eintr = BigInt.asIntN(64, cpu.getReg(0)) === -4n;
        syscallDone();
        const restart = eintr && syscalls.hasPendingDeliverable() && syscalls.anyPendingRestart();
        const retPC = restart ? svcPC : svcPC + 4n;
        if (restart) cpu.setReg(0, origX0);
        if (syscalls.hasPendingDeliverable()) {
          const next = deliverQueuedSignals(retPC);
          if (next === null) return;
          pc = next;
        } else {
          pc = retPC;
        }
        continue;
      }

      // a syscall that dereferences an out-of-range guest pointer throws a RangeError from the
      // underlying typed-array/DataView op. linux answers a bad user address with EFAULT, not by
      // killing the process, so convert it here instead of letting the worker crash.
      let result;
      try {
        result = syscalls.handle(cpu, memory);
      } catch (e) {
        if (e instanceof RangeError) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); result = { status: 'ok' as const }; }
        else throw e;
      }
      syscallDone();
      if (takeSwapAtSyscall(svcPC)) return;
      view = new DataView(memory.buffer);

      switch (result.status) {
        case 'ok':
          pc = svcPC + 4n;
          break;

        case 'exit':
          postExit(result.code, result.group === true);
          return;

        case 'fork':
          pc = handleFork(result);
          break;

        case 'execve':
          try {
            pc = handleExec(result.path, result.argv, result.envp);
            // once a child has exec'd it is no longer the program its parent forked, and the
            // parent loses the right to move it between process groups.
            Atomics.store(procTable, (procSlotOffset(mySlot) / 4) + PROC_EXECED, 1);
          } catch (e) {
            // exec of a missing/unreadable binary must fail the syscall (ENOENT), not crash
            // the worker: readGuestFile throws before any process state is touched, so the
            // caller is intact. return the errno so the shell prints "not found".
            const errno = (e as { errno?: number })?.errno;
            if (typeof errno === 'number') { cpu.setReg(0, BigInt.asUintN(64, BigInt(-errno))); pc = svcPC + 4n; }
            else throw e;
          }
          break;

        case 'blocked_read':
          // Block on stdin via Atomics.wait on the stdin ring
          handleBlockedRead();
          // Retry the read syscall
          pc = svcPC;
          break;

        case 'bridge':
          // thread this process's creds into every filesystem bridge call so NativeFS can enforce
          // emulated POSIX ownership/permissions against the caller (fd-only ops ignore it).
          if (result.method.startsWith('fs_')) {
            (result.args as Record<string, unknown>).creds = syscalls.getCreds();
          }
          // Bridge calls go through main thread
          handleBridgeCall(result.method, result.args, result.apply);
          // a call that could not finish yet (a blocking accept with nobody waiting) asks to be
          // run again. the waiting happens HERE rather than inside the bridge, so a signal can
          // end it and the bridge watchdog never sees a call held open.
          if (syscalls.takeRetry()) {
            if (pollAndQueueSignals()) {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(-4)));   // EINTR
              pc = svcPC + 4n;
            } else {
              waitIo(readIoGeneration(), 20);
              pc = svcPC;
            }
            break;
          }
          // a bridge call returns straight into the guest without passing through the syscall
          // handler's return, so the failure census has to be told here or filesystem errors
          // would be the one kind it never sees.
          syscalls.noteResult(nr, BigInt.asIntN(64, cpu.getReg(0)));
          pc = svcPC + 4n;
          break;

        default:
          pc = svcPC + 4n;
          break;
      }
      traceCall(nr, tA0, tA1, tA2, tPath);
      // a blocking syscall may have queued a signal (e.g. wait4/read interrupted by SIGALRM
      // or an incoming kill). deliver it now, before executing the resumed instruction.
      if (drainSignalQueue()) {
        const next = deliverQueuedSignals(pc);
        if (next === null) return;
        pc = next;
      }
      continue;
    }
  }

  postExit(0);
}

// ── Syscall helpers ─────────────────────────────────────────────────────────

// pack a guest memory region into a sparse snapshot: only the non-zero 64KB pages travel. a
// fresh child memory is all-zero, so the omitted pages are already correct. the ~33MB contiguous
// fork copy (which forced every page resident and OOMed fork storms) becomes "the few MB the
// process actually touched". the zero-scan is a Uint32 word compare (exact — no float -0/NaN
// footguns) with early-exit, run once per fork (not a hot path). also reused by memory swap.
// WHERE A FORK PUTS THE PAGES IT HANDS ITS CHILD.
//
// this used to be a fresh ArrayBuffer per fork, packed and transferred. that is one allocation of
// everything the process has dirtied, every time it forks -- and a guest that forks a few hundred
// times a second makes garbage faster than the collector takes it back. the web content process
// reaches its memory limit and is killed, which is how four ltp tests were taking the whole app
// down with them.
//
// so the pages go into a block of the shared arena instead, kept between forks and grown only when
// a snapshot outgrows it. steady-state cost: nothing at all.
/**
 * ask the kernel for a block to stage this fork's pages in. the block belongs to the CHILD from
 * the moment it is handed over, and the kernel gives it back when the child has copied out of it
 * or has died -- so the parent never waits on its own child, and two forks never share a block.
 *
 * -1 means the arena is full, and the caller falls back to packing into a buffer of its own. that
 * is the slow path this whole arrangement exists to avoid, but under real pressure it is better
 * than refusing a fork.
 */
function takeForkStage(bytes: number): number {
  Atomics.store(myMailbox, MBOX_READY, 0);
  (self as any).postMessage({ type: 'fork_stage', bytes });
  awaitMailbox('fork_stage');
  return Atomics.load(myMailbox, MBOX_VALUE0);
}

function buildSparseSnapshot(buffer: ArrayBufferLike, totalLen: number,
                            ranges?: Array<{ start: number; end: number }>,
                            stage = false): SparseMemSnapshot {
  const pageCount = Math.ceil(totalLen / MEM_PAGE_SIZE);
  const wordsPerPage = MEM_PAGE_SIZE >>> 2;
  const u32 = new Uint32Array(buffer, 0, (pageCount * MEM_PAGE_SIZE) >>> 2);
  const bytes = new Uint8Array(buffer);
  const idx: number[] = [];
  // scan only the pages the process actually has. reading every word of every ZERO page to find
  // the few non-zero ones is what made a fork cost 7-20 ms: an idle shell's address space is 34 MiB
  // of which about 2 is real. `ranges` are host-address ranges (guest_mem_base already added).
  //
  // the ranges OVERLAP -- a MAP_SHARED region is both a vma and a shared map, and two mappings can
  // cover the same page -- so a page must be recorded once however many ranges name it. counting it
  // per range made the packed copy a multiple of the address space rather than a subset of it, and
  // a process with enough overlap could ask for more memory than exists: a guest that forks then
  // threw "Out of memory" here and took its worker down with it.
  const seen = new Uint8Array(pageCount);
  const scan = ranges ?? [{ start: 0, end: pageCount * MEM_PAGE_SIZE }];
  for (const r of scan) {
    const first = Math.max(0, Math.floor(r.start / MEM_PAGE_SIZE));
    const last = Math.min(pageCount, Math.ceil(r.end / MEM_PAGE_SIZE));
    for (let p = first; p < last; p++) {
      if (seen[p]) continue;
      seen[p] = 1;
      const wStart = p * wordsPerPage;
      let nz = false;
      for (let w = 0; w < wordsPerPage; w++) { if (u32[wStart + w] !== 0) { nz = true; break; } }
      if (nz) idx.push(p);
    }
  }
  idx.sort((a, b) => a - b);
  const need = idx.length * MEM_PAGE_SIZE;
  const packed = (dst: Uint8Array) => {
    for (let i = 0; i < idx.length; i++) {
      const off = idx[i] * MEM_PAGE_SIZE;
      dst.set(bytes.subarray(off, off + MEM_PAGE_SIZE), i * MEM_PAGE_SIZE);
    }
  };
  if (stage && need > 0) {
    const off = takeForkStage(need);
    if (off >= 0) {
      packed(new Uint8Array(shmArena, off, need));
      return { pages: Uint32Array.from(idx), stageOff: off, byteLen: need,
               totalLen: pageCount * MEM_PAGE_SIZE };
    }
  }
  const data = new Uint8Array(need);
  packed(data);
  return { pages: Uint32Array.from(idx), data: data.buffer, totalLen: pageCount * MEM_PAGE_SIZE };
}

// scatter a sparse snapshot into a (zero) destination memory, growing it to fit first.
function applySparseSnapshot(mem: WebAssembly.Memory, snap: SparseMemSnapshot): void {
  if (snap.totalLen > mem.buffer.byteLength) {
    const need = Math.ceil(snap.totalLen / MEM_PAGE_SIZE);
    const cur = mem.buffer.byteLength / MEM_PAGE_SIZE;
    if (need > cur) mem.grow(need - cur);
  }
  const dst = new Uint8Array(mem.buffer);
  // the bodies are either a buffer that travelled with the message or a block of the shared arena.
  const data = snap.data ? new Uint8Array(snap.data)
                         : new Uint8Array(shmArena, snap.stageOff!, snap.byteLen!);
  for (let i = 0; i < snap.pages.length; i++) {
    dst.set(data.subarray(i * MEM_PAGE_SIZE, (i + 1) * MEM_PAGE_SIZE), snap.pages[i] * MEM_PAGE_SIZE);
  }
}

function handleFork(req: { resumePC: bigint; flags: number; childStack: bigint; ptid: bigint; tls: bigint; ctid: bigint }): bigint {
  const resumePC = req.resumePC;
  const tFork0 = performance.now();
  // Snapshot our memory for the child (sparse: only touched pages)
  const snapSize = Math.min(
    Number(syscalls.getMmapBase()) + GUEST_MEM_BASE + 0x10000,
    memory.buffer.byteLength,
  );
  // the host region below the guest (the page-protection table, the register block) is state the
  // child rebuilds for itself, but it is also where the cpu state block lives, so page 0 has to
  // travel: keep [0, GUEST_MEM_BASE) in the scan and add the guest's own mapped ranges on top.
  const guestRanges = [
    ...syscalls.snapshotRanges(GUEST_STACK_TOP),
    // MAP_SHARED regions are shadowed into guest memory and reconciled at every syscall boundary,
    // so a child that starts with zeros there does not merely miss data -- it PUBLISHES the zeros
    // over the arena and destroys what the parent wrote. they are in the vma list too, but this is
    // the one category where being wrong is silent and destructive, so it is stated outright.
    ...sharedMaps.map((m) => ({ start: m.addr, end: m.addr + m.len })),
  ].map((r) => ({
    start: r.start + GUEST_MEM_BASE, end: Math.min(r.end + GUEST_MEM_BASE, snapSize),
  }));
  // a snapshot the host cannot allocate is a fork the host cannot make. linux answers ENOMEM;
  // letting the allocation throw would take this whole process down for something the guest is
  // entitled to be told about and handle.
  let memSnapshot: SparseMemSnapshot;
  try {
    memSnapshot = buildSparseSnapshot(memory.buffer, snapSize,
      [{ start: 0, end: GUEST_MEM_BASE }, ...guestRanges], true);
  } catch (e) {
    // say WHY. a fork that answers ENOMEM and swallows the reason is a dead end to debug: the
    // number the guest sees is the same whether the snapshot was too big, the heap was full, or
    // the size was computed wrong, and only one of those is the guest's fault.
    (self as any).postMessage({ type: 'worker_error',
      detail: `pid ${myPid} fork snapshot failed (snapSize=${snapSize}, ranges=${guestRanges.length}): ${e}` });
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-12)));   // ENOMEM
    return resumePC;
  }
  if (tracing) (self as any).postMessage({ type: 'trace', line:
    `pid${myPid} FORKRANGES snapSize=0x${snapSize.toString(16)} ` +
    guestRanges.map((r) => `[0x${(r.start - GUEST_MEM_BASE).toString(16)},0x${(r.end - GUEST_MEM_BASE).toString(16)})`).join(' ') });
  const syscallState = syscalls.getState();

  // Take the host-side references the child's descriptors need NOW, synchronously, on the parent's
  // thread. the child's own init runs arbitrarily later than the parent resuming, and the parent is
  // free to close everything the instant fork(2) returns -- which is precisely what a server that
  // forks a worker and closes its own copy of the listening socket does. done in the child, the
  // reference arrives after the descriptor is already gone.
  //
  // a native fd is dup'd, once per distinct swift fd so that descriptors the parent had dup2'd
  // together keep sharing one file offset in the child, exactly as they do in the parent (darwin's
  // dup shares the offset, which is also what fork's shared open file description means).
  // a socket cannot be dup'd that way -- both processes must keep talking through the SAME
  // connection -- so it takes a reference on the one socket instead.
  const forkNativeDups = new Map<number, number>();
  for (const [guestFd, entry] of syscallState.fdTable) {
    if (entry.type !== 'native') continue;
    let newFd = forkNativeDups.get(entry.nativeFd);
    if (newFd === undefined) {
      newFd = -1;
      handleBridgeCall('fs_dup', { fd: entry.nativeFd }, (resp) => { newFd = resp.ret; });
      forkNativeDups.set(entry.nativeFd, newFd);
    }
    if (newFd < 0) continue;
    syscallState.fdTable.set(guestFd, { ...entry, nativeFd: newFd });
  }
  // the inherited refcounts are keyed by swift fd, so they have to be recounted against the child's
  // own fds rather than carried over against the parent's.
  const childNativeRefs = new Map<number, number>();
  for (const e of syscallState.fdTable.values()) {
    if (e.type === 'native') childNativeRefs.set(e.nativeFd, (childNativeRefs.get(e.nativeFd) ?? 0) + 1);
  }
  syscallState.nativeRefs = childNativeRefs;
  const forkSocketIds = syscalls.inheritedSocketIds();
  for (const sockId of forkSocketIds) handleBridgeCall('net_ref', { socketId: sockId }, () => {});
  // a unix socket is a pair of rings and, sometimes, a name. the child inherits both, and the
  // references are taken here for the same reason the pipes' are: the parent may close its own
  // copy the instant fork returns. without this the child had the DESCRIPTOR but nothing behind
  // it -- every read, write and recvmsg on an inherited unix socket failed.
  const unixSocksSnapshot = refUnixSocksForChild();

  // Store our PC so the child knows where to resume
  cpu.setPC(resumePC);

  // Serialize kernel pipe map for child inheritance
  const kernelPipesSnapshot = [...kernelPipes.entries()];

  // Bump the refcount of every inherited pipe end NOW, synchronously, on behalf of the child.
  // the child worker adds refs only in its async init, which runs arbitrarily later than the
  // parent resuming — so without this the parent can close both ends and read EOF before the
  // child ever registers, leaving the child's later write staring at zero readers (EPIPE) and
  // truncating the pipe. incrementing here (parent thread, before we return to guest code)
  // makes the child's ends accounted for the instant fork(2) returns.
  for (const [, info] of kernelPipesSnapshot) {
    const ring = new PipeRing(shmArena, info.pipeOff);
    if (info.mode === 'write') ring.addWriter();
    else ring.addReader();
  }

  // Request fork from kernel
  (self as any).postMessage({
    type: 'fork',
    memSnapshot,
    syscallState,
    kernelPipesSnapshot,
    unixSocksSnapshot,
    inotifyIds: syscalls.inotifyInstanceIds(),
    shmBlocks: sharedMaps.map((m) => m.arenaOff),
    // which segment is attached where travels with the mapping: shmdt names an ADDRESS, and a
    // child that inherited an attach has to be able to detach it.
    shmMaps: sharedMaps.map((m) => ({ addr: m.addr, len: m.len, off: m.arenaOff,
                                     sysv: sysvAttached.get(m.addr) })),
    protArmed: protReady,
    resumePC: resumePC.toString(),
    // clone(2)'s child-context arguments. these are not thread-only: vfork, posix_spawn and
    // system() are all fork-style clones WITH a stack, and the child has to start on it.
    cloneFlags: req.flags,
    childStack: req.childStack.toString(),
    childTls: req.tls.toString(),
    childCtid: req.ctid.toString(),
  });   // the pages are already in the shared arena; nothing to transfer

  // Block until kernel responds with child PID
  const tPost = performance.now();
  Atomics.store(myMailbox, MBOX_READY, 0);
  awaitMailbox('fork');
  const childPid = Atomics.load(myMailbox, MBOX_VALUE0);
  if (tracing) (self as any).postMessage({ type: 'trace', line:
    `pid${myPid} FORK snapshot=${(tPost - tFork0).toFixed(1)}ms reply=${(performance.now() - tPost).toFixed(1)}ms` });

  if (childPid < 0) {
    // fork failed (kernel proc table full -> EAGAIN). the child never spawned, so undo the
    // pipe-end refcounts we optimistically bumped for it; otherwise every failed fork in a
    // storm leaks a reader/writer ref and the ring never reaches EOF.
    for (const [, info] of kernelPipesSnapshot) {
      const ring = new PipeRing(shmArena, info.pipeOff);
      if (info.mode === 'write') ring.closeWriter();
      else ring.closeReader();
    }
    // same for the descriptor references taken above: a child that never spawned holds none of them,
    // and a leaked one keeps a file or a socket alive for the rest of the run.
    for (const newFd of forkNativeDups.values()) {
      if (newFd >= 0) handleBridgeCall('fs_close', { fd: newFd }, () => {});
    }
    for (const sockId of forkSocketIds) handleBridgeCall('net_close', { socketId: sockId }, () => {});
    releaseUnixSocksSnapshot(unixSocksSnapshot);
    cpu.setReg(0, BigInt.asUintN(64, BigInt(childPid))); // -EAGAIN
    return resumePC;
  }

  // CLONE_PARENT_SETTID: the PARENT stores the child's tid where it was asked to. glibc's
  // posix_spawn and pthread creation both read the pid back from there rather than from x0.
  if ((req.flags & CLONE_PARENT_SETTID) && req.ptid !== 0n) {
    const p = Number(req.ptid);
    if (!syscalls.ptrFaults(new Uint8Array(memory.buffer), p, 4)) {
      new DataView(memory.buffer).setInt32(p + GUEST_MEM_BASE, childPid, true);
    }
  }

  // an eventfd's counter is shared with the child, so the child holds it too from the moment fork
  // returns. the parent still holds it while this runs, so the counter cannot go away underneath.
  for (const efd of syscalls.inheritedEventfdIds()) eventfdCall({ op: 'ref', id: efd, pid: childPid });

  // CLONE_VFORK: the parent does not run again until the child execs or _exits. programs rely on
  // that -- posix_spawn and every shell's fork+exec do -- and a parent that carried on regardless
  // raced its child for the memory they briefly share.
  if (req.flags & CLONE_VFORK) {
    const cslot = slotOfPid(childPid);
    if (cslot >= 0) {
      const cbase = cslot;   // slotOfPid returns the WORD index of the slot
      // ...and stop waiting the moment the child is gone: a child KILLED before it could exec
      // never runs its own exit path, and its slot stays a zombie with the pid still in it.
      while (Atomics.load(procTable, cbase + PROC_VFORK) === 1 &&
             Atomics.load(procTable, cbase + PROC_PID) === childPid &&
             (Atomics.load(procTable, cbase + PROC_STATE) === PROC_RUNNING ||
              Atomics.load(procTable, cbase + PROC_STATE) === PROC_STOPPED)) {
        setSleeping(true);
        try { sleepingWait(procTable, cbase + PROC_VFORK, 1, 50); } finally { setSleeping(false); }
        if (pollAndQueueSignals()) break;   // a signal ends the wait, as it does on linux
      }
    }
  }

  // Parent: set X0 = childPid
  cpu.setReg(0, BigInt(childPid));
  return resumePC;
}

// ── threads (clone with CLONE_THREAD) ────────────────────────────────────────

const CLONE_PARENT_SETTID = 0x00100000;
const CLONE_VFORK = 0x00004000;

// every unix-domain (and netlink) socket this worker holds, with a reference taken on each ring on
// behalf of the worker about to receive the snapshot. taken HERE, synchronously, because the sender
// may close its own copy before the recipient exists. undo with releaseUnixSocksSnapshot.
function refUnixSocksForChild(): Array<readonly [number, any]> {
  const snap = [...unixSocks.entries()].map(([fd, s]) => [fd, { ...s }] as const);
  // the RING counts move here, on the child's behalf. the kernel's holder record does not: it
  // copies the parent's wholesale when the child is created (inheritPipeRefs), and telling it
  // about these would count them twice.
  for (const [, sock] of snap) {
    if (sock.rx !== undefined) new PipeRing(shmArena, sock.rx).addReader();
    if (sock.tx !== undefined) new PipeRing(shmArena, sock.tx).addWriter();
  }
  return snap;
}

function releaseUnixSocksSnapshot(snap: Array<readonly [number, any]>): void {
  for (const [, sock] of snap) {
    if (sock.rx !== undefined) new PipeRing(shmArena, sock.rx).closeReader();
    if (sock.tx !== undefined) new PipeRing(shmArena, sock.tx).closeWriter();
  }
}

// a clone(CLONE_THREAD) that's mid-promote: the run loop yielded after posting 'clone_thread' to the
// kernel; onmessage('promote') finishes it once the kernel hands back the (main-thread-created)
// shared memory. saved so the resume knows where to continue + how to set up the child.
let pendingClone: { svcPC: bigint; childStack: number; tls: number; parentTidPtr: number; childTidPtr: number; flags: number } | null = null;

// clone(CLONE_THREAD|CLONE_VM|...): spawn a new thread. WORKERS CANNOT ALLOCATE SHARED MEMORY on
// WKWebView/JSC (it OOMs), so the kernel (main thread) creates it. first thread: post 'clone_thread'
// and YIELD the run loop — onmessage('promote') copies our memory into the shared one, rebinds to
// threaded mode, and resumes. subsequent threads: the process memory is already shared and the kernel
// holds it, so a plain blocking round-trip spawns the thread. returns true if the run loop must yield.
function handleCloneThread(cpu: CpuState, svcPC: bigint): boolean {
  const flags = Number(cpu.getReg(0));
  const childStack = Number(cpu.getReg(1));
  const parentTidPtr = Number(cpu.getReg(2));
  const tls = Number(cpu.getReg(3));
  const childTidPtr = Number(cpu.getReg(4));

  if (!threaded) {
    // a threaded process's address space is smaller than a single-threaded one's (see types.ts), so a
    // process that already mapped past that ceiling cannot be promoted without losing memory. fail the
    // clone honestly (EAGAIN) rather than silently truncating its image.
    if (Number(syscalls.getMmapBase()) > threadLayout.regsBase) {
      cpu.setReg(0, BigInt.asUintN(64, -11n)); // EAGAIN
      return false;
    }
    pendingClone = { svcPC, childStack, tls, parentTidPtr, childTidPtr, flags };
    (self as any).postMessage({ type: 'clone_thread', curPages: memory.buffer.byteLength / MEM_PAGE_SIZE });
    return true; // yield the run loop; resumed by onmessage('promote')
  }

  // already threaded: the kernel owns this process's shared memory (keyed by tgid). blocking round-trip.
  Atomics.store(myMailbox, MBOX_READY, 0);
  const threadSocks = refUnixSocksForChild();
  (self as any).postMessage({
    type: 'spawn_thread', tgid: syscalls.tgid || myPid,
    startPC: (svcPC + 4n).toString(), childStack, tls, childTidPtr, flags,
    parentRegBase: stateBase, syscallState: syscalls.getState(), unixSocksSnapshot: threadSocks,
    shmMaps: sharedMaps.map((m) => ({ addr: m.addr, len: m.len, off: m.arenaOff,
                                      sysv: sysvAttached.get(m.addr) })),
  });
  awaitMailbox('clone_thread');
  const tid = Atomics.load(myMailbox, MBOX_VALUE0);
  if (tid < 0) {
    releaseUnixSocksSnapshot(threadSocks);
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-11))); return false; // EAGAIN
  }
  if ((flags & CLONE_PARENT_SETTID) && parentTidPtr) new DataView(memory.buffer).setInt32(parentTidPtr + GUEST_MEM_BASE, tid, true);
  cpu.setReg(0, BigInt(tid));
  return false;
}

// onmessage('promote'): the kernel created the shared memory + allocated the child tid/reg-block.
// copy our current memory into it, switch to threaded mode, finish the pending clone, tell the kernel
// to spawn the thread-Worker (now that the shared memory holds our state), and resume the run loop.
function completePromote(sharedMemory: WebAssembly.Memory, tid: number, regBaseAddr: number, staleHigh: number, layout?: ThreadLayout): void {
  if (layout) threadLayout = layout; // adopt the tier the kernel sized this process to
  const pc = pendingClone!;
  // copy only the GUEST span. our non-shared memory can extend past the threaded ceiling (the page
  // protection table lives far above it, and a single-threaded guest may map higher than a threaded
  // one is allowed to), and the threaded layout defines everything at/above regsBase itself
  // (register blocks + atomics shadow) — copying the whole buffer both overflowed the smaller shared
  // memory (RangeError -> the process died on pthread_create) and would have pasted the old image's
  // side-tables over the new layout.
  const copyLen = Math.min(memory.buffer.byteLength, threadLayout.regsBase);
  const shared = new Uint8Array(sharedMemory.buffer);
  shared.set(new Uint8Array(memory.buffer, 0, copyLen));
  // a REUSED spare still holds the previous tenant's bytes beyond our image: zero the stale guest span
  // [copyLen, staleHigh) plus the whole atomics shadow (stale reservation bits would corrupt LDXR/STXR).
  // beyond staleHigh the shared memory was never touched, so it's already zero; reg blocks are
  // overwritten before use. only the actually-used span is touched, so RSS stays bounded.
  if (staleHigh > copyLen) shared.fill(0, copyLen, staleHigh);
  if (staleHigh > 0) shared.fill(0, threadLayout.shadowBase, threadLayout.shadowBase + (threadLayout.regsBase >> 9));
  memory = sharedMemory;
  threaded = true;
  // the address space becomes SHARED at this moment. hand the list we have to the group's table --
  // which every thread from here on reads and writes instead of its own private copy -- and arm
  // the guard for the group NOW, while this is still the only thread and its view of the address
  // space is therefore the authoritative one. leaving either to whichever thread happened to mmap
  // first would have let a thread work from the snapshot it was born with.
  syscalls.attachVmaTable(memory, threadLayout.vmaBase);
  syscalls.publishVmaTable();
  // ...and the group's signal dispositions, seeded from what this thread has, since it is still
  // the only one. sigaction from any thread from here on writes them for all of them.
  syscalls.attachSigTable(memory, threadLayout.sigBase);
  syscalls.publishSigTable();
  if (protActive && protBase) {
    if (!protReady) { paintProtTable(); protReady = true; }
    new Int32Array(memory.buffer, PROT_PAINTED_FLAG, 1)[0] = 1;
    protActive.value = 1;
  }
  stateBase = 0;
  regBaseGlobal = new WebAssembly.Global({ value: 'i32', mutable: false }, 0);
  cpu = new CpuState(memory, stateBase);
  regionCache.clear();

  syscalls.setThreadIds(myPid, myPid); // the leader's gettid==getpid==tgid==pid; names our thread group
  syscalls.maxBytes = threadLayout.regsBase; // threaded guests cap brk/mmap below the register blocks

  // snapshot our register block (the clone-time state) INTO the child's reg block NOW, while we're
  // still paused. once we resume below we race the child-Worker's own read of our block, so the child
  // starts from a copy taken at the clone instant instead (parentRegBase == its own base -> no re-copy).
  new Uint8Array(memory.buffer).copyWithin(regBaseAddr, 0, STATE_SIZE);

  cpu.setReg(0, BigInt(tid)); // clone returns the child tid to the parent
  if ((pc.flags & CLONE_PARENT_SETTID) && pc.parentTidPtr) new DataView(memory.buffer).setInt32(pc.parentTidPtr + GUEST_MEM_BASE, tid, true);

  (self as any).postMessage({
    type: 'thread_go', tid, regBaseAddr,
    startPC: (pc.svcPC + 4n).toString(), childStack: pc.childStack, tls: pc.tls, childTidPtr: pc.childTidPtr,
    flags: pc.flags, parentRegBase: regBaseAddr, // block already holds the clone-time snapshot
    syscallState: syscalls.getState(), // thread inherits the group's fds/cwd/creds
    unixSocksSnapshot: refUnixSocksForChild(),
    shmMaps: sharedMaps.map((m) => ({ addr: m.addr, len: m.len, off: m.arenaOff,
                                      sysv: sysvAttached.get(m.addr) })),
  });
  const resume = pc.svcPC + 4n;
  pendingClone = null;
  runToCompletion(resume);
}

// onmessage('clone_fail'): the kernel couldn't create the shared memory or find a free slot. the
// promote never happened, so we're still single-threaded — resume the clone returning EAGAIN.
function completePromoteFail(): void {
  const pc = pendingClone!;
  cpu.setReg(0, BigInt.asUintN(64, BigInt(-11))); // EAGAIN
  const resume = pc.svcPC + 4n;
  pendingClone = null;
  runToCompletion(resume);
}

// futex(uaddr, op, val, timeout, ...): real cross-thread blocking on the SHARED guest memory via
// Atomics.wait/notify (only reachable when threaded, so memory.buffer is a SharedArrayBuffer).
// returns true if a signal interrupted a WAIT (caller delivers it). waits in bounded slices so an
// incoming kill/timer is still observed.
/** index into the arena's Int32Array for a guest address inside a shared mapping, or -1. */
function sharedFutexSlot(uaddr: number): number {
  for (const m of sharedMaps) {
    if (uaddr < m.addr || uaddr + 4 > m.addr + m.len) continue;
    // ...as long as the group still says it is shared. this list is a snapshot, and a sibling thread
    // that unmapped the region leaves a stale entry behind; the address can then be handed out again
    // as ordinary memory, and keying a futex on the arena for one thread and on the address for
    // another is a wakeup that reaches nobody.
    if (!syscalls.rangeIsShared(uaddr & ~0xfff, (uaddr & ~0xfff) + 4096)) continue;
    return (m.arenaOff + (uaddr - m.addr)) >> 2;
  }
  return -1;
}

// linux's futex operations, and the errnos this file answers with.
const FUTEX_WAIT = 0, FUTEX_WAKE = 1, FUTEX_REQUEUE = 3, FUTEX_CMP_REQUEUE = 4, FUTEX_WAKE_OP = 5,
      FUTEX_WAIT_BITSET = 9, FUTEX_WAKE_BITSET = 10;
const FUTEX_CLOCK_REALTIME = 256;
const FX_EINTR = 4, FX_EFAULT = 14, FX_EINVAL = 22, FX_EAGAIN = 11, FX_ENOMEM = 12,
      FX_ENOSYS = 38, FX_ETIMEDOUT = 110;
// the whole point of a futex is that two waiters on the same memory are on the same queue, whatever
// address each of them calls it. so the key is the memory: its offset in the shared arena when the
// page is shared, and (thread group, address) when it is not -- a private mapping cannot be reached
// from another process, so its address is only ambiguous across processes that cannot collide.
interface FutexRef { i32: Int32Array; idx: number; keyLo: number; keyHi: number; }
function futexRef(uaddr: number): FutexRef {
  const shared = sharedFutexSlot(uaddr);
  if (shared >= 0) return { i32: new Int32Array(shmArena), idx: shared, keyLo: shared, keyHi: 0 };
  return {
    i32: new Int32Array(memory.buffer), idx: (uaddr + GUEST_MEM_BASE) >> 2,
    keyLo: uaddr | 0, keyHi: (syscalls.tgid || myPid) | 0,
  };
}

/** ms from now that an absolute timeout on `clk` falls due, or Infinity for "no timeout". */
function futexDeadline(view: DataView, tsPtr: number, absolute: boolean, clk: number): number | 'fault' | 'invalid' {
  if (tsPtr === 0) return Infinity;
  if (syscalls.ptrFaults(new Uint8Array(memory.buffer), tsPtr, 16)) return 'fault';
  const sec = view.getBigInt64(tsPtr + GUEST_MEM_BASE, true);
  const nsec = view.getBigInt64(tsPtr + GUEST_MEM_BASE + 8, true);
  if (nsec < 0n || nsec >= 1_000_000_000n) return 'invalid';
  const want = sec * 1_000_000_000n + nsec;
  if (!absolute) {
    if (want < 0n) return 'invalid';
    return performance.now() + Number(want) / 1e6;
  }
  const left = want - syscalls.clockNs(clk);
  return performance.now() + Math.max(0, Number(left) / 1e6);
}

/**
 * park on an already-queued futex entry until it is woken, the deadline passes, or a signal
 * arrives. the entry is released here whichever of those happens, since the waiter is the only one
 * that knows it has stopped waiting.
 *
 * returns 0 for a wake, or the negative errno to report. `signalled` says a signal is pending
 * regardless of the outcome, because a wake that races a signal still has to run the handler.
 */
function futexPark(slot: number, deadline: number): { ret: number; signalled: boolean } {
  const stateIdx = futexStateIndex(slot);
  let ret = 0, signalled = false;
  for (;;) {
    if (Atomics.load(procTable, stateIdx) === FUTEX_WOKEN) break;
    const left = deadline - performance.now();
    if (left <= 0) { ret = -FX_ETIMEDOUT; break; }
    // in slices, so a kill that arrives during a long wait is noticed rather than slept through.
    sleepingWait(procTable, stateIdx, FUTEX_QUEUED, Math.min(left, 50));
    if (Atomics.load(procTable, stateIdx) === FUTEX_WOKEN) break;
    if (pollAndQueueSignals()) { ret = -FX_EINTR; signalled = true; break; }
  }
  futexLock(procTable);
  // a wake that landed while we were giving up still counts: the waker has spent its wakeup on us.
  if (Atomics.load(procTable, stateIdx) === FUTEX_WOKEN) ret = 0;
  const woken = Atomics.load(procTable, (FUTEX_TABLE_OFFSET / 4) + FUTEX_HDR_WORDS + slot * FUTEX_SLOT_WORDS + FUTEX_RESULT);
  futexDequeue(procTable, slot);
  futexUnlock(procTable);
  return { ret: ret === 0 ? Math.max(0, woken) : ret, signalled };
}

// futex(uaddr, op, val, timeout/val2, uaddr2, val3). returns true when a signal is pending and the
// caller should deliver it.
function handleFutex(cpu: CpuState): boolean {
  const bytes = new Uint8Array(memory.buffer);
  const view = new DataView(memory.buffer);
  const uaddr = Number(BigInt.asUintN(32, cpu.getReg(0)));
  const rawOp = Number(BigInt.asIntN(32, cpu.getReg(1)));
  const op = rawOp & 0x7f;
  const realtime = (rawOp & FUTEX_CLOCK_REALTIME) !== 0;
  const val = Number(BigInt.asIntN(32, cpu.getReg(2)));
  const arg3 = cpu.getReg(3);
  const uaddr2 = Number(BigInt.asUintN(32, cpu.getReg(4)));
  const val3 = Number(BigInt.asIntN(32, cpu.getReg(5)));
  const err = (e: number) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(e))); };

  if (uaddr & 3) { err(-FX_EINVAL); return false; }

  if (op === FUTEX_WAIT || op === FUTEX_WAIT_BITSET) {
    const bits = op === FUTEX_WAIT_BITSET ? (val3 >>> 0) : 0xffffffff;
    if (bits === 0) { err(-FX_EINVAL); return false; }
    if (syscalls.ptrFaults(bytes, uaddr, 4)) { err(-FX_EFAULT); return false; }
    const dl = futexDeadline(view, Number(BigInt.asUintN(32, arg3)),
                             op === FUTEX_WAIT_BITSET, realtime ? 0 : 1);
    if (dl === 'fault') { err(-FX_EFAULT); return false; }
    if (dl === 'invalid') { err(-FX_EINVAL); return false; }
    const t = futexRef(uaddr);
    futexLock(procTable);
    // the value is read under the queue lock, so a wake that changes it cannot slip between the
    // check and the enqueue -- the lost-wakeup race futex exists to close.
    if (Atomics.load(t.i32, t.idx) !== (val | 0)) { futexUnlock(procTable); err(-FX_EAGAIN); return false; }
    const slot = futexEnqueue(procTable, t.keyLo, t.keyHi, bits, myPid, -1);
    futexUnlock(procTable);
    if (slot < 0) { err(-FX_ENOMEM); return false; }
    const r = futexPark(slot, dl);
    err(r.ret > 0 ? 0 : r.ret);
    return r.signalled;
  }

  if (op === FUTEX_WAKE || op === FUTEX_WAKE_BITSET) {
    const bits = op === FUTEX_WAKE_BITSET ? (val3 >>> 0) : 0xffffffff;
    if (bits === 0) { err(-FX_EINVAL); return false; }
    if (syscalls.ptrFaults(bytes, uaddr, 4)) { err(-FX_EFAULT); return false; }
    const t = futexRef(uaddr);
    futexLock(procTable);
    const r = futexWake(procTable, t.keyLo, t.keyHi, bits, Math.max(0, val));
    futexUnlock(procTable);
    cpu.setReg(0, BigInt(r.woke));
    return false;
  }

  if (op === FUTEX_REQUEUE || op === FUTEX_CMP_REQUEUE) {
    if (uaddr2 & 3) { err(-FX_EINVAL); return false; }
    if (syscalls.ptrFaults(bytes, uaddr, 4) || syscalls.ptrFaults(bytes, uaddr2, 4)) { err(-FX_EFAULT); return false; }
    const nrRequeue = Number(BigInt.asIntN(32, arg3));
    const t = futexRef(uaddr), t2 = futexRef(uaddr2);
    futexLock(procTable);
    if (op === FUTEX_CMP_REQUEUE && Atomics.load(t.i32, t.idx) !== (val3 | 0)) {
      futexUnlock(procTable); err(-FX_EAGAIN); return false;
    }
    const r = futexWake(procTable, t.keyLo, t.keyHi, 0xffffffff, Math.max(0, val),
                        Math.max(0, nrRequeue), t2.keyLo, t2.keyHi);
    futexUnlock(procTable);
    cpu.setReg(0, BigInt(r.woke + r.moved));
    return false;
  }

  if (op === FUTEX_WAKE_OP) {
    if (uaddr2 & 3) { err(-FX_EINVAL); return false; }
    if (syscalls.ptrFaults(bytes, uaddr, 4) || syscalls.ptrFaults(bytes, uaddr2, 4)) { err(-FX_EFAULT); return false; }
    const enc = val3 >>> 0;
    const opcode = (enc >> 28) & 0x7;
    const shifted = ((enc >> 28) & 0x8) !== 0;
    const cmp = (enc >> 24) & 0xf;
    let oparg = (enc << 8) >> 20;      // sign-extended 12 bits, as the kernel reads it
    const cmparg = (enc << 20) >> 20;
    if (shifted) {
      if (oparg < 0 || oparg > 31) { err(-FX_EINVAL); return false; }
      oparg = 1 << oparg;
    }
    const t = futexRef(uaddr), t2 = futexRef(uaddr2);
    let old: number;
    switch (opcode) {
      case 0: old = Atomics.exchange(t2.i32, t2.idx, oparg); break;
      case 1: old = Atomics.add(t2.i32, t2.idx, oparg); break;
      case 2: old = Atomics.or(t2.i32, t2.idx, oparg); break;
      case 3: old = Atomics.and(t2.i32, t2.idx, ~oparg); break;
      case 4: old = Atomics.xor(t2.i32, t2.idx, oparg); break;
      default: err(-FX_ENOSYS); return false;
    }
    let second: boolean;
    switch (cmp) {
      case 0: second = old === cmparg; break;
      case 1: second = old !== cmparg; break;
      case 2: second = old < cmparg; break;
      case 3: second = old <= cmparg; break;
      case 4: second = old > cmparg; break;
      case 5: second = old >= cmparg; break;
      default: err(-FX_ENOSYS); return false;
    }
    const nrWake2 = Number(BigInt.asIntN(32, arg3));
    futexLock(procTable);
    let total = futexWake(procTable, t.keyLo, t.keyHi, 0xffffffff, Math.max(0, val)).woke;
    if (second) total += futexWake(procTable, t2.keyLo, t2.keyHi, 0xffffffff, Math.max(0, nrWake2)).woke;
    futexUnlock(procTable);
    cpu.setReg(0, BigInt(total));
    return false;
  }

  // FUTEX_FD and the priority-inheritance family. nothing in the system uses them, and answering
  // 0 would be a lie a caller cannot detect.
  err(-FX_ENOSYS);
  return false;
}

// futex_waitv(waiters, nr_futexes, flags, timeout, clockid): wait on several futexes at once and
// report WHICH one woke. the waiters share one queue entry to sleep on -- the first of the group --
// so a wake on any of them wakes the thread exactly once.
function handleFutexWaitv(cpu: CpuState): boolean {
  const bytes = new Uint8Array(memory.buffer);
  const view = new DataView(memory.buffer);
  const listPtr = Number(BigInt.asUintN(32, cpu.getReg(0)));
  const nr = Number(BigInt.asUintN(32, cpu.getReg(1)));
  const flags = Number(BigInt.asUintN(32, cpu.getReg(2)));
  const tsPtr = Number(BigInt.asUintN(32, cpu.getReg(3)));
  const clockid = Number(BigInt.asIntN(32, cpu.getReg(4)));
  const err = (e: number) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(e))); };
  const FUTEX2_SIZE_MASK = 0x3, FUTEX2_32 = 0x2, FUTEX2_KNOWN = 0x83; // size bits | private

  if (flags !== 0) { err(-FX_EINVAL); return false; }
  if (nr === 0 || nr > 128) { err(-FX_EINVAL); return false; }
  if (listPtr === 0) { err(-FX_EINVAL); return false; }
  if (clockid !== 0 && clockid !== 1) { err(-FX_EINVAL); return false; }
  if (syscalls.ptrFaults(bytes, listPtr, nr * 24)) { err(-FX_EFAULT); return false; }

  const want: Array<{ uaddr: number; val: number }> = [];
  for (let i = 0; i < nr; i++) {
    const rec = listPtr + GUEST_MEM_BASE + i * 24;
    const wval = view.getBigUint64(rec, true);
    const waddr = Number(BigInt.asUintN(32, view.getBigUint64(rec + 8, true)));
    const wflags = view.getUint32(rec + 16, true);
    const reserved = view.getUint32(rec + 20, true);
    if (reserved !== 0) { err(-FX_EINVAL); return false; }
    if (wflags & ~FUTEX2_KNOWN) { err(-FX_EINVAL); return false; }
    if ((wflags & FUTEX2_SIZE_MASK) !== FUTEX2_32) { err(-FX_EINVAL); return false; }
    if (waddr & 3) { err(-FX_EINVAL); return false; }
    if (syscalls.ptrFaults(bytes, waddr, 4)) { err(-FX_EFAULT); return false; }
    want.push({ uaddr: waddr, val: Number(BigInt.asIntN(32, BigInt.asUintN(32, wval))) });
  }

  const dl = futexDeadline(view, tsPtr, true, clockid);
  if (dl === 'fault') { err(-FX_EFAULT); return false; }
  if (dl === 'invalid') { err(-FX_EINVAL); return false; }

  futexLock(procTable);
  const slots: number[] = [];
  let leader = -1, bad = 0;
  for (let i = 0; i < want.length; i++) {
    const t = futexRef(want[i].uaddr);
    if (Atomics.load(t.i32, t.idx) !== (want[i].val | 0)) { bad = -FX_EAGAIN; break; }
    const slot = futexEnqueue(procTable, t.keyLo, t.keyHi, 0xffffffff, myPid, leader, true);
    if (slot < 0) { bad = -FX_ENOMEM; break; }
    if (leader < 0) leader = slot;
    slots.push(slot);
  }
  if (bad !== 0) {
    if (leader >= 0) futexDequeue(procTable, leader);
    futexUnlock(procTable);
    err(bad);
    return false;
  }
  futexUnlock(procTable);

  const r = futexPark(leader, dl);
  if (r.ret < 0) { err(r.ret); return r.signalled; }
  // the queue reports WHICH entry was woken; the caller wants its index in the list it passed.
  const woken = slots.indexOf(r.ret);
  cpu.setReg(0, BigInt(woken < 0 ? 0 : woken));
  return r.signalled;
}

function handleExec(path: string, argv: string[], envp: string[]): bigint {
  // read the NEW binary from the guest FS (real distros exec different binaries; busybox
  // multicall re-reads /bin/<applet> -> the busybox image). update elfBytes so any later
  // re-exec / interpreter reload sees the current image. resolveExec also follows `#!` lines
  // and refuses a non-ELF -- and it does so BEFORE any process state is touched, since the
  // memory wipe below is the point of no return and the caller must survive to see the error.
  // exec is a permission check before it is anything else: the file has to BE there, be reachable
  // through every directory on the way, and be executable BY THIS PROCESS. reading it as the kernel
  // does answers none of that -- a program with mode 0700 owned by root ran happily as nobody, and
  // a path through a file reported ENOENT where the walk should have said ENOTDIR.
  const execAbs = syscalls.absolutePath(path);
  if (execAbs !== null) {
    let execAcc = 0;
    handleBridgeCall('fs_faccessat',
      { path: execAbs, mode: 1, eaccess: true, creds: syscalls.getCreds() },
      (resp: any) => { execAcc = resp?.ret ?? 0; });
    if (execAcc < 0) throw { errno: -execAcc };
  }
  const target = resolveExec(path, argv, (p) => { try { return readGuestFile(p); } catch { return null; } });
  path = target.path;
  argv = target.argv;
  const bytes = target.bytes;
  elfBytes = bytes;

  // clear guest memory (best-effort; loadProgram overwrites the segments it needs)
  new Uint8Array(memory.buffer).fill(0, GUEST_MEM_BASE, GUEST_MEM_BASE + 0x4000000);
  resetPageProt(); // the old image's page protections do not apply to the new one

  const prog = loadProgram(memory, elfBytes, argv, envp, readGuestFile);
  // loadProgram may have grown (detached) memory; rebuild cpu + views afresh
  cpu = new CpuState(memory, stateBase);
  cpu.setSP(prog.sp);
  new DataView(memory.buffer).setBigUint64(stateBase + 0x118, BigInt(0x900000), true); // TLS

  releaseSharedMaps(0, Number.MAX_SAFE_INTEGER); // exec tears down the address space
  // exec closes the FD_CLOEXEC descriptors, and a kernel pipe end is one of them. these live here
  // rather than in the syscall handler's table, so nothing else can release their ring reference:
  // without this the pipe keeps a phantom reader or writer forever and never reaches EOF, so the
  // other end blocks on a pipe whose only user is gone.
  for (const fd of syscalls.cloexecFds()) {
    const info = kernelPipes.get(fd);
    if (info) {
      pipeRef(info.pipeOff, info.mode, -1);
      kernelPipes.delete(fd);
    }
    // a unix socket is the same kind of thing: its rings live here, and an entry left behind after
    // exec shadows whatever descriptor number the new program later opens.
    if (unixSocks.has(fd)) closeUnixSock(fd);
  }
  publishIdentity(argv);
  // tell the kernel which file this process is now running: a write to a running program is
  // ETXTBSY, and the kernel is where every open passes through.
  { const img = syscalls.absolutePath(path); if (img) (self as any).postMessage({ type: 'exec_image', path: img }); }
  syscalls.setComm((argv[0] ?? '').split('/').pop() ?? '');
  syscalls.resetForExec(prog.brkStart, (fd) => handleBridgeCall('fs_close', { fd }, () => {}));
  // AFTER resetForExec, which empties the vma list -- adopting first left every exec'd process
  // with no address space at all. that is silent for a static binary loaded low, because fork's
  // snapshot also carries [0, max(brk, stack_top)] unconditionally, but a dynamically linked
  // image sits well above the stack: ld-musl put alpine's busybox at 16 MB, the snapshot stopped
  // at 8 MB, and every fork that did NOT go on to exec handed its child a text segment of zeros.
  // that is `timeout`, `time`, `xargs` and every vfork+exec in a distro shell.
  syscalls.adoptImage(prog.segs); // the loader's segments are mappings too
  regionCache.clear(); // new binary image
  syscalls.installSignalTrampoline(memory); // exec may have wiped low memory; re-install

  // a vforked parent has been suspended since the fork; the exec is what releases it.
  releaseVforkParent();

  // a traced process stops for its tracer the moment it has exec'd, with SIGTRAP -- that is the
  // hook every debugger uses to take control of a program before its first instruction.
  if (Atomics.load(procTable, procSlotOffset(mySlot) / 4 + PROC_TRACER) !== 0) syscalls.raiseSignal(5);

  return prog.entry;
}

// suspend this process for the requested duration by parking on its wait futex. a signal
// (e.g. SIGTERM from busybox timeout's watchdog) wakes it early -> EINTR + remaining time.
function handleNanosleep(cpu: CpuState, view: DataView, reqAddr: number, remAddr: number, absolute: boolean, clockId: number): void {
  const bytes = new Uint8Array(memory.buffer);
  if (reqAddr === 0 || syscalls.ptrFaults(bytes, reqAddr, 16)) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return;             // EFAULT
  }
  // a clock that does not exist cannot be slept on, and neither can this thread's own cpu clock:
  // it does not advance while the thread is asleep, so the wait could never end.
  if (clockId < 0 || clockId > 11 || clockId === 3) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return;             // EINVAL
  }
  // the request is a SIGNED timespec, and every part of it has to be a real duration. read as
  // unsigned, a negative second became nine billion and a nanosecond field of 1e9 was accepted --
  // both of which linux refuses outright.
  const sec = Number(view.getBigInt64(reqAddr + GUEST_MEM_BASE, true));
  const nsec = Number(view.getBigInt64(reqAddr + GUEST_MEM_BASE + 8, true));
  if (sec < 0 || nsec < 0 || nsec >= 1_000_000_000) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return;             // EINVAL
  }
  if (remAddr !== 0 && !absolute && syscalls.ptrWriteFaults(bytes, remAddr, 16)) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return;             // EFAULT
  }
  const targetNs = sec * 1e9 + nsec;
  const ms = absolute ? (targetNs - Number(syscalls.clockNs(clockId))) / 1e6 : targetNs / 1e6;
  if (ms <= 0) { cpu.setReg(0, 0n); return; }

  const slotBase = procSlotOffset(mySlot) / 4;
  const deadline = performance.now() + ms;
  let interrupted = false;
  for (;;) {
    const rem = deadline - performance.now();
    if (rem <= 0) break;
    // the last millisecond is spun rather than waited on: a futex wait cannot promise better than
    // whole-millisecond wakeups, so a 1 ms sleep asked of it comes back at 2 ms. a program that
    // sleeps in the sub-millisecond range -- which is most of what a timer test measures, and what
    // any polling loop does -- gets the resolution it asked for this way, at the cost of at most
    // one millisecond of spinning.
    if (rem <= 1.2) {
      while (performance.now() < deadline) {
        if (pollAndQueueSignals()) { interrupted = true; break; }
      }
      break;
    }
    Atomics.store(procTable, slotBase + PROC_WAIT_FUTEX, 0);
    // a futex wait overshoots by about a quarter of what it was asked for, so asking for the whole
    // remaining time sleeps 25% too long. asking for less and looping converges on the deadline:
    // each pass overshoots a smaller number, and the last millisecond is spun.
    sleepingWait(procTable, slotBase + PROC_WAIT_FUTEX, 0, Math.max(1.2, Math.min(rem * 0.7, 1000)));
    if (pollAndQueueSignals()) { interrupted = true; break; }
  }

  if (interrupted) {
    if (remAddr && !absolute) {
      const remMs = Math.max(0, deadline - performance.now());
      view.setBigUint64(remAddr + GUEST_MEM_BASE, BigInt(Math.floor(remMs / 1000)), true);
      view.setBigUint64(remAddr + GUEST_MEM_BASE + 8, BigInt(Math.round((remMs % 1000) * 1e6)), true);
    }
    cpu.setReg(0, BigInt.asUintN(64, -4n)); // -EINTR
  } else {
    if (remAddr && !absolute) {
      view.setBigUint64(remAddr + GUEST_MEM_BASE, 0n, true);
      view.setBigUint64(remAddr + GUEST_MEM_BASE + 8, 0n, true);
    }
    cpu.setReg(0, 0n);
  }
}

// rt_sigsuspend(mask, sigsetsize): atomically install `mask` and block until a signal not in it is
// delivered, then restore the pre-suspend mask and return EINTR (the caller delivers the handler).
// parks on the proc wait futex — the same one handleKill notifies — so a cross-worker kill wakes it.
function handleSigsuspend(cpu: CpuState): void {
  const maskPtr = Number(cpu.getReg(0));
  // the mask is read before the wait begins, so a bad pointer is EFAULT rather than a process that
  // suspends forever waiting under a mask it never managed to read.
  if (maskPtr === 0 || syscalls.ptrFaults(new Uint8Array(memory.buffer), maskPtr, 8)) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-14)));
    return;
  }
  const tempMask = new DataView(memory.buffer).getBigUint64(maskPtr + GUEST_MEM_BASE, true);
  const savedMask = syscalls.getSignalMask();
  // the wait runs under the replacement mask, and so does the handler that ends it: putting the
  // old mask back here would re-block the very signal the caller unblocked to wait for, and the
  // handler would never run. rt_sigreturn restores it instead, out of the frame.
  syscalls.setSuspendMask(tempMask, savedMask);
  const slotBase = procSlotOffset(mySlot) / 4;
  for (;;) {
    pollAndQueueSignals(); // fold in timers + anything another process queued for us
    if (syscalls.hasDeliverableSignal(tempMask)) break;
    Atomics.store(procTable, slotBase + PROC_WAIT_FUTEX, 0);
    const rem = syscalls.timerRemainingMs();
    sleepingWait(procTable, slotBase + PROC_WAIT_FUTEX, 0, rem > 0 ? Math.min(1000, rem) : 1000);
  }
  cpu.setReg(0, BigInt.asUintN(64, -4n)); // EINTR
}

// ppoll(NULL, 0, tmo, sigmask): pause()/timed sigwait. optionally swap in `sigmask` for the wait, then
// park on the proc wait futex until a deliverable signal (EINTR) or the timeout expires (0).
function handlePpollWait(cpu: CpuState): void {
  const view = new DataView(memory.buffer);
  const tmoPtr = Number(cpu.getReg(2));
  const sigmaskPtr = Number(cpu.getReg(3));
  let deadline = -1;
  if (tmoPtr !== 0) {
    const sec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE, true));
    const nsec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE + 8, true));
    deadline = performance.now() + sec * 1000 + nsec / 1e6;
  }
  const savedMask = syscalls.getSignalMask();
  const mask = sigmaskPtr ? view.getBigUint64(sigmaskPtr + GUEST_MEM_BASE, true) : savedMask;
  if (sigmaskPtr) syscalls.setSignalMask(mask);
  const slotBase = procSlotOffset(mySlot) / 4;
  let ret = 0n;
  for (;;) {
    pollAndQueueSignals();
    if (syscalls.hasDeliverableSignal(mask)) { ret = BigInt.asUintN(64, -4n); break; } // EINTR
    if (deadline >= 0 && performance.now() >= deadline) { ret = 0n; break; }
    Atomics.store(procTable, slotBase + PROC_WAIT_FUTEX, 0);
    const timerRem = syscalls.timerRemainingMs();
    let wait = deadline >= 0 ? Math.max(0, deadline - performance.now()) : (timerRem > 0 ? timerRem : 1000);
    wait = Math.min(1000, wait || 1000);
    sleepingWait(procTable, slotBase + PROC_WAIT_FUTEX, 0, wait);
  }
  if (sigmaskPtr) syscalls.setSignalMask(savedMask);
  cpu.setReg(0, ret);
}

/**
 * waitid(idtype, id, infop, options, rusage): wait for a child and describe what happened to it in
 * a siginfo_t rather than a packed status word.
 *
 * the difference from wait4 that matters is that the caller says WHICH state changes it wants to
 * hear about, and can ask (WNOWAIT) to be told without consuming the notification.
 */
function handleWaitid(cpu: CpuState, view: DataView): void {
  const err = (e: number) => cpu.setReg(0, BigInt.asUintN(64, BigInt(-e)));
  const P_ALL = 0, P_PID = 1, P_PGID = 2, P_PIDFD = 3;
  const WNOHANG = 1, WSTOPPED = 2, WEXITED = 4, WCONTINUED = 8;
  const WNOWAIT = 0x01000000, WNOTHREAD = 0x20000000, WALL = 0x40000000, WCLONE = 0x80000000;
  const KNOWN = WNOHANG | WSTOPPED | WEXITED | WCONTINUED | WNOWAIT | WNOTHREAD | WALL | WCLONE;

  let idtype = Number(cpu.getReg(0));
  let id = Number(BigInt.asIntN(32, cpu.getReg(1)));
  const infop = Number(cpu.getReg(2));
  const options = Number(BigInt.asUintN(32, cpu.getReg(3)));

  // P_PIDFD names the process through a descriptor; from here on it is an ordinary wait for that
  // pid, which is safe precisely because the descriptor pinned which process it meant.
  let pidfdNonblock = false;
  if (idtype === P_PIDFD) {
    const pid = syscalls.pidfdTarget(id);
    if (pid < 0) { err(9); return; } // EBADF
    // a pidfd opened PIDFD_NONBLOCK makes this wait non-blocking, and "nothing to report" comes
    // back as EAGAIN rather than as the all-zero siginfo WNOHANG would give.
    pidfdNonblock = syscalls.isNonBlockingFd(id);
    idtype = P_PID; id = pid;
  }
  if (options & ~KNOWN) { err(22); return; }
  // the caller has to name at least one state change; asking about none is not a wait, it is a
  // mistake, and linux says so rather than blocking forever.
  if ((options & (WEXITED | WSTOPPED | WCONTINUED)) === 0) { err(22); return; }
  if (idtype === P_PID && id <= 0) { err(22); return; }
  if (idtype === P_PGID && id < 0) { err(22); return; }
  if (idtype !== P_ALL && idtype !== P_PID && idtype !== P_PGID) { err(22); return; }
  if (infop !== 0 && !guestAccessible(infop, 128, true)) { err(14); return; }

  const bytes = new Uint8Array(memory.buffer);
  const zeroInfo = () => { if (infop !== 0) bytes.fill(0, infop + GUEST_MEM_BASE, infop + GUEST_MEM_BASE + 128); };

  for (;;) {
    let zombie = -1;
    let stopped = -1;
    let cont = -1;
    let living = false;
    for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
      const base = procSlotOffset(slot) / 4;
      const childPid = Atomics.load(procTable, base + PROC_PID);
      if (childPid === 0) continue;
      if (Atomics.load(procTable, base + PROC_PPID) !== myPid &&
          Atomics.load(procTable, base + PROC_TRACER) !== myPid) continue;
      const baseTgid = Atomics.load(procTable, base + PROC_TGID);
      if (baseTgid !== 0 && baseTgid !== childPid) continue; // a thread is not a child
      if (idtype === P_PID && childPid !== id) continue;
      if (idtype === P_PGID) {
        const want = id === 0 ? Atomics.load(procTable, (procSlotOffset(mySlot) / 4) + PROC_PGID) : id;
        if (Atomics.load(procTable, base + PROC_PGID) !== want) continue;
      }
      const state = Atomics.load(procTable, base + PROC_STATE);
      if (state === PROC_ZOMBIE) { if (zombie < 0) zombie = slot; }
      else if (state === PROC_RUNNING || state === PROC_STOPPED) {
        living = true;
        if (stopped < 0 && ((options & WSTOPPED) !== 0 ||
                            Atomics.load(procTable, base + PROC_TRACER) === myPid) && state === PROC_STOPPED &&
            Atomics.load(procTable, base + PROC_STOP_SEEN) === 0) stopped = slot;
        if (cont < 0 && (options & WCONTINUED) &&
            Atomics.load(procTable, base + PROC_CONT_PENDING) === 1) cont = slot;
      }
    }

    // a stop or a continue is reported the same way an exit is, with the reason in si_code.
    if (zombie < 0 && (stopped >= 0 || cont >= 0)) {
      const isStop = stopped >= 0;
      const base = procSlotOffset(isStop ? stopped : cont) / 4;
      const childPid = Atomics.load(procTable, base + PROC_PID);
      const uid = Atomics.load(procCreds, procCredOffset(isStop ? stopped : cont) / 4);
      const CLD_STOPPED = 5, CLD_CONTINUED = 6;
      const status = isStop ? Atomics.load(procTable, base + PROC_STOP_SIG) : 18 /* SIGCONT */;
      if (infop !== 0) {
        zeroInfo();
        const b = infop + GUEST_MEM_BASE;
        view.setInt32(b, 17, true);                                    // si_signo = SIGCHLD
        view.setInt32(b + 8, isStop ? CLD_STOPPED : CLD_CONTINUED, true);
        view.setInt32(b + 16, childPid, true);
        view.setUint32(b + 20, uid, true);
        view.setInt32(b + 24, status, true);
      }
      // WNOWAIT looks without collecting, so the next wait sees the same change again.
      if (!(options & WNOWAIT)) {
        if (isStop) Atomics.store(procTable, base + PROC_STOP_SEEN, 1);
        else Atomics.store(procTable, base + PROC_CONT_PENDING, 0);
      }
      cpu.setReg(0, 0n);
      return;
    }

    // a zombie is a child that has EXITED, so it is only news to a caller that asked for exits.
    if (zombie >= 0 && (options & WEXITED)) {
      const base = procSlotOffset(zombie) / 4;
      const childPid = Atomics.load(procTable, base + PROC_PID);
      const exitCode = Atomics.load(procTable, base + PROC_EXIT_CODE);
      const uid = Atomics.load(procCreds, procCredOffset(zombie) / 4);
      const killed = (exitCode & 0x10000) !== 0;
      const CLD_EXITED = 1, CLD_KILLED = 2, CLD_DUMPED = 3;
      const code = killed ? ((exitCode & 0x20000) ? CLD_DUMPED : CLD_KILLED) : CLD_EXITED;
      const status = killed ? (exitCode & 0x7f) : (exitCode & 0xff);
      const cUtime = Atomics.load(procTable, base + PROC_UTIME_MS);
      const cStime = Atomics.load(procTable, base + PROC_STIME_MS);

      if (infop !== 0) {
        // siginfo_t: si_signo, si_errno, si_code, then _sifields at 16 -- for SIGCHLD that is
        // pid, uid, status, utime, stime.
        zeroInfo();
        const b = infop + GUEST_MEM_BASE;
        view.setInt32(b, 17, true);          // si_signo = SIGCHLD
        view.setInt32(b + 8, code, true);    // si_code
        view.setInt32(b + 16, childPid, true);
        view.setUint32(b + 20, uid, true);
        view.setInt32(b + 24, status, true);
        view.setBigInt64(b + 32, BigInt(Math.floor(cUtime / 10)), true);
        view.setBigInt64(b + 40, BigInt(Math.floor(cStime / 10)), true);
      }
      const rusageAddr = Number(cpu.getReg(4));
      // WNOWAIT reports the change and LEAVES it pending, so the next wait sees it again. the
      // child stays a zombie, which is the point: one caller can look without consuming.
      if (!(options & WNOWAIT)) {
        syscalls.noteChildCpu(cUtime, cStime);
        if (rusageAddr !== 0) syscalls.writeRusage(view, rusageAddr, cUtime, cStime, 0);
        Atomics.store(procTable, base + PROC_STATE, PROC_EMPTY);
        Atomics.store(procTable, base + PROC_PID, 0);
      }
      cpu.setReg(0, 0n);
      return;
    }

    if (!living && zombie < 0) { err(10); return; }  // ECHILD
    // a live child that has not done the thing being waited for: WNOHANG answers "nothing yet",
    // and an all-zero siginfo is how the caller tells that from a real report.
    if (options & WNOHANG) { zeroInfo(); cpu.setReg(0, 0n); return; }
    if (pidfdNonblock) { err(11); return; }   // EAGAIN

    const mySlotBase = procSlotOffset(mySlot) / 4;
    const rem = syscalls.timerRemainingMs();
    Atomics.store(procTable, mySlotBase + PROC_WAIT_FUTEX, 0);
    sleepingWait(procTable, mySlotBase + PROC_WAIT_FUTEX, 0, rem > 0 ? Math.min(1000, rem) : 1000);
    if (pollAndQueueSignals()) { err(4); return; }   // EINTR
  }
}

function handleWait4(cpu: CpuState, view: DataView): void {
  const targetPid = Number(BigInt.asIntN(32, cpu.getReg(0))); // pid argument (signed: -1 = any)
  const wstatusAddr = Number(cpu.getReg(1));
  const options = Number(BigInt.asUintN(32, cpu.getReg(2)));
  const rusageAddr = Number(cpu.getReg(3));
  const WNOHANG = 1, WUNTRACED = 2, WCONTINUED = 8;
  const WNOTHREAD = 0x20000000, WALL = 0x40000000, WCLONE = 0x80000000;
  if (options & ~(WNOHANG | WUNTRACED | WCONTINUED | WNOTHREAD | WALL | WCLONE)) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; // EINVAL
  }

  // a negative pid names a process GROUP. linux resolves -pid as an id NUMBER first and answers
  // ESRCH when nothing was ever allocated under it -- which is not the same question as whether
  // any process is currently IN that group, and answering the second one made waiting for a live
  // group with no children of yours ESRCH instead of ECHILD. -INT_MIN is still negative, so the
  // classic INT_MIN argument resolves to nothing at all.
  let wantPgid = -1;
  if (targetPid === 0) {
    wantPgid = Atomics.load(procTable, (procSlotOffset(mySlot) / 4) + PROC_PGID);
  } else if (targetPid < -1) {
    wantPgid = -targetPid;
    let idExists = wantPgid > 0;
    if (idExists) {
      idExists = false;
      for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
        const base = procSlotOffset(slot) / 4;
        if (Atomics.load(procTable, base + PROC_PID) === wantPgid) { idExists = true; break; }
      }
    }
    if (!idExists) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-3))); return; } // ESRCH
  }

  while (true) {
    // Scan process table for matching zombie children
    let foundZombie = -1;
    let foundStopped = -1;
    let foundCont = -1;
    let hasLivingChild = false;

    for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
      const slotBase = procSlotOffset(slot) / 4;
      const childPid = Atomics.load(procTable, slotBase + PROC_PID);
      if (childPid === 0) continue; // empty slot
      // a tracer waits for its TRACEES as well as its children: that is how a debugger hears about
      // a process it attached to but did not start.
      const ppid = Atomics.load(procTable, slotBase + PROC_PPID);
      if (ppid !== myPid && Atomics.load(procTable, slotBase + PROC_TRACER) !== myPid) continue;
      // a THREAD of this process is not a child of it. we record a thread's creator as its
      // parent so the table stays a tree, but wait(2) only ever reports thread group LEADERS --
      // counting threads here made a threaded process's own tst_reap_children() block on threads
      // that were waiting, in turn, for that process to reach its cleanup.
      const slotTgid = Atomics.load(procTable, slotBase + PROC_TGID);
      if (slotTgid !== 0 && slotTgid !== childPid) continue;

      // Check if this child matches the target
      if (targetPid > 0 && childPid !== targetPid) continue;
      // targetPid == -1 is any child; anything else negative, or zero, names a process group.
      if (wantPgid >= 0 && Atomics.load(procTable, slotBase + PROC_PGID) !== wantPgid) continue;

      const state = Atomics.load(procTable, slotBase + PROC_STATE);
      if (state === PROC_ZOMBIE) {
        foundZombie = slot;
        break;
      }
      if (state === PROC_RUNNING || state === PROC_STOPPED) {
        hasLivingChild = true;
        // a stop and a continue are state CHANGES a waiting parent asked to hear about, and each
        // is reported once. WUNTRACED/WCONTINUED are how a shell learns its job hit ^Z or came back.
        // a stop is reported when the caller asked for stops (WUNTRACED) -- and always to a
        // TRACER, which is the whole mechanism a debugger waits on.
        if (foundStopped < 0 && state === PROC_STOPPED &&
            ((options & WUNTRACED) !== 0 ||
             Atomics.load(procTable, slotBase + PROC_TRACER) === myPid) &&
            Atomics.load(procTable, slotBase + PROC_STOP_SEEN) === 0) foundStopped = slot;
        if (foundCont < 0 && (options & WCONTINUED) &&
            Atomics.load(procTable, slotBase + PROC_CONT_PENDING) === 1) foundCont = slot;
      }
    }

    // a dead child outranks a stopped one, but either beats waiting.
    if (foundZombie < 0 && (foundStopped >= 0 || foundCont >= 0)) {
      const stopped = foundStopped >= 0;
      const slotBase = procSlotOffset(stopped ? foundStopped : foundCont) / 4;
      const childPid = Atomics.load(procTable, slotBase + PROC_PID);
      if (stopped) Atomics.store(procTable, slotBase + PROC_STOP_SEEN, 1);
      else Atomics.store(procTable, slotBase + PROC_CONT_PENDING, 0);
      if (wstatusAddr !== 0) {
        // WIFSTOPPED is (status & 0xff) == 0x7f with the signal above it; WIFCONTINUED is 0xffff.
        const stopSig = Atomics.load(procTable, slotBase + PROC_STOP_SIG);
        view.setUint32(wstatusAddr + GUEST_MEM_BASE, stopped ? ((stopSig << 8) | 0x7f) : 0xffff, true);
      }
      cpu.setReg(0, BigInt(childPid));
      return;
    }

    if (foundZombie >= 0) {
      // Reap the zombie
      const slotBase = procSlotOffset(foundZombie) / 4;
      const childPid = Atomics.load(procTable, slotBase + PROC_PID);
      const exitCode = Atomics.load(procTable, slotBase + PROC_EXIT_CODE);

      // Write wstatus if requested
      if (wstatusAddr !== 0) {
        // a death BY SIGNAL is encoded with the signal in the low 7 bits and no WIFEXITED byte;
        // masking to (code & 0xff) << 8 unconditionally reported every killed child as having
        // exited normally, so WIFSIGNALED was false for every process the kernel ever killed.
        // (0x10000 is the marker the signal-delivery path sets -- syscall.ts decodes it the same way.)
        // 0x80 on top of the signal number is WCOREDUMP: the signal was one whose default action
        // dumps core. we write no core file, but a parent tells a crash from an orderly death by
        // signal with exactly this bit.
        const wstatus = (exitCode & 0x10000)
          ? ((exitCode & 0x7f) | ((exitCode & 0x20000) ? 0x80 : 0))
          : ((exitCode & 0xFF) << 8);
        view.setUint32(wstatusAddr + GUEST_MEM_BASE, wstatus, true);
      }

      // the child's cpu accounting is the last thing it wrote before announcing its exit. take it
      // now, while the slot still holds it: wait4's fourth argument is a struct rusage FOR THE
      // CHILD, and it is also what getrusage(RUSAGE_CHILDREN) sums.
      const cUtime = Atomics.load(procTable, slotBase + PROC_UTIME_MS);
      const cStime = Atomics.load(procTable, slotBase + PROC_STIME_MS);
      const cMaxRss = Atomics.load(procTable, slotBase + PROC_MAXRSS_KB);
      syscalls.noteChildCpu(cUtime, cStime, cMaxRss);
      if (rusageAddr !== 0) syscalls.writeRusage(view, rusageAddr, cUtime, cStime, cMaxRss);

      // Clear the process table slot (reap)
      Atomics.store(procTable, slotBase + PROC_STATE, PROC_EMPTY);
      Atomics.store(procTable, slotBase + PROC_PID, 0);

      // Return child PID
      cpu.setReg(0, BigInt(childPid));
      return;
    }

    if (!hasLivingChild) {
      // No children at all — ECHILD
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-10))); // -ECHILD
      return;
    }

    if (options & WNOHANG) {
      // No zombie yet but don't block
      cpu.setReg(0, 0n);
      return;
    }

    // Block until a child exits: wait on our own slot's wait_futex
    // wake at the interval-timer deadline if one is armed (busybox timeout waits here),
    // else re-poll at 1s. this makes SIGALRM-based timeouts punctual.
    const rem = syscalls.timerRemainingMs();
    const waitMs = rem > 0 ? Math.min(1000, rem) : 1000;
    const mySlotBase = procSlotOffset(mySlot) / 4;
    Atomics.store(procTable, mySlotBase + PROC_WAIT_FUTEX, 0);
    sleepingWait(procTable, mySlotBase + PROC_WAIT_FUTEX, 0, waitMs);
    // Loop back and re-check for zombies

    // a fired timer or an incoming signal interrupts the wait with EINTR; the run loop
    // delivers the queued signal (running the handler or terminating) on return.
    if (pollAndQueueSignals()) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-4))); // -EINTR
      return;
    }
  }
}

// kill(pid, sig) / tgkill(tgid, tid, sig): cross-worker signal delivery. sets the target's
// signal queue and wakes it if it is blocked on its wait
// futex. handles pid>0 (specific), 0 (own pgroup), -1 (all but init), <-1 (specific pgroup).
/**
 * does the target in `slot` have room for one more queued signal?
 *
 * the budget is the target's RLIMIT_SIGPENDING, spent by every signal queued to any process of the
 * same real user and not yet taken -- which is why this counts across the table rather than asking
 * the one slot.
 */
function sigpendingAllows(slot: number): boolean {
  const limit = Atomics.load(procTable, procSlotOffset(slot) / 4 + PROC_RLIM_SIGPENDING);
  if (limit <= 0) return false;
  const ruid = Atomics.load(procCreds, procCredOffset(slot) / 4);
  let pending = 0;
  for (let s = 0; s < PROC_TABLE_SLOTS; s++) {
    if (Atomics.load(procTable, procSlotOffset(s) / 4 + PROC_PID) === 0) continue;
    if (Atomics.load(procCreds, procCredOffset(s) / 4) !== ruid) continue;
    pending += sigqDepth(sigQueue, s);
    if (pending >= limit) return false;
  }
  return true;
}

/**
 * clock_gettime and clock_getres on a task's cpu clock.
 *
 * linux packs the task into the clockid itself: the top bits are ~pid, bit 2 says the id names a
 * THREAD rather than a thread group, and the low two say which of that task's clocks it is. this is
 * what pthread_getcpuclockid and clock_getcpuclockid hand back, and the only way to ask how much
 * cpu someone else has had.
 */
function handleCpuClock(nr: number, cpu: CpuState, view: DataView): void {
  const err = (e: number) => cpu.setReg(0, BigInt.asUintN(64, BigInt(-e)));
  const clk = Number(BigInt.asIntN(32, cpu.getReg(0)));
  const ptr = Number(cpu.getReg(1));
  const id = ~(clk >> 3);
  const perThread = (clk & 4) !== 0;
  const which = clk & 3;
  if (which === 3 || id <= 0) { err(22); return; }
  if (ptr !== 0 && syscalls.ptrFaults(new Uint8Array(memory.buffer), ptr, 16)) { err(14); return; }

  if (nr === 114) { // clock_getres: cpu time is accounted as the scheduler ticks
    if (ptr !== 0) {
      view.setBigInt64(ptr + GUEST_MEM_BASE, 0n, true);
      view.setBigInt64(ptr + GUEST_MEM_BASE + 8, 1000000n, true);
    }
    cpu.setReg(0, 0n);
    return;
  }

  // our own entry is only as fresh as the last syscall boundary, and this IS one.
  publishCpuTime();
  const readMs = (slot: number): number => {
    const b = procSlotOffset(slot) / 4;
    const u = Atomics.load(procTable, b + PROC_UTIME_MS);
    const sy = Atomics.load(procTable, b + PROC_STIME_MS);
    return which === 1 ? u : u + sy;   // VIRT is user time alone; PROF and SCHED count both
  };

  let ms = 0;
  if (perThread) {
    const slot = schedSlotOf(id);
    if (slot < 0) { err(22); return; }
    ms = readMs(slot);
  } else {
    let found = false;
    for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
      const b = procSlotOffset(slot) / 4;
      const pid = Atomics.load(procTable, b + PROC_PID);
      if (pid === 0 || Atomics.load(procTable, b + PROC_STATE) === PROC_EMPTY) continue;
      if ((Atomics.load(procTable, b + PROC_TGID) || pid) !== id) continue;
      found = true;
      ms += readMs(slot);
    }
    if (!found) { err(22); return; }
  }
  if (ptr !== 0) {
    view.setBigInt64(ptr + GUEST_MEM_BASE, BigInt(Math.floor(ms / 1000)), true);
    view.setBigInt64(ptr + GUEST_MEM_BASE + 8, BigInt((ms % 1000) * 1000000), true);
  }
  cpu.setReg(0, 0n);
}

/** has this process ended? a pidfd reports readable from that moment, and so does poll on one. */
function pidExited(pid: number): boolean {
  const slot = schedSlotOf(pid);
  if (slot < 0) return true;   // the slot is gone, so the process is too
  return Atomics.load(procTable, (procSlotOffset(slot) / 4) + PROC_STATE) === PROC_ZOMBIE;
}

/**
 * pidfd_open and pidfd_send_signal.
 *
 * a pidfd names a PROCESS rather than a pid, so it cannot be confused by pid reuse: the descriptor
 * stays valid after the process dies, reports readable from that moment, and refuses to deliver a
 * signal to whoever inherits the number next.
 */
function handlePidfd(nr: number, cpu: CpuState, view: DataView): void {
  const err = (e: number) => cpu.setReg(0, BigInt.asUintN(64, BigInt(-e)));
  const PIDFD_NONBLOCK = 0o4000; // O_NONBLOCK: a pidfd is a file, and this is the file's flag

  if (nr === 434) { // pidfd_open(pid, flags)
    const pid = Number(BigInt.asIntN(32, cpu.getReg(0)));
    const flags = Number(BigInt.asUintN(32, cpu.getReg(1)));
    if (pid <= 0) { err(22); return; }
    if (flags & ~PIDFD_NONBLOCK) { err(22); return; }
    const slot = schedSlotOf(pid);
    if (slot < 0) { err(3); return; } // ESRCH
    // only a thread-group leader is a process. a thread has a tid, not a pid, and no pidfd.
    const base = procSlotOffset(slot) / 4;
    if ((Atomics.load(procTable, base + PROC_TGID) || pid) !== pid) { err(22); return; }
    const fd = syscalls.allocLowestFd();
    if (fd < 0) { err(24); return; } // EMFILE
    syscalls.registerFd(fd, { type: 'pidfd', pid }, flags & PIDFD_NONBLOCK);
    // a pidfd is always close-on-exec: it is a handle one program holds, and passing it through an
    // exec to a program that never asked for it is not something the caller can opt into.
    syscalls.markCloexec(fd);
    cpu.setReg(0, BigInt(fd));
    return;
  }

  // pidfd_send_signal(pidfd, sig, info, flags)
  const target = syscalls.pidfdTarget(Number(BigInt.asIntN(32, cpu.getReg(0))));
  const sig = Number(BigInt.asIntN(32, cpu.getReg(1)));
  const infoPtr = Number(cpu.getReg(2));
  const flags = Number(BigInt.asUintN(32, cpu.getReg(3)));
  if (flags !== 0) { err(22); return; }
  if (target < 0) { err(9); return; }  // EBADF: not a pidfd, and not a /proc/<pid> directory
  if (sig < 0 || sig > 64) { err(22); return; }
  // the descriptor outlives the process; the signal must not. a dead target is ESRCH rather than a
  // signal to whoever holds the pid now, which is the whole reason pidfds exist.
  if (pidExited(target)) { err(3); return; }
  if (infoPtr !== 0) {
    if (syscalls.ptrFaults(new Uint8Array(memory.buffer), infoPtr, 128)) { err(14); return; }
    // the siginfo has to agree with the signal being sent: they are one request, not two.
    if (view.getInt32(infoPtr + GUEST_MEM_BASE, true) !== sig) { err(22); return; }
  }
  // with a siginfo this is rt_sigqueueinfo, and without one it is kill; both are already here, and
  // both enforce the permission rules a signal has to pass.
  handleKill(infoPtr !== 0 ? 138 : 129, cpu, view, { targetPid: target, sig, infoPtr });
}

function handleKill(nr: number, cpu: CpuState, view: DataView,
                    from?: { targetPid: number; sig: number; infoPtr: number }): void {
  // `from` is pidfd_send_signal, which is kill(2) or rt_sigqueueinfo(2) with the target named by a
  // descriptor instead of by a pid. its arguments are passed rather than read out of the registers,
  // because those still hold the caller's own and a syscall must not rewrite them.
  const targetPid = from ? from.targetPid : Number(BigInt.asIntN(32, cpu.getReg(0)));
  // kill(pid, sig); tkill(tid, sig); tgkill(tgid, tid, sig);
  // rt_sigqueueinfo(tgid, sig, info); rt_tgsigqueueinfo(tgid, tid, sig, info).
  const sig = from ? from.sig
            : (nr === 131 || nr === 240) ? Number(BigInt.asIntN(32, cpu.getReg(2)))
                                         : Number(BigInt.asIntN(32, cpu.getReg(1)));
  // an out-of-range signal is EINVAL, and refusing it is not pedantry: the pending-signal set is a
  // 32-bit mask, and javascript's shift takes its count modulo 32, so kill(pid, 80) would have set
  // the bit for signal 16 on the target. a caller could raise a signal it never named.
  if (sig < 0 || sig > 64) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; } // EINVAL
  // tkill and tgkill name a THREAD, and a thread id is a positive number. kill(2) is the only one
  // of these where a non-positive argument means something (a process group, or every process),
  // so for the others it is a malformed request rather than a search that finds nothing.
  const namedTid = from ? 0 : (nr === 131 || nr === 240) ? Number(BigInt.asIntN(32, cpu.getReg(1))) : 0;
  if (nr === 130 && targetPid <= 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
  // rt_sigqueueinfo names ONE thread group. kill(2)'s reading of a non-positive pid -- a process
  // group, or every process -- does not apply, so there is simply no such group: ESRCH.
  if (nr === 138 && targetPid <= 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-3))); return; }
  if ((nr === 131 || nr === 240) && (targetPid <= 0 || namedTid <= 0)) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return;
  }
  const myPgid = Atomics.load(procTable, (procSlotOffset(mySlot) / 4) + PROC_PGID);
  const myTgid = Atomics.load(procTable, (procSlotOffset(mySlot) / 4) + PROC_TGID) || myPid;
  const myCredBase = procCredOffset(mySlot) / 4;
  const myRuid = Atomics.load(procCreds, myCredBase);
  const myEuid = Atomics.load(procCreds, myCredBase + 1);

  // the two rt_*queueinfo calls hand the kernel a siginfo to deliver verbatim. two things in it
  // are not theirs to choose: si_code >= 0 means "a user sent this", and SI_TKILL means "a
  // tgkill(2) sent it" -- both of which carry a sender identity the kernel vouches for. a process
  // may claim either only about ITSELF, which is what stops one from forging the other's mail.
  let info: QueuedSignal | null = null;
  if (nr === 138 || nr === 240) {
    const uinfo = from ? from.infoPtr : Number(cpu.getReg(nr === 138 ? 2 : 3));
    if (syscalls.ptrFaults(new Uint8Array(memory.buffer), uinfo, 128)) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return;   // EFAULT
    }
    const ib = uinfo + GUEST_MEM_BASE;
    const code = view.getInt32(ib + 8, true);
    const SI_TKILL = -6;
    if ((code >= 0 || code === SI_TKILL) && targetPid !== myTgid) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-1))); return;    // EPERM
    }
    info = {
      signo: sig, code,
      pid: view.getInt32(ib + 16, true),
      uid: view.getInt32(ib + 20, true),
      value: view.getBigUint64(ib + 24, true),
    };
  }
  // tkill(2) and tgkill(2) say who sent the signal and how: si_code is SI_TKILL, not SI_USER.
  // the distinction is not cosmetic -- linux exempts SI_USER from RLIMIT_SIGPENDING and holds
  // everything else to it, which is the whole difference between kill() and tgkill() here.
  const SI_TKILL = -6;
  const sentCode = info ? info.code : ((nr === 130 || nr === 131) ? SI_TKILL : 0 /* SI_USER */);
  let delivered = false;
  let denied = false;
  let overLimit = false;
  for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
    const base = procSlotOffset(slot) / 4;
    const pid = Atomics.load(procTable, base + PROC_PID);
    if (pid === 0) continue;
    const pgid = Atomics.load(procTable, base + PROC_PGID);
    let match: boolean;
    // tgkill/rt_tgsigqueueinfo name a THREAD inside a thread group. the signal belongs in that
    // thread's own queue -- matching on the group id instead delivered every one of them to the
    // leader, so a program that signals one of its threads saw the handler run on the wrong one.
    if (namedTid > 0) match = pid === namedTid &&
      Atomics.load(procTable, base + PROC_TGID) === targetPid;
    else if (targetPid > 0) match = pid === targetPid;
    else if (targetPid === 0) match = pgid === myPgid;
    else if (targetPid === -1) match = pid !== 1 && pid !== myPid; // broadcast excludes the sender
    else match = pgid === -targetPid;
    if (!match) continue;
    // one user may not signal another's processes. linux allows it when the sender's real or
    // effective uid equals the target's real or saved uid, or when the sender is root.
    if (myEuid !== 0 && pid !== myPid) {
      const tb = procCredOffset(slot) / 4;
      const tRuid = Atomics.load(procCreds, tb);
      const tSuid = Atomics.load(procCreds, tb + 2);
      if (myRuid !== tRuid && myRuid !== tSuid && myEuid !== tRuid && myEuid !== tSuid) {
        denied = true;
        continue;
      }
    }
    delivered = true;
    if (sig === 0) continue; // existence check only
    // SIGCONT restarts a stopped process. that cannot be done BY the target -- it is not running --
    // so the sender does it, and leaves a note for the target's parent to collect with WCONTINUED.
    if (sig === 18 && Atomics.load(procTable, base + PROC_STATE) === PROC_STOPPED) {
      Atomics.store(procTable, base + PROC_CONT_PENDING, 1);
      Atomics.store(procTable, base + PROC_STATE, PROC_RUNNING);
      Atomics.store(procTable, base + PROC_WAIT_FUTEX, 1);
      Atomics.notify(procTable, base + PROC_WAIT_FUTEX);
      wakeParentOf(pid);
    }
    // a queued signal is charged to the TARGET's real user, against the TARGET's RLIMIT_SIGPENDING.
    // over the limit, a real-time signal is refused outright (there is nowhere to keep the siginfo
    // it must carry); a standard signal is still delivered, since its pending bit says all it needs
    // to. linux exempts si_code >= 0, so an ordinary kill(2) never sees this.
    if (sig !== 9 && sentCode < 0 && !sigpendingAllows(slot)) {
      if (sig >= 32) { overLimit = true; delivered = false; continue; }
    }
    if (pid === myPid) {
      syscalls.raiseSignal(sig, info ?? { pid: myPid, uid: syscalls.currentEuid(), code: sentCode, value: 0n });
      continue;
    }
    // SIGKILL is unconditional and un-catchable: a target blocked in Atomics.wait (a pipe
    // read, the bridge mailbox, ...) never runs guest code to observe the signal bit and
    // self-terminate, so setting a bit can't kill it. only the kernel, which owns the Worker
    // objects, can. mirror linux: SIGKILL is enforced by the kernel, the target has no say.
    if (sig === 9) { (self as any).postMessage({ type: 'kill', pid, sig }); continue; }
    // one queue, every signal number: a real-time signal used to have no bit to set here and was
    // simply dropped on its way to another process.
    sigqPush(sigQueue, slot, info ?? { signo: sig, code: sentCode, pid: myPid, uid: syscalls.currentEuid(), value: 0n });
    // wake the target if it is parked on its wait futex (e.g. blocked in wait4)
    Atomics.store(procTable, base + PROC_WAIT_FUTEX, 1);
    Atomics.notify(procTable, base + PROC_WAIT_FUTEX);
    // ...and if it is parked waiting for io. poll, select and epoll sleep on the io generation,
    // not on that futex, so a signal aimed at a process sitting in poll(2) went unseen until
    // whatever it was polling for happened -- which for a program waiting to be interrupted is
    // never. one bump wakes every poller, which is what the generation is for.
    bumpIoGeneration();
  }
  // a target that exists but may not be signalled is EPERM; no such process at all is ESRCH.
  const err = delivered ? 0 : (overLimit ? -11 : (denied ? -1 : -3));
  cpu.setReg(0, BigInt.asUintN(64, BigInt(err)));
}

// setpgid(pid, pgid): move a process into a process group. pid 0 means the caller; pgid 0
// means "use pid as the new group id" (become a group leader). the membership is a field in
// the target's shared procTable slot, so we write it directly here.
function slotOfPid(pid: number): number {
  for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
    const base = procSlotOffset(slot) / 4;
    if (Atomics.load(procTable, base + PROC_PID) === pid &&
        Atomics.load(procTable, base + PROC_STATE) !== PROC_EMPTY) return base;
  }
  return -1;
}

/** the session a pid belongs to, or -1 if there is no such process. */
function sessionOf(base: number): number { return Atomics.load(procTable, base + PROC_SID); }

function handleSetpgid(cpu: CpuState): void {
  let pid = Number(BigInt.asIntN(32, cpu.getReg(0)));
  let pgid = Number(BigInt.asIntN(32, cpu.getReg(1)));
  if (pid === 0) pid = myPid;
  if (pgid === 0) pgid = pid;
  if (pgid < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; } // -EINVAL
  const target = slotOfPid(pid);
  if (target < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-3))); return; }  // ESRCH
  const me = procSlotOffset(mySlot) / 4;
  if (pid !== myPid) {
    // a process may move itself, or a child it forked. someone else's process is not ours to
    // move, and neither is a child that has already exec'd -- it is no longer the program we
    // forked, so its group is its own business.
    if (Atomics.load(procTable, target + PROC_PPID) !== myPid) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-3))); return;                  // ESRCH
    }
    if (sessionOf(target) !== sessionOf(me)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-1))); return; } // EPERM
    if (Atomics.load(procTable, target + PROC_EXECED)) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-13))); return;                 // EACCES
    }
  }
  // a session leader is anchored: its group is its session and cannot be changed.
  if (sessionOf(target) === pid) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-1))); return; } // EPERM
  if (pgid !== pid) {
    // joining an existing group means the group must exist, in this session. a pgid that names
    // nothing is EPERM, not a group brought into being by the request.
    let found = false;
    for (let slot = 0; slot < PROC_TABLE_SLOTS && !found; slot++) {
      const base = procSlotOffset(slot) / 4;
      if (Atomics.load(procTable, base + PROC_STATE) === PROC_EMPTY) continue;
      if (Atomics.load(procTable, base + PROC_PGID) !== pgid) continue;
      found = sessionOf(base) === sessionOf(target);
    }
    if (!found) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-1))); return; }     // EPERM
  }
  Atomics.store(procTable, target + PROC_PGID, pgid);
  cpu.setReg(0, 0n);
}

// setsid(): start a new session, with the caller as its leader and the only member of a new
// process group of the same id. a process that already leads a group cannot -- the group would be
// split across two sessions.
function handleSetsid(cpu: CpuState): void {
  const me = procSlotOffset(mySlot) / 4;
  for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
    const base = procSlotOffset(slot) / 4;
    if (Atomics.load(procTable, base + PROC_STATE) === PROC_EMPTY) continue;
    if (Atomics.load(procTable, base + PROC_PGID) === myPid) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-1))); return;                   // EPERM
    }
  }
  Atomics.store(procTable, me + PROC_SID, myPid);
  Atomics.store(procTable, me + PROC_PGID, myPid);
  cpu.setReg(0, BigInt(myPid));
}

// getsid(pid): the session of a process. pid 0 means the caller.
function handleGetsid(cpu: CpuState): void {
  let pid = Number(BigInt.asIntN(32, cpu.getReg(0)));
  if (pid === 0) pid = myPid;
  if (pid < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-3))); return; }       // ESRCH
  const base = slotOfPid(pid);
  if (base < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-3))); return; }      // ESRCH
  cpu.setReg(0, BigInt(sessionOf(base)));
}

// getpgid(pid): report a process's group. pid 0 means the caller.
function handleGetpgid(cpu: CpuState): void {
  let pid = Number(BigInt.asIntN(32, cpu.getReg(0)));
  if (pid === 0) pid = myPid;
  if (pid < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-3))); return; }       // ESRCH
  const base = slotOfPid(pid);
  if (base < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-3))); return; }      // ESRCH
  cpu.setReg(0, BigInt(Atomics.load(procTable, base + PROC_PGID)));
}

// fcntl(fd, F_GETLK/F_SETLK/F_SETLKW, &flock): round-trip a record-lock request to the kernel's
// shared lock table. the flock struct (aarch64) is { i16 l_type; i16 l_whence; i64 l_start;
// i64 l_len; i32 l_pid; }. we key locks by the fd's resolved path; a non-file fd has no path and
// gets the historical always-grant behaviour (locks on pipes/sockets are meaningless here).
function handleFcntlLock(cpu: CpuState, view: DataView, cmd: number, scope: 'posix' | 'ofd'): void {
  const fd = Number(cpu.getReg(0));
  const flockAddr = Number(cpu.getReg(2));
  // EFAULT comes before everything else, and a bad pointer is not only an out-of-range one: a
  // PROT_NONE mapping is in range and still unreadable, and that is precisely what a caller uses
  // when it wants an address guaranteed to fault.
  if (syscalls.ptrWouldFault(new Uint8Array(memory.buffer), flockAddr, 32)) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; // EFAULT: bad flock pointer
  }
  const flockPtr = flockAddr + GUEST_MEM_BASE;
  const ltype = view.getInt16(flockPtr, true);
  const whence = view.getInt16(flockPtr + 2, true);
  const lstart = Number(view.getBigInt64(flockPtr + 8, true));
  const len = Number(view.getBigInt64(flockPtr + 16, true));
  // order matters, and it is the kernel's order: the descriptor is resolved FIRST (a bad fd is
  // EBADF before anything looks at the request), then the request is validated (a malformed
  // l_type or l_whence is EINVAL), then the lock is checked against how the descriptor was
  // opened -- posix wants a readable descriptor for a shared lock and a writable one for an
  // exclusive lock, and answers EBADF otherwise. that last rule is what makes a read lock on a
  // write-only stdout fail, and it applies whether or not the descriptor names a file.
  if (!syscalls.fdIsOpen(fd) && !kernelPipes.has(fd)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-9))); return; }
  if (ltype !== 0 && ltype !== 1 && ltype !== 2) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
  if (whence !== 0 && whence !== 1 && whence !== 2) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
  if (cmd !== 5 && cmd !== 12 && cmd !== 36) { // F_GETLK only asks, so it is exempt
    const acc = syscalls.fdAccess(fd);
    if (ltype === 0 && !acc.readable) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-9))); return; }
    if (ltype === 1 && !acc.writable) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-9))); return; }
  }

  const path = syscalls.fdPath(fd);
  // a pipe or tty has no on-disk identity to lock against, so an otherwise-valid request is a no-op
  if (path === null) { cpu.setReg(0, 0n); return; }

  // resolve l_whence to an absolute start. SEEK_SET is the common case, but the locking tests
  // lean on SEEK_CUR and SEEK_END (often with a negative l_start), and answering those with a
  // hardcoded 0 turned a legal request into EINVAL.
  let start = lstart;
  if (whence === 1) start += nativeFdOffset(fd);
  else if (whence === 2) start += nativeFdSize(fd, path);
  if (start < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; } // EINVAL

  Atomics.store(myMailbox, MBOX_READY, 0);
  // an ofd lock must name the description; l_pid is required to be 0 on the way in.
  if (scope === 'ofd' && view.getInt32(flockPtr + 24, true) !== 0) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; // EINVAL
  }
  const ownerId = scope === 'ofd' ? syscalls.fdDescriptionId(fd) : 0;
  usedLocks = true;
  (self as any).postMessage({ type: 'lock_req', space: 'record', scope, ownerId, path, cmd, ltype, start, len });
  const r = waitLockReply();

  if (cmd === 5 || cmd === 12) { // F_GETLK: write the (conflicting) lock back into the struct
    const rtype = Atomics.load(myMailbox, MBOX_VALUE0 + 1);
    view.setInt16(flockPtr, rtype, true);
    if (rtype !== 2) {
      const rpid = Atomics.load(myMailbox, MBOX_VALUE0 + 2);
      const cstart = (Atomics.load(myMailbox, MBOX_VALUE0 + 3) >>> 0) + Atomics.load(myMailbox, MBOX_VALUE0 + 4) * 4294967296;
      const clen = (Atomics.load(myMailbox, MBOX_VALUE0 + 5) >>> 0) + Atomics.load(myMailbox, MBOX_VALUE0 + 6) * 4294967296;
      view.setInt16(flockPtr + 2, 0, true); // l_whence = SEEK_SET
      view.setBigInt64(flockPtr + 8, BigInt(cstart), true);
      view.setBigInt64(flockPtr + 16, BigInt(clen), true);
      view.setInt32(flockPtr + 24, rpid, true);
    }
    cpu.setReg(0, 0n);
    return;
  }
  cpu.setReg(0, r < 0 ? BigInt.asUintN(64, BigInt(r)) : 0n);
}

// publish what this process is now running, so /proc/<pid> can describe it from any other
// process without a message. comm is the 16-byte name in the process slot; cmdline is the full
// NUL-separated argv in this slot's cmdline page.
/**
 * one place that builds the syscall handler, so the hooks it needs are attached wherever it is
 * created -- there are three such places (boot, fork resume, exec) and forgetting one leaves a
 * process whose shared mappings silently never register.
 */
function newSyscallHandler(
  brk: bigint,
  io: ConstructorParameters<typeof SyscallHandler>[1],
  fs: ConstructorParameters<typeof SyscallHandler>[2],
): SyscallHandler {
  const h = new SyscallHandler(brk, io, fs);
  h.sharedMapHook = registerSharedMap;
  h.bindSysctls = (vfs) => {
    vfs.setSysctl('/proc/sys/fs/pipe-max-size',
      () => `${Atomics.load(globalWords, GLOBAL_PIPE_MAX)}\n`,
      (text: string) => {
        const v = parseInt(text, 10);
        if (!Number.isFinite(v) || v < 4096) return -22;   // EINVAL: below one page
        if (!h.hasCap(21)) return -1;                      // EPERM (CAP_SYS_ADMIN): a system limit
        Atomics.store(globalWords, GLOBAL_PIPE_MAX, v);
        return 0;
      });
    // kernel.msgmni caps how many sysv message queues may exist at once.
    vfs.setSysctl('/proc/sys/kernel/msgmni',
      () => `${Atomics.load(globalWords, GLOBAL_MSGMNI)}\n`,
      (text: string) => {
        const v = parseInt(text, 10);
        if (!Number.isFinite(v) || v < 0 || v > 32768) return -22;  // EINVAL: linux's own range
        if (!h.hasCap(21)) return -1;                               // EPERM (CAP_SYS_ADMIN)
        Atomics.store(globalWords, GLOBAL_MSGMNI, v);
        return 0;
      });
    // the id the next sysv object of each kind must be given. checkpoint/restore is what these are
    // for: an image is only restorable if the ids in it can be recreated. -1 means "no request",
    // and the value is consumed by the first create that follows.
    for (const [path, slot] of [['msg', GLOBAL_MSG_NEXT_ID], ['sem', GLOBAL_SEM_NEXT_ID],
                                ['shm', GLOBAL_SHM_NEXT_ID]] as Array<[string, number]>) {
      vfs.setSysctl(`/proc/sys/kernel/${path}_next_id`,
        () => `${Atomics.load(globalWords, slot)}\n`,
        (text: string) => {
          const v = parseInt(text, 10);
          if (!Number.isFinite(v) || v < -1 || v > 0x7fffffff) return -22;  // EINVAL
          if (!h.hasCap(21)) return -1;                                     // EPERM (CAP_SYS_ADMIN)
          Atomics.store(globalWords, slot, v);
          return 0;
        });
    }
    // how many distinct mappings one process may hold. it is per-process state rather than a
    // system-wide word: the limit is checked against this process's own vma count.
    vfs.setSysctl('/proc/sys/vm/max_map_count',
      () => `${h.getMaxMapCount()}\n`,
      (text: string) => {
        const v = parseInt(text, 10);
        if (!Number.isFinite(v) || v < 1) return -22;      // EINVAL
        if (!h.hasCap(21)) return -1;                      // EPERM (CAP_SYS_ADMIN)
        return h.setMaxMapCount(v);
      });
    // the posix message queue limits. mq_open reads them out of the same shared words, so
    // lowering one here is felt by the very next call to it, in any process.
    const mqLimit = (path: string, word: number, min: number, max: number) => {
      vfs.setSysctl(path,
        () => `${Atomics.load(globalWords, word)}\n`,
        (text: string) => {
          const v = parseInt(text, 10);
          if (!Number.isFinite(v) || v < min || v > max) return -22;  // EINVAL
          if (!h.hasCap(21)) return -1;                               // EPERM (CAP_SYS_ADMIN)
          Atomics.store(globalWords, word, v);
          return 0;
        });
    };
    mqLimit('/proc/sys/fs/mqueue/msg_max', GLOBAL_MQ_MSG_MAX, 1, 65536);
    mqLimit('/proc/sys/fs/mqueue/msgsize_max', GLOBAL_MQ_MSGSIZE_MAX, 128, 16 * 1024 * 1024);
    mqLimit('/proc/sys/fs/mqueue/queues_max', GLOBAL_MQ_QUEUES_MAX, 0, 1024);
    // the SysV shared-memory limits live with the segments they bound, in the kernel, so these
    // read and write through it. both are 64-bit, and both are settings a program may lower.
    const shmLimit = (path: string, which: number, idx: number) => {
      vfs.setSysctl(path,
        () => {
          const r = sysvShmCall({ op: 'limits' });
          const v = (BigInt(r[idx + 1] >>> 0) << 32n) | BigInt(r[idx] >>> 0);
          return `${v}\n`;
        },
        (text: string) => {
          let v: bigint;
          try { v = BigInt(text.trim()); } catch { return -22; }
          if (v < 0n || v > 0xffffffffffffffffn) return -22;
          if (!h.hasCap(21)) return -1;                    // EPERM (CAP_SYS_ADMIN)
          sysvShmCall({ op: 'setlimit', which,
                        lo: Number(v & 0xffffffffn) | 0, hi: Number((v >> 32n) & 0xffffffffn) | 0 });
          return 0;
        });
    };
    // the timer slack a process is willing to accept on its sleeps. it is per-process, and the
    // owner (or anyone with CAP_SYS_NICE) may change it through this file as well as by prctl.
    vfs.setSysctl('/proc/self/timerslack_ns',
      () => `${h.getTimerSlack()}\n`,
      (text: string) => {
        const v = parseInt(text, 10);
        if (!Number.isFinite(v)) return -22;                 // EINVAL
        if (!h.hasCap(23) && !h.hasCap(21)) return -1;       // EPERM (CAP_SYS_NICE / CAP_SYS_ADMIN)
        return h.setTimerSlack(v);
      });
    shmLimit('/proc/sys/kernel/shmmax', 0, 1);
    shmLimit('/proc/sys/kernel/shmall', 1, 3);
    // uptime is the monotonic clock, which counts from the system's boot rather than this
    // process's start -- so it says the same thing in every process, as it must.
    vfs.setDynamicFile('/proc/uptime', () => new TextEncoder().encode(
      `${(Number(h.clockNs(1) / 1_000_000n) / 1000).toFixed(2)} 0.00\n`));
  };
  h.pipeMaxHook = {
    get: () => Atomics.load(globalWords, GLOBAL_PIPE_MAX),
    // changing a system-wide limit is root's business, exactly as writing /proc/sys is on linux
    set: (v) => { if (!h.hasCap(21)) return -1; Atomics.store(globalWords, GLOBAL_PIPE_MAX, v); return 0; }, // CAP_SYS_ADMIN
  };
  h.refusalHook = (nr, errno) => { (self as any).postMessage({ type: 'refusal', nr, errno }); };
  h.readNativeHook = (nativeFd, count) => {
    if (count <= 0) return new Uint8Array(0);
    const CHUNK = 12288; // what fits base64 in the bridge response slot
    const out = new Uint8Array(count);
    let got = 0;
    while (got < count) {
      const want = Math.min(CHUNK, count - got);
      const h2 = { data: new Uint8Array(0), err: 0 };
      handleBridgeCall('fs_read', { fd: nativeFd, count: want }, (r) => {
        if (r && typeof r.ret === 'number' && r.ret < 0) { h2.err = -r.ret; return; }
        if (r && typeof r.data === 'string' && r.data.length) {
          const bin = atob(r.data);
          const u = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
          h2.data = u;
        }
      });
      if (h2.err) return got > 0 ? out.subarray(0, got) : h2.err; // a partial read still counts
      if (h2.data.length === 0) break;                            // end of file
      out.set(h2.data, got);
      got += h2.data.length;
      if (h2.data.length < want) break;                           // the file ended mid-chunk
    }
    return out.subarray(0, got);
  };
  h.pipeResizeHook = (fd, bytes) => {
    const info = kernelPipes.get(fd);
    if (!info) return -1;
    return new PipeRing(shmArena, info.pipeOff).resize(bytes);
  };
  h.ownerChanged = (fd) => {
    const info = kernelPipes.get(fd);
    if (!info) return; // only pipes notify today; a regular file is always "ready"
    const a = h.asyncState(fd);
    (self as any).postMessage({ type: 'pipe_async', off: info.pipeOff, mode: info.mode, sig: a.sig, on: a.armed });
  };
  h.pipeCapacityHook = (fd) => {
    const info = kernelPipes.get(fd);
    return info ? new PipeRing(shmArena, info.pipeOff).capacity : null;
  };
  // a thread's parent is its thread GROUP's parent. linux has no notion of one thread being the
  // child of another, so getppid() inside a pthread has to answer what the process would answer --
  // we record the creator in the table only so the kernel can tear a group down.
  h.ppidHook = () => {
    const myBase = procSlotOffset(mySlot) / 4;
    const tgid = Atomics.load(procTable, myBase + PROC_TGID);
    if (tgid !== 0 && tgid !== myPid) {
      const lead = slotOfPid(tgid);
      if (lead >= 0) return Atomics.load(procTable, lead + PROC_PPID);
    }
    return Atomics.load(procTable, myBase + PROC_PPID);
  };
  // close_range closes a span of descriptors in one syscall, and a syscall has one reply: the
  // host-side closes have to happen inline rather than as a returned bridge call.
  h.closeNativeHook = (nativeFd) => { handleBridgeCall('fs_close', { fd: nativeFd }, () => {}); };
  h.closeSocketHook = (socketId) => { handleBridgeCall('net_close', { socketId }, () => {}); };
    h.closeUnixHook = (fd) => { closeUnixSock(fd); };
    h.handleHook = {
      put: (ino, path) => { handleBridgeCall('handle_put', { ino, path }, () => {}); },
      get: (ino) => {
        let out: string | null = null;
        handleBridgeCall('handle_get', { ino }, (resp) => {
          if (resp.ret > 0 && resp.bytes) out = new TextDecoder().decode(resp.bytes);
        });
        return out;
      },
    };
  h.peerCredHook = (fd) => unixSocks.get(fd)?.peer ?? null;
  h.subreaperHook = {
    get: () => Atomics.load(procTable, (procSlotOffset(mySlot) / 4) + PROC_SUBREAPER),
    set: (on) => { Atomics.store(procTable, (procSlotOffset(mySlot) / 4) + PROC_SUBREAPER, on); },
  };
  h.sharedUnmapHook = releaseSharedMaps;
  // the /proc/sys files have to point at THIS handler, so they are bound once it exists. binding
  // them earlier in init left a fresh worker with the static placeholder text where a live setting
  // should be, and a write to one of those files went into a file nobody ever read again.
  if (globalWords && vfs) h.bindSysctls?.(vfs);
  return h;
}

// ── MAP_SHARED regions ───────────────────────────────────────────────────────
// see shared-map.ts for the model. this side owns the per-process view: which guest ranges are
// shared, where their authoritative bytes live in the arena, and the shadow that lets a merge
// tell our writes from stale reads.

let shmArena: SharedArrayBuffer;
let sharedMaps: SharedMapping[] = [];

/**
 * wait for a kernel reply that may never come on its own -- a blocking lock request parks until
 * some other process releases the range. a signal has to be able to break that, or an F_SETLKW
 * that no one will ever satisfy is unkillable and the process just disappears into the watchdog.
 *
 * cancelling is race-free by construction: the kernel answers a parked request exactly once, so
 * after asking it to cancel we keep waiting for that one answer. it is either the grant that beat
 * us to it, or the EINTR we asked for.
 */
function waitLockReply(): number {
  let asked = false;
  while (Atomics.load(myMailbox, MBOX_READY) === 0) {
    Atomics.wait(myMailbox, MBOX_READY, 0, 50);
    if (Atomics.load(myMailbox, MBOX_READY) !== 0) break;
    if (asked) continue;
    if (sigqPending(procTable, mySlot) || syscalls.hasPendingDeliverable()) {
      asked = true;
      (self as any).postMessage({ type: 'lock_cancel' });
    }
  }
  return Atomics.load(myMailbox, MBOX_VALUE0);
}


// ── AF_UNIX sockets ──────────────────────────────────────────────────────────
// a unix socket is a bidirectional byte stream between two processes plus a filesystem name to
// meet on. the stream is two arena rings (one per direction), which is exactly what a pipe is;
// the kernel owns only the name -> listener map and the queue of connections waiting to be
// accepted. so nothing here goes near the swift network bridge -- these never touch the network.
// `path` is the key this socket is registered under and only a BOUND socket has one -- it is what
// close gives back. `name` is the same address as the caller spelled it, which is what getsockname
// has to answer, and `peerName` is the far end's, for getpeername.
// MSG_DONTWAIT: this one call does not block, whatever O_NONBLOCK says about the descriptor.
const MSG_DONTWAIT = 0x40;
const unixSocks = new Map<number, { path?: string; name?: string; peerName?: string;
                                    listening: boolean; rx?: number; tx?: number;
                                    // `dgram` is CONNECTIONLESS -- reached by name, no accept.
                                    // `framed` is message boundaries, which a seqpacket keeps too
                                    // even though it is connected like a stream.
                                    dgram?: boolean; framed?: boolean;
                                    peer?: { pid: number; uid: number; gid: number } }>();

/**
 * read a sockaddr_un: { u16 sun_family; char sun_path[108] }.
 *
 * three kinds of address live in that one struct, and they are told apart by the FIRST BYTE of
 * sun_path and by addrlen, not by the string alone: nothing but the family is an autobind request,
 * a leading NUL is an abstract name (whose length comes from addrlen, so it may hold anything),
 * and the rest are filesystem pathnames. reading it as a C string made every abstract name the
 * empty one, so the second program to bind an abstract address collided with the first.
 */
function readSunAddr(view: DataView, addr: number, addrLen: number):
    { raw: string; abstract: boolean } | null {
  if (addr === 0 || addrLen < 2 || addrLen > 110) return null;
  const base = addr + GUEST_MEM_BASE;
  if (view.getUint16(base, true) !== 1) return null; // not AF_UNIX
  const bytes = new Uint8Array(memory.buffer, base + 2, Math.min(108, addrLen - 2));
  if (bytes.length === 0) return { raw: '', abstract: false }; // autobind
  if (bytes[0] === 0) return { raw: '\0' + new TextDecoder().decode(bytes.subarray(1)), abstract: true };
  let n = bytes.indexOf(0);
  if (n < 0) n = bytes.length;
  return { raw: new TextDecoder().decode(bytes.subarray(0, n)), abstract: false };
}

/**
 * the key a name is registered under.
 *
 * an abstract name is its own key. a pathname is keyed by the file it resolves to, so two
 * processes with different working directories that name the same socket find each other.
 */
function unixKey(a: { raw: string; abstract: boolean }): string {
  return a.abstract ? a.raw : (syscalls.socketBindPath(a.raw) ?? a.raw);
}

/** write an address back into a caller's (addr, addrlen) pair, truncating the way linux does. */
function putSunAddr(view: DataView, addrPtr: number, lenPtr: number, name: string | undefined): number {
  const bytes = new Uint8Array(memory.buffer);
  if (lenPtr === 0 || syscalls.ptrFaults(bytes, lenPtr, 4)) return -14; // EFAULT
  const want = view.getInt32(lenPtr + GUEST_MEM_BASE, true);
  if (want < 0) return -22; // EINVAL
  const enc = new TextEncoder();
  let body: Uint8Array;
  if (!name) body = new Uint8Array(0);
  else if (name.charCodeAt(0) === 0) {
    // abstract: the leading NUL is part of the name, and no terminator follows it.
    const nm = enc.encode(name.slice(1));
    body = new Uint8Array(1 + nm.length);
    body.set(nm, 1);
  } else {
    const nm = enc.encode(name);
    body = new Uint8Array(nm.length + 1); // pathnames are reported NUL-terminated
    body.set(nm, 0);
  }
  const full = 2 + body.length;
  const copy = Math.min(Math.max(0, want), full);
  if (copy > 0 && (addrPtr === 0 || syscalls.ptrFaults(bytes, addrPtr, copy))) return -14;
  const buf = new Uint8Array(full);
  buf[0] = 1; buf[1] = 0; // AF_UNIX, little-endian
  buf.set(body, 2);
  if (copy > 0) bytes.set(buf.subarray(0, copy), addrPtr + GUEST_MEM_BASE);
  view.setUint32(lenPtr + GUEST_MEM_BASE, full, true);
  return 0;
}

// linux hands an unnamed socket that binds with nothing but a family a unique abstract name of
// five hex digits, and programs (and ltp) do rely on getsockname reporting one afterwards.
let autobindSeq = 0;

/** ask the kernel about a name; replies are [errno] or [0, rx, tx]. */
function unixCall(type: string, path: string, dgram = false): number[] {
  Atomics.store(myMailbox, MBOX_READY, 0);
  // the caller's credentials travel with every unix-domain call: bind records who owns the name,
  // connect records who is knocking, and SO_PEERCRED reads them back on the far side.
  const uc = syscalls.getCreds();
  (self as any).postMessage({ type, path, dgram, uid: uc.euid, gid: uc.egid });
  awaitMailbox('unix');
  return [0, 1, 2, 3, 4, 5].map((i) => Atomics.load(myMailbox, MBOX_VALUE0 + i));
}

function handleUnixSocket(cpu: CpuState, sockType: number): void {
  // the unix domain carries streams, datagrams and seqpackets. socketpair already accepted all
  // three; socket() refusing two of them was an inconsistency, not a policy.
  const kind = sockType & 0xff;
  if (kind !== 1 && kind !== 2 && kind !== 5) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-94))); return; } // ESOCKTNOSUPPORT
  if (Number(cpu.getReg(2)) !== 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-93))); return; }            // EPROTONOSUPPORT
  const fd = syscalls.allocLowestFd();
  if (fd < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-24))); return; } // EMFILE
  syscalls.registerFd(fd, { type: 'unixsock', sockType: kind });
  // a datagram (or seqpacket) socket keeps message boundaries and is reachable by name rather
  // than by connection, and both of those start here: everything downstream keys off `dgram`.
  unixSocks.set(fd, { listening: false, dgram: kind === 2, framed: kind !== 1 });
  cpu.setReg(0, BigInt(fd));
}

/**
 * socket(AF_NETLINK, SOCK_RAW|SOCK_DGRAM, ...): a descriptor the kernel can push a message into.
 *
 * this is not a general netlink implementation -- there is no routing table or interface database
 * behind it. it exists because mq_notify(SIGEV_THREAD) is specified in terms of one: libc opens a
 * netlink socket, hands the kernel its descriptor, and parks a thread reading it. without the
 * socket, the notification has nowhere to land and libc fails before the syscall.
 */
function handleNetlinkSocket(cpu: CpuState, sockType: number): void {
  const kind = sockType & 0xf;
  if (kind !== 3 /* SOCK_RAW */ && kind !== 2 /* SOCK_DGRAM */) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-94))); return;   // ESOCKTNOSUPPORT
  }
  const fd = syscalls.allocLowestFd();
  if (fd < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-24))); return; } // EMFILE
  Atomics.store(myMailbox, MBOX_READY, 0);
  (self as any).postMessage({ type: 'nl_socket', fd });
  awaitMailbox('netlink');
  const r = Atomics.load(myMailbox, MBOX_VALUE0);
  if (r !== 0) { syscalls.unregisterFd(fd); cpu.setReg(0, BigInt.asUintN(64, BigInt(r))); return; }
  const ring = Atomics.load(myMailbox, MBOX_VALUE0 + 1);
  syscalls.registerFd(fd, { type: 'unixsock' });
  // read-only: nothing this process writes has anywhere to go.
  unixSocks.set(fd, { listening: false, rx: ring });
  if (sockType & 0o2000000) syscalls.markCloexec(fd);          // SOCK_CLOEXEC
  if (sockType & 0o4000) syscalls.registerFd(fd, { type: 'unixsock' }, 0o4000); // SOCK_NONBLOCK
  cpu.setReg(0, BigInt(fd));
}

/**
 * socketpair(2): two endpoints connected to each other, with no name anywhere.
 *
 * the kernel allocates the two rings; this side crosses them so what one endpoint writes the
 * other reads. it is the same stream machinery a connect/accept pair ends up with, minus the
 * rendezvous -- which is why it does not need a path.
 */
function handleUnixSocketpair(cpu: CpuState, view: DataView, sockType: number): void {
  const SOCK_STREAM = 1, SOCK_DGRAM = 2, SOCK_SEQPACKET = 5;
  const kind = sockType & 0xf;
  // a protocol other than 0 does not exist in the unix domain
  if (Number(cpu.getReg(2)) !== 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-93))); return; } // EPROTONOSUPPORT
  if (kind !== SOCK_STREAM && kind !== SOCK_DGRAM && kind !== SOCK_SEQPACKET) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; // EINVAL
  }
  const SOCK_NONBLOCK_F = 0o4000, SOCK_CLOEXEC_F = 0o2000000;
  if (sockType & ~(0xf | SOCK_NONBLOCK_F | SOCK_CLOEXEC_F)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
  const svPtr = Number(cpu.getReg(3));
  if (svPtr === 0 || badGuestPtr(new Uint8Array(memory.buffer), svPtr, 8)) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; // EFAULT
  }
  const fd0 = syscalls.allocLowestFd();
  if (fd0 < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-24))); return; } // EMFILE
  syscalls.registerFd(fd0, { type: 'unixsock' });
  const fd1 = syscalls.allocLowestFd();
  if (fd1 < 0) { syscalls.unregisterFd(fd0); cpu.setReg(0, BigInt.asUintN(64, BigInt(-24))); return; }
  syscalls.registerFd(fd1, { type: 'unixsock' });

  Atomics.store(myMailbox, MBOX_READY, 0);
  (self as any).postMessage({ type: 'unix_pair' });
  awaitMailbox('unix');
  const r = Atomics.load(myMailbox, MBOX_VALUE0);
  if (r !== 0) {
    syscalls.unregisterFd(fd0); syscalls.unregisterFd(fd1);
    cpu.setReg(0, BigInt.asUintN(64, BigInt(r)));
    return;
  }
  const a = Atomics.load(myMailbox, MBOX_VALUE0 + 1);
  const b = Atomics.load(myMailbox, MBOX_VALUE0 + 2);
  // crossed: fd0 writes into `a` and reads from `b`; fd1 is the mirror. a datagram or seqpacket
  // pair keeps message boundaries, which a byte ring does not: those are framed with a length
  // prefix on write and unframed on read (see handleUnixIO).
  const dgram = kind !== SOCK_STREAM;
  // both ends of a socketpair belong to the process that made it, so each is its own peer.
  const spCreds = syscalls.getCreds();
  const spPeer = { pid: myPid, uid: spCreds.euid, gid: spCreds.egid };
  unixSocks.set(fd0, { listening: false, tx: a, rx: b, framed: dgram, peer: spPeer });
  unixSocks.set(fd1, { listening: false, tx: b, rx: a, framed: dgram, peer: spPeer });
  const SOCK_NONBLOCK = SOCK_NONBLOCK_F, SOCK_CLOEXEC = SOCK_CLOEXEC_F;
  const O_NONBLOCK_FLAG = 0o4000;
  if (sockType & SOCK_NONBLOCK) { syscalls.registerFd(fd0, { type: 'unixsock' }, O_NONBLOCK_FLAG); syscalls.registerFd(fd1, { type: 'unixsock' }, O_NONBLOCK_FLAG); }
  if (sockType & SOCK_CLOEXEC) { syscalls.markCloexec(fd0); syscalls.markCloexec(fd1); }
  view.setInt32(svPtr + GUEST_MEM_BASE, fd0, true);
  view.setInt32(svPtr + GUEST_MEM_BASE + 4, fd1, true);
  cpu.setReg(0, 0n);
}

function handleUnixBind(cpu: CpuState, view: DataView): void {
  const fd = Number(cpu.getReg(0));
  const sock = unixSocks.get(fd)!;
  const a = readSunAddr(view, Number(cpu.getReg(1)), Number(cpu.getReg(2)));
  if (a === null) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; } // EINVAL
  // a socket may be bound once.
  if (sock.path) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }     // EINVAL
  if (a.raw === '') { // autobind: keep trying names until one is free
    for (let i = 0; i < 0x100000; i++) {
      const nm = '\0' + (autobindSeq++ & 0xfffff).toString(16).padStart(5, '0');
      const [rc, inbox] = unixCall('unix_bind', nm, !!sock.dgram);
      if (rc === 0) {
        sock.path = nm; sock.name = nm;
        if (sock.dgram && inbox >= 0) sock.rx = inbox;
        cpu.setReg(0, 0n); return;
      }
      if (rc !== -98) { cpu.setReg(0, BigInt.asUintN(64, BigInt(rc))); return; }
    }
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-98))); return; // EADDRINUSE
  }
  const path = unixKey(a);
  // the name is a FILE. creating it is what makes the socket visible to ls, removable by unlink,
  // and what makes a second bind to the same path fail -- and it is also where every path error
  // comes from: a missing directory is ENOENT, a non-directory component is ENOTDIR, an
  // unwritable directory is EACCES. registering the name only in the kernel's own table meant
  // none of that happened and the name existed nowhere a program could see it.
  const disk = a.abstract ? null : syscalls.socketBindPath(a.raw);
  if (disk !== null) {
    let r = -5;
    handleBridgeCall('fs_mksock', { path: disk, mode: 0o777 },
                     (resp: any) => { r = typeof resp?.ret === 'number' ? resp.ret : -5; });
    if (r < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(r))); return; }
  }
  const [r, inbox] = unixCall('unix_bind', path, !!sock.dgram);
  if (r === 0) {
    sock.path = path; sock.name = a.raw;
    // a datagram socket's name IS its mailbox: the kernel hands back the ring anyone who knows
    // that name writes into, and this socket reads from it.
    if (sock.dgram && inbox >= 0) sock.rx = inbox;
  }
  cpu.setReg(0, r === 0 ? 0n : BigInt.asUintN(64, BigInt(r)));
}

/** getsockname(204) / getpeername(205) on a unix socket: only this layer knows the names. */
function handleUnixGetName(cpu: CpuState, view: DataView, peer: boolean): void {
  const sock = unixSocks.get(Number(cpu.getReg(0)))!;
  if (peer && sock.rx === undefined && sock.tx === undefined) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-107))); return; // ENOTCONN
  }
  const r = putSunAddr(view, Number(cpu.getReg(1)), Number(cpu.getReg(2)),
                       peer ? sock.peerName : sock.name);
  cpu.setReg(0, r === 0 ? 0n : BigInt.asUintN(64, BigInt(r)));
}

/** the ring a name currently points at, or a negative errno. */
function dgramRingOf(name: string): number {
  const [r, inbox] = unixCall('unix_lookup', unixKey({ raw: name, abstract: name.charCodeAt(0) === 0 }));
  return r === 0 ? inbox : r;
}

/** where a send with no address of its own goes: the name this socket was connected to. */
function dgramDefaultRing(sock: { dgram?: boolean; tx?: number; peerName?: string }): number {
  if (sock.dgram && sock.peerName !== undefined) return dgramRingOf(sock.peerName);
  if (sock.tx !== undefined) return sock.tx;
  return -107; // ENOTCONN
}

/** give a datagram socket a name of its own if it has none, so its messages can be answered. */
function autobindIfUnnamed(sock: { dgram?: boolean; path?: string; name?: string; rx?: number }): number {
  if (sock.path) return 0;
  for (let i = 0; i < 0x100000; i++) {
    const nm = '\0' + (autobindSeq++ & 0xfffff).toString(16).padStart(5, '0');
    const [rc, inbox] = unixCall('unix_bind', nm, true);
    if (rc === 0) { sock.path = nm; sock.name = nm; if (inbox >= 0) sock.rx = inbox; return 0; }
    if (rc !== -98) return rc;
  }
  return -98; // EADDRINUSE: every name taken, which cannot really happen
}

function handleUnixListen(cpu: CpuState): void {
  const sock = unixSocks.get(Number(cpu.getReg(0)))!;
  if (!sock.path) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; } // EINVAL: bind first
  const r = unixCall('unix_listen', sock.path)[0];
  if (r === 0) {
    sock.listening = true;
    const e = syscalls.getFdEntry(Number(cpu.getReg(0)));
    if (e && e.type === 'unixsock') e.listening = true;   // SO_ACCEPTCONN reports this
  }
  cpu.setReg(0, r === 0 ? 0n : BigInt.asUintN(64, BigInt(r)));
}

function handleUnixConnect(cpu: CpuState, view: DataView): void {
  const fd = Number(cpu.getReg(0));
  const sock = unixSocks.get(fd)!;
  const addr = readSunAddr(view, Number(cpu.getReg(1)), Number(cpu.getReg(2)));
  if (addr === null || addr.raw === '') { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
  if (sock.dgram) {
    // linux gives an unbound socket a name of its own the moment it connects or sends, so that a
    // datagram it sends can be replied to. without that the far end's recvfrom reports no sender.
    const ab = autobindIfUnnamed(sock);
    if (ab < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(ab))); return; }
    const [dr, , dpid, duid, dgid] = unixCall('unix_lookup', unixKey(addr));
    if (dr !== 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(dr))); return; }
    // the destination is remembered by NAME, not by ring: the socket behind it may close and
    // another may take the name, and a datagram sent to a name nobody holds is ECONNREFUSED
    // rather than bytes into a ring that has been given back to the arena.
    sock.peerName = addr.raw;
    sock.peer = { pid: dpid, uid: duid, gid: dgid };
    cpu.setReg(0, 0n);
    return;
  }
  const [r, a, b, ppid, puid, pgid] = unixCall('unix_connect', unixKey(addr));
  if (r !== 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(r))); return; }
  // the name belongs to the far end. recording it as this socket's own made close() give away
  // the LISTENER's address, so the next bind to that path failed while the server still held it.
  sock.tx = a; sock.rx = b; sock.peerName = addr.raw;
  sock.peer = { pid: ppid, uid: puid, gid: pgid };
  cpu.setReg(0, 0n);
}

function handleUnixAccept(cpu: CpuState, view: DataView): void {
  const sock = unixSocks.get(Number(cpu.getReg(0)))!;
  if (!sock.path || !sock.listening) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
  // a blocking accept waits for someone to arrive; a non-blocking one says EAGAIN at once.
  const nonblock = syscalls.isNonBlockingFd(Number(cpu.getReg(0)));
  for (;;) {
    const [r, rx, tx, ppid, puid, pgid] = unixCall('unix_accept', sock.path);
    if (r === 0) {
      const nfd = syscalls.allocLowestFd();
      if (nfd < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-24))); return; }
      syscalls.registerFd(nfd, { type: 'unixsock' });
      // an accepted socket carries the listener's address as its OWN name (that is what
      // getsockname answers on it) but holds no registration: closing it must not unbind the
      // listener that is still accepting on that name.
      // the accepted end is the same KIND of socket as the one that was listening: a seqpacket
      // keeps message boundaries on both sides, and framing one end but not the other means the
      // reader sees the length header as data.
      unixSocks.set(nfd, { listening: false, rx, tx, name: sock.name, framed: sock.framed,
                           peer: { pid: ppid, uid: puid, gid: pgid } });
      // the peer is whoever connected. an unbound client has no address at all, which linux
      // reports as a bare family and a length of two.
      const addrPtr = Number(cpu.getReg(1));
      if (addrPtr !== 0) putSunAddr(view, addrPtr, Number(cpu.getReg(2)), undefined);
      cpu.setReg(0, BigInt(nfd));
      return;
    }
    if (r !== -11 || nonblock) { cpu.setReg(0, BigInt.asUintN(64, BigInt(r))); return; }
    if (pollAndQueueSignals()) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-4))); return; } // EINTR
    Atomics.wait(myMailbox, MBOX_READY, Atomics.load(myMailbox, MBOX_READY), 20);
  }
}

/**
 * whether a unix socket would read or write without blocking. for a listener that is "is a
 * connection queued", asked of the kernel without taking it; for a connected socket it is just
 * the two rings. poll/select/epoll ask through IOInterface.unixReady, because none of this is
 * visible from the syscall layer.
 */
function unixReady(fd: number): { readable: boolean; writable: boolean } {
  const sock = unixSocks.get(fd);
  if (!sock) return { readable: false, writable: false };
  if (sock.listening && sock.path) {
    return { readable: unixCall('unix_pending', sock.path)[1] > 0, writable: false };
  }
  // each direction is answered from its own ring, and a socket may have only one of them: a
  // netlink socket is readable (the kernel writes to it) and never writable.
  // a datagram socket writes to whoever it is told to, so it is writable whether or not it holds
  // a ring of the far end's; what it can READ is its own inbox, and a datagram socket never sees
  // an end-of-file the way a stream does.
  if (sock.dgram && (sock.peerName !== undefined || sock.path !== undefined)) {
    const inbox = sock.rx !== undefined ? new PipeRing(shmArena, sock.rx) : null;
    return { readable: inbox ? inbox.available > 0 : false, writable: true };
  }
  if (sock.rx === undefined && sock.tx === undefined) return { readable: false, writable: false };
  if (sock.rx === undefined || sock.tx === undefined) {
    const one = sock.rx !== undefined ? new PipeRing(shmArena, sock.rx) : null;
    const outRing = sock.tx !== undefined ? new PipeRing(shmArena, sock.tx) : null;
    return {
      readable: one ? one.available > 0 || one.writerClosed : false,
      writable: outRing ? !outRing.readerClosed && outRing.available < outRing.capacity : false,
    };
  }
  const rx = new PipeRing(shmArena, sock.rx);
  const tx = new PipeRing(shmArena, sock.tx);
  // a peer that closed reads as readable, exactly like a pipe: the read returns 0 rather than
  // blocking, and that is what tells the caller the connection is over.
  return {
    readable: rx.available > 0 || rx.writerClosed,
    writable: !tx.readerClosed && tx.available < tx.capacity,
  };
}

/**
 * park until some descriptor may have become ready. `seen` is the generation read BEFORE the
 * readiness check that came up empty, so an event landing in between makes this return at once
 * instead of being slept through. bounded by the same slice interruptibleWaiter uses, so an
 * incoming signal is noticed even though it does not touch this futex.
 */
function waitIo(seen: number, timeoutMs: number): boolean {
  const rem = syscalls.timerRemainingMs();
  const cap = rem > 0 ? Math.min(1000, rem) : 1000;
  // the same shaping the sleep path uses: a wait for the whole remaining time comes back late, so
  // the caller would report a timeout after longer than it asked for. shorter waits converge, and
  // a sub-millisecond remainder is not worth parking for at all.
  const want = timeoutMs < 0 ? cap : Math.min(cap, timeoutMs);
  if (timeoutMs >= 0 && timeoutMs <= 1.2) return waitIoGeneration(seen, 0);
  // parked, and it has to SAY so: a process blocked writing to a full pipe is sleeping, and
  // /proc/<pid>/stat has to report S. it read R for every one of these waits, so anything watching
  // for a blocked process -- ltp does this constantly -- waited for a state that never came.
  setSleeping(true);
  try {
    const r = waitIoGeneration(seen, Math.max(1, want * 0.7));
    // an io wait is a place a signal has to land: poll, select and epoll are all built on this
    // park, and a signal that only reached the queue would sit there until the syscall returned.
    // draining here is what lets them answer EINTR while they are still waiting.
    drainSignalQueue();
    return r;
  } finally {
    setSleeping(false);
  }
}

/** readiness of a message queue, which only the kernel holding it can answer. */
function mqReady(fd: number): { readable: boolean; writable: boolean } | null {
  const id = syscalls.mqIdOf(fd);
  if (id < 0) return null;
  const r = posixMqCall({ op: 'getattr', id });
  if (r[0] < 0) return null;
  const access = syscalls.fdAccess(fd);
  return { readable: access.readable && r[3] > 0, writable: access.writable && r[3] < r[1] };
}

/**
 * readiness of a kernel pipe end, straight off its ring. null when the descriptor is not one.
 *
 * poll, select and epoll ask through IOInterface.pipeReady, because the ring lives in the shared
 * arena and the syscall handler's fd table knows nothing about it.
 */
function pipeReady(fd: number): { readable: boolean; writable: boolean } | null {
  const info = kernelPipes.get(fd);
  if (!info) return null;
  const ring = new PipeRing(shmArena, info.pipeOff);
  if (info.mode === 'read') {
    // a reader is ready when there are bytes, and also when the last writer is gone: that read
    // returns 0 immediately, which is a poll-visible event (EOF), not a reason to keep waiting.
    return { readable: ring.bytesAvailable > 0 || ring.writerClosed, writable: false };
  }
  // a pipe is writable when a whole PIPE_BUF write would fit, not merely when one byte would:
  // POSIX makes a write of up to PIPE_BUF atomic, so poll may only promise room for one. (a pipe
  // resized below PIPE_BUF has to be empty instead.) linux gets the same answer a different way,
  // by asking whether every page-sized buffer is in use, and ltp's edge-triggered test checks it:
  // draining half of a one-page pipe must NOT report the writer as ready.
  const PIPE_BUF = 4096;
  return {
    readable: false,
    writable: ring.freeBytes >= Math.min(PIPE_BUF, ring.capacity) || ring.readerClosed,
  };
}

/** drop a unix socket descriptor: release its rings, give up its name, forget the fd. */
function closeUnixSock(fd: number): void {
  const sock = unixSocks.get(fd);
  if (!sock) return;
  // one end each, and through pipeRef so the KERNEL's holder record shrinks with it. a socket
  // READS its rx ring and WRITES its tx ring; closing both ends of both rings gave every ring two
  // closes for one open, and closing them behind the kernel's back left it convinced this process
  // still held ends it had given up -- so when the process exited the kernel closed them AGAIN and
  // the peer read end-of-file on a socket that was still open at both ends.
  if (sock.rx !== undefined) pipeRef(sock.rx, 'read', -1);
  if (sock.tx !== undefined) pipeRef(sock.tx, 'write', -1);
  // the NAME goes with the socket, whether or not it ever listened -- a bound socket holds the
  // address either way, and leaving the registration behind made every later bind to that path
  // fail. the FILE stays: linux leaves it for unlink to remove, which is why every unix-socket
  // program unlinks its path before binding.
  if (sock.path) unixCall('unix_unbind', sock.path);
  unixSocks.delete(fd);
  syscalls.unregisterFd(fd);
}

/**
 * sendto(2) on a unix socket. a datagram may name its destination on every send, which is the
 * whole point of binding a name: any process that knows it can write to that socket without a
 * connection. on a stream (or with no address) this is write(2) with a flags word.
 */
function handleUnixSendto(cpu: CpuState, view: DataView): void {
  const fd = Number(cpu.getReg(0));
  const sock = unixSocks.get(fd)!;
  const addrPtr = Number(cpu.getReg(4));
  // MSG_DONTWAIT makes this one call non-blocking whatever the descriptor's own flags say.
  const dontwait = (Number(BigInt.asIntN(32, cpu.getReg(3))) & MSG_DONTWAIT) !== 0;
  if (!sock.dgram || addrPtr === 0) { handleUnixIO(64, cpu, dontwait); return; }
  const a = readSunAddr(view, addrPtr, Number(cpu.getReg(5)));
  if (a === null || a.raw === '') { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; } // EINVAL
  const ab = autobindIfUnnamed(sock);
  if (ab < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(ab))); return; }
  const ring = dgramRingOf(a.raw);
  if (ring < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(ring))); return; }
  const buf = Number(cpu.getReg(1));
  const count = Number(cpu.getReg(2));
  const n = unixDgramWrite(fd, ring, new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, count), sock.name, dontwait);
  cpu.setReg(0, n < 0 ? BigInt.asUintN(64, BigInt(n)) : BigInt(n));
}

/** recvfrom(2): read(2) plus the address the message came from, which only a datagram carries. */
function handleUnixRecvfrom(cpu: CpuState, view: DataView): void {
  const fd = Number(cpu.getReg(0));
  const sock = unixSocks.get(fd)!;
  const addrPtr = Number(cpu.getReg(4));
  const dontwait = (Number(BigInt.asIntN(32, cpu.getReg(3))) & MSG_DONTWAIT) !== 0;
  if (!sock.dgram || addrPtr === 0) { handleUnixIO(63, cpu, dontwait); return; }
  if (sock.rx === undefined) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-107))); return; } // ENOTCONN
  const buf = Number(cpu.getReg(1));
  const count = Number(cpu.getReg(2));
  const got = unixDgramRead(fd, sock.rx, count, dontwait);
  if (typeof got === 'number') { cpu.setReg(0, BigInt.asUintN(64, BigInt(got))); return; }
  if (got.data.length > 0) {
    new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, got.data.length).set(got.data);
  }
  putSunAddr(view, addrPtr, Number(cpu.getReg(5)), got.from || undefined);
  cpu.setReg(0, BigInt(got.data.length));
}

/** read/write/close on a connected unix socket, driven straight off its rings. */
function handleUnixIO(nr: number, cpu: CpuState, dontwait = false): void {
  const fd = Number(cpu.getReg(0));
  const sock = unixSocks.get(fd)!;
  if (nr === 57) { // close
    closeUnixSock(fd);
    cpu.setReg(0, 0n);
    return;
  }
  // a socket needs the ring for the direction it is being used in: a netlink socket has only the
  // one the kernel writes into, and reading it is the whole point.
  const canSend = sock.tx !== undefined || (sock.dgram === true && sock.peerName !== undefined);
  if ((nr === 63 && sock.rx === undefined) || (nr !== 63 && !canSend)) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-107))); return; // ENOTCONN
  }
  const buf = Number(cpu.getReg(1));
  const count = Number(cpu.getReg(2));
  // a datagram or seqpacket socket keeps message boundaries; the ring underneath is a byte
  // stream, so each message travels with a four-byte length in front of it.
  if (sock.framed) { handleUnixDgramIO(nr, cpu, sock, fd, buf, count, dontwait); return; }
  if (nr === 63) {
    const got = unixStreamRead(fd, sock.rx!, count, dontwait);
    if (typeof got === 'number') { cpu.setReg(0, BigInt.asUintN(64, BigInt(got))); return; }
    if (got.data.length > 0) {
      new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, got.data.length).set(got.data);
    }
    cpu.setReg(0, BigInt(got.data.length));
    return;
  }
  const n = unixStreamWrite(fd, sock.tx!, new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, count), dontwait);
  cpu.setReg(0, n < 0 ? BigInt.asUintN(64, BigInt(n)) : BigInt(n));
}

/**
 * the stream side of a unix socket, as bytes in and bytes out rather than guest pointers -- write,
 * sendto and sendmsg are the same act on the same ring, and only differ in where the bytes came
 * from.
 */
function unixStreamWrite(fd: number, txOff: number, src: Uint8Array, dontwait = false): number {
  const ring = new PipeRing(shmArena, txOff);
  // O_NONBLOCK means "take what fits and tell me": a socket whose buffer is full answers EAGAIN
  // rather than parking. programs fill a socket on purpose to find its capacity, and one that
  // blocks instead never comes back.
  if (dontwait || syscalls.isNonBlockingFd(fd)) {
    const n = ring.write(src);
    if (n === -1) { syscalls.raiseSignal(13); return -32; }  // EPIPE
    return n === 0 && src.length > 0 ? -11 : n;              // EAGAIN
  }
  const n = ring.writeAll(src, interruptibleWaiter);
  if (n === -1) { syscalls.raiseSignal(13); return -32; }
  return n === -2 ? -4 : n;                                  // EINTR
}

function unixStreamRead(fd: number, rxOff: number, cap: number, dontwait = false):
    number | { data: Uint8Array; truncated: boolean } {
  const ring = new PipeRing(shmArena, rxOff);
  const dst = new Uint8Array(cap);
  if (dontwait || syscalls.isNonBlockingFd(fd)) {
    const n = ring.tryRead(dst);
    if (n === 0 && ring.writerClosed) return { data: dst.subarray(0, 0), truncated: false }; // EOF
    if (n === 0 && cap > 0) return -11;                                                      // EAGAIN
    return { data: dst.subarray(0, n), truncated: false };
  }
  const n = ring.read(dst, interruptibleWaiter);
  if (n < 0) return -4;                                                                      // EINTR
  return { data: dst.subarray(0, n), truncated: false };
}

/**
 * read/write on a DATAGRAM (or seqpacket) unix socket.
 *
 * the difference from a stream is the boundary: one write is one message, one read takes exactly
 * one message, and a read whose buffer is too small keeps the head and DISCARDS the tail rather
 * than leaving it for next time. the ring is a byte stream, so each message carries a four-byte
 * little-endian length in front of it.
 */
function handleUnixDgramIO(nr: number, cpu: CpuState, sock: { rx?: number; tx?: number; dgram?: boolean;
                                                               name?: string; peerName?: string },
                           fd: number, buf: number, count: number, dontwait = false): void {
  if (nr === 63) {
    const got = unixDgramRead(fd, sock.rx!, count, dontwait);
    if (typeof got === 'number') { cpu.setReg(0, BigInt.asUintN(64, BigInt(got))); return; }
    if (got.data.length > 0) {
      new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, got.data.length).set(got.data);
    }
    cpu.setReg(0, BigInt(got.data.length));
    return;
  }
  const ring = dgramDefaultRing(sock);
  if (ring < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(ring))); return; }
  const n = unixDgramWrite(fd, ring, new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, count), sock.name, dontwait);
  cpu.setReg(0, n < 0 ? BigInt.asUintN(64, BigInt(n)) : BigInt(n));
}

/**
 * one message onto the ring: its length, the sender's name, then the bytes.
 *
 * the name is what recvfrom answers with. a datagram socket reached by name has no connection to
 * ask, so the address has to travel with the message or the receiver cannot reply.
 */
function unixDgramWrite(fd: number, txOff: number, src: Uint8Array, fromName?: string, dontwait = false): number {
  const ring = new PipeRing(shmArena, txOff);
  const count = src.length;
  const nameBytes = fromName ? new TextEncoder().encode(fromName) : new Uint8Array(0);
  const framed = new Uint8Array(8 + nameBytes.length + count);
  const fv = new DataView(framed.buffer);
  fv.setUint32(0, count, true);
  fv.setUint32(4, nameBytes.length, true);
  framed.set(nameBytes, 8);
  if (count > 0) framed.set(src, 8 + nameBytes.length);
  // a datagram is all-or-nothing, so a nonblocking send either has room for the whole message
  // or answers EAGAIN -- half a message would be a message the receiver never asked for.
  if (dontwait || syscalls.isNonBlockingFd(fd)) {
    if (ring.freeBytes < framed.length) {
      if (ring.readerClosed) { syscalls.raiseSignal(13); return -32; }
      return -11;
    }
    const w = ring.write(framed);
    if (w === -1) { syscalls.raiseSignal(13); return -32; }
    return count;
  }
  const n = ring.writeAll(framed, interruptibleWaiter);
  if (n === -1) { syscalls.raiseSignal(13); return -32; }
  if (n === -2) return -4;
  return count;
}

/** exactly one message off the ring; what does not fit in `cap` is discarded, not kept. */
function unixDgramRead(fd: number, rxOff: number, cap: number, dontwait = false):
    number | { data: Uint8Array; truncated: boolean; from: string } {
  const ring = new PipeRing(shmArena, rxOff);
  const hdr = new Uint8Array(8);
  const empty = { data: hdr.subarray(0, 0), truncated: false, from: '' };
  if ((dontwait || syscalls.isNonBlockingFd(fd)) && ring.bytesAvailable < 8) {
    if (ring.writerClosed) return empty; // EOF
    return -11;
  }
  const fill = (dst: Uint8Array): number => {
    let n = 0;
    while (n < dst.length) {
      const r = ring.read(dst.subarray(n), interruptibleWaiter);
      if (r < 0) return -4;
      if (r === 0) break;
      n += r;
    }
    return n;
  };
  const got = fill(hdr);
  if (got < 0) return -4;
  if (got < 8) return empty;
  const hv = new DataView(hdr.buffer);
  const len = hv.getUint32(0, true);
  const nameLen = hv.getUint32(4, true);
  let from = '';
  if (nameLen > 0) {
    const nb = new Uint8Array(nameLen);
    if (fill(nb) < 0) return -4;
    from = new TextDecoder().decode(nb);
  }
  const body = new Uint8Array(len);
  const have = fill(body);
  if (have < 0) return -4;
  const copied = Math.min(have, cap);
  return { data: body.subarray(0, copied), truncated: have > cap, from };
}

// ── sendmsg/recvmsg on a unix socket, and the descriptors they can carry ────
//
// SCM_RIGHTS is how one process hands another an open file: the receiver gets its OWN descriptor
// for the same object, not a number that means nothing outside the sender. so the sender takes the
// reference at send time (a dup'd swift fd, another reference on a socket, another end on a ring)
// and describes the object; the kernel holds that description against the stream position it was
// sent at; the receiver rebuilds a descriptor from it when it reads past that position. that is
// what makes it work across workers, where an fd number is meaningless.

/** take a reference to what `fd` names, and describe it so another process can rebuild it. */
function describeFdForScm(fd: number): any | null {
  const e = syscalls.getFdEntry(fd);
  if (e?.type === 'native') {
    let nfd = -1;
    handleBridgeCall('fs_dup', { fd: e.nativeFd }, (resp) => { nfd = resp.ret; });
    if (nfd < 0) return null;
    // the ofd id travels too: the receiver shares the sender's open file description, which is
    // what makes the offset and any OFD lock the same object on both sides.
    return { k: 'native', nfd, path: e.path, ofdId: e.ofdId, isDir: e.isDir, unlinked: e.unlinked };
  }
  if (e?.type === 'socket') {
    handleBridgeCall('net_ref', { socketId: e.socketId }, () => {});
    return { k: 'socket', socketId: e.socketId, domain: e.domain, sockType: e.sockType };
  }
  if (e?.type === 'unixsock') {
    const s = unixSocks.get(fd);
    if (!s || s.rx === undefined || s.tx === undefined) return null;
    new PipeRing(shmArena, s.rx).addReader();
    new PipeRing(shmArena, s.tx).addWriter();
    return { k: 'unix', rx: s.rx, tx: s.tx, dgram: s.dgram, framed: s.framed, peer: s.peer,
             ends: [[s.rx, 'r'], [s.tx, 'w']] };
  }
  const kp = kernelPipes.get(fd);
  if (kp) {
    const ring = new PipeRing(shmArena, kp.pipeOff);
    if (kp.mode === 'read') ring.addReader(); else ring.addWriter();
    return { k: 'pipe', off: kp.pipeOff, mode: kp.mode,
             ends: [[kp.pipeOff, kp.mode === 'read' ? 'r' : 'w']] };
  }
  const vp = vfs?.fdPath(fd);
  if (vp) return { k: 'vfs', path: vp };
  return null;
}

/** build a descriptor in THIS process for something another process sent. */
function materializeScmFd(desc: any): number {
  const fd = syscalls.allocLowestFd();
  if (fd < 0) return -1;
  switch (desc?.k) {
    case 'native':
      syscalls.registerFd(fd, { type: 'native', nativeFd: desc.nfd, path: desc.path,
                                ofdId: desc.ofdId ?? 0, isDir: desc.isDir, unlinked: desc.unlinked });
      return fd;
    case 'socket':
      syscalls.registerFd(fd, { type: 'socket', socketId: desc.socketId, domain: desc.domain,
                                sockType: desc.sockType });
      return fd;
    case 'unix':
      // the two rings arrive the way the SENDER named them, and this end reads and writes the same
      // way it did: a descriptor is the same object on both sides, not a mirrored one.
      syscalls.registerFd(fd, { type: 'unixsock' });
      unixSocks.set(fd, { listening: false, rx: desc.rx, tx: desc.tx, dgram: desc.dgram, framed: desc.framed,
                          peer: desc.peer });
      return fd;
    case 'pipe':
      syscalls.registerFd(fd, { type: desc.mode === 'read' ? 'pipe_read' : 'pipe_write',
                                pipe: { data: [], writeClosed: false, readClosed: false } });
      kernelPipes.set(fd, { pipeOff: desc.off, mode: desc.mode });
      if (fd >= nextLocalFd) nextLocalFd = fd + 1;
      return fd;
    case 'vfs': {
      syscalls.unregisterFd(fd);
      const r = vfs ? vfs.open(desc.path, 0, 0) : -1;
      return r;
    }
    default:
      syscalls.unregisterFd(fd);
      return -1;
  }
}

/** the parts of a struct msghdr this kernel reads. */
function readMsghdr(view: DataView, p: number) {
  const b = p + GUEST_MEM_BASE;
  return {
    name: Number(view.getBigUint64(b, true)),
    nameLen: view.getUint32(b + 8, true),
    iov: Number(view.getBigUint64(b + 16, true)),
    iovLen: Number(view.getBigUint64(b + 24, true)),
    control: Number(view.getBigUint64(b + 32, true)),
    controlLen: Number(view.getBigUint64(b + 40, true)),
    flags: view.getInt32(b + 48, true),
  };
}

function gatherIov(view: DataView, iov: number, n: number): Uint8Array {
  let total = 0;
  for (let i = 0; i < n; i++) total += Number(view.getBigUint64(iov + i * 16 + 8 + GUEST_MEM_BASE, true));
  const out = new Uint8Array(total);
  let off = 0;
  for (let i = 0; i < n; i++) {
    const base = Number(view.getBigUint64(iov + i * 16 + GUEST_MEM_BASE, true));
    const len = Number(view.getBigUint64(iov + i * 16 + 8 + GUEST_MEM_BASE, true));
    out.set(new Uint8Array(memory.buffer, base + GUEST_MEM_BASE, len), off);
    off += len;
  }
  return out;
}

function scatterIov(view: DataView, iov: number, n: number, data: Uint8Array): number {
  let left = data.length, src = 0;
  for (let i = 0; i < n && left > 0; i++) {
    const base = Number(view.getBigUint64(iov + i * 16 + GUEST_MEM_BASE, true));
    const len = Number(view.getBigUint64(iov + i * 16 + 8 + GUEST_MEM_BASE, true));
    const take = Math.min(len, left);
    new Uint8Array(memory.buffer, base + GUEST_MEM_BASE, take).set(data.subarray(src, src + take));
    src += take; left -= take;
  }
  return data.length - left;
}

/** every SCM_RIGHTS descriptor named in a control buffer, described for transport. */
function collectScmRights(view: DataView, control: number, controlLen: number): { fds: any[]; err: number } {
  const out: any[] = [];
  let off = 0;
  while (off + 16 <= controlLen) {
    const b = control + off + GUEST_MEM_BASE;
    const len = Number(view.getBigUint64(b, true));
    const level = view.getInt32(b + 8, true);
    const type = view.getInt32(b + 12, true);
    if (len < 16 || off + len > controlLen) return { fds: out, err: 22 }; // EINVAL
    if (level === 1 && type === 1) {
      const count = Math.floor((len - 16) / 4);
      for (let i = 0; i < count; i++) {
        const gfd = view.getInt32(b + 16 + i * 4, true);
        if (!syscalls.fdIsOpen(gfd)) return { fds: out, err: 9 };  // EBADF
        const d = describeFdForScm(gfd);
        if (!d) return { fds: out, err: 9 };
        out.push(d);
      }
    } else if (level !== 1) {
      return { fds: out, err: 92 }; // ENOPROTOOPT: a level this socket has no ancillary data for
    }
    off += (len + 7) & ~7;
  }
  return { fds: out, err: 0 };
}

/** sendmsg(2) on a unix socket: the same bytes as write, plus whatever rides alongside them. */
function handleUnixSendmsg(cpu: CpuState, view: DataView): void {
  const fd = Number(cpu.getReg(0));
  const sock = unixSocks.get(fd)!;
  const hp = Number(cpu.getReg(1));
  const flags = Number(BigInt.asIntN(32, cpu.getReg(2)));
  const bytes = new Uint8Array(memory.buffer);
  if (hp === 0 || syscalls.ptrFaults(bytes, hp, 56)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; }
  const h = readMsghdr(view, hp);
  if (h.iovLen > 1024) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-90))); return; } // EMSGSIZE
  if (h.iovLen > 0 && (h.iov === 0 || syscalls.ptrFaults(bytes, h.iov, h.iovLen * 16))) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return;
  }
  for (let i = 0; i < h.iovLen; i++) {
    const base = Number(view.getBigUint64(h.iov + i * 16 + GUEST_MEM_BASE, true));
    const len = Number(view.getBigUint64(h.iov + i * 16 + 8 + GUEST_MEM_BASE, true));
    if (len > 0 && syscalls.ptrFaults(bytes, base, len)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; }
  }
  if (h.controlLen > 0 && (h.control === 0 || syscalls.ptrFaults(bytes, h.control, h.controlLen))) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return;
  }
  // MSG_OOB has no meaning in the unix domain, and neither does anything outside this set:
  // MSG_NOSIGNAL, MSG_DONTWAIT, MSG_MORE, MSG_EOR and MSG_DONTROUTE (which costs nothing here).
  if (flags & ~(0x4000 | 0x40 | 0x8000 | 0x80 | 0x4)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-95))); return; }
  const msgDontwait = (flags & MSG_DONTWAIT) !== 0;
  // where these bytes go: the address this message carries if it has one, else wherever the socket
  // sends by default. only a datagram may name a destination per message.
  let dest: number;
  if (sock.dgram && h.name !== 0) {
    const a = readSunAddr(view, h.name, h.nameLen);
    if (a === null || a.raw === '') { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
    const ab = autobindIfUnnamed(sock);
    if (ab < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(ab))); return; }
    dest = dgramRingOf(a.raw);
  } else {
    dest = dgramDefaultRing(sock);
  }
  if (dest < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(dest))); return; }
  const scm = h.controlLen > 0 ? collectScmRights(view, h.control, h.controlLen) : { fds: [], err: 0 };
  if (scm.err) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-scm.err))); return; }
  const payload = gatherIov(view, h.iov, h.iovLen);
  const tx = new PipeRing(shmArena, dest);
  // the descriptors belong to the FIRST byte of this message, which is where the writer is now.
  // they are queued BEFORE the bytes go in: a reader that takes those bytes must be able to find
  // them, and a record queued afterwards can be missed by a reader that got there first.
  const pos = tx.writePos;
  if (scm.fds.length) {
    handleBridgeCall('scm_send', { ring: dest, pos, fds: scm.fds }, () => {});
    tx.addScm(scm.fds.length);
  }
  const n = sock.framed ? unixDgramWrite(fd, dest, payload, sock.name, msgDontwait) : unixStreamWrite(fd, dest, payload, msgDontwait);
  if (n < 0) {
    if (scm.fds.length) {
      handleBridgeCall('scm_unsend', { ring: dest, pos, n: scm.fds.length }, () => {});
      tx.addScm(-scm.fds.length);
    }
    for (const d of scm.fds) releaseScmDescriptor(d);
    cpu.setReg(0, BigInt.asUintN(64, BigInt(n)));
    return;
  }
  cpu.setReg(0, BigInt(n));
}

/** give back a descriptor that was taken for a send which then failed. */
function releaseScmDescriptor(d: any): void {
  if (d?.k === 'native') handleBridgeCall('fs_close', { fd: d.nfd }, () => {});
  else if (d?.k === 'socket') handleBridgeCall('net_close', { socketId: d.socketId }, () => {});
  else if (d?.ends) {
    for (const [off, mode] of d.ends as Array<[number, string]>) {
      const r = new PipeRing(shmArena, off);
      if (mode === 'w') r.closeWriter(); else r.closeReader();
    }
  }
}

/** recvmsg(2) on a unix socket. */
function handleUnixRecvmsg(cpu: CpuState, view: DataView): void {
  const fd = Number(cpu.getReg(0));
  const sock = unixSocks.get(fd)!;
  const hp = Number(cpu.getReg(1));
  const flags = Number(BigInt.asIntN(32, cpu.getReg(2)));
  const bytes = new Uint8Array(memory.buffer);
  if (hp === 0 || syscalls.ptrFaults(bytes, hp, 56)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; }
  const h = readMsghdr(view, hp);
  if (h.iovLen > 1024) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-90))); return; }
  if (h.iovLen > 0 && (h.iov === 0 || syscalls.ptrFaults(bytes, h.iov, h.iovLen * 16))) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return;
  }
  if (sock.rx === undefined) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-107))); return; // ENOTCONN
  }
  let cap = 0;
  for (let i = 0; i < h.iovLen; i++) cap += Number(view.getBigUint64(h.iov + i * 16 + 8 + GUEST_MEM_BASE, true));
  const rx = new PipeRing(shmArena, sock.rx);
  const msgDontwait = (flags & MSG_DONTWAIT) !== 0;
  const got = sock.framed ? unixDgramRead(fd, sock.rx, cap, msgDontwait) : unixStreamRead(fd, sock.rx, cap, msgDontwait);
  if (typeof got === 'number') { cpu.setReg(0, BigInt.asUintN(64, BigInt(got))); return; }
  const wrote = scatterIov(view, h.iov, h.iovLen, got.data);
  // whatever the sender attached to the bytes just consumed. an empty read still claims records
  // queued at this position, which is how a message of nothing but descriptors arrives.
  let msgFlags = got.truncated ? 0x20 : 0;   // MSG_TRUNC
  let controlUsed = 0;
  const after = rx.readPos;
  let records: any[] = [];
  // records belonging to the bytes just consumed. a read that took nothing instead claims a record
  // sitting exactly where it is: that is a message of descriptors and no data, which linux delivers
  // as a zero-length read with ancillary data.
  const claimTo = got.data.length > 0 ? (after - 1) | 0 : after | 0;
  // nothing in flight means nothing to ask about, which is the case on every ordinary read.
  if (rx.scmPending > 0) {
    handleBridgeCall('scm_take', { ring: sock.rx, to: claimTo }, (resp) => {
      if (resp.ret > 0 && resp.bytes) {
        try { records = JSON.parse(new TextDecoder().decode(resp.bytes)); } catch { records = []; }
      }
    });
    if (records.length) rx.addScm(-records.length);
  }
  if (records.length) {
    const room = h.controlLen >= 16 ? Math.floor((h.controlLen - 16) / 4) : 0;
    const give = Math.min(room, records.length);
    if (give < records.length) msgFlags |= 0x8;  // MSG_CTRUNC
    const newFds: number[] = [];
    for (let i = 0; i < give; i++) {
      const nfd = materializeScmFd(records[i]);
      if (nfd >= 0) {
        newFds.push(nfd);
        if (flags & 0x40000000) syscalls.markCloexec(nfd);  // MSG_CMSG_CLOEXEC
      }
    }
    // anything that did not fit is closed, exactly as linux does with a truncated SCM_RIGHTS.
    for (let i = give; i < records.length; i++) releaseScmDescriptor(records[i]);
    if (newFds.length) {
      const cb = h.control + GUEST_MEM_BASE;
      const len = 16 + newFds.length * 4;
      view.setBigUint64(cb, BigInt(len), true);
      view.setInt32(cb + 8, 1, true);   // SOL_SOCKET
      view.setInt32(cb + 12, 1, true);  // SCM_RIGHTS
      for (let i = 0; i < newFds.length; i++) view.setInt32(cb + 16 + i * 4, newFds[i], true);
      controlUsed = (len + 7) & ~7;
    }
  }
  // who sent this. a datagram carries its sender's name with it; a connected pair has no address
  // to report on either side, which is a name of zero bytes rather than a lie.
  let nameLen = 0;
  if (h.name !== 0 && sock.framed && (got as any).from) {
    const buf = new Uint8Array(memory.buffer);
    const nm: string = (got as any).from;
    const enc = new TextEncoder().encode(nm.charCodeAt(0) === 0 ? nm.slice(1) : nm);
    const body = new Uint8Array(nm.charCodeAt(0) === 0 ? 1 + enc.length : enc.length + 1);
    body.set(enc, nm.charCodeAt(0) === 0 ? 1 : 0);
    const full = 2 + body.length;
    const copy = Math.min(h.nameLen, full);
    if (copy > 0 && !syscalls.ptrFaults(buf, h.name, copy)) {
      const out = new Uint8Array(full);
      out[0] = 1; out[1] = 0; // AF_UNIX
      out.set(body, 2);
      buf.set(out.subarray(0, copy), h.name + GUEST_MEM_BASE);
      nameLen = full;
    }
  }
  view.setUint32(hp + 8 + GUEST_MEM_BASE, nameLen, true);
  view.setBigUint64(hp + 40 + GUEST_MEM_BASE, BigInt(controlUsed), true);
  view.setInt32(hp + 48 + GUEST_MEM_BASE, msgFlags, true);
  cpu.setReg(0, BigInt(wrote));
}

/** where a native fd currently sits, for resolving a SEEK_CUR lock range. */
/**
 * copy_file_range(fd_in, off_in, fd_out, off_out, len, flags) between two real files.
 *
 * an explicit offset leaves the descriptor's position alone and is written back to the caller; a
 * null one uses the position and advances it. that distinction is the whole interface, and it is
 * what lets a copy loop interleave with ordinary reads on the same descriptor.
 */
function handleCopyFileRange(cpu: CpuState, view: DataView): void {
  const err = (e: number) => cpu.setReg(0, BigInt.asUintN(64, BigInt(-e)));
  const bytes = new Uint8Array(memory.buffer);
  const inFd = Number(BigInt.asIntN(32, cpu.getReg(0)));
  const offInPtr = Number(cpu.getReg(1));
  const outFd = Number(BigInt.asIntN(32, cpu.getReg(2)));
  const offOutPtr = Number(cpu.getReg(3));
  const len = Number(BigInt.asIntN(64, cpu.getReg(4)));
  const flags = Number(BigInt.asUintN(32, cpu.getReg(5)));

  if (flags !== 0) { err(22); return; }                                  // EINVAL: no flags defined
  if (len < 0) { err(22); return; }
  if (!syscalls.fdAccess(inFd).readable || !syscalls.fdAccess(outFd).writable) { err(9); return; }
  if (offInPtr !== 0 && syscalls.ptrWriteFaults(bytes, offInPtr, 8)) { err(14); return; }
  if (offOutPtr !== 0 && syscalls.ptrWriteFaults(bytes, offOutPtr, 8)) { err(14); return; }

  const inNative = syscalls.fdNative(inFd);
  const outNative = syscalls.fdNative(outFd);
  let inOff = offInPtr !== 0 ? Number(view.getBigInt64(offInPtr + GUEST_MEM_BASE, true))
                             : nativeFdOffset(inFd);
  let outOff = offOutPtr !== 0 ? Number(view.getBigInt64(offOutPtr + GUEST_MEM_BASE, true))
                               : nativeFdOffset(outFd);
  if (inOff < 0 || outOff < 0) { err(22); return; }

  // the same file cannot be both ends of an overlapping copy: the result would depend on the order
  // the kernel happened to move the bytes in.
  if (inNative === outNative && inOff < outOff + len && outOff < inOff + len) { err(22); return; }

  let copied = 0;
  while (copied < len) {
    const chunk = readNativeFileAt(inNative, inOff + copied, Math.min(65536, len - copied));
    if (chunk.length === 0) break;   // end of the source
    let wrote = 0;
    handleBridgeCall('fs_pwrite', { fd: outNative, offset: outOff + copied,
                                    data: bytesToBase64Fast(chunk) }, (r) => {
      if (r && typeof r.ret === 'number') wrote = r.ret;
    });
    if (wrote < 0) { if (copied === 0) { err(-wrote); return; } break; }
    copied += wrote;
    if (wrote < chunk.length) break;
  }

  if (offInPtr !== 0) view.setBigInt64(offInPtr + GUEST_MEM_BASE, BigInt(inOff + copied), true);
  else handleBridgeCall('fs_lseek', { fd: inNative, offset: inOff + copied, whence: 0 }, () => {});
  if (offOutPtr !== 0) view.setBigInt64(offOutPtr + GUEST_MEM_BASE, BigInt(outOff + copied), true);
  else handleBridgeCall('fs_lseek', { fd: outNative, offset: outOff + copied, whence: 0 }, () => {});
  // the destination grew and its mtime moved, and nothing on the ordinary write path saw it happen.
  if (copied > 0) syscalls.invalidateStatFd(outFd);
  cpu.setReg(0, BigInt(copied));
}

function nativeFdOffset(fd: number): number {
  const nfd = syscalls.fdNative(fd);
  if (nfd < 0) return 0;
  let off = 0;
  handleBridgeCall('fs_lseek', { fd: nfd, offset: 0, whence: 1 }, (r) => {
    if (r && typeof r.ret === 'number' && r.ret >= 0) off = r.ret;
  });
  return off;
}

/** how big the file behind an fd is, for resolving a SEEK_END lock range. */
function nativeFdSize(fd: number, path: string): number {
  let size = 0;
  handleBridgeCall('fs_stat', { path, creds: syscalls.getCreds() }, (r) => {
    if (r && r._meta && typeof r._meta[1] === 'number') size = r._meta[1];
  });
  return size;
}

/** called when a syscall completes, by whichever path handled it. */
function syscallDone(): void {
  publishSyscall(-1);
  // a syscall that blocked spent most of its time parked, and parked time is nobody's cpu time.
  if (sysEnterMs >= 0) {
    sysAccumMs += Math.max(0, (performance.now() - sysEnterMs) - (parkedTotalMs() - sysEnterParkedMs));
    sysEnterMs = -1;
  }
  if (sharedMaps.length) reconcileShared();
}

function reconcileShared(): void {
  const guest = new Uint8Array(memory.buffer);
  const arena = new Uint8Array(shmArena);
  for (const m of sharedMaps) {
    const start = m.addr + GUEST_MEM_BASE;
    if (start + m.len > guest.length) continue; // mapping outlived a memory shrink; nothing to do
    reconcile(guest.subarray(start, start + m.len), arena.subarray(m.arenaOff, m.arenaOff + m.len), m.shadow);
  }
}

/** ask the kernel for arena space and start tracking a new shared mapping. */
function registerSharedMap(addr: number, len: number, key?: string): void {
  Atomics.store(myMailbox, MBOX_READY, 0);
  (self as any).postMessage({ type: 'shm_alloc', len, key });
  awaitMailbox('shm_map');
  const off = Atomics.load(myMailbox, MBOX_VALUE0);
  if (off < 0) return; // arena exhausted: the mapping still works, just privately
  const created = Atomics.load(myMailbox, MBOX_VALUE0 + 1) === 1;

  const guest = new Uint8Array(memory.buffer, addr + GUEST_MEM_BASE, len);
  const block = new Uint8Array(shmArena, off, len);
  if (created) {
    // ours to seed: a file mapping has just been read in, an anonymous one is zero.
    block.set(guest);
  } else {
    // joining someone else's mapping: their bytes are the truth, ours were only a fresh read
    // of the file and would clobber writes they have already made.
    guest.set(block);
  }
  sharedMaps.push({ addr, len, arenaOff: off, shadow: new Uint8Array(block) });
}

function releaseSharedMaps(start: number, end: number): void {
  const kept: SharedMapping[] = [];
  for (const m of sharedMaps) {
    if (m.addr + m.len <= start || m.addr >= end) { kept.push(m); continue; }
    Atomics.store(myMailbox, MBOX_READY, 0);
    (self as any).postMessage({ type: 'shm_free', off: m.arenaOff });
    awaitMailbox('shm_release');
  }
  sharedMaps = kept;
}

// where each SysV segment is attached in this process, so shmdt (which names an address) can find
// the id it belongs to.
let sysvAttached = new Map<number, number>();

let currentArgv: string[] = [];

function publishIdentity(argv: string[]): void {
  currentArgv = argv;
  if (vfs) vfs.selfPid = myPid;
  const name = (argv[0] ?? '').split('/').pop() ?? '';
  const comm = new TextEncoder().encode(name).subarray(0, 15);
  const commBytes = new Uint8Array(16);
  commBytes.set(comm);
  new Uint8Array(kernelSAB, procSlotOffset(mySlot) + PROC_COMM * 4, 16).set(commBytes);

  const joined = argv.join('\0') + '\0';
  const raw = new TextEncoder().encode(joined).subarray(0, PROC_CMDLINE_SLOT_SIZE - 1);
  const slot = new Uint8Array(kernelSAB, procCmdlineOffset(mySlot), PROC_CMDLINE_SLOT_SIZE);
  slot.fill(0);
  slot.set(raw);
}

// every live process, read straight out of the shared table. this is what /proc/<pid> is built
// from, so `ps` costs one scan of 64 slots and no round trips.
function procSnapshot(): Array<{ pid: number; ppid: number; pgid: number; tgid: number; state: number; blocked: boolean; comm: string; cmdline: string }> {
  const out = [];
  for (let s = 0; s < PROC_TABLE_SLOTS; s++) {
    const base = procSlotOffset(s) / 4;
    const state = Atomics.load(procTable, base + PROC_STATE);
    if (state === PROC_EMPTY) continue;
    const pid = Atomics.load(procTable, base + PROC_PID);
    if (pid <= 0) continue;
    const commRaw = new Uint8Array(kernelSAB, procSlotOffset(s) + PROC_COMM * 4, 16);
    let n = commRaw.indexOf(0);
    if (n < 0) n = 16;
    const cmdRaw = new Uint8Array(kernelSAB, procCmdlineOffset(s), PROC_CMDLINE_SLOT_SIZE);
    let cn = cmdRaw.length;
    while (cn > 0 && cmdRaw[cn - 1] === 0) cn--;
    out.push({
      pid,
      ppid: Atomics.load(procTable, base + PROC_PPID),
      pgid: Atomics.load(procTable, base + PROC_PGID),
      sid: Atomics.load(procTable, base + PROC_SID),
      tgid: Atomics.load(procTable, base + PROC_TGID),
      state,
      blocked: Atomics.load(procTable, base + PROC_BLOCKED) !== 0,
      utimeMs: Atomics.load(procTable, base + PROC_UTIME_MS),
      stimeMs: Atomics.load(procTable, base + PROC_STIME_MS),
      nice: Atomics.load(procTable, base + PROC_NICE),
      syscall: Atomics.load(procTable, base + PROC_SYSCALL),
      comm: new TextDecoder().decode(commRaw.subarray(0, n)),
      cmdline: new TextDecoder().decode(cmdRaw.subarray(0, cn)),
    });
  }
  return out;
}

// set once this process touches advisory locking at all. it gates the close-time notification
// below so processes that never lock never pay for it.
let usedLocks = false;

// tell the kernel that a descriptor is going away. `lastRef` says whether this was the process's
// final reference to that open file description, which is what decides an ofd lock's fate.
function releaseLocksForClosingFd(fd: number): void {
  const path = syscalls.fdPath(fd);
  if (path === null) return;
  const ofdId = syscalls.fdDescriptionId(fd);
  Atomics.store(myMailbox, MBOX_READY, 0);
  (self as any).postMessage({
    type: 'lock_close', path, ofdId, lastRef: syscalls.countFdsForDescription(ofdId) <= 1,
  });
  awaitMailbox('lock_close');
}

// flock(fd, op): LOCK_SH=1, LOCK_EX=2, LOCK_UN=8, or'd with LOCK_NB=4 for the non-blocking form.
// the lock covers the whole file and belongs to the open file description, so dup'd and inherited
// descriptors share one lock while two separate opens of the same file contend.
function handleFlock(cpu: CpuState): void {
  const fd = Number(cpu.getReg(0));
  const op = Number(cpu.getReg(1)) | 0;
  const path = syscalls.fdPath(fd);
  if (path === null) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-9))); return; } // EBADF

  const nonblock = (op & 4) !== 0;
  const base = op & ~4;
  let ltype: number;
  if (base === 1) ltype = 0;        // LOCK_SH -> read
  else if (base === 2) ltype = 1;   // LOCK_EX -> write
  else if (base === 8) ltype = 2;   // LOCK_UN
  else { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; } // EINVAL

  // reuse the record-lock transport: cmd picks blocking vs not, and the whole file is [0, EOF).
  const cmd = ltype === 2 ? 6 : nonblock ? 6 : 7;
  usedLocks = true;
  Atomics.store(myMailbox, MBOX_READY, 0);
  (self as any).postMessage({
    type: 'lock_req', space: 'flock', scope: 'ofd', ownerId: syscalls.fdDescriptionId(fd),
    path, cmd, ltype, start: 0, len: 0,
  });
  const r = waitLockReply();
  // flock reports contention as EWOULDBLOCK, which is EAGAIN's value; leave the errno as-is.
  cpu.setReg(0, r < 0 ? BigInt.asUintN(64, BigInt(r)) : 0n);
}

// ── Kernel pipe dup/close ────────────────────────────────────────────────────

/**
 * dup(2) / fcntl(F_DUPFD) of a kernel pipe end.
 *
 * The copy is a second reference to the same ring, so the ring's holder count has to rise with it
 * -- otherwise closing the original drops the ring to zero holders while a live descriptor still
 * names it, and the readers on the other side see EOF while a writer is still there.
 */
function handleKernelPipeDup(cpu: CpuState, minFd: number, cloexec: boolean): void {
  const oldFd = Number(cpu.getReg(0));
  const info = kernelPipes.get(oldFd)!;
  const newFd = syscalls.allocLowestFd(minFd);
  if (newFd < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-24))); return; } // EMFILE
  kernelPipes.set(newFd, { ...info });
  pipeRef(info.pipeOff, info.mode, +1);
  // the syscall handler keeps its own view of the fd so fcntl, fstat and the rest still work.
  // a dup SHARES the file description, so the status flags travel with it; FD_CLOEXEC does not.
  syscalls.registerFd(newFd, {
    type: info.mode === 'read' ? 'pipe_read' : 'pipe_write',
    pipe: { data: [], writeClosed: false, readClosed: false },
  }, syscalls.statusFlags(oldFd));
  if (cloexec) syscalls.markCloexec(newFd);
  cpu.setReg(0, BigInt(newFd));
}

function handleKernelPipeDup3(cpu: CpuState): void {
  const oldFd = Number(cpu.getReg(0));
  const newFd = Number(cpu.getReg(1));
  // dup3's only flag is O_CLOEXEC, and it is checked before the descriptors are.
  if (Number(BigInt.asIntN(32, cpu.getReg(2))) & ~0o2000000) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; // EINVAL
  }
  if (oldFd === newFd) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
  const pipeInfo = kernelPipes.get(oldFd)!;

  // If newFd already has a kernel pipe, close it first
  const existing = kernelPipes.get(newFd);
  if (existing) {
    pipeRef(existing.pipeOff, existing.mode, -1);
    kernelPipes.delete(newFd);
  }

  // Duplicate: newFd now points to the same pipe ring, increment refcount
  kernelPipes.set(newFd, { ...pipeInfo });
  pipeRef(pipeInfo.pipeOff, pipeInfo.mode, +1);

  // Also let the SyscallHandler know (it manages fd table for other ops like fcntl)
  syscalls.registerFd(newFd, {
    type: pipeInfo.mode === 'read' ? 'pipe_read' : 'pipe_write',
    pipe: { data: [], writeClosed: false, readClosed: false },
  }, syscalls.statusFlags(oldFd));
  // dup3's flag applies to the NEW descriptor. dup(2) and dup2(2) never set it, so a redundant
  // clear here is what makes "dup2 then dup3 with the flag" and the reverse both come out right.
  if (Number(BigInt.asIntN(32, cpu.getReg(2))) & 0o2000000) syscalls.markCloexec(newFd);
  else syscalls.clearCloexec(newFd);

  cpu.setReg(0, BigInt(newFd));
}

// ── Kernel pipe operations ──────────────────────────────────────────────────

function allocPipeSlot(): number {
  // Ask kernel for a pipe slot via mailbox
  Atomics.store(myMailbox, MBOX_READY, 0);
  (self as any).postMessage({ type: 'pipe_alloc', priv: syscalls.hasCap(24) }); // CAP_SYS_RESOURCE
  awaitMailbox('pipe_alloc');
  return Atomics.load(myMailbox, MBOX_VALUE0);
}

function handlePipe2(cpu: CpuState, view: DataView): void {
  const pipefdAddr = Number(cpu.getReg(0));
  // the pair is copied out to user memory at the end, but a wild pipefd has to be caught HERE:
  // with the write unguarded, pipe2(NULL) wrote two descriptors over whatever lives at guest
  // address 0 and reported success.
  if (syscalls.ptrFaults(new Uint8Array(memory.buffer), pipefdAddr, 8)) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; // EFAULT
  }
  // pipe2 takes three flags and nothing else. they were dropped on the floor except O_NONBLOCK,
  // so a caller that asked for close-on-exec got a descriptor its exec kept, and one that asked
  // for packet mode was told it had it when F_GETFL said otherwise.
  const O_DIRECT = 0o200000, O_NONBLOCK = 0o4000, O_CLOEXEC = 0o2000000;
  const p2Flags = Number(BigInt.asIntN(32, cpu.getReg(1)));
  if (p2Flags & ~(O_DIRECT | O_NONBLOCK | O_CLOEXEC)) {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; // EINVAL
  }
  // the caller's own descriptor limit is checked first, and it is a DIFFERENT answer from running
  // the system out of pipe buffers: EMFILE means "you are holding too many", ENFILE means "the
  // machine has none left". a caller that gets the wrong one looks in the wrong place.
  if (!syscalls.fdRoomFor(2)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-24))); return; } // EMFILE
  const pipeOff = allocPipeSlot();
  if (pipeOff < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-23))); return; } // ENFILE: arena exhausted

  // Initialize the ring buffer (the kernel already claimed+reset it; this is idempotent)
  const ring = new PipeRing(shmArena, pipeOff);
  ring.reset();

  // descriptors come from the same allocator as every other kind, lowest-free-first and bounded
  // by RLIMIT_NOFILE. a private monotonic counter handed out numbers that climbed past the limit
  // and never reused a closed one -- linux does neither. each is registered as it is taken so the
  // second allocation cannot pick the same number as the first.
  const readFd = syscalls.allocLowestFd();
  if (readFd < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-24))); return; } // EMFILE
  // O_CLOEXEC is a DESCRIPTOR flag, not a status flag, so it is recorded separately below.
  const pipeStatus = p2Flags & (O_NONBLOCK | O_DIRECT);
  syscalls.registerFd(readFd, { type: 'pipe_read', pipe: { data: [], writeClosed: false, readClosed: false } }, pipeStatus);
  const writeFd = syscalls.allocLowestFd();
  if (writeFd < 0) {
    syscalls.unregisterFd(readFd);
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-24))); return; // EMFILE
  }
  syscalls.registerFd(writeFd, { type: 'pipe_write', pipe: { data: [], writeClosed: false, readClosed: false } }, pipeStatus);

  if (p2Flags & O_CLOEXEC) { syscalls.markCloexec(readFd); syscalls.markCloexec(writeFd); }

  kernelPipes.set(readFd, { pipeOff, mode: 'read' });
  kernelPipes.set(writeFd, { pipeOff, mode: 'write' });

  // Write [readFd, writeFd] to guest memory
  view.setInt32(pipefdAddr + GUEST_MEM_BASE, readFd, true);
  view.setInt32(pipefdAddr + GUEST_MEM_BASE + 4, writeFd, true);

  cpu.setReg(0, 0n);
}

/**
 * Take or drop a reference to a pipe end, and tell the kernel.
 *
 * The refcount in the ring is what readers and writers actually consult, but only the kernel
 * outlives a process. It keeps a per-process tally so that when a process dies WITHOUT running its
 * own close path -- killed, crashed, reaped as an orphan -- it can still drop what that process was
 * holding. Otherwise the ring keeps a writer that no longer exists and every reader waits forever
 * for an EOF that can never come.
 *
 * Fire-and-forget: no round trip, and message order from one worker is FIFO, so the kernel always
 * sees these before the exit that follows them.
 */
/** a write landed: if anyone armed O_ASYNC on the far end, ask the kernel to signal them.
 *  the flag lives in the ring header so the common case -- nobody listening -- costs one load. */
function notePipeReadable(off: number): void {
  const hdr = new Int32Array(shmArena, off, PIPE_HEADER_SIZE / 4);
  if (Atomics.load(hdr, PIPE_ASYNC_W) === 0) return;
  (self as any).postMessage({ type: 'pipe_io', off, mode: 'read' });
}

function pipeRef(off: number, mode: 'read' | 'write', delta: number): void {
  const ring = new PipeRing(shmArena, off);
  if (delta > 0) { if (mode === 'write') ring.addWriter(); else ring.addReader(); }
  else { if (mode === 'write') ring.closeWriter(); else ring.closeReader(); }
  (self as any).postMessage({ type: 'pipe_ref', off, mode, delta });
}


// ── ptrace ──────────────────────────────────────────────────────────────────────────────────────
//
// a tracer cannot reach into its tracee: every process has its own wasm memory, and the kernel --
// which is the main thread -- holds none of them. so a ptrace-stopped tracee does not simply sit
// there. it parks in a service loop and answers its tracer's peeks, pokes and register reads out
// of the exchange slot it shares with everyone. the tracer writes a request, bumps REQ, wakes the
// tracee's wait futex, and waits for ACK to catch up.
const PTRACE_TRACEME = 0, PTRACE_PEEKTEXT = 1, PTRACE_PEEKDATA = 2, PTRACE_PEEKUSER = 3,
      PTRACE_POKETEXT = 4, PTRACE_POKEDATA = 5, PTRACE_POKEUSER = 6, PTRACE_CONT = 7,
      PTRACE_KILL = 8, PTRACE_SINGLESTEP = 9, PTRACE_ATTACH = 16, PTRACE_DETACH = 17,
      PTRACE_SETOPTIONS = 0x4200, PTRACE_GETREGSET = 0x4204, PTRACE_SETREGSET = 0x4205;
// what the service loop is being asked for.
const PT_OP_PEEK = 1, PT_OP_POKE = 2, PT_OP_GETREGS = 3, PT_OP_SETREGS = 4;
// struct user_pt_regs: x0..x30, sp, pc, pstate.
const PTRACE_REGS_BYTES = 34 * 8;

/** the slot a live pid occupies, or -1. */
function traceSlotOfPid(pid: number): number {
  for (let s = 0; s < PROC_TABLE_SLOTS; s++) {
    const b = procSlotOffset(s) / 4;
    if (Atomics.load(procTable, b + PROC_PID) === pid &&
        Atomics.load(procTable, b + PROC_STATE) !== PROC_EMPTY) return s;
  }
  return -1;
}

/** ask a stopped tracee to do something in its own memory, and wait for the answer. */
function ptraceAsk(slot: number, op: number, addr: number, len: number, data: bigint):
    { status: number; data: bigint } {
  const w = ptraceSlotWord(slot);
  const seq = (Atomics.load(procTable, w + PT_REQ) + 1) | 0;
  Atomics.store(procTable, w + PT_OP, op);
  Atomics.store(procTable, w + PT_ADDR, addr | 0);
  Atomics.store(procTable, w + PT_LEN, len | 0);
  Atomics.store(procTable, w + PT_DATA_LO, Number(data & 0xffffffffn) | 0);
  Atomics.store(procTable, w + PT_DATA_HI, Number((data >> 32n) & 0xffffffffn) | 0);
  Atomics.store(procTable, w + PT_STATUS, 0);
  Atomics.store(procTable, w + PT_REQ, seq);
  // the tracee is parked on its wait futex; that is the door to knock on.
  const tb = procSlotOffset(slot) / 4;
  Atomics.store(procTable, tb + PROC_WAIT_FUTEX, 1);
  Atomics.notify(procTable, tb + PROC_WAIT_FUTEX);
  const deadline = performance.now() + 2000;
  while (Atomics.load(procTable, w + PT_ACK) !== seq) {
    if (performance.now() > deadline) return { status: -3, data: 0n }; // ESRCH: it never answered
    if (Atomics.load(procTable, tb + PROC_PID) <= 0) return { status: -3, data: 0n };
    sleepingWait(procTable, w + PT_ACK, Atomics.load(procTable, w + PT_ACK), 5);
  }
  const lo = BigInt(Atomics.load(procTable, w + PT_DATA_LO) >>> 0);
  const hi = BigInt(Atomics.load(procTable, w + PT_DATA_HI) >>> 0);
  return { status: Atomics.load(procTable, w + PT_STATUS), data: (hi << 32n) | lo };
}

/** answer one request while stopped. returns true if there was one. */
function ptraceServe(): boolean {
  const w = ptraceSlotWord(mySlot);
  const seq = Atomics.load(procTable, w + PT_REQ);
  if (seq === Atomics.load(procTable, w + PT_ACK)) return false;
  const op = Atomics.load(procTable, w + PT_OP);
  const addr = Atomics.load(procTable, w + PT_ADDR) >>> 0;
  const len = Atomics.load(procTable, w + PT_LEN);
  const bytes = new Uint8Array(memory.buffer);
  const view = new DataView(memory.buffer);
  let status = 0;
  let out = 0n;
  try {
    if (op === PT_OP_PEEK) {
      // ptrace reads a word at a time, and an address it cannot reach is EIO -- not a fault.
      if (syscalls.ptrFaults(bytes, addr, 8)) status = -5;
      else out = view.getBigUint64(addr + GUEST_MEM_BASE, true);
    } else if (op === PT_OP_POKE) {
      const lo = BigInt(Atomics.load(procTable, w + PT_DATA_LO) >>> 0);
      const hi = BigInt(Atomics.load(procTable, w + PT_DATA_HI) >>> 0);
      if (syscalls.ptrWriteFaults(bytes, addr, 8)) status = -5;
      else view.setBigUint64(addr + GUEST_MEM_BASE, (hi << 32n) | lo, true);
    } else if (op === PT_OP_GETREGS) {
      const sc = ptraceScratchOffset(mySlot);
      const sab = new DataView(procTable.buffer);
      for (let i = 0; i < 31; i++) sab.setBigUint64(sc + i * 8, cpu.getReg(i), true);
      sab.setBigUint64(sc + 31 * 8, cpu.getSP(), true);
      sab.setBigUint64(sc + 32 * 8, cpu.getPC(), true);
      sab.setBigUint64(sc + 33 * 8, cpu.getNZCV(), true);
      out = BigInt(PTRACE_REGS_BYTES);
    } else if (op === PT_OP_SETREGS) {
      const sc = ptraceScratchOffset(mySlot);
      const sab = new DataView(procTable.buffer);
      const n = Math.min(len, PTRACE_REGS_BYTES);
      for (let i = 0; i < 31 && (i + 1) * 8 <= n; i++) cpu.setReg(i, sab.getBigUint64(sc + i * 8, true));
      if (n >= 32 * 8) cpu.setSP(sab.getBigUint64(sc + 31 * 8, true));
      if (n >= 33 * 8) cpu.setPC(sab.getBigUint64(sc + 32 * 8, true));
      out = BigInt(n);
    } else {
      status = -22;
    }
  } catch {
    status = -5;
  }
  Atomics.store(procTable, w + PT_DATA_LO, Number(out & 0xffffffffn) | 0);
  Atomics.store(procTable, w + PT_DATA_HI, Number((out >> 32n) & 0xffffffffn) | 0);
  Atomics.store(procTable, w + PT_STATUS, status);
  Atomics.store(procTable, w + PT_ACK, seq);
  Atomics.notify(procTable, w + PT_ACK);
  return true;
}

/**
 * stop for the tracer, and stay useful while stopped.
 *
 * returns the signal to deliver when the tracer lets go: the one it was stopped for, whatever the
 * tracer swapped in, or 0 to swallow it -- which is what makes a debugger able to hide a signal
 * from the program it is tracing.
 */
function ptraceStop(sig: number): number {
  const base = procSlotOffset(mySlot) / 4;
  Atomics.store(procTable, base + PROC_TRACE_CMD, PT_RESUME_NONE);
  Atomics.store(procTable, base + PROC_STOP_SIG, sig);
  Atomics.store(procTable, base + PROC_STOP_SEEN, 0);
  Atomics.store(procTable, base + PROC_STATE, PROC_STOPPED);
  // the TRACER is the one waiting on this stop, and it is not always the parent -- and even when
  // it is, waking "the parent of the tracer" is a different process entirely.
  wakeWatcher(Atomics.load(procTable, base + PROC_TRACER));
  for (;;) {
    while (ptraceServe()) { /* answer everything queued before parking again */ }
    const cmd = Atomics.load(procTable, base + PROC_TRACE_CMD);
    if (cmd !== PT_RESUME_NONE) break;
    if (Atomics.load(procTable, base + PROC_PID) <= 0) return 0;   // killed while stopped
    Atomics.store(procTable, base + PROC_WAIT_FUTEX, 0);
    if (Atomics.load(procTable, base + PROC_TRACE_CMD) !== PT_RESUME_NONE) break;
    sleepingWait(procTable, base + PROC_WAIT_FUTEX, 0, 200);
  }
  Atomics.store(procTable, base + PROC_TRACE_CMD, PT_RESUME_NONE);
  Atomics.store(procTable, base + PROC_STATE, PROC_RUNNING);
  const inject = Atomics.load(procTable, base + PROC_STOP_SIG);
  Atomics.store(procTable, base + PROC_STOP_SIG, 0);
  return inject;
}

/** ptrace(request, pid, addr, data). */
function handlePtrace(cpu2: CpuState, view: DataView): void {
  const req = Number(BigInt.asIntN(32, cpu2.getReg(0)));
  const pid = Number(BigInt.asIntN(32, cpu2.getReg(1)));
  const addr = Number(BigInt.asUintN(64, cpu2.getReg(2)));
  const data = cpu2.getReg(3);
  const err = (e: number) => cpu2.setReg(0, BigInt.asUintN(64, BigInt(e)));
  const myBase = procSlotOffset(mySlot) / 4;

  if (req === PTRACE_TRACEME) {
    if (Atomics.load(procTable, myBase + PROC_TRACER) !== 0) { err(-1); return; }  // EPERM
    Atomics.store(procTable, myBase + PROC_TRACER, Atomics.load(procTable, myBase + PROC_PPID));
    cpu2.setReg(0, 0n);
    return;
  }

  const slot = traceSlotOfPid(pid);
  if (slot < 0) { err(-3); return; }                                   // ESRCH
  const base = procSlotOffset(slot) / 4;

  if (req === PTRACE_ATTACH) {
    // a process may not trace itself, nor take one that is already traced.
    if (pid === myPid) { err(-1); return; }
    if (Atomics.load(procTable, base + PROC_TRACER) !== 0) { err(-1); return; }
    // and it may only trace what it could signal: linux wants the tracer's real uid to match ALL
    // THREE of the target's, or CAP_SYS_PTRACE. without this any user could attach to root's
    // processes and read their memory a word at a time.
    const meCred = procCredOffset(mySlot) / 4;
    const myEuid2 = Atomics.load(procCreds, meCred + 1);
    if (myEuid2 !== 0) {
      const tb = procCredOffset(slot) / 4;
      const myRuid2 = Atomics.load(procCreds, meCred);
      const tRuid = Atomics.load(procCreds, tb);
      const tEuid = Atomics.load(procCreds, tb + 1);
      const tSuid = Atomics.load(procCreds, tb + 2);
      if (myRuid2 !== tRuid || myRuid2 !== tEuid || myRuid2 !== tSuid) { err(-1); return; }
    }
    Atomics.store(procTable, base + PROC_TRACER, myPid);
    sigqPush(sigQueue, slot, { signo: 19 /* SIGSTOP */, code: 0, pid: myPid, uid: syscalls.currentEuid(), value: 0n });
    Atomics.store(procTable, base + PROC_WAIT_FUTEX, 1);
    Atomics.notify(procTable, base + PROC_WAIT_FUTEX);
    bumpIoGeneration();
    cpu2.setReg(0, 0n);
    return;
  }

  // everything else is between a tracer and ITS tracee.
  if (Atomics.load(procTable, base + PROC_TRACER) !== myPid) { err(-3); return; }  // ESRCH

  if (req === PTRACE_KILL) {
    (self as any).postMessage({ type: 'kill', pid, sig: 9 });
    cpu2.setReg(0, 0n);
    return;
  }
  if (req === PTRACE_SETOPTIONS) { cpu2.setReg(0, 0n); return; }

  // the rest need the tracee stopped: linux answers ESRCH when it is not.
  if (Atomics.load(procTable, base + PROC_STATE) !== PROC_STOPPED) { err(-3); return; }

  switch (req) {
    case PTRACE_CONT:
    case PTRACE_DETACH: {
      const sig = Number(BigInt.asIntN(32, data));
      if (sig < 0 || sig > 64) { err(-22); return; }
      if (req === PTRACE_DETACH) Atomics.store(procTable, base + PROC_TRACER, 0);
      Atomics.store(procTable, base + PROC_STOP_SIG, sig);
      Atomics.store(procTable, base + PROC_TRACE_CMD,
                    req === PTRACE_DETACH ? PT_RESUME_DETACH : PT_RESUME_CONT);
      Atomics.store(procTable, base + PROC_WAIT_FUTEX, 1);
      Atomics.notify(procTable, base + PROC_WAIT_FUTEX);
      cpu2.setReg(0, 0n);
      return;
    }
    case PTRACE_PEEKTEXT:
    case PTRACE_PEEKDATA: {
      const r = ptraceAsk(slot, PT_OP_PEEK, addr, 8, 0n);
      if (r.status < 0) { err(r.status); return; }
      // the raw syscall stores the word where `data` points and returns 0; only the libc wrapper
      // returns it in the register.
      const out = Number(data);
      if (out === 0 || syscalls.ptrFaults(new Uint8Array(memory.buffer), out, 8)) { err(-14); return; }
      view.setBigUint64(out + GUEST_MEM_BASE, r.data, true);
      cpu2.setReg(0, 0n);
      return;
    }
    case PTRACE_POKETEXT:
    case PTRACE_POKEDATA: {
      const r = ptraceAsk(slot, PT_OP_POKE, addr, 8, BigInt.asUintN(64, data));
      if (r.status < 0) { err(r.status); return; }
      cpu2.setReg(0, 0n);
      return;
    }
    case PTRACE_PEEKUSER:
    case PTRACE_POKEUSER: {
      // the user area starts with struct user_pt_regs, and nothing past it is readable here.
      if (addr % 8 !== 0 || addr >= PTRACE_REGS_BYTES) { err(-5); return; }  // EIO
      const g = ptraceAsk(slot, PT_OP_GETREGS, 0, PTRACE_REGS_BYTES, 0n);
      if (g.status < 0) { err(g.status); return; }
      const sc = ptraceScratchOffset(slot);
      const sab = new DataView(procTable.buffer);
      if (req === PTRACE_PEEKUSER) {
        const word = sab.getBigUint64(sc + addr, true);
        const out = Number(data);
        if (out === 0 || syscalls.ptrFaults(new Uint8Array(memory.buffer), out, 8)) { err(-14); return; }
        view.setBigUint64(out + GUEST_MEM_BASE, word, true);
      } else {
        sab.setBigUint64(sc + addr, BigInt.asUintN(64, data), true);
        const r = ptraceAsk(slot, PT_OP_SETREGS, 0, PTRACE_REGS_BYTES, 0n);
        if (r.status < 0) { err(r.status); return; }
      }
      cpu2.setReg(0, 0n);
      return;
    }
    case PTRACE_GETREGSET:
    case PTRACE_SETREGSET: {
      const NT_PRSTATUS = 1;
      if (addr !== NT_PRSTATUS) { err(-22); return; }
      const iov = Number(data);
      const bytes = new Uint8Array(memory.buffer);
      if (iov === 0 || syscalls.ptrFaults(bytes, iov, 16)) { err(-14); return; }
      const bufPtr = Number(view.getBigUint64(iov + GUEST_MEM_BASE, true));
      const bufLen = Number(view.getBigUint64(iov + GUEST_MEM_BASE + 8, true));
      const n = Math.min(bufLen, PTRACE_REGS_BYTES);
      if (n <= 0) { err(-22); return; }
      const sc = ptraceScratchOffset(slot);
      const sabBytes = new Uint8Array(procTable.buffer);
      if (req === PTRACE_GETREGSET) {
        const g = ptraceAsk(slot, PT_OP_GETREGS, 0, PTRACE_REGS_BYTES, 0n);
        if (g.status < 0) { err(g.status); return; }
        if (syscalls.ptrWriteFaults(bytes, bufPtr, n)) { err(-14); return; }
        bytes.set(sabBytes.subarray(sc, sc + n), bufPtr + GUEST_MEM_BASE);
      } else {
        if (syscalls.ptrFaults(bytes, bufPtr, n)) { err(-14); return; }
        sabBytes.set(bytes.subarray(bufPtr + GUEST_MEM_BASE, bufPtr + GUEST_MEM_BASE + n), sc);
        const r = ptraceAsk(slot, PT_OP_SETREGS, 0, n, 0n);
        if (r.status < 0) { err(r.status); return; }
      }
      view.setBigUint64(iov + GUEST_MEM_BASE + 8, BigInt(n), true);
      cpu2.setReg(0, 0n);
      return;
    }
    default:
      err(-5); // EIO: a request this kernel does not carry
  }
}

/**
 * rt_sigtimedwait(set, info, timeout, sigsetsize).
 *
 * Blocks until one of `set` is pending for us, then CONSUMES it (no handler runs) and reports it.
 * A signal outside the set that can be delivered interrupts the wait with EINTR, which is what lets
 * the run loop take it on the way out. Timeout expiry is EAGAIN. Programs block the signals they
 * wait for first, which is why a blocked signal still has to accumulate in the pending queue.
 */
function handleSigtimedwait(cpu: CpuState, view: DataView): void {
  const setPtr = Number(cpu.getReg(0));
  const infoPtr = Number(cpu.getReg(1));
  const tsPtr = Number(cpu.getReg(2));
  const bytes = new Uint8Array(memory.buffer);
  // the size argument is the sigset's, and a caller that gets it wrong was compiled against a
  // different kernel. it is checked before anything else, ahead of even the pointer.
  if (Number(BigInt.asIntN(64, cpu.getReg(3))) !== 8) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
  if (setPtr === 0 || syscalls.ptrFaults(bytes, setPtr, 8)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; }
  // SIGKILL and SIGSTOP cannot be waited for any more than they can be blocked.
  const want = view.getBigUint64(setPtr + GUEST_MEM_BASE, true) & ~((1n << 8n) | (1n << 18n));

  let deadline = Infinity;
  if (tsPtr !== 0) {
    if (syscalls.ptrFaults(bytes, tsPtr, 16)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; }
    const sec = Number(view.getBigInt64(tsPtr + GUEST_MEM_BASE, true));
    const nsec = Number(view.getBigInt64(tsPtr + GUEST_MEM_BASE + 8, true));
    if (sec < 0 || nsec < 0 || nsec >= 1_000_000_000) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
    deadline = performance.now() + sec * 1000 + nsec / 1e6;
  }

  const slotBase = procSlotOffset(mySlot) / 4;
  for (;;) {
    pollAndQueueSignals();
    const got = syscalls.takePendingIn(want);
    if (got) {
      // the signal has already been taken out of the queue by now, so a buffer that cannot be
      // written is EFAULT and the signal is gone -- which is exactly what linux does with it.
      if (infoPtr !== 0 && syscalls.ptrWriteFaults(bytes, infoPtr, 128)) {
        cpu.setReg(0, BigInt.asUintN(64, BigInt(-14)));
        return;
      }
      if (infoPtr !== 0) {
        const base = infoPtr + GUEST_MEM_BASE;
        bytes.fill(0, base, base + 128);
        view.setInt32(base + 0, got.sig, true);
        view.setInt32(base + 8, got.pid > 0 ? 0 /* SI_USER */ : 0x80 /* SI_KERNEL */, true);
        if (got.pid > 0) { view.setInt32(base + 16, got.pid, true); view.setInt32(base + 20, got.uid, true); }
      }
      cpu.setReg(0, BigInt(got.sig));
      return;
    }
    // something we are NOT waiting for wants to be delivered: leave, and let the run loop take it
    if (syscalls.hasPendingDeliverable()) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-4))); return; } // EINTR
    const rem = deadline - performance.now();
    if (rem <= 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-11))); return; } // EAGAIN
    Atomics.store(procTable, slotBase + PROC_WAIT_FUTEX, 0);
    sleepingWait(procTable, slotBase + PROC_WAIT_FUTEX, 0, Math.min(rem, 50));
  }
}

/** one kernel round trip for a SysV request; returns the reply words. */

// ── scheduling ─────────────────────────────────────────────────────────────────────────────────
// none of this changes which thread the host runs next -- workers are os threads and the browser
// schedules them. what it does is keep the state linux keeps: a policy, a priority, an affinity
// mask and a deadline reservation, per thread, in the shared table, validated and permission
// checked the way linux validates them, so that a program that sets them reads back what it set,
// including from another process.
const SCHED_OTHER = 0, SCHED_FIFO = 1, SCHED_RR = 2, SCHED_BATCH = 3, SCHED_IDLE = 5, SCHED_DEADLINE = 6;
const SCHED_RESET_ON_FORK = 0x40000000;
const SCHED_ATTR_SIZE_VER0 = 48;
/** how many cpus the guest believes it has. everything that names a cpu agrees with this. */
const GUEST_CPUS = guestCpus();
/** cpumask bytes as linux reports them: whole longs, enough to hold every cpu. */
const CPUMASK_BYTES = Math.ceil(GUEST_CPUS / 64) * 8;
/** SCHED_RR's quantum, matching /proc/sys/kernel/sched_rr_timeslice_ms. */
const RR_TIMESLICE_MS = 100;
/** the slice a fair-policy thread gets, which is what linux reports for SCHED_OTHER and friends. */
const FAIR_SLICE_MS = 3;

function isRtPolicy(p: number): boolean { return p === SCHED_FIFO || p === SCHED_RR; }

function validPolicy(p: number): boolean {
  return p === SCHED_OTHER || p === SCHED_FIFO || p === SCHED_RR ||
         p === SCHED_BATCH || p === SCHED_IDLE || p === SCHED_DEADLINE;
}

/** the table slot a pid occupies, or -1. pid 0 means the calling thread. */
/**
 * which threads an ioprio call is about: one process, every process in a group, or every process of
 * one user. `who` of zero means the caller's own pid, group or user.
 */
function ioprioTargets(which: number, who: number): number[] {
  if (which === 1) { const s = schedSlotOf(who); return s < 0 ? [] : [s]; }
  const meBase = procSlotOffset(mySlot) / 4;
  const want = who !== 0 ? who
             : Atomics.load(procTable, meBase + (which === 2 ? PROC_PGID : PROC_RUID));
  const out: number[] = [];
  for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
    const base = procSlotOffset(slot) / 4;
    if (Atomics.load(procTable, base + PROC_STATE) === PROC_EMPTY) continue;
    if (Atomics.load(procTable, base + (which === 2 ? PROC_PGID : PROC_RUID)) !== want) continue;
    out.push(slot);
  }
  return out;
}

function schedSlotOf(pid: number): number {
  if (pid === 0) return mySlot;
  for (let slot = 0; slot < PROC_TABLE_SLOTS; slot++) {
    const base = procSlotOffset(slot) / 4;
    if (Atomics.load(procTable, base + PROC_PID) !== pid) continue;
    if (Atomics.load(procTable, base + PROC_STATE) === PROC_EMPTY) continue;
    return slot;
  }
  return -1;
}

/**
 * may the caller change this thread's scheduling? linux asks whether the two are the same user
 * (check_same_owner), and lets anyone with CAP_SYS_NICE past regardless.
 */
function schedSameOwner(slot: number): boolean {
  const myBase = procCredOffset(mySlot) / 4;
  const myEuid = Atomics.load(procCreds, myBase + 1);
  if (myEuid === 0) return true;
  const tb = procCredOffset(slot) / 4;
  return myEuid === Atomics.load(procCreds, tb + 1) || myEuid === Atomics.load(procCreds, tb);
}

/** CAP_SYS_NICE, which is what raising a thread onto a real-time policy costs. */
function schedPrivileged(): boolean {
  return Atomics.load(procCreds, (procCredOffset(mySlot) / 4) + 1) === 0;
}

function schedRead(slot: number, field: number): number {
  return Atomics.load(procTable, (procSlotOffset(slot) / 4) + field);
}
function schedWrite(slot: number, field: number, value: number): void {
  Atomics.store(procTable, (procSlotOffset(slot) / 4) + field, value | 0);
}
/** a 64-bit scheduling quantity, stored as a (low, high) pair of words. */
function schedRead64(slot: number, field: number): bigint {
  const base = procSlotOffset(slot) / 4;
  const lo = BigInt(Atomics.load(procTable, base + field) >>> 0);
  const hi = BigInt(Atomics.load(procTable, base + field + 1) >>> 0);
  return (hi << 32n) | lo;
}
function schedWrite64(slot: number, field: number, v: bigint): void {
  const base = procSlotOffset(slot) / 4;
  Atomics.store(procTable, base + field, Number(v & 0xffffffffn) | 0);
  Atomics.store(procTable, base + field + 1, Number((v >> 32n) & 0xffffffffn) | 0);
}

/** the affinity mask of a thread that has never set one: every cpu it could run on. */
function schedAffinity(slot: number): bigint {
  const base = procSlotOffset(slot) / 4;
  const lo = BigInt(Atomics.load(procTable, base + PROC_AFFINITY) >>> 0);
  const hi = BigInt(Atomics.load(procTable, base + PROC_AFFINITY + 1) >>> 0);
  const m = (hi << 32n) | lo;
  return m === 0n ? (GUEST_CPUS >= 64 ? ~0n & ((1n << 64n) - 1n) : (1n << BigInt(GUEST_CPUS)) - 1n) : m;
}

/**
 * sched_setscheduler / sched_setparam / sched_getscheduler / sched_getparam / the priority bounds /
 * affinity / rr quantum / sched_setattr / sched_getattr.
 */
function handleSched(nr: number, cpu: CpuState, view: DataView): void {
  const bytes = new Uint8Array(memory.buffer);
  const err = (e: number) => cpu.setReg(0, BigInt.asUintN(64, BigInt(-e)));
  const EPERM = 1, ESRCH = 3, EFAULT = 14, EINVAL = 22, E2BIG = 7;

  // getcpu: which cpu this thread is on. we do not place threads ourselves -- the browser does --
  // so the honest answer is the one the caller asked for: a thread pinned to one cpu by
  // sched_setaffinity runs there, and the lowest cpu it is allowed on is where an unpinned one is
  // reported. answering zero unconditionally contradicted the affinity the caller had just set.
  if (nr === 168) {
    const cpuPtr = Number(cpu.getReg(0));
    const nodePtr = Number(cpu.getReg(1));
    const bytes168 = new Uint8Array(memory.buffer);
    if (cpuPtr !== 0 && syscalls.ptrWriteFaults(bytes168, cpuPtr, 4)) { err(EFAULT); return; }
    if (nodePtr !== 0 && syscalls.ptrWriteFaults(bytes168, nodePtr, 4)) { err(EFAULT); return; }
    const mask = schedAffinity(mySlot);
    let which = 0;
    while (which < 64 && (mask & (1n << BigInt(which))) === 0n) which++;
    if (which >= 64) which = 0;
    if (cpuPtr !== 0) view.setUint32(cpuPtr + GUEST_MEM_BASE, which, true);
    if (nodePtr !== 0) view.setUint32(nodePtr + GUEST_MEM_BASE, 0, true);
    cpu.setReg(0, 0n);
    return;
  }

  // ioprio_set/ioprio_get: which = 1 process, 2 process group, 3 user; who = 0 means "me". the
  // value packs a class in its top three bits and a level in the rest.
  if (nr === 30 || nr === 31) {
    const which = Number(BigInt.asIntN(32, cpu.getReg(0)));
    const who = Number(BigInt.asIntN(32, cpu.getReg(1)));
    if (which < 1 || which > 3) { err(EINVAL); return; }
    const slots = ioprioTargets(which, who);
    if (slots.length === 0) { err(ESRCH); return; }
    if (nr === 31) {
      // a group or a user can hold several; the answer is the best priority any of them has, and
      // a smaller class (and level within it) is the better one.
      let best = -1;
      for (const s of slots) {
        const v = schedRead(s, PROC_IOPRIO);
        if (best < 0 || v < best) best = v;
      }
      cpu.setReg(0, BigInt(best < 0 ? 0 : best));
      return;
    }
    const value = Number(BigInt.asIntN(32, cpu.getReg(2)));
    const cls = (value >> 13) & 0x7;
    const level = value & 0x1fff;
    // real-time i/o starves every other class, so only a privileged caller may ask for it. the
    // other classes bound the level at eight, and "none" carries no level at all.
    if (cls === 1 && !syscalls.hasCap(21 /* CAP_SYS_ADMIN */)) { err(EPERM); return; }
    if (cls === 1 || cls === 2) { if (level >= 8) { err(EINVAL); return; } }
    else if (cls === 0) { if (level !== 0) { err(EINVAL); return; } }
    else if (cls !== 3) { err(EINVAL); return; }
    for (const s of slots) schedWrite(s, PROC_IOPRIO, value);
    cpu.setReg(0, 0n);
    return;
  }

  if (nr === 125 || nr === 126) { // sched_get_priority_max / sched_get_priority_min
    const policy = Number(BigInt.asIntN(32, cpu.getReg(0)));
    if (!validPolicy(policy)) { err(EINVAL); return; }
    if (!isRtPolicy(policy)) { cpu.setReg(0, 0n); return; }
    cpu.setReg(0, nr === 125 ? 99n : 1n);
    return;
  }

  if (nr === 120) { // sched_getscheduler(pid)
    const pid = Number(BigInt.asIntN(32, cpu.getReg(0)));
    if (pid < 0) { err(EINVAL); return; }
    const slot = schedSlotOf(pid);
    if (slot < 0) { err(ESRCH); return; }
    const reset = (schedRead(slot, PROC_SCHED_FLAGS) & 1) ? SCHED_RESET_ON_FORK : 0;
    cpu.setReg(0, BigInt(schedRead(slot, PROC_POLICY) | reset));
    return;
  }

  if (nr === 121) { // sched_getparam(pid, param)
    const pid = Number(BigInt.asIntN(32, cpu.getReg(0)));
    const ptr = Number(cpu.getReg(1));
    if (pid < 0 || ptr === 0) { err(EINVAL); return; }
    const slot = schedSlotOf(pid);
    if (slot < 0) { err(ESRCH); return; }
    if (syscalls.ptrFaults(bytes, ptr, 4)) { err(EFAULT); return; }
    view.setInt32(ptr + GUEST_MEM_BASE, schedRead(slot, PROC_RTPRIO), true);
    cpu.setReg(0, 0n);
    return;
  }

  if (nr === 118 || nr === 119) { // sched_setparam(pid, param) / sched_setscheduler(pid, policy, param)
    const pid = Number(BigInt.asIntN(32, cpu.getReg(0)));
    const ptr = Number(nr === 118 ? cpu.getReg(1) : cpu.getReg(2));
    if (pid < 0 || ptr === 0) { err(EINVAL); return; }
    if (syscalls.ptrFaults(bytes, ptr, 4)) { err(EFAULT); return; }
    const prio = view.getInt32(ptr + GUEST_MEM_BASE, true);
    const slot = schedSlotOf(pid);
    if (slot < 0) { err(ESRCH); return; }

    let policy: number, resetOnFork: number;
    if (nr === 119) {
      const raw = Number(BigInt.asIntN(32, cpu.getReg(1)));
      resetOnFork = (raw & SCHED_RESET_ON_FORK) ? 1 : 0;
      policy = raw & ~SCHED_RESET_ON_FORK;
      if (!validPolicy(policy)) { err(EINVAL); return; }
      // SCHED_DEADLINE carries a reservation that a struct sched_param cannot express, so it can
      // only be entered through sched_setattr.
      if (policy === SCHED_DEADLINE) { err(EINVAL); return; }
    } else {
      policy = schedRead(slot, PROC_POLICY);
      resetOnFork = schedRead(slot, PROC_SCHED_FLAGS) & 1;
      if (policy === SCHED_DEADLINE) { err(EINVAL); return; }
    }
    // a real-time priority is 1..99 and belongs only to a real-time policy; every other policy
    // takes exactly 0, and says so rather than quietly ignoring the number.
    if (isRtPolicy(policy) ? (prio < 1 || prio > 99) : prio !== 0) { err(EINVAL); return; }
    if (!schedSameOwner(slot)) { err(EPERM); return; }
    // moving onto a real-time policy, or raising a real-time priority, needs CAP_SYS_NICE --
    // RLIMIT_RTPRIO is 0 by default, so an ordinary user has no real-time budget at all.
    if (isRtPolicy(policy) && !schedPrivileged()) { err(EPERM); return; }

    schedWrite(slot, PROC_POLICY, policy);
    schedWrite(slot, PROC_RTPRIO, prio);
    schedWrite(slot, PROC_SCHED_FLAGS, resetOnFork);
    if (policy !== SCHED_DEADLINE) {
      schedWrite64(slot, PROC_DL_RUNTIME, 0n);
      schedWrite64(slot, PROC_DL_DEADLINE, 0n);
      schedWrite64(slot, PROC_DL_PERIOD, 0n);
    }
    cpu.setReg(0, 0n);
    return;
  }

  if (nr === 122 || nr === 123) { // sched_setaffinity / sched_getaffinity
    const pid = Number(BigInt.asIntN(32, cpu.getReg(0)));
    const len = Number(cpu.getReg(1));
    const ptr = Number(cpu.getReg(2));
    if (pid < 0) { err(EINVAL); return; }
    if (len % 8 !== 0) { err(EINVAL); return; }
    if (nr === 123 && len * 8 < GUEST_CPUS) { err(EINVAL); return; }
    if (nr === 123 && len === 0) { err(EINVAL); return; }

    if (nr === 122) {
      const take = Math.min(len, CPUMASK_BYTES);
      if (syscalls.ptrFaults(bytes, ptr, take)) { err(EFAULT); return; }
      let mask = 0n;
      for (let i = 0; i < take; i++) mask |= BigInt(bytes[ptr + GUEST_MEM_BASE + i]) << BigInt(i * 8);
      const slot = schedSlotOf(pid);
      if (slot < 0) { err(ESRCH); return; }
      if (!schedSameOwner(slot)) { err(EPERM); return; }
      // bits past the last cpu name nothing. a mask that names only those names no cpu at all,
      // which is not a request the kernel can honour.
      const allowed = GUEST_CPUS >= 64 ? (1n << 64n) - 1n : (1n << BigInt(GUEST_CPUS)) - 1n;
      const eff = mask & allowed;
      if (eff === 0n) { err(EINVAL); return; }
      schedWrite(slot, PROC_AFFINITY, Number(eff & 0xffffffffn));
      schedWrite(slot, PROC_AFFINITY + 1, Number((eff >> 32n) & 0xffffffffn));
      cpu.setReg(0, 0n);
      return;
    }

    const slot = schedSlotOf(pid);
    if (slot < 0) { err(ESRCH); return; }
    const out = Math.min(len, CPUMASK_BYTES);
    if (syscalls.ptrFaults(bytes, ptr, out)) { err(EFAULT); return; }
    const mask = schedAffinity(slot);
    for (let i = 0; i < out; i++) {
      bytes[ptr + GUEST_MEM_BASE + i] = Number((mask >> BigInt(i * 8)) & 0xffn);
    }
    cpu.setReg(0, BigInt(out));
    return;
  }

  if (nr === 127) { // sched_rr_get_interval(pid, tp)
    const pid = Number(BigInt.asIntN(32, cpu.getReg(0)));
    const ptr = Number(cpu.getReg(1));
    if (pid < 0) { err(EINVAL); return; }
    const slot = schedSlotOf(pid);
    if (slot < 0) { err(ESRCH); return; }
    if (ptr === 0 || syscalls.ptrFaults(bytes, ptr, 16)) { err(EFAULT); return; }
    const policy = schedRead(slot, PROC_POLICY);
    // only SCHED_RR round-robins, so only SCHED_RR has a quantum. a SCHED_FIFO thread runs until
    // it yields or blocks, and reports zero to say exactly that.
    const ms = policy === SCHED_RR ? RR_TIMESLICE_MS
             : (policy === SCHED_FIFO || policy === SCHED_DEADLINE) ? 0 : FAIR_SLICE_MS;
    view.setBigInt64(ptr + GUEST_MEM_BASE, BigInt(Math.floor(ms / 1000)), true);
    view.setBigInt64(ptr + GUEST_MEM_BASE + 8, BigInt((ms % 1000) * 1000000), true);
    cpu.setReg(0, 0n);
    return;
  }

  if (nr === 274) { // sched_setattr(pid, attr, flags)
    const pid = Number(BigInt.asIntN(32, cpu.getReg(0)));
    const ptr = Number(cpu.getReg(1));
    const flags = Number(cpu.getReg(2));
    if (flags !== 0) { err(EINVAL); return; }
    if (ptr === 0 || pid < 0) { err(EINVAL); return; }
    if (syscalls.ptrFaults(bytes, ptr, 4)) { err(EFAULT); return; }
    const size = view.getUint32(ptr + GUEST_MEM_BASE, true);
    // a struct too small to hold the fields the kernel needs is answered with the size it wanted.
    if (size < SCHED_ATTR_SIZE_VER0 || size > 4096) {
      if (!syscalls.ptrFaults(bytes, ptr, 4)) view.setUint32(ptr + GUEST_MEM_BASE, SCHED_ATTR_SIZE_VER0, true);
      err(E2BIG); return;
    }
    if (syscalls.ptrFaults(bytes, ptr, SCHED_ATTR_SIZE_VER0)) { err(EFAULT); return; }
    const base = ptr + GUEST_MEM_BASE;
    const rawPolicy = view.getUint32(base + 4, true);
    const attrFlags = view.getBigUint64(base + 8, true);
    const nice = view.getInt32(base + 16, true);
    const prio = view.getUint32(base + 20, true);
    const runtime = view.getBigUint64(base + 24, true);
    const deadline = view.getBigUint64(base + 32, true);
    const period = view.getBigUint64(base + 40, true);

    const resetOnFork = (attrFlags & 1n) ? 1 : 0;
    const policy = rawPolicy & ~SCHED_RESET_ON_FORK;
    if (!validPolicy(policy)) { err(EINVAL); return; }
    const slot = schedSlotOf(pid);
    if (slot < 0) { err(ESRCH); return; }
    if (isRtPolicy(policy) ? (prio < 1 || prio > 99) : prio !== 0) { err(EINVAL); return; }
    if (policy === SCHED_DEADLINE) {
      // the reservation has to be one a scheduler could actually meet: a slice of work that fits
      // inside its deadline, which in turn fits inside the period it repeats over.
      const eff = period === 0n ? deadline : period;
      if (runtime === 0n || deadline === 0n || runtime > deadline || deadline > eff) { err(EINVAL); return; }
    }
    if (!schedSameOwner(slot)) { err(EPERM); return; }
    if ((isRtPolicy(policy) || policy === SCHED_DEADLINE) && !schedPrivileged()) { err(EPERM); return; }

    schedWrite(slot, PROC_POLICY, policy);
    schedWrite(slot, PROC_RTPRIO, prio);
    schedWrite(slot, PROC_SCHED_FLAGS, resetOnFork);
    if (policy === SCHED_OTHER || policy === SCHED_BATCH) {
      schedWrite(slot, PROC_NICE, Math.max(-20, Math.min(19, nice)));
    }
    schedWrite64(slot, PROC_DL_RUNTIME, policy === SCHED_DEADLINE ? runtime : 0n);
    schedWrite64(slot, PROC_DL_DEADLINE, policy === SCHED_DEADLINE ? deadline : 0n);
    schedWrite64(slot, PROC_DL_PERIOD, policy === SCHED_DEADLINE ? (period === 0n ? deadline : period) : 0n);
    cpu.setReg(0, 0n);
    return;
  }

  // sched_getattr(pid, attr, size, flags)
  const pid = Number(BigInt.asIntN(32, cpu.getReg(0)));
  const ptr = Number(cpu.getReg(1));
  const size = Number(cpu.getReg(2));
  const flags = Number(cpu.getReg(3));
  if (ptr === 0 || size < SCHED_ATTR_SIZE_VER0 || size > 4096 || flags !== 0 || pid < 0) { err(EINVAL); return; }
  const slot = schedSlotOf(pid);
  if (slot < 0) { err(ESRCH); return; }
  const out = Math.min(size, SCHED_ATTR_SIZE_VER0);
  if (syscalls.ptrFaults(bytes, ptr, out)) { err(EFAULT); return; }
  const base = ptr + GUEST_MEM_BASE;
  view.setUint32(base, out, true);
  view.setUint32(base + 4, schedRead(slot, PROC_POLICY), true);
  view.setBigUint64(base + 8, BigInt(schedRead(slot, PROC_SCHED_FLAGS) & 1), true);
  view.setInt32(base + 16, schedRead(slot, PROC_NICE), true);
  view.setUint32(base + 20, schedRead(slot, PROC_RTPRIO), true);
  view.setBigUint64(base + 24, schedRead64(slot, PROC_DL_RUNTIME), true);
  view.setBigUint64(base + 32, schedRead64(slot, PROC_DL_DEADLINE), true);
  view.setBigUint64(base + 40, schedRead64(slot, PROC_DL_PERIOD), true);
  cpu.setReg(0, 0n);
}


/** may the guest read (or write, with `wantWrite`) this range? the per-page permission table is
 *  the truth once it has been painted; before that, the mapping list is all there is. */
function guestAccessible(addr: number, len: number, wantWrite: boolean): boolean {
  if (syscalls.ptrFaults(new Uint8Array(memory.buffer), addr, len)) return false;
  if (!protPainted()) return true;
  const bytes = new Uint8Array(memory.buffer);
  const need = wantWrite ? 2 : 1;
  for (let p = addr >>> 12, last = (addr + len - 1) >>> 12; p <= last; p++) {
    if ((bytes[PROT_TABLE_BASE + p] & need) === 0) return false;
  }
  return true;
}

/**
 * splice, tee and vmsplice: moving bytes without a trip through the caller's address space.
 *
 * a pipe here is an arena ring, so "moving pages between a pipe and a file" is a copy through a
 * worker-local buffer rather than a page steal -- the observable behaviour is what matters, and
 * the observable behaviour is that the bytes arrive and the offsets advance. what does have to be
 * exact is the bookkeeping: bytes are only taken OUT of a pipe once the destination has accepted
 * them (peek, write, then discard), because a short write that consumed the whole peek would lose
 * everything that did not fit.
 */
const SPLICE_F_NONBLOCK = 2;
const SPLICE_F_ALL = 1 | 2 | 4 | 8;   // MOVE | NONBLOCK | MORE | GIFT
const SPLICE_CHUNK = 65536;

/** the ring behind a pipe descriptor, and which end of it this descriptor is. */
function pipeEnd(fd: number): { ring: PipeRing; off: number; mode: 'read' | 'write' } | null {
  const info = kernelPipes.get(fd);
  if (!info) return null;
  return { ring: new PipeRing(shmArena, info.pipeOff), off: info.pipeOff, mode: info.mode };
}

/**
 * the ring a stream unix socket reads from, if this descriptor is one.
 *
 * splice needs a pipe on ONE side; the other may be a socket, and a connected stream socket is a
 * byte ring exactly like a pipe. a datagram socket is not: its bytes travel behind a length
 * header, and handing those raw to a pipe would splice the framing along with the message.
 */
function unixStreamRx(fd: number): PipeRing | null {
  const s = unixSocks.get(fd);
  if (!s || s.framed || s.rx === undefined) return null;
  return new PipeRing(shmArena, s.rx);
}

/** the ring a stream unix socket writes into, if this descriptor is one. */
function unixStreamTx(fd: number): PipeRing | null {
  const s = unixSocks.get(fd);
  if (!s || s.framed || s.tx === undefined) return null;
  return new PipeRing(shmArena, s.tx);
}

/** read from a native file, at the current position or at an explicit one. */
function spliceReadFile(nativeFd: number, count: number, at: number | null): Uint8Array | number {
  let out: Uint8Array | null = null;
  let e = 0;
  // one reply carries at most one slot's worth; asking for more is a short read, not an error.
  const want = Math.min(count, BRIDGE_DATA_MAX);
  const method = at === null ? 'fs_read' : 'fs_pread';
  const args = at === null ? { fd: nativeFd, count: want } : { fd: nativeFd, offset: at, count: want };
  handleBridgeCall(method, args, (resp: any) => {
    if (typeof resp?.ret === 'number' && resp.ret < 0) e = resp.ret;
    else out = resp?.bytes ?? new Uint8Array(0);
  });
  return e !== 0 ? e : (out ?? new Uint8Array(0));
}

/** write to a native file at an explicit offset, for splice's off_out. */
function spliceWriteAt(nativeFd: number, data: Uint8Array, at: number): number {
  let r = -5;
  handleBridgeCall('fs_pwrite', { fd: nativeFd, offset: at, data: bytesToBase64Fast(data) },
                   (resp: any) => { r = typeof resp?.ret === 'number' ? resp.ret : -5; });
  return r;
}

function handleSplice(nr: number, cpu: CpuState, view: DataView): void {
  const err = (e: number) => cpu.setReg(0, BigInt.asUintN(64, BigInt(-e)));
  if (nr === 75) { handleVmsplice(cpu, view); return; }
  if (nr === 77) { handleTee(cpu); return; }

  const inFd = Number(BigInt.asIntN(32, cpu.getReg(0)));
  const offInPtr = Number(cpu.getReg(1));
  const outFd = Number(BigInt.asIntN(32, cpu.getReg(2)));
  const offOutPtr = Number(cpu.getReg(3));
  const len = Number(BigInt.asIntN(64, cpu.getReg(4)));
  const flags = Number(cpu.getReg(5));

  if (flags & ~SPLICE_F_ALL) { err(22); return; }
  if (len < 0) { err(22); return; }
  const inEntry = syscalls.getFdEntry(inFd);
  const outEntry = syscalls.getFdEntry(outFd);
  if (!inEntry || !syscalls.fdAccess(inFd).readable) { err(9); return; }
  if (!outEntry || !syscalls.fdAccess(outFd).writable) { err(9); return; }
  // an O_PATH descriptor names a file without opening it: it can be stat'd and passed around,
  // and every operation that would move bytes through it is EBADF.
  if (syscalls.fdIsPath(inFd) || syscalls.fdIsPath(outFd)) { err(9); return; }
  // a directory has no byte stream to splice.
  if (inEntry.type === 'native' && inEntry.isDir) { err(22); return; }
  if (outEntry.type === 'native' && outEntry.isDir) { err(22); return; }

  const src = pipeEnd(inFd), dst = pipeEnd(outFd);
  const inIsPipe = src?.mode === 'read', outIsPipe = dst?.mode === 'write';
  // splice always has a pipe on one side; that is the whole shape of the call.
  if (!inIsPipe && !outIsPipe) { err(22); return; }
  // a pipe has no position, so asking to splice at an offset in one is ESPIPE.
  if (inIsPipe && offInPtr !== 0) { err(29); return; }
  if (outIsPipe && offOutPtr !== 0) { err(29); return; }
  // and a pipe cannot be spliced into itself.
  if (inIsPipe && outIsPipe && src!.off === dst!.off) { err(22); return; }
  if (len === 0) { cpu.setReg(0, 0n); return; }

  const nonblock = (flags & SPLICE_F_NONBLOCK) !== 0;

  // O_APPEND makes the destination's position the kernel's to choose, and splice refuses to
  // write where it cannot say where the bytes went.
  if (!outIsPipe && (syscalls.fdStatusFlags(outFd) & 0o2000)) { err(22); return; }

  const srcRing = inIsPipe ? src!.ring : unixStreamRx(inFd);
  if (srcRing) {
    // a ring (pipe or stream socket) -> anything that takes bytes
    if (!outIsPipe && outEntry.type !== 'native' && !unixSocks.has(outFd)) { err(22); return; }
    let outAt: bigint | null = null;
    if (offOutPtr !== 0) {
      if (!guestAccessible(offOutPtr, 8, true)) { err(14); return; }
      outAt = view.getBigInt64(offOutPtr + GUEST_MEM_BASE, true);
      if (outAt < 0n) { err(22); return; }
    }
    // a ring destination is written directly rather than through the generic write path: that
    // path blocks until everything fits, which would ignore SPLICE_F_NONBLOCK, and taking bytes
    // out of the source before knowing they fit would lose whatever did not.
    const dstRing = outIsPipe ? dst!.ring : unixStreamTx(outFd);
    const buf = new Uint8Array(Math.min(len, SPLICE_CHUNK));
    let got = 0;
    for (;;) {
      if (dstRing?.readerClosed) { syscalls.raiseSignal(13); err(32); return; }  // SIGPIPE + EPIPE
      got = srcRing.peek(buf);
      if (dstRing) got = Math.min(got, dstRing.freeBytes);
      if (got > 0) break;
      // nothing to move: either the source is dry (and may be finished) or the destination is full.
      if (srcRing.bytesAvailable === 0 && !srcRing.writersOpen()) { cpu.setReg(0, 0n); return; }
      if (nonblock || syscalls.isNonBlockingFd(inFd) || syscalls.isNonBlockingFd(outFd)) { err(11); return; }
      if (pollAndQueueSignals()) { err(4); return; }
      waitIo(readIoGeneration(), 20);
    }
    const data = buf.subarray(0, got);
    const w = dstRing ? dstRing.write(data)
      : outAt !== null ? spliceWriteAt((outEntry as any).nativeFd, data, Number(outAt))
      : sendfileWrite(outFd, data);
    if (w < 0) { err(-w); return; }
    srcRing.discard(w);
    bumpIoGeneration();
    if (outAt !== null) view.setBigInt64(offOutPtr + GUEST_MEM_BASE, outAt + BigInt(w), true);
    cpu.setReg(0, BigInt(w));
    return;
  }

  // file -> pipe
  if (inEntry.type !== 'native') { err(22); return; }
  let inAt: bigint | null = null;
  if (offInPtr !== 0) {
    if (!guestAccessible(offInPtr, 8, true)) { err(14); return; }
    inAt = view.getBigInt64(offInPtr + GUEST_MEM_BASE, true);
    if (inAt < 0n) { err(22); return; }
  }
  // read only as much as the pipe can take right now: a read that overshoots would advance the
  // file position past bytes that never reached the pipe.
  let room = 0;
  for (;;) {
    if (dst!.ring.readerClosed) { syscalls.raiseSignal(13); err(32); return; }  // SIGPIPE + EPIPE
    room = Math.min(len, SPLICE_CHUNK, dst!.ring.freeBytes);
    if (room > 0) break;
    if (nonblock || syscalls.isNonBlockingFd(outFd)) { err(11); return; }
    if (pollAndQueueSignals()) { err(4); return; }
    waitIo(readIoGeneration(), 20);
  }
  const got = spliceReadFile(inEntry.nativeFd, room, inAt === null ? null : Number(inAt));
  if (typeof got === 'number') { err(-got); return; }
  if (got.length === 0) { cpu.setReg(0, 0n); return; }              // end of file
  const n = dst!.ring.write(got);
  bumpIoGeneration();
  if (inAt !== null) view.setBigInt64(offInPtr + GUEST_MEM_BASE, inAt + BigInt(n), true);
  cpu.setReg(0, BigInt(n));
}

/** tee(2): duplicate a pipe's contents into another pipe, leaving the first one readable. */
function handleTee(cpu: CpuState): void {
  const err = (e: number) => cpu.setReg(0, BigInt.asUintN(64, BigInt(-e)));
  const inFd = Number(BigInt.asIntN(32, cpu.getReg(0)));
  const outFd = Number(BigInt.asIntN(32, cpu.getReg(1)));
  const len = Number(BigInt.asIntN(64, cpu.getReg(2)));
  const flags = Number(cpu.getReg(3));
  if (flags & ~SPLICE_F_ALL) { err(22); return; }
  if (len < 0) { err(22); return; }
  if (!syscalls.getFdEntry(inFd) || !syscalls.getFdEntry(outFd)) { err(9); return; }
  const src = pipeEnd(inFd), dst = pipeEnd(outFd);
  // tee has a pipe on BOTH sides -- that is what distinguishes it from splice -- and they have to
  // be different pipes.
  if (src?.mode !== 'read' || dst?.mode !== 'write') { err(22); return; }
  if (src.off === dst.off) { err(22); return; }
  if (len === 0) { cpu.setReg(0, 0n); return; }
  const nonblock = (flags & SPLICE_F_NONBLOCK) !== 0;
  const buf = new Uint8Array(Math.min(len, SPLICE_CHUNK));
  let got = 0;
  for (;;) {
    if (dst.ring.readerClosed) { syscalls.raiseSignal(13); err(32); return; }
    got = Math.min(src.ring.peek(buf), dst.ring.freeBytes);
    if (got > 0) break;
    if (src.ring.bytesAvailable === 0 && !src.ring.writersOpen()) { cpu.setReg(0, 0n); return; }
    if (nonblock || syscalls.isNonBlockingFd(inFd)) { err(11); return; }
    if (pollAndQueueSignals()) { err(4); return; }
    waitIo(readIoGeneration(), 20);
  }
  const n = dst.ring.write(buf.subarray(0, got));   // and the source keeps its copy
  bumpIoGeneration();
  cpu.setReg(0, BigInt(n));
}

/** vmsplice(2): the caller's own memory on one side of the pipe instead of a descriptor. */
function handleVmsplice(cpu: CpuState, view: DataView): void {
  const err = (e: number) => cpu.setReg(0, BigInt.asUintN(64, BigInt(-e)));
  const fd = Number(BigInt.asIntN(32, cpu.getReg(0)));
  const iovPtr = Number(cpu.getReg(1));
  const nrSegs = Number(cpu.getReg(2));
  const flags = Number(cpu.getReg(3));
  if (flags & ~SPLICE_F_ALL) { err(22); return; }
  if (!syscalls.getFdEntry(fd)) { err(9); return; }
  const end = pipeEnd(fd);
  if (!end) { err(9); return; }                                   // not a pipe: EBADF
  if (nrSegs > 1024) { err(22); return; }                         // UIO_MAXIOV
  if (nrSegs === 0) { cpu.setReg(0, 0n); return; }
  if (!guestAccessible(iovPtr, nrSegs * 16, false)) { err(14); return; }
  const segs: Array<{ base: number; len: number }> = [];
  let total = 0;
  for (let i = 0; i < nrSegs; i++) {
    const base = Number(view.getBigUint64(iovPtr + GUEST_MEM_BASE + i * 16, true));
    const l = Number(view.getBigUint64(iovPtr + GUEST_MEM_BASE + i * 16 + 8, true));
    if (l > 0 && !guestAccessible(base, l, end.mode === 'read')) { err(14); return; }
    segs.push({ base, len: l });
    total += l;
  }
  if (total === 0) { cpu.setReg(0, 0n); return; }
  const bytes = new Uint8Array(memory.buffer);
  const nonblock = (flags & SPLICE_F_NONBLOCK) !== 0;

  if (end.mode === 'write') {
    // gather the caller's buffers into the pipe
    const data = new Uint8Array(Math.min(total, SPLICE_CHUNK));
    let filled = 0;
    for (const s of segs) {
      if (filled >= data.length) break;
      const take = Math.min(s.len, data.length - filled);
      data.set(bytes.subarray(s.base + GUEST_MEM_BASE, s.base + GUEST_MEM_BASE + take), filled);
      filled += take;
    }
    for (;;) {
      if (end.ring.readerClosed) { syscalls.raiseSignal(13); err(32); return; }
      if (end.ring.freeBytes > 0) break;
      if (nonblock || syscalls.isNonBlockingFd(fd)) { err(11); return; }
      if (pollAndQueueSignals()) { err(4); return; }
      waitIo(readIoGeneration(), 20);
    }
    const n = end.ring.write(data.subarray(0, filled));
    bumpIoGeneration();
    cpu.setReg(0, BigInt(n));
    return;
  }

  // scatter the pipe's bytes into the caller's buffers
  const buf = new Uint8Array(Math.min(total, SPLICE_CHUNK));
  let got = 0;
  for (;;) {
    got = end.ring.peek(buf);
    if (got > 0) break;
    if (!end.ring.writersOpen()) { cpu.setReg(0, 0n); return; }
    if (nonblock || syscalls.isNonBlockingFd(fd)) { err(11); return; }
    if (pollAndQueueSignals()) { err(4); return; }
    waitIo(readIoGeneration(), 20);
  }
  let placed = 0;
  for (const s of segs) {
    if (placed >= got) break;
    const take = Math.min(s.len, got - placed);
    bytes.set(buf.subarray(placed, placed + take), s.base + GUEST_MEM_BASE);
    placed += take;
  }
  end.ring.discard(placed);
  bumpIoGeneration();
  cpu.setReg(0, BigInt(placed));
}

/**
 * hand `data` to a descriptor from outside the write(2) path, for sendfile.
 *
 * returns bytes written, or a negative errno. a partial write is a real answer: sendfile stops
 * there and reports what it moved, exactly as the kernel does.
 */
function sendfileWrite(fd: number, data: Uint8Array): number {
  const e = syscalls.getFdEntry(fd);
  if (!e) return -9; // EBADF
  if (e.type === 'native') {
    let r = -5;
    handleBridgeCall('fs_write', { fd: e.nativeFd, data: bytesToBase64Fast(data) }, (resp: any) => {
      r = typeof resp?.ret === 'number' ? resp.ret : -5;
    });
    return r;
  }
  // a unix socket goes through the same write path everything else does. it used to frame its own
  // datagrams here, with a four-byte header that stopped matching the one the reader expects the
  // moment the frame grew a sender name -- two spellings of one wire format is one too many.
  const sock = unixSocks.get(fd);
  if (sock) {
    if (sock.framed) {
      const dest = dgramDefaultRing(sock);
      if (dest < 0) return dest;
      return unixDgramWrite(fd, dest, data, sock.name);
    }
    if (sock.tx === undefined) return -107;                   // ENOTCONN
    return unixStreamWrite(fd, sock.tx, data);
  }
  const pipe = kernelPipes.get(fd);
  if (pipe?.mode === 'write') {
    const ring = new PipeRing(shmArena, pipe.pipeOff);
    if (syscalls.isNonBlockingFd(fd)) {
      const n = ring.write(data);
      if (n === -1) { syscalls.raiseSignal(13); return -32; } // EPIPE
      if (n === 0) return -11;                                // EAGAIN
      return n;
    }
    const n = ring.writeAll(data, interruptibleWaiter);
    if (n === -1) { syscalls.raiseSignal(13); return -32; }
    if (n === -2) return -4;                                  // EINTR
    return n;
  }
  return syscalls.writeOutOfBand(fd, data);
}

/**
 * sendfile(out_fd, in_fd, offset, count): move bytes between descriptors without the guest
 * providing a buffer. the source has to be something seekable -- a pipe has no offset to read
 * from, which is why linux answers EINVAL for one -- and the destination can be anything.
 */
function handleSendfile(cpu: CpuState, view: DataView): void {
  const err = (e: number) => cpu.setReg(0, BigInt.asUintN(64, BigInt(-e)));
  const outFd = Number(BigInt.asIntN(32, cpu.getReg(0)));
  const inFd = Number(BigInt.asIntN(32, cpu.getReg(1)));
  const offPtr = Number(cpu.getReg(2));
  let count = Number(BigInt.asIntN(64, cpu.getReg(3)));

  const inEntry = syscalls.getFdEntry(inFd);
  if (!inEntry || !syscalls.fdAccess(inFd).readable) { err(9); return; }   // EBADF
  // the source must be a file with a position to read from: a pipe or a socket has none, which is
  // what linux answers EINVAL for. /proc's synthetic files are generated per read rather than
  // stored, so they are not a source either.
  if (inEntry.type !== 'native') { err(22); return; }                      // EINVAL

  let pos = -1n;
  if (offPtr !== 0) {
    // the offset is read AND written, so a read-only page is as much a fault as an unmapped one.
    if (!guestAccessible(offPtr, 8, true)) { err(14); return; } // EFAULT
    pos = view.getBigInt64(offPtr + GUEST_MEM_BASE, true);
    if (pos < 0n) { err(22); return; }                          // EINVAL
  }

  const outEntry = syscalls.getFdEntry(outFd);
  if (!outEntry || !syscalls.fdAccess(outFd).writable) { err(9); return; } // EBADF
  if (count < 0) { err(22); return; }

  let moved = 0;
  while (count > 0) {
    const chunk = Math.min(count, BRIDGE_DATA_MAX);
    let data: Uint8Array | null = null;
    let readErr = 0;
    const args = offPtr !== 0
      ? { fd: inEntry.nativeFd, offset: Number(pos) + moved, count: chunk }
      : { fd: inEntry.nativeFd, count: chunk };
    handleBridgeCall(offPtr !== 0 ? 'fs_pread' : 'fs_read', args, (resp: any) => {
      if (typeof resp?.ret === 'number' && resp.ret < 0) readErr = resp.ret;
      else data = resp?.bytes ?? new Uint8Array(0);
    });
    if (readErr !== 0) { if (moved === 0) { err(-readErr); return; } break; }
    const got: Uint8Array = data ?? new Uint8Array(0);
    if (got.length === 0) break;                                // end of file

    const w = sendfileWrite(outFd, got);
    if (w < 0) { if (moved === 0) { err(-w); return; } break; }
    moved += w;
    count -= w;
    if (w < got.length) break;                                  // the destination took what it could
  }

  // with an offset the source's own position is untouched and the caller's offset moves instead,
  // which is the whole difference between the two forms.
  if (offPtr !== 0) view.setBigInt64(offPtr + GUEST_MEM_BASE, pos + BigInt(moved), true);
  cpu.setReg(0, BigInt(moved));
}

/** one round trip to the kernel's inotify registry. */
/** the native filesystem is one device, and writeBridgeStat reports st_dev 1 for everything on it. */
const NATIVE_FS_DEV = 1;

function inotifyCall(msg: Record<string, unknown>, words = 3): number[] {
  Atomics.store(myMailbox, MBOX_READY, 0);
  (self as any).postMessage({ type: 'inotify', ...msg });
  awaitMailbox('inotify');
  const out: number[] = [];
  for (let i = 0; i < words; i++) out.push(Atomics.load(myMailbox, MBOX_VALUE0 + i));
  return out;
}

/**
 * the inotify part of /proc/<pid>/fdinfo/<fd>: one line per watch, naming the object being watched
 * and the mask it is watched with. programs read this to find out what a descriptor they inherited
 * is actually watching, which they cannot ask the descriptor itself.
 */
function inotifyFdinfo(id: number): string {
  let out = '', start = 0;
  for (;;) {
    const r = inotifyCall({ op: 'watches', id, start }, 30);
    if (r[0] !== 0) return out;
    const total = r[1], n = r[2];
    for (let i = 0; i < n; i++) {
      const wd = r[3 + i * 3], mask = r[4 + i * 3] >>> 0, ino = r[5 + i * 3] >>> 0;
      out += `inotify wd:${wd} ino:${ino.toString(16)} sdev:${NATIVE_FS_DEV.toString(16)}`
           + ` mask:${mask.toString(16)} ignored_mask:0 fhandle-bytes:8 fhandle-type:1`
           + ` f_handle:${ino.toString(16).padStart(16, '0')}\n`;
    }
    start += n;
    if (start >= total || n === 0) return out;
  }
}

/** are there any watches at all? one atomic load, so a mutation costs nothing when nobody looks. */
function inotifyActive(): boolean {
  return Atomics.load(globalWords, GLOBAL_INOTIFY_WATCHES) > 0;
}

/**
 * report a filesystem change so watchers hear about it.
 *
 * this is fire-and-forget: the reporter does not wait for the kernel, because nothing it does
 * next depends on the event having been queued.
 */
function reportInotify(path: string, mask: number, cookie = 0, isDir?: boolean,
                       selfOnly = false, probe = path, unlinked = false): void {
  if (!inotifyActive()) return;
  // whether the object is a directory is part of the event (IN_ISDIR), and the only way to know for
  // an object nobody told us about is to ask. the cost is confined to the case where somebody is
  // actually watching, which the check above has already established.
  if (isDir === undefined) {
    handleBridgeCall('fs_stat', { path: probe }, (r) => {
      if (r && r.ret === 0 && r._meta) isDir = ((r._meta[0] | 0) & 0o170000) === 0o040000;
    });
  }
  (self as any).postMessage({ type: 'inotify_event', path, mask, cookie, isdir: isDir === true,
                              selfonly: selfOnly, moved: probe, unlinked });
}

/**
 * inotify_init1 / inotify_add_watch / inotify_rm_watch, and read(2) on the resulting descriptor.
 *
 * the watches live in the kernel because they are about what OTHER processes do; this side owns
 * only the descriptor. reads take one arena block of packed events at a time.
 */
function handleInotify(nr: number, cpu: CpuState, view: DataView): void {
  const bytes = new Uint8Array(memory.buffer);
  if (nr === 26) { // inotify_init1(flags)
    const IN_NONBLOCK = 0o4000, IN_CLOEXEC = 0o2000000;
    const flags = Number(cpu.getReg(0));
    if (flags & ~(IN_NONBLOCK | IN_CLOEXEC)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
    const fd = syscalls.allocLowestFd();
    if (fd < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-24))); return; } // EMFILE
    const id = inotifyCall({ op: 'init', id: 0 })[1];
    syscalls.registerFd(fd, { type: 'inotify', id }, flags & IN_NONBLOCK);
    if (flags & IN_CLOEXEC) syscalls.markCloexec(fd);
    cpu.setReg(0, BigInt(fd));
    return;
  }
  if (nr === 27) { // inotify_add_watch(fd, pathname, mask)
    const id = syscalls.inotifyIdOf(Number(cpu.getReg(0)));
    if (id < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-9))); return; } // EBADF
    const pathPtr = Number(cpu.getReg(1));
    if (badGuestPtr(bytes, pathPtr, 1)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; }
    const mask = Number(cpu.getReg(2)) >>> 0;
    // a mask with no event bits at all watches nothing, which linux rejects rather than accepting
    const IN_ALL_EVENTS = 0x00000fff;
    if ((mask & IN_ALL_EVENTS) === 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
    const raw = readCString(pathPtr);
    if (raw === null || raw.length === 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-2))); return; } // ENOENT
    // /proc and /dev are synthesised per-process and have no host path, so nothing outside this
    // process could ever change them in a way a watch would see.
    const path = syscalls.resolveForWatch(raw);
    if (path === null) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; } // EINVAL
    // the path has to exist: a watch on nothing is ENOENT, not a watch that never fires.
    let exists = false, ino = 0;
    handleBridgeCall('fs_stat', { path }, (r) => {
      exists = (r?.ret ?? -1) >= 0;
      if (exists && r._meta) ino = r._meta[6] | 0;
    });
    if (!exists) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-2))); return; }
    const r = inotifyCall({ op: 'add', id, path, mask, ino });
    cpu.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
    return;
  }
  if (nr === 28) { // inotify_rm_watch(fd, wd)
    const id = syscalls.inotifyIdOf(Number(cpu.getReg(0)));
    if (id < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-9))); return; }
    const r = inotifyCall({ op: 'rm', id, wd: Number(BigInt.asIntN(32, cpu.getReg(1))) });
    cpu.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
    return;
  }
  if (nr === 57) { // close
    inotifyCall({ op: 'close', id: syscalls.inotifyIdOf(Number(cpu.getReg(0))) });
    syscalls.unregisterFd(Number(cpu.getReg(0)));
    cpu.setReg(0, 0n);
    return;
  }
  // read(fd, buf, count): hand back as many whole events as fit.
  const fd = Number(cpu.getReg(0));
  const rid = syscalls.inotifyIdOf(fd);
  const buf = Number(cpu.getReg(1));
  const count = Number(cpu.getReg(2));
  if (count > 0 && badGuestPtr(bytes, buf, count)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; }
  for (;;) {
    const r = inotifyCall({ op: 'read', id: rid, maxlen: count });
    if (r[0] === 0) {
      const off = r[1], len = r[2];
      new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, len).set(new Uint8Array(shmArena, off, len));
      inotifyCall({ op: 'free', id: rid, off, len });
      cpu.setReg(0, BigInt(len));
      return;
    }
    if (r[0] !== -11) { cpu.setReg(0, BigInt.asUintN(64, BigInt(r[0]))); return; }
    if (syscalls.isNonBlockingFd(fd)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-11))); return; } // EAGAIN
    if (pollAndQueueSignals()) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-4))); return; }         // EINTR
    waitIo(readIoGeneration(), 20);
  }
}

/** one round trip to the kernel's posix message-queue registry. */
function eventfdCall(msg: Record<string, unknown>): number[] {
  Atomics.store(myMailbox, MBOX_READY, 0);
  (self as any).postMessage({ type: 'eventfd', ...msg });
  awaitMailbox('eventfd');
  return [
    Atomics.load(myMailbox, MBOX_VALUE0),
    Atomics.load(myMailbox, MBOX_VALUE0 + 1),
    Atomics.load(myMailbox, MBOX_VALUE0 + 2),
  ];
}

function posixMqCall(msg: Record<string, unknown>): number[] {
  Atomics.store(myMailbox, MBOX_READY, 0);
  (self as any).postMessage({ type: 'posix_mq', ...msg });
  awaitMailbox('posix_mq');
  const out: number[] = [];
  for (let i = 0; i < 6; i++) out.push(Atomics.load(myMailbox, MBOX_VALUE0 + i));
  return out;
}

/**
 * a posix queue name, as the kernel's flat namespace sees it, or an errno.
 *
 * libc strips the leading slash the interface demands before making the call, but a test that
 * makes the syscall itself will not have, so both spellings arrive here. what cannot arrive is a
 * name with a slash left INSIDE it: the namespace is flat, so there is no directory for it to
 * name, and nothing could ever be created there.
 */
function mqName(ptr: number): string | number {
  const bytes = new Uint8Array(memory.buffer);
  if (syscalls.ptrFaults(bytes, ptr, 1)) return -14;           // EFAULT
  const raw = readCString(ptr);
  if (raw === null) return -14;
  const name = raw.startsWith('/') ? raw.slice(1) : raw;
  if (name.length === 0 || name === '.' || name === '..' || name.includes('/')) return -22; // EINVAL
  if (name.length > 255) return -36;                           // ENAMETOOLONG
  return name;
}

/**
 * an absolute deadline in a timespec, as milliseconds from now. null means "no deadline"; an
 * already-expired one is a negative number rather than an error, because the call still gets its
 * one attempt before ETIMEDOUT. the errno comes back in its own field: a deadline fourteen
 * milliseconds in the past is a perfectly good number and must not read as EFAULT.
 */
function mqDeadline(ptr: number, view: DataView): { err: number } | { leftMs: number | null } {
  if (ptr === 0) return { leftMs: null };
  const bytes = new Uint8Array(memory.buffer);
  if (syscalls.ptrFaults(bytes, ptr, 16)) return { err: -14 };          // EFAULT
  const sec = view.getBigInt64(ptr + GUEST_MEM_BASE, true);
  const nsec = view.getBigInt64(ptr + GUEST_MEM_BASE + 8, true);
  if (nsec < 0n || nsec >= 1_000_000_000n || sec < 0n) return { err: -22 };  // EINVAL
  const targetMs = Number(sec) * 1000 + Number(nsec) / 1e6;
  return { leftMs: targetMs - Number(syscalls.clockNs(0) / 1_000_000n) };
}

/**
 * mq_open / mq_unlink / mq_timedsend / mq_timedreceive / mq_notify / mq_getsetattr, and the close
 * of a descriptor onto a queue.
 *
 * the queues are named and system-wide, so every one of these is a kernel round trip; the message
 * BODIES live in the shared arena, so a sender hands its bytes over once and a receiver in another
 * process copies them straight out rather than having them relayed twice.
 */
function handlePosixMq(nr: number, cpu: CpuState, view: DataView): void {
  const bytes = new Uint8Array(memory.buffer);
  const creds = syscalls.getCreds();
  const err = (e: number) => cpu.setReg(0, BigInt.asUintN(64, BigInt(e)));
  const O_NONBLOCK = 0o4000, O_CLOEXEC = 0o2000000, O_CREAT = 0o100;
  const MQ_PRIO_MAX = 32768;

  if (nr === 57) { // close(mqdes)
    const fd = Number(cpu.getReg(0));
    const id = syscalls.mqIdOf(fd);
    // the kernel only wants to hear about the LAST descriptor this process holds onto the queue:
    // a dup or a second mq_open of the same name is still one hold.
    if (!syscalls.mqHeldElsewhere(fd, id)) posixMqCall({ op: 'close', id, pid: myPid });
    syscalls.unregisterFd(fd);
    cpu.setReg(0, 0n);
    return;
  }

  if (nr === 180) { // mq_open(name, oflag, mode, attr)
    const name = mqName(Number(cpu.getReg(0)));
    if (typeof name === 'number') { err(name); return; }
    const oflag = Number(BigInt.asIntN(32, cpu.getReg(1)));
    const accmode = oflag & 3;
    if (accmode === 3) { err(-22); return; }                   // EINVAL: not an access mode
    const mode = Number(cpu.getReg(2));
    const attrPtr = Number(cpu.getReg(3));
    let hasAttr = false, maxmsg = 0, msgsize = 0;
    // an attr is only consulted when the call may CREATE the queue; on a plain open it is ignored.
    if ((oflag & O_CREAT) && attrPtr !== 0) {
      if (syscalls.ptrFaults(bytes, attrPtr, 64)) { err(-14); return; }
      // struct mq_attr: mq_flags, mq_maxmsg, mq_msgsize, mq_curmsgs, then four reserved longs.
      maxmsg = Number(view.getBigInt64(attrPtr + GUEST_MEM_BASE + 8, true));
      msgsize = Number(view.getBigInt64(attrPtr + GUEST_MEM_BASE + 16, true));
      hasAttr = true;
    }
    const fd = syscalls.allocLowestFd(0);
    if (fd < 0) { err(-24); return; }                          // EMFILE
    const r = posixMqCall({ op: 'open', name, oflag, accmode, mode, hasAttr, maxmsg, msgsize,
                            uid: creds.euid, gid: creds.egid, pid: myPid });
    if (r[0] < 0) { err(r[0]); return; }
    syscalls.registerFd(fd, { type: 'mqueue', id: r[0], name }, oflag & (3 | O_NONBLOCK));
    if (oflag & O_CLOEXEC) syscalls.markCloexec(fd);
    cpu.setReg(0, BigInt(fd));
    return;
  }

  if (nr === 181) { // mq_unlink(name)
    const name = mqName(Number(cpu.getReg(0)));
    if (typeof name === 'number') { err(name); return; }
    err(posixMqCall({ op: 'unlink', name, uid: creds.euid, gid: creds.egid })[0]);
    return;
  }

  const SIGEV_SIGNAL = 0, SIGEV_NONE = 1, SIGEV_THREAD = 2;
  const NOTIFY_COOKIE_LEN = 32;
  let notifySigno = -1;
  let notifyRing = -1;   // -1: not a SIGEV_THREAD registration (offset 0 is a real ring)
  let notifyFd = -1;
  let notifyCookie: Uint8Array | undefined;
  if (nr === 184) {
    const sevp = Number(cpu.getReg(1));
    // linux judges the sigevent before it so much as looks at the descriptor, which is why
    // mq_notify(0, <nonsense>) is EINVAL rather than the EBADF stdin would otherwise earn.
    if (sevp !== 0) {
      if (syscalls.ptrFaults(bytes, sevp, 64)) { err(-14); return; }
      const signo = view.getInt32(sevp + GUEST_MEM_BASE + 8, true);
      const kind = view.getInt32(sevp + GUEST_MEM_BASE + 12, true);
      if (kind !== SIGEV_NONE && kind !== SIGEV_SIGNAL && kind !== SIGEV_THREAD) { err(-22); return; }
      if (kind === SIGEV_SIGNAL && (signo <= 0 || signo > 64)) { err(-22); return; }
      // SIGEV_THREAD asks the kernel to write a cookie to a netlink socket the caller opened, and
      // names that socket in the field a signal number would otherwise occupy.
      if (kind === SIGEV_THREAD) {
        // the socket may have been opened by a SIBLING THREAD -- musl's mq_notify opens it in one
        // thread and registers from another -- so a local miss is not an error: the kernel knows
        // which ring the descriptor names for the whole thread group.
        const nlSock = unixSocks.get(signo);
        notifyRing = nlSock?.rx ?? -1;
        notifyFd = signo;
        // the notification IS the 32-byte cookie the caller left behind, returned to it with its
        // last byte saying why it woke. libc reads it to tell a real notification from a teardown.
        const cookiePtr = Number(view.getBigUint64(sevp + GUEST_MEM_BASE, true) & 0xffffffffn);
        if (syscalls.ptrFaults(bytes, cookiePtr, NOTIFY_COOKIE_LEN)) { err(-14); return; }
        notifyCookie = bytes.slice(cookiePtr + GUEST_MEM_BASE, cookiePtr + GUEST_MEM_BASE + NOTIFY_COOKIE_LEN);
      }
      notifySigno = kind === SIGEV_NONE ? 0 : signo;
    }
  }

  const fd = Number(cpu.getReg(0));
  const id = syscalls.mqIdOf(fd);
  if (id < 0) { err(-9); return; }                             // EBADF: not a queue descriptor
  const access = syscalls.fdAccess(fd);
  const nonblock = syscalls.isNonBlockingFd(fd);

  if (nr === 182) { // mq_timedsend(mqdes, msg_ptr, msg_len, msg_prio, abs_timeout)
    if (!access.writable) { err(-9); return; }
    const ptr = Number(cpu.getReg(1));
    const len = Number(BigInt.asIntN(64, cpu.getReg(2)));
    const prio = Number(BigInt.asIntN(32, cpu.getReg(3)));
    if (prio < 0 || prio >= MQ_PRIO_MAX) { err(-22); return; }
    if (len < 0) { err(-22); return; }
    // a bounds check is not enough: ltp hands these calls a PROT_NONE page, which is inside the
    // guest's address space and unreadable all the same. reading it anyway silently enqueued a
    // message of zeros, and the test that later received it saw the zeros rather than an EFAULT.
    if (len > 0 && syscalls.ptrFaults(bytes, ptr, len)) { err(-14); return; }
    const dl = mqDeadline(Number(cpu.getReg(4)), view);
    if ('err' in dl) { err(dl.err); return; }
    const leftMs = dl.leftMs;
    const data = len > 0 ? bytes.slice(ptr + GUEST_MEM_BASE, ptr + GUEST_MEM_BASE + len) : new Uint8Array(0);
    const startMs = Number(syscalls.clockNs(0) / 1_000_000n);
    for (;;) {
      const r = posixMqCall({ op: 'send', id, prio, len, data });
      if (r[0] !== -11) { err(r[0]); return; }
      if (nonblock) { err(-11); return; }                      // EAGAIN
      if (leftMs !== null && Number(syscalls.clockNs(0) / 1_000_000n) - startMs >= leftMs) { err(-110); return; } // ETIMEDOUT
      if (pollAndQueueSignals()) { err(-4); return; }           // EINTR
      waitIo(readIoGeneration(), 20);
    }
  }

  if (nr === 183) { // mq_timedreceive(mqdes, msg_ptr, msg_len, msg_prio, abs_timeout)
    if (!access.readable) { err(-9); return; }
    const ptr = Number(cpu.getReg(1));
    const maxlen = Number(BigInt.asIntN(64, cpu.getReg(2)));
    const prioPtr = Number(cpu.getReg(3));
    if (maxlen < 0) { err(-22); return; }
    if (maxlen > 0 && syscalls.ptrWriteFaults(bytes, ptr, maxlen)) { err(-14); return; }
    if (prioPtr !== 0 && syscalls.ptrWriteFaults(bytes, prioPtr, 4)) { err(-14); return; }
    const dl = mqDeadline(Number(cpu.getReg(4)), view);
    if ('err' in dl) { err(dl.err); return; }
    const leftMs = dl.leftMs;
    const startMs = Number(syscalls.clockNs(0) / 1_000_000n);
    for (;;) {
      const r = posixMqCall({ op: 'recv', id, maxlen });
      if (r[0] === 0) {
        const off = r[1], len = r[2];
        if (len > 0) bytes.set(new Uint8Array(shmArena, off, len), ptr + GUEST_MEM_BASE);
        if (prioPtr !== 0) view.setUint32(prioPtr + GUEST_MEM_BASE, r[3] >>> 0, true);
        posixMqCall({ op: 'free', off, len });
        cpu.setReg(0, BigInt(len));
        return;
      }
      if (r[0] !== -11) { err(r[0]); return; }
      if (nonblock) { err(-11); return; }
      if (leftMs !== null && Number(syscalls.clockNs(0) / 1_000_000n) - startMs >= leftMs) { err(-110); return; }
      if (pollAndQueueSignals()) { err(-4); return; }
      waitIo(readIoGeneration(), 20);
    }
  }

  if (nr === 184) { // mq_notify(mqdes, sevp)
    // a null sigevent withdraws this process's registration, and says nothing if it had none.
    err(posixMqCall({ op: 'notify', id, pid: myPid, signo: notifySigno, ring: notifyRing, nlFd: notifyFd, cookie: notifyCookie })[0]);
    return;
  }

  // mq_getsetattr(mqdes, newattr, oldattr): the only settable thing is O_NONBLOCK, which belongs
  // to the descriptor rather than the queue, so it never reaches the kernel at all.
  const newPtr = Number(cpu.getReg(1));
  const oldPtr = Number(cpu.getReg(2));
  if (newPtr !== 0 && syscalls.ptrFaults(bytes, newPtr, 64)) { err(-14); return; }
  if (oldPtr !== 0 && syscalls.ptrWriteFaults(bytes, oldPtr, 64)) { err(-14); return; }
  const attr = posixMqCall({ op: 'getattr', id });
  if (attr[0] < 0) { err(attr[0]); return; }
  if (oldPtr !== 0) {
    const b = oldPtr + GUEST_MEM_BASE;
    bytes.fill(0, b, b + 64);
    view.setBigInt64(b, BigInt(nonblock ? O_NONBLOCK : 0), true);
    view.setBigInt64(b + 8, BigInt(attr[1]), true);
    view.setBigInt64(b + 16, BigInt(attr[2]), true);
    view.setBigInt64(b + 24, BigInt(attr[3]), true);
  }
  if (newPtr !== 0) {
    const want = Number(view.getBigInt64(newPtr + GUEST_MEM_BASE, true));
    syscalls.setFdNonBlocking(fd, (want & O_NONBLOCK) !== 0);
  }
  cpu.setReg(0, 0n);
}

/** one round trip to the kernel's message-queue registry. */
function sysvMsgCall(msg: Record<string, unknown>): number[] {
  Atomics.store(myMailbox, MBOX_READY, 0);
  (self as any).postMessage({ type: 'sysv_msg', ...msg });
  awaitMailbox('sysv_msg');
  const out: number[] = [];
  // 14: the stat reply is thirteen fields, and MSG_STAT adds the queue's real id on the end.
  for (let i = 0; i < 14; i++) out.push(Atomics.load(myMailbox, MBOX_VALUE0 + i));
  return out;
}

/**
 * msgget / msgsnd / msgrcv / msgctl.
 *
 * the registry is system-wide, so every call is a kernel round trip. the message BODIES live in
 * the shared arena: a sender hands the kernel its bytes once, and a receiver in another process
 * copies them straight out of the arena rather than having them relayed through the kernel again.
 */
function handleSysvMsg(nr: number, cpu: CpuState, view: DataView): void {
  const creds = syscalls.getCreds();
  const bytes = new Uint8Array(memory.buffer);

  if (nr === 186) { // msgget(key, msgflg)
    const r = sysvMsgCall({ op: 'get', key: Number(BigInt.asIntN(32, cpu.getReg(0))),
                            flg: Number(cpu.getReg(1)), uid: creds.euid, gid: creds.egid });
    cpu.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
    return;
  }

  if (nr === 189) { // msgsnd(msqid, msgp, msgsz, msgflg)
    const id = Number(BigInt.asIntN(32, cpu.getReg(0)));
    const ptr = Number(cpu.getReg(1));
    const sz = Number(BigInt.asIntN(64, cpu.getReg(2)));
    const flg = Number(cpu.getReg(3));
    const IPC_NOWAIT = 0o4000;
    // struct msgbuf is `{ long mtype; char mtext[]; }`: the size argument counts mtext ALONE.
    if (sz < 0 || sz > MSGMAX) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; } // EINVAL
    if (badGuestPtr(bytes, ptr, 8 + Math.max(0, sz))) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; }
    const mtype = view.getBigInt64(ptr + GUEST_MEM_BASE, true);
    if (mtype < 1n) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }             // EINVAL
    const data = bytes.slice(ptr + GUEST_MEM_BASE + 8, ptr + GUEST_MEM_BASE + 8 + sz);
    // a queue that vanishes while we are asleep on it is EIDRM; an id that never named one is
    // EINVAL. the kernel only knows the id is unknown, so the difference is ours to keep.
    let slept = false;
    for (;;) {
      const r = sysvMsgCall({ op: 'snd', id, mtype: mtype.toString(), len: sz, data,
                              uid: creds.euid, gid: creds.egid });
      if (r[0] === -22 && slept) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-43))); return; }
      if (r[0] !== -11) { cpu.setReg(0, BigInt.asUintN(64, BigInt(r[0]))); return; }
      // the queue is full. IPC_NOWAIT says say so; otherwise wait for a receiver to make room.
      if (flg & IPC_NOWAIT) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-11))); return; }
      if (pollAndQueueSignals()) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-4))); return; } // EINTR
      slept = true;
      waitIo(readIoGeneration(), 20);
    }
  }

  if (nr === 188) { // msgrcv(msqid, msgp, msgsz, msgtyp, msgflg)
    const id = Number(BigInt.asIntN(32, cpu.getReg(0)));
    const ptr = Number(cpu.getReg(1));
    const sz = Number(BigInt.asIntN(64, cpu.getReg(2)));
    const mtyp = BigInt.asIntN(64, cpu.getReg(3));
    const flg = Number(cpu.getReg(4));
    const IPC_NOWAIT = 0o4000, MSG_COPY = 0o40000, MSG_EXCEPT = 0o20000;
    if (sz < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
    if (badGuestPtr(bytes, ptr, 8 + Math.max(0, sz))) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; }
    // MSG_COPY reads the message at an INDEX and leaves it in the queue. it is only meaningful
    // without waiting (there is nothing to wait for) and cannot be combined with a type filter.
    if (flg & MSG_COPY) {
      if (!(flg & IPC_NOWAIT) || (flg & MSG_EXCEPT)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
      const rc = sysvMsgCall({ op: 'copy', id, mtype: mtyp.toString(), maxlen: sz, flg,
                               uid: creds.euid, gid: creds.egid });
      if (rc[0] !== 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(rc[0]))); return; }
      const ctype = (BigInt(rc[4] >>> 0) << 32n) | BigInt(rc[3] >>> 0);
      const ccopied = Math.min(rc[2], sz);
      view.setBigInt64(ptr + GUEST_MEM_BASE, BigInt.asIntN(64, ctype), true);
      if (ccopied > 0) bytes.set(new Uint8Array(shmArena, rc[1], ccopied), ptr + GUEST_MEM_BASE + 8);
      cpu.setReg(0, BigInt(ccopied));
      return;
    }
    let slept = false;
    for (;;) {
      const r = sysvMsgCall({ op: 'rcv', id, mtype: mtyp.toString(), maxlen: sz, flg,
                              uid: creds.euid, gid: creds.egid });
      if (r[0] === -22 && slept) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-43))); return; } // EIDRM
      if (r[0] === 0) {
        const off = r[1], len = r[2];
        const type = (BigInt(r[4] >>> 0) << 32n) | BigInt(r[3] >>> 0);
        const copied = Math.min(len, sz); // MSG_NOERROR truncates rather than failing
        view.setBigInt64(ptr + GUEST_MEM_BASE, BigInt.asIntN(64, type), true);
        if (copied > 0) bytes.set(new Uint8Array(shmArena, off, copied), ptr + GUEST_MEM_BASE + 8);
        sysvMsgCall({ op: 'free', off, len });
        cpu.setReg(0, BigInt(copied));
        return;
      }
      if (r[0] !== -11) { cpu.setReg(0, BigInt.asUintN(64, BigInt(r[0]))); return; }
      if (flg & IPC_NOWAIT) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-42))); return; }     // ENOMSG
      if (pollAndQueueSignals()) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-4))); return; } // EINTR
      slept = true;
      waitIo(readIoGeneration(), 20);
    }
  }

  // msgctl(msqid, cmd, buf)
  const id = Number(BigInt.asIntN(32, cpu.getReg(0)));
  const cmd = Number(cpu.getReg(1));
  const buf = Number(cpu.getReg(2));
  const IPC_RMID = 0, IPC_SET = 1, IPC_STAT = 2, IPC_INFO = 3;
  const MSG_STAT = 11, MSG_INFO = 12, MSG_STAT_ANY = 13;
  if (cmd === IPC_INFO || cmd === MSG_INFO) {
    // struct msginfo: 7 ints then a ushort. IPC_INFO reports the system LIMITS; MSG_INFO reuses
    // the same struct to report current USAGE, which is why they share a case.
    if (badGuestPtr(bytes, buf, 32)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; }
    const r = sysvMsgCall({ op: 'info' });
    const base = buf + GUEST_MEM_BASE;
    bytes.fill(0, base, base + 32);
    const usage = cmd === MSG_INFO;
    view.setInt32(base + 0, usage ? r[3] : MSGPOOL, true);   // msgpool: bytes in use / pool size
    view.setInt32(base + 4, usage ? r[2] : 0, true);         // msgmap: messages in use / map size
    view.setInt32(base + 8, MSGMAX, true);                   // msgmax
    view.setInt32(base + 12, r[4], true);                    // msgmnb
    view.setInt32(base + 16, MSGMNI, true);                  // msgmni
    view.setInt32(base + 20, 16, true);                      // msgssz
    view.setInt32(base + 24, usage ? r[2] : MSGTQL, true);   // msgtql
    view.setUint16(base + 28, 0xffff, true);                 // msgseg
    // both return the highest index in the queue array, which is one less than the count.
    cpu.setReg(0, BigInt(Math.max(0, r[1] - 1)));
    return;
  }
  if (cmd === MSG_STAT || cmd === MSG_STAT_ANY) {
    if (badGuestPtr(bytes, buf, MSQID_DS_SIZE)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; }
    const r = sysvMsgCall({ op: 'statidx', id, uid: creds.euid, gid: creds.egid });
    if (r[0] < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(r[0]))); return; }
    writeMsqidDs(view, buf, r);
    cpu.setReg(0, BigInt(r[13])); // the real id of the queue at that index
    return;
  }
  if (cmd === IPC_RMID) {
    const r = sysvMsgCall({ op: 'rm', id, uid: creds.euid, gid: creds.egid });
    cpu.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
    return;
  }
  if (cmd === IPC_STAT) {
    if (badGuestPtr(bytes, buf, MSQID_DS_SIZE)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; }
    const r = sysvMsgCall({ op: 'stat', id, uid: creds.euid, gid: creds.egid });
    if (r[0] < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(r[0]))); return; }
    writeMsqidDs(view, buf, r);
    cpu.setReg(0, 0n);
    return;
  }
  if (cmd === IPC_SET) {
    if (badGuestPtr(bytes, buf, MSQID_DS_SIZE)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; }
    const base = buf + GUEST_MEM_BASE;
    // ipc64_perm: key@0 uid@4 gid@8 cuid@12 cgid@16 mode@20, then the queue fields from 48.
    const r = sysvMsgCall({ op: 'set', id, uid: creds.euid, gid: creds.egid,
                            newUid: view.getUint32(base + 4, true),
                            newGid: view.getUint32(base + 8, true),
                            mode: view.getUint32(base + 20, true),
                            qbytes: Number(view.getBigUint64(base + 88, true)) });
    cpu.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
    return;
  }
  cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); // EINVAL: an ipc command we do not implement
}

// struct msqid_ds on aarch64: ipc64_perm (48 bytes: key, uid, gid, cuid, cgid, mode, __pad1,
// seq, __pad2, __unused1, __unused2) then msg_stime/rtime/ctime (8 each), __msg_cbytes,
// msg_qnum, msg_qbytes, msg_lspid, msg_lrpid.
const MSQID_DS_SIZE = 120;

function writeMsqidDs(view: DataView, buf: number, r: number[]): void {
  const base = buf + GUEST_MEM_BASE;
  new Uint8Array(memory.buffer).fill(0, base, base + MSQID_DS_SIZE);
  view.setInt32(base + 0, r[1], true);       // key
  view.setUint32(base + 4, r[2], true);      // uid
  view.setUint32(base + 8, r[3], true);      // gid
  view.setUint32(base + 12, r[2], true);     // cuid
  view.setUint32(base + 16, r[3], true);     // cgid
  view.setUint32(base + 20, r[4], true);     // mode
  view.setBigUint64(base + 48, BigInt(r[10]), true); // msg_stime
  view.setBigUint64(base + 56, BigInt(r[11]), true); // msg_rtime
  view.setBigUint64(base + 64, BigInt(r[12]), true); // msg_ctime
  view.setBigUint64(base + 72, BigInt(r[5]), true);  // __msg_cbytes
  view.setBigUint64(base + 80, BigInt(r[6]), true);  // msg_qnum
  view.setBigUint64(base + 88, BigInt(r[7]), true);  // msg_qbytes
  view.setInt32(base + 96, r[8], true);              // msg_lspid
  view.setInt32(base + 100, r[9], true);             // msg_lrpid
}

// linux's SysV message-queue limits, as /proc/sys/kernel reports them.
const MSGMAX = 8192;   // the largest mtext a single message may carry
const MSGMNI = 32000;  // how many queues may exist
const MSGPOOL = 1024;  // the (long obsolete) pool size, in kilobytes
const MSGTQL = 32000;  // how many messages may exist system-wide

/** one round trip to the kernel's semaphore registry. */
function sysvSemCall(msg: Record<string, unknown>): number[] {
  Atomics.store(myMailbox, MBOX_READY, 0);
  (self as any).postMessage({ type: 'sysv_sem', ...msg });
  awaitMailbox('sysv_sem');
  const out: number[] = [];
  // 11: the stat reply is the widest -- status plus ten fields of semid_ds.
  for (let i = 0; i < 11; i++) out.push(Atomics.load(myMailbox, MBOX_VALUE0 + i));
  return out;
}

/** the text of /proc/sysvipc/<which>. the registries live in the kernel, so this is a round trip. */
function sysvipcListing(which: 'sem' | 'msg' | 'shm'): string {
  const r = sysvSemCall({ op: 'proclist', which });
  if (r[0] !== 0) return '';
  const off = r[1], len = r[2];
  const text = new TextDecoder().decode(new Uint8Array(shmArena, off, len).slice());
  sysvSemCall({ op: 'free', off, len });
  return text;
}

// struct semid64_ds on aarch64: ipc64_perm (48 bytes) then sem_otime, sem_ctime, sem_nsems and
// two unused longs.
const SEMID_DS_SIZE = 88;
// struct sembuf is three shorts, packed: sem_num, sem_op, sem_flg.
const SEMBUF_SIZE = 6;

// linux's semaphore limits, as /proc/sys/kernel/sem reports them.
const SEMMSL = 32000, SEMMNS = 1024000000, SEMOPM = 500, SEMMNI = 32000;
const SEMVMX = 32767, SEMAEM = SEMVMX, SEMUME = SEMOPM, SEMUSZ = 20;

function writeSemidDs(view: DataView, buf: number, r: number[]): void {
  const base = buf + GUEST_MEM_BASE;
  new Uint8Array(memory.buffer).fill(0, base, base + SEMID_DS_SIZE);
  view.setInt32(base + 0, r[1], true);        // key
  view.setUint32(base + 4, r[2], true);       // uid
  view.setUint32(base + 8, r[3], true);       // gid
  view.setUint32(base + 12, r[9], true);      // cuid
  view.setUint32(base + 16, r[10], true);     // cgid
  view.setUint32(base + 20, r[4], true);      // mode
  view.setBigUint64(base + 48, BigInt(r[6] >>> 0), true);  // sem_otime
  view.setBigUint64(base + 56, BigInt(r[7] >>> 0), true);  // sem_ctime
  view.setBigUint64(base + 64, BigInt(r[5] >>> 0), true);  // sem_nsems
}

/**
 * semget / semctl / semtimedop / semop.
 *
 * the set itself lives in the kernel, because semop is atomic across the whole array and only a
 * serial view can decide whether a call's operations all fit. this side owns the WAITING: the
 * kernel only ever answers about the state right now, and a semop that cannot proceed comes back
 * EAGAIN for this loop to retry -- which is also what makes it interruptible by a signal.
 */
function handleSysvSem(nr: number, cpu: CpuState, view: DataView): void {
  const creds = syscalls.getCreds();
  const uid = creds.euid, gid = creds.egid;
  const bytes = new Uint8Array(memory.buffer);
  const fail = (e: number) => cpu.setReg(0, BigInt.asUintN(64, BigInt(e)));

  if (nr === 190) { // semget(key, nsems, semflg)
    const r = sysvSemCall({ op: 'get', key: Number(BigInt.asIntN(32, cpu.getReg(0))),
                            nsems: Number(BigInt.asIntN(32, cpu.getReg(1))),
                            flg: Number(cpu.getReg(2)), uid, gid });
    fail(r[0]);
    return;
  }

  if (nr === 192 || nr === 193) { // semtimedop / semop(semid, sops, nsops, timeout)
    const id = Number(BigInt.asIntN(32, cpu.getReg(0)));
    const ptr = Number(cpu.getReg(1));
    const nsops = Number(cpu.getReg(2));
    if (nsops === 0) { fail(-22); return; }                     // EINVAL
    if (nsops > SEMOPM) { fail(-7); return; }                   // E2BIG
    if (!guestAccessible(ptr, nsops * SEMBUF_SIZE, false)) { fail(-14); return; }
    let deadline = Infinity;
    if (nr === 192) {
      const tsp = Number(cpu.getReg(3));
      if (tsp !== 0) {
        if (!guestAccessible(tsp, 16, false)) { fail(-14); return; }
        const sec = view.getBigInt64(tsp + GUEST_MEM_BASE, true);
        const nsec = view.getBigInt64(tsp + GUEST_MEM_BASE + 8, true);
        if (sec < 0n || nsec < 0n || nsec >= 1000000000n) { fail(-22); return; }
        deadline = Date.now() + Number(sec) * 1000 + Number(nsec) / 1e6;
      }
    }
    const ops: number[] = [];
    for (let i = 0; i < nsops; i++) {
      const o = ptr + GUEST_MEM_BASE + i * SEMBUF_SIZE;
      ops.push(view.getUint16(o, true), view.getInt16(o + 2, true), view.getInt16(o + 4, true));
    }
    let waiting = 0;
    for (;;) {
      const r = sysvSemCall({ op: 'op', id, ops, wait: waiting, uid, gid });
      if (r[0] === 0) { cpu.setReg(0, 0n); return; }
      // a set removed while we were asleep on it is EIDRM; an id that never named one is EINVAL.
      if (r[0] === -22 && waiting) { fail(-43); return; }
      if (r[0] !== -11) { fail(r[0]); return; }
      if (r[1] === 1) { fail(-11); return; }                    // the blocking op said IPC_NOWAIT
      if (Date.now() >= deadline) {
        sysvSemCall({ op: 'unwait', id });
        fail(-11);                                              // EAGAIN: the timeout ran out
        return;
      }
      if (pollAndQueueSignals()) { sysvSemCall({ op: 'unwait', id }); fail(-4); return; } // EINTR
      // register as a waiter BEFORE parking, not after. semctl(GETNCNT) counts the processes
      // sleeping on a semaphore, and anything watching for one to fall asleep -- ltp does exactly
      // that -- would see it sleeping a whole polling interval before it was counted.
      if (!waiting) { waiting = 1; continue; }
      waitIo(readIoGeneration(), 20);
    }
  }

  // semctl(semid, semnum, cmd, arg)
  const id = Number(BigInt.asIntN(32, cpu.getReg(0)));
  const num = Number(BigInt.asIntN(32, cpu.getReg(1)));
  // some libcs or the caller's own IPC_64 flag on the command word; no semctl command uses that
  // bit, so masking it costs nothing and keeps a glibc guest working.
  const cmd = Number(BigInt.asIntN(32, cpu.getReg(2))) & ~0x100;
  const arg = cpu.getReg(3);
  const IPC_RMID = 0, IPC_SET = 1, IPC_STAT = 2, IPC_INFO = 3;
  const GETPID = 11, GETVAL = 12, GETALL = 13, GETNCNT = 14, GETZCNT = 15, SETVAL = 16, SETALL = 17;
  const SEM_STAT = 18, SEM_INFO = 19, SEM_STAT_ANY = 20;

  if (cmd === IPC_INFO || cmd === SEM_INFO) {
    // struct seminfo is ten ints. IPC_INFO reports the system LIMITS; SEM_INFO reuses the same
    // struct to report current USAGE, which is why they share a case.
    const buf = Number(arg);
    if (!guestAccessible(buf, 40, true)) { fail(-14); return; }
    const r = sysvSemCall({ op: 'info' });
    const base = buf + GUEST_MEM_BASE;
    const usage = cmd === SEM_INFO;
    view.setInt32(base + 0, usage ? r[1] : SEMMNS, true);   // semmap: sets in use / map size
    view.setInt32(base + 4, SEMMNI, true);                  // semmni
    view.setInt32(base + 8, usage ? r[2] : SEMMNS, true);   // semmns: semaphores in use / limit
    view.setInt32(base + 12, usage ? r[1] : SEMMNS, true);  // semmnu
    view.setInt32(base + 16, SEMMSL, true);                 // semmsl
    view.setInt32(base + 20, SEMOPM, true);                 // semopm
    view.setInt32(base + 24, SEMUME, true);                 // semume
    view.setInt32(base + 28, usage ? r[1] : SEMUSZ, true);  // semusz
    view.setInt32(base + 32, SEMVMX, true);                 // semvmx
    view.setInt32(base + 36, usage ? r[2] : SEMAEM, true);  // semaem
    // both return the highest index in the array of sets, which is one less than the count.
    cpu.setReg(0, BigInt(Math.max(0, r[1] - 1)));
    return;
  }

  if (cmd === IPC_STAT || cmd === SEM_STAT || cmd === SEM_STAT_ANY) {
    const buf = Number(arg);
    if (!guestAccessible(buf, SEMID_DS_SIZE, true)) { fail(-14); return; }
    const byIndex = cmd !== IPC_STAT;
    const r = sysvSemCall({ op: byIndex ? 'statidx' : 'stat', id, uid, gid });
    if (r[0] < 0) { fail(r[0]); return; }
    writeSemidDs(view, buf, r);
    // SEM_STAT reads by index and hands back the real id; IPC_STAT returns zero.
    cpu.setReg(0, byIndex ? BigInt(r[8]) : 0n);
    return;
  }

  if (cmd === IPC_SET) {
    const buf = Number(arg);
    if (!guestAccessible(buf, SEMID_DS_SIZE, false)) { fail(-14); return; }
    const base = buf + GUEST_MEM_BASE;
    const r = sysvSemCall({ op: 'set', id, uid, gid,
                            nuid: view.getUint32(base + 4, true),
                            ngid: view.getUint32(base + 8, true),
                            mode: view.getUint32(base + 20, true) });
    fail(r[0]);
    return;
  }

  if (cmd === IPC_RMID) { fail(sysvSemCall({ op: 'rm', id, uid, gid })[0]); return; }

  if (cmd === GETALL) {
    const buf = Number(arg);
    const r = sysvSemCall({ op: 'getall', id, uid, gid });
    if (r[0] < 0) { fail(r[0]); return; }
    const off = r[1], len = r[2];
    if (!guestAccessible(buf, len, true)) { sysvSemCall({ op: 'free', off, len }); fail(-14); return; }
    bytes.set(new Uint8Array(shmArena, off, len), buf + GUEST_MEM_BASE);
    sysvSemCall({ op: 'free', off, len });
    cpu.setReg(0, 0n);
    return;
  }

  if (cmd === SETALL) {
    const buf = Number(arg);
    // the caller's array is nsems unsigned shorts, and only the set knows how many that is.
    const st = sysvSemCall({ op: 'nsems', id });
    if (st[0] < 0) { fail(st[0]); return; }
    const nsems = st[0];
    if (!guestAccessible(buf, nsems * 2, false)) { fail(-14); return; }
    const vals: number[] = [];
    for (let i = 0; i < nsems; i++) vals.push(view.getUint16(buf + GUEST_MEM_BASE + i * 2, true));
    fail(sysvSemCall({ op: 'setall', id, vals, uid, gid })[0]);
    return;
  }

  if (cmd === SETVAL) {
    fail(sysvSemCall({ op: 'setval', id, num, val: Number(BigInt.asIntN(32, arg)), uid, gid })[0]);
    return;
  }

  if (cmd === GETVAL || cmd === GETPID || cmd === GETNCNT || cmd === GETZCNT) {
    const what = cmd === GETVAL ? 'val' : cmd === GETPID ? 'pid' : cmd === GETNCNT ? 'ncnt' : 'zcnt';
    fail(sysvSemCall({ op: 'getone', id, num, what, uid, gid })[0]);
    return;
  }

  fail(-22); // EINVAL: an ipc command we do not implement
}

function sysvShmCall(msg: Record<string, unknown>): number[] {
  Atomics.store(myMailbox, MBOX_READY, 0);
  (self as any).postMessage({ type: 'sysv_shm', ...msg });
  awaitMailbox('sysv_shm');
  const out: number[] = [];
  // 13: the stat reply is twelve fields, and SHM_STAT adds the segment's real id on the end.
  for (let i = 0; i < 13; i++) out.push(Atomics.load(myMailbox, MBOX_VALUE0 + i));
  return out;
}

/**
 * shmget / shmctl / shmat / shmdt.
 *
 * The segment itself is an arena block keyed "sysvshm:<id>", which is the same thing a MAP_SHARED
 * mapping is -- so attaching one is registering a shared mapping, and the coherence that already
 * keeps two processes' MAP_SHARED views in step keeps two attachers in step for free.
 */
function handleSysvShm(nr: number, cpu: CpuState, view: DataView): void {
  const creds = syscalls.getCreds();
  if (nr === 194) { // shmget(key, size, shmflg)
    // there is no hugetlbfs here, and a segment that quietly came back in 4 KiB pages would be a
    // lie to a caller who asked for huge ones. linux answers EINVAL when the kernel has no
    // hugetlb support, which is exactly our situation.
    const SHM_HUGETLB = 0o4000;
    if (Number(BigInt.asIntN(32, cpu.getReg(2))) & SHM_HUGETLB) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
    const r = sysvShmCall({ op: 'get', key: Number(BigInt.asIntN(32, cpu.getReg(0))),
                            size: Number(cpu.getReg(1)), flg: Number(cpu.getReg(2)),
                            uid: creds.euid, gid: creds.egid });
    cpu.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
    return;
  }

  if (nr === 196) { // shmat(shmid, shmaddr, shmflg)
    const id = Number(BigInt.asIntN(32, cpu.getReg(0)));
    const SHM_RDONLY = 0o10000, SHM_RND = 0o20000, SHMLBA = 4096;
    const flg = Number(BigInt.asIntN(32, cpu.getReg(2)));
    // an address the caller names must sit on a SHMLBA boundary. SHM_RND says round it down to one
    // instead of refusing, and rounding to zero is refused either way: shmat(0) means "you choose".
    let want = Number(cpu.getReg(1));
    if (want !== 0) {
      if (flg & SHM_RND) want -= want % SHMLBA;
      else if (want % SHMLBA !== 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
      if (want === 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
    }
    const r = sysvShmCall({ op: 'attach', id, rdonly: (flg & SHM_RDONLY) ? 1 : 0,
                            uid: creds.euid, gid: creds.egid });
    if (r[0] < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(r[0]))); return; }
    const size = r[0];
    const prot = (flg & SHM_RDONLY) ? 0x1 : 0x3;
    const addr = syscalls.attachShared(memory, size, prot, `sysvshm:${id}`, want);
    if (addr < 0) {
      sysvShmCall({ op: 'detach', id }); // the attach count must not outlive the failure
      cpu.setReg(0, BigInt.asUintN(64, BigInt(addr)));
      return;
    }
    sysvAttached.set(addr, id);
    cpu.setReg(0, BigInt(addr));
    return;
  }

  if (nr === 197) { // shmdt(shmaddr)
    const addr = Number(cpu.getReg(0));
    const id = sysvAttached.get(addr);
    const region = syscalls.sharedRegionAt(addr);
    // shmdt names an address, and one that was never the START of an attach is EINVAL.
    if (id === undefined || !region) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); return; }
    releaseSharedMaps(addr, addr + region.len);
    syscalls.detachShared(addr, region.len);
    sysvAttached.delete(addr);
    sysvShmCall({ op: 'detach', id });
    cpu.setReg(0, 0n);
    return;
  }

  // shmctl(shmid, cmd, buf)
  const id = Number(BigInt.asIntN(32, cpu.getReg(0)));
  // read as ints: a command of -1 taken 64 bits wide is a number no comparison matches, and the
  // fall-through answer was the wrong errno.
  const cmd = Number(BigInt.asIntN(32, cpu.getReg(1))) & ~0x100;
  const buf = Number(cpu.getReg(2));
  const shmCreds = syscalls.getCreds();
  const IPC_RMID = 0, IPC_SET = 1, IPC_STAT = 2, SHM_LOCK = 11, SHM_UNLOCK = 12;
  if (cmd === IPC_RMID) {
    const r = sysvShmCall({ op: 'rm', id, uid: shmCreds.euid, gid: shmCreds.egid });
    cpu.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
    return;
  }
  const IPC_INFO = 3, SHM_INFO = 14;
  if (cmd === IPC_INFO || cmd === SHM_INFO) {
    // struct shminfo is four unsigned longs then shmall; struct shm_info is an int followed by
    // five unsigned longs. IPC_INFO reports the LIMITS, SHM_INFO the current USAGE, and both
    // return the highest index in the segment array.
    if (buf === 0 || syscalls.ptrFaults(new Uint8Array(memory.buffer), buf, 48)) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; // EFAULT
    }
    const r = sysvShmCall({ op: 'info' });
    const b = buf + GUEST_MEM_BASE;
    new Uint8Array(memory.buffer).fill(0, b, b + 48);
    if (cmd === IPC_INFO) {
      const w64 = (lo: number, hi: number) => (BigInt(hi >>> 0) << 32n) | BigInt(lo >>> 0);
      view.setBigUint64(b + 0, w64(r[4], r[5]), true);                        // shmmax
      view.setBigUint64(b + 8, 1n, true);                                    // shmmin
      view.setBigUint64(b + 16, BigInt(r[3]), true);                         // shmmni
      view.setBigUint64(b + 24, BigInt(r[3]), true);                         // shmseg
      view.setBigUint64(b + 32, w64(r[6], r[7]), true);                      // shmall
    } else {
      view.setInt32(b + 0, r[1], true);                                      // used_ids
      view.setBigUint64(b + 8, BigInt(r[2]), true);                          // shm_tot (pages)
      view.setBigUint64(b + 16, BigInt(r[2]), true);                         // shm_rss
      view.setBigUint64(b + 24, 0n, true);                                   // shm_swp
      view.setBigUint64(b + 32, 0n, true);                                   // swap_attempts
      view.setBigUint64(b + 40, 0n, true);                                   // swap_successes
    }
    cpu.setReg(0, BigInt(Math.max(0, r[1] - 1)));
    return;
  }
  // SHM_LOCK / SHM_UNLOCK: pin a segment. everything already lives in the shared arena, so the
  // observable part is the permission -- and it is the owner's call, not anyone's.
  if (cmd === SHM_LOCK || cmd === SHM_UNLOCK) {
    const r = sysvShmCall({ op: 'lock', id, on: cmd === SHM_LOCK ? 1 : 0,
                            uid: shmCreds.euid, gid: shmCreds.egid });
    cpu.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
    return;
  }
  const SHM_STAT = 13, SHM_STAT_ANY = 15;
  if (cmd === IPC_STAT || cmd === SHM_STAT || cmd === SHM_STAT_ANY) {
    if (buf === 0 || syscalls.ptrFaults(new Uint8Array(memory.buffer), buf, 112)) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; // EFAULT
    }
    const byIdx = cmd !== IPC_STAT;
    const r = sysvShmCall({ op: byIdx ? 'statidx' : 'stat', id, uid: shmCreds.euid, gid: shmCreds.egid });
    if (r[0] < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(r[0]))); return; }
    const [, key, uid, gid, mode, size, cpid, lpid, nattch, atime, dtime, ctime] = r;
    // struct shmid64_ds on aarch64: ipc64_perm (48 bytes) then segsz, atime, dtime, ctime,
    // cpid, lpid, nattch.
    const b = buf + GUEST_MEM_BASE;
    for (let i = 0; i < 112; i += 8) view.setBigUint64(b + i, 0n, true);
    view.setInt32(b + 0, key, true);        // shm_perm.key
    view.setUint32(b + 4, uid, true);       // uid
    view.setUint32(b + 8, gid, true);       // gid
    view.setUint32(b + 12, uid, true);      // cuid
    view.setUint32(b + 16, gid, true);      // cgid
    view.setUint32(b + 20, mode, true);     // mode
    view.setBigUint64(b + 48, BigInt(size), true);
    view.setBigInt64(b + 56, BigInt(atime), true);
    view.setBigInt64(b + 64, BigInt(dtime), true);
    view.setBigInt64(b + 72, BigInt(ctime), true);
    view.setInt32(b + 80, cpid, true);
    view.setInt32(b + 84, lpid, true);
    view.setBigUint64(b + 88, BigInt(nattch), true);
    // SHM_STAT reads by index and hands back the real id; IPC_STAT returns zero.
    cpu.setReg(0, byIdx ? BigInt(r[12]) : 0n);
    return;
  }
  if (cmd === IPC_SET) {
    if (buf === 0 || syscalls.ptrFaults(new Uint8Array(memory.buffer), buf, 112)) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return;
    }
    const b = buf + GUEST_MEM_BASE;
    // the credentials that decide whether this is allowed are the CALLER's; the uid and gid in
    // the buffer are the new owner it is asking for. sending only the latter made the check ask
    // "does the new owner own it", which any caller can answer yes to by naming root.
    const r = sysvShmCall({ op: 'set', id, uid: shmCreds.euid, gid: shmCreds.egid,
                            mode: view.getUint32(b + 20, true),
                            newUid: view.getUint32(b + 4, true), newGid: view.getUint32(b + 8, true) });
    cpu.setReg(0, BigInt.asUintN(64, BigInt(r[0])));
    return;
  }
  cpu.setReg(0, BigInt.asUintN(64, BigInt(-22))); // EINVAL: a command we do not implement
}

function handleKernelPipeIO(nr: number, cpu: CpuState, view: DataView): void {
  const fd = Number(cpu.getReg(0));
  const pipeInfo = kernelPipes.get(fd)!;
  const ring = new PipeRing(shmArena, pipeInfo.pipeOff);

  // a pipe end goes one way. reading the write end or writing the read end is EBADF, not a
  // no-op and certainly not a success -- the descriptor simply was not opened for that.
  if ((nr === 63 || nr === 65) && pipeInfo.mode === 'write') { cpu.setReg(0, BigInt.asUintN(64, BigInt(-9))); return; }
  if ((nr === 64 || nr === 66) && pipeInfo.mode === 'read') { cpu.setReg(0, BigInt.asUintN(64, BigInt(-9))); return; }

  if (nr === 57) {
    // close. dropping the ring reference is only half of it: the DESCRIPTOR has to go back to the
    // fd table too. a pipe end is registered there like every other fd (that is what makes
    // lowest-free-first allocation and RLIMIT_NOFILE work), and leaving it registered meant a
    // closed pipe still occupied its number forever -- fd numbers climbed instead of being reused
    // and pipe() died with EMFILE after ~500 cycles. any program that pipes in a loop hits that:
    // a shell running pipelines, a build system, a server spawning workers.
    // it also left the descriptor half-alive. with the entry gone from kernelPipes but still in the
    // fd table, a later read or write on it stopped being intercepted and fell through to the
    // legacy in-memory pipe path, quietly succeeding against an empty buffer instead of EBADF.
    pipeRef(pipeInfo.pipeOff, pipeInfo.mode, -1);
    kernelPipes.delete(fd);
    syscalls.unregisterFd(fd);
    cpu.setReg(0, 0n);
    return;
  }

  if (nr === 63) {
    // read — interruptible: a pending signal (armed timer, incoming kill/term) breaks the
    // block with EINTR instead of parking forever, so a pipe-blocked process stays reapable.
    const buf = Number(cpu.getReg(1));
    const count = Number(cpu.getReg(2));
    if (count > 0 && badGuestPtr(new Uint8Array(memory.buffer), buf, count)) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; // -EFAULT
    }
    const dst = new Uint8Array(count);
    // O_NONBLOCK on the READ end was never honoured (only the write end checked it), so a program
    // that polls a pipe rather than committing to it blocked forever on the first empty one. that
    // is the ordinary shape of a supervisor loop -- and of ltp's kill02, which reads four pipes in
    // a row precisely to find out which of them stayed silent.
    if (syscalls.isNonBlockingFd(fd)) {
      const n = ring.tryRead(dst);
      if (n === 0 && ring.writersOpen()) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-11))); return; } // -EAGAIN
      if (n > 0) new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, n).set(dst.subarray(0, n));
      cpu.setReg(0, BigInt(n)); // 0 here is a real EOF: empty ring, no writers left
      return;
    }
    const n = ring.read(dst, interruptibleWaiter);
    if (n === -1) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-4))); return; } // -EINTR
    if (n > 0) {
      new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, n).set(dst.subarray(0, n));
    }
    cpu.setReg(0, BigInt(n));
    return;
  }

  if (nr === 64) {
    // write
    const buf = Number(cpu.getReg(1));
    const count = Number(cpu.getReg(2));
    if (count > 0 && badGuestPtr(new Uint8Array(memory.buffer), buf, count)) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; // -EFAULT
    }
    const src = new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, count);
    if (syscalls.isNonBlockingFd(fd)) {
      const n = ring.write(src);
      if (n > 0) notePipeReadable(pipeInfo.pipeOff);
      if (n === -1) { syscalls.raiseSignal(13); cpu.setReg(0, BigInt.asUintN(64, BigInt(-32))); return; } // SIGPIPE + EPIPE
      if (n === 0 && count > 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-11))); return; } // EAGAIN
      cpu.setReg(0, BigInt(n));
      return;
    }
    // blocking fd: park until the whole buffer is written (a slow reader must not lose data).
    const n = ring.writeAll(src, interruptibleWaiter);
    if (n > 0) notePipeReadable(pipeInfo.pipeOff);
    if (n === -1) { syscalls.raiseSignal(13); cpu.setReg(0, BigInt.asUintN(64, BigInt(-32))); return; } // SIGPIPE + EPIPE
    if (n === -2) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-4))); return; }         // EINTR
    cpu.setReg(0, BigInt(n));
    return;
  }

  if (nr === 65) {
    // readv -- scatter read into an iovec array. this was simply missing, so readv on a pipe fell
    // through to the per-process handler, which knows nothing about kernel pipes and answered
    // EBADF. readv is not exotic: busybox's od reads its stdin that way, and so does anything
    // that wants a header and a body in one call.
    const iovAddr = Number(cpu.getReg(1));
    const iovcnt = Number(cpu.getReg(2));
    const bytes = new Uint8Array(memory.buffer);
    if (iovcnt > 0 && badGuestPtr(bytes, iovAddr, iovcnt * 16)) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; // -EFAULT
    }
    // the segments are one read: linux fills them in order from a single fetch, so gather the
    // total first and hand out the bytes, rather than reading the pipe once per segment (which
    // would block again on an empty pipe after the first segment was satisfied).
    let total = 0;
    const segs: Array<{ addr: number; len: number }> = [];
    for (let i = 0; i < iovcnt; i++) {
      const base = Number(view.getBigUint64(iovAddr + i * 16 + GUEST_MEM_BASE, true));
      const len = Number(view.getBigUint64(iovAddr + i * 16 + 8 + GUEST_MEM_BASE, true));
      if (len === 0) continue;
      if (badGuestPtr(bytes, base, len)) {
        if (total === 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; } // -EFAULT
        break;
      }
      segs.push({ addr: base, len });
      total += len;
    }
    if (total === 0) { cpu.setReg(0, 0n); return; }

    const dst = new Uint8Array(total);
    let n: number;
    if (syscalls.isNonBlockingFd(fd)) {
      n = ring.tryRead(dst);
      if (n === 0 && ring.writersOpen()) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-11))); return; } // -EAGAIN
    } else {
      n = ring.read(dst, interruptibleWaiter);
      if (n === -1) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-4))); return; } // -EINTR
    }
    let off = 0;
    for (const sg of segs) {
      if (off >= n) break;
      const take = Math.min(sg.len, n - off);
      bytes.set(dst.subarray(off, off + take), sg.addr + GUEST_MEM_BASE);
      off += take;
    }
    cpu.setReg(0, BigInt(n));
    return;
  }

  if (nr === 66) {
    // writev — gather write from iovec array
    const iovAddr = Number(cpu.getReg(1));
    const iovcnt = Number(cpu.getReg(2));
    let totalWritten = 0;
    const bytes = new Uint8Array(memory.buffer);
    if (iovcnt > 0 && badGuestPtr(bytes, iovAddr, iovcnt * 16)) {
      cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; // -EFAULT
    }
    for (let i = 0; i < iovcnt; i++) {
      const base = view.getBigUint64(iovAddr + i * 16 + GUEST_MEM_BASE, true);
      const len = view.getBigUint64(iovAddr + i * 16 + 8 + GUEST_MEM_BASE, true);
      if (len === 0n) continue;
      if (badGuestPtr(bytes, Number(base), Number(len))) {
        if (totalWritten === 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; } // -EFAULT
        break;
      }
      const src = bytes.subarray(Number(base) + GUEST_MEM_BASE, Number(base) + GUEST_MEM_BASE + Number(len));
      const n = ring.write(src);
      if (n > 0) notePipeReadable(pipeInfo.pipeOff);
      if (n === -1) {
        if (totalWritten === 0) {
          syscalls.raiseSignal(13); // SIGPIPE: default disposition kills the writer
          cpu.setReg(0, BigInt.asUintN(64, BigInt(-32))); // -EPIPE
          return;
        }
        break;
      }
      totalWritten += n;
      if (n < Number(len)) break; // short write
    }
    cpu.setReg(0, BigInt(totalWritten));
    return;
  }
}

function handleBlockedRead(): void {
  if (stdinRing) {
    // Block until data appears in stdin ring
    // The main thread writes data and notifies via Atomics
    const stdinRingIdx = Atomics.load(procTable, (procSlotOffset(mySlot) / 4) + 7);
    const ringI32 = new Int32Array(kernelSAB, stdinRingOffset(stdinRingIdx), 4);
    while (stdinRing.empty && !stdinRing.writerClosed) {
      const wp = Atomics.load(ringI32, 0);
      // bound the wait so an armed timer / incoming kill is observed within <=1s even
      // though it notifies the wait futex, not the stdin ring.
      const rem = syscalls.timerRemainingMs();
      sleepingWait(ringI32, 0, wp, rem > 0 ? Math.min(1000, rem) : 1000);
      // any pending signal interrupts the read; the run loop delivers it (handler or exit).
      if (pollAndQueueSignals()) return;
    }
  }
}

// read an entire file from the guest FS (bridge) synchronously, in <=slot-sized chunks.
// used to load the dynamic interpreter (ld.so) and execve targets. slow for big files
// (many round-trips) but correct; a bulk-transfer channel is a later optimization.
/** base64 in blocks, not one character at a time -- the naive loop is what made this expensive. */
function bytesToBase64Fast(u: Uint8Array): string {
  let out = '';
  const BLOCK = 0x8000;
  for (let i = 0; i < u.length; i += BLOCK) {
    out += String.fromCharCode.apply(null, u.subarray(i, i + BLOCK) as unknown as number[]);
  }
  return btoa(out);
}

function readGuestFile(path: string): Uint8Array {
  const fdh = { fd: -1 };
  handleBridgeCall('fs_open', { path, flags: 0, mode: 0 }, (r) => { fdh.fd = r.ret ?? -1; });
  if (fdh.fd < 0) {
    const e = new Error(`readGuestFile: cannot open ${path} (ret ${fdh.fd})`) as Error & { errno: number };
    e.errno = -fdh.fd || 2; // ENOENT default
    throw e;
  }
  const chunks: Uint8Array[] = [];
  let total = 0;
  const CHUNK = BRIDGE_DATA_MAX; // raw bytes per read; the slot no longer holds base64
  try {
    for (;;) {
      const h = { data: new Uint8Array(0) };
      handleBridgeCall('fs_read', { fd: fdh.fd, count: CHUNK }, (r) => {
        if (r.bytes && r.bytes.length) h.data = r.bytes;
      });
      if (h.data.length === 0) break;
      chunks.push(h.data); total += h.data.length;
      if (h.data.length < CHUNK) break;
    }
  } finally {
    handleBridgeCall('fs_close', { fd: fdh.fd }, () => {});
  }
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

// synchronous offset-read of a NATIVE (bridge-backed) file, chunked to fit the bridge response slot.
// this is what makes file-backed mmap + pread work on real files -> the dynamic loader can map shared
// libraries from NativeFS. returns the bytes actually read (short at EOF).
function readNativeFileAt(fd: number, offset: number, count: number): Uint8Array {
  const CHUNK = 12288; // raw bytes per read; base64 fits the 16KB response slot
  const out = new Uint8Array(count);
  let got = 0;
  while (got < count) {
    const want = Math.min(CHUNK, count - got);
    const h = { data: new Uint8Array(0) };
    handleBridgeCall('fs_pread', { fd, offset: offset + got, count: want }, (r) => {
      if (r && typeof r.data === 'string' && r.data.length) {
        const bin = atob(r.data);
        const u = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
        h.data = u;
      }
    });
    if (h.data.length === 0) break; // EOF or error
    out.set(h.data, got);
    got += h.data.length;
    if (h.data.length < want) break; // short read -> EOF
  }
  return got === count ? out : out.subarray(0, got);
}

function handleBridgeCall(method: string, args: Record<string, unknown>, apply: (resp: any) => void): void {
  // arm the mailbox BEFORE announcing the request. the other order loses the wakeup: the kernel
  // can answer in the window between the postMessage and the store, setting READY=1, and the store
  // then wipes that answer out -- leaving this loop parked on an Atomics.wait for a response that
  // already came and will never come again. the process stays RUNNING with a live worker and
  // simply stops, which is exactly what it looked like.
  Atomics.store(myMailbox, MBOX_READY, 0);
  (self as any).postMessage({ type: 'bridge_call', id: 0, method, args });

  // bounded slices, for the same reason as awaitMailbox: an unbounded wait here turns any lost
  // notify into a process that stops forever while still looking alive. the loop re-reads the flag,
  // so a reply that arrived without waking us is picked up on the next pass.
  const bridgeStart = performance.now();
  let bridgeWarned = false;
  while (Atomics.load(myMailbox, MBOX_READY) === 0) {
    Atomics.wait(myMailbox, MBOX_READY, 0, 100);
    // Woke up — check if it's a signal rather than a bridge response
    if (Atomics.load(myMailbox, MBOX_READY) === 0) {
      if (!bridgeWarned && performance.now() - bridgeStart > 20000) {
        bridgeWarned = true;
        (self as any).postMessage({ type: 'worker_error',
          detail: `pid ${myPid} has waited 20s for bridge '${method}'` });
      }
      // not a bridge response: a signal may have arrived instead, and a bridge call that would
      // otherwise wait forever has to notice one that is meant to interrupt it.
      const queued = sigqDrain(sigQueue, mySlot);
      for (const q of queued) syscalls.raiseSignal(q.signo, { pid: q.pid, uid: q.uid, code: q.code, value: q.value });
      if (queued.some((q) => q.signo === 2)) {
        // Check if this process has a custom SIGINT handler (shell does)
        if (syscalls.hasSignalHandler(2)) {
          // Deliver signal through the normal dispatch path — queue it and return
          // The bridge call will get a fake EINTR response
          syscalls.raiseSignal(2);
          // Return EINTR to the caller
          apply({ ret: -4 }); // EINTR
          return;
        }
        // Default action: terminate
        postExit(130); // 128 + SIGINT(2)
        throw new Error('SIGINT');
      }
    }
  }

  // Read response from the bridge response area in kernel SAB
  const respOffset = BRIDGE_RESP_OFFSET + mySlot * BRIDGE_RESP_SLOT_SIZE;
  const respView = new DataView(kernelSAB, respOffset, BRIDGE_RESP_SLOT_SIZE);

  // Response layout: [ret:i32][metaCount:u16][dataLen:u32][meta: metaCount×f64][data: dataLen bytes]
  const ret = respView.getInt32(0, true);
  const metaCount = respView.getUint16(4, true);
  const dataLen = respView.getUint32(6, true);
  const dataOffset = 10 + metaCount * 8;

  const resp: any = { ret };

  // Read meta values
  if (metaCount > 0) {
    resp._meta = new Array(metaCount);
    for (let i = 0; i < metaCount; i++) {
      resp._meta[i] = respView.getFloat64(10 + i * 8, true);
    }
  }

  // Read data.
  //
  // The payload is ALREADY here, as bytes, in shared memory. It used to be turned into a base64
  // string first -- built one character at a time with `str +=`, then base64'd, then decoded again
  // by whoever wanted it, then copied out character by character. Three full passes and a
  // quadratic-ish string build over every byte the guest ever reads from a file. exec reads a whole
  // binary that way, which is where most of a 40 ms process creation was going.
  //
  // So: one memcpy into `bytes`, and `data` stays available as a LAZY base64 view for the callers
  // that genuinely want text (readlink, getdents' json) and pay for it only when they ask.
  if (dataLen > 0) {
    const raw = new Uint8Array(dataLen);
    raw.set(new Uint8Array(kernelSAB, respOffset + dataOffset, dataLen));
    resp.bytes = raw;
    Object.defineProperty(resp, 'data', {
      configurable: true, enumerable: true,
      get: () => bytesToBase64Fast(raw),
    });
  }

  // same guard as the synchronous syscall path: a bridge result copied to a bad guest buffer
  // faults with EFAULT rather than crashing the worker.
  try {
    apply(resp);
  } catch (e) {
    if (e instanceof RangeError) cpu.setReg(0, BigInt.asUintN(64, BigInt(-14)));
    else throw e;
  }
}

// ── I/O helpers ─────────────────────────────────────────────────────────────

function readStdin(count: number): Uint8Array | null {
  if (!stdinRing) return null;
  if (stdinRing.empty && !stdinRing.writerClosed) return null; // signal blocked_read

  const buf = new Uint8Array(count);
  const n = stdinRing.tryRead(buf);
  if (n === 0 && stdinRing.writerClosed) {
    return new Uint8Array(0); // EOF
  }
  if (n === 0) return null; // no data, will block
  return buf.subarray(0, n);
}

function waitStdin(timeoutMs: number): 'data' | 'timeout' {
  if (!stdinRing) return 'timeout';
  if (!stdinRing.empty || stdinRing.writerClosed) return 'data';
  const stdinRingIdx = Atomics.load(procTable, (procSlotOffset(mySlot) / 4) + 7);
  if (stdinRingIdx < 0) return 'timeout';
  const ringI32 = new Int32Array(kernelSAB, stdinRingOffset(stdinRingIdx), 4);
  const wp = Atomics.load(ringI32, 0);
  const result = timeoutMs < 0
    ? sleepingWait(ringI32, 0, wp)
    : sleepingWait(ringI32, 0, wp, timeoutMs);
  return result === 'timed-out' ? 'timeout' : 'data';
}

function postWrite(fd: number, data: Uint8Array): void {
  (self as any).postMessage({ type: 'write', fd, data: data.slice() });
}

/** a thread of a group whose leader is someone else. its exit ends a thread, not a process, so it
 *  must not release anything the group still shares. */
function isNonLeaderThread(): boolean {
  return threaded && syscalls.tgid !== 0 && syscalls.tgid !== syscalls.currentPid;
}

/** let a vfork parent go: this process has exec'd or is exiting, which is what it was waiting for. */
function releaseVforkParent(): void {
  const base = procSlotOffset(mySlot) / 4;
  if (Atomics.exchange(procTable, base + PROC_VFORK, 0) === 1) {
    Atomics.notify(procTable, base + PROC_VFORK);
  }
}

function postExit(code: number, group = false): void {
  releaseVforkParent();
  publishSyscall(-6);   // exiting: handing back native descriptors
  stamp('exit'); reportStages();
  // hand back the native descriptors FIRST, while this Worker still owns its mailbox.
  //
  // announcing the exit is what lets the kernel park this Worker for reuse, and a parked Worker is
  // handed to the next process with a NEW slot and a NEW onmessage handler. anything the dying
  // tenant says after that point is answered into the new slot's mailbox, while it waits on its
  // own -- forever. it never returns to the event loop, so the incoming init never runs either:
  // the Worker is simultaneously "free" and permanently stuck. a process with a couple of open
  // files usually got away with it; ltp's fcntl12 exits holding 1024 and wedged the whole run.
  //
  // a non-leader THREAD is not a process ending: its siblings are still running and the
  // descriptors belong to all of them. closing them here shut the group's files out from under
  // the threads still using them.
  if (!pendingClone && !isNonLeaderThread()) {
    try {
      syscalls.closeNativeFds((fd) => handleBridgeCall('fs_close', { fd }, () => {}));
      syscalls.releaseEventfds();
    } catch {}
  }
  // CLONE_CHILD_CLEARTID: a thread exiting must zero its clear_child_tid word and futex-wake it, so
  // a joiner blocked in pthread_join (futex-waiting on that word) is released.
  if (threaded && clearChildTid) {
    try {
      const t = futexRef(clearChildTid);
      Atomics.store(t.i32, t.idx, 0);
      // the waiters are in the kernel's futex queue, not parked on the word itself, so the wake
      // has to go through it -- notifying the word directly would reach nobody.
      futexLock(procTable);
      futexWake(procTable, t.keyLo, t.keyHi, 0xffffffff, 0x7fffffff);
      futexUnlock(procTable);
    } catch {}
    clearChildTid = 0;
  }
  publishSyscall(-7);   // exiting: descriptors done, releasing pipes
  // Close all open kernel pipe fds before exiting
  for (const [, info] of kernelPipes) pipeRef(info.pipeOff, info.mode, -1);
  kernelPipes.clear();
  // and the unix sockets, whose rings this process holds a reference to whether it closed the
  // descriptors or not.
  for (const fd of [...unixSocks.keys()]) closeUnixSock(fd);

  // a threaded process reports its touched high-water so the kernel, when it parks this shared memory
  // as the reusable spare, knows how much of it the next tenant must zero (only the used span is stale).
  publishCpuTime();
  publishSyscall(-8);   // exiting: about to announce it
  const high = threaded ? Number(syscalls.getMmapBase()) + GUEST_MEM_BASE : 0;
  // report our memory size too: a wasm memory never shrinks, so the kernel uses it to decide whether
  // this Worker is cheap enough to keep in the idle pool or should be retired (see POOL_MAX_MEM_BYTES).
  (self as any).postMessage({ type: 'exit', pid: myPid, code, high, group,
                              memBytes: memory.buffer.byteLength });
}
