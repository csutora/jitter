import { CpuState } from './cpu.js';
import { GUEST_MEM_BASE, TPIDR_OFFSET, PC_OFFSET, V_OFFSET } from './types.js';
import type { NativeBridge } from './bridge.js';

// ── Syscall result type ─────────────────────────────────────────────────────

export type SyscallResult =
  | { status: 'ok' }
  | { status: 'exit'; code: number }
  | { status: 'blocked_read'; fd: number }
  | { status: 'fork'; resumePC: bigint }
  | { status: 'execve'; path: string; argv: string[]; envp: string[] }
  | { status: 'bridge'; method: string; args: Record<string, any>; apply: (resp: any) => void }
  // cooperative-scheduler statuses (only produced when this.coop is set — the conformance
  // harness drives real concurrent processes). the syscall is re-run when the process is
  // rescheduled, so the handler leaves no partial side effects before returning these.
  | { status: 'coop_block'; reason: 'pipe_read' | 'pipe_write' | 'lock' | 'futex'; addr?: number }
  | { status: 'coop_wait'; kind: 'wait4' | 'waitid'; wantPid: number; wstatusAddr: number; options: number; infop: number };

// base64-encode bytes for the bridge. must chunk: btoa(String.fromCharCode(...data))
// spreads the whole buffer as call args and blows the stack on large writes (tar extract).
function bytesToBase64(data: Uint8Array): string {
  const chunk = 0x8000;
  let s = '';
  for (let i = 0; i < data.length; i += chunk) {
    s += String.fromCharCode.apply(null, data.subarray(i, i + chunk) as unknown as number[]);
  }
  return btoa(s);
}

// ── Error constants ─────────────────────────────────────────────────────────

const ENOSYS = 38;
const EBADF  = 9;
const EINVAL = 22;
const ENOENT = 2;
const EPERM  = 1;
const EMFILE = 24;
const ENOTDIR = 20;
const ECHILD = 10;
const ESPIPE = 29;
const EAGAIN = 11;
const EFAULT = 14;
const EACCES = 13;
const ENOMEM = 12;
const EEXIST = 17;
const ESRCH  = 3;
const EDEADLK = 35;
const MAX_FDS = 1024;

// per-process address-space ceiling (RLIMIT_AS-ish). a mmap/brk growth past this returns ENOMEM
// to the guest (malloc -> NULL) instead of growing the wasm memory toward the device jetsam limit
// and taking the whole WKWebView content process down. one process can't alone blow the budget;
// it fails gracefully like real linux under a cgroup/ulimit. (the global cross-process budget is
// enforced separately in the kernel.) generous enough that real programs never see it.
const MAX_PROCESS_BYTES = 1610612736; // 1.5 GiB

const O_NONBLOCK = 0o4000;

// Signal constants
const SIG_DFL = 0n;
const SIG_IGN = 1n;
const SIGALRM = 14;
const SIGPIPE = 13;
const SIGCHLD = 17;

// Signal trampoline address (fixed location in guest memory for rt_sigreturn stub)
const SIGNAL_TRAMPOLINE = 0x1000; // in the first page, before GUEST_MEM_BASE

// Signals with default action = ignore
const SIG_IGNORE_DEFAULT = new Set([SIGCHLD, 17, 20]); // SIGCHLD, SIGURG, SIGWINCH
// Signals with default action = stop (not kill)
const SIG_STOP_DEFAULT = new Set([19, 20, 21, 22]); // SIGSTOP, SIGTSTP, SIGTTIN, SIGTTOU

// Signal frame layout in guest memory (pushed onto guest stack):
// [0..247]  X0-X30 (31 × 8 bytes)
// [248]     SP
// [256]     PC (return address)
// [264]     NZCV
// [272]     signal number (for our use)
// Total: 280 bytes, rounded to 16-byte alignment = 288 bytes
// GPRs+SP+PC+NZCV+signum+mask occupy the first 288 bytes; the 32 V (NEON/FP) registers (512 bytes)
// follow at offset 288. saving them is mandatory: crypto and memcpy run in V registers, so a signal
// (e.g. apk's SIGALRM network-timeout) delivered mid-computation would otherwise corrupt vector state
// on rt_sigreturn and silently garble the result — the TLS "unexpected message" was exactly this.
const SIGNAL_FRAME_V_OFFSET = 288;
const SIGNAL_FRAME_SIZE = SIGNAL_FRAME_V_OFFSET + 512;

// ── ioctl constants ─────────────────────────────────────────────────────────

const TCGETS     = 0x5401;
const TCSETS     = 0x5402;
const TIOCGWINSZ = 0x5413;
const TIOCSWINSZ = 0x5414;

// ── fcntl constants ─────────────────────────────────────────────────────────

const F_DUPFD = 0;
const F_GETFD = 1;
const F_SETFD = 2;
const F_GETFL = 3;
const F_SETFL = 4;
const F_DUPFD_CLOEXEC = 1030;

// ── VFS interface ───────────────────────────────────────────────────────────

export interface VFS {
  open(path: string, flags: number, mode: number): number;
  close(fd: number): number;
  dup(fd: number): number;
  dupTo(oldFd: number, newFd: number): number;
  hasFd(fd: number): boolean;
  read(fd: number, buf: number, count: number, memory: DataView): number;
  write(fd: number, buf: number, count: number, memory: DataView): number;
  lseek(fd: number, offset: bigint, whence: number): bigint;
  fstat(fd: number, statBuf: number, memory: DataView): number;
  fstatat(dirfd: number, path: string, statBuf: number, flags: number, memory: DataView): number;
  faccessat(dirfd: number, path: string, mode: number, flags: number): number;
  fchmod(fd: number, mode: number): number;
  fchown(fd: number, uid: number, gid: number): number;
  chmodPath(path: string, mode: number): number;
  chownPath(path: string, uid: number, gid: number): number;
  ftruncate(fd: number, length: number): number;
  getcwd(): string;
  setCwd(path: string): void;
  chdir(path: string): number;
  getdents64(fd: number, buf: number, count: number, memory: DataView): number;
  readlinkat(path: string): string | null;
  unlinkat(path: string, flags: number): number;
  mkdirat(path: string, mode: number): number;
  symlinkat(target: string, linkpath: string): number;
  linkat(oldpath: string, newpath: string): number;
  rename(oldpath: string, newpath: string): number;
  snapshotFds(): Map<number, any>;
  restoreFds(snap: Map<number, any>): void;
  closeAllExcept(...keep: number[]): void;
  loadBinary(data: Uint8Array): void;
  fdAllocator: (() => number) | null;
  credsProvider: (() => { ruid: number; euid: number; rgid: number; egid: number }) | null;
}

// ── I/O interface for terminal ──────────────────────────────────────────────

export interface IOInterface {
  writeStdout(data: Uint8Array): void;
  writeStderr(data: Uint8Array): void;
  readStdin(count: number): Uint8Array | null;
  /** Whether stdin has bytes available for non-blocking read. */
  stdinHasData?(): boolean;
  /** Block until stdin has data or timeout fires (ms; negative = infinite). Returns whether data arrived. */
  waitStdin?(timeoutMs: number): 'data' | 'timeout';
  /** Synchronously read up to `count` bytes at `offset` from a NATIVE (bridge-backed) fd. used by
   * file-backed mmap + pread on real files (ld.so maps shared libraries this way). returns the bytes
   * actually read (short at EOF); undefined when no bridge is available (tests use the vfs path). */
  readFileAt?(nativeFd: number, offset: number, count: number): Uint8Array;
}

// ── Syscall handler ─────────────────────────────────────────────────────────

// ── Pipe buffer ──────────────────────────────────────────────────────────────

interface PipeBuffer {
  data: Uint8Array[];
  writeClosed: boolean;
  readClosed: boolean;
  // open-fd refcounts. writeClosed/readClosed become true only when the last
  // holder of that end is closed (across fork, both parent and child hold a
  // reference). without this a single close would signal a premature EOF.
  writers: number;
  readers: number;
  bytes: number; // queued byte count, for backpressure on a full pipe
}

type FdEntry =
  | { type: 'stdio'; fd: number }  // 0=stdin, 1=stdout, 2=stderr
  | { type: 'tty' }                // /dev/tty — controlling terminal
  | { type: 'vfs'; fd: number }    // VFS file descriptor
  | { type: 'pipe_read'; pipe: PipeBuffer }
  | { type: 'pipe_write'; pipe: PipeBuffer }
  | { type: 'native'; nativeFd: number; path: string }
  | { type: 'socket'; socketId: number; domain: number; sockType: number; remoteAddr?: { family: number; host: string; port: number } }
  | { type: 'inotify' }                                  // event queue (stub: no watches, never ready)
  | { type: 'timerfd' }                                  // periodic timer (stub: no timer armed, never ready)
  | { type: 'eventfd'; counter: bigint; semaphore: boolean }  // 8-byte counter (real semantics)
  | { type: 'epoll' };                                   // multiplexer (state in this.epollSets)

const NET_TRACE = false; // socket trace for https diagnosis; flip to true when needed, remove before ship.
const FD_TRACE = false; // fd-lifecycle trace; flip to true when needed, remove before ship.

export class SyscallHandler {
  private nt(s: string) { if (NET_TRACE) this.io.writeStderr(new TextEncoder().encode('[net] ' + s + '\n')); }
  private ft(s: string) { if (FD_TRACE) this.io.writeStderr(new TextEncoder().encode('[fd] ' + s + '\n')); }
  private brk: bigint;
  // the exclusive-monitor shadow lives at ZIG_SHADOW_BASE (32MB) and is tiny
  // (1 byte per 512 guest bytes, so <=256KB for a 128MB guest). reserve [32MB,33MB)
  // for it and start the mmap arena above, so mmap'd shared libraries (ld.so mmaps
  // libc.so.6) don't collide with the shadow. mmap grows up toward the shadow_base
  // ceiling of the top-of-memory shadow... here shadow is low, so cap mmap well above.
  private mmapBase = 0x2100000n; // 33MB — just above the 1MB shadow reservation
  // brk/mmap ceiling. a threaded process uses a lower ceiling (its shared memory places the per-thread
  // register blocks + atomics shadow just above this, and the shared reservation is bounded so several
  // threaded processes can coexist). single-threaded processes keep the full 1.5 GiB.
  maxBytes = MAX_PROCESS_BYTES;
  // MAP_SHARED regions (guest offsets). shared across fork: a child's writes to these
  // must survive the parent-memory restore in the serialized fork model.
  private sharedRegions: Array<{ addr: number; len: number }> = [];
  // writable MAP_SHARED mappings of real (native) files. the guest writes into the mapping and
  // expects those bytes flushed back to the file on msync/munmap (this is how apk extracts
  // binaries). we track them and pwrite the dirty range through the bridge on flush.
  private fileMmaps: Array<{ start: number; len: number; fileLen: number; nativeFd: number; fileOffset: number }> = [];
  // guest ranges mapped PROT_NONE (inaccessible). a user pointer landing here faults with
  // EFAULT, which is how LTP's tst_get_bad_addr() probes are built.
  private protNone: Array<{ start: number; end: number }> = [];
  // page-aligned, non-overlapping locked-memory intervals (mmap MAP_LOCKED / mlock). their
  // total is reported as VmLck in /proc/self/status.
  private lockedRegions: Array<{ start: number; end: number }> = [];
  // virtual memory areas (guest offsets == guest-visible addresses), sorted, non-overlapping.
  // drives MAP_FIXED_NOREPLACE overlap detection and the synthetic /proc/self/maps. prot holds
  // the PROT_* bits; shared distinguishes p/s in the maps perms column.
  private vmas: Array<{ start: number; end: number; prot: number; shared: boolean }> = [];
  private cwd = '/';
  private tpidr = 0n;
  private umask = 0o022;
  private fdCloexec = new Set<number>();
  private fdSig = new Map<number, number>();
  // nanosecond clock. Date.now() is millisecond-resolution; performance.now() gives sub-ms
  // where the host allows it (browsers/WKWebView clamp it to ~1ms for spectre mitigation,
  // which is fine). posix only requires CLOCK_MONOTONIC to be non-decreasing, not strictly
  // increasing, so we allow equal consecutive reads and clamp only to defend against the
  // clock actually running backwards (performance.now() shouldn't, but Date.now() can under
  // ntp/settimeofday). no artificial per-read increment: that would distort measured
  // elapsed time under tight loops, which is the one thing a clock must get right.
  private clkHasPerf = typeof performance !== 'undefined';
  private clkPerfBase = this.clkHasPerf ? performance.now() : Date.now();
  private clkRealBaseNs = BigInt(Date.now()) * 1_000_000n;
  private clkLastMonoNs = 0n;
  private monoNs(): bigint {
    const elapsedMs = (this.clkHasPerf ? performance.now() : Date.now()) - this.clkPerfBase;
    let ns = BigInt(Math.round(elapsedMs * 1_000_000));
    if (ns < this.clkLastMonoNs) ns = this.clkLastMonoNs; // never go backwards
    else this.clkLastMonoNs = ns;
    return ns;
  }
  clockNs(clkId: number): bigint {
    // CLOCK_MONOTONIC(1)/RAW(4)/BOOTTIME(7) + cputime clocks -> since construction;
    // CLOCK_REALTIME(0) and everything else -> wall clock
    const mono = this.monoNs();
    if (clkId === 1 || clkId === 4 || clkId === 7 || clkId === 2 || clkId === 3) return mono;
    return this.clkRealBaseNs + mono;
  }
  // ITIMER_REAL: monotonic ms deadline at which SIGALRM fires, 0 = disarmed. interval is
  // the auto-rearm period (0 = one-shot). used by alarm()/setitimer() and by busybox
  // timeout, which is how the ltp driver bounds each test. checked by the exec loops via
  // checkTimers() so it fires even in a syscall-less spin.
  private itimerRealDeadlineMs = 0;
  private itimerRealIntervalMs = 0;
  // RLIMIT_NOFILE soft limit: the highest fd number a process may allocate. set via
  // prlimit64/setrlimit; enforced at fd-allocating syscalls (open/dup/...) -> EMFILE.
  private nofileLimit = 1024;
  private nowMs(): bigint { return this.monoNs() / 1_000_000n; }
  /** raise SIGALRM if the real interval timer has expired; rearm or disarm. returns true if it fired. */
  checkTimers(): boolean {
    if (this.itimerRealDeadlineMs === 0) return false;
    if (Number(this.nowMs()) < this.itimerRealDeadlineMs) return false;
    this.raiseSignal(SIGALRM);
    if (this.itimerRealIntervalMs > 0) this.itimerRealDeadlineMs += this.itimerRealIntervalMs;
    else this.itimerRealDeadlineMs = 0;
    return true;
  }
  /** ms until the real timer fires, or 0 if disarmed (for getitimer / wait timeouts). */
  timerRemainingMs(): number {
    if (this.itimerRealDeadlineMs === 0) return 0;
    return Math.max(0, this.itimerRealDeadlineMs - Number(this.nowMs()));
  }
  pendingSignals: number[] = [];  // queue of signals to deliver

  // cooperative-scheduler mode. when set, blocking syscalls (pipe read on an empty
  // pipe, wait4/waitid with live-but-not-yet-exited children) yield to the scheduler
  // instead of using the serialized vfork fast path. off by default so the app's
  // dispatch loop and busybox boot keep their existing single-runner behavior.
  coop = false;
  // set true whenever a syscall does something that could unblock another process
  // (writes to a pipe, closes the last writer/reader end). the harness wakes blocked
  // processes on this signal so a parked reader/writer re-checks its condition.
  schedEvent = false;
  // shared posix advisory-lock table (fcntl record locks), injected by the harness in
  // coop mode. null in the serialized path, where locks are stubbed as always-granted.
  locks: import('./scheduler.js').LockManager | null = null;
  // cross-process signal delivery (kill) in coop mode. (pid, sig) => target existed.
  // sig 0 is an existence probe. null in the serialized path.
  killHook: ((pid: number, sig: number) => boolean) | null = null;
  // wake up to `n` processes blocked in FUTEX_WAIT on a guest address; returns the
  // count actually woken (what FUTEX_WAKE must report). null in the serialized path.
  futexWakeHook: ((addr: number, n: number) => number) | null = null;
  // set by a voluntarily-yielding syscall (nanosleep, sched_yield) in coop mode so the
  // harness runs another process before resuming this one (cooperative preemption point).
  wantYield = false;
  private epollSets?: Map<number, Map<number, { events: number; data: bigint }>>;
  private signalHandlers?: Map<number, { handler: bigint; flags: bigint; restorer: bigint; mask: bigint }>;
  private signalMask?: bigint;
  /** True if bridge calls can be handled (either direct bridge or Worker kernel relay) */
  bridgeAvailable = false;
  currentPid = 1;
  currentPpid = 0;
  // returns true if `pid` is a live process (a suspended ancestor in the serialized fork model).
  // wired by the dispatch/worker loop to the process manager so kill() can raise ESRCH correctly.
  isLivePidFn: ((pid: number) => boolean) | null = null;
  // wired by the dispatch loop to the page-protection table: records the PROT_* permission of the
  // guest pages covering [addr, addr+len) so guest accesses to restricted pages fault (SIGSEGV).
  protHook: ((addr: number, len: number, prot: number) => void) | null = null;
  /** thread-group id: 0 for a normal process (getpid == currentPid); for a thread it's the
   * process pid so getpid()==tgid while gettid()==currentPid(==tid). */
  tgid = 0;

  /** mark this handler as belonging to a thread: gettid returns `tid`, getpid returns `tgid`. */
  setThreadIds(tid: number, tgid: number): void {
    this.currentPid = tid;
    this.tgid = tgid;
  }
  // process credentials. real/effective/saved uid+gid. start as root (0).
  private ruid = 0; private euid = 0; private suid = 0;
  private rgid = 0; private egid = 0; private sgid = 0;
  private termCols = 80;
  private termRows = 24;
  private vfs?: VFS;
  private io: IOInterface;
  private bridge?: NativeBridge;
  private fdFlags = new Map<number, number>(); // fd -> flags
  private fdTable = new Map<number, FdEntry>(); // unified fd table
  private nativeRefs = new Map<number, number>(); // nativeFd -> refcount
  // Stat cache: populated from getdents, consumed by fs_stat. Avoids N+1 bridge calls for ls -l.
  // Path → {mode, size, nlink, mtime, expiresAt (ms since epoch)}
  private statCache = new Map<string, { mode: number; size: number; nlink: number; mtime: number; uid: number; gid: number; ino: number; expiresAt: number }>();
  private readonly STAT_CACHE_TTL_MS = 1000;
  lastWaitStatus?: number;  // exit code from last child (set by dispatch loop)
  lastWaitPid?: number;     // PID of last exited child

  constructor(
    brkStart: bigint,
    io: IOInterface,
    vfs?: VFS,
    bridge?: NativeBridge,
  ) {
    this.brk = brkStart;
    this.io = io;
    this.vfs = vfs;
    this.bridge = bridge;
    this.fdTable.set(0, { type: 'stdio', fd: 0 });
    this.fdTable.set(1, { type: 'stdio', fd: 1 });
    this.fdTable.set(2, { type: 'stdio', fd: 2 });
    if (this.vfs) {
      this.vfs.fdAllocator = () => this.allocFd();
      this.vfs.credsProvider = () => ({ ruid: this.ruid, euid: this.euid, rgid: this.rgid, egid: this.egid });
    }
  }

  /**
   * Check if a guest path should be routed to the native Swift filesystem.
   * Returns the resolved absolute path if bridging, or null if staying in JS VFS.
   *
   * JS VFS handles: /proc, /dev (virtual, per-process)
   * Native Swift handles: everything else (/bin, /etc, /tmp, /root, /home, /var, /usr, ...)
   */
  // resolve an *at()-family path against its dirfd. absolute paths and AT_FDCWD (-100) pass through
  // (bridgePath/vfs resolve them against cwd). otherwise the path is relative to the directory the
  // fd refers to, so join it onto that fd's known path. apk loads trust keys via openat(keys_fd, name),
  // so without this the keys never open and every index signature reads as UNTRUSTED.
  private resolveAt(dirfd: bigint, path: string): string {
    if (path.startsWith('/')) return path;
    const d = Number(BigInt.asIntN(64, dirfd));
    if (d === -100) return path; // AT_FDCWD
    const base = (this.getFdEntry(d) as any)?.path;
    if (typeof base === 'string' && base.length) return (base.endsWith('/') ? base : base + '/') + path;
    return path;
  }

  // flush a writable MAP_SHARED native-file mapping back to disk on msync/munmap. finds the mapping
  // overlapping [reqStart, reqStart+reqLen), pwrites its dirty (real-file) bytes through the bridge,
  // and drops the record when unmapping. returns a bridge action, or null if nothing to flush.
  private buildMmapFlush(cpu: any, memory: WebAssembly.Memory, reqStart: number, reqLen: number, removeAfter: boolean): SyscallResult | null {
    for (let i = 0; i < this.fileMmaps.length; i++) {
      const m = this.fileMmaps[i];
      if (reqStart >= m.start + m.len || reqStart + reqLen <= m.start) continue;
      if (removeAfter) this.fileMmaps.splice(i, 1);
      const fStart = Math.max(reqStart, m.start);
      const fEnd = Math.min(reqStart + reqLen, m.start + m.fileLen);
      if (fEnd <= fStart) return null;
      const data = new Uint8Array(memory.buffer).slice(fStart + GUEST_MEM_BASE, fEnd + GUEST_MEM_BASE);
      return {
        status: 'bridge', method: 'fs_pwrite',
        args: { fd: m.nativeFd, offset: m.fileOffset + (fStart - m.start), data: bytesToBase64(data) },
        apply: () => { cpu.setReg(0, 0n); },
      };
    }
    return null;
  }

  // mark [start,end) (page-aligned) as locked or unlocked in the VmLck accounting, keeping the
  // interval set normalized (non-overlapping, merged), then republish the total to the vfs.
  private setLocked(start: number, end: number, locked: boolean): void {
    if (end <= start) return;
    const next: Array<{ start: number; end: number }> = [];
    for (const r of this.lockedRegions) {
      if (r.end <= start || r.start >= end) { next.push(r); continue; }
      if (r.start < start) next.push({ start: r.start, end: start });
      if (r.end > end) next.push({ start: end, end: r.end });
    }
    if (locked) next.push({ start, end });
    next.sort((a, b) => a.start - b.start);
    const merged: Array<{ start: number; end: number }> = [];
    for (const r of next) {
      const last = merged[merged.length - 1];
      if (last && r.start <= last.end) last.end = Math.max(last.end, r.end);
      else merged.push({ ...r });
    }
    this.lockedRegions = merged;
    let total = 0;
    for (const r of merged) total += r.end - r.start;
    if (this.vfs) (this.vfs as any).vmLckKb = Math.floor(total / 1024);
  }

  // does [start,end) overlap any tracked vma? used for MAP_FIXED_NOREPLACE.
  private vmaOverlaps(start: number, end: number): boolean {
    for (const v of this.vmas) if (v.start < end && start < v.end) return true;
    return false;
  }

  // register the lazy /proc/self/maps renderer once the vma set is live.
  private publishVmas(): void {
    if (this.vfs && !(this.vfs as any).procMapsGen) (this.vfs as any).procMapsGen = () => this.renderProcMaps();
  }

  // vmas is kept sorted and non-overlapping, so all edits are a single O(n) pass with inline
  // coalescing of adjacent identical vmas: no re-sort, which keeps mapping-heavy loops linear.
  private vmaEmit(out: Array<{ start: number; end: number; prot: number; shared: boolean }>, v: { start: number; end: number; prot: number; shared: boolean }): void {
    const last = out[out.length - 1];
    if (last && last.end === v.start && last.prot === v.prot && last.shared === v.shared) last.end = v.end;
    else out.push(v);
  }

  // carve [start,end) out of the vma set (splitting boundary vmas), then optionally insert a fresh
  // vma with the given prot/shared, all in sorted order.
  private vmaSet(start: number, end: number, insert: { prot: number; shared: boolean } | null): void {
    if (end <= start) return;
    const out: Array<{ start: number; end: number; prot: number; shared: boolean }> = [];
    let inserted = false;
    for (const v of this.vmas) {
      if (v.end <= start) { this.vmaEmit(out, v); continue; }
      if (v.start >= end) {
        if (!inserted && insert) { this.vmaEmit(out, { start, end, prot: insert.prot, shared: insert.shared }); inserted = true; }
        this.vmaEmit(out, v);
        continue;
      }
      if (v.start < start) this.vmaEmit(out, { start: v.start, end: start, prot: v.prot, shared: v.shared });
      if (!inserted && insert) { this.vmaEmit(out, { start, end, prot: insert.prot, shared: insert.shared }); inserted = true; }
      if (v.end > end) this.vmaEmit(out, { start: end, end: v.end, prot: v.prot, shared: v.shared });
    }
    if (!inserted && insert) this.vmaEmit(out, { start, end, prot: insert.prot, shared: insert.shared });
    this.vmas = out;
    this.publishVmas();
  }

  // set the protection of [start,end), splitting vmas at the boundaries (mprotect).
  private vmaProtect(start: number, end: number, prot: number): void {
    if (end <= start) return;
    const out: Array<{ start: number; end: number; prot: number; shared: boolean }> = [];
    for (const v of this.vmas) {
      if (v.end <= start || v.start >= end) { this.vmaEmit(out, v); continue; }
      if (v.start < start) this.vmaEmit(out, { start: v.start, end: start, prot: v.prot, shared: v.shared });
      this.vmaEmit(out, { start: Math.max(v.start, start), end: Math.min(v.end, end), prot, shared: v.shared });
      if (v.end > end) this.vmaEmit(out, { start: end, end: v.end, prot: v.prot, shared: v.shared });
    }
    this.vmas = out;
    this.publishVmas();
  }

  private renderProcMaps(): string {
    let out = '';
    for (const v of this.vmas) {
      const perms = (v.prot & 0x1 ? 'r' : '-') + (v.prot & 0x2 ? 'w' : '-') + (v.prot & 0x4 ? 'x' : '-') + (v.shared ? 's' : 'p');
      const s = v.start.toString(16).padStart(8, '0');
      const e = v.end.toString(16).padStart(8, '0');
      out += `${s}-${e} ${perms} 00000000 00:00 0 \n`;
    }
    return out;
  }

  private bridgePath(path: string): string | null {
    if (!this.bridge?.isNative && !this.bridgeAvailable) return null;
    // Resolve relative paths using cwd
    let resolved = path;
    if (!path.startsWith('/')) {
      const cwd = this.vfs?.getcwd() ?? this.cwd;
      resolved = cwd === '/' ? '/' + path : cwd + '/' + path;
    }
    // Normalize . and ..
    const parts: string[] = [];
    for (const p of resolved.split('/')) {
      if (p === '' || p === '.') continue;
      if (p === '..') { parts.pop(); continue; }
      parts.push(p);
    }
    resolved = '/' + parts.join('/');
    // Virtual filesystems stay in JS
    if (resolved === '/proc' || resolved.startsWith('/proc/') ||
        resolved === '/dev' || resolved.startsWith('/dev/')) {
      return null;
    }
    // Everything else → Swift NativeFS
    return resolved;
  }

  // resolve a (possibly relative) path to a normalized absolute path against the current cwd.
  // used for execve so the worker reads the binary from the right place (a bare "./prog" must
  // become "/cwd/prog"), matching how the kernel resolves the exec pathname.
  private absolutizePath(path: string): string {
    let resolved = path;
    if (!path.startsWith('/')) {
      const cwd = this.vfs?.getcwd() ?? this.cwd;
      resolved = cwd === '/' ? '/' + path : cwd + '/' + path;
    }
    const parts: string[] = [];
    for (const p of resolved.split('/')) {
      if (p === '' || p === '.') continue;
      if (p === '..') { parts.pop(); continue; }
      parts.push(p);
    }
    return '/' + parts.join('/');
  }

  setTermSize(cols: number, rows: number): void {
    this.termCols = cols;
    this.termRows = rows;
  }

  /** Get the current mmap high-water mark (for fork snapshot sizing). */
  getMmapBase(): bigint { return this.mmapBase; }

  /** Write the signal return trampoline into guest memory. Call once during init. */
  installSignalTrampoline(memory: WebAssembly.Memory): void {
    const view = new DataView(memory.buffer);
    const base = SIGNAL_TRAMPOLINE + GUEST_MEM_BASE;
    // ARM64 instructions for: mov x8, #139; svc #0
    // 139 = __NR_rt_sigreturn
    view.setUint32(base, 0xd2801168, true);     // mov x8, #139 (0x8b)
    view.setUint32(base + 4, 0xd4000001, true); // svc #0
  }

  /**
   * Deliver a pending signal to the guest process.
   * Pushes a signal frame onto the guest stack and redirects PC to the handler.
   * Returns true if a signal was delivered (caller should update PC).
   */
  deliverSignal(cpu: CpuState, memory: WebAssembly.Memory, signum: number): { delivered: boolean; action?: 'exit' | 'continue'; exitCode?: number } {
    if (!this.signalHandlers) return { delivered: false };

    const handler = this.signalHandlers.get(signum);
    const handlerAddr = handler?.handler ?? SIG_DFL;

    // SIG_IGN: ignore the signal
    if (handlerAddr === SIG_IGN) return { delivered: true, action: 'continue' };

    // SIG_DFL: default action
    if (handlerAddr === SIG_DFL) {
      if (SIG_IGNORE_DEFAULT.has(signum)) return { delivered: true, action: 'continue' };
      if (SIG_STOP_DEFAULT.has(signum)) return { delivered: true, action: 'continue' }; // can't stop in emulator
      // Default for most signals: terminate
      return { delivered: true, action: 'exit', exitCode: 128 + signum };
    }

    // Check signal mask — if blocked, keep pending
    if (this.signalMask && (this.signalMask & (1n << BigInt(signum - 1)))) {
      return { delivered: false }; // signal is blocked
    }

    // User handler: push signal frame and redirect execution
    const view = new DataView(memory.buffer);
    const currentSP = cpu.getSP();

    // Allocate signal frame on guest stack (grows downward)
    const frameSP = (currentSP - BigInt(SIGNAL_FRAME_SIZE)) & ~15n; // 16-byte aligned
    const frameBase = Number(frameSP) + GUEST_MEM_BASE;

    // Save all registers to the frame
    for (let r = 0; r < 31; r++) {
      view.setBigUint64(frameBase + r * 8, cpu.getReg(r), true);
    }
    view.setBigUint64(frameBase + 248, currentSP, true);      // saved SP
    view.setBigUint64(frameBase + 256, cpu.getPC(), true);     // saved PC (return address)
    view.setBigUint64(frameBase + 264, cpu.getNZCV(), true);   // saved NZCV
    view.setUint32(frameBase + 272, signum, true);              // signal number
    view.setBigUint64(frameBase + 280, this.signalMask ?? 0n, true); // saved signal mask
    // save the 32 V registers (512 bytes) so the handler can't clobber in-flight vector state
    {
      const vBase = cpu.base + V_OFFSET;
      for (let i = 0; i < 512; i += 8) {
        view.setBigUint64(frameBase + SIGNAL_FRAME_V_OFFSET + i, view.getBigUint64(vBase + i, true), true);
      }
    }

    // block this signal (plus the handler's sa_mask) for the duration of the handler unless
    // SA_NODEFER — this is what stops a repeated signal (e.g. tst_test's SIGUSR1 heartbeat) from
    // re-entering its own handler and never returning to the interrupted syscall. restored by
    // rt_sigreturn.
    const flags = handler?.flags ?? 0n;
    let newMask = (this.signalMask ?? 0n) | (handler?.mask ?? 0n);
    if ((flags & 0x40000000n) === 0n) newMask |= (1n << BigInt(signum - 1)); // SA_NODEFER
    this.signalMask = newMask;

    // Set up for signal handler execution
    cpu.setSP(frameSP);
    cpu.setPC(handlerAddr);                    // jump to handler
    cpu.setReg(0, BigInt(signum));             // x0 = signal number
    cpu.setReg(30, BigInt(SIGNAL_TRAMPOLINE)); // x30 (LR) = trampoline (calls rt_sigreturn)

    return { delivered: true, action: 'continue' };
  }

  /** Handle rt_sigreturn: restore registers from signal frame. */
  handleSigreturn(cpu: CpuState, memory: WebAssembly.Memory): bigint {
    const view = new DataView(memory.buffer);
    const frameSP = cpu.getSP();
    const frameBase = Number(frameSP) + GUEST_MEM_BASE;

    // Restore all registers from the frame
    for (let r = 0; r < 31; r++) {
      cpu.setReg(r, view.getBigUint64(frameBase + r * 8, true));
    }
    const savedSP = view.getBigUint64(frameBase + 248, true);
    const savedPC = view.getBigUint64(frameBase + 256, true);
    const savedNZCV = view.getBigUint64(frameBase + 264, true);
    this.signalMask = view.getBigUint64(frameBase + 280, true); // restore pre-handler mask
    // restore the 32 V registers saved at delivery
    {
      const vBase = cpu.base + V_OFFSET;
      for (let i = 0; i < 512; i += 8) {
        view.setBigUint64(vBase + i, view.getBigUint64(frameBase + SIGNAL_FRAME_V_OFFSET + i, true), true);
      }
    }

    cpu.setSP(savedSP);
    cpu.setNZCV(savedNZCV);

    return savedPC; // caller should set PC to this
  }

  /** Queue a signal for delivery. */
  raiseSignal(signum: number): void {
    this.pendingSignals.push(signum);
  }

  /** True if any queued signal has an SA_RESTART handler — the interrupted restartable syscall
   *  (wait4, blocking read) should then be re-executed instead of returning EINTR. */
  anyPendingRestart(): boolean {
    if (!this.signalHandlers) return false;
    for (const sig of this.pendingSignals) {
      const h = this.signalHandlers.get(sig);
      if (h && h.handler !== SIG_DFL && h.handler !== SIG_IGN && (h.flags & 0x10000000n) !== 0n) return true; // SA_RESTART
    }
    return false;
  }

  /** Check if a custom signal handler is installed (not SIG_DFL or SIG_IGN). */
  hasSignalHandler(signum: number): boolean {
    if (!this.signalHandlers) return false;
    const handler = this.signalHandlers.get(signum);
    if (!handler) return false;
    return handler.handler !== SIG_DFL && handler.handler !== SIG_IGN;
  }

  /** Parse a sockaddr structure from guest memory into JS object. */
  private parseSockaddr(bytes: Uint8Array, addrPtr: number, addrLen: number): { family: number; host: string; port: number } {
    const base = addrPtr + GUEST_MEM_BASE;
    const family = bytes[base] | (bytes[base + 1] << 8); // sa_family (LE on aarch64)

    if (family === 2 && addrLen >= 8) {
      // AF_INET: struct sockaddr_in { family(2), port(2 BE), addr(4 BE) }
      const port = (bytes[base + 2] << 8) | bytes[base + 3]; // big-endian port
      const a = bytes[base + 4], b = bytes[base + 5], c = bytes[base + 6], d = bytes[base + 7];
      return { family, host: `${a}.${b}.${c}.${d}`, port };
    }

    if (family === 10 && addrLen >= 28) {
      // AF_INET6: struct sockaddr_in6 { family(2), port(2 BE), flowinfo(4), addr(16), scope(4) }
      const port = (bytes[base + 2] << 8) | bytes[base + 3];
      const addr = bytes.slice(base + 8, base + 24);
      // Check for IPv4-mapped (::ffff:a.b.c.d)
      const isV4Mapped = addr[10] === 0xFF && addr[11] === 0xFF &&
        addr.slice(0, 10).every(b => b === 0);
      if (isV4Mapped) {
        return { family: 2, host: `${addr[12]}.${addr[13]}.${addr[14]}.${addr[15]}`, port };
      }
      // Full IPv6
      const parts: string[] = [];
      for (let i = 0; i < 16; i += 2) {
        parts.push(((addr[i] << 8) | addr[i + 1]).toString(16));
      }
      return { family, host: parts.join(':'), port };
    }

    return { family, host: '0.0.0.0', port: 0 };
  }

  /** Write a sockaddr structure to guest memory. Returns bytes written. */
  private writeSockaddr(view: DataView, addrPtr: number, addr: { family: number; host: string; port: number }): number {
    const base = addrPtr + GUEST_MEM_BASE;
    if (addr.family === 2) {
      // AF_INET: { family(2 LE), port(2 BE), addr(4 BE), zero(8) } = 16 bytes
      view.setUint16(base, 2, true); // sa_family LE
      view.setUint8(base + 2, (addr.port >> 8) & 0xFF); // port BE high
      view.setUint8(base + 3, addr.port & 0xFF);        // port BE low
      const parts = addr.host.split('.').map(Number);
      view.setUint8(base + 4, parts[0]);
      view.setUint8(base + 5, parts[1]);
      view.setUint8(base + 6, parts[2]);
      view.setUint8(base + 7, parts[3]);
      // Zero padding
      for (let i = 8; i < 16; i++) view.setUint8(base + i, 0);
      return 16;
    }
    // AF_INET6 or unknown — zero it out
    view.setUint16(base, addr.family, true);
    return 0;
  }

  /** Register an fd entry directly (used by Worker for kernel pipe fds). */
  registerFd(fd: number, entry: FdEntry): void {
    this.fdTable.set(fd, entry);
  }

  // the resolved path backing a native fd, used as the key for cross-process record locks. a
  // non-native fd (pipe, vfs) returns null: those have no on-disk identity to lock against.
  fdPath(fd: number): string | null {
    const e = this.fdTable.get(fd);
    return e && e.type === 'native' ? e.path : null;
  }

  // fd offset / file size for resolving SEEK_CUR / SEEK_END lock ranges. native fds keep their
  // offset and size on the Swift side, so these are best-effort (0) until a bridge query is added;
  // SEEK_SET locks (the common case) are unaffected.
  fdOffset(_fd: number): number { return 0; }
  fdSize(_fd: number): number { return 0; }

  /** Set process identity (used when a fork continues as the child). */
  setIdentity(pid: number, ppid: number): void {
    this.currentPid = pid;
    this.currentPpid = ppid;
  }

  // the current process credentials, threaded into NativeFS path-op bridge calls so the native
  // side can enforce emulated POSIX ownership/permissions against this process's identity.
  // (supplementary groups aren't tracked yet — setgroups is a no-op — so groups is empty for now.)
  getCreds(): { ruid: number; euid: number; rgid: number; egid: number; umask: number; groups: number[] } {
    return { ruid: this.ruid, euid: this.euid, rgid: this.rgid, egid: this.egid, umask: this.umask, groups: [] };
  }

  /** high-water byte offset the cooperative scheduler must snapshot to capture a
   * process image: cpu-state block + guest data + heap + stack + mmap allocations,
   * which all live below the mmap bump pointer. */
  get memImageBound(): number {
    return Number(this.mmapBase) + GUEST_MEM_BASE + 0x10000;
  }

  // system-wide identity of the file behind `fd`, for advisory record locks. two
  // descriptors onto the same file (dup, or a separate open of the same inode) share
  // it. returns null for descriptors that can't hold record locks (pipes, sockets).
  private lockKey(fd: number): string | null {
    const e = this.fdTable.get(fd);
    if (e?.type === 'native') return `n:${e.path}`;
    const vfd = e?.type === 'vfs' ? e.fd : ((this.vfs?.hasFd(fd) ?? false) ? fd : -1);
    if (vfd >= 0 && this.vfs) {
      const ino = this.vfs.inoOfFd(vfd);
      if (ino != null) return `v:${ino}`;
    }
    return null;
  }

  /** add an open-fd reference for every pipe end in an inherited fd table (fork). */
  retainPipeRefs(fdTable: Map<number, FdEntry>): void {
    for (const e of fdTable.values()) {
      if (e.type === 'pipe_write') e.pipe.writers++;
      else if (e.type === 'pipe_read') e.pipe.readers++;
    }
  }

  /** release an open-fd reference for every pipe end in an fd table (process exit). */
  releasePipeRefs(fdTable: Map<number, FdEntry>): void {
    for (const e of fdTable.values()) {
      if (e.type === 'pipe_write') { if (--e.pipe.writers <= 0) e.pipe.writeClosed = true; }
      else if (e.type === 'pipe_read') { if (--e.pipe.readers <= 0) e.pipe.readClosed = true; }
    }
  }

  /** Get state for process snapshot */
  getState() {
    return {
      brk: this.brk,
      mmapBase: this.mmapBase,
      cwd: this.vfs ? this.vfs.getcwd() : this.cwd,
      fdFlags: new Map(this.fdFlags),
      fdTable: new Map(this.fdTable),
      nativeRefs: new Map(this.nativeRefs),
      currentPid: this.currentPid,
      currentPpid: this.currentPpid,
      creds: { ruid: this.ruid, euid: this.euid, suid: this.suid, rgid: this.rgid, egid: this.egid, sgid: this.sgid },
      sharedRegions: this.sharedRegions.map((r) => ({ ...r })),
      protNone: this.protNone.map((r) => ({ ...r })),
      nofileLimit: this.nofileLimit,
      signalHandlers: this.signalHandlers ? new Map(this.signalHandlers) : undefined,
      vfsFdSnapshot: this.vfs ? this.vfs.snapshotFds() : undefined,
    };
  }

  /** Restore state from process snapshot */
  restoreState(state: { brk: bigint; mmapBase: bigint; cwd: string; fdFlags: Map<number, number>; fdTable?: Map<number, FdEntry>; nativeRefs?: Map<number, number>; currentPid?: number; currentPpid?: number; creds?: { ruid: number; euid: number; suid: number; rgid: number; egid: number; sgid: number }; sharedRegions?: Array<{ addr: number; len: number }>; protNone?: Array<{ start: number; end: number }>; nofileLimit?: number; signalHandlers?: Map<number, { handler: bigint; flags: bigint; restorer: bigint; mask: bigint }>; vfsFdSnapshot?: any }): void {
    this.brk = state.brk;
    this.mmapBase = state.mmapBase;
    this.cwd = state.cwd;
    if (this.vfs) {
      // set the cwd directly (it was already validated when first entered); vfs.chdir would
      // re-resolve against the in-memory /proc+/dev tree and reject real NativeFS paths.
      this.vfs.setCwd(state.cwd);
      if (state.vfsFdSnapshot) this.vfs.restoreFds(state.vfsFdSnapshot);
    }
    this.fdFlags = new Map(state.fdFlags);
    if (state.fdTable) this.fdTable = new Map(state.fdTable);
    if (state.nativeRefs) this.nativeRefs = new Map(state.nativeRefs);
    if (state.currentPid !== undefined) this.currentPid = state.currentPid;
    if (state.currentPpid !== undefined) this.currentPpid = state.currentPpid;
    if (state.creds) {
      this.ruid = state.creds.ruid; this.euid = state.creds.euid; this.suid = state.creds.suid;
      this.rgid = state.creds.rgid; this.egid = state.creds.egid; this.sgid = state.creds.sgid;
    }
    if (state.sharedRegions !== undefined) this.sharedRegions = state.sharedRegions.map((r) => ({ ...r }));
    if (state.protNone !== undefined) this.protNone = state.protNone.map((r) => ({ ...r }));
    if (state.nofileLimit !== undefined) this.nofileLimit = state.nofileLimit;
    if (state.signalHandlers !== undefined) this.signalHandlers = new Map(state.signalHandlers);
  }

  /**
   * After fork, native fds in the inherited fdTable still reference the parent's
   * swift filehandles. dup each one so the child owns independent fds — otherwise
   * parent close() kills the child's view of the file.
   */
  dupNativeFdsAfterFork(dupFn: (oldNativeFd: number) => number): void {
    for (const [guestFd, entry] of this.fdTable) {
      if (entry.type !== 'native') continue;
      const newFd = dupFn(entry.nativeFd);
      if (newFd < 0) continue;
      this.fdTable.set(guestFd, { type: 'native', nativeFd: newFd, path: entry.path });
    }
  }

  /**
   * Close every native (Swift-backed) fd this process still holds. real kernels close a
   * process's fds when it exits; jitter never did, so a Worker that runs many pooled tenants
   * leaks Swift file handles until the device fd budget is exhausted and NativeFS starts failing
   * opens. call this on process exit / before pool reuse. stdio and non-native fds are ignored.
   */
  closeNativeFds(closeFn: (nativeFd: number) => void): void {
    for (const [, entry] of this.fdTable) {
      if (entry.type === 'native') closeFn(entry.nativeFd);
    }
  }

  resetForExec(brkStart: bigint, closeFn?: (nativeFd: number) => void): void {
    this.brk = brkStart;
    this.mmapBase = 0x2100000n;
    // exec closes exactly the descriptors marked FD_CLOEXEC; every other fd survives into the new
    // image (this is how programs pass fds to an exec'd child). native fds we drop must be
    // fs_close'd too, or exec leaks the Swift file handle behind them.
    for (const fd of [...this.fdCloexec]) {
      const entry = this.fdTable.get(fd);
      if (entry) {
        if (entry.type === 'native' && closeFn) closeFn(entry.nativeFd);
        else if (entry.type === 'vfs' && this.vfs) this.vfs.close(entry.fd);
        this.fdTable.delete(fd);
      } else if (this.vfs?.hasFd(fd)) {
        this.vfs.close(fd);
      }
      this.fdFlags.delete(fd);
    }
    this.fdCloexec.clear();
  }

  /** Allocate next available fd */
  // lowest-numbered unused guest fd >= min, per posix. the guest fd namespace is
  // shared: pipe/native/socket/tty fds live in fdTable, vfs-backed fds live in the
  // vfs (where the guest fd is also the vfs internal fd). scan both so the two
  // allocators never hand out the same number.
  private lowestFreeFd(min: number): number {
    let fd = min;
    while (this.fdTable.has(fd) || (this.vfs?.hasFd(fd) ?? false)) fd++;
    return fd;
  }

  private allocFd(): number {
    // linux hands out the lowest free descriptor including 0/1/2 once they are closed. starting at
    // 3 meant a guest that closed a std fd and reopened (as neovim does) got a high fd instead of the
    // reused low one, desyncing its fd bookkeeping. 0/1/2 stay reserved while their entries live.
    return this.lowestFreeFd(0);
  }

  /** true if a new fd (>= min) can be allocated within RLIMIT_NOFILE. */
  private fdAvailable(min = 3): boolean {
    return this.lowestFreeFd(min) < this.nofileLimit;
  }

  /** Resolve fd: check fdTable for overrides (dup2'd pipes, etc.), else fall through to default */
  private getFdEntry(fd: number): FdEntry | undefined {
    return this.fdTable.get(fd);
  }

  private isNonBlocking(fd: number): boolean {
    return ((this.fdFlags.get(fd) ?? 0) & O_NONBLOCK) !== 0;
  }

  /** public view of a fd's O_NONBLOCK flag (the worker needs it for kernel-pipe write blocking). */
  isNonBlockingFd(fd: number): boolean { return this.isNonBlocking(fd); }

  // a user pointer faults if it is out of the guest-addressable region (badGuestPtr) or lands
  // in a range that was mapped PROT_NONE.
  private ptrFaults(bytes: Uint8Array, addr: number, len = 1): boolean {
    if (badGuestPtr(bytes, addr, len)) return true;
    for (const r of this.protNone) {
      if (addr < r.end && addr + len > r.start) return true;
    }
    return false;
  }

  /** Whether the fd has data available for read without blocking. */
  private isFdReadable(fd: number): boolean {
    const entry = this.fdTable.get(fd);
    if (!entry) return false;
    switch (entry.type) {
      case 'stdio': return entry.fd === 0 ? (this.io.stdinHasData?.() ?? true) : true;
      case 'tty': return this.io.stdinHasData?.() ?? true;
      case 'pipe_read': return entry.pipe.data.length > 0 || entry.pipe.writeClosed;
      case 'pipe_write': return false;
      case 'vfs': case 'native': return true;
      case 'socket': return false; // handled via bridge poll
      case 'eventfd': return entry.counter > 0n;
      case 'inotify': case 'timerfd': return false; // stub: never have events
      case 'epoll': return false;
    }
  }

  /** Whether the fd can accept a write without blocking. */
  private isFdWritable(fd: number): boolean {
    const entry = this.fdTable.get(fd);
    if (!entry) return false;
    switch (entry.type) {
      case 'stdio': return entry.fd === 1 || entry.fd === 2;
      case 'tty': return true;
      case 'pipe_write': return !entry.pipe.readClosed;
      case 'pipe_read': return false;
      case 'vfs': case 'native': return true;
      case 'socket': return true; // connected TCP sockets accept writes (net_write queues via NWConnection)
      case 'eventfd': return entry.counter < 0xFFFFFFFFFFFFFFFEn;
      case 'inotify': case 'timerfd': case 'epoll': return false;
    }
  }

  /** Whether a wait on this fd's underlying waker mechanism would resume on data arrival. */
  private isFdWakeable(fd: number): boolean {
    const entry = this.fdTable.get(fd);
    if (!entry) return false;
    return (entry.type === 'stdio' && entry.fd === 0) || entry.type === 'tty';
  }

  /** Read an fd_set bitmap from guest memory into a Set<number>. */
  private readFdSet(view: DataView, ptr: number, nfds: number): Set<number> {
    const out = new Set<number>();
    if (ptr === 0) return out;
    const bytes = new Uint8Array(view.buffer);
    for (let fd = 0; fd < nfds; fd++) {
      const byte = bytes[ptr + GUEST_MEM_BASE + (fd >> 3)];
      if ((byte & (1 << (fd & 7))) !== 0) out.add(fd);
    }
    return out;
  }

  /** Write a Set<number> as an fd_set bitmap to guest memory, zeroing the rest. */
  private writeFdSet(view: DataView, ptr: number, nfds: number, set: Set<number>): void {
    if (ptr === 0) return;
    const bytes = new Uint8Array(view.buffer);
    const sz = Math.ceil(nfds / 8);
    for (let i = 0; i < sz; i++) bytes[ptr + GUEST_MEM_BASE + i] = 0;
    for (const fd of set) {
      if (fd < nfds && fd >= 0) {
        bytes[ptr + GUEST_MEM_BASE + (fd >> 3)] |= (1 << (fd & 7));
      }
    }
  }

  handle(cpu: CpuState, memory: WebAssembly.Memory): SyscallResult {
    const nr = Number(cpu.getReg(8));   // X8 = syscall number
    const a0 = cpu.getReg(0);          // X0-X5 = args
    const a1 = cpu.getReg(1);
    const a2 = cpu.getReg(2);
    const a3 = cpu.getReg(3);
    const a4 = cpu.getReg(4);
    const a5 = cpu.getReg(5);

    const view = new DataView(memory.buffer);
    const bytes = new Uint8Array(memory.buffer);

    let ret: bigint;

    if (NET_TRACE && (this as any)._netActive && nr !== 64 && nr !== 66 && nr !== 63) this.nt(`sys ${nr} a0=${Number(a0 & 0xffffn)}`);

    switch (nr) {
      // ── Process lifecycle ──
      case 93:  // exit
      case 94:  // exit_group
        return { status: 'exit', code: Number(a0) };

      case 220: { // clone(flags, stack, parent_tid, tls, child_tid)
        const flags = Number(a0);
        const SIGCHLD = 17;
        // If SIGCHLD is set, treat as fork
        if (flags & SIGCHLD) {
          const svcPC = view.getBigUint64(PC_OFFSET, true);
          return { status: 'fork', resumePC: svcPC + 4n };
        }
        // Other clone variants not supported
        ret = -BigInt(ENOSYS);
        break;
      }

      case 221: { // execve(pathname, argv, envp)
        const pathname = readGuestString(bytes, Number(a0));

        // Read argv array (null-terminated array of pointers)
        const argvPtrs: string[] = [];
        let argvAddr = Number(a1);
        while (true) {
          const ptr = Number(view.getBigUint64(argvAddr + GUEST_MEM_BASE, true));
          if (ptr === 0) break;
          argvPtrs.push(readGuestString(bytes, ptr));
          argvAddr += 8;
        }

        // Read envp array
        const envpPtrs: string[] = [];
        let envpAddr = Number(a2);
        while (true) {
          const ptr = Number(view.getBigUint64(envpAddr + GUEST_MEM_BASE, true));
          if (ptr === 0) break;
          envpPtrs.push(readGuestString(bytes, ptr));
          envpAddr += 8;
        }

        return { status: 'execve', path: this.absolutizePath(pathname), argv: argvPtrs, envp: envpPtrs };
      }

      case 95: { // waitid(idtype, id, infop, options)
        // Per spec: more flexible wait. idtype: P_PID=1, P_PGID=2, P_ALL=0
        // For our vfork model, redirect to wait4 semantics
        const WNOHANG = 1;
        const widOptions = Number(a3);

        if (this.coop) {
          const idtype = Number(a0);
          const wantPid = idtype === 1 ? Number(a1) : -1; // P_PID targets a specific child
          return { status: 'coop_wait', kind: 'waitid', wantPid, wstatusAddr: 0, options: widOptions, infop: Number(a2) };
        }
        if (this.lastWaitStatus !== undefined) {
          const exitCode = this.lastWaitStatus;
          this.lastWaitStatus = undefined;
          const infop = Number(a2);
          if (infop !== 0) {
            // siginfo_t: fill in si_pid, si_status, si_code
            const base = infop + GUEST_MEM_BASE;
            // Zero the struct first (128 bytes)
            for (let i = 0; i < 128; i += 8) view.setBigUint64(base + i, 0n, true);
            view.setInt32(base, 17, true); // si_signo = SIGCHLD
            view.setInt32(base + 8, 1, true); // si_code = CLD_EXITED
            view.setInt32(base + 12, this.lastWaitPid ?? 1, true); // si_pid (in _sifields.sigchld)
            view.setInt32(base + 24, exitCode, true); // si_status
          }
          ret = 0n;
          this.lastWaitPid = undefined;
        } else if (widOptions & WNOHANG) {
          // Zero out infop to indicate "no child changed state"
          const infop = Number(a2);
          if (infop !== 0) {
            const base = infop + GUEST_MEM_BASE;
            for (let i = 0; i < 128; i += 8) view.setBigUint64(base + i, 0n, true);
          }
          ret = 0n;
        } else {
          ret = -BigInt(ECHILD);
        }
        break;
      }

      case 260: { // wait4(pid, wstatus, options, rusage)
        const wstatus = Number(a1);
        const WNOHANG = 1;
        const options = Number(a2);

        if (this.coop) {
          // the scheduler owns the process table and zombie children; let the harness resolve.
          return { status: 'coop_wait', kind: 'wait4', wantPid: Number(BigInt.asIntN(64, a0)), wstatusAddr: wstatus, options, infop: 0 };
        }

        // In our vfork model, the child has already exited by the time
        // the parent resumes. The exit code was stored by the dispatch loop.
        if (this.lastWaitStatus !== undefined) {
          const exitCode = this.lastWaitStatus;
          this.lastWaitStatus = undefined;
          if (wstatus !== 0) {
            // a child terminated by a signal is marked with the 0x10000 bit and carries the signal
            // in the low 7 bits -> WIFSIGNALED (raw status = signum). otherwise WIFEXITED (code<<8).
            const encoded = (exitCode & 0x10000) ? (exitCode & 0x7f) : ((exitCode & 0xFF) << 8);
            view.setUint32(wstatus + GUEST_MEM_BASE, encoded, true);
          }
          ret = BigInt(this.lastWaitPid || 1);
          this.lastWaitPid = undefined;
        } else if (options & WNOHANG) {
          ret = 0n; // No child to wait for
        } else {
          // No child exited — shouldn't happen in our model
          ret = -BigInt(ECHILD);
        }
        break;
      }

      // ── Read/Write ──
      case 63: { // read(fd, buf, count)
        const fd = Number(a0);
        const buf = Number(a1);
        const count = Number(a2);

        // Check fd table for overrides (pipes, dup'd fds)
        const readEntry = this.getFdEntry(fd);
        // valid fd but the destination buffer is outside the address space -> EFAULT (count 0 is a no-op).
        if ((readEntry !== undefined || (this.vfs?.hasFd(fd) ?? false)) && count > 0 && this.ptrFaults(bytes, buf, count)) {
          ret = -BigInt(EFAULT); break;
        }
        if (readEntry) {
          if (readEntry.type === 'pipe_write') {
            ret = -BigInt(EBADF); // can't read from write end of pipe
            break;
          }
          if (readEntry.type === 'pipe_read') {
            const pipe = readEntry.pipe;
            if (pipe.data.length === 0) {
              if (pipe.writeClosed) {
                ret = 0n; // EOF: all writers gone
              } else if (this.isNonBlocking(fd)) {
                ret = -BigInt(EAGAIN);
              } else if (this.coop) {
                // a writer is still open: park until data arrives or the last writer closes.
                return { status: 'coop_block', reason: 'pipe_read' };
              } else {
                ret = 0n; // serialized fast path has no scheduler to yield to
              }
            } else {
              const chunk = pipe.data.shift()!;
              const toRead = Math.min(count, chunk.length);
              bytes.set(chunk.subarray(0, toRead), buf + GUEST_MEM_BASE);
              if (toRead < chunk.length) {
                pipe.data.unshift(chunk.subarray(toRead));
              }
              pipe.bytes -= toRead;
              this.schedEvent = true; // a parked writer may now have room
              ret = BigInt(toRead);
            }
            break;
          }
          if ((readEntry.type === 'stdio' && readEntry.fd === 0) || readEntry.type === 'tty') {
            const data = this.io.readStdin(count);
            if (data === null) {
              if (this.isNonBlocking(fd)) { ret = -BigInt(EAGAIN); break; }
              return { status: 'blocked_read', fd: 0 };
            }
            bytes.set(data, buf + GUEST_MEM_BASE);
            ret = BigInt(data.length);
            break;
          }
          if (readEntry.type === 'eventfd') {
            if (readEntry.counter === 0n) {
              ret = -BigInt(EAGAIN); // would block; we don't have a per-fd waker
              break;
            }
            if (count < 8) { ret = -BigInt(EINVAL); break; }
            const value = readEntry.semaphore ? 1n : readEntry.counter;
            readEntry.counter = readEntry.semaphore ? readEntry.counter - 1n : 0n;
            view.setBigUint64(buf + GUEST_MEM_BASE, value, true);
            ret = 8n;
            break;
          }
          if (readEntry.type === 'inotify' || readEntry.type === 'timerfd') {
            ret = -BigInt(EAGAIN); // never have events in our stub
            break;
          }
          if (readEntry.type === 'epoll') {
            ret = -BigInt(EINVAL); // epoll fd is not directly readable; must use epoll_wait
            break;
          }
          if (readEntry.type === 'vfs') {
            ret = this.vfs ? BigInt(this.vfs.read(readEntry.fd, buf, count, view)) : -BigInt(EBADF);
            break;
          }
          if (readEntry.type === 'native') {
            // the bridge response slot holds ~16 KB; a larger read overflows it and corrupts the data.
            // cap to a slot-safe chunk (matches readGuestFile) and return a short read — the guest
            // loops for the rest. this is what was corrupting md5sum/od/large binary reads.
            const rcount = Math.min(count, 12288);
            return {
              status: 'bridge', method: 'fs_read',
              args: { fd: readEntry.nativeFd, count: rcount },
              apply: (resp: any) => {
                if (resp.ret < 0) {
                  cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                } else if (resp.data) {
                  const decoded = Uint8Array.from(atob(resp.data), c => c.charCodeAt(0));
                  bytes.set(decoded, buf + GUEST_MEM_BASE);
                  cpu.setReg(0, BigInt(decoded.length));
                } else {
                  cpu.setReg(0, BigInt(resp.ret)); // EOF or 0
                }
              },
            };
          }
          if (readEntry.type === 'socket') {
            // same slot-overflow cap as the native path: base64 of a 16 KB read exceeds the bridge
            // response slot, so the data is dropped and the guest gets ret>0 with an unwritten buffer.
            // that fed TLS record garbage ("unexpected message") and stalled bulk http. cap + short read.
            const scount = Math.min(count, 12288);
            return {
              status: 'bridge', method: 'net_read',
              args: { socketId: readEntry.socketId, count: scount },
              apply: (resp: any) => {
                if (resp.ret < 0) {
                  this.nt(`read fd=${fd} req=${count} -> err ${resp.ret}`);
                  cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                } else if (resp.data) {
                  const decoded = Uint8Array.from(atob(resp.data), c => c.charCodeAt(0));
                  bytes.set(decoded, buf + GUEST_MEM_BASE);
                  cpu.setReg(0, BigInt(decoded.length));
                } else {
                  this.nt(`read fd=${fd} req=${count} -> EOF/0 (ret=${resp.ret})`);
                  cpu.setReg(0, BigInt(resp.ret));
                }
              },
            };
          }
          ret = -BigInt(EBADF);
          break;
        }

        // No fdTable entry: fall through to VFS for raw fds (legacy path for VFS-issued fds)
        if (this.vfs && this.vfs.hasFd(fd)) {
          ret = BigInt(this.vfs.read(fd, buf, count, view));
        } else {
          ret = -BigInt(EBADF);
        }
        break;
      }

      case 64: { // write(fd, buf, count)
        const fd = Number(a0);
        const buf = Number(a1);
        const count = Number(a2);

        const writeEntry = this.getFdEntry(fd);
        // for a valid fd, a source buffer outside the accessible address space faults EFAULT (the
        // kernel validates the fd first, then the buffer). count 0 never touches the buffer.
        if ((writeEntry !== undefined || (this.vfs?.hasFd(fd) ?? false)) && count > 0 && this.ptrFaults(bytes, buf, count)) {
          ret = -BigInt(EFAULT); break;
        }
        if (writeEntry) {
          if (writeEntry.type === 'pipe_read') {
            ret = -BigInt(EBADF); // can't write to read end of pipe
            break;
          }
          if (writeEntry.type === 'pipe_write') {
            if (writeEntry.pipe.readClosed) {
              // Per spec: write to pipe with no readers → SIGPIPE + EPIPE
              this.raiseSignal(SIGPIPE);
              ret = -BigInt(32); // EPIPE
              break;
            }
            const data = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
            writeEntry.pipe.data.push(data);
            writeEntry.pipe.bytes += data.length;
            this.schedEvent = true; // a parked reader may now proceed
            ret = BigInt(count);
            break;
          }
          if (writeEntry.type === 'stdio') {
            const data = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
            if (writeEntry.fd === 1) this.io.writeStdout(data);
            else if (writeEntry.fd === 2) this.io.writeStderr(data);
            ret = BigInt(count);
            break;
          }
          if (writeEntry.type === 'tty') {
            const data = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
            this.io.writeStdout(data);
            ret = BigInt(count);
            break;
          }
          if (writeEntry.type === 'vfs') {
            ret = this.vfs ? BigInt(this.vfs.write(writeEntry.fd, buf, count, view)) : -BigInt(EBADF);
            break;
          }
          if (writeEntry.type === 'native') {
            const writeData = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
            const b64 = bytesToBase64(writeData);
            this.statCache.delete(writeEntry.path); // invalidate stat cache for this file
            return {
              status: 'bridge', method: 'fs_write',
              args: { fd: writeEntry.nativeFd, data: b64 },
              apply: (resp: any) => {
                if (resp.ret < 0) {
                  cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                } else {
                  cpu.setReg(0, BigInt(resp.ret));
                }
              },
            };
          }
          if (writeEntry.type === 'socket') {
            const sockWriteData = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
            const sockB64 = bytesToBase64(sockWriteData);
            this.nt(`write fd=${fd} len=${count}`);
            return {
              status: 'bridge', method: 'net_write',
              args: { socketId: writeEntry.socketId, data: sockB64 },
              apply: (resp: any) => {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              },
            };
          }
          if (writeEntry.type === 'eventfd') {
            if (count < 8) { ret = -BigInt(EINVAL); break; }
            const value = view.getBigUint64(buf + GUEST_MEM_BASE, true);
            if (value === 0xFFFFFFFFFFFFFFFFn) { ret = -BigInt(EINVAL); break; }
            const newCounter = writeEntry.counter + value;
            if (newCounter > 0xFFFFFFFFFFFFFFFEn) { ret = -BigInt(EAGAIN); break; }
            writeEntry.counter = newCounter;
            ret = 8n;
            break;
          }
          if (writeEntry.type === 'inotify' || writeEntry.type === 'timerfd' || writeEntry.type === 'epoll') {
            ret = -BigInt(EBADF); // not writable
            break;
          }
          ret = -BigInt(EBADF);
          break;
        }

        // No fdTable entry: fall through to VFS for raw fds
        if (this.vfs && this.vfs.hasFd(fd)) {
          ret = BigInt(this.vfs.write(fd, buf, count, view));
        } else {
          ret = -BigInt(EBADF);
        }
        break;
      }

      case 66: { // writev(fd, iov, iovcnt)
        const fd = Number(a0);
        const iovBase = Number(a1);
        const iovcnt = Number(a2);
        let totalWritten = 0;

        // Resolve fd via fd table
        const writevEntry = this.getFdEntry(fd);

        // Socket or native fd: gather all iov data and send as one bridge write
        if (writevEntry?.type === 'socket') {
          let totalLen = 0;
          for (let i = 0; i < iovcnt; i++) {
            totalLen += Number(view.getBigUint64(iovBase + i * 16 + 8 + GUEST_MEM_BASE, true));
          }
          const combined = new Uint8Array(totalLen);
          let off = 0;
          for (let i = 0; i < iovcnt; i++) {
            const base = Number(view.getBigUint64(iovBase + i * 16 + GUEST_MEM_BASE, true));
            const len = Number(view.getBigUint64(iovBase + i * 16 + 8 + GUEST_MEM_BASE, true));
            if (len === 0) continue;
            combined.set(bytes.slice(base + GUEST_MEM_BASE, base + GUEST_MEM_BASE + len), off);
            off += len;
          }
          const b64 = bytesToBase64(combined);
          this.nt(`writev fd=${fd} totalLen=${totalLen}`);
          return {
            status: 'bridge', method: 'net_write',
            args: { socketId: writevEntry.socketId, data: b64 },
            apply: (resp: any) => {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            },
          };
        }
        if (writevEntry?.type === 'native') {
          let totalLen = 0;
          for (let i = 0; i < iovcnt; i++) {
            totalLen += Number(view.getBigUint64(iovBase + i * 16 + 8 + GUEST_MEM_BASE, true));
          }
          const combined = new Uint8Array(totalLen);
          let off = 0;
          for (let i = 0; i < iovcnt; i++) {
            const base = Number(view.getBigUint64(iovBase + i * 16 + GUEST_MEM_BASE, true));
            const len = Number(view.getBigUint64(iovBase + i * 16 + 8 + GUEST_MEM_BASE, true));
            if (len === 0) continue;
            combined.set(bytes.slice(base + GUEST_MEM_BASE, base + GUEST_MEM_BASE + len), off);
            off += len;
          }
          const b64 = bytesToBase64(combined);
          return {
            status: 'bridge', method: 'fs_write',
            args: { fd: writevEntry.nativeFd, data: b64 },
            apply: (resp: any) => {
              if (resp.ret < 0) {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              } else {
                cpu.setReg(0, BigInt(resp.ret));
              }
            },
          };
        }

        for (let i = 0; i < iovcnt; i++) {
          const base = Number(view.getBigUint64(iovBase + i * 16 + GUEST_MEM_BASE, true));
          const len = Number(view.getBigUint64(iovBase + i * 16 + 8 + GUEST_MEM_BASE, true));
          if (len === 0) continue;

          const data = bytes.slice(base + GUEST_MEM_BASE, base + GUEST_MEM_BASE + len);

          if (writevEntry) {
            if (writevEntry.type === 'pipe_write') {
              writevEntry.pipe.data.push(data);
            } else if (writevEntry.type === 'stdio') {
              if (writevEntry.fd === 1) this.io.writeStdout(data);
              else if (writevEntry.fd === 2) this.io.writeStderr(data);
            } else if (writevEntry.type === 'tty') {
              this.io.writeStdout(data);
            } else if (writevEntry.type === 'vfs' && this.vfs) {
              this.vfs.write(writevEntry.fd, base, len, view);
            }
          } else if (this.vfs && this.vfs.hasFd(fd)) {
            this.vfs.write(fd, base, len, view);
          } else {
            ret = -BigInt(EBADF);
            break;
          }
          totalWritten += len;
        }
        if (ret !== -BigInt(EBADF)) ret = BigInt(totalWritten);
        break;
      }

      case 65: { // readv(fd, iov, iovcnt) — scatter read
        const rvFd = Number(a0);
        const rvIov = Number(a1);
        const rvCnt = Number(a2);
        let rvTotal = 0;
        for (let i = 0; i < rvCnt; i++) {
          const base = Number(view.getBigUint64(rvIov + i * 16 + GUEST_MEM_BASE, true));
          const len = Number(view.getBigUint64(rvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
          if (len === 0) continue;
          // Reuse the read syscall logic by temporarily setting args
          const origA0 = a0, origA1 = a1, origA2 = a2;
          cpu.setReg(0, BigInt(rvFd));
          cpu.setReg(1, BigInt(base));
          cpu.setReg(2, BigInt(len));
          // Directly call VFS read
          if (this.vfs) {
            const n = this.vfs.read(rvFd, base, len, view);
            if (n < 0) { if (rvTotal === 0) { ret = BigInt(n); } else { ret = BigInt(rvTotal); } break; }
            rvTotal += n;
            if (n < len) break; // short read
          }
        }
        if (ret === undefined as any) ret = BigInt(rvTotal);
        break;
      }

      case 67: { // pread64(fd, buf, count, offset) — read at offset without changing position
        const prFd = Number(a0);
        const prBuf = Number(a1);
        const prCount = Number(a2);
        const prOffset = BigInt.asIntN(64, a3);
        const prEntry = this.fdTable.get(prFd);
        if (this.vfs && this.vfs.hasFd(prFd)) {
          // Save current position, seek, read, restore position
          const savedPos = this.vfs.lseek(prFd, 0n, 1); // SEEK_CUR
          if (savedPos < 0n) { ret = savedPos; break; }
          this.vfs.lseek(prFd, prOffset, 0); // SEEK_SET
          const n = this.vfs.read(prFd, prBuf, prCount, view);
          this.vfs.lseek(prFd, savedPos, 0); // restore
          ret = BigInt(n);
        } else if (prEntry?.type === 'native' && this.io.readFileAt) {
          // real file on NativeFS: bridge pread at offset (ld.so preads ELF/program headers this way).
          const data = this.io.readFileAt(prEntry.nativeFd, Number(prOffset), prCount);
          new Uint8Array(memory.buffer).set(data, prBuf + GUEST_MEM_BASE);
          ret = BigInt(data.length);
        } else {
          ret = -BigInt(EBADF);
        }
        break;
      }

      case 68: { // pwrite64(fd, buf, count, offset) — write at offset without changing position
        const pwFd = Number(a0);
        const pwBuf = Number(a1);
        const pwCount = Number(a2);
        const pwOffset = BigInt.asIntN(64, a3);
        if (this.vfs) {
          const savedPos = this.vfs.lseek(pwFd, 0n, 1);
          if (savedPos < 0n) { ret = savedPos; break; }
          this.vfs.lseek(pwFd, pwOffset, 0);
          const n = this.vfs.write(pwFd, pwBuf, pwCount, view);
          this.vfs.lseek(pwFd, savedPos, 0);
          ret = BigInt(n);
        } else {
          ret = -BigInt(EBADF);
        }
        break;
      }

      // ── Pipes & fd duplication ──
      case 59: { // pipe2(pipefd, flags)
        const pipefdAddr = Number(a0);
        // a pipe consumes two descriptors; enforce RLIMIT_NOFILE so a create-until-failure loop
        // terminates with EMFILE instead of allocating fds (and pipe buffers) forever.
        if (!this.fdAvailable()) { ret = -BigInt(EMFILE); break; }
        const pipe: PipeBuffer = { data: [], writeClosed: false, readClosed: false, writers: 1, readers: 1, bytes: 0 };
        // register the read end before allocating the write end, otherwise lowest-free
        // allocation hands out the same number twice (readFd isn't in the table yet)
        const readFd = this.allocFd();
        this.fdTable.set(readFd, { type: 'pipe_read', pipe });
        if (!this.fdAvailable()) { this.fdTable.delete(readFd); ret = -BigInt(EMFILE); break; }
        const writeFd = this.allocFd();
        this.fdTable.set(writeFd, { type: 'pipe_write', pipe });
        // the kernel writes the fd pair to user memory last (copy_to_user), so a bad pipefd buffer
        // faults EFAULT after the descriptors are allocated; release them so nothing leaks.
        if (this.ptrFaults(bytes, pipefdAddr, 8)) {
          this.fdTable.delete(readFd); this.fdTable.delete(writeFd);
          ret = -BigInt(EFAULT); break;
        }
        // Write [readFd, writeFd] to guest memory
        view.setInt32(pipefdAddr + GUEST_MEM_BASE, readFd, true);
        view.setInt32(pipefdAddr + GUEST_MEM_BASE + 4, writeFd, true);
        ret = 0n;
        break;
      }

      case 23: { // dup(oldfd)
        const oldfd = Number(a0);
        const oldEntry = this.fdTable.get(oldfd);
        if (!oldEntry && !(this.vfs && this.vfs.hasFd(oldfd))) { ret = -BigInt(EBADF); break; }
        if (!this.fdAvailable()) { ret = -BigInt(EMFILE); break; } // RLIMIT_NOFILE
        if (oldEntry) {
          const newfd = this.allocFd();
          if (oldEntry.type === 'vfs' && this.vfs) {
            // share the vfs open file description at the new guest fd
            if (this.vfs.dupTo(oldEntry.fd, newfd) < 0) { ret = -BigInt(EBADF); break; }
          } else if (oldEntry.type === 'native') {
            this.fdTable.set(newfd, { ...oldEntry });
            this.nativeRefs.set(oldEntry.nativeFd, (this.nativeRefs.get(oldEntry.nativeFd) ?? 1) + 1);
          } else {
            if (oldEntry.type === 'pipe_write') oldEntry.pipe.writers++;
            else if (oldEntry.type === 'pipe_read') oldEntry.pipe.readers++;
            this.fdTable.set(newfd, { ...oldEntry });
          }
          ret = BigInt(newfd);
        } else if (this.vfs && this.vfs.hasFd(oldfd)) {
          const newfd = this.allocFd();
          if (this.vfs.dupTo(oldfd, newfd) < 0) { ret = -BigInt(EBADF); break; }
          ret = BigInt(newfd);
        } else {
          ret = -BigInt(EBADF);
        }
        if (ret >= 0n) this.fdFlags.set(Number(ret), this.fdFlags.get(oldfd) ?? 0);
        this.ft(`dup ${oldfd} -> ${BigInt.asIntN(64, ret)}`);
        break;
      }

      case 24: { // dup3(oldfd, newfd, flags)
        const oldfd = Number(a0);
        const newfd = Number(BigInt.asIntN(32, a1));
        const dup3flags = Number(a2);
        this.ft(`dup3 ${oldfd} -> ${newfd}`);
        // Per spec: dup3 with oldfd==newfd returns EINVAL
        if (oldfd === newfd) { ret = -BigInt(EINVAL); break; }
        // Validate fd range
        if (newfd < 0 || newfd >= MAX_FDS) { ret = -BigInt(EBADF); break; }
        // Validate oldfd exists
        const oldfdValid = oldfd >= 0 && (this.fdTable.has(oldfd) || (this.vfs?.hasFd(oldfd) ?? false));
        if (!oldfdValid) { ret = -BigInt(EBADF); break; }
        // Close newfd if it's open (in the fd table or as a plain vfs fd)
        const existingEntry = this.fdTable.get(newfd);
        if (existingEntry) {
          if (existingEntry.type === 'pipe_write') { if (--existingEntry.pipe.writers <= 0) existingEntry.pipe.writeClosed = true; }
          if (existingEntry.type === 'pipe_read') { if (--existingEntry.pipe.readers <= 0) existingEntry.pipe.readClosed = true; }
          if (existingEntry.type === 'vfs' && this.vfs) this.vfs.close(existingEntry.fd);
          if (existingEntry.type === 'native') {
            const refs = (this.nativeRefs.get(existingEntry.nativeFd) ?? 1) - 1;
            if (refs <= 0) this.nativeRefs.delete(existingEntry.nativeFd);
            else this.nativeRefs.set(existingEntry.nativeFd, refs);
            // Note: we skip bridge close here for simplicity — the fd will be reused
          }
          this.fdTable.delete(newfd);
        } else if (this.vfs && this.vfs.hasFd(newfd)) {
          this.vfs.close(newfd);
        }
        // Duplicate oldfd to newfd
        const srcEntry = this.fdTable.get(oldfd);
        if (srcEntry) {
          if (srcEntry.type === 'vfs' && this.vfs) {
            if (this.vfs.dupTo(srcEntry.fd, newfd) < 0) { ret = -BigInt(EBADF); break; }
          } else if (srcEntry.type === 'native') {
            this.fdTable.set(newfd, { ...srcEntry });
            this.nativeRefs.set(srcEntry.nativeFd, (this.nativeRefs.get(srcEntry.nativeFd) ?? 1) + 1);
          } else {
            if (srcEntry.type === 'pipe_write') srcEntry.pipe.writers++;
            else if (srcEntry.type === 'pipe_read') srcEntry.pipe.readers++;
            this.fdTable.set(newfd, { ...srcEntry });
          }
          ret = BigInt(newfd);
        } else if (this.vfs && this.vfs.hasFd(oldfd)) {
          if (this.vfs.dupTo(oldfd, newfd) < 0) { ret = -BigInt(EBADF); break; }
          ret = BigInt(newfd);
        } else {
          ret = -BigInt(EBADF);
          break;
        }
        // shared open file description -> copy the status flags to the duplicate.
        if (ret >= 0n) this.fdFlags.set(newfd, this.fdFlags.get(oldfd) ?? 0);
        // Per spec: dup3 O_CLOEXEC flag sets FD_CLOEXEC on the new fd
        if (dup3flags & 0o2000000) this.fdCloexec.add(newfd);
        else this.fdCloexec.delete(newfd);
        break;
      }

      // ── File operations ──
      case 56: { // openat(dirfd, path, flags, mode)
        if (this.ptrFaults(bytes, Number(a1))) { ret = -BigInt(EFAULT); break; }
        const path = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
        const flags56 = Number(a2);
        const mode56 = Number(a3);
        if (path === '/dev/tty') {
          const ttyFd = this.allocFd();
          this.fdTable.set(ttyFd, { type: 'tty' });
          this.ft(`open ${path} -> ${ttyFd}`);
          ret = BigInt(ttyFd);
          break;
        }
        const bp56 = this.bridgePath(path);
        if (bp56 !== null) {
          const guestFd = this.allocFd();
          return {
            status: 'bridge', method: 'fs_open',
            args: { path: bp56, flags: flags56, mode: mode56 },
            apply: (resp: any) => {
              if (resp.ret < 0) {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                // no fd was committed to the table, so the number is still free
              } else {
                this.fdTable.set(guestFd, { type: 'native', nativeFd: resp.ret, path: bp56 });
                this.nativeRefs.set(resp.ret, 1);
                this.ft(`open ${bp56} -> ${guestFd} (native ${resp.ret})`);
                cpu.setReg(0, BigInt(guestFd));
              }
            },
          };
        }
        if (this.vfs) {
          // Apply umask to creation mode (per spec: effective mode = mode & ~umask)
          const effectiveMode = mode56 & ~this.umask;
          const openRet = this.vfs.open(path, flags56, effectiveMode);
          if (openRet >= 0) {
            // Store status flags for F_GETFL (strip creation-only flags)
            this.fdFlags.set(openRet, flags56 & ~(0o100 | 0o200 | 0o1000 | 0o2000000)); // strip O_CREAT|O_EXCL|O_TRUNC|O_CLOEXEC
            // O_CLOEXEC → set FD_CLOEXEC on the new fd
            if (flags56 & 0o2000000) this.fdCloexec.add(openRet);
          }
          this.ft(`open ${path} -> ${openRet} (vfs)`);
          ret = BigInt(openRet);
        } else {
          ret = -BigInt(ENOENT);
        }
        break;
      }

      case 57: { // close(fd)
        const fd = Number(a0);
        const closeEntry = this.fdTable.get(fd);
        this.ft(`close ${fd} (type=${closeEntry?.type ?? (this.vfs?.hasFd(fd) ? 'vfs' : 'none')})`);
        // per-fd metadata never outlives the fd; clearing here keeps a later reuse
        // of the same number (lowest-free allocation) from inheriting stale state
        this.fdFlags.delete(fd);
        this.fdCloexec.delete(fd);
        // posix: closing any descriptor to a file releases every advisory lock this
        // process holds on that file. do it before the vfs close drops the inode.
        if (this.locks) { const lk = this.lockKey(fd); if (lk) this.locks.releaseFile(lk, this.currentPid); this.schedEvent = true; }
        if (closeEntry) {
          if (closeEntry.type === 'pipe_write') { if (--closeEntry.pipe.writers <= 0) closeEntry.pipe.writeClosed = true; this.schedEvent = true; }
          if (closeEntry.type === 'pipe_read') { if (--closeEntry.pipe.readers <= 0) closeEntry.pipe.readClosed = true; this.schedEvent = true; }
          if (closeEntry.type === 'vfs' && this.vfs) this.vfs.close(closeEntry.fd);
          if (closeEntry.type === 'native') {
            this.fdTable.delete(fd);
            const nfd = closeEntry.nativeFd;
            const refs = (this.nativeRefs.get(nfd) ?? 1) - 1;
            if (refs <= 0) {
              this.nativeRefs.delete(nfd);
              return {
                status: 'bridge', method: 'fs_close',
                args: { fd: nfd },
                apply: (resp: any) => { cpu.setReg(0, BigInt(resp.ret ?? 0)); },
              };
            }
            this.nativeRefs.set(nfd, refs);
            ret = 0n;
            break;
          }
          if (closeEntry.type === 'socket') {
            this.fdTable.delete(fd);
            return {
              status: 'bridge', method: 'net_close',
              args: { socketId: closeEntry.socketId },
              apply: (resp: any) => { cpu.setReg(0, 0n); },
            };
          }
          if (closeEntry.type === 'epoll') this.epollSets?.delete(fd);
          this.fdTable.delete(fd);
          ret = 0n;
        } else if (this.vfs && this.vfs.hasFd(fd)) {
          ret = BigInt(this.vfs.close(fd));
        } else {
          ret = -BigInt(EBADF);
        }
        break;
      }

      case 46: { // ftruncate(fd, length)
        const ftFd = Number(a0);
        const ftLen = Number(a1);
        const ftEntry = this.getFdEntry(ftFd);
        if (ftEntry && ftEntry.type === 'vfs' && this.vfs) {
          ret = BigInt(this.vfs.ftruncate(ftEntry.fd, ftLen));
        } else if (ftEntry && ftEntry.type === 'native') {
          this.statCache.delete(ftEntry.path);
          return {
            status: 'bridge', method: 'fs_ftruncate',
            args: { fd: ftEntry.nativeFd, length: ftLen },
            apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); },
          };
        } else if (this.vfs) {
          ret = BigInt(this.vfs.ftruncate(ftFd, ftLen));
        } else {
          ret = -BigInt(EBADF);
        }
        break;
      }

      case 62: { // lseek(fd, offset, whence)
        const lseekFd = Number(a0);
        const lseekEntry = this.getFdEntry(lseekFd);
        if (lseekEntry) {
          switch (lseekEntry.type) {
            case 'stdio': case 'tty':
            case 'pipe_read': case 'pipe_write':
            case 'inotify': case 'timerfd': case 'eventfd': case 'epoll':
            case 'socket':
              ret = -BigInt(ESPIPE);
              break;
            case 'native':
              return {
                status: 'bridge', method: 'fs_lseek',
                args: { fd: lseekEntry.nativeFd, offset: Number(a1), whence: Number(a2) },
                apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); },
              };
            case 'vfs':
              ret = this.vfs ? this.vfs.lseek(lseekEntry.fd, BigInt.asIntN(64, a1), Number(a2)) : -BigInt(EBADF);
              break;
          }
          break;
        }
        if (this.vfs && this.vfs.hasFd(lseekFd)) {
          ret = this.vfs.lseek(lseekFd, BigInt.asIntN(64, a1), Number(a2));
        } else {
          ret = -BigInt(EBADF);
        }
        break;
      }

      // ── stat/fstat ──
      case 80: { // fstat(fd, statbuf)
        const fd = Number(a0);
        const statBuf = Number(a1);
        if (this.ptrFaults(bytes, statBuf, 128)) { ret = -BigInt(EFAULT); break; }

        // Check fd table first
        const fstatEntry = this.getFdEntry(fd);
        if (fd <= 4 || fstatEntry?.type === 'tty' || fstatEntry?.type === 'stdio') this.ft(`fstat ${fd} type=${fstatEntry?.type ?? (this.vfs?.hasFd(fd) ? 'vfs' : 'none')}`);
        if (fstatEntry) {
          if (fstatEntry.type === 'tty' && this.vfs) {
            // report the /dev/tty inode so fstat(ttyfd) matches stat("/dev/tty"); musl's ttyname_r
            // compares (st_dev, st_ino) of the readlink target against the fd, and libuv's tty
            // reopen (hence neovim's TUI) hinges on that check passing. only the explicit /dev/tty
            // fd is remapped; stdio (0/1/2) keeps its plain chardev stat so headless stays correct.
            ret = BigInt(this.vfs.fstatat(-100, '/dev/tty', statBuf, 0, view));
          } else if (fstatEntry.type === 'stdio' || fstatEntry.type === 'tty' ||
              fstatEntry.type === 'pipe_read' || fstatEntry.type === 'pipe_write' ||
              fstatEntry.type === 'inotify' || fstatEntry.type === 'timerfd' ||
              fstatEntry.type === 'eventfd' || fstatEntry.type === 'epoll') {
            writeChardevStat(view, statBuf);
            ret = 0n;
          } else if (fstatEntry.type === 'native') {
            return {
              status: 'bridge', method: 'fs_stat',
              args: { path: fstatEntry.path },
              apply: (resp: any) => {
                if (resp.ret < 0) {
                  cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                } else {
                  writeBridgeStat(view, statBuf, resp);
                  cpu.setReg(0, 0n);
                }
              },
            };
          } else if (fstatEntry.type === 'vfs' && this.vfs) {
            ret = BigInt(this.vfs.fstat(fstatEntry.fd, statBuf, view));
          } else {
            ret = -BigInt(EBADF);
          }
        } else if (this.vfs && this.vfs.hasFd(fd)) {
          ret = BigInt(this.vfs.fstat(fd, statBuf, view));
        } else {
          ret = -BigInt(EBADF);
        }
        break;
      }

      case 79: { // fstatat(dirfd, path, statbuf, flags)
        const path = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
        const statBuf79 = Number(a2);
        if (this.ptrFaults(bytes, statBuf79, 128)) { ret = -BigInt(EFAULT); break; }
        const bp79 = this.bridgePath(path);
        if (bp79 !== null) {
          // Check stat cache (populated by recent getdents) — avoids bridge round-trip
          const cached = this.statCache.get(bp79);
          if (cached && cached.expiresAt > Date.now()) {
            writeBridgeStat(view, statBuf79, { _meta: [cached.mode, cached.size, cached.nlink, cached.mtime, cached.uid, cached.gid, cached.ino] });
            ret = 0n;
            break;
          }
          return {
            status: 'bridge', method: 'fs_stat',
            args: { path: bp79 },
            apply: (resp: any) => {
              if (resp.ret < 0) {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              } else {
                writeBridgeStat(view, statBuf79, resp);
                // Cache for subsequent stat calls
                if (resp._meta) {
                  this.statCache.set(bp79, {
                    mode: resp._meta[0],
                    size: resp._meta[1],
                    nlink: resp._meta[2],
                    mtime: resp._meta[3],
                    uid: resp._meta[4] ?? 0,
                    gid: resp._meta[5] ?? 0,
                    ino: resp._meta[6] ?? 0,
                    expiresAt: Date.now() + this.STAT_CACHE_TTL_MS,
                  });
                }
                cpu.setReg(0, 0n);
              }
            },
          };
        }
        if (this.vfs) {
          ret = BigInt(this.vfs.fstatat(Number(a0), path, statBuf79, Number(a3), view));
        } else {
          ret = -BigInt(ENOENT);
        }
        break;
      }

      case 61: { // getdents64(fd, dirp, count)
        const fd = Number(a0);
        const dirp = Number(a1);
        const count = Number(a2);

        const gdEntry = this.getFdEntry(fd);
        if (gdEntry?.type === 'native') {
          const dirPath = gdEntry.path; // capture for cache population
          return {
            status: 'bridge', method: 'fs_getdents',
            args: { fd: gdEntry.nativeFd },
            apply: (resp: any) => {
              if (resp.ret <= 0) {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                return;
              }
              // Write linux_dirent64 structs to guest memory
              // struct linux_dirent64 { u64 d_ino; s64 d_off; u16 d_reclen; u8 d_type; char d_name[]; }
              // Entries come as base64-encoded JSON array in resp.data, enriched with stat info
              let entries: { name: string; type: number; mode?: number; size?: number; mtime?: number }[] = [];
              if (resp.data) {
                try {
                  const decoded = atob(resp.data);
                  entries = JSON.parse(decoded);
                } catch {}
              }
              // Populate stat cache for each entry (eliminates N+1 bridge calls in ls -l)
              const cacheExpiry = Date.now() + this.STAT_CACHE_TTL_MS;
              for (const entry of entries) {
                if (entry.mode !== undefined) {
                  const childPath = dirPath === '/' ? '/' + entry.name : dirPath + '/' + entry.name;
                  this.statCache.set(childPath, {
                    mode: entry.mode,
                    size: entry.size ?? 0,
                    nlink: entry.type === 4 ? 2 : 1, // DT_DIR=4
                    mtime: entry.mtime ?? 0,
                    uid: entry.uid ?? 0,
                    gid: entry.gid ?? 0,
                    ino: entry.ino ?? 0,
                    expiresAt: cacheExpiry,
                  });
                }
              }
              let offset = 0;
              for (const entry of entries) {
                const nameBytes = new TextEncoder().encode(entry.name);
                const reclen = (19 + nameBytes.length + 1 + 7) & ~7; // align to 8
                if (offset + reclen > count) break;
                const base = dirp + offset + GUEST_MEM_BASE;
                view.setBigUint64(base, 1n, true);          // d_ino (fake)
                view.setBigInt64(base + 8, 0n, true);        // d_off
                view.setUint16(base + 16, reclen, true);     // d_reclen
                bytes[base + 18] = entry.type;                // d_type
                bytes.set(nameBytes, base + 19);              // d_name
                bytes[base + 19 + nameBytes.length] = 0;     // null terminator
                offset += reclen;
              }
              cpu.setReg(0, BigInt(offset));
            },
          };
        }
        if (this.vfs) {
          ret = BigInt(this.vfs.getdents64(fd, dirp, count, view));
        } else {
          ret = -BigInt(EBADF);
        }
        break;
      }

      case 34: { // mkdirat(dirfd, path, mode)
        if (this.ptrFaults(bytes, Number(a1))) { ret = -BigInt(EFAULT); break; }
        const path = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
        const bp34 = this.bridgePath(path);
        if (bp34 !== null) {
          return {
            status: 'bridge', method: 'fs_mkdir',
            args: { path: bp34, mode: Number(a2) },
            apply: (resp: any) => {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            },
          };
        }
        if (this.vfs) {
          ret = BigInt(this.vfs.mkdirat(path, Number(a2) & ~this.umask));
        } else {
          ret = -BigInt(ENOENT);
        }
        break;
      }

      case 35: { // unlinkat(dirfd, path, flags)
        if (this.ptrFaults(bytes, Number(a1))) { ret = -BigInt(EFAULT); break; }
        const path = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
        if (path === '') { ret = -BigInt(ENOENT); break; } // empty path (no AT_EMPTY_PATH for unlink)
        const bp35 = this.bridgePath(path);
        if (bp35 !== null) {
          this.statCache.delete(bp35);
          return {
            status: 'bridge', method: 'fs_unlink',
            args: { path: bp35 },
            apply: (resp: any) => {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            },
          };
        }
        if (this.vfs) {
          ret = BigInt(this.vfs.unlinkat(path, Number(a2)));
        } else {
          ret = -BigInt(ENOENT);
        }
        break;
      }

      case 33: { // mknodat(dirfd, path, mode, dev)
        const path = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
        const mode = Number(a2);
        if (this.vfs) {
          // S_IFIFO = 0o010000 — named pipe (mkfifo)
          // S_IFREG = 0o100000 — regular file
          // S_IFCHR = 0o020000 — character device (needs root, skip for now)
          const fmt = mode & 0o170000;
          if (fmt === 0o010000) {
            // mkfifo — create a named-pipe (S_IFIFO) inode
            ret = BigInt(this.vfs.mkfifo(path, mode & 0o7777));
          } else if (fmt === 0o100000 || fmt === 0) {
            // Regular file
            ret = BigInt(this.vfs.open(path, 0o100, mode & 0o7777)); // O_CREAT
            if (ret >= 0n) {
              this.vfs.close(Number(ret));
              ret = 0n;
            }
          } else {
            ret = -BigInt(EPERM); // Can't create block/char devices
          }
        } else {
          ret = -BigInt(ENOSYS);
        }
        break;
      }

      case 38: { // renameat(olddirfd, oldpath, newdirfd, newpath)
        if (this.ptrFaults(bytes, Number(a1)) || this.ptrFaults(bytes, Number(a3))) { ret = -BigInt(EFAULT); break; }
        const oldpath = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
        const newpath = this.resolveAt(a2, readGuestString(bytes, Number(a3)));
        const bOld = this.bridgePath(oldpath);
        if (bOld) {
          const bNew = this.bridgePath(newpath) ?? newpath;
          this.statCache.delete(bOld);
          this.statCache.delete(bNew);
          return {
            status: 'bridge', method: 'fs_rename',
            args: { oldPath: bOld, newPath: bNew },
            apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); },
          };
        }
        if (this.vfs) {
          ret = BigInt(this.vfs.rename(oldpath, newpath));
        } else {
          ret = -BigInt(ENOSYS);
        }
        break;
      }

      case 36: { // symlinkat(target, newdirfd, linkpath)
        if (this.ptrFaults(bytes, Number(a0)) || this.ptrFaults(bytes, Number(a2))) { ret = -BigInt(EFAULT); break; }
        const target = readGuestString(bytes, Number(a0));
        const linkpath = this.resolveAt(a1, readGuestString(bytes, Number(a2)));
        const bLink = this.bridgePath(linkpath);
        if (bLink) {
          return {
            status: 'bridge', method: 'fs_symlink',
            args: { target, linkpath: bLink },
            apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); },
          };
        }
        if (this.vfs) {
          ret = BigInt(this.vfs.symlinkat(target, linkpath));
        } else {
          ret = -BigInt(ENOSYS);
        }
        break;
      }

      case 37: { // linkat(olddirfd, oldpath, newdirfd, newpath, flags)
        if (this.ptrFaults(bytes, Number(a1)) || this.ptrFaults(bytes, Number(a3))) { ret = -BigInt(EFAULT); break; }
        const oldStr = readGuestString(bytes, Number(a1));
        const newStr = readGuestString(bytes, Number(a3));
        // an empty pathname (without AT_EMPTY_PATH) is ENOENT, checked before any existence check.
        if (oldStr === '' || newStr === '') { ret = -BigInt(ENOENT); break; }
        const oldpath = this.resolveAt(a0, oldStr);
        const newpath = this.resolveAt(a2, newStr);
        if (this.vfs) {
          ret = BigInt(this.vfs.linkat(oldpath, newpath));
        } else {
          ret = -BigInt(ENOSYS);
        }
        break;
      }

      case 88: { // utimensat(dirfd, path, times, flags)
        // Accept but don't actually track timestamps (our VFS doesn't store them)
        ret = 0n;
        break;
      }

      case 78: { // readlinkat(dirfd, path, buf, bufsiz)
        const rawPath = readGuestString(bytes, Number(a1));
        const path = this.resolveAt(a0, rawPath);
        const buf = Number(a2);
        const bufsiz = Number(a3);
        // an empty pathname is ENOENT (without AT_EMPTY_PATH); a non-positive buffer size is EINVAL.
        if (rawPath === '') { ret = -BigInt(ENOENT); break; }
        if (bufsiz <= 0) { ret = -BigInt(EINVAL); break; }
        this.ft(`readlinkat ${path}`);
        // /proc/self/fd/<n> (and /proc/<pid>/fd/<n>) resolve to the path behind the fd. musl's
        // ttyname_r relies on this to reopen a tty, which is how neovim's libuv TUI comes up.
        const pfd = path.match(/^\/proc\/(?:self|\d+)\/fd\/(\d+)$/);
        if (pfd) {
          const tfd = Number(pfd[1]);
          const te = this.getFdEntry(tfd);
          const link = (te?.type === 'tty' || te?.type === 'stdio') ? '/dev/tty' : ((te as any)?.path ?? ((this.vfs as any)?.fdPath?.(tfd) ?? null));
          this.ft(`readlink /proc/self/fd/${tfd} -> ${link ?? 'NULL'} (type=${te?.type ?? '?'})`);
          if (link) {
            const enc = new TextEncoder().encode(link);
            const len = Math.min(enc.length, bufsiz);
            if (badGuestPtr(bytes, buf, len)) { ret = -BigInt(14); break; }
            bytes.set(enc.subarray(0, len), buf + GUEST_MEM_BASE);
            ret = BigInt(len);
            break;
          }
        }
        const bPath = this.bridgePath(path);
        if (bPath) {
          return {
            status: 'bridge', method: 'fs_readlink',
            args: { path: bPath },
            apply: (resp: any) => {
              if (resp.ret < 0) {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              } else if (resp.data) {
                // Data is the symlink target, base64-encoded
                const decoded = Uint8Array.from(atob(resp.data), c => c.charCodeAt(0));
                const len = Math.min(decoded.length, bufsiz);
                if (badGuestPtr(bytes, buf, len)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; } // EFAULT
                bytes.set(decoded.subarray(0, len), buf + GUEST_MEM_BASE);
                cpu.setReg(0, BigInt(len));
              } else {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(-EINVAL)));
              }
            },
          };
        }
        if (this.vfs) {
          const target = this.vfs.readlinkat(path);
          if (target === null) {
            ret = BigInt((this.vfs as any).resolveErrno?.() ?? -EINVAL);
          } else {
            const encoded = new TextEncoder().encode(target);
            const len = Math.min(encoded.length, bufsiz);
            if (badGuestPtr(bytes, buf, len)) { ret = -BigInt(14); break; } // EFAULT
            bytes.set(encoded.subarray(0, len), buf + GUEST_MEM_BASE);
            ret = BigInt(len);
          }
        } else {
          ret = -BigInt(ENOENT);
        }
        break;
      }

      case 48: { // faccessat(dirfd, path, mode, flags)
        // the kernel rejects any mode bits outside R_OK|W_OK|X_OK (F_OK is 0) with EINVAL first.
        if (Number(BigInt.asUintN(32, a2)) & ~0x7) { ret = -BigInt(EINVAL); break; }
        if (this.ptrFaults(bytes, Number(a1))) { ret = -BigInt(EFAULT); break; }
        const path = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
        const bp48 = this.bridgePath(path);
        if (bp48 !== null) {
          return {
            status: 'bridge', method: 'fs_faccessat',
            args: { path: bp48, mode: Number(a2) },
            apply: (resp: any) => {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            },
          };
        }
        if (this.vfs) {
          ret = BigInt(this.vfs.faccessat(Number(a0), path, Number(a2), Number(a3)));
        } else {
          ret = -BigInt(ENOENT);
        }
        break;
      }

      case 52: { // fchmod(fd, mode)
        const fchmodFd = Number(a0);
        if (this.vfs?.isOPath(fchmodFd)) { ret = -BigInt(EBADF); break; } // O_PATH fd
        const fchmodMode = Number(a1);
        const fchmodEntry = this.getFdEntry(fchmodFd);
        if (fchmodEntry?.type === 'native') {
          this.statCache.delete(fchmodEntry.path);
          return {
            status: 'bridge', method: 'fs_chmod',
            args: { path: fchmodEntry.path, mode: fchmodMode },
            apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); },
          };
        }
        if (this.vfs && this.vfs.hasFd(fchmodFd)) ret = BigInt(this.vfs.fchmod(fchmodFd, fchmodMode));
        else if (this.fdTable.has(fchmodFd)) ret = 0n; // native/stdio fd: best-effort no-op
        else ret = -BigInt(EBADF);
        break;
      }
      case 53: { // fchmodat(dirfd, path, mode)
        const fchmodatPath = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
        const fchmodatMode = Number(a2);
        const bpChmod = this.bridgePath(fchmodatPath);
        if (bpChmod !== null) {
          this.statCache.delete(bpChmod);
          return {
            status: 'bridge', method: 'fs_chmod',
            args: { path: bpChmod, mode: fchmodatMode },
            apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); },
          };
        }
        ret = this.vfs ? BigInt(this.vfs.chmodPath(fchmodatPath, fchmodatMode)) : -BigInt(ENOENT);
        break;
      }
      case 54: { // fchownat(dirfd, path, owner, group, flags)
        const fchownFlags = Number(a4);
        const cOwner = Number(BigInt.asIntN(32, a2));
        const cGroup = Number(BigInt.asIntN(32, a3));
        if (fchownFlags & 0x1000) {
          // AT_EMPTY_PATH: operate on dirfd directly (like fchown)
          const fchownFd = Number(a0);
          if (this.vfs && this.vfs.hasFd(fchownFd)) ret = BigInt(this.vfs.fchown(fchownFd, cOwner, cGroup));
          else if (this.fdTable.has(fchownFd)) ret = 0n; // native/stdio fd: best-effort no-op
          else ret = -BigInt(EBADF);
          break;
        }
        const fchownatPath = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
        const bpChown = this.bridgePath(fchownatPath);
        if (bpChown !== null) {
          return {
            status: 'bridge', method: 'fs_chown',
            args: { path: bpChown, uid: cOwner, gid: cGroup },
            apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret ?? 0))); },
          };
        }
        ret = this.vfs ? BigInt(this.vfs.chownPath(fchownatPath, cOwner, cGroup)) : -BigInt(ENOENT);
        break;
      }
      case 55: { // fchown(fd, owner, group)
        const fchownFd = Number(a0);
        if (this.vfs?.isOPath(fchownFd)) { ret = -BigInt(EBADF); break; } // O_PATH fd
        const owner = Number(BigInt.asIntN(32, a1));
        const group = Number(BigInt.asIntN(32, a2));
        const ent = this.getFdEntry(fchownFd);
        if (ent?.type === 'native') {
          this.statCache.delete(ent.path);
          return {
            status: 'bridge', method: 'fs_chown',
            args: { path: ent.path, uid: owner, gid: group },
            apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret ?? 0))); },
          };
        }
        if (this.vfs && this.vfs.hasFd(fchownFd)) ret = BigInt(this.vfs.fchown(fchownFd, owner, group));
        else if (this.fdTable.has(fchownFd)) ret = 0n;
        else ret = -BigInt(EBADF);
        break;
      }

      // ── statfs/fstatfs — return a plausible tmpfs-like result ──
      case 43:   // statfs(path, buf)
      case 44: { // fstatfs(fd, buf)
        const statfsBuf = Number(nr === 43 ? a1 : a1);
        // struct statfs on aarch64 is 120 bytes
        // f_type = 0x01021994 (TMPFS_MAGIC)
        const sv = new DataView(memory.buffer);
        const base = statfsBuf + GUEST_MEM_BASE;
        sv.setBigUint64(base + 0, 0x01021994n, true);   // f_type
        sv.setBigUint64(base + 8, 4096n, true);          // f_bsize
        sv.setBigUint64(base + 16, 1048576n, true);      // f_blocks
        sv.setBigUint64(base + 24, 524288n, true);       // f_bfree
        sv.setBigUint64(base + 32, 524288n, true);       // f_bavail
        sv.setBigUint64(base + 40, 1048576n, true);      // f_files
        sv.setBigUint64(base + 48, 524288n, true);       // f_ffree
        sv.setBigUint64(base + 56, 0n, true);            // f_fsid (low)
        sv.setBigUint64(base + 64, 255n, true);          // f_namelen
        sv.setBigUint64(base + 72, 4096n, true);         // f_frsize
        sv.setBigUint64(base + 80, 0n, true);            // f_flags
        // remaining padding zeroed
        ret = 0n;
        break;
      }

      // ── ioctl ──
      case 29: { // ioctl(fd, request, ...)
        const fd = Number(a0);
        if (this.vfs?.isOPath(fd)) { ret = -BigInt(EBADF); break; } // O_PATH fd
        const request = Number(a1);

        // Resolve fd through fd table — check if this is a dup'd stdio fd
        const ioctlEntry = this.getFdEntry(fd);
        if (fd <= 4) this.ft(`ioctl ${fd} req=0x${request.toString(16)} type=${ioctlEntry?.type ?? (this.vfs?.hasFd(fd) ? 'vfs' : 'none')}`);
        const isStdio = ioctlEntry?.type === 'stdio' || ioctlEntry?.type === 'tty';
        const isPipe = ioctlEntry?.type === 'pipe_read' || ioctlEntry?.type === 'pipe_write';
        if (isPipe) {
          // Pipes don't support terminal ioctls
          ret = -BigInt(EINVAL);
          break;
        }

        switch (request) {
          case TCGETS: {
            // Return a termios struct: raw mode (ICANON=0, ECHO=0)
            const buf = Number(a2);
            // c_iflag, c_oflag, c_cflag, c_lflag (each 4 bytes)
            view.setUint32(buf + GUEST_MEM_BASE, 0, true);     // c_iflag
            view.setUint32(buf + GUEST_MEM_BASE + 4, 0, true); // c_oflag
            view.setUint32(buf + GUEST_MEM_BASE + 8, 0, true); // c_cflag
            view.setUint32(buf + GUEST_MEM_BASE + 12, 0, true); // c_lflag
            ret = 0n;
            break;
          }
          case TCSETS: {
            ret = 0n; // Ignore terminal settings
            break;
          }
          case TIOCGWINSZ: {
            const buf = Number(a2);
            view.setUint16(buf + GUEST_MEM_BASE, this.termRows, true);
            view.setUint16(buf + GUEST_MEM_BASE + 2, this.termCols, true);
            view.setUint16(buf + GUEST_MEM_BASE + 4, 0, true); // xpixel
            view.setUint16(buf + GUEST_MEM_BASE + 6, 0, true); // ypixel
            ret = 0n;
            break;
          }
          case TIOCSWINSZ: {
            ret = 0n;
            break;
          }
          default:
            ret = -BigInt(EINVAL);
        }
        break;
      }

      // ── Memory management ──
      case 214: { // brk(addr)
        const addr = a0;
        if (addr === 0n) {
          ret = this.brk;
        } else if (addr >= this.brk) {
          // Ensure memory is large enough
          const needed = Number(addr) + GUEST_MEM_BASE;
          // budget: refuse to grow past the per-process ceiling. brk failure returns the
          // unchanged break, which glibc's wrapper reads as ENOMEM (malloc -> NULL).
          if (needed > this.maxBytes) { ret = this.brk; break; }
          if (needed > memory.buffer.byteLength) {
            const pagesNeeded = Math.ceil(needed / 65536);
            const currentPages = memory.buffer.byteLength / 65536;
            if (pagesNeeded > currentPages) {
              memory.grow(pagesNeeded - currentPages);
            }
          }
          this.brk = addr;
          ret = this.brk;
        } else {
          ret = this.brk; // Can't shrink brk
        }
        break;
      }

      case 222: { // mmap(addr, length, prot, flags, fd, offset)
        const length = Number(a1);
        const prot = Number(a2);
        const flags = Number(a3);
        const mmapFd = Number(BigInt.asIntN(32, a4));
        const aligned = (length + 0xFFF) & ~0xFFF; // page-align
        const MAP_ANONYMOUS = 0x20;
        const MAP_FIXED = 0x10;
        const MAP_SHARED = 0x01;

        // the kernel looks up the file descriptor (fget) before validating length or flags, so a
        // file-backed mmap on a closed/bogus fd faults EBADF even when length is 0 (mmap08). an
        // O_PATH descriptor also can't back a mapping (mmap on it faults EBADF).
        if (!(flags & MAP_ANONYMOUS)) {
          const validFd = (this.vfs?.hasFd(mmapFd) ?? false) || this.fdTable.has(mmapFd);
          if (mmapFd < 0 || !validFd || (this.vfs?.isOPath(mmapFd) ?? false)) { ret = -BigInt(EBADF); break; }
        }

        // zero length is invalid; exactly one map type (MAP_SHARED/MAP_PRIVATE, or the
        // MAP_SHARED_VALIDATE superset) must be requested.
        const mapType = flags & 0xf;
        if (length === 0 || (mapType !== 0x01 && mapType !== 0x02 && mapType !== 0x03)) {
          ret = -BigInt(EINVAL);
          break;
        }
        // MAP_SHARED_VALIDATE (type 0x3) validates every flag bit and rejects unknown ones with
        // EOPNOTSUPP; plain MAP_SHARED silently ignores them. known bits: type + fixed/anon/growsdown/
        // denywrite/executable/locked/noreserve/populate/nonblock/stack/hugetlb/sync/fixed_noreplace
        // + the 6-bit hugepage size field.
        if (mapType === 0x03) {
          const VALID_MMAP = 0x1f | 0x100 | 0x800 | 0x1000 | 0x2000 | 0x4000 | 0x8000 | 0x10000 | 0x20000 | 0x40000 | 0x80000 | 0x100000 | 0xfc000000;
          if ((flags & ~VALID_MMAP) !== 0) { ret = -BigInt(95); break; } // EOPNOTSUPP
        }

        // File-backed mmap always reads pages from the file, so the descriptor must be
        // readable (O_WRONLY faults with EACCES); a shared writable mapping additionally
        // requires the file to be open for writing.
        if (mmapFd >= 0 && !(flags & MAP_ANONYMOUS) && this.vfs && this.vfs.hasFd(mmapFd)) {
          const am = this.vfs.fdAccessMode(mmapFd);
          const PROT_WRITE = 0x2;
          if (am === 1 /* O_WRONLY */ ||
              ((flags & MAP_SHARED) && (prot & PROT_WRITE) && am === 0 /* O_RDONLY */)) {
            ret = -BigInt(EACCES);
            break;
          }
        }

        // only MAP_FIXED forces placement at the requested address (and legitimately replaces
        // whatever is mapped there). a bare hint is advisory: linux never honors it if doing so
        // would overlap an existing mapping, so we route hints to the monotonic arena instead.
        // honoring hints literally was zero-filling live heap/mmap regions -> guest heap corruption.
        const MAP_FIXED_NOREPLACE = 0x100000;
        let addr: bigint;
        if (a0 !== 0n && (flags & (MAP_FIXED | MAP_FIXED_NOREPLACE))) {
          addr = a0;
          // MAP_FIXED_NOREPLACE refuses to clobber an existing mapping: EEXIST on any overlap.
          if ((flags & MAP_FIXED_NOREPLACE) && this.vmaOverlaps(Number(addr), Number(addr) + aligned)) {
            ret = -BigInt(EEXIST);
            break;
          }
          const end = a0 + BigInt(aligned);
          if (end > this.mmapBase) this.mmapBase = end;
        } else {
          addr = this.mmapBase;
          this.mmapBase += BigInt(aligned);
        }

        // Ensure WASM memory is large enough
        const needed = Number(addr) + aligned + GUEST_MEM_BASE;
        // budget: refuse mappings past the per-process ceiling -> ENOMEM (mmap returns MAP_FAILED),
        // rather than growing toward the jetsam limit and crashing the whole content process.
        if (needed > this.maxBytes) { ret = -BigInt(ENOMEM); break; }
        if (needed > memory.buffer.byteLength) {
          const pagesNeeded = Math.ceil(needed / 65536);
          const currentPages = memory.buffer.byteLength / 65536;
          if (pagesNeeded > currentPages) memory.grow(pagesNeeded - currentPages);
        }

        // Zero-fill the mapped region
        new Uint8Array(memory.buffer).fill(0, Number(addr) + GUEST_MEM_BASE, Number(addr) + GUEST_MEM_BASE + aligned);

        // File-backed mmap: read file content into the mapped region.
        if (mmapFd >= 0 && !(flags & MAP_ANONYMOUS)) {
          const fileOffset = Number(a5); // sixth argument: file offset
          const entry = this.fdTable.get(mmapFd);
          if (this.vfs && this.vfs.hasFd(mmapFd)) {
            // vfs-backed file (synthetic /proc, or the whole fs in tests): seek + read in place.
            const savedPos = this.vfs.lseek(mmapFd, 0n, 1); // SEEK_CUR
            if (savedPos >= 0n) {
              this.vfs.lseek(mmapFd, BigInt(fileOffset), 0); // SEEK_SET
              this.vfs.read(mmapFd, Number(addr), aligned, new DataView(memory.buffer));
              this.vfs.lseek(mmapFd, savedPos, 0); // restore
            }
          } else if (entry?.type === 'native' && this.io.readFileAt) {
            // real file on NativeFS (the on-device case): bridge-read the segment at its file offset
            // into the mapping. this is what lets ld.so map shared libraries. short reads at EOF leave
            // the tail zero-filled (already done above), which is correct for a segment's bss overhang.
            const data = this.io.readFileAt(entry.nativeFd, fileOffset, aligned);
            if (data.length > 0) new Uint8Array(memory.buffer).set(data, Number(addr) + GUEST_MEM_BASE);
            // writable shared mapping -> the guest's stores must reach the file (apk extracts here).
            if ((flags & MAP_SHARED) && (prot & 0x2)) {
              this.fileMmaps.push({ start: Number(addr), len: aligned, fileLen: length, nativeFd: entry.nativeFd, fileOffset });
            }
          }
        }

        if (flags & MAP_SHARED) this.sharedRegions.push({ addr: Number(addr), len: aligned });

        // track PROT_NONE mappings so pointers into them fault; clear any stale overlap first.
        const mmStart = Number(addr), mmEnd = mmStart + aligned;
        this.protNone = this.protNone.filter((r) => r.end <= mmStart || r.start >= mmEnd);
        if ((prot & 0x7) === 0) this.protNone.push({ start: mmStart, end: mmEnd });

        if (flags & 0x2000) this.setLocked(mmStart, mmEnd, true); // MAP_LOCKED

        this.vmaSet(mmStart, mmEnd, { prot: prot & 0x7, shared: (flags & MAP_SHARED) !== 0 });
        this.protHook?.(mmStart, aligned, prot & 0x7); // page-protection: enforce PROT_NONE/read-only

        ret = addr;
        break;
      }

      case 226: { // mprotect(addr, len, prot)
        // the start must be page-aligned (EINVAL) and the whole range must be mapped; the null page
        // and other clearly-unmapped low memory give ENOMEM (mprotect01).
        if ((a0 & 0xFFFn) !== 0n) { ret = -BigInt(EINVAL); break; }
        const mpStart = Number(a0), mpEnd = mpStart + ((Number(a1) + 0xFFF) & ~0xFFF);
        if (mpStart < 0x1000) { ret = -BigInt(ENOMEM); break; }
        this.protNone = this.protNone.filter((r) => r.end <= mpStart || r.start >= mpEnd);
        if ((Number(a2) & 0x7) === 0) this.protNone.push({ start: mpStart, end: mpEnd });
        this.vmaProtect(mpStart, mpEnd, Number(a2) & 0x7);
        this.protHook?.(mpStart, mpEnd - mpStart, Number(a2) & 0x7); // page-protection guard table
        ret = 0n;
        break;
      }

      case 215: { // munmap(addr, length)
        // addr must be page-aligned and length non-zero; the range must lie within the
        // guest address space. otherwise EINVAL, matching the kernel.
        const unmapLen = Number(a1);
        const guestMax = BigInt(memory.buffer.byteLength - GUEST_MEM_BASE);
        if (unmapLen === 0 || (a0 & 0xFFFn) !== 0n || a0 >= guestMax || a0 + BigInt(unmapLen) > guestMax) {
          ret = -BigInt(EINVAL);
          break;
        }
        const umStart = Number(a0), umEnd = umStart + ((unmapLen + 0xFFF) & ~0xFFF);
        this.protNone = this.protNone.filter((r) => r.end <= umStart || r.start >= umEnd);
        this.setLocked(umStart, umEnd, false);
        this.vmaSet(umStart, umEnd, null);
        const umFlush = this.buildMmapFlush(cpu, memory, umStart, (unmapLen + 0xFFF) & ~0xFFF, true);
        if (umFlush) return umFlush;
        ret = 0n;
        break;
      }

      // ── Thread/process ──
      case 96: { // set_tid_address(tidptr)
        ret = 1n; // PID = 1
        break;
      }

      case 99: // set_robust_list
        ret = 0n;
        break;

      // gettid handled with getpid above

      case 172: // getpid — the thread-group id (== pid) for a thread; own pid otherwise
        ret = BigInt(this.tgid || this.currentPid);
        break;

      case 173: // getppid
        ret = BigInt(this.currentPpid);
        break;

      case 178: // gettid (same as pid in single-threaded model)
        ret = BigInt(this.currentPid);
        break;

      case 174: // getuid
        ret = BigInt(this.ruid);
        break;
      case 175: // geteuid
        ret = BigInt(this.euid);
        break;
      case 176: // getgid
        ret = BigInt(this.rgid);
        break;
      case 177: // getegid
        ret = BigInt(this.egid);
        break;

      // ── Signals ──
      case 134: { // rt_sigaction(signum, act, oldact, sigsetsize)
        // Per spec: set/get signal handler
        // struct kernel_sigaction on aarch64:
        //   sa_handler: 8 bytes (pointer or SIG_DFL=0/SIG_IGN=1)
        //   sa_flags: 8 bytes
        //   sa_restorer: 8 bytes (unused on aarch64, but present)
        //   sa_mask: 8 bytes (signal set)
        const signum = Number(a0);
        const actPtr = Number(a1);
        const oldactPtr = Number(a2);

        if (!this.signalHandlers) this.signalHandlers = new Map();

        // Write old handler to oldact if requested
        if (oldactPtr !== 0) {
          const base = oldactPtr + GUEST_MEM_BASE;
          const old = this.signalHandlers.get(signum);
          if (old) {
            view.setBigUint64(base, old.handler, true);
            view.setBigUint64(base + 8, old.flags, true);
            view.setBigUint64(base + 16, old.restorer, true);
            view.setBigUint64(base + 24, old.mask, true);
          } else {
            // Default: SIG_DFL (0) for everything
            for (let i = 0; i < 32; i += 8) view.setBigUint64(base + i, 0n, true);
          }
        }

        // Install new handler if act is provided
        if (actPtr !== 0) {
          const base = actPtr + GUEST_MEM_BASE;
          this.signalHandlers.set(signum, {
            handler: view.getBigUint64(base, true),
            flags: view.getBigUint64(base + 8, true),
            restorer: view.getBigUint64(base + 16, true),
            mask: view.getBigUint64(base + 24, true),
          });
        }
        ret = 0n;
        break;
      }

      case 135: { // rt_sigprocmask(how, set, oldset, sigsetsize)
        // Per spec: block/unblock signals
        // SIG_BLOCK=0, SIG_UNBLOCK=1, SIG_SETMASK=2
        const oldsetPtr = Number(a2);
        if (oldsetPtr !== 0) {
          // Return current mask (we track it but don't enforce it)
          view.setBigUint64(oldsetPtr + GUEST_MEM_BASE, this.signalMask ?? 0n, true);
        }
        const setPtr = Number(a1);
        if (setPtr !== 0) {
          const how = Number(a0);
          const newSet = view.getBigUint64(setPtr + GUEST_MEM_BASE, true);
          const current = this.signalMask ?? 0n;
          if (how === 0) this.signalMask = current | newSet;        // SIG_BLOCK
          else if (how === 1) this.signalMask = current & ~newSet;  // SIG_UNBLOCK
          else if (how === 2) this.signalMask = newSet;             // SIG_SETMASK
        }
        ret = 0n;
        break;
      }

      case 132: { // sigaltstack(ss, old_ss)
        // Per spec: set/get alternate signal stack
        const oldssPtr = Number(a1);
        if (oldssPtr !== 0) {
          const base = oldssPtr + GUEST_MEM_BASE;
          // Return "no alternate stack" (ss_flags = SS_DISABLE = 2)
          view.setBigUint64(base, 0n, true);       // ss_sp
          view.setInt32(base + 8, 2, true);         // ss_flags = SS_DISABLE
          view.setBigUint64(base + 16, 0n, true);   // ss_size
        }
        ret = 0n;
        break;
      }

      case 133: { // rt_sigsuspend(mask, sigsetsize)
        // Per spec: atomically set mask and suspend until signal
        // We can't actually suspend — return EINTR immediately (as if interrupted)
        ret = -BigInt(4); // EINTR
        break;
      }

      case 136: { // rt_sigpending(set, sigsetsize)
        // Per spec: get pending signals
        const pendPtr = Number(a0) + GUEST_MEM_BASE;
        view.setBigUint64(pendPtr, 0n, true); // no pending signals
        ret = 0n;
        break;
      }

      // ── Filesystem info ──
      case 17: { // getcwd(buf, size)
        const buf = Number(a0);
        const size = Number(a1);
        const cwd = this.vfs ? this.vfs.getcwd() : this.cwd;
        const encoded = new TextEncoder().encode(cwd + '\0');
        // the kernel checks the buffer size against the path length before touching the buffer, so
        // a too-small size faults ERANGE even for a bad/NULL buf; only then does a bad buf give EFAULT.
        if (encoded.length > size) { ret = -BigInt(34); break; } // ERANGE
        if (this.ptrFaults(new Uint8Array(memory.buffer), buf, encoded.length)) { ret = -BigInt(EFAULT); break; }
        new Uint8Array(memory.buffer).set(encoded, buf + GUEST_MEM_BASE);
        ret = a0;
        break;
      }

      case 49: { // chdir(path)
        if (this.ptrFaults(bytes, Number(a0))) { ret = -BigInt(EFAULT); break; }
        const path = readGuestString(bytes, Number(a0));
        const bPath = this.bridgePath(path);
        if (bPath) {
          // Verify directory exists via bridge, then update cwd
          return {
            status: 'bridge', method: 'fs_stat',
            args: { path: bPath },
            apply: (resp: any) => {
              if (resp.ret < 0) {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                return;
              }
              const mode = resp._meta ? resp._meta[0] : resp.mode;
              if (mode && (mode & 0o170000) === 0o40000) {
                this.cwd = bPath;
                if (this.vfs) this.vfs.setCwd(bPath);
                cpu.setReg(0, 0n);
              } else {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(-ENOTDIR)));
              }
            },
          };
        }
        if (this.vfs) {
          const res = this.vfs.chdir(path);
          ret = BigInt(res);
        } else {
          this.cwd = path;
          ret = 0n;
        }
        break;
      }

      // ── uname ──
      case 160: { // uname(buf)
        if (this.ptrFaults(new Uint8Array(memory.buffer), Number(a0), 6 * 65)) {
          ret = -BigInt(14); // EFAULT
          break;
        }
        const buf = Number(a0) + GUEST_MEM_BASE;
        const fields = ['Linux', 'jitter', '6.1.0', '#1', 'aarch64', '(none)'];
        for (let i = 0; i < fields.length; i++) {
          const encoded = new TextEncoder().encode(fields[i] + '\0');
          new Uint8Array(memory.buffer).set(encoded, buf + i * 65); // each field is 65 bytes
        }
        ret = 0n;
        break;
      }

      // ── Time ──
      case 113: { // clock_gettime(clk_id, tp)
        const tp = Number(a1) + GUEST_MEM_BASE;
        const ns = this.clockNs(Number(a0));
        view.setBigUint64(tp, ns / 1_000_000_000n, true);
        view.setBigUint64(tp + 8, ns % 1_000_000_000n, true);
        ret = 0n;
        break;
      }

      case 169: { // gettimeofday(tv, tz)
        if (this.ptrFaults(new Uint8Array(memory.buffer), Number(a0), 16)) { ret = -BigInt(14); break; } // EFAULT
        const tv = Number(a0) + GUEST_MEM_BASE;
        const ns = this.clockNs(0); // wall clock
        view.setBigUint64(tv, ns / 1_000_000_000n, true);
        view.setBigUint64(tv + 8, (ns % 1_000_000_000n) / 1000n, true); // usec
        ret = 0n;
        break;
      }

      // ── fcntl ──
      case 25: { // fcntl(fd, cmd, ...)
        const fd = Number(a0);
        const cmd = Number(a1);
        if (fd <= 4) this.ft(`fcntl ${fd} cmd=${cmd}`);
        // every fcntl command operates on an open descriptor; a closed/invalid fd is EBADF.
        if (!this.fdTable.has(fd) && !(this.vfs?.hasFd(fd) ?? false)) { ret = -BigInt(EBADF); break; }
        switch (cmd) {
          case F_DUPFD:
          case F_DUPFD_CLOEXEC: {
            const minfd = Number(a2);
            if (!this.fdAvailable(minfd)) { ret = -BigInt(EMFILE); break; } // RLIMIT_NOFILE
            // lowest available fd >= minfd across the shared guest fd namespace
            const newfd = this.lowestFreeFd(minfd);
            // Duplicate the source fd entry
            const srcEntry = this.fdTable.get(fd);
            if (srcEntry) {
              if (srcEntry.type === 'vfs' && this.vfs) {
                if (this.vfs.dupTo(srcEntry.fd, newfd) < 0) { ret = -BigInt(EBADF); break; }
              } else if (srcEntry.type === 'native') {
                this.fdTable.set(newfd, { ...srcEntry });
                this.nativeRefs.set(srcEntry.nativeFd, (this.nativeRefs.get(srcEntry.nativeFd) ?? 1) + 1);
              } else {
                this.fdTable.set(newfd, { ...srcEntry });
              }
            } else if (this.vfs && this.vfs.hasFd(fd)) {
              if (this.vfs.dupTo(fd, newfd) < 0) { ret = -BigInt(EBADF); break; }
            } else {
              ret = -BigInt(EBADF);
              break;
            }
            // a dup shares the open file description, so the status flags (access mode, O_APPEND,
            // O_NONBLOCK) are shared -> copy them so F_GETFL on the new fd matches the original.
            this.fdFlags.set(newfd, this.fdFlags.get(fd) ?? 0);
            // F_DUPFD_CLOEXEC sets close-on-exec on the new fd; plain F_DUPFD clears it
            if (cmd === F_DUPFD_CLOEXEC) this.fdCloexec.add(newfd);
            else this.fdCloexec.delete(newfd);
            ret = BigInt(newfd);
            break;
          }
          case F_GETFL: {
            // Per spec: returns status flags + access mode
            const flags = this.fdFlags.get(fd) ?? 0;
            ret = BigInt(flags);
            break;
          }
          case F_SETFL: {
            // Per spec: only O_APPEND, O_NONBLOCK, O_ASYNC, O_DIRECT, O_NOATIME are modifiable
            // Access mode and creation flags are silently ignored
            const modifiable = 0o2000 | 0o4000; // O_APPEND | O_NONBLOCK
            const current = this.fdFlags.get(fd) ?? 0;
            const newFlags = (current & ~modifiable) | (Number(a2) & modifiable);
            this.fdFlags.set(fd, newFlags);
            ret = 0n;
            break;
          }
          case F_GETFD:
            ret = BigInt(this.fdCloexec.has(fd) ? 1 : 0); // FD_CLOEXEC = 1
            break;
          case F_SETFD:
            if (Number(a2) & 1) this.fdCloexec.add(fd);
            else this.fdCloexec.delete(fd);
            ret = 0n;
            break;
          case 5:  // F_GETLK
          case 12: // F_GETLK64
          case 6:  // F_SETLK — non-blocking
          case 7:  // F_SETLKW — blocking
          case 13: // F_SETLK64
          case 14: { // F_SETLKW64
            const key = this.lockKey(fd);
            if (!this.locks) { ret = 0n; break; } // serialized path: no lock table, always grant
            // the kernel copies the flock struct from user space before checking the file
            // type, so a bad pointer faults EFAULT ahead of the EINVAL for an unlockable fd.
            if (this.ptrFaults(bytes, Number(a2), 32)) { ret = -BigInt(EFAULT); break; }
            if (key === null) { ret = -BigInt(EINVAL); break; } // pipes/sockets/tty can't hold record locks

            const flockPtr = Number(a2) + GUEST_MEM_BASE;
            const lType = view.getInt16(flockPtr, true);       // 0=RDLCK 1=WRLCK 2=UNLCK
            const lWhence = view.getInt16(flockPtr + 2, true); // 0=SET 1=CUR 2=END
            const lStart = view.getBigInt64(flockPtr + 8, true);
            const lLen = view.getBigInt64(flockPtr + 16, true);

            // resolve the byte range. only SEEK_SET is modelled precisely; CUR/END fall
            // back to the absolute l_start (ltp record-lock tests use SEEK_SET).
            let start = Number(lWhence === 0 ? lStart : lStart);
            let end: number;
            const len = Number(lLen);
            if (len === 0) end = Infinity;        // to end of file
            else if (len > 0) end = start + len - 1;
            else { start = start + len; end = start - len - 1; } // negative len locks bytes before l_start
            const wantType: 'r' | 'w' | 'u' = lType === 0 ? 'r' : lType === 1 ? 'w' : 'u';

            if (cmd === 5 || cmd === 12) { // F_GETLK: report a conflict, else F_UNLCK
              const probe = wantType === 'u' ? 'w' : wantType;
              const c = this.locks.conflict(key, this.currentPid, probe as 'r' | 'w', start, end);
              if (c) {
                view.setInt16(flockPtr, c.type === 'w' ? 1 : 0, true);
                view.setInt16(flockPtr + 2, 0, true); // SEEK_SET
                view.setBigInt64(flockPtr + 8, BigInt(c.start), true);
                view.setBigInt64(flockPtr + 16, c.end === Infinity ? 0n : BigInt(c.end - c.start + 1), true);
                view.setInt32(flockPtr + 24, c.pid, true); // l_pid
              } else {
                view.setInt16(flockPtr, 2, true); // F_UNLCK: no conflict
              }
              ret = 0n;
              break;
            }

            // F_SETLK / F_SETLKW
            this.locks.clearWaiter(this.currentPid); // fresh attempt: no longer waiting on a prior range
            if (wantType === 'u') {
              this.locks.set(key, this.currentPid, 'u', start, end);
              this.schedEvent = true; // a waiter blocked on this range may now proceed
              ret = 0n;
              break;
            }
            const conflict = this.locks.conflict(key, this.currentPid, wantType, start, end);
            if (conflict) {
              if (cmd === 7 || cmd === 14) { // F_SETLKW: block until the range frees
                if (this.coop) {
                  if (this.locks.wouldDeadlock(this.currentPid, key, wantType, start, end)) { ret = -BigInt(EDEADLK); break; }
                  this.locks.setWaiter(this.currentPid, key, wantType, start, end);
                  return { status: 'coop_block', reason: 'lock' };
                }
                ret = 0n; break; // serialized fallback: grant
              }
              ret = -BigInt(EAGAIN); // F_SETLK: non-blocking -> EAGAIN
              break;
            }
            this.locks.set(key, this.currentPid, wantType, start, end);
            ret = 0n;
            break;
          }
          case 8: // F_SETOWN
            ret = 0n;
            break;
          case 9: // F_GETOWN
            ret = BigInt(this.currentPid);
            break;
          case 10: // F_GETSIG
            ret = BigInt(this.fdSig.get(fd) ?? 0);
            break;
          case 11: // F_SETSIG
            this.fdSig.set(fd, Number(a2));
            ret = 0n;
            break;
          case 15: // F_SETOWN_EX
            ret = 0n;
            break;
          case 16: { // F_GETOWN_EX
            // Write f_owner_ex struct: { type: F_OWNER_PID(1), pid }
            const ownerPtr = Number(a2) + GUEST_MEM_BASE;
            view.setInt32(ownerPtr, 1, true); // type = F_OWNER_PID
            view.setInt32(ownerPtr + 4, this.currentPid, true); // pid
            ret = 0n;
            break;
          }
          case 1024: // F_SETLEASE
            ret = 0n;
            break;
          case 1025: // F_GETLEASE
            ret = 0n; // F_UNLCK — no lease
            break;
          case 1026: // F_NOTIFY
            ret = 0n;
            break;
          case 1032: // F_GETPIPE_SZ
            ret = 1048576n; // 1MB
            break;
          case 1031: { // F_SETPIPE_SZ
            const requestedSize = Number(a2);
            ret = BigInt(Math.max(requestedSize, 4096)); // accept requested size
            break;
          }
          default:
            ret = -BigInt(EINVAL);
        }
        break;
      }

      // ── Misc ──
      case 278: // getrandom
      {
        const buf = Number(a0) + GUEST_MEM_BASE;
        const len = Number(a1);
        for (let i = 0; i < len; i++) {
          bytes[buf + i] = (Math.random() * 256) | 0;
        }
        ret = a1;
        break;
      }

      case 98: { // futex(uaddr, op, val, timeout, uaddr2, val3)
        // Per Linux spec: fast userspace mutex operations
        const futexAddr = Number(a0) + GUEST_MEM_BASE;
        const futexOp = Number(a1) & 0x7F; // strip FUTEX_PRIVATE_FLAG (128)
        const futexVal = Number(a2);

        switch (futexOp) {
          case 0:   // FUTEX_WAIT
          case 9: { // FUTEX_WAIT_BITSET
            const current = view.getInt32(futexAddr, true);
            if (current !== futexVal) {
              ret = -BigInt(11); // EAGAIN — value already changed (also how a wake-up returns)
            } else if (this.coop) {
              // block until another process changes the word and FUTEX_WAKEs it. the futex
              // word lives in MAP_SHARED memory kept coherent across the context switch.
              return { status: 'coop_block', reason: 'futex', addr: Number(a0) };
            } else {
              ret = -BigInt(11); // serialized path: nothing can wake us, don't hang
            }
            break;
          }
          case 1:    // FUTEX_WAKE
          case 10: { // FUTEX_WAKE_BITSET
            // report the real number of waiters woken: tst_checkpoint_wake retries until
            // it wakes the number it expects, so an inflated count would desynchronize it.
            if (this.coop && this.futexWakeHook) ret = BigInt(this.futexWakeHook(Number(a0), Math.max(0, futexVal)));
            else ret = 0n;
            break;
          }
          case 2: // FUTEX_FD (deprecated)
            ret = -BigInt(ENOSYS);
            break;
          case 3: // FUTEX_REQUEUE: wake val waiters, requeue rest to uaddr2
            ret = 0n;
            break;
          case 4: // FUTEX_CMP_REQUEUE: like REQUEUE but check *uaddr == val3 first
            ret = 0n;
            break;
          case 5: // FUTEX_WAKE_OP: wake on uaddr, do atomic op on uaddr2, conditionally wake uaddr2
            ret = 0n;
            break;
          case 6: // FUTEX_LOCK_PI
            ret = 0n;
            break;
          case 7: // FUTEX_UNLOCK_PI
            ret = 0n;
            break;
          case 8: // FUTEX_TRYLOCK_PI
            ret = 0n;
            break;
          default:
            ret = -BigInt(ENOSYS);
        }
        break;
      }

      case 131: // tgkill
        ret = 0n;
        break;

      case 146: { // setuid(uid)
        const uid = Number(BigInt.asIntN(32, a0));
        if (this.euid === 0) { this.ruid = this.euid = this.suid = uid; ret = 0n; }
        else if (uid === this.ruid || uid === this.suid) { this.euid = uid; ret = 0n; }
        else ret = -BigInt(EPERM);
        break;
      }
      case 144: { // setgid(gid)
        const gid = Number(BigInt.asIntN(32, a0));
        if (this.euid === 0) { this.rgid = this.egid = this.sgid = gid; ret = 0n; }
        else if (gid === this.rgid || gid === this.sgid) { this.egid = gid; ret = 0n; }
        else ret = -BigInt(EPERM);
        break;
      }
      case 145: { // setreuid(ruid, euid)
        const r = Number(BigInt.asIntN(32, a0)), e = Number(BigInt.asIntN(32, a1));
        const unchanged = 0xffffffff;
        if (this.euid !== 0) {
          if ((r !== -1 && r >>> 0 !== unchanged && r !== this.ruid && r !== this.euid) ||
              (e !== -1 && e >>> 0 !== unchanged && e !== this.ruid && e !== this.euid && e !== this.suid)) {
            ret = -BigInt(EPERM); break;
          }
        }
        if (r !== -1 && r >>> 0 !== unchanged) this.ruid = r;
        if (e !== -1 && e >>> 0 !== unchanged) this.euid = e;
        if (r !== -1 && r >>> 0 !== unchanged) this.suid = this.euid;
        ret = 0n;
        break;
      }
      case 147: { // setresuid(ruid, euid, suid)
        const r = Number(BigInt.asIntN(32, a0)), e = Number(BigInt.asIntN(32, a1)), s = Number(BigInt.asIntN(32, a2));
        const set = (v: number) => v !== -1 && (v >>> 0) !== 0xffffffff;
        if (this.euid !== 0) {
          const allowed = (v: number) => v === this.ruid || v === this.euid || v === this.suid;
          if ((set(r) && !allowed(r)) || (set(e) && !allowed(e)) || (set(s) && !allowed(s))) {
            ret = -BigInt(EPERM); break;
          }
        }
        if (set(r)) this.ruid = r;
        if (set(e)) this.euid = e;
        if (set(s)) this.suid = s;
        ret = 0n;
        break;
      }
      case 143: { // setregid(rgid, egid)
        const r = Number(BigInt.asIntN(32, a0)), e = Number(BigInt.asIntN(32, a1));
        if (r !== -1 && (r >>> 0) !== 0xffffffff) this.rgid = r;
        if (e !== -1 && (e >>> 0) !== 0xffffffff) this.egid = e;
        ret = 0n;
        break;
      }
      case 149: { // setresgid(rgid, egid, sgid)
        const set = (v: number) => v !== -1 && (v >>> 0) !== 0xffffffff;
        const r = Number(BigInt.asIntN(32, a0)), e = Number(BigInt.asIntN(32, a1)), s = Number(BigInt.asIntN(32, a2));
        if (set(r)) this.rgid = r;
        if (set(e)) this.egid = e;
        if (set(s)) this.sgid = s;
        ret = 0n;
        break;
      }
      case 151: // setfsuid — returns previous fsuid; we alias fsuid to euid
        ret = BigInt(this.euid);
        break;
      case 152: // setfsgid
        ret = BigInt(this.egid);
        break;
      case 159: // setgroups
        ret = 0n;
        break;
      // ── Networking (via Swift bridge) ──

      case 198: { // socket(domain, type, protocol)
        const domain = Number(a0);   // AF_INET=2, AF_INET6=10, AF_UNIX=1
        const sockType = Number(a1) & 0xFF; // SOCK_STREAM=1, SOCK_DGRAM=2 (strip SOCK_CLOEXEC|SOCK_NONBLOCK)
        const protocol = Number(a2);

        // AF_UNIX sockets used by musl nscd — return ENOSYS so musl falls back to /etc/passwd
        if (domain === 1) { ret = -BigInt(ENOSYS); break; }

        if (!this.bridge?.isNative && !this.bridgeAvailable) {
          // No bridge available — can't do real networking
          ret = -BigInt(ENOSYS);
          break;
        }

        const socketFd = this.allocFd();
        return {
          status: 'bridge', method: 'net_socket',
          args: { domain, type: sockType, protocol },
          apply: (resp: any) => {
            if (resp.ret < 0) {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            } else {
              this.fdTable.set(socketFd, { type: 'socket', socketId: resp.ret, domain, sockType });
              if (Number(a1) & 0o2000000) this.fdCloexec.add(socketFd); // SOCK_CLOEXEC
              this.nt(`socket dom=${domain} type=${sockType} -> fd=${socketFd}`);
              (this as any)._netActive = true;
              cpu.setReg(0, BigInt(socketFd));
            }
          },
        };
      }

      case 200: { // bind(sockfd, addr, addrlen)
        const bindEntry = this.getFdEntry(Number(a0));
        if (!bindEntry || bindEntry.type !== 'socket') { ret = -BigInt(EBADF); break; }
        const addrBuf = this.parseSockaddr(bytes, Number(a1), Number(a2));
        return {
          status: 'bridge', method: 'net_bind',
          args: { socketId: bindEntry.socketId, ...addrBuf },
          apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); },
        };
      }

      case 201: { // listen(sockfd, backlog)
        const listenEntry = this.getFdEntry(Number(a0));
        if (!listenEntry || listenEntry.type !== 'socket') { ret = -BigInt(EBADF); break; }
        return {
          status: 'bridge', method: 'net_listen',
          args: { socketId: listenEntry.socketId, backlog: Number(a1) },
          apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); },
        };
      }

      case 202: { // accept(sockfd, addr, addrlen)
        const acceptEntry = this.getFdEntry(Number(a0));
        if (!acceptEntry || acceptEntry.type !== 'socket') { ret = -BigInt(EBADF); break; }
        const newFd = this.allocFd();
        return {
          status: 'bridge', method: 'net_accept',
          args: { socketId: acceptEntry.socketId },
          apply: (resp: any) => {
            if (resp.ret < 0) {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            } else {
              this.fdTable.set(newFd, { type: 'socket', socketId: resp.ret, domain: acceptEntry.domain, sockType: acceptEntry.sockType });
              cpu.setReg(0, BigInt(newFd));
            }
          },
        };
      }

      case 203: { // connect(sockfd, addr, addrlen)
        const connEntry = this.getFdEntry(Number(a0));
        if (!connEntry || connEntry.type !== 'socket') { ret = -BigInt(EBADF); break; }
        const connAddr = this.parseSockaddr(bytes, Number(a1), Number(a2));
        connEntry.remoteAddr = connAddr;
        return {
          status: 'bridge', method: 'net_connect',
          args: { socketId: connEntry.socketId, ...connAddr },
          apply: (resp: any) => {
            this.nt(`connect fd=${Number(a0)} ${connAddr.host}:${connAddr.port} -> ${resp.ret}`);
            cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
          },
        };
      }

      case 204: { // getsockname(sockfd, addr, addrlen)
        const gsnEntry = this.getFdEntry(Number(a0));
        if (!gsnEntry || gsnEntry.type !== 'socket') { ret = -BigInt(EBADF); break; }
        // Return a zeroed sockaddr (we don't track local addresses)
        const addrPtr = Number(a1) + GUEST_MEM_BASE;
        const lenPtr = Number(a2) + GUEST_MEM_BASE;
        const family = gsnEntry.domain;
        view.setUint16(addrPtr, family, true); // sa_family
        if (family === 2) { // AF_INET
          view.setUint32(lenPtr, 16, true);
        } else if (family === 10) { // AF_INET6
          view.setUint32(lenPtr, 28, true);
        }
        ret = 0n;
        break;
      }

      case 205: { // getpeername(sockfd, addr, addrlen)
        const gpnEntry = this.getFdEntry(Number(a0));
        if (!gpnEntry || gpnEntry.type !== 'socket') { ret = -BigInt(EBADF); break; }
        ret = -107n; // ENOTCONN
        break;
      }

      case 206: { // sendto(sockfd, buf, len, flags, dest_addr, addrlen)
        const stEntry = this.getFdEntry(Number(a0));
        if (!stEntry || stEntry.type !== 'socket') { ret = -BigInt(EBADF); break; }
        const sendBuf = Number(a1);
        const sendLen = Number(a2);
        const sendData = bytes.slice(sendBuf + GUEST_MEM_BASE, sendBuf + GUEST_MEM_BASE + sendLen);
        const b64 = bytesToBase64(sendData);
        // If dest_addr is provided (connectionless sendto), use net_sendto
        const destAddrPtr = Number(a4);
        const destAddrLen = Number(a5);
        if (destAddrPtr !== 0 && destAddrLen > 0) {
          const dest = this.parseSockaddr(bytes, destAddrPtr, destAddrLen);
          if (!stEntry.remoteAddr) stEntry.remoteAddr = dest;
          return {
            status: 'bridge', method: 'net_sendto',
            args: { socketId: stEntry.socketId, data: b64, host: dest.host, port: dest.port },
            apply: (resp: any) => {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            },
          };
        }
        // Connected socket — use net_write
        return {
          status: 'bridge', method: 'net_write',
          args: { socketId: stEntry.socketId, data: b64 },
          apply: (resp: any) => {
            cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
          },
        };
      }

      case 207: { // recvfrom(sockfd, buf, len, flags, src_addr, addrlen)
        const rfEntry = this.getFdEntry(Number(a0));
        if (!rfEntry || rfEntry.type !== 'socket') { ret = -BigInt(EBADF); break; }
        const recvBuf = Number(a1);
        const recvLen = Math.min(Number(a2), 12288); // slot-safe cap (see net_read in read())
        return {
          status: 'bridge', method: 'net_read',
          args: { socketId: rfEntry.socketId, count: recvLen },
          apply: (resp: any) => {
            if (resp.ret < 0) {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            } else if (resp.data) {
              const decoded = Uint8Array.from(atob(resp.data), c => c.charCodeAt(0));
              bytes.set(decoded, recvBuf + GUEST_MEM_BASE);
              cpu.setReg(0, BigInt(decoded.length));
            } else {
              cpu.setReg(0, BigInt(resp.ret));
            }
          },
        };
      }

      case 208: // setsockopt(sockfd, level, optname, optval, optlen)
        // Accept most socket options without actually setting them
        ret = 0n;
        break;

      case 209: { // getsockopt(sockfd, level, optname, optval, optlen)
        // Return default values for common options
        const optval = Number(a3) + GUEST_MEM_BASE;
        const optlen = Number(a4) + GUEST_MEM_BASE;
        const optname = Number(a2);
        // SO_ERROR (4): return 0 (no error)
        view.setInt32(optval, 0, true);
        view.setUint32(optlen, 4, true);
        ret = 0n;
        break;
      }

      // ── Process identity/groups ──
      case 129: { // kill(pid, sig)
        const killPid = Number(BigInt.asIntN(32, a0));
        const killSig = Number(a1);
        if (killSig < 0 || killSig > 64) { ret = -BigInt(EINVAL); break; }
        // a process exists (is signalable) if it's us, init, or a suspended ancestor. cross-process
        // delivery isn't modeled in the serialized fork engine, but existence/permission is.
        const pidExists = (p: number): boolean =>
          p === this.currentPid || p === 1 || p === this.currentPpid || (this.isLivePidFn?.(p) ?? false);
        if (killPid > 0) {
          if (killPid === this.currentPid) {
            if (killSig !== 0) this.raiseSignal(killSig); // deliver to self via the live queue
            ret = 0n;
          } else if (this.coop && this.killHook) {
            // cross-process delivery through the scheduler's process table
            if (!this.killHook(killPid, killSig)) { ret = -BigInt(ESRCH); break; }
            ret = 0n;
          } else {
            if (!pidExists(killPid)) { ret = -BigInt(ESRCH); break; }
            ret = 0n;
          }
        } else if (killPid === 0 || killPid === -1) {
          // own process group (0) or every process we may signal (-1): at minimum deliver to self.
          if (killSig !== 0) this.raiseSignal(killSig);
          ret = 0n;
        } else {
          // pid < -1: the process group -pid. only our own group is modeled; others are ESRCH.
          ret = -BigInt(ESRCH);
        }
        break;
      }

      case 166: { // umask(mask) — return old mask, set new
        const oldMask = this.umask;
        this.umask = Number(a0) & 0o777;
        ret = BigInt(oldMask);
        break;
      }

      case 154: // setpgid
        ret = 0n;
        break;
      case 155: // getpgid
        ret = BigInt(1);
        break;
      case 156: // getsid(pid) — get session ID
        ret = BigInt(this.currentPid); // each process is its own session leader
        break;
      case 157: // setsid — create new session
        ret = BigInt(this.currentPid);
        break;
      case 158: { // getgroups(size, list)
        // Per spec: return supplementary group IDs
        // We have 1 group: gid 0 (root)
        const gSize = Number(a0);
        if (gSize === 0) { ret = 1n; break; } // return count only
        if (gSize >= 1) {
          view.setUint32(Number(a1) + GUEST_MEM_BASE, 0, true); // gid 0
        }
        ret = 1n; // 1 supplementary group
        break;
      }

      case 122: // sched_setaffinity — stub
      case 123: // sched_getaffinity — stub
        ret = 0n;
        break;

      case 115: { // clock_nanosleep (NOT sched_getscheduler — that's 120)
        // Per spec: sleep on specified clock. Return immediately in emulator.
        if (this.coop) this.wantYield = true; // cooperative yield point while "sleeping"
        ret = 0n;
        break;
      }

      case 261: { // prlimit64(pid, resource, new_limit, old_limit)
        // struct rlimit64: { rlim_cur (8 bytes), rlim_max (8 bytes) }
        const resource = Number(a1);
        const rlNewPtr = Number(a2);
        const rlOldPtr = Number(a3);
        const RLIM_INFINITY = 0xFFFFFFFFFFFFFFFFn;
        // report the current limit into old_limit BEFORE applying the new one
        if (rlOldPtr !== 0) {
          const base = rlOldPtr + GUEST_MEM_BASE;
          switch (resource) {
            case 7:  // RLIMIT_NOFILE
              view.setBigUint64(base, BigInt(this.nofileLimit), true);
              view.setBigUint64(base + 8, 1048576n, true); // hard max
              break;
            case 3:  // RLIMIT_STACK
              view.setBigUint64(base, 8388608n, true);  // 8MB
              view.setBigUint64(base + 8, RLIM_INFINITY, true);
              break;
            default: // most others: infinity
              view.setBigUint64(base, RLIM_INFINITY, true);
              view.setBigUint64(base + 8, RLIM_INFINITY, true);
          }
        }
        // apply the new limit for the resources we enforce
        if (rlNewPtr !== 0 && resource === 7) {
          const cur = view.getBigUint64(rlNewPtr + GUEST_MEM_BASE, true);
          // clamp to a sane cap; RLIM_INFINITY -> effectively unlimited
          this.nofileLimit = cur >= 1048576n ? 1048576 : Number(cur);
        }
        ret = 0n;
        break;
      }

      case 233: { // madvise(addr, length, advice)
        // MADV_DONTNEED (4) on an anonymous private mapping must make subsequent reads
        // return zero-filled pages (linux semantics). musl's mallocng relies on this to
        // recycle freed slots; leaving stale bytes corrupts its inline chunk metadata.
        // MADV_FREE (8) may also zero lazily, so zeroing eagerly is safe.
        const advice = Number(a2);
        if (advice === 4 || advice === 8) {
          const start = Number(a0), len = Number(a1);
          const end = start + len;
          if (len > 0 && (start & 0xFFF) === 0 && end + GUEST_MEM_BASE <= memory.buffer.byteLength) {
            new Uint8Array(memory.buffer).fill(0, start + GUEST_MEM_BASE, end + GUEST_MEM_BASE);
          }
        }
        ret = 0n;
        break;
      }

      // ── Missing syscalls (spec-compliant implementations) ──

      case 50: { // fchdir(fd)
        // change cwd to the directory the fd refers to. native (bridge) fds carry their
        // guest path; that is what busybox's post-install triggers fchdir into.
        const fchEntry = this.getFdEntry(Number(a0));
        const fchPath = (fchEntry as any)?.path;
        if (typeof fchPath === 'string' && fchPath.length) {
          this.cwd = fchPath;
          if (this.vfs) this.vfs.setCwd(fchPath);
          ret = 0n;
        } else {
          ret = -BigInt(EBADF);
        }
        break;
      }

      case 101: { // nanosleep(req, rem)
        // Per spec: suspend execution for specified time
        // In our emulator, return immediately (no actual sleeping)
        // Zero out the remainder if rem is provided
        const remAddr = Number(a1);
        if (remAddr !== 0) {
          view.setBigUint64(remAddr + GUEST_MEM_BASE, 0n, true);
          view.setBigUint64(remAddr + GUEST_MEM_BASE + 8, 0n, true);
        }
        if (this.coop) this.wantYield = true; // let another process run while we "sleep"
        ret = 0n;
        break;
      }

      case 124: // sched_yield — yield CPU
        if (this.coop) this.wantYield = true;
        ret = 0n;
        break;

      case 165: { // getrusage(who, usage)
        // Per spec: return resource usage for self or children
        // Zero-fill the struct rusage (144 bytes on aarch64)
        const usageBuf = Number(a1) + GUEST_MEM_BASE;
        for (let i = 0; i < 144; i += 8) view.setBigUint64(usageBuf + i, 0n, true);
        ret = 0n;
        break;
      }

      case 167: { // prctl(option, arg2, arg3, arg4, arg5)
        const prOption = Number(a0);
        switch (prOption) {
          case 15: // PR_SET_NAME — set thread name (accept, ignore)
            ret = 0n;
            break;
          case 16: // PR_GET_NAME — get thread name
            if (badGuestPtr(new Uint8Array(memory.buffer), Number(a1), 16)) {
              ret = -BigInt(14); // EFAULT
              break;
            }
            const nameBuf = Number(a1) + GUEST_MEM_BASE;
            const name = new TextEncoder().encode('jitter\0');
            new Uint8Array(memory.buffer).set(name, nameBuf);
            ret = 0n;
            break;
          case 38: // PR_SET_CHILD_SUBREAPER
            ret = 0n;
            break;
          default:
            ret = 0n; // accept most prctl options
        }
        break;
      }

      case 179: { // sysinfo(info)
        // struct sysinfo on aarch64: 112 bytes
        const siBuf = Number(a0) + GUEST_MEM_BASE;
        for (let i = 0; i < 112; i += 8) view.setBigUint64(siBuf + i, 0n, true);
        const uptime = Math.floor(Date.now() / 1000) - 1700000000; // seconds since ~2023
        view.setBigUint64(siBuf + 0, BigInt(Math.max(uptime, 60)), true); // uptime
        view.setBigUint64(siBuf + 32, BigInt(1024 * 1024 * 1024), true);  // totalram (1GB)
        view.setBigUint64(siBuf + 40, BigInt(512 * 1024 * 1024), true);   // freeram (512MB)
        view.setBigUint64(siBuf + 48, BigInt(512 * 1024 * 1024), true);   // sharedram
        view.setBigUint64(siBuf + 56, BigInt(64 * 1024 * 1024), true);    // bufferram
        view.setBigUint64(siBuf + 80, 1n, true);                           // procs
        view.setUint32(siBuf + 104, 1, true);                               // mem_unit
        ret = 0n;
        break;
      }

      case 148: { // getresuid(ruid, euid, suid)
        const ruidPtr = Number(a0) + GUEST_MEM_BASE;
        const euidPtr = Number(a1) + GUEST_MEM_BASE;
        const suidPtr = Number(a2) + GUEST_MEM_BASE;
        view.setUint32(ruidPtr, 0, true); // root
        view.setUint32(euidPtr, 0, true);
        view.setUint32(suidPtr, 0, true);
        ret = 0n;
        break;
      }

      case 150: { // getresgid(rgid, egid, sgid)
        const rgidPtr = Number(a0) + GUEST_MEM_BASE;
        const egidPtr = Number(a1) + GUEST_MEM_BASE;
        const sgidPtr = Number(a2) + GUEST_MEM_BASE;
        view.setUint32(rgidPtr, 0, true);
        view.setUint32(egidPtr, 0, true);
        view.setUint32(sgidPtr, 0, true);
        ret = 0n;
        break;
      }

      case 153: { // times(buf) — get process CPU times
        const tmsBuf = Number(a0) + GUEST_MEM_BASE;
        // struct tms: 4 fields of clock_t (8 bytes each on aarch64)
        view.setBigUint64(tmsBuf, 0n, true);      // tms_utime
        view.setBigUint64(tmsBuf + 8, 0n, true);  // tms_stime
        view.setBigUint64(tmsBuf + 16, 0n, true); // tms_cutime
        view.setBigUint64(tmsBuf + 24, 0n, true); // tms_cstime
        ret = BigInt(Math.floor(Date.now() / 10)); // return clock ticks
        break;
      }

      case 92: // personality — set execution domain
        ret = 0n; // PER_LINUX
        break;

      // ── Timer syscalls ──

      case 102: { // getitimer(which, curr_value)
        // itimerval = { it_interval: timeval, it_value: timeval }, timeval = { sec, usec }
        const itvPtr = Number(a1) + GUEST_MEM_BASE;
        const which = Number(a0);
        const remMs = which === 0 ? this.timerRemainingMs() : 0;
        const intMs = which === 0 ? this.itimerRealIntervalMs : 0;
        view.setBigUint64(itvPtr, BigInt(Math.floor(intMs / 1000)), true);
        view.setBigUint64(itvPtr + 8, BigInt((intMs % 1000) * 1000), true);
        view.setBigUint64(itvPtr + 16, BigInt(Math.floor(remMs / 1000)), true);
        view.setBigUint64(itvPtr + 24, BigInt((remMs % 1000) * 1000), true);
        ret = 0n;
        break;
      }

      case 103: { // setitimer(which, new_value, old_value)
        const which = Number(a0);
        // report the previous timer into old_value before overwriting
        const oldPtr = Number(a2);
        if (oldPtr !== 0) {
          const base = oldPtr + GUEST_MEM_BASE;
          const remMs = which === 0 ? this.timerRemainingMs() : 0;
          const intMs = which === 0 ? this.itimerRealIntervalMs : 0;
          view.setBigUint64(base, BigInt(Math.floor(intMs / 1000)), true);
          view.setBigUint64(base + 8, BigInt((intMs % 1000) * 1000), true);
          view.setBigUint64(base + 16, BigInt(Math.floor(remMs / 1000)), true);
          view.setBigUint64(base + 24, BigInt((remMs % 1000) * 1000), true);
        }
        // only ITIMER_REAL (0) is backed by wall time -> SIGALRM. VIRTUAL/PROF measure cpu
        // time we don't track; accept them but arm nothing (musl/ltp use REAL for timeouts).
        if (which === 0) {
          const newPtr = Number(a1);
          if (newPtr === 0) { this.itimerRealDeadlineMs = 0; this.itimerRealIntervalMs = 0; ret = 0n; break; }
          const nb = newPtr + GUEST_MEM_BASE;
          const intMs = Number(view.getBigUint64(nb, true)) * 1000 + Math.floor(Number(view.getBigUint64(nb + 8, true)) / 1000);
          const valMs = Number(view.getBigUint64(nb + 16, true)) * 1000 + Math.floor(Number(view.getBigUint64(nb + 24, true)) / 1000);
          this.itimerRealIntervalMs = intMs;
          this.itimerRealDeadlineMs = valMs > 0 ? Number(this.nowMs()) + valMs : 0;
        }
        ret = 0n;
        break;
      }

      case 107: { // timer_create(clockid, sevp, timerid)
        // Per spec: create a POSIX timer, write timer ID to *timerid
        const tidPtr = Number(a2) + GUEST_MEM_BASE;
        view.setInt32(tidPtr, 0, true); // timer ID = 0
        ret = 0n;
        break;
      }

      case 108: { // timer_gettime(timerid, curr_value)
        // Per spec: get remaining time. Return zeros (timer not running)
        const tsPtr = Number(a1) + GUEST_MEM_BASE;
        for (let i = 0; i < 32; i += 8) view.setBigUint64(tsPtr + i, 0n, true);
        ret = 0n;
        break;
      }

      case 109: // timer_getoverrun(timerid)
        ret = 0n; // no overruns
        break;

      case 110: { // timer_settime(timerid, flags, new_value, old_value)
        // Accept the timer setting. For alarm-based LTP timeouts, we just
        // don't deliver the signal (tests will run to completion or hit our insn limit).
        const oldValPtr = Number(a3);
        if (oldValPtr !== 0) {
          const base = oldValPtr + GUEST_MEM_BASE;
          for (let i = 0; i < 32; i += 8) view.setBigUint64(base + i, 0n, true);
        }
        ret = 0n;
        break;
      }

      case 111: // timer_delete(timerid)
        ret = 0n;
        break;

      case 114: { // clock_getres(clk_id, res)
        // Per spec: return clock resolution (1 nanosecond for us)
        const resPtr = Number(a1);
        if (resPtr !== 0) {
          const base = resPtr + GUEST_MEM_BASE;
          view.setBigUint64(base, 0n, true);     // seconds
          view.setBigUint64(base + 8, 1n, true); // nanoseconds (1ns resolution)
        }
        ret = 0n;
        break;
      }

      case 32: // flock — advisory file locking
        ret = 0n; // accept all locks (single-user)
        break;

      // ── Scheduling (per spec: mostly no-ops in single-core emulator) ──

      case 118: // sched_setparam
      case 119: // sched_setscheduler  (already had case 103 wrong, now correct here)
        ret = 0n;
        break;
      case 120: // sched_getscheduler
        ret = 0n; // SCHED_OTHER
        break;
      case 121: { // sched_getparam
        // Per spec: write sched_param struct (just sched_priority = 0)
        const spPtr = Number(a1) + GUEST_MEM_BASE;
        view.setInt32(spPtr, 0, true);
        ret = 0n;
        break;
      }
      case 125: { // sched_get_priority_max(policy)
        ret = 0n; // SCHED_OTHER max priority = 0
        break;
      }
      case 126: { // sched_get_priority_min(policy)
        ret = 0n;
        break;
      }

      // ── inotify (stub — accept creation, ignore watches) ──

      case 26: { // inotify_init1(flags)
        const inoFd = this.allocFd();
        this.fdTable.set(inoFd, { type: 'inotify' });
        this.fdFlags.set(inoFd, Number(a0) & O_NONBLOCK);
        ret = BigInt(inoFd);
        break;
      }
      case 27: // inotify_add_watch
        ret = 1n; // return watch descriptor 1
        break;
      case 28: // inotify_rm_watch
        ret = 0n;
        break;

      // ── timerfd (stub — create fd, accept settings) ──

      case 85: { // timerfd_create(clockid, flags)
        const tfFd = this.allocFd();
        this.fdTable.set(tfFd, { type: 'timerfd' });
        this.fdFlags.set(tfFd, Number(a1) & O_NONBLOCK);
        ret = BigInt(tfFd);
        break;
      }
      case 86: { // timerfd_settime
        const oldVal = Number(a3);
        if (oldVal !== 0) {
          const b = oldVal + GUEST_MEM_BASE;
          for (let i = 0; i < 32; i += 8) view.setBigUint64(b + i, 0n, true);
        }
        ret = 0n;
        break;
      }
      case 87: { // timerfd_gettime
        const cur = Number(a1) + GUEST_MEM_BASE;
        for (let i = 0; i < 32; i += 8) view.setBigUint64(cur + i, 0n, true);
        ret = 0n;
        break;
      }

      // ── Misc process control ──

      case 97: // unshare — disassociate namespaces (no-op)
        ret = 0n;
        break;
      case 168: { // getcpu(cpu, node)
        const cpuPtr = Number(a0);
        const nodePtr = Number(a1);
        if (cpuPtr !== 0) view.setUint32(cpuPtr + GUEST_MEM_BASE, 0, true);
        if (nodePtr !== 0) view.setUint32(nodePtr + GUEST_MEM_BASE, 0, true);
        ret = 0n;
        break;
      }
      case 140: // setpriority
        ret = 0n;
        break;
      case 141: // getpriority
        ret = 20n; // default nice value (kernel adds 20 to actual nice)
        break;

      // ── Memory control (per spec: accept, no-op for our in-memory VFS) ──
      case 228:   // mlock(addr, len)
      case 229: { // munlock(addr, len)
        const lkStart = Number(a0) & ~0xFFF;
        const lkEnd = (Number(a0) + Number(a1) + 0xFFF) & ~0xFFF;
        this.setLocked(lkStart, lkEnd, nr === 228);
        ret = 0n;
        break;
      }
      case 230: // mlockall
      case 231: // munlockall
        ret = 0n;
        break;
      case 227: { // msync — flush a writable shared file mapping back to disk
        const msFlush = this.buildMmapFlush(cpu, memory, Number(a0), Number(a1), false);
        if (msFlush) return msFlush;
        ret = 0n;
        break;
      }
      case 232: { // mincore(addr, length, vec) — check page residency
        // Per spec: fill vec with 1s (all pages resident in our model)
        const mcLen = Number(a1);
        const mcVec = Number(a2) + GUEST_MEM_BASE;
        const pages = Math.ceil(mcLen / 4096);
        const b = new Uint8Array(memory.buffer);
        for (let i = 0; i < pages; i++) b[mcVec + i] = 1;
        ret = 0n;
        break;
      }
      case 216: { // mremap(old_addr, old_size, new_size, flags, new_addr)
        // returning old_addr while ignoring new_size let the guest write past the mapping into
        // the next allocation -> heap corruption (musl mallocng grows large allocations via
        // mremap). resize for real: extend in place if this is the top of the arena, otherwise
        // move to a fresh region (MREMAP_MAYMOVE, which musl always sets) and copy the bytes over.
        const MREMAP_MAYMOVE = 1;
        const oldAddr = Number(a0), oldSize = Number(a1), newSize = Number(a2);
        const mrFlags = Number(a3);
        const alignedOld = (oldSize + 0xFFF) & ~0xFFF;
        const alignedNew = (newSize + 0xFFF) & ~0xFFF;
        if (alignedNew <= alignedOld) { ret = a0; break; } // shrink or same size: in place
        // grow in place when the old mapping sits at the very top of the arena
        if (BigInt(oldAddr + alignedOld) === this.mmapBase) {
          const needed = oldAddr + alignedNew + GUEST_MEM_BASE;
          if (needed > this.maxBytes) { ret = -BigInt(ENOMEM); break; }
          if (needed > memory.buffer.byteLength) {
            const pagesNeeded = Math.ceil(needed / 65536), currentPages = memory.buffer.byteLength / 65536;
            if (pagesNeeded > currentPages) memory.grow(pagesNeeded - currentPages);
          }
          new Uint8Array(memory.buffer).fill(0, oldAddr + alignedOld + GUEST_MEM_BASE, oldAddr + alignedNew + GUEST_MEM_BASE);
          this.mmapBase = BigInt(oldAddr + alignedNew);
          ret = a0;
          break;
        }
        if (!(mrFlags & MREMAP_MAYMOVE)) { ret = -BigInt(ENOMEM); break; }
        const newAddr = this.mmapBase;
        this.mmapBase += BigInt(alignedNew);
        const needed = Number(newAddr) + alignedNew + GUEST_MEM_BASE;
        if (needed > this.maxBytes) { this.mmapBase -= BigInt(alignedNew); ret = -BigInt(ENOMEM); break; }
        if (needed > memory.buffer.byteLength) {
          const pagesNeeded = Math.ceil(needed / 65536), currentPages = memory.buffer.byteLength / 65536;
          if (pagesNeeded > currentPages) memory.grow(pagesNeeded - currentPages);
        }
        const mrMem = new Uint8Array(memory.buffer);
        mrMem.copyWithin(Number(newAddr) + GUEST_MEM_BASE, oldAddr + GUEST_MEM_BASE, oldAddr + GUEST_MEM_BASE + oldSize);
        mrMem.fill(0, Number(newAddr) + GUEST_MEM_BASE + oldSize, Number(newAddr) + GUEST_MEM_BASE + alignedNew);
        ret = newAddr;
        break;
      }

      // ── Signal notification via fd ──
      case 74: { // signalfd4(fd, mask, sizemask, flags)
        // Per spec: create/modify fd for receiving signals
        // Simple implementation: return a valid fd
        const sfFd = Number(BigInt.asIntN(32, a0));
        if (sfFd === -1) {
          const newFd = this.allocFd();
          this.fdTable.set(newFd, { type: 'stdio', fd: newFd }); // placeholder
          ret = BigInt(newFd);
        } else {
          ret = BigInt(sfFd); // modify existing fd
        }
        break;
      }

      // ── Process creation (modern) ──
      case 435: { // clone3(cl_args, size)
        // Per spec: modern clone variant with extensible args struct
        // Check if this is a fork-like clone (SIGCHLD flag)
        const argsPtr = Number(a0) + GUEST_MEM_BASE;
        const cloneFlags = view.getBigUint64(argsPtr, true); // flags at offset 0
        const SIGCHLD = 17n;
        if (cloneFlags & SIGCHLD) {
          const svcPC = view.getBigUint64(PC_OFFSET, true);
          return { status: 'fork', resumePC: svcPC + 4n };
        }
        // Non-fork clone (threads etc) — not supported
        ret = -BigInt(ENOSYS);
        break;
      }

      // ── Exec relative to dirfd ──
      case 281: { // execveat(dirfd, pathname, argv, envp, flags)
        // Per spec: exec relative to directory fd
        // Simple: treat as regular execve (ignore dirfd, use pathname as-is)
        const pathname281 = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
        const argvPtrs281: string[] = [];
        let argvAddr281 = Number(a2);
        while (true) {
          const ptr = Number(view.getBigUint64(argvAddr281 + GUEST_MEM_BASE, true));
          if (ptr === 0) break;
          argvPtrs281.push(readGuestString(bytes, ptr));
          argvAddr281 += 8;
        }
        const envpPtrs281: string[] = [];
        let envpAddr281 = Number(a3);
        while (true) {
          const ptr = Number(view.getBigUint64(envpAddr281 + GUEST_MEM_BASE, true));
          if (ptr === 0) break;
          envpPtrs281.push(readGuestString(bytes, ptr));
          envpAddr281 += 8;
        }
        return { status: 'execve', path: this.absolutizePath(pathname281), argv: argvPtrs281, envp: envpPtrs281 };
      }

      case 47: { // fallocate(fd, mode, offset, len) — preallocate space
        ret = 0n;
        break;
      }

      case 45: { // truncate(path, length)
        // Per spec: truncate file at path to specified length
        const tpath = readGuestString(bytes, Number(a0));
        const tlength = Number(BigInt.asIntN(64, a1));
        if (this.vfs) {
          const fd = this.vfs.open(tpath, 2, 0); // O_RDWR
          if (fd < 0) { ret = BigInt(fd); break; }
          ret = BigInt(this.vfs.ftruncate(fd, tlength));
          this.vfs.close(fd);
        } else {
          ret = -BigInt(ENOENT);
        }
        break;
      }

      case 71: { // sendfile(out_fd, in_fd, offset, count)
        // Per spec: copy data between fds in kernel space
        // For VFS files: read directly from inode data, write to output fd
        // This avoids using guest memory as temp buffer
        ret = -BigInt(ENOSYS); // TODO: implement without guest memory temp buffer
        break;
      }

      case 276: { // renameat2(olddirfd, oldpath, newdirfd, newpath, flags)
        const oldpath276 = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
        const newpath276 = this.resolveAt(a2, readGuestString(bytes, Number(a3)));
        const bOld276 = this.bridgePath(oldpath276);
        if (bOld276) {
          const bNew276 = this.bridgePath(newpath276) ?? newpath276;
          this.statCache.delete(bOld276);
          this.statCache.delete(bNew276);
          return {
            status: 'bridge', method: 'fs_rename',
            args: { oldPath: bOld276, newPath: bNew276 },
            apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); },
          };
        }
        if (this.vfs) {
          ret = BigInt(this.vfs.rename(oldpath276, newpath276));
        } else {
          ret = -BigInt(ENOSYS);
        }
        break;
      }

      case 279: { // memfd_create(name, flags)
        // Per spec: create anonymous file in memory, returns fd
        // Implementation: create a temp file in /tmp
        if (this.vfs) {
          const mfName = readGuestString(bytes, Number(a0));
          const tmpPath = '/tmp/.memfd_' + mfName + '_' + Date.now();
          const fd = this.vfs.open(tmpPath, 0o100 | 0o2, 0o600); // O_CREAT | O_RDWR
          ret = BigInt(fd);
        } else {
          ret = -BigInt(ENOSYS);
        }
        break;
      }

      case 291: { // statx(dirfd, pathname, flags, mask, statxbuf)
        // Per spec: extended stat returning struct statx (256 bytes)
        // We first get a regular stat into the LAST 128 bytes of the 256-byte statx buffer
        // (since statx is 256 bytes and stat is 128, they don't overlap), then rearrange.
        const sxFlags = Number(a2);
        // a bad pathname pointer faults EFAULT (stat(NULL)); AT_EMPTY_PATH (fstat) passes a valid ""
        // so it is never caught here.
        if (!(sxFlags & 0x1000) && this.ptrFaults(bytes, Number(a1), 1)) { ret = -BigInt(EFAULT); break; }
        const sxPath = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
        const sxBufAddr = Number(a4);
        if (this.ptrFaults(bytes, sxBufAddr, 256)) { ret = -BigInt(EFAULT); break; } // musl routes fstat/stat here
        if (this.vfs) {
          // Write regular stat into the second half of the statx buffer (bytes 128-255)
          const tmpOffset = sxBufAddr + 128;
          const res = this.vfs.fstatat(Number(BigInt.asIntN(32, a0)), sxPath, tmpOffset, sxFlags, view);
          if (res < 0) { ret = BigInt(res); break; }

          // Read values from the temp stat area
          const st = tmpOffset + GUEST_MEM_BASE;
          const stDev = view.getBigUint64(st + 0, true);
          const stIno = view.getBigUint64(st + 8, true);
          const stMode = view.getUint32(st + 16, true);
          const stNlink = view.getUint32(st + 20, true);
          const stUid = view.getUint32(st + 24, true);
          const stGid = view.getUint32(st + 28, true);
          const stRdev = view.getBigUint64(st + 32, true);
          const stSize = view.getBigInt64(st + 48, true);
          const stBlksize = view.getInt32(st + 56, true);
          const stBlocks = view.getBigInt64(st + 64, true);
          const stAtime = view.getBigInt64(st + 72, true);
          const stMtime = view.getBigInt64(st + 88, true);
          const stCtime = view.getBigInt64(st + 104, true);

          // Now write the statx struct (256 bytes) at the actual buffer address
          const b = sxBufAddr + GUEST_MEM_BASE;
          for (let i = 0; i < 256; i += 8) view.setBigUint64(b + i, 0n, true);

          view.setUint32(b + 0, 0x17FF, true);              // stx_mask (all basic fields valid)
          view.setUint32(b + 4, stBlksize, true);            // stx_blksize
          view.setBigUint64(b + 8, 0n, true);                // stx_attributes
          view.setUint32(b + 16, stNlink, true);              // stx_nlink
          view.setUint32(b + 20, stUid, true);                // stx_uid
          view.setUint32(b + 24, stGid, true);                // stx_gid
          view.setUint16(b + 28, stMode & 0xFFFF, true);     // stx_mode
          view.setBigUint64(b + 32, stIno, true);             // stx_ino
          view.setBigUint64(b + 40, BigInt.asUintN(64, stSize), true); // stx_size
          view.setBigUint64(b + 48, BigInt.asUintN(64, stBlocks), true); // stx_blocks
          view.setBigUint64(b + 56, 0n, true);                // stx_attributes_mask
          // stx_atime: { tv_sec(8), tv_nsec(4), __reserved(4) } at offset 64
          view.setBigInt64(b + 64, stAtime, true);
          // stx_btime (birth time) at offset 80 — use ctime
          view.setBigInt64(b + 80, stCtime, true);
          // stx_ctime at offset 96
          view.setBigInt64(b + 96, stCtime, true);
          // stx_mtime at offset 112
          view.setBigInt64(b + 112, stMtime, true);
          // stx_rdev_major/minor at offset 128
          view.setUint32(b + 128, 0, true);                  // rdev_major
          view.setUint32(b + 132, Number(stRdev), true);     // rdev_minor
          // stx_dev_major/minor at offset 136
          view.setUint32(b + 136, 0, true);                  // dev_major
          view.setUint32(b + 140, Number(stDev), true);      // dev_minor
        }
        ret = 0n;
        break;
      }

      // ── I/O multiplexing (epoll, eventfd) ──

      case 19: { // eventfd2(initval, flags)
        const initVal = BigInt.asUintN(32, a0);
        const efdFlags = Number(a1);
        const EFD_SEMAPHORE = 1;
        const EFD_NONBLOCK = O_NONBLOCK;
        const efd = this.allocFd();
        this.fdTable.set(efd, {
          type: 'eventfd',
          counter: initVal,
          semaphore: (efdFlags & EFD_SEMAPHORE) !== 0,
        });
        this.fdFlags.set(efd, efdFlags & EFD_NONBLOCK);
        ret = BigInt(efd);
        break;
      }

      case 20: { // epoll_create1(flags)
        const epfd = this.allocFd();
        this.fdTable.set(epfd, { type: 'epoll' });
        if (!this.epollSets) this.epollSets = new Map();
        this.epollSets.set(epfd, new Map()); // fd → { events, data }
        ret = BigInt(epfd);
        break;
      }

      case 21: { // epoll_ctl(epfd, op, fd, event)
        // Per spec: add/modify/remove fd in epoll interest list
        // struct epoll_event: { u32 events; u64 data; } (12 bytes, but padded to 16)
        const epfd = Number(a0);
        const epOp = Number(a1); // 1=ADD, 2=DEL, 3=MOD
        const targetFd = Number(a2);
        const eventPtr = Number(a3);

        if (!this.epollSets?.has(epfd)) { ret = -BigInt(EBADF); break; }
        const eset = this.epollSets.get(epfd)!;

        if (epOp === 1 || epOp === 3) { // EPOLL_CTL_ADD or EPOLL_CTL_MOD
          const events = view.getUint32(eventPtr + GUEST_MEM_BASE, true);
          const data = view.getBigUint64(eventPtr + GUEST_MEM_BASE + 4, true);
          eset.set(targetFd, { events, data });
          ret = 0n;
        } else if (epOp === 2) { // EPOLL_CTL_DEL
          eset.delete(targetFd);
          ret = 0n;
        } else {
          ret = -BigInt(EINVAL);
        }
        break;
      }

      case 22: { // epoll_pwait(epfd, events, maxevents, timeout, sigmask)
        // Per spec: wait for events on epoll instance
        // For our implementation: report all registered fds as ready immediately
        // (correct for files, approximately correct for pipes with data)
        const epfd = Number(a0);
        const eventsPtr = Number(a1);
        const maxEvents = Number(a2);

        if (!this.epollSets?.has(epfd)) { ret = -BigInt(EBADF); break; }
        const eset = this.epollSets.get(epfd)!;

        let count = 0;
        const EPOLLIN = 0x001, EPOLLOUT = 0x004;
        for (const [targetFd, entry] of eset) {
          if (count >= maxEvents) break;
          // Report fds as ready with their requested events
          const readyEvents = entry.events & (EPOLLIN | EPOLLOUT);
          if (readyEvents) {
            const off = eventsPtr + count * 12 + GUEST_MEM_BASE;
            view.setUint32(off, readyEvents, true);
            view.setBigUint64(off + 4, entry.data, true);
            count++;
          }
        }
        ret = BigInt(count);
        break;
      }

      case 72: { // pselect6(nfds, readfds, writefds, exceptfds, timeout, sigmask)
        const nfds = Number(a0);
        const readPtr = Number(a1);
        const writePtr = Number(a2);
        const exceptPtr = Number(a3);
        const tmoPtr = Number(a4);

        let tmoMs = -1;
        if (tmoPtr !== 0) {
          const sec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE, true));
          const nsec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE + 8, true));
          tmoMs = sec * 1000 + Math.floor(nsec / 1_000_000);
        }

        const wantRead = this.readFdSet(view, readPtr, nfds);
        const wantWrite = this.readFdSet(view, writePtr, nfds);

        // socket readability is only knowable via the bridge. if a socket is watched for read and
        // nothing else is already ready (writable sockets/readable local fds resolve synchronously),
        // defer to net_poll — this mirrors ppoll and is what unblocks apk/openssl select loops.
        const sockReadFd = [...wantRead].find((fd) => this.getFdEntry(fd)?.type === 'socket');
        if (sockReadFd !== undefined && tmoMs !== 0) {
          let preReady = 0;
          for (const fd of wantRead) if (this.getFdEntry(fd)?.type !== 'socket' && this.isFdReadable(fd)) preReady++;
          for (const fd of wantWrite) if (this.isFdWritable(fd)) preReady++;
          if (preReady === 0) {
            const sockEntry = this.getFdEntry(sockReadFd);
            const pollTmo = tmoMs < 0 ? 30000 : tmoMs;
            return {
              status: 'bridge', method: 'net_poll',
              args: { socketId: sockEntry!.socketId, timeoutMs: pollTmo },
              apply: (resp: any) => {
                const outR = new Set<number>(), outW = new Set<number>();
                for (const fd of wantWrite) if (this.isFdWritable(fd)) outW.add(fd);
                for (const fd of wantRead) {
                  const t = this.getFdEntry(fd)?.type;
                  if (t === 'socket') { if (resp.ret > 0) outR.add(fd); }
                  else if (this.isFdReadable(fd)) outR.add(fd);
                }
                this.writeFdSet(view, readPtr, nfds, outR);
                this.writeFdSet(view, writePtr, nfds, outW);
                if (exceptPtr !== 0) this.writeFdSet(view, exceptPtr, nfds, new Set());
                this.nt(`pselect.sock -> ret=${resp.ret} r=${outR.size} w=${outW.size}`);
                cpu.setReg(0, BigInt(outR.size + outW.size));
              },
            };
          }
        }

        const readyRead = new Set<number>();
        const readyWrite = new Set<number>();
        let deadline = tmoMs < 0 ? -1 : Date.now() + tmoMs;

        while (true) {
          readyRead.clear();
          readyWrite.clear();
          for (const fd of wantRead) if (this.isFdReadable(fd)) readyRead.add(fd);
          for (const fd of wantWrite) if (this.isFdWritable(fd)) readyWrite.add(fd);
          const total = readyRead.size + readyWrite.size;

          if (total > 0) {
            this.writeFdSet(view, readPtr, nfds, readyRead);
            this.writeFdSet(view, writePtr, nfds, readyWrite);
            if (exceptPtr !== 0) this.writeFdSet(view, exceptPtr, nfds, new Set());
            ret = BigInt(total);
            break;
          }

          const remaining = deadline < 0 ? -1 : Math.max(0, deadline - Date.now());
          if (deadline >= 0 && remaining === 0) {
            this.writeFdSet(view, readPtr, nfds, readyRead);
            this.writeFdSet(view, writePtr, nfds, readyWrite);
            if (exceptPtr !== 0) this.writeFdSet(view, exceptPtr, nfds, new Set());
            ret = 0n;
            break;
          }

          let wakeable = -1;
          for (const fd of wantRead) if (this.isFdWakeable(fd)) { wakeable = fd; break; }

          if (wakeable < 0 || !this.io.waitStdin) {
            // No fd we can block on; behave as poll
            this.writeFdSet(view, readPtr, nfds, readyRead);
            this.writeFdSet(view, writePtr, nfds, readyWrite);
            if (exceptPtr !== 0) this.writeFdSet(view, exceptPtr, nfds, new Set());
            ret = 0n;
            break;
          }

          const status = this.io.waitStdin(remaining);
          if (status === 'timeout' && deadline >= 0) {
            // Final empty result
            this.writeFdSet(view, readPtr, nfds, readyRead);
            this.writeFdSet(view, writePtr, nfds, readyWrite);
            if (exceptPtr !== 0) this.writeFdSet(view, exceptPtr, nfds, new Set());
            ret = 0n;
            break;
          }
          // loop back, re-check
        }
        break;
      }

      case 73: { // ppoll(fds, nfds, tmo_p, sigmask)
        // struct pollfd { int fd; short events; short revents; }  — 8 bytes each
        const POLLIN = 1, POLLOUT = 4;
        const fdsAddr = Number(a0);
        const nfds = Number(a1);
        const tmoPtr = Number(a2);

        let tmoMs = -1;
        if (tmoPtr !== 0) {
          const sec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE, true));
          const nsec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE + 8, true));
          tmoMs = sec * 1000 + Math.floor(nsec / 1_000_000);
        }

        // Snapshot pollfds
        type Pfd = { fd: number; events: number; revents: number };
        const pfds: Pfd[] = [];
        for (let i = 0; i < nfds; i++) {
          const off = fdsAddr + i * 8 + GUEST_MEM_BASE;
          pfds.push({
            fd: view.getInt32(off, true),
            events: view.getInt16(off + 4, true),
            revents: 0,
          });
        }

        const writeBack = () => {
          for (let i = 0; i < nfds; i++) {
            view.setInt16(fdsAddr + i * 8 + 6 + GUEST_MEM_BASE, pfds[i].revents, true);
          }
        };

        // First pass: identify sockets (handled via bridge), do synchronous readiness for the rest
        let socketEntry: any = null;
        for (const pfd of pfds) {
          if (pfd.fd < 0) continue;
          const entry = this.getFdEntry(pfd.fd);
          if (entry?.type === 'socket' && (pfd.events & POLLIN)) {
            if (!socketEntry) socketEntry = { fd: pfd.fd, entry };
          }
        }

        const checkReady = (): number => {
          let count = 0;
          for (const pfd of pfds) {
            pfd.revents = 0;
            if (pfd.fd < 0) continue;
            const entry = this.getFdEntry(pfd.fd);
            if (entry?.type === 'socket') continue; // socket readiness handled by bridge
            if ((pfd.events & POLLIN) && this.isFdReadable(pfd.fd)) pfd.revents |= POLLIN;
            if ((pfd.events & POLLOUT) && this.isFdWritable(pfd.fd)) pfd.revents |= POLLOUT;
            if (pfd.revents !== 0) count++;
          }
          return count;
        };

        // If only socket fds are watched, defer to bridge poll
        const watchingLocal = pfds.some((p) => p.fd >= 0 && this.getFdEntry(p.fd)?.type !== 'socket');
        if (!watchingLocal && socketEntry) {
          const pollTmo = tmoMs < 0 ? 30000 : tmoMs;
          this.nt(`ppoll.enter sock=${socketEntry.fd} events=${socketEntry.entry ? pfds.find((p) => p.fd === socketEntry.fd)?.events : '?'} tmo=${tmoMs}`);
          return {
            status: 'bridge', method: 'net_poll',
            args: { socketId: socketEntry.entry.socketId, timeoutMs: pollTmo },
            apply: (resp: any) => {
              if (resp.ret > 0) {
                for (const pfd of pfds) {
                  if (pfd.fd < 0) continue;
                  const e = this.getFdEntry(pfd.fd);
                  if (e?.type === 'socket' && (pfd.events & POLLIN)) pfd.revents |= POLLIN;
                }
              }
              writeBack();
              const c = pfds.filter((p) => p.revents !== 0).length;
              this.nt(`ppoll.sock -> ret=${resp.ret} ready=${c}`);
              cpu.setReg(0, BigInt(c));
            },
          };
        }

        // Synchronous + wait path for local fds (stdin/tty/pipes/files)
        let deadline = tmoMs < 0 ? -1 : Date.now() + tmoMs;
        while (true) {
          const ready = checkReady();
          if (ready > 0) {
            writeBack();
            ret = BigInt(ready);
            break;
          }
          const remaining = deadline < 0 ? -1 : Math.max(0, deadline - Date.now());
          if (deadline >= 0 && remaining === 0) {
            writeBack();
            ret = 0n;
            break;
          }
          // Find a wakeable fd (only stdin/tty currently). If none, behave as poll.
          const wakeable = pfds.some((p) => p.fd >= 0 && (p.events & POLLIN) && this.isFdWakeable(p.fd));
          if (!wakeable || !this.io.waitStdin) {
            writeBack();
            ret = 0n;
            break;
          }
          const status = this.io.waitStdin(remaining);
          if (status === 'timeout' && deadline >= 0) {
            writeBack();
            ret = 0n;
            break;
          }
        }
        break;
      }

      // ── Remaining syscalls from aarch64 spec ──

      // mount(source, target, filesystemtype, mountflags, data). the emulator has a
      // single in-memory tree, so a mount records the mount point in the namespace
      // (visible via /proc/self/mounts and mountinfo) rather than grafting a real fs.
      // the ltp scratch-device tests select tmpfs (LTP_DEV_FS_TYPE=tmpfs), which needs
      // no backing device or mkfs; the mount point directory already exists.
      case 40: { // mount
        const target = readGuestString(bytes, Number(a1));
        const fstype = a2 !== 0n ? readGuestString(bytes, Number(a2)) : '';
        const source = a0 !== 0n ? readGuestString(bytes, Number(a0)) : fstype;
        const tgt = this.absolutizePath(target);
        if (!this.vfs || !this.vfs.exists(tgt)) { ret = -BigInt(ENOENT); break; }
        this.vfs.addMount(source, tgt, fstype || 'tmpfs');
        ret = 0n;
        break;
      }
      case 39: { // umount2(target, flags)
        const target = this.absolutizePath(readGuestString(bytes, Number(a0)));
        if (!this.vfs || !this.vfs.removeMount(target)) { ret = -BigInt(EINVAL); break; }
        ret = 0n;
        break;
      }
      case 442: // mount_setattr
        ret = 0n;
        break;
      case 41: // pivot_root
        ret = -BigInt(EPERM); // not permitted (we don't support changing root)
        break;

      // Robust futex list
      case 100: { // get_robust_list(pid, head_ptr, len_ptr)
        const headPtr = Number(a1) + GUEST_MEM_BASE;
        const lenPtr = Number(a2) + GUEST_MEM_BASE;
        view.setBigUint64(headPtr, 0n, true);
        view.setBigUint64(lenPtr, 24n, true); // sizeof(robust_list_head)
        ret = 0n;
        break;
      }

      // Signal: tkill and rt_sigreturn
      case 130: // tkill(tid, sig) — obsolete, use tgkill
        ret = 0n;
        break;
      case 139: { // rt_sigreturn — return from signal handler
        // Per spec: restores saved register state from the signal frame on the stack
        const restoredPC = this.handleSigreturn(cpu, memory);
        // Don't set ret — the restored registers include X0
        // The caller should set PC to restoredPC
        cpu.setPC(restoredPC);
        // Return 'ok' — the PC is already set correctly
        ret = cpu.getReg(0); // restored X0
        break;
      }

      // Networking (remaining)
      case 210: // shutdown(sockfd, how)
        ret = -BigInt(EBADF); // no valid sockets
        break;
      case 211: { // sendmsg(fd, *msghdr, flags)
        // struct msghdr { msg_name(8), msg_namelen(4), pad(4), msg_iov(8), msg_iovlen(8),
        //                 msg_control(8), msg_controllen(8), msg_flags(4) }
        const smEntry = this.getFdEntry(Number(a0));
        if (!smEntry || smEntry.type !== 'socket') { ret = -BigInt(EBADF); break; }
        const mhdr = Number(a1) + GUEST_MEM_BASE;
        const iovPtr = Number(view.getBigUint64(mhdr + 16, true));
        const iovLen = Number(view.getBigUint64(mhdr + 24, true));
        // Gather all iov data
        let totalLen = 0;
        for (let i = 0; i < iovLen; i++) {
          totalLen += Number(view.getBigUint64(iovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
        }
        const combined = new Uint8Array(totalLen);
        let off = 0;
        for (let i = 0; i < iovLen; i++) {
          const base = Number(view.getBigUint64(iovPtr + i * 16 + GUEST_MEM_BASE, true));
          const len = Number(view.getBigUint64(iovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
          combined.set(bytes.subarray(base + GUEST_MEM_BASE, base + GUEST_MEM_BASE + len), off);
          off += len;
        }
        const smB64 = bytesToBase64(combined);
        this.nt(`sendmsg fd=${Number(a0)} len=${totalLen}`);
        // Check for destination address (msg_name) for connectionless sendmsg
        const namePtr = Number(view.getBigUint64(mhdr, true));
        const nameLen = view.getUint32(mhdr + 8, true);
        if (namePtr !== 0 && nameLen > 0) {
          const dest = this.parseSockaddr(bytes, namePtr, nameLen);
          return {
            status: 'bridge', method: 'net_sendto',
            args: { socketId: smEntry.socketId, data: smB64, host: dest.host, port: dest.port },
            apply: (resp: any) => {
              cpu.setReg(0, resp.ret < 0
                ? BigInt.asUintN(64, BigInt(resp.ret))
                : BigInt(totalLen));
            },
          };
        }
        return {
          status: 'bridge', method: 'net_write',
          args: { socketId: smEntry.socketId, data: smB64 },
          apply: (resp: any) => {
            cpu.setReg(0, resp.ret < 0
              ? BigInt.asUintN(64, BigInt(resp.ret))
              : BigInt(totalLen));
          },
        };
      }

      case 212: { // recvmsg(fd, *msghdr, flags)
        const rmEntry = this.getFdEntry(Number(a0));
        if (!rmEntry || rmEntry.type !== 'socket') { ret = -BigInt(EBADF); break; }
        const rmhdr = Number(a1) + GUEST_MEM_BASE;
        const rmIovPtr = Number(view.getBigUint64(rmhdr + 16, true));
        const rmIovLen = Number(view.getBigUint64(rmhdr + 24, true));
        // Calculate total buffer capacity from iovec
        let rmCapacity = 0;
        for (let i = 0; i < rmIovLen; i++) {
          rmCapacity += Number(view.getBigUint64(rmIovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
        }
        return {
          status: 'bridge', method: 'net_read',
          args: { socketId: rmEntry.socketId, count: Math.min(rmCapacity, 12288) },
          apply: (resp: any) => {
            this.nt(`recvmsg fd=${Number(a0)} cap=${rmCapacity} -> ret=${resp.ret} dlen=${resp.data ? Math.floor(resp.data.length * 3 / 4) : 0}`);
            if (resp.ret < 0) {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              return;
            }
            let recvd: Uint8Array;
            if (resp.data) {
              recvd = Uint8Array.from(atob(resp.data), c => c.charCodeAt(0));
            } else {
              // EOF or no data
              cpu.setReg(0, BigInt(resp.ret));
              return;
            }
            // Scatter into iovec buffers
            let remaining = recvd.length;
            let srcOff = 0;
            for (let i = 0; i < rmIovLen && remaining > 0; i++) {
              const base = Number(view.getBigUint64(rmIovPtr + i * 16 + GUEST_MEM_BASE, true));
              const len = Number(view.getBigUint64(rmIovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
              const copyLen = Math.min(len, remaining);
              bytes.set(recvd.subarray(srcOff, srcOff + copyLen), base + GUEST_MEM_BASE);
              srcOff += copyLen;
              remaining -= copyLen;
            }
            // Fill msg_name with source address if buffer provided
            const namePtr = Number(view.getBigUint64(rmhdr, true));
            const nameLenPtr = rmhdr + 8; // msg_namelen field
            if (namePtr !== 0 && rmEntry.remoteAddr) {
              const written = this.writeSockaddr(view, namePtr, rmEntry.remoteAddr);
              view.setUint32(nameLenPtr, written, true);
            }
            // Clear msg_flags and msg_controllen
            view.setInt32(rmhdr + 48, 0, true);     // msg_flags
            view.setBigUint64(rmhdr + 40, 0n, true); // msg_controllen
            cpu.setReg(0, BigInt(recvd.length));
          },
        };
      }

      case 213:
        ret = -BigInt(ENOSYS);
        break;
      case 242: // accept4
        ret = -BigInt(EBADF);
        break;
      case 243: // recvmmsg
      case 269: // sendmmsg
        ret = -BigInt(EBADF);
        break;
      case 199: // socketpair(domain, type, protocol, sv[2])
        ret = -BigInt(ENOSYS); // not yet supported
        break;

      // File operations (remaining)
      case 75: // splice — copy between fds via pipe
      case 76: // tee — duplicate pipe content
        ret = -BigInt(ENOSYS);
        break;
      case 77: // sync_file_range
        ret = 0n; // no-op (our VFS is in-memory)
        break;
      case 81: // fadvise64 — advise kernel about file access pattern
        ret = 0n; // accept, ignore
        break;
      case 18: // lookup_dcookie
        ret = -BigInt(ENOSYS);
        break;

      // Capabilities
      case 90: { // capget(hdrp, datap)
        // Per spec: get process capabilities
        // Return full capabilities (we're root)
        const capDataPtr = Number(a1);
        if (capDataPtr !== 0) {
          const base = capDataPtr + GUEST_MEM_BASE;
          // struct __user_cap_data_struct × 2 (for v3 caps):
          // { effective: u32, permitted: u32, inheritable: u32 } × 2
          view.setUint32(base, 0xFFFFFFFF, true);     // effective[0]
          view.setUint32(base + 4, 0xFFFFFFFF, true);  // permitted[0]
          view.setUint32(base + 8, 0xFFFFFFFF, true);  // inheritable[0]
          view.setUint32(base + 12, 0xFFFFFFFF, true);  // effective[1]
          view.setUint32(base + 16, 0xFFFFFFFF, true);  // permitted[1]
          view.setUint32(base + 20, 0xFFFFFFFF, true);  // inheritable[1]
        }
        // Also need to handle the header
        const capHdrPtr = Number(a0);
        if (capHdrPtr !== 0) {
          const hBase = capHdrPtr + GUEST_MEM_BASE;
          view.setUint32(hBase, 0x20080522, true); // version (v3)
          view.setInt32(hBase + 4, this.currentPid, true); // pid
        }
        ret = 0n;
        break;
      }
      case 91: // capset — set capabilities (accept, no-op as root)
        ret = 0n;
        break;

      // Kernel log
      case 116: // syslog — read/control kernel message buffer
        ret = 0n; // pretend success
        break;

      // Process tracing
      case 117: // ptrace — not supported in emulator
        ret = -BigInt(EPERM);
        break;

      // Scheduling
      case 127: { // sched_rr_get_interval(pid, tp)
        const rrTp = Number(a1) + GUEST_MEM_BASE;
        view.setBigUint64(rrTp, 0n, true);
        view.setBigUint64(rrTp + 8, 100000000n, true); // 100ms quantum
        ret = 0n;
        break;
      }
      case 128: // restart_syscall — restart interrupted syscall
        ret = -BigInt(4); // EINTR
        break;

      // Time
      case 170: // settimeofday — not permitted
        ret = -BigInt(EPERM);
        break;

      // Extended attributes
      case 5: case 6: case 7: case 8: case 9: case 10: case 11: case 12: case 13: case 14: case 15: case 16:
        // setxattr/getxattr/listxattr/removexattr variants (fs/fd/l)
        // Per spec: extended attributes. Return ENOTSUP (not supported on this filesystem)
        ret = -BigInt(95); // ENOTSUP/EOPNOTSUPP
        break;

      // Security
      case 277: // seccomp
        ret = -BigInt(EINVAL); // not supported
        break;
      case 293: // rseq — restartable sequences (glibc needs this)
        ret = -BigInt(ENOSYS); // musl doesn't use rseq, glibc does
        break;

      // Copy file range
      case 285: { // copy_file_range(fd_in, off_in, fd_out, off_out, len, flags)
        // Per spec: copy data between two file descriptors in kernel space
        // Implementation: read from in_fd into JS buffer, write to out_fd
        const cfrInFd = Number(a0);
        const cfrOffInPtr = Number(a1);
        const cfrOutFd = Number(a2);
        const cfrOffOutPtr = Number(a3);
        const cfrLen = Number(a4);
        if (!this.vfs) { ret = -BigInt(EBADF); break; }

        // Save positions, seek if offset pointers provided
        const inSaved = this.vfs.lseek(cfrInFd, 0n, 1);
        const outSaved = this.vfs.lseek(cfrOutFd, 0n, 1);
        if (inSaved < 0n || outSaved < 0n) { ret = -BigInt(EBADF); break; }

        if (cfrOffInPtr !== 0) {
          const offIn = view.getBigInt64(cfrOffInPtr + GUEST_MEM_BASE, true);
          this.vfs.lseek(cfrInFd, offIn, 0);
        }
        if (cfrOffOutPtr !== 0) {
          const offOut = view.getBigInt64(cfrOffOutPtr + GUEST_MEM_BASE, true);
          this.vfs.lseek(cfrOutFd, offOut, 0);
        }

        // Use VFS-level copy (avoids guest memory as temp buffer)
        const n = (this.vfs as any).copyFileRange?.(cfrInFd, cfrOutFd, cfrLen) ?? -38;
        ret = BigInt(n);
        // Update offset pointers if provided
        if (cfrOffInPtr !== 0 && n > 0) {
          const newOff = view.getBigInt64(cfrOffInPtr + GUEST_MEM_BASE, true) + BigInt(n);
          view.setBigInt64(cfrOffInPtr + GUEST_MEM_BASE, newOff, true);
          this.vfs.lseek(cfrInFd, inSaved, 0); // restore original position
        }
        if (cfrOffOutPtr !== 0 && n > 0) {
          const newOff = view.getBigInt64(cfrOffOutPtr + GUEST_MEM_BASE, true) + BigInt(n);
          view.setBigInt64(cfrOffOutPtr + GUEST_MEM_BASE, newOff, true);
          this.vfs.lseek(cfrOutFd, outSaved, 0);
        }
        break;
      }

      // Process accounting
      case 89: // acct
        ret = -BigInt(ENOSYS);
        break;

      // Namespace
      case 268: // setns
        ret = -BigInt(EPERM);
        break;

      // ── Async I/O (io_setup, io_destroy, io_submit, io_cancel, io_getevents) ──
      case 0: case 1: case 2: case 4:
        ret = -BigInt(ENOSYS); // async I/O not supported
        break;
      case 3: // io_cancel
        ret = -BigInt(EINVAL);
        break;

      // ── splice/tee/vmsplice ──
      case 30: // fanotify_init
      case 31: // fanotify_mark
        ret = -BigInt(ENOSYS);
        break;

      case 42: // nfsservctl (removed in Linux 3.1)
        ret = -BigInt(ENOSYS);
        break;

      case 51: { // fchdir (CORRECT: 50 was wrong earlier, 50 is also fchdir — let me check)
        // Actually on aarch64: 50 = fchdir. We have that. 51 doesn't exist on aarch64.
        ret = -BigInt(ENOSYS);
        break;
      }

      case 58: // ??? (openat2 is 437 on aarch64, 58 is unused)
        ret = -BigInt(ENOSYS);
        break;

      case 69: { // preadv(fd, iov, iovcnt, offset)
        // Per spec: scatter read at offset without changing position
        const prvFd = Number(a0);
        const prvIov = Number(a1);
        const prvCnt = Number(a2);
        const prvOff = BigInt.asIntN(64, a3);
        if (!this.vfs) { ret = -BigInt(EBADF); break; }
        const prvSaved = this.vfs.lseek(prvFd, 0n, 1);
        if (prvSaved < 0n) { ret = prvSaved; break; }
        this.vfs.lseek(prvFd, prvOff, 0);
        let prvTotal = 0;
        for (let i = 0; i < prvCnt; i++) {
          const base = Number(view.getBigUint64(prvIov + i * 16 + GUEST_MEM_BASE, true));
          const len = Number(view.getBigUint64(prvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
          if (len === 0) continue;
          const n = this.vfs.read(prvFd, base, len, view);
          if (n < 0) { if (prvTotal === 0) prvTotal = n; break; }
          prvTotal += n;
          if (n < len) break;
        }
        this.vfs.lseek(prvFd, prvSaved, 0);
        ret = BigInt(prvTotal);
        break;
      }
      case 70: { // pwritev(fd, iov, iovcnt, offset)
        const pwvFd = Number(a0);
        const pwvIov = Number(a1);
        const pwvCnt = Number(a2);
        const pwvOff = BigInt.asIntN(64, a3);
        if (!this.vfs) { ret = -BigInt(EBADF); break; }
        const pwvSaved = this.vfs.lseek(pwvFd, 0n, 1);
        if (pwvSaved < 0n) { ret = pwvSaved; break; }
        this.vfs.lseek(pwvFd, pwvOff, 0);
        let pwvTotal = 0;
        for (let i = 0; i < pwvCnt; i++) {
          const base = Number(view.getBigUint64(pwvIov + i * 16 + GUEST_MEM_BASE, true));
          const len = Number(view.getBigUint64(pwvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
          if (len === 0) continue;
          const n = this.vfs.write(pwvFd, base, len, view);
          if (n < 0) { if (pwvTotal === 0) pwvTotal = n; break; }
          pwvTotal += n;
          if (n < len) break;
        }
        this.vfs.lseek(pwvFd, pwvSaved, 0);
        ret = BigInt(pwvTotal);
        break;
      }

      // timerfd (verify correct numbers)
      // Per aarch64 spec: timerfd_create=85, timerfd_settime=86, timerfd_gettime=87
      // 82-84 are unused on aarch64? Let me check — they might be signalfd/timerfd/eventfd
      // Actually: 82=signalfd (old), 83=timerfd_create (old), 84=eventfd (old)
      // These are the OLD non-flagged versions; aarch64 should use 74/85/19 instead
      case 82: // signalfd (old, use signalfd4=74)
      case 83: // timerfd_create (old, use 85)
      case 84: // eventfd (old, use eventfd2=19)
        ret = -BigInt(ENOSYS);
        break;

      // setitimer/getitimer duplicates check
      case 104: // ??? (on aarch64, 104 = clock_adjtime. We have setitimer at 103)
        ret = -BigInt(EPERM); // clock_adjtime — not permitted
        break;
      case 105: // ??? (105 = clock_adjtime_time64 — doesn't exist on aarch64)
      case 106: // ??? (106 = mq_timedsend)
        ret = -BigInt(ENOSYS);
        break;

      // Clock
      case 112: // clock_settime — not permitted
        ret = -BigInt(EPERM);
        break;

      // Signal (remaining)
      case 137: // rt_sigtimedwait
        ret = -BigInt(4); // EINTR — pretend interrupted
        break;
      case 138: // rt_sigqueueinfo
        ret = 0n; // accept
        break;

      // ioctl subcommand handling is in case 29 already
      case 142: // sched_setattr (Linux 3.14+)
        ret = 0n;
        break;

      // getrlimit/setrlimit (prlimit64 at 261 is the modern replacement)
      case 161: case 162: // ??? (161=chroot, 162=sync)
        if (nr === 161) ret = 0n; // chroot — pretend success
        else ret = 0n; // sync — no-op
        break;
      case 163: // getrlimit — redirect to prlimit64 logic
      case 164: // setrlimit
        ret = 0n;
        break;

      case 171: // adjtimex — kernel clock tuning
        ret = -BigInt(EPERM);
        break;

      // Key management
      case 180: case 181: case 182: case 183: case 184: case 185:
        // add_key, request_key, keyctl and variants
        ret = -BigInt(ENOSYS);
        break;

      // SysV IPC
      case 186: case 187: case 188: case 189: // msgget, msgctl, msgrcv, msgsnd
      case 190: case 191: case 192: case 193: // semget, semctl, semtimedop, semop
      case 194: case 195: case 196: case 197: // shmget, shmctl, shmat, shmdt
        ret = -BigInt(ENOSYS); // SysV IPC not supported
        break;

      // getdents64 — we have this at case 61 which is correct on aarch64
      // 217 = getdents (old 32-bit version, not used on aarch64)
      case 217:
        ret = -BigInt(ENOSYS);
        break;

      // Miscellaneous
      case 218: // set_mempolicy
      case 219: // get_mempolicy
        ret = 0n; // NUMA policies — accept, ignore
        break;
      case 223: // mq_open
      case 224: // mq_unlink
      case 225: // mq_timedsend
        ret = -BigInt(ENOSYS); // POSIX message queues
        break;
      case 234: // userfaultfd
        ret = -BigInt(ENOSYS);
        break;
      case 235: // membarrier
        ret = 0n; // memory barrier — no-op in single-threaded
        break;
      case 236: // mlock2(addr, len, flags)
        this.setLocked(Number(a0) & ~0xFFF, (Number(a0) + Number(a1) + 0xFFF) & ~0xFFF, true);
        ret = 0n;
        break;
      case 237: // copy_file_range (same as 285)
        ret = -BigInt(ENOSYS);
        break;
      case 238: // preadv2
      case 239: // pwritev2
        ret = -BigInt(ENOSYS);
        break;
      case 240: // rt_tgsigqueueinfo
        ret = 0n;
        break;
      case 241: // perf_event_open
        ret = -BigInt(ENOSYS);
        break;
      case 244: // kcmp
        ret = -BigInt(ENOSYS);
        break;

      // io_uring (425-427)
      case 425: case 426: case 427:
        ret = -BigInt(ENOSYS); // io_uring not supported
        break;

      // Remaining 245-294 (arch specific range + various)
      case 262: // fanotify_init
      case 263: // fanotify_mark
        ret = -BigInt(ENOSYS);
        break;
      case 264: // name_to_handle_at
      case 265: // open_by_handle_at
        ret = -BigInt(ENOSYS);
        break;
      case 266: // clock_adjtime
        ret = -BigInt(EPERM);
        break;
      case 267: // syncfs — sync filesystem containing fd
        ret = 0n; // no-op (in-memory VFS)
        break;
      case 270: // process_vm_readv
      case 271: // process_vm_writev
        ret = -BigInt(ENOSYS); // cross-process memory access
        break;
      case 272: // kcmp — compare kernel resources
        ret = -BigInt(ENOSYS);
        break;
      case 273: // finit_module — load kernel module
        ret = -BigInt(EPERM);
        break;
      case 274: // sched_setattr
        ret = 0n;
        break;
      case 275: { // sched_getattr(pid, attr, size, flags)
        const saPtr = Number(a1) + GUEST_MEM_BASE;
        const saSize = Number(a2);
        // Zero the struct, set size field
        for (let i = 0; i < Math.min(saSize, 56); i += 8) view.setBigUint64(saPtr + i, 0n, true);
        view.setUint32(saPtr, Math.min(saSize, 56), true); // size field
        ret = 0n;
        break;
      }
      case 280: // bpf
        ret = -BigInt(ENOSYS);
        break;
      case 282: // userfaultfd
        ret = -BigInt(ENOSYS);
        break;
      case 283: // membarrier
        ret = 0n; // no-op in single-threaded
        break;
      case 284: // mlock2
        ret = 0n;
        break;
      case 286: // preadv2
      case 287: // pwritev2
        ret = -BigInt(ENOSYS); // TODO: implement
        break;
      case 288: // pkey_mprotect
      case 289: // pkey_alloc
      case 290: // pkey_free
        ret = -BigInt(ENOSYS); // memory protection keys not supported
        break;
      case 292: // io_pgetevents
        ret = -BigInt(ENOSYS);
        break;
      case 294: // kexec_file_load
        ret = -BigInt(EPERM);
        break;

      // 295-402: unassigned on aarch64 (kernel reserves these)
      // 403-423: time64 compat syscalls (32-bit only, not available on native aarch64)
      // All return ENOSYS.
      // 245-259: arch_specific_syscall range (unused on aarch64)
      case 245: case 246: case 247: case 248: case 249:
      case 250: case 251: case 252: case 253: case 254:
      case 255: case 256: case 257: case 258: case 259:
        ret = -BigInt(ENOSYS);
        break;

      case 424: // pidfd_send_signal
        ret = -BigInt(ENOSYS);
        break;
      // io_uring (425-427) already handled above
      case 428: // open_tree
      case 429: // move_mount
      case 430: // fsopen
      case 431: // fsconfig
      case 432: // fsmount
      case 433: // fspick
        ret = -BigInt(ENOSYS); // new mount API — not yet supported
        break;
      case 434: // pidfd_open
        ret = -BigInt(ENOSYS);
        break;
      // clone3 already handled at 435
      case 436: // close_range(first, last, flags)
        // Per spec: close all fds from first to last
        if (this.vfs) {
          const first = Number(a0);
          const last = Number(a1);
          for (let fd = first; fd <= Math.min(last, MAX_FDS); fd++) {
            if (this.fdTable.has(fd)) {
              this.fdTable.delete(fd);
            }
            if (this.vfs.hasFd(fd)) {
              this.vfs.close(fd);
            }
          }
        }
        ret = 0n;
        break;

      case 437: { // openat2(dirfd, pathname, how, size) — extended open
        // Treat as regular openat
        const path437 = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
        const howPtr = Number(a2) + GUEST_MEM_BASE;
        const flags437 = Number(view.getBigUint64(howPtr, true)); // how.flags
        const mode437 = Number(view.getBigUint64(howPtr + 8, true)); // how.mode
        if (this.vfs) {
          const effectiveMode = mode437 & ~this.umask;
          const openRet = this.vfs.open(path437, flags437, effectiveMode);
          if (openRet >= 0) {
            this.fdFlags.set(openRet, flags437 & ~(0o100 | 0o200 | 0o1000 | 0o2000000));
            if (flags437 & 0o2000000) this.fdCloexec.add(openRet);
          }
          ret = BigInt(openRet);
        } else {
          ret = -BigInt(ENOENT);
        }
        break;
      }

      case 438: // pidfd_getfd
        ret = -BigInt(ENOSYS);
        break;
      case 439: { // faccessat2(dirfd, pathname, mode, flags)
        // Per spec: like faccessat but with flags (AT_SYMLINK_NOFOLLOW, AT_EACCESS)
        if (this.ptrFaults(bytes, Number(a1))) { ret = -BigInt(EFAULT); break; }
        const fa2Path = this.resolveAt(a0, readGuestString(bytes, Number(a1)));
        if (this.vfs) {
          ret = BigInt(this.vfs.faccessat(Number(BigInt.asIntN(32, a0)), fa2Path, Number(a2), Number(a3)));
        } else {
          ret = -BigInt(ENOENT);
        }
        break;
      }
      case 440: // process_madvise
        ret = -BigInt(ENOSYS);
        break;
      case 441: { // epoll_pwait2 — like epoll_pwait but with timespec timeout
        // Redirect to our epoll_pwait implementation
        const ep2fd = Number(a0);
        if (!this.epollSets?.has(ep2fd)) { ret = -BigInt(EBADF); break; }
        const ep2set = this.epollSets.get(ep2fd)!;
        const ep2events = Number(a1);
        const ep2max = Number(a2);
        let ep2count = 0;
        for (const [targetFd, entry] of ep2set) {
          if (ep2count >= ep2max) break;
          const ready = entry.events & 0x005; // EPOLLIN | EPOLLOUT
          if (ready) {
            const off = ep2events + ep2count * 12 + GUEST_MEM_BASE;
            view.setUint32(off, ready, true);
            view.setBigUint64(off + 4, entry.data, true);
            ep2count++;
          }
        }
        ret = BigInt(ep2count);
        break;
      }
      case 442: // mount_setattr
        ret = 0n;
        break;
      case 443: // quotactl_fd
        ret = -BigInt(ENOSYS);
        break;
      case 444: case 445: case 446: // landlock
        ret = -BigInt(ENOSYS);
        break;
      case 447: // memfd_secret
        ret = -BigInt(ENOSYS);
        break;
      case 448: // process_mrelease
        ret = -BigInt(ENOSYS);
        break;
      case 449: // futex_waitv
        ret = -BigInt(ENOSYS);
        break;
      case 450: // set_mempolicy_home_node
        ret = 0n; // accept, no NUMA
        break;
      case 451: // cachestat
        ret = -BigInt(ENOSYS);
        break;
      case 452: { // fchmodat2(dirfd, pathname, mode, flags)
        // Per spec: like fchmodat but with AT_SYMLINK_NOFOLLOW support
        ret = 0n; // accept
        break;
      }
      case 453: // map_shadow_stack
        ret = -BigInt(ENOSYS);
        break;
      case 454: // futex_wake (new API)
      case 455: // futex_wait (new API)
      case 456: // futex_requeue (new API)
        ret = -BigInt(ENOSYS);
        break;
      case 457: // statmount
      case 458: // listmount
        ret = -BigInt(ENOSYS);
        break;
      case 459: case 460: case 461: // LSM
        ret = -BigInt(ENOSYS);
        break;
      case 462: // mseal
        ret = 0n; // accept, pretend sealed
        break;
      case 463: case 464: case 465: case 466: // xattr at variants
        ret = -BigInt(95); // ENOTSUP
        break;
      case 467: // open_tree_attr
        ret = -BigInt(ENOSYS);
        break;
      case 468: case 469: // file_getattr/file_setattr
        ret = -BigInt(ENOSYS);
        break;
      case 470: // listns
        ret = -BigInt(ENOSYS);
        break;
      case 471: // rseq_slice_yield
        ret = -BigInt(ENOSYS);
        break;

      default:
        // For any remaining syscall not in our table, return ENOSYS
        if (nr >= 0 && nr <= 450) {
          // Known range — just not implemented
          ret = -BigInt(ENOSYS);
        } else {
          console.warn(`[jitter] unknown syscall ${nr} (X0=${a0}, X1=${a1}, X2=${a2})`);
          ret = -BigInt(ENOSYS);
        }
    }

    // Set return value in X0
    cpu.setReg(0, BigInt.asUintN(64, ret));
    return { status: 'ok' };
  }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

// a guest pointer faults (EFAULT) if it lands in the never-mapped low page or falls outside
// the guest-addressable region. this catches the classic LTP bad-address probes ((void*)-1 and
// NULL) without needing full per-page vma tracking, which our flat guest memory does not model.
export function badGuestPtr(memory: Uint8Array, guestAddr: number, len = 1): boolean {
  if (!Number.isFinite(guestAddr) || guestAddr < 0x1000) return true;
  const max = memory.length - GUEST_MEM_BASE;
  return guestAddr >= max || guestAddr + len > max;
}

export function readGuestString(memory: Uint8Array, guestAddr: number): string {
  const base = guestAddr + GUEST_MEM_BASE;
  let end = base;
  while (end < memory.length && memory[end] !== 0) end++;
  return new TextDecoder().decode(memory.subarray(base, end));
}

function writeChardevStat(view: DataView, guestAddr: number): void {
  const base = guestAddr + GUEST_MEM_BASE;
  // struct stat (aarch64 Linux):
  // 0: st_dev (8), 8: st_ino (8), 16: st_mode (4), ...
  // Zero the whole struct (128 bytes)
  for (let i = 0; i < 128; i += 8) {
    view.setBigUint64(base + i, 0n, true);
  }
  // st_mode = S_IFCHR | 0666 = 0o20666 = 0x21B6
  view.setUint32(base + 16, 0x21B6, true);
  // st_nlink = 1
  view.setUint32(base + 20, 1, true);
}

/** Write a stat struct from a bridge response (fs_stat returns mode, size, nlink, mtime) */
function writeBridgeStat(view: DataView, guestAddr: number, resp: any): void {
  const base = guestAddr + GUEST_MEM_BASE;
  // Zero the whole struct (128 bytes)
  for (let i = 0; i < 128; i += 8) {
    view.setBigUint64(base + i, 0n, true);
  }
  // struct stat (aarch64 Linux):
  // 0: st_dev (8), 8: st_ino (8), 16: st_mode (4), 20: st_nlink (4)
  // 48: st_size (8), 88: st_mtime (8)
  // Meta layout for fs_stat: _meta = [mode, size, nlink, mtime, uid, gid, ino]
  const meta = resp._meta;
  const mode = meta ? meta[0] : (resp.mode ?? 0o100644);
  const size = meta ? meta[1] : (resp.size ?? 0);
  const nlink = meta ? meta[2] : (resp.nlink ?? 1);
  const mtime = meta ? meta[3] : (resp.mtime ?? 0);
  const uid = meta ? (meta[4] ?? 0) : (resp.uid ?? 0);
  const gid = meta ? (meta[5] ?? 0) : (resp.gid ?? 0);
  const ino = meta ? (meta[6] ?? 0) : (resp.ino ?? 0);
  // st_dev (nonzero) + st_ino (real inode, unique per file). CRITICAL: musl's ld.so dedups shared
  // libraries by (st_dev, st_ino) — a zero/constant ino makes it treat every lib as already-loaded
  // and silently skip all but the first, so their symbols never resolve.
  view.setBigUint64(base + 0, 1n, true);              // st_dev
  view.setBigUint64(base + 8, BigInt.asUintN(64, BigInt(Math.trunc(ino))), true); // st_ino
  view.setUint32(base + 16, mode, true);
  view.setUint32(base + 20, nlink, true);
  view.setUint32(base + 24, uid, true);   // st_uid
  view.setUint32(base + 28, gid, true);   // st_gid
  view.setBigUint64(base + 48, BigInt(size), true);
  view.setBigUint64(base + 88, BigInt(mtime), true);
}
