import { CpuState } from './cpu.js';
import { GUEST_MEM_BASE, TPIDR_OFFSET, PC_OFFSET, V_OFFSET } from './types.js';
import type { NativeBridge } from './bridge.js';
import { BRIDGE_DATA_MAX } from './kernel-ipc.js';
import { VMA_TABLE_BYTES, VMA_TABLE_ENTRIES, VMA_TABLE_FIRST, SIG_TABLE_BYTES, SIG_TABLE_FIRST } from './types.js';

/** one mapped range. `maxProt` is the most it may ever be granted -- see the vmas field.
 *  `grow` marks a MAP_GROWSDOWN mapping: a stack, which the kernel extends downward on a fault
 *  just below it rather than killing the process. */
interface Vma { start: number; end: number; prot: number; shared: boolean; maxProt: number; grow?: boolean }

// ── Syscall result type ─────────────────────────────────────────────────────

export type SyscallResult =
  | { status: 'ok' }
  | { status: 'exit'; code: number; group?: boolean }
  | { status: 'blocked_read'; fd: number }
  | { status: 'fork'; resumePC: bigint; flags: number; childStack: bigint; ptid: bigint; tls: bigint; ctid: bigint }
  | { status: 'execve'; path: string; argv: string[]; envp: string[] }
  | { status: 'bridge'; method: string; args: Record<string, any>; apply: (resp: any) => void };

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
const ELOOP = 40;
const ENOENT = 2;
const EPERM  = 1;
const EOPNOTSUPP = 95;
// the read/write flags preadv2 and pwritev2 accept: HIPRI, DSYNC, SYNC, NOWAIT, APPEND, NOAPPEND.
// anything outside this set is a request for behaviour the kernel does not have, and is refused.
const RWF_SUPPORTED = 0x3f;
const AT_SYMLINK_NOFOLLOW = 0x100;
// the two sentinels utimensat takes in place of a nanosecond count: "use the current time" and
// "leave this timestamp alone".
const UTIME_NOW = (1 << 30) - 1;
const UTIME_OMIT = (1 << 30) - 2;
const EINTR = 4;
const E2BIG = 7;
const EXDEV = 18;
const EFBIG = 27;
const ENOTSOCK = 88;

/**
 * does this (level, optname) pair name an option linux actually has?
 *
 * getsockopt and setsockopt answer ENOPROTOOPT for anything else, and that answer is the only way
 * a program can discover what a kernel supports. saying yes to everything -- which is what this
 * layer used to do -- makes feature detection report the opposite of the truth.
 *
 * the ranges are contiguous in linux's headers, so they are written as ranges rather than as
 * hundreds of names; each one ends at the last option the current uapi defines.
 */
/**
 * can a socket of this (domain, type, protocol) exist? returns 0 if so, else the errno linux
 * gives -- checked in linux's own order, because the order is observable: a bad type in a good
 * family is EINVAL, and a good type with a protocol it cannot carry is EPROTONOSUPPORT.
 */
function sockTripleErrno(domain: number, typeArg: number, protocol: number, euid: number): number {
  const SOCK_STREAM = 1, SOCK_DGRAM = 2, SOCK_RAW = 3, SOCK_SEQPACKET = 5;
  const IPPROTO_ICMP = 1, IPPROTO_TCP = 6, IPPROTO_UDP = 17;
  const type = typeArg & 0xFF;
  if (domain !== 1 && domain !== 2 && domain !== 10) return EAFNOSUPPORT;
  if (type !== SOCK_STREAM && type !== SOCK_DGRAM && type !== SOCK_RAW && type !== SOCK_SEQPACKET) return EINVAL;
  if (typeArg & ~(0xFF | 0o2000000 | 0o4000)) return EINVAL;   // only SOCK_CLOEXEC | SOCK_NONBLOCK
  if (domain === 1) return protocol === 0 ? 0 : EPROTONOSUPPORT;
  const ok =
    protocol === 0 ? type !== SOCK_RAW
    : type === SOCK_STREAM ? protocol === IPPROTO_TCP
    : type === SOCK_DGRAM ? protocol === IPPROTO_UDP
    : type === SOCK_RAW ? protocol === IPPROTO_ICMP || protocol === IPPROTO_TCP || protocol === IPPROTO_UDP
    : false;
  if (!ok) return EPROTONOSUPPORT;
  if (type === SOCK_RAW && euid !== 0) return EPERM;
  return 0;
}

function sockOptLevel(level: number): boolean {
  return level === 0 || level === 1 || level === 6 || level === 41 || level === 58 || level === 132;
}

/** an ipv6 address as its eight 16-bit groups, from any of the shorthands text can use. */
function ipv6Groups(host: string): number[] {
  const out = new Array(8).fill(0);
  const [head, tail] = host.split('::');
  const h = head ? head.split(':').filter((x) => x.length) : [];
  const t = tail !== undefined ? tail.split(':').filter((x) => x.length) : [];
  for (let i = 0; i < h.length && i < 8; i++) out[i] = parseInt(h[i], 16) & 0xffff;
  for (let i = 0; i < t.length && i < 8; i++) out[8 - t.length + i] = parseInt(t[i], 16) & 0xffff;
  return out;
}

/** the canonical text of an ipv6 address: the longest run of zero groups collapsed to "::". */
function ipv6Text(groups: number[]): string {
  let bestStart = -1, bestLen = 0;
  for (let i = 0; i < 8; i++) {
    if (groups[i] !== 0) continue;
    let j = i;
    while (j < 8 && groups[j] === 0) j++;
    if (j - i > bestLen) { bestLen = j - i; bestStart = i; }
    i = j - 1;
  }
  if (bestLen > 1) {
    const head = groups.slice(0, bestStart).map((g) => g.toString(16)).join(':');
    const tail = groups.slice(bestStart + bestLen).map((g) => g.toString(16)).join(':');
    return `${head}::${tail}`;
  }
  return groups.map((g) => g.toString(16)).join(':');
}

function sockOptKnown(level: number, name: number): boolean {
  switch (level) {
    case 1:  return name >= 1 && name <= 75;   // SOL_SOCKET: SO_DEBUG .. SO_RCVMARK
    case 0:  return (name >= 1 && name <= 25) || (name >= 32 && name <= 50); // IPPROTO_IP
    case 6:  return name >= 1 && name <= 38;   // IPPROTO_TCP: TCP_NODELAY .. TCP_TX_DELAY
    case 41: return name >= 1 && name <= 80;   // IPPROTO_IPV6
    case 58: return name >= 1 && name <= 24;   // IPPROTO_ICMPV6
    case 132: return name >= 0 && name <= 33;  // IPPROTO_SCTP
    default: return false;
  }
}
const EPROTONOSUPPORT = 93;
const EAFNOSUPPORT = 97;
const ENOPROTOOPT = 92;
const EADDRINUSE = 98;
const EADDRNOTAVAIL = 99;
const ENOTCONN = 107;
const EISCONN = 106;
// CAP_DAC_READ_SEARCH: the capability that lets a process reach a file it could not name its way to.
const CAP_DAC_READ_SEARCH = 2;
// the one filesystem under everything but /proc and /dev, as /proc/mounts and file handles name it.
const NATIVE_FS_MOUNT_ID = 1;
const EMSGSIZE = 90;
const EPIPE = 32;
const ENODEV = 19;
const EMFILE = 24;
const ENOTDIR = 20;
const ECHILD = 10;
const ESPIPE = 29;
const EISDIR = 21;
const EBUSY = 16;
// the largest file this filesystem will hold, which is what fallocate and truncate measure
// their arguments against. ext4's s_maxbytes on a 4 KiB block size.
const MAX_FILE_BYTES = 0x1000000000000n;
const EAGAIN = 11;
const EFAULT = 14;
const EACCES = 13;
const ENOMEM = 12;
const EEXIST = 17;
const ENAMETOOLONG = 36;
// every madvise advice linux recognises. an advice it does not is EINVAL, and accepting all of
// them made madvise a call that could never fail.
const MADV_VALID = new Set([0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 100, 101]);
const IOV_MAX = 1024;
// the resource limits this system enforces; everything else is unlimited.
// fs.nr_open: the system-wide ceiling on a descriptor number, and on RLIMIT_NOFILE's hard limit.
const NR_OPEN = 1048576;
const RLIMIT_STACK = 3, RLIMIT_FSIZE = 1, RLIMIT_NOFILE = 7, RLIMIT_MEMLOCK = 8,
      RLIMIT_SIGPENDING = 11, RLIMIT_NICE = 13, RLIMIT_RTPRIO = 14;
// linux derives the default from the thread ceiling (max_threads/2). ours is a fixed number of the
// same order, and it is a real limit rather than infinity because a program is entitled to lower
// it and then observe the refusal that follows.
const SIGPENDING_DEFAULT = 15000;
const RLIM_INFINITY = 0xFFFFFFFFFFFFFFFFn;
const RLIM_NLIMITS = 16;
// reboot(2)'s second magic number: linux accepts any of four, each a kernel developer's birthday.
const REBOOT_MAGIC2 = new Set([0x28121969, 0x05121996, 0x16041998, 0x20112000]);
// inotify event masks (linux/inotify.h)
const IN_ACCESS = 0x001, IN_MODIFY = 0x002, IN_ATTRIB = 0x004;
const IN_CLOSE_WRITE = 0x008, IN_CLOSE_NOWRITE = 0x010, IN_OPEN = 0x020;
const IN_MOVED_FROM = 0x040, IN_MOVED_TO = 0x080, IN_CREATE = 0x100;
const IN_DELETE = 0x200, IN_DELETE_SELF = 0x400, IN_MOVE_SELF = 0x800;
const IN_ISDIR = 0x40000000;
const PRIO_PROCESS = 0;
const PRIO_PGRP = 1;
const PRIO_USER = 2;
const ESRCH  = 3;
const MAX_FDS = 1024;

// struct epoll_event is `{ uint32_t events; uint64_t data; }` and is __attribute__((packed)) ONLY
// on x86_64 -- linux defines __EPOLL_PACKED for that arch alone. everywhere else, aarch64 included,
// it takes its natural layout: 16 bytes with the union at offset 8, not 12 bytes with it at 4.
const EPOLL_EVENT_SIZE = 16;
/** how many epoll instances may be chained, one watching the next. linux's EP_MAX_NESTS plus one. */
const EPOLL_MAX_DEPTH = 5;
const EPOLL_DATA_OFFSET = 8;

// per-process address-space ceiling (RLIMIT_AS-ish). a mmap/brk growth past this returns ENOMEM
// to the guest (malloc -> NULL) instead of growing the wasm memory toward the device jetsam limit
// and taking the whole WKWebView content process down. one process can't alone blow the budget;
// it fails gracefully like real linux under a cgroup/ulimit. (the global cross-process budget is
// enforced separately in the kernel.) generous enough that real programs never see it.
// how much guest address space one process may have. 1.5 GiB was chosen when we believed the
// platform ceiling was about 1.25 GiB; it is not -- a single wasm memory reaches 4 GiB, written,
// and what actually bounds us is total committed bytes across the app (~3.7 GiB measured). the
// limit now is the permission table, which must fit between PROT_TABLE_BASE and GUEST_MEM_BASE at
// one byte per 4 KiB page. 1856 MiB needs 464 KiB of table and leaves room to spare.
const MAX_PROCESS_BYTES = 1856 * 1024 * 1024;

const O_NONBLOCK = 0o4000;

// Signal constants
const SIG_DFL = 0n;
const SIG_IGN = 1n;
const SIGALRM = 14;

/**
 * where a pending signal came from, and what it carries. si_code is the sender's claim about
 * itself -- SI_USER for a kill, SI_QUEUE for a sigqueue, SI_TIMER for a timer -- and a handler
 * reads it before trusting anything else in the struct.
 */
export interface SignalOrigin {
  pid: number;
  uid: number;
  code: number;
  value: bigint;
}
const SIGVTALRM = 26;
const SIGPROF = 27;
const ITIMER_SIGNAL = [SIGALRM, SIGVTALRM, SIGPROF];

// clock ids, as the guest spells them.
const CLOCK_REALTIME = 0;
const CLOCK_MONOTONIC = 1;
// CLOCK_PROCESS_CPUTIME_ID: the clock that only runs while this process does. it backs
// ITIMER_VIRTUAL and ITIMER_PROF, which is why they cannot fire while the process is parked.
const CLOCK_CPU_OWN = 2;
const CLOCK_THREAD_CPUTIME_ID = 3;
const CLOCK_BOOTTIME = 7;
const CLOCK_REALTIME_ALARM = 8;
const CLOCK_BOOTTIME_ALARM = 9;

const SIGEV_SIGNAL = 0;
const SIGEV_NONE = 1;
const SIGEV_THREAD = 2;
const SIGEV_THREAD_ID = 4;
const SIGRTMAX = 64;

// the open flags openat2 will accept. it is strict where openat is forgiving: openat quietly
// drops bits it does not know, openat2 refuses them, and that refusal is how a program discovers
// what the kernel it is talking to supports.
// aarch64 does NOT use the asm-generic numbering for four of these: O_DIRECTORY is 0o40000,
// O_NOFOLLOW 0o100000, O_DIRECT 0o200000 and O_LARGEFILE 0o400000. reading the generic values
// here would have made O_TMPFILE mean __O_TMPFILE|O_DIRECT, which is not a thing.
const O_TMPFILE_BIT = 0o20000000n;                 // __O_TMPFILE, without the O_DIRECTORY half
const O_DIRECTORY_B = 0o40000n;
const O_NOFOLLOW_B = 0o100000n;
const O_PATH_B = 0o10000000n;
const O_CLOEXEC_B = 0o2000000n;
const VALID_OPEN_FLAGS =
  3n                    // O_ACCMODE
  | 0o100n              // O_CREAT
  | 0o200n              // O_EXCL
  | 0o400n              // O_NOCTTY
  | 0o1000n             // O_TRUNC
  | 0o2000n             // O_APPEND
  | 0o4000n             // O_NONBLOCK
  | 0o10000n            // O_DSYNC
  | 0o20000n            // FASYNC
  | O_DIRECTORY_B
  | O_NOFOLLOW_B
  | 0o200000n           // O_DIRECT
  | 0o400000n           // O_LARGEFILE
  | 0o1000000n          // O_NOATIME
  | O_CLOEXEC_B
  | 0o4000000n          // __O_SYNC
  | O_PATH_B
  | O_TMPFILE_BIT;
// O_PATH opens a reference to a file rather than the file, so almost nothing else may be asked for.
const O_PATH_FLAGS = O_DIRECTORY_B | O_NOFOLLOW_B | O_PATH_B | O_CLOEXEC_B;

const RESOLVE_NO_XDEV = 0x01n;
const RESOLVE_NO_MAGICLINKS = 0x02n;
const RESOLVE_NO_SYMLINKS = 0x04n;
const RESOLVE_BENEATH = 0x08n;
const RESOLVE_IN_ROOT = 0x10n;
const RESOLVE_CACHED = 0x20n;
const VALID_RESOLVE = RESOLVE_NO_XDEV | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_SYMLINKS |
                      RESOLVE_BENEATH | RESOLVE_IN_ROOT | RESOLVE_CACHED;

const TFD_CLOEXEC = 0o2000000;
const TFD_NONBLOCK = 0o4000;
const TIMER_ABSTIME = 1;
// timer_getoverrun returns an int, and a timer with a nanosecond period left pending for a few
// seconds passes two billion long before anyone asks. linux saturates rather than wrapping, which
// is the whole of CVE-2018-12896.
function clampOverrun(n: number): number { return n > 0x7fffffff ? 0x7fffffff : n; }
// the running total is held one above that ceiling, so that subtracting the expiry the signal
// itself announced still leaves INT_MAX rather than one short of it.
const OVERRUN_CEIL = 0x80000000;
const TFD_TIMER_ABSTIME = 1;
const TFD_TIMER_CANCEL_ON_SET = 2;

/** a deadline on some clock, plus the period it repeats with (0 = one-shot). */
interface ArmedTimer {
  clockid: number;
  nextNs: bigint;      // absolute, on clockid. 0 means disarmed.
  intervalNs: bigint;
}
interface PosixTimer extends ArmedTimer {
  id: number;
  notify: number;      // SIGEV_*
  signo: number;
  sigval: bigint;
  overrun: number;     // expirations the most recent DELIVERY stood for, minus the one it announced
  accrued: number;     // expirations since that delivery, waiting to be folded into the next one
}
interface TimerFdTimer extends ArmedTimer {
  ticks: bigint;       // expirations not yet read off the descriptor
}
const SIGPIPE = 13;
const SIGCHLD = 17;

// Signal trampoline address (fixed location in guest memory for rt_sigreturn stub)
const SIGNAL_TRAMPOLINE = 0x1000; // in the first page, before GUEST_MEM_BASE

// Signals with default action = ignore
const SIG_IGNORE_DEFAULT = new Set([SIGCHLD, 23, 28]); // SIGCHLD(17), SIGURG(23), SIGWINCH(28)
// Signals with default action = stop (not kill)
const SIG_STOP_DEFAULT = new Set([19, 20, 21, 22]); // SIGSTOP, SIGTSTP, SIGTTIN, SIGTTOU
// signals whose default action is "terminate AND dump core". we write no core file, but the fact
// that the signal is a core-dumping one is reported in the wait status (WCOREDUMP), and a parent
// uses it to tell a crash from an orderly death by signal.
const SIG_CORE_DEFAULT = new Set([
  3,  // SIGQUIT
  4,  // SIGILL
  5,  // SIGTRAP
  6,  // SIGABRT
  7,  // SIGBUS
  8,  // SIGFPE
  11, // SIGSEGV
  24, // SIGXCPU
  25, // SIGXFSZ
  31, // SIGSYS
]);

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
// arm64 rt_sigframe layout (what the kernel pushes on signal delivery). the handler receives
// x1 = &siginfo, x2 = &ucontext; on return the rt_sigreturn trampoline restores register state from
// ucontext.uc_mcontext, so a handler that edits uc_mcontext.pc/regs (fault recovery) takes effect.
const SI_CODE = 8;                    // siginfo_t.si_code
const SI_ADDR = 16;                   // siginfo_t._sifields._sigfault.si_addr (SIGSEGV/SIGBUS/SIGILL/SIGFPE)
const SI_PID = 16;                    // siginfo_t._sifields._kill.si_pid  (the union starts at 16)
const SI_UID = 20;                    // siginfo_t._sifields._kill.si_uid
const SI_VALUE = 24;                  // siginfo_t._sifields._rt.si_sigval
const UC_OFF = 128;                   // ucontext follows the 128-byte siginfo
const UC_SIGMASK = UC_OFF + 40;       // ucontext.uc_sigmask (after uc_flags,uc_link,uc_stack)
const UC_MCONTEXT = UC_OFF + 176;     // ucontext.uc_mcontext (struct sigcontext) — fixed arm64 offset
const MC_REGS = UC_MCONTEXT + 8;      // sigcontext.regs[31] (after fault_address)
const MC_SP = UC_MCONTEXT + 256;      // sigcontext.sp
const MC_PC = UC_MCONTEXT + 264;      // sigcontext.pc
const MC_PSTATE = UC_MCONTEXT + 272;  // sigcontext.pstate
const MC_VREGS = UC_MCONTEXT + 296;   // fpsimd_context.vregs[32] (fault_addr+regs+sp+pc+pstate+8B fpsimd hdr)
const SIGNAL_FRAME_SIZE = 1216;       // covers vregs end (296+512) with slack, 16-byte aligned
// signals whose siginfo carries a faulting address in _sigfault.si_addr.
const SIG_WITH_ADDR = new Set([4, 7, 8, 11]); // SIGILL, SIGBUS, SIGFPE, SIGSEGV

// ── ioctl constants ─────────────────────────────────────────────────────────

const ENOTTY = 25;
const SIOCATMARK    = 0x8905;
const SIOCGIFNAME   = 0x8910;
const SIOCGIFCONF   = 0x8912;
const SIOCGIFFLAGS  = 0x8913;
const SIOCSIFFLAGS  = 0x8914;
const SIOCGIFADDR   = 0x8915;
const SIOCGIFDSTADDR = 0x8917;
const SIOCGIFBRDADDR = 0x8919;
const SIOCGIFNETMASK = 0x891b;
const SIOCGIFMTU    = 0x8921;
const SIOCGIFHWADDR = 0x8927;
const SIOCGIFINDEX  = 0x8933;
const TCGETS     = 0x5401;
const TCSETS     = 0x5402;
const TIOCGWINSZ = 0x5413;
const TIOCSWINSZ = 0x5414;

// ── fcntl constants ─────────────────────────────────────────────────────────

const F_DUPFD = 0;
// struct f_owner_ex types
// capability numbers we actually consult (linux capability.h)
const CAP_SYS_PTRACE = 19;
const CAP_SYS_CHROOT = 18;
const CAP_SYS_NICE = 23;
const CAP_IPC_LOCK = 14;
const CAP_SETPCAP = 8;
const CAP_SETGID = 6;
const CAP_SYS_TIME = 25;
const CAP_SYS_BOOT = 22;
const CAP_SYS_ADMIN = 21;
const CAP_SYS_RESOURCE = 24;
const CAP_WAKE_ALARM = 35;
const ALL_CAPS = (1n << 41n) - 1n; // CAP_LAST_CAP is 40 on current kernels

// a page with no backing store behind a legal mapping: touching it is SIGBUS, not SIGSEGV.
const PAGE_NO_BACKING = 8;

const F_OWNER_TID = 0;
const F_OWNER_PID = 1;
const F_OWNER_PGRP = 2;
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
  fdIsDir(fd: number): boolean;
  isCharDev(fd: number): boolean;
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
  mkfifo(path: string, mode: number): number;
  isOPath(fd: number): boolean;
  fdPath(fd: number): string | null;
  openFdNumbers(): number[];
  fdAccessMode(fd: number): number;
  // plain data, because it crosses a postMessage: see InMemoryVFS.snapshotFds.
  fdOfdType(fd: number): 'file' | 'dir' | 'chardev' | 'fifo' | null;
  fdDevice(fd: number): { major: number; minor: number } | null;
  snapshotFds(): { ofds: any[]; fds: Array<[number, number]> };
  restoreFds(snap: { ofds: any[]; fds: Array<[number, number]> } | undefined): void;
  closeAllExcept(...keep: number[]): void;
  loadBinary(data: Uint8Array): void;
  fdAllocator: (() => number) | null;
  credsProvider: (() => { ruid: number; euid: number; rgid: number; egid: number }) | null;
  selfCpuProvider: (() => { utimeMs: number; stimeMs: number }) | null;
}

/** a descriptor that names a socket, whichever domain it lives in. */
type SockEntry = Extract<FdEntry, { type: 'socket' } | { type: 'unixsock' }>;

// ── I/O interface for terminal ──────────────────────────────────────────────

export interface IOInterface {
  writeStdout(data: Uint8Array): void;
  writeStderr(data: Uint8Array): void;
  readStdin(count: number): Uint8Array | null;
  /** Whether stdin has bytes available for non-blocking read. */
  stdinHasData?(): boolean;
  /** Block until stdin has data or timeout fires (ms; negative = infinite). Returns whether data arrived. */
  waitStdin?(timeoutMs: number): 'data' | 'timeout';
  /** the io readiness generation, and a park on it. this is what lets poll/select/epoll block on
   * ANY descriptor rather than only on stdin: read the generation, decide nothing is ready, then
   * park until it moves. only a Worker can park, so both are optional. */
  ioGeneration?(): number;
  waitIo?(seen: number, timeoutMs: number): boolean;
  /** readiness of an AF_UNIX socket. it is the worker that owns the rings and the kernel that owns
   * the accept queue, so the answer is not visible from here and has to be asked for. */
  unixReady?(fd: number): { readable: boolean; writable: boolean };
  /**
   * Readiness of a KERNEL pipe end, or null if this descriptor is not one.
   *
   * Real pipes live in the shared arena and are tracked by the worker, not by this handler's fd
   * table -- so every readiness question about one used to fall through to "not in the table,
   * therefore not ready". poll, select and epoll all reported a pipe with data waiting as having
   * nothing, forever.
   */
  pipeReady?(fd: number): { readable: boolean; writable: boolean } | null;
  /** how many inotify events are queued for this descriptor. the queue lives in the kernel. */
  inotifyPending?(fd: number): number;
  /** a message queue's readiness, which only the kernel that holds the queue can answer. */
  mqReady?(fd: number): { readable: boolean; writable: boolean } | null;
  /** has this process exited? a pidfd becomes readable at that moment, and only the shared process
   * table knows. */
  pidExited?(pid: number): boolean;
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
}

type FdEntry =
  | { type: 'stdio'; fd: number }  // 0=stdin, 1=stdout, 2=stderr
  | { type: 'tty' }                // /dev/tty — controlling terminal
  // /dev/jitter — the control device the `jitter` command talks to. the guest writes a request line,
  // then reads [int32 status][body]; the request is dispatched host-side on the first read.
  | { type: 'jitterctl'; req: string; resp: Uint8Array | null; pos: number }
  | { type: 'vfs'; fd: number }    // VFS file descriptor
  | { type: 'pipe_read'; pipe: PipeBuffer }
  | { type: 'pipe_write'; pipe: PipeBuffer }
  // isDir and unlinked are what the descriptor knows about its object that its path no longer does:
  // whether it is a directory (IN_ISDIR) and whether the name has since been removed (IN_EXCL_UNLINK).
  | { type: 'native'; nativeFd: number; path: string; ofdId: number; isDir?: boolean; ftype?: number; unlinked?: boolean; sealable?: boolean }
  | { type: 'socket'; socketId: number; domain: number; sockType: number; protocol?: number; listening?: boolean; shutRd?: boolean; shutWr?: boolean; opts?: Map<number, number>; mcast?: Set<string>; remoteAddr?: { family: number; host: string; port: number }; localAddr?: { family: number; host: string; port: number } }
  // AF_UNIX: the stream lives in arena rings the worker drives directly, so there is no swift-side
  // socket behind it. the entry exists so close, dup and poll see a real descriptor.
  | { type: 'unixsock'; sockType?: number; listening?: boolean; opts?: Map<number, number>; mcast?: Set<string> }
  | { type: 'signalfd'; mask: bigint }
  | { type: 'inotify'; id: number }                     // event queue; `id` names it in the kernel registry
  | { type: 'timerfd'; id: number }                     // timerfd_create: an armed timer you read expirations from
  | { type: 'mqueue'; id: number; name: string }        // mq_open: a named posix message queue
  | { type: 'eventfd'; id: number; semaphore: boolean }  // the counter lives in the kernel: fork shares it
  | { type: 'epoll' }                                    // multiplexer (state in this.epollSets)
  // pidfd_open: a stable reference to a process. it outlives the pid (which may be reused), becomes
  // readable when that process exits, and is what pidfd_send_signal and waitid(P_PIDFD) name.
  | { type: 'pidfd'; pid: number }
  // a char/block device node opened by path. what it DOES is decided by its major:minor, exactly
  // as a driver would; the file on disk behind it is only a placeholder for the node's existence.
  | { type: 'devnode'; major: number; minor: number };

// the most a single bridge read may ask for. the response slot is shared memory of a fixed size,
// and a reply that does not fit is dropped -- which the guest would see as a successful read into
// an untouched buffer. short reads are legal; silently empty ones are not.
const BRIDGE_READ_MAX = BRIDGE_DATA_MAX;

/**
 * The devices we can actually be.
 *
 * A device node is its major:minor -- that pair IS the file, and it is the only thing that says
 * what opening it should do. These are linux's own assignments, so a node made by a distro's
 * MAKEDEV, by apk, or by hand with the numbers from devices.txt lands on the right behaviour.
 * Anything else is a node with no driver behind it, which linux answers with ENXIO.
 */
type DevKind = 'null' | 'zero' | 'full' | 'random' | 'tty';
const DEVICES = new Map<string, DevKind>([
  ['1:3', 'null'], ['1:5', 'zero'], ['1:7', 'full'],
  ['1:8', 'random'], ['1:9', 'random'],
  ['5:0', 'tty'], ['5:1', 'tty'],
]);
function devKind(major: number, minor: number): DevKind | undefined {
  return DEVICES.get(`${major}:${minor}`);
}

const NET_TRACE = false; // socket trace for https diagnosis; flip to true when needed, remove before ship.
const FD_TRACE = false; // fd-lifecycle trace; flip to true when needed, remove before ship.

export class SyscallHandler {
  private nt(s: string) { if (NET_TRACE) this.io.writeStderr(new TextEncoder().encode('[net] ' + s + '\n')); }
  private ft(s: string) { if (FD_TRACE) this.io.writeStderr(new TextEncoder().encode('[fd] ' + s + '\n')); }
  private brk: bigint;
  // where the heap started. brk(2) may give memory back, but never below the address the program
  // was loaded with -- below that lies the image itself.
  private brkFloor: bigint;
  /** which membarrier registrations this process has made: bit 0 private expedited, bit 1 sync core. */
  private membarrierRegistered = 0;
  /** where set_robust_list left this thread's list of futexes to release if it dies holding them. */
  private robustListHead = 0n;
  /**
   * file seals, by open-file id. a seal is a promise about a file that cannot be taken back --
   * that it will never grow, never shrink, or never be written again -- and it is what makes a
   * memfd safe to hand to something you do not trust.
   */
  private fileSeals = new Map<number, number>();
  private sealsOf(e: FdEntry | undefined): number {
    return e && e.type === 'native' ? (this.fileSeals.get(e.ofdId) ?? 0) : 0;
  }
  // the exclusive-monitor shadow lives at ZIG_SHADOW_BASE (32MB) and is tiny
  // (1 byte per 512 guest bytes, so <=256KB for a 128MB guest). reserve [32MB,33MB)
  // for it and start the mmap arena above, so mmap'd shared libraries (ld.so mmaps
  // libc.so.6) don't collide with the shadow. mmap grows up toward the shadow_base
  // ceiling of the top-of-memory shadow... here shadow is low, so cap mmap well above.
  private mmapBase = 0x2100000n; // 33MB — just above the 1MB shadow reservation
  /** the bottom of the mmap arena. everything above it was handed out by mmap and is described
   *  by a vma, so the gaps between those vmas are genuinely free and may be handed out again. */
  private readonly mmapArenaStart = 0x2100000;
  /** where to begin looking for free address space. see allocMmapRange. */
  private mmapHint = 0x2100000;

  /**
   * grow the guest's wasm memory to cover `needed` bytes. false if the engine would not.
   *
   * a refused grow throws a RangeError, and RangeError is what a bad guest pointer throws too, so
   * it was being reported as EFAULT -- an errno mmap can never return, and one that tells a program
   * its arguments were wrong when what actually happened is that the machine ran out of memory.
   */
  /** report WHY a grow was refused. "out of memory" is what the guest sees either way, but the
   *  numbers say whether the platform said no or we did. */
  growFailed: ((cur: number, min: number, want: number, cap: number, err: string) => void) | null = null;

  private growTo(memory: WebAssembly.Memory, needed: number, chunkPages = 1): boolean {
    if (needed <= memory.buffer.byteLength) return true;
    const currentPages = memory.buffer.byteLength / 65536;
    const minPages = Math.ceil(needed / 65536);
    const capPages = Math.floor(this.maxBytes / 65536);
    // grow GEOMETRICALLY, not in fixed steps. a fixed 1 MiB chunk means a mapping-heavy program
    // grows every few dozen mmaps forever, and each grow costs in proportion to how big the memory
    // already is -- so the per-mmap cost climbed with the process's size (measured: 390 us per
    // mapping at five thousand, 989 us at thirty-five). asking for half again as much each time
    // makes the number of grows logarithmic and their total cost amortized constant.
    const chunked = Math.ceil(minPages / chunkPages) * chunkPages;
    const geometric = chunkPages > 1 ? Math.ceil(currentPages * 1.5) : 0;
    const want = Math.min(Math.max(chunked, geometric, minPages), capPages);
    if (want <= currentPages) return needed <= memory.buffer.byteLength;
    try {
      memory.grow(want - currentPages);
    } catch (e: any) {
      // the rounded-up request may be what the engine refused; ask for exactly what is needed.
      if (want !== minPages) {
        try { memory.grow(minPages - currentPages); } catch (e2: any) { this.growFailed?.(currentPages, minPages, want, capPages, String(e2?.name)); return false; }
      } else {
        this.growFailed?.(currentPages, minPages, want, capPages, String(e?.name));
        return false;
      }
    }
    return needed <= memory.buffer.byteLength;
  }
  /** vm.max_map_count: how many distinct mappings one process may hold. linux's default. */
  private maxMapCount = 65530;
  getMaxMapCount(): number { return this.maxMapCount; }
  setMaxMapCount(n: number): number {
    if (!Number.isFinite(n) || n < 1) return -EINVAL;
    // a thread group's mappings live in a fixed-size shared table, so the limit can never be set
    // past what that table holds -- silently truncating it on the next publish would destroy
    // mappings the guest is still using.
    this.maxMapCount = Math.min(n, VMA_TABLE_ENTRIES);
    return 0;
  }
  /** how many mappings the process holds, for /proc and for the limit check. */
  vmaCount(): number { return this.vmas.length; }

  /**
   * What this process actually COSTS, as opposed to how far its memory has been grown.
   *
   * wasm growth is lazy: measured on device, 158 memories grown to 32 MiB coexist happily, exactly
   * like 156 that merely DECLARED 32 MiB -- pages become real when written, not when grown. so
   * charging a process for buffer.byteLength (which is what the kernel's budget used to do) is not
   * conservative, it is wrong: it refused forks that would have fit comfortably. what a process
   * genuinely holds is its image and stack, its heap up to the break, and its mappings.
   */
  footprintBytes(stackTop: number): number {
    let n = Math.max(Number(this.brk), stackTop) + GUEST_MEM_BASE;
    for (const v of this.vmas) n += v.end - v.start;
    return n;
  }

  /**
   * find `len` bytes of free guest address space for a mapping.
   *
   * a bump allocator over the arena never reuses anything, so a program that maps and unmaps in a
   * loop -- which is what a general-purpose allocator does with any large request -- walks its
   * address space upward forever, and since wasm memory can only grow, every lap COMMITS more real
   * memory. mapping and unmapping one megabyte two hundred times cost two hundred megabytes and
   * 12 ms a lap. so: first fit through the holes, and only extend the arena when none is big enough.
   */
  private allocMmapRange(len: number): bigint {
    // start where the last one finished. first-fit from the very bottom every time is O(n) per
    // mmap and so O(n^2) over a program's lifetime: LTP's munmap04 makes 65530 mappings and went
    // from failing for lack of memory to failing for lack of TIME. the hint makes the common case
    // -- a program that keeps allocating -- constant, and freeing pulls it back down so a hole is
    // still found. it is only a hint: if the scan from it comes up empty we start over from the
    // bottom before growing the arena, so nothing is ever lost, only looked at later.
    const found = this.scanForGap(Math.max(this.mmapHint, this.mmapArenaStart), len);
    if (found >= 0) { this.mmapHint = found + len; return BigInt(found); }
    if (this.mmapHint > this.mmapArenaStart) {
      const retry = this.scanForGap(this.mmapArenaStart, len);
      if (retry >= 0) { this.mmapHint = retry + len; return BigInt(retry); }
    }
    const addr = this.mmapBase;
    this.mmapBase += BigInt(len);
    this.mmapHint = Number(this.mmapBase);
    this.vmaDirty = true; // the arena cursor moved, and that is the group's state too
    return addr;
  }

  /** lowest free address >= `from` with `len` bytes behind it, or -1 if the arena has no such gap. */
  private scanForGap(from: number, len: number): number {
    const top = Number(this.mmapBase);
    let cur = from;
    for (let k = this.vmaLower(cur); k < this.vmas.length; k++) {
      const v = this.vmas[k];
      if (v.start >= top) break;
      if (v.start - cur >= len) return cur;
      if (v.end > cur) cur = v.end;
    }
    return top - cur >= len ? cur : -1;
  }
  // brk/mmap ceiling. a threaded process uses a lower ceiling (its shared memory places the per-thread
  // register blocks + atomics shadow just above this, and the shared reservation is bounded so several
  // threaded processes can coexist). single-threaded processes keep the full 1.5 GiB.
  maxBytes = MAX_PROCESS_BYTES;
  // MAP_SHARED regions (guest offsets). shared across fork: a child's writes to these
  // must survive the parent-memory restore in the serialized fork model.
  private sharedRegions: Array<{ addr: number; len: number }> = [];
  // installed by the worker: a new MAP_SHARED region needs a block in the shared arena, and an
  // unmapped one has to give it back. see shared-map.ts for why the arena exists at all.
  sharedMapHook: ((addr: number, len: number, key?: string) => void) | null = null;
  /** capacity of the ring behind a pipe fd, or null if the fd is not a pipe. pipes are not all the
   *  same size -- the kernel shrinks new ones when many are open -- and F_GETPIPE_SZ is a promise
   *  a program sizes its writes by, so it has to come from the actual ring. */
  pipeCapacityHook: ((fd: number) => number | null) | null = null;
  /** called when a descriptor's async-notification state changes (owner, signal, or the O_ASYNC
   *  flag). the worker uses it to tell the kernel who to signal when the pipe becomes ready. */
  ownerChanged: ((fd: number) => void) | null = null;
  /** called when the signal mask changes so a newly-deliverable pending signal is not left waiting
   *  for the next periodic check. */
  signalsUnblocked: (() => void) | null = null;
  /** reports the first time each (syscall, errno) refusal happens. a call we answer ENOSYS or
   *  EINVAL to is often invisible: libc tries something, we refuse, it falls back and succeeds, and
   *  the only trace left is a stale errno that surfaces somewhere unrelated. */
  refusalHook: ((nr: number, errno: number) => void) | null = null;
  private seenRefusals = new Set<number>();

  /**
   * Record that a syscall answered with an error, once per (syscall, errno) pair.
   *
   * A failing syscall is usually invisible: the caller recovers, carries on, and the only evidence
   * is a wrong result somewhere else entirely. EAGAIN and EINTR are left out because they are
   * ordinary flow control rather than answers -- everything else is worth knowing about.
   */
  noteResult(nr: number, ret: bigint): void {
    if (ret >= 0n || ret < -4095n) return;
    const errno = Number(-ret);
    if (errno === 11 || errno === 4) return; // EAGAIN, EINTR
    const key = nr * 4096 + errno;
    if (this.seenRefusals.has(key)) return;
    this.seenRefusals.add(key);
    this.refusalHook?.(nr, errno);
  }
  /** fs.pipe-max-size, shared across every process. the ceiling on any one pipe's capacity, and
   *  what F_SETPIPE_SZ measures an unprivileged request against. */
  pipeMaxHook: { get: () => number; set: (v: number) => number } | null = null;
  /** read from a bridge-backed file, gathering as many transport-sized pieces as the caller asked
   *  for. returns the bytes read (possibly short only at end of file) or a positive errno. */
  readNativeHook: ((nativeFd: number, count: number) => Uint8Array | number) | null = null;
  /** resize a pipe's ring to `bytes`, returning what it actually got (or -1). */
  pipeResizeHook: ((fd: number, bytes: number) => number) | null = null;
  /** wire the /proc/sys files that name real settings to those settings. called once the vfs exists. */
  bindSysctls: ((vfs: any) => void) | null = null;
  /** PR_SET_CHILD_SUBREAPER lives in the shared process table: the kernel reads it when it
   *  reparents orphans, and it survives an exec, which this handler does not. */
  subreaperHook: { get: () => number; set: (on: number) => void } | null = null;
  /**
   * read and validate a capability header, as capget and capset both must.
   *
   * an unknown version is EINVAL, and the kernel writes back the one it prefers so a caller can
   * discover it by asking with a bad one -- which is exactly what libcap does at startup.
   */
  private capHeader(bytes: Uint8Array, view: DataView, ptr: number):
      { version: number; pid: number } | { err: number } {
    if (ptr === 0 || this.ptrWriteFaults(bytes, ptr, 8)) return { err: EFAULT };
    const base = ptr + GUEST_MEM_BASE;
    const version = view.getUint32(base, true);
    if (version !== 0x19980330 && version !== 0x20071026 && version !== 0x20080522) {
      view.setUint32(base, 0x20080522, true); // _LINUX_CAPABILITY_VERSION_3
      return { err: EINVAL };
    }
    return { version, pid: view.getInt32(base + 4, true) };
  }

  /** whether a pid names a live process, for the calls that answer ESRCH about one. */
  private pidExists(pid: number): boolean {
    if (pid === this.currentPid) return true;
    const list = this.prioHook?.list();
    return list ? list.some((p) => p.pid === pid) : false;
  }

  /** every socket this process holds, so a forked child can take its own reference to each. */
  inheritedSocketIds(): number[] {
    const ids = new Set<number>();
    for (const e of this.fdTable.values()) if (e.type === 'socket') ids.add(e.socketId);
    return [...ids];
  }

  /** set by a bridge apply that could not finish: the worker runs the syscall again. */
  private retryAfterBridge = false;
  takeRetry(): boolean { const r = this.retryAfterBridge; this.retryAfterBridge = false; return r; }

  /** the CURRENT parent, read from the shared process table rather than remembered. */
  ppidHook: (() => number) | null = null;
  /** hand a host descriptor back, for closes that cannot wait for a syscall reply. */
  closeNativeHook: ((nativeFd: number) => void) | null = null;
  closeSocketHook: ((socketId: number) => void) | null = null;
  /** how to give up a unix socket: its rings and its name live in the worker, not in this table. */
  closeUnixHook: ((fd: number) => void) | null = null;

  /** the timer slack this process is using, for /proc/self/timerslack_ns. */
  getTimerSlack(): number { return this.timerSlackNs; }
  /** set it from the /proc file. writing zero restores the inherited default, as prctl does. */
  setTimerSlack(v: number): number {
    if (!Number.isFinite(v) || v < 0) return -EINVAL;
    this.timerSlackNs = v === 0 ? this.defaultTimerSlackNs : v;
    return 0;
  }

  /** who is on the other end of a unix-domain socket, for SO_PEERCRED. only the worker knows:
   *  the connection is made through the kernel's name table, not through this fd table. */
  peerCredHook: ((fd: number) => { pid: number; uid: number; gid: number } | null) | null = null;
  sharedUnmapHook: ((start: number, end: number) => void) | null = null;
  // writable MAP_SHARED mappings of real (native) files. the guest writes into the mapping and
  // expects those bytes flushed back to the file on msync/munmap (this is how apk extracts
  // binaries). we track them and pwrite the dirty range through the bridge on flush.
  private fileMmaps: Array<{ start: number; len: number; fileLen: number; nativeFd: number; fileOffset: number }> = [];
  // guest ranges mapped PROT_NONE (inaccessible). a user pointer landing here faults with
  // EFAULT, which is how LTP's tst_get_bad_addr() probes are built.
  private protNone: Array<{ start: number; end: number }> = [];
  // page-aligned, non-overlapping locked-memory intervals (mmap MAP_LOCKED / mlock). their
  // total is reported as VmLck in /proc/self/status.
  // MLOCK_ONFAULT locks a range without populating it: the pages join the lock as they are
  // touched. that is the difference between "these pages are here now" and "these pages will stay
  // once they arrive", and mincore has to tell them apart.
  private lockedRegions: Array<{ start: number; end: number; onFault?: boolean }> = [];
  // set by mlockall(MCL_FUTURE): every mapping made from now on is locked as it is made.
  private lockFuture = false;
  // virtual memory areas (guest offsets == guest-visible addresses), sorted, non-overlapping.
  // drives MAP_FIXED_NOREPLACE overlap detection and the synthetic /proc/self/maps. prot holds
  // the PROT_* bits; shared distinguishes p/s in the maps perms column.
  // `maxProt` is the most this mapping may EVER be granted. mmap already refuses to hand out access
  // the descriptor lacks, but mprotect can be asked for it later, and a mapping is not allowed to
  // acquire by a second call what it was refused by the first.
  private vmas: Vma[] = [];

  /**
   * The regions of the address space a fork actually has to copy.
   *
   * Everything else is untouched, and sweeping it is not free: a snapshot used to scan from zero to
   * the top of the mmap arena looking for non-zero pages, which for an ordinary shell means reading
   * 34 MiB to find the two that matter -- 7 to 20 ms on EVERY fork, most of a process's whole
   * creation cost. It also missed anything mapped MAP_FIXED above the bump pointer, which a fork
   * silently failed to copy.
   *
   * [0, brk) covers the loaded image and the heap; STACK_TOP bounds the stack, which grows down and
   * has no vma of its own; the rest is whatever is mapped.
   */
  snapshotRanges(stackTop: number): Array<{ start: number; end: number }> {
    return this.readAddressSpace(() => this.snapshotRangesLocked(stackTop));
  }

  private snapshotRangesLocked(stackTop: number): Array<{ start: number; end: number }> {
    const out: Array<{ start: number; end: number }> = [
      { start: 0, end: Math.max(Number(this.brk), stackTop) },
    ];
    for (const v of this.vmas) out.push({ start: v.start, end: v.end });
    out.sort((a, b) => a.start - b.start);
    const merged: Array<{ start: number; end: number }> = [];
    for (const r of out) {
      const last = merged[merged.length - 1];
      if (last && r.start <= last.end) last.end = Math.max(last.end, r.end);
      else merged.push({ ...r });
    }
    return merged;
  }
  private cwd = '/';
  // the filesystem identity. linux keeps it equal to the effective one except when setfsuid or
  // setfsgid has deliberately moved it, and every file permission check uses THIS rather than the
  // effective id -- which is the entire reason the pair exists.
  private fsuid = 0;
  private fsgid = 0;

  private tpidr = 0n;
  private umask = 0o022;
  private fdCloexec = new Set<number>();
  // who receives the I/O-ready signal for a descriptor. linux calls this the descriptor's OWNER,
  // and it is a real piece of state: F_SETOWN records it and F_GETOWN reads it back. answering
  // F_GETOWN with the caller's own pid regardless of what was set is a plausible-looking lie that
  // breaks the moment a program hands a descriptor's notifications to somebody else.
  private fdOwner = new Map<number, { type: number; pid: number }>();
  // linux privilege is not "am I uid 0", it is a set of CAPABILITIES. a process can be uid 0 and
  // still lack the one permission it is asking for, and a process can hold a capability without
  // being root at all -- which is how containers give a program exactly the power it needs. these
  // used to be faked: capget always answered "all of them" and capset accepted anything and did
  // nothing, so dropping a capability appeared to work and changed nothing.
  private capEff = ALL_CAPS;
  private capPerm = ALL_CAPS;
  private capInh = 0n;
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
  // how far settimeofday/clock_settime has moved the wall clock from the host's. the host clock is
  // not ours to change, so the guest's realtime is the host's plus this.
  private clkRealOffsetNs = 0n;
  /** where the system-wide clock offset lives, once the worker has somewhere to put it. */
  clockOffsetHook: { get(): bigint; set(ns: bigint): void } | null = null;
  private get clockOffsetNs(): bigint { return this.clockOffsetHook ? this.clockOffsetHook.get() : this.clkRealOffsetNs; }
  private set clockOffsetNs(ns: bigint) {
    if (this.clockOffsetHook) this.clockOffsetHook.set(ns); else this.clkRealOffsetNs = ns;
    // the never-go-backwards clamp exists to hide DRIFT correction, not to overrule the caller:
    // settimeofday and clock_settime are allowed to move the clock anywhere, including back, and
    // holding the old floor made a receding set look like it had not happened.
    this.clkLastRealNs = 0n;
  }
  /** nanoseconds since this handler was constructed. the base every other clock is built on. */
  private elapsedNs(): bigint {
    const elapsedMs = (this.clkHasPerf ? performance.now() : Date.now()) - this.clkPerfBase;
    let ns = BigInt(Math.round(elapsedMs * 1_000_000));
    if (ns < this.clkLastMonoNs) ns = this.clkLastMonoNs; // never go backwards
    else this.clkLastMonoNs = ns;
    return ns;
  }
  // how old the system already was when this process started. CLOCK_MONOTONIC measures the
  // machine's uptime, so a process that starts an hour in must not read it as zero.
  private clkBootOffsetNs = 0n;
  setSystemBootMs(bootMs: number): void {
    if (!Number.isFinite(bootMs) || bootMs <= 0) return;
    const age = Date.now() - bootMs;
    this.clkBootOffsetNs = age > 0 ? BigInt(Math.round(age * 1_000_000)) : 0n;
  }
  private monoNs(): bigint { return this.clkBootOffsetNs + this.elapsedNs(); }

  /**
   * the wall clock, at nanosecond resolution, without drifting away from the machine's.
   *
   * the fine-grained answer is an epoch captured at startup plus monotonic elapsed time, because
   * Date.now() only counts milliseconds. but the two run at slightly different rates, and the
   * kernel timestamps its own objects (an ipc segment's ctime) from Date.now() -- so after a while
   * the process and the kernel disagreed by a whole second about what time it was, and a test that
   * checks a timestamp against its own clock reads a creation time from before it asked.
   */
  private realNs(): bigint {
    const wall = BigInt(Date.now()) * 1_000_000n;
    const est = this.clkRealBaseNs + this.elapsedNs();
    const drift = est - wall;
    // behind the machine's clock: jump forward to it. AHEAD of it: hold still and let it catch up,
    // never step back -- a wall clock that goes backwards breaks everything that measures with it,
    // and a few milliseconds of standing still breaks nothing.
    if (drift < -2_000_000n) this.clkRealBaseNs -= drift;
    let v = this.clkRealBaseNs + this.elapsedNs() + this.clockOffsetNs;
    if (drift > 2_000_000n) v -= drift - 2_000_000n;
    if (v < this.clkLastRealNs) v = this.clkLastRealNs; else this.clkLastRealNs = v;
    return v;
  }
  private clkLastRealNs = 0n;

  /**
   * the clock a timeout is measured against, in milliseconds.
   *
   * it has to be the SAME clock the guest reads through clock_gettime, or a wait can end before
   * the guest's own clock says the time has passed -- and every test that measures a sleep against
   * CLOCK_MONOTONIC calls that "woken up early". Date.now() has millisecond granularity, so a
   * deadline built from it was up to a millisecond short of what was asked for.
   */
  private nowMs(): number { return Number(this.monoNs() / 1000n) / 1000; }
  clockNs(clkId: number): bigint {
    // CLOCK_MONOTONIC(1)/RAW(4)/BOOTTIME(7) -> since construction; the cputime clocks -> this
    // process's own cpu time; CLOCK_REALTIME(0) and the rest -> wall clock, plus whatever
    // settimeofday moved it by
    // the cpu clocks measure this process rather than the world, so they are the one place the
    // answer is not some offset from the monotonic clock.
    if (clkId === CLOCK_CPU_OWN || clkId === CLOCK_THREAD_CPUTIME_ID) {
      return BigInt(Math.round(this.ownCpuMs() * 1_000_000));
    }
    // MONOTONIC(1), MONOTONIC_RAW(4), MONOTONIC_COARSE(6), BOOTTIME(7) and BOOTTIME_ALARM(9) all
    // count from boot; REALTIME(0), REALTIME_COARSE(5), REALTIME_ALARM(8) and TAI(11) are the wall
    // clock. the ALARM clocks differ from their bases only in waking a suspended machine, which
    // ours never is, and TAI differs by an offset only ntp can set.
    const isMono = clkId === 1 || clkId === 4 || clkId === 6 || clkId === 7 || clkId === 9;
    const ns = isMono ? this.monoNs() : this.realNs();
    // a coarse clock is the last tick rather than a fresh read, so it must not carry more
    // precision than it claims: a program comparing it against clock_getres would catch that.
    const res = this.clockResNs(clkId);
    return res > 1n ? (ns / res) * res : ns;
  }

  /**
   * whether a clock id names a clock at all.
   *
   * a NEGATIVE id is linux's encoding for a dynamic clock backed by a file descriptor
   * (CLOCK_INVALID and the /dev/ptp family); we have none, so those are EINVAL too. answering
   * successfully for every id, as this used to, meant clock_gettime could not fail.
   */
  private knownClock(clkId: number): boolean {
    return clkId >= 0 && clkId <= 11 && clkId !== 10;
  }

  /**
   * a clock's resolution, in nanoseconds.
   *
   * the COARSE clocks are the ones read straight off the last timer tick without a hardware read,
   * so their resolution is the tick -- linux reports 1/CONFIG_HZ. everything else is nanosecond.
   */
  private clockResNs(clkId: number): bigint {
    return (clkId === 5 || clkId === 6) ? 1_000_000n : 1n; // 1 ms at HZ=1000
  }
  // every timer this process owns -- the three interval timers, the posix timers of
  // timer_create, and the descriptors of timerfd_create -- is a deadline on some clock plus a
  // period. nothing preempts a running guest, so expiry is EVALUATED rather than scheduled:
  // wherever the process could observe a timer (the exec loop's periodic signal poll, a blocking
  // wait, a read of a timerfd) every armed timer is compared against its clock and the whole run
  // of expirations since the last look is accounted for at once. the COUNT matters as much as the
  // fact, because that count is exactly what overrun and a timerfd read report.
  private itimers: ArmedTimer[] = [
    { clockid: CLOCK_MONOTONIC, nextNs: 0n, intervalNs: 0n },  // ITIMER_REAL    -> SIGALRM
    { clockid: CLOCK_CPU_OWN,   nextNs: 0n, intervalNs: 0n },  // ITIMER_VIRTUAL -> SIGVTALRM
    { clockid: CLOCK_CPU_OWN,   nextNs: 0n, intervalNs: 0n },  // ITIMER_PROF    -> SIGPROF
  ];
  private posixTimers = new Map<number, PosixTimer>();
  private nextPosixTimerId = 0;
  private timerFds = new Map<number, TimerFdTimer>();
  private nextTimerFdId = 0;
  // prctl's per-process settings. these are real state: a program sets one and reads it back, and
  // answering a constant made every get/set pair disagree.
  private pdeathSig = 0;
  /** publish the parent-death signal into the shared process table. */
  pdeathHook: ((sig: number) => void) | null = null;
  private dumpable = 1;
  private keepCaps = 0;
  private noNewPrivs = 0;
  private thpDisable = 0;
  private childSubreaper = 0;
  private secureBits = 0;
  private timerSlackNs = 50000;
  // PR_SET_TIMERSLACK(0) restores the value this process STARTED with, which is whatever its
  // parent was using when it forked -- not a constant. keeping only the current value made the
  // reset land on 50 us for a child whose parent had chosen something else.
  private defaultTimerSlackNs = 50000;
  private comm = 'jitter';
  /** publish a new process name so /proc and ps see it too. */
  commHook: ((name: string) => void) | null = null;
  /** exec renames the process after the binary it loaded, and resets the dump flag with it. */
  setComm(name: string): void {
    this.comm = name.slice(0, 15);
    this.dumpable = 1;
    // execve destroys every posix timer. the interval timers survive it, which is what makes
    // alarm() still fire in the program you exec'd.
    this.posixTimers.clear();
  }
  // RLIMIT_MEMLOCK's soft limit, in bytes. linux's default is 8 MiB on most distributions, and
  // ltp's mlock tests lower it and expect the lower value to bite.
  private memlockLimit = 8 * 1024 * 1024;
  // the execution domain, as personality(2) sets and reads it. it survives exec and fork, so it
  // travels with the rest of the process state.
  private personality = 0;
  // the supplementary groups this process belongs to. they survive fork and exec, and they decide
  // group access to every file the process opens.
  private groups: number[] = [];
  // RLIMIT_NOFILE soft limit: the highest fd number a process may allocate. set via
  // prlimit64/setrlimit; enforced at fd-allocating syscalls (open/dup/...) -> EMFILE.
  private nofileLimit = 1024;
  // every resource limit a program has explicitly set, by resource number. what is not here takes
  // the system default from rlimitOf.
  private rlimits: Array<{ cur: bigint; max: bigint } | undefined> = [];
  /** cpu time this process has burned, in nanoseconds -- wall time minus every moment parked. */
  private cpuNs(): bigint { return BigInt(Math.round(this.ownCpuMs() * 1_000_000)); }

  /** the current value of the clock a timer is armed against. */
  private timerClockNs(clockid: number): bigint {
    return clockid === CLOCK_CPU_OWN ? this.cpuNs() : this.clockNs(clockid);
  }

  /**
   * how many times a deadline+period pair has expired by `now`, advancing it past all of them.
   *
   * a one-shot disarms. a periodic timer moves forward by whole periods, so a process that was
   * away for ten periods sees one expiry with an overrun of nine rather than ten separate ones --
   * which is what linux reports and what a timerfd read returns as its count.
   */
  private static expire(t: ArmedTimer, now: bigint): number {
    if (t.nextNs === 0n || now < t.nextNs) return 0;
    if (t.intervalNs <= 0n) { t.nextNs = 0n; return 1; }
    const n = 1n + (now - t.nextNs) / t.intervalNs;
    t.nextNs += n * t.intervalNs;
    return Number(n);
  }

  /** nothing armed anywhere: the common case, and worth not walking three collections for. */
  private noTimers(): boolean {
    return this.posixTimers.size === 0 && this.timerFds.size === 0
        && this.itimers[0].nextNs === 0n && this.itimers[1].nextNs === 0n && this.itimers[2].nextNs === 0n;
  }

  /**
   * settle every armed timer against its clock: queue the signals that are due, and credit a
   * timerfd with the expirations it can now report. returns true if a signal became pending.
   */
  checkTimers(): boolean {
    if (this.noTimers()) return false;
    let fired = false;
    for (let w = 0; w < 3; w++) {
      const t = this.itimers[w];
      if (t.nextNs === 0n) continue;
      if (SyscallHandler.expire(t, this.timerClockNs(t.clockid)) > 0) {
        this.raiseSignal(ITIMER_SIGNAL[w]);
        fired = true;
      }
    }
    for (const t of this.posixTimers.values()) {
      if (t.nextNs === 0n) continue;
      const n = SyscallHandler.expire(t, this.timerClockNs(t.clockid));
      if (n === 0) continue;
      t.accrued = Math.min(t.accrued + n, OVERRUN_CEIL);
      // SIGEV_NONE still runs the clock down -- it just has nobody to tell, so nothing is ever
      // delivered and the count it would have carried is never latched.
      if (t.notify === SIGEV_NONE) continue;
      // one signal stands for the whole run of expirations behind it, and it goes on standing for
      // them until somebody takes it: a process with the signal blocked, or merely slow to reach
      // its handler, can sit on a single pending SIGALRM through millions of periods. overrun is
      // how it finds that out, so the count is latched at the delivery it belongs to and left
      // alone afterwards -- recomputing it when asked would report the delay since, not the run.
      if (this.pendingSignals.includes(t.signo)) continue;
      t.overrun = clampOverrun(t.accrued - 1);
      t.accrued = 0;
      // siginfo for a timer is _sifields._timer: the timer's id, its overrun, and the value the
      // program attached to it -- laid out where si_pid, si_uid and si_value sit for a kill.
      this.raiseSignal(t.signo, { pid: t.id, uid: t.overrun, code: -2 /* SI_TIMER */, value: t.sigval });
      fired = true;
    }
    for (const t of this.timerFds.values()) {
      if (t.nextNs === 0n) continue;
      const n = SyscallHandler.expire(t, this.timerClockNs(t.clockid));
      if (n > 0) t.ticks += BigInt(n);
    }
    return fired;
  }

  /**
   * ms until the next timer that will deliver a SIGNAL fires, or 0 if there is none -- the bound a
   * blocking wait must not sleep past.
   *
   * timers on a cpu clock are deliberately absent: cpu time does not advance while the process is
   * parked, so such a timer cannot fire during the wait however long it lasts.
   */
  timerRemainingMs(): number {
    if (this.noTimers()) return 0;
    let best = -1;
    const consider = (t: ArmedTimer) => {
      if (t.nextNs === 0n || t.clockid === CLOCK_CPU_OWN) return;
      const ms = Math.max(1, Number((t.nextNs - this.timerClockNs(t.clockid)) / 1_000_000n));
      if (best < 0 || ms < best) best = ms;
    };
    for (const t of this.itimers) consider(t);
    for (const t of this.posixTimers.values()) if (t.notify !== SIGEV_NONE) consider(t);
    return best < 0 ? 0 : best;
  }

  /**
   * read an itimerval (frac = 1e6, microseconds) or itimerspec (frac = 1e9, nanoseconds): two
   * {seconds, fraction} pairs, the period first and the initial expiry second. null when either
   * is not a valid time -- a negative second count, or a fraction that should have been carried.
   */
  private readItimer(view: DataView, base: number, frac: bigint): { intervalNs: bigint; valueNs: bigint } | null {
    const perUnit = 1_000_000_000n / frac;
    const out: bigint[] = [];
    for (const off of [0, 16]) {
      const sec = view.getBigInt64(base + off, true);
      const sub = view.getBigInt64(base + off + 8, true);
      if (sec < 0n || sub < 0n || sub >= frac) return null;
      out.push(sec * 1_000_000_000n + sub * perUnit);
    }
    return { intervalNs: out[0], valueNs: out[1] };
  }

  private writeItimer(view: DataView, base: number, intervalNs: bigint, valueNs: bigint, frac: bigint): void {
    const perUnit = 1_000_000_000n / frac;
    view.setBigInt64(base, intervalNs / 1_000_000_000n, true);
    view.setBigInt64(base + 8, (intervalNs % 1_000_000_000n) / perUnit, true);
    view.setBigInt64(base + 16, valueNs / 1_000_000_000n, true);
    view.setBigInt64(base + 24, (valueNs % 1_000_000_000n) / perUnit, true);
  }

  /** the timer behind a descriptor, or the errno that descriptor deserves. */
  private timerFdOf(fd: number): TimerFdTimer | number {
    const e = this.getFdEntry(fd);
    if (!e) return this.fdIsOpen(fd) ? -EINVAL : -EBADF;
    if (e.type !== 'timerfd') return -EINVAL;
    return this.timerFds.get(e.id) ?? -EINVAL;
  }

  /** the itimerval/itimerspec a getter should report for an armed timer: remaining, then period. */
  private timerRemaining(t: ArmedTimer): { valueNs: bigint; intervalNs: bigint } {
    if (t.nextNs === 0n) return { valueNs: 0n, intervalNs: t.intervalNs };
    const rem = t.nextNs - this.timerClockNs(t.clockid);
    return { valueNs: rem > 0n ? rem : 1n, intervalNs: t.intervalNs };
  }
  pendingSignals: number[] = [];  // queue of signals to deliver
  private epollSets?: Map<number, Map<number, { events: number; data: bigint; lastReady: number }>>;
  private signalHandlers?: Map<number, { handler: bigint; flags: bigint; restorer: bigint; mask: bigint }>;
  /** the thread group's shared dispositions, once this process has more than one thread. */
  private sigTable?: Int32Array;
  private sigTableSeen = -1;
  private signalMask?: bigint;
  /** faulting address for the next fault-signal delivery (SIGSEGV/SIGBUS); set by the run loop's
   *  page-protection fault handler so the handler's siginfo.si_addr is accurate. */
  faultAddr = 0n;
  /** the stack pointer of the last rt_sigreturn we refused, for diagnosing where it came from. */
  badSigreturnSP = 0n;
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
  // publish this process's user ids where other processes can see them. kill(2) checks the
  // TARGET's ids, and those live in the target's own worker, so they have to be shared.
  credsHook: ((ruid: number, euid: number, suid: number, rgid: number) => void) | null = null;

  /**
   * the system's UTS names, read and written where every process can see them.
   *
   * a hostname belongs to the machine, not to whoever last asked: keeping a copy per process would
   * let two shells disagree about what host they are on. the worker points this at the shared
   * global page, so there is no round trip either.
   */
  utsHook: { get(which: 'host' | 'domain'): string; set(which: 'host' | 'domain', name: string): void } | null = null;
  /**
   * the kernel's eventfd counters. an eventfd is one counter with descriptors onto it, and fork
   * gives parent and child the SAME counter -- so it cannot live in either of them.
   */
  eventfdHook: ((m: Record<string, unknown>) => number[]) | null = null;
  private efdCount(id: number): bigint {
    const r = this.eventfdHook?.({ op: 'peek', id });
    if (!r || r[0] < 0) return 0n;
    return (BigInt(r[2] >>> 0) << 32n) | BigInt(r[1] >>> 0);
  }
  /** give up every eventfd this process holds. the process is ending or being replaced. */
  releaseEventfds(): void {
    for (const id of this.inheritedEventfdIds()) {
      this.eventfdHook?.({ op: 'close', id, pid: this.currentPid });
    }
  }
  /** every eventfd this process holds, so fork can take a reference for the child. */
  inheritedEventfdIds(): number[] {
    const ids = new Set<number>();
    for (const [, e] of this.fdTable) if (e.type === 'eventfd') ids.add(e.id);
    return [...ids];
  }
  /** file leases live in the kernel, where every process's opens are visible. */
  leaseHook: ((op: 'set' | 'get' | 'sig', nativeFd: number, path: string, arg: number,
               breakSig: number, isOwner: boolean, isRegular: boolean) => number) | null = null;
  /** the queued-signal budget a sender must respect, in the shared table. */
  sigpendingHook: ((limit: number) => void) | null = null;
  publishSigpending(): void {
    const cur = this.rlimitOf(RLIMIT_SIGPENDING).cur;
    this.sigpendingHook?.(cur >= 0x7fffffffn ? 0x7fffffff : Number(cur));
  }
  /** call after any change to the credentials, and once at startup. */
  publishCreds(): void {
    // every credential change carries the filesystem identity with it: linux resets fsuid to the
    // new euid whenever the effective id changes, so setfsuid only holds until the next setuid.
    // doing it here means it happens wherever credentials change, rather than in six places that
    // each have to remember.
    this.fsuid = this.euid;
    this.fsgid = this.egid;
    this.credsHook?.(this.ruid, this.euid, this.suid, this.rgid);
  }
  // spans of a file mapping that the file does not reach. touching one is SIGBUS rather than
  // SIGSEGV -- the mapping is legal, the memory behind it simply does not exist -- and keeping the
  // reason here rather than inferring it from the permission table means the fault handler asks a
  // question with a definite answer.
  private noBacking: Array<{ start: number; end: number }> = [];

  /** the recorded no-backing spans, for diagnostics. */
  noBackingRanges(): string { return this.noBacking.map((r) => `${r.start}-${r.end}`).join(','); }

  /** is this address inside a mapping the file does not reach? */
  isNoBacking(addr: number): boolean {
    for (const r of this.noBacking) if (addr >= r.start && addr < r.end) return true;
    return false;
  }
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
  // the whole meta array is kept, not a chosen few fields: a cache that remembers less than the
  // bridge told us answers a later stat with zeros where the timestamps should be.
  private statCache = new Map<string, { meta: number[]; expiresAt: number }>();
  private readonly STAT_CACHE_TTL_MS = 1000;
  /** entries live at most a second, but nothing was ever pruning the expired ones: a process that
   *  walked a large tree (one getdents fills an entry PER FILE) grew this map for as long as it
   *  ran. a cache that cannot forget is not a cache, it is a leak with a hit rate. */
  private readonly STAT_CACHE_MAX = 4096;

  private cacheStat(key: string, value: { meta: number[]; expiresAt: number }): void {
    if (this.statCache.size >= this.STAT_CACHE_MAX) {
      const now = Date.now();
      for (const [k, v] of this.statCache) if (v.expiresAt <= now) this.statCache.delete(k);
      // still full: drop the oldest, which a Map hands us in insertion order.
      while (this.statCache.size >= this.STAT_CACHE_MAX) {
        const oldest = this.statCache.keys().next().value;
        if (oldest === undefined) break;
        this.statCache.delete(oldest);
      }
    }
    this.statCache.set(key, value);
  }

  /** forget what we knew about a path. a file has TWO cache entries -- the follow view and the
   *  nofollow one lstat reads -- and dropping only one of them leaves the other to answer with
   *  the state before the change. */
  /**
   * forget the cached stat of the file behind a descriptor. copy_file_range writes through the
   * bridge rather than through write(2), so nothing on the normal write path knows the file has
   * changed -- and the next fstat would report the size and mtime it had before the copy.
   */
  invalidateStatFd(fd: number): void {
    const e = this.getFdEntry(fd);
    if (e?.type === 'native') this.invalidateStat(e.path);
  }

  private invalidateStat(path: string): void {
    this.statCache.delete(path);
    this.statCache.delete('L:' + path);
  }
  lastWaitStatus?: number;  // exit code from last child (set by dispatch loop)
  lastWaitPid?: number;     // PID of last exited child

  constructor(
    brkStart: bigint,
    io: IOInterface,
    vfs?: VFS,
    bridge?: NativeBridge,
  ) {
    this.brk = brkStart;
    this.brkFloor = brkStart;
    this.io = io;
    this.vfs = vfs;
    this.bridge = bridge;
    this.fdTable.set(0, { type: 'stdio', fd: 0 });
    this.fdTable.set(1, { type: 'stdio', fd: 1 });
    this.fdTable.set(2, { type: 'stdio', fd: 2 });
    // the standard descriptors arrive with real access modes, and anything that asks how a
    // descriptor was opened has to get the truth: without these, stdout looked O_RDONLY, so a
    // read lock on it was accepted where the kernel answers EBADF.
    this.fdFlags.set(0, 0);          // O_RDONLY
    this.fdFlags.set(1, 1);          // O_WRONLY
    this.fdFlags.set(2, 1);          // O_WRONLY
    if (this.vfs) {
      this.vfs.fdAllocator = () => this.allocFd();
      this.vfs.credsProvider = () => ({ ruid: this.ruid, euid: this.euid, rgid: this.rgid, egid: this.egid });
      this.vfs.selfCpuProvider = () => ({ utimeMs: this.ownUserMs(), stimeMs: this.ownSysMs() });
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
  /**
   * what an *at() call should answer about its DIRECTORY descriptor, before it looks at the name:
   * EBADF for one that is not open, ENOTDIR for one that is not a directory. a relative name is the
   * only case where the descriptor means anything -- an absolute one ignores it entirely.
   */
  private atDirfdError(dirfd: bigint, path: string): number {
    if (path.startsWith('/')) return 0;
    const d = Number(BigInt.asIntN(32, dirfd));
    if (d === -100) return 0;                       // AT_FDCWD
    const e = this.getFdEntry(d);
    if (!e) return this.vfs?.hasFd(d) ? 0 : EBADF;
    if (e.type === 'native') return e.isDir === false ? ENOTDIR : 0;
    // a pipe, a socket or an eventfd is not a directory either.
    return e.type === 'vfs' ? 0 : ENOTDIR;
  }

  private resolveAt(dirfd: bigint, path: string): string {
    if (path.startsWith('/')) return path;
    // a dirfd is an INT in every prototype that takes one, so only the low 32 bits mean anything.
    // reading all 64 matters: a variadic syscall() wrapper leaves the upper half of an int
    // argument unspecified, so AT_FDCWD arrives as 0x00000000ffffff9c about as often as it
    // arrives sign-extended -- and read as 64 bits that is 4294967196, not -100.
    const d = Number(BigInt.asIntN(32, dirfd));
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

  /**
   * re-read a shared file mapping from the file, for msync(MS_INVALIDATE).
   *
   * the point of MS_INVALIDATE is to throw away what the mapping has cached so the next read sees
   * the FILE. our mapping is a copy taken when it was made, so a write through the descriptor is
   * invisible to it until this happens -- which is exactly what a program does when it uses the
   * two interchangeably.
   */
  private buildMmapReload(cpu: any, memory: WebAssembly.Memory, reqStart: number, reqLen: number): SyscallResult | null {
    for (const m of this.fileMmaps) {
      if (reqStart >= m.start + m.len || reqStart + reqLen <= m.start) continue;
      const fStart = Math.max(reqStart, m.start);
      const fEnd = Math.min(reqStart + reqLen, m.start + m.fileLen);
      if (fEnd <= fStart) return null;
      const want = Math.min(fEnd - fStart, BRIDGE_READ_MAX);
      return {
        status: 'bridge', method: 'fs_pread',
        args: { fd: m.nativeFd, offset: m.fileOffset + (fStart - m.start), count: want },
        apply: (resp: any) => {
          if (resp.ret >= 0 && resp.bytes) {
            new Uint8Array(memory.buffer).set(resp.bytes, fStart + GUEST_MEM_BASE);
          }
          cpu.setReg(0, 0n);
        },
      };
    }
    return null;
  }

  // mark [start,end) (page-aligned) as locked or unlocked in the VmLck accounting, keeping the
  // interval set normalized (non-overlapping, merged), then republish the total to the vfs.
  /** whether any part of [start, end) is pinned in memory. */
  private rangeLocked(start: number, end: number): boolean {
    return this.lockedRegions.some((r) => r.start < end && r.end > start);
  }

  private setLocked(start: number, end: number, locked: boolean, onFault = false): void {
    if (end <= start) return;
    const next: Array<{ start: number; end: number; onFault?: boolean }> = [];
    for (const r of this.lockedRegions) {
      if (r.end <= start || r.start >= end) { next.push(r); continue; }
      if (r.start < start) next.push({ start: r.start, end: start });
      if (r.end > end) next.push({ start: end, end: r.end });
    }
    if (locked) next.push({ start, end, onFault });
    next.sort((a, b) => a.start - b.start);
    const merged: Array<{ start: number; end: number; onFault?: boolean }> = [];
    for (const r of next) {
      const last = merged[merged.length - 1];
      // only ranges locked the same way merge: one that was populated and one that was not are
      // different promises about the same kind of thing.
      if (last && r.start <= last.end && !!last.onFault === !!r.onFault) {
        last.end = Math.max(last.end, r.end);
      } else {
        merged.push({ ...r });
      }
    }
    this.lockedRegions = merged;
    let total = 0;
    for (const r of merged) total += r.end - r.start;
    if (this.vfs) (this.vfs as any).vmLckKb = Math.floor(total / 1024);
  }

  // does [start,end) overlap any tracked vma? used for MAP_FIXED_NOREPLACE.
  vmaOverlaps(start: number, end: number): boolean {
    const v = this.vmas[this.vmaLower(start)];
    return !!v && v.start < end;
  }

  // register the lazy /proc/self/maps renderer once the vma set is live.
  private publishVmas(): void {
    if (this.vfs && !(this.vfs as any).procMapsGen) (this.vfs as any).procMapsGen = () => this.renderProcMaps();
    if (this.vfs && !(this.vfs as any).procSmapsGen) (this.vfs as any).procSmapsGen = () => this.renderProcSmaps();
  }

  // vmas is kept sorted and non-overlapping, so all edits are a single O(n) pass with inline
  // coalescing of adjacent identical vmas: no re-sort, which keeps mapping-heavy loops linear.
  private vmaEmit(out: Vma[], v: Vma): void {
    const last = out[out.length - 1];
    // only merge neighbours that agree about their ceiling too, or a read-only file mapping could
    // be absorbed into a writable neighbour and quietly inherit permission to be made writable.
    if (last && last.end === v.start && last.prot === v.prot && last.shared === v.shared
        && last.maxProt === v.maxProt && !!last.grow === !!v.grow) last.end = v.end;
    else out.push(v);
  }

  // carve [start,end) out of the vma set (splitting boundary vmas), then optionally insert a fresh
  // vma with the given prot/shared, all in sorted order.
  /** set x0 to -errno and finish the syscall. */
  private retErr(cpu: any, errno: number): SyscallResult {
    cpu.setReg(0, BigInt.asUintN(64, BigInt(-errno)));
    return { status: 'ok' };
  }

  // ── the thread group's shared address space ───────────────────────────────
  //
  // threads share one address space, so they must share one map of it. each thread worker used to
  // get a COPY of the vma list at clone and never resync, which meant an mmap in one thread was
  // invisible to its siblings: two threads could be handed the SAME address, /proc/self/maps was
  // wrong in every thread but the one that made the mapping, and a fork from a non-leader thread
  // lost whatever its siblings had mapped.
  //
  // the list therefore lives in the shared memory, and each thread keeps its js copy in step with
  // it through a generation counter: pull when the shared copy has moved on, push after an edit.
  // the js array stays the working representation, so the sorted-splice logic is single-sourced
  // and costs the same as before -- a splice was already O(n), and so is a serialize.
  private vmaTable: Int32Array | null = null;
  private vmaGen = 0;
  private vmaDepth = 0;
  /** set by any edit, so a section that only read publishes nothing and siblings keep their view. */
  private vmaDirty = false;

  /** point this handler at its thread group's shared vma table. */
  attachVmaTable(memory: WebAssembly.Memory, base: number): void {
    this.vmaTable = new Int32Array(memory.buffer, base, VMA_TABLE_BYTES >> 2);
  }

  /**
   * the thread group's shared signal dispositions.
   *
   * sigaction(2) is a PROCESS-wide setting: a handler installed by any thread is the handler every
   * thread of the group runs. keeping it per-thread meant a handler installed by a worker thread
   * was invisible to the main one, which then took the DEFAULT action for a signal the program had
   * plainly asked to catch.
   *
   * the layout is a header -- lock, generation, and which signals have been set at all -- then one
   * 8-word record per signal. readers keep a local copy and refresh it when the generation moves,
   * which is checked at every syscall boundary; that is the only place a signal can be delivered.
   */
  attachSigTable(memory: WebAssembly.Memory, base: number): void {
    this.sigTable = new Int32Array(memory.buffer, base, SIG_TABLE_BYTES >> 2);
    this.sigTableSeen = -1;
  }

  /** publish what this thread has as the group's initial state (called once, at promote). */
  publishSigTable(): void {
    const t = this.sigTable;
    if (!t) return;
    this.sigAcquire();
    try {
      Atomics.store(t, 2, 0); Atomics.store(t, 3, 0);
      for (const [sig, h] of this.signalHandlers ?? []) this.sigWriteLocked(sig, h);
      this.sigTableSeen = Atomics.add(t, 1, 1) + 1;
    } finally { this.sigRelease(); }
  }

  /** refresh the local copy if another thread has changed a disposition since we last looked. */
  sigSync(): void {
    const t = this.sigTable;
    if (!t) return;
    const gen = Atomics.load(t, 1);
    if (gen === this.sigTableSeen) return;
    this.sigAcquire();
    try {
      const present = (BigInt(Atomics.load(t, 3) >>> 0) << 32n) | BigInt(Atomics.load(t, 2) >>> 0);
      const next = new Map<number, { handler: bigint; flags: bigint; restorer: bigint; mask: bigint }>();
      for (let sig = 1; sig <= 64; sig++) {
        if ((present & (1n << BigInt(sig - 1))) === 0n) continue;
        const w = SIG_TABLE_FIRST + (sig - 1) * 8;
        const rd = (i: number) => (BigInt(Atomics.load(t, w + i + 1) >>> 0) << 32n) | BigInt(Atomics.load(t, w + i) >>> 0);
        next.set(sig, { handler: rd(0), flags: rd(2), restorer: rd(4), mask: rd(6) });
      }
      this.signalHandlers = next;
      this.sigTableSeen = Atomics.load(t, 1);
    } finally { this.sigRelease(); }
  }

  /** record one disposition for the whole group. the caller holds the lock. */
  private sigWriteLocked(sig: number, h: { handler: bigint; flags: bigint; restorer: bigint; mask: bigint }): void {
    const t = this.sigTable!;
    const w = SIG_TABLE_FIRST + (sig - 1) * 8;
    const wr = (i: number, v: bigint) => {
      Atomics.store(t, w + i, Number(v & 0xffffffffn) | 0);
      Atomics.store(t, w + i + 1, Number((v >> 32n) & 0xffffffffn) | 0);
    };
    wr(0, h.handler); wr(2, h.flags); wr(4, h.restorer); wr(6, h.mask);
    const bit = sig - 1;
    if (bit < 32) Atomics.or(t, 2, 1 << bit); else Atomics.or(t, 3, 1 << (bit - 32));
  }

  /** publish one changed disposition to the group. */
  private sigPublish(sig: number, h: { handler: bigint; flags: bigint; restorer: bigint; mask: bigint } | undefined): void {
    const t = this.sigTable;
    if (!t) return;
    this.sigAcquire();
    try {
      if (h) this.sigWriteLocked(sig, h);
      else {
        const bit = sig - 1;
        if (bit < 32) Atomics.and(t, 2, ~(1 << bit)); else Atomics.and(t, 3, ~(1 << (bit - 32)));
      }
      this.sigTableSeen = Atomics.add(t, 1, 1) + 1;
    } finally { this.sigRelease(); }
  }

  private sigAcquire(): void {
    const t = this.sigTable;
    if (!t) return;
    for (let spin = 0; spin < 64; spin++) {
      if (Atomics.compareExchange(t, 0, 0, 1) === 0) return;
    }
    while (Atomics.compareExchange(t, 0, 0, 1) !== 0) Atomics.wait(t, 0, 1, 5);
  }

  private sigRelease(): void {
    const t = this.sigTable;
    if (!t) return;
    Atomics.store(t, 0, 0);
    Atomics.notify(t, 0, 1);
  }

  /** publish the current list as the group's initial shared state (called once, at promote). */
  publishVmaTable(): void {
    if (!this.vmaTable) return;
    this.vmaAcquire();
    try { this.vmaPush(); } finally { this.vmaRelease(); }
  }

  private vmaAcquire(): void {
    const t = this.vmaTable;
    if (!t) return;
    // the critical sections here are a handful of microseconds of array work with no i/o, so a
    // brief spin then a futex wait is the right shape: contention is rare and short.
    for (let spin = 0; spin < 64; spin++) {
      if (Atomics.compareExchange(t, 0, 0, 1) === 0) return;
    }
    while (Atomics.compareExchange(t, 0, 0, 1) !== 0) Atomics.wait(t, 0, 1, 5);
  }

  private vmaRelease(): void {
    const t = this.vmaTable;
    if (!t) return;
    Atomics.store(t, 0, 0);
    Atomics.notify(t, 0, 1);
  }

  /** adopt the shared list if another thread has changed it since we last looked. */
  private vmaPull(): void {
    const t = this.vmaTable;
    if (!t) return;
    const gen = Atomics.load(t, 1);
    if (gen === this.vmaGen) return;
    // the break and the arena cursor are address-space state too: a thread that extended the heap
    // or took the next arena slot has moved them for the whole group, and a sibling working from
    // its own copy would hand out memory that is already in use.
    this.brk = BigInt(Atomics.load(t, 3) >>> 0);
    this.mmapBase = BigInt(Atomics.load(t, 4) >>> 0);
    const n = Atomics.load(t, 2);
    const out: Vma[] = new Array(n);
    for (let i = 0; i < n; i++) {
      const b = VMA_TABLE_FIRST + i * 3;
      const packed = t[b + 2];
      out[i] = {
        start: t[b] >>> 0, end: t[b + 1] >>> 0,
        prot: packed & 7, shared: (packed & 16) !== 0,
        maxProt: (packed >>> 8) & 7, grow: (packed & 32) !== 0,
      };
    }
    this.vmas = out;
    this.vmaGen = gen;
  }

  /** hand our list to the group and mark it as the newest. */
  private vmaPush(): void {
    const t = this.vmaTable;
    if (!t) return;
    // maxMapCount is capped at the table size and enforced on every path that can add a vma, so
    // this can only ever be the whole list; the clamp is a backstop against silent truncation.
    const n = Math.min(this.vmas.length, VMA_TABLE_ENTRIES);
    for (let i = 0; i < n; i++) {
      const v = this.vmas[i], b = VMA_TABLE_FIRST + i * 3;
      t[b] = v.start; t[b + 1] = v.end;
      t[b + 2] = (v.prot & 7) | (v.shared ? 16 : 0) | (v.grow ? 32 : 0) | ((v.maxProt & 7) << 8);
    }
    Atomics.store(t, 2, n);
    Atomics.store(t, 3, Number(this.brk) | 0);
    Atomics.store(t, 4, Number(this.mmapBase) | 0);
    this.vmaGen = Atomics.add(t, 1, 1) + 1;
  }

  /** read the group's address space without changing it. takes the lock: a plain read could
   *  otherwise land in the middle of a sibling's write and see half of two different lists. */
  private readAddressSpace<T>(fn: () => T): T {
    if (!this.vmaTable || this.vmaDepth > 0) return fn();
    this.vmaAcquire();
    this.vmaDepth = 1;
    this.vmaDirty = false;
    try {
      this.vmaPull();
      const r = fn();
      // a "read" that turned out to edit still has to publish, or the change is lost.
      if (this.vmaDirty) this.vmaPush();
      return r;
    } finally {
      this.vmaDepth = 0;
      this.vmaDirty = false;
      this.vmaRelease();
    }
  }

  /**
   * Run `fn` with the thread group's address space held still.
   *
   * choosing an address and claiming it have to be one indivisible step, or two threads mmapping
   * at the same moment both first-fit into the same hole and are handed the same memory. reentrant,
   * so the individual vma helpers can use it without caring whether a caller already holds it.
   */
  withAddressSpace<T>(fn: () => T): T {
    if (!this.vmaTable || this.vmaDepth > 0) {
      this.vmaDepth++;
      try { return fn(); } finally { this.vmaDepth--; }
    }
    this.vmaAcquire();
    this.vmaDepth = 1;
    this.vmaDirty = false;
    try {
      this.vmaPull();
      const r = fn();
      // publish only a section that actually changed something: a needless generation bump would
      // make every sibling thread re-read the whole list for nothing.
      if (this.vmaDirty) this.vmaPush();
      return r;
    } finally {
      this.vmaDepth = 0;
      this.vmaDirty = false;
      this.vmaRelease();
    }
  }

  /** index of the first vma that reaches past `addr`. the list is sorted, so this is a search
   *  rather than a walk -- which is what keeps a process with tens of thousands of mappings from
   *  paying a full pass per mmap. */
  private vmaLower(addr: number): number {
    let lo = 0, hi = this.vmas.length;
    while (lo < hi) {
      const m = (lo + hi) >> 1;
      if (this.vmas[m].end <= addr) lo = m + 1; else hi = m;
    }
    return lo;
  }

  /** the half-open slice of vmas that [start,end) touches. */
  private vmaSlice(start: number, end: number): { i: number; j: number } {
    const i = this.vmaLower(start);
    let j = i;
    while (j < this.vmas.length && this.vmas[j].start < end) j++;
    return { i, j };
  }

  /** what the slice [i,j) becomes when [start,end) is carved out and `insert` put in its place. */
  private vmaPieces(start: number, end: number, i: number, j: number,
                    insert: { prot: number; shared: boolean; maxProt: number; grow?: boolean } | null): Vma[] {
    const pieces: Vma[] = [];
    if (j > i) {
      const first = this.vmas[i];
      if (first.start < start) pieces.push({ start: first.start, end: start, prot: first.prot, shared: first.shared, maxProt: first.maxProt, grow: first.grow });
    }
    if (insert) pieces.push({ start, end, prot: insert.prot, shared: insert.shared, maxProt: insert.maxProt, grow: insert.grow });
    if (j > i) {
      const last = this.vmas[j - 1];
      if (last.end > end) pieces.push({ start: end, end: last.end, prot: last.prot, shared: last.shared, maxProt: last.maxProt, grow: last.grow });
    }
    return pieces;
  }

  /** replace vmas[i,i+del) with `pieces`, coalescing only at the two seams -- everything further
   *  out was already maximally merged, so there is nothing else that could have become mergeable. */
  private vmaSplice(i: number, del: number, pieces: Vma[]): void {
    this.vmaDirty = true;
    const merged: Vma[] = [];
    for (const p of pieces) this.vmaEmit(merged, p);
    // splice is the only whole-array operation on this path, and it made every mmap cost time
    // proportional to how many mappings the process already had: 113 us at ten thousand, 257 us at
    // twenty, which is O(n^2) over a program's life and why 65530 mappings took a minute and a half.
    // the two shapes that actually happen -- appending a new mapping at the top of the arena, and
    // replacing one range with one range -- need no shifting at all.
    if (del === 0 && merged.length === 1 && i === this.vmas.length) {
      this.vmas.push(merged[0]);
    } else if (del === 1 && merged.length === 1) {
      this.vmas[i] = merged[0];
    } else {
      this.vmas.splice(i, del, ...merged);
    }
    const lo = Math.max(0, i - 1);
    const hi = Math.min(this.vmas.length - 1, i + merged.length);
    for (let k = hi; k > lo; k--) {
      const a = this.vmas[k - 1], b = this.vmas[k];
      if (a.end === b.start && a.prot === b.prot && a.shared === b.shared && a.maxProt === b.maxProt
          && !!a.grow === !!b.grow) {
        a.end = b.end;
        this.vmas.splice(k, 1);
      }
    }
    this.publishVmas();
  }

  /**
   * would this edit push the process past vm.max_map_count?
   *
   * splitting a mapping costs a vma, which is why unmapping the MIDDLE of one can fail with ENOMEM
   * on a process that is already at the limit even though it is giving memory back.
   */
  private vmaWouldExceed(start: number, end: number,
                         insert: { prot: number; shared: boolean; maxProt: number; grow?: boolean } | null): boolean {
    if (end <= start) return false;
    const { i, j } = this.vmaSlice(start, end);
    const delta = this.vmaPieces(start, end, i, j, insert).length - (j - i);
    return delta > 0 && this.vmas.length + delta > this.maxMapCount;
  }

  private vmaSet(start: number, end: number, insert: { prot: number; shared: boolean; maxProt: number; grow?: boolean } | null): void {
    if (end <= start) return;
    // giving address space back has to be visible to the allocator's hint, or the hole is only
    // found on the fallback scan and the arena grows past perfectly good free space.
    if (!insert && start < this.mmapHint) this.mmapHint = start;
    this.withAddressSpace(() => {
      const { i, j } = this.vmaSlice(start, end);
      this.vmaSplice(i, j - i, this.vmaPieces(start, end, i, j, insert));
    });
  }

  // set the protection of [start,end), splitting vmas at the boundaries (mprotect).
  private vmaProtect(start: number, end: number, prot: number): void {
    if (end <= start) return;
    this.withAddressSpace(() => this.vmaProtectLocked(start, end, prot));
  }

  private vmaProtectLocked(start: number, end: number, prot: number): void {
    const { i, j } = this.vmaSlice(start, end);
    const pieces: Vma[] = [];
    for (let k = i; k < j; k++) {
      const v = this.vmas[k];
      if (v.start < start) pieces.push({ start: v.start, end: start, prot: v.prot, shared: v.shared, maxProt: v.maxProt, grow: v.grow });
      pieces.push({ start: Math.max(v.start, start), end: Math.min(v.end, end), prot, shared: v.shared, maxProt: v.maxProt, grow: v.grow });
      if (v.end > end) pieces.push({ start: end, end: v.end, prot: v.prot, shared: v.shared, maxProt: v.maxProt, grow: v.grow });
    }
    this.vmaSplice(i, j - i, pieces);
  }

  /**
   * paint the initial page-protection table.
   *
   * it used to start as "everything accessible" and only mark what was explicitly unmapped, which
   * meant a wild pointer into address space that was never mapped at all read and wrote real bytes
   * instead of faulting. start from the opposite end: nothing is reachable unless something put it
   * there. the program image, its heap and its stack all live below the mmap arena in one run, and
   * everything above is described by a vma.
   */
  /** record the program image's loaded segments as mappings. the loader puts them there, so
   *  nothing else knows about them, and the vma set is what fork, /proc/self/maps and the page
   *  guard all read. */
  adoptImage(segs: Array<{ start: number; end: number; prot: number }>): void {
    for (const g of segs) this.vmaSet(g.start, g.end, { prot: g.prot, shared: false, maxProt: 0x7 });
  }

  paintProtTable(stackTop: number, set: (start: number, end: number, prot: number) => void): void {
    this.readAddressSpace(() => this.paintProtTableLocked(stackTop, set));
  }

  private paintProtTableLocked(stackTop: number, set: (start: number, end: number, prot: number) => void): void {
    // everything below the mmap arena is the process's fixed layout -- the program image, its
    // stack, its tls block, the dynamic loader -- all of which exists by construction. above it,
    // only what mmap handed out exists, and that is exactly what the vma set describes.
    set(0, Math.max(Number(this.brk), stackTop, this.mmapArenaStart), 7);
    for (const v of this.vmas) set(v.start, v.end, v.prot);
    for (const r of this.noBacking) set(r.start, r.end, PAGE_NO_BACKING);
  }

  /** the guard gap the kernel keeps between a growable stack and the next mapping below it. */
  private static readonly STACK_GUARD_GAP = 256 * 4096;

  /**
   * a fault just below a MAP_GROWSDOWN mapping is the stack asking for more room, not an error.
   * extend the mapping down to cover `addr` -- unless doing so would come within the guard gap of
   * whatever is mapped below, which is the whole point of the gap: a stack must never be able to
   * grow silently into its neighbour. returns the newly covered range, or null to let it fault.
   */
  growStackFor(addr: number): { start: number; end: number; prot: number } | null {
    return this.withAddressSpace(() => this.growStackForLocked(addr));
  }

  private growStackForLocked(addr: number): { start: number; end: number; prot: number } | null {
    const page = addr & ~0xFFF;
    const i = this.vmaLower(page);
    const v = this.vmas[i];
    if (!v || !v.grow || v.start <= page) return null;
    if (v.start - page > SyscallHandler.STACK_GUARD_GAP) return null; // too far below to be the stack
    const below = i > 0 ? this.vmas[i - 1].end : 0;
    if (page - below < SyscallHandler.STACK_GUARD_GAP) return null;   // would breach the guard gap
    const grown = { start: page, end: v.start, prot: v.prot };
    this.vmaSet(page, v.start, { prot: v.prot, shared: v.shared, maxProt: v.maxProt, grow: true });
    return grown;
  }

  /**
   * whether [start, end) is covered by mappings with no hole. madvise, mincore, mlock and msync
   * all answer ENOMEM for a range that is not, and a hole in the MIDDLE counts -- which is why
   * this walks rather than just checking the ends.
   */
  private rangeMapped(start: number, end: number): boolean {
    return this.readAddressSpace(() => {
      let at = start;
      let i = this.vmaLower(at);
      while (at < end) {
        const v = this.vmas[i];
        if (!v || v.start > at) return false;
        at = v.end;
        i++;
      }
      return true;
    });
  }

  /**
   * what kind of memory [start, end) is, as far as madvise cares.
   *
   * madvise's advice values are not universally applicable: MADV_FREE and MADV_WIPEONFORK are
   * defined only on private anonymous memory, MADV_REMOVE only on shared memory that has no file
   * behind it, and MADV_DONTNEED refuses a locked range. accepting them everywhere made madvise a
   * call that could not fail.
   */
  private rangeKind(start: number, end: number): { shared: boolean; fileBacked: boolean; locked: boolean } {
    const shared = this.readAddressSpace(() => {
      let at = start, i = this.vmaLower(at);
      while (at < end) {
        const v = this.vmas[i];
        if (!v || v.start > at) break;
        if (v.shared) return true;
        at = v.end; i++;
      }
      return false;
    });
    const fileBacked = this.fileMmaps.some((m) => m.start < end && m.start + m.len > start);
    const locked = this.lockedRegions.some((r) => r.start < end && r.end > start);
    return { shared, fileBacked, locked };
  }

  /** the vma covering `addr`, or null. */
  private vmaAt(addr: number): Vma | null {
    const i = this.vmaLower(addr);
    const v = this.vmas[i];
    return v && v.start <= addr ? v : null;
  }

  /**
   * cpu time this process has used, in milliseconds. supplied by the worker, which is the only
   * layer that can see the difference between running and being parked in a blocking wait. zero
   * when there is no worker (the host-side unit tests), which is honest: nothing ran.
   */
  cpuMsHook: (() => number) | null = null;
  private ownCpuMs(): number { return this.cpuMsHook ? this.cpuMsHook() : 0; }
  /** the part of that cpu time spent inside the kernel, and the rest of it. */
  sysMsHook: (() => number) | null = null;
  private ownSysMs(): number { return Math.min(this.sysMsHook ? this.sysMsHook() : 0, this.ownCpuMs()); }
  private ownUserMs(): number { return Math.max(0, this.ownCpuMs() - this.ownSysMs()); }
  /** resident size in KiB. the worker owns the stack top, so it is the one that can ask. */
  rssKbHook: (() => number) | null = null;

  // what reaped children have cost, which is what getrusage(RUSAGE_CHILDREN) and times()' cutime
  // report. linux accumulates these across every child a process ever waits for, so they only grow.
  private childUtimeMs = 0;
  private childStimeMs = 0;
  private childMaxRssKb = 0;
  noteChildCpu(utimeMs: number, stimeMs: number, maxRssKb = 0): void {
    this.childUtimeMs += utimeMs;
    this.childStimeMs += stimeMs;
    if (maxRssKb > this.childMaxRssKb) this.childMaxRssKb = maxRssKb;
  }

  /**
   * fill a struct rusage (144 bytes on aarch64) at a guest address.
   *
   * the fields we do not measure are written as zero, which is the point: wait4(2) used to leave
   * the caller's buffer untouched entirely, so a program read back whatever its own stack happened
   * to hold. `time` reported 2329 hours of system time for a 50ms command that way.
   */
  writeRusage(view: DataView, addr: number, utimeMs: number, stimeMs: number, maxRssKb: number): void {
    const base = addr + GUEST_MEM_BASE;
    for (let i = 0; i < 144; i += 8) view.setBigUint64(base + i, 0n, true);
    view.setBigUint64(base + 0, BigInt(Math.floor(utimeMs / 1000)), true);
    view.setBigUint64(base + 8, BigInt(Math.round((utimeMs % 1000) * 1000)), true);
    view.setBigUint64(base + 16, BigInt(Math.floor(stimeMs / 1000)), true);
    view.setBigUint64(base + 24, BigInt(Math.round((stimeMs % 1000) * 1000)), true);
    view.setBigUint64(base + 32, BigInt(Math.max(0, Math.floor(maxRssKb))), true); // ru_maxrss
  }

  /**
   * one line describing what the address space says about `addr`: the vma covering it, or the
   * gap it falls in with its neighbours named. an undecodable instruction that reads as
   * 0x00000000 is almost never a translator gap -- it is memory that is not there -- and the two
   * are indistinguishable from the opcode alone, so the fault report has to say which.
   */
  describeAddr(addr: number): string {
    return this.readAddressSpace(() => {
      const v = this.vmaAt(addr);
      if (v) return `in vma [0x${v.start.toString(16)},0x${v.end.toString(16)}) prot=${v.prot}${v.shared ? ' shared' : ''}`;
      const i = this.vmaLower(addr);
      const before = this.vmas[i];
      const after = this.vmas[i] && this.vmas[i].start > addr ? this.vmas[i] : this.vmas[i + 1];
      const b = before && before.end <= addr ? `0x${before.end.toString(16)}` : 'none';
      const a = after ? `0x${after.start.toString(16)}` : 'none';
      return `UNMAPPED (brk=0x${this.brk.toString(16)}, prev vma ends ${b}, next begins ${a})`;
    });
  }

  private renderProcMaps(): string {
    return this.readAddressSpace(() => this.renderProcMapsLocked());
  }

  private renderProcMapsLocked(): string {
    let out = '';
    for (const v of this.vmas) {
      const perms = (v.prot & 0x1 ? 'r' : '-') + (v.prot & 0x2 ? 'w' : '-') + (v.prot & 0x4 ? 'x' : '-') + (v.shared ? 's' : 'p');
      // linux writes these with %08lx, which on a 64-bit machine never pads because every user
      // address is longer than eight digits. ours are shorter, so padding them would print an
      // address no program recognises: a caller looking for its own mapping formats the pointer it
      // holds with %lx and finds nothing. the address as %lx writes it is the honest form here.
      const s = v.start.toString(16);
      const e = v.end.toString(16);
      out += `${s}-${e} ${perms} 00000000 00:00 0 \n`;
    }
    return out;
  }

  /**
   * /proc/self/smaps: every mapping, with what it costs.
   *
   * the numbers are this system's truth rather than a copy of linux's shape. a mapping here is
   * backed by wasm memory that is committed when the mapping is made, so every page of it IS
   * resident -- there is no second-level paging to make Rss smaller than Size. what mlock changes
   * is whether the range may be handed back, and that is recorded exactly.
   */
  private renderProcSmaps(): string {
    return this.readAddressSpace(() => {
      let out = '';
      for (const v of this.vmas) {
        const perms = (v.prot & 0x1 ? 'r' : '-') + (v.prot & 0x2 ? 'w' : '-') +
                      (v.prot & 0x4 ? 'x' : '-') + (v.shared ? 's' : 'p');
        const s = v.start.toString(16);
        const e = v.end.toString(16);
        const kb = Math.ceil((v.end - v.start) / 1024);
        const lockedKb = Math.ceil(this.lockedBytesIn(v.start, v.end) / 1024);
        const anon = v.shared ? 0 : kb;
        out += `${s}-${e} ${perms} 00000000 00:00 0 \n`;
        out += `Size:           ${String(kb).padStart(8)} kB\n`;
        out += `KernelPageSize:        4 kB\n`;
        out += `MMUPageSize:           4 kB\n`;
        out += `Rss:            ${String(kb).padStart(8)} kB\n`;
        out += `Pss:            ${String(kb).padStart(8)} kB\n`;
        out += `Shared_Clean:          0 kB\n`;
        out += `Shared_Dirty:   ${String(v.shared ? kb : 0).padStart(8)} kB\n`;
        out += `Private_Clean:         0 kB\n`;
        out += `Private_Dirty:  ${String(anon).padStart(8)} kB\n`;
        out += `Referenced:     ${String(kb).padStart(8)} kB\n`;
        out += `Anonymous:      ${String(anon).padStart(8)} kB\n`;
        out += `LazyFree:              0 kB\n`;
        out += `AnonHugePages:         0 kB\n`;
        out += `ShmemPmdMapped:        0 kB\n`;
        out += `FilePmdMapped:         0 kB\n`;
        out += `Shared_Hugetlb:        0 kB\n`;
        out += `Private_Hugetlb:       0 kB\n`;
        out += `Swap:                  0 kB\n`;
        out += `SwapPss:               0 kB\n`;
        out += `Locked:         ${String(lockedKb).padStart(8)} kB\n`;
        out += `THPeligible:    0\n`;
        out += `VmFlags:${v.prot & 1 ? ' rd' : ''}${v.prot & 2 ? ' wr' : ''}${v.prot & 4 ? ' ex' : ''}` +
               `${v.shared ? ' sh' : ' mr mw me'}${lockedKb ? ' lo' : ''} ac\n`;
      }
      return out;
    });
  }

  /** where file handles are remembered: the kernel holds the map, so a handle taken in one process
   *  can be opened in another. */
  handleHook: { put(ino: number, path: string): void; get(ino: number): string | null } | null = null;

  /** a fresh open-file-description id, for the paths outside open() that make one. */
  newOfdIdPublic(): number { return this.newOfdId(); }

  /** the path a descriptor was opened by, for the *at() calls that accept an empty name. */
  private pathOfFd(fd: number): string | null {
    const e = this.getFdEntry(fd);
    if (e && (e as any).path) return (e as any).path as string;
    return this.vfs?.fdPath?.(fd) ?? null;
  }

  /** what the address space says about an address -- which mapping it is in, or that it is in none.
   *  the fault trace prints it, because "SIGSEGV at 0x..." alone cannot tell a guard page from a
   *  mapping this kernel forgot it made. */
  addrInfo(addr: number): string {
    return this.readAddressSpace(() => {
      const v = this.vmas.find((x) => addr >= x.start && addr < x.end);
      if (v) return `vma[0x${v.start.toString(16)},0x${v.end.toString(16)}) prot=${v.prot}${v.shared ? ' shared' : ''}`;
      const pn = this.protNone.find((r) => addr >= r.start && addr < r.end);
      if (pn) return `hole[0x${pn.start.toString(16)},0x${pn.end.toString(16)})`;
      // nothing covers it: name the neighbours, because "no mapping" one byte past the end of a
      // mapping and "no mapping" in the middle of nowhere are entirely different bugs.
      let below: Vma | undefined, above: Vma | undefined;
      for (const x of this.vmas) {
        if (x.end <= addr && (!below || x.end > below.end)) below = x;
        if (x.start > addr && (!above || x.start < above.start)) above = x;
      }
      const near = (v: Vma | undefined) => v ? `0x${v.start.toString(16)}-0x${v.end.toString(16)}` : 'none';
      return `no mapping (below=${near(below)} above=${near(above)} mmapBase=0x${this.mmapBase.toString(16)})`;
    });
  }

  /**
   * the file-size ceiling to hand the filesystem with a write, or -1 for none.
   *
   * RLIMIT_FSIZE is enforced where the offset is known, which is on the other side of the bridge:
   * a write that starts past the limit is EFBIG (and SIGXFSZ), one that crosses it is short.
   */
  private fsizeLimitArg(): number {
    const lim = this.rlimitOf(RLIMIT_FSIZE);
    if (lim.cur === RLIM_INFINITY || lim.cur > BigInt(Number.MAX_SAFE_INTEGER)) return -1;
    return Number(lim.cur);
  }

  /**
   * is [start,end) still a SHARED mapping, as the thread group's live vma table sees it?
   *
   * a worker's own list of shared mappings is a snapshot: a sibling thread that unmapped one leaves
   * every other thread believing it is still there, and if the address is later handed out again the
   * two threads disagree about what the memory IS. that disagreement is invisible until a futex on
   * it keys differently in each thread and the wake reaches nobody.
   */
  rangeIsShared(start: number, end: number): boolean {
    return this.rangeKind(start, end).shared;
  }

  /** how many bytes of [start,end) are mlocked. */
  private lockedBytesIn(start: number, end: number): number {
    let n = 0;
    for (const r of this.lockedRegions) {
      const a = Math.max(r.start, start), b = Math.min(r.end, end);
      if (b > a) n += b - a;
    }
    return n;
  }

  /** how many bytes of [start,end) are locked AND populated -- an MLOCK_ONFAULT range is neither
   *  resident nor absent by the lock alone, so it does not count here. */
  private populatedBytesIn(start: number, end: number): number {
    let n = 0;
    for (const r of this.lockedRegions) {
      if (r.onFault) continue;
      const a = Math.max(r.start, start), b = Math.min(r.end, end);
      if (b > a) n += b - a;
    }
    return n;
  }

  /**
   * the process's root directory, as a host path. chroot(2) moves it; everything below resolves
   * against it, and nothing can climb out with `..` (linux clamps the same way -- a chrooted
   * process cannot escape by walking up).
   *
   * one deliberate divergence: /proc and /dev stay synthesized after a chroot rather than
   * disappearing. on linux a chroot without those mounted has neither, but ours are the only
   * source of process information in the system and losing them costs more than the fidelity is
   * worth. every other path goes through the new root.
   */
  private rootDir = '/';

  /**
   * which of the two filesystems a path lives on. /proc/mounts advertises exactly two -- the root
   * and /proc -- and that is not a simplification: /proc really is synthesised per process while
   * everything else is one real filesystem underneath.
   */
  private mountOf(path: string): string {
    return path === '/proc' || path.startsWith('/proc/') ? '/proc' : '/';
  }

  /**
   * the /proc links that name an OBJECT rather than a path -- exe, cwd, root and the fd/ entries.
   * following one lands somewhere no amount of reading the path could have predicted, which is
   * why openat2 lets a caller refuse to be led by one.
   */
  private isMagicLink(path: string): boolean {
    const m = /^\/proc\/(self|thread-self|[0-9]+)\/(.*)$/.exec(path);
    if (!m) return false;
    return m[2] === 'exe' || m[2] === 'cwd' || m[2] === 'root' || /^fd\/[0-9]+$/.test(m[2]);
  }

  /** the working directory as the guest names it -- the vfs owns it, so it is the one to ask. */
  private currentCwd(): string {
    return this.hostToGuest(this.vfs ? this.vfs.getcwd() : this.cwd);
  }

  /**
   * where a path lands once openat2's scoping flags have had their say, or the errno they refuse
   * it with.
   *
   * RESOLVE_BENEATH and RESOLVE_IN_ROOT both say the walk may not leave the directory the
   * descriptor names. they differ in what an absolute path or a climbing ".." means: BENEATH
   * refuses one outright, IN_ROOT reinterprets it against that directory, which is what a chroot
   * would do. NO_XDEV refuses a walk that ends on a different filesystem from the one it started
   * on, and NO_MAGICLINKS refuses to be taken through a /proc link that names an object.
   * RESOLVE_NO_SYMLINKS is the one this cannot answer, because only the filesystem can see the
   * links -- it travels down to the open itself.
   */
  private scopeOpenPath(dirfd: bigint, raw: string, resolve: bigint): string | number {
    const checked = resolve & (RESOLVE_BENEATH | RESOLVE_IN_ROOT | RESOLVE_NO_XDEV | RESOLVE_NO_MAGICLINKS);
    if (!checked) return this.resolveAt(dirfd, raw);
    const d = Number(BigInt.asIntN(32, dirfd));
    const base = this.normalizePath(d === -100 ? this.currentCwd() : ((this.getFdEntry(d) as any)?.path ?? ''));
    if (!base.startsWith('/')) return EBADF;

    let full: string;
    if (resolve & RESOLVE_BENEATH) {
      if (raw.startsWith('/')) return EXDEV;
      full = this.normalizePath(base === '/' ? '/' + raw : base + '/' + raw);
      const prefix = base === '/' ? '/' : base + '/';
      if (full !== base && !full.startsWith(prefix)) return EXDEV;
    } else if (resolve & RESOLVE_IN_ROOT) {
      // the base IS the root, so an absolute path is taken from it and normalizePath's own
      // clamping at '/' does the rest -- a climb out of the root simply stays at the root.
      const rel = this.normalizePath(raw.startsWith('/') ? raw : '/' + raw);
      full = base === '/' ? rel : (rel === '/' ? base : base + rel);
    } else {
      const at = this.resolveAt(dirfd, raw);
      full = at.startsWith('/') ? this.normalizePath(at)
           : this.normalizePath(base === '/' ? '/' + at : base + '/' + at);
    }

    if ((resolve & RESOLVE_NO_XDEV) && this.mountOf(full) !== this.mountOf(base)) return EXDEV;
    if (resolve & RESOLVE_NO_MAGICLINKS) {
      // every component, not just the last: being led through one is the thing being refused.
      const parts = full.split('/').filter((c) => c.length > 0);
      let walked = '';
      for (const c of parts) {
        walked += '/' + c;
        if (this.isMagicLink(walked)) return ELOOP;
      }
    }
    return full;
  }

  /** collapse `.` and `..` and duplicate separators. the result always starts with `/`. */
  private normalizePath(p: string): string {
    const parts: string[] = [];
    for (const seg of p.split('/')) {
      if (seg === '' || seg === '.') continue;
      if (seg === '..') { parts.pop(); continue; }
      parts.push(seg);
    }
    return '/' + parts.join('/');
  }

  /** a guest-visible absolute path, as the host sees it. */
  private guestToHost(guestAbs: string): string {
    if (this.rootDir === '/') return guestAbs;
    return guestAbs === '/' ? this.rootDir : this.rootDir + guestAbs;
  }

  /** a host path, as the guest sees it. paths outside the root report as the root itself. */
  private hostToGuest(hostAbs: string): string {
    if (this.rootDir === '/') return hostAbs;
    if (hostAbs === this.rootDir) return '/';
    if (hostAbs.startsWith(this.rootDir + '/')) return hostAbs.slice(this.rootDir.length);
    return '/';
  }

  /**
   * the on-disk path a unix-socket name binds to, or null if it does not live on the real
   * filesystem. bind(AF_UNIX) creates a file there, so it needs the same resolution every other
   * path-taking call gets -- relative to the cwd, pushed under the root.
   */
  socketBindPath(path: string): string | null {
    return this.bridgePath(this.resolveAt(-100n, path));
  }

  /**
   * an empty pathname names nothing.
   *
   * every path syscall without AT_EMPTY_PATH answers ENOENT for "" -- resolving it lands on the
   * cwd instead, which then reports whatever is true of the DIRECTORY (EEXIST, EISDIR) rather
   * than the fact that the caller named no file at all.
   */
  private isEmptyPath(raw: string): boolean { return raw.length === 0; }

  /**
   * what is wrong with a pathname before anyone tries to resolve it.
   *
   * an empty name is ENOENT, a name longer than PATH_MAX is ENAMETOOLONG, and so is a single
   * COMPONENT longer than NAME_MAX -- the last of which used to be reported as whatever the
   * walk happened to hit first (usually ENOTDIR).
   */
  private pathCheck(raw: string): number {
    if (raw.length === 0) return ENOENT;
    if (raw.length > 4095) return ENAMETOOLONG;
    for (const part of raw.split('/')) if (part.length > 255) return ENAMETOOLONG;
    return 0;
  }

  /**
   * the host path a name resolves to, or null when it names something only this process has (the
   * synthesised /proc and /dev). exec asks for this so it can check the file's permissions before
   * it replaces the address space with it.
   */
  absolutePath(path: string): string | null { return this.bridgePath(path); }

  private bridgePath(path: string): string | null {
    if (!this.bridge?.isNative && !this.bridgeAvailable) return null;
    // a relative path resolves against the cwd, which is already a host path; an absolute one is
    // guest-visible and has to be pushed under the root.
    let host: string;
    if (path.startsWith('/')) {
      host = this.guestToHost(this.normalizePath(path));
    } else {
      const cwd = this.vfs?.getcwd() ?? this.cwd;
      host = this.normalizePath(cwd === '/' ? '/' + path : cwd + '/' + path);
    }
    // `..` must not climb out of the root. normalizePath already clamps at `/`, so this only bites
    // when the root is deeper than that.
    if (this.rootDir !== '/' && host !== this.rootDir && !host.startsWith(this.rootDir + '/')) {
      host = this.rootDir;
    }
    // Virtual filesystems stay in JS, and they are named by their GUEST path.
    const guest = this.hostToGuest(host);
    // /dev/shm is the exception: a posix shared-memory object is a FILE, and two processes only
    // share one if they see the same file. the synthesised /dev is per-process, so shm_open there
    // would give each of them a private copy of what is supposed to be their common ground.
    if (guest === '/dev/shm' || guest.startsWith('/dev/shm/')) return host;
    if (guest === '/proc' || guest.startsWith('/proc/') ||
        guest === '/dev' || guest.startsWith('/dev/')) {
      return null;
    }
    return host;
  }

  // resolve a (possibly relative) path to a normalized absolute path against the current cwd.
  // used for execve so the worker reads the binary from the right place (a bare "./prog" must
  // become "/cwd/prog"), matching how the kernel resolves the exec pathname.
  private absolutizePath(path: string): string {
    // the worker opens this through the same filesystem the bridge does, so it wants the HOST
    // path: under a chroot, exec("/bin/sh") has to find the root's /bin/sh, not the system one.
    if (path.startsWith('/')) return this.guestToHost(this.normalizePath(path));
    const cwd = this.vfs?.getcwd() ?? this.cwd;
    return this.normalizePath(cwd === '/' ? '/' + path : cwd + '/' + path);
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
  deliverSignal(cpu: CpuState, memory: WebAssembly.Memory, signum: number): { delivered: boolean; action?: 'exit' | 'continue' | 'stop'; exitCode?: number } {
    // no sigaction() has ever been called, so every signal is at its default disposition. that is
    // NOT the same as "ignore everything", which is what returning undelivered here amounted to: a
    // program that installs no handlers -- `head`, `true`, most of coreutils -- became immune to
    // SIGPIPE and SIGTERM alike, so `producer | head -3` ran until the producer ran out.
    const handler = this.signalHandlers?.get(signum);
    // SIGKILL and SIGSTOP cannot be caught, blocked or ignored -- not as policy, but because a
    // process that could refuse them would be unkillable and unstoppable.
    const handlerAddr = (signum === 9 || signum === 19) ? SIG_DFL : (handler?.handler ?? SIG_DFL);

    // a blocked signal stays pending whatever its disposition (SIGKILL and SIGSTOP excepted, and
    // neither reaches here). checking the mask only on the handler path meant a blocked fatal
    // default still killed the process -- so musl's raise(), which blocks everything first, was
    // lethal for any signal whose default is terminate.
    if (signum !== 9 && signum !== 19 && this.signalMask && (this.signalMask & (1n << BigInt(signum - 1)))) {
      return { delivered: false };
    }

    // SIG_IGN: ignore the signal
    if (handlerAddr === SIG_IGN) return { delivered: true, action: 'continue' };

    // SIG_DFL: default action
    if (handlerAddr === SIG_DFL) {
      if (SIG_IGNORE_DEFAULT.has(signum)) return { delivered: true, action: 'continue' };
      // SIGSTOP and friends stop the process. it keeps everything it has and resumes where it left
      // off when SIGCONT arrives; the caller parks it, because only the worker can stop running.
      if (SIG_STOP_DEFAULT.has(signum)) return { delivered: true, action: 'stop', exitCode: signum };
      // SIGCONT's default is to resume, which a running process already is. the resume itself
      // happens where the signal is SENT, since the target is not executing to act on it.
      if (signum === 18) return { delivered: true, action: 'continue' };
      // Default for most signals: terminate
      // mark this as a SIGNALLED death, not a plain exit: wait4/waitid encode it as WIFSIGNALED with
      // WTERMSIG == signum (see the 0x10000 decode there). without the marker a killed child looked
      // like it had exited normally with status 128+sig, so every "did it die from SIGSEGV/SIGBUS?"
      // check saw a normal exit instead.
      return { delivered: true, action: 'exit', exitCode: 0x10000 | (SIG_CORE_DEFAULT.has(signum) ? 0x20000 : 0) | signum };
    }

    // User handler: push signal frame and redirect execution
    const view = new DataView(memory.buffer);
    const currentSP = cpu.getSP();

    // Allocate the rt_sigframe on the guest stack (grows downward), 16-byte aligned.
    const frameSP = (currentSP - BigInt(SIGNAL_FRAME_SIZE)) & ~15n;
    const frameBase = Number(frameSP) + GUEST_MEM_BASE;
    const withAddr = SIG_WITH_ADDR.has(signum);
    // zero the siginfo + ucontext header (uc_flags/uc_link/uc_stack + siginfo padding).
    new Uint8Array(memory.buffer).fill(0, frameBase, frameBase + UC_MCONTEXT);

    // siginfo_t: si_signo, si_code, and si_addr (fault signals).
    view.setInt32(frameBase, signum, true);
    const sender = this.sigSender.get(signum);
    this.sigSender.delete(signum);
    // SI_USER(0) is what kill(2) stamps, SI_KERNEL(0x80) what the kernel does on its own account.
    // a handler tells them apart by si_code before trusting si_pid.
    view.setInt32(frameBase + SI_CODE,
      withAddr ? 2 /* SEGV_ACCERR */ : (sender ? sender.code : 0x80 /* SI_KERNEL */), true);
    if (withAddr) view.setBigUint64(frameBase + SI_ADDR, this.faultAddr, true);
    else if (sender) {
      view.setInt32(frameBase + SI_PID, sender.pid, true);
      view.setInt32(frameBase + SI_UID, sender.uid, true);
      // _sifields._rt.si_sigval, the word sigqueue(3) and a posix timer carry. it sits after the
      // pid and uid, and it is the only reason either of those interfaces exists.
      view.setBigUint64(frameBase + SI_VALUE, sender.value, true);
    }

    // ucontext.uc_mcontext: the register state the handler sees and may edit for recovery. rt_sigreturn
    // restores from here, so `uc_mcontext.pc += 4` (skip a faulting insn) takes effect.
    if (withAddr) view.setBigUint64(frameBase + UC_MCONTEXT, this.faultAddr, true); // sigcontext.fault_address
    for (let r = 0; r < 31; r++) view.setBigUint64(frameBase + MC_REGS + r * 8, cpu.getReg(r), true);
    view.setBigUint64(frameBase + MC_SP, currentSP, true);
    view.setBigUint64(frameBase + MC_PC, cpu.getPC(), true);
    view.setBigUint64(frameBase + MC_PSTATE, cpu.getNZCV(), true);
    // the mask to restore when the handler returns. after a sigsuspend that is the mask from
    // BEFORE the suspend, not the one the wait ran under.
    view.setBigUint64(frameBase + UC_SIGMASK, this.suspendRestoreMask ?? this.signalMask ?? 0n, true);
    this.suspendRestoreMask = null;
    // preserve the 32 V registers across the handler so it can't clobber in-flight vector state.
    {
      const vBase = cpu.base + V_OFFSET;
      for (let i = 0; i < 512; i += 8) view.setBigUint64(frameBase + MC_VREGS + i, view.getBigUint64(vBase + i, true), true);
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
    cpu.setReg(1, frameSP);                    // x1 = &siginfo   (SA_SIGINFO 3-arg handlers)
    cpu.setReg(2, frameSP + BigInt(UC_OFF));   // x2 = &ucontext
    cpu.setReg(30, BigInt(SIGNAL_TRAMPOLINE)); // x30 (LR) = trampoline (calls rt_sigreturn)
    this.faultAddr = 0n;                       // consumed by this delivery

    return { delivered: true, action: 'continue' };
  }

  /** Handle rt_sigreturn: restore registers from signal frame. */
  handleSigreturn(cpu: CpuState, memory: WebAssembly.Memory): bigint {
    const view = new DataView(memory.buffer);
    const frameSP = cpu.getSP();
    const frameBase = Number(frameSP) + GUEST_MEM_BASE;

    // rt_sigreturn reads a whole signal frame from wherever the stack pointer happens to point, and
    // the guest owns that pointer. A wild one must be the GUEST's problem: without this check the
    // read runs off the end of memory and traps, which kills the worker outright -- an uncatchable
    // host crash caused by ordinary guest misbehaviour. linux answers a bad frame with SIGSEGV, and
    // so should we, so the process dies and everything else keeps running.
    if (frameSP < 0n || frameBase < GUEST_MEM_BASE ||
        frameBase + SIGNAL_FRAME_SIZE > memory.buffer.byteLength) {
      this.badSigreturnSP = frameSP;
      this.raiseSignal(11); // SIGSEGV
      return cpu.getPC();
    }

    // restore register state from ucontext.uc_mcontext — reading what the handler left there, so any
    // edits (e.g. uc_mcontext.pc advanced past a faulting instruction) take effect on return.
    for (let r = 0; r < 31; r++) cpu.setReg(r, view.getBigUint64(frameBase + MC_REGS + r * 8, true));
    const savedSP = view.getBigUint64(frameBase + MC_SP, true);
    const savedPC = view.getBigUint64(frameBase + MC_PC, true);
    const savedNZCV = view.getBigUint64(frameBase + MC_PSTATE, true);
    this.signalMask = view.getBigUint64(frameBase + UC_SIGMASK, true); // restore pre-handler mask
    {
      const vBase = cpu.base + V_OFFSET;
      for (let i = 0; i < 512; i += 8) view.setBigUint64(vBase + i, view.getBigUint64(frameBase + MC_VREGS + i, true), true);
    }

    cpu.setSP(savedSP);
    cpu.setNZCV(savedNZCV);

    return savedPC; // caller should set PC to this
  }

  // who raised each pending signal, when it was a kill(2) rather than the kernel. a handler
  // installed with SA_SIGINFO reads si_pid to find out, and 0 means "the kernel" -- which for a
  // signal a process explicitly sent is simply false, and a program that replies to si_pid ends up
  // signalling process group 0 instead of the sender.
  private sigSender = new Map<number, SignalOrigin>();

  /**
   * Take a pending signal that is in `mask` (bit sig-1), WITHOUT running its handler.
   *
   * This is what sigwait/sigtimedwait do: the signal is accepted by the caller directly rather than
   * being delivered, which is why programs block the signals they intend to wait for. Returns the
   * sender too, so the caller can fill in siginfo.
   */
  takePendingIn(mask: bigint): { sig: number; pid: number; uid: number } | null {
    for (let i = 0; i < this.pendingSignals.length; i++) {
      const sig = this.pendingSignals[i];
      if (sig < 1 || sig > 64) continue;
      if (!(mask & (1n << BigInt(sig - 1)))) continue;
      this.pendingSignals.splice(i, 1);
      const from = this.sigSender.get(sig);
      this.sigSender.delete(sig);
      return { sig, pid: from?.pid ?? 0, uid: from?.uid ?? 0 };
    }
    return null;
  }

  /** Queue a signal for delivery. `from` names the sender for a kill(2)-generated signal. */
  /**
   * would delivering this signal DO anything? a signal set to ignore -- explicitly with SIG_IGN, or
   * by its own default like SIGCHLD and SIGWINCH -- is consumed without a trace.
   */
  private signalIsNoop(signum: number): boolean {
    const h = this.signalHandlers?.get(signum);
    const addr = (signum === 9 || signum === 19) ? SIG_DFL : (h?.handler ?? SIG_DFL);
    if (addr === SIG_IGN) return true;
    return addr === SIG_DFL && SIG_IGNORE_DEFAULT.has(signum);
  }

  /** is this process being traced? a tracer sees signals its tracee would otherwise discard. */
  tracedHook: (() => boolean) | null = null;

  raiseSignal(signum: number, from?: SignalOrigin): void {
    // an unblocked signal that is going to be ignored is discarded here, exactly as linux discards
    // one generated for a process that ignores it. queueing it instead meant every SIGCHLD woke the
    // parent out of wait() with EINTR for a signal that was then thrown away -- a wait that returned
    // -1 the moment a child did anything at all. (a BLOCKED one still queues: it may be waited for
    // with sigwait, or unblocked later.)
    const blocked = this.signalMask ? (this.signalMask & (1n << BigInt(signum - 1))) !== 0n : false;
    // ...unless someone is TRACING this process. a tracer is told about every signal, including the
    // ones the program would ignore -- stopping for them is the whole point, and dropping them here
    // meant a traced program ran straight past a signal its debugger was waiting for.
    if (!blocked && !this.tracedHook?.() && this.signalIsNoop(signum)) return;
    // a CLASSIC signal is not queued: a second instance of one already waiting adds nothing, and
    // linux collapses them. only the real-time signals (32 and up) stack.
    if (signum < 32 && this.pendingSignals.includes(signum)) return;
    this.pendingSignals.push(signum);
    if (from) this.sigSender.set(signum, from); else this.sigSender.delete(signum);
  }

  /**
   * raise a signal caused by the instruction that is about to re-run: SIGILL, SIGSEGV, SIGBUS,
   * SIGFPE from a fault.
   *
   * such a signal cannot be blocked or ignored into oblivion. if it were, the faulting instruction
   * would simply run again, fault again, and the process would spin at 100% forever instead of
   * dying -- which is exactly what it did. linux forces the disposition back to the default and
   * unblocks it (force_sig_info), turning an unfixable loop into the death it already is.
   */
  raiseFaultSignal(signum: number): void {
    const h = this.signalHandlers?.get(signum);
    if (h && h.handler === SIG_IGN) this.signalHandlers!.delete(signum);
    if (this.signalMask) this.signalMask &= ~(1n << BigInt(signum - 1));
    this.raiseSignal(signum);
  }

  /** this process's blocked-signal mask, for diagnostics. */
  currentMask(): bigint { return this.signalMask ?? 0n; }

  /** the caller's effective uid, for stamping si_uid on a signal it sends. */
  currentEuid(): number { return this.euid; }

  /** signal mask accessors — used by the worker's blocking rt_sigsuspend, which temporarily swaps in
   *  the caller's mask while parked and restores the original on return. */
  getSignalMask(): bigint { return this.signalMask ?? 0n; }
  setSignalMask(m: bigint): void { this.signalMask = m; }
  /**
   * the mask sigsuspend must put back, once the handler it waited for has run.
   *
   * sigsuspend replaces the mask, waits, and lets the signal be TAKEN under the replacement --
   * that is the whole point, and unblocking a signal only to re-block it before the handler could
   * run is the same as not unblocking it at all. so the frame the handler returns through carries
   * the ORIGINAL mask, and rt_sigreturn is what restores it.
   */
  private suspendRestoreMask: bigint | null = null;
  setSuspendMask(temp: bigint, saved: bigint): void {
    this.signalMask = temp;
    this.suspendRestoreMask = saved;
  }
  /** sigsuspend ended without a handler running, so the mask goes back here instead. */
  clearSuspendMask(): void {
    if (this.suspendRestoreMask !== null) { this.signalMask = this.suspendRestoreMask; this.suspendRestoreMask = null; }
  }
  /**
   * Is there a signal we could actually take RIGHT NOW?
   *
   * Not the same question as "is anything queued". A blocked signal stays queued until it is
   * unblocked -- that is what blocking means -- and treating it as an interrupt would abandon every
   * blocking call with EINTR forever while never running a handler, because the thing that would
   * clear the queue is the delivery that the mask is preventing. libc blocks signals constantly
   * (musl's raise() blocks ALL of them), so one signal arriving at the wrong moment would leave the
   * process spinning out of every wait it ever entered.
   */
  hasPendingDeliverable(): boolean { return this.hasDeliverableSignal(this.signalMask ?? 0n); }

  /** true if a queued signal is deliverable under `mask` (unblocked) — the wake condition for suspend. */
  hasDeliverableSignal(mask: bigint): boolean {
    return this.pendingSignals.some((s) => (mask & (1n << BigInt(s - 1))) === 0n);
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
  /**
   * resolve a descriptor that must name a socket.
   *
   * EBADF means "there is no such descriptor"; ENOTSOCK means "there is, and it is not a socket".
   * they are not interchangeable: a program handed the wrong fd needs to know which mistake it
   * made, and ltp checks the distinction on bind, listen, setsockopt, getsockopt and shutdown.
   */
  private sockFd(fd: number): { e: SockEntry } | { err: number } {
    // an O_PATH descriptor is not open for I/O at all, so it is EBADF rather than "not a socket".
    if (this.fdIsPath(fd)) return { err: EBADF };
    const entry = this.getFdEntry(fd);
    if (entry) return entry.type === 'socket' || entry.type === 'unixsock' ? { e: entry } : { err: ENOTSOCK };
    // the vfs keeps its own descriptors (/dev/null, /proc/..., anything synthetic), and they are
    // just as open as the ones in this table. asking only this table said EBADF about a perfectly
    // valid /dev/null -- which is exactly the descriptor ltp uses to test ENOTSOCK.
    return { err: this.fdIsOpen(fd) ? ENOTSOCK : EBADF };
  }

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
      // Full IPv6, written the way everything else writes it: the longest run of zero groups
      // collapsed to "::". spelling ::1 out as 0:0:0:0:0:0:0:1 made it a string no comparison
      // against the machine's own addresses ever matched, so binding to the loopback failed.
      const groups: number[] = [];
      for (let i = 0; i < 16; i += 2) groups.push((addr[i] << 8) | addr[i + 1]);
      let bestStart = -1, bestLen = 0;
      for (let i = 0; i < 8; i++) {
        if (groups[i] !== 0) continue;
        let j = i;
        while (j < 8 && groups[j] === 0) j++;
        if (j - i > bestLen) { bestLen = j - i; bestStart = i; }
        i = j - 1;
      }
      let host: string;
      if (bestLen > 1) {
        const head = groups.slice(0, bestStart).map((g) => g.toString(16)).join(':');
        const tail = groups.slice(bestStart + bestLen).map((g) => g.toString(16)).join(':');
        host = `${head}::${tail}`;
      } else {
        host = groups.map((g) => g.toString(16)).join(':');
      }
      return { family, host, port };
    }

    return { family, host: '0.0.0.0', port: 0 };
  }

  /** Write a sockaddr structure to guest memory. Returns bytes written. */
  /** the size a struct sockaddr of this family occupies, which is what getsockname reports. */
  private sockaddrLen(family: number): number {
    return family === 10 ? 28 : (family === 1 ? 2 : 16);
  }

  /**
   * copy an address out to a caller's (addr, addrlen) pair, the way the kernel does: at most
   * *addrlen bytes are written, and *addrlen is set to the address's FULL size so the caller can
   * tell it was truncated. accept, recvfrom and recvmsg all answer with one of these.
   */
  private putSockaddr(bytes: Uint8Array, view: DataView, addrPtr: number, lenPtr: number,
                      addr: { family: number; host: string; port: number }): number {
    if (lenPtr === 0 || this.ptrWriteFaults(bytes, lenPtr, 4)) return -EFAULT;
    const want = view.getInt32(lenPtr + GUEST_MEM_BASE, true);
    if (want < 0) return -EINVAL;
    const full = this.sockaddrLen(addr.family);
    const copy = Math.min(want, full);
    if (addrPtr === 0 || (copy > 0 && this.ptrWriteFaults(bytes, addrPtr, copy))) return -EFAULT;
    const buf = new Uint8Array(full);
    const bv = new DataView(buf.buffer);
    bv.setUint16(0, addr.family, true);
    if (addr.family === 2) {
      bv.setUint16(2, addr.port, false);
      const oct = addr.host.split('.').map((n) => Number(n) & 0xff);
      for (let i = 0; i < 4; i++) buf[4 + i] = oct[i] ?? 0;
    } else if (addr.family === 10) {
      bv.setUint16(2, addr.port, false);
      // an ipv4 address on an ipv6 socket is written the way it is carried there: v4-mapped.
      let g: number[];
      if (addr.host.includes(':')) {
        g = ipv6Groups(addr.host);
      } else {
        const o = addr.host.split('.').map((n) => Number(n) & 0xff);
        g = [0, 0, 0, 0, 0, 0xffff, ((o[0] ?? 0) << 8) | (o[1] ?? 0), ((o[2] ?? 0) << 8) | (o[3] ?? 0)];
      }
      for (let i = 0; i < 8; i++) bv.setUint16(8 + i * 2, g[i], false);
    }
    bytes.set(buf.subarray(0, copy), addrPtr + GUEST_MEM_BASE);
    view.setUint32(lenPtr + GUEST_MEM_BASE, full, true);
    return 0;
  }

  /** the interface list a net_ifaddrs reply carries. */
  private ifaceList(resp: any): Array<{ name: string; index: number; family: number; flags: number;
                                        addr: string; netmask: string; broadcast: string; mtu: number }> {
    if (!resp?.bytes) return [];
    try { return JSON.parse(new TextDecoder().decode(resp.bytes)); } catch { return []; }
  }

  /** an interface name in its fixed 16-byte field, nul-padded. */
  private putIfName(bytes: Uint8Array, addr: number, name: string): void {
    const enc = new TextEncoder().encode(name).subarray(0, 15);
    bytes.fill(0, addr + GUEST_MEM_BASE, addr + GUEST_MEM_BASE + 16);
    bytes.set(enc, addr + GUEST_MEM_BASE);
  }

  private readIfName(bytes: Uint8Array, addr: number): string {
    let end = 0;
    while (end < 16 && bytes[addr + GUEST_MEM_BASE + end] !== 0) end++;
    return new TextDecoder().decode(bytes.subarray(addr + GUEST_MEM_BASE, addr + GUEST_MEM_BASE + end));
  }

  /** a sockaddr written straight into a struct that has room for one, with no length to report. */
  private putSockaddrRaw(view: DataView, addr: number,
                         a: { family: number; host: string; port: number }): void {
    const base = addr + GUEST_MEM_BASE;
    view.setUint16(base, a.family, true);
    view.setUint16(base + 2, a.port, false);
    if (a.family === 2) {
      const oct = a.host.split('.').map((n) => Number(n) & 0xff);
      for (let i = 0; i < 4; i++) view.setUint8(base + 4 + i, oct[i] ?? 0);
    } else if (a.family === 10) {
      const g = ipv6Groups(a.host);
      for (let i = 0; i < 8; i++) view.setUint16(base + 8 + i * 2, g[i], false);
    }
  }

  /** the peer a bridge reply describes: the host side packs it as family, port and eight groups. */
  private peerFromMeta(resp: any): { family: number; host: string; port: number } | null {
    const m = resp?._meta;
    if (!m || m.length < 10) return null;
    const family = m[1] | 0;
    if (family !== 2 && family !== 10) return null;
    const g: number[] = [];
    for (let i = 0; i < 8; i++) g.push(m[2 + i] | 0);
    const host = family === 2
      ? `${(g[0] >> 8) & 0xff}.${g[0] & 0xff}.${(g[1] >> 8) & 0xff}.${g[1] & 0xff}`
      : ipv6Text(g);
    return { family, host, port: m[0] | 0 };
  }

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

  /**
   * Reserve address space for a SysV attach, and give it back on detach.
   *
   * shmat is mmap's anonymous-shared case wearing a different name: the segment's bytes live in the
   * same arena, and what it needs from the address space is identical. Sharing the mechanism keeps
   * one story about where mappings come from and what the vma list says about them -- which matters,
   * because fork's memory snapshot now reads that list.
   */
  attachShared(memory: WebAssembly.Memory, len: number, prot: number, key: string,
               want = 0): number {
    const aligned = (len + 0xFFF) & ~0xFFF;
    // shmat may name the address it wants, exactly as mmap does: the caller has already rounded it
    // to SHMLBA (or asked us to refuse an unrounded one), so what is left to check here is that the
    // space is real and nothing already lives in it.
    if (want !== 0) {
      if (want <= 0 || (want & 0xFFF) !== 0) return -EINVAL;
      if (want + aligned + GUEST_MEM_BASE > this.maxBytes) return -EINVAL;
      if (this.vmaOverlaps(want, want + aligned)) return -EINVAL;
      if (!this.growTo(memory, want + aligned + GUEST_MEM_BASE)) return -ENOMEM;
      new Uint8Array(memory.buffer).fill(0, want + GUEST_MEM_BASE, want + GUEST_MEM_BASE + aligned);
      this.vmaSet(want, want + aligned, { prot: prot & 0x7, shared: true, maxProt: 0x7 });
      try { this.protHook?.(want, aligned, prot & 0x7); } catch {}
      this.sharedRegions.push({ addr: want, len: aligned });
      this.sharedMapHook?.(want, aligned, key);
      return want;
    }
    const addr = Number(this.mmapBase);
    const needed = addr + aligned + GUEST_MEM_BASE;
    if (needed > this.maxBytes) return -ENOMEM;
    this.mmapBase += BigInt(aligned);
    if (!this.growTo(memory, needed)) return -ENOMEM;
    new Uint8Array(memory.buffer).fill(0, addr + GUEST_MEM_BASE, addr + GUEST_MEM_BASE + aligned);
    this.vmaSet(addr, addr + aligned, { prot: prot & 0x7, shared: true, maxProt: 0x7 });
    try { this.protHook?.(addr, aligned, prot & 0x7); } catch {}
    this.sharedRegions.push({ addr, len: aligned });
    this.sharedMapHook?.(addr, aligned, key);
    return addr;
  }

  /** the segment attached at this address, if any -- shmdt names an address, not an id. */
  sharedRegionAt(addr: number): { addr: number; len: number } | undefined {
    return this.sharedRegions.find((r) => r.addr === addr);
  }

  detachShared(addr: number, len: number): void {
    this.vmaSet(addr, addr + len, null);
    this.sharedRegions = this.sharedRegions.filter((r) => r.addr !== addr);
    try { this.protHook?.(addr, len, 0); } catch {}
  }

  /** Register an fd entry directly (used by Worker for kernel pipe fds). */
  /**
   * Register a descriptor created outside this handler (kernel pipes, unix sockets).
   *
   * `status` is the O_* STATUS flags of the new open file description, and passing it is not
   * optional bookkeeping: descriptor numbers are reused lowest-free-first, so a number that is
   * silently left with its previous holder's flags starts life non-blocking (or appending) for no
   * reason anyone can see. a fresh description starts with exactly the flags it was opened with.
   */
  registerFd(fd: number, entry: FdEntry, status = 0): void {
    this.fdTable.set(fd, entry);
    this.fdFlags.set(fd, status);
  }

  /** the O_* status flags of a descriptor, for dup: a duplicate SHARES the description, flags and all. */
  statusFlags(fd: number): number { return this.fdFlags.get(fd) ?? 0; }

  /**
   * the process's resource limits, in one place.
   *
   * getrlimit, setrlimit and prlimit64 are three spellings of the same state, and each used to
   * carry its own copy of the table. they drifted the moment one of them learned about a limit the
   * others did not: prlimit64 reported RLIMIT_MEMLOCK as 8 MiB while getrlimit still said
   * infinity, and ltp's getrlimit03 exists precisely to catch that disagreement.
   */
  private rlimitOf(resource: number): { cur: bigint; max: bigint } {
    const stored = this.rlimits[resource];
    if (stored) return stored;
    switch (resource) {
      case RLIMIT_STACK:   return { cur: 8388608n, max: RLIM_INFINITY };
      case RLIMIT_NOFILE:  return { cur: BigInt(this.nofileLimit), max: 1048576n };
      case RLIMIT_MEMLOCK: return { cur: BigInt(this.memlockLimit), max: RLIM_INFINITY };
      // an ordinary user gets no nice budget and no real-time budget at all: linux starts both at
      // zero, and that zero is exactly what makes an unprivileged sched_setscheduler EPERM.
      case RLIMIT_SIGPENDING: return { cur: BigInt(SIGPENDING_DEFAULT), max: BigInt(SIGPENDING_DEFAULT) };
      case RLIMIT_NICE:    return { cur: 0n, max: 0n };
      case RLIMIT_RTPRIO:  return { cur: 0n, max: 0n };
      default:             return { cur: RLIM_INFINITY, max: RLIM_INFINITY };
    }
  }

  /**
   * record a new limit, and apply it where this system enforces one.
   *
   * every resource is REMEMBERED even when nothing here acts on it: a limit a program sets and
   * reads back has to be the limit it set. storing only the two we enforce meant setrlimit
   * appeared to succeed and getrlimit then reported the default.
   */
  private setRlimit(resource: number, cur: bigint, max: bigint): void {
    this.rlimits[resource] = { cur, max };
    if (resource === RLIMIT_NOFILE) this.nofileLimit = cur >= 1048576n ? 1048576 : Number(cur);
    else if (resource === RLIMIT_MEMLOCK) this.memlockLimit = cur >= 0x7fffffffn ? 0x7fffffff : Number(cur);
    else if (resource === RLIMIT_SIGPENDING) this.publishSigpending();
  }

  /**
   * report a filesystem change to whoever is watching it.
   *
   * supplied by the worker, which owns the channel to the kernel's watch registry. it costs one
   * atomic load when nobody is watching, which is the normal case.
   */
  inotifyHook: ((path: string, mask: number, cookie: number, isDir: boolean | undefined,
                 selfOnly: boolean, probe: string, unlinked: boolean) => void) | null = null;
  /** pairs the two halves of a rename in the event stream. */
  private renameCookie = 0;
  // whether ctrl-alt-del reboots immediately, per reboot(LINUX_REBOOT_CMD_CAD_ON/OFF). there is no
  // ctrl-alt-del here, but the setting is real state a program can toggle.
  private cadEnabled = false;
  /** stop the guest system. supplied by the worker; the argument is true for a restart. */
  haltHook: ((restart: boolean) => void) | null = null;
  /**
   * report a filesystem change to whoever is watching.
   *
   * `isDir` decides IN_ISDIR, which watchers use to tell a directory apart from a file of the same
   * name. leave it undefined and the reporter stats the object to find out; pass it explicitly when
   * the object is already gone (a delete, the departing half of a rename) or when the answer is
   * free. `opts.probe` names a different path to stat for that answer, for the same reason.
   *
   * `opts.selfOnly` keeps the event off the parent directory's watch. an unlink drops the link
   * count, which is an attribute change of the file itself and no business of the directory.
   */
  private notify(path: string | null, mask: number, cookie = 0, isDir?: boolean,
                 opts?: { selfOnly?: boolean; probe?: string; unlinked?: boolean }): void {
    if (path) {
      this.inotifyHook?.(path, mask, cookie, isDir, opts?.selfOnly ?? false, opts?.probe ?? path,
                         opts?.unlinked ?? false);
    }
  }

  /** report an event about whatever file a descriptor refers to. */
  private notifyFd(fd: number, mask: number): void {
    if (!this.inotifyHook) return;
    const e = this.fdTable.get(fd);
    if (e && e.type === 'native' && e.path) {
      this.inotifyHook(e.path, mask, 0, e.isDir === true, false, e.path, e.unlinked === true);
    }
  }

  /**
   * a guest path as the host filesystem sees it, for a caller outside this handler.
   *
   * inotify watches name a path rather than a descriptor, and the watch registry lives in the
   * kernel -- which has no idea what this process's cwd or root directory are. null for /proc
   * and /dev, which are synthesised per-process and cannot be watched from outside.
   */
  resolveForWatch(path: string): string | null { return this.bridgePath(path); }

  /** the kernel-side inotify instance a descriptor names, or -1. */
  inotifyIdOf(fd: number): number {
    const e = this.fdTable.get(fd);
    return e && e.type === 'inotify' ? e.id : -1;
  }

  /** the kernel-side message queue a descriptor names, or -1. */
  mqIdOf(fd: number): number {
    const e = this.fdTable.get(fd);
    return e && e.type === 'mqueue' ? e.id : -1;
  }

  /** does any OTHER descriptor still name this queue? a close is only the last one's business. */
  mqHeldElsewhere(fd: number, id: number): boolean {
    for (const [other, e] of this.fdTable) {
      if (other !== fd && e.type === 'mqueue' && e.id === id) return true;
    }
    return false;
  }

  /** every inotify instance this process holds, so a fork can take a reference on each. */
  inotifyInstanceIds(): number[] {
    const out = new Set<number>();
    for (const e of this.fdTable.values()) if (e.type === 'inotify') out.add(e.id);
    return [...out];
  }

  /** mark a descriptor close-on-exec, for the creators that live outside this handler. */
  markCloexec(fd: number): void { this.fdCloexec.add(fd); }
  clearCloexec(fd: number): void { this.fdCloexec.delete(fd); }

  /** undo a registerFd -- a rollback, or a close handled outside the close(2) case. per-fd metadata
   *  never outlives the fd, or the next holder of the number inherits it. */
  unregisterFd(fd: number): void {
    this.fdTable.delete(fd);
    this.fdFlags.delete(fd);
    this.fdCloexec.delete(fd);
  }

  // the resolved path backing a native fd, used as the key for cross-process record locks. a
  // non-native fd (pipe, vfs) returns null: those have no on-disk identity to lock against.
  fdPath(fd: number): string | null {
    const e = this.fdTable.get(fd);
    return e && e.type === 'native' ? e.path : null;
  }

  // the open file description behind an fd, which is what owns an F_OFD_* lock and an flock. dup
  // and fork copy it (they share the description, exactly as linux does); a fresh open gets a new
  // one. ids are unique across processes because the pid is folded into them, so the kernel can
  // compare descriptions from different workers without coordinating a counter.
  fdDescriptionId(fd: number): number {
    const e = this.fdTable.get(fd);
    return e && e.type === 'native' ? e.ofdId : 0;
  }

  // how many of this process's descriptors still point at one open file description.
  countFdsForDescription(ofdId: number): number {
    let n = 0;
    for (const e of this.fdTable.values()) if (e.type === 'native' && e.ofdId === ofdId) n++;
    return n;
  }

  private ofdSeq = 0;
  private newOfdId(): number {
    return this.currentPid * 0x100000 + (++this.ofdSeq);
  }

  /** the swift-side descriptor behind a native fd, or -1. */
  fdNative(fd: number): number {
    const e = this.fdTable.get(fd);
    return e && e.type === 'native' ? e.nativeFd : -1;
  }

  /** Set process identity (used when a fork continues as the child). */
  setIdentity(pid: number, ppid: number): void {
    this.currentPid = pid;
    this.currentPpid = ppid;
  }

  // the current process credentials, threaded into NativeFS path-op bridge calls so the native
  // side can enforce emulated POSIX ownership/permissions against this process's identity.
  // (supplementary groups aren't tracked yet — setgroups is a no-op — so groups is empty for now.)
  // may this process traverse a directory with these owner/mode bits? root always may, provided
  // some execute bit is set anywhere -- matching the kernel, and InMemoryVFS.permitted.
  private mayExec(mode: number, uid: number, gid: number): boolean {
    if (this.euid === 0) return (mode & 0o111) !== 0;
    if (this.euid === uid) return (mode & 0o100) !== 0;
    if (this.egid === gid) return (mode & 0o010) !== 0;
    return (mode & 0o001) !== 0;
  }

  getCreds(): { ruid: number; euid: number; rgid: number; egid: number; umask: number; groups: number[] } {
    // the FILESYSTEM identity is what a file operation is checked against, which is what fsuid is
    // for. it equals the effective id unless setfsuid moved it, so this is the same answer as
    // before in every case except the one the pair exists to express.
    return { ruid: this.ruid, euid: this.fsuid, rgid: this.rgid, egid: this.fsgid,
             umask: this.umask, groups: [...this.groups] };
  }

  /** Get state for process snapshot */
  getState() {
    return {
      brk: this.brk,
      mmapBase: this.mmapBase,
      cwd: this.vfs ? this.vfs.getcwd() : this.cwd,
      // a child inherits its parent's root directory, and exec keeps it: that is what makes a
      // chroot stick to everything a process goes on to run.
      rootDir: this.rootDir,
      // resource limits survive both fork and exec, which is how `ulimit -n 64; ./prog` works.
      rlimits: this.rlimits.map((r) => (r ? { ...r } : undefined)),
      fdFlags: new Map(this.fdFlags),
      // WHICH descriptors are close-on-exec is per-process state that fork copies, exactly like the
      // fd table itself. leaving it out meant a forked child began with an empty set, so the exec
      // that followed closed nothing and every FD_CLOEXEC descriptor survived into the new program.
      // that is the one thing FD_CLOEXEC exists to prevent: it is how a process keeps its private
      // descriptors out of a child it execs, and a leaked one holds files open, keeps pipes from
      // ever reaching EOF, and hands the new program access it was never meant to have.
      fdCloexec: new Set(this.fdCloexec),
      fdTable: new Map(this.fdTable),
      nativeRefs: new Map(this.nativeRefs),
      currentPid: this.currentPid,
      currentPpid: this.currentPpid,
      creds: { ruid: this.ruid, euid: this.euid, suid: this.suid, rgid: this.rgid, egid: this.egid,
               sgid: this.sgid, fsuid: this.fsuid, fsgid: this.fsgid },
      caps: { eff: this.capEff, perm: this.capPerm, inh: this.capInh },
      sharedRegions: this.sharedRegions.map((r) => ({ ...r })),
      // a fork inherits the parent's ADDRESS SPACE, so it inherits the mappings that describe it.
      // leaving them behind gave every child an empty vma list: /proc/self/maps showed nothing,
      // munmap and mprotect had no record to edit, and mmap could hand back an address that was
      // already in use. it also made a fork's own memory snapshot incomplete once that snapshot
      // started asking which regions are mapped rather than sweeping the whole address space.
      vmas: this.vmas.map((v) => ({ ...v })),
      // an epoll descriptor survives fork, and so does what it is watching: linux copies the
      // interest list into the child. without it every epoll_wait in a child was EINVAL, since the
      // descriptor was there and the set behind it was not.
      epollSets: this.epollSets
        ? [...this.epollSets].map(([fd, set]) => [fd, [...set].map(([t, e]) => [t, { ...e }])] as [number, Array<[number, { events: number; data: bigint; lastReady: number }]>])
        : undefined,
      protNone: this.protNone.map((r) => ({ ...r })),
      // a forked child inherits its parent's mappings, so it inherits which parts of them the
      // file does not reach. without this the child faulted on the same page and was told it was
      // a bad pointer rather than a short file.
      noBacking: this.noBacking.map((r) => ({ ...r })),
      nofileLimit: this.nofileLimit,
      // the execution domain and what the process has pinned in memory both survive a fork.
      personality: this.personality,
      groups: [...this.groups],
      // a child inherits the timer slack its parent was using, and that value becomes the child's
      // own default -- what PR_SET_TIMERSLACK(0) puts back.
      timerSlackNs: this.timerSlackNs,
      lockedRegions: this.lockedRegions.map((r) => ({ ...r })),
      lockFuture: this.lockFuture,
      signalHandlers: this.signalHandlers ? new Map(this.signalHandlers) : undefined,
      vfsFdSnapshot: this.vfs ? this.vfs.snapshotFds() : undefined,
    };
  }

  /** Restore state from process snapshot */
  restoreState(state: { epollSets?: Array<[number, Array<[number, { events: number; data: bigint; lastReady: number }]>]>; brk: bigint; mmapBase: bigint; cwd: string; rootDir?: string; rlimits?: Array<{ cur: bigint; max: bigint } | undefined>; fdFlags: Map<number, number>; fdCloexec?: Set<number>; fdTable?: Map<number, FdEntry>; nativeRefs?: Map<number, number>; currentPid?: number; currentPpid?: number; creds?: { ruid: number; euid: number; suid: number; rgid: number; egid: number; sgid: number; fsuid?: number; fsgid?: number }; caps?: { eff: bigint; perm: bigint; inh: bigint }; sharedRegions?: Array<{ addr: number; len: number }>; vmas?: Vma[]; protNone?: Array<{ start: number; end: number }>; noBacking?: Array<{ start: number; end: number }>; nofileLimit?: number; personality?: number; groups?: number[]; timerSlackNs?: number; lockedRegions?: Array<{ start: number; end: number }>; lockFuture?: boolean; signalHandlers?: Map<number, { handler: bigint; flags: bigint; restorer: bigint; mask: bigint }>; vfsFdSnapshot?: any }): void {
    this.brk = state.brk;
    this.mmapBase = state.mmapBase;
    if (state.rootDir) this.rootDir = state.rootDir;
    if (state.rlimits) this.rlimits = state.rlimits.map((r) => (r ? { ...r } : undefined));
    this.cwd = state.cwd;
    if (state.fdCloexec) this.fdCloexec = new Set(state.fdCloexec);
    if (state.caps) { this.capEff = state.caps.eff; this.capPerm = state.caps.perm; this.capInh = state.caps.inh; }
    if (state.noBacking) this.noBacking = state.noBacking.map((r) => ({ ...r }));
    if (state.personality !== undefined) this.personality = state.personality;
    if (state.groups) this.groups = [...state.groups];
    if (state.timerSlackNs !== undefined) {
      this.timerSlackNs = state.timerSlackNs;
      this.defaultTimerSlackNs = state.timerSlackNs;
    }
    if (state.lockedRegions) {
      this.lockedRegions = state.lockedRegions.map((r) => ({ ...r }));
      let lkTotal = 0;
      for (const r of this.lockedRegions) lkTotal += r.end - r.start;
      if (this.vfs) (this.vfs as any).vmLckKb = Math.floor(lkTotal / 1024);
    }
    if (state.lockFuture !== undefined) this.lockFuture = state.lockFuture;
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
      // a fork inherits the filesystem identity too; older snapshots did not carry it, and for
      // those the effective id is the right answer because that is what it would have been.
      this.fsuid = state.creds.fsuid ?? state.creds.euid;
      this.fsgid = state.creds.fsgid ?? state.creds.egid;
    }
    if (state.sharedRegions !== undefined) this.sharedRegions = state.sharedRegions.map((r) => ({ ...r }));
    if (state.vmas !== undefined) this.vmas = state.vmas.map((v) => ({ ...v }));
    if (state.epollSets !== undefined) {
      this.epollSets = new Map(state.epollSets.map(([fd, set]) =>
        [fd, new Map(set.map(([t, e]) => [t, { ...e }]))]));
    }
    if (state.protNone !== undefined) this.protNone = state.protNone.map((r) => ({ ...r }));
    if (state.nofileLimit !== undefined) this.nofileLimit = state.nofileLimit;
    if (state.signalHandlers !== undefined) this.signalHandlers = new Map(state.signalHandlers);
    // the /proc/self/maps renderer is registered when the vma set is first EDITED. a forked child
    // inherits its mappings rather than making them, so a child that never mmapped anything had no
    // renderer at all and reported an empty address space.
    this.publishVmas();
  }

  /**
   * Close every native (Swift-backed) fd this process still holds. real kernels close a
   * process's fds when it exits; jitter never did, so a Worker that runs many pooled tenants
   * leaks Swift file handles until the device fd budget is exhausted and NativeFS starts failing
   * opens. call this on process exit / before pool reuse. stdio and non-native fds are ignored.
   */
  /** close one descriptor, for the callers that cannot return a bridge result -- close_range
   *  closes a whole span in a single syscall, and there is only ever one reply to give. */
  private closeOne(fd: number): void {
    const e = this.fdTable.get(fd);
    this.fdFlags.delete(fd);
    this.fdCloexec.delete(fd);
    if (!e) { if (this.vfs?.hasFd(fd)) this.vfs.close(fd); return; }
    this.fdTable.delete(fd);
    switch (e.type) {
      case 'pipe_write': e.pipe.writeClosed = true; break;
      case 'pipe_read': e.pipe.readClosed = true; break;
      case 'vfs': this.vfs?.close(e.fd); break;
      case 'epoll': this.epollSets?.delete(fd); break;
      case 'native': {
        const refs = (this.nativeRefs.get(e.nativeFd) ?? 1) - 1;
        if (refs <= 0) { this.nativeRefs.delete(e.nativeFd); this.closeNativeHook?.(e.nativeFd); }
        else this.nativeRefs.set(e.nativeFd, refs);
        break;
      }
      case 'socket': this.closeSocketHook?.(e.socketId); break;
      case 'unixsock': this.closeUnixHook?.(fd); break;
      default: break;
    }
  }

  closeNativeFds(closeFn: (nativeFd: number) => void): void {
    // each underlying descriptor once, and each only ever once: two guest fds can share one native
    // fd (that is what dup does), and a second close of a number swift has already handed back
    // would land on whatever it was reassigned to. dropping the entries as we go is what makes a
    // second call -- the exit-path backstop -- a no-op rather than a hazard.
    const seen = new Set<number>();
    for (const [guestFd, entry] of [...this.fdTable]) {
      if (entry.type !== 'native') continue;
      if (!seen.has(entry.nativeFd)) { seen.add(entry.nativeFd); closeFn(entry.nativeFd); }
      this.fdTable.delete(guestFd);
    }
  }

  /** descriptors marked FD_CLOEXEC, for owners of fd kinds this handler does not itself track. */
  cloexecFds(): number[] {
    return [...this.fdCloexec];
  }

  resetForExec(brkStart: bigint, closeFn?: (nativeFd: number) => void): void {
    this.brk = brkStart;
    this.brkFloor = brkStart;
    this.mmapBase = 0x2100000n;
    this.mmapHint = 0x2100000;
    // exec replaces the address space: every mapping and page protection of the OLD image is gone.
    // keeping them left the new image inheriting stale PROT_NONE/read-only ranges and phantom vmas
    // (a stale RELRO range from the previous program faulted the new one's relocation writes).
    this.protNone = [];
    this.vmas = [];
    // exec ends every other thread, so the group's shared address space ends with them: the new
    // image gets a private one again, and must not keep writing into the old group's table.
    this.vmaTable = null;
    this.vmaGen = 0;
    this.vmaDepth = 0;
    this.sharedRegions = [];
    this.fileMmaps = [];
    this.noBacking = [];
    this.lockedRegions = [];
    // exec replaces the code, so every handler ADDRESS in the old image points at whatever now
    // happens to live there. posix resets every caught signal to its default across exec for
    // exactly that reason; a signal set to SIG_IGN stays ignored, because that is a decision about
    // the signal rather than a pointer into the program. without this, a shell's SIGCHLD handler
    // was inherited by everything it ran, and the first SIGCHLD after that jumped the process into
    // the middle of its own heap -- which then faulted forever instead of dying, because a blocked
    // fault signal was re-queued rather than forced.
    if (this.signalHandlers) {
      for (const [sig, h] of [...this.signalHandlers]) {
        if (h.handler !== SIG_IGN) this.signalHandlers.delete(sig);
      }
    }
    // exec ends every other thread of the group, so the group's shared table -- which lives in a
    // shared memory this image is about to stop using -- is no longer ours to read or write.
    this.sigTable = undefined;
    this.sigTableSeen = -1;
    // the pending queue was raised against a program that no longer exists.
    this.pendingSignals = [];
    // exec closes exactly the descriptors marked FD_CLOEXEC; every other fd survives into the new
    // image (this is how programs pass fds to an exec'd child). native fds we drop must be
    // fs_close'd too, or exec leaks the Swift file handle behind them.
    for (const fd of [...this.fdCloexec]) {
      const entry = this.fdTable.get(fd);
      if (entry) {
        if (entry.type === 'native' && closeFn) closeFn(entry.nativeFd);
        if (entry.type === 'eventfd') this.eventfdHook?.({ op: 'close', id: entry.id, pid: this.currentPid });
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

  /** the lowest free descriptor at or above `min`, or -1 when RLIMIT_NOFILE leaves none. */
  allocLowestFd(min = 0): number {
    const fd = this.lowestFreeFd(min);
    return fd < this.nofileLimit ? fd : -1;
  }

  /** can `n` more descriptors be allocated within RLIMIT_NOFILE? pipe() needs two at once. */
  fdRoomFor(n: number): boolean {
    let free = 0;
    for (let fd = 0; fd < this.nofileLimit && free < n; fd++) {
      if (!this.fdTable.has(fd) && !(this.vfs?.hasFd(fd) ?? false)) free++;
    }
    return free >= n;
  }

  /** true if a new fd (>= min) can be allocated within RLIMIT_NOFILE. */
  private fdAvailable(min = 3): boolean {
    return this.lowestFreeFd(min) < this.nofileLimit;
  }

  /** Resolve fd: check fdTable for overrides (dup2'd pipes, etc.), else fall through to default */
  getFdEntry(fd: number): FdEntry | undefined {
    return this.fdTable.get(fd);
  }

  private isNonBlocking(fd: number): boolean {
    return ((this.fdFlags.get(fd) ?? 0) & O_NONBLOCK) !== 0;
  }

  /** a descriptor's open status flags, for the worker-side handlers that need them. */
  fdStatusFlags(fd: number): number { return this.fdFlags.get(fd) ?? 0; }

  /** public view of a fd's O_NONBLOCK flag (the worker needs it for kernel-pipe write blocking). */
  isNonBlockingFd(fd: number): boolean { return this.isNonBlocking(fd); }
  /** turn O_NONBLOCK on or off for a descriptor, for the callers that live outside this handler. */
  setFdNonBlocking(fd: number, on: boolean): void {
    const cur = this.fdFlags.get(fd) ?? 0;
    this.fdFlags.set(fd, on ? (cur | O_NONBLOCK) : (cur & ~O_NONBLOCK));
  }

  /**
   * would touching this guest range fault? bounds AND the process's PROT_NONE mappings, which is
   * how a caller distinguishes "you handed me an unmapped address" (EFAULT) from every other
   * complaint. worker-side syscall handlers need it too: bounds alone miss a PROT_NONE page,
   * which is exactly what a test reaches for when it wants a guaranteed-bad pointer.
   */
  ptrWouldFault(bytes: Uint8Array, addr: number, len = 1): boolean {
    return this.ptrFaults(bytes, addr, len);
  }

  /**
   * O_PATH opens a descriptor that NAMES a file without granting access to its contents: it is
   * for *at() calls and fstat, and reading or writing through it is EBADF. it survives dup, so
   * the duplicate is just as inert.
   */
  fdIsPath(fd: number): boolean {
    return ((this.fdFlags.get(fd) ?? 0) & 0o10000000) !== 0;
  }

  /**
   * the process a descriptor names, for the calls that take a pidfd: one made by pidfd_open, or an
   * open /proc/<pid> directory, which linux accepts wherever a pidfd is wanted. -1 for anything
   * else, which is EBADF to the caller.
   */
  pidfdTarget(fd: number): number {
    const e = this.getFdEntry(fd);
    if (e?.type === 'pidfd') return e.pid;
    // a /proc descriptor may not be in the fd table at all: the synthesised filesystem keeps its
    // own, and only the ones it shares with the rest of the system appear in both.
    if (e !== undefined && e.type !== 'vfs') return -1;
    const path = (this.vfs as any)?.fdPath?.(fd) ?? null;
    const m = typeof path === 'string' ? path.match(/^\/proc\/(self|\d+)\/?$/) : null;
    if (!m) return -1;
    return m[1] === 'self' ? (this.tgid || this.currentPid) : Number(m[1]);
  }

  /** is a signal this descriptor reports pending right now? */
  signalfdReady(fd: number): boolean {
    const e = this.getFdEntry(fd);
    if (e?.type !== 'signalfd') return false;
    return this.pendingSignals.some((sig) => (e.mask & (1n << BigInt(sig - 1))) !== 0n);
  }

  /**
   * read(2) on a signalfd: hand back one struct signalfd_siginfo per signal in the descriptor's
   * mask that is pending, CONSUMING it -- a signal read this way is not delivered again.
   *
   * returns the byte count, -EINVAL for a buffer too small for one record, or -EAGAIN when
   * nothing matches (the caller decides whether to wait).
   */
  signalfdRead(fd: number, buf: number, count: number, view: DataView, bytes: Uint8Array): number {
    const e = this.getFdEntry(fd);
    if (e?.type !== 'signalfd') return -EBADF;
    const SIGINFO_SIZE = 128;
    if (count < SIGINFO_SIZE) return -EINVAL;
    if (this.ptrWriteFaults(bytes, buf, Math.min(count, SIGINFO_SIZE))) return -EFAULT;
    let written = 0;
    while (written + SIGINFO_SIZE <= count) {
      const idx = this.pendingSignals.findIndex((sig) => (e.mask & (1n << BigInt(sig - 1))) !== 0n);
      if (idx < 0) break;
      const sig = this.pendingSignals[idx];
      this.pendingSignals.splice(idx, 1);
      const from = this.sigSender.get(sig);
      this.sigSender.delete(sig);
      const off = buf + written + GUEST_MEM_BASE;
      bytes.fill(0, off, off + SIGINFO_SIZE);
      view.setUint32(off + 0, sig, true);          // ssi_signo
      view.setInt32(off + 4, 0, true);             // ssi_errno
      // SI_USER (0) when a process sent it, SI_KERNEL (0x80) otherwise -- the same distinction
      // a handler reads out of si_code.
      view.setInt32(off + 8, from ? 0 : 0x80, true);
      view.setUint32(off + 12, from?.pid ?? 0, true); // ssi_pid
      view.setUint32(off + 16, from?.uid ?? 0, true); // ssi_uid
      written += SIGINFO_SIZE;
    }
    return written === 0 ? -EAGAIN : written;
  }

  /** does this descriptor name a directory? a relative *at() path can only resolve against one. */
  fdIsDir(fd: number): boolean {
    const entry = this.getFdEntry(fd);
    if (entry) return entry.type === 'native' && entry.isDir === true;
    return this.vfs?.fdIsDir(fd) ?? false;
  }

  /** is this descriptor open at all, by any of the kinds this handler tracks? */
  fdIsOpen(fd: number): boolean {
    return this.fdTable.has(fd) || (this.vfs?.hasFd(fd) ?? false);
  }

  /**
   * what /proc/self/fd should show: every open descriptor and what it names. the targets are the
   * strings linux uses -- a real path for a file, "pipe:[n]" / "socket:[n]" / "anon_inode:[kind]"
   * for the things that have no path -- because programs read them and match on exactly those.
   */
  /**
   * write bytes to a descriptor that this handler owns end to end, for a caller outside write(2).
   * the descriptors the WORKER owns (native files, pipes, unix sockets) are not here: it writes to
   * those itself. returns bytes written, or a negative errno.
   */
  writeOutOfBand(fd: number, data: Uint8Array): number {
    const e = this.fdTable.get(fd);
    if (!e) return -EBADF;
    switch (e.type) {
      case 'stdio':
        if (e.fd === 1) this.io.writeStdout(data);
        else if (e.fd === 2) this.io.writeStderr(data);
        else return -EBADF;
        return data.length;
      case 'tty':
        this.io.writeStdout(data);
        return data.length;
      // /proc and /dev are synthesised rather than stored, so there is nothing to append bytes to.
      case 'vfs':
        return -EINVAL;
      case 'devnode': {
        const kind = devKind(e.major, e.minor);
        // /dev/null and /dev/zero swallow whatever they are given; anything else here has no
        // sink to write into.
        return kind === 'null' || kind === 'zero' ? data.length : -EINVAL;
      }
      default:
        return -EINVAL;
    }
  }

  /** the inotify watch lines for a descriptor's fdinfo; supplied by the worker, which can reach
   *  the kernel's watch registry. */
  inotifyFdinfo: ((id: number) => string) | null = null;
  /** where a native descriptor currently sits; the offset lives on the host side of the bridge. */
  fdPosition: ((fd: number) => number) | null = null;

  openFdList(): Array<{ fd: number; target: string; info: string }> {
    const out: Array<{ fd: number; target: string; info: string }> = [];
    for (const [fd, e] of this.fdTable) {
      let target: string;
      switch (e.type) {
        case 'native':      target = e.path; break;
        case 'vfs':         target = this.vfs?.fdPath(e.fd) ?? `anon_inode:[vfs]`; break;
        case 'stdio':       target = '/dev/console'; break;
        case 'tty':         target = '/dev/tty'; break;
        case 'jitterctl':   target = '/dev/jitter'; break;
        case 'pipe_read':
        case 'pipe_write':  target = `pipe:[${fd}]`; break;
        case 'socket':      target = `socket:[${e.socketId}]`; break;
        case 'unixsock':    target = `socket:[${fd}]`; break;
        case 'inotify':     target = 'anon_inode:inotify'; break;
        case 'timerfd':     target = 'anon_inode:[timerfd]'; break;
        case 'eventfd':     target = 'anon_inode:[eventfd]'; break;
        case 'epoll':       target = 'anon_inode:[eventpoll]'; break;
        case 'signalfd':    target = 'anon_inode:[signalfd]'; break;
        case 'mqueue':      target = `/dev/mqueue/${e.name}`; break;
        case 'devnode':     target = `/dev/${e.major}:${e.minor}`; break;
        case 'pidfd':       target = 'anon_inode:[pidfd]'; break;
      }
      // fdinfo is the descriptor's own state rather than the file's: where it is, how it was opened,
      // and whatever the thing behind it wants to say about itself.
      let info = `pos:\t${e.type === 'native' ? this.fdPosition?.(fd) ?? 0 : 0}\n`
               + `flags:\t0${((this.fdFlags.get(fd) ?? 0) >>> 0).toString(8)}\n`
               + `mnt_id:\t1\n`;
      if (e.type === 'inotify') info += this.inotifyFdinfo?.(e.id) ?? '';
      // which process a pidfd refers to. it is the only way to ask, since the descriptor carries
      // no path and the pid it names may already be gone.
      if (e.type === 'pidfd') info += `Pid:\t${e.pid}\nNSpid:\t${e.pid}\n`;
      out.push({ fd, target, info });
    }
    if (this.vfs) {
      for (const fd of this.vfs.openFdNumbers()) {
        if (!this.fdTable.has(fd)) {
          out.push({ fd, target: this.vfs.fdPath(fd) ?? 'anon_inode:[vfs]', info: 'pos:\t0\nflags:\t0100000\nmnt_id:\t1\n' });
        }
      }
    }
    out.sort((a, b) => a.fd - b.fd);
    return out;
  }

  /**
   * whether the descriptor was opened for reading / writing. record locks are gated on this:
   * posix says a shared lock needs a descriptor open for reading and an exclusive lock one open
   * for writing, and answers EBADF otherwise -- which is what makes fcntl(1, F_SETLK, F_RDLCK)
   * fail on a write-only stdout.
   */
  /** the descriptor's async-notification state: who to signal, with what, and whether O_ASYNC is
   *  armed at all. O_ASYNC is 0o20000 on aarch64 (asm-generic FASYNC). */
  asyncState(fd: number): { pid: number; sig: number; armed: boolean } {
    const o = this.fdOwner.get(fd);
    return {
      pid: o?.pid ?? 0,
      sig: this.fdSig.get(fd) ?? 0,
      armed: ((this.fdFlags.get(fd) ?? 0) & 0o20000) !== 0 && (o?.pid ?? 0) !== 0,
    };
  }

  /** is this process running as root? */
  isRoot(): boolean { return this.euid === 0; }

  /** does this process hold a capability in its EFFECTIVE set? this, not the uid, is what a
   *  privileged operation should ask. */
  hasCap(cap: number): boolean { return (this.capEff & (1n << BigInt(cap))) !== 0n; }

  /**
   * recompute the capability sets after a uid change, the way linux's cap_emulate_setxuid does.
   *
   * capabilities and uid 0 are not the same thing, but they are tied: a process that gives up root
   * gives up the powers root came with, and one that returns to root gets back only what it still
   * has PERMITTED. only setuid(2) did any of this, so a process that dropped privilege with
   * setresuid or setreuid kept every capability it had -- and then chroot, mknod and the rest all
   * succeeded for a caller that was no longer root.
   */
  private recomputeCapsAfterUid(prevRuid: number, prevEuid: number, prevSuid: number): void {
    const wasRootAnywhere = prevRuid === 0 || prevEuid === 0 || prevSuid === 0;
    const isRootAnywhere = this.ruid === 0 || this.euid === 0 || this.suid === 0;
    // giving up the last trace of root gives up the capabilities permanently
    if (wasRootAnywhere && !isRootAnywhere) {
      this.capPerm = 0n;
      this.capEff = 0n;
      return;
    }
    // and the effective set simply follows the effective uid
    if (prevEuid === 0 && this.euid !== 0) this.capEff = 0n;
    else if (prevEuid !== 0 && this.euid === 0) this.capEff = this.capPerm;
  }

  fdAccess(fd: number): { readable: boolean; writable: boolean } {
    // an access mode is a property of how a FILE was opened. most descriptors in this table were
    // never opened at all -- they were created, by socket() or eventfd() or by naming a device --
    // and asking them for their O_RDONLY/O_WRONLY bit gets the default, zero, which reads as
    // "opened read-only" and refuses every write with EBADF. that is how `jitter ls` came to fail
    // with "write: Bad file descriptor" on a control device it had just opened for writing.
    // so: descriptors with a real open mode are judged by it, and descriptors without one are
    // judged by what they ARE.
    const entry = this.fdTable.get(fd);
    switch (entry?.type) {
      // bidirectional by construction: nothing about them is one-way.
      case 'tty': case 'jitterctl': case 'socket': case 'unixsock': case 'eventfd':
        return { readable: true, writable: true };
      // a pipe end goes exactly one way, whatever flags happen to be recorded for it.
      case 'pipe_read': return { readable: true, writable: false };
      case 'pipe_write': return { readable: false, writable: true };
      // event queues you can only consume from.
      case 'signalfd':
      case 'inotify': case 'timerfd': case 'epoll':
        return { readable: true, writable: false };
      default: break; // stdio, vfs and native files carry a genuine open mode
    }
    const mode = (this.fdFlags.get(fd) ?? 0) & 0x3; // O_RDONLY / O_WRONLY / O_RDWR
    return { readable: mode === 0 || mode === 2, writable: mode === 1 || mode === 2 };
  }

  // a user pointer faults if it is out of the guest-addressable region (badGuestPtr) or lands
  // in a range that was mapped PROT_NONE.
  ptrFaults(bytes: Uint8Array, addr: number, len = 1): boolean {
    if (badGuestPtr(bytes, addr, len)) return true;
    for (const r of this.protNone) {
      if (addr < r.end && addr + len > r.start) return true;
    }
    return false;
  }

  /**
   * The drivers behind the device nodes we back. These are linux's semantics, not approximations:
   * a program that reads /dev/zero for a megabyte of zeros, or writes a log to /dev/null, or fills
   * a disk-full test against /dev/full, is relying on exactly this.
   */
  private deviceRead(major: number, minor: number, bytes: Uint8Array, buf: number, count: number): number {
    switch (devKind(major, minor)) {
      case 'null':   return 0;                       // always at end of file
      case 'zero':
      case 'full':   bytes.fill(0, buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count); return count;
      case 'random': {
        const dst = bytes.subarray(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
        crypto.getRandomValues(dst);
        return count;
      }
      case 'tty': {
        const d = this.io.readStdin(count);
        if (!d || d.length === 0) return 0;
        bytes.set(d, buf + GUEST_MEM_BASE);
        return d.length;
      }
      default:       return -6; // ENXIO
    }
  }

  private deviceWrite(major: number, minor: number, bytes: Uint8Array, buf: number, count: number): number {
    switch (devKind(major, minor)) {
      case 'null':   return count;                   // swallowed
      case 'zero':   return count;                   // linux accepts and discards
      case 'full':   return -28;                     // ENOSPC -- that is what /dev/full is FOR
      case 'random': return count;                   // writing seeds the pool; accepting is enough
      case 'tty':
        this.io.writeStdout(bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count));
        return count;
      default:       return -6; // ENXIO
    }
  }

  /**
   * the checks pread/pwrite/preadv/pwritev all share, in linux's order.
   *
   * Returns a negative errno, or 0 to proceed. The positional calls are defined only on seekable
   * descriptors: a pipe, socket or fifo is ESPIPE, not a short read. And the OFFSET is checked
   * before the buffer, so a negative offset is EINVAL even when the buffer is also bad.
   */
  private positionalIoCheck(fd: number, offset: bigint, forWrite: boolean): number {
    // a kernel pipe is not seekable, and it is not in the fd table either
    if (this.io.pipeReady?.(fd)) return -ESPIPE;
    if (!this.fdExists(fd)) return -EBADF;
    const acc = this.fdAccess(fd);
    if (forWrite ? !acc.writable : !acc.readable) return -EBADF;
    const entry = this.fdTable.get(fd);
    switch (entry?.type) {
      case 'pipe_read': case 'pipe_write': case 'socket': case 'unixsock':
      case 'tty': case 'stdio': case 'eventfd': case 'epoll':
      case 'inotify': case 'timerfd': case 'jitterctl': case 'signalfd': case 'mqueue':
        return -ESPIPE;
      default: break;
    }
    // O_PATH descriptors are not open for I/O at all
    if (this.fdIsPath(fd)) return -EBADF;
    // a directory can be opened and seeked, but its contents are not a byte stream: reading one is
    // EISDIR, which is a different answer from "that descriptor is not readable".
    if (entry?.type === 'native' && entry.isDir) return -EISDIR;
    if (offset < 0n) return -EINVAL;
    return 0;
  }

  /**
   * total length of an iovec array, or a negative errno.
   *
   * iovcnt is bounded by IOV_MAX (1024 on linux) and each length is a size_t that the kernel reads
   * as signed: a negative one is EINVAL, not an enormous transfer. the sum overflowing is EINVAL too.
   */
  private iovTotal(view: DataView, iov: number, iovcnt: number): number {
    if (iovcnt < 0 || iovcnt > IOV_MAX) return -EINVAL;
    let total = 0;
    for (let i = 0; i < iovcnt; i++) {
      const len = Number(view.getBigInt64(iov + i * 16 + 8 + GUEST_MEM_BASE, true));
      if (len < 0) return -EINVAL;
      total += len;
      if (total > 0x7fffffff) return -EINVAL;
    }
    return total;
  }

  /**
   * whether this descriptor number refers to anything at all.
   *
   * the guest fd namespace is split: pipes/sockets/native files live in fdTable, vfs-backed files
   * live in the vfs. a check against only one of them calls half the open descriptors in the
   * process closed.
   */
  private fdExists(fd: number): boolean {
    return this.fdTable.has(fd) || (this.vfs?.hasFd(fd) ?? false) || !!this.io.pipeReady?.(fd);
  }

  /**
   * whether epoll (and poll, and select) can meaningfully wait on this descriptor.
   *
   * linux answers EPERM for the ones it cannot: a regular file and a directory are always ready,
   * so there is nothing to wait for, and letting a caller add one would turn its event loop into
   * a spin. everything with a readiness state -- pipes, sockets, ttys, eventfds, timerfds, other
   * epoll instances -- is fair game.
   */
  private fdPollable(fd: number): boolean {
    if (this.io.pipeReady?.(fd)) return true;
    const entry = this.fdTable.get(fd);
    if (!entry) return false; // vfs-backed descriptors are plain files
    switch (entry.type) {
      case 'native': case 'vfs': return false;
      default: return true;
    }
  }

  /** Whether the fd has data available for read without blocking. */
  private isFdReadable(fd: number): boolean {
    const pr = this.io.pipeReady?.(fd);
    if (pr) return pr.readable;
    const entry = this.fdTable.get(fd);
    if (!entry) return false;
    switch (entry.type) {
      case 'stdio': return entry.fd === 0 ? (this.io.stdinHasData?.() ?? true) : true;
      case 'tty': return this.io.stdinHasData?.() ?? true;
      case 'pipe_read': return entry.pipe.data.length > 0 || entry.pipe.writeClosed;
      case 'pipe_write': return false;
      case 'vfs': case 'native': return true;
      case 'socket': return false; // handled via bridge poll
      case 'eventfd': return this.efdCount(entry.id) > 0n;
      // an inotify descriptor is readable exactly when an event is waiting for it, and the queue
      // lives in the kernel -- so poll, select and epoll have to ask.
      case 'inotify': return (this.io.inotifyPending?.(fd) ?? 0) > 0;
      // a timerfd is readable exactly when it has expirations nobody has collected, and that
      // is a question about the clock rather than about any event we were told of.
      case 'timerfd': { this.checkTimers(); return (this.timerFds.get(entry.id)?.ticks ?? 0n) > 0n; }
      case 'epoll': return false;
      // a pidfd reports readable exactly once its process has exited, which is how poll waits for
      // a process without being its parent.
      case 'pidfd': return this.io.pidExited?.(entry.pid) ?? false;
      case 'jitterctl': return entry.resp !== null; // readable once the host has answered
      // a listening socket is readable when a connection is queued; a connected one when its rx
      // ring has bytes or the peer is gone. answering "always" instead is a lie with teeth: the
      // caller believes there is something there and issues a blocking accept or read that never
      // returns. libuv asserts and aborts on it, which is how neovim died.
      case 'unixsock': return this.io.unixReady?.(fd).readable ?? false;
      case 'signalfd': return this.signalfdReady(fd);
      case 'mqueue': return (this.io.mqReady?.(fd)?.readable ?? false);
      // /dev/null, /dev/zero and friends are always ready; a tty node follows the terminal.
      case 'devnode': return devKind(entry.major, entry.minor) === 'tty'
        ? (this.io.stdinHasData?.() ?? true) : true;
    }
  }

  /** Whether the fd can accept a write without blocking. */
  private isFdWritable(fd: number): boolean {
    const pw = this.io.pipeReady?.(fd);
    if (pw) return pw.writable;
    const entry = this.fdTable.get(fd);
    if (!entry) return false;
    switch (entry.type) {
      case 'stdio': return entry.fd === 1 || entry.fd === 2;
      case 'tty': return true;
      case 'pipe_write': return !entry.pipe.readClosed;
      case 'pipe_read': return false;
      case 'vfs': case 'native': return true;
      case 'socket': return true; // connected TCP sockets accept writes (net_write queues via NWConnection)
      case 'eventfd': return this.efdCount(entry.id) < 0xFFFFFFFFFFFFFFFEn;
      case 'inotify': case 'timerfd': case 'epoll': case 'signalfd': case 'pidfd': return false;
      case 'jitterctl': return true; // the request side always accepts a write
      case 'unixsock': return this.io.unixReady?.(fd).writable ?? false;
      case 'mqueue': return (this.io.mqReady?.(fd)?.writable ?? false);
      case 'devnode': return true; // every device we back accepts a write (/dev/full fails it, but not by blocking)
    }
  }

  /**
   * epoll_pwait and epoll_pwait2, which differ only in how the timeout is spelled.
   *
   * struct epoll_event is `{ uint32_t events; uint64_t data; }`, and it is packed ONLY on x86_64
   * (__EPOLL_PACKED is defined for that arch and nothing else). On aarch64 it takes its natural
   * layout: 16 bytes, with data at offset 8. Writing the x86 layout put every data pointer four
   * bytes early and made every event the caller read back garbage -- which is most of what the
   * epoll tests were failing on.
   *
   * Returns the number of events, or a negative errno.
   */
  /**
   * may the guest WRITE this range? a mapping it can only read is as much a fault as none at all,
   * and a syscall that fills a buffer has to say so before filling it.
   */
  ptrWriteFaults(bytes: Uint8Array, addr: number, len: number): boolean {
    if (this.ptrFaults(bytes, addr, len)) return true;
    for (const v of this.vmas) {
      if (addr < v.end && addr + len > v.start && !(v.prot & 2)) return true;
    }
    return false;
  }

  /** how many epoll instances deep a chain rooted here goes; a non-epoll descriptor is zero. */
  private epollDepth(fd: number, seen = new Set<number>()): number {
    const set = this.epollSets?.get(fd);
    if (!set || seen.has(fd)) return 0;
    seen.add(fd);
    let d = 1;
    for (const child of set.keys()) d = Math.max(d, 1 + this.epollDepth(child, seen));
    seen.delete(fd);
    return d;
  }

  /** can `target` be reached from `from` by following epoll-watches-epoll edges? */
  private epollReaches(from: number, target: number, seen = new Set<number>()): boolean {
    if (from === target) return true;
    const set = this.epollSets?.get(from);
    if (!set || seen.has(from)) return false;
    seen.add(from);
    for (const child of set.keys()) if (this.epollReaches(child, target, seen)) return true;
    return false;
  }

  private epollWait(view: DataView, bytes: Uint8Array, epfd: number, eventsPtr: number,
                    maxEvents: number, timeoutMs: number): number {
    if (!this.fdExists(epfd)) return -EBADF;
    if (!this.epollSets?.has(epfd)) return -EINVAL;
    // maxevents must be positive: zero would be a wait that can never report anything.
    if (maxEvents <= 0) return -EINVAL;
    if (this.ptrWriteFaults(bytes, eventsPtr, maxEvents * EPOLL_EVENT_SIZE)) return -EFAULT;
    const eset = this.epollSets.get(epfd)!;

    const EPOLLIN = 0x001, EPOLLOUT = 0x004, EPOLLHUP = 0x010, EPOLLRDHUP = 0x2000;
    const EPOLLONESHOT = 0x40000000, EPOLLET = 0x80000000;
    const gather = (): number => {
      let n = 0;
      for (const [targetFd, entry] of eset) {
        if (n >= maxEvents) break;
        let readyEvents = 0;
        if ((entry.events & EPOLLIN) && this.isFdReadable(targetFd)) readyEvents |= EPOLLIN;
        if ((entry.events & EPOLLOUT) && this.isFdWritable(targetFd)) readyEvents |= EPOLLOUT;
        // EPOLLHUP and EPOLLERR are reported whether or not the caller asked for them.
        const fdEntry = this.fdTable.get(targetFd);
        if (fdEntry?.type === 'pipe_read' && fdEntry.pipe.writeClosed) readyEvents |= EPOLLHUP;
        if (fdEntry?.type === 'pipe_write' && fdEntry.pipe.readClosed) readyEvents |= EPOLLHUP;
        // a socket whose receiving half is down will never deliver another byte, and that is what
        // EPOLLRDHUP reports. it is a maskable event, so it is only reported when asked for; a
        // socket shut down both ways is a hangup, which is reported either way.
        if (fdEntry?.type === 'socket') {
          if ((entry.events & EPOLLRDHUP) && fdEntry.shutRd) readyEvents |= EPOLLRDHUP;
          if (fdEntry.shutRd && fdEntry.shutWr) readyEvents |= EPOLLHUP;
        }
        // edge-triggered: the caller is told when readiness ARRIVES, not for as long as it lasts.
        // a descriptor that was already readable last time it looked is not news, which is the
        // whole point -- an EPOLLET loop drains a descriptor and expects silence until more comes.
        const report = (entry.events & EPOLLET) ? (readyEvents & ~entry.lastReady) : readyEvents;
        entry.lastReady = readyEvents;
        if (!report) continue;
        const off = eventsPtr + n * EPOLL_EVENT_SIZE + GUEST_MEM_BASE;
        view.setUint32(off, report, true);
        view.setBigUint64(off + EPOLL_DATA_OFFSET, entry.data, true);
        // EPOLLONESHOT fires once and then goes quiet: the descriptor stays in the interest list
        // but is disarmed until EPOLL_CTL_MOD arms it again.
        if (entry.events & EPOLLONESHOT) entry.events &= ~(EPOLLIN | EPOLLOUT);
        n++;
      }
      return n;
    };

    // wait when nothing is ready, rather than reporting a timeout that did not happen. the wait is
    // bounded by the caller's timeout, or by a slice when it asked to wait forever, so the syscall
    // stays interruptible and a caller that blocks indefinitely still yields.
    const deadline = timeoutMs < 0 ? -1 : this.nowMs() + timeoutMs;
    let seen = this.ioGeneration();
    let count = gather();
    while (count === 0 && timeoutMs !== 0) {
      const remaining = deadline < 0 ? -1 : Math.max(0, deadline - this.nowMs());
      if (deadline >= 0 && remaining === 0) break;
      if (!this.parkForIo(seen, remaining)) break; // nothing here can block at all
      seen = this.ioGeneration();
      count = gather();
      // a signal ends the wait. with events already gathered they are reported (they happened);
      // with none, the answer is EINTR -- which is what epoll_pwait's caller is testing for, and
      // returning 0 instead told it the wait had timed out when it had not.
      if (this.hasPendingDeliverable()) {
        if (count === 0) return -EINTR;
        break;
      }
    }
    return count;
  }

  /**
   * who setpriority/getpriority are talking about. the worker supplies the process table, because
   * the answer is about OTHER processes and nothing in this handler can see them.
   */
  prioHook: {
    list: () => Array<{ pid: number; pgid: number; ruid: number; nice: number }>;
    setNice: (pid: number, nice: number) => void;
  } | null = null;

  private priorityTargets(which: number, who: number): Array<{ pid: number; pgid: number; ruid: number; nice: number }> {
    const all = this.prioHook?.list() ?? [];
    // `who == 0` means "me", spelled differently for each kind of set
    switch (which) {
      case PRIO_PROCESS: {
        const want = who === 0 ? this.currentPid : who;
        return all.filter((p) => p.pid === want);
      }
      case PRIO_PGRP: {
        const self = all.find((p) => p.pid === this.currentPid);
        const want = who === 0 ? (self?.pgid ?? this.currentPid) : who;
        return all.filter((p) => p.pgid === want);
      }
      default: {
        const want = who === 0 ? this.ruid : who;
        return all.filter((p) => p.ruid === want);
      }
    }
  }

  /** the current io readiness generation; 0 when nothing here can produce events. */
  private ioGeneration(): number {
    return this.io.ioGeneration?.() ?? 0;
  }

  /**
   * park until something may have become ready, or `remainingMs` elapses (negative = no deadline
   * of our own). `seen` must be the generation read BEFORE the readiness check that came up
   * empty; otherwise an event landing between the check and the park is slept through.
   *
   * returns false when there is no way to block at all, which is the case off-device where poll
   * degrades to a non-blocking check. the park itself is bounded internally so a pending signal
   * is still noticed.
   */
  private parkForIo(seen: number, remainingMs: number): boolean {
    if (this.io.waitIo) { this.io.waitIo(seen, remainingMs); return true; }
    if (this.io.waitStdin) { this.io.waitStdin(remainingMs); return true; }
    return false;
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
    // not const: openat2 is openat with its flags and mode in a struct rather than in registers,
    // so it validates the struct and then falls into the openat case with these two set from it.
    let a2 = cpu.getReg(2);
    let a3 = cpu.getReg(3);
    const a4 = cpu.getReg(4);
    const a5 = cpu.getReg(5);

    // set by the openat2 preamble before it falls into the openat case, which then opens this
    // rather than resolving the arguments again -- the scoping flags have already had their say.
    let openat2Path: string | null = null;
    // RESOLVE_NO_SYMLINKS: only the filesystem can see the links, so the refusal has to travel
    // with the open rather than being decided here.
    let openat2NoSymlinks = false;

    const view = new DataView(memory.buffer);
    const bytes = new Uint8Array(memory.buffer);

    let ret: bigint = -BigInt(ENOSYS);

    // a sibling thread may have changed a signal disposition since our last syscall; this is the
    // boundary where a pending signal is delivered, so it is where the group's table is re-read.
    this.sigSync();

    // linux copies every pathname in through getname() BEFORE the syscall's own logic runs, so a
    // bad path pointer is EFAULT and nothing else happens. doing that here, once, from a table of
    // which argument holds a path, is the same guarantee -- and it is the only version of this that
    // stays true: a check per case is a check you can forget to add to the thirty-first case.
    const pathArgs = PATH_ARG_REGS[nr];
    if (pathArgs !== undefined) {
      const args = [a0, a1, a2, a3, a4, a5];
      for (const i of pathArgs) {
        if (this.ptrFaults(bytes, Number(args[i]))) return this.retErr(cpu, EFAULT);
        // getname() also bounds the name at PATH_MAX, and it does so BEFORE the syscall resolves
        // anything -- including the descriptor a relative name hangs off. checking it later gave
        // fchmodat(fd_of_a_file, <4098 chars>) ENOTDIR, which is the answer to a question linux
        // never got as far as asking. per-component limits are a resolution-time rule and stay
        // where resolution happens.
        if (readGuestString(bytes, Number(args[i])).length > 4095) {
          return this.retErr(cpu, ENAMETOOLONG);
        }
      }
    }

    // and the dirfd each relative path resolves against, for the same reason and in the same place.
    // an ABSOLUTE path ignores the dirfd entirely, and an EMPTY one is the AT_EMPTY_PATH case where
    // the descriptor names the object itself and may legitimately not be a directory -- so neither
    // is checked here.
    const flagSpec = AT_FLAG_ARGS[nr];
    if (flagSpec !== undefined) {
      const args = [a0, a1, a2, a3, a4, a5];
      const given = Number(BigInt.asIntN(32, args[flagSpec[0]]));
      if ((given & ~flagSpec[1]) !== 0) return this.retErr(cpu, EINVAL);
    }

    const dirfdArgs = DIRFD_ARG_REGS[nr];
    if (dirfdArgs !== undefined) {
      const args = [a0, a1, a2, a3, a4, a5];
      for (const [dfdReg, pathReg] of dirfdArgs) {
        const raw = readGuestString(bytes, Number(args[pathReg]));
        if (raw.length === 0 || raw.startsWith('/')) continue;
        const dfd = Number(BigInt.asIntN(32, args[dfdReg]));
        if (dfd === -100) continue; // AT_FDCWD
        if (!this.fdIsOpen(dfd)) return this.retErr(cpu, EBADF);
        if (!this.fdIsDir(dfd)) return this.retErr(cpu, ENOTDIR);
      }
    }

    if (NET_TRACE && (this as any)._netActive && nr !== 64 && nr !== 66 && nr !== 63) this.nt(`sys ${nr} a0=${Number(a0 & 0xffffn)}`);

    switch (nr) {
      // ── Process lifecycle ──
      case 93:  // exit
        // only the low 8 bits of an exit status survive, exactly as on linux -- and here it is
        // also what keeps a guest's exit code out of the bits the signalled-death marker uses.
        return { status: 'exit', code: Number(a0) & 0xff };
      case 94:  // exit_group
        // every thread of the group ends, not just this one. exit(3) and _exit(2) both issue this,
        // so a thread calling either used to end only itself and leave the process running -- which
        // is how mq_notify's callback could exit(0) and the program carried on waiting for it.
        return { status: 'exit', code: Number(a0) & 0xff, group: true };

      case 220: { // clone(flags, stack, parent_tid, tls, child_tid)
        const flags = Number(a0);
        const SIGCHLD = 17;
        // If SIGCHLD is set, treat as fork
        if (flags & SIGCHLD) {
          const svcPC = view.getBigUint64(PC_OFFSET, true);
          // the remaining arguments are not thread-only. a fork-style clone with a child stack is
          // exactly what vfork(), posix_spawn() and glibc/musl's system() issue, and every libc's
          // __clone stub stashes the function pointer at the top of THAT stack and reloads it from
          // sp in the child. dropping the stack left the child reading its parent's frame and
          // branching into garbage -- a SIGILL that looked like a translator bug.
          return { status: 'fork', resumePC: svcPC + 4n, flags, childStack: a1, ptid: a2, tls: a3, ctid: a4 };
        }
        // Other clone variants not supported
        ret = -BigInt(ENOSYS);
        break;
      }

      case 221: { // execve(pathname, argv, envp)
        if (this.ptrFaults(bytes, Number(a0))) { ret = -BigInt(EFAULT); break; }
        const pathname = readGuestString(bytes, Number(a0));
        { const pe221 = this.pathCheck(pathname); if (pe221) { ret = -BigInt(pe221); break; } }

        // Read argv array (null-terminated array of pointers)
        const argvPtrs: string[] = [];
        let argvAddr = Number(a1);
        while (true) {
          const ptr = Number(view.getBigUint64(argvAddr + GUEST_MEM_BASE, true));
          if (ptr === 0) break;
          argvPtrs.push(readGuestString(bytes, ptr));
          argvAddr += 8;
        }

        // an EMPTY argv still gives the new program an argv[0]: the kernel fills in one empty
        // string rather than letting argc be zero. a program that trusts argv[0] to exist (and
        // most do, starting with every error message that names itself) reads whatever follows
        // it otherwise -- which is CVE-2021-4034.
        if (argvPtrs.length === 0) argvPtrs.push('');

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

        // In our vfork model, the child has already exited by the time
        // the parent resumes. The exit code was stored by the dispatch loop.
        if (this.lastWaitStatus !== undefined) {
          const exitCode = this.lastWaitStatus;
          this.lastWaitStatus = undefined;
          if (wstatus !== 0) {
            // a child terminated by a signal is marked with the 0x10000 bit and carries the signal
            // in the low 7 bits -> WIFSIGNALED (raw status = signum). otherwise WIFEXITED (code<<8).
            const encoded = (exitCode & 0x10000)
              ? ((exitCode & 0x7f) | ((exitCode & 0x20000) ? 0x80 : 0)) // WIFSIGNALED (+ WCOREDUMP)
              : ((exitCode & 0xFF) << 8);
            view.setUint32(wstatus + GUEST_MEM_BASE, encoded, true);
          }
          // the fourth argument is a struct rusage for the reaped child. we have no accounting for
          // it on this path, but leaving the buffer untouched is worse than saying zero: the caller
          // reads back whatever its own stack held.
          if (Number(a3) !== 0) this.writeRusage(view, Number(a3), 0, 0, 0);
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
        // a descriptor opened write-only cannot be read, and says so with EBADF rather than
        // quietly handing back bytes. (the standard descriptors carry their real modes now, so
        // this also covers reading from stdout.)
        if (readEntry !== undefined && (!this.fdAccess(fd).readable || this.fdIsPath(fd))) { ret = -BigInt(EBADF); break; }
        // valid fd but the destination buffer is outside the address space -> EFAULT (count 0 is a no-op).
        if ((readEntry !== undefined || (this.vfs?.hasFd(fd) ?? false)) && count > 0 && this.ptrFaults(bytes, buf, count)) {
          ret = -BigInt(EFAULT); break;
        }
        if (readEntry) {
          if (readEntry.type === 'devnode') {
            ret = BigInt(this.deviceRead(readEntry.major, readEntry.minor, bytes, buf, count));
            break;
          }
          if (readEntry.type === 'pipe_write') {
            ret = -BigInt(EBADF); // can't read from write end of pipe
            break;
          }
          if (readEntry.type === 'pipe_read') {
            const pipe = readEntry.pipe;
            if (pipe.data.length === 0) {
              if (pipe.writeClosed) {
                ret = 0n; // EOF
              } else if (this.isNonBlocking(fd)) {
                ret = -BigInt(EAGAIN);
              } else {
                ret = 0n; // Pipes currently don't block; legacy behavior
              }
            } else {
              const chunk = pipe.data.shift()!;
              const toRead = Math.min(count, chunk.length);
              bytes.set(chunk.subarray(0, toRead), buf + GUEST_MEM_BASE);
              if (toRead < chunk.length) {
                pipe.data.unshift(chunk.subarray(toRead));
              }
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
            // the buffer-too-small check comes FIRST: a read of fewer than 8 bytes is EINVAL
            // whether or not the counter has anything in it, and answering EAGAIN instead told a
            // caller with a bad buffer to try again forever.
            if (count < 8) { ret = -BigInt(EINVAL); break; }
            let efdR = this.eventfdHook?.({ op: 'read', id: readEntry.id }) ?? [-EBADF];
            if (efdR[0] === -11) {
              if (this.isNonBlocking(fd)) { ret = -BigInt(EAGAIN); break; }
              // a blocking read waits for a write, exactly as it does on a pipe. returning EAGAIN
              // to a descriptor that never asked for O_NONBLOCK is a lie the caller cannot handle:
              // read(2) does not fail that way unless it was told to.
              let seen = this.ioGeneration();
              let efdIntr = false;
              while (efdR[0] === -11) {
                if (!this.parkForIo(seen, -1)) break; // nothing here can ever wake it
                seen = this.ioGeneration();
                if (this.hasPendingDeliverable()) { efdIntr = true; break; }
                efdR = this.eventfdHook?.({ op: 'read', id: readEntry.id }) ?? [-EBADF];
              }
              if (efdR[0] === -11) { ret = -BigInt(efdIntr ? EINTR : EAGAIN); break; }
            }
            if (efdR[0] < 0) { ret = BigInt(efdR[0]); break; }
            view.setBigUint64(buf + GUEST_MEM_BASE,
                              (BigInt(efdR[2] >>> 0) << 32n) | BigInt(efdR[1] >>> 0), true);
            ret = 8n;
            break;
          }
          if (readEntry.type === 'inotify') {
            // the worker intercepts a read on an inotify descriptor, because the event queue lives
            // in the kernel. reaching here means there is no kernel (the host-side test path).
            ret = -BigInt(EAGAIN);
            break;
          }
          if (readEntry.type === 'timerfd') {
            // a read yields the number of expirations since the last one, as a u64, and clears it.
            // like eventfd, the too-small buffer is EINVAL regardless of whether anything is armed.
            if (count < 8) { ret = -BigInt(EINVAL); break; }
            const tfr = this.timerFds.get(readEntry.id);
            if (!tfr) { ret = -BigInt(EINVAL); break; }
            this.checkTimers();
            let tfErr = 0;
            while (tfr.ticks === 0n) {
              // a disarmed timer will never become ready, so waiting for it is waiting forever.
              if (tfr.nextNs === 0n || this.isNonBlocking(fd)) { tfErr = EAGAIN; break; }
              // sleep no longer than the wait itself: no io event announces a timer, so the
              // deadline is the only thing that can end this park on time.
              const untilMs = Number((tfr.nextNs - this.timerClockNs(tfr.clockid)) / 1_000_000n);
              if (!this.parkForIo(this.ioGeneration(), Math.max(1, untilMs))) { tfErr = EAGAIN; break; }
              this.checkTimers();
              if (tfr.ticks === 0n && this.pendingSignals.length) { tfErr = EINTR; break; }
            }
            if (tfErr !== 0) { ret = -BigInt(tfErr); break; }
            view.setBigUint64(buf + GUEST_MEM_BASE, tfr.ticks, true);
            tfr.ticks = 0n;
            ret = 8n;
            break;
          }
          if (readEntry.type === 'epoll' || readEntry.type === 'mqueue') {
            ret = -BigInt(EINVAL); // neither is read(2)-able: use epoll_wait / mq_timedreceive
            break;
          }
          if (readEntry.type === 'jitterctl') {
            // first read runs the command host-side; later reads stream the rest of the reply.
            if (readEntry.resp === null) {
              const e = readEntry;
              return {
                status: 'bridge', method: 'jitter_ctl', args: { cmd: e.req },
                apply: (resp: any) => {
                  // `ret` carries the command's exit status, `data` its stdout (base64).
                  const body = resp?.bytes ?? new Uint8Array(0);
                  const out = new Uint8Array(4 + body.length);
                  new DataView(out.buffer).setInt32(0, Number(resp?.ret ?? 1), true);
                  out.set(body, 4);
                  e.resp = out;
                  e.pos = 0;
                  const n = Math.min(count, out.length);
                  bytes.set(out.subarray(0, n), buf + GUEST_MEM_BASE);
                  e.pos = n;
                  cpu.setReg(0, BigInt(n));
                },
              };
            }
            const remaining = readEntry.resp.length - readEntry.pos;
            const n = Math.min(count, remaining);
            if (n > 0) {
              bytes.set(readEntry.resp.subarray(readEntry.pos, readEntry.pos + n), buf + GUEST_MEM_BASE);
              readEntry.pos += n;
            }
            ret = BigInt(n);
            break;
          }
          if (readEntry.type === 'vfs') {
            ret = this.vfs ? BigInt(this.vfs.read(readEntry.fd, buf, count, view)) : -BigInt(EBADF);
            break;
          }
          if (readEntry.type === 'native') {
            // A READ OF A REGULAR FILE RETURNS WHAT WAS ASKED FOR. linux only short-reads one at
            // end of file or on a signal, and real software relies on that: apk sizes a buffer to
            // the file it is extracting and reads into it, and a short read told it the archive had
            // ended early -- so every file over its 128 KiB buffer threshold was installed empty.
            //
            // the bridge's response slot holds ~16 KB, which is a fact about our transport and not
            // about the file, so the chunking belongs BELOW the syscall rather than above it. the
            // hook gathers as many slot-sized pieces as the caller asked for and hands back one
            // result, exactly as the kernel would.
            if (this.readNativeHook) {
              const got = this.readNativeHook(readEntry.nativeFd, count);
              if (typeof got === 'number') { ret = -BigInt(got); break; }
              if (got.length) { bytes.set(got, buf + GUEST_MEM_BASE); this.notify(readEntry.path, IN_ACCESS, 0, false, { unlinked: readEntry.unlinked === true }); }
              ret = BigInt(got.length);
              break;
            }
            const rcount = Math.min(count, BRIDGE_READ_MAX);
            return {
              status: 'bridge', method: 'fs_read',
              args: { fd: readEntry.nativeFd, count: rcount },
              apply: (resp: any) => {
                if (resp.ret < 0) {
                  cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                } else if (resp.bytes) {
                  this.notify(readEntry.path, IN_ACCESS, 0, false, { unlinked: readEntry.unlinked === true });
                  bytes.set(resp.bytes, buf + GUEST_MEM_BASE);
                  cpu.setReg(0, BigInt(resp.bytes.length));
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
            const scount = Math.min(count, BRIDGE_READ_MAX);
            return {
              status: 'bridge', method: 'net_read',
              args: { socketId: readEntry.socketId, count: scount,
                      nonblock: this.isNonBlockingFd(fd) ? 1 : 0 },
              apply: (resp: any) => {
                if (resp.ret < 0) {
                  this.nt(`read fd=${fd} req=${count} -> err ${resp.ret}`);
                  cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                } else if (resp.bytes) {
                  bytes.set(resp.bytes, buf + GUEST_MEM_BASE);
                  cpu.setReg(0, BigInt(resp.bytes.length));
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
        // the mirror of the read rule: a descriptor opened read-only cannot be written.
        if (writeEntry !== undefined && (!this.fdAccess(fd).writable || this.fdIsPath(fd))) { ret = -BigInt(EBADF); break; }
        // for a valid fd, a source buffer outside the accessible address space faults EFAULT (the
        // kernel validates the fd first, then the buffer). count 0 never touches the buffer.
        if ((writeEntry !== undefined || (this.vfs?.hasFd(fd) ?? false)) && count > 0 && this.ptrFaults(bytes, buf, count)) {
          ret = -BigInt(EFAULT); break;
        }
        if (writeEntry) {
          if (writeEntry.type === 'devnode') {
            ret = BigInt(this.deviceWrite(writeEntry.major, writeEntry.minor, bytes, buf, count));
            break;
          }
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
          if (writeEntry.type === 'jitterctl') {
            // accumulate the request; it is dispatched host-side on the first read (see there).
            writeEntry.req += new TextDecoder().decode(bytes.subarray(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count));
            ret = BigInt(count);
            break;
          }
          if (writeEntry.type === 'vfs') {
            ret = this.vfs ? BigInt(this.vfs.write(writeEntry.fd, buf, count, view)) : -BigInt(EBADF);
            break;
          }
          if (writeEntry.type === 'native') {
            // a sealed file keeps its promise: F_SEAL_WRITE refuses the write outright, and
            // F_SEAL_GROW refuses one that would take the file past where it ends.
            const wSeals = this.sealsOf(writeEntry);
            if (wSeals & 0x8) { ret = -BigInt(EPERM); break; }
            const writeData = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
            const b64 = bytesToBase64(writeData);
            this.invalidateStat(writeEntry.path); // invalidate stat cache for this file
            return {
              status: 'bridge', method: 'fs_write',
              args: { fd: writeEntry.nativeFd, data: b64, limit: this.fsizeLimitArg(),
                      // F_SEAL_GROW: the file may not get longer, so a write past the end is
                      // refused rather than shortened. only the filesystem knows where the end is.
                      noGrow: (wSeals & 0x4) ? 1 : 0 },
              apply: (resp: any) => {
                if (resp.ret < 0) {
                  // a file that may not grow raises SIGXFSZ as well as failing, which is what makes
                  // a shell say "File size limit exceeded" instead of losing the tail in silence.
                  if (resp.ret === -EFBIG) this.raiseSignal(25);
                  cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                } else {
                  this.notify(writeEntry.path, IN_MODIFY, 0, false, { unlinked: writeEntry.unlinked === true });
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
            const efdWrite = () => this.eventfdHook?.({ op: 'write', id: writeEntry.id, value: value.toString() }) ?? [-EBADF];
            let efdW = efdWrite();
            if (efdW[0] === -11) {
              if (this.isNonBlocking(fd)) { ret = -BigInt(EAGAIN); break; }
              let seen = this.ioGeneration();
              let efdWIntr = false;
              while (efdW[0] === -11) {
                if (!this.parkForIo(seen, -1)) break;
                seen = this.ioGeneration();
                if (this.hasPendingDeliverable()) { efdWIntr = true; break; }
                efdW = efdWrite();
              }
              if (efdW[0] === -11) { ret = -BigInt(efdWIntr ? EINTR : EAGAIN); break; }
            }
            if (efdW[0] < 0) { ret = BigInt(efdW[0]); break; }
            ret = 8n;
            break;
          }
          if (writeEntry.type === 'inotify' || writeEntry.type === 'timerfd' || writeEntry.type === 'epoll') {
            ret = -BigInt(EBADF); // not writable
            break;
          }
          if (writeEntry.type === 'mqueue') { ret = -BigInt(EINVAL); break; } // mq_timedsend only
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
        if (!this.fdExists(fd) || !this.fdAccess(fd).writable || this.fdIsPath(fd)) { ret = -BigInt(EBADF); break; }
        if (iovcnt < 0 || iovcnt > IOV_MAX) { ret = -BigInt(EINVAL); break; }
        if (iovcnt > 0 && this.ptrFaults(bytes, iovBase, iovcnt * 16)) { ret = -BigInt(EFAULT); break; }
        const wvLen = this.iovTotal(view, iovBase, iovcnt);
        if (wvLen < 0) { ret = BigInt(wvLen); break; }
        // every buffer is checked BEFORE any of them is written. linux imports the whole iovec
        // array up front, so a bad pointer in the second entry means EFAULT and nothing written --
        // writing the first entry and then reporting a positive count says the call partly
        // succeeded, which is a different (and unrecoverable) thing to tell a caller.
        let wvFault = false;
        for (let i = 0; i < iovcnt; i++) {
          const bPtr = Number(view.getBigUint64(iovBase + i * 16 + GUEST_MEM_BASE, true));
          const bLen = Number(view.getBigUint64(iovBase + i * 16 + 8 + GUEST_MEM_BASE, true));
          if (bLen > 0 && this.ptrFaults(bytes, bPtr, bLen)) { wvFault = true; break; }
        }
        if (wvFault) { ret = -BigInt(EFAULT); break; }
        let totalWritten = 0;
        let writevBad = false;

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
          this.invalidateStat(writevEntry.path);
          return {
            status: 'bridge', method: 'fs_write',
            args: { fd: writevEntry.nativeFd, data: b64, limit: this.fsizeLimitArg() },
            apply: (resp: any) => {
              if (resp.ret < 0) {
                if (resp.ret === -EFBIG) this.raiseSignal(25);
                cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              } else {
                this.notifyFd(fd, IN_MODIFY);
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
            writevBad = true;
            break;
          }
          totalWritten += len;
        }
        ret = writevBad ? -BigInt(EBADF) : BigInt(totalWritten);
        break;
      }

      case 65: { // readv(fd, iov, iovcnt) - scatter read
        const rvFd = Number(a0);
        const rvIov = Number(a1);
        const rvCnt = Number(a2);
        // the argument checks readv used to skip entirely: a bad descriptor, an iovcnt past
        // IOV_MAX, a negative iov_len and an unreadable iovec array all reported success.
        if (!this.fdExists(rvFd) || !this.fdAccess(rvFd).readable || this.fdIsPath(rvFd)) { ret = -BigInt(EBADF); break; }
        if (rvCnt < 0 || rvCnt > IOV_MAX) { ret = -BigInt(EINVAL); break; }
        if (rvCnt > 0 && this.ptrFaults(bytes, rvIov, rvCnt * 16)) { ret = -BigInt(EFAULT); break; }
        const rvLen = this.iovTotal(view, rvIov, rvCnt);
        if (rvLen < 0) { ret = BigInt(rvLen); break; }
        const rvEntry = this.getFdEntry(rvFd);
        // a directory can be opened and seeked; reading it is EISDIR, not a short read of nothing.
        if (rvEntry?.type === 'native' && rvEntry.isDir) { ret = -BigInt(EISDIR); break; }
        // every buffer the read would land in has to be writable BEFORE any of it happens, which is
        // what makes a bad address in the second vector fail the whole call rather than half of it.
        // a zero-length vector names no memory at all, so its address is never looked at.
        let rvFault = false;
        for (let i = 0; i < rvCnt; i++) {
          const len = Number(view.getBigUint64(rvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
          if (len === 0) continue;
          const base = Number(view.getBigUint64(rvIov + i * 16 + GUEST_MEM_BASE, true));
          if (this.ptrWriteFaults(bytes, base, len)) { rvFault = true; break; }
        }
        if (rvFault) { ret = -BigInt(EFAULT); break; }
        // readv SCATTERS: it fills the vectors in order from one read. filling only the first and
        // calling the rest a short read gave a caller with three buffers a third of its data.
        const rvScatter = (d: Uint8Array) => {
          let off = 0;
          for (let i = 0; i < rvCnt && off < d.length; i++) {
            const len = Number(view.getBigUint64(rvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
            if (len === 0) continue;
            const base = Number(view.getBigUint64(rvIov + i * 16 + GUEST_MEM_BASE, true));
            const n = Math.min(len, d.length - off);
            bytes.set(d.subarray(off, off + n), base + GUEST_MEM_BASE);
            off += n;
          }
        };
        const rvWant = Math.min(rvLen, BRIDGE_READ_MAX);
        const applyBridgeRead = (resp: any) => {
          if (resp.ret < 0) { this.nt(`readv fd=${rvFd} req=${rvWant} -> err ${resp.ret}`); cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); }
          else if (resp.bytes) { const d = resp.bytes; rvScatter(d); this.nt(`readv fd=${rvFd} req=${rvWant} -> ${d.length} (ret=${resp.ret})`); cpu.setReg(0, BigInt(d.length)); }
          else { this.nt(`readv fd=${rvFd} req=${rvWant} -> EOF ret=${resp.ret}`); cpu.setReg(0, BigInt(resp.ret)); }
        };
        if (rvEntry?.type === 'socket') {
          if (rvWant === 0) { ret = 0n; break; }
          return { status: 'bridge', method: 'net_read', args: { socketId: rvEntry.socketId, count: rvWant }, apply: applyBridgeRead };
        }
        if (rvEntry?.type === 'native') {
          if (rvWant === 0) { ret = 0n; break; }
          return { status: 'bridge', method: 'fs_read', args: { fd: rvEntry.nativeFd, count: rvWant }, apply: applyBridgeRead };
        }
        let rvTotal = 0;
        let rvErr: bigint | null = null;
        for (let i = 0; i < rvCnt; i++) {
          const base = Number(view.getBigUint64(rvIov + i * 16 + GUEST_MEM_BASE, true));
          const len = Number(view.getBigUint64(rvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
          if (len === 0) continue;
          if (this.vfs) {
            const n = this.vfs.read(rvFd, base, len, view);
            if (n < 0) { rvErr = rvTotal === 0 ? BigInt(n) : BigInt(rvTotal); break; }
            rvTotal += n;
            if (n < len) break; // short read
          }
        }
        ret = rvErr ?? BigInt(rvTotal);
        break;
      }

      case 67: { // pread64(fd, buf, count, offset) - read at offset without moving the position
        const prFd = Number(a0), prBuf = Number(a1), prCount = Number(a2);
        const prOffset = BigInt.asIntN(64, a3);
        const prErr = this.positionalIoCheck(prFd, prOffset, false);
        if (prErr !== 0) { ret = BigInt(prErr); break; }
        if (prCount > 0 && this.ptrFaults(bytes, prBuf, prCount)) { ret = -BigInt(EFAULT); break; }
        if (prCount === 0) { ret = 0n; break; }
        const prEntry = this.fdTable.get(prFd);
        if (prEntry?.type === 'native') {
          return {
            status: 'bridge', method: 'fs_pread',
            args: { fd: prEntry.nativeFd, offset: Number(prOffset), count: Math.min(prCount, BRIDGE_READ_MAX) },
            apply: (resp: any) => {
              if (resp.ret < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); return; }
              const d = resp.bytes ?? new Uint8Array(0);
              new Uint8Array(memory.buffer).set(d, prBuf + GUEST_MEM_BASE);
              cpu.setReg(0, BigInt(d.length));
            },
          };
        }
        if (this.vfs?.hasFd(prFd)) {
          const savedPos = this.vfs.lseek(prFd, 0n, 1); // SEEK_CUR
          if (savedPos < 0n) { ret = savedPos; break; }
          this.vfs.lseek(prFd, prOffset, 0);
          const n = this.vfs.read(prFd, prBuf, prCount, view);
          this.vfs.lseek(prFd, savedPos, 0); // pread must not move the position
          ret = BigInt(n);
          break;
        }
        ret = -BigInt(EBADF);
        break;
      }

      case 68: { // pwrite64(fd, buf, count, offset) - write at offset without moving the position
        const pwFd = Number(a0), pwBuf = Number(a1), pwCount = Number(a2);
        const pwOffset = BigInt.asIntN(64, a3);
        const pwErr = this.positionalIoCheck(pwFd, pwOffset, true);
        if (pwErr !== 0) { ret = BigInt(pwErr); break; }
        if (pwCount > 0 && this.ptrFaults(bytes, pwBuf, pwCount)) { ret = -BigInt(EFAULT); break; }
        if (pwCount === 0) { ret = 0n; break; }
        const pwEntry = this.fdTable.get(pwFd);
        // a sealed file keeps its promise however the write is spelled: F_SEAL_WRITE refuses a
        // positional write exactly as it refuses an ordinary one.
        if (this.sealsOf(pwEntry) & 0x8) { ret = -BigInt(EPERM); break; }
        if (pwEntry?.type === 'native') {
          // this whole branch was missing: pwrite on a real file fell through to the vfs, which
          // does not have the descriptor, and every positional write to disk returned EBADF.
          const payload = bytes.slice(pwBuf + GUEST_MEM_BASE, pwBuf + GUEST_MEM_BASE + pwCount);
          this.invalidateStat(pwEntry.path);
          return {
            status: 'bridge', method: 'fs_pwrite',
            args: { fd: pwEntry.nativeFd, offset: Number(pwOffset), data: bytesToBase64(payload),
                    noGrow: (this.sealsOf(pwEntry) & 0x4) ? 1 : 0 },
            apply: (resp: any) => {
              if (resp.ret >= 0) this.notifyFd(pwFd, IN_MODIFY);
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            },
          };
        }
        if (this.vfs?.hasFd(pwFd)) {
          const savedPos = this.vfs.lseek(pwFd, 0n, 1);
          if (savedPos < 0n) { ret = savedPos; break; }
          this.vfs.lseek(pwFd, pwOffset, 0);
          const n = this.vfs.write(pwFd, pwBuf, pwCount, view);
          this.vfs.lseek(pwFd, savedPos, 0);
          ret = BigInt(n);
          break;
        }
        ret = -BigInt(EBADF);
        break;
      }

      // ── Pipes & fd duplication ──
      case 59: { // pipe2(pipefd, flags)
        const pipefdAddr = Number(a0);
        // a pipe consumes two descriptors; enforce RLIMIT_NOFILE so a create-until-failure loop
        // terminates with EMFILE instead of allocating fds (and pipe buffers) forever.
        if (!this.fdAvailable()) { ret = -BigInt(EMFILE); break; }
        const pipe: PipeBuffer = { data: [], writeClosed: false, readClosed: false };
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
        const dup3flags = Number(BigInt.asIntN(32, a2));
        this.ft(`dup3 ${oldfd} -> ${newfd}`);
        // the flags come first, before any question about the descriptors: dup3's only flag is
        // O_CLOEXEC, and anything else is EINVAL even when both descriptors are nonsense.
        if (dup3flags & ~0o2000000) { ret = -BigInt(EINVAL); break; }
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
          if (existingEntry.type === 'pipe_write') existingEntry.pipe.writeClosed = true;
          if (existingEntry.type === 'pipe_read') existingEntry.pipe.readClosed = true;
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
      /**
       * openat2(dirfd, pathname, how, size).
       *
       * openat with its flags and mode moved into a struct that can grow, plus a `resolve` word
       * of restrictions on how the path may be walked. the extensible struct has real rules: a
       * short one is EINVAL, a long one is only allowed if everything past what we understand is
       * zero, and anything else is E2BIG -- that is how a program discovers what this kernel
       * supports. this used to accept anything, ignore `resolve` entirely, and then look only in
       * the per-process /proc and /dev tree rather than at the real filesystem.
       */
      case 437: {
        const OPEN_HOW_SIZE = 24;    // flags, mode, resolve -- three u64s
        const howPtr = Number(a2);
        const howSize = Number(BigInt.asUintN(64, a3));
        if (howSize < OPEN_HOW_SIZE) { ret = -BigInt(EINVAL); break; }
        if (this.ptrFaults(bytes, howPtr, howSize)) { ret = -BigInt(EFAULT); break; }
        const howBase = howPtr + GUEST_MEM_BASE;
        // a struct longer than we understand is only acceptable if all of the extra is zero:
        // otherwise the caller is asking for something this kernel would silently not do.
        let tooBig = false;
        for (let off = OPEN_HOW_SIZE; off < howSize; off++) if (bytes[howBase + off] !== 0) { tooBig = true; break; }
        if (tooBig) { ret = -BigInt(E2BIG); break; }
        const howFlags = view.getBigUint64(howBase, true);
        const howMode = view.getBigUint64(howBase + 8, true);
        const howResolve = view.getBigUint64(howBase + 16, true);
        if ((howFlags & ~VALID_OPEN_FLAGS) !== 0n) { ret = -BigInt(EINVAL); break; }
        if ((howResolve & ~VALID_RESOLVE) !== 0n) { ret = -BigInt(EINVAL); break; }
        // the two scoping flags are different answers to the same question.
        if ((howResolve & RESOLVE_BENEATH) && (howResolve & RESOLVE_IN_ROOT)) { ret = -BigInt(EINVAL); break; }
        // a mode is only meaningful when the call may create a file, and then only permission bits.
        const willCreate = (howFlags & (0o100n | O_TMPFILE_BIT)) !== 0n;
        if (willCreate ? (howMode & ~0o7777n) !== 0n : howMode !== 0n) { ret = -BigInt(EINVAL); break; }
        if (howFlags & O_TMPFILE_BIT) {
          // O_TMPFILE is __O_TMPFILE|O_DIRECTORY and must be spelled that way, without O_CREAT;
          // and an unnamed file you cannot write to is of no use to anyone.
          if ((howFlags & (O_TMPFILE_BIT | O_DIRECTORY_B | 0o100n)) !== (O_TMPFILE_BIT | O_DIRECTORY_B)) {
            ret = -BigInt(EINVAL); break;
          }
          if ((howFlags & 3n) !== 1n && (howFlags & 3n) !== 2n) { ret = -BigInt(EINVAL); break; }
        }
        if ((howFlags & O_PATH_B) && (howFlags & ~O_PATH_FLAGS) !== 0n) { ret = -BigInt(EINVAL); break; }

        if (this.ptrFaults(bytes, Number(a1))) { ret = -BigInt(EFAULT); break; }
        const scoped = this.scopeOpenPath(a0, readGuestString(bytes, Number(a1)), howResolve);
        if (typeof scoped === 'number') { ret = -BigInt(scoped); break; }
        // the scoped path is already absolute, so the dirfd has nothing left to contribute.
        openat2Path = scoped;
        openat2NoSymlinks = (howResolve & RESOLVE_NO_SYMLINKS) !== 0n;
        a2 = howFlags;
        a3 = howMode;
        // fall through: with the struct unpacked, the rest is exactly openat.
      }
      // eslint-disable-next-line no-fallthrough
      case 56: { // openat(dirfd, path, flags, mode)
        if (openat2Path === null && this.ptrFaults(bytes, Number(a1))) { ret = -BigInt(EFAULT); break; }
        // an empty pathname names nothing. every one of these calls used to resolve it to the
        // working directory and operate on THAT, which is a different file from the one the
        // caller failed to name.
        const raw56 = openat2Path === null ? readGuestString(bytes, Number(a1)) : null;
        if (raw56 !== null && this.isEmptyPath(raw56)) { ret = -BigInt(ENOENT); break; }
        const path = openat2Path ?? this.resolveAt(a0, raw56!);
        const flags56 = Number(a2);
        const mode56 = Number(a3);
        if (path === '/dev/jitter') {
          const jfd = this.allocFd();
          this.fdTable.set(jfd, { type: 'jitterctl', req: '', resp: null, pos: 0 });
          ret = BigInt(jfd);
          break;
        }
        if (path === '/dev/tty') {
          const ttyFd = this.allocFd();
          this.fdTable.set(ttyFd, { type: 'tty' });
          this.ft(`open ${path} -> ${ttyFd}`);
          ret = BigInt(ttyFd);
          break;
        }
        // RLIMIT_NOFILE bounds every descriptor a process can hold, and open is the main way it
        // gets them. without this the limit was enforced on dup but not on open, so a program
        // could walk straight past its own ceiling.
        if (!this.fdAvailable(0)) { ret = -BigInt(EMFILE); break; }
        // /proc/<pid>/fd/<n> is a MAGIC LINK: opening it opens the object the descriptor holds,
        // not the name the link reads back as. for an anonymous file -- a memfd, an O_TMPFILE, or
        // anything unlinked since it was opened -- there is no name to open, and following the
        // link text opened nothing at all.
        const magic56 = path.match(/^\/proc\/(?:self|\d+)\/fd\/(\d+)$/);
        if (magic56) {
          const srcFd = Number(magic56[1]);
          const srcEnt = this.getFdEntry(srcFd);
          if (srcEnt?.type === 'native') {
            const reFd = this.allocFd();
            const reFlags = flags56;
            return {
              status: 'bridge', method: 'fs_reopen',
              // O_TRUNC through the link is a shrink of the object, and a file sealed against
              // shrinking refuses it.
              args: { fd: srcEnt.nativeFd, flags: reFlags,
                      noShrink: (this.sealsOf(srcEnt) & 0x2) ? 1 : 0 },
              apply: (resp: any) => {
                if (resp.ret < 0) { this.unregisterFd(reFd); cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); return; }
                this.fdTable.set(reFd, {
                  type: 'native', nativeFd: resp.ret, path: srcEnt.path, ofdId: this.newOfdIdPublic(),
                  isDir: srcEnt.isDir, ftype: srcEnt.ftype, unlinked: srcEnt.unlinked,
                  sealable: srcEnt.sealable,
                });
                this.nativeRefs.set(resp.ret, 1);
                this.fdFlags.set(reFd, reFlags & ~(0o100 | 0o1000 | 0o2000000));
                if (reFlags & 0o2000000) this.fdCloexec.add(reFd);
                cpu.setReg(0, BigInt(reFd));
              },
            };
          }
        }
        const bp56 = this.bridgePath(path);
        if (bp56 !== null) {
          const guestFd = this.allocFd();
          return {
            status: 'bridge', method: 'fs_open',
            args: { path: bp56, flags: flags56, mode: mode56, noSymlinks: openat2NoSymlinks ? 1 : 0 },
            apply: (resp: any) => {
              // -ENODEV means the path names a DEVICE NODE. the filesystem stores the node; which
              // driver sits behind its major:minor is the kernel's business, so it is decided here.
              if (resp.ret === -19 && resp._meta) {
                const dmode = resp._meta[0] ?? 0;
                const drdev = resp._meta[1] ?? 0;
                const major = (drdev >> 8) & 0xfff;
                const minor = (drdev & 0xff) | ((drdev >> 12) & 0xfff00);
                if (!devKind(major, minor)) {
                  cpu.setReg(0, BigInt.asUintN(64, BigInt(-6))); // ENXIO: a node with no driver
                  return;
                }
                this.fdTable.set(guestFd, { type: 'devnode', major, minor });
                this.fdFlags.set(guestFd, flags56 & ~(0o100 | 0o200 | 0o1000 | 0o2000000));
                if (flags56 & 0o2000000) this.fdCloexec.add(guestFd);
                this.ft(`open ${bp56} -> ${guestFd} (dev ${major}:${minor} mode ${dmode.toString(8)})`);
                cpu.setReg(0, BigInt(guestFd));
                return;
              }
              if (resp.ret < 0) {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                // no fd was committed to the table, so the number is still free
              } else {
                // the reply carries the mode, so the descriptor knows from the start whether it is a
                // directory. asking the filesystem again on every event it generates would be a
                // round trip per event, for an answer that cannot change while the fd is open.
                const openedDir = ((resp._meta?.[0] ?? 0) & 0o170000) === 0o040000;
                this.fdTable.set(guestFd, {
                  type: 'native', nativeFd: resp.ret, path: bp56, ofdId: this.newOfdId(),
                  isDir: openedDir, ftype: (resp._meta?.[0] ?? 0) & 0o170000,
                });
                this.nativeRefs.set(resp.ret, 1);
                // O_CREAT on a path that was not there is a creation; every open is an open. only
                // the filesystem knows which happened, so it says so in the reply.
                if (resp._meta?.[2]) this.notify(bp56, IN_CREATE, 0, false);
                this.notify(bp56, IN_OPEN, 0, openedDir);
                if (flags56 & 0o1000 /* O_TRUNC */) this.notify(bp56, IN_MODIFY, 0, false);
                // status flags for F_GETFL (strip creation-only flags); O_CLOEXEC -> FD_CLOEXEC
                this.fdFlags.set(guestFd, flags56 & ~(0o100 | 0o200 | 0o1000 | 0o2000000));
                if (flags56 & 0o2000000) this.fdCloexec.add(guestFd);
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
        // which close event a watcher sees depends on the access mode, so read it BEFORE the
        // flags are cleared below -- reading it after gets the default and calls every close of a
        // writable file a read-only one.
        const closedWritable = this.fdAccess(fd).writable;
        // per-fd metadata never outlives the fd; clearing here keeps a later reuse
        // of the same number (lowest-free allocation) from inheriting stale state
        this.fdFlags.delete(fd);
        this.fdCloexec.delete(fd);
        if (closeEntry) {
          if (closeEntry.type === 'pipe_write') closeEntry.pipe.writeClosed = true;
          if (closeEntry.type === 'pipe_read') closeEntry.pipe.readClosed = true;
          if (closeEntry.type === 'vfs' && this.vfs) this.vfs.close(closeEntry.fd);
          if (closeEntry.type === 'native') {
            // a watcher waiting for a file to be FINISHED is waiting for IN_CLOSE_WRITE.
            const closedPath = closeEntry.path;
            const closedDir = closeEntry.isDir === true, closedGone = closeEntry.unlinked === true;
            this.fdTable.delete(fd);
            this.notify(closedPath, closedWritable ? IN_CLOSE_WRITE : IN_CLOSE_NOWRITE, 0, closedDir,
                        { unlinked: closedGone });
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
          if (closeEntry.type === 'eventfd') {
            this.fdTable.delete(fd);
            // one process may hold several descriptors onto the same counter (dup, or an inherited
            // one); the kernel only wants to hear about the last of them.
            let stillHeld = false;
            for (const [, e] of this.fdTable) if (e.type === 'eventfd' && e.id === closeEntry.id) { stillHeld = true; break; }
            if (!stillHeld) this.eventfdHook?.({ op: 'close', id: closeEntry.id, pid: this.currentPid });
            ret = 0n;
            break;
          }
          if (closeEntry.type === 'epoll') this.epollSets?.delete(fd);
          this.fdTable.delete(fd);
          if (closeEntry.type === 'timerfd') {
            // a dup keeps the same timer alive; the last descriptor onto it takes it away.
            let stillOpen = false;
            for (const other of this.fdTable.values()) {
              if (other.type === 'timerfd' && other.id === closeEntry.id) { stillOpen = true; break; }
            }
            if (!stillOpen) this.timerFds.delete(closeEntry.id);
          }
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
        // ftruncate only means anything for a regular file. a socket, a pipe or a terminal has no
        // length to set, and linux says so with EINVAL -- not EBADF, which claims the descriptor
        // does not exist. an unwritable descriptor is EINVAL for the same reason: the request is
        // ill-formed, not the fd.
        if (ftEntry !== undefined && ftEntry.type !== 'vfs' && ftEntry.type !== 'native') {
          ret = -BigInt(EINVAL); break;
        }
        if (ftLen < 0) { ret = -BigInt(EINVAL); break; }
        const ftLimit = this.rlimitOf(RLIMIT_FSIZE);
        if (ftLimit.cur !== RLIM_INFINITY && BigInt(ftLen) > ftLimit.cur) {
          this.raiseSignal(25); // SIGXFSZ
          ret = -BigInt(EFBIG);
          break;
        }
        if (this.fdIsOpen(ftFd) && (!this.fdAccess(ftFd).writable || this.fdIsPath(ftFd))) {
          ret = -BigInt(EINVAL); break;
        }
        if (ftEntry && ftEntry.type === 'vfs' && this.vfs) {
          ret = BigInt(this.vfs.ftruncate(ftEntry.fd, ftLen));
        } else if (ftEntry && ftEntry.type === 'native') {
          this.invalidateStat(ftEntry.path);
          // a sealed file keeps its promise, and only the filesystem knows the current size to
          // compare the new one against: F_SEAL_SHRINK refuses a smaller length and F_SEAL_GROW a
          // larger one. F_SEAL_WRITE is NOT one of them -- it is a promise about the CONTENTS, and
          // a write-sealed file stays as resizable as it was, which is what memfd_create01 checks.
          const ftSeals = this.sealsOf(ftEntry);
          return {
            status: 'bridge', method: 'fs_ftruncate',
            args: {
              fd: ftEntry.nativeFd, length: ftLen,
              noShrink: (ftSeals & 0x2) ? 1 : 0,
              noGrow: (ftSeals & 0x4) ? 1 : 0,
            },
            apply: (resp: any) => {
              if (resp.ret >= 0) this.notify(ftEntry.path, IN_MODIFY, 0, false);
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            },
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
            // a memory device has no position to lose: linux lets you seek one and hands back the
            // offset you asked for, which is how `dd skip=` over /dev/zero works.
            case 'devnode':
              ret = devKind(lseekEntry.major, lseekEntry.minor) === 'tty'
                ? -BigInt(ESPIPE) : BigInt.asUintN(64, BigInt.asIntN(64, a1));
              break;
            case 'native':
              // the offset is SIGNED, and seeking backwards is not an edge case: reading a trailer
              // with SEEK_END and a negative offset, or pushing back an over-read with SEEK_CUR and
              // a negative one, is how every streaming archive reader works. taking the raw
              // register as a number turned -100 into 18446744073709551516, which the host quite
              // reasonably refused -- so the seek failed, the stream lost its place, and the file
              // being extracted came out empty.
              return {
                status: 'bridge', method: 'fs_lseek',
                args: { fd: lseekEntry.nativeFd, offset: Number(BigInt.asIntN(64, a1)), whence: Number(a2) },
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
          } else if (fstatEntry.type === 'devnode') {
            writeChardevStat(view, statBuf, (fstatEntry.major << 8) | (fstatEntry.minor & 0xff)
              | ((fstatEntry.minor & ~0xff) << 12));
            ret = 0n;
          } else if (fstatEntry.type === 'stdio' || fstatEntry.type === 'tty' ||
              fstatEntry.type === 'pipe_read' || fstatEntry.type === 'pipe_write' ||
              fstatEntry.type === 'inotify' || fstatEntry.type === 'timerfd' ||
              fstatEntry.type === 'eventfd' || fstatEntry.type === 'epoll') {
            writeChardevStat(view, statBuf);
            ret = 0n;
          } else if (fstatEntry.type === 'native') {
            // through the DESCRIPTOR, not the name. a name is not what fstat was given, and for an
            // anonymous file (memfd, O_TMPFILE) or one unlinked since it was opened there is no
            // name to stat -- so this answered ENOENT for a file the caller plainly had open.
            return {
              status: 'bridge', method: 'fs_fstat',
              args: { fd: fstatEntry.nativeFd },
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
        const raw79 = readGuestString(bytes, Number(a1));
        if (this.isEmptyPath(raw79)) { if (!(Number(a3) & 0x1000)) { ret = -BigInt(ENOENT); break; } }
        else { const pe79 = this.pathCheck(raw79); if (pe79) { ret = -BigInt(pe79); break; } }
        const path = this.resolveAt(a0, raw79);
        const statBuf79 = Number(a2);
        if (this.ptrFaults(bytes, statBuf79, 128)) { ret = -BigInt(EFAULT); break; }
        const bp79 = this.bridgePath(path);
        if (bp79 !== null) {
          // AT_SYMLINK_NOFOLLOW is how every libc spells lstat(), so it describes the link itself.
          // the two flavours cannot share a cache entry -- one says "symlink", the other says what
          // it points at -- so nofollow entries are kept under their own key. getdents fills the
          // nofollow side (attributesOfItem does not follow), which is what `ls -l` reads back.
          const nofollow79 = (Number(a3) & 0x100) !== 0;
          const key79 = nofollow79 ? 'L:' + bp79 : bp79;
          const cached = this.statCache.get(key79);
          if (cached && cached.expiresAt > Date.now()) {
            writeBridgeStat(view, statBuf79, { _meta: cached.meta });
            ret = 0n;
            break;
          }
          return {
            status: 'bridge', method: nofollow79 ? 'fs_lstat' : 'fs_stat',
            args: { path: bp79 },
            apply: (resp: any) => {
              if (resp.ret < 0) {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              } else {
                writeBridgeStat(view, statBuf79, resp);
                // Cache for subsequent stat calls
                if (resp._meta) {
                  this.cacheStat(key79, { meta: resp._meta.slice(),
                                          expiresAt: Date.now() + this.STAT_CACHE_TTL_MS });
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
              let entries: { name: string; type: number; mode?: number; size?: number; mtime?: number;
                             uid?: number; gid?: number; ino?: number; rdev?: number; nlink?: number;
                             atime?: number; ctime?: number; mtimeNsec?: number; atimeNsec?: number;
                             ctimeNsec?: number; blocks?: number; blksize?: number }[] = [];
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
                  // nofollow flavour: these come from a non-following host call, so a symlink entry
                  // describes the link. serving a plain stat() from them would misreport the target.
                  this.cacheStat('L:' + childPath, {
                    meta: [entry.mode, entry.size ?? 0, entry.nlink ?? 1, entry.mtime ?? 0,
                           entry.uid ?? 0, entry.gid ?? 0, entry.ino ?? 0, entry.rdev ?? 0,
                           entry.atime ?? 0, entry.ctime ?? 0, entry.mtimeNsec ?? 0,
                           entry.atimeNsec ?? 0, entry.ctimeNsec ?? 0,
                           entry.blocks ?? 0, entry.blksize ?? 4096],
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
        const raw34 = readGuestString(bytes, Number(a1));
        { const pe = this.pathCheck(raw34); if (pe) { ret = -BigInt(pe); break; } }
        const path = this.resolveAt(a0, raw34);
        const bp34 = this.bridgePath(path);
        if (bp34 !== null) {
          this.invalidateStat(bp34);
          return {
            status: 'bridge', method: 'fs_mkdir',
            args: { path: bp34, mode: Number(a2) },
            apply: (resp: any) => {
              if (resp.ret >= 0) this.notify(bp34, IN_CREATE | IN_ISDIR);
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
          this.invalidateStat(bp35);
          return {
            status: 'bridge', method: 'fs_unlink',
            // AT_REMOVEDIR is what tells unlink from rmdir. dropping it made both the same call,
            // so unlink(directory) succeeded (and recursively) instead of answering EISDIR.
            args: { path: bp35, removeDir: (Number(a2) & 0x200) !== 0 },
            apply: (resp: any) => {
              if (resp.ret >= 0) {
                const removedDir = (Number(a2) & 0x200) !== 0;
                // unlinking drops the link count, and that is an attribute change of the file --
                // reported to the file's own watch, and to nobody else. rmdir has no link count to
                // drop, so it says nothing of the sort.
                for (const e of this.fdTable.values()) {
                  if (e.type === 'native' && e.path === bp35) e.unlinked = true;
                }
                if (!removedDir) this.notify(bp35, IN_ATTRIB, 0, false, { selfOnly: true });
                this.notify(bp35, IN_DELETE_SELF, 0, removedDir);
                this.notify(bp35, IN_DELETE, 0, removedDir);
              }
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
        const rawMknod = readGuestString(bytes, Number(a1));
        { const pe = this.pathCheck(rawMknod); if (pe) { ret = -BigInt(pe); break; } }
        const path = this.resolveAt(a0, rawMknod);
        const mode = Number(a2);
        const bMknod = this.bridgePath(path);
        if (bMknod) {
          this.invalidateStat(bMknod);
          return {
            status: 'bridge', method: 'fs_mknod',
            // the device number matters: a char/block node IS its major:minor, and dropping it left
            // every device the guest made pointing at nothing.
            args: { path: bMknod, mode, dev: Number(a3) },
            apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); },
          };
        }
        if (this.vfs) {
          // S_IFIFO = 0o010000 — named pipe (mkfifo)
          // S_IFREG = 0o100000 — regular file
          // S_IFCHR = 0o020000 — character device. the in-memory tree is /proc and /dev, which are
          // ours and synthesised; a guest adding device nodes belongs on the real filesystem, where
          // mknod does support them.
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
        const rawOld38 = readGuestString(bytes, Number(a1));
        const rawNew38 = readGuestString(bytes, Number(a3));
        { const pe = this.pathCheck(rawOld38) || this.pathCheck(rawNew38); if (pe) { ret = -BigInt(pe); break; } }
        const oldpath = this.resolveAt(a0, rawOld38);
        const newpath = this.resolveAt(a2, rawNew38);
        const bOld = this.bridgePath(oldpath);
        if (bOld) {
          const bNew = this.bridgePath(newpath) ?? newpath;
          this.invalidateStat(bOld);
          this.invalidateStat(bNew);
          return {
            status: 'bridge', method: 'fs_rename',
            args: { oldPath: bOld, newPath: bNew },
            apply: (resp: any) => {
              // the two halves of a rename share a cookie, which is how a watcher pairs them up
              // into one move rather than reading them as an unrelated delete and create.
              if (resp.ret >= 0) {
                const ck = ++this.renameCookie;
                // the old name is gone by now, so the object has to be asked about under its new one.
                this.notify(bOld, IN_MOVED_FROM, ck, undefined, { probe: bNew });
                this.notify(bNew, IN_MOVED_TO, ck);
                this.notify(bOld, IN_MOVE_SELF, 0, undefined, { probe: bNew });
              }
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            },
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
        const rawLink = readGuestString(bytes, Number(a2));
        // an empty linkpath (the name to create) is ENOENT (symlink03). an empty target is legal.
        if (rawLink === '') { ret = -BigInt(ENOENT); break; }
        const linkpath = this.resolveAt(a1, rawLink);
        const bLink = this.bridgePath(linkpath);
        if (bLink) {
          this.invalidateStat(bLink);
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
        // naming an O_TMPFILE file. linux gives an unnamed file its name through the descriptor
        // that is its only handle, addressed as /proc/self/fd/N -- so this is not really a link
        // between two paths at all, and resolving the source as one would find nothing.
        const selfFd = /^\/proc\/(self|\d+)\/fd\/(\d+)$/.exec(oldpath);
        if (selfFd) {
          const srcFd = Number(selfFd[2]);
          const nativeSrc = this.fdNative(srcFd);
          if (nativeSrc >= 0) {
            const bLinkNew2 = this.bridgePath(newpath) ?? newpath;
            this.invalidateStat(bLinkNew2);
            return {
              status: 'bridge', method: 'fs_link_fd',
              args: { fd: nativeSrc, newPath: bLinkNew2, creds: this.getCreds() },
              apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); },
            };
          }
        }
        const bLinkOld = this.bridgePath(oldpath);
        if (bLinkOld) {
          const bLinkNew = this.bridgePath(newpath) ?? newpath;
          this.invalidateStat(bLinkNew);
          return {
            status: 'bridge', method: 'fs_link',
            args: { oldPath: bLinkOld, newPath: bLinkNew },
            apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); },
          };
        }
        if (this.vfs) {
          ret = BigInt(this.vfs.linkat(oldpath, newpath));
        } else {
          ret = -BigInt(ENOSYS);
        }
        break;
      }

      case 88: { // utimensat(dirfd, path, times, flags)
        const utFlags = Number(BigInt.asIntN(32, a3));
        if (utFlags & ~AT_SYMLINK_NOFOLLOW) { ret = -BigInt(EINVAL); break; }
        const utTimes = Number(a2);
        // times is two timespecs. a null pointer means "both now", and each half may instead say
        // UTIME_NOW or UTIME_OMIT in its nanoseconds -- the seconds are then ignored entirely,
        // which is why a garbage tv_sec beside UTIME_OMIT is not an error.
        const ts: number[] = [0, UTIME_NOW, 0, UTIME_NOW];
        if (utTimes !== 0) {
          if (this.ptrFaults(bytes, utTimes, 32)) { ret = -BigInt(EFAULT); break; }
          let bad = false;
          for (let i = 0; i < 2; i++) {
            const sec = view.getBigInt64(utTimes + i * 16 + GUEST_MEM_BASE, true);
            const nsec = Number(view.getBigInt64(utTimes + i * 16 + 8 + GUEST_MEM_BASE, true));
            if (nsec !== UTIME_NOW && nsec !== UTIME_OMIT && (nsec < 0 || nsec >= 1_000_000_000)) bad = true;
            ts[i * 2] = Number(sec);
            ts[i * 2 + 1] = nsec;
          }
          if (bad) { ret = -BigInt(EINVAL); break; }
          if (ts[1] === UTIME_OMIT && ts[3] === UTIME_OMIT) { ret = 0n; break; }
        }
        // a null pathname names the descriptor itself, which is how futimens is spelled.
        const utRaw = Number(a1) === 0 ? null : readGuestString(bytes, Number(a1));
        if (utRaw === null) {
          const utEnt = this.getFdEntry(Number(BigInt.asIntN(32, a0)));
          if (!utEnt || utEnt.type !== 'native') {
            ret = this.fdExists(Number(BigInt.asIntN(32, a0))) ? 0n : -BigInt(EBADF);
            break;
          }
          this.invalidateStat(utEnt.path);
          return {
            status: 'bridge', method: 'fs_utimens',
            args: { path: utEnt.path, times: ts, nofollow: false },
            apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret ?? 0))); },
          };
        }
        { const peUt = this.pathCheck(utRaw); if (peUt) { ret = -BigInt(peUt); break; } }
        const utPath = this.resolveAt(a0, utRaw);
        const bUt = this.bridgePath(utPath);
        if (bUt !== null) {
          this.invalidateStat(bUt);
          return {
            status: 'bridge', method: 'fs_utimens',
            args: { path: bUt, times: ts, nofollow: (utFlags & AT_SYMLINK_NOFOLLOW) !== 0 },
            apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret ?? 0))); },
          };
        }
        // the synthetic filesystems carry no timestamps a caller may set; the call is accepted
        // for a file that is there, which is all /proc and /dev can honestly promise.
        ret = 0n;
        break;
      }

      case 78: { // readlinkat(dirfd, path, buf, bufsiz)
        const rawPath = readGuestString(bytes, Number(a1));
        const path = this.resolveAt(a0, rawPath);
        const buf = Number(a2);
        const bufsiz = Number(a3);
        // an empty pathname names the DESCRIPTOR, if it was opened O_PATH|O_NOFOLLOW on a
        // symlink -- that is the documented way to read a link you already hold open. anything
        // else with an empty name is ENOENT.
        if (rawPath === '') {
          const rlFd = Number(BigInt.asIntN(32, a0));
          const rlEnt = this.getFdEntry(rlFd);
          if (rlEnt?.type === 'native' && this.fdIsPath(rlFd) && bufsiz > 0) {
            return {
              status: 'bridge', method: 'fs_readlink',
              args: { path: rlEnt.path },
              apply: (resp: any) => {
                if (resp.ret < 0 || !resp.data) { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret < 0 ? resp.ret : -EINVAL))); return; }
                const target = Uint8Array.from(atob(resp.data), (c) => c.charCodeAt(0));
                const n = Math.min(target.length, bufsiz);
                if (badGuestPtr(bytes, buf, n)) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-14))); return; }
                bytes.set(target.subarray(0, n), buf + GUEST_MEM_BASE);
                cpu.setReg(0, BigInt(n));
              },
            };
          }
          ret = -BigInt(ENOENT); break;
        }
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
        const raw48 = readGuestString(bytes, Number(a1));
        { const pe = this.pathCheck(raw48); if (pe) { ret = -BigInt(pe); break; } }
        const path = this.resolveAt(a0, raw48);
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
        if (this.fdIsPath(fchmodFd) || this.vfs?.isOPath(fchmodFd)) { ret = -BigInt(EBADF); break; }
        const fchmodMode = Number(a1);
        const fchmodEntry = this.getFdEntry(fchmodFd);
        if (fchmodEntry?.type === 'native') {
          this.invalidateStat(fchmodEntry.path);
          return {
            status: 'bridge', method: 'fs_chmod',
            args: { path: fchmodEntry.path, mode: fchmodMode },
            apply: (resp: any) => {
              if (resp.ret >= 0) this.notify(fchmodEntry.path, IN_ATTRIB, 0, fchmodEntry.isDir === true,
                                             { unlinked: fchmodEntry.unlinked === true });
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            },
          };
        }
        if (this.vfs && this.vfs.hasFd(fchmodFd)) ret = BigInt(this.vfs.fchmod(fchmodFd, fchmodMode));
        else if (this.fdTable.has(fchmodFd)) ret = 0n; // native/stdio fd: best-effort no-op
        else ret = -BigInt(EBADF);
        break;
      }
      case 53: { // fchmodat(dirfd, path, mode)
        if (this.ptrFaults(bytes, Number(a1), 1)) { ret = -BigInt(EFAULT); break; }
        const raw53 = readGuestString(bytes, Number(a1));
        // the NAME is checked first (linux copies it in before it looks anything up), then the
        // descriptor it is relative to.
        { const pe = this.pathCheck(raw53); if (pe) { ret = -BigInt(pe); break; } }
        { const de = this.atDirfdError(a0, raw53); if (de) { ret = -BigInt(de); break; } }
        const fchmodatPath = this.resolveAt(a0, raw53);
        const fchmodatMode = Number(a2);
        const bpChmod = this.bridgePath(fchmodatPath);
        if (bpChmod !== null) {
          this.invalidateStat(bpChmod);
          return {
            status: 'bridge', method: 'fs_chmod',
            args: { path: bpChmod, mode: fchmodatMode },
            apply: (resp: any) => {
              if (resp.ret >= 0) this.notify(bpChmod, IN_ATTRIB);
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            },
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
        const raw54 = readGuestString(bytes, Number(a1));
        { const pe = this.pathCheck(raw54); if (pe) { ret = -BigInt(pe); break; } }
        const fchownatPath = this.resolveAt(a0, raw54);
        const bpChown = this.bridgePath(fchownatPath);
        if (bpChown !== null) {
          this.invalidateStat(bpChown);
          return {
            status: 'bridge', method: 'fs_chown',
            args: { path: bpChown, uid: cOwner, gid: cGroup, nofollow: (fchownFlags & 0x100) !== 0 },
            apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret ?? 0))); },
          };
        }
        ret = this.vfs ? BigInt(this.vfs.chownPath(fchownatPath, cOwner, cGroup)) : -BigInt(ENOENT);
        break;
      }
      case 55: { // fchown(fd, owner, group)
        const fchownFd = Number(a0);
        if (this.fdIsPath(fchownFd) || this.vfs?.isOPath(fchownFd)) { ret = -BigInt(EBADF); break; }
        const owner = Number(BigInt.asIntN(32, a1));
        const group = Number(BigInt.asIntN(32, a2));
        const ent = this.getFdEntry(fchownFd);
        if (ent?.type === 'native') {
          this.invalidateStat(ent.path);
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
        // the numbers are the same whichever file you ask about -- there is one filesystem -- but
        // the QUESTION still has to be valid. this used to fill the struct unconditionally, so
        // statfs of a path that does not exist, of a component that is not a directory, or through
        // a descriptor that was never opened all reported a healthy filesystem and returned 0.
        const statfsBuf = Number(a1);
        if (this.ptrWriteFaults(bytes, statfsBuf, 120)) { ret = -BigInt(EFAULT); break; }
        const fillStatfs = () => {
          const sv = new DataView(memory.buffer);
          const base = statfsBuf + GUEST_MEM_BASE;
          new Uint8Array(memory.buffer).fill(0, base, base + 120);
          sv.setBigUint64(base + 0, 0x01021994n, true);   // f_type (TMPFS_MAGIC)
          sv.setBigUint64(base + 8, 4096n, true);          // f_bsize
          sv.setBigUint64(base + 16, 1048576n, true);      // f_blocks
          sv.setBigUint64(base + 24, 524288n, true);       // f_bfree
          sv.setBigUint64(base + 32, 524288n, true);       // f_bavail
          sv.setBigUint64(base + 40, 1048576n, true);      // f_files
          sv.setBigUint64(base + 48, 524288n, true);       // f_ffree
          sv.setBigUint64(base + 64, 255n, true);          // f_namelen
          sv.setBigUint64(base + 72, 4096n, true);         // f_frsize
        };
        if (nr === 44) {
          if (!this.fdExists(Number(a0))) { ret = -BigInt(EBADF); break; }
          fillStatfs();
          ret = 0n;
          break;
        }
        if (this.ptrFaults(bytes, Number(a0), 1)) { ret = -BigInt(EFAULT); break; }
        const sfPath = readGuestString(bytes, Number(a0));
        if (sfPath.length > 4095) { ret = -BigInt(ENAMETOOLONG); break; }
        const sfBridge = this.bridgePath(sfPath);
        if (sfBridge === null) {
          // /proc and /dev are synthesised here, so the walk is ours to answer.
          if (!this.vfs || this.vfs.faccessat(-100, this.absolutizePath(sfPath), 0, 0) < 0) {
            ret = BigInt(this.vfs?.faccessat(-100, this.absolutizePath(sfPath), 0, 0) ?? -ENOENT);
            break;
          }
          fillStatfs();
          ret = 0n;
          break;
        }
        return {
          status: 'bridge', method: 'fs_stat',
          args: { path: sfBridge, creds: this.getCreds() },
          apply: (resp: any) => {
            if (resp.ret < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); return; }
            fillStatfs();
            cpu.setReg(0, 0n);
          },
        };
      }

      // ── ioctl ──
      case 29: { // ioctl(fd, request, ...)
        const fd = Number(a0);
        // O_PATH names a file without opening it, so every operation ON THE FILE is EBADF --
        // including ioctl, which must not fall through to "not a terminal".
        if (this.fdIsPath(fd) || this.vfs?.isOPath(fd)) { ret = -BigInt(EBADF); break; }
        // and a descriptor that is not open at all is EBADF before the request is even looked at:
        // "there is no such descriptor" is a different thing from "it does not do that".
        if (!this.fdIsOpen(fd)) { ret = -BigInt(EBADF); break; }
        // an ioctl request is a 32-bit encoding, and musl's wrapper takes it as an INT -- so any
        // request with the read/write direction bits set arrives sign-extended (RNDGETENTCNT as
        // 0xffffffff80045200). reading all 64 bits matched none of them.
        const request = Number(BigInt.asUintN(32, a1));

        // Resolve fd through fd table — check if this is a dup'd stdio fd
        const ioctlEntry = this.getFdEntry(fd);
        if (fd <= 4) this.ft(`ioctl ${fd} req=0x${request.toString(16)} type=${ioctlEntry?.type ?? (this.vfs?.hasFd(fd) ? 'vfs' : 'none')}`);
        const isStdio = ioctlEntry?.type === 'stdio' || ioctlEntry?.type === 'tty';

        // A TERMINAL IOCTL ONLY WORKS ON A TERMINAL. this is not pedantry: it is the ONLY way a
        // program can tell whether it is talking to one. isatty() is a TCGETS that succeeded, and
        // musl decides a stream's buffering by whether TIOCGWINSZ works on it. answering both for
        // every descriptor told every program it was on a terminal -- so a regular file got line
        // buffering, and anything that colours its output, draws a progress bar or asks before
        // overwriting behaved as if a person were watching, even when its output was a file.
        // linux says ENOTTY: "this object does not answer that request".
        if (!isStdio && (request === TCGETS || request === TCSETS ||
                         request === TIOCGWINSZ || request === TIOCSWINSZ)) {
          ret = -BigInt(ENOTTY);
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
          // RNDGETENTCNT: how much entropy the random pool holds. ours is the host's csprng, which
          // is never short of it, so the answer is the full pool -- the same number
          // /proc/sys/kernel/random/entropy_avail reports, which is what a caller compares it to.
          case 0x80045200: {
            const rndDev = ioctlEntry?.type === 'devnode'
              ? { major: ioctlEntry.major, minor: ioctlEntry.minor }
              : (this.vfs?.fdDevice(fd) ?? null);
            if (!rndDev || devKind(rndDev.major, rndDev.minor) !== 'random') { ret = -BigInt(ENOTTY); break; }
            const rndBuf = Number(a2);
            if (this.ptrFaults(bytes, rndBuf, 4)) { ret = -BigInt(14); break; }
            view.setInt32(rndBuf + GUEST_MEM_BASE, 4096, true);
            ret = 0n;
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
          // ── the socket ioctls ────────────────────────────────────────────────────
          // these are how a program asks what the machine is attached to. every one of them used
          // to be ENOTTY, so `ifconfig` and `ip` could not list an interface, let alone bring one
          // up, and a program looking for its own address found none.
          case SIOCATMARK: {
            if (ioctlEntry?.type !== 'socket') { ret = -BigInt(ENOTTY); break; }
            // urgent data is a stream idea. a datagram socket's protocol has no handler for the
            // request at all, which is ENOTTY -- the same answer a file gives.
            if (ioctlEntry.sockType !== 1) { ret = -BigInt(ENOTTY); break; }
            if (Number(a2) === 0 || this.ptrWriteFaults(bytes, Number(a2), 4)) { ret = -BigInt(EFAULT); break; }
            view.setInt32(Number(a2) + GUEST_MEM_BASE, 0, true); // no out-of-band data pending
            ret = 0n;
            break;
          }
          case SIOCGIFCONF: {
            if (ioctlEntry?.type !== 'socket') { ret = -BigInt(ENOTTY); break; }
            const ifcPtr = Number(a2);
            if (ifcPtr === 0 || this.ptrWriteFaults(bytes, ifcPtr, 16)) { ret = -BigInt(EFAULT); break; }
            return {
              status: 'bridge', method: 'net_ifaddrs', args: {},
              apply: (resp: any) => {
                const list = this.ifaceList(resp);
                // SIOCGIFCONF reports ipv4 addresses only; ipv6 has its own interfaces.
                const v4 = list.filter((i) => i.family === 2);
                const want = view.getInt32(ifcPtr + GUEST_MEM_BASE, true);
                const bufPtr = Number(view.getBigUint64(ifcPtr + 8 + GUEST_MEM_BASE, true));
                // a null buffer asks how much room the answer needs.
                if (bufPtr === 0) {
                  view.setInt32(ifcPtr + GUEST_MEM_BASE, v4.length * 40, true);
                  cpu.setReg(0, 0n);
                  return;
                }
                const n = Math.max(0, Math.min(v4.length, Math.floor(want / 40)));
                if (n > 0 && this.ptrWriteFaults(bytes, bufPtr, n * 40)) {
                  cpu.setReg(0, BigInt.asUintN(64, BigInt(-EFAULT)));
                  return;
                }
                for (let i = 0; i < n; i++) {
                  const base = bufPtr + i * 40;
                  bytes.fill(0, base + GUEST_MEM_BASE, base + GUEST_MEM_BASE + 40);
                  this.putIfName(bytes, base, v4[i].name);
                  this.putSockaddrRaw(view, base + 16, { family: 2, host: v4[i].addr, port: 0 });
                }
                view.setInt32(ifcPtr + GUEST_MEM_BASE, n * 40, true);
                cpu.setReg(0, 0n);
              },
            };
          }
          case SIOCGIFNAME:
          case SIOCGIFFLAGS:
          case SIOCSIFFLAGS:
          case SIOCGIFADDR:
          case SIOCGIFDSTADDR:
          case SIOCGIFBRDADDR:
          case SIOCGIFNETMASK:
          case SIOCGIFMTU:
          case SIOCGIFHWADDR:
          case SIOCGIFINDEX: {
            if (ioctlEntry?.type !== 'socket') { ret = -BigInt(ENOTTY); break; }
            const ifrPtr = Number(a2);
            if (ifrPtr === 0 || this.ptrWriteFaults(bytes, ifrPtr, 40)) { ret = -BigInt(EFAULT); break; }
            // changing an interface is an administrative act, and this kernel does not hand out
            // the address of a device it does not own. it can still answer "yes, it is up".
            if (request === SIOCSIFFLAGS && !this.hasCap(12)) { ret = -BigInt(EPERM); break; } // CAP_NET_ADMIN
            const wantName = this.readIfName(bytes, ifrPtr);
            const wantIndex = view.getInt32(ifrPtr + 16 + GUEST_MEM_BASE, true);
            return {
              status: 'bridge', method: 'net_ifaddrs', args: {},
              apply: (resp: any) => {
                const list = this.ifaceList(resp);
                const iface = request === SIOCGIFNAME
                  ? list.find((i) => i.index === wantIndex)
                  : list.find((i) => i.name === wantName);
                if (!iface) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-19))); return; } // ENODEV
                switch (request) {
                  case SIOCGIFNAME:
                    this.putIfName(bytes, ifrPtr, iface.name);
                    break;
                  case SIOCSIFFLAGS:
                    // the flags a program can set -- up, promiscuous, multicast -- are all already
                    // what they are on a device this kernel does not administer. accept and report.
                    break;
                  case SIOCGIFFLAGS:
                    view.setUint16(ifrPtr + 16 + GUEST_MEM_BASE, iface.flags & 0xffff, true);
                    break;
                  case SIOCGIFADDR:
                    this.putSockaddrRaw(view, ifrPtr + 16, { family: iface.family, host: iface.addr, port: 0 });
                    break;
                  case SIOCGIFDSTADDR:
                  case SIOCGIFBRDADDR:
                    this.putSockaddrRaw(view, ifrPtr + 16,
                      { family: iface.family, host: iface.broadcast || '0.0.0.0', port: 0 });
                    break;
                  case SIOCGIFNETMASK:
                    this.putSockaddrRaw(view, ifrPtr + 16,
                      { family: iface.family, host: iface.netmask || '0.0.0.0', port: 0 });
                    break;
                  case SIOCGIFMTU:
                    view.setInt32(ifrPtr + 16 + GUEST_MEM_BASE, iface.mtu, true);
                    break;
                  case SIOCGIFINDEX:
                    view.setInt32(ifrPtr + 16 + GUEST_MEM_BASE, iface.index, true);
                    break;
                  case SIOCGIFHWADDR:
                    // ARPHRD_LOOPBACK is 772, ARPHRD_ETHER is 1. the address itself is not ours to
                    // give: ios does not report a hardware address for any interface.
                    view.setUint16(ifrPtr + 16 + GUEST_MEM_BASE, (iface.flags & 8) ? 772 : 1, true);
                    bytes.fill(0, ifrPtr + 18 + GUEST_MEM_BASE, ifrPtr + 24 + GUEST_MEM_BASE);
                    break;
                }
                cpu.setReg(0, 0n);
              },
            };
          }
          default:
            // an ioctl the driver does not implement is ENOTTY, not EINVAL. EINVAL says the
            // ARGUMENTS were wrong, which sends a caller looking at its own code; ENOTTY says
            // "this object does not answer that request", which is the truth and what libc
            // expects when it probes a descriptor to decide how to buffer it.
            ret = -BigInt(ENOTTY);
        }
        break;
      }

      // ── Memory management ──
      case 214: { // brk(addr)
        const addr = a0;
        if (addr === 0n) {
          ret = this.brk;
        } else if (addr >= this.brk) {
          // the break belongs to the address space, so in a thread group it is shared: reading it,
          // deciding, and moving it all happen with the group held still.
          const grownFrom = this.withAddressSpace(() => {
            if (addr < this.brk) return -1; // a sibling already took it past here
            const needed = Number(addr) + GUEST_MEM_BASE;
            // budget: refuse to grow past the per-process ceiling. brk failure returns the
            // unchanged break, which glibc's wrapper reads as ENOMEM (malloc -> NULL).
            if (needed > this.maxBytes || !this.growTo(memory, needed)) return -1;
            const from = Number(this.brk);
            this.brk = addr;
            this.vmaDirty = true;
            return from;
          });
          // the heap the break just handed out is reachable memory. now that the guard table
          // starts out empty, saying so is what keeps malloc's new pages from faulting.
          if (grownFrom >= 0) {
            // a break that was given back and is now taken again is mapped memory once more: the
            // hole the shrink left has to go, or the guard would fault on the heap we just handed out.
            const lo = grownFrom & ~0xFFF, hi = (Number(addr) + 0xFFF) & ~0xFFF;
            this.protNone = this.protNone.filter((r) => r.end <= lo || r.start >= hi);
            this.noBacking = this.noBacking.filter((r) => r.end <= lo || r.start >= hi);
            try { this.protHook?.(grownFrom, Number(addr) - grownFrom, 0x3); } catch {}
          }
          ret = this.brk;
        } else if (addr >= this.brkFloor) {
          // giving the heap back. linux keeps mm->brk exactly where it was asked to, page-aligning
          // only the mapping behind it -- so an unaligned break is a break the caller reads back
          // unchanged. this used to refuse every shrink, which meant a program could not release
          // heap at all and sbrk(-n) silently did nothing.
          const shrunkFrom = this.withAddressSpace(() => {
            if (addr >= this.brk) return -1;   // a sibling moved it while we waited
            const from = Number(this.brk);
            this.brk = addr;
            this.vmaDirty = true;
            return from;
          });
          if (shrunkFrom >= 0) {
            const lo = (Number(addr) + 0xFFF) & ~0xFFF;
            const hi = (shrunkFrom + 0xFFF) & ~0xFFF;
            if (hi > lo) {
              // the pages behind the released heap become a hole, exactly as munmap leaves one:
              // touching them faults instead of reading whatever the program last wrote there.
              this.protNone = this.protNone.filter((r) => r.end <= lo || r.start >= hi);
              this.protNone.push({ start: lo, end: hi });
              this.setLocked(lo, hi, false);
              try { this.protHook?.(lo, hi - lo, 0); } catch {}
            }
          }
          ret = this.brk;
        } else {
          ret = this.brk;  // below the image: refused, and the break is unchanged
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
          if (mmapFd < 0 || !validFd || this.fdIsPath(mmapFd) || (this.vfs?.isOPath(mmapFd) ?? false)) { ret = -BigInt(EBADF); break; }
        }

        // zero length is invalid; exactly one map type (MAP_SHARED/MAP_PRIVATE, or the
        // MAP_SHARED_VALIDATE superset) must be requested.
        const mapType = flags & 0xf;
        if (length === 0 || (mapType !== 0x01 && mapType !== 0x02 && mapType !== 0x03)) {
          ret = -BigInt(EINVAL);
          break;
        }
        // only once the request itself is well-formed does WHO is asking matter: a mapping cannot
        // grant access the descriptor does not have. every file mapping needs the file readable
        // (the kernel has to fault pages in), and a MAP_SHARED mapping that may be written needs
        // it writable. EINVAL for a malformed request comes first, as it does in the kernel.
        if (!(flags & MAP_ANONYMOUS)) {
          const macc = this.fdAccess(mmapFd);
          if (!macc.readable) { ret = -BigInt(EACCES); break; }
          if ((flags & MAP_SHARED) && (prot & 0x2) && !macc.writable) { ret = -BigInt(EACCES); break; }
          // F_SEAL_WRITE is a promise that the file will never be written again, and a writable
          // shared mapping is a way to break it. the seal outranks the descriptor's own mode.
          if ((flags & MAP_SHARED) && (prot & 0x2) && (this.sealsOf(this.getFdEntry(mmapFd)) & 0x8)) {
            ret = -BigInt(EPERM); break;
          }
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
        // a shared mapping of a read-only file may never become writable; anything else may be
        // re-protected freely (a private mapping's writes are its own copy).
        // ...and a shared mapping of a WRITE-SEALED file may not become writable either, which is
        // the other half of the seal: refusing the writable mapping and then letting mprotect add
        // the write bit would leave the promise no stronger than a suggestion.
        const mmSealedRO = !(flags & MAP_ANONYMOUS) && (flags & MAP_SHARED)
          && ((this.sealsOf(this.getFdEntry(mmapFd)) & 0x8) !== 0 || !this.fdAccess(mmapFd).writable);
        const mmMaxProt = mmSealedRO ? 0x5 : 0x7;
        const mmVma = { prot: prot & 0x7, shared: (flags & MAP_SHARED) !== 0, maxProt: mmMaxProt,
                        grow: (flags & 0x100) !== 0 /* MAP_GROWSDOWN */ };
        // picking an address and claiming it is ONE step. between them, a sibling thread first-
        // fitting into the same hole would be handed the same memory.
        let mmErr = 0;
        const addr: bigint = this.withAddressSpace(() => {
          if (a0 !== 0n && (flags & (MAP_FIXED | MAP_FIXED_NOREPLACE))) {
            // MAP_FIXED_NOREPLACE refuses to clobber an existing mapping: EEXIST on any overlap.
            if ((flags & MAP_FIXED_NOREPLACE) && this.vmaOverlaps(Number(a0), Number(a0) + aligned)) {
              mmErr = EEXIST;
              return 0n;
            }
            const end = a0 + BigInt(aligned);
            if (end > this.mmapBase) { this.mmapBase = end; this.vmaDirty = true; }
            if (this.vmaWouldExceed(Number(a0), Number(a0) + aligned, mmVma)) { mmErr = ENOMEM; return 0n; }
            this.vmaSet(Number(a0), Number(a0) + aligned, mmVma);
            return a0;
          }
          // a bare hint is advisory, but linux DOES place the mapping there when the range is
          // free, and allocators lean on it: musl's mallocng asks for each new group next to the
          // last one so the heap stays packed. we honour it only inside the arena and only below
          // the cursor -- never over the image, the stack or the break, which is what made an
          // earlier attempt at this zero-fill live memory.
          if (a0 !== 0n) {
            const hint = Number(a0 & ~0xFFFn);
            if (hint >= this.mmapArenaStart && hint + aligned <= Number(this.mmapBase) &&
                !this.vmaOverlaps(hint, hint + aligned)) {
              if (this.vmaWouldExceed(hint, hint + aligned, mmVma)) { mmErr = ENOMEM; return 0n; }
              this.vmaSet(hint, hint + aligned, mmVma);
              return BigInt(hint);
            }
          }
          const picked = this.allocMmapRange(aligned);
          if (this.vmaWouldExceed(Number(picked), Number(picked) + aligned, mmVma)) { mmErr = ENOMEM; return 0n; }
          this.vmaSet(Number(picked), Number(picked) + aligned, mmVma);
          return picked;
        });
        if (mmErr) { ret = -BigInt(mmErr); break; }

        // Ensure WASM memory is large enough
        const needed = Number(addr) + aligned + GUEST_MEM_BASE;
        // budget: refuse mappings past the per-process ceiling -> ENOMEM (mmap returns MAP_FAILED),
        // rather than growing toward the jetsam limit and crashing the whole content process.
        // the range is already claimed by now, so a refusal has to give it back -- otherwise a
        // failed mmap would quietly consume address space nothing can ever use.
        // grow in 1 MiB steps rather than by exactly what was asked. a grow is not free -- the
        // engine may have to move the whole buffer -- and a page-at-a-time mmap loop was paying
        // for one every sixteenth call, which is where 196 us of a 4 KiB mmap was going.
        if (needed > this.maxBytes || !this.growTo(memory, needed, 16)) {
          this.vmaSet(Number(addr), Number(addr) + aligned, null);
          ret = -BigInt(ENOMEM);
          break;
        }

        // Zero-fill the mapped region
        new Uint8Array(memory.buffer).fill(0, Number(addr) + GUEST_MEM_BASE, Number(addr) + GUEST_MEM_BASE + aligned);

        // File-backed mmap: read file content into the mapped region.
        let sharedKey: string | undefined;
        // how many bytes of this mapping the FILE actually provides; -1 for an anonymous mapping,
        // which always has backing. anything past this has none, and touching it is SIGBUS.
        let fileBackedBytes = -1;
        if (mmapFd >= 0 && !(flags & MAP_ANONYMOUS)) {
          const fileOffset = Number(a5); // sixth argument: file offset
          const entry = this.fdTable.get(mmapFd);
          // a file-backed mapping is identified by what it maps, so two independent mappers of the
          // same bytes end up sharing one authoritative copy rather than two private ones.
          if (entry?.type === 'native') sharedKey = `${entry.path}:${fileOffset}:${aligned}`;
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
            fileBackedBytes = data.length;
            // writable shared mapping -> the guest's stores must reach the file (apk extracts here).
            // the writable span is what the file ACTUALLY holds, not what was asked for: a mapping
            // may run past EOF (the request is rounded up to a page), and writing that tail back
            // would extend the file, which linux never does. the short read above already told us
            // where the file ends.
            if ((flags & MAP_SHARED) && (prot & 0x2)) {
              const backed = Math.min(length, data.length);
              this.fileMmaps.push({ start: Number(addr), len: aligned, fileLen: backed, nativeFd: entry.nativeFd, fileOffset });
            }
          }
        }

        if (flags & MAP_SHARED) {
          this.sharedRegions.push({ addr: Number(addr), len: aligned });
          this.sharedMapHook?.(Number(addr), aligned, sharedKey);
        }

        // track PROT_NONE mappings so pointers into them fault; clear any stale overlap first.
        const mmStart = Number(addr), mmEnd = mmStart + aligned;
        this.protNone = this.protNone.filter((r) => r.end <= mmStart || r.start >= mmEnd);
        if ((prot & 0x7) === 0) this.protNone.push({ start: mmStart, end: mmEnd });

        if ((flags & 0x2000) || this.lockFuture) this.setLocked(mmStart, mmEnd, true); // MAP_LOCKED

        // enforcement is best-effort (same as mprotect): a mapping must never fail because the
        // guard's permission table couldn't be materialized.
        try { this.protHook?.(mmStart, aligned, prot & 0x7); } catch {}
        // pages of a FILE mapping that lie wholly past the end of the file have no backing store.
        // the mapping is legal -- mmap rounds the length up to a page and a caller may map more
        // than the file holds -- but touching one is SIGBUS, and that is how a program notices its
        // file is shorter than it expected. the partial last page is NOT one of these: linux reads
        // it as zeros and discards writes to the tail, so only whole pages beyond it are marked.
        this.noBacking = this.noBacking.filter((r) => r.end <= mmStart || r.start >= mmEnd);
        if (fileBackedBytes >= 0) {
          const firstDead = mmStart + Math.ceil(fileBackedBytes / 4096) * 4096;
          if (firstDead < mmEnd) {
            this.noBacking.push({ start: firstDead, end: mmEnd });
            try { this.protHook?.(firstDead, mmEnd - firstDead, PAGE_NO_BACKING); } catch {}
          }
        }

        ret = addr;
        break;
      }

      case 226: { // mprotect(addr, len, prot)
        // the start must be page-aligned (EINVAL) and the whole range must be mapped; the null page
        // and other clearly-unmapped low memory give ENOMEM (mprotect01).
        if ((a0 & 0xFFFn) !== 0n) { ret = -BigInt(EINVAL); break; }
        const mpStart = Number(a0), mpEnd = mpStart + ((Number(a1) + 0xFFF) & ~0xFFF);
        if (mpStart < 0x1000) { ret = -BigInt(ENOMEM); break; }
        // asking for more than the mapping is ever allowed is EACCES. without this a program could
        // map a read-only file shared, be refused write access by mmap, and then simply ask again.
        const mpWant = Number(a2) & 0x7;
        // checking the ceiling and applying the change is one step: a sibling thread must not be
        // able to slip a different mapping in between the two.
        const mpDenied = this.withAddressSpace(() => {
          const { i, j } = this.vmaSlice(mpStart, mpEnd);
          for (let k = i; k < j; k++) if (mpWant & ~this.vmas[k].maxProt) return EACCES;
          // re-protecting the middle of a mapping splits it in three, and those are mappings the
          // process has to be allowed to have. at the limit, linux refuses with ENOMEM.
          let extra = 0;
          for (let k = i; k < j; k++) {
            const v = this.vmas[k];
            if (v.prot === mpWant) continue;
            if (v.start < mpStart) extra++;
            if (v.end > mpEnd) extra++;
          }
          if (extra > 0 && this.vmas.length + extra > this.maxMapCount) return ENOMEM;
          this.vmaProtectLocked(mpStart, mpEnd, mpWant);
          return 0;
        });
        if (mpDenied) return this.retErr(cpu, mpDenied);
        this.protNone = this.protNone.filter((r) => r.end <= mpStart || r.start >= mpEnd);
        if ((Number(a2) & 0x7) === 0) this.protNone.push({ start: mpStart, end: mpEnd });
        // arming the guard must never be able to fail the guest's mprotect: the permission table is
        // how we ENFORCE the mapping, not what makes it exist.
        try { this.protHook?.(mpStart, mpEnd - mpStart, Number(a2) & 0x7); } catch {}
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
        // unmapping the middle of a mapping splits it in two, which costs a vma. at the limit that
        // cannot be paid, so giving memory back fails with ENOMEM -- the kernel does the same.
        if (this.withAddressSpace(() => this.vmaWouldExceed(umStart, umEnd, null))) { ret = -BigInt(ENOMEM); break; }
        this.sharedRegions = this.sharedRegions.filter((r) => r.addr + r.len <= umStart || r.addr >= umEnd);
        this.sharedUnmapHook?.(umStart, umEnd);
        // an unmapped range is not "free memory", it is a HOLE: touching it must fault. dropping
        // the PROT_NONE records for the range did the opposite -- it made the address freely
        // accessible again, so a use-after-unmap quietly read whatever bytes were still there.
        // record the hole instead, and arm the jit's page guard on it; a later mmap over the same
        // address clears both.
        this.protNone = this.protNone.filter((r) => r.end <= umStart || r.start >= umEnd);
        this.protNone.push({ start: umStart, end: umEnd });
        this.noBacking = this.noBacking.filter((r) => r.end <= umStart || r.start >= umEnd);
        try { this.protHook?.(umStart, umEnd - umStart, 0); } catch {}
        this.setLocked(umStart, umEnd, false);
        this.vmaSet(umStart, umEnd, null);
        const umFlush = this.buildMmapFlush(cpu, memory, umStart, (unmapLen + 0xFFF) & ~0xFFF, true);
        if (umFlush) return umFlush;
        ret = 0n;
        break;
      }

      // ── Thread/process ──
      case 96: { // set_tid_address(tidptr): returns the CALLER's thread id
        // it returned a hardcoded 1. libc stores this as the thread's own id for the rest of the
        // process's life, so EVERY process believed its tid was 1 -- and musl builds raise() out of
        // tkill(self_tid, sig). raise() therefore signalled PID 1, the init shell, instead of the
        // caller: abort() did not abort, assertion failures walked on, and a stray SIGTERM or
        // SIGPIPE landed on the shell from a program that never meant to signal anything at all.
        ret = BigInt(this.currentPid);
        break;
      }

      case 99: // set_robust_list(head, len)
        // the size is the kernel's version check on the structure: anything but its own is EINVAL.
        if (Number(BigInt.asUintN(64, a1)) !== 24) { ret = -BigInt(EINVAL); break; }
        this.robustListHead = BigInt.asUintN(64, a0);
        ret = 0n;
        break;

      // gettid handled with getpid above

      case 172: // getpid — the thread-group id (== pid) for a thread; own pid otherwise
        ret = BigInt(this.tgid || this.currentPid);
        break;

      case 173: // getppid
        // a process's parent CHANGES when the old one dies: the kernel hands it to init, or to
        // the nearest subreaper. reading the value cached at fork meant a process could never
        // find out it had been orphaned, which is the one thing getppid is asked for.
        ret = BigInt(this.ppidHook ? this.ppidHook() : this.currentPpid);
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
        // struct kernel_sigaction on aarch64: sa_handler, sa_flags, sa_restorer, sa_mask -- four
        // 8-byte fields. the size argument is the sigset's, and it is not advisory: a caller that
        // passes the wrong one is compiled against a different kernel and has to be told so.
        if (Number(BigInt.asIntN(64, a3)) !== 8) { ret = -BigInt(EINVAL); break; }
        const signum = Number(BigInt.asIntN(32, a0));
        const actPtr = Number(a1);
        const oldactPtr = Number(a2);
        if (actPtr !== 0 && this.ptrFaults(bytes, actPtr, 32)) { ret = -BigInt(EFAULT); break; }
        // SIGKILL and SIGSTOP belong to the kernel: there is no handler for them, and no way to
        // ignore them either, which is the whole point of having them.
        if (signum < 1 || signum > 64 || (actPtr !== 0 && (signum === 9 || signum === 19))) {
          ret = -BigInt(EINVAL); break;
        }
        if (!this.signalHandlers) this.signalHandlers = new Map();
        const oldSa = this.signalHandlers.get(signum);
        if (actPtr !== 0) {
          const base = actPtr + GUEST_MEM_BASE;
          const entry = {
            handler: view.getBigUint64(base, true),
            flags: view.getBigUint64(base + 8, true),
            restorer: view.getBigUint64(base + 16, true),
            mask: view.getBigUint64(base + 24, true),
          };
          this.signalHandlers.set(signum, entry);
          // the disposition belongs to the thread GROUP, so a threaded process publishes it.
          this.sigPublish(signum, entry);
        }
        if (oldactPtr !== 0) {
          if (this.ptrWriteFaults(bytes, oldactPtr, 32)) { ret = -BigInt(EFAULT); break; }
          const base = oldactPtr + GUEST_MEM_BASE;
          view.setBigUint64(base, oldSa?.handler ?? 0n, true);
          view.setBigUint64(base + 8, oldSa?.flags ?? 0n, true);
          view.setBigUint64(base + 16, oldSa?.restorer ?? 0n, true);
          view.setBigUint64(base + 24, oldSa?.mask ?? 0n, true);
        }
        ret = 0n;
        break;
      }

      case 135: { // rt_sigprocmask(how, set, oldset, sigsetsize)
        if (Number(BigInt.asIntN(64, a3)) !== 8) { ret = -BigInt(EINVAL); break; }
        const setPtr = Number(a1), oldsetPtr = Number(a2);
        // the old mask is the one in force BEFORE this call, so it is taken now and written after.
        const previous = this.signalMask ?? 0n;
        if (setPtr !== 0) {
          if (this.ptrFaults(bytes, setPtr, 8)) { ret = -BigInt(EFAULT); break; }
          const how = Number(BigInt.asIntN(32, a0));
          if (how !== 0 && how !== 1 && how !== 2) { ret = -BigInt(EINVAL); break; }
          // SIGKILL and SIGSTOP cannot be blocked. linux drops them from the set rather than
          // refusing the call, so a program that blocks "everything" still stops and still dies.
          const newSet = view.getBigUint64(setPtr + GUEST_MEM_BASE, true) & ~((1n << 8n) | (1n << 18n));
          if (how === 0) this.signalMask = previous | newSet;        // SIG_BLOCK
          else if (how === 1) this.signalMask = previous & ~newSet;  // SIG_UNBLOCK
          else this.signalMask = newSet;                             // SIG_SETMASK
          // unblocking is a DELIVERY POINT: linux hands over anything that was waiting on the mask
          // before it lets the caller run again. the pending queue is drained at the next check
          // either way, but a signal that just became deliverable should not wait on a poll.
          if (this.signalMask !== previous) this.signalsUnblocked?.();
        }
        if (oldsetPtr !== 0) {
          if (this.ptrWriteFaults(bytes, oldsetPtr, 8)) { ret = -BigInt(EFAULT); break; }
          view.setBigUint64(oldsetPtr + GUEST_MEM_BASE, previous, true);
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
        // what has been raised at this process and not yet taken -- which in practice means what
        // its own mask is holding back. answering "nothing" made sigpending useless as the way to
        // find out whether a blocked signal had arrived, which is what it is for.
        const pendSize = Number(BigInt.asIntN(64, a1));
        // linux takes a set SMALLER than its own here, and only refuses a larger one.
        if (pendSize > 8) { ret = -BigInt(EINVAL); break; }
        const pendPtr = Number(a0);
        if (pendSize > 0 && this.ptrWriteFaults(bytes, pendPtr, pendSize)) { ret = -BigInt(EFAULT); break; }
        let pendMask = 0n;
        for (const sig of this.pendingSignals) if (sig >= 1 && sig <= 64) pendMask |= 1n << BigInt(sig - 1);
        const outBytes = new Uint8Array(8);
        new DataView(outBytes.buffer).setBigUint64(0, pendMask, true);
        bytes.set(outBytes.subarray(0, pendSize), pendPtr + GUEST_MEM_BASE);
        ret = 0n;
        break;
      }

      // ── Filesystem info ──
      case 17: { // getcwd(buf, size)
        const buf = Number(a0);
        const size = Number(a1);
        // a chroot does not move the working directory. if the cwd is left OUTSIDE the new root
        // there is no path from the root that reaches it, and getcwd says so rather than
        // pretending the directory is the root -- which is CVE-2018-1000001, where realpath(".")
        // came back with a path that named a different directory entirely.
        const hostCwd = this.vfs ? this.vfs.getcwd() : this.cwd;
        if (this.rootDir !== '/' && hostCwd !== this.rootDir && !hostCwd.startsWith(this.rootDir + '/')) {
          ret = -BigInt(ENOENT); break;
        }
        const cwd = this.hostToGuest(hostCwd);
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
              if (!mode || (mode & 0o170000) !== 0o40000) {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(-ENOTDIR)));
                return;
              }
              // entering a directory needs search (execute) permission on it, which is the whole
              // point of a mode-0 directory. the stat reply carries the emulated owner, so the
              // check costs nothing extra.
              const dUid = resp._meta ? resp._meta[4] : 0;
              const dGid = resp._meta ? resp._meta[5] : 0;
              if (!this.mayExec(mode, dUid, dGid)) {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(-EACCES)));
                return;
              }
              // the working directory is the DIRECTORY, not the name used to reach it: linux stores
              // the resolved dentry, so getcwd after `cd some-symlink` reports the target. keeping
              // the caller's spelling made two paths to the same directory report differently.
              let realCwd = bPath;
              if (resp.data) { try { const d = atob(resp.data); if (d.length > 0) realCwd = d; } catch {} }
              this.cwd = realCwd;
              if (this.vfs) this.vfs.setCwd(realCwd);
              cpu.setReg(0, 0n);
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
        const fields = ['Linux', this.utsHook?.get('host') || 'jitter', '6.1.0', '#1', 'aarch64',
                        this.utsHook?.get('domain') || '(none)'];
        for (let i = 0; i < fields.length; i++) {
          const encoded = new TextEncoder().encode(fields[i] + '\0');
          new Uint8Array(memory.buffer).set(encoded, buf + i * 65); // each field is 65 bytes
        }
        ret = 0n;
        break;
      }

      // ── Time ──
      case 113: { // clock_gettime(clk_id, tp)
        const gtClk = Number(BigInt.asIntN(32, a0));
        if (!this.knownClock(gtClk)) { ret = -BigInt(EINVAL); break; }
        if (this.ptrFaults(bytes, Number(a1), 16)) { ret = -BigInt(EFAULT); break; }
        const tp = Number(a1) + GUEST_MEM_BASE;
        const ns = this.clockNs(gtClk);
        view.setBigUint64(tp, ns / 1_000_000_000n, true);
        view.setBigUint64(tp + 8, ns % 1_000_000_000n, true);
        ret = 0n;
        break;
      }

      case 112: { // clock_settime(clk_id, tp)
        const stClk = Number(BigInt.asIntN(32, a0));
        if (!this.knownClock(stClk)) { ret = -BigInt(EINVAL); break; }
        // only the settable clocks can be set: a monotonic clock that could be moved would not be
        // monotonic, and the cpu-time clocks measure something nobody gets to choose.
        if (stClk !== 0 && stClk !== 8 && stClk !== 11) { ret = -BigInt(EINVAL); break; }
        if (this.ptrFaults(bytes, Number(a1), 16)) { ret = -BigInt(EFAULT); break; }
        const stSec = view.getBigInt64(Number(a1) + GUEST_MEM_BASE, true);
        const stNsec = view.getBigInt64(Number(a1) + GUEST_MEM_BASE + 8, true);
        if (stNsec < 0n || stNsec >= 1_000_000_000n || stSec < 0n) { ret = -BigInt(EINVAL); break; }
        if (!this.hasCap(CAP_SYS_TIME)) { ret = -BigInt(EPERM); break; }
        // the host clock is not ours to move, so record the difference and add it on every read.
        const want = stSec * 1_000_000_000n + stNsec;
        this.clockOffsetNs = want - (this.clkRealBaseNs + this.elapsedNs());
        ret = 0n;
        break;
      }

      case 169: { // gettimeofday(tv, tz)
        // the timezone argument is obsolete and its contents are ignored, but the kernel still
        // WRITES to it when it is non-null, so a wild tz faults just as a wild tv does.
        if (this.ptrFaults(new Uint8Array(memory.buffer), Number(a0), 16)) { ret = -BigInt(EFAULT); break; }
        if (a1 !== 0n && this.ptrFaults(new Uint8Array(memory.buffer), Number(a1), 8)) { ret = -BigInt(EFAULT); break; }
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
            // O_ASYNC belongs in this set: it is how a program ARMS the descriptor to signal its
            // owner when it becomes ready, and leaving it out meant the request was accepted and
            // silently dropped -- the notification simply never came.
            const modifiable = 0o2000 | 0o4000 | 0o20000; // O_APPEND | O_NONBLOCK | O_ASYNC
            const current = this.fdFlags.get(fd) ?? 0;
            const newFlags = (current & ~modifiable) | (Number(a2) & modifiable);
            this.fdFlags.set(fd, newFlags);
            if ((current ^ newFlags) & 0o20000) this.ownerChanged?.(fd);
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
          case 5: // F_GETLK — query lock status
          case 12: { // F_GETLK64
            // cross-process advisory locks are enforced by the kernel on device; the
            // per-worker handler reports no conflicting lock (l_type = F_UNLCK).
            const flockPtr = Number(a2) + GUEST_MEM_BASE;
            view.setInt16(flockPtr, 2, true); // l_type = F_UNLCK
            ret = 0n;
            break;
          }
          case 6:  // F_SETLK
          case 7:  // F_SETLKW
          case 13: // F_SETLK64
          case 14: // F_SETLKW64
            ret = 0n; // granted here; real cross-process contention is arbitrated by the kernel
            break;
          case 8: { // F_SETOWN: positive is a pid, negative is a process GROUP, zero clears
            const who = Number(BigInt.asIntN(32, a2));
            if (who === 0) this.fdOwner.delete(fd);
            else this.fdOwner.set(fd, who > 0 ? { type: F_OWNER_PID, pid: who } : { type: F_OWNER_PGRP, pid: -who });
            this.ownerChanged?.(fd);
            ret = 0n;
            break;
          }
          case 9: { // F_GETOWN: a process group comes back NEGATED, the way it went in
            const o = this.fdOwner.get(fd);
            ret = BigInt(!o ? 0 : o.type === F_OWNER_PGRP ? -o.pid : o.pid);
            break;
          }
          case 10: { // F_SETSIG (linux: F_SETSIG=10, F_GETSIG=11 — these were swapped)
            this.fdSig.set(fd, Number(a2));
            this.ownerChanged?.(fd);
            // a lease already held on this descriptor breaks with the NEW signal, not the one that
            // was current when it was taken.
            const se = this.getFdEntry(fd);
            if (se?.type === 'native') this.leaseHook?.('sig', se.nativeFd, se.path, 0, Number(a2), false, true);
            ret = 0n;
            break;
          }
          case 11: // F_GETSIG
            ret = BigInt(this.fdSig.get(fd) ?? 0);
            break;
          case 15: { // F_SETOWN_EX: struct f_owner_ex { int type; pid_t pid; }
            const exPtr = Number(a2);
            if (this.ptrFaults(bytes, exPtr, 8)) { ret = -BigInt(EFAULT); break; }
            const otype = view.getInt32(exPtr + GUEST_MEM_BASE, true);
            const opid = view.getInt32(exPtr + GUEST_MEM_BASE + 4, true);
            if (otype !== F_OWNER_TID && otype !== F_OWNER_PID && otype !== F_OWNER_PGRP) {
              ret = -BigInt(EINVAL); break;
            }
            if (opid === 0) this.fdOwner.delete(fd);
            else this.fdOwner.set(fd, { type: otype, pid: opid });
            this.ownerChanged?.(fd);
            ret = 0n;
            break;
          }
          case 16: { // F_GETOWN_EX
            const exPtr = Number(a2);
            if (this.ptrFaults(bytes, exPtr, 8)) { ret = -BigInt(EFAULT); break; }
            const o = this.fdOwner.get(fd);
            view.setInt32(exPtr + GUEST_MEM_BASE, o?.type ?? F_OWNER_PID, true);
            view.setInt32(exPtr + GUEST_MEM_BASE + 4, o?.pid ?? 0, true);
            ret = 0n;
            break;
          }
          case 1024: { // F_SETLEASE
            const le = this.getFdEntry(fd);
            if (!le || le.type !== 'native') { ret = -BigInt(EINVAL); break; }
            const larg = Number(a2);
            if (larg !== 0 && larg !== 1 && larg !== 2) { ret = -BigInt(EINVAL); break; }
            const lfd = le.nativeFd, lpath = le.path;
            const lsig = this.fdSig.get(fd) || 29; // SIGIO unless F_SETSIG said otherwise
            const leuid = this.euid, lhook = this.leaseHook;
            // the grant depends on who owns the file and whether it is a regular one, and only
            // the filesystem knows either -- so ask it before deciding.
            return {
              status: 'bridge', method: 'fs_stat', args: { path: lpath },
              apply: (resp: any) => {
                if (resp.ret < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); return; }
                const mode = (resp._meta ? resp._meta[0] : resp.mode) ?? 0;
                const uid = (resp._meta ? resp._meta[4] : resp.uid) ?? 0;
                const regular = (mode & 0xf000) === 0x8000;
                const owner = leuid === 0 || uid === leuid;
                const r = lhook ? lhook('set', lfd, lpath, larg, lsig, owner, regular) : -EINVAL;
                cpu.setReg(0, BigInt.asUintN(64, BigInt(r)));
              },
            };
          }
          case 1025: { // F_GETLEASE
            const ge = this.getFdEntry(fd);
            if (!ge || ge.type !== 'native') { ret = 2n; break; } // F_UNLCK
            ret = BigInt(this.leaseHook?.('get', ge.nativeFd, ge.path, 0, 0, false, true) ?? 2);
            break;
          }
          case 1026: // F_NOTIFY
            ret = 0n;
            break;
          // a pipe's capacity is a promise about how much can be buffered before a writer
          // blocks, so it has to be the truth: our rings are a fixed slot in the kernel's shared
          // buffer, and claiming a megabyte would strand any program that sized its writes by the
          // answer. linux rounds a request up to a power of two and clamps it, so report what we
          // can actually hold and clamp requests to it.
          case 1033: { // F_ADD_SEALS(fd, seals)
            const sealEntry = this.getFdEntry(fd);
            const F_SEAL_ALL = 0x1 | 0x2 | 0x4 | 0x8 | 0x10; // SEAL|SHRINK|GROW|WRITE|FUTURE_WRITE
            const want = Number(BigInt.asUintN(32, a2));
            if (want & ~F_SEAL_ALL) { ret = -BigInt(EINVAL); break; }
            // only a file that was created ready to be sealed can be, and only through a
            // descriptor that could have written to it.
            if (sealEntry?.type !== 'native' || !sealEntry.sealable) { ret = -BigInt(EPERM); break; }
            if (!this.fdAccess(fd).writable) { ret = -BigInt(EPERM); break; }
            const have = this.fileSeals.get(sealEntry.ofdId) ?? 0;
            if (have & 0x1) { ret = -BigInt(EPERM); break; }   // F_SEAL_SEAL: the set is closed
            this.fileSeals.set(sealEntry.ofdId, have | want);
            ret = 0n;
            break;
          }
          case 1034: { // F_GET_SEALS(fd)
            const getSealEntry = this.getFdEntry(fd);
            if (getSealEntry?.type !== 'native' || !getSealEntry.sealable) { ret = -BigInt(EINVAL); break; }
            ret = BigInt(this.fileSeals.get(getSealEntry.ofdId) ?? 0);
            break;
          }
          case 1032: { // F_GETPIPE_SZ
            const cap = this.pipeCapacityHook?.(fd) ?? null;
            if (cap === null) { ret = -BigInt(EINVAL); break; } // not a pipe
            ret = BigInt(cap);
            break;
          }
          case 1031: { // F_SETPIPE_SZ
            const cap = this.pipeCapacityHook?.(fd) ?? null;
            if (cap === null) { ret = -BigInt(EINVAL); break; } // not a pipe
            // the size is UNSIGNED, and anything below a page becomes a page rather than an error:
            // F_SETPIPE_SZ with 0 is how a program asks for the smallest pipe there is. only a
            // request larger than the whole address space is refused.
            const want = Number(BigInt.asUintN(32, a2));
            if (want > 0x8000_0000) { ret = -BigInt(EINVAL); break; }
            // linux rounds the request UP to a power of two, at least one page
            let rounded = 4096;
            while (rounded < want) rounded *= 2;
            // fs.pipe-max-size is a real ceiling and asking past it is EPERM, not a smaller pipe.
            // silently handing back less than was asked for is the worst of both: the caller is
            // told it succeeded and then sizes its writes by a number it never agreed to.
            const max = this.pipeMaxHook?.get() ?? cap;
            if (rounded > max && !this.hasCap(CAP_SYS_RESOURCE)) { ret = -BigInt(EPERM); break; }
            const granted = this.pipeResizeHook?.(fd, Math.min(rounded, max));
            if (granted === undefined) { ret = -BigInt(EINVAL); break; }
            // pass the ring's own errno through. EBUSY ("there is data in the pipe") and EINVAL
            // ("that size is not allowed") are different problems with different fixes, and
            // collapsing them tells the caller to go looking in the wrong place.
            ret = granted < 0 ? -BigInt(-granted) : BigInt(granted);
            break;
          }
          default:
            ret = -BigInt(EINVAL);
        }
        break;
      }

      // ── Misc ──
      case 278: { // getrandom(buf, buflen, flags)
        // GRND_NONBLOCK and GRND_RANDOM are the only flags there are; the pool is always ready
        // here, so neither changes what happens. a buffer that cannot be written is EFAULT --
        // filling it unchecked wrote random bytes into whatever the address happened to name.
        const GRND_NONBLOCK = 1, GRND_RANDOM = 2, GRND_INSECURE = 4;
        if (Number(BigInt.asIntN(32, a2)) & ~(GRND_NONBLOCK | GRND_RANDOM | GRND_INSECURE)) { ret = -BigInt(EINVAL); break; }
        const len = Number(a1);
        if (len < 0) { ret = -BigInt(EINVAL); break; }
        if (len === 0) { ret = 0n; break; }
        if (this.ptrWriteFaults(bytes, Number(a0), len)) { ret = -BigInt(EFAULT); break; }
        const buf = Number(a0) + GUEST_MEM_BASE;
        for (let i = 0; i < len; i++) {
          bytes[buf + i] = (Math.random() * 256) | 0;
        }
        ret = BigInt(len);
        break;
      }

      case 98: { // futex(uaddr, op, val, timeout, uaddr2, val3)
        // Per Linux spec: fast userspace mutex operations
        const futexAddr = Number(a0) + GUEST_MEM_BASE;
        const futexOp = Number(a1) & 0x7F; // strip FUTEX_PRIVATE_FLAG (128)
        const futexVal = Number(a2);

        switch (futexOp) {
          case 0:  // FUTEX_WAIT
          case 9:  // FUTEX_WAIT_BITSET: real cross-thread blocking is handled in worker.ts
            ret = -BigInt(11); // EAGAIN: single-threaded handler has no one to wake it
            break;
          case 1:  // FUTEX_WAKE
          case 10: // FUTEX_WAKE_BITSET
            ret = 0n; // no in-handler waiters (threaded futex lives in worker.ts)
            break;
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
        const pR = this.ruid, pE = this.euid, pS = this.suid;
        if (this.euid === 0) {
          this.ruid = this.euid = this.suid = uid; this.publishCreds();
          this.recomputeCapsAfterUid(pR, pE, pS);
          ret = 0n;
        }
        else if (uid === this.ruid || uid === this.suid) {
          this.euid = uid; this.publishCreds();
          this.recomputeCapsAfterUid(pR, pE, pS);
          ret = 0n;
        }
        else ret = -BigInt(EPERM);
        break;
      }
      case 144: { // setgid(gid)
        // the group setters had no permission checks and, worse, never carried the change into
        // the FILESYSTEM identity -- so a process that changed its group went on creating files
        // owned by the group it used to have.
        const gid = Number(BigInt.asIntN(32, a0));
        if (this.euid === 0) { this.rgid = this.egid = this.sgid = gid; }
        else if (gid === this.rgid || gid === this.sgid) { this.egid = gid; }
        else { ret = -BigInt(EPERM); break; }
        this.publishCreds();
        ret = 0n;
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
        const preR = this.ruid, preE = this.euid, preS = this.suid;
        if (r !== -1 && r >>> 0 !== unchanged) this.ruid = r;
        if (e !== -1 && e >>> 0 !== unchanged) this.euid = e;
        // a real change, or an effective one to something that was not already the REAL uid,
        // fixes the saved uid at the new effective one. the second half was missing, which left a
        // process still entitled to a uid the saved slot no longer named.
        if ((r !== -1 && r >>> 0 !== unchanged) ||
            (e !== -1 && e >>> 0 !== unchanged && e !== preR)) this.suid = this.euid;
        this.publishCreds();
        this.recomputeCapsAfterUid(preR, preE, preS);
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
        const p3R = this.ruid, p3E = this.euid, p3S = this.suid;
        if (set(r)) this.ruid = r;
        if (set(e)) this.euid = e;
        if (set(s)) this.suid = s;
        this.publishCreds();
        this.recomputeCapsAfterUid(p3R, p3E, p3S);
        ret = 0n;
        break;
      }
      case 143: { // setregid(rgid, egid)
        const set143 = (v: number) => v !== -1 && (v >>> 0) !== 0xffffffff;
        const r = Number(BigInt.asIntN(32, a0)), e = Number(BigInt.asIntN(32, a1));
        if (this.euid !== 0) {
          // the real group may become the real or the effective one; the effective one may also
          // become the saved one. anything else needs CAP_SETGID.
          if (set143(r) && r !== this.rgid && r !== this.egid) { ret = -BigInt(EPERM); break; }
          if (set143(e) && e !== this.rgid && e !== this.egid && e !== this.sgid) { ret = -BigInt(EPERM); break; }
        }
        const preRgid = this.rgid;
        if (set143(r)) this.rgid = r;
        if (set143(e)) this.egid = e;
        // a real change, or an effective one to something that was not already the REAL group,
        // fixes the saved group at the new effective one -- which is what stops a process from
        // switching back later. comparing against the old EFFECTIVE group instead loses the saved
        // group a process is still entitled to, and the next switch back to it is refused.
        if (set143(r) || (set143(e) && e !== preRgid)) this.sgid = this.egid;
        this.publishCreds();
        ret = 0n;
        break;
      }
      case 149: { // setresgid(rgid, egid, sgid)
        const set = (v: number) => v !== -1 && (v >>> 0) !== 0xffffffff;
        const r = Number(BigInt.asIntN(32, a0)), e = Number(BigInt.asIntN(32, a1)), s = Number(BigInt.asIntN(32, a2));
        if (this.euid !== 0) {
          const allowed = (v: number) => v === this.rgid || v === this.egid || v === this.sgid;
          if ((set(r) && !allowed(r)) || (set(e) && !allowed(e)) || (set(s) && !allowed(s))) {
            ret = -BigInt(EPERM); break;
          }
        }
        if (set(r)) this.rgid = r;
        if (set(e)) this.egid = e;
        if (set(s)) this.sgid = s;
        this.publishCreds();
        ret = 0n;
        break;
      }
      // setfsuid / setfsgid: the identity the kernel uses for FILESYSTEM access, separate from the
      // effective one so a server can act as a client for file checks without becoming that client
      // for signals. these used to answer with the euid and change nothing, so setfsuid(-1) -- the
      // query form -- reported the wrong identity and setting one had no effect at all.
      //
      // neither ever reports an error: an unpermitted change is silently ignored, and the caller
      // learns what happened by calling again with -1 and comparing.
      case 151: {
        const prevFsuid = this.fsuid;
        const wantUid = Number(BigInt.asUintN(32, a0));
        if (wantUid !== 0xffffffff &&
            (this.euid === 0 || wantUid === this.ruid || wantUid === this.euid ||
             wantUid === this.suid || wantUid === this.fsuid)) {
          this.fsuid = wantUid;
        }
        ret = BigInt(prevFsuid);
        break;
      }
      case 152: {
        const prevFsgid = this.fsgid;
        const wantGid = Number(BigInt.asUintN(32, a0));
        if (wantGid !== 0xffffffff &&
            (this.euid === 0 || wantGid === this.rgid || wantGid === this.egid ||
             wantGid === this.sgid || wantGid === this.fsgid)) {
          this.fsgid = wantGid;
        }
        ret = BigInt(prevFsgid);
        break;
      }
      case 159: { // setgroups(size, list)
        const sgSize = Number(BigInt.asIntN(32, a0));
        if (sgSize < 0 || sgSize > 65536) { ret = -BigInt(EINVAL); break; }   // NGROUPS_MAX
        if (!this.hasCap(CAP_SETGID)) { ret = -BigInt(EPERM); break; }
        if (sgSize > 0 && this.ptrFaults(bytes, Number(a1), sgSize * 4)) { ret = -BigInt(EFAULT); break; }
        const next: number[] = [];
        for (let i = 0; i < sgSize; i++) {
          next.push(view.getUint32(Number(a1) + GUEST_MEM_BASE + i * 4, true));
        }
        this.groups = next;
        ret = 0n;
        break;
      }
      // ── Networking (via Swift bridge) ──

      case 198: { // socket(domain, type, protocol)
        const domain = Number(a0);   // AF_INET=2, AF_INET6=10, AF_UNIX=1
        const sockType = Number(a1) & 0xFF; // SOCK_STREAM=1, SOCK_DGRAM=2 (strip SOCK_CLOEXEC|SOCK_NONBLOCK)
        const protocol = Number(a2);

        // AF_UNIX never reaches here -- the worker answers it out of the kernel's own ring
        // machinery, because a unix socket never touches the network. anything else is either a
        // family we route over the bridge or one this kernel does not have.
        const sockErr = sockTripleErrno(domain, Number(BigInt.asIntN(32, a1)), protocol, this.euid);
        if (sockErr !== 0) { ret = -BigInt(sockErr); break; }

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
              this.fdTable.set(socketFd, { type: 'socket', socketId: resp.ret, domain, sockType, protocol });
              if (Number(a1) & 0o2000000) this.fdCloexec.add(socketFd); // SOCK_CLOEXEC
              if (Number(a1) & 0o4000) this.fdFlags.set(socketFd, (this.fdFlags.get(socketFd) ?? 0) | O_NONBLOCK);
              this.nt(`socket dom=${domain} type=${sockType} -> fd=${socketFd}`);
              (this as any)._netActive = true;
              cpu.setReg(0, BigInt(socketFd));
            }
          },
        };
      }

      case 200: { // bind(sockfd, addr, addrlen)
        const bindSock = this.sockFd(Number(a0));
        if ('err' in bindSock) { ret = -BigInt(bindSock.err); break; }
        if (bindSock.e.type !== 'socket') { ret = -BigInt(EOPNOTSUPP); break; }
        const bindEntry = bindSock.e;
        const bindLen = Number(BigInt.asIntN(32, a2));
        // an address shorter than its own family field, or longer than any sockaddr_storage, is
        // not an address. so is a sockaddr_in that does not reach past the port.
        if (bindLen < 2 || bindLen > 128) { ret = -BigInt(EINVAL); break; }
        if (bindEntry.domain === 2 && bindLen < 16) { ret = -BigInt(EINVAL); break; }
        if (bindEntry.domain === 10 && bindLen < 28) { ret = -BigInt(EINVAL); break; }
        if (this.ptrFaults(bytes, Number(a1), bindLen)) { ret = -BigInt(EFAULT); break; }
        const addrBuf = this.parseSockaddr(bytes, Number(a1), bindLen);
        if (addrBuf.family !== bindEntry.domain && !(bindEntry.domain === 10 && addrBuf.family === 2)) {
          ret = -BigInt(EAFNOSUPPORT); break;
        }
        // the first 1024 ports belong to root. binding one without CAP_NET_BIND_SERVICE is EACCES,
        // which is the whole reason a program drops privileges only after it has bound its port.
        if (addrBuf.port > 0 && addrBuf.port < 1024 && !this.hasCap(10)) { ret = -BigInt(EACCES); break; }
        return {
          status: 'bridge', method: 'net_bind',
          args: { socketId: bindEntry.socketId, ...addrBuf },
          apply: (resp: any) => {
            // the port that came back is the one actually bound, which for a request of 0 is the
            // one the system picked.
            if (resp.ret >= 0) {
              const bound = resp._meta ? (resp._meta[0] | 0) : 0;
              bindEntry.localAddr = { ...addrBuf, port: bound || addrBuf.port };
            }
            cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
          },
        };
      }

      case 201: { // listen(sockfd, backlog)
        const listenSock = this.sockFd(Number(a0));
        if ('err' in listenSock) { ret = -BigInt(listenSock.err); break; }
        if (listenSock.e.type !== 'socket') { ret = -BigInt(EOPNOTSUPP); break; }
        const listenEntry = listenSock.e;
        // listen is a connection-oriented idea. a datagram socket has no connections to queue,
        // and linux says so rather than pretending the call did something.
        if (listenEntry.sockType !== 1 && listenEntry.sockType !== 5) { ret = -BigInt(EOPNOTSUPP); break; }
        return {
          status: 'bridge', method: 'net_listen',
          args: { socketId: listenEntry.socketId, backlog: Number(a1) },
          apply: (resp: any) => {
            if (resp.ret >= 0) listenEntry.listening = true;
            cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
          },
        };
      }

      case 202:   // accept(sockfd, addr, addrlen)
      case 242: { // accept4(sockfd, addr, addrlen, flags)
        const acceptEntrySock = this.sockFd(Number(a0));
        if ('err' in acceptEntrySock) { ret = -BigInt(acceptEntrySock.err); break; }
        if (acceptEntrySock.e.type !== 'socket') { ret = -BigInt(EOPNOTSUPP); break; }
        const acceptEntry = acceptEntrySock.e;
        const acFlags = nr === 242 ? Number(BigInt.asIntN(32, a3)) : 0;
        if (acFlags & ~(0o4000 | 0o2000000)) { ret = -BigInt(EINVAL); break; }   // SOCK_NONBLOCK|SOCK_CLOEXEC
        // accepting is a connection-oriented idea: a datagram socket has no queue of them.
        if (acceptEntry.sockType !== 1 && acceptEntry.sockType !== 5) { ret = -BigInt(EOPNOTSUPP); break; }
        // and a socket that never listened has no queue either, however well-formed the call is.
        // this used to block forever instead, which is what took the test down with a timeout.
        if (!acceptEntry.listening) { ret = -BigInt(EINVAL); break; }
        const acAddrPtr = Number(a1);
        const acLenPtr = Number(a2);
        const newFd = this.allocFd();
        return {
          status: 'bridge', method: 'net_accept',
          args: { socketId: acceptEntry.socketId },
          apply: (resp: any) => {
            if (resp.ret === -11 && !this.isNonBlockingFd(Number(a0))) {
              // nobody is waiting yet. a blocking accept keeps asking rather than holding the
              // bridge open, which is what makes it interruptible by a signal.
              this.retryAfterBridge = true;
              return;
            }
            if (resp.ret < 0) {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              return;
            }
            const peer = this.peerFromMeta(resp);
            this.fdTable.set(newFd, { type: 'socket', socketId: resp.ret, domain: acceptEntry.domain,
                                      sockType: acceptEntry.sockType,
                                      remoteAddr: peer ?? undefined,
                                      localAddr: acceptEntry.localAddr });
            if (acFlags & 0o2000000) this.fdCloexec.add(newFd);
            if (acFlags & 0o4000) this.fdFlags.set(newFd, (this.fdFlags.get(newFd) ?? 0) | O_NONBLOCK);
            // who connected, if the caller asked. the kernel copies this out AFTER the descriptor
            // exists, and a buffer it cannot write to costs the connection.
            if (acAddrPtr !== 0 && peer) {
              const e = this.putSockaddr(bytes, view, acAddrPtr, acLenPtr, peer);
              if (e < 0) {
                this.closeSocketHook?.(resp.ret);
                this.fdTable.delete(newFd);
                cpu.setReg(0, BigInt.asUintN(64, BigInt(e)));
                return;
              }
            }
            cpu.setReg(0, BigInt(newFd));
          },
        };
      }

      case 203: { // connect(sockfd, addr, addrlen)
        const connEntrySock = this.sockFd(Number(a0));
        if ('err' in connEntrySock) { ret = -BigInt(connEntrySock.err); break; }
        if (connEntrySock.e.type !== 'socket') { ret = -BigInt(EOPNOTSUPP); break; }
        const connEntry = connEntrySock.e;
        // the address has to BE one before it is dialled: too short for its own family, longer than
        // any sockaddr_storage, unreadable, or of a family this socket cannot reach. none of these
        // were checked, so every one of them became a connection attempt that timed out instead.
        const connLen = Number(BigInt.asIntN(32, a2));
        if (connLen < 2 || connLen > 128) { ret = -BigInt(EINVAL); break; }
        if (Number(a1) === 0 || this.ptrFaults(bytes, Number(a1), connLen)) { ret = -BigInt(EFAULT); break; }
        const connFamily = bytes[Number(a1) + GUEST_MEM_BASE] | (bytes[Number(a1) + GUEST_MEM_BASE + 1] << 8);
        // AF_UNSPEC dissolves the association rather than making one, which is how a datagram
        // socket is un-connected and how IPV6_ADDRFORM is exercised.
        if (connFamily === 0) {
          connEntry.remoteAddr = undefined;
          ret = 0n;
          break;
        }
        if (connFamily !== connEntry.domain && !(connEntry.domain === 10 && connFamily === 2)) {
          ret = -BigInt(EAFNOSUPPORT); break;
        }
        if (connFamily === 2 && connLen < 16) { ret = -BigInt(EINVAL); break; }
        if (connFamily === 10 && connLen < 28) { ret = -BigInt(EINVAL); break; }
        // a stream socket connects once. asking again is EISCONN, not a second connection.
        if (connEntry.sockType === 1 && connEntry.remoteAddr) { ret = -BigInt(EISCONN); break; }
        const connAddr = this.parseSockaddr(bytes, Number(a1), Number(a2));
        // connecting to the wildcard address means "this machine", which is what linux turns it
        // into. left as 0.0.0.0 it names no host at all and the connection simply times out.
        if (connAddr.host === '0.0.0.0') connAddr.host = '127.0.0.1';
        if (connAddr.host === '::') connAddr.host = '::1';
        return {
          status: 'bridge', method: 'net_connect',
          args: { socketId: connEntry.socketId, ...connAddr },
          apply: (resp: any) => {
            this.nt(`connect fd=${Number(a0)} ${connAddr.host}:${connAddr.port} -> ${resp.ret}`);
            if (resp.ret >= 0) connEntry.remoteAddr = connAddr;
            cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
          },
        };
      }

      case 204: { // getsockname(sockfd, addr, addrlen)
        const gsnEntrySock = this.sockFd(Number(a0));
        if ('err' in gsnEntrySock) { ret = -BigInt(gsnEntrySock.err); break; }
        const gsnEntry = gsnEntrySock.e;
        // both pointers are the caller's to get right, and neither was checked: a null buffer or
        // a null length wrote through it anyway, and a negative length was accepted.
        const gsnLenPtr = Number(a2);
        if (gsnLenPtr === 0 || this.ptrWriteFaults(bytes, gsnLenPtr, 4)) { ret = -BigInt(EFAULT); break; }
        const gsnWant = view.getInt32(gsnLenPtr + GUEST_MEM_BASE, true);
        if (gsnWant < 0) { ret = -BigInt(EINVAL); break; }
        // a unix-domain socket that was never bound has an address of nothing but its family,
        // which is two bytes -- and that IS its name, not a missing one.
        const family = gsnEntry.type === 'socket' ? gsnEntry.domain : 1;
        // the local address is whatever bind() was given; an unbound socket has the wildcard. an
        // ipv6 socket used to get nothing but its family here, so the port it had just been given
        // read back as zero and every program that binds a port and connects to it went nowhere.
        const gsnAddr = gsnEntry.type === 'socket'
          ? (gsnEntry.localAddr ?? { family, host: family === 10 ? '::' : '0.0.0.0', port: 0 })
          : { family: 1, host: '', port: 0 };
        const gsnErr = this.putSockaddr(bytes, view, Number(a1), gsnLenPtr,
                                        { ...gsnAddr, family });
        if (gsnErr < 0) { ret = -BigInt(-gsnErr); break; }
        ret = 0n;
        break;
      }

      case 205: { // getpeername(sockfd, addr, addrlen)
        const gpnSock = this.sockFd(Number(a0));
        if ('err' in gpnSock) { ret = -BigInt(gpnSock.err); break; }
        const gpnLenPtr = Number(a2);
        if (gpnLenPtr === 0 || this.ptrWriteFaults(bytes, gpnLenPtr, 4)) { ret = -BigInt(EFAULT); break; }
        const gpnWant = view.getInt32(gpnLenPtr + GUEST_MEM_BASE, true);
        if (gpnWant < 0) { ret = -BigInt(EINVAL); break; }
        // a unix-domain pair IS connected, and its peer's address is the unnamed one: two bytes
        // of family. only a socket with nobody on the other end has no name to report.
        const gpnUnix = gpnSock.e.type === 'unixsock';
        const gpnPeer = gpnUnix
          ? (this.peerCredHook?.(Number(a0)) ? { family: 1, host: '', port: 0 } : undefined)
          : (gpnSock.e.type === 'socket' ? gpnSock.e.remoteAddr : undefined);
        if (!gpnPeer) { ret = -107n; break; } // ENOTCONN
        const gpnErr = this.putSockaddr(bytes, view, Number(a1), gpnLenPtr, gpnPeer);
        if (gpnErr < 0) { ret = -BigInt(-gpnErr); break; }
        ret = 0n;
        break;
      }

      case 206: { // sendto(sockfd, buf, len, flags, dest_addr, addrlen)
        const stEntrySock = this.sockFd(Number(a0));
        if ('err' in stEntrySock) { ret = -BigInt(stEntrySock.err); break; }
        if (stEntrySock.e.type !== 'socket') { ret = -BigInt(EOPNOTSUPP); break; }
        const stEntry = stEntrySock.e;
        const sendBuf = Number(a1);
        const sendLen = Number(BigInt.asUintN(64, a2));
        const sendFlags = Number(BigInt.asIntN(32, a3));
        // out-of-band data is a stream idea; a datagram socket has no urgent pointer.
        if ((sendFlags & 1) && stEntry.sockType !== 1) { ret = -BigInt(EOPNOTSUPP); break; }
        if (sendLen > 0 && this.ptrFaults(bytes, sendBuf, sendLen)) { ret = -BigInt(EFAULT); break; }
        // a datagram is sent whole or not at all, and one that cannot fit in a packet is refused
        // rather than truncated. ipv6 has 20 more bytes to spend than ipv4.
        if (stEntry.sockType === 2 && sendLen > (stEntry.domain === 10 ? 65527 : 65507)) {
          ret = -BigInt(EMSGSIZE); break;
        }
        // writing to a stream that has nowhere to go: no peer, or the sending half already shut
        // down. linux answers EPIPE and raises SIGPIPE, which is what makes a broken pipe fatal
        // to a program that did not ask to hear about it.
        if (stEntry.sockType === 1 && (!stEntry.remoteAddr || stEntry.shutWr)) {
          if (!(sendFlags & 0x4000)) this.raiseSignal(13); // unless MSG_NOSIGNAL
          ret = -BigInt(EPIPE); break;
        }
        const sendData = bytes.slice(sendBuf + GUEST_MEM_BASE, sendBuf + GUEST_MEM_BASE + sendLen);
        const b64 = bytesToBase64(sendData);
        // If dest_addr is provided (connectionless sendto), use net_sendto
        const destAddrPtr = Number(a4);
        const destAddrLen = Number(BigInt.asIntN(32, a5));
        // a destination has to be an address: long enough for its family, and readable. neither was
        // checked, so a garbage pointer and a two-byte sockaddr were both sent somewhere.
        // a connected stream ignores the destination entirely, however malformed it is: the
        // connection already says where the bytes go.
        const stConnectedStream = stEntry.sockType === 1 && !!stEntry.remoteAddr;
        if (!stConnectedStream && (destAddrPtr !== 0 || destAddrLen !== 0)) {
          if (destAddrLen < 2 || destAddrLen > 128) { ret = -BigInt(EINVAL); break; }
          if (destAddrPtr === 0 || this.ptrFaults(bytes, destAddrPtr, destAddrLen)) { ret = -BigInt(EFAULT); break; }
          const destFamily = bytes[destAddrPtr + GUEST_MEM_BASE] | (bytes[destAddrPtr + GUEST_MEM_BASE + 1] << 8);
          if (destFamily === 2 && destAddrLen < 16) { ret = -BigInt(EINVAL); break; }
          if (destFamily === 10 && destAddrLen < 28) { ret = -BigInt(EINVAL); break; }
          if (destFamily !== stEntry.domain && !(stEntry.domain === 10 && destFamily === 2)) {
            ret = -BigInt(EAFNOSUPPORT); break;
          }
        }
        if (!stConnectedStream && destAddrPtr !== 0 && destAddrLen > 0) {
          const dest = this.parseSockaddr(bytes, destAddrPtr, destAddrLen);
          if (!stEntry.remoteAddr) stEntry.remoteAddr = dest;
          return {
            status: 'bridge', method: 'net_sendto',
            args: { socketId: stEntry.socketId, data: b64, host: dest.host, port: dest.port,
                    more: (sendFlags & 0x8000) ? 1 : 0 },
            apply: (resp: any) => {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            },
          };
        }
        // Connected socket — use net_write
        return {
          status: 'bridge', method: 'net_write',
          args: { socketId: stEntry.socketId, data: b64, more: (sendFlags & 0x8000) ? 1 : 0 },
          apply: (resp: any) => {
            cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
          },
        };
      }

      case 207: { // recvfrom(sockfd, buf, len, flags, src_addr, addrlen)
        const rfEntrySock = this.sockFd(Number(a0));
        if ('err' in rfEntrySock) { ret = -BigInt(rfEntrySock.err); break; }
        if (rfEntrySock.e.type !== 'socket') { ret = -BigInt(EOPNOTSUPP); break; }
        const rfEntry = rfEntrySock.e;
        const recvBuf = Number(a1);
        const recvLen = Math.min(Number(a2), BRIDGE_READ_MAX); // slot-safe cap (see net_read in read())
        const rfFlags = Number(BigInt.asIntN(32, a3));
        // there is no urgent data to read, and no error queue to read from. both used to be
        // accepted silently, so a program asking for either got ordinary data instead of an answer.
        if (rfFlags & 1) { ret = -BigInt(EINVAL); break; }        // MSG_OOB
        if (rfFlags & 0x2000) { ret = -BigInt(EAGAIN); break; }   // MSG_ERRQUEUE
        if (recvLen > 0 && this.ptrWriteFaults(bytes, recvBuf, recvLen)) { ret = -BigInt(EFAULT); break; }
        const rfAddrPtr = Number(a4);
        const rfLenPtr = Number(a5);
        // the caller's buffer length is checked even when nothing will be written into it.
        if (rfLenPtr !== 0) {
          if (this.ptrWriteFaults(bytes, rfLenPtr, 4)) { ret = -BigInt(EFAULT); break; }
          if (view.getInt32(rfLenPtr + GUEST_MEM_BASE, true) < 0) { ret = -BigInt(EINVAL); break; }
        }
        // a socket whose receiving half was shut down reads end of file, without asking the host.
        if (rfEntry.shutRd) { ret = 0n; break; }
        // a connected stream has no per-message sender to report: linux leaves the buffer alone and
        // says the name is zero bytes long. only a datagram answers with an address.
        const rfWantFrom = rfEntry.sockType === 2;
        const rfNonblock = this.isNonBlockingFd(Number(a0)) || (rfFlags & 0x40) !== 0; // MSG_DONTWAIT
        return {
          status: 'bridge', method: 'net_read',
          args: { socketId: rfEntry.socketId, count: recvLen,
                  nonblock: rfNonblock ? 1 : 0, peek: (rfFlags & 2) ? 1 : 0 },
          apply: (resp: any) => {
            if (resp.ret === -11 && !rfNonblock) {
              this.retryAfterBridge = true;
              return;
            }
            if (resp.ret < 0) {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              return;
            }
            const from = this.peerFromMeta(resp) ?? rfEntry.remoteAddr;
            if (rfLenPtr !== 0) {
              if (rfWantFrom && rfAddrPtr !== 0 && from) {
                const e = this.putSockaddr(bytes, view, rfAddrPtr, rfLenPtr, from);
                if (e < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(e))); return; }
              } else {
                view.setUint32(rfLenPtr + GUEST_MEM_BASE, 0, true);
              }
            }
            if (resp.bytes) {
              bytes.set(resp.bytes, recvBuf + GUEST_MEM_BASE);
              cpu.setReg(0, BigInt(resp.bytes.length));
            } else {
              cpu.setReg(0, BigInt(resp.ret));
            }
          },
        };
      }

      // setsockopt / getsockopt. these used to answer 0 to everything and write a zero back,
      // which meant a program could not tell a supported option from one this kernel has never
      // heard of -- and every wrong call looked like it worked.
      case 208: { // setsockopt(sockfd, level, optname, optval, optlen)
        const soSock = this.sockFd(Number(a0));
        if ('err' in soSock) { ret = -BigInt(soSock.err); break; }
        const soLevel = Number(BigInt.asIntN(32, a1));
        const soName = Number(BigInt.asIntN(32, a2));
        const soLen = Number(BigInt.asIntN(32, a4));
        if (!sockOptKnown(soLevel, soName)) { ret = -BigInt(ENOPROTOOPT); break; }
        // every option this kernel carries is at least an int wide, so a shorter buffer cannot
        // hold one. a null buffer is a fault, not a zero.
        if (soLen < 4) { ret = -BigInt(EINVAL); break; }
        if (Number(a3) === 0 || this.ptrFaults(bytes, Number(a3), soLen)) { ret = -BigInt(EFAULT); break; }
        // SO_TYPE, SO_DOMAIN, SO_PROTOCOL and SO_ACCEPTCONN describe the socket; they are read-only.
        if (soLevel === 1 && (soName === 3 || soName === 30 || soName === 38 || soName === 39)) {
          ret = -BigInt(ENOPROTOOPT); break;
        }
        // multicast membership is a LIST on the socket, not a value. we carry no multicast traffic,
        // but the list itself is observable: dropping a group this socket never joined is
        // EADDRNOTAVAIL, and that is exactly what a program checks after accept(2) to be sure the
        // listener's memberships were not copied onto the accepted socket (CVE-2017-8890).
        const IP_ADD_MEMBERSHIP = 35, IP_DROP_MEMBERSHIP = 36;
        if (soLevel === 0 && (soName === IP_ADD_MEMBERSHIP || soName === IP_DROP_MEMBERSHIP)) {
          if (soLen < 8) { ret = -BigInt(EINVAL); break; }
          const b = Number(a3) + GUEST_MEM_BASE;
          // struct ip_mreq(n): the group, then the local interface (address or index).
          const group = view.getUint32(b, false);
          const iface = view.getUint32(b + 4, false);
          const key = `${group}:${iface}`;
          if (!soSock.e.mcast) soSock.e.mcast = new Set<string>();
          if (soName === IP_ADD_MEMBERSHIP) {
            if (soSock.e.mcast.has(key)) { ret = -BigInt(EADDRINUSE); break; }
            soSock.e.mcast.add(key);
          } else {
            if (!soSock.e.mcast.delete(key)) { ret = -BigInt(EADDRNOTAVAIL); break; }
          }
          ret = 0n;
          break;
        }
        if (!soSock.e.opts) soSock.e.opts = new Map();
        soSock.e.opts.set(soLevel * 65536 + soName, view.getInt32(Number(a3) + GUEST_MEM_BASE, true));
        ret = 0n;
        break;
      }

      case 209: { // getsockopt(sockfd, level, optname, optval, optlen)
        const goSock = this.sockFd(Number(a0));
        if ('err' in goSock) { ret = -BigInt(goSock.err); break; }
        const goLevel = Number(BigInt.asIntN(32, a1));
        const goName = Number(BigInt.asIntN(32, a2));
        if (!sockOptLevel(goLevel)) { ret = -BigInt(EOPNOTSUPP); break; }
        if (!sockOptKnown(goLevel, goName)) { ret = -BigInt(ENOPROTOOPT); break; }
        if (Number(a4) === 0 || this.ptrWriteFaults(bytes, Number(a4), 4)) { ret = -BigInt(EFAULT); break; }
        const goLen = view.getInt32(Number(a4) + GUEST_MEM_BASE, true);
        if (goLen < 0) { ret = -BigInt(EINVAL); break; }
        if (Number(a3) === 0 || this.ptrWriteFaults(bytes, Number(a3), Math.min(Math.max(goLen, 1), 4))) {
          ret = -BigInt(EFAULT); break;
        }
        if (goLevel === 1 && goName === 17) {   // SO_PEERCRED: struct ucred { pid, uid, gid }
          if (goLen < 12) { ret = -BigInt(EINVAL); break; }
          if (this.ptrWriteFaults(bytes, Number(a3), 12)) { ret = -BigInt(EFAULT); break; }
          // SO_PEERCRED names the process on the OTHER end. reporting our own credentials made
          // every connection look like it came from us, which is the one thing this option exists
          // to tell apart. an unconnected socket has no peer, and reports zeroes.
          const peer = this.peerCredHook?.(Number(a0)) ?? null;
          view.setInt32(Number(a3) + GUEST_MEM_BASE, peer?.pid ?? 0, true);
          view.setUint32(Number(a3) + GUEST_MEM_BASE + 4, peer?.uid ?? 0, true);
          view.setUint32(Number(a3) + GUEST_MEM_BASE + 8, peer?.gid ?? 0, true);
          view.setInt32(Number(a4) + GUEST_MEM_BASE, 12, true);
          ret = 0n;
          break;
        }
        // the four that describe the socket itself have real answers; the rest report what was
        // last set, or zero.
        const e = goSock.e;
        let value: number;
        if (goLevel === 1 && goName === 3) value = e.sockType ?? 1;
        else if (goLevel === 1 && goName === 4) value = 0;                       // SO_ERROR
        else if (goLevel === 1 && goName === 30) value = e.listening ? 1 : 0;    // SO_ACCEPTCONN
        else if (goLevel === 1 && goName === 38) value = (e.type === 'socket' ? e.protocol : 0) ?? 0;
        else if (goLevel === 1 && goName === 39) value = e.type === 'socket' ? e.domain : 1; // SO_DOMAIN
        else value = e.opts?.get(goLevel * 65536 + goName) ?? 0;
        view.setInt32(Number(a3) + GUEST_MEM_BASE, value, true);
        view.setInt32(Number(a4) + GUEST_MEM_BASE, 4, true);
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
          if (!pidExists(killPid)) { ret = -BigInt(ESRCH); break; }
          if (killSig !== 0 && killPid === this.currentPid) this.raiseSignal(killSig);
          ret = 0n;
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

      case 158: { // getgroups(size, list)
        // the supplementary groups are real now: they were a hardcoded "you are in group 0", so
        // setgroups changed nothing and a file readable only through a group was unreachable.
        const gSize = Number(BigInt.asIntN(32, a0));
        if (gSize < 0) { ret = -BigInt(EINVAL); break; }
        if (gSize === 0) { ret = BigInt(this.groups.length); break; } // the count alone
        if (gSize < this.groups.length) { ret = -BigInt(EINVAL); break; }
        if (this.ptrWriteFaults(bytes, Number(a1), this.groups.length * 4)) { ret = -BigInt(EFAULT); break; }
        for (let i = 0; i < this.groups.length; i++) {
          view.setUint32(Number(a1) + GUEST_MEM_BASE + i * 4, this.groups[i] >>> 0, true);
        }
        ret = BigInt(this.groups.length);
        break;
      }


      case 115: { // clock_nanosleep (NOT sched_getscheduler — that's 120)
        // the REAL sleep is in the worker, which intercepts 101/115 before this handler and parks
        // on the process's wait futex so a signal can interrupt it. this arm is only reached off
        // the device (the host-side test path), where there is no futex to park on.
        ret = 0n;
        break;
      }

      case 261: { // prlimit64(pid, resource, new_limit, old_limit)
        // struct rlimit64: { rlim_cur (8 bytes), rlim_max (8 bytes) }
        const resource = Number(BigInt.asIntN(32, a1));
        if (resource < 0 || resource >= RLIM_NLIMITS) { ret = -BigInt(EINVAL); break; }
        // a pid of 0 means this process. we have no way to reach into another one's limits, and
        // saying so is better than silently answering with our own.
        const rlPid = Number(BigInt.asIntN(32, a0));
        if (rlPid !== 0 && rlPid !== this.currentPid) { ret = -BigInt(ESRCH); break; }
        const rlNewPtr = Number(a2);
        const rlOldPtr = Number(a3);
        if (rlOldPtr !== 0 && this.ptrFaults(bytes, rlOldPtr, 16)) { ret = -BigInt(EFAULT); break; }
        if (rlNewPtr !== 0 && this.ptrFaults(bytes, rlNewPtr, 16)) { ret = -BigInt(EFAULT); break; }
        // validate the new limit BEFORE reporting the old one: a rejected call must leave the
        // caller's old_limit buffer untouched.
        let rlNewCur = 0n;
        if (rlNewPtr !== 0) {
          rlNewCur = view.getBigUint64(rlNewPtr + GUEST_MEM_BASE, true);
          const newMax = view.getBigUint64(rlNewPtr + GUEST_MEM_BASE + 8, true);
          if (rlNewCur > newMax) { ret = -BigInt(EINVAL); break; }
          if (resource === RLIMIT_NOFILE && newMax > BigInt(NR_OPEN)) { ret = -BigInt(EPERM); break; }
          if (newMax > this.rlimitOf(resource).max && !this.hasCap(CAP_SYS_RESOURCE)) { ret = -BigInt(EPERM); break; }
        }
        if (rlOldPtr !== 0) {
          const lim = this.rlimitOf(resource);
          view.setBigUint64(rlOldPtr + GUEST_MEM_BASE, lim.cur, true);
          view.setBigUint64(rlOldPtr + GUEST_MEM_BASE + 8, lim.max, true);
        }
        if (rlNewPtr !== 0) this.setRlimit(resource, rlNewCur, view.getBigUint64(rlNewPtr + GUEST_MEM_BASE + 8, true));
        ret = 0n;
        break;
      }

      case 233: { // madvise(addr, length, advice)
        const advice = Number(BigInt.asIntN(32, a2));
        const mvStart = Number(a0);
        const mvLen = Number(BigInt.asIntN(64, a1));
        // the checks, in linux's order. this used to accept absolutely anything and answer 0,
        // including an unaligned address, an advice value that does not exist, and a range with
        // nothing mapped in it.
        if (mvStart & 0xFFF) { ret = -BigInt(EINVAL); break; }
        if (mvLen < 0) { ret = -BigInt(EINVAL); break; }
        if (!MADV_VALID.has(advice)) { ret = -BigInt(EINVAL); break; }
        // poisoning a page is a fault-injection facility and needs CAP_SYS_ADMIN
        if ((advice === 100 || advice === 101) && !this.hasCap(CAP_SYS_ADMIN)) { ret = -BigInt(EPERM); break; }
        if (mvLen === 0) { ret = 0n; break; }
        const mvEnd = (mvStart + mvLen + 0xFFF) & ~0xFFF;
        if (mvEnd <= mvStart) { ret = -BigInt(EINVAL); break; } // wrapped
        if (!this.rangeMapped(mvStart, mvEnd)) { ret = -BigInt(ENOMEM); break; }
        const mvKind = this.rangeKind(mvStart, mvEnd);
        const privAnon = !mvKind.shared && !mvKind.fileBacked;
        let mvBad = false;
        switch (advice) {
          // MADV_DONTNEED refuses a locked range: dropping the pages would unlock it.
          case 4: mvBad = mvKind.locked; break;
          // MADV_FREE and MADV_WIPEONFORK are defined only on private anonymous memory.
          case 8: case 18: mvBad = !privAnon; break;
          // MADV_REMOVE punches a hole, which only shared memory with no file behind it has.
          case 9: mvBad = !mvKind.shared || mvKind.fileBacked; break;
          // KSM merges anonymous private pages and nothing else.
          case 12: case 13: mvBad = !privAnon; break;
          default: break;
        }
        if (mvBad) { ret = -BigInt(EINVAL); break; }
        // MADV_DONTNEED (4) on an anonymous private mapping must make subsequent reads return
        // zero-filled pages. musl's mallocng relies on this to recycle freed slots; leaving stale
        // bytes corrupts its inline chunk metadata. MADV_FREE (8) may also zero lazily, so
        // zeroing eagerly is safe.
        if (advice === 4 || advice === 8) {
          if (mvEnd + GUEST_MEM_BASE <= memory.buffer.byteLength) {
            new Uint8Array(memory.buffer).fill(0, mvStart + GUEST_MEM_BASE, mvEnd + GUEST_MEM_BASE);
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
        if (typeof fchPath !== 'string' || !fchPath.length) { ret = -BigInt(EBADF); break; }
        if (fchEntry?.type === 'native' && fchEntry.isDir === false) { ret = -BigInt(ENOTDIR); break; }
        const fchBridge = this.bridgePath(fchPath);
        if (fchBridge === null) {
          this.cwd = fchPath;
          if (this.vfs) this.vfs.setCwd(fchPath);
          ret = 0n;
          break;
        }
        // moving INTO a directory needs search permission on it, and that is checked now rather
        // than when the descriptor was opened: a directory whose mode changed after the open is
        // no longer somewhere this process may stand.
        return {
          status: 'bridge', method: 'fs_faccessat',
          args: { path: fchBridge, mode: 1 /* X_OK */, eaccess: true },
          apply: (resp: any) => {
            if (resp.ret < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); return; }
            this.cwd = fchPath;
            if (this.vfs) this.vfs.setCwd(fchPath);
            cpu.setReg(0, 0n);
          },
        };
      }

      case 101: { // nanosleep(req, rem)
        // as with clock_nanosleep above: the worker handles this for real before we get here, and
        // this arm exists for the host-side test path. report no remaining time.
        const remAddr = Number(a1);
        if (remAddr !== 0) {
          view.setBigUint64(remAddr + GUEST_MEM_BASE, 0n, true);
          view.setBigUint64(remAddr + GUEST_MEM_BASE + 8, 0n, true);
        }
        ret = 0n;
        break;
      }

      case 124: // sched_yield — yield CPU
        ret = 0n;
        break;

      case 165: { // getrusage(who, usage)
        const RUSAGE_SELF = 0, RUSAGE_CHILDREN = -1, RUSAGE_THREAD = 1;
        const who = Number(BigInt.asIntN(32, a0));
        if (who !== RUSAGE_SELF && who !== RUSAGE_CHILDREN && who !== RUSAGE_THREAD) {
          ret = -BigInt(EINVAL); break;
        }
        if (this.ptrFaults(bytes, Number(a1), 144)) { ret = -BigInt(EFAULT); break; }
        if (who === RUSAGE_CHILDREN) {
          // children contribute nothing to OUR resident set, so maxrss is theirs, not ours: linux
          // reports the largest any reaped child reached.
          this.writeRusage(view, Number(a1), this.childUtimeMs, this.childStimeMs, this.childMaxRssKb);
        } else {
          this.writeRusage(view, Number(a1), this.ownUserMs(), this.ownSysMs(), this.rssKbHook ? this.rssKbHook() : 0);
        }
        ret = 0n;
        break;
      }

      case 167: { // prctl(option, arg2, arg3, arg4, arg5)
        // this used to answer 0 to EVERY option, including options that do not exist and options
        // that are supposed to hand something back. a program that asked for its own name got
        // "jitter" no matter what it had set, and one that probed for a feature by prctl'ing for
        // it was told the feature was there.
        const prOption = Number(BigInt.asIntN(32, a0));
        const prArg2 = Number(BigInt.asIntN(64, a1));
        switch (prOption) {
          case 1: // PR_SET_PDEATHSIG - signal to send us when our parent dies
            if (prArg2 < 0 || prArg2 > 64) { ret = -BigInt(EINVAL); break; }
            this.pdeathSig = prArg2;
            // the kernel is the one that delivers it, long after this process has stopped running
            // guest code, so it has to be published where the kernel can read it.
            this.pdeathHook?.(prArg2);
            ret = 0n;
            break;
          case 2: // PR_GET_PDEATHSIG
            if (this.ptrFaults(bytes, Number(a1), 4)) { ret = -BigInt(EFAULT); break; }
            view.setInt32(Number(a1) + GUEST_MEM_BASE, this.pdeathSig, true);
            ret = 0n;
            break;
          case 3: // PR_GET_DUMPABLE
            ret = BigInt(this.dumpable);
            break;
          case 4: // PR_SET_DUMPABLE
            // SUID_DUMP_ROOT (2) is a state the kernel puts a process into by itself; a process
            // cannot ask for it, and accepting it let a caller claim a dump mode linux refuses.
            if (prArg2 !== 0 && prArg2 !== 1) { ret = -BigInt(EINVAL); break; }
            this.dumpable = prArg2;
            ret = 0n;
            break;
          case 7: // PR_GET_KEEPCAPS
            ret = BigInt(this.keepCaps);
            break;
          case 8: // PR_SET_KEEPCAPS
            if (prArg2 !== 0 && prArg2 !== 1) { ret = -BigInt(EINVAL); break; }
            this.keepCaps = prArg2;
            ret = 0n;
            break;
          case 15: { // PR_SET_NAME - the thread's comm, truncated to 16 bytes including the NUL
            if (this.ptrFaults(bytes, Number(a1), 1)) { ret = -BigInt(EFAULT); break; }
            const raw = readGuestString(bytes, Number(a1));
            this.comm = raw.slice(0, 15);
            this.commHook?.(this.comm);
            ret = 0n;
            break;
          }
          case 16: { // PR_GET_NAME
            if (this.ptrFaults(bytes, Number(a1), 16)) { ret = -BigInt(EFAULT); break; }
            const out = new Uint8Array(16);
            const enc = new TextEncoder().encode(this.comm.slice(0, 15));
            out.set(enc.subarray(0, 15));
            new Uint8Array(memory.buffer).set(out, Number(a1) + GUEST_MEM_BASE);
            ret = 0n;
            break;
          }
          case 21: // PR_GET_SECCOMP - not in any filter mode
            ret = 0n;
            break;
          case 22: // PR_SET_SECCOMP
            ret = -BigInt(EINVAL); // no seccomp support, and pretending otherwise is worse
            break;
          case 23: // PR_CAPBSET_READ
            if (prArg2 < 0 || prArg2 > 40) { ret = -BigInt(EINVAL); break; }
            ret = this.hasCap(prArg2) ? 1n : 0n;
            break;
          case 24: // PR_CAPBSET_DROP
            if (prArg2 < 0 || prArg2 > 40) { ret = -BigInt(EINVAL); break; }
            if (!this.hasCap(CAP_SETPCAP)) { ret = -BigInt(EPERM); break; }
            this.capPerm &= ~(1n << BigInt(prArg2));
            this.capEff &= ~(1n << BigInt(prArg2));
            ret = 0n;
            break;
          case 29: // PR_SET_TIMERSLACK - 0 restores the inherited default
            this.timerSlackNs = prArg2 === 0 ? this.defaultTimerSlackNs : prArg2;
            ret = 0n;
            break;
          case 30: // PR_GET_TIMERSLACK
            ret = BigInt(this.timerSlackNs);
            break;
          case 36: // PR_SET_CHILD_SUBREAPER
            this.childSubreaper = prArg2 !== 0 ? 1 : 0;
            // the kernel is the one that reparents orphans, so it has to be able to see this.
            this.subreaperHook?.set(this.childSubreaper);
            ret = 0n;
            break;
          case 37: // PR_GET_CHILD_SUBREAPER
            if (this.ptrFaults(bytes, Number(a1), 4)) { ret = -BigInt(EFAULT); break; }
            view.setInt32(Number(a1) + GUEST_MEM_BASE,
                          this.subreaperHook ? this.subreaperHook.get() : this.childSubreaper, true);
            ret = 0n;
            break;
          case 38: // PR_SET_NO_NEW_PRIVS - a one-way switch: once on it can never be turned off
            if (prArg2 !== 1) { ret = -BigInt(EINVAL); break; }
            if (Number(a2) !== 0 || Number(a3) !== 0 || Number(a4) !== 0) { ret = -BigInt(EINVAL); break; }
            this.noNewPrivs = 1;
            ret = 0n;
            break;
          case 39: // PR_GET_NO_NEW_PRIVS - a pure query, so every argument must be zero
            if (prArg2 !== 0 || Number(a2) !== 0 || Number(a3) !== 0 || Number(a4) !== 0) { ret = -BigInt(EINVAL); break; }
            ret = BigInt(this.noNewPrivs);
            break;
          case 41: // PR_SET_THP_DISABLE - arg2 is the flag, and the rest are unused
            if (Number(a2) !== 0 || Number(a3) !== 0 || Number(a4) !== 0) { ret = -BigInt(EINVAL); break; }
            this.thpDisable = prArg2 !== 0 ? 1 : 0;
            ret = 0n;
            break;
          case 42: // PR_GET_THP_DISABLE
            if (prArg2 !== 0 || Number(a2) !== 0 || Number(a3) !== 0 || Number(a4) !== 0) { ret = -BigInt(EINVAL); break; }
            ret = BigInt(this.thpDisable);
            break;
          // options that exist but describe hardware or kernel facilities this system does not
          // have. linux answers EINVAL for these on an arch that lacks them, and so do we.
          case 5: case 6:   // PR_GET/SET_UNALIGN
          case 9: case 10:  // PR_GET/SET_FPEMU
          case 11: case 12: // PR_GET/SET_FPEXC
          case 19: case 20: // PR_GET/SET_ENDIAN
          case 25: case 26: // PR_GET/SET_TSC
          case 43: case 44: // PR_MPX_ENABLE/DISABLE_MANAGEMENT
          case 45: case 46: // PR_SET/GET_FP_MODE
          case 50: case 51: // PR_SVE_SET/GET_VL
          case 63: case 64: // PR_SME_SET/GET_VL
            ret = -BigInt(EINVAL);
            break;
          case 13: // PR_GET_TIMING - statistical process timing, the only kind linux has
            ret = 0n;
            break;
          case 14: // PR_SET_TIMING - anything but statistical is EINVAL
            ret = prArg2 === 0 ? 0n : -BigInt(EINVAL);
            break;
          case 27: // PR_GET_SECUREBITS
            ret = BigInt(this.secureBits);
            break;
          case 28: // PR_SET_SECUREBITS
            if (!this.hasCap(CAP_SETPCAP)) { ret = -BigInt(EPERM); break; }
            this.secureBits = prArg2;
            ret = 0n;
            break;
          case 31: case 32: // PR_TASK_PERF_EVENTS_DISABLE/ENABLE
          case 33: case 34: // PR_MCE_KILL / PR_MCE_KILL_GET
          case 35:          // PR_SET_MM
          case 47:          // PR_CAP_AMBIENT
          case 52: case 53: // PR_GET/SET_SPECULATION_CTRL
          case 54:          // PR_PAC_RESET_KEYS
          case 55: case 56: // PR_SET/GET_TAGGED_ADDR_CTRL
          case 57: case 58: // PR_SET/GET_IO_FLUSHER
          case 59:          // PR_SET_SYSCALL_USER_DISPATCH
          case 60: case 61: // PR_PAC_SET/GET_ENABLED_KEYS
          case 62:          // PR_SCHED_CORE
          case 65: case 66: // PR_SET/GET_MDWE
          case 67: case 68: // PR_SET/GET_MEMORY_MERGE
            ret = -BigInt(EINVAL);
            break;
          case 40: // PR_GET_TID_ADDRESS
            if (this.ptrFaults(bytes, Number(a1), 8)) { ret = -BigInt(EFAULT); break; }
            view.setBigUint64(Number(a1) + GUEST_MEM_BASE, 0n, true);
            ret = 0n;
            break;
          case 0x53564d41: // PR_SET_VMA - name an anonymous mapping. we keep no such name.
            ret = 0n;
            break;
          case 0x59616d61: // PR_SET_PTRACER - the yama LSM knob, and yama is not loaded
            ret = -BigInt(EINVAL);
            break;
          default:
            // an option linux does not know is EINVAL, and that is how a program probes for one.
            ret = -BigInt(EINVAL);
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

      // these answered root, root, root no matter what the process had done. a program that drops
      // privilege and then checks that it worked -- which is the whole point of dropping it -- was
      // told it had not.
      case 148: { // getresuid(ruid, euid, suid)
        if (this.ptrFaults(bytes, Number(a0), 4) || this.ptrFaults(bytes, Number(a1), 4) ||
            this.ptrFaults(bytes, Number(a2), 4)) { ret = -BigInt(EFAULT); break; }
        view.setUint32(Number(a0) + GUEST_MEM_BASE, this.ruid, true);
        view.setUint32(Number(a1) + GUEST_MEM_BASE, this.euid, true);
        view.setUint32(Number(a2) + GUEST_MEM_BASE, this.suid, true);
        ret = 0n;
        break;
      }

      case 150: { // getresgid(rgid, egid, sgid)
        if (this.ptrFaults(bytes, Number(a0), 4) || this.ptrFaults(bytes, Number(a1), 4) ||
            this.ptrFaults(bytes, Number(a2), 4)) { ret = -BigInt(EFAULT); break; }
        view.setUint32(Number(a0) + GUEST_MEM_BASE, this.rgid, true);
        view.setUint32(Number(a1) + GUEST_MEM_BASE, this.egid, true);
        view.setUint32(Number(a2) + GUEST_MEM_BASE, this.sgid, true);
        ret = 0n;
        break;
      }

      case 153: { // times(buf) - get process CPU times
        // clock_t here is in USER_HZ ticks, which linux fixes at 100/second regardless of the
        // kernel's internal HZ. reporting the same numbers times(2) and getrusage(2) report is not
        // optional: programs cross-check them, and a `times` that always said zero made every
        // elapsed-cpu measurement come out as zero no matter how long the work took.
        if (Number(a0) !== 0) {
          if (this.ptrFaults(bytes, Number(a0), 32)) { ret = -BigInt(EFAULT); break; }
          const tmsBuf = Number(a0) + GUEST_MEM_BASE;
          view.setBigUint64(tmsBuf, BigInt(Math.floor(this.ownUserMs() / 10)), true);       // tms_utime
          view.setBigUint64(tmsBuf + 8, BigInt(Math.floor(this.ownSysMs() / 10)), true);     // tms_stime
          view.setBigUint64(tmsBuf + 16, BigInt(Math.floor(this.childUtimeMs / 10)), true); // tms_cutime
          view.setBigUint64(tmsBuf + 24, BigInt(Math.floor(this.childStimeMs / 10)), true); // tms_cstime
        }
        // the return value is ticks since an arbitrary point in the past, and its only defined use
        // is differencing two calls. monotonic uptime gives that; wall-clock does not.
        ret = BigInt(Math.floor(Number(this.monoNs() / 1000000n) / 10));
        break;
      }

      case 92: { // personality — set execution domain
        // 0xffffffff is the "tell me what it is" spelling: it is not a domain anyone can be in, so
        // the kernel treats it as a query. every other value is set, and the PREVIOUS one is
        // returned -- which is what makes the query work at all.
        const persNew = Number(BigInt.asUintN(32, a0));
        const persOld = this.personality;
        if (persNew !== 0xffffffff) this.personality = persNew;
        ret = BigInt(persOld);
        break;
      }

      // ── Timer syscalls ──

      case 102: { // getitimer(which, curr_value)
        // struct itimerval = { it_interval, it_value }, both struct timeval { sec, usec }.
        const which = Number(BigInt.asIntN(32, a0));
        if (which < 0 || which > 2) { ret = -BigInt(EINVAL); break; }
        if (this.ptrWriteFaults(bytes, Number(a1), 32)) { ret = -BigInt(EFAULT); break; }
        this.checkTimers();
        const cur = this.timerRemaining(this.itimers[which]);
        this.writeItimer(view, Number(a1) + GUEST_MEM_BASE, cur.intervalNs, cur.valueNs, 1_000_000n);
        ret = 0n;
        break;
      }

      case 103: { // setitimer(which, new_value, old_value)
        // ITIMER_REAL counts wall time and raises SIGALRM; ITIMER_VIRTUAL and ITIMER_PROF count
        // this process's own cpu time and raise SIGVTALRM and SIGPROF. they are the same machinery
        // pointed at different clocks, which is why a cpu timer cannot fire while the process is
        // parked -- its clock is not running either.
        const which = Number(BigInt.asIntN(32, a0));
        const newPtr = Number(a1), oldPtr = Number(a2);
        let want = { intervalNs: 0n, valueNs: 0n };
        if (newPtr !== 0) {
          if (this.ptrFaults(bytes, newPtr, 32)) { ret = -BigInt(EFAULT); break; }
          const got = this.readItimer(view, newPtr + GUEST_MEM_BASE, 1_000_000n);
          if (!got) { ret = -BigInt(EINVAL); break; }
          want = got;
        }
        if (which < 0 || which > 2) { ret = -BigInt(EINVAL); break; }
        this.checkTimers();
        const t = this.itimers[which];
        const prev = this.timerRemaining(t);
        // arming ITIMER_REAL with a zero it_value drops its period too; the cpu timers keep theirs.
        t.intervalNs = (which === 0 && want.valueNs === 0n) ? 0n : want.intervalNs;
        t.nextNs = want.valueNs > 0n ? this.timerClockNs(t.clockid) + want.valueNs : 0n;
        if (oldPtr !== 0) {
          if (this.ptrWriteFaults(bytes, oldPtr, 32)) { ret = -BigInt(EFAULT); break; }
          this.writeItimer(view, oldPtr + GUEST_MEM_BASE, prev.intervalNs, prev.valueNs, 1_000_000n);
        }
        ret = 0n;
        break;
      }
      case 107: { // timer_create(clockid, sevp, timerid)
        // a posix timer is a deadline plus a way of announcing it: a signal, or nothing at all.
        // SIGEV_THREAD is a libc construction -- it asks the kernel for a signal on a specific
        // thread and runs the callback itself -- so the kernel side of it is SIGEV_THREAD_ID.
        const tcClk = Number(BigInt.asIntN(32, a0));
        if (!this.knownClock(tcClk)) { ret = -BigInt(EINVAL); break; }
        // the coarse and raw clocks are readable but cannot carry a timer: they have no way to
        // schedule anything, so linux answers EOPNOTSUPP rather than pretending.
        if (tcClk === 4 || tcClk === 5 || tcClk === 6) { ret = -BigInt(EOPNOTSUPP); break; }
        let tcNotify = SIGEV_SIGNAL, tcSigno = SIGALRM, tcVal = BigInt(this.nextPosixTimerId);
        const tcEvp = Number(a1);
        if (tcEvp !== 0) {
          if (this.ptrFaults(bytes, tcEvp, 64)) { ret = -BigInt(EFAULT); break; }
          const eb = tcEvp + GUEST_MEM_BASE;
          tcVal = view.getBigUint64(eb, true);
          tcSigno = view.getInt32(eb + 8, true);
          tcNotify = view.getInt32(eb + 12, true);
          if (tcNotify === SIGEV_SIGNAL || tcNotify === SIGEV_THREAD || tcNotify === SIGEV_THREAD_ID) {
            if (tcSigno <= 0 || tcSigno > SIGRTMAX) { ret = -BigInt(EINVAL); break; }
          } else if (tcNotify !== SIGEV_NONE) { ret = -BigInt(EINVAL); break; }
        }
        if (this.ptrWriteFaults(bytes, Number(a2), 4)) { ret = -BigInt(EFAULT); break; }
        const tcId = this.nextPosixTimerId++;
        this.posixTimers.set(tcId, {
          id: tcId, clockid: tcClk, nextNs: 0n, intervalNs: 0n,
          notify: tcNotify, signo: tcSigno, sigval: tcVal, overrun: 0, accrued: 0,
        });
        view.setInt32(Number(a2) + GUEST_MEM_BASE, tcId, true);
        ret = 0n;
        break;
      }

      case 108: { // timer_gettime(timerid, curr_value)
        const tgT = this.posixTimers.get(Number(BigInt.asIntN(32, a0)));
        if (!tgT) { ret = -BigInt(EINVAL); break; }
        if (this.ptrWriteFaults(bytes, Number(a1), 32)) { ret = -BigInt(EFAULT); break; }
        this.checkTimers();
        const tgCur = this.timerRemaining(tgT);
        this.writeItimer(view, Number(a1) + GUEST_MEM_BASE, tgCur.intervalNs, tgCur.valueNs, 1_000_000_000n);
        ret = 0n;
        break;
      }

      case 109: { // timer_getoverrun(timerid)
        // how many expirations the most recent delivery stood for, beyond the one it announced.
        // a pure read: settling the timers here would replace that number with a fresh count of
        // the microseconds between the signal and the handler that is asking.
        const toT = this.posixTimers.get(Number(BigInt.asIntN(32, a0)));
        if (!toT) { ret = -BigInt(EINVAL); break; }
        ret = BigInt(toT.overrun);
        break;
      }

      case 110: { // timer_settime(timerid, flags, new_value, old_value)
        const tsNew = Number(a2), tsOld = Number(a3);
        if (tsNew === 0) { ret = -BigInt(EINVAL); break; }
        if (this.ptrFaults(bytes, tsNew, 32)) { ret = -BigInt(EFAULT); break; }
        const tsWant = this.readItimer(view, tsNew + GUEST_MEM_BASE, 1_000_000_000n);
        if (!tsWant) { ret = -BigInt(EINVAL); break; }
        const tsFlags = Number(BigInt.asIntN(32, a1));
        if (tsFlags & ~TIMER_ABSTIME) { ret = -BigInt(EINVAL); break; }
        const tsT = this.posixTimers.get(Number(BigInt.asIntN(32, a0)));
        if (!tsT) { ret = -BigInt(EINVAL); break; }
        this.checkTimers();
        const tsPrev = this.timerRemaining(tsT);
        tsT.intervalNs = tsWant.intervalNs;
        tsT.nextNs = tsWant.valueNs === 0n ? 0n
                   : (tsFlags & TIMER_ABSTIME) ? tsWant.valueNs
                   : this.timerClockNs(tsT.clockid) + tsWant.valueNs;
        tsT.overrun = 0;
        tsT.accrued = 0;
        if (tsOld !== 0) {
          if (this.ptrWriteFaults(bytes, tsOld, 32)) { ret = -BigInt(EFAULT); break; }
          this.writeItimer(view, tsOld + GUEST_MEM_BASE, tsPrev.intervalNs, tsPrev.valueNs, 1_000_000_000n);
        }
        ret = 0n;
        break;
      }

      case 111: // timer_delete(timerid)
        ret = this.posixTimers.delete(Number(BigInt.asIntN(32, a0))) ? 0n : -BigInt(EINVAL);
        break;
      case 114: { // clock_getres(clk_id, res)
        const grClk = Number(BigInt.asIntN(32, a0));
        if (!this.knownClock(grClk)) { ret = -BigInt(EINVAL); break; }
        // a null pointer is explicitly allowed: it makes clock_getres a pure "does this clock
        // exist" probe, which is how a program tests for one.
        const resPtr = Number(a1);
        if (resPtr !== 0) {
          if (this.ptrFaults(bytes, resPtr, 16)) { ret = -BigInt(EFAULT); break; }
          const base = resPtr + GUEST_MEM_BASE;
          view.setBigUint64(base, 0n, true);
          view.setBigUint64(base + 8, this.clockResNs(grClk), true);
        }
        ret = 0n;
        break;
      }

      case 32: // flock — advisory file locking
        ret = 0n; // accept all locks (single-user)
        break;

      // ── Scheduling ──
      // the sched_* family (118..123, 125..127, 274, 275) is handled by the worker, which can reach
      // the shared process table these calls read and write -- they name a pid rather than the
      // caller. they only arrive here on the host-side test path, where there is no table.
      case 118: case 119: case 120: case 121: case 122: case 123:
      case 125: case 126:
        ret = -BigInt(ENOSYS);
        break;

      // inotify (26/27/28) is handled by the worker, which can reach the kernel's watch registry.
      // it only arrives here on the host-side test path, where there is no kernel to ask.
      case 26: case 27: case 28:
        ret = -BigInt(ENOSYS);
        break;

      // ── timerfd ──
      // the same timer as timer_create, announced by making a descriptor readable instead of by
      // raising a signal. a read yields the number of expirations since the last read and clears
      // it, which is why the count has to be carried rather than a mere "it fired" flag.

      case 85: { // timerfd_create(clockid, flags)
        const tfClk = Number(BigInt.asIntN(32, a0));
        const tfFlags = Number(BigInt.asIntN(32, a1));
        if (tfFlags & ~(TFD_CLOEXEC | TFD_NONBLOCK)) { ret = -BigInt(EINVAL); break; }
        if (tfClk !== CLOCK_REALTIME && tfClk !== CLOCK_MONOTONIC && tfClk !== CLOCK_BOOTTIME &&
            tfClk !== CLOCK_REALTIME_ALARM && tfClk !== CLOCK_BOOTTIME_ALARM) {
          ret = -BigInt(EINVAL); break;
        }
        // the alarm clocks are the ones that wake a suspended machine, so they are privileged.
        if ((tfClk === CLOCK_REALTIME_ALARM || tfClk === CLOCK_BOOTTIME_ALARM) && !this.hasCap(CAP_WAKE_ALARM)) {
          ret = -BigInt(EPERM); break;
        }
        const tfFd = this.allocFd();
        if (tfFd < 0) { ret = -BigInt(EMFILE); break; }
        const tfId = this.nextTimerFdId++;
        this.timerFds.set(tfId, { clockid: tfClk, nextNs: 0n, intervalNs: 0n, ticks: 0n });
        this.fdTable.set(tfFd, { type: 'timerfd', id: tfId });
        if (tfFlags & TFD_CLOEXEC) this.fdCloexec.add(tfFd);
        this.fdFlags.set(tfFd, tfFlags & TFD_NONBLOCK);
        ret = BigInt(tfFd);
        break;
      }

      case 86: { // timerfd_settime(fd, flags, new_value, old_value)
        const tsfNew = Number(a2), tsfOld = Number(a3);
        if (this.ptrFaults(bytes, tsfNew, 32)) { ret = -BigInt(EFAULT); break; }
        const tsfWant = this.readItimer(view, tsfNew + GUEST_MEM_BASE, 1_000_000_000n);
        if (!tsfWant) { ret = -BigInt(EINVAL); break; }
        const tsfFlags = Number(BigInt.asIntN(32, a1));
        // TFD_TIMER_CANCEL_ON_SET only means anything for an absolute deadline: it asks to be told
        // when the clock it is measured against jumps underneath it.
        if ((tsfFlags & ~(TFD_TIMER_ABSTIME | TFD_TIMER_CANCEL_ON_SET)) !== 0 ||
            ((tsfFlags & TFD_TIMER_CANCEL_ON_SET) && !(tsfFlags & TFD_TIMER_ABSTIME))) {
          ret = -BigInt(EINVAL); break;
        }
        const tsfT = this.timerFdOf(Number(BigInt.asIntN(32, a0)));
        if (typeof tsfT === 'number') { ret = BigInt(tsfT); break; }
        this.checkTimers();
        const tsfPrev = this.timerRemaining(tsfT);
        tsfT.intervalNs = tsfWant.intervalNs;
        tsfT.nextNs = tsfWant.valueNs === 0n ? 0n
                    : (tsfFlags & TFD_TIMER_ABSTIME) ? tsfWant.valueNs
                    : this.timerClockNs(tsfT.clockid) + tsfWant.valueNs;
        tsfT.ticks = 0n;
        if (tsfOld !== 0) {
          if (this.ptrWriteFaults(bytes, tsfOld, 32)) { ret = -BigInt(EFAULT); break; }
          this.writeItimer(view, tsfOld + GUEST_MEM_BASE, tsfPrev.intervalNs, tsfPrev.valueNs, 1_000_000_000n);
        }
        ret = 0n;
        break;
      }

      case 87: { // timerfd_gettime(fd, curr_value)
        const tgfT = this.timerFdOf(Number(BigInt.asIntN(32, a0)));
        if (typeof tgfT === 'number') { ret = BigInt(tgfT); break; }
        if (this.ptrWriteFaults(bytes, Number(a1), 32)) { ret = -BigInt(EFAULT); break; }
        this.checkTimers();
        const tgfCur = this.timerRemaining(tgfT);
        this.writeItimer(view, Number(a1) + GUEST_MEM_BASE, tgfCur.intervalNs, tgfCur.valueNs, 1_000_000_000n);
        ret = 0n;
        break;
      }

      // ── Misc process control ──

      case 97: { // unshare(flags) — disassociate parts of the process's execution context
        // the parts we CAN unshare are the ones a single-process view makes trivial: a private
        // file-descriptor table, a private root and cwd, a private semaphore-undo list. every
        // namespace flag names something this system does not have, and saying yes to one would
        // claim an isolation that is not there.
        const CLONE_FILES = 0x400, CLONE_FS = 0x200, CLONE_SYSVSEM = 0x40000;
        const CLONE_NEWNS = 0x20000, CLONE_NEWCGROUP = 0x2000000, CLONE_NEWUTS = 0x4000000;
        const CLONE_NEWIPC = 0x8000000, CLONE_NEWUSER = 0x10000000, CLONE_NEWPID = 0x20000000;
        const CLONE_NEWNET = 0x40000000, CLONE_NEWTIME = 0x80;
        const NAMESPACES = CLONE_NEWNS | CLONE_NEWCGROUP | CLONE_NEWUTS | CLONE_NEWIPC |
                           CLONE_NEWUSER | CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWTIME;
        const unFlags = Number(BigInt.asUintN(32, a0));
        if (unFlags & ~(CLONE_FILES | CLONE_FS | CLONE_SYSVSEM | NAMESPACES)) {
          ret = -BigInt(EINVAL); break;
        }
        // creating any namespace needs CAP_SYS_ADMIN, and that check comes before the refusal --
        // an unprivileged caller learns it was not allowed rather than that we lack the feature.
        if ((unFlags & NAMESPACES) && !this.hasCap(CAP_SYS_ADMIN)) { ret = -BigInt(EPERM); break; }
        if (unFlags & NAMESPACES) { ret = -BigInt(EINVAL); break; }
        ret = 0n;
        break;
      }
      case 168: { // getcpu(cpu, node, tcache)
        // a pointer the caller gave has to be writable; both are optional, but a bad one is a
        // bad one. writing through it unchecked reported success for a wild address.
        const cpuPtr = Number(a0);
        const nodePtr = Number(a1);
        if (cpuPtr !== 0 && this.ptrWriteFaults(bytes, cpuPtr, 4)) { ret = -BigInt(EFAULT); break; }
        if (nodePtr !== 0 && this.ptrWriteFaults(bytes, nodePtr, 4)) { ret = -BigInt(EFAULT); break; }
        if (cpuPtr !== 0) view.setUint32(cpuPtr + GUEST_MEM_BASE, 0, true);
        if (nodePtr !== 0) view.setUint32(nodePtr + GUEST_MEM_BASE, 0, true);
        ret = 0n;
        break;
      }
      // ── scheduling priority ─────────────────────────────────────────────────
      // these used to accept anything and always answer "nice 0". that is not a simplification:
      // setpriority reported success for a `which` that does not exist and for processes that do
      // not exist, and nothing a caller set was ever visible again.
      case 140: { // setpriority(which, who, prio)
        const spWhich = Number(BigInt.asIntN(32, a0));
        const spWho = Number(BigInt.asIntN(32, a1));
        let spPrio = Number(BigInt.asIntN(32, a2));
        if (spWhich < PRIO_PROCESS || spWhich > PRIO_USER) { ret = -BigInt(EINVAL); break; }
        // linux clamps rather than rejecting: anything below -20 becomes -20, above 19 becomes 19.
        if (spPrio < -20) spPrio = -20;
        if (spPrio > 19) spPrio = 19;
        const spTargets = this.priorityTargets(spWhich, spWho);
        if (spTargets.length === 0) { ret = -BigInt(ESRCH); break; }
        // a dedicated error variable, not `ret`: `ret` arrives here holding the dispatch's default
        // of -ENOSYS, so testing it after the loop reported "not implemented" for every call.
        let spErr = 0;
        for (const t of spTargets) {
          // whose process it is comes FIRST: a caller who may not touch it at all is told that,
          // rather than being told the priority it asked for was too ambitious.
          if (this.euid !== 0 && t.ruid !== this.ruid && t.ruid !== this.euid) { spErr = EPERM; break; }
          // raising priority (a LOWER nice) is privileged; lowering it is always allowed.
          if (spPrio < t.nice && !this.hasCap(CAP_SYS_NICE)) { spErr = EACCES; break; }
        }
        if (spErr !== 0) { ret = -BigInt(spErr); break; }
        for (const t of spTargets) this.prioHook?.setNice(t.pid, spPrio);
        ret = 0n;
        break;
      }
      case 141: { // getpriority(which, who)
        const gpWhich = Number(BigInt.asIntN(32, a0));
        const gpWho = Number(BigInt.asIntN(32, a1));
        if (gpWhich < PRIO_PROCESS || gpWhich > PRIO_USER) { ret = -BigInt(EINVAL); break; }
        const gpTargets = this.priorityTargets(gpWhich, gpWho);
        if (gpTargets.length === 0) { ret = -BigInt(ESRCH); break; }
        // the HIGHEST priority in the set, which is the LOWEST nice value. the syscall is biased by
        // 20 so it never returns a negative number that would be read as an errno.
        let best = 19;
        for (const t of gpTargets) if (t.nice < best) best = t.nice;
        ret = BigInt(20 - best);
        break;
      }

      // ── Memory control (per spec: accept, no-op for our in-memory VFS) ──
      case 228:   // mlock(addr, len)
      case 284:   // mlock2(addr, len, flags) -- mlock with MLOCK_ONFAULT
      case 229: { // munlock(addr, len)
        // mlock2's third argument is the only difference. MLOCK_ONFAULT says "lock the pages as
        // they are faulted in rather than now", which for a wasm memory that is already resident
        // is the same locked range; any other bit names a behaviour that does not exist.
        const MLOCK_ONFAULT = 1;
        if (nr === 284 && (Number(BigInt.asIntN(32, a2)) & ~MLOCK_ONFAULT)) { ret = -BigInt(EINVAL); break; }
        // these accepted anything at all, so locking a range that was never mapped succeeded.
        //
        // linux asks "may this process lock memory at all" FIRST, before it looks at the range or
        // the length: a zero RLIMIT_MEMLOCK without CAP_IPC_LOCK is EPERM, and a limit that is
        // merely too small is ENOMEM. answering ENOMEM to both loses the distinction between "you
        // are not allowed to do this" and "you asked for more than your share".
        if (nr !== 229 && this.memlockLimit === 0 && !this.hasCap(CAP_IPC_LOCK)) {
          ret = -BigInt(EPERM); break;
        }
        const lkLen = Number(BigInt.asIntN(64, a1));
        if (lkLen < 0) { ret = -BigInt(EINVAL); break; }
        if (lkLen === 0) { ret = 0n; break; }
        const lkStart = Number(a0) & ~0xFFF;
        const lkEnd = (Number(a0) + lkLen + 0xFFF) & ~0xFFF;
        if (lkEnd <= lkStart) { ret = -BigInt(EINVAL); break; } // wrapped
        if (!this.rangeMapped(lkStart, lkEnd)) { ret = -BigInt(ENOMEM); break; }
        // pinning memory is rationed by RLIMIT_MEMLOCK unless the process holds CAP_IPC_LOCK.
        if (nr !== 229 && !this.hasCap(CAP_IPC_LOCK) && (lkEnd - lkStart) > this.memlockLimit) {
          ret = -BigInt(ENOMEM); break;
        }
        // mlock2(MLOCK_ONFAULT) locks the range without pulling it in: the pages join the lock as
        // they are touched, so the range is locked but not (yet) resident.
        const lkOnFault = nr === 284 && (Number(BigInt.asIntN(32, a2)) & 1) !== 0;
        this.setLocked(lkStart, lkEnd, nr !== 229, lkOnFault);
        ret = 0n;
        break;
      }
      case 230: { // mlockall(flags)
        const MCL_CURRENT = 1, MCL_FUTURE = 2, MCL_ONFAULT = 4;
        const mlaFlags = Number(BigInt.asIntN(32, a0));
        // at least one of CURRENT/FUTURE is required, ONFAULT means nothing without them, and
        // anything else does not exist. accepting 0 made mlockall(0) a silent success.
        if (mlaFlags === 0 || (mlaFlags & ~(MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT))) { ret = -BigInt(EINVAL); break; }
        if ((mlaFlags & MCL_ONFAULT) && !(mlaFlags & MCL_CURRENT)) { ret = -BigInt(EINVAL); break; }
        if (!this.hasCap(CAP_IPC_LOCK) && this.memlockLimit === 0) { ret = -BigInt(EPERM); break; }
        // locking the WHOLE address space is rationed by the same limit a single mlock is: if it
        // does not fit, nothing is locked and the answer is ENOMEM.
        if ((mlaFlags & MCL_CURRENT) && !this.hasCap(CAP_IPC_LOCK)) {
          const mapped = this.readAddressSpace(() => this.vmas.reduce((n, v) => n + (v.end - v.start), 0));
          if (mapped > this.memlockLimit) { ret = -BigInt(ENOMEM); break; }
        }
        // MCL_CURRENT locks what is mapped NOW, which is the whole address space as the vma list
        // has it. saying yes without locking anything left VmLck at zero, so a program that pins
        // its memory and then checks could see that it had not happened.
        if (mlaFlags & MCL_CURRENT) {
          for (const v of this.readAddressSpace(() => this.vmas.map((x) => ({ start: x.start, end: x.end })))) {
            this.setLocked(v.start, v.end, true);
          }
        }
        // MCL_FUTURE outlives this call: everything mapped from here on is locked as it appears.
        this.lockFuture = (mlaFlags & MCL_FUTURE) !== 0;
        ret = 0n;
        break;
      }
      case 231: // munlockall
        this.lockedRegions = [];
        this.lockFuture = false;
        if (this.vfs) (this.vfs as any).vmLckKb = 0;
        ret = 0n;
        break;
      case 227: { // msync — flush a writable shared file mapping back to disk
        // the arguments were never checked at all: an unaligned address, a nonsense flag, or a
        // range that maps nothing all reported a successful flush of nothing.
        const MS_ASYNC = 1, MS_INVALIDATE = 2, MS_SYNC = 4;
        const msFlags = Number(BigInt.asIntN(32, a2));
        if (msFlags & ~(MS_ASYNC | MS_INVALIDATE | MS_SYNC)) { ret = -BigInt(EINVAL); break; }
        // ASYNC and SYNC are the two ways of asking, and asking both ways at once is nonsense.
        if ((msFlags & MS_ASYNC) && (msFlags & MS_SYNC)) { ret = -BigInt(EINVAL); break; }
        const msAddrB = BigInt.asUintN(64, a0);
        if ((msAddrB & 0xFFFn) !== 0n) { ret = -BigInt(EINVAL); break; }
        const msAddr = Number(msAddrB);
        const msLenU = BigInt.asUintN(64, a1);
        if (BigInt(msAddr) + msLenU > BigInt(MAX_PROCESS_BYTES)) { ret = -BigInt(ENOMEM); break; }
        const msEnd = msAddr + ((Number(msLenU) + 0xFFF) & ~0xFFF);
        if (msEnd === msAddr) { ret = 0n; break; }
        if (!this.rangeMapped(msAddr, msEnd)) { ret = -BigInt(ENOMEM); break; }
        // discarding the pages of a mapping that is pinned in memory contradicts the pin.
        if ((msFlags & MS_INVALIDATE) && this.rangeLocked(msAddr, msEnd)) { ret = -BigInt(EBUSY); break; }
        // MS_INVALIDATE on its own discards what the mapping holds and takes the file's word for
        // it; asking to sync as well writes back first, which is the direction that matters then.
        if ((msFlags & MS_INVALIDATE) && !(msFlags & (MS_SYNC | MS_ASYNC))) {
          const msReload = this.buildMmapReload(cpu, memory, msAddr, Number(a1));
          if (msReload) return msReload;
          ret = 0n; break;
        }
        const msFlush = this.buildMmapFlush(cpu, memory, msAddr, Number(a1), false);
        if (msFlush) return msFlush;
        ret = 0n;
        break;
      }
      case 232: { // mincore(addr, length, vec) - which pages of a mapping are resident
        const mcAddr = Number(a0);
        // length is a size_t, and reading it as SIGNED turned "a length that runs off the end of
        // the address space" into EINVAL. linux checks the range against the user address space
        // and answers ENOMEM, which is exactly the case ltp tests with RLIM_INFINITY arithmetic.
        const mcLenU = BigInt.asUintN(64, a1);
        if (mcAddr & 0xFFF) { ret = -BigInt(EINVAL); break; }
        if (BigInt(mcAddr) + mcLenU > BigInt(MAX_PROCESS_BYTES)) { ret = -BigInt(ENOMEM); break; }
        const mcLen = Number(mcLenU);
        const mcPages = Math.ceil(mcLen / 4096);
        // ENOMEM for an unmapped range comes BEFORE the vec check: linux looks at the range it was
        // asked about before it looks at where to put the answer.
        if (mcPages > 0 && !this.rangeMapped(mcAddr, mcAddr + mcPages * 4096)) { ret = -BigInt(ENOMEM); break; }
        if (this.ptrFaults(bytes, Number(a2), Math.max(1, mcPages))) { ret = -BigInt(EFAULT); break; }
        // which pages are really there. nothing below this evicts a page, but a page of an
        // anonymous mapping that has never been written is not backed by anything either -- it
        // reads as zeros because it was never touched, and that is exactly what "not resident"
        // means to the caller. so the answer is the same one fork's snapshot uses: a page with a
        // byte in it is present, an all-zero page is not. (a page deliberately filled with zeros
        // is indistinguishable from an untouched one here; linux would call that one present.)
        const mcVec = Number(a2) + GUEST_MEM_BASE;
        const mcB = new Uint8Array(memory.buffer);
        // a mapping with something BEHIND it -- a file we read in, or a shared block in the arena --
        // is materialised here in full: there is no second-level paging to make part of it absent.
        // saying otherwise reported a file mapping the caller had just faulted in as not resident.
        const mcKind = this.rangeKind(mcAddr, mcAddr + mcPages * 4096);
        const mcBacked = mcKind.shared || mcKind.fileBacked;
        for (let i = 0; i < mcPages; i++) {
          const base = mcAddr + i * 4096 + GUEST_MEM_BASE;
          // an mlocked page IS resident, by definition: locking a range is a promise that it stays
          // in memory, and linux populates it to keep that promise.
          let present = this.populatedBytesIn(mcAddr + i * 4096, mcAddr + (i + 1) * 4096) > 0 ? 1 : 0;
          if (!present && mcBacked) {
            const k = this.rangeKind(mcAddr + i * 4096, mcAddr + (i + 1) * 4096);
            if (k.shared || k.fileBacked) present = 1;
          }
          if (!present && base + 4096 <= mcB.length) {
            for (let off = 0; off < 4096; off += 8) {
              if (mcB[base + off] || mcB[base + off + 1] || mcB[base + off + 2] || mcB[base + off + 3] ||
                  mcB[base + off + 4] || mcB[base + off + 5] || mcB[base + off + 6] || mcB[base + off + 7]) {
                present = 1; break;
              }
            }
          }
          mcB[mcVec + i] = present;
        }
        ret = 0n;
        break;
      }
      case 216: { // mremap(old_addr, old_size, new_size, flags, new_addr)
        // returning old_addr while ignoring new_size let the guest write past the mapping into
        // the next allocation -> heap corruption (musl mallocng grows large allocations via
        // mremap). resize for real: extend in place if this is the top of the arena, otherwise
        // move to a fresh region (MREMAP_MAYMOVE, which musl always sets) and copy the bytes over.
        const MREMAP_MAYMOVE = 1, MREMAP_FIXED = 2, MREMAP_DONTUNMAP = 4;
        const oldAddr = Number(a0), oldSize = Number(a1), newSize = Number(a2);
        const mrFlags = Number(BigInt.asIntN(32, a3));
        const alignedOld = (oldSize + 0xFFF) & ~0xFFF;
        const alignedNew = (newSize + 0xFFF) & ~0xFFF;
        // the argument checks, in linux's order. none of these were made before: an unaligned old
        // address was silently rounded, a zero new size unmapped everything, and MREMAP_FIXED was
        // ignored outright -- so a caller naming its destination got a mapping somewhere else.
        if (mrFlags & ~(MREMAP_MAYMOVE | MREMAP_FIXED | MREMAP_DONTUNMAP)) { ret = -BigInt(EINVAL); break; }
        if (oldAddr & 0xFFF) { ret = -BigInt(EINVAL); break; }
        if ((mrFlags & MREMAP_FIXED) && !(mrFlags & MREMAP_MAYMOVE)) { ret = -BigInt(EINVAL); break; }
        if ((mrFlags & MREMAP_DONTUNMAP) &&
            (!(mrFlags & MREMAP_MAYMOVE) || (mrFlags & MREMAP_FIXED))) { ret = -BigInt(EINVAL); break; }
        if (alignedNew === 0) { ret = -BigInt(EINVAL); break; }
        const mrVma = this.readAddressSpace(() => {
          const v = this.vmaAt(oldAddr);
          return v ? { ...v } : null;
        });
        // an address that maps nothing is EFAULT, and so is a length that runs off the end of the
        // mapping it starts in: mremap moves ONE mapping, and a range that spans a hole (or an
        // unrelated neighbour) is not a mapping it can be asked about.
        if (!mrVma) { ret = -BigInt(EFAULT); break; }
        if (alignedOld > mrVma.end - oldAddr) { ret = -BigInt(EFAULT); break; }
        if (mrFlags & MREMAP_FIXED) {
          const fixedAddr = Number(a4);
          if (fixedAddr & 0xFFF) { ret = -BigInt(EINVAL); break; }
          // the two ranges must be disjoint -- a move onto itself has no meaning, and linux will
          // not try to work out which bytes survive it.
          if (oldAddr + alignedOld > fixedAddr && fixedAddr + alignedNew > oldAddr) {
            ret = -BigInt(EINVAL); break;
          }
          const fixedEnd = fixedAddr + alignedNew;
          if (fixedEnd + GUEST_MEM_BASE > this.maxBytes) { ret = -BigInt(ENOMEM); break; }
          if (!this.growTo(memory, fixedEnd + GUEST_MEM_BASE)) { ret = -BigInt(ENOMEM); break; }
          const fixedMem = new Uint8Array(memory.buffer);
          const keep = Math.min(alignedOld, alignedNew);
          fixedMem.copyWithin(fixedAddr + GUEST_MEM_BASE, oldAddr + GUEST_MEM_BASE,
                              oldAddr + GUEST_MEM_BASE + keep);
          fixedMem.fill(0, fixedAddr + GUEST_MEM_BASE + keep, fixedEnd + GUEST_MEM_BASE);
          this.withAddressSpace(() => {
            this.vmaSet(fixedAddr, fixedEnd,
                        { prot: mrVma.prot, shared: mrVma.shared, maxProt: mrVma.maxProt });
            this.vmaSet(oldAddr, oldAddr + alignedOld, null);
            return 0n;
          });
          try { this.protHook?.(fixedAddr, alignedNew, mrVma.prot); } catch {}
          try { this.protHook?.(oldAddr, alignedOld, 0); } catch {}
          ret = BigInt(fixedAddr);
          break;
        }
        if (alignedNew <= alignedOld) {
          // a shrink releases the tail, which is address space someone else may now have.
          if (alignedNew < alignedOld) {
            this.vmaSet(oldAddr + alignedNew, oldAddr + alignedOld, null);
            try { this.protHook?.(oldAddr + alignedNew, alignedOld - alignedNew, 0); } catch {}
          }
          ret = a0; break;
        }
        // grow in place when the old mapping sits at the very top of the arena
        if (BigInt(oldAddr + alignedOld) === this.mmapBase) {
          const needed = oldAddr + alignedNew + GUEST_MEM_BASE;
          if (needed > this.maxBytes) { ret = -BigInt(ENOMEM); break; }
          if (!this.growTo(memory, needed)) { ret = -BigInt(ENOMEM); break; }
          new Uint8Array(memory.buffer).fill(0, oldAddr + alignedOld + GUEST_MEM_BASE, oldAddr + alignedNew + GUEST_MEM_BASE);
          this.mmapBase = BigInt(oldAddr + alignedNew);
          // the grown tail is part of the mapping now. without a vma for it, fork would not copy
          // it, /proc/self/maps would not show it, and the arena allocator would hand it out again.
          this.vmaSet(oldAddr, oldAddr + alignedNew, {
            prot: mrVma.prot, shared: mrVma.shared, maxProt: mrVma.maxProt,
          });
          try { this.protHook?.(oldAddr + alignedOld, alignedNew - alignedOld, mrVma.prot); } catch {}
          ret = a0;
          break;
        }
        if (!(mrFlags & MREMAP_MAYMOVE)) { ret = -BigInt(ENOMEM); break; }
        // pick the destination and claim it in one step, for the same reason mmap does.
        // pick the destination, make sure it can actually be backed, and only THEN move: a failure
        // after the old mapping was released would lose the guest's memory outright, where linux
        // leaves an unmovable mapping exactly where it was and answers ENOMEM.
        const newAddr = this.withAddressSpace(() => {
          const picked = this.allocMmapRange(alignedNew);
          const need = Number(picked) + alignedNew + GUEST_MEM_BASE;
          if (need > this.maxBytes || !this.growTo(memory, need)) return -1n;
          this.vmaSet(Number(picked), Number(picked) + alignedNew, {
            prot: mrVma.prot, shared: mrVma.shared, maxProt: mrVma.maxProt,
          });
          this.vmaSet(oldAddr, oldAddr + alignedOld, null);
          return picked;
        });
        if (newAddr < 0n) { ret = -BigInt(ENOMEM); break; }
        const mrMem = new Uint8Array(memory.buffer);
        mrMem.copyWithin(Number(newAddr) + GUEST_MEM_BASE, oldAddr + GUEST_MEM_BASE, oldAddr + GUEST_MEM_BASE + oldSize);
        mrMem.fill(0, Number(newAddr) + GUEST_MEM_BASE + oldSize, Number(newAddr) + GUEST_MEM_BASE + alignedNew);
        // the mapping moved: it exists at its new address and is gone from the old one. neither
        // half of that was recorded before, so a moved mapping was invisible to fork, to
        // /proc/self/maps and to the page guard, and its old range stayed claimed forever.
        try { this.protHook?.(Number(newAddr), alignedNew, mrVma.prot); } catch {}
        try { this.protHook?.(oldAddr, alignedOld, 0); } catch {}
        ret = newAddr;
        break;
      }

      // ── Signal notification via fd ──
      case 74: { // signalfd4(fd, mask, sizemask, flags)
        // a descriptor that hands back the signals a process has BLOCKED, instead of running a
        // handler for them. this used to allocate a placeholder of the wrong kind, so reading the
        // descriptor answered EBADF and the whole point was missing.
        const sfFd = Number(BigInt.asIntN(32, a0));
        const sfMaskPtr = Number(a1);
        const sfSize = Number(BigInt.asIntN(64, a2));
        const sfFlags = Number(BigInt.asIntN(32, a3));
        const SFD_NONBLOCK = 0o4000, SFD_CLOEXEC = 0o2000000;
        if (sfSize !== 8) { ret = -BigInt(EINVAL); break; }
        if (sfFlags & ~(SFD_NONBLOCK | SFD_CLOEXEC)) { ret = -BigInt(EINVAL); break; }
        if (this.ptrFaults(bytes, sfMaskPtr, 8)) { ret = -BigInt(EFAULT); break; }
        // SIGKILL and SIGSTOP cannot be caught by any means, signalfd included; linux drops them
        // from the mask rather than refusing the call.
        const sfMask = view.getBigUint64(sfMaskPtr + GUEST_MEM_BASE, true) &
                       ~((1n << 8n) | (1n << 18n));
        if (sfFd === -1) {
          const newFd = this.allocFd();
          if (newFd < 0) { ret = -BigInt(EMFILE); break; }
          this.fdTable.set(newFd, { type: 'signalfd', mask: sfMask });
          if (sfFlags & SFD_CLOEXEC) this.fdCloexec.add(newFd);
          if (sfFlags & SFD_NONBLOCK) this.fdFlags.set(newFd, (this.fdFlags.get(newFd) ?? 0) | O_NONBLOCK);
          ret = BigInt(newFd);
          break;
        }
        const sfEntry = this.getFdEntry(sfFd);
        if (!sfEntry) { ret = -BigInt(EBADF); break; }
        if (sfEntry.type !== 'signalfd') { ret = -BigInt(EINVAL); break; }
        sfEntry.mask = sfMask;   // a second call replaces the set it reports
        ret = BigInt(sfFd);
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
          // struct clone_args: flags(0) pidfd(8) child_tid(16) parent_tid(24) exit_signal(32)
          // stack(40) stack_size(48) tls(56). the stack field is the LOW address of the region and
          // the child starts at stack+stack_size, unlike clone(2) where the argument is already the
          // top of the stack.
          const c3Stack = view.getBigUint64(argsPtr + 40, true);
          const c3Size = view.getBigUint64(argsPtr + 48, true);
          return {
            status: 'fork', resumePC: svcPC + 4n, flags: Number(cloneFlags & 0xffffffffn),
            childStack: c3Stack === 0n ? 0n : c3Stack + c3Size,
            ptid: view.getBigUint64(argsPtr + 24, true),
            tls: view.getBigUint64(argsPtr + 56, true),
            ctid: view.getBigUint64(argsPtr + 16, true),
          };
        }
        // Non-fork clone (threads etc) — not supported
        ret = -BigInt(ENOSYS);
        break;
      }

      // ── Exec relative to dirfd ──
      case 281: { // execveat(dirfd, pathname, argv, envp, flags)
        // the flags and the descriptor are checked before anything is read, exactly as the *at()
        // calls next door do. this used to ignore both: execveat(-1, "", AT_EMPTY_PATH) reported
        // whatever reading the empty name produced instead of EBADF.
        const raw281 = readGuestString(bytes, Number(a1));
        const flags281 = Number(BigInt.asIntN(32, a4));
        const AT_EMPTY_PATH = 0x1000, AT_SYMLINK_NOFOLLOW = 0x100;
        if (flags281 & ~(AT_EMPTY_PATH | AT_SYMLINK_NOFOLLOW)) { ret = -BigInt(EINVAL); break; }
        const dirfd281 = Number(BigInt.asIntN(32, a0));
        if ((this.isEmptyPath(raw281) || !raw281.startsWith('/')) && dirfd281 !== -100) {
          if (!this.fdIsOpen(dirfd281)) { ret = -BigInt(EBADF); break; }
        }
        if (this.isEmptyPath(raw281) && !(flags281 & AT_EMPTY_PATH)) { ret = -BigInt(ENOENT); break; }
        const pathname281 = this.resolveAt(a0, raw281);
        const argvPtrs281: string[] = [];
        let argvAddr281 = Number(a2);
        while (true) {
          const ptr = Number(view.getBigUint64(argvAddr281 + GUEST_MEM_BASE, true));
          if (ptr === 0) break;
          argvPtrs281.push(readGuestString(bytes, ptr));
          argvAddr281 += 8;
        }
        if (argvPtrs281.length === 0) argvPtrs281.push('');
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

      case 47: { // fallocate(fd, mode, offset, len)
        // ALLOCATING SPACE HAS TO ACTUALLY ALLOCATE IT. this answered "done" and did nothing, which
        // is the most dangerous shape a stub can take: the caller is told the room exists and then
        // uses it. apk asks for the size of the package, maps the file MAP_SHARED and memcpys the
        // contents in -- no write() anywhere -- so with the file still empty there was nothing to
        // map, nothing to flush back, and every large package installed as a zero-byte file.
        const faEntry = this.getFdEntry(Number(a0));
        const faMode = Number(a1);
        const faOff = Number(BigInt.asIntN(64, a2));
        const faLen = Number(BigInt.asIntN(64, a3));
        if (!faEntry) { ret = this.vfs?.hasFd(Number(a0)) ? 0n : -BigInt(EBADF); break; }
        // allocating space is a write. a descriptor that was not opened for one cannot do it.
        if (this.fdIsPath(Number(a0)) || !this.fdAccess(Number(a0)).writable) { ret = -BigInt(EBADF); break; }
        if (faLen <= 0 || faOff < 0) { ret = -BigInt(EINVAL); break; }
        if (faEntry.type !== 'native') {
          // a pipe or socket has no blocks to reserve; linux says ESPIPE/ENODEV rather than yes.
          ret = (faEntry.type === 'pipe_read' || faEntry.type === 'pipe_write') ? -BigInt(ESPIPE) : -BigInt(ENODEV);
          break;
        }
        // mode 0 grows the file to cover the range. FALLOC_FL_KEEP_SIZE (1) reserves the space
        // without changing the size, which for us is a no-op we can honestly report as done --
        // we do not report free space we cannot deliver. FALLOC_FL_PUNCH_HOLE (2) makes the range
        // read as zeros and is only defined together with KEEP_SIZE, since a hole cannot change
        // the length. anything else we do not implement.
        const FALLOC_KEEP_SIZE = 1, FALLOC_PUNCH_HOLE = 2;
        if (faMode & ~(FALLOC_KEEP_SIZE | FALLOC_PUNCH_HOLE)) { ret = -BigInt(EOPNOTSUPP); break; }
        if ((faMode & FALLOC_PUNCH_HOLE) && !(faMode & FALLOC_KEEP_SIZE)) { ret = -BigInt(EINVAL); break; }
        // a range that ends past the largest file there can be is EFBIG, not a very large file.
        if (BigInt(faOff) + BigInt(faLen) > MAX_FILE_BYTES) { ret = -BigInt(EFBIG); break; }
        if (faMode & FALLOC_PUNCH_HOLE) {
          // punching a hole is a write, and a sealed file has promised there will be no more.
          if (this.sealsOf(faEntry) & 0x8) { ret = -BigInt(EPERM); break; }
          this.invalidateStat(faEntry.path);
          return {
            status: 'bridge', method: 'fs_punch_hole',
            args: { fd: faEntry.nativeFd, offset: faOff, length: faLen },
            apply: (resp: any) => {
              if (resp.ret >= 0) this.notify(faEntry.path, IN_MODIFY, 0, false);
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            },
          };
        }
        if (faMode & FALLOC_KEEP_SIZE) { ret = 0n; break; }
        this.invalidateStat(faEntry.path);
        return {
          status: 'bridge', method: 'fs_fallocate',
          args: { fd: faEntry.nativeFd, size: faOff + faLen,
                  noGrow: (this.sealsOf(faEntry) & 0x4) ? 1 : 0 },
          apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); },
        };
      }

      case 45: { // truncate(path, length)
        // this only ever looked in the in-memory tree, which holds /proc and /dev and nothing
        // else -- so truncating any real file answered ENOENT for a file that plainly existed.
        const traw = readGuestString(bytes, Number(a0));
        // an empty pathname names nothing. resolving it would land on the cwd, and answering
        // EISDIR for a file the caller never named is a different mistake from the one they made.
        { const pe = this.pathCheck(traw); if (pe) { ret = -BigInt(pe); break; } }
        const tpath = this.resolveAt(-100n, traw);
        const tlength = Number(BigInt.asIntN(64, a1));
        if (tlength < 0) { ret = -BigInt(EINVAL); break; }
        // a file may not be grown past RLIMIT_FSIZE. linux answers EFBIG and raises SIGXFSZ, which
        // is what makes a shell print "File size limit exceeded" rather than silently truncating.
        const fsize = this.rlimitOf(RLIMIT_FSIZE);
        if (fsize.cur !== RLIM_INFINITY && BigInt(tlength) > fsize.cur) {
          this.raiseSignal(25); // SIGXFSZ
          ret = -BigInt(EFBIG);
          break;
        }
        const bTrunc = this.bridgePath(tpath);
        if (bTrunc) {
          this.invalidateStat(bTrunc);
          return {
            status: 'bridge', method: 'fs_truncate',
            args: { path: bTrunc, length: tlength },
            apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); },
          };
        }
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
        // the flags are the whole reason this call exists, and they were ignored: RENAME_NOREPLACE
        // silently overwrote, and RENAME_EXCHANGE moved one name over the other instead of swapping.
        const RENAME_NOREPLACE = 1, RENAME_EXCHANGE = 2, RENAME_WHITEOUT = 4;
        const r2Flags = Number(BigInt.asUintN(32, a4));
        if (r2Flags & ~(RENAME_NOREPLACE | RENAME_EXCHANGE | RENAME_WHITEOUT)) { ret = -BigInt(EINVAL); break; }
        if ((r2Flags & RENAME_EXCHANGE) && (r2Flags & (RENAME_NOREPLACE | RENAME_WHITEOUT))) {
          ret = -BigInt(EINVAL); break;
        }
        // a whiteout is an overlayfs marker -- a character device with device number 0 -- and this
        // filesystem has no overlay to put one in.
        if (r2Flags & RENAME_WHITEOUT) { ret = -BigInt(EINVAL); break; }
        const rawOld276 = readGuestString(bytes, Number(a1));
        const rawNew276 = readGuestString(bytes, Number(a3));
        { const pe = this.pathCheck(rawOld276) || this.pathCheck(rawNew276); if (pe) { ret = -BigInt(pe); break; } }
        const oldpath276 = this.resolveAt(a0, rawOld276);
        const newpath276 = this.resolveAt(a2, rawNew276);
        const bOld276 = this.bridgePath(oldpath276);
        if (bOld276) {
          const bNew276 = this.bridgePath(newpath276) ?? newpath276;
          this.invalidateStat(bOld276);
          this.invalidateStat(bNew276);
          return {
            status: 'bridge', method: 'fs_rename',
            args: { oldPath: bOld276, newPath: bNew276, flags: r2Flags },
            apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); },
          };
        }
        if (this.vfs) {
          // the synthetic filesystems have no atomic swap, and nothing in them is renamed anyway.
          if (r2Flags !== 0) { ret = -BigInt(EINVAL); break; }
          ret = BigInt(this.vfs.rename(oldpath276, newpath276));
        } else {
          ret = -BigInt(ENOSYS);
        }
        break;
      }

      case 279: { // memfd_create(name, flags)
        // an anonymous file: real bytes, no name anywhere in the filesystem. that is exactly what
        // O_TMPFILE already gives us, so a memfd IS one -- it used to try to create a file in the
        // synthesised /tmp, which has not existed since the filesystem moved to disk, so every
        // memfd_create in the system answered ENOENT.
        const MFD_CLOEXEC = 1, MFD_ALLOW_SEALING = 2, MFD_HUGETLB = 4, MFD_NOEXEC_SEAL = 8, MFD_EXEC = 16;
        const mfFlags = Number(BigInt.asUintN(32, a1));
        if (mfFlags & ~(MFD_CLOEXEC | MFD_ALLOW_SEALING | MFD_HUGETLB | MFD_NOEXEC_SEAL | MFD_EXEC)) {
          ret = -BigInt(EINVAL); break;
        }
        if ((mfFlags & MFD_NOEXEC_SEAL) && (mfFlags & MFD_EXEC)) { ret = -BigInt(EINVAL); break; }
        if (this.ptrFaults(bytes, Number(a0), 1)) { ret = -BigInt(EFAULT); break; }
        const mfName = readGuestString(bytes, Number(a0));
        // the name is only ever seen in /proc/<pid>/fd, prefixed with "memfd:", and the whole
        // thing has to fit in a filename.
        if (mfName.length > 249) { ret = -BigInt(EINVAL); break; }
        const mfDir = this.bridgePath('/tmp');
        if (mfDir === null) { ret = -BigInt(ENOSYS); break; }
        if (!this.fdAvailable(0)) { ret = -BigInt(EMFILE); break; }
        const mfFd = this.allocFd();
        return {
          status: 'bridge', method: 'fs_open',
          args: { path: mfDir, flags: Number(O_TMPFILE_BIT | O_DIRECTORY_B | 2n), mode: 0o600 },
          apply: (resp: any) => {
            if (resp.ret < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); return; }
            const mfOfd = this.newOfdIdPublic();
            this.fdTable.set(mfFd, {
              type: 'native', nativeFd: resp.ret, path: '/memfd:' + mfName,
              ofdId: mfOfd, isDir: false, ftype: 0o100000, unlinked: true, sealable: true,
            });
            // a memfd always ANSWERS about its seals; one created without MFD_ALLOW_SEALING simply
            // starts out sealed against sealing, which is how linux says "not this one".
            if (!(mfFlags & MFD_ALLOW_SEALING)) this.fileSeals.set(mfOfd, 0x1);
            this.nativeRefs.set(resp.ret, 1);
            this.fdFlags.set(mfFd, 2); // O_RDWR
            if (mfFlags & MFD_CLOEXEC) this.fdCloexec.add(mfFd);
            cpu.setReg(0, BigInt(mfFd));
          },
        };
      }

      case 291: { // statx(dirfd, pathname, flags, mask, statxbuf)
        // Per spec: extended stat returning struct statx (256 bytes)
        // We first get a regular stat into the LAST 128 bytes of the 256-byte statx buffer
        // (since statx is 256 bytes and stat is 128, they don't overlap), then rearrange.
        const sxFlags = Number(BigInt.asIntN(32, a2));
        // the flags and the mask are checked before the path is even looked at. only the *at()
        // flags and the sync-mode bits exist; the mask names which fields the caller wants, and
        // the reserved bit is reserved. accepting anything meant statx(-1, -1) succeeded.
        const AT_STATX_FLAGS = 0x100 | 0x800 | 0x1000 | 0x2000 | 0x4000 | 0x6000; // NOFOLLOW|NOAUTOMOUNT|EMPTY_PATH|SYNC_TYPE
        const STATX_RESERVED = 0x80000000;
        if (sxFlags & ~AT_STATX_FLAGS) { ret = -BigInt(EINVAL); break; }
        if ((sxFlags & 0x6000) === 0x6000) { ret = -BigInt(EINVAL); break; }  // both sync bits at once
        if (Number(BigInt.asUintN(32, a3)) & STATX_RESERVED) { ret = -BigInt(EINVAL); break; }
        // a bad pathname pointer faults EFAULT (stat(NULL)); AT_EMPTY_PATH (fstat) passes a valid ""
        // so it is never caught here.
        if (!(sxFlags & 0x1000) && this.ptrFaults(bytes, Number(a1), 1)) { ret = -BigInt(EFAULT); break; }
        const rawSx = readGuestString(bytes, Number(a1));
        if (this.isEmptyPath(rawSx)) { if (!(sxFlags & 0x1000)) { ret = -BigInt(ENOENT); break; } }
        else { const peSx = this.pathCheck(rawSx); if (peSx) { ret = -BigInt(peSx); break; } }
        const sxPath = this.resolveAt(a0, rawSx);
        const sxBufAddr = Number(a4);
        if (this.ptrFaults(bytes, sxBufAddr, 256)) { ret = -BigInt(EFAULT); break; } // musl routes fstat/stat here
        // a plain stat has already been laid down at sxBufAddr+128 (the halves cannot overlap);
        // widen it in place into the 256-byte statx struct at sxBufAddr.
        const widenToStatx = () => {
          const tmpOffset = sxBufAddr + 128;
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
          const stAtimeNs = view.getBigInt64(st + 80, true);
          const stMtime = view.getBigInt64(st + 88, true);
          const stMtimeNs = view.getBigInt64(st + 96, true);
          const stCtime = view.getBigInt64(st + 104, true);
          const stCtimeNs = view.getBigInt64(st + 112, true);

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
          view.setUint32(b + 72, Number(stAtimeNs), true);
          // stx_btime (birth time) at offset 80 — use ctime
          view.setBigInt64(b + 80, stCtime, true);
          view.setUint32(b + 88, Number(stCtimeNs), true);
          // stx_ctime at offset 96
          view.setBigInt64(b + 96, stCtime, true);
          view.setUint32(b + 104, Number(stCtimeNs), true);
          // stx_mtime at offset 112
          view.setBigInt64(b + 112, stMtime, true);
          view.setUint32(b + 120, Number(stMtimeNs), true);
          // statx reports a device as its two HALVES, not as the packed number stat uses -- that is
          // the point of the field pair. writing the packed value into the minor half and zero into
          // the major reported /dev/sda1 (8:1) as 0:2049.
          //
          // the packing is linux's: minor is bits 0-7 and 20-31, major is bits 8-19.
          const devMajor = (d: bigint) => Number((d >> 8n) & 0xfffn);
          const devMinor = (d: bigint) => Number((d & 0xffn) | ((d >> 12n) & 0xfff00n));
          // stx_rdev_major/minor at offset 128
          view.setUint32(b + 128, devMajor(BigInt.asUintN(64, stRdev)), true);
          view.setUint32(b + 132, devMinor(BigInt.asUintN(64, stRdev)), true);
          // stx_dev_major/minor at offset 136
          view.setUint32(b + 136, devMajor(BigInt.asUintN(64, stDev)), true);
          view.setUint32(b + 140, devMinor(BigInt.asUintN(64, stDev)), true);
          // stx_mnt_id at offset 144. stx_mask already claims the field is valid, and a caller
          // checks it against /proc/self/mountinfo -- which lists exactly one mount, id 1.
          view.setBigUint64(b + 144, 1n, true);
        };

        const sxTmp = sxBufAddr + 128;
        const sxEntry = (sxFlags & 0x1000) ? this.getFdEntry(Number(BigInt.asIntN(32, a0))) : undefined;
        if (sxEntry?.type === 'native') {
          return {
            status: 'bridge', method: 'fs_stat',
            args: { path: sxEntry.path },
            apply: (resp: any) => {
              if (resp.ret < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); return; }
              writeBridgeStat(view, sxTmp, resp);
              widenToStatx();
              cpu.setReg(0, 0n);
            },
          };
        }
        if (sxEntry && sxEntry.type !== 'vfs') {
          // glibc implements fstat() as statx(fd, "", AT_EMPTY_PATH), so anonymous fds land here too
          writeChardevStat(view, sxTmp);
          widenToStatx();
          ret = 0n;
          break;
        }
        const bp291 = (sxFlags & 0x1000) ? null : this.bridgePath(sxPath);
        if (bp291 !== null) {
          return {
            status: 'bridge', method: (sxFlags & 0x100) ? 'fs_lstat' : 'fs_stat',
            args: { path: bp291 },
            apply: (resp: any) => {
              if (resp.ret < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); return; }
              writeBridgeStat(view, sxTmp, resp);
              widenToStatx();
              cpu.setReg(0, 0n);
            },
          };
        }
        if (this.vfs) {
          const res = this.vfs.fstatat(Number(BigInt.asIntN(32, a0)), sxPath, sxTmp, sxFlags, view);
          if (res < 0) { ret = BigInt(res); break; }
          widenToStatx();
        } else {
          ret = -BigInt(ENOENT);
          break;
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
        const EFD_CLOEXEC = 0o2000000;
        // three flags exist and nothing else does. accepting the rest also meant EFD_CLOEXEC was
        // silently dropped, so an eventfd asked to close on exec survived it.
        if (efdFlags & ~(EFD_SEMAPHORE | EFD_NONBLOCK | EFD_CLOEXEC)) { ret = -BigInt(EINVAL); break; }
        const efd = this.allocFd();
        if (efd < 0) { ret = -BigInt(EMFILE); break; }
        const efdId = this.eventfdHook?.({
          op: 'create', initval: Number(initVal), semaphore: (efdFlags & EFD_SEMAPHORE) !== 0,
        })?.[0] ?? -1;
        if (efdId < 0) { ret = -BigInt(ENOMEM); break; }
        this.fdTable.set(efd, {
          type: 'eventfd',
          id: efdId,
          semaphore: (efdFlags & EFD_SEMAPHORE) !== 0,
        });
        this.fdFlags.set(efd, efdFlags & EFD_NONBLOCK);
        if (efdFlags & EFD_CLOEXEC) this.fdCloexec.add(efd);
        ret = BigInt(efd);
        break;
      }

      case 20: { // epoll_create1(flags)
        // the only flag that exists is EPOLL_CLOEXEC, and anything else is EINVAL. accepting
        // whatever was passed also meant EPOLL_CLOEXEC itself did nothing, so an epoll fd survived
        // exec when the caller had asked for exactly the opposite.
        const EPOLL_CLOEXEC = 0o2000000;
        const epFlags = Number(a0);
        if (epFlags & ~EPOLL_CLOEXEC) { ret = -BigInt(EINVAL); break; }
        const epfd = this.allocFd();
        if (epfd < 0) { ret = -BigInt(EMFILE); break; }
        this.fdTable.set(epfd, { type: 'epoll' });
        if (epFlags & EPOLL_CLOEXEC) this.fdCloexec.add(epfd);
        if (!this.epollSets) this.epollSets = new Map();
        this.epollSets.set(epfd, new Map());
        ret = BigInt(epfd);
        break;
      }

      case 21: { // epoll_ctl(epfd, op, fd, event)
        const epfd = Number(BigInt.asIntN(32, a0));
        // every one of these is an int in the kernel's prototype. read as a 64-bit value, fd -1
        // becomes 1.8e19 and op -1 becomes a number no comparison matches -- which is how an
        // epoll_ctl on a descriptor that does not exist came back as success.
        const epOp = Number(BigInt.asIntN(32, a1)); // 1=ADD, 2=DEL, 3=MOD
        const targetFd = Number(BigInt.asIntN(32, a2));
        const eventPtr = Number(a3);
        const EPOLL_CTL_ADD = 1, EPOLL_CTL_DEL = 2, EPOLL_CTL_MOD = 3;

        // the order of these checks is observable, and ltp checks it. a bad epfd is EBADF before
        // anything else; an epfd that is a valid descriptor but not an epoll one is EINVAL; and
        // watching an epoll instance with itself is EINVAL, not a loop.
        if (!this.fdExists(epfd)) { ret = -BigInt(EBADF); break; }
        if (!this.epollSets?.has(epfd)) { ret = -BigInt(EINVAL); break; }
        if (!this.fdExists(targetFd)) { ret = -BigInt(EBADF); break; }
        if (epfd === targetFd) { ret = -BigInt(EINVAL); break; }
        if (epOp !== EPOLL_CTL_ADD && epOp !== EPOLL_CTL_DEL && epOp !== EPOLL_CTL_MOD) {
          ret = -BigInt(EINVAL); break;
        }
        const eset = this.epollSets.get(epfd)!;

        if (epOp === EPOLL_CTL_DEL) {
          if (!eset.has(targetFd)) { ret = -BigInt(ENOENT); break; }
          eset.delete(targetFd);
          ret = 0n;
          break;
        }
        // ADD and MOD both read the event struct; DEL is defined to ignore it entirely (and is
        // documented as accepting NULL there, which is why the read has to come after the branch).
        if (this.ptrFaults(bytes, eventPtr, EPOLL_EVENT_SIZE)) { ret = -BigInt(EFAULT); break; }
        // a regular file or a directory cannot be polled: it is always ready, so the kernel refuses
        // to let you wait on one rather than spinning the caller forever. EPERM, and linux decides
        // it from the file itself BEFORE looking the descriptor up in the interest list, so it wins
        // over both EEXIST and ENOENT.
        if (!this.fdPollable(targetFd)) { ret = -BigInt(EPERM); break; }
        if (epOp === EPOLL_CTL_ADD && eset.has(targetFd)) { ret = -BigInt(EEXIST); break; }
        if (epOp === EPOLL_CTL_MOD && !eset.has(targetFd)) { ret = -BigInt(ENOENT); break; }
        // watching one epoll instance with another is allowed, but only so far: a cycle would make
        // a wait chase itself, and a deep chain would make one event walk arbitrarily far. linux
        // answers the first with ELOOP and the second with EINVAL, and checks them in that order.
        if (epOp === EPOLL_CTL_ADD && this.epollSets.has(targetFd)) {
          if (this.epollReaches(targetFd, epfd)) { ret = -BigInt(ELOOP); break; }
          if (Math.max(this.epollDepth(epfd), 1 + this.epollDepth(targetFd)) > EPOLL_MAX_DEPTH) {
            ret = -BigInt(EINVAL); break;
          }
        }
        const events = view.getUint32(eventPtr + GUEST_MEM_BASE, true);
        const data = view.getBigUint64(eventPtr + GUEST_MEM_BASE + EPOLL_DATA_OFFSET, true);
        // a fresh registration has seen nothing yet, so an edge-triggered watcher counts whatever
        // is ready now as an arrival. a MOD re-arms: it keeps the descriptor but starts over.
        eset.set(targetFd, { events, data, lastReady: 0 });
        ret = 0n;
        break;
      }

      case 22: { // epoll_pwait(epfd, events, maxevents, timeout, sigmask)
        // maxevents is an int: read it as one. taken as a 64-bit unsigned, -1 became four billion
        // and the bounds check on the output array turned an EINVAL into an EFAULT.
        const epSigmask = Number(a4);
        if (epSigmask !== 0 && this.ptrFaults(bytes, epSigmask, 8)) { ret = -BigInt(EFAULT); break; }
        const epSaved = epSigmask !== 0 ? (this.signalMask ?? 0n) : null;
        if (epSigmask !== 0) this.setSignalMask(view.getBigUint64(epSigmask + GUEST_MEM_BASE, true));
        ret = BigInt(this.epollWait(view, bytes, Number(a0), Number(a1),
                                    Number(BigInt.asIntN(32, a2)), Number(BigInt.asIntN(32, a3))));
        if (epSaved !== null) this.setSignalMask(epSaved);
        break;
      }

      case 72: { // pselect6(nfds, readfds, writefds, exceptfds, timeout, sigmask)
        // nfds is an int, and a negative one is EINVAL. read as a 64-bit unsigned, -1 became
        // eighteen quintillion and the scan over the fd sets never finished -- a hang, not an
        // error. linux also clamps a too-large count to the size of the fd table rather than
        // walking past it.
        const nfdsArg = Number(BigInt.asIntN(32, a0));
        if (nfdsArg < 0) { ret = -BigInt(EINVAL); break; }
        const nfds = Math.min(nfdsArg, this.nofileLimit);
        const readPtr = Number(a1);
        const writePtr = Number(a2);
        const exceptPtr = Number(a3);
        const tmoPtr = Number(a4);

        // select MODIFIES the sets it is given, so each one has to be writable, not merely readable.
        const setBytes = (nfds + 7) >> 3;
        let selFault = false;
        for (const p of [readPtr, writePtr, exceptPtr]) {
          if (p !== 0 && setBytes > 0 && this.ptrWriteFaults(bytes, p, setBytes)) { selFault = true; break; }
        }
        if (selFault) { ret = -BigInt(EFAULT); break; }
        if (tmoPtr !== 0 && this.ptrFaults(bytes, tmoPtr, 16)) { ret = -BigInt(EFAULT); break; }

        let tmoMs = -1;
        if (tmoPtr !== 0) {
          const sec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE, true));
          const nsec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE + 8, true));
          if (sec < 0 || nsec < 0 || nsec >= 1_000_000_000) { ret = -BigInt(EINVAL); break; }
          tmoMs = sec * 1000 + Math.floor(nsec / 1_000_000);
        }

        const wantRead = this.readFdSet(view, readPtr, nfds);
        const wantWrite = this.readFdSet(view, writePtr, nfds);
        const wantExcept = this.readFdSet(view, exceptPtr, nfds);
        // every descriptor named in any of the sets has to be open. one that is not is EBADF, and
        // waiting on it instead is a wait for something that can never happen.
        let selBad = false;
        for (const set of [wantRead, wantWrite, wantExcept]) {
          for (const fd of set) if (!this.fdIsOpen(fd)) { selBad = true; break; }
          if (selBad) break;
        }
        if (selBad) { ret = -BigInt(EBADF); break; }

        // pselect's sixth argument is { const sigset_t *ss; size_t ss_len }, and the mask it points
        // at is installed for the duration of the wait -- that atomicity is the only reason to call
        // pselect rather than select. it comes off again at every exit below.
        let selSaved: bigint | null = null;
        const selSigPtr = Number(a5);
        if (selSigPtr !== 0) {
          if (this.ptrFaults(bytes, selSigPtr, 16)) { ret = -BigInt(EFAULT); break; }
          const ssPtr = Number(view.getBigUint64(selSigPtr + GUEST_MEM_BASE, true));
          const ssLen = Number(view.getBigUint64(selSigPtr + GUEST_MEM_BASE + 8, true));
          if (ssPtr !== 0) {
            if (ssLen !== 8) { ret = -BigInt(EINVAL); break; }
            if (this.ptrFaults(bytes, ssPtr, 8)) { ret = -BigInt(EFAULT); break; }
            selSaved = this.signalMask ?? 0n;
            this.setSignalMask(view.getBigUint64(ssPtr + GUEST_MEM_BASE, true));
          }
        }

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
            if (sockEntry?.type !== 'socket') {
              if (selSaved !== null) this.setSignalMask(selSaved);
              ret = -BigInt(EBADF); break;
            }
            const pollTmo = tmoMs < 0 ? 30000 : tmoMs;
            return {
              status: 'bridge', method: 'net_poll',
              args: { socketId: sockEntry.socketId, timeoutMs: pollTmo },
              apply: (resp: any) => {
                if (selSaved !== null) this.setSignalMask(selSaved);
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
        let deadline = tmoMs < 0 ? -1 : this.nowMs() + tmoMs;

        let selSeen = this.ioGeneration();
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

          const remaining = deadline < 0 ? -1 : Math.max(0, deadline - this.nowMs());
          if (deadline >= 0 && remaining === 0) {
            this.writeFdSet(view, readPtr, nfds, readyRead);
            this.writeFdSet(view, writePtr, nfds, readyWrite);
            if (exceptPtr !== 0) this.writeFdSet(view, exceptPtr, nfds, new Set());
            ret = 0n;
            break;
          }

          if (!this.parkForIo(selSeen, remaining)) {
            // nothing here can block at all; behave as poll
            this.writeFdSet(view, readPtr, nfds, readyRead);
            this.writeFdSet(view, writePtr, nfds, readyWrite);
            if (exceptPtr !== 0) this.writeFdSet(view, exceptPtr, nfds, new Set());
            ret = 0n;
            break;
          }
          // a signal interrupts the wait, and linux leaves the descriptor sets untouched when it
          // does. without this a select() slept through every signal aimed at it.
          if (this.hasPendingDeliverable()) { ret = -BigInt(EINTR); break; }
          selSeen = this.ioGeneration();

          if (deadline >= 0 && this.nowMs() >= deadline) {
            // Final empty result
            this.writeFdSet(view, readPtr, nfds, readyRead);
            this.writeFdSet(view, writePtr, nfds, readyWrite);
            if (exceptPtr !== 0) this.writeFdSet(view, exceptPtr, nfds, new Set());
            ret = 0n;
            break;
          }
          // loop back, re-check
        }
        if (selSaved !== null) this.setSignalMask(selSaved);
        break;
      }

      case 73: { // ppoll(fds, nfds, tmo_p, sigmask)
        // struct pollfd { int fd; short events; short revents; }  — 8 bytes each
        const POLLIN = 1, POLLOUT = 4;
        const fdsAddr = Number(a0);
        // nfds is unsigned, and linux refuses one larger than the process may have descriptors --
        // it is a promise about how much of the array is real. taking it at face value walked off
        // the end of the array instead.
        const nfds = Number(BigInt.asUintN(64, a1));
        if (nfds > this.nofileLimit) { ret = -BigInt(EINVAL); break; }
        if (nfds > 0 && (fdsAddr === 0 || this.ptrWriteFaults(bytes, fdsAddr, nfds * 8))) {
          ret = -BigInt(EFAULT); break;
        }
        const tmoPtr = Number(a2);
        if (tmoPtr !== 0 && this.ptrFaults(bytes, tmoPtr, 16)) { ret = -BigInt(EFAULT); break; }
        // the mask is the whole difference between poll and ppoll: it is installed for the wait
        // and taken off afterwards, which is how a program waits for exactly the signals it names
        // without racing between unblocking them and starting to wait.
        const ppSigmask = Number(a3);
        if (ppSigmask !== 0 && this.ptrFaults(bytes, ppSigmask, 8)) { ret = -BigInt(EFAULT); break; }
        const ppSaved = ppSigmask !== 0 ? (this.signalMask ?? 0n) : null;
        if (ppSigmask !== 0) this.setSignalMask(view.getBigUint64(ppSigmask + GUEST_MEM_BASE, true));

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
            // a descriptor that is not open is not a wait, it is an answer: POLLNVAL, reported
            // whether or not the caller asked for it. polling one forever instead is a hang.
            if (!this.fdIsOpen(pfd.fd)) { pfd.revents = 0x20; count++; continue; }
            const entry = this.getFdEntry(pfd.fd);
            if (entry?.type === 'socket') continue; // socket readiness handled by bridge
            if ((pfd.events & POLLIN) && this.isFdReadable(pfd.fd)) pfd.revents |= POLLIN;
            if ((pfd.events & POLLOUT) && this.isFdWritable(pfd.fd)) pfd.revents |= POLLOUT;
            const pe = this.fdTable.get(pfd.fd);
            if (pe?.type === 'socket') {
              if ((pfd.events & 0x2000) && pe.shutRd) pfd.revents |= 0x2000; // POLLRDHUP
              if (pe.shutRd && pe.shutWr) pfd.revents |= 0x10;               // POLLHUP
            }
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
        let deadline = tmoMs < 0 ? -1 : this.nowMs() + tmoMs;
        let pollSeen = this.ioGeneration();
        while (true) {
          const ready = checkReady();
          if (ready > 0) {
            writeBack();
            ret = BigInt(ready);
            break;
          }
          const remaining = deadline < 0 ? -1 : Math.max(0, deadline - this.nowMs());
          if (deadline >= 0 && remaining === 0) {
            writeBack();
            ret = 0n;
            break;
          }
          if (!this.parkForIo(pollSeen, remaining)) {
            writeBack();
            ret = 0n;
            break;
          }
          if (this.hasPendingDeliverable()) { ret = -BigInt(EINTR); break; }
          pollSeen = this.ioGeneration();
          if (deadline >= 0 && this.nowMs() >= deadline) {
            writeBack();
            ret = 0n;
            break;
          }
        }
        if (ppSaved !== null) this.setSignalMask(ppSaved);
        break;
      }

      // ── Remaining syscalls from aarch64 spec ──

      // mount(source, target, filesystemtype, mountflags, data). the emulator has a
      // single in-memory tree, so a mount records the mount point in the namespace
      // (visible via /proc/self/mounts and mountinfo) rather than grafting a real fs.
      // the ltp scratch-device tests select tmpfs (LTP_DEV_FS_TYPE=tmpfs), which needs
      // no backing device or mkfs; the mount point directory already exists.
      case 40:  // mount
        // there are no block devices to mount: the filesystem is the host's, and a guest cannot
        // add another. reporting success was worse than useless -- a caller that believes its
        // filesystem got mounted (ltp's device-backed tests do) then fails much later and much
        // more confusingly. ENODEV is what linux says when the fs type is not available.
        ret = -BigInt(19);
        break;
      case 39:  // umount2
      case 60:  // (legacy no-op slot)
      case 442: // mount_setattr
        ret = 0n; // nothing was ever mounted, so unmounting is vacuously fine
        break;
      case 41: // pivot_root
        ret = -BigInt(EPERM); // not permitted (we don't support changing root)
        break;

      // Robust futex list
      case 100: { // get_robust_list(pid, head_ptr, len_ptr)
        // both out-parameters are required: the kernel writes the head AND its size, so a null
        // either side is EFAULT. this wrote through whatever address it was handed, including 0.
        const grlPid = Number(BigInt.asIntN(32, a0));
        const grlHead = Number(a1), grlLen = Number(a2);
        if (grlHead === 0 || grlLen === 0) { ret = -BigInt(EFAULT); break; }
        if (this.ptrFaults(bytes, grlHead, 8) || this.ptrFaults(bytes, grlLen, 8)) { ret = -BigInt(EFAULT); break; }
        if (grlPid !== 0 && grlPid !== this.currentPid && !this.pidExists(grlPid)) { ret = -BigInt(ESRCH); break; }
        // another process's list is not ours to read: linux wants ptrace-level access for that.
        if (grlPid !== 0 && grlPid !== this.currentPid && !this.hasCap(CAP_SYS_PTRACE)) { ret = -BigInt(EPERM); break; }
        view.setBigUint64(grlHead + GUEST_MEM_BASE, this.robustListHead, true);
        view.setBigUint64(grlLen + GUEST_MEM_BASE, 24n, true); // sizeof(struct robust_list_head)
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
      case 210: { // shutdown(sockfd, how)
        const shSock = this.sockFd(Number(a0));
        if ('err' in shSock) { ret = -BigInt(shSock.err); break; }
        const how = Number(BigInt.asIntN(32, a1));
        if (how < 0 || how > 2) { ret = -BigInt(EINVAL); break; }
        // shutting down half of a connection that was never made is ENOTCONN, which is how a
        // program learns the peer went away before it got there.
        if (shSock.e.type === 'socket' && !shSock.e.remoteAddr && !shSock.e.listening) {
          ret = -BigInt(ENOTCONN); break;
        }
        // which half was shut down is not a formality: writing to a socket whose sending half is
        // gone is EPIPE, and reading from one whose receiving half is gone is end of file.
        if (shSock.e.type === 'socket') {
          if (how === 0 || how === 2) shSock.e.shutRd = true;
          if (how === 1 || how === 2) shSock.e.shutWr = true;
        }
        ret = 0n;
        break;
      }
      case 211: { // sendmsg(fd, *msghdr, flags)
        // struct msghdr { msg_name(8), msg_namelen(4), pad(4), msg_iov(8), msg_iovlen(8),
        //                 msg_control(8), msg_controllen(8), msg_flags(4) }
        const smSock = this.sockFd(Number(a0));
        if ('err' in smSock) { ret = -BigInt(smSock.err); break; }
        if (smSock.e.type !== 'socket') { ret = -BigInt(EOPNOTSUPP); break; }
        const smEntry = smSock.e;
        // the header, the iovec array and every buffer it names have to be readable before any of
        // this is a message. none of them were checked: a bad pointer anywhere read whatever
        // happened to be at that address and sent it.
        if (Number(a1) === 0 || this.ptrFaults(bytes, Number(a1), 56)) { ret = -BigInt(EFAULT); break; }
        const mhdr = Number(a1) + GUEST_MEM_BASE;
        const iovPtr = Number(view.getBigUint64(mhdr + 16, true));
        const iovLen = Number(view.getBigUint64(mhdr + 24, true));
        if (iovLen > 1024) { ret = -BigInt(EMSGSIZE); break; }
        if (iovLen > 0 && (iovPtr === 0 || this.ptrFaults(bytes, iovPtr, iovLen * 16))) {
          ret = -BigInt(EFAULT); break;
        }
        const smFlags = Number(BigInt.asIntN(32, a2));
        if ((smFlags & 1) && smEntry.sockType !== 1) { ret = -BigInt(EOPNOTSUPP); break; } // MSG_OOB
        // Gather all iov data
        let totalLen = 0;
        let smFault = false;
        for (let i = 0; i < iovLen; i++) {
          const base = Number(view.getBigUint64(iovPtr + i * 16 + GUEST_MEM_BASE, true));
          const len = Number(view.getBigUint64(iovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
          if (len > 0 && this.ptrFaults(bytes, base, len)) { smFault = true; break; }
          totalLen += len;
        }
        if (smFault) { ret = -BigInt(EFAULT); break; }
        if (smEntry.sockType === 2 && totalLen > (smEntry.domain === 10 ? 65527 : 65507)) {
          ret = -BigInt(EMSGSIZE); break;
        }
        const smNamePtr0 = Number(view.getBigUint64(mhdr, true));
        if (smEntry.sockType === 1 && (!smEntry.remoteAddr || smEntry.shutWr) && smNamePtr0 === 0) {
          if (!(smFlags & 0x4000)) this.raiseSignal(13);
          ret = -BigInt(EPIPE); break;
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
            args: { socketId: smEntry.socketId, data: smB64, host: dest.host, port: dest.port,
                    more: (Number(BigInt.asIntN(32, a2)) & 0x8000) ? 1 : 0 },
            apply: (resp: any) => {
              cpu.setReg(0, resp.ret < 0
                ? BigInt.asUintN(64, BigInt(resp.ret))
                : BigInt(totalLen));
            },
          };
        }
        return {
          status: 'bridge', method: 'net_write',
          args: { socketId: smEntry.socketId, data: smB64,
                  more: (Number(BigInt.asIntN(32, a2)) & 0x8000) ? 1 : 0 },
          apply: (resp: any) => {
            cpu.setReg(0, resp.ret < 0
              ? BigInt.asUintN(64, BigInt(resp.ret))
              : BigInt(totalLen));
          },
        };
      }

      case 212: { // recvmsg(fd, *msghdr, flags)
        const rmSock = this.sockFd(Number(a0));
        if ('err' in rmSock) { ret = -BigInt(rmSock.err); break; }
        if (rmSock.e.type !== 'socket') { ret = -BigInt(EOPNOTSUPP); break; }
        const rmEntry = rmSock.e;
        if (Number(a1) === 0 || this.ptrWriteFaults(bytes, Number(a1), 56)) { ret = -BigInt(EFAULT); break; }
        const rmhdr = Number(a1) + GUEST_MEM_BASE;
        const rmIovPtr = Number(view.getBigUint64(rmhdr + 16, true));
        const rmIovLen = Number(view.getBigUint64(rmhdr + 24, true));
        if (rmIovLen > 1024) { ret = -BigInt(EMSGSIZE); break; }
        if (rmIovLen > 0 && (rmIovPtr === 0 || this.ptrFaults(bytes, rmIovPtr, rmIovLen * 16))) {
          ret = -BigInt(EFAULT); break;
        }
        // Calculate total buffer capacity from iovec
        let rmCapacity = 0;
        for (let i = 0; i < rmIovLen; i++) {
          rmCapacity += Number(view.getBigUint64(rmIovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
        }
        const rmFlags = Number(BigInt.asIntN(32, a2));
        if (rmFlags & 1) { ret = -BigInt(EINVAL); break; }        // MSG_OOB
        if (rmFlags & 0x2000) { ret = -BigInt(EAGAIN); break; }   // MSG_ERRQUEUE
        const rmNonblock = this.isNonBlockingFd(Number(a0)) || (rmFlags & 0x40) !== 0;
        return {
          status: 'bridge', method: 'net_read',
          args: { socketId: rmEntry.socketId, count: Math.min(rmCapacity, BRIDGE_READ_MAX),
                  nonblock: rmNonblock ? 1 : 0, peek: (rmFlags & 2) ? 1 : 0 },
          apply: (resp: any) => {
            this.nt(`recvmsg fd=${Number(a0)} cap=${rmCapacity} -> ret=${resp.ret} dlen=${resp.data ? Math.floor(resp.data.length * 3 / 4) : 0}`);
            if (resp.ret === -11 && !rmNonblock) { this.retryAfterBridge = true; return; }
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
            const rmFrom = this.peerFromMeta(resp) ?? rmEntry.remoteAddr;
            if (namePtr !== 0 && rmEntry.sockType === 2 && rmFrom) {
              const e = this.putSockaddr(bytes, view, namePtr, nameLenPtr - GUEST_MEM_BASE, rmFrom);
              if (e < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(e))); return; }
            } else {
              view.setUint32(nameLenPtr, 0, true);
            }
            // Clear msg_flags and msg_controllen
            view.setInt32(rmhdr + 48, 0, true);     // msg_flags
            view.setBigUint64(rmhdr + 40, 0n, true); // msg_controllen
            cpu.setReg(0, BigInt(recvd.length));
          },
        };
      }

      case 213: { // readahead(fd, offset, count)
        const raFd = Number(BigInt.asIntN(32, a0));
        // read-ahead is a hint to a page cache, so it needs a descriptor open for reading and
        // something with a page cache behind it. a pipe or a socket has neither.
        if (!this.fdIsOpen(raFd) || this.fdIsPath(raFd) || !this.fdAccess(raFd).readable) { ret = -BigInt(EBADF); break; }
        const raEnt = this.getFdEntry(raFd);
        if (raEnt?.type === 'pipe_read' || raEnt?.type === 'pipe_write') { ret = -BigInt(29); break; } // ESPIPE
        // only something with a page cache behind it: a regular file, or one of the synthesised
        // /proc files. a directory, a device node and a socket have none.
        if (raEnt?.type === 'native') {
          if (raEnt.isDir === true) { ret = -BigInt(EINVAL); break; }
        } else if ((this.vfs?.fdOfdType(raFd) ?? null) !== 'file') {
          ret = -BigInt(EINVAL); break;
        }
        // the pages are already wherever they are; nothing to do but say so.
        ret = 0n;
        break;
      }
      case 269: { // sendmmsg(sockfd, mmsghdr*, vlen, flags)
        // struct mmsghdr { struct msghdr msg_hdr; unsigned msg_len; } -- 56 bytes of header, then
        // the count the kernel fills in, padded out to 64.
        const smmSock = this.sockFd(Number(a0));
        if ('err' in smmSock) { ret = -BigInt(smmSock.err); break; }
        if (smmSock.e.type !== 'socket') { ret = -BigInt(EOPNOTSUPP); break; }
        const smmEntry = smmSock.e;
        const smmVec = Number(a1);
        let smmLen = Number(BigInt.asUintN(32, a2));
        if (smmLen === 0) { ret = 0n; break; }
        if (smmLen > 1024) smmLen = 1024; // UIO_MAXIOV: linux caps the batch rather than refusing it
        if (this.ptrWriteFaults(bytes, smmVec, smmLen * 64)) { ret = -BigInt(EFAULT); break; }
        const smmMsgs: Array<{ data: string; host: string; port: number }> = [];
        const smmSizes: number[] = [];
        let smmErr = 0;
        for (let m = 0; m < smmLen; m++) {
          const hdr = smmVec + m * 64 + GUEST_MEM_BASE;
          const iovPtr = Number(view.getBigUint64(hdr + 16, true));
          const iovCount = Number(view.getBigUint64(hdr + 24, true));
          if (iovCount > 1024) { smmErr = EMSGSIZE; break; }
          let total = 0;
          for (let i = 0; i < iovCount; i++) {
            total += Number(view.getBigUint64(iovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
          }
          if (smmEntry.sockType === 2 && total > (smmEntry.domain === 10 ? 65527 : 65507)) {
            smmErr = EMSGSIZE; break;
          }
          const payload = new Uint8Array(total);
          let off = 0;
          for (let i = 0; i < iovCount; i++) {
            const base = Number(view.getBigUint64(iovPtr + i * 16 + GUEST_MEM_BASE, true));
            const len = Number(view.getBigUint64(iovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
            if (len > 0 && this.ptrFaults(bytes, base, len)) { smmErr = EFAULT; break; }
            payload.set(bytes.subarray(base + GUEST_MEM_BASE, base + GUEST_MEM_BASE + len), off);
            off += len;
          }
          if (smmErr) break;
          const namePtr = Number(view.getBigUint64(hdr, true));
          const nameLen = view.getUint32(hdr + 8, true);
          const dest = namePtr !== 0 && nameLen > 0
            ? this.parseSockaddr(bytes, namePtr, nameLen) : null;
          smmMsgs.push({ data: bytesToBase64(payload), host: dest ? dest.host : '', port: dest ? dest.port : 0 });
          smmSizes.push(total);
        }
        // a batch that could not even build its first message reports why; one that built some
        // sends those, exactly as linux does with a partial batch.
        if (smmErr && smmMsgs.length === 0) { ret = -BigInt(smmErr); break; }
        if (smmEntry.sockType === 1 && (!smmEntry.remoteAddr || smmEntry.shutWr) && !smmMsgs.some((m) => m.host)) {
          this.raiseSignal(13);
          ret = -BigInt(EPIPE); break;
        }
        return {
          status: 'bridge', method: 'net_sendmmsg',
          args: { socketId: smmEntry.socketId, msgs: smmMsgs },
          apply: (resp: any) => {
            if (resp.ret < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); return; }
            for (let m = 0; m < resp.ret && m < smmSizes.length; m++) {
              view.setUint32(smmVec + m * 64 + 56 + GUEST_MEM_BASE, smmSizes[m], true);
            }
            cpu.setReg(0, BigInt(resp.ret));
          },
        };
      }

      case 243: { // recvmmsg(sockfd, mmsghdr*, vlen, flags, timeout)
        const rmmSock = this.sockFd(Number(a0));
        if ('err' in rmmSock) { ret = -BigInt(rmmSock.err); break; }
        if (rmmSock.e.type !== 'socket') { ret = -BigInt(EOPNOTSUPP); break; }
        const rmmEntry = rmmSock.e;
        const rmmVec = Number(a1);
        let rmmLen = Number(BigInt.asUintN(32, a2));
        if (rmmLen === 0) { ret = 0n; break; }
        if (rmmLen > 1024) rmmLen = 1024;
        if (this.ptrWriteFaults(bytes, rmmVec, rmmLen * 64)) { ret = -BigInt(EFAULT); break; }
        if (rmmEntry.shutRd) { ret = 0n; break; }
        // the per-message capacity, so a datagram longer than the buffer it lands in is truncated
        // per message rather than by whatever the largest one happened to be.
        const rmmCaps: number[] = [];
        for (let m = 0; m < rmmLen; m++) {
          const hdr = rmmVec + m * 64 + GUEST_MEM_BASE;
          const iovPtr = Number(view.getBigUint64(hdr + 16, true));
          const iovCount = Number(view.getBigUint64(hdr + 24, true));
          let cap = 0;
          for (let i = 0; i < iovCount; i++) {
            cap += Number(view.getBigUint64(iovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
          }
          rmmCaps.push(cap);
        }
        // MSG_DONTWAIT, or a zero timeout, means "whatever is already here".
        const rmmFlags = Number(BigInt.asIntN(32, a3));
        let rmmTimeout = -1;
        if (Number(a4) !== 0) {
          if (this.ptrFaults(bytes, Number(a4), 16)) { ret = -BigInt(EFAULT); break; }
          const tsec = Number(view.getBigInt64(Number(a4) + GUEST_MEM_BASE, true));
          const tnsec = Number(view.getBigInt64(Number(a4) + 8 + GUEST_MEM_BASE, true));
          if (tsec < 0 || tnsec < 0 || tnsec > 999999999) { ret = -BigInt(EINVAL); break; }
          rmmTimeout = tsec * 1000 + Math.ceil(tnsec / 1e6);
        }
        if (rmmFlags & 0x40) rmmTimeout = 0;                       // MSG_DONTWAIT
        if (this.isNonBlockingFd(Number(a0))) rmmTimeout = 0;
        if (rmmTimeout < 0) rmmTimeout = 30000;
        const rmmMax = Math.min(Math.max(...rmmCaps, 0), BRIDGE_READ_MAX);
        return {
          status: 'bridge', method: 'net_recvmmsg',
          args: { socketId: rmmEntry.socketId, vlen: rmmLen, maxLen: rmmMax, timeoutMs: rmmTimeout },
          apply: (resp: any) => {
            if (resp.ret < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); return; }
            const msgs: Array<{ data: string; host: string; port: number }> =
              resp.bytes ? JSON.parse(new TextDecoder().decode(resp.bytes)) : [];
            for (let m = 0; m < msgs.length; m++) {
              const hdr = rmmVec + m * 64 + GUEST_MEM_BASE;
              const payload = Uint8Array.from(atob(msgs[m].data), (c) => c.charCodeAt(0));
              const iovPtr = Number(view.getBigUint64(hdr + 16, true));
              const iovCount = Number(view.getBigUint64(hdr + 24, true));
              let remaining = Math.min(payload.length, rmmCaps[m]);
              const written = remaining;
              let src = 0;
              for (let i = 0; i < iovCount && remaining > 0; i++) {
                const base = Number(view.getBigUint64(iovPtr + i * 16 + GUEST_MEM_BASE, true));
                const len = Number(view.getBigUint64(iovPtr + i * 16 + 8 + GUEST_MEM_BASE, true));
                const take = Math.min(len, remaining);
                bytes.set(payload.subarray(src, src + take), base + GUEST_MEM_BASE);
                src += take; remaining -= take;
              }
              const namePtr = Number(view.getBigUint64(hdr, true));
              if (namePtr !== 0 && msgs[m].host) {
                const wrote = this.writeSockaddr(view, namePtr,
                  { family: msgs[m].host.includes(':') ? 10 : 2, host: msgs[m].host, port: msgs[m].port });
                view.setUint32(hdr + 8, wrote, true);
              }
              // MSG_TRUNC says the datagram was longer than the buffers it was read into.
              view.setInt32(hdr + 48, payload.length > rmmCaps[m] ? 0x20 : 0, true);
              view.setBigUint64(hdr + 40, 0n, true);
              view.setUint32(hdr + 56, written, true);
            }
            cpu.setReg(0, BigInt(msgs.length));
          },
        };
      }
      // socketpair is handled by the worker (case 199), which owns the unix-socket rings. it can
      // only reach here on the host-side test path, where there is no kernel to allocate them.
      case 199: { // socketpair(domain, type, protocol, sv[2])
        // AF_UNIX pairs are made by the worker out of the kernel's rings and never reach here.
        // for every other family the triple is still validated first -- linux creates the socket
        // before it discovers the family has no socketpair -- and only then answers EOPNOTSUPP.
        const spErr = sockTripleErrno(Number(BigInt.asIntN(32, a0)), Number(BigInt.asIntN(32, a1)),
                                      Number(BigInt.asIntN(32, a2)), this.euid);
        ret = -BigInt(spErr !== 0 ? spErr : EOPNOTSUPP);
        break;
      }

      // File operations (remaining)
      case 75: // vmsplice
      case 76: // splice
      case 77: // tee
        ret = -BigInt(ENOSYS);
        break;
      case 81: // sync -- flush everything; we have no writeback cache of our own to flush
        ret = 0n;
        break;
      case 84: { // sync_file_range(fd, offset, nbytes, flags)
        // a hint about WHEN to write back, not a durability promise -- but the arguments still
        // have to make sense, and this used to accept a closed descriptor and a made-up flag.
        const SYNC_FILE_RANGE_ALL = 7; // WAIT_BEFORE | WRITE | WAIT_AFTER
        if (Number(BigInt.asUintN(32, a3)) & ~SYNC_FILE_RANGE_ALL) { ret = -BigInt(EINVAL); break; }
        const sfrOff = BigInt.asIntN(64, a1);
        const sfrLen = BigInt.asIntN(64, a2);
        if (sfrOff < 0n || sfrLen < 0n) { ret = -BigInt(EINVAL); break; }
        const sfrFd = Number(BigInt.asIntN(32, a0));
        if (!this.fdIsOpen(sfrFd)) { ret = -BigInt(EBADF); break; }
        // writing back a RANGE only means something for a file that has ranges. a pipe, a socket
        // or a character device is ESPIPE, exactly as lseek on one would be.
        const sfrEntry = this.getFdEntry(sfrFd);
        const sfrType = sfrEntry?.type === 'native' ? (sfrEntry.ftype ?? 0) : 0;
        const sfrPipe = sfrEntry?.type === 'pipe_read' || sfrEntry?.type === 'pipe_write' ||
                        sfrEntry?.type === 'socket' || sfrEntry?.type === 'unixsock' ||
                        sfrType === 0o010000 || sfrType === 0o140000 || sfrType === 0o020000 ||
                        (!sfrEntry && !!this.vfs?.isCharDev?.(sfrFd));
        if (sfrPipe) { ret = -BigInt(ESPIPE); break; }
        ret = 0n;
        break;
      }
      case 18: // lookup_dcookie
        ret = -BigInt(ENOSYS);
        break;

      // Capabilities
      case 90: { // capget(hdrp, datap): report what this process ACTUALLY holds
        // the header was read AFTER the data and only if it was non-null, so capget(NULL, data)
        // succeeded and a version the kernel does not speak was accepted.
        const capHdr = this.capHeader(bytes, view, Number(a0));
        if ('err' in capHdr && capHdr.err !== EINVAL) { ret = -BigInt(capHdr.err); break; }
        const capDataPtr = Number(a1);
        // a null data pointer is how a caller ASKS which version the kernel prefers: the header
        // now holds the answer, and there is nothing else to report.
        if (capDataPtr === 0) { ret = 0n; break; }
        if ('err' in capHdr) { ret = -BigInt(capHdr.err); break; }
        if (capHdr.pid < 0) { ret = -BigInt(EINVAL); break; }
        if (capHdr.pid !== 0 && capHdr.pid !== this.currentPid && !this.pidExists(capHdr.pid)) {
          ret = -BigInt(ESRCH); break;
        }
        const capWords = capHdr.version === 0x19980330 ? 12 : 24;
        if (this.ptrWriteFaults(bytes, capDataPtr, capWords)) { ret = -BigInt(EFAULT); break; }
        const base = capDataPtr + GUEST_MEM_BASE;
        // struct __user_cap_data_struct x2 for v3: the low 32 bits of each set, then the high.
        const lo = (v: bigint) => Number(v & 0xFFFFFFFFn) >>> 0;
        const hi = (v: bigint) => Number((v >> 32n) & 0xFFFFFFFFn) >>> 0;
        view.setUint32(base, lo(this.capEff), true);
        view.setUint32(base + 4, lo(this.capPerm), true);
        view.setUint32(base + 8, lo(this.capInh), true);
        if (capWords === 24) {
          view.setUint32(base + 12, hi(this.capEff), true);
          view.setUint32(base + 16, hi(this.capPerm), true);
          view.setUint32(base + 20, hi(this.capInh), true);
        }
        ret = 0n;
        break;
      }
      case 91: { // capset(hdrp, datap)
        const setHdr = this.capHeader(bytes, view, Number(a0));
        if ('err' in setHdr) { ret = -BigInt(setHdr.err); break; }
        // a process may only change its OWN capabilities. naming another one is refused outright
        // rather than checked for permission -- there is no way to be allowed to do it.
        if (setHdr.pid !== 0 && setHdr.pid !== this.currentPid) { ret = -BigInt(EPERM); break; }
        const dp = Number(a1);
        if (dp === 0) { ret = -BigInt(EFAULT); break; }
        if (this.ptrFaults(bytes, dp, setHdr.version === 0x19980330 ? 12 : 24)) { ret = -BigInt(EFAULT); break; }
        const b = dp + GUEST_MEM_BASE;
        const join = (loOff: number, hiOff: number) =>
          BigInt(view.getUint32(b + loOff, true) >>> 0) | (BigInt(view.getUint32(b + hiOff, true) >>> 0) << 32n);
        const wantEff = join(0, 12), wantPerm = join(4, 16), wantInh = join(8, 20);
        // the rules that make capabilities a one-way door: you may only keep a subset of what you
        // already had, and your effective set may never exceed your permitted set. without them
        // dropping a capability would be reversible, which is the opposite of the point.
        if ((wantPerm & ~this.capPerm) !== 0n) { ret = -BigInt(EPERM); break; }
        if ((wantEff & ~wantPerm) !== 0n) { ret = -BigInt(EPERM); break; }
        if ((wantInh & ~(this.capInh | this.capPerm)) !== 0n) { ret = -BigInt(EPERM); break; }
        this.capPerm = wantPerm; this.capEff = wantEff; this.capInh = wantInh;
        ret = 0n;
        break;
      }

      // Kernel log
      case 116: { // syslog(type, buf, len) — read/control the kernel message buffer
        // there is no kernel ring buffer here, so every read comes back empty -- but the command
        // itself is checked, because "that is not a command" and "there was nothing to read" are
        // different answers and a caller acts on them differently.
        const slType = Number(BigInt.asIntN(32, a0));
        const slBuf = Number(a1);
        const slLen = Number(BigInt.asIntN(32, a2));
        if (slType < 0 || slType > 10) { ret = -BigInt(EINVAL); break; }
        // reading or clearing the buffer, and setting the console level, are privileged.
        const slPriv = slType !== 3 && slType !== 9 && slType !== 10;
        if (slPriv && !this.hasCap(CAP_SYS_ADMIN)) { ret = -BigInt(EPERM); break; }
        if (slType === 2 || slType === 3 || slType === 4) {
          if (slBuf === 0) { ret = -BigInt(EINVAL); break; }
          if (slLen < 0) { ret = -BigInt(EINVAL); break; }
          if (slLen > 0 && this.ptrWriteFaults(bytes, slBuf, slLen)) { ret = -BigInt(EFAULT); break; }
          ret = 0n; break;    // nothing buffered to hand back
        }
        if (slType === 8) {
          // the console level runs 1..8; 0 and 9 name no level.
          if (slLen < 1 || slLen > 8) { ret = -BigInt(EINVAL); break; }
        }
        // 9 (unread size) and 10 (buffer size) both describe a buffer that is empty and zero-sized.
        ret = 0n;
        break;
      }

      // Process tracing
      case 117: // ptrace — not supported in emulator
        ret = -BigInt(EPERM);
        break;

      case 127: // sched_rr_get_interval — see the sched_* note above
        ret = -BigInt(ENOSYS);
        break;
      case 128: // restart_syscall — restart interrupted syscall
        ret = -BigInt(4); // EINTR
        break;

      case 170: { // settimeofday(tv, tz)
        // the same clock clock_settime moves, spelled the older way. answering EPERM to a caller
        // that HOLDS CAP_SYS_TIME was simply wrong: the guest is root and root can set the clock.
        if (Number(a0) === 0) { ret = 0n; break; }   // tv NULL: only the (ignored) timezone was set
        if (this.ptrFaults(bytes, Number(a0), 16)) { ret = -BigInt(EFAULT); break; }
        const stvSec = view.getBigInt64(Number(a0) + GUEST_MEM_BASE, true);
        const stvUsec = view.getBigInt64(Number(a0) + GUEST_MEM_BASE + 8, true);
        if (stvUsec < 0n || stvUsec >= 1_000_000n || stvSec < 0n) { ret = -BigInt(EINVAL); break; }
        if (!this.hasCap(CAP_SYS_TIME)) { ret = -BigInt(EPERM); break; }
        // elapsedNs, not monoNs: the monotonic clock counts from the system's boot rather than
        // from this handler's construction, and the wall clock is built on the latter.
        this.clockOffsetNs = (stvSec * 1_000_000_000n + stvUsec * 1000n) -
                             (this.clkRealBaseNs + this.elapsedNs());
        ret = 0n;
        break;
      }

      // Extended attributes
      //
      // twelve syscalls, three operations x (path, link, descriptor). the differences between the
      // flavours are the target and whether a final symlink is followed; everything after that is
      // the same call into the filesystem, which owns the namespace rules ("user.*" only on regular
      // files and directories, "trusted.*" privileged) and the storage.
      case 5: case 6: case 7: case 8: case 9: case 10: case 11: case 12: case 13: case 14: case 15: case 16: {
        const byFd = nr === 7 || nr === 10 || nr === 13 || nr === 16;
        const nofollow = nr === 6 || nr === 9 || nr === 12 || nr === 15;
        const isList = nr >= 11 && nr <= 13;
        let target: string;
        if (byFd) {
          const xfd = Number(BigInt.asIntN(32, a0));
          if (!this.fdIsOpen(xfd) || this.fdIsPath(xfd) || this.vfs?.isOPath(xfd)) { ret = -BigInt(EBADF); break; }
          const xe = this.getFdEntry(xfd);
          if (xe?.type !== 'native') { ret = -BigInt(EOPNOTSUPP); break; }
          target = xe.path;
        } else {
          if (this.ptrFaults(bytes, Number(a0), 1)) { ret = -BigInt(14); break; }
          const rawX = readGuestString(bytes, Number(a0));
          const pe = this.pathCheck(rawX);
          if (pe) { ret = -BigInt(pe); break; }
          // the synthesised /proc and /dev keep no attributes, and neither does linux's procfs.
          const bp = this.bridgePath(rawX);
          if (bp === null) { ret = -BigInt(EOPNOTSUPP); break; }
          target = bp;
        }

        // the name comes second everywhere except the list calls, which have none.
        let xname = '';
        if (!isList) {
          if (this.ptrFaults(bytes, Number(a1), 1)) { ret = -BigInt(14); break; }
          xname = readGuestString(bytes, Number(a1));
        }
        const creds = this.getCreds();

        if (nr <= 7) { // setxattr / lsetxattr / fsetxattr
          const xsize = Number(a3);
          if (xsize < 0) { ret = -BigInt(EINVAL); break; }
          if (xsize > 0 && this.ptrFaults(bytes, Number(a2), xsize)) { ret = -BigInt(14); break; }
          const val = xsize > 0
            ? bytes.subarray(Number(a2) + GUEST_MEM_BASE, Number(a2) + GUEST_MEM_BASE + xsize)
            : new Uint8Array(0);
          return {
            status: 'bridge', method: 'fs_setxattr',
            args: { path: target, name: xname, value: bytesToBase64(val),
                    flags: Number(BigInt.asIntN(32, a4)), nofollow, creds },
            apply: (resp: any) => cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))),
          };
        }

        if (nr >= 14) { // removexattr / lremovexattr / fremovexattr
          return {
            status: 'bridge', method: 'fs_removexattr',
            args: { path: target, name: xname, nofollow, creds },
            apply: (resp: any) => cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))),
          };
        }

        // getxattr and listxattr differ only in what they name and where the buffer sits: a size of
        // zero asks how big the answer is without writing it, and a buffer too small is ERANGE.
        const bufPtr = Number(isList ? a1 : a2);
        const bufLen = Number(isList ? a2 : a3);
        if (bufLen < 0) { ret = -BigInt(EINVAL); break; }
        if (bufLen > 0 && this.ptrFaults(bytes, bufPtr, bufLen)) { ret = -BigInt(14); break; }
        return {
          status: 'bridge', method: isList ? 'fs_listxattr' : 'fs_getxattr',
          args: isList ? { path: target, nofollow, creds }
                       : { path: target, name: xname, nofollow, creds },
          apply: (resp: any) => {
            if (resp.ret < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); return; }
            const out = resp.data ? Uint8Array.from(atob(resp.data), (c: string) => c.charCodeAt(0)) : new Uint8Array(0);
            if (bufLen === 0) { cpu.setReg(0, BigInt(out.length)); return; }
            if (out.length > bufLen) { cpu.setReg(0, BigInt.asUintN(64, BigInt(-34))); return; } // ERANGE
            bytes.set(out, bufPtr + GUEST_MEM_BASE);
            cpu.setReg(0, BigInt(out.length));
          },
        };
      }

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
      // i/o scheduling class and priority (fanotify is 262/263). handled by the worker, which can
      // reach the shared process table these name a pid in; they only arrive here on the host-side
      // test path, where there is no table.
      case 30: // ioprio_set
      case 31: // ioprio_get
        ret = -BigInt(ENOSYS);
        break;

      case 42: // nfsservctl (removed in Linux 3.1)
        ret = -BigInt(ENOSYS);
        break;

      case 51: { // chroot(path)
        // 51 is chroot on aarch64 (fchdir is 50, which we handle separately). this used to answer
        // ENOSYS under a comment claiming 51 did not exist.
        if (this.ptrFaults(bytes, Number(a0))) { ret = -BigInt(EFAULT); break; }
        const crPath = readGuestString(bytes, Number(a0));
        if (crPath.length === 0) { ret = -BigInt(ENOENT); break; }
        if (crPath.length > 4095) { ret = -BigInt(ENAMETOOLONG); break; }
        const crBridge = this.bridgePath(crPath);
        if (!crBridge) { ret = -BigInt(EPERM); break; } // /proc and /dev are not real directories
        return {
          status: 'bridge', method: 'fs_stat', args: { path: crBridge },
          apply: (resp: any) => {
            if (resp.ret < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); return; }
            const mode = resp._meta ? resp._meta[0] : resp.mode;
            if (!mode || (mode & 0o170000) !== 0o40000) {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(-ENOTDIR))); return;
            }
            // entering the new root needs search permission on it, as chdir does
            const rUid = resp._meta ? resp._meta[4] : 0;
            const rGid = resp._meta ? resp._meta[5] : 0;
            if (!this.mayExec(mode, rUid, rGid)) {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(-EACCES))); return;
            }
            // the capability is checked LAST, after the path has been resolved and its search
            // permission tested: a caller without it still learns that the directory was closed
            // to it, which is the order the kernel answers in.
            if (!this.hasCap(CAP_SYS_CHROOT)) {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(-EPERM))); return;
            }
            // the resolved path, so a chroot through a symlink records the directory itself
            let newRoot = crBridge;
            if (resp.data) { try { const d = atob(resp.data); if (d.length > 0) newRoot = d; } catch {} }
            this.rootDir = newRoot === '/' ? '/' : newRoot.replace(/\/+$/, '');
            cpu.setReg(0, 0n);
          },
        };
      }

      case 58: // ??? (openat2 is 437 on aarch64, 58 is unused)
        ret = -BigInt(ENOSYS);
        break;

      case 69:        // preadv(fd, iov, iovcnt, offset)
      case 286: {     // preadv2(fd, iov, iovcnt, offset, flags)
        const prvFd = Number(a0), prvIov = Number(a1), prvCnt = Number(a2);
        let prvOff = BigInt.asIntN(64, a3);
        // preadv2 with an offset of -1 means "use the current file position", which is plain readv.
        const prvCurrent = nr === 286 && prvOff === -1n;
        if (prvCurrent) prvOff = 0n;
        const prvErr = this.positionalIoCheck(prvFd, prvCurrent ? 0n : prvOff, false);
        if (prvErr !== 0) { ret = BigInt(prvErr); break; }
        // preadv2 carries per-call flags in its sixth argument. a bit outside the set linux knows
        // is EOPNOTSUPP -- accepting it silently would promise behaviour (HIPRI, NOWAIT) we never
        // delivered. the flags we do accept describe scheduling, not results, so honouring them by
        // doing the ordinary read is correct.
        if (nr === 286 && (Number(BigInt.asIntN(32, a5)) & ~RWF_SUPPORTED) !== 0) {
          ret = -BigInt(EOPNOTSUPP); break;
        }
        const prvLen = this.iovTotal(view, prvIov, prvCnt);
        if (prvLen < 0) { ret = BigInt(prvLen); break; }
        if (prvLen === 0) { ret = 0n; break; }
        let prvFault = false;
        for (let i = 0; i < prvCnt; i++) {
          const len = Number(view.getBigUint64(prvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
          if (len === 0) continue;
          const base = Number(view.getBigUint64(prvIov + i * 16 + GUEST_MEM_BASE, true));
          if (this.ptrWriteFaults(bytes, base, len)) { prvFault = true; break; }
        }
        if (prvFault) { ret = -BigInt(EFAULT); break; }
        const prvScatter = (d: Uint8Array) => {
          let off = 0;
          for (let i = 0; i < prvCnt && off < d.length; i++) {
            const len = Number(view.getBigUint64(prvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
            if (len === 0) continue;
            const base = Number(view.getBigUint64(prvIov + i * 16 + GUEST_MEM_BASE, true));
            const n = Math.min(len, d.length - off);
            bytes.set(d.subarray(off, off + n), base + GUEST_MEM_BASE);
            off += n;
          }
        };
        const prvWant = Math.min(prvLen, BRIDGE_READ_MAX);
        const prvEntry = this.fdTable.get(prvFd);
        if (prvEntry?.type === 'native') {
          const prvAt = prvCurrent ? -1 : Number(prvOff);
          return {
            status: 'bridge',
            method: prvCurrent ? 'fs_read' : 'fs_pread',
            args: prvCurrent
              ? { fd: prvEntry.nativeFd, count: prvWant }
              : { fd: prvEntry.nativeFd, offset: prvAt, count: prvWant },
            apply: (resp: any) => {
              if (resp.ret < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); return; }
              const d = resp.bytes ?? new Uint8Array(0);
              prvScatter(d);
              cpu.setReg(0, BigInt(d.length));
            },
          };
        }
        if (this.vfs?.hasFd(prvFd)) {
          const prvSaved = this.vfs.lseek(prvFd, 0n, 1);
          if (prvSaved < 0n) { ret = prvSaved; break; }
          if (!prvCurrent) this.vfs.lseek(prvFd, prvOff, 0);
          let prvTotal = 0;
          for (let i = 0; i < prvCnt; i++) {
            const base = Number(view.getBigUint64(prvIov + i * 16 + GUEST_MEM_BASE, true));
            const len = Number(view.getBigUint64(prvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
            if (len === 0) continue;
            const n = this.vfs.read(prvFd, base, len, view);
            if (n < 0) { if (prvTotal === 0) prvTotal = n; break; }
            prvTotal += n;
            if (n < len) break; // short read ends the gather
          }
          if (!prvCurrent) this.vfs.lseek(prvFd, prvSaved, 0);
          ret = BigInt(prvTotal);
          break;
        }
        ret = -BigInt(EBADF);
        break;
      }

      case 70:        // pwritev(fd, iov, iovcnt, offset)
      case 287: {     // pwritev2(fd, iov, iovcnt, offset, flags)
        const pwvFd = Number(a0), pwvIov = Number(a1), pwvCnt = Number(a2);
        let pwvOff = BigInt.asIntN(64, a3);
        const pwvCurrent = nr === 287 && pwvOff === -1n;
        if (pwvCurrent) pwvOff = 0n;
        const pwvErr = this.positionalIoCheck(pwvFd, pwvCurrent ? 0n : pwvOff, true);
        if (pwvErr !== 0) { ret = BigInt(pwvErr); break; }
        if (nr === 287 && (Number(BigInt.asIntN(32, a5)) & ~RWF_SUPPORTED) !== 0) {
          ret = -BigInt(EOPNOTSUPP); break;
        }
        const pwvLen = this.iovTotal(view, pwvIov, pwvCnt);
        if (pwvLen < 0) { ret = BigInt(pwvLen); break; }
        if (pwvLen === 0) { ret = 0n; break; }
        const pwvEntry = this.fdTable.get(pwvFd);
        // F_SEAL_WRITE refuses a scattered positional write as it refuses every other kind.
        if (this.sealsOf(pwvEntry) & 0x8) { ret = -BigInt(EPERM); break; }
        if (pwvEntry?.type === 'native') {
          // gather every iovec into one buffer: a positional write is not allowed to interleave
          // with anything else, so it has to reach the file as a single operation.
          const combined = new Uint8Array(pwvLen);
          let off = 0;
          let faulted = false;
          for (let i = 0; i < pwvCnt; i++) {
            const base = Number(view.getBigUint64(pwvIov + i * 16 + GUEST_MEM_BASE, true));
            const len = Number(view.getBigUint64(pwvIov + i * 16 + 8 + GUEST_MEM_BASE, true));
            if (len === 0) continue;
            if (this.ptrFaults(bytes, base, len)) { faulted = true; break; }
            combined.set(bytes.subarray(base + GUEST_MEM_BASE, base + GUEST_MEM_BASE + len), off);
            off += len;
          }
          if (faulted) { ret = -BigInt(EFAULT); break; }
          return {
            status: 'bridge',
            method: pwvCurrent ? 'fs_write' : 'fs_pwrite',
            args: pwvCurrent
              ? { fd: pwvEntry.nativeFd, data: bytesToBase64(combined) }
              : { fd: pwvEntry.nativeFd, offset: Number(pwvOff), data: bytesToBase64(combined) },
            apply: (resp: any) => {
              if (resp.ret >= 0) this.notifyFd(pwvFd, IN_MODIFY);
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            },
          };
        }
        if (this.vfs?.hasFd(pwvFd)) {
          const pwvSaved = this.vfs.lseek(pwvFd, 0n, 1);
          if (pwvSaved < 0n) { ret = pwvSaved; break; }
          if (!pwvCurrent) this.vfs.lseek(pwvFd, pwvOff, 0);
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
          if (!pwvCurrent) this.vfs.lseek(pwvFd, pwvSaved, 0);
          ret = BigInt(pwvTotal);
          break;
        }
        ret = -BigInt(EBADF);
        break;
      }

      // timerfd (verify correct numbers)
      // Per aarch64 spec: timerfd_create=85, timerfd_settime=86, timerfd_gettime=87
      // 82-84 are unused on aarch64? Let me check — they might be signalfd/timerfd/eventfd
      // Actually: 82=signalfd (old), 83=timerfd_create (old), 84=eventfd (old)
      // These are the OLD non-flagged versions; aarch64 should use 74/85/19 instead
      // fsync/fdatasync. these were numbered as if this were x86 -- as old signalfd/timerfd/eventfd
      // -- and answered ENOSYS, so the one call a program makes to be SURE its data is on disk was
      // the one call that failed. apk builds every file it installs through an output stream whose
      // commit step is "fsync, then rename into place"; the fsync failed, so it treated the whole
      // stream as failed and left an empty file behind. a durability primitive that reports failure
      // is worse than a slow one: everything that cares stops, and everything that does not care
      // never asked.
      case 82:   // fsync
      case 83: { // fdatasync
        const syncEntry = this.getFdEntry(Number(a0));
        if (!syncEntry) {
          // a descriptor the synthesised filesystem owns. /dev/null and its neighbours are character
          // devices with nothing to write back, and linux answers EINVAL for those rather than
          // pretending. this branch used to answer 0 for every one of them.
          if (!this.vfs?.hasFd(Number(a0))) { ret = -BigInt(EBADF); break; }
          ret = this.vfs?.isCharDev?.(Number(a0)) ? -BigInt(EINVAL) : 0n;
          break;
        }
        // a fifo, a socket or a character device opened by NAME is still not a file with data to
        // write back: linux answers EINVAL for those, and only the type the open reported tells
        // them apart from the regular file next to them in the same directory.
        const syncType = syncEntry.type === 'native' ? (syncEntry.ftype ?? 0) : 0;
        if (syncType === 0o010000 || syncType === 0o140000 || syncType === 0o020000) {
          ret = -BigInt(EINVAL);
          break;
        }
        if (syncEntry.type === 'native') {
          return {
            status: 'bridge', method: 'fs_fsync',
            args: { fd: syncEntry.nativeFd, data: nr === 83 },
            apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); },
          };
        }
        // anything not backed by a real file has nothing to write back, and linux answers EINVAL
        // for a pipe, a socket or a character device rather than pretending it flushed something.
        const syncNoData = syncEntry.type === 'pipe_read' || syncEntry.type === 'pipe_write' ||
                           syncEntry.type === 'socket' || syncEntry.type === 'unixsock' ||
                           (syncEntry.type === 'vfs' && !!this.vfs?.isCharDev?.(syncEntry.fd));
        ret = syncNoData ? -BigInt(EINVAL) : 0n;
        break;
      }

      // setitimer/getitimer duplicates check
      case 104: // ??? (on aarch64, 104 = clock_adjtime. We have setitimer at 103)
        ret = -BigInt(EPERM); // clock_adjtime — not permitted
        break;
      case 105: // ??? (105 = clock_adjtime_time64 — doesn't exist on aarch64)
      case 106: // ??? (106 = mq_timedsend)
        ret = -BigInt(ENOSYS);
        break;

      // Clock
        ret = -BigInt(EPERM);
        break;

      // Signal (remaining)
      // 137 rt_sigtimedwait is handled in the worker (it has to BLOCK, and only the worker can park
      // on the proc-table futex). it reaches here only in the host-side dispatch used by tests.
      case 137:
        ret = -BigInt(4); // EINTR
        break;
      case 138: // rt_sigqueueinfo
        ret = 0n; // accept
        break;

      // ioctl subcommand handling is in case 29 already
      case 142: { // reboot(magic1, magic2, cmd, arg)
        // NOT sched_setattr, which is 274. this used to answer 0 to everything, which told a
        // caller the machine had accepted a reboot it never performed.
        //
        // the capability check comes FIRST, before the magic numbers: linux refuses an
        // unprivileged caller without telling it whether its arguments were even well formed.
        if (!this.hasCap(CAP_SYS_BOOT)) { ret = -BigInt(EPERM); break; }
        const rbMagic1 = Number(BigInt.asIntN(32, a0)) >>> 0;
        const rbMagic2 = Number(BigInt.asIntN(32, a1)) >>> 0;
        if (rbMagic1 !== 0xfee1dead || !REBOOT_MAGIC2.has(rbMagic2)) { ret = -BigInt(EINVAL); break; }
        const rbCmd = Number(BigInt.asIntN(32, a2)) >>> 0;
        switch (rbCmd) {
          // CAD_ON/CAD_OFF only choose what ctrl-alt-del does; they reboot nothing.
          case 0x89abcdef: this.cadEnabled = true; ret = 0n; break;
          case 0x00000000: this.cadEnabled = false; ret = 0n; break;
          // the ones that actually stop the machine. our machine is the guest system, and stopping
          // it is what `halt` and `poweroff` are for -- so they do exactly that rather than
          // pretending. reboot(2) does not return when it succeeds.
          case 0x01234567: // RESTART
          case 0xcdef0123: // HALT
          case 0x4321fedc: // POWER_OFF
          case 0xa1b2c3d4: // RESTART2
            this.haltHook?.(rbCmd === 0x01234567 || rbCmd === 0xa1b2c3d4);
            return { status: 'exit', code: 0 };
          case 0xd000fce2: // SW_SUSPEND: there is no hibernation image to write
          case 0x45584543: // KEXEC: no second kernel to jump to
            ret = -BigInt(EINVAL);
            break;
          default: ret = -BigInt(EINVAL);
        }
        break;
      }

      // getrlimit/setrlimit (prlimit64 at 261 is the modern replacement)
      case 161:   // sethostname(name, len)
      case 162: { // setdomainname(name, len)
        // these were labelled chroot and sync, which are 51 and 81 -- so setting the hostname did
        // nothing and uname kept answering with a constant.
        const utsLen = Number(BigInt.asIntN(64, a1));
        if (utsLen < 0 || utsLen > 64) { ret = -BigInt(EINVAL); break; }
        if (this.ptrFaults(bytes, Number(a0), Math.max(utsLen, 1))) { ret = -BigInt(EFAULT); break; }
        // changing the machine's name is the administrator's to do.
        if (!this.hasCap(CAP_SYS_ADMIN)) { ret = -BigInt(EPERM); break; }
        const base = Number(a0) + GUEST_MEM_BASE;
        const name = new TextDecoder().decode(bytes.subarray(base, base + utsLen));
        this.utsHook?.set(nr === 161 ? 'host' : 'domain', name);
        ret = 0n;
        break;
      }
      // getrlimit/setrlimit are the older two-argument spellings of prlimit64, with the same
      // struct rlimit { u64 cur; u64 max; } on a 64-bit target. they used to return success
      // without doing anything: setrlimit silently ignored the new limit, so a process that
      // lowered its own ceiling was never held to it, and getrlimit left the caller's struct
      // untouched while claiming to have filled it.
      case 163: { // getrlimit(resource, rlim)
        const res = Number(BigInt.asIntN(32, a0));
        if (res < 0 || res >= RLIM_NLIMITS) { ret = -BigInt(EINVAL); break; }
        const ptr = Number(a1);
        if (ptr === 0 || this.ptrFaults(bytes, ptr, 16)) { ret = -BigInt(EFAULT); break; }
        const lim = this.rlimitOf(res);
        view.setBigUint64(ptr + GUEST_MEM_BASE, lim.cur, true);
        view.setBigUint64(ptr + GUEST_MEM_BASE + 8, lim.max, true);
        ret = 0n;
        break;
      }
      case 164: { // setrlimit(resource, rlim)
        const res = Number(BigInt.asIntN(32, a0));
        if (res < 0 || res >= RLIM_NLIMITS) { ret = -BigInt(EINVAL); break; }
        const ptr = Number(a1);
        if (ptr === 0 || this.ptrFaults(bytes, ptr, 16)) { ret = -BigInt(EFAULT); break; }
        const cur = view.getBigUint64(ptr + GUEST_MEM_BASE, true);
        const max = view.getBigUint64(ptr + GUEST_MEM_BASE + 8, true);
        if (cur > max) { ret = -BigInt(EINVAL); break; }
        // the descriptor ceiling is a system-wide number (fs.nr_open), and no process may set its
        // own hard limit above it -- not even root, because the table it indexes is sized by that
        // number. linux answers EPERM.
        if (res === RLIMIT_NOFILE && max > BigInt(NR_OPEN)) { ret = -BigInt(EPERM); break; }
        // raising the HARD limit needs CAP_SYS_RESOURCE; lowering it is one-way and always allowed.
        if (max > this.rlimitOf(res).max && !this.hasCap(CAP_SYS_RESOURCE)) { ret = -BigInt(EPERM); break; }
        this.setRlimit(res, cur, max);
        ret = 0n;
        break;
      }

      case 171:   // adjtimex(buf)
      case 266: { // clock_adjtime(clk_id, buf) - the same call, with a clock named up front
        const axBuf = nr === 171 ? Number(a0) : Number(a1);
        if (nr === 266 && !this.knownClock(Number(BigInt.asIntN(32, a0)))) { ret = -BigInt(EINVAL); break; }
        // struct timex is 208 bytes on aarch64 (16 longs, 6 ints, a timeval, and padding).
        if (this.ptrFaults(bytes, axBuf, 208)) { ret = -BigInt(EFAULT); break; }
        const axBase = axBuf + GUEST_MEM_BASE;
        const modes = view.getInt32(axBase, true);
        // modes == 0 is a pure READ of the kernel's timekeeping state, and it is unprivileged.
        // answering EPERM to it, as this used to, made even `adjtimex --print` fail.
        // the mode word was never checked, so every nonsense combination "succeeded". these are
        // linux's rules, in its order.
        const ADJ_OFFSET = 0x0001, ADJ_FREQUENCY = 0x0002, ADJ_SETOFFSET = 0x0100;
        const ADJ_NANO = 0x2000, ADJ_TICK = 0x4000, ADJ_ADJTIME = 0x8000;
        if (modes & ADJ_ADJTIME) {
          // ADJ_ADJTIME is the old adjtime(2) wearing a mode bit, and it names an OFFSET. on its
          // own it asks for nothing, which is what CVE-2018-11508 read uninitialised memory for.
          if (!(modes & ADJ_OFFSET)) { ret = -BigInt(EINVAL); break; }
          if (!this.hasCap(CAP_SYS_TIME)) { ret = -BigInt(EPERM); break; }
        } else if (modes !== 0) {
          if (!this.hasCap(CAP_SYS_TIME)) { ret = -BigInt(EPERM); break; }
          // a tick more than 10% away from nominal means the caller has the units wrong.
          if (modes & ADJ_TICK) {
            const tick = view.getBigInt64(axBase + 88, true);
            if (tick < 9000n || tick > 11000n) { ret = -BigInt(EINVAL); break; }
          }
        }
        if (modes & ADJ_SETOFFSET) {
          // ADJ_SETOFFSET shifts the clock by struct timex's `time`, which is a timeval unless
          // ADJ_NANO says it is a timespec.
          const secOff = view.getBigInt64(axBase + 72, true);
          const fracOff = view.getBigInt64(axBase + 80, true);
          const fracMax = (modes & ADJ_NANO) ? 1_000_000_000n : 1_000_000n;
          if (fracOff < 0n || fracOff >= fracMax) { ret = -BigInt(EINVAL); break; }
          this.clockOffsetNs = this.clockOffsetNs + secOff * 1_000_000_000n +
                               fracOff * ((modes & ADJ_NANO) ? 1n : 1000n);
        }
        if (modes & ADJ_FREQUENCY) {
          // freq is in parts-per-million scaled by 2^16, and the kernel refuses a value whose
          // scaling would overflow rather than wrapping it.
          const freq = view.getBigInt64(axBase + 16, true);
          if (freq < -32768000n * 65536n || freq > 32768000n * 65536n) { ret = -BigInt(EINVAL); break; }
        }
        // report the state a caller reads back: no offset under discipline, a nominal tick, and
        // the current time in the struct's timeval.
        const nowNs = this.clockNs(0);
        view.setBigInt64(axBase + 8, 0n, true);        // offset
        view.setBigInt64(axBase + 16, 0n, true);       // freq
        view.setBigInt64(axBase + 24, 500000n, true);  // maxerror
        view.setBigInt64(axBase + 32, 0n, true);       // esterror
        view.setInt32(axBase + 40, 0x0040, true);      // status: STA_UNSYNC
        view.setBigInt64(axBase + 48, 0n, true);       // constant
        view.setBigInt64(axBase + 56, 0n, true);       // precision
        view.setBigInt64(axBase + 64, 500000n, true);  // tolerance
        view.setBigInt64(axBase + 72, nowNs / 1_000_000_000n, true);            // time.tv_sec
        view.setBigInt64(axBase + 80, (nowNs % 1_000_000_000n) / 1000n, true);  // time.tv_usec
        view.setBigInt64(axBase + 88, 10000n, true);   // tick (microseconds per tick at HZ=100)
        ret = 5n; // TIME_ERROR: the clock is not disciplined by anything
        break;
      }

      // posix message queues (mq_open, mq_unlink, mq_timedsend, mq_timedreceive, mq_notify,
      // mq_getsetattr) are handled by the worker: the queues are named system-wide, so they need
      // the kernel round trip only the worker can make. reaching here means there is no kernel.
      case 180: case 181: case 182: case 183: case 184: case 185:
        ret = -BigInt(ENOSYS);
        break;

      // SysV IPC
      // 186-197 (msg*, sem*, shm*) are all handled in the worker: every one of these registries is
      // system-wide, so they need a round trip to the kernel that only the worker can make. these
      // cases are unreachable from a real guest and exist so a stray number still gets an answer.
      case 186: case 187: case 188: case 189: // msgget, msgctl, msgrcv, msgsnd
      case 190: case 191: case 192: case 193: // semget, semctl, semtimedop, semop
      case 194: case 195: case 196: case 197: // shmget, shmctl, shmat, shmdt
        ret = -BigInt(ENOSYS);
        break;

      // getdents64 — we have this at case 61 which is correct on aarch64
      // 217 = getdents (old 32-bit version, not used on aarch64)
      case 217:
        ret = -BigInt(ENOSYS);
        break;

      // Miscellaneous
      // 218 and 219 are request_key and keyctl, not the NUMA policy calls they were labelled as
      // (those are 235..237). answering 0 meant a program asking the kernel keyring for a key was
      // told it had succeeded and then read back nothing.
      case 218: // request_key
      case 219: // keyctl
        ret = -BigInt(ENOSYS);
        break;
      case 223: { // fadvise64(fd, offset, len, advice)
        // the advice itself can always be ignored -- it is a hint about future access and we keep
        // no page cache of our own to act on. what cannot be ignored is whether the CALL made
        // sense: a descriptor that is not open, a stream with no position to advise about, or an
        // advice number that names nothing are all errors, and programs check for them.
        const faFd = Number(a0);
        const faAdvice = Number(BigInt.asIntN(32, a3));
        if (!this.fdExists(faFd)) { ret = -BigInt(EBADF); break; }
        const faEntry = this.fdTable.get(faFd);
        if (this.io.pipeReady?.(faFd) || faEntry?.type === 'pipe_read' || faEntry?.type === 'pipe_write' ||
            faEntry?.type === 'socket' || faEntry?.type === 'unixsock' || faEntry?.type === 'tty' ||
            faEntry?.type === 'stdio' || faEntry?.type === 'eventfd' || faEntry?.type === 'epoll' ||
            faEntry?.type === 'inotify' || faEntry?.type === 'timerfd' || faEntry?.type === 'jitterctl') {
          ret = -BigInt(ESPIPE); break;
        }
        // POSIX_FADV_NORMAL/RANDOM/SEQUENTIAL/WILLNEED/DONTNEED/NOREUSE, in the numbering every
        // architecture but 31-bit s390 uses.
        if (faAdvice < 0 || faAdvice > 5) { ret = -BigInt(EINVAL); break; }
        ret = 0n;
        break;
      }
      case 224: // swapon
      case 225: // swapoff
        ret = -BigInt(EPERM); // needs CAP_SYS_ADMIN, and there is no swap device to name anyway
        break;
      case 234: // remap_file_pages (removed from linux in all but name)
        ret = -BigInt(ENOSYS);
        break;
      // 235..239 are the NUMA calls: mbind, get_mempolicy, set_mempolicy, migrate_pages,
      // move_pages. they were labelled membarrier/mlock2/copy_file_range/preadv2/pwritev2, which
      // are 283..287 -- and 236 in particular did not merely answer the wrong question, it CALLED
      // setLocked() and mlocked a range of the caller's address space on a get_mempolicy.
      case 235: // mbind - a placement policy for a range. one node, so nothing to place.
      case 237: // set_mempolicy
        ret = 0n;
        break;
      case 236: { // get_mempolicy(mode, nodemask, maxnode, addr, flags)
        // one node, always node 0: report MPOL_DEFAULT and a mask with bit 0 set.
        if (Number(a0) !== 0) {
          if (this.ptrFaults(bytes, Number(a0), 4)) { ret = -BigInt(EFAULT); break; }
          view.setInt32(Number(a0) + GUEST_MEM_BASE, 0, true); // MPOL_DEFAULT
        }
        if (Number(a1) !== 0 && Number(a2) > 0) {
          if (this.ptrFaults(bytes, Number(a1), 8)) { ret = -BigInt(EFAULT); break; }
          view.setBigUint64(Number(a1) + GUEST_MEM_BASE, 1n, true);
        }
        ret = 0n;
        break;
      }
      case 238: // migrate_pages
      case 239: // move_pages
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
      case 264: { // name_to_handle_at(dirfd, path, handle, mount_id, flags)
        // struct file_handle { __u32 handle_bytes; int handle_type; unsigned char f_handle[]; }
        //
        // the handle's contents are the filesystem's own business -- linux hands back an opaque
        // blob no caller may interpret -- so ours is the inode number, which is what identifies a
        // file here. the kernel remembers which path each one named, and open_by_handle_at asks it.
        // the checks happen in the kernel's order, and the order is the point of the test: the
        // NAME is looked up before the handle is measured, so a bad descriptor is EBADF even when
        // the handle would not have fitted anyway.
        const nhFlags = Number(BigInt.asIntN(32, a4));
        // name_to_handle_at does NOT follow a final symlink unless asked -- the flag that asks is
        // AT_SYMLINK_FOLLOW, not the NOFOLLOW every other *at() call takes.
        const AT_EMPTY_PATH_NH = 0x1000, AT_SYMLINK_FOLLOW_NH = 0x400, AT_HANDLE_FID = 0x200,
              AT_HANDLE_MNT_ID_UNIQUE = 0x1;
        if (nhFlags & ~(AT_EMPTY_PATH_NH | AT_SYMLINK_FOLLOW_NH | AT_HANDLE_FID | AT_HANDLE_MNT_ID_UNIQUE)) {
          ret = -BigInt(EINVAL); break;
        }
        const nhHandle = Number(a2);
        if (nhHandle === 0 || this.ptrWriteFaults(bytes, nhHandle, 8)) { ret = -BigInt(EFAULT); break; }
        const nhRoom = view.getUint32(nhHandle + GUEST_MEM_BASE, true);
        const NH_SIZE = 8;        // an inode number
        const MAX_HANDLE_SZ = 128;
        if (nhRoom > MAX_HANDLE_SZ) { ret = -BigInt(EINVAL); break; }
        if (Number(a3) !== 0 && this.ptrWriteFaults(bytes, Number(a3), 4)) { ret = -BigInt(EFAULT); break; }
        if (this.ptrFaults(bytes, Number(a1))) { ret = -BigInt(EFAULT); break; }
        const nhRaw = readGuestString(bytes, Number(a1));
        if (this.isEmptyPath(nhRaw) && !(nhFlags & AT_EMPTY_PATH_NH)) { ret = -BigInt(ENOENT); break; }
        if (!this.isEmptyPath(nhRaw)) { const pe = this.pathCheck(nhRaw); if (pe) { ret = -BigInt(pe); break; } }
        // with AT_EMPTY_PATH the descriptor names the FILE, so "not a directory" is not an error --
        // it is the whole point of the call.
        if (!this.isEmptyPath(nhRaw)) {
          const de = this.atDirfdError(a0, nhRaw); if (de) { ret = -BigInt(de); break; }
        }
        // an empty name with AT_FDCWD names the working directory, not nothing.
        const nhDfd = Number(BigInt.asIntN(32, a0));
        const nhPath = !this.isEmptyPath(nhRaw) ? this.resolveAt(a0, nhRaw)
                     : (nhDfd === -100 ? (this.vfs?.getcwd() ?? this.cwd) : this.pathOfFd(nhDfd));
        if (nhPath === null) { ret = -BigInt(EBADF); break; }
        const nhBridge = this.bridgePath(nhPath);
        if (nhBridge === null) { ret = -BigInt(EOPNOTSUPP); break; } // /proc and /dev have no handles
        return {
          status: 'bridge', method: 'fs_stat',
          args: { path: nhBridge, nofollow: (nhFlags & AT_SYMLINK_FOLLOW_NH) === 0 },
          apply: (resp: any) => {
            if (resp.ret < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); return; }
            // the name exists; only now does the handle's size matter. the caller is told how much
            // room it needs and asks again.
            if (nhRoom < NH_SIZE) {
              view.setUint32(nhHandle + GUEST_MEM_BASE, NH_SIZE, true);
              cpu.setReg(0, BigInt.asUintN(64, BigInt(-75)));   // EOVERFLOW
              return;
            }
            const ino = resp._meta ? Math.round(resp._meta[6]) : 0;
            view.setUint32(nhHandle + GUEST_MEM_BASE, NH_SIZE, true);
            view.setInt32(nhHandle + 4 + GUEST_MEM_BASE, 1, true);            // handle_type
            view.setBigUint64(nhHandle + 8 + GUEST_MEM_BASE, BigInt(ino), true);
            if (Number(a3) !== 0) view.setInt32(Number(a3) + GUEST_MEM_BASE, NATIVE_FS_MOUNT_ID, true);
            // remember what this handle names, so it can be opened again.
            this.handleHook?.put(ino, nhBridge);
            cpu.setReg(0, 0n);
          },
        };
      }

      case 265: { // open_by_handle_at(mount_fd, handle, flags)
        // the first argument is a descriptor for the FILESYSTEM the handle came from: it has to be
        // open, and it has to be on the filesystem that issued the handle. this accepted anything.
        const ohMountFd = Number(BigInt.asIntN(32, a0));
        if (ohMountFd !== -100 && !this.fdIsOpen(ohMountFd)) { ret = -BigInt(EBADF); break; }
        const ohHandle = Number(a1);
        if (ohHandle === 0 || this.ptrFaults(bytes, ohHandle, 8)) { ret = -BigInt(EFAULT); break; }
        const ohBytes = view.getUint32(ohHandle + GUEST_MEM_BASE, true);
        const ohType = view.getInt32(ohHandle + 4 + GUEST_MEM_BASE, true);
        if (ohBytes !== 8 || ohType !== 1) { ret = -BigInt(22); break; }        // EINVAL: not ours
        if (this.ptrFaults(bytes, ohHandle, 16)) { ret = -BigInt(EFAULT); break; }
        // opening by handle is a privileged operation: it reaches a file by identity rather than
        // by name, so it walks past every directory permission on the way there.
        if (!this.hasCap(CAP_DAC_READ_SEARCH)) { ret = -BigInt(EPERM); break; }
        // a mount descriptor that is not on this filesystem cannot resolve the handle: linux
        // answers ESTALE, which is how a caller learns it asked the wrong filesystem.
        if (ohMountFd !== -100) {
          const mEnt = this.getFdEntry(ohMountFd);
          if (mEnt?.type !== 'native') { ret = -BigInt(116); break; }          // ESTALE
        }
        const ohIno = Number(view.getBigUint64(ohHandle + 8 + GUEST_MEM_BASE, true));
        const ohPath = this.handleHook?.get(ohIno) ?? null;
        if (ohPath === null) { ret = -BigInt(116); break; }                    // ESTALE
        const ohFd = this.allocFd();
        const ohFlags = Number(BigInt.asIntN(32, a2));
        return {
          status: 'bridge', method: 'fs_open',
          // a handle names an inode, and a symlink's inode is the LINK: there is no following it
          // from here, so opening one is ELOOP -- which is exactly what O_NOFOLLOW answers.
          args: { path: ohPath, flags: ohFlags | 0o100000, mode: 0 },
          apply: (resp: any) => {
            if (resp.ret < 0) { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); return; }
            this.fdTable.set(ohFd, { type: 'native', nativeFd: resp.ret, path: ohPath,
                                     ofdId: this.newOfdIdPublic() });
            this.nativeRefs.set(resp.ret, 1);
            cpu.setReg(0, BigInt(ohFd));
          },
        };
      }
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
      case 274: case 275: // sched_setattr / sched_getattr — see the sched_* note above
        ret = -BigInt(ENOSYS);
        break;
      case 280: // bpf
        ret = -BigInt(ENOSYS);
        break;
      case 282: // userfaultfd
        ret = -BigInt(ENOSYS);
        break;
      case 283: { // membarrier(cmd, flags, cpu_id)
        // every thread of a process here is a real thread over one shared memory, and the engine
        // already emits the atomics the guest asked for -- so the barrier itself costs nothing.
        // what a caller can still observe is the PROTOCOL: which commands exist, that a private
        // expedited barrier needs registering first, and that anything else is EINVAL.
        // the commands are BITS, not an enumeration: MEMBARRIER_CMD_QUERY answers with the mask of
        // the ones this kernel has, and each command IS its bit.
        const MB_QUERY = 0, MB_GLOBAL = 1, MB_GLOBAL_EXPEDITED = 2, MB_REG_GLOBAL_EXPEDITED = 4,
              MB_PRIVATE_EXPEDITED = 8, MB_REG_PRIVATE_EXPEDITED = 16,
              MB_PRIVATE_EXPEDITED_SYNC_CORE = 32, MB_REG_PRIVATE_EXPEDITED_SYNC_CORE = 64;
        const MB_SUPPORTED = MB_GLOBAL | MB_GLOBAL_EXPEDITED | MB_REG_GLOBAL_EXPEDITED |
                             MB_PRIVATE_EXPEDITED | MB_REG_PRIVATE_EXPEDITED |
                             MB_PRIVATE_EXPEDITED_SYNC_CORE | MB_REG_PRIVATE_EXPEDITED_SYNC_CORE;
        const mbCmd = Number(BigInt.asIntN(32, a0));
        if (Number(BigInt.asIntN(32, a1)) !== 0) { ret = -BigInt(EINVAL); break; }
        switch (mbCmd) {
          case MB_QUERY: ret = BigInt(MB_SUPPORTED); break;
          case MB_GLOBAL: case MB_GLOBAL_EXPEDITED: case MB_REG_GLOBAL_EXPEDITED: ret = 0n; break;
          case MB_REG_PRIVATE_EXPEDITED: this.membarrierRegistered |= 1; ret = 0n; break;
          case MB_REG_PRIVATE_EXPEDITED_SYNC_CORE: this.membarrierRegistered |= 2; ret = 0n; break;
          // an expedited private barrier is only for a process that said it wanted one.
          case MB_PRIVATE_EXPEDITED:
            ret = (this.membarrierRegistered & 1) ? 0n : -BigInt(EPERM);
            break;
          case MB_PRIVATE_EXPEDITED_SYNC_CORE:
            ret = (this.membarrierRegistered & 2) ? 0n : -BigInt(EPERM);
            break;
          default: ret = -BigInt(EINVAL); break;
        }
        break;
      }

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
      case 436: { // close_range(first, last, flags)
        // the arguments were never looked at: a reversed range, an unknown flag and a request to
        // only SET close-on-exec all closed the same descriptors regardless.
        const CLOSE_RANGE_UNSHARE = 2, CLOSE_RANGE_CLOEXEC = 4;
        const crFlags = Number(BigInt.asUintN(32, a2));
        if (crFlags & ~(CLOSE_RANGE_UNSHARE | CLOSE_RANGE_CLOEXEC)) { ret = -BigInt(EINVAL); break; }
        const crFirst = Number(BigInt.asUintN(32, a0));
        const crLast = Number(BigInt.asUintN(32, a1));
        if (crFirst > crLast) { ret = -BigInt(EINVAL); break; }
        const crEnd = Math.min(crLast, MAX_FDS);
        for (let fd = crFirst; fd <= crEnd; fd++) {
          if (!this.fdIsOpen(fd)) continue;
          // CLOSE_RANGE_CLOEXEC marks rather than closes: the descriptors stay usable until the
          // next exec, which is the point of asking for it instead of a plain close.
          if (crFlags & CLOSE_RANGE_CLOEXEC) { this.fdCloexec.add(fd); continue; }
          this.closeOne(fd);
        }
        ret = 0n;
        break;
      }

      case 438: // pidfd_getfd
        ret = -BigInt(ENOSYS);
        break;
      case 439: { // faccessat2(dirfd, pathname, mode, flags)
        // this never reached the bridge, so every real file on disk answered ENOENT -- the whole
        // point of faccessat2 over faccessat is the flags, and it could not see the filesystem.
        const AT_EACCESS = 0x200, AT_EMPTY_PATH = 0x1000;
        const fa2Flags = Number(BigInt.asIntN(32, a3));
        if (fa2Flags & ~(AT_EACCESS | AT_SYMLINK_NOFOLLOW | AT_EMPTY_PATH)) { ret = -BigInt(EINVAL); break; }
        if (Number(BigInt.asUintN(32, a2)) & ~0x7) { ret = -BigInt(EINVAL); break; }
        if (this.ptrFaults(bytes, Number(a1))) { ret = -BigInt(EFAULT); break; }
        const rawFa2 = readGuestString(bytes, Number(a1));
        if (this.isEmptyPath(rawFa2)) { if (!(fa2Flags & AT_EMPTY_PATH)) { ret = -BigInt(ENOENT); break; } }
        else { const peFa2 = this.pathCheck(rawFa2); if (peFa2) { ret = -BigInt(peFa2); break; } }
        const fa2Path = this.resolveAt(a0, rawFa2);
        const bpFa2 = this.bridgePath(fa2Path);
        if (bpFa2 !== null) {
          return {
            status: 'bridge', method: 'fs_faccessat',
            // AT_EACCESS asks the question of the EFFECTIVE ids rather than the real ones, which
            // is the difference between "may this program reach it" and "may the user it runs for".
            args: { path: bpFa2, mode: Number(a2), eaccess: (fa2Flags & AT_EACCESS) !== 0 },
            apply: (resp: any) => { cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret))); },
          };
        }
        if (this.vfs) {
          ret = BigInt(this.vfs.faccessat(Number(BigInt.asIntN(32, a0)), fa2Path, Number(a2), fa2Flags));
        } else {
          ret = -BigInt(ENOENT);
        }
        break;
      }
      case 440: // process_madvise
        ret = -BigInt(ENOSYS);
        break;
      case 441: { // epoll_pwait2 - epoll_pwait with a timespec instead of a millisecond count
        // this used to be a second, worse copy of the wait: it reported every registered
        // descriptor as ready without asking whether it was, and never waited. a caller then
        // issued a blocking read on a descriptor with nothing in it.
        let ep2Timeout = -1;
        if (Number(a3) !== 0) {
          if (this.ptrFaults(bytes, Number(a3), 16)) { ret = -BigInt(EFAULT); break; }
          const tsSec = Number(view.getBigInt64(Number(a3) + GUEST_MEM_BASE, true));
          const tsNsec = Number(view.getBigInt64(Number(a3) + GUEST_MEM_BASE + 8, true));
          if (tsSec < 0 || tsNsec < 0 || tsNsec >= 1_000_000_000) { ret = -BigInt(EINVAL); break; }
          ep2Timeout = tsSec * 1000 + Math.ceil(tsNsec / 1_000_000);
        }
        const ep2Sigmask = Number(a4);
        if (ep2Sigmask !== 0 && this.ptrFaults(bytes, ep2Sigmask, 8)) { ret = -BigInt(EFAULT); break; }
        const ep2Saved = ep2Sigmask !== 0 ? (this.signalMask ?? 0n) : null;
        if (ep2Sigmask !== 0) this.setSignalMask(view.getBigUint64(ep2Sigmask + GUEST_MEM_BASE, true));
        ret = BigInt(this.epollWait(view, bytes, Number(a0), Number(a1),
                                    Number(BigInt.asIntN(32, a2)), ep2Timeout));
        if (ep2Saved !== null) this.setSignalMask(ep2Saved);
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
        // fchmodat with a flags argument that is actually honoured. it used to answer 0 to
        // everything, including a closed descriptor and a flag that does not exist.
        const AT_SYMLINK_NOFOLLOW_452 = 0x100, AT_EMPTY_PATH_452 = 0x1000;
        const flags452 = Number(BigInt.asIntN(32, a3));
        if (flags452 & ~(AT_SYMLINK_NOFOLLOW_452 | AT_EMPTY_PATH_452)) { ret = -BigInt(EINVAL); break; }
        if (this.ptrFaults(bytes, Number(a1), 1)) { ret = -BigInt(EFAULT); break; }
        const raw452 = readGuestString(bytes, Number(a1));
        { const pe = this.pathCheck(raw452); if (pe) { ret = -BigInt(pe); break; } }
        { const de = this.atDirfdError(a0, raw452); if (de) { ret = -BigInt(de); break; } }
        const path452 = this.resolveAt(a0, raw452);
        const mode452 = Number(BigInt.asUintN(32, a2)) & 0o7777;
        const bp452 = this.bridgePath(path452);
        if (bp452 !== null) {
          this.invalidateStat(bp452);
          return {
            status: 'bridge', method: 'fs_chmod',
            args: { path: bp452, mode: mode452, nofollow: (flags452 & AT_SYMLINK_NOFOLLOW_452) ? 1 : 0 },
            apply: (resp: any) => {
              if (resp.ret >= 0) this.notify(bp452, IN_ATTRIB);
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            },
          };
        }
        ret = this.vfs ? BigInt(this.vfs.chmodPath(path452, mode452)) : -BigInt(ENOENT);
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
      case 462: // mseal — memory sealing is not implemented, and saying otherwise would promise
                //         that a later mprotect of the range will be refused when it will not.
        ret = -BigInt(ENOSYS);
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
    // a refusal that libc quietly recovers from still leaves its errno lying around, and that stale
    // value is what eventually gets reported by something unrelated. record each one once.
    this.noteResult(nr, ret);
    cpu.setReg(0, BigInt.asUintN(64, ret));
    return { status: 'ok' };
  }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

// a guest pointer faults (EFAULT) if it lands in the never-mapped low page or falls outside
// the guest-addressable region. this catches the classic LTP bad-address probes ((void*)-1 and
// NULL) without needing full per-page vma tracking, which our flat guest memory does not model.
// which argument register of each path-taking syscall holds a pathname. checked centrally in
// handle() so a wild path pointer is EFAULT before the call does anything, exactly as getname()
// does in linux. AT_EMPTY_PATH is not an exemption: it means the string must be EMPTY, not that
// there need not be one, and linux still faults on a null pointer.
// the *at() flag words, and which bits linux actually defines for each. anything outside the mask
// is EINVAL, checked before the syscall runs -- the same reasoning as PATH_ARG_REGS and
// DIRFD_ARG_REGS. [flags register, allowed bits].
//   AT_SYMLINK_NOFOLLOW 0x100  AT_REMOVEDIR 0x200  AT_EACCESS 0x200  AT_SYMLINK_FOLLOW 0x400
//   AT_NO_AUTOMOUNT 0x800      AT_EMPTY_PATH 0x1000  AT_STATX_SYNC_TYPE 0x6000
// fchmodat (53) and faccessat (48) are absent on purpose: neither takes a flags argument on
// aarch64 -- that is what fchmodat2 and faccessat2 were added for.
const AT_FLAG_ARGS: Record<number, [number, number]> = {
  35: [2, 0x200],            // unlinkat: AT_REMOVEDIR
  37: [4, 0x400 | 0x1000],   // linkat: AT_SYMLINK_FOLLOW | AT_EMPTY_PATH
  54: [4, 0x100 | 0x1000],   // fchownat: AT_SYMLINK_NOFOLLOW | AT_EMPTY_PATH
  79: [3, 0x100 | 0x800 | 0x1000],           // fstatat
  88: [3, 0x100],            // utimensat: AT_SYMLINK_NOFOLLOW
  291: [2, 0x100 | 0x800 | 0x1000 | 0x6000], // statx
  439: [3, 0x100 | 0x200 | 0x1000],          // faccessat2: NOFOLLOW | EACCESS | EMPTY_PATH
};

// which register holds the DIRFD that each relative path in a syscall resolves against.
// pairs of [dirfd register, path register], in the same spirit as PATH_ARG_REGS: linux validates
// the dirfd before the syscall's own logic runs, and a check per case is a check you can forget
// to add to the thirty-first case. an entry here is enough to make a syscall answer EBADF for a
// descriptor that is not open and ENOTDIR for one that is not a directory -- which resolveAt used
// to paper over by quietly falling back to the cwd, so openat(closed_fd, "x") "worked".
const DIRFD_ARG_REGS: Record<number, Array<[number, number]>> = {
  33: [[0, 1]],           // mknodat
  34: [[0, 1]],           // mkdirat
  35: [[0, 1]],           // unlinkat
  36: [[1, 2]],           // symlinkat(target, newdirfd, linkpath)
  37: [[0, 1], [2, 3]],   // linkat
  38: [[0, 1], [2, 3]],   // renameat
  48: [[0, 1]],           // faccessat
  53: [[0, 1]],           // fchmodat
  54: [[0, 1]],           // fchownat
  56: [[0, 1]],           // openat
  78: [[0, 1]],           // readlinkat
  79: [[0, 1]],           // fstatat
  88: [[0, 1]],           // utimensat
  276: [[0, 1], [2, 3]],  // renameat2
  281: [[0, 1]],          // execveat
  291: [[0, 1]],          // statx
  437: [[0, 1]],          // openat2
  439: [[0, 1]],          // faccessat2
};

const PATH_ARG_REGS: Record<number, number[]> = {
  5: [0], 6: [0],            // setxattr, lsetxattr
  8: [0], 9: [0],            // getxattr, lgetxattr
  11: [0], 12: [0],          // listxattr, llistxattr
  14: [0], 15: [0],          // removexattr, lremovexattr
  33: [1],      // mknodat
  34: [1],      // mkdirat
  35: [1],      // unlinkat
  36: [0, 2],   // symlinkat(target, newdirfd, linkpath)
  37: [1, 3],   // linkat
  38: [1, 3],   // renameat
  45: [0],      // truncate
  48: [1],      // faccessat
  49: [0],      // chdir
  53: [1],      // fchmodat
  54: [1],      // fchownat
  56: [1],      // openat
  78: [1],      // readlinkat
  79: [1],      // fstatat
  221: [0],     // execve
  276: [1, 3],  // renameat2
  279: [0],     // memfd_create (a name, not a path, but read the same way)
  281: [1],     // execveat
  291: [1],     // statx
  437: [1],     // openat2
  439: [1],     // faccessat2
};

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

function writeChardevStat(view: DataView, guestAddr: number, rdev = 0): void {
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
  // a device node reports WHICH device it is; the anonymous chardevs (a pipe end, a tty) have none.
  if (rdev !== 0) view.setBigUint64(base + 32, BigInt(rdev), true);
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
  // Meta layout for fs_stat: _meta = [mode, size, nlink, mtime, uid, gid, ino, rdev]
  const meta = resp._meta;
  const mode = meta ? meta[0] : (resp.mode ?? 0o100644);
  const size = meta ? meta[1] : (resp.size ?? 0);
  const nlink = meta ? meta[2] : (resp.nlink ?? 1);
  const mtime = meta ? meta[3] : (resp.mtime ?? 0);
  const uid = meta ? (meta[4] ?? 0) : (resp.uid ?? 0);
  const gid = meta ? (meta[5] ?? 0) : (resp.gid ?? 0);
  const ino = meta ? (meta[6] ?? 0) : (resp.ino ?? 0);
  const rdev = meta ? (meta[7] ?? 0) : (resp.rdev ?? 0);
  const atime = meta ? (meta[8] ?? 0) : (resp.atime ?? 0);
  const ctime = meta ? (meta[9] ?? 0) : (resp.ctime ?? 0);
  const mtimeNsec = meta ? (meta[10] ?? 0) : 0;
  const atimeNsec = meta ? (meta[11] ?? 0) : 0;
  const ctimeNsec = meta ? (meta[12] ?? 0) : 0;
  const blocks = meta ? (meta[13] ?? 0) : 0;
  const blksize = meta ? (meta[14] ?? 4096) : 4096;
  // st_dev (nonzero) + st_ino (real inode, unique per file). CRITICAL: musl's ld.so dedups shared
  // libraries by (st_dev, st_ino) — a zero/constant ino makes it treat every lib as already-loaded
  // and silently skip all but the first, so their symbols never resolve.
  view.setBigUint64(base + 0, 1n, true);              // st_dev
  view.setBigUint64(base + 8, BigInt.asUintN(64, BigInt(Math.trunc(ino))), true); // st_ino
  view.setUint32(base + 16, mode, true);
  view.setUint32(base + 20, nlink, true);
  view.setUint32(base + 24, uid, true);   // st_uid
  view.setUint32(base + 28, gid, true);   // st_gid
  // st_rdev IS the identity of a device node: major:minor is what the node is for, and a program
  // that opens one and asks what it got reads it from here.
  view.setBigUint64(base + 32, BigInt.asUintN(64, BigInt(Math.trunc(rdev))), true);
  view.setBigUint64(base + 48, BigInt(size), true);
  // st_blksize (4) then st_blocks (8) at 64: `du`, `tar --sparse` and stat(1) all read them, and
  // a file that claims zero blocks reads as a hole the size of itself.
  view.setUint32(base + 56, blksize || 4096, true);
  view.setBigUint64(base + 64, BigInt(Math.trunc(blocks)), true);
  view.setBigInt64(base + 72, BigInt(Math.trunc(atime)), true);
  view.setBigInt64(base + 80, BigInt(Math.trunc(atimeNsec)), true);
  view.setBigUint64(base + 88, BigInt(mtime), true);
  view.setBigInt64(base + 96, BigInt(Math.trunc(mtimeNsec)), true);
  view.setBigInt64(base + 104, BigInt(Math.trunc(ctime)), true);
  view.setBigInt64(base + 112, BigInt(Math.trunc(ctimeNsec)), true);
}
