/**
 * POSIX-compliant in-memory VFS for the jitter ARM64 emulator.
 *
 * Implements the Linux three-level fd model:
 *   fd (per-process integer, has FD_CLOEXEC) →
 *   OFD (open file description: shared offset + flags) →
 *   Inode (actual file/dir/chardev/symlink data)
 *
 * Reference: Linux man pages (man7.org), aarch64 struct stat layout.
 */

import { GUEST_MEM_BASE } from './types.js';
import { PIPE_PAGES_SOFT } from './kernel-ipc.js';
import type { VFS } from './syscall.js';

// ── Error constants (Linux errno values) ────────────────────────────────────

const EPERM       = 1;
const ENOENT      = 2;
const ENXIO       = 6;
const EBADF       = 9;
const EAGAIN      = 11;
const EACCES      = 13;
const EEXIST      = 17;
const ENOTDIR     = 20;
const EISDIR      = 21;
const EINVAL      = 22;
const EMFILE      = 24;
const ESPIPE      = 29;
const EPIPE       = 32;
const ERANGE      = 34;
const ENAMETOOLONG = 36;
const ENOTEMPTY   = 39;
const ELOOP       = 40;

// ── O_ flags (aarch64 Linux values) ────────────────────────────────────────

const O_ACCMODE   = 0o3;
const O_RDONLY    = 0o0;
const O_WRONLY    = 0o1;
const O_RDWR      = 0o2;
const O_CREAT     = 0o100;
const O_EXCL      = 0o200;
const O_TRUNC     = 0o1000;
const O_APPEND    = 0o2000;
const O_NONBLOCK  = 0o4000;
const O_DIRECTORY = 0o40000;   // 0x4000
const O_NOFOLLOW  = 0o100000;  // 0x8000 (0o400000 is O_LARGEFILE, which musl sets on every open)
const O_PATH      = 0o10000000; // 0x200000: open the location only; read/write/etc. give EBADF
const O_CLOEXEC   = 0o2000000;

// ── S_IF constants ─────────────────────────────────────────────────────────

const S_IFMT   = 0o170000;
const S_IFIFO  = 0o010000;
const S_IFCHR  = 0o020000;
const S_IFDIR  = 0o040000;
const S_IFBLK  = 0o060000;
const S_IFREG  = 0o100000;
const S_IFLNK  = 0o120000;
const S_IFSOCK = 0o140000;

const NAME_MAX = 255;
const PATH_MAX = 4096;
const MAX_SYMLINK_DEPTH = 40;
const MAX_FDS = 1024;

// ── Inode types ─────────────────────────────────────────────────────────────

interface InodeBase {
  ino: number;       // stable inode number
  mode: number;      // file type + permission bits
  nlink: number;     // hard link count
  uid: number;
  gid: number;
  atime: number;     // seconds since epoch
  mtime: number;
  ctime: number;
}

interface FileInode extends InodeBase {
  type: 'file';
  data: Uint8Array;
  // synthetic /proc files whose content depends on live kernel state: refreshed on every open.
  dynamic?: () => Uint8Array;
  // a file whose content is computed per RANGE rather than materialised. /proc/self/pagemap is one
  // 64-bit word per page of the address space -- megabytes of mostly-identical entries -- and a
  // reader seeks straight to the page it cares about. building the whole thing on every open would
  // be pure waste.
  ranged?: { size: number; read: (offset: number, len: number) => Uint8Array };
  /** a sysctl: writing it changes a system setting rather than a file's contents. returns 0 or a
   *  negative errno. without this, `echo 4096 > /proc/sys/...` succeeds, changes the bytes, and
   *  changes nothing about the system -- a setting that reads back the value you wrote and is
   *  ignored by everything is worse than one that refuses. */
  sysctlWrite?: (text: string) => number;
}

interface DirInode extends InodeBase {
  type: 'dir';
  children: Map<string, Inode>;
  parent: DirInode | null;  // null for root
}

interface CharDevInode extends InodeBase {
  type: 'chardev';
  // the REAL linux device numbers. these are not decoration: st_rdev is how a program identifies
  // a device (and how it tells /dev/null from /dev/zero without trusting the name), and a node
  // copied out of here onto a real filesystem has to carry numbers that mean the same thing there.
  major: number;
  minor: number;
}

interface SymlinkInode extends InodeBase {
  type: 'symlink';
  target: string;
}

interface FifoInode extends InodeBase {
  type: 'fifo';
  readers?: number; // count of open read-ends, so O_WRONLY|O_NONBLOCK with no reader gives ENXIO
}

type Inode = FileInode | DirInode | CharDevInode | SymlinkInode | FifoInode;

/** one live process, as read out of the kernel's shared process table. */
export interface ProcInfo {
  pid: number;
  ppid: number;
  pgid: number;
  sid?: number;
  tgid: number;
  state: number;
  blocked: boolean;
  comm: string;
  cmdline: string;
  utimeMs?: number;
  syscall?: number;
  stimeMs?: number;
  nice?: number;
}

// ── Open File Description (OFD) ────────────────────────────────────────────
// Created by open(), shared between dup'd fds and across fork.

interface FileOFD {
  type: 'file';
  path: string;
  inode: FileInode;
  pos: number;       // shared seek position
  flags: number;     // O_ status flags (O_APPEND, O_NONBLOCK, etc.)
  accessMode: number; // O_RDONLY/O_WRONLY/O_RDWR (immutable after open)
  refcount: number;
  oPath?: boolean;   // opened O_PATH: location-only, read/write/etc. give EBADF
}

interface DirOFD {
  type: 'dir';
  path: string;
  inode: DirInode;
  entries: string[]; // snapshot of children at open time, including . and ..
  pos: number;       // index for getdents64
  refcount: number;
}

interface CharDevOFD {
  type: 'chardev';
  path: string;
  inode: CharDevInode;
  refcount: number;
}

interface FifoOFD {
  type: 'fifo';
  path: string;
  inode: FifoInode;
  accessMode: number;
  refcount: number;
}

type OFD = FileOFD | DirOFD | CharDevOFD | FifoOFD;

// ── InMemoryVFS ─────────────────────────────────────────────────────────────

/**
 * how many cpus the guest has.
 *
 * a guest process is a worker and a guest thread is a worker, and the browser runs them on the
 * machine's real cores -- so the honest number is the host's, bounded at 64 because that is what
 * one word of an affinity mask holds.
 */
/**
 * a gzip stream whose deflate blocks are STORED (uncompressed).
 *
 * /proc/config.gz is gzip because that is the name and format the kernel exposes it under, and
 * programs open it with a gzip reader. stored blocks are a legal deflate encoding, so the file is
 * a real gzip file that any reader accepts -- and it needs neither a compressor nor an async api
 * to produce, which matters because this filesystem is built synchronously.
 */
function gzipStored(data: Uint8Array): Uint8Array {
  const blocks = Math.max(1, Math.ceil(data.length / 65535));
  const out = new Uint8Array(10 + blocks * 5 + data.length + 8);
  out.set([0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3], 0);
  let o = 10;
  for (let i = 0; i < blocks; i++) {
    const start = i * 65535;
    const len = Math.min(65535, data.length - start);
    out[o++] = i === blocks - 1 ? 1 : 0;        // BFINAL on the last, BTYPE=00 (stored)
    out[o++] = len & 0xff; out[o++] = (len >> 8) & 0xff;
    out[o++] = ~len & 0xff; out[o++] = (~len >> 8) & 0xff;
    out.set(data.subarray(start, start + len), o);
    o += len;
  }
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i++) {
    crc ^= data[i];
    for (let b = 0; b < 8; b++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  crc = (crc ^ 0xffffffff) >>> 0;
  const dv = new DataView(out.buffer);
  dv.setUint32(o, crc, true);
  dv.setUint32(o + 4, data.length >>> 0, true);
  return out;
}

export function guestCpus(): number {
  const n = (globalThis as any).navigator?.hardwareConcurrency;
  return Math.max(1, Math.min(64, typeof n === 'number' && n > 0 ? Math.floor(n) : 1));
}

// one pagemap entry per page of the largest address space a process can hold (1.5 GiB), which is
// what bounds a seek into /proc/self/pagemap.
const PAGEMAP_PAGES = 1610612736 / 4096;

export class InMemoryVFS implements VFS {
  private root: DirInode;
  private openFds = new Map<number, OFD>();
  // when set (by the syscall handler), fd numbers are drawn from the shared guest
  // namespace so vfs fds never collide with pipe/native/socket fds. left null for
  // standalone use, where we fall back to our own lowest-free scan over openFds.
  fdAllocator: (() => number) | null = null;
  // credentials of the calling process, injected by the syscall handler for permission
  // checks. null (standalone/tests) means "act as root" (all access granted).
  credsProvider: (() => { ruid: number; euid: number; rgid: number; egid: number }) | null = null;
  // this process's own cpu accounting, live. the shared table is only refreshed periodically, and
  // a program that spins on its own utime waiting for it to become nonzero needs the exact number
  // rather than the last one published.
  selfCpuProvider: (() => { utimeMs: number; stimeMs: number }) | null = null;
  // path-resolution search-permission state. resolve() enforces +x (search) on each directory
  // component; searchDenied records when a walk failed on a permission (EACCES) vs a missing
  // entry (ENOENT). searchEff selects effective vs real creds for the walk (access() uses real
  // by default, most syscalls use effective). set per-resolve.
  private searchDenied = false;
  private searchEff = true;
  // errno for the most recent failed resolve() (ENOENT/ENOTDIR/EACCES/ELOOP/ENAMETOOLONG).
  // callers that get a null inode use resolveErrno() to return the correct negative errno.
  private lastResolveErrno = ENOENT;
  // locked-memory accounting (kb) reported via /proc/self/status VmLck. updated by the syscall
  // handler on mmap(MAP_LOCKED)/mlock/munlock/munmap of locked regions.
  vmLckKb = 0;
  // lazy /proc/self/smaps, alongside maps: the same walk with what each mapping costs.
  procSmapsGen: (() => string) | null = null;

  // lazy /proc/self/maps: the syscall handler registers a renderer that walks its live vma set
  // only when the file is actually read, so mapping-heavy workloads don't pay per-mmap rendering.
  procMapsGen: (() => string) | null = null;

  private nextIno = 1;      // inode counter
  private cwd = '/';
  private cwdInode: DirInode;

  constructor(opts: { bridgeHandlesRealFs?: boolean } = {}) {
    this.root = this.makeDir(null, 0o755);
    this.cwdInode = this.root;
    this.populateVirtual();
    if (!opts.bridgeHandlesRealFs) {
      this.populateNativeFallback();
    }
  }

  // ── Inode creation helpers ──────────────────────────────────────────────

  private now(): number { return Math.floor(Date.now() / 1000); }

  // stamp a freshly created inode with its creator's ownership. new files take the
  // creating process's euid; the gid is inherited from a set-group-ID parent directory,
  // otherwise the creator's egid. a new directory under a sgid parent is itself sgid.
  private applyCreatorOwnership(inode: Inode, parent: DirInode): void {
    const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
    inode.uid = c.euid;
    inode.gid = (parent.mode & 0o2000) ? parent.gid : c.egid;
    if (inode.type === 'dir' && (parent.mode & 0o2000)) inode.mode |= 0o2000;
  }

  private makeFile(mode: number, data: Uint8Array = new Uint8Array(0)): FileInode {
    const t = this.now();
    return { type: 'file', ino: this.nextIno++, mode: S_IFREG | (mode & 0o7777), nlink: 1, uid: 0, gid: 0, atime: t, mtime: t, ctime: t, data };
  }

  private makeDir(parent: DirInode | null, mode: number): DirInode {
    const t = this.now();
    const dir: DirInode = { type: 'dir', ino: this.nextIno++, mode: S_IFDIR | (mode & 0o7777), nlink: 2, uid: 0, gid: 0, atime: t, mtime: t, ctime: t, children: new Map(), parent: null };
    dir.parent = parent ?? dir; // root's parent is itself
    return dir;
  }

  private makeCharDev(major: number, minor: number, mode: number): CharDevInode {
    const t = this.now();
    return { type: 'chardev', ino: this.nextIno++, mode: S_IFCHR | (mode & 0o7777), nlink: 1, uid: 0, gid: 0, atime: t, mtime: t, ctime: t, major, minor };
  }

  private makeSymlink(target: string): SymlinkInode {
    const t = this.now();
    return { type: 'symlink', ino: this.nextIno++, mode: S_IFLNK | 0o777, nlink: 1, uid: 0, gid: 0, atime: t, mtime: t, ctime: t, target };
  }

  private makeFifo(mode: number): FifoInode {
    const t = this.now();
    return { type: 'fifo', ino: this.nextIno++, mode: S_IFIFO | (mode & 0o7777), nlink: 1, uid: 0, gid: 0, atime: t, mtime: t, ctime: t };
  }

  mkfifo(path: string, mode: number): number {
    const parent = this.resolveParent(path);
    if (!parent) return this.resolveErrno();
    const [dir, name] = parent;
    if (dir.children.has(name)) return -EEXIST;
    if (!this.canWrite(dir)) return -EACCES;
    const fifo = this.makeFifo(mode);
    this.applyCreatorOwnership(fifo, dir);
    dir.children.set(name, fifo);
    dir.mtime = dir.ctime = this.now();
    return 0;
  }

  // ── Default filesystem population ───────────────────────────────────────

  private populateVirtual(): void {
    this.mkdirp('/dev');
    this.mkdirp('/proc/self/fd');
    this.mkdirp('/proc/sys/kernel');
    this.mkdirp('/proc/sys/fs');
    this.mkdirp('/proc/sys/vm');

    // linux's own assignments (Documentation/admin-guide/devices.txt), so a program that checks
    // st_rdev, or a node copied to a real filesystem, means the same thing here as anywhere else.
    this.addCharDev('/dev/null', 1, 3, 0o666);
    this.addCharDev('/dev/zero', 1, 5, 0o666);
    this.addCharDev('/dev/full', 1, 7, 0o666);
    this.addCharDev('/dev/random', 1, 8, 0o444);
    this.addCharDev('/dev/urandom', 1, 9, 0o444);
    this.addCharDev('/dev/tty', 5, 0, 0o666);
    // the jitter control device. reads/writes are special-cased in the syscall layer (the request is
    // dispatched host-side); the node exists here so it shows up in /dev and can be stat'd like any
    // other device.
    // 240-254 are linux's local/experimental character majors, which is exactly what this is.
    this.addCharDev('/dev/jitter', 240, 0, 0o600);

    this.addSymlinkNode('/proc/self/exe', '/bin/busybox');

    this.addFileContent('/proc/meminfo',
      'MemTotal:       1048576 kB\nMemFree:         524288 kB\nMemAvailable:    786432 kB\n' +
      'Buffers:          65536 kB\nCached:          131072 kB\nSwapTotal:            0 kB\nSwapFree:             0 kB\n');
    // one block per cpu the guest can actually run on. processes are workers and threads are
    // workers, and the browser spreads them over the machine's real cores, so the count is the
    // host's -- claiming a single processor made every program that sizes a thread pool by it run
    // single-threaded on hardware that is not.
    this.addFileContent('/proc/cpuinfo',
      Array.from({ length: guestCpus() }, (_, i) =>
        `processor\t: ${i}\nBogoMIPS\t: 48.00\nFeatures\t: fp asimd aes sha1 sha2 crc32\n` +
        'CPU implementer\t: 0x61\nCPU architecture: 8\nCPU variant\t: 0x1\nCPU part\t: 0x000\n').join('\n'));
    // the entropy the random pool holds. ours is the host csprng, which is never short of it, and
    // RNDGETENTCNT on /dev/random has to report the same number this file does.
    this.addFileContent('/proc/sys/kernel/random/entropy_avail', '4096\n');
    this.addFileContent('/proc/sys/kernel/random/poolsize', '4096\n');
    this.addFileContent('/proc/sys/kernel/pid_max', '32768\n');
    this.addFileContent('/proc/sys/kernel/sched_rr_timeslice_ms', '100\n');
    // the id the next sysv object of each kind must be given (-1 = no request). the worker binds
    // these to the shared setting; seeding them is what gives the binding something to replace.
    this.addFileContent('/proc/sys/kernel/msg_next_id', '-1\n');
    this.addFileContent('/proc/sys/kernel/sem_next_id', '-1\n');
    this.addFileContent('/proc/sys/kernel/shm_next_id', '-1\n');
    // SEMMSL SEMMNS SEMOPM SEMMNI, in that order, exactly as linux prints them.
    this.addFileContent('/proc/sys/kernel/sem', '32000\t1024000000\t500\t32000\n');
    // the per-process timer slack, as /proc/<pid>/timerslack_ns reports it. the worker binds it
    // to the live setting; this seeds the file so the binding has something to replace.
    this.addFileContent('/proc/self/timerslack_ns', '50000\n');
    // the sysv message queue limits, which msgget/msgsnd enforce: how many queues may exist, how
    // large one message may be, and how many bytes a queue holds before a sender waits.
    // how many threads may exist at once. this is the real ceiling -- the process table's -- not a
    // number copied from a linux box: a program that sizes a thread pool by it should get an answer
    // this system can keep.
    this.addFileContent('/proc/sys/kernel/threads-max', '248\n');
    // which mappings a core dump would include. no core file is ever written, but the value is
    // read by programs deciding what to do about their own crashes.
    this.addFileContent('/proc/self/coredump_filter', '00000033\n');
    this.addFileContent('/proc/sys/kernel/msgmni', '32000\n');
    this.addFileContent('/proc/sys/kernel/msgmax', '8192\n');
    this.addFileContent('/proc/sys/kernel/msgmnb', '16384\n');
    this.addFileContent('/proc/sys/kernel/shmmax', '18446744073692774399\n');
    this.addFileContent('/proc/sys/kernel/shmall', '18446744073692774399\n');
    this.addFileContent('/proc/sys/kernel/shmmni', '4096\n');
    // no core file is ever written, but the pattern is how a program learns what WOULD be written
    // and whether a handler pipe is in the way -- which decides whether a crash reports CLD_DUMPED.
    this.addFileContent('/proc/sys/kernel/core_pattern', 'core\n');
    // the kernel's own damage report. every ltp test reads it before and after itself to notice a
    // kernel it broke, and without the file the library aborts the test rather than the check.
    this.addFileContent('/proc/sys/kernel/tainted', '0\n');
    // the posix message queue limits, which mq_open enforces.
    this.mkdirp('/proc/sys/fs/mqueue');
    this.addFileContent('/proc/sys/fs/mqueue/msg_max', '10\n');
    this.addFileContent('/proc/sys/fs/mqueue/msgsize_max', '8192\n');
    this.addFileContent('/proc/sys/fs/mqueue/queues_max', '256\n');
    this.addFileContent('/proc/sys/fs/pipe-user-pages-soft', `${PIPE_PAGES_SOFT}\n`);
    // inotify's limits. programs read these to size their own buffers, and ltp reads them to know
    // what to expect -- a missing file there is not a missing feature, it is a test that cannot run.
    this.mkdirp('/proc/sys/fs/inotify');
    this.addFileContent('/proc/sys/fs/inotify/max_queued_events', '16384\n');
    this.addFileContent('/proc/sys/fs/inotify/max_user_instances', '128\n');
    this.addFileContent('/proc/sys/fs/inotify/max_user_watches', '65536\n');
    // must agree with what F_GETPIPE_SZ actually reports; claiming a megabyte here while handing
    // out 64 KiB pipes is the kind of inconsistency a program resolves by deadlocking.
    // seeded here and immediately re-pointed at the live setting by the worker (setSysctl), so a
    // read reports what the system will actually do and a write by root changes it.
    this.addFileContent('/proc/sys/fs/pipe-max-size', '65536\n');
    // console_loglevel, default_message_loglevel, minimum_console_loglevel, default_console_loglevel.
    // ltp's library saves and restores this around any test that talks to the kernel log.
    this.addFileContent('/proc/sys/kernel/printk', '7\t4\t1\t7\n');
    this.addFileContent('/proc/sys/vm/max_map_count', '65530\n');
    // /proc/self describes whoever is asking. it used to report Pid 1 and the name "jitter" to
    // every process, which is wrong for all but the first one and is exactly what a reader uses
    // this file to find out.
    // /proc/self/pagemap. every page of guest memory is genuinely resident -- it is a wasm linear
    // memory, nothing swaps underneath it -- so PRESENT is not a convenient fiction, it is the
    // answer. the page frame number is the page's own index, which is as true as any other
    // identity we could invent for it and keeps distinct pages distinct.
    this.addFileContent('/proc/self/pagemap', '');
    this.setRangedFile('/proc/self/pagemap', PAGEMAP_PAGES * 8, (offset, len) => {
      const out = new Uint8Array(len);
      const dv = new DataView(out.buffer);
      const first = Math.floor(offset / 8);
      const limit = this.memSizeProvider ? Math.floor(this.memSizeProvider() / 4096) : 0;
      for (let i = 0; i * 8 < len; i++) {
        const page = first + i;
        if (page >= limit) continue; // outside the address space: no entry, all zero
        dv.setBigUint64(i * 8, (1n << 63n) | BigInt(page), true); // PRESENT | pfn
      }
      return out;
    });

    this.addFileContent('/proc/self/status', '');
    this.setDynamicFile('/proc/self/status', () => new TextEncoder().encode(this.procStatus(this.selfPid)));
    this.addFileContent('/proc/self/syscall', '');
    this.setDynamicFile('/proc/self/syscall', () => new TextEncoder().encode(this.procSyscall(this.selfPid)));
    this.addFileContent('/proc/self/stat', '');
    this.setDynamicFile('/proc/self/stat', () => new TextEncoder().encode(this.procStat(this.selfPid)));
    this.addFileContent('/proc/self/cmdline', '');
    this.setDynamicFile('/proc/self/cmdline', () => new TextEncoder().encode(this.procOf(this.selfPid)?.cmdline ?? ''));
    this.addFileContent('/proc/self/comm', '');
    this.setDynamicFile('/proc/self/comm', () => new TextEncoder().encode((this.procOf(this.selfPid)?.comm ?? '') + '\n'));
    this.addFileContent('/proc/self/maps', '');
    this.setDynamicFile('/proc/self/maps', () => new TextEncoder().encode(this.procMapsGen ? this.procMapsGen() : ''));
    this.addFileContent('/proc/self/smaps', '');
    this.setDynamicFile('/proc/self/smaps', () => new TextEncoder().encode(this.procSmapsGen ? this.procSmapsGen() : ''));
    const mountTable = 'tmpfs / tmpfs rw,relatime 0 0\nproc /proc proc rw,relatime 0 0\n';
    this.addFileContent('/proc/mounts', mountTable);
    // /proc/self/mounts is the per-process view of the same table, and it is the one software
    // actually opens -- a process asks what IT can see, not what the system as a whole has. we
    // have no mount namespaces, so the two are the same list, but the name has to exist.
    this.addFileContent('/proc/self/mounts', mountTable);
    this.addFileContent('/proc/self/mountinfo', '1 0 0:1 / / rw,relatime - tmpfs tmpfs rw\n');
    this.addFileContent('/proc/filesystems', 'nodev\ttmpfs\n');
    // every linux has this, and enough software reads it that not having it is a bug of its own.
    // it says the same thing uname does, because it is the same kernel being described.
    this.addFileContent('/proc/version', 'Linux version 6.1.0 (jitter@localhost) #1 SMP aarch64\n');
    this.addFileContent('/proc/cmdline', 'console=hvc0\n');
    // seconds since boot, then idle seconds across all cpus. we do not account idle time
    // separately, and a machine that has been busy the whole time reports zero there.
    this.addFileContent('/proc/uptime', '0.00 0.00\n');
    // /proc/sysvipc: the only way to ask what SysV ipc objects exist without owning them. ipcs(1)
    // reads these, and so does anything that wants to know how close the system is to its limits.
    // the registries are kernel-owned, so each read is a round trip rather than a cached copy.
    this.mkdirp('/proc/sysvipc');
    for (const which of ['sem', 'msg', 'shm'] as const) {
      this.addFileContent(`/proc/sysvipc/${which}`, '');
      this.setDynamicFile(`/proc/sysvipc/${which}`,
        () => new TextEncoder().encode(this.sysvipcProvider ? this.sysvipcProvider(which) : ''));
    }
    // honest kernel config, so ltp's .needs_kconfigs checks skip (tconf) the features we do not
    // provide instead of aborting during setup. only what the syscall layer actually implements is
    // listed: claiming an option we lack turns a clean "not supported" into a mystery failure.
    // written out here rather than pasted in as a base64 gzip blob, which nobody could edit.
    this.setFileData('/proc/config.gz', gzipStored(new TextEncoder().encode(
      '#\n# jitter emulated kernel (arm64) config\n#\n' +
      ['CONFIG_ARM64', 'CONFIG_64BIT', 'CONFIG_MMU', 'CONFIG_FUTEX', 'CONFIG_EPOLL',
       'CONFIG_EVENTFD', 'CONFIG_TIMERFD', 'CONFIG_INOTIFY_USER', 'CONFIG_SIGNALFD',
       'CONFIG_POSIX_TIMERS', 'CONFIG_POSIX_MQUEUE', 'CONFIG_PROC_FS', 'CONFIG_TMPFS',
       'CONFIG_DEVTMPFS', 'CONFIG_MULTIUSER', 'CONFIG_SYSVIPC', 'CONFIG_FILE_LOCKING',
       'CONFIG_SHMEM', 'CONFIG_ADVISE_SYSCALLS', 'CONFIG_MEMBARRIER',
       // msgrcv(MSG_COPY) is the part of checkpoint/restore this kernel has.
       'CONFIG_CHECKPOINT_RESTORE'].map((k) => k + '=y\n').join(''))), 0o444);
  }

  /// in-memory backing for real paths, used by tests and non-bridged hosts.
  /// on ipad these paths live on disk via NativeFS instead
  private populateNativeFallback(): void {
    this.mkdirp('/etc');
    this.mkdirp('/tmp');
    this.mkdirp('/root');
    this.mkdirp('/bin');
    this.mkdirp('/home');
    this.mkdirp('/var');

    this.addFileContent('/etc/passwd',
      'root:x:0:0:root:/root:/bin/sh\n' +
      'daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin\n' +
      'nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n');
    this.addFileContent('/etc/group', 'root:x:0:\ndaemon:x:1:\nnobody:x:65534:\nnogroup:x:65534:\n');
    this.addFileContent('/etc/hostname', 'jitter\n');
    this.addFileContent('/etc/shells', '/bin/sh\n');
    this.addFileContent('/etc/resolv.conf', 'nameserver 1.1.1.1\nnameserver 1.0.0.1\n');
    this.addFileContent('/etc/nsswitch.conf', 'hosts: files dns\n');
  }

  loadBinary(data: Uint8Array): void {
    this.setFileData('/bin/busybox', data, 0o755);
    const applets = [
      'sh', 'ash', 'ls', 'cat', 'echo', 'grep', 'find', 'wc', 'head', 'tail',
      'sort', 'uniq', 'sed', 'awk', 'cut', 'tr', 'tee', 'xargs',
      'cp', 'mv', 'rm', 'mkdir', 'rmdir', 'ln', 'chmod', 'chown', 'touch',
      'pwd', 'env', 'test', '[', 'true', 'false', 'yes', 'seq', 'expr',
      'date', 'sleep', 'uname', 'hostname', 'whoami', 'id', 'which',
      'du', 'df', 'free', 'uptime', 'ps', 'kill', 'basename', 'dirname',
      'readlink', 'realpath', 'stat', 'md5sum', 'sha256sum',
      'dd', 'hexdump', 'od', 'strings', 'diff', 'patch',
      'tar', 'gzip', 'gunzip', 'zcat', 'bzip2', 'bunzip2',
      'vi', 'less', 'more', 'clear',
      'wget', 'nc', 'ping', 'nslookup', 'traceroute',
    ];
    this.mkdirp('/usr/bin');
    for (const name of applets) {
      this.addSymlinkNode(`/bin/${name}`, '/bin/busybox');
      this.addSymlinkNode(`/usr/bin/${name}`, '/bin/busybox');
    }
  }

  // ── Internal tree manipulation ──────────────────────────────────────────

  private mkdirp(path: string): void {
    const parts = path.split('/').filter(p => p.length > 0);
    let current = this.root;
    for (const part of parts) {
      let child = current.children.get(part);
      if (!child) {
        const dir = this.makeDir(current, 0o755);
        current.children.set(part, dir);
        current.nlink++; // parent gains a link from child's ..
        child = dir;
      }
      if (child.type !== 'dir') return;
      current = child;
    }
  }

  private addFileContent(path: string, content: string): void {
    this.setFileData(path, new TextEncoder().encode(content), 0o644);
  }

  /** wire a /proc/sys file to the setting it actually names: reads report the live value, writes
   *  change it (or report a negative errno). */
  setSysctl(path: string, get: () => string, set: (text: string) => number): void {
    this.setDynamicFile(path, () => new TextEncoder().encode(get()));
    const parts = path.split('/').filter((p) => p.length > 0);
    const name = parts.pop()!;
    let dir = this.root;
    for (const part of parts) {
      const child = dir.children.get(part);
      if (!child || child.type !== 'dir') return;
      dir = child;
    }
    const inode = dir.children.get(name);
    if (inode && inode.type === 'file') inode.sysctlWrite = set;
  }

  /** how much guest memory exists, for the files that describe the address space. */
  memSizeProvider: (() => number) | null = null;

  private setRangedFile(path: string, size: number, read: (offset: number, len: number) => Uint8Array): void {
    const parts = path.split('/').filter((p) => p.length > 0);
    const name = parts.pop()!;
    let dir = this.root;
    for (const part of parts) {
      const child = dir.children.get(part);
      if (!child || child.type !== 'dir') return;
      dir = child;
    }
    const inode = dir.children.get(name);
    if (inode && inode.type === 'file') inode.ranged = { size, read };
  }

  private setDynamicFile(path: string, gen: () => Uint8Array): void {
    const parts = path.split('/').filter(p => p.length > 0);
    const name = parts.pop()!;
    let dir = this.root;
    for (const part of parts) {
      const child = dir.children.get(part);
      if (!child || child.type !== 'dir') return;
      dir = child;
    }
    const inode = dir.children.get(name);
    if (inode && inode.type === 'file') inode.dynamic = gen;
  }

  // public: inject a file (creating parent directories) with raw bytes. used to seed
  // the rootfs with binaries like /lib/ld-linux-aarch64.so.1 and /lib/libc.so.6.
  writeFileBytes(path: string, data: Uint8Array, mode = 0o755): void {
    const parts = path.split('/').filter(p => p.length > 0);
    parts.pop();
    let dirPath = '';
    for (const part of parts) { dirPath += '/' + part; this.mkdirp(dirPath); }
    this.setFileData(path, data, mode);
  }

  private setFileData(path: string, data: Uint8Array, mode: number): void {
    const parts = path.split('/').filter(p => p.length > 0);
    const name = parts.pop()!;
    // the directories a synthesised file sits in are part of synthesising it: bailing out when one
    // was missing dropped the file silently, and /proc/sys/kernel/random/entropy_avail simply did
    // not exist because nothing had made the directory above it first.
    this.mkdirp('/' + parts.join('/'));
    let dir = this.root;
    for (const part of parts) {
      const child = dir.children.get(part);
      if (!child || child.type !== 'dir') return;
      dir = child;
    }
    const existing = dir.children.get(name);
    if (existing && existing.type === 'file') {
      existing.data = data;
      existing.mode = S_IFREG | (mode & 0o7777);
      existing.mtime = existing.ctime = this.now();
    } else {
      dir.children.set(name, this.makeFile(mode, data));
    }
  }

  private addCharDev(path: string, major: number, minor: number, mode: number): void {
    const parts = path.split('/').filter(p => p.length > 0);
    const name = parts.pop()!;
    let dir = this.root;
    for (const part of parts) {
      const child = dir.children.get(part);
      if (!child || child.type !== 'dir') return;
      dir = child;
    }
    dir.children.set(name, this.makeCharDev(major, minor, mode));
  }

  private addSymlinkNode(path: string, target: string): void {
    const parts = path.split('/').filter(p => p.length > 0);
    const name = parts.pop()!;
    let dir = this.root;
    for (const part of parts) {
      const child = dir.children.get(part);
      if (!child || child.type !== 'dir') return;
      dir = child;
    }
    dir.children.set(name, this.makeSymlink(target));
  }

  // ── Path resolution (follows Linux symlink rules) ───────────────────────

  /**
   * Normalize a path: resolve relative paths against cwd, handle . and ..
   * Returns array of path components (avoids join+split overhead in resolve).
   */
  private normalizeParts(path: string): string[] {
    // Make absolute
    if (path.charCodeAt(0) !== 47) { // '/'
      path = this.cwd + (this.cwd.length > 1 ? '/' : '') + path;
    }
    // Split and normalize in one pass
    const result: string[] = [];
    let start = 0;
    for (let i = 0; i <= path.length; i++) {
      if (i === path.length || path.charCodeAt(i) === 47) {
        if (i > start) {
          const part = path.substring(start, i);
          if (part === '..') { result.pop(); }
          else if (part !== '.') { result.push(part); }
        }
        start = i + 1;
      }
    }
    return result;
  }

  private resolvePath(path: string): string {
    const parts = this.normalizeParts(path);
    return parts.length === 0 ? '/' : '/' + parts.join('/');
  }

  /**
   * Resolve a path to an inode, following symlinks per Linux rules.
   * Uses pre-split path components to avoid redundant string splitting.
   */
  /** live process list, supplied by the worker (which owns the shared process table). */
  procListProvider: (() => ProcInfo[]) | null = null;

  /** the text of /proc/sysvipc/<which>, straight from the kernel's registries. */
  sysvipcProvider: ((which: 'sem' | 'msg' | 'shm') => string) | null = null;
  // what /proc/self/fd should contain right now. the fd table lives in the syscall layer, so the
  // directory can only be honest if that layer hands it over. it was an empty directory, which is a
  // lie with consequences: anything that counts its own descriptors by listing it (ltp's pipe07,
  // every "close all fds above 3" loop, lsof-shaped code) concluded it had none open.
  fdListProvider: (() => Array<{ fd: number; target: string; info: string }>) | null = null;
  /** which pid /proc/self refers to; the worker keeps it current across fork and exec. */
  selfPid = 1;

  /**
   * bring /proc's per-process directories in line with who is actually alive. the tree is a plain
   * inode structure, so the pid directories are materialised here rather than being faked at read
   * time -- that way stat, open, getdents and readlink all see the same thing with no special
   * cases. called on any walk into /proc, which is the only place it can be observed.
   */
  private syncProcPids(): void {
    if (!this.procListProvider) return;
    const procDir = this.root.children.get('proc');
    if (!procDir || procDir.type !== 'dir') return;

    const live = this.procListProvider();
    const want = new Set(live.map((p) => String(p.pid)));
    for (const name of [...procDir.children.keys()]) {
      if (/^\d+$/.test(name) && !want.has(name)) procDir.children.delete(name);
    }

    for (const p of live) {
      const dir = String(p.pid);
      if (procDir.children.has(dir)) continue;
      this.mkdirp(`/proc/${dir}`);
      this.addFileContent(`/proc/${dir}/stat`, '');
      this.setDynamicFile(`/proc/${dir}/stat`, () => new TextEncoder().encode(this.procStat(p.pid)));
      // /proc/<pid>/syscall: the number of the call the process is inside, or "running".
      this.addFileContent(`/proc/${dir}/syscall`, '');
      this.setDynamicFile(`/proc/${dir}/syscall`, () => new TextEncoder().encode(this.procSyscall(p.pid)));
      this.addFileContent(`/proc/${dir}/status`, '');
      this.setDynamicFile(`/proc/${dir}/status`, () => new TextEncoder().encode(this.procStatus(p.pid)));
      this.addFileContent(`/proc/${dir}/cmdline`, '');
      this.setDynamicFile(`/proc/${dir}/cmdline`, () => new TextEncoder().encode(this.procOf(p.pid)?.cmdline ?? ''));
      this.addFileContent(`/proc/${dir}/comm`, '');
      this.setDynamicFile(`/proc/${dir}/comm`, () => new TextEncoder().encode((this.procOf(p.pid)?.comm ?? '') + '\n'));
    }

    // /proc/<pid>/task/<tid>: one directory per thread of the group. a thread's tgid names the
    // process it belongs to, so the group's members are exactly the entries sharing that tgid.
    // programs read their own name out of /proc/self/task/<tid>/comm rather than /proc/self/comm
    // when they care about the THREAD's name -- prctl(PR_SET_NAME) renames a thread, not a process.
    for (const p of live) {
      const taskPath = `/proc/${p.pid}/task`;
      this.mkdirp(taskPath);
      const members = live.filter((t) => t.tgid === p.pid);
      const taskDir = this.lookupDir(taskPath);
      if (taskDir) {
        const wantTids = new Set(members.map((t) => String(t.pid)));
        for (const name of [...taskDir.children.keys()]) if (!wantTids.has(name)) taskDir.children.delete(name);
      }
      for (const t of members) {
        const tp = `${taskPath}/${t.pid}`;
        if (taskDir?.children.has(String(t.pid))) continue;
        this.addTaskDir(tp, t.pid);
      }
    }

    // /proc/self/task mirrors the caller's own group. /proc/self is a real directory here rather
    // than a symlink, so it needs its own copy.
    const selfProc = this.procOf(this.selfPid);
    const selfGroup = selfProc ? live.filter((t) => t.tgid === selfProc.tgid) : [];
    this.mkdirp('/proc/self/task');
    const selfTaskDir = this.lookupDir('/proc/self/task');
    if (selfTaskDir) {
      const want = new Set(selfGroup.map((t) => String(t.pid)));
      for (const name of [...selfTaskDir.children.keys()]) if (!want.has(name)) selfTaskDir.children.delete(name);
      for (const t of selfGroup) {
        if (selfTaskDir.children.has(String(t.pid))) continue;
        this.addTaskDir(`/proc/self/task/${t.pid}`, t.pid);
      }
    }
  }

  /** the per-thread files under a /proc/<pid>/task/<tid> directory. */
  private addTaskDir(path: string, tid: number): void {
    this.mkdirp(path);
    this.addFileContent(`${path}/comm`, '');
    this.setDynamicFile(`${path}/comm`, () => new TextEncoder().encode((this.procOf(tid)?.comm ?? '') + '\n'));
    this.addFileContent(`${path}/stat`, '');
    this.setDynamicFile(`${path}/stat`, () => new TextEncoder().encode(this.procStat(tid)));
    this.addFileContent(`${path}/status`, '');
    this.setDynamicFile(`${path}/status`, () => new TextEncoder().encode(this.procStatus(tid)));
  }

  /** the directory inode at an absolute path, or null. */
  private lookupDir(path: string): DirInode | null {
    let node: Inode | undefined = this.root;
    for (const part of path.split('/')) {
      if (!part) continue;
      if (!node || node.type !== 'dir') return null;
      node = node.children.get(part);
    }
    return node && node.type === 'dir' ? node : null;
  }

  private syncSelfFds(): void {
    if (!this.fdListProvider) return;
    const live = this.fdListProvider();
    // a process reaches its own descriptors by either name, so both have to be there and agree.
    this.fillFdDirs('/proc/self', live);
    this.fillFdDirs(`/proc/${this.selfPid}`, live);
  }

  /** the fd and fdinfo directories under one /proc process directory. */
  private fillFdDirs(base: string, live: Array<{ fd: number; target: string; info: string }>): void {
    this.mkdirp(`${base}/fd`);
    this.mkdirp(`${base}/fdinfo`);
    const dir = this.lookupDir(`${base}/fd`);
    const infoDir = this.lookupDir(`${base}/fdinfo`);
    if (!dir || !infoDir) return;
    dir.children.clear();
    infoDir.children.clear();
    for (const e of live) {
      dir.children.set(String(e.fd), this.makeSymlink(e.target));
      this.addFileContent(`${base}/fdinfo/${e.fd}`, e.info);
    }
  }

  private procOf(pid: number): ProcInfo | null {
    return this.procListProvider?.().find((p) => p.pid === pid) ?? null;
  }

  private stateLetter(p: ProcInfo): string {
    if (p.state === 2) return 'Z';   // PROC_ZOMBIE
    if (p.state === 4) return 'T';   // PROC_STOPPED -- job control, not sleeping and not gone
    return p.blocked ? 'S' : 'R';    // parked in a blocking syscall : on cpu
  }

  /**
   * /proc/<pid>/stat. the fields we can answer honestly are filled in and the rest are zero, which
   * is what a reader gets for a process with no accounting anyway. the layout matters more than the
   * values: ps and friends index fields positionally, and comm is parenthesised because it may
   * contain spaces (every parser finds it by the LAST ')').
   */
  private procStat(pid: number): string {
    const p = this.procOf(pid) ?? { pid, ppid: 0, pgid: pid, tgid: pid, state: 1, blocked: false, comm: '', cmdline: '' };
    const own = pid === this.selfPid ? this.selfCpuProvider?.() : undefined;
    // cpu times here are in clock ticks, and USER_HZ is 100 -- so a hundredth of a second each.
    const ticks = (ms: number) => Math.floor(ms / 10);
    const nice = p.nice ?? 0;
    const f: (string | number)[] = [p.pid, `(${p.comm})`, this.stateLetter(p), p.ppid, p.pgid, p.sid ?? p.pgid, 0, -1, 0];
    while (f.length < 52) f.push(0);
    f[13] = ticks(own?.utimeMs ?? p.utimeMs ?? 0);   // utime
    f[14] = ticks(own?.stimeMs ?? p.stimeMs ?? 0);   // stime
    f[17] = 20 + nice;                               // priority
    f[18] = nice;
    f[19] = 1; // num_threads
    return f.join(' ') + '\n';
  }

  private procSyscall(pid: number): string {
    const n = this.procOf(pid)?.syscall ?? -1;
    return n < 0 ? 'running\n' : `${n} 0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x0\n`;
  }

  private procStatus(pid: number): string {
    const p = this.procOf(pid) ?? { pid, ppid: 0, pgid: pid, tgid: pid, state: 1, blocked: false, comm: '', cmdline: '' };
    const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
    const L = this.stateLetter(p);
    const state = L === 'Z' ? 'Z (zombie)' : (L === 'S' ? 'S (sleeping)' : 'R (running)');
    return `Name:\t${p.comm}\nUmask:\t0022\nState:\t${state}\nTgid:\t${p.tgid}\nNgid:\t0\n` +
      `Pid:\t${p.pid}\nPPid:\t${p.ppid}\nTracerPid:\t0\n` +
      `Uid:\t${c.ruid}\t${c.euid}\t${c.euid}\t${c.euid}\nGid:\t${c.rgid}\t${c.egid}\t${c.egid}\t${c.egid}\n` +
      `FDSize:\t64\nGroups:\t0\nVmPeak:\t    65536 kB\nVmSize:\t    65536 kB\nVmRSS:\t    32768 kB\n` +
      `VmLck:\t${String(this.vmLckKb).padStart(8)} kB\nVmSwap:\t       0 kB\nThreads:\t1\n`;
  }

  private resolve(path: string, followFinal = true, depth = 0, searchEff = true): Inode | null {
    if (depth === 0) {
      this.searchDenied = false; this.searchEff = searchEff; this.lastResolveErrno = ENOENT; // fresh walk
      if (path.startsWith('/proc') || this.cwd.startsWith('/proc')) { this.syncProcPids(); this.syncSelfFds(); }
    }
    if (depth > MAX_SYMLINK_DEPTH) { this.lastResolveErrno = ELOOP; return null; }
    const parts = this.normalizeParts(path);
    for (const part of parts) {
      if (part.length > NAME_MAX) { this.lastResolveErrno = ENAMETOOLONG; return null; }
    }
    return this.resolveFromParts(parts, followFinal, depth);
  }

  // the errno (negative) for the most recent failed resolve().
  resolveErrno(): number { return -this.lastResolveErrno; }

  // execute (search) permission on a directory: needed to traverse into/through it. root can
  // always search. non-root needs the x bit for owner/group/other as applicable.
  private canSearch(dir: InodeBase): boolean {
    const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
    const uid = this.searchEff ? c.euid : c.ruid;
    const gid = this.searchEff ? c.egid : c.rgid;
    if (uid === 0) return true;
    let perms: number;
    if (dir.uid === uid) perms = (dir.mode >> 6) & 7;
    else if (dir.gid === gid) perms = (dir.mode >> 3) & 7;
    else perms = dir.mode & 7;
    return (perms & 1) !== 0;
  }

  // normalize a raw list of path segments (which may contain '', '.', '..') into clean
  // components, resolving '..' against earlier entries. used when splicing a symlink target
  // into a path mid-resolution, where cwd-prepending must not happen.
  private normalizePartsArray(segments: string[]): string[] {
    const out: string[] = [];
    for (const p of segments) {
      if (p === '' || p === '.') continue;
      if (p === '..') out.pop();
      else out.push(p);
    }
    return out;
  }

  // write (modify) permission on a directory: needed to create, remove, or rename entries in
  // it. root bypasses; others need the +w bit for owner/group/other as applicable (effective).
  private canWrite(dir: InodeBase): boolean {
    return this.canAccess(dir, 2);
  }

  // DAC check: does the effective uid/gid have all of `need` (4=read|2=write|1=exec) on the inode?
  // root (euid 0) bypasses. mirrors the kernel's owner/group/other mode-bit selection.
  private canAccess(inode: InodeBase, need: number): boolean {
    const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
    if (c.euid === 0) return true;
    let perms: number;
    if (inode.uid === c.euid) perms = (inode.mode >> 6) & 7;
    else if (inode.gid === c.egid) perms = (inode.mode >> 3) & 7;
    else perms = inode.mode & 7;
    return (perms & need) === need;
  }

  private resolveFromParts(parts: string[], followFinal: boolean, depth: number): Inode | null {
    if (depth > MAX_SYMLINK_DEPTH) { this.lastResolveErrno = ELOOP; return null; }
    if (parts.length === 0) return this.root;

    let current: Inode = this.root;
    for (let i = 0; i < parts.length; i++) {
      // a non-directory component in the middle of the path is ENOTDIR
      if (current.type !== 'dir') { this.lastResolveErrno = ENOTDIR; return null; }
      // must be able to search this directory to look up the next component
      if (!this.canSearch(current)) { this.searchDenied = true; this.lastResolveErrno = EACCES; return null; }
      const child = current.children.get(parts[i]);
      if (!child) { this.lastResolveErrno = ENOENT; return null; }

      const isLast = i === parts.length - 1;
      if (child.type === 'symlink' && (followFinal || !isLast)) {
        // rebuild the path from the symlink target plus the unvisited components. a relative
        // target resolves against the directory containing the link (parts[0..i-1]); an
        // absolute target resolves from the root.
        const target = child.target;
        const base = target.startsWith('/') ? [] : parts.slice(0, i);
        const combined = [...base, ...target.split('/'), ...parts.slice(i + 1)];
        return this.resolveFromParts(this.normalizePartsArray(combined), followFinal, depth + 1);
      }

      current = child;
    }
    return current;
  }

  /** Resolve parent directory of path. Returns [parentDirInode, childName] or null; on null,
   *  lastResolveErrno holds the reason (ENOENT/ENOTDIR/ENAMETOOLONG/EACCES/ELOOP). */
  private resolveParent(path: string, depth = 0): [DirInode, string] | null {
    if (depth === 0) {
      this.lastResolveErrno = ENOENT;
      if (path.startsWith('/proc') || this.cwd.startsWith('/proc')) { this.syncProcPids(); this.syncSelfFds(); }
    }
    if (depth > MAX_SYMLINK_DEPTH) { this.lastResolveErrno = ELOOP; return null; }

    const resolved = this.resolvePath(path);
    const parts = resolved.split('/').filter(p => p.length > 0);
    if (parts.length === 0) return null; // can't get parent of root
    const name = parts.pop()!;
    if (name.length > NAME_MAX) { this.lastResolveErrno = ENAMETOOLONG; return null; }

    let dir = this.root;
    for (let idx = 0; idx < parts.length; idx++) {
      const part = parts[idx];
      if (part.length > NAME_MAX) { this.lastResolveErrno = ENAMETOOLONG; return null; }
      if (!this.canSearch(dir)) { this.searchDenied = true; this.lastResolveErrno = EACCES; return null; }
      const child = dir.children.get(part);
      if (!child) { this.lastResolveErrno = ENOENT; return null; }
      if (child.type === 'symlink') {
        // a relative symlink target resolves against the directory holding the link (parts[0..idx-1]);
        // only an absolute target resolves from root. omitting that prefix made relative "../x"
        // self-loops resolve against cwd and disappear into ENOENT instead of looping to ELOOP.
        const target = child.target;
        const base = target.startsWith('/') ? '' : '/' + parts.slice(0, idx).join('/');
        const remaining = parts.slice(idx + 1).join('/');
        const newPath = [base, target, remaining, name].filter(s => s.length > 0).join('/').replace(/\/+/g, '/');
        return this.resolveParent(newPath, depth + 1);
      }
      if (child.type !== 'dir') { this.lastResolveErrno = ENOTDIR; return null; }
      dir = child;
    }
    if (!this.canSearch(dir)) { this.searchDenied = true; this.lastResolveErrno = EACCES; return null; }
    return [dir, name];
  }

  // ── VFS interface implementation ────────────────────────────────────────

  private takeFd(): number {
    if (this.fdAllocator) return this.fdAllocator();
    let fd = 3;
    while (this.openFds.has(fd)) fd++;
    return fd;
  }

  open(path: string, flags: number, mode: number): number {
    const resolved = this.resolvePath(path);
    // Validate path length
    if (path.length > PATH_MAX) return -ENAMETOOLONG;
    for (const part of path.split('/')) {
      if (part.length > NAME_MAX) return -ENAMETOOLONG;
    }

    // Check fd limit
    if (this.openFds.size >= MAX_FDS) return -EMFILE;

    const followSymlinks = !(flags & O_NOFOLLOW);
    let inode = this.resolve(path, followSymlinks);
    let created = false;

    // O_NOFOLLOW: if the final component is itself a symlink, fail ELOOP (unless O_CREAT would make
    // it, or O_PATH is set which we don't model). resolve(path,false) returns the symlink inode.
    if ((flags & O_NOFOLLOW) && !(flags & O_CREAT)) {
      if (inode && inode.type === 'symlink') return -ELOOP;
      if (!inode) {
        const raw = this.resolve(path, false);
        if (raw && raw.type === 'symlink') return -ELOOP;
      }
    }

    // O_CREAT | O_EXCL: fail if exists
    if (inode && (flags & O_CREAT) && (flags & O_EXCL)) {
      return -EEXIST;
    }

    // O_CREAT: create if not exists
    if (!inode && (flags & O_CREAT)) {
      const parent = this.resolveParent(path);
      if (!parent) return this.resolveErrno();
      const [dir, name] = parent;
      if (!this.canWrite(dir)) return -EACCES;
      const newFile = this.makeFile(mode & 0o7777);
      this.applyCreatorOwnership(newFile, dir);
      dir.children.set(name, newFile);
      dir.mtime = dir.ctime = this.now();
      inode = newFile;
      created = true;
    }

    if (!inode) return this.resolveErrno();

    // enforce the target's DAC permissions against the requested access mode. skipped for a file we
    // just created (the creator is granted access), for directories (handled via search/EISDIR), and
    // for O_PATH (which opens the location only and needs no read/write permission on the file).
    if (!created && inode.type !== 'dir' && !(flags & O_PATH)) {
      const acc = flags & O_ACCMODE;
      let need = 0;
      if (acc === O_RDONLY || acc === O_RDWR) need |= 4;
      if (acc === O_WRONLY || acc === O_RDWR) need |= 2;
      if (need && !this.canAccess(inode, need)) return -EACCES;

      // O_NOATIME is only honored for the file's owner (or root); others get EPERM.
      if (flags & 0o1000000) {
        const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
        if (c.euid !== 0 && inode.uid !== c.euid) return -EPERM;
      }
    }

    // O_DIRECTORY: must be a directory
    if ((flags & O_DIRECTORY) && inode.type !== 'dir') return -ENOTDIR;

    // Directory handling
    if (inode.type === 'dir') {
      // O_WRONLY or O_RDWR on directory is EISDIR
      if ((flags & O_ACCMODE) !== O_RDONLY) return -EISDIR;
      const fd = this.takeFd();
      // Snapshot entries including . and ..
      const entries = ['.', '..', ...inode.children.keys()];
      this.openFds.set(fd, { type: 'dir', path: resolved, inode, entries, pos: 0, refcount: 1 });
      return fd;
    }

    // Character device
    if (inode.type === 'chardev') {
      const fd = this.takeFd();
      this.openFds.set(fd, { type: 'chardev', path: resolved, inode, refcount: 1 });
      return fd;
    }

    // Named pipe (fifo). we model it as a seekless endpoint: open succeeds (the test opens
    // O_RDWR, which does not block), stat reports S_IFIFO, and lseek returns ESPIPE. a write-only
    // non-blocking open with no reader present faults ENXIO, matching the kernel.
    if (inode.type === 'fifo') {
      const acc = flags & O_ACCMODE;
      if (acc === O_WRONLY && (flags & O_NONBLOCK) && (inode.readers ?? 0) === 0) return -ENXIO;
      const fd = this.takeFd();
      if (acc === O_RDONLY || acc === O_RDWR) inode.readers = (inode.readers ?? 0) + 1;
      this.openFds.set(fd, { type: 'fifo', path: resolved, inode, accessMode: acc, refcount: 1 });
      return fd;
    }

    // Symlink (only reached if O_NOFOLLOW and it's a symlink — already handled above)
    if (inode.type !== 'file') return -ENOENT;

    if (inode.dynamic) inode.data = inode.dynamic();

    // O_TRUNC: truncate to 0 (requires write access)
    if ((flags & O_TRUNC) && ((flags & O_ACCMODE) !== O_RDONLY)) {
      inode.data = new Uint8Array(0);
      inode.mtime = inode.ctime = this.now();
    }

    const fd = this.takeFd();
    const accessMode = flags & O_ACCMODE;
    const pos = (flags & O_APPEND) ? inode.data.length : 0;
    this.openFds.set(fd, { type: 'file', path: resolved, inode, pos, flags, accessMode, refcount: 1, oPath: (flags & O_PATH) !== 0 });
    return fd;
  }

  /** true if the fd was opened O_PATH: it references a location only, so read/write/fchmod/fchown/
   *  ioctl/mmap on it must fail EBADF. */
  /** the path an open descriptor was opened by, for /proc/self/fd. */
  fdPath(fd: number): string | null { return this.openFds.get(fd)?.path ?? null; }

  /** true when the descriptor names a character device. fsync has nothing to flush for one, which
   *  is why linux answers EINVAL rather than pretending it wrote something back. */
  isCharDev(fd: number): boolean { return this.openFds.get(fd)?.type === 'chardev'; }

  /** does this descriptor name a directory? *at() syscalls need to know before they resolve. */
  fdIsDir(fd: number): boolean { return this.openFds.get(fd)?.type === 'dir'; }

  /** every descriptor this vfs holds open. */
  openFdNumbers(): number[] { return [...this.openFds.keys()]; }

  isOPath(fd: number): boolean {
    return (this.openFds.get(fd) as { oPath?: boolean } | undefined)?.oPath === true;
  }

  close(fd: number): number {
    const ofd = this.openFds.get(fd);
    if (!ofd) return -EBADF;
    this.openFds.delete(fd);
    ofd.refcount--;
    // last reference to a fifo read-end: drop the reader count so a later O_WRONLY|O_NONBLOCK
    // open sees no reader and faults ENXIO.
    if (ofd.refcount <= 0 && ofd.type === 'fifo' && (ofd.accessMode === O_RDONLY || ofd.accessMode === O_RDWR)) {
      ofd.inode.readers = Math.max(0, (ofd.inode.readers ?? 0) - 1);
    }
    return 0;
  }

  hasFd(fd: number): boolean {
    return this.openFds.has(fd);
  }

  // the O_RDONLY/O_WRONLY/O_RDWR access mode a file descriptor was opened with, or -1 if
  // the fd is unknown or has no access mode (e.g. a directory handle).
  fdAccessMode(fd: number): number {
    const ofd = this.openFds.get(fd) as { accessMode?: number } | undefined;
    return ofd && typeof ofd.accessMode === 'number' ? ofd.accessMode : -1;
  }

  dup(fd: number): number {
    const ofd = this.openFds.get(fd);
    if (!ofd) return -EBADF;
    if (this.openFds.size >= MAX_FDS) return -EMFILE;
    const newFd = this.takeFd();
    ofd.refcount++;
    this.openFds.set(newFd, ofd); // same OFD object = shared position
    return newFd;
  }

  // install a shared duplicate of oldFd at a caller-chosen newFd. the caller owns
  // the fd namespace (dup2/dup3/F_DUPFD pick the target), so newFd must already be
  // free. keeps the guest fd == vfs internal fd invariant for dup'd fds.
  dupTo(oldFd: number, newFd: number): number {
    const ofd = this.openFds.get(oldFd);
    if (!ofd) return -EBADF;
    ofd.refcount++;
    this.openFds.set(newFd, ofd);
    return newFd;
  }

  read(fd: number, buf: number, count: number, memory: DataView): number {
    const ofd = this.openFds.get(fd);
    if (!ofd) return -EBADF;
    if (ofd.type === 'file' && ofd.oPath) return -EBADF; // O_PATH fd: not readable

    if (ofd.type === 'chardev') {
      const bytes = new Uint8Array(memory.buffer);
      switch (`${ofd.inode.major}:${ofd.inode.minor}`) {
        case '1:3': return 0;                       // /dev/null -- always at end of file
        case '1:5': case '1:7':                     // /dev/zero, /dev/full
          bytes.fill(0, buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
          return count;
        case '1:8': case '1:9': {                   // /dev/random, /dev/urandom
          crypto.getRandomValues(bytes.subarray(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count));
          return count;
        }
        default: return 0;                          // tty and the control device read through the syscall layer
      }
    }

    if (ofd.type === 'dir') return -EISDIR;
    // fifos carry no buffered data in our model, so a read never has bytes ready: it would block,
    // which a non-blocking reader observes as EAGAIN (LTP read03 reads an empty O_NONBLOCK fifo).
    if (ofd.type === 'fifo') return -EAGAIN;
    if (ofd.type !== 'file') return -EBADF;

    // Check read permission
    if (ofd.accessMode === O_WRONLY) return -EBADF;

    const bytes = new Uint8Array(memory.buffer);
    const ranged = ofd.inode.ranged;
    if (ranged) {
      const avail = ranged.size - ofd.pos;
      const n = Math.min(count, Math.max(0, avail));
      if (n <= 0) return 0;
      bytes.set(ranged.read(ofd.pos, n), buf + GUEST_MEM_BASE);
      ofd.pos += n;
      return n;
    }

    const available = ofd.inode.data.length - ofd.pos;
    const toRead = Math.min(count, Math.max(0, available));
    if (toRead <= 0) return 0; // EOF

    bytes.set(ofd.inode.data.subarray(ofd.pos, ofd.pos + toRead), buf + GUEST_MEM_BASE);
    ofd.pos += toRead;
    ofd.inode.atime = this.now();
    return toRead;
  }

  write(fd: number, buf: number, count: number, memory: DataView): number {
    const ofd = this.openFds.get(fd);
    if (!ofd) return -EBADF;
    if (ofd.type === 'file' && ofd.oPath) return -EBADF; // O_PATH fd: not writable

    // /dev/full is the one device whose whole purpose is to fail a write, and programs test
    // out-of-space handling against exactly it.
    if (ofd.type === 'chardev') return (ofd.inode.major === 1 && ofd.inode.minor === 7) ? -28 : count;
    if (ofd.type === 'dir') return -EBADF;
    if (ofd.type !== 'file') return -EBADF;

    // Check write permission
    if (ofd.accessMode === O_RDONLY) return -EBADF;

    if (count === 0) return 0;

    const guestMax = memory.buffer.byteLength - GUEST_MEM_BASE;
    if (buf < 0 || buf >= guestMax || buf + count > guestMax) return -14; // EFAULT

    if (ofd.inode.sysctlWrite) {
      const bytes = new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, count);
      const rc = ofd.inode.sysctlWrite(new TextDecoder().decode(bytes).trim());
      return rc < 0 ? rc : count;
    }

    // O_APPEND: atomically seek to end
    if (ofd.flags & O_APPEND) {
      ofd.pos = ofd.inode.data.length;
    }

    const src = new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, count);
    const endPos = ofd.pos + count;

    // Grow file if needed
    if (endPos > ofd.inode.data.length) {
      const newData = new Uint8Array(endPos);
      newData.set(ofd.inode.data);
      ofd.inode.data = newData;
    }

    ofd.inode.data.set(src, ofd.pos);
    ofd.pos = endPos;
    ofd.inode.mtime = ofd.inode.ctime = this.now();
    return count;
  }

  lseek(fd: number, offset: bigint, whence: number): bigint {
    const ofd = this.openFds.get(fd);
    if (!ofd) return BigInt(-EBADF);
    if (ofd.type === 'chardev' || ofd.type === 'fifo') return BigInt(-ESPIPE);
    if (ofd.type !== 'file') return BigInt(-EBADF);

    let newPos: number;
    const off = Number(offset); // already signed from caller
    switch (whence) {
      case 0: newPos = off; break;                         // SEEK_SET
      case 1: newPos = ofd.pos + off; break;               // SEEK_CUR
      case 2: newPos = (ofd.inode.ranged ? ofd.inode.ranged.size : ofd.inode.data.length) + off; break; // SEEK_END
      default: return BigInt(-EINVAL);
    }
    if (newPos < 0) return BigInt(-EINVAL);
    ofd.pos = newPos;
    return BigInt(newPos);
  }

  // only the file's owner (or root) may change its mode; otherwise EPERM.
  private chmodInode(inode: InodeBase, mode: number): number {
    const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
    if (c.euid !== 0 && c.euid !== inode.uid) return -EPERM;
    inode.mode = (inode.mode & S_IFMT) | (mode & 0o7777);
    inode.ctime = this.now();
    return 0;
  }

  fchmod(fd: number, mode: number): number {
    const ofd = this.openFds.get(fd);
    if (!ofd) return -EBADF;
    const inode = ofd.type === 'file' ? ofd.inode : ofd.type === 'dir' ? ofd.inode : null;
    if (!inode) return -EBADF;
    return this.chmodInode(inode, mode);
  }

  // shared chown logic, matching the kernel's setattr_prepare. -1 (or 0xffffffff) means
  // "don't change". without CAP_CHOWN (non-root euid): changing the owner requires that the
  // caller already owns the file and the target uid equals the current one (i.e. no real
  // change); changing the group requires ownership and that the target gid is one the caller
  // belongs to (its egid/rgid) or the file's current gid. a successful chown of a non-directory
  // drops the set-user-ID bit, and the set-group-ID bit for group-executable files.
  private chownInode(inode: InodeBase, uid: number, gid: number): number {
    const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
    const noChg = (v: number) => v === -1 || v === 0xFFFFFFFF;
    const uidPresent = !noChg(uid);
    const gidPresent = !noChg(gid);
    const wantUid = uidPresent ? uid : inode.uid;
    const wantGid = gidPresent ? gid : inode.gid;
    if (c.euid !== 0) {
      if (uidPresent && (inode.uid !== c.euid || wantUid !== inode.uid)) return -EPERM;
      if (gidPresent && (inode.uid !== c.euid ||
          (wantGid !== c.egid && wantGid !== c.rgid && wantGid !== inode.gid))) return -EPERM;
    }
    inode.uid = wantUid;
    inode.gid = wantGid;
    if ((inode.mode & S_IFMT) !== S_IFDIR) {
      inode.mode &= ~0o4000;                       // clear set-user-ID
      if (inode.mode & 0o010) inode.mode &= ~0o2000; // clear set-group-ID if group-executable
    }
    inode.ctime = this.now();
    return 0;
  }

  fchown(fd: number, uid: number, gid: number): number {
    const ofd = this.openFds.get(fd);
    if (!ofd) return -EBADF;
    const inode = ofd.type === 'file' ? ofd.inode : ofd.type === 'dir' ? ofd.inode : null;
    if (!inode) return -EBADF;
    return this.chownInode(inode, uid, gid);
  }

  chmodPath(path: string, mode: number): number {
    const inode = this.resolve(path);
    if (!inode) return this.resolveErrno();
    return this.chmodInode(inode, mode);
  }

  chownPath(path: string, uid: number, gid: number): number {
    const inode = this.resolve(path);
    if (!inode) return this.resolveErrno();
    return this.chownInode(inode, uid, gid);
  }

  ftruncate(fd: number, length: number): number {
    const ofd = this.openFds.get(fd);
    if (!ofd || ofd.type !== 'file') return -EBADF;
    // Must be open for writing
    if (ofd.accessMode === O_RDONLY) return -EBADF;
    if (length < 0) return -EINVAL;

    const inode = ofd.inode;
    if (length < inode.data.length) {
      inode.data = inode.data.slice(0, length);
    } else if (length > inode.data.length) {
      const newData = new Uint8Array(length); // zero-filled
      newData.set(inode.data);
      inode.data = newData;
    }
    inode.mtime = inode.ctime = this.now();
    return 0;
  }

  fstat(fd: number, statBuf: number, memory: DataView): number {
    const ofd = this.openFds.get(fd);
    if (!ofd) return -EBADF;
    const inode = ofd.type === 'file' ? ofd.inode : ofd.type === 'dir' ? ofd.inode : ofd.type === 'chardev' ? ofd.inode : null;
    if (!inode) return -EBADF;
    this.writeStat(memory, statBuf, inode);
    return 0;
  }

  fstatat(dirfd: number, path: string, statBuf: number, flags: number, memory: DataView): number {
    const AT_SYMLINK_NOFOLLOW = 0x100;
    const AT_EMPTY_PATH = 0x1000;

    if ((flags & AT_EMPTY_PATH) && path === '') {
      return this.fstat(dirfd, statBuf, memory);
    }

    const follow = !(flags & AT_SYMLINK_NOFOLLOW);
    const inode = this.resolve(path, follow);
    if (!inode) return this.resolveErrno();
    this.writeStat(memory, statBuf, inode);
    return 0;
  }

  // posix access-mode permission check. mode bits: R_OK=4, W_OK=2, X_OK=1, F_OK=0.
  // returns true if the effective creds may access the inode in the requested mode.
  // useEffective selects effective vs real uid/gid. access()/faccessat default to the
  // REAL ids; only faccessat with AT_EACCESS uses the effective ids.
  private permitted(inode: InodeBase, mode: number, useEffective: boolean): boolean {
    if (mode === 0) return true; // F_OK: existence, already resolved
    const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
    const uid = useEffective ? c.euid : c.ruid;
    const gid = useEffective ? c.egid : c.rgid;
    if (uid === 0) {
      // root bypasses read/write; for execute it still needs at least one x bit set.
      if (mode & 1) return (inode.mode & 0o111) !== 0;
      return true;
    }
    let perms: number;
    if (inode.uid === uid) perms = (inode.mode >> 6) & 7;
    else if (inode.gid === gid) perms = (inode.mode >> 3) & 7;
    else perms = inode.mode & 7;
    return (perms & mode) === mode;
  }

  faccessat(dirfd: number, path: string, mode: number, flags: number): number {
    const AT_EACCESS = 0x200;
    if ((mode & ~7) !== 0) return -EINVAL;      // mode must be a subset of R_OK|W_OK|X_OK
    if (path === '') return -ENOENT;            // empty path (no AT_EMPTY_PATH for access)
    const useEff = (flags & AT_EACCESS) !== 0; // access() walks + checks with the real uid
    const inode = this.resolve(path, true, 0, useEff);
    if (!inode) return this.resolveErrno();
    return this.permitted(inode, mode & 7, useEff) ? 0 : -EACCES;
  }

  getcwd(): string { return this.cwd; }

  getdents64(fd: number, buf: number, count: number, memory: DataView): number {
    const ofd = this.openFds.get(fd);
    if (!ofd || ofd.type !== 'dir') return -EBADF;

    const bytes = new Uint8Array(memory.buffer);
    const base = buf + GUEST_MEM_BASE;
    let offset = 0;

    while (ofd.pos < ofd.entries.length) {
      const name = ofd.entries[ofd.pos];

      // Determine d_type and d_ino
      let d_type = 4; // DT_DIR for . and ..
      let d_ino = ofd.inode.ino;
      if (name === '.') {
        d_ino = ofd.inode.ino;
      } else if (name === '..') {
        d_ino = ofd.inode.parent ? ofd.inode.parent.ino : ofd.inode.ino;
      } else {
        const child = ofd.inode.children.get(name);
        if (!child) { ofd.pos++; continue; }
        d_ino = child.ino;
        if (child.type === 'file') d_type = 8;      // DT_REG
        else if (child.type === 'dir') d_type = 4;   // DT_DIR
        else if (child.type === 'symlink') d_type = 10; // DT_LNK
        else if (child.type === 'chardev') d_type = 2;  // DT_CHR
      }

      const nameBytes = new TextEncoder().encode(name);
      const reclen = (19 + nameBytes.length + 1 + 7) & ~7; // 8-byte aligned

      if (offset + reclen > count) break; // buffer full

      const pos = base + offset;
      memory.setBigUint64(pos, BigInt(d_ino), true);            // d_ino
      memory.setBigUint64(pos + 8, BigInt(ofd.pos + 1), true);  // d_off (cookie for next call)
      memory.setUint16(pos + 16, reclen, true);                  // d_reclen
      bytes[pos + 18] = d_type;                                   // d_type
      bytes.set(nameBytes, pos + 19);                              // d_name
      bytes[pos + 19 + nameBytes.length] = 0;                     // null terminator
      // Zero padding
      for (let i = 19 + nameBytes.length + 1; i < reclen; i++) bytes[pos + i] = 0;

      offset += reclen;
      ofd.pos++;
    }

    return offset; // 0 = end of directory
  }

  readlinkat(path: string): string | null {
    const inode = this.resolve(path, false); // don't follow final symlink
    if (!inode) return null; // lastResolveErrno holds ENOENT/ENOTDIR/ENAMETOOLONG/ELOOP/EACCES
    if (inode.type !== 'symlink') { this.lastResolveErrno = EINVAL; return null; } // exists, not a link
    return inode.target;
  }

  unlinkat(path: string, flags: number): number {
    const AT_REMOVEDIR = 0x200;
    const parent = this.resolveParent(path);
    if (!parent) return this.resolveErrno();
    const [dir, name] = parent;
    if (name === '.' || name === '..') return -EINVAL;

    const child = dir.children.get(name);
    if (!child) return -ENOENT;
    if (!this.canWrite(dir)) return -EACCES;

    if (flags & AT_REMOVEDIR) {
      if (child.type !== 'dir') return -ENOTDIR;
      if (child.children.size > 0) return -ENOTEMPTY;
      dir.nlink--; // remove .. link
    } else {
      if (child.type === 'dir') return -EISDIR;
      if (child.type === 'file') child.nlink--;
    }

    dir.children.delete(name);
    dir.mtime = dir.ctime = this.now();
    return 0;
  }

  mkdirat(path: string, mode: number): number {
    const parent = this.resolveParent(path);
    if (!parent) return this.resolveErrno();
    const [dir, name] = parent;
    if (dir.children.has(name)) return -EEXIST;
    if (!this.canWrite(dir)) return -EACCES;

    const newDir = this.makeDir(dir, mode & 0o7777);
    this.applyCreatorOwnership(newDir, dir);
    dir.children.set(name, newDir);
    dir.nlink++; // new dir's .. points to parent
    dir.mtime = dir.ctime = this.now();
    return 0;
  }

  symlinkat(target: string, linkpath: string): number {
    const parent = this.resolveParent(linkpath);
    if (!parent) return this.resolveErrno();
    const [dir, name] = parent;
    if (dir.children.has(name)) return -EEXIST;
    if (!this.canWrite(dir)) return -EACCES;
    const link = this.makeSymlink(target);
    this.applyCreatorOwnership(link, dir);
    dir.children.set(name, link);
    dir.mtime = dir.ctime = this.now();
    return 0;
  }

  linkat(oldpath: string, newpath: string): number {
    const inode = this.resolve(oldpath);
    if (!inode) return this.resolveErrno();
    if (inode.type === 'dir') return -EPERM; // can't hard link directories
    const parent = this.resolveParent(newpath);
    if (!parent) return this.resolveErrno();
    const [dir, name] = parent;
    if (dir.children.has(name)) return -EEXIST;
    if (!this.canWrite(dir)) return -EACCES;
    dir.children.set(name, inode); // same inode object = hard link
    inode.nlink++;
    inode.ctime = this.now();
    dir.mtime = dir.ctime = this.now();
    return 0;
  }

  rename(oldpath: string, newpath: string): number {
    const oldParent = this.resolveParent(oldpath);
    if (!oldParent) return this.resolveErrno();
    const [oldDir, oldName] = oldParent;
    const srcNode = oldDir.children.get(oldName);
    if (!srcNode) return -ENOENT;

    const newParent = this.resolveParent(newpath);
    if (!newParent) return this.resolveErrno();
    const [newDir, newName] = newParent;

    const dstNode = newDir.children.get(newName);
    if (dstNode) {
      if (srcNode.type === 'dir' && dstNode.type !== 'dir') return -ENOTDIR;
      if (srcNode.type !== 'dir' && dstNode.type === 'dir') return -EISDIR;
      if (dstNode.type === 'dir' && dstNode.children.size > 0) return -ENOTEMPTY;
    }

    oldDir.children.delete(oldName);
    newDir.children.set(newName, srcNode);
    if (srcNode.type === 'dir') {
      (srcNode as DirInode).parent = newDir;
      if (oldDir !== newDir) { oldDir.nlink--; newDir.nlink++; }
    }
    oldDir.mtime = oldDir.ctime = this.now();
    newDir.mtime = newDir.ctime = this.now();
    return 0;
  }

  /** Set cwd directly (used when chdir target is on native FS, not in VFS) */
  setCwd(path: string): void {
    this.cwd = path;
  }

  chdir(path: string): number {
    if (path.length > PATH_MAX) return -ENAMETOOLONG;
    for (const part of path.split('/')) {
      if (part.length > NAME_MAX) return -ENAMETOOLONG;
    }
    const inode = this.resolve(path);
    if (!inode) return -ENOENT;
    if (inode.type !== 'dir') return -ENOTDIR;
    this.cwd = this.resolvePath(path);
    this.cwdInode = inode;
    return 0;
  }

  /** Copy data between two open files entirely in JS (no guest memory needed). */
  copyFileRange(fdIn: number, fdOut: number, len: number): number {
    const ofdIn = this.openFds.get(fdIn);
    const ofdOut = this.openFds.get(fdOut);
    if (!ofdIn || ofdIn.type !== 'file') return -EBADF;
    if (!ofdOut || ofdOut.type !== 'file') return -EBADF;
    if (ofdIn.accessMode === 1) return -EBADF; // O_WRONLY
    if (ofdOut.accessMode === 0) return -EBADF; // O_RDONLY

    const available = ofdIn.inode.data.length - ofdIn.pos;
    const toCopy = Math.min(len, Math.max(0, available));
    if (toCopy <= 0) return 0;

    // Grow output file if needed
    const endPos = ofdOut.pos + toCopy;
    if (endPos > ofdOut.inode.data.length) {
      const newData = new Uint8Array(endPos);
      newData.set(ofdOut.inode.data);
      ofdOut.inode.data = newData;
    }

    // Copy
    ofdOut.inode.data.set(ofdIn.inode.data.subarray(ofdIn.pos, ofdIn.pos + toCopy), ofdOut.pos);
    ofdIn.pos += toCopy;
    ofdOut.pos = endPos;
    ofdOut.inode.mtime = ofdOut.inode.ctime = this.now();
    return toCopy;
  }

  // ── Snapshot/restore for fork ───────────────────────────────────────────

  /**
   * the open descriptions, as plain data.
   *
   * this crosses a postMessage (fork, and the promotion of a process to its first thread), so it
   * must contain nothing that structured clone refuses. an OFD points at its inode, and a
   * synthesised /proc inode carries the FUNCTION that computes its contents -- copying the object
   * graph therefore threw DataCloneError and killed the worker, which is what any program that
   * held a /proc descriptor across pthread_create saw.
   *
   * what the other side needs is the NAME. its own tree is built the same way, so it re-resolves.
   * dup'd descriptors share one description, and that sharing survives as an index.
   */
  snapshotFds(): { ofds: any[]; fds: Array<[number, number]> } {
    const idx = new Map<OFD, number>();
    const ofds: any[] = [];
    const fds: Array<[number, number]> = [];
    for (const [fd, ofd] of this.openFds) {
      if (!idx.has(ofd)) {
        idx.set(ofd, ofds.length);
        const d: any = { type: ofd.type, path: ofd.path, refcount: ofd.refcount };
        if (ofd.type === 'file') {
          d.pos = ofd.pos; d.flags = ofd.flags; d.accessMode = ofd.accessMode; d.oPath = ofd.oPath;
        } else if (ofd.type === 'dir') {
          d.entries = [...ofd.entries]; d.pos = ofd.pos;
        } else if (ofd.type === 'fifo') {
          d.accessMode = ofd.accessMode;
        }
        ofds.push(d);
      }
      fds.push([fd, idx.get(ofd)!]);
    }
    return { ofds, fds };
  }

  restoreFds(snap: { ofds: any[]; fds: Array<[number, number]> } | undefined): void {
    this.openFds = new Map();
    if (!snap || !Array.isArray(snap.ofds)) return;
    // /proc/self/fd is filled on demand, so a descriptor into it may name something this tree has
    // not built yet. build it once before giving up on a name.
    let synced = false;
    const rebuilt: Array<OFD | null> = snap.ofds.map((d) => {
      let node = this.lookupNode(d.path);
      if (!node && !synced) { synced = true; this.syncSelfFds(); node = this.lookupNode(d.path); }
      if (!node) return null;
      if (d.type === 'file' && node.type === 'file') {
        return { type: 'file', path: d.path, inode: node, pos: d.pos ?? 0, flags: d.flags ?? 0,
                 accessMode: d.accessMode ?? 0, refcount: d.refcount ?? 1, oPath: d.oPath };
      }
      if (d.type === 'dir' && node.type === 'dir') {
        return { type: 'dir', path: d.path, inode: node, entries: [...(d.entries ?? [])],
                 pos: d.pos ?? 0, refcount: d.refcount ?? 1 };
      }
      if (d.type === 'chardev' && node.type === 'chardev') {
        return { type: 'chardev', path: d.path, inode: node, refcount: d.refcount ?? 1 };
      }
      if (d.type === 'fifo' && node.type === 'fifo') {
        return { type: 'fifo', path: d.path, inode: node, accessMode: d.accessMode ?? 0,
                 refcount: d.refcount ?? 1 };
      }
      return null;
    });
    for (const [fd, i] of snap.fds) {
      const ofd = rebuilt[i];
      if (ofd) this.openFds.set(fd, ofd);
    }
  }

  /** the device numbers behind a descriptor open on one of our char devices, else null. */
  fdDevice(fd: number): { major: number; minor: number } | null {
    const ofd = this.openFds.get(fd);
    if (!ofd || ofd.type !== 'chardev') return null;
    return { major: ofd.inode.major, minor: ofd.inode.minor };
  }

  /** what kind of thing a descriptor of ours is open on, or null when it is not one of ours. */
  fdOfdType(fd: number): 'file' | 'dir' | 'chardev' | 'fifo' | null {
    return this.openFds.get(fd)?.type ?? null;
  }

  /** the inode a path names in this tree, of whatever type, or null. */
  private lookupNode(path: string): Inode | null {
    let node: Inode | undefined = this.root;
    for (const part of path.split('/')) {
      if (!part) continue;
      if (!node || node.type !== 'dir') return null;
      node = node.children.get(part);
    }
    return node ?? null;
  }

  closeAllExcept(...keep: number[]): void {
    const keepSet = new Set(keep);
    for (const [fd, ofd] of [...this.openFds.entries()]) {
      if (!keepSet.has(fd)) {
        this.openFds.delete(fd);
        ofd.refcount--;
      }
    }
  }

  // ── Stat writing ────────────────────────────────────────────────────────

  private writeStat(memory: DataView, guestAddr: number, inode: Inode): void {
    const b = guestAddr + GUEST_MEM_BASE;
    // linux packs a device number as major in bits 8-19 with the minor split around it.
    const rdev = inode.type === 'chardev'
      ? ((inode.major << 8) | (inode.minor & 0xff) | ((inode.minor & ~0xff) << 12)) : 0;
    const size = inode.type === 'file' ? inode.data.length
      : inode.type === 'dir' ? 4096
      : inode.type === 'symlink' ? inode.target.length : 0;
    const blocks = Math.ceil(size / 512);
    const at = BigInt(inode.atime), mt = BigInt(inode.mtime), ct = BigInt(inode.ctime);

    // Write all 128 bytes in order (avoids separate zero-fill pass)
    memory.setBigUint64(b + 0, 1n, true);                   // st_dev
    memory.setBigUint64(b + 8, BigInt(inode.ino), true);     // st_ino
    memory.setUint32(b + 16, inode.mode, true);              // st_mode
    memory.setUint32(b + 20, inode.nlink, true);             // st_nlink
    memory.setUint32(b + 24, inode.uid, true);               // st_uid
    memory.setUint32(b + 28, inode.gid, true);               // st_gid
    memory.setBigUint64(b + 32, BigInt(rdev), true);         // st_rdev
    memory.setBigUint64(b + 40, 0n, true);                   // __pad1
    memory.setBigInt64(b + 48, BigInt(size), true);          // st_size
    memory.setInt32(b + 56, 4096, true);                     // st_blksize
    memory.setInt32(b + 60, 0, true);                        // __pad2
    memory.setBigInt64(b + 64, BigInt(blocks), true);        // st_blocks
    memory.setBigInt64(b + 72, at, true);                    // st_atime_sec
    memory.setBigUint64(b + 80, 0n, true);                   // st_atime_nsec
    memory.setBigInt64(b + 88, mt, true);                    // st_mtime_sec
    memory.setBigUint64(b + 96, 0n, true);                   // st_mtime_nsec
    memory.setBigInt64(b + 104, ct, true);                   // st_ctime_sec
    memory.setBigUint64(b + 112, 0n, true);                  // st_ctime_nsec
    memory.setUint32(b + 120, 0, true);                      // __unused4
    memory.setUint32(b + 124, 0, true);                      // __unused5
  }
}
