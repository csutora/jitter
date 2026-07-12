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
import type { VFS } from './syscall.js';

// ── Error constants (Linux errno values) ────────────────────────────────────

const EPERM       = 1;
const ENOENT      = 2;
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
const O_DIRECTORY = 0o200000;
const O_NOFOLLOW  = 0o400000;
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
}

interface DirInode extends InodeBase {
  type: 'dir';
  children: Map<string, Inode>;
  parent: DirInode | null;  // null for root
}

interface CharDevInode extends InodeBase {
  type: 'chardev';
  devno: number;     // device number (1=null, 2=zero, 3=tty, 4=urandom)
}

interface SymlinkInode extends InodeBase {
  type: 'symlink';
  target: string;
}

interface FifoInode extends InodeBase {
  type: 'fifo';
}

type Inode = FileInode | DirInode | CharDevInode | SymlinkInode | FifoInode;

// ── Open File Description (OFD) ────────────────────────────────────────────
// Created by open(), shared between dup'd fds and across fork.

interface FileOFD {
  type: 'file';
  inode: FileInode;
  pos: number;       // shared seek position
  flags: number;     // O_ status flags (O_APPEND, O_NONBLOCK, etc.)
  accessMode: number; // O_RDONLY/O_WRONLY/O_RDWR (immutable after open)
  refcount: number;
}

interface DirOFD {
  type: 'dir';
  inode: DirInode;
  entries: string[]; // snapshot of children at open time, including . and ..
  pos: number;       // index for getdents64
  refcount: number;
}

interface CharDevOFD {
  type: 'chardev';
  inode: CharDevInode;
  refcount: number;
}

interface FifoOFD {
  type: 'fifo';
  inode: FifoInode;
  accessMode: number;
  refcount: number;
}

type OFD = FileOFD | DirOFD | CharDevOFD | FifoOFD;

// ── InMemoryVFS ─────────────────────────────────────────────────────────────

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
  // path-resolution search-permission state. resolve() enforces +x (search) on each directory
  // component; searchDenied records when a walk failed on a permission (EACCES) vs a missing
  // entry (ENOENT). searchEff selects effective vs real creds for the walk (access() uses real
  // by default, most syscalls use effective). set per-resolve.
  private searchDenied = false;
  private searchEff = true;
  // errno for the most recent failed resolve() (ENOENT/ENOTDIR/EACCES/ELOOP/ENAMETOOLONG).
  // callers that get a null inode use resolveErrno() to return the correct negative errno.
  private lastResolveErrno = ENOENT;
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
  private applyCreatorOwnership(inode: InodeBase, parent: DirInode): void {
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

  private makeCharDev(devno: number, mode: number): CharDevInode {
    const t = this.now();
    return { type: 'chardev', ino: this.nextIno++, mode: S_IFCHR | (mode & 0o7777), nlink: 1, uid: 0, gid: 0, atime: t, mtime: t, ctime: t, devno };
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

    this.addCharDev('/dev/null', 1, 0o666);
    this.addCharDev('/dev/zero', 2, 0o666);
    this.addCharDev('/dev/tty', 3, 0o666);
    this.addCharDev('/dev/urandom', 4, 0o444);

    this.addSymlinkNode('/proc/self/exe', '/bin/busybox');

    this.addFileContent('/proc/meminfo',
      'MemTotal:       1048576 kB\nMemFree:         524288 kB\nMemAvailable:    786432 kB\n' +
      'Buffers:          65536 kB\nCached:          131072 kB\nSwapTotal:            0 kB\nSwapFree:             0 kB\n');
    this.addFileContent('/proc/cpuinfo',
      'processor\t: 0\nBogoMIPS\t: 48.00\nFeatures\t: fp asimd aes sha1 sha2 crc32\n' +
      'CPU implementer\t: 0x61\nCPU architecture: 8\nCPU variant\t: 0x1\nCPU part\t: 0x000\n');
    this.addFileContent('/proc/sys/kernel/pid_max', '32768\n');
    this.addFileContent('/proc/sys/fs/pipe-user-pages-soft', '16384\n');
    this.addFileContent('/proc/sys/fs/pipe-max-size', '1048576\n');
    this.addFileContent('/proc/self/status',
      'Name:\tjitter\nUmask:\t0022\nState:\tR (running)\nTgid:\t1\nNgid:\t0\nPid:\t1\nPPid:\t0\n' +
      'TracerPid:\t0\nUid:\t0\t0\t0\t0\nGid:\t0\t0\t0\t0\n' +
      'FDSize:\t64\nGroups:\t0\nVmPeak:\t    65536 kB\nVmSize:\t    65536 kB\nVmRSS:\t    32768 kB\nThreads:\t1\n');
    this.addFileContent('/proc/mounts', 'tmpfs / tmpfs rw,relatime 0 0\nproc /proc proc rw,relatime 0 0\n');
    this.addFileContent('/proc/filesystems', 'nodev\ttmpfs\n');
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

  private addCharDev(path: string, devno: number, mode: number): void {
    const parts = path.split('/').filter(p => p.length > 0);
    const name = parts.pop()!;
    let dir = this.root;
    for (const part of parts) {
      const child = dir.children.get(part);
      if (!child || child.type !== 'dir') return;
      dir = child;
    }
    dir.children.set(name, this.makeCharDev(devno, mode));
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
  private resolve(path: string, followFinal = true, depth = 0, searchEff = true): Inode | null {
    if (depth === 0) { this.searchDenied = false; this.searchEff = searchEff; this.lastResolveErrno = ENOENT; } // fresh walk
    if (depth > MAX_SYMLINK_DEPTH) { this.lastResolveErrno = ELOOP; return null; }
    const parts = this.normalizeParts(path);
    for (const part of parts) {
      if (part.length > NAME_MAX) { this.lastResolveErrno = ENAMETOOLONG; return null; }
    }
    return this.resolveFromParts(parts, followFinal, depth);
  }

  // the errno (negative) for the most recent failed resolve().
  private resolveErrno(): number { return -this.lastResolveErrno; }

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
    const c = this.credsProvider ? this.credsProvider() : { ruid: 0, euid: 0, rgid: 0, egid: 0 };
    if (c.euid === 0) return true;
    let perms: number;
    if (dir.uid === c.euid) perms = (dir.mode >> 6) & 7;
    else if (dir.gid === c.egid) perms = (dir.mode >> 3) & 7;
    else perms = dir.mode & 7;
    return (perms & 2) !== 0;
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
    if (depth === 0) this.lastResolveErrno = ENOENT;
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
        const target = child.target;
        const remaining = parts.slice(idx + 1).join('/');
        const newPath = remaining ? target + '/' + remaining + '/' + name : target + '/' + name;
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
    // Validate path length
    if (path.length > PATH_MAX) return -ENAMETOOLONG;
    for (const part of path.split('/')) {
      if (part.length > NAME_MAX) return -ENAMETOOLONG;
    }

    // Check fd limit
    if (this.openFds.size >= MAX_FDS) return -EMFILE;

    const followSymlinks = !(flags & O_NOFOLLOW);
    let inode = this.resolve(path, followSymlinks);

    // O_NOFOLLOW: if final component is symlink, return ELOOP
    if ((flags & O_NOFOLLOW) && !inode) {
      const raw = this.resolve(path, false);
      if (raw && raw.type === 'symlink') return -ELOOP;
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
    }

    if (!inode) return this.resolveErrno();

    // O_DIRECTORY: must be a directory
    if ((flags & O_DIRECTORY) && inode.type !== 'dir') return -ENOTDIR;

    // Directory handling
    if (inode.type === 'dir') {
      // O_WRONLY or O_RDWR on directory is EISDIR
      if ((flags & O_ACCMODE) !== O_RDONLY) return -EISDIR;
      const fd = this.takeFd();
      // Snapshot entries including . and ..
      const entries = ['.', '..', ...inode.children.keys()];
      this.openFds.set(fd, { type: 'dir', inode, entries, pos: 0, refcount: 1 });
      return fd;
    }

    // Character device
    if (inode.type === 'chardev') {
      const fd = this.takeFd();
      this.openFds.set(fd, { type: 'chardev', inode, refcount: 1 });
      return fd;
    }

    // Named pipe (fifo). we model it as a seekless endpoint: open succeeds (the test opens
    // O_RDWR, which does not block), stat reports S_IFIFO, and lseek returns ESPIPE.
    if (inode.type === 'fifo') {
      const fd = this.takeFd();
      this.openFds.set(fd, { type: 'fifo', inode, accessMode: flags & O_ACCMODE, refcount: 1 });
      return fd;
    }

    // Symlink (only reached if O_NOFOLLOW and it's a symlink — already handled above)
    if (inode.type !== 'file') return -ENOENT;

    // O_TRUNC: truncate to 0 (requires write access)
    if ((flags & O_TRUNC) && ((flags & O_ACCMODE) !== O_RDONLY)) {
      inode.data = new Uint8Array(0);
      inode.mtime = inode.ctime = this.now();
    }

    const fd = this.takeFd();
    const accessMode = flags & O_ACCMODE;
    const pos = (flags & O_APPEND) ? inode.data.length : 0;
    this.openFds.set(fd, { type: 'file', inode, pos, flags, accessMode, refcount: 1 });
    return fd;
  }

  close(fd: number): number {
    const ofd = this.openFds.get(fd);
    if (!ofd) return -EBADF;
    this.openFds.delete(fd);
    ofd.refcount--;
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

    if (ofd.type === 'chardev') {
      const bytes = new Uint8Array(memory.buffer);
      switch (ofd.inode.devno) {
        case 1: return 0; // /dev/null → EOF
        case 2: // /dev/zero → zeros
          bytes.fill(0, buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
          return count;
        case 3: return 0; // /dev/tty → EOF
        case 4: // /dev/urandom → random
          for (let i = 0; i < count; i++)
            bytes[buf + GUEST_MEM_BASE + i] = (Math.random() * 256) | 0;
          return count;
        default: return 0;
      }
    }

    if (ofd.type === 'dir') return -EISDIR;
    if (ofd.type !== 'file') return -EBADF;

    // Check read permission
    if (ofd.accessMode === O_WRONLY) return -EBADF;

    const available = ofd.inode.data.length - ofd.pos;
    const toRead = Math.min(count, Math.max(0, available));
    if (toRead <= 0) return 0; // EOF

    const bytes = new Uint8Array(memory.buffer);
    bytes.set(ofd.inode.data.subarray(ofd.pos, ofd.pos + toRead), buf + GUEST_MEM_BASE);
    ofd.pos += toRead;
    ofd.inode.atime = this.now();
    return toRead;
  }

  write(fd: number, buf: number, count: number, memory: DataView): number {
    const ofd = this.openFds.get(fd);
    if (!ofd) return -EBADF;

    if (ofd.type === 'chardev') return count; // /dev/null etc: discard
    if (ofd.type === 'dir') return -EBADF;
    if (ofd.type !== 'file') return -EBADF;

    // Check write permission
    if (ofd.accessMode === O_RDONLY) return -EBADF;

    if (count === 0) return 0;

    const guestMax = memory.buffer.byteLength - GUEST_MEM_BASE;
    if (buf < 0 || buf >= guestMax || buf + count > guestMax) return -14; // EFAULT

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
      case 2: newPos = ofd.inode.data.length + off; break; // SEEK_END
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
    if (!inode || inode.type !== 'symlink') return null;
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

  snapshotFds(): Map<number, any> {
    // Deep copy OFDs, preserving sharing (dup'd fds share same OFD)
    const ofdMap = new Map<OFD, OFD>();
    const snap = new Map<number, any>();
    for (const [fd, ofd] of this.openFds) {
      if (!ofdMap.has(ofd)) {
        if (ofd.type === 'file') {
          ofdMap.set(ofd, { ...ofd });
        } else if (ofd.type === 'dir') {
          ofdMap.set(ofd, { ...ofd, entries: [...ofd.entries] });
        } else {
          ofdMap.set(ofd, { ...ofd });
        }
      }
      snap.set(fd, ofdMap.get(ofd)!);
    }
    return snap;
  }

  restoreFds(snap: Map<number, any>): void {
    this.openFds = new Map(snap);
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
    const rdev = inode.type === 'chardev' ? inode.devno : 0;
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
