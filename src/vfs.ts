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

type Inode = FileInode | DirInode | CharDevInode | SymlinkInode;

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

type OFD = FileOFD | DirOFD | CharDevOFD;

// ── InMemoryVFS ─────────────────────────────────────────────────────────────

export class InMemoryVFS implements VFS {
  private root: DirInode;
  private openFds = new Map<number, OFD>();
  private nextFd = 3;       // 0=stdin, 1=stdout, 2=stderr
  private nextIno = 1;      // inode counter
  private cwd = '/';
  private cwdInode: DirInode;

  constructor() {
    this.root = this.makeDir(null, 0o755);
    this.cwdInode = this.root;
    this.populateDefaults();
  }

  // ── Inode creation helpers ──────────────────────────────────────────────

  private now(): number { return Math.floor(Date.now() / 1000); }

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

  // ── Default filesystem population ───────────────────────────────────────

  private populateDefaults(): void {
    this.mkdirp('/dev');
    this.mkdirp('/etc');
    this.mkdirp('/proc/self/fd');
    this.mkdirp('/proc/sys/kernel');
    this.mkdirp('/proc/sys/fs');
    this.mkdirp('/tmp');
    this.mkdirp('/root');
    this.mkdirp('/bin');
    this.mkdirp('/home');
    this.mkdirp('/var');

    this.addCharDev('/dev/null', 1, 0o666);
    this.addCharDev('/dev/zero', 2, 0o666);
    this.addCharDev('/dev/tty', 3, 0o666);
    this.addCharDev('/dev/urandom', 4, 0o444);

    this.addFileContent('/etc/passwd',
      'root:x:0:0:root:/root:/bin/sh\n' +
      'daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin\n' +
      'nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n');
    this.addFileContent('/etc/group', 'root:x:0:\ndaemon:x:1:\nnobody:x:65534:\nnogroup:x:65534:\n');
    this.addFileContent('/etc/hostname', 'jitter\n');
    this.addFileContent('/etc/shells', '/bin/sh\n');
    this.addFileContent('/etc/resolv.conf', 'nameserver 1.1.1.1\nnameserver 1.0.0.1\n');
    this.addFileContent('/etc/nsswitch.conf', 'hosts: files dns\n');

    this.addSymlinkNode('/proc/self/exe', '/bin/busybox');

    // Synthetic /proc files
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
  private resolve(path: string, followFinal = true, depth = 0): Inode | null {
    if (depth > MAX_SYMLINK_DEPTH) return null; // ELOOP
    const parts = this.normalizeParts(path);
    return this.resolveFromParts(parts, followFinal, depth);
  }

  private resolveFromParts(parts: string[], followFinal: boolean, depth: number): Inode | null {
    if (parts.length === 0) return this.root;

    let current: Inode = this.root;
    for (let i = 0; i < parts.length; i++) {
      if (current.type !== 'dir') return null;
      const child = current.children.get(parts[i]);
      if (!child) return null;

      const isLast = i === parts.length - 1;
      if (child.type === 'symlink' && (followFinal || !isLast)) {
        // Build remaining path from symlink target + unvisited components
        const target = child.target;
        const remaining = parts.slice(i + 1);
        const targetParts = this.normalizeParts(target);
        if (remaining.length > 0) targetParts.push(...remaining);
        return this.resolveFromParts(targetParts, followFinal, depth + 1);
      }

      current = child;
    }
    return current;
  }

  /** Resolve parent directory of path. Returns [parentDirInode, childName] or null. */
  private resolveParent(path: string, depth = 0): [DirInode, string] | null {
    if (depth > MAX_SYMLINK_DEPTH) return null;

    const resolved = this.resolvePath(path);
    const parts = resolved.split('/').filter(p => p.length > 0);
    if (parts.length === 0) return null; // can't get parent of root
    const name = parts.pop()!;

    let dir = this.root;
    for (const part of parts) {
      const child = dir.children.get(part);
      if (!child) return null;
      if (child.type === 'symlink') {
        const target = child.target;
        const remaining = parts.slice(parts.indexOf(part) + 1).join('/');
        const newPath = remaining ? target + '/' + remaining + '/' + name : target + '/' + name;
        return this.resolveParent(newPath, depth + 1);
      }
      if (child.type !== 'dir') return null;
      dir = child;
    }
    return [dir, name];
  }

  // ── VFS interface implementation ────────────────────────────────────────

  open(path: string, flags: number, mode: number): number {
    // Validate path length
    if (path.length > PATH_MAX) return -ENAMETOOLONG;
    for (const part of path.split('/')) {
      if (part.length > NAME_MAX) return -ENAMETOOLONG;
    }

    // Check fd limit
    if (this.nextFd >= MAX_FDS) return -EMFILE;

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
      if (!parent) return -ENOENT;
      const [dir, name] = parent;
      const newFile = this.makeFile(mode & 0o7777);
      dir.children.set(name, newFile);
      dir.mtime = dir.ctime = this.now();
      inode = newFile;
    }

    if (!inode) return -ENOENT;

    // O_DIRECTORY: must be a directory
    if ((flags & O_DIRECTORY) && inode.type !== 'dir') return -ENOTDIR;

    // Directory handling
    if (inode.type === 'dir') {
      // O_WRONLY or O_RDWR on directory is EISDIR
      if ((flags & O_ACCMODE) !== O_RDONLY) return -EISDIR;
      const fd = this.nextFd++;
      // Snapshot entries including . and ..
      const entries = ['.', '..', ...inode.children.keys()];
      this.openFds.set(fd, { type: 'dir', inode, entries, pos: 0, refcount: 1 });
      return fd;
    }

    // Character device
    if (inode.type === 'chardev') {
      const fd = this.nextFd++;
      this.openFds.set(fd, { type: 'chardev', inode, refcount: 1 });
      return fd;
    }

    // Symlink (only reached if O_NOFOLLOW and it's a symlink — already handled above)
    if (inode.type !== 'file') return -ENOENT;

    // O_TRUNC: truncate to 0 (requires write access)
    if ((flags & O_TRUNC) && ((flags & O_ACCMODE) !== O_RDONLY)) {
      inode.data = new Uint8Array(0);
      inode.mtime = inode.ctime = this.now();
    }

    const fd = this.nextFd++;
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

  dup(fd: number): number {
    const ofd = this.openFds.get(fd);
    if (!ofd) return -EBADF;
    if (this.nextFd >= MAX_FDS) return -EMFILE;
    const newFd = this.nextFd++;
    ofd.refcount++;
    this.openFds.set(newFd, ofd); // same OFD object = shared position
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
    if (ofd.type === 'chardev') return BigInt(-ESPIPE);
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

  fchmod(fd: number, mode: number): number {
    const ofd = this.openFds.get(fd);
    if (!ofd) return -EBADF;
    const inode = ofd.type === 'file' ? ofd.inode : ofd.type === 'dir' ? ofd.inode : null;
    if (!inode) return -EBADF;
    // Preserve file type bits, replace permission bits
    inode.mode = (inode.mode & S_IFMT) | (mode & 0o7777);
    inode.ctime = this.now();
    return 0;
  }

  fchown(fd: number, uid: number, gid: number): number {
    const ofd = this.openFds.get(fd);
    if (!ofd) return -EBADF;
    const inode = ofd.type === 'file' ? ofd.inode : ofd.type === 'dir' ? ofd.inode : null;
    if (!inode) return -EBADF;
    if (uid !== -1 && uid !== 0xFFFFFFFF) inode.uid = uid;
    if (gid !== -1 && gid !== 0xFFFFFFFF) inode.gid = gid;
    inode.ctime = this.now();
    return 0;
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
    if (!inode) return -ENOENT;
    this.writeStat(memory, statBuf, inode);
    return 0;
  }

  faccessat(dirfd: number, path: string, mode: number, flags: number): number {
    const inode = this.resolve(path);
    if (!inode) return -ENOENT;
    return 0; // assume access granted (we're always root)
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
    if (!parent) return -ENOENT;
    const [dir, name] = parent;
    if (name === '.' || name === '..') return -EINVAL;

    const child = dir.children.get(name);
    if (!child) return -ENOENT;

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
    if (!parent) return -ENOENT;
    const [dir, name] = parent;
    if (dir.children.has(name)) return -EEXIST;

    const newDir = this.makeDir(dir, mode & 0o7777);
    dir.children.set(name, newDir);
    dir.nlink++; // new dir's .. points to parent
    dir.mtime = dir.ctime = this.now();
    return 0;
  }

  symlinkat(target: string, linkpath: string): number {
    const parent = this.resolveParent(linkpath);
    if (!parent) return -ENOENT;
    const [dir, name] = parent;
    if (dir.children.has(name)) return -EEXIST;
    dir.children.set(name, this.makeSymlink(target));
    dir.mtime = dir.ctime = this.now();
    return 0;
  }

  linkat(oldpath: string, newpath: string): number {
    const inode = this.resolve(oldpath);
    if (!inode) return -ENOENT;
    if (inode.type === 'dir') return -EPERM; // can't hard link directories
    const parent = this.resolveParent(newpath);
    if (!parent) return -ENOENT;
    const [dir, name] = parent;
    if (dir.children.has(name)) return -EEXIST;
    dir.children.set(name, inode); // same inode object = hard link
    inode.nlink++;
    inode.ctime = this.now();
    dir.mtime = dir.ctime = this.now();
    return 0;
  }

  rename(oldpath: string, newpath: string): number {
    const oldParent = this.resolveParent(oldpath);
    if (!oldParent) return -ENOENT;
    const [oldDir, oldName] = oldParent;
    const srcNode = oldDir.children.get(oldName);
    if (!srcNode) return -ENOENT;

    const newParent = this.resolveParent(newpath);
    if (!newParent) return -ENOENT;
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
    this.nextFd = 3;
    for (const fd of this.openFds.keys()) {
      if (fd >= this.nextFd) this.nextFd = fd + 1;
    }
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
