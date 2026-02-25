import { GUEST_MEM_BASE } from './types.js';
import type { VFS } from './syscall.js';

// ── VFS node types ──────────────────────────────────────────────────────────

interface VFSFile {
  type: 'file';
  data: Uint8Array;
  mode: number; // permission bits
}

interface VFSDir {
  type: 'dir';
  children: Map<string, VFSNode>;
  mode: number;
}

interface VFSCharDev {
  type: 'chardev';
  devno: number;
  mode: number;
}

interface VFSSymlink {
  type: 'symlink';
  target: string;
}

type VFSNode = VFSFile | VFSDir | VFSCharDev | VFSSymlink;

// ── Open file descriptor ────────────────────────────────────────────────────

interface OpenFile {
  type: 'file';
  node: VFSFile;
  pos: number;
  flags: number;
}

interface OpenDir {
  type: 'dir';
  node: VFSDir;
  entries: string[];  // snapshot of child names at open time
  pos: number;        // index into entries (how many have been read)
}

type OpenFd = OpenFile | OpenDir;

// ── Error constants ─────────────────────────────────────────────────────────

const ENOENT = 2;
const EEXIST = 17;
const EBADF  = 9;
const EINVAL = 22;
const EISDIR = 21;
const ENOTDIR = 20;
const ENOTEMPTY = 39;

// O_ flags (aarch64 Linux)
const O_ACCMODE   = 3;
const O_WRONLY    = 1;
const O_RDWR      = 2;
const O_CREAT     = 0o100;
const O_TRUNC     = 0o1000;
const O_APPEND    = 0o2000;
const O_DIRECTORY = 0o200000;  // 0x10000
const O_CLOEXEC   = 0o2000000;

// S_IF constants
const S_IFDIR  = 0o040000;
const S_IFREG  = 0o100000;
const S_IFCHR  = 0o020000;
const S_IFLNK  = 0o120000;

// ── InMemoryVFS ─────────────────────────────────────────────────────────────

export class InMemoryVFS implements VFS {
  private root: VFSDir;
  private openFds = new Map<number, OpenFd>();
  private nextFd = 3; // 0=stdin, 1=stdout, 2=stderr
  private cwd = '/';

  constructor() {
    this.root = { type: 'dir', children: new Map(), mode: 0o755 };
    this.populateDefaults();
  }

  private populateDefaults(): void {
    this.mkdirp('/dev');
    this.mkdirp('/etc');
    this.mkdirp('/proc/self');
    this.mkdirp('/tmp');
    this.mkdirp('/root');
    this.mkdirp('/bin');

    this.addCharDev('/dev/null', 1);
    this.addCharDev('/dev/zero', 5);
    this.addCharDev('/dev/tty', 5);
    this.addCharDev('/dev/urandom', 1);

    this.addFile('/etc/passwd', 'root:x:0:0:root:/root:/bin/sh\n');
    this.addFile('/etc/group', 'root:x:0:\n');
    this.addFile('/etc/hostname', 'jitter\n');
    this.addFile('/etc/shells', '/bin/sh\n');

    this.addSymlink('/proc/self/exe', '/bin/busybox');
  }

  /** Load the busybox binary into /bin/busybox and create applet symlinks */
  loadBinary(data: Uint8Array): void {
    this.setFile('/bin/busybox', data, 0o755);

    // Common busybox applet symlinks
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
    ];
    for (const name of applets) {
      this.addSymlink(`/bin/${name}`, '/bin/busybox');
    }

    // Also populate /usr/bin
    this.mkdirp('/usr/bin');
    for (const name of applets) {
      this.addSymlink(`/usr/bin/${name}`, '/bin/busybox');
    }
  }

  private mkdirp(path: string): void {
    const parts = path.split('/').filter(p => p.length > 0);
    let current = this.root;
    for (const part of parts) {
      let child = current.children.get(part);
      if (!child) {
        child = { type: 'dir', children: new Map(), mode: 0o755 };
        current.children.set(part, child);
      }
      if (child.type !== 'dir') return;
      current = child;
    }
  }

  private addFile(path: string, content: string): void {
    this.setFile(path, new TextEncoder().encode(content), 0o644);
  }

  private setFile(path: string, data: Uint8Array, mode: number): void {
    const parts = path.split('/').filter(p => p.length > 0);
    const name = parts.pop()!;
    let dir = this.root;
    for (const part of parts) {
      const child = dir.children.get(part);
      if (!child || child.type !== 'dir') return;
      dir = child;
    }
    dir.children.set(name, { type: 'file', data, mode });
  }

  private addCharDev(path: string, devno: number): void {
    const parts = path.split('/').filter(p => p.length > 0);
    const name = parts.pop()!;
    let dir = this.root;
    for (const part of parts) {
      const child = dir.children.get(part);
      if (!child || child.type !== 'dir') return;
      dir = child;
    }
    dir.children.set(name, { type: 'chardev', devno, mode: 0o666 });
  }

  private addSymlink(path: string, target: string): void {
    const parts = path.split('/').filter(p => p.length > 0);
    const name = parts.pop()!;
    let dir = this.root;
    for (const part of parts) {
      const child = dir.children.get(part);
      if (!child || child.type !== 'dir') return;
      dir = child;
    }
    dir.children.set(name, { type: 'symlink', target });
  }

  private resolvePath(path: string): string {
    if (!path.startsWith('/')) {
      path = this.cwd + (this.cwd.endsWith('/') ? '' : '/') + path;
    }
    // Normalize: resolve . and ..
    const parts = path.split('/').filter(p => p.length > 0);
    const result: string[] = [];
    for (const part of parts) {
      if (part === '.') continue;
      if (part === '..') { result.pop(); continue; }
      result.push(part);
    }
    return '/' + result.join('/');
  }

  private lookupNode(path: string, followSymlinks = true, depth = 0): VFSNode | null {
    if (depth > 16) return null; // symlink loop protection

    const resolved = this.resolvePath(path);
    if (resolved === '/') return this.root;

    const parts = resolved.split('/').filter(p => p.length > 0);
    let current: VFSNode = this.root;

    for (let i = 0; i < parts.length; i++) {
      if (current.type !== 'dir') return null;
      const child = current.children.get(parts[i]);
      if (!child) return null;

      if (child.type === 'symlink' && (followSymlinks || i < parts.length - 1)) {
        const target = child.target;
        const remaining = parts.slice(i + 1).join('/');
        const newPath = remaining ? target + '/' + remaining : target;
        return this.lookupNode(newPath, followSymlinks, depth + 1);
      }

      current = child;
    }

    return current;
  }

  // ── VFS interface implementation ──

  /** Lookup parent directory and return [parentDir, childName] */
  private lookupParent(path: string): [VFSDir, string] | null {
    const resolved = this.resolvePath(path);
    const parts = resolved.split('/').filter(p => p.length > 0);
    if (parts.length === 0) return null; // can't get parent of root
    const name = parts.pop()!;
    let dir = this.root;
    for (const part of parts) {
      const child = dir.children.get(part);
      if (!child || child.type !== 'dir') return null;
      // Follow symlinks for intermediate components
      if (child.type === 'dir') dir = child;
    }
    return [dir, name];
  }

  open(path: string, flags: number, mode: number): number {
    let node = this.lookupNode(path);

    // Handle O_CREAT: create file if not exists
    if (!node && (flags & O_CREAT)) {
      const parent = this.lookupParent(path);
      if (!parent) return -ENOENT;
      const [dir, name] = parent;
      const newFile: VFSFile = { type: 'file', data: new Uint8Array(0), mode: mode & 0o777 || 0o644 };
      dir.children.set(name, newFile);
      node = newFile;
    }

    if (!node) return -ENOENT;

    if (node.type === 'dir') {
      // Allow opening directories for getdents64
      const fd = this.nextFd++;
      const entries = [...node.children.keys()];
      this.openFds.set(fd, { type: 'dir', node, entries, pos: 0 });
      return fd;
    }
    if (node.type === 'chardev') {
      return this.nextFd++;
    }
    if (node.type !== 'file') return -ENOENT;

    // Reject O_DIRECTORY on non-directories
    if (flags & O_DIRECTORY) return -ENOTDIR;

    // Handle O_TRUNC: truncate to 0
    if (flags & O_TRUNC) {
      node.data = new Uint8Array(0);
    }

    const fd = this.nextFd++;
    const pos = (flags & O_APPEND) ? node.data.length : 0;
    this.openFds.set(fd, { type: 'file', node, pos, flags });
    return fd;
  }

  close(fd: number): number {
    this.openFds.delete(fd);
    return 0;
  }

  read(fd: number, buf: number, count: number, memory: DataView): number {
    const entry = this.openFds.get(fd);
    if (!entry || entry.type !== 'file') return -EBADF;

    const available = entry.node.data.length - entry.pos;
    const toRead = Math.min(count, available);
    if (toRead <= 0) return 0;

    const bytes = new Uint8Array(memory.buffer);
    bytes.set(entry.node.data.subarray(entry.pos, entry.pos + toRead), buf + GUEST_MEM_BASE);
    entry.pos += toRead;
    return toRead;
  }

  write(fd: number, buf: number, count: number, memory: DataView): number {
    const entry = this.openFds.get(fd);
    if (!entry || entry.type !== 'file') return -EBADF;

    if (count === 0) return 0;

    const src = new Uint8Array(memory.buffer, buf + GUEST_MEM_BASE, count);

    // Handle O_APPEND: always write at end
    if (entry.flags & O_APPEND) {
      entry.pos = entry.node.data.length;
    }

    // Grow file if needed
    const endPos = entry.pos + count;
    if (endPos > entry.node.data.length) {
      const newData = new Uint8Array(endPos);
      newData.set(entry.node.data);
      entry.node.data = newData;
    }

    // Copy data
    entry.node.data.set(src, entry.pos);
    entry.pos = endPos;
    return count;
  }

  lseek(fd: number, offset: bigint, whence: number): bigint {
    const entry = this.openFds.get(fd);
    if (!entry || entry.type !== 'file') return BigInt(-EBADF);

    let newPos: number;
    switch (whence) {
      case 0: newPos = Number(offset); break; // SEEK_SET
      case 1: newPos = entry.pos + Number(offset); break; // SEEK_CUR
      case 2: newPos = entry.node.data.length + Number(offset); break; // SEEK_END
      default: return BigInt(-EINVAL);
    }
    if (newPos < 0) return BigInt(-EINVAL);
    entry.pos = newPos;
    return BigInt(newPos);
  }

  fstat(fd: number, statBuf: number, memory: DataView): number {
    const entry = this.openFds.get(fd);
    if (!entry) return -EBADF;
    if (entry.type === 'dir') {
      writeDirStat(memory, statBuf, entry.node.mode);
    } else {
      writeFileStat(memory, statBuf, entry.node.data.length, entry.node.mode);
    }
    return 0;
  }

  fstatat(dirfd: number, path: string, statBuf: number, flags: number, memory: DataView): number {
    const node = this.lookupNode(path);
    if (!node) return -ENOENT;

    if (node.type === 'dir') {
      writeDirStat(memory, statBuf, node.mode);
    } else if (node.type === 'file') {
      writeFileStat(memory, statBuf, node.data.length, node.mode);
    } else if (node.type === 'chardev') {
      writeCharStat(memory, statBuf, node.mode);
    } else {
      return -ENOENT;
    }
    return 0;
  }

  faccessat(dirfd: number, path: string, mode: number, flags: number): number {
    const node = this.lookupNode(path);
    if (!node) return -ENOENT;
    return 0; // File exists, assume access is granted
  }

  getcwd(): string {
    return this.cwd;
  }

  getdents64(fd: number, buf: number, count: number, memory: DataView): number {
    const entry = this.openFds.get(fd);
    if (!entry || entry.type !== 'dir') return -EBADF;

    const bytes = new Uint8Array(memory.buffer);
    const base = buf + GUEST_MEM_BASE;
    let offset = 0;

    while (entry.pos < entry.entries.length) {
      const name = entry.entries[entry.pos];
      const child = entry.node.children.get(name);
      if (!child) { entry.pos++; continue; }

      // d_type values
      let d_type = 0; // DT_UNKNOWN
      if (child.type === 'dir') d_type = 4;       // DT_DIR
      else if (child.type === 'file') d_type = 8;  // DT_REG
      else if (child.type === 'symlink') d_type = 10; // DT_LNK
      else if (child.type === 'chardev') d_type = 2;  // DT_CHR

      // struct linux_dirent64: d_ino(8) + d_off(8) + d_reclen(2) + d_type(1) + d_name(NUL)
      const nameBytes = new TextEncoder().encode(name);
      const reclen = (19 + nameBytes.length + 1 + 7) & ~7; // 8-byte aligned

      if (offset + reclen > count) break; // buffer full

      const pos = base + offset;
      // d_ino — use pos+1 as a simple nonzero inode
      memory.setBigUint64(pos, BigInt(entry.pos + 1), true);
      // d_off — offset of *next* entry
      memory.setBigUint64(pos + 8, BigInt(entry.pos + 1), true);
      // d_reclen
      memory.setUint16(pos + 16, reclen, true);
      // d_type
      bytes[pos + 18] = d_type;
      // d_name (null-terminated)
      bytes.set(nameBytes, pos + 19);
      bytes[pos + 19 + nameBytes.length] = 0;
      // Zero padding
      for (let i = 19 + nameBytes.length + 1; i < reclen; i++) {
        bytes[pos + i] = 0;
      }

      offset += reclen;
      entry.pos++;
    }

    return offset; // total bytes written (0 = end of directory)
  }

  readlinkat(path: string): string | null {
    const node = this.lookupNode(path, false); // don't follow final symlink
    if (!node || node.type !== 'symlink') return null;
    return node.target;
  }

  unlinkat(path: string, flags: number): number {
    const AT_REMOVEDIR = 0x200;
    const parent = this.lookupParent(path);
    if (!parent) return -ENOENT;
    const [dir, name] = parent;

    const child = dir.children.get(name);
    if (!child) return -ENOENT;

    if (flags & AT_REMOVEDIR) {
      if (child.type !== 'dir') return -ENOTDIR;
      if (child.children.size > 0) return -ENOTEMPTY;
    } else {
      if (child.type === 'dir') return -EISDIR;
    }

    dir.children.delete(name);
    return 0;
  }

  mkdirat(path: string, mode: number): number {
    const parent = this.lookupParent(path);
    if (!parent) return -ENOENT;
    const [dir, name] = parent;

    if (dir.children.has(name)) return -EEXIST;
    dir.children.set(name, { type: 'dir', children: new Map(), mode: mode & 0o777 || 0o755 });
    return 0;
  }

  chdir(path: string): number {
    const resolved = this.resolvePath(path);
    const node = this.lookupNode(resolved);
    if (!node) return -ENOENT;
    if (node.type !== 'dir') return -ENOTDIR;
    this.cwd = resolved;
    return 0;
  }
}

// ── stat helpers ────────────────────────────────────────────────────────────

function writeFileStat(memory: DataView, guestAddr: number, size: number, mode: number): void {
  const base = guestAddr + GUEST_MEM_BASE;
  // Zero the struct
  for (let i = 0; i < 128; i += 8) {
    memory.setBigUint64(base + i, 0n, true);
  }
  memory.setUint32(base + 16, S_IFREG | mode, true); // st_mode
  memory.setUint32(base + 20, 1, true); // st_nlink
  memory.setBigUint64(base + 48, BigInt(size), true); // st_size
}

function writeDirStat(memory: DataView, guestAddr: number, mode: number): void {
  const base = guestAddr + GUEST_MEM_BASE;
  for (let i = 0; i < 128; i += 8) {
    memory.setBigUint64(base + i, 0n, true);
  }
  memory.setUint32(base + 16, S_IFDIR | mode, true);
  memory.setUint32(base + 20, 2, true);
}

function writeCharStat(memory: DataView, guestAddr: number, mode: number): void {
  const base = guestAddr + GUEST_MEM_BASE;
  for (let i = 0; i < 128; i += 8) {
    memory.setBigUint64(base + i, 0n, true);
  }
  memory.setUint32(base + 16, S_IFCHR | mode, true);
  memory.setUint32(base + 20, 1, true);
}
