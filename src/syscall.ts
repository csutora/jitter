import { CpuState } from './cpu.js';
import { GUEST_MEM_BASE, TPIDR_OFFSET, PC_OFFSET } from './types.js';
import type { NativeBridge } from './bridge.js';

// ── Syscall result type ─────────────────────────────────────────────────────

export type SyscallResult =
  | { status: 'ok' }
  | { status: 'exit'; code: number }
  | { status: 'blocked_read'; fd: number }
  | { status: 'fork'; resumePC: bigint }
  | { status: 'execve'; path: string; argv: string[]; envp: string[] }
  | { status: 'bridge'; method: string; args: Record<string, any>; apply: (resp: any) => void };

// ── Error constants ─────────────────────────────────────────────────────────

const ENOSYS = 38;
const EBADF  = 9;
const EINVAL = 22;
const ENOENT = 2;
const EPERM  = 1;
const ENOTDIR = 20;
const ECHILD = 10;
const ESPIPE = 29;

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
  read(fd: number, buf: number, count: number, memory: DataView): number;
  write(fd: number, buf: number, count: number, memory: DataView): number;
  lseek(fd: number, offset: bigint, whence: number): bigint;
  fstat(fd: number, statBuf: number, memory: DataView): number;
  fstatat(dirfd: number, path: string, statBuf: number, flags: number, memory: DataView): number;
  faccessat(dirfd: number, path: string, mode: number, flags: number): number;
  getcwd(): string;
  chdir(path: string): number;
  getdents64(fd: number, buf: number, count: number, memory: DataView): number;
  readlinkat(path: string): string | null;
  unlinkat(path: string, flags: number): number;
  mkdirat(path: string, mode: number): number;
}

// ── I/O interface for terminal ──────────────────────────────────────────────

export interface IOInterface {
  writeStdout(data: Uint8Array): void;
  writeStderr(data: Uint8Array): void;
  readStdin(count: number): Uint8Array | null;
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
  | { type: 'vfs'; fd: number }    // VFS file descriptor
  | { type: 'pipe_read'; pipe: PipeBuffer }
  | { type: 'pipe_write'; pipe: PipeBuffer }
  | { type: 'native'; nativeFd: number; path: string };  // Native Swift fd (via bridge)

export class SyscallHandler {
  private brk: bigint;
  private mmapBase = 0x2000000n; // mmap region starts at 32MB
  private cwd = '/';
  private tpidr = 0n;
  private termCols = 80;
  private termRows = 24;
  private vfs?: VFS;
  private io: IOInterface;
  private bridge?: NativeBridge;
  private fdFlags = new Map<number, number>(); // fd -> flags
  private fdTable = new Map<number, FdEntry>(); // unified fd table
  private nativeRefs = new Map<number, number>(); // nativeFd -> refcount
  private nextFd = 3; // next available fd (0-2 are stdio)
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
  }

  /**
   * Check if a guest path should be routed to the native Swift filesystem.
   * Returns the resolved absolute path if bridging, or null if staying in JS.
   *
   * JS VFS handles: /proc, /dev, /tmp (virtual/ephemeral), /bin, /etc, /usr (busybox + config)
   * Native Swift handles: /root, /home, /var (persistent user storage)
   */
  private bridgePath(path: string): string | null {
    if (!this.bridge?.isNative) return null;
    // Resolve relative paths using cwd
    let resolved = path;
    if (!path.startsWith('/')) {
      const cwd = this.vfs?.getcwd() ?? this.cwd;
      resolved = cwd === '/' ? '/' + path : cwd + '/' + path;
    }
    // Only route known persistent-storage paths to native
    if (resolved.startsWith('/root') || resolved.startsWith('/home') || resolved.startsWith('/var')) {
      return resolved;
    }
    return null;
  }

  setTermSize(cols: number, rows: number): void {
    this.termCols = cols;
    this.termRows = rows;
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
      nextFd: this.nextFd,
    };
  }

  /** Restore state from process snapshot */
  restoreState(state: { brk: bigint; mmapBase: bigint; cwd: string; fdFlags: Map<number, number>; fdTable?: Map<number, FdEntry>; nativeRefs?: Map<number, number>; nextFd?: number }): void {
    this.brk = state.brk;
    this.mmapBase = state.mmapBase;
    if (this.vfs) {
      this.vfs.chdir(state.cwd);
    } else {
      this.cwd = state.cwd;
    }
    this.fdFlags = new Map(state.fdFlags);
    if (state.fdTable) this.fdTable = new Map(state.fdTable);
    if (state.nativeRefs) this.nativeRefs = new Map(state.nativeRefs);
    if (state.nextFd !== undefined) this.nextFd = state.nextFd;
  }

  /** Reset state for execve (new process image) */
  resetForExec(brkStart: bigint): void {
    this.brk = brkStart;
    this.mmapBase = 0x2000000n;
    // Preserve stdio redirections (fds 0, 1, 2) — real Linux closes CLOEXEC fds,
    // we close all fds >= 3 (pipe ends the child set up before exec)
    const preserved = new Map<number, FdEntry>();
    for (const [fd, entry] of this.fdTable) {
      if (fd <= 2) preserved.set(fd, entry);
    }
    this.fdTable = preserved;
    this.nextFd = 3;
  }

  /** Allocate next available fd */
  private allocFd(): number {
    return this.nextFd++;
  }

  /** Resolve fd: check fdTable for overrides (dup2'd pipes, etc.), else fall through to default */
  private getFdEntry(fd: number): FdEntry | undefined {
    return this.fdTable.get(fd);
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

        return { status: 'execve', path: pathname, argv: argvPtrs, envp: envpPtrs };
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
            // Write wstatus: exit code << 8 (WEXITSTATUS format)
            view.setUint32(wstatus + GUEST_MEM_BASE, (exitCode & 0xFF) << 8, true);
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
        if (readEntry) {
          if (readEntry.type === 'pipe_read') {
            const pipe = readEntry.pipe;
            if (pipe.data.length === 0) {
              if (pipe.writeClosed) {
                ret = 0n; // EOF
              } else {
                // Would block — in our sequential model, write side hasn't finished
                ret = 0n; // Return EOF for now (sequential pipe)
              }
            } else {
              // Read from pipe buffer
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
          if (readEntry.type === 'stdio' && readEntry.fd === 0) {
            const data = this.io.readStdin(count);
            if (data === null) return { status: 'blocked_read', fd: 0 };
            bytes.set(data, buf + GUEST_MEM_BASE);
            ret = BigInt(data.length);
            break;
          }
          if (readEntry.type === 'vfs') {
            ret = this.vfs ? BigInt(this.vfs.read(readEntry.fd, buf, count, view)) : -BigInt(EBADF);
            break;
          }
          if (readEntry.type === 'native') {
            return {
              status: 'bridge', method: 'fs_read',
              args: { fd: readEntry.nativeFd, count },
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
          ret = -BigInt(EBADF);
          break;
        }

        // Default behavior for non-overridden fds
        if (fd === 0) {
          const data = this.io.readStdin(count);
          if (data === null) return { status: 'blocked_read', fd: 0 };
          bytes.set(data, buf + GUEST_MEM_BASE);
          ret = BigInt(data.length);
        } else if (this.vfs) {
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

        // Check fd table for overrides
        const writeEntry = this.getFdEntry(fd);
        if (writeEntry) {
          if (writeEntry.type === 'pipe_write') {
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
          if (writeEntry.type === 'vfs') {
            ret = this.vfs ? BigInt(this.vfs.write(writeEntry.fd, buf, count, view)) : -BigInt(EBADF);
            break;
          }
          if (writeEntry.type === 'native') {
            const writeData = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
            const b64 = btoa(String.fromCharCode(...writeData));
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
          ret = -BigInt(EBADF);
          break;
        }

        // Default behavior
        if (fd === 1) {
          const data = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
          this.io.writeStdout(data);
          ret = BigInt(count);
        } else if (fd === 2) {
          const data = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
          this.io.writeStderr(data);
          ret = BigInt(count);
        } else if (this.vfs) {
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

        // Native fd: gather all iov data and send as one bridge write
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
          const b64 = btoa(String.fromCharCode(...combined));
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
            } else if (writevEntry.type === 'vfs' && this.vfs) {
              this.vfs.write(writevEntry.fd, base, len, view);
            }
          } else if (fd === 1) {
            this.io.writeStdout(data);
          } else if (fd === 2) {
            this.io.writeStderr(data);
          } else if (this.vfs) {
            this.vfs.write(fd, base, len, view);
          }
          totalWritten += len;
        }
        ret = BigInt(totalWritten);
        break;
      }

      // ── Pipes & fd duplication ──
      case 59: { // pipe2(pipefd, flags)
        const pipefdAddr = Number(a0);
        const pipe: PipeBuffer = { data: [], writeClosed: false, readClosed: false };
        const readFd = this.allocFd();
        const writeFd = this.allocFd();
        this.fdTable.set(readFd, { type: 'pipe_read', pipe });
        this.fdTable.set(writeFd, { type: 'pipe_write', pipe });
        // Write [readFd, writeFd] to guest memory
        view.setInt32(pipefdAddr + GUEST_MEM_BASE, readFd, true);
        view.setInt32(pipefdAddr + GUEST_MEM_BASE + 4, writeFd, true);
        ret = 0n;
        break;
      }

      case 23: { // dup(oldfd)
        const oldfd = Number(a0);
        const oldEntry = this.fdTable.get(oldfd);
        if (oldEntry) {
          const newfd = this.allocFd();
          if (oldEntry.type === 'vfs' && this.vfs) {
            // Duplicate the underlying VFS fd so it has its own lifetime
            const vfsFd = this.vfs.dup(oldEntry.fd);
            if (vfsFd < 0) { ret = BigInt(vfsFd); break; }
            this.fdTable.set(newfd, { type: 'vfs', fd: vfsFd });
          } else if (oldEntry.type === 'native') {
            this.fdTable.set(newfd, { ...oldEntry });
            this.nativeRefs.set(oldEntry.nativeFd, (this.nativeRefs.get(oldEntry.nativeFd) ?? 1) + 1);
          } else {
            this.fdTable.set(newfd, { ...oldEntry });
          }
          ret = BigInt(newfd);
        } else if (oldfd <= 2) {
          const newfd = this.allocFd();
          this.fdTable.set(newfd, { type: 'stdio', fd: oldfd });
          ret = BigInt(newfd);
        } else if (this.vfs) {
          // oldfd is a VFS fd not in our table — dup it in VFS
          const vfsFd = this.vfs.dup(oldfd);
          if (vfsFd < 0) { ret = BigInt(vfsFd); break; }
          const newfd = this.allocFd();
          this.fdTable.set(newfd, { type: 'vfs', fd: vfsFd });
          ret = BigInt(newfd);
        } else {
          ret = -BigInt(EBADF);
        }
        break;
      }

      case 24: { // dup3(oldfd, newfd, flags)
        const oldfd = Number(a0);
        const newfd = Number(a1);
        // Close newfd if it's open
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
        }
        // Duplicate oldfd to newfd
        const srcEntry = this.fdTable.get(oldfd);
        if (srcEntry) {
          if (srcEntry.type === 'vfs' && this.vfs) {
            const vfsFd = this.vfs.dup(srcEntry.fd);
            if (vfsFd < 0) { ret = BigInt(vfsFd); break; }
            this.fdTable.set(newfd, { type: 'vfs', fd: vfsFd });
          } else if (srcEntry.type === 'native') {
            this.fdTable.set(newfd, { ...srcEntry });
            this.nativeRefs.set(srcEntry.nativeFd, (this.nativeRefs.get(srcEntry.nativeFd) ?? 1) + 1);
          } else {
            this.fdTable.set(newfd, { ...srcEntry });
          }
          ret = BigInt(newfd);
        } else if (oldfd <= 2) {
          this.fdTable.set(newfd, { type: 'stdio', fd: oldfd });
          ret = BigInt(newfd);
        } else if (this.vfs) {
          // oldfd is a VFS fd not in our table — dup it in VFS
          const vfsFd = this.vfs.dup(oldfd);
          if (vfsFd < 0) { ret = BigInt(vfsFd); break; }
          this.fdTable.set(newfd, { type: 'vfs', fd: vfsFd });
          ret = BigInt(newfd);
        } else {
          ret = -BigInt(EBADF);
        }
        break;
      }

      // ── File operations ──
      case 56: { // openat(dirfd, path, flags, mode)
        const path = readGuestString(bytes, Number(a1));
        const flags56 = Number(a2);
        const mode56 = Number(a3);
        const bp56 = this.bridgePath(path);
        if (bp56 !== null) {
          const guestFd = this.allocFd();
          return {
            status: 'bridge', method: 'fs_open',
            args: { path: bp56, flags: flags56, mode: mode56 },
            apply: (resp: any) => {
              if (resp.ret < 0) {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                this.nextFd--; // reclaim
              } else {
                this.fdTable.set(guestFd, { type: 'native', nativeFd: resp.ret, path: bp56 });
                this.nativeRefs.set(resp.ret, 1);
                cpu.setReg(0, BigInt(guestFd));
              }
            },
          };
        }
        if (this.vfs) {
          ret = BigInt(this.vfs.open(path, flags56, mode56));
        } else {
          ret = -BigInt(ENOENT);
        }
        break;
      }

      case 57: { // close(fd)
        const fd = Number(a0);
        const closeEntry = this.fdTable.get(fd);
        if (closeEntry) {
          if (closeEntry.type === 'pipe_write') closeEntry.pipe.writeClosed = true;
          if (closeEntry.type === 'pipe_read') closeEntry.pipe.readClosed = true;
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
          this.fdTable.delete(fd);
          ret = 0n;
        } else if (fd <= 2) {
          ret = 0n; // don't close stdio
        } else if (this.vfs) {
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
        // Stdio and pipes are not seekable — return ESPIPE
        if (lseekFd <= 2 && !lseekEntry) {
          ret = -BigInt(ESPIPE);
          break;
        }
        if (lseekEntry) {
          if (lseekEntry.type === 'stdio' || lseekEntry.type === 'pipe_read' || lseekEntry.type === 'pipe_write') {
            ret = -BigInt(ESPIPE);
            break;
          }
          if (lseekEntry.type === 'native') {
            return {
              status: 'bridge', method: 'fs_lseek',
              args: { fd: lseekEntry.nativeFd, offset: Number(a1), whence: Number(a2) },
              apply: (resp: any) => {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              },
            };
          }
          if (lseekEntry.type === 'vfs') {
            ret = this.vfs ? this.vfs.lseek(lseekEntry.fd, a1, Number(a2)) : -BigInt(EBADF);
            break;
          }
        }
        if (this.vfs) {
          ret = this.vfs.lseek(lseekFd, a1, Number(a2));
        } else {
          ret = -BigInt(EBADF);
        }
        break;
      }

      // ── stat/fstat ──
      case 80: { // fstat(fd, statbuf)
        const fd = Number(a0);
        const statBuf = Number(a1);

        // Check fd table first
        const fstatEntry = this.getFdEntry(fd);
        if (fstatEntry) {
          if (fstatEntry.type === 'stdio' || fstatEntry.type === 'pipe_read' || fstatEntry.type === 'pipe_write') {
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
        } else if (fd <= 2) {
          writeChardevStat(view, statBuf);
          ret = 0n;
        } else if (this.vfs) {
          ret = BigInt(this.vfs.fstat(fd, statBuf, view));
        } else {
          ret = -BigInt(EBADF);
        }
        break;
      }

      case 79: { // fstatat(dirfd, path, statbuf, flags)
        const path = readGuestString(bytes, Number(a1));
        const statBuf79 = Number(a2);
        const bp79 = this.bridgePath(path);
        if (bp79 !== null) {
          return {
            status: 'bridge', method: 'fs_stat',
            args: { path: bp79 },
            apply: (resp: any) => {
              if (resp.ret < 0) {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
              } else {
                writeBridgeStat(view, statBuf79, resp);
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
              const entries: { name: string; type: number }[] = resp.entries ?? [];
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
        const path = readGuestString(bytes, Number(a1));
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
          ret = BigInt(this.vfs.mkdirat(path, Number(a2)));
        } else {
          ret = -BigInt(ENOENT);
        }
        break;
      }

      case 35: { // unlinkat(dirfd, path, flags)
        const path = readGuestString(bytes, Number(a1));
        const bp35 = this.bridgePath(path);
        if (bp35 !== null) {
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

      case 78: { // readlinkat(dirfd, path, buf, bufsiz)
        const path = readGuestString(bytes, Number(a1));
        const buf = Number(a2);
        const bufsiz = Number(a3);
        if (this.vfs) {
          const target = this.vfs.readlinkat(path);
          if (target === null) {
            ret = -BigInt(EINVAL);
          } else {
            const encoded = new TextEncoder().encode(target);
            const len = Math.min(encoded.length, bufsiz);
            bytes.set(encoded.subarray(0, len), buf + GUEST_MEM_BASE);
            ret = BigInt(len);
          }
        } else {
          ret = -BigInt(ENOENT);
        }
        break;
      }

      case 48: { // faccessat(dirfd, path, mode, flags)
        const path = readGuestString(bytes, Number(a1));
        const bp48 = this.bridgePath(path);
        if (bp48 !== null) {
          // Use stat to check existence/access on the native side
          return {
            status: 'bridge', method: 'fs_stat',
            args: { path: bp48 },
            apply: (resp: any) => {
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret < 0 ? resp.ret : 0)));
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

      // ── chmod/chown (stubs — we run as root, permissions are a no-op) ──
      case 53:   // fchmod(fd, mode)
      case 54: { // fchmodat(dirfd, path, mode, flags)
        ret = 0n;
        break;
      }
      case 55: { // fchownat(dirfd, path, owner, group, flags)
        ret = 0n;
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
        const request = Number(a1);

        // Resolve fd through fd table — check if this is a dup'd stdio fd
        const ioctlEntry = this.getFdEntry(fd);
        const isStdio = fd <= 2 || (ioctlEntry?.type === 'stdio');
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
        const flags = Number(a3);
        const aligned = (length + 0xFFF) & ~0xFFF; // page-align
        const MAP_FIXED = 0x10;

        // Honor requested address if nonzero (MAP_FIXED or hint)
        let addr: bigint;
        if (a0 !== 0n) {
          addr = a0;
          // Update mmapBase if this extends past it
          const end = a0 + BigInt(aligned);
          if (end > this.mmapBase) {
            this.mmapBase = end;
          }
        } else {
          addr = this.mmapBase;
          this.mmapBase += BigInt(aligned);
        }

        // Ensure memory
        const needed = Number(addr) + aligned + GUEST_MEM_BASE;
        if (needed > memory.buffer.byteLength) {
          const pagesNeeded = Math.ceil(needed / 65536);
          const currentPages = memory.buffer.byteLength / 65536;
          if (pagesNeeded > currentPages) {
            memory.grow(pagesNeeded - currentPages);
          }
        }

        // Zero-fill the mapped region
        new Uint8Array(memory.buffer).fill(0, Number(addr) + GUEST_MEM_BASE, Number(addr) + GUEST_MEM_BASE + aligned);

        ret = addr;
        break;
      }

      case 226: // mprotect
        ret = 0n;
        break;

      case 215: // munmap
        ret = 0n;
        break;

      // ── Thread/process ──
      case 96: { // set_tid_address(tidptr)
        ret = 1n; // PID = 1
        break;
      }

      case 99: // set_robust_list
        ret = 0n;
        break;

      case 178: // gettid
        ret = 1n;
        break;

      case 172: // getpid
        ret = 1n;
        break;

      case 173: // getppid
        ret = 0n;
        break;

      case 174: // getuid
      case 175: // geteuid
      case 176: // getgid
      case 177: // getegid
        ret = 0n; // root
        break;

      // ── Signals ──
      case 134: // rt_sigaction
        ret = 0n;
        break;

      case 135: // rt_sigprocmask
        ret = 0n;
        break;

      case 132: // sigaltstack
        ret = 0n;
        break;

      // ── Filesystem info ──
      case 17: { // getcwd(buf, size)
        const buf = Number(a0);
        const size = Number(a1);
        const cwd = this.vfs ? this.vfs.getcwd() : this.cwd;
        const encoded = new TextEncoder().encode(cwd + '\0');
        if (encoded.length > size) {
          ret = -BigInt(EINVAL);
        } else {
          new Uint8Array(memory.buffer).set(encoded, buf + GUEST_MEM_BASE);
          ret = a0;
        }
        break;
      }

      case 49: { // chdir(path)
        const path = readGuestString(bytes, Number(a0));
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
        const now = Date.now();
        const sec = BigInt(Math.floor(now / 1000));
        const nsec = BigInt((now % 1000) * 1000000);
        view.setBigUint64(tp, sec, true);
        view.setBigUint64(tp + 8, nsec, true);
        ret = 0n;
        break;
      }

      case 169: { // gettimeofday(tv, tz)
        const tv = Number(a0) + GUEST_MEM_BASE;
        const now = Date.now();
        const sec = BigInt(Math.floor(now / 1000));
        const usec = BigInt((now % 1000) * 1000);
        view.setBigUint64(tv, sec, true);
        view.setBigUint64(tv + 8, usec, true);
        ret = 0n;
        break;
      }

      // ── fcntl ──
      case 25: { // fcntl(fd, cmd, ...)
        const fd = Number(a0);
        const cmd = Number(a1);
        switch (cmd) {
          case F_DUPFD:
          case F_DUPFD_CLOEXEC: {
            const minfd = Number(a2);
            // Find lowest available fd >= minfd
            let newfd = minfd;
            while (this.fdTable.has(newfd)) newfd++;
            // Duplicate the source fd entry
            const srcEntry = this.fdTable.get(fd);
            if (srcEntry) {
              if (srcEntry.type === 'vfs' && this.vfs) {
                const vfsFd = this.vfs.dup(srcEntry.fd);
                if (vfsFd < 0) { ret = BigInt(vfsFd); break; }
                this.fdTable.set(newfd, { type: 'vfs', fd: vfsFd });
              } else if (srcEntry.type === 'native') {
                this.fdTable.set(newfd, { ...srcEntry });
                this.nativeRefs.set(srcEntry.nativeFd, (this.nativeRefs.get(srcEntry.nativeFd) ?? 1) + 1);
              } else {
                this.fdTable.set(newfd, { ...srcEntry });
              }
            } else if (fd <= 2) {
              this.fdTable.set(newfd, { type: 'stdio', fd });
            } else if (this.vfs) {
              // fd is a VFS fd not in our table — dup it in VFS
              const vfsFd = this.vfs.dup(fd);
              if (vfsFd < 0) { ret = BigInt(vfsFd); break; }
              this.fdTable.set(newfd, { type: 'vfs', fd: vfsFd });
            } else {
              ret = -BigInt(EBADF);
              break;
            }
            // Keep nextFd above any manually allocated fd
            if (newfd >= this.nextFd) this.nextFd = newfd + 1;
            ret = BigInt(newfd);
            break;
          }
          case F_GETFL:
            ret = BigInt(this.fdFlags.get(fd) ?? 0);
            break;
          case F_SETFL:
            this.fdFlags.set(fd, Number(a2));
            ret = 0n;
            break;
          case F_GETFD:
            ret = 0n;
            break;
          case F_SETFD:
            ret = 0n;
            break;
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

      case 98: // futex — stub
        ret = 0n;
        break;

      case 131: // tgkill
        ret = 0n;
        break;

      case 103: // sched_setscheduler — stub
      case 122: // sched_setaffinity — stub
      case 123: // sched_getaffinity — stub
        ret = 0n;
        break;

      case 115: // sched_getscheduler
        ret = 0n; // SCHED_OTHER
        break;

      case 227: { // clock_nanosleep
        // Just return immediately (no actual sleeping in emulator)
        ret = 0n;
        break;
      }

      case 261: // prlimit64 — stub, return 0
        ret = 0n;
        break;

      case 233: // madvise
        ret = 0n;
        break;

      case 73: { // ppoll(fds, nfds, tmo_p, sigmask)
        // struct pollfd { int fd; short events; short revents; }  (8 bytes each)
        const fdsAddr = Number(a0);
        const nfds = Number(a1);

        for (let i = 0; i < nfds; i++) {
          const pfd = fdsAddr + i * 8 + GUEST_MEM_BASE;
          const events = view.getInt16(pfd + 4, true);
          // Report all fds as ready — actual blocking happens in read()
          view.setInt16(pfd + 6, events, true);
        }
        ret = BigInt(nfds);
        break;
      }

      default:
        console.warn(`[jitter] unimplemented syscall ${nr} (X0=${a0}, X1=${a1}, X2=${a2})`);
        ret = -BigInt(ENOSYS);
    }

    // Set return value in X0
    cpu.setReg(0, BigInt.asUintN(64, ret));
    return { status: 'ok' };
  }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

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
  view.setUint32(base + 16, resp.mode ?? 0o100644, true);
  view.setUint32(base + 20, resp.nlink ?? 1, true);
  view.setBigUint64(base + 48, BigInt(resp.size ?? 0), true);
  view.setBigUint64(base + 88, BigInt(resp.mtime ?? 0), true);
}
