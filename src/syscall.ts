import { CpuState } from './cpu.js';
import { GUEST_MEM_BASE, TPIDR_OFFSET, PC_OFFSET } from './types.js';

// ── Syscall result type ─────────────────────────────────────────────────────

export type SyscallResult =
  | { status: 'ok' }
  | { status: 'exit'; code: number }
  | { status: 'blocked_read'; fd: number }
  | { status: 'fork'; resumePC: bigint }
  | { status: 'execve'; path: string; argv: string[]; envp: string[] };

// ── Error constants ─────────────────────────────────────────────────────────

const ENOSYS = 38;
const EBADF  = 9;
const EINVAL = 22;
const ENOENT = 2;
const EPERM  = 1;
const ENOTDIR = 20;
const ECHILD = 10;

// ── ioctl constants ─────────────────────────────────────────────────────────

const TCGETS     = 0x5401;
const TCSETS     = 0x5402;
const TIOCGWINSZ = 0x5413;
const TIOCSWINSZ = 0x5414;

// ── fcntl constants ─────────────────────────────────────────────────────────

const F_GETFL = 3;
const F_SETFL = 4;
const F_GETFD = 1;
const F_SETFD = 2;

// ── VFS interface ───────────────────────────────────────────────────────────

export interface VFS {
  open(path: string, flags: number, mode: number): number;
  close(fd: number): number;
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
  | { type: 'pipe_write'; pipe: PipeBuffer };

export class SyscallHandler {
  private brk: bigint;
  private mmapBase = 0x2000000n; // mmap region starts at 32MB
  private cwd = '/';
  private tpidr = 0n;
  private termCols = 80;
  private termRows = 24;
  private vfs?: VFS;
  private io: IOInterface;
  private fdFlags = new Map<number, number>(); // fd -> flags
  private fdTable = new Map<number, FdEntry>(); // unified fd table
  private nextFd = 3; // next available fd (0-2 are stdio)
  lastWaitStatus?: number;  // exit code from last child (set by dispatch loop)
  lastWaitPid?: number;     // PID of last exited child

  constructor(
    brkStart: bigint,
    io: IOInterface,
    vfs?: VFS,
  ) {
    this.brk = brkStart;
    this.io = io;
    this.vfs = vfs;
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
      nextFd: this.nextFd,
    };
  }

  /** Restore state from process snapshot */
  restoreState(state: { brk: bigint; mmapBase: bigint; cwd: string; fdFlags: Map<number, number>; fdTable?: Map<number, FdEntry>; nextFd?: number }): void {
    this.brk = state.brk;
    this.mmapBase = state.mmapBase;
    if (this.vfs) {
      this.vfs.chdir(state.cwd);
    } else {
      this.cwd = state.cwd;
    }
    this.fdFlags = new Map(state.fdFlags);
    if (state.fdTable) this.fdTable = new Map(state.fdTable);
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
          this.fdTable.set(newfd, { ...oldEntry });
          ret = BigInt(newfd);
        } else if (oldfd <= 2) {
          // Dup stdio
          const newfd = this.allocFd();
          this.fdTable.set(newfd, { type: 'stdio', fd: oldfd });
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
          this.fdTable.delete(newfd);
        }
        // Duplicate oldfd to newfd
        const srcEntry = this.fdTable.get(oldfd);
        if (srcEntry) {
          this.fdTable.set(newfd, { ...srcEntry });
          ret = BigInt(newfd);
        } else if (oldfd <= 2) {
          this.fdTable.set(newfd, { type: 'stdio', fd: oldfd });
          ret = BigInt(newfd);
        } else {
          ret = -BigInt(EBADF);
        }
        break;
      }

      // ── File operations ──
      case 56: { // openat(dirfd, path, flags, mode)
        const path = readGuestString(bytes, Number(a1));
        if (this.vfs) {
          ret = BigInt(this.vfs.open(path, Number(a2), Number(a3)));
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

      case 62: { // lseek(fd, offset, whence)
        if (this.vfs) {
          ret = this.vfs.lseek(Number(a0), a1, Number(a2));
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
        if (this.vfs) {
          ret = BigInt(this.vfs.fstatat(Number(a0), path, Number(a2), Number(a3), view));
        } else {
          ret = -BigInt(ENOENT);
        }
        break;
      }

      case 61: { // getdents64(fd, dirp, count)
        const fd = Number(a0);
        const dirp = Number(a1);
        const count = Number(a2);

        if (this.vfs) {
          ret = BigInt(this.vfs.getdents64(fd, dirp, count, view));
        } else {
          ret = -BigInt(EBADF);
        }
        break;
      }

      case 34: { // mkdirat(dirfd, path, mode)
        const path = readGuestString(bytes, Number(a1));
        if (this.vfs) {
          ret = BigInt(this.vfs.mkdirat(path, Number(a2)));
        } else {
          ret = -BigInt(ENOENT);
        }
        break;
      }

      case 35: { // unlinkat(dirfd, path, flags)
        const path = readGuestString(bytes, Number(a1));
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
        if (this.vfs) {
          ret = BigInt(this.vfs.faccessat(Number(a0), path, Number(a2), Number(a3)));
        } else {
          ret = -BigInt(ENOENT);
        }
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
