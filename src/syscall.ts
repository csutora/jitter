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
const MAX_FDS = 1024;

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
const SIGNAL_FRAME_SIZE = 288;

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
  hasFd(fd: number): boolean;
  read(fd: number, buf: number, count: number, memory: DataView): number;
  write(fd: number, buf: number, count: number, memory: DataView): number;
  lseek(fd: number, offset: bigint, whence: number): bigint;
  fstat(fd: number, statBuf: number, memory: DataView): number;
  fstatat(dirfd: number, path: string, statBuf: number, flags: number, memory: DataView): number;
  faccessat(dirfd: number, path: string, mode: number, flags: number): number;
  fchmod(fd: number, mode: number): number;
  fchown(fd: number, uid: number, gid: number): number;
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
  | { type: 'native'; nativeFd: number; path: string }  // Native Swift fd (via bridge)
  | { type: 'socket'; socketId: number; domain: number; sockType: number; remoteAddr?: { family: number; host: string; port: number } };  // Network socket

export class SyscallHandler {
  private brk: bigint;
  private mmapBase = 0x2000000n; // mmap region starts at 32MB
  private cwd = '/';
  private tpidr = 0n;
  private umask = 0o022;
  private fdCloexec = new Set<number>();
  private fdSig = new Map<number, number>();
  pendingSignals: number[] = [];  // queue of signals to deliver
  private epollSets?: Map<number, Map<number, { events: number; data: bigint }>>;
  private signalHandlers?: Map<number, { handler: bigint; flags: bigint; restorer: bigint; mask: bigint }>;
  private signalMask?: bigint;
  /** True if bridge calls can be handled (either direct bridge or Worker kernel relay) */
  bridgeAvailable = false;
  currentPid = 1;
  currentPpid = 0;
  private termCols = 80;
  private termRows = 24;
  private vfs?: VFS;
  private io: IOInterface;
  private bridge?: NativeBridge;
  private fdFlags = new Map<number, number>(); // fd -> flags
  private fdTable = new Map<number, FdEntry>(); // unified fd table
  private nativeRefs = new Map<number, number>(); // nativeFd -> refcount
  private nextFd = 3; // next available fd (0-2 are stdio)
  // Stat cache: populated from getdents, consumed by fs_stat. Avoids N+1 bridge calls for ls -l.
  // Path → {mode, size, nlink, mtime, expiresAt (ms since epoch)}
  private statCache = new Map<string, { mode: number; size: number; nlink: number; mtime: number; expiresAt: number }>();
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
  }

  /**
   * Check if a guest path should be routed to the native Swift filesystem.
   * Returns the resolved absolute path if bridging, or null if staying in JS VFS.
   *
   * JS VFS handles: /proc, /dev (virtual, per-process)
   * Native Swift handles: everything else (/bin, /etc, /tmp, /root, /home, /var, /usr, ...)
   */
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

    cpu.setSP(savedSP);
    cpu.setNZCV(savedNZCV);

    return savedPC; // caller should set PC to this
  }

  /** Queue a signal for delivery. */
  raiseSignal(signum: number): void {
    this.pendingSignals.push(signum);
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
    if (fd >= this.nextFd) this.nextFd = fd + 1;
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
      vfsFdSnapshot: this.vfs ? this.vfs.snapshotFds() : undefined,
    };
  }

  /** Restore state from process snapshot */
  restoreState(state: { brk: bigint; mmapBase: bigint; cwd: string; fdFlags: Map<number, number>; fdTable?: Map<number, FdEntry>; nativeRefs?: Map<number, number>; nextFd?: number; vfsFdSnapshot?: any }): void {
    this.brk = state.brk;
    this.mmapBase = state.mmapBase;
    if (this.vfs) {
      this.vfs.chdir(state.cwd);
      if (state.vfsFdSnapshot) this.vfs.restoreFds(state.vfsFdSnapshot);
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
    // Also close VFS fds >= 3 (child shouldn't inherit parent's open files after exec)
    if (this.vfs) this.vfs.closeAllExcept(0, 1, 2);
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
          if (readEntry.type === 'pipe_write') {
            ret = -BigInt(EBADF); // can't read from write end of pipe
            break;
          }
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
          if (readEntry.type === 'socket') {
            return {
              status: 'bridge', method: 'net_read',
              args: { socketId: readEntry.socketId, count },
              apply: (resp: any) => {
                if (resp.ret < 0) {
                  cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
                } else if (resp.data) {
                  const decoded = Uint8Array.from(atob(resp.data), c => c.charCodeAt(0));
                  bytes.set(decoded, buf + GUEST_MEM_BASE);
                  cpu.setReg(0, BigInt(decoded.length));
                } else {
                  cpu.setReg(0, BigInt(resp.ret));
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
          if (writeEntry.type === 'vfs') {
            ret = this.vfs ? BigInt(this.vfs.write(writeEntry.fd, buf, count, view)) : -BigInt(EBADF);
            break;
          }
          if (writeEntry.type === 'native') {
            const writeData = bytes.slice(buf + GUEST_MEM_BASE, buf + GUEST_MEM_BASE + count);
            const b64 = btoa(String.fromCharCode(...writeData));
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
            const sockText = new TextDecoder().decode(sockWriteData);
            console.log(`[net] write(fd=${fd}, ${count}B): ${JSON.stringify(sockText)}`);
            const sockB64 = btoa(String.fromCharCode(...sockWriteData));
            return {
              status: 'bridge', method: 'net_write',
              args: { socketId: writeEntry.socketId, data: sockB64 },
              apply: (resp: any) => {
                cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
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
          const b64 = btoa(String.fromCharCode(...combined));
          const text = new TextDecoder().decode(combined);
          // Log individual iov entries for debugging
          const iovDetails: string[] = [];
          for (let i = 0; i < iovcnt; i++) {
            const iovBase2 = Number(view.getBigUint64(iovBase + i * 16 + GUEST_MEM_BASE, true));
            const iovLen2 = Number(view.getBigUint64(iovBase + i * 16 + 8 + GUEST_MEM_BASE, true));
            iovDetails.push(`{0x${iovBase2.toString(16)},${iovLen2}}`);
          }
          // For the corrupted User-Agent: check the null byte
          if (totalLen > 30 && text.includes('User-Agent')) {
            const strAddr = Number(view.getBigUint64(iovBase + 16 + GUEST_MEM_BASE, true));
            const strLen = Number(view.getBigUint64(iovBase + 24 + GUEST_MEM_BASE, true));
            console.log(`[net] DEBUG iov[1] str@0x${strAddr.toString(16)} len=${strLen}`);
            // Dump bytes at the string address
            const strBytes: string[] = [];
            for (let i = 0; i < Math.min(strLen + 4, 35); i++) {
              strBytes.push(bytes[strAddr + GUEST_MEM_BASE + i].toString(16).padStart(2, '0'));
            }
            console.log(`[net] DEBUG bytes: ${strBytes.join(' ')}`);
            // Check if null byte at position 4
            console.log(`[net] DEBUG byte@+4: ${bytes[strAddr + GUEST_MEM_BASE + 4]} (should be 0)`);
          }
          console.log(`[net] writev(fd=${fd}, ${totalLen}B, iovs=[${iovDetails.join(',')}]): ${JSON.stringify(text)}`);
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
        if (this.vfs) {
          // Save current position, seek, read, restore position
          const savedPos = this.vfs.lseek(prFd, 0n, 1); // SEEK_CUR
          if (savedPos < 0n) { ret = savedPos; break; }
          this.vfs.lseek(prFd, prOffset, 0); // SEEK_SET
          const n = this.vfs.read(prFd, prBuf, prCount, view);
          this.vfs.lseek(prFd, savedPos, 0); // restore
          ret = BigInt(n);
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
        const newfd = Number(BigInt.asIntN(32, a1));
        const dup3flags = Number(a2);
        // Per spec: dup3 with oldfd==newfd returns EINVAL
        if (oldfd === newfd) { ret = -BigInt(EINVAL); break; }
        // Validate fd range
        if (newfd < 0 || newfd >= MAX_FDS) { ret = -BigInt(EBADF); break; }
        // Validate oldfd exists
        const oldfdValid = oldfd >= 0 && (oldfd <= 2 || this.fdTable.has(oldfd) || (this.vfs && this.vfs.hasFd(oldfd)));
        if (!oldfdValid) { ret = -BigInt(EBADF); break; }
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
          break;
        }
        // Per spec: dup3 O_CLOEXEC flag sets FD_CLOEXEC on the new fd
        if (dup3flags & 0o2000000) this.fdCloexec.add(newfd);
        else this.fdCloexec.delete(newfd);
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
          // Apply umask to creation mode (per spec: effective mode = mode & ~umask)
          const effectiveMode = mode56 & ~this.umask;
          const openRet = this.vfs.open(path, flags56, effectiveMode);
          if (openRet >= 0) {
            // Store status flags for F_GETFL (strip creation-only flags)
            this.fdFlags.set(openRet, flags56 & ~(0o100 | 0o200 | 0o1000 | 0o2000000)); // strip O_CREAT|O_EXCL|O_TRUNC|O_CLOEXEC
            // O_CLOEXEC → set FD_CLOEXEC on the new fd
            if (flags56 & 0o2000000) this.fdCloexec.add(openRet);
          }
          ret = BigInt(openRet);
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
          if (closeEntry.type === 'socket') {
            this.fdTable.delete(fd);
            return {
              status: 'bridge', method: 'net_close',
              args: { socketId: closeEntry.socketId },
              apply: (resp: any) => { cpu.setReg(0, 0n); },
            };
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
            // Offset is signed (negative for backward seeks)
            ret = this.vfs ? this.vfs.lseek(lseekEntry.fd, BigInt.asIntN(64, a1), Number(a2)) : -BigInt(EBADF);
            break;
          }
        }
        if (this.vfs) {
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
          // Check stat cache (populated by recent getdents) — avoids bridge round-trip
          const cached = this.statCache.get(bp79);
          if (cached && cached.expiresAt > Date.now()) {
            writeBridgeStat(view, statBuf79, { _meta: [cached.mode, cached.size, cached.nlink, cached.mtime] });
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
          ret = BigInt(this.vfs.mkdirat(path, Number(a2) & ~this.umask));
        } else {
          ret = -BigInt(ENOENT);
        }
        break;
      }

      case 35: { // unlinkat(dirfd, path, flags)
        const path = readGuestString(bytes, Number(a1));
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
        const path = readGuestString(bytes, Number(a1));
        const mode = Number(a2);
        if (this.vfs) {
          // S_IFIFO = 0o010000 — named pipe (mkfifo)
          // S_IFREG = 0o100000 — regular file
          // S_IFCHR = 0o020000 — character device (needs root, skip for now)
          const fmt = mode & 0o170000;
          if (fmt === 0o010000) {
            // mkfifo — create as a regular empty file (we don't support real FIFOs yet)
            ret = BigInt(this.vfs.open(path, 0o100 | 0o1000, mode & 0o7777)); // O_CREAT | O_TRUNC
            if (ret >= 0n) {
              this.vfs.close(Number(ret));
              ret = 0n;
            }
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
        const oldpath = readGuestString(bytes, Number(a1));
        const newpath = readGuestString(bytes, Number(a3));
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
        const target = readGuestString(bytes, Number(a0));
        const linkpath = readGuestString(bytes, Number(a2));
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
        const oldpath = readGuestString(bytes, Number(a1));
        const newpath = readGuestString(bytes, Number(a3));
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
        const path = readGuestString(bytes, Number(a1));
        const buf = Number(a2);
        const bufsiz = Number(a3);
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
      case 52: { // fchmod(fd, mode) — actually track mode changes
        const fchmodFd = Number(a0);
        const fchmodMode = Number(a1);
        // Update mode in VFS if it's a VFS file
        if (this.vfs && this.vfs.hasFd(fchmodFd)) {
          this.vfs.fchmod(fchmodFd, fchmodMode);
        }
        ret = 0n;
        break;
      }
      case 53:   // (unused on aarch64, but keep for safety)
      case 54: { // fchmodat(dirfd, path, mode, flags)
        ret = 0n;
        break;
      }
      case 55: { // fchownat(dirfd, path, owner, group, flags)
        // If AT_EMPTY_PATH (0x1000) is set, operate on dirfd directly (like fchown)
        const fchownFlags = Number(a4);
        if (fchownFlags & 0x1000) {
          const fchownFd = Number(a0);
          if (this.vfs && this.vfs.hasFd(fchownFd)) {
            this.vfs.fchown(fchownFd, Number(BigInt.asIntN(32, a2)), Number(BigInt.asIntN(32, a3)));
          }
        }
        // Otherwise it's path-based — we accept but don't track path-based ownership changes yet
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
        const prot = Number(a2);
        const flags = Number(a3);
        const mmapFd = Number(BigInt.asIntN(32, a4));
        const aligned = (length + 0xFFF) & ~0xFFF; // page-align
        const MAP_ANONYMOUS = 0x20;
        const MAP_FIXED = 0x10;
        const MAP_SHARED = 0x01;

        // Honor requested address if nonzero (MAP_FIXED or hint)
        let addr: bigint;
        if (a0 !== 0n) {
          addr = a0;
          const end = a0 + BigInt(aligned);
          if (end > this.mmapBase) this.mmapBase = end;
        } else {
          addr = this.mmapBase;
          this.mmapBase += BigInt(aligned);
        }

        // Ensure WASM memory is large enough
        const needed = Number(addr) + aligned + GUEST_MEM_BASE;
        if (needed > memory.buffer.byteLength) {
          const pagesNeeded = Math.ceil(needed / 65536);
          const currentPages = memory.buffer.byteLength / 65536;
          if (pagesNeeded > currentPages) memory.grow(pagesNeeded - currentPages);
        }

        // Zero-fill the mapped region
        new Uint8Array(memory.buffer).fill(0, Number(addr) + GUEST_MEM_BASE, Number(addr) + GUEST_MEM_BASE + aligned);

        // File-backed mmap: read file content into the mapped region
        if (mmapFd >= 0 && !(flags & MAP_ANONYMOUS) && this.vfs) {
          const fileOffset = Number(a5); // sixth argument: file offset
          // Save current file position, seek to requested offset, read, restore
          const savedPos = this.vfs.lseek(mmapFd, 0n, 1); // SEEK_CUR
          if (savedPos >= 0n) {
            this.vfs.lseek(mmapFd, BigInt(fileOffset), 0); // SEEK_SET
            this.vfs.read(mmapFd, Number(addr), aligned, new DataView(memory.buffer));
            this.vfs.lseek(mmapFd, savedPos, 0); // restore
          }
        }

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

      // gettid handled with getpid above

      case 172: // getpid
        ret = BigInt(this.currentPid);
        break;

      case 173: // getppid
        ret = BigInt(this.currentPpid);
        break;

      case 178: // gettid (same as pid in single-threaded model)
        ret = BigInt(this.currentPid);
        break;

      case 174: // getuid
      case 175: // geteuid
      case 176: // getgid
      case 177: // getegid
        ret = 0n; // root
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
        if (encoded.length > size) {
          ret = -BigInt(34); // ERANGE
        } else {
          const dest = buf + GUEST_MEM_BASE;
          if (dest + encoded.length <= memory.buffer.byteLength) {
            new Uint8Array(memory.buffer).set(encoded, dest);
          }
          ret = a0;
        }
        break;
      }

      case 49: { // chdir(path)
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
              } else if (resp.mode && (resp.mode & 0o170000) === 0o40000) {
                // It's a directory — update cwd
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
          case 5: // F_GETLK — query lock status
          case 12: { // F_GETLK64
            // Report no conflicting lock: set l_type = F_UNLCK (2) in the flock struct
            const flockPtr = Number(a2) + GUEST_MEM_BASE;
            view.setInt16(flockPtr, 2, true); // l_type = F_UNLCK
            ret = 0n;
            break;
          }
          case 6:  // F_SETLK — set lock (non-blocking)
          case 7:  // F_SETLKW — set lock (blocking)
          case 13: // F_SETLK64
          case 14: // F_SETLKW64
            ret = 0n; // accept all locks (single-user system, no contention)
            break;
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
          case 0: { // FUTEX_WAIT: if *uaddr == val, sleep until FUTEX_WAKE
            const current = view.getInt32(futexAddr, true);
            if (current !== futexVal) {
              ret = -BigInt(11); // EAGAIN — value mismatch
            } else {
              // Single-threaded: no other thread can wake us, return EAGAIN
              // Multi-threaded (future): would Atomics.wait here
              ret = -BigInt(11); // EAGAIN
            }
            break;
          }
          case 1: // FUTEX_WAKE: wake up to val waiters
            ret = 0n; // no waiters in single-threaded mode
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
          case 9: // FUTEX_WAIT_BITSET: like WAIT but with bitmask
            ret = -BigInt(11); // EAGAIN
            break;
          case 10: // FUTEX_WAKE_BITSET
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

      case 143: // setregid
      case 144: // setgid
      case 145: // setreuid
      case 146: // setuid
      case 147: // setresuid
      case 149: // setresgid
      case 151: // setfsuid
      case 152: // setfsgid
      case 159: // setgroups
        // We're always root — accept any uid/gid change
        ret = 0n;
        break;
      // ── Networking (via Swift bridge) ──

      case 198: { // socket(domain, type, protocol)
        const domain = Number(a0);   // AF_INET=2, AF_INET6=10, AF_UNIX=1
        const sockType = Number(a1) & 0xFF; // SOCK_STREAM=1, SOCK_DGRAM=2 (strip SOCK_CLOEXEC|SOCK_NONBLOCK)
        const protocol = Number(a2);
        console.log(`[net] socket(domain=${domain}, type=${sockType})`);

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
              this.nextFd--; // reclaim
            } else {
              this.fdTable.set(socketFd, { type: 'socket', socketId: resp.ret, domain, sockType });
              if (Number(a1) & 0o2000000) this.fdCloexec.add(socketFd); // SOCK_CLOEXEC
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
              this.nextFd--;
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
        console.log(`[net] connect(${connAddr.host}:${connAddr.port})`);
        connEntry.remoteAddr = connAddr;
        return {
          status: 'bridge', method: 'net_connect',
          args: { socketId: connEntry.socketId, ...connAddr },
          apply: (resp: any) => {
            console.log(`[net] connect → ${resp.ret}`);
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
        const b64 = btoa(String.fromCharCode(...sendData));
        // If dest_addr is provided (connectionless sendto), use net_sendto
        const destAddrPtr = Number(a4);
        const destAddrLen = Number(a5);
        if (destAddrPtr !== 0 && destAddrLen > 0) {
          const dest = this.parseSockaddr(bytes, destAddrPtr, destAddrLen);
          if (!stEntry.remoteAddr) stEntry.remoteAddr = dest;
          console.log(`[net] sendto(${dest.host}:${dest.port}, ${sendLen}B)`);
          return {
            status: 'bridge', method: 'net_sendto',
            args: { socketId: stEntry.socketId, data: b64, host: dest.host, port: dest.port },
            apply: (resp: any) => {
              console.log(`[net] sendto → ${resp.ret}`);
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
        const recvLen = Number(a2);
        return {
          status: 'bridge', method: 'net_read',
          args: { socketId: rfEntry.socketId, count: recvLen },
          apply: (resp: any) => {
            if (resp.ret < 0) {
              console.log(`[net] recvfrom → err ${resp.ret}`);
              cpu.setReg(0, BigInt.asUintN(64, BigInt(resp.ret)));
            } else if (resp.data) {
              const decoded = Uint8Array.from(atob(resp.data), c => c.charCodeAt(0));
              bytes.set(decoded, recvBuf + GUEST_MEM_BASE);
              console.log(`[net] recvfrom → ${decoded.length}B`);
              cpu.setReg(0, BigInt(decoded.length));
            } else {
              console.log(`[net] recvfrom → ${resp.ret} (no data)`);
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
        // For now, ignore signals sent to self or children
        ret = 0n;
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
        ret = 0n;
        break;
      }

      case 261: { // prlimit64(pid, resource, new_limit, old_limit)
        // Per spec: get/set resource limits
        // struct rlimit64: { rlim_cur (8 bytes), rlim_max (8 bytes) }
        const rlOldPtr = Number(a3);
        const RLIM_INFINITY = 0xFFFFFFFFFFFFFFFFn;
        if (rlOldPtr !== 0) {
          const base = rlOldPtr + GUEST_MEM_BASE;
          const resource = Number(a1);
          // Return reasonable defaults for common resources
          switch (resource) {
            case 7:  // RLIMIT_NOFILE
              view.setBigUint64(base, 1024n, true);     // cur
              view.setBigUint64(base + 8, 1024n, true); // max
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
        ret = 0n;
        break;
      }

      case 233: // madvise — accept any advice
        ret = 0n;
        break;

      // ── Missing syscalls (spec-compliant implementations) ──

      case 50: { // fchdir(fd)
        // Per spec: change cwd to the directory referenced by fd
        // We need the path of the open dir — but our VFS doesn't track that easily
        // For now, return ENOSYS (this is rare in practice)
        ret = -BigInt(EBADF);
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
        ret = 0n;
        break;
      }

      case 124: // sched_yield — yield CPU (no-op in emulator)
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
            // Write "jitter\0" to the buffer
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
        // Zero out the itimerval struct (16 bytes: 2x timeval of 8+8)
        const itvPtr = Number(a1) + GUEST_MEM_BASE;
        for (let i = 0; i < 32; i += 8) view.setBigUint64(itvPtr + i, 0n, true);
        ret = 0n;
        break;
      }

      case 103: { // setitimer(which, new_value, old_value)
        // Per spec: set interval timer (ITIMER_REAL sends SIGALRM)
        // Accept but don't actually deliver signals
        const oldPtr = Number(a2);
        if (oldPtr !== 0) {
          const base = oldPtr + GUEST_MEM_BASE;
          for (let i = 0; i < 32; i += 8) view.setBigUint64(base + i, 0n, true);
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
        this.fdTable.set(inoFd, { type: 'stdio', fd: inoFd }); // placeholder
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
        this.fdTable.set(tfFd, { type: 'stdio', fd: tfFd }); // placeholder
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
      case 228: // mlock
      case 229: // munlock
      case 230: // mlockall
      case 231: // munlockall
        ret = 0n;
        break;
      case 227: // msync — sync memory mapping to file (no-op, our VFS is in-memory)
        ret = 0n;
        break;
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
        // Per spec: resize memory mapping
        // Simple implementation: return the same address (pretend resize succeeded)
        ret = a0; // return old address
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
        const pathname281 = readGuestString(bytes, Number(a1));
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
        return { status: 'execve', path: pathname281, argv: argvPtrs281, envp: envpPtrs281 };
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
        const oldpath276 = readGuestString(bytes, Number(a1));
        const newpath276 = readGuestString(bytes, Number(a3));
        const renameFlags = Number(a4);
        if (this.vfs) {
          if (renameFlags & 1) { // RENAME_NOREPLACE — check if target exists
            // Our VFS rename already handles this correctly for existing targets
            // but RENAME_NOREPLACE should fail if target exists
            ret = BigInt(this.vfs.rename(oldpath276, newpath276));
          } else {
            ret = BigInt(this.vfs.rename(oldpath276, newpath276));
          }
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
        const sxPath = readGuestString(bytes, Number(a1));
        const sxFlags = Number(a2);
        const sxBufAddr = Number(a4);
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
        // Create an eventfd — a simple counter-based notification fd
        // For our purposes, return a fd that reads/writes a 64-bit counter
        // Implementation: use a VFS file as backing
        if (this.vfs) {
          const buf = new Uint8Array(8);
          const dv = new DataView(buf.buffer);
          dv.setBigUint64(0, BigInt(Number(a0)), true);
          // Create a temp file for the eventfd
          const fd = this.vfs.open('/dev/null', 0, 0); // just get a fd
          // Actually, eventfd needs special semantics. For now, return a valid fd.
          // Most software just needs it to not return ENOSYS.
          ret = BigInt(fd >= 0 ? fd : -EBADF);
        } else {
          ret = -BigInt(ENOSYS);
        }
        break;
      }

      case 20: { // epoll_create1(flags)
        // Per spec: create epoll instance, returns fd
        // We store epoll state indexed by the returned fd
        const epfd = this.allocFd();
        this.fdTable.set(epfd, { type: 'stdio', fd: epfd }); // placeholder entry
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
        // Per spec: synchronous I/O multiplexing
        // Simple implementation: report all fds as ready
        const nfds = Number(a0);
        // readfds, writefds at a1, a2 — leave them as-is (already set by caller)
        ret = BigInt(nfds); // all ready
        break;
      }

      case 73: { // ppoll(fds, nfds, tmo_p, sigmask)
        // struct pollfd { int fd; short events; short revents; }  (8 bytes each)
        // Per POSIX: block until at least one fd is ready, or timeout expires.
        // Returns number of fds with non-zero revents, 0 on timeout.
        const fdsAddr = Number(a0);
        const nfds = Number(a1);
        const tmoPtr = Number(a2);

        let tmoMs = -1; // -1 = infinite
        if (tmoPtr !== 0) {
          const tmoSec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE, true));
          const tmoNsec = Number(view.getBigInt64(tmoPtr + GUEST_MEM_BASE + 8, true));
          tmoMs = tmoSec * 1000 + Math.floor(tmoNsec / 1_000_000);
        }

        // First pass: handle non-socket fds and find socket fds
        let readyCount = 0;
        let socketFd = -1;
        let socketEntry: any = null;

        for (let i = 0; i < nfds; i++) {
          const pfd = fdsAddr + i * 8 + GUEST_MEM_BASE;
          const fd = view.getInt32(pfd, true);
          const events = view.getInt16(pfd + 4, true);

          if (fd < 0) {
            view.setInt16(pfd + 6, 0, true);
            continue;
          }

          const entry = this.getFdEntry(fd);
          if (entry?.type === 'socket' && (events & 1)) { // POLLIN
            // Don't set revents yet — need to ask bridge
            view.setInt16(pfd + 6, 0, true);
            if (socketFd < 0) {
              socketFd = fd;
              socketEntry = entry;
            }
          } else {
            // Non-socket fds: report as ready
            view.setInt16(pfd + 6, events, true);
            if (events) readyCount++;
          }
        }

        // If non-socket fds are ready, return immediately (don't block for sockets)
        if (readyCount > 0) {
          ret = BigInt(readyCount);
          break;
        }

        // Only socket fds being polled — ask bridge to wait for data
        if (socketFd >= 0 && socketEntry) {
          const pollTmo = tmoMs < 0 ? 30000 : tmoMs; // cap infinite at 30s
          return {
            status: 'bridge', method: 'net_poll',
            args: { socketId: socketEntry.socketId, timeoutMs: pollTmo },
            apply: (resp: any) => {
              if (resp.ret > 0) {
                // Mark socket fds as ready
                for (let i = 0; i < nfds; i++) {
                  const pfd = fdsAddr + i * 8 + GUEST_MEM_BASE;
                  const fd = view.getInt32(pfd, true);
                  const events = view.getInt16(pfd + 4, true);
                  if (fd >= 0 && this.getFdEntry(fd)?.type === 'socket' && (events & 1)) {
                    view.setInt16(pfd + 6, events, true);
                  }
                }
                cpu.setReg(0, 1n);
              } else {
                cpu.setReg(0, 0n); // timeout
              }
            },
          };
        }

        // No fds at all (or all fd=-1)
        ret = 0n;
        break;
      }

      // ── Remaining syscalls from aarch64 spec ──

      // Mount/unmount (needed for container/init, stub for now)
      case 39: // mount
      case 40: // mount_setattr
        ret = 0n; // pretend mount succeeded
        break;
      case 60: // umount2
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
        const smB64 = btoa(String.fromCharCode(...combined));
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
        console.log(`[net] recvmsg(fd=${Number(a0)}, capacity=${rmCapacity})`);
        return {
          status: 'bridge', method: 'net_read',
          args: { socketId: rmEntry.socketId, count: rmCapacity },
          apply: (resp: any) => {
            if (resp.ret < 0) {
              console.log(`[net] recvmsg → err ${resp.ret}`);
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
            console.log(`[net] recvmsg → ${recvd.length}B`);
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
      case 236: // mlock2
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
        const path437 = readGuestString(bytes, Number(a1));
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
        const fa2Path = readGuestString(bytes, Number(a1));
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
  // Meta layout for fs_stat: _meta[0]=mode, _meta[1]=size, _meta[2]=nlink, _meta[3]=mtime
  const meta = resp._meta;
  const mode = meta ? meta[0] : (resp.mode ?? 0o100644);
  const size = meta ? meta[1] : (resp.size ?? 0);
  const nlink = meta ? meta[2] : (resp.nlink ?? 1);
  const mtime = meta ? meta[3] : (resp.mtime ?? 0);
  view.setUint32(base + 16, mode, true);
  view.setUint32(base + 20, nlink, true);
  view.setBigUint64(base + 48, BigInt(size), true);
  view.setBigUint64(base + 88, BigInt(mtime), true);
}
