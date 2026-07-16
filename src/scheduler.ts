// cooperative fork scheduler for the single-threaded conformance harness.
//
// real linux concurrency semantics (a parent and its children alive at the same
// time, blocking on pipes / wait / locks and handing control to one another) can't
// be expressed by the serialized vfork model in process.ts, which runs a child to
// completion before the parent resumes. bidirectional handshakes (child waits for
// the parent to write a pipe, parent waits for the child) deadlock there.
//
// this scheduler keeps one process image swapped into the single live
// WebAssembly.Memory at a time. because it is the same Memory instance throughout,
// the emulator's compiled-region cache stays valid (all forked processes run the
// same code at the same guest addresses). a context switch saves the outgoing
// process's used memory span (which includes its cpu-state block at offset 0) into a
// side buffer and loads the incoming one. switches only happen at blocking points,
// which are rare, so the copy cost is bounded.

import { REG_OFFSET, GUEST_MEM_BASE } from './types.js';
import type { SyscallHandler } from './syscall.js';

export type BlockReason = 'pipe_read' | 'pipe_write' | 'wait' | 'lock' | 'futex';

// posix advisory record locks (fcntl F_SETLK/F_GETLK/F_SETLKW). locks are keyed by
// a system-wide file identity (vfs inode / native path) and owned by a pid. a write
// lock conflicts with any other pid's lock on an overlapping byte range; a read lock
// conflicts only with another pid's write lock. same-pid locks never conflict and
// replace one another. ranges use inclusive [start, end]; end === Infinity means EOF.
interface RecordLock { pid: number; type: 'r' | 'w'; start: number; end: number; }

export class LockManager {
  private byFile = new Map<string, RecordLock[]>();
  // what each process currently blocked in F_SETLKW is waiting for, for deadlock
  // detection (posix EDEADLK). cleared when the process acquires, unblocks, or exits.
  private waiters = new Map<number, { key: string; type: 'r' | 'w'; start: number; end: number }>();

  setWaiter(pid: number, key: string, type: 'r' | 'w', start: number, end: number): void {
    this.waiters.set(pid, { key, type, start, end });
  }
  clearWaiter(pid: number): void { this.waiters.delete(pid); }

  // pids (other than `pid`) holding a lock that conflicts with the given request.
  private holdersOf(key: string, pid: number, type: 'r' | 'w', start: number, end: number): number[] {
    const list = this.byFile.get(key);
    if (!list) return [];
    const out: number[] = [];
    for (const l of list) {
      if (l.pid === pid) continue;
      if (l.end < start || l.start > end) continue;
      if (type === 'w' || l.type === 'w') out.push(l.pid);
    }
    return out;
  }

  // would `pid` blocking on the given request close a cycle in the wait-for graph?
  // (pid waits for the lock holders; each blocked holder waits for its own holders;
  // if that chain returns to pid, granting-by-waiting would deadlock -> EDEADLK.)
  wouldDeadlock(pid: number, key: string, type: 'r' | 'w', start: number, end: number): boolean {
    const seen = new Set<number>();
    const toVisit = this.holdersOf(key, pid, type, start, end);
    while (toVisit.length) {
      const h = toVisit.pop()!;
      if (h === pid) return true;
      if (seen.has(h)) continue;
      seen.add(h);
      const w = this.waiters.get(h);
      if (w) for (const nh of this.holdersOf(w.key, h, w.type, w.start, w.end)) toVisit.push(nh);
    }
    return false;
  }

  conflict(key: string, pid: number, type: 'r' | 'w', start: number, end: number): RecordLock | null {
    const list = this.byFile.get(key);
    if (!list) return null;
    let best: RecordLock | null = null;
    for (const l of list) {
      if (l.pid === pid) continue;
      if (l.end < start || l.start > end) continue; // disjoint
      if (type === 'w' || l.type === 'w') { if (!best || l.start < best.start) best = l; }
    }
    return best;
  }

  // merge each (pid, type) group's overlapping or adjacent locks into maximal ranges
  // so f_getlk reports one coherent region rather than fragments (posix coalescing).
  private coalesce(key: string): void {
    const list = this.byFile.get(key);
    if (!list || list.length < 2) return;
    const groups = new Map<string, RecordLock[]>();
    for (const l of list) {
      const gk = `${l.pid}|${l.type}`;
      const g = groups.get(gk); if (g) g.push(l); else groups.set(gk, [l]);
    }
    const merged: RecordLock[] = [];
    for (const g of groups.values()) {
      g.sort((a, b) => a.start - b.start);
      let cur = { ...g[0] };
      for (let i = 1; i < g.length; i++) {
        const n = g[i];
        if (n.start <= cur.end + 1) cur.end = Math.max(cur.end, n.end);
        else { merged.push(cur); cur = { ...n }; }
      }
      merged.push(cur);
    }
    this.byFile.set(key, merged);
  }

  // trim/split this pid's locks so none covers any byte in [start, end].
  private removeRange(key: string, pid: number, start: number, end: number): void {
    const list = this.byFile.get(key);
    if (!list) return;
    const out: RecordLock[] = [];
    for (const l of list) {
      if (l.pid !== pid || l.end < start || l.start > end) { out.push(l); continue; }
      if (l.start < start) out.push({ pid, type: l.type, start: l.start, end: start - 1 });
      if (l.end > end) out.push({ pid, type: l.type, start: end + 1, end: l.end });
    }
    if (out.length) this.byFile.set(key, out); else this.byFile.delete(key);
  }

  set(key: string, pid: number, type: 'r' | 'w' | 'u', start: number, end: number): void {
    this.removeRange(key, pid, start, end);
    if (type === 'u') return;
    const list = this.byFile.get(key) ?? [];
    list.push({ pid, type, start, end });
    this.byFile.set(key, list);
    this.coalesce(key);
  }

  releaseFile(key: string, pid: number): void {
    const list = this.byFile.get(key);
    if (!list) return;
    const kept = list.filter((l) => l.pid !== pid);
    if (kept.length) this.byFile.set(key, kept); else this.byFile.delete(key);
  }

  releasePid(pid: number): void {
    this.waiters.delete(pid);
    for (const [key, list] of this.byFile) {
      const kept = list.filter((l) => l.pid !== pid);
      if (kept.length) this.byFile.set(key, kept); else this.byFile.delete(key);
    }
  }
}

export interface ProcCtx {
  pid: number;
  ppid: number;
  state: 'run' | 'blocked' | 'zombie';
  blockReason: BlockReason | null;
  blockAddr: number;          // futex guest address when blocked on a futex, else 0
  pc: bigint;                 // guest pc to resume at
  mem: Uint8Array | null;     // saved image [0, memLen); null while this proc is live
  memLen: number;
  syscallState: ReturnType<SyscallHandler['getState']> | null;
  pendingSignals: number[];
  exitCode: number;           // encoded wait status once zombie (0x10000 bit => signalled)
  reaped: boolean;
  children: Set<number>;
}

export class Scheduler {
  procs = new Map<number, ProcCtx>();
  current: ProcCtx;
  private memory: WebAssembly.Memory;
  private syscalls: SyscallHandler;
  private nextPid: number;
  private hwm = 0; // shared high-water mark (bytes) for context-switch image size
  private rr = 0;  // round-robin cursor over pids
  // MAP_SHARED anonymous regions are common to every process that inherited them
  // (ltp aggregates test results and synchronizes through such a mapping). the
  // per-process image would give each process a private copy, so a coherent backing
  // is kept here: persisted from live memory when a process yields, overlaid back
  // onto live memory when a process is scheduled in. keyed by guest address.
  private sharedBacking = new Map<number, Uint8Array>();
  readonly locks = new LockManager();

  constructor(memory: WebAssembly.Memory, syscalls: SyscallHandler, rootPid: number, rootPpid: number, rootPC: bigint) {
    this.memory = memory;
    this.syscalls = syscalls;
    this.nextPid = rootPid + 1;
    this.current = {
      pid: rootPid, ppid: rootPpid, state: 'run', blockReason: null, blockAddr: 0,
      pc: rootPC, mem: null, memLen: 0,
      syscallState: null, pendingSignals: [], exitCode: 0, reaped: false,
      children: new Set(),
    };
    this.procs.set(rootPid, this.current);
    this.growHwm();
  }

  // ── memory image bounds ──

  private growHwm(): void {
    const clamped = Math.min(this.syscalls.memImageBound, this.memory.buffer.byteLength);
    if (clamped > this.hwm) this.hwm = clamped;
  }

  /** copy a process's MAP_SHARED regions from live memory into the coherent backing. */
  private persistShared(state: ReturnType<SyscallHandler['getState']> | null): void {
    const regions = state?.sharedRegions;
    if (!regions || regions.length === 0) return;
    const buf = this.memory.buffer;
    for (const r of regions) {
      const start = r.addr + GUEST_MEM_BASE;
      this.sharedBacking.set(r.addr, new Uint8Array(buf.slice(start, start + r.len)));
    }
  }

  /** overlay the coherent backing for a process's MAP_SHARED regions onto live memory. */
  private applyShared(state: ReturnType<SyscallHandler['getState']> | null): void {
    const regions = state?.sharedRegions;
    if (!regions || regions.length === 0) return;
    const live = new Uint8Array(this.memory.buffer);
    for (const r of regions) {
      const backing = this.sharedBacking.get(r.addr);
      if (backing) live.set(backing, r.addr + GUEST_MEM_BASE);
    }
  }

  private saveCurrent(pc: bigint): void {
    this.growHwm();
    const cur = this.current;
    cur.pc = pc;
    cur.syscallState = this.syscalls.getState();
    cur.pendingSignals = this.syscalls.pendingSignals.slice();
    this.persistShared(cur.syscallState);
    cur.mem = new Uint8Array(this.memory.buffer.slice(0, this.hwm));
    cur.memLen = this.hwm;
  }

  private loadInto(next: ProcCtx): void {
    const live = new Uint8Array(this.memory.buffer);
    live.set(next.mem!, 0);
    // if the shared image grew since this process was last saved, the tail belongs
    // to no process yet and must read as zero (a fresh anonymous page), not stale
    // bytes left by whoever ran last.
    if (next.memLen < this.hwm) live.fill(0, next.memLen, this.hwm);
    this.syscalls.restoreState(next.syscallState!);
    this.syscalls.pendingSignals = next.pendingSignals;
    this.applyShared(next.syscallState); // bring in writes other processes made to shared memory
    next.mem = null;
    this.current = next;
  }

  /** switch the live memory from the current process to `next`, returning next's resume pc. */
  switchTo(next: ProcCtx, curPC: bigint): bigint {
    if (next === this.current) return curPC;
    if (this.current.state !== 'zombie') this.saveCurrent(curPC);
    this.loadInto(next);
    return next.pc;
  }

  // ── process lifecycle ──

  /**
   * fork the current (parent) process. the child is a full copy of the parent's
   * live memory and syscall state, with x0 = 0 baked into its saved cpu block. the
   * parent stays live and continues; its x0 is set to the child pid by the caller.
   */
  fork(resumePC: bigint): number {
    this.growHwm();
    const childPid = this.nextPid++;
    const img = new Uint8Array(this.memory.buffer.slice(0, this.hwm));
    // child's fork() returns 0
    new DataView(img.buffer).setBigUint64(REG_OFFSET, 0n, true);

    const childState = this.syscalls.getState();
    childState.currentPid = childPid;
    childState.currentPpid = this.current.pid;
    // parent and child both hold every inherited pipe end open
    this.syscalls.retainPipeRefs(childState.fdTable);

    const child: ProcCtx = {
      pid: childPid, ppid: this.current.pid, state: 'run', blockReason: null, blockAddr: 0,
      pc: resumePC, mem: img, memLen: this.hwm,
      syscallState: childState, pendingSignals: [], exitCode: 0, reaped: false,
      children: new Set(),
    };
    this.procs.set(childPid, child);
    this.current.children.add(childPid);
    return childPid;
  }

  /** turn the current process into a zombie with an encoded wait status. */
  zombify(encodedStatus: number): void {
    const cur = this.current;
    // dropping the process closes all its still-open fds; releasing pipe ends can
    // signal EOF to a reader parked in another process.
    const st = this.syscalls.getState();
    this.syscalls.releasePipeRefs(st.fdTable);
    this.locks.releasePid(cur.pid); // advisory locks are dropped when the owner dies
    // its final writes to shared memory must be visible to whoever reaps it.
    this.persistShared(st);
    cur.exitCode = encodedStatus;
    cur.state = 'zombie';
    cur.blockReason = null;
    cur.mem = null; // a zombie's memory is gone; only its exit status remains
    this.wakeAll(); // a parent parked in wait() may now reap
  }

  blockCurrent(reason: BlockReason, svcPC: bigint, addr = 0): void {
    this.current.state = 'blocked';
    this.current.blockReason = reason;
    this.current.blockAddr = addr;
    this.current.pc = svcPC; // re-run the blocking syscall when rescheduled
  }

  /** wake up to `n` processes blocked in FUTEX_WAIT on `addr`; return the count woken. */
  futexWake(addr: number, n: number): number {
    let woken = 0;
    for (const p of this.procs.values()) {
      if (woken >= n) break;
      if (p.state === 'blocked' && p.blockReason === 'futex' && p.blockAddr === addr) {
        p.state = 'run'; p.blockReason = null; p.blockAddr = 0;
        woken++;
      }
    }
    return woken;
  }

  /** mark every blocked process runnable so each re-checks its wait condition. */
  wakeAll(): void {
    for (const p of this.procs.values()) if (p.state === 'blocked') { p.state = 'run'; p.blockReason = null; }
  }

  /** pick a runnable process, round-robin, preferring one other than `current`. */
  pickNext(): ProcCtx | null {
    const runnable: ProcCtx[] = [];
    for (const p of this.procs.values()) if (p.state === 'run') runnable.push(p);
    if (runnable.length === 0) return null;
    runnable.sort((a, b) => a.pid - b.pid);
    // advance the cursor for fairness, then prefer a non-current runnable
    this.rr = (this.rr + 1) % runnable.length;
    const start = this.rr;
    for (let i = 0; i < runnable.length; i++) {
      const cand = runnable[(start + i) % runnable.length];
      if (cand !== this.current || runnable.length === 1) return cand;
    }
    return runnable[start];
  }

  anyLive(): boolean {
    for (const p of this.procs.values()) if (p.state !== 'zombie') return true;
    return false;
  }

  anyRunnable(): boolean {
    for (const p of this.procs.values()) if (p.state === 'run') return true;
    return false;
  }

  // ── wait/reap ──

  /** a zombie child of `parentPid` matching `wantPid` (-1 => any), or null. */
  findZombieChild(parentPid: number, wantPid: number): ProcCtx | null {
    const parent = this.procs.get(parentPid);
    if (!parent) return null;
    for (const cpid of parent.children) {
      const c = this.procs.get(cpid);
      if (!c || c.reaped || c.state !== 'zombie') continue;
      if (wantPid > 0 && cpid !== wantPid) continue;
      return c;
    }
    return null;
  }

  /** does `parentPid` still have an unreaped child (zombie or alive) matching wantPid? */
  hasChild(parentPid: number, wantPid: number): boolean {
    const parent = this.procs.get(parentPid);
    if (!parent) return false;
    for (const cpid of parent.children) {
      const c = this.procs.get(cpid);
      if (!c || c.reaped) continue;
      if (wantPid > 0 && cpid !== wantPid) continue;
      return true;
    }
    return false;
  }

  reap(child: ProcCtx): void {
    child.reaped = true;
    const parent = this.procs.get(child.ppid);
    if (parent) parent.children.delete(child.pid);
    this.procs.delete(child.pid);
  }

  // ── signals ──

  /** deliver a signal to a target process (kill). signum 0 is an existence probe.
   * returns false if the pid is unknown (ESRCH); a zombie still "exists" but takes no
   * signal. delivery to the current process is handled by the caller (its live pending
   * queue differs from the saved ctx), so this targets other processes. */
  signal(pid: number, signum: number): boolean {
    const p = this.procs.get(pid);
    if (!p) return false;
    if (signum !== 0 && p.state !== 'zombie') {
      p.pendingSignals.push(signum);
      this.wakeAll(); // let a blocked target act on the signal
    }
    return true;
  }
}
