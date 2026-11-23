/**
 * Shared constants and protocol definitions for the multicore kernel.
 *
 * The kernel uses a SharedArrayBuffer ("kernel bus") for all cross-Worker IPC:
 * process table, syscall mailboxes, pipe ring buffers, and stdin buffers.
 *
 * Layout:
 *   0x00000  Process Table     (64 slots × 64 bytes = 4KB)
 *   0x01000  Syscall Mailboxes (64 slots × 128 bytes = 8KB)
 *   0x03000  Pipe Ring Buffers (32 pipes × 16KB = 512KB)
 *   0x83000  Stdin Ring Buffers (8 × 8KB = 64KB)
 *   0x93000  Global State      (4KB)
 *   Total: ~600KB
 */

import type { ThreadLayout } from './types.js';

// ── Kernel SAB Layout ───────────────────────────────────────────────────────
//
// every region here is sized from PROC_TABLE_SLOTS and laid out end to end, rather than pinned to
// hand-picked offsets. the offsets used to be constants that happened to fit 64 slots, so the
// process limit was not a number anyone could change: raising it would have silently overlapped
// the next region. deriving them means the ceiling is one edit.
//
// how many processes can exist at once. 64 was a layout artifact, not a considered limit, and
// programs hit it: ltp's pipe13 wants a hundred children and got EAGAIN at sixty-four. a process
// that is merely declared costs no committed memory (measured: 135 guest-sized memories coexist
// untouched), so the real budget is what processes actually write, and the table should not be
// what stops them first.
export const PROC_TABLE_SLOTS = 256;
export const PROC_SLOT_SIZE = 256; // bytes per process entry
export const MAILBOX_SLOTS = PROC_TABLE_SLOTS;
export const MAILBOX_SLOT_SIZE = 128; // bytes per mailbox
export const BRIDGE_RESP_SLOTS = PROC_TABLE_SLOTS;
// 64 KiB per worker. the slot bounds how much a single bridge round trip can carry, and a round
// trip is a postMessage plus a futex wake -- so reading a 1 MiB binary in 12 KiB pieces cost ~90 of
// them just to start a process. it is shared memory, not a copy per worker.
export const BRIDGE_RESP_SLOT_SIZE = 65536;
// the most payload one response can carry: the slot minus the largest header any method writes
// (10 bytes plus fifteen 8-byte meta words). a caller that asks for more than this gets a reply
// its data does not fit in -- which used to mean the data was dropped and the count was kept, so
// a read said "65536 bytes" and delivered none. every read path caps its request at this.
export const BRIDGE_DATA_MAX = BRIDGE_RESP_SLOT_SIZE - 256;
// A PER-PROCESS SIGNAL QUEUE.
//
// what this replaces was a 32-bit word with one bit per signal, plus a two-int note beside it
// saying who sent it. that spans the classic signals and nothing else -- kill(pid, SIGRTMIN) from
// another process set no bit and was simply lost -- and a bit cannot carry a VALUE, which is the
// whole point of sigqueue(3) and of a posix timer's sigev_value.
//
// so: a ring per process. any process may write one, only its owner reads it, and every entry
// carries the whole siginfo its sender meant. an entry is published by stamping its sequence
// LAST, so a reader never sees one half-written.
export const SIGQ_ENTRIES = 64;
export const SIGQ_ENTRY_WORDS = 7;      // seq, signo, code, pid, uid, value lo, value hi
export const SIGQ_SLOT_WORDS = 2 + SIGQ_ENTRIES * SIGQ_ENTRY_WORDS;
export const SIGQ_HEAD = 0;
export const SIGQ_TAIL = 1;
export const PROC_CRED_WORDS = 4;       // ruid, euid, suid, rgid
export const PROC_CMDLINE_SLOT_SIZE = 256;
export const PIPE_RING_COUNT = 32;
export const STDIN_RING_SIZE = 8192; // 8KB per stdin buffer
export const STDIN_RING_COUNT = 8;

/** round up to a 4 KiB boundary, so every region starts page-aligned. */
const pg = (n: number) => (n + 0xFFF) & ~0xFFF;

export const PROC_TABLE_OFFSET = 0;
export const MAILBOX_OFFSET = PROC_TABLE_OFFSET + pg(PROC_TABLE_SLOTS * PROC_SLOT_SIZE);
export const STDIN_RING_OFFSET = MAILBOX_OFFSET + pg(MAILBOX_SLOTS * MAILBOX_SLOT_SIZE);
export const GLOBAL_STATE_OFFSET = STDIN_RING_OFFSET + pg(STDIN_RING_COUNT * STDIN_RING_SIZE);
export const BRIDGE_RESP_OFFSET = GLOBAL_STATE_OFFSET + pg(4096);
export const PROC_CMDLINE_OFFSET = BRIDGE_RESP_OFFSET + pg(BRIDGE_RESP_SLOTS * BRIDGE_RESP_SLOT_SIZE);
export const SIG_SENDER_OFFSET = PROC_CMDLINE_OFFSET + pg(PROC_TABLE_SLOTS * PROC_CMDLINE_SLOT_SIZE);
export const SIGQ_OFFSET = SIG_SENDER_OFFSET + pg(PROC_TABLE_SLOTS * SIGQ_SLOT_WORDS * 4);
export const PROC_CRED_OFFSET = SIGQ_OFFSET + pg(PROC_TABLE_SLOTS * SIGQ_SLOT_WORDS * 4);
export const FUTEX_TABLE_OFFSET = PROC_CRED_OFFSET + pg(PROC_TABLE_SLOTS * PROC_CRED_WORDS * 4);
export const PTRACE_TABLE_OFFSET = FUTEX_TABLE_OFFSET + pg(16 * 4 + 1024 * 8 * 4);
// one exchange slot per process: eight header words and a scratch area (see "ptrace exchange").
export const PTRACE_HDR_WORDS = 8;
export const PTRACE_SCRATCH = 1024;
export const PTRACE_SLOT_SIZE = PTRACE_HDR_WORDS * 4 + PTRACE_SCRATCH;
export const KERNEL_SAB_SIZE = PTRACE_TABLE_OFFSET + pg(PROC_TABLE_SLOTS * PTRACE_SLOT_SIZE);

// ── ptrace exchange ─────────────────────────────────────────────────────────────────────────────
// a tracer cannot read its tracee's memory: every process has its own wasm memory and the kernel
// (the main thread) holds none of them. so a STOPPED tracee is not idle -- it parks in a service
// loop and answers peeks, pokes and register reads out of one of these slots. the tracer writes a
// request, bumps REQ, and waits for ACK to catch up.
export const PT_REQ = 0;      // request sequence, bumped by the tracer
export const PT_ACK = 1;      // equal to REQ once the answer is there
export const PT_OP = 2;
export const PT_ADDR = 3;
export const PT_LEN = 4;      // bytes of scratch the request carries or asks for
export const PT_DATA_LO = 5;
export const PT_DATA_HI = 6;
export const PT_STATUS = 7;   // 0, or a negative errno
/** first word of `slot`'s ptrace exchange, in a whole-buffer Int32Array. */
export function ptraceSlotWord(slot: number): number {
  return (PTRACE_TABLE_OFFSET + slot * PTRACE_SLOT_SIZE) / 4;
}
/** byte offset of `slot`'s scratch area. */
export function ptraceScratchOffset(slot: number): number {
  return PTRACE_TABLE_OFFSET + slot * PTRACE_SLOT_SIZE + PTRACE_HDR_WORDS * 4;
}

// ── The futex wait queue ────────────────────────────────────────────────────────────────────────
//
// every blocked futex waiter in the system is one entry here, in memory every worker AND the kernel
// can see. it exists because Atomics.wait alone cannot express what futex(2) is: a wait that times
// out, a wake that counts what it woke, a bitset that selects among waiters, and a requeue that
// moves a waiter from one address to another without waking it.
//
// a waiter parks on its OWN entry's state word, so waking one waiter never disturbs another, and a
// futex_waitv group parks on its leader's. the key is the identity of the memory, not the address:
// two processes sharing a MAP_SHARED page must land on the same queue from different addresses, so
// a shared futex is keyed by its offset in the arena and a private one by (tgid, address).
export const FUTEX_SLOTS = 1024;      // must match the reservation in KERNEL_SAB_SIZE above
export const FUTEX_SLOT_WORDS = 8;
export const FUTEX_HDR_WORDS = 16;   // [0] lock, [1] allocation hint

export const FUTEX_STATE = 0;   // FUTEX_FREE / FUTEX_QUEUED / FUTEX_WOKEN -- the word waited on
export const FUTEX_KEYLO = 1;
export const FUTEX_KEYHI = 2;
export const FUTEX_BITS = 3;    // the waiter's bitset; a wake matches when the masks intersect
export const FUTEX_PID = 4;     // owner, so a process that dies queued does not leak its entries
export const FUTEX_LEADER = 5;  // slot whose state word this waiter sleeps on (itself, or its group's)
export const FUTEX_RESULT = 6;  // for a group: which member was woken
export const FUTEX_GROUPED = 7; // 1 when this entry is one futex of a futex_waitv

export const FUTEX_FREE = 0, FUTEX_QUEUED = 1, FUTEX_WOKEN = 2;

const futexLockIdx = FUTEX_TABLE_OFFSET / 4;
const futexSlotIdx = (slot: number) => (FUTEX_TABLE_OFFSET / 4) + FUTEX_HDR_WORDS + slot * FUTEX_SLOT_WORDS;

/** the state word a waiter parks on, as an index into the kernel SAB's Int32Array. */
export function futexStateIndex(slot: number): number { return futexSlotIdx(slot) + FUTEX_STATE; }

/**
 * take the table lock. every critical section is a bounded scan of the table, so this spins rather
 * than parks -- and the kernel's main thread, which cannot Atomics.wait at all, has to be able to
 * take it too (a dying process's entries are released there).
 */
export function futexLock(i32: Int32Array): void {
  const start = performance.now();
  for (;;) {
    if (Atomics.compareExchange(i32, futexLockIdx, 0, 1) === 0) return;
    for (let spin = 0; spin < 128 && Atomics.load(i32, futexLockIdx) !== 0; spin++) { /* back off */ }
    if (Atomics.load(i32, futexLockIdx) === 0) continue;
    // every critical section here is a bounded scan with no i/o, so a lock still held after a
    // quarter of a second belongs to a process the kernel terminated mid-section. take it: the
    // alternative is that every futex in the system stops working for the rest of the run.
    if (performance.now() - start > 2000) { Atomics.store(i32, futexLockIdx, 1); return; }
    // a holder that got descheduled: park briefly where that is allowed (workers), spin where it
    // is not (the main thread, which throws).
    try { Atomics.wait(i32, futexLockIdx, 1, 1); } catch { /* main thread: keep spinning */ }
  }
}

export function futexUnlock(i32: Int32Array): void {
  Atomics.store(i32, futexLockIdx, 0);
  Atomics.notify(i32, futexLockIdx, 1);
}

/** queue a waiter. returns its slot, or -1 when the table is full. caller holds the lock. */
export function futexEnqueue(i32: Int32Array, keyLo: number, keyHi: number, bits: number,
                             pid: number, leader: number, grouped = false): number {
  const hint = Atomics.load(i32, futexLockIdx + 1);
  for (let n = 0; n < FUTEX_SLOTS; n++) {
    const slot = (hint + n) % FUTEX_SLOTS;
    const base = futexSlotIdx(slot);
    if (Atomics.load(i32, base + FUTEX_STATE) !== FUTEX_FREE) continue;
    Atomics.store(i32, base + FUTEX_KEYLO, keyLo | 0);
    Atomics.store(i32, base + FUTEX_KEYHI, keyHi | 0);
    Atomics.store(i32, base + FUTEX_BITS, bits | 0);
    Atomics.store(i32, base + FUTEX_PID, pid | 0);
    Atomics.store(i32, base + FUTEX_LEADER, leader < 0 ? slot : leader);
    Atomics.store(i32, base + FUTEX_RESULT, -1);
    Atomics.store(i32, base + FUTEX_GROUPED, grouped ? 1 : 0);
    Atomics.store(i32, base + FUTEX_STATE, FUTEX_QUEUED);
    Atomics.store(i32, futexLockIdx + 1, (slot + 1) % FUTEX_SLOTS);
    // the highest slot ever used, so a scan stops there instead of at the end of the table.
    if (slot > Atomics.load(i32, futexLockIdx + 2)) Atomics.store(i32, futexLockIdx + 2, slot);
    return slot;
  }
  return -1;
}

/**
 * drop a waiter's entry, and the rest of its group with it. caller holds the lock.
 *
 * the group sweep is skipped for an ordinary waiter, which is all but one caller in a thousand:
 * scanning the table per wakeup made every futex operation cost the whole table, and with thirty
 * threads on one lock that is where the time went.
 */
export function futexDequeue(i32: Int32Array, slot: number): void {
  const own = futexSlotIdx(slot);
  const grouped = Atomics.load(i32, own + FUTEX_GROUPED) === 1;
  Atomics.store(i32, own + FUTEX_STATE, FUTEX_FREE);
  if (!grouped) return;
  for (let s = 0; s < FUTEX_SLOTS; s++) {
    const base = futexSlotIdx(s);
    if (Atomics.load(i32, base + FUTEX_LEADER) === slot) Atomics.store(i32, base + FUTEX_STATE, FUTEX_FREE);
  }
}

/**
 * wake up to `max` waiters on a key, and requeue up to `requeue` of the rest onto another key.
 * returns how many were woken plus how many moved, which is what FUTEX_CMP_REQUEUE reports.
 * caller holds the lock.
 */
export function futexWake(i32: Int32Array, keyLo: number, keyHi: number, bits: number, max: number,
                          requeue = 0, reqLo = 0, reqHi = 0): { woke: number; moved: number } {
  let woke = 0, moved = 0;
  // stop as soon as the quota is met. a wake of one -- which is what every uncontended lock does --
  // used to walk the whole table looking for a second waiter it had no use for, and at thousands of
  // futex operations a second that scan IS the cost of the futex.
  const top = Math.min(FUTEX_SLOTS, Atomics.load(i32, futexLockIdx + 2) + 1);
  for (let s = 0; s < top; s++) {
    if (woke >= max && moved >= requeue) break;
    const base = futexSlotIdx(s);
    if (Atomics.load(i32, base + FUTEX_STATE) !== FUTEX_QUEUED) continue;
    if (Atomics.load(i32, base + FUTEX_KEYLO) !== (keyLo | 0)) continue;
    if (Atomics.load(i32, base + FUTEX_KEYHI) !== (keyHi | 0)) continue;
    if ((Atomics.load(i32, base + FUTEX_BITS) & bits) === 0) continue;
    if (woke < max) {
      const leader = Atomics.load(i32, base + FUTEX_LEADER);
      const lbase = futexSlotIdx(leader);
      // the members of a woken group are done with; only the leader's entry survives, holding
      // which futex it was that woke, until the waiter picks it up. an ordinary waiter has no
      // group, and paying a table scan for one is what made a thirty-thread barrier crawl.
      if (Atomics.load(i32, base + FUTEX_GROUPED) === 1) {
        for (let g = 0; g < FUTEX_SLOTS; g++) {
          const gbase = futexSlotIdx(g);
          if (g !== leader && Atomics.load(i32, gbase + FUTEX_LEADER) === leader) {
            Atomics.store(i32, gbase + FUTEX_STATE, FUTEX_FREE);
          }
        }
      }
      Atomics.store(i32, lbase + FUTEX_RESULT, s);
      Atomics.store(i32, lbase + FUTEX_STATE, FUTEX_WOKEN);
      Atomics.notify(i32, lbase + FUTEX_STATE, 1);
      woke++;
      continue;
    }
    if (moved < requeue) {
      Atomics.store(i32, base + FUTEX_KEYLO, reqLo | 0);
      Atomics.store(i32, base + FUTEX_KEYHI, reqHi | 0);
      moved++;
      continue;
    }
  }
  return { woke, moved };
}

/**
 * every queued futex waiter, for the page's inspector. a deadlock in here looks like nothing at all
 * from the guest side -- every thread simply stops -- so being able to ask who is waiting on what
 * is the difference between a bug report and a guess.
 */
export function bindFutexInspector(i32: Int32Array): void {
  if (typeof globalThis === 'undefined') return;
  (globalThis as any).__jitterFutexes = () => {
    const out: Array<Record<string, number>> = [];
    for (let s = 0; s < FUTEX_SLOTS; s++) {
      const base = futexSlotIdx(s);
      const state = Atomics.load(i32, base + FUTEX_STATE);
      if (state === FUTEX_FREE) continue;
      out.push({
        slot: s, state,
        keyLo: Atomics.load(i32, base + FUTEX_KEYLO), keyHi: Atomics.load(i32, base + FUTEX_KEYHI),
        bits: Atomics.load(i32, base + FUTEX_BITS), pid: Atomics.load(i32, base + FUTEX_PID),
        leader: Atomics.load(i32, base + FUTEX_LEADER), grouped: Atomics.load(i32, base + FUTEX_GROUPED),
      });
    }
    return { lock: Atomics.load(i32, futexLockIdx), waiters: out };
  };
}

/** a process that dies while queued leaves entries behind; this is where they go. */
export function futexReleasePid(i32: Int32Array, pid: number): void {
  futexLock(i32);
  for (let s = 0; s < FUTEX_SLOTS; s++) {
    const base = futexSlotIdx(s);
    if (Atomics.load(i32, base + FUTEX_STATE) !== FUTEX_FREE &&
        Atomics.load(i32, base + FUTEX_PID) === pid) {
      Atomics.store(i32, base + FUTEX_STATE, FUTEX_FREE);
    }
  }
  futexUnlock(i32);
}

// WHO sent a pending signal. the pending set is a bitmask, so it already collapses two instances of
// the same signal into one -- which is exactly what linux does for the classic signals -- and one
// sender per (process, signal) is therefore the right granularity. it has to be recorded somewhere:
// a handler installed with SA_SIGINFO reads si_pid to find out who signalled it, and answering 0
// tells it "the kernel", which for kill(2) is a lie with teeth. ltp's kill10 replies to si_pid, so
// a zero there made it signal its own process group in a loop.
// A process's credentials, visible system-wide. kill(2) is only allowed between processes whose
// user ids match (that is the whole reason one user cannot signal another's processes), and the
// sender has to be able to check the TARGET's ids -- which live in the target's own worker. So the
// few that the check needs are published here.
export function procCredOffset(slot: number): number {
  return PROC_CRED_OFFSET + slot * PROC_CRED_WORDS * 4;
}

/** byte offset of the (pid, uid) pair recording who raised `sig` for the process in `slot`. */
/** the first WORD of a process's signal queue, in a whole-buffer Int32Array. */
export function sigqSlotWord(slot: number): number {
  return SIG_SENDER_OFFSET / 4 + slot * SIGQ_SLOT_WORDS;
}
function sigqEntryWord(slot: number, index: number): number {
  return sigqSlotWord(slot) + 2 + (((index % SIGQ_ENTRIES) + SIGQ_ENTRIES) % SIGQ_ENTRIES) * SIGQ_ENTRY_WORDS;
}

/** everything a siginfo carries that a sender gets to choose. */
export interface QueuedSignal {
  signo: number;
  code: number;      // SI_USER(0), SI_KERNEL(0x80), SI_QUEUE(-1), SI_TIMER(-2), SI_TKILL(-6)
  pid: number;
  uid: number;
  value: bigint;     // sigval: the word sigqueue(3) and a posix timer deliver
}

/**
 * queue a signal for the process in `slot`. false means its queue is full, which for a real-time
 * signal is EAGAIN and for a classic one is nothing at all -- a second instance of a signal that
 * is already waiting adds nothing.
 *
 * head and tail are monotonic counters compared with two's-complement subtraction, so they stay
 * correct across the point where an int32 wraps.
 */
export function sigqPush(q: Int32Array, slot: number, info: QueuedSignal): boolean {
  const base = sigqSlotWord(slot);
  for (;;) {
    const head = Atomics.load(q, base + SIGQ_HEAD);
    const tail = Atomics.load(q, base + SIGQ_TAIL);
    if (((head - tail) | 0) >= SIGQ_ENTRIES) return false;
    if (Atomics.compareExchange(q, base + SIGQ_HEAD, head, (head + 1) | 0) !== head) continue;
    const e = sigqEntryWord(slot, head);
    Atomics.store(q, e + 1, info.signo | 0);
    Atomics.store(q, e + 2, info.code | 0);
    Atomics.store(q, e + 3, info.pid | 0);
    Atomics.store(q, e + 4, info.uid | 0);
    Atomics.store(q, e + 5, Number(info.value & 0xffffffffn) | 0);
    Atomics.store(q, e + 6, Number((info.value >> 32n) & 0xffffffffn) | 0);
    // last, and never zero: zero is what an unwritten entry holds.
    Atomics.store(q, e + 0, ((head + 1) | 0) || 1);
    return true;
  }
}

/** how many signals are queued for `slot` and not yet taken. */
export function sigqDepth(q: Int32Array, slot: number): number {
  const d = (Atomics.load(q, sigqSlotWord(slot) + SIGQ_HEAD) -
             Atomics.load(q, sigqSlotWord(slot) + SIGQ_TAIL)) | 0;
  return d > 0 ? d : 0;
}

/** take everything queued for `slot`, oldest first. only the owning process may call this. */
export function sigqDrain(q: Int32Array, slot: number): QueuedSignal[] {
  const base = sigqSlotWord(slot);
  const out: QueuedSignal[] = [];
  for (;;) {
    const tail = Atomics.load(q, base + SIGQ_TAIL);
    if (tail === Atomics.load(q, base + SIGQ_HEAD)) return out;
    const e = sigqEntryWord(slot, tail);
    // a writer stamps the sequence last, so an entry whose sequence has not arrived is one still
    // being written: leave it and come back rather than reading half of it.
    if (Atomics.load(q, e + 0) !== (((tail + 1) | 0) || 1)) return out;
    out.push({
      signo: Atomics.load(q, e + 1),
      code: Atomics.load(q, e + 2),
      pid: Atomics.load(q, e + 3),
      uid: Atomics.load(q, e + 4),
      value: (BigInt(Atomics.load(q, e + 6) >>> 0) << 32n) | BigInt(Atomics.load(q, e + 5) >>> 0),
    });
    Atomics.store(q, e + 0, 0);
    Atomics.store(q, base + SIGQ_TAIL, (tail + 1) | 0);
  }
}

/** empty a slot's queue, so a new process never inherits signals meant for the last one. */
/** is anything waiting for this process? cheap enough for a wait loop to ask every time round. */
export function sigqPending(q: Int32Array, slot: number): boolean {
  const base = sigqSlotWord(slot);
  return Atomics.load(q, base + SIGQ_HEAD) !== Atomics.load(q, base + SIGQ_TAIL);
}

export function sigqReset(q: Int32Array, slot: number): void {
  const base = sigqSlotWord(slot);
  Atomics.store(q, base + SIGQ_HEAD, 0);
  Atomics.store(q, base + SIGQ_TAIL, 0);
  for (let i = 0; i < SIGQ_ENTRIES; i++) Atomics.store(q, sigqEntryWord(slot, i), 0);
}

// Bridge response area: 64 slots × 16KB = 1MB
// Used for passing bridge response data (e.g., network read data) to Workers
// 64 KiB per worker. the slot bounds how much a single bridge round trip can carry, and a round
// trip is a postMessage plus a futex wake -- so reading a 1 MiB binary in 12 KiB pieces cost ~90 of
// them just to start a process. it is shared memory, not a copy per worker: 64 slots is 4 MiB.



export const PIPE_WRITE_POS = 0;     // +0  (Atomics)
export const PIPE_READ_POS = 1;      // +4  (Atomics)
export const PIPE_WRITER_CLOSED = 2; // +8  (Atomics)
export const PIPE_READER_CLOSED = 3; // +12 (Atomics)
// a ring's capacity is a property of THE RING, not of the build: pipes are no longer all the same
// size, so a user that assumed a compile-time constant would index straight off the end of a small
// one and into whatever the arena happened to put next to it. every user reads this word instead.
export const PIPE_CAPACITY_W = 4;    // +16 usable data bytes in this ring
// nonzero when somebody has asked to be signalled when this pipe becomes readable (O_ASYNC).
// the writer checks it on every write, so it has to be free to look at when nobody is using it:
// one relaxed load of a word already in cache, and only then a message to the kernel.
export const PIPE_ASYNC_W = 5;       // +20 async owners registered on the read end
// how much room the ring's ARENA BLOCK actually has. capacity can be moved within this by
// F_SETPIPE_SZ, but never past it -- the block was allocated once and cannot grow in place.
export const PIPE_ALLOC_W = 6;       // +24 data bytes the block can hold
// how many descriptors are queued in flight over this ring (SCM_RIGHTS). a receiver that sees zero
// -- which is every receiver, almost always -- can skip asking the kernel entirely, so ancillary
// data costs nothing on the path that does not use it.
export const PIPE_SCM_W = 7;         // +28 descriptors queued on this ring
export const PIPE_HEADER_SIZE = 32;  // 8 words, all spoken for

// linux's default pipe capacity is exactly 64 KiB, and F_GETPIPE_SZ is a promise about how much can
// be buffered before a writer blocks -- programs size their writes by it -- so the ring has to be
// able to keep that promise rather than merely report it.
//
// but a pipe does not get that capacity unconditionally. linux charges every pipe's pages to the
// user and, once they are past `pipe-user-pages-soft`, hands new pipes ONE page instead of sixteen.
// that is what lets a process open a thousand pipes without a thousand times 64 KiB going missing.
// we need the same rule for the same reason: the arena is shared with MAP_SHARED, and a test that
// opens pipes until it runs out (ltp's pipe06 does exactly that) would otherwise take the arena
// with it and starve everything else in the system.
export const PAGE_SIZE = 4096;
export const PIPE_DEF_PAGES = 16;  // linux PIPE_DEF_BUFFERS -- the 64 KiB every pipe wants
export const PIPE_MIN_PAGES = 1;   // what a pipe gets once the soft budget is spent
export const PIPE_MAX_PAGES = 256; // linux pipe-max-size, the ceiling on F_SETPIPE_SZ

/** total bytes a ring of `pages` data pages occupies, header included. */
export function pipeRingSize(pages: number): number {
  return PIPE_HEADER_SIZE + pages * PAGE_SIZE;
}
export const PIPE_RING_SIZE = pipeRingSize(PIPE_DEF_PAGES);

// storage for every MAP_SHARED region AND every pipe buffer in the system. one allocation for the
// whole app, sized for ipc rather than bulk data.
// the shared arena backs pipes, SysV and posix message bodies, and the per-process block a fork
// stages its pages in. the last of those is why it grew: a forking process holds one, and a
// system with a few dozen of them should not have to choose between forking and piping.
export const SHM_ARENA_SIZE = 64 * 1024 * 1024;
// linux's pipe-user-pages-soft: how many pages pipes may hold at full size before new ones are
// downgraded to a single page. expressed as a fraction of the arena so it stays meaningful if the
// arena is resized. HALF is the point -- whatever pipes do, MAP_SHARED always has the other half,
// so no amount of pipe pressure can make an unrelated mmap fail.
export const PIPE_PAGES_SOFT = SHM_ARENA_SIZE / PAGE_SIZE / 2;



// full argv for each process slot, NUL-separated and NUL-terminated exactly as
// /proc/<pid>/cmdline reports it. 255 usable bytes; a longer command line is truncated, which is
// what the kernel does too (it stops at the end of the mapped arg page).
export function procCmdlineOffset(slot: number): number {
  return PROC_CMDLINE_OFFSET + slot * PROC_CMDLINE_SLOT_SIZE;
}

// ── Process States ──────────────────────────────────────────────────────────

export const PROC_EMPTY = 0;
export const PROC_RUNNING = 1;
export const PROC_ZOMBIE = 2;
// the process is frozen: its image is held by the kernel and its Worker (and with it the whole
// WebAssembly.Memory) has been destroyed. dropping the Memory is the only way to give its bytes
// back -- wasm memory never shrinks and jsc ships no discard, so evicting individual pages inside
// a live memory reclaims nothing. measured on device: fill to the cap, drop every reference, and
// the full amount allocates again.
export const PROC_SWAPPED = 3;
// job control: a process that received SIGSTOP (or SIGTSTP/SIGTTIN/SIGTTOU with no handler) is
// STOPPED, not gone. it holds its slot, its memory and its descriptors, and resumes exactly where
// it was when SIGCONT arrives. the state has to be visible system-wide because the thing that
// resumes it is another process, and the thing that reports it is its parent's wait().
export const PROC_STOPPED = 4;

// ── Process Table Entry Layout (64 bytes, Int32Array indexed) ───────────────
// Each field is a 32-bit int. Index = byte offset / 4.

export const PROC_PID = 0;        // +0
export const PROC_PPID = 1;       // +4
export const PROC_PGID = 2;       // +8
export const PROC_STATE = 3;      // +12
export const PROC_EXIT_CODE = 4;  // +16
export const PROC_WAIT_FUTEX = 5; // +20  — Atomics.wait target for parent waiting on child
export const PROC_SIGNAL = 6;     // +24  — unused: signals travel through the per-process queue
export const PROC_STDIN_RING = 7; // +28  — stdin ring buffer index (-1 = none)
export const PROC_REPARENTED = 8; // +32  — 1 if orphaned + reparented to init (kernel auto-reaps)
export const PROC_TGID = 9;       // +36  — thread-group id (== pid for a normal process/leader)
// linux reports a process parked in a blocking syscall as 'S' in /proc/<pid>/stat, and programs
// wait for exactly that -- ltp's TST_PROCESS_STATE_WAIT spins until a child it just forked shows up
// as sleeping. saying 'R' unconditionally makes every such wait spin forever.
export const PROC_BLOCKED = 15;   // +60  — 1 while parked in a blocking syscall ('S' rather than 'R')
export const PROC_SWAP_REQ = 14;  // +56  — kernel asks the process to freeze itself at its next
                                  //        safe point; it answers with a swap_image message
// +40..+55 — the process name, NUL-padded, in linux's 16-byte TASK_COMM_LEN. lives in the slot
// (rather than beside the cmdline) because /proc/<pid>/stat and ps need it for every process and
// reading it must not cost a message.
export const PROC_COMM = 10;      // 4 words
// a dying process leaves its cpu accounting here for whoever reaps it. wait4(2) hands the parent
// a struct rusage for the child it just reaped, and there is nowhere else for those numbers to
// come from once the worker is gone -- leaving the struct untouched (as we used to) means the
// parent reads whatever was on its own stack, which is how `time` reported 2329 hours of system
// time for a command that ran in 50 milliseconds.
export const PROC_UTIME_MS = 16;  // +64  - cpu time the process spent running guest code
export const PROC_STIME_MS = 17;  // +68  - cpu time it spent inside the syscall layer
// the scheduling nice value, -20..19. it lives in the shared table rather than in the process
// because setpriority(2) and getpriority(2) are defined ACROSS processes: one process sets another
// one's, or a whole process group's, and neither can see the other's private state.
export const PROC_NICE = 18;      // +72
// the signal this process asked to receive when its PARENT dies (prctl PR_SET_PDEATHSIG). the
// kernel is the only thing that can deliver it -- the parent is gone by then -- so it has to be
// readable from the table rather than kept in the process.
export const PROC_PDEATHSIG = 19; // +76

// scheduling state, per THREAD (linux schedules threads, and sched_setattr(0, ...) inside a pthread
// sets that thread's policy). it lives in the shared table because these calls name a pid: one
// process sets another's policy, priority or affinity, and reads it back afterwards.
export const PROC_POLICY = 20;    // +80  - SCHED_OTHER/FIFO/RR/BATCH/IDLE/DEADLINE
export const PROC_RTPRIO = 21;    // +84  - 1..99 under SCHED_FIFO and SCHED_RR, 0 otherwise
export const PROC_SCHED_FLAGS = 22; // +88 - sched_attr's sched_flags (SCHED_FLAG_RESET_ON_FORK etc)
export const PROC_AFFINITY = 23;  // +92  - cpu affinity, 2 words = up to 64 cpus
export const PROC_AFFINITY_WORDS = 2;
// SCHED_DEADLINE's reservation, nanoseconds, each as a (low, high) pair of 32-bit words.
export const PROC_DL_RUNTIME = 25;  // +100
export const PROC_DL_DEADLINE = 27; // +108
export const PROC_DL_PERIOD = 29;   // +116
// which signal stopped it, what its parent has already been told about that stop, and whether a
// SIGCONT has arrived that the parent has not yet collected. wait(WUNTRACED) and wait(WCONTINUED)
// are defined in terms of these three, and each event is reported once.
export const PROC_STOP_SIG = 31;    // +124
export const PROC_STOP_SEEN = 32;   // +128 - 1 once the parent has been told about the stop
export const PROC_CONT_PENDING = 33; // +132 - 1 from SIGCONT until the parent collects it
// which syscall the process is inside right now, or -1 when it is running its own code. linux
// publishes the same thing as /proc/<pid>/syscall, and for the same reason: when a process is
// stuck, the first question is always "stuck in what".
export const PROC_SYSCALL = 34;
// the session a process belongs to, and whether it has exec'd since its parent forked it. both
// are things one process asks about ANOTHER (setpgid checks the target's session; getsid names a
// pid), so they live in the shared table rather than in the process that owns them.
export const PROC_SID = 35;       // +140
export const PROC_EXECED = 36;    // +144
// PR_SET_CHILD_SUBREAPER: this process adopts the orphans of its descendants instead of letting
// them go to init. the kernel walks the ancestor chain looking for it, so it has to be readable
// from outside the process that set it.
export const PROC_SUBREAPER = 37; // +148
// RLIMIT_SIGPENDING, published by the process that owns it. a SENDER needs the TARGET's limit to
// answer EAGAIN, and every other resource limit is private to the process it applies to.
export const PROC_RLIM_SIGPENDING = 38; // +152
// the process tracing this one, or 0. it lives here because wait4 has to know, without asking
// anyone, whether a stopped child is stopped FOR ITS TRACER -- which is reported whether or not
// the caller passed WUNTRACED.
export const PROC_TRACER = 39;    // +156
// what a ptrace-stopped tracee should do when it is let go: PT_RESUME_* below, with the signal to
// deliver (0 = swallow it) in the high bits.
export const PROC_TRACE_CMD = 40; // +160
export const PT_RESUME_NONE = 0, PT_RESUME_CONT = 1, PT_RESUME_DETACH = 2;
// set in a vforked CHILD's slot while its parent is suspended waiting for it. the child clears it
// when it execs or exits, which is exactly when vfork(2) says the parent may run again.
export const PROC_VFORK = 41;     // +164
// the high-water mark of this process's resident size, in kilobytes. a parent that reaps it wants
// ru_maxrss for the child, and the number is gone with the worker by then.
export const PROC_MAXRSS_KB = 42; // +168
// i/o scheduling class and level, the pair ioprio_set/ioprio_get carry. per THREAD, and in the
// shared table for the same reason the sched_* fields are: the calls name a pid rather than the
// caller.
export const PROC_IOPRIO = 43;    // +172
// the real user id, republished at every syscall boundary. a call that names a USER rather than a
// pid (ioprio_set(IOPRIO_WHO_USER, ...)) has no other way to tell whose processes these are.
export const PROC_RUID = 44;      // +176
// +80..+127 reserved

// ── Signal Bits ─────────────────────────────────────────────────────────────

export const SIG_INT = 1 << 2;  // SIGINT (Ctrl+C)
export const SIG_PIPE = 1 << 13; // SIGPIPE (broken pipe)
export const SIG_CHLD = 1 << 17; // SIGCHLD (child exited)

// ── Syscall Mailbox Layout (128 bytes, Int32Array indexed) ──────────────────
// Used for synchronous Worker↔Main thread communication.
// Worker writes request, posts message to main, then Atomics.wait on MBOX_READY.
// Main thread processes, writes response, Atomics.store+notify on MBOX_READY.

export const MBOX_READY = 0;     // +0  — 0=pending, 1=response ready (Atomics.wait target)
export const MBOX_TYPE = 1;      // +4  — request/response type
export const MBOX_VALUE0 = 2;    // +8  — generic value fields
export const MBOX_VALUE1 = 3;    // +12
export const MBOX_VALUE2 = 4;    // +16
export const MBOX_VALUE3 = 5;    // +20
// +24..+127 — payload area for larger data (e.g., serialized state)

export const MBOX_TYPE_NONE = 0;
export const MBOX_TYPE_FORK_RESULT = 1;
export const MBOX_TYPE_PIPE_ALLOC = 2;
export const MBOX_TYPE_BRIDGE_RESP = 3;

// ── Pipe Ring Buffer Layout ─────────────────────────────────────────────────
// Header: 4 × Int32 (16 bytes), then data ring.


// ── Stdin Ring Buffer — same layout as pipe ring ────────────────────────────

export const STDIN_DATA_SIZE = STDIN_RING_SIZE - PIPE_HEADER_SIZE; // 8176 bytes

// ── Global State Layout ─────────────────────────────────────────────────────

export const GLOBAL_NEXT_PID = 0;       // +0  next PID to allocate (Atomics)
export const GLOBAL_NEXT_PIPE = 1;      // +4  next pipe slot to allocate
export const GLOBAL_NEXT_STDIN = 2;     // +8  next stdin ring to allocate
export const GLOBAL_FG_PGID = 3;        // +12 foreground process group ID
// fs.pipe-max-size: the ceiling on any one pipe's capacity. it is a SYSTEM-wide setting that a
// privileged process changes through /proc and every process must then see, so it lives here in
// shared state rather than in each process's own view of /proc.
export const GLOBAL_PIPE_MAX = 4;       // +16 fs.pipe-max-size, in bytes
// a monotonic counter bumped by anything that can make some descriptor ready: a pipe write or
// close, a unix socket connecting or being accepted, stdin arriving. poll/select/epoll read it,
// look at every descriptor they were given, and if none is ready park on THIS word until it
// moves. one futex for all of them, because Atomics.wait can only watch a single address and a
// poller is watching many descriptors at once. wakeups are therefore broadcast: a woken poller
// re-checks its own set and parks again if the event was not for it.
export const GLOBAL_IO_GEN = 5;         // +20 io readiness generation
// how many inotify watches exist anywhere in the system. a filesystem mutation only has to tell
// the kernel about itself when something is actually watching, so with no watches -- the normal
// case -- notification costs one atomic load per mutating syscall and nothing else.
export const GLOBAL_INOTIFY_WATCHES = 6; // +24
// wall-clock milliseconds at which the system booted, split across two words because it does not
// fit one. every worker derives CLOCK_MONOTONIC from it, so a process started an hour in does not
// think the machine is one second old -- and two processes comparing timestamps agree.
export const GLOBAL_BOOT_MS_LO = 7;     // +28
export const GLOBAL_BOOT_MS_HI = 8;     // +32
// the posix message queue limits, live: fs.mqueue.msg_max, msgsize_max and queues_max. they are
// settings rather than constants, and mq_open has to enforce whatever the last write left here.
export const GLOBAL_MQ_MSG_MAX = 9;      // +36
export const GLOBAL_MQ_MSGSIZE_MAX = 10; // +40
export const GLOBAL_MQ_QUEUES_MAX = 11;  // +44
// how far settimeofday/clock_settime has moved the wall clock, as a seqlock: a writer makes the
// sequence odd while it is mid-update, so a reader never sees a half-written offset. the clock is
// the SYSTEM's, not a process's -- a time set in one process has to be the time every other one
// reads, which is exactly what a per-process offset could not do.
export const GLOBAL_CLK_SEQ = 12;        // +48
export const GLOBAL_CLK_OFF_SEC = 13;    // +52
export const GLOBAL_CLK_OFF_NSEC = 14;   // +56
// kernel.msgmni: how many sysv message queues may exist at once. a setting, not a constant --
// lowering it is how an administrator caps them, and msgget has to answer ENOSPC at whatever the
// last write left here.
export const GLOBAL_MSGMNI = 15;         // +60

// the system's UTS names, in the shared page so every process reads the SAME ones without a round
// trip -- a hostname is a property of the machine, not of whoever last asked. each is a
// NUL-padded byte range of __NEW_UTS_LEN + 1, laid out after the counters above.
export const GLOBAL_UTS_LEN = 65;
export const GLOBAL_HOSTNAME_OFFSET = 64;   // bytes into the global page
export const GLOBAL_DOMAINNAME_OFFSET = GLOBAL_HOSTNAME_OFFSET + GLOBAL_UTS_LEN;

// kernel.{msg,sem,shm}_next_id: the id the NEXT sysv object of that kind must be given, or -1 for
// "whatever comes next". checkpoint/restore is what these are for -- an image is only restorable
// if the ids in it can be recreated -- and the value is consumed by the first create that follows.
// placed after the uts names, which are byte ranges rather than words.
export const GLOBAL_MSG_NEXT_ID = 49;    // +196
export const GLOBAL_SEM_NEXT_ID = 50;    // +200
export const GLOBAL_SHM_NEXT_ID = 51;    // +204

// ── Worker Message Types (postMessage protocol) ─────────────────────────────

// A sparse snapshot of a guest linear memory: only the non-zero 64KB pages travel, keyed by
// page index, plus the logical length so the child grows to match. a fresh child memory is all
// zero, so the omitted (zero) pages are already correct — this turns a ~33MB contiguous fork
// copy into "the few MB the process actually touched".
export interface SparseMemSnapshot {
  pages: Uint32Array;   // page indices present, in order
  // the page bodies travel one of two ways. a FORK stages them in the shared arena, in a block the
  // process keeps and reuses -- allocating a buffer per fork made garbage faster than the
  // collector could take it back, and a guest that forked a few hundred times a second reached
  // the web content process's memory limit and had it killed. a SWAP IMAGE is a one-off that may
  // be far larger than the arena, so it still travels as a transferred buffer.
  data?: ArrayBuffer;   // packed bodies, pages.length * MEM_PAGE_SIZE bytes
  stageOff?: number;    // ...or where those same bytes sit in the shared arena
  byteLen?: number;
  totalLen: number;     // logical byte length of the source memory region
}

export const MEM_PAGE_SIZE = 65536; // wasm page; also the sparse-snapshot granularity

export type KernelMessage =
  // Worker → Main
  | { type: 'fork'; memSnapshot: SparseMemSnapshot; syscallState: any; resumePC: string;
      shmBlocks?: number[]; shmMaps?: Array<{ addr: number; len: number; off: number; sysv?: number }> }
  | { type: 'sysv_shm'; op: 'limits' }
  | { type: 'sysv_shm'; op: 'setlimit'; which: number; lo: number; hi: number }
  | { type: 'fork_stage'; bytes: number }
  | { type: 'fork_ready' }
  | { type: 'exit'; pid: number; code: number; high?: number; memBytes?: number } // high = threaded touched high-water; memBytes = this worker's (never-shrinking) memory size
  | { type: 'kill'; pid: number; sig: number }
  | { type: 'write'; fd: number; data: Uint8Array }
  // `priv` is the caller's privilege: fs.pipe-max-size clamps a new pipe for everyone EXCEPT a
  // privileged process, which linux exempts (CAP_SYS_RESOURCE).
  | { type: 'pipe_alloc'; priv?: boolean }
  // a process took or dropped a reference to a pipe end. fire-and-forget: the kernel keeps its own
  // per-process tally so it can release what a process still holds when the process DIES, which the
  // process itself may never get the chance to do.
  | { type: 'pipe_ref'; off: number; mode: 'read' | 'write'; delta: number }
  // O_ASYNC bookkeeping: who wants a signal when this pipe end becomes ready, and which signal.
  // `sig` 0 means the default, SIGIO. `on` false deregisters this process.
  | { type: 'pipe_async'; off: number; mode: 'read' | 'write'; sig: number; on: boolean }
  // a write landed in a ring that has async owners; the kernel delivers to them.
  | { type: 'pipe_io'; off: number; mode: 'read' | 'write' }
  // a syscall this process asked for and we refused (ENOSYS/EINVAL), reported once per pair
  | { type: 'refusal'; nr: number; errno: number }
  // one traced syscall, when tracing is on. answers "what did this program actually do", which
  // black-box probing cannot when every call succeeds and the result is still wrong.
  | { type: 'trace'; line: string }
  // first pthread_create: workers can't allocate shared memory on WKWebView/JSC, so the worker asks
  // the kernel (main thread) to create it. 'clone_thread' -> kernel makes the shared memory + reserves
  // the child tid/reg-block and replies 'promote'; the worker copies its state in then 'thread_go'
  // tells the kernel to actually spawn the thread-Worker on that (now-populated) shared memory.
  | { type: 'clone_thread'; curPages: number }
  | { type: 'thread_go'; tid: number; regBaseAddr: number; startPC: string; childStack: number; tls: number; childTidPtr: number; flags: number; parentRegBase: number }
  // subsequent threads: the kernel already holds the process's shared memory (keyed by tgid), so no
  // memory travels; the caller names its thread-group and the kernel reuses that shared memory.
  | { type: 'spawn_thread'; tgid: number; startPC: string; childStack: number; tls: number; childTidPtr: number; flags: number; parentRegBase: number }
  | { type: 'lock_req'; path: string; cmd: number; ltype: number; start: number; len: number;
      space?: 'record' | 'flock'; scope?: 'posix' | 'ofd'; ownerId?: number }
  | { type: 'lock_close'; path: string; ofdId: number; lastRef: boolean }
  | { type: 'worker_error'; detail: string }
  // a grow the platform refused. the guest already has its ENOMEM; this is for the log only.
  | { type: 'grow_refused'; detail: string }
  | { type: 'lease_req'; op: 'set' | 'get' | 'sig'; path: string; nativeFd: number;
      arg: number; breakSig: number; isOwner: boolean; isRegular: boolean }
  | { type: 'shm_alloc'; len: number; key?: string }
  // SysV shared memory. ids and the IPC_CREAT/IPC_EXCL rules are system-wide, so the registry is
  // the kernel's; the segment's bytes are an ordinary arena block keyed "sysvshm:<id>".
  | { type: 'sysv_shm'; op: 'get'; key: number; size: number; flg: number; uid: number; gid: number }
  | { type: 'sysv_shm'; op: 'attach' | 'detach'; id: number }
  | { type: 'sysv_shm'; op: 'stat' | 'statidx' | 'rm'; id: number; uid: number; gid: number }
  | { type: 'sysv_shm'; op: 'lock'; id: number; on: number; uid: number; gid: number }
  | { type: 'sysv_shm'; op: 'info' }
  | { type: 'sysv_shm'; op: 'set'; id: number; mode: number; uid: number; gid: number }
  // SysV message queues. bodies travel as arena offsets, so 'snd' carries the bytes once and
  // 'rcv' answers with where they landed; the receiver frees the block when it has copied them.
  | { type: 'sysv_msg'; op: 'get'; key: number; flg: number; uid: number; gid: number }
  | { type: 'sysv_msg'; op: 'snd'; id: number; mtype: string; len: number; data: Uint8Array }
  | { type: 'sysv_msg'; op: 'rcv'; id: number; mtype: string; maxlen: number; flg: number }
  | { type: 'sysv_msg'; op: 'free'; off: number; len: number }
  | { type: 'sysv_msg'; op: 'stat' | 'statidx' | 'rm'; id: number }
  | { type: 'sysv_msg'; op: 'info' }
  // posix message queues. the name is what the guest passed, with its leading slash already off.
  | { type: 'posix_mq'; op: 'open'; name: string; oflag: number; accmode: number; mode: number;
      hasAttr: boolean; maxmsg: number; msgsize: number; uid: number; gid: number; pid: number }
  | { type: 'posix_mq'; op: 'ref' | 'close'; id: number; pid: number }
  | { type: 'exec_image'; path: string }
  | { type: 'klog'; line: string }
  | { type: 'eventfd'; op: 'create'; initval: number; semaphore: boolean }
  | { type: 'eventfd'; op: 'ref' | 'close'; id: number; pid: number }
  | { type: 'eventfd'; op: 'peek' | 'read'; id: number }
  | { type: 'eventfd'; op: 'write'; id: number; value: string }
  | { type: 'posix_mq'; op: 'unlink'; name: string; uid: number; gid: number }
  | { type: 'posix_mq'; op: 'send'; id: number; prio: number; len: number; data: Uint8Array }
  | { type: 'posix_mq'; op: 'recv'; id: number; maxlen: number }
  | { type: 'posix_mq'; op: 'free'; off: number; len: number }
  | { type: 'posix_mq'; op: 'getattr'; id: number }
  | { type: 'posix_mq'; op: 'notify'; id: number; pid: number; signo: number; ring?: number; nlFd?: number; cookie?: Uint8Array }
  | { type: 'sysv_sem'; op: 'get'; key: number; nsems: number; flg: number; uid: number; gid: number }
  | { type: 'sysv_sem'; op: 'op'; id: number; ops: number[]; wait: number; uid: number; gid: number }
  | { type: 'sysv_sem'; op: 'unwait'; id: number }
  | { type: 'sysv_sem'; op: 'getone'; id: number; num: number; what: 'val' | 'pid' | 'ncnt' | 'zcnt'; uid: number; gid: number }
  | { type: 'sysv_sem'; op: 'setval'; id: number; num: number; val: number; uid: number; gid: number }
  | { type: 'sysv_sem'; op: 'getall' | 'stat' | 'statidx' | 'rm'; id: number; uid: number; gid: number }
  | { type: 'sysv_sem'; op: 'nsems'; id: number }
  | { type: 'sysv_sem'; op: 'proclist'; which: 'sem' | 'msg' | 'shm' }
  | { type: 'sysv_sem'; op: 'setall'; id: number; vals: number[]; uid: number; gid: number }
  | { type: 'sysv_sem'; op: 'set'; id: number; mode: number; nuid: number; ngid: number; uid: number; gid: number }
  | { type: 'sysv_sem'; op: 'free'; off: number; len: number }
  | { type: 'sysv_sem'; op: 'info' }
  // inotify. the watch registry is system-wide; events are packed into an arena block for the
  // reader, which frees the block once it has copied them out.
  | { type: 'inotify'; op: 'init' | 'close' | 'pending'; id: number }
  | { type: 'inotify'; op: 'add'; id: number; path: string; mask: number }
  | { type: 'inotify'; op: 'rm'; id: number; wd: number }
  | { type: 'inotify'; op: 'read'; id: number; maxlen: number }
  | { type: 'inotify'; op: 'free'; id: number; off: number; len: number }
  | { type: 'inotify_event'; path: string; mask: number; cookie?: number }
  // reboot(2) asking for the guest system to stop. `restart` distinguishes a reboot from a halt;
  // there is nothing to restart INTO, so both end the same way -- the flag is for the message.
  | { type: 'halt'; restart: boolean }
  | { type: 'sysv_msg'; op: 'set'; id: number; mode: number; uid: number; gid: number; qbytes: number }
  // AF_UNIX rendezvous. the kernel owns the name -> listener map and the pending-connection
  // queues; the byte streams themselves are ordinary rings in the shared arena, one per direction.
  // uid/gid are the CALLER's, carried so the kernel can answer SO_PEERCRED later.
  | { type: 'unix_bind'; path: string; uid: number; gid: number; dgram?: boolean }
  | { type: 'unix_lookup'; path: string }
  | { type: 'unix_unbind'; path: string }
  | { type: 'unix_listen'; path: string }
  | { type: 'unix_connect'; path: string; uid: number; gid: number }
  | { type: 'unix_accept'; path: string }
  | { type: 'unix_pending'; path: string }
  // socketpair(2): a nameless connected pair, so no path is involved
  | { type: 'unix_pair' }
  | { type: 'nl_socket' }
  | { type: 'shm_free'; off: number }
  | { type: 'lock_cancel' }
  | { type: 'bridge_call'; id: number; method: string; args: Record<string, unknown> }
  | { type: 'jit_compiled'; pc: string; wasmBytes: Uint8Array }
  // Main → Worker
  | { type: 'init'; config: WorkerInitConfig }
  | { type: 'fork_result'; childPid: number }
  | { type: 'pipe_result'; pipeSlot: number }
  | { type: 'bridge_response'; id: number; resp: unknown }
  | { type: 'promote'; sharedMemory: WebAssembly.Memory; tid: number; regBaseAddr: number; staleHigh: number; layout: ThreadLayout }
  | { type: 'clone_fail' } // the kernel couldn't create the shared memory / slot -> clone returns EAGAIN
  | { type: 'stdin_data' } // notification only; data is in the stdin ring buffer
  | { type: 'signal'; sig: number }
  | { type: 'term_size'; cols: number; rows: number }
  // a frozen process handing its whole life back to the kernel, so the Worker (and the wasm
  // memory it owns) can be destroyed
  // a process's committed wasm bytes, reported as it grows. jsc counts shared and non-shared
  // against one 4 GiB total, so budgeting on the shared reservation alone under-counts.
  // `bytes` is the process's real FOOTPRINT (image, stack, heap, mappings); `grown` is how far its
  // wasm memory has been extended. they are very different numbers -- growth is lazy -- and the
  // budget wants the first while the worker-pooling decision wants the second.
  | { type: 'mem_size'; bytes: number; grown?: number }
  | { type: 'swap_image'; memSnapshot: SparseMemSnapshot; syscallState: unknown;
      kernelPipesSnapshot: unknown; unixSocksSnapshot?: unknown; shmMaps?: Array<{ addr: number; len: number; off: number; sysv?: number }>;
      protArmed?: boolean };

export interface WorkerInitConfig {
  /** this is a thaw, not a fork: the snapshot is the process's own, so it keeps its identity,
   *  its descriptors and its return value, and resumes at the pc the freeze recorded */
  resume?: boolean;
  pid: number;
  ppid: number;
  pgid: number;
  slot: number;
  kernelSAB: SharedArrayBuffer;
  // authoritative storage for MAP_SHARED regions; see shared-map.ts
  shmArena: SharedArrayBuffer;
  // for fork: the parent's shared mappings, which the child goes on sharing
  shmMaps?: Array<{ addr: number; len: number; off: number; sysv?: number }>;
  // for fork: whether the parent had the page-protection guard armed. the table itself lives in
  // guest memory and is copied with it, but the "is it on" flag is a wasm global, which a fresh
  // instance starts at zero -- so without this the child inherits the protections and ignores them.
  protArmed?: boolean;
  elfBytes: Uint8Array;
  argv: string[];
  envp: string[];
  // For fork: parent's sparse memory snapshot to restore
  memSnapshot?: SparseMemSnapshot;
  // Serialized syscall state (brk, mmapBase, cwd, fdTable descriptors)
  syscallState?: unknown;
  // For fork: kernel pipe map inherited from parent
  inotifyIds?: number[];
  kernelPipesSnapshot?: Array<[number, { pipeOff: number; mode: 'read' | 'write' }]>;
  // the unix sockets a forked child inherits: which rings each descriptor reads and writes.
  unixSocksSnapshot?: Array<[number, { path?: string; listening: boolean; rx?: number; tx?: number;
                                       dgram?: boolean; peer?: { pid: number; uid: number; gid: number } }]>;
  // For fork: clone(2)'s child-context arguments. a fork-style clone (SIGCHLD, no CLONE_THREAD) can
  // still carry a stack and a tls -- vfork, posix_spawn and system() all issue exactly that -- and
  // the child has to start on them. bigints travel as strings through structured clone.
  cloneFlags?: number;
  childStack?: string;
  childTls?: string;
  childCtid?: string;
  // Cached JIT modules from parent (compile once, instantiate per-worker)
  jitModules?: Array<{ pc: string; wasmBytes: ArrayBuffer }>;
  termCols?: number;
  termRows?: number;
  // translator.wasm bytes — when present, the worker uses the zig translator with ts fallback
  zigTranslatorBytes?: ArrayBuffer;
  // pre-compiled translator module — the kernel compiles translator.wasm once and shares it with
  // every worker (structured-cloneable, shared code) so forks skip the per-worker 1.5MB compile.
  zigTranslatorModule?: WebAssembly.Module;
  // For CLONE_THREAD: this worker is a THREAD of an existing process. it imports the shared memory
  // instead of creating its own, runs at its own register-block base, and starts mid-program.
  threadStart?: {
    sharedMemory: WebAssembly.Memory;
    startPC: string;       // where the thread begins (just after the parent's clone)
    childStack: number;    // guest SP for the thread
    tls: number;           // guest TPIDR_EL0 for the thread
    tid: number;           // this thread's tid
    tgid: number;          // thread-group id (== the process pid)
    regBaseAddr: number;   // linear-memory base of this thread's register block
    parentRegBase: number; // copy the parent thread's registers as the starting state
    childTidPtr: number;   // CLONE_CHILD_SETTID / CLONE_CHILD_CLEARTID address (0 if unset)
    flags: number;
    layout?: ThreadLayout; // this group's address-space tier (register-block + shadow placement)
    syscallState?: unknown; // the parent's descriptor table, so the new thread shares it
    unixSocksSnapshot?: Array<readonly [number, any]>; // unix/netlink sockets, refs already taken
    shmMaps?: Array<{ addr: number; len: number; off: number; sysv?: number }>; // the group's MAP_SHARED regions
  };
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/** Get the byte offset of a process table slot in the kernel SAB. */
export function procSlotOffset(slot: number): number {
  return PROC_TABLE_OFFSET + slot * PROC_SLOT_SIZE;
}

/** Get the byte offset of a mailbox slot in the kernel SAB. */
export function mailboxSlotOffset(slot: number): number {
  return MAILBOX_OFFSET + slot * MAILBOX_SLOT_SIZE;
}

/** Get the byte offset of a pipe ring buffer in the kernel SAB. */

/** Get the byte offset of a stdin ring buffer in the kernel SAB. */
export function stdinRingOffset(stdinIndex: number): number {
  return STDIN_RING_OFFSET + stdinIndex * STDIN_RING_SIZE;
}

// ── IO readiness notification ───────────────────────────────────────────────
// bound once per thread to the kernel globals array. it lives here rather than in either the
// kernel or the worker because both sides produce events and PipeRing (which neither owns) has
// to be able to raise one.

let ioGenArr: Int32Array | null = null;

export function bindIoGeneration(globals: Int32Array): void {
  ioGenArr = globals;
}

/** announce that some descriptor may have become ready. safe to over-call. */
export function bumpIoGeneration(): void {
  if (!ioGenArr) return;
  Atomics.add(ioGenArr, GLOBAL_IO_GEN, 1);
  Atomics.notify(ioGenArr, GLOBAL_IO_GEN);
}

export function readIoGeneration(): number {
  return ioGenArr ? Atomics.load(ioGenArr, GLOBAL_IO_GEN) : 0;
}

/**
 * park until the generation moves past `seen`, or the timeout expires. only a Worker may call
 * this. returns false when it timed out with nothing new.
 */
export function waitIoGeneration(seen: number, timeoutMs: number): boolean {
  if (!ioGenArr) return false;
  const r = timeoutMs < 0
    ? Atomics.wait(ioGenArr, GLOBAL_IO_GEN, seen)
    : Atomics.wait(ioGenArr, GLOBAL_IO_GEN, seen, timeoutMs);
  return r !== 'timed-out';
}
