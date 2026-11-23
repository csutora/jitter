// MAP_SHARED memory between processes.
//
// every guest process is a Web Worker with its OWN WebAssembly.Memory, and wasm gives no way to
// alias one page of a memory into another (no multi-memory in Safari). so two processes physically
// cannot write the same bytes, and a MAP_SHARED mapping has to be reconstructed rather than
// pointed at: the authoritative copy lives in a SharedArrayBuffer arena that every worker holds,
// and each process reconciles its own view against it.
//
// THE MODEL, stated plainly: a shared mapping is coherent AT SYSCALL BOUNDARIES. a process
// publishes its writes and observes everyone else's on entry to and exit from every syscall. that
// covers every way real programs actually synchronise -- pipes, signals, waitpid, futex, file
// locks are all syscalls -- and it is what lets ltp's harness collect results from its children.
// what it does NOT give is visibility to a process spinning on a shared word without ever making
// a syscall; such a loop would never see the update. that is the honest boundary of this design,
// and lifting it needs all processes to share one wasm memory.
//
// reconciliation is a three-way merge against a shadow of what we last agreed on, so it is
// symmetric: whoever wrote a word wins, and a word nobody touched stays put. without the shadow
// there would be no way to tell "I wrote this" from "this is stale".

/**
 * Reconcile one region. `mine` is the process's view (guest memory), `arena` the shared copy,
 * `shadow` what the two last agreed on. All three are the same length, and `shadow` is updated
 * in place to the merged result.
 */
export function reconcile(mine: Uint8Array, arena: Uint8Array, shadow: Uint8Array): void {
  const n = mine.length;
  // word-at-a-time over the aligned body, byte-at-a-time over any tail
  const words = n >>> 2;
  const m32 = new Uint32Array(mine.buffer, mine.byteOffset, words);
  const a32 = new Uint32Array(arena.buffer, arena.byteOffset, words);
  const s32 = new Uint32Array(shadow.buffer, shadow.byteOffset, words);

  for (let i = 0; i < words; i++) {
    const mv = m32[i];
    const sv = s32[i];
    if (mv !== sv) {          // we wrote it: publish
      a32[i] = mv;
      s32[i] = mv;
    } else {
      const av = a32[i];
      if (av !== sv) {        // someone else wrote it: observe
        m32[i] = av;
        s32[i] = av;
      }
    }
  }

  for (let i = words << 2; i < n; i++) {
    const mv = mine[i];
    const sv = shadow[i];
    if (mv !== sv) { arena[i] = mv; shadow[i] = mv; }
    else if (arena[i] !== sv) { mine[i] = arena[i]; shadow[i] = arena[i]; }
  }
}

/** A live MAP_SHARED mapping in one process. */
export interface SharedMapping {
  /** guest address of the mapping (not yet offset by GUEST_MEM_BASE) */
  addr: number;
  len: number;
  /** byte offset of this region's authoritative copy inside the arena */
  arenaOff: number;
  /** what this process and the arena last agreed on */
  shadow: Uint8Array;
}

/**
 * A bump allocator with free-list reuse for the shared arena. Lives on the kernel side; workers
 * only ever see offsets. Regions are page-granular so a partial unmap cannot split a block into
 * something unallocatable.
 */
export class ArenaAllocator {
  private next = 0;
  private free: Array<{ off: number; len: number }> = [];

  constructor(private readonly size: number, private readonly pageSize = 4096) {}

  /** returns a byte offset, or -1 when the arena is exhausted (the caller reports ENOMEM). */
  alloc(bytes: number): number {
    const len = Math.ceil(bytes / this.pageSize) * this.pageSize;
    for (let i = 0; i < this.free.length; i++) {
      const b = this.free[i];
      if (b.len < len) continue;
      const off = b.off;
      if (b.len === len) this.free.splice(i, 1);
      else { b.off += len; b.len -= len; }
      return off;
    }
    if (this.next + len > this.size) return -1;
    const off = this.next;
    this.next += len;
    return off;
  }

  release(off: number, bytes: number): void {
    const len = Math.ceil(bytes / this.pageSize) * this.pageSize;
    // releasing a block twice would put it in the free list twice, and the next two allocations
    // would be handed the SAME memory -- a pipe and a shared mapping on one block, each destroying
    // the other. a leak is survivable; that is not. so the double release is dropped here, loudly,
    // rather than trusted.
    for (const b of this.free) {
      if (off < b.off + b.len && off + len > b.off) {
        this.doubleReleases++;
        return;
      }
    }
    this.free.push({ off, len });
    this.coalesce();
  }

  /** how many releases named memory that was already free. anything but zero is a refcount bug. */
  doubleReleases = 0;

  /** join neighbouring free blocks so a long-lived process cannot fragment the arena to death. */
  private coalesce(): void {
    this.free.sort((a, b) => a.off - b.off);
    const out: Array<{ off: number; len: number }> = [];
    for (const b of this.free) {
      const prev = out[out.length - 1];
      if (prev && prev.off + prev.len === b.off) prev.len += b.len;
      else out.push({ ...b });
    }
    // hand back any block that reaches the bump pointer, so sequential map/unmap does not creep
    while (out.length && out[out.length - 1].off + out[out.length - 1].len === this.next) {
      this.next = out[out.length - 1].off;
      out.pop();
    }
    this.free = out;
  }

  /** for tests and diagnostics: bytes handed out and not yet released. */
  used(): number {
    return this.next - this.free.reduce((n, b) => n + b.len, 0);
  }
}
