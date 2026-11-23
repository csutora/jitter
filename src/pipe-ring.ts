/**
 * Lock-free ring buffer for inter-Worker pipe and stdin IPC.
 *
 * Operates on a region of SharedArrayBuffer using Atomics for synchronization.
 * Writers never block (they return short writes if the ring is full).
 * Readers block via Atomics.wait() when the ring is empty.
 *
 * Layout (16 bytes header + data ring):
 *   +0  i32  write_pos  (Atomics — monotonically increasing byte offset)
 *   +4  i32  read_pos   (Atomics — monotonically increasing byte offset)
 *   +8  i32  writer_closed  (0 or 1)
 *   +12 i32  reader_closed  (0 or 1)
 *   +16 ...  data ring  (dataSize bytes, accessed with modular indexing)
 *
 * Positions are monotonically increasing; the actual ring index is pos % dataSize.
 * This avoids the full/empty ambiguity of same-value head/tail pointers.
 */

import {
  PIPE_WRITE_POS, PIPE_READ_POS, PIPE_WRITER_CLOSED, PIPE_READER_CLOSED,
  PIPE_CAPACITY_W, PIPE_ALLOC_W, PIPE_SCM_W, PIPE_HEADER_SIZE, bumpIoGeneration,
} from './kernel-ipc.js';

export class PipeRing {
  private readonly i32: Int32Array;
  private readonly data: Uint8Array;
  private readonly dataSize: number;
  private readonly baseOffset: number;

  /**
   * @param sab   The kernel SharedArrayBuffer (or the shared arena)
   * @param offset  Byte offset of this ring's header within it
   * @param totalSize  Total size of this ring (header + data). Pass it when CREATING the ring;
   *   omit it to adopt an existing one, whose size is recorded in its own header. Pipes are not all
   *   the same size any more, so a caller that guessed from a build constant would run off the end
   *   of a small ring -- omitting the argument is both shorter and the safe default.
   */
  constructor(sab: SharedArrayBuffer, offset: number, totalSize?: number) {
    this.baseOffset = offset;
    const hdr = new Int32Array(sab, offset, PIPE_HEADER_SIZE / 4);
    this.dataSize = totalSize !== undefined
      ? totalSize - PIPE_HEADER_SIZE
      : Atomics.load(hdr, PIPE_CAPACITY_W);
    if (!(this.dataSize > 0)) {
      throw new RangeError(`PipeRing at ${offset}: ring has no capacity (adopted before reset?)`);
    }
    this.i32 = hdr;
    this.data = new Uint8Array(sab, offset + PIPE_HEADER_SIZE, this.dataSize);
  }

  /** usable data bytes -- what F_GETPIPE_SZ reports for this particular pipe. */
  get capacity(): number { return this.dataSize; }

  /** the monotonic stream positions. ancillary data (SCM_RIGHTS) is delivered with the byte it was
   *  sent alongside, so both ends need to name a place in the stream, not just a count. */
  get writePos(): number { return Atomics.load(this.i32, PIPE_WRITE_POS); }
  get readPos(): number { return Atomics.load(this.i32, PIPE_READ_POS); }

  /** how many descriptors are queued in flight over this ring, and how to change that count. */
  get scmPending(): number { return Atomics.load(this.i32, PIPE_SCM_W); }
  addScm(n: number): void { Atomics.add(this.i32, PIPE_SCM_W, n); }

  /** how much the underlying block can hold; capacity may be set anywhere up to this. */
  get allocated(): number { return Atomics.load(this.i32, PIPE_ALLOC_W) || this.dataSize; }

  /**
   * Change this pipe's capacity (F_SETPIPE_SZ). Returns the size actually set, or a negative errno.
   *
   * The ring's indexing is modular in the capacity, so every buffered byte's position depends on it:
   * changing the number while data is in flight would move the data out from under whoever is
   * reading it. Linux resizes by copying under the pipe mutex; having no such lock, this refuses a
   * resize of a pipe with anything buffered -- EBUSY, which is the same answer linux gives when the
   * request cannot be satisfied without losing data. A program that sizes its pipe does it on a
   * fresh one, before any bytes flow, which is exactly the case that works.
   */
  resize(bytes: number): number {
    const room = this.allocated;
    const want = Math.min(bytes, room);
    if (want < 4096) return -22;                       // EINVAL: below one page
    if (this.available !== 0) return -16;              // EBUSY: cannot move buffered data safely
    Atomics.store(this.i32, PIPE_WRITE_POS, 0);
    Atomics.store(this.i32, PIPE_READ_POS, 0);
    Atomics.store(this.i32, PIPE_CAPACITY_W, want);
    return want;
  }

  /** Reset the ring to empty state with the given number of open reader/writer ends. */
  reset(writers = 1, readers = 1): void {
    Atomics.store(this.i32, PIPE_WRITE_POS, 0);
    Atomics.store(this.i32, PIPE_READ_POS, 0);
    // Store open writer/reader counts (0 = closed/EOF)
    Atomics.store(this.i32, PIPE_WRITER_CLOSED, writers);
    Atomics.store(this.i32, PIPE_READER_CLOSED, readers);
    // publish the size so everyone who later adopts this ring agrees with us about where it ends,
    // and the block's real extent so a later resize knows how far it may go
    Atomics.store(this.i32, PIPE_CAPACITY_W, this.dataSize);
    Atomics.store(this.i32, PIPE_ALLOC_W, this.dataSize);
    // a recycled ring carries nothing over from whoever had it before.
    Atomics.store(this.i32, PIPE_SCM_W, 0);
  }

  /** Add a writer reference (e.g., after fork inherits the write end). */
  addWriter(): void { Atomics.add(this.i32, PIPE_WRITER_CLOSED, 1); }

  /** Add a reader reference (e.g., after fork inherits the read end). */
  addReader(): void { Atomics.add(this.i32, PIPE_READER_CLOSED, 1); }

  // ── Writer API ──────────────────────────────────────────────────────────

  /**
   * Write data into the ring. Returns the number of bytes written.
   * Non-blocking: returns 0 if the ring is full.
   * Returns -1 (EPIPE) if the reader has closed.
   */
  write(src: Uint8Array): number {
    if (Atomics.load(this.i32, PIPE_READER_CLOSED) === 0) return -1; // all readers closed

    const wp = Atomics.load(this.i32, PIPE_WRITE_POS);
    const rp = Atomics.load(this.i32, PIPE_READ_POS);

    // the whole buffer is usable. the reserved byte here was the classic trick for telling a full
    // ring from an empty one when head and tail are plain indices -- but these positions are
    // MONOTONIC (see the header note), so used === dataSize already means full and used === 0
    // already means empty. keeping the gap only made the pipe one byte smaller than the capacity
    // it advertises, which a writer filling the pipe exactly notices as a short write.
    const used = wp - rp;
    const available = this.dataSize - used;
    if (available <= 0) return 0;

    const toWrite = Math.min(src.length, available);
    const ringPos = wp % this.dataSize;

    // Write with wrap-around
    const firstChunk = Math.min(toWrite, this.dataSize - ringPos);
    this.data.set(src.subarray(0, firstChunk), ringPos);
    if (toWrite > firstChunk) {
      this.data.set(src.subarray(firstChunk, toWrite), 0);
    }

    // Update write position (release semantics — readers see data before new wp)
    Atomics.store(this.i32, PIPE_WRITE_POS, wp + toWrite);
    // Wake any reader waiting for data. the targeted notify releases a reader parked on this
    // exact ring; the generation bump releases pollers, who are parked on one shared word
    // because they are watching several descriptors at once and cannot park on all of them.
    Atomics.notify(this.i32, PIPE_WRITE_POS);
    bumpIoGeneration();

    return toWrite;
  }

  /**
   * Blocking write: keep writing `src` until it is fully in the ring, parking on reader progress
   * (PIPE_READ_POS) whenever the ring is full. `waiter` is the interruptible park (same one the
   * blocking read uses): it returns true if a signal interrupted the wait.
   * Returns bytes written (== src.length on success); if nothing could be written it returns
   * -1 (all readers closed -> EPIPE) or -2 (interrupted -> EINTR). A partial write that then hits
   * reader-close/interrupt returns the partial count, per posix.
   */
  writeAll(src: Uint8Array, waiter: (i32: Int32Array, idx: number, expected: number) => boolean): number {
    let written = 0;
    while (written < src.length) {
      const n = this.write(src.subarray(written));
      if (n === -1) return written > 0 ? written : -1; // readers gone: partial, else EPIPE
      written += n;
      if (written >= src.length) break;
      // ring full: wait for a reader to consume before writing the rest. load rp *after* the write
      // filled the ring so a reader that advances it wakes us (no lost wakeup).
      const rp = Atomics.load(this.i32, PIPE_READ_POS);
      const wp = Atomics.load(this.i32, PIPE_WRITE_POS);
      if (this.dataSize - (wp - rp) <= 0) { // same capacity rule as write()
        if (waiter(this.i32, PIPE_READ_POS, rp)) return written > 0 ? written : -2; // interrupted
      }
    }
    return written;
  }

  /** room left in the ring: how much a write could take without blocking. */
  get freeBytes(): number {
    return this.dataSize - (Atomics.load(this.i32, PIPE_WRITE_POS) - Atomics.load(this.i32, PIPE_READ_POS));
  }

  /** bytes sitting in the ring right now, which is what FIONREAD reports. */
  get bytesAvailable(): number {
    return Atomics.load(this.i32, PIPE_WRITE_POS) - Atomics.load(this.i32, PIPE_READ_POS);
  }

  /** Decrement writer refcount. Last close triggers EOF for readers. */
  closeWriter(): void {
    const remaining = Atomics.sub(this.i32, PIPE_WRITER_CLOSED, 1) - 1;
    if (remaining <= 0) {
      Atomics.store(this.i32, PIPE_WRITER_CLOSED, 0);
      Atomics.notify(this.i32, PIPE_WRITE_POS);  // wake readers
      bumpIoGeneration();  // and pollers, for whom writer-close reads as readable (EOF)
    }
  }

  /** Check if all read ends have been closed (broken pipe). */
  get readerClosed(): boolean {
    return Atomics.load(this.i32, PIPE_READER_CLOSED) === 0;
  }

  // ── Reader API ──────────────────────────────────────────────────────────

  /**
   * Read data from the ring into dst. Returns the number of bytes read.
   * Returns 0 for EOF (writer closed and ring empty).
   * Returns -1 if `waiter` reported an interruption (EINTR — a pending signal).
   * Blocks via Atomics.wait() (or `waiter`) if the ring is empty and writer is still open.
   *
   * `waiter`, when given, replaces the bare Atomics.wait: it parks on the passed futex and
   * returns true if the wait should be abandoned because a signal is pending. this keeps the
   * signal machinery in the Worker while letting the ring own the empty/EOF logic. without it
   * the read blocks indefinitely, which is un-interruptible by anything but SIGKILL.
   *
   * IMPORTANT: Only call from a Worker thread (Atomics.wait blocks the thread).
   */
  read(dst: Uint8Array, waiter?: (i32: Int32Array, idx: number, expected: number) => boolean): number {
    while (true) {
      const wp = Atomics.load(this.i32, PIPE_WRITE_POS);
      const rp = Atomics.load(this.i32, PIPE_READ_POS);
      const available = wp - rp;

      if (available > 0) {
        const toRead = Math.min(dst.length, available);
        const ringPos = rp % this.dataSize;

        // Read with wrap-around
        const firstChunk = Math.min(toRead, this.dataSize - ringPos);
        dst.set(this.data.subarray(ringPos, ringPos + firstChunk));
        if (toRead > firstChunk) {
          dst.set(this.data.subarray(0, toRead - firstChunk), firstChunk);
        }

        // Update read position
        Atomics.store(this.i32, PIPE_READ_POS, rp + toRead);
        // Wake any writer waiting for space, and pollers watching for writability
        Atomics.notify(this.i32, PIPE_READ_POS);
        bumpIoGeneration();

        return toRead;
      }

      // Ring is empty — check if all writers have closed
      if (Atomics.load(this.i32, PIPE_WRITER_CLOSED) === 0) {
        return 0; // EOF — no more writers
      }

      // Block until writer adds data or closes. wp hasn't changed, so wait for it to change.
      if (waiter) {
        if (waiter(this.i32, PIPE_WRITE_POS, wp)) return -1; // interrupted by a signal
      } else {
        Atomics.wait(this.i32, PIPE_WRITE_POS, wp);
      }
    }
  }

  /** does this pipe still have a writer? distinguishes "empty for now" (EAGAIN) from EOF. */
  writersOpen(): boolean { return Atomics.load(this.i32, PIPE_WRITER_CLOSED) > 0; }

  /**
   * Non-blocking read. Returns 0 if empty (not EOF — caller must check writerClosed).
   * Safe to call from the main thread.
   */
  tryRead(dst: Uint8Array): number {
    const wp = Atomics.load(this.i32, PIPE_WRITE_POS);
    const rp = Atomics.load(this.i32, PIPE_READ_POS);
    const available = wp - rp;

    if (available <= 0) return 0;

    const toRead = Math.min(dst.length, available);
    const ringPos = rp % this.dataSize;

    const firstChunk = Math.min(toRead, this.dataSize - ringPos);
    dst.set(this.data.subarray(ringPos, ringPos + firstChunk));
    if (toRead > firstChunk) {
      dst.set(this.data.subarray(0, toRead - firstChunk), firstChunk);
    }

    Atomics.store(this.i32, PIPE_READ_POS, rp + toRead);
    Atomics.notify(this.i32, PIPE_READ_POS);
    bumpIoGeneration();

    return toRead;
  }

  /**
   * copy bytes out WITHOUT consuming them.
   *
   * tee(2) duplicates a pipe's contents into another pipe and leaves the original readable, so it
   * needs a read that does not move the read pointer. splice needs it too, for a different reason:
   * it must not take bytes out of the pipe until it knows the destination accepted them, or a
   * short write silently loses whatever did not fit.
   */
  peek(dst: Uint8Array): number {
    const wp = Atomics.load(this.i32, PIPE_WRITE_POS);
    const rp = Atomics.load(this.i32, PIPE_READ_POS);
    const available = wp - rp;
    if (available <= 0) return 0;
    const toRead = Math.min(dst.length, available);
    const ringPos = rp % this.dataSize;
    const firstChunk = Math.min(toRead, this.dataSize - ringPos);
    dst.set(this.data.subarray(ringPos, ringPos + firstChunk));
    if (toRead > firstChunk) dst.set(this.data.subarray(0, toRead - firstChunk), firstChunk);
    return toRead;
  }

  /** consume `n` bytes that a previous peek already copied out. */
  discard(n: number): number {
    const wp = Atomics.load(this.i32, PIPE_WRITE_POS);
    const rp = Atomics.load(this.i32, PIPE_READ_POS);
    const dropped = Math.min(n, Math.max(0, wp - rp));
    if (dropped <= 0) return 0;
    Atomics.store(this.i32, PIPE_READ_POS, rp + dropped);
    Atomics.notify(this.i32, PIPE_READ_POS);
    return dropped;
  }

  /** Decrement reader refcount. Last close signals broken pipe to writers. */
  closeReader(): void {
    const remaining = Atomics.sub(this.i32, PIPE_READER_CLOSED, 1) - 1;
    if (remaining <= 0) {
      Atomics.store(this.i32, PIPE_READER_CLOSED, 0);
      Atomics.notify(this.i32, PIPE_READ_POS);  // wake writers (broken pipe)
      bumpIoGeneration();
    }
  }

  /** Check if all write ends have been closed (EOF for readers). */
  get writerClosed(): boolean {
    return Atomics.load(this.i32, PIPE_WRITER_CLOSED) === 0;
  }

  /** Number of bytes currently in the ring. */
  get available(): number {
    return Atomics.load(this.i32, PIPE_WRITE_POS) - Atomics.load(this.i32, PIPE_READ_POS);
  }

  /** Whether the ring is empty. */
  get empty(): boolean {
    return this.available === 0;
  }
}
