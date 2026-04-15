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
  PIPE_HEADER_SIZE,
} from './kernel-ipc.js';

export class PipeRing {
  private readonly i32: Int32Array;
  private readonly data: Uint8Array;
  private readonly dataSize: number;
  private readonly baseOffset: number;

  /**
   * @param sab   The kernel SharedArrayBuffer
   * @param offset  Byte offset of this ring's header within the SAB
   * @param totalSize  Total size of this ring (header + data)
   */
  constructor(sab: SharedArrayBuffer, offset: number, totalSize: number) {
    this.baseOffset = offset;
    this.dataSize = totalSize - PIPE_HEADER_SIZE;
    // Int32Array view over the 4-int header
    this.i32 = new Int32Array(sab, offset, 4);
    // Uint8Array view over the data region
    this.data = new Uint8Array(sab, offset + PIPE_HEADER_SIZE, this.dataSize);
  }

  /** Reset the ring to empty state with the given number of open reader/writer ends. */
  reset(writers = 1, readers = 1): void {
    Atomics.store(this.i32, PIPE_WRITE_POS, 0);
    Atomics.store(this.i32, PIPE_READ_POS, 0);
    // Store open writer/reader counts (0 = closed/EOF)
    Atomics.store(this.i32, PIPE_WRITER_CLOSED, writers);
    Atomics.store(this.i32, PIPE_READER_CLOSED, readers);
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

    // Available space: dataSize - 1 - (wp - rp), capped at dataSize - 1
    // (we keep 1 byte gap to distinguish full from empty)
    const used = wp - rp;
    const available = this.dataSize - 1 - used;
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
    // Wake any reader waiting for data
    Atomics.notify(this.i32, PIPE_WRITE_POS);

    return toWrite;
  }

  /** Decrement writer refcount. Last close triggers EOF for readers. */
  closeWriter(): void {
    const remaining = Atomics.sub(this.i32, PIPE_WRITER_CLOSED, 1) - 1;
    if (remaining <= 0) {
      Atomics.store(this.i32, PIPE_WRITER_CLOSED, 0);
      Atomics.notify(this.i32, PIPE_WRITE_POS);  // wake readers
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
   * Blocks via Atomics.wait() if the ring is empty and writer is still open.
   *
   * IMPORTANT: Only call from a Worker thread (Atomics.wait blocks the thread).
   */
  read(dst: Uint8Array): number {
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
        // Wake any writer waiting for space (future: back-pressure)
        Atomics.notify(this.i32, PIPE_READ_POS);

        return toRead;
      }

      // Ring is empty — check if all writers have closed
      if (Atomics.load(this.i32, PIPE_WRITER_CLOSED) === 0) {
        return 0; // EOF — no more writers
      }

      // Block until writer adds data or closes
      // wp hasn't changed, so wait for it to change
      Atomics.wait(this.i32, PIPE_WRITE_POS, wp);
    }
  }

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

    return toRead;
  }

  /** Decrement reader refcount. Last close signals broken pipe to writers. */
  closeReader(): void {
    const remaining = Atomics.sub(this.i32, PIPE_READER_CLOSED, 1) - 1;
    if (remaining <= 0) {
      Atomics.store(this.i32, PIPE_READER_CLOSED, 0);
      Atomics.notify(this.i32, PIPE_READ_POS);  // wake writers (broken pipe)
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
