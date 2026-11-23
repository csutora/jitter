import { describe, it, expect } from 'vitest';
import { PipeRing } from '../src/pipe-ring';
import { PIPE_HEADER_SIZE, PAGE_SIZE, pipeRingSize, PIPE_DEF_PAGES, PIPE_MIN_PAGES } from '../src/kernel-ipc';

const arena = () => new SharedArrayBuffer(1 << 20);
const enc = (s: string) => new TextEncoder().encode(s);

describe('ring size is a property of the ring', () => {
  it('publishes its capacity on reset so a later adopter agrees where it ends', () => {
    const sab = arena();
    const size = pipeRingSize(PIPE_MIN_PAGES);
    new PipeRing(sab, 0, size).reset(1, 1);

    const adopted = new PipeRing(sab, 0); // no size argument: read it from the header
    expect(adopted.capacity).toBe(PAGE_SIZE);
  });

  it('refuses to adopt a ring that was never reset instead of guessing', () => {
    // the dangerous case: a stale offset. a guessed size would index into whatever the arena put
    // next to it, so this has to fail loudly rather than read a neighbour's bytes.
    expect(() => new PipeRing(arena(), 0)).toThrow(RangeError);
  });

  it('keeps two differently-sized rings out of each other', () => {
    const sab = arena();
    const smallSize = pipeRingSize(PIPE_MIN_PAGES);
    const small = new PipeRing(sab, 0, smallSize);
    small.reset(1, 1);
    const big = new PipeRing(sab, smallSize, pipeRingSize(2));
    big.reset(1, 1);

    // fill the small ring completely; it must not spill into the big one that follows it
    const written = small.write(new Uint8Array(PAGE_SIZE * 2).fill(0xab));
    expect(written).toBe(PAGE_SIZE);
    expect(new PipeRing(sab, smallSize).available).toBe(0);
  });
});

describe('a downgraded (single page) ring still behaves like a pipe', () => {
  const mk = (pages: number) => {
    const sab = arena();
    const r = new PipeRing(sab, 0, pipeRingSize(pages));
    r.reset(1, 1);
    return r;
  };

  it('round-trips data', () => {
    const r = mk(PIPE_MIN_PAGES);
    expect(r.write(enc('hello'))).toBe(5);
    const dst = new Uint8Array(16);
    expect(r.read(dst)).toBe(5);
    expect(new TextDecoder().decode(dst.subarray(0, 5))).toBe('hello');
  });

  it('reports a short write at ITS capacity, not the default one', () => {
    const r = mk(PIPE_MIN_PAGES);
    expect(r.write(new Uint8Array(PAGE_SIZE + 100))).toBe(PAGE_SIZE);
    expect(r.write(enc('x'))).toBe(0); // full
  });

  it('holds a full PIPE_BUF write atomically, which is the posix guarantee', () => {
    // POSIX promises writes of PIPE_BUF (4096) or less are atomic. one page is exactly that, so
    // even the smallest pipe we hand out can still keep the promise.
    const r = mk(PIPE_MIN_PAGES);
    expect(r.write(new Uint8Array(4096).fill(7))).toBe(4096);
  });

  it('wraps around correctly when the ring is smaller than the traffic', () => {
    const r = mk(PIPE_MIN_PAGES);
    const dst = new Uint8Array(1000);
    let sent = 0;
    for (let i = 0; i < 20; i++) {
      const chunk = new Uint8Array(1000).fill(i);
      expect(r.write(chunk)).toBe(1000);
      sent += 1000;
      expect(r.read(dst)).toBe(1000);
      expect(dst[0]).toBe(i);
      expect(dst[999]).toBe(i);
    }
    expect(sent).toBeGreaterThan(PAGE_SIZE * 4); // definitely wrapped
  });
});

describe('a default ring still keeps the 64 KiB promise', () => {
  it('buffers exactly the capacity F_GETPIPE_SZ would report', () => {
    const sab = arena();
    const r = new PipeRing(sab, 0, pipeRingSize(PIPE_DEF_PAGES));
    r.reset(1, 1);
    expect(r.capacity).toBe(65536);
    expect(r.write(new Uint8Array(65536))).toBe(65536);
    expect(r.write(enc('x'))).toBe(0);
  });

  it('leaves room for the header inside its allocation', () => {
    expect(pipeRingSize(PIPE_DEF_PAGES)).toBe(PIPE_HEADER_SIZE + 65536);
  });
});

describe('peek and discard', () => {
  const fresh = () => {
    const sab = arena();
    const size = pipeRingSize(PIPE_MIN_PAGES);
    const ring = new PipeRing(sab, 0, size);
    ring.reset(1, 1);
    return ring;
  };

  it('peek copies without consuming, discard consumes', () => {
    const ring = fresh();
    ring.write(enc('hello world'));
    const a = new Uint8Array(5);
    expect(ring.peek(a)).toBe(5);
    expect(new TextDecoder().decode(a)).toBe('hello');
    // still all there: peek moved nothing
    expect(ring.bytesAvailable).toBe(11);
    const b = new Uint8Array(5);
    expect(ring.peek(b)).toBe(5);
    expect(new TextDecoder().decode(b)).toBe('hello');
    expect(ring.discard(5)).toBe(5);
    expect(ring.bytesAvailable).toBe(6);
    const c = new Uint8Array(16);
    expect(ring.peek(c)).toBe(6);
    expect(new TextDecoder().decode(c.subarray(0, 6))).toBe(' world');
  });

  it('peek wraps around the end of the ring', () => {
    const ring = fresh();
    const cap = ring.capacity;
    ring.write(new Uint8Array(cap - 4).fill(1));
    expect(ring.discard(cap - 4)).toBe(cap - 4);
    const payload = new Uint8Array(8);
    for (let i = 0; i < 8; i++) payload[i] = i + 10;
    ring.write(payload);
    const out = new Uint8Array(8);
    expect(ring.peek(out)).toBe(8);
    expect([...out]).toEqual([...payload]);
    expect(ring.bytesAvailable).toBe(8);
  });

  it('discard never runs past what is in the ring', () => {
    const ring = fresh();
    ring.write(new Uint8Array(3).fill(7));
    expect(ring.discard(99)).toBe(3);
    expect(ring.bytesAvailable).toBe(0);
    expect(ring.discard(1)).toBe(0);
  });
});
