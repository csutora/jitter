import { describe, it, expect } from 'vitest';
import { reconcile, ArenaAllocator } from '../src/shared-map';

function region(n = 16): { mine: Uint8Array; arena: Uint8Array; shadow: Uint8Array } {
  return { mine: new Uint8Array(n), arena: new Uint8Array(n), shadow: new Uint8Array(n) };
}

describe('reconcile', () => {
  it('publishes our writes to the arena', () => {
    const r = region();
    r.mine[0] = 7;
    reconcile(r.mine, r.arena, r.shadow);
    expect(r.arena[0]).toBe(7);
    expect(r.shadow[0]).toBe(7);
  });

  it('observes the other side writes', () => {
    const r = region();
    r.arena[4] = 9;
    reconcile(r.mine, r.arena, r.shadow);
    expect(r.mine[4]).toBe(9);
  });

  it('leaves untouched words alone', () => {
    const r = region();
    r.mine.fill(3); r.arena.fill(3); r.shadow.fill(3);
    reconcile(r.mine, r.arena, r.shadow);
    expect([...r.mine]).toEqual(new Array(16).fill(3));
    expect([...r.arena]).toEqual(new Array(16).fill(3));
  });

  it('lets a write win over a stale read of the same word', () => {
    const r = region();
    // both sides agreed on 5, then we wrote 6; the arena must take our value, not overwrite it
    r.mine.fill(5); r.arena.fill(5); r.shadow.fill(5);
    r.mine[0] = 6;
    reconcile(r.mine, r.arena, r.shadow);
    expect(r.arena[0]).toBe(6);
    expect(r.mine[0]).toBe(6);
  });

  it('merges disjoint writes from both sides', () => {
    const r = region();
    r.mine[0] = 1;   // our word 0
    r.arena[8] = 2;  // their word 2
    reconcile(r.mine, r.arena, r.shadow);
    expect(r.arena[0]).toBe(1);
    expect(r.mine[8]).toBe(2);
  });

  it('round-trips a two-process exchange', () => {
    // one arena, two independent views, as a fork would give
    const arena = new Uint8Array(16);
    const a = { mine: new Uint8Array(16), shadow: new Uint8Array(16) };
    const b = { mine: new Uint8Array(16), shadow: new Uint8Array(16) };

    a.mine[0] = 42;                        // parent writes
    reconcile(a.mine, arena, a.shadow);    // ...and hits a syscall
    reconcile(b.mine, arena, b.shadow);    // child hits a syscall
    expect(b.mine[0]).toBe(42);            // child observes it

    b.mine[4] = 99;                        // child replies
    reconcile(b.mine, arena, b.shadow);
    reconcile(a.mine, arena, a.shadow);
    expect(a.mine[4]).toBe(99);
    expect(a.mine[0]).toBe(42);            // and the earlier value survived
  });

  it('handles a length that is not a multiple of four', () => {
    const r = region(7);
    r.mine[6] = 5;
    r.arena[5] = 4;
    reconcile(r.mine, r.arena, r.shadow);
    expect(r.arena[6]).toBe(5);
    expect(r.mine[5]).toBe(4);
  });
});

describe('ArenaAllocator', () => {
  it('hands out page-aligned blocks and refuses when full', () => {
    const a = new ArenaAllocator(8192);
    expect(a.alloc(1)).toBe(0);
    expect(a.alloc(4096)).toBe(4096);
    expect(a.alloc(1)).toBe(-1);
  });

  it('reuses a released block', () => {
    const a = new ArenaAllocator(8192);
    const first = a.alloc(4096);
    a.alloc(4096);
    a.release(first, 4096);
    expect(a.alloc(4096)).toBe(first);
  });

  it('coalesces neighbours so fragmentation does not accumulate', () => {
    const a = new ArenaAllocator(4096 * 4);
    const p = [a.alloc(4096), a.alloc(4096), a.alloc(4096)];
    a.release(p[0], 4096);
    a.release(p[1], 4096);
    expect(a.alloc(8192)).toBe(p[0]); // only possible if the two were joined
  });

  it('refuses a double release rather than handing the same block to two callers', () => {
    const a = new ArenaAllocator(4096 * 4);
    const x = a.alloc(4096);
    const y = a.alloc(4096);
    a.release(x, 4096);
    a.release(x, 4096); // the bug: one block, freed twice
    expect(a.doubleReleases).toBe(1);
    const p = a.alloc(4096);
    const q = a.alloc(4096);
    expect(p).not.toBe(q);
    expect(y).not.toBe(p);
    expect(y).not.toBe(q);
  });

  it('gives the bump pointer back when the tail is released', () => {
    const a = new ArenaAllocator(4096 * 2);
    const x = a.alloc(4096);
    a.alloc(4096);
    expect(a.alloc(4096)).toBe(-1);
    a.release(4096, 4096); // release the tail block
    expect(a.used()).toBe(4096);
    expect(a.alloc(4096)).toBe(4096);
    expect(x).toBe(0);
  });
});

describe('reconcile across fork', () => {
  // a forked child's shadow is what it INHERITED, not what the arena holds when its worker starts.
  // seeding from the arena makes every byte the parent wrote in between look like a write by the
  // child of the older value, and the child publishes the stale byte back over it.
  it('does not publish bytes the child never wrote', () => {
    const LEN = 16;
    const parent = new Uint8Array(LEN);
    const arena = new Uint8Array(LEN);
    const parentShadow = new Uint8Array(LEN);

    // the parent writes the first half and publishes it; that is the state the child forks from.
    parent[0] = 11;
    reconcile(parent, arena, parentShadow);
    const inherited = new Uint8Array(parent);

    // after the fork, the parent writes one more byte and publishes it.
    parent[1] = 22;
    reconcile(parent, arena, parentShadow);
    expect(arena[1]).toBe(22);

    // the child starts, with a shadow seeded from what it inherited.
    const child = new Uint8Array(inherited);
    const childShadow = new Uint8Array(inherited);
    reconcile(child, arena, childShadow);

    expect(arena[1]).toBe(22);  // the parent's write survives
    expect(child[1]).toBe(22);  // and the child sees it
  });
});
