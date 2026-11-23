import { describe, it, expect } from 'vitest';
import { LockTable, LeaseTable, EOF_END, LEASE_RD, LEASE_WR, LEASE_UN, type LockRequest, type LockScope } from '../src/locks';

const F = '/tmp/f';

function req(
  opts: Partial<LockRequest> & { kind: LockRequest['kind']; pid: number },
): LockRequest {
  const scope: LockScope = opts.owner?.scope ?? 'posix';
  return {
    space: opts.space ?? 'record',
    file: opts.file ?? F,
    kind: opts.kind,
    start: opts.start ?? 0,
    end: opts.end ?? EOF_END,
    owner: opts.owner ?? { scope, id: opts.pid, pid: opts.pid },
  };
}

describe('record locks', () => {
  it('grants a write lock and blocks a conflicting one from another process', () => {
    const t = new LockTable();
    expect(t.trySet(req({ kind: 'write', pid: 1 }))).toBe(0);
    expect(t.trySet(req({ kind: 'write', pid: 2 }))).toBe(-11); // EAGAIN
    expect(t.trySet(req({ kind: 'read', pid: 2 }))).toBe(-11);
  });

  it('lets read locks share', () => {
    const t = new LockTable();
    expect(t.trySet(req({ kind: 'read', pid: 1 }))).toBe(0);
    expect(t.trySet(req({ kind: 'read', pid: 2 }))).toBe(0);
    expect(t.trySet(req({ kind: 'write', pid: 3 }))).toBe(-11);
  });

  it('does not let disjoint ranges conflict', () => {
    const t = new LockTable();
    expect(t.trySet(req({ kind: 'write', pid: 1, start: 0, end: 10 }))).toBe(0);
    expect(t.trySet(req({ kind: 'write', pid: 2, start: 10, end: 20 }))).toBe(0);
    expect(t.trySet(req({ kind: 'write', pid: 3, start: 9, end: 11 }))).toBe(-11);
  });

  it('replaces rather than stacks a same-owner lock, and splits on partial unlock', () => {
    const t = new LockTable();
    t.trySet(req({ kind: 'write', pid: 1, start: 0, end: 100 }));
    t.trySet(req({ kind: 'unlock', pid: 1, start: 40, end: 60 }));
    const held = t.snapshot('record', F).sort((a, b) => a.start - b.start);
    expect(held.map((h) => [h.start, h.end])).toEqual([[0, 40], [60, 100]]);
    // the hole is now free for someone else
    expect(t.trySet(req({ kind: 'write', pid: 2, start: 40, end: 60 }))).toBe(0);
  });

  it('downgrades a write lock to a read lock in place', () => {
    const t = new LockTable();
    t.trySet(req({ kind: 'write', pid: 1, start: 0, end: 100 }));
    t.trySet(req({ kind: 'read', pid: 1, start: 0, end: 100 }));
    expect(t.snapshot('record', F)).toEqual([
      { start: 0, end: 100, kind: 'read', pid: 1, scope: 'posix', id: 1 },
    ]);
    expect(t.trySet(req({ kind: 'read', pid: 2, start: 0, end: 100 }))).toBe(0);
  });

  it('coalesces adjacent same-owner ranges', () => {
    const t = new LockTable();
    t.trySet(req({ kind: 'write', pid: 1, start: 0, end: 10 }));
    t.trySet(req({ kind: 'write', pid: 1, start: 10, end: 20 }));
    expect(t.snapshot('record', F)).toHaveLength(1);
  });

  it('reports the blocking lock via test(), and nothing when free', () => {
    const t = new LockTable();
    t.trySet(req({ kind: 'write', pid: 7, start: 5, end: 15 }));
    expect(t.test(req({ kind: 'write', pid: 8, start: 0, end: 100 }))).toMatchObject({
      start: 5, end: 15, kind: 'write', pid: 7,
    });
    expect(t.test(req({ kind: 'write', pid: 8, start: 50, end: 60 }))).toBeNull();
  });
});

describe('ownership scopes', () => {
  it('treats a process lock and an ofd lock in the same process as different owners', () => {
    const t = new LockTable();
    t.trySet(req({ kind: 'write', pid: 1, owner: { scope: 'posix', id: 1, pid: 1 } }));
    expect(t.trySet(req({ kind: 'write', pid: 1, owner: { scope: 'ofd', id: 99, pid: 1 } }))).toBe(-11);
  });

  it('keeps two ofd locks from one process independent', () => {
    const t = new LockTable();
    t.trySet(req({ kind: 'write', pid: 1, owner: { scope: 'ofd', id: 10, pid: 1 } }));
    expect(t.trySet(req({ kind: 'write', pid: 1, owner: { scope: 'ofd', id: 11, pid: 1 } }))).toBe(-11);
    expect(t.trySet(req({ kind: 'write', pid: 1, owner: { scope: 'ofd', id: 10, pid: 1 } }))).toBe(0);
  });

  it('keeps record locks and flock locks from seeing each other', () => {
    const t = new LockTable();
    t.trySet(req({ kind: 'write', pid: 1 }));
    expect(t.trySet(req({ kind: 'write', pid: 2, space: 'flock' }))).toBe(0);
  });
});

describe('blocking waits', () => {
  it('grants a waiter as soon as the holder unlocks', () => {
    const t = new LockTable();
    t.trySet(req({ kind: 'write', pid: 1 }));
    const seen: number[] = [];
    t.setBlocking(req({ kind: 'write', pid: 2 }), (e) => seen.push(e));
    expect(seen).toEqual([]);
    expect(t.waitingCount()).toBe(1);
    t.apply(req({ kind: 'unlock', pid: 1 }));
    expect(seen).toEqual([0]);
    expect(t.snapshot('record', F)[0].pid).toBe(2);
  });

  it('takes an uncontended blocking lock immediately', () => {
    const t = new LockTable();
    const seen: number[] = [];
    t.setBlocking(req({ kind: 'write', pid: 1 }), (e) => seen.push(e));
    expect(seen).toEqual([0]);
  });

  it('wakes several waiters when a write lock becomes shareable', () => {
    const t = new LockTable();
    t.trySet(req({ kind: 'write', pid: 1 }));
    const seen: number[] = [];
    t.setBlocking(req({ kind: 'read', pid: 2 }), (e) => seen.push(e));
    t.setBlocking(req({ kind: 'read', pid: 3 }), (e) => seen.push(e));
    t.apply(req({ kind: 'unlock', pid: 1 }));
    expect(seen).toEqual([0, 0]);
  });

  it('refuses a wait that would close a cycle with EDEADLK', () => {
    const t = new LockTable();
    t.trySet(req({ kind: 'write', pid: 1, file: '/a' }));
    t.trySet(req({ kind: 'write', pid: 2, file: '/b' }));
    const a: number[] = [];
    t.setBlocking(req({ kind: 'write', pid: 1, file: '/b' }), (e) => a.push(e));
    expect(a).toEqual([]); // 1 now waits on 2
    const b: number[] = [];
    t.setBlocking(req({ kind: 'write', pid: 2, file: '/a' }), (e) => b.push(e));
    expect(b).toEqual([-35]); // EDEADLK
  });

  it('cuts a wait short with EINTR when the process is signalled', () => {
    const t = new LockTable();
    t.trySet(req({ kind: 'write', pid: 1 }));
    const seen: number[] = [];
    t.setBlocking(req({ kind: 'write', pid: 2 }), (e) => seen.push(e));
    t.interrupt(2);
    expect(seen).toEqual([-4]);
    expect(t.waitingCount()).toBe(0);
  });
});

describe('release', () => {
  it('drops a process posix locks when any descriptor for the file closes', () => {
    const t = new LockTable();
    t.trySet(req({ kind: 'write', pid: 1, start: 0, end: 10 }));
    t.trySet(req({ kind: 'write', pid: 1, start: 50, end: 60 }));
    t.releaseProcessFile(1, F);
    expect(t.snapshot('record', F)).toEqual([]);
  });

  it('leaves other files alone when one is closed', () => {
    const t = new LockTable();
    t.trySet(req({ kind: 'write', pid: 1, file: '/a' }));
    t.trySet(req({ kind: 'write', pid: 1, file: '/b' }));
    t.releaseProcessFile(1, '/a');
    expect(t.snapshot('record', '/a')).toEqual([]);
    expect(t.snapshot('record', '/b')).toHaveLength(1);
  });

  it('drops ofd locks only when the description goes', () => {
    const t = new LockTable();
    t.trySet(req({ kind: 'write', pid: 1, owner: { scope: 'ofd', id: 42, pid: 1 } }));
    t.releaseProcessFile(1, F); // posix-only release must not touch it
    expect(t.snapshot('record', F)).toHaveLength(1);
    t.releaseDescription(42);
    expect(t.snapshot('record', F)).toEqual([]);
  });

  it('releases everything a dead process held and unblocks its waiters', () => {
    const t = new LockTable();
    t.trySet(req({ kind: 'write', pid: 1 }));
    const seen: number[] = [];
    t.setBlocking(req({ kind: 'write', pid: 2 }), (e) => seen.push(e));
    t.releaseProcess(1);
    expect(seen).toEqual([0]);
    expect(t.snapshot('record', F)[0].pid).toBe(2);
  });
});

describe('leases', () => {
  const sigs: Array<[number, number]> = [];
  const mk = () => { sigs.length = 0; return new LeaseTable((pid, sig) => { sigs.push([pid, sig]); }); };
  const LF = '/tmp/leased';

  it('grants a read lease only when nobody has the file open for writing', () => {
    const t = mk();
    t.noteOpen(1, 3, LF, false);
    expect(t.set(1, 3, LF, LEASE_RD, 29, true, true)).toBe(0);
    expect(t.get(1, 3)).toBe(LEASE_RD);

    const u = mk();
    u.noteOpen(1, 3, LF, false);
    u.noteOpen(2, 9, LF, true);
    expect(u.set(1, 3, LF, LEASE_RD, 29, true, true)).toBe(-11); // EAGAIN
  });

  it('grants a write lease only to the sole opener', () => {
    const t = mk();
    t.noteOpen(1, 3, LF, false);
    t.noteOpen(2, 9, LF, false);
    expect(t.set(1, 3, LF, LEASE_WR, 29, true, true)).toBe(-11);
    t.noteClose(2, 9);
    expect(t.set(1, 3, LF, LEASE_WR, 29, true, true)).toBe(0);
    expect(t.get(1, 3)).toBe(LEASE_WR);
  });

  it('refuses a lease on a file the caller does not own, or on a non-regular file', () => {
    const t = mk();
    t.noteOpen(1, 3, LF, false);
    expect(t.set(1, 3, LF, LEASE_RD, 29, false, true)).toBe(-13); // EACCES
    expect(t.set(1, 3, LF, LEASE_RD, 29, true, false)).toBe(-22); // EINVAL
    expect(t.set(1, 3, LF, 7, 29, true, true)).toBe(-22);
  });

  it('signals the holder and holds the opener back until the lease is released', async () => {
    const t = mk();
    t.noteOpen(1, 3, LF, false);
    t.set(1, 3, LF, LEASE_WR, 12, true, true);

    let opened = false;
    const p = t.breakOn(LF, false, false).then((r) => { opened = true; return r; });
    await Promise.resolve();
    expect(sigs).toEqual([[1, 12]]);
    expect(opened).toBe(false);

    expect(t.set(1, 3, LF, LEASE_UN, 12, true, true)).toBe(0);
    expect(await p).toBe(0);
    expect(t.get(1, 3)).toBe(LEASE_UN);
  });

  it('lets a reader through on a downgrade, but keeps a writer waiting', async () => {
    const t = mk();
    t.noteOpen(1, 3, LF, false);
    t.set(1, 3, LF, LEASE_WR, 29, true, true);

    let reader = false, writer = false;
    const pr = t.breakOn(LF, false, false).then(() => { reader = true; });
    const pw = t.breakOn(LF, true, false).then(() => { writer = true; });

    t.set(1, 3, LF, LEASE_RD, 29, true, true);
    await pr;
    expect(reader).toBe(true);
    expect(writer).toBe(false);

    t.set(1, 3, LF, LEASE_UN, 29, true, true);
    await pw;
    expect(writer).toBe(true);
  });

  it('answers a non-blocking opener immediately, but still starts the break', async () => {
    const t = mk();
    t.noteOpen(1, 3, LF, false);
    t.set(1, 3, LF, LEASE_WR, 29, true, true);
    expect(await t.breakOn(LF, true, true)).toBe(-11); // EWOULDBLOCK
    expect(sigs).toEqual([[1, 29]]);
  });

  it('drops a lease when the holder closes the descriptor or dies', () => {
    const t = mk();
    t.noteOpen(1, 3, LF, false);
    t.set(1, 3, LF, LEASE_WR, 29, true, true);
    t.noteClose(1, 3);
    expect(t.get(1, 3)).toBe(LEASE_UN);
    expect(t.openCount(LF)).toBe(0);

    t.noteOpen(2, 4, LF, false);
    t.set(2, 4, LF, LEASE_WR, 29, true, true);
    t.dropProcess(2);
    expect(t.get(2, 4)).toBe(LEASE_UN);
    expect(t.openCount(LF)).toBe(0);
  });

  it('counts a dup as another opener', () => {
    const t = mk();
    t.noteOpen(1, 3, LF, false);
    t.noteDupFrom(1, 3, 7);
    expect(t.openCount(LF)).toBe(2);
    expect(t.set(1, 3, LF, LEASE_WR, 29, true, true)).toBe(-11);
  });
});
