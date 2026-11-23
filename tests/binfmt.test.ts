import { describe, it, expect } from 'vitest';
import { parseShebang, isElf, resolveExec, ExecError, BINPRM_BUF_SIZE } from '../src/binfmt';

const enc = (s: string) => new TextEncoder().encode(s);
const elf = (tag = '') => {
  const b = new Uint8Array(16 + tag.length);
  b.set([0x7f, 0x45, 0x4c, 0x46]);
  b.set(enc(tag), 16);
  return b;
};

describe('parseShebang', () => {
  it('reads a bare interpreter', () => {
    expect(parseShebang(enc('#!/bin/sh\necho hi'))).toEqual({ interp: '/bin/sh' });
  });

  it('keeps the argument as ONE string, unsplit', () => {
    // the kernel does not tokenise -- this is why `#!/usr/bin/env -S a b` needs env's own -S
    expect(parseShebang(enc('#!/usr/bin/env -S python -u\n'))).toEqual({
      interp: '/usr/bin/env', arg: '-S python -u',
    });
  });

  it('tolerates spaces around the interpreter', () => {
    expect(parseShebang(enc('#!  /bin/sh  \n'))).toEqual({ interp: '/bin/sh' });
  });

  it('handles a file with no trailing newline', () => {
    expect(parseShebang(enc('#!/bin/sh'))).toEqual({ interp: '/bin/sh' });
  });

  it('rejects anything that is not #!', () => {
    expect(parseShebang(enc('#/bin/sh\n'))).toBeNull();
    expect(parseShebang(elf())).toBeNull();
    expect(parseShebang(new Uint8Array([0x23]))).toBeNull();
    expect(parseShebang(enc('#!\n'))).toBeNull();
  });

  it('stops at the kernel buffer size, not at the newline', () => {
    const long = '#!/bin/' + 'x'.repeat(BINPRM_BUF_SIZE * 2) + '\n';
    const sb = parseShebang(enc(long))!;
    expect(sb.interp.length).toBeLessThanOrEqual(BINPRM_BUF_SIZE);
  });
});

describe('isElf', () => {
  it('matches the magic and nothing else', () => {
    expect(isElf(elf())).toBe(true);
    expect(isElf(enc('#!/bin/sh'))).toBe(false);
    expect(isElf(new Uint8Array([0x7f, 0x45, 0x4c]))).toBe(false);
  });
});

describe('resolveExec', () => {
  const fs = (files: Record<string, Uint8Array>) => (p: string) => files[p] ?? null;

  it('passes an ELF straight through', () => {
    const r = resolveExec('/bin/ls', ['ls', '-l'], fs({ '/bin/ls': elf() }));
    expect(r.path).toBe('/bin/ls');
    expect(r.argv).toEqual(['ls', '-l']);
  });

  it('re-points a script at its interpreter and inserts the script path', () => {
    const r = resolveExec('/bin/ltp', ['/bin/ltp', 'p', 'q'], fs({
      '/bin/ltp': enc('#!/bin/sh\n...'), '/bin/sh': elf(),
    }));
    expect(r.path).toBe('/bin/sh');
    expect(r.argv).toEqual(['/bin/sh', '/bin/ltp', 'p', 'q']);
  });

  it('places the optional argument before the script path', () => {
    const r = resolveExec('/s', ['/s', 'x'], fs({
      '/s': enc('#!/usr/bin/awk -f\n'), '/usr/bin/awk': elf(),
    }));
    expect(r.argv).toEqual(['/usr/bin/awk', '-f', '/s', 'x']);
  });

  it('follows a script whose interpreter is itself a script', () => {
    const r = resolveExec('/a', ['/a'], fs({
      '/a': enc('#!/b\n'), '/b': enc('#!/bin/sh\n'), '/bin/sh': elf(),
    }));
    expect(r.path).toBe('/bin/sh');
    expect(r.argv).toEqual(['/bin/sh', '/b', '/a']);
  });

  it('gives up on an interpreter loop rather than recursing forever', () => {
    const files = { '/a': enc('#!/b\n'), '/b': enc('#!/a\n') };
    expect(() => resolveExec('/a', ['/a'], fs(files))).toThrow(ExecError);
    try { resolveExec('/a', ['/a'], fs(files)); } catch (e) { expect((e as ExecError).errno).toBe(40); } // ELOOP
  });

  it('reports a MISSING INTERPRETER as ENOENT against the exec', () => {
    // the classic confusing error: "no such file or directory" naming a script that plainly exists
    try {
      resolveExec('/s', ['/s'], fs({ '/s': enc('#!/bin/nope\n') }));
      throw new Error('should have thrown');
    } catch (e) {
      expect((e as ExecError).errno).toBe(2);
    }
  });

  it('reports a data file as ENOEXEC', () => {
    try {
      resolveExec('/d', ['/d'], fs({ '/d': enc('just text\n') }));
      throw new Error('should have thrown');
    } catch (e) {
      expect((e as ExecError).errno).toBe(8);
    }
  });

  it('reports an unreadable target as ENOENT', () => {
    try {
      resolveExec('/gone', ['/gone'], fs({}));
      throw new Error('should have thrown');
    } catch (e) {
      expect((e as ExecError).errno).toBe(2);
    }
  });
});
