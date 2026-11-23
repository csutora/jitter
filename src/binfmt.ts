// what makes a file runnable.
//
// execve does not just jump into a file: it asks each binary format handler whether it recognises
// the image. we support two, the same two that matter in practice: ELF, and the `#!` script, where
// the kernel silently re-points the exec at an interpreter and hands it the script as an argument.
//
// this lives on its own because there are two callers that must agree: a process calling execve,
// and the kernel spawning a process from outside (the boot shell, a detached run). linux draws no
// distinction between those -- a script is a program either way -- and when only one path knew
// about `#!`, /bin/ltp was executable from a shell but "not an executable" when spawned directly.

/** BINPRM_BUF_SIZE: the kernel only ever reads this much of a file to identify it. */
export const BINPRM_BUF_SIZE = 256;

/** how many times exec will follow one interpreter to another before giving up with ELOOP. */
export const MAX_INTERP_DEPTH = 4;

export function isElf(bytes: Uint8Array): boolean {
  return bytes.length >= 4 && bytes[0] === 0x7f && bytes[1] === 0x45 && bytes[2] === 0x4c && bytes[3] === 0x46;
}

export interface Shebang {
  interp: string;
  /** everything after the interpreter, as ONE argument. the kernel does not split it on spaces,
   *  which is why `#!/usr/bin/env -S foo bar` needs env's own -S to re-split it. */
  arg?: string;
}

export function parseShebang(bytes: Uint8Array): Shebang | null {
  if (bytes.length < 2 || bytes[0] !== 0x23 || bytes[1] !== 0x21) return null; // '#!'
  const limit = Math.min(bytes.length, BINPRM_BUF_SIZE);
  let end = 2;
  while (end < limit && bytes[end] !== 0x0a) end++;
  const line = new TextDecoder().decode(bytes.subarray(2, end)).trim();
  if (!line) return null;
  const sp = line.search(/\s/);
  if (sp < 0) return { interp: line };
  const arg = line.slice(sp).trim();
  return arg ? { interp: line.slice(0, sp), arg } : { interp: line.slice(0, sp) };
}

export interface ExecTarget {
  path: string;
  argv: string[];
  bytes: Uint8Array;
}

export class ExecError extends Error {
  constructor(message: string, readonly errno: number) { super(message); }
}

/**
 * Follow `#!` lines until the target is a real binary. `read` returns the file's bytes or null if
 * it cannot be read; a missing INTERPRETER is ENOENT reported against the exec as a whole, which is
 * the kernel's behaviour and the reason "no such file or directory" can name a file that plainly
 * exists.
 */
export function resolveExec(path: string, argv: string[], read: (p: string) => Uint8Array | null): ExecTarget {
  let bytes = read(path);
  if (!bytes) throw new ExecError(`exec: cannot read ${path}`, 2); // ENOENT

  for (let depth = 0; ; depth++) {
    const sb = parseShebang(bytes);
    if (!sb) break;
    if (depth >= MAX_INTERP_DEPTH) throw new ExecError(`exec: interpreter loop at ${path}`, 40); // ELOOP
    const next: string[] = [sb.interp];
    if (sb.arg) next.push(sb.arg);
    next.push(path, ...argv.slice(1));
    path = sb.interp;
    argv = next;
    bytes = read(path);
    if (!bytes) throw new ExecError(`exec: interpreter ${path} not found`, 2); // ENOENT
  }

  if (!isElf(bytes)) throw new ExecError(`exec: not an ELF file: ${path}`, 8); // ENOEXEC
  return { path, argv, bytes };
}
