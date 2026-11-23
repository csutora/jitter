// compare the syscall numbers src/syscall.ts dispatches on against the real aarch64 table.
//
// the numbers are the one part of a syscall implementation that cannot be got right by testing the
// happy path: a wrong number silently answers a DIFFERENT call, and the only symptom is a distant
// program misbehaving. chroot was implemented as fchdir this way, and sethostname was labelled
// chroot and returned success.
//
//   npx tsx tools/audit-syscall-numbers.ts <path-to-aarch64-syscalls.txt>
//
// the table file is "<number> <name>" per line; generate it with
//   zig cc -target aarch64-linux-musl -E -dM - <<< '#include <sys/syscall.h>'
import { readFileSync } from 'fs';

const tablePath = process.argv[2];
if (!tablePath) { console.error('usage: audit-syscall-numbers.ts <table>'); process.exit(2); }

const official = new Map<number, string>();
for (const line of readFileSync(tablePath, 'utf8').split('\n')) {
  const m = line.match(/^(\d+)\s+(\S+)/);
  if (m) official.set(Number(m[1]), m[2]);
}

const src = readFileSync('src/syscall.ts', 'utf8').split('\n');
// a case label, plus whatever the line (or the one after it) calls the syscall in a comment
const seen = new Map<number, { line: number; label: string }[]>();
for (let i = 0; i < src.length; i++) {
  const m = src[i].match(/^\s*case (\d+):(.*)$/);
  if (!m) continue;
  const nr = Number(m[1]);
  let label = m[2];
  // a run of `case N: case M: {` shares one comment; look ahead a line for it
  if (!/\/\//.test(label) && i + 1 < src.length) label += ' ' + src[i + 1];
  const cm = label.match(/\/\/\s*([A-Za-z_][A-Za-z0-9_]*)/);
  if (!seen.has(nr)) seen.set(nr, []);
  seen.get(nr)!.push({ line: i + 1, label: cm ? cm[1] : '?' });
}

let bad = 0;
for (const [nr, uses] of [...seen].sort((a, b) => a[0] - b[0])) {
  const real = official.get(nr);
  if (real === undefined) {
    console.log(`MISSING  ${nr} is not an aarch64 syscall at all (we call it "${uses[0].label}") at syscall.ts:${uses[0].line}`);
    bad++;
    continue;
  }
  for (const u of uses) {
    if (u.label === '?') continue;
    // the comment names the syscall; allow the common aliases the table spells differently
    const alias: Record<string, string> = {
      epoll_pwait: 'epoll_pwait', newfstatat: 'fstatat', fstatat: 'newfstatat',
      getdents: 'getdents64', sigaction: 'rt_sigaction', sigprocmask: 'rt_sigprocmask',
      sigreturn: 'rt_sigreturn', sigsuspend: 'rt_sigsuspend', sigtimedwait: 'rt_sigtimedwait',
      wait4: 'wait4', clone: 'clone', mmap: 'mmap', eventfd: 'eventfd2', epoll_create: 'epoll_create1',
      stat: 'newfstatat', lseek: 'lseek', pread: 'pread64', pwrite: 'pwrite64',
    };
    if (u.label !== real && alias[u.label] !== real && !real.startsWith(u.label) && !u.label.startsWith(real)) {
      console.log(`MISLABEL ${nr} is ${real}, we call it "${u.label}" at syscall.ts:${u.line}`);
      bad++;
    }
  }
}
console.log(`\n${seen.size} numbers dispatched, ${bad} suspicious`);
