// trace what busybox `timeout` does, in the fork-capable node dispatch path, to find why
// it returns 127. logs every syscall (nr/args/result) + fork/execve/exit transitions.
import { readFileSync } from 'fs';
import { join } from 'path';
import { Emulator, IOManager } from '../src/dispatch.js';
import { loadElf, setupStack } from '../src/elf.js';
import { SyscallHandler } from '../src/syscall.js';
import { InMemoryVFS } from '../src/vfs.js';
import { TPIDR_OFFSET } from '../src/types.js';
import { initZigTranslator } from '../src/zig-translator.js';

initZigTranslator(new Uint8Array(readFileSync('engine/zig-out/bin/translator.wasm')));
const elfBytes = new Uint8Array(readFileSync(join('assets', 'busybox-aarch64')));
const cmd = process.argv[2] ?? 'timeout 5 echo TIMEOUT_OK\n';

const emu = new Emulator(1024);
emu.elfBytes = elfBytes;
const vfs = new InMemoryVFS();
vfs.loadBinary(elfBytes);
// seed /bin applet files so PATH lookup finds them (dispatch execve always loads busybox
// bytes regardless of path; argv[0]'s basename selects the applet).
for (const a of ['timeout', 'sleep', 'echo', 'sh', 'cat', 'ls', 'true']) {
  try { (vfs as any).writeFileBytes('/bin/' + a, elfBytes, 0o755); } catch (e) { console.error('seed fail', a, (e as any).message); }
}
const elfInfo = loadElf(elfBytes, emu.memory);
emu.cpu.setSP(setupStack(emu.memory, elfInfo, ['sh'], ['HOME=/root', 'TERM=linux', 'PATH=/bin', 'PWD=/']));
new DataView(emu.memory.buffer).setBigUint64(TPIDR_OFFSET, 0x900000n, true);

const io = new IOManager();
io.pushStdin(new TextEncoder().encode(cmd));
let stdout = '';
const syscalls = new SyscallHandler(elfInfo.brkStart, {
  writeStdout: (d) => { stdout += new TextDecoder().decode(d); },
  writeStderr: (d) => { stdout += new TextDecoder().decode(d); },
  readStdin: (n) => io.readStdin(n),
}, vfs);
emu.syscallHandlerInstance = syscalls;

const NAME: Record<number, string> = {
  56: 'openat', 57: 'close', 63: 'read', 64: 'write', 93: 'exit', 94: 'exit_group',
  220: 'clone', 221: 'execve', 260: 'wait4', 129: 'kill', 79: 'fstatat', 80: 'fstat',
  101: 'nanosleep', 103: 'setitimer', 134: 'rt_sigaction', 135: 'rt_sigprocmask',
  172: 'getpid', 173: 'getppid', 178: 'gettid', 233: 'madvise', 96: 'set_tid_address',
};
const dec = new TextDecoder();
function gstr(addr: number): string {
  if (!addr) return '';
  const b = new Uint8Array(emu.memory.buffer); const base = addr + 0; // GUEST_MEM_BASE folded in reader below
  return '';
}

let pc = elfInfo.entry;
const HALT = 0xDEAD0000n, SVC = 0xDEAD0001n;
let blocks = 0, depth = 0, traceOn = false, sc = 0;
while (pc !== HALT) {
  blocks++;
  pc = (emu as any).getRegion(pc).run();
  const view = new DataView(emu.memory.buffer);
  if (pc === SVC) {
    const svcPC = view.getBigUint64(256, true);
    const nr = Number(emu.cpu.getReg(8));
    const a0 = emu.cpu.getReg(0), a1 = emu.cpu.getReg(1), a2 = emu.cpu.getReg(2);
    // start tracing once we exec timeout (the first execve after boot)
    const before = `[d${depth}] ${NAME[nr] ?? 'sys' + nr}(0x${a0.toString(16)}, 0x${a1.toString(16)}, 0x${a2.toString(16)})`;
    const result: any = syscalls.handle(emu.cpu, emu.memory);
    sc++;
    const KEY = new Set([220, 221, 260, 93, 94, 103, 129, 130, 131, 101, 134]);
    if (nr === 221) { // execve
      const path = readStr(emu, Number(a0));
      console.error(`[d${depth}] EXECVE "${path}" argv1="${readStr(emu, Number(new DataView(emu.memory.buffer).getBigUint64(Number(a1) + 0x10000, true)))}" -> ${result.status}`);
    } else if (KEY.has(nr)) {
      console.error(`${before} -> ${result.status ?? ''} x0=0x${emu.cpu.getReg(0).toString(16)}`);
    }
    if (result.status === 'exit') {
      const restored = emu.processManager.exit(emu.memory, emu.cpu, result.code);
      if (traceOn) console.error(`[d${depth}] EXIT code=${result.code} restored=${!!restored}`);
      if (restored) {
        syscalls.lastWaitStatus = result.code; syscalls.lastWaitPid = restored.childPid;
        syscalls.restoreState(restored.syscallState); pc = restored.resumePC; depth--; continue;
      }
      break;
    }
    if (result.status === 'blocked_read') { console.error('BLOCKED_READ'); break; }
    if (result.status === 'fork') {
      emu.processManager.fork(emu.memory, emu.cpu, syscalls.getState(), result.resumePC);
      depth++; if (traceOn) console.error(`[d${depth}] FORK`); pc = result.resumePC; continue;
    }
    if (result.status === 'execve') {
      const { entry, brkStart } = emu.processManager.execve(emu.memory, emu.cpu, elfBytes, result.argv, result.envp);
      syscalls.resetForExec(brkStart); pc = entry; continue;
    }
    pc = svcPC + 4n; continue;
  }
  if (blocks >= 3_000_000) { console.error('BUDGET'); break; }
}
console.error('--- STDOUT ---'); console.error(JSON.stringify(stdout));

function readStr(emu: Emulator, addr: number): string {
  if (!addr) return '';
  const b = new Uint8Array(emu.memory.buffer);
  const GUEST = 0x10000;
  let s = ''; let p = addr + GUEST;
  for (let i = 0; i < 256 && b[p]; i++, p++) s += String.fromCharCode(b[p]);
  return s;
}
