// run a static aarch64 binary with fork support (dispatch ProcessManager) + syscall logging,
// to see where an LTP test hangs. copy-fork semantics match the device worker path.
import { readFileSync } from 'fs';
import { Emulator, IOManager } from '../src/dispatch.js';
import { loadElf, setupStack } from '../src/elf.js';
import { SyscallHandler } from '../src/syscall.js';
import { InMemoryVFS } from '../src/vfs.js';
import { TPIDR_OFFSET } from '../src/types.js';
import { initZigTranslator } from '../src/zig-translator.js';
initZigTranslator(new Uint8Array(readFileSync('engine/zig-out/bin/translator.wasm')));
const path = process.argv[2];
const budget = Number(process.argv[3] || 400);
const elfBytes = new Uint8Array(readFileSync(path));
const emu = new Emulator(1024); emu.elfBytes = elfBytes;
const vfs = new InMemoryVFS();
const elfInfo = loadElf(elfBytes, emu.memory);
emu.cpu.setSP(setupStack(emu.memory, elfInfo, [path], ['PATH=/bin','PWD=/','TMPDIR=/tmp']));
new DataView(emu.memory.buffer).setBigUint64(TPIDR_OFFSET, 0x900000n, true);
let out = '';
const syscalls = new SyscallHandler(elfInfo.brkStart, {
  writeStdout: (d)=>{out+=new TextDecoder().decode(d);}, writeStderr:(d)=>{out+=new TextDecoder().decode(d);}, readStdin:()=>new Uint8Array(),
}, vfs);
emu.syscallHandlerInstance = syscalls;
const N: Record<number,string> = {56:'openat',57:'close',63:'read',64:'write',93:'exit',94:'exit_grp',220:'clone',221:'execve',260:'wait4',129:'kill',79:'fstatat',80:'fstat',101:'nanosleep',115:'clk_nanosleep',103:'setitimer',134:'sigaction',135:'sigprocmask',96:'set_tid_addr',98:'futex',222:'mmap',226:'mprotect',214:'brk',233:'madvise',48:'faccessat',439:'faccessat2',278:'getrandom',146:'setuid',147:'setresuid',148:'setresgid',149:'setgid',151:'setfsuid',153:'setgroups',48:'faccessat',439:'faccessat2'};
let pc = elfInfo.entry; const HALT=0xDEAD0000n, SVC=0xDEAD0001n; let blocks=0, sc=0, depth=0;
while (pc !== HALT) {
  blocks++;
  pc = (emu as any).getRegion(pc).run();
  const view = new DataView(emu.memory.buffer);
  if (pc === SVC) {
    const svcPC = view.getBigUint64(256, true);
    const nr = Number(emu.cpu.getReg(8));
    const a0=emu.cpu.getReg(0);
    const r: any = syscalls.handle(emu.cpu, emu.memory);
    sc++;
    const tag = `[d${depth}#${sc}] ${N[nr]??'sys'+nr}`;
    if (sc > budget) { console.error(`BUDGET. last=${tag}`); break; }
    if ([146,147,149,153,48,439,220,260].includes(nr)) console.error(`${tag}(a0=0x${a0.toString(16)}) euid=${(syscalls as any).euid} -> ${r.status??('0x'+emu.cpu.getReg(0).toString(16))}`);
    if (r.status==='exit'){ const rst=emu.processManager.exit(emu.memory,emu.cpu,r.code); console.error(`${tag} EXIT ${r.code} reaped=${!!rst}`); if(rst){syscalls.lastWaitStatus=r.code;syscalls.lastWaitPid=rst.childPid;syscalls.restoreState(rst.syscallState);pc=rst.resumePC;depth--;continue;} break; }
    if (r.status==='blocked_read'){console.error(`${tag} BLOCKED_READ`);break;}
    if (r.status==='fork'){ emu.processManager.fork(emu.memory,emu.cpu,syscalls.getState(),r.resumePC); depth++; console.error(`${tag} -> FORK child (depth ${depth})`); pc=r.resumePC; continue; }
    if (r.status==='execve'){ const {entry,brkStart}=emu.processManager.execve(emu.memory,emu.cpu,elfBytes,r.argv,r.envp); syscalls.resetForExec(brkStart); pc=entry; continue; }
    pc = svcPC + 4n; continue;
  }
  if (blocks > 8_000_000){console.error('BLOCK BUDGET (spin?)');break;}
}
console.error('--- stdout tail ---'); console.error(out.slice(-1000));
