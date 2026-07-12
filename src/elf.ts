import { GUEST_MEM_BASE } from './types.js';

// ── ELF64 constants ─────────────────────────────────────────────────────────

const ELF_MAGIC = [0x7F, 0x45, 0x4C, 0x46]; // \x7FELF
const ELFCLASS64 = 2;
const ELFDATA2LSB = 1;       // Little-endian
const EM_AARCH64 = 0xB7;
const PT_LOAD = 1;
const PT_DYNAMIC = 2;
const PT_INTERP = 3;
const PT_PHDR = 6;

// Dynamic section tags
const DT_NULL   = 0;
const DT_RELA   = 7;
const DT_RELASZ = 8;
const DT_RELAENT = 9;
const DT_RELR   = 36; // 0x24
const DT_RELRSZ = 35; // 0x23
const DT_RELRENT = 37; // 0x25

// ELF type
const ET_DYN = 3; // PIE / shared object

// ── ELF info returned after loading ─────────────────────────────────────────

export interface ElfInfo {
  entry: bigint;        // entry point guest address (load-base adjusted)
  phdr: bigint;         // program headers address in guest memory
  phentSize: number;    // size of one program header entry
  phnum: number;        // number of program headers
  brkStart: bigint;     // end of last PT_LOAD segment (heap starts here)
  loadBase: bigint;     // base address for PIE, 0 for non-PIE
  interpPath?: string;  // PT_INTERP path (dynamic executables); undefined if static
  interpBase?: bigint;  // load base of the interpreter, once loaded (for AT_BASE)
  interpEntry?: bigint; // interpreter entry point — execution starts here when dynamic
}

// base address for the dynamic linker (ld.so). sits above the program (0x400000) and
// stack (0x800000) but below the shadow reservation at 32MB.
const INTERP_LOAD_BASE = 0x1000000n; // 16MB

// ── Load an ELF64 binary into WASM linear memory ───────────────────────────

const PIE_LOAD_BASE = 0x400000n; // conventional load address for PIE

export function loadElf(
  elfBytes: Uint8Array,
  memory: WebAssembly.Memory,
): ElfInfo {
  const dv = new DataView(elfBytes.buffer, elfBytes.byteOffset, elfBytes.byteLength);

  // Validate ELF magic
  for (let i = 0; i < 4; i++) {
    if (elfBytes[i] !== ELF_MAGIC[i]) {
      throw new Error('Not an ELF file');
    }
  }

  // Validate ELF64, little-endian, aarch64
  if (elfBytes[4] !== ELFCLASS64) throw new Error('Not a 64-bit ELF');
  if (elfBytes[5] !== ELFDATA2LSB) throw new Error('Not little-endian');
  const eMachine = dv.getUint16(18, true);
  if (eMachine !== EM_AARCH64) throw new Error(`Not aarch64 (e_machine=0x${eMachine.toString(16)})`);

  const eType = dv.getUint16(16, true);
  const isPie = eType === ET_DYN;

  // Read ELF header fields
  const rawEntry = dv.getBigUint64(24, true);  // e_entry
  const phoff = Number(dv.getBigUint64(32, true));
  const phentSize = dv.getUint16(54, true);
  const phnum = dv.getUint16(56, true);

  // For PIE, find the lowest vaddr to determine load base
  let lowestVaddr = 0xFFFFFFFFFFFFFFFFn;
  for (let i = 0; i < phnum; i++) {
    const phBase = phoff + i * phentSize;
    const pType = dv.getUint32(phBase, true);
    if (pType === PT_LOAD) {
      const pVaddr = dv.getBigUint64(phBase + 16, true);
      if (pVaddr < lowestVaddr) lowestVaddr = pVaddr;
    }
  }

  const loadBase = isPie ? PIE_LOAD_BASE - lowestVaddr : 0n;
  const entry = rawEntry + loadBase;

  let brkStart = 0n;
  let phdrAddr = 0n;
  let dynOffset = 0;
  let dynSize = 0;
  let interpPath: string | undefined;

  // Process program headers — load segments
  for (let i = 0; i < phnum; i++) {
    const phBase = phoff + i * phentSize;
    const pType = dv.getUint32(phBase, true);
    const pOffset = Number(dv.getBigUint64(phBase + 8, true));
    const pVaddr = dv.getBigUint64(phBase + 16, true) + loadBase;
    const pFilesz = Number(dv.getBigUint64(phBase + 32, true));
    const pMemsz = Number(dv.getBigUint64(phBase + 40, true));

    if (pType === PT_PHDR) {
      phdrAddr = pVaddr;
    }

    if (pType === PT_DYNAMIC) {
      dynOffset = pOffset;
      dynSize = pFilesz;
    }

    if (pType === PT_INTERP) {
      let end = pOffset;
      while (end < pOffset + pFilesz && elfBytes[end] !== 0) end++;
      interpPath = new TextDecoder().decode(elfBytes.subarray(pOffset, end));
    }

    if (pType !== PT_LOAD) continue;

    const guestAddr = Number(pVaddr);
    const wasmAddr = guestAddr + GUEST_MEM_BASE;

    // Ensure memory is large enough
    const needed = wasmAddr + pMemsz;
    if (needed > memory.buffer.byteLength) {
      const pagesNeeded = Math.ceil(needed / 65536);
      const currentPages = memory.buffer.byteLength / 65536;
      if (pagesNeeded > currentPages) {
        memory.grow(pagesNeeded - currentPages);
      }
    }

    // Re-get view after potential grow
    const freshBytes = new Uint8Array(memory.buffer);

    // Copy file contents
    if (pFilesz > 0) {
      freshBytes.set(elfBytes.subarray(pOffset, pOffset + pFilesz), wasmAddr);
    }

    // Zero-fill BSS (memsz - filesz)
    if (pMemsz > pFilesz) {
      freshBytes.fill(0, wasmAddr + pFilesz, wasmAddr + pMemsz);
    }

    // Track end of loaded segments for brk
    const segEnd = pVaddr + BigInt(pMemsz);
    if (segEnd > brkStart) {
      brkStart = segEnd;
    }
  }

  // If no PT_PHDR, compute phdr address from first PT_LOAD + phoff
  if (phdrAddr === 0n) {
    phdrAddr = loadBase + BigInt(phoff);
  }

  // NOTE: Do NOT apply RELR/RELA relocations here for PIE binaries.
  // Static PIE binaries (e.g., musl-based busybox) apply their own relocations
  // during startup via rcrt1/ldso. Applying them here would cause double relocation.

  // Align brkStart to page boundary
  brkStart = (brkStart + 0xFFFn) & ~0xFFFn;

  return {
    entry,
    phdr: phdrAddr,
    phentSize,
    phnum,
    brkStart,
    loadBase,
    interpPath,
  };
}

// ── Load the dynamic linker (ld.so) alongside a dynamic executable ──────────
// ld.so is itself an ET_DYN; load its PT_LOAD segments at `base` and return its
// entry point. execution for a dynamic program starts here, not at the program entry.
export function loadInterp(
  interpBytes: Uint8Array,
  memory: WebAssembly.Memory,
  base: bigint = INTERP_LOAD_BASE,
): { entry: bigint; base: bigint } {
  const dv = new DataView(interpBytes.buffer, interpBytes.byteOffset, interpBytes.byteLength);
  for (let i = 0; i < 4; i++) if (interpBytes[i] !== ELF_MAGIC[i]) throw new Error('interp not an ELF');

  const rawEntry = dv.getBigUint64(24, true);
  const phoff = Number(dv.getBigUint64(32, true));
  const phentSize = dv.getUint16(54, true);
  const phnum = dv.getUint16(56, true);

  let lowestVaddr = 0xFFFFFFFFFFFFFFFFn;
  for (let i = 0; i < phnum; i++) {
    const phBase = phoff + i * phentSize;
    if (dv.getUint32(phBase, true) === PT_LOAD) {
      const v = dv.getBigUint64(phBase + 16, true);
      if (v < lowestVaddr) lowestVaddr = v;
    }
  }
  const loadBase = base - lowestVaddr;

  for (let i = 0; i < phnum; i++) {
    const phBase = phoff + i * phentSize;
    if (dv.getUint32(phBase, true) !== PT_LOAD) continue;
    const pOffset = Number(dv.getBigUint64(phBase + 8, true));
    const pVaddr = dv.getBigUint64(phBase + 16, true) + loadBase;
    const pFilesz = Number(dv.getBigUint64(phBase + 32, true));
    const pMemsz = Number(dv.getBigUint64(phBase + 40, true));
    const wasmAddr = Number(pVaddr) + GUEST_MEM_BASE;

    const needed = wasmAddr + pMemsz;
    if (needed > memory.buffer.byteLength) {
      const pagesNeeded = Math.ceil(needed / 65536);
      const currentPages = memory.buffer.byteLength / 65536;
      if (pagesNeeded > currentPages) memory.grow(pagesNeeded - currentPages);
    }
    const freshBytes = new Uint8Array(memory.buffer);
    if (pFilesz > 0) freshBytes.set(interpBytes.subarray(pOffset, pOffset + pFilesz), wasmAddr);
    if (pMemsz > pFilesz) freshBytes.fill(0, wasmAddr + pFilesz, wasmAddr + pMemsz);
  }

  return { entry: rawEntry + loadBase, base: loadBase };
}

// ── Apply RELR relocations ─────────────────────────────────────────────────

function applyRelocations(
  elfBytes: Uint8Array,
  memory: WebAssembly.Memory,
  loadBase: bigint,
  dynOffset: number,
  dynSize: number,
): void {
  const elfDv = new DataView(elfBytes.buffer, elfBytes.byteOffset, elfBytes.byteLength);
  const memDv = new DataView(memory.buffer);

  // Parse DYNAMIC section to find RELR entries
  let relrAddr = 0n;
  let relrSize = 0;
  let relaAddr = 0n;
  let relaSize = 0;
  let relaEnt = 0;

  for (let pos = dynOffset; pos < dynOffset + dynSize; pos += 16) {
    const tag = Number(elfDv.getBigInt64(pos, true));
    const val = elfDv.getBigUint64(pos + 8, true);
    if (tag === DT_NULL) break;

    switch (tag) {
      case DT_RELR: relrAddr = val; break;
      case DT_RELRSZ: relrSize = Number(val); break;
      case DT_RELA: relaAddr = val; break;
      case DT_RELASZ: relaSize = Number(val); break;
      case DT_RELAENT: relaEnt = Number(val); break;
    }
  }

  // Apply RELR (compact relative relocations)
  if (relrSize > 0) {
    // RELR entries are in the loaded memory (file offset = relrAddr for PIE where first LOAD starts at 0)
    // But since they're now loaded at loadBase, we read from WASM memory
    const base = Number(relrAddr + loadBase) + GUEST_MEM_BASE;
    const nEntries = relrSize / 8;
    let where = 0n;

    for (let i = 0; i < nEntries; i++) {
      const entry = memDv.getBigUint64(base + i * 8, true);

      if ((entry & 1n) === 0n) {
        // Pointer entry: apply relocation at this address
        where = entry + loadBase;
        const wasmAddr = Number(where) + GUEST_MEM_BASE;
        const oldVal = memDv.getBigUint64(wasmAddr, true);
        memDv.setBigUint64(wasmAddr, oldVal + loadBase, true);
        where += 8n;
      } else {
        // Bitmap entry: each set bit is a relocation at where + bit * 8
        let bitmap = entry >> 1n;
        let pos = 0;
        while (bitmap !== 0n) {
          if (bitmap & 1n) {
            const addr = where + BigInt(pos * 8);
            const wasmAddr = Number(addr) + GUEST_MEM_BASE;
            const oldVal = memDv.getBigUint64(wasmAddr, true);
            memDv.setBigUint64(wasmAddr, oldVal + loadBase, true);
          }
          bitmap >>= 1n;
          pos++;
        }
        where += BigInt((1 + 63) * 8); // advance by 63 slots (bitmap covers slots 0..62)
      }
    }
  }

  // Apply RELA relocations if any
  if (relaSize > 0 && relaEnt > 0) {
    const base = Number(relaAddr + loadBase) + GUEST_MEM_BASE;
    const nEntries = relaSize / relaEnt;

    for (let i = 0; i < nEntries; i++) {
      const offset = memDv.getBigUint64(base + i * relaEnt, true) + loadBase;
      const info = memDv.getBigUint64(base + i * relaEnt + 8, true);
      const addend = memDv.getBigInt64(base + i * relaEnt + 16, true);
      const rType = Number(info & 0xFFFFFFFFn);

      // R_AARCH64_RELATIVE = 1027
      if (rType === 1027) {
        const wasmAddr = Number(offset) + GUEST_MEM_BASE;
        memDv.setBigUint64(wasmAddr, BigInt.asUintN(64, loadBase + addend), true);
      }
    }
  }
}

// ── High-level program loader (main ELF + optional dynamic interpreter) ──────
// loads the main ELF, and if it's dynamic (has PT_INTERP), loads the interpreter
// (ld.so) via the `readInterp` callback and arranges for execution to start at the
// interpreter's entry with a full auxv. `readInterp(path)` returns the interpreter's
// bytes (or throws). returns the entry PC to jump to and the initial SP.
export function loadProgram(
  memory: WebAssembly.Memory,
  elfBytes: Uint8Array,
  argv: string[],
  envp: string[],
  readInterp: (path: string) => Uint8Array,
): { entry: bigint; sp: bigint; brkStart: bigint; interp: boolean } {
  const elfInfo = loadElf(elfBytes, memory);
  let startPC = elfInfo.entry;
  if (elfInfo.interpPath) {
    const interpBytes = readInterp(elfInfo.interpPath);
    const { entry: iEntry, base: iBase } = loadInterp(interpBytes, memory);
    elfInfo.interpBase = iBase;
    elfInfo.interpEntry = iEntry;
    startPC = iEntry;
  }
  const sp = setupStack(memory, elfInfo, argv, envp);
  return { entry: startPC, sp, brkStart: elfInfo.brkStart, interp: !!elfInfo.interpPath };
}

// ── Stack setup (Linux ABI) ─────────────────────────────────────────────────

const STACK_TOP = 0x800000; // 8MB

// Auxiliary vector types
const AT_NULL = 0;
const AT_PHDR = 3;
const AT_PHENT = 4;
const AT_PHNUM = 5;
const AT_PAGESZ = 6;
const AT_BASE = 7;       // interpreter (ld.so) load base
const AT_FLAGS = 8;
const AT_ENTRY = 9;
const AT_UID = 11;
const AT_EUID = 12;
const AT_GID = 13;
const AT_EGID = 14;
const AT_PLATFORM = 15;
const AT_HWCAP = 16;
const AT_CLKTCK = 17;
const AT_SECURE = 23;
const AT_RANDOM = 25;
const AT_EXECFN = 31;

// hwcap bits we can honestly back: FP + AdvSIMD (both fully translated). deliberately
// NOT advertising SVE/SME/etc that we can't run, so glibc ifunc resolvers stay on code
// paths we support.
const HWCAP_FP = 1 << 0;
const HWCAP_ASIMD = 1 << 1;

export function setupStack(
  memory: WebAssembly.Memory,
  elfInfo: ElfInfo,
  argv: string[],
  envp: string[],
): bigint {
  const mem = new DataView(memory.buffer);
  const bytes = new Uint8Array(memory.buffer);

  let sp = STACK_TOP;

  // ── Write string data at top of stack area ──
  const stringAddrs: number[] = [];

  for (const arg of argv) {
    const encoded = new TextEncoder().encode(arg + '\0');
    sp -= encoded.length;
    bytes.set(encoded, sp + GUEST_MEM_BASE);
    stringAddrs.push(sp);
  }

  const envAddrs: number[] = [];
  for (const env of envp) {
    const encoded = new TextEncoder().encode(env + '\0');
    sp -= encoded.length;
    bytes.set(encoded, sp + GUEST_MEM_BASE);
    envAddrs.push(sp);
  }

  // Write 16 random bytes for AT_RANDOM
  sp -= 16;
  const randomAddr = sp;
  for (let i = 0; i < 16; i++) {
    bytes[sp + GUEST_MEM_BASE + i] = (Math.random() * 256) | 0;
  }

  // AT_PLATFORM string
  const platform = new TextEncoder().encode('aarch64\0');
  sp -= platform.length;
  const platformAddr = sp;
  bytes.set(platform, sp + GUEST_MEM_BASE);

  // AT_EXECFN points at the program path (argv[0])
  const execfnAddr = stringAddrs.length > 0 ? stringAddrs[0] : 0;

  // ── Align SP to 16 bytes ──
  sp = sp & ~0xF;

  // auxv entries
  const auxv: [number, bigint][] = [
    [AT_PHDR, elfInfo.phdr],
    [AT_PHENT, BigInt(elfInfo.phentSize)],
    [AT_PHNUM, BigInt(elfInfo.phnum)],
    [AT_PAGESZ, 4096n],
    [AT_ENTRY, elfInfo.entry],
    [AT_FLAGS, 0n],
    [AT_HWCAP, BigInt(HWCAP_FP | HWCAP_ASIMD)],
    [AT_CLKTCK, 100n],
    [AT_SECURE, 0n],
    [AT_UID, 0n],
    [AT_EUID, 0n],
    [AT_GID, 0n],
    [AT_EGID, 0n],
    [AT_PLATFORM, BigInt(platformAddr)],
    [AT_EXECFN, BigInt(execfnAddr)],
    [AT_RANDOM, BigInt(randomAddr)],
  ];
  // AT_BASE only for dynamic executables (interpreter present)
  if (elfInfo.interpBase !== undefined) auxv.push([AT_BASE, elfInfo.interpBase]);
  auxv.push([AT_NULL, 0n]);

  const stackEntries =
    8 + // argc
    (argv.length + 1) * 8 + // argv + NULL
    (envp.length + 1) * 8 + // envp + NULL
    auxv.length * 16; // auxv pairs

  sp -= stackEntries;
  sp = sp & ~0xF; // re-align

  let pos = sp;

  // argc
  mem.setBigUint64(pos + GUEST_MEM_BASE, BigInt(argv.length), true);
  pos += 8;

  // argv pointers
  for (let i = 0; i < argv.length; i++) {
    mem.setBigUint64(pos + GUEST_MEM_BASE, BigInt(stringAddrs[i]), true);
    pos += 8;
  }
  mem.setBigUint64(pos + GUEST_MEM_BASE, 0n, true); // NULL
  pos += 8;

  // envp pointers
  for (let i = 0; i < envp.length; i++) {
    mem.setBigUint64(pos + GUEST_MEM_BASE, BigInt(envAddrs[i]), true);
    pos += 8;
  }
  mem.setBigUint64(pos + GUEST_MEM_BASE, 0n, true); // NULL
  pos += 8;

  // auxv
  for (const [type, val] of auxv) {
    mem.setBigUint64(pos + GUEST_MEM_BASE, BigInt(type), true);
    pos += 8;
    mem.setBigUint64(pos + GUEST_MEM_BASE, val, true);
    pos += 8;
  }

  return BigInt(sp);
}
