import { describe, it, expect } from 'vitest';
import { loadElf, setupStack } from '../src/elf.js';
import { GUEST_MEM_BASE } from '../src/types.js';

// Build a minimal ELF64 aarch64 binary with one PT_LOAD segment
function buildMinimalElf(opts: {
  entry: bigint;
  vaddr: bigint;
  code: Uint8Array;
}): Uint8Array {
  const { entry, vaddr, code } = opts;
  const ehdrSize = 64;
  const phdrSize = 56;
  const phoff = ehdrSize;
  const codeOff = ehdrSize + phdrSize;
  const totalSize = codeOff + code.length;

  const buf = new ArrayBuffer(totalSize);
  const dv = new DataView(buf);
  const bytes = new Uint8Array(buf);

  // ELF header
  bytes[0] = 0x7F; bytes[1] = 0x45; bytes[2] = 0x4C; bytes[3] = 0x46; // magic
  bytes[4] = 2;    // ELFCLASS64
  bytes[5] = 1;    // ELFDATA2LSB
  bytes[6] = 1;    // EV_CURRENT
  dv.setUint16(16, 2, true);  // e_type = ET_EXEC
  dv.setUint16(18, 0xB7, true); // e_machine = EM_AARCH64
  dv.setUint32(20, 1, true);  // e_version
  dv.setBigUint64(24, entry, true); // e_entry
  dv.setBigUint64(32, BigInt(phoff), true); // e_phoff
  dv.setBigUint64(40, 0n, true); // e_shoff
  dv.setUint32(48, 0, true);  // e_flags
  dv.setUint16(52, 64, true); // e_ehsize
  dv.setUint16(54, 56, true); // e_phentsize
  dv.setUint16(56, 1, true);  // e_phnum

  // Program header (PT_LOAD)
  const ph = phoff;
  dv.setUint32(ph, 1, true);     // p_type = PT_LOAD
  dv.setUint32(ph + 4, 5, true); // p_flags = PF_R|PF_X
  dv.setBigUint64(ph + 8, BigInt(codeOff), true);  // p_offset
  dv.setBigUint64(ph + 16, vaddr, true);   // p_vaddr
  dv.setBigUint64(ph + 24, vaddr, true);   // p_paddr
  dv.setBigUint64(ph + 32, BigInt(code.length), true); // p_filesz
  dv.setBigUint64(ph + 40, BigInt(code.length), true); // p_memsz
  dv.setBigUint64(ph + 48, 0x1000n, true); // p_align

  // Code section
  bytes.set(code, codeOff);

  return bytes;
}

describe('elf', () => {
  it('loads a minimal ELF and returns correct info', () => {
    const code = new Uint8Array([0x01, 0x02, 0x03, 0x04]);
    const elf = buildMinimalElf({
      entry: 0x400000n,
      vaddr: 0x400000n,
      code,
    });

    const memory = new WebAssembly.Memory({ initial: 256 });
    const info = loadElf(elf, memory);

    expect(info.entry).toBe(0x400000n);
    expect(info.phentSize).toBe(56);
    expect(info.phnum).toBe(1);
    expect(info.brkStart).toBeGreaterThan(0x400000n);

    // Verify code was loaded at guest address + GUEST_MEM_BASE
    const view = new Uint8Array(memory.buffer);
    expect(view[0x400000 + GUEST_MEM_BASE]).toBe(0x01);
    expect(view[0x400001 + GUEST_MEM_BASE]).toBe(0x02);
    expect(view[0x400002 + GUEST_MEM_BASE]).toBe(0x03);
    expect(view[0x400003 + GUEST_MEM_BASE]).toBe(0x04);
  });

  it('rejects non-ELF files', () => {
    const memory = new WebAssembly.Memory({ initial: 16 });
    const badFile = new Uint8Array([0x00, 0x00, 0x00, 0x00]);
    expect(() => loadElf(badFile, memory)).toThrow('Not an ELF file');
  });

  it('sets up stack with correct layout', () => {
    const memory = new WebAssembly.Memory({ initial: 256 });
    const elfInfo = {
      entry: 0x400000n,
      phdr: 0x400040n,
      phentSize: 56,
      phnum: 1,
      brkStart: 0x401000n,
    };

    const sp = setupStack(memory, elfInfo, ['sh'], ['HOME=/root']);
    const dv = new DataView(memory.buffer);

    // SP should be 16-byte aligned
    expect(Number(sp) % 16).toBe(0);
    expect(sp).toBeLessThan(0x800000n);
    expect(sp).toBeGreaterThan(0x700000n);

    // First 8 bytes at SP: argc = 1
    const argc = dv.getBigUint64(Number(sp) + GUEST_MEM_BASE, true);
    expect(argc).toBe(1n);

    // argv[0] pointer (offset 8 from SP)
    const argv0Ptr = Number(dv.getBigUint64(Number(sp) + 8 + GUEST_MEM_BASE, true));
    expect(argv0Ptr).toBeGreaterThan(0);

    // Read the string at argv[0]
    const bytes = new Uint8Array(memory.buffer);
    let str = '';
    for (let i = 0; bytes[argv0Ptr + GUEST_MEM_BASE + i] !== 0; i++) {
      str += String.fromCharCode(bytes[argv0Ptr + GUEST_MEM_BASE + i]);
    }
    expect(str).toBe('sh');

    // argv[1] should be NULL
    const argv1 = dv.getBigUint64(Number(sp) + 16 + GUEST_MEM_BASE, true);
    expect(argv1).toBe(0n);

    // envp[0] pointer (after NULL)
    const envp0Ptr = Number(dv.getBigUint64(Number(sp) + 24 + GUEST_MEM_BASE, true));
    let envStr = '';
    for (let i = 0; bytes[envp0Ptr + GUEST_MEM_BASE + i] !== 0; i++) {
      envStr += String.fromCharCode(bytes[envp0Ptr + GUEST_MEM_BASE + i]);
    }
    expect(envStr).toBe('HOME=/root');
  });
});
