import { describe, it } from 'vitest';
import { decode } from '../src/decode.js';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const BASE = 0x400000;
const elfPath = resolve(import.meta.dirname!, '..', 'assets', 'busybox-aarch64');
const elf = readFileSync(elfPath);

function dumpRange(label: string, startPC: number, endPC: number) {
  console.log(`\n=== ${label} ===`);
  for (let pc = startPC; pc < endPC; pc += 4) {
    const fileOff = pc - BASE;
    const insn =
      elf[fileOff] |
      (elf[fileOff + 1] << 8) |
      (elf[fileOff + 2] << 16) |
      ((elf[fileOff + 3] << 24) >>> 0);
    // Force unsigned 32-bit
    const uinsn = insn >>> 0;
    const decoded = decode(uinsn);
    console.log(
      `0x${pc.toString(16)}: 0x${uinsn.toString(16).padStart(8, '0')} \u2192 ${JSON.stringify(decoded, (_k, v) => typeof v === 'bigint' ? '0x' + v.toString(16) : v)}`,
    );
  }
}

describe('decode-dump: busybox heap corruption investigation', () => {
  it('Range 1: 0x4ce5ac-0x4ce5d0 (string function called from setvar)', () => {
    dumpRange('string function called from setvar', 0x4ce5ac, 0x4ce5d0);
  });

  it('Range 2: 0x4ce6dc-0x4ce750 (strcpy setup + loop)', () => {
    dumpRange('strcpy setup + loop', 0x4ce6dc, 0x4ce750);
  });

  it('Range 3: 0x4cea44-0x4cea60 (function called before strcpy)', () => {
    dumpRange('function called before strcpy', 0x4cea44, 0x4cea60);
  });

  it('Range 4: 0x44ac20-0x44ac60 (setvar calling string/strcpy functions)', () => {
    dumpRange('setvar calling string/strcpy functions', 0x44ac20, 0x44ac60);
  });

  it('Range 5: 0x404250-0x4042c0 (malloc implementation)', () => {
    dumpRange('malloc implementation', 0x404250, 0x4042c0);
  });

  it('Range 6: 0x4ceb60-0x4cebb8 (strlen-like function called from setvar)', () => {
    dumpRange('strlen-like function called from setvar', 0x4ceb60, 0x4cebb8);
  });
});
