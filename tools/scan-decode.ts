// list every instruction in a set of binaries that the translator cannot decode.
//
// llvm-objdump does the linear disassembly (it knows where code ends and literal pools begin, so
// data does not masquerade as an undecodable instruction), and each word it calls an instruction
// is handed to the zig decoder. anything the decoder answers "unknown" to is a real gap, reported
// with its mnemonic and how often it appears.
//
// gaps are aggregated across every input, because that is the useful question: an emulator is
// complete when NOTHING in the corpus is undecodable, and one binary at a time cannot tell you
// which gap is worth closing first.
//
// usage: npx tsx tools/scan-decode.ts <elf> [<elf> ...]
//        npx tsx tools/scan-decode.ts --list <file-with-one-path-per-line>

import { readFileSync } from 'fs';
import { join } from 'path';
import { execFileSync } from 'child_process';
import { initZigTranslator, zigDecodeKind } from '../src/zig-translator.js';

initZigTranslator(new Uint8Array(readFileSync(
  join(import.meta.dirname, '..', 'engine', 'zig-out', 'bin', 'translator.wasm'))));

// objdump lines look like: "  400b24: 4e379633     \tmla\tv19.16b, v17.16b, v23.16b"
const LINE = /^\s*[0-9a-f]+:\s+([0-9a-f]{8})\s+(.+)$/;

const gaps = new Map<string, { count: number; word: number; first: string; files: Set<string> }>();
let total = 0, unknown = 0;
let currentFile = '';

// one objdump per batch rather than per file: the process launch dominates for small binaries,
// and the corpus is over a thousand of them.
const BATCH = 24;

function objdump(paths: string[]): string {
  return execFileSync('llvm-objdump', ['-d', ...paths], { encoding: 'utf8', maxBuffer: 1 << 30 });
}

function scanBatch(paths: string[]): void {
  let out: string;
  try { out = objdump(paths); } catch { return; }
  for (const line of out.split('\n')) {
    const fm = /^(\S+):\s*file format/.exec(line);
    if (fm) { currentFile = fm[1]; continue; }
    const m = LINE.exec(line);
    if (!m) continue;
    const word = parseInt(m[1], 16) >>> 0;
    const text = m[2].replace(/\s+/g, ' ').trim();
    if (text.startsWith('<') || text === '' || text.startsWith('.')) continue;
    total++;
    if (zigDecodeKind(word) !== 0) continue;
    unknown++;
    // group by mnemonic + operand shape, so every `mla vX.16b, ...` counts as one gap
    const key = text.split(' ')[0] + ' ' + (text.split(' ')[1] ?? '').replace(/v\d+/g, 'v').replace(/[wx]\d+/g, 'r');
    const e = gaps.get(key);
    if (e) { e.count++; e.files.add(currentFile); }
    else gaps.set(key, { count: 1, word, first: text, files: new Set([currentFile]) });
  }
}

const args = process.argv.slice(2);
let paths: string[];
if (args[0] === '--list') paths = readFileSync(args[1], 'utf8').split('\n').filter(Boolean);
else paths = args;
if (!paths.length) { console.error('usage: scan-decode <elf> [<elf> ...] | --list <file>'); process.exit(2); }

for (let i = 0; i < paths.length; i += BATCH) {
  scanBatch(paths.slice(i, i + BATCH));
  if (i % (BATCH * 10) === 0) process.stderr.write(`\r${i}/${paths.length} binaries`);
}
process.stderr.write(`\r${paths.length}/${paths.length} binaries\n`);

console.log(`\n${paths.length} binaries, ${total} instructions, ${unknown} undecodable (${(100 * unknown / Math.max(1, total)).toFixed(4)}%)`);
const rows = [...gaps.entries()].sort((a, b) => b[1].count - a[1].count);
for (const [, e] of rows) {
  console.log(`  ${String(e.count).padStart(7)}  in ${String(e.files.size).padStart(4)} files  0x${e.word.toString(16).padStart(8, '0')}  ${e.first}`);
}
if (!rows.length) console.log('  (none)');
