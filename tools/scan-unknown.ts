// find real instruction decode gaps: read objdump -d output (which knows instruction boundaries and
// skips data/literal pools), and for each disassembled instruction run its encoding through the zig
// engine's decode-kind probe. report the UNDECODED ones grouped by mnemonic — exactly what to add.
// usage: objdump -d a.so b.so > /tmp/dis.txt && tsx tools/scan-unknown.ts /tmp/dis.txt
import { readFileSync } from 'fs';
import { initZigTranslator, zigDecodeKind } from '../src/zig-translator.js';

await initZigTranslator(new Uint8Array(readFileSync('engine/zig-out/bin/translator.wasm')));

const text = readFileSync(process.argv[2], 'utf8');
// objdump line: "  <hexaddr>: <hex8>  \t<mnemonic> <operands>"
const re = /^\s*[0-9a-f]+:\s+([0-9a-f]{8})\s+(\S+)(.*)$/;
const byMnem = new Map<string, { count: number; example: string; insn: number }>();
for (const line of text.split('\n')) {
  const m = re.exec(line);
  if (!m) continue;
  const insn = parseInt(m[1], 16) >>> 0;
  const mnem = m[2];
  if (mnem === '.word' || mnem === '.inst' || mnem === 'udf' || mnem === '...') continue; // data / undefined
  if (zigDecodeKind(insn) !== 0) continue; // engine handles it
  const key = mnem;
  const e = byMnem.get(key);
  if (e) e.count++;
  else byMnem.set(key, { count: 1, example: (mnem + m[3]).trim().slice(0, 44), insn });
}
const sorted = [...byMnem.entries()].sort((a, b) => b[1].count - a[1].count);
console.log(`undecoded instruction mnemonics: ${sorted.length}`);
for (const [mnem, { count, example, insn }] of sorted) {
  console.log(`  ${String(count).padStart(6)}  ${mnem.padEnd(12)} 0x${insn.toString(16).padStart(8, '0')}  e.g. ${example}`);
}
