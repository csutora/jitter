import { readFileSync } from 'fs';
import { initZigTranslator, zigDecodeKind } from '../src/zig-translator.js';
await initZigTranslator(new Uint8Array(readFileSync('engine/zig-out/bin/translator.wasm')));
for (const i of [0xd4207d00, 0xd4200000, 0xd4200020]) console.log(`${zigDecodeKind(i)!==0?'OK ':'GAP'} 0x${i.toString(16)} (brk #${(i>>>5)&0xffff})`);
