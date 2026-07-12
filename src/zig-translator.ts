// zig translator loader. instantiates engine/zig-out/bin/translator.wasm once
// and wraps it with a callback compatible with the ts emulator's CompiledRegion.
//
// the translator.wasm runs in its own linear memory (separate from the guest
// memory). guest instruction bytes are copied into its input buffer; the
// produced wasm module is read out, compiled, and instantiated against the
// guest's actual memory.
//
// shadow_base must match the guest memory layout — every translated store
// atomic.ANDs the shadow bit at that offset.

import { GUEST_MEM_BASE, PC_OFFSET, HALT_PC, SVC_PC } from './types.js';
import type { CompiledRegion } from './types.js';
import { detectBlock } from './block.js';
import { createEnv } from './helpers.js';

// dev-only debug switches. resolved once and guarded, because `process` does not
// exist in the browser/webview worker where this same code runs (referencing it
// unguarded throws "Can't find variable: process" and kills the worker).
const HAS_PROCESS = typeof process !== 'undefined' && !!(process as any).env;
const JIT_DUMP_PC: string = HAS_PROCESS ? ((process as any).env.JIT_DUMP_PC ?? '') : '';
const JIT_DEBUG: boolean = HAS_PROCESS ? !!(process as any).env.JIT_DEBUG : false;
const dbgWrite = (s: string) => { if (HAS_PROCESS) (process as any).stderr.write(s); };

interface TranslatorExports {
  memory: WebAssembly.Memory;
  translator_input_buf_offset(): number;
  translator_input_buf_size(): number;
  translator_output_buf_offset(): number;
  translator_reset(): void;
  translator_translate_region(
    base_pc_lo: number,
    base_pc_hi: number,
    entry_pc_lo: number,
    entry_pc_hi: number,
    code_len: number,
    shadow_base: number,
    shared_monitor: number,
  ): number;
  translator_result_offset(): number;
  translator_result_len(): number;
  translator_decode_kind(insn: number): number;
}

let exports_: TranslatorExports | null = null;
let autoloadTried = false;

/// initialize the zig translator synchronously. `wasmBytes` is the translator.wasm bytecode.
export function initZigTranslator(wasmBytes: Uint8Array | ArrayBuffer): void {
  const bytes = wasmBytes instanceof Uint8Array ? wasmBytes : new Uint8Array(wasmBytes);
  initZigTranslatorModule(new WebAssembly.Module(bytes));
}

/// initialize from an already-compiled module. the kernel compiles translator.wasm ONCE on the
/// main thread and structured-clones the WebAssembly.Module to every worker (compiled modules
/// share their code across agents), so a fork storm doesn't pay N x a 1.5MB compile — the spike
/// that dominated fork-storm OOM. each worker still needs its own Instance (its own scratch mem).
export function initZigTranslatorModule(mod: WebAssembly.Module): void {
  const instance = new WebAssembly.Instance(mod);
  exports_ = instance.exports as unknown as TranslatorExports;
}

/// under node, load translator.wasm from the engine build output on first use so
/// tests and tools don't each need explicit init. browser contexts must call
/// initZigTranslator with fetched bytes (the worker does this at startup).
function ensureInit(): boolean {
  if (exports_) return true;
  if (autoloadTried) return false;
  autoloadTried = true;
  if (typeof process === 'undefined' || !process.versions?.node) return false;
  try {
    const fs = process.getBuiltinModule('fs');
    const path = process.getBuiltinModule('path');
    const url = process.getBuiltinModule('url');
    const here = path.dirname(url.fileURLToPath(import.meta.url));
    const bytes = fs.readFileSync(path.join(here, '..', 'engine', 'zig-out', 'bin', 'translator.wasm'));
    initZigTranslator(new Uint8Array(bytes));
    return true;
  } catch {
    return false;
  }
}

/// reads guest code spanning [pc - back, pc + forward) and returns the bytes
/// together with the base pc the buffer corresponds to (used as base_pc for the translator)
function readGuestBytesWindow(view: DataView, pc: bigint, back: number, forward: number): { bytes: Uint8Array; basePC: bigint } {
  const memSize = view.byteLength;
  let basePCNum = Number(pc) - back;
  if (basePCNum < 0) basePCNum = 0;
  const startOffset = basePCNum + GUEST_MEM_BASE;
  if (startOffset >= memSize) return { bytes: new Uint8Array(0), basePC: pc };
  const endOffset = Math.min(Number(pc) + GUEST_MEM_BASE + forward, memSize);
  const bytes = new Uint8Array(view.buffer, startOffset, endOffset - startOffset).slice();
  return { bytes, basePC: BigInt(basePCNum) };
}

/// reads up to `maxBytes` of guest code starting at `pc` from the guest memory.
/// returns the number of bytes actually read (bounded by the memory limit).
function readGuestBytes(view: DataView, pc: bigint, maxBytes: number): Uint8Array {
  const baseOffset = Number(pc) + GUEST_MEM_BASE;
  const memSize = view.byteLength;
  const limit = Math.min(maxBytes, memSize - baseOffset);
  if (limit <= 0) return new Uint8Array(0);
  return new Uint8Array(view.buffer, baseOffset, limit).slice();
}

/// translate a region starting at entryPC using the zig translator.
/// returns null if translation fails (caller falls back to ts translator).
///
/// the produced wasm module imports `env.memory`, which must be the guest's
/// shared memory containing both guest data (at GUEST_MEM_BASE) and the
/// exclusive-monitor shadow bitmap (at shadowBase).
export function translateRegionZig(
  view: DataView,
  entryPC: bigint,
  memory: WebAssembly.Memory,
  shadowBase: number,
  singleBlock = false,
  threaded = false,
  regBase: WebAssembly.Global | null = null,
): CompiledRegion | null {
  if (!ensureInit()) return null;

  let codeBytes: Uint8Array;
  const basePC = entryPC;
  if (singleBlock) {
    const block = detectBlock(view, entryPC);
    codeBytes = readGuestBytes(view, entryPC, block.insns.length * 4);
  } else {
    // multi-block: read a forward window. backward branch targets outside this window
    // will cause rc=2 from the block builder and we fall back to ts (acceptable in practice)
    codeBytes = readGuestBytes(view, entryPC, 4096);
  }
  if (codeBytes.length < 4) return null;

  const ex = exports_;
  const tbuf = new Uint8Array(ex.memory.buffer);
  const inOff = ex.translator_input_buf_offset();
  tbuf.set(codeBytes, inOff);

  ex.translator_reset();

  const basePCLo = Number(basePC & 0xffffffffn);
  const basePCHi = Number((basePC >> 32n) & 0xffffffffn);
  const entryLo = Number(entryPC & 0xffffffffn);
  const entryHi = Number((entryPC >> 32n) & 0xffffffffn);

  // shared_monitor: threaded processes emit reg_base-relative registers, real atomics (LDXR/STXR
  // shadow protocol + LSE), and fences on a SHARED memory. single-threaded (each fork has its own
  // private memory) uses plain value-CAS and skips the shadow protocol entirely.
  const rc = ex.translator_translate_region(basePCLo, basePCHi, entryLo, entryHi, codeBytes.length, shadowBase, threaded ? 1 : 0);
  if (rc !== 0) return null;

  const outLen = ex.translator_result_len();
  const outOff = ex.translator_result_offset();
  const moduleBytes = new Uint8Array(outLen);
  const tbufFresh = new Uint8Array(ex.memory.buffer); // re-read in case translator memory grew
  moduleBytes.set(tbufFresh.subarray(outOff, outOff + outLen));

  let mod: WebAssembly.Module;
  try {
    mod = new WebAssembly.Module(moduleBytes);
  } catch {
    return null;
  }

  if (JIT_DUMP_PC && entryPC.toString(16) === JIT_DUMP_PC) {
    dbgWrite(`zig dump for pc=0x${entryPC.toString(16)} module bytes:\n`);
    let hex = '';
    for (let i = 0; i < moduleBytes.length; i++) hex += moduleBytes[i].toString(16).padStart(2, '0');
    dbgWrite(hex + '\n');
    dbgWrite('source bytes:\n');
    let src = '';
    for (let i = 0; i < codeBytes.length && i < 64; i++) src += codeBytes[i].toString(16).padStart(2, '0');
    dbgWrite(src + '\n');
  }

  let inst: WebAssembly.Instance;
  try {
    inst = new WebAssembly.Instance(mod, { env: createEnv(memory, regBase) });
  } catch {
    return null;
  }

  const runZig = inst.exports.run as () => bigint;
  // computed branches (ret/br/blr) write the 64-bit target to the register block's PC slot, which
  // in threaded mode sits at reg_base + PC_OFFSET (not the absolute low offset). read it there.
  const pcSlot = (regBase ? (regBase.value as number) : 0) + PC_OFFSET;

  let callCount = 0;
  const region: CompiledRegion & { lastExitReason?: number } = {
    run: (): bigint => {
      const packed = runZig();
      const reason = Number((packed >> 32n) & 0xffffn);
      region.lastExitReason = reason;
      const view = new DataView(memory.buffer);
      const nextPC = view.getBigUint64(pcSlot, true);
      callCount++;
      if (JIT_DEBUG && callCount <= 200) {
        dbgWrite(`#${callCount} zig entry=0x${entryPC.toString(16)} reason=${reason} -> 0x${nextPC.toString(16)}\n`);
      }
      switch (reason) {
        case 0: return HALT_PC;
        case 1: return SVC_PC;
        default: return nextPC;
      }
    },
    entryPC,
    blockPCs: [entryPC], // we don't expose per-block PCs from zig yet; entry only
  };
  return region;
}

/// quick lookup: is the instruction at `insn` decodable by the zig translator?
/// returns 0 if "unknown", nonzero kind otherwise. used for pre-scanning before
/// committing to a zig translation
export function zigDecodeKind(insn: number): number {
  if (!exports_) throw new Error('zig translator not initialized');
  return exports_.translator_decode_kind(insn);
}
