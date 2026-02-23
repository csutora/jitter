import { decode } from './decode.js';
import { isBranch, GUEST_MEM_BASE } from './types.js';
import type { DecodedInsn } from './types.js';

export interface BasicBlock {
  startPC: bigint;
  endPC: bigint;         // PC after the last instruction
  insns: DecodedInsn[];
  isSelfLoop: boolean;   // does backward branch target startPC?
}

export interface Region {
  blocks: BasicBlock[];
  blockMap: Map<bigint, number>;  // PC → index in blocks[]
  entryIndex: number;             // always 0
}

export function detectBlock(memory: DataView, pc: bigint): BasicBlock {
  const insns: DecodedInsn[] = [];
  const startPC = pc;
  let currentPC = pc;

  for (;;) {
    const memOffset = Number(currentPC) + GUEST_MEM_BASE;
    const raw = memory.getUint32(memOffset, true); // little-endian
    const insn = decode(raw);
    insns.push(insn);
    currentPC += 4n;

    if (isBranch(insn) || insn.kind === 'unknown') {
      break;
    }

    // Safety: limit block size
    if (insns.length >= 256) break;
  }

  // Check if this is a self-loop: does the block's branch target its own start?
  let isSelfLoop = false;
  const lastInsn = insns[insns.length - 1];
  if (lastInsn) {
    switch (lastInsn.kind) {
      case 'b': {
        const target = Number(startPC) + insns.length * 4 - 4 + lastInsn.offset;
        isSelfLoop = BigInt(target) === startPC;
        break;
      }
      case 'b_cond':
      case 'cbz':
      case 'cbnz':
      case 'tbz':
      case 'tbnz': {
        const branchPC = Number(startPC) + (insns.length - 1) * 4;
        const target = branchPC + lastInsn.offset;
        isSelfLoop = BigInt(target) === startPC;
        break;
      }
    }

  }

  return {
    startPC,
    endPC: currentPC,
    insns,
    isSelfLoop,
  };
}

// Extract branch target PCs from a block's terminating instruction
function getBranchTargets(block: BasicBlock): bigint[] {
  const lastInsn = block.insns[block.insns.length - 1];
  if (!lastInsn) return [];
  const branchPC = Number(block.endPC) - 4; // PC of last instruction
  const targets: bigint[] = [];

  switch (lastInsn.kind) {
    case 'b':
      targets.push(BigInt(branchPC + lastInsn.offset));
      break;
    case 'b_cond':
    case 'cbz':
    case 'cbnz':
    case 'tbz':
    case 'tbnz':
      targets.push(BigInt(branchPC + lastInsn.offset)); // taken
      targets.push(block.endPC);                         // fallthrough
      break;
    case 'bl':
      targets.push(block.endPC); // fallthrough only (skip call target)
      break;
    // br, blr, ret, svc: no static targets
  }
  return targets;
}

const MAX_REGION_BLOCKS = 16;
const MAX_REGION_INSNS = 256;

export function detectRegion(memory: DataView, entryPC: bigint): Region {
  const blocks: BasicBlock[] = [];
  const blockMap = new Map<bigint, number>();
  const queue: bigint[] = [entryPC];
  let totalInsns = 0;

  while (queue.length > 0 && blocks.length < MAX_REGION_BLOCKS && totalInsns < MAX_REGION_INSNS) {
    const pc = queue.shift()!;
    if (blockMap.has(pc)) continue;

    const block = detectBlock(memory, pc);
    const idx = blocks.length;
    blocks.push(block);
    blockMap.set(pc, idx);
    totalInsns += block.insns.length;

    // Queue branch targets
    for (const target of getBranchTargets(block)) {
      if (!blockMap.has(target)) {
        queue.push(target);
      }
    }
  }

  return { blocks, blockMap, entryIndex: 0 };
}
