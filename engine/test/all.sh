#!/bin/sh
# run every engine test through jsc.
set -e
cd "$(dirname "$0")/.."

JSC=/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc

for t in test/smoke.js test/e2e.js test/branches.js test/loop.js test/cmp.js test/csel.js test/fib.js test/stp_ldp.js test/bitfield.js test/multiply.js test/ldr_reg.js test/ldxr_stxr.js test/stxr_fail.js test/simd_ldst.js test/multiblock.js test/multiblock_loop.js; do
  printf '== %-25s ==\n' "$(basename "$t")"
  "$JSC" "$t"
  echo
done
