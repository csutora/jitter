#!/bin/sh
# run every engine test through jsc.
#
# the tests report by PRINTING "PASS ..." or "FAIL: ...", and jsc exits 0 either way -- it has no
# process.exit and an uncaught rejection does not set a status. so the result is read out of the
# output, and one broken test no longer hides every test after it (the old `set -e` did).
cd "$(dirname "$0")/.."

JSC=/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc

failed=""
for t in test/smoke.js test/e2e.js test/branches.js test/loop.js test/cmp.js test/csel.js \
         test/fib.js test/stp_ldp.js test/bitfield.js test/multiply.js test/ldr_reg.js \
         test/ldxr_stxr.js test/stxr_fail.js test/ldxp_stxp.js test/simd_ldst.js \
         test/multiblock.js test/multiblock_loop.js; do
  name=$(basename "$t")
  printf '== %-25s ==\n' "$name"
  output=$("$JSC" "$t" 2>&1)
  echo "$output"
  echo
  case "$output" in
    *FAIL*) failed="$failed $name" ;;
    *PASS*) ;;
    *) failed="$failed $name(no-verdict)" ;;
  esac
done

if [ -n "$failed" ]; then
  echo "FAILED:$failed"
  exit 1
fi
echo "all engine tests passed"
