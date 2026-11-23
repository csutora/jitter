#!/usr/bin/env bash
# summarise an `ltp` sweep captured from the device: totals, then the non-passing tests grouped by
# their subsystem prefix so a cluster is obvious at a glance rather than buried in 285 lines.
#   tools/ltp-report.sh <sweep-output-file>
set -uo pipefail

raw="${1:?usage: ltp-report.sh <sweep-output>}"
# the device shell speaks CRLF, and a trailing \r makes every status fail an anchored match
f="$(mktemp)"
trap 'rm -f "$f"' EXIT
tr -d '\r' < "$raw" > "$f"

echo "== totals =="
awk '{print $2}' "$f" | grep -E '^(PASS|FAIL|BROK|SKIP|TIMEOUT|EXIT[0-9]+)$' | sort | uniq -c | sort -rn

total=$(awk '{print $2}' "$f" | grep -cE '^(PASS|FAIL|BROK|SKIP|TIMEOUT|EXIT[0-9]+)$' || true)
pass=$(awk '{print $2}' "$f" | grep -c '^PASS$' || true)
skip=$(awk '{print $2}' "$f" | grep -c '^SKIP$' || true)
if [ "$total" -gt 0 ]; then
  echo
  echo "ran $total | pass $pass ($((pass * 100 / total))%) | of the non-skipped: $((pass * 100 / (total - skip)))%"
fi

echo
echo "== not passing, by subsystem =="
grep -vE '\<(PASS|SKIP)\>' "$f" \
  | grep -E '\<(FAIL|BROK|TIMEOUT|EXIT[0-9]+)\>' \
  | sed 's/[0-9_]*$//' \
  | awk '{print $1}' \
  | sed -E 's/[0-9].*$//' \
  | sort | uniq -c | sort -rn

echo
echo "== the list =="
grep -vE '\<(PASS|SKIP)\>' "$f" | grep -E '\<(FAIL|BROK|TIMEOUT|EXIT[0-9]+)\>'
