#!/usr/bin/env bash
# strip the in-repo ltp test binaries and pack them into a single app resource (ltp.tar,
# UNCOMPRESSED). the /bin/ltp driver extracts it with `tar xf` on first run -- plain tar
# needs no decompressor child, so it avoids busybox's fork+pipe+inflate path entirely
# (guest gzip is a separate, fork-dependent concern). re-run when the ltp set changes.
set -euo pipefail

src="tests/conformance/bin/ltp"
out="Jitter/Jitter/Resources/ltp.tar"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

[ -d "$src" ] || { echo "no ltp binaries at $src"; exit 1; }

n=0
for f in "$src"/*; do
  [ -f "$f" ] || continue
  if command -v llvm-strip >/dev/null 2>&1; then
    llvm-strip -o "$stage/$(basename "$f")" "$f" 2>/dev/null || cp "$f" "$stage/$(basename "$f")"
  elif command -v aarch64-unknown-linux-gnu-strip >/dev/null 2>&1; then
    aarch64-unknown-linux-gnu-strip -o "$stage/$(basename "$f")" "$f" 2>/dev/null || cp "$f" "$stage/$(basename "$f")"
  else
    cp "$f" "$stage/$(basename "$f")"
  fi
  chmod 755 "$stage/$(basename "$f")"
  n=$((n+1))
done

( cd "$stage" && tar cf - . ) > "$out"
echo "staged $n ltp binaries -> $out ($(du -h "$out" | cut -f1))"
