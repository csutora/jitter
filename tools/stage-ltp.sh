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

# the corpus keeps its <subdir>_<test> filenames so the source tree records where each test came
# from, but they are staged under the bare upstream name. several tests re-exec themselves through
# ltp's self_exec, which looks the binary up by its own name -- under a renamed binary that exec
# fails ENOENT and the test reports TBROK before it has tested anything. upstream installs them
# flat under these names too, and stripping the prefix leaves them unique.
n=0
for f in "$src"/*; do
  [ -f "$f" ] || continue
  base="$(basename "$f")"
  name="${base#*_}"
  if command -v llvm-strip >/dev/null 2>&1; then
    llvm-strip -o "$stage/$name" "$f" 2>/dev/null || cp "$f" "$stage/$name"
  elif command -v aarch64-unknown-linux-gnu-strip >/dev/null 2>&1; then
    aarch64-unknown-linux-gnu-strip -o "$stage/$name" "$f" 2>/dev/null || cp "$f" "$stage/$name"
  else
    cp "$f" "$stage/$name"
  fi
  chmod 755 "$stage/$name"
  n=$((n+1))
done

( cd "$stage" && tar cf - . ) > "$out"
echo "staged $n ltp binaries -> $out ($(du -h "$out" | cut -f1))"
