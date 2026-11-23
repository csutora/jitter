default:
    @just --list

# Install dependencies
setup:
    nix-shell --run "npm install"

# Run tests
test:
    nix-shell --run "npx vitest run"

# Run tests in watch mode
dev:
    nix-shell --run "npx vitest"

# Build dist/index.html
build:
    nix-shell --run "npx tsx build.ts"

# Typecheck without bundling (build does this too, and refuses to bundle if it fails)
check:
    nix-shell -p nodejs --run "npx tsc --noEmit -p tsconfig.json"

# Host-side unit tests: pure logic that needs no device (translator, locks, elf)
test-host:
    nix-shell -p nodejs --run "npx vitest run tests/neon.test.ts tests/locks.test.ts tests/shared-map.test.ts tests/pipe-ring.test.ts tests/binfmt.test.ts tests/elf.test.ts"

# Build and serve locally
serve: build
    nix-shell --run "npx -y serve dist"

# Build zig engine (translator.wasm)
build-engine:
    cd engine && nix-shell -p zig_0_15 --run "zig build --release=fast"

# Run engine native tests
test-engine:
    cd engine && nix-shell -p zig_0_15 --run "zig build test --summary all"

# Run all engine wasm tests through JSC
test-engine-jsc: build-engine
    engine/test/all.sh

# Build ghostty-vt.xcframework into Jitter/Frameworks
build-ghostty:
    cd ghostty && nix develop --command zig build \
        -Demit-lib-vt=true \
        -Demit-xcframework=true \
        -Dxcframework-target=universal \
        -Doptimize=ReleaseFast
    mkdir -p Jitter/Frameworks
    rm -rf Jitter/Frameworks/ghostty-vt.xcframework
    cp -R ghostty/zig-out/lib/ghostty-vt.xcframework Jitter/Frameworks/

clean-ghostty:
    rm -rf ghostty/.zig-cache ghostty/zig-out ghostty/macos/GhosttyKit.xcframework
    rm -rf Jitter/Frameworks/GhosttyKit.xcframework Jitter/Frameworks/ghostty-vt.xcframework

# build the bundle, build the app, install it and relaunch. every step is checked: a swift compile
# error used to leave the previous app binary sitting in derived data, so `devicectl install`
# happily shipped a stale executable next to freshly synced javascript, and the failure looked like
# a bug in whatever was being tested.
deploy:
    #!/usr/bin/env bash
    set -euo pipefail
    nix-shell -p nodejs --run "npx tsc --noEmit -p ."
    nix-shell -p nodejs --run "npx tsx build.ts"
    out=$(xcrun xcodebuild -project Jitter/Jitter.xcodeproj -scheme Jitter -configuration Debug \
        -destination 'generic/platform=iOS' -derivedDataPath /tmp/jitterbuild \
        -allowProvisioningUpdates build 2>&1) || true
    if ! grep -q "BUILD SUCCEEDED" <<<"$out"; then
        grep -E "error:" <<<"$out" | head
        echo "app build failed"
        exit 1
    fi
    xcrun devicectl device install app --device maxwell \
        /tmp/jitterbuild/Build/Products/Debug-iphoneos/Jitter.app >/dev/null
    xcrun devicectl device process launch --terminate-existing --device maxwell com.csutora.Jitter >/dev/null
    echo deployed
