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
