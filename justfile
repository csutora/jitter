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
