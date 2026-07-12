import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    // tests/dynamic holds rootfs fixtures (alpine minirootfs, glibc libs), not tests.
    // some are symlinks into macos-protected paths that error on scandir, so keep the
    // file walker out of there entirely.
    include: ['tests/**/*.test.ts'],
    exclude: ['node_modules/**', 'tests/dynamic/**'],
  },
});
