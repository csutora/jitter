import * as esbuild from 'esbuild';
import { readFileSync, writeFileSync, mkdirSync, existsSync, copyFileSync } from 'fs';
import { execFileSync } from 'child_process';

// esbuild strips types without checking them, so a ReferenceError-class mistake (a renamed field,
// a handler missing a parameter) bundles cleanly and only shows up as a dead syscall on the
// device, half an hour of build-deploy-reproduce later. typecheck first; it costs a few seconds.
// JITTER_SKIP_TSC=1 escapes it when you knowingly want a bundle from a broken tree.
if (!process.env.JITTER_SKIP_TSC) {
  try {
    execFileSync('npx', ['tsc', '--noEmit', '-p', 'tsconfig.json'], { stdio: 'inherit' });
  } catch {
    console.error('\ntypecheck failed -- not bundling. set JITTER_SKIP_TSC=1 to bundle anyway.');
    process.exit(1);
  }
}

mkdirSync('dist', { recursive: true });

// Bundle main entry point
const result = await esbuild.build({
  entryPoints: ['src/index.ts'],
  bundle: true,
  format: 'esm',
  target: 'es2022',
  minify: false,
  write: false,
});

const js = result.outputFiles![0].text;

// Bundle worker entry point (separate file — Workers can't be inlined)
// Safari/WebKit doesn't support module Workers, so use IIFE format
await esbuild.build({
  entryPoints: ['src/worker.ts'],
  bundle: true,
  format: 'iife',
  target: 'es2022',
  minify: false,
  outfile: 'dist/worker.js',
});
console.log('Built dist/worker.js');

const html = readFileSync('index.html', 'utf-8');
const output = html.replace('/* BUNDLE */', js);
writeFileSync('dist/index.html', output);
console.log('Built dist/index.html');

writeFileSync('dist/serve.json', JSON.stringify({
  headers: [{
    source: '**/*',
    headers: [
      { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
      { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
    ],
  }],
}, null, 2));

// Copy busybox binary if present
const busyboxPath = 'assets/busybox-aarch64';
if (existsSync(busyboxPath)) {
  copyFileSync(busyboxPath, 'dist/busybox-aarch64');
  console.log('Copied busybox-aarch64 to dist/');
} else {
  console.log('Note: assets/busybox-aarch64 not found — Boot Shell will not work');
}

// Copy zig translator.wasm if built — worker.js fetches it at startup
const translatorPath = 'engine/zig-out/bin/translator.wasm';
if (existsSync(translatorPath)) {
  copyFileSync(translatorPath, 'dist/translator.wasm');
  console.log('Copied translator.wasm to dist/');
} else {
  console.log('Note: engine/zig-out/bin/translator.wasm not found — zig translator disabled. run `just build-engine` first.');
}

const iosResources = 'Jitter/Jitter/Resources';
if (existsSync(iosResources)) {
  for (const f of ['index.html', 'worker.js', 'busybox-aarch64', 'translator.wasm']) {
    const src = `dist/${f}`;
    if (existsSync(src)) copyFileSync(src, `${iosResources}/${f}`);
  }
  console.log(`Synced bundle into ${iosResources}/`);
}
