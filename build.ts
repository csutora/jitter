import * as esbuild from 'esbuild';
import { readFileSync, writeFileSync, mkdirSync, existsSync, copyFileSync } from 'fs';

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

// Read xterm.js CSS
let xtermCss = '';
const xtermCssPath = 'node_modules/@xterm/xterm/css/xterm.css';
if (existsSync(xtermCssPath)) {
  xtermCss = `<style>${readFileSync(xtermCssPath, 'utf-8')}</style>`;
}

const html = readFileSync('index.html', 'utf-8');
const output = html
  .replace('/* BUNDLE */', js)
  .replace('</head>', `${xtermCss}\n</head>`);
writeFileSync('dist/index.html', output);
console.log('Built dist/index.html');

// Copy busybox binary if present
const busyboxPath = 'assets/busybox-aarch64';
if (existsSync(busyboxPath)) {
  copyFileSync(busyboxPath, 'dist/busybox-aarch64');
  console.log('Copied busybox-aarch64 to dist/');
} else {
  console.log('Note: assets/busybox-aarch64 not found — Boot Shell will not work');
}
