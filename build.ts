import * as esbuild from 'esbuild';
import { readFileSync, writeFileSync, mkdirSync } from 'fs';

mkdirSync('dist', { recursive: true });

const result = await esbuild.build({
  entryPoints: ['src/index.ts'],
  bundle: true,
  format: 'esm',
  target: 'es2022',
  minify: false,
  write: false,
});

const js = result.outputFiles![0].text;
const html = readFileSync('index.html', 'utf-8');
const output = html.replace('/* BUNDLE */', js);
writeFileSync('dist/index.html', output);
console.log('Built dist/index.html');
