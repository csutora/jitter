import { NativeBridge } from './bridge.js';
import { Kernel } from './kernel.js';

function uint8ToBase64(data: Uint8Array): string {
  const chunk = 0x8000;
  let s = '';
  for (let i = 0; i < data.length; i += chunk) {
    s += String.fromCharCode.apply(null, data.subarray(i, i + chunk) as unknown as number[]);
  }
  return btoa(s);
}

function makeBrowserSink(): (data: Uint8Array) => void {
  const pre = document.createElement('pre');
  pre.id = 'fallback-term';
  pre.style.cssText = 'background:#000;color:#0f0;font-family:Menlo,monospace;font-size:14px;padding:8px;white-space:pre-wrap;outline:none;height:80vh;overflow:auto;margin:0;';
  pre.tabIndex = 0;
  document.body.appendChild(pre);
  pre.focus();
  const decoder = new TextDecoder();
  return (data) => {
    pre.append(decoder.decode(data, { stream: true }));
    pre.scrollTop = pre.scrollHeight;
  };
}

function browserKeyToBytes(e: KeyboardEvent): Uint8Array | null {
  if (e.ctrlKey && e.key.length === 1) {
    const c = e.key.toLowerCase().charCodeAt(0);
    if (c >= 0x60 && c < 0x80) return new Uint8Array([c - 0x60]);
  }
  switch (e.key) {
    case 'Enter': return new Uint8Array([0x0D]);
    case 'Backspace': return new Uint8Array([0x7F]);
    case 'Tab': return new Uint8Array([0x09]);
    case 'Escape': return new Uint8Array([0x1B]);
    case 'ArrowUp': return new Uint8Array([0x1B, 0x5B, 0x41]);
    case 'ArrowDown': return new Uint8Array([0x1B, 0x5B, 0x42]);
    case 'ArrowRight': return new Uint8Array([0x1B, 0x5B, 0x43]);
    case 'ArrowLeft': return new Uint8Array([0x1B, 0x5B, 0x44]);
  }
  if (e.key.length === 1 && !e.metaKey && !e.altKey) {
    return new TextEncoder().encode(e.key);
  }
  return null;
}

function installInputSources(isNative: boolean, feed: (bytes: Uint8Array) => void): void {
  if (isNative) {
    (window as any).__jitterStdin = (b64: string) => {
      const bin = atob(b64);
      const out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      feed(out);
    };
    return;
  }
  document.addEventListener('keydown', (e) => {
    const bytes = browserKeyToBytes(e);
    if (bytes) {
      e.preventDefault();
      feed(bytes);
    }
  });
}

function appendScrollbackWipeAfterClear(data: Uint8Array): Uint8Array {
  let count = 0;
  for (let i = 0; i + 5 < data.length; i++) {
    if (data[i] === 0x1B && data[i + 1] === 0x5B && data[i + 2] === 0x48 &&
        data[i + 3] === 0x1B && data[i + 4] === 0x5B && data[i + 5] === 0x4A) {
      count++;
    }
  }
  if (count === 0) return data;
  const out = new Uint8Array(data.length + count * 4);
  let j = 0;
  let i = 0;
  while (i < data.length) {
    if (i + 5 < data.length &&
        data[i] === 0x1B && data[i + 1] === 0x5B && data[i + 2] === 0x48 &&
        data[i + 3] === 0x1B && data[i + 4] === 0x5B && data[i + 5] === 0x4A) {
      out[j++] = 0x1B; out[j++] = 0x5B; out[j++] = 0x48;
      out[j++] = 0x1B; out[j++] = 0x5B; out[j++] = 0x4A;
      out[j++] = 0x1B; out[j++] = 0x5B; out[j++] = 0x33; out[j++] = 0x4A;
      i += 6;
    } else {
      out[j++] = data[i];
      i++;
    }
  }
  return out;
}

function translateNewlines(data: Uint8Array): Uint8Array {
  let extra = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i] === 0x0A && (i === 0 || data[i - 1] !== 0x0D)) extra++;
  }
  if (extra === 0) return data;

  const out = new Uint8Array(data.length + extra);
  let j = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i] === 0x0A && (i === 0 || data[i - 1] !== 0x0D)) {
      out[j++] = 0x0D;
    }
    out[j++] = data[i];
  }
  return out;
}



// ── Shell boot ──────────────────────────────────────────────────────────────

let bootStarted = false;
(window as any).bootShell = async () => {
  // idempotent: the native auto-boot can fire from both the DOMContentLoaded listener
  // and the readyState check, and a double boot spins up two competing kernels/shells
  // that deadlock each other (no prompt).
  if (bootStarted) return;
  bootStarted = true;

  const btn = document.getElementById('btn-shell') as HTMLButtonElement;
  if (btn) { btn.disabled = true; btn.textContent = 'Loading...'; }

  // hoisted so the catch handler can surface errors on the terminal itself.
  // in the native app the status bar is hidden behind the terminal, so a thrown
  // boot error would otherwise be completely invisible.
  let writeOutput: ((data: Uint8Array) => void) | null = null;
  const term = (s: string) => writeOutput?.(new TextEncoder().encode(s));

  try {
    // Setup native bridge (active only in WKWebView)
    const bridge = new NativeBridge();
    if (bridge.isNative) {
      console.log('[jitter] Native bridge detected — filesystem calls will route to Swift');
    }

    // bring the output path up FIRST, so every later step (busybox load, translator
    // fetch, kernel boot) can report failures onto the visible terminal. if this
    // banner never appears, the terminal rendering path itself is broken.
    writeOutput = bridge.isNative
      ? (data: Uint8Array) => {
          bridge.call('term_data', { data: uint8ToBase64(data) }).catch(() => {});
        }
      : makeBrowserSink();
    term('[jitter] boot: output path up\r\n');

    const statusBar = document.getElementById('status-bar')!;
    statusBar.classList.add('active');
    if (btn) btn.style.display = 'none';
    statusBar.textContent = 'Booting...';

    // Load the init binary. in the native app, boot the ACTIVE root's login shell so `jitter switch`
    // boots the chosen distro (the worker's exec path is dynamic-aware, so alpine's musl /bin/sh just
    // works). fall back to the bundled static busybox if the root has no shell yet (fresh install).
    const b64ToBytes = (s: string) => { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; };
    let elfBytes: Uint8Array | null = null;
    if (bridge.isNative) {
      term('[jitter] boot: loading init shell...\r\n');
      for (const p of ['/bin/sh', '/bin/busybox']) {
        const r = await bridge.call('load_root_file', { path: p });
        if (!r.error && typeof r.data === 'string') { elfBytes = b64ToBytes(r.data); break; }
      }
      if (!elfBytes) {
        const resp = await bridge.call('load_bundle_file', { name: 'busybox-aarch64' });
        if (resp.error) throw new Error(`Failed to load init shell (no root shell, no bundle): ${resp.error}`);
        elfBytes = b64ToBytes(resp.data as string);
      }
    } else {
      const resp = await fetch('busybox-aarch64');
      if (!resp.ok) throw new Error(`Failed to fetch busybox: ${resp.status}`);
      elfBytes = new Uint8Array(await resp.arrayBuffer());
    }
    term(`[jitter] boot: init loaded (${elfBytes!.length} bytes)\r\n`);

    // ── Multicore path: use Kernel + Web Workers when cross-origin isolated ──
    if (!self.crossOriginIsolated) {
      throw new Error(
        'crossOriginIsolated is false. Serve with COOP=same-origin and COEP=require-corp headers.'
      );
    }

    statusBar.textContent = 'Booting...';

    // the zig translator is mandatory now (the ts translator was removed). a failed
    // fetch here means every worker will throw on init, so make it loud, not silent.
    term('[jitter] boot: fetching translator.wasm...\r\n');
    let zigTranslatorBytes: ArrayBuffer;
    const tresp = await fetch('translator.wasm');
    if (!tresp.ok) throw new Error(`translator.wasm fetch failed: HTTP ${tresp.status}`);
    zigTranslatorBytes = await tresp.arrayBuffer();
    term(`[jitter] boot: translator ready (${zigTranslatorBytes.byteLength} bytes)\r\n`);

    const out = writeOutput!;
    const kernel = new Kernel({
      elfBytes: elfBytes!,
      workerURL: 'worker.js',
      onOutput: (data: Uint8Array) => {
        out(appendScrollbackWipeAfterClear(translateNewlines(data)));
      },
      bridge: bridge.isNative ? bridge : undefined,
      zigTranslatorBytes,
    });

    const feedStdin = (bytes: Uint8Array) => {
      if (bytes.length === 1 && bytes[0] === 0x03) {
        kernel.sendInterrupt();
        out(new Uint8Array([0x5E, 0x43, 0x0D, 0x0A]));
        return;
      }
      kernel.pushStdin(bytes);
    };

    installInputSources(bridge.isNative, feedStdin);

    (window as any).__jitterTermSize = (cols: number, rows: number) => {
      kernel.setTermSize(cols, rows);
    };

    if (bridge.isNative) {
      const sizeResp = await bridge.call('term_size_query');
      const cols = Number(sizeResp.cols) || 0;
      const rows = Number(sizeResp.rows) || 0;
      if (cols > 0 && rows > 0) {
        kernel.setTermSize(cols, rows);
      }
    }

    term('[jitter] boot: starting sh (if no prompt follows, the emulator/worker stalled)\r\n');
    kernel.boot(
      ['/bin/sh'],
      ['HOME=/root', 'TERM=xterm-256color', 'PATH=/bin:/usr/bin:/sbin:/usr/sbin', 'PWD=/'],
    );

    statusBar.textContent = 'Running';
  } catch (err: any) {
    console.error('Boot failed:', err);
    // surface the failure on the terminal (visible) and the status bar (browser)
    term(`\r\n[jitter] boot failed: ${err?.message ?? err}\r\n${err?.stack ?? ''}\r\n`);
    const sb = document.getElementById('status-bar');
    if (sb) sb.textContent = `Boot failed: ${err.message}`;
    if (btn) { btn.disabled = false; btn.textContent = 'Boot Shell'; }
  }
};

if ((window as any).webkit?.messageHandlers?.jitter) {
  document.documentElement.style.background = '#000';
  document.body.style.background = '#000';
  const hide = (id: string) => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  };
  for (const id of ['controls', 'status-bar']) hide(id);
  // boot exactly once: either the DOM is already parsed (fire now) or wait for it.
  // bootShell is also idempotent as a backstop.
  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', () => (window as any).bootShell?.(), { once: true });
  } else {
    (window as any).bootShell?.();
  }
}
