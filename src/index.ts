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

(window as any).bootShell = async () => {
  const btn = document.getElementById('btn-shell') as HTMLButtonElement;
  btn.disabled = true;
  btn.textContent = 'Loading...';

  try {
    // Setup native bridge (active only in WKWebView)
    const bridge = new NativeBridge();
    if (bridge.isNative) {
      console.log('[jitter] Native bridge detected — filesystem calls will route to Swift');
    }

    // Load busybox binary — via bridge in native app, via fetch in browser
    let elfBytes: Uint8Array;
    if (bridge.isNative) {
      const resp = await bridge.call('load_bundle_file', { name: 'busybox-aarch64' });
      if (resp.error) throw new Error(`Failed to load busybox via bridge: ${resp.error}`);
      const binary = atob(resp.data as string);
      elfBytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) elfBytes[i] = binary.charCodeAt(i);
    } else {
      const resp = await fetch('busybox-aarch64');
      if (!resp.ok) throw new Error(`Failed to fetch busybox: ${resp.status}`);
      elfBytes = new Uint8Array(await resp.arrayBuffer());
    }

    const statusBar = document.getElementById('status-bar')!;
    statusBar.classList.add('active');
    btn.style.display = 'none';
    statusBar.textContent = 'Booting...';

    const writeOutput = bridge.isNative
      ? (data: Uint8Array) => {
          bridge.call('term_data', { data: uint8ToBase64(data) }).catch(() => {});
        }
      : makeBrowserSink();

    // ── Multicore path: use Kernel + Web Workers when cross-origin isolated ──
    if (!self.crossOriginIsolated) {
      throw new Error(
        'crossOriginIsolated is false. Serve with COOP=same-origin and COEP=require-corp headers.'
      );
    }

    statusBar.textContent = 'Booting...';

    const kernel = new Kernel({
      elfBytes,
      workerURL: 'worker.js',
      onOutput: (data: Uint8Array) => {
        writeOutput(appendScrollbackWipeAfterClear(translateNewlines(data)));
      },
      bridge: bridge.isNative ? bridge : undefined,
    });

    const feedStdin = (bytes: Uint8Array) => {
      if (bytes.length === 1 && bytes[0] === 0x03) {
        kernel.sendInterrupt();
        writeOutput(new Uint8Array([0x5E, 0x43, 0x0D, 0x0A]));
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

    kernel.boot(
      ['sh'],
      ['HOME=/root', 'TERM=xterm-256color', 'PATH=/bin', 'PWD=/'],
    );

    statusBar.textContent = 'Running';
  } catch (err: any) {
    console.error('Boot failed:', err);
    const sb = document.getElementById('status-bar');
    if (sb) sb.textContent = `Boot failed: ${err.message}`;
    btn.disabled = false;
    btn.textContent = 'Boot Shell';
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
  window.addEventListener('DOMContentLoaded', () => {
    (window as any).bootShell?.();
  });
  if (document.readyState !== 'loading') (window as any).bootShell?.();
}
