import { NativeBridge } from './bridge.js';
import { Kernel } from './kernel.js';
import { Emulator } from './dispatch.js';
import { initZigTranslator } from './zig-translator.js';
import { GUEST_MEM_BASE, V_OFFSET } from './types.js';

// the guest's environment. shared by the boot shell and by /ctl/run so a directly-run binary sees
// exactly what a shell-launched one does.
// LTP_* advertise our environment to the ltp harness: one host-backed namespace (a tmpfs, since we
// have no block layer / mkfs) and a scratch device path, so device-requiring tests run on tmpfs
// instead of failing to acquire a loop device. inert for every non-ltp program.
// /opt/ltp/testcases/bin on PATH lets the old-api tests that re-exec themselves (self_exec via execvp
// with the bare test name, e.g. fcntl07 -T <case>) find their own binary, as on a real ltp install.
const GUEST_ENV = [
  'HOME=/root', 'TERM=xterm-256color',
  'PATH=/bin:/usr/bin:/sbin:/usr/sbin:/opt/ltp/testcases/bin', 'PWD=/',
  'LTP_DEV=/tmp/.ltp-scratch.img', 'LTP_SINGLE_FS_TYPE=tmpfs',
];

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

    // /ctl/run: execute ONE binary as its own process with its output captured, bypassing the shell
    // and the tty entirely. this is how a root whose init/shell doesn't load can still be driven and
    // debugged, and it can't drop input the way typing into a busy shell does.
    (window as any).__jitterRun = (path: string, argv: string[], timeoutMs: number) =>
      kernel.runDetached(path, argv ?? [], GUEST_ENV, timeoutMs || 30000);

    // ── differential test entry point ────────────────────────────────────────
// the translator emits wasm, and the wasm is run by whatever engine is hosting us. that means a
// differential test executed on a development machine validates "translator + that machine's
// engine", not the thing that ships. the two are not interchangeable: wasm leaves NaN payload
// propagation to the implementation, and engines differ on which proposals they even accept. so
// the subject side of the comparison runs HERE, on the device, and only the reference (real
// hardware) and the comparison stay on the host.
//
// takes a batch of records so a sweep is a handful of round trips rather than one per case.
// record layout matches tools/difftest.ts: u32 insn, i32 base_reg, then the register state.
    (window as any).__jitterDiff = (() => {
      let emu: Emulator | null = null;
      const NREG = 28, NVEC = 28, SCRATCH = 512;
      const STATE = NREG * 8 + 16 + NVEC * 16 + SCRATCH;
      const REC = 8 + STATE;
      const PC = 0x600000, GUEST_SCRATCH = 0x710000, NZCV_OFFSET = 0x108;
      return (b64: string): string => {
        // the page never needed the translator itself (the workers each init their own), so it
        // has to be brought up here before a region can be built
        if (!emu) { initZigTranslator(zigTranslatorBytes); emu = new Emulator(1024); }
        const raw = atob(b64);
        const inp = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) inp[i] = raw.charCodeAt(i);
        const n = inp.length / REC;
        const out = new Uint8Array(n * (STATE + 8));
        const iv = new DataView(inp.buffer), ov = new DataView(out.buffer);
        const mem = new DataView(emu.memory.buffer), bytes = new Uint8Array(emu.memory.buffer);
        for (let r = 0; r < n; r++) {
          const o = r * REC, so = o + 8;
          const insn = iv.getUint32(o, true), baseReg = iv.getInt32(o + 4, true);
          mem.setUint32(PC + GUEST_MEM_BASE, insn, true);
          mem.setUint32(PC + 4 + GUEST_MEM_BASE, 0xd65f03c0, true); // ret
          for (let i = 0; i < 32; i++) {
            mem.setBigUint64(i * 8, 0n, true);
            mem.setBigUint64(V_OFFSET + i * 16, 0n, true);
            mem.setBigUint64(V_OFFSET + i * 16 + 8, 0n, true);
          }
          for (let i = 0; i < NREG; i++) mem.setBigUint64(i * 8, iv.getBigUint64(so + i * 8, true), true);
          for (let i = 0; i < NVEC; i++) bytes.set(inp.subarray(so + NREG * 8 + 16 + i * 16, so + NREG * 8 + 16 + (i + 1) * 16), V_OFFSET + i * 16);
          mem.setUint32(NZCV_OFFSET, Number(iv.getBigUint64(so + NREG * 8, true) & 0xffffffffn), true);
          mem.setBigUint64(0x100, BigInt(PC), true);
          mem.setBigUint64(0xf8, 0x700000n, true);
          mem.setBigUint64(30 * 8, BigInt(PC + 0x800), true);
          bytes.set(inp.subarray(so + NREG * 8 + 16 + NVEC * 16, so + STATE), GUEST_SCRATCH + GUEST_MEM_BASE);
          if (baseReg >= 0) mem.setBigUint64(baseReg * 8, BigInt(GUEST_SCRATCH + SCRATCH / 2), true);

          (emu as any).regionCache.clear();
          let exitPc = 0n;
          try { exitPc = (emu as any).getRegion(BigInt(PC)).run(); } catch { exitPc = BigInt(PC); }

          const qo = r * (STATE + 8);
          // a region that refuses an instruction exits at that instruction's own pc, which is how
          // "not implemented" is told apart from "computed the wrong answer"
          ov.setUint32(qo, exitPc === BigInt(PC) ? 1 : 0, true);
          const q = qo + 8;
          for (let i = 0; i < NREG; i++) {
            let val = mem.getBigUint64(i * 8, true);
            if (i === baseReg) val = BigInt.asUintN(64, val - BigInt(GUEST_SCRATCH + SCRATCH / 2));
            ov.setBigUint64(q + i * 8, val, true);
          }
          ov.setBigUint64(q + NREG * 8, BigInt(mem.getUint32(NZCV_OFFSET, true)), true);
          for (let i = 0; i < NVEC; i++) out.set(bytes.subarray(V_OFFSET + i * 16, V_OFFSET + (i + 1) * 16), q + NREG * 8 + 16 + i * 16);
          out.set(bytes.subarray(GUEST_SCRATCH + GUEST_MEM_BASE, GUEST_SCRATCH + GUEST_MEM_BASE + SCRATCH), q + NREG * 8 + 16 + NVEC * 16);
        }
        return uint8ToBase64(out);
      };
    })();

    // memory swap driver: freeze a process (its Worker and wasm memory are destroyed) and bring
    // it back. exposed for now so the pipeline can be exercised on its own, before any automatic
    // pressure policy decides who to freeze.
    (window as any).__jitterSwap = (pid: number) => kernel.swapOut(pid);
    (window as any).__jitterThaw = (pid: number) => kernel.thaw(pid);
    (window as any).__jitterSwapStat = () => bridge.call('swap_stat');
    (window as any).__jitterSwapPolicy = (on: boolean) => kernel.setSwapPolicy(on);
    (window as any).__jitterSwapForceDisk = (on: boolean) => bridge.call('swap_force_disk', { on });
    (window as any).__jitterSwapped = () => kernel.swappedProcesses();
    (window as any).__jitterMemStats = () => kernel.memStats();
    (window as any).__jitterCensus = () => kernel.census();
    (window as any).__jitterTrace = (on?: boolean) => {
      if (on !== undefined) kernel.setTrace(on);
      return kernel.getTrace();
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
    kernel.boot(['/bin/sh'], GUEST_ENV);

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
