/**
 * Native bridge for JS↔Swift communication via WKWebView.
 *
 * When running in a WKWebView with a registered "jitter" message handler,
 * this bridge enables async syscall routing to native Swift code.
 * When running in a regular browser, isNative is false and the bridge is inert.
 */

interface BridgeResponse {
  error?: string;
  [key: string]: any;
}

export class NativeBridge {
  private nextId = 0;
  private pending = new Map<number, { resolve: (r: BridgeResponse) => void }>();
  readonly isNative: boolean;

  constructor() {
    this.isNative = !!(window as any).webkit?.messageHandlers?.jitter;
    if (this.isNative) {
      // Register global callback for Swift → JS responses
      (window as any).__jitterResponse = (id: number, result: BridgeResponse) => {
        const p = this.pending.get(id);
        if (p) {
          this.pending.delete(id);
          p.resolve(result);
        }
      };
    }
  }

  /** Send a message to Swift and await the response */
  async call(method: string, args: Record<string, any> = {}): Promise<BridgeResponse> {
    if (!this.isNative) throw new Error('NativeBridge not available');
    const id = this.nextId++;
    return new Promise(resolve => {
      this.pending.set(id, { resolve });
      (window as any).webkit.messageHandlers.jitter.postMessage({ id, method, args });
    });
  }
}
