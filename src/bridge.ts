/**
 * Native bridge for JS↔Swift communication via WKWebView.
 *
 * Uses WKScriptMessageHandlerWithReply (iOS 14+): JS postMessage returns a Promise
 * that Swift fulfills directly via replyHandler. No evaluateJavaScript round-trip.
 */

interface BridgeResponse {
  error?: string;
  [key: string]: any;
}

export class NativeBridge {
  readonly isNative: boolean;

  constructor() {
    this.isNative = !!(window as any).webkit?.messageHandlers?.jitter;
  }

  /** Send a message to Swift and await the response. */
  async call(method: string, args: Record<string, any> = {}): Promise<BridgeResponse> {
    if (!this.isNative) throw new Error('NativeBridge not available');
    // postMessage now returns a Promise directly (WKScriptMessageHandlerWithReply)
    const result = await (window as any).webkit.messageHandlers.jitter.postMessage({ method, args });
    return (result ?? {}) as BridgeResponse;
  }
}
