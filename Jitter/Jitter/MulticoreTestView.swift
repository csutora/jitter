import SwiftUI
import WebKit

/// Temporary PoC view that serves content via local HTTP with COOP/COEP headers
/// to test SharedArrayBuffer / Web Workers / Atomics support in WKWebView.
struct MulticoreTestView: UIViewRepresentable {
    let server: LocalServer

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.preferences.javaScriptCanOpenWindowsAutomatically = true

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.isOpaque = false
        webView.backgroundColor = UIColor(red: 10/255, green: 10/255, blue: 10/255, alpha: 1)

        // Load from the local HTTP server (not file://)
        let testURL = server.baseURL.appendingPathComponent("multicore-test.html")
        webView.load(URLRequest(url: testURL))

        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {}
}
