import SwiftUI
import WebKit
import ObjectiveC

/// UIViewRepresentable wrapping a WKWebView that loads the jitter emulator.
/// Content is served via a local HTTP server with COOP/COEP headers to enable
/// SharedArrayBuffer, Web Workers, and Atomics for multicore process execution.
struct EmulatorWebView: UIViewRepresentable {
    let messageRouter: MessageRouter
    let baseURL: URL

    func makeUIView(context: Context) -> JitterWebView {
        let config = WKWebViewConfiguration()
        config.preferences.javaScriptCanOpenWindowsAutomatically = false

        // Register JS→Swift bridge using WithReply API (iOS 14+).
        // JS postMessage returns a Promise that Swift fulfills directly — no evaluateJavaScript round-trip.
        let contentController = WKUserContentController()
        contentController.addScriptMessageHandler(messageRouter, contentWorld: .page, name: "jitter")
        config.userContentController = contentController

        let webView = JitterWebView(frame: .zero, configuration: config)
        webView.isInspectable = true  // Enable Safari Web Inspector
        webView.isOpaque = true
        webView.backgroundColor = .black
        webView.scrollView.backgroundColor = .black

        // Disable bouncing/scrolling — the terminal handles its own scrolling
        webView.scrollView.isScrollEnabled = false
        webView.scrollView.bounces = false

        // The webview is purely the compute+bridge engine and renders nothing the user touches
        // (the terminal is layered on top and owns all input, delivered programmatically via the
        // bridge). turn OFF its user interaction entirely — otherwise WKWebView keeps live scroll
        // gesture recognizers that swallow trackpad / mouse-wheel / continuity-cursor scroll events
        // before the terminal's pan gesture can see them (isScrollEnabled=false is not enough).
        webView.isUserInteractionEnabled = false
        webView.scrollView.isUserInteractionEnabled = false

        // Load from local HTTP server (not file:// — we need COOP/COEP headers for SharedArrayBuffer)
        let indexURL = baseURL.appendingPathComponent("index.html")
        webView.load(URLRequest(url: indexURL))

        // Keep a reference so MessageRouter can call evaluateJavaScript
        messageRouter.webView = webView

        return webView
    }

    func updateUIView(_ webView: JitterWebView, context: Context) {}
}

/// WKWebView subclass that suppresses iOS keyboard chrome.
class JitterWebView: WKWebView {
    override var inputAccessoryView: UIView? { nil }

    private var contentViewPatched = false

    override func didMoveToSuperview() {
        super.didMoveToSuperview()
        patchContentView()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        if !contentViewPatched {
            patchContentView()
        }
    }

    private func patchContentView() {
        guard let contentView = scrollView.subviews.first(where: {
            String(describing: type(of: $0)).hasPrefix("WKContent")
        }) else { return }

        let className = "JitterWKContentView_NoAccessory"
        var cls: AnyClass? = NSClassFromString(className)

        if cls == nil,
           let parentClass = object_getClass(contentView),
           let cName = className.cString(using: .ascii) {
            cls = objc_allocateClassPair(parentClass, cName, 0)

            let nilHelper = NilReturner.self
            if let m = class_getInstanceMethod(nilHelper, #selector(getter: NilReturner.inputAccessoryView)) {
                class_addMethod(cls, #selector(getter: UIResponder.inputAccessoryView),
                                method_getImplementation(m), method_getTypeEncoding(m))
            }

            objc_registerClassPair(cls!)
        }

        if let cls = cls {
            object_setClass(contentView, cls)
        }

        if let responder = contentView as? UIResponder {
            let item = responder.inputAssistantItem
            item.leadingBarButtonGroups = []
            item.trailingBarButtonGroups = []
        }

        contentViewPatched = true
    }
}

private class NilReturner: NSObject {
    @objc var inputAccessoryView: AnyObject? { nil }
}
