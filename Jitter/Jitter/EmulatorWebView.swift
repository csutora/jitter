import SwiftUI
import WebKit
import ObjectiveC

/// UIViewRepresentable wrapping a WKWebView that loads the jitter emulator.
struct EmulatorWebView: UIViewRepresentable {
    let messageRouter: MessageRouter

    func makeUIView(context: Context) -> JitterWebView {
        let config = WKWebViewConfiguration()
        config.preferences.javaScriptCanOpenWindowsAutomatically = false

        // Allow file:// access for fetch('busybox-aarch64')
        config.preferences.setValue(true, forKey: "allowFileAccessFromFileURLs")

        // Register JS→Swift bridge
        let contentController = WKUserContentController()
        contentController.add(messageRouter, name: "jitter")
        config.userContentController = contentController

        let webView = JitterWebView(frame: .zero, configuration: config)
        webView.isOpaque = false
        webView.backgroundColor = UIColor(red: 10/255, green: 10/255, blue: 10/255, alpha: 1)
        webView.scrollView.backgroundColor = webView.backgroundColor

        // Disable bouncing/scrolling — the terminal handles its own scrolling
        webView.scrollView.isScrollEnabled = false
        webView.scrollView.bounces = false

        // Copy bundle assets to tmp so WKWebView can fetch() them without sandbox issues
        let fm = FileManager.default
        let webDir = fm.temporaryDirectory.appendingPathComponent("jitter-web", isDirectory: true)
        try? fm.createDirectory(at: webDir, withIntermediateDirectories: true)

        let filesToCopy = ["index.html", "busybox-aarch64"]
        for name in filesToCopy {
            let dest = webDir.appendingPathComponent(name)
            // Always overwrite to pick up new builds
            try? fm.removeItem(at: dest)
            if let src = Bundle.main.url(forResource: name == "index.html" ? "index" : "busybox-aarch64",
                                         withExtension: name == "index.html" ? "html" : nil) {
                try? fm.copyItem(at: src, to: dest)
            }
        }

        let htmlURL = webDir.appendingPathComponent("index.html")
        webView.loadFileURL(htmlURL, allowingReadAccessTo: webDir)

        // Keep a reference so MessageRouter can call evaluateJavaScript
        messageRouter.webView = webView

        return webView
    }

    func updateUIView(_ webView: JitterWebView, context: Context) {}
}

/// WKWebView subclass that suppresses iOS keyboard chrome (accessory bar, assistant pill, etc.)
///
/// WKWebView delegates text input to an internal WKContentView. We use the ObjC runtime to
/// create a subclass of WKContentView that overrides inputAccessoryView (the bar above the
/// keyboard) and various text input traits. We also strip the inputAssistantItem bar button
/// groups (the floating language/dictate/arrow pill on iPadOS).
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

        // --- 1. Swizzle the class to override inputAccessoryView ---
        let className = "JitterWKContentView_NoAccessory"
        var cls: AnyClass? = NSClassFromString(className)

        if cls == nil,
           let parentClass = object_getClass(contentView),
           let cName = className.cString(using: .ascii) {
            cls = objc_allocateClassPair(parentClass, cName, 0)

            // inputAccessoryView → nil
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

        // --- 2. Strip the keyboard assistant item (floating pill on iPadOS) ---
        if let responder = contentView as? UIResponder {
            let item = responder.inputAssistantItem
            item.leadingBarButtonGroups = []
            item.trailingBarButtonGroups = []
        }

        contentViewPatched = true
    }
}

/// ObjC helper: returns nil for inputAccessoryView.
private class NilReturner: NSObject {
    @objc var inputAccessoryView: AnyObject? { nil }
}
