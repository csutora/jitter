import Foundation
import WebKit
import UIKit

/// a second, throwaway WKWebView used to answer one question: does each webview get its own
/// WebContent process with its own wasm memory budget, or is the 4 GiB cap shared across the app?
/// it is attached to the window (1x1, effectively invisible) rather than left detached, because a
/// content process that is not backing a visible view gets suppressed and then suspended.
@MainActor
final class SecondWebView: NSObject {
    static let shared = SecondWebView()
    private var web: WKWebView?

    func view() -> WKWebView {
        if let w = web { return w }
        let cfg = WKWebViewConfiguration()
        let w = WKWebView(frame: CGRect(x: 0, y: 0, width: 1, height: 1), configuration: cfg)
        if let win = UIApplication.shared.connectedScenes
            .compactMap({ ($0 as? UIWindowScene)?.keyWindow }).first {
            w.alpha = 0.01
            win.addSubview(w)
        }
        w.loadHTMLString("<html><body>probe</body></html>", baseURL: URL(string: "http://localhost/"))
        web = w
        return w
    }

}

/// callers arrive on the server's queue, so the main-thread work is dispatched async and waited on
/// from there. blocking the main thread instead would deadlock: callAsyncJavaScript delivers its
/// completion on that same thread.
func secondWebViewEval(_ js: String, timeoutMs: Int) -> String {
    let sem = DispatchSemaphore(value: 0)
    var out = "[ctl/js2] no response"
    DispatchQueue.main.async {
        let w = SecondWebView.shared.view()
        w.callAsyncJavaScript(js, arguments: [:], in: nil, in: .page) { r in
            switch r {
            case .success(let v): out = String(describing: v)
            case .failure(let e): out = "[ctl/js2] error: \(e.localizedDescription)"
            }
            sem.signal()
        }
    }
    if sem.wait(timeout: .now() + .milliseconds(timeoutMs)) == .timedOut { return "[ctl/js2] timed out\n" }
    return out + "\n"
}

func webProcessCount() -> String {
    let sem = DispatchSemaphore(value: 0)
    var out = "unavailable"
    DispatchQueue.main.async {
        let sel = NSSelectorFromString("_webProcessCount")
        let pool = WKProcessPool()
        if pool.responds(to: sel), let n = (pool as AnyObject).perform(sel) {
            out = String(describing: n.takeUnretainedValue())
        }
        sem.signal()
    }
    _ = sem.wait(timeout: .now() + .seconds(5))
    return out
}
