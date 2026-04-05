import Foundation
import WebKit

/// Routes messages from JS (via WKScriptMessageHandler) to native handlers,
/// and sends responses back via evaluateJavaScript.
///
/// JS sends: { id: number, method: string, args: { ... } }
/// Swift responds: window.__jitterResponse(id, { ... })
class MessageRouter: NSObject, WKScriptMessageHandler {
    weak var webView: WKWebView?
    private var handlers: [String: (Int, [String: Any]) -> Void] = [:]
    private(set) var nativeFS: NativeFS?

    override init() {
        super.init()
        setupHandlers()
    }

    private func setupHandlers() {
        let fs = NativeFS()
        self.nativeFS = fs

        handlers["ping"] = { [weak self] id, _ in
            self?.respond(id: id, result: ["pong": true])
        }

        handlers["load_bundle_file"] = { [weak self] id, args in
            guard let name = args["name"] as? String else {
                self?.respond(id: id, result: ["error": "missing name"])
                return
            }
            if let url = Bundle.main.url(forResource: name, withExtension: nil),
               let data = try? Data(contentsOf: url) {
                let b64 = data.base64EncodedString()
                self?.respond(id: id, result: ["data": b64])
            } else {
                self?.respond(id: id, result: ["error": "not found: \(name)"])
            }
        }

        // ── Filesystem bridge ──

        handlers["fs_open"] = { [weak self] id, args in
            guard let path = args["path"] as? String,
                  let flags = args["flags"] as? Int,
                  let mode = args["mode"] as? Int else {
                self?.respond(id: id, result: ["error": "invalid args", "ret": -22])
                return
            }
            let ret = fs.open(path: path, flags: Int32(flags), mode: Int32(mode))
            self?.respond(id: id, result: ["ret": ret])
        }

        handlers["fs_close"] = { [weak self] id, args in
            guard let fd = args["fd"] as? Int else {
                self?.respond(id: id, result: ["error": "invalid args", "ret": -9])
                return
            }
            let ret = fs.close(fd: Int32(fd))
            self?.respond(id: id, result: ["ret": ret])
        }

        handlers["fs_read"] = { [weak self] id, args in
            guard let fd = args["fd"] as? Int,
                  let count = args["count"] as? Int else {
                self?.respond(id: id, result: ["error": "invalid args", "ret": -9])
                return
            }
            let (ret, data) = fs.read(fd: Int32(fd), count: count)
            if let data = data {
                let b64 = data.base64EncodedString()
                self?.respond(id: id, result: ["ret": ret, "data": b64])
            } else {
                self?.respond(id: id, result: ["ret": ret])
            }
        }

        handlers["fs_write"] = { [weak self] id, args in
            guard let fd = args["fd"] as? Int,
                  let b64 = args["data"] as? String,
                  let data = Data(base64Encoded: b64) else {
                self?.respond(id: id, result: ["error": "invalid args", "ret": -22])
                return
            }
            let ret = fs.write(fd: Int32(fd), data: data)
            self?.respond(id: id, result: ["ret": ret])
        }

        handlers["fs_stat"] = { [weak self] id, args in
            guard let path = args["path"] as? String else {
                self?.respond(id: id, result: ["error": "invalid args", "ret": -22])
                return
            }
            let result = fs.stat(path: path)
            self?.respond(id: id, result: result)
        }

        handlers["fs_lseek"] = { [weak self] id, args in
            guard let fd = args["fd"] as? Int,
                  let offset = args["offset"] as? Int,
                  let whence = args["whence"] as? Int else {
                self?.respond(id: id, result: ["error": "invalid args", "ret": -22])
                return
            }
            let ret = fs.lseek(fd: Int32(fd), offset: Int64(offset), whence: Int32(whence))
            self?.respond(id: id, result: ["ret": ret])
        }

        handlers["fs_getdents"] = { [weak self] id, args in
            guard let fd = args["fd"] as? Int else {
                self?.respond(id: id, result: ["error": "invalid args", "ret": -9])
                return
            }
            let (ret, entries) = fs.getdents(fd: Int32(fd))
            self?.respond(id: id, result: ["ret": ret, "entries": entries])
        }

        handlers["fs_mkdir"] = { [weak self] id, args in
            guard let path = args["path"] as? String,
                  let mode = args["mode"] as? Int else {
                self?.respond(id: id, result: ["error": "invalid args", "ret": -22])
                return
            }
            let ret = fs.mkdir(path: path, mode: Int32(mode))
            self?.respond(id: id, result: ["ret": ret])
        }

        handlers["fs_unlink"] = { [weak self] id, args in
            guard let path = args["path"] as? String else {
                self?.respond(id: id, result: ["error": "invalid args", "ret": -22])
                return
            }
            let ret = fs.unlink(path: path)
            self?.respond(id: id, result: ["ret": ret])
        }

        handlers["fs_rename"] = { [weak self] id, args in
            guard let oldPath = args["oldPath"] as? String,
                  let newPath = args["newPath"] as? String else {
                self?.respond(id: id, result: ["error": "invalid args", "ret": -22])
                return
            }
            let ret = fs.rename(oldPath: oldPath, newPath: newPath)
            self?.respond(id: id, result: ["ret": ret])
        }
    }

    // MARK: - WKScriptMessageHandler

    func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage
    ) {
        guard let body = message.body as? [String: Any],
              let id = body["id"] as? Int,
              let method = body["method"] as? String else {
            return
        }

        let args = body["args"] as? [String: Any] ?? [:]

        if let handler = handlers[method] {
            handler(id, args)
        } else {
            respond(id: id, result: ["error": "unknown method: \(method)"])
        }
    }

    // MARK: - Response

    func respond(id: Int, result: [String: Any]) {
        guard let jsonData = try? JSONSerialization.data(withJSONObject: result),
              let jsonString = String(data: jsonData, encoding: .utf8) else {
            return
        }

        let js = "window.__jitterResponse(\(id), \(jsonString))"
        DispatchQueue.main.async { [weak self] in
            self?.webView?.evaluateJavaScript(js)
        }
    }
}
