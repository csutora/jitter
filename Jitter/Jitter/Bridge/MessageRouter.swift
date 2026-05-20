import Foundation
import WebKit

/// Routes messages from JS to native handlers using WKScriptMessageHandlerWithReply (iOS 14+).
///
/// JS calls: `const result = await window.webkit.messageHandlers.jitter.postMessage({method, args})`
/// The Promise is fulfilled directly by Swift via replyHandler — no evaluateJavaScript round-trip.
class MessageRouter: NSObject, WKScriptMessageHandlerWithReply {
    weak var webView: WKWebView?
    private var handlers: [String: ([String: Any], @escaping (Any?, String?) -> Void) -> Void] = [:]
    private(set) var nativeFS: NativeFS?
    var onTerminalData: ((Data) -> Void)?
    private var lastTermCols: Int = 0
    private var lastTermRows: Int = 0

    override init() {
        super.init()
        setupHandlers()
    }

    private func setupHandlers() {
        let fs = NativeFS()
        self.nativeFS = fs
        let net = NetworkBridge()

        // ── Network bridge ──

        handlers["net_socket"] = { args, reply in
            guard let domain = args["domain"] as? Int,
                  let type = args["type"] as? Int,
                  let proto = args["protocol"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(net.socket(domain: domain, type: type, protocol: proto), nil)
        }

        handlers["net_connect"] = { args, reply in
            guard let socketId = args["socketId"] as? Int,
                  let family = args["family"] as? Int,
                  let host = args["host"] as? String,
                  let port = args["port"] as? Int else {
                reply(["ret": -22], nil); return
            }
            net.connect(socketId: socketId, family: family, host: host, port: port) { result in
                reply(result, nil)
            }
        }

        handlers["net_bind"] = { args, reply in
            guard let socketId = args["socketId"] as? Int,
                  let family = args["family"] as? Int,
                  let host = args["host"] as? String,
                  let port = args["port"] as? Int else {
                reply(["ret": -22], nil); return
            }
            net.bind(socketId: socketId, family: family, host: host, port: port) { result in
                reply(result, nil)
            }
        }

        handlers["net_listen"] = { args, reply in
            guard let socketId = args["socketId"] as? Int,
                  let backlog = args["backlog"] as? Int else {
                reply(["ret": -22], nil); return
            }
            net.listen(socketId: socketId, backlog: backlog) { result in
                reply(result, nil)
            }
        }

        handlers["net_accept"] = { args, reply in
            guard let socketId = args["socketId"] as? Int else {
                reply(["ret": -9], nil); return
            }
            net.accept(socketId: socketId) { result in
                reply(result, nil)
            }
        }

        handlers["net_read"] = { args, reply in
            guard let socketId = args["socketId"] as? Int,
                  let count = args["count"] as? Int else {
                reply(["ret": -9], nil); return
            }
            net.read(socketId: socketId, count: count) { result in
                reply(result, nil)
            }
        }

        handlers["net_write"] = { args, reply in
            guard let socketId = args["socketId"] as? Int,
                  let data = args["data"] as? String else {
                reply(["ret": -22], nil); return
            }
            net.write(socketId: socketId, data: data) { result in
                reply(result, nil)
            }
        }

        handlers["net_close"] = { args, reply in
            guard let socketId = args["socketId"] as? Int else {
                reply(["ret": -9], nil); return
            }
            reply(net.close(socketId: socketId), nil)
        }

        handlers["net_poll"] = { args, reply in
            guard let socketId = args["socketId"] as? Int,
                  let timeoutMs = args["timeoutMs"] as? Int else {
                reply(["ret": 0], nil); return
            }
            net.poll(socketId: socketId, timeoutMs: timeoutMs) { result in
                reply(result, nil)
            }
        }

        handlers["net_sendto"] = { args, reply in
            guard let socketId = args["socketId"] as? Int,
                  let data = args["data"] as? String,
                  let host = args["host"] as? String,
                  let port = args["port"] as? Int else {
                reply(["ret": -22], nil); return
            }
            net.sendto(socketId: socketId, data: data, host: host, port: port) { result in
                reply(result, nil)
            }
        }

        handlers["net_sockopt"] = { _, reply in
            reply(["ret": 0], nil)
        }

        handlers["ping"] = { _, reply in
            reply(["pong": true], nil)
        }

        handlers["load_bundle_file"] = { args, reply in
            guard let name = args["name"] as? String else {
                reply(["error": "missing name"], nil); return
            }
            if let url = Bundle.main.url(forResource: name, withExtension: nil),
               let data = try? Data(contentsOf: url) {
                reply(["data": data.base64EncodedString()], nil)
            } else {
                reply(["error": "not found: \(name)"], nil)
            }
        }

        // ── Filesystem bridge ──

        handlers["fs_open"] = { args, reply in
            guard let path = args["path"] as? String,
                  let flags = args["flags"] as? Int,
                  let mode = args["mode"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.open(path: path, flags: Int32(flags), mode: Int32(mode))], nil)
        }

        handlers["fs_close"] = { args, reply in
            guard let fd = args["fd"] as? Int else {
                reply(["ret": -9], nil); return
            }
            reply(["ret": fs.close(fd: Int32(fd))], nil)
        }

        handlers["fs_read"] = { args, reply in
            guard let fd = args["fd"] as? Int,
                  let count = args["count"] as? Int else {
                reply(["ret": -9], nil); return
            }
            let (ret, data) = fs.read(fd: Int32(fd), count: count)
            if let data = data {
                reply(["ret": ret, "data": data.base64EncodedString()], nil)
            } else {
                reply(["ret": ret], nil)
            }
        }

        handlers["fs_write"] = { args, reply in
            guard let fd = args["fd"] as? Int,
                  let b64 = args["data"] as? String,
                  let data = Data(base64Encoded: b64) else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.write(fd: Int32(fd), data: data)], nil)
        }

        handlers["fs_stat"] = { args, reply in
            guard let path = args["path"] as? String else {
                reply(["ret": -22], nil); return
            }
            reply(fs.stat(path: path), nil)
        }

        handlers["fs_lseek"] = { args, reply in
            guard let fd = args["fd"] as? Int,
                  let offset = args["offset"] as? Int,
                  let whence = args["whence"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.lseek(fd: Int32(fd), offset: Int64(offset), whence: Int32(whence))], nil)
        }

        handlers["fs_getdents"] = { args, reply in
            guard let fd = args["fd"] as? Int else {
                reply(["ret": -9], nil); return
            }
            let (ret, entries) = fs.getdents(fd: Int32(fd))
            reply(["ret": ret, "entries": entries], nil)
        }

        handlers["fs_mkdir"] = { args, reply in
            guard let path = args["path"] as? String,
                  let mode = args["mode"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.mkdir(path: path, mode: Int32(mode))], nil)
        }

        handlers["fs_unlink"] = { args, reply in
            guard let path = args["path"] as? String else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.unlink(path: path)], nil)
        }

        handlers["fs_rename"] = { args, reply in
            guard let oldPath = args["oldPath"] as? String,
                  let newPath = args["newPath"] as? String else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.rename(oldPath: oldPath, newPath: newPath)], nil)
        }

        handlers["fs_symlink"] = { args, reply in
            guard let target = args["target"] as? String,
                  let linkpath = args["linkpath"] as? String else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.symlink(target: target, linkpath: linkpath)], nil)
        }

        handlers["fs_readlink"] = { args, reply in
            guard let path = args["path"] as? String else {
                reply(["ret": -22], nil); return
            }
            let (ret, target) = fs.readlink(path: path)
            if let target = target {
                reply(["ret": ret, "target": target], nil)
            } else {
                reply(["ret": ret], nil)
            }
        }

        handlers["fs_fstat"] = { args, reply in
            guard let fd = args["fd"] as? Int else {
                reply(["ret": -9], nil); return
            }
            reply(fs.fstat(fd: Int32(fd)), nil)
        }

        handlers["fs_ftruncate"] = { args, reply in
            guard let fd = args["fd"] as? Int,
                  let length = args["length"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.ftruncate(fd: Int32(fd), length: Int64(length))], nil)
        }

        handlers["fs_dup"] = { args, reply in
            guard let fd = args["fd"] as? Int else {
                reply(["ret": -9], nil); return
            }
            reply(["ret": fs.dup(fd: Int32(fd))], nil)
        }

        handlers["fs_faccessat"] = { args, reply in
            guard let path = args["path"] as? String,
                  let mode = args["mode"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.faccessat(path: path, mode: Int32(mode))], nil)
        }

        handlers["fs_chmod"] = { args, reply in
            guard let path = args["path"] as? String,
                  let mode = args["mode"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.chmod(path: path, mode: Int32(mode))], nil)
        }

        handlers["term_size_query"] = { [weak self] _, reply in
            reply(["cols": self?.lastTermCols ?? 0, "rows": self?.lastTermRows ?? 0], nil)
        }

        handlers["term_data"] = { [weak self] args, reply in
            if let b64 = args["data"] as? String,
               let data = Data(base64Encoded: b64),
               let cb = self?.onTerminalData {
                DispatchQueue.main.async { cb(data) }
            }
            reply([:], nil)
        }
    }

    func sendStdin(_ data: Data) {
        guard !data.isEmpty else { return }
        let b64 = data.base64EncodedString()
        let js = "window.__jitterStdin && window.__jitterStdin('\(b64)')"
        DispatchQueue.main.async { [weak self] in
            self?.webView?.evaluateJavaScript(js, completionHandler: nil)
        }
    }

    func sendTermSize(cols: Int, rows: Int) {
        lastTermCols = cols
        lastTermRows = rows
        let js = "window.__jitterTermSize && window.__jitterTermSize(\(cols), \(rows))"
        DispatchQueue.main.async { [weak self] in
            self?.webView?.evaluateJavaScript(js, completionHandler: nil)
        }
    }

    // MARK: - WKScriptMessageHandlerWithReply

    func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage,
        replyHandler: @escaping (Any?, String?) -> Void
    ) {
        guard let body = message.body as? [String: Any],
              let method = body["method"] as? String else {
            replyHandler(nil, "invalid message")
            return
        }

        let args = body["args"] as? [String: Any] ?? [:]

        if let handler = handlers[method] {
            handler(args, replyHandler)
        } else {
            replyHandler(nil, "unknown method: \(method)")
        }
    }
}
