import Foundation
import Network

/// Minimal HTTP server using Network.framework.
/// Serves files from a directory with COOP/COEP headers to enable SharedArrayBuffer.
final class LocalServer {
    private var listener: NWListener?
    private let rootDir: URL
    private let port: UInt16

    /// Debug control channel: (method, path, body) -> (contentType, responseBody).
    /// Set by ContentView to bridge /ctl/* requests to the terminal (inject stdin,
    /// read output, reset the guest). nil = disabled (serves static files only).
    var controlHandler: ((_ method: String, _ path: String, _ body: Data) -> (String, Data))?

    /// The URL to load in WKWebView once the server is running.
    var baseURL: URL { URL(string: "http://localhost:\(port)")! }

    init(rootDir: URL, port: UInt16 = 9080) {
        self.rootDir = rootDir
        self.port = port
    }

    func start() throws {
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true

        listener = try NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!)
        listener?.newConnectionHandler = { [weak self] conn in
            self?.handleConnection(conn)
        }
        listener?.stateUpdateHandler = { state in
            if case .failed(let err) = state {
                print("[LocalServer] failed: \(err)")
            }
        }
        listener?.start(queue: .global(qos: .userInitiated))
        print("[LocalServer] serving \(rootDir.path) on http://localhost:\(port)")
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }

    // MARK: - Connection handling

    private func handleConnection(_ conn: NWConnection) {
        conn.start(queue: .global(qos: .userInitiated))
        readRequest(conn, accumulated: Data())
    }

    // accumulate bytes until we have the full headers + body (per Content-Length), then route.
    private func readRequest(_ conn: NWConnection, accumulated: Data) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            guard let self = self, error == nil else { conn.cancel(); return }
            var buf = accumulated
            if let data = data { buf.append(data) }
            // find header terminator
            guard let headerEnd = buf.range(of: Data("\r\n\r\n".utf8)) else {
                if isComplete { conn.cancel() } else { self.readRequest(conn, accumulated: buf) }
                return
            }
            let header = String(data: buf.subdata(in: buf.startIndex..<headerEnd.lowerBound), encoding: .utf8) ?? ""
            let contentLength = header.split(separator: "\r\n")
                .first(where: { $0.lowercased().hasPrefix("content-length:") })
                .flatMap { Int($0.split(separator: ":")[1].trimmingCharacters(in: .whitespaces)) } ?? 0
            let bodyStart = headerEnd.upperBound
            let have = buf.distance(from: bodyStart, to: buf.endIndex)
            if have < contentLength && !isComplete {
                self.readRequest(conn, accumulated: buf)
                return
            }
            let body = buf.subdata(in: bodyStart..<buf.endIndex)
            self.respond(header: header, body: body, on: conn)
        }
    }

    private func respond(header request: String, body: Data, on conn: NWConnection) {
        // Parse "METHOD /path HTTP/1.1"
        let firstLine = request.split(separator: "\r\n").first ?? ""
        let parts = firstLine.split(separator: " ")
        let method = parts.count >= 1 ? String(parts[0]) : "GET"
        let rawPath = parts.count >= 2 ? String(parts[1]) : "/"

        // debug control channel: /ctl/* routes to the injected handler
        if rawPath.hasPrefix("/ctl/"), let handler = controlHandler {
            let (contentType, respBody) = handler(method, rawPath, body)
            send(httpResponse(status: "200 OK", contentType: contentType, body: respBody), on: conn)
            return
        }

        let path = rawPath == "/" ? "/index.html" : rawPath

        // Decode percent-encoding and resolve to filesystem
        let decoded = path.removingPercentEncoding ?? path
        let fileURL = rootDir.appendingPathComponent(decoded)

        guard let fileData = try? Data(contentsOf: fileURL) else {
            let body = "404 Not Found: \(decoded)"
            let resp = httpResponse(status: "404 Not Found", contentType: "text/plain",
                                    body: body.data(using: .utf8)!)
            send(resp, on: conn)
            return
        }

        let contentType = mimeType(for: decoded)
        let resp = httpResponse(status: "200 OK", contentType: contentType, body: fileData)
        send(resp, on: conn)
    }

    private func httpResponse(status: String, contentType: String, body: Data) -> Data {
        var header = "HTTP/1.1 \(status)\r\n"
        header += "Content-Type: \(contentType)\r\n"
        header += "Content-Length: \(body.count)\r\n"
        // These two headers enable cross-origin isolation → SharedArrayBuffer
        header += "Cross-Origin-Opener-Policy: same-origin\r\n"
        header += "Cross-Origin-Embedder-Policy: require-corp\r\n"
        header += "Cache-Control: no-cache\r\n"
        header += "Connection: close\r\n"
        header += "\r\n"

        var data = header.data(using: .utf8)!
        data.append(body)
        return data
    }

    private func send(_ data: Data, on conn: NWConnection) {
        conn.send(content: data, completion: .contentProcessed { _ in
            conn.cancel()
        })
    }

    private func mimeType(for path: String) -> String {
        let ext = (path as NSString).pathExtension.lowercased()
        switch ext {
        case "html": return "text/html; charset=utf-8"
        case "js":   return "application/javascript; charset=utf-8"
        case "css":  return "text/css; charset=utf-8"
        case "wasm": return "application/wasm"
        case "json": return "application/json"
        case "png":  return "image/png"
        case "svg":  return "image/svg+xml"
        default:     return "application/octet-stream"
        }
    }
}
