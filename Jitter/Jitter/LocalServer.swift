import Foundation
import Network

/// Minimal HTTP server using Network.framework.
/// Serves files from a directory with COOP/COEP headers to enable SharedArrayBuffer.
final class LocalServer {
    private var listener: NWListener?
    private let rootDir: URL
    private let port: UInt16

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
        // Read the HTTP request (up to 8KB is plenty for GET requests)
        conn.receive(minimumIncompleteLength: 1, maximumLength: 8192) { [weak self] data, _, _, error in
            guard let self = self, let data = data, error == nil else {
                conn.cancel()
                return
            }
            let request = String(data: data, encoding: .utf8) ?? ""
            self.respond(to: request, on: conn)
        }
    }

    private func respond(to request: String, on conn: NWConnection) {
        // Parse GET path from first line: "GET /path HTTP/1.1"
        let firstLine = request.split(separator: "\r\n").first ?? ""
        let parts = firstLine.split(separator: " ")
        let rawPath = parts.count >= 2 ? String(parts[1]) : "/"
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
