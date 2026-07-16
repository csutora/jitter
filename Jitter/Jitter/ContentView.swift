import SwiftUI

struct ContentView: View {
    @State private var messageRouter = MessageRouter()
    @State private var server: LocalServer?
    @State private var ghosttyProxy = TerminalProxy()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            Group {
                if let server = server {
                    EmulatorWebView(messageRouter: messageRouter, baseURL: server.baseURL)
                } else {
                    Color.black
                }
            }
            .ignoresSafeArea()
            TerminalView(
                proxy: ghosttyProxy,
                onStdin: { [router = messageRouter] data in router.sendStdin(data) },
                onTermSize: { [router = messageRouter] cols, rows in router.sendTermSize(cols: cols, rows: rows) }
            )
            .ignoresSafeArea()
        }
        .preferredColorScheme(.dark)
        .persistentSystemOverlays(.hidden)
        .onAppear {
            messageRouter.onTerminalData = { [proxy = ghosttyProxy] data in
                proxy.feed(data)
            }
            startServer()
        }
    }

    private func startServer() {
        let fm = FileManager.default
        let webDir = fm.temporaryDirectory.appendingPathComponent("jitter-web", isDirectory: true)
        try? fm.createDirectory(at: webDir, withIntermediateDirectories: true)

        let filesToCopy = [
            ("index", "html"),
            ("busybox-aarch64", nil),
            ("worker", "js"),
            ("translator", "wasm"),
            ("multicore-test", "html"),
        ]
        for (base, ext) in filesToCopy {
            let name = ext != nil ? "\(base).\(ext!)" : base
            let dest = webDir.appendingPathComponent(name)
            try? fm.removeItem(at: dest)
            if let src = Bundle.main.url(forResource: base, withExtension: ext) {
                try? fm.copyItem(at: src, to: dest)
            }
        }

        let s = LocalServer(rootDir: webDir, port: 9080)
        // debug control channel (always on for now; make it a settings toggle later).
        // lets an external driver inject stdin / read output / reset the guest over http:
        //   POST /ctl/input  (body=text)  -> type into the terminal
        //   GET  /ctl/output              -> current terminal scrollback
        //   POST /ctl/reset               -> reload the webview (fresh guest, disk persists)
        s.controlHandler = { [router = messageRouter] method, path, body in
            if path.hasPrefix("/ctl/input") {
                router.ctlInject(String(data: body, encoding: .utf8) ?? "")
                return ("text/plain", Data("ok".utf8))
            } else if path.hasPrefix("/ctl/output") {
                return ("application/octet-stream", router.ctlReadOutput())
            } else if path.hasPrefix("/ctl/reset") {
                router.ctlReset()
                return ("text/plain", Data("reset".utf8))
            } else if path.hasPrefix("/ctl/pull") {
                // body: "<name> <url>" — download + extract a distro tarball into roots/<name>
                let s = String(data: body, encoding: .utf8) ?? ""
                let parts = s.trimmingCharacters(in: .whitespacesAndNewlines).split(separator: " ", maxSplits: 1).map(String.init)
                guard parts.count == 2 else { return ("text/plain", Data("usage: <name> <url>".utf8)) }
                return ("text/plain", Data(router.performPull(parts[0], parts[1]).utf8))
            } else if path.hasPrefix("/ctl/exec") {
                // body: a shell command, optionally prefixed with "@<ms> " to override the 30s timeout.
                // blocks until it completes (or times out, sending SIGINT); returns output + [exit N].
                var cmd = String(data: body, encoding: .utf8) ?? ""
                var ms = 30000
                if cmd.hasPrefix("@"), let sp = cmd.firstIndex(of: " ") {
                    ms = Int(cmd[cmd.index(after: cmd.startIndex)..<sp]) ?? 30000
                    cmd = String(cmd[cmd.index(after: sp)...])
                }
                return ("text/plain", Data(router.execCommand(cmd, timeoutMs: ms).utf8))
            } else if path.hasPrefix("/ctl/switch") {
                // body: "<name>" — repoint the active root + reboot the session into it
                let name = (String(data: body, encoding: .utf8) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                let ok = router.nativeFS?.switchRoot(name) ?? false
                if ok { router.ctlReset() }
                return ("text/plain", Data((ok ? "switched to \(name)" : "no such root: \(name)").utf8))
            } else if path.hasPrefix("/ctl/instkat") {
                // temporary crypto-diagnosis helper: drop a bundled diagnostic binary into the active root.
                // body = resource name (default crypto_kat).
                let resName = (String(data: body, encoding: .utf8) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                let name = resName.isEmpty ? "crypto_kat" : resName
                guard let src = Bundle.main.url(forResource: name, withExtension: nil),
                      let data = try? Data(contentsOf: src),
                      let fs = router.nativeFS else { return ("text/plain", Data("no \(name) or fs".utf8)) }
                let dstDir = fs.rootPath.appendingPathComponent("root", isDirectory: true)
                let dst = dstDir.appendingPathComponent(name)
                do {
                    try FileManager.default.createDirectory(at: dstDir, withIntermediateDirectories: true)
                    try data.write(to: dst)
                    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dst.path)
                    return ("text/plain", Data("installed \(data.count) bytes to /root/\(name)".utf8))
                } catch { return ("text/plain", Data("err: \(error)".utf8)) }
            }
            return ("text/plain", Data("unknown ctl path".utf8))
        }
        do {
            try s.start()
            server = s
        } catch {
            print("[Jitter] Failed to start local server: \(error)")
        }
    }
}

#Preview {
    ContentView()
}
