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
