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
            if path.hasPrefix("/ctl/js2") {
                var js = String(data: body, encoding: .utf8) ?? ""
                var ms = 30000
                if js.hasPrefix("@"), let sp = js.firstIndex(of: " ") {
                    ms = Int(js[js.index(after: js.startIndex)..<sp]) ?? 30000
                    js = String(js[js.index(after: sp)...])
                }
                return ("text/plain", Data(secondWebViewEval(js, timeoutMs: ms).utf8))
            } else if path.hasPrefix("/ctl/procs") {
                return ("text/plain", Data("web content processes: \(webProcessCount())\n".utf8))
            } else if path.hasPrefix("/ctl/wk") {
                // body: "" for the selector probe, or a substring to filter the feature list
                let f = (String(data: body, encoding: .utf8) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                if f.isEmpty { return ("text/plain", Data(WebKitKnobs.probeSelectors().utf8)) }
                return ("text/plain", Data(WebKitKnobs.features(matching: f == "all" ? "" : f).utf8))
            } else if path.hasPrefix("/ctl/mem") {
                // body: "" for a report, or "commit <stepMiB> <guardMiB> <capMiB>" /
                // "reserve <stepMiB> <capMiB>" to walk this process up to its ceiling
                let a = (String(data: body, encoding: .utf8) ?? "").split(separator: " ").map(String.init)
                if a.first == "commit" {
                    return ("text/plain", Data(MemoryProbe.commitProbe(
                        stepMiB: Int(a.count > 1 ? a[1] : "") ?? 64,
                        guardMiB: Int(a.count > 2 ? a[2] : "") ?? 512,
                        capMiB: Int(a.count > 3 ? a[3] : "") ?? 32768).utf8))
                }
                if a.first == "reserve" {
                    return ("text/plain", Data(MemoryProbe.reserveProbe(
                        stepMiB: Int(a.count > 1 ? a[1] : "") ?? 256,
                        capMiB: Int(a.count > 2 ? a[2] : "") ?? 262144).utf8))
                }
                return ("text/plain", Data(MemoryProbe.report().utf8))
            } else if path.hasPrefix("/ctl/input") {
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
            } else if path.hasPrefix("/ctl/js") {
                // body: "[@<ms> ]<javascript>" — evaluate in the page (async fn body: use `return`).
                var s = String(data: body, encoding: .utf8) ?? ""
                var ms = 30000
                if s.hasPrefix("@"), let sp = s.firstIndex(of: " ") {
                    ms = Int(s[s.index(after: s.startIndex)..<sp]) ?? 30000
                    s = String(s[s.index(after: sp)...])
                }
                return ("text/plain", Data(router.evalJS(s, timeoutMs: ms).utf8))
            } else if path.hasPrefix("/ctl/run") {
                // body: "[@<ms> ]<binary> [args...]" — run one binary as its own process, bypassing the
                // shell/tty, and return its output + exit code. works on a root with no working shell.
                var s = (String(data: body, encoding: .utf8) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                var ms = 30000
                if s.hasPrefix("@"), let sp = s.firstIndex(of: " ") {
                    ms = Int(s[s.index(after: s.startIndex)..<sp]) ?? 30000
                    s = String(s[s.index(after: sp)...])
                }
                let parts = splitArgv(s)
                guard let bin = parts.first else { return ("text/plain", Data("usage: [@ms] <binary> [args]".utf8)) }
                return ("text/plain", Data(router.runBinary(bin, args: Array(parts.dropFirst()), timeoutMs: ms).utf8))
            } else if path.hasPrefix("/ctl/switch") {
                // body: "<name>" — repoint the active root + reboot the session into it
                let name = (String(data: body, encoding: .utf8) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                let ok = router.nativeFS?.switchRoot(name) ?? false
                if ok { router.ctlReset() }
                return ("text/plain", Data((ok ? "switched to \(name)" : "no such root: \(name)").utf8))
            } else if path.hasPrefix("/ctl/put/") {
                // POST /ctl/put/<guest path>, body = the file's bytes. writes it into the active root
                // and marks it executable, which is how a locally built probe or test binary gets onto
                // the device without a rebuild.
                guard let fs = router.nativeFS else { return ("text/plain", Data("no fs".utf8)) }
                let guestPath = String(path.dropFirst("/ctl/put".count)).removingPercentEncoding ?? ""
                guard guestPath.count > 1, !guestPath.contains("..") else {
                    return ("text/plain", Data("usage: POST /ctl/put/<path>".utf8))
                }
                let dst = fs.rootPath.appendingPathComponent(String(guestPath.dropFirst()))
                do {
                    try FileManager.default.createDirectory(at: dst.deletingLastPathComponent(),
                                                            withIntermediateDirectories: true)
                    try body.write(to: dst)
                    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dst.path)
                    return ("text/plain", Data("wrote \(body.count) bytes to \(guestPath)".utf8))
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

// split a control-route command line into argv the way a shell would, honouring single quotes,
// double quotes and backslash escapes. a naive split on spaces passed the quote characters through
// as literal argv words, so `sh -c "ls -d /opt"` reached the guest as an unterminated string.
func splitArgv(_ line: String) -> [String] {
    var out: [String] = []
    var cur = ""
    var started = false
    var quote: Character? = nil
    var escaped = false

    for ch in line {
        if escaped { cur.append(ch); escaped = false; continue }
        if let q = quote {
            if ch == "\\" && q == "\"" { escaped = true; continue }
            if ch == q { quote = nil; continue }
            cur.append(ch)
            continue
        }
        switch ch {
        case "\\": escaped = true; started = true
        case "'", "\"": quote = ch; started = true
        case " ", "\t":
            if started { out.append(cur); cur = ""; started = false }
        default:
            cur.append(ch); started = true
        }
    }
    if started { out.append(cur) }
    return out
}
