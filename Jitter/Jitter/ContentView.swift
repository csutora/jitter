import SwiftUI

struct ContentView: View {
    @State private var messageRouter = MessageRouter()
    @State private var server: LocalServer?

    var body: some View {
        Group {
            if let server = server {
                EmulatorWebView(messageRouter: messageRouter, baseURL: server.baseURL)
            } else {
                ProgressView("Starting server...")
                    .foregroundStyle(.white)
            }
        }
        .ignoresSafeArea()
        .preferredColorScheme(.dark)
        .persistentSystemOverlays(.hidden)
        .onAppear {
            startServer()
        }
    }

    private func startServer() {
        let fm = FileManager.default
        let webDir = fm.temporaryDirectory.appendingPathComponent("jitter-web", isDirectory: true)
        try? fm.createDirectory(at: webDir, withIntermediateDirectories: true)

        // Copy all resources to the serving directory
        let filesToCopy = [
            ("index", "html"),
            ("busybox-aarch64", nil),
            ("worker", "js"),
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
