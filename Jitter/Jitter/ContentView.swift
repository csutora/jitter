import SwiftUI

struct ContentView: View {
    @State private var messageRouter = MessageRouter()
    @State private var server: LocalServer?

    // Set to true to run the multicore capability test instead of the emulator
    private let multicoreTest = true

    var body: some View {
        Group {
            if multicoreTest, let server = server {
                MulticoreTestView(server: server)
            } else {
                EmulatorWebView(messageRouter: messageRouter)
            }
        }
        .ignoresSafeArea()
        .preferredColorScheme(.dark)
        .persistentSystemOverlays(.hidden)
        .onAppear {
            if multicoreTest {
                startServer()
            }
        }
    }

    private func startServer() {
        // Serve from the app bundle's Resources directory
        let fm = FileManager.default
        let webDir = fm.temporaryDirectory.appendingPathComponent("jitter-web", isDirectory: true)
        try? fm.createDirectory(at: webDir, withIntermediateDirectories: true)

        // Copy test page + any other resources to the serving directory
        let filesToCopy = ["multicore-test.html", "index.html", "busybox-aarch64"]
        for name in filesToCopy {
            let dest = webDir.appendingPathComponent(name)
            try? fm.removeItem(at: dest)
            let ext = (name as NSString).pathExtension
            let base = (name as NSString).deletingPathExtension
            if let src = Bundle.main.url(forResource: base, withExtension: ext.isEmpty ? nil : ext) {
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
