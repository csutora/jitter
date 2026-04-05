import SwiftUI

struct ContentView: View {
    @State private var messageRouter = MessageRouter()

    var body: some View {
        EmulatorWebView(messageRouter: messageRouter)
            .ignoresSafeArea()
            .preferredColorScheme(.dark)
            .persistentSystemOverlays(.hidden)
    }
}

#Preview {
    ContentView()
}
