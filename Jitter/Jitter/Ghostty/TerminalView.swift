import SwiftUI

final class TerminalProxy {
    fileprivate weak var view: TerminalUIView?

    func feed(_ data: Data) {
        view?.feed(data)
    }
}

struct TerminalView: UIViewRepresentable {
    let proxy: TerminalProxy
    var fontSize: CGFloat = 14
    var onStdin: (Data) -> Void = { _ in }
    var onTermSize: (Int, Int) -> Void = { _, _ in }

    func makeUIView(context: Context) -> TerminalUIView {
        let v = TerminalUIView(frame: .zero, fontSize: fontSize)
        v.onStdin = onStdin
        v.onTermSize = onTermSize
        proxy.view = v
        DispatchQueue.main.async { _ = v.becomeFirstResponder() }
        return v
    }

    func updateUIView(_ uiView: TerminalUIView, context: Context) {
        uiView.onStdin = onStdin
        uiView.onTermSize = onTermSize
    }
}
