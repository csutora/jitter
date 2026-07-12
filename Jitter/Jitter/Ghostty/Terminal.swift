import Foundation
import GhosttyVt

/// Swift wrapper around a libghostty-vt terminal handle.
/// Owns the underlying `GhosttyTerminal` (opaque struct) for its lifetime.
final class Terminal {
    private var handle: OpaquePointer?

    init?(cols: UInt16, rows: UInt16, maxScrollback: Int = 10_000) {
        let opts = GhosttyTerminalOptions(
            cols: cols,
            rows: rows,
            max_scrollback: maxScrollback
        )
        var terminal: OpaquePointer?
        let result = ghostty_terminal_new(nil, &terminal, opts)
        guard result == GHOSTTY_SUCCESS, terminal != nil else { return nil }
        self.handle = terminal
    }

    deinit {
        if let handle {
            ghostty_terminal_free(handle)
        }
    }

    func write(_ data: Data) {
        guard let handle else { return }
        data.withUnsafeBytes { (buf: UnsafeRawBufferPointer) in
            guard let base = buf.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            ghostty_terminal_vt_write(handle, base, buf.count)
        }
    }

    func write(bytes: UnsafePointer<UInt8>, count: Int) {
        guard let handle else { return }
        ghostty_terminal_vt_write(handle, bytes, count)
    }

    @discardableResult
    func resize(cols: UInt16, rows: UInt16, cellWidthPx: UInt32, cellHeightPx: UInt32) -> Bool {
        guard let handle else { return false }
        return ghostty_terminal_resize(handle, cols, rows, cellWidthPx, cellHeightPx) == GHOSTTY_SUCCESS
    }

    func reset() {
        guard let handle else { return }
        ghostty_terminal_reset(handle)
    }

    func scrollViewport(delta: Int) {
        guard let handle, delta != 0 else { return }
        var behavior = GhosttyTerminalScrollViewport()
        behavior.tag = GHOSTTY_SCROLL_VIEWPORT_DELTA
        behavior.value.delta = delta
        ghostty_terminal_scroll_viewport(handle, behavior)
    }

    func scrollViewportToBottom() {
        guard let handle else { return }
        var behavior = GhosttyTerminalScrollViewport()
        behavior.tag = GHOSTTY_SCROLL_VIEWPORT_BOTTOM
        ghostty_terminal_scroll_viewport(handle, behavior)
    }

    /// Absolute viewport position within the full scrollback. `offset` is the row index of the
    /// top visible row measured from the very top of the scrollable area (stable content anchor);
    /// `total` is the scrollable height in rows, `len` the visible rows. nil if unavailable.
    func scrollbar() -> (total: Int, offset: Int, len: Int)? {
        guard let handle else { return nil }
        var sb = GhosttyTerminalScrollbar(total: 0, offset: 0, len: 0)
        guard ghostty_terminal_get(handle, GHOSTTY_TERMINAL_DATA_SCROLLBAR, &sb) == GHOSTTY_SUCCESS else { return nil }
        return (Int(sb.total), Int(sb.offset), Int(sb.len))
    }

    var rawHandle: OpaquePointer? { handle }
}
