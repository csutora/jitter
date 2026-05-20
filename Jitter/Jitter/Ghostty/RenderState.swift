import Foundation
import GhosttyVt

struct CellRGB: Equatable {
    var r: UInt8
    var g: UInt8
    var b: UInt8
    static let black = CellRGB(r: 0, g: 0, b: 0)
    static let white = CellRGB(r: 255, g: 255, b: 255)
}

struct CellStyle {
    var bold: Bool
    var italic: Bool
    var faint: Bool
    var underline: Bool
    var strikethrough: Bool
    var inverse: Bool
}

struct Cell {
    var codepoints: [UInt32]
    var fg: CellRGB
    var bg: CellRGB
    var style: CellStyle
    var isBlank: Bool { codepoints.isEmpty }
}

enum CursorShape {
    case bar, block, underline, blockHollow
}

struct CursorInfo {
    var x: Int
    var y: Int
    var shape: CursorShape
    var color: CellRGB?
}

final class RenderState {
    private var handle: OpaquePointer?
    private var rowIter: OpaquePointer?
    private var cellsIter: OpaquePointer?

    init?() {
        var state: OpaquePointer?
        guard ghostty_render_state_new(nil, &state) == GHOSTTY_SUCCESS, state != nil else { return nil }
        self.handle = state

        var rIter: OpaquePointer?
        guard ghostty_render_state_row_iterator_new(nil, &rIter) == GHOSTTY_SUCCESS, rIter != nil else {
            ghostty_render_state_free(state)
            return nil
        }
        self.rowIter = rIter

        var cIter: OpaquePointer?
        guard ghostty_render_state_row_cells_new(nil, &cIter) == GHOSTTY_SUCCESS, cIter != nil else {
            ghostty_render_state_row_iterator_free(rIter)
            ghostty_render_state_free(state)
            return nil
        }
        self.cellsIter = cIter
    }

    deinit {
        if let cellsIter { ghostty_render_state_row_cells_free(cellsIter) }
        if let rowIter { ghostty_render_state_row_iterator_free(rowIter) }
        if let handle { ghostty_render_state_free(handle) }
    }

    @discardableResult
    func update(from terminal: Terminal) -> Bool {
        guard let handle, let termHandle = terminal.rawHandle else { return false }
        guard ghostty_render_state_update(handle, termHandle) == GHOSTTY_SUCCESS else { return false }
        if var ri = rowIter {
            ghostty_render_state_get(handle, GHOSTTY_RENDER_STATE_DATA_ROW_ITERATOR, &ri)
        }
        return true
    }

    var defaultBackground: CellRGB {
        guard let handle else { return .black }
        var c = GhosttyColorRgb(r: 0, g: 0, b: 0)
        ghostty_render_state_get(handle, GHOSTTY_RENDER_STATE_DATA_COLOR_BACKGROUND, &c)
        return CellRGB(r: c.r, g: c.g, b: c.b)
    }

    var defaultForeground: CellRGB {
        guard let handle else { return .white }
        var c = GhosttyColorRgb(r: 255, g: 255, b: 255)
        ghostty_render_state_get(handle, GHOSTTY_RENDER_STATE_DATA_COLOR_FOREGROUND, &c)
        return CellRGB(r: c.r, g: c.g, b: c.b)
    }

    var cursorInfo: CursorInfo? {
        guard let handle else { return nil }
        var visible = false
        ghostty_render_state_get(handle, GHOSTTY_RENDER_STATE_DATA_CURSOR_VISIBLE, &visible)
        guard visible else { return nil }
        var inViewport = false
        ghostty_render_state_get(handle, GHOSTTY_RENDER_STATE_DATA_CURSOR_VIEWPORT_HAS_VALUE, &inViewport)
        guard inViewport else { return nil }
        var x: UInt16 = 0, y: UInt16 = 0
        ghostty_render_state_get(handle, GHOSTTY_RENDER_STATE_DATA_CURSOR_VIEWPORT_X, &x)
        ghostty_render_state_get(handle, GHOSTTY_RENDER_STATE_DATA_CURSOR_VIEWPORT_Y, &y)
        var styleRaw = GHOSTTY_RENDER_STATE_CURSOR_VISUAL_STYLE_BLOCK
        ghostty_render_state_get(handle, GHOSTTY_RENDER_STATE_DATA_CURSOR_VISUAL_STYLE, &styleRaw)
        let shape: CursorShape
        switch styleRaw {
        case GHOSTTY_RENDER_STATE_CURSOR_VISUAL_STYLE_BAR: shape = .bar
        case GHOSTTY_RENDER_STATE_CURSOR_VISUAL_STYLE_UNDERLINE: shape = .underline
        case GHOSTTY_RENDER_STATE_CURSOR_VISUAL_STYLE_BLOCK_HOLLOW: shape = .blockHollow
        default: shape = .block
        }
        var hasColor = false
        ghostty_render_state_get(handle, GHOSTTY_RENDER_STATE_DATA_COLOR_CURSOR_HAS_VALUE, &hasColor)
        var color: CellRGB? = nil
        if hasColor {
            var c = GhosttyColorRgb(r: 255, g: 255, b: 255)
            ghostty_render_state_get(handle, GHOSTTY_RENDER_STATE_DATA_COLOR_CURSOR, &c)
            color = CellRGB(r: c.r, g: c.g, b: c.b)
        }
        return CursorInfo(x: Int(x), y: Int(y), shape: shape, color: color)
    }

    /// Iterate the viewport's rows. For each row, `body` is called with the row index
    /// and an array of Cells across that row.
    func forEachRow(_ body: (Int, [Cell]) -> Void) {
        guard let rowIter, var cells = cellsIter else { return }
        var y = 0
        while ghostty_render_state_row_iterator_next(rowIter) {
            let r = ghostty_render_state_row_get(rowIter, GHOSTTY_RENDER_STATE_ROW_DATA_CELLS, &cells)
            if r != GHOSTTY_SUCCESS { y += 1; continue }
            body(y, readCells(cells))
            y += 1
        }
    }

    private func readCells(_ cells: OpaquePointer) -> [Cell] {
        var row: [Cell] = []
        var buf = [UInt32](repeating: 0, count: 16)
        while ghostty_render_state_row_cells_next(cells) {
            var len: UInt32 = 0
            ghostty_render_state_row_cells_get(cells, GHOSTTY_RENDER_STATE_ROW_CELLS_DATA_GRAPHEMES_LEN, &len)

            var codepoints: [UInt32] = []
            if len > 0 {
                let n = min(Int(len), buf.count)
                buf.withUnsafeMutableBufferPointer { bp in
                    ghostty_render_state_row_cells_get(cells, GHOSTTY_RENDER_STATE_ROW_CELLS_DATA_GRAPHEMES_BUF, bp.baseAddress)
                }
                codepoints = Array(buf.prefix(n))
            }

            var fg = GhosttyColorRgb(r: 255, g: 255, b: 255)
            ghostty_render_state_row_cells_get(cells, GHOSTTY_RENDER_STATE_ROW_CELLS_DATA_FG_COLOR, &fg)
            var bg = GhosttyColorRgb(r: 0, g: 0, b: 0)
            ghostty_render_state_row_cells_get(cells, GHOSTTY_RENDER_STATE_ROW_CELLS_DATA_BG_COLOR, &bg)

            var style = GhosttyStyle()
            style.size = MemoryLayout<GhosttyStyle>.size
            ghostty_render_state_row_cells_get(cells, GHOSTTY_RENDER_STATE_ROW_CELLS_DATA_STYLE, &style)

            row.append(Cell(
                codepoints: codepoints,
                fg: CellRGB(r: fg.r, g: fg.g, b: fg.b),
                bg: CellRGB(r: bg.r, g: bg.g, b: bg.b),
                style: CellStyle(
                    bold: style.bold,
                    italic: style.italic,
                    faint: style.faint,
                    underline: style.underline != GHOSTTY_SGR_UNDERLINE_NONE.rawValue,
                    strikethrough: style.strikethrough,
                    inverse: style.inverse
                )
            ))
        }
        return row
    }

    func markClean() {
        guard let handle else { return }
        var clean = GHOSTTY_RENDER_STATE_DIRTY_FALSE
        ghostty_render_state_set(handle, GHOSTTY_RENDER_STATE_OPTION_DIRTY, &clean)
    }
}
