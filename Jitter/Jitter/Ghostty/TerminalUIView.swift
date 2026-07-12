import UIKit
import CoreText
import GhosttyVt

final class TerminalUIView: UIView, UIKeyInput {
    private var terminal: Terminal?
    private var renderState: RenderState?
    private let font: UIFont
    private let boldFont: UIFont
    private let italicFont: UIFont
    private let boldItalicFont: UIFont
    private let ctFont: CTFont
    private let ctBoldFont: CTFont
    private let ctItalicFont: CTFont
    private let ctBoldItalicFont: CTFont
    private let cellSize: CGSize
    private let baselineFromBottom: CGFloat

    var onStdin: ((Data) -> Void)?
    var onTermSize: ((Int, Int) -> Void)?

    private var lastReportedCols: Int = 0
    private var lastReportedRows: Int = 0

    private struct Selection {
        var anchor: (col: Int, row: Int)
        var head: (col: Int, row: Int)

        var normalized: (start: (col: Int, row: Int), end: (col: Int, row: Int)) {
            if anchor.row < head.row || (anchor.row == head.row && anchor.col <= head.col) {
                return (anchor, head)
            }
            return (head, anchor)
        }

        // anchor/head columns are cell BOUNDARIES (0...cols), boundary b being the left edge of
        // cell b. a cell is inside the selection when it sits between the two boundaries, so the
        // end boundary is exclusive at the cell level.
        func contains(col: Int, row: Int) -> Bool {
            let (s, e) = normalized
            if row < s.row || row > e.row { return false }
            if s.row == e.row { return col >= s.col && col < e.col }
            if row == s.row { return col >= s.col }
            if row == e.row { return col < e.col }
            return true
        }
    }

    private var selection: Selection?
    private let selectionBg = CellRGB(r: 60, g: 110, b: 200)

    var contentInsets: UIEdgeInsets = UIEdgeInsets(top: 8, left: 18, bottom: 8, right: 18) {
        didSet { setNeedsLayout(); setNeedsDisplay() }
    }

    private func effectiveInsets() -> UIEdgeInsets {
        let sa = safeAreaInsets
        return UIEdgeInsets(
            top: contentInsets.top + sa.top,
            left: contentInsets.left + sa.left,
            bottom: contentInsets.bottom + sa.bottom,
            right: contentInsets.right + sa.right
        )
    }

    init(frame: CGRect, fontSize: CGFloat = 14) {
        let f = UIFont.monospacedSystemFont(ofSize: fontSize, weight: .regular)
        let bf = UIFont.monospacedSystemFont(ofSize: fontSize, weight: .bold)
        let italicDesc = f.fontDescriptor.withSymbolicTraits(.traitItalic) ?? f.fontDescriptor
        let boldItalicDesc = bf.fontDescriptor.withSymbolicTraits(.traitItalic) ?? bf.fontDescriptor
        self.font = f
        self.boldFont = bf
        self.italicFont = UIFont(descriptor: italicDesc, size: fontSize)
        self.boldItalicFont = UIFont(descriptor: boldItalicDesc, size: fontSize)
        self.ctFont = CTFontCreateWithFontDescriptor(f.fontDescriptor, fontSize, nil)
        self.ctBoldFont = CTFontCreateWithFontDescriptor(bf.fontDescriptor, fontSize, nil)
        self.ctItalicFont = CTFontCreateWithFontDescriptor(italicDesc, fontSize, nil)
        self.ctBoldItalicFont = CTFontCreateWithFontDescriptor(boldItalicDesc, fontSize, nil)
        let m = "M".size(withAttributes: [.font: f])
        self.cellSize = CGSize(width: ceil(m.width), height: ceil(f.lineHeight))
        self.baselineFromBottom = CTFontGetDescent(self.ctFont)
        super.init(frame: frame)
        self.backgroundColor = .black
        self.isOpaque = true
        self.contentMode = .redraw

        if let term = Terminal(cols: 80, rows: 24),
           let rs = RenderState() {
            self.terminal = term
            self.renderState = rs
        }

        // scroll is driven by a real UIScrollView layered over the terminal. it natively handles
        // finger drag, trackpad / mouse-wheel / continuity-cursor scroll, and momentum — the exact
        // "feels like any other view" behaviour that a custom pan gesture never reliably got for
        // indirect (trackpad) input here. the content is empty and oversized; we never move it, we
        // just translate the scroll offset's deltas into line-locked viewport scrolls.
        scrollProxy.showsVerticalScrollIndicator = false
        scrollProxy.showsHorizontalScrollIndicator = false
        scrollProxy.backgroundColor = .clear
        scrollProxy.isOpaque = false
        scrollProxy.contentInsetAdjustmentBehavior = .never
        scrollProxy.delaysContentTouches = false
        scrollProxy.delegate = self
        // the proxy's pan drives scrolling for FINGER drags only (plus indirect trackpad/wheel
        // scroll events, which aren't touches and so aren't gated here). a pointer click-drag
        // (.indirectPointer) is left for the selection gesture below, matching the platform
        // convention: finger drags scroll, a cursor click-drag selects.
        scrollProxy.panGestureRecognizer.allowedTouchTypes = [NSNumber(value: UITouch.TouchType.direct.rawValue)]
        addSubview(scrollProxy)

        let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap))
        tap.cancelsTouchesInView = false
        scrollProxy.addGestureRecognizer(tap)

        // FINGER selection: press-and-hold then drag (disables scrolling while active so the drag
        // extends the selection; a quick finger drag just scrolls).
        let select = UILongPressGestureRecognizer(target: self, action: #selector(handleSelect(_:)))
        select.minimumPressDuration = 0.35
        select.allowedTouchTypes = [NSNumber(value: UITouch.TouchType.direct.rawValue)]
        scrollProxy.addGestureRecognizer(select)

        // POINTER selection: a trackpad/mouse click-drag selects immediately (no hold needed). it
        // only sees .indirectPointer touches, which the scroll pan ignores, so the two never fight.
        let pointerSelect = DownLocationPan(target: self, action: #selector(handlePointerSelect(_:)))
        pointerSelect.allowedTouchTypes = [NSNumber(value: UITouch.TouchType.indirectPointer.rawValue)]
        scrollProxy.addGestureRecognizer(pointerSelect)
    }

    // map a point to the nearest cell BOUNDARY column (round, not truncate) so a cell is grabbed
    // once the cursor passes its midpoint — no need to click pixel-perfectly at a cell's edge. the
    // column ranges 0...cols (cols = right edge of the last cell); the row is the containing cell.
    private func selBoundary(point p: CGPoint) -> (col: Int, row: Int) {
        let inner = bounds.inset(by: effectiveInsets())
        let col = max(0, min(lastReportedCols, Int(((p.x - inner.minX) / cellSize.width).rounded())))
        let row = max(0, min(lastReportedRows - 1, Int((p.y - inner.minY) / cellSize.height)))
        return (col, viewportTop + row) // absolute row (survives scroll/output)
    }

    // capture the absolute viewport top at the start of a selection so its coordinates are absolute
    // from the first point (draw() only refreshes viewportTop once a selection exists).
    private func beginSelectionCoords() { viewportTop = terminal?.scrollbar()?.offset ?? 0 }

    @objc private func handlePointerSelect(_ g: DownLocationPan) {
        let p = g.location(in: self)
        switch g.state {
        case .began:
            beginSelectionCoords()
            // anchor at the true click point (captured at touch-down), not wherever the pointer had
            // already drifted to by the time the pan began.
            let start = g.downLocation ?? p
            selection = Selection(anchor: selBoundary(point: start), head: selBoundary(point: p))
            setNeedsDisplay()
        case .changed:
            selection?.head = selBoundary(point: p)
            setNeedsDisplay()
        default:
            break
        }
    }

    // ── scrolling (UIScrollView-driven, line-locked) ─────────────────────────
    private let scrollProxy = UIScrollView()
    private let scrollCenter: CGFloat = 100_000_000 // oversized so the offset never reaches an edge
    private var lastScrollY: CGFloat = 0
    private var scrollAccum: CGFloat = 0
    private var isSelecting = false
    private var suppressScrollMap = false
    // the absolute row index of the top visible row within the full scrollback (ghostty's scrollbar
    // offset), refreshed every draw. the selection is stored in these ABSOLUTE coordinates, so it
    // stays pinned to its content as the viewport scrolls or new output arrives — no delta tracking
    // to drift. converted back to a screen row (absolute - viewportTop) at render/copy time.
    private var viewportTop = 0

    private func layoutScrollProxy() {
        scrollProxy.frame = bounds
        let h = bounds.height + 2 * scrollCenter
        if scrollProxy.contentSize.height != h {
            scrollProxy.contentSize = CGSize(width: bounds.width, height: h)
            recenterScroll()
        }
    }

    // park the offset in the middle of the huge content so it can be dragged either way ~forever;
    // we only ever read deltas, never the absolute position.
    private func recenterScroll() {
        suppressScrollMap = true
        scrollProxy.contentOffset = CGPoint(x: 0, y: scrollCenter)
        lastScrollY = scrollCenter
        scrollAccum = 0
        suppressScrollMap = false
    }

    required init?(coder: NSCoder) { fatalError("unsupported") }

    func feed(_ data: Data) {
        terminal?.write(data)
        setNeedsDisplay()
    }

    @objc private func handleTap() {
        if selection != nil {
            selection = nil
            setNeedsDisplay()
        }
        if !isFirstResponder { becomeFirstResponder() }
    }


    @objc private func handleSelect(_ g: UILongPressGestureRecognizer) {
        let p = g.location(in: self)
        switch g.state {
        case .began:
            isSelecting = true
            scrollProxy.isScrollEnabled = false // don't scroll while extending a selection
            beginSelectionCoords()
            selection = Selection(anchor: selBoundary(point: p), head: selBoundary(point: p))
            setNeedsDisplay()
        case .changed:
            selection?.head = selBoundary(point: p)
            setNeedsDisplay()
        case .ended, .cancelled, .failed:
            isSelecting = false
            scrollProxy.isScrollEnabled = true
            recenterScroll()
        default:
            break
        }
    }

    override var canBecomeFirstResponder: Bool { true }

    private func snapViewportToBottom() {
        // the selection is stored in absolute scrollback coordinates, so it stays pinned to its
        // content across the jump — nothing to adjust here.
        terminal?.scrollViewportToBottom()
        recenterScroll()
    }

    // MARK: UIKeyInput

    var hasText: Bool { true }

    func insertText(_ text: String) {
        var bytes = Array(text.utf8)
        for i in 0..<bytes.count where bytes[i] == 0x0A { bytes[i] = 0x0D }
        clearSelection()
        snapViewportToBottom()
        onStdin?(Data(bytes))
    }

    func deleteBackward() {
        clearSelection()
        snapViewportToBottom()
        onStdin?(Data([0x7F]))
    }

    private func clearSelection() {
        if selection != nil {
            selection = nil
            setNeedsDisplay()
        }
    }

    override func copy(_ sender: Any?) {
        copySelectionToClipboard()
    }

    override func paste(_ sender: Any?) {
        pasteFromClipboard()
    }

    override func canPerformAction(_ action: Selector, withSender sender: Any?) -> Bool {
        if action == #selector(copy(_:)) { return selection != nil }
        if action == #selector(paste(_:)) { return UIPasteboard.general.hasStrings }
        return super.canPerformAction(action, withSender: sender)
    }

    private func copySelectionToClipboard() {
        guard let sel = selection, let renderState, let terminal else { return }
        guard renderState.update(from: terminal) else { return }
        let (start, end) = sel.normalized
        // selection rows are absolute; map each visible row to its absolute index via the viewport
        // top. (only copies the on-screen portion of a selection — a scrollback-spanning copy is a
        // separate feature.)
        let top = terminal.scrollbar()?.offset ?? viewportTop
        var lines: [String] = []
        renderState.forEachRow { y, row in
            let absY = top + y
            if absY < start.row || absY > end.row { return }
            let startCol = (absY == start.row) ? start.col : 0
            // end column is a boundary (exclusive at the cell level); a non-end row runs to the
            // line end.
            let endColExcl = (absY == end.row) ? end.col : row.count
            var line = ""
            var x = startCol
            while x < min(endColExcl, row.count) {
                for cp in row[x].codepoints {
                    if let s = Unicode.Scalar(cp) { line.append(Character(s)) }
                }
                if row[x].codepoints.isEmpty { line.append(" ") }
                x += 1
            }
            while line.hasSuffix(" ") { line.removeLast() }
            lines.append(line)
        }
        UIPasteboard.general.string = lines.joined(separator: "\n")
    }

    private func pasteFromClipboard() {
        guard let s = UIPasteboard.general.string else { return }
        clearSelection()
        snapViewportToBottom()
        onStdin?(Data(s.utf8))
    }

    override var keyCommands: [UIKeyCommand]? {
        var cmds: [UIKeyCommand] = [
            UIKeyCommand(input: UIKeyCommand.inputUpArrow, modifierFlags: [], action: #selector(handleKey(_:))),
            UIKeyCommand(input: UIKeyCommand.inputDownArrow, modifierFlags: [], action: #selector(handleKey(_:))),
            UIKeyCommand(input: UIKeyCommand.inputRightArrow, modifierFlags: [], action: #selector(handleKey(_:))),
            UIKeyCommand(input: UIKeyCommand.inputLeftArrow, modifierFlags: [], action: #selector(handleKey(_:))),
            UIKeyCommand(input: UIKeyCommand.inputEscape, modifierFlags: [], action: #selector(handleKey(_:))),
            UIKeyCommand(input: "\t", modifierFlags: [], action: #selector(handleKey(_:))),
        ]
        for c in "abcdefghijklmnopqrstuvwxyz" {
            cmds.append(UIKeyCommand(input: String(c), modifierFlags: .control, action: #selector(handleKey(_:))))
        }
        return cmds
    }

    @objc private func handleKey(_ cmd: UIKeyCommand) {
        guard let input = cmd.input else { return }
        clearSelection()
        snapViewportToBottom()
        if cmd.modifierFlags.contains(.control), input.count == 1,
           let ch = input.lowercased().unicodeScalars.first, ch.value >= 0x60, ch.value < 0x80 {
            onStdin?(Data([UInt8(ch.value - 0x60)]))
            return
        }
        switch input {
        case UIKeyCommand.inputUpArrow: onStdin?(Data("\u{1B}[A".utf8))
        case UIKeyCommand.inputDownArrow: onStdin?(Data("\u{1B}[B".utf8))
        case UIKeyCommand.inputRightArrow: onStdin?(Data("\u{1B}[C".utf8))
        case UIKeyCommand.inputLeftArrow: onStdin?(Data("\u{1B}[D".utf8))
        case UIKeyCommand.inputEscape: onStdin?(Data([0x1B]))
        case "\t": onStdin?(Data([0x09]))
        default: break
        }
    }

    override func safeAreaInsetsDidChange() {
        super.safeAreaInsetsDidChange()
        setNeedsLayout()
        setNeedsDisplay()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        layoutScrollProxy()
        guard let terminal else { return }
        let inner = bounds.inset(by: effectiveInsets())
        let cols = max(1, Int(inner.width / cellSize.width))
        let rows = max(1, Int(inner.height / cellSize.height))
        let scale = window?.windowScene?.screen.scale ?? traitCollection.displayScale
        terminal.resize(
            cols: UInt16(cols),
            rows: UInt16(rows),
            cellWidthPx: UInt32(cellSize.width * scale),
            cellHeightPx: UInt32(cellSize.height * scale)
        )
        if cols != lastReportedCols || rows != lastReportedRows {
            lastReportedCols = cols
            lastReportedRows = rows
            onTermSize?(cols, rows)
        }
        setNeedsDisplay()
    }

    // MARK: - Rendering

    private func ctFont(bold: Bool, italic: Bool) -> CTFont {
        switch (bold, italic) {
        case (true, true): return ctBoldItalicFont
        case (true, false): return ctBoldFont
        case (false, true): return ctItalicFont
        case (false, false): return ctFont
        }
    }

    private func uiFont(bold: Bool, italic: Bool) -> UIFont {
        switch (bold, italic) {
        case (true, true): return boldItalicFont
        case (true, false): return boldFont
        case (false, true): return italicFont
        case (false, false): return font
        }
    }

    override func draw(_ rect: CGRect) {
        guard let ctx = UIGraphicsGetCurrentContext(),
              let terminal,
              let renderState else { return }
        guard renderState.update(from: terminal) else { return }
        // absolute row of the top visible line; the selection is stored in absolute coords and this
        // maps them to screen rows. only queried when there's a live selection (it can be costly).
        viewportTop = (selection != nil ? terminal.scrollbar()?.offset : 0) ?? 0

        let cellW = cellSize.width
        let cellH = cellSize.height
        let inner = bounds.inset(by: effectiveInsets())
        let defaultBg = renderState.defaultBackground
        let cur = renderState.cursorInfo
        let cursorVisuallyOn = cur != nil
        let cursorColor: CellRGB = cur?.color ?? renderState.defaultForeground
        let blockCursorActive = cursorVisuallyOn && cur?.shape == .block

        ctx.setFillColor(red: CGFloat(defaultBg.r)/255, green: CGFloat(defaultBg.g)/255, blue: CGFloat(defaultBg.b)/255, alpha: 1)
        ctx.fill(bounds)

        ctx.textMatrix = .identity
        ctx.translateBy(x: 0, y: bounds.height)
        ctx.scaleBy(x: 1, y: -1)

        renderState.forEachRow { y, row in
            let rowFlippedBottom = bounds.height - inner.minY - CGFloat(y + 1) * cellH
            let textY = rowFlippedBottom + baselineFromBottom

            // Pass 1: background fills, coalesced
            var i = 0
            while i < row.count {
                let bg = effectiveBg(row[i], x: i, y: y, cur: cur, blockCursorActive: blockCursorActive, cursorColor: cursorColor, defaultBg: defaultBg)
                if bg == defaultBg { i += 1; continue }
                var j = i + 1
                while j < row.count {
                    let nbg = effectiveBg(row[j], x: j, y: y, cur: cur, blockCursorActive: blockCursorActive, cursorColor: cursorColor, defaultBg: defaultBg)
                    if nbg != bg { break }
                    j += 1
                }
                ctx.setFillColor(red: CGFloat(bg.r)/255, green: CGFloat(bg.g)/255, blue: CGFloat(bg.b)/255, alpha: 1)
                let x0 = inner.minX + CGFloat(i) * cellW
                let width = CGFloat(j - i) * cellW
                ctx.fill(CGRect(x: x0, y: rowFlippedBottom, width: width, height: cellH))
                i = j
            }

            // Pass 2: glyphs + decorations per cell
            for (x, cell) in row.enumerated() {
                let isCursorCell = cur != nil && cur!.x == x && cur!.y == y
                let fg = effectiveFg(cell, isCursorCell: isCursorCell, blockCursorActive: blockCursorActive, defaultBg: defaultBg)
                let xPx = inner.minX + CGFloat(x) * cellW

                if !cell.isBlank {
                    let alpha: CGFloat = cell.style.faint ? 0.5 : 1.0
                    drawCell(ctx, codepoints: cell.codepoints, fg: fg, alpha: alpha,
                             bold: cell.style.bold, italic: cell.style.italic,
                             at: CGPoint(x: xPx, y: textY))
                }
                if cell.style.underline {
                    ctx.setFillColor(red: CGFloat(fg.r)/255, green: CGFloat(fg.g)/255, blue: CGFloat(fg.b)/255, alpha: 1)
                    ctx.fill(CGRect(x: xPx, y: rowFlippedBottom + max(1, baselineFromBottom * 0.3), width: cellW, height: 1))
                }
                if cell.style.strikethrough {
                    ctx.setFillColor(red: CGFloat(fg.r)/255, green: CGFloat(fg.g)/255, blue: CGFloat(fg.b)/255, alpha: 1)
                    ctx.fill(CGRect(x: xPx, y: rowFlippedBottom + cellH * 0.5, width: cellW, height: 1))
                }
            }
        }

        // Non-block cursor decorations
        if cursorVisuallyOn, let cur, cur.shape != .block {
            let cellX = inner.minX + CGFloat(cur.x) * cellW
            let cellY = inner.minY + CGFloat(cur.y) * cellH
            let flippedY = bounds.height - cellY - cellH
            let c = cursorColor
            ctx.setFillColor(red: CGFloat(c.r)/255, green: CGFloat(c.g)/255, blue: CGFloat(c.b)/255, alpha: 1)
            ctx.setStrokeColor(red: CGFloat(c.r)/255, green: CGFloat(c.g)/255, blue: CGFloat(c.b)/255, alpha: 1)
            switch cur.shape {
            case .blockHollow:
                ctx.setLineWidth(1)
                ctx.stroke(CGRect(x: cellX, y: flippedY, width: cellW, height: cellH).insetBy(dx: 0.5, dy: 0.5))
            case .bar:
                ctx.fill(CGRect(x: cellX, y: flippedY, width: max(1, cellW * 0.1), height: cellH))
            case .underline:
                ctx.fill(CGRect(x: cellX, y: flippedY, width: cellW, height: max(1, cellH * 0.1)))
            case .block:
                break
            }
        }

        renderState.markClean()
    }

    private func effectiveBg(_ cell: Cell, x: Int, y: Int, cur: CursorInfo?, blockCursorActive: Bool, cursorColor: CellRGB, defaultBg: CellRGB) -> CellRGB {
        if let sel = selection, sel.contains(col: x, row: viewportTop + y) { return selectionBg }
        if blockCursorActive, let cur, cur.x == x, cur.y == y { return cursorColor }
        return cell.style.inverse ? cell.fg : cell.bg
    }

    private func effectiveFg(_ cell: Cell, isCursorCell: Bool, blockCursorActive: Bool, defaultBg: CellRGB) -> CellRGB {
        if isCursorCell && blockCursorActive { return defaultBg }
        return cell.style.inverse ? cell.bg : cell.fg
    }

    private func drawCell(_ ctx: CGContext, codepoints: [UInt32], fg: CellRGB, alpha: CGFloat, bold: Bool, italic: Bool, at origin: CGPoint) {
        let font = ctFont(bold: bold, italic: italic)
        ctx.setFillColor(red: CGFloat(fg.r)/255, green: CGFloat(fg.g)/255, blue: CGFloat(fg.b)/255, alpha: alpha)

        if codepoints.count == 1, codepoints[0] <= 0xFFFF {
            var ch = unichar(codepoints[0])
            var glyph: CGGlyph = 0
            if CTFontGetGlyphsForCharacters(font, &ch, &glyph, 1) {
                var pos = origin
                CTFontDrawGlyphs(font, &glyph, &pos, 1, ctx)
                return
            }
        }

        var scalars = String.UnicodeScalarView()
        for cp in codepoints {
            if let s = Unicode.Scalar(cp) { scalars.append(s) }
        }
        let s = String(scalars)
        if s.isEmpty { return }
        let uifont = uiFont(bold: bold, italic: italic)
        let attrs: [NSAttributedString.Key: Any] = [
            .font: uifont,
            .foregroundColor: UIColor(red: CGFloat(fg.r)/255, green: CGFloat(fg.g)/255, blue: CGFloat(fg.b)/255, alpha: alpha),
        ]
        let attributed = NSAttributedString(string: s, attributes: attrs)
        let line = CTLineCreateWithAttributedString(attributed)
        ctx.textPosition = origin
        CTLineDraw(line, ctx)
    }
}

// a pan that remembers where the touch actually went down. UIPanGestureRecognizer only begins
// after some movement, and its translation at .began can be ~0, so `location - translation` does
// NOT reliably recover the click point — the selection anchor would land wherever the pointer had
// already drifted to. capturing the down location in touchesBegan fixes the selection start.
final class DownLocationPan: UIPanGestureRecognizer {
    var downLocation: CGPoint?
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent) {
        // capture in the proxy's SUPERVIEW (the terminal), not the proxy itself: the proxy is a
        // UIScrollView whose bounds origin is its (huge, recentered) contentOffset, so a location
        // in its own space is offset by ~100M points. the handler reads location(in: terminal),
        // so match that space here.
        if downLocation == nil, let t = touches.first {
            downLocation = t.location(in: view?.superview)
        }
        super.touchesBegan(touches, with: event)
    }
    override func reset() {
        super.reset()
        downLocation = nil
    }
}

extension TerminalUIView: UIScrollViewDelegate {
    // translate the scroll proxy's offset deltas into line-locked viewport scrolls. offset
    // increasing (dragging toward the bottom of content) reveals newer output -> positive delta;
    // decreasing reveals older -> negative. we never move the (empty) content, only read deltas.
    // the selection is pinned to absolute scrollback coords, so it needs no adjustment here — the
    // renderer maps it to screen rows each frame via the viewport offset.
    func scrollViewDidScroll(_ sv: UIScrollView) {
        if suppressScrollMap || isSelecting { return }
        let delta = sv.contentOffset.y - lastScrollY
        lastScrollY = sv.contentOffset.y
        scrollAccum += delta
        let lines = Int((scrollAccum / cellSize.height).rounded(.towardZero))
        if lines != 0 {
            scrollAccum -= CGFloat(lines) * cellSize.height
            terminal?.scrollViewport(delta: lines)
            setNeedsDisplay()
        }
        // never let the offset drift near the oversized content's edge over a very long session.
        if abs(sv.contentOffset.y - scrollCenter) > scrollCenter * 0.9 { recenterScroll() }
    }
}
