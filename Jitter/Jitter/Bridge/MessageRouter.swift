import Foundation
import WebKit

// Parse the guest process credentials the TS layer threads into path-op bridge args. Absent
// creds default to root — only the trusted worker builds these, so "no creds" means an internal
// app operation, which runs privileged.
private func parseCreds(_ args: [String: Any]) -> FsCreds {
    guard let c = args["creds"] as? [String: Any] else { return .root }
    func u(_ k: String, _ d: UInt32) -> UInt32 { (c[k] as? Int).map { UInt32(truncatingIfNeeded: $0) } ?? d }
    let groups = (c["groups"] as? [Int])?.map { UInt32(truncatingIfNeeded: $0) } ?? []
    return FsCreds(ruid: u("ruid", 0), euid: u("euid", 0), rgid: u("rgid", 0), egid: u("egid", 0),
                   umask: u("umask", 0o022), groups: groups)
}

/// Routes messages from JS to native handlers using WKScriptMessageHandlerWithReply (iOS 14+).
///
/// JS calls: `const result = await window.webkit.messageHandlers.jitter.postMessage({method, args})`
/// The Promise is fulfilled directly by Swift via replyHandler — no evaluateJavaScript round-trip.
class MessageRouter: NSObject, WKScriptMessageHandlerWithReply {
    weak var webView: WKWebView?
    private var handlers: [String: ([String: Any], @escaping (Any?, String?) -> Void) -> Void] = [:]
    private(set) var nativeFS: NativeFS?
    var onTerminalData: ((Data) -> Void)?
    private var lastTermCols: Int = 0
    private var lastTermRows: Int = 0

    // ── debug control channel ────────────────────────────────────────────────
    // taps a rolling copy of everything written to the terminal and lets an external
    // driver inject stdin / read output / reboot the guest over http (see LocalServer).
    // dev tooling for the in-vivo test loop; gated on by ContentView.
    private let ctlLock = NSLock()
    private var ctlBuffer = Data()
    private let ctlCap = 8 * 1024 * 1024
    private func ctlAppend(_ d: Data) {
        ctlLock.lock(); defer { ctlLock.unlock() }
        ctlBuffer.append(d)
        if ctlBuffer.count > ctlCap { ctlBuffer.removeFirst(ctlBuffer.count - ctlCap) }
    }
    func ctlReadOutput() -> Data {
        ctlLock.lock(); defer { ctlLock.unlock() }
        return ctlBuffer
    }
    func ctlClearOutput() {
        ctlLock.lock(); ctlBuffer.removeAll(keepingCapacity: true); ctlLock.unlock()
    }
    func ctlInject(_ text: String) { sendStdin(Data(text.utf8)) }

    // run a shell command and BLOCK until it finishes, returning exactly its output + exit code.
    // injects `<cmd>; echo <marker>$?<end>` and polls the in-memory output tap for the marker (which
    // carries the exit status) so the caller (a dev driver over http) never has to guess a wait time.
    // the marker literal is split in the injected source ("==J""X..") so the shell's echo of the typed
    // line doesn't itself match the search string — only the command's actual output does.
    private var execCounter = 0

    /// frozen process images, held in THIS process rather than in WebKit's content process.
    /// that is the whole point of the tier: jsc caps total WebAssembly.Memory at 4 GiB inside the
    /// content process, and no entitlement or feature flag moves it, but the app process with
    /// increased-memory-limit gets 16 GiB (measured: 5120 -> 16384 MiB, and 10 GiB committed with
    /// room to spare). only the bulk memory image travels; the small bookkeeping stays in js.
    private var swapStore: [Int: Data] = [:]

    /// images spilled to disk once the app process is itself under pressure. the tier only exists
    /// because os_proc_available_memory tells us honestly how much room is left, so the decision
    /// is made on a real number rather than a guess. cold images are rare by construction: a
    /// process only reaches this tier if it has been frozen and then not wanted for a while.
    private var swapOnDisk: [Int: URL] = [:]
    private lazy var swapDir: URL = {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("jitter-swap", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }()
    /// spill when the app process has less than this much headroom left
    private let swapSpillThreshold = 2048 * 1024 * 1024
    /// forces the disk path regardless of headroom, so the cold tier can be exercised without
    /// first pushing the device to within 2 GiB of its limit -- a tier only ever tested under real
    /// pressure is a tier that gets tested once
    private var swapForceDisk = false
    func execCommand(_ cmd: String, timeoutMs: Int = 30000) -> String {
        ctlLock.lock()
        execCounter += 1
        let id = execCounter
        let startLen = ctlBuffer.count
        ctlLock.unlock()
        let marker = [UInt8]("==JX\(id)=".utf8) // appears only in output (echoed source is ==J""X..)
        // settle the line first. every process reading fd 0 shares one stdin ring, so a command
        // interrupted a moment ago can still be draining it and will swallow the first characters
        // of the next one -- "echo ready" reaching the shell as "ho ready". a bare newline gives
        // any such straggler something harmless to eat, and the pause lets it finish doing so.
        // (the real fix is terminal ownership by process group; see the task.)
        ctlInject("\n")
        usleep(60000)
        ctlInject("\(cmd); echo \"==J\"\"X\(id)=$?==\"\n")

        // find the first occurrence of `needle` in `hay` at/after `from`, byte-wise (Swift Characters
        // fold "\r\n" into one grapheme, so all of this must operate on raw bytes).
        func find(_ hay: [UInt8], _ needle: [UInt8], _ from: Int = 0) -> Int? {
            guard !needle.isEmpty, hay.count >= needle.count else { return nil }
            var i = from
            while i <= hay.count - needle.count {
                if Array(hay[i..<i+needle.count]) == needle { return i }
                i += 1
            }
            return nil
        }

        let deadline = Date().addingTimeInterval(Double(timeoutMs) / 1000.0)
        while Date() < deadline {
            ctlLock.lock()
            let tail = ctlBuffer.count > startLen ? [UInt8](ctlBuffer.subdata(in: startLen..<ctlBuffer.count)) : []
            ctlLock.unlock()
            if let m = find(tail, marker) {
                // exit code = digits right after the marker
                var j = m + marker.count
                var rc = ""
                while j < tail.count, tail[j] >= 0x30, tail[j] <= 0x39 { rc.append(Character(UnicodeScalar(tail[j]))); j += 1 }
                // output = everything after the echoed command line (ends at the first \n) up to the marker
                let lf = tail.firstIndex(of: 0x0a)
                let start = (lf != nil && lf! + 1 <= m) ? lf! + 1 : 0
                let out = start < m ? String(decoding: tail[start..<m], as: UTF8.self) : ""
                return out + "[exit \(rc.isEmpty ? "?" : rc)]\n"
            }
            usleep(15000) // 15ms — local, cheap, responsive
        }
        // the command never produced the marker within the budget (likely a hang). send ^C to
        // interrupt it so the shell recovers for the next exec, and return what output we captured.
        sendStdin(Data([0x03]))
        ctlLock.lock()
        let tail = ctlBuffer.count > startLen ? ctlBuffer.subdata(in: startLen..<ctlBuffer.count) : Data()
        ctlLock.unlock()
        return String(decoding: tail, as: UTF8.self) + "\n[timeout after \(timeoutMs)ms — sent SIGINT]\n"
    }

    // synchronous pull for the debug /ctl channel: download a tarball + install into roots/<name>.
    // blocks the caller (a LocalServer connection thread) until done. the guest-facing async path is
    // the jitter_pull bridge handler.
    func performPull(_ name: String, _ urlStr: String) -> String {
        guard let url = URL(string: urlStr), let fs = nativeFS else { return "bad url or no fs" }
        let sem = DispatchSemaphore(value: 0)
        var result = "download failed"
        URLSession.shared.dataTask(with: url) { data, _, err in
            defer { sem.signal() }
            if let err = err { result = "download error: \(err.localizedDescription)"; return }
            guard let data = data, !data.isEmpty else { result = "empty download"; return }
            let r = fs.installRootfs(data, into: name)
            result = "\(r.note): \(r.files) files -> roots/\(name) (\(data.count) bytes)"
        }.resume()
        _ = sem.wait(timeout: .now() + 180)
        return result
    }
    func ctlReset() {
        ctlClearOutput()
        DispatchQueue.main.async { [weak self] in self?.webView?.reload() }
    }

    override init() {
        super.init()
        setupHandlers()
    }

    private func setupHandlers() {
        let fs = NativeFS()
        self.nativeFS = fs
        let net = NetworkBridge()

        // ── Network bridge ──

        handlers["net_socket"] = { args, reply in
            guard let domain = args["domain"] as? Int,
                  let type = args["type"] as? Int,
                  let proto = args["protocol"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(net.socket(domain: domain, type: type, protocol: proto), nil)
        }

        handlers["net_connect"] = { args, reply in
            guard let socketId = args["socketId"] as? Int,
                  let family = args["family"] as? Int,
                  let host = args["host"] as? String,
                  let port = args["port"] as? Int else {
                reply(["ret": -22], nil); return
            }
            net.connect(socketId: socketId, family: family, host: host, port: port) { result in
                reply(result, nil)
            }
        }

        handlers["net_bind"] = { args, reply in
            guard let socketId = args["socketId"] as? Int,
                  let family = args["family"] as? Int,
                  let host = args["host"] as? String,
                  let port = args["port"] as? Int else {
                reply(["ret": -22], nil); return
            }
            net.bind(socketId: socketId, family: family, host: host, port: port) { result in
                reply(result, nil)
            }
        }

        handlers["net_listen"] = { args, reply in
            guard let socketId = args["socketId"] as? Int,
                  let backlog = args["backlog"] as? Int else {
                reply(["ret": -22], nil); return
            }
            net.listen(socketId: socketId, backlog: backlog) { result in
                reply(result, nil)
            }
        }

        handlers["net_accept"] = { args, reply in
            guard let socketId = args["socketId"] as? Int else {
                reply(["ret": -9], nil); return
            }
            net.accept(socketId: socketId) { result in
                reply(result, nil)
            }
        }

        handlers["net_read"] = { args, reply in
            guard let socketId = args["socketId"] as? Int,
                  let count = args["count"] as? Int else {
                reply(["ret": -9], nil); return
            }
            net.read(socketId: socketId, count: count,
                     nonblock: (args["nonblock"] as? Int ?? 0) != 0,
                     peek: (args["peek"] as? Int ?? 0) != 0) { result in
                reply(result, nil)
            }
        }

        handlers["net_write"] = { args, reply in
            guard let socketId = args["socketId"] as? Int,
                  let data = args["data"] as? String else {
                reply(["ret": -22], nil); return
            }
            net.write(socketId: socketId, data: data,
                      more: (args["more"] as? Int ?? 0) != 0) { result in
                reply(result, nil)
            }
        }

        handlers["net_close"] = { args, reply in
            guard let socketId = args["socketId"] as? Int else {
                reply(["ret": -9], nil); return
            }
            reply(net.close(socketId: socketId), nil)
        }

        handlers["net_sendmmsg"] = { args, reply in
            guard let socketId = args["socketId"] as? Int,
                  let msgs = args["msgs"] as? [[String: Any]] else {
                reply(["ret": -22], nil); return
            }
            net.sendmmsg(socketId: socketId, msgs: msgs) { result in reply(result, nil) }
        }

        handlers["net_recvmmsg"] = { args, reply in
            guard let socketId = args["socketId"] as? Int,
                  let vlen = args["vlen"] as? Int,
                  let maxLen = args["maxLen"] as? Int else {
                reply(["ret": -22], nil); return
            }
            let timeoutMs = args["timeoutMs"] as? Int ?? 0
            net.recvmmsg(socketId: socketId, vlen: vlen, maxLen: maxLen, timeoutMs: timeoutMs) { result in
                reply(result, nil)
            }
        }

        handlers["net_ifaddrs"] = { _, reply in
            reply(net.interfaces(), nil)
        }

        handlers["net_ref"] = { args, reply in
            guard let socketId = args["socketId"] as? Int else {
                reply(["ret": -9], nil); return
            }
            reply(net.ref(socketId: socketId), nil)
        }

        handlers["net_poll"] = { args, reply in
            guard let socketId = args["socketId"] as? Int,
                  let timeoutMs = args["timeoutMs"] as? Int else {
                reply(["ret": 0], nil); return
            }
            net.poll(socketId: socketId, timeoutMs: timeoutMs) { result in
                reply(result, nil)
            }
        }

        handlers["net_sendto"] = { args, reply in
            guard let socketId = args["socketId"] as? Int,
                  let data = args["data"] as? String,
                  let host = args["host"] as? String,
                  let port = args["port"] as? Int else {
                reply(["ret": -22], nil); return
            }
            net.sendto(socketId: socketId, data: data, host: host, port: port,
                       more: (args["more"] as? Int ?? 0) != 0) { result in
                reply(result, nil)
            }
        }

        handlers["net_sockopt"] = { _, reply in
            reply(["ret": 0], nil)
        }

        handlers["ping"] = { _, reply in
            reply(["pong": true], nil)
        }

        // reclaim every descriptor a process still holds. the guest side closes its own on any
        // exit path it survives to run, but a force-terminated one (SIGKILL) never executes
        // another line, so without this its FileHandles live until the app dies.
        handlers["fs_close_owner"] = { args, reply in
            guard let pid = args["pid"] as? Int else { reply(["closed": 0], nil); return }
            reply(["closed": fs.closeOwner(pid: pid), "stillOpen": fs.openFdCount()], nil)
        }

        handlers["fs_stat_fds"] = { _, reply in
            reply(["open": fs.openFdCount()], nil)
        }

        // ── swapped process images ──

        handlers["swap_store"] = { [weak self] args, reply in
            guard let self, let pid = args["pid"] as? Int,
                  let b64 = args["data"] as? String,
                  let d = Data(base64Encoded: b64) else { reply(["ret": -22], nil); return }
            // hold it in memory while there is room, otherwise put it straight on disk. either
            // way the content process is rid of it, which is the part that matters.
            if !self.swapForceDisk && MemoryProbe.availableBytes() > self.swapSpillThreshold {
                self.swapStore[pid] = d
                reply(["ret": d.count, "where": "memory"], nil)
            } else {
                let url = self.swapDir.appendingPathComponent("\(pid).img")
                do {
                    try d.write(to: url, options: .atomic)
                    self.swapOnDisk[pid] = url
                    reply(["ret": d.count, "where": "disk"], nil)
                } catch {
                    self.swapStore[pid] = d // disk refused; memory is still better than nothing
                    reply(["ret": d.count, "where": "memory"], nil)
                }
            }
        }

        handlers["swap_load"] = { [weak self] args, reply in
            guard let self, let pid = args["pid"] as? Int else { reply(["error": "no image"], nil); return }
            if let d = self.swapStore[pid] { reply(["data": d.base64EncodedString()], nil); return }
            if let url = self.swapOnDisk[pid], let d = try? Data(contentsOf: url) {
                reply(["data": d.base64EncodedString()], nil); return
            }
            reply(["error": "no image"], nil)
        }

        handlers["swap_drop"] = { [weak self] args, reply in
            guard let self, let pid = args["pid"] as? Int else { reply(["ret": -22], nil); return }
            self.swapStore.removeValue(forKey: pid)
            if let url = self.swapOnDisk.removeValue(forKey: pid) { try? FileManager.default.removeItem(at: url) }
            reply(["ret": 0], nil)
        }

        handlers["swap_force_disk"] = { [weak self] args, reply in
            guard let self else { reply(["ret": -1], nil); return }
            self.swapForceDisk = (args["on"] as? Bool) ?? false
            reply(["ret": 0, "forceDisk": self.swapForceDisk], nil)
        }

        handlers["swap_stat"] = { [weak self] _, reply in
            guard let self else { reply(["count": 0, "bytes": 0], nil); return }
            let total = self.swapStore.values.reduce(0) { $0 + $1.count }
            let diskBytes = self.swapOnDisk.values.reduce(0) { acc, url in
                let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
                return acc + ((attrs?[.size] as? Int) ?? 0)
            }
            reply(["count": self.swapStore.count + self.swapOnDisk.count,
                   "bytes": total, "onDisk": self.swapOnDisk.count, "diskBytes": diskBytes,
                   "availableMiB": MemoryProbe.availableBytes() / 1048576], nil)
        }

        handlers["load_bundle_file"] = { args, reply in
            guard let name = args["name"] as? String else {
                reply(["error": "missing name"], nil); return
            }
            if let url = Bundle.main.url(forResource: name, withExtension: nil),
               let data = try? Data(contentsOf: url) {
                reply(["data": data.base64EncodedString()], nil)
            } else {
                reply(["error": "not found: \(name)"], nil)
            }
        }

        // ── Filesystem bridge ──

        handlers["fs_open"] = { args, reply in
            guard let path = args["path"] as? String,
                  let flags = args["flags"] as? Int,
                  let mode = args["mode"] as? Int else {
                reply(["ret": -22], nil); return
            }
            // truncatingIfNeeded, not Int32(_:): the guest can pass a 64-bit value with junk
            // high bits (e.g. open() without O_CREAT leaves mode uninitialized), and Int32(_:)
            // would trap on overflow, taking down the whole app.
            let creds = parseCreds(args)
            var didCreate = false
            // openat2's RESOLVE_NO_SYMLINKS: the guest kernel asks the filesystem to refuse a
            // path with a link anywhere in it, because only the filesystem can see the links.
            let noSymlinks = (args["noSymlinks"] as? Int ?? 0) != 0
            let fd = fs.open(path: path, flags: Int32(truncatingIfNeeded: flags), mode: Int32(truncatingIfNeeded: mode), creds: creds, didCreate: &didCreate, noSymlinks: noSymlinks)
            // ENODEV from open means "this is a device node" -- the filesystem holds it, the guest
            // kernel owns the driver. only then is the extra lookup for its identity worth doing.
            if fd == -19, let dev = fs.deviceAt(path: path, creds: creds) {
                reply(["ret": -19, "mode": Int(dev.mode), "rdev": Int(dev.rdev)], nil); return
            }
            // `created` is what tells an IN_CREATE from an IN_OPEN of a file that was already there,
            // and `mode` is how the guest learns it opened a directory without a second round trip.
            let openedMode = fd >= 0 ? (fs.fstat(fd: fd)["mode"] as? Int ?? 0) : 0
            reply(["ret": fd, "created": didCreate ? 1 : 0, "mode": openedMode], nil)
        }

        handlers["fs_close"] = { args, reply in
            guard let fd = args["fd"] as? Int else {
                reply(["ret": -9], nil); return
            }
            reply(["ret": fs.close(fd: Int32(truncatingIfNeeded: fd))], nil)
        }

        handlers["fs_read"] = { args, reply in
            guard let fd = args["fd"] as? Int,
                  let count = args["count"] as? Int else {
                reply(["ret": -9], nil); return
            }
            let (ret, data) = fs.read(fd: Int32(truncatingIfNeeded: fd), count: count)
            if let data = data {
                reply(["ret": ret, "data": data.base64EncodedString()], nil)
            } else {
                reply(["ret": ret], nil)
            }
        }

        // read a whole file from the ACTIVE root (symlinks followed) — used to boot the root's login
        // shell as pid 1 so `jitter switch` boots the chosen distro.
        handlers["load_root_file"] = { args, reply in
            guard let path = args["path"] as? String else { reply(["error": "no path"], nil); return }
            if let url = fs.resolve(path), let data = try? Data(contentsOf: url) {
                reply(["data": data.base64EncodedString()], nil)
            } else {
                reply(["error": "cannot read \(path)"], nil)
            }
        }

        // pull a distro tarball into roots/<name>. async download; installRootfs does gunzip + untar.
        handlers["jitter_pull"] = { args, reply in
            guard let name = args["name"] as? String, let urlStr = args["url"] as? String,
                  let url = URL(string: urlStr) else { reply(["ret": -22, "output": "usage: name url"], nil); return }
            URLSession.shared.dataTask(with: url) { data, _, err in
                if let err = err { reply(["ret": -1, "output": "download failed: \(err.localizedDescription)"], nil); return }
                guard let data = data, !data.isEmpty else { reply(["ret": -1, "output": "empty download"], nil); return }
                let r = fs.installRootfs(data, into: name)
                reply(["ret": r.ok ? 0 : -1, "output": "\(r.note): \(r.files) files -> roots/\(name)"], nil)
            }.resume()
        }

        handlers["jitter_ls"] = { _, reply in
            let fsRef = fs
            reply(["roots": fsRef.listRoots(), "active": fsRef.activeRoot], nil)
        }

        handlers["fs_pread"] = { args, reply in
            guard let fd = args["fd"] as? Int,
                  let offset = args["offset"] as? Int,
                  let count = args["count"] as? Int else {
                reply(["ret": -9], nil); return
            }
            let (ret, data) = fs.pread(fd: Int32(truncatingIfNeeded: fd), offset: UInt64(max(0, offset)), count: count)
            if let data = data {
                reply(["ret": ret, "data": data.base64EncodedString()], nil)
            } else {
                reply(["ret": ret], nil)
            }
        }

        handlers["fs_write"] = { args, reply in
            guard let fd = args["fd"] as? Int,
                  let b64 = args["data"] as? String,
                  let data = Data(base64Encoded: b64) else {
                reply(["ret": -22], nil); return
            }
            let wLimit = (args["limit"] as? Int).map { Int64($0) } ?? -1
            reply(["ret": fs.write(fd: Int32(truncatingIfNeeded: fd), data: data, limit: wLimit,
                                   noGrow: ((args["noGrow"] as? Int) ?? 0) != 0)], nil)
        }

        handlers["fs_pwrite"] = { args, reply in
            guard let fd = args["fd"] as? Int,
                  let offset = args["offset"] as? Int,
                  let b64 = args["data"] as? String,
                  let data = Data(base64Encoded: b64) else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.pwrite(fd: Int32(truncatingIfNeeded: fd), offset: UInt64(max(0, offset)),
                                    data: data, noGrow: ((args["noGrow"] as? Int) ?? 0) != 0)], nil)
        }

        handlers["fs_stat"] = { args, reply in
            guard let path = args["path"] as? String else {
                reply(["ret": -22], nil); return
            }
            reply(fs.stat(path: path, creds: parseCreds(args)), nil)
        }

        handlers["fs_lstat"] = { args, reply in
            guard let path = args["path"] as? String else {
                reply(["ret": -22], nil); return
            }
            reply(fs.lstat(path: path, creds: parseCreds(args)), nil)
        }

        handlers["fs_lseek"] = { args, reply in
            guard let fd = args["fd"] as? Int,
                  let offset = args["offset"] as? Int,
                  let whence = args["whence"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.lseek(fd: Int32(truncatingIfNeeded: fd), offset: Int64(truncatingIfNeeded: offset), whence: Int32(truncatingIfNeeded: whence))], nil)
        }

        handlers["fs_getdents"] = { args, reply in
            guard let fd = args["fd"] as? Int else {
                reply(["ret": -9], nil); return
            }
            let (ret, entries) = fs.getdents(fd: Int32(truncatingIfNeeded: fd))
            reply(["ret": ret, "entries": entries], nil)
        }

        handlers["fs_mkdir"] = { args, reply in
            guard let path = args["path"] as? String,
                  let mode = args["mode"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.mkdir(path: path, mode: Int32(truncatingIfNeeded: mode), creds: parseCreds(args))], nil)
        }

        handlers["fs_unlink"] = { args, reply in
            guard let path = args["path"] as? String else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.unlink(path: path, removeDir: (args["removeDir"] as? Bool) ?? false,
                                    creds: parseCreds(args))], nil)
        }

        handlers["fs_rename"] = { args, reply in
            guard let oldPath = args["oldPath"] as? String,
                  let newPath = args["newPath"] as? String else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.rename(oldPath: oldPath, newPath: newPath,
                                    flags: args["flags"] as? Int ?? 0, creds: parseCreds(args))], nil)
        }

        handlers["fs_symlink"] = { args, reply in
            guard let target = args["target"] as? String,
                  let linkpath = args["linkpath"] as? String else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.symlink(target: target, linkpath: linkpath, creds: parseCreds(args))], nil)
        }

        // the /dev/jitter control device. the guest's `jitter` shim writes a request here and reads
        // [int32 status][body]; all policy (download, extract, switch) runs host-side, right here.
        handlers["jitter_ctl"] = { [weak self] args, reply in
            let cmd = (args["cmd"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            let (status, output) = self?.runJitterCommand(cmd) ?? (1, "jitter: control channel unavailable\n")
            // the generic bridge shape: `ret` is the command's exit status, `data` is base64 stdout.
            // (a worker's response travels through the shared-memory relay, which only carries those.)
            reply(["ret": status, "data": Data(output.utf8).base64EncodedString()], nil)
        }

        handlers["fs_mknod"] = { args, reply in
            guard let path = args["path"] as? String, let mode = args["mode"] as? Int else {
                reply(["ret": -22], nil); return
            }
            let dev = (args["dev"] as? Int) ?? 0
            reply(["ret": fs.mknod(path: path, mode: Int32(truncatingIfNeeded: mode),
                                   dev: Int32(truncatingIfNeeded: dev), creds: parseCreds(args))], nil)
        }

        handlers["fs_mksock"] = { args, reply in
            guard let path = args["path"] as? String, let mode = args["mode"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.mksock(path: path, mode: Int32(truncatingIfNeeded: mode),
                                    creds: parseCreds(args))], nil)
        }

        handlers["fs_link"] = { args, reply in
            guard let oldPath = args["oldPath"] as? String,
                  let newPath = args["newPath"] as? String else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.link(oldPath: oldPath, newPath: newPath, creds: parseCreds(args))], nil)
        }

        // give an unnamed (O_TMPFILE) descriptor its name. the guest reached here through
        // linkat("/proc/self/fd/N", ...), which is how linux materialises a file that was written
        // before it had any name at all.
        handlers["fs_link_fd"] = { args, reply in
            guard let fd = args["fd"] as? Int, let newPath = args["newPath"] as? String else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.linkTmpfile(fd: Int32(fd), newPath: newPath, creds: parseCreds(args))], nil)
        }

        handlers["fs_readlink"] = { args, reply in
            guard let path = args["path"] as? String else {
                reply(["ret": -22], nil); return
            }
            let (ret, target) = fs.readlink(path: path, creds: parseCreds(args))
            if let target = target {
                reply(["ret": ret, "target": target], nil)
            } else {
                reply(["ret": ret], nil)
            }
        }

        handlers["fs_fstat"] = { args, reply in
            guard let fd = args["fd"] as? Int else {
                reply(["ret": -9], nil); return
            }
            reply(fs.fstat(fd: Int32(truncatingIfNeeded: fd)), nil)
        }

        handlers["fs_fallocate"] = { args, reply in
            guard let fd = args["fd"] as? Int, let size = args["size"] as? Int, size >= 0 else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.fallocate(fd: Int32(fd), size: UInt64(size),
                                       noGrow: ((args["noGrow"] as? Int) ?? 0) != 0)], nil)
        }

        handlers["fs_fsync"] = { args, reply in
            guard let fd = args["fd"] as? Int else { reply(["ret": -9], nil); return }
            reply(["ret": fs.fsync(fd: Int32(fd), dataOnly: (args["data"] as? Bool) ?? false)], nil)
        }

        handlers["fs_truncate"] = { args, reply in
            guard let path = args["path"] as? String, let length = args["length"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.truncate(path: path, length: Int64(length), creds: parseCreds(args))], nil)
        }

        handlers["fs_ftruncate"] = { args, reply in
            guard let fd = args["fd"] as? Int,
                  let length = args["length"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.ftruncate(fd: Int32(truncatingIfNeeded: fd), length: Int64(truncatingIfNeeded: length),
                                       noShrink: (args["noShrink"] as? Int) == 1,
                                       noGrow: (args["noGrow"] as? Int) == 1)], nil)
        }

        handlers["fs_dup"] = { args, reply in
            guard let fd = args["fd"] as? Int else {
                reply(["ret": -9], nil); return
            }
            reply(["ret": fs.dup(fd: Int32(truncatingIfNeeded: fd))], nil)
        }

        handlers["fs_punch_hole"] = { args, reply in
            guard let fd = args["fd"] as? Int, let offset = args["offset"] as? Int,
                  let length = args["length"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": Int(fs.punchHole(fd: Int32(truncatingIfNeeded: fd),
                                           offset: Int64(offset), length: Int64(length)))], nil)
        }

        handlers["fs_reopen"] = { args, reply in
            guard let fd = args["fd"] as? Int, let flags = args["flags"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": Int(fs.reopen(fd: Int32(truncatingIfNeeded: fd),
                                        flags: Int32(truncatingIfNeeded: flags),
                                        noShrink: ((args["noShrink"] as? Int) ?? 0) != 0))], nil)
        }

        handlers["fs_faccessat"] = { args, reply in
            guard let path = args["path"] as? String,
                  let mode = args["mode"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.faccessat(path: path, mode: Int32(truncatingIfNeeded: mode),
                                       creds: parseCreds(args),
                                       eaccess: (args["eaccess"] as? Bool) ?? false)], nil)
        }

        handlers["fs_chmod"] = { args, reply in
            guard let path = args["path"] as? String,
                  let mode = args["mode"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.chmod(path: path, mode: Int32(truncatingIfNeeded: mode), creds: parseCreds(args))], nil)
        }

        handlers["fs_chown"] = { args, reply in
            guard let path = args["path"] as? String,
                  let uid = args["uid"] as? Int,
                  let gid = args["gid"] as? Int else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": fs.chown(path: path,
                                   uid: Int32(truncatingIfNeeded: uid),
                                   gid: Int32(truncatingIfNeeded: gid),
                                   creds: parseCreds(args),
                                   nofollow: (args["nofollow"] as? Bool) ?? false)], nil)
        }

        handlers["fs_utimens"] = { args, reply in
            guard let path = args["path"] as? String,
                  let t = args["times"] as? [Int], t.count == 4 else {
                reply(["ret": -22], nil); return
            }
            let times = [timespec(tv_sec: t[0], tv_nsec: t[1]), timespec(tv_sec: t[2], tv_nsec: t[3])]
            reply(["ret": fs.utimens(path: path, times: times, creds: parseCreds(args),
                                     nofollow: (args["nofollow"] as? Bool) ?? false)], nil)
        }

        handlers["fs_getxattr"] = { args, reply in
            guard let path = args["path"] as? String, let name = args["name"] as? String else {
                reply(["ret": -22], nil); return
            }
            let (rc, data) = fs.getXattr(path: path, name: name,
                                         nofollow: (args["nofollow"] as? Bool) ?? false,
                                         creds: parseCreds(args))
            reply(["ret": Int(rc), "data": data?.base64EncodedString() ?? ""], nil)
        }

        handlers["fs_setxattr"] = { args, reply in
            guard let path = args["path"] as? String, let name = args["name"] as? String,
                  let b64 = args["value"] as? String, let flags = args["flags"] as? Int else {
                reply(["ret": -22], nil); return
            }
            let value = Data(base64Encoded: b64) ?? Data()
            reply(["ret": Int(fs.setXattr(path: path, name: name, value: value,
                                          flags: Int32(truncatingIfNeeded: flags),
                                          nofollow: (args["nofollow"] as? Bool) ?? false,
                                          creds: parseCreds(args)))], nil)
        }

        handlers["fs_removexattr"] = { args, reply in
            guard let path = args["path"] as? String, let name = args["name"] as? String else {
                reply(["ret": -22], nil); return
            }
            reply(["ret": Int(fs.removeXattr(path: path, name: name,
                                             nofollow: (args["nofollow"] as? Bool) ?? false,
                                             creds: parseCreds(args)))], nil)
        }

        handlers["fs_listxattr"] = { args, reply in
            guard let path = args["path"] as? String else { reply(["ret": -22], nil); return }
            let (rc, data) = fs.listXattr(path: path,
                                          nofollow: (args["nofollow"] as? Bool) ?? false,
                                          creds: parseCreds(args))
            reply(["ret": Int(rc), "data": data?.base64EncodedString() ?? ""], nil)
        }

        handlers["term_size_query"] = { [weak self] _, reply in
            reply(["cols": self?.lastTermCols ?? 0, "rows": self?.lastTermRows ?? 0], nil)
        }

        handlers["term_data"] = { [weak self] args, reply in
            if let b64 = args["data"] as? String,
               let data = Data(base64Encoded: b64) {
                self?.ctlAppend(data) // tee for the debug control channel
                if let cb = self?.onTerminalData {
                    DispatchQueue.main.async { cb(data) }
                }
            }
            reply([:], nil)
        }
    }

    func sendStdin(_ data: Data) {
        guard !data.isEmpty else { return }
        let b64 = data.base64EncodedString()
        let js = "window.__jitterStdin && window.__jitterStdin('\(b64)')"
        DispatchQueue.main.async { [weak self] in
            self?.webView?.evaluateJavaScript(js, completionHandler: nil)
        }
    }

    // ── the `jitter` command (see /dev/jitter above) ─────────────────────────────────────────────
    // one entry point so the guest shim stays trivial and every policy decision is host-side. returns
    // (exit status, text for stdout).
    func runJitterCommand(_ cmd: String) -> (Int, String) {
        let words = cmd.split(separator: " ").map(String.init)
        guard let verb = words.first else { return (2, jitterUsage) }
        let rest = Array(words.dropFirst())
        guard let fs = nativeFS else { return (1, "jitter: no filesystem\n") }

        switch verb {
        case "help", "-h", "--help":
            return (0, jitterUsage)

        case "ls":
            let active = fs.activeRoot
            let names = fs.listRoots().sorted()
            if names.isEmpty { return (0, "no roots installed\n") }
            var out = ""
            for n in names { out += (n == active ? "* " : "  ") + n + "\n" }
            return (0, out)

        case "pull":
            guard let ref = rest.first else { return (2, "usage: jitter pull <name|url> [as]\n") }
            let name = rest.count > 1 ? rest[1] : defaultNameFor(ref)
            guard let url = resolveSource(ref) else {
                return (1, "jitter: don't know how to pull '\(ref)'. known: \(catalog.map(\.name).joined(separator: ", "))\n")
            }
            let r = performPull(name, url)
            // a freshly installed root has no shim; put one in so `jitter` works after switching.
            if fs.rootExists(name) { installShim(into: name) }
            return (r.hasPrefix("ok") ? 0 : 1, r + "\n")

        case "switch":
            guard let name = rest.first else { return (2, "usage: jitter switch <name>\n") }
            guard fs.rootExists(name) else { return (1, "jitter: no such root '\(name)'\n") }
            installShim(into: name)
            guard fs.switchRoot(name) else { return (1, "jitter: could not switch to '\(name)'\n") }
            // reboot the session into the new root, but only after this reply has made it back to the
            // guest and been printed — resetting inline tears the webview down mid-write and the user
            // never sees the confirmation.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in self?.ctlReset() }
            return (0, "switched to \(name)\n")

        case "rm":
            guard let name = rest.first else { return (2, "usage: jitter rm <name>\n") }
            if name == fs.activeRoot { return (1, "jitter: '\(name)' is the active root\n") }
            return fs.removeRoot(name) ? (0, "removed \(name)\n") : (1, "jitter: could not remove '\(name)'\n")

        case "default":
            guard let name = rest.first else { return (2, "usage: jitter default <name>\n") }
            guard fs.rootExists(name) else { return (1, "jitter: no such root '\(name)'\n") }
            installShim(into: name)
            _ = fs.switchRoot(name)
            return (0, "default root is now \(name)\n")

        default:
            return (2, "jitter: unknown command '\(verb)'\n\n" + jitterUsage)
        }
    }

    private var jitterUsage: String {
        """
        usage: jitter <command> [args]

          pull <name|url> [as]   download a root and install it
          switch <name>          restart the session in <name>
          ls                     list installed roots
          rm <name>              delete a root
          default <name>         set the root to boot into
          help                   this text

        known sources: \(catalog.map(\.name).joined(separator: ", "))
        """ + "\n"
    }

    /// the source catalog. manual entries only for now; the config-file form (and later, catalog feeds
    /// and OCI registries) slots in here without changing anything above.
    private struct Source { let name: String; let description: String; let url: String }
    private var catalog: [Source] {
        [
            // pulls run natively through URLSession (real TLS, and iOS ATS requires https anyway), so
            // the canonical CDN is fine here — the truncation seen from the guest's own socket stack
            // affects guest-side http only.
            Source(name: "alpine",
                   description: "Alpine Linux 3.20 aarch64 minirootfs",
                   url: "https://dl-cdn.alpinelinux.org/alpine/v3.20/releases/aarch64/alpine-minirootfs-3.20.3-aarch64.tar.gz"),
        ]
    }

    private func resolveSource(_ ref: String) -> String? {
        if let hit = catalog.first(where: { $0.name == ref }) { return hit.url }
        if ref.hasPrefix("http://") || ref.hasPrefix("https://") { return ref }
        return nil
    }

    private func defaultNameFor(_ ref: String) -> String {
        if catalog.contains(where: { $0.name == ref }) { return ref }
        // url -> last path component without extensions, else "root"
        let base = URL(string: ref)?.lastPathComponent ?? ref
        let stem = base.split(separator: ".").first.map(String.init) ?? "root"
        return stem.isEmpty ? "root" : stem
    }

    /// make a root usable: the `jitter` overlay plus the bits a distro image doesn't ship because it
    /// expects its runtime to supply them. everything here is additive — a distro's own files are never
    /// modified, and anything it DOES provide is left alone.
    private func installShim(into rootName: String) {
        guard let fs = nativeFS else { return }
        let root = fs.rootURL(rootName)
        let fm = FileManager.default

        // the `jitter` command itself
        if let src = Bundle.main.url(forResource: "jitter-shim", withExtension: nil),
           let data = try? Data(contentsOf: src) {
            let binDir = root.appendingPathComponent("bin", isDirectory: true)
            try? fm.createDirectory(at: binDir, withIntermediateDirectories: true)
            let dst = binDir.appendingPathComponent("jitter")
            try? data.write(to: dst)
            try? fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dst.path)
        }

        // resolver config. a minirootfs ships no /etc/resolv.conf (the runtime is expected to inject
        // one, exactly as a container engine does), so without this NOTHING in a pulled root can
        // resolve a name — apk included. only written when the root doesn't already have one.
        let etc = root.appendingPathComponent("etc", isDirectory: true)
        try? fm.createDirectory(at: etc, withIntermediateDirectories: true)
        let resolv = etc.appendingPathComponent("resolv.conf")
        if !fm.fileExists(atPath: resolv.path) {
            try? "nameserver 1.1.1.1\nnameserver 8.8.8.8\n".write(to: resolv, atomically: true, encoding: .utf8)
        }
        let hosts = etc.appendingPathComponent("hosts")
        if !fm.fileExists(atPath: hosts.path) {
            try? "127.0.0.1 localhost\n::1 localhost\n".write(to: hosts, atomically: true, encoding: .utf8)
        }
    }

    // evaluate javascript in the page and return its result as text. debug channel only (same class as
    // runBinary): the way to measure what the engine/OS will actually give us — wasm memory limits,
    // allocator behaviour — from outside the guest.
    func evalJS(_ src: String, timeoutMs: Int) -> String {
        let sem = DispatchSemaphore(value: 0)
        var result = "[ctl/js] no response"
        DispatchQueue.main.async { [weak self] in
            guard let wv = self?.webView else { result = "[ctl/js] no webview"; sem.signal(); return }
            wv.callAsyncJavaScript(src, arguments: [:], in: nil, in: .page) { r in
                switch r {
                case .success(let v): result = String(describing: v)
                case .failure(let e): result = "[ctl/js] error: \(e.localizedDescription)"
                }
                sem.signal()
            }
        }
        if sem.wait(timeout: .now() + .milliseconds(timeoutMs)) == .timedOut { return "[ctl/js] timed out\n" }
        return result + "\n"
    }

    // run ONE guest binary as its own process and block until it exits, returning its output + status.
    // goes straight to the kernel (window.__jitterRun), so it needs no working shell and never types
    // into the tty — which makes it both the way to debug a root whose init won't load and a driver
    // that can't lose characters the way stdin injection into a busy shell can.
    func runBinary(_ path: String, args: [String], timeoutMs: Int) -> String {
        let argJSON = (try? JSONSerialization.data(withJSONObject: args))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
        let esc = path.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "'", with: "\\'")
        // callAsyncJavaScript runs this as an async function BODY, so it must `return` directly
        // (an IIFE expression would evaluate and be discarded).
        let js = """
        try {
          if (!window.__jitterRun) return JSON.stringify({output: '[ctl/run] kernel not ready', code: -1});
          const r = await window.__jitterRun('\(esc)', \(argJSON), \(timeoutMs));
          return JSON.stringify(r);
        } catch (e) {
          return JSON.stringify({output: '[ctl/run] ' + e, code: -1});
        }
        """
        let sem = DispatchSemaphore(value: 0)
        var result = "[ctl/run] no response"
        DispatchQueue.main.async { [weak self] in
            guard let wv = self?.webView else { result = "[ctl/run] no webview"; sem.signal(); return }
            wv.callAsyncJavaScript(js, arguments: [:], in: nil, in: .page) { r in
                switch r {
                case .success(let v): result = (v as? String) ?? "[ctl/run] bad result"
                case .failure(let e): result = "[ctl/run] js error: \(e.localizedDescription)"
                }
                sem.signal()
            }
        }
        // the js side has its own timeout and always resolves; this is just a backstop so a wedged
        // webview can't hold the http connection open forever.
        if sem.wait(timeout: .now() + .milliseconds(timeoutMs + 5000)) == .timedOut {
            return "[ctl/run] timed out waiting for the webview\n"
        }
        // unwrap {output, code} into plain text + the exit line /ctl/exec callers already expect.
        if let d = result.data(using: .utf8),
           let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
            let out = o["output"] as? String ?? ""
            let code = o["code"] as? Int ?? -1
            return out + "[exit \(code)]\n"
        }
        return result
    }

    func sendTermSize(cols: Int, rows: Int) {
        lastTermCols = cols
        lastTermRows = rows
        let js = "window.__jitterTermSize && window.__jitterTermSize(\(cols), \(rows))"
        DispatchQueue.main.async { [weak self] in
            self?.webView?.evaluateJavaScript(js, completionHandler: nil)
        }
    }

    // MARK: - WKScriptMessageHandlerWithReply

    func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage,
        replyHandler: @escaping (Any?, String?) -> Void
    ) {
        guard let body = message.body as? [String: Any],
              let method = body["method"] as? String else {
            replyHandler(nil, "invalid message")
            return
        }

        let args = body["args"] as? [String: Any] ?? [:]

        // stamp who is asking, so a descriptor opened during this call can be reclaimed later if
        // the process is force-terminated and never gets to close it itself
        nativeFS?.currentOwnerPid = (args["_owner"] as? Int) ?? 0
        defer { nativeFS?.currentOwnerPid = 0 }

        if let handler = handlers[method] {
            handler(args, replyHandler)
        } else {
            replyHandler(nil, "unknown method: \(method)")
        }
    }
}
