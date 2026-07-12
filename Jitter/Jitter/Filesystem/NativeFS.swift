import Foundation
import os.lock

// File-scope wrappers for POSIX stat/lstat to disambiguate from NativeFS.stat method
private func statCWrapper(_ path: UnsafePointer<CChar>, _ sb: UnsafeMutablePointer<stat>) -> Int32 {
    return stat(path, sb)
}
private func lstatCWrapper(_ path: UnsafePointer<CChar>, _ sb: UnsafeMutablePointer<stat>) -> Int32 {
    return lstat(path, sb)
}

// Int(Double) traps on NaN/infinite; real files never have those, but guard anyway so a
// corrupt mtime can't take down the whole app.
private func safeInt(_ d: Double) -> Int { d.isFinite ? Int(d) : 0 }

// Emulated ownership metadata stored per inode as a 12-byte xattr. `mode` is the full emulated
// mode (type + suid/sgid/sticky + perm bits); the type bits are always re-derived from the real
// file so a stale/foreign xattr can't misreport the file kind.
struct FileMeta { var uid: UInt32; var gid: UInt32; var mode: UInt32 }

// The calling guest process's credentials, passed with every path-resolving bridge op (NativeFS
// is shared across all worker processes, each with its own creds, so they can't be held globally).
struct FsCreds {
    var ruid: UInt32; var euid: UInt32; var rgid: UInt32; var egid: UInt32
    var umask: UInt32; var groups: [UInt32]
    static let root = FsCreds(ruid: 0, euid: 0, rgid: 0, egid: 0, umask: 0o022, groups: [])
}

/// Native filesystem backed by the app's Documents/rootfs/ directory.
/// Maps guest Linux paths to real paths on disk.
/// Serves as the single source of truth for all real file I/O — shared across all Web Workers.
///
/// Guest path "/root/test.txt" → <Documents>/rootfs/root/test.txt
final class NativeFS {
    let rootPath: URL
    private let fm = FileManager.default
    private var openFiles: [Int32: FileHandle] = [:]
    private var openFilePaths: [Int32: String] = [:] // fd -> real path, for fstat ownership lookup
    private var openDirs: [Int32: DirState] = [:]
    private var nextFd: Int32 = 100

    // Read cache (LRU, configurable size, lives outside WebView memory budget)
    private var cache = ReadCache()

    // Emulated POSIX ownership/permissions. iOS won't let us chown the real files and their
    // real uid/gid is meaningless to the guest, so we store the emulated {uid,gid,mode} as a
    // 12-byte xattr per inode (authoritative), enforce access here during the native path walk,
    // and never rely on the real file's owner. metaCache memoises path -> meta (write-through:
    // invalidated on any op that mutates ownership/existence). see memory/ltp_grind_status.md.
    private var metaCache: [String: FileMeta] = [:]

    // Linux errno values
    private let ENOENT: Int32 = 2
    private let EEXIST: Int32 = 17
    private let EBADF: Int32 = 9
    private let EINVAL: Int32 = 22
    private let EISDIR: Int32 = 21
    private let ENOTDIR: Int32 = 20
    private let ENOTEMPTY: Int32 = 39
    private let EPERM: Int32 = 1

    // Linux open flags
    private let O_RDONLY: Int32 = 0
    private let O_WRONLY: Int32 = 1
    private let O_RDWR: Int32 = 2
    private let O_CREAT: Int32 = 64
    private let O_EXCL: Int32 = 128
    private let O_TRUNC: Int32 = 512
    private let O_APPEND: Int32 = 1024
    private let O_DIRECTORY: Int32 = 0x10000
    private let O_NOFOLLOW: Int32 = 0x20000

    // Linux seek whence
    private let SEEK_SET: Int32 = 0
    private let SEEK_CUR: Int32 = 1
    private let SEEK_END: Int32 = 2

    // Current bootstrap version — bump when busybox or default configs change
    static let bootstrapVersion = 9

    // Fallback applet set if busybox-applets.txt isn't bundled (the real, complete list is
    // loaded from that resource at bootstrap). keep this small but bootable.
    static let fallbackApplets = [
        "sh", "ash", "ls", "cat", "echo", "grep", "find", "wc", "head", "tail",
        "sort", "uniq", "sed", "awk", "cut", "tr", "tee", "xargs",
        "cp", "mv", "rm", "mkdir", "rmdir", "ln", "chmod", "chown", "touch",
        "pwd", "env", "test", "[", "true", "false", "yes", "seq", "expr",
        "date", "sleep", "uname", "hostname", "whoami", "id", "which",
        "du", "df", "free", "uptime", "ps", "kill", "basename", "dirname",
        "readlink", "realpath", "stat", "md5sum", "sha256sum",
        "dd", "hexdump", "od", "strings", "diff", "patch",
        "tar", "gzip", "gunzip", "zcat", "bzip2", "bunzip2",
        "vi", "less", "more", "clear",
        "wget", "nc", "ping", "nslookup", "traceroute",
        "printf", "type",
    ]

    private struct DirState {
        let url: URL
        var entries: [String]?
        var position: Int
    }

    init() {
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first!
        rootPath = docs.appendingPathComponent("rootfs", isDirectory: true)
        try? fm.createDirectory(at: rootPath, withIntermediateDirectories: true)
        bootstrapIfNeeded()
        loadCacheConfig()
    }

    // MARK: - Bootstrap

    /// On first launch (or after app update), create the rootfs directory tree
    /// and populate it with busybox + default configs.
    private func bootstrapIfNeeded() {
        let stateURL = rootPath
            .appendingPathComponent("etc/jitter", isDirectory: true)

        // Read current state
        let stateFile = stateURL.appendingPathComponent("state.json")
        var currentVersion = 0
        if let data = try? Data(contentsOf: stateFile),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let v = json["bootstrapVersion"] as? Int {
            currentVersion = v
        }

        guard currentVersion < Self.bootstrapVersion else { return }

        print("[NativeFS] Bootstrapping rootfs (v\(currentVersion) → v\(Self.bootstrapVersion))...")

        // Create directory tree
        let dirs = [
            "bin", "etc", "etc/jitter", "tmp", "root", "home", "var",
            "usr", "usr/bin", "usr/sbin", "sbin", "opt", "opt/ltp",
        ]
        for dir in dirs {
            let url = rootPath.appendingPathComponent(dir, isDirectory: true)
            try? fm.createDirectory(at: url, withIntermediateDirectories: true)
        }

        // Copy busybox from app bundle
        if let bundleURL = Bundle.main.url(forResource: "busybox-aarch64", withExtension: nil) {
            let dest = rootPath.appendingPathComponent("bin/busybox")
            try? fm.removeItem(at: dest)
            do {
                try fm.copyItem(at: bundleURL, to: dest)
                try fm.setAttributes([.posixPermissions: NSNumber(value: 0o755)], ofItemAtPath: dest.path)
                print("[NativeFS] Copied busybox to \(dest.path)")
            } catch {
                print("[NativeFS] ERROR copying busybox: \(error)")
            }
        } else {
            print("[NativeFS] WARNING: busybox-aarch64 not found in bundle")
        }

        // Create a symlink for every busybox applet. the list is the exact + complete set
        // this busybox provides, bundled as busybox-applets.txt (generated from the binary via
        // `busybox --list`; see tools/bb-list.ts). fall back to a minimal set if it's missing.
        let busyboxDest = "busybox" // relative symlink within same directory
        var applets = Self.fallbackApplets
        if let listURL = Bundle.main.url(forResource: "busybox-applets", withExtension: "txt"),
           let text = try? String(contentsOf: listURL, encoding: .utf8) {
            let parsed = text.split(whereSeparator: \.isNewline).map(String.init).filter { !$0.isEmpty }
            if !parsed.isEmpty { applets = parsed }
        }
        var symlinkCount = 0
        for applet in applets {
            for dir in ["bin", "usr/bin"] {
                let link = rootPath.appendingPathComponent("\(dir)/\(applet)")
                try? fm.removeItem(at: link)
                do {
                    try fm.createSymbolicLink(atPath: link.path, withDestinationPath: busyboxDest)
                    symlinkCount += 1
                } catch {
                    print("[NativeFS] ERROR creating symlink \(applet): \(error)")
                }
            }
        }
        print("[NativeFS] Created \(symlinkCount) applet symlinks (\(applets.count) applets)")

        // Seed the ltp suite tarball (the /bin/ltp driver extracts it with guest tar on
        // first run) + write the driver. this is the in-vivo ltp runner: each test is a
        // real fork+exec through the worker-kernel, bounded by busybox timeout.
        // extract the ltp binaries natively (direct disk I/O). doing it guest-side would
        // stream 100MB+ through the FS bridge per read -- far too slow/fragile.
        let ltpBin = rootPath.appendingPathComponent("opt/ltp/testcases/bin")
        try? fm.removeItem(at: rootPath.appendingPathComponent("opt/ltp/ltp.tar")) // remove stale seed
        try? fm.removeItem(at: rootPath.appendingPathComponent("opt/ltp/ltp.tgz"))
        if let ltpURL = Bundle.main.url(forResource: "ltp", withExtension: "tar") {
            let n = extractTar(ltpURL, to: ltpBin)
            print("[NativeFS] extracted \(n) ltp binaries")
        } else {
            print("[NativeFS] note: ltp.tar not bundled (ltp runner will be absent)")
        }
        writeText("bin/ltp", """
        #!/bin/sh
        # jitter in-vivo ltp runner: exec each test under busybox timeout (real fork+exec)
        # and tally by ltp exit code. ctrl+c aborts. suite is pre-extracted at bootstrap.
        BIN=/opt/ltp/testcases/bin
        T=${LTP_TIMEOUT:-20}
        if [ ! -d "$BIN" ]; then echo "ltp: suite not installed"; exit 1; fi
        pass=0 fail=0 brok=0 conf=0 to=0 n=0
        for t in "$BIN"/*; do
          [ -f "$t" ] || continue
          name=${t##*/}
          n=$((n+1))
          timeout -k 5 "$T" "$t" >/dev/null 2>&1
          rc=$?
          case $rc in
            0)       pass=$((pass+1)); st=PASS ;;
            1)       fail=$((fail+1)); st=FAIL ;;
            2)       brok=$((brok+1)); st=BROK ;;
            32)      conf=$((conf+1)); st=SKIP ;;
            137|143) to=$((to+1));     st=TIMEOUT ;;
            *)       fail=$((fail+1)); st=EXIT$rc ;;
          esac
          printf '%-42s %s\\n' "$name" "$st"
        done
        echo "------------------------------------------------"
        echo "ltp: $n tests | pass $pass  fail $fail  broke $brok  skip $conf  timeout $to"
        """)
        let ltpDriver = rootPath.appendingPathComponent("bin/ltp")
        try? fm.setAttributes([.posixPermissions: NSNumber(value: 0o755)], ofItemAtPath: ltpDriver.path)

        // Write default config files
        writeText("etc/passwd",
                  "root:x:0:0:root:/root:/bin/sh\n" +
                  "daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin\n" +
                  "nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n")
        writeText("etc/group", "root:x:0:\ndaemon:x:1:\nnobody:x:65534:\nnogroup:x:65534:\n")
        writeText("etc/hostname", "jitter\n")
        writeText("etc/shells", "/bin/sh\n")
        writeText("etc/resolv.conf", "nameserver 1.1.1.1\nnameserver 1.0.0.1\n")
        writeText("etc/nsswitch.conf", "hosts: files dns\n")

        // Write jitter config (only if doesn't exist — preserve user config)
        let configFile = stateURL.appendingPathComponent("config.json")
        if !fm.fileExists(atPath: configFile.path) {
            writeText("etc/jitter/config.json", "{\n  \"fsCacheMaxKBytes\": 131072\n}\n")
        }

        // Write state
        let state: [String: Any] = ["bootstrapVersion": Self.bootstrapVersion]
        if let data = try? JSONSerialization.data(withJSONObject: state, options: .prettyPrinted) {
            try? data.write(to: stateFile)
        }

        // Verify bootstrap
        print("[NativeFS] rootPath: \(rootPath.path)")
        if let binContents = try? fm.contentsOfDirectory(atPath: rootPath.appendingPathComponent("bin").path) {
            print("[NativeFS] /bin has \(binContents.count) entries: \(binContents.prefix(5).joined(separator: ", "))...")
        }
        if let etcContents = try? fm.contentsOfDirectory(atPath: rootPath.appendingPathComponent("etc").path) {
            print("[NativeFS] /etc has \(etcContents.count) entries: \(etcContents.joined(separator: ", "))")
        }
        print("[NativeFS] Bootstrap complete.")
    }

    /// Minimal ustar extractor: writes the regular files from a tar into destDir (flat, all
    /// chmod 755). Streams via a FileHandle so a 100MB+ archive is never fully resident.
    /// Handles the "./name" prefixes that `tar cf .` produces; skips dirs. Writes a
    /// diagnostic to <destDir>/../extract.log. Returns the count written.
    private func extractTar(_ tarURL: URL, to destDir: URL) -> Int {
        var log = "extractTar \(tarURL.lastPathComponent) -> \(destDir.path)\n"
        func finish(_ n: Int, _ note: String) -> Int {
            log += note + "\nwrote \(n) files\n"
            try? log.data(using: .utf8)?.write(to: destDir.deletingLastPathComponent().appendingPathComponent("extract.log"))
            return n
        }
        try? fm.removeItem(at: destDir)
        do { try fm.createDirectory(at: destDir, withIntermediateDirectories: true) }
        catch { return finish(0, "createDirectory failed: \(error)") }
        guard let fh = try? FileHandle(forReadingFrom: tarURL) else { return finish(0, "cannot open tar") }
        defer { try? fh.close() }

        var count = 0, firstErr = ""
        while true {
            guard let header = try? fh.read(upToCount: 512), header.count == 512 else { break }
            let name = String(decoding: header.prefix(100).prefix(while: { $0 != 0 }), as: UTF8.self)
            if name.isEmpty { break } // zero block => end of archive
            let sizeField = header.subdata(in: 124..<136)
            let sizeStr = String(decoding: sizeField.prefix(while: { $0 != 0 && $0 != 0x20 }), as: UTF8.self)
            let size = Int(sizeStr, radix: 8) ?? 0
            let typeflag = header[header.startIndex + 156]
            let fileData = size > 0 ? (try? fh.read(upToCount: size)) ?? Data() : Data()
            // skip the zero padding up to the next 512 boundary
            let pad = (512 - size % 512) % 512
            if pad > 0 { _ = try? fh.read(upToCount: pad) }

            if typeflag == 0x30 || typeflag == 0 { // '0' or NUL = regular file
                var base = name
                while base.hasPrefix("./") { base = String(base.dropFirst(2)) }
                if !base.isEmpty && !base.contains("/") && fileData.count > 0 {
                    let dest = destDir.appendingPathComponent(base)
                    do {
                        try fileData.write(to: dest)
                        try? fm.setAttributes([.posixPermissions: NSNumber(value: 0o755)], ofItemAtPath: dest.path)
                        count += 1
                    } catch {
                        if firstErr.isEmpty { firstErr = "write \(base): \(error)" }
                    }
                }
            }
        }
        return finish(count, firstErr.isEmpty ? "ok" : firstErr)
    }

    private func writeText(_ relativePath: String, _ content: String) {
        let url = rootPath.appendingPathComponent(relativePath)
        try? content.write(to: url, atomically: true, encoding: .utf8)
    }

    private func loadCacheConfig() {
        let configURL = rootPath.appendingPathComponent("etc/jitter/config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let maxKB = json["fsCacheMaxKBytes"] as? Int else { return }
        cache.maxBytes = maxKB * 1024
    }

    // MARK: - Path Resolution

    /// Map a guest Linux path to a real filesystem URL.
    func resolve(_ guestPath: String) -> URL {
        var parts: [String] = []
        for c in guestPath.split(separator: "/").map(String.init) {
            if c == "." { continue }
            if c == ".." { _ = parts.popLast(); continue }
            parts.append(c)
        }
        var url = rootPath
        for c in parts { url = url.appendingPathComponent(c) }
        return url
    }

    // MARK: - Emulated ownership/permissions (xattr-backed)

    private static let metaXattrName = "com.csutora.jitter.meta"

    private func readMetaXattr(_ path: String) -> FileMeta? {
        var buf = [UInt8](repeating: 0, count: 12)
        let n = buf.withUnsafeMutableBytes { getxattr(path, NativeFS.metaXattrName, $0.baseAddress, 12, 0, 0) }
        guard n == 12 else { return nil }
        func rd(_ o: Int) -> UInt32 { UInt32(buf[o]) | (UInt32(buf[o+1]) << 8) | (UInt32(buf[o+2]) << 16) | (UInt32(buf[o+3]) << 24) }
        return FileMeta(uid: rd(0), gid: rd(4), mode: rd(8))
    }

    // write the emulated metadata and refresh the cache (write-through).
    func writeMetaXattr(_ path: String, _ m: FileMeta) {
        var buf = [UInt8](repeating: 0, count: 12)
        func wr(_ v: UInt32, _ o: Int) { buf[o] = UInt8(v & 0xff); buf[o+1] = UInt8((v >> 8) & 0xff); buf[o+2] = UInt8((v >> 16) & 0xff); buf[o+3] = UInt8((v >> 24) & 0xff) }
        wr(m.uid, 0); wr(m.gid, 4); wr(m.mode, 8)
        _ = buf.withUnsafeBytes { setxattr(path, NativeFS.metaXattrName, $0.baseAddress, 12, 0, 0) }
        metaCache[path] = m
    }

    // owner (uid,gid) of a path for foreign-import inheritance: the nearest stamped ancestor,
    // defaulting to root (0,0) at/above the rootfs boundary.
    private func ownerOf(_ url: URL) -> (UInt32, UInt32) {
        if url.path.count <= rootPath.path.count { return (0, 0) }
        if let c = metaCache[url.path] { return (c.uid, c.gid) }
        if let x = readMetaXattr(url.path) { return (x.uid, x.gid) }
        return ownerOf(url.deletingLastPathComponent())
    }

    // resolve the emulated {uid,gid,mode} for a real path. xattr is authoritative; a missing
    // xattr means a foreign import -> inherit the parent dir's owner + the real file's mode. type
    // bits always come from `realMode` (reality) so a stale xattr type can't lie.
    private func effectiveMeta(_ url: URL, realMode: UInt32) -> FileMeta {
        let typeBits = realMode & 0o170000
        if let c = metaCache[url.path] { return FileMeta(uid: c.uid, gid: c.gid, mode: typeBits | (c.mode & 0o7777)) }
        if let x = readMetaXattr(url.path) {
            metaCache[url.path] = x
            return FileMeta(uid: x.uid, gid: x.gid, mode: typeBits | (x.mode & 0o7777))
        }
        let (u, g) = ownerOf(url.deletingLastPathComponent())
        return FileMeta(uid: u, gid: g, mode: realMode)
    }

    private func realMode(_ path: String, followLinks: Bool = true) -> UInt32? {
        var sb = Darwin.stat()
        let r: Int32 = path.withCString { followLinks ? statCWrapper($0, &sb) : lstatCWrapper($0, &sb) }
        return r == 0 ? UInt32(sb.st_mode) : nil
    }

    // r/w/x access check (want = bitmask R_OK=4|W_OK=2|X_OK=1). root bypasses read/write and, for
    // execute, needs at least one x bit set. mirrors InMemoryVFS.permitted (src/vfs.ts).
    private func permits(_ m: FileMeta, _ want: UInt32, uid: UInt32, gid: UInt32, groups: [UInt32]) -> Bool {
        if want == 0 { return true }
        if uid == 0 { return (want & 1) != 0 ? (m.mode & 0o111) != 0 : true }
        var perms: UInt32
        if m.uid == uid { perms = (m.mode >> 6) & 7 }
        else if m.gid == gid || groups.contains(m.gid) { perms = (m.mode >> 3) & 7 }
        else { perms = m.mode & 7 }
        return (perms & want) == want
    }

    // ensure a path has a persisted xattr (adopt-on-touch): the first metadata-affecting op on a
    // foreign file stamps its inherited default so it's stable thereafter.
    @discardableResult
    private func adopt(_ url: URL, realMode: UInt32) -> FileMeta {
        if let c = metaCache[url.path] { return FileMeta(uid: c.uid, gid: c.gid, mode: (realMode & 0o170000) | (c.mode & 0o7777)) }
        if let x = readMetaXattr(url.path) {
            metaCache[url.path] = x
            return FileMeta(uid: x.uid, gid: x.gid, mode: (realMode & 0o170000) | (x.mode & 0o7777))
        }
        let (u, g) = ownerOf(url.deletingLastPathComponent())
        let m = FileMeta(uid: u, gid: g, mode: realMode)
        writeMetaXattr(url.path, m)
        return m
    }

    private func allocFd() -> Int32 {
        let fd = nextFd; nextFd += 1; return fd
    }

    // MARK: - File Operations

    func open(path: String, flags: Int32, mode: Int32, creds: FsCreds) -> Int32 {
        let url = resolve(path)
        let accessMode = flags & 0x3
        var isDir: ObjCBool = false
        let exists = fm.fileExists(atPath: url.path, isDirectory: &isDir)

        // O_EXCL: fail if exists
        if exists && (flags & O_CREAT != 0) && (flags & O_EXCL != 0) {
            return -EEXIST
        }

        // Opening a directory
        if flags & O_DIRECTORY != 0 || (exists && isDir.boolValue) {
            if !exists { return -ENOENT }
            if !isDir.boolValue { return -ENOTDIR }
            if let rm = realMode(url.path) {
                let m = adopt(url, realMode: rm)
                let want: UInt32 = accessMode == O_WRONLY ? 2 : (accessMode == O_RDWR ? 6 : 4)
                if !permits(m, want, uid: creds.euid, gid: creds.egid, groups: creds.groups) { return -EACCES }
            }
            let fd = allocFd()
            openDirs[fd] = DirState(url: url, entries: nil, position: 0)
            return fd
        }

        // O_CREAT: create file if doesn't exist. the real file stays app-accessible (0600); the
        // emulated perms/owner live in the xattr, stamped here from the creator's creds + umask,
        // with set-group-ID directory gid inheritance.
        let created = !exists && (flags & O_CREAT != 0)
        if created {
            let parent = url.deletingLastPathComponent()
            try? fm.createDirectory(at: parent, withIntermediateDirectories: true)
            fm.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: NSNumber(value: 0o600)])
            let pmeta = realMode(parent.path).map { effectiveMeta(parent, realMode: $0) }
            let sgid = (pmeta?.mode ?? 0) & 0o2000 != 0
            let gid = sgid ? (pmeta?.gid ?? creds.egid) : creds.egid
            let cmode = (UInt32(bitPattern: mode) & 0o7777) & ~creds.umask
            writeMetaXattr(url.path, FileMeta(uid: creds.euid, gid: gid, mode: 0o100000 | cmode))
        }

        if !fm.fileExists(atPath: url.path) { return -ENOENT }

        // access enforcement on an EXISTING file (a freshly created file is opened regardless of
        // its new mode, POSIX-style — the check for O_CREAT is on the parent dir, not the file).
        if !created, let rm = realMode(url.path) {
            let m = adopt(url, realMode: rm)
            var want: UInt32 = 0
            if accessMode == O_RDONLY || accessMode == O_RDWR { want |= 4 }
            if accessMode == O_WRONLY || accessMode == O_RDWR || (flags & O_TRUNC) != 0 { want |= 2 }
            if !permits(m, want, uid: creds.euid, gid: creds.egid, groups: creds.groups) { return -EACCES }
        }

        // O_TRUNC: truncate
        if flags & O_TRUNC != 0 {
            try? Data().write(to: url)
            cache.evict(path: url.path)
        }

        do {
            let handle: FileHandle
            switch accessMode {
            case O_RDONLY: handle = try FileHandle(forReadingFrom: url)
            case O_WRONLY: handle = try FileHandle(forWritingTo: url)
            case O_RDWR:   handle = try FileHandle(forUpdating: url)
            default:       handle = try FileHandle(forReadingFrom: url)
            }
            if flags & O_APPEND != 0 { handle.seekToEndOfFile() }
            let fd = allocFd()
            openFiles[fd] = handle
            openFilePaths[fd] = url.path
            return fd
        } catch {
            return -ENOENT
        }
    }

    func close(fd: Int32) -> Int32 {
        if let handle = openFiles.removeValue(forKey: fd) {
            openFilePaths.removeValue(forKey: fd)
            try? handle.close()
            return 0
        }
        if openDirs.removeValue(forKey: fd) != nil { return 0 }
        return -EBADF
    }

    func read(fd: Int32, count: Int) -> (Int32, Data?) {
        guard let handle = openFiles[fd] else { return (-EBADF, nil) }
        do {
            let data = try handle.read(upToCount: count)
            if let data = data, !data.isEmpty {
                return (Int32(data.count), data)
            }
            return (0, nil) // EOF
        } catch {
            return (-EBADF, nil)
        }
    }

    func write(fd: Int32, data: Data) -> Int32 {
        guard let handle = openFiles[fd] else { return -EBADF }
        do {
            // Evict cache for this file (if we can determine the path)
            // Note: FileHandle doesn't expose its path, so we track it separately
            // if needed. For now, writes through the bridge always evict by path
            // at the caller level (syscall handler passes path context).
            try handle.write(contentsOf: data)
            return Int32(data.count)
        } catch {
            return -EBADF
        }
    }

    func lseek(fd: Int32, offset: Int64, whence: Int32) -> Int64 {
        guard let handle = openFiles[fd] else { return Int64(-EBADF) }
        do {
            let newOffset: UInt64
            switch whence {
            case SEEK_SET:
                newOffset = UInt64(max(0, offset))
                try handle.seek(toOffset: newOffset)
            case SEEK_CUR:
                let current = handle.offsetInFile
                newOffset = UInt64(Int64(current) + offset)
                try handle.seek(toOffset: newOffset)
            case SEEK_END:
                let end = handle.seekToEndOfFile()
                newOffset = UInt64(Int64(end) + offset)
                try handle.seek(toOffset: newOffset)
            default:
                return Int64(-EINVAL)
            }
            return Int64(newOffset)
        } catch {
            return Int64(-EBADF)
        }
    }

    func stat(path: String) -> [String: Any] {
        return posixStat(resolve(path), followLinks: true)
    }

    /// lstat-style: don't follow symlinks (for fstatat with AT_SYMLINK_NOFOLLOW)
    func lstat(path: String) -> [String: Any] {
        return posixStat(resolve(path), followLinks: false)
    }

    private func posixStat(_ url: URL, followLinks: Bool) -> [String: Any] {
        var sb = Darwin.stat()
        let result: Int32 = url.path.withCString { followLinks ? statCWrapper($0, &sb) : lstatCWrapper($0, &sb) }
        guard result == 0 else { return ["ret": Int(-ENOENT)] }
        return statDict(sb, meta: effectiveMeta(url, realMode: UInt32(sb.st_mode)))
    }

    func fstat(fd: Int32) -> [String: Any] {
        if let handle = openFiles[fd] {
            var sb = Darwin.stat()
            guard Darwin.fstat(handle.fileDescriptor, &sb) == 0 else { return ["ret": Int(-EBADF)] }
            let m: FileMeta
            if let p = openFilePaths[fd] { m = effectiveMeta(URL(fileURLWithPath: p), realMode: UInt32(sb.st_mode)) }
            else { m = FileMeta(uid: 0, gid: 0, mode: UInt32(sb.st_mode)) }
            return statDict(sb, meta: m)
        }
        if let dirState = openDirs[fd] { return statURL(dirState.url) }
        return ["ret": Int(-EBADF)]
    }

    private func statURL(_ url: URL) -> [String: Any] {
        var sb = Darwin.stat()
        guard url.path.withCString({ statCWrapper($0, &sb) }) == 0 else { return ["ret": Int(-ENOENT)] }
        return statDict(sb, meta: effectiveMeta(url, realMode: UInt32(sb.st_mode)))
    }

    // merge the real stat (size/nlink/mtime/ino) with the emulated ownership (uid/gid + full mode).
    private func statDict(_ sb: Darwin.stat, meta: FileMeta) -> [String: Any] {
        // st_ino is a 64-bit UInt; on real APFS it can exceed Int64.max, so Int(_:) would trap.
        // it's just an identifier for the guest, so reinterpret the bit pattern.
        return [
            "ret": 0,
            "mode": Int(meta.mode),
            "size": Int(sb.st_size),
            "nlink": Int(sb.st_nlink),
            "mtime": Int(sb.st_mtimespec.tv_sec),
            "ino": Int(bitPattern: UInt(sb.st_ino)),
            "uid": Int(meta.uid),
            "gid": Int(meta.gid),
        ]
    }

    func getdents(fd: Int32) -> (Int32, [[String: Any]]) {
        guard var dirState = openDirs[fd] else { return (-EBADF, []) }

        // Lazily enumerate on first call
        if dirState.entries == nil {
            do {
                let entries = try fm.contentsOfDirectory(atPath: dirState.url.path)
                dirState.entries = entries
                openDirs[fd] = dirState
            } catch {
                return (-ENOENT, [])
            }
        }

        guard let entries = dirState.entries else { return (-ENOENT, []) }
        let pos = dirState.position
        if pos >= entries.count { return (0, []) }

        // batch entries so the enriched response fits the 16KB bridge slot; the guest calls
        // getdents repeatedly until it returns 0. without this, a big dir (e.g. the 285 ltp
        // binaries) overflows the slot, the response is dropped, and the guest sees nothing.
        let batchEnd = min(pos + 48, entries.count)

        var result: [[String: Any]] = []
        for i in pos..<batchEnd {
            let name = entries[i]
            let entryURL = dirState.url.appendingPathComponent(name)

            // derive the type from lstat-like attributes (attributesOfItem does NOT follow
            // symlinks), NOT fileExists (which does) — otherwise a symlink reports its target's
            // type and never DT_LNK, breaking rm -r / find / ls -F on links.
            let attrs = try? fm.attributesOfItem(atPath: entryURL.path)
            let ftype = attrs?[.type] as? FileAttributeType
            let dtype: Int
            let typeBits: UInt32
            switch ftype {
            case .some(.typeDirectory):        dtype = 4;  typeBits = 0o040000
            case .some(.typeSymbolicLink):     dtype = 10; typeBits = 0o120000
            case .some(.typeCharacterSpecial): dtype = 2;  typeBits = 0o020000
            case .some(.typeBlockSpecial):     dtype = 6;  typeBits = 0o060000
            case .some(.typeSocket):           dtype = 12; typeBits = 0o140000
            default:                           dtype = 8;  typeBits = 0o100000 // regular / unknown
            }

            var entry: [String: Any] = ["name": name, "type": dtype]

            // Enrich with stat + emulated ownership so an `ls -l` gets the whole listing's
            // uid/gid/mode in this batch, no per-entry stat storm on the guest side.
            if let attrs = attrs {
                let size = attrs[.size] as? Int ?? 0
                let realPerm = UInt32(attrs[.posixPermissions] as? Int ?? 0o644)
                let modDate = attrs[.modificationDate] as? Date ?? Date()
                let m = effectiveMeta(entryURL, realMode: typeBits | realPerm)
                entry["mode"] = Int(m.mode)
                entry["uid"] = Int(m.uid)
                entry["gid"] = Int(m.gid)
                entry["size"] = size
                entry["mtime"] = safeInt(modDate.timeIntervalSince1970)
            }

            result.append(entry)
        }

        dirState.position = batchEnd
        openDirs[fd] = dirState
        return (Int32(result.count), result)
    }

    func mkdir(path: String, mode: Int32, creds: FsCreds) -> Int32 {
        let url = resolve(path)
        if fm.fileExists(atPath: url.path) { return -EEXIST }
        do {
            // real dir stays app-traversable (0700); emulated perms/owner in the xattr
            try fm.createDirectory(at: url, withIntermediateDirectories: false, attributes: [.posixPermissions: NSNumber(value: 0o700)])
            let parent = url.deletingLastPathComponent()
            let pmeta = realMode(parent.path).map { effectiveMeta(parent, realMode: $0) }
            let sgid = (pmeta?.mode ?? 0) & 0o2000 != 0
            let gid = sgid ? (pmeta?.gid ?? creds.egid) : creds.egid
            var cmode = (UInt32(bitPattern: mode) & 0o7777) & ~creds.umask
            if sgid { cmode |= 0o2000 } // a new dir under a set-group-ID parent is itself sgid
            writeMetaXattr(url.path, FileMeta(uid: creds.euid, gid: gid, mode: 0o040000 | cmode))
            return 0
        } catch {
            return -ENOENT
        }
    }

    func unlink(path: String) -> Int32 {
        let url = resolve(path)
        if !fm.fileExists(atPath: url.path) { return -ENOENT }
        do {
            try fm.removeItem(at: url)
            cache.evict(path: url.path)
            metaCache.removeValue(forKey: url.path)
            return 0
        } catch {
            return -ENOENT
        }
    }

    func rename(oldPath: String, newPath: String) -> Int32 {
        let oldURL = resolve(oldPath)
        let newURL = resolve(newPath)
        do {
            if fm.fileExists(atPath: newURL.path) {
                try fm.removeItem(at: newURL)
            }
            try fm.moveItem(at: oldURL, to: newURL)
            cache.evict(path: oldURL.path)
            cache.evict(path: newURL.path)
            // the xattr rides along with the inode; just invalidate both cache keys so the next
            // stat re-reads it at the new path.
            metaCache.removeValue(forKey: oldURL.path)
            metaCache.removeValue(forKey: newURL.path)
            return 0
        } catch {
            return -ENOENT
        }
    }

    // MARK: - New Methods

    func symlink(target: String, linkpath: String) -> Int32 {
        let linkURL = resolve(linkpath)
        // Ensure parent exists
        let parent = linkURL.deletingLastPathComponent()
        try? fm.createDirectory(at: parent, withIntermediateDirectories: true)
        do {
            // Target is stored as-is (relative or absolute guest path)
            try fm.createSymbolicLink(atPath: linkURL.path, withDestinationPath: target)
            return 0
        } catch {
            return -EEXIST
        }
    }

    func readlink(path: String) -> (Int32, String?) {
        let url = resolve(path)
        do {
            let target = try fm.destinationOfSymbolicLink(atPath: url.path)
            return (0, target)
        } catch {
            return (-ENOENT, nil)
        }
    }

    func ftruncate(fd: Int32, length: Int64) -> Int32 {
        guard let handle = openFiles[fd] else { return -EBADF }
        do {
            try handle.truncate(atOffset: UInt64(length))
            return 0
        } catch {
            return -EINVAL
        }
    }

    /// Duplicate a file handle — gives the caller a new independent fd
    /// pointing to the same underlying file at the same position.
    func dup(fd: Int32) -> Int32 {
        guard let handle = openFiles[fd] else { return -EBADF }
        // Use POSIX dup on the underlying file descriptor
        let posixFd = Darwin.dup(handle.fileDescriptor)
        guard posixFd >= 0 else { return -EBADF }
        let newHandle = FileHandle(fileDescriptor: posixFd, closeOnDealloc: true)
        let newFd = allocFd()
        openFiles[newFd] = newHandle
        return newFd
    }

    // access(2) checks with the REAL uid/gid (F_OK=0, R_OK=4, W_OK=2, X_OK=1).
    func faccessat(path: String, mode: Int32, creds: FsCreds) -> Int32 {
        let url = resolve(path)
        guard let rm = realMode(url.path) else { return -ENOENT }
        if mode == 0 { return 0 } // F_OK: existence
        let m = effectiveMeta(url, realMode: rm)
        return permits(m, UInt32(mode) & 7, uid: creds.ruid, gid: creds.rgid, groups: creds.groups) ? 0 : -EACCES
    }

    // only the owner (or root) may chmod. the emulated mode lives in the xattr; we never touch
    // the real file's perms (it stays app-accessible).
    func chmod(path: String, mode: Int32, creds: FsCreds) -> Int32 {
        let url = resolve(path)
        guard let rm = realMode(url.path) else { return -ENOENT }
        let m = adopt(url, realMode: rm)
        if creds.euid != 0 && creds.euid != m.uid { return -EPERM }
        let newMode = (m.mode & 0o170000) | (UInt32(bitPattern: mode) & 0o7777)
        writeMetaXattr(url.path, FileMeta(uid: m.uid, gid: m.gid, mode: newMode))
        return 0
    }

    // chown against the emulated model (kernel setattr_prepare semantics; mirrors chownInode in
    // src/vfs.ts). uid/gid == -1 mean "leave unchanged". a successful chown of a non-directory
    // drops the set-user-ID bit, and set-group-ID if group-executable.
    func chown(path: String, uid: Int32, gid: Int32, creds: FsCreds) -> Int32 {
        let url = resolve(path)
        guard let rm = realMode(url.path) else { return -ENOENT }
        let m = adopt(url, realMode: rm)
        let uidPresent = uid != -1
        let gidPresent = gid != -1
        let wantUid = uidPresent ? UInt32(bitPattern: uid) : m.uid
        let wantGid = gidPresent ? UInt32(bitPattern: gid) : m.gid
        if creds.euid != 0 {
            if uidPresent && (m.uid != creds.euid || wantUid != m.uid) { return -EPERM }
            if gidPresent && (m.uid != creds.euid ||
                (wantGid != creds.egid && wantGid != creds.rgid && wantGid != m.gid && !creds.groups.contains(wantGid))) { return -EPERM }
        }
        var newMode = m.mode
        if (m.mode & 0o170000) != 0o040000 {
            newMode &= ~0o4000
            if (newMode & 0o010) != 0 { newMode &= ~0o2000 }
        }
        writeMetaXattr(url.path, FileMeta(uid: wantUid, gid: wantGid, mode: newMode))
        return 0
    }
}

// MARK: - Read Cache

/// LRU read cache for frequently accessed files.
/// Lives in Swift (outside WebView memory budget).
/// Thread-safe via os_unfair_lock. Write-through: writes evict before returning.
private final class ReadCache {
    var maxBytes: Int = 128 * 1024 * 1024 // Default 128MB, configurable via /etc/jitter/config.json
    private var entries: [String: CacheEntry] = [:]
    private var totalBytes: Int = 0
    private var accessOrder: [String] = [] // LRU: most recent at end
    private var lock = os_unfair_lock()

    struct CacheEntry {
        let data: Data
        let offset: UInt64 // file offset this data starts at
    }

    /// Read from cache. Returns nil if not cached.
    func get(path: String) -> Data? {
        guard maxBytes > 0 else { return nil }
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        guard let entry = entries[path] else { return nil }
        // Move to end of LRU
        if let idx = accessOrder.firstIndex(of: path) {
            accessOrder.remove(at: idx)
            accessOrder.append(path)
        }
        return entry.data
    }

    /// Store in cache.
    func put(path: String, data: Data) {
        guard maxBytes > 0 else { return }
        guard data.count <= maxBytes else { return } // Don't cache files bigger than max
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        // Remove old entry if exists
        if let old = entries.removeValue(forKey: path) {
            totalBytes -= old.data.count
            accessOrder.removeAll { $0 == path }
        }

        // Evict LRU entries until we have space
        while totalBytes + data.count > maxBytes, let oldest = accessOrder.first {
            if let entry = entries.removeValue(forKey: oldest) {
                totalBytes -= entry.data.count
            }
            accessOrder.removeFirst()
        }

        entries[path] = CacheEntry(data: data, offset: 0)
        totalBytes += data.count
        accessOrder.append(path)
    }

    /// Evict a specific path (called on write/unlink/rename).
    func evict(path: String) {
        guard maxBytes > 0 else { return }
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        if let entry = entries.removeValue(forKey: path) {
            totalBytes -= entry.data.count
            accessOrder.removeAll { $0 == path }
        }
    }
}
