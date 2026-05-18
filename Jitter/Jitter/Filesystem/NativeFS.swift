import Foundation
import os.lock

// File-scope wrappers for POSIX stat/lstat to disambiguate from NativeFS.stat method
private func statCWrapper(_ path: UnsafePointer<CChar>, _ sb: UnsafeMutablePointer<stat>) -> Int32 {
    return stat(path, sb)
}
private func lstatCWrapper(_ path: UnsafePointer<CChar>, _ sb: UnsafeMutablePointer<stat>) -> Int32 {
    return lstat(path, sb)
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
    private var openDirs: [Int32: DirState] = [:]
    private var nextFd: Int32 = 100

    // Read cache (LRU, configurable size, lives outside WebView memory budget)
    private var cache = ReadCache()

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
    static let bootstrapVersion = 2

    // Busybox applets
    static let busyboxApplets = [
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
            "usr", "usr/bin", "usr/sbin", "sbin",
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

        // Create busybox applet symlinks
        let busyboxDest = "busybox" // relative symlink within same directory
        var symlinkCount = 0
        for applet in Self.busyboxApplets {
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
        print("[NativeFS] Created \(symlinkCount) applet symlinks")

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

    private func allocFd() -> Int32 {
        let fd = nextFd; nextFd += 1; return fd
    }

    // MARK: - File Operations

    func open(path: String, flags: Int32, mode: Int32) -> Int32 {
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
            let fd = allocFd()
            openDirs[fd] = DirState(url: url, entries: nil, position: 0)
            return fd
        }

        // O_CREAT: create file if doesn't exist
        if !exists && (flags & O_CREAT != 0) {
            let parent = url.deletingLastPathComponent()
            try? fm.createDirectory(at: parent, withIntermediateDirectories: true)
            fm.createFile(atPath: url.path, contents: nil, attributes: [
                .posixPermissions: NSNumber(value: mode & 0o777)
            ])
        }

        if !fm.fileExists(atPath: url.path) { return -ENOENT }

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
            return fd
        } catch {
            return -ENOENT
        }
    }

    func close(fd: Int32) -> Int32 {
        if let handle = openFiles.removeValue(forKey: fd) {
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
        let url = resolve(path)
        // Use POSIX stat() which follows symlinks (vs FileManager.attributesOfItem which doesn't)
        return posixStat(url.path, followLinks: true)
    }

    /// lstat-style: don't follow symlinks (for fstatat with AT_SYMLINK_NOFOLLOW)
    func lstat(path: String) -> [String: Any] {
        let url = resolve(path)
        return posixStat(url.path, followLinks: false)
    }

    private func posixStat(_ path: String, followLinks: Bool) -> [String: Any] {
        var sb = Darwin.stat()
        let result: Int32 = path.withCString { ptr in
            return followLinks ? statCWrapper(ptr, &sb) : lstatCWrapper(ptr, &sb)
        }
        guard result == 0 else {
            return ["ret": Int(-ENOENT)]
        }
        return statFromPosix(sb)
    }

    func fstat(fd: Int32) -> [String: Any] {
        guard let handle = openFiles[fd] else {
            // Check if it's a directory fd
            if let dirState = openDirs[fd] {
                return statURL(dirState.url)
            }
            return ["ret": Int(-EBADF)]
        }
        // Use POSIX fstat on the underlying file descriptor
        var sb = Darwin.stat()
        guard Darwin.fstat(handle.fileDescriptor, &sb) == 0 else {
            return ["ret": Int(-EBADF)]
        }
        return statFromPosix(sb)
    }

    private func statURL(_ url: URL) -> [String: Any] {
        do {
            let attrs = try fm.attributesOfItem(atPath: url.path)
            let fileType = attrs[.type] as? FileAttributeType
            let size = attrs[.size] as? Int ?? 0
            let permissions = attrs[.posixPermissions] as? Int ?? 0o644
            let modDate = attrs[.modificationDate] as? Date ?? Date()
            let isDir = fileType == .typeDirectory
            let isSymlink = fileType == .typeSymbolicLink

            var stMode = permissions
            if isDir { stMode |= 0o40000 }
            else if isSymlink { stMode |= 0o120000 }
            else { stMode |= 0o100000 }

            return [
                "ret": 0,
                "mode": stMode,
                "size": size,
                "nlink": isDir ? 2 : 1,
                "mtime": Int(modDate.timeIntervalSince1970),
                "ino": 0, // Could use inode number if needed
            ]
        } catch {
            return ["ret": Int(-ENOENT)]
        }
    }

    private func statFromPosix(_ sb: Darwin.stat) -> [String: Any] {
        return [
            "ret": 0,
            "mode": Int(sb.st_mode),
            "size": Int(sb.st_size),
            "nlink": Int(sb.st_nlink),
            "mtime": Int(sb.st_mtimespec.tv_sec),
            "ino": Int(sb.st_ino),
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

        var result: [[String: Any]] = []
        for i in pos..<entries.count {
            let name = entries[i]
            let entryURL = dirState.url.appendingPathComponent(name)

            // Include stat data for performance (eliminates per-entry stat calls)
            var isDir: ObjCBool = false
            fm.fileExists(atPath: entryURL.path, isDirectory: &isDir)
            let dtype: Int = isDir.boolValue ? 4 : 8

            var entry: [String: Any] = ["name": name, "type": dtype]

            // Enrich with stat data
            if let attrs = try? fm.attributesOfItem(atPath: entryURL.path) {
                let size = attrs[.size] as? Int ?? 0
                let permissions = attrs[.posixPermissions] as? Int ?? 0o644
                let modDate = attrs[.modificationDate] as? Date ?? Date()
                var mode = permissions
                if isDir.boolValue { mode |= 0o40000 } else { mode |= 0o100000 }
                entry["mode"] = mode
                entry["size"] = size
                entry["mtime"] = Int(modDate.timeIntervalSince1970)
            }

            result.append(entry)
        }

        dirState.position = entries.count
        openDirs[fd] = dirState
        return (Int32(result.count), result)
    }

    func mkdir(path: String, mode: Int32) -> Int32 {
        let url = resolve(path)
        if fm.fileExists(atPath: url.path) { return -EEXIST }
        do {
            try fm.createDirectory(at: url, withIntermediateDirectories: false, attributes: [
                .posixPermissions: NSNumber(value: mode & 0o777)
            ])
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

    func faccessat(path: String, mode: Int32) -> Int32 {
        let url = resolve(path)
        // F_OK=0, R_OK=4, W_OK=2, X_OK=1
        if mode == 0 {
            return fm.fileExists(atPath: url.path) ? 0 : -ENOENT
        }
        if !fm.fileExists(atPath: url.path) { return -ENOENT }
        if (mode & 4 != 0) && !fm.isReadableFile(atPath: url.path) { return -EPERM }
        if (mode & 2 != 0) && !fm.isWritableFile(atPath: url.path) { return -EPERM }
        // X_OK: check executable permission
        if (mode & 1 != 0) && !fm.isExecutableFile(atPath: url.path) { return -EPERM }
        return 0
    }

    func chmod(path: String, mode: Int32) -> Int32 {
        let url = resolve(path)
        do {
            try fm.setAttributes([.posixPermissions: NSNumber(value: mode & 0o7777)],
                                 ofItemAtPath: url.path)
            return 0
        } catch {
            return -ENOENT
        }
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
