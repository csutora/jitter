import Foundation

/// Native filesystem backed by the app container's Documents directory.
/// Maps guest Linux paths to real paths under Documents/rootfs/.
///
/// Guest path "/root/test.txt" → <AppContainer>/Documents/rootfs/root/test.txt
class NativeFS {
    private let rootPath: URL
    private let fm = FileManager.default
    private var openFiles: [Int32: FileHandle] = [:]
    private var openDirs: [Int32: (URL, [String]?, Int)] = [:] // (dirURL, entries cache, position)
    private var nextFd: Int32 = 100 // Start high to avoid collisions with JS-side fds

    // Linux errno values
    private let ENOENT: Int32 = 2
    private let EEXIST: Int32 = 17
    private let EBADF: Int32 = 9
    private let EINVAL: Int32 = 22
    private let EISDIR: Int32 = 21
    private let ENOTDIR: Int32 = 20
    private let ENOTEMPTY: Int32 = 39

    // Linux open flags
    private let O_RDONLY: Int32 = 0
    private let O_WRONLY: Int32 = 1
    private let O_RDWR: Int32 = 2
    private let O_CREAT: Int32 = 64
    private let O_TRUNC: Int32 = 512
    private let O_APPEND: Int32 = 1024
    private let O_DIRECTORY: Int32 = 0x10000

    // Linux seek whence
    private let SEEK_SET: Int32 = 0
    private let SEEK_CUR: Int32 = 1
    private let SEEK_END: Int32 = 2

    init() {
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first!
        rootPath = docs.appendingPathComponent("rootfs", isDirectory: true)

        // Ensure rootfs directory exists
        try? fm.createDirectory(at: rootPath, withIntermediateDirectories: true)

        // Create default directories if they don't exist
        for dir in ["root", "home", "var", "tmp"] {
            let dirURL = rootPath.appendingPathComponent(dir, isDirectory: true)
            try? fm.createDirectory(at: dirURL, withIntermediateDirectories: true)
        }
    }

    /// Map a guest Linux path to a real filesystem URL
    private func resolve(_ guestPath: String) -> URL {
        // Normalize: strip leading /, resolve . and ..
        var components = guestPath.split(separator: "/").map(String.init)
        var resolved: [String] = []
        for c in components {
            if c == "." { continue }
            if c == ".." { resolved.popLast(); continue }
            resolved.append(c)
        }
        var url = rootPath
        for c in resolved {
            url = url.appendingPathComponent(c)
        }
        return url
    }

    private func allocFd() -> Int32 {
        let fd = nextFd
        nextFd += 1
        return fd
    }

    // MARK: - File Operations

    func open(path: String, flags: Int32, mode: Int32) -> Int32 {
        let url = resolve(path)
        let accessMode = flags & 0x3 // O_RDONLY=0, O_WRONLY=1, O_RDWR=2
        var isDir: ObjCBool = false
        let exists = fm.fileExists(atPath: url.path, isDirectory: &isDir)

        // Opening a directory
        if flags & O_DIRECTORY != 0 || (exists && isDir.boolValue) {
            if !exists { return -ENOENT }
            if !isDir.boolValue { return -ENOTDIR }
            let fd = allocFd()
            openDirs[fd] = (url, nil, 0)
            return fd
        }

        // O_CREAT: create file if it doesn't exist
        if !exists && (flags & O_CREAT != 0) {
            // Ensure parent directory exists
            let parent = url.deletingLastPathComponent()
            try? fm.createDirectory(at: parent, withIntermediateDirectories: true)
            fm.createFile(atPath: url.path, contents: nil, attributes: [
                .posixPermissions: NSNumber(value: mode & 0o777)
            ])
        }

        if !fm.fileExists(atPath: url.path) {
            return -ENOENT
        }

        // O_TRUNC: truncate to zero
        if flags & O_TRUNC != 0 {
            try? Data().write(to: url)
        }

        // Open the file handle
        do {
            let handle: FileHandle
            switch accessMode {
            case O_RDONLY:
                handle = try FileHandle(forReadingFrom: url)
            case O_WRONLY:
                handle = try FileHandle(forWritingTo: url)
            case O_RDWR:
                handle = try FileHandle(forUpdating: url)
            default:
                handle = try FileHandle(forReadingFrom: url)
            }

            // O_APPEND: seek to end
            if flags & O_APPEND != 0 {
                handle.seekToEndOfFile()
            }

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
        if openDirs.removeValue(forKey: fd) != nil {
            return 0
        }
        return -EBADF
    }

    func read(fd: Int32, count: Int) -> (Int32, Data?) {
        guard let handle = openFiles[fd] else {
            return (-EBADF, nil)
        }
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
        guard let handle = openFiles[fd] else {
            return -EBADF
        }
        do {
            try handle.write(contentsOf: data)
            return Int32(data.count)
        } catch {
            return -EBADF
        }
    }

    func lseek(fd: Int32, offset: Int64, whence: Int32) -> Int64 {
        guard let handle = openFiles[fd] else {
            return Int64(-EBADF)
        }
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
        do {
            let attrs = try fm.attributesOfItem(atPath: url.path)
            let fileType = attrs[.type] as? FileAttributeType
            let size = attrs[.size] as? Int ?? 0
            let permissions = attrs[.posixPermissions] as? Int ?? 0o644
            let modDate = attrs[.modificationDate] as? Date ?? Date()
            let isDir = fileType == .typeDirectory
            let isSymlink = fileType == .typeSymbolicLink

            // st_mode: S_IFREG=0o100000, S_IFDIR=0o40000, S_IFLNK=0o120000
            var stMode = permissions
            if isDir {
                stMode |= 0o40000
            } else if isSymlink {
                stMode |= 0o120000
            } else {
                stMode |= 0o100000
            }

            return [
                "ret": 0,
                "mode": stMode,
                "size": size,
                "nlink": isDir ? 2 : 1,
                "mtime": Int(modDate.timeIntervalSince1970),
            ]
        } catch {
            return ["ret": Int(-ENOENT)]
        }
    }

    func getdents(fd: Int32) -> (Int32, [[String: Any]]) {
        guard var dirState = openDirs[fd] else {
            return (-EBADF, [])
        }

        // Lazily enumerate directory on first getdents call
        if dirState.1 == nil {
            do {
                let entries = try fm.contentsOfDirectory(atPath: dirState.0.path)
                dirState.1 = entries
                openDirs[fd] = dirState
            } catch {
                return (-ENOENT, [])
            }
        }

        guard let entries = dirState.1 else {
            return (-ENOENT, [])
        }

        // Return remaining entries from current position
        let pos = dirState.2
        if pos >= entries.count {
            return (0, []) // No more entries
        }

        var result: [[String: Any]] = []
        for i in pos..<entries.count {
            let name = entries[i]
            let entryURL = dirState.0.appendingPathComponent(name)
            var isDir: ObjCBool = false
            fm.fileExists(atPath: entryURL.path, isDirectory: &isDir)

            // d_type: DT_DIR=4, DT_REG=8, DT_LNK=10
            let dtype: Int = isDir.boolValue ? 4 : 8

            result.append([
                "name": name,
                "type": dtype,
            ])
        }

        // Mark all entries as read
        dirState.2 = entries.count
        openDirs[fd] = dirState

        return (Int32(result.count), result)
    }

    func mkdir(path: String, mode: Int32) -> Int32 {
        let url = resolve(path)
        if fm.fileExists(atPath: url.path) {
            return -EEXIST
        }
        do {
            try fm.createDirectory(at: url, withIntermediateDirectories: false, attributes: [
                .posixPermissions: NSNumber(value: mode & 0o777)
            ])
            return 0
        } catch {
            return -ENOENT // parent doesn't exist
        }
    }

    func unlink(path: String) -> Int32 {
        let url = resolve(path)
        if !fm.fileExists(atPath: url.path) {
            return -ENOENT
        }
        do {
            try fm.removeItem(at: url)
            return 0
        } catch {
            return -ENOENT
        }
    }

    func rename(oldPath: String, newPath: String) -> Int32 {
        let oldURL = resolve(oldPath)
        let newURL = resolve(newPath)
        do {
            // Remove destination if it exists (Linux rename semantics)
            if fm.fileExists(atPath: newURL.path) {
                try fm.removeItem(at: newURL)
            }
            try fm.moveItem(at: oldURL, to: newURL)
            return 0
        } catch {
            return -ENOENT
        }
    }
}
