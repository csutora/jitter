import Foundation
import os.lock
import Compression

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

// Emulated ownership metadata stored per inode as a 16-byte xattr. `mode` is the full emulated
// mode (type + suid/sgid/sticky + perm bits) and `rdev` the device number of a char/block node.
//
// The type bits are re-derived from the real file for every kind the host can actually represent
// -- regular files, directories, symlinks, fifos -- so a stale or foreign xattr cannot misreport
// what a file is. A char or block device is the exception: the host has no way to hold one, so for
// those the xattr is the only record there is and its type has to be believed.
//
// A 12-byte xattr is the older layout and reads back with rdev 0, so nothing already on disk needs
// rewriting.
struct FileMeta { var uid: UInt32; var gid: UInt32; var mode: UInt32; var rdev: UInt32 = 0 }

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
    // the ACTIVE root's on-disk directory. all guest paths resolve under it. mutable: `switchRoot`
    // repoints it at a different userspace under `rootsPath`. (the session is torn down + rebooted on
    // switch, so no open fd outlives the change.)
    private(set) var rootPath: URL
    // <Documents>/roots — holds one directory per installed userspace (default, alpine, ...).
    let rootsPath: URL
    private(set) var activeRoot: String = "default"
    private let fm = FileManager.default
    private var openFiles: [Int32: FileHandle] = [:]
    private var openFilePaths: [Int32: String] = [:] // fd -> real path, for fstat ownership lookup
    private var openDirs: [Int32: DirState] = [:]
    private var nextFd: Int32 = 100

    /// which guest process opened each descriptor. a process that exits normally closes its own,
    /// but one the kernel force-terminates (SIGKILL, and the ltp runner sends `kill -9 -1` between
    /// every test) never runs a line of javascript again -- so nothing on the guest side can give
    /// these back, and the FileHandles live until the app dies. closeOwner() is the kernel's way to
    /// reclaim them, the same way it already reclaims a dead process's pipe ends and record locks.
    private var fdOwner: [Int32: Int] = [:]
    /// set by the router for the duration of one bridge call, so allocFd can stamp the opener
    var currentOwnerPid: Int = 0

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
    private let ENAMETOOLONG: Int32 = 36  // linux's value; darwin says 63
    // returned by open() for a char/block node: this layer stores the node, the guest kernel owns
    // the driver behind it. see the comment at the check in open().
    private let ENODEV: Int32 = 19

    // Linux open flags
    private let O_RDONLY: Int32 = 0
    private let O_WRONLY: Int32 = 1
    private let O_RDWR: Int32 = 2
    private let O_CREAT: Int32 = 64
    private let O_EXCL: Int32 = 128
    private let O_TRUNC: Int32 = 512
    private let O_APPEND: Int32 = 1024
    private let O_DIRECTORY: Int32 = 0x10000
    // linux aarch64 O_TMPFILE is __O_TMPFILE(0o20000000) | O_DIRECTORY(0o40000); both bits must be
    // set, which is how the kernel distinguishes it from a plain directory open.
    private let O_TMPFILE_LINUX: Int32 = 0o20040000
    /// descriptors opened O_TMPFILE that have not been given a name yet, and the hidden file
    /// standing in for them. closing one without naming it destroys the file, as linux does.
    private var unnamedFiles: [Int32: String] = [:]
    private var tmpfileStore: URL { rootPath.appendingPathComponent(".jitter-unnamed") }
    private let O_NOFOLLOW: Int32 = 0x20000

    // Linux seek whence
    private let SEEK_SET: Int32 = 0
    private let SEEK_CUR: Int32 = 1
    private let SEEK_END: Int32 = 2

    // Current bootstrap version — bump when busybox or default configs change
    static let bootstrapVersion = 21  // v21: /dev/shm again, for roots bootstrapped before it landed

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
        rootsPath = docs.appendingPathComponent("roots", isDirectory: true)
        try? fm.createDirectory(at: rootsPath, withIntermediateDirectories: true)

        // migrate the pre-roots single rootfs (<Documents>/rootfs) into roots/default so existing
        // installs keep their state. one-time; harmless once roots/default exists.
        let legacy = docs.appendingPathComponent("rootfs", isDirectory: true)
        let defaultRoot = rootsPath.appendingPathComponent("default", isDirectory: true)
        if fm.fileExists(atPath: legacy.path) && !fm.fileExists(atPath: defaultRoot.path) {
            try? fm.moveItem(at: legacy, to: defaultRoot)
        }
        try? fm.createDirectory(at: defaultRoot, withIntermediateDirectories: true)

        // busybox + default configs are only ever seeded into `default` (pulled roots bring their own
        // userspace). point rootPath at default for the bootstrap (which resolves writes via rootPath),
        // then switch it to the persisted active root below.
        rootPath = defaultRoot
        bootstrapIfNeeded()
        installSyscallProbe()

        // pick the persisted active root; fall back to default if it's gone.
        let persisted = (try? String(contentsOf: rootsPath.appendingPathComponent(".active"), encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let p = persisted, !p.isEmpty,
           fm.fileExists(atPath: rootsPath.appendingPathComponent(p, isDirectory: true).path) {
            activeRoot = p
        }
        rootPath = rootsPath.appendingPathComponent(activeRoot, isDirectory: true)
        loadCacheConfig()
    }

    // MARK: - Roots (userspace management)

    func rootURL(_ name: String) -> URL { rootsPath.appendingPathComponent(name, isDirectory: true) }

    /// installed roots (directory names under rootsPath), sorted.
    func listRoots() -> [String] {
        let items = (try? fm.contentsOfDirectory(atPath: rootsPath.path)) ?? []
        return items.filter { !$0.hasPrefix(".") &&
            (try? rootURL($0).resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true }.sorted()
    }

    func rootExists(_ name: String) -> Bool { fm.fileExists(atPath: rootURL(name).path) }

    /// create an empty root directory (a pull extracts into it). returns false if it already exists.
    @discardableResult
    func createRoot(_ name: String) -> Bool {
        guard isValidRootName(name), !rootExists(name) else { return false }
        try? fm.createDirectory(at: rootURL(name), withIntermediateDirectories: true)
        return true
    }

    /// repoint the active root. the caller (kernel/session) is responsible for the reboot; this just
    /// changes where guest paths resolve and persists the choice.
    @discardableResult
    func switchRoot(_ name: String) -> Bool {
        guard rootExists(name) else { return false }
        activeRoot = name
        rootPath = rootURL(name)
        try? name.write(to: rootsPath.appendingPathComponent(".active"), atomically: true, encoding: .utf8)
        return true
    }

    func removeRoot(_ name: String) -> Bool {
        guard name != "default", name != activeRoot, rootExists(name) else { return false }
        try? fm.removeItem(at: rootURL(name))
        return true
    }

    // root names are single path components used to build directory paths + guest-visible under /roots.
    private func isValidRootName(_ name: String) -> Bool {
        !name.isEmpty && name != "." && name != ".." && !name.contains("/") && !name.hasPrefix(".")
    }

    // MARK: - Bootstrap

    /// On first launch (or after app update), create the rootfs directory tree
    /// and populate it with busybox + default configs.
    /// syscall_probe (tools/syscall_probe.c): raw-svc conformance checks. ltp's binaries are static
    /// glibc, so a failing case there cannot distinguish "the kernel is wrong" from "the libc wrapper
    /// answered without asking"; this asks the kernel directly. refreshed on every launch rather than
    /// at bootstrap, because a diagnostic that does not match the build it is diagnosing is worse
    /// than no diagnostic.
    private func installSyscallProbe() {
        for name in ["syscall_probe", "tiny_exit"] {
            guard let src = Bundle.main.url(forResource: name, withExtension: nil) else { continue }
            let dest = rootPath.appendingPathComponent("bin/" + name)
            try? fm.removeItem(at: dest)
            try? fm.copyItem(at: src, to: dest)
            try? fm.setAttributes([.posixPermissions: NSNumber(value: 0o755)], ofItemAtPath: dest.path)
        }
    }

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

        // Create directory tree. the modes are not decoration: /tmp is world-writable with the
        // sticky bit on every real unix, and anything that drops to an unprivileged uid before
        // making a temp directory (which is most of ltp's error-path tests) gets EACCES without it.
        // /root is 0700 for the same reason in reverse -- a test that checks "an ordinary user
        // cannot read root's home" needs that to be true.
        let dirs: [(String, Int)] = [
            ("bin", 0o755), ("etc", 0o755), ("etc/jitter", 0o755),
            ("tmp", 0o1777), ("root", 0o700), ("home", 0o755),
            ("var", 0o755), ("var/tmp", 0o1777), ("var/log", 0o755),
            ("usr", 0o755), ("usr/bin", 0o755), ("usr/sbin", 0o755),
            ("sbin", 0o755), ("opt", 0o755), ("opt/ltp", 0o755),
            // posix shared memory is a directory of files, and it has to be a REAL one: two
            // processes share an object only if they open the same file.
            ("dev", 0o755), ("dev/shm", 0o1777),
        ]
        for (dir, mode) in dirs {
            let url = rootPath.appendingPathComponent(dir, isDirectory: true)
            try? fm.createDirectory(at: url, withIntermediateDirectories: true)
            try? fm.setAttributes([.posixPermissions: NSNumber(value: mode)], ofItemAtPath: url.path)
            // the guest sees the EMULATED mode from the xattr, not the host one, so setting the host
            // permission alone changes nothing a program can observe. stamp both.
            writeMetaXattr(url.path, FileMeta(uid: 0, gid: 0, mode: 0o040000 | UInt32(mode)))
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

        // the `jitter` command. it is an OVERLAY: a small static shim dropped into every root's /bin
        // (here for the seeded default, and on pull/switch for the others), so the command exists
        // wherever you are without a distro's own filesystem being otherwise touched. it just talks to
        // /dev/jitter — all policy lives host-side (MessageRouter.runJitterCommand).
        if let shimURL = Bundle.main.url(forResource: "jitter-shim", withExtension: nil) {
            let dest = rootPath.appendingPathComponent("bin/jitter")
            try? fm.removeItem(at: dest)
            try? fm.copyItem(at: shimURL, to: dest)
            try? fm.setAttributes([.posixPermissions: NSNumber(value: 0o755)], ofItemAtPath: dest.path)
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
        // drop the old tree first: the staged names changed once, and extracting over the top
        // would leave both spellings of every test sitting in the suite directory.
        let ltpBin = rootPath.appendingPathComponent("opt/ltp/testcases/bin")
        try? fm.removeItem(at: ltpBin)
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
        #
        #   ltp            run everything (about an hour and a half)
        #   ltp mmap       run just the tests whose name starts with mmap
        #   ltp mmap fcntl several prefixes
        #
        # a full sweep is far too slow to iterate against, and nearly all of a fix's evidence is in
        # one subsystem, so being able to ask for that subsystem is the difference between a
        # minute and an afternoon.
        BIN=/opt/ltp/testcases/bin
        # 30s matches ltp's OWN per-test default. 20 was an arbitrary harness number, and it was
        # cutting off tests that are slow BY DESIGN rather than slow because we are: kill12 loops
        # thirteen times with a sleep(1) on each side of a fork, so it takes about 26 seconds on
        # any kernel.
        T=${LTP_TIMEOUT:-30}
        if [ ! -d "$BIN" ]; then echo "ltp: suite not installed"; exit 1; fi
        sel=""
        for a in "$@"; do sel="$sel $BIN/$a*"; done
        [ -n "$sel" ] || sel="$BIN/*"
        pass=0 fail=0 brok=0 conf=0 to=0 n=0
        # a sweep takes an hour and its output only arrives at the end, so a run that wedges
        # partway tells you nothing about where. mirror every line to a log as it happens, and
        # name the test BEFORE running it, so the last two lines of the log are always
        # "what finished" and "what it died in".
        LOG=${LTP_LOG:-/tmp/ltp.log}
        : > "$LOG"
        for t in $sel; do
          [ -f "$t" ] || continue
          name=${t##*/}
          # *_child binaries are HELPERS that a test re-execs with arguments, not tests. run alone
          # they print their usage and exit non-zero, which the tally then filed as a failure.
          case $name in *_child) continue ;; esac
          n=$((n+1))
          echo "RUNNING $name" >> "$LOG"
          timeout -k 5 "$T" "$t" >/dev/null 2>&1
          rc=$?
          # ltp's exit status is a BITMASK of the result kinds it saw, not a single code:
          # TFAIL=1, TBROK=2, TWARN=4, TINFO=16, TCONF=32. reading it as a plain number filed
          # every combination (a skip that also warned exits 36) under "unknown, call it a fail".
          case $rc in
            137|143) to=$((to+1)); st=TIMEOUT ;;
            *)
              if [ $((rc & 1)) -ne 0 ];  then fail=$((fail+1)); st=FAIL
              elif [ $((rc & 2)) -ne 0 ]; then brok=$((brok+1)); st=BROK
              elif [ $((rc & 32)) -ne 0 ]; then conf=$((conf+1)); st=SKIP
              elif [ $rc -eq 0 ] || [ $rc -eq 4 ] || [ $rc -eq 16 ] || [ $rc -eq 20 ]; then pass=$((pass+1)); st=PASS
              else fail=$((fail+1)); st=EXIT$rc
              fi ;;
          esac
          # a test that had to be killed leaves its own children behind -- the timeout reaches the
          # harness, not the processes it forked -- and those orphans hold process slots forever.
          # past the live-process cap every later fork fails, so tests start failing for no reason
          # of their own and the run stops meaning anything. reclaim before moving on. kill -1 is
          # defined to spare pid 1 and the sender, so this only sweeps up strays.
          kill -9 -1 2>/dev/null
          printf '%-42s %s\\n' "$name" "$st"
          printf '%-42s %s\\n' "$name" "$st" >> "$LOG"
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

    // MARK: - Distro pull (extract a rootfs tarball into a root)

    /// decompress a gzip member to a Data. strips the gzip wrapper (RFC 1952) and raw-inflates the
    /// DEFLATE stream via the Compression framework (which speaks raw DEFLATE, not the gzip wrapper).
    static func gunzip(_ data: Data) -> Data? {
        let bytes = [UInt8](data)
        guard bytes.count > 18, bytes[0] == 0x1f, bytes[1] == 0x8b, bytes[2] == 0x08 else { return nil }
        let flg = bytes[3]
        var pos = 10
        if flg & 0x04 != 0 { // FEXTRA
            guard pos + 2 <= bytes.count else { return nil }
            let xlen = Int(bytes[pos]) | (Int(bytes[pos + 1]) << 8); pos += 2 + xlen
        }
        if flg & 0x08 != 0 { while pos < bytes.count && bytes[pos] != 0 { pos += 1 }; pos += 1 } // FNAME
        if flg & 0x10 != 0 { while pos < bytes.count && bytes[pos] != 0 { pos += 1 }; pos += 1 } // FCOMMENT
        if flg & 0x02 != 0 { pos += 2 } // FHCRC
        let n = bytes.count
        guard pos < n - 8 else { return nil }
        // gzip footer: crc32(4) + isize(4, uncompressed size mod 2^32) — trust isize for the buffer.
        var isize = Int(bytes[n - 4]) | (Int(bytes[n - 3]) << 8) | (Int(bytes[n - 2]) << 16) | (Int(bytes[n - 1]) << 24)
        if isize <= 0 { isize = max(1, (n - pos - 8) * 8) } // fallback if isize is bogus/huge-wrapped
        let deflate = Array(bytes[pos..<(n - 8)])
        var dst = [UInt8](repeating: 0, count: isize)
        let decoded = dst.withUnsafeMutableBufferPointer { d in
            deflate.withUnsafeBufferPointer { s in
                compression_decode_buffer(d.baseAddress!, isize, s.baseAddress!, deflate.count, nil, COMPRESSION_ZLIB)
            }
        }
        guard decoded > 0 else { return nil }
        return Data(dst.prefix(decoded))
    }

    /// extract a (possibly gzipped) rootfs tarball into roots/<name>, preserving the directory tree,
    /// symlinks, and permissions. returns (ok, fileCount, note). the root is wiped + recreated first.
    func installRootfs(_ tarball: Data, into name: String) -> (ok: Bool, files: Int, note: String) {
        guard isValidRootName(name) else { return (false, 0, "invalid root name") }
        let tar: Data
        if tarball.count > 2 && tarball[tarball.startIndex] == 0x1f && tarball[tarball.startIndex + 1] == 0x8b {
            guard let un = Self.gunzip(tarball) else { return (false, 0, "gunzip failed") }
            tar = un
        } else {
            tar = tarball
        }
        let dest = rootURL(name)
        try? fm.removeItem(at: dest)
        try? fm.createDirectory(at: dest, withIntermediateDirectories: true)
        let n = extractUstar(tar, into: dest)
        return (n >= 0, max(0, n), n >= 0 ? "ok" : "extract failed")
    }

    // proper ustar/gnu tar extractor: nested paths, dirs, symlinks, hardlinks, GNU long names, modes.
    // operates on an in-memory Data (decompressed tar). returns files+links written, or -1 on fatal.
    private func extractUstar(_ tar: Data, into destDir: URL) -> Int {
        let b = [UInt8](tar)
        var pos = 0
        var count = 0
        var longName: String? = nil

        func octal(_ off: Int, _ len: Int) -> Int {
            var s = ""
            for i in off..<min(off + len, b.count) { let c = b[pos + i]; if c == 0 || c == 0x20 { break }; s.append(Character(UnicodeScalar(c))) }
            return Int(s, radix: 8) ?? 0
        }
        func cstr(_ off: Int, _ len: Int) -> String {
            var end = off
            while end < off + len && pos + end < b.count && b[pos + end] != 0 { end += 1 }
            return String(decoding: b[(pos + off)..<(pos + end)], as: UTF8.self)
        }
        // sanitize a tar path to a safe relative path (strip leading ./ and /, drop any .. component).
        func sanitize(_ raw: String) -> String? {
            var parts: [String] = []
            for c in raw.split(separator: "/").map(String.init) {
                if c.isEmpty || c == "." { continue }
                if c == ".." { return nil }
                parts.append(c)
            }
            return parts.isEmpty ? nil : parts.joined(separator: "/")
        }

        while pos + 512 <= b.count {
            // a name field starting with NUL => zero block => end of archive
            if b[pos] == 0 { break }
            let rawName = cstr(0, 100)
            let prefix = cstr(345, 155)
            let size = octal(124, 12)
            let mode = octal(100, 8)
            let typeflag = b[pos + 156]
            let linkname = cstr(157, 100)
            pos += 512

            let dataStart = pos
            pos += (size + 511) & ~511 // advance past the (padded) file body

            let fullName = longName ?? (prefix.isEmpty ? rawName : prefix + "/" + rawName)
            longName = nil

            switch typeflag {
            case 0x4C: // 'L' GNU long name: body is the real name of the NEXT entry
                longName = String(decoding: b[dataStart..<min(dataStart + size, b.count)], as: UTF8.self)
                    .trimmingCharacters(in: CharacterSet(charactersIn: "\0"))
                continue
            case 0x78, 0x67, 0x4B: // 'x'/'g' pax, 'K' GNU long link — skip (short names cover alpine)
                continue
            default: break
            }

            guard let rel = sanitize(fullName) else { continue }
            let target = destDir.appendingPathComponent(rel)

            switch typeflag {
            case 0x35: // '5' directory
                try? fm.createDirectory(at: target, withIntermediateDirectories: true)
            case 0x32: // '2' symlink
                try? fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? fm.removeItem(at: target)
                if (try? fm.createSymbolicLink(atPath: target.path, withDestinationPath: linkname)) != nil { count += 1 }
            case 0x31: // '1' hardlink -> copy the linked file's contents (simpler than real hardlinks here)
                try? fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                if let src = sanitize(linkname) {
                    try? fm.removeItem(at: target)
                    try? fm.copyItem(at: destDir.appendingPathComponent(src), to: target)
                    count += 1
                }
            case 0x30, 0: // '0' or NUL regular file
                try? fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                let body = size > 0 ? Data(b[dataStart..<min(dataStart + size, b.count)]) : Data()
                do {
                    try body.write(to: target)
                    try? fm.setAttributes([.posixPermissions: NSNumber(value: mode == 0 ? 0o644 : mode)], ofItemAtPath: target.path)
                    count += 1
                } catch { /* skip unwritable entry */ }
            default: break // char/block/fifo devices etc — not represented on the host fs
            }
        }
        return count
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

    private static let symlinkHopLimit = 40

    /// Map a guest Linux path to a real filesystem URL, walking it the way a chroot would: every
    /// component is resolved inside the rootfs and symlinks are followed by hand. letting the host
    /// kernel follow a guest symlink would escape the root, because the guest writes the target
    /// verbatim -- `ln -s /etc/passwd x` leaves a host symlink to the *host's* /etc/passwd, and a
    /// target of `../../..` climbs past a boundary the host knows nothing about.
    /// `follow` applies to the last component only; intermediate ones are always followed.
    /// nil means the walk gave up (a symlink cycle); callers report ELOOP. it must never fall back
    /// to the root -- an unlink or truncate aimed at the whole rootfs is not a recoverable mistake.
    func resolve(_ guestPath: String, follow: Bool = true, creds: FsCreds = .root) -> URL? {
        let (url, err) = walk(guestPath, follow: follow, creds: creds)
        lastWalkErrno = err
        return url
    }

    /// why the last resolve() gave up. filesystem calls are serialised on one queue, so this is a
    /// simpler way to keep the real errno (EACCES from a directory that denies search, ELOOP from a
    /// cycle) than threading a result type through fifteen call sites.
    private var lastWalkErrno: Int32 = 0

    /// set for the duration of one open that asked for RESOLVE_NO_SYMLINKS, which refuses a path
    /// with a symlink anywhere in it rather than following it. a property for the same reason
    /// lastWalkErrno is one: filesystem calls are serialised on a single queue, so there is never
    /// a second walk in flight to confuse it with.
    private var walkRefusesSymlinks = false
    private var walkError: Int32 { lastWalkErrno != 0 ? lastWalkErrno : -linuxErrno(ELOOP) }

    /// the resolution above, keeping the errno so callers can report ELOOP rather than guess.
    func walk(_ guestPath: String, follow: Bool, creds: FsCreds = .root) -> (URL?, Int32) {
        // linux bounds a path at PATH_MAX and each component at NAME_MAX, and says so with
        // ENAMETOOLONG. this lives in the walk rather than in any one syscall, because every
        // path-taking call has the same limit -- checking it only in open left stat, link, unlink
        // and the rest reporting ENOENT, which is a different thing and hides the real mistake.
        if guestPath.utf8.count > 4095 { return (nil, -ENAMETOOLONG) }
        var pending = guestPath.split(separator: "/").map(String.init)
        for c in pending where c.utf8.count > 255 { return (nil, -ENAMETOOLONG) }
        var taken: [String] = []
        var hops = 0

        func assemble(_ parts: [String], _ extra: String? = nil) -> URL {
            var url = rootPath
            for p in parts { url = url.appendingPathComponent(p) }
            if let e = extra { url = url.appendingPathComponent(e) }
            return url
        }

        while !pending.isEmpty {
            let c = pending.removeFirst()
            if c == "." { continue }
            if c == ".." { if !taken.isEmpty { taken.removeLast() }; continue }

            // looking inside a directory needs search permission on it. this is checked for EVERY
            // component, which is what makes a file unreachable through a directory that denies
            // search even when the file itself is world-readable.
            if !canSearch(assemble(taken), creds) { return (nil, -EACCES) }

            if pending.isEmpty && !follow {
                // openat2's RESOLVE_NO_SYMLINKS refuses the link itself, not just links in the
                // middle of the path, so the last component has to be looked at even when the
                // caller was not going to follow it.
                if walkRefusesSymlinks {
                    var lb = Darwin.stat()
                    let there = assemble(taken, c).path.withCString { lstatCWrapper($0, &lb) } == 0
                    if there && (lb.st_mode & S_IFMT) == S_IFLNK { return (nil, -linuxErrno(ELOOP)) }
                }
                taken.append(c); break
            }

            let candidate = assemble(taken, c)
            var sb = Darwin.stat()
            let ok = candidate.path.withCString { lstatCWrapper($0, &sb) } == 0
            if ok && (sb.st_mode & S_IFMT) == S_IFLNK {
                if walkRefusesSymlinks { return (nil, -linuxErrno(ELOOP)) }
                hops += 1
                if hops > NativeFS.symlinkHopLimit { return (nil, -linuxErrno(ELOOP)) }
                guard let target = try? fm.destinationOfSymbolicLink(atPath: candidate.path) else {
                    return (nil, -ENOENT)
                }
                let parts = target.split(separator: "/").map(String.init)
                // a link with no components names nothing; without this the walk would quietly
                // fall through to the link's PARENT directory.
                if parts.isEmpty { return (nil, -ENOENT) }
                if target.hasPrefix("/") { taken.removeAll() }
                pending.insert(contentsOf: parts, at: 0)
                continue
            }
            // a component in the MIDDLE of a path has to EXIST and be a directory. a missing one is
            // ENOENT: linux does not create intermediate directories for anybody, and letting the
            // walk append it anyway meant open("a/nodir/x", O_CREAT) reported success and created
            // nothing -- the worst of both answers.
            if !ok && !pending.isEmpty { return (nil, -ENOENT) }
            // when it exists but is not a directory the answer is ENOTDIR -- "you tried to walk
            // through a file" -- and not ENOENT, which says the name does not exist and sends the
            // caller looking for a typo instead.
            if ok && !pending.isEmpty && (sb.st_mode & S_IFMT) != S_IFDIR { return (nil, -ENOTDIR) }
            taken.append(c)
        }
        return (assemble(taken), 0)
    }

    /// may these credentials look inside this directory? root always may -- unlike execute on a
    /// regular file, directory search is not gated on an x bit for the superuser.
    private func canSearch(_ dir: URL, _ creds: FsCreds) -> Bool {
        if creds.euid == 0 { return true }
        guard let rm = realMode(dir.path) else { return true } // missing: let the real op report it
        if (rm & 0o170000) != 0o040000 { return true }         // not a directory: ENOTDIR, not EACCES
        return permits(effectiveMeta(dir, realMode: rm), 1, uid: creds.euid, gid: creds.egid, groups: creds.groups)
    }

    /// creating, removing or renaming a name happens IN a directory, so it needs write permission
    /// on that directory -- not on the file, which may not even exist yet. this was only checked
    /// when making a symlink, so every other way of changing a directory's contents ignored it.
    private func mayWriteParent(_ url: URL, _ creds: FsCreds) -> Bool {
        if creds.euid == 0 { return true }
        let parent = url.deletingLastPathComponent()
        guard let rm = realMode(parent.path) else { return true } // missing: let the real op report it
        return permits(effectiveMeta(parent, realMode: rm), 2, uid: creds.euid, gid: creds.egid, groups: creds.groups)
    }

    /// existence without following: a dangling symlink is still a name that exists, so unlink and
    /// the create paths must see it (fileExists stats through the link and reports false).
    private func lexists(_ url: URL) -> Bool {
        var sb = Darwin.stat()
        return url.path.withCString { lstatCWrapper($0, &sb) } == 0
    }

    /// true when the last component of an already-walked path is itself a symlink.
    /// whether a "do not follow" request should read the LINK's own metadata record.
    ///
    /// only a symlink has two of them. asking for the nofollow record of an ordinary file finds
    /// nothing stamped there and falls back to the host's real mode -- which is how every lstat
    /// of a plain file started reporting 0755 instead of what the guest had set.
    private func linkRecord(_ url: URL, _ nofollow: Bool) -> Bool {
        return nofollow && isSymlink(url)
    }

    private func isSymlink(_ url: URL) -> Bool {
        var sb = Darwin.stat()
        guard url.path.withCString({ lstatCWrapper($0, &sb) }) == 0 else { return false }
        return (sb.st_mode & S_IFMT) == S_IFLNK
    }

    // MARK: - Emulated ownership/permissions (xattr-backed)

    private static let metaXattrName = "com.csutora.jitter.meta"

    // a symlink and its target are distinct metadata objects, so they need distinct cache keys:
    // getxattr/setxattr without XATTR_NOFOLLOW address the target, with it the link itself.
    private func metaKey(_ path: String, _ nofollow: Bool) -> String { nofollow ? "L:" + path : path }

    // 16 bytes now; 12 is the older layout, which reads back with rdev 0. accepting both means
    // nothing already stamped on disk has to be rewritten.
    private static let metaXattrSize = 16

    private func readMetaXattr(_ path: String, nofollow: Bool = false) -> FileMeta? {
        var buf = [UInt8](repeating: 0, count: NativeFS.metaXattrSize)
        let opts: Int32 = nofollow ? XATTR_NOFOLLOW : 0
        let n = buf.withUnsafeMutableBytes {
            getxattr(path, NativeFS.metaXattrName, $0.baseAddress, NativeFS.metaXattrSize, 0, opts)
        }
        guard n == NativeFS.metaXattrSize || n == 12 else { return nil }
        func rd(_ o: Int) -> UInt32 { UInt32(buf[o]) | (UInt32(buf[o+1]) << 8) | (UInt32(buf[o+2]) << 16) | (UInt32(buf[o+3]) << 24) }
        return FileMeta(uid: rd(0), gid: rd(4), mode: rd(8), rdev: n == 12 ? 0 : rd(12))
    }

    // write the emulated metadata and refresh the cache (write-through).
    func writeMetaXattr(_ path: String, _ m: FileMeta, nofollow: Bool = false) {
        var buf = [UInt8](repeating: 0, count: NativeFS.metaXattrSize)
        func wr(_ v: UInt32, _ o: Int) { buf[o] = UInt8(v & 0xff); buf[o+1] = UInt8((v >> 8) & 0xff); buf[o+2] = UInt8((v >> 16) & 0xff); buf[o+3] = UInt8((v >> 24) & 0xff) }
        wr(m.uid, 0); wr(m.gid, 4); wr(m.mode, 8); wr(m.rdev, 12)
        let opts: Int32 = nofollow ? XATTR_NOFOLLOW : 0
        _ = buf.withUnsafeBytes {
            setxattr(path, NativeFS.metaXattrName, $0.baseAddress, NativeFS.metaXattrSize, 0, opts)
        }
        metaCache[metaKey(path, nofollow)] = m
    }

    // owner (uid,gid) of a path for foreign-import inheritance: the nearest stamped ancestor,
    // defaulting to root (0,0) at/above the rootfs boundary.
    private func ownerOf(_ url: URL) -> (UInt32, UInt32) {
        if url.path.count <= rootPath.path.count { return (0, 0) }
        if let c = metaCache[url.path] { return (c.uid, c.gid) }
        if let x = readMetaXattr(url.path) { return (x.uid, x.gid) }
        return ownerOf(url.deletingLastPathComponent())
    }

    /// The type a stamped file claims to be, when the host cannot hold that type itself.
    ///
    /// Reality wins for every kind the host filesystem can actually represent, which is what keeps
    /// a stale or foreign xattr from misreporting a file's kind. A character or block device is the
    /// one thing it cannot represent -- the node is a placeholder regular file on disk -- so there
    /// the xattr is the only record of what the file is, and it has to be believed.
    private func emulatedType(_ x: FileMeta, realMode: UInt32) -> UInt32 {
        let claimed = x.mode & 0o170000
        // S_IFCHR / S_IFBLK / S_IFSOCK: three types the host filesystem cannot hold, so what is on
        // disk is an empty placeholder and the stamp is the only record of what it really is.
        if claimed == 0o020000 || claimed == 0o060000 || claimed == 0o140000 { return claimed }
        return realMode & 0o170000
    }

    // resolve the emulated {uid,gid,mode,rdev} for a real path. xattr is authoritative for owner
    // and permission bits; a missing xattr means a foreign import -> inherit the parent dir's owner
    // + the real file's mode.
    // `nofollow` picks WHICH of the two records to read: a symlink has ownership of its own,
    // separate from its target's, and lchown writes the one lstat has to read back.
    private func effectiveMeta(_ url: URL, realMode: UInt32, nofollow: Bool = false) -> FileMeta {
        let key = metaKey(url.path, nofollow)
        if let c = metaCache[key] {
            return FileMeta(uid: c.uid, gid: c.gid, mode: emulatedType(c, realMode: realMode) | (c.mode & 0o7777), rdev: c.rdev)
        }
        if let x = readMetaXattr(url.path, nofollow: nofollow) {
            metaCache[key] = x
            return FileMeta(uid: x.uid, gid: x.gid, mode: emulatedType(x, realMode: realMode) | (x.mode & 0o7777), rdev: x.rdev)
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
    private func adopt(_ url: URL, realMode: UInt32, nofollow: Bool = false) -> FileMeta {
        let key = metaKey(url.path, nofollow)
        if let c = metaCache[key] { return FileMeta(uid: c.uid, gid: c.gid, mode: emulatedType(c, realMode: realMode) | (c.mode & 0o7777), rdev: c.rdev) }
        if let x = readMetaXattr(url.path, nofollow: nofollow) {
            metaCache[key] = x
            return FileMeta(uid: x.uid, gid: x.gid, mode: emulatedType(x, realMode: realMode) | (x.mode & 0o7777), rdev: x.rdev)
        }
        let (u, g) = ownerOf(url.deletingLastPathComponent())
        let m = FileMeta(uid: u, gid: g, mode: realMode)
        writeMetaXattr(url.path, m, nofollow: nofollow)
        return m
    }

    // MARK: - Extended attributes
    //
    // the guest's attributes are kept under a prefix of their own. two reasons: the metadata xattr
    // this filesystem stamps on every inode must stay invisible to the guest, and listxattr has to
    // report exactly what the guest set rather than whatever else apfs keeps on the file
    // (com.apple.quarantine and friends), which would appear out of nowhere as unnamed attributes.
    private static let guestXattrPrefix = "com.csutora.jitter.x."

    // darwin's XATTR_CREATE/XATTR_REPLACE are 0x2/0x4; linux's are 1/2, and the guest speaks linux.
    private static let linuxXattrCreate: Int32 = 1
    private static let linuxXattrReplace: Int32 = 2

    // linux caps a name at 255 bytes and a value at 64 KiB.
    private static let xattrNameMax = 255
    private static let xattrSizeMax = 65536

    private static let eNoData: Int32 = 61       // linux ENODATA; darwin calls it ENOATTR (93)
    private static let eNotSupported: Int32 = 95 // linux EOPNOTSUPP

    /// who may touch an attribute, by namespace. unknown namespaces do not exist at all, "user.*"
    /// lives only on regular files and directories (which is why setting one on a fifo is EPERM),
    /// "trusted.*" is privileged, and everything else follows the file's own permissions.
    private func xattrPolicy(name: String, meta: FileMeta, write: Bool, creds: FsCreds) -> Int32 {
        if name.isEmpty || name.utf8.count > NativeFS.xattrNameMax { return -ERANGE }
        if name.hasPrefix("user.") {
            let type = meta.mode & 0o170000
            if type != 0o100000 && type != 0o040000 { return -EPERM }
        } else if name.hasPrefix("trusted.") {
            if creds.euid != 0 { return -EPERM }
            return 0
        } else if !name.hasPrefix("security.") {
            return -NativeFS.eNotSupported
        }
        let want: UInt32 = write ? 2 : 4
        if !permits(meta, want, uid: creds.euid, gid: creds.egid, groups: creds.groups) { return -EACCES }
        return 0
    }

    /// resolve for an attribute call: the name, its emulated metadata, and the errno if either step
    /// fails. `nofollow` addresses a symlink itself rather than what it points at.
    private func xattrTarget(_ path: String, _ nofollow: Bool, _ creds: FsCreds) -> (URL?, FileMeta?, Int32) {
        let (u, err) = walk(path, follow: !nofollow, creds: creds)
        guard let url = u else { return (nil, nil, err != 0 ? err : -linuxErrno(ELOOP)) }
        guard let rm = realMode(url.path, followLinks: !nofollow) else { return (nil, nil, -ENOENT) }
        return (url, adopt(url, realMode: rm, nofollow: nofollow), 0)
    }

    func getXattr(path: String, name: String, nofollow: Bool, creds: FsCreds) -> (Int32, Data?) {
        let (u, meta, err) = xattrTarget(path, nofollow, creds)
        guard let url = u, let meta else { return (err, nil) }
        let pol = xattrPolicy(name: name, meta: meta, write: false, creds: creds)
        if pol != 0 { return (pol, nil) }
        let full = NativeFS.guestXattrPrefix + name
        let opts: Int32 = nofollow ? XATTR_NOFOLLOW : 0
        let n = Darwin.getxattr(url.path, full, nil, 0, 0, opts)
        if n < 0 { return (errno == ENOATTR ? -NativeFS.eNoData : -linuxErrno(errno), nil) }
        if n == 0 { return (0, Data()) }
        var buf = [UInt8](repeating: 0, count: n)
        let got = buf.withUnsafeMutableBytes { Darwin.getxattr(url.path, full, $0.baseAddress, n, 0, opts) }
        if got < 0 { return (errno == ENOATTR ? -NativeFS.eNoData : -linuxErrno(errno), nil) }
        return (Int32(got), Data(buf[0..<got]))
    }

    func setXattr(path: String, name: String, value: Data, flags: Int32, nofollow: Bool, creds: FsCreds) -> Int32 {
        if (flags & (NativeFS.linuxXattrCreate | NativeFS.linuxXattrReplace))
            == (NativeFS.linuxXattrCreate | NativeFS.linuxXattrReplace) { return -EINVAL }
        let (u, meta, err) = xattrTarget(path, nofollow, creds)
        guard let url = u, let meta else { return err }
        let pol = xattrPolicy(name: name, meta: meta, write: true, creds: creds)
        if pol != 0 { return pol }
        if value.count > NativeFS.xattrSizeMax { return -E2BIG }
        var opts: Int32 = nofollow ? XATTR_NOFOLLOW : 0
        if (flags & NativeFS.linuxXattrCreate) != 0 { opts |= XATTR_CREATE }
        if (flags & NativeFS.linuxXattrReplace) != 0 { opts |= XATTR_REPLACE }
        let full = NativeFS.guestXattrPrefix + name
        // an empty value is legal and distinct from an absent attribute, but an empty Data has no
        // base address, so it needs a real pointer with a zero length.
        var bytes = [UInt8](value)
        if bytes.isEmpty { bytes = [0] }
        let rc = bytes.withUnsafeBytes {
            Darwin.setxattr(url.path, full, $0.baseAddress, value.count, 0, opts)
        }
        if rc != 0 { return errno == ENOATTR ? -NativeFS.eNoData : -linuxErrno(errno) }
        return 0
    }

    func removeXattr(path: String, name: String, nofollow: Bool, creds: FsCreds) -> Int32 {
        let (u, meta, err) = xattrTarget(path, nofollow, creds)
        guard let url = u, let meta else { return err }
        let pol = xattrPolicy(name: name, meta: meta, write: true, creds: creds)
        if pol != 0 { return pol }
        let opts: Int32 = nofollow ? XATTR_NOFOLLOW : 0
        let rc = Darwin.removexattr(url.path, NativeFS.guestXattrPrefix + name, opts)
        if rc != 0 { return errno == ENOATTR ? -NativeFS.eNoData : -linuxErrno(errno) }
        return 0
    }

    /// the guest's attribute names on a file, NUL-terminated one after another, which is the layout
    /// listxattr(2) hands back.
    func listXattr(path: String, nofollow: Bool, creds: FsCreds) -> (Int32, Data?) {
        let (u, meta, err) = xattrTarget(path, nofollow, creds)
        guard let url = u, meta != nil else { return (err, nil) }
        let opts: Int32 = nofollow ? XATTR_NOFOLLOW : 0
        let n = Darwin.listxattr(url.path, nil, 0, opts)
        if n < 0 { return (-linuxErrno(errno), nil) }
        if n == 0 { return (0, Data()) }
        var buf = [CChar](repeating: 0, count: n)
        let got = buf.withUnsafeMutableBufferPointer { Darwin.listxattr(url.path, $0.baseAddress, n, opts) }
        if got < 0 { return (-linuxErrno(errno), nil) }
        var out = Data()
        var start = 0
        let raw = buf.prefix(got).map { UInt8(bitPattern: $0) }
        for i in 0..<raw.count where raw[i] == 0 {
            if let s = String(bytes: raw[start..<i], encoding: .utf8), s.hasPrefix(NativeFS.guestXattrPrefix) {
                out.append(contentsOf: Array(s.dropFirst(NativeFS.guestXattrPrefix.count).utf8))
                out.append(0)
            }
            start = i + 1
        }
        return (Int32(out.count), out)
    }

    private func allocFd() -> Int32 {
        let fd = nextFd; nextFd += 1
        if currentOwnerPid != 0 { fdOwner[fd] = currentOwnerPid }
        return fd
    }

    /// close every descriptor still owned by a process, and report how many there were. safe to
    /// call for a process that already cleaned up after itself: it simply finds nothing.
    func closeOwner(pid: Int) -> Int {
        let mine = fdOwner.filter { $0.value == pid }.map { $0.key }
        for fd in mine { _ = close(fd: fd); fdOwner.removeValue(forKey: fd) }
        return mine.count
    }

    /// descriptors currently open, for diagnosing exactly the leak this exists to prevent
    func openFdCount() -> Int { return openFiles.count + openDirs.count }

    // MARK: - File Operations

    /// `didCreate` reports whether this open BROUGHT THE FILE INTO EXISTENCE, which the caller
    /// cannot work out afterwards -- and inotify has to distinguish IN_CREATE from a plain
    /// IN_OPEN of a file that was already there.
    func open(path: String, flags: Int32, mode: Int32, creds: FsCreds, didCreate: inout Bool,
              noSymlinks: Bool = false) -> Int32 {
        didCreate = false
        walkRefusesSymlinks = noSymlinks
        defer { walkRefusesSymlinks = false }
        // O_NOFOLLOW and O_CREAT|O_EXCL both act on the link itself, so they stop the walk one
        // component short; every other open resolves the final symlink like the kernel does.
        // aarch64 does not use the asm-generic flag numbering: O_DIRECTORY is 0o40000, O_NOFOLLOW
        // 0o100000, O_DIRECT 0o200000, O_LARGEFILE 0o400000. musl ors O_LARGEFILE into every open,
        // so reading 0o400000 as O_NOFOLLOW makes literally every open look like one.
        // O_TMPFILE: the PATH names a DIRECTORY, and what comes back is a writable file with no
        // name in it. modern software creates files this way -- write the contents first, give it a
        // name only when it is complete, and if anything goes wrong there is never a half-written
        // file to find. apk builds every package it installs this way, which is why refusing it
        // (EISDIR, because the path really is a directory) left correctly-named EMPTY files behind.
        //
        // there is no unnamed-file primitive to borrow here, so the file is real and lives out of
        // the guest's sight until it is named: no directory entry exists in the guest namespace,
        // which is the property that matters. linkat gives it its name by moving it into place.
        if flags & O_TMPFILE_LINUX == O_TMPFILE_LINUX {
            let (dirURL, dirErr) = walk(path, follow: true, creds: creds)
            guard let dir = dirURL else { return dirErr }
            var isD: ObjCBool = false
            guard fm.fileExists(atPath: dir.path, isDirectory: &isD), isD.boolValue else { return -ENOTDIR }
            if creds.euid != 0, let rm = realMode(dir.path),
               !permits(effectiveMeta(dir, realMode: rm), 2, uid: creds.euid, gid: creds.egid, groups: creds.groups) {
                return -EACCES
            }
            let hidden = tmpfileStore.appendingPathComponent(UUID().uuidString)
            try? fm.createDirectory(at: tmpfileStore, withIntermediateDirectories: true)
            guard fm.createFile(atPath: hidden.path, contents: nil,
                                attributes: [.posixPermissions: NSNumber(value: 0o600)]) else { return -EIO }
            let cmode = (UInt32(bitPattern: mode) & 0o7777) & ~creds.umask
            writeMetaXattr(hidden.path, FileMeta(uid: creds.euid, gid: inheritedGid(hidden, creds),
                                                 mode: 0o100000 | cmode))
            let rawfd = hidden.path.withCString { Darwin.open($0, O_RDWR) }
            if rawfd < 0 { try? fm.removeItem(at: hidden); return -linuxErrno(errno) }
            let fd = allocFd()
            openFiles[fd] = FileHandle(fileDescriptor: rawfd, closeOnDealloc: true)
            openFilePaths[fd] = hidden.path
            unnamedFiles[fd] = hidden.path
            return fd
        }
        let wantNofollow = flags & 0o100000 != 0
        let exclusiveCreate = (flags & 0o100 != 0) && (flags & 0o200 != 0)
        let (walked, walkErr) = walk(path, follow: !(wantNofollow || exclusiveCreate), creds: creds)
        guard let url = walked else { return walkErr }
        // O_PATH opens the NAME rather than the file behind it, so O_PATH|O_NOFOLLOW on a
        // symlink is how a caller takes hold of the link itself -- the one case where refusing a
        // symlink under O_NOFOLLOW would refuse exactly what was asked for.
        if wantNofollow && (flags & 0o10000000) == 0 && isSymlink(url) { return walkError }
        if exclusiveCreate && lexists(url) { return -EEXIST }
        let accessMode = flags & 0x3
        // the guest sends linux aarch64 open-flag values, whose bits differ from darwin's. notably
        // linux O_CLOEXEC (0o2000000 = 0x100000) collides with darwin O_DIRECTORY, which used to make
        // realpath's O_PATH|O_CLOEXEC opens spuriously fail ENOTDIR. interpret the linux bits directly.
        let wantCreat = flags & 0o100 != 0
        let wantExcl = flags & 0o200 != 0
        let wantTrunc = flags & 0o1000 != 0
        let wantAppend = flags & 0o2000 != 0
        let wantDir = flags & 0o40000 != 0
        var isDir: ObjCBool = false
        let exists = fm.fileExists(atPath: url.path, isDirectory: &isDir)

        // O_EXCL: fail if exists
        if exists && wantCreat && wantExcl {
            return -EEXIST
        }

        // Opening a directory
        if wantDir || (exists && isDir.boolValue) {
            if !exists { return -ENOENT }
            if !isDir.boolValue { return -ENOTDIR }
            // a directory can be opened for reading, never for writing. this is EISDIR, and it is
            // a different answer from "you lack permission" -- a caller distinguishes "that is a
            // directory" from "that is yours but closed to you" by exactly this errno.
            if accessMode == O_WRONLY || accessMode == O_RDWR { return -EISDIR }
            // O_CREAT names a file you are prepared to create, and you cannot create a directory
            // this way, so asking for one that already exists is EISDIR whatever the access mode.
            if wantCreat { return -EISDIR }
            if let rm = realMode(url.path) {
                let m = adopt(url, realMode: rm)
                let want: UInt32 = accessMode == O_WRONLY ? 2 : (accessMode == O_RDWR ? 6 : 4)
                if !permits(m, want, uid: creds.euid, gid: creds.egid, groups: creds.groups) { return -EACCES }
            }
            let fd = allocFd()
            openDirs[fd] = DirState(url: url, entries: nil, position: 0)
            return fd
        }

        // named pipe (fifo): open the host fifo NON-BLOCKING regardless of the requested mode, so a
        // blocking O_WRONLY open of a readerless fifo can't wedge the bridge thread (it returns ENXIO,
        // matching linux) and O_RDONLY returns immediately. proper blocking rendezvous would need
        // kernel-coordinated named pipes; this at least keeps fifo fixtures (dup/lseek/unlink) working.
        if exists && !isDir.boolValue {
            var lsb = Darwin.stat()
            if url.path.withCString({ lstatCWrapper($0, &lsb) }) == 0 && (lsb.st_mode & S_IFMT) == S_IFIFO {
                let rawfd = url.path.withCString { Darwin.open($0, accessMode | O_NONBLOCK) }
                if rawfd < 0 { return -linuxErrno(errno) }
                let fd = allocFd()
                openFiles[fd] = FileHandle(fileDescriptor: rawfd, closeOnDealloc: true)
                openFilePaths[fd] = url.path
                return fd
            }
        }

        // A DEVICE NODE IS NOT THIS LAYER'S TO OPEN.
        //
        // What is on disk is a placeholder; the thing the guest asked for is whatever driver sits
        // behind that major:minor, and which drivers exist is the guest kernel's business, not the
        // filesystem's. So say what it is and let the kernel open it. ENODEV is the signal: nothing
        // else in this path produces it, and `deviceAt` gives the caller the identity to act on.
        if exists, let rm = realMode(url.path, followLinks: false) {
            let fmt = effectiveMeta(url, realMode: rm).mode & 0o170000
            if fmt == 0o020000 || fmt == 0o060000 { return -ENODEV }
        }

        // O_CREAT: create file if doesn't exist. the real file stays app-accessible (0600); the
        // emulated perms/owner live in the xattr, stamped here from the creator's creds + umask,
        // with set-group-ID directory gid inheritance.
        let created = !exists && wantCreat
        didCreate = created
        if created {
            let parent = url.deletingLastPathComponent()
            try? fm.createDirectory(at: parent, withIntermediateDirectories: true)
            fm.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: NSNumber(value: 0o600)])
            let cmode = (UInt32(bitPattern: mode) & 0o7777) & ~creds.umask
            writeMetaXattr(url.path, FileMeta(uid: creds.euid, gid: inheritedGid(url, creds),
                                              mode: 0o100000 | cmode))
        }

        if !fm.fileExists(atPath: url.path) { return -ENOENT }

        // access enforcement on an EXISTING file (a freshly created file is opened regardless of
        // its new mode, POSIX-style — the check for O_CREAT is on the parent dir, not the file).
        if !created, let rm = realMode(url.path) {
            let m = adopt(url, realMode: rm)
            // O_NOATIME suppresses access-time updates, which is only yours to ask for on your own
            // files: it is how a reader avoids writing to a filesystem it may not own. anyone else
            // gets EPERM, and that check comes before the ordinary access check.
            if (flags & 0o1000000) != 0 && creds.euid != 0 && creds.euid != m.uid { return -EPERM }
            var want: UInt32 = 0
            if accessMode == O_RDONLY || accessMode == O_RDWR { want |= 4 }
            if accessMode == O_WRONLY || accessMode == O_RDWR || wantTrunc { want |= 2 }
            if !permits(m, want, uid: creds.euid, gid: creds.egid, groups: creds.groups) { return -EACCES }
        }

        // O_TRUNC: truncate
        if wantTrunc {
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
            // O_APPEND is a property of every write, not a one-off seek at open: the kernel moves to
            // the end of the file as part of each write, atomically. seeking once at open made the
            // FIRST write append and every later one land wherever the offset happened to be, so a
            // program that seeks and then writes -- a log rotator, ltp's write06 -- overwrote in
            // place. darwin's F_SETFL takes O_APPEND, so hand the real flag to the real descriptor
            // and let the host kernel do it properly.
            if wantAppend {
                let raw = handle.fileDescriptor
                let fl = Darwin.fcntl(raw, F_GETFL)
                if fl >= 0 { _ = Darwin.fcntl(raw, F_SETFL, fl | Darwin.O_APPEND) }
                handle.seekToEndOfFile()
            }
            let fd = allocFd()
            openFiles[fd] = handle
            openFilePaths[fd] = url.path
            return fd
        } catch {
            return -ENOENT
        }
    }

    /// The device a path names, if it names one. Answers the ENODEV that `open` returns for a node.
    func deviceAt(path: String, creds: FsCreds = .root) -> (mode: UInt32, rdev: UInt32)? {
        guard let url = resolve(path, creds: creds) else { return nil }
        guard let rm = realMode(url.path, followLinks: false) else { return nil }
        let m = effectiveMeta(url, realMode: rm)
        let fmt = m.mode & 0o170000
        guard fmt == 0o020000 || fmt == 0o060000 else { return nil }
        return (m.mode, m.rdev)
    }

    func close(fd: Int32) -> Int32 {
        fdOwner.removeValue(forKey: fd)
        if let handle = openFiles.removeValue(forKey: fd) {
            openFilePaths.removeValue(forKey: fd)
            try? handle.close()
            // an O_TMPFILE descriptor that was never given a name takes the file with it: that is
            // the guarantee the whole mechanism exists for -- an abandoned write leaves nothing.
            if let hidden = unnamedFiles.removeValue(forKey: fd) {
                try? fm.removeItem(atPath: hidden)
            }
            return 0
        }
        if openDirs.removeValue(forKey: fd) != nil { return 0 }
        return -EBADF
    }

    func read(fd: Int32, count: Int) -> (Int32, Data?) {
        // a directory descriptor is a real, open descriptor -- you just cannot read() it, you
        // read it with getdents. EISDIR says that; EBADF would claim the descriptor is not open.
        if openDirs[fd] != nil { return (-EISDIR, nil) }
        guard let handle = openFiles[fd] else { return (-EBADF, nil) }
        if count <= 0 { return (0, nil) }
        // raw read(2), not FileHandle -- for the same reason lseek is raw (see below): the
        // FileHandle APIs report failure by raising an OBJC exception that swift cannot catch, so
        // any descriptor that is not a plain regular file could take the whole app down. they also
        // flatten every failure into one error, when the errno IS the answer the caller needs:
        // EAGAIN on an empty non-blocking fifo is not the same as EBADF, and a guest that cannot
        // tell them apart cannot implement a pipe.
        var buf = [UInt8](repeating: 0, count: count)
        let n = buf.withUnsafeMutableBytes { Darwin.read(handle.fileDescriptor, $0.baseAddress, count) }
        if n < 0 { return (-linuxErrno(errno), nil) }
        if n == 0 { return (0, nil) } // EOF
        return (Int32(n), Data(buf[0..<n]))
    }

    // read at an explicit offset without disturbing the fd's current position. used by file-backed
    // mmap + pread (the dynamic loader maps shared libraries this way).
    func pread(fd: Int32, offset: UInt64, count: Int) -> (Int32, Data?) {
        guard let handle = openFiles[fd] else { return (-EBADF, nil) }
        if count <= 0 { return (0, nil) }
        // pread(2) reads at an offset ATOMICALLY. the old save-seek-read-restore dance was three
        // chances to leave the file position somewhere the caller did not put it, and could not
        // work at all on a descriptor that has no position.
        var buf = [UInt8](repeating: 0, count: count)
        let n = buf.withUnsafeMutableBytes { Darwin.pread(handle.fileDescriptor, $0.baseAddress, count, off_t(offset)) }
        if n < 0 { return (-linuxErrno(errno), nil) }
        if n == 0 { return (0, nil) } // EOF
        return (Int32(n), Data(buf[0..<n]))
    }

    func pwrite(fd: Int32, offset: UInt64, data: Data, noGrow: Bool = false) -> Int32 {
        guard let handle = openFiles[fd] else { return -EBADF }
        // F_SEAL_GROW: the file has promised never to get longer, so a write that would run past
        // the end is refused outright rather than shortened.
        if noGrow, let end = fileEnd(handle), offset + UInt64(data.count) > end { return -EPERM }
        do {
            let saved = handle.offsetInFile
            try handle.seek(toOffset: offset)
            try handle.write(contentsOf: data)
            try handle.seek(toOffset: saved)
            return Int32(data.count)
        } catch {
            return -EBADF
        }
    }

    /// reserve space for a file, growing it if the requested range runs past the end. this is
    /// fallocate's mode 0: the caller is promising to use the room, and a great deal of software
    /// then writes into it through a mapping rather than with write().
    func fallocate(fd: Int32, size: UInt64, noGrow: Bool = false) -> Int32 {
        guard let handle = openFiles[fd] else {
            if openDirs[fd] != nil { return -ENODEV }
            return -EBADF
        }
        var sb = Darwin.stat()
        if Darwin.fstat(handle.fileDescriptor, &sb) != 0 { return -linuxErrno(errno) }
        if UInt64(sb.st_size) >= size { return 0 }          // already covers the range
        if noGrow { return -EPERM }                          // F_SEAL_GROW
        // ask for contiguous space first, then set the size. F_PREALLOCATE only reserves blocks;
        // the file is not longer until ftruncate says so, which is what a mapping needs.
        var pre = fstore_t(fst_flags: UInt32(F_ALLOCATECONTIG), fst_posmode: F_PEOFPOSMODE,
                           fst_offset: 0, fst_length: off_t(size) - sb.st_size, fst_bytesalloc: 0)
        if fcntl(handle.fileDescriptor, F_PREALLOCATE, &pre) == -1 {
            pre.fst_flags = UInt32(F_ALLOCATEALL)           // fall back to a fragmented allocation
            _ = fcntl(handle.fileDescriptor, F_PREALLOCATE, &pre)
        }
        if Darwin.ftruncate(handle.fileDescriptor, off_t(size)) != 0 { return -linuxErrno(errno) }
        return 0
    }

    /// flush a descriptor's data to storage. fdatasync differs only in not promising the metadata,
    /// and there is nothing to gain here by distinguishing them.
    func fsync(fd: Int32, dataOnly: Bool) -> Int32 {
        guard let handle = openFiles[fd] else {
            if openDirs[fd] != nil { return 0 }   // a directory descriptor has nothing buffered
            return -EBADF
        }
        let rc = Darwin.fsync(handle.fileDescriptor)
        if rc != 0 { return -linuxErrno(errno) }
        return 0
    }

    /// `limit` is the process's RLIMIT_FSIZE, or -1 for no limit. a write that would take the file
    /// past it is refused with EFBIG (and the caller raises SIGXFSZ); one that STARTS below it and
    /// would cross it writes what fits, which is what makes `ulimit -f` truncate rather than fail.
    func write(fd: Int32, data: Data, limit: Int64 = -1, noGrow: Bool = false) -> Int32 {
        guard let handle = openFiles[fd] else { return -EBADF }
        if data.isEmpty { return 0 }
        // F_SEAL_GROW, as in pwrite: a write past the end is refused rather than shortened.
        if noGrow, let end = fileEnd(handle) {
            let here = Darwin.lseek(handle.fileDescriptor, 0, SEEK_CUR)
            if here >= 0 && UInt64(here) + UInt64(data.count) > end { return -EPERM }
        }
        var data = data
        if limit >= 0 {
            let here = Darwin.lseek(handle.fileDescriptor, 0, SEEK_CUR)
            if here >= 0 {
                if here >= limit { return -linuxErrno(EFBIG) }
                let room = limit - Int64(here)
                if Int64(data.count) > room { data = data.prefix(Int(room)) }
            }
        }
        // raw write(2). FileHandle.write reports failure by raising an OBJC exception that swift
        // cannot catch: writing to a full non-blocking fifo -- an ordinary thing a program does,
        // and precisely what ltp's write04 does -- killed the entire app. a guest must never be
        // able to do that.
        // it also returns the REAL count, so a short write stays short instead of being reported
        // as if everything landed. that is the difference between a fifo that fills up and one
        // that silently loses the tail of every large write.
        let n = data.withUnsafeBytes { Darwin.write(handle.fileDescriptor, $0.baseAddress, data.count) }
        if n < 0 { return -linuxErrno(errno) }
        return Int32(n)
    }

    /// Seek a descriptor. This goes through raw lseek(2) rather than FileHandle: the legacy
    /// FileHandle seek API raises OBJC exceptions on failure, which Swift cannot catch, so seeking
    /// a fifo -- a thing lseek is specified to answer with ESPIPE -- took the whole process down
    /// instead. the raw call also gives the exact errno for free.
    func lseek(fd: Int32, offset: Int64, whence: Int32) -> Int64 {
        guard let handle = openFiles[fd] else {
            // a directory descriptor is seekable in linux; report position 0 for the simple cases
            if openDirs[fd] != nil { return whence == SEEK_SET ? max(0, offset) : 0 }
            return Int64(-EBADF)
        }
        // SEEK_DATA and SEEK_HOLE are numbered the OTHER WAY ROUND on darwin (3 is SEEK_HOLE
        // there, 4 is SEEK_DATA; linux has them the other way). passing the guest's number through
        // silently asked for the opposite of what it wanted -- "where does the data start" answered
        // with where it ends.
        var w = whence
        if whence == 3 { w = SEEK_DATA } else if whence == 4 { w = SEEK_HOLE }  // linux 3/4 -> darwin 4/3
        let r = Darwin.lseek(handle.fileDescriptor, off_t(offset), w)
        if r < 0 { return Int64(-linuxErrno(errno)) }
        return Int64(r)
    }

    func stat(path: String, creds: FsCreds = .root) -> [String: Any] {
        guard let u = resolve(path, creds: creds) else { return ["ret": Int(walkError)] }
        var d = posixStat(u, followLinks: true)
        // the guest-visible path with every symlink resolved. chdir needs it: the working directory
        // is the DIRECTORY, not the name it was reached by, so getcwd after `cd symlink` must report
        // where the symlink pointed.
        d["real"] = guestPath(u)
        return d
    }

    /// turn a real host url back into the path the guest knows it by (strip the active root).
    private func guestPath(_ url: URL) -> String {
        let root = rootPath.path
        var p = url.path
        if p.hasPrefix(root) { p = String(p.dropFirst(root.count)) }
        if !p.hasPrefix("/") { p = "/" + p }
        return p.count > 1 && p.hasSuffix("/") ? String(p.dropLast()) : p
    }

    /// lstat-style: don't follow symlinks (for fstatat with AT_SYMLINK_NOFOLLOW)
    func lstat(path: String, creds: FsCreds = .root) -> [String: Any] {
        guard let u = resolve(path, follow: false, creds: creds) else { return ["ret": Int(walkError)] }
        return posixStat(u, followLinks: false)
    }

    private func posixStat(_ url: URL, followLinks: Bool) -> [String: Any] {
        var sb = Darwin.stat()
        let result: Int32 = url.path.withCString { followLinks ? statCWrapper($0, &sb) : lstatCWrapper($0, &sb) }
        guard result == 0 else { return ["ret": Int(-ENOENT)] }
        return statDict(sb, meta: effectiveMeta(url, realMode: UInt32(sb.st_mode),
                                               nofollow: (sb.st_mode & S_IFMT) == S_IFLNK && !followLinks))
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
            "rdev": Int(meta.rdev),
            "atime": Int(sb.st_atimespec.tv_sec),
            "ctime": Int(sb.st_ctimespec.tv_sec),
            "mtimeNsec": Int(sb.st_mtimespec.tv_nsec),
            "atimeNsec": Int(sb.st_atimespec.tv_nsec),
            "ctimeNsec": Int(sb.st_ctimespec.tv_nsec),
            "blocks": Int(sb.st_blocks),
            "blksize": Int(sb.st_blksize),
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

            // lstat, not fileExists (which follows symlinks) — otherwise a symlink reports its
            // target's type and never DT_LNK, breaking rm -r / find / ls -F on links. it is also
            // the whole of struct stat in one call, which is what lets the guest's stat cache
            // answer a later stat() of these names without going back to disk.
            var sb = Darwin.stat()
            let statOK = Darwin.lstat(entryURL.path, &sb) == 0
            let dtype: Int
            switch sb.st_mode & S_IFMT {
            case S_IFDIR:  dtype = 4
            case S_IFLNK:  dtype = 10
            case S_IFCHR:  dtype = 2
            case S_IFBLK:  dtype = 6
            case S_IFSOCK: dtype = 12
            case S_IFIFO:  dtype = 1
            default:       dtype = 8   // regular / unknown
            }

            var entry: [String: Any] = ["name": name, "type": dtype]

            // Enrich with stat + emulated ownership so an `ls -l` gets the whole listing's
            // uid/gid/mode in this batch, no per-entry stat storm on the guest side.
            if statOK {
                let m = effectiveMeta(entryURL, realMode: UInt32(sb.st_mode),
                                      nofollow: (sb.st_mode & S_IFMT) == S_IFLNK)
                entry["mode"] = Int(m.mode)
                entry["uid"] = Int(m.uid)
                entry["gid"] = Int(m.gid)
                entry["size"] = Int(sb.st_size)
                entry["nlink"] = Int(sb.st_nlink)
                entry["ino"] = Int(bitPattern: UInt(sb.st_ino))
                entry["mtime"] = Int(sb.st_mtimespec.tv_sec)
                entry["atime"] = Int(sb.st_atimespec.tv_sec)
                entry["ctime"] = Int(sb.st_ctimespec.tv_sec)
                entry["mtimeNsec"] = Int(sb.st_mtimespec.tv_nsec)
                entry["atimeNsec"] = Int(sb.st_atimespec.tv_nsec)
                entry["ctimeNsec"] = Int(sb.st_ctimespec.tv_nsec)
                entry["blocks"] = Int(sb.st_blocks)
                entry["blksize"] = Int(sb.st_blksize)
                entry["rdev"] = Int(m.rdev)
                // a device node is a placeholder regular file on disk, so the host's idea of its
                // type is wrong. d_type is what `ls` colours by and what readdir consumers switch
                // on, so it has to follow the emulated type rather than the file underneath.
                switch m.mode & 0o170000 {
                case 0o020000: entry["type"] = 2   // DT_CHR
                case 0o060000: entry["type"] = 6   // DT_BLK
                default: break
                }
            }

            result.append(entry)
        }

        dirState.position = batchEnd
        openDirs[fd] = dirState
        return (Int32(result.count), result)
    }

    /// the group a newly created name gets.
    ///
    /// normally the creating process's, but a set-group-ID DIRECTORY hands its own group to
    /// everything made inside it -- which is the whole mechanism behind a shared project
    /// directory staying readable by its group no matter who writes into it.
    private func inheritedGid(_ url: URL, _ creds: FsCreds) -> UInt32 {
        let parent = url.deletingLastPathComponent()
        guard let rm = realMode(parent.path) else { return creds.egid }
        let pm = effectiveMeta(parent, realMode: rm)
        return (pm.mode & 0o2000) != 0 ? pm.gid : creds.egid
    }

    func mkdir(path: String, mode: Int32, creds: FsCreds) -> Int32 {
        guard let url = resolve(path, follow: false, creds: creds) else { return walkError }
        if fm.fileExists(atPath: url.path) { return -EEXIST }
        if !mayWriteParent(url, creds) { return -EACCES }
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

    /// the sticky bit on a directory (/tmp is 1777 everywhere) means "you may only remove your own
    /// things here". without it, any user who can write the directory can delete anyone's file in
    /// it, which is exactly what a shared temp directory must not allow. root is exempt.
    private func stickyDenies(_ url: URL, _ creds: FsCreds) -> Bool {
        if creds.euid == 0 { return false }
        let parent = url.deletingLastPathComponent()
        guard let pm = realMode(parent.path) else { return false }
        let pmeta = effectiveMeta(parent, realMode: pm)
        if (pmeta.mode & 0o1000) == 0 { return false }        // no S_ISVTX
        if pmeta.uid == creds.euid { return false }            // owner of the directory
        guard let vm = realMode(url.path, followLinks: false) else { return false }
        // the victim may itself be a symlink, and it is the LINK's owner that decides.
        let victim = effectiveMeta(url, realMode: vm, nofollow: (vm & UInt32(S_IFMT)) == UInt32(S_IFLNK))
        return victim.uid != creds.euid   // owner of the victim
    }

    /// remove a name. `removeDir` is unlinkat's AT_REMOVEDIR, and it is not a detail: unlink and
    /// rmdir are different calls with different rules, and routing both through removeItem meant
    /// unlink(directory) SUCCEEDED and took the whole subtree with it. the order of the checks is
    /// the kernel's own (may_delete): permission on the parent first, then the sticky bit, then the
    /// type mismatch, so a caller that trips more than one gets the errno linux would give.
    func unlink(path: String, removeDir: Bool = false, creds: FsCreds = .root) -> Int32 {
        guard let url = resolve(path, follow: false, creds: creds) else { return walkError }
        if !lexists(url) { return -ENOENT }
        if !mayWriteParent(url, creds) { return -EACCES }
        if stickyDenies(url, creds) { return -EPERM }
        let isDir = (realMode(url.path, followLinks: false).map { $0 & 0o170000 } ?? 0) == 0o040000
        if removeDir && !isDir { return -ENOTDIR }
        if !removeDir && isDir { return -EISDIR }
        // raw unlink(2)/rmdir(2) rather than removeItem: they are non-recursive by construction and
        // they report the real errno (ENOTEMPTY, EBUSY) instead of a blanket failure.
        let rc = url.path.withCString { p in removeDir ? Darwin.rmdir(p) : Darwin.unlink(p) }
        if rc != 0 { return -linuxErrno(errno) }
        cache.evict(path: url.path)
        metaCache.removeValue(forKey: url.path)
        metaCache.removeValue(forKey: "L:" + url.path)
        return 0
    }

    /// darwin's errno for a failed rename, in linux's numbering.
    private func linuxFsErrno(_ e: Int32) -> Int32 {
        switch e {
        case Darwin.EEXIST:    return 17
        case Darwin.ENOTEMPTY: return 39
        case Darwin.EXDEV:     return 18
        case Darwin.ENOENT:    return 2
        case Darwin.EACCES:    return 13
        case Darwin.EPERM:     return 1
        case Darwin.EISDIR:    return 21
        case Darwin.ENOTDIR:   return 20
        case Darwin.EINVAL:    return 22
        case Darwin.ENOSPC:    return 28
        default:               return e
        }
    }

    func rename(oldPath: String, newPath: String, flags: Int = 0, creds: FsCreds = .root) -> Int32 {
        guard let oldURL = resolve(oldPath, follow: false, creds: creds),
              let newURL = resolve(newPath, follow: false, creds: creds) else { return walkError }
        // rename REMOVES a name from one directory and creates one in another, so both ends need
        // the same permission a delete and a create would need -- including the sticky bit.
        if !mayWriteParent(oldURL, creds) || !mayWriteParent(newURL, creds) { return -EACCES }
        if stickyDenies(oldURL, creds) { return -EPERM }
        if lexists(newURL) && stickyDenies(newURL, creds) { return -EPERM }
        // renameat2's two real flags are both atomic operations the kernel underneath already has:
        // "do not replace an existing name" and "swap the two names". doing either by hand -- stat
        // then rename -- would be a race, and the whole point of these flags is that they are not.
        if flags != 0 {
            let RENAME_NOREPLACE = 1, RENAME_EXCHANGE = 2
            var darwinFlags: UInt32 = 0
            if flags & RENAME_NOREPLACE != 0 { darwinFlags |= UInt32(RENAME_EXCL) }
            if flags & RENAME_EXCHANGE != 0 { darwinFlags |= UInt32(RENAME_SWAP) }
            let rc = oldURL.path.withCString { o in
                newURL.path.withCString { n in renamex_np(o, n, darwinFlags) }
            }
            if rc != 0 { return -linuxFsErrno(errno) }
            cache.evict(path: oldURL.path); cache.evict(path: newURL.path)
            metaCache.removeValue(forKey: oldURL.path); metaCache.removeValue(forKey: newURL.path)
            return 0
        }
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

    func symlink(target: String, linkpath: String, creds: FsCreds) -> Int32 {
        guard let linkURL = resolve(linkpath, follow: false, creds: creds) else { return walkError }
        // emulated-credential check: creating the link needs write on the parent dir. the host runs as
        // the app uid so raw symlink(2) wouldn't enforce the guest's euid (symlink03 as nobody).
        let parent = linkURL.deletingLastPathComponent()
        if let rm = realMode(parent.path, followLinks: true) {
            let m = effectiveMeta(parent, realMode: rm)
            if !permits(m, 2, uid: creds.euid, gid: creds.egid, groups: creds.groups) { return -EACCES }
        }
        // raw symlink(2) surfaces the real errno (ENOTDIR/ENOENT/ENAMETOOLONG/EEXIST) instead of a
        // blanket EEXIST. the target is stored verbatim so readlink returns exactly what the guest
        // wrote; resolve() is what keeps a "/..." target inside the root.
        let rc = target.withCString { d in linkURL.path.withCString { l in Darwin.symlink(d, l) } }
        return rc == 0 ? 0 : -linuxErrno(errno)
    }

    // errno values 1-34 are identical on darwin and linux; several above diverge, so a raw darwin
    // syscall's errno must be remapped before it reaches the (linux) guest. covers the codes the fs
    // syscalls here can actually return.
    private func linuxErrno(_ e: Int32) -> Int32 {
        switch e {
        case 35: return 11   // EAGAIN
        case 45: return 95   // ENOTSUP -> EOPNOTSUPP
        case 62: return 40   // ELOOP
        case 63: return 36   // ENAMETOOLONG
        case 66: return 39   // ENOTEMPTY
        case 70: return 116  // ESTALE
        case 78: return 38   // ENOSYS
        case 84: return 75   // EOVERFLOW
        default: return e
        }
    }

    /// create the filesystem name for bind(AF_UNIX).
    ///
    /// this is deliberately NOT a mode that mknod accepts: linux refuses mknod(S_IFSOCK) for the
    /// same reason, because a socket file is made by bind and by nothing else. what lands on disk
    /// is an empty placeholder and the xattr carries S_IFSOCK, so lstat reports a socket -- which
    /// is what `ls -l` shows, what unlink removes, and what a second bind to the same path trips
    /// over. an existing name is EADDRINUSE (98 on linux, not darwin's 48) rather than EEXIST.
    func mksock(path: String, mode: Int32, creds: FsCreds) -> Int32 {
        guard let url = resolve(path, follow: false, creds: creds) else { return walkError }
        if lexists(url) { return -98 }
        let parent = url.deletingLastPathComponent()
        if !fm.fileExists(atPath: parent.path) { return -ENOENT }
        if !mayWriteParent(url, creds) { return -EACCES }
        let perm = (UInt32(bitPattern: mode) & 0o7777) & ~creds.umask
        fm.createFile(atPath: url.path, contents: nil,
                      attributes: [.posixPermissions: NSNumber(value: 0o600)])
        writeMetaXattr(url.path, FileMeta(uid: creds.euid, gid: inheritedGid(url, creds),
                                          mode: 0o140000 | perm, rdev: 0))
        return 0
    }

    func mknod(path: String, mode: Int32, dev: Int32 = 0, creds: FsCreds) -> Int32 {
        guard let url = resolve(path, follow: false, creds: creds) else { return walkError }
        if lexists(url) { return -EEXIST }
        let parent = url.deletingLastPathComponent()
        if !fm.fileExists(atPath: parent.path) { return -ENOENT }
        if !mayWriteParent(url, creds) { return -EACCES }
        let fmt = mode & 0o170000
        let perm = (UInt32(bitPattern: mode) & 0o7777) & ~creds.umask
        if fmt == 0o010000 {
            // named pipe: a real host fifo so lstat reports S_IFIFO and open() gets pipe semantics.
            let rc = url.path.withCString { mkfifo($0, mode_t(perm)) }
            if rc != 0 { return -linuxErrno(errno) }
        } else if fmt == 0o100000 || fmt == 0 {
            fm.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: NSNumber(value: 0o600)])
        } else if fmt == 0o020000 || fmt == 0o060000 {
            // character / block device. the host filesystem cannot hold one, so what lands on disk
            // is an empty placeholder and the xattr carries the type and the device number. making
            // a device node needs CAP_MKNOD, which here means being root.
            if creds.euid != 0 { return -EPERM }
            fm.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: NSNumber(value: 0o600)])
        } else if fmt == 0o140000 {
            // a socket file. mknod(S_IFSOCK) is allowed and needs no privilege -- it makes the
            // NAME, which is all a socket file is; bind() is what attaches a socket to one. this
            // used to answer EPERM, which is the errno for a directory, not for this.
            fm.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: NSNumber(value: 0o600)])
        } else if fmt == 0o040000 {
            return -EPERM   // a directory is mkdir's to make, and linux says EPERM to mknod
        } else {
            return -EINVAL  // not a file type at all
        }
        writeMetaXattr(url.path, FileMeta(uid: creds.euid, gid: inheritedGid(url, creds),
                                          mode: UInt32(bitPattern: fmt) | perm,
                                          rdev: UInt32(bitPattern: dev)))
        return 0
    }

    /// give an unnamed (O_TMPFILE) descriptor a name. linux does this with
    /// linkat(AT_FDCWD, "/proc/self/fd/N", ..., AT_SYMLINK_FOLLOW); the descriptor is the only
    /// handle on the file, so this is the moment it becomes visible, fully written, all at once.
    func linkTmpfile(fd: Int32, newPath: String, creds: FsCreds = .root) -> Int32 {
        guard let hidden = unnamedFiles[fd] else { return -EBADF }
        guard let newURL = resolve(newPath, follow: false, creds: creds) else { return walkError }
        if lexists(newURL) { return -EEXIST }
        if !mayWriteParent(newURL, creds) { return -EACCES }
        // flush anything still buffered before the file becomes reachable under its name
        if let h = openFiles[fd] { try? h.synchronize() }
        do { try fm.moveItem(atPath: hidden, toPath: newURL.path) } catch { return -EIO }
        unnamedFiles.removeValue(forKey: fd)
        openFilePaths[fd] = newURL.path
        return 0
    }

    /// is this descriptor an unnamed O_TMPFILE?
    func isUnnamed(fd: Int32) -> Bool { unnamedFiles[fd] != nil }

    func link(oldPath: String, newPath: String, creds: FsCreds = .root) -> Int32 {
        guard let oldURL = resolve(oldPath, follow: false, creds: creds),
              let newURL = resolve(newPath, follow: false, creds: creds) else { return walkError }
        if !lexists(oldURL) { return -ENOENT }
        if lexists(newURL) { return -EEXIST }
        if !mayWriteParent(newURL, creds) { return -EACCES }
        let rc = oldURL.path.withCString { op in newURL.path.withCString { np in Darwin.link(op, np) } }
        if rc != 0 { return -linuxErrno(errno) }
        return 0
    }

    func readlink(path: String, creds: FsCreds = .root) -> (Int32, String?) {
        guard let url = resolve(path, follow: false, creds: creds) else { return (walkError, nil) }
        // distinguish missing (ENOENT) from exists-but-not-a-symlink (EINVAL): realpath resolves a
        // path component-by-component via readlink and relies on EINVAL to mean "a real file, stop
        // descending". returning ENOENT there made realpath of any native path fail spuriously.
        var sb = Darwin.stat()
        if url.path.withCString({ lstatCWrapper($0, &sb) }) != 0 { return (-ENOENT, nil) }
        if (sb.st_mode & S_IFMT) != S_IFLNK { return (-EINVAL, nil) }
        do {
            return (0, try fm.destinationOfSymbolicLink(atPath: url.path))
        } catch {
            return (-EINVAL, nil)
        }
    }

    /// truncate(2): set a file's length by PATH rather than by descriptor.
    ///
    /// this is not ftruncate with an open in front of it -- the errors differ. truncate follows
    /// symlinks, refuses a directory with EISDIR, needs WRITE permission on the file (not an open
    /// descriptor), and rejects a negative length with EINVAL.
    func truncate(path: String, length: Int64, creds: FsCreds) -> Int32 {
        if length < 0 { return -EINVAL }
        guard let url = resolve(path, creds: creds) else { return walkError }
        guard let rm = realMode(url.path) else { return -ENOENT }
        let m = adopt(url, realMode: rm)
        if (m.mode & 0o170000) == 0o040000 { return -EISDIR }
        if !permits(m, 2, uid: creds.euid, gid: creds.egid, groups: creds.groups) { return -EACCES }
        if url.path.withCString({ Darwin.truncate($0, off_t(length)) }) != 0 {
            return -linuxErrno(errno)
        }
        return 0
    }

    /// `noShrink`/`noGrow` are the caller's file seals: a sealed file may not change size in the
    /// sealed direction, and only here is the current size known to compare against.
    func ftruncate(fd: Int32, length: Int64, noShrink: Bool = false, noGrow: Bool = false) -> Int32 {
        guard let handle = openFiles[fd] else { return -EBADF }
        if noShrink || noGrow {
            var sb = Darwin.stat()
            if Darwin.fstat(handle.fileDescriptor, &sb) == 0 {
                if noShrink && length < Int64(sb.st_size) { return -EPERM }
                if noGrow && length > Int64(sb.st_size) { return -EPERM }
            }
        }
        // ftruncate(2) changes the file's LENGTH and nothing else. FileHandle.truncate(atOffset:)
        // also moves the offset there, so a program that sized a file and then read it from the
        // start read from the end instead and got nothing -- which is what every memfd did, since
        // memfd_create is followed by exactly that ftruncate.
        if Darwin.ftruncate(handle.fileDescriptor, off_t(length)) != 0 { return -linuxErrno(errno) }
        return 0
    }

    /// how long the file behind a handle is right now, for the seals that are defined against it.
    private func fileEnd(_ handle: FileHandle) -> UInt64? {
        var sb = Darwin.stat()
        guard Darwin.fstat(handle.fileDescriptor, &sb) == 0 else { return nil }
        return UInt64(sb.st_size)
    }

    /// fallocate(FALLOC_FL_PUNCH_HOLE): make a range of the file read as zeros without changing
    /// its length. apfs can do it for free when the range is block aligned; otherwise the bytes are
    /// zeroed, which is the same file to every reader and differs only in what st_blocks says.
    func punchHole(fd: Int32, offset: Int64, length: Int64) -> Int32 {
        guard let handle = openFiles[fd] else { return -EBADF }
        if length <= 0 || offset < 0 { return -EINVAL }
        var fp = fpunchhole_t(fp_flags: 0, reserved: 0, fp_offset: off_t(offset), fp_length: off_t(length))
        if Darwin.fcntl(handle.fileDescriptor, F_PUNCHHOLE, &fp) == 0 { return 0 }
        let zeros = [UInt8](repeating: 0, count: 65536)
        var done: Int64 = 0
        while done < length {
            let n = Int(min(Int64(zeros.count), length - done))
            let w = zeros.withUnsafeBytes {
                Darwin.pwrite(handle.fileDescriptor, $0.baseAddress, n, off_t(offset + done))
            }
            if w <= 0 { return -linuxErrno(errno) }
            done += Int64(w)
        }
        return 0
    }

    /// Reopen the file behind a descriptor, the way /proc/<pid>/fd/<n> does.
    ///
    /// that path is a MAGIC LINK: opening it opens the object, not the name. for an anonymous file
    /// -- a memfd, an O_TMPFILE, anything unlinked since it was opened -- there is no name to open,
    /// and following the link text as an ordinary path opens nothing or the wrong thing. the new
    /// descriptor starts at offset zero and is independent of the old one, as a fresh open is.
    func reopen(fd: Int32, flags: Int32, noShrink: Bool = false) -> Int32 {
        guard let handle = openFiles[fd] else { return -EBADF }
        // O_TRUNC here truncates the OBJECT, which is the point: the memfd behind the link shrinks
        // to nothing. that makes it a shrink like any other, so a file sealed against shrinking
        // refuses it -- and refuses it before the descriptor is duplicated.
        if (flags & 0o1000) != 0 && noShrink {
            var sb = Darwin.stat()
            if Darwin.fstat(handle.fileDescriptor, &sb) == 0 && sb.st_size > 0 { return -EPERM }
        }
        let posixFd = Darwin.dup(handle.fileDescriptor)
        guard posixFd >= 0 else { return -linuxErrno(errno) }
        if (flags & 0o1000) != 0 && Darwin.ftruncate(posixFd, 0) != 0 {
            let e = errno; Darwin.close(posixFd); return -linuxErrno(e)
        }
        if (flags & 0o2000) != 0 { _ = Darwin.fcntl(posixFd, F_SETFL, O_APPEND) }
        _ = Darwin.lseek(posixFd, 0, SEEK_SET)
        let newFd = allocFd()
        openFiles[newFd] = FileHandle(fileDescriptor: posixFd, closeOnDealloc: true)
        if let p = openFilePaths[fd] { openFilePaths[newFd] = p }
        return newFd
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
    // access(2) asks the question of the REAL ids -- that is its whole purpose, letting a
    // set-user-ID program check what the user who ran it may reach. AT_EACCESS asks it of the
    // effective ids instead, which is what faccessat2 adds.
    func faccessat(path: String, mode: Int32, creds: FsCreds, eaccess: Bool = false) -> Int32 {
        guard let url = resolve(path, creds: creds) else { return walkError }
        guard let rm = realMode(url.path) else { return -ENOENT }
        if mode == 0 { return 0 } // F_OK: existence
        let m = effectiveMeta(url, realMode: rm)
        let uid = eaccess ? creds.euid : creds.ruid
        let gid = eaccess ? creds.egid : creds.rgid
        return permits(m, UInt32(mode) & 7, uid: uid, gid: gid, groups: creds.groups) ? 0 : -EACCES
    }

    // only the owner (or root) may chmod. the emulated mode lives in the xattr; we never touch
    // the real file's perms (it stays app-accessible).
    func chmod(path: String, mode: Int32, creds: FsCreds) -> Int32 {
        guard let url = resolve(path, creds: creds) else { return walkError }
        guard let rm = realMode(url.path) else { return -ENOENT }
        let m = adopt(url, realMode: rm)
        if creds.euid != 0 && creds.euid != m.uid { return -EPERM }
        var newMode = (m.mode & 0o170000) | (UInt32(bitPattern: mode) & 0o7777)
        // an unprivileged caller cannot set the set-group-ID bit on a file whose group it is not in.
        // the kernel does not refuse the chmod for it -- it silently drops the bit -- which is what
        // stops "chmod g+s" from being a way to acquire a group you do not belong to.
        if creds.euid != 0 && m.gid != creds.egid && !creds.groups.contains(m.gid) {
            newMode &= ~0o2000
        }
        writeMetaXattr(url.path, FileMeta(uid: m.uid, gid: m.gid, mode: newMode))
        return 0
    }

    // chown against the emulated model (kernel setattr_prepare semantics; mirrors chownInode in
    // src/vfs.ts). uid/gid == -1 mean "leave unchanged". a successful chown of a non-directory
    // drops the set-user-ID bit, and set-group-ID if group-executable.
    func chown(path: String, uid: Int32, gid: Int32, creds: FsCreds, nofollow: Bool = false) -> Int32 {
        guard let url = resolve(path, follow: !nofollow, creds: creds) else { return walkError }
        guard let rm = realMode(url.path, followLinks: !nofollow) else { return -ENOENT }
        let linkOwn = linkRecord(url, nofollow)
        let m = adopt(url, realMode: rm, nofollow: linkOwn)
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
        writeMetaXattr(url.path, FileMeta(uid: wantUid, gid: wantGid, mode: newMode), nofollow: linkOwn)
        return 0
    }

    // set a file's access and modification times. UTIME_NOW/UTIME_OMIT travel through untouched --
    // darwin's utimensat understands both -- so "leave the other one alone" stays one operation.
    // setting explicit times is the owner's right; asking only for "now" needs write permission,
    // which is what makes `touch` on someone else's writable file work.
    func utimens(path: String, times: [timespec], creds: FsCreds, nofollow: Bool) -> Int32 {
        guard let url = resolve(path, follow: !nofollow, creds: creds) else { return walkError }
        guard let rm = realMode(url.path, followLinks: !nofollow) else { return -ENOENT }
        let m = effectiveMeta(url, realMode: rm, nofollow: linkRecord(url, nofollow))
        let bothNow = times[0].tv_nsec == UTIME_NOW && times[1].tv_nsec == UTIME_NOW
        if creds.euid != 0 && m.uid != creds.euid {
            if !bothNow { return -EPERM }
            if !permits(m, 2, uid: creds.euid, gid: creds.egid, groups: creds.groups) { return -EACCES }
        }
        var ts = times
        let flag: Int32 = nofollow ? AT_SYMLINK_NOFOLLOW : 0
        let rc = ts.withUnsafeBufferPointer { utimensat(AT_FDCWD, url.path, $0.baseAddress, flag) }
        return rc == 0 ? 0 : -errno
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
