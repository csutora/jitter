import Foundation
import Network

/// Manages network connections for the jitter emulator.
/// Each socket fd maps to an NWConnection (for TCP) or similar.
/// Called from MessageRouter when the JS bridge sends net_* requests.
///
/// Implements a receive buffer per socket, mirroring kernel socket buffers.
/// This allows poll() to check data availability without consuming it,
/// and read()/recvfrom() to drain from the buffer.
final class NetworkBridge {
    private var connections: [Int: NWConnection] = [:]
    private var listeners: [Int: NWListener] = [:]
    // connections that have arrived at a listener and not yet been accepted. the handler is
    // installed when the listener is created rather than when accept is called: a connection that
    // arrives first has to WAIT somewhere, and holding an open bridge call to wait in is what the
    // watchdog ends after twenty seconds.
    private var pendingConns: [Int: [NWConnection]] = [:]
    private var nextId = 1
    private let queue = DispatchQueue(label: "jitter.network", qos: .userInitiated)

    // MARK: - Socket state

    private var socketTypes: [Int: Int] = [:]  // socketId → SOCK_STREAM(1) or SOCK_DGRAM(2)
    private var recvBuffers: [Int: Data] = [:] // socketId → buffered received data
    // sockets whose background receive failed. the error used to be dropped on the floor, so a broken
    // connection produced neither data nor EOF and readers/pollers spun forever; now it is reported to
    // the guest once as EIO.
    private var recvFailed = Set<Int>()
    private var recvEOF: Set<Int> = []         // sockets that have seen EOF/complete
    private var sendBuffers: [Int: Data] = [:] // socketId → buffered outgoing data (TCP write coalescing)
    // sockets with a flush already scheduled, so a burst of small writes still becomes one send.
    private var flushScheduled = Set<Int>()
    // how many guest descriptors name each socket. fork gives the child its own reference to the
    // parent's sockets, and a server that forks a worker and closes its own copy of the listening
    // socket -- which is the ordinary shape of a server -- must not take the listener away from
    // the child that is about to accept on it.
    private var socketRefs: [Int: Int] = [:]

    // ── datagram sockets ────────────────────────────────────────────────────
    // a stream socket's receive buffer is a byte queue, but a datagram socket's is a queue of
    // MESSAGES: recvfrom returns one datagram and discards what did not fit, and two sends are two
    // reads no matter how they were timed. buffering udp into the same Data as tcp lost every
    // boundary, so a test that sends "one"+"two" and then "three" read "onetwothree" once.
    private struct Datagram { let data: Data; let host: String; let port: Int }
    private var datagrams: [Int: [Datagram]] = [:]
    // which local port a socket was bound to.
    private var boundPorts: [Int: Int] = [:]
    // a datagram socket IS a bsd socket here. network.framework models connections, and a udp
    // socket is not one: it has a fixed local port, many peers, message boundaries, and a source
    // address per message. every one of those had to be faked on top of NWConnection, and the last
    // of them cannot be -- a reply has to leave from the port the request arrived on, and a fresh
    // connection never does. the kernel's own socket gives all of it exactly.
    private var udpFds: [Int: Int32] = [:]
    private var udpSources: [Int: DispatchSourceRead] = [:]
    private var socketDomains: [Int: Int] = [:]


    // ── thread confinement ──────────────────────────────────────────────────
    // every bit of socket state (recvBuffers, recvEOF, recvFailed, pendingReceives, sendBuffers,
    // connections) is touched both by NWConnection callbacks (which run on `queue`) and by the bridge
    // handlers (which run on the caller's thread). that was unsynchronized: a receive appending to
    // recvBuffers could interleave with a read reassigning it, and the appended chunk was simply lost —
    // an intermittent short download, worse the faster the data arrived. every entry point now hops onto
    // `queue`, so all state is confined to one serial queue and the callbacks are already on it.

    func socket(domain: Int, type: Int, protocol proto: Int) -> [String: Any] {
        queue.sync { socketImpl(domain: domain, type: type, protocol: proto) }
    }
    func close(socketId: Int) -> [String: Any] {
        queue.sync { closeImpl(socketId: socketId) }
    }
    func connect(socketId: Int, family: Int, host: String, port: Int, completion: @escaping ([String: Any]) -> Void) {
        queue.async { self.connectImpl(socketId: socketId, family: family, host: host, port: port, completion: completion) }
    }
    func bind(socketId: Int, family: Int, host: String, port: Int, completion: @escaping ([String: Any]) -> Void) {
        queue.async { self.bindImpl(socketId: socketId, family: family, host: host, port: port, completion: completion) }
    }
    func sendto(socketId: Int, data: String, host: String, port: Int, more: Bool,
                completion: @escaping ([String: Any]) -> Void) {
        queue.async {
            self.sendtoImpl(socketId: socketId, data: data, host: host, port: port, more: more,
                            completion: completion)
        }
    }
    func poll(socketId: Int, timeoutMs: Int, completion: @escaping ([String: Any]) -> Void) {
        queue.async { self.pollImpl(socketId: socketId, timeoutMs: timeoutMs, completion: completion) }
    }
    func read(socketId: Int, count: Int, nonblock: Bool, peek: Bool,
              completion: @escaping ([String: Any]) -> Void) {
        queue.async {
            self.readImpl(socketId: socketId, count: count, nonblock: nonblock, peek: peek,
                          completion: completion)
        }
    }
    func listen(socketId: Int, backlog: Int, completion: @escaping ([String: Any]) -> Void) {
        queue.async { self.listenImpl(socketId: socketId, backlog: backlog, completion: completion) }
    }
    func accept(socketId: Int, completion: @escaping ([String: Any]) -> Void) {
        queue.async { self.acceptImpl(socketId: socketId, completion: completion) }
    }
    func write(socketId: Int, data: String, more: Bool, completion: @escaping ([String: Any]) -> Void) {
        queue.async { self.writeImpl(socketId: socketId, data: data, more: more, completion: completion) }
    }
    func sendmmsg(socketId: Int, msgs: [[String: Any]], completion: @escaping ([String: Any]) -> Void) {
        queue.async { self.sendmmsgImpl(socketId: socketId, msgs: msgs, index: 0, sent: 0, completion: completion) }
    }
    func recvmmsg(socketId: Int, vlen: Int, maxLen: Int, timeoutMs: Int, completion: @escaping ([String: Any]) -> Void) {
        queue.async {
            self.recvmmsgImpl(socketId: socketId, vlen: vlen, maxLen: maxLen,
                              deadline: .now() + .milliseconds(timeoutMs > 0 ? timeoutMs : 0),
                              blocking: timeoutMs != 0, out: [], completion: completion)
        }
    }

    // MARK: - endpoint helpers

    /// the host and port of an endpoint, with the %en0 scope suffix a link-local address carries
    /// stripped: the guest never sends one and cannot parse one back.
    private func endpointParts(_ ep: NWEndpoint?) -> (String, Int) {
        guard case .hostPort(let host, let port)? = ep else { return ("0.0.0.0", 0) }
        var text = "\(host)"
        if let pct = text.firstIndex(of: "%") { text = String(text[text.startIndex..<pct]) }
        return (text, Int(port.rawValue))
    }

    /// an address as the eight 16-bit groups of a struct sockaddr, so it can travel over the bridge
    /// as numbers rather than a string the response has no room for. ipv4 uses the first two.
    private func addrGroups(_ host: String) -> (family: Int, groups: [Int]) {
        var v4 = in_addr()
        if inet_pton(AF_INET, host, &v4) == 1 {
            let be = v4.s_addr.bigEndian
            return (2, [Int((be >> 16) & 0xFFFF), Int(be & 0xFFFF), 0, 0, 0, 0, 0, 0])
        }
        var v6 = in6_addr()
        if inet_pton(AF_INET6, host, &v6) == 1 {
            var out: [Int] = []
            withUnsafeBytes(of: &v6) { raw in
                for i in 0..<8 { out.append(Int(raw[i * 2]) << 8 | Int(raw[i * 2 + 1])) }
            }
            return (10, out)
        }
        return (0, [0, 0, 0, 0, 0, 0, 0, 0])
    }

    /// MSG_MORE on a datagram socket: hold this payload back and make it the head of the next send.
    /// returns nil when the payload was held, or the whole message when it is time to send.
    private func cork(socketId: Int, payload: Data, more: Bool) -> Data? {
        guard socketTypes[socketId] == 2 else { return payload }
        if more {
            sendBuffers[socketId, default: Data()].append(payload)
            return nil
        }
        guard let corked = sendBuffers.removeValue(forKey: socketId), !corked.isEmpty else { return payload }
        return corked + payload
    }

    // MARK: - Datagram sockets (bsd)

    /// how much a datagram socket will hold before it starts dropping, which is what linux's
    /// default net.core.rmem_default buys.
    private static let udpRecvBufBytes = 212992

    /// darwin's errno numbers are not linux's, and the guest only understands linux's.
    private static func linuxErrno(_ e: Int32) -> Int {
        switch e {
        case EAGAIN:        return 11
        case EADDRINUSE:    return 98
        case EADDRNOTAVAIL: return 99
        case EAFNOSUPPORT:  return 97
        case ECONNREFUSED:  return 111
        case ENETUNREACH:   return 101
        case EHOSTUNREACH:  return 113
        case EHOSTDOWN:     return 112
        case EMSGSIZE:      return 90
        case ENOTCONN:      return 107
        case EDESTADDRREQ:  return 89
        case EISCONN:       return 106
        case ENOBUFS:       return 105
        case EOPNOTSUPP:    return 95
        case EPIPE:         return 32
        case ETIMEDOUT:     return 110
        default:            return Int(e)   // the low numbers agree across both
        }
    }

    /// a sockaddr for this socket's family. an ipv4 literal on an ipv6 socket is carried v4-mapped,
    /// which is how a dual-stack socket reaches an ipv4 peer at all.
    private func sockaddrFor(socketId: Int, host: String, port: Int)
            -> (addr: sockaddr_storage, len: socklen_t)? {
        var storage = sockaddr_storage()
        let v6 = socketDomains[socketId] == 10
        if v6 {
            var sin6 = sockaddr_in6()
            sin6.sin6_len = UInt8(MemoryLayout<sockaddr_in6>.size)
            sin6.sin6_family = sa_family_t(AF_INET6)
            sin6.sin6_port = in_port_t(UInt16(port)).bigEndian
            let text = host.contains(":") || host.isEmpty ? (host.isEmpty ? "::" : host) : "::ffff:" + host
            if inet_pton(AF_INET6, text, &sin6.sin6_addr) != 1 { return nil }
            withUnsafeMutableBytes(of: &storage) { dst in
                withUnsafeBytes(of: &sin6) { src in
                    dst.copyMemory(from: UnsafeRawBufferPointer(rebasing: src[0..<MemoryLayout<sockaddr_in6>.size]))
                }
            }
            return (storage, socklen_t(MemoryLayout<sockaddr_in6>.size))
        }
        var sin = sockaddr_in()
        sin.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        sin.sin_family = sa_family_t(AF_INET)
        sin.sin_port = in_port_t(UInt16(port)).bigEndian
        if inet_pton(AF_INET, host.isEmpty ? "0.0.0.0" : host, &sin.sin_addr) != 1 { return nil }
        withUnsafeMutableBytes(of: &storage) { dst in
            withUnsafeBytes(of: &sin) { src in
                dst.copyMemory(from: UnsafeRawBufferPointer(rebasing: src[0..<MemoryLayout<sockaddr_in>.size]))
            }
        }
        return (storage, socklen_t(MemoryLayout<sockaddr_in>.size))
    }

    /// the host and port a sockaddr names, as text.
    private func addressParts(_ storage: inout sockaddr_storage) -> (String, Int) {
        var buf = [CChar](repeating: 0, count: Int(NI_MAXHOST))
        var port = 0
        withUnsafeBytes(of: &storage) { raw in
            let sa = raw.baseAddress!.assumingMemoryBound(to: sockaddr.self)
            if sa.pointee.sa_family == sa_family_t(AF_INET6) {
                let s6 = raw.baseAddress!.assumingMemoryBound(to: sockaddr_in6.self)
                port = Int(UInt16(bigEndian: s6.pointee.sin6_port))
                var a = s6.pointee.sin6_addr
                inet_ntop(AF_INET6, &a, &buf, socklen_t(buf.count))
            } else {
                let s4 = raw.baseAddress!.assumingMemoryBound(to: sockaddr_in.self)
                port = Int(UInt16(bigEndian: s4.pointee.sin_port))
                var a = s4.pointee.sin_addr
                inet_ntop(AF_INET, &a, &buf, socklen_t(buf.count))
            }
        }
        var text = String(cString: buf)
        if let pct = text.firstIndex(of: "%") { text = String(text[text.startIndex..<pct]) }
        return (text, port)
    }

    private func localPort(fd: Int32) -> Int {
        var storage = sockaddr_storage()
        var len = socklen_t(MemoryLayout<sockaddr_storage>.size)
        let rc = withUnsafeMutablePointer(to: &storage) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &len) }
        }
        if rc < 0 { return 0 }
        return addressParts(&storage).1
    }

    /// keep every datagram that arrives, with the address it came from. the read source fires on
    /// the same queue everything else is confined to, so the queue itself is the lock.
    private func armDatagramReader(socketId: Int, fd: Int32) {
        let src = DispatchSource.makeReadSource(fileDescriptor: fd, queue: queue)
        src.setEventHandler { [weak self] in
            guard let self = self else { return }
            var buf = [UInt8](repeating: 0, count: 65536)
            while true {
                var from = sockaddr_storage()
                var fromLen = socklen_t(MemoryLayout<sockaddr_storage>.size)
                let n = withUnsafeMutablePointer(to: &from) { fp in
                    fp.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                        recvfrom(fd, &buf, buf.count, 0, sa, &fromLen)
                    }
                }
                if n < 0 {
                    if errno != EAGAIN && errno != EINTR { self.recvFailed.insert(socketId) }
                    return
                }
                let (h, p) = self.addressParts(&from)
                // a socket's receive buffer is finite, and a datagram that arrives when it is full
                // is DROPPED -- that is what udp is. keeping every message instead would let a
                // flood grow this queue without limit.
                var queued = self.datagrams[socketId] ?? []
                let held = queued.reduce(0) { $0 + $1.data.count }
                if held + n > NetworkBridge.udpRecvBufBytes || queued.count >= 512 { continue }
                queued.append(Datagram(data: Data(buf[0..<n]), host: h, port: p))
                self.datagrams[socketId] = queued
            }
        }
        src.resume()
        udpSources[socketId] = src
    }

    /// send one datagram, either to a named peer or to whoever the socket was connected to.
    private func sendDatagram(socketId: Int, payload: Data, host: String, port: Int, userLen: Int) -> Int {
        guard let fd = udpFds[socketId] else { return -9 }
        var n = -1
        if host.isEmpty {
            n = payload.withUnsafeBytes { Darwin.send(fd, $0.baseAddress, payload.count, 0) }
        } else {
            guard var sa = sockaddrFor(socketId: socketId, host: host, port: port) else { return -97 }
            n = payload.withUnsafeBytes { p in
                withUnsafePointer(to: &sa.addr) {
                    $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                        Darwin.sendto(fd, p.baseAddress, payload.count, 0, $0, sa.len)
                    }
                }
            }
        }
        if n < 0 { return -NetworkBridge.linuxErrno(errno) }
        // a partial datagram is not a thing: what the caller handed us either went or it did not.
        return userLen
    }

    /// linux's errno for a connection that did not happen.
    private static func connectErrno(_ error: NWError) -> Int {
        guard case .posix(let code) = error else { return -110 } // ETIMEDOUT
        switch code {
        case .ECONNREFUSED: return -111
        case .ENETUNREACH:  return -101
        case .EHOSTUNREACH: return -113
        case .EHOSTDOWN:    return -112
        case .EADDRNOTAVAIL: return -99
        case .EACCES:       return -13
        default:            return -110
        }
    }

    private func peerFields(_ host: String, _ port: Int) -> [String: Any] {
        let a = addrGroups(host)
        return ["peerHost": host, "peerPort": port, "peerFamily": a.family, "peerAddr": a.groups]
    }

    // MARK: - Socket operations

    private func socketImpl(domain: Int, type: Int, protocol proto: Int) -> [String: Any] {
        let id = nextId
        nextId += 1
        socketTypes[id] = type
        socketDomains[id] = domain
        recvBuffers[id] = Data()
        socketRefs[id] = 1
        if type == 2 {
            datagrams[id] = []
            let fd = Darwin.socket(domain == 10 ? AF_INET6 : AF_INET, SOCK_DGRAM, 0)
            if fd < 0 { return ["ret": -NetworkBridge.linuxErrno(errno)] }
            var on: Int32 = 1
            setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, socklen_t(MemoryLayout<Int32>.size))
            _ = fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK)
            udpFds[id] = fd
            armDatagramReader(socketId: id, fd: fd)
        }
        return ["ret": id]
    }

    private func connectImpl(socketId: Int, family: Int, host: String, port: Int, completion: @escaping ([String: Any]) -> Void) {
        if socketTypes[socketId] == 2 {
            guard let fd = udpFds[socketId] else { completion(["ret": -9]); return }
            guard var sa = sockaddrFor(socketId: socketId, host: host, port: port) else {
                completion(["ret": -97]); return
            }
            let rc = withUnsafePointer(to: &sa.addr) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.connect(fd, $0, sa.len) }
            }
            if rc < 0 { completion(["ret": -NetworkBridge.linuxErrno(errno)]); return }
            if boundPorts[socketId] == nil { boundPorts[socketId] = localPort(fd: fd) }
            completion(["ret": 0])
            return
        }
        let nwHost = NWEndpoint.Host(host)
        let nwPort = NWEndpoint.Port(integerLiteral: UInt16(port))
        let isUDP = socketTypes[socketId] == 2
        let params = isUDP ? NWParameters.udp : NWParameters.tcp
        let conn = NWConnection(host: nwHost, port: nwPort, using: params)

        var didComplete = false
        conn.stateUpdateHandler = { [weak self] state in
            guard !didComplete else { return }
            switch state {
            case .ready:
                didComplete = true
                completion(["ret": 0])
            case .failed(let error):
                didComplete = true
                print("[NetworkBridge] connect failed: \(error)")
                completion(["ret": NetworkBridge.connectErrno(error)])
            case .waiting(let error):
                didComplete = true
                print("[NetworkBridge] connect waiting: \(error)")
                // network.framework reports a refused connection as "waiting" and keeps retrying,
                // but the guest asked for a blocking connect and linux would already have said no.
                // reporting every one of these as a timeout hid the difference between a closed
                // port and an unreachable host.
                completion(["ret": NetworkBridge.connectErrno(error)])
            default:
                break
            }
        }

        connections[socketId] = conn
        conn.start(queue: queue)
    }

    /// the network interfaces as the guest should see them: the ones this device actually has.
    /// the guest's sockets ARE these interfaces -- it binds through them and its packets leave from
    /// them -- so anything else here would be a fiction the rest of the system does not share.
    func interfaces() -> [String: Any] {
        var out: [[String: Any]] = []
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddr) == 0, let first = ifaddr else { return ["ret": 0, "ifs": out] }
        defer { freeifaddrs(ifaddr) }
        for ptr in sequence(first: first, next: { $0.pointee.ifa_next }) {
            guard let sa = ptr.pointee.ifa_addr else { continue }
            let fam = sa.pointee.sa_family
            guard fam == UInt8(AF_INET) || fam == UInt8(AF_INET6) else { continue }
            var rawName = String(cString: ptr.pointee.ifa_name)
            // linux calls its loopback "lo", and every program that touches it by name says "lo".
            if rawName == "lo0" { rawName = "lo" }
            let addr = Self.text(of: sa)
            let mask = ptr.pointee.ifa_netmask.map { Self.text(of: $0) } ?? ""
            let bcast = ptr.pointee.ifa_dstaddr.map { Self.text(of: $0) } ?? ""
            out.append([
                "name": rawName,
                "index": Int(if_nametoindex(ptr.pointee.ifa_name)),
                "family": fam == UInt8(AF_INET) ? 2 : 10,
                "flags": Self.linuxIfFlags(ptr.pointee.ifa_flags),
                "addr": addr,
                "netmask": mask,
                "broadcast": bcast,
                "mtu": rawName == "lo" ? 65536 : 1500,
            ])
        }
        return ["ret": 0, "ifs": out]
    }

    private static func text(of sa: UnsafeMutablePointer<sockaddr>) -> String {
        var buf = [CChar](repeating: 0, count: Int(NI_MAXHOST))
        let len = socklen_t(sa.pointee.sa_family == UInt8(AF_INET)
            ? MemoryLayout<sockaddr_in>.size : MemoryLayout<sockaddr_in6>.size)
        guard getnameinfo(sa, len, &buf, socklen_t(buf.count), nil, 0, NI_NUMERICHOST) == 0 else { return "" }
        var text = String(cString: buf)
        if let pct = text.firstIndex(of: "%") { text = String(text[text.startIndex..<pct]) }
        return text
    }

    /// bsd and linux agree on the low interface flags and disagree above them: MULTICAST is 0x8000
    /// on darwin and 0x1000 on linux, and the bits between are darwin's own.
    private static func linuxIfFlags(_ bsd: UInt32) -> Int {
        var out = Int(bsd & 0x3FF)
        if bsd & 0x8000 != 0 { out |= 0x1000 }
        return out
    }

    /// every address this device actually has, plus the two that always work.
    ///
    /// bind() to an address the machine does not hold is EADDRNOTAVAIL -- linux decides it by
    /// asking whether the address is local (inet_addr_type), and so does this. ignoring the host
    /// entirely, which is what this used to do, meant binding 10.3.2.1 "succeeded" and a program
    /// checking which interface it was reachable on was told a lie.
    private func localAddresses() -> Set<String> {
        var found: Set<String> = ["0.0.0.0", "127.0.0.1", "::", "::1"]
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddr) == 0, let first = ifaddr else { return found }
        defer { freeifaddrs(ifaddr) }
        for ptr in sequence(first: first, next: { $0.pointee.ifa_next }) {
            guard let sa = ptr.pointee.ifa_addr else { continue }
            let family = sa.pointee.sa_family
            guard family == UInt8(AF_INET) || family == UInt8(AF_INET6) else { continue }
            var buf = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            let len = socklen_t(family == UInt8(AF_INET)
                ? MemoryLayout<sockaddr_in>.size : MemoryLayout<sockaddr_in6>.size)
            if getnameinfo(sa, len, &buf, socklen_t(buf.count), nil, 0, NI_NUMERICHOST) == 0 {
                var text = String(cString: buf)
                // link-local addresses come back with a %en0 scope suffix the guest never sends.
                if let pct = text.firstIndex(of: "%") { text = String(text[text.startIndex..<pct]) }
                found.insert(text)
            }
        }
        return found
    }

    private func bindImpl(socketId: Int, family: Int, host: String, port: Int, completion: @escaping ([String: Any]) -> Void) {
        if !host.isEmpty && !localAddresses().contains(host) {
            completion(["ret": -99])   // EADDRNOTAVAIL (linux's number, not darwin's 49)
            return
        }
        let isUDP = socketTypes[socketId] == 2
        if isUDP {
            guard let fd = udpFds[socketId] else { completion(["ret": -9]); return }
            guard var sa = sockaddrFor(socketId: socketId, host: host.isEmpty ? "0.0.0.0" : host, port: port) else {
                completion(["ret": -97]) // EAFNOSUPPORT
                return
            }
            let rc = withUnsafePointer(to: &sa.addr) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.bind(fd, $0, sa.len) }
            }
            if rc < 0 { completion(["ret": -NetworkBridge.linuxErrno(errno)]); return }
            boundPorts[socketId] = localPort(fd: fd)
            completion(["ret": 0, "port": boundPorts[socketId] ?? port])
            return
        }
        do {
            let params = NWParameters.tcp
            // a port the guest may bind again after closing, which is what SO_REUSEADDR means to
            // every server that restarts.
            params.allowLocalEndpointReuse = true
            // port 0 means "any", and NWEndpoint.Port has no such value: asking for it built a
            // listener that failed to start, so the bind looked fine and nothing ever connected.
            let listener = port == 0
                ? try NWListener(using: params)
                : try NWListener(using: params, on: NWEndpoint.Port(integerLiteral: UInt16(port)))
            listeners[socketId] = listener
            pendingConns[socketId] = []
            listener.newConnectionHandler = { [weak self] conn in
                guard let self = self else { return }
                self.queue.async {
                    conn.start(queue: self.queue)
                    self.pendingConns[socketId, default: []].append(conn)
                }
            }
            // bind(port 0) asks the system to CHOOSE a port, and the caller then learns which one
            // through getsockname. replying before the listener is ready meant the answer was
            // always the zero it asked with -- so every test that binds a port, reads it back and
            // connects to it was connecting to port 0 and timing out.
            var replied = false
            listener.stateUpdateHandler = { [weak self] state in
                guard let self = self else { return }
                self.queue.async {
                    switch state {
                    case .ready:
                        if !replied {
                            replied = true
                            let bound = Int(listener.port?.rawValue ?? UInt16(port))
                            self.boundPorts[socketId] = bound
                            completion(["ret": 0, "port": bound])
                        }
                    case .failed:
                        if !replied { replied = true; completion(["ret": -98]) } // EADDRINUSE
                    default:
                        break
                    }
                }
            }
            listener.start(queue: queue)
        } catch {
            print("[NetworkBridge] bind failed: \(error)")
            completion(["ret": -98]) // EADDRINUSE
        }
    }

    private func sendtoImpl(socketId: Int, data: String, host: String, port: Int, more: Bool = false,
                            completion: @escaping ([String: Any]) -> Void) {
        guard let raw = Data(base64Encoded: data) else {
            completion(["ret": -22]) // EINVAL
            return
        }
        guard let decoded = cork(socketId: socketId, payload: raw, more: more) else {
            completion(["ret": raw.count])   // held for the send that finishes the message
            return
        }
        let userLen = raw.count

        if socketTypes[socketId] == 2 {
            completion(["ret": sendDatagram(socketId: socketId, payload: decoded, host: host,
                                            port: port, userLen: userLen)])
            return
        }

        // If we already have a connection (from connect() or a previous sendto), reuse it
        if let conn = connections[socketId] {
            conn.send(content: decoded, completion: .contentProcessed { error in
                if let error = error {
                    print("[NetworkBridge] sendto write error: \(error)")
                    completion(["ret": -32]) // EPIPE
                } else {
                    completion(["ret": userLen])
                }
            })
            return
        }

        // No connection yet — create one to the given destination (connectionless UDP pattern)
        let nwHost = NWEndpoint.Host(host)
        let nwPort = NWEndpoint.Port(integerLiteral: UInt16(port))
        let isUDP = socketTypes[socketId] == 2
        let params = isUDP ? NWParameters.udp : NWParameters.tcp
        // a socket that bound a port sends FROM that port, which is how the peer knows where to
        // answer. without this the source port was a fresh ephemeral one every time.
        if let local = boundPorts[socketId], let lp = NWEndpoint.Port(rawValue: UInt16(local)) {
            params.allowLocalEndpointReuse = true
            // the local endpoint has to be of the same family as the destination, or the connection
            // has no route at all -- which is what an ipv6 send from an ipv4-any source reported.
            let anyHost: NWEndpoint.Host = host.contains(":") ? .ipv6(.any) : .ipv4(.any)
            params.requiredLocalEndpoint = NWEndpoint.hostPort(host: anyHost, port: lp)
        }
        let conn = NWConnection(host: nwHost, port: nwPort, using: params)

        var didComplete = false
        conn.stateUpdateHandler = { [weak self] state in
            guard !didComplete else { return }
            switch state {
            case .ready:
                didComplete = true
                conn.send(content: decoded, completion: .contentProcessed { error in
                    if let error = error {
                        print("[NetworkBridge] sendto write error: \(error)")
                        completion(["ret": -32])
                    } else {
                        completion(["ret": userLen])
                    }
                })
                // Start background receive after sending
                if !isUDP { self?.startBackgroundReceive(socketId: socketId) }
            case .failed(let error):
                didComplete = true
                print("[NetworkBridge] sendto connect failed: \(error)")
                completion(["ret": NetworkBridge.connectErrno(error)])
            case .waiting(let error):
                didComplete = true
                print("[NetworkBridge] sendto connect waiting: \(error)")
                completion(["ret": NetworkBridge.connectErrno(error)])
            default:
                break
            }
        }

        connections[socketId] = conn
        conn.start(queue: queue)
    }

    // MARK: - Background receive (fills buffer without blocking caller)

    /// Initiates a non-blocking receive on the NWConnection, buffering data for later poll/read.
    /// Only starts if the buffer is currently empty and no receive is already pending.
    private var pendingReceives: Set<Int> = []

    private func startBackgroundReceive(socketId: Int) {
        // datagram sockets have their own receive loop, and it frames messages. a stream receive on
        // the same connection would take the bytes out of it.
        guard socketTypes[socketId] != 2 else { return }
        // Don't start if already receiving or already have data
        guard !pendingReceives.contains(socketId) else { return }
        guard let buf = recvBuffers[socketId], buf.isEmpty else { return }
        guard !recvEOF.contains(socketId) else { return }
        guard connections[socketId] != nil else { return }

        pendingReceives.insert(socketId)
        // flush any buffered request first — non-blocking clients (apk/openssl) write the request and
        // then poll for the reply without ever issuing a blocking read, so this is the only place the
        // send gets pushed out. without it we wait forever for a response the peer never asked for.
        flushSendBuffer(socketId: socketId) { [weak self] in
            guard let self = self, let conn = self.connections[socketId] else {
                self?.pendingReceives.remove(socketId)
                return
            }
            conn.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
                guard let self = self else { return }
                self.pendingReceives.remove(socketId)

                if let error = error {
                    print("[NetworkBridge] background receive error on \(socketId): \(error)")
                    self.recvFailed.insert(socketId)
                }
                if let data = data, !data.isEmpty {
                    if self.recvBuffers[socketId] != nil {
                        self.recvBuffers[socketId]!.append(data)
                    } else {
                        self.recvBuffers[socketId] = data
                    }
                }
                // For TCP, isComplete means the peer closed. it can be reported ALONGSIDE a chunk while
                // more data is still queued for delivery, so treating it as "nothing further" truncated
                // downloads: we marked EOF, stopped issuing receives, and dropped the tail — which is why
                // a 3.9 MB file arrived as an intermittent 2.8-3.9 MB with a clean (wrong) EOF, and why it
                // was worse for fast/cached responses where the body and the FIN land together. only call
                // it EOF when a receive completes with NO data; otherwise keep draining (the reader re-arms
                // the next receive when it empties the buffer).
                // For UDP, isComplete just means "datagram fully received" — never EOF.
                if isComplete && self.socketTypes[socketId] != 2 && (data?.isEmpty ?? true) {
                    self.recvEOF.insert(socketId)
                }
            }
        }
    }

    // MARK: - Multi-message send and receive

    /// sendmmsg: one datagram per message, in order, stopping at the first failure. linux returns
    /// how many it managed to send and only reports an error if that is none.
    private func sendmmsgImpl(socketId: Int, msgs: [[String: Any]], index: Int, sent: Int,
                              completion: @escaping ([String: Any]) -> Void) {
        if index >= msgs.count {
            completion(["ret": sent])
            return
        }
        let msg = msgs[index]
        let payload = msg["data"] as? String ?? ""
        let host = msg["host"] as? String ?? ""
        let port = msg["port"] as? Int ?? 0
        let next: ([String: Any]) -> Void = { [weak self] result in
            guard let self = self else { return }
            let r = result["ret"] as? Int ?? -5
            if r < 0 {
                completion(["ret": sent > 0 ? sent : r])
                return
            }
            self.sendmmsgImpl(socketId: socketId, msgs: msgs, index: index + 1, sent: sent + 1,
                              completion: completion)
        }
        if host.isEmpty {
            writeImpl(socketId: socketId, data: payload, more: false, completion: next)
        } else {
            sendtoImpl(socketId: socketId, data: payload, host: host, port: port, more: false,
                       completion: next)
        }
    }

    /// recvmmsg: drain up to vlen queued datagrams. the timeout bounds the wait for the FIRST one;
    /// once something has arrived, whatever else is already queued comes back with it.
    private func recvmmsgImpl(socketId: Int, vlen: Int, maxLen: Int, deadline: DispatchTime,
                              blocking: Bool, out: [[String: Any]],
                              completion: @escaping ([String: Any]) -> Void) {
        var collected = out
        while collected.count < vlen, datagrams[socketId]?.isEmpty == false {
            let dg = datagrams[socketId]!.removeFirst()
            let chunk = dg.data.prefix(maxLen)
            collected.append(["data": chunk.base64EncodedString(), "host": dg.host, "port": dg.port])
        }
        if collected.count >= vlen || !collected.isEmpty {
            completion(["ret": collected.count, "msgs": collected])
            return
        }
        if recvFailed.contains(socketId) { recvFailed.remove(socketId); completion(["ret": -5]); return }
        if udpFds[socketId] == nil && connections[socketId] == nil { completion(["ret": -9]); return }
        if !blocking || DispatchTime.now() >= deadline {
            completion(["ret": -11]) // EAGAIN
            return
        }
        queue.asyncAfter(deadline: .now() + .milliseconds(2)) { [weak self] in
            self?.recvmmsgImpl(socketId: socketId, vlen: vlen, maxLen: maxLen, deadline: deadline,
                               blocking: blocking, out: collected, completion: completion)
        }
    }

    // MARK: - Poll (check buffer, optionally wait with timeout)

    private func pollImpl(socketId: Int, timeoutMs: Int, completion: @escaping ([String: Any]) -> Void) {
        // always push any buffered request out first. non-blocking clients (apk/openssl over TLS) write
        // the request then poll for the reply without a blocking read, and startBackgroundReceive skips
        // the flush when a receive is already pending — so this is the only point the send is guaranteed.
        flushSendBuffer(socketId: socketId) { }
        // Check if data is already buffered
        if let buf = recvBuffers[socketId], !buf.isEmpty {
            completion(["ret": 1])
            return
        }
        if datagrams[socketId]?.isEmpty == false {
            completion(["ret": 1])
            return
        }
        // a listening socket is readable when a connection is waiting to be accepted, which is how
        // a server that selects before accepting learns there is one.
        if pendingConns[socketId]?.isEmpty == false {
            completion(["ret": 1])
            return
        }
        if recvEOF.contains(socketId) || recvFailed.contains(socketId) {
            completion(["ret": 1]) // EOF (or a failed connection) is a readable condition
            return
        }

        // No data yet. If timeout is 0, return immediately.
        if timeoutMs == 0 {
            completion(["ret": 0])
            return
        }

        // Ensure a background receive is running
        startBackgroundReceive(socketId: socketId)

        // Poll with retry until data arrives or timeout expires
        let deadline = DispatchTime.now() + .milliseconds(timeoutMs > 0 ? timeoutMs : 30000)
        pollRetry(socketId: socketId, deadline: deadline, completion: completion)
    }

    private func pollRetry(socketId: Int, deadline: DispatchTime, completion: @escaping ([String: Any]) -> Void) {
        // Check if data arrived
        if let buf = recvBuffers[socketId], !buf.isEmpty {
            completion(["ret": 1])
            return
        }
        if datagrams[socketId]?.isEmpty == false || pendingConns[socketId]?.isEmpty == false {
            completion(["ret": 1])
            return
        }
        if recvEOF.contains(socketId) || recvFailed.contains(socketId) {
            completion(["ret": 1])
            return
        }
        // Check timeout
        if DispatchTime.now() >= deadline {
            completion(["ret": 0])
            return
        }
        // Retry after a short delay
        queue.asyncAfter(deadline: .now() + .milliseconds(5)) { [weak self] in
            self?.pollRetry(socketId: socketId, deadline: deadline, completion: completion)
        }
    }

    // MARK: - Read (drain from buffer)

    private func readImpl(socketId: Int, count: Int, nonblock: Bool = false, peek: Bool = false,
                          completion: @escaping ([String: Any]) -> Void) {
        // a datagram socket reads one message, and what did not fit is discarded rather than kept
        // for the next read: that is the difference between a datagram and a stream.
        if socketTypes[socketId] == 2 {
            if datagrams[socketId]?.isEmpty == false {
                // MSG_PEEK looks without taking: the next read must see the same message again.
                let dg = peek ? datagrams[socketId]![0] : datagrams[socketId]!.removeFirst()
                let chunk = dg.data.prefix(count)
                var reply: [String: Any] = ["ret": chunk.count, "data": chunk.base64EncodedString()]
                reply.merge(peerFields(dg.host, dg.port)) { a, _ in a }
                completion(reply)
                return
            }
            if recvFailed.contains(socketId) { recvFailed.remove(socketId); completion(["ret": -5]); return }
            guard udpFds[socketId] != nil else {
                completion(["ret": -9]) // EBADF
                return
            }
            completion(["ret": -11]) // EAGAIN -- the guest owns the blocking, and keeps asking
            return
        }

        // If buffer has data, return from it
        if let buf = recvBuffers[socketId], !buf.isEmpty {
            let readCount = min(count, buf.count)
            let chunk = buf.prefix(readCount)
            if !peek { recvBuffers[socketId] = Data(buf.dropFirst(readCount)) }
            let b64 = chunk.base64EncodedString()
            completion(["ret": readCount, "data": b64])

            // If buffer is now empty, start another background receive for next time
            if recvBuffers[socketId]?.isEmpty == true {
                startBackgroundReceive(socketId: socketId)
            }
            return
        }

        // Buffer empty — check EOF
        if recvEOF.contains(socketId) {
            completion(["ret": 0]) // EOF
            return
        }

        // Buffer empty, not EOF — for UDP return EAGAIN, for TCP block
        guard connections[socketId] != nil else {
            completion(["ret": -9]) // EBADF
            return
        }

        // TCP, no data yet. do NOT issue our own conn.receive here: Network.framework allows only ONE
        // outstanding receive per connection, and startBackgroundReceive may already have one in flight
        // (a read that drains the buffer arms one on its way out). a second, overlapping receive fails
        // the connection, which surfaced to the guest as EIO — and because it is a race it only bit
        // sustained transfers, e.g. a multi-MB download dying around 88%. instead: make sure the single
        // background receive is running and wait for it to fill the buffer.
        flushSendBuffer(socketId: socketId) { }
        startBackgroundReceive(socketId: socketId)
        // a read that must not wait says so and gets told there is nothing, rather than parking the
        // caller for two minutes: that is what O_NONBLOCK and MSG_DONTWAIT are for.
        if nonblock { completion(["ret": -11]); return }
        readRetry(socketId: socketId, count: count, deadline: .now() + .seconds(120), completion: completion)
    }

    /// wait for the (single) background receive to deliver, then serve the read from the buffer.
    private func readRetry(socketId: Int, count: Int, deadline: DispatchTime, completion: @escaping ([String: Any]) -> Void) {
        if let buf = recvBuffers[socketId], !buf.isEmpty {
            let readCount = min(count, buf.count)
            let chunk = buf.prefix(readCount)
            recvBuffers[socketId] = Data(buf.dropFirst(readCount))
            completion(["ret": readCount, "data": chunk.base64EncodedString()])
            if recvBuffers[socketId]?.isEmpty == true { startBackgroundReceive(socketId: socketId) }
            return
        }
        if recvFailed.contains(socketId) { recvFailed.remove(socketId); completion(["ret": -5]); return } // EIO, reported once
        if recvEOF.contains(socketId) { completion(["ret": 0]); return }
        if connections[socketId] == nil { completion(["ret": -9]); return }                                // EBADF
        if DispatchTime.now() >= deadline { completion(["ret": -5]); return }                              // backstop: never wedge forever
        queue.asyncAfter(deadline: .now() + .milliseconds(2)) { [weak self] in
            guard let self = self else { return }
            self.startBackgroundReceive(socketId: socketId) // re-arm if the last one completed empty
            self.readRetry(socketId: socketId, count: count, deadline: deadline, completion: completion)
        }
    }

    private func listenImpl(socketId: Int, backlog: Int, completion: @escaping ([String: Any]) -> Void) {
        guard listeners[socketId] != nil else {
            completion(["ret": -9]) // EBADF
            return
        }
        completion(["ret": 0])
    }

    private func acceptImpl(socketId: Int, completion: @escaping ([String: Any]) -> Void) {
        guard listeners[socketId] != nil else {
            completion(["ret": -9]) // EBADF
            return
        }
        // nobody waiting is EAGAIN, not a call that never returns: the guest owns the blocking,
        // and it can keep asking (and be interrupted by a signal while it does).
        guard let conn = pendingConns[socketId]?.first, !(pendingConns[socketId]?.isEmpty ?? true) else {
            completion(["ret": -11]) // EAGAIN
            return
        }
        pendingConns[socketId]?.removeFirst()
        let newId = nextId
        nextId += 1
        connections[newId] = conn
        recvBuffers[newId] = Data()
        socketTypes[newId] = socketTypes[socketId] ?? 1
        socketRefs[newId] = 1
        startBackgroundReceive(socketId: newId)
        let (h, p) = endpointParts(conn.endpoint)
        var reply: [String: Any] = ["ret": newId]
        reply.merge(peerFields(h, p)) { a, _ in a }
        completion(reply)
    }

    private func writeImpl(socketId: Int, data: String, more: Bool = false,
                           completion: @escaping ([String: Any]) -> Void) {
        guard connections[socketId] != nil || udpFds[socketId] != nil else {
            completion(["ret": -9]) // EBADF
            return
        }
        guard let raw = Data(base64Encoded: data) else {
            completion(["ret": -22]) // EINVAL
            return
        }
        // a datagram corked with MSG_MORE is not sent yet; it becomes part of the one that follows.
        // a stream does its own coalescing further down, where the buffer is also the flush queue.
        var decoded = raw
        if socketTypes[socketId] == 2 {
            guard let whole = cork(socketId: socketId, payload: raw, more: more) else {
                completion(["ret": raw.count])
                return
            }
            decoded = whole
        }
        let userLen = raw.count

        // TCP: coalesce writes (apps like wget send headers line-by-line) -- but only for as long
        // as it takes the caller to issue the next one. the buffer used to be flushed ONLY when
        // the same socket started a receive, so a program that writes without reading (which is
        // every server answering a request, and every client streaming a body) never sent anything.
        if socketTypes[socketId] == 1 {
            if sendBuffers[socketId] == nil { sendBuffers[socketId] = Data() }
            sendBuffers[socketId]!.append(decoded)
            // MSG_MORE says more of this message is coming, so hold it until a send that does not
            // say so. without this the "more" was flushed 2 ms later and the peer saw two messages.
            if more { completion(["ret": decoded.count]); return }
            if !flushScheduled.contains(socketId) {
                flushScheduled.insert(socketId)
                queue.asyncAfter(deadline: .now() + .milliseconds(2)) { [weak self] in
                    guard let self = self else { return }
                    self.flushScheduled.remove(socketId)
                    self.flushSendBuffer(socketId: socketId) {}
                }
            }
            completion(["ret": decoded.count])
            return
        }

        // UDP: send immediately, to whoever this socket was connected to
        completion(["ret": sendDatagram(socketId: socketId, payload: decoded, host: "", port: 0,
                                        userLen: userLen)])
    }

    /// Flush buffered TCP writes to the connection.
    private func flushSendBuffer(socketId: Int, completion: @escaping () -> Void) {
        guard let data = sendBuffers.removeValue(forKey: socketId), !data.isEmpty,
              let conn = connections[socketId] else {
            completion()
            return
        }
        conn.send(content: data, completion: .contentProcessed { error in
            if let error = error {
                print("[NetworkBridge] flush write error: \(error)")
            }
            completion()
        })
    }

    private func closeImpl(socketId: Int) -> [String: Any] {
        // one descriptor's close is not the socket's end while another process still holds it.
        let refs = (socketRefs[socketId] ?? 1) - 1
        if refs > 0 { socketRefs[socketId] = refs; return ["ret": 0] }
        socketRefs.removeValue(forKey: socketId)
        if let conn = connections.removeValue(forKey: socketId) {
            // close(2) on a stream still sends what the program already wrote. cancelling with a
            // buffered write outstanding threw it away, so a server that answered a request and
            // then closed the connection delivered nothing but the eof.
            if let pending = sendBuffers.removeValue(forKey: socketId), !pending.isEmpty {
                conn.send(content: pending, isComplete: true,
                          completion: .contentProcessed { _ in conn.cancel() })
            } else {
                conn.cancel()
            }
        }
        if let listener = listeners.removeValue(forKey: socketId) {
            listener.cancel()
        }
        socketTypes.removeValue(forKey: socketId)
        recvBuffers.removeValue(forKey: socketId)
        recvFailed.remove(socketId)
        sendBuffers.removeValue(forKey: socketId)
        recvEOF.remove(socketId)
        pendingReceives.remove(socketId)
        pendingConns.removeValue(forKey: socketId)
        flushScheduled.remove(socketId)
        datagrams.removeValue(forKey: socketId)
        boundPorts.removeValue(forKey: socketId)
        socketDomains.removeValue(forKey: socketId)
        if let src = udpSources.removeValue(forKey: socketId) { src.cancel() }
        if let fd = udpFds.removeValue(forKey: socketId) { Darwin.close(fd) }
        return ["ret": 0]
    }

    /// take another reference to a socket, for a child that inherited the descriptor.
    func ref(socketId: Int) -> [String: Any] {
        queue.sync {
            socketRefs[socketId] = (socketRefs[socketId] ?? 1) + 1
            return ["ret": 0]
        }
    }
}
