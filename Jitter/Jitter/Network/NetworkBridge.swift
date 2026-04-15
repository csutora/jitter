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
    private var nextId = 1
    private let queue = DispatchQueue(label: "jitter.network", qos: .userInitiated)

    // MARK: - Socket state

    private var socketTypes: [Int: Int] = [:]  // socketId → SOCK_STREAM(1) or SOCK_DGRAM(2)
    private var recvBuffers: [Int: Data] = [:] // socketId → buffered received data
    private var recvEOF: Set<Int> = []         // sockets that have seen EOF/complete
    private var sendBuffers: [Int: Data] = [:] // socketId → buffered outgoing data (TCP write coalescing)

    // MARK: - Socket operations

    func socket(domain: Int, type: Int, protocol proto: Int) -> [String: Any] {
        let id = nextId
        nextId += 1
        socketTypes[id] = type
        recvBuffers[id] = Data()
        return ["ret": id]
    }

    func connect(socketId: Int, family: Int, host: String, port: Int, completion: @escaping ([String: Any]) -> Void) {
        let nwHost = NWEndpoint.Host(host)
        let nwPort = NWEndpoint.Port(integerLiteral: UInt16(port))
        let isUDP = socketTypes[socketId] == 2
        let params = isUDP ? NWParameters.udp : NWParameters.tcp
        let conn = NWConnection(host: nwHost, port: nwPort, using: params)

        var didComplete = false
        conn.stateUpdateHandler = { state in
            guard !didComplete else { return }
            switch state {
            case .ready:
                didComplete = true
                completion(["ret": 0])
            case .failed(let error):
                didComplete = true
                print("[NetworkBridge] connect failed: \(error)")
                completion(["ret": -111]) // ECONNREFUSED
            case .waiting(let error):
                didComplete = true
                print("[NetworkBridge] connect waiting: \(error)")
                completion(["ret": -110]) // ETIMEDOUT
            default:
                break
            }
        }

        connections[socketId] = conn
        conn.start(queue: queue)
    }

    func bind(socketId: Int, family: Int, host: String, port: Int, completion: @escaping ([String: Any]) -> Void) {
        let isUDP = socketTypes[socketId] == 2
        if isUDP {
            // UDP bind is a no-op — NWConnection picks an ephemeral source port automatically
            completion(["ret": 0])
            return
        }
        do {
            let params = NWParameters.tcp
            let listener = try NWListener(using: params, on: NWEndpoint.Port(integerLiteral: UInt16(port)))
            listeners[socketId] = listener
            listener.start(queue: queue)
            completion(["ret": 0])
        } catch {
            print("[NetworkBridge] bind failed: \(error)")
            completion(["ret": -98]) // EADDRINUSE
        }
    }

    func sendto(socketId: Int, data: String, host: String, port: Int, completion: @escaping ([String: Any]) -> Void) {
        guard let decoded = Data(base64Encoded: data) else {
            completion(["ret": -22]) // EINVAL
            return
        }

        // If we already have a connection (from connect() or a previous sendto), reuse it
        if let conn = connections[socketId] {
            conn.send(content: decoded, completion: .contentProcessed { error in
                if let error = error {
                    print("[NetworkBridge] sendto write error: \(error)")
                    completion(["ret": -32]) // EPIPE
                } else {
                    completion(["ret": decoded.count])
                }
            })
            // After sending on UDP, start a background receive to fill the buffer
            self.startBackgroundReceive(socketId: socketId)
            return
        }

        // No connection yet — create one to the given destination (connectionless UDP pattern)
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
                conn.send(content: decoded, completion: .contentProcessed { error in
                    if let error = error {
                        print("[NetworkBridge] sendto write error: \(error)")
                        completion(["ret": -32])
                    } else {
                        completion(["ret": decoded.count])
                    }
                })
                // Start background receive after sending
                self?.startBackgroundReceive(socketId: socketId)
            case .failed(let error):
                didComplete = true
                print("[NetworkBridge] sendto connect failed: \(error)")
                completion(["ret": -111]) // ECONNREFUSED
            case .waiting(let error):
                didComplete = true
                print("[NetworkBridge] sendto connect waiting: \(error)")
                completion(["ret": -101]) // ENETUNREACH
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
        // Don't start if already receiving or already have data
        guard !pendingReceives.contains(socketId) else { return }
        guard let buf = recvBuffers[socketId], buf.isEmpty else { return }
        guard !recvEOF.contains(socketId) else { return }
        guard let conn = connections[socketId] else { return }

        pendingReceives.insert(socketId)
        conn.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            guard let self = self else { return }
            self.pendingReceives.remove(socketId)

            if let data = data, !data.isEmpty {
                if self.recvBuffers[socketId] != nil {
                    self.recvBuffers[socketId]!.append(data)
                } else {
                    self.recvBuffers[socketId] = data
                }
            }
            // For TCP, isComplete means connection closed (EOF).
            // For UDP, isComplete just means "datagram fully received" — NOT EOF.
            if isComplete && self.socketTypes[socketId] != 2 {
                self.recvEOF.insert(socketId)
            }
        }
    }

    // MARK: - Poll (check buffer, optionally wait with timeout)

    func poll(socketId: Int, timeoutMs: Int, completion: @escaping ([String: Any]) -> Void) {
        // Check if data is already buffered
        if let buf = recvBuffers[socketId], !buf.isEmpty {
            completion(["ret": 1])
            return
        }
        if recvEOF.contains(socketId) {
            completion(["ret": 1]) // EOF is also a readable condition
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
        if recvEOF.contains(socketId) {
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

    func read(socketId: Int, count: Int, completion: @escaping ([String: Any]) -> Void) {
        // If buffer has data, return from it
        if let buf = recvBuffers[socketId], !buf.isEmpty {
            let readCount = min(count, buf.count)
            let chunk = buf.prefix(readCount)
            recvBuffers[socketId] = Data(buf.dropFirst(readCount))
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

        if socketTypes[socketId] == 2 {
            // UDP: no data available right now → EAGAIN
            // Start background receive so poll can detect when data arrives
            startBackgroundReceive(socketId: socketId)
            completion(["ret": -11]) // EAGAIN
            return
        }

        // TCP: flush any buffered writes first, then block until data arrives
        flushSendBuffer(socketId: socketId) { [weak self] in
            guard let self = self, let conn = self.connections[socketId] else {
                completion(["ret": -9])
                return
            }
            conn.receive(minimumIncompleteLength: 1, maximumLength: count) { [weak self] data, _, isComplete, error in
                guard let self = self else { return }
                if let error = error {
                    print("[NetworkBridge] read error: \(error)")
                    completion(["ret": -5]) // EIO
                    return
                }
                if let data = data, !data.isEmpty {
                    let b64 = data.base64EncodedString()
                    completion(["ret": data.count, "data": b64])
                } else if isComplete {
                    self.recvEOF.insert(socketId)
                    completion(["ret": 0]) // EOF
                } else {
                    completion(["ret": 0])
                }
            }
        }
    }

    func listen(socketId: Int, backlog: Int, completion: @escaping ([String: Any]) -> Void) {
        guard listeners[socketId] != nil else {
            completion(["ret": -9]) // EBADF
            return
        }
        completion(["ret": 0])
    }

    func accept(socketId: Int, completion: @escaping ([String: Any]) -> Void) {
        guard let listener = listeners[socketId] else {
            completion(["ret": -9]) // EBADF
            return
        }
        listener.newConnectionHandler = { [weak self] conn in
            guard let self = self else { return }
            let newId = self.nextId
            self.nextId += 1
            self.connections[newId] = conn
            self.recvBuffers[newId] = Data()
            conn.start(queue: self.queue)
            completion(["ret": newId])
        }
    }

    func write(socketId: Int, data: String, completion: @escaping ([String: Any]) -> Void) {
        guard connections[socketId] != nil else {
            completion(["ret": -9]) // EBADF
            return
        }
        guard let decoded = Data(base64Encoded: data) else {
            completion(["ret": -22]) // EINVAL
            return
        }

        // TCP: buffer writes and coalesce them (apps like wget send headers line-by-line)
        if socketTypes[socketId] == 1 {
            if sendBuffers[socketId] == nil { sendBuffers[socketId] = Data() }
            sendBuffers[socketId]!.append(decoded)
            completion(["ret": decoded.count])
            return
        }

        // UDP: send immediately
        connections[socketId]!.send(content: decoded, completion: .contentProcessed { error in
            if let error = error {
                print("[NetworkBridge] write error: \(error)")
                completion(["ret": -32]) // EPIPE
            } else {
                completion(["ret": decoded.count])
            }
        })
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

    func close(socketId: Int) -> [String: Any] {
        if let conn = connections.removeValue(forKey: socketId) {
            conn.cancel()
        }
        if let listener = listeners.removeValue(forKey: socketId) {
            listener.cancel()
        }
        socketTypes.removeValue(forKey: socketId)
        recvBuffers.removeValue(forKey: socketId)
        sendBuffers.removeValue(forKey: socketId)
        recvEOF.remove(socketId)
        pendingReceives.remove(socketId)
        return ["ret": 0]
    }
}
