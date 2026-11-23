import Foundation
import os

/// where the memory ceilings actually are, measured rather than assumed.
///
/// two limits matter and they are not the same thing. the footprint limit is jetsam's: exceed it
/// and the kernel kills the process. the address-space limit is the kernel's virtual mapping cap,
/// which a wasm shared memory consumes in full at creation whatever it commits. they also belong
/// to different processes: this file measures OUR process, while the guest's memory lives in
/// WebKit's content process, which has its own of each. the point of the probe is to find out
/// whether the entitlements move either one, and whether the two processes are budgeted apart.
enum MemoryProbe {

    /// bytes remaining before jetsam, per the kernel's own accounting. this is the limit minus
    /// the current footprint, so it already accounts for everything the app is holding.
    static func availableBytes() -> Int {
        return Int(os_proc_available_memory())
    }

    /// current physical footprint, the number jetsam actually compares against the limit
    static func footprintBytes() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int(info.phys_footprint) : -1
    }

    /// virtual size: every mapping, resident or not. this is what an address-space limit bounds.
    static func virtualBytes() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int(info.virtual_size) : -1
    }

    private static func mib(_ bytes: Int) -> String {
        return bytes < 0 ? "?" : String(bytes / 1048576)
    }

    static func report() -> String {
        var out = "device: \(ProcessInfo.processInfo.physicalMemory / 1048576) MiB physical\n"
        out += "app process: footprint \(mib(footprintBytes())) MiB, virtual \(mib(virtualBytes())) MiB\n"
        out += "os_proc_available_memory: \(mib(availableBytes())) MiB\n"
        out += "=> jetsam limit for this process: ~\(mib(availableBytes() + footprintBytes())) MiB\n"
        let ents = ["com.apple.developer.kernel.increased-memory-limit",
                    "com.apple.developer.kernel.extended-virtual-addressing"]
        for e in ents {
            let on = (Bundle.main.object(forInfoDictionaryKey: e) as? Bool) ?? false
            out += "entitlement \(e): \(on ? "present in Info.plist" : "not visible via Info.plist")\n"
        }
        return out
    }

    /// commit memory in this process a step at a time, reporting the kernel's own remaining-bytes
    /// figure as it goes. every page is written so the memory is genuinely resident rather than a
    /// lazy mapping. stops at the guard rather than riding it into a jetsam kill.
    static func commitProbe(stepMiB: Int, guardMiB: Int, capMiB: Int) -> String {
        var blocks: [UnsafeMutableRawPointer] = []
        defer { for b in blocks { free(b) } }
        let step = stepMiB * 1048576
        var out = "commit probe: \(stepMiB) MiB steps, stopping at \(guardMiB) MiB remaining\n"
        out += report()
        var committed = 0
        while committed < capMiB * 1048576 {
            let avail = availableBytes()
            if avail < guardMiB * 1048576 {
                out += "stop: \(mib(avail)) MiB remaining is under the guard\n"
                break
            }
            guard let p = malloc(step) else {
                out += "stop: malloc refused at \(mib(committed)) MiB committed\n"
                break
            }
            // touch a byte per 16 KiB page so the pages are really resident
            var off = 0
            while off < step {
                p.storeBytes(of: UInt8(1), toByteOffset: off, as: UInt8.self)
                off += 16384
            }
            blocks.append(p)
            committed += step
            if committed % (512 * 1048576) == 0 {
                out += "  committed \(mib(committed)) MiB -> available \(mib(availableBytes())) MiB, footprint \(mib(footprintBytes())) MiB\n"
            }
        }
        out += "peak committed: \(mib(committed)) MiB\n"
        out += "at peak: available \(mib(availableBytes())) MiB, footprint \(mib(footprintBytes())) MiB, virtual \(mib(virtualBytes())) MiB\n"
        return out
    }

    /// reserve address space WITHOUT committing it, which is what a shared wasm memory does to the
    /// content process. mmap of anonymous pages that are never written costs virtual size only, so
    /// this finds the address-space ceiling independently of the footprint one.
    static func reserveProbe(stepMiB: Int, capMiB: Int) -> String {
        var blocks: [(UnsafeMutableRawPointer, Int)] = []
        defer { for (p, n) in blocks { munmap(p, n) } }
        let step = stepMiB * 1048576
        var out = "reserve probe: \(stepMiB) MiB steps of untouched anonymous mapping\n"
        var reserved = 0
        while reserved < capMiB * 1048576 {
            let p = mmap(nil, step, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0)
            if p == MAP_FAILED {
                out += "stop: mmap refused at \(mib(reserved)) MiB reserved (errno \(errno))\n"
                break
            }
            blocks.append((p!, step))
            reserved += step
        }
        out += "peak reserved: \(mib(reserved)) MiB\n"
        out += "at peak: virtual \(mib(virtualBytes())) MiB, footprint \(mib(footprintBytes())) MiB, available \(mib(availableBytes())) MiB\n"
        return out
    }
}
