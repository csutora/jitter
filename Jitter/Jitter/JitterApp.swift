//
//  JitterApp.swift
//  Jitter
//
//  Created by numi on 2026. 02. 26..
//

import SwiftUI
import GhosttyVt

/// every file a guest process opens costs a real descriptor in THIS process, and every guest
/// process has its own budget of 1024. the default soft limit is a couple of hundred, so a handful
/// of busy guests exhaust it and NativeFS starts failing opens for reasons that have nothing to do
/// with the guest that hit the wall. raise the soft limit to whatever the hard limit allows.
private func raiseDescriptorLimit() {
    var lim = rlimit()
    guard getrlimit(RLIMIT_NOFILE, &lim) == 0 else { return }
    let before = lim.rlim_cur
    lim.rlim_cur = lim.rlim_max
    if setrlimit(RLIMIT_NOFILE, &lim) != 0 {
        // some systems refuse RLIM_INFINITY here and want a concrete number instead
        lim.rlim_cur = 65536
        _ = setrlimit(RLIMIT_NOFILE, &lim)
    }
    var now = rlimit()
    _ = getrlimit(RLIMIT_NOFILE, &now)
    print("[Jitter] RLIMIT_NOFILE \(before) -> \(now.rlim_cur) (hard \(now.rlim_max))")
}

@main
struct JitterApp: App {
    init() {
        raiseDescriptorLimit()
        var version = GhosttyString(ptr: nil, len: 0)
        if ghostty_build_info(GHOSTTY_BUILD_INFO_VERSION_STRING, &version) == GHOSTTY_SUCCESS,
           let ptr = version.ptr {
            let bytes = UnsafeBufferPointer(start: ptr, count: version.len)
            let s = String(decoding: bytes, as: UTF8.self)
            print("[Jitter] linked against libghostty-vt \(s)")
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
