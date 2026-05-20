//
//  JitterApp.swift
//  Jitter
//
//  Created by numi on 2026. 02. 26..
//

import SwiftUI
import GhosttyVt

@main
struct JitterApp: App {
    init() {
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
