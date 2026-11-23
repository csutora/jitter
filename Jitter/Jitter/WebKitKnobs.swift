import Foundation
import WebKit

/// enumerate the settings WebKit exposes at runtime, so "is there a knob for this" is answered by
/// asking the framework on the device rather than by reading source that may not match the build
/// shipped here. _features() is SPI, so everything goes through the objc runtime: a selector that
/// is not there simply reports absent instead of failing to link.
enum WebKitKnobs {

    private static func featureList() -> [NSObject] {
        let sel = NSSelectorFromString("_features")
        guard WKPreferences.responds(to: sel) else { return [] }
        guard let r = (WKPreferences.self as AnyObject).perform(sel) else { return [] }
        return (r.takeUnretainedValue() as? [NSObject]) ?? []
    }

    /// dump every feature whose name or key matches `filter` (empty matches all), with its default
    static func features(matching filter: String) -> String {
        let all = featureList()
        if all.isEmpty { return "WKPreferences._features() unavailable on this build\n" }
        let want = filter.lowercased()
        var lines: [String] = []
        for f in all {
            let key = (f.value(forKey: "key") as? String) ?? "?"
            let name = (f.value(forKey: "name") as? String) ?? "?"
            let def = (f.value(forKey: "defaultValue") as? Bool).map { $0 ? "on" : "off" } ?? "?"
            if want.isEmpty || key.lowercased().contains(want) || name.lowercased().contains(want) {
                lines.append("  \(key) = \(def)   -- \(name)")
            }
        }
        return "\(all.count) features total, \(lines.count) matching \"\(filter)\"\n" + lines.sorted().joined(separator: "\n") + "\n"
    }

    /// which of the memory-shaped SPI actually exists on WKWebViewConfiguration, WKProcessPool and
    /// its configuration. presence is what decides whether a knob is reachable at all.
    static func probeSelectors() -> String {
        var out = "selector presence:\n"
        func check(_ cls: AnyClass, _ names: [String]) {
            for n in names {
                let sel = NSSelectorFromString(n)
                let inst = cls.instancesRespond(to: sel)
                let meta = (cls as AnyObject).responds(to: sel)
                out += "  \(NSStringFromClass(cls)).\(n): \(inst || meta ? "present" : "absent")\n"
            }
        }
        check(WKWebViewConfiguration.self, [
            "_setCPULimit:", "_cpuLimit",
            "_setAllowUniversalAccessFromFileURLs:",
            "setLimitsNavigationsToAppBoundDomains:",
        ])
        check(WKProcessPool.self, [
            "_setObjectStorageLimit:", "_webProcessCount",
        ])
        if let poolCfgCls = NSClassFromString("_WKProcessPoolConfiguration") {
            check(poolCfgCls, [
                "setJavaScriptConfigurationDirectory:",
                "javaScriptConfigurationDirectory",
                "setMemoryFootprintPollIntervalForTesting:",
                "setMemoryFootprintNotificationThresholds:",
                "setShouldTakeUIBackgroundAssertion:",
                "setUsesSingleWebProcess:",
                "setProcessSwapsOnNavigation:",
            ])
        } else {
            out += "  _WKProcessPoolConfiguration: class not found\n"
        }
        if let prefCls = NSClassFromString("WKPreferences") {
            check(prefCls, ["_setEnabled:forFeature:", "_isEnabledForFeature:"])
        }
        return out
    }
}
