const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
        .cpu_features_add = std.Target.wasm.featureSet(&.{
            .bulk_memory,
            .nontrapping_fptoint,
            .sign_ext,
            .mutable_globals,
            .reference_types,
            .simd128,
            .multivalue,
        }),
    });

    const translator = b.addExecutable(.{
        .name = "translator",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/translator.zig"),
            .target = wasm_target,
            .optimize = optimize,
        }),
    });
    translator.entry = .disabled;
    translator.rdynamic = true;
    b.installArtifact(translator);

    const test_step = b.step("test", "run tests");
    const native_target = b.standardTargetOptions(.{});

    const src_modules = .{
        .{ "decode", "src/decode.zig" },
        .{ "cpu", "src/cpu.zig" },
    };

    inline for (.{"decode"}) |name| {
        const test_mod = b.createModule(.{
            .root_source_file = b.path("test/" ++ name ++ "_test.zig"),
            .target = native_target,
            .optimize = optimize,
        });
        inline for (src_modules) |entry| {
            const m = b.createModule(.{
                .root_source_file = b.path(entry[1]),
                .target = native_target,
                .optimize = optimize,
            });
            test_mod.addImport(entry[0], m);
        }
        const t = b.addTest(.{ .root_module = test_mod });
        const run_test = b.addRunArtifact(t);
        test_step.dependOn(&run_test.step);
    }
}
