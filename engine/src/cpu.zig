//! guest cpu state layout in linear memory. matches the ts side exactly
//! so existing tests and guest wasm modules keep working during the rewrite.

pub const reg_offset: u32 = 0x00; // x0..x30 at 0x00..0xf0 (31 * 8 = 0xf8)
pub const sp_offset: u32 = 0xf8;
pub const pc_offset: u32 = 0x100;
pub const nzcv_offset: u32 = 0x108;
pub const iter_offset: u32 = 0x110; // self-loop iteration counter
pub const tpidr_offset: u32 = 0x118; // tpidr_el0 (tls base)
// exclusive monitor for ldxr/stxr (arm ARM b2.9.5). a guest store anywhere within
// this granule's range clears the matching shadow bit; STXR verifies bit still set
pub const monitor_addr_offset: u32 = 0x120;
pub const monitor_value_offset: u32 = 0x128;
pub const monitor_size_offset: u32 = 0x130; // bytes (1/2/4/8); 0 = invalid
pub const v_offset: u32 = 0x200; // v0..v31 simd registers, 16 bytes each
pub const state_size: u32 = 0x400;
pub const guest_mem_base: u32 = 0x10000;

// exclusive-reservation-granule. arm ARM permits 8..128, we pick 64 (arm cache line)
pub const erg_log2: u5 = 6; // 1 << 6 == 64
pub const erg_bytes: u32 = 1 << erg_log2;

// halt/svc sentinel pcs used by the dispatcher to detect exit reasons
pub const halt_pc: u64 = 0xdead_0000;
pub const svc_pc: u64 = 0xdead_0001;

pub fn gpr_offset(idx: u5) u32 {
    return reg_offset + @as(u32, idx) * 8;
}

pub fn vreg_offset(idx: u5) u32 {
    return v_offset + @as(u32, idx) * 16;
}

// nzcv layout: bit 31=N, bit 30=Z, bit 29=C, bit 28=V (matches arm64 abi)
pub const nzcv_n: u32 = 1 << 31;
pub const nzcv_z: u32 = 1 << 30;
pub const nzcv_c: u32 = 1 << 29;
pub const nzcv_v: u32 = 1 << 28;
