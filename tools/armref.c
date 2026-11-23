// executes one aarch64 instruction natively and reports the architectural state it produced.
// this is the reference side of the differential fuzzer: the host is Apple Silicon, so "what
// does this instruction really do" can be answered by the hardware rather than by reading the
// manual and hoping.
//
// protocol on stdin/stdout, little-endian binary, one record in -> one record out:
//   in:  u32 insn, i32 base_reg, State
//   out: u32 status (0 = ran, 1 = trapped), u32 pad, State
// when base_reg >= 0 that register is pointed at State.scratch just before the instruction
// runs, so load/store forms address a buffer both sides can compare afterwards.
// State is 28 x-registers, nzcv, then 28 v-registers. x28..x30 and the v registers above 27 are
// reserved for the harness, so generated instructions must only name 0..27.
//
// build: cc -O2 -o armref tools/armref.c   (macOS arm64)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <setjmp.h>
#include <signal.h>
#include <sys/mman.h>
#include <pthread.h>
#include <libkern/OSCacheControl.h>

#define NREG 28
#define NVEC 28

#define SCRATCH 512

typedef struct {
    uint64_t x[NREG];
    uint64_t nzcv;
    uint64_t pad;   // keeps `v` 16-byte aligned: LDR/STR q scale their offset by 16, so an
                    // unaligned v would silently round down and land on nzcv
    uint8_t  v[NVEC][16];
    uint8_t  scratch[SCRATCH];  // what load/store instructions address
} State;

#define OFF_X    ((uint32_t)offsetof(State, x))
#define OFF_NZCV ((uint32_t)offsetof(State, nzcv))
#define OFF_V    ((uint32_t)offsetof(State, v))
#define OFF_SCR  ((uint32_t)offsetof(State, scratch))
_Static_assert(OFF_V % 16 == 0, "v must be 16-byte aligned for LDR/STR q");
_Static_assert(OFF_X % 8 == 0 && OFF_NZCV % 8 == 0, "x and nzcv must be 8-byte aligned");

// ── instruction encoders (only what the harness itself needs) ──

static uint32_t *code;
static uint32_t *cp;

static void emit(uint32_t w) { *cp++ = w; }

// LDP/STP xt1, xt2, [xn, #imm]  (imm is a multiple of 8, -512..504)
static void ldp_x(int t1, int t2, int n, int imm) {
    emit(0xa9400000u | ((uint32_t)((imm / 8) & 0x7f) << 15) | ((uint32_t)t2 << 10) | ((uint32_t)n << 5) | (uint32_t)t1);
}
static void stp_x(int t1, int t2, int n, int imm) {
    emit(0xa9000000u | ((uint32_t)((imm / 8) & 0x7f) << 15) | ((uint32_t)t2 << 10) | ((uint32_t)n << 5) | (uint32_t)t1);
}
// LDR/STR xt, [xn, #imm]  (imm a multiple of 8, 0..32760)
static void ldr_x(int t, int n, unsigned imm) {
    emit(0xf9400000u | ((imm / 8) << 10) | ((uint32_t)n << 5) | (uint32_t)t);
}
static void str_x(int t, int n, unsigned imm) {
    emit(0xf9000000u | ((imm / 8) << 10) | ((uint32_t)n << 5) | (uint32_t)t);
}
// LDR/STR qt, [xn, #imm]  (imm a multiple of 16, 0..65520)
static void ldr_q(int t, int n, unsigned imm) {
    emit(0x3dc00000u | ((imm / 16) << 10) | ((uint32_t)n << 5) | (uint32_t)t);
}
static void str_q(int t, int n, unsigned imm) {
    emit(0x3d800000u | ((imm / 16) << 10) | ((uint32_t)n << 5) | (uint32_t)t);
}
static void msr_nzcv(int t) { emit(0xd51b4200u | (uint32_t)t); }
static void mrs_nzcv(int t) { emit(0xd53b4200u | (uint32_t)t); }
static void mov_x(int d, int n) { emit(0xaa0003e0u | ((uint32_t)n << 16) | (uint32_t)d); }
static void ret_(void) { emit(0xd65f03c0u); }
// ADD xd, xn, #imm12
static void add_imm(int d, int n, unsigned imm) {
    emit(0x91000000u | (imm << 10) | ((uint32_t)n << 5) | (uint32_t)d);
}

// the state pointer lives in x28 across the instruction under test, which is why generated
// instructions are restricted to registers 0..27.
#define SP_REG 28

static void build(uint32_t insn, int base_reg) {
    cp = code;
    // prologue: the caller passes &state in x0. save what the ABI says we must.
    emit(0xa9bf7bfdu);              // stp x29, x30, [sp, #-16]!
    emit(0xd10403ffu);              // sub sp, sp, #256
    stp_x(19, 20, 31, 0);
    stp_x(21, 22, 31, 16);
    stp_x(23, 24, 31, 32);
    stp_x(25, 26, 31, 48);
    stp_x(27, SP_REG, 31, 64);
    mov_x(SP_REG, 0);

    for (int i = 0; i < NVEC; i++) ldr_q(i, SP_REG, OFF_V + (unsigned)i * 16);
    ldr_x(0, SP_REG, OFF_NZCV);
    msr_nzcv(0);
    for (int i = 0; i + 1 < NREG; i += 2) ldp_x(i, i + 1, SP_REG, (int)OFF_X + i * 8);
    // point the addressing base at the scratch buffer, in the middle so both negative and
    // positive offsets stay inside it
    if (base_reg >= 0) add_imm(base_reg, SP_REG, OFF_SCR + SCRATCH / 2);

    emit(insn);

    // read the flags before anything else runs. x0 goes out first so the mrs is free to clobber it.
    str_x(0, SP_REG, OFF_X);
    mrs_nzcv(0);
    str_x(0, SP_REG, OFF_NZCV);
    for (int i = 1; i < NREG; i++) str_x(i, SP_REG, (unsigned)((int)OFF_X + i * 8));
    for (int i = 0; i < NVEC; i++) str_q(i, SP_REG, OFF_V + (unsigned)i * 16);

    ldp_x(19, 20, 31, 0);
    ldp_x(21, 22, 31, 16);
    ldp_x(23, 24, 31, 32);
    ldp_x(25, 26, 31, 48);
    ldp_x(27, SP_REG, 31, 64);
    emit(0x910403ffu);              // add sp, sp, #256
    emit(0xa8c17bfdu);              // ldp x29, x30, [sp], #16
    ret_();
}

// ── trap handling ──
// an encoding the hardware rejects, or one that faults, must not take the harness down with it.

static sigjmp_buf trap_env;
static volatile sig_atomic_t trapped;

static void on_trap(int sig) { (void)sig; trapped = 1; siglongjmp(trap_env, 1); }

int main(int argc, char **argv) {
    // -d <hexinsn>: dump the generated harness for that instruction and exit, so the harness
    // itself can be disassembled rather than trusted.
    if (argc == 3 && strcmp(argv[1], "-d") == 0) {
        static uint32_t buf[4096];
        code = buf;
        build((uint32_t)strtoul(argv[2], NULL, 0), -1);
        fwrite(buf, 4, (size_t)(cp - buf), stdout);
        return 0;
    }
    // apple silicon refuses a mapping that is writable and executable at once. MAP_JIT gives a
    // page whose W/X state is switched per thread; without it, fall back to flipping mprotect
    // around each rebuild.
    size_t page = 1u << 16;
    int jit = 1;
    code = mmap(NULL, page, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
    if (code == MAP_FAILED) {
        jit = 0;
        code = mmap(NULL, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (code == MAP_FAILED) { perror("mmap"); return 1; }
    }

    struct sigaction sa; memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_trap;
    sa.sa_flags = SA_NODEFER;
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
    sigaction(SIGTRAP, &sa, NULL);

    for (;;) {
        uint32_t hdr[2];
        State st;
        if (fread(hdr, sizeof hdr, 1, stdin) != 1) break;
        if (fread(&st, sizeof st, 1, stdin) != 1) break;

        if (jit) pthread_jit_write_protect_np(0);
        else if (mprotect(code, page, PROT_READ | PROT_WRITE) != 0) { perror("mprotect w"); return 1; }
        build(hdr[0], (int32_t)hdr[1]);
        if (jit) pthread_jit_write_protect_np(1);
        else if (mprotect(code, page, PROT_READ | PROT_EXEC) != 0) { perror("mprotect x"); return 1; }
        sys_icache_invalidate(code, page);

        trapped = 0;
        if (sigsetjmp(trap_env, 1) == 0) {
            ((void (*)(State *))code)(&st);
        }

        // the base register comes back holding a host address, which means nothing to the
        // caller; report how far the instruction moved it instead, which is what write-back
        // addressing actually defines.
        if (!trapped && (int32_t)hdr[1] >= 0)
            st.x[(int32_t)hdr[1]] -= (uint64_t)(uintptr_t)st.scratch + SCRATCH / 2;
        uint32_t out[2] = { trapped ? 1u : 0u, 0u };
        fwrite(out, sizeof out, 1, stdout);
        fwrite(&st, sizeof st, 1, stdout);
        fflush(stdout);
    }
    return 0;
}
