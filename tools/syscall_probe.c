// a guest-side probe for syscall edge cases that ltp exercises indirectly through libc. ltp's
// binaries are static glibc, so a failing case leaves it ambiguous whether the kernel or the libc
// wrapper produced the answer. this asks the kernel directly.
//
//   nix-shell -p zig_0_15 --run "zig cc -target aarch64-linux-musl -static -O2 \
//       -o Jitter/Jitter/Resources/syscall_probe tools/syscall_probe.c"
//
// then run it in the guest; each line is "name = ret errno".

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <pthread.h>
#include <sys/stat.h>

static long sc(long n, long a, long b, long c, long d, long e, long f) {
    register long x8 __asm__("x8") = n;
    register long x0 __asm__("x0") = a;
    register long x1 __asm__("x1") = b;
    register long x2 __asm__("x2") = c;
    register long x3 __asm__("x3") = d;
    register long x4 __asm__("x4") = e;
    register long x5 __asm__("x5") = f;
    __asm__ volatile ("svc 0" : "+r"(x0) : "r"(x8), "r"(x1), "r"(x2), "r"(x3), "r"(x4), "r"(x5) : "memory", "cc");
    return x0;
}

// set by the lease-break handler so the holder can see that it was told before it releases
static volatile int lease_broken = 0;
static void lease_hit(int sig) { (void)sig; lease_broken = 1; }

// write one byte from a throwaway child, and report the signal that killed it (0 if it survived).
// the write has to happen somewhere expendable: a fault here is the answer, not a failure.
static int child_touch(long addr) {
    long k = sc(SYS_clone, 17, 0, 0, 0, 0, 0);
    if (k == 0) {
        *(volatile char *)addr = 1;
        sc(SYS_exit, 0, 0, 0, 0, 0, 0);
    }
    int st = 0;
    sc(SYS_wait4, k, (long)&st, 0, 0, 0, 0);
    return (st & 0x7f) ? (st & 0x7f) : 0;
}

// a thread that maps a page, so the main thread can check that it can see it
static volatile long thread_map;
static void *thread_mapper(void *arg) {
    (void)arg;
    thread_map = sc(SYS_mmap, 0, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return 0;
}

// a thread that writes to an unmapped address: it should die, and take its process with it.
static volatile long hole_addr;
static void *thread_touch(void *arg) {
    (void)arg;
    *(volatile char *)hole_addr = 1;
    return 0;
}

// a SIGUSR1 handler that ends the process with a recognisable status
static void group_hit(int sig) { (void)sig; sc(SYS_exit, 42, 0, 0, 0, 0, 0); }

// the third field of /proc/<pid>/stat, which is how a program asks "has it gone to sleep yet".
// it is also the thing ltp's TST_PROCESS_STATE_WAIT spins on.
static char proc_state(long pid) {
    char path[64], buf[256];
    int n = 0;
    path[n++] = '/'; path[n++] = 'p'; path[n++] = 'r'; path[n++] = 'o'; path[n++] = 'c'; path[n++] = '/';
    char digits[16]; int d = 0;
    long v = pid;
    do { digits[d++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (d) path[n++] = digits[--d];
    const char *tail = "/stat";
    for (const char *t = tail; *t; t++) path[n++] = *t;
    path[n] = 0;
    // give the child a moment to actually reach its pause before asking
    for (int tries = 0; tries < 200; tries++) {
        int fd2 = (int)sc(SYS_openat, AT_FDCWD, (long)path, 0, 0, 0, 0);
        if (fd2 < 0) { usleep(5000); continue; }
        long r = sc(SYS_read, fd2, (long)buf, sizeof(buf) - 1, 0, 0, 0);
        sc(SYS_close, fd2, 0, 0, 0, 0, 0);
        if (r > 0) {
            buf[r] = 0;
            char *close_paren = strrchr(buf, ')');
            if (close_paren && close_paren[1] == ' ') {
                if (close_paren[2] == 'S') return 'S';
                usleep(5000); continue;
            }
        }
        usleep(5000);
    }
    return '?';
}

static void report(const char *name, long r) {
    if (r < 0) printf("%-28s = -1 %s(%ld)\n", name, strerror((int)-r), -r);
    else       printf("%-28s = %ld\n", name, r);
}

int main(void) {
    // unbuffered: this probe forks constantly, and a buffered line sitting in the parent's stdio
    // buffer at fork time is inherited by the child and printed AGAIN when the child flushes. that
    // turned one measurement into three contradictory-looking ones and cost real debugging time.
    setbuf(stdout, NULL);
    char buf[512];
    void *none = mmap(NULL, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int fd = (int)sc(SYS_openat, AT_FDCWD, (long)"/tmp/.syscall_probe", O_RDWR | O_CREAT | O_TRUNC, 0644, 0, 0);

    report("fstat(fd, NULL)",            sc(SYS_fstat, fd, 0, 0, 0, 0, 0));
    report("fstat(fd, PROT_NONE)",       sc(SYS_fstat, fd, (long)none, 0, 0, 0, 0));
    report("newfstatat(cwd, NULL, buf)", sc(SYS_newfstatat, AT_FDCWD, 0, (long)buf, 0, 0, 0));
    report("newfstatat(cwd, none, buf)", sc(SYS_newfstatat, AT_FDCWD, (long)none, (long)buf, 0, 0, 0));
    report("newfstatat(fd, \"\", NULL)",   sc(SYS_newfstatat, fd, (long)"", 0, 0x1000, 0, 0));
    report("gettimeofday(buf, -1)",      sc(SYS_gettimeofday, (long)buf, -1L, 0, 0, 0, 0));
    report("gettimeofday(-1, 0)",        sc(SYS_gettimeofday, -1L, 0, 0, 0, 0, 0));
    report("pipe2(NULL, 0)",             sc(SYS_pipe2, 0, 0, 0, 0, 0, 0));
    report("pipe2(none, 0)",             sc(SYS_pipe2, (long)none, 0, 0, 0, 0, 0));
    report("statx(cwd, none, ...)",      sc(SYS_statx, AT_FDCWD, (long)none, 0, 0x7ff, (long)buf, 0));

    // O_APPEND: seek into the middle, write, and see where the offset lands
    sc(SYS_write, fd, (long)buf, 2048, 0, 0, 0);
    sc(SYS_close, fd, 0, 0, 0, 0, 0);
    fd = (int)sc(SYS_openat, AT_FDCWD, (long)"/tmp/.syscall_probe", O_RDWR | O_APPEND, 0, 0, 0);
    sc(SYS_lseek, fd, 1024, SEEK_SET, 0, 0, 0);
    sc(SYS_write, fd, (long)buf, 1024, 0, 0, 0);
    report("O_APPEND offset (want 3072)", sc(SYS_lseek, fd, 0, SEEK_CUR, 0, 0, 0));
    sc(SYS_close, fd, 0, 0, 0, 0, 0);

    // unlink of a directory must be EISDIR
    sc(SYS_mkdirat, AT_FDCWD, (long)"/tmp/.syscall_probe_dir", 0755, 0, 0, 0);
    report("unlink(dir) (want EISDIR)",  sc(SYS_unlinkat, AT_FDCWD, (long)"/tmp/.syscall_probe_dir", 0, 0, 0, 0));
    sc(SYS_unlinkat, AT_FDCWD, (long)"/tmp/.syscall_probe_dir", 0x200 /* AT_REMOVEDIR */, 0, 0, 0);

    // ftruncate on a socket must be EINVAL
    int s = (int)sc(SYS_socket, 1 /* AF_UNIX */, 1 /* SOCK_STREAM */, 0, 0, 0, 0);
    report("ftruncate(socket)",          sc(SYS_ftruncate, s, 4, 0, 0, 0, 0));
    sc(SYS_close, s, 0, 0, 0, 0, 0);

    // O_PATH descriptors name a file without granting access to it
    int p = (int)sc(SYS_openat, AT_FDCWD, (long)"/tmp/.syscall_probe", 0x200000 /* O_PATH */, 0, 0, 0);
    report("O_PATH open",                p);
    report("O_PATH fchmod",              sc(SYS_fchmod, p, 0600, 0, 0, 0, 0));
    report("O_PATH fchown",              sc(SYS_fchown, p, 0, 0, 0, 0, 0));
    report("O_PATH ioctl",               sc(SYS_ioctl, p, 0x5401, (long)buf, 0, 0, 0));
    report("O_PATH fgetxattr",           sc(SYS_fgetxattr, p, (long)"user.x", (long)buf, 16, 0, 0));
    sc(SYS_close, p, 0, 0, 0, 0, 0);

    // glibc's system()/posix_spawn do not fork: they clone with CLONE_VM|CLONE_VFORK and exec on a
    // caller-supplied stack. ltp's access02 runs its X_OK case through system(), so if this shape is
    // mishandled the test reports "execute failed" with no errno to explain it.
    {
        static char child_stack[65536];
        long pid = sc(SYS_clone, 0x00000100 /* CLONE_VM */ | 0x00004000 /* CLONE_VFORK */ | 17 /* SIGCHLD */,
                      (long)(child_stack + sizeof(child_stack)), 0, 0, 0, 0);
        if (pid == 0) {
            char *argv[] = { (char *)"/bin/true", NULL };
            char *envp[] = { NULL };
            sc(SYS_execve, (long)"/bin/true", (long)argv, (long)envp, 0, 0, 0);
            sc(SYS_exit, 127, 0, 0, 0, 0, 0);
        }
        report("clone(VM|VFORK)", pid);
        if (pid > 0) {
            int status = 0;
            long w = sc(SYS_wait4, pid, (long)&status, 0, 0, 0, 0);
            printf("%-28s = wait=%ld status=0x%x\n", "clone(VM|VFORK) child", w, status);
        }
    }

    printf("%-28s = %d\n", "system(/bin/true)", system("/bin/true"));

    // narrow it down: plain fork+exec of the applet, then the same through the shell, which is what
    // system() actually does.
    {
        char *envp[] = { (char *)"PATH=/bin:/usr/bin", NULL };
        struct { const char *name; char *const *argv; const char *file; } cases[] = {
            { "fork+exec /bin/true", (char *const[]){ (char *)"true", NULL }, "/bin/true" },
            { "fork+exec sh -c true", (char *const[]){ (char *)"sh", (char *)"-c", (char *)"/bin/true", NULL }, "/bin/sh" },
        };
        for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            long pid = sc(SYS_clone, 17 /* SIGCHLD */, 0, 0, 0, 0, 0);
            if (pid == 0) {
                sc(SYS_execve, (long)cases[i].file, (long)cases[i].argv, (long)envp, 0, 0, 0);
                sc(SYS_exit, 126, 0, 0, 0, 0, 0);
            }
            int status = 0;
            sc(SYS_wait4, pid, (long)&status, 0, 0, 0, 0);
            printf("%-28s = status=0x%x\n", cases[i].name, status);
        }
    }

    // ltp runs each test body in a FORKED CHILD while setup() ran in the parent, so anything that
    // travels badly across fork shows up as a failure the test itself cannot explain. repeat the
    // descriptor checks on the far side of a fork.
    sc(SYS_openat, AT_FDCWD, (long)"/tmp/.syscall_probe", O_RDWR | O_CREAT, 0644, 0, 0);
    p = (int)sc(SYS_openat, AT_FDCWD, (long)"/tmp/.syscall_probe", 0x200000 /* O_PATH */, 0, 0, 0);
    {
        long pid = sc(SYS_clone, 17 /* SIGCHLD */, 0, 0, 0, 0, 0);
        if (pid == 0) {
            report("after fork: O_PATH fchmod", sc(SYS_fchmod, p, 0600, 0, 0, 0, 0));
            report("after fork: O_PATH fchown", sc(SYS_fchown, p, 0, 0, 0, 0, 0));
            report("after fork: fstat(fd,NULL)", sc(SYS_fstat, fd, 0, 0, 0, 0, 0));
            fflush(stdout);
            sc(SYS_exit, 0, 0, 0, 0, 0, 0);
        }
        int st = 0;
        sc(SYS_wait4, pid, (long)&st, 0, 0, 0, 0);
    }
    sc(SYS_close, p, 0, 0, 0, 0, 0);

    // statx is what a modern glibc uses for stat/fstat; a null result buffer must fault.
    report("statx(fd,\"\",EMPTY,NULL)",  sc(SYS_statx, 1, (long)"", 0x1000, 0x7ff, 0, 0));

    // readlink with a non-positive buffer size is EINVAL, whatever the link is
    sc(SYS_symlinkat, (long)"/tmp/.syscall_probe", AT_FDCWD, (long)"/tmp/.syscall_probe_link", 0, 0, 0);
    report("readlink(link, buf, 0)",     sc(SYS_readlinkat, AT_FDCWD, (long)"/tmp/.syscall_probe_link", (long)buf, 0, 0, 0));
    report("readlink(link, buf, 256)",   sc(SYS_readlinkat, AT_FDCWD, (long)"/tmp/.syscall_probe_link", (long)buf, 256, 0, 0));
    report("readlink(regular file)",     sc(SYS_readlinkat, AT_FDCWD, (long)"/tmp/.syscall_probe", (long)buf, 256, 0, 0));

    // getcwd must report the RESOLVED directory, not the symlink that was chdir'd through
    sc(SYS_mkdirat, AT_FDCWD, (long)"/tmp/.syscall_probe_d", 0755, 0, 0, 0);
    sc(SYS_symlinkat, (long)"/tmp/.syscall_probe_d", AT_FDCWD, (long)"/tmp/.syscall_probe_dl", 0, 0, 0);
    if (sc(SYS_chdir, (long)"/tmp/.syscall_probe_dl", 0, 0, 0, 0, 0) == 0) {
        sc(SYS_getcwd, (long)buf, sizeof(buf), 0, 0, 0, 0);
        printf("%-28s = %s\n", "getcwd via symlinked cwd", buf);
    }
    sc(SYS_chdir, (long)"/tmp", 0, 0, 0, 0, 0);
    sc(SYS_unlinkat, AT_FDCWD, (long)"/tmp/.syscall_probe_dl", 0, 0, 0, 0);
    sc(SYS_unlinkat, AT_FDCWD, (long)"/tmp/.syscall_probe_d", 0x200, 0, 0, 0);
    sc(SYS_unlinkat, AT_FDCWD, (long)"/tmp/.syscall_probe_link", 0, 0, 0, 0);

    // SysV shared memory: two processes that never shared a mapping must see each other's writes
    // through a segment they both attached by id.
    {
        const long IPC_PRIVATE = 0, IPC_CREAT = 01000, IPC_RMID = 0, IPC_STAT = 2;
        long id = sc(194 /* shmget */, IPC_PRIVATE, 4096, IPC_CREAT | 0666, 0, 0, 0);
        report("shmget", id);
        if (id >= 0) {
            long addr = sc(196 /* shmat */, id, 0, 0, 0, 0, 0);
            printf("%-28s = %s\n", "shmat", addr < 0 && addr > -4096 ? "failed" : "ok");
            if (!(addr < 0 && addr > -4096)) {
                char *seg = (char *)addr;
                seg[0] = 0; seg[1] = 0;
                long kid = sc(SYS_clone, 17, 0, 0, 0, 0, 0);
                if (kid == 0) {
                    long a2 = sc(196, id, 0, 0, 0, 0, 0);   // a fresh attach in the child
                    if (!(a2 < 0 && a2 > -4096)) {
                        char *s2 = (char *)a2;
                        s2[0] = 42;                          // the child writes
                        // yield rather than spin: shared memory here coheres at SYSCALL
                        // BOUNDARIES, so a loop that makes no syscall never observes the other
                        // side. that is what any real waiter does anyway.
                        for (int i = 0; i < 20000 && s2[1] != 43; i++) usleep(200);
                        sc(197 /* shmdt */, a2, 0, 0, 0, 0, 0);
                    }
                    sc(SYS_exit, 0, 0, 0, 0, 0, 0);
                }
                for (int i = 0; i < 20000 && seg[0] != 42; i++) usleep(200);
                seg[1] = 43;
                usleep(1000);
                int st = 0;
                sc(SYS_wait4, kid, (long)&st, 0, 0, 0, 0);
                printf("%-28s = child wrote %d, parent answered\n", "shm shared both ways", seg[0]);

                char sbuf[256];
                memset(sbuf, 0, sizeof(sbuf));
                long r = sc(195 /* shmctl */, id, IPC_STAT, (long)sbuf, 0, 0, 0);
                unsigned long segsz; long nattch;
                memcpy(&segsz, sbuf + 48, 8);
                memcpy(&nattch, sbuf + 88, 8);
                printf("%-28s = ret=%ld segsz=%lu nattch=%ld\n", "shmctl(IPC_STAT)", r, segsz, nattch);
                report("shmdt", sc(197, addr, 0, 0, 0, 0, 0));
            }
            report("shmctl(IPC_RMID)", sc(195, id, IPC_RMID, 0, 0, 0, 0));
            report("shmat after rm", sc(196, id, 0, 0, 0, 0, 0) < 0 ? -1 : 0);
        }
    }

    // device nodes: mknod one, check stat sees a character device with the right major:minor, and
    // check that opening it gets the driver rather than the placeholder file underneath.
    {
        sc(SYS_unlinkat, AT_FDCWD, (long)"/tmp/.probe_zero", 0, 0, 0, 0);
        report("mknod(c 1:5)", sc(SYS_mknodat, AT_FDCWD, (long)"/tmp/.probe_zero",
                                  0020000 | 0666, (1 << 8) | 5, 0, 0));
        struct stat st;
        memset(&st, 0, sizeof(st));
        long r = sc(SYS_newfstatat, AT_FDCWD, (long)"/tmp/.probe_zero", (long)&st, 0, 0, 0);
        printf("%-28s = ret=%ld mode=0%o rdev=%u:%u\n", "stat(node)", r,
               (unsigned)st.st_mode, (unsigned)((st.st_rdev >> 8) & 0xfff), (unsigned)(st.st_rdev & 0xff));
        int zfd = (int)sc(SYS_openat, AT_FDCWD, (long)"/tmp/.probe_zero", O_RDWR, 0, 0, 0);
        report("open(node)", zfd);
        if (zfd >= 0) {
            memset(buf, 0xaa, 64);
            long n = sc(SYS_read, zfd, (long)buf, 64, 0, 0, 0);
            int zeroed = 1;
            for (int i = 0; i < 64; i++) if (buf[i] != 0) zeroed = 0;
            printf("%-28s = n=%ld zeroed=%d\n", "read(/dev/zero node)", n, zeroed);
            report("write(node)", sc(SYS_write, zfd, (long)buf, 16, 0, 0, 0));
            sc(SYS_close, zfd, 0, 0, 0, 0, 0);
        }
        // a node whose major:minor has no driver behind it is ENXIO, as on linux
        sc(SYS_unlinkat, AT_FDCWD, (long)"/tmp/.probe_nodev", 0, 0, 0, 0);
        sc(SYS_mknodat, AT_FDCWD, (long)"/tmp/.probe_nodev", 0020000 | 0666, (99 << 8) | 1, 0, 0);
        report("open(node, no driver)", sc(SYS_openat, AT_FDCWD, (long)"/tmp/.probe_nodev", O_RDONLY, 0, 0, 0));
        sc(SYS_unlinkat, AT_FDCWD, (long)"/tmp/.probe_nodev", 0, 0, 0, 0);
        sc(SYS_unlinkat, AT_FDCWD, (long)"/tmp/.probe_zero", 0, 0, 0, 0);
    }

    // kill(0, sig) must reach every process in the CALLER'S group and nothing outside it, and a
    // process parked in pause() must wake for it. ltp's kill02 hangs on exactly this. we move into
    // a group of our own first, so the probe cannot signal the shell that started it.
    {
        sc(SYS_setpgid, 0, 0, 0, 0, 0, 0);
        signal(SIGUSR1, SIG_IGN);           // the parent is in the group too and would die otherwise

        long same = sc(SYS_clone, 17, 0, 0, 0, 0, 0);
        if (same == 0) {
            signal(SIGUSR1, group_hit);
            for (;;) sc(SYS_ppoll, 0, 0, 0, 0, 8, 0);   // pause(): sleep until a signal arrives
        }
        long other = sc(SYS_clone, 17, 0, 0, 0, 0, 0);
        if (other == 0) {
            sc(SYS_setpgid, 0, 0, 0, 0, 0, 0);          // leave the group before the signal flies
            signal(SIGUSR1, group_hit);
            for (;;) sc(SYS_ppoll, 0, 0, 0, 0, 8, 0);
        }
        // let both children reach their pause. /proc/<pid>/stat is the honest way to know, so use it.
        printf("%-28s = %c %c\n", "children sleeping (want S S)", proc_state(same), proc_state(other));

        sc(SYS_kill, 0, SIGUSR1, 0, 0, 0, 0);
        int st1 = 0, st2 = 0;
        long w1 = sc(SYS_wait4, same, (long)&st1, 0, 0, 0, 0);
        printf("%-28s = wait=%ld status=0x%x\n", "same group got SIGUSR1", w1, st1);
        long w2 = sc(SYS_wait4, other, (long)&st2, 1 /* WNOHANG */, 0, 0, 0);
        printf("%-28s = wait=%ld (want 0)\n", "other group spared", w2);
        sc(SYS_kill, other, 9, 0, 0, 0, 0);
        sc(SYS_wait4, other, (long)&st2, 0, 0, 0, 0);
    }

    // a write lease says "tell me before anyone else gets at this file". the holder must be
    // signalled and the newcomer's open must WAIT until the lease is gone -- that ordering is the
    // whole guarantee, so the child stamps a shared word before and after its open.
    {
        const char *lp = "/tmp/.probe_lease";
        sc(SYS_unlinkat, AT_FDCWD, (long)lp, 0, 0, 0, 0);
        int lfd = (int)sc(SYS_openat, AT_FDCWD, (long)lp, O_RDWR | O_CREAT, 0644, 0, 0);
        report("lease: open", lfd);
        signal(SIGIO, lease_hit);
        report("lease: F_SETLEASE F_WRLCK", sc(SYS_fcntl, lfd, 1024 /* F_SETLEASE */, 1 /* F_WRLCK */, 0, 0, 0));
        printf("%-28s = %ld (want 1)\n", "lease: F_GETLEASE",
               sc(SYS_fcntl, lfd, 1025 /* F_GETLEASE */, 0, 0, 0, 0));

        long kid = sc(SYS_clone, 17, 0, 0, 0, 0, 0);
        if (kid == 0) {
            int c = (int)sc(SYS_openat, AT_FDCWD, (long)lp, O_RDONLY, 0, 0, 0);
            sc(SYS_exit, c >= 0 ? 7 : 8, 0, 0, 0, 0, 0);
        }
        // the child is now blocked in open(); the signal should already be here
        for (int i = 0; i < 200 && !lease_broken; i++) sc(SYS_nanosleep, (long)&(struct timespec){0, 5000000}, 0, 0, 0, 0, 0);
        printf("%-28s = %d (want 1), child still blocked=%c\n", "lease: holder signalled",
               lease_broken, proc_state(kid));
        report("lease: release", sc(SYS_fcntl, lfd, 1024, 2 /* F_UNLCK */, 0, 0, 0));
        int lst = 0;
        sc(SYS_wait4, kid, (long)&lst, 0, 0, 0, 0);
        printf("%-28s = status=0x%x (want 0x700)\n", "lease: child open completed", lst);
        printf("%-28s = %ld (want 2)\n", "lease: F_GETLEASE after",
               sc(SYS_fcntl, lfd, 1025, 0, 0, 0, 0));
        sc(SYS_close, lfd, 0, 0, 0, 0, 0);
        sc(SYS_unlinkat, AT_FDCWD, (long)lp, 0, 0, 0, 0);
    }

    // address space that nothing mapped must not be readable. a hole is not free memory: it is a
    // hole, and touching it is how a program finds out it used a pointer it had already given back.
    {
        long ps = 4096;
        long m = sc(SYS_mmap, 0, ps * 3, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        printf("%-28s = sig=%d (want 0)\n", "write inside a mapping", child_touch(m));
        sc(SYS_munmap, m + ps, ps, 0, 0, 0, 0);
        printf("%-28s = sig=%d (want 11)\n", "write into an unmapped hole", child_touch(m + ps));
        printf("%-28s = sig=%d (want 0)\n", "the kept page still works", child_touch(m));
        printf("%-28s = sig=%d (want 11)\n", "write where nothing was ever mapped", child_touch(0x60000000L));
        sc(SYS_munmap, m, ps, 0, 0, 0, 0);
        sc(SYS_munmap, m + ps * 2, ps, 0, 0, 0, 0);

        // the note the walking child leaves for its parent has to be allocated BEFORE the reserve
        // below, or first-fit reuse drops it straight into the hole this is about to test.
        long shared = sc(SYS_mmap, 0, ps, PROT_READ | PROT_WRITE, 0x01 | MAP_ANONYMOUS, -1, 0);
        volatile long *reached = (volatile long *)shared;

        // a MAP_GROWSDOWN mapping is a stack: a fault just below it grows it instead of killing
        // the process, unless something else is mapped inside the kernel's 256-page guard gap.
        long reserve = ps * 400;
        long area = sc(SYS_mmap, 0, reserve, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        sc(SYS_munmap, area, reserve, 0, 0, 0, 0);
        long stack = sc(SYS_mmap, area + reserve - ps * 4, ps * 4, PROT_READ | PROT_WRITE,
                        MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | 0x0100 /* MAP_GROWSDOWN */, -1, 0);
        printf("%-28s = sig=%d (want 0)\n", "growsdown stack extends down", child_touch(stack - ps));

        // the guard gap is the real subject: the stack may grow until its bottom would come within
        // 256 pages of whatever is mapped below it, and not one page further. block the bottom of
        // the reserve, then have a child walk DOWN a page at a time (which is what a recursing
        // stack does) recording how far it got in memory the parent can still read afterwards.
        sc(SYS_mmap, area, ps, PROT_READ | PROT_WRITE,
           MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        *reached = 0;
        long w = sc(SYS_clone, 17, 0, 0, 0, 0, 0);
        if (w == 0) {
            // MAP_SHARED coheres at syscall boundaries, so the note has to be followed by one
            for (long a = stack - ps; a > area; a -= ps) {
                *(volatile char *)a = 1;
                *reached = a;
                sc(SYS_getppid, 0, 0, 0, 0, 0, 0);
            }
            sc(SYS_exit, 0, 0, 0, 0, 0, 0);
        }
        int wst = 0;
        sc(SYS_wait4, w, (long)&wst, 0, 0, 0, 0);
        long gap_pages = (*reached - (area + ps)) / ps;
        printf("%-28s = sig=%d (want 11), stopped %ld pages above the block (want 256)\n",
               "guard gap stops the stack", wst & 0x7f, gap_pages);
        sc(SYS_munmap, shared, ps, 0, 0, 0, 0);
        sc(SYS_munmap, area, reserve, 0, 0, 0, 0);
    }

    // the address space belongs to the PROCESS, so a thread has to honour it too. a thread that
    // walks off a mapping must fault exactly like the main thread does -- the guard was wired for
    // the non-threaded path only, so a process with threads had no memory protection at all.
    {
        long ps = 4096;
        long m = sc(SYS_mmap, 0, ps * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        sc(SYS_munmap, m + ps, ps, 0, 0, 0, 0);
        hole_addr = m + ps;
        // a SIGSEGV kills the whole process, threads included, so this has to happen in a child.
        long k = sc(SYS_clone, 17, 0, 0, 0, 0, 0);
        if (k == 0) {
            pthread_t th;
            if (pthread_create(&th, 0, thread_touch, 0) != 0) sc(SYS_exit, 3, 0, 0, 0, 0, 0);
            pthread_join(th, 0);
            sc(SYS_exit, 0, 0, 0, 0, 0, 0);   // reached only if the write did NOT fault
        }
        int tst = 0;
        sc(SYS_wait4, k, (long)&tst, 0, 0, 0, 0);
        printf("%-28s = sig=%d (want 11)\n", "a thread faults on a hole", tst & 0x7f);
        sc(SYS_munmap, m, ps, 0, 0, 0, 0);
    }

    // threads share ONE address space, so a mapping made by one is real for all of them. with a
    // per-thread copy of the mapping list the main thread never learned about the thread's mmap
    // and its own next mmap was handed the very same address -- two live allocations, one memory.
    {
        thread_map = 0;
        pthread_t th;
        if (pthread_create(&th, 0, thread_mapper, 0) != 0) {
            printf("%-28s = pthread_create failed\n", "sibling mmap is visible");
        } else {
            pthread_join(th, 0);
            long mine = sc(SYS_mmap, 0, 4096, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            printf("%-28s = thread=%#lx main=%#lx distinct=%d (want 1)\n",
                   "sibling mmap is visible", thread_map, mine, thread_map != mine);
            // and the thread's mapping is genuinely ours to use and to give back
            *(volatile char *)thread_map = 7;
            printf("%-28s = %d, unmap=%ld\n", "thread's mapping usable", *(volatile char *)thread_map,
                   sc(SYS_munmap, thread_map, 4096, 0, 0, 0, 0));
            sc(SYS_munmap, mine, 4096, 0, 0, 0, 0);
        }
    }

    sc(SYS_unlinkat, AT_FDCWD, (long)"/tmp/.syscall_probe", 0, 0, 0, 0);
    return 0;
}
