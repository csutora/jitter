// the `jitter` command. a thin shim over the /dev/jitter control device: it joins its argv into a
// request, writes it to the device, then streams the reply to stdout and exits with the host's status.
// all policy (download, extract, switch) lives host-side; this exists purely for ergonomics, and an
// advanced user can talk to /dev/jitter directly. built static so it runs in any root, whatever libc
// that root ships.
//
// wire format
//   request:  the argv words joined by single spaces, no trailing newline
//   response: [int32 status][body...]  — body streamed to stdout until EOF
//
// cross-compile:
//   aarch64-unknown-linux-musl-gcc -static -Os -o jitter tools/jitter_shim.c
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static const char usage[] =
    "usage: jitter <command> [args]\n"
    "\n"
    "  pull <name|url> [as]   download a root and install it\n"
    "  switch <name>          restart the session in <name>\n"
    "  ls                     list installed roots\n"
    "  rm <name>              delete a root\n"
    "  default <name>         set the root to boot into\n"
    "  help                   this text\n";

int main(int argc, char **argv) {
    if (argc < 2) { fputs(usage, stderr); return 2; }

    char req[4096];
    size_t n = 0;
    for (int i = 1; i < argc; i++) {
        size_t len = strlen(argv[i]);
        if (n + len + 2 > sizeof req) { fputs("jitter: request too long\n", stderr); return 2; }
        if (i > 1) req[n++] = ' ';
        memcpy(req + n, argv[i], len);
        n += len;
    }

    int fd = open("/dev/jitter", O_RDWR);
    if (fd < 0) {
        fputs("jitter: /dev/jitter is unavailable (is this running under jitter?)\n", stderr);
        return 1;
    }
    if (write(fd, req, n) < 0) { perror("jitter: write"); close(fd); return 1; }

    // the host answers with a 4-byte status followed by the body. read the status first so a partial
    // body (a long-running pull that dies midway) still reports a real exit code.
    int status = 1;
    unsigned char hdr[4];
    size_t got = 0;
    while (got < sizeof hdr) {
        ssize_t r = read(fd, hdr + got, sizeof hdr - got);
        if (r <= 0) { fputs("jitter: no response from the host\n", stderr); close(fd); return 1; }
        got += (size_t)r;
    }
    status = (int)((unsigned)hdr[0] | ((unsigned)hdr[1] << 8) | ((unsigned)hdr[2] << 16) | ((unsigned)hdr[3] << 24));

    char buf[8192];
    for (;;) {
        ssize_t r = read(fd, buf, sizeof buf);
        if (r <= 0) break;
        ssize_t off = 0;
        while (off < r) {
            ssize_t w = write(1, buf + off, (size_t)(r - off));
            if (w <= 0) { close(fd); return 1; }
            off += w;
        }
    }
    close(fd);
    return status;
}
