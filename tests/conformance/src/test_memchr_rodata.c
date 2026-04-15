#include <string.h>
#include <stdio.h>
#include <unistd.h>

/*
 * Test memchr behavior with a string that matches the exact layout
 * busybox wget encounters in .rodata: "Wget\0Connecting to %s (%s)\n\0"
 *
 * The key: the string "Wget" is at an alignment offset within a page
 * that exercises the word-at-a-time loop in musl's memchr.
 */

/* Force the string to be at a known offset to test alignment handling */
static const char rodata[] = "Wget\0Connecting to %s (%s)\n\0GET %s://";

int main() {
    /* Test 1: memchr on the string directly */
    const char *p1 = memchr(rodata, 0, 100);
    printf("test1: memchr offset=%td (expect 4)\n", p1 ? p1 - rodata : -1);

    /* Test 2: strnlen */
    size_t len = strnlen(rodata, 100);
    printf("test2: strnlen=%zu (expect 4)\n", len);

    /* Test 3: fprintf to FILE* opened on fd via fdopen */
    int fds[2];
    pipe(fds);
    FILE *fp = fdopen(fds[1], "w");

    /* This is EXACTLY what busybox wget does */
    fprintf(fp, "User-Agent: %s\r\n", rodata);
    fflush(fp);
    fclose(fp);

    char buf[256];
    int n = read(fds[0], buf, sizeof(buf));
    close(fds[0]);

    printf("test3: wrote %d bytes (expect 18)\n", n);
    printf("test3: data=\"");
    for (int i = 0; i < n && i < 50; i++) {
        unsigned char c = buf[i];
        if (c == '\r') printf("\\r");
        else if (c == '\n') printf("\\n");
        else if (c == 0) printf("\\0");
        else if (c >= 32 && c < 127) putchar(c);
        else printf("\\x%02x", c);
    }
    printf("\"\n");

    int ok = (p1 - rodata == 4) && (len == 4) && (n == 18);
    printf("result: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
