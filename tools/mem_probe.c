// musl's aarch64 memcpy/memset/strlen/memcmp are hand-written NEON, so a single wrongly
// translated vector instruction shows up as rare, data-dependent corruption in unrelated
// programs rather than as a crash. this checks each of them against a naive byte loop across
// every length and alignment that changes which code path they take.
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#define BUF 8192
static unsigned char src[BUF + 64], dst[BUF + 64], ref[BUF + 64];
static int fails;

static void report(const char *what, size_t len, size_t off, size_t bad) {
    if (fails++ < 20)
        printf("FAIL %s len=%zu off=%zu first bad byte at %zu (got 0x%02x want 0x%02x)\n",
               what, len, off, bad, dst[bad], ref[bad]);
}

static size_t firstdiff(const unsigned char *a, const unsigned char *b, size_t n) {
    for (size_t i = 0; i < n; i++) if (a[i] != b[i]) return i;
    return (size_t)-1;
}

static void fill(unsigned char *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; i++) p[i] = (unsigned char)(seed * 1103515245u + i * 31u + 7u);
}

int main(void) {
    // lengths that straddle every branch these routines take (16/32/64/128-byte chunks)
    static const size_t lens[] = {
        0,1,2,3,4,5,6,7,8,9,15,16,17,31,32,33,47,48,63,64,65,95,96,127,128,129,
        191,192,255,256,257,383,384,511,512,513,1023,1024,1025,2047,2048,4095,4096,8191,
    };
    const size_t nlens = sizeof lens / sizeof *lens;

    fill(src, sizeof src, 1);

    for (size_t li = 0; li < nlens; li++) {
        size_t len = lens[li];
        for (size_t off = 0; off < 17; off++) {
            memset(dst, 0xcc, sizeof dst);
            memset(ref, 0xcc, sizeof ref);
            memcpy(dst + off, src + off + 1, len);
            for (size_t i = 0; i < len; i++) ref[off + i] = src[off + 1 + i];
            size_t bad = firstdiff(dst, ref, sizeof dst);
            if (bad != (size_t)-1) report("memcpy", len, off, bad);

            // overlapping forward and backward, which is where memmove picks a direction
            fill(dst, sizeof dst, 2); memcpy(ref, dst, sizeof dst);
            memmove(dst + off, dst + off + 3, len);
            for (size_t i = 0; i < len; i++) ref[off + i] = ref[off + 3 + i];
            bad = firstdiff(dst, ref, sizeof dst);
            if (bad != (size_t)-1) report("memmove.fwd", len, off, bad);

            fill(dst, sizeof dst, 3); memcpy(ref, dst, sizeof dst);
            memmove(dst + off + 3, dst + off, len);
            for (size_t i = len; i-- > 0;) ref[off + 3 + i] = ref[off + i];
            bad = firstdiff(dst, ref, sizeof dst);
            if (bad != (size_t)-1) report("memmove.bwd", len, off, bad);

            memset(dst, 0xcc, sizeof dst);
            memset(ref, 0xcc, sizeof ref);
            memset(dst + off, 0x5a, len);
            for (size_t i = 0; i < len; i++) ref[off + i] = 0x5a;
            bad = firstdiff(dst, ref, sizeof dst);
            if (bad != (size_t)-1) report("memset", len, off, bad);

            // strlen / strnlen over a run of non-NUL bytes ending exactly at len
            memset(dst, 'x', sizeof dst);
            dst[off + len] = 0;
            size_t got = strlen((char *)dst + off);
            if (got != len) {
                if (fails++ < 20) printf("FAIL strlen len=%zu off=%zu got %zu\n", len, off, got);
            }

            // memcmp: equal, then a single differing byte at each end of the range
            memcpy(dst, src, sizeof dst);
            memcpy(ref, src, sizeof ref);
            if (memcmp(dst + off, ref + off, len) != 0) {
                if (fails++ < 20) printf("FAIL memcmp equal len=%zu off=%zu\n", len, off);
            }
            if (len) {
                ref[off + len - 1] ^= 0x80;
                if (memcmp(dst + off, ref + off, len) == 0) {
                    if (fails++ < 20) printf("FAIL memcmp lastbyte len=%zu off=%zu\n", len, off);
                }
                ref[off + len - 1] ^= 0x80;
                ref[off] ^= 0x01;
                if (memcmp(dst + off, ref + off, len) == 0) {
                    if (fails++ < 20) printf("FAIL memcmp firstbyte len=%zu off=%zu\n", len, off);
                }
                ref[off] ^= 0x01;
            }

            // memchr, both a hit at the very end and a miss
            memset(dst, 'a', sizeof dst);
            if (len) {
                dst[off + len - 1] = 'Z';
                if (memchr(dst + off, 'Z', len) != dst + off + len - 1) {
                    if (fails++ < 20) printf("FAIL memchr hit len=%zu off=%zu\n", len, off);
                }
                dst[off + len - 1] = 'a';
            }
            if (memchr(dst + off, 'Z', len) != NULL) {
                if (fails++ < 20) printf("FAIL memchr miss len=%zu off=%zu\n", len, off);
            }
        }
    }

    // strcmp across a difference at every position of a medium string
    for (size_t n = 1; n < 200; n++) {
        char a[256], b[256];
        memset(a, 'q', sizeof a); memset(b, 'q', sizeof b);
        a[n] = 0; b[n] = 0;
        if (strcmp(a, b) != 0) { if (fails++ < 20) printf("FAIL strcmp equal n=%zu\n", n); }
        for (size_t i = 0; i < n; i++) {
            b[i] = 'r';
            if (strcmp(a, b) >= 0) { if (fails++ < 20) printf("FAIL strcmp lt n=%zu i=%zu\n", n, i); }
            if (strcmp(b, a) <= 0) { if (fails++ < 20) printf("FAIL strcmp gt n=%zu i=%zu\n", n, i); }
            b[i] = 'q';
        }
    }

    printf(fails ? "\n%d FAILURES\n" : "\nall clean\n", fails);
    return fails != 0;
}
