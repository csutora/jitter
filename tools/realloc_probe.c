// luajit grows its bytecode array with realloc as it parses, so if a growing allocation ever
// loses or shifts its contents the parser sees garbage jump targets and reports nonsense like
// "control structure too long". musl services large reallocs with mremap and small ones with
// malloc+memcpy+free, so both paths are exercised here, with a checksum that names the first
// byte that changed.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

static int fails;

static void fillpat(unsigned char *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; i++) p[i] = (unsigned char)(i * 2654435761u + seed);
}

static int checkpat(const unsigned char *p, size_t n, unsigned seed, const char *what, size_t step) {
    for (size_t i = 0; i < n; i++) {
        unsigned char want = (unsigned char)(i * 2654435761u + seed);
        if (p[i] != want) {
            if (fails++ < 20)
                printf("FAIL %s step=%zu byte %zu of %zu: got 0x%02x want 0x%02x\n",
                       what, step, i, n, p[i], want);
            return 1;
        }
    }
    return 0;
}

// the exact shape luajit uses: double the vector, keep everything already in it
static void growvec(const char *what, size_t start, size_t limit) {
    size_t n = start;
    unsigned char *p = malloc(n);
    if (!p) { printf("FAIL %s: malloc\n", what); fails++; return; }
    fillpat(p, n, 0x5a);
    for (size_t step = 0; n < limit; step++) {
        size_t old = n;
        n *= 2;
        unsigned char *q = realloc(p, n);
        if (!q) { printf("FAIL %s: realloc to %zu\n", what, n); fails++; free(p); return; }
        p = q;
        if (checkpat(p, old, 0x5a, what, step)) { free(p); return; }
        fillpat(p + old, n - old, 0x5a - (unsigned)old * 2654435761u);
        // the tail pattern continues the same sequence, so one check covers the whole buffer
        for (size_t i = old; i < n; i++) p[i] = (unsigned char)(i * 2654435761u + 0x5a);
    }
    free(p);
}

int main(void) {
    growvec("growvec.small", 64, 1u << 12);
    growvec("growvec.medium", 4096, 1u << 20);
    growvec("growvec.large", 1u << 18, 1u << 23);

    // shrink as well as grow -- musl may move the block either way
    for (size_t sz = 1u << 20; sz > 32; sz /= 3) {
        unsigned char *p = malloc(1u << 20);
        if (!p) { fails++; printf("FAIL shrink: malloc\n"); break; }
        fillpat(p, 1u << 20, 0x21);
        unsigned char *q = realloc(p, sz);
        if (!q) { fails++; printf("FAIL shrink: realloc %zu\n", sz); free(p); break; }
        checkpat(q, sz, 0x21, "shrink", sz);
        free(q);
    }

    // mremap directly, which is the path musl takes for a large growing block
    long ps = sysconf(_SC_PAGESIZE);
    size_t sz = (size_t)ps * 16;
    unsigned char *m = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) { printf("FAIL mremap: mmap\n"); fails++; }
    else {
        fillpat(m, sz, 0x77);
        for (int i = 0; i < 6; i++) {
            size_t nsz = sz * 2;
            unsigned char *r = mremap(m, sz, nsz, MREMAP_MAYMOVE);
            if (r == MAP_FAILED) { printf("FAIL mremap grow %zu -> %zu\n", sz, nsz); fails++; break; }
            m = r;
            if (checkpat(m, sz, 0x77, "mremap", (size_t)i)) break;
            fillpat(m, nsz, 0x77);
            sz = nsz;
        }
        munmap(m, sz);
    }

    // and interleaved alloc/free churn, which is where a bad free-list shows up
    void *keep[64];
    for (int i = 0; i < 64; i++) {
        keep[i] = malloc(128 + i * 97);
        fillpat(keep[i], 128 + i * 97, (unsigned)i);
    }
    for (int round = 0; round < 8; round++) {
        for (int i = round % 2; i < 64; i += 2) {
            size_t n = 128 + i * 97;
            if (checkpat(keep[i], n, (unsigned)i, "churn", (size_t)round)) goto done;
            free(keep[i]);
            keep[i] = malloc(n * 2);
            fillpat(keep[i], n * 2, (unsigned)i);
        }
    }
done:
    printf(fails ? "\n%d FAILURES\n" : "\nall clean\n", fails);
    return fails != 0;
}
