#include "../harness.h"
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/syscall.h>

int main(void) {
    TEST("brk(0) returns current break");
    long brk0 = syscall(SYS_brk, 0);
    ASSERT(brk0 > 0);

    TEST("brk(brk0 + 4096) extends heap");
    long brk1 = syscall(SYS_brk, brk0 + 4096);
    ASSERT_EQ(brk1, brk0 + 4096);

    TEST("can write to extended heap region");
    char *p = (char *)brk0;
    memset(p, 0xAA, 4096);
    ASSERT(p[0] == (char)0xAA);
    ASSERT(p[4095] == (char)0xAA);

    TEST("malloc works (uses brk internally)");
    char *m = malloc(1024);
    ASSERT(m != NULL);
    memset(m, 0x55, 1024);
    ASSERT(m[0] == 0x55);
    ASSERT(m[1023] == 0x55);
    free(m);

    TEST("mmap anonymous page succeeds");
    void *mp = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT(mp != MAP_FAILED);

    TEST("can write to mmap'd page");
    memset(mp, 0x55, 4096);
    ASSERT(((unsigned char *)mp)[0] == 0x55);
    ASSERT(((unsigned char *)mp)[4095] == 0x55);

    TEST("munmap succeeds");
    int ret = munmap(mp, 4096);
    ASSERT_EQ(ret, 0);

    DONE();
}
