#include <string.h>
#include <stdio.h>
#include <limits.h>

/* Test strnlen / memchr behavior with null-adjacent strings */
int main() {
    /* Simulate the .rodata layout: "Wget\0Connecting to %s (%s)\n\0" */
    char buf[64];
    memcpy(buf, "Wget\0Connecting to %s (%s)\n\0", 28);

    /* Test 1: strlen */
    size_t len1 = strlen(buf);
    printf("strlen(\"Wget\")=%zu (expect 4)\n", len1);

    /* Test 2: strnlen with INT_MAX */
    size_t len2 = strnlen(buf, INT_MAX);
    printf("strnlen(\"Wget\",INT_MAX)=%zu (expect 4)\n", len2);

    /* Test 3: strnlen with large n */
    size_t len3 = strnlen(buf, 1000);
    printf("strnlen(\"Wget\",1000)=%zu (expect 4)\n", len3);

    /* Test 4: memchr for null byte */
    char *p = memchr(buf, 0, 100);
    printf("memchr null offset=%td (expect 4)\n", p ? p - buf : -1);

    /* Test 5: fprintf %s to a buffer via sprintf */
    char out[128];
    int n = sprintf(out, "User-Agent: %s\r\n", buf);
    printf("sprintf len=%d (expect 18)\n", n);
    printf("sprintf out=\"");
    for (int i = 0; i < n; i++) {
        if (out[i] == '\r') printf("\\r");
        else if (out[i] == '\n') printf("\\n");
        else if (out[i] == 0) printf("\\0");
        else putchar(out[i]);
    }
    printf("\"\n");

    /* Test 6: test with the actual Wget string from .rodata by pointing into it */
    const char *wget = "Wget";
    size_t len6 = strnlen(wget, INT_MAX);
    printf("strnlen(.rodata \"Wget\")=%zu (expect 4)\n", len6);

    return (len1 == 4 && len2 == 4 && len3 == 4 && n == 18) ? 0 : 1;
}
