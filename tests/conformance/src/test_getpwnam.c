#include "../harness.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <pwd.h>
#include <errno.h>

int main(void) {
    // Test fseek/rewind works
    TEST("fopen /etc/passwd");
    FILE *f = fopen("/etc/passwd", "rbe");
    ASSERT(f != NULL);

    char line[256];
    fgets(line, sizeof(line), f);
    write(2, "FIRST:", 6); write(2, line, strlen(line));

    TEST("rewind works");
    rewind(f);
    fgets(line, sizeof(line), f);
    write(2, "AFTER_REWIND:", 13); write(2, line, strlen(line));
    ASSERT(strstr(line, "root") != NULL);

    TEST("fseek(0, SEEK_SET) works");
    fseek(f, 0, SEEK_SET);
    fgets(line, sizeof(line), f);
    write(2, "AFTER_FSEEK:", 12); write(2, line, strlen(line));
    ASSERT(strstr(line, "root") != NULL);
    fclose(f);

    // Now the real test
    TEST("getpwnam(root) works");
    struct passwd *pw = getpwnam("root");
    ASSERT(pw != NULL);

    TEST("getpwnam(nobody) works");
    errno = 0;
    pw = getpwnam("nobody");
    if (!pw) {
        char msg[64];
        int len = snprintf(msg, sizeof(msg), "getpwnam failed errno=%d\n", errno);
        write(2, msg, len);
    }
    ASSERT(pw != NULL);

    DONE();
}
