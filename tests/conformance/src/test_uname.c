#include "../harness.h"
#include <sys/utsname.h>
#include <string.h>

int main(void) {
    struct utsname buf;

    TEST("uname succeeds");
    int ret = uname(&buf);
    ASSERT_EQ(ret, 0);

    TEST("sysname is Linux");
    ASSERT(strcmp(buf.sysname, "Linux") == 0);

    TEST("machine is aarch64");
    ASSERT(strcmp(buf.machine, "aarch64") == 0);

    TEST("nodename is jitter");
    ASSERT(strcmp(buf.nodename, "jitter") == 0);

    DONE();
}
