#include "../harness.h"
#include <unistd.h>
#include <string.h>

int main(void) {
    char buf[256];

    TEST("getcwd returns /");
    char *cwd = getcwd(buf, sizeof(buf));
    ASSERT(cwd != NULL);
    ASSERT(strcmp(buf, "/") == 0);

    TEST("chdir to /tmp succeeds");
    int ret = chdir("/tmp");
    ASSERT_EQ(ret, 0);

    TEST("getcwd returns /tmp after chdir");
    cwd = getcwd(buf, sizeof(buf));
    ASSERT(cwd != NULL);
    ASSERT(strcmp(buf, "/tmp") == 0);

    TEST("chdir to nonexistent dir fails");
    ret = chdir("/nonexistent_dir_12345");
    ASSERT_EQ(ret, -1);

    TEST("chdir back to / succeeds");
    ret = chdir("/");
    ASSERT_EQ(ret, 0);

    DONE();
}
