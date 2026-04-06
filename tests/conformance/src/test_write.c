#include "../harness.h"
#include <unistd.h>
#include <string.h>

int main(void) {
    TEST("write to stdout returns correct count");
    const char *msg = "hello from conformance test\n";
    ssize_t n = write(STDOUT_FILENO, msg, strlen(msg));
    ASSERT_EQ(n, (ssize_t)strlen(msg));

    TEST("write to stderr returns correct count");
    const char *err = "stderr works\n";
    n = write(STDERR_FILENO, err, strlen(err));
    ASSERT_EQ(n, (ssize_t)strlen(err));

    TEST("write with count=0 returns 0");
    n = write(STDOUT_FILENO, "x", 0);
    ASSERT_EQ(n, 0);

    DONE();
}
