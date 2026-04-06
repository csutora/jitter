#include "../harness.h"
#include <unistd.h>
#include <string.h>

int main(void) {
    int fds[2];

    TEST("pipe creates two fds");
    int ret = pipe(fds);
    ASSERT_EQ(ret, 0);
    ASSERT(fds[0] >= 0);
    ASSERT(fds[1] >= 0);
    ASSERT(fds[0] != fds[1]);

    TEST("write to pipe succeeds");
    const char *msg = "pipe test data";
    ssize_t n = write(fds[1], msg, strlen(msg));
    ASSERT_EQ(n, (ssize_t)strlen(msg));

    TEST("read from pipe returns correct data");
    char buf[64];
    n = read(fds[0], buf, sizeof(buf));
    ASSERT_EQ(n, (ssize_t)strlen(msg));
    buf[n] = '\0';
    ASSERT(strcmp(buf, "pipe test data") == 0);

    TEST("close pipe fds succeeds");
    ASSERT_EQ(close(fds[0]), 0);
    ASSERT_EQ(close(fds[1]), 0);

    DONE();
}
