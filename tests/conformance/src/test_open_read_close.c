#include "../harness.h"
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

int main(void) {
    TEST("open /etc/passwd succeeds");
    int fd = open("/etc/passwd", O_RDONLY);
    ASSERT(fd >= 0);

    TEST("read returns data");
    char buf[256];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    ASSERT(n > 0);
    buf[n] = '\0';

    TEST("data contains root");
    ASSERT(strstr(buf, "root") != NULL);

    TEST("close succeeds");
    int ret = close(fd);
    ASSERT_EQ(ret, 0);

    TEST("open nonexistent file fails with ENOENT");
    fd = open("/nonexistent_file_12345", O_RDONLY);
    ASSERT_EQ(fd, -1);

    TEST("close invalid fd fails");
    ret = close(9999);
    ASSERT_EQ(ret, -1);

    DONE();
}
