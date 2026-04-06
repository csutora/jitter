#include "../harness.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int main(void) {
    struct stat st;

    TEST("fstat on stdin succeeds");
    int ret = fstat(STDIN_FILENO, &st);
    ASSERT_EQ(ret, 0);

    TEST("stat /etc/passwd succeeds");
    ret = stat("/etc/passwd", &st);
    ASSERT_EQ(ret, 0);

    TEST("/etc/passwd is a regular file");
    ASSERT(S_ISREG(st.st_mode));

    TEST("/etc/passwd has nonzero size");
    ASSERT(st.st_size > 0);

    TEST("stat / succeeds");
    ret = stat("/", &st);
    ASSERT_EQ(ret, 0);

    TEST("/ is a directory");
    ASSERT(S_ISDIR(st.st_mode));

    TEST("stat nonexistent file fails");
    ret = stat("/nonexistent_12345", &st);
    ASSERT_EQ(ret, -1);

    DONE();
}
