#include "../harness.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int main(void) {
    TEST("mkdir /tmp/testdir succeeds");
    int ret = mkdir("/tmp/testdir", 0755);
    ASSERT_EQ(ret, 0);

    TEST("stat /tmp/testdir shows directory");
    struct stat st;
    ret = stat("/tmp/testdir", &st);
    ASSERT_EQ(ret, 0);
    ASSERT(S_ISDIR(st.st_mode));

    TEST("create file in new dir");
    int fd = open("/tmp/testdir/file.txt", O_WRONLY | O_CREAT, 0644);
    ASSERT(fd >= 0);
    write(fd, "test", 4);
    close(fd);

    TEST("unlink file succeeds");
    ret = unlink("/tmp/testdir/file.txt");
    ASSERT_EQ(ret, 0);

    TEST("stat unlinked file fails");
    ret = stat("/tmp/testdir/file.txt", &st);
    ASSERT_EQ(ret, -1);

    TEST("rmdir /tmp/testdir succeeds");
    ret = rmdir("/tmp/testdir");
    ASSERT_EQ(ret, 0);

    DONE();
}
