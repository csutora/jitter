#include "../harness.h"
#include <time.h>
#include <sys/time.h>

int main(void) {
    struct timespec ts;

    TEST("clock_gettime CLOCK_REALTIME succeeds");
    int ret = clock_gettime(CLOCK_REALTIME, &ts);
    ASSERT_EQ(ret, 0);

    TEST("CLOCK_REALTIME returns nonzero seconds");
    ASSERT(ts.tv_sec > 1000000000); /* after ~2001 */

    TEST("clock_gettime CLOCK_MONOTONIC succeeds");
    ret = clock_gettime(CLOCK_MONOTONIC, &ts);
    ASSERT_EQ(ret, 0);

    struct timeval tv;
    TEST("gettimeofday succeeds");
    ret = gettimeofday(&tv, NULL);
    ASSERT_EQ(ret, 0);

    TEST("gettimeofday returns nonzero seconds");
    ASSERT(tv.tv_sec > 1000000000);

    DONE();
}
