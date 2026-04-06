/* Minimal TAP (Test Anything Protocol) test harness for jitter conformance tests.
 *
 * Usage:
 *   #include "harness.h"
 *   int main(void) {
 *     TEST("open returns valid fd");
 *     int fd = open("/etc/passwd", O_RDONLY);
 *     ASSERT(fd >= 0);
 *     DONE();
 *   }
 *
 * Output (TAP format):
 *   # Test 1: open returns valid fd
 *   ok 1
 *   1..1
 */

#ifndef HARNESS_H
#define HARNESS_H

#include <stdio.h>
#include <stdlib.h>

static int _test_num = 0;
static int _test_fail = 0;

#define TEST(name) do { \
    _test_num++; \
    fprintf(stderr, "# Test %d: %s\n", _test_num, (name)); \
} while (0)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "not ok %d - %s (at %s:%d)\n", _test_num, #cond, __FILE__, __LINE__); \
        _test_fail++; \
    } else { \
        fprintf(stderr, "ok %d\n", _test_num); \
    } \
} while (0)

#define ASSERT_EQ(a, b) do { \
    long long _a = (long long)(a); \
    long long _b = (long long)(b); \
    if (_a != _b) { \
        fprintf(stderr, "not ok %d - %s == %s (got %lld, expected %lld) at %s:%d\n", \
                _test_num, #a, #b, _a, _b, __FILE__, __LINE__); \
        _test_fail++; \
    } else { \
        fprintf(stderr, "ok %d\n", _test_num); \
    } \
} while (0)

#define DONE() do { \
    fprintf(stderr, "1..%d\n", _test_num); \
    fprintf(stderr, "# %d passed, %d failed\n", _test_num - _test_fail, _test_fail); \
    return _test_fail ? 1 : 0; \
} while (0)

#endif /* HARNESS_H */
