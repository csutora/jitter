// does a sleep return when it should? kill12 spends 26 of its 30 second budget asleep by design,
// so a sleep that overshoots is the difference between passing and timing out.
#include <stdio.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

int main(void) {
    setbuf(stdout, NULL);
    const int ms[] = {1, 10, 100, 1000};
    for (int i = 0; i < 4; i++) {
        struct timespec req = { ms[i] / 1000, (long)(ms[i] % 1000) * 1000000L };
        double a = now_ms();
        nanosleep(&req, 0);
        double got = now_ms() - a;
        printf("sleep %5d ms -> %8.2f ms (%+.1f%%)\n", ms[i], got, (got - ms[i]) * 100.0 / ms[i]);
    }
    return 0;
}
