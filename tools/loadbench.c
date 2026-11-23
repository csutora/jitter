// what the last two ltp holdouts actually need, measured rather than guessed: fcntl14 wants ~1800
// forks in 20 s, and kill12 wants sleeps that do not overshoot a 30 s budget by design.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    int n = argc > 1 ? atoi(argv[1]) : 200;

    double a = now_ms();
    for (int i = 0; i < n; i++) {
        pid_t p = fork();
        if (p == 0) _exit(0);
        if (p < 0) { printf("fork failed at %d\n", i); return 1; }
        int st; waitpid(p, &st, 0);
    }
    double per = (now_ms() - a) / n;
    printf("fork+exit+wait   %6.2f ms each -> %.0f/s (fcntl14 needs ~91/s)\n", per, 1000.0 / per);

    // how many can be ALIVE at once, which is what pipe13 and fcntl14 both lean on
    int alive = 0;
    a = now_ms();
    for (int i = 0; i < n; i++) {
        pid_t p = fork();
        if (p == 0) { pause(); _exit(0); }
        if (p < 0) break;
        alive++;
    }
    printf("concurrent forks %d in %.0f ms\n", alive, now_ms() - a);
    for (int i = 0; i < alive; i++) wait(NULL); // they are paused; reaped after the kill below
    return 0;
}
