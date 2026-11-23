#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <time.h>

// fork as fast as the system will let us, reaping each child immediately. argv[1] = how many.
// this is the shape of ltp's epoll-ltp, which wraps every one of its 27,648 assertions in a fork.
int main(int argc, char **argv) {
	long n = argc > 1 ? atol(argv[1]) : 2000;
	struct timespec t0, t1;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	long ok = 0, fail = 0;
	for (long i = 0; i < n; i++) {
		pid_t p = fork();
		if (p == 0) _exit(0);
		if (p < 0) { fail++; if (fail < 4) perror("fork"); continue; }
		int st; waitpid(p, &st, 0);
		ok++;
		if ((i % 250) == 0) { printf("%ld ok=%ld fail=%ld\n", i, ok, fail); fflush(stdout); }
	}
	clock_gettime(CLOCK_MONOTONIC, &t1);
	double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
	printf("done ok=%ld fail=%ld in %.1fs (%.0f forks/s)\n", ok, fail, secs, ok / secs);
	return 0;
}
