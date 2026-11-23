#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <unistd.h>
#include <sys/epoll.h>

// mimics the phases of ltp's epoll-ltp one at a time, so the phase that takes the app down can be
// found without running the whole thing. argv[1] picks the phase.
static void phase_create(void) {
	printf("create(-1)=%d\n", epoll_create(-1));
	for (int i = 0; i < 16; i++) {
		int fd = epoll_create(INT_MAX - i);
		printf("create(INT_MAX-%d)=%d\n", i, fd);
		if (fd >= 0) close(fd);
	}
	fflush(stdout);
}

static void phase_ctl(int useUnaligned, int useNull) {
	int ep = epoll_create1(0);
	int fds[2] = { -1, INT_MAX };
	int eps[4] = { 0, -1, 0, INT_MAX };
	int ops[9] = { EPOLL_CTL_ADD, EPOLL_CTL_MOD, EPOLL_CTL_DEL, INT_MAX, -1,
	               EPOLL_CTL_ADD, EPOLL_CTL_MOD, EPOLL_CTL_DEL, 0 };
	char raw[sizeof(struct epoll_event) + 8];
	struct epoll_event *aligned = (struct epoll_event *)raw;
	struct epoll_event *unaligned = (struct epoll_event *)(raw + 1);
	long calls = 0;
	for (int e = 0; e < 64; e++) {
		for (int p = 0; p < 3; p++) {
			struct epoll_event *ev = p == 0 ? aligned : (p == 1 ? unaligned : NULL);
			if (p == 1 && !useUnaligned) continue;
			if (p == 2 && !useNull) continue;
			if (ev) { memset(ev, 0, sizeof(*ev)); ev->events = (uint32_t)e; }
			for (int f = 0; f < 2; f++) {
				for (int o = 0; o < 9; o++) {
					for (int x = 0; x < 4; x++) {
						epoll_ctl(x == 0 ? ep : eps[x], ops[o], fds[f], ev);
						calls++;
					}
				}
			}
		}
	}
	printf("ctl calls=%ld ok\n", calls);
	fflush(stdout);
	close(ep);
}

int main(int argc, char **argv) {
	int phase = argc > 1 ? atoi(argv[1]) : 0;
	switch (phase) {
	case 1: phase_create(); break;
	case 2: phase_ctl(0, 0); break;
	case 3: phase_ctl(1, 0); break;
	case 4: phase_ctl(0, 1); break;
	default: phase_create(); phase_ctl(1, 1); break;
	}
	printf("phase %d done\n", phase);
	return 0;
}
