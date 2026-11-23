#define _GNU_SOURCE
#include <stdio.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <sys/syscall.h>

static const char *names[] = {"REALTIME","MONOTONIC","PROCESS_CPUTIME","THREAD_CPUTIME","MONOTONIC_RAW","REALTIME_COARSE","MONOTONIC_COARSE","BOOTTIME","REALTIME_ALARM","BOOTTIME_ALARM","(10)","TAI"};

int main(void) {
	for (int c = 0; c < 12; c++) {
		struct timespec a, b, r;
		errno = 0;
		long ra = syscall(SYS_clock_gettime, c, &a);
		long rr = syscall(SYS_clock_getres, c, &r);
		if (ra != 0) { printf("%-16s gettime=%ld errno=%d\n", names[c], ra, errno); continue; }
		for (volatile long i = 0; i < 3000000; i++) ;
		syscall(SYS_clock_gettime, c, &b);
		long long d = (long long)(b.tv_sec - a.tv_sec) * 1000000000LL + (b.tv_nsec - a.tv_nsec);
		printf("%-16s %lld.%09ld res=%ld advanced=%lldns\n", names[c], (long long)a.tv_sec, a.tv_nsec, rr == 0 ? r.tv_nsec : -1, d);
	}
	return 0;
}
