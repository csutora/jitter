// thread_probe2: isolate the contended-mutex failure. unbuffered markers via write(2) so
// output survives a crash; each stage narrows spawn-count vs shared-writes vs futex contention.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

static long counter = 0;
static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
static void M(const char *s) { write(1, s, strlen(s)); }

static void *noop(void *a) { (void)a; volatile long x = 0; for (int i = 0; i < 2000; i++) x += i; return NULL; }
static void *racy(void *a) { (void)a; for (int i = 0; i < 2000; i++) counter++; return NULL; }
static void *mtxw(void *a) { (void)a; for (int i = 0; i < 2000; i++) { pthread_mutex_lock(&mtx); counter++; pthread_mutex_unlock(&mtx); } return NULL; }

static void run(const char *name, void *(*fn)(void *), int n) {
	M("T "); M(name); M(" start\n");
	pthread_t t[8];
	for (int i = 0; i < n; i++) {
		int rc = pthread_create(&t[i], NULL, fn, NULL);
		if (rc) { M("  pthread_create FAIL\n"); }
	}
	for (int i = 0; i < n; i++) pthread_join(t[i], NULL);
	M("T "); M(name); M(" done\n");
}

int main(void) {
	setvbuf(stdout, NULL, _IONBF, 0);
	run("noop4", noop, 4);          // spawn+join 4 threads, no sharing
	run("racy4", racy, 4);          // 4 threads write a shared global (no lock)
	counter = 0; run("mtx2", mtxw, 2);   // 2 threads, contended mutex (futex)
	counter = 0; run("mtx4", mtxw, 4);   // 4 threads, contended mutex (the failing case)
	char b[64]; snprintf(b, sizeof b, "counter=%ld (want 8000)\n", counter); M(b);
	M("ALL DONE\n");
	return 0;
}
