// thread_probe: exercises the real device threading path (pthreads sharing one
// address space via web-worker + SharedArrayBuffer) to surface gaps the bundled
// ltp set can't (it has only gettid). prints PASS/FAIL per property.
//
// cross-compile: aarch64-unknown-linux-musl-gcc -static -O2 -pthread thread_probe.c -o thread_probe
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <sys/syscall.h>

static int shared_global = 0;
static __thread int tls_var = 0;
static int g_fd = -1;
static long fd_write_ret = -777;
static pid_t main_tid, thr_tid;

static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
static long counter = 0;

static void *worker(void *arg) {
	(void)arg;
	tls_var = 42;                       // must NOT clobber main's tls
	shared_global = 123;                // shared address space
	thr_tid = (pid_t)syscall(SYS_gettid);
	const char *msg = "thread-wrote-this\n";
	ssize_t n = write(g_fd, msg, strlen(msg));   // shared fd table?
	fd_write_ret = (n < 0) ? -errno : n;
	return NULL;
}

static void *bumper(void *arg) {
	(void)arg;
	for (int i = 0; i < 100000; i++) {
		pthread_mutex_lock(&mtx);
		counter++;
		pthread_mutex_unlock(&mtx);
	}
	return NULL;
}

#define CHECK(cond, name, ...) \
	do { printf("%s %s ", (cond) ? "PASS" : "FAIL", name); printf(__VA_ARGS__); printf("\n"); } while (0)

int main(void) {
	main_tid = (pid_t)syscall(SYS_gettid);
	tls_var = 7;

	g_fd = open("/tmp/probe_shared.txt", O_RDWR | O_CREAT | O_TRUNC, 0644);
	printf("open /tmp/probe_shared.txt -> fd=%d (errno=%d)\n", g_fd, errno);

	pthread_t t;
	int rc = pthread_create(&t, NULL, worker, NULL);
	CHECK(rc == 0, "pthread_create", "(rc=%d errno=%d)", rc, errno);
	if (rc != 0) return 1;
	rc = pthread_join(t, NULL);
	CHECK(rc == 0, "pthread_join", "(rc=%d)", rc);

	CHECK(shared_global == 123, "shared_global", "(=%d want 123)", shared_global);
	CHECK(tls_var == 7, "tls_isolation", "(main tls=%d want 7, thread set its own to 42)", tls_var);
	CHECK(thr_tid != 0 && thr_tid != main_tid, "gettid_distinct", "(main=%d thread=%d)", main_tid, thr_tid);
	CHECK(fd_write_ret == 18, "shared_fd_write", "(write ret=%ld want 18)", fd_write_ret);

	char buf[64] = {0};
	lseek(g_fd, 0, SEEK_SET);
	ssize_t rn = read(g_fd, buf, sizeof(buf) - 1);
	CHECK(rn > 0 && strstr(buf, "thread-wrote") != NULL, "shared_fd_readback", "(read %zd: '%.24s')", rn, buf);

	pthread_t th[4];
	for (int i = 0; i < 4; i++) pthread_create(&th[i], NULL, bumper, NULL);
	for (int i = 0; i < 4; i++) pthread_join(th[i], NULL);
	CHECK(counter == 400000, "mutex_counter", "(=%ld want 400000)", counter);

	return 0;
}
