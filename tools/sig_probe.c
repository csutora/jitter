#define _GNU_SOURCE
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <string.h>

static volatile sig_atomic_t got;
static void onchld(int s) { (void)s; got = 1; }

int main(void) {
	struct sigaction sa, old;
	memset(&sa, 0, sizeof sa);
	sa.sa_handler = onchld;
	sigaction(SIGCHLD, &sa, NULL);
	sigaction(SIGCHLD, NULL, &old);
	printf("handler installed: %p (want %p)\n", (void *)old.sa_handler, (void *)onchld);
	fflush(stdout);

	sigset_t block, orig, wait;
	sigemptyset(&block); sigaddset(&block, SIGCHLD);
	sigprocmask(SIG_BLOCK, &block, &orig);
	wait = orig; sigdelset(&wait, SIGCHLD);

	pid_t p = fork();
	if (p == 0) { _exit(0); }
	printf("forked %d, suspending\n", (int)p);
	fflush(stdout);
	alarm(5);                       // so a hang ends rather than lasting forever
	sigsuspend(&wait);
	printf("suspend returned, got=%d\n", (int)got);
	int st = 0;
	printf("waitpid=%d\n", (int)waitpid(-1, &st, WNOHANG));
	fflush(stdout);
	return 0;
}
