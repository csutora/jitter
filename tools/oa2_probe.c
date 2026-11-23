#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/syscall.h>

struct open_how { unsigned long long flags, mode, resolve; };
#define R_NO_XDEV 0x01
#define R_NO_MAGICLINKS 0x02
#define R_NO_SYMLINKS 0x04
#define R_BENEATH 0x08
#define R_IN_ROOT 0x10

static void td(const char *what, int dirfd, const char *path, unsigned long long resolve) {
	struct open_how how; memset(&how, 0, sizeof how);
	how.flags = O_RDONLY; how.resolve = resolve;
	errno = 0;
	long r = syscall(SYS_openat2, dirfd, path, &how, sizeof how);
	printf("%-22s %-16s -> %ld errno=%d (%s)\n", what, path, r, r < 0 ? errno : 0, r < 0 ? strerror(errno) : "ok");
	if (r >= 0) close(r);
}

static void t(const char *what, const char *path, unsigned long long resolve) {
	struct open_how how; memset(&how, 0, sizeof how);
	how.flags = O_RDONLY; how.resolve = resolve;
	errno = 0;
	long r = syscall(SYS_openat2, AT_FDCWD, path, &how, sizeof how);
	printf("%-22s %-16s -> %ld errno=%d (%s)\n", what, path, r, r < 0 ? errno : 0, r < 0 ? strerror(errno) : "ok");
	if (r >= 0) close(r);
}

int main(void) {
	char cwd[4096];
	printf("cwd=%s\n", getcwd(cwd, sizeof cwd));
	t("plain", "/proc/version", 0);
	t("no_xdev", "/proc/version", R_NO_XDEV);
	t("no_magiclinks", "/proc/self/exe", R_NO_MAGICLINKS);
	t("beneath-abs", "/proc/version", R_BENEATH);
	t("beneath-dotdot", "../foo", R_BENEATH);
	t("in_root-abs", "/proc/version", R_IN_ROOT);
	// decisive: a name that exists only in the cwd. if the walk starts at "/" instead, ENOENT.
	int fd = open("uniqzz", O_CREAT | O_RDWR, 0600);
	if (fd >= 0) close(fd);
	t("beneath-cwdfile", "uniqzz", R_BENEATH);
	t("plain-cwdfile", "uniqzz", 0);
	printf("cwd-after=%s\n", getcwd(cwd, sizeof cwd));
	int dfd = open(".", O_RDONLY | O_DIRECTORY);
	printf("dirfd=%d\n", dfd);
	td("beneath-dirfd-file", dfd, "uniqzz", R_BENEATH);
	td("beneath-dirfd-dotdot", dfd, "../foo", R_BENEATH);
	return 0;
}
