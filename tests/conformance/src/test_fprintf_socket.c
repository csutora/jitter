#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Simulate what busybox wget does: fprintf to a FILE opened on a pipe/fd */
int main() {
    int fds[2];
    if (pipe(fds) < 0) {
        perror("pipe");
        return 1;
    }

    /* Open write end as a FILE (like busybox's fdopen on the socket) */
    FILE *sfp = fdopen(fds[1], "w");
    if (!sfp) {
        perror("fdopen");
        return 1;
    }

    /* Write exactly like busybox wget */
    const char *user_agent = "Wget";
    fprintf(sfp, "User-Agent: %s\r\n", user_agent);
    fflush(sfp);
    fclose(sfp);

    /* Read what was written */
    char buf[256];
    int n = read(fds[0], buf, sizeof(buf));
    close(fds[0]);

    printf("Read %d bytes: \"", n);
    for (int i = 0; i < n; i++) {
        if (buf[i] == '\r') printf("\\r");
        else if (buf[i] == '\n') printf("\\n");
        else if (buf[i] == 0) printf("\\0");
        else putchar(buf[i]);
    }
    printf("\"\n");
    printf("Expected 18 bytes: \"User-Agent: Wget\\r\\n\"\n");

    return (n == 18) ? 0 : 1;
}
