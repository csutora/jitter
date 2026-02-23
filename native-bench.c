#include <stdio.h>
#include <stdint.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// ── Fibonacci(40) ───────────────────────────────────────────────────────────

// Prevent the compiler from optimizing away the loop body
#define CLOBBER() __asm__ volatile("" ::: "memory")

__attribute__((noinline))
static uint64_t fib(int n) {
    uint64_t a = 1, b = 0;
    for (int i = n; i > 1; i--) {
        uint64_t t = a + b;
        b = a;
        a = t;
        CLOBBER();
    }
    return a;
}

// ── Sum 1..N ────────────────────────────────────────────────────────────────

__attribute__((noinline))
static uint64_t sum(uint64_t n) {
    uint64_t s = 0, i = 1;
    while (i <= n) {
        s += i;
        i++;
        CLOBBER();
    }
    return s;
}

// ── Sum of squares 1..N ─────────────────────────────────────────────────────

__attribute__((noinline))
static uint64_t sumsq(uint64_t n) {
    uint64_t s = 0, i = 1;
    while (i <= n) {
        s += i * i;
        i++;
        CLOBBER();
    }
    return s;
}

static double median(double *arr, int n) {
    // simple insertion sort
    for (int i = 1; i < n; i++) {
        double key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) { arr[j+1] = arr[j]; j--; }
        arr[j+1] = key;
    }
    return (n & 1) ? arr[n/2] : (arr[n/2 - 1] + arr[n/2]) / 2.0;
}

#define SAMPLES 11

int main(void) {
    double times[SAMPLES];
    double med, ips;

    // ── Fibonacci(40) ───────────────────────────────────────────────
    // Very fast, so batch 1M repetitions per sample
    {
        uint64_t result = 0;
        int batch = 1000000;
        for (int s = 0; s < SAMPLES; s++) {
            double t0 = now_ms();
            for (int b = 0; b < batch; b++) {
                result = fib(40);
            }
            times[s] = (now_ms() - t0) / batch;
        }
        int insns = 238; // same as jitter: 39 iters * 6 insns/iter + overhead
        med = median(times, SAMPLES);
        ips = (insns / med) * 1000.0;
        printf("Fibonacci(40)\n");
        printf("  Result:     %llu (expected 102334155)\n", (unsigned long long)result);
        printf("  Median:     %.4f ms  (%d samples, %d runs/sample)\n", med, SAMPLES, batch);
        printf("  Throughput: %.1fM IPS  (%d insns/run)\n\n", ips / 1e6, insns);
    }

    // ── Sum 1..100M ─────────────────────────────────────────────────
    {
        uint64_t n = 100000000ULL;
        uint64_t result = 0;
        int insns = (int)(n * 4 + 1); // 4 insns/iter + halt
        for (int s = 0; s < SAMPLES; s++) {
            double t0 = now_ms();
            result = sum(n);
            times[s] = now_ms() - t0;
        }
        med = median(times, SAMPLES);
        ips = (insns / med) * 1000.0;
        printf("Sum 1..100M\n");
        printf("  Result:     %llu (expected 5000000050000000)\n", (unsigned long long)result);
        printf("  Median:     %.3f ms  (%d samples)\n", med, SAMPLES);
        printf("  Throughput: %.2fG IPS  (%d insns/run)\n\n", ips / 1e9, insns);
    }

    // ── Sum of Squares 1..10M ───────────────────────────────────────
    {
        uint64_t n = 10000000ULL;
        uint64_t result = 0;
        int insns = (int)(n * 4 + 1); // 4 insns/iter + halt
        for (int s = 0; s < SAMPLES; s++) {
            double t0 = now_ms();
            result = sumsq(n);
            times[s] = now_ms() - t0;
        }
        med = median(times, SAMPLES);
        ips = (insns / med) * 1000.0;
        printf("Sum of Squares 1..10M\n");
        printf("  Result:     %llu (expected 1291990006563070912)\n", (unsigned long long)result);
        printf("  Median:     %.3f ms  (%d samples)\n", med, SAMPLES);
        printf("  Throughput: %.2fG IPS  (%d insns/run)\n", ips / 1e9, insns);
    }

    return 0;
}
