/* cpu_bench.c — two hot functions with opposite microarchitectural character,
 * so the perfdigest digest of a `perf` report shows a clean cache-bound vs
 * compute-bound contrast on the CPU vocabulary (ipc, cache_miss_rate, self_pct).
 *
 *   mem_bound_chase   : pointer-chase over a buffer larger than LLC
 *                       -> high cache_miss_rate, low IPC.
 *   compute_bound_poly: tight in-register FMA-style polynomial
 *                       -> high IPC, near-zero cache misses.
 *
 * Build:  cc -O2 -fno-inline -fno-omit-frame-pointer -o cpu_bench cpu_bench.c -lm
 * stat:   perf stat -j -o cpu_stat.json -- ./cpu_bench
 * report: perf record -e cycles:u -g -o perf.data -- ./cpu_bench
 *         perf report -i perf.data --stdio -n > cpu_report.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define BUF (128u * 1024u * 1024u / sizeof(size_t))   /* 128 MB >> LLC */

__attribute__((noinline))
static size_t mem_bound_chase(size_t *idx, size_t steps) {
    size_t p = 0, acc = 0;
    for (size_t s = 0; s < steps; ++s) {
        p = idx[p];          /* dependent load -> serialized cache misses */
        acc += p;
    }
    return acc;
}

__attribute__((noinline))
static double compute_bound_poly(double x, size_t iters) {
    double a = x, c = 1.0000001;
    for (size_t k = 0; k < iters; ++k) {
        a = a * c + 0.5;     /* dependent FMA chain, all in registers */
        a = a * 0.9999999 + 0.25;
    }
    return a;
}

int main(void) {
    size_t n = BUF;
    size_t *idx = malloc(n * sizeof(size_t));
    if (!idx) return 1;
    /* a single random-ish cycle so the prefetcher cannot help */
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    for (size_t i = n - 1; i > 0; --i) {
        size_t j = (size_t)((i * 2654435761u) % (i + 1));
        size_t t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }

    volatile size_t sink1 = 0;
    volatile double sink2 = 0.0;
    for (int rep = 0; rep < 6; ++rep) {
        sink1 += mem_bound_chase(idx, 40u * 1000u * 1000u);
        sink2 += compute_bound_poly((double)rep + 1.0, 400u * 1000u * 1000u);
    }
    printf("done %zu %f\n", (size_t)sink1, (double)sink2);
    free(idx);
    return 0;
}
