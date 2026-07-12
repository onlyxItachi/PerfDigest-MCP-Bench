/* simd_bench.c — scalar vs vector ISA contrast for the linux_perf digest study.
 *
 * Two units doing the SAME fma-heavy work over the same buffers:
 *   scalar_fma_loop : vectorization disabled per-function -> scalar SSE math
 *   vector_fma_loop : -O3 autovectorized -> AVX/AVX-512 on capable hosts
 *                     (Zen 5 has a full 512-bit datapath)
 *
 * What this stresses in perfdigest: the linux_perf vocabulary has no SIMD
 * terms yet, so the interesting counters only exist behind vendor event names
 * that differ per CPU maker — AMD Zen: fp_ret_sse_avx_ops.all; Intel:
 * fp_arith_inst_retired.{scalar_single,128b_packed_single,256b_packed_single,
 * 512b_packed_single}. The digest study documents the gap and the expand()
 * escape hatch; a v1.2 vocabulary addition proposal comes out of it.
 *
 * Build:   cc -O3 -march=native -fno-inline -fno-omit-frame-pointer -o simd_bench simd_bench.c -lm
 * Capture (needs kernel.perf_event_paranoid<=2):
 *   AMD:   LC_ALL=C perf stat -j -o simd_stat.json \
 *            -e task-clock,cycles,instructions,fp_ret_sse_avx_ops.all -- ./simd_bench
 *   Intel: ... -e task-clock,cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.256b_packed_single -- ./simd_bench
 *   perf record -e cycles:u -g -o simd.data -- ./simd_bench
 *   LC_ALL=C perf report -i simd.data --stdio -n --no-children -g none --percent-limit 0.5 > simd_report.txt
 */

#include <stdio.h>
#include <stdlib.h>

#define N (1 << 22) /* 4M floats = 16 MB, larger than L3 slice, streaming */
#define REPS 40

#if defined(__clang__)
#define NO_VECTORIZE
#define NO_VEC_LOOP _Pragma("clang loop vectorize(disable) interleave(disable)")
#else
#define NO_VECTORIZE __attribute__((optimize("no-tree-vectorize")))
#define NO_VEC_LOOP
#endif

NO_VECTORIZE
static float scalar_fma_loop(const float *restrict x, float *restrict y) {
    float acc = 0.0f;
    for (int r = 0; r < REPS; r++) {
        NO_VEC_LOOP
        for (int i = 0; i < N; i++) {
            y[i] = 1.0000001f * y[i] + x[i];
            acc += y[i];
        }
    }
    return acc;
}

static float vector_fma_loop(const float *restrict x, float *restrict y) {
    float acc = 0.0f;
    for (int r = 0; r < REPS; r++) {
        for (int i = 0; i < N; i++) {
            y[i] = 1.0000001f * y[i] + x[i];
            acc += y[i];
        }
    }
    return acc;
}

int main(void) {
    float *x = malloc(N * sizeof(float));
    float *y = malloc(N * sizeof(float));
    if (!x || !y) return 1;
    for (int i = 0; i < N; i++) {
        x[i] = (float)(i & 1023) * 0.001f;
        y[i] = 1.0f;
    }
    float a = scalar_fma_loop(x, y);
    float b = vector_fma_loop(x, y);
    /* print so the work cannot be optimized away */
    fprintf(stderr, "checksums: scalar=%.3f vector=%.3f\n", a, b);
    free(x);
    free(y);
    return 0;
}
