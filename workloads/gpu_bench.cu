// gpu_bench.cu — two kernels with deliberately opposite roofline character,
// so the perfdigest digest of an ncu --set full report shows a clean
// memory-bound vs compute-bound contrast.
//
//   saxpy_mem_bound   : y = a*x + y  (2 loads + 1 store, 2 FLOP/elem)
//                       -> tiny arithmetic intensity -> DRAM-bandwidth bound.
//   fma_compute_bound : long in-register FMA chain, ~no global traffic
//                       -> SM-throughput bound, DRAM near-idle.
//
// Build:  nvcc -O3 -arch=sm_89 -o gpu_bench gpu_bench.cu
// Profile: ncu --set full -o gpu_bench.ncu-rep ./gpu_bench   (stdout suppressed to file)
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 1; } } while (0)

__global__ void saxpy_mem_bound(float a, const float* __restrict__ x,
                                float* __restrict__ y, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

__global__ void fma_compute_bound(const float* __restrict__ x,
                                  float* __restrict__ y, long n, int iters) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = x[i];
    float c = 1.0000001f;
    #pragma unroll 8
    for (int k = 0; k < iters; ++k) {
        acc = fmaf(acc, c, 0.5f);   // stays in registers; no global traffic
    }
    y[i] = acc;                     // single store keeps the kernel honest
}

int main() {
    const long n = 32L * 1024 * 1024;        // 32M floats = 128 MB per array
    const size_t bytes = n * sizeof(float);
    float *dx, *dy;
    CHECK(cudaMalloc(&dx, bytes));
    CHECK(cudaMalloc(&dy, bytes));
    CHECK(cudaMemset(dx, 1, bytes));
    CHECK(cudaMemset(dy, 2, bytes));

    const int tpb = 256;
    const int blocks = (int)((n + tpb - 1) / tpb);

    // warm up (not profiled meaningfully; ncu replays the measured launches)
    saxpy_mem_bound<<<blocks, tpb>>>(2.0f, dx, dy, n);
    fma_compute_bound<<<blocks, tpb>>>(dx, dy, n, 512);
    CHECK(cudaDeviceSynchronize());

    // measured launches
    saxpy_mem_bound<<<blocks, tpb>>>(2.0f, dx, dy, n);
    fma_compute_bound<<<blocks, tpb>>>(dx, dy, n, 512);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(dx));
    CHECK(cudaFree(dy));
    printf("done n=%ld\n", n);
    return 0;
}
