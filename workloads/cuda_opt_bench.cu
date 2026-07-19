// cuda_opt_bench.cu — the CUDA optimization loop, in one file, three variants
// of the SAME computation (a square matrix transpose) so one `nvcc -Xptxas -v`
// compile gives ptxas codegen stats for all three, and one `ncu --set full`
// capture gives counters the digest tools can `compare_metrics` step-by-step.
//
//   kernel_v0_naive       : direct index-swap transpose. Global READ is
//                            coalesced (x = threadIdx.x is the fast dimension
//                            on the input row); global WRITE is NOT — the
//                            output index multiplies threadIdx.x by n, so
//                            consecutive threads in a warp write n*4 bytes
//                            apart. Textbook uncoalesced store.
//   kernel_v1_coalesced   : shared-memory tile (32x33, padded to kill bank
//                            conflicts) — the transpose happens in the
//                            *shared-memory indexing*, so BOTH the global
//                            read and the global write are per-warp
//                            contiguous. This is the classic Harris/Ruetsch
//                            NVIDIA-blog tiled transpose.
//   kernel_v2_vectorized  : v1's tile/padding, plus float4 (128-bit) vector
//                            loads AND stores on the global-memory side —
//                            one vectorized instruction moves what took 4
//                            scalar instructions in v1, and blockDim covers
//                            the whole 32-row tile directly (no BLOCK_ROWS
//                            loop needed).
//
// Matrix is square N x N float32, N = 8192 by default -> 256 MB per array,
// 8x the RTX 4060's 32 MB L2, chosen deliberately so the working set cannot
// be cache-absorbed and the coalescing penalty shows up as real DRAM traffic
// (verified against `cudaDeviceProp.l2CacheSize`, not assumed).
//
// Correctness: this is pure data movement (no arithmetic), so a correct
// transpose is EXACTLY reproducible bit-for-bit. main() runs a full
// element-wise device-side compare of v1's and v2's output against v0's
// output (not a sample) plus a random-sample direct index check against the
// untransposed input, and aborts (nonzero exit) on any mismatch. A
// position-weighted checksum per variant is also printed to stderr.
//
// Build:  nvcc -O3 -arch=sm_89 -o cuda_opt_bench cuda_opt_bench.cu
// ptxas codegen only (no run): nvcc -O3 -arch=sm_89 -Xptxas -v -c cuda_opt_bench.cu -o /dev/null
// Profile (see workloads/README.md for the full command + rationale)

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

static const int TILE_DIM = 32;
static const int BLOCK_ROWS = 8; // v0/v1 block shape (32,8) -> 4-iter loop over the 32-row tile

// ---------------------------------------------------------------------------
// v0: naive — coalesced read, uncoalesced (strided by n) write.
__global__ void kernel_v0_naive(float* __restrict__ odata,
                                 const float* __restrict__ idata, int n) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        odata[x * n + (y + j)] = idata[(y + j) * n + x];
    }
}

// ---------------------------------------------------------------------------
// v1: shared-memory tile, padded [TILE_DIM][TILE_DIM+1] to kill bank
// conflicts. Transpose happens in shared-memory indexing only — both global
// accesses are per-warp contiguous.
__global__ void kernel_v1_coalesced(float* __restrict__ odata,
                                     const float* __restrict__ idata, int n) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * n + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
}

// ---------------------------------------------------------------------------
// v2: v1's padded tile, plus float4 vectorized global loads AND stores.
// blockDim is (8,32) here (TILE_DIM/4, TILE_DIM): each thread reads one
// float4 (4 contiguous input columns) in the load phase — one 128-bit LDG
// where v1 issued 4 scalar LDGs across its BLOCK_ROWS loop. The store phase
// gathers 4 scalar shared-memory reads (a strided-by-33 tile column, not
// contiguous in shared memory, so this part stays scalar) into a register
// float4 and issues ONE 128-bit STG — again replacing 4 of v1's scalar
// stores. Requires n % 4 == 0 (checked in main()).
__global__ void kernel_v2_vectorized(float* __restrict__ odata,
                                      const float* __restrict__ idata, int n) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int tx = threadIdx.x; // 0..7  (float4 lane group)
    int ty = threadIdx.y; // 0..31 (full tile row, no loop needed)

    int x4 = blockIdx.x * TILE_DIM + tx * 4;
    int y  = blockIdx.y * TILE_DIM + ty;
    const float4* in4 = reinterpret_cast<const float4*>(idata + (size_t)y * n + x4);
    float4 v = *in4;
    tile[ty][tx * 4 + 0] = v.x;
    tile[ty][tx * 4 + 1] = v.y;
    tile[ty][tx * 4 + 2] = v.z;
    tile[ty][tx * 4 + 3] = v.w;

    __syncthreads();

    float4 outv;
    outv.x = tile[tx * 4 + 0][ty];
    outv.y = tile[tx * 4 + 1][ty];
    outv.z = tile[tx * 4 + 2][ty];
    outv.w = tile[tx * 4 + 3][ty];

    int out_col_base = blockIdx.y * TILE_DIM + tx * 4;
    int out_row       = blockIdx.x * TILE_DIM + ty;
    float4* out4 = reinterpret_cast<float4*>(odata + (size_t)out_row * n + out_col_base);
    *out4 = outv;
}

// ---------------------------------------------------------------------------
// Correctness helpers (not part of the timed region).

__global__ void init_kernel(float* data, long n_elems) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) {
        // deterministic, not a trivial constant, cheap to generate on-device
        data[i] = __sinf(i * 0.0001f) + (float)(i % 7);
    }
}

// Direct-index spot check: odata[col*n+row] must equal idata[row*n+col] for
// a strided sample of (row,col) pairs. Sets *mismatch=1 on any failure.
__global__ void spot_check_kernel(const float* idata, const float* odata,
                                   int n, long n_samples, int* mismatch) {
    long s = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_samples) return;
    // spread samples across the full n x n index space
    long lin = (s * 104729L) % ((long)n * (long)n); // 104729 is prime, decorrelates stride
    int row = (int)(lin / n);
    int col = (int)(lin % n);
    if (idata[(long)row * n + col] != odata[(long)col * n + row]) {
        atomicExch(mismatch, 1);
    }
}

// Full element-wise compare between two output buffers (checks every element).
__global__ void compare_kernel(const float* a, const float* b, long n_elems, int* mismatch) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems && a[i] != b[i]) atomicExch(mismatch, 1);
}

// Position-weighted checksum (order-sensitive enough to catch a permutation
// bug, unlike a plain sum): accumulate (double)value * ((i % 97) + 1).
__global__ void checksum_kernel(const float* data, long n_elems, double* out) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) {
        double w = (double)((i % 97) + 1);
        atomicAdd(out, (double)data[i] * w);
    }
}

static double device_checksum(const float* d, long n_elems) {
    double* d_acc;
    CHECK(cudaMalloc(&d_acc, sizeof(double)));
    CHECK(cudaMemset(d_acc, 0, sizeof(double)));
    int tpb = 256;
    long blocks = (n_elems + tpb - 1) / tpb;
    checksum_kernel<<<(unsigned)blocks, tpb>>>(d, n_elems, d_acc);
    CHECK(cudaGetLastError());
    double h_acc = 0.0;
    CHECK(cudaMemcpy(&h_acc, d_acc, sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_acc));
    return h_acc;
}

static int device_mismatch(const float* a, const float* b, long n_elems) {
    int* d_flag;
    CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CHECK(cudaMemset(d_flag, 0, sizeof(int)));
    int tpb = 256;
    long blocks = (n_elems + tpb - 1) / tpb;
    compare_kernel<<<(unsigned)blocks, tpb>>>(a, b, n_elems, d_flag);
    CHECK(cudaGetLastError());
    int h_flag = 0;
    CHECK(cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_flag));
    return h_flag;
}

static int device_spot_check(const float* idata, const float* odata, int n, long n_samples) {
    int* d_flag;
    CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CHECK(cudaMemset(d_flag, 0, sizeof(int)));
    int tpb = 256;
    long blocks = (n_samples + tpb - 1) / tpb;
    spot_check_kernel<<<(unsigned)blocks, tpb>>>(idata, odata, n, n_samples, d_flag);
    CHECK(cudaGetLastError());
    int h_flag = 0;
    CHECK(cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_flag));
    return h_flag;
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int n = 8192;
    if (argc > 1) n = atoi(argv[1]);
    if (n % TILE_DIM != 0) { fprintf(stderr, "n must be a multiple of %d\n", TILE_DIM); return 1; }
    if (n % 4 != 0)        { fprintf(stderr, "n must be a multiple of 4 (v2 float4)\n"); return 1; }

    const long n_elems = (long)n * (long)n;
    const size_t bytes = (size_t)n_elems * sizeof(float);

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "device=%s l2CacheSize=%.1f MB n=%d array_bytes=%.1f MB (%.1fx L2)\n",
            prop.name, prop.l2CacheSize / 1024.0 / 1024.0, n,
            bytes / 1024.0 / 1024.0, bytes / (double)prop.l2CacheSize);

    float *dA, *dB0, *dB1, *dB2;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB0, bytes));
    CHECK(cudaMalloc(&dB1, bytes));
    CHECK(cudaMalloc(&dB2, bytes));

    int tpb = 256;
    long init_blocks = (n_elems + tpb - 1) / tpb;
    init_kernel<<<(unsigned)init_blocks, tpb>>>(dA, n_elems);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    dim3 grid(n / TILE_DIM, n / TILE_DIM);
    dim3 block01(TILE_DIM, BLOCK_ROWS);      // v0, v1
    dim3 block2(TILE_DIM / 4, TILE_DIM);     // v2

    const int ROUNDS = 3; // launches per variant: ncu -c sized to catch all of these (see workloads/README.md)

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    // --- v0_naive ---
    float sum_ms_v0 = 0.f;
    for (int r = 0; r < ROUNDS; ++r) {
        CHECK(cudaEventRecord(t0));
        kernel_v0_naive<<<grid, block01>>>(dB0, dA, n);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaGetLastError());
        CHECK(cudaEventSynchronize(t1));
        float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
        sum_ms_v0 += ms;
        fprintf(stdout, "kernel_v0_naive      round=%d time_ms=%.4f\n", r, ms);
    }

    // --- v1_coalesced ---
    float sum_ms_v1 = 0.f;
    for (int r = 0; r < ROUNDS; ++r) {
        CHECK(cudaEventRecord(t0));
        kernel_v1_coalesced<<<grid, block01>>>(dB1, dA, n);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaGetLastError());
        CHECK(cudaEventSynchronize(t1));
        float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
        sum_ms_v1 += ms;
        fprintf(stdout, "kernel_v1_coalesced  round=%d time_ms=%.4f\n", r, ms);
    }

    // --- v2_vectorized ---
    float sum_ms_v2 = 0.f;
    for (int r = 0; r < ROUNDS; ++r) {
        CHECK(cudaEventRecord(t0));
        kernel_v2_vectorized<<<grid, block2>>>(dB2, dA, n);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaGetLastError());
        CHECK(cudaEventSynchronize(t1));
        float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
        sum_ms_v2 += ms;
        fprintf(stdout, "kernel_v2_vectorized round=%d time_ms=%.4f\n", r, ms);
    }

    fprintf(stdout, "avg_ms v0_naive=%.4f v1_coalesced=%.4f v2_vectorized=%.4f\n",
            sum_ms_v0 / ROUNDS, sum_ms_v1 / ROUNDS, sum_ms_v2 / ROUNDS);

    // --- correctness: full element-wise compare v1 vs v0, v2 vs v0 ---
    int mismatch01 = device_mismatch(dB0, dB1, n_elems);
    int mismatch02 = device_mismatch(dB0, dB2, n_elems);
    // --- correctness: direct-index spot check (independent of any variant being "the" reference) ---
    long n_samples = 2000000L;
    int spot0 = device_spot_check(dA, dB0, n, n_samples);
    int spot1 = device_spot_check(dA, dB1, n, n_samples);
    int spot2 = device_spot_check(dA, dB2, n, n_samples);

    // --- checksums (position-weighted, printed to stderr per task spec) ---
    double cs0 = device_checksum(dB0, n_elems);
    double cs1 = device_checksum(dB1, n_elems);
    double cs2 = device_checksum(dB2, n_elems);
    fprintf(stderr, "checksum v0_naive=%.6f v1_coalesced=%.6f v2_vectorized=%.6f\n", cs0, cs1, cs2);
    fprintf(stderr, "correctness: v1_vs_v0_mismatch=%d v2_vs_v0_mismatch=%d "
                     "spot_check_v0=%d spot_check_v1=%d spot_check_v2=%d "
                     "(0=PASS, 1=FAIL, %ld samples each)\n",
            mismatch01, mismatch02, spot0, spot1, spot2, n_samples);

    int failed = mismatch01 || mismatch02 || spot0 || spot1 || spot2;
    if (failed) {
        fprintf(stderr, "CORRECTNESS FAILURE — variants do not agree, or a variant does not "
                         "match the untransposed input. Aborting with nonzero exit.\n");
    } else {
        fprintf(stderr, "correctness OK: v0, v1, v2 outputs are bit-identical and match idata.\n");
    }

    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB0));
    CHECK(cudaFree(dB1));
    CHECK(cudaFree(dB2));

    return failed ? 1 : 0;
}
