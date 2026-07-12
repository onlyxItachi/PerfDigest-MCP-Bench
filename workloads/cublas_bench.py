"""cuBLAS digest workload — closed-source vendor kernels for the perfdigest A/B.

The point of this workload: when torch dispatches a matmul, the kernels that
actually run are cuBLAS/cuBLASLt internals (mangled names like
`ampere_h16816gemm_...` or `nvjet_...`) whose source nobody can read. An agent
diagnosing them has ONLY the counters — exactly the scenario perfdigest's
digest tier exists for.

Four GEMM configs on one 2048x2048x2048 shape so the capture shows how the
library picks different kernel families per precision path:
  * fp32, TF32 off  -> classic sgemm path
  * fp32, TF32 on   -> tensor-core tf32 path
  * fp16            -> half tensor-core gemm
  * bf16            -> bfloat16 tensor-core gemm

Capture (stdout stays quiet by design — perfdigest rule 7):
  ncu -c 24 --set full -f -o <out> <venv-python> workloads/cublas_bench.py
Then digest with format='ncu-rep'. The surrounding randn/cast kernels are
deliberate noise: summarize_report must surface the gemms on its own.
"""

import torch

N = 2048
ITERS = 3


def gemm(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype, tf32: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = tf32
    x = a.to(dtype)
    y = b.to(dtype)
    for _ in range(ITERS):
        _ = x @ y
    torch.cuda.synchronize()


def main() -> None:
    torch.manual_seed(0)
    a = torch.randn(N, N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, N, device="cuda", dtype=torch.float32)

    gemm(a, b, torch.float32, tf32=False)
    gemm(a, b, torch.float32, tf32=True)
    gemm(a, b, torch.float16, tf32=False)
    gemm(a, b, torch.bfloat16, tf32=False)


if __name__ == "__main__":
    main()
