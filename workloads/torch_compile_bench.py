"""torch.compile digest workload — Inductor/Triton generated-kernel names.

Inductor emits kernels with machine-generated names (`triton_poi_fused_add_mul_
relu_sigmoid_0`, `triton_per_fused__softmax_1`, ...) that change whenever the
fused graph changes. That stresses two perfdigest seams at once:

  * chrome_trace aggregation — one trace mixes framework_op spans (aten/dynamo
    bookkeeping) with gpu_kernel spans whose names came out of a code generator;
  * cross-report matching — an eager trace and a compiled trace of the SAME
    model share almost no unit names, which is exactly the `kernel_b` override
    scenario compare_metrics was built for.

The model body is a fusable pointwise chain + a matmul + a softmax reduction,
so Inductor produces at least one `poi` (pointwise) and one `per` (persistent
reduction) kernel and keeps the mm as an extern cuBLAS call.

Usage:
  python torch_compile_bench.py --mode compile --trace compiled.torch-trace.json
  python torch_compile_bench.py --mode eager   --trace eager.torch-trace.json
  ncu -c 24 --set full -f -o inductor python torch_compile_bench.py --mode compile
"""

import argparse

import torch

N = 1024
ITERS = 8


def model(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.relu(x) * torch.sigmoid(y) + x
    w = z @ y
    return torch.softmax(w, dim=-1).sum(dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eager", "compile"], default="compile")
    parser.add_argument("--trace", help="export a Kineto chrome trace to this path")
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn(N, N, device="cuda")
    y = torch.randn(N, N, device="cuda")

    fn = torch.compile(model) if args.mode == "compile" else model

    # Warmup runs compilation OUTSIDE the profiled region so the trace shows
    # steady-state kernels, not Inductor's one-time autotuning launches.
    for _ in range(3):
        fn(x, y)
    torch.cuda.synchronize()

    if args.trace:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(ITERS):
                fn(x, y)
            torch.cuda.synchronize()
        prof.export_chrome_trace(args.trace)
    else:
        for _ in range(ITERS):
            fn(x, y)
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
