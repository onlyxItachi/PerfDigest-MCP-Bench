"""torch.compile TRAINING-step digest workload — demanding workloads pt. 2.

Part 1 (`torch_compile_bench.py`) measured inference forward only. Training is
the harder subject: the compiled graph gains a BACKWARD pass (Inductor emits
`triton_*_backward`-flavored fused kernels reached through the AOTAutograd
joint graph) and an optimizer step (`_foreach_` multi-tensor-apply kernels).
That stresses digest ranking on many more units, kernel_b renames between
eager and compiled backward, and bookkeeping tagging on autograd spans.

Model: 2-layer MLP (1024 -> 2048 -> 1024) with LayerNorm + GELU between the
layers, batch 256, MSE loss (regression target of the same 1024-wide shape as
the output — the natural loss for this shape, and it keeps the reduction
kernel distinct from part 1's softmax-sum so this study adds a new unit
rather than repeating one), AdamW(foreach=True). One full step = forward,
loss, backward, optimizer.step, zero_grad.

**--mode compile compiles the MODEL only** (`torch.compile(model)`), not the
whole step function. Measured before deciding (see the probe in this repo's
scratch history): wrapping the whole step (forward+backward+opt.step) in
`torch.compile` graph-breaks at `loss.backward()` itself ("Unsupported
Tensor.backward() call" — Dynamo cannot trace the imperative backward() call
without `trace_autograd_ops`), AND it additionally sweeps optimizer.step()'s
`_foreach_*` ops into a *second*, separately Inductor-compiled graph — which
would contaminate exactly the eager-vs-compiled optimizer comparison this
study wants to make. Compiling just the model still gets Inductor-compiled
forward AND backward kernels (AOTAutograd emits the backward joint graph at
compile time regardless of whether the `.backward()` call site itself is
traced), while leaving `opt.step()` as genuine, undisturbed eager
multi_tensor_apply `_foreach_*` kernels in BOTH modes — a clean, single-axis
A/B on the model compilation only.

Usage:
  python torch_train_bench.py --mode compile --trace compiled.torch-trace.json
  python torch_train_bench.py --mode eager   --trace eager.torch-trace.json
  ncu -k "regex:triton" -c 24 --set full -f -o train_triton \
      python torch_train_bench.py --mode compile
"""

import argparse
import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress noisy stdout (compile/autotune warnings, TF32 advisory, SM-count
# advisory) so a no-trace ncu capture run stays quiet, per house rule 7.
warnings.filterwarnings("ignore")
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

BATCH = 256
IN_DIM = 1024
HIDDEN = 2048
OUT_DIM = 1024
ITERS = 8
WARMUP = 5  # compile + autotune happen here, OUTSIDE the profiled region


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, HIDDEN)
        self.ln1 = nn.LayerNorm(HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.ln1(h)
        h = F.gelu(h)
        return self.fc2(h)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eager", "compile"], default="compile")
    parser.add_argument("--trace", help="export a Kineto chrome trace to this path")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = MLP().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
    fn = torch.compile(model) if args.mode == "compile" else model

    x = torch.randn(BATCH, IN_DIM, device="cuda")
    target = torch.randn(BATCH, OUT_DIM, device="cuda")

    def step() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        out = fn(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt.step()
        return loss

    # Warmup runs compilation OUTSIDE the profiled region so the trace shows
    # steady-state kernels, not Inductor's one-time autotuning launches.
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()

    if args.trace:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(ITERS):
                step()
            torch.cuda.synchronize()
        prof.export_chrome_trace(args.trace)
    else:
        for _ in range(ITERS):
            step()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
