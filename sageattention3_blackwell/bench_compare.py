"""
Bench: original sageattn3 vs sageattn3_simplified.
Same kernel, just docs differ. Should be near-identical perf + exact same output.

HOW IT WORKS:
  1. Run both on same random Q/K/V inputs.
  2. Compare outputs (max abs diff, mean abs diff).
  3. Time both with CUDA events (warmup + timed iters).
  4. Print TFLOPS for each.

TFLOPS FORMULA:
  Standard attention FLOPs = 4 * B * H * seqlen_q * seqlen_k * D
  (two matmuls: QK^T and PV, each is 2*B*H*sq*sk*D)
"""

import torch
import time
import argparse

# import both versions
from sageattn3 import sageattn3_blackwell as orig_attn
from sageattn3_simplified.api import sageattn3_blackwell as simp_attn


def tflops(B, H, sq, sk, D, ms):
    # 4 * B * H * sq * sk * D flops for attention
    flops = 4.0 * B * H * sq * sk * D
    return flops / (ms * 1e-3) / 1e12


def bench_one(fn, q, k, v, is_causal, warmup=10, iters=50):
    # warmup - gpu needs to warm up jit / triton
    for _ in range(warmup):
        out = fn(q.clone(), k.clone(), v.clone(), is_causal=is_causal)
    torch.cuda.synchronize()

    # timed iters with CUDA events (more accurate than time.time)
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn(q.clone(), k.clone(), v.clone(), is_causal=is_causal)
    end.record()
    torch.cuda.synchronize()

    ms_total = start.elapsed_time(end)  # ms
    ms_avg   = ms_total / iters
    return out, ms_avg


def run(B, H, sq, sk, D, dtype, is_causal):
    print(f"\n{'='*60}")
    print(f"B={B} H={H} sq={sq} sk={sk} D={D} dtype={dtype} causal={is_causal}")
    print(f"{'='*60}")

    # same random input for both
    torch.manual_seed(42)
    q = torch.randn(B, H, sq, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, sk, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, sk, D, dtype=dtype, device="cuda")

    # accuracy: run once each, compare outputs
    with torch.no_grad():
        out_orig = orig_attn(q.clone(), k.clone(), v.clone(), is_causal=is_causal)
        out_simp = simp_attn(q.clone(), k.clone(), v.clone(), is_causal=is_causal)

    diff = (out_orig.float() - out_simp.float()).abs()
    print(f"  max_diff  = {diff.max().item():.6f}")
    print(f"  mean_diff = {diff.mean().item():.8f}")

    # perf
    with torch.no_grad():
        _, ms_orig = bench_one(orig_attn, q, k, v, is_causal)
        _, ms_simp = bench_one(simp_attn, q, k, v, is_causal)

    tf_orig = tflops(B, H, sq, sk, D, ms_orig)
    tf_simp = tflops(B, H, sq, sk, D, ms_simp)

    print(f"  original   : {ms_orig:.3f} ms  |  {tf_orig:.2f} TFLOPS")
    print(f"  simplified : {ms_simp:.3f} ms  |  {tf_simp:.2f} TFLOPS")
    print(f"  ratio (simp/orig): {ms_simp/ms_orig:.3f}x  (1.0 = same speed)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--causal", action="store_true")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # configs: (B, H, seqlen_q, seqlen_k, headdim)
    configs = [
        (1, 16, 1024,  1024,  64),
        (1, 16, 2048,  2048,  64),
        (1, 16, 4096,  4096,  64),
        (1, 16, 8192,  8192,  64),
        (1, 16, 1024,  1024,  128),
        (1, 16, 2048,  2048,  128),
        (1, 16, 4096,  4096,  128),
        (1, 16, 8192,  8192,  128),
        (2,  8, 4096,  4096,  128),
    ]

    for (B, H, sq, sk, D) in configs:
        run(B, H, sq, sk, D, dtype, args.causal)

    print("\nDone.")


if __name__ == "__main__":
    main()
