"""
Benchmark + accuracy comparison:
  1. sageattn3           – original FP4 CUTLASS kernel (Blackwell SM120)
  2. sageattn3_simplified – identical algorithm, cleaned-up docs
  3. triton_sage_attn3   – our pure-Triton, CUTLASS-free reimplementation

HOW IT WORKS
------------
  • Same random Q/K/V fed to all three implementations.
  • Accuracy: max/mean absolute difference vs. a float32 SDPA reference.
  • Performance: CUDA-event timing (warmup + timed iterations) → TFLOPS.

TFLOPS FORMULA
--------------
  Standard attention FLOPs = 4 * B * H * seqlen_q * seqlen_k * D
  (two matmuls: QK^T and PV, each = 2*B*H*sq*sk*D ops)

USAGE
-----
  # non-causal, BF16
  python bench_compare.py

  # causal, FP16
  python bench_compare.py --causal --dtype fp16

  # skip FP4 CUDA kernels (e.g. not on SM120)
  python bench_compare.py --skip-cuda
"""

import sys, os, argparse
import torch
import torch.nn.functional as F

# ── Import paths ─────────────────────────────────────────────────────────────
# triton_sage_attn3 lives one level up
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from triton_sage_attn3.api import sageattn3_triton as triton_attn


def _try_import_cuda():
    """Import the FP4 CUDA kernels; return None if unavailable."""
    try:
        from sageattn3 import sageattn3_blackwell as orig
        from sageattn3_simplified.api import sageattn3_blackwell as simp
        return orig, simp
    except Exception as e:
        print(f"  [warn] CUDA FP4 kernels not importable: {e}")
        return None, None


# ── Helpers ───────────────────────────────────────────────────────────────────

def tflops(B, H, sq, sk, D, ms):
    """TFLOPS from attention shape and elapsed ms."""
    flops = 4.0 * B * H * sq * sk * D
    return flops / (ms * 1e-3) / 1e12


def sdpa_ref(q, k, v, is_causal):
    """Float32 SDPA reference (ground truth)."""
    return F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=is_causal
    ).to(q.dtype)


def bench_one(fn, q, k, v, is_causal, warmup=10, iters=50):
    """Time fn over iters runs (after warmup). Returns (output, ms_avg)."""
    for _ in range(warmup):
        fn(q.clone(), k.clone(), v.clone(), is_causal=is_causal)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(q.clone(), k.clone(), v.clone(), is_causal=is_causal)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def accuracy_row(label, out, ref):
    diff = (out.float() - ref.float()).abs()
    a = out.float().flatten(); b = ref.float().flatten()
    cos = (a @ b / (a.norm() * b.norm())).item()
    return (f"  {label:<22}  max={diff.max().item():.5f}"
            f"  mean={diff.mean().item():.7f}  cos={cos:.6f}")


# ── Per-config run ────────────────────────────────────────────────────────────

def run(B, H, sq, sk, D, dtype, is_causal,
        orig_fn, simp_fn, skip_cuda):
    print(f"\n{'='*68}")
    print(f"  B={B}  H={H}  sq={sq}  sk={sk}  D={D}"
          f"  dtype={'bf16' if dtype==torch.bfloat16 else 'fp16'}"
          f"  causal={is_causal}")
    print(f"{'='*68}")

    torch.manual_seed(42)
    q = torch.randn(B, H, sq, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, sk, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, sk, D, dtype=dtype, device="cuda")

    # ── Accuracy ──────────────────────────────────────────────────────────────
    print("  ACCURACY  (vs float32 SDPA reference)")
    print(f"  {'':22}  {'max abs diff':>14}  {'mean abs diff':>14}  {'cosine sim':>12}")

    with torch.no_grad():
        ref = sdpa_ref(q, k, v, is_causal)

        if not skip_cuda and orig_fn is not None:
            try:
                out_orig = orig_fn(q.clone(), k.clone(), v.clone(), is_causal=is_causal)
                print(accuracy_row("original (FP4)", out_orig, ref))
            except Exception as e:
                print(f"  original (FP4)         ERROR: {e}")

        if not skip_cuda and simp_fn is not None:
            try:
                out_simp = simp_fn(q.clone(), k.clone(), v.clone(), is_causal=is_causal)
                print(accuracy_row("simplified (FP4)", out_simp, ref))
            except Exception as e:
                print(f"  simplified (FP4)       ERROR: {e}")

        out_tri = triton_attn(q.clone(), k.clone(), v.clone(), is_causal=is_causal)
        print(accuracy_row("triton (BF16)", out_tri, ref))

    # ── Performance ───────────────────────────────────────────────────────────
    print()
    print(f"  PERFORMANCE  (warmup=10, iters=50)")
    print(f"  {'label':<22}  {'ms':>8}  {'TFLOPS':>10}")

    with torch.no_grad():
        if not skip_cuda and orig_fn is not None:
            try:
                ms_orig = bench_one(orig_fn, q, k, v, is_causal)
                tf_orig = tflops(B, H, sq, sk, D, ms_orig)
                print(f"  {'original (FP4)':<22}  {ms_orig:>8.3f}  {tf_orig:>10.2f}")
            except Exception as e:
                print(f"  {'original (FP4)':<22}  ERROR: {e}")

        if not skip_cuda and simp_fn is not None:
            try:
                ms_simp = bench_one(simp_fn, q, k, v, is_causal)
                tf_simp = tflops(B, H, sq, sk, D, ms_simp)
                print(f"  {'simplified (FP4)':<22}  {ms_simp:>8.3f}  {tf_simp:>10.2f}")
            except Exception as e:
                print(f"  {'simplified (FP4)':<22}  ERROR: {e}")

        ms_tri = bench_one(triton_attn, q, k, v, is_causal)
        tf_tri = tflops(B, H, sq, sk, D, ms_tri)
        print(f"  {'triton (BF16)':<22}  {ms_tri:>8.3f}  {tf_tri:>10.2f}")

        # SDPA baseline (PyTorch built-in)
        ms_sdpa = bench_one(
            lambda q, k, v, is_causal:
                F.scaled_dot_product_attention(q, k, v, is_causal=is_causal),
            q, k, v, is_causal
        )
        tf_sdpa = tflops(B, H, sq, sk, D, ms_sdpa)
        print(f"  {'sdpa (PyTorch)':<22}  {ms_sdpa:>8.3f}  {tf_sdpa:>10.2f}")

        # Ratios relative to SDPA
        print()
        if not skip_cuda and orig_fn is not None:
            try:
                print(f"  triton / sdpa speedup:   {ms_sdpa / ms_tri:.2f}x")
            except Exception:
                pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype",      choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--causal",     action="store_true")
    p.add_argument("--skip-cuda",  action="store_true",
                   help="Skip FP4 CUDA kernels (use on non-SM120 GPUs)")
    args = p.parse_args()

    dtype     = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    skip_cuda = args.skip_cuda

    orig_fn, simp_fn = (None, None) if skip_cuda else _try_import_cuda()
    if orig_fn is None and not skip_cuda:
        print("  [info] FP4 CUDA kernels unavailable – benchmarking Triton only.")
        skip_cuda = True

    # ── Config table ──────────────────────────────────────────────────────────
    # (B, H, seqlen_q, seqlen_k, headdim)
    configs = [
        (1, 16,  1024,  1024,  64),
        (1, 16,  2048,  2048,  64),
        (1, 16,  4096,  4096,  64),
        (1, 16,  8192,  8192,  64),
        (1, 16,  1024,  1024, 128),
        (1, 16,  2048,  2048, 128),
        (1, 16,  4096,  4096, 128),
        (1, 16,  8192,  8192, 128),
        (2,  8,  4096,  4096, 128),
    ]

    for (B, H, sq, sk, D) in configs:
        run(B, H, sq, sk, D, dtype, args.causal,
            orig_fn, simp_fn, skip_cuda)

    print("\nDone.")


if __name__ == "__main__":
    main()
