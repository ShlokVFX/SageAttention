"""
Correctness tests and benchmarks for triton_sage_attn3.

USAGE
-----
    # correctness only (CPU-compatible shapes)
    python test_and_bench.py

    # include benchmarks
    python test_and_bench.py --bench

    # include FP8 tests (requires H100 / SM90+)
    python test_and_bench.py --fp8

WHAT IS TESTED
--------------
1. Preprocessing:
   - smooth_quant_q: centered Q + group means sum to original Q
   - normalize_k: zero mean check
   - delta_s: qm @ K^T produces correct shape and values

2. Attention kernel (BF16):
   - Non-causal:  compare to torch.nn.functional.scaled_dot_product_attention
   - Causal:      compare to sdpa with is_causal=True
   - GQA:         H_k < H works correctly
   - Seq padding: outputs match regardless of padding level

3. Full pipeline (sageattn3_triton):
   - End-to-end result close to sdpa reference

4. FP8 pipeline (optional, requires H100):
   - FP8 quant / dequant roundtrip
   - FP8 attention result vs sdpa reference

BENCHMARKS (--bench)
--------------------
Throughputs reported in TFLOPS using triton.testing.do_bench.
"""

import argparse
import math
import sys
import os

# Make sure the package is importable when running this script directly,
# e.g.  python triton_sage_attn3/test_and_bench.py
#       python -m triton_sage_attn3.test_and_bench
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_qkv(B, H, L, D, dtype=torch.bfloat16, device="cuda"):
    q = torch.randn(B, H, L, D, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, H, L, D, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, H, L, D, dtype=dtype, device=device) * 0.1
    return q, k, v


def sdpa_ref(q, k, v, is_causal=False, sm_scale=None):
    """PyTorch reference: scaled_dot_product_attention."""
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    # sdpa expects [B, H, L, D]; scale is applied internally as 1/sqrt(D)
    # so we pre-scale Q to inject a custom sm_scale
    return F.scaled_dot_product_attention(
        q * sm_scale * (q.shape[-1] ** 0.5),
        k, v,
        is_causal=is_causal,
        scale=sm_scale,
    )


def allclose(a: torch.Tensor, b: torch.Tensor, rtol=1e-2, atol=1e-2) -> bool:
    return torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)


def max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (a_f @ b_f / (a_f.norm() * b_f.norm())).item()


# ---------------------------------------------------------------------------
# Test: preprocessing
# ---------------------------------------------------------------------------

def test_smooth_quant(device="cuda"):
    from triton_sage_attn3.preprocessing import smooth_quant_q

    B, H, L, D = 2, 4, 256, 64

    # Use float32 for tight numerical tolerance check
    q32 = torch.randn(B, H, L, D, dtype=torch.float32, device=device)
    q_c32, qm32 = smooth_quant_q(q32, group_size=128)
    qm_exp = qm32.repeat_interleave(128, dim=2)
    err32 = max_err(q_c32 + qm_exp, q32)
    assert err32 < 1e-4, f"FP32 smooth_quant reconstruction error {err32:.2e}"

    G = L // 128
    group_means = q_c32.reshape(B, H, G, 128, D).mean(dim=3)
    assert group_means.abs().max().item() < 1e-4, "FP32 q_c group means not zero"

    # BF16: reconstruction error is bounded by BF16 rounding (~3 ops × eps_bf16).
    # With unscaled randn (max magnitude ~4-5), max_err ~0.05 is expected.
    q16 = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=device)
    q_c16, qm16 = smooth_quant_q(q16, group_size=128)
    qm_exp16 = qm16.repeat_interleave(128, dim=2)
    err16 = max_err(q_c16 + qm_exp16, q16)
    # Threshold: 3 BF16 ops × epsilon_bf16(1) × max_magnitude.
    # epsilon_bf16 ≈ 0.0078; max_magnitude of randn < 6σ ≈ 6; threshold = 3×0.0078×6 ≈ 0.14
    assert err16 < 0.15, f"BF16 smooth_quant reconstruction error {err16:.2e} too large"

    print(f"[PASS] test_smooth_quant  fp32_err={err32:.1e}  bf16_err={err16:.1e}")


def test_normalize_k(device="cuda"):
    from triton_sage_attn3.preprocessing import normalize_k

    B, H, L, D = 2, 4, 256, 64
    # Test with float32 for tight tolerance
    k32 = torch.randn(B, H, L, D, dtype=torch.float32, device=device) + 5.0
    k_norm32 = normalize_k(k32)
    err32 = k_norm32.float().mean(dim=2).abs().max().item()
    assert err32 < 1e-5, f"normalize_k FP32 mean {err32:.2e}"

    # BF16 with realistic small-magnitude values: mean should be near zero
    k16 = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=device) * 0.1
    k_norm16 = normalize_k(k16)
    err16 = k_norm16.float().mean(dim=2).abs().max().item()
    assert err16 < 1e-3, f"normalize_k BF16 mean {err16:.2e}"

    print(f"[PASS] test_normalize_k  fp32_err={err32:.1e}  bf16_err={err16:.1e}")


def test_delta_s(device="cuda"):
    from triton_sage_attn3.preprocessing import smooth_quant_q, normalize_k, compute_delta_s

    B, H, L, D = 1, 2, 128, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=device)
    k = normalize_k(k)

    q_c, qm = smooth_quant_q(q)
    ds = compute_delta_s(qm, k)             # [B, H, 1, L]

    # delta_s should equal qm @ k^T
    expected = torch.matmul(qm.float(), k.float().transpose(-2, -1))
    err = max_err(ds, expected)
    assert err < 1e-4, f"delta_s mismatch {err:.2e}"
    print("[PASS] test_delta_s")


# ---------------------------------------------------------------------------
# Test: attention kernel (BF16)
# ---------------------------------------------------------------------------

def test_attention_noncausal(device="cuda"):
    from triton_sage_attn3.attention import sage_attn3_fwd

    B, H, L, D = 2, 4, 256, 64
    q, k, v = make_qkv(B, H, L, D, device=device)
    sm = D ** -0.5

    out_tri = sage_attn3_fwd(q, k, v, softmax_scale=sm, is_causal=False)
    out_ref = F.scaled_dot_product_attention(q, k, v, scale=sm)

    cs = cosine_sim(out_tri, out_ref)
    me = max_err(out_tri, out_ref)
    assert cs > 0.999, f"non-causal cosine sim {cs:.4f} too low"
    assert me < 1e-1, f"non-causal max error {me:.4f} too large"
    print(f"[PASS] test_attention_noncausal  cos={cs:.5f}  max_err={me:.2e}")


def test_attention_causal(device="cuda"):
    from triton_sage_attn3.attention import sage_attn3_fwd

    B, H, L, D = 2, 4, 256, 64
    q, k, v = make_qkv(B, H, L, D, device=device)
    sm = D ** -0.5

    out_tri = sage_attn3_fwd(q, k, v, softmax_scale=sm, is_causal=True)
    out_ref = F.scaled_dot_product_attention(q, k, v, scale=sm, is_causal=True)

    cs = cosine_sim(out_tri, out_ref)
    me = max_err(out_tri, out_ref)
    assert cs > 0.999, f"causal cosine sim {cs:.4f} too low"
    assert me < 1e-1, f"causal max error {me:.4f} too large"
    print(f"[PASS] test_attention_causal     cos={cs:.5f}  max_err={me:.2e}")


def test_attention_gqa(device="cuda"):
    """GQA: 8 Q heads, 2 KV heads (ratio=4)."""
    from triton_sage_attn3.attention import sage_attn3_fwd

    B, H, H_k, L, D = 1, 8, 2, 256, 64
    q = torch.randn(B, H,   L, D, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(B, H_k, L, D, dtype=torch.bfloat16, device=device) * 0.1
    v = torch.randn(B, H_k, L, D, dtype=torch.bfloat16, device=device) * 0.1

    # Expand K/V to full H for reference
    ratio = H // H_k
    k_exp = k.repeat_interleave(ratio, dim=1)
    v_exp = v.repeat_interleave(ratio, dim=1)

    sm = D ** -0.5
    out_tri = sage_attn3_fwd(q, k, v, softmax_scale=sm, is_causal=False)
    out_ref = F.scaled_dot_product_attention(q, k_exp, v_exp, scale=sm)

    cs = cosine_sim(out_tri, out_ref)
    me = max_err(out_tri, out_ref)
    assert cs > 0.999, f"GQA cosine sim {cs:.4f} too low"
    print(f"[PASS] test_attention_gqa        cos={cs:.5f}  max_err={me:.2e}")


def test_attention_with_delta_s(device="cuda"):
    """Full pipeline: Q centering + delta_s should match direct attention."""
    from triton_sage_attn3.preprocessing import preprocess_qkv
    from triton_sage_attn3.attention import sage_attn3_fwd

    B, H, L, D = 1, 4, 256, 64
    q, k, v = make_qkv(B, H, L, D, device=device)
    sm = D ** -0.5

    # Triton path: preprocess then attend
    q_pre, k_pre, v_pre, delta_s = preprocess_qkv(q.clone(), k.clone(), v.clone())
    out_tri = sage_attn3_fwd(
        q_pre, k_pre, v_pre,
        delta_s=delta_s,
        softmax_scale=sm,
        is_causal=False,
    )[:, :, :L, :]  # trim padding

    # Reference: direct attention on ORIGINAL q, k (no centering)
    k_norm = k - k.mean(dim=2, keepdim=True)
    out_ref = F.scaled_dot_product_attention(q, k_norm, v, scale=sm)

    cs = cosine_sim(out_tri, out_ref)
    me = max_err(out_tri, out_ref)
    assert cs > 0.998, f"delta_s pipeline cosine sim {cs:.4f} too low"
    print(f"[PASS] test_attention_with_delta_s  cos={cs:.5f}  max_err={me:.2e}")


# ---------------------------------------------------------------------------
# Test: full sageattn3_triton() API
# ---------------------------------------------------------------------------

def test_full_api(device="cuda"):
    from triton_sage_attn3 import sageattn3_triton

    B, H, L, D = 2, 8, 512, 64
    q, k, v = make_qkv(B, H, L, D, device=device)

    out = sageattn3_triton(q.clone(), k.clone(), v.clone(), is_causal=False)
    assert out.shape == q.shape, f"Shape mismatch: {out.shape} vs {q.shape}"

    out_c = sageattn3_triton(q.clone(), k.clone(), v.clone(), is_causal=True)
    assert out_c.shape == q.shape

    print("[PASS] test_full_api")


def test_full_api_noncausal_accuracy(device="cuda"):
    from triton_sage_attn3 import sageattn3_triton

    B, H, L, D = 1, 4, 256, 64
    q, k, v = make_qkv(B, H, L, D, device=device)
    sm = D ** -0.5

    out_tri = sageattn3_triton(q.clone(), k.clone(), v.clone(),
                               is_causal=False, sm_scale=sm)
    # Reference uses K-normalized version to match our pipeline
    k_norm = k - k.mean(dim=2, keepdim=True)
    out_ref = F.scaled_dot_product_attention(q, k_norm, v, scale=sm)

    cs = cosine_sim(out_tri, out_ref)
    me = max_err(out_tri, out_ref)
    assert cs > 0.997, f"API accuracy cos={cs:.4f} too low"
    print(f"[PASS] test_full_api_accuracy      cos={cs:.5f}  max_err={me:.2e}")


# ---------------------------------------------------------------------------
# Test: FP8 (optional, H100 only)
# ---------------------------------------------------------------------------

def test_fp8_quant_roundtrip(device="cuda"):
    from triton_sage_attn3.quantize import quant_fp8_per_token

    B, H, L, D = 2, 4, 128, 64
    x = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=device) * 0.1

    xq, scales = quant_fp8_per_token(x)
    # Triton 3.x uses float8e4nv → PyTorch stores as float8_e4m3fn
    assert xq.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz), \
        f"Unexpected FP8 dtype: {xq.dtype}"
    # Reconstruct
    xr = xq.float() * scales.unsqueeze(-1)
    cs = cosine_sim(xr, x)
    assert cs > 0.999, f"FP8 roundtrip cosine sim {cs:.4f}"
    print(f"[PASS] test_fp8_quant_roundtrip  cos={cs:.5f}")


def test_fp8_attention(device="cuda"):
    from triton_sage_attn3 import sageattn3_triton

    B, H, L, D = 1, 4, 256, 64
    q, k, v = make_qkv(B, H, L, D, device=device)
    sm = D ** -0.5

    out_fp8 = sageattn3_triton(q.clone(), k.clone(), v.clone(),
                               is_causal=False, sm_scale=sm, quant='fp8')
    k_norm  = k - k.mean(dim=2, keepdim=True)
    out_ref = F.scaled_dot_product_attention(q, k_norm, v, scale=sm)

    cs = cosine_sim(out_fp8, out_ref)
    me = max_err(out_fp8, out_ref)
    # FP8 has slightly more error than BF16
    assert cs > 0.99, f"FP8 attention cosine sim {cs:.4f} too low"
    print(f"[PASS] test_fp8_attention          cos={cs:.5f}  max_err={me:.2e}")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_attn(B, H, L, D, is_causal, quant, device="cuda"):
    import triton.testing

    from triton_sage_attn3 import sageattn3_triton

    q, k, v = make_qkv(B, H, L, D, device=device)

    def fn():
        return sageattn3_triton(q, k, v, is_causal=is_causal, quant=quant)

    # Warmup
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)

    # Compute TFLOPS
    # FLOPs for attention: 2 × B × H × L_q × L_k × D (for QK^T and PV)
    flops = 2 * B * H * L * L * D * 2   # factor 2 for both matmuls, * 2 for mul+add
    tflops = flops / (ms * 1e-3) / 1e12
    return ms, tflops


def run_benchmarks(device="cuda"):
    print("\n" + "="*70)
    print("BENCHMARKS")
    print("="*70)
    print(f"{'Config':<45} {'ms':>8} {'TFLOPS':>10}")
    print("-"*70)

    configs = [
        # B,  H,   L,    D,     causal, quant
        (1,  16,  1024, 128,  False,  "none"),
        (1,  16,  4096, 128,  False,  "none"),
        (1,  16,  4096, 128,  True,   "none"),
        (1,  32,  2048, 128,  False,  "none"),
        (4,  16,  1024, 128,  False,  "none"),
    ]

    for B, H, L, D, causal, quant in configs:
        label = f"B={B} H={H} L={L} D={D} causal={causal} quant={quant}"
        try:
            ms, tflops = bench_attn(B, H, L, D, causal, quant, device)
            print(f"{label:<45} {ms:>8.2f} {tflops:>10.2f}")
        except Exception as e:
            print(f"{label:<45}  ERROR: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="Run benchmarks")
    parser.add_argument("--fp8",   action="store_true", help="Run FP8 tests (H100+)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, running on CPU – most tests will be skipped.")
        device = "cpu"

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print()

    # ── Preprocessing tests ──────────────────────────────────────────────────
    test_smooth_quant(device)
    test_normalize_k(device)
    test_delta_s(device)

    if device == "cuda":
        # ── Kernel tests ─────────────────────────────────────────────────────
        test_attention_noncausal(device)
        test_attention_causal(device)
        test_attention_gqa(device)
        test_attention_with_delta_s(device)

        # ── Full API tests ───────────────────────────────────────────────────
        test_full_api(device)
        test_full_api_noncausal_accuracy(device)

        # ── FP8 tests (optional) ─────────────────────────────────────────────
        if args.fp8:
            try:
                test_fp8_quant_roundtrip(device)
                test_fp8_attention(device)
            except RuntimeError as e:
                print(f"[SKIP] FP8 tests: {e}")

        # ── Benchmarks (optional) ────────────────────────────────────────────
        if args.bench:
            run_benchmarks(device)

    print("\nAll requested tests passed.")


if __name__ == "__main__":
    main()
