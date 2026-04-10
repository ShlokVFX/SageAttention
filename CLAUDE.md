# SageAttention3 — Triton FP4 Implementation Guide

This file is the single source of truth for any new Claude session working on this repo.
Read it first, skip reading all the source files, save tokens.

---

## 1. What This Repo Is

A **pure-Python/Triton implementation of SageAttention3** targeting NVIDIA Blackwell (SM120,
RTX 5090) without CUTLASS or CuTe. The goal is to match the original `sageattention3_blackwell`
FP4 CUDA/CUTLASS implementation in both quality and throughput.

### Directory layout

```
triton_sage_attn3/       ← our Triton implementation (the main deliverable)
  api.py                 ← entry point: sageattn3_triton(q,k,v, quant='fp4')
  attention.py           ← Triton kernels: _fwd_kernel_bf16, _fwd_kernel_fp4
  quantize.py            ← MXFP4 quant kernels: _quant_mxfp4_kernel, _fused_smooth_fp4_kernel
  preprocessing.py       ← smooth_quant_q, normalize_k, compute_delta_s, preprocess_qkv

sageattention3_blackwell/ ← original CUDA/CUTLASS reference (comparison target)
  sageattn3/api.py        ← sageattn3_blackwell(q,k,v)  ← what we must match
  fp4attn_cuda.so         ← built .so (gitignored); rebuild with:
                              cd sageattention3_blackwell && python setup.py build_ext --inplace
  csrc/cutlass/include/   ← CUTLASS headers needed for build (gitignored)
                              Download: github.com/NVIDIA/cutlass/releases/tag/v3.9.2

example/
  sage3_video_compare.py  ← end-to-end video benchmark (Wan2.1-T2V-1.3B)
  modify_model/           ← model patching utilities for Wan
  videos/compare/         ← generated .mp4 outputs (gitignored)

RESULTS.md               ← all benchmark numbers, keep this up to date
```

---

## 2. Current State (as of this session)

### What works

| Feature | Status |
|---|---|
| `quant='none'` — BF16 flash-attention | ✅ Working, cosine ~0.9999 vs FP32 |
| `quant='fp8'` — FP8 E4M3 (H100+) | ✅ Working (not benchmarked here) |
| `quant='fp4'` — MXFP4 E2M1 (SM120) | ✅ Working, cosine ~0.986 vs FP32 |
| GQA (H_q ≠ H_k) | ✅ Fixed this session |
| Causal masking | ✅ |
| Video benchmark vs sage3 | ✅ Done: 20.85 dB PSNR, 95.4s vs 79.7s |

### Key numbers (RTX 5090, D=128)

**Kernel TFLOPS** (attention kernel only, not preprocessing):

| L | Triton FP4 | SDPA | sage3 CUDA (RTX 5060) |
|---|---|---|---|
| 2048 | 122 | 158 | 68 |
| 4096 | 172 | 162 | 86 |
| 8192 | **223** | 187 | 105 |

**End-to-end video generation (Wan, 20 steps, 81 frames):**

| Backend | Time | PSNR vs sage3 |
|---|---|---|
| sage3 CUDA | 79.7s | reference |
| triton_fp4 | 95.4s | 20.85 dB |

**Gap: ~20% slower end-to-end.** Root cause: preprocessing overhead at short sequences.

---

## 3. How the FP4 Pipeline Works

```
Input: q [B,H,L,D] BF16,  k [B,H_k,L,D] BF16,  v [B,H_k,L,D] BF16

Step 1 — Pad L to multiple of 128
Step 2 — normalize_k(k)          → k_norm [B,H_k,L,D]  (subtract global mean)
Step 3 — smooth_quant_and_fp4(q) → q_packed [B,H,L,D//2] uint8   (fused: center+quantize)
                                    q_scales  [B,H,L,D//32] uint8
                                    qm        [B,H,G,D]     BF16   (group means)
Step 4 — compute_delta_s(qm, k)  → delta_s [B,H,G,L]  FP32  (correction term)
Step 5 — quant_mxfp4(k_norm)     → k_packed [B,H_k,L,D//2] uint8
                                    k_scales  [B,H_k,L,D//32] uint8
Step 6 — k_packed.transpose(-2,-1) → k_T_packed [B,H_k,D//2,L]  (for coalesced kernel loads)
Step 7 — _fwd_kernel_fp4(...)    → out [B,H,L,D] BF16
```

### FP4 format (MXFP4 E2M1)
- 4 bits: `[sign | exp[1] | exp[0] | mant]`
- Values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}
- Packed 2 per byte: low nibble = even index, high nibble = odd index
- Scale: E8M0 uint8 per group-of-32, value = 2^(stored−127)
- Scale formula: `exp = ceil(log2(max_abs / 6.0))` — maps max_abs → ≤6 (FP4 max)

### Triton instruction
```python
tl.dot_scaled(q_packed, q_scales, 'e2m1',
              k_T_packed, k_scales, 'e2m1',
              out_dtype=tl.float32)
# Lowers to: mma.sync.aligned.m16n8k64.kind::mxf4nvf4.block_scale.ue8m0
```

### Tuned kernel config (RTX 5090)
```python
BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=2
# BLOCK_M=128, BLOCK_N=128 exceeds SRAM (>101 KB limit on SM120)
```

---

## 4. The Performance Gap — Root Cause Analysis

**Profile at L=1024, D=128, B=1, H=16:**

| Step | Time |
|---|---|
| normalize_k | 0.018 ms |
| smooth_quant_and_fp4 (fused) | 0.026 ms |
| quant_mxfp4 (K) | 0.018 ms |
| transpose K | 0.008 ms |
| compute_delta_s | 0.023 ms |
| **attention kernel** | **0.049 ms** |
| **Total** | **0.144 ms** |

At L=1024, the attention kernel is only **34% of total time**. Five separate kernel launches
each add ~5–20 µs of launch overhead. The Wan model has many attention layers with L < 2048
where this overhead dominates.

**Profile at L=8192** — overhead shrinks to 18%; kernel is 82% of total. We're already
faster than SDPA here (1.19×).

---

## 5. Future Plan: Closing the Gap with sage3 CUDA

Priority order (highest ROI first):

### 5.1 Fuse preprocessing into 1–2 kernels  ← **most impactful**

Replace the 5 separate Python-dispatch kernels with a single fused kernel covering:
`normalize_k + smooth_quant_q + compute_delta_s + quant_k + quant_q`

**Design:**
- Grid: `(B * H * G,)` where G = L // 128
- One program per (batch, head, token-group):
  1. Load K tile [GROUP_SIZE, D], compute global K mean, subtract, quantize K → FP4
  2. Load Q tile [GROUP_SIZE, D], compute group mean (qm), subtract, quantize Q → FP4
  3. Compute `delta_s_partial = qm @ K_tile^T`
- Challenge: `compute_delta_s` needs ALL K tiles for each Q group. Options:
  - Two-pass: separate K normalization + quantization pass, then Q pass
  - Or: for short sequences, the full `[G, L]` delta_s fits in a single matmul

**Simpler near-term fix:**  fuse just `normalize_k + quant_k` into one kernel
(saves 1 launch, ~20 µs at L=1024):
```python
# Current: 2 passes over K
k_norm = normalize_k(k)            # pass 1
k_packed, k_scales = quant_mxfp4(k_norm)  # pass 2
# Target: 1 pass
k_packed, k_scales, k_mean = normalize_and_quant_k(k)  # 1 kernel
```

### 5.2 Kernel-level pipelining (Triton async)

The `_fwd_kernel_fp4` uses `num_stages=2` which enables some async prefetch.
Explore `num_stages=3` and warp-group pipelining (Triton 3.x `warp_specialize`):
```python
# TMA-style persistent kernel: producer warps prefetch K/V, consumer warps compute
# Expected: ~15-20% kernel speedup for L >= 4096
```

### 5.3 V quantization (FP4 or FP8)

Currently V stays in BF16. Quantizing V to FP4/FP8 would:
- Halve V memory bandwidth → ~15% speedup on memory-bound sizes
- Slight quality reduction (V FP4 less critical than QK precision)

```python
# In _fwd_kernel_fp4:
v_packed, v_scales = quant_mxfp4(v)   # [B, H_k, N_k, D//2]
# Then in inner loop:
v_tile = tl.dot_scaled(p, p_scales, 'e2m1', v_packed, v_scales, 'e2m1')
```
Note: tl.dot_scaled for P@V requires P to be quantized (currently FP32 after softmax).
Need to requantize P → FP4 after softmax. This adds complexity but cuts V bandwidth.

### 5.4 Block sizes for specific Wan layer shapes

Profile the actual Q/K shapes in Wan's attention layers:
```bash
python -c "
import sys; sys.path.insert(0, '.')
from triton_sage_attn3.api import sageattn3_triton
import torch

orig = sageattn3_triton
shapes = []
def patched(q, k, v, **kw):
    shapes.append((q.shape, k.shape))
    return orig(q, k, v, **kw)
# ... patch and run one step
"
```
Then tune `BLOCK_M/BLOCK_N` per-shape with `@triton.autotune`.

### 5.5 Autotune decorator

Replace the fixed `BLOCK_M=128, BLOCK_N=64` with `@triton.autotune` over a small
config grid. Triton caches the best config per input shape, so first-call overhead
amortizes quickly in video generation (same shapes repeat every step).

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=8, num_stages=2),
    ],
    key=['N_q', 'N_k', 'HEAD_DIM'],
)
@triton.jit
def _fwd_kernel_fp4(...):
```

### 5.6 Expected impact

| Optimization | Estimated gain |
|---|---|
| Fuse normalize_k + quant_k | −0.025ms/layer = ~5% e2e |
| Fuse all 5 preproc steps | −0.08ms/layer = ~15% e2e |
| Kernel pipelining (num_stages=3+) | ~10% kernel speed |
| Autotune per shape | ~5–10% across mixed-L workloads |
| V quantization (FP8) | ~8% for large L |
| **Combined** | **~30–35% e2e** → match sage3 CUDA |

---

## 6. Quick Start / Reproducing Results

### Run the video benchmark
```bash
cd /workspace/SageAttention/example

# FP4 Triton vs sage3 CUDA (default)
python sage3_video_compare.py \
  --backends sage3 triton_fp4 \
  --ref sage3 \
  --num-prompts 1 --num-inference-steps 20 --num-frames 81 \
  --metrics

# FP4 Triton vs SDPA
python sage3_video_compare.py \
  --backends sdpa triton_fp4 \
  --ref sdpa \
  --num-prompts 1 --metrics
```

### Run kernel accuracy + throughput benchmark
```bash
cd /workspace/SageAttention
python - << 'EOF'
import torch, sys
sys.path.insert(0, '.')
from triton_sage_attn3 import sageattn3_triton
import torch.nn.functional as F

for L in [1024, 4096, 8192]:
    B,H,D = 1,16,128
    q=torch.randn(B,H,L,D,dtype=torch.bfloat16,device='cuda')
    k=torch.randn(B,H,L,D,dtype=torch.bfloat16,device='cuda')
    v=torch.randn(B,H,L,D,dtype=torch.bfloat16,device='cuda')
    ref = F.scaled_dot_product_attention(q.float(),k.float(),v.float()).bfloat16()
    out = sageattn3_triton(q.clone(),k.clone(),v.clone(),quant='fp4')
    a=out.float().flatten();b=ref.float().flatten()
    cos=(a@b/(a.norm()*b.norm())).item()
    print(f'L={L}  cosine={cos:.6f}')
EOF
```

### Rebuild sage3 CUDA extension (if .so missing)
```bash
# First get CUTLASS headers
python -c "
import urllib.request, tarfile
urllib.request.urlretrieve('https://github.com/NVIDIA/cutlass/archive/refs/tags/v3.9.2.tar.gz', '/tmp/cutlass.tar.gz')
with tarfile.open('/tmp/cutlass.tar.gz', 'r:gz') as t:
    members = [m for m in t.getmembers() if '/include/' in m.name]
    t.extractall('/tmp/cutlass_src', members=members, filter='data')
import shutil
shutil.copytree('/tmp/cutlass_src/cutlass-3.9.2/include/cutlass',
    'sageattention3_blackwell/csrc/cutlass/include/cutlass')
shutil.copytree('/tmp/cutlass_src/cutlass-3.9.2/include/cute',
    'sageattention3_blackwell/csrc/cutlass/include/cute')
"
cd sageattention3_blackwell && python setup.py build_ext --inplace
```

---

## 7. Key Files to Edit for Each Task

| Task | Files |
|---|---|
| Change FP4 quantization formula | `triton_sage_attn3/quantize.py` — `_quant_mxfp4_kernel` (~line 327) |
| Change FP4 attention kernel | `triton_sage_attn3/attention.py` — `_fwd_kernel_fp4` (~line 420) |
| Change block sizes / autotune | `triton_sage_attn3/attention.py` — `sage_attn3_fwd_fp4()` (~line 586) |
| Fuse preprocessing | `triton_sage_attn3/preprocessing.py` + `quantize.py` |
| Change API / routing | `triton_sage_attn3/api.py` — `sageattn3_triton()` (~line 73) |
| Add V quantization | `triton_sage_attn3/attention.py` — inner V-load section (~line 565) |
| Video benchmark | `example/sage3_video_compare.py` |

---

## 8. Known Pitfalls (save time, don't repeat these)

1. **`tl.clamp` only accepts float dtypes** — use `tl.clamp(x.to(tl.float32), -127.0, 127.0)`, not `tl.clamp(int_tensor, ...)`.

2. **`tl.constexpr` can't be declared inside a loop body** — declare all `constexpr` values at the top of the kernel function. Inside `tl.static_range`, variables computed from the loop index ARE constexpr, but literals must be hoisted.

3. **No nested `def` inside `@triton.jit`** — Triton can't parse nested function definitions. Inline everything.

4. **No 2D tensor indexing with mixed constexpr/vector** — `q_c[t_local, offs_e]` fails when `t_local` is a constexpr loop var and `offs_e` is a runtime vector. Use pointer arithmetic instead:
   ```python
   # BAD:  q_c[t_local, offs_e]
   # GOOD: tl.load(ptr + t_local * stride_row + offs_e * stride_col)
   ```

5. **Write-then-reload memory hazard** — if you `tl.store(x)` then immediately `tl.load(x)` in the same kernel program, the load may not see the store. Keep computed values in registers and pass them directly.

6. **GQA: `compute_delta_s` needs head expansion** — qm has H query heads, k has H_k KV heads. The matmul `qm @ k^T` needs `k.repeat_interleave(H//H_k, dim=1)` first.

7. **K pre-transpose layout** — K is stored as `[B, H_k, D//2, N_k]` (transposed from token-major). The scale stays `[B, H_k, N_k, D//32]` (NOT transposed). `tl.dot_scaled` doc says "do NOT transpose rhs_scale".

8. **SM120 SRAM limit is ~101 KB** — `BLOCK_M=128, BLOCK_N=128` with FP4 exceeds this. Max viable: `BLOCK_M=128, BLOCK_N=64`.

9. **`@triton.jit` functions must be defined in a .py file**, not in `python -c` inline code (raises `ValueError: @jit functions should be defined in a Python file`).

10. **Double-centering bug** — `preprocess_qkv` calls `smooth_quant_q` internally. For `quant='fp4'`, bypass it and call `normalize_k` + `smooth_quant_and_fp4` directly. The api.py `fp4` branch does this correctly; don't accidentally re-add `preprocess_qkv` for the fp4 path.
