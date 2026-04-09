from typing import Tuple
import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# ARCH DETECTION
# ---------------------------------------------------------------------------


def get_device_arch():
    if torch.version.hip is not None:
        return "amd"  # MI300X gfx942
    elif torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 9:
            return "sm90"   # H100
        elif major >= 8:
            return "sm80"   # A100
    return "unknown"

def quantise_qkv_fp4(q, k, v):
    """
    Stub for MXFP4 (Blackwell only).

    On unsupported architectures (e.g., MI300X), this gracefully falls back.
    """
    arch = get_device_arch()

    if arch != "sm120":
        # safe fallback
        return quantise_qkv_int8(q, k, v)

    raise RuntimeError("FP4 path not implemented in this build")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

INT8_MAX = 127.0
FP8_MAX  = 448.0


# ---------------------------------------------------------------------------
# INT8 KERNEL (FAST PATH FOR AMD)
# ---------------------------------------------------------------------------

@triton.jit
def _per_token_quant_int8_kernel(
    x_ptr, xq_ptr, scale_ptr,
    stride_xn, stride_xd,
    stride_sqn,
    N_tokens,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(
        x_ptr + n * stride_xn + offs * stride_xd,
        mask=mask, other=0.0
    ).to(tl.float32)

    abs_x = tl.abs(x)
    abs_max = tl.max(abs_x)

    scale = tl.where(abs_max == 0.0, 1.0, abs_max / INT8_MAX)

    xq = tl.clamp(
        tl.floor(x / scale + 0.5),
        -INT8_MAX, INT8_MAX
    ).to(tl.int8)

    tl.store(xq_ptr + n * stride_xn + offs * stride_xd, xq, mask=mask)
    tl.store(scale_ptr + n * stride_sqn, scale)


def quant_int8_per_token(x: torch.Tensor):
    orig_shape = x.shape
    D = x.shape[-1]

    x_flat = x.reshape(-1, D).contiguous()
    N = x_flat.shape[0]

    BLOCK_D = max(128, triton.next_power_of_2(D))

    x_int8 = torch.empty_like(x_flat, dtype=torch.int8)
    scales = torch.empty(N, dtype=torch.float32, device=x.device)

    _per_token_quant_int8_kernel[(N,)](
        x_flat, x_int8, scales,
        x_flat.stride(0), x_flat.stride(1),
        scales.stride(0),
        N_tokens=N,
        D=D,
        BLOCK_D=BLOCK_D,
    )

    return x_int8.reshape(orig_shape), scales.reshape(orig_shape[:-1])


def dequant_int8(x_int8, scales):
    return x_int8.float() * scales.unsqueeze(-1)


# ---------------------------------------------------------------------------
# FP8 KERNEL (ONLY FOR SM90+)
# ---------------------------------------------------------------------------

if torch.version.hip is None:  # disable on AMD

    @triton.jit
    def _per_token_quant_fp8_kernel(
        x_ptr, xq_ptr, scale_ptr,
        stride_xn, stride_xd,
        stride_sqn,
        N_tokens,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        n = tl.program_id(0)

        offs = tl.arange(0, BLOCK_D)
        mask = offs < D

        x = tl.load(
            x_ptr + n * stride_xn + offs * stride_xd,
            mask=mask, other=0.0
        ).to(tl.float32)

        abs_max = tl.max(tl.abs(x))
        scale = tl.where(abs_max == 0.0, 1.0, abs_max / FP8_MAX)

        xq = tl.clamp(x / scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)

        tl.store(xq_ptr + n * stride_xn + offs * stride_xd, xq, mask=mask)
        tl.store(scale_ptr + n * stride_sqn, scale)


def quant_fp8_per_token(x: torch.Tensor):
    arch = get_device_arch()

    if arch != "sm90":
        raise RuntimeError(f"FP8 not supported on {arch}")

    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("PyTorch FP8 not available")

    orig_shape = x.shape
    D = x.shape[-1]

    x_flat = x.reshape(-1, D).contiguous()
    N = x_flat.shape[0]

    BLOCK_D = max(128, triton.next_power_of_2(D))

    x_fp8 = torch.empty_like(x_flat, dtype=torch.float8_e4m3fn)
    scales = torch.empty(N, dtype=torch.float32, device=x.device)

    _per_token_quant_fp8_kernel[(N,)](
        x_flat, x_fp8, scales,
        x_flat.stride(0), x_flat.stride(1),
        scales.stride(0),
        N_tokens=N,
        D=D,
        BLOCK_D=BLOCK_D,
    )

    return x_fp8.reshape(orig_shape), scales.reshape(orig_shape[:-1])


# ---------------------------------------------------------------------------
# AUTO DISPATCH (KEY PART)
# ---------------------------------------------------------------------------

def quantise_qkv_auto(q, k, v, mode="auto"):
    arch = get_device_arch()

    if mode == "int8" or (mode == "auto" and arch == "amd"):
        q_i, qs = quant_int8_per_token(q)
        k_i, ks = quant_int8_per_token(k)
        return q_i, k_i, v, qs, ks

    elif mode == "fp8" or (mode == "auto" and arch == "sm90"):
        q_f, qs = quant_fp8_per_token(q)
        k_f, ks = quant_fp8_per_token(k)
        return q_f, k_f, v, qs, ks

    else:
        # fallback = no quant
        return q, k, v, None, None


# ---------------------------------------------------------------------------
# OPTIONAL: FORCE MODES
# ---------------------------------------------------------------------------

def quantise_qkv_int8(q, k, v):
    q_i, qs = quant_int8_per_token(q)
    k_i, ks = quant_int8_per_token(k)
    return q_i, k_i, v, qs, ks


def quantise_qkv_fp8(q, k, v):
    return quant_fp8_per_token(q)[0], quant_fp8_per_token(k)[0], v