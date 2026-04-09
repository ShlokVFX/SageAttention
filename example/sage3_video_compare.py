"""
Video generation comparison: sageattn3 (original) vs triton_sage_attn3 vs sdpa baseline.

WHAT THIS DOES
--------------
Generates the same video with three attention backends using Wan2.1-T2V-1.3B:
  1. sdpa        – PyTorch scaled_dot_product_attention (ground truth)
  2. triton      – our pure-Triton SageAttention3 (BF16, portable)
  3. sage3       – original FP4 CUDA SageAttention3 (SM120 Blackwell only)

Each run uses an identical fixed seed so outputs are directly comparable.
Videos are written to:
  videos/compare/<prompt_id>_sdpa.mp4
  videos/compare/<prompt_id>_triton.mp4
  videos/compare/<prompt_id>_sage3.mp4

USAGE
-----
  # generate 1 prompt with sdpa + triton (safe on any GPU, no FP4 needed)
  python sage3_video_compare.py --backends sdpa triton --num-prompts 1

  # generate 2 prompts with all three backends (requires SM120 for sage3)
  python sage3_video_compare.py --backends sdpa triton sage3 --num-prompts 2

  # use a custom prompt string
  python sage3_video_compare.py --backends sdpa triton --prompt "a cat surfing a wave"

  # skip model download check (use cached)
  python sage3_video_compare.py --backends triton --num-prompts 1 --no-check

QUALITY METRICS (printed after generation)
-------------------------------------------
  • Frame-level PSNR between triton and sdpa outputs (dB; higher = closer)
  • Frame-level SSIM (0–1; higher = closer)
  • Per-frame max pixel difference

Requires: diffusers >= 0.30, transformers, torch >= 2.1, imageio, scikit-image
"""

import os, sys, gc, time, argparse
import torch
import torch.nn.functional as F
from diffusers import WanPipeline
from diffusers.utils import export_to_video
import os
os.environ["TRITON_ALLOW_NON_CONSTEXPR_GLOBALS"] = "1"
# ── Path setup ────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_SA_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _SA_ROOT)   # for triton_sage_attn3

from modify_model.modify_wan import set_sage_attn_wan, SageWanAttnProcessor
from triton_sage_attn3.api import sageattn3_triton


# ── Attention backend registry ────────────────────────────────────────────────

def _make_triton_fn():
    """Wrap sageattn3_triton to match the (q,k,v,**kw) signature Wan expects."""
    def _triton(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **_):
        return sageattn3_triton(q, k, v, is_causal=is_causal)
    return _triton


def _make_sage3_fn():
    """Import and wrap the original FP4 sageattn3 kernel."""
    try:
        from sageattn3 import sageattn3_blackwell
        def _sage3(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **_):
            return sageattn3_blackwell(q, k, v, is_causal=is_causal)
        return _sage3
    except ImportError as e:
        raise ImportError(
            f"Could not import sageattn3 (FP4 CUDA kernel): {e}\n"
            "Make sure you installed sageattn3 from "
            "SageAttention/sageattention3_blackwell/ and are on SM120."
        )


def _make_sdpa_fn():
    def _sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **_):
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
    return _sdpa


BACKEND_FACTORIES = {
    "sdpa":   _make_sdpa_fn,
    "triton": _make_triton_fn,
    "sage3":  _make_sage3_fn,
}


# ── Metric helpers ────────────────────────────────────────────────────────────

def _load_video_frames(path: str):
    """Load MP4 → list of (H,W,3) uint8 numpy arrays."""
    try:
        import imageio
        reader = imageio.get_reader(path, "ffmpeg")
        frames = [f for f in reader]
        reader.close()
        return frames
    except ImportError:
        print("  [warn] imageio not installed – skipping pixel metrics")
        return None


def _psnr(a, b):
    """PSNR between two uint8 numpy arrays."""
    import numpy as np
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def _ssim(a, b):
    """Structural similarity between two uint8 numpy arrays (per-frame)."""
    try:
        from skimage.metrics import structural_similarity as ski_ssim
        import numpy as np
        # skimage expects (H, W, C) or (H, W); channel_axis for colour
        return ski_ssim(a, b, channel_axis=2, data_range=255)
    except ImportError:
        return None


def compute_video_metrics(ref_path: str, cmp_path: str, label: str):
    """Print PSNR / SSIM between two video files."""
    import numpy as np
    ref_frames = _load_video_frames(ref_path)
    cmp_frames = _load_video_frames(cmp_path)

    if ref_frames is None or cmp_frames is None:
        return

    n = min(len(ref_frames), len(cmp_frames))
    psnrs, ssims, maxdiffs = [], [], []

    for i in range(n):
        r, c = np.array(ref_frames[i]), np.array(cmp_frames[i])
        psnrs.append(_psnr(r, c))
        s = _ssim(r, c)
        if s is not None:
            ssims.append(s)
        maxdiffs.append(np.abs(r.astype(float) - c.astype(float)).max())

    print(f"\n  Metrics  ref=sdpa  cmp={label}  ({n} frames)")
    print(f"    PSNR    avg={np.mean(psnrs):.2f} dB  min={np.min(psnrs):.2f} dB")
    if ssims:
        print(f"    SSIM    avg={np.mean(ssims):.4f}     min={np.min(ssims):.4f}")
    print(f"    MaxDiff avg={np.mean(maxdiffs):.1f}      max={np.max(maxdiffs):.1f}")


# ── Generation ────────────────────────────────────────────────────────────────

def generate_video(
    pipe,
    attn_fn,
    prompt: str,
    out_path: str,
    seed: int,
    height: int = 480,
    width:  int = 832,
    num_frames: int = 81,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 20,
):
    """Patch attention, run pipeline, export video. Returns elapsed seconds."""
    # Swap attention backend in-place
    set_sage_attn_wan(pipe.transformer, attn_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen    = torch.Generator(device=device).manual_seed(seed)

    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
        frames = pipe(
            prompt=prompt,
            negative_prompt=(
                "Bright tones, overexposed, static, blurred details, "
                "subtitles, style, works, paintings, images, static, "
                "overall gray, worst quality, low quality, JPEG compression "
                "residue, ugly, incomplete"
            ),
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=gen,
        ).frames[0]
    elapsed = time.perf_counter() - t0

    export_to_video(frames, out_path, fps=16)
    del frames
    gc.collect()
    torch.cuda.empty_cache()
    return elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backends", nargs="+",
        choices=["sdpa", "triton", "sage3"],
        default=["sdpa", "triton"],
        help="Which attention backends to run (default: sdpa triton)",
    )
    p.add_argument(
        "--num-prompts", type=int, default=1,
        help="Number of prompts to use from testing_prompts.txt (default: 1)",
    )
    p.add_argument(
        "--prompt", type=str, default=None,
        help="Single custom prompt string (overrides --num-prompts)",
    )
    p.add_argument(
        "--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="HuggingFace model ID or local path",
    )
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--num-inference-steps", type=int,  default=20)
    p.add_argument("--height",             type=int,   default=480)
    p.add_argument("--width",              type=int,   default=832)
    p.add_argument("--num-frames",         type=int,   default=81)
    p.add_argument("--metrics",            action="store_true",
                   help="Compute PSNR/SSIM after generation (requires imageio, scikit-image)")
    args = p.parse_args()

    # ── Prompts ───────────────────────────────────────────────────────────────
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompt_file = os.path.join(_HERE, "videos", "testing_prompts.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = [l.strip() for l in f if l.strip()]
        prompts = prompts[: args.num_prompts]

    out_dir = os.path.join(_HERE, "videos", "compare")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load backends ─────────────────────────────────────────────────────────
    ordered = sorted(args.backends, key=lambda x: 0 if x == "triton" else 1)

    backend_fns = {}
    for name in ordered:
        try:
            backend_fns[name] = BACKEND_FACTORIES[name]()
            print(f"[OK]   backend '{name}' loaded")
        except Exception as e:
            print(f"[SKIP] backend '{name}': {e}")

    if not backend_fns:
        print("No backends available. Exiting.")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    pipe = WanPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    print("Model loaded.\n")

    # ── Generate ──────────────────────────────────────────────────────────────
    results = {}   # {(prompt_id, backend): path}

    for pid, prompt in enumerate(prompts):
        short = prompt[:60].replace(" ", "_").replace(",", "")
        print(f"\n{'─'*60}")
        print(f"Prompt {pid}: {prompt[:80]}{'...' if len(prompt)>80 else ''}")
        print(f"{'─'*60}")

        for name, fn in backend_fns.items():
            out_path = os.path.join(out_dir, f"{pid}_{name}.mp4")
            print(f"  [{name}] generating → {out_path}")
            try:
                elapsed = generate_video(
                    pipe, fn, prompt, out_path,
                    seed=args.seed,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                )
                results[(pid, name)] = out_path
                print(f"  [{name}] done in {elapsed:.1f}s  →  {out_path}")
            except Exception as e:
                print(f"  [{name}] FAILED: {e}")
                import traceback; traceback.print_exc()

    # ── Metrics ───────────────────────────────────────────────────────────────
    if args.metrics and "sdpa" in backend_fns:
        print(f"\n{'='*60}")
        print("PIXEL METRICS  (reference = sdpa)")
        print(f"{'='*60}")
        for pid in range(len(prompts)):
            ref_path = results.get((pid, "sdpa"))
            if ref_path is None:
                continue
            for name in args.backends:
                if name == "sdpa":
                    continue
                cmp_path = results.get((pid, name))
                if cmp_path:
                    compute_video_metrics(ref_path, cmp_path, f"{name}[prompt{pid}]")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("OUTPUT SUMMARY")
    print(f"{'='*60}")
    for (pid, name), path in sorted(results.items()):
        print(f"  prompt={pid}  backend={name:<8}  {path}")

    print(f"\nAll videos saved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
