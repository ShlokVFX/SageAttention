"""
Setup script for sageattn3_simplified - documented version of sageattn3.

SAME build as original setup.py. Only difference: package name and source path.
See original setup.py for build flags explanation.

KEY BUILD FLAGS:
  -DQBLKSIZE=128   : query block size (kBlockM = 128)
  -DKBLKSIZE=128   : key/value block size (kBlockN = 128)
  -DCTA256         : CTA (thread block) = 256 threads
  -DDQINRMEM       : load Q into registers (not re-loaded each iteration)
  cc_flag: arch=compute_100a for B100/B200 (SM100), compute_120a for B200X (SM120)

REQUIRES:
  - CUDA 12.8+
  - Blackwell GPU (SM100 or SM120)
  - PyTorch with CUDA
"""
import warnings
import os
from pathlib import Path
from packaging.version import parse, Version
from setuptools import setup, find_packages
import subprocess
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "sageattn3_simplified"

FORCE_BUILD = os.getenv("FAHOPPER_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FAHOPPER_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
FORCE_CXX11_ABI = os.getenv("FAHOPPER_FORCE_CXX11_ABI", "FALSE") == "TRUE"


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])
    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]


cmdclass = {}
ext_modules = []

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.8"):
        raise RuntimeError("Sage3 is only supported on CUDA 12.8 and above")
    cc_major, cc_minor = torch.cuda.get_device_capability()
    if (cc_major, cc_minor) == (10, 0):  # sm_100 (B100/B200)
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_100a,code=sm_100a")
    elif (cc_major, cc_minor) == (12, 0):  # sm_120 (B200X)
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_120a,code=sm_120a")
    elif (cc_major, cc_minor) == (12, 1):  # sm_121
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_121a,code=sm_121a")
    else:
        raise RuntimeError("Unsupported GPU")

    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    repo_dir = Path(this_dir)
    cutlass_dir = repo_dir / "csrc" / "cutlass"
    (repo_dir / "csrc").mkdir(parents=True, exist_ok=True)
    if not cutlass_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/NVIDIA/cutlass.git", str(cutlass_dir)],
            check=True
        )

    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ptxas-options=--verbose,--warn-on-local-memory-usage",
        "-lineinfo",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "-DNDEBUG",
        "-DQBLKSIZE=128",
        "-DKBLKSIZE=128",
        "-DCTA256",
        "-DDQINRMEM",
    ]

    include_dirs = [
        repo_dir / "sageattn3_simplified",  # point to simplified package
        cutlass_dir / "include",
        cutlass_dir / "tools" / "util" / "include",
    ]

    ext_modules.append(
        CUDAExtension(
            name="fp4attn_cuda",
            sources=["sageattn3_simplified/blackwell/api.cu"],  # simplified source
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(
                    nvcc_flags + ["-DEXECMODE=0"] + cc_flag
                ),
            },
            include_dirs=include_dirs,
            libraries=["cuda"]
        )
    )
    ext_modules.append(
        CUDAExtension(
            name="fp4quant_cuda",
            sources=["sageattn3_simplified/quantization/fp4_quantization_4d.cu"],  # simplified source
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(
                    nvcc_flags + ["-DEXECMODE=0"] + cc_flag
                ),
            },
            include_dirs=include_dirs,
            libraries=["cuda"]
        )
    )


class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        super().run()

setup(
    name=PACKAGE_NAME,
    version="1.0.0",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    description="FP4FlashAttention (Simplified/Documented Version)",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension}
    if ext_modules
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
)
