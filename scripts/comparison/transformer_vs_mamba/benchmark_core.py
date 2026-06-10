#!/usr/bin/env python
"""Shared benchmark helpers for Transformer vs BiMamba comparisons."""

from __future__ import annotations

import json
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Add repository root to path so `import stormer` works when running this file directly.
ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from stormer.models.hub.bimamba_stormer import BiMambaStormer
from stormer.models.hub.stormer import Stormer


VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
] + [f"geopotential_{p}" for p in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]] \
  + [f"u_component_of_wind_{p}" for p in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]] \
  + [f"v_component_of_wind_{p}" for p in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]] \
  + [f"temperature_{p}" for p in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]] \
  + [f"specific_humidity_{p}" for p in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]]


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_memory(model, x, variables, t, warmup=10, runs=50):
    """Benchmark peak memory usage in GB."""
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x, variables, t)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(x, variables, t)

    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
    return peak_memory


def benchmark_speed(model, x, variables, t, warmup=10, runs=100):
    """Benchmark inference speed in seconds per forward pass."""
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x, variables, t)

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(x, variables, t)

    torch.cuda.synchronize()
    elapsed = time.time() - start
    return elapsed / runs


def benchmark_model(model_class, config, x, variables, t, name="Model", warmup=10, runs=100, dtype=torch.float32):
    """Run memory and speed benchmarks for one model."""
    print(f"\nBenchmarking {name}...")
    print("-" * 60)

    model = model_class(**config).to(device="cuda", dtype=dtype)
    model.eval()
    
    x = x.to(dtype=dtype)
    t = t.to(dtype=dtype)

    num_params = count_parameters(model)
    print(f"  Parameters: {num_params:,}")

    print("  Measuring memory usage...")
    peak_memory = benchmark_memory(model, x, variables, t, warmup=warmup, runs=runs)
    print(f"  Peak memory: {peak_memory:.2f} GB")

    print("  Measuring inference speed...")
    time_per_sample = benchmark_speed(model, x, variables, t, warmup=warmup, runs=runs)
    print(f"  Time per sample: {time_per_sample * 1000:.2f} ms")
    print(f"  Throughput: {1 / time_per_sample:.2f} samples/sec")

    results = {
        "num_parameters": int(num_params),
        "peak_memory_gb": float(peak_memory),
        "time_per_sample_sec": float(time_per_sample),
        "time_per_sample_ms": float(time_per_sample * 1000),
        "throughput_samples_per_sec": float(1 / time_per_sample),
    }

    del model
    torch.cuda.empty_cache()
    return results


def build_base_config(img_height, img_width, patch_size, hidden_size, depth):
    return {
        "in_img_size": [img_height, img_width],
        "variables": VARIABLES,
        "patch_size": patch_size,
        "hidden_size": hidden_size,
        "depth": depth,
    }


def build_transformer_config(base_config):
    return {
        **base_config,
        "num_heads": 16,
        "mlp_ratio": 4.0,
    }


def build_bimamba_config(base_config):
    return {
        **base_config,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "d_state": 32, # Wide
        "d_conv": 4,
        "expand": 2, # Production BiMamba-Vim setting
    }


def create_inputs(batch_size, img_height, img_width, variables=None):
    variables = variables or VARIABLES
    x = torch.randn(batch_size, len(variables), img_height, img_width).cuda()
    t = torch.full((batch_size,), 6.0).cuda()
    return x, t, variables


def compute_comparison(transformer_results, candidate_results):
    return {
        "memory_reduction_pct": 100 * (1 - candidate_results["peak_memory_gb"] / transformer_results["peak_memory_gb"]),
        "speedup": transformer_results["time_per_sample_sec"] / candidate_results["time_per_sample_sec"],
        "parameter_difference": candidate_results["num_parameters"] - transformer_results["num_parameters"],
        "parameter_difference_pct": 100
        * (candidate_results["num_parameters"] - transformer_results["num_parameters"])
        / transformer_results["num_parameters"],
    }


def benchmark_pair(
    batch_size,
    img_height,
    img_width,
    patch_size,
    hidden_size,
    transformer_depth,
    candidate_depth,
    warmup,
    runs,
    dtype=torch.float32,
):
    x, t, variables = create_inputs(batch_size, img_height, img_width, VARIABLES)

    # Benchmark Transformer
    transformer_base = build_base_config(img_height, img_width, patch_size, hidden_size, transformer_depth)
    transformer_results = benchmark_model(
        Stormer,
        build_transformer_config(transformer_base),
        x,
        variables,
        t,
        name="Transformer",
        warmup=warmup,
        runs=runs,
        dtype=dtype,
    )

    # Benchmark Candidate
    candidate_base = build_base_config(img_height, img_width, patch_size, hidden_size, candidate_depth)
    candidate_results = benchmark_model(
        BiMambaStormer,
        build_bimamba_config(candidate_base),
        x,
        variables,
        t,
        name="BiMamba",
        warmup=warmup,
        runs=runs,
        dtype=dtype,
    )

    return {
        "transformer": transformer_results,
        "bimamba": candidate_results,
        "comparison": compute_comparison(transformer_results, candidate_results),
        "config": {
            "batch_size": batch_size,
            "img_size": [img_height, img_width],
            "patch_size": patch_size,
            "hidden_size": hidden_size,
            "transformer_depth": transformer_depth,
            "candidate_depth": candidate_depth,
            "warmup": warmup,
            "runs": runs,
            "candidate_model": "bimamba",
            "candidate_label": "BiMamba",
            "dtype": str(dtype),
        },
    }


def _safe_version(package_name):
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(package_name)
    except Exception:
        return None


def detect_environment_label():
    host = socket.gethostname().lower()
    gpu = ""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0).lower()

    cluster = host.split(".")[0]

    if "h200" in gpu:
        accelerator = "h200"
    elif "3090" in gpu:
        accelerator = "rtx3090"
    elif gpu:
        accelerator = gpu.replace("nvidia", "").strip().replace(" ", "_")
    else:
        accelerator = "cpu"

    return f"{cluster}_{accelerator}".replace("__", "_")


def get_environment_metadata():
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "environment_label": detect_environment_label(),
        "python_version": sys.version.split()[0],
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": gpu_name,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "xformers_version": _safe_version("xformers"),
        "mamba_ssm_version": _safe_version("mamba-ssm"),
        "causal_conv1d_version": _safe_version("causal-conv1d"),
    }


def ensure_cuda_or_raise():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Benchmarking requires GPU.")


def is_oom_error(exc):
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def get_dtype(dtype_str):
    if dtype_str == "fp32":
        return torch.float32
    elif dtype_str == "bf16":
        return torch.bfloat16
    elif dtype_str == "fp16":
        return torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
