#!/usr/bin/env python
"""Run Transformer vs BiMamba benchmarks across multiple input resolutions."""

from __future__ import annotations

import argparse
import csv
import traceback
from datetime import datetime
from pathlib import Path
import sys

# Add repository root to path so `import stormer` works when running this file directly.
ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.comparison.transformer_vs_mamba.benchmark_core import (  # noqa: E402
    benchmark_pair,
    cleanup_cuda,
    detect_environment_label,
    ensure_cuda_or_raise,
    get_environment_metadata,
    get_dtype,
    is_oom_error,
    write_json,
)


DEFAULT_RESOLUTIONS = [
    "32x64",
    "64x128",
    "121x240",
    "128x256",
    "256x256",
    "256x512",
    "512x256",
]


def parse_resolution(value):
    normalized = value.lower().replace("*", "x")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid resolution '{value}'. Expected HxW, for example 121x240.")

    try:
        height = int(parts[0])
        width = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid resolution '{value}'. Height and width must be integers.") from exc

    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError(f"Invalid resolution '{value}'. Height and width must be positive.")

    return height, width


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark Transformer vs BiMamba across multiple resolutions")
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=DEFAULT_RESOLUTIONS,
        help="List of resolutions in HxW form, for example 32x64 121x240 512x256",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./artifacts/benchmarks/transformer_vs_bimamba",
        help="Root directory under which a timestamped benchmark run folder will be created",
    )
    parser.add_argument(
        "--env_label",
        type=str,
        default=None,
        help="Optional environment label for the run directory",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for benchmarking")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size")
    parser.add_argument("--transformer_depth", type=int, default=24, help="Transformer model depth")
    parser.add_argument("--candidate_depth", type=int, default=14, help="BiMamba model depth")
    parser.add_argument(
        "--inference_dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help="Inference dtype: fp32, bf16, fp16",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Number of measurement iterations")
    return parser


def create_run_dir(output_root, env_label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{env_label}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def flatten_row(resolution, status, result_path, comparison=None, error_type=None, error_message=None):
    row = {
        "resolution": resolution,
        "status": status,
        "error_type": error_type or "",
        "error_message": error_message or "",
        "result_path": str(result_path),
    }

    if comparison is None:
        row.update(
            {
                "transformer_num_parameters": "",
                "transformer_peak_memory_gb": "",
                "transformer_time_per_sample_ms": "",
                "transformer_throughput_samples_per_sec": "",
                "bimamba_num_parameters": "",
                "bimamba_peak_memory_gb": "",
                "bimamba_time_per_sample_ms": "",
                "bimamba_throughput_samples_per_sec": "",
                "speedup": "",
                "memory_reduction_pct": "",
                "parameter_difference_pct": "",
            }
        )
        return row

    row.update(
        {
            "transformer_num_parameters": comparison["transformer"]["num_parameters"],
            "transformer_peak_memory_gb": comparison["transformer"]["peak_memory_gb"],
            "transformer_time_per_sample_ms": comparison["transformer"]["time_per_sample_ms"],
            "transformer_throughput_samples_per_sec": comparison["transformer"]["throughput_samples_per_sec"],
            "bimamba_num_parameters": comparison["bimamba"]["num_parameters"],
            "bimamba_peak_memory_gb": comparison["bimamba"]["peak_memory_gb"],
            "bimamba_time_per_sample_ms": comparison["bimamba"]["time_per_sample_ms"],
            "bimamba_throughput_samples_per_sec": comparison["bimamba"]["throughput_samples_per_sec"],
            "speedup": comparison["comparison"]["speedup"],
            "memory_reduction_pct": comparison["comparison"]["memory_reduction_pct"],
            "parameter_difference_pct": comparison["comparison"]["parameter_difference_pct"],
        }
    )
    return row


def write_summary_csv(summary_csv, rows):
    fieldnames = [
        "resolution",
        "status",
        "error_type",
        "error_message",
        "transformer_num_parameters",
        "transformer_peak_memory_gb",
        "transformer_time_per_sample_ms",
        "transformer_throughput_samples_per_sec",
        "bimamba_num_parameters",
        "bimamba_peak_memory_gb",
        "bimamba_time_per_sample_ms",
        "bimamba_throughput_samples_per_sec",
        "speedup",
        "memory_reduction_pct",
        "parameter_difference_pct",
        "result_path",
    ]

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        ensure_cuda_or_raise()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    parsed_resolutions = [parse_resolution(value) for value in args.resolutions]
    env_label = args.env_label or detect_environment_label()
    run_dir = create_run_dir(args.output_root, env_label)
    resolutions_dir = run_dir / "resolutions"
    resolutions_dir.mkdir(parents=True, exist_ok=True)

    metadata = get_environment_metadata()
    metadata["requested_config"] = {
        "resolutions": [f"{height}x{width}" for height, width in parsed_resolutions],
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "hidden_size": args.hidden_size,
        "transformer_depth": args.transformer_depth,
        "candidate_depth": args.candidate_depth,
        "candidate_model": "bimamba",
        "inference_dtype": args.inference_dtype,
        "warmup": args.warmup,
        "runs": args.runs,
        "output_root": str(Path(args.output_root)),
    }
    metadata["run_directory"] = str(run_dir)
    write_json(run_dir / "run_metadata.json", metadata)

    print("=" * 70)
    print("Transformer vs BiMamba Resolution Sweep")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print(f"GPU: {metadata['gpu_name']}")
    print(f"Resolutions: {', '.join(metadata['requested_config']['resolutions'])}")
    print(f"Transformer Depth: {args.transformer_depth}")
    print(f"BiMamba Depth: {args.candidate_depth}")
    print(f"Inference Dtype: {args.inference_dtype}")
    print()

    rows = []
    
    skip_oom = False
    dtype = get_dtype(args.inference_dtype)

    for height, width in parsed_resolutions:
        resolution_label = f"{height}x{width}"
        resolution_dir = resolutions_dir / resolution_label
        resolution_dir.mkdir(parents=True, exist_ok=True)
        result_path = resolution_dir / "benchmark_results.json"

        print("-" * 70)
        print(f"Running benchmark for {resolution_label}")
        print("-" * 70)
        
        if skip_oom:
            print(f"Skipping {resolution_label} due to previous OOM")
            rows.append(flatten_row(resolution_label, "skipped", result_path, error_type="oom_cascade"))
            continue

        try:
            comparison = benchmark_pair(
                batch_size=args.batch_size,
                img_height=height,
                img_width=width,
                patch_size=args.patch_size,
                hidden_size=args.hidden_size,
                transformer_depth=args.transformer_depth,
                candidate_depth=args.candidate_depth,
                warmup=args.warmup,
                runs=args.runs,
                dtype=dtype,
            )
            comparison["environment"] = metadata
            comparison["status"] = "ok"
            write_json(result_path, comparison)
            rows.append(flatten_row(resolution_label, "ok", result_path, comparison=comparison))
            print(f"Completed {resolution_label}: speedup {comparison['comparison']['speedup']:.2f}x")
        except Exception as exc:  # noqa: BLE001
            cleanup_cuda()
            error_type = "oom" if is_oom_error(exc) else exc.__class__.__name__
            if error_type == "oom":
                # For resolution sweep, we only skip if BOTH models failed. 
                # But here we skip the resolution if either fails to be safe.
                skip_oom = True
            error_payload = {
                "status": "oom" if error_type == "oom" else "error",
                "error_type": error_type,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "config": {
                    "batch_size": args.batch_size,
                    "img_size": [height, width],
                    "patch_size": args.patch_size,
                    "hidden_size": args.hidden_size,
                    "transformer_depth": args.transformer_depth,
                    "candidate_depth": args.candidate_depth,
                    "candidate_model": "bimamba",
                    "inference_dtype": args.inference_dtype,
                    "warmup": args.warmup,
                    "runs": args.runs,
                },
                "environment": metadata,
            }
            write_json(result_path, error_payload)
            rows.append(
                flatten_row(
                    resolution_label,
                    error_payload["status"],
                    result_path,
                    error_type=error_type,
                    error_message=str(exc),
                )
            )
            print(f"Failed {resolution_label}: {error_type} - {exc}")

    summary = {
        "run_directory": str(run_dir),
        "environment": metadata,
        "results": rows,
    }
    write_json(run_dir / "summary.json", summary)
    write_summary_csv(run_dir / "summary.csv", rows)

    print("\n" + "=" * 70)
    print("Sweep complete")
    print("=" * 70)
    print(f"Metadata: {run_dir / 'run_metadata.json'}")
    print(f"Summary JSON: {run_dir / 'summary.json'}")
    print(f"Summary CSV: {run_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
