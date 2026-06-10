#!/usr/bin/env python
"""Benchmark memory usage and inference speed for Transformer vs BiMamba."""

import argparse
import os
import sys
from pathlib import Path

# Add repository root to path so `import stormer` works when running this file directly.
ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.comparison.transformer_vs_mamba.benchmark_core import (  # noqa: E402
    benchmark_pair,
    ensure_cuda_or_raise,
    get_environment_metadata,
    get_dtype,
    write_json,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark Transformer vs BiMamba")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for benchmarking")
    parser.add_argument("--img_height", type=int, default=64, help="Image height")
    parser.add_argument("--img_width", type=int, default=32, help="Image width")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size")
    parser.add_argument("--transformer_depth", type=int, default=24, help="Transformer depth")
    parser.add_argument("--candidate_depth", type=int, default=14, help="Candidate model depth")
    parser.add_argument(
        "--inference_dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help="Inference dtype: fp32, bf16, fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getenv("EXPERIMENT_DIR", "./experiments/comparison"),
        help="Output directory for results",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Number of measurement iterations")
    return parser


def print_summary(results_file, comparison):
    candidate_key = "bimamba"
    candidate_label = comparison["config"].get("candidate_label", candidate_key.title())
    transformer_results = comparison["transformer"]
    candidate_results = comparison[candidate_key]
    metrics = comparison["comparison"]

    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    print(f"✓ Results saved to: {results_file}")

    print("\n" + "-" * 60)
    print(f"{'Metric':<35} {'Transformer':<15} {candidate_label:<15} {'Improvement':<15}")
    print("-" * 60)

    print(
        f"{'Parameters':<35} {transformer_results['num_parameters']:>14,} "
        f"{candidate_results['num_parameters']:>14,} {metrics['parameter_difference_pct']:>+13.1f}%"
    )
    print(
        f"{'Peak Memory (GB)':<35} {transformer_results['peak_memory_gb']:>14.2f} "
        f"{candidate_results['peak_memory_gb']:>14.2f} {metrics['memory_reduction_pct']:>+13.1f}%"
    )
    print(
        f"{'Inference Time (ms)':<35} {transformer_results['time_per_sample_ms']:>14.2f} "
        f"{candidate_results['time_per_sample_ms']:>14.2f} {metrics['speedup']:>13.2f}x"
    )
    print(
        f"{'Throughput (samples/s)':<35} {transformer_results['throughput_samples_per_sec']:>14.2f} "
        f"{candidate_results['throughput_samples_per_sec']:>14.2f} {metrics['speedup']:>13.2f}x"
    )
    print("-" * 60)

    print("\n" + "=" * 60)
    print("Interpretation")
    print("=" * 60)

    mem_reduction = metrics["memory_reduction_pct"]
    speedup = metrics["speedup"]

    if mem_reduction > 0:
        print(f"✓ {candidate_label} uses {mem_reduction:.1f}% less memory than Transformer")
    else:
        print(f"⚠ {candidate_label} uses {-mem_reduction:.1f}% more memory than Transformer")

    if speedup > 1:
        print(f"✓ {candidate_label} is {speedup:.2f}x faster than Transformer")
    else:
        print(f"⚠ {candidate_label} is {1 / speedup:.2f}x slower than Transformer")

    print("\n" + "=" * 60)
    print("Benchmarking Complete!")


def main():
    parser = build_parser()
    args = parser.parse_args()

    candidate_label = "BiMamba"

    print("=" * 60)
    print(f"Transformer vs {candidate_label} Benchmarking")
    print("=" * 60)
    print("Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.img_height} x {args.img_width}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Transformer Depth: {args.transformer_depth}")
    print(f"  Candidate Depth: {args.candidate_depth}")
    print(f"  Inference Dtype: {args.inference_dtype}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Measurement iterations: {args.runs}")
    print(f"  Candidate model: {candidate_label}")
    print()

    try:
        ensure_cuda_or_raise()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    metadata = get_environment_metadata()
    print(f"GPU: {metadata['gpu_name']}")
    print()

    dtype = get_dtype(args.inference_dtype)

    comparison = benchmark_pair(
        batch_size=args.batch_size,
        img_height=args.img_height,
        img_width=args.img_width,
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        transformer_depth=args.transformer_depth,
        candidate_depth=args.candidate_depth,
        warmup=args.warmup,
        runs=args.runs,
        dtype=dtype,
    )
    comparison["environment"] = metadata

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "benchmark_results.json"
    write_json(results_file, comparison)

    print_summary(results_file, comparison)


if __name__ == "__main__":
    main()
