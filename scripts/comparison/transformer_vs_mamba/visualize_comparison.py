#!/usr/bin/env python
"""
Visualization script for Transformer vs BiMamba comparison results.

This script generates:
- RMSE vs Lead Time plots
- Per-variable performance comparison
- Performance improvement bar charts
- Memory and speed comparison charts

Requires matplotlib and seaborn.
"""

import pandas as pd
import json
import os
from pathlib import Path
import argparse
import sys

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("WARNING: matplotlib and/or seaborn not installed. Install with:")
    print("  pip install matplotlib seaborn")


def plot_rmse_vs_leadtime(results_df: pd.DataFrame, output_dir: Path, candidate_label: str):
    """Plot RMSE vs Lead Time for both models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in ['Transformer', candidate_label]:
        data = results_df[results_df['model'] == model]
        mean_rmse = data.groupby('lead_time')['rmse'].mean()
        std_rmse = data.groupby('lead_time')['rmse'].std()

        ax.plot(mean_rmse.index, mean_rmse.values, 'o-', label=model, linewidth=2, markersize=8)
        ax.fill_between(
            mean_rmse.index,
            mean_rmse - std_rmse,
            mean_rmse + std_rmse,
            alpha=0.2
        )

    ax.set_xlabel('Forecast Lead Time (hours)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(f'Forecast Skill: {candidate_label} vs Transformer', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = output_dir / 'rmse_vs_leadtime.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_per_variable_comparison(results_df: pd.DataFrame, output_dir: Path, lead_time=72):
    """Plot per-variable comparison at a specific lead time."""
    data_lt = results_df[results_df['lead_time'] == lead_time]
    if data_lt.empty:
        print(f"  ⚠ No evaluation rows found for lead time {lead_time}h, skipping per-variable plot")
        return

    fig, ax = plt.subplots(figsize=(16, 8))

    # Get top 20 variables by average RMSE (for readability)
    top_vars = data_lt.groupby('variable')['rmse'].mean().nlargest(20).index

    data_lt_filtered = data_lt[data_lt['variable'].isin(top_vars)]
    pivot = data_lt_filtered.pivot_table(values='rmse', index='variable', columns='model')

    pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Variable', fontsize=12)
    ax.set_ylabel(f'RMSE at {lead_time}h', fontsize=12)
    ax.set_title(f'Per-Variable Performance at {lead_time}h Lead Time (Top 20 Variables)', fontsize=14, fontweight='bold')
    ax.legend(title='Model', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_file = output_dir / f'per_variable_{lead_time}h.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_improvement(results_df: pd.DataFrame, output_dir: Path, candidate_label: str):
    """Plot performance improvement (candidate vs Transformer)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    trans_mean = results_df[results_df['model'] == 'Transformer'].groupby('lead_time')['rmse'].mean()
    candidate_mean = results_df[results_df['model'] == candidate_label].groupby('lead_time')['rmse'].mean()

    improvement = 100 * (trans_mean - candidate_mean) / trans_mean

    colors = ['green' if x > 0 else 'red' for x in improvement.values]
    ax.bar(improvement.index, improvement.values, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)

    ax.set_xlabel('Forecast Lead Time (hours)', fontsize=12)
    ax.set_ylabel('Performance Improvement (%)', fontsize=12)
    ax.set_title(
        f'{candidate_label} Performance Relative to Transformer\n(Positive = {candidate_label} Better)',
        fontsize=14,
        fontweight='bold',
    )
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    output_file = output_dir / 'improvement.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_benchmark_comparison(benchmark_file: Path, output_dir: Path):
    """Plot memory and speed comparison from benchmark results."""
    if not benchmark_file.exists():
        print(f"  ⚠ Benchmark file not found: {benchmark_file}")
        return

    with open(benchmark_file) as f:
        data = json.load(f)

    trans = data['transformer']
    candidate_key = data.get("config", {}).get("candidate_model", "bimamba")
    candidate_label = data.get("config", {}).get("candidate_label", candidate_key.title())
    candidate = data[candidate_key]

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Memory usage
    ax = axes[0]
    models = ['Transformer', candidate_label]
    memory = [trans['peak_memory_gb'], candidate['peak_memory_gb']]
    colors = ['#3498db', '#e74c3c']
    ax.bar(models, memory, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Peak Memory (GB)', fontsize=12)
    ax.set_title('GPU Memory Usage', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(memory):
        ax.text(i, v + 0.1, f'{v:.2f} GB', ha='center', fontsize=11)

    # Plot 2: Inference speed
    ax = axes[1]
    speed = [trans['time_per_sample_ms'], candidate['time_per_sample_ms']]
    ax.bar(models, speed, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Time per Sample (ms)', fontsize=12)
    ax.set_title('Inference Speed', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(speed):
        ax.text(i, v + 0.5, f'{v:.2f} ms', ha='center', fontsize=11)

    # Plot 3: Parameters
    ax = axes[2]
    params = [trans['num_parameters'] / 1e6, candidate['num_parameters'] / 1e6]
    ax.bar(models, params, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Parameters (millions)', fontsize=12)
    ax.set_title('Model Size', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(params):
        ax.text(i, v + 0.5, f'{v:.1f}M', ha='center', fontsize=11)

    plt.tight_layout()

    output_file = output_dir / 'benchmark_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_training_curves(output_dir: Path, candidate_label: str, candidate_subdir: str, candidate_log_name: str):
    """Plot training curves from CSV logs if available."""
    transformer_log_dir = output_dir / 'transformer' / 'transformer_logs'
    candidate_log_dir = output_dir / candidate_subdir / candidate_log_name

    # Try to find CSV log files
    transformer_csv = None
    candidate_csv = None

    if transformer_log_dir.exists():
        csv_files = list(transformer_log_dir.glob('**/metrics.csv'))
        if csv_files:
            transformer_csv = csv_files[0]

    if candidate_log_dir.exists():
        csv_files = list(candidate_log_dir.glob('**/metrics.csv'))
        if csv_files:
            candidate_csv = csv_files[0]

    if not transformer_csv or not candidate_csv:
        print("  ⚠ Training log CSV files not found, skipping training curves")
        return

    # Read CSV files
    trans_df = pd.read_csv(transformer_csv)
    candidate_df = pd.read_csv(candidate_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training loss
    ax = axes[0]
    if 'train/loss' in trans_df.columns:
        ax.plot(trans_df['epoch'], trans_df['train/loss'], label='Transformer', linewidth=2)
        ax.plot(candidate_df['epoch'], candidate_df['train/loss'], label=candidate_label, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title('Training Loss Over Time', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    # Plot 2: Validation metric
    ax = axes[1]
    val_col = 'val/w_mse_aggregate_72_hrs_ensemble_mean'
    if val_col in trans_df.columns:
        ax.plot(trans_df['epoch'], trans_df[val_col], label='Transformer', linewidth=2)
        ax.plot(candidate_df['epoch'], candidate_df[val_col], label=candidate_label, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation wMSE (72h)', fontsize=12)
        ax.set_title('Validation Performance Over Time', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / 'training_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Transformer vs candidate-model comparison')
    parser.add_argument('--results_dir', type=str,
                        default=os.getenv('EXPERIMENT_DIR', './artifacts/stage2_evaluation'),
                        help='Directory containing comparison results')
    parser.add_argument(
        '--per_variable_lead_times',
        type=str,
        default='6,72',
        help='Comma-separated lead times for per-variable comparison plots'
    )
    parser.add_argument(
        '--candidate_label',
        type=str,
        default='BiMamba',
        help='Display label for the candidate model'
    )
    parser.add_argument(
        '--candidate_subdir',
        type=str,
        default='finetune_bimamba',
        help='Experiment subdirectory for candidate model'
    )
    parser.add_argument(
        '--candidate_log_name',
        type=str,
        default='finetune_bimamba_logs',
        help='Logger directory name for candidate model'
    )

    args = parser.parse_args()

    if not PLOTTING_AVAILABLE:
        print("ERROR: matplotlib and seaborn are required for visualization")
        print("Install with: pip install matplotlib seaborn")
        sys.exit(1)

    results_dir = Path(args.results_dir)
    per_variable_lead_times = [int(v.strip()) for v in args.per_variable_lead_times.split(',') if v.strip()]

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100

    print("=" * 60)
    print("Generating Comparison Visualizations")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print()

    # Create plots directory
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Load evaluation results
    eval_file = results_dir / 'evaluation_results.csv'
    if eval_file.exists():
        print("Generating evaluation plots...")
        results_df = pd.read_csv(eval_file)

        plot_rmse_vs_leadtime(results_df, plots_dir, args.candidate_label)
        for lead_time in per_variable_lead_times:
            plot_per_variable_comparison(results_df, plots_dir, lead_time=lead_time)
        plot_improvement(results_df, plots_dir, args.candidate_label)
        print()
    else:
        print(f"⚠ Evaluation results not found: {eval_file}")
        print()

    # Plot benchmark comparison
    benchmark_file = results_dir / 'benchmark_results.json'
    if benchmark_file.exists():
        print("Generating benchmark plots...")
        plot_benchmark_comparison(benchmark_file, plots_dir)
        print()
    else:
        print(f"⚠ Benchmark results not found: {benchmark_file}")
        print()

    # Plot training curves
    print("Generating training curves...")
    plot_training_curves(
        results_dir,
        args.candidate_label,
        args.candidate_subdir,
        args.candidate_log_name,
    )
    print()

    print("=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print(f"Plots saved to: {plots_dir}")
    print()
    print("Generated plots:")
    for plot_file in sorted(plots_dir.glob('*.png')):
        print(f"  - {plot_file.name}")


if __name__ == "__main__":
    main()
