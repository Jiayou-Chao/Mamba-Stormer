"""Regenerate the BiMamba speedup figure from benchmark sweep outputs."""
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import math

# Target resolutions and their expected sequence lengths (L = H/4 * W/4)
RESOLUTIONS = {
    "32x64": 128,
    "64x128": 512,
    "121x240": 1860, # (124/4 * 240/4)
    "128x256": 2048,
    "256x256": 4096,
    "256x512": 8192,
    "512x256": 8192, # Same scale as 256x512
    "512x512": 16384
}

def get_latest_results(benchmark_root: Path):
    all_summaries = list(benchmark_root.glob("**/summary.json"))
    
    # Store latest result per resolution: {res: {"speedup": float, "timestamp": datetime, "gpu": str}}
    best_results = {}
    
    for summary_path in all_summaries:
        try:
            with open(summary_path) as f:
                data = json.load(f)
            
            # Extract timestamp and config
            ts_str = data["environment"]["timestamp"]
            ts = datetime.fromisoformat(ts_str)
            gpu = data["environment"]["gpu_name"]
            config = data["environment"].get("requested_config", {})
            
            batch_size = config.get("batch_size", 0)
            dtype = config.get("inference_dtype", "fp32")
            
            # Heuristic for "Production Quality":
            # 1. Prefer batch_size=4
            # 2. Prefer fp32
            # 3. Then latest timestamp
            is_standard = (batch_size == 4 and dtype == "fp32")
            
            for res_data in data["results"]:
                res = res_data["resolution"]
                if res_data["status"] == "ok":
                    speedup = res_data["speedup"]
                    
                    is_h200 = "H200" in gpu
                    
                    if res not in best_results:
                        best_results[res] = {"speedup": speedup, "timestamp": ts, "gpu": gpu, "is_standard": is_standard}
                    else:
                        current = best_results[res]
                        
                        # Tie-breaking logic:
                        # 1. Prioritize H200 for consistency if requested by user
                        # 2. Prefer standard configuration (batch_size=4, fp32)
                        # 3. Then latest timestamp
                        replace = False
                        is_h200 = "H200" in gpu
                        was_h200 = "H200" in current["gpu"]

                        if is_h200 and not was_h200:
                            replace = True
                        elif is_h200 == was_h200:
                            if is_standard and not current["is_standard"]:
                                replace = True
                            elif is_standard == current["is_standard"] and ts > current["timestamp"]:
                                replace = True
                        
                        if replace:
                            best_results[res] = {"speedup": speedup, "timestamp": ts, "gpu": gpu, "is_standard": is_standard}
        except Exception as e:
            print(f"Warning: Could not parse {summary_path}: {e}")
            
    return best_results

def plot_results(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })

    TRANSFORMER_COLOR = '#1565C0'
    MAMBA_COLOR       = '#E65100'

    # Prepare data for plotting
    plot_data = []
    for res, L in RESOLUTIONS.items():
        if res in results:
            plot_data.append((L, results[res]["speedup"], res, results[res]["gpu"]))
    
    plot_data.sort() # Sort by sequence length
    
    if not plot_data:
        print("Error: No valid results found to plot!")
        return

    seq_lens = [d[0] for d in plot_data]
    speedups = [d[1] for d in plot_data]
    
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.set_facecolor('#FAFAFA')

    # Shade speedup / slowdown regions
    ax.fill_between([90, 20000], [1.0, 1.0], [3.0, 3.0], alpha=0.04, color='green')
    ax.fill_between([90, 20000], [0.5, 0.5], [1.0, 1.0], alpha=0.04, color='red')

    # 1× baseline
    ax.axhline(y=1.0, color='#C62828', linestyle='--', lw=1.5,
               label='1× baseline (no speedup)', alpha=0.8)

    # BiMamba main line
    ax.plot(seq_lens, speedups,
            'o-', color=MAMBA_COLOR, lw=2.2, ms=8,
            markerfacecolor='white', markeredgewidth=2.0,
            label='BiMamba (Latest Benchmarks)', zorder=5)

    # Annotate points
    for i, (L, s, res, gpu) in enumerate(plot_data):
        # Handle identical or crowded sequence lengths
        same_L_points = [p for p in plot_data if abs(p[0] - L) < 1.0]
        
        y_off = 12
        x_off = 0
        ha = 'center'
        va = 'bottom'
        
        if len(same_L_points) > 1:
            # Multiple points at the same X (e.g. 8K cluster)
            # Shift labels horizontally to avoid overlap
            idx = same_L_points.index((L, s, res, gpu))
            x_off = -20 if idx == 0 else 20
            ha = 'right' if idx == 0 else 'left'
        else:
            # Handle close-but-not-identical points (e.g. 2K cluster)
            is_crowded = False
            if i > 0 and (math.log2(L) - math.log2(plot_data[i-1][0]) < 0.25):
                is_crowded = True
            if i < len(plot_data) - 1 and (math.log2(plot_data[i+1][0]) - math.log2(L) < 0.25):
                is_crowded = True

            if is_crowded and res == "128x256":
                y_off = -12 # Below the point, but closer than before
                va = 'top'

        label = f"{s:.2f}x\n({res})"
        ax.annotate(label,
                    xy=(L, s),
                    xytext=(x_off, y_off),
                    textcoords='offset points',
                    fontsize=8, color=MAMBA_COLOR, ha=ha, fontweight='bold',
                    va=va)

    # Axes
    ax.set_xscale('log', base=2)
    ax.set_xticks([128, 512, 1024, 2048, 4096, 8192, 16384])
    ax.set_xticklabels(['128', '512', '1K', '2K', '4K', '8K', '16K'])
    ax.set_xlim(90, 20000)
    ax.set_ylim(0.7, max(max(speedups) * 1.2, 2.0))
    ax.set_xlabel('Sequence Length  $L$  (log scale)', fontsize=12)
    ax.set_ylabel('Throughput Speedup  (BiMamba / Transformer)', fontsize=12)
    
    # Identify GPUs used
    gpus_used = sorted(list(set(d[3] for d in plot_data)))
    gpu_str = " & ".join(gpus_used)
    
    ax.set_title(f'BiMamba Throughput Speedup vs. Sequence Length\n'
                 f'Auto-collected from: {gpu_str}',
                 fontsize=13, fontweight='bold')

    # Resolution labels on top x-axis - Group by L to avoid collision
    res_groups = {}
    for L, s, res, gpu in plot_data:
        L_key = round(float(L), 1)
        if L_key not in res_groups: res_groups[L_key] = []
        res_groups[L_key].append(res)
    
    top_xticks = sorted(res_groups.keys())
    # Use newline and ampersand for merged labels; sort to put 512x256 first for the 8K pair
    top_xticklabels = ["\n& ".join(sorted(res_groups[k], reverse=True)) for k in top_xticks]

    ax2 = ax.twiny()
    ax2.set_xscale('log', base=2)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(top_xticks)
    ax2.set_xticklabels(top_xticklabels, fontsize=7.5, rotation=45, ha='left')

    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}x'))

    plt.tight_layout()
    path = output_dir / 'speedup_vs_resolution.png'
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')
    
    # Print a summary table for the user
    print("\nSummary of results collected:")
    print(f"{'Resolution':<12} | {'L':<8} | {'Speedup':<8} | {'GPU':<20}")
    print("-" * 55)
    for L, s, res, gpu in plot_data:
        print(f"{res:<12} | {int(L):<8} | {s:<8.3f} | {gpu:<20}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the BiMamba speedup figure")
    parser.add_argument(
        "--benchmark_root",
        default="./artifacts/benchmarks/transformer_vs_bimamba",
        help="Root directory containing benchmark sweep runs",
    )
    parser.add_argument(
        "--output_dir",
        default="./artifacts/paper_figures",
        help="Directory for the generated figure",
    )
    args = parser.parse_args()

    results = get_latest_results(Path(args.benchmark_root))
    plot_results(results, Path(args.output_dir))
