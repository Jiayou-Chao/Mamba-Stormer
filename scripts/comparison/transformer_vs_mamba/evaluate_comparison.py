#!/usr/bin/env python
"""
Comprehensive evaluation script for Transformer vs BiMamba comparison.

This script can run the legacy per-batch comparison flow or a WeatherBench 2
comparison mode that:
- restricts evaluation to selected years (for example 2020 only),
- evaluates an ensemble mean over valid rollout intervals, and
- aggregates RMSE as sqrt(mean_t mean_space mse), matching WB2's definition.
"""

import argparse
import gc
import importlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

# Add stormer to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stormer.data.multi_step_datamodule import (
    MultiStepDataRandomizedModule,
    collate_fn_val,
)
from stormer.models.iterative_module import GlobalForecastIterativeModule
from stormer.utils.metrics import _lat_weight_broadcast


def _import_from_class_path(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    value = value.strip()
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def load_model_from_config(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Instantiate model from YAML config and load weights from Lightning checkpoint."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    net_cfg = model_cfg["net"]
    net_class_path = net_cfg["class_path"]
    net_init_args = dict(net_cfg["init_args"])

    # Prefer checkpoint-side model config when available (captures CLI overrides
    # such as in_img_size used during training).
    ckpt_model_cfg = Path(checkpoint_path).with_name("model_config.json")
    if ckpt_model_cfg.exists():
        with open(ckpt_model_cfg, "r") as f:
            saved_cfg = json.load(f)
        net_class_path = saved_cfg.get("model_class", net_class_path)
        for k, v in saved_cfg.get("model_args", {}).items():
            if v is not None:
                net_init_args[k] = v

    net_cls = _import_from_class_path(net_class_path)
    net = net_cls(**net_init_args)

    module_args = {k: v for k, v in model_cfg.items() if k != "net"}
    module = GlobalForecastIterativeModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        net=net,
        **module_args,
    )
    module = module.to(device)
    module.eval()
    return module


def select_test_dataset(datamodule, filter_years: Optional[Sequence[int]]):
    dataset = datamodule.data_test
    if not filter_years:
        return dataset, {
            "filter_years": [],
            "selected_samples": len(dataset),
        }

    selected_years = set(int(year) for year in filter_years)
    indices = []
    year_counts = {}

    for idx, path in enumerate(dataset.inp_file_paths):
        year = int(Path(path).stem.split("_")[0])
        if year in selected_years:
            indices.append(idx)
            year_counts[year] = year_counts.get(year, 0) + 1

    if not indices:
        raise ValueError(f"No test samples found for years: {sorted(selected_years)}")

    return Subset(dataset, indices), {
        "filter_years": sorted(selected_years),
        "selected_samples": len(indices),
        "year_counts": year_counts,
    }


def build_test_dataloader(datamodule, dataset):
    return DataLoader(
        dataset,
        batch_size=datamodule.hparams.val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=datamodule.hparams.num_workers,
        pin_memory=datamodule.hparams.pin_memory,
        collate_fn=collate_fn_val,
    )


def rollout_prediction(
    model: GlobalForecastIterativeModule,
    x: torch.Tensor,
    variables,
    lead_time: int,
    eval_mode: str,
) -> torch.Tensor:
    if eval_mode == "base_6h":
        intervals = [6]
    elif eval_mode == "ensemble_mean":
        intervals = [base for base in model.list_train_intervals if lead_time % base == 0]
    else:
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")

    if not intervals:
        raise ValueError(f"No valid rollout intervals found for lead time {lead_time}h.")

    preds = []
    for base_interval in intervals:
        steps = lead_time // base_interval
        preds.append(model.forward_validation(x, variables, base_interval, steps))

    if len(preds) == 1:
        return preds[0]
    return torch.stack(preds, dim=0).mean(0)


def evaluate_model(
    checkpoint_path: str,
    config_path: str,
    datamodule,
    test_dataset,
    lead_times: Sequence[int],
    eval_mode: str,
    wb2_mode: bool,
    device="cuda",
    num_batches: Optional[int] = None,
    inference_dtype="fp32",
    low_memory=False,
):
    """
    Evaluate a single model checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to YAML config file
        datamodule: Data module for loading transforms and coordinates
        test_dataset: Dataset or subset to evaluate
        lead_times: Lead times to evaluate in hours
        eval_mode: Rollout mode, either base_6h or ensemble_mean
        wb2_mode: Whether to aggregate RMSE in WB2 style
        device: Device to run evaluation on
        num_batches: Number of batches to evaluate, None means full selected dataset

    Returns:
        DataFrame with one aggregated RMSE per variable and lead time
    """
    model = load_model_from_config(checkpoint_path, config_path, device)

    model_dtype = torch.float32
    if str(device).startswith("cuda"):
        if inference_dtype == "bf16":
            model_dtype = torch.bfloat16
        elif inference_dtype == "fp16":
            model_dtype = torch.float16
    model = model.to(dtype=model_dtype)

    lat, lon = datamodule.get_lat_lon()
    model.set_lat_lon(lat, lon)
    model.set_transforms(*datamodule.get_transforms())
    model.set_base_intervals_and_lead_times(
        datamodule.hparams.list_train_intervals,
        lead_times,
    )

    test_dataloader = build_test_dataloader(datamodule, test_dataset)
    total_batches = len(test_dataloader)
    if num_batches is None or num_batches <= 0:
        max_batches = total_batches
    else:
        max_batches = min(num_batches, total_batches)

    print(
        f"Evaluating on {max_batches}/{total_batches} batches "
        f"(mode={eval_mode}, wb2_mode={wb2_mode})..."
    )

    sum_metric: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    variable_names = None

    with torch.inference_mode():
        for batch_idx, batch in enumerate(test_dataloader):
            if batch_idx >= max_batches:
                break

            x, dict_y, variables = batch
            if variable_names is None:
                variable_names = list(variables)
            x = x.to(device=device, dtype=model_dtype)

            for lead_time in lead_times:
                if str(device).startswith("cuda") and inference_dtype in ("bf16", "fp16"):
                    amp_dtype = torch.bfloat16 if inference_dtype == "bf16" else torch.float16
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
                else:
                    autocast_ctx = nullcontext()

                with autocast_ctx:
                    pred = rollout_prediction(model, x, variables, lead_time, eval_mode)

                gt = dict_y[lead_time].to(device=device, dtype=pred.dtype)
                pred_phys = model.reverse_inp_transform(pred.float())
                gt_phys = model.reverse_inp_transform(gt.float())

                error = (pred_phys - gt_phys) ** 2
                w_lat = _lat_weight_broadcast(model.lat, error).unsqueeze(1)
                spatial_mse = (error * w_lat).mean(dim=(-2, -1))

                if wb2_mode:
                    batch_metric = spatial_mse.sum(dim=0).double().cpu()
                else:
                    batch_metric = spatial_mse.sqrt().sum(dim=0).double().cpu()

                if lead_time not in sum_metric:
                    sum_metric[lead_time] = batch_metric
                    counts[lead_time] = spatial_mse.shape[0]
                else:
                    sum_metric[lead_time] += batch_metric
                    counts[lead_time] += spatial_mse.shape[0]

                if str(device).startswith("cuda") and low_memory:
                    del pred
                    del gt
                    del pred_phys
                    del gt_phys
                    del error
                    del spatial_mse
                    gc.collect()
                    torch.cuda.empty_cache()

            if (batch_idx + 1) % 5 == 0 or batch_idx + 1 == max_batches:
                print(f"  Processed {batch_idx + 1}/{max_batches} batches")

            if str(device).startswith("cuda") and low_memory:
                del x
                del dict_y
                gc.collect()
                torch.cuda.empty_cache()

    print("Evaluation complete!")
    if variable_names is None:
        raise RuntimeError("No batches were processed during evaluation.")

    rows = []
    aggregation = "wb2_rmse" if wb2_mode else "mean_sample_rmse"
    for lead_time in sorted(lead_times):
        for var_idx, var_name in enumerate(variable_names):
            mean_metric = sum_metric[lead_time][var_idx].item() / counts[lead_time]
            rmse = float(np.sqrt(mean_metric)) if wb2_mode else float(mean_metric)
            rows.append(
                {
                    "lead_time": lead_time,
                    "variable": var_name,
                    "rmse": rmse,
                    "num_samples": counts[lead_time],
                    "eval_mode": eval_mode,
                    "aggregation": aggregation,
                }
            )

    if str(device).startswith("cuda"):
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def compute_summary_statistics(results_df: pd.DataFrame) -> Dict:
    """Compute summary statistics from aggregated results."""
    summary = {}

    for lead_time in sorted(results_df["lead_time"].unique()):
        lt_data = results_df[results_df["lead_time"] == lead_time]
        summary[f"rmse_{lead_time}h"] = {
            "mean": float(lt_data["rmse"].mean()),
            "std": float(lt_data["rmse"].std()),
            "min": float(lt_data["rmse"].min()),
            "max": float(lt_data["rmse"].max()),
            "num_variables": int(len(lt_data)),
            "num_samples": int(lt_data["num_samples"].max()),
        }

    summary["overall"] = {
        "mean_rmse": float(results_df["rmse"].mean()),
        "std_rmse": float(results_df["rmse"].std()),
    }
    return summary


def statistical_significance_test(transformer_results: pd.DataFrame, candidate_results: pd.DataFrame, candidate_label: str):
    """
    Perform paired statistical tests comparing Transformer and a candidate model RMSE values
    on the same variables.

    In WB2 mode this compares aggregated per-variable RMSE values, not per-sample
    errors. The pairing is by variable name, which is more appropriate than an
    unpaired test because each variable is evaluated under both models.
    """
    from scipy import stats

    sig_tests = {}
    candidate_key = candidate_label.lower().replace(" ", "_")

    for lead_time in sorted(transformer_results["lead_time"].unique()):
        trans_lt = transformer_results[transformer_results["lead_time"] == lead_time][["variable", "rmse"]]
        candidate_lt = candidate_results[candidate_results["lead_time"] == lead_time][["variable", "rmse"]]

        paired = (
            trans_lt.rename(columns={"rmse": "transformer_rmse"})
            .merge(
                candidate_lt.rename(columns={"rmse": f"{candidate_key}_rmse"}),
                on="variable",
                how="inner",
            )
            .sort_values("variable")
        )

        if paired.empty:
            raise ValueError(f"No overlapping variables found for lead time {lead_time}h")

        trans_rmse = paired["transformer_rmse"]
        candidate_rmse = paired[f"{candidate_key}_rmse"]
        diff = trans_rmse - candidate_rmse

        t_stat, p_value = stats.ttest_rel(trans_rmse, candidate_rmse)

        sig_tests[f"{lead_time}h"] = {
            "test": "paired_t_test_across_variables",
            "p_value": float(p_value),
            "t_statistic": float(t_stat),
            "transformer_mean": float(trans_rmse.mean()),
            f"{candidate_key}_mean": float(candidate_rmse.mean()),
            "mean_paired_difference": float(diff.mean()),
            "std_paired_difference": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
            "difference_pct": float(100 * (trans_rmse.mean() - candidate_rmse.mean()) / trans_rmse.mean()),
            "num_variables": int(len(paired)),
            "transformer_better_count": int((diff < 0).sum()),
            f"{candidate_key}_better_count": int((diff > 0).sum()),
            "ties_count": int((diff == 0).sum()),
        }

    return sig_tests


def build_parser(experiment_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Transformer vs BiMamba")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./dataset/wb2_h5df_240x121",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--transformer_ckpt",
        type=str,
        default=f"{experiment_dir}/transformer/transformer_logs/checkpoints/best.ckpt",
        help="Path to Transformer checkpoint",
    )
    parser.add_argument(
        "--transformer_config",
        type=str,
        default="./configs/comparison/transformer.yaml",
        help="Path to Transformer config YAML",
    )
    parser.add_argument(
        "--candidate_ckpt",
        "--mamba_ckpt",
        dest="candidate_ckpt",
        type=str,
        default=f"{experiment_dir}/bimamba/bimamba_logs/checkpoints/best.ckpt",
        help="Path to candidate model checkpoint",
    )
    parser.add_argument(
        "--candidate_config",
        "--mamba_config",
        dest="candidate_config",
        type=str,
        default="./configs/comparison/bimamba.yaml",
        help="Path to candidate model config YAML",
    )
    parser.add_argument(
        "--candidate_label",
        type=str,
        default="BiMamba",
        help="Display label for the candidate model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=experiment_dir,
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of batches to evaluate; <=0 means full selected dataset",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers for evaluation",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=64,
        help="Validation batch size for evaluation dataloader",
    )
    parser.add_argument(
        "--inference_dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help="Inference precision for model rollout",
    )
    parser.add_argument(
        "--low_memory",
        action="store_true",
        help="Aggressively free tensors between rollouts to reduce peak VRAM",
    )
    parser.add_argument(
        "--lead_times",
        type=str,
        default="6,72,120",
        help="Comma-separated lead times in hours to evaluate",
    )
    parser.add_argument(
        "--filter_years",
        type=str,
        default="",
        help="Comma-separated input years to include from the local test split",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="base_6h",
        choices=["base_6h", "ensemble_mean"],
        help="Rollout mode for multi-step evaluation",
    )
    parser.add_argument(
        "--wb2_mode",
        action="store_true",
        help="Enable WB2-style comparison: filtered years + final aggregated RMSE output",
    )
    return parser


def main():
    experiment_dir = os.getenv("EXPERIMENT_DIR", "./experiments/full_comparison")
    parser = build_parser(experiment_dir)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lead_times = parse_int_list(args.lead_times)
    filter_years = parse_int_list(args.filter_years)

    print("=" * 60)
    print(f"Transformer vs {args.candidate_label} Evaluation")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Transformer checkpoint: {args.transformer_ckpt}")
    print(f"Transformer config: {args.transformer_config}")
    print(f"{args.candidate_label} checkpoint: {args.candidate_ckpt}")
    print(f"{args.candidate_label} config: {args.candidate_config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Lead times: {lead_times}")
    print(f"Filter years: {filter_years or 'all'}")
    print(f"Evaluation mode: {args.eval_mode}")
    print(f"WB2 mode: {args.wb2_mode}")
    print(f"Number of batches: {args.num_batches}")
    print(f"Inference dtype: {args.inference_dtype}")
    print(f"Low-memory mode: {args.low_memory}")
    print()

    for path_str, label in [
        (args.transformer_ckpt, "Transformer checkpoint"),
        (args.transformer_config, "Transformer config"),
        (args.candidate_ckpt, f"{args.candidate_label} checkpoint"),
        (args.candidate_config, f"{args.candidate_label} config"),
    ]:
        if not Path(path_str).exists():
            print(f"ERROR: {label} not found: {path_str}")
            sys.exit(1)

    variables = [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
    ] + [f"geopotential_{p}" for p in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]] \
      + [f"u_component_of_wind_{p}" for p in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]] \
      + [f"v_component_of_wind_{p}" for p in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]] \
      + [f"temperature_{p}" for p in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]] \
      + [f"specific_humidity_{p}" for p in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]]

    print("Loading datamodule...")
    datamodule = MultiStepDataRandomizedModule(
        root_dir=args.data_root,
        variables=variables,
        list_train_intervals=[6, 12, 24],
        steps=20,
        val_lead_times=lead_times,
        batch_size=1,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup("test")
    print("Datamodule loaded successfully")

    test_dataset, dataset_metadata = select_test_dataset(datamodule, filter_years)
    print(f"Selected {dataset_metadata['selected_samples']} test samples")
    if dataset_metadata.get("year_counts"):
        print(f"Year counts: {dataset_metadata['year_counts']}")
    print()

    metadata = {
        "lead_times": lead_times,
        "filter_years": dataset_metadata["filter_years"],
        "selected_samples": dataset_metadata["selected_samples"],
        "eval_mode": args.eval_mode,
        "wb2_mode": args.wb2_mode,
        "num_batches_requested": args.num_batches,
        "val_batch_size": args.val_batch_size,
        "inference_dtype": args.inference_dtype,
        "candidate_label": args.candidate_label,
    }
    metadata_file = output_dir / "evaluation_config.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Evaluation config saved to: {metadata_file}")
    print()

    print("[1/2] Evaluating Transformer...")
    print("-" * 60)
    transformer_results = evaluate_model(
        args.transformer_ckpt,
        args.transformer_config,
        datamodule,
        test_dataset=test_dataset,
        lead_times=lead_times,
        eval_mode=args.eval_mode,
        wb2_mode=args.wb2_mode,
        device=args.device,
        num_batches=args.num_batches,
        inference_dtype=args.inference_dtype,
        low_memory=args.low_memory,
    )
    transformer_results["model"] = "Transformer"
    print()

    print(f"[2/2] Evaluating {args.candidate_label}...")
    print("-" * 60)
    candidate_results = evaluate_model(
        args.candidate_ckpt,
        args.candidate_config,
        datamodule,
        test_dataset=test_dataset,
        lead_times=lead_times,
        eval_mode=args.eval_mode,
        wb2_mode=args.wb2_mode,
        device=args.device,
        num_batches=args.num_batches,
        inference_dtype=args.inference_dtype,
        low_memory=args.low_memory,
    )
    candidate_results["model"] = args.candidate_label
    print()

    all_results = pd.concat([transformer_results, candidate_results], ignore_index=True)

    results_csv = output_dir / "evaluation_results.csv"
    all_results.to_csv(results_csv, index=False)
    print(f"✓ Results saved to: {results_csv}")

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    summary = {
        "metadata": metadata,
        "transformer": compute_summary_statistics(transformer_results),
        args.candidate_label.lower().replace(" ", "_"): compute_summary_statistics(candidate_results),
    }

    summary_file = output_dir / "evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_file}")

    print("\nMean RMSE by Lead Time:")
    print("-" * 60)
    print(f"{'Lead Time':<15} {'Transformer':<15} {args.candidate_label:<15} {'Diff %':<10}")
    print("-" * 60)

    for lead_time in sorted(transformer_results["lead_time"].unique()):
        trans_mean = transformer_results[transformer_results["lead_time"] == lead_time]["rmse"].mean()
        candidate_mean = candidate_results[candidate_results["lead_time"] == lead_time]["rmse"].mean()
        diff_pct = 100 * (trans_mean - candidate_mean) / trans_mean
        print(
            f"{lead_time:>3}h {'':<10} {trans_mean:>8.4f} "
            f"{'':<6} {candidate_mean:>8.4f} {'':<6} {diff_pct:>+7.2f}%"
        )

    print("\n" + "=" * 60)
    print("Statistical Significance Tests")
    print("=" * 60)

    try:
        sig_tests = statistical_significance_test(transformer_results, candidate_results, args.candidate_label)

        sig_file = output_dir / "significance_tests.json"
        with open(sig_file, "w") as f:
            json.dump(sig_tests, f, indent=2)
        print(f"✓ Significance tests saved to: {sig_file}")

        print(f"\n{'Lead Time':<10} {'p-value':<12} {'Significant?':<15} {'Winner Count (T/C)'}")
        print("-" * 72)
        candidate_count_key = f"{args.candidate_label.lower().replace(' ', '_')}_better_count"
        for key, value in sig_tests.items():
            significant = "Yes (p<0.05)" if value["p_value"] < 0.05 else "No"
            print(
                f"{key:<10} {value['p_value']:<12.4f} {significant:<15} "
                f"{value['transformer_better_count']}/{value[candidate_count_key]}"
            )
    except Exception as e:
        print(f"Could not perform significance tests: {e}")
        print("(scipy may not be installed)")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("  - Generate visualizations: python scripts/comparison/transformer_vs_mamba/visualize_with_baselines.py")


if __name__ == "__main__":
    main()
