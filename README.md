# Mamba-Stormer

Code for reproducing the experiments described in:

**Mamba-Stormer v1.0: A Bidirectional Vision Mamba Backbone for Accurate and Scalable Global Weather Forecasting**

The repository contains code to reproduce the comparison between:

- `Transformer-Stormer`
- `Mamba-Stormer (BiMamba)`

at the WeatherBench 2 `240x121` ERA5 resolution, including:

- WeatherBench 2 preprocessing
- one-step pretraining
- multi-step finetuning
- evaluation and significance testing
- throughput benchmarking
- figure regeneration from evaluation and benchmark outputs

## Repository Contents

- core Transformer and BiMamba model implementations
- training entrypoint (`train.py`)
- WeatherBench 2 preprocessing pipeline under `stormer/data_preprocessing/`
- comparison configs under `configs/comparison/`
- evaluation and benchmark scripts under `scripts/comparison/transformer_vs_mamba/`
- figure generation scripts under `scripts/`

## Environment Setup

The supported path is a CUDA-enabled Python environment with `mamba-ssm`, `causal-conv1d`, and `xformers`.

To create the environment from scratch:

```bash
bash scripts/setup/setup_env.sh
source stormer_mamba_env/bin/activate
```

Notes:

- The setup script assumes `nvcc` is available.
- By default it installs `torch==2.8.0` with CUDA 12.8 wheels.
- `mamba-ssm` and `causal-conv1d` are built in the active environment with `--no-build-isolation`.

If you manage the environment manually, install:

```bash
pip install -r requirements.txt
pip install xformers --no-deps
pip install mamba-ssm==2.2.6.post3 causal-conv1d --no-build-isolation
pip install -e .
```

## Dataset Preparation

The code expects a processed WeatherBench 2 HDF5 dataset at:

```bash
./dataset/wb2_h5df_240x121
```

You can override that location with `DATA_ROOT`.

### 1. Download ERA5 from WeatherBench 2

```bash
python stormer/data_preprocessing/download_wb2.py \
  --file "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr" \
  --save_dir ./dataset/wb2_raw_240x121
```

### 2. Build train / val / test splits

```bash
python stormer/data_preprocessing/process_one_step_data.py \
  --root_dir ./dataset/wb2_raw_240x121 \
  --save_dir ./dataset/wb2_h5df_240x121 \
  --start_year 1979 \
  --end_year 2019 \
  --split train \
  --chunk_size 10

python stormer/data_preprocessing/process_one_step_data.py \
  --root_dir ./dataset/wb2_raw_240x121 \
  --save_dir ./dataset/wb2_h5df_240x121 \
  --start_year 2019 \
  --end_year 2020 \
  --split val \
  --chunk_size 10

python stormer/data_preprocessing/process_one_step_data.py \
  --root_dir ./dataset/wb2_raw_240x121 \
  --save_dir ./dataset/wb2_h5df_240x121 \
  --start_year 2020 \
  --end_year 2024 \
  --split test \
  --chunk_size 10
```

### 3. Compute normalization constants

```bash
for lead_time in -1 6 12 24; do
  python stormer/data_preprocessing/compute_normalization.py \
    --root_dir ./dataset/wb2_raw_240x121 \
    --save_dir ./dataset/wb2_h5df_240x121 \
    --start_year 1979 \
    --end_year 2018 \
    --chunk_size 100 \
    --lead_time "${lead_time}" \
    --data_frequency 6
done
```

`normalization_constants/` is included because the figure scripts use climatological standard deviations for normalization.

## Training

Set the environment variables first:

```bash
export DATA_ROOT=./dataset/wb2_h5df_240x121
export EXPERIMENT_ROOT=./experiments/full_comparison
```

### Transformer pretraining

```bash
python train.py \
  --config configs/comparison/transformer.yaml \
  --data.batch_size 4 \
  --data.num_workers 12
```

### BiMamba pretraining

```bash
python train.py \
  --config configs/comparison/bimamba.yaml \
  --data.batch_size 4 \
  --data.num_workers 12
```

### Transformer finetuning

```bash
python train.py \
  --config configs/comparison/finetune_transformer.yaml \
  --model.pretrained_path ./experiments/full_comparison/transformer/transformer_logs/checkpoints/last.ckpt \
  --data.batch_size 2 \
  --data.num_workers 12
```

### BiMamba finetuning

```bash
python train.py \
  --config configs/comparison/finetune_bimamba.yaml \
  --model.pretrained_path ./experiments/full_comparison/bimamba/bimamba_logs/checkpoints/last.ckpt \
  --data.batch_size 2 \
  --data.num_workers 12
```

### Multi-GPU note

For SLURM launches, keep `trainer.devices: 1` inside YAML and override at the CLI:

```bash
python train.py \
  --config configs/comparison/bimamba.yaml \
  --trainer.devices 4 \
  --trainer.strategy ddp
```

This follows the required Lightning-SLURM device and strategy override pattern for multi-GPU launches.

## Evaluation

The evaluation command below uses:

- full processed test split (`2020-2023`)
- `ensemble_mean` rollout mode
- WB2-style RMSE aggregation
- `bf16` inference is optional here for lower-memory GPUs; it is not part of the throughput benchmark protocol

```bash
python scripts/comparison/transformer_vs_mamba/evaluate_comparison.py \
  --data_root "$DATA_ROOT" \
  --transformer_ckpt "$EXPERIMENT_ROOT/finetune_transformer/finetune_transformer_logs/checkpoints/best.ckpt" \
  --transformer_config ./configs/comparison/transformer.yaml \
  --candidate_ckpt "$EXPERIMENT_ROOT/finetune_bimamba/finetune_bimamba_logs/checkpoints/best.ckpt" \
  --candidate_config ./configs/comparison/bimamba.yaml \
  --candidate_label BiMamba \
  --output_dir ./artifacts/evaluation \
  --num_batches 0 \
  --val_batch_size 4 \
  --eval_mode ensemble_mean \
  --inference_dtype bf16 \
  --device cuda \
  --low_memory \
  --wb2_mode
```

This produces:

- `artifacts/evaluation/evaluation_results.csv`
- `artifacts/evaluation/evaluation_summary.json`
- `artifacts/evaluation/significance_tests.json`

Optional visualization:

```bash
python scripts/comparison/transformer_vs_mamba/visualize_comparison.py \
  --results_dir ./artifacts/evaluation \
  --candidate_label BiMamba \
  --candidate_subdir finetune_bimamba \
  --candidate_log_name finetune_bimamba_logs
```

## Throughput Benchmarking

Single-resolution benchmark at the training resolution used in the comparison:

```bash
python scripts/comparison/transformer_vs_mamba/benchmark_comparison.py \
  --img_height 121 \
  --img_width 240 \
  --batch_size 4 \
  --output_dir ./artifacts/benchmarks/121x240
```

Multi-resolution sweep:

```bash
bash scripts/comparison/transformer_vs_mamba/run_benchmark_sweep.sh \
  --output_root ./artifacts/benchmarks/transformer_vs_bimamba \
  --resolutions 32x64 64x128 121x240 128x256 256x256 256x512 512x256 512x512 \
  --batch_size 4
```

## Figures

```bash
python scripts/comparison/transformer_vs_mamba/visualize_comparison.py \
  --results_dir ./artifacts/evaluation \
  --candidate_label BiMamba \
  --candidate_subdir finetune_bimamba \
  --candidate_log_name finetune_bimamba_logs

python scripts/gen_speedup_figure.py \
  --benchmark_root ./artifacts/benchmarks/transformer_vs_bimamba \
  --output_dir ./artifacts/figures
```

These figures are generated from the evaluation and benchmark artifacts produced by the commands above.

Generated figures are written to:

```bash
./artifacts/evaluation/plots
./artifacts/figures
```

## Reproducibility Defaults

The configs use the following defaults:

- seed: `42`
- resolution: `240x121`
- Transformer depth: `24`
- BiMamba depth: `14`
- BiMamba `d_state=32`, `expand=2`, `d_conv=4`
- BiMamba learning rate cap: `2e-4`
- gradient clipping: `0.5` for BiMamba and finetuning
