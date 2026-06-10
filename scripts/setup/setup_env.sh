#!/bin/bash

set -euo pipefail

ENV_DIR="${1:-stormer_mamba_env}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.8.0}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"

if [ ! -f "train.py" ]; then
    echo "Run this script from the repository root."
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc is required to build mamba-ssm and causal-conv1d."
    exit 1
fi

export TORCH_CUDA_ARCH_LIST

python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

pip install --upgrade pip
pip install ninja wheel packaging
pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "$TORCH_INDEX_URL"
pip install -r requirements.txt
pip install xformers --no-deps
pip install "mamba-ssm==2.2.6.post3" causal-conv1d --no-build-isolation
pip install -e .

python -c "
import torch
from stormer.models.hub import BiMambaStormer, Stormer
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Transformer class: {Stormer.__name__}')
print(f'BiMamba class: {BiMambaStormer.__name__}')
"

echo "Environment ready: source ${ENV_DIR}/bin/activate"
