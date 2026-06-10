"""Utility functions for loading models from checkpoints."""
import json
import importlib
from pathlib import Path
import torch


def load_model_from_checkpoint(checkpoint_path):
    """
    Load model from checkpoint using saved model_config.json.

    Args:
        checkpoint_path: Path to Lightning checkpoint (.ckpt file)

    Returns:
        Model in eval mode on CUDA
    """
    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path.parent / 'model_config.json'

    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Import model class
    module_path, class_name = config['model_class'].rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    # Instantiate model
    model_args = {k: v for k, v in config['model_args'].items() if v is not None}
    model = model_class(**model_args)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    state_dict = {k.replace('net.', ''): v for k, v in checkpoint['state_dict'].items()
                  if k.startswith('net.')}
    model.load_state_dict(state_dict, strict=False)

    return model.cuda().eval()
