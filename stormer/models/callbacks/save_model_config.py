"""Callback to save model configuration alongside checkpoints for easy evaluation loading."""
import json
from pathlib import Path
from lightning.pytorch.callbacks import Callback


class SaveModelConfig(Callback):
    """Save model configuration as JSON when checkpoint is saved."""

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Save model config when checkpoint is saved."""
        if not hasattr(pl_module, 'net'):
            return

        model = pl_module.net
        config = {
            'model_class': f"{model.__class__.__module__}.{model.__class__.__name__}",
            'model_args': {
                'in_img_size': getattr(model, 'original_in_img_size', getattr(model, 'in_img_size', None)),
                'patch_size': getattr(model, 'patch_size', None),
                'hidden_size': getattr(model, 'hidden_size', None),
                'depth': getattr(model, 'depth', None),
                'num_heads': getattr(model, 'num_heads', None),
                'mlp_ratio': getattr(model, 'mlp_ratio', None),
                'variables': getattr(model, 'variables', None),
            }
        }

        # Add model-specific args
        if hasattr(model, 'd_state'):
            config['model_args']['d_state'] = model.d_state
        if hasattr(model, 'd_conv'):
            config['model_args']['d_conv'] = model.d_conv
        if hasattr(model, 'expand'):
            config['model_args']['expand'] = model.expand

        # Save to checkpoint directory
        ckpt_dir = Path(trainer.checkpoint_callback.dirpath)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        config_path = ckpt_dir / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
