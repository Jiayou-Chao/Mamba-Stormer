"""Utility functions for auto-resuming training."""
from pathlib import Path


def get_resume_checkpoint(output_dir):
    """
    Auto-detect checkpoint to resume from.

    Args:
        output_dir: Training output directory

    Returns:
        Path to checkpoint if found, None otherwise
    """
    ckpt_path = Path(output_dir) / "checkpoints" / "last.ckpt"
    if ckpt_path.exists():
        print(f"✓ Found checkpoint, resuming from: {ckpt_path}")
        return str(ckpt_path)
    print("✓ No checkpoint found, starting fresh training")
    return None
