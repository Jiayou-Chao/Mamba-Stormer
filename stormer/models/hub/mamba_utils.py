import torch

try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
    MAMBA_ERROR = None
except Exception as exc:
    MAMBA_AVAILABLE = False
    MAMBA_ERROR = f"Failed to import mamba-ssm: {exc}"
    Mamba = None


def build_mamba_module(hidden_size, d_state=16, d_conv=4, expand=2, **mamba_kwargs):
    if not MAMBA_AVAILABLE:
        raise ImportError(MAMBA_ERROR)

    try:
        return Mamba(
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **mamba_kwargs,
        )
    except Exception as exc:
        error_msg = str(exc)
        if "undefined symbol" in error_msg and "selective_scan_cuda" in error_msg:
            raise RuntimeError(
                "Mamba CUDA compatibility error detected.\n\n"
                "This usually means mamba-ssm was compiled against a different "
                "PyTorch/CUDA stack than the active environment.\n\n"
                f"PyTorch: {torch.__version__}\n"
                f"CUDA available: {torch.cuda.is_available()}\n"
                f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}\n\n"
                "Reinstall mamba-ssm and causal-conv1d in the active environment."
            ) from exc
        raise RuntimeError(f"Unexpected error during Mamba initialization: {error_msg}") from exc
