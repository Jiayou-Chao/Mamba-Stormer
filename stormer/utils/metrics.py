import numpy as np
import torch


def _lat_weight_broadcast(lat: np.ndarray, tensor: torch.Tensor) -> torch.Tensor:
    """
    Build a broadcastable latitude weight tensor that matches the spatial
    dimensions of `tensor` (expected layout: [B, V, H, W] or [B, H, W]).

    The function automatically detects whether the provided `lat` vector
    length corresponds to the height (H) or width (W) dimension and
    returns weights with shape (1, H, 1) or (1, 1, W) respectively.
    """
    # Accept tensors of shape [B, V, H, W] or [B, H, W]
    if tensor.dim() not in (3, 4):
        raise ValueError(f"Unexpected tensor rank {tensor.dim()}; expected 3 or 4 dims.")

    # Identify H and W indices
    if tensor.dim() == 4:
        H, W = tensor.shape[-2], tensor.shape[-1]
    else:
        H, W = tensor.shape[-2], tensor.shape[-1]

    # Normalize cosine weights of latitude
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()

    if len(lat) == H:
        # weight along height dimension
        w_lat = torch.from_numpy(w).to(dtype=tensor.dtype, device=tensor.device).view(1, H, 1)
    elif len(lat) == W:
        # weight along width dimension
        w_lat = torch.from_numpy(w).to(dtype=tensor.dtype, device=tensor.device).view(1, 1, W)
    else:
        raise ValueError(
            f"Latitude vector length {len(lat)} does not match spatial dims H={H} or W={W}."
        )
    return w_lat


def lat_weighted_mse(pred, y, vars, lat, weighted=False, weight_dict=None):
    """Latitude weighted mean squared error

    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [B, V, H, W]
    # latitude weights (auto-detect axis)
    w_lat = _lat_weight_broadcast(lat, error)  # (1, H, 1) or (1, 1, W)

    loss_dict = {}
    for i, var in enumerate(vars):
        loss_dict[f"w_mse_{var}"] = (error[:, i] * w_lat).mean()
        
    if weighted:
        weights = torch.Tensor([weight_dict[var] for var in vars]).to(device=error.device).view(1, -1, 1, 1)
        weights = weights / weights.sum()
    else:
        weights = torch.ones(len(vars)).to(device=error.device).view(1, -1, 1, 1) / len(vars)
    
    # expand w_lat along variable dimension for aggregation
    loss_dict["w_mse_aggregate"] = (error * w_lat.unsqueeze(1) * weights).sum(dim=1).mean()

    return loss_dict


def lat_weighted_mse_val(pred, y, transform, vars, lat, log_postfix, weighted=False, weight_dict=None):
    """Latitude weighted mean squared error
    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [B, V, H, W]
    w_lat = _lat_weight_broadcast(lat, error)  # (1, H, 1) or (1, 1, W)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_mse_{var}_{log_postfix}"] = (error[:, i] * w_lat).mean()
            
    if weighted:
        weights = torch.Tensor([weight_dict[var] for var in vars]).to(device=error.device).view(1, -1, 1, 1)
        weights = weights / weights.sum()
    else:
        weights = torch.ones(len(vars)).to(device=error.device).view(1, -1, 1, 1) / len(vars)

    loss_dict[f"w_mse_aggregate_{log_postfix}"] = (error * w_lat.unsqueeze(1) * weights).sum(dim=1).mean()

    return loss_dict


def lat_weighted_rmse(pred, y, transform, vars, lat, log_postfix, weighted=False, weight_dict=None):
    """Latitude weighted root mean squared error

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    pred = transform(pred)
    y = transform(y)

    error = (pred - y) ** 2  # [B, V, H, W]
    w_lat = _lat_weight_broadcast(lat, error)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_rmse_{var}_{log_postfix}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
            )

    # loss_dict[f"w_rmse_aggregate_{log_postfix}"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict

def lat_weighted_crps(pred: torch.Tensor, y: torch.Tensor, transform, vars, lat, log_postfix, weighted=False, weight_dict=None):
    assert len(pred.shape) == len(y.shape) + 1
    # pred: [B, N, V, H, W] because there are N ensemble members
    # y: [B, V, H, W]
    pred = transform(pred)
    y = transform(y)
    
    H, W = pred.shape[-2], pred.shape[-1]
    N = pred.shape[1]
    # Prepare 2 broadcastable variants for convenience
    w_lat_H = _lat_weight_broadcast(lat, pred[:, 0])  # [B, H, W] -> (1,H,1) or (1,1,W)
    # For CRPS we need shapes for [B,N,H,W] broadcasting:
    if w_lat_H.shape == (1, H, 1):
        w_for_err = w_lat_H.view(1, 1, H, 1)
        w_for_spread = w_lat_H.view(1, 1, 1, H, 1)
    else:  # (1,1,W)
        w_for_err = w_lat_H.view(1, 1, 1, W)
        w_for_spread = w_lat_H.view(1, 1, 1, 1, W)

    def crps_var(pred_var: torch.Tensor, y_var: torch.Tensor):
        # pred_var: [B, N, H, W]
        # y: [B, H, W]
        # first term: prediction errors
        with torch.no_grad():
            error_term = torch.abs(pred_var - y_var.unsqueeze(1))  # [B, N, H, W]
            error_term = error_term * w_for_err  # [B, N, H, W]
            error_term = torch.mean(error_term)
        
        # second term: ensemble spread
        with torch.no_grad():
            spread_term = torch.abs(pred_var.unsqueeze(2) - pred_var.unsqueeze(1))  # [B, N, N, H, W]
            spread_term = spread_term * w_for_spread  # [B, N, N, H, W]
            spread_term = spread_term.mean(dim=(-2, -1)) # [B, N, N]
            spread_term = spread_term.sum(dim=(1, 2)) / (2 * N * (N - 1)) # [B]
            spread_term = spread_term.mean()
            
        return error_term - spread_term
    
    loss_dict = {}
    for i, var in enumerate(vars):
        loss_dict[f"w_crps_{var}_{log_postfix}"] = crps_var(pred[:, :, i], y[:, i])
        
    return loss_dict

def lat_weighted_spread_skill_ratio(pred: torch.Tensor, y: torch.Tensor, transform, vars, lat, log_postfix, weighted=False, weight_dict=None):
    assert len(pred.shape) == len(y.shape) + 1
    # pred: [B, N, V, H, W] because there are N ensemble members
    # y: [B, V, H, W]
    rmse_dict = lat_weighted_rmse(pred.mean(dim=1), y, transform, vars, lat, log_postfix, weighted, weight_dict)
    
    pred = transform(pred)
    y = transform(y)
    
    H, W = pred.shape[-2], pred.shape[-1]
    w_lat_b = _lat_weight_broadcast(lat, pred[:, 0])  # (1,H,1) or (1,1,W)
    var = torch.var(pred, dim=1)  # [B, V, H, W]
    var = var * w_lat_b.unsqueeze(1)  # [B, V, H, W]
    spread = var.mean(dim=(-2, -1)).sqrt().mean(dim=0) # [V]
    
    loss_dict = {}
    for i, var in enumerate(vars):
        loss_dict[f"w_ssr_{var}_{log_postfix}"] = spread[i] / rmse_dict[f"w_rmse_{var}_{log_postfix}"]
        
    return loss_dict

def lat_weighted_acc(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)

    w_lat = _lat_weight_broadcast(lat, pred)  # [1,H,1] or [1,1,W]

    # clim = torch.mean(y, dim=(0, 1), keepdim=True)
    clim = clim.to(device=y.device).unsqueeze(0)
    pred = pred - clim
    y = y - clim
    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_prime = pred[:, i] - torch.mean(pred[:, i])
            y_prime = y[:, i] - torch.mean(y[:, i])
            loss_dict[f"acc_{var}_{log_postfix}"] = torch.sum(w_lat * pred_prime * y_prime) / torch.sqrt(
                torch.sum(w_lat * pred_prime**2) * torch.sum(w_lat * y_prime**2)
            )

    # loss_dict[f"acc_aggregate_{log_postfix}"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict
