import torch

def quantile_huber_loss(pred_q: torch.Tensor, y: torch.Tensor, quantiles, delta: float = 1.0, reduction="mean"):
    if y.ndim == 1:
        y = y.view(-1, 1)
    qs = torch.tensor(list(quantiles), device=pred_q.device, dtype=pred_q.dtype).view(1, -1)
    e = y - pred_q
    abs_e = torch.abs(e)
    huber = torch.where(abs_e <= delta, 0.5 * (e ** 2) / delta, abs_e - 0.5 * delta)
    qloss = torch.maximum(qs * huber, (qs - 1.0) * huber)
    if reduction == "mean":
        return qloss.mean()
    if reduction == "sum":
        return qloss.sum()
    return qloss
