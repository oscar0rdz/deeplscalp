import torch


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: list[float], reduction: str = "mean") -> torch.Tensor:
    """
    pred: (B, Q)
    target: (B,)
    returns:
      - "none": (B,)
      - "mean": scalar
      - "sum" : scalar
    """
    y = target.unsqueeze(1)  # (B,1)
    losses = []
    for i, q in enumerate(quantiles):
        e = y - pred[:, i:i+1]
        # pinball
        losses.append(torch.maximum((q - 1) * e, q * e))
    # (B,Q) -> (B,)
    loss_per_sample = torch.sum(torch.cat(losses, dim=1), dim=1)

    if reduction == "none":
        return loss_per_sample
    if reduction == "sum":
        return loss_per_sample.sum()
    return loss_per_sample.mean()
