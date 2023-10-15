# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn.functional as F


def scaledlp_loss(input: torch.Tensor, target: torch.Tensor, p: int = 2, reduction: str = "mean"):
    B = input.size(0)
    diff_norms = torch.norm(input.reshape(B, -1) - target.reshape(B, -1), p, 1)
    target_norms = torch.norm(target.reshape(B, -1), p, 1)
    val = diff_norms / target_norms
    if reduction == "mean":
        return torch.mean(val)
    elif reduction == "sum":
        return torch.sum(val)
    elif reduction == "none":
        return val
    else:
        raise NotImplementedError(reduction)


def custommse_loss(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
    loss = F.mse_loss(input, target, reduction="none")
    # avg across space
    reduced_loss = torch.mean(loss, dim=tuple(range(3, loss.ndim)))
    # sum across time + fields
    reduced_loss = reduced_loss.sum(dim=(1, 2))
    # reduce along batch
    if reduction == "mean":
        return torch.mean(reduced_loss)
    elif reduction == "sum":
        return torch.sum(reduced_loss)
    elif reduction == "none":
        return reduced_loss
    else:
        raise NotImplementedError(reduction)


def pearson_correlation(input: torch.Tensor, target: torch.Tensor, reduce_batch: bool = False):
    B = input.size(0)
    T = input.size(1)
    input = input.reshape(B, T, -1)
    target = target.reshape(B, T, -1)
    input_mean = torch.mean(input, dim=(2), keepdim=True)
    target_mean = torch.mean(target, dim=(2), keepdim=True)
    # Unbiased since we use unbiased estimates in covariance
    input_std = torch.std(input, dim=(2), unbiased=False)
    target_std = torch.std(target, dim=(2), unbiased=False)

    corr = torch.mean((input - input_mean) * (target - target_mean), dim=2) / (input_std * target_std).clamp(
        min=torch.finfo(torch.float32).tiny
    )  # shape (B, T)
    if reduce_batch:
        corr = torch.mean(corr, dim=0)
    return corr


class ScaledLpLoss(torch.nn.Module):
    """Scaled Lp loss for PDEs.

    Args:
        p (int, optional): p in Lp norm. Defaults to 2.
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, p: int = 2, reduction: str = "mean") -> None:
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return scaledlp_loss(input, target, p=self.p, reduction=self.reduction)


class CustomMSELoss(torch.nn.Module):
    """Custom MSE loss for PDEs.

    MSE but summed over time and fields, then averaged over space and batch.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return custommse_loss(input, target, reduction=self.reduction)


class PearsonCorrelationScore(torch.nn.Module):
    """Pearson Correlation Score for PDEs."""

    def __init__(self, channel: int = None, reduce_batch: bool = False) -> None:
        super().__init__()
        self.channel = channel
        self.reduce_batch = reduce_batch

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.channel is not None:
            input = input[:, :, self.channel]
            target = target[:, :, self.channel]
        return pearson_correlation(input, target, reduce_batch=self.reduce_batch)
