"""SIGReg: Sliced Isotropic Gaussian Regularization for collapse prevention.

Implements the Cramér-Wold approach from LeWorldModel: test high-dimensional
Gaussianity by checking that all 1D random projections are Gaussian, using the
Epps-Pulley (EP) test statistic.

Extension: per-subspace SIGReg for multi-modal disentanglement, with
adaptive lambda scaling and cross-covariance penalties.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def epps_pulley_statistic(x: Tensor) -> Tensor:
    """Compute the Epps-Pulley test statistic for 1D Gaussianity.

    The EP statistic measures departure from normality.  Lower values
    indicate more Gaussian distributions.  The statistic is fully
    differentiable (no sorting or discrete operations).

    For a sample x_1 ... x_n with sample mean mu and variance sigma^2:

        EP = (2/n) sum_{i<j} exp(-0.5 * (x_i - x_j)^2 / (2*sigma^2))
             - sqrt(2) * mean_i exp(-0.5 * (x_i - mu)^2 / (sigma^2 + sigma^2))
             + sqrt(1 / (1 + 2*sigma^2 / sigma^2))

    Using bandwidth s^2 = sigma^2 (sample variance), which is the standard
    choice from the EP paper.

    Args:
        x: 1D tensor of shape (N,) with N >= 2.

    Returns:
        Scalar EP statistic (lower = more Gaussian).
    """
    n = x.shape[0]
    if n < 2:
        return x.new_tensor(0.0)

    mu = x.mean()
    sigma_sq = x.var(unbiased=True).clamp(min=1e-8)

    # Bandwidth s^2 = sigma^2 (sample variance)
    s_sq = sigma_sq

    # Pairwise term: (2/n^2) sum_{i,j} exp(-0.5 * (x_i - x_j)^2 / (2*s^2))
    # Using the full matrix (includes i=j and double-counts), then adjusting.
    diff = x.unsqueeze(0) - x.unsqueeze(1)  # (N, N)
    pairwise = torch.exp(-0.25 * diff.pow(2) / s_sq)  # -0.5 / (2*s^2) = -0.25/s^2
    # sum_{i<j} = (sum_all - sum_diag) / 2
    pair_sum = (pairwise.sum() - pairwise.diag().sum()) / 2.0
    term1 = (2.0 / (n * (n - 1))) * pair_sum

    # Marginal term: mean_i exp(-0.5 * (x_i - mu)^2 / (s^2 + sigma^2))
    centered = x - mu
    marginal = torch.exp(-0.5 * centered.pow(2) / (s_sq + sigma_sq))
    term2 = math.sqrt(2.0) * marginal.mean()

    # Normalizing constant
    term3 = math.sqrt(1.0 / 3.0)  # sqrt(s^2 / (2*s^2 + sigma^2)) = sqrt(1/3) when s^2=sigma^2

    return term1 - term2 + term3


def cramer_wold_sigreg(
    z: Tensor,
    n_projections: int = 1024,
) -> tuple[Tensor, Tensor]:
    """Compute the Cramér-Wold SIGReg loss via random projections.

    Projects D-dimensional embeddings onto K random directions and
    computes the mean EP statistic across all projections.  By the
    Cramér-Wold theorem, low mean EP implies approximate Gaussianity
    in the original high-dimensional space.

    Args:
        z: Embeddings of shape (B, D) with B >= 2.
        n_projections: Number of random 1D projections (K).

    Returns:
        Tuple of (sigreg_loss, mean_ep_statistic).
        The loss is the mean EP across projections.
        The mean_ep_statistic is the same value, detached, for adaptive lambda.
    """
    batch_size, dim = z.shape
    if batch_size < 2:
        zero = z.new_tensor(0.0)
        return zero, zero.detach()

    # Sample random directions from unit sphere
    directions = torch.randn(dim, n_projections, device=z.device, dtype=z.dtype)
    directions = F.normalize(directions, dim=0)

    # Project: (B, D) @ (D, K) -> (B, K)
    projections = z @ directions

    # Compute EP statistic for each 1D slice
    # Vectorized: compute all K slices at once for efficiency
    ep_stats = _batch_epps_pulley(projections)  # (K,)

    loss = ep_stats.mean()
    return loss, loss.detach()


def _batch_epps_pulley(projections: Tensor) -> Tensor:
    """Vectorized EP statistic across K projections.

    Args:
        projections: (B, K) tensor of 1D projections.

    Returns:
        (K,) tensor of EP statistics, one per projection.
    """
    n, k = projections.shape
    if n < 2:
        return projections.new_zeros(k)

    mu = projections.mean(dim=0)  # (K,)
    sigma_sq = projections.var(dim=0, unbiased=True).clamp(min=1e-8)  # (K,)
    s_sq = sigma_sq

    # Pairwise term: for each projection k, compute sum_{i<j} exp(...)
    # diff[i,j,k] = projections[i,k] - projections[j,k]
    # shape: (B, 1, K) - (1, B, K) = (B, B, K)
    diff = projections.unsqueeze(1) - projections.unsqueeze(0)  # (B, B, K)
    pairwise = torch.exp(-0.25 * diff.pow(2) / s_sq.unsqueeze(0).unsqueeze(0))  # (B, B, K)

    # sum_{i<j} via (sum_all - diag) / 2
    # diagonal(dim1=0, dim2=1) on (B, B, K) returns (K, B), so sum over dim=1
    diag_sum = pairwise.diagonal(dim1=0, dim2=1).sum(dim=1)  # (K,)
    pair_sum = (pairwise.sum(dim=(0, 1)) - diag_sum) / 2.0  # (K,)
    term1 = (2.0 / (n * (n - 1))) * pair_sum

    # Marginal term
    centered = projections - mu.unsqueeze(0)  # (B, K)
    marginal = torch.exp(-0.5 * centered.pow(2) / (s_sq + sigma_sq).unsqueeze(0))  # (B, K)
    term2 = math.sqrt(2.0) * marginal.mean(dim=0)  # (K,)

    # Normalizing constant (same for all K when s^2 = sigma^2)
    term3 = math.sqrt(1.0 / 3.0)

    return term1 - term2 + term3


def cross_covariance_loss(z1: Tensor, z2: Tensor) -> Tensor:
    """Penalize correlation between two subspaces.

    Computes the Frobenius norm of the cross-covariance matrix between
    z1 and z2, normalized by dimension.  Encourages the two subspaces
    to capture independent information.

    Args:
        z1: Embeddings from subspace 1, shape (B, D1).
        z2: Embeddings from subspace 2, shape (B, D2).

    Returns:
        Scalar cross-covariance penalty.
    """
    batch_size = z1.shape[0]
    if batch_size < 2:
        return z1.new_tensor(0.0)

    # Center
    z1_c = z1 - z1.mean(dim=0, keepdim=True)
    z2_c = z2 - z2.mean(dim=0, keepdim=True)

    # Cross-covariance: (D1, D2)
    cov = (z1_c.T @ z2_c) / (batch_size - 1)

    # Frobenius norm squared, normalized
    d = max(z1.shape[1], z2.shape[1])
    return cov.pow(2).sum() / d


def cross_correlation_loss(z1: Tensor, z2: Tensor, *, eps: float = 1.0e-6) -> Tensor:
    """Diagnose cross-subspace entanglement after per-dimension standardization.

    This is not currently used in the training objective. It exists to separate
    scale-driven cross-covariance spikes from genuinely correlated subspaces.
    """
    batch_size = z1.shape[0]
    if batch_size < 2:
        return z1.new_tensor(0.0)

    z1_c = z1 - z1.mean(dim=0, keepdim=True)
    z2_c = z2 - z2.mean(dim=0, keepdim=True)
    z1_std = z1_c.std(dim=0, unbiased=True).clamp_min(eps)
    z2_std = z2_c.std(dim=0, unbiased=True).clamp_min(eps)
    z1_z = z1_c / z1_std
    z2_z = z2_c / z2_std
    corr = (z1_z.T @ z2_z) / (batch_size - 1)
    d = max(z1.shape[1], z2.shape[1])
    return corr.pow(2).sum() / d


def adaptive_lambda(
    base_lambda: float,
    alpha: float,
    ep_statistic: Tensor,
) -> Tensor:
    """Compute adaptive regularization weight from EP statistic.

    When a subspace is close to collapse (high EP), increase the
    regularization.  When healthy (low EP), let prediction loss dominate.

    Args:
        base_lambda: Base regularization weight.
        alpha: Scaling factor for adaptation. 0 = fixed lambda.
        ep_statistic: Detached EP statistic for this subspace.

    Returns:
        Adaptive lambda value (scalar tensor).
    """
    return ep_statistic.new_tensor(base_lambda) * (1.0 + alpha * ep_statistic)


def vicreg_regularization(
    z: Tensor,
    *,
    variance_target: float = 1.0,
    variance_epsilon: float = 1.0e-4,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute a VICReg-style variance/covariance regularizer for one embedding space.

    This omits the invariance term because the Paper 2 comparison uses VICReg as
    an anti-collapse regularizer on a single latent space rather than between two
    augmented views.
    """
    batch_size, dim = z.shape
    if batch_size < 2 or dim < 1:
        zero = z.new_tensor(0.0)
        return zero, zero.detach(), zero.detach()

    centered = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=True) + variance_epsilon)
    variance_loss = F.relu(z.new_tensor(float(variance_target)) - std).mean()

    cov = (centered.T @ centered) / (batch_size - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    covariance_loss = off_diag.pow(2).sum() / max(1, dim * (dim - 1))

    total = (variance_weight * variance_loss) + (covariance_weight * covariance_loss)
    return total, variance_loss.detach(), covariance_loss.detach()
