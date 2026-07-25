"""Losses for the joint CT + segmentation VAE.

The organ channels are extremely sparse -- the chiasm occupies ~127 of 2.1M
voxels (6e-5). Plain BCE is minimised almost perfectly by predicting zero
everywhere, so the segmentation term is Dice-dominated, with BCE kept only as a
smooth gradient signal. Every organ term is masked by `presence` so that
un-contoured organs contribute nothing rather than being supervised as absent.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(probs: torch.Tensor, target: torch.Tensor,
                   presence: torch.Tensor, eps: float = 1.0):
    """Per-sample, per-channel soft Dice, averaged over present channels only.

    Returns (mean_loss, per_channel_dice_sum, per_channel_count) so the caller
    can log real per-organ Dice without a second pass.
    """
    dims = tuple(range(2, probs.dim()))
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)          # (B, C)

    m = presence.to(dice.dtype)
    n = m.sum().clamp(min=1.0)
    loss = ((1.0 - dice) * m).sum() / n
    return loss, (dice.detach() * m).sum(0), m.sum(0)


def masked_bce(logits: torch.Tensor, target: torch.Tensor,
               presence: torch.Tensor, pos_weight=50.0):
    """BCE restricted to contoured organs, with positives upweighted.

    `pos_weight` may be a scalar or a per-channel sequence. A single value does
    not fit once a dense derived channel is in play: 50x is calibrated for organs
    at ~1e-3 of the volume and would wildly over-weight bone at ~1e-1.
    """
    pw = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    if pw.ndim == 1:
        pw = pw.view(1, -1, *([1] * (logits.dim() - 2)))
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pw, reduction="none")
    dims = tuple(range(2, logits.dim()))
    per = bce.mean(dims)                                  # (B, C)
    m = presence.to(per.dtype)
    return (per * m).sum() / m.sum().clamp(min=1.0)


def ct_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor,
                           body: torch.Tensor | None = None,
                           body_weight: float = 4.0):
    """L1 on the normalised CT.

    L1 rather than MSE because dental amalgam produces extreme streak artefacts;
    MSE would let a handful of saturated voxels dominate the gradient. Voxels
    inside the body mask are upweighted so the model does not spend capacity
    perfecting the air around the patient.
    """
    err = (pred - target).abs()
    if body is None:
        return err.mean()
    w = 1.0 + (body_weight - 1.0) * body
    return (err * w).sum() / w.sum().clamp(min=1.0)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0):
    """KL(q(z|x) || N(0, I)) in nats, averaged over the batch.

    `free_bits` applies a floor per latent dimension. With 40 training volumes
    an unconstrained KL term collapses most dimensions to the prior within a few
    hundred steps; the floor keeps them carrying information.
    """
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)   # (B, D)
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl = kl_per_dim.sum(1).mean()
    active = (kl_per_dim.detach().mean(0) > 0.01).sum()
    return kl, active


def beta_schedule(step: int, warmup_steps: int, beta_max: float,
                  start_step: int = 0) -> float:
    """Linear KL warm-up. Reconstruction must get established first, otherwise
    the decoder learns to ignore z entirely (posterior collapse)."""
    if step < start_step:
        return 0.0
    if warmup_steps <= 0:
        return beta_max
    t = (step - start_step) / warmup_steps
    return float(min(1.0, max(0.0, t)) * beta_max)


def vae_loss(out: dict, ct: torch.Tensor, labels: torch.Tensor,
             presence: torch.Tensor, body: torch.Tensor, beta: float,
             w_ct: float = 1.0, w_dice: float = 1.0, w_bce: float = 0.1,
             free_bits: float = 0.5, body_weight: float = 4.0,
             pos_weight=50.0):
    probs = torch.sigmoid(out["label_logits"])

    l_ct = ct_reconstruction_loss(out["ct"], ct, body, body_weight)
    l_dice, dice_sum, dice_cnt = soft_dice_loss(probs, labels, presence)
    l_bce = masked_bce(out["label_logits"], labels, presence, pos_weight)
    kl, active = kl_divergence(out["mu"], out["logvar"], free_bits)

    total = w_ct * l_ct + w_dice * l_dice + w_bce * l_bce + beta * kl

    return total, {
        "loss": total.detach(),
        "ct_l1": l_ct.detach(),
        "dice_loss": l_dice.detach(),
        "bce": l_bce.detach(),
        "kl": kl.detach(),
        "beta": torch.as_tensor(beta),
        "active_dims": active.detach().float(),
        "_dice_sum": dice_sum,
        "_dice_cnt": dice_cnt,
    }
