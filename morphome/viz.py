"""Figure rendering for training monitoring: reconstructions and prior samples."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .constants import N_STRUCT
from .model import build_input

PALETTE = np.array([
    [0.90, 0.10, 0.10], [1.00, 0.95, 0.20], [0.20, 0.60, 1.00],
    [0.20, 1.00, 0.40], [0.00, 0.70, 0.25], [1.00, 0.45, 0.00],
    [0.85, 0.30, 0.00], [0.80, 0.30, 1.00], [0.55, 0.10, 0.85],
    [0.00, 0.90, 0.90],                                  # Bone (derived)
])


def _boundary(mask2d: np.ndarray) -> np.ndarray:
    """1-pixel outline of a binary mask (mask minus its 4-neighbour erosion)."""
    er = mask2d.copy()
    er[1:] &= mask2d[:-1]
    er[:-1] &= mask2d[1:]
    er[:, 1:] &= mask2d[:, :-1]
    er[:, :-1] &= mask2d[:, 1:]
    return mask2d & ~er


def _overlay(ct2d: np.ndarray, prob2d: np.ndarray, thr: float = 0.5, alpha: float = 0.55):
    """ct2d in [-1,1]; prob2d is (C,H,W) of probabilities.

    Derived channels past the OARs (bone) cover 5-10 % of the slice, so they are
    drawn as an outline rather than a fill -- a solid overlay would hide the
    anatomy the figure exists to show, and the edge is the thing being judged.
    """
    g = np.clip((ct2d + 1) / 2, 0, 1)
    rgb = np.repeat(g[..., None], 3, axis=-1)
    for i in range(prob2d.shape[0]):
        m = prob2d[i] > thr
        if not m.any():
            continue
        if i >= N_STRUCT:
            m = _boundary(m)
            rgb[m] = 0.15 * rgb[m] + 0.85 * PALETTE[i]
        else:
            rgb[m] = (1 - alpha) * rgb[m] + alpha * PALETTE[i]
    return np.clip(rgb, 0, 1)


def _three_views(ct: np.ndarray, prob: np.ndarray, center=None):
    """ct (D,H,W), prob (C,D,H,W) -> list of 3 RGB images."""
    if center is None:
        # Centre on the OARs only: bone spans most of the volume and would drag
        # the view centre away from the organ complex.
        any_lab = prob[:N_STRUCT].max(0) > 0.5
        center = (np.argwhere(any_lab).mean(0).astype(int)
                  if any_lab.any() else np.array(ct.shape) // 2)
    z, y, x = [int(np.clip(center[i], 0, ct.shape[i] - 1)) for i in range(3)]
    return [
        _overlay(ct[z], prob[:, z]),
        _overlay(ct[:, y][::-1], prob[:, :, y][:, ::-1]),
        _overlay(ct[:, :, x][::-1], prob[:, :, :, x][:, ::-1]),
    ]


@torch.no_grad()
def save_recon_figure(model, loader, device, path: Path, n: int = 3):
    was_training = model.training
    model.eval()
    batch = next(iter(loader))
    ct = batch["ct"].to(device)
    labels = batch["labels"].to(device)
    presence = batch["presence"].to(device)
    n = min(n, ct.shape[0])

    x = build_input(ct[:n], labels[:n], presence[:n])
    rec_ct, rec_prob, _ = model.reconstruct(x)

    ct_np = ct[:n, 0].float().cpu().numpy()
    lab_np = labels[:n].float().cpu().numpy()
    rct_np = rec_ct[:, 0].float().cpu().numpy()
    rpr_np = rec_prob.float().cpu().numpy()

    fig, axes = plt.subplots(2 * n, 3, figsize=(9, 6 * n), squeeze=False)
    for i in range(n):
        gt = _three_views(ct_np[i], lab_np[i])
        rc = _three_views(rct_np[i], rpr_np[i])
        for k in range(3):
            axes[2 * i, k].imshow(gt[k])
            axes[2 * i, k].set_title(f"{batch['case'][i]} GT" if k == 0 else "", fontsize=8)
            axes[2 * i, k].axis("off")
            axes[2 * i + 1, k].imshow(rc[k])
            axes[2 * i + 1, k].set_title("recon" if k == 0 else "", fontsize=8)
            axes[2 * i + 1, k].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    if was_training:
        model.train()


@torch.no_grad()
def save_sample_figure(model, device, path: Path, n: int = 4, temperature: float = 1.0):
    was_training = model.training
    model.eval()
    ct, prob, _ = model.sample(n, device, temperature=temperature)
    ct_np = ct[:, 0].float().cpu().numpy()
    pr_np = prob.float().cpu().numpy()

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for i in range(n):
        views = _three_views(ct_np[i], pr_np[i])
        for k in range(3):
            axes[i, k].imshow(views[k])
            axes[i, k].set_title(f"prior sample {i}" if k == 0 else "", fontsize=8)
            axes[i, k].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    if was_training:
        model.train()
