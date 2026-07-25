"""Dataset and GPU augmentation for the canonical-grid head-and-neck cache.

Two things here are not boilerplate and matter a lot:

1. `presence` masking. Mandible is missing in 8/48 cases and the submandibular
   glands in 7-12. An absent contour means "nobody drew it", not "the organ is
   not there". Feeding zeros as ground truth would teach the model to delete
   organs, so presence is carried through to the loss and the encoder input.

2. Left-right flip augmentation must permute the paired label channels. With 48
   cases this flip is the single most valuable augmentation available, and
   getting the permutation wrong silently trains the model that parotids swap
   sides at random.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .constants import (
    BONE_HU_THRESHOLD,
    HU_MAX,
    HU_MIN,
    MODEL_STRUCTURES,
    N_STRUCT,
    STRUCTURES,
)

# Index permutation applied to label channels under an L-R flip.
_FLIP_PARTNER = {
    "OpticNerve_L": "OpticNerve_R",
    "OpticNerve_R": "OpticNerve_L",
    "Parotid_L": "Parotid_R",
    "Parotid_R": "Parotid_L",
    "Submandibular_L": "Submandibular_R",
    "Submandibular_R": "Submandibular_L",
}
FLIP_PERM = tuple(
    STRUCTURES.index(_FLIP_PARTNER.get(name, name)) for name in STRUCTURES
)
# Same permutation over the full model channel list. Derived channels (bone) are
# laterally symmetric and map to themselves, and they sit after the OARs, so
# MODEL_FLIP_PERM[:C] is valid for any C the model was built with.
MODEL_FLIP_PERM = tuple(
    MODEL_STRUCTURES.index(_FLIP_PARTNER.get(name, name)) for name in MODEL_STRUCTURES
)


def normalize_hu(ct_hu: np.ndarray | torch.Tensor):
    """HU -> [-1, 1]."""
    return 2.0 * (ct_hu - HU_MIN) / (HU_MAX - HU_MIN) - 1.0


def denormalize_hu(x):
    """[-1, 1] -> HU."""
    return (x + 1.0) / 2.0 * (HU_MAX - HU_MIN) + HU_MIN


def unpack_labels_torch(packed: np.ndarray) -> np.ndarray:
    out = np.zeros((N_STRUCT, *packed.shape), dtype=np.float32)
    for i in range(N_STRUCT):
        out[i] = (packed >> i) & 1
    return out


def default_split(cases: list[str], n_val: int = 8, seed: int = 0):
    """Deterministic case-level split. 48 cases is small enough that the split
    choice measurably moves validation numbers, so it is fixed by seed and
    recorded alongside every run."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(cases))
    val = sorted(cases[i] for i in idx[:n_val])
    train = sorted(cases[i] for i in idx[n_val:])
    return train, val


class HNCache(Dataset):
    """Loads preprocessed volumes. Returns CPU float tensors; augmentation runs
    on the GPU in the training loop where it is essentially free."""

    def __init__(self, cache_dir: str | Path, cases: list[str] | None = None,
                 in_memory: bool = True, with_bone: bool = False):
        self.with_bone = with_bone
        self.dir = Path(cache_dir)
        meta_path = self.dir / "meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        all_cases = sorted(p.stem for p in self.dir.glob("0522c*.npz"))
        self.cases = cases if cases is not None else all_cases
        missing = set(self.cases) - set(all_cases)
        if missing:
            raise FileNotFoundError(f"cases not in cache: {sorted(missing)}")
        self.in_memory = in_memory
        self._cache: dict[str, tuple] = {}
        if in_memory:
            for c in self.cases:
                self._cache[c] = self._load(c)

    def _load(self, case: str):
        d = np.load(self.dir / f"{case}.npz", allow_pickle=True)
        ct_hu = d["ct_hu"].astype(np.float32)
        ct = normalize_hu(ct_hu)[None]                                   # (1,D,H,W)
        lab = unpack_labels_torch(d["labels_packed"])                    # (9,D,H,W)
        presence = d["presence"].astype(np.float32)                      # (9,)
        if self.with_bone:
            # Derived from the cached (pre-jitter) HU. Thresholding *after*
            # intensity augmentation would make the target wobble by a few HU
            # per step in a way that corresponds to no real anatomy.
            bone = (ct_hu > BONE_HU_THRESHOLD).astype(np.float32)[None]
            lab = np.concatenate([lab, bone], axis=0)                    # (10,D,H,W)
            presence = np.concatenate([presence, np.ones(1, np.float32)])
        shape = d["ct_hu"].shape
        body = np.unpackbits(d["body"])[: int(np.prod(shape))]
        body = body.reshape(shape).astype(np.float32)[None]              # (1,D,H,W)
        return (torch.from_numpy(ct), torch.from_numpy(lab),
                torch.from_numpy(presence), torch.from_numpy(body))

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, i: int):
        case = self.cases[i]
        ct, lab, presence, body = self._cache[case] if self.in_memory else self._load(case)
        return {"ct": ct, "labels": lab, "presence": presence, "body": body, "case": case}


# --------------------------------------------------------------------------
# GPU augmentation
# --------------------------------------------------------------------------

def _affine_theta(batch: int, device, rot_deg: float, scale_rng: tuple[float, float],
                  shift_frac: float, generator=None) -> torch.Tensor:
    """Random rigid+scale matrices in normalised [-1,1] grid coordinates."""
    def u(lo, hi, shape):
        return torch.rand(shape, device=device, generator=generator) * (hi - lo) + lo

    ang = u(-rot_deg, rot_deg, (batch, 3)) * torch.pi / 180.0
    cx, sx = torch.cos(ang[:, 0]), torch.sin(ang[:, 0])
    cy, sy = torch.cos(ang[:, 1]), torch.sin(ang[:, 1])
    cz, sz = torch.cos(ang[:, 2]), torch.sin(ang[:, 2])
    one = torch.ones_like(cx)
    zero = torch.zeros_like(cx)

    Rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], 1).view(batch, 3, 3)
    Ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], 1).view(batch, 3, 3)
    Rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], 1).view(batch, 3, 3)
    R = Rz @ Ry @ Rx

    s = u(scale_rng[0], scale_rng[1], (batch, 1, 1))
    R = R * s

    t = u(-shift_frac, shift_frac, (batch, 3, 1))
    return torch.cat([R, t], dim=2)  # (B, 3, 4)


def augment(ct: torch.Tensor, labels: torch.Tensor, body: torch.Tensor,
            rot_deg: float = 8.0, scale_rng: tuple[float, float] = (0.92, 1.08),
            shift_frac: float = 0.06, p_flip: float = 0.5,
            intensity_shift: float = 0.05, intensity_scale: float = 0.05,
            noise_std: float = 0.01, generator=None):
    """Random affine + L-R flip + intensity jitter, applied consistently to CT,
    labels and body mask."""
    b, device = ct.shape[0], ct.device
    n_lab = labels.shape[1]

    # --- left-right flip (axis W == index 4 of NCDHW == patient L-R) ---------
    if p_flip > 0:
        do = torch.rand(b, device=device, generator=generator) < p_flip
        if do.any():
            sel = do.view(b, 1, 1, 1, 1)
            ct = torch.where(sel, ct.flip(-1), ct)
            body = torch.where(sel, body.flip(-1), body)
            flipped = labels.flip(-1)[:, list(MODEL_FLIP_PERM[:n_lab])]
            labels = torch.where(sel, flipped, labels)

    # --- affine -------------------------------------------------------------
    theta = _affine_theta(b, device, rot_deg, scale_rng, shift_frac, generator)
    stacked = torch.cat([ct, labels, body], dim=1)
    grid = F.affine_grid(theta, list(stacked.shape), align_corners=False)
    # Padding with the normalised value of air keeps the border physically
    # meaningful instead of introducing a hard zero (which is ~ -250 HU).
    air = float(normalize_hu(torch.tensor(HU_MIN)))
    stacked = stacked - air
    stacked = F.grid_sample(stacked, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False)
    stacked = stacked + air

    ct = stacked[:, :1]
    labels = stacked[:, 1:1 + n_lab].clamp_(0, 1)
    body = (stacked[:, 1 + n_lab:] > 0.5).float()

    # --- intensity ----------------------------------------------------------
    if intensity_scale or intensity_shift:
        gain = 1.0 + (torch.rand(b, 1, 1, 1, 1, device=device, generator=generator)
                      * 2 - 1) * intensity_scale
        bias = (torch.rand(b, 1, 1, 1, 1, device=device, generator=generator)
                * 2 - 1) * intensity_shift
        ct = ct * gain + bias
    if noise_std:
        ct = ct + torch.randn(ct.shape, device=device, generator=generator) * noise_std
    ct = ct.clamp_(-1.0, 1.0)

    return ct, labels, body
