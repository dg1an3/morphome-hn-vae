"""Composite rendering: turn a crisp predicted bone mask into crisp bone
*appearance*.

Why this exists
---------------
The CT channel is trained with L1, whose optimum is a conditional median, so
wherever the model is unsure about the exact position of a bone edge it emits a
blurred ramp between the two possibilities. The bone *mask* channel is trained
with Dice, which does not reward hedging, so it comes out committed. This module
spends the second to fix the first.

Two distinct defects are corrected, and both matter:

1. **The halo.** Blur pushes bone-ish intensity a few millimetres *outside* the
   true cortex. Clamping everything outside the mask to a soft-tissue ceiling
   removes it. Without this step the composite still looks smeared, because the
   bright ramp survives just beyond the mask edge.
2. **The washed-out interior.** Inside the mask the predicted intensities are
   compressed into a narrow band well below real cortical HU. Rescaling that
   band across a bone window restores the cortex/marrow distinction instead of
   painting the whole mask one flat value -- a flat fill reads as a cartoon and
   throws away the (real) intensity structure the decoder did learn.

This is a post-hoc renderer, not a loss change: it never touches training, and
it is only as good as the mask it is given.
"""

from __future__ import annotations

import numpy as np

from .data import denormalize_hu, normalize_hu

# Above this the model is claiming bone; below, soft tissue. Matches the
# threshold the mask channel was supervised at.
DEFAULT_THR = 0.5

# Everything outside the bone mask is capped here. ~150 HU sits above every soft
# tissue in the head and below trabecular bone, so the cap removes the blur halo
# without touching muscle, fat or contrast.
SOFT_TISSUE_MAX_HU = 150.0

# Target window for the composited skeleton.
BONE_LO_HU = 250.0
BONE_HI_HU = 1300.0


def _soft_mask(prob: np.ndarray, thr: float, edge: float) -> np.ndarray:
    """Probability -> [0,1] alpha with a controlled transition width.

    A hard threshold aliases badly at 1.6 mm; a raw probability is too soft and
    reintroduces the blur this function exists to remove.
    """
    if edge <= 0:
        return (prob > thr).astype(np.float32)
    return np.clip((prob - thr) / edge + 0.5, 0.0, 1.0).astype(np.float32)


def bone_composite(ct_norm: np.ndarray, bone_prob: np.ndarray,
                   thr: float = DEFAULT_THR, edge: float = 0.15,
                   soft_tissue_max_hu: float = SOFT_TISSUE_MAX_HU,
                   bone_lo_hu: float = BONE_LO_HU, bone_hi_hu: float = BONE_HI_HU,
                   lo_pct: float = 5.0, hi_pct: float = 99.0) -> np.ndarray:
    """Re-impose bone intensity inside a predicted bone mask.

    `ct_norm` is a (D,H,W) volume in [-1,1]; `bone_prob` is the matching bone
    channel probability. Returns a volume in [-1,1].
    """
    hu = denormalize_hu(ct_norm.astype(np.float32))
    alpha = _soft_mask(bone_prob, thr, edge)

    # 1. kill the blur halo outside the mask
    outside = np.minimum(hu, soft_tissue_max_hu)

    # 2. stretch the in-mask intensities across the bone window. Percentiles
    #    rather than min/max: a single dental-amalgam voxel would otherwise set
    #    the top of the range and flatten everything else.
    core = hu[bone_prob > thr]
    if core.size < 64:
        # Nothing credible to rescale; leave the CT alone rather than inventing
        # a skeleton out of noise.
        return ct_norm.astype(np.float32)
    lo, hi = np.percentile(core, [lo_pct, hi_pct])
    if hi - lo < 1.0:
        inside = np.full_like(hu, 0.5 * (bone_lo_hu + bone_hi_hu))
    else:
        t = (hu - lo) / (hi - lo)
        inside = bone_lo_hu + np.clip(t, 0.0, 1.0) * (bone_hi_hu - bone_lo_hu)

    out = (1.0 - alpha) * outside + alpha * inside
    return normalize_hu(np.clip(out, -1000.0, 1500.0)).astype(np.float32)


def bone_composite_batch(ct_norm: np.ndarray, bone_prob: np.ndarray, **kw) -> np.ndarray:
    """`bone_composite` over a leading batch axis: (N,D,H,W) -> (N,D,H,W)."""
    return np.stack([bone_composite(ct_norm[i], bone_prob[i], **kw)
                     for i in range(ct_norm.shape[0])])
