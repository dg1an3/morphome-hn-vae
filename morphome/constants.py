"""Shared constants for the active dataset profile.

These were literals for the PDDCA head-and-neck corpus. They are now re-exported
from `profiles.active()` so a second corpus (NSCLC-Radiomics thorax) can reuse the
pipeline without editing this file per run; the rationale for each value lives
with its profile. Default is head and neck, so existing behaviour is unchanged --
set `MORPHOME_PROFILE=thorax` to switch.
"""

from .profiles import active

_P = active()

PROFILE = _P.name

# Channel order is fixed everywhere: preprocessing, model, losses, visualisation.
STRUCTURES = _P.structures
N_STRUCT = len(STRUCTURES)

# Present in (nearly) every case -> safe to use as the anatomical anchor for
# cropping. See the profile for coverage and why these structures specifically.
ANCHOR_STRUCTURES = _P.anchor_structures

# LPS axes: +x left, +y posterior, +z superior.
ANCHOR_OFFSET_MM = _P.anchor_offset_mm

# Canonical grid, in voxels, (x, y, z) = (L-R, A-P, S-I).
GRID_DIMS = _P.grid_dims
GRID_SPACING = _P.grid_spacing

# Cache filename pattern, recorded in meta.json so HNCache finds the cases and
# skips sidecars. Corpora do not share a naming convention: PDDCA cases are
# 0522c*, LUNG1 cases are LUNG1-*.
CASE_GLOB = _P.case_glob

# True when the grid/offset above are a placeholder rather than fitted to the
# corpus. Callers that build a cache should refuse or warn loudly.
PROVISIONAL_FRAME = _P.provisional_frame

# HU window used for normalisation. Air is -1000; cortical bone / dental work
# saturates well above 1500, but clipping there keeps the dynamic range useful
# for soft tissue, which is what the OARs live in.
HU_MIN = _P.hu_min
HU_MAX = _P.hu_max

# Anything below this is air/couch gap for body-mask extraction.
BODY_HU_THRESHOLD = _P.body_hu_threshold

# --------------------------------------------------------------------------
# Derived channels
# --------------------------------------------------------------------------
# Bone is supervised as an extra "structure" with the same masked Dice + BCE used
# for the OARs. The CT channel is trained with L1, whose optimum is a conditional
# median, so uncertain bone edges come out as a blurred ramp; Dice does not
# reward hedging, which is why the contour channels stay crisp. Giving bone a
# segmentation-like representation buys that crispness for the skeleton.
#
# It is *derived* from the CT rather than cached, so STRUCTURES above remains the
# on-disk contract and no cache rebuild is needed.
BONE_HU_THRESHOLD = _P.bone_hu_threshold

# Body is the external patient contour, cached alongside the CT and appended as
# a Dice-supervised channel for the same reason bone is: a dose engine needs a
# committed, closed contour per slice, and re-deriving one by thresholding a
# blurry L1-decoded CT reintroduces exactly the soft edge that supervision fixes.
DERIVED_STRUCTURES = _P.derived_structures

# Channel order for model input/output. Derived channels go last, so a model
# trained without them stays index-compatible with this list.
MODEL_STRUCTURES = STRUCTURES + DERIVED_STRUCTURES
N_MODEL_STRUCT = len(MODEL_STRUCTURES)
