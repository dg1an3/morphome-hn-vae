"""Per-corpus facts, selected once and re-exported by `constants`.

Twenty-three call sites across `morphome/` and `scripts/` read these as plain
module-level names (`from morphome.constants import STRUCTURES, N_STRUCT`).
Threading a profile object through all of them would touch every script for no
benefit, so the profile is chosen once -- via `MORPHOME_PROFILE`, default `hn` --
and `constants` re-exports its fields under the existing names. Nothing that
works today changes behaviour.

The risk this trades for is a silent mismatch: a cache built under one profile
read under another would misinterpret the label bitplanes. `HNCache` guards that
by checking `meta.json["structures"]` against the active profile, which is why
the cache has recorded its structure list from the start.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    """Everything that differs between corpora. Field order is the on-disk
    contract: `structures` indexes the label bitplanes in the cache."""

    name: str
    structures: tuple[str, ...]
    anchor_structures: tuple[str, ...]
    anchor_offset_mm: tuple[float, float, float]
    grid_dims: tuple[int, int, int]        # voxels, (x, y, z) = (L-R, A-P, S-I)
    grid_spacing: float                    # mm, isotropic
    # How cache files are named. HNCache globs for this rather than taking every
    # .npz, so sidecars (latents.npz, and anything else written beside the cases)
    # are excluded by construction instead of by a blacklist.
    case_glob: str = "*.npz"
    # Defaults below hold for any CT corpus; override only with a reason.
    derived_structures: tuple[str, ...] = ("Bone", "Body")
    hu_min: float = -1000.0
    hu_max: float = 1500.0
    body_hu_threshold: float = -500.0
    bone_hu_threshold: float = 300.0
    provisional_frame: bool = False        # grid/offset not yet fitted to data


# --------------------------------------------------------------------------
# PDDCA head and neck
# --------------------------------------------------------------------------
# ANCHOR: present in all 48 cases -> safe as the cropping anchor.
#
# OFFSET, LPS axes (+x left, +y posterior, +z superior): derived by
# scripts/fit_dose_frame.py against per-slice patient contours from all 48 source
# scans. The frame is sized for DOSE, which needs a closed external contour in
# every transverse slice -- not just for the OARs.
#
# The previous frame (204.8 mm cube, offset (0, -37, -8)) was sized around the
# union OAR bounding box and truncated the patient posteriorly in 94.1 % of
# transverse slices. This one clips 0 of 4218 slices.
#
# The +30 mm superior shift is load-bearing: 10-12 of 48 source scans have the
# shoulders/arms running outside the reconstruction FOV, which no crop can
# recover, so the window is held above them. z=+35 would reach zero scan-FOV
# clipping but leaves the submandibular glands only 0.5 mm of margin; z=+40 clips
# them outright. +30 keeps 5.5 mm of OAR margin and costs 2 slices.
#
# GRID: 2.5 mm is native dose-grid resolution; dimensions are multiples of 32 for
# the 5 encoder downsampling stages. Sized to the patient extent *within this z
# window* (408 x 287 mm over all 48 cases), not to the whole scan: holding the
# window above the shoulders means the lateral extent needed is far smaller than
# the shoulder width, and the box shrinks from 3.44 M to 2.36 M voxels -- about
# the same cost as the old 128^3 frame, while clipping 0 of 4218 transverse
# slices instead of 94 %.
HN = Profile(
    name="hn",
    structures=(
        "BrainStem",
        "Chiasm",
        "Mandible",
        "OpticNerve_L",
        "OpticNerve_R",
        "Parotid_L",
        "Parotid_R",
        "Submandibular_L",
        "Submandibular_R",
    ),
    anchor_structures=("BrainStem", "Parotid_L", "Parotid_R"),
    anchor_offset_mm=(0.0, 15.0, 30.0),
    grid_dims=(192, 128, 96),              # 480 x 320 x 240 mm
    grid_spacing=2.5,
    case_glob="0522c*.npz",
)


# --------------------------------------------------------------------------
# NSCLC-Radiomics (LUNG1) thorax
# --------------------------------------------------------------------------
# Structure order is the on-disk bitplane contract and must not be reordered once
# a cache exists. OARs first, target last.
#
# Availability over 422 patients, from the DICOM SEG objects: GTV 421,
# SpinalCord 411, Esophagus 355, Lung_L/Lung_R 312 each, Lungs 409 (the 312 unions
# plus 97 cases contoured only as a combined "Lungs-Total"), Heart 127.
#
# Heart is included despite reaching only 127/422 because absence is already a
# first-class state: the cache stores a per-case presence bitmask and the loss
# masks unsupervised channels, which is the same contract PDDCA uses for organs
# nobody contoured. Dropping the channel would lose a real OAR; keeping it costs
# the cases that lack one nothing.
#
# ANCHOR: Lungs, the only large structure with near-complete coverage (409/422).
# The 13 cases with no lung contour at all cannot be anchored and are skipped by
# preprocess_case, which raises when no anchor structure resolves.
#
# FRAME: sized for DOSE, i.e. a closed external contour in every transverse slice
# of the window, since path length and scatter depend on tissue no OAR occupies.
#
# Measured by analyze_body_extent.py over 40 cases, relative to the Lungs anchor:
# the body needs 497 mm laterally (p95 490), 351 mm A-P (p95 337), and spans
# 326 mm in z at the median case (p95 459, max 591).
#
# Unlike head and neck, nothing here is truncated by the scanner: 0/40 cases have
# the body touching the reconstruction FOV in x or y. (All 40 touch z_lo and
# z_hi, but that is just the patient continuing past the ends of the scan, which
# no frame needs to contain.) So full contour coverage is purely a sizing
# question here, where in HN it was impossible for the 10-12 of 48 scans whose
# shoulders ran outside the FOV.
#
# 3.0 mm rather than HN's 2.5 mm is what makes that affordable: 192 x 128 x 96 at
# 3.0 mm spans 576 x 384 x 288 mm, clearing the widest measured body by 79 mm
# laterally and 33 mm A-P, at 2,359,296 voxels -- the identical count to the HN
# frame, so compute and peak VRAM stay where they are known to work. The same
# dims at 2.5 mm span only 480 x 320 mm and clip both in-plane axes.
#
# The offset is anterior (-y), opposite HN's +15: the lung centroid sits
# posterior to the torso centre, so a box centred on it would waste posterior
# margin and clip the anterior chest wall.
#
# The offset is NOT the one fit_dose_frame.py reports as best. That search
# minimises the raw count of clipped slices, which rewards windows containing
# fewer slices: it returns z=-80, where 6 500 fewer slices fall in the window and
# the upper thorax is discarded rather than contained. Body clipping is flat at
# ~0.5 % for every z from -80 to +40, so z costs nothing there and is decided
# instead by anatomy. Measured containment over 40 cases at 192x128x96:
#
#     offset        Lungs    GTV   Esophagus  SpinalCord
#     (0,-20,-80)   85.9%   84.0%   67.1%      62.8%      <- fit's "best"
#     (0,-14,+14)   99.9%  100.0%   99.2%      83.9%      <- used here
#
# SpinalCord never reaches 100 % because the cord is contoured over the full scan
# length and cannot fit any 288 mm window; only the treated span matters.
#
# In-plane dims are sized to clip nothing at all: 224 x 160 gives half-extents of
# 336 and 240 mm against measured worst-case needs of 303 and 213. Verified over
# all 410 cases with a lung contour -- 0 of 38 865 transverse slices clipped by
# the box, 0 by the scan FOV.
#
# 192 x 128 would have cost 2.36 M voxels instead of 3.44 M (exactly the head and
# neck frame) but clipped 6 cases: LUNG1-367/362/354/170/138/097, all large body
# habitus. Excluding them would have biased the corpus by body size in the
# direction that matters most for dose, since path length and scatter are
# precisely where large patients differ, so the 1.46x is bought deliberately.
#
# Unlike head and neck, none of this was forced by the scanner: scan-FOV clipping
# is 0.00 % at every offset tried, where 10-12 of 48 HN scans were unrecoverable.
THORAX = Profile(
    name="thorax",
    structures=(
        "Esophagus",
        "Heart",
        "Lung_L",
        "Lung_R",
        "Lungs",
        "SpinalCord",
        "GTV",
    ),
    anchor_structures=("Lungs",),
    anchor_offset_mm=(0.0, -14.0, 14.0),
    grid_dims=(224, 160, 96),              # 672 x 480 x 288 mm
    grid_spacing=3.0,
    case_glob="LUNG1-*.npz",
)


PROFILES: dict[str, Profile] = {p.name: p for p in (HN, THORAX)}


def active() -> Profile:
    """The profile named by MORPHOME_PROFILE, defaulting to head and neck."""
    key = os.environ.get("MORPHOME_PROFILE", "hn").strip().lower()
    if key not in PROFILES:
        raise SystemExit(f"MORPHOME_PROFILE={key!r} is not one of "
                         f"{sorted(PROFILES)}")
    return PROFILES[key]


def flip_partners(names: tuple[str, ...]) -> dict[str, str]:
    """Left/right partner map derived from the `_L`/`_R` suffix.

    Deriving beats enumerating: it reproduces the head-and-neck pairs exactly
    (optic nerves, parotids, submandibulars) and picks up Lung_L/Lung_R with no
    edit. A structure whose partner is not in the list maps to itself, so an
    unpaired lateral organ is left alone rather than silently swapped away.
    """
    out: dict[str, str] = {}
    for n in names:
        if n.endswith("_L"):
            partner = n[:-2] + "_R"
        elif n.endswith("_R"):
            partner = n[:-2] + "_L"
        else:
            continue
        if partner in names:
            out[n] = partner
    return out
