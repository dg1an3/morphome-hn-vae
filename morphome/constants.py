"""Shared constants for the PDDCA head-and-neck corpus."""

# Channel order is fixed everywhere: preprocessing, model, losses, visualisation.
STRUCTURES = (
    "BrainStem",
    "Chiasm",
    "Mandible",
    "OpticNerve_L",
    "OpticNerve_R",
    "Parotid_L",
    "Parotid_R",
    "Submandibular_L",
    "Submandibular_R",
)
N_STRUCT = len(STRUCTURES)

# Present in all 48 cases -> safe to use as the anatomical anchor for cropping.
ANCHOR_STRUCTURES = ("BrainStem", "Parotid_L", "Parotid_R")

# The anchor centroid sits posterior/superior to the centre of the OAR complex,
# because the mandible extends far anteriorly and the submandibulars inferiorly.
# Derived by scripts/analyze_frame.py: over all 48 cases the union OAR bounding
# box relative to the anchor spans x[-90.8, 87.1] y[-117.1, 43.2] z[-84.5, 69.0]
# mm, whose centre is offset by this much. Without it the mandible clipped the
# crop face in ~24/48 cases. LPS axes: +x left, +y posterior, +z superior.
ANCHOR_OFFSET_MM = (0.0, -37.0, -8.0)

# HU window used for normalisation. Air is -1000; cortical bone / dental work
# saturates well above 1500, but clipping there keeps the dynamic range useful
# for soft tissue, which is what the OARs live in.
HU_MIN = -1000.0
HU_MAX = 1500.0

# Anything below this is air/couch gap for body-mask extraction.
BODY_HU_THRESHOLD = -500.0

# --------------------------------------------------------------------------
# Derived channels
# --------------------------------------------------------------------------
# Bone is supervised as a tenth "structure" with the same masked Dice + BCE used
# for the OARs. The CT channel is trained with L1, whose optimum is a conditional
# median, so uncertain bone edges come out as a blurred ramp; Dice does not
# reward hedging, which is why the contour channels stay crisp. Giving bone a
# segmentation-like representation buys that crispness for the skeleton.
#
# It is *derived* from the CT rather than cached, so STRUCTURES above remains the
# on-disk contract (9 bitplanes) and no cache rebuild is needed.
BONE_HU_THRESHOLD = 300.0

DERIVED_STRUCTURES = ("Bone",)

# Channel order for model input/output. Derived channels go last, so a model
# trained without them stays index-compatible with this list.
MODEL_STRUCTURES = STRUCTURES + DERIVED_STRUCTURES
N_MODEL_STRUCT = len(MODEL_STRUCTURES)
