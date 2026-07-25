# TODO / experiment queue

---

## 0. Shrink the latent to its intrinsic dimensionality  *(DONE)*

**Status:** trained as `runs/v1_ld32` (`--latent-dim 32 --beta-max 3e-4`,
3000 epochs, 216 min). Validation Dice unchanged (0.591 vs 0.590 at 256-d) and
`z ~ N(0, I)` now decodes to coherent anatomy instead of noise — but it still
under-generates organ volume by ~2x, so the PCA-Gaussian prior remains the
default sampler. Numbers in the README. The beta question below is answered:
3e-4 showed no Dice erosion, unlike 1e-3 at 256-d.

Residual: the posterior radius drifts with beta (‖mu‖ 9.61 at ep 550 -> 4.36 at
ep 2999), so `fit_prior.py` must be re-run per checkpoint.

Measured on the 48-case corpus: `‖mu‖` = 10.04 against 16.00 for an `N(0, I)`
draw in 256-d, and **17 PCA components carry 90 % of the latent variance**. The
latent is ~15x larger than the data supports, so `z ~ N(0, I)` samples land far
off-manifold and decode to noise.

`scripts/fit_prior.py` fixes this post-hoc (PCA-Gaussian prior, all 9 organs in
6/6 samples) and needs no retraining, so it is not urgent. But the next run
should set **`--latent-dim 32`** so the native prior is usable and the model is
better regularised for a 40-volume training set.

Open question worth an ablation: with a 32-d latent, does `beta` still need to
be as high? The KL pressure was partly compensating for an oversized latent, and
it measurably eroded validation Dice (0.625 at epoch 600 -> ~0.59 by epoch 900
as beta ramped 2.5e-4 -> 5.4e-4).

---

## 1. Bone-enhancing channel, to fix blurry bone reconstruction

**Status:** variants (a) and (d) implemented; first run is `runs/v2_bone`
(`--with-bone --latent-dim 32 --beta-max 3e-4`, otherwise identical to
`v1_ld32`). Bone is derived in `HNCache._load` as `ct_hu > BONE_HU_THRESHOLD`
(300 HU) and appended as a tenth structure, so no cache rebuild was needed.
`morphome/render.py` implements the composite renderer, `scripts/sample_bone.py`
samples the PCA-Gaussian prior and writes raw-vs-composite figures.

**Outcome: (a) worked, (d) turned out to be mostly redundant.** Numbers and the
surface-distance table are in the README.

- HD95 of the generated bone surface against real held-out CT: **7.60 mm ->
  6.12 mm** using the mask channel instead of a thresholded CT. Success criterion
  met.
- Unplanned bonus: the auxiliary task sharpened the **CT channel itself**
  (HD95 7.17 mm thresholding v2's own CT; mean |grad HU| on generated bone
  217 -> 230). Nobody asked it to; supervising bone as shape appears to
  regularise the intensity head.
- OAR Dice unchanged within split noise (0.583 vs 0.591), latent geometry
  unchanged (‖mu‖ 4.38 vs 4.36, k=14 vs 15).
- Bone volumetric Dice plateaus at 0.62 by epoch 600 and never moves. At 1.6 mm
  the >300 HU mask is speckly trabecular noise; **do not use volumetric Dice to
  judge this channel**, use the surface metrics.
- The composite renderer is worth 230 -> 238 |grad HU| on top of the bone
  channel, against 136 -> 233 on a simulated blur. It was built for a defect the
  bone channel had already fixed at source.

Remaining gap to real anatomy: 238 vs 346. Bone-specific supervision closes about
a sixth of it; the rest is the L1 conditional median. **Item 4 is now the only
thing that will move it**, and (b), (c), (e) are not worth trying first -- they
are all variations on supervising the same intensity field.
**Motivation:** the CT channel reconstructs bone as a soft grey smear. Cortical
edges, the mandibular ramus and the vertebral cortex are all washed out, while
the *segmentation* channels come out crisp.

### Why the asymmetry exists

This is not undertraining, and more epochs will not fix it:

- The CT channel is trained with **L1**, whose optimum is the conditional
  *median* of the posterior over voxel intensity. Where the model is uncertain
  about the exact location of a bone edge (a millimetre either way), the
  loss-minimising output is a blurred ramp between the two possibilities.
- The organ channels are trained with **Dice**, which does not reward hedging —
  a half-confident blob scores worse than a committed one. So they come out
  sharp.

The fix therefore is to give bone a **Dice-supervised, segmentation-like
representation**, rather than to keep tuning the intensity loss.

### Proposed variants (cheap to expensive)

**(a) Binary bone mask channel — try first.**
Add one output channel, `bone = CT > ~300 HU`, supervised with the same masked
Dice + BCE used for the OARs. Bone becomes the tenth "structure". Expected
outcome: a crisp bone shape, decoupled from the blurry intensity field.

**(b) Two-level bone channels.**
Split into cortical (>600 HU) and trabecular/cancellous (200-600 HU). Better
matches the actual bimodal appearance and may reconstruct the mandible cortex
distinctly from the medullary space. Costs one more channel.

**(c) Bone-window CT channel.**
A second intensity channel clipped to a bone window (e.g. 150-1500 HU) and
renormalised, supervised with L1. Recovers bone *contrast* that the wide
[-1000, 1500] window compresses, but is still an L1 target, so it will still
blur. Lower expected value than (a); mainly useful in combination.

**(d) Composite rendering at generation time.**
Once (a) works, synthesise the final CT as a blend: take the smooth predicted CT
and re-impose bone intensity inside the predicted bone mask. Turns the crisp
shape channel into crisp *appearance*. This is a post-hoc renderer, not a loss
change, and is the cheapest route to a good-looking sample.

**(e) Gradient / Laplacian-pyramid loss on the CT.**
Add an L1 term on image gradients or across a multi-scale pyramid. Helps
sharpness generally, but tends to amplify dental-streak artefacts, which is
exactly why plain L1 was chosen over MSE in the first place. Try only after
(a)-(d).

### Implementation notes

- **No cache rebuild needed.** The bone channel is a deterministic function of
  the CT, so derive it on GPU in the data path rather than re-running
  `morphome.preprocess` over all 48 cases.
- **As an *input* channel it adds no information** — it is a pure function of a
  channel the encoder already sees. The value is almost entirely on the
  **output + loss** side. Adding it to the encoder input is optional and mildly
  helpful at best; do not expect the gain to come from there.
- **Derive it before intensity jitter, not after.** Augmentation applies a
  random gain/bias to the CT; thresholding the jittered volume would make the
  bone mask wobble in a way that does not correspond to any real anatomy.
  Compute the mask from the pre-jitter CT, then carry it through the same affine
  and flip as the other label channels.
- `presence` for the bone channel is always 1 (it is derived, never missing), so
  it needs no special masking — but the channel count in
  `constants.N_STRUCT`, the flip permutation, the palette in `viz.py` and the
  label-head bias init all need updating together. The bone channel occupies
  ~5-10 % of the volume, far denser than any OAR, so its **logit bias should
  *not* be -6.0** like the others; initialise it near `logit(0.08) ≈ -2.4`.
- Evaluate with a bone-specific metric (surface Dice / Hausdorff at 95th pct
  against the thresholded real CT), not just volumetric Dice — the whole point
  is edge fidelity, which volumetric Dice barely penalises.

### Success criterion

A prior sample whose mandible and cervical vertebrae have a visible cortical
boundary at 1.6 mm, and a measurable drop in 95th-percentile surface distance
between the reconstructed and real bone surfaces on the held-out cases.

---

## 2. Label-channel dropout, to allow uncontoured H&N CT

**Status:** proposed (discussed, not implemented)

The loss already supports partial supervision via `presence`, so cases with
missing contours train correctly. But the **encoder takes labels as input**, so
a completely uncontoured case would present zeroed label channels — a
train/test mismatch.

Fix: randomly zero label channels during training (and set their `presence` to
0) so the encoder learns to infer shape from CT alone. This is the prerequisite
for ingesting any new H&N CT collection that lacks contours.

---

## 3. More data

Local survey (see `scripts/survey_thorax_coverage.py`,
`scripts/identify_collections.py`) confirmed PDDCA's 48 cases are the only H&N
CT with OAR contours on E:. Thorax collections (NLST 1.45 TB, NSCLC-Radiomics)
reach a **median of 0 mm above the lung apex** — 0/12 sampled series were
usable, and none contain the anchor structures.

Candidates to fetch (verify sizes): SegRap2023 (~200 cases, ~45 OARs),
HaN-Seg (~42, ~30 OARs), StructSeg2019 (~50, ~22 OARs), TCIA HNSCC and
Head-Neck-Radiomics-HN1 (RTSTRUCT, needs a converter).

---

## 4. Sharp CT appearance (v2 architecture)

Beyond the bone channel: an adversarial or diffusion decoder over the learned
latent. This is the principled fix for texture in general, of which bone is the
most conspicuous symptom. Larger scope than items 1-3.
