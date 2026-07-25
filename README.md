# morphome-hn-vae

A generative 3D VAE over head-and-neck CT **and** organ-at-risk segmentations —
the first component of the morphome latent anatomy model.

The model learns a single global latent — 32-d in v1 — that jointly encodes
image and shape, so a sample from the prior is a complete synthetic case: a CT
volume *plus* its nine organ contours, mutually consistent by construction.

---

## Data

**Source:** PDDCA 1.4.1 (MICCAI 2015 Head & Neck Auto-Segmentation Challenge),
already extracted at `E:\datasets\medical\miccai_hn_sharpe`.

> The three `PDDCA-1.4.1_part{1,2,3}.zip` archives contain **exactly** the same
> data — verified file-by-file: 486 files each side, zero files unique to
> either, zero size mismatches. They are a redundant archival copy; there is
> nothing extra to extract.

**48 cases.** Nine structures, of which three are incompletely contoured:

| structure | cases | median voxels @1.6 mm |
|---|---|---|
| BrainStem | 48/48 | 6 480 |
| Chiasm | 48/48 | 127 |
| Mandible | **40/48** | 12 713 |
| OpticNerve_L / _R | 48/48 | 145 / 124 |
| Parotid_L / _R | 48/48 | 7 268 / 7 382 |
| Submandibular_L | **41/48** | 1 688 |
| Submandibular_R | **36/48** | 1 797 |

A missing contour means *nobody drew it*, not *the organ is absent*. Every such
channel is masked out of the loss via a per-case `presence` vector. Treating
them as background would train the model to delete organs.

The corpus is geometrically heterogeneous — in-plane 0.76–1.27 mm, slice
1.25–3.0 mm, 76–360 slices — so nothing can assume voxel counts are comparable
across cases. All 48 share identity direction cosines.

## Canonical frame

Every case is resampled to a **128³ grid at 1.6 mm** (a 204.8 mm cube), anchored
on the centroid of BrainStem + both parotids (the three structures present in
all 48 cases) plus a fixed offset of `(0, −37, −8)` mm in LPS.

That offset is derived, not guessed. `scripts/analyze_frame.py` measures the
union OAR bounding box relative to the anchor across all cases:

```
x [-90.8,  87.1]   y [-117.1, 43.2]   z [-84.5, 69.0]  mm
required side lengths: 177.8 x 160.3 x 153.5 mm
```

The anchor centroid sits ~37 mm posterior to the centre of that box, because the
mandible extends far anteriorly. **A first pass without the offset clipped the
mandible against the crop face in 24/48 cases.** With it, QC reports zero
boundary contacts and zero label voxels outside the body mask.

Resampling details that matter:
- Structure masks are resampled as **floats and re-thresholded at 0.5**. Nearest
  neighbour at 1.6 mm shreds the chiasm (median 184 source voxels).
- A **body mask** (largest connected component above −500 HU, holes filled along
  all three axes) removes the treatment couch and immobilisation shell, so
  capacity goes to anatomy rather than to scanner hardware.
- HU clipped to [−1000, 1500], normalised to [−1, 1].

Sanity check: the population mean CT over all 48 resampled cases is *sharp* —
mandibular arch, spinal canal and airway are all crisp, and the parotid
probability maps are bilaterally symmetric. Per-structure centroid spread is
small (BrainStem σ = [1.2, 3.7, 3.5] mm), so the centroid frame aligns cases
well enough that no registration is needed for v1.

## Model

`morphome/model.py` — a global-latent 3D convolutional VAE.

```
input   10ch @ 128³   (1 CT + 9 binary masks)
enc     128³×16 → 64³×32 → 32³×64 → 16³×128 → 8³×192 → 4³×256
        flatten 16 384 → Linear → mu, logvar (32-d)
dec     mirror, nearest-upsample + conv
output  1ch CT (tanh)  +  9ch organ logits
        19.15 M params (enc 9.63 M / dec 9.52 M)
```

The first full run used a 256-d latent (30.16 M params). It reached the same
validation Dice but its prior was unusable; see
[the prior section](#the-aggregate-posterior-never-matches-n0-i--use-scriptsfit_priorpy).

**Why a global vector latent, not a spatial one.** A spatial latent
reconstructs better, but sampling one from N(0, I) yields spatially incoherent
anatomy unless you fit a second-stage prior over it. The goal here is an
anatomy manifold you can sample and interpolate *directly*, so the bottleneck is
global by construction.

Load-bearing details:
- **GroupNorm, not BatchNorm** — batches are 2–4 volumes.
- **Label-head bias initialised to −6.0.** Organs occupy 6e-5 to 3e-3 of the
  volume; without it the model emits p≈0.5 everywhere on step 1 and the Dice
  gradient explodes.
- **`logvar` clamped to [−8, 8]**, so an untrained encoder cannot blow up the
  reparameterisation.
- Nearest-upsample + conv instead of transposed conv, which checkerboards in 3D.

## Losses

- **CT:** L1, not MSE — dental amalgam streaks are extreme outliers that would
  dominate a squared error. Voxels inside the body mask are weighted 4×.
- **Organs:** soft Dice (dominant) + BCE (weak, for smooth gradients). Plain BCE
  is nearly minimised by predicting zero everywhere at these sparsities. Both
  are masked by `presence`.
- **KL:** free-bits floor of 0.5 nats/dim, linear β warm-up. With 40 training
  volumes an unconstrained KL collapses most dimensions within a few hundred
  steps.

## Augmentation

All on GPU, applied consistently to CT, labels and body mask:

1. **L-R flip (p=0.5)** — the highest-value augmentation available, effectively
   doubling the corpus. It **permutes the paired label channels**
   (`OpticNerve_L↔R`, `Parotid_L↔R`, `Submandibular_L↔R`); getting this wrong
   silently teaches the model that parotids swap sides at random.
   `scripts/smoke_test.py` asserts the permutation is a correct involution.
2. **Affine** — rotation ±8°, scale 0.92–1.08, translation ±6 % (±12 mm).
3. **Intensity** — gain ±5 %, bias ±5 %, Gaussian noise σ = 0.01.

`grid_sample` pads with zeros, but **0 is not air** in a [−1,1] volume (it is
~+250 HU). The code shifts by the normalised air value before sampling and back
after, so rotations do not paint a shell of fake dense tissue around the patient.

## Results

`runs/v1_ld32` — 3000 epochs, 216 min on an RTX 3090, `--latent-dim 32
--beta-max 3e-4`. Validation mean Dice on EMA weights peaks at **0.613** around
epoch 500–550 and settles at **0.591**:

| organ | Dice (ep 2999) |
|---|---|
| BrainStem | 0.85 |
| Parotid_R / _L | 0.79 / 0.77 |
| Mandible | 0.67 |
| Submandibular_L / _R | 0.66 / 0.65 |
| Chiasm | 0.36 |
| OpticNerve_R / _L | 0.32 / 0.21 |

**Shrinking the latent 8× costs nothing.** The first full run (`runs/full`,
256-d, `--beta-max 1e-3`) peaked at 0.625 (epoch 600) and settled at 0.590 —
the same place, with 30.16 M parameters instead of 19.15 M. All 32 dimensions
stay active for the whole run at ~3 nats/dim, so the 0.5-nat free-bits floor
never binds and nothing collapses; the lower beta also avoids the late-run Dice
erosion the 256-d run showed as beta ramped past 5e-4.

**Everything after ~epoch 700 is overfitting.** Train CT L1 keeps falling to
0.065 while validation CT L1 bottoms at 0.117 (ep 500–750) and drifts back up to
0.124. `best.pt` — scored on `ct_l1 + dice_loss` — is epoch 550. Prior-sample
quality nevertheless keeps improving over those same epochs (see below), so the
long tail is not wasted if generation rather than reconstruction is the goal.

Small structures (chiasm, optic nerves) sit at Dice ≈ 0.008 until ~epoch 180 and
only then begin to learn — short runs badly understate them.

**The CT channel is blurry and will remain so.** An L1/Gaussian VAE decoder
predicts a conditional mean, so fine texture averages out. Segmentations stay
crisp because Dice does not reward hedging. Sharp CT texture requires an
adversarial or diffusion decoder over this latent — a v2 item, not a tuning fix.

Bone is the most conspicuous casualty of this, and it is the one case where a
cheap fix exists — see below.

### The bone channel — `--with-bone`

Bone is derived from the CT (`ct_hu > 300`) and supervised as a tenth structure
with the same masked Dice + BCE as the OARs, so the skeleton gets a
representation that *cannot* hedge. It costs no cache rebuild (the channel is a
pure function of a channel already in the cache) and it needs two things not to
break: a logit bias of `-2.4` rather than `-6.0` (bone is ~4.7 % of the volume,
two orders of magnitude denser than an OAR), and its own BCE `pos_weight` of 2
rather than 50.

`runs/v2_bone` is `v1_ld32` plus that channel, all else equal. Bone Dice plateaus
at **0.62 by epoch 600** and does not move for the remaining 2400 epochs —
thresholding at 1.6 mm produces a speckly trabecular mask whose exact voxels are
not reproducible, so volumetric Dice understates the result. The metric that
matters is the surface distance to bone extracted from the *real* CT of the eight
held-out cases (`scripts/eval_bone_surface.py`):

| source of the bone surface | Dice | MSD | HD95 | sDice@1.6 mm |
|---|---|---|---|---|
| `v1_ld32` CT thresholded at 300 HU | 0.599 | 2.61 mm | 7.60 mm | 0.590 |
| `v2_bone` CT thresholded at 300 HU | 0.593 | 2.49 mm | 7.17 mm | 0.619 |
| **`v2_bone` bone channel** | **0.615** | **2.37 mm** | **6.12 mm** | 0.614 |

Two separate effects, and it is worth keeping them apart:

1. **The mask is a better bone surface than the CT** — HD95 6.12 mm against
   7.60 mm, a 19 % drop. This is the Dice-vs-L1 argument paying out as predicted.
2. **The auxiliary task also sharpened the CT channel itself**, which was not the
   stated goal: the same model's thresholded CT improves to 7.17 mm without ever
   being asked to. On generated anatomy, mean |grad HU| across the bone surface
   rises from 217 (`v1_ld32`) to 230 (`v2_bone`); real cases sit at 346.

OAR Dice is unchanged within split noise (0.583 vs 0.591), so the extra channel
costs nothing.

### Composite rendering — `morphome/render.py`

The crisp mask can be spent on the CT: clamp everything outside it to 150 HU to
kill the blur halo, and rescale the in-mask intensities across a bone window to
restore cortex/marrow contrast. Against a *simulated* decoder blur (Gaussian
sigma 1.6 on a real CT, composited with the ground-truth mask) this is dramatic —
re-thresholded bone Dice 0.806 -> 0.963, surface gradient 136 -> 233.

Against the real model it is worth much less: 230 -> 238 mean |grad HU| on
generated samples. **The bone channel had already fixed most of what the renderer
was built to fix**, at source rather than post-hoc. It is still the right thing to
apply at generation time, but it is a polish step, not the main event.

The honest ceiling: 238 against 346 for real anatomy. Bone-specific supervision
closes roughly a sixth of the gap. The rest is the L1 conditional median — and
that is what the diffusion refiner below is for.

### The diffusion refiner — `morphome/diffusion.py`

```
z ~ PCA-Gaussian  ->  VAE decoder  ->  blurry CT + 10 crisp masks
                                              |
                                       diffusion refiner
                                              v
                                          sharp CT
```

A conditional diffusion model, `p(sharp CT | blurry CT, masks)`, v-parameterised
on a cosine schedule. An 11 M-param 3D UNet, **trained on 64^3 patches and run on
128^3 volumes** — it is fully convolutional, so the resolution change is free.
`runs/refiner`: 20 000 steps, 155 min, DDIM-50 sampling at 13.7 s per volume.

It refines rather than replaces the decoder because of the data. 40 volumes
cannot support learning 3D anatomy from noise, but they hold thousands of 64^3
patches, and with the anatomy supplied as conditioning the model only has to
learn *local texture*. The latent manifold, the interpolation and the
PCA-Gaussian prior are all untouched; refinement is a post-step.

Two design points carry most of the risk:

- **Train on the VAE's own reconstructions, never on synthetic blur.** A
  Gaussian-sigma-1.6 stand-in predicted the composite renderer would be worth
  136 -> 233; against the real decoder it delivered 230 -> 238. The decoder's
  failure mode is not a Gaussian blur.
- **Jitter `z` during training.** At generation time the conditioning is a prior
  sample, which is smoother than a reconstruction. Training on decodes of
  `mu + U[0, 0.3]·sigma·eps` widens the conditioning distribution toward what the
  sampler actually produces while keeping the real CT a valid target.

Mean |grad HU| across the bone surface of **generated** samples (n=6):

| | sharpness | gap to real closed |
|---|---|---|
| raw VAE | 230.0 | — |
| composite render | 238.2 | 7 % |
| **diffusion refined** | **349.9** | **~100 %** |
| real cases | 345.6 | — |

**Sharpness alone proves nothing** — it is maximised by plausible noise, and an
untrained net scores 991. The test that matters is held-out *reconstruction*
(`scripts/eval_refiner.py`, n=8): refine the VAE output for a validation case and
measure L1 against that case's real CT.

| | L1 to real | bone sharpness |
|---|---|---|
| VAE output | 0.0884 | 230 |
| refined | **0.0848** | **345** |
| real | — | 368 |

L1 *falls* in 7 of 8 cases. The refiner is not overwriting the anatomy it was
handed; it is closer to the real CT than the blurry input it started from.

**It is not memorising.** With 40 training volumes, patch-level copying is the
obvious failure. Nearest-neighbour L2 from 16^3 bone-region patches into the
training corpus, with held-out real cases setting the floor for how similar
independent anatomy naturally looks:

| query patches | mean NN-L2 | p05 | min |
|---|---|---|---|
| generated + refined | 14.73 | 8.23 | 4.13 |
| held-out real | 15.55 | 8.39 | 5.56 |
| refined held-out recon | 14.06 | 7.67 | 4.97 |

Generated texture sits 5 % nearer the corpus than genuinely independent anatomy
does — a mild pull toward canonical texture, not copying, which would show as a
ratio far below 1. Note that refining *real* held-out anatomy lands even closer
(14.06), confirming the effect is a property of the refiner's texture in general
rather than of specific memorised cases.

**What this does not license.** Fine structure — sinus air cells, trabecular
pattern — is invented, not inferred: none of it is in the 32-d latent. That is
fine for synthetic anatomy, which is the entire output here, but a refined volume
can never support a claim about a real patient's imaging.

This retires the composite renderer (238 vs 350) and most of TODO item 4.

### The aggregate posterior never matches N(0, I) — use `scripts/fit_prior.py`

Sampling `z ~ N(0, I)` from the **256-d** latent produces **incoherent anatomy**.
This is not undertraining; it is a dimensionality mismatch, and it is measurable:
`‖mu‖` over the corpus is 10.04 while an `N(0, I)` draw in 256-d sits at 16.00,
and 17 PCA components carry 90 % of the latent variance. 48 cases can span at
most 47 dimensions and effectively use ~17. The remaining ~239 directions carry
no anatomy, but an `N(0, I)` draw puts unit variance into every one of them, so
samples land on a shell of radius 16 while all real data lives at radius ~10 and
the decoder is evaluated far off-manifold.

Shrinking the latent to 32-d (`runs/v1_ld32`) fixes that, but does **not** make
the native prior correct — and the direction of the residual mismatch changes
over training as beta ramps:

| checkpoint | `‖mu‖` | `N(0, I)` shell | matched `T` | PCA k for 90 % |
|---|---|---|---|---|
| 256-d, ep 900 | 10.04 | 16.00 | 0.63 (shrink) | 17 |
| 32-d, ep 550 (`best.pt`) | 9.61 | 5.66 | 1.70 (stretch) | 12 |
| 32-d, ep 2999 (`last.pt`) | 4.36 | 5.66 | 0.77 (shrink) | 15 |

At epoch 550 beta is still warming (6e-5) and the posterior means sit *outside*
the prior shell; by epoch 2999 beta has been pinned at its 3e-4 maximum for 1500
epochs and KL pressure has pulled them *inside* it. **The latent radius is a
property of the checkpoint, not of the architecture** — measure it, never assume
it.

Two remedies, both post-hoc — **no retraining needed**:

1. **Radius-matched temperature**, `T = ‖mu‖ / sqrt(latent_dim)`. Cheap and
   effective in either direction, but isotropic, and the latent is not.
2. **PCA-Gaussian prior** (`scripts/fit_prior.py`) — a full-covariance Gaussian
   in the leading-k PCA subspace of the posterior means, plus isotropic residual
   noise orthogonal to it.

Measured over n=6 samples, organs counted at >50 voxels; real cases average
~37 700 organ voxels:

| checkpoint | sampler | `‖z‖` | organs > 50 vox per sample | mean organ voxels |
|---|---|---|---|---|
| 256-d ep 900 | isotropic `T=1.0` | 15.95 | 9, 3, 9, 6, 9, 6 | 23 918 |
| | isotropic `T=0.63` | 9.97 | 8, 9, 5, 8, 6, 6 | 17 494 |
| | **PCA-Gaussian, k=17** | 9.93 | **9, 9, 9, 9, 9, 9** | 46 930 |
| 32-d ep 550 | isotropic `T=1.0` | 5.12 | 9, 6, 7, 5, 3, 7 | 15 630 |
| | isotropic `T=1.70` | 9.93 | 7, 9, 8, 8, 8, 8 | 31 736 |
| | **PCA-Gaussian, k=12** | 9.19 | **9, 9, 9, 9, 9, 9** | 44 352 |
| 32-d ep 2999 | isotropic `T=1.0` | 5.12 | 9, 6, 8, 6, 3, 6 | 20 021 |
| | isotropic `T=0.77` | 4.51 | 6, 8, 8, 9, 9, 9 | 30 810 |
| | **PCA-Gaussian, k=15** | 4.91 | **9, 9, 9, 9, 9, 9** | 43 165 |
| `v2_bone` ep 2999 | isotropic `T=1.0` | 5.12 | 9, 5, 9, 7, 6, 6 | 22 897 |
| | isotropic `T=0.77` | 4.52 | 7, 5, 7, 6, 8, 9 | 26 161 |
| | **PCA-Gaussian, k=14** | 4.35 | **9, 9, 9, 9, 9, 9** | 40 593 |

The bone channel leaves the latent geometry essentially untouched — `‖mu‖` 4.38
against 4.36, k=14 against 15 — so everything above transfers unchanged.

Three things this measures:

1. **`N(0, I)` is never the right sampler, at any latent size.** At 32-d it does
   what the shrink was meant to achieve — it decodes to coherent, recognisable
   heads instead of noise — but it still under-generates badly: 15.6 k / 20.0 k
   organ voxels against ~37.7 k real, and only 3–7 of 9 organs in most samples.
   *Coherent* and *complete* are different failures.
2. **Radius matching is worth a lot and costs nothing.** It raises generated
   organ volume by 50–100 % whether the correction is a stretch (`T=1.70`) or a
   shrink (`T=0.77`).
3. **The PCA-Gaussian prior remains the best sampler by a wide margin** — nine
   organs in 6/6 samples at every checkpoint tested. Its ep-2999 `k=15` samples
   are the sharpest generated anatomy in the project so far: cortical mandible,
   bilateral parotids and submandibulars, chiasm and both optic nerves. It does
   over-generate slightly (~43 k vs ~37.7 k voxels), consistent with a
   marginally over-wide fit in the leading subspace.

So the latent shrink did its job — it made `N(0, I)` usable — but it demotes
`fit_prior.py` from mandatory to merely better rather than retiring it.
**Sample with `pca_gaussian` on `last.pt`.**

k = 12–15 of 32 dimensions still carry 90 % of the variance, so the latent
remains ~2× the intrinsic dimensionality. 48 cases put a hard floor under how
far this can usefully be squeezed, and there is no accuracy left to win — only a
slightly better-behaved native prior.

## Usage

```powershell
.venv\Scripts\python.exe scripts\probe_dataset.py          # survey raw corpus
.venv\Scripts\python.exe scripts\analyze_frame.py          # derive crop frame
.venv\Scripts\python.exe -m morphome.preprocess `
    --out E:\datasets\medical\morphome_cache\hn_128_1.6mm --spacing 1.6
.venv\Scripts\python.exe scripts\qc_cache.py --cache ...   # verify geometry
.venv\Scripts\python.exe scripts\smoke_test.py             # shapes/VRAM/timing

.venv\Scripts\python.exe -m morphome.train --out runs\v1_ld32 --batch-size 4 `
    --epochs 3000 --latent-dim 32 `
    --beta-max 3e-4 --beta-start 3000 --beta-warmup 12000

.venv\Scripts\python.exe -m morphome.train --out runs\v2_bone --batch-size 4 `
    --epochs 3000 --latent-dim 32 --with-bone `
    --beta-max 3e-4 --beta-start 3000 --beta-warmup 12000

.venv\Scripts\python.exe scripts\explore_latent.py --ckpt runs\v1_ld32\last.pt
.venv\Scripts\python.exe scripts\fit_prior.py --ckpt runs\v1_ld32\last.pt `
    --out notes\prior_fit_ld32_ep2999          # fit + sample the PCA-Gaussian prior
.venv\Scripts\python.exe scripts\read_tb.py --run runs\v1_ld32\tb

.venv\Scripts\python.exe scripts\sample_bone.py --ckpt runs\v2_bone\last.pt
.venv\Scripts\python.exe scripts\eval_bone_surface.py `
    --ckpt runs\v1_ld32\last.pt --ckpt runs\v2_bone\last.pt

.venv\Scripts\python.exe scripts\train_refiner.py --out runs\refiner --steps 20000
.venv\Scripts\python.exe scripts\sample_refined.py    # z -> anatomy -> sharp CT
.venv\Scripts\python.exe scripts\eval_refiner.py      # fidelity + memorisation
```

`explore_latent.py` produces prior samples, latent interpolation between real
cases, PCA of the posterior means (the principal axes of anatomical variation),
and NRRD export of generated anatomy for 3D Slicer.

## Layout

```
morphome/
  constants.py    structure list, HU window, anchor offset
  preprocess.py   NRRD -> canonical-grid .npz cache
  data.py         dataset, presence masking, GPU augmentation
  model.py        encoder / decoder / VAE
  losses.py       masked Dice + BCE, weighted L1, KL with free bits
  ema.py          weight EMA for sampling
  viz.py          overlay figures
  render.py       composite bone rendering
  diffusion.py    conditional diffusion refiner (schedule, 3D UNet, DDIM)
  train.py        training loop
scripts/
  probe_dataset.py  analyze_frame.py   qc_cache.py
  smoke_test.py     explore_latent.py  read_tb.py
  fit_prior.py      render_from_ckpt.py
  sample_bone.py    eval_bone_surface.py
  train_refiner.py  sample_refined.py   eval_refiner.py
```

## Known limitations

- **48 cases** is very small for a generative model; augmentation and the small
  latent are doing heavy lifting. Expect the model to interpolate well and
  extrapolate poorly. Validation Dice stops improving around epoch 550 and the
  rest of the run is overfitting the 40 training volumes.
- No inter-patient registration — the frame is centroid-anchored only. Fine for
  the large OARs; likely the limiting factor for chiasm/optic nerves.
- The 8-case validation split is small enough that the split seed measurably
  moves the numbers. It is fixed (`--split-seed 0`) and recorded in
  `runs/*/config.json`.
- The aggregate posterior does not match `N(0, I)` at any latent size tested,
  and its radius drifts with beta over the course of a run, so **every new
  checkpoint needs `fit_prior.py` re-run before its samples mean anything**.
  `explore_latent.py` reports `‖mu‖` against the expected `sqrt(latent_dim)`, so
  this is measurable rather than assumed.
