"""Generate a synthetic corpus: latent draw -> anatomy -> refined CT -> cache.

Writes cases in exactly the format `morphome.preprocess` produces, so `HNCache`
loads them unchanged and any downstream training only has to be pointed at a
different `--cache`. Bone is *not* stored: `HNCache(with_bone=True)` derives it
from `ct_hu > 300`, so writing the refined CT keeps the derived channel
consistent by construction.

Two honest limits on what "more data" means here:

- **Diversity saturates.** The prior is a Gaussian fitted to 48 posterior means
  with k~14 components. Drawing 1000 samples does not add anatomical information
  beyond what those 48 cases determine; it resamples the same estimated
  distribution more densely. The script reports nearest-neighbour spacing among
  the generated latents against the real corpus so the saturation is visible
  rather than assumed.
- **Fine texture is invented.** The refiner supplies plausible detail that is not
  implied by the 32-d latent. Synthetic cases are legitimate training data for a
  model that must generalise; they are not evidence about any real patient.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from morphome.constants import HU_MAX, HU_MIN, N_STRUCT, STRUCTURES
from morphome.data import HNCache, denormalize_hu
from morphome.diffusion import ddim_sample
from morphome.preprocess import body_mask
from morphome.render import bone_sharpness
from explore_latent import decode_grid, encode_all, load_model, n_derived, wants_bone
from fit_prior import PCAGaussianPrior
from sample_refined import load_refiner


def pack_case(ct_norm: np.ndarray, probs: np.ndarray, min_voxels: int,
              spacing: float, apply_body_mask: bool = True) -> dict:
    """Refined CT + mask probabilities -> the on-disk cache record."""
    ct_hu = np.clip(denormalize_hu(ct_norm), HU_MIN, HU_MAX)
    body = body_mask(ct_hu)
    if apply_body_mask:
        ct_hu = np.where(body, ct_hu, HU_MIN)

    labels = probs[:N_STRUCT] > 0.5
    counts = labels.reshape(N_STRUCT, -1).sum(1)
    # An organ the sampler failed to produce is marked absent rather than
    # supervised as empty -- the same contract the real cache uses for organs
    # nobody contoured. Training on an empty-but-present channel teaches deletion.
    presence = counts >= min_voxels
    labels[~presence] = False

    packed = np.zeros(ct_hu.shape, dtype=np.uint16)
    for i in range(N_STRUCT):
        packed |= (labels[i].astype(np.uint16) << i)

    # Arrays are (z, y, x); `grid_dims` is (x, y, z) to match what preprocess
    # writes. The spacing is the source corpus's, not a constant -- the dose
    # frame is 2.5 mm, and a hardcoded 1.6 would mislabel every case.
    return {"ct_hu": ct_hu.astype(np.int16), "labels_packed": packed,
            "body": np.packbits(body), "presence": presence,
            "grid_dims": np.asarray(ct_hu.shape[::-1], dtype=np.int32),
            "grid_spacing": np.float32(spacing)}, counts, float(body.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-ckpt", default="runs/v2_bone/last.pt")
    ap.add_argument("--refiner", default="runs/refiner/last.pt")
    ap.add_argument("--cache", default=r"E:\datasets\medical\morphome_cache\hn_128_1.6mm")
    ap.add_argument("--out", default=r"E:\datasets\medical\morphome_cache\hn_synth_v1")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2, help="volumes refined at once")
    ap.add_argument("--min-volume-mm3", type=float, default=204.8,
                    help="presence threshold as a physical volume; 204.8 mm3 is "
                         "the 50-voxel threshold the 1.6 mm corpora used")
    ap.add_argument("--min-voxels", type=int, default=0,
                    help="0 = derive from --min-volume-mm3 at the corpus "
                         "spacing; set explicitly to override")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--no-refine", action="store_true",
                    help="write the raw VAE CT instead (ablation corpus)")
    ap.add_argument("--append", action="store_true",
                    help="extend an existing corpus in --out rather than "
                         "overwriting it; case numbering continues")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    start, prev_z, prev_stats, tranches = 0, None, [], []
    meta_path = out / "meta.json"
    if args.append:
        if not meta_path.exists():
            raise SystemExit(f"--append given but {meta_path} does not exist")
        prev_meta = json.loads(meta_path.read_text())
        prev_z = np.load(out / "latents.npz")["z"]
        prev_stats = json.loads((out / "generate_stats.json").read_text())
        start = len(prev_stats)
        tranches = prev_meta.get("tranches", [{"seed": prev_meta.get("seed"),
                                               "n": start, "start": 0}])
        used = [t.get("seed") for t in tranches]
        if args.seed in used:
            # Same seed re-draws the same latents from the same fitted prior, so
            # the "new" tranche would be a byte-for-byte duplicate of an old one.
            raise SystemExit(f"seed {args.seed} already used by this corpus "
                             f"(seeds {used}); pick another or drop --append")
        print(f"appending to {start} existing cases (seeds {used})")
    elif meta_path.exists():
        print(f"WARNING: overwriting the corpus already in {out} "
              f"(use --append to extend it)")

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    vae, vcfg, _ = load_model(args.vae_ckpt, device, prefer_ema=True)
    if not wants_bone(vcfg):
        raise SystemExit("needs a bone-channel VAE")
    net, sched = (None, None)
    if not args.no_refine:
        net, sched, _ = load_refiner(args.refiner, device)

    src = HNCache(args.cache, derived=n_derived(vcfg), in_memory=False)
    spacing = float(src.meta.get("grid", {}).get("spacing", 1.6))
    # A voxel count is not portable across grids: 50 voxels is 205 mm3 at 1.6 mm
    # but 781 mm3 at 2.5 mm, which silently marks chiasm and optic nerves absent
    # on the coarser frame even though the sampler produced them at the right
    # physical size. Threshold on volume so the test means the same thing.
    min_voxels = args.min_voxels or max(1, round(args.min_volume_mm3 / spacing ** 3))
    print(f"presence threshold: {min_voxels} voxels "
          f"({min_voxels * spacing ** 3:.0f} mm3 at {spacing} mm)")
    mu, real_cases = encode_all(vae, src, device)
    prior = PCAGaussianPrior(mu, var_target=0.90)
    z = prior.sample(args.n, rng)
    print(f"prior k={prior.k} ({prior.explained*100:.1f}% var) -> {args.n} draws")

    # Diversity: how far apart are the generated latents, against the real corpus?
    d_real = np.linalg.norm(mu[:, None] - mu[None], axis=-1)
    np.fill_diagonal(d_real, np.inf)
    d_gen = np.linalg.norm(z[:, None] - z[None], axis=-1)
    np.fill_diagonal(d_gen, np.inf)
    d_gen_real = np.linalg.norm(z[:, None] - mu[None], axis=-1).min(1)
    print(f"latent NN spacing: real-real {d_real.min(1).mean():.2f}   "
          f"gen-gen {d_gen.min(1).mean():.2f}   gen->nearest real "
          f"{d_gen_real.mean():.2f}")

    t0 = time.time()
    stats = []
    dims_xyz = None
    for i0 in range(0, args.n, args.batch):
        zb = torch.from_numpy(z[i0:i0 + args.batch])
        cts, probs = decode_grid(vae, zb, device)
        if net is not None:
            cond = torch.cat([cts, probs], dim=1).to(device)
            ct_out = ddim_sample(net, sched, cond, steps=args.steps,
                                 generator=gen).float().cpu().numpy()[:, 0]
        else:
            ct_out = cts[:, 0].numpy()

        for k in range(ct_out.shape[0]):
            idx = i0 + k
            rec, counts, bodyfrac = pack_case(ct_out[k], probs[k].numpy(),
                                              min_voxels, spacing)
            name = f"synth{start + idx:04d}"
            dims_xyz = rec["grid_dims"].tolist()
            np.savez_compressed(out / f"{name}.npz", **rec)
            stats.append({"case": name, "n_present": int(rec["presence"].sum()),
                          "body_frac": bodyfrac,
                          "sharpness": bone_sharpness(ct_out[k]),
                          "organ_voxels": {STRUCTURES[j]: int(counts[j])
                                           for j in range(N_STRUCT)}})
        el = time.time() - t0
        done = min(i0 + args.batch, args.n)
        print(f"[{done:4d}/{args.n}] {el/60:.1f}m elapsed, "
              f"eta {el/done*(args.n-done)/60:.1f}m", flush=True)

    tranches.append({"seed": args.seed, "n": args.n, "start": start,
                     "refined": not args.no_refine, "ddim_steps": args.steps})
    z_all = z if prev_z is None else np.concatenate([prev_z, z])
    stats = prev_stats + stats
    np.savez_compressed(out / "latents.npz", z=z_all, prior_mean=prior.mean,
                        prior_components=prior.components,
                        prior_coord_mean=prior.coord_mean, prior_chol=prior.chol,
                        prior_resid_std=np.float32(prior.resid_std))
    (out / "meta.json").write_text(json.dumps(
        {"grid": {"dims": dims_xyz, "spacing": spacing},
         "structures": list(STRUCTURES), "hu_range": [HU_MIN, HU_MAX],
         "body_masked": True, "synthetic": True, "case_glob": "synth*.npz",
         "vae_ckpt": args.vae_ckpt,
         "refiner": None if args.no_refine else args.refiner,
         "tranches": tranches, "n": len(stats),
         "min_voxels": min_voxels,
         "min_volume_mm3": round(min_voxels * spacing ** 3, 1),
         "prior": {"kind": "PCAGaussian", "k": int(prior.k),
                   "explained": float(prior.explained),
                   "fit_on": len(real_cases)},
         "source_corpus": args.cache}, indent=2))
    (out / "generate_stats.json").write_text(json.dumps(stats, indent=2))

    new = stats[start:]
    present = np.array([s["n_present"] for s in new])
    sharp = np.array([s["sharpness"] for s in new], dtype=float)
    print(f"\n{args.n} new cases -> {out}  (corpus now {len(stats)})")
    print(f"  organs present: mean {present.mean():.2f}/{N_STRUCT}, "
          f"all-{N_STRUCT} in {(present == N_STRUCT).sum()}/{args.n}")
    print(f"  bone sharpness: {np.nanmean(sharp):.1f} +/- {np.nanstd(sharp):.1f}")
    print(f"  wall clock {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
