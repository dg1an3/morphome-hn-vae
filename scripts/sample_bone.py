"""Generate synthetic cases from a bone-channel checkpoint, with composite
bone rendering.

Puts the three pieces together:
  * the PCA-Gaussian prior from `fit_prior.py` (N(0, I) is never the right
    sampler for this model -- see the README),
  * the Dice-supervised bone mask channel, which is crisp where the L1 CT is not,
  * `morphome.render.bone_composite`, which spends the mask on the CT.

Writes a raw-vs-composite figure so the two are always judged side by side, and
optionally NRRD for 3D Slicer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from morphome.constants import N_STRUCT
from morphome.data import HNCache
from morphome.render import bone_composite
from explore_latent import decode_grid, encode_all, export_nrrd, load_model, wants_bone
from fit_prior import PCAGaussianPrior


def _slices(vol: np.ndarray, center) -> list[np.ndarray]:
    z, y, x = center
    return [vol[z], vol[:, y][::-1], vol[:, :, x][::-1]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/v2_bone/last.pt")
    ap.add_argument("--cache", default=r"E:\datasets\medical\morphome_cache\hn_128_1.6mm")
    ap.add_argument("--out", default="notes/bone_samples")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--var-target", type=float, default=0.90)
    ap.add_argument("--raw-weights", action="store_true")
    ap.add_argument("--export-nrrd", type=int, default=0)
    ap.add_argument("--edge", type=float, default=0.15,
                    help="composite transition width in probability units")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    model, cfg, ck = load_model(args.ckpt, device, prefer_ema=not args.raw_weights)
    if not wants_bone(cfg):
        raise SystemExit(
            f"{args.ckpt} has {cfg.out_label_channels} label channels and no bone "
            "channel; retrain with --with-bone")

    # Streamed rather than in-memory: this is a one-pass encode and the training
    # process may still be holding the corpus.
    ds = HNCache(args.cache, with_bone=True, in_memory=False)
    spacing = float(ds.meta.get("grid", {}).get("spacing", 1.6))
    mu, _ = encode_all(model, ds, device)
    prior = PCAGaussianPrior(mu, var_target=args.var_target)
    print(f"PCA-Gaussian prior: k={prior.k} ({prior.explained*100:.1f}% variance)")

    z = torch.from_numpy(prior.sample(args.n, rng))
    cts, probs = decode_grid(model, z, device)
    ct_np = cts[:, 0].numpy()
    prob_np = probs.numpy()
    bone_np = prob_np[:, N_STRUCT]

    comp = np.stack([bone_composite(ct_np[i], bone_np[i], edge=args.edge)
                     for i in range(args.n)])

    bone_frac = (bone_np > 0.5).mean(axis=(1, 2, 3))
    oar_vox = (prob_np[:, :N_STRUCT] > 0.5).sum(axis=(1, 2, 3, 4))
    n_organs = ((prob_np[:, :N_STRUCT] > 0.5).sum(axis=(2, 3, 4)) > 50).sum(1)
    for i in range(args.n):
        print(f"  sample {i}: bone {bone_frac[i]*100:5.2f}% of volume   "
              f"organs>50vox {n_organs[i]}/9   organ voxels {oar_vox[i]}")

    # Two rows per view: raw decoder CT above, composite below, same window.
    views = ("axial", "coronal", "sagittal")
    fig, axes = plt.subplots(6, args.n, figsize=(2.6 * args.n, 15.5), squeeze=False)
    for i in range(args.n):
        any_oar = prob_np[i, :N_STRUCT].max(0) > 0.5
        center = (np.argwhere(any_oar).mean(0).astype(int) if any_oar.any()
                  else np.array(ct_np[i].shape) // 2)
        center = [int(np.clip(center[k], 0, ct_np[i].shape[k] - 1)) for k in range(3)]
        raw_v = _slices(ct_np[i], center)
        cmp_v = _slices(comp[i], center)
        for v in range(3):
            for r, img in ((2 * v, raw_v[v]), (2 * v + 1, cmp_v[v])):
                axes[r, i].imshow(np.clip((img + 1) / 2, 0, 1), cmap="gray",
                                  vmin=0, vmax=1)
                axes[r, i].axis("off")
            if i == 0:
                axes[2 * v, i].set_ylabel(views[v])
        axes[0, i].set_title(f"#{i} raw", fontsize=8)
        axes[1, i].set_title("composite", fontsize=8)
    fig.tight_layout()
    path = out / "samples_raw_vs_composite.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

    if args.export_nrrd > 0:
        for i in range(min(args.export_nrrd, args.n)):
            export_nrrd(comp[i], prob_np[i], out / "nrrd", f"sample_{i}", spacing)
        print(f"wrote {out/'nrrd'} (CT is the composite render)")


if __name__ == "__main__":
    main()
