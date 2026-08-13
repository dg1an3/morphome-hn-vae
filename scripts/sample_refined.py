"""Generate synthetic cases end to end: latent -> anatomy -> refined CT.

    z ~ PCA-Gaussian prior        (the only sampler that works for this model)
      -> VAE decoder              anatomy: masks are crisp, CT is L1-blurry
      -> diffusion refiner        texture, conditioned on that anatomy

Renders raw / composite / refined side by side against real cases, and reports
the sharpness statistic each variant achieves, so the refiner is judged against
the two cheaper things it is supposed to beat rather than in isolation.
"""

from __future__ import annotations

import argparse
import json
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
from morphome.diffusion import Schedule, UNet3d, UNetConfig, ddim_sample
from morphome.render import bone_composite, bone_sharpness
from explore_latent import (decode_grid, encode_all, export_nrrd, load_model,
                            n_derived, wants_bone)
from fit_prior import PCAGaussianPrior


def load_refiner(path: str, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = UNet3d(UNetConfig(**ck["unet"])).to(device)
    net.load_state_dict(ck.get("ema") or ck["model"])
    net.eval()
    print(f"loaded refiner {path} (ema) step={ck['step']} over {ck['vae_ckpt']}")
    return net, Schedule(ck["timesteps"], device=device), ck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-ckpt", default="runs/v2_bone/last.pt")
    ap.add_argument("--refiner", default="runs/refiner/last.pt")
    ap.add_argument("--cache", default=r"E:\datasets\medical\morphome_cache\hn_128_1.6mm")
    ap.add_argument("--out", default="notes/refined_samples")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--export-nrrd", type=int, default=0)
    ap.add_argument("--save-volumes", action="store_true",
                    help="also write the raw/composite/refined arrays (~50 MB each)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    # The diffusion step draws x_T from torch's RNG. Without this the latents are
    # reproducible but the volumes decoded from them are not, which makes the
    # saved z useless for regenerating a specific sample.
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    vae, vcfg, _ = load_model(args.vae_ckpt, device, prefer_ema=True)
    if not wants_bone(vcfg):
        raise SystemExit("needs a bone-channel VAE")
    net, sched, _ = load_refiner(args.refiner, device)

    ds = HNCache(args.cache, derived=n_derived(vcfg), in_memory=False)
    spacing = float(ds.meta.get("grid", {}).get("spacing", 1.6))
    mu, _ = encode_all(vae, ds, device)
    prior = PCAGaussianPrior(mu, var_target=0.90)
    z = torch.from_numpy(prior.sample(args.n, rng))
    print(f"PCA-Gaussian prior: k={prior.k} ({prior.explained*100:.1f}% variance)")

    cts, probs = decode_grid(vae, z, device)
    raw = cts[:, 0].numpy()
    prob_np = probs.numpy()
    comp = np.stack([bone_composite(raw[i], prob_np[i, N_STRUCT]) for i in range(args.n)])

    refined = np.empty_like(raw)
    for i in range(args.n):
        cond = torch.cat([cts[i:i + 1], probs[i:i + 1]], dim=1).to(device)
        refined[i] = ddim_sample(net, sched, cond, steps=args.steps,
                                 generator=gen)[0, 0].float().cpu().numpy()
        print(f"  refined sample {i}")

    real = np.stack([ds[i]["ct"][0].numpy() for i in range(args.n)])

    rows = {"raw (VAE)": raw, "composite": comp, "refined": refined, "real (ref)": real}
    stats = {k: [bone_sharpness(v[i]) for i in range(args.n)] for k, v in rows.items()}
    print("\nmean |grad HU| across the bone surface:")
    for k, v in stats.items():
        print(f"  {k:<14} {np.nanmean(v):7.1f}")

    fig, axes = plt.subplots(len(rows), args.n, figsize=(2.5 * args.n, 2.6 * len(rows)),
                             squeeze=False)
    for r, (tag, vols) in enumerate(rows.items()):
        for i in range(args.n):
            axes[r, i].imshow(np.clip((vols[i][:, :, 64][::-1] + 1) / 2, 0, 1),
                              cmap="gray", vmin=0, vmax=1)
            axes[r, i].axis("off")
        axes[r, 0].set_title(tag, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "raw_composite_refined.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out/'raw_composite_refined.png'}")

    (out / "sharpness.json").write_text(json.dumps(
        {k: {"per_sample": v, "mean": float(np.nanmean(v))} for k, v in stats.items()},
        indent=2))

    # Provenance. A sample is worthless without the z that produced it and the
    # exact weights that decoded it: the refiner is only valid over the VAE it
    # was trained against, and both RNG streams have to be pinned to reproduce a
    # specific volume.
    np.savez_compressed(
        out / "latents.npz", z=z.numpy(),
        prior_mean=prior.mean, prior_components=prior.components,
        prior_coord_mean=prior.coord_mean, prior_chol=prior.chol,
        prior_resid_std=np.float32(prior.resid_std))
    (out / "provenance.json").write_text(json.dumps({
        "vae_ckpt": args.vae_ckpt, "refiner": args.refiner,
        "seed": args.seed, "ddim_steps": args.steps, "n": args.n,
        "prior": {"kind": "PCAGaussian", "k": int(prior.k),
                  "explained": float(prior.explained),
                  "var_target": 0.90, "fit_on_cases": len(ds)},
        "latent_dim": int(z.shape[1]),
        "note": "z rows correspond to sample index; regenerate with the same "
                "--seed and checkpoints. The 'real (ref)' row in the figure is "
                "the first n corpus cases and is NOT paired with the samples.",
    }, indent=2))
    print(f"wrote {out/'latents.npz'} and provenance.json")

    if args.save_volumes:
        np.savez_compressed(out / "volumes.npz", raw=raw, composite=comp,
                            refined=refined, masks=(prob_np > 0.5))
        print(f"wrote {out/'volumes.npz'}")

    if args.export_nrrd > 0:
        for i in range(min(args.export_nrrd, args.n)):
            export_nrrd(refined[i], prob_np[i], out / "nrrd", f"sample_{i}", spacing)
        print(f"wrote {out/'nrrd'} (CT is the refined volume)")


if __name__ == "__main__":
    main()
