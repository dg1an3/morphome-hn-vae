"""Two questions the sharpness statistic cannot answer.

1. **Does the refiner respect its conditioning?** Sharpness alone is maximised by
   plausible noise -- an untrained net scores 991 against real anatomy's ~345.
   The test is held-out *reconstruction*: refine the VAE's output for a
   validation case and measure L1 against that case's real CT. If the refiner is
   using the anatomy it was handed, L1 should fall below the VAE's own.

2. **Is it memorising texture?** 40 training volumes is squarely the regime where
   a patch-level model copies. The probe: for bone-region patches, compare the
   nearest-neighbour distance into the training corpus for *generated* patches
   against the same statistic for *held-out real* patches. Held-out real cases
   set the floor for how similar independent anatomy naturally looks. If
   generated patches sit clearly below that floor, the model is copying rather
   than generalising.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from morphome.constants import BONE_HU_THRESHOLD, N_STRUCT
from morphome.data import HNCache, cache_cases, default_split, denormalize_hu
from morphome.diffusion import ddim_sample
from morphome.model import build_input
from morphome.render import bone_sharpness
from explore_latent import decode_grid, encode_all, load_model, n_derived, wants_bone
from fit_prior import PCAGaussianPrior
from sample_refined import load_refiner


def bone_patches(vol: np.ndarray, bone: np.ndarray, size: int, stride: int,
                 min_frac: float = 0.10, cap: int | None = None,
                 rng: np.random.RandomState | None = None) -> np.ndarray:
    """Grid of patches whose bone content exceeds `min_frac`.

    Axes are swept independently: the frames are not cubes (thorax is
    96 x 160 x 224, head and neck 192 x 128 x 96), and sweeping all three over
    shape[0] would confine every patch to a corner of the volume rather than
    raising an error.
    """
    out = []
    D, H, W = vol.shape
    for z in range(0, D - size + 1, stride):
        for y in range(0, H - size + 1, stride):
            for x in range(0, W - size + 1, stride):
                sl = (slice(z, z + size), slice(y, y + size), slice(x, x + size))
                if bone[sl].mean() >= min_frac:
                    out.append(vol[sl].ravel())
    if not out:
        return np.zeros((0, size ** 3), np.float32)
    arr = np.stack(out).astype(np.float32)
    if cap is not None and len(arr) > cap:
        rng = rng or np.random.RandomState(0)
        arr = arr[rng.choice(len(arr), cap, replace=False)]
    return arr


def nn_distance(queries: np.ndarray, corpus: torch.Tensor, device,
                chunk: int = 256) -> np.ndarray:
    """Min L2 distance from each query patch to the corpus."""
    q = torch.from_numpy(queries).to(device)
    mins = []
    for i in range(0, len(q), chunk):
        d = torch.cdist(q[i:i + chunk], corpus)
        mins.append(d.min(dim=1).values.cpu().numpy())
    return np.concatenate(mins)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-ckpt", default="runs/v2_bone/last.pt")
    ap.add_argument("--refiner", default="runs/refiner/last.pt")
    ap.add_argument("--cache", default=r"E:\datasets\medical\morphome_cache\hn_128_1.6mm")
    ap.add_argument("--out", default="notes/refiner_eval")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--n-gen", type=int, default=6)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--patches-per-vol", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)   # ddim_sample draws x_T from torch's RNG

    vae, vcfg, _ = load_model(args.vae_ckpt, device, prefer_ema=True)
    if not wants_bone(vcfg):
        raise SystemExit("needs a bone-channel VAE")
    net, sched, _ = load_refiner(args.refiner, device)

    all_cases = cache_cases(args.cache)
    train_cases, val_cases = default_split(all_cases, 8, 0)

    # ---------------- 1. held-out reconstruction fidelity ----------------
    nd = n_derived(vcfg)
    val_ds = HNCache(args.cache, val_cases, in_memory=False, derived=nd)
    rows = []
    refined_vols = []
    with torch.no_grad():
        for i in range(len(val_ds)):
            s = val_ds[i]
            real = s["ct"][0].numpy()
            mu, _ = vae.encoder(build_input(s["ct"][None].to(device),
                                            s["labels"][None].to(device),
                                            s["presence"][None].to(device)))
            c_ct, c_pr = vae.decoder(mu)
            c_pr = torch.sigmoid(c_pr)
            cond = torch.cat([c_ct, c_pr], dim=1)
            ref = ddim_sample(net, sched, cond, steps=args.steps)
            raw_np = c_ct[0, 0].float().cpu().numpy()
            ref_np = ref[0, 0].float().cpu().numpy()
            refined_vols.append(ref_np)
            rows.append({
                "case": s["case"],
                "l1_vae": float(np.abs(raw_np - real).mean()),
                "l1_refined": float(np.abs(ref_np - real).mean()),
                "sharp_vae": bone_sharpness(raw_np),
                "sharp_refined": bone_sharpness(ref_np),
                "sharp_real": bone_sharpness(real),
            })
            print(f"  {s['case']}  L1 {rows[-1]['l1_vae']:.4f} -> "
                  f"{rows[-1]['l1_refined']:.4f}   sharp {rows[-1]['sharp_vae']:.0f} -> "
                  f"{rows[-1]['sharp_refined']:.0f}  (real {rows[-1]['sharp_real']:.0f})")

    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "case"}
    print("\nheld-out reconstruction (n=8):")
    print(f"  L1 to real     VAE {agg['l1_vae']:.4f}  ->  refined {agg['l1_refined']:.4f}")
    print(f"  bone sharpness VAE {agg['sharp_vae']:.0f}    ->  refined "
          f"{agg['sharp_refined']:.0f}   (real {agg['sharp_real']:.0f})")

    # ---------------- 2. memorisation probe ----------------
    print("\nbuilding training patch corpus...")
    train_ds = HNCache(args.cache, train_cases, in_memory=False, derived=nd)
    corpus = []
    for i in range(len(train_ds)):
        s = train_ds[i]
        v = s["ct"][0].numpy()
        corpus.append(bone_patches(v, s["labels"][N_STRUCT].numpy(), args.patch,
                                   args.stride, cap=args.patches_per_vol, rng=rng))
    corpus = torch.from_numpy(np.concatenate(corpus)).to(device)
    print(f"  corpus {tuple(corpus.shape)} ({args.patch}^3 patches, stride {args.stride})")

    # queries A: generated + refined
    ds_all = HNCache(args.cache, derived=nd, in_memory=False)
    mu_all, _ = encode_all(vae, ds_all, device)
    prior = PCAGaussianPrior(mu_all, var_target=0.90)
    z = torch.from_numpy(prior.sample(args.n_gen, np.random.RandomState(args.seed)))
    cts, probs = decode_grid(vae, z, device)
    q_gen = []
    for i in range(args.n_gen):
        cond = torch.cat([cts[i:i + 1], probs[i:i + 1]], dim=1).to(device)
        g = ddim_sample(net, sched, cond, steps=args.steps)[0, 0].float().cpu().numpy()
        bone = (denormalize_hu(g) > BONE_HU_THRESHOLD).astype(np.float32)
        q_gen.append(bone_patches(g, bone, args.patch, args.stride,
                                  cap=args.patches_per_vol, rng=rng))
    q_gen = np.concatenate(q_gen)

    # queries B: held-out real cases -- the floor for natural similarity
    q_real = []
    for i in range(len(val_ds)):
        s = val_ds[i]
        q_real.append(bone_patches(s["ct"][0].numpy(), s["labels"][N_STRUCT].numpy(),
                                   args.patch, args.stride,
                                   cap=args.patches_per_vol, rng=rng))
    q_real = np.concatenate(q_real)

    # queries C: refined held-out reconstructions
    q_ref = []
    for i, v in enumerate(refined_vols):
        bone = (denormalize_hu(v) > BONE_HU_THRESHOLD).astype(np.float32)
        q_ref.append(bone_patches(v, bone, args.patch, args.stride,
                                  cap=args.patches_per_vol, rng=rng))
    q_ref = np.concatenate(q_ref)

    probe = {}
    for name, q in (("generated+refined", q_gen), ("held-out real", q_real),
                    ("refined held-out recon", q_ref)):
        d = nn_distance(q, corpus, device)
        probe[name] = {"n": int(len(d)), "mean": float(d.mean()),
                       "p05": float(np.percentile(d, 5)),
                       "median": float(np.median(d)),
                       "min": float(d.min())}
        print(f"  {name:<24} n={len(d):5d}  NN-L2 mean {d.mean():.3f}  "
              f"p05 {np.percentile(d,5):.3f}  min {d.min():.3f}")

    ratio = probe["generated+refined"]["mean"] / probe["held-out real"]["mean"]
    print(f"\n  generated / held-out-real mean NN distance = {ratio:.3f}")
    print("  (<< 1 would indicate the refiner is copying training texture)")

    (out / "refiner_eval.json").write_text(json.dumps(
        {"held_out": {"per_case": rows, "mean": agg}, "memorisation": probe,
         "gen_over_real_nn_ratio": ratio}, indent=2))
    print(f"\nwrote {out/'refiner_eval.json'}")


if __name__ == "__main__":
    main()
