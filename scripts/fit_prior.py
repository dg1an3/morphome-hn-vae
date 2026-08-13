"""Fit an ex-post prior to the aggregate posterior and sample from it.

Why this exists
---------------
With 48 cases and a 256-d latent, sampling z ~ N(0, I) fails badly: the
posterior means sit on a shell of radius ~10 while N(0, I) draws sit at ~16, so
every prior sample lands off-manifold and decodes to noise. The latent's
intrinsic dimensionality is ~17 (90 % of variance), not 256.

Isotropic temperature scaling (z ~ N(0, T*I) with T ~= 0.63) fixes the radius
and already produces coherent anatomy, but treats every direction alike. The
aggregate posterior is anisotropic, so a better prior is a Gaussian fitted in
the leading PCA subspace of the posterior means -- the standard "ex-post density
estimation" / two-stage-VAE remedy.

This does not require retraining; it replaces N(0, I) at sampling time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from morphome.constants import N_STRUCT
from morphome.data import HNCache
from explore_latent import decode_grid, encode_all, load_model, n_derived, strip_figure


class PCAGaussianPrior:
    """Gaussian fitted in the top-k PCA subspace of the posterior means.

    Full covariance in 256-d is rank-deficient with only 48 samples, so the fit
    is done in the k-dim subspace where the data actually lives, with a small
    ridge for numerical safety.
    """

    def __init__(self, mu: np.ndarray, k: int | None = None,
                 var_target: float = 0.90, ridge: float = 1e-4):
        self.mean = mu.mean(0)
        X = mu - self.mean
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        var = S ** 2 / max(1, len(mu) - 1)
        ratio = var / var.sum()
        if k is None:
            k = int(np.searchsorted(np.cumsum(ratio), var_target) + 1)
        self.k = k
        self.components = Vt[:k]                 # (k, D)
        coords = X @ self.components.T           # (N, k)
        self.coord_mean = coords.mean(0)
        cov = np.cov(coords, rowvar=False) + ridge * np.eye(k)
        self.chol = np.linalg.cholesky(cov)
        self.explained = float(np.cumsum(ratio)[k - 1])
        # residual energy outside the subspace, added back as isotropic noise so
        # samples are not confined to a perfectly flat slab
        self.resid_std = float(np.sqrt(max(0.0, var[k:].sum() / max(1, mu.shape[1] - k))))

    def sample(self, n: int, rng: np.random.RandomState,
               scale: float = 1.0, with_residual: bool = True) -> np.ndarray:
        e = rng.randn(n, self.k)
        coords = self.coord_mean + (e @ self.chol.T) * scale
        z = coords @ self.components + self.mean
        if with_residual and self.resid_std > 0:
            perp = rng.randn(n, self.components.shape[1]) * self.resid_std * scale
            perp -= (perp @ self.components.T) @ self.components   # keep it orthogonal
            z = z + perp
        return z.astype(np.float32)


def radius_report(name: str, z: np.ndarray, target: float) -> None:
    r = np.linalg.norm(z, axis=1)
    print(f"  {name:<28} ||z|| mean={r.mean():6.2f} std={r.std():5.2f}   "
          f"(data shell {target:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/full/last.pt")
    ap.add_argument("--cache", default=r"E:\datasets\medical\morphome_cache\hn_128_1.6mm")
    ap.add_argument("--out", default="notes/prior_fit")
    ap.add_argument("--k", type=int, default=0, help="0 = choose by --var-target")
    ap.add_argument("--var-target", type=float, default=0.90)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--raw-weights", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    model, cfg, ck = load_model(args.ckpt, device, prefer_ema=not args.raw_weights)
    ds = HNCache(args.cache, derived=n_derived(cfg))
    mu, cases = encode_all(model, ds, device)

    data_r = float(np.linalg.norm(mu, axis=1).mean())
    iso_r = float(np.sqrt(cfg.latent_dim))
    print(f"\nlatent_dim={cfg.latent_dim}  n_cases={len(cases)}")
    print(f"data shell ||mu||={data_r:.2f}   N(0,I) shell={iso_r:.2f}   "
          f"radius-matched temperature={data_r/iso_r:.3f}")

    prior = PCAGaussianPrior(mu, k=(args.k or None), var_target=args.var_target)
    print(f"\nPCA-Gaussian prior: k={prior.k} components, "
          f"{prior.explained*100:.1f}% variance, residual std={prior.resid_std:.4f}")

    T = data_r / iso_r
    variants = {
        "isotropic_T1.0": rng.randn(args.n, cfg.latent_dim).astype(np.float32),
        f"isotropic_T{T:.2f}": (rng.randn(args.n, cfg.latent_dim) * T).astype(np.float32),
        f"pca_gaussian_k{prior.k}": prior.sample(args.n, rng),
    }

    print("\nsample radii:")
    for name, z in variants.items():
        radius_report(name, z, data_r)

    for name, z in variants.items():
        cts, probs = decode_grid(model, torch.from_numpy(z), device)
        # How much organ is actually generated -- a blank sample is a failure.
        # OAR channels only: a derived bone channel is ~100x denser and would
        # swamp the total, and these counts are compared across checkpoints.
        vol = probs.numpy()[:, :N_STRUCT].reshape(args.n, N_STRUCT, -1)
        present = (vol > 0.5).sum(-1)
        n_organs = (present > 50).sum(1)
        print(f"  {name:<28} organs>50vox per sample: {n_organs.tolist()}  "
              f"mean total organ voxels={present.sum(1).mean():.0f}")
        strip_figure(cts, probs, [f"{name}\n#{i}" for i in range(args.n)],
                     out / f"{name}.png")

    (out / "prior_fit.json").write_text(json.dumps({
        "latent_dim": cfg.latent_dim,
        "n_cases": len(cases),
        "data_shell_radius": data_r,
        "isotropic_shell_radius": iso_r,
        "radius_matched_temperature": T,
        "pca_k": prior.k,
        "pca_explained": prior.explained,
        "epoch": int(ck["epoch"]),
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
