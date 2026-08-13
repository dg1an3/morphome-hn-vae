"""Exercise the learned anatomy manifold.

Produces:
  * prior samples at several temperatures
  * latent interpolation between two real cases
  * a PCA of the posterior means over the whole corpus, plus traversals along
    the leading components (what does the model think the principal axes of
    head-and-neck anatomical variation are?)
  * optional NRRD export so generated anatomy can be opened in 3D Slicer

This is the deliverable that makes the model a "latent space model of anatomy"
rather than just an autoencoder.
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

from morphome.constants import MODEL_STRUCTURES, N_STRUCT, STRUCTURES
from morphome.data import HNCache, denormalize_hu
from morphome.model import HNVAE, VAEConfig, build_input
from morphome.viz import _three_views


def n_derived(cfg: VAEConfig) -> int:
    """How many derived channels (Bone, Body) this checkpoint was trained with.

    Derived channels are appended after the OARs, so the channel count alone
    identifies them. Always pass this to `HNCache(derived=...)` rather than a
    boolean: a 10-channel model wants Bone only, an 11-channel model wants Bone
    and Body, and feeding the wrong count is a silent shape mismatch at the
    encoder.
    """
    return max(0, cfg.out_label_channels - N_STRUCT)


def wants_bone(cfg: VAEConfig) -> bool:
    """True if this checkpoint has any derived channel."""
    return n_derived(cfg) > 0


def load_model(ckpt_path: str, device, prefer_ema: bool = True):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = VAEConfig(**ck["cfg"])
    model = HNVAE(cfg).to(device)
    sd = ck["ema"] if (prefer_ema and ck.get("ema") is not None) else ck["model"]
    model.load_state_dict(sd)
    model.eval()
    which = "ema" if (prefer_ema and ck.get("ema") is not None) else "raw"
    print(f"loaded {ckpt_path} ({which} weights) epoch={ck['epoch']} step={ck['step']}")
    return model, cfg, ck


@torch.no_grad()
def encode_all(model, ds, device, batch: int = 4):
    mus, cases = [], []
    for i in range(0, len(ds), batch):
        items = [ds[j] for j in range(i, min(i + batch, len(ds)))]
        ct = torch.stack([b["ct"] for b in items]).to(device)
        lab = torch.stack([b["labels"] for b in items]).to(device)
        pres = torch.stack([b["presence"] for b in items]).to(device)
        mu, _ = model.encoder(build_input(ct, lab, pres))
        mus.append(mu.float().cpu())
        cases += [b["case"] for b in items]
    return torch.cat(mus).numpy(), cases


@torch.no_grad()
def decode_grid(model, z: torch.Tensor, device, chunk: int = 2):
    cts, probs = [], []
    for i in range(0, z.shape[0], chunk):
        ct, logit = model.decoder(z[i:i + chunk].to(device))
        cts.append(ct.float().cpu())
        probs.append(torch.sigmoid(logit).float().cpu())
    return torch.cat(cts), torch.cat(probs)


def strip_figure(cts, probs, titles, path: Path, view: int = 0):
    n = cts.shape[0]
    fig, axes = plt.subplots(3, n, figsize=(2.6 * n, 8), squeeze=False)
    for i in range(n):
        views = _three_views(cts[i, 0].numpy(), probs[i].numpy())
        for r in range(3):
            axes[r, i].imshow(views[r])
            axes[r, i].axis("off")
            if r == 0:
                axes[r, i].set_title(titles[i], fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def export_nrrd(ct_norm: np.ndarray, prob: np.ndarray, out_dir: Path,
                name: str, spacing: float, thr: float = 0.5):
    """Write generated anatomy as NRRD (CT + one label map) for 3D Slicer."""
    import SimpleITK as sitk

    out_dir.mkdir(parents=True, exist_ok=True)
    hu = denormalize_hu(ct_norm).astype(np.int16)
    img = sitk.GetImageFromArray(hu)
    img.SetSpacing((spacing,) * 3)
    sitk.WriteImage(img, str(out_dir / f"{name}_ct.nrrd"), True)

    # Winner-take-all label map; 0 = background, i+1 = STRUCTURES[i]. Restricted
    # to the OARs: bone covers ~5-10 % of the volume and would win the argmax
    # over most of the head, so it is exported as its own binary segment.
    oar = prob[:N_STRUCT]
    lab = np.zeros(oar.shape[1:], dtype=np.uint8)
    best = oar.argmax(0)
    hit = oar.max(0) > thr
    lab[hit] = (best[hit] + 1).astype(np.uint8)
    seg = sitk.GetImageFromArray(lab)
    seg.SetSpacing((spacing,) * 3)
    sitk.WriteImage(seg, str(out_dir / f"{name}_seg.nrrd"), True)

    counts = {STRUCTURES[i]: int((lab == i + 1).sum()) for i in range(N_STRUCT)}
    for i in range(N_STRUCT, prob.shape[0]):
        derived = (prob[i] > thr).astype(np.uint8)
        img_d = sitk.GetImageFromArray(derived)
        img_d.SetSpacing((spacing,) * 3)
        sitk.WriteImage(img_d, str(out_dir / f"{name}_{MODEL_STRUCTURES[i].lower()}.nrrd"),
                        True)
        counts[MODEL_STRUCTURES[i]] = int(derived.sum())
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/hnvae/best.pt")
    ap.add_argument("--cache", default=r"E:\datasets\medical\morphome_cache\hn_128_1.6mm")
    ap.add_argument("--out", default="notes/latent")
    ap.add_argument("--raw-weights", action="store_true", help="use non-EMA weights")
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--n-interp", type=int, default=7)
    ap.add_argument("--n-pcs", type=int, default=4)
    ap.add_argument("--export-nrrd", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, cfg, ck = load_model(args.ckpt, device, prefer_ema=not args.raw_weights)
    ds = HNCache(args.cache, derived=n_derived(cfg))
    spacing = float(ds.meta.get("grid", {}).get("spacing", 1.6))
    print(f"{len(ds)} cases, grid spacing {spacing} mm")

    # ---------------- posterior geometry ----------------
    mu, cases = encode_all(model, ds, device)
    print(f"\nposterior means: {mu.shape}")
    print(f"  per-dim std over dataset: mean={mu.std(0).mean():.3f} "
          f"min={mu.std(0).min():.3f} max={mu.std(0).max():.3f}")
    print(f"  ||mu|| mean={np.linalg.norm(mu, axis=1).mean():.2f} "
          f"(a well-matched N(0,I) posterior would give ~{np.sqrt(cfg.latent_dim):.1f})")

    mu_c = mu - mu.mean(0)
    U, S, Vt = np.linalg.svd(mu_c, full_matrices=False)
    var = S ** 2 / max(1, len(mu) - 1)
    ratio = var / var.sum()
    print(f"  PCA explained variance, first 10: {np.round(ratio[:10], 3).tolist()}")
    print(f"  components for 90% variance: {int(np.searchsorted(np.cumsum(ratio), 0.90) + 1)}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(np.cumsum(ratio)[:40], marker="o", ms=3)
    ax[0].axhline(0.9, ls="--", c="r", lw=1)
    ax[0].set_xlabel("component"); ax[0].set_ylabel("cumulative explained variance")
    ax[0].set_title("latent PCA spectrum"); ax[0].grid(alpha=0.3)
    ax[1].scatter(U[:, 0] * S[0], U[:, 1] * S[1], s=28)
    for i, c in enumerate(cases):
        ax[1].annotate(c[-4:], (U[i, 0] * S[0], U[i, 1] * S[1]), fontsize=6)
    ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
    ax[1].set_title("cases in latent space"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "latent_pca.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out/'latent_pca.png'}")

    # ---------------- prior samples ----------------
    for temp in (0.6, 0.8, 1.0):
        z = torch.randn(args.n_samples, cfg.latent_dim) * temp
        cts, probs = decode_grid(model, z, device)
        strip_figure(cts, probs, [f"T={temp} #{i}" for i in range(args.n_samples)],
                     out / f"prior_samples_T{temp}.png")

    # ---------------- interpolation ----------------
    i, j = 0, min(1, len(ds) - 1)
    t = torch.linspace(0, 1, args.n_interp).view(-1, 1)
    za, zb = torch.tensor(mu[i]), torch.tensor(mu[j])
    z = (1 - t) * za + t * zb
    cts, probs = decode_grid(model, z, device)
    strip_figure(cts, probs,
                 [f"{cases[i]}" if k == 0 else (f"{cases[j]}" if k == args.n_interp - 1
                  else f"t={float(t[k]):.2f}") for k in range(args.n_interp)],
                 out / "interpolation.png")

    # ---------------- PC traversals ----------------
    for pc in range(min(args.n_pcs, Vt.shape[0])):
        sd = np.sqrt(var[pc])
        offs = np.linspace(-2.5, 2.5, 5)
        z = torch.tensor(np.stack([mu.mean(0) + o * sd * Vt[pc] for o in offs]),
                         dtype=torch.float32)
        cts, probs = decode_grid(model, z, device)
        strip_figure(cts, probs, [f"PC{pc+1} {o:+.1f}σ" for o in offs],
                     out / f"pc{pc+1}_traversal.png")

    # ---------------- NRRD export ----------------
    if args.export_nrrd > 0:
        z = torch.randn(args.export_nrrd, cfg.latent_dim) * 0.8
        cts, probs = decode_grid(model, z, device)
        summary = {}
        for k in range(args.export_nrrd):
            summary[f"sample_{k}"] = export_nrrd(
                cts[k, 0].numpy(), probs[k].numpy(),
                out / "nrrd", f"sample_{k}", spacing)
        print("\nNRRD export, generated organ volumes (voxels):")
        for k, v in summary.items():
            print(f"  {k}: " + "  ".join(f"{n[:8]}={c}" for n, c in v.items()))
        (out / "nrrd" / "summary.json").write_text(json.dumps(summary, indent=2))

    (out / "latent_stats.json").write_text(json.dumps({
        "cases": cases,
        "explained_variance_ratio": ratio.tolist(),
        "mu_norm_mean": float(np.linalg.norm(mu, axis=1).mean()),
        "latent_dim": cfg.latent_dim,
        "epoch": int(ck["epoch"]),
    }, indent=2))
    print("\ndone")


if __name__ == "__main__":
    main()
