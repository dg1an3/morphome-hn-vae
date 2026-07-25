"""Quality control for the preprocessed cache.

Checks that (a) small OARs survived resampling, (b) labels sit inside the body
mask and inside the crop, (c) the crop is anatomically consistent across cases,
then renders an overlay montage for eyeball verification.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from morphome.constants import HU_MAX, HU_MIN, N_STRUCT, STRUCTURES
from morphome.preprocess import unpack_labels

# Distinguishable colours for 9 structures.
PALETTE = np.array([
    [0.90, 0.10, 0.10],  # BrainStem
    [1.00, 0.95, 0.20],  # Chiasm
    [0.20, 0.60, 1.00],  # Mandible
    [0.20, 1.00, 0.40],  # OpticNerve_L
    [0.00, 0.70, 0.25],  # OpticNerve_R
    [1.00, 0.45, 0.00],  # Parotid_L
    [0.85, 0.30, 0.00],  # Parotid_R
    [0.80, 0.30, 1.00],  # Submandibular_L
    [0.55, 0.10, 0.85],  # Submandibular_R
])


def load(path: Path):
    d = np.load(path, allow_pickle=True)
    ct = d["ct_hu"].astype(np.float32)
    lab = unpack_labels(d["labels_packed"])
    body = np.unpackbits(d["body"])[: ct.size].reshape(ct.shape).astype(bool)
    return ct, lab, body, d["presence"], d


def overlay(ct_slice, lab_slice, alpha=0.55):
    g = (np.clip(ct_slice, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
    rgb = np.repeat(g[..., None], 3, axis=-1)
    for i in range(N_STRUCT):
        m = lab_slice[i]
        if m.any():
            rgb[m] = (1 - alpha) * rgb[m] + alpha * PALETTE[i]
    return np.clip(rgb, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=r"E:\datasets\medical\morphome_cache\hn_128_1.5mm")
    ap.add_argument("--out", default="notes/qc")
    ap.add_argument("--n-montage", type=int, default=12)
    args = ap.parse_args()

    cache = Path(args.cache)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(cache.glob("0522c*.npz"))
    print(f"{len(files)} cached cases")

    rows = []
    problems = []
    for f in files:
        ct, lab, body, presence, _ = load(f)
        rec = {"case": f.stem, "hu_mean": float(ct[body].mean()) if body.any() else None,
               "body_frac": float(body.mean())}
        for i, name in enumerate(STRUCTURES):
            n = int(lab[i].sum())
            rec[name] = n
            if presence[i] and n == 0:
                problems.append(f"{f.stem}: {name} present in source but EMPTY after resample")
            if presence[i] and n > 0:
                # touching the crop face means the organ is clipped
                idx = np.argwhere(lab[i])
                lo, hi = idx.min(0), idx.max(0)
                if (lo == 0).any() or (hi == ct.shape[0] - 1).any():
                    problems.append(f"{f.stem}: {name} touches crop boundary (clipped)")
                outside = int((lab[i] & ~body).sum())
                if outside > 0.02 * n:
                    problems.append(
                        f"{f.stem}: {name} has {outside}/{n} voxels outside body mask")
        rows.append(rec)

    # ---- table ----
    print(f"\n{'structure':<18} {'cases':>6} {'median':>8} {'min':>7} {'max':>8}")
    for name in STRUCTURES:
        v = np.array([r[name] for r in rows if r[name] > 0])
        print(f"{name:<18} {len(v):>6} {int(np.median(v)):>8} {v.min():>7} {v.max():>8}")

    print("\n--- issues ---")
    if problems:
        for p in problems:
            print("  " + p)
    else:
        print("  none")
    print(f"total issues: {len(problems)}")

    (outdir / "qc_stats.json").write_text(json.dumps({"rows": rows, "problems": problems}, indent=2))

    # ---- montage: axial/coronal/sagittal through the label centroid ----
    sel = files[: args.n_montage]
    fig, axes = plt.subplots(len(sel), 3, figsize=(9, 3 * len(sel)))
    for r, f in enumerate(sel):
        ct, lab, body, presence, _ = load(f)
        any_lab = lab.any(0)
        c = np.argwhere(any_lab).mean(0).astype(int) if any_lab.any() else np.array(ct.shape) // 2
        views = [
            (overlay(ct[c[0]], lab[:, c[0]]), f"{f.stem} axial z={c[0]}"),
            (overlay(ct[:, c[1]][::-1], lab[:, :, c[1]][:, ::-1]), f"coronal y={c[1]}"),
            (overlay(ct[:, :, c[2]][::-1], lab[:, :, :, c[2]][:, ::-1]), f"sagittal x={c[2]}"),
        ]
        for k, (im, title) in enumerate(views):
            ax = axes[r, k]
            ax.imshow(im)
            ax.set_title(title, fontsize=7)
            ax.axis("off")
    fig.tight_layout()
    p = outdir / "montage.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    print(f"\nwrote {p}")

    # ---- population mean maps: is the crop anatomically consistent? ----
    acc_ct = np.zeros((128, 128, 128), dtype=np.float64)
    acc_lab = np.zeros((N_STRUCT, 128, 128, 128), dtype=np.float64)
    cnt_lab = np.zeros(N_STRUCT)
    for f in files:
        ct, lab, body, presence, _ = load(f)
        acc_ct += ct
        for i in range(N_STRUCT):
            if presence[i]:
                acc_lab[i] += lab[i]
                cnt_lab[i] += 1
    acc_ct /= len(files)
    for i in range(N_STRUCT):
        if cnt_lab[i]:
            acc_lab[i] /= cnt_lab[i]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    mid = 64
    prob = acc_lab.max(0)
    for k, (ctv, pv, t) in enumerate([
        (acc_ct[mid], prob[mid], "mean CT + label prob, axial"),
        (acc_ct[:, mid][::-1], prob[:, mid][::-1], "coronal"),
        (acc_ct[:, :, mid][::-1], prob[:, :, mid][::-1], "sagittal"),
    ]):
        g = (np.clip(ctv, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
        axes[k].imshow(g, cmap="gray")
        axes[k].imshow(np.ma.masked_where(pv < 0.05, pv), cmap="inferno", alpha=0.65, vmin=0, vmax=1)
        axes[k].set_title(t, fontsize=9)
        axes[k].axis("off")
    fig.tight_layout()
    p = outdir / "population_mean.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    print(f"wrote {p}")

    # Spatial agreement: how peaked is each structure's population probability?
    print("\nper-structure population overlap (max prob over grid, and volume of p>0.5 region):")
    for i, name in enumerate(STRUCTURES):
        print(f"  {name:<18} maxprob={acc_lab[i].max():.2f}  vox(p>0.5)={int((acc_lab[i] > 0.5).sum()):>6}")


if __name__ == "__main__":
    main()
