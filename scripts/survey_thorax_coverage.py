"""Does a thorax CT reach far enough superiorly to be usable for a head-and-neck
anatomy model?

The morphome H&N frame is anchored on the centroid of BrainStem + both parotids.
A scan is only usable if it actually contains that anatomy. Rather than assume,
this samples series, measures the craniocaudal extent, and locates the lung
apex -- then reports how much anatomy exists ABOVE the apex, which is where the
H&N organs live.

Reference anatomy (adult, approximate, relative to the lung apex):
  lung apex        0 mm
  hyoid         ~ +55 mm
  mandible base ~ +75 mm
  parotid centre~ +115 mm
  brainstem     ~ +150 mm
  orbits/chiasm ~ +180 mm
The H&N canonical cube needs roughly +40 mm to +190 mm above the apex.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

# Above the lung apex, this much extra coverage is needed for the H&N frame.
NEED_ABOVE_APEX_MM = 190.0


def find_series_dirs(root: Path, limit: int, seed: int = 0) -> list[Path]:
    """Directories that directly contain .dcm files."""
    pats = sorted(p for p in root.iterdir() if p.is_dir())
    rng = random.Random(seed)
    rng.shuffle(pats)
    out = []
    for p in pats:
        if len(out) >= limit:
            break
        best = None
        for d in [p, *(x for x in p.rglob("*") if x.is_dir())]:
            n = len(list(d.glob("*.dcm")))
            if n > 40 and (best is None or n > best[1]):
                best = (d, n)
        if best:
            out.append(best[0])
    return out


def read_series(d: Path):
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(d))
    if not ids:
        return None
    best, bestn = None, 0
    for sid in ids:
        files = reader.GetGDCMSeriesFileNames(str(d), sid)
        if len(files) > bestn:
            best, bestn = files, len(files)
    if not best or bestn < 40:
        return None
    reader.SetFileNames(best)
    return reader.Execute()


def lung_apex_index(arr: np.ndarray) -> int | None:
    """Most superior slice index (in z-increasing-superior order) containing a
    substantial air pocket enclosed by body -- i.e. the lung apex."""
    d = arr.shape[0]
    body = arr > -500
    air_in_body = (arr < -700)
    for z in range(d - 1, -1, -1):
        b = body[z]
        if b.sum() < 500:
            continue
        # air voxels that are interior to the body cross-section
        from scipy import ndimage
        filled = ndimage.binary_fill_holes(b)
        interior_air = air_in_body[z] & filled & ~b
        if interior_air.sum() > 400:
            return z
    return None


def analyse(img: sitk.Image) -> dict:
    arr = sitk.GetArrayFromImage(img).astype(np.int16)  # (z,y,x) in acquisition order
    sp = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()

    # Ensure z increases superiorly.
    zsign = 1.0 if direction[8] >= 0 else -1.0
    if zsign < 0:
        arr = arr[::-1]
    z_mm = arr.shape[0] * sp[2]

    apex = lung_apex_index(arr)
    above = (arr.shape[0] - 1 - apex) * sp[2] if apex is not None else None

    return {
        "size": list(img.GetSize()),
        "spacing": [round(s, 3) for s in sp],
        "z_extent_mm": round(z_mm, 1),
        "apex_index": apex,
        "mm_above_apex": round(above, 1) if above is not None else None,
        "arr": arr,
        "sp": sp,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="nsclc,nlst")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="notes/thorax_coverage")
    args = ap.parse_args()

    roots = {
        "nsclc": Path(r"E:\datasets\medical\nsclc-radiomics\NSCLC-Radiomics"),
        "nlst": Path(r"E:\datasets\medical\manifest-NLST_allCT\NLST"),
        "lymph": Path(r"E:\datasets\medical\CT Lymph Nodes\CT Lymph Nodes"),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    panels = []
    for key in [s.strip() for s in args.sets.split(",")]:
        root = roots.get(key)
        if not root or not root.exists():
            print(f"skip {key}: {root} not found")
            continue
        dirs = find_series_dirs(root, args.n)
        print(f"\n=== {key}: {len(dirs)} series sampled from {root}")
        for d in dirs:
            try:
                img = read_series(d)
                if img is None:
                    print(f"  {d.name}: no usable series")
                    continue
                r = analyse(img)
            except Exception as e:
                print(f"  {d.name}: FAILED {type(e).__name__}: {e}")
                continue
            arr, sp = r.pop("arr"), r.pop("sp")
            r["dataset"] = key
            r["dir"] = str(d)
            usable = (r["mm_above_apex"] is not None
                      and r["mm_above_apex"] >= NEED_ABOVE_APEX_MM)
            r["hn_usable"] = bool(usable)
            records.append(r)
            print(f"  {d.parent.name[:22]:<22} size={r['size']} sp={r['spacing']} "
                  f"z={r['z_extent_mm']}mm apex@{r['apex_index']} "
                  f"above_apex={r['mm_above_apex']}mm  HN_usable={usable}")

            # coronal slice through the middle for the montage
            mid = arr.shape[1] // 2
            cor = arr[:, mid, :][::-1]
            # rescale to display aspect
            zoom_y = sp[2] / sp[0]
            panels.append((f"{key}/{d.parent.name[:14]}\n"
                           f"z={r['z_extent_mm']:.0f}mm above_apex="
                           f"{r['mm_above_apex']}mm", cor, zoom_y,
                           r["apex_index"], arr.shape[0]))

    if panels:
        n = len(panels)
        cols = min(6, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 4.6 * rows),
                                 squeeze=False)
        for i, (title, cor, zoom_y, apex, nz) in enumerate(panels):
            ax = axes[i // cols][i % cols]
            ax.imshow(np.clip(cor, -1000, 500), cmap="gray", aspect=zoom_y)
            if apex is not None:
                ax.axhline(nz - 1 - apex, color="lime", lw=1.2)
            ax.set_title(title, fontsize=7)
            ax.axis("off")
        for j in range(n, rows * cols):
            axes[j // cols][j % cols].axis("off")
        fig.tight_layout()
        p = out / "coronal_montage.png"
        fig.savefig(p, dpi=110, bbox_inches="tight")
        print(f"\nwrote {p}  (green line = detected lung apex)")

    (out / "coverage.json").write_text(json.dumps(records, indent=2))

    if records:
        ok = [r for r in records if r["hn_usable"]]
        above = [r["mm_above_apex"] for r in records if r["mm_above_apex"] is not None]
        print("\n================ SUMMARY ================")
        print(f"series analysed: {len(records)}")
        if above:
            print(f"mm above lung apex: min={min(above):.0f} median={np.median(above):.0f} "
                  f"max={max(above):.0f}")
        print(f"H&N-usable (>= {NEED_ABOVE_APEX_MM:.0f} mm above apex): "
              f"{len(ok)}/{len(records)}")


if __name__ == "__main__":
    main()
