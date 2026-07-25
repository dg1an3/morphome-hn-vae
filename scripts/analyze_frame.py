"""Measure, in physical mm relative to the anatomical anchor, how big the crop
must be to contain every organ in every case -- and where it must be centred.

The first cache used a 192 mm cube centred on the BrainStem+Parotid centroid.
QC showed the Mandible clipping in ~24/48 cases, which means the anchor is
offset from the true OAR bounding box centre (mandible is anterior+inferior to
the brainstem). This script derives the correct offset and size from data.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from morphome.constants import ANCHOR_STRUCTURES, N_STRUCT, STRUCTURES


def centroid_and_bbox_mm(mask_img):
    arr = sitk.GetArrayFromImage(mask_img) > 0
    if not arr.any():
        return None, None, None
    idx = np.argwhere(arr)  # z, y, x
    def to_mm(izyx):
        return np.asarray(mask_img.TransformContinuousIndexToPhysicalPoint(
            [float(izyx[2]), float(izyx[1]), float(izyx[0])]))
    cen = to_mm(idx.mean(0))
    lo = to_mm(idx.min(0))
    hi = to_mm(idx.max(0))
    return cen, np.minimum(lo, hi), np.maximum(lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\datasets\medical\miccai_hn_sharpe")
    ap.add_argument("--out", default="notes/frame_analysis.json")
    args = ap.parse_args()

    root = Path(args.root)
    cases = sorted(d for d in root.iterdir() if d.is_dir())

    recs = []
    for c in cases:
        anchors = []
        for name in ANCHOR_STRUCTURES:
            p = c / "structures" / f"{name}.nrrd"
            if p.exists():
                cen, _, _ = centroid_and_bbox_mm(sitk.ReadImage(str(p)))
                if cen is not None:
                    anchors.append(cen)
        anchor = np.mean(anchors, axis=0)

        per = {}
        los, his = [], []
        for name in STRUCTURES:
            p = c / "structures" / f"{name}.nrrd"
            if not p.exists():
                per[name] = None
                continue
            cen, lo, hi = centroid_and_bbox_mm(sitk.ReadImage(str(p)))
            if cen is None:
                per[name] = None
                continue
            per[name] = {"cen_rel": (cen - anchor).tolist(),
                         "lo_rel": (lo - anchor).tolist(),
                         "hi_rel": (hi - anchor).tolist()}
            los.append(lo - anchor)
            his.append(hi - anchor)
        recs.append({"case": c.name, "anchor": anchor.tolist(), "per": per,
                     "union_lo_rel": np.min(los, 0).tolist(),
                     "union_hi_rel": np.max(his, 0).tolist()})
        print(f"{c.name}  union_rel lo={np.min(los,0).round(0)} hi={np.max(his,0).round(0)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(recs, indent=2))

    lo = np.array([r["union_lo_rel"] for r in recs])
    hi = np.array([r["union_hi_rel"] for r in recs])
    ax = ["x (L-R)", "y (A-P)", "z (I-S)"]

    print("\n=========== union OAR bbox relative to anchor (mm) ===========")
    for a in range(3):
        print(f"{ax[a]:>10}:  lo min={lo[:,a].min():8.1f} med={np.median(lo[:,a]):8.1f}   "
              f"hi max={hi[:,a].max():8.1f} med={np.median(hi[:,a]):8.1f}   "
              f"span max={(hi[:,a]-lo[:,a]).max():6.1f}")

    print("\nTo contain EVERY case with the anchor-centred axis-aligned cube:")
    need_lo = lo.min(0)
    need_hi = hi.max(0)
    print("  required lo (mm):", need_lo.round(1).tolist())
    print("  required hi (mm):", need_hi.round(1).tolist())
    print("  required centre offset from anchor (mm):", ((need_hi + need_lo) / 2).round(1).tolist())
    print("  required side length (mm):", (need_hi - need_lo).round(1).tolist())

    print("\nSame, but ignoring the single worst case per axis (robust):")
    rlo = np.percentile(lo, 2, axis=0)
    rhi = np.percentile(hi, 98, axis=0)
    print("  offset:", ((rhi + rlo) / 2).round(1).tolist(),
          " side:", (rhi - rlo).round(1).tolist())

    print("\nper-structure centroid offset from anchor (median mm) and spread (std):")
    for name in STRUCTURES:
        v = np.array([r["per"][name]["cen_rel"] for r in recs if r["per"][name]])
        print(f"  {name:<18} n={len(v):2d}  med={np.median(v,0).round(1)}  std={v.std(0).round(1)}")

    print("\nper-structure extreme extent from anchor (min lo / max hi over cases):")
    for name in STRUCTURES:
        l = np.array([r["per"][name]["lo_rel"] for r in recs if r["per"][name]])
        h = np.array([r["per"][name]["hi_rel"] for r in recs if r["per"][name]])
        print(f"  {name:<18} lo={l.min(0).round(0)}  hi={h.max(0).round(0)}")


if __name__ == "__main__":
    main()
