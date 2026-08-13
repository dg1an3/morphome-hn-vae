"""How large must the canonical frame be to contain the whole patient?

`analyze_frame.py` sized the current frame around the *union OAR bounding box*,
which is the right target for a segmentation model and the wrong one for dose:
dose calculation needs the complete external contour in every transverse slice,
because path length and scatter depend on tissue that no OAR occupies.

The consequence is measurable in the cache — ~76 % of transverse slices have
their posterior body edge lying on a single plane, i.e. cut flat — so this script
goes back to the *source* NRRDs and measures the true body extent relative to the
same anchor, in LPS mm. Output is the box the frame would need in order to stop
truncating, plus what that costs in voxels at a range of spacings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy import ndimage

from morphome.constants import ANCHOR_OFFSET_MM, ANCHOR_STRUCTURES, BODY_HU_THRESHOLD


def patient_mask(hu: np.ndarray, erode: int = 3) -> np.ndarray:
    """Patient tissue with the couch and immobilisation shell separated off.

    `preprocess.body_mask` takes the largest connected component, which works on
    the cropped volume where the couch is usually disconnected. On the *full*
    source volume the patient rests on the couch and the two merge into one
    component -- measuring that gives a 575 mm "body", i.e. shoulders plus table.

    The contact is thin, so eroding before the connected-component step splits
    them, and dilating back restores the true surface.
    """
    m = hu > BODY_HU_THRESHOLD
    core = ndimage.binary_erosion(m, iterations=erode, border_value=0)
    lab, n = ndimage.label(core)
    if n == 0:
        return ndimage.binary_fill_holes(m)
    core = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    grown = ndimage.binary_dilation(core, iterations=erode) & m
    return ndimage.binary_fill_holes(grown)


def centroid_mm(path: Path) -> np.ndarray | None:
    img = sitk.ReadImage(str(path))
    a = sitk.GetArrayFromImage(img)
    idx = np.argwhere(a > 0)
    if idx.size == 0:
        return None
    # (z,y,x) -> (x,y,z) index order for TransformContinuousIndexToPhysicalPoint
    mean_zyx = idx.mean(0)
    return np.array(img.TransformContinuousIndexToPhysicalPoint(
        [float(mean_zyx[2]), float(mean_zyx[1]), float(mean_zyx[0])]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\datasets\medical\miccai_hn_sharpe")
    ap.add_argument("--out", default="notes/body_extent")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    cases = sorted(p for p in root.rglob("img.nrrd"))
    if not cases:
        cases = sorted(p for p in root.rglob("*.nrrd") if p.name.lower().startswith("img"))
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} source volumes under {root}")

    rows = []
    for i, ct_path in enumerate(cases):
        case_dir = ct_path.parent
        cents = []
        for name in ANCHOR_STRUCTURES:
            p = case_dir / "structures" / f"{name}.nrrd"
            if p.exists():
                c = centroid_mm(p)
                if c is not None:
                    cents.append(c)
        if not cents:
            print(f"  {case_dir.name}: no anchor structures, skipped")
            continue
        anchor = np.mean(cents, axis=0) + np.asarray(ANCHOR_OFFSET_MM, float)

        img = sitk.ReadImage(str(ct_path))
        hu = sitk.GetArrayFromImage(img).astype(np.float32)
        body = patient_mask(hu)
        idx = np.argwhere(body)                       # (z,y,x)
        if idx.size == 0:
            continue
        lo_zyx, hi_zyx = idx.min(0), idx.max(0)
        lo = np.array(img.TransformContinuousIndexToPhysicalPoint(
            [float(lo_zyx[2]), float(lo_zyx[1]), float(lo_zyx[0])]))
        hi = np.array(img.TransformContinuousIndexToPhysicalPoint(
            [float(hi_zyx[2]), float(hi_zyx[1]), float(hi_zyx[0])]))
        lo_r, hi_r = np.minimum(lo, hi) - anchor, np.maximum(lo, hi) - anchor

        # Does the body touch the source FOV edge? If so the scan itself is
        # truncated and no crop box can recover it.
        touches = {
            "z_lo": bool(lo_zyx[0] == 0), "z_hi": bool(hi_zyx[0] == body.shape[0] - 1),
            "y_lo": bool(lo_zyx[1] == 0), "y_hi": bool(hi_zyx[1] == body.shape[1] - 1),
            "x_lo": bool(lo_zyx[2] == 0), "x_hi": bool(hi_zyx[2] == body.shape[2] - 1),
        }
        rows.append({"case": case_dir.name, "lo": lo_r.tolist(), "hi": hi_r.tolist(),
                     "touches_fov": touches})
        print(f"  [{i+1:2d}/{len(cases)}] {case_dir.name}  "
              f"x[{lo_r[0]:7.1f},{hi_r[0]:7.1f}] y[{lo_r[1]:7.1f},{hi_r[1]:7.1f}] "
              f"z[{lo_r[2]:7.1f},{hi_r[2]:7.1f}]"
              f"{'  FOV-LIMITED' if any(touches.values()) else ''}", flush=True)

    lo = np.array([r["lo"] for r in rows])
    hi = np.array([r["hi"] for r in rows])
    axes = "xyz"
    print("\nbody extent relative to the anchor, over all cases (mm):")
    for k in range(3):
        print(f"  {axes[k]}: [{lo[:,k].min():8.1f}, {hi[:,k].max():8.1f}]   "
              f"span {hi[:,k].max()-lo[:,k].min():7.1f}   "
              f"p95 span {np.percentile(hi[:,k],95)-np.percentile(lo[:,k],5):7.1f}")

    need = np.array([hi[:, k].max() - lo[:, k].min() for k in range(3)])
    print(f"\ncurrent frame: 128 voxels x 1.6 mm = 204.8 mm cube")
    print(f"required to contain every body: {need[0]:.0f} x {need[1]:.0f} x {need[2]:.0f} mm")
    for sp in (1.6, 2.0, 2.4, 3.0):
        vox = np.ceil(need / sp).astype(int)
        print(f"  at {sp:.1f} mm -> {vox[0]} x {vox[1]} x {vox[2]} voxels "
              f"(cube {int(np.ceil(need.max()/sp))}^3)")

    n_fov = sum(1 for r in rows if any(r["touches_fov"].values()))
    print(f"\n{n_fov}/{len(rows)} cases have body touching the source FOV edge "
          f"(unrecoverable by re-cropping)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "body_extent.json").write_text(json.dumps(
        {"cases": rows, "required_mm": need.tolist()}, indent=2))
    print(f"wrote {out/'body_extent.json'}")


if __name__ == "__main__":
    main()
