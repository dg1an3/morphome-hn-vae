"""Choose the crop frame for a dose-capable cache, from measured per-slice contours.

The existing frame was sized around the union OAR bounding box, which truncates
the patient posteriorly in ~76% of transverse slices -- fatal for dose, which
needs a closed external contour in every slice.

This measures, per case, the patient's in-plane bounding box for every transverse
slice as a function of height above the anchor, then searches offsets for a
candidate grid and reports how many slices would still be clipped. Two distinct
failure modes are separated, because only one is fixable:

  * clipped by OUR box   -> fix by moving/enlarging the frame
  * clipped by the SCAN  -> the reconstruction FOV itself cuts the patient
                            (10-12 of 48 cases laterally, at shoulder level);
                            no crop recovers this, it can only be avoided by
                            keeping the frame above the shoulders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from morphome.constants import ANCHOR_STRUCTURES
from analyze_body_extent import centroid_mm, patient_mask


def profile_case(ct_path: Path) -> dict | None:
    """Per-transverse-slice in-plane extent, in mm relative to the raw anchor."""
    case_dir = ct_path.parent
    cents = []
    for name in ANCHOR_STRUCTURES:
        p = case_dir / "structures" / f"{name}.nrrd"
        if p.exists():
            c = centroid_mm(p)
            if c is not None:
                cents.append(c)
    if not cents:
        return None
    anchor = np.mean(cents, axis=0)          # raw centroid, no offset applied

    img = sitk.ReadImage(str(ct_path))
    hu = sitk.GetArrayFromImage(img).astype(np.float32)
    m = patient_mask(hu)
    sp, org = np.array(img.GetSpacing()), np.array(img.GetOrigin())
    xs = org[0] + np.arange(hu.shape[2]) * sp[0] - anchor[0]
    ys = org[1] + np.arange(hu.shape[1]) * sp[1] - anchor[1]
    zs = org[2] + np.arange(hu.shape[0]) * sp[2] - anchor[2]

    rows = []
    for k in range(hu.shape[0]):
        s = m[k]
        if not s.any():
            continue
        cols, rws = s.any(0), s.any(1)
        rows.append({
            "z": float(zs[k]),
            "x0": float(xs[cols.argmax()]), "x1": float(xs[len(cols) - 1 - cols[::-1].argmax()]),
            "y0": float(ys[rws.argmax()]), "y1": float(ys[len(rws) - 1 - rws[::-1].argmax()]),
            "fov_x": bool(s[:, 0].any() or s[:, -1].any()),
            "fov_y": bool(s[0].any() or s[-1].any()),
        })
    return {"case": case_dir.name, "slices": rows}


def evaluate(profiles, dims, spacing, off):
    """Fraction of in-window slices whose contour is clipped, by cause."""
    half = np.array(dims) * spacing / 2.0
    ours = scan = total = 0
    per_case = []
    for p in profiles:
        c_ours = c_scan = c_tot = 0
        for s in p["slices"]:
            if not (off[2] - half[2] <= s["z"] <= off[2] + half[2]):
                continue
            c_tot += 1
            inside = (s["x0"] >= off[0] - half[0] and s["x1"] <= off[0] + half[0] and
                      s["y0"] >= off[1] - half[1] and s["y1"] <= off[1] + half[1])
            if s["fov_x"] or s["fov_y"]:
                c_scan += 1
            elif not inside:
                c_ours += 1
        ours += c_ours; scan += c_scan; total += c_tot
        per_case.append({"case": p["case"], "slices": c_tot,
                         "clipped_by_box": c_ours, "clipped_by_scan": c_scan})
    return {"total_slices": total, "clipped_by_box": ours, "clipped_by_scan": scan,
            "per_case": per_case}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\datasets\medical\miccai_hn_sharpe")
    ap.add_argument("--profiles", default="notes/body_extent/slice_profiles.json")
    ap.add_argument("--dims", type=int, nargs=3, default=[224, 160, 96])
    ap.add_argument("--spacing", type=float, default=2.5)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    pf = Path(args.profiles)
    if pf.exists() and not args.recompute:
        profiles = json.loads(pf.read_text())
        print(f"loaded {len(profiles)} slice profiles from {pf}")
    else:
        cases = sorted(Path(args.root).rglob("img.nrrd"))
        profiles = []
        for i, c in enumerate(cases):
            r = profile_case(c)
            if r:
                profiles.append(r)
            print(f"  [{i+1}/{len(cases)}] {c.parent.name}  "
                  f"{len(r['slices']) if r else 0} slices", flush=True)
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text(json.dumps(profiles))
        print(f"wrote {pf}")

    dims, sp = args.dims, args.spacing
    print(f"\ngrid {dims[0]}x{dims[1]}x{dims[2]} @ {sp} mm = "
          f"{dims[0]*sp:.0f} x {dims[1]*sp:.0f} x {dims[2]*sp:.0f} mm")

    best = None
    for oy in range(-20, 121, 5):
        for oz in range(-80, 41, 5):
            r = evaluate(profiles, dims, sp, (0.0, float(oy), float(oz)))
            key = (r["clipped_by_box"], r["clipped_by_scan"])
            if best is None or key < best[0]:
                best = (key, (0.0, float(oy), float(oz)), r)
    (_, off, r) = best
    print(f"\nbest offset (x, y, z) = ({off[0]:.0f}, {off[1]:.0f}, {off[2]:.0f}) mm")
    print(f"  slices in window        {r['total_slices']}")
    print(f"  clipped by OUR box      {r['clipped_by_box']}  "
          f"({r['clipped_by_box']/max(1,r['total_slices'])*100:.2f}%)")
    print(f"  clipped by the SCAN FOV {r['clipped_by_scan']}  "
          f"({r['clipped_by_scan']/max(1,r['total_slices'])*100:.2f}%)")
    bad = [c for c in r["per_case"] if c["clipped_by_scan"]]
    print(f"  cases with any scan-FOV clipping: {len(bad)}/{len(r['per_case'])}")

    cur = evaluate(profiles, [128, 128, 128], 1.6, (0.0, -37.0, -8.0))
    print(f"\nfor comparison, the current frame (128^3 @ 1.6 mm, offset 0,-37,-8):")
    print(f"  clipped by OUR box      {cur['clipped_by_box']}/{cur['total_slices']}  "
          f"({cur['clipped_by_box']/max(1,cur['total_slices'])*100:.1f}%)")

    Path("notes/body_extent/frame_choice.json").write_text(json.dumps(
        {"dims": dims, "spacing": sp, "offset_mm": off, "result": r,
         "current_frame": cur}, indent=2))


if __name__ == "__main__":
    main()
