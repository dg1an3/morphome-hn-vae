"""Survey the PDDCA head-and-neck NRRD corpus: geometry, intensity range, label coverage.

Run once before preprocessing so the resampling grid is chosen from data, not guesswork.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk

STRUCTURES = [
    "BrainStem",
    "Chiasm",
    "Mandible",
    "OpticNerve_L",
    "OpticNerve_R",
    "Parotid_L",
    "Parotid_R",
    "Submandibular_L",
    "Submandibular_R",
]


def probe_case(case_dir: Path) -> dict:
    img_path = case_dir / "img.nrrd"
    img = sitk.ReadImage(str(img_path))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)

    rec = {
        "case": case_dir.name,
        "size_xyz": list(img.GetSize()),
        "spacing_xyz": [round(s, 4) for s in img.GetSpacing()],
        "origin_xyz": [round(o, 2) for o in img.GetOrigin()],
        "direction": [round(d, 3) for d in img.GetDirection()],
        "pixel_type": img.GetPixelIDTypeAsString(),
        "hu_min": float(arr.min()),
        "hu_max": float(arr.max()),
        "hu_p01": float(np.percentile(arr, 0.1)),
        "hu_p99": float(np.percentile(arr, 99.9)),
        "extent_mm_xyz": [round(n * s, 1) for n, s in zip(img.GetSize(), img.GetSpacing())],
    }

    structs = {}
    for name in STRUCTURES:
        p = case_dir / "structures" / f"{name}.nrrd"
        if not p.exists():
            structs[name] = None
            continue
        m = sitk.ReadImage(str(p))
        marr = sitk.GetArrayFromImage(m)
        nvox = int((marr > 0).sum())
        entry = {
            "n_voxels": nvox,
            "size_matches_img": list(m.GetSize()) == list(img.GetSize()),
            "spacing_matches_img": np.allclose(m.GetSpacing(), img.GetSpacing(), atol=1e-4),
            "max_value": int(marr.max()),
        }
        if nvox:
            idx = np.argwhere(marr > 0)  # (z, y, x)
            entry["bbox_zyx_min"] = idx.min(0).tolist()
            entry["bbox_zyx_max"] = idx.max(0).tolist()
        structs[name] = entry
    rec["structures"] = structs

    # Union bounding box over all available structures, in physical mm,
    # so we can size a canonical crop that always contains the OARs.
    zs, ys, xs = [], [], []
    for name, e in structs.items():
        if e and e.get("n_voxels"):
            zs += [e["bbox_zyx_min"][0], e["bbox_zyx_max"][0]]
            ys += [e["bbox_zyx_min"][1], e["bbox_zyx_max"][1]]
            xs += [e["bbox_zyx_min"][2], e["bbox_zyx_max"][2]]
    if zs:
        sx, sy, sz = img.GetSpacing()
        rec["oar_bbox_vox_zyx"] = [[min(zs), min(ys), min(xs)], [max(zs), max(ys), max(xs)]]
        rec["oar_extent_mm_xyz"] = [
            round((max(xs) - min(xs) + 1) * sx, 1),
            round((max(ys) - min(ys) + 1) * sy, 1),
            round((max(zs) - min(zs) + 1) * sz, 1),
        ]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\datasets\medical\miccai_hn_sharpe")
    ap.add_argument("--out", default="notes/dataset_probe.json")
    args = ap.parse_args()

    root = Path(args.root)
    cases = sorted(d for d in root.iterdir() if d.is_dir())
    records = []
    for i, c in enumerate(cases, 1):
        rec = probe_case(c)
        records.append(rec)
        print(
            f"[{i:2d}/{len(cases)}] {rec['case']:>10}  size={rec['size_xyz']}  "
            f"spacing={rec['spacing_xyz']}  HU=[{rec['hu_min']:.0f},{rec['hu_max']:.0f}]",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2))

    # --- summary ---
    print("\n================ SUMMARY ================")
    print(f"cases: {len(records)}")

    sp = np.array([r["spacing_xyz"] for r in records])
    print("\nspacing x/y (in-plane): min=%.4f max=%.4f" % (sp[:, 0].min(), sp[:, 0].max()))
    print("spacing z (slice):      min=%.4f max=%.4f" % (sp[:, 2].min(), sp[:, 2].max()))
    print("unique z spacings:", sorted(Counter(sp[:, 2].round(3)).items()))

    sz = np.array([r["size_xyz"] for r in records])
    print("\nsize x:", sorted(Counter(sz[:, 0]).items()))
    print("size y:", sorted(Counter(sz[:, 1]).items()))
    print("size z: min=%d max=%d median=%d" % (sz[:, 2].min(), sz[:, 2].max(), np.median(sz[:, 2])))

    ext = np.array([r["extent_mm_xyz"] for r in records])
    print("\nphysical extent mm  x: %.0f..%.0f  y: %.0f..%.0f  z: %.0f..%.0f"
          % (ext[:, 0].min(), ext[:, 0].max(), ext[:, 1].min(), ext[:, 1].max(),
             ext[:, 2].min(), ext[:, 2].max()))

    oar = np.array([r["oar_extent_mm_xyz"] for r in records if "oar_extent_mm_xyz" in r])
    print("\nOAR union bbox extent mm  x: %.0f..%.0f  y: %.0f..%.0f  z: %.0f..%.0f"
          % (oar[:, 0].min(), oar[:, 0].max(), oar[:, 1].min(), oar[:, 1].max(),
             oar[:, 2].min(), oar[:, 2].max()))
    print("OAR extent 95th pct:", np.percentile(oar, 95, axis=0).round(0).tolist())

    print("\nstructure availability / voxel counts:")
    for name in STRUCTURES:
        present = [r["structures"][name] for r in records if r["structures"][name] is not None]
        nonempty = [e["n_voxels"] for e in present if e["n_voxels"] > 0]
        bad_geom = sum(1 for e in present if not (e["size_matches_img"] and e["spacing_matches_img"]))
        print(f"  {name:<18} files={len(present):2d}/{len(records)}  nonempty={len(nonempty):2d}  "
              f"vox median={int(np.median(nonempty)) if nonempty else 0:>7}  "
              f"min={min(nonempty) if nonempty else 0:>6}  max={max(nonempty) if nonempty else 0:>7}  "
              f"geom_mismatch={bad_geom}")

    hu = np.array([[r["hu_min"], r["hu_max"]] for r in records])
    print("\nHU min across cases: %.0f..%.0f ; HU max: %.0f..%.0f"
          % (hu[:, 0].min(), hu[:, 0].max(), hu[:, 1].min(), hu[:, 1].max()))

    dirs = Counter(tuple(r["direction"]) for r in records)
    print("\ndirection cosines (distinct):")
    for d, n in dirs.items():
        print(f"  n={n}: {d}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
