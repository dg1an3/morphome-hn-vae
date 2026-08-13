"""Verify the LUNG1 ingest: are the masks actually on the anatomy they name?

The ingest places each SEG frame by its patient-space origin, using the CT grid
that SimpleITK built. When a series has missing slices, SimpleITK still reports a
single uniform spacing (it warns: "Non uniform sampling or missing slices
detected"), so a frame past the gap can round to the wrong index. Nothing about
the file sizes or shapes would look wrong afterwards.

Mean HU inside each mask is the test that catches it, because it is anatomy, not
bookkeeping: lung parenchyma is air-filled and sits near -700 HU, and no
misplacement of a lung mask survives that check. Structures are scored against
plausible HU windows and volume ranges; anything outside is reported per case.

`--src` additionally reads the source DICOM slice positions to report which
series are non-uniform, so a failure can be attributed rather than guessed at.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# (min_hu, max_hu, min_mL, max_mL) -- deliberately wide; this catches gross
# misplacement, not contour quality.
EXPECTED = {
    "Lungs":      (-900, -400,  1000, 9000),
    "Lung_L":     (-900, -400,   400, 5000),
    "Lung_R":     (-900, -400,   400, 5000),
    "Heart":      (  -50,  200,   300, 1600),
    "Esophagus":  ( -200,  150,     5,  200),
    "SpinalCord": (  -50,  150,     5,  200),
    "GTV":        ( -900,  300,   0.1, 2000),
}


def slice_uniformity(ct_series_dir: Path) -> float | None:
    """Max ratio between consecutive slice gaps; 1.0 is perfectly uniform."""
    import pydicom
    zs = []
    for f in ct_series_dir.glob("*.dcm"):
        try:
            d = pydicom.dcmread(f, stop_before_pixels=True)
            zs.append(float(d.ImagePositionPatient[2]))
        except Exception:
            continue
    if len(zs) < 3:
        return None
    gaps = np.diff(np.sort(np.asarray(zs)))
    gaps = gaps[gaps > 1e-6]
    return float(gaps.max() / gaps.min()) if len(gaps) else None


def find_ct_dir(patient_dir: Path) -> Path | None:
    import pydicom
    best, n = None, 0
    for study in (p for p in patient_dir.iterdir() if p.is_dir()):
        for series in (p for p in study.iterdir() if p.is_dir()):
            files = sorted(series.glob("*.dcm"))
            if len(files) <= n:
                continue
            try:
                if pydicom.dcmread(files[0], stop_before_pixels=True).Modality == "CT":
                    best, n = series, len(files)
            except Exception:
                continue
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\datasets\medical\lung1_nrrd")
    ap.add_argument("--src", default="", help="source DICOM root, to attribute failures")
    ap.add_argument("--out", default="notes/lung1_qc.json")
    args = ap.parse_args()

    root = Path(args.root)
    cases = sorted(p for p in root.iterdir() if p.is_dir())
    print(f"{len(cases)} ingested cases in {root}")

    records, bad = {}, []
    for i, case in enumerate(cases):
        ct = sitk.ReadImage(str(case / "img.nrrd"))
        a = sitk.GetArrayFromImage(ct)
        mL = float(np.prod(ct.GetSpacing())) / 1000.0
        rec, flags = {}, []
        for f in sorted((case / "structures").glob("*.nrrd")):
            m = sitk.GetArrayFromImage(sitk.ReadImage(str(f))).astype(bool)
            if not m.any():
                flags.append(f"{f.stem}: empty")
                continue
            hu, vol = float(a[m].mean()), float(m.sum()) * mL
            rec[f.stem] = {"mean_hu": round(hu, 1), "mL": round(vol, 1)}
            lo_hu, hi_hu, lo_v, hi_v = EXPECTED.get(f.stem, (-1e9, 1e9, 0, 1e9))
            if not lo_hu <= hu <= hi_hu:
                flags.append(f"{f.stem}: mean HU {hu:.0f} outside [{lo_hu},{hi_hu}]")
            if not lo_v <= vol <= hi_v:
                flags.append(f"{f.stem}: {vol:.0f} mL outside [{lo_v},{hi_v}]")
        records[case.name] = {"structures": rec, "flags": flags}
        if flags:
            bad.append(case.name)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(cases)}]", flush=True)

    # Attribute failures to non-uniform source series where possible.
    if args.src:
        src = Path(args.src)
        for name in bad:
            d = src / name
            if d.is_dir():
                ct_dir = find_ct_dir(d)
                if ct_dir is not None:
                    records[name]["slice_uniformity"] = slice_uniformity(ct_dir)

    print(f"\n{len(cases) - len(bad)} clean, {len(bad)} flagged")
    for name in bad[:25]:
        u = records[name].get("slice_uniformity")
        tag = f"  [gap ratio {u:.2f}]" if u else ""
        print(f"  {name}{tag}")
        for f in records[name]["flags"]:
            print(f"      {f}")

    # Population summary of the structures that carry the alignment signal.
    print("\nmean HU by structure (population):")
    for s in EXPECTED:
        vals = [r["structures"][s]["mean_hu"] for r in records.values()
                if s in r["structures"]]
        if vals:
            v = np.asarray(vals)
            print(f"  {s:11s} n={len(v):4d}  mean {v.mean():8.1f}  "
                  f"p05 {np.percentile(v,5):8.1f}  p95 {np.percentile(v,95):8.1f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_cases": len(cases), "n_flagged": len(bad),
                               "flagged": bad, "cases": records}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
