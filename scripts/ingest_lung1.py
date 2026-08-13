"""NSCLC-Radiomics (LUNG1) DICOM -> the PDDCA-style layout `preprocess.py` reads.

Writes `<out>/<case>/img.nrrd` plus `<out>/<case>/structures/<Name>.nrrd`, which is
exactly the contract `morphome.preprocess.preprocess_case` expects, so the existing
resample/pack pipeline runs against thorax without I/O changes.

**Structures come from the DICOM SEG objects, not the RTSTRUCT.** Both are present
for every patient, but they are not equivalent in quality:

* RTSTRUCT is the original planning contour set, and its ROI naming is clinical
  free-text: across 422 patients there are 55 distinct names, including `gtv-2`,
  `gtv_4r`, `gtv_10_li`, `gtv-supral`, `gtv_nr_7` and `gtv-pre-operat`. Only the
  first six names are used consistently.
* SEG is TCIA's curated conversion. It carries the same primary structures at the
  same per-patient counts, drops the ad-hoc nodal GTV variants, and codes anatomy
  properly -- laterality is in `SegmentedPropertyTypeModifierCodeSequence`, not the
  label, so the two segments both labelled "Lung" are unambiguous.

Left/right lungs and a combined "Lungs-Total" are mutually exclusive across
patients (312 have L/R, 97 have the union). Rather than leave a channel that means
different things for different cases, `Lungs` is always written: the union where
laterality exists, the combined mask where it does not. `Lung_L`/`Lung_R` are
additionally written for the 312 that have them, and simply absent for the rest --
the same per-case presence contract PDDCA already uses for uncontoured organs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

# (SegmentLabel, laterality modifier) -> canonical structure name. The modifier is
# "" for unpaired organs and "Left"/"Right"/"Right and left" for the lungs.
SEG_NAME_MAP = {
    ("Neoplasm, Primary", ""): "GTV",
    ("Spinal cord", ""): "SpinalCord",
    ("Esophagus", ""): "Esophagus",
    ("Heart", ""): "Heart",
    ("Lung", "Left"): "Lung_L",
    ("Lung", "Right"): "Lung_R",
    ("Lung", "Right and left"): "Lungs",
}


def segment_key(seg) -> tuple[str, str]:
    """(label, laterality) for one item of SegmentSequence."""
    label = str(getattr(seg, "SegmentLabel", "")).strip()
    modifier = ""
    for t in getattr(seg, "SegmentedPropertyTypeCodeSequence", []):
        for m in getattr(t, "SegmentedPropertyTypeModifierCodeSequence", []):
            modifier = str(getattr(m, "CodeMeaning", "")).strip()
    return label, modifier


def find_series(patient_dir: Path) -> tuple[Path | None, Path | None]:
    """(ct_series_dir, seg_file). The CT series with the most slices wins."""
    ct_dir, ct_n, seg_file = None, 0, None
    for study in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
        for series in sorted(p for p in study.iterdir() if p.is_dir()):
            files = sorted(series.glob("*.dcm"))
            if not files:
                continue
            try:
                d = pydicom.dcmread(files[0], stop_before_pixels=True)
            except Exception:
                continue
            if d.Modality == "CT" and len(files) > ct_n:
                ct_dir, ct_n = series, len(files)
            elif d.Modality == "SEG":
                seg_file = files[0]
    return ct_dir, seg_file


def read_ct(ct_dir: Path, tol: float = 0.01) -> tuple[sitk.Image, dict]:
    """Read a CT series onto a grid whose spacing is the one it was acquired at.

    A series with missing slices is the failure this guards against. SimpleITK
    still returns a single uniform spacing for one -- the *average* gap -- so
    every slice lands at the wrong z except by coincidence, and any mask placed
    by physical position rounds to the wrong index. Nothing about the resulting
    array shape looks wrong. LUNG1-014 is the worst local example: 87 slices at
    3.0 mm with 16 missing, averaged to 3.558 mm, an 18.6 % error.

    So the true gaps are measured from the headers, and if they are irregular the
    stack is resampled onto a regular grid at the *modal* gap -- the spacing the
    scanner actually used -- interpolating each output slice from its two real
    neighbours. That is what a planning system does on importing an irregular
    series.

    The honest cost: slices that were never acquired are interpolated, so their
    attenuation is invented. For dose that is fabricated tissue. It is still
    strictly better than the silent averaging it replaces, which fabricates the
    same slices *and* misplaces every real one, but it is not free -- callers
    that care should read `n_interpolated` from the returned info.
    """
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(ct_dir))
    if not names:
        raise RuntimeError(f"no DICOM series found in {ct_dir}")
    reader.SetFileNames(names)
    img = reader.Execute()

    ipp = np.array([[float(v) for v in
                     pydicom.dcmread(n, stop_before_pixels=True).ImagePositionPatient]
                    for n in names])
    order = np.argsort(ipp[:, 2])
    ipp = ipp[order]
    zs = ipp[:, 2]
    gaps = np.diff(zs)
    info = {"n_src": len(zs), "resampled": False}
    if len(gaps) < 2 or float(gaps.max() - gaps.min()) <= tol:
        return img, info

    # The modal gap is the acquisition spacing; larger gaps are dropped slices.
    vals, counts = np.unique(np.round(gaps, 2), return_counts=True)
    modal = float(vals[counts.argmax()])
    n_out = int(round((zs[-1] - zs[0]) / modal)) + 1
    tz = zs[0] + modal * np.arange(n_out)

    src = sitk.GetArrayFromImage(img)[order]                 # (n, rows, cols)
    lo = np.clip(np.searchsorted(zs, tz, side="right") - 1, 0, len(zs) - 2)
    out = np.empty((n_out, *src.shape[1:]), dtype=src.dtype)
    for i in range(n_out):                                   # per-slice: bounded memory
        a, b = lo[i], lo[i] + 1
        w = (tz[i] - zs[a]) / (zs[b] - zs[a])
        out[i] = np.rint((1.0 - w) * src[a] + w * src[b]).astype(src.dtype)

    res = sitk.GetImageFromArray(out)
    sp = img.GetSpacing()
    res.SetSpacing((sp[0], sp[1], modal))
    res.SetOrigin(tuple(float(v) for v in ipp[0]))
    # Rebuild the direction from the in-plane orientation rather than reusing the
    # reader's: the stack has been re-sorted to increasing z, so the slice normal
    # must point that way regardless of the order the files came in.
    d = pydicom.dcmread(names[0], stop_before_pixels=True)
    row = np.array([float(v) for v in d.ImageOrientationPatient[:3]])
    col = np.array([float(v) for v in d.ImageOrientationPatient[3:]])
    nrm = np.cross(row, col)
    if nrm[2] < 0:
        nrm = -nrm
    res.SetDirection(tuple(np.stack([row, col, nrm], axis=1).ravel()))

    # Signed: positive means slices were missing and have been interpolated,
    # negative means the series carried an extra off-grid slice that the regular
    # grid absorbs. Both are irregularities; only the first invents tissue.
    info.update(resampled=True, modal_spacing=modal,
                mean_spacing_before=float(gaps.mean()),
                n_out=n_out, n_slice_delta=int(n_out - len(zs)))
    return res, info


def decode_seg(seg_path: Path, ct: sitk.Image
               ) -> tuple[dict[str, np.ndarray], list[tuple[str, str]], int]:
    """DICOM SEG -> ({canonical name: bool volume on the CT grid}, unmapped, dropped).

    Frames are placed by their patient-space origin rather than by frame order:
    a SEG stores only the slices a segment actually touches, and the per-segment
    frame blocks are not required to be contiguous or ordered.
    """
    d = pydicom.dcmread(seg_path)
    if getattr(d, "SegmentationType", "BINARY") != "BINARY":
        raise RuntimeError(f"{seg_path.name}: unsupported SegmentationType "
                           f"{d.SegmentationType}")

    num_to_name: dict[int, str] = {}
    unmapped: list[tuple[str, str]] = []
    for s in d.SegmentSequence:
        key = segment_key(s)
        if key in SEG_NAME_MAP:
            num_to_name[int(s.SegmentNumber)] = SEG_NAME_MAP[key]
        else:
            unmapped.append(key)

    size_x, size_y, size_z = ct.GetSize()
    frames = d.pixel_array
    if frames.ndim == 2:                       # single-frame SEG
        frames = frames[None]
    if frames.shape[1:] != (size_y, size_x):
        raise RuntimeError(f"{seg_path.name}: SEG plane {frames.shape[1:]} != "
                           f"CT plane {(size_y, size_x)}")

    out = {name: np.zeros((size_z, size_y, size_x), dtype=bool)
           for name in set(num_to_name.values())}
    dropped = 0
    for i, fg in enumerate(d.PerFrameFunctionalGroupsSequence):
        num = int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        name = num_to_name.get(num)
        if name is None:
            continue
        ipp = [float(v) for v in fg.PlanePositionSequence[0].ImagePositionPatient]
        k = ct.TransformPhysicalPointToIndex(ipp)[2]
        if not 0 <= k < size_z:
            dropped += 1
            continue
        out[name][k] |= frames[i].astype(bool)

    # A consistent lung channel regardless of how this case was contoured.
    if "Lungs" not in out and ("Lung_L" in out or "Lung_R" in out):
        union = np.zeros((size_z, size_y, size_x), dtype=bool)
        for side in ("Lung_L", "Lung_R"):
            if side in out:
                union |= out[side]
        out["Lungs"] = union

    out = {k: v for k, v in out.items() if v.any()}
    return out, unmapped, dropped



def write_mask(mask: np.ndarray, ct: sitk.Image, path: Path) -> None:
    """Write a bool volume carrying the CT's geometry, so resampling lines up."""
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img.CopyInformation(ct)
    sitk.WriteImage(img, str(path), useCompression=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"E:\datasets\medical\nsclc-radiomics\NSCLC-Radiomics")
    ap.add_argument("--out", default=r"E:\datasets\medical\lung1_nrrd")
    ap.add_argument("--limit", type=int, default=0, help="0 = all patients")
    ap.add_argument("--cases", default="", help="comma-separated case names; "
                                                "re-ingest just these")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args(argv)

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    patients = sorted(p for p in src.iterdir() if p.is_dir())
    if args.cases:
        want = {c.strip() for c in args.cases.split(",") if c.strip()}
        patients = [p for p in patients if p.name in want]
        missing = want - {p.name for p in patients}
        if missing:
            raise SystemExit(f"no such case(s) under {src}: {sorted(missing)}")
    if args.limit:
        patients = patients[: args.limit]
    print(f"{len(patients)} patients -> {out}")

    # Merge rather than clobber: a --cases run re-ingests a handful of cases and
    # must not drop the record of every other case from the manifest.
    manifest_path = out / "ingest_manifest.json"
    manifest, failures = {}, []
    if manifest_path.exists():
        prev = json.loads(manifest_path.read_text())
        manifest = dict(prev.get("cases", {}))
        failures = [f for f in prev.get("failures", [])
                    if f["case"] not in {p.name for p in patients}]
    all_unmapped: dict[str, int] = {}
    t0 = time.time()
    for i, pat in enumerate(patients):
        case_out = out / pat.name
        if args.skip_existing and (case_out / "img.nrrd").exists():
            continue
        try:
            ct_dir, seg_file = find_series(pat)
            if ct_dir is None:
                raise RuntimeError("no CT series")
            if seg_file is None:
                raise RuntimeError("no SEG object")
            ct, ct_info = read_ct(ct_dir)
            masks, unmapped, dropped = decode_seg(seg_file, ct)
            if not masks:
                raise RuntimeError("SEG decoded to no usable structures")

            (case_out / "structures").mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(ct, str(case_out / "img.nrrd"), useCompression=True)
            for name, m in sorted(masks.items()):
                write_mask(m, ct, case_out / "structures" / f"{name}.nrrd")

            for key in unmapped:
                all_unmapped[str(key)] = all_unmapped.get(str(key), 0) + 1
            manifest[pat.name] = {
                "size": list(ct.GetSize()), "spacing": list(ct.GetSpacing()),
                "structures": sorted(masks), "frames_dropped": dropped,
                **ct_info,
            }
            if ct_info["resampled"]:
                d = ct_info["n_slice_delta"]
                what = (f"{d} interpolated" if d > 0 else
                        f"{-d} extra slice(s) absorbed" if d < 0 else
                        "slice count unchanged")
                print(f"  {pat.name}: irregular slices, resampled "
                      f"{ct_info['mean_spacing_before']:.3f} -> "
                      f"{ct_info['modal_spacing']:.3f} mm, {what}", flush=True)
        except Exception as e:
            failures.append({"case": pat.name, "error": f"{type(e).__name__}: {e}"})
            print(f"  FAILED {pat.name}: {type(e).__name__}: {e}", flush=True)

        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(patients)}] {el/60:.1f}m elapsed, "
                  f"eta {el/(i+1)*(len(patients)-i-1)/60:.1f}m", flush=True)

    manifest_path.write_text(json.dumps(
        {"source": str(src), "n_ok": len(manifest), "n_failed": len(failures),
         "unmapped_segments": all_unmapped, "failures": failures,
         "cases": manifest}, indent=1))

    counts: dict[str, int] = {}
    for rec in manifest.values():
        for s in rec["structures"]:
            counts[s] = counts.get(s, 0) + 1
    print(f"\n{len(manifest)} ok, {len(failures)} failed  ({(time.time()-t0)/60:.1f} min)")
    print("structure availability:")
    for name, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {c:4d}  {name}")
    if all_unmapped:
        print("unmapped SEG segments (not written):")
        for k, c in sorted(all_unmapped.items(), key=lambda x: -x[1]):
            print(f"  {c:4d}  {k}")


if __name__ == "__main__":
    main()
