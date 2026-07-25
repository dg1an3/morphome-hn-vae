"""Read DICOM headers to identify what body part / modality each collection on E:
actually contains, so data hunting is driven by metadata rather than folder names.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pydicom

ROOTS = {
    "tciaDownload": r"E:\datasets\medical\tciaDownload",
    "CT Lymph Nodes": r"E:\datasets\medical\CT Lymph Nodes\CT Lymph Nodes",
    "nsclc-radiomics": r"E:\datasets\medical\nsclc-radiomics\NSCLC-Radiomics",
    "upenn-gbm": r"E:\datasets\medical\upenn-gbm",
    "IXI": r"E:\datasets\medical\IXI",
}

FIELDS = ["Modality", "BodyPartExamined", "StudyDescription", "SeriesDescription",
          "ProtocolName", "Manufacturer"]


def sample_files(root: Path, n: int, seed: int = 0) -> list[Path]:
    dirs = [d for d in root.iterdir() if d.is_dir()] if root.is_dir() else []
    rng = random.Random(seed)
    rng.shuffle(dirs)
    out: list[Path] = []
    for d in dirs[: n * 4]:
        found = None
        for f in d.rglob("*"):
            if f.suffix.lower() in (".dcm", "") and f.is_file():
                try:
                    pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                    found = f
                    break
                except Exception:
                    continue
        if found:
            out.append(found)
        if len(out) >= n:
            break
    return out


def main() -> None:
    for name, r in ROOTS.items():
        root = Path(r)
        print(f"\n=== {name}  ({r})")
        if not root.exists():
            print("  MISSING")
            continue
        files = sample_files(root, 6)
        if not files:
            print("  no readable DICOM found (may be NIfTI/other format)")
            exts = Counter(p.suffix.lower() for p in list(root.rglob("*"))[:4000] if p.is_file())
            print("  extensions seen:", dict(exts.most_common(6)))
            continue
        agg = {f: Counter() for f in FIELDS}
        for f in files:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            for fld in FIELDS:
                agg[fld][str(getattr(ds, fld, "")).strip() or "-"] += 1
        for fld in FIELDS:
            vals = ", ".join(f"{v}({c})" for v, c in agg[fld].most_common(4))
            print(f"  {fld:<18} {vals}")
        # RTSTRUCT presence is what makes a collection useful for OAR training
        rt = sum(1 for _ in list(root.rglob("*"))[:20000]
                 if _.is_file() and _.suffix.lower() == ".dcm" and "RTSTRUCT" in _.name.upper())
        print(f"  files with RTSTRUCT in name (first 20k scanned): {rt}")


if __name__ == "__main__":
    main()
