"""Resample PDDCA cases onto a canonical isotropic grid and cache them as .npz.

Design notes
------------
* The corpus is geometrically heterogeneous (in-plane 0.76-1.27 mm, slice
  1.25-3.0 mm, 76-360 slices). Nothing downstream can assume voxel counts mean
  the same thing across cases, so every case is resampled to one physical grid.
* The crop is anchored on the centroid of BrainStem + both Parotids. Those three
  are present in all 48 cases and sit at a stable anatomical position, which
  gives a reproducible frame without needing registration.
* Structure masks are resampled as floats and re-thresholded at 0.5. Nearest
  neighbour at 1.5 mm would shred the small OARs (Chiasm has a median of 184
  source voxels).
* Missing structure files are recorded in a presence vector. They are *unknown*,
  not empty -- the training loss masks them out.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from .constants import (
    ANCHOR_OFFSET_MM,
    ANCHOR_STRUCTURES,
    CASE_GLOB,
    GRID_DIMS,
    GRID_SPACING,
    BODY_HU_THRESHOLD,
    HU_MAX,
    HU_MIN,
    N_STRUCT,
    STRUCTURES,
)


@dataclass
class GridSpec:
    """Canonical output grid, expressed physically.

    Anisotropic, because the patient is not cubic: 560 x 400 x 240 mm at 2.5 mm
    contains the full external contour in every transverse slice of all 48 cases
    while a cube covering the same lateral extent would waste ~2.3x the voxels
    on air. See scripts/fit_dose_frame.py.
    """

    dims: tuple[int, int, int] = GRID_DIMS   # voxels, (x, y, z) = (L-R, A-P, S-I)
    spacing: float = GRID_SPACING            # mm, isotropic

    @property
    def extent_mm(self) -> np.ndarray:
        return np.asarray(self.dims, dtype=float) * self.spacing

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        """numpy array shape, which is the reverse of the SimpleITK size."""
        return (self.dims[2], self.dims[1], self.dims[0])


def _read(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def _centroid_mm(mask: sitk.Image) -> np.ndarray | None:
    """Physical-space centroid (x, y, z in mm) of a binary mask, or None if empty."""
    arr = sitk.GetArrayFromImage(mask) > 0
    if not arr.any():
        return None
    idx_zyx = np.argwhere(arr).mean(0)
    idx_xyz = [float(idx_zyx[2]), float(idx_zyx[1]), float(idx_zyx[0])]
    return np.asarray(mask.TransformContinuousIndexToPhysicalPoint(idx_xyz))


def _resample(
    image: sitk.Image,
    grid: GridSpec,
    center_mm: np.ndarray,
    interpolator: int,
    default_value: float,
) -> np.ndarray:
    """Resample onto an axis-aligned box of `grid` centred on `center_mm`."""
    half = grid.extent_mm / 2.0
    # Origin is the centre of the first voxel, hence the +spacing/2 offset.
    origin = center_mm - half + grid.spacing / 2.0

    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing([grid.spacing] * 3)
    rs.SetSize([int(v) for v in grid.dims])
    rs.SetOutputOrigin(origin.tolist())
    rs.SetOutputDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    rs.SetInterpolator(interpolator)
    rs.SetDefaultPixelValue(default_value)
    out = rs.Execute(sitk.Cast(image, sitk.sitkFloat32))
    return sitk.GetArrayFromImage(out)  # (z, y, x)


def body_mask(ct_hu: np.ndarray) -> np.ndarray:
    """Largest connected soft-tissue component, holes filled.

    Removes the treatment couch, immobilisation shell and scanner-bore air so the
    model spends its capacity on anatomy rather than on hardware that varies by
    scanner.
    """
    m = ct_hu > BODY_HU_THRESHOLD
    m = ndimage.binary_closing(m, structure=np.ones((3, 3, 3)), iterations=2)
    lab, n = ndimage.label(m)
    if n == 0:
        return np.ones_like(m, dtype=bool)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = lab == (int(np.argmax(sizes)) + 1)
    # Fill holes slice-wise rather than in 3D: a full 3D fill leaks out through
    # the airway where it meets the top/bottom faces of the crop. Filling along
    # all three axes and taking the union recovers concavities (nasal cavity,
    # oropharynx) that a z-only pass leaves open.
    filled = np.zeros_like(m)
    for axis in range(3):
        f = np.zeros_like(m)
        for i in range(m.shape[axis]):
            sl = [slice(None)] * 3
            sl[axis] = i
            f[tuple(sl)] = ndimage.binary_fill_holes(m[tuple(sl)])
        filled |= f
    return filled


def preprocess_case(
    case_dir: Path,
    grid: GridSpec,
    apply_body_mask: bool = True,
) -> dict:
    img = _read(case_dir / "img.nrrd")

    # --- anatomical anchor -------------------------------------------------
    centroids = []
    for name in ANCHOR_STRUCTURES:
        p = case_dir / "structures" / f"{name}.nrrd"
        if p.exists():
            c = _centroid_mm(_read(p))
            if c is not None:
                centroids.append(c)
    if not centroids:
        raise RuntimeError(f"{case_dir.name}: no anchor structures available")
    center = np.mean(centroids, axis=0) + np.asarray(ANCHOR_OFFSET_MM, dtype=float)

    # Clamp so the cube stays inside the acquired field of view where possible.
    size_xyz = np.array(img.GetSize(), dtype=float)
    spacing_xyz = np.array(img.GetSpacing(), dtype=float)
    origin_xyz = np.array(img.GetOrigin(), dtype=float)
    fov_lo = origin_xyz - spacing_xyz / 2.0
    fov_hi = fov_lo + size_xyz * spacing_xyz
    half = grid.extent_mm / 2.0
    for a in range(3):
        if fov_hi[a] - fov_lo[a] >= grid.extent_mm[a]:
            center[a] = float(np.clip(center[a], fov_lo[a] + half[a], fov_hi[a] - half[a]))

    # --- CT ----------------------------------------------------------------
    ct = _resample(img, grid, center, sitk.sitkLinear, default_value=HU_MIN)
    ct = np.clip(ct, HU_MIN, HU_MAX)

    body = body_mask(ct)
    if apply_body_mask:
        ct = np.where(body, ct, HU_MIN)

    # --- structures --------------------------------------------------------
    labels = np.zeros((N_STRUCT, *grid.shape_zyx), dtype=bool)
    presence = np.zeros(N_STRUCT, dtype=bool)
    for i, name in enumerate(STRUCTURES):
        p = case_dir / "structures" / f"{name}.nrrd"
        if not p.exists():
            continue
        m = _read(p)
        soft = _resample(m, grid, center, sitk.sitkLinear, default_value=0.0)
        labels[i] = soft >= 0.5
        presence[i] = True

    # Pack the 9 boolean channels into one uint16 bitfield: 4 MB instead of 34 MB.
    packed = np.zeros(grid.shape_zyx, dtype=np.uint16)
    for i in range(N_STRUCT):
        packed |= (labels[i].astype(np.uint16) << i)

    return {
        "ct_hu": ct.astype(np.int16),
        "labels_packed": packed,
        "body": np.packbits(body),
        "presence": presence,
        "center_mm": center.astype(np.float32),
        "grid_dims": np.asarray(grid.dims, dtype=np.int32),
        "grid_spacing": np.float32(grid.spacing),
        "case": case_dir.name,
        "_stats": {
            "case": case_dir.name,
            "center_mm": [round(float(v), 2) for v in center],
            "n_present": int(presence.sum()),
            "missing": [STRUCTURES[i] for i in range(N_STRUCT) if not presence[i]],
            "label_voxels": {STRUCTURES[i]: int(labels[i].sum()) for i in range(N_STRUCT)},
            "body_frac": round(float(body.mean()), 4),
            "hu_mean_in_body": round(float(ct[body].mean()), 1),
        },
    }


def unpack_labels(packed: np.ndarray) -> np.ndarray:
    """uint16 bitfield -> (N_STRUCT, D, H, W) bool."""
    out = np.zeros((N_STRUCT, *packed.shape), dtype=bool)
    for i in range(N_STRUCT):
        out[i] = (packed >> i) & 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the canonical-grid cache.")
    ap.add_argument("--root", default=r"E:\datasets\medical\miccai_hn_sharpe")
    ap.add_argument("--out", default=r"E:\datasets\medical\morphome_cache\hn_dose_2.5mm")
    ap.add_argument("--dims", type=int, nargs=3, default=list(GRID_DIMS),
                    help="voxels, (x y z) = (L-R A-P S-I)")
    ap.add_argument("--spacing", type=float, default=GRID_SPACING)
    ap.add_argument("--no-body-mask", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    grid = GridSpec(dims=tuple(args.dims), spacing=args.spacing)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = sorted(d for d in root.iterdir() if d.is_dir() and (d / "img.nrrd").exists())
    e = grid.extent_mm
    print(f"{len(cases)} cases -> {grid.dims[0]}x{grid.dims[1]}x{grid.dims[2]} @ "
          f"{grid.spacing} mm ({e[0]:.0f} x {e[1]:.0f} x {e[2]:.0f} mm), "
          f"body_mask={not args.no_body_mask}")

    stats, skipped = [], []
    for i, c in enumerate(cases, 1):
        dst = out / f"{c.name}.npz"
        if dst.exists() and not args.overwrite:
            print(f"[{i:2d}/{len(cases)}] {c.name} skip (exists)", flush=True)
            continue
        # A case with none of the anchor structures cannot be positioned, which
        # preprocess_case reports by raising. Skipping is right -- 10 of the 421
        # thorax cases have no lung contour at all -- but it must not take the
        # rest of the corpus down with it.
        try:
            rec = preprocess_case(c, grid, apply_body_mask=not args.no_body_mask)
        except RuntimeError as e:
            skipped.append({"case": c.name, "error": str(e)})
            print(f"[{i:2d}/{len(cases)}] {c.name}  SKIPPED: {e}", flush=True)
            continue
        st = rec.pop("_stats")
        np.savez_compressed(dst, **rec)
        stats.append(st)
        print(f"[{i:2d}/{len(cases)}] {c.name}  present={st['n_present']}/{N_STRUCT}  "
              f"body={st['body_frac']:.2f}  missing={','.join(st['missing']) or '-'}", flush=True)

    if stats:
        (out / "preprocess_stats.json").write_text(json.dumps(stats, indent=2))
    meta = {"grid": asdict(grid), "structures": list(STRUCTURES),
            "hu_range": [HU_MIN, HU_MAX], "body_masked": not args.no_body_mask,
            "case_glob": CASE_GLOB,
            "source_root": str(root), "skipped": skipped}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ncache written to {out}  ({len(stats)} written, {len(skipped)} skipped)")
    for s in skipped:
        print(f"  skipped {s['case']}: {s['error']}")


if __name__ == "__main__":
    main()
