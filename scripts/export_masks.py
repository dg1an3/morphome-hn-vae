"""Export true per-structure masks for the viewer's 3D surfaces.

The viewer already ships `data/vol/<case>_seg.bin`, a winner-take-all label map:
one byte per voxel, earlier structure index wins. That is the right shape for a
slice overlay -- a voxel gets one colour -- but it is lossy exactly where
contours overlap, and marching cubes has nothing to fall back on. Measured on
the thorax corpus:

    LUNG1-001   Lungs      0 / 194 130      GTV   5 181 / 5 773
    LUNG1-003   Lungs      0 / 145 196      GTV     821 / 1 286
    synth0003   Lungs  5 177 / 141 062      GTV       0 / 599

`Lungs` is the union of `Lung_L`/`Lung_R`, which both outrank it, so it is wiped
out; `GTV` is the last index and sits inside a lung, so it loses most or all of
its voxels. Contouring the label map would render an empty tumour on some cases
and no combined lung anywhere -- the two structures a thorax reader most wants
in 3D.

So surfaces get their own input: the cache's own label bitplanes, one binary
mask per structure, unmodified by any overlap rule.

Size is why this is a separate file rather than nine full-grid planes. Each mask
is cropped to its own bounding box and bit-packed, which is what makes the whole
corpus affordable -- about 250 kB per case against 3.4 MB for one uncropped
plane -- and has the side benefit that marching cubes in the browser then runs
over the crop rather than the full grid, which for GTV is ~70x less work.

Format, little-endian throughout. One file per case:

    0   char[4]  "MMSK"
    4   uint8    version = 1
    5   uint8    n_struct
    6   uint16   0
    8   n_struct x 20-byte records, in STRUCTURES order:
          uint16 x0, y0, z0     crop origin, voxels, in the full grid
          uint16 nx, ny, nz     crop size, voxels (all zero => absent)
          uint32 offset         from file start
          uint32 length         bytes
    ...  packed bits, x fastest, LSB first, ceil(nx*ny*nz / 8) bytes each

The structure list comes from the viewer's embedding.json rather than from
`morphome.constants`, so this runs against either corpus without setting
MORPHOME_PROFILE -- and it cross-checks that list against each cache's meta.json
so a profile mismatch is an error here rather than a mis-tinted surface later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HEADER = 8
RECORD = 20


def bbox(mask: np.ndarray) -> tuple[int, int, int, int, int, int] | None:
    """(z0, z1, y0, y1, x0, x1) of the True voxels, or None if there are none.

    Reducing each axis first is what keeps this cheap: three `any` passes over
    the volume rather than materialising the coordinates of every set voxel,
    which for a lung is a few hundred thousand triples per structure per case.
    """
    az = mask.any(axis=(1, 2))
    if not az.any():
        return None
    ay = mask.any(axis=(0, 2))
    ax = mask.any(axis=(0, 1))

    def span(a):
        return int(np.argmax(a)), int(len(a) - np.argmax(a[::-1]))

    z0, z1 = span(az)
    y0, y1 = span(ay)
    x0, x1 = span(ax)
    return z0, z1, y0, y1, x0, x1


def pack_case(labels_packed: np.ndarray, n_struct: int) -> bytes:
    """One case's bitplanes -> one MMSK blob."""
    recs, blobs = [], []
    for i in range(n_struct):
        m = ((labels_packed >> i) & 1).astype(bool)          # (z, y, x)
        box = bbox(m)
        if box is None:
            recs.append((0, 0, 0, 0, 0, 0))
            blobs.append(b"")
            continue
        z0, z1, y0, y1, x0, x1 = box
        crop = m[z0:z1, y0:y1, x0:x1]
        # C-order ravel of a (z, y, x) crop is x-fastest, which is the order the
        # browser walks when it unpacks -- and the same convention the volume
        # export already relies on.
        blobs.append(np.packbits(crop.ravel(order="C"), bitorder="little").tobytes())
        recs.append((x0, y0, z0, x1 - x0, y1 - y0, z1 - z0))

    out = bytearray(b"MMSK")
    out += bytes([1, n_struct, 0, 0])
    off = HEADER + RECORD * n_struct
    for rec, blob in zip(recs, blobs):
        out += np.array(rec, dtype="<u2").tobytes()
        out += np.array([off if blob else 0, len(blob)], dtype="<u4").tobytes()
        off += len(blob)
    for blob in blobs:
        out += blob
    return bytes(out)


def read_masks(path: Path) -> list[dict | None]:
    """Parse one MMSK file back into cropped boolean masks."""
    b = path.read_bytes()
    if b[:4] != b"MMSK":
        raise SystemExit(f"{path} is not a mask file")
    if b[4] != 1:
        raise SystemExit(f"{path}: mask format v{b[4]} is not supported")
    out: list[dict | None] = []
    for i in range(b[5]):
        p = HEADER + RECORD * i
        x0, y0, z0, nx, ny, nz = (int(v) for v in np.frombuffer(b, "<u2", 6, p))
        off, ln = (int(v) for v in np.frombuffer(b, "<u4", 2, p + 12))
        if ln == 0:
            out.append(None)
            continue
        bits = np.unpackbits(np.frombuffer(b, "u1", ln, off), bitorder="little")
        out.append({"x0": x0, "y0": y0, "z0": z0,
                    "m": bits[:nx * ny * nz].reshape(nz, ny, nx).astype(bool)})
    return out


def redundant(masks, thresh: float = 0.95) -> list[int]:
    """Structures that are just a union of smaller ones.

    Some profiles carry a combined channel alongside its parts -- thorax has
    `Lungs` beside `Lung_L`/`Lung_R`, and on a real case it is their union
    exactly. Anything drawing all the structures at once wants to skip it: two
    coincident surfaces in the same place z-fight into speckle, and the shell
    buries whatever sits inside it.

    "Covered by the union of strictly smaller structures" is what separates a
    union channel from a nested one. Lungs is covered by the two lobes, which
    are smaller, so it goes. GTV is covered by Lungs, which is larger, so it
    stays -- which matters, because on this corpus GTV is the structure most
    worth looking at. A case contoured only as `Lungs` with no lobes keeps it,
    since there is then nothing smaller to cover it.
    """
    vols = {i: int(r["m"].sum()) for i, r in enumerate(masks) if r is not None}
    drop = []
    for i, vi in vols.items():
        smaller = [j for j, vj in vols.items() if vj < vi]
        if not smaller or not vi:
            continue
        rec = masks[i]
        acc = np.zeros_like(rec["m"], dtype=bool)
        for j in smaller:
            o = masks[j]
            # Overlap in i's own crop frame; a structure that does not reach it
            # contributes nothing.
            dz, dy, dx = o["z0"] - rec["z0"], o["y0"] - rec["y0"], o["x0"] - rec["x0"]
            zs = slice(max(0, dz), min(acc.shape[0], dz + o["m"].shape[0]))
            ys = slice(max(0, dy), min(acc.shape[1], dy + o["m"].shape[1]))
            xs = slice(max(0, dx), min(acc.shape[2], dx + o["m"].shape[2]))
            if zs.start >= zs.stop or ys.start >= ys.stop or xs.start >= xs.stop:
                continue
            acc[zs, ys, xs] |= o["m"][zs.start - dz: zs.stop - dz,
                                      ys.start - dy: ys.stop - dy,
                                      xs.start - dx: xs.stop - dx]
        if (rec["m"] & acc).sum() / vi >= thresh:
            drop.append(i)
    return drop


def write_mask_meta(out_dir: Path, cases: list[str], dst: Path) -> None:
    """Per-case sidecar naming the union channels and the present structures.

    The viewer defaults its 3D surfaces to "everything present that is not a
    union of something else", which needs this to be known before any mask is
    contoured. Computing it here keeps one implementation of the rule for both
    the browser and the offline thumbnail renderer, instead of a JavaScript
    copy that could drift from the Python one.
    """
    meta = {}
    for name in cases:
        masks = read_masks(out_dir / f"{name}.bin")
        meta[name] = {
            "present": [i for i, m in enumerate(masks) if m is not None],
            "redundant": redundant(masks),
        }
    dst.write_text(json.dumps({"cases": meta}, separators=(",", ":")))


def check_structures(cache: Path, expected: list[str]) -> None:
    meta_path = cache / "meta.json"
    if not meta_path.exists():
        return
    got = list(json.loads(meta_path.read_text()).get("structures", []))
    if got and got != expected:
        raise SystemExit(f"{cache} was built with structures {got}, but the "
                         f"viewer's embedding.json expects {expected}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", required=True,
                    help="viewer directory built by build_viewer.py")
    ap.add_argument("--synth", required=True, help="synthetic cache directory")
    ap.add_argument("--real", required=True, help="real cache directory")
    ap.add_argument("--force", action="store_true",
                    help="rewrite masks that already exist")
    ap.add_argument("--meta-only", action="store_true",
                    help="recompute mask_meta.json from the masks already on "
                         "disk, without touching the caches")
    args = ap.parse_args()

    viewer = Path(args.viewer)
    emb_path = viewer / "data" / "embedding.json"
    if not emb_path.exists():
        raise SystemExit(f"no embedding.json under {viewer}; build the viewer first")
    emb = json.loads(emb_path.read_text())
    structures = list(emb["structures"])
    n_struct = len(structures)

    caches = {0: Path(args.synth), 1: Path(args.real)}
    for c in caches.values():
        check_structures(c, structures)

    out_dir = viewer / "data" / "mask"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = emb["cases"]
    names = [c["name"] for c in cases]
    meta_dst = viewer / "data" / "mask_meta.json"
    if args.meta_only:
        write_mask_meta(out_dir, names, meta_dst)
        print(f"{meta_dst}  ({meta_dst.stat().st_size/1e3:.0f} kB)")
        return

    print(f"{len(cases)} cases, {n_struct} structures -> {out_dir}")
    written = skipped = 0
    total = 0
    for n, rec in enumerate(cases):
        dst = out_dir / f"{rec['name']}.bin"
        if dst.exists() and not args.force:
            skipped += 1
            total += dst.stat().st_size
            continue
        src = caches[int(rec["real"])] / f"{rec['name']}.npz"
        # Only the bitplanes are needed: reading ct_hu and body as well would
        # more than triple the I/O for data no surface uses.
        with np.load(src, allow_pickle=True) as z:
            packed = z["labels_packed"]
        blob = pack_case(packed, n_struct)
        dst.write_bytes(blob)
        written += 1
        total += len(blob)
        if (n + 1) % 50 == 0:
            print(f"  [{n+1}/{len(cases)}] {total/1e6:.0f} MB", flush=True)

    print(f"\n{written} written, {skipped} already present, "
          f"{total/1e6:.0f} MB total ({total/max(len(cases),1)/1e3:.0f} kB/case)")
    write_mask_meta(out_dir, names, meta_dst)
    print(f"{meta_dst.name}  ({meta_dst.stat().st_size/1e3:.0f} kB)")


if __name__ == "__main__":
    sys.exit(main())
