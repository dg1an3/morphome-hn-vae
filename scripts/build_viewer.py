"""Build a local, self-hosted browser explorer for a synthetic corpus.

Two halves:

* **Latent map.** A cascaded UMAP, 32 -> 5 -> 2 and 32 -> 5 -> 3, fitted on the
  synthetic latents *and* the real posterior means together so the two are
  co-embedded. Co-embedding is the point: it shows whether the sampler covers
  the real distribution or collapses into part of it, which a separate embedding
  per corpus could never reveal.

* **Ortho viewer.** vtk.js renders true transverse / coronal / sagittal slices
  from the actual volume, with a scrubbable slice index and an optional label
  overlay, in either of two layouts: the three planes intersecting in one
  rotatable 3D scene (default, with an LPS orientation cube), or as three
  separate parallel-projection panes. Volumes are fetched per case rather than
  embedded, which is why this is served locally instead of shipped as a single
  file.

Axis convention: cached arrays are (z, y, x) in C order, so a flat copy lands in
VTK's x-fastest layout unchanged and K/J/I slices are axial/coronal/sagittal
respectively.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from morphome.constants import N_STRUCT, STRUCTURES
from morphome.data import HNCache
from morphome.render import bone_sharpness
from explore_latent import encode_all, load_model, n_derived

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "viewer_assets"


def grid_of(meta: dict) -> tuple[list[int], float]:
    """(dims_xyz, spacing_mm) from a cache meta.

    Two layouts are in circulation: the older caches record a single cubic
    `size`, the 2.5 mm dose frame records explicit `dims` in (x, y, z) order --
    which is already what VTK's setDimensions wants, since the arrays are
    (z, y, x) C-order.
    """
    g = meta.get("grid", {})
    spacing = float(g.get("spacing", 1.6))
    if "dims" in g:
        return [int(v) for v in g["dims"]], spacing
    n = int(g.get("size", 128))
    return [n, n, n], spacing


def to_uint8(ct_norm: np.ndarray) -> np.ndarray:
    """[-1,1] -> 0..255. Window/level is applied in the browser on this scale."""
    return np.clip((ct_norm + 1.0) * 127.5, 0, 255).astype(np.uint8)


def seg_uint8(labels: np.ndarray) -> np.ndarray:
    """(9,D,H,W) binary -> winner-take-all label map, 0 = background."""
    out = np.zeros(labels.shape[1:], dtype=np.uint8)
    for i in range(labels.shape[0] - 1, -1, -1):     # earlier index wins
        out[labels[i] > 0.5] = i + 1
    return out


def cascade_umap(x: np.ndarray, seed: int, n_neighbors: int, min_dist: float):
    import umap

    mid = umap.UMAP(n_components=5, n_neighbors=n_neighbors, min_dist=min_dist,
                    random_state=seed, verbose=False).fit_transform(x)
    two = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                    random_state=seed, verbose=False).fit_transform(mid)
    three = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
                      random_state=seed, verbose=False).fit_transform(mid)
    return mid, two, three


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-ckpt", default="runs/v2_bone/last.pt")
    ap.add_argument("--synth", default=r"E:\datasets\medical\morphome_cache\hn_synth_v1")
    ap.add_argument("--real", default=r"E:\datasets\medical\morphome_cache\hn_128_1.6mm")
    ap.add_argument("--out", default=r"E:\datasets\medical\morphome_cache\hn_synth_v1_viewer")
    ap.add_argument("--vtk", default="", help="path to the vtk.js UMD bundle")
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-volumes", action="store_true", help="rebuild metadata only")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "data" / "vol").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    synth = HNCache(args.synth, with_bone=False, in_memory=False)
    real = HNCache(args.real, with_bone=False, in_memory=False)
    print(f"{len(synth)} synthetic, {len(real)} real cases")

    # Both corpora feed one vtkImageData pipeline with a single dims/spacing, so
    # a mismatch would not degrade gracefully -- one of the two would be read
    # with the wrong stride and render as noise. Fail here instead.
    dims, spacing = grid_of(real.meta)
    if (dims, spacing) != grid_of(synth.meta):
        raise SystemExit(f"grid mismatch: real {grid_of(real.meta)} vs "
                         f"synthetic {grid_of(synth.meta)}; the viewer renders "
                         f"both on one grid")
    print(f"grid {dims[0]}x{dims[1]}x{dims[2]} @ {spacing} mm")

    z_synth = np.load(Path(args.synth) / "latents.npz")["z"].astype(np.float32)
    assert len(z_synth) == len(synth), "latents.npz does not match the case count"

    vae, vcfg, _ = load_model(args.vae_ckpt, device, prefer_ema=True)
    real_bone = HNCache(args.real, derived=n_derived(vcfg), in_memory=False)
    mu_real, real_names = encode_all(vae, real_bone, device)
    print(f"latents: synthetic {z_synth.shape}, real {mu_real.shape}")

    x = np.concatenate([z_synth, mu_real]).astype(np.float32)
    is_real = np.array([0] * len(z_synth) + [1] * len(mu_real))
    names = [synth.cases[i] for i in range(len(synth))] + list(real_names)

    print("fitting cascaded UMAP 32 -> 5 -> 2/3 ...")
    mid, two, three = cascade_umap(x, args.seed, args.n_neighbors, args.min_dist)

    # Export volumes and gather per-case metadata.
    records = []
    for gi, (name, real_flag) in enumerate(zip(names, is_real)):
        ds = real if real_flag else synth
        idx = ds.cases.index(name)
        s = ds[idx]
        ct = s["ct"][0].numpy()
        lab = s["labels"].numpy()
        if not args.no_volumes:
            (out / "data" / "vol" / f"{name}_ct.bin").write_bytes(to_uint8(ct).tobytes())
            (out / "data" / "vol" / f"{name}_seg.bin").write_bytes(seg_uint8(lab).tobytes())
        vox = lab.reshape(N_STRUCT, -1).sum(1)
        records.append({
            "name": name, "real": int(real_flag),
            "umap2": [float(v) for v in two[gi]],
            "umap3": [float(v) for v in three[gi]],
            "umap5": [float(v) for v in mid[gi]],
            "sharpness": bone_sharpness(ct),
            "body_frac": float(s["body"].mean()),
            "n_present": int((vox >= 50).sum()),
            "organ_voxels": {STRUCTURES[j]: int(vox[j]) for j in range(N_STRUCT)},
        })
        if (gi + 1) % 25 == 0:
            print(f"  [{gi+1}/{len(names)}]", flush=True)

    (out / "data" / "embedding.json").write_text(json.dumps({
        "structures": list(STRUCTURES),
        "dims": dims, "spacing": spacing,
        "umap": {"n_neighbors": args.n_neighbors, "min_dist": args.min_dist,
                 "seed": args.seed, "cascade": "32 -> 5 -> 2/3"},
        "vae_ckpt": args.vae_ckpt,
        "synth_meta": json.loads((Path(args.synth) / "meta.json").read_text()),
        "cases": records,
    }, indent=1))

    vtkjs = Path(args.vtk) if args.vtk else None
    if vtkjs and vtkjs.exists():
        shutil.copy2(vtkjs, out / "vtk.js")
        print(f"copied vtk.js ({vtkjs.stat().st_size/1e6:.1f} MB)")
    else:
        print("WARNING: no --vtk bundle given; the viewer will not render slices")

    for asset in ("index.html", "app.js"):
        shutil.copy2(ASSETS / asset, out / asset)

    n_vol = len(list((out / "data" / "vol").glob("*.bin")))
    size = sum(p.stat().st_size for p in out.rglob("*")) / 1e9
    print(f"\nviewer -> {out}")
    print(f"  {len(records)} cases, {n_vol} volume files, {size:.2f} GB")
    print(f"\nserve it:\n  cd {out}\n  python -m http.server 8000\nthen open "
          f"http://localhost:8000/")


if __name__ == "__main__":
    main()
