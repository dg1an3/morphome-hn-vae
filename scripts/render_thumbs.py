"""Render a 3D-orthogonal thumbnail per synthetic case, packed into one atlas.

The latent map draws each synthetic case as a billboard of its own anatomy
instead of a dot, so the cloud shows what the sampler actually produced rather
than only where it landed. 400 live scenes is not an option -- two 3.4 MB
volumes and seven surfaces per case -- so the thumbnails are rendered once,
here, and the page composites them as sprites.

Inputs are the viewer's own files, not the caches: `data/vol/<case>_ct.bin` is
already the exact uint8 the browser windows, and `data/mask/<case>.bin` already
holds the per-structure masks it contours. Reading those instead of the npz
caches makes a thumbnail agree with the interactive view by construction -- the
same voxels through the same isosurface at the same camera -- and skips
decompressing 400 archives for data that has already been extracted once.

The camera matches `orient3D` in app.js exactly (right-anterior-superior
oblique, +z up, the same 1.35 zoom), so clicking a sprite opens the view it is
a picture of.

Output:
    <viewer>/data/thumbs.png    RGBA atlas, `cell` px per cell, `cols` per row
    <viewer>/data/thumbs.json   {cell, cols, rows, order: [case, ...]}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import vtk
    from vtk.util import numpy_support
except ImportError:                                     # pragma: no cover
    raise SystemExit("this script needs VTK: pip install vtk")

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The union-channel rule and the mask reader are shared with the exporter, so
# a thumbnail and the browser agree on which structures to draw.
from export_masks import read_masks, redundant

# Same palette as app.js LABEL_RGB, index 0 being background.
LABEL_RGB = [(0, 0, 0), (0.90, 0.10, 0.10), (1.00, 0.95, 0.20), (0.20, 0.60, 1.00),
             (0.20, 1.00, 0.40), (0.00, 0.70, 0.25), (1.00, 0.45, 0.00),
             (0.85, 0.30, 0.00), (0.80, 0.30, 1.00), (0.55, 0.10, 0.85)]

AIR_ALPHA = 0.05        # matches app.js
SURF_SMOOTH = 2         # [1,2,1] passes per axis, matches app.js

# Opacity by size. In the interactive view the reader picks which structures to
# show; a thumbnail has to choose, and drawing every structure at one opacity
# means the largest simply hides the rest -- a solid Lungs surface swallows the
# heart, both lobes and the tumour sitting inside it. Scaling alpha down with
# volume turns the big structures into glassy shells that the small ones read
# through, which is the whole point of a per-case picture.
BIG_VOX, BIG_ALPHA = 150_000, 0.22
SMALL_VOX, SMALL_ALPHA = 20_000, 0.95


def surf_alpha(n_vox: int) -> float:
    """Interpolate alpha on log volume between the two anchors above."""
    if n_vox <= SMALL_VOX:
        return SMALL_ALPHA
    if n_vox >= BIG_VOX:
        return BIG_ALPHA
    t = (np.log(n_vox) - np.log(SMALL_VOX)) / (np.log(BIG_VOX) - np.log(SMALL_VOX))
    return float(SMALL_ALPHA + t * (BIG_ALPHA - SMALL_ALPHA))


def to_vtk_image(arr: np.ndarray, dims_xyz, spacing, origin=(0.0, 0.0, 0.0)):
    """(z,y,x) C-order numpy -> vtkImageData. The ravel is x-fastest, which is
    already VTK's layout, so no transpose is involved."""
    img = vtk.vtkImageData()
    img.SetDimensions(*dims_xyz)
    img.SetSpacing(spacing, spacing, spacing)
    img.SetOrigin(*origin)
    flat = np.ascontiguousarray(arr).ravel()
    va = numpy_support.numpy_to_vtk(flat, deep=True)
    va.SetName("scalars")
    img.GetPointData().SetScalars(va)
    return img


def ct_lookup_table() -> vtk.vtkLookupTable:
    """Grayscale 0..255 with the air floor punched nearly transparent.

    app.js gets this from a piecewise opacity function on the image property;
    VTK's image property has no scalar-opacity of its own, so the alpha rides
    in the lookup table instead. Same result: scalar 0 is the outside of a
    body-masked CT and nothing inside lands on it.
    """
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetTableRange(0, 255)
    for i in range(256):
        g = i / 255.0
        lut.SetTableValue(i, g, g, g, AIR_ALPHA if i == 0 else 1.0)
    lut.Build()
    return lut


def build_scene(ct: np.ndarray, masks, dims, spacing, zoom: float):
    """One renderer holding three ortho slices and the structure surfaces."""
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.04, 0.05, 0.07)
    ren.SetBackgroundAlpha(0.0)

    img = to_vtk_image(ct, dims, spacing)
    lut = ct_lookup_table()
    for axis in range(3):
        mapper = vtk.vtkImageSliceMapper()
        mapper.SetInputData(img)
        mapper.SetOrientation(axis)
        mapper.SetSliceNumber((dims[axis] - 1) // 2)
        actor = vtk.vtkImageSlice()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetLookupTable(lut)
        prop.UseLookupTableScalarRangeOn()
        ren.AddViewProp(actor)

    drop = redundant(masks)
    # Largest first, so the glassy shells are laid down before the small solid
    # organs that read through them.
    order = sorted((i for i, r in enumerate(masks) if r is not None and i not in drop),
                   key=lambda i: -int(masks[i]["m"].sum()))
    for i in order:
        rec = masks[i]
        # One zero voxel of pad so a structure flush against its bounding box
        # closes instead of contouring to an open shell.
        m = np.pad(rec["m"], 1)
        vol = to_vtk_image(
            smooth(m, SURF_SMOOTH), (m.shape[2], m.shape[1], m.shape[0]), spacing,
            ((rec["x0"] - 1) * spacing, (rec["y0"] - 1) * spacing,
             (rec["z0"] - 1) * spacing))
        mc = vtk.vtkFlyingEdges3D() if hasattr(vtk, "vtkFlyingEdges3D") \
            else vtk.vtkMarchingCubes()
        mc.SetInputData(vol)
        mc.SetValue(0, 0.5)
        mc.ComputeNormalsOn()
        mc.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(mc.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        c = LABEL_RGB[i + 1] if i + 1 < len(LABEL_RGB) else (0.6, 0.6, 0.6)
        actor.GetProperty().SetColor(*c)
        actor.GetProperty().SetOpacity(surf_alpha(int(rec["m"].sum())))
        ren.AddActor(actor)

    # Identical to orient3D in app.js: right-anterior-superior oblique, +z up.
    cx, cy, cz = ((d - 1) * spacing / 2 for d in dims)
    dist = max(dims) * spacing * 1.6
    cam = ren.GetActiveCamera()
    cam.SetFocalPoint(cx, cy, cz)
    cam.SetPosition(cx - 0.8 * dist, cy - 0.8 * dist, cz + 0.5 * dist)
    cam.SetViewUp(0, 0, 1)
    ren.ResetCamera()
    # app.js homes to 1.35, tuned for a wide panel. ResetCamera fits the
    # bounding sphere, so a square thumbnail wastes the corners; the default
    # here is tighter. Orientation is what has to match the interactive view,
    # not framing.
    cam.Zoom(zoom)
    ren.ResetCameraClippingRange()
    return ren


def grab(ren, cell: int) -> np.ndarray:
    """Render one scene offscreen and return an (cell, cell, 4) uint8 image."""
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetAlphaBitPlanes(1)
    rw.SetMultiSamples(8)
    rw.AddRenderer(ren)
    rw.SetSize(cell, cell)
    rw.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.SetInputBufferTypeToRGBA()
    w2i.ReadFrontBufferOff()
    w2i.Update()
    out = w2i.GetOutput()
    w, h, _ = out.GetDimensions()
    arr = numpy_support.vtk_to_numpy(out.GetPointData().GetScalars())
    arr = arr.reshape(h, w, -1)[::-1]          # VTK origin is bottom-left
    rw.Finalize()
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", required=True)
    ap.add_argument("--cell", type=int, default=128, help="thumbnail size, px")
    ap.add_argument("--limit", type=int, default=0, help="render only the first N")
    ap.add_argument("--zoom", type=float, default=1.9, help="camera zoom")
    ap.add_argument("--real", action="store_true",
                    help="also render the real cases (default: synthetic only)")
    ap.add_argument("--preview", action="store_true",
                    help="also write thumbs_preview.png, flattened onto the "
                         "page background so the atlas can be judged as shown")
    args = ap.parse_args()

    viewer = Path(args.viewer)
    emb = json.loads((viewer / "data" / "embedding.json").read_text())
    structures = list(emb["structures"])
    dims, spacing = emb["dims"], float(emb["spacing"])
    n_vox = dims[0] * dims[1] * dims[2]

    cases = [c["name"] for c in emb["cases"] if args.real or not c["real"]]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise SystemExit("no cases to render")

    mask_dir = viewer / "data" / "mask"
    if not mask_dir.exists():
        raise SystemExit(f"no {mask_dir}; run export_masks.py first")

    cell = args.cell
    cols = int(np.ceil(np.sqrt(len(cases))))
    rows = int(np.ceil(len(cases) / cols))
    atlas = np.zeros((rows * cell, cols * cell, 4), np.uint8)
    print(f"{len(cases)} cases -> {cols}x{rows} atlas of {cell} px cells "
          f"({cols*cell}x{rows*cell})")

    t0 = time.time()
    for n, name in enumerate(cases):
        ct = np.fromfile(viewer / "data" / "vol" / f"{name}_ct.bin", np.uint8)
        if ct.size != n_vox:
            raise SystemExit(f"{name}: {ct.size} voxels, expected {n_vox}")
        ct = ct.reshape(dims[2], dims[1], dims[0])
        masks = read_masks(mask_dir / f"{name}.bin")
        tile = grab(build_scene(ct, masks, dims, spacing, args.zoom), cell)
        r, c = divmod(n, cols)
        atlas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = tile[..., :4]
        if (n + 1) % 25 == 0:
            per = (time.time() - t0) / (n + 1)
            print(f"  [{n+1}/{len(cases)}] {per:.2f} s/case, "
                  f"{per*(len(cases)-n-1)/60:.1f} min left", flush=True)

    from PIL import Image
    png = viewer / "data" / "thumbs.png"
    Image.fromarray(atlas, "RGBA").save(png, optimize=True)
    if args.preview:
        # The atlas is mostly transparent, so on a white background the faint
        # air planes look like solid grey slabs. Flatten onto the page colour
        # to see what the reader will actually see.
        bg = np.empty_like(atlas[..., :3])
        bg[:] = (np.array([0.04, 0.05, 0.07]) * 255).astype(np.uint8)
        a = atlas[..., 3:4].astype(np.float32) / 255.0
        flat = (atlas[..., :3] * a + bg * (1 - a)).astype(np.uint8)
        Image.fromarray(flat, "RGB").save(viewer / "data" / "thumbs_preview.png")
    (viewer / "data" / "thumbs.json").write_text(json.dumps({
        "cell": cell, "cols": cols, "rows": rows, "order": cases}, indent=1))
    print(f"\n{png}  {png.stat().st_size/1e6:.1f} MB  "
          f"({time.time()-t0:.0f} s total)")


if __name__ == "__main__":
    sys.exit(main())
