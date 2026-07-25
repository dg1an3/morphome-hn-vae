"""Bone-specific evaluation: is the Dice-supervised mask a better bone surface
than thresholding the reconstructed CT?

Volumetric Dice barely penalises a blurred edge -- a mask can be a millimetre
soft all the way round and still score well -- so the metrics that decide TODO
item 1 are surface distances against the bone extracted from the *real* CT of
each held-out case.

Compared, per validation case:
  ct@300         reconstructed CT thresholded at BONE_HU_THRESHOLD. This is what
                 you get without a bone channel, and is the baseline `v1_ld32`
                 offers.
  bone_ch        the Dice-supervised bone channel (bone-channel models only).
  composite@300  the composite render re-thresholded; a consistency check on
                 morphome.render, not an independent result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from morphome.constants import BONE_HU_THRESHOLD, N_STRUCT
from morphome.data import HNCache, default_split, denormalize_hu
from morphome.model import build_input
from morphome.render import bone_composite
from explore_latent import load_model, wants_bone


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask, iterations=1, border_value=0)


def surface_metrics(pred: np.ndarray, ref: np.ndarray, spacing: float,
                    tolerances=(1.6, 3.2)) -> dict:
    """Symmetric surface distances in mm, plus surface Dice at tolerances."""
    out = {"dice": float("nan"), "msd": float("nan"), "hd95": float("nan")}
    if not pred.any() or not ref.any():
        return out
    inter = float((pred & ref).sum())
    out["dice"] = 2 * inter / float(pred.sum() + ref.sum())

    p_surf, r_surf = _surface(pred), _surface(ref)
    if not p_surf.any() or not r_surf.any():
        return out
    # Distance to the *other* surface, sampled on this one.
    d_to_r = distance_transform_edt(~r_surf, sampling=spacing)[p_surf]
    d_to_p = distance_transform_edt(~p_surf, sampling=spacing)[r_surf]
    both = np.concatenate([d_to_r, d_to_p])
    out["msd"] = float(both.mean())
    out["hd95"] = float(np.percentile(both, 95))
    for t in tolerances:
        # Surface Dice: fraction of both surfaces lying within t mm of the other.
        n_ok = float((d_to_r <= t).sum() + (d_to_p <= t).sum())
        out[f"sdice@{t}"] = n_ok / float(both.size)
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="checkpoint to evaluate; repeatable")
    ap.add_argument("--cache", default=r"E:\datasets\medical\morphome_cache\hn_128_1.6mm")
    ap.add_argument("--out", default="notes/bone_surface")
    ap.add_argument("--raw-weights", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_cases = sorted(p.stem for p in Path(args.cache).glob("0522c*.npz"))
    _, val_cases = default_split(all_cases, 8, 0)
    print(f"val cases: {val_cases}\n")

    results: dict[str, dict] = {}
    for ckpt in args.ckpt:
        model, cfg, ck = load_model(ckpt, device, prefer_ema=not args.raw_weights)
        has_bone = wants_bone(cfg)
        ds = HNCache(args.cache, val_cases, in_memory=False, with_bone=has_bone)
        spacing = float(ds.meta.get("grid", {}).get("spacing", 1.6))
        tag = f"{Path(ckpt).parent.name}/{Path(ckpt).stem}@ep{ck['epoch']}"

        per_variant: dict[str, list] = {}
        for i in range(len(ds)):
            s = ds[i]
            ct = s["ct"][None].to(device)
            lab = s["labels"][None].to(device)
            pres = s["presence"][None].to(device)
            rec_ct, prob, _ = model.reconstruct(build_input(ct, lab, pres))

            ref = (denormalize_hu(s["ct"][0].numpy()) > BONE_HU_THRESHOLD)
            rec_np = rec_ct[0, 0].float().cpu().numpy()
            variants = {"ct@300": denormalize_hu(rec_np) > BONE_HU_THRESHOLD}
            if has_bone:
                bone_p = prob[0, N_STRUCT].float().cpu().numpy()
                variants["bone_ch"] = bone_p > 0.5
                comp = bone_composite(rec_np, bone_p)
                variants["composite@300"] = denormalize_hu(comp) > BONE_HU_THRESHOLD

            for name, pred in variants.items():
                m = surface_metrics(pred, ref, spacing)
                m["case"] = s["case"]
                per_variant.setdefault(name, []).append(m)

        print(f"--- {tag} ---")
        summary = {}
        for name, rows in per_variant.items():
            agg = {k: float(np.mean([r[k] for r in rows]))
                   for k in rows[0] if k != "case"}
            summary[name] = {"mean": agg, "per_case": rows}
            print(f"  {name:<16} Dice {agg['dice']:.3f}  MSD {agg['msd']:5.2f} mm  "
                  f"HD95 {agg['hd95']:5.2f} mm  "
                  f"sDice@1.6 {agg['sdice@1.6']:.3f}  sDice@3.2 {agg['sdice@3.2']:.3f}")
        results[tag] = summary
        print()

    (out / "bone_surface.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out/'bone_surface.json'}")


if __name__ == "__main__":
    main()
