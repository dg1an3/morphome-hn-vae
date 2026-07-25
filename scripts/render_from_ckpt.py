"""Render reconstruction and prior-sample figures from a checkpoint on demand,
without disturbing a run in progress.

Copies the checkpoint before loading, so a concurrent trainer writing to
last.pt cannot produce a torn read.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from morphome.constants import MODEL_STRUCTURES, N_STRUCT
from morphome.data import HNCache, default_split
from morphome.model import HNVAE, VAEConfig, build_input
from morphome.viz import save_recon_figure, save_sample_figure


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/full/last.pt")
    ap.add_argument("--cache", default=r"E:\datasets\medical\morphome_cache\hn_128_1.6mm")
    ap.add_argument("--out", default="notes/live")
    ap.add_argument("--n-recon", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--raw-weights", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "ckpt.pt"
        shutil.copy2(args.ckpt, tmp)
        ck = torch.load(tmp, map_location=device, weights_only=False)

    cfg = VAEConfig(**ck["cfg"])
    model = HNVAE(cfg).to(device)
    use_ema = (not args.raw_weights) and ck.get("ema") is not None
    model.load_state_dict(ck["ema"] if use_ema else ck["model"])
    model.eval()
    epoch = ck["epoch"]
    print(f"checkpoint epoch={epoch} step={ck['step']} weights={'ema' if use_ema else 'raw'}")

    val_cases = ck["args"]["_val_cases"] if "_val_cases" in ck.get("args", {}) else None
    if val_cases is None:
        all_cases = sorted(p.stem for p in Path(args.cache).glob("0522c*.npz"))
        _, val_cases = default_split(all_cases, ck["args"].get("n_val", 8),
                                     ck["args"].get("split_seed", 0))
    n_lab = cfg.out_label_channels
    names = MODEL_STRUCTURES[:n_lab]
    ds = HNCache(args.cache, val_cases, with_bone=n_lab > N_STRUCT)
    loader = DataLoader(ds, batch_size=args.n_recon, shuffle=False)
    print(f"val cases: {val_cases}")

    tag = f"ep{epoch:05d}{'_ema' if use_ema else '_raw'}"
    save_recon_figure(model, loader, device, out / f"recon_{tag}.png", n=args.n_recon)
    save_sample_figure(model, device, out / f"sample_{tag}.png",
                       n=args.n_samples, temperature=args.temperature)
    print(f"wrote {out/f'recon_{tag}.png'}")
    print(f"wrote {out/f'sample_{tag}.png'}")

    # quick per-organ Dice on the val set with these weights
    dice_sum = np.zeros(n_lab)
    dice_cnt = np.zeros(n_lab)
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=2):
            ct = b["ct"].to(device); lab = b["labels"].to(device)
            pres = b["presence"].to(device)
            _, prob, _ = model.reconstruct(build_input(ct, lab, pres))
            pred = (prob > 0.5).float()
            dims = (2, 3, 4)
            inter = (pred * lab).sum(dims)
            den = pred.sum(dims) + lab.sum(dims)
            d = ((2 * inter + 1e-6) / (den + 1e-6)).cpu().numpy()
            m = pres.cpu().numpy()
            dice_sum += (d * m).sum(0); dice_cnt += m.sum(0)
    per = dice_sum / np.maximum(dice_cnt, 1)
    print(f"\nval Dice @0.5 threshold (epoch {epoch}, {'ema' if use_ema else 'raw'}):")
    for i, n in enumerate(names):
        print(f"  {n:<18} {per[i]:.3f}")
    # Mean over OARs only, so it stays comparable to runs without bone.
    print(f"  {'MEAN (OARs)':<18} {per[:N_STRUCT].mean():.3f}")


if __name__ == "__main__":
    main()
