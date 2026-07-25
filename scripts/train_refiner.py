"""Train the conditional diffusion refiner over a frozen VAE.

Training pairs are `(VAE output, real CT)` -- the VAE's *own* reconstruction, not
a synthetic blur. This is load-bearing: a Gaussian-sigma-1.6 stand-in predicted
the composite renderer would be worth 136 -> 233 mean |grad HU|, and against the
real decoder it delivered 230 -> 238. The decoder's failure mode is not a
Gaussian blur and a refiner trained on the wrong corruption inherits the same
error.

The conditioning distribution at *generation* time is a prior sample, which is
systematically smoother than a reconstruction, so z is jittered around each
case's posterior mean during training. The anatomy stays close enough to the case
that the real CT remains a valid target, while the conditioning statistics widen
toward what the sampler actually produces.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from morphome.constants import BONE_HU_THRESHOLD, N_STRUCT
from morphome.data import MODEL_FLIP_PERM, HNCache, default_split, denormalize_hu
from morphome.diffusion import Schedule, UNet3d, UNetConfig, count_parameters, ddim_sample
from morphome.ema import ModelEMA
from morphome.model import build_input
from explore_latent import load_model, wants_bone


def get_args(argv=None):
    p = argparse.ArgumentParser(description="Train the diffusion refiner.")
    p.add_argument("--vae-ckpt", default="runs/v2_bone/last.pt")
    p.add_argument("--cache", default=r"E:\datasets\medical\morphome_cache\hn_128_1.6mm")
    p.add_argument("--out", default="runs/refiner")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--patch", type=int, default=64)
    p.add_argument("--batch", type=int, default=8, help="patches per step")
    p.add_argument("--vols-per-batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--z-jitter", type=float, default=0.3,
                   help="max z noise as a fraction of the per-dim posterior std")
    p.add_argument("--p-bone-patch", type=float, default=0.6,
                   help="fraction of patches centred on bone rather than body")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--lr-warmup", type=int, default=500)
    p.add_argument("--amp", default="bf16", choices=["off", "bf16", "fp16"])
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-minutes", type=float, default=0.0)
    return p.parse_args(argv)


@torch.no_grad()
def decode_conditioning(vae, z, device):
    """z -> (conditioning CT, mask probabilities). The VAE is frozen throughout."""
    ct, logits = vae.decoder(z.to(device))
    return ct, torch.sigmoid(logits)


def sample_patches(cond: torch.Tensor, target: torch.Tensor, body: torch.Tensor,
                   n: int, size: int, p_bone: float, rng: np.random.RandomState):
    """Crop `n` patches per volume, biased toward bone.

    Uniform sampling would spend most of the capacity on flat soft tissue and
    air; the whole point of the refiner is the bone edge, which occupies ~5 % of
    the volume.
    """
    b, _, D, H, W = cond.shape
    half = size // 2
    bone = cond[:, 1 + N_STRUCT] > 0.5
    out_c, out_t = [], []
    for i in range(b):
        # One device->host transfer per volume, not one per patch.
        pool_bone = torch.nonzero(bone[i], as_tuple=False).cpu().numpy()
        pool_body = torch.nonzero(body[i, 0] > 0.5, as_tuple=False).cpu().numpy()
        for _ in range(n):
            pool = pool_bone if (rng.rand() < p_bone and len(pool_bone) > 0) else pool_body
            if len(pool) == 0:
                c = np.array([D // 2, H // 2, W // 2])
            else:
                c = pool[rng.randint(len(pool))]
            z0, y0, x0 = [int(np.clip(c[k] - half, 0, (D, H, W)[k] - size)) for k in range(3)]
            sl = (slice(z0, z0 + size), slice(y0, y0 + size), slice(x0, x0 + size))
            out_c.append(cond[i][(slice(None), *sl)])
            out_t.append(target[i][(slice(None), *sl)])
    return torch.stack(out_c), torch.stack(out_t)


def bone_sharpness(ct_norm: np.ndarray) -> float:
    """Mean |grad HU| across the surface of the volume's own >300 HU bone."""
    hu = denormalize_hu(ct_norm)
    m = hu > BONE_HU_THRESHOLD
    if m.sum() < 1000:
        return float("nan")
    g = np.sqrt(sum(x ** 2 for x in np.gradient(hu)))
    return float(g[m].mean())


def main(argv=None) -> None:
    args = get_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    vae, vcfg, _ = load_model(args.vae_ckpt, device, prefer_ema=True)
    for p in vae.parameters():
        p.requires_grad_(False)
    if not wants_bone(vcfg):
        raise SystemExit("refiner expects a bone-channel VAE; use --with-bone weights")
    n_lab = vcfg.out_label_channels

    all_cases = sorted(p.stem for p in Path(args.cache).glob("0522c*.npz"))
    train_cases, val_cases = default_split(all_cases, 8, 0)
    train_ds = HNCache(args.cache, train_cases, with_bone=True)
    val_ds = HNCache(args.cache, val_cases, with_bone=True)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    # Posterior means, computed once: the refiner conditions on decodes of these.
    mus = []
    with torch.no_grad():
        for i in range(len(train_ds)):
            s = train_ds[i]
            mu, _ = vae.encoder(build_input(s["ct"][None].to(device),
                                            s["labels"][None].to(device),
                                            s["presence"][None].to(device)))
            mus.append(mu[0])
    mu_all = torch.stack(mus)
    z_std = mu_all.std(0)
    print(f"posterior means {tuple(mu_all.shape)}  per-dim std "
          f"[{float(z_std.min()):.2f}, {float(z_std.max()):.2f}]")

    ucfg = UNetConfig(in_channels=1 + 1 + n_lab, base_channels=args.base_channels,
                      blocks_per_stage=args.blocks, dropout=args.dropout)
    model = UNet3d(ucfg).to(device)
    print(f"refiner UNet: {count_parameters(model)/1e6:.2f}M params, "
          f"in_channels={ucfg.in_channels}, patch {args.patch}^3")

    sched = Schedule(args.timesteps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.99))
    ema = ModelEMA(model, decay=args.ema_decay)
    amp_dtype = {"off": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16"))
    writer = SummaryWriter(str(outdir / "tb"))
    (outdir / "config.json").write_text(json.dumps(
        {"args": vars(args), "unet": asdict(ucfg), "vae_ckpt": args.vae_ckpt,
         "train_cases": train_cases, "val_cases": val_cases}, indent=2, default=str))

    n_per_vol = max(1, args.batch // args.vols_per_batch)
    t0 = time.time()
    run_loss, run_n = 0.0, 0

    for step in range(args.steps):
        lr = args.lr * min(1.0, (step + 1) / max(1, args.lr_warmup))
        for g in opt.param_groups:
            g["lr"] = lr

        idx = rng.choice(len(train_ds), args.vols_per_batch, replace=False)
        items = [train_ds[int(i)] for i in idx]
        target = torch.stack([it["ct"] for it in items]).to(device)
        body = torch.stack([it["body"] for it in items]).to(device)

        # Jitter z toward the sampler's distribution, not just the posterior mean.
        z = mu_all[idx].clone()
        z = z + torch.randn_like(z) * z_std * (rng.rand() * args.z_jitter)
        with torch.no_grad():
            cond_ct, cond_prob = decode_conditioning(vae, z, device)
        cond = torch.cat([cond_ct, cond_prob], dim=1)

        if rng.rand() < 0.5:                      # L-R flip, masks permuted
            cond = cond.flip(-1)
            cond = torch.cat([cond[:, :1], cond[:, 1:][:, list(MODEL_FLIP_PERM[:n_lab])]], 1)
            target = target.flip(-1)
            body = body.flip(-1)

        c_p, t_p = sample_patches(cond, target, body, n_per_vol, args.patch,
                                  args.p_bone_patch, rng)

        t = torch.randint(0, args.timesteps, (t_p.shape[0],), device=device)
        noise = torch.randn_like(t_p)
        x_t, v_target = sched.add_noise(t_p, noise, t)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            v_pred = model(x_t, t, c_p)
            loss = torch.nn.functional.mse_loss(v_pred.float(), v_target.float())

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        ema.update(model)

        run_loss += float(loss.detach())
        run_n += 1
        if (step + 1) % args.log_every == 0:
            writer.add_scalar("train/loss", run_loss / run_n, step)
            writer.add_scalar("train/grad_norm", float(gn), step)
            writer.add_scalar("train/lr", lr, step)
            print(f"step {step+1:6d}/{args.steps} | v-mse {run_loss/run_n:.4f} "
                  f"gn {float(gn):.2f} | {(time.time()-t0)/60:.1f}m", flush=True)
            run_loss, run_n = 0.0, 0

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            model.eval()
            with torch.no_grad():
                s = val_ds[0]
                mu, _ = vae.encoder(build_input(s["ct"][None].to(device),
                                                s["labels"][None].to(device),
                                                s["presence"][None].to(device)))
                c_ct, c_pr = decode_conditioning(vae, mu, device)
                c_full = torch.cat([c_ct, c_pr], dim=1)
                ref = ddim_sample(ema.module, sched, c_full, steps=args.sample_steps)
            sh_raw = bone_sharpness(c_ct[0, 0].float().cpu().numpy())
            sh_ref = bone_sharpness(ref[0, 0].float().cpu().numpy())
            sh_real = bone_sharpness(s["ct"][0].numpy())
            l1 = float((ref[0, 0].cpu() - s["ct"][0]).abs().mean())
            writer.add_scalar("val/sharpness_raw", sh_raw, step)
            writer.add_scalar("val/sharpness_refined", sh_ref, step)
            writer.add_scalar("val/l1_to_real", l1, step)
            print(f"  [eval] |grad HU| on bone: VAE {sh_raw:.0f} -> refined "
                  f"{sh_ref:.0f}   (real {sh_real:.0f})   L1 to real {l1:.4f}",
                  flush=True)
            model.train()

        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                        "unet": asdict(ucfg), "step": step + 1,
                        "timesteps": args.timesteps, "vae_ckpt": args.vae_ckpt,
                        "args": vars(args)}, outdir / "last.pt")

        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            print(f"reached --max-minutes {args.max_minutes}; stopping at step {step+1}")
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                        "unet": asdict(ucfg), "step": step + 1,
                        "timesteps": args.timesteps, "vae_ckpt": args.vae_ckpt,
                        "args": vars(args)}, outdir / "last.pt")
            break

    writer.close()
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
