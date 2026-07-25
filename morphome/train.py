"""Training loop for the head-and-neck 3D VAE."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .constants import MODEL_STRUCTURES, N_MODEL_STRUCT, N_STRUCT
from .data import HNCache, augment, default_split
from .ema import ModelEMA
from .losses import beta_schedule, vae_loss
from .model import HNVAE, VAEConfig, build_input, count_parameters
from .viz import save_recon_figure, save_sample_figure


def lr_at(step: int, total_steps: int, base_lr: float,
          warmup_steps: int = 500, min_ratio: float = 0.05) -> float:
    """Linear warm-up then cosine decay."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    t = min(1.0, max(0.0, t))
    cos = 0.5 * (1.0 + math.cos(math.pi * t))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cos)


def get_args(argv=None):
    p = argparse.ArgumentParser(description="Train the HN 3D VAE.")
    p.add_argument("--cache", default=r"E:\datasets\medical\morphome_cache\hn_128_1.6mm")
    p.add_argument("--out", default="runs/hnvae")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--n-val", type=int, default=8)
    p.add_argument("--split-seed", type=int, default=0)

    p.add_argument("--beta-max", type=float, default=1e-4)
    p.add_argument("--beta-warmup", type=int, default=20000)
    p.add_argument("--beta-start", type=int, default=2000)
    p.add_argument("--free-bits", type=float, default=0.5)
    p.add_argument("--w-ct", type=float, default=1.0)
    p.add_argument("--w-dice", type=float, default=1.0)
    p.add_argument("--w-bce", type=float, default=0.1)
    p.add_argument("--body-weight", type=float, default=4.0)
    p.add_argument("--pos-weight", type=float, default=50.0)
    p.add_argument("--with-bone", action="store_true",
                   help="add a derived, Dice-supervised bone channel (CT > "
                        "BONE_HU_THRESHOLD) as a tenth structure")
    p.add_argument("--bone-pos-weight", type=float, default=2.0,
                   help="BCE positive weight for the bone channel; it is ~100x "
                        "denser than an OAR, so --pos-weight does not apply")

    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--rot-deg", type=float, default=8.0)
    p.add_argument("--shift-frac", type=float, default=0.06)

    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--lr-warmup", type=int, default=500)
    p.add_argument("--lr-min-ratio", type=float, default=0.05)
    p.add_argument("--no-lr-decay", action="store_true")

    p.add_argument("--amp", default="bf16", choices=["off", "bf16", "fp16"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--fig-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default="")
    p.add_argument("--max-minutes", type=float, default=0.0,
                   help="stop cleanly after this many minutes (0 = no limit)")
    return p.parse_args(argv)


def move(batch, device):
    return (batch["ct"].to(device, non_blocking=True),
            batch["labels"].to(device, non_blocking=True),
            batch["presence"].to(device, non_blocking=True),
            batch["body"].to(device, non_blocking=True))


def label_channels(args) -> int:
    """Number of supervised mask channels, derived channels included."""
    return N_MODEL_STRUCT if args.with_bone else N_STRUCT


def pos_weight_vector(args, device):
    """Per-channel BCE positive weight, or a scalar when there is nothing dense."""
    if not args.with_bone:
        return args.pos_weight
    return torch.tensor([args.pos_weight] * N_STRUCT + [args.bone_pos_weight],
                        device=device)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, args):
    model.eval()
    n_lab = label_channels(args)
    pos_weight = pos_weight_vector(args, device)
    agg: dict[str, float] = {}
    dice_sum = torch.zeros(n_lab, device=device)
    dice_cnt = torch.zeros(n_lab, device=device)
    n = 0
    for batch in loader:
        ct, labels, presence, body = move(batch, device)
        x = build_input(ct, labels, presence)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(x)
            # Evaluate at beta=0: validation reconstruction quality is the
            # quantity of interest, and the KL term is logged separately.
            _, logs = vae_loss(out, ct, labels, presence, body, beta=0.0,
                               w_ct=args.w_ct, w_dice=args.w_dice, w_bce=args.w_bce,
                               free_bits=args.free_bits, body_weight=args.body_weight,
                               pos_weight=pos_weight)
        dice_sum += logs.pop("_dice_sum").float()
        dice_cnt += logs.pop("_dice_cnt").float()
        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + float(v)
        n += 1
    model.train()
    metrics = {k: v / max(n, 1) for k, v in agg.items()}
    per_organ = (dice_sum / dice_cnt.clamp(min=1)).cpu().numpy()
    return metrics, per_organ


def main(argv=None) -> None:
    args = get_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)

    all_cases = sorted(p.stem for p in Path(args.cache).glob("0522c*.npz"))
    train_cases, val_cases = default_split(all_cases, args.n_val, args.split_seed)
    print(f"train={len(train_cases)} val={len(val_cases)}")
    print(f"val cases: {val_cases}")

    train_ds = HNCache(args.cache, train_cases, with_bone=args.with_bone)
    val_ds = HNCache(args.cache, val_cases, with_bone=args.with_bone)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    n_lab = label_channels(args)
    names = MODEL_STRUCTURES[:n_lab]
    pos_weight = pos_weight_vector(args, device)
    cfg = VAEConfig(latent_dim=args.latent_dim, base_channels=args.base_channels,
                    dropout=args.dropout, in_channels=1 + n_lab,
                    out_label_channels=n_lab)
    model = HNVAE(cfg).to(device)
    model = model.to(memory_format=torch.channels_last_3d)
    counts = count_parameters(model)
    print(f"params: encoder={counts['encoder']/1e6:.2f}M "
          f"decoder={counts['decoder']/1e6:.2f}M total={counts['total']/1e6:.2f}M")
    print(f"bottleneck: {cfg.bottleneck_channels}ch @ {cfg.bottleneck_size}^3 "
          f"-> latent {cfg.latent_dim}")

    if args.compile:
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.99))
    amp_dtype = {"off": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16"))

    writer = SummaryWriter(str(outdir / "tb"))
    (outdir / "config.json").write_text(json.dumps(
        {"args": vars(args), "model": asdict(cfg),
         "train_cases": train_cases, "val_cases": val_cases}, indent=2, default=str))

    ema = None if args.no_ema else ModelEMA(model, decay=args.ema_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch

    start_epoch, step = 0, 0
    best_val = float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        best_val = ck.get("best_val", best_val)
        if ema is not None and ck.get("ema") is not None:
            ema.load_state_dict(ck["ema"])
            ema.step_count = ck.get("ema_steps", step)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    def snapshot(epoch, step):
        d = {"model": model.state_dict(), "opt": opt.state_dict(),
             "epoch": epoch, "step": step, "cfg": asdict(cfg),
             "best_val": best_val, "args": vars(args)}
        if ema is not None:
            d["ema"] = ema.state_dict()
            d["ema_steps"] = ema.step_count
        return d

    t0 = time.time()
    stop = False
    for epoch in range(start_epoch, args.epochs):
        ep_logs: dict[str, float] = {}
        nb = 0
        opt.zero_grad(set_to_none=True)

        for it, batch in enumerate(train_loader):
            ct, labels, presence, body = move(batch, device)
            if not args.no_augment:
                ct, labels, body = augment(ct, labels, body,
                                           rot_deg=args.rot_deg,
                                           shift_frac=args.shift_frac)
            x = build_input(ct, labels, presence)
            x = x.to(memory_format=torch.channels_last_3d)

            if not args.no_lr_decay:
                cur_lr = lr_at(step, total_steps, args.lr,
                               args.lr_warmup, args.lr_min_ratio)
                for g in opt.param_groups:
                    g["lr"] = cur_lr
            else:
                cur_lr = args.lr

            beta = beta_schedule(step, args.beta_warmup, args.beta_max, args.beta_start)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(x)
                loss, logs = vae_loss(out, ct, labels, presence, body, beta=beta,
                                      w_ct=args.w_ct, w_dice=args.w_dice,
                                      w_bce=args.w_bce, free_bits=args.free_bits,
                                      body_weight=args.body_weight,
                                      pos_weight=pos_weight)

            scaler.scale(loss / args.accum).backward()
            if (it + 1) % args.accum == 0:
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                logs["grad_norm"] = gn.detach()
                if ema is not None:
                    ema.update(model)

            logs.pop("_dice_sum")
            logs.pop("_dice_cnt")
            logs["lr"] = torch.as_tensor(cur_lr)
            for k, v in logs.items():
                ep_logs[k] = ep_logs.get(k, 0.0) + float(v)
            nb += 1
            step += 1

        for k in ep_logs:
            ep_logs[k] /= max(nb, 1)
        for k, v in ep_logs.items():
            writer.add_scalar(f"train/{k}", v, epoch)

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            vm, per_organ = evaluate(model, val_loader, device, amp_dtype, args)
            for k, v in vm.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            for i, name in enumerate(names):
                writer.add_scalar(f"val_dice/{name}", per_organ[i], epoch)
            # Mean over the OARs only, so the number stays comparable across
            # runs with and without the derived bone channel (bone scores ~0.9
            # and would inflate a 10-channel mean by ~0.03).
            mean_dice = float(np.mean(per_organ[:N_STRUCT]))
            writer.add_scalar("val/mean_dice", mean_dice, epoch)
            bone_dice = float(per_organ[N_STRUCT]) if args.with_bone else float("nan")
            if args.with_bone:
                writer.add_scalar("val/bone_dice", bone_dice, epoch)

            ema_dice = float("nan")
            ema_bone = float("nan")
            if ema is not None:
                evm, eper = evaluate(ema.module, val_loader, device, amp_dtype, args)
                for k, v in evm.items():
                    writer.add_scalar(f"val_ema/{k}", v, epoch)
                ema_dice = float(np.mean(eper[:N_STRUCT]))
                writer.add_scalar("val_ema/mean_dice", ema_dice, epoch)
                if args.with_bone:
                    ema_bone = float(eper[N_STRUCT])
                    writer.add_scalar("val_ema/bone_dice", ema_bone, epoch)

            el = time.time() - t0
            bone_str = (f" bone {bone_dice:.3f} ema_bone {ema_bone:.3f}"
                        if args.with_bone else "")
            print(f"ep {epoch:5d} step {step:7d} | "
                  f"train loss {ep_logs.get('loss', 0):.4f} ct {ep_logs.get('ct_l1', 0):.4f} "
                  f"dice_l {ep_logs.get('dice_loss', 0):.4f} kl {ep_logs.get('kl', 0):.1f} "
                  f"beta {ep_logs.get('beta', 0):.2e} act {ep_logs.get('active_dims', 0):.0f} | "
                  f"VAL ct {vm['ct_l1']:.4f} dice {mean_dice:.3f} "
                  f"ema_dice {ema_dice:.3f}{bone_str} | {el/60:.1f}m",
                  flush=True)
            print("        per-organ val dice: " + "  ".join(
                f"{n.split('_')[0][:6]}{'_'+n.split('_')[1] if '_' in n else ''}"
                f"={per_organ[i]:.3f}" for i, n in enumerate(names)), flush=True)

            score = vm["ct_l1"] + vm["dice_loss"]
            if score < best_val:
                best_val = score
                torch.save(snapshot(epoch, step), outdir / "best.pt")

        if epoch % args.fig_every == 0 or epoch == args.epochs - 1:
            save_recon_figure(model, val_loader, device,
                              outdir / "figures" / f"recon_{epoch:05d}.png")
            # Sample from EMA weights: the raw iterate produces noticeably
            # noisier prior samples.
            sample_src = ema.module if ema is not None else model
            save_sample_figure(sample_src, device,
                               outdir / "figures" / f"sample_{epoch:05d}.png")

        if epoch % args.ckpt_every == 0 or epoch == args.epochs - 1:
            torch.save(snapshot(epoch, step), outdir / "last.pt")

        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            print(f"reached --max-minutes {args.max_minutes}; stopping at epoch {epoch}")
            stop = True

        if stop:
            torch.save(snapshot(epoch, step), outdir / "last.pt")
            break

    writer.close()
    print(f"done in {(time.time()-t0)/60:.1f} min; best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
