"""Conditional diffusion in voxel space, used as a refiner over the VAE output.

Why a refiner and not a decoder
-------------------------------
The VAE's CT channel is trained with L1, whose optimum is a conditional median,
so it emits the blur-averaged image and no amount of further training fixes it.
The organ and bone channels are Dice-supervised and come out committed. This
module learns `p(sharp CT | blurry CT, masks)` so the committed shape can be
spent on the intensity field -- the learned generalisation of
`morphome.render.bone_composite`, which does the same thing with three
hand-picked constants.

It refines rather than replaces because of the data: 40 training volumes cannot
support learning 3D anatomy from noise, but they contain thousands of 64^3
patches, and with the anatomy supplied as conditioning the model only has to
learn *local texture*. The latent manifold, the interpolation and the
PCA-Gaussian prior all keep working untouched; refinement is a post-step.

Parameterisation is `v` (Salimans & Ho) on a cosine schedule: better behaved than
epsilon at high noise and under the small step counts we sample with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------

def cosine_alphas_cumprod(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule."""
    t = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
    f = torch.cos((t + s) / (1.0 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    betas = torch.clamp(1.0 - ac[1:] / ac[:-1], 0.0, 0.999)
    return torch.cumprod(1.0 - betas, dim=0).float()


class Schedule:
    """Holds alphas_cumprod and the v <-> (x0, eps) conversions."""

    def __init__(self, timesteps: int = 1000, device=None):
        self.timesteps = timesteps
        self.ac = cosine_alphas_cumprod(timesteps).to(device)

    def to(self, device):
        self.ac = self.ac.to(device)
        return self

    def _broadcast(self, t: torch.Tensor, ndim: int) -> torch.Tensor:
        return self.ac[t].view(-1, *([1] * (ndim - 1)))

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        """Returns (x_t, v_target)."""
        ac = self._broadcast(t, x0.dim())
        sa, sb = ac.sqrt(), (1.0 - ac).sqrt()
        x_t = sa * x0 + sb * noise
        v = sa * noise - sb * x0
        return x_t, v

    def to_x0_eps(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor):
        ac = self._broadcast(t, x_t.dim())
        sa, sb = ac.sqrt(), (1.0 - ac).sqrt()
        x0 = sa * x_t - sb * v
        eps = sb * x_t + sa * v
        return x0, eps


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


def _norm(ch: int) -> nn.Module:
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class ResBlock(nn.Module):
    """Pre-norm residual block with FiLM-style timestep conditioning."""

    def __init__(self, cin: int, cout: int, temb: int, dropout: float = 0.0):
        super().__init__()
        self.n1 = _norm(cin)
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(temb, 2 * cout)
        self.n2 = _norm(cout)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()
        # Zero-init the output conv so the block starts as identity; with 40
        # volumes an untrained residual stack otherwise injects a lot of noise
        # into the first few hundred steps.
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(F.silu(temb))[:, :, None, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1.0 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return h + self.skip(x)


@dataclass
class UNetConfig:
    in_channels: int = 12          # 1 noisy CT + 1 conditioning CT + 10 masks
    out_channels: int = 1
    base_channels: int = 32
    channel_mult: tuple[int, ...] = (1, 2, 4)
    blocks_per_stage: int = 2
    dropout: float = 0.0
    temb_dim: int = 128


class UNet3d(nn.Module):
    """Fully convolutional, so it trains on 64^3 patches and runs on 128^3."""

    def __init__(self, cfg: UNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or UNetConfig()
        cfg = self.cfg
        chans = [cfg.base_channels * m for m in cfg.channel_mult]
        temb = cfg.temb_dim

        self.temb = nn.Sequential(
            nn.Linear(temb, temb * 4), nn.SiLU(), nn.Linear(temb * 4, temb * 4))
        temb4 = temb * 4

        self.stem = nn.Conv3d(cfg.in_channels, chans[0], 3, padding=1)

        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        skip_ch = [chans[0]]
        for i, ch in enumerate(chans):
            blocks = nn.ModuleList()
            cin = chans[i - 1] if i > 0 else chans[0]
            for b in range(cfg.blocks_per_stage):
                blocks.append(ResBlock(cin if b == 0 else ch, ch, temb4, cfg.dropout))
                skip_ch.append(ch)
            self.down.append(blocks)
            last = i == len(chans) - 1
            self.downsample.append(
                nn.Identity() if last else nn.Conv3d(ch, ch, 3, stride=2, padding=1))
            if not last:
                skip_ch.append(ch)

        self.mid1 = ResBlock(chans[-1], chans[-1], temb4, cfg.dropout)
        self.mid2 = ResBlock(chans[-1], chans[-1], temb4, cfg.dropout)

        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i, ch in reversed(list(enumerate(chans))):
            blocks = nn.ModuleList()
            for b in range(cfg.blocks_per_stage + 1):
                blocks.append(ResBlock(ch + skip_ch.pop(), ch, temb4, cfg.dropout))
            self.up.append(blocks)
            self.upsample.append(
                nn.Identity() if i == 0 else
                nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                              nn.Conv3d(ch, chans[i - 1], 3, padding=1)))

        self.out_norm = _norm(chans[0])
        self.out = nn.Conv3d(chans[0], cfg.out_channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        temb = self.temb(timestep_embedding(t, self.cfg.temb_dim))
        h = self.stem(torch.cat([x_t, cond], dim=1))
        skips = [h]
        for blocks, ds in zip(self.down, self.downsample):
            for blk in blocks:
                h = blk(h, temb)
                skips.append(h)
            if not isinstance(ds, nn.Identity):
                h = ds(h)
                skips.append(h)

        h = self.mid2(self.mid1(h, temb), temb)

        for blocks, us in zip(self.up, self.upsample):
            for blk in blocks:
                h = blk(torch.cat([h, skips.pop()], dim=1), temb)
            h = us(h)

        return self.out(F.silu(self.out_norm(h)))


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(model: UNet3d, sched: Schedule, cond: torch.Tensor,
                steps: int = 50, eta: float = 0.0, shape=None,
                generator=None, x_T: torch.Tensor | None = None) -> torch.Tensor:
    """Deterministic (eta=0) DDIM. `cond` is (B, C, D, H, W)."""
    device = cond.device
    b = cond.shape[0]
    shape = shape or (b, 1, *cond.shape[2:])
    x = torch.randn(shape, device=device, generator=generator) if x_T is None else x_T

    ts = torch.linspace(sched.timesteps - 1, 0, steps, device=device).long()
    for i, t in enumerate(ts):
        t_b = t.repeat(b)
        v = model(x, t_b, cond)
        x0, eps = sched.to_x0_eps(x, v, t_b)
        x0 = x0.clamp(-1.0, 1.0)
        if i == len(ts) - 1:
            x = x0
            break
        t_prev = ts[i + 1].repeat(b)
        ac_prev = sched.ac[t_prev].view(-1, 1, 1, 1, 1)
        sigma = eta * ((1 - ac_prev) / (1 - sched.ac[t_b].view(-1, 1, 1, 1, 1))).sqrt()
        sigma = sigma * (1 - sched.ac[t_b].view(-1, 1, 1, 1, 1) / ac_prev).sqrt()
        x = ac_prev.sqrt() * x0 + (1 - ac_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        if eta > 0:
            x = x + sigma * torch.randn(x.shape, device=device, generator=generator)
    return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
