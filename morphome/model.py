"""A 3D VAE over joint CT + organ-at-risk segmentation volumes.

Latent design
-------------
The latent is a single global vector, not a spatial feature map. A spatial
latent reconstructs better, but sampling one from N(0, I) produces spatially
incoherent anatomy unless you fit a second-stage prior over it. The goal here is
a low-dimensional *anatomy manifold* that can be sampled and interpolated
directly, so the bottleneck is global by construction.

The encoder sees 1 CT channel plus one binary mask per structure, and the decoder
emits 1 CT channel plus the matching mask logits, so the model learns image and
shape jointly rather than treating segmentation as a downstream task. With
`--with-bone` the structure list gains a derived bone channel (see
`constants.MODEL_STRUCTURES`), making the counts 11 in / 1 + 10 out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import N_STRUCT


def _norm(ch: int) -> nn.Module:
    # GroupNorm rather than BatchNorm: batches are 2-4 volumes, far too small
    # for stable batch statistics.
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class ResBlock3d(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        self.n1 = _norm(cin)
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.n2 = _norm(cout)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


@dataclass
class VAEConfig:
    in_channels: int = 1 + N_STRUCT
    out_ct_channels: int = 1
    out_label_channels: int = N_STRUCT
    latent_dim: int = 256
    base_channels: int = 16
    # Logit bias at init, per output label channel. OARs occupy 6e-5 to 3e-3 of
    # the volume; bone occupies ~5-10 %, so it needs a far less negative start.
    oar_bias_init: float = -6.0
    derived_bias_init: float = -2.4
    # One entry per resolution step: 128 -> 64 -> 32 -> 16 -> 8 -> 4
    channel_mult: tuple[int, ...] = (1, 2, 4, 8, 12, 16)
    blocks_per_stage: int = 1
    dropout: float = 0.0
    input_size: int = 128
    # (x, y, z) in voxels. None keeps the old cubic behaviour, so checkpoints
    # saved before the frame became anisotropic still load.
    input_dims: tuple[int, int, int] | None = None

    @property
    def dims_xyz(self) -> tuple[int, int, int]:
        return tuple(self.input_dims) if self.input_dims else (self.input_size,) * 3

    @property
    def bottleneck_size(self) -> int:
        return self.input_size // 2 ** (len(self.channel_mult) - 1)

    @property
    def bottleneck_shape(self) -> tuple[int, int, int]:
        """Bottleneck feature-map shape as torch orders it: (D, H, W) = (z, y, x)."""
        f = 2 ** (len(self.channel_mult) - 1)
        d = self.dims_xyz
        return (d[2] // f, d[1] // f, d[0] // f)

    @property
    def bottleneck_numel(self) -> int:
        s = self.bottleneck_shape
        return s[0] * s[1] * s[2]

    @property
    def bottleneck_channels(self) -> int:
        return self.base_channels * self.channel_mult[-1]


class Encoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        chans = [cfg.base_channels * m for m in cfg.channel_mult]

        self.stem = nn.Conv3d(cfg.in_channels, chans[0], 3, padding=1)

        stages = []
        for i in range(len(chans) - 1):
            blocks = [ResBlock3d(chans[i], chans[i], cfg.dropout)
                      for _ in range(cfg.blocks_per_stage)]
            # Strided conv downsample, one resolution step per stage.
            blocks.append(nn.Conv3d(chans[i], chans[i + 1], 4, stride=2, padding=1))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

        self.out_norm = _norm(chans[-1])
        flat = chans[-1] * cfg.bottleneck_numel
        self.to_mu = nn.Linear(flat, cfg.latent_dim)
        self.to_logvar = nn.Linear(flat, cfg.latent_dim)

    def forward(self, x):
        h = self.stem(x)
        for s in self.stages:
            h = s(h)
        h = F.silu(self.out_norm(h)).flatten(1)
        mu = self.to_mu(h)
        # Clamp keeps the reparameterisation numerically sane early in training,
        # when an untrained encoder can emit extreme log-variances.
        logvar = self.to_logvar(h).clamp(-8.0, 8.0)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        chans = [cfg.base_channels * m for m in cfg.channel_mult]
        self.chans = chans
        self.from_latent = nn.Linear(cfg.latent_dim, chans[-1] * cfg.bottleneck_numel)

        stages = []
        for i in range(len(chans) - 1, 0, -1):
            blocks = [ResBlock3d(chans[i], chans[i], cfg.dropout)
                      for _ in range(cfg.blocks_per_stage)]
            # Nearest-neighbour upsample + conv rather than transposed conv,
            # which checkerboards badly in 3D.
            blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
            blocks.append(nn.Conv3d(chans[i], chans[i - 1], 3, padding=1))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

        self.out_norm = _norm(chans[0])
        self.to_ct = nn.Conv3d(chans[0], cfg.out_ct_channels, 3, padding=1)
        self.to_labels = nn.Conv3d(chans[0], cfg.out_label_channels, 3, padding=1)
        # Start every organ logit strongly negative. Organs occupy 1e-4 to 3e-3
        # of the volume; without this the first steps emit ~0.5 everywhere and
        # the Dice term produces enormous, unstable gradients.
        nn.init.constant_(self.to_labels.bias, cfg.oar_bias_init)
        # Derived channels (bone) are two orders of magnitude denser. Starting
        # them at -6.0 too would stall them at p~0.002 for hundreds of steps.
        if cfg.out_label_channels > N_STRUCT:
            with torch.no_grad():
                self.to_labels.bias[N_STRUCT:] = cfg.derived_bias_init

    def forward(self, z):
        h = self.from_latent(z).view(-1, self.chans[-1], *self.cfg.bottleneck_shape)
        for s in self.stages:
            h = s(h)
        h = F.silu(self.out_norm(h))
        ct = torch.tanh(self.to_ct(h))
        label_logits = self.to_labels(h)
        return ct, label_logits


class HNVAE(nn.Module):
    def __init__(self, cfg: VAEConfig | None = None):
        super().__init__()
        self.cfg = cfg or VAEConfig()
        self.encoder = Encoder(self.cfg)
        self.decoder = Decoder(self.cfg)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        ct, label_logits = self.decoder(z)
        return {"ct": ct, "label_logits": label_logits, "mu": mu, "logvar": logvar, "z": z}

    @torch.no_grad()
    def sample(self, n: int, device, temperature: float = 1.0):
        z = torch.randn(n, self.cfg.latent_dim, device=device) * temperature
        ct, logits = self.decoder(z)
        return ct, torch.sigmoid(logits), z

    @torch.no_grad()
    def reconstruct(self, x, use_mean: bool = True):
        mu, logvar = self.encoder(x)
        z = mu if use_mean else self.reparameterize(mu, logvar)
        ct, logits = self.decoder(z)
        return ct, torch.sigmoid(logits), z


def build_input(ct: torch.Tensor, labels: torch.Tensor,
                presence: torch.Tensor | None = None) -> torch.Tensor:
    """Assemble the 10-channel encoder input.

    Absent contours are zeroed explicitly. They are already zero in the cache,
    but doing it here makes the contract obvious and survives augmentation
    resampling, which can smear a channel that should stay empty.
    """
    if presence is not None:
        labels = labels * presence.view(labels.shape[0], -1, 1, 1, 1)
    return torch.cat([ct, labels], dim=1)


def count_parameters(model: nn.Module) -> dict:
    enc = sum(p.numel() for p in model.encoder.parameters())
    dec = sum(p.numel() for p in model.decoder.parameters())
    return {"encoder": enc, "decoder": dec, "total": enc + dec}
