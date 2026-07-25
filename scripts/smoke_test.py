"""Fast sanity checks: shapes, parameter count, VRAM, step time, and the
correctness of the L-R flip channel permutation."""

import time

import torch

from morphome.constants import (
    BONE_HU_THRESHOLD,
    MODEL_STRUCTURES,
    N_MODEL_STRUCT,
    N_STRUCT,
    STRUCTURES,
)
from morphome.data import FLIP_PERM, MODEL_FLIP_PERM, HNCache, augment, default_split
from morphome.losses import vae_loss
from morphome.model import HNVAE, VAEConfig, build_input, count_parameters

CACHE = r"E:\datasets\medical\morphome_cache\hn_128_1.6mm"


def check_flip_perm():
    print("--- L-R flip permutation ---")
    ok = True
    for i, name in enumerate(STRUCTURES):
        j = FLIP_PERM[i]
        partner = STRUCTURES[j]
        expect = name.replace("_L", "_R") if name.endswith("_L") else (
            name.replace("_R", "_L") if name.endswith("_R") else name)
        good = partner == expect
        ok &= good
        print(f"  {name:<18} -> {partner:<18} {'ok' if good else 'WRONG (expect ' + expect + ')'}")
    assert ok, "flip permutation is wrong"
    # involution
    assert tuple(FLIP_PERM[FLIP_PERM[i]] for i in range(len(STRUCTURES))) == tuple(range(len(STRUCTURES)))
    print("  permutation is an involution: ok\n")
    # The model-channel permutation must agree with the OAR one and leave the
    # derived channels alone -- bone is laterally symmetric.
    assert MODEL_FLIP_PERM[:N_STRUCT] == FLIP_PERM
    assert MODEL_FLIP_PERM[N_STRUCT:] == tuple(range(N_STRUCT, N_MODEL_STRUCT))
    print(f"  derived channels {MODEL_STRUCTURES[N_STRUCT:]} map to themselves: ok\n")


def check_bone_channel(device):
    """The derived bone channel: shape, density, flip behaviour, logit bias."""
    print("--- derived bone channel ---")
    cases = sorted(p.stem for p in __import__("pathlib").Path(CACHE).glob("0522c*.npz"))
    ds = HNCache(CACHE, cases[:2], with_bone=True)
    s = ds[0]
    assert tuple(s["labels"].shape)[0] == N_MODEL_STRUCT, s["labels"].shape
    assert s["presence"].shape[0] == N_MODEL_STRUCT
    assert float(s["presence"][N_STRUCT]) == 1.0, "bone is derived, never missing"
    frac = float(s["labels"][N_STRUCT].mean())
    print(f"  labels {tuple(s['labels'].shape)}  presence {s['presence'].numpy().astype(int)}")
    print(f"  bone fraction of volume: {frac:.4f} (CT > {BONE_HU_THRESHOLD:.0f} HU)")
    assert 0.01 < frac < 0.25, f"implausible bone fraction {frac}"

    ct = torch.stack([ds[i]["ct"] for i in range(2)]).to(device)
    lab = torch.stack([ds[i]["labels"] for i in range(2)]).to(device)
    body = torch.stack([ds[i]["body"] for i in range(2)]).to(device)
    a_ct, a_lab, a_body = augment(ct, lab, body)
    assert a_lab.shape[1] == N_MODEL_STRUCT, a_lab.shape
    print(f"  augmented labels {tuple(a_lab.shape)} body frac {float(a_body.mean()):.3f}")

    # Bone must survive a forced flip in place (it is its own partner).
    _, f_lab, _ = augment(ct, lab, body, p_flip=1.0, rot_deg=0.0,
                          scale_rng=(1.0, 1.0), shift_frac=0.0,
                          intensity_shift=0.0, intensity_scale=0.0, noise_std=0.0)
    err = (f_lab[:, N_STRUCT] - lab[:, N_STRUCT].flip(-1)).abs().max()
    print(f"  bone under L-R flip == flipped bone (max err {float(err):.3g})")
    assert float(err) < 1e-3

    cfg = VAEConfig(latent_dim=32, in_channels=1 + N_MODEL_STRUCT,
                    out_label_channels=N_MODEL_STRUCT)
    model = HNVAE(cfg).to(device)
    x = build_input(a_ct, a_lab, torch.stack([ds[i]["presence"] for i in range(2)]).to(device))
    assert x.shape[1] == 1 + N_MODEL_STRUCT, x.shape
    _, prob, _ = model.sample(2, device)
    oar_p, bone_p = float(prob[:, :N_STRUCT].mean()), float(prob[:, N_STRUCT].mean())
    print(f"  encoder input {tuple(x.shape)}  decoder out {tuple(prob.shape)}")
    print(f"  initial prob: OARs {oar_p:.5f} (~0.002)  bone {bone_p:.4f} (~0.083)")
    assert 0.05 < bone_p < 0.12, "bone logit bias is not at logit(0.08)"
    print("  bone channel: ok\n")


def main():
    device = torch.device("cuda")
    check_flip_perm()
    check_bone_channel(device)

    cases = sorted(p.stem for p in __import__("pathlib").Path(CACHE).glob("0522c*.npz"))
    train_cases, val_cases = default_split(cases, 8, 0)
    ds = HNCache(CACHE, train_cases[:4])
    print(f"--- data ---\nloaded {len(ds)} cases, meta={ds.meta.get('grid')}")
    s = ds[0]
    print(f"  ct {tuple(s['ct'].shape)} range [{s['ct'].min():.2f},{s['ct'].max():.2f}]")
    print(f"  labels {tuple(s['labels'].shape)} sum={s['labels'].sum():.0f}")
    print(f"  presence {s['presence'].numpy().astype(int)}")
    print(f"  body frac {s['body'].mean():.3f}\n")

    cfg = VAEConfig()
    model = HNVAE(cfg).to(device).to(memory_format=torch.channels_last_3d)
    c = count_parameters(model)
    print(f"--- model ---\nencoder {c['encoder']/1e6:.2f}M  decoder {c['decoder']/1e6:.2f}M  "
          f"total {c['total']/1e6:.2f}M")
    print(f"  bottleneck {cfg.bottleneck_channels}ch @ {cfg.bottleneck_size}^3 "
          f"= {cfg.bottleneck_channels * cfg.bottleneck_size**3} feats -> latent {cfg.latent_dim}")
    ch = [cfg.base_channels * m for m in cfg.channel_mult]
    res = [cfg.input_size // 2**i for i in range(len(ch))]
    print("  encoder resolutions: " + " -> ".join(f"{r}^3x{c_}" for r, c_ in zip(res, ch)))

    bs = 2
    ct = torch.stack([ds[i]["ct"] for i in range(bs)]).to(device)
    labels = torch.stack([ds[i]["labels"] for i in range(bs)]).to(device)
    presence = torch.stack([ds[i]["presence"] for i in range(bs)]).to(device)
    body = torch.stack([ds[i]["body"] for i in range(bs)]).to(device)

    a_ct, a_lab, a_body = augment(ct, labels, body)
    print(f"\n--- augment ---\n  ct {tuple(a_ct.shape)} [{a_ct.min():.2f},{a_ct.max():.2f}]  "
          f"labels [{a_lab.min():.2f},{a_lab.max():.2f}]  body frac {a_body.mean():.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    x = build_input(a_ct, a_lab, presence).to(memory_format=torch.channels_last_3d)
    print(f"  encoder input {tuple(x.shape)}")

    print("\n--- fwd/bwd (bf16 autocast) ---")
    for i in range(4):
        torch.cuda.synchronize()
        t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            loss, logs = vae_loss(out, a_ct, a_lab, presence, a_body, beta=1e-4)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t
        print(f"  step {i}: loss={float(loss):.4f} ct_l1={float(logs['ct_l1']):.4f} "
              f"dice_l={float(logs['dice_loss']):.4f} kl={float(logs['kl']):.1f} "
              f"gn={float(gn):.2f} | {dt*1000:.0f} ms")

    print(f"\n  out ct {tuple(out['ct'].shape)}  logits {tuple(out['label_logits'].shape)}  "
          f"mu {tuple(out['mu'].shape)}")
    print(f"  peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
          f"of {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    ct_s, prob_s, z = model.sample(2, device)
    print(f"\n--- sample ---\n  ct {tuple(ct_s.shape)} prob {tuple(prob_s.shape)} z {tuple(z.shape)}")
    print(f"  initial organ prob mean={float(prob_s.mean()):.5f} (should be ~0.002, "
          f"from the -6.0 logit bias)")
    print("\nOK")


if __name__ == "__main__":
    main()
