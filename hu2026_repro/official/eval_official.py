"""Evaluate a checkpoint produced by the official Fed_Unet_Seg code on the held-out target test half.

Reports, per image: OD Dice (disc incl. cup), OC Dice, vCDR, referral flip — and, for comparison, the
official repo's own numbers computed the official way:
  official_dice  = mean Dice over one-hot classes 1 and 2 (rim and cup as separate classes; empty/empty = 1)
  official_f1    = torchmetrics-style macro F1 over ALL classes incl. background (explains F1 != Dice in Table II)
Output (summary.json / preds.csv / preds/*.npy) is compatible with make_evidence.py.

  python eval_official.py --repo /path/to/fundusgan --ckpt .../G1020_fused_s42.pth.pth --test-dir work/G1020/test \
      --target G1020 --cond fused --seed 42 --out runs_official/best/G1020_fused_s42
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import case_metrics, postprocess  # noqa: E402
from fed_train import summarize  # noqa: E402

COND = {  # -> fields make_evidence.py uses to name conditions
    "pooled": {"mode": "pooled", "style": "none", "fusion": False},
    "raw": {"mode": "fedavg", "style": "none", "fusion": False},
    "gen": {"mode": "fedavg", "style": "gan", "fusion": False},
    "fused": {"mode": "fedavg", "style": "gan", "fusion": True},
}


def load_like_official(img_path, mask_path, scale, mask_values):
    """Mirror BasicDataset.preprocess: PIL resize (bicubic / nearest), /255, mask values -> class index."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    nw, nh = max(int(scale * w), 32), max(int(scale * h), 32)
    x = np.asarray(img.resize((nw, nh), Image.BICUBIC)).transpose(2, 0, 1) / 255.0
    m = np.asarray(Image.open(mask_path).resize((nw, nh), Image.NEAREST))
    if m.ndim == 3:
        m = m[..., 0]
    lab = np.zeros(m.shape, np.uint8)
    for k, v in enumerate(mask_values):
        lab[m == v] = k
    return x.astype(np.float32), lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--cond", required=True, choices=list(COND))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--which", default="best", choices=["best", "last"])
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--alpha0", type=float, default=0.3)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.repo) / "Fed_Unet_Seg"))
    from model.unet_model import UNet  # official model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    net = UNet(n_channels=3, n_classes=3, bilinear=False).to(dev)
    net.load_state_dict(st["model"])
    net.eval()

    td = Path(a.test_dir)
    ids = sorted(p.stem for p in (td / "images").glob("*.png"))
    mask_values = sorted(np.unique(np.concatenate([np.unique(np.asarray(Image.open(td / "masks" / (i + ".png")))) for i in ids])).tolist())
    if len(mask_values) != 3:
        raise SystemExit(f"expected 3 mask values (bg, disc, cup) in the test set, got {mask_values}")

    out = Path(a.out)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    rows, tp, fp, fn, off_dice = [], np.zeros(3), np.zeros(3), np.zeros(3), []
    size = None
    with torch.no_grad():
        for i in ids:
            x, gt = load_like_official(td / "images" / (i + ".png"), td / "masks" / (i + ".png"), a.scale, mask_values)
            size = gt.shape[0]
            raw = net(torch.from_numpy(x)[None].to(dev)).argmax(1)[0].cpu().numpy().astype(np.uint8)
            # official metrics on the raw prediction
            d = []
            for c in (1, 2):
                p_, g_ = raw == c, gt == c
                s_ = p_.sum() + g_.sum()
                d.append(1.0 if s_ == 0 else 2 * np.logical_and(p_, g_).sum() / s_)
            off_dice.append(np.mean(d))
            for c in range(3):
                tp[c] += np.logical_and(raw == c, gt == c).sum()
                fp[c] += np.logical_and(raw == c, gt != c).sum()
                fn[c] += np.logical_and(raw != c, gt == c).sum()
            pred = postprocess(raw)
            r = case_metrics(pred, gt)
            r.update({"official_dice": float(np.mean(d)), "dataset": a.target, "id": i})
            rows.append(r)
            np.save(out / "preds" / f"{a.target}_{i}.npy", pred)

    with open(out / "preds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = summarize(rows)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    summ["official_macro_f1_incl_background"] = float(f1.mean())
    summ["official_macro_f1_fg_only"] = float(f1[1:].mean())
    summ["checkpoint"] = {"path": a.ckpt, "which": a.which, "round": st.get("round"), "selection_score": st.get("score")}
    summ["config"] = {"target": a.target, "seed": a.seed, "size": size, "rounds": a.rounds, "local_epochs": a.local_epochs,
                      "target_split": "disjoint", "alpha0": a.alpha0, "sigma": a.sigma, "backbone": "official UNet",
                      **COND[a.cond]}
    summ["train_log"] = []
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print(f"{a.target} {a.cond} s{a.seed} [{a.which}]: OD {summ['dice_od']['mean']:.4f} OC {summ['dice_oc']['mean']:.4f} "
          f"| official Dice {summ['official_dice']['mean']:.4f} | official macro-F1 (incl. bg) {summ['official_macro_f1_incl_background']:.4f} "
          f"| vCDR MAE {summ['vcdr_abs_err']['mean']:.3f} | flip@0.6 {summ['refer_flip@0.6']['mean']:.3f} | n={len(rows)}")


if __name__ == "__main__":
    main()
