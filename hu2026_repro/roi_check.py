"""Diagnose the paper's ROI crop (brightest pixel -> 1000x1000) on RAW images. Read-only.

For a sample of images per dataset: where the brightest pixel is vs. where the disc is, how many pixels share the
max value (ties -> np.argmax picks the first, top-left one), and how much of the disc survives the crop,
for the paper's rule and for the 'smooth' variant (blur before argmax). Saves a picture of a few cases.

  python hu2026_repro/roi_check.py --raw raw --n 40 --out data/_roi_check.png
"""
import argparse
import glob
import os
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_roi_center  # noqa: E402
from prepare_data import mask_labelmap, mask_papila  # noqa: E402

SETS = {  # name: (image glob, mask resolver)
    "G1020": ("raw/G1020/Images/*.jpg", lambda p: mask_labelmap(p.replace("/Images/", "/Masks/")[:-4] + ".png", (0, 1, 2))),
    "ORIGA": ("raw/ORIGA/Images/*.jpg", lambda p: mask_labelmap(p.replace("/Images/", "/Masks/")[:-4] + ".png", (0, 1, 2))),
    "REFUGE": ("raw/REFUGE/*/Images/*.jpg", lambda p: mask_labelmap(p.replace("/Images/", "/Masks/")[:-4] + ".png", (0, 1, 2))),
    "PAPILA": ("raw/PAPILA/FundusImages/*.jpg", None),
}


def kept_fraction(lab, center, size=1000):
    r, c = center
    h = size // 2
    disc = lab >= 1
    inside = disc[max(r - h, 0):r - h + size, max(c - h, 0):c - h + size].sum()
    return inside / max(disc.sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="data/_roi_check.png")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    examples = []
    modes = ("paper", "smooth", "blob")
    print("miss = <90% of disc inside the 1000x1000 crop; worst = smallest kept fraction; dist = median px to disc centre")
    print(f"{'dataset':<8}{'n':>4} " + " ".join(f"| {m:<6} miss  worst  dist" for m in modes) + " | ties>1")
    for name, (pat, mfn) in SETS.items():
        files = sorted(f for f in glob.glob(os.path.join(a.raw, pat.split("/", 1)[1])) if not os.path.basename(f).startswith("."))
        if not files:
            print(f"{name:<8} no files for {pat}")
            continue
        pick = [files[i] for i in rng.permutation(len(files))[: a.n]]
        stats = {m: [] for m in modes}
        ties = []
        for f in pick:
            img = np.asarray(Image.open(f).convert("RGB"))
            if mfn is None:
                iid = os.path.splitext(os.path.basename(f))[0]
                lab = mask_papila(os.path.join(a.raw, "PAPILA/ExpertsSegmentations/Contours"), iid, img.shape, "exp1")
            else:
                lab = mfn(f)
            g = img.astype(np.float32).mean(2)
            ties.append(int((g == g.max()).sum()))
            ys, xs = np.nonzero(lab >= 1)
            dc = (ys.mean(), xs.mean())
            for mode in modes:
                ctr = find_roi_center(img, mode)
                stats[mode].append((kept_fraction(lab, ctr), np.hypot(ctr[0] - dc[0], ctr[1] - dc[1]), ctr))
            if len([e for e in examples if e[0] == name]) < 2 and min(stats["smooth"][-1][0], stats["blob"][-1][0]) < 0.9:
                examples.append((name, f, img, lab, stats["paper"][-1][2], stats["smooth"][-1][2], stats["blob"][-1][2]))
        def summ(m):
            k = np.array([s[0] for s in stats[m]])
            d = np.array([s[1] for s in stats[m]])
            return f"| {m:<6}{(k < 0.9).mean() * 100:5.1f}% {k.min():5.2f} {np.median(d):5.0f}"
        print(f"{name:<8}{len(pick):>4} " + " ".join(summ(m) for m in modes) + f" | {np.mean(np.array(ties) > 1) * 100:5.1f}%")
    if examples:
        fig, axes = plt.subplots(1, len(examples), figsize=(3.2 * len(examples), 3.4), squeeze=False)
        for ax, (name, f, img, lab, cp, cs, cb) in zip(axes[0], examples):
            ax.imshow(img)
            ax.contour(lab >= 1, levels=[0.5], colors="lime", linewidths=0.8)
            for (r, c), col, lb in ((cp, "red", "paper"), (cs, "cyan", "smooth"), (cb, "yellow", "blob")):
                ax.add_patch(plt.Rectangle((c - 500, r - 500), 1000, 1000, fill=False, ec=col, lw=1.2, label=lb))
            ax.set_title(f"{name} {os.path.basename(f)}", fontsize=8)
            ax.axis("off")
        axes[0][0].legend(fontsize=7, loc="lower left")
        fig.suptitle("green = disc (GT) | crops: red = paper, cyan = smooth, yellow = blob  (cases where smooth or blob miss)", fontsize=9)
        fig.tight_layout()
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        fig.savefig(a.out, dpi=110)
        print("saved", a.out)


if __name__ == "__main__":
    main()
