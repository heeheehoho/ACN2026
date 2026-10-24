"""Convert raw datasets to the standard format (see common.py) using the paper's preprocessing.

Raw mask formats differ by dataset and by download source. CHECK each one before running:
  refuge   : *.bmp/png, 0 = cup, 128 = disc rim, 255 = background (official REFUGE format)
  labelmap : integer label map; pass --map "bg,disc,cup" pixel values (e.g. G1020 / ORIGA PNGs: "0,1,2")
  papila   : expert contour .txt files (x y per line), <id>_disc_exp1.txt / <id>_cup_exp1.txt
  synth    : toy images for smoke tests (no raw data needed)

Examples
  python prepare_data.py --kind refuge  --name REFUGE --images raw/REFUGE/images --masks raw/REFUGE/masks --out data
  python prepare_data.py --kind labelmap --map 0,1,2 --name G1020 --images raw/G1020/Images --masks raw/G1020/Masks --out data
  python prepare_data.py --kind papila  --name PAPILA --images raw/PAPILA/FundusImages --masks raw/PAPILA/Contours --expert exp1 --out data
  python prepare_data.py --kind synth --out data_synth --size 128 --n 40
"""
import argparse
import glob
import json
import os
import numpy as np
from skimage import io, draw
from common import preprocess, synth_fundus, DATASETS


def mask_refuge(path):
    m = io.imread(path)
    if m.ndim == 3:
        m = m[..., 0]
    lab = np.zeros(m.shape, np.uint8)
    lab[m < 200] = 1          # rim (128) and cup (0) are both disc
    lab[m < 64] = 2           # cup
    return lab


def mask_labelmap(path, mapping):
    m = io.imread(path)
    if m.ndim == 3:
        m = m[..., 0]
    bg, disc, cup = mapping
    lab = np.zeros(m.shape, np.uint8)
    lab[m == disc] = 1
    lab[m == cup] = 2         # assumes cup pixels carry the cup value only (check per source)
    unknown = ~np.isin(m, [bg, disc, cup])
    if unknown.any():
        raise ValueError(f"{path}: unexpected mask values {np.unique(m[unknown])[:10]}")
    return lab


def mask_papila(contour_dir, img_id, shape, expert):
    lab = np.zeros(shape[:2], np.uint8)
    for part, val in (("disc", 1), ("cup", 2)):
        f = os.path.join(contour_dir, f"{img_id}_{part}_{expert}.txt")
        if not os.path.exists(f):
            raise FileNotFoundError(f)
        xy = np.loadtxt(f)
        rr, cc = draw.polygon(xy[:, 1], xy[:, 0], shape=shape[:2])
        lab[rr, cc] = val     # disc drawn first, cup overwrites
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["refuge", "labelmap", "papila", "synth"])
    ap.add_argument("--name")
    ap.add_argument("--images")
    ap.add_argument("--masks")
    ap.add_argument("--map", default="0,1,2")
    ap.add_argument("--expert", default="exp1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--crop", default="1000", help="int pixels (paper) or float fraction of width")
    ap.add_argument("--roi", default="paper", choices=["paper", "smooth", "blob", "gt"])
    ap.add_argument("--n", type=int, default=40, help="synth: images per domain")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.kind == "synth":
        rng = np.random.default_rng(a.seed)
        for k, name in enumerate(DATASETS):
            os.makedirs(os.path.join(a.out, name, "images"), exist_ok=True)
            os.makedirs(os.path.join(a.out, name, "masks"), exist_ok=True)
            for i in range(a.n):
                img, lab = synth_fundus(rng, a.size, style=k)
                io.imsave(os.path.join(a.out, name, "images", f"{i:04d}.png"), img, check_contrast=False)
                io.imsave(os.path.join(a.out, name, "masks", f"{i:04d}.png"), lab, check_contrast=False)
        print("synthetic data written to", a.out)
        return

    crop = float(a.crop) if "." in a.crop else int(a.crop)
    od, om = os.path.join(a.out, a.name, "images"), os.path.join(a.out, a.name, "masks")
    os.makedirs(od, exist_ok=True)
    os.makedirs(om, exist_ok=True)
    mapping = tuple(int(x) for x in a.map.split(","))
    log_path = os.path.join(a.out, a.name, "prepare_log.json")
    log = {"ok": [], "failed": [], "roi_misses_disc": [], "sources": []}
    if os.path.exists(log_path):                     # several raw folders -> one dataset (e.g. REFUGE train/val/test)
        log.update(json.load(open(log_path)))
    log.setdefault("sources", []).append({"images": a.images, "masks": a.masks, "kind": a.kind})
    imgs = sorted(p for p in sum([glob.glob(os.path.join(a.images, e)) for e in ("*.jpg", "*.JPG", "*.png", "*.bmp", "*.tif")], [])
                  if not os.path.basename(p).startswith("."))          # skip macOS '._*' metadata files
    for ip in imgs:
        img_id = os.path.splitext(os.path.basename(ip))[0]
        try:
            img = io.imread(ip)[..., :3]
            if a.kind == "papila":
                lab = mask_papila(a.masks, img_id, img.shape, a.expert)
            else:
                cands = glob.glob(os.path.join(a.masks, img_id + ".*"))
                if not cands:
                    raise FileNotFoundError("mask for " + img_id)
                lab = mask_refuge(cands[0]) if a.kind == "refuge" else mask_labelmap(cands[0], mapping)
            pimg, plab, center = preprocess(img, lab, a.size, crop, a.roi)
            # Paper's brightest-pixel ROI can miss the disc; record instead of silently keeping.
            if (lab >= 1).any() and (plab >= 1).sum() < 0.9 * (lab >= 1).sum() * (a.size / (crop if isinstance(crop, int) else crop * img.shape[1])) ** 2:
                log["roi_misses_disc"].append(img_id)
            io.imsave(os.path.join(od, img_id + ".png"), pimg, check_contrast=False)
            io.imsave(os.path.join(om, img_id + ".png"), plab, check_contrast=False)
            log["ok"].append(img_id)
        except Exception as e:  # keep going but record every failure
            log["failed"].append([img_id, repr(e)])
    log["ok"] = sorted(set(log["ok"]))
    with open(log_path, "w") as f:
        json.dump(log, f, indent=1)
    print(f"{a.name}: ok={len(log['ok'])} failed={len(log['failed'])} roi_misses_disc={len(log['roi_misses_disc'])}")


if __name__ == "__main__":
    main()
