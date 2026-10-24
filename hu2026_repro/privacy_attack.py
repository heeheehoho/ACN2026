"""Empirical test of the paper's privacy claim for fusion I_f = a*I_r + (1-a)*I_g, a = clip(0.3+N(0,0.05^2)).

The paper claims fusion "prevents exact recovery of the original source images" but reports no attack.
Two attacks, on one client's images:

A. Inversion (attacker holds I_f and I_g, e.g. leaked generator outputs):
   I_r = I_g + (I_f - I_g)/a. The only unknown is ONE scalar per image.
   Attacker plugs in the public base weight a0 = 0.3. Residual = (I_r - I_g)(a/a0 - 1): on average ~13% of |I_r - I_g| for sigma=0.05.
B. Re-identification (attacker holds only I_f, plus a gallery of real images, e.g. the hospital PACS):
   match by normalised cross-correlation of vessel maps (green-channel black top-hat). Top-1 accuracy.
   Controls: query = I_g (translation only), query = real image of the same eye with flip (sanity).

  python privacy_attack.py --data data --site REFUGE --style fda --style-site G1020 --n 200 --out runs/privacy_refuge
"""
import argparse
import json
import os
import numpy as np
from skimage import io, morphology, transform
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from common import list_dataset, load_pair
from fed_train import fda


def vessel_map(img):
    g = img[..., 1].astype(np.float32)
    v = morphology.black_tophat(g, morphology.disk(max(3, img.shape[0] // 64)))
    v = transform.resize(v, (128, 128), anti_aliasing=True)
    v = (v - v.mean()) / (v.std() + 1e-6)
    return v.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--site", required=True, help="client whose images are attacked")
    ap.add_argument("--style", default="fda", help="fda | <dir of generated images>/<site>/<id>.png")
    ap.add_argument("--style-site", help="target dataset providing FDA style references")
    ap.add_argument("--alpha0", type=float, default=0.3)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    items = list_dataset(a.data, a.site)
    items = [items[i] for i in rng.permutation(len(items))[: a.n]]
    refs = list_dataset(a.data, a.style_site) if a.style == "fda" else None

    reals, fused, gens, rows = [], [], [], []
    for it in items:
        r = load_pair(it, a.size)[0].astype(np.float32)
        if a.style == "fda":
            g = fda(r.astype(np.uint8), load_pair(refs[rng.integers(len(refs))], a.size)[0]).astype(np.float32)
        else:
            g = io.imread(os.path.join(a.style, it[0], it[1] + ".png"))[..., :3].astype(np.float32)
        al = float(np.clip(a.alpha0 + rng.normal(0, a.sigma), 0, 1))
        f = al * r + (1 - al) * g
        f = np.clip(np.round(f), 0, 255)                    # fused images are stored as uint8
        rec = np.clip(g + (f - g) / a.alpha0, 0, 255)      # inversion with the public base weight
        row = {"id": it[1], "alpha_true": al}
        for tag, x in (("fused", f), ("gen", g), ("inverted", rec)):
            row[f"psnr_{tag}"] = float(psnr(r, x, data_range=255))
            row[f"ssim_{tag}"] = float(ssim(r, x, data_range=255, channel_axis=2))
        rows.append(row)
        reals.append(vessel_map(r.astype(np.uint8)))
        fused.append(vessel_map(f.astype(np.uint8)))
        gens.append(vessel_map(g.astype(np.uint8)))

    R = np.stack(reals)
    def top1(Q):
        S = Q @ R.T / R.shape[1]
        return float((S.argmax(1) == np.arange(len(Q))).mean())
    summ = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "id"}
    summ.update({"n": len(rows), "reid_top1_fused": top1(np.stack(fused)), "reid_top1_gen": top1(np.stack(gens)),
                 "reid_chance": 1 / len(rows)})
    json.dump({"summary": summ, "rows": rows, "config": vars(a)}, open(os.path.join(a.out, "privacy.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
