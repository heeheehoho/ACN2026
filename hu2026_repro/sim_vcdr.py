"""Geometry illustration (synthetic ellipses, NOT real data): does cup Dice predict vCDR error?

For elliptical disc/cup masks with known vCDR we perturb ONLY the predicted cup with four error types
(isotropic scale, horizontal-only, vertical-only, translation), sweep the magnitude, and record
cup Dice, |vCDR error| and referral flips at vCDR >= 0.6. The disc is predicted perfectly.
The point is qualitative: equal Dice can mean very different vCDR error depending on error direction.

  python sim_vcdr.py --n 400 --out sim_vcdr.json
"""
import argparse
import json
import numpy as np
from common import dice, vcdr


def ellipse(size, cy, cx, ry, rx):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1


def lab_from(disc, cup):
    lab = np.zeros(disc.shape, np.uint8)
    lab[disc] = 1
    lab[cup & disc] = 2
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sim_vcdr.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    S = a.size
    cases = []
    for _ in range(a.n):
        rd = S * rng.uniform(0.30, 0.38)
        v = rng.uniform(0.3, 0.85)                       # GT vCDR spread over normal..glaucomatous range
        rc = rd * v
        cases.append((rd, rd * rng.uniform(0.85, 1.0), rc, rc * rng.uniform(0.75, 1.0)))

    kinds = {
        "scale": lambda ry, rx, m: (ry * (1 + m), rx * (1 + m), 0.0),
        "horizontal": lambda ry, rx, m: (ry, rx * (1 + 2 * m), 0.0),
        "vertical": lambda ry, rx, m: (ry * (1 + m), rx, 0.0),
        "shift": lambda ry, rx, m: (ry, rx, 2 * m),        # shift in units of cup radius
    }
    mags = [-0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3]
    res = []
    for kname, fn in kinds.items():
        for m in mags:
            if kname == "shift" and m < 0:
                continue
            d_l, e_l, f_l = [], [], []
            for rd, rdx, rc, rcx in cases:
                c = S / 2
                disc = ellipse(S, c, c, rd, rdx)
                gt = lab_from(disc, ellipse(S, c, c, rc, rcx))
                ry, rx, sh = fn(rc, rcx, m)
                pred = lab_from(disc, ellipse(S, c, c + sh * rcx, ry, rx))
                vg, vp = vcdr(gt), vcdr(pred)
                d_l.append(dice(pred == 2, gt == 2))
                e_l.append(abs(vp - vg))
                f_l.append(int((vp >= a.thr) != (vg >= a.thr)))
            res.append({"error": kname, "magnitude": m, "cup_dice": float(np.mean(d_l)),
                        "vcdr_mae": float(np.mean(e_l)), "flip_rate": float(np.mean(f_l))})
    print(f"{'error':<11}{'mag':>6}{'cupDice':>9}{'vCDR_MAE':>10}{'flip@'+str(a.thr):>10}")
    for r in res:
        print(f"{r['error']:<11}{r['magnitude']:>6.2f}{r['cup_dice']:>9.3f}{r['vcdr_mae']:>10.3f}{r['flip_rate']:>10.3f}")
    json.dump({"rows": res, "config": vars(a)}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
