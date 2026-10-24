"""Build the reproducibility evidence pack from YOUR runs (no numbers are invented here).

Reads every runs/**/summary.json (+ preds.csv, preds/*.npy) written by fed_train.py and produces:
  evidence/results_table.md     paper vs reproduction, mean ± SD over seeds, n runs per row
  evidence/results_runs.csv     one line per run (for the appendix)
  evidence/curves.png           source-validation Dice per FL round (FedAvg runs)
  evidence/vcdr_scatter.png     predicted vs ground-truth vCDR per condition, 0.6 referral lines
  evidence/overlays.png         GT (green) vs prediction (red) contours: best / median / worst cup Dice
  evidence/manifest.json        file list with SHA-256, run configs, timestamp

  python make_evidence.py --runs runs --data data --target G1020 --out evidence
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import time
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import io, measure
from common import load_pair

# Reported values from Hu et al. (IEEE IoT-J 2026). Table II = G1020 target (OD, OC Dice); Table III = mean Dice.
PAPER = {
    "G1020": {"source-only": (0.7439, 0.5986), "fedavg+gen": (0.8022, 0.6225), "fedavg+gen+fusion": (0.7941, 0.6271)},
    "ORIGA": {"source-only": 0.7830, "fedavg+gen": 0.8369, "fedavg+gen+fusion": 0.8304},
    "PAPILA": {"source-only": 0.7010, "fedavg+gen": 0.7279, "fedavg+gen+fusion": 0.7209},
    "REFUGE": {"source-only": 0.7946, "fedavg+gen": 0.8652, "fedavg+gen+fusion": 0.8525},
}
ORDER = ["source-only", "local", "fedavg", "fedavg+gen", "fedavg+gen+fusion"]


def condition(cfg):
    if cfg["mode"] == "pooled":
        return "source-only"
    if cfg["mode"] == "local":
        return "local"
    if cfg["style"] == "none":
        return "fedavg"
    return "fedavg+gen+fusion" if cfg["fusion"] else "fedavg+gen"


def gen_label(cfg):
    s = cfg["style"]
    return "-" if s == "none" else ("FDA" if s == "fda" else "GAN")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fmt(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return "—"
    return f"{np.mean(v):.4f}" + (f" ± {np.std(v, ddof=1):.4f}" if len(v) > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--target", default="G1020")
    ap.add_argument("--out", default="evidence")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = []
    for sp in sorted(glob.glob(os.path.join(a.runs, "**", "summary.json"), recursive=True)):
        s = json.load(open(sp))
        cfg = s["config"]
        if cfg["target"] != a.target:
            continue
        runs.append({"dir": os.path.dirname(sp), "cfg": cfg, "s": s, "mtime": os.path.getmtime(sp),
                     "cond": condition(cfg), "gen": gen_label(cfg)})
    if not runs:
        raise SystemExit(f"no runs for target {a.target} under {a.runs}")
    # Runs with different training settings must never be averaged together; a repeated seed keeps the newest run.
    def setting(r):
        c = r["cfg"]
        return (r["cond"], r["gen"], c["size"], c["rounds"], c["local_epochs"], c["target_split"], c["alpha0"], c["sigma"])
    latest, dropped = {}, []
    for r in sorted(runs, key=lambda r: r["mtime"]):
        k = setting(r) + (r["cfg"]["seed"],)
        if k in latest:
            dropped.append(latest[k]["dir"])
        latest[k] = r
    runs = list(latest.values())
    for d in dropped:
        print("WARNING: duplicate seed, older run ignored:", d)

    # ---- per-run CSV
    keys = ["dice_od", "dice_oc", "iou_od", "iou_oc", "vcdr_abs_err", "refer_flip@0.6"]
    with open(os.path.join(a.out, "results_runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_dir", "condition", "generator", "seed", "size", "target_split", "n_test"] + keys)
        for r in runs:
            c = r["cfg"]
            w.writerow([r["dir"], r["cond"], r["gen"], c["seed"], c["size"], c["target_split"], r["s"]["n"]]
                       + [r["s"][k]["mean"] for k in keys])

    # ---- aggregated table vs paper
    groups = defaultdict(list)
    for r in runs:
        groups[setting(r)[:2]].append(r)
    mixed = {k: {setting(r)[2:] for r in rs} for k, rs in groups.items()}
    mixed = {k: v for k, v in mixed.items() if len(v) > 1}
    if mixed:
        raise SystemExit(f"runs with different settings in the same condition: {mixed}. Use --runs on one setting only.")
    lines = [f"# Reproduction results — target {a.target}", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} from {len(runs)} runs. Mean ± SD over seeds.", ""]
    paper_t = PAPER.get(a.target, {})
    if a.target == "G1020":
        lines += ["| Condition | Gen | Seeds | OD Dice (ours) | OD Dice (paper) | OC Dice (ours) | OC Dice (paper) | Mean Dice (ours) | Mean (paper) | vCDR MAE | Referral flip @0.6 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
    else:
        lines += ["| Condition | Gen | Seeds | Mean Dice (ours) | Mean Dice (paper) | vCDR MAE | Referral flip @0.6 |",
                  "|---|---|---|---|---|---|---|"]
    for cond in ORDER:
        for (c, g), rs in sorted(groups.items()):
            if c != cond:
                continue
            od = [r["s"]["dice_od"]["mean"] for r in rs]
            oc = [r["s"]["dice_oc"]["mean"] for r in rs]
            mean = [(x + y) / 2 for x, y in zip(od, oc)]
            ve = [r["s"]["vcdr_abs_err"]["mean"] for r in rs]
            fl = [r["s"]["refer_flip@0.6"]["mean"] for r in rs]
            seeds = ",".join(str(r["cfg"]["seed"]) for r in rs)
            p = paper_t.get(cond)
            if a.target == "G1020":
                pod, poc = (f"{p[0]:.4f}", f"{p[1]:.4f}") if p else ("not reported", "not reported")
                pm = f"{(p[0] + p[1]) / 2:.4f}" if p else "not reported"
                lines.append(f"| {cond} | {g} | {seeds} | {fmt(od)} | {pod} | {fmt(oc)} | {poc} | {fmt(mean)} | {pm} | {fmt(ve)} | {fmt(fl)} |")
            else:
                pm = f"{p:.4f}" if p else "not reported"
                lines.append(f"| {cond} | {g} | {seeds} | {fmt(mean)} | {pm} | {fmt(ve)} | {fmt(fl)} |")
    if dropped:
        lines += ["", f"Ignored {len(dropped)} older duplicate-seed run(s): " + ", ".join(dropped)]
    lines += ["", "Settings (from run configs): "
              f"size={sorted({r['cfg']['size'] for r in runs})}, rounds×local epochs={sorted({str(r['cfg']['rounds']) + '×' + str(r['cfg']['local_epochs']) for r in runs})}, "
              f"target split={sorted({r['cfg']['target_split'] for r in runs})}, alpha0={sorted({r['cfg']['alpha0'] for r in runs})}, sigma={sorted({r['cfg']['sigma'] for r in runs})}.",
              "Paper uses U-Mamba and its own GAN; differences must be listed in the report."]
    open(os.path.join(a.out, "results_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- FL curves
    fed = [r for r in runs if r["cfg"]["mode"] == "fedavg" and r["s"].get("train_log")]
    if fed:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        for r in fed:
            lg = r["s"]["train_log"]
            lab = f"{r['cond']} ({r['gen']}) s{r['cfg']['seed']}"
            ax[0].plot([x["round"] for x in lg], [x["src_val_dice_od"] for x in lg], label=lab)
            ax[1].plot([x["round"] for x in lg], [x["src_val_dice_oc"] for x in lg], label=lab)
        for k, t in enumerate(["Optic disc", "Optic cup"]):
            ax[k].set_title(f"{t} — source-validation Dice per round")
            ax[k].set_xlabel("FL round")
            ax[k].grid(alpha=0.3)
        ax[0].set_ylabel("Dice")
        ax[1].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "curves.png"), dpi=150)
        plt.close(fig)

    # ---- vCDR scatter (first seed of each condition)
    firsts = {}
    for r in runs:
        firsts.setdefault((r["cond"], r["gen"]), r)
    fig, axes = plt.subplots(1, len(firsts), figsize=(3.3 * len(firsts), 3.3), squeeze=False)
    for ax, ((c, g), r) in zip(axes[0], sorted(firsts.items(), key=lambda kv: ORDER.index(kv[0][0]))):
        rows = list(csv.DictReader(open(os.path.join(r["dir"], "preds.csv"))))
        vg = np.array([float(x["vcdr_gt"]) for x in rows])
        vp = np.array([float(x["vcdr_pred"]) for x in rows])
        ok = ~(np.isnan(vg) | np.isnan(vp))
        flip = ok & ((vg >= 0.6) != (vp >= 0.6))
        ax.scatter(vg[ok & ~flip], vp[ok & ~flip], s=6, alpha=0.5, label="same decision")
        ax.scatter(vg[flip], vp[flip], s=8, color="#d55e00", label="referral flipped")
        ax.plot([0, 1], [0, 1], color="0.5", lw=0.8)
        ax.axvline(0.6, color="0.6", ls="--", lw=0.8)
        ax.axhline(0.6, color="0.6", ls="--", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{c} ({g}) s{r['cfg']['seed']}\nflipped {int(flip.sum())}/{int(ok.sum())}", fontsize=9)
        ax.set_xlabel("GT vCDR")
    axes[0][0].set_ylabel("Predicted vCDR")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "vcdr_scatter.png"), dpi=150)
    plt.close(fig)

    # ---- contour overlays from a run that saved predictions
    with_preds = [r for r in runs if os.path.isdir(os.path.join(r["dir"], "preds"))]
    if with_preds:
        r = with_preds[-1]
        rows = [x for x in csv.DictReader(open(os.path.join(r["dir"], "preds.csv"))) if x["dice_oc"] not in ("", "nan")]
        rows.sort(key=lambda x: float(x["dice_oc"]))
        pick = [("worst", rows[0]), ("median", rows[len(rows) // 2]), ("best", rows[-1])]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
        for ax, (tag, x) in zip(axes, pick):
            item = (x["dataset"], x["id"], os.path.join(a.data, x["dataset"], "images", x["id"] + ".png"),
                    os.path.join(a.data, x["dataset"], "masks", x["id"] + ".png"))
            img, gt = load_pair(item, r["cfg"]["size"])
            pred = np.load(os.path.join(r["dir"], "preds", f"{x['dataset']}_{x['id']}.npy"))
            ys, xs = np.where(gt >= 1)
            if len(ys):
                pad = 40
                y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, gt.shape[0])
                x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, gt.shape[1])
            else:
                y0, y1, x0, x1 = 0, gt.shape[0], 0, gt.shape[1]
            ax.imshow(img[y0:y1, x0:x1])
            for m, col in ((gt, "#009e73"), (pred, "#d55e00")):
                for lvl in (1, 2):
                    padded = np.pad((m[y0:y1, x0:x1] >= lvl).astype(float), 1)   # closes contours touching the edge
                    for cnt in measure.find_contours(padded, 0.5):
                        ax.plot(cnt[:, 1] - 1, cnt[:, 0] - 1, color=col, lw=1.2)
            ax.set_title(f"{tag}: cup Dice {float(x['dice_oc']):.3f}\nvCDR GT {float(x['vcdr_gt']):.2f} / pred {float(x['vcdr_pred']):.2f}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{r['cond']} ({r['gen']}) seed {r['cfg']['seed']} — green = ground truth, red = prediction", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "overlays.png"), dpi=150)
        plt.close(fig)

    # ---- manifest with hashes (proof the figures come from these files)
    files = {}
    for r in runs:
        for fn in ("summary.json", "preds.csv"):
            p = os.path.join(r["dir"], fn)
            if os.path.exists(p):
                files[p] = sha256(p)
    for fn in sorted(os.listdir(a.out)):
        if fn != "manifest.json":
            files[os.path.join(a.out, fn)] = sha256(os.path.join(a.out, fn))
    json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "target": a.target,
               "runs": [{"dir": r["dir"], "config": r["cfg"]} for r in runs], "sha256": files},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"evidence written to {a.out}/")


if __name__ == "__main__":
    main()
