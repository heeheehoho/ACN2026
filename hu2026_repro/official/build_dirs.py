"""Folder plumbing between the standard data format (prepare_data.py) and the official FundusGAN code.

Subcommands (run in this order):
  split    : fixed target split -> half 'adapt' (unlabeled GAN target images), half 'test'
  gan-data : Fundus_GAN dataroots  <work>/<T>/gan_data/<S>/{trainA,trainB,testA,testB}
  collect  : translated images from Fundus_GAN results -> <work>/<T>/gen/<S>/<id>.png
  fuse     : Eq. (5) fused images, ONE alpha per image (logged)  -> <work>/<T>/fused/<S>/images
  fl-config: client_config.json for Fed_Unet_Seg, one per condition (pooled | raw | gen | fused)

  python build_dirs.py split    --data data --target G1020 --work work
  python build_dirs.py gan-data --data data --target G1020 --work work
  python build_dirs.py collect  --target G1020 --work work --results <Fundus_GAN results_dir>
  python build_dirs.py fuse     --data data --target G1020 --work work --alpha0 0.3 --sigma 0.05
  python build_dirs.py fl-config --data data --target G1020 --work work
Files are hard-linked when possible (no extra disk, works on Windows NTFS), copied otherwise.
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
import numpy as np
from skimage import io, transform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATASETS, list_dataset  # noqa: E402


def place(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def sources_of(a):
    """Client datasets: --sources if given, else every dataset except the target."""
    if a.sources:
        src = a.sources.split(",")
        bad = [d for d in src if d not in DATASETS or d == a.target]
        if bad:
            raise SystemExit(f"invalid --sources {bad}")
        return src
    return [d for d in DATASETS if d != a.target]


def split_path(work, target):
    return Path(work) / target / "split.json"


def load_split(work, target):
    p = split_path(work, target)
    if not p.exists():
        raise SystemExit(f"run `split` first ({p} missing)")
    return json.loads(p.read_text())


def cmd_split(a):
    items = list_dataset(a.data, a.target)
    ids = sorted(i[1] for i in items)
    rng = np.random.default_rng(a.split_seed)
    perm = [ids[k] for k in rng.permutation(len(ids))]
    half = len(ids) // 2
    sp = {"target": a.target, "split_seed": a.split_seed, "adapt": sorted(perm[:half]), "test": sorted(perm[half:])}
    p = split_path(a.work, a.target)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sp, indent=1))
    tdir = Path(a.work) / a.target / "test"
    for i in sp["test"]:
        place(Path(a.data, a.target, "images", i + ".png"), tdir / "images" / (i + ".png"))
        place(Path(a.data, a.target, "masks", i + ".png"), tdir / "masks" / (i + ".png"))
    print(f"{a.target}: adapt={len(sp['adapt'])} test={len(sp['test'])} -> {p}")


def cmd_gan_data(a):
    sp = load_split(a.work, a.target)
    for s in sources_of(a):
        root = Path(a.work) / a.target / "gan_data" / s
        src = list_dataset(a.data, s)
        for it in src:                                  # all source images get translated
            for sub in ("trainA", "testA"):
                place(it[2], root / sub / (it[1] + ".png"))
        for i in sp["adapt"]:                           # target style refs: adapt half only, never test
            for sub in ("trainB", "testB"):
                place(Path(a.data, a.target, "images", i + ".png"), root / sub / (i + ".png"))
        print(f"{s}->{a.target}: trainA={len(src)} trainB={len(sp['adapt'])} -> {root}")


def cmd_collect(a):
    for s in sources_of(a):
        fake = Path(a.results) / f"{a.target}_{s}" / "train_latest" / "images" / "fake_B"
        files = sorted(fake.glob("*.png"))
        n_src = len(list_dataset(a.data, s)) if a.data else None
        if not files:
            raise SystemExit(f"no generated images in {fake}")
        out = Path(a.work) / a.target / "gen" / s
        for f in files:
            place(f, out / f.name)
        msg = f"{s}: {len(files)} generated" + (f" / {n_src} source images" if n_src else "")
        if n_src and len(files) != n_src:
            msg += "  <-- MISMATCH (check --num_test)"
        print(msg)


def cmd_fuse(a):
    rng = np.random.default_rng(a.seed)
    for s in sources_of(a):
        out = Path(a.work) / a.target / "fused" / s / "images"
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for it in list_dataset(a.data, s):
            g_path = Path(a.work) / a.target / "gen" / s / (it[1] + ".png")
            if not g_path.exists():
                raise SystemExit(f"missing generated image {g_path} (run collect)")
            r = io.imread(it[2])[..., :3].astype(np.float32)
            g = io.imread(g_path)[..., :3].astype(np.float32)
            if g.shape != r.shape:
                g = transform.resize(g, r.shape, order=1, preserve_range=True, anti_aliasing=True)
            alpha = float(np.clip(a.alpha0 + rng.normal(0, a.sigma), 0, 1))     # Eq. (5), one alpha per image
            f = np.clip(np.round(alpha * r + (1 - alpha) * g), 0, 255).astype(np.uint8)
            io.imsave(out / (it[1] + ".png"), f, check_contrast=False)
            rows.append({"id": it[1], "alpha": alpha})
        with open(out.parent / "alphas.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "alpha"])
            w.writeheader()
            w.writerows(rows)
        al = np.array([x["alpha"] for x in rows])
        print(f"{s}: fused {len(rows)} images, alpha mean {al.mean():.3f} sd {al.std():.3f}")


def cmd_fl_config(a):
    base = Path(a.work).resolve() / a.target
    test = {"name": a.target + "_test", "img_dir": str(base / "test" / "images"), "mask_dir": str(base / "test" / "masks")}
    sources = sources_of(a)
    data = Path(a.data).resolve()
    conds = {
        "raw": [{"name": s, "img_dir": str(data / s / "images"), "mask_dir": str(data / s / "masks")} for s in sources],
        "gen": [{"name": s, "img_dir": str(base / "gen" / s), "mask_dir": str(data / s / "masks")} for s in sources],
        "fused": [{"name": s, "img_dir": str(base / "fused" / s / "images"), "mask_dir": str(data / s / "masks")} for s in sources],
    }
    pooled = base / "pooled"                       # one client holding all sources = centralised "source only"
    for s in sources:
        for it in list_dataset(a.data, s):
            place(it[2], pooled / "images" / f"{s}__{it[1]}.png")
            place(it[3], pooled / "masks" / f"{s}__{it[1]}.png")
    conds["pooled"] = [{"name": "pooled", "img_dir": str(pooled / "images"), "mask_dir": str(pooled / "masks")}]
    cdir = base / "configs"
    cdir.mkdir(parents=True, exist_ok=True)
    for c, clients in conds.items():
        missing = [x["img_dir"] for x in clients if not Path(x["img_dir"]).is_dir()]
        if missing:
            print(f"skip {c}: missing {missing}")
            continue
        (cdir / f"{c}.json").write_text(json.dumps({"clients": clients, "server": [test]}, indent=1))
        print(f"{c}: {[x['name'] for x in clients]} -> {cdir / (c + '.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "gan-data", "collect", "fuse", "fl-config"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--target", required=True, choices=DATASETS)
    ap.add_argument("--work", default="work")
    ap.add_argument("--sources", default="", help="comma list of client datasets (default: all except target)")
    ap.add_argument("--results", help="Fundus_GAN results_dir (collect)")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--alpha0", type=float, default=0.3)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0, help="fusion noise seed")
    a = ap.parse_args()
    {"split": cmd_split, "gan-data": cmd_gan_data, "collect": cmd_collect,
     "fuse": cmd_fuse, "fl-config": cmd_fl_config}[a.cmd](a)


if __name__ == "__main__":
    main()
