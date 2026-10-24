"""One driver for the whole reproduction with the OFFICIAL code (GAN -> fusion -> FedAvg U-Net -> evaluation).

Stages (each skips work that is already done, so it can be re-run after a crash):
  prep : split target, build GAN dataroots
  gan  : train one generator per source->target (Fundus_GAN/train.py --model ppfg), translate all source images
  fuse : collect translations, build fused datasets (Eq. 5), write client configs
  fl   : Fed_Unet_Seg/main.py for each condition x seed (paper: 20 rounds x 5 local epochs)
  eval : eval_official.py on the BEST (test-selected, as in the official code) and LAST round checkpoints

  python run_official.py --repo ../fundusgan --data data --target G1020 --stage all --dry-run   # print commands only
  python run_official.py --repo ../fundusgan --data data --target G1020 --stage all
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
CONDS = ["pooled", "raw", "gen", "fused"]


def run(cmd, cwd=None, env=None, dry=False, log=None):
    line = (f"(cd {cwd} && " if cwd else "") + " ".join(shlex.quote(str(c)) for c in cmd) + (")" if cwd else "")
    print("$", line, flush=True)
    if dry:
        return
    e = dict(os.environ, **(env or {}))
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as fh:
            fh.write("$ " + line + "\n")
            p = subprocess.run([str(c) for c in cmd], cwd=cwd, env=e, stdout=fh, stderr=subprocess.STDOUT)
    else:
        p = subprocess.run([str(c) for c in cmd], cwd=cwd, env=e)
    if p.returncode != 0:
        raise SystemExit(f"FAILED ({p.returncode}): {line}" + (f"\n  see log: {log}" if log else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="clone of github.com/JRHU0048/FundusGAN (patched)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--target", default="G1020")
    ap.add_argument("--work", default="work")
    ap.add_argument("--stage", default="all", choices=["prep", "gan", "fuse", "fl", "eval", "all"])
    ap.add_argument("--sources", default="", help="client datasets, e.g. ORIGA,PAPILA (default: all except target)")
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--gan-epochs", default="50,50", help="n_epochs,n_epochs_decay (paper 50,50; repo default 10,10)")
    ap.add_argument("--gan-size", type=int, default=512)
    ap.add_argument("--gan-extra", default="", help="extra Fundus_GAN args, e.g. '--gpu_ids -1'")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--scale", type=float, default=0.5, help="official default image scale for the U-Net")
    ap.add_argument("--fl-extra", default="", help="extra Fed_Unet_Seg/main.py args")
    ap.add_argument("--out", default="runs_official")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    repo = Path(a.repo).resolve()
    work = Path(a.work).resolve()
    data = Path(a.data).resolve()
    T = a.target
    sources = a.sources.split(",") if a.sources else [d for d in ["G1020", "ORIGA", "REFUGE", "PAPILA"] if d != T]
    stages = ["prep", "gan", "fuse", "fl", "eval"] if a.stage == "all" else [a.stage]
    conds = a.conds.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    logs = Path(a.out).resolve() / "logs"
    bd = [PY, HERE / "build_dirs.py"]
    common = ["--data", data, "--target", T, "--work", work, "--sources", ",".join(sources)]

    if "prep" in stages:
        run(bd + ["split"] + common, dry=a.dry_run)
        run(bd + ["gan-data"] + common, dry=a.dry_run)

    gan_dir = repo / "Fundus_GAN"
    ckpt_dir, res_dir = work / T / "gan_checkpoints", work / T / "gan_results"
    if "gan" in stages:
        ne, nd = a.gan_epochs.split(",")
        for s in sources:
            name = f"{T}_{s}"
            dr = work / T / "gan_data" / s
            opts = ["--dataroot", dr, "--name", name, "--model", "ppfg", "--CUT_mode", "cut",
                    "--checkpoints_dir", ckpt_dir, "--load_size", a.gan_size, "--crop_size", a.gan_size] + shlex.split(a.gan_extra)
            if not (ckpt_dir / name / "latest_net_G.pth").exists():
                # official train.py saves 'latest' only every --save_epoch_freq (5) epochs -> force a save at the last epoch
                run([PY, "train.py"] + opts + ["--n_epochs", ne, "--n_epochs_decay", nd, "--display_id", "0", "--no_html",
                                               "--save_epoch_freq", int(ne) + int(nd)],
                    cwd=gan_dir, dry=a.dry_run, log=logs / f"gan_train_{name}.log")
            if not (res_dir / name / "train_latest" / "images" / "fake_B").is_dir():
                run([PY, "test.py"] + opts + ["--phase", "train", "--num_test", "1000000", "--results_dir", res_dir],
                    cwd=gan_dir, dry=a.dry_run, log=logs / f"gan_test_{name}.log")

    if "fuse" in stages:
        run(bd + ["collect", "--results", res_dir] + common, dry=a.dry_run)
        run(bd + ["fuse"] + common, dry=a.dry_run)
        run(bd + ["fl-config"] + common, dry=a.dry_run)

    seg_dir = repo / "Fed_Unet_Seg"
    if "fl" in stages:
        (Path(a.out).resolve() / "fl_logs").mkdir(parents=True, exist_ok=True)   # official logger does not create it
        for c in conds:
            for s in seeds:
                tag = f"{T}_{c}_s{s}"
                if (seg_dir / "checkpoints" / f"{tag}_last.pth").exists() and not a.dry_run:
                    rounds_done = __import__("torch").load(seg_dir / "checkpoints" / f"{tag}_last.pth", map_location="cpu")["round"]
                    if rounds_done >= a.rounds:
                        print(f"skip {tag}: done ({rounds_done} rounds)")
                        continue
                run([PY, "main.py", "--client-config", work / T / "configs" / f"{c}.json",
                     "--global-epochs", a.rounds, "--local-epochs", a.local_epochs, "--scale", a.scale,
                     "--log_file", Path(a.out).resolve() / "fl_logs" / f"{tag}.xlsx"] + shlex.split(a.fl_extra),
                    cwd=seg_dir, env={"SEED": str(s), "CKPT_NAME": tag}, dry=a.dry_run, log=logs / f"fl_{tag}.log")

    if "eval" in stages:
        for c in conds:
            for s in seeds:
                tag = f"{T}_{c}_s{s}"
                for which, fn in (("best", f"{tag}.pth"), ("last", f"{tag}_last.pth")):
                    ck = seg_dir / "checkpoints" / fn
                    if not ck.exists() and not a.dry_run:
                        print(f"missing {ck}, skipped")
                        continue
                    run([PY, HERE / "eval_official.py", "--repo", repo, "--ckpt", ck, "--test-dir", work / T / "test",
                         "--target", T, "--cond", c, "--seed", s, "--which", which, "--scale", a.scale,
                         "--rounds", a.rounds, "--local-epochs", a.local_epochs,
                         "--out", Path(a.out).resolve() / which / tag], dry=a.dry_run)
        print(f"\nnext: python {HERE.parent / 'make_evidence.py'} --runs {Path(a.out) / 'last'} --data {a.data} --target {T} --out evidence_last"
              f"\n      (repeat with {Path(a.out) / 'best'} to show the effect of test-set model selection)")


if __name__ == "__main__":
    main()
