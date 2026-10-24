"""Minimal, logged patches so the official code (github.com/JRHU0048/FundusGAN @ 71e9354) runs as the paper describes.

Only fixes that stop the code from running or from being reproducible are applied; the method itself
(losses, MGDA, FedAvg, metrics, model selection) is left untouched so that we reproduce THEIR pipeline.

  python patch_official.py /path/to/fundusgan          # idempotent; prints every change
"""
import sys
from pathlib import Path

PATCHES = [
    # (file, old, new, why)
    ("Fundus_GAN/models/ppfg_model.py",
     "parser.add_argument('--CUT_model', type=str,",
     "parser.add_argument('--CUT_mode', type=str,",
     "option is declared as --CUT_model but read as opt.CUT_mode -> AttributeError"),
    ("Fundus_GAN/models/ppfg_model.py",
     None,  # append
     "\n\n# [repro patch] `--model ppfg` looks for a class named PpfgModel; the class is called CUTModel.\nPpfgModel = CUTModel\n",
     "default --model cut needs models/cut_model.py, which is not in the repo; alias lets `--model ppfg` load"),
    ("Fed_Unet_Seg/train.py",
     "random_seed = 42\n",
     "random_seed = int(os.environ.get('SEED', 42))  # [repro patch] seeds 42/123/2026 from the paper\n",
     "seed is hard-coded; the paper reports 3 seeds"),
    ("Fed_Unet_Seg/train.py",
     "checkpoint_name = 'best_model.pth'   # The name of the saved model",
     "checkpoint_name = os.environ.get('CKPT_NAME', 'best_model')  # [repro patch] one name per run",
     "every run overwrites checkpoints/best_model.pth.pth"),
    ("Fed_Unet_Seg/train.py",
     "            if self.logger:\n                self._log_metrics(global_epoch, scores, server_metrics)\n",
     "            if self.logger:\n                self._log_metrics(global_epoch, scores, server_metrics)\n"
     "                self.logger.save_to_excel()  # [repro patch] keep the per-round log even if a run dies\n"
     "            # [repro patch] also keep the LAST-round model: the official code keeps only the round with the\n"
     "            # best Dice on the target (= test) set, i.e. model selection on the test data.\n"
     "            Path('checkpoints').mkdir(exist_ok=True)\n"
     "            torch.save({'model': self.global_model.state_dict(), 'round': global_epoch},\n"
     "                       f'checkpoints/{checkpoint_name}_last.pth')\n",
     "only the test-selected best round is saved; last round is needed for an unbiased number"),
    ("Fed_Unet_Seg/train.py",
     "            state_dict = model.state_dict()\n            del model, optimizer, grad_scaler",
     "            # [repro patch] move the 2 GB client weights off the GPU; aggregation already runs on CPU\n"
     "            state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}\n"
     "            del model, optimizer, grad_scaler",
     "client state_dicts stay on the GPU until aggregation (3 x 2 GB) -> OOM risk on a 16 GB card; numerically identical"),
]


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    for rel, old, new, why in PATCHES:
        p = root / rel
        s = p.read_text(encoding="utf-8")
        if new.strip() in s:
            print(f"[already] {rel}: {why}")
            continue
        if old is None:
            s = s + new
        else:
            if old not in s:
                raise SystemExit(f"[FAILED] {rel}: anchor not found ({why}). Repo version differs from 71e9354?")
            s = s.replace(old, new, 1)
        p.write_text(s, encoding="utf-8")
        print(f"[patched] {rel}: {why}")


if __name__ == "__main__":
    main()
