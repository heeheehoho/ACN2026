"""Leave-one-domain-out OD/OC segmentation: pooled / isolated-local / FedAvg, optional style transfer + fusion.

Replicates the protocol of Hu et al. (IEEE IoT-J 2026), Sec. IV-B:
  one dataset = target (test), other three = FL clients; each client split 80/20 train/val;
  FedAvg weighted by client size, 20 rounds x 5 local epochs; seeds 42/123/2026;
  fusion I_f = a*I_r + (1-a)*I_g, a = clip(0.3 + N(0, 0.05^2), 0, 1)   (Eq. 5).
Backbone: plain U-Net by default (U-Mamba needs mamba-ssm CUDA kernels; see README).

Every per-image result is saved (preds.csv); a checkpoint is written each round and resumed automatically.

  python fed_train.py --data data --target G1020 --mode fedavg --style fda --fusion --seed 42 --out runs/g1020_fda_fus_s42
  python fed_train.py --data data --target G1020 --mode fedavg --dry-run      # pre-check: counts, 1 step, time/epoch
"""
import argparse
import csv
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
from common import DATASETS, list_dataset, load_pair, case_metrics, postprocess


# ------------------------------------------------------------------ model
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, ch=(32, 64, 128, 256, 320), n_cls=3):
        super().__init__()
        self.enc = nn.ModuleList()
        prev = 3
        for c in ch:
            self.enc.append(block(prev, c))
            prev = c
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(len(ch) - 1, 0, -1)])
        self.dec = nn.ModuleList([block(ch[i - 1] * 2, ch[i - 1]) for i in range(len(ch) - 1, 0, -1)])
        self.head = nn.Conv2d(ch[0], n_cls, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.head(x)


def seg_loss(logits, lab):
    ce = F.cross_entropy(logits, lab)
    p = logits.softmax(1)
    pd_, pc = p[:, 1] + p[:, 2], p[:, 2]
    gd, gc = (lab >= 1).float(), (lab == 2).float()
    def sd(a, b):
        return 1 - (2 * (a * b).sum((1, 2)) + 1) / (a.sum((1, 2)) + b.sum((1, 2)) + 1)
    return ce + 0.5 * (sd(pd_, gd).mean() + sd(pc, gc).mean())


# ------------------------------------------------------------------ style transfer stand-ins
def fda(src, tgt, beta=0.01):
    """Fourier Domain Adaptation (Yang & Soatto, CVPR 2020): swap low-freq amplitude. uint8 HxWx3 -> uint8."""
    s = src.astype(np.float32)
    t = tgt.astype(np.float32)
    fs, ft = np.fft.fft2(s, axes=(0, 1)), np.fft.fft2(t, axes=(0, 1))
    amp_s, pha_s = np.abs(fs), np.angle(fs)
    amp_t = np.abs(ft)
    amp_s, amp_t = np.fft.fftshift(amp_s, axes=(0, 1)), np.fft.fftshift(amp_t, axes=(0, 1))
    h, w = s.shape[:2]
    b = max(1, int(np.floor(min(h, w) * beta)))
    ch, cw = h // 2, w // 2
    amp_s[ch - b:ch + b + 1, cw - b:cw + b + 1] = amp_t[ch - b:ch + b + 1, cw - b:cw + b + 1]
    amp_s = np.fft.ifftshift(amp_s, axes=(0, 1))
    out = np.real(np.fft.ifft2(amp_s * np.exp(1j * pha_s), axes=(0, 1)))
    return np.clip(out, 0, 255).astype(np.uint8)


class SegSet(torch.utils.data.Dataset):
    """style: 'none' | 'fda' | path to pre-generated images (<dir>/<dataset>/<id>.png, e.g. FundusGAN output)."""
    def __init__(self, items, size, train, style="none", style_refs=None, fusion=False,
                 alpha0=0.3, sigma=0.05, fda_beta=0.01, seed=0):
        self.items, self.size, self.train = items, size, train
        self.style, self.refs, self.fusion = style, style_refs or [], fusion
        self.alpha0, self.sigma, self.beta = alpha0, sigma, fda_beta
        self.rng = np.random.default_rng(seed)
        self._ref_cache = {}

    def __len__(self):
        return len(self.items)

    def _ref(self):
        k = int(self.rng.integers(len(self.refs)))
        if k not in self._ref_cache:
            self._ref_cache[k] = load_pair(self.refs[k], self.size)[0]
        return self._ref_cache[k]

    def generated(self, idx, img):
        if self.style == "fda":
            return fda(img, self._ref(), self.beta)
        name, iid = self.items[idx][0], self.items[idx][1]
        g = io.imread(os.path.join(self.style, name, iid + ".png"))[..., :3]
        if g.shape[0] != self.size:
            from skimage import transform
            g = transform.resize(g, (self.size, self.size), order=1, preserve_range=True).astype(np.uint8)
        return g

    def __getitem__(self, idx):
        img, lab = load_pair(self.items[idx], self.size)
        if self.train and self.style != "none":
            g = self.generated(idx, img)
            if self.fusion:                                     # Eq. (5)
                a = float(np.clip(self.alpha0 + self.rng.normal(0, self.sigma), 0, 1))
                img = np.clip(a * img.astype(np.float32) + (1 - a) * g.astype(np.float32), 0, 255).astype(np.uint8)
            else:
                img = g
        if self.train and self.rng.random() < 0.5:
            img, lab = img[:, ::-1], lab[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return x, torch.from_numpy(np.ascontiguousarray(lab)).long(), idx


# ------------------------------------------------------------------ train / eval
def run_epochs(model, loader, epochs, lr, dev, scaler):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    tot, n = 0.0, 0
    for _ in range(epochs):
        for x, y, _ in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                loss = seg_loss(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot, n = tot + loss.item() * len(x), n + len(x)
    return tot / max(n, 1)


@torch.no_grad()
def evaluate(model, items, size, dev, bs=8, save_dir=None):
    model.eval()
    ds = SegSet(items, size, train=False)
    rows = []
    for x, y, idx in torch.utils.data.DataLoader(ds, batch_size=bs):
        pred = model(x.to(dev)).argmax(1).cpu().numpy().astype(np.uint8)
        for p, g, i in zip(pred, y.numpy().astype(np.uint8), idx.numpy()):
            p = postprocess(p)
            r = case_metrics(p, g)
            r.update({"dataset": items[i][0], "id": items[i][1]})
            rows.append(r)
            if save_dir:
                np.save(os.path.join(save_dir, f"{items[i][0]}_{items[i][1]}.npy"), p)
    return rows


def summarize(rows):
    out = {"n": len(rows)}
    for k in rows[0]:
        if k in ("dataset", "id"):
            continue
        v = np.array([r[k] for r in rows], dtype=float)
        ok = ~np.isnan(v)
        out[k] = {"mean": float(v[ok].mean()) if ok.any() else None,
                  "sd": float(v[ok].std(ddof=1)) if ok.sum() > 1 else None, "n_valid": int(ok.sum())}
    return out


def fedavg(states, weights):
    tot = float(sum(weights))
    avg = {}
    for k in states[0]:
        if states[0][k].dtype in (torch.int64, torch.long):      # BN num_batches_tracked
            avg[k] = states[0][k].clone()
        else:
            avg[k] = sum(s[k].float() * (w / tot) for s, w in zip(states, weights)).to(states[0][k].dtype)
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True, choices=DATASETS)
    ap.add_argument("--mode", default="fedavg", choices=["pooled", "local", "fedavg"])
    ap.add_argument("--style", default="none", help="none | fda | <dir of generated images>")
    ap.add_argument("--fusion", action="store_true")
    ap.add_argument("--alpha0", type=float, default=0.3)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--target-split", default="disjoint", choices=["disjoint", "paper"],
                    help="disjoint: half of target = unlabeled style refs, other half = test. paper: all for both (transductive)")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--width", type=int, default=32, help="U-Net base channels")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-preds", action="store_true", help="save predicted label maps (.npy) for vCDR / report work")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="load <out>/ckpt.pt and re-evaluate on the same test split (for a live demo); writes eval_only.json")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)

    sources = [d for d in DATASETS if d != a.target]
    clients = {}
    for d in sources:
        it = list_dataset(a.data, d)
        if not it:
            raise SystemExit(f"no data for {d} under {a.data}")
        perm = rng.permutation(len(it))
        cut = int(0.8 * len(it))
        clients[d] = ([it[i] for i in perm[:cut]], [it[i] for i in perm[cut:]])
    tgt = list_dataset(a.data, a.target)
    perm = rng.permutation(len(tgt))
    if a.target_split == "disjoint":
        refs, test = [tgt[i] for i in perm[: len(tgt) // 2]], [tgt[i] for i in perm[len(tgt) // 2:]]
    else:
        refs, test = tgt, tgt
    print(f"device={dev} target={a.target} test={len(test)} style_refs={len(refs)} "
          + " ".join(f"{d}:train={len(tr)}/val={len(va)}" for d, (tr, va) in clients.items()))

    def loader(items, seed):
        ds = SegSet(items, a.size, True, a.style, refs, a.fusion, a.alpha0, a.sigma, seed=seed)
        def reseed(wid):  # distinct fusion/FDA randomness per worker
            info = torch.utils.data.get_worker_info()
            info.dataset.rng = np.random.default_rng(seed * 100 + wid)
        return torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                                           drop_last=len(items) > a.bs, worker_init_fn=reseed)

    width = a.width
    ch = (width, width * 2, width * 4, width * 8, width * 10)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    if a.dry_run:
        m = UNet(ch).to(dev)
        t0 = time.time()
        run_epochs(m, loader(clients[sources[0]][0][: a.bs * 2], 0), 1, a.lr, dev, scaler)
        dt = (time.time() - t0) / (a.bs * 2)
        n_train = sum(len(tr) for tr, _ in clients.values())
        est = dt * n_train * a.rounds * a.local_epochs / 3600
        print(f"dry-run OK: {dt:.3f}s/img -> ~{est:.1f} h for {a.rounds}x{a.local_epochs} epochs over {n_train} imgs")
        return

    ckpt = os.path.join(a.out, "ckpt.pt")
    if a.eval_only:
        if not os.path.exists(ckpt):
            raise SystemExit(f"no checkpoint at {ckpt} (pooled/local modes do not save one)")
        m = UNet(ch).to(dev)
        st = torch.load(ckpt, map_location=dev)
        m.load_state_dict(st["model"])
        t0 = time.time()
        summ = summarize(evaluate(m, test, a.size, dev))
        summ.update({"checkpoint_round": st["round"], "seconds": round(time.time() - t0, 1)})
        with open(os.path.join(a.out, "eval_only.json"), "w") as f:
            json.dump(summ, f, indent=1)
        print(f"[eval-only] {a.target} round {st['round']}: Dice OD {summ['dice_od']['mean']:.4f} "
              f"OC {summ['dice_oc']['mean']:.4f} | vCDR MAE {summ['vcdr_abs_err']['mean']:.3f} "
              f"| n={summ['n']} | {summ['seconds']}s")
        return
    log = []
    total_epochs = a.rounds * a.local_epochs
    pred_dir = os.path.join(a.out, "preds") if a.save_preds else None
    if pred_dir:
        os.makedirs(pred_dir, exist_ok=True)

    if a.mode == "pooled":
        m = UNet(ch).to(dev)
        tr = sum([c[0] for c in clients.values()], [])
        run_epochs(m, loader(tr, a.seed), total_epochs, a.lr, dev, scaler)
        rows = evaluate(m, test, a.size, dev, save_dir=pred_dir)
    elif a.mode == "local":
        rows = []
        for k, (d, (tr, _)) in enumerate(clients.items()):
            m = UNet(ch).to(dev)
            run_epochs(m, loader(tr, a.seed + k), total_epochs, a.lr, dev, scaler)
            r = evaluate(m, test, a.size, dev)
            for x in r:
                x["client"] = d
            rows += r
    else:
        glob_m = UNet(ch).to(dev)
        start = 0
        if os.path.exists(ckpt):
            st = torch.load(ckpt, map_location=dev)
            glob_m.load_state_dict(st["model"])
            start, log = st["round"], st["log"]
            print("resumed at round", start)
        sizes = [len(tr) for tr, _ in clients.values()]
        for rd in range(start, a.rounds):
            states, losses = [], []
            for k, (d, (tr, _)) in enumerate(clients.items()):
                m = UNet(ch).to(dev)
                m.load_state_dict(glob_m.state_dict())
                losses.append(run_epochs(m, loader(tr, a.seed * 1000 + rd * 10 + k), a.local_epochs, a.lr, dev, scaler))
                states.append({k2: v.detach().cpu() for k2, v in m.state_dict().items()})
            glob_m.load_state_dict(fedavg(states, sizes))
            val = summarize(evaluate(glob_m, sum([c[1] for c in clients.values()], []), a.size, dev))
            log.append({"round": rd + 1, "loss": losses, "src_val_dice_od": val["dice_od"]["mean"],
                        "src_val_dice_oc": val["dice_oc"]["mean"]})
            print(json.dumps(log[-1]))
            torch.save({"model": glob_m.state_dict(), "round": rd + 1, "log": log}, ckpt)
        rows = evaluate(glob_m, test, a.size, dev, save_dir=pred_dir)

    with open(os.path.join(a.out, "preds.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = summarize([{k: v for k, v in r.items() if k != "client"} for r in rows])
    summ["config"] = vars(a)
    summ["train_log"] = log
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(f"{a.target} {a.mode} style={a.style} fusion={a.fusion}: "
          f"Dice OD {summ['dice_od']['mean']:.4f} OC {summ['dice_oc']['mean']:.4f} | "
          f"vCDR MAE {summ['vcdr_abs_err']['mean']:.3f} | flip@0.6 {summ['refer_flip@0.6']['mean']:.3f} "
          f"| empty-GT-cup {sum(r['gt_cup_empty'] for r in rows)}")


if __name__ == "__main__":
    main()
