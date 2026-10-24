"""Shared utilities for replicating Hu et al. (IEEE IoT-J 2026).

Standard on-disk format produced by prepare_data.py:
    <root>/<DATASET>/images/<id>.png   RGB, preprocessed (ROI crop -> resize -> CLAHE -> median)
    <root>/<DATASET>/masks/<id>.png    uint8 label map: 0 background, 1 disc rim, 2 cup
Disc = label >= 1 (disc includes cup), Cup = label == 2.
"""
import os
import glob
import numpy as np
from skimage import io, transform, exposure, filters
from scipy import ndimage

DATASETS = ["G1020", "ORIGA", "REFUGE", "PAPILA"]


# ---------------------------------------------------------------- preprocessing
def find_roi_center(img_rgb, mode="paper"):
    """Paper: brightest pixel of the grayscale image (np.argmax -> first pixel on ties).
    'smooth': blur first (robust to specular dots).
    'blob'  : blur, keep the brightest 0.3% of pixels inside the field of view, centroid of the largest blob."""
    g = img_rgb.astype(np.float32).mean(axis=2)
    if mode in ("smooth", "blob"):
        g = ndimage.gaussian_filter(g, sigma=max(3, g.shape[1] // 100))
    if mode == "blob":
        fov = ndimage.binary_erosion(g > 15, iterations=max(3, g.shape[1] // 20))   # drop the bright FOV rim / borders
        vals = g[fov] if fov.any() else g.ravel()
        m = (g >= np.percentile(vals, 99.7)) & (fov if fov.any() else True)
        lab, n = ndimage.label(m)
        if n:
            k = 1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1))))
            r, c = ndimage.center_of_mass(lab == k)
            return int(r), int(c)
    r, c = np.unravel_index(np.argmax(g), g.shape)
    return int(r), int(c)


def crop_square(arr, center, size):
    """Crop size x size around center, zero-padding where the window leaves the image."""
    r, c = center
    h, w = arr.shape[:2]
    half = size // 2
    out = np.zeros((size, size) + arr.shape[2:], dtype=arr.dtype)
    r0, c0 = r - half, c - half
    sr0, sc0 = max(r0, 0), max(c0, 0)
    sr1, sc1 = min(r0 + size, h), min(c0 + size, w)
    out[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = arr[sr0:sr1, sc0:sc1]
    return out


def preprocess(img_rgb, mask, out_size=512, crop=1000, roi="paper"):
    """Paper pipeline: ROI at brightest pixel, crop 1000x1000, resize 512, CLAHE, 3x3 median.

    `crop` may be an int (paper) or a float in (0,1] = fraction of image width (scale-normalised variant).
    Returns (img uint8 HxWx3, mask uint8 HxW, roi_center).
    """
    size = int(round(crop * img_rgb.shape[1])) if isinstance(crop, float) else int(crop)
    if roi == "gt":   # crop centred on the ground-truth disc: no ROI failures, but uses labels (state in report)
        ys, xs = np.nonzero(mask >= 1)
        center = (int(ys.mean()), int(xs.mean())) if len(ys) else find_roi_center(img_rgb, "smooth")
    else:
        center = find_roi_center(img_rgb, roi)
    img = crop_square(img_rgb, center, size)
    m = crop_square(mask, center, size)
    img = transform.resize(img, (out_size, out_size), order=1, preserve_range=True, anti_aliasing=True)
    m = transform.resize(m, (out_size, out_size), order=0, preserve_range=True, anti_aliasing=False)
    img = exposure.equalize_adapthist(img.astype(np.uint8))           # CLAHE per channel -> float [0,1]
    img = (img * 255).astype(np.uint8)
    img = np.stack([ndimage.median_filter(img[..., k], size=3) for k in range(3)], axis=2)
    return img, m.astype(np.uint8), center


# ---------------------------------------------------------------- dataset index
def list_dataset(root, name):
    ims = sorted(glob.glob(os.path.join(root, name, "images", "*.png")))
    out = []
    for p in ims:
        mid = os.path.basename(p)
        mp = os.path.join(root, name, "masks", mid)
        if os.path.exists(mp):
            out.append((name, os.path.splitext(mid)[0], p, mp))
    return out


def load_pair(item, size=None):
    _, _, ip, mp = item
    img = io.imread(ip)[..., :3]
    m = io.imread(mp)
    if size is not None and img.shape[0] != size:
        img = transform.resize(img, (size, size), order=1, preserve_range=True, anti_aliasing=True).astype(np.uint8)
        m = transform.resize(m, (size, size), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
    return img, m


# ---------------------------------------------------------------- metrics
def dice(pred, gt):
    """Returns NaN when GT is empty (caller must count these, not treat as 0 or 1)."""
    gs = gt.sum()
    if gs == 0:
        return float("nan")
    return 2.0 * np.logical_and(pred, gt).sum() / (pred.sum() + gs)


def iou(pred, gt):
    if gt.sum() == 0:
        return float("nan")
    u = np.logical_or(pred, gt).sum()
    return np.logical_and(pred, gt).sum() / u


def vertical_extent(m):
    rows = np.where(m.any(axis=1))[0]
    return 0 if len(rows) == 0 else rows[-1] - rows[0] + 1


def vcdr(label_map):
    """Vertical cup-to-disc ratio from a label map (0 bg, 1 rim, 2 cup). NaN if no disc."""
    d = vertical_extent(label_map >= 1)
    c = vertical_extent(label_map == 2)
    return float("nan") if d == 0 else c / d


def largest_component(m):
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def postprocess(label_map):
    """Keep largest disc component, cup restricted to disc. Standard OD/OC post-processing."""
    disc = largest_component(label_map >= 1)
    cup = largest_component((label_map == 2) & disc)
    out = np.zeros_like(label_map)
    out[disc] = 1
    out[cup] = 2
    return out


def case_metrics(pred_lab, gt_lab, thresholds=(0.6, 0.7)):
    pd_, gd = pred_lab >= 1, gt_lab >= 1
    pc, gc = pred_lab == 2, gt_lab == 2
    v_p, v_g = vcdr(pred_lab), vcdr(gt_lab)
    r = {
        "dice_od": dice(pd_, gd), "dice_oc": dice(pc, gc),
        "iou_od": iou(pd_, gd), "iou_oc": iou(pc, gc),
        "vcdr_pred": v_p, "vcdr_gt": v_g,
        "vcdr_abs_err": abs(v_p - v_g) if not (np.isnan(v_p) or np.isnan(v_g)) else float("nan"),
        "gt_cup_empty": int(gc.sum() == 0),
    }
    for t in thresholds:
        if np.isnan(v_p) or np.isnan(v_g):
            r[f"refer_flip@{t}"] = float("nan")
        else:
            r[f"refer_flip@{t}"] = int((v_p >= t) != (v_g >= t))
    return r


# ---------------------------------------------------------------- synthetic data (smoke tests only)
def synth_fundus(rng, size=128, style=0, vcdr_target=None):
    """Toy fundus-like image + label map. NOT a substitute for real data; used only to test code paths.

    style: integer domain id -> different colour cast / gamma / blur.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy, cx = size / 2 + rng.normal(0, size * 0.05, 2)
    rd = size * rng.uniform(0.18, 0.24)                    # disc vertical radius
    v = vcdr_target if vcdr_target is not None else rng.uniform(0.3, 0.85)
    rc = rd * v
    asp_d, asp_c = rng.uniform(0.85, 1.0), rng.uniform(0.75, 1.0)
    disc = ((yy - cy) / rd) ** 2 + ((xx - cx) / (rd * asp_d)) ** 2 <= 1
    cup = ((yy - cy) / rc) ** 2 + ((xx - cx) / (rc * asp_c)) ** 2 <= 1
    lab = np.zeros((size, size), np.uint8)
    lab[disc] = 1
    lab[cup & disc] = 2
    base = np.exp(-(((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (2 * (size * 0.6) ** 2)))
    img = np.stack([0.75 * base, 0.35 * base, 0.15 * base], axis=2)
    for _ in range(rng.integers(4, 8)):                    # vessel-like lines from the disc
        ang = rng.uniform(0, 2 * np.pi)
        t = np.linspace(0, size, 4 * size)
        py = cy + t * np.sin(ang) + rng.normal(0, 3) * np.sin(t / 15)
        px = cx + t * np.cos(ang)
        ok = (py >= 0) & (py < size) & (px >= 0) & (px < size)
        img[py[ok].astype(int), px[ok].astype(int)] *= 0.55
    img[disc] = img[disc] * 0.5 + np.array([0.95, 0.75, 0.45]) * 0.5
    img[cup & disc] = img[cup & disc] * 0.4 + np.array([1.0, 0.95, 0.8]) * 0.6
    casts = [np.array([1.0, 1.0, 1.0]), np.array([1.1, 0.8, 0.7]), np.array([0.85, 1.05, 1.1]), np.array([1.0, 0.9, 1.2])]
    gammas = [1.0, 1.4, 0.75, 1.15]
    img = np.clip(img * casts[style % 4], 0, 1) ** gammas[style % 4]
    img = ndimage.gaussian_filter(img, sigma=(0.4 * (style % 4), 0.4 * (style % 4), 0))
    img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)
    fov = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) <= (0.62 * size) ** 2   # black corners like a real fundus FOV
    img[~fov] = 0
    return (img * 255).astype(np.uint8), lab
