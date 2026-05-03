"""
Compute floor metrics (JS, Spearman, SSIM, mIoU) between seed-42 and seed-43
baseline Grad-CAM maps.

Reads students/<arch>/results/gradcam_full/arrays_seed43/*.npz.
Writes students/<arch>/results/metrics/<arch>_floor_scores.csv.

Usage (from project root):
    python shared/score_floor.py --student resnet18
    python shared/score_floor.py --student mobilenet
    python shared/score_floor.py --student densenet
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim_fn

_ROOT = Path(__file__).parent.parent    # project root


def js(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(jensenshannon(a, b))


def spear(a, b):
    return float(spearmanr(a.ravel(), b.ravel()).statistic)


def ssim(a, b):
    return float(ssim_fn(
        a.astype(np.float64).reshape(7, 7),
        b.astype(np.float64).reshape(7, 7),
        data_range=1.0, win_size=7,
    ))


def miou(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    ious = []
    for p in range(0, 101, 5):  # 21 thresholds: 0, 5, 10, ..., 100
        t1 = np.percentile(a, p)
        t2 = np.percentile(b, p)
        mask1 = a >= t1
        mask2 = b >= t2
        intersection = float(np.sum(mask1 & mask2))
        union = float(np.sum(mask1 | mask2))
        ious.append(intersection / union if union > 0 else 0.0)
    return float(np.mean(ious))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True,
                        choices=["resnet18", "mobilenet", "densenet"],
                        help="Student architecture to score")
    args = parser.parse_args()

    arrays_dir = _ROOT / "students" / args.student / "results" / "gradcam_full" / "arrays_seed43"
    out_csv    = _ROOT / "students" / args.student / "results" / "metrics" / f"{args.student}_floor_scores.csv"

    npz_files = sorted(arrays_dir.glob("*.npz"))
    n_total   = len(npz_files)
    print(f"Student  : {args.student}")
    print(f"Arrays   : {arrays_dir}  ({n_total} files)")
    print(f"Output   : {out_csv}\n")

    rows = []
    for i, path in enumerate(npz_files):
        data = np.load(path)
        map42 = data["baseline_seed42"]
        map43 = data["baseline_seed43"]

        rows.append({
            "filename":       path.name,
            "js_floor":       js(map42, map43),
            "spearman_floor": spear(map42, map43),
            "ssim_floor":     ssim(map42, map43),
            "miou_floor":     miou(map42, map43),
        })

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n_total} …")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "js_floor", "spearman_floor", "ssim_floor", "miou_floor"])
        writer.writeheader()
        writer.writerows(rows)

    js_vals   = np.array([r["js_floor"]       for r in rows])
    sp_vals   = np.array([r["spearman_floor"] for r in rows])
    ssim_vals = np.array([r["ssim_floor"]     for r in rows])
    miou_vals = np.array([r["miou_floor"]     for r in rows])
    print(f"\n{args.student} floor summary ({len(rows)} images):")
    print(f"  JS       mean={np.mean(js_vals):.4f}  std={np.std(js_vals):.4f}")
    print(f"  Spearman mean={np.mean(sp_vals):.4f}  std={np.std(sp_vals):.4f}")
    print(f"  SSIM     mean={np.mean(ssim_vals):.4f}  std={np.std(ssim_vals):.4f}")
    print(f"  mIoU     mean={np.mean(miou_vals):.4f}  std={np.std(miou_vals):.4f}")
    print(f"\nSaved -> {out_csv}")


if __name__ == "__main__":
    main()
