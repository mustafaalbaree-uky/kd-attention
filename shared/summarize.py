"""
Compute summary statistics and produce paper-ready figures from divergence and floor CSVs.

Reads:
  students/<arch>/results/metrics/<arch>_divergence_scores.csv
  students/<arch>/results/metrics/<arch>_floor_scores.csv
  students/<arch>/results/metrics/<arch>_accuracy.csv

Writes:
  students/<arch>/results/metrics/<arch>_summary_stats.json
  students/<arch>/results/figures/figure{1-5}_*.png

Usage (from project root):
    python shared/summarize.py --student resnet18
    python shared/summarize.py --student mobilenet
    python shared/summarize.py --student densenet
"""
import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import mannwhitneyu

_HERE = Path(__file__).parent       # shared/
_ROOT = _HERE.parent                # project root


def load_divergence(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def load_accuracy(path: Path) -> dict[str, float]:
    acc = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            acc[row["model_name"]] = float(row["test_accuracy"])
    return acc


def load_floor(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def outcome_masks(student_correct, teacher_correct):
    both_correct  = (student_correct == 1) & (teacher_correct == 1)
    student_wrong = (student_correct == 0) & (teacher_correct == 1)
    both_wrong    = (student_correct == 0) & (teacher_correct == 0)
    return both_correct, student_wrong, both_wrong


def group_stats(arr: np.ndarray, mask: np.ndarray) -> dict:
    subset = arr[mask]
    if len(subset) == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": round(float(subset.mean()), 6),
            "std":  round(float(subset.std()),  6),
            "n":    int(mask.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True,
                        choices=["resnet18", "mobilenet", "densenet"],
                        help="Student architecture to summarize")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    student_dir    = _ROOT / "students" / args.student
    divergence_csv = student_dir / "results" / "metrics" / f"{args.student}_divergence_scores.csv"
    floor_csv      = student_dir / "results" / "metrics" / f"{args.student}_floor_scores.csv"
    accuracy_csv   = student_dir / "results" / "metrics" / f"{args.student}_accuracy.csv"
    stats_json     = student_dir / "results" / "metrics" / f"{args.student}_summary_stats.json"
    figures_dir    = student_dir / "results" / "figures"

    figures_dir.mkdir(parents=True, exist_ok=True)
    stats_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"Student : {args.student}")
    print(f"CSV     : {divergence_csv}")
    print(f"Figures : {figures_dir}\n")

    rows       = load_divergence(divergence_csv)
    floor_rows = load_floor(floor_csv)
    acc        = load_accuracy(accuracy_csv)

    js_kd   = np.array([float(r["js_teacher_kd"])             for r in rows])
    js_bl   = np.array([float(r["js_teacher_baseline"])       for r in rows])
    sp_kd   = np.array([float(r["spearman_teacher_kd"])       for r in rows])
    sp_bl   = np.array([float(r["spearman_teacher_baseline"]) for r in rows])
    ss_kd   = np.array([float(r["ssim_teacher_kd"])           for r in rows])
    ss_bl   = np.array([float(r["ssim_teacher_baseline"])     for r in rows])
    miou_kd = np.array([float(r["miou_teacher_kd"])           for r in rows])
    miou_bl = np.array([float(r["miou_teacher_baseline"])     for r in rows])

    teacher_correct  = np.array([int(r["teacher_correct"])  for r in rows])
    kd_correct       = np.array([int(r["kd_correct"])       for r in rows])
    baseline_correct = np.array([int(r["baseline_correct"]) for r in rows])

    kd_bc, kd_sw, kd_bw = outcome_masks(kd_correct,       teacher_correct)
    bl_bc, bl_sw, bl_bw = outcome_masks(baseline_correct,  teacher_correct)

    arch = args.student
    stats = {
        "js_divergence": {
            "kd_student": {"mean": round(float(js_kd.mean()), 6), "std": round(float(js_kd.std()), 6)},
            "baseline":   {"mean": round(float(js_bl.mean()), 6), "std": round(float(js_bl.std()), 6)},
        },
        "spearman_r": {
            "kd_student": {"mean": round(float(sp_kd.mean()), 6), "std": round(float(sp_kd.std()), 6)},
            "baseline":   {"mean": round(float(sp_bl.mean()), 6), "std": round(float(sp_bl.std()), 6)},
        },
        "ssim": {
            "kd_student": {"mean": round(float(ss_kd.mean()), 6), "std": round(float(ss_kd.std()), 6)},
            "baseline":   {"mean": round(float(ss_bl.mean()), 6), "std": round(float(ss_bl.std()), 6)},
        },
        "miou": {
            "kd_student": {"mean": round(float(miou_kd.mean()), 6), "std": round(float(miou_kd.std()), 6)},
            "baseline":   {"mean": round(float(miou_bl.mean()), 6), "std": round(float(miou_bl.std()), 6)},
        },
        "js_by_outcome": {
            "kd_student": {
                "both_correct":                  group_stats(js_kd, kd_bc),
                "student_wrong_teacher_correct": group_stats(js_kd, kd_sw),
                "both_wrong":                    group_stats(js_kd, kd_bw),
            },
            "baseline": {
                "both_correct":                  group_stats(js_bl, bl_bc),
                "student_wrong_teacher_correct": group_stats(js_bl, bl_sw),
                "both_wrong":                    group_stats(js_bl, bl_bw),
            },
        },
        "ssim_by_outcome": {
            "kd_student": {
                "both_correct":                  group_stats(ss_kd, kd_bc),
                "student_wrong_teacher_correct": group_stats(ss_kd, kd_sw),
                "both_wrong":                    group_stats(ss_kd, kd_bw),
            },
            "baseline": {
                "both_correct":                  group_stats(ss_bl, bl_bc),
                "student_wrong_teacher_correct": group_stats(ss_bl, bl_sw),
                "both_wrong":                    group_stats(ss_bl, bl_bw),
            },
        },
        "miou_by_outcome": {
            "kd_student": {
                "both_correct":                  group_stats(miou_kd, kd_bc),
                "student_wrong_teacher_correct": group_stats(miou_kd, kd_sw),
                "both_wrong":                    group_stats(miou_kd, kd_bw),
            },
            "baseline": {
                "both_correct":                  group_stats(miou_bl, bl_bc),
                "student_wrong_teacher_correct": group_stats(miou_bl, bl_sw),
                "both_wrong":                    group_stats(miou_bl, bl_bw),
            },
        },
        "top1_accuracy": {
            "teacher":    acc.get("teacher_resnet50"),
            "kd_student": acc.get(f"student_kd_{arch}") or acc.get("student_kd_resnet18"),
            "baseline":   acc.get(f"student_baseline_{arch}") or acc.get("student_baseline_resnet18"),
        },
    }

    if floor_rows:
        fl_js   = np.array([float(r["js_floor"])       for r in floor_rows])
        fl_sp   = np.array([float(r["spearman_floor"]) for r in floor_rows])
        fl_ss   = np.array([float(r["ssim_floor"])     for r in floor_rows])
        stats["floor_js_mean"]       = round(float(fl_js.mean()), 6)
        stats["floor_js_std"]        = round(float(fl_js.std()),  6)
        stats["floor_spearman_mean"] = round(float(fl_sp.mean()), 6)
        stats["floor_spearman_std"]  = round(float(fl_sp.std()),  6)
        stats["floor_ssim_mean"]     = round(float(fl_ss.mean()), 6)
        stats["floor_ssim_std"]      = round(float(fl_ss.std()),  6)
        if "miou_floor" in floor_rows[0]:
            fl_miou = np.array([float(r["miou_floor"]) for r in floor_rows])
            stats["floor_miou_mean"] = round(float(fl_miou.mean()), 6)
            stats["floor_miou_std"]  = round(float(fl_miou.std()),  6)
        else:
            stats["floor_miou_mean"] = None
            stats["floor_miou_std"]  = None

    mw_u, mw_p = mannwhitneyu(js_kd, js_bl, alternative="less")
    stats["mann_whitney_u_statistic"] = float(mw_u)
    stats["mann_whitney_p_value"]     = float(mw_p)
    print(f"Mann-Whitney U (JS)   = {mw_u:.1f},  p = {mw_p:.6e}")

    mw_u_miou, mw_p_miou = mannwhitneyu(miou_kd, miou_bl, alternative="greater")
    stats["mann_whitney_u_miou"] = float(mw_u_miou)
    stats["mann_whitney_p_miou"] = float(mw_p_miou)
    print(f"Mann-Whitney U (mIoU) = {mw_u_miou:.1f},  p = {mw_p_miou:.6e}")

    with stats_json.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved -> {stats_json}")

    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
    })

    KD_COLOR     = "#2166ac"
    BL_COLOR     = "#d6604d"
    GROUP_COLORS = ["#4dac26", "#f4a582", "#bababa"]

    arch_label = arch.title()

    # Figure 1 - JS divergence bar chart
    fig, ax = plt.subplots(figsize=(4.5, 4))
    means = [js_kd.mean(), js_bl.mean()]
    stds  = [js_kd.std(),  js_bl.std()]
    x     = np.array([0, 1])
    bars = ax.bar(x, means, 0.5, yerr=stds, capsize=5,
                  color=[KD_COLOR, BL_COLOR],
                  error_kw={"elinewidth": 1.2, "capthick": 1.2}, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"KD Student\n({arch_label})", f"Baseline\n({arch_label})"])
    ax.set_ylabel("Mean JS Distance vs Teacher")
    ax.set_title("Grad-CAM Spatial Divergence from Teacher", pad=10)
    ax.set_ylim(0, max(means) * 1.55)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.004,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    ax.annotate("vs Teacher (ResNet-50)", xy=(0.5, 0.93), xycoords="axes fraction",
                ha="center", fontsize=9, color="grey")
    plt.tight_layout()
    fig.savefig(figures_dir / "figure1_js_divergence_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {figures_dir / 'figure1_js_divergence_bar.png'}")

    # Figure 2 - JS divergence by outcome group (KD student)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    group_labels = ["Both Correct", "Student Wrong\n+ Teacher Correct", "Both Wrong"]
    group_masks  = [kd_bc, kd_sw, kd_bw]
    means_g, stds_g, ns_g = [], [], []
    for mask in group_masks:
        sub = js_kd[mask]
        means_g.append(sub.mean() if len(sub) else 0.0)
        stds_g.append(sub.std()   if len(sub) else 0.0)
        ns_g.append(int(mask.sum()))
    x = np.arange(len(group_labels))
    bars = ax.bar(x, means_g, 0.55, yerr=stds_g, capsize=5,
                  color=GROUP_COLORS,
                  error_kw={"elinewidth": 1.2, "capthick": 1.2}, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Mean JS Distance vs Teacher")
    ax.set_title(f"KD Student - JS Divergence by Prediction Outcome", pad=10)
    ax.set_ylim(0, max(means_g) * 1.6)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)
    for bar, m, s, n in zip(bars, means_g, stds_g, ns_g):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.006,
                f"{m:.3f}\n(n={n})", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(figures_dir / "figure2_js_by_outcome.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {figures_dir / 'figure2_js_by_outcome.png'}")

    # Figure 3 - Spearman r distribution (overlaid histograms)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(-0.5, 1.0, 31)
    ax.hist(sp_kd, bins=bins, alpha=0.6, color=KD_COLOR, label="KD Student", density=True, zorder=3)
    ax.hist(sp_bl, bins=bins, alpha=0.6, color=BL_COLOR,  label="Baseline",   density=True, zorder=3)
    ax.axvline(sp_kd.mean(), color=KD_COLOR, linestyle="--", linewidth=1.4,
               label=f"KD mean = {sp_kd.mean():.3f}")
    ax.axvline(sp_bl.mean(), color=BL_COLOR,  linestyle="--", linewidth=1.4,
               label=f"Baseline mean = {sp_bl.mean():.3f}")
    ax.set_xlabel("Spearman r  (Teacher vs Student Grad-CAM)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Spatial Rank Correlation with Teacher", pad=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)
    plt.tight_layout()
    fig.savefig(figures_dir / "figure3_spearman_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {figures_dir / 'figure3_spearman_distribution.png'}")

    # Figure 4 - SSIM by outcome group (KD + baseline side by side)
    fig, ax = plt.subplots(figsize=(7, 4))
    group_labels = ["Both Correct", "Student Wrong\n+ Teacher Correct", "Both Wrong"]
    kd_means, kd_stds = [], []
    bl_means, bl_stds = [], []
    for kd_mask, bl_mask in zip([kd_bc, kd_sw, kd_bw], [bl_bc, bl_sw, bl_bw]):
        kd_sub = ss_kd[kd_mask]; bl_sub = ss_bl[bl_mask]
        kd_means.append(kd_sub.mean() if len(kd_sub) else 0.0)
        kd_stds.append(kd_sub.std()   if len(kd_sub) else 0.0)
        bl_means.append(bl_sub.mean() if len(bl_sub) else 0.0)
        bl_stds.append(bl_sub.std()   if len(bl_sub) else 0.0)
    x     = np.arange(len(group_labels))
    width = 0.35
    err_kw = {"elinewidth": 1.2, "capthick": 1.2, "capsize": 4}
    bars_kd = ax.bar(x - width / 2, kd_means, width, yerr=kd_stds,
                     color=KD_COLOR, label="KD Student", error_kw=err_kw, zorder=3)
    bars_bl = ax.bar(x + width / 2, bl_means, width, yerr=bl_stds,
                     color=BL_COLOR, label="Baseline", error_kw=err_kw, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Mean SSIM vs Teacher")
    ax.set_title("SSIM Alignment by Prediction Outcome", pad=10)
    y_max = max(max(kd_means), max(bl_means))
    ax.set_ylim(min(min(kd_means), min(bl_means)) - 0.1, y_max * 1.35)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)
    ax.legend(frameon=False, fontsize=9)
    for bar, m, s in zip(bars_kd, kd_means, kd_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, m, s in zip(bars_bl, bl_means, bl_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(figures_dir / "figure4_ssim_by_outcome.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {figures_dir / 'figure4_ssim_by_outcome.png'}")

    # Figure 5 - mIoU by outcome group (KD + baseline side by side)
    fig, ax = plt.subplots(figsize=(7, 4))
    kd_means5, kd_stds5 = [], []
    bl_means5, bl_stds5 = [], []
    for kd_mask, bl_mask in zip([kd_bc, kd_sw, kd_bw], [bl_bc, bl_sw, bl_bw]):
        kd_sub = miou_kd[kd_mask]; bl_sub = miou_bl[bl_mask]
        kd_means5.append(kd_sub.mean() if len(kd_sub) else 0.0)
        kd_stds5.append(kd_sub.std()   if len(kd_sub) else 0.0)
        bl_means5.append(bl_sub.mean() if len(bl_sub) else 0.0)
        bl_stds5.append(bl_sub.std()   if len(bl_sub) else 0.0)
    x     = np.arange(len(group_labels))
    width = 0.35
    bars_kd5 = ax.bar(x - width / 2, kd_means5, width, yerr=kd_stds5,
                      color=KD_COLOR, label="KD Student", error_kw=err_kw, zorder=3)
    bars_bl5 = ax.bar(x + width / 2, bl_means5, width, yerr=bl_stds5,
                      color=BL_COLOR, label="Baseline", error_kw=err_kw, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Mean mIoU vs Teacher")
    ax.set_title("mIoU by Outcome Group", pad=10)
    y_max5 = max(max(kd_means5), max(bl_means5))
    y_min5 = min(min(kd_means5), min(bl_means5))
    ax.set_ylim(max(0.0, y_min5 - 0.05), y_max5 * 1.35)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)
    ax.legend(frameon=False, fontsize=9)
    for bar, m, s in zip(bars_kd5, kd_means5, kd_stds5):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, m, s in zip(bars_bl5, bl_means5, bl_stds5):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(figures_dir / "figure5_miou_by_outcome.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {figures_dir / 'figure5_miou_by_outcome.png'}")

    print(f"\nSanity check:")
    print(f"  mean js_teacher_kd         = {js_kd.mean():.6f}")
    print(f"  mean js_teacher_baseline   = {js_bl.mean():.6f}")
    print(f"  mean ssim_teacher_kd       = {ss_kd.mean():.6f}")
    print(f"  mean ssim_teacher_baseline = {ss_bl.mean():.6f}")
    print(f"  mean miou_teacher_kd       = {miou_kd.mean():.6f}")
    print(f"  mean miou_teacher_baseline = {miou_bl.mean():.6f}")


if __name__ == "__main__":
    main()
