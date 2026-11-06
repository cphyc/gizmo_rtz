"""Export accepted GPU trial speedups and timing provenance as a PDF."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "build/matplotlib-cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import numpy as np

RESULTS = ROOT / "benchmarks/results"
COLORS = {"float64": "#2878a0", "mixed": "#2c8b57", "historical": "#ba812b"}
# Each entry names its own measured parent; ratios are not multiplied across branches.
TRIALS = [
    ("T02", "Active-cell compaction", "trial02", "baseline", "float64"),
    ("T03", "32-thread teams", "trial03", "trial02", "float64"),
    ("Layout*", "Compact tables + descriptor, F64", "float64-compact-baseline", "trial03", "float64"),
    ("T08", "Cache metal recombination / ionization", "rate-cache-gpu", "float64-compact-baseline", "float64"),
    ("T11", "Separate H/He kernel", "split-hhe-gpu", "rate-cache-gpu", "float64"),
    ("T12", "Skip zero charge-transfer fits", "zero-ct-gpu", "zero-ct-baseline-recheck", "float64"),
    ("T13", "Cache metal charge-transfer rates", "ct-cache-gpu", "ct-cache-baseline-recheck", "float64"),
    ("P02", "Charge transfer → float32", "precision-timing-charge", "precision-timing-float64", "mixed"),
    ("P04", "Cosmic rays → float32", "precision-timing-cosmic", "precision-timing-charge", "mixed"),
    ("P05", "Dust recombination → float32", "precision-timing-dust", "precision-timing-cosmic", "mixed"),
    ("P06", "Molecular helpers → float32", "precision-timing-molecules", "precision-timing-dust", "mixed"),
    ("P07", "UV background → float32", "precision-timing-uvb", "precision-timing-molecules", "mixed"),
    ("T04†", "Float32 chemistry storage", "trial04", "trial03", "historical"),
    ("Layout*†", "Compact table mapping, F32", "compact-policy-final-gpu", "trial05", "historical"),
    ("T06†", "Device-resident descriptor, F32", "descriptor-gpu", "compact-policy-final-gpu", "historical"),
    ("P01†", "Collisional ionization → float32", "per-file-collisional", "ct-cache-gpu", "historical"),
]


def load(stem):
    path = RESULTS / f"{stem}.json"
    result = json.loads(path.read_text())
    samples = np.asarray(result["seconds"], dtype=float)
    if len(samples) != 5 or not np.isfinite(samples).all():
        raise ValueError(f"Five finite warmed samples required: {path}")
    return result, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/figures/accepted_gpu_trials.pdf")
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    rows = []
    for tag, label, candidate, parent, kind in TRIALS:
        result, samples = load(candidate)
        _, before = load(parent)
        if kind != "historical" and result.get("accuracy_passed") is False:
            raise ValueError(f"Accepted current trial failed accuracy: {candidate}")
        ratio = float(np.median(before) / np.median(samples))
        rows.append(dict(tag=tag, label=label, candidate=candidate, parent=parent,
                         category=kind, before_seconds=float(np.median(before)),
                         after_seconds=float(np.median(samples)), speedup=ratio,
                         range_low=float(before.min() / samples.max()),
                         range_high=float(before.max() / samples.min())))
    selected, selected_samples = load("precision-timing-selected")
    control, control_samples = load("precision-timing-float64")
    if not selected["accuracy_passed"] or not control["accuracy_passed"]:
        raise ValueError("Final selected/control accuracy must pass")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output, metadata={"Title": "Accepted galaxy GPU trials and relative speedups"}) as pdf:
        fig, ax = plt.subplots(figsize=(10.5, 13))
        positions = np.arange(len(rows))[::-1].astype(float)
        positions[:12] += 0.6  # Visual separation of superseded precision paths.
        labels = []
        for y, row in zip(positions, rows):
            x = row["speedup"]
            color = COLORS[row["category"]]
            ax.barh(y, x - 1, left=1, height=0.5, color=color, alpha=0.8)
            ax.errorbar(x, y, xerr=[[x - row["range_low"]], [row["range_high"] - x]],
                        fmt="o", color=color, capsize=2, markersize=3.5, linewidth=0.8)
            ax.text(x + 0.035, y, f"{x:.3f}×", va="center", fontsize=9)
            labels.append(f"{row['tag']}  {row['label']}\n"
                          f"{row['before_seconds']:.3f} → {row['after_seconds']:.3f} s")
        ax.axvline(1, color="#333333", linewidth=1)
        ax.axhline(3.8, color="#bbbbbb", linestyle="--", linewidth=0.8)
        ax.set(yticks=positions, yticklabels=labels, xlim=(0.5, 2.65),
               xlabel="Speedup vs measured parent (parent median / trial median)",
               title="Accepted GPU trials — incremental comparisons")
        ax.tick_params(axis="y", length=0, labelsize=8.5)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        handles = [Patch(facecolor=COLORS[k], label=v) for k, v in [
            ("float64", "Float64; exact-output branch"),
            ("mixed", "Current mixed precision; 1e-5 gate"),
            ("historical", "Historical precision / superseded acceptance†")]]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.975), fontsize=8, ncol=1)
        fig.text(0.04, 0.017,
                 "10,000 cells · 100 yr · Quadro T1000 Max-Q · medians of five warmed solves.\n"
                 "Whiskers span sample-extrema ratios, not confidence intervals; near-1 values can be timing noise.\n"
                 "* Retained layout functionality, including measured regressions. † Historical paths do not pass today's 1e-5 gate.\n"
                 "Parents differ by branch; do not multiply all bars. Rejected trials are omitted. JSON sources accompany this PDF.",
                 fontsize=7.5, va="bottom")
        fig.subplots_adjust(left=0.43, right=0.97, top=0.90, bottom=0.09)
        pdf.savefig(fig)
        fig.savefig(ROOT / "build/accepted_gpu_trials_page1.png", dpi=130)
        plt.close(fig)

        progression = [
            ("Base", "baseline"), ("T02", "trial02"), ("T03", "trial03"),
            ("Layout64", "float64-compact-baseline"), ("T08", "rate-cache-gpu"),
            ("T11", "split-hhe-gpu"), ("T12", "zero-ct-gpu"), ("T13", "ct-cache-gpu"),
            ("P02", "precision-timing-charge"), ("P04", "precision-timing-cosmic"),
            ("P05", "precision-timing-dust"), ("P06", "precision-timing-molecules"),
            ("P07", "precision-timing-uvb"), ("Selected", "precision-timing-selected"),
        ]
        datasets = [load(stem)[1] for _, stem in progression]
        times = np.array([np.median(a) for a in datasets])
        low = times - np.array([a.min() for a in datasets])
        high = np.array([a.max() for a in datasets]) - times
        x = np.arange(len(times))
        colors = [COLORS["float64"] if i < 8 else COLORS["mixed"] for i in x]
        fig, axes = plt.subplots(2, 1, figsize=(11.5, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [1.3, 1]})
        axes[0].bar(x, times, color=colors, alpha=0.8)
        axes[0].errorbar(x, times, yerr=[low, high], fmt="none", color="#333333", capsize=2)
        axes[0].set(ylabel="Median solve time (s)", title="Observed progression on the local GPU")
        axes[0].text(x[-1], times[-1] + 1, f"{times[-1]:.3f} s", ha="center", weight="bold")
        axes[1].plot(x, times[0] / times, "o-", color=COLORS["mixed"])
        axes[1].set(ylabel="Speedup vs original baseline", xticks=x,
                    xticklabels=[tag for tag, _ in progression])
        axes[1].axhline(1, color="#777777", linestyle="--", linewidth=0.7)
        for axis in axes:
            axis.grid(axis="y", alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_axisbelow(True)
        control_speedup = float(np.median(control_samples) / np.median(selected_samples))
        fig.text(0.08, 0.035,
                 f"Final mixed / saved optimized float64 control: {control_speedup:.3f}×  "
                 f"({np.median(control_samples):.3f} → {np.median(selected_samples):.3f} s).\n"
                 "Historical medians and a fresh per-file cohort; precision branches are identified on page 1.\n"
                 "Atomic benchmark disables H2/CO. Molecular accuracy was checked separately; neutral timing changes are not claimed as gains.",
                 fontsize=8)
        fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.16, hspace=0.22)
        pdf.savefig(fig)
        fig.savefig(ROOT / "build/accepted_gpu_trials_page2.png", dpi=130)
        plt.close(fig)
    data = {"rows": rows, "progression": [{"tag": tag, "source": stem,
            "median_seconds": float(t)} for (tag, stem), t in zip(progression, times)],
            "final_saved_float64_control_speedup": control_speedup,
            "note": "Historical precision branches are not eligible under the current 1e-5 gate."}
    (RESULTS / "accepted_gpu_trials_plot_data.json").write_text(json.dumps(data, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
