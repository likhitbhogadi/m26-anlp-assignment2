"""Post-hoc analysis of the saved generations (CPU only; no model needed).

Usage:
    uv run python -m src.part3.analyze

Reads `<output_dir>/results.csv` + `<output_dir>/<name>.jsonl`, writes
`<output_dir>/analysis.csv` (extra diagnostics per experiment) and the report figures
to `src/part3/assets/` (PDF for LaTeX, PNG for quick viewing).
"""

import argparse
import csv
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ASSETS = "src/part3/assets"

# Reference categorical palette (slots 1-3 validate all-pairs) + neutral for the baseline.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAMILY_STYLE = {  # family -> (label, color, marker)
    "greedy/beam": ("Greedy / beam", BLUE, "o"),
    "top_k": ("Top-k", ORANGE, "s"),
    "top_p": ("Top-p", AQUA, "^"),
    "baseline": ("Greedy, 0-shot", GRAY, "D"),
}


def family(row: dict) -> str:
    if row["n_shots"] == "0":
        return "baseline"
    return {"greedy": "greedy/beam", "beam": "greedy/beam"}.get(row["strategy"], row["strategy"])


def short_label(row: dict) -> str:
    s, name = row["strategy"], row["name"]
    if name == "greedy_0shot":
        return "greedy (0-shot)"
    if s == "greedy":
        return "greedy"
    if s == "beam":
        return f"beam w={row['num_beams']}"
    t = float(row["temperature"])
    knob = f"k={row['k']}" if s == "top_k" else f"p={float(row['p']):g}"
    return knob + (f", T={t:g}" if t != 1.0 else "")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def repeated_4gram_rate(ids: list[int]) -> float:
    grams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def distinct_n(records: list[dict], n: int) -> float:
    grams = [tuple(r["gen_ids"][i : i + n]) for r in records for i in range(len(r["gen_ids"]) - n + 1)]
    return len(set(grams)) / len(grams)


def positional_accuracy_curve(records: list[dict], max_pos: int) -> list[float]:
    """Fraction of samples whose generated token at position i equals the reference token."""
    curve = []
    for i in range(max_pos):
        valid = [r for r in records if len(r["ref_ids"]) > i]
        curve.append(st.fmean(len(r["gen_ids"]) > i and r["gen_ids"][i] == r["ref_ids"][i] for r in valid))
    return curve


def diagnostics(records: list[dict]) -> dict:
    return {
        "acc_first_token": st.fmean(r["gen_ids"][0] == r["ref_ids"][0] for r in records),
        "acc_first10": st.fmean(
            sum(g == t for g, t in zip(r["gen_ids"][:10], r["ref_ids"][:10])) / 10 for r in records
        ),
        "zero_acc_frac": st.fmean(r["acc"] == 0 for r in records),
        "newline_start_frac": st.fmean(r["generation"].startswith("\n") for r in records),
        "rep4": st.fmean(repeated_4gram_rate(r["gen_ids"]) for r in records),
        "distinct1": distinct_n(records, 1),
        "distinct2": distinct_n(records, 2),
        "eos_frac": st.fmean(r["gen_ids"][-1] == 0 for r in records),
        "mean_sum_logprob": st.fmean(sum(r["logprobs"]) for r in records),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=8, width=0.6)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def save(fig, name):
    os.makedirs(ASSETS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ASSETS, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# per-panel label offsets (points, ha) so direct labels don't collide; None = no label
LABEL_OFFSET = {
    "accuracy": {
        "greedy": (7, -3, "left"), "beam w=2": (7, -3, "left"), "beam w=4": (7, -3, "left"),
        "greedy (0-shot)": (7, -3, "left"),
        "k=5": (3, -12, "left"), "k=20": (7, -3, "left"), "k=50": (7, -3, "left"),
        "k=50, T=0.7": (7, -3, "left"),
        "p=0.5": (-7, -3, "right"), "p=0.9": (-7, -10, "right"), "p=0.95": (0, 7, "center"),
        "p=0.9, T=0.7": (0, 7, "center"),
    },
    "distinct2": {
        "greedy": None, "beam w=2": None, "beam w=4": (12, -9, "left"),
        "greedy (0-shot)": (7, 2, "left"),
        "k=5": (7, -6, "left"), "k=20": (7, -3, "left"), "k=50": (7, -3, "left"),
        "k=50, T=0.7": (7, -3, "left"),
        "p=0.5": (-7, -3, "right"), "p=0.9": (-7, 0, "right"), "p=0.95": (-7, 0, "right"),
        "p=0.9, T=0.7": (-7, 3, "right"),
    },
}


def fig_quality_tradeoff(rows: list[dict]):
    """(a) accuracy vs perplexity, (b) diversity vs perplexity, one point per configuration."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    panels = [("accuracy", "Positional accuracy (%)", 100), ("distinct2", "Distinct-2 (%)", 100)]
    for ax, (key, ylabel, scale) in zip(axes, panels):
        for fam, (label, color, marker) in FAMILY_STYLE.items():
            pts = [r for r in rows if family(r) == fam and r["name"] != "beam_w1"]  # beam w=1 == greedy
            ax.scatter([float(r["ppl_median"]) for r in pts], [scale * float(r[key]) for r in pts],
                       s=42, color=color, marker=marker, edgecolors="white", linewidths=1.2,
                       label=label, zorder=3)
            for r in pts:
                lab = short_label(r)
                off = LABEL_OFFSET[key].get(lab, (7, -3, "left"))
                if off is None:
                    continue
                if key == "distinct2" and lab == "beam w=4":
                    lab = "greedy, beam w=2,4"
                ax.annotate(lab, (float(r["ppl_median"]), scale * float(r[key])), xytext=off[:2],
                            textcoords="offset points", ha=off[2], fontsize=6.5, color=INK_2)
        ax.set_xscale("log")
        ax.set_xticks([1.5, 2, 3, 5, 10, 20])
        ax.set_xlim(1.3, 30)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel("Median perplexity (log scale; lower = more confident)", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        style_axes(ax)
    axes[0].set_title("(a) Accuracy vs. perplexity", fontsize=9.5, color=INK, loc="left")
    axes[1].set_title("(b) Diversity vs. perplexity", fontsize=9.5, color=INK, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    save(fig, "fig_quality_tradeoff")


def fig_beam_cost(rows: list[dict]):
    """(a) wall-clock time and (b) peak GPU memory vs beam width, with least-squares lines."""
    beams = sorted((r for r in rows if r["strategy"] == "beam"), key=lambda r: int(r["num_beams"]))
    widths = [int(r["num_beams"]) for r in beams]
    series = [
        ("total_time_s", "Total decoding time (s)", "(a) Time on 1000 prompts", "s"),
        ("peak_mem_mb", "Peak GPU memory (MiB)", "(b) Peak GPU memory", "MiB"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    fits = {}
    for ax, (key, ylabel, title, unit) in zip(axes, series):
        ys = [float(r[key]) for r in beams]
        mx, my = st.fmean(widths), st.fmean(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(widths, ys)) / sum((x - mx) ** 2 for x in widths)
        icpt = my - slope * mx
        fits[key] = (icpt, slope)
        xs = [0.8, 4.2]
        ax.plot(xs, [icpt + slope * x for x in xs], color=BLUE, linewidth=1.2, linestyle="--", alpha=0.6,
                zorder=2, label="least-squares fit")
        ax.plot(widths, ys, color=BLUE, linewidth=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3, label="measured")
        for x, y in zip(widths, ys):
            ax.annotate(f"{y:.1f} {unit}" if unit == "s" else f"{y:.0f} {unit}", (x, y),
                        xytext=(6, -13), textcoords="offset points", fontsize=7.5, color=INK_2, ha="left")
        ax.text(0.97, 0.06, f"fit: {icpt:.1f} + {slope:.1f}·w {unit}", transform=ax.transAxes,
                ha="right", fontsize=7.5, color=INK_2)
        ax.set_xticks(widths)
        ax.set_xlim(0.6, 4.4)
        ax.set_ylim(0, max(ys) * 1.25)
        ax.set_xlabel("Beam width w", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_beam_cost")
    return fits


def fig_positional_accuracy(curves: dict[str, list[float]]):
    """Per-position token accuracy for representative strategies."""
    lines = [  # name, label, color, linestyle
        ("greedy", "Greedy", BLUE, "-"),
        ("beam_w4", "Beam w=4", BLUE, "--"),
        ("top_p0.9_T1.0", "Top-p 0.9", AQUA, "-"),
        ("top_k50_T0.7", "Top-k 50, T=0.7", ORANGE, "-"),
        ("greedy_0shot", "Greedy, 0-shot", GRAY, "-"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for name, label, color, ls in lines:
        ys = [100 * v for v in curves[name]]
        ax.plot(range(1, len(ys) + 1), ys, color=color, linestyle=ls, linewidth=1.8, label=label,
                marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.6)
    n = len(next(iter(curves.values())))
    ax.set_xlim(0.5, n + 0.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0, 60)
    ax.set_xlabel("Token position in the continuation", fontsize=8.5)
    ax.set_ylabel("Samples matching the reference\nat this position (%)", fontsize=8.5)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7.5, ncol=1, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_positional_accuracy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/part3/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = cfg["output_dir"]

    with open(os.path.join(out, "results.csv")) as f:
        rows = list(csv.DictReader(f))
    records = {}
    for r in rows:
        with open(os.path.join(out, f"{r['name']}.jsonl")) as f:
            records[r["name"]] = [json.loads(line) for line in f]

    extra = []
    for r in rows:
        d = {"name": r["name"], **diagnostics(records[r["name"]])}
        r.update({k: str(v) for k, v in d.items() if k != "name"})
        extra.append(d)
    with open(os.path.join(out, "analysis.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(extra[0]))
        writer.writeheader()
        writer.writerows(extra)

    # beam search vs greedy at the sequence level
    g, b1, b4 = records["greedy"], records["beam_w1"], records["beam_w4"]
    print(f"greedy vs beam_w1: {sum(x['gen_ids'] != y['gen_ids'] for x, y in zip(g, b1))} of {len(g)} differ")
    better = st.fmean(sum(y["logprobs"]) >= sum(x["logprobs"]) - 1e-6 for x, y in zip(g, b4))
    print(f"beam_w4 sequence log-prob >= greedy on {100 * better:.1f}% of prompts")

    fig_quality_tradeoff(rows)
    fits = fig_beam_cost(rows)
    for key, (a, s) in fits.items():
        print(f"beam {key}: {a:.2f} + {s:.2f} * width")
    curves = {n: positional_accuracy_curve(records[n], 20)
              for n in ("greedy", "beam_w4", "top_p0.9_T1.0", "top_k50_T0.7", "greedy_0shot")}
    fig_positional_accuracy(curves)

    print(f"\n{'name':16}" + "".join(f"{k:>12}" for k in extra[0] if k != "name"))
    for d in extra:
        print(f"{d['name']:16}" + "".join(f"{v:12.4f}" for k, v in d.items() if k != "name"))
    print(f"\nwrote {out}/analysis.csv and figures in {ASSETS}/")


if __name__ == "__main__":
    main()
