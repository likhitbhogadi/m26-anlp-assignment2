"""Plots and tables for Q2 from the CSVs written by train.py.

Usage:
    uv run python -m src.part2.analyze

Runs named `final_<opt>` are the full 1x runs; `probe_<opt>_lr<lr>` are the LR probes.
Writes `<assets_dir>/*.png` and `<output_dir>/summary.md` (markdown tables for the report).
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.part1.common import load_config
from src.part2.optimizers import NAMES

LABEL = {"adamw": "AdamW", "mars": "MARS", "lion": "Lion", "muon": "Muon"}
COLOR = {"adamw": "#2a78d6", "mars": "#eb6834", "lion": "#1baf7a", "muon": "#eda100"}  # fixed slot order
MARKER = {"adamw": "o", "mars": "s", "lion": "^", "muon": "D"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
})


def read_csv(path: str) -> list[dict]:
    with open(path) as f:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def tokens_to_reach(rows: list[dict], target: float) -> float:
    """First token count at which val loss <= target (linear interpolation between evals); nan if never."""
    t, l = col(rows, "tokens_seen"), col(rows, "val_loss")
    for i in range(1, len(t)):
        if l[i] <= target:
            return float(np.interp(target, [l[i], l[i - 1]], [t[i], t[i - 1]]))
    return float("nan")


def save(fig, assets, name):
    path = os.path.join(assets, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/part2/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out, assets = cfg["output_dir"], cfg["assets_dir"]
    os.makedirs(assets, exist_ok=True)
    results = {r["run"]: r for r in read_csv(os.path.join(out, "results.csv"))}
    md = []

    # ------------------------------------------------ LR probe
    probes = {o: sorted([r for r in results.values() if str(r["run"]).startswith(f"probe_{o}_")], key=lambda r: r["lr"])
              for o in NAMES}
    if any(probes.values()):
        fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
        md += ["## LR probe (500 steps ≈ 0.2× dataset, final val loss)", "",
               "| optimizer | peak LR | val loss | val loss (human) | val loss (LLM) |", "|---|---|---|---|---|"]
        for ax, o in zip(axes, NAMES):
            rs = probes[o]
            if not rs:
                continue
            lrs, losses = col(rs, "lr"), col(rs, "val_loss")
            ax.plot(lrs, losses, color=COLOR[o], marker=MARKER[o], markersize=7)
            best = int(np.argmin(losses))
            ax.plot(lrs[best], losses[best], marker=MARKER[o], markersize=11, color=COLOR[o], markeredgecolor=INK)
            ax.annotate(f"{losses[best]:.3f}", (lrs[best], losses[best]), textcoords="offset points", xytext=(0, 8),
                        ha="center", color=INK)
            ax.set_xscale("log")
            ax.set_xticks(lrs, [f"{x:g}" for x in lrs])
            ax.minorticks_off()
            ax.set_title(LABEL[o], color=INK)
            ax.set_xlabel("peak LR")
            for r in rs:
                md.append(f"| {LABEL[o]} | {r['lr']:g} | {r['val_loss']:.4f} | {r['val_loss_human']:.4f} | "
                          f"{r['val_loss_llm']:.4f} |")
        axes[0].set_ylabel("val loss after probe")
        fig.suptitle("LR probe: 500 steps (≈0.2× dataset), warmup + cosine compressed to the probe", color=INK, y=1.06)
        save(fig, assets, "lr_probe.png")

        fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
        for ax, o in zip(axes, NAMES):
            for r, alpha in zip(probes[o], np.linspace(0.3, 1.0, len(probes[o]))):
                p = os.path.join(out, f"train_log_{r['run']}.csv")
                if os.path.exists(p):
                    tl = read_csv(p)
                    ax.plot(col(tl, "tokens_seen") / 1e6, col(tl, "train_loss"), color=COLOR[o], alpha=alpha,
                            linewidth=1.5, label=f"lr {r['lr']:g}")
            ax.set_title(LABEL[o], color=INK)
            ax.set_xlabel("tokens (M)")
            ax.set_ylim(3, 8)
            ax.legend(fontsize=7)
        axes[0].set_ylabel("train loss")
        save(fig, assets, "lr_probe_curves.png")
        md.append("")

    # ------------------------------------------------ final runs
    finals = [o for o in NAMES if os.path.exists(os.path.join(out, f"eval_log_final_{o}.csv"))]
    if not finals:
        print("no final runs yet")
    else:
        ev = {o: read_csv(os.path.join(out, f"eval_log_final_{o}.csv")) for o in finals}
        tr = {o: read_csv(os.path.join(out, f"train_log_final_{o}.csv")) for o in finals}
        res = {o: results[f"final_{o}"] for o in finals}
        dataset_tokens = res[finals[0]]["tokens_seen"] / res[finals[0]]["dataset_fraction"]
        chin = dataset_tokens / (20 * res[finals[0]]["total_params"])
        xlab = f"fraction of the training set (1× = {dataset_tokens / 1e6:.1f}M tokens ≈ {chin:.3f}× Chinchilla)"

        # val loss vs tokens (step-0 point dropped: loss ~10.4 would flatten the curves)
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for o in finals:
            rows = [r for r in ev[o] if r["step"] > 0]
            ax.plot(col(rows, "dataset_fraction"), col(rows, "val_loss"), color=COLOR[o], marker=MARKER[o],
                    markersize=5, label=LABEL[o])
            ax.annotate(LABEL[o], (rows[-1]["dataset_fraction"], rows[-1]["val_loss"]), textcoords="offset points",
                        xytext=(6, 0), va="center", color=INK, fontsize=8)
        ax.set_xticks(np.arange(0.1, 1.01, 0.1))
        ax.set_xlabel(xlab)
        ax.set_ylabel("validation loss (nats/token)")
        ax.set_title("Validation loss vs tokens", color=INK, loc="left")
        ax.legend()
        save(fig, assets, "val_loss_vs_tokens.png")

        # BLEU vs tokens: multi-reference and human-only reference
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, key, title in [(axes[0], "bleu", "Test BLEU, 7 references (human + 6 LLM continuations)"),
                               (axes[1], "bleu_human", "Test BLEU, human continuation only")]:
            for o in finals:
                ax.plot(col(ev[o], "dataset_fraction"), col(ev[o], key), color=COLOR[o], marker=MARKER[o],
                        markersize=5, label=LABEL[o])
            ax.set_xticks(np.arange(0, 1.01, 0.1))
            ax.set_xlabel("fraction of the training set")
            ax.set_ylabel("BLEU")
            ax.set_title(title, color=INK, loc="left", fontsize=9)
        axes[0].legend()
        save(fig, assets, "bleu_vs_tokens.png")

        # train loss
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for o in finals:
            ax.plot(col(tr[o], "dataset_fraction"), col(tr[o], "train_loss"), color=COLOR[o], linewidth=1.5,
                    label=LABEL[o])
        ax.set_ylim(None, 6)
        ax.set_xlabel("fraction of the training set")
        ax.set_ylabel(f"train loss (mean over {cfg['train']['log_every']} steps)")
        ax.set_title("Training loss", color=INK, loc="left")
        ax.legend()
        save(fig, assets, "train_loss.png")

        # human vs LLM val loss at 1x
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        w = 0.38
        x = np.arange(len(finals))
        for k, (key, name, hatch) in enumerate([("val_loss_human", "human-written", ""), ("val_loss_llm", "LLM-written", "//")]):
            vals = [ev[o][-1][key] for o in finals]
            ax.bar(x + (k - 0.5) * w, vals, w * 0.95, color=[COLOR[o] for o in finals], hatch=hatch,
                   edgecolor="white", label=name)
            for xi, v in zip(x, vals):
                ax.text(xi + (k - 0.5) * w, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=INK)
        ax.set_xticks(x, [LABEL[o] for o in finals])
        ax.set_ylabel("val loss at 1×")
        ax.set_ylim(min(ev[o][-1]["val_loss_llm"] for o in finals) - 0.6, None)
        ax.legend()
        ax.set_title("Human- vs LLM-written validation text", color=INK, loc="left")
        save(fig, assets, "val_loss_by_source.png")

        # tables
        base = "adamw" if "adamw" in finals else finals[0]
        target = res[base]["val_loss"]
        md += [f"## Final runs (1× = {dataset_tokens / 1e6:.2f}M tokens ≈ {chin:.4f}× Chinchilla)", "",
               "| optimizer | peak LR | val loss | test loss | test PPL | BLEU (7 refs) | BLEU (human ref) | "
               f"tokens to reach {LABEL[base]} final loss | speedup vs {LABEL[base]} | ms/step | optimizer state (MiB) |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for o in finals:
            r = res[o]
            reach = tokens_to_reach(ev[o], target)
            speedup = "—" if np.isnan(reach) else f"{res[base]['tokens_seen'] / reach:.2f}×"
            md.append(f"| {LABEL[o]} | {r['lr']:g} | {r['val_loss']:.4f} | {r['test_loss']:.4f} | {r['test_ppl']:.2f} | "
                      f"{r['bleu']:.2f} | {r['bleu_human']:.2f} | "
                      + ("—" if np.isnan(reach) else f"{reach / 1e6:.1f}M ({reach / dataset_tokens:.2f}×)")
                      + f" | {speedup} | {r['ms_per_step']:.0f} | {r['opt_state_mib']:.0f} |")
        md += ["", "### Validation loss / BLEU (7 refs) at every 0.1× of the dataset", "",
               "| fraction | " + " | ".join(f"{LABEL[o]} loss | {LABEL[o]} BLEU" for o in finals) + " |",
               "|---|" + "---|---|" * len(finals)]
        for i in range(len(ev[finals[0]])):
            md.append(f"| {ev[finals[0]][i]['dataset_fraction']:.1f} | " + " | ".join(
                f"{ev[o][i]['val_loss']:.4f} | {ev[o][i]['bleu']:.2f}" for o in finals) + " |")
        genres = [k[len("val_loss_"):] for k in ev[finals[0]][-1] if k.startswith("val_loss_") and
                  k not in ("val_loss_human", "val_loss_llm")]
        md += ["", "### Val loss at 1× by source and genre", "",
               "| optimizer | human | LLM | " + " | ".join(genres) + " |", "|---|---|---|" + "---|" * len(genres)]
        for o in finals:
            e = ev[o][-1]
            md.append(f"| {LABEL[o]} | {e['val_loss_human']:.4f} | {e['val_loss_llm']:.4f} | "
                      + " | ".join(f"{e['val_loss_' + g]:.4f}" for g in genres) + " |")

    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
