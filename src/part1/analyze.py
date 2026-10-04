"""Plots + markdown tables for the Q1 report, built only from the CSVs in output_dir.

Usage:
    uv run python -m src.part1.analyze
    uv run python -m src.part1.analyze --runs sweep_dense_lr3e-4,sweep_dense_lr6e-4 --out-name lr_sweep_dense.png

Reads train_log_* / val_log_* / train_routing_* (train.py), results.csv / routing.csv
(evaluate.py); writes <assets_dir>/*.png and <assets_dir>/tables.md. Variants that have no
CSVs yet are skipped, so this can be re-run at any point.
"""

import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.part1.common import LANGS, load_config

# Categorical slots in fixed order (one per variant, same color in every figure).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LANG_LABEL = {"vi": "vi→en", "ja": "ja→en", "all": "both"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2,
})


def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows])


def smooth(x: np.ndarray, w: int = 5) -> np.ndarray:
    if len(x) < w:
        return x
    return np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")


def legend_below(fig, ax, ncol: int = 3):
    """One shared legend under the panels, so it never covers the data."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=ncol)


def save(fig, assets: str, name: str):
    path = os.path.join(assets, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------- training curves

def plot_curves(out: str, assets: str, runs: list[str], colors: dict, fname: str, baseline: str | None = None):
    """Train / val CE vs tokens; with `baseline`, a third panel shows val CE minus the
    baseline's at the same step (the curves themselves overlap almost completely)."""
    base = read_csv(os.path.join(out, f"val_log_{baseline}.csv")) if baseline else []
    fig, axes = plt.subplots(1, 3 if base else 2, figsize=(15 if base else 10, 3.6))
    for run in runs:
        tr, va = read_csv(os.path.join(out, f"train_log_{run}.csv")), read_csv(os.path.join(out, f"val_log_{run}.csv"))
        if tr:
            axes[0].plot(col(tr, "tokens_seen") / 1e6, smooth(col(tr, "train_ce")), color=colors[run], label=run)
        if va:
            axes[1].plot(col(va, "tokens_seen") / 1e6, col(va, "val_ce"), color=colors[run], label=run,
                         marker="o", markersize=3)
            if base and len(va) == len(base):
                skip = 2  # first 1k steps: everything is still ~6+ nats and the gaps are noise
                axes[2].plot(col(va, "tokens_seen")[skip:] / 1e6, (col(va, "val_ce") - col(base, "val_ce"))[skip:],
                             color=colors[run], marker="o", markersize=3)
    axes[0].set(title="Train target CE (smoothed)", xlabel="training tokens (M)", ylabel="cross-entropy (nats)")
    axes[1].set(title="Validation target CE (2k pairs)", xlabel="training tokens (M)", ylabel="cross-entropy (nats)")
    if base:
        axes[2].axhline(0, color=INK_2, linewidth=0.8)
        axes[2].set(title=f"Validation CE − {baseline} (same step)", xlabel="training tokens (M)",
                    ylabel="Δ cross-entropy (nats)")
    for ax in axes[:2]:  # cut the first ~10% (initial drop from ~10 nats) so the end of training is readable
        ys = [l.get_ydata() for l in ax.get_lines()]
        if ys:
            lo, hi = min(y.min() for y in ys), max(np.percentile(y, 85) for y in ys)
            ax.set_ylim(lo - 0.05 * (hi - lo), hi)
    legend_below(fig, axes[1])
    save(fig, assets, fname)


def plot_aux(out: str, assets: str, variants: list[str], colors: dict):
    moe = [v for v in variants if read_csv(os.path.join(out, f"train_routing_{v}.csv"))]
    if not moe:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for v in moe:
        tr = read_csv(os.path.join(out, f"train_log_{v}.csv"))
        axes[0].plot(col(tr, "tokens_seen") / 1e6, smooth(col(tr, "aux_lb")), color=colors[v], label=v)
        rt = read_csv(os.path.join(out, f"train_routing_{v}.csv"))
        by_step = defaultdict(list)
        for r in rt:
            by_step[int(r["step"])].append(float(r["load_frac"]))
        steps = sorted(by_step)
        n_exp = max(int(r["expert"]) for r in rt) + 1
        # max expert load / uniform share, averaged over layers (1 = perfectly balanced)
        imbalance = [np.mean(np.max(np.array(by_step[s]).reshape(-1, n_exp), axis=1)) * n_exp for s in steps]
        tok = {int(r["step"]): float(r["tokens_seen"]) for r in tr}
        axes[1].plot([tok[s] / 1e6 for s in steps], smooth(np.array(imbalance)), color=colors[v], label=v)
    axes[0].set(title="Load-balancing loss (1.0 = uniform)", xlabel="training tokens (M)", ylabel="E · Σ fᵢ Pᵢ, mean over layers")
    axes[1].set(title="Busiest expert's load ÷ uniform share", xlabel="training tokens (M)", ylabel="max load × E, mean over layers")
    legend_below(fig, axes[1], ncol=4)
    save(fig, assets, "moe_aux_and_balance.png")


# ---------------------------------------------------------------- test metrics

def plot_test(results: list[dict], assets: str, variants: list[str], colors: dict):
    got = [v for v in variants if any(r["variant"] == v for r in results)]
    if not got:
        return
    get = {(r["variant"], r["lang"]): r for r in results}
    for metric, title, fname in [("test_ppl", "Test perplexity (target tokens, lower is better)", "test_ppl.png"),
                                 ("bleu", "Test BLEU (greedy, sacrebleu 13a, higher is better)", "test_bleu.png")]:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
        for ax, lang in zip(axes, [*LANGS, "all"]):
            vals = [float(get[(v, lang)][metric]) for v in got]
            bars = ax.bar(range(len(got)), vals, color=[colors[v] for v in got], width=0.7, edgecolor="white", linewidth=2)
            for b, val in zip(bars, vals):
                ax.annotate(f"{val:.2f}", (b.get_x() + b.get_width() / 2, val), ha="center", va="bottom",
                            fontsize=8, color=INK, xytext=(0, 2), textcoords="offset points")
            ax.set_xticks(range(len(got)), got, rotation=30, ha="right")
            ax.set_title(LANG_LABEL[lang])
            ax.grid(axis="x", visible=False)
        fig.suptitle(title, color=INK, fontsize=10)
        save(fig, assets, fname)


def plot_params(results: list[dict], assets: str, variants: list[str], colors: dict):
    rows = {r["variant"]: r for r in results if r["lang"] == "all" and r["variant"] in variants}
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, metric, ylabel in [(axes[0], "test_ppl", "test perplexity (both languages)"),
                               (axes[1], "bleu", "test BLEU (both languages)")]:
        for v, r in rows.items():
            x, y = float(r["active_params"]) / 1e6, float(r[metric])
            filled = "dense" not in rows or math.isclose(float(r["total_params"]), float(rows["dense"]["total_params"]), rel_tol=0.01)
            ax.scatter(x, y, s=70, color=colors[v] if filled else "white", edgecolor=colors[v], linewidth=2, zorder=3)
            ax.annotate(v, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
        ax.set(xlabel="active parameters per token (M)", ylabel=ylabel)
        ax.margins(x=0.25, y=0.15)
    fig.suptitle("Quality vs active parameters  ·  filled: same total params as dense (35.3M)  ·  "
                 "hollow: 2× FFN params (47.9M total)", color=INK, fontsize=10)
    save(fig, assets, "params_vs_quality.png")


# ---------------------------------------------------------------- routing

def routing_tensor(routing: list[dict], variant: str):
    """frac[layer, lang, role, expert] (each (layer, lang, role) row sums to 1)."""
    rows = [r for r in routing if r["variant"] == variant]
    if not rows:
        return None
    L = max(int(r["layer"]) for r in rows) + 1
    E = max(int(r["expert"]) for r in rows) + 1
    counts = np.zeros((L, len(LANGS), 2, E))
    for r in rows:
        counts[int(r["layer"]), LANGS.index(r["src_lang"]), ["src", "tgt"].index(r["token_role"]), int(r["expert"])] = float(r["count"])
    return counts


def _heat(ax, m, cmap, vmin, vmax, title, fmt="{:.2f}"):
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for (i, j), val in np.ndenumerate(m):
        dark = abs(val - vmin) / max(vmax - vmin, 1e-9) > 0.6 if cmap != "RdBu_r" else abs(val) > 0.6 * vmax
        ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=7, color="white" if dark else INK)
    ax.set_xticks(range(m.shape[1]), [f"E{e}" for e in range(m.shape[1])])
    ax.set_yticks(range(m.shape[0]), [f"L{l}" for l in range(m.shape[0])])
    ax.grid(False)
    ax.set_title(title, fontsize=9)
    return im


def plot_routing_heatmaps(routing: list[dict], assets: str, variant: str, shared: bool):
    counts = routing_tensor(routing, variant)
    if counts is None:
        return
    frac = counts / counts.sum(-1, keepdims=True)
    E = frac.shape[-1]
    vmax_d = max(0.05, np.abs(frac[:, 0] - frac[:, 1]).max())
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), constrained_layout=True)
    for r, role in enumerate(["source tokens (vi / ja)", "target-side tokens (en)"]):
        for c, lang in enumerate(LANGS):
            im = _heat(axes[r, c], frac[:, c, r], "Blues", 0, max(0.5, frac.max()), f"{LANG_LABEL[lang]} — {role}")
        imd = _heat(axes[r, 2], frac[:, 0, r] - frac[:, 1, r], "RdBu_r", -vmax_d, vmax_d, f"vi − ja — {role}", "{:+.2f}")
    fig.colorbar(im, ax=axes[:, :2], shrink=0.6, label="share of routed assignments")
    fig.colorbar(imd, ax=axes[:, 2], shrink=0.6, label="vi share − ja share")
    note = " (routed experts only; the shared expert sees every token)" if shared else ""
    fig.suptitle(f"{variant}: expert usage per layer by source language{note}  ·  uniform = {1 / E:.2f}",
                 color=INK, fontsize=10)
    save(fig, assets, f"routing_heatmap_{variant}.png")


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon divergence in bits (0 = identical, 1 = disjoint)."""
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log2(np.maximum(a, 1e-12) / np.maximum(b, 1e-12)), 0)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def routing_summary(routing: list[dict], variants: list[str]) -> dict:
    """Per variant: per-layer load CV / max-load ratio / normalised entropy (all test tokens)
    and vi-vs-ja JS divergence for source and target-side tokens."""
    out = {}
    for v in variants:
        counts = routing_tensor(routing, v)
        if counts is None:
            continue
        E = counts.shape[-1]
        load = counts.sum((1, 2))
        load = load / load.sum(-1, keepdims=True)                          # (L, E)
        frac = counts / counts.sum(-1, keepdims=True)
        out[v] = {
            "cv": load.std(-1) / load.mean(-1),
            "max_ratio": load.max(-1) * E,
            "entropy": -(load * np.log(np.maximum(load, 1e-12))).sum(-1) / np.log(E),
            "js_src": np.array([js_divergence(frac[l, 0, 0], frac[l, 1, 0]) for l in range(len(load))]),
            "js_tgt": np.array([js_divergence(frac[l, 0, 1], frac[l, 1, 1]) for l in range(len(load))]),
            "load": load,
        }
    return out


def plot_routing_summary(summary: dict, assets: str, colors: dict):
    if not summary:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for v, s in summary.items():
        layers = np.arange(len(s["cv"]))
        axes[0].plot(layers, s["cv"], color=colors[v], marker="o", markersize=4, label=v)
        axes[1].plot(layers, s["js_src"], color=colors[v], marker="o", markersize=4, label=v)
        axes[2].plot(layers, s["js_tgt"], color=colors[v], marker="o", markersize=4, label=v)
    axes[0].set(title="Load imbalance (CV of expert load, 0 = uniform)", xlabel="layer", ylabel="coefficient of variation")
    axes[1].set(title="Language specialisation: source tokens", xlabel="layer", ylabel="JS divergence vi vs ja (bits)")
    axes[2].set(title="Language specialisation: target-side tokens", xlabel="layer", ylabel="JS divergence vi vs ja (bits)")
    for ax in axes:
        ax.set_xticks(layers)
        ax.set_ylim(bottom=0)
    legend_below(fig, axes[0], ncol=4)
    save(fig, assets, "routing_summary.png")


# ---------------------------------------------------------------- tables

def fmt_m(x) -> str:
    return f"{float(x) / 1e6:.2f}M"


def write_tables(cfg: dict, results: list[dict], summary: dict, variants: list[str], assets: str):
    get = {(r["variant"], r["lang"]): r for r in results}
    lines = ["## Parameters", "", "| variant | total | active | FFN total | FFN active | non-emb total | non-emb active |",
             "|---|---|---|---|---|---|---|"]
    for v in variants:
        if (v, "all") in get:
            r = get[(v, "all")]
            lines.append(f"| {v} | {fmt_m(r['total_params'])} | {fmt_m(r['active_params'])} | {fmt_m(r['ffn_params'])} | "
                         f"{fmt_m(r['active_ffn_params'])} | {fmt_m(r['non_embedding_params'])} | {fmt_m(r['active_non_embedding_params'])} |")
    lines += ["", "## Test results", "",
              "| variant | PPL vi→en | PPL ja→en | PPL both | BLEU vi→en | BLEU ja→en | BLEU both |", "|---|---|---|---|---|---|---|"]
    for v in variants:
        if (v, "all") in get:
            p = [float(get[(v, l)]["test_ppl"]) for l in [*LANGS, "all"]]
            b = [float(get[(v, l)]["bleu"]) for l in [*LANGS, "all"]]
            lines.append(f"| {v} | " + " | ".join(f"{x:.2f}" for x in p) + " | " + " | ".join(f"{x:.2f}" for x in b) + " |")
    if summary:
        lines += ["", "## Expert usage (all test tokens; mean over layers, [min–max] across layers)", "",
                  "| variant | load CV | busiest ÷ uniform | normalised entropy | JS vi/ja src | JS vi/ja tgt |",
                  "|---|---|---|---|---|---|"]
        for v, s in summary.items():
            cell = lambda a: f"{a.mean():.3f} [{a.min():.3f}–{a.max():.3f}]"
            lines.append(f"| {v} | {cell(s['cv'])} | {cell(s['max_ratio'])} | {cell(s['entropy'])} | "
                         f"{cell(s['js_src'])} | {cell(s['js_tgt'])} |")
    lines += token_table(read_csv(os.path.join(cfg["output_dir"], "routing_tokens.csv")), variants)
    lines += sample_table(cfg["output_dir"], variants)
    with open(os.path.join(assets, "tables.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(assets, 'tables.md')}")


def _md_tok(t: str) -> str:
    return "`" + t.replace("`", "'").replace("|", "\\|") + "`"


def token_table(rows: list[dict], variants: list[str], n_show: int = 8) -> list[str]:
    """Per MoE variant, first and last layer: each expert's most 'loyal' frequent tokens."""
    if not rows:
        return []
    lines = ["", "## Tokens most consistently routed to each expert (test set, tokens with ≥200 occurrences)", "",
             "purity = share of the token's occurrences routed to that expert "
             "(uniform routing: top_k / n_experts).", ""]
    for v in variants:
        vr = [r for r in rows if r["variant"] == v]
        if not vr:
            continue
        last = max(int(r["layer"]) for r in vr)
        lines += [f"**{v}**", "", "| layer | expert | top tokens (purity) |", "|---|---|---|"]
        for layer in (0, last):
            for e in sorted({int(r["expert"]) for r in vr}):
                top = sorted((r for r in vr if int(r["layer"]) == layer and int(r["expert"]) == e),
                             key=lambda r: int(r["rank"]))[:n_show]
                lines.append(f"| L{layer} | E{e} | " + ", ".join(f"{_md_tok(r['token'])} ({float(r['purity']):.2f})" for r in top) + " |")
        lines.append("")
    return lines


def sample_table(out: str, variants: list[str], per_lang: int = 4, seed: int = 0) -> list[str]:
    """Same randomly chosen short test sentences translated by every variant. Also written to
    translation_samples.csv (small; the full translation CSVs live on the HF Hub)."""
    trans = {v: read_csv(os.path.join(out, f"translations_{v}.csv")) for v in variants}
    trans = {v: t for v, t in trans.items() if t}
    if not trans:
        return []
    first = next(iter(trans.values()))
    rng = np.random.default_rng(seed)
    picked = []
    for lang in LANGS:
        cand = [r for r in first if r["lang"] == lang and 15 <= len(r["reference"]) <= 90]
        picked += [(cand[i]["idx"], lang) for i in rng.choice(len(cand), per_lang, replace=False)]
    keyed = {v: {(r["idx"], r["lang"]): r for r in t} for v, t in trans.items()}
    rows, lines = [], ["", "## Sample translations (same random short test sentences for every variant)", ""]
    for idx, lang in picked:
        ref = keyed[next(iter(keyed))][(idx, lang)]
        lines += [f"**[{LANG_LABEL[lang]}, test #{idx}]** {ref['source']}  ", f"*reference:* {ref['reference']}", "",
                  "| variant | hypothesis | sentence BLEU |", "|---|---|---|"]
        for v, k in keyed.items():
            r = k[(idx, lang)]
            rows.append(r)
            lines.append(f"| {v} | {r['hypothesis'].replace('|', '/')} | {float(r['sentence_bleu']):.1f} |")
        lines.append("")
    with open(os.path.join(out, "translation_samples.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/part1/config.yaml")
    parser.add_argument("--runs", default=None, help="comma-separated run names: only plot their training curves")
    parser.add_argument("--out-name", default="lr_sweep_curves.png")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out, assets = cfg["output_dir"], cfg["assets_dir"]
    os.makedirs(assets, exist_ok=True)

    if args.runs:  # e.g. the LR sweep
        runs = args.runs.split(",")
        plot_curves(out, assets, runs, {r: SERIES[i % len(SERIES)] for i, r in enumerate(runs)}, args.out_name)
        return

    variants = [v["name"] for v in cfg["variants"]]
    colors = {v: SERIES[i] for i, v in enumerate(variants)}
    shared = {v["name"]: v.get("n_shared", 0) > 0 for v in cfg["variants"]}
    results, routing = read_csv(os.path.join(out, "results.csv")), read_csv(os.path.join(out, "routing.csv"))

    plot_curves(out, assets, [v for v in variants if os.path.exists(os.path.join(out, f"train_log_{v}.csv"))],
                colors, "loss_curves.png", baseline="dense")
    plot_aux(out, assets, variants, colors)
    plot_test(results, assets, variants, colors)
    plot_params(results, assets, variants, colors)
    for v in variants:
        plot_routing_heatmaps(routing, assets, v, shared[v])
    summary = routing_summary(routing, variants)
    plot_routing_summary(summary, assets, colors)
    write_tables(cfg, results, summary, variants, assets)


if __name__ == "__main__":
    main()
