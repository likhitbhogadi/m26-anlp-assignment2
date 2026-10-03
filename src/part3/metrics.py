"""Perplexity (self-scored from the decoding loop) and positional token accuracy."""

import math
import statistics


def sample_perplexity(logprobs: list[float]) -> float:
    """exp of the mean negative log-likelihood of the generated tokens.

    `logprobs` are log p_model(chosen token) under the model's unmodified softmax,
    recorded inside the decoding loop (no temperature / truncation applied)."""
    if not logprobs:
        return float("nan")
    return math.exp(-sum(logprobs) / len(logprobs))


def positional_accuracy(gen_ids: list[int], ref_ids: list[int]) -> float:
    """Fraction of reference positions i with gen[i] == ref[i].

    Missing positions (generation shorter than reference) count as wrong;
    generated tokens beyond the reference length are ignored."""
    matches = sum(g == r for g, r in zip(gen_ids, ref_ids))
    return matches / len(ref_ids)


def aggregate(records: list[dict]) -> dict:
    ppls = [r["ppl"] for r in records if not math.isnan(r["ppl"])]
    all_lp = [lp for r in records for lp in r["logprobs"]]
    return {
        "ppl_mean": statistics.fmean(ppls),
        "ppl_median": statistics.median(ppls),
        "ppl_corpus": math.exp(-sum(all_lp) / len(all_lp)),
        "accuracy": statistics.fmean(r["acc"] for r in records),
        "avg_gen_tokens": statistics.fmean(len(r["gen_ids"]) for r in records),
    }
