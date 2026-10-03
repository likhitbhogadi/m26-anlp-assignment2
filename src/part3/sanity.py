"""Correctness checks for the decoders on a few prompts (not part of the experiments).

Usage:
    uv run python -m src.part3.sanity
"""

import argparse

import torch
import torch.nn.functional as F
import yaml

from src.part3 import data, decoding
from src.part3.run import load_model, prepare_samples


def run(model, samples, fn, batch_size, pad_id, device, **kw):
    outs = []
    for batch in data.batches(samples, batch_size):
        ids, mask = data.collate_left_pad([s["prompt_ids"] for s in batch], pad_id, device)
        outs += fn(model, ids, mask, pad_id=pad_id, **kw)
    return outs


@torch.no_grad()
def rescore(model, prompt_ids, gen_ids, device) -> torch.Tensor:
    """log p(gen token) from a single un-padded, un-cached forward over prompt + generation."""
    ids = torch.tensor([prompt_ids + gen_ids], device=device)
    logits = model(ids).logits[0, len(prompt_ids) - 1 : -1].float()
    return F.log_softmax(logits, -1).gather(-1, torch.tensor(gen_ids, device=device)[:, None]).squeeze(-1)


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/part3/config.yaml")
    parser.add_argument("--n", type=int, default=48)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(cfg, device)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    T = cfg["max_new_tokens"]
    common = dict(max_new_tokens=T, eos_id=eos_id)

    # Reference length distribution over the full evaluation set (tokenizer only).
    all_samples = prepare_samples(cfg, tok, cfg["n_samples"])
    ref_lens = torch.tensor([len(s["ref_ids"]) for s in all_samples], dtype=torch.float)
    q = torch.quantile(ref_lens, torch.tensor([0.5, 0.9, 0.99, 1.0]))
    print(f"reference tokens: p50={q[0]:.0f} p90={q[1]:.0f} p99={q[2]:.0f} max={q[3]:.0f}; "
          f"{(ref_lens > T).sum().item()} of {len(ref_lens)} exceed max_new_tokens={T}")
    print(f"eos_id={eos_id} pad_id={pad_id} device={device}\n")

    samples = all_samples[: args.n]
    ids_of = lambda outs: [o["token_ids"] for o in outs]

    greedy = run(model, samples, decoding.greedy_decode, 16, pad_id, device, **common)
    greedy_unbatched = run(model, samples[:16], decoding.greedy_decode, 1, pad_id, device, **common)
    same = sum(a == b for a, b in zip(ids_of(greedy[:16]), ids_of(greedy_unbatched)))
    report("greedy batched == unbatched (left padding / position_ids)", same == 16, f"{same}/16")

    beam1 = run(model, samples, decoding.beam_search, 16, pad_id, device, num_beams=1, **common)
    same = sum(a == b for a, b in zip(ids_of(greedy), ids_of(beam1)))
    report("beam width 1 == greedy", same == len(samples), f"{same}/{len(samples)}")

    topk1 = run(model, samples, decoding.top_k_decode, 16, pad_id, device, k=1, **common)
    same = sum(a == b for a, b in zip(ids_of(greedy), ids_of(topk1)))
    report("top-k k=1 == greedy", same == len(samples), f"{same}/{len(samples)}")

    topp0 = run(model, samples, decoding.top_p_decode, 16, pad_id, device, p=1e-6, **common)
    same = sum(a == b for a, b in zip(ids_of(greedy), ids_of(topp0)))
    report("top-p p=1e-6 == greedy", same == len(samples), f"{same}/{len(samples)}")

    beam2 = run(model, samples[:16], decoding.beam_search, 8, pad_id, device, num_beams=2, **common)
    beam2_unbatched = run(model, samples[:16], decoding.beam_search, 1, pad_id, device, num_beams=2, **common)
    same = sum(a == b for a, b in zip(ids_of(beam2), ids_of(beam2_unbatched)))
    report("beam width 2 batched == unbatched", same == 16, f"{same}/16")

    gen = torch.Generator(device=device).manual_seed(0)
    topp = run(model, samples, decoding.top_p_decode, 16, pad_id, device, p=0.9, generator=gen, **common)
    beam4 = run(model, samples, decoding.beam_search, 16, pad_id, device, num_beams=4, **common)

    # Log-probs recorded in the loop must equal a fresh full forward (checks the KV cache,
    # padding and, for beam search, the cache reordering).
    for name, outs in [("greedy", greedy), ("top_p0.9", topp), ("beam_w4", beam4)]:
        err = max(
            (rescore(model, s["prompt_ids"], o["token_ids"], device) - torch.tensor(o["logprobs"], device=device))
            .abs().max().item()
            for s, o in zip(samples[:16], outs[:16])
        )
        report(f"{name} loop logprobs == full-forward logprobs", err < 1e-3, f"max abs err {err:.2e}")

    mean_sum = lambda outs: sum(sum(o["logprobs"]) for o in outs) / len(outs)
    mean_len = lambda outs: sum(len(o["token_ids"]) for o in outs) / len(outs)
    print(f"\nmean sum-logprob / length:  greedy {mean_sum(greedy):.2f} / {mean_len(greedy):.1f}   "
          f"beam4 {mean_sum(beam4):.2f} / {mean_len(beam4):.1f}   top_p0.9 {mean_sum(topp):.2f} / {mean_len(topp):.1f}")

    s = samples[0]
    print(f"\nPROMPT:    {s['prompt']}\nREFERENCE: {s['reference']}")
    for name, outs in [("greedy", greedy), ("beam_w4", beam4), ("top_p0.9", topp)]:
        print(f"{name.upper():9}: {tok.decode(outs[0]['token_ids'], skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
