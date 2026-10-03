"""Decoding strategies built directly on `model.forward()` (no `model.generate()`).

All strategies run batched with left padding and a KV cache: the prompt is fed once
(prefill), after which only the newly chosen token is fed at every step.

Every strategy returns, per prompt, the generated token ids and, for each generated
token, log p_model(token | prefix) under the model's *unmodified* softmax. These
log-probs are what the self-scored perplexity is computed from, so temperature and
top-k / top-p truncation change which token is picked but not how it is scored.
"""

from typing import Callable

import torch
import torch.nn.functional as F


@torch.no_grad()
def _forward(model, input_ids, attention_mask, past_key_values):
    """One forward pass; returns next-token logits (float32) and the updated cache.

    position_ids are derived from the attention mask so that left padding does not
    shift the positions of the real tokens."""
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)[:, -input_ids.shape[1] :]
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    return out.logits[:, -1, :].float(), out.past_key_values


def _reorder_cache(past_key_values, rows: torch.Tensor):
    """Selects batch rows of the KV cache (used by beam search to follow surviving beams)."""
    if hasattr(past_key_values, "layers"):  # transformers >= 4.56 Cache objects
        for layer in past_key_values.layers:
            layer.keys = layer.keys.index_select(0, rows)
            layer.values = layer.values.index_select(0, rows)
        return past_key_values
    if hasattr(past_key_values, "key_cache"):  # older DynamicCache
        for i in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[i] = past_key_values.key_cache[i].index_select(0, rows)
            past_key_values.value_cache[i] = past_key_values.value_cache[i].index_select(0, rows)
        return past_key_values
    return tuple(tuple(t.index_select(0, rows) for t in layer) for layer in past_key_values)


# ---------------------------------------------------------------------------
# Token selection rules for greedy / top-k / top-p
# ---------------------------------------------------------------------------

def greedy_select(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def make_top_k_select(k: int, temperature: float = 1.0, generator=None) -> Callable:
    def select(logits: torch.Tensor) -> torch.Tensor:
        top_logits, top_ids = (logits / temperature).topk(k, dim=-1)
        probs = F.softmax(top_logits, dim=-1)
        choice = torch.multinomial(probs, 1, generator=generator)
        return top_ids.gather(-1, choice).squeeze(-1)

    return select


def make_top_p_select(p: float, temperature: float = 1.0, generator=None) -> Callable:
    def select(logits: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_ids = (logits / temperature).sort(dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        # Keep the smallest prefix whose mass reaches p: drop a token if the mass
        # *before* it already exceeds p. The top token is therefore always kept.
        mass_before = probs.cumsum(dim=-1) - probs
        probs = probs.masked_fill(mass_before > p, 0.0)
        choice = torch.multinomial(probs, 1, generator=generator)  # renormalises internally
        return sorted_ids.gather(-1, choice).squeeze(-1)

    return select


@torch.no_grad()
def sample_loop(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    select_fn: Callable,
    max_new_tokens: int,
    eos_id: int,
    pad_id: int,
) -> list[dict]:
    """Autoregressive loop shared by greedy, top-k and top-p (they differ only in `select_fn`)."""
    batch = input_ids.shape[0]
    device = input_ids.device
    logits, past = _forward(model, input_ids, attention_mask, None)

    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    step_tokens, step_logprobs, step_valid = [], [], []
    for step in range(max_new_tokens):
        raw_logprobs = F.log_softmax(logits, dim=-1)
        next_tok = select_fn(logits)
        next_tok = torch.where(finished, torch.full_like(next_tok, pad_id), next_tok)

        step_tokens.append(next_tok)
        step_logprobs.append(raw_logprobs.gather(-1, next_tok[:, None]).squeeze(-1))
        step_valid.append(~finished)

        finished = finished | (next_tok == eos_id)
        if finished.all() or step == max_new_tokens - 1:
            break
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(batch, 1)], dim=-1)
        logits, past = _forward(model, next_tok[:, None], attention_mask, past)

    tokens = torch.stack(step_tokens, dim=1).tolist()
    logprobs = torch.stack(step_logprobs, dim=1).tolist()
    valid = torch.stack(step_valid, dim=1).tolist()
    return [
        {
            "token_ids": [t for t, v in zip(tok_row, v_row) if v],
            "logprobs": [lp for lp, v in zip(lp_row, v_row) if v],
        }
        for tok_row, lp_row, v_row in zip(tokens, logprobs, valid)
    ]


def greedy_decode(model, input_ids, attention_mask, max_new_tokens, eos_id, pad_id):
    return sample_loop(model, input_ids, attention_mask, greedy_select, max_new_tokens, eos_id, pad_id)


def top_k_decode(model, input_ids, attention_mask, max_new_tokens, eos_id, pad_id,
                 k, temperature=1.0, generator=None):
    select = make_top_k_select(k, temperature, generator)
    return sample_loop(model, input_ids, attention_mask, select, max_new_tokens, eos_id, pad_id)


def top_p_decode(model, input_ids, attention_mask, max_new_tokens, eos_id, pad_id,
                 p, temperature=1.0, generator=None):
    select = make_top_p_select(p, temperature, generator)
    return sample_loop(model, input_ids, attention_mask, select, max_new_tokens, eos_id, pad_id)


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_search(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_beams: int,
    max_new_tokens: int,
    eos_id: int,
    pad_id: int,
    length_penalty: float = 1.0,
) -> list[dict]:
    """Batched beam search. Each prompt keeps `num_beams` live hypotheses, stored as
    rows n*K .. n*K+K-1 of a (N*K)-row batch, so one forward pass advances every beam
    of every prompt. Hypotheses that emit EOS move to a per-prompt finished pool scored
    by sum(log p) / length**length_penalty."""
    N, K = input_ids.shape[0], num_beams
    device = input_ids.device

    input_ids = input_ids.repeat_interleave(K, dim=0)
    attention_mask = attention_mask.repeat_interleave(K, dim=0)
    logits, past = _forward(model, input_ids, attention_mask, None)

    # Only beam 0 is live initially; otherwise the K identical copies of the prompt
    # would all propose the same first tokens.
    beam_scores = torch.zeros(N, K, device=device)
    beam_scores[:, 1:] = float("-inf")
    beam_scores = beam_scores.view(-1)

    seqs = [[] for _ in range(N * K)]       # generated token ids per beam row
    seq_logprobs = [[] for _ in range(N * K)]
    finished = [[] for _ in range(N)]       # (normalised score, ids, logprobs)
    done = [False] * N

    for step in range(max_new_tokens):
        cur_len = step + 1
        logprobs = F.log_softmax(logits, dim=-1)                       # (N*K, V)
        vocab = logprobs.shape[-1]
        cand_scores = (beam_scores[:, None] + logprobs).view(N, K * vocab)
        # 2K candidates guarantee at least K non-EOS continuations (each beam has one EOS).
        top_scores, top_idx = cand_scores.topk(2 * K, dim=-1)
        top_logprobs = logprobs.view(N, K * vocab).gather(-1, top_idx)
        top_beam, top_tok = top_idx // vocab, top_idx % vocab

        top_scores, top_logprobs = top_scores.tolist(), top_logprobs.tolist()
        top_beam, top_tok = top_beam.tolist(), top_tok.tolist()

        new_beams = []  # (source row, token, score, token logprob) for every next row
        for n in range(N):
            alive = []
            if not done[n]:
                for rank in range(2 * K):
                    row = n * K + top_beam[n][rank]
                    tok, score, lp = top_tok[n][rank], top_scores[n][rank], top_logprobs[n][rank]
                    if tok == eos_id:
                        if rank < K:  # only EOS that would have made the top-K counts
                            finished[n].append((score / cur_len**length_penalty,
                                                seqs[row] + [tok], seq_logprobs[row] + [lp]))
                    else:
                        alive.append((row, tok, score, lp))
                    if len(alive) == K:
                        break
                finished[n] = sorted(finished[n], key=lambda h: h[0], reverse=True)[:K]
                # Stop this prompt once K finished hypotheses beat the best live beam.
                best_alive = alive[0][2] / cur_len**length_penalty
                if len(finished[n]) == K and finished[n][-1][0] >= best_alive:
                    done[n] = True
            if done[n]:  # placeholder rows; ignored from now on
                alive = [(n * K + j, pad_id, 0.0, 0.0) for j in range(K)]
            new_beams.extend(alive)

        seqs = [seqs[row] + [tok] for row, tok, _, _ in new_beams]
        seq_logprobs = [seq_logprobs[row] + [lp] for row, _, _, lp in new_beams]
        beam_scores = torch.tensor([score for _, _, score, _ in new_beams], device=device)
        if all(done) or cur_len == max_new_tokens:
            break

        rows = torch.tensor([row for row, _, _, _ in new_beams], device=device)
        past = _reorder_cache(past, rows)
        attention_mask = attention_mask.index_select(0, rows)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(N * K, 1)], dim=-1)
        next_tok = torch.tensor([tok for _, tok, _, _ in new_beams], device=device)[:, None]
        logits, past = _forward(model, next_tok, attention_mask, past)

    outputs = []
    for n in range(N):
        hyps = list(finished[n])
        if not done[n]:  # hit max_new_tokens: live beams compete with finished ones
            for j in range(K):
                row = n * K + j
                hyps.append((beam_scores[row].item() / len(seqs[row]) ** length_penalty,
                             seqs[row], seq_logprobs[row]))
        _, ids, lps = max(hyps, key=lambda h: h[0])
        outputs.append({"token_ids": ids, "logprobs": lps})
    return outputs
