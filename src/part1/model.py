"""Decoder-only transformer with a drop-in FFN: dense 2-layer MLP or Mixture of Experts.

Self-test (CPU, seconds):
    uv run python -m src.part1.model

MoE (written from scratch): linear router -> softmax -> top-k experts per token. With
k > 1 the chosen gates are renormalised to sum to 1; with k = 1 the raw probability is
kept as the gate (otherwise the router gets no gradient from the LM loss). Dispatch is
dropless: each expert gathers exactly the tokens routed to it (no capacity factor), and
the gated outputs are scattered back with index_add_. An optional always-on shared expert
is added on top. Padding tokens are never routed and never counted in the aux losses.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 512          # dimension of the model
    n_ctx: int = 128            # number of context tokens (training length)
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048            # dense MLP hidden size
    dropout: float = 0.0
    use_moe: bool = False       # whether to use MoE or MLP
    n_experts: int = 4          # routed experts
    top_k: int = 1              # routed experts active per token
    n_shared: int = 0           # always-active shared experts
    d_expert: int = 512         # hidden size of each expert
    aux_lb_coef: float = 0.01   # load-balancing loss weight
    aux_z_coef: float = 0.001   # router z-loss weight
    rope_base: float = 10000.0


# ---------------------------------------------------------------- FFNs

class MLP(nn.Module):
    """Standard 2-layer MLP: d_model -> d_hidden -> GELU -> d_model."""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MoE(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_experts, self.top_k = config.n_experts, config.top_k
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        self.experts = nn.ModuleList(MLP(config.d_model, config.d_expert) for _ in range(config.n_experts))
        self.shared = nn.ModuleList(MLP(config.d_model, config.d_expert) for _ in range(config.n_shared))

    def forward(self, x, valid=None):
        # Input shape: (B, N_ctx, D_model); valid: (B, N_ctx) bool, False = padding
        # Output shape: (B, N_ctx, D_model), plus routing info for the aux losses / stats
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        tok_idx = torch.arange(B * T, device=x.device) if valid is None else valid.reshape(-1).nonzero(as_tuple=True)[0]
        h = flat.index_select(0, tok_idx)                                 # (N, D) real tokens only
        N, E, k = h.shape[0], self.n_experts, self.top_k

        logits = self.router(h).float()                                   # (N, E)
        probs = logits.softmax(dim=-1)
        top_p, top_e = probs.topk(k, dim=-1)                              # (N, k)
        gates = top_p / top_p.sum(dim=-1, keepdim=True) if k > 1 else top_p

        out = torch.zeros_like(h)
        assign_e = top_e.reshape(-1)                                      # (N*k,) expert of each assignment
        assign_tok = torch.arange(N, device=x.device).repeat_interleave(k)
        assign_gate = gates.reshape(-1)
        for e, expert in enumerate(self.experts):
            sel = (assign_e == e).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            rows = assign_tok[sel]
            y = expert(h.index_select(0, rows)) * assign_gate[sel, None].to(h.dtype)
            out.index_add_(0, rows, y)
        for expert in self.shared:
            out = out + expert(h)

        # Switch-Transformer load balancing: E * sum_i f_i * P_i (= 1 when perfectly uniform),
        # f_i = fraction of assignments to expert i, P_i = mean router probability of expert i.
        counts = torch.bincount(assign_e, minlength=E)
        frac = counts.float() / max(N * k, 1)
        mean_prob = probs.mean(dim=0)
        lb_loss = E * (frac * mean_prob).sum()
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()

        y = torch.zeros_like(flat).index_copy(0, tok_idx, out).reshape(B, T, D)
        info = {
            "lb_loss": lb_loss,
            "z_loss": z_loss,
            "counts": counts.detach(),                      # (E,)
            "mean_prob": mean_prob.detach(),                # (E,)
            "entropy": -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean().detach(),
            "top_e": top_e.detach(),                        # (N, k) for the real tokens, row-major order
            "gates": gates.detach(),                        # (N, k)
        }
        return y, info


# ---------------------------------------------------------------- attention

def rope(x, positions, base: float):
    """Rotary embeddings. x: (B, H, T, Dh), positions: (B, T) or (T,)."""
    d = x.shape[-1]
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = positions.float()[..., None] * inv_freq                         # (B, T, d/2) or (T, d/2)
    if ang.dim() == 2:
        ang = ang[None]
    cos, sin = ang.cos()[:, None].to(x.dtype), ang.sin()[:, None].to(x.dtype)  # (B, 1, T, d/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads, self.rope_base, self.dropout = config.n_heads, config.rope_base, config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x, positions, attn_mask=None, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k = rope(q, positions, self.rope_base), rope(k, positions, self.rope_base)
        if cache is not None:  # incremental decoding: append to this layer's K/V
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        drop = self.dropout if self.training else 0.0
        if attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.norm2 = nn.RMSNorm(config.d_model)
        self.use_moe = config.use_moe
        self.ffn = MoE(config) if config.use_moe else MLP(config.d_model, config.d_ff)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, positions, valid=None, attn_mask=None, cache=None):
        x = x + self.drop(self.attn(self.norm1(x), positions, attn_mask, cache))
        if self.use_moe:
            h, info = self.ffn(self.norm2(x), valid)
        else:
            h, info = self.ffn(self.norm2(x)), None
        return x + self.drop(h), info


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tied embeddings
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2 style scaled init of residual projections
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids, valid=None, positions=None, attn_mask=None, caches=None):
        # Input: token ids (B, N_ctx). Output: logits (B, N_ctx, N_vocab) and aux info:
        #   aux["loss"]  = weighted sum of the MoE aux losses over layers (0 for dense)
        #   aux["layers"] = per-layer routing info (empty for dense)
        T = input_ids.shape[1]
        if positions is None:
            positions = torch.arange(T, device=input_ids.device)
        x = self.embed(input_ids)
        layers = []
        for i, block in enumerate(self.blocks):
            x, info = block(x, positions, valid, attn_mask, None if caches is None else caches[i])
            if info is not None:
                layers.append(info)
        logits = self.lm_head(self.norm(x))
        aux = {"layers": layers, "loss": logits.new_zeros((), dtype=torch.float32),
               "lb_loss": 0.0, "z_loss": 0.0}
        if layers:
            lb = torch.stack([l["lb_loss"] for l in layers]).sum()
            z = torch.stack([l["z_loss"] for l in layers]).sum()
            aux.update(loss=self.config.aux_lb_coef * lb + self.config.aux_z_coef * z,
                       lb_loss=lb.detach() / len(layers), z_loss=z.detach() / len(layers))
        return logits, aux


def count_params(model: Transformer) -> dict[str, int]:
    """Total vs active (per-token) parameters; tied embeddings are counted once."""
    c = model.config
    total = sum(p.numel() for p in model.parameters())
    embed = model.embed.weight.numel()
    inactive = 0
    if c.use_moe:
        per_expert = sum(p.numel() for p in model.blocks[0].ffn.experts[0].parameters())
        inactive = c.n_layers * (c.n_experts - c.top_k) * per_expert
    ffn_total = sum(p.numel() for b in model.blocks for p in b.ffn.parameters())
    return {
        "total_params": total,
        "active_params": total - inactive,
        "non_embedding_params": total - embed,
        "active_non_embedding_params": total - embed - inactive,
        "ffn_params": ffn_total,
        "active_ffn_params": ffn_total - inactive,
    }


# ---------------------------------------------------------------- generation helpers

def prefill_mask(valid: torch.Tensor) -> torch.Tensor:
    """(B, 1, T, T) bool mask for left-padded prompts: causal and key-not-padding. Every
    query may always see itself, so padded rows never become all-masked (NaN)."""
    T = valid.shape[1]
    causal = torch.ones(T, T, dtype=torch.bool, device=valid.device).tril()
    eye = torch.eye(T, dtype=torch.bool, device=valid.device)
    return ((causal & valid[:, None, :]) | eye)[:, None]


# ---------------------------------------------------------------- self-test

def _naive_moe(moe: MoE, x: torch.Tensor) -> torch.Tensor:
    """Per-token reference implementation of the MoE forward pass."""
    out = torch.zeros_like(x)
    for n in range(x.shape[0]):
        p = moe.router(x[n]).float().softmax(-1)
        top_p, top_e = p.topk(moe.top_k)
        g = top_p / top_p.sum() if moe.top_k > 1 else top_p
        for gi, ei in zip(g, top_e):
            out[n] += gi * moe.experts[int(ei)](x[n])
        for s in moe.shared:
            out[n] += s(x[n])
    return out


def _self_test():
    from src.part1.common import load_config, model_config

    torch.manual_seed(0)
    cfg = load_config("src/part1/config.yaml")
    print(f"{'variant':22s} {'total':>12s} {'active':>12s} {'ffn total':>12s} {'ffn active':>12s}")
    for v in cfg["variants"]:
        model = Transformer(model_config(cfg, v, cfg["tokenizer"]["vocab_size"]))
        c = count_params(model)
        print(f"{v['name']:22s} {c['total_params']:12,d} {c['active_params']:12,d} "
              f"{c['ffn_params']:12,d} {c['active_ffn_params']:12,d}")

    for v in cfg["variants"]:
        mc = model_config(cfg, v, 100)
        mc.d_model, mc.d_ff, mc.d_expert, mc.n_layers, mc.n_heads = 32, 64, 16, 2, 4
        model = Transformer(mc).eval()
        checks = []

        if mc.use_moe:
            # 1. vectorised dropless dispatch == naive per-token loop
            x = torch.randn(3, 7, 32)
            y, _ = model.blocks[0].ffn(x)
            ref = _naive_moe(model.blocks[0].ffn, x.reshape(-1, 32)).reshape(3, 7, 32)
            assert torch.allclose(y, ref, atol=1e-5), f"{v['name']}: dispatch mismatch"

            # 2. padding does not change real tokens; pad rows are zero
            valid = torch.ones(3, 7, dtype=torch.bool)
            valid[1, 4:] = False
            yv, info = model.blocks[0].ffn(x, valid)
            assert torch.allclose(yv[valid], y[valid], atol=1e-5) and yv[~valid].abs().max() == 0
            assert info["top_e"].shape == (int(valid.sum()), mc.top_k)
            checks += ["dispatch == naive", "padding ok"]

        # 3. left-padded prefill + KV-cache decoding == full forward pass.
        #    row 0: 6-token prompt; row 1: 3 pads + 3-token prompt; then 2 decode steps.
        ids = torch.randint(6, 100, (2, 9))
        full, _ = model(ids)
        prompts = torch.stack([ids[0, :6], torch.cat([torch.zeros(3, dtype=torch.long), ids[1, :3]])])
        pvalid = torch.tensor([[1] * 6, [0] * 3 + [1] * 3], dtype=torch.bool)
        positions = (pvalid.long().cumsum(1) - 1).clamp_min(0)
        caches = [{} for _ in model.blocks]
        logits, _ = model(prompts, valid=pvalid, positions=positions, attn_mask=prefill_mask(pvalid), caches=caches)
        assert torch.allclose(logits[0, -1], full[0, 5], atol=1e-4), f"{v['name']}: prefill row 0"
        assert torch.allclose(logits[1, -1], full[1, 2], atol=1e-4), f"{v['name']}: prefill row 1"
        key_valid, pos = pvalid, positions[:, -1]
        for step in range(2):
            nxt = torch.stack([ids[0, 6 + step], ids[1, 3 + step]])[:, None]
            key_valid = torch.cat([key_valid, torch.ones(2, 1, dtype=torch.bool)], 1)
            pos = pos + 1
            logits, _ = model(nxt, valid=torch.ones(2, 1, dtype=torch.bool), positions=pos[:, None],
                              attn_mask=key_valid[:, None, None, :], caches=caches)
            assert torch.allclose(logits[0, -1], full[0, 6 + step], atol=1e-4), f"{v['name']}: kv-cache row 0"
            assert torch.allclose(logits[1, -1], full[1, 3 + step], atol=1e-4), f"{v['name']}: kv-cache row 1"
        checks.append("kv-cache ok")
        print(f"{v['name']}: " + ", ".join(checks))
    print("self-test passed")


if __name__ == "__main__":
    _self_test()
