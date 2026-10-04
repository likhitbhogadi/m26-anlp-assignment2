"""From-scratch optimizers, one per category of Wen et al. (2025), Table 1.

    AdamW  (baseline)                      Loshchilov & Hutter, 2019
    MARS   (variance-reduced AdamW)        Yuan et al., 2024 -- MARS-AdamW, approximate variant
    Lion   (memory-efficient)              Chen et al., 2023
    Muon   (matrix-based)                  Jordan et al., 2024 (+ AdamW for non-matrix params)

Only `torch.optim.Optimizer` is used, as a base class (param groups / state bookkeeping).
Every group carries a `peak_lr`; the training loop sets `lr = peak_lr * schedule(step)`.
Weight decay is decoupled everywhere: p <- p * (1 - lr * wd).

Self-test (CPU, seconds):
    uv run python -m src.part2.optimizers
"""

import math

import torch
from torch.optim import Optimizer


def _adamw_step(p, g, state, lr, betas, eps, wd):
    """One AdamW update of p with gradient-like tensor g (bias-corrected moments)."""
    b1, b2 = betas
    if "m" not in state:
        state["step"], state["m"], state["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
    state["step"] += 1
    t, m, v = state["step"], state["m"], state["v"]
    m.lerp_(g, 1 - b1)                                  # m = b1 m + (1 - b1) g
    v.mul_(b2).addcmul_(g, g, value=1 - b2)             # v = b2 v + (1 - b2) g^2
    denom = (v / (1 - b2 ** t)).sqrt_().add_(eps)
    p.mul_(1 - lr * wd)
    p.addcdiv_(m, denom, value=-lr / (1 - b1 ** t))


class AdamW(Optimizer):
    """m_t = b1 m + (1-b1) g;  v_t = b2 v + (1-b2) g^2;
    w <- w - lr * (m_t/(1-b1^t)) / (sqrt(v_t/(1-b2^t)) + eps) - lr * wd * w."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    _adamw_step(p, p.grad, self.state[p], group["lr"], group["betas"], group["eps"],
                                group["weight_decay"])


class MARS(Optimizer):
    """MARS-AdamW (approximate): variance reduction with the previous step's gradient.
        c_t = g_t + gamma * b1/(1-b1) * (g_t - g_{t-1});   c_t <- c_t / max(1, ||c_t||_2)
        then an AdamW step on c_t.
    The exact MARS needs grad(w_{t-1}) on the *current* batch (a second backward pass); the
    approximate version (used in the MARS paper's experiments and in the benchmark) reuses
    g_{t-1} from the previous batch. As in the reference implementation, the correction is
    applied to matrices only; 1-D params (norm gains) take a plain AdamW step."""

    def __init__(self, params, lr=3e-3, betas=(0.95, 0.99), eps=1e-8, weight_decay=0.0, gamma=0.025):
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay, gamma=gamma))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1 = group["betas"][0]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, state = p.grad, self.state[p]
                if p.dim() >= 2:
                    if "last_grad" not in state:
                        state["last_grad"] = torch.zeros_like(p)   # as in the reference code
                    c = g + (group["gamma"] * b1 / (1 - b1)) * (g - state["last_grad"])
                    c.div_(c.norm().clamp_min(1.0))              # clip ||c||_2 to 1
                    state["last_grad"].copy_(g)
                else:
                    c = g
                _adamw_step(p, c, state, group["lr"], group["betas"], group["eps"], group["weight_decay"])


class Lion(Optimizer):
    """u = sign(b1 m + (1-b1) g);  w <- w - lr * u - lr * wd * w;  m <- b2 m + (1-b2) g.
    One state tensor per parameter (AdamW keeps two)."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=tuple(betas), weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, state = p.grad, self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                u = m.lerp(g, 1 - b1).sign_()
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u, alpha=-group["lr"])
                m.lerp_(g, 1 - b2)


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate orthogonalisation U V^T of G = U S V^T (batched over leading dims) with the
    quintic Newton-Schulz iteration X <- aX + (b XX^T + c (XX^T)^2) X, in bf16. The coefficients
    push all singular values into roughly [0.7, 1.2] in 5 steps."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)   # spectral norm <= Frobenius norm <= 1
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return (X.mT if tall else X).to(G.dtype)


class Muon(Optimizer):
    """Groups with `use_muon=True` (2-D hidden weight matrices):
        m <- mu m + (1-mu) g;   d = (1-mu) g + mu m  (Nesterov) or m;
        O = NewtonSchulz5(d) * sqrt(max(1, rows/cols));   w <- w - lr * O - lr * wd * w.
    A matrix with `n_split > 1` (the fused qkv projection) is orthogonalised as n_split
    separate (rows/n_split x cols) blocks, i.e. Q, K and V each get their own update.
    Other groups (embeddings, norm gains) take an AdamW step with that group's settings."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0,
                 betas=(0.9, 0.95), eps=1e-8):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay, betas=tuple(betas), eps=eps,
                                      use_muon=True, n_split=1))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, state = p.grad, self.state[p]
                if not group["use_muon"]:
                    _adamw_step(p, g, state, group["lr"], group["betas"], group["eps"], group["weight_decay"])
                    continue
                mu = group["momentum"]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"].lerp_(g, 1 - mu)
                d = g.lerp(m, mu) if group["nesterov"] else m
                k = group["n_split"]
                blocks = d.view(k, d.size(0) // k, d.size(1))
                rows, cols = blocks.shape[-2:]
                O = newton_schulz(blocks, group["ns_steps"]).view_as(p) * math.sqrt(max(1.0, rows / cols))
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(O, alpha=-group["lr"])


# ---------------------------------------------------------------- construction

NAMES = ["adamw", "mars", "lion", "muon"]


def build_optimizer(name: str, model: torch.nn.Module, hp: dict) -> Optimizer:
    """Param groups as in Q1: weight decay on >= 2-D params only. For Muon, the 2-D weights inside
    the transformer blocks use Muon; the (tied) embedding and norm gains use AdamW with adam_*.
    Every group gets `peak_lr` (the scheduler multiplies it)."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    decay = [p for _, p in named if p.dim() >= 2]
    no_decay = [p for _, p in named if p.dim() < 2]
    wd = hp.get("weight_decay", 0.0)
    if name in ("adamw", "mars", "lion"):
        groups = [{"params": decay, "weight_decay": wd, "peak_lr": hp["lr"]},
                  {"params": no_decay, "weight_decay": 0.0, "peak_lr": hp["lr"]}]
        if name == "adamw":
            return AdamW(groups, lr=hp["lr"], betas=hp["betas"], eps=hp["eps"])
        if name == "mars":
            return MARS(groups, lr=hp["lr"], betas=hp["betas"], eps=hp["eps"], gamma=hp["gamma"])
        return Lion(groups, lr=hp["lr"], betas=hp["betas"])
    if name == "muon":
        matrix = [(n, p) for n, p in named if p.dim() == 2 and n.startswith("blocks.")]
        matrix_ids = {id(p) for _, p in matrix}
        qkv = [p for n, p in matrix if n.endswith("attn.qkv.weight")]
        other_mat = [p for n, p in matrix if not n.endswith("attn.qkv.weight")]
        adam_kw = dict(use_muon=False, peak_lr=hp["adam_lr"], betas=tuple(hp["adam_betas"]), eps=hp["adam_eps"])
        groups = [
            {"params": qkv, "n_split": 3, "weight_decay": wd, "peak_lr": hp["lr"]},
            {"params": other_mat, "n_split": 1, "weight_decay": wd, "peak_lr": hp["lr"]},
            {"params": [p for p in decay if id(p) not in matrix_ids], "weight_decay": wd, **adam_kw},
            {"params": no_decay, "weight_decay": 0.0, **adam_kw},
        ]
        return Muon(groups, lr=hp["lr"], momentum=hp["momentum"], nesterov=hp["nesterov"], ns_steps=hp["ns_steps"])
    raise SystemExit(f"unknown optimizer {name!r}; choose from {NAMES}")


def state_bytes(opt: Optimizer) -> int:
    """Bytes held in optimizer state tensors (moments, last gradients, momentum buffers)."""
    return sum(v.numel() * v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))


def set_lr(opt: Optimizer, mult: float):
    for g in opt.param_groups:
        g["lr"] = g["peak_lr"] * mult


# ---------------------------------------------------------------- self-test

def _self_test():
    torch.manual_seed(0)
    shapes = [(6, 4), (4,), (9, 4)]

    def params():
        torch.manual_seed(1)
        return [torch.randn(s, requires_grad=True) for s in shapes]

    grads = [[torch.randn(s) for s in shapes] for _ in range(10)]

    def run(make_opt):
        """Fresh params, `make_opt(params)`, 10 steps with the fixed grads; returns the params."""
        ps = params()
        opt = make_opt(ps)
        for gs in grads:
            for p, g in zip(ps, gs):
                p.grad = g.clone()
            opt.step()
        return ps

    # 1. AdamW == torch.optim.AdamW (reference used only here, never for training)
    ours = run(lambda ps: AdamW(ps, lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1))
    ref = run(lambda ps: torch.optim.AdamW(ps, lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1))
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(ours, ref)), "AdamW mismatch"
    print("AdamW == torch.optim.AdamW")

    # 2. Lion and MARS vs straightforward per-step reference formulas
    lr, wd, (b1, b2) = 1e-2, 0.1, (0.9, 0.99)
    ours = run(lambda ps: Lion(ps, lr=lr, betas=(b1, b2), weight_decay=wd))
    ref, ms = params(), [torch.zeros(s) for s in shapes]
    with torch.no_grad():
        for gs in grads:
            for p, m, g in zip(ref, ms, gs):
                p.copy_(p * (1 - lr * wd) - lr * torch.sign(b1 * m + (1 - b1) * g))
                m.copy_(b2 * m + (1 - b2) * g)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(ours, ref)), "Lion mismatch"
    print("Lion == reference")

    gamma, eps, (b1, b2) = 0.025, 1e-8, (0.95, 0.99)
    ours = run(lambda ps: MARS(ps, lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd, gamma=gamma))
    ref = params()
    st = [dict(m=torch.zeros(s), v=torch.zeros(s), last=torch.zeros(s)) for s in shapes]
    with torch.no_grad():
        for t, gs in enumerate(grads, 1):
            for p, s, g in zip(ref, st, gs):
                c = g
                if p.dim() >= 2:
                    c = g + gamma * b1 / (1 - b1) * (g - s["last"])
                    if c.norm() > 1:
                        c = c / c.norm()
                    s["last"] = g
                s["m"] = b1 * s["m"] + (1 - b1) * c
                s["v"] = b2 * s["v"] + (1 - b2) * c * c
                upd = (s["m"] / (1 - b1 ** t)) / ((s["v"] / (1 - b2 ** t)).sqrt() + eps)
                p.copy_(p * (1 - lr * wd) - lr * upd)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(ours, ref)), "MARS mismatch"
    print("MARS == reference")

    # 3. Newton-Schulz: singular values of the output are ~1 (wide, tall, batched)
    for shape in [(32, 64), (64, 32), (3, 16, 48)]:
        s = torch.linalg.svdvals(newton_schulz(torch.randn(shape)).float())
        assert s.min() > 0.5 and s.max() < 1.3, f"NS singular values {s.min():.2f}..{s.max():.2f}"
        print(f"Newton-Schulz {shape}: singular values in [{s.min():.2f}, {s.max():.2f}]")

    # 4. every optimizer (through build_optimizer, incl. Muon's qkv split) trains a tiny transformer
    from src.part1.model import Transformer, TransformerConfig

    hps = {"adamw": dict(lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1),
           "mars": dict(lr=3e-3, betas=(0.95, 0.99), eps=1e-8, weight_decay=0.1, gamma=0.025),
           "lion": dict(lr=1e-3, betas=(0.9, 0.99), weight_decay=1.0),
           "muon": dict(lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.1,
                        adam_lr=3e-3, adam_betas=(0.9, 0.95), adam_eps=1e-8)}
    data = torch.randint(0, 50, (16, 17))
    data[:, 1::2] = data[:, 0::2][:, :8] % 7          # learnable structure
    for name in NAMES:
        torch.manual_seed(0)
        model = Transformer(TransformerConfig(vocab_size=50, d_model=32, n_ctx=16, n_layers=2, n_heads=4, d_ff=64))
        opt = build_optimizer(name, model, hps[name])
        n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
        assert n_opt == sum(p.numel() for p in model.parameters()), f"{name}: params missing from groups"
        losses = []
        for _ in range(60):
            logits, _ = model(data[:, :-1])
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), data[:, 1:].flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            set_lr(opt, 1.0)
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < 0.8 * losses[0], f"{name}: loss {losses[0]:.3f} -> {losses[-1]:.3f}"
        print(f"{name:6s} toy LM loss {losses[0]:.3f} -> {losses[-1]:.3f}, state {state_bytes(opt) / 1024:.1f} KiB")
    print("self-test passed")


if __name__ == "__main__":
    _self_test()
