# Q3: Decoding strategies (Pythia-160M on ROCStories)

Greedy, top-k, top-p and beam search (widths 1, 2, 4) implemented from scratch on
`model.forward()` (no `generate()`), evaluated on 1000 ROCStories validation prompts.

- **WandB runs:** https://wandb.ai/likhitbhogadi-iiit-hyderabad/anlp-a2-q3
- **Raw generations:** https://huggingface.co/datasets/<hf-username>/anlp-a2-q3-generations  <!-- TODO -->
- **Model:** pretrained `EleutherAI/pythia-160m`, unchanged (no new checkpoint).

| File | Purpose |
|---|---|
| `decoding.py` | the four decoding strategies |
| `run.py` | runs every experiment in `config.yaml`, writes `outputs/` and logs to WandB |
| `sanity.py` | correctness checks (beam w=1 == greedy, batching, cached vs. full-forward log-probs) |
| `analyze.py` | diagnostics (`outputs/analysis.csv`) and report figures (`assets/`) |
| `data.py`, `metrics.py`, `logging_utils.py` | data loading, PPL / positional accuracy, run logs |
| `outputs/results.csv` | PPL, accuracy and timing for all 13 configurations |
| `../report.md` | full report (Q3 = Section 3) |

```bash
uv run python -m src.part3.sanity --device cuda:0       # ~1 min
uv run python -m src.part3.run --device cuda:0          # all 13 configs, ~5 min on an L40S
uv run python -m src.part3.analyze                      # CPU only: diagnostics + figures
```
