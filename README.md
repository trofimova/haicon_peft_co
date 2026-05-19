# HAICON 2026 — A Practical Tour of PEFT

Minimal, Colab-friendly workshop materials for building intuition around **parameter-efficient fine-tuning (PEFT)** applied to vision transformers.

## Repo layout

```text
.
├── notebooks/
│   ├── 01_peft_building_blocks.ipynb   # anatomy guide — what each method touches
│   └── 02_peft_your_model.ipynb        # interactive template — plug in your data
└── src/
    ├── training.py                     # train loop, evaluation, helpers
    └── methods/
        ├── configs.py                  # hyperparameter dataclasses for each method
        ├── linear_probe.py
        ├── bitfit.py
        ├── prompt_tuning.py
        ├── lora.py
        ├── adapters.py
        └── partial_ft.py
```

## Notebooks

### `01_peft_building_blocks.ipynb` — Anatomy guide

Loads `WinKawaks/vit-small-patch16-224` and walks through all six PEFT methods one by one, with a dedicated visualisation for each.

| Section | Method | What you see |
|---------|--------|--------------|
| 1 | Linear Probe | head weight heatmap |
| 2 | BitFit | per-block bias param counts (log scale) + bias type breakdown |
| 3 | Visual Prompt Tuning | token sequence diagram, prompt value heatmap, per-position L2 norm |
| 4 | LoRA | W₀, A, B and ΔW matrix heatmaps for the last adapted layer |
| 5 | Adapter | bottleneck down/up projection heatmaps + near-zero residual check |
| 6 | Partial Fine-tuning | trainable vs frozen params per block for last-4 / last-8 / all-12 blocks |
| 7 | Summary | trainable param count bar chart + "what does each method touch?" grid |

### `02_peft_your_model.ipynb` — Interactive template

A fill-in-the-blanks notebook for applying any PEFT method to your own model and dataset.

- **Section 1 — Config:** choose `METHOD`, dataset size, training hyperparams
- **Section 2 — Backbone:** loads a HuggingFace ViT; swap in any backbone
- **Section 3 — Dataset:** CIFAR-10 by default with configurable subset size; replace with any `torch.utils.data.Dataset`
- **Section 4 — Build model:** `build_model()` reads hyperparams from `src/methods/configs.py`
- **Section 5 — Train:** standard AdamW loop with loss + accuracy curves
- **Section 6 — Evaluate:** training curves and final accuracy
- **Section 7 — Compare all methods:** runs all six methods and produces a 6-panel efficiency dashboard (accuracy, params, training time, inference latency, peak memory)

## `src/methods/` — Building blocks

Each file is a self-contained PyTorch `nn.Module` wrapper around a frozen backbone.

| File | Class | What it trains |
|------|-------|----------------|
| `linear_probe.py` | `LinearProbeModel` | Linear head only |
| `bitfit.py` | `BitFitClassifier` | All bias terms in the backbone (Zaken et al. 2021) |
| `prompt_tuning.py` | `PromptTunedClassifier` | k learnable token vectors inserted between CLS and patch tokens (VPT, Jia et al. 2022) |
| `lora.py` | `LoRAClassifier` | Low-rank decomposition ΔW = BA on selected attention projections (Hu et al. 2022) |
| `adapters.py` | `AdapterHeadClassifier` | Bottleneck module (down → GELU → up + residual) appended after the backbone |
| `partial_ft.py` | `PartialFineTuneClassifier` | Selected transformer blocks unfrozen |

### `configs.py` — Method hyperparameters

Default hyperparameters for each method live in `src/methods/configs.py` as dataclasses, so the notebook config cell stays clean:

```python
# src/methods/configs.py
LoRAConfig(rank=8, target_modules=["query", "value"])
VisualPromptConfig(num_prompt_tokens=10)
AdapterConfig(bottleneck_dim=32)
PartialFTConfig(n_blocks=1)
```

Edit this file to change hyperparams without touching the notebook.

## Running in Colab

Both notebooks include an auto-setup cell that clones the repo and installs dependencies when running in Colab. Just open the notebook and run all cells — no manual setup needed.

```
https://github.com/trofimova/haicon_peft_co
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision transformers peft matplotlib pandas tqdm
```

## Quick method reference

| Method | Trainable params | Changes forward pass? | Good default when… |
|--------|-----------------|----------------------|---------------------|
| Linear Probe | ~4 K | no | features already align with your task |
| BitFit | ~56 K | no | very tight param budget |
| VPT | ~16 K | yes | you want zero backbone modification |
| LoRA | ~150 K | no | general-purpose first PEFT baseline |
| Adapter | ~29 K | no | modular per-task packaging matters |
| Partial FT | 1 M+ | no | you have data and want maximum accuracy |

*(Numbers for ViT-small-patch16-224, 10-class head.)*
