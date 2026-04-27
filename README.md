# HAICON 2026 — A Practical Tour of PEFT & Co.

Minimal, Colab-friendly workshop materials for building intuition around **parameter-efficient finetuning (PEFT)**.

This repo is organized around three layers:

1. **Concept notebook:** *where* different methods operate  
2. **Dynamics notebook:** *how* they converge and what tradeoffs appear in practice  
3. **Realistic notebook:** a small ViT + CIFAR benchmark to connect the intuition back to a real pretrained vision model
4. **Interactive notebook:** a choose-your-method playground for attendees

## Repo layout

```text
.
├── README.md
├── requirements.txt
├── instructor_guide.md
├── .gitignore
├── notebooks/
│   ├── 01_where_peft_operates.ipynb
│   ├── 02_convergence_and_what_to_use_when.ipynb
│   ├── 03_vit_cifar_peft_comparison.ipynb
│   └── 04_interactive_peft_explorer.ipynb
└── src/
    ├── __init__.py
    ├── data.py
    ├── training.py
    ├── visualization.py
    └── methods/
        ├── __init__.py
        ├── linear_probe.py
        ├── prompt_tuning.py
        ├── adapters.py
        └── lora.py
```

## Who this is for

This material is meant for a workshop audience that wants:
- a **shared mental model** for adaptation strategies
- small, inspectable experiments instead of giant training runs
- a practical answer to **“what should I use when?”**

## Notebook overview

### `01_where_peft_operates.ipynb`
A tiny frozen transformer-style setup that shows **where each method intervenes**.

Methods highlighted:
- linear probing
- prompt tuning / soft prompts
- adapters
- LoRA
- BitFit

Focus:
- input-space vs hidden-state vs weight-space intervention
- frozen vs trainable parameters
- visual intuition rather than benchmark realism

### `02_convergence_and_what_to_use_when.ipynb`
A shared training setup comparing methods on:
- train loss
- validation accuracy
- trainable parameter count
- practical takeaways

Focus:
- which methods are strong cheap baselines
- how parameter budget interacts with performance
- a first pass at “what to use when”

### `03_vit_cifar_peft_comparison.ipynb`
A more realistic **vision** example using a pretrained ViT checkpoint and CIFAR-10 (or a subset).

Included by default:
- linear probing
- partial finetuning
- LoRA
- visual prompt tuning

Optional extension:
- adapters

This notebook is intentionally scoped to stay shareable in Colab.

### `04_interactive_peft_explorer.ipynb`
A widget-based notebook for attendees to **pick a PEFT method from a dropdown**, choose a task regime, and run a very small experiment.

Included methods:
- linear probing
- prompt tuning
- adapters
- LoRA
- BitFit
- partial finetuning

Focus:
- self-directed exploration
- method cards and quick heuristics
- toy curves + one method-specific visualization
- “what should I use when?” discussion

This notebook works well as the interactive capstone at the end of the workshop. It now includes a top-of-notebook Colab setup cell for `ipywidgets`, plus a simple parameter-budget game and leaderboard-ready result block.

## Suggested workshop flow

A 75–90 minute version works well:

1. **10–15 min** — explain the adaptation landscape
2. **20 min** — run Notebook 1 together
3. **20 min** — run Notebook 2 and discuss curves
4. **15–20 min** — use Notebook 3 to connect the toy intuition to a real pretrained model
5. **10–15 min** — let attendees explore Notebook 4 in pairs or individually
6. **5 min** — wrap up with the selection heuristics

## Minimal setup

### Local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Colab

Upload the repo or clone it, then install:

```python
!pip install -q -r requirements.txt
```

If running the ViT notebook in Colab, a GPU is recommended.

## “What to use when?” — rough workshop heuristics

These are intentionally simplified teaching heuristics:

- **Linear probing**
  - start here when you want a very cheap baseline
  - great when pretrained features already separate your target task well

- **Prompt tuning / visual prompt tuning**
  - useful when you want to preserve the backbone almost entirely
  - attractive when you want a tiny adaptation footprint or easy swapping across tasks

- **Adapters**
  - useful when you want modularity and explicit hidden-state intervention
  - often nice for multi-task or per-task packaging

- **LoRA**
  - often the best default first PEFT baseline
  - strong balance of parameter efficiency, flexibility, and performance

- **Partial / full finetuning**
  - use when you have enough data/compute and the gap matters
  - still the reference point for maximum task-specific adaptation

## Notes

- The first two notebooks are **intentionally minimalistic** and designed for understanding.
- The third notebook is more realistic but still optimized for workshop runtime.
- The helper code in `src/` is small on purpose so participants can actually read it.

## License / reuse

Feel free to adapt this structure for teaching, tutorials, and GitHub sharing.


## Notebooks

- `01_where_peft_operates.ipynb` — conceptual map of where PEFT intervenes
- `02_convergence_and_what_to_use_when.ipynb` — compare training behavior and trade-offs
- `03_vit_cifar_peft_comparison.ipynb` — more realistic vision benchmark
- `04_interactive_peft_explorer.ipynb` — attendee playground with method dropdowns, an in-notebook setup cell for `ipywidgets`, and a parameter-budget score
- `05_inspect_the_change.ipynb` — inspect what each method actually changes inside the model
