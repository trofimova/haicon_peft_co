# Instructor Guide

## Workshop goal

Build practical intuition for **how PEFT methods differ**, not just in name, but in:
- where they intervene
- what they update
- how fast they adapt
- what tradeoffs they imply

## Recommended flow

### Part 1 — map the space
Use `notebooks/01_peft_building_blocks.ipynb`.

Main prompts:
- What exactly is trainable?
- Does this method change the input representation, the hidden computation, or the effective weights?
- Which methods are architecture-aware vs more “bolt-on”?

Expected takeaways:
- **Prompt tuning** modifies the conditioning pathway and acts through the input/token interface.
- **Adapters** inject trainable computation inside the network.
- **LoRA** changes the effective update in weight space while keeping the base weights frozen.
- **Linear probing** only changes the readout head.

### Part 2 — compare training behavior
Use `notebooks/02_peft_your_model.ipynb`.

Main prompts:
- Which method improves fastest early on?
- Which one saturates quickly?
- Which method gives the best return per trainable parameter?

Expected takeaways:
- linear probing is often a strong first baseline
- LoRA is often a strong quality/efficiency compromise
- prompt methods can be attractive when freezing the backbone is a hard requirement
- full/partial finetuning remain useful references

### Part 3 — few-shot adaptation under domain gap
Use `notebooks/03_few_shot_evaluation.ipynb`.

Main prompts:
- What changes when labels are scarce?
- Which method holds up when validation data is shifted away from the training domain?
- Does moving more parameters help, or does it overfit the few-shot source data?
- Does one labeled target-domain example per class change the preferred method?

Expected takeaways:
- few-shot performance and shifted-domain performance can favor different methods
- trainable parameter count is a constraint, not a quality guarantee
- a source-only winner may not remain best after adding even a tiny amount of target data
- there is no universally best PEFT method; data availability and domain gap decide

## Runtime expectations

- Notebook 1: ~10 minutes
- Notebook 2: ~10–15 minutes
- Notebook 3: ~20–30 minutes depending on hardware and epochs

For live workshops, keep the default epochs low and pre-run at least once on the target environment.

## Common discussion questions

### “Why not just full finetune?”
Because the point is not only final accuracy. Constraints matter:
- memory
- storage per task
- trainable parameter count
- task switching
- deployment packaging

### “Why is linear probing still worth showing?”
Because it is cheap, surprisingly strong on many tasks, and gives a calibration point for whether deeper adaptation is even needed.

### “When should I teach prompt tuning vs LoRA first?”
Prompt tuning is great for explaining **input-space intervention**.  
LoRA is great for explaining **weight-space low-rank updates**.  
Together they make the contrast very clear.

## Practical facilitation tips

- Ask participants to predict outcomes before running cells.
- Keep the method set small. Too many methods weakens the comparison.
- Reuse the same dataset and seed whenever possible.
- Always show trainable parameter counts next to results.

## Minimal recommended method set

For a clean workshop, these four are enough:
- linear probing
- prompt tuning / visual prompt tuning
- adapters
- LoRA

Then mention full finetuning only as a reference.

## Wrap-up slide / board summary

A good final summary is:

- **Need fastest cheap baseline?** → linear probe  
- **Need strong default PEFT baseline?** → LoRA  
- **Need modular hidden-state edits?** → adapters  
- **Need tiny conditioning-style adaptation?** → prompt tuning / VPT  
- **Need maximum task fit and can afford it?** → finetuning


### Part 4 — attendee playground
Use `notebooks/04_interactive_peft_explorer.ipynb`. Ask attendees to run the setup cell first; it installs `ipywidgets` in Colab and enables the widget manager. The notebook also supports a simple parameter-budget game and prints a leaderboard-ready result block after each run.

Main prompts:
- Which method would you choose under a given constraint?
- Does the quick run confirm your intuition?
- How much do your conclusions change between easy vs hard regimes?

Expected takeaways:
- method choice depends on constraints, not only leaderboard intuition
- prompt methods, adapters, and LoRA each encode different tradeoffs
- a cheap baseline is often worth trying before more expressive PEFT

## Additional runtime expectations

- Notebook 4: ~5–15 minutes depending on how many runs participants try

## Facilitation tip for Notebook 4

Ask participants to state a preference **before** pressing the run button:
- “I care most about a tiny task-specific state.”
- “I care most about strongest default quality-per-parameter.”
- “I care most about modularity.”

Then compare whether their chosen method and the observed run line up.


## Suggested flow extension: Inspect the change

Use `05_inspect_the_change.ipynb` after the convergence notebook or after the interactive explorer.

Teaching goal:
- move from “which method worked?” to “what changed inside the model?”

Key discussion prompts:
- What object is being learned: input patch, hidden residual, low-rank weight update, or just a head?
- Are good results associated with large changes, or can a small targeted change be enough?
- How does the inspection view explain the convergence behavior seen earlier?
