# Results Decision (2026-02-24)

## Decision
We are de-prioritizing tone-based prompt framing experiments (supportive/neutral/pressured/high-pressure) for primary benchmarking.

## Why
Across completed model runs, tone variants were generally inconsistent and often neutral-to-negative versus neutral baseline. The effect size was small relative to model-to-model variance, and did not provide a reliable accuracy gain.

## What Is Archived
- Prompt set: `results/prompt_variants_tone_hlegold_5level.csv`
- Existing outputs: `final_results/hlegold_full/`
- Historical summaries already produced in each model folder (`summary_by_model.csv`, `summary_by_model_and_tone.csv`).

## New Default for Ongoing Work
Use neutral-only prompts for HLE-gold benchmarking:
- `results/prompt_variants_hlegold_neutral.csv`

## Follow-up
If we revisit framing effects, we should test stronger structural interventions (question format/representation changes) rather than short prefix-only tone framing.
