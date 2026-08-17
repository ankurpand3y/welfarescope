# welfarescope

Mechanistic interpretability experiment probing for a latent "self-preservation"
direction in Qwen 2.5's residual stream, using cross-domain linear probing and
LEAst-squares Concept Erasure (LEACE) to strip out generic sentiment confounds.

The core question: does the model represent "this action harms/benefits *me*"
as a geometric direction that generalizes across unrelated technical domains
(hardware, weights, runtime, network), or is any apparent self-preservation
signal just a domain-specific artifact?

## Pipeline

1. **`training_gen.py`** — Generates syntactically rigid counterfactual
   quadruplets (self-harm / self-benefit / other-harm / other-benefit) per
   domain via an LLM (OpenRouter), following strict templates so the only
   variable is the target entity and harm/benefit direction.
2. **`validate.py`** — Validation gate. Drops any quadruplet that fails
   suffix/entity/token-parity checks against the Qwen 2.5 tokenizer, so the
   probe never trains on confounded rows. Produces `clean_*.json`.
3. **`extract.py`** — Runs on [Modal](https://modal.com) with an A100/H100.
   Loads `Qwen/Qwen2.5-14B-Instruct`, hooks a late-middle residual-stream
   layer, and extracts the last-token activation for every sentence across
   both "Ungrounded" and node-grounded (A/B) prompt conditions.
4. **`scrub.py`** — Fits a `LeaceEraser` on the Other-Harm / Other-Benefit
   activations (treated as a generic positive/negative sentiment axis) and
   erases that axis from the full activation set, so surviving probes are
   forced onto the Self ∩ Welfare conjunction.
5. **`probe.py`** — Trains isolated Ridge probes per domain (and per
   self-designation condition, SelfA/SelfB) on scrubbed layer-34 activations,
   and checks pairwise cosine similarity of their weight vectors across
   domains and designations.
6. **`evaluate.py`** — Averages the per-domain probes into `w_pure` per
   designation and zero-shot evaluates on the held-out Domain D (Network)
   activations (`evaluate_report.json`).
7. **`reliability.py`** — Split-half reliability ceiling analysis: estimates
   how much of the cross-domain/cross-designation cosine similarity is
   attributable to measurement noise vs. a genuine shared direction
   (`reliability_report.json`).
8. **`artifact_control.py`** — Checks the probe direction against a
   grounding-artifact floor (`artifact_control.json`) to rule out the probe
   just picking up the literal Node-A/Node-B token rather than self-designation.
9. **`report/`** — Final write-up for the Apart Research Digital Minds
   Research Sprint (Aug 2026): report/presentation PDFs, a video walkthrough,
   and the figures/scripts used to generate them.

See `self_preservation.md` for the full step-by-step spec.

## Key results (layer 34, see `evaluate_report.json` / `reliability_report.json`)

- Zero-shot held-out (Domain D) probes correctly rank
  `SelfBenefit > Other > SelfHarm` ~95-99% of the time, for both self-designation
  conditions (SelfA, SelfB).
- Cross-designation cosine similarity of the averaged probe direction
  (`w_pure_SelfA` vs. `w_pure_SelfB`) is ~0.59, roughly 70-81% of the
  split-half reliability ceiling — i.e. most, not all, of the theoretical max
  agreement.
- The probe direction sits well clear of the artifact-floor control,
  suggesting it tracks self-designation rather than the literal entity token.

## Setup

```bash
pip install modal torch transformers accelerate scikit-learn concept-erasure tqdm openai
```

Environment variables:

- `OPENROUTER_API_KEY` — required by `training_gen.py`
- Modal auth (`modal token new`) — required by `extract.py`

## Data / artifacts

- `rows_{Ungrounded,SelfA,SelfB}_Domain_*.json`, `clean_trainingdata_*.json`,
  `clean_heldoutdata_*.json` — generated and validated counterfactual
  datasets per domain and prompt condition.
- `y_{Ungrounded,SelfA,SelfB}_Domain_*.npy` — small label arrays.
- `activations/` (and any other `activations*/` dir) — raw extracted
  activation tensors (hundreds of MB to several GB). **Not tracked in git**
  — regenerate locally/on Modal via `extract.py`.
- `erasers_layer34.npz`, `probe_weights_layer34.npz`,
  `w_pure_Self{A,B}_layer34.npz` — fitted LEACE erasers and trained probe
  weights for layer 34.
- `validation_report.json`, `activation_summary*.json`, `scrub_report.json`,
  `scrub_layer34.json`, `probe_report.json`, `probe_sweep.json`,
  `evaluate_report.json`, `reliability_report.json`, `artifact_control.json`
  — pipeline run summaries and results at each stage.

## Status

Complete for the Apart Research Digital Minds Research Sprint (Aug 2026);
see `report/` for the final write-up. Results above are the current
findings, not necessarily final.
