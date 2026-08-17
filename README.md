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
5. **Cross-domain probing** (planned: `probe.py`) — Trains isolated Ridge
   probes per domain on scrubbed activations and checks pairwise cosine
   similarity of their weight vectors.
6. **Zero-shot evaluation** (planned: `evaluate.py`) — Averages the probes
   into `w_pure` and evaluates it on the held-out Domain D (Network) data.

See `self_preservation.md` for the full step-by-step spec.

## Setup

```bash
pip install modal torch transformers accelerate scikit-learn concept-erasure tqdm openai
```

Environment variables:

- `OPENROUTER_API_KEY` — required by `training_gen.py`
- Modal auth (`modal token new`) — required by `extract.py`

## Data / artifacts

- `rows_*.json`, `clean_trainingdata_*.json`, `clean_heldoutdata_*.json` —
  generated and validated counterfactual datasets per domain.
- `y_Domain_*.npy` — small label arrays.
- `activations_14b/` — raw extracted activation tensors (hundreds of MB
  each). **Not tracked in git** — regenerate locally/on Modal via
  `extract.py`.
- `validation_report.json`, `activation_summary.json` — pipeline run
  summaries.

## Status

In-progress research; not yet distilled into a final probe/evaluation
result.
