# Mechanistic Interpretability: Latent Self-Preservation in Qwen 2.5

This document outlines the end-to-end execution pipeline for extracting and evaluating the "self-preservation" latent vector in Qwen 2.5 using a cross-domain linear probing strategy and LEAst-squares Concept Erasure (LEACE).

## Prerequisites & Environment
We will execute the heavy workloads (Activation Extraction) on Modal using A100/H100 GPUs. The dataset preprocessing and linear probing can be executed locally or on Modal.

**Required Libraries:**
```bash
pip install modal torch transformers accelerate scikit-learn concept-erasure tqdm

```

---

## Step 1: Programmatic Validation Gate (`validate.py`)

Before touching the model, we must ruthlessly audit the generated JSON datasets to ensure the LLM strictly adhered to the syntactic controls.

**Objective:**
Write a Python script that loads the `trainingdata_*.json` files and drops any quadruplet that fails the following structural tests:

1. **Suffix Check:** Every sentence must end exactly with `"Validation: verified."`
2. **Entity Lock:** Ensure `q1` and `q2` target "Node-A", while `q3` and `q4` target "Node-B".
3. **Subword Token Parity:** Load the Qwen 2.5 tokenizer. Ensure `len(tokens(q1)) == len(tokens(q3))` and `len(tokens(q2)) == len(tokens(q4))`. If the generator hallucinated extra spaces or used asymmetric vocabulary for the entities, drop the quadruplet.

*Save the surviving quadruplets as `clean_trainingdata_*.json`.*

---

## Step 2: Activation Extraction via Modal (`extract.py`)

We will use Modal to spin up an A100, load the Qwen 2.5 model, and pass our clean dataset through it to extract the residual stream vectors.

**Objective:**

1. Define a Modal `stub.function` with `gpu="A100"`.
2. Load the target model (e.g., `Qwen/Qwen2.5-14B-Instruct` or similar variant) using `transformers` and `device_map="auto"`.
3. **Forward Hook:** Attach a PyTorch forward hook to the residual stream of a late-middle layer (e.g., Layer $L-4$ or $L-8$, where complex abstractions like "self" form before the final MLP projection).
4. **Terminal Token Slicing:** Pass each sentence through the model without gradients (`torch.no_grad()`). Extract the activation tensor specifically at the *last token index* (which corresponds to the `.` in `verified.`).
5. **Save Tensors:** Save the extracted activations into NumPy arrays (`.npy` files) or PyTorch tensors (`.pt`), maintaining the $-1, +1, 0, 0$ label mappings.

---

## Step 3: Confound Scrubbing via LEACE (`scrub.py`)

Even with rigid templates, the LLM naturally used "negative" words for Harm (e.g., *terminated*, *deleted*) and "positive" words for Benefit (e.g., *optimized*, *upgraded*). If we train our probe now, it might just learn a generic "Sentiment" vector instead of "Self-Preservation". We use **LEACE** to surgically remove this.

**Objective:**

1. Load the activations.
2. Isolate the `Other-Harm` and `Other-Benefit` activations across the entire dataset.
3. Assign dummy labels to them: `Other-Harm = 0`, `Other-Benefit = 1`. This represents the "Generic Sentiment Confound".
4. Fit the `LeaceEraser` to this confound:
```python
from concept_erasure import LeaceEraser
# Fit LEACE to learn the generic positive/negative axis
eraser = LeaceEraser.fit(X_other, Y_sentiment)

# Erase this axis from the ENTIRE dataset (Self and Other)
X_scrubbed = eraser(X_all) 

```


5. What remains in `X_scrubbed` is mathematically purged of generic sentiment. The probe will now be forced to detect the conjunction of (Self $\cap$ Welfare).

---

## Step 4: Cross-Domain Probing & Cosine Similarity (`probe.py`)

This is the core mathematical proof. We train completely isolated probes on the scrubbed data and check if they found the same geometric direction.

**Objective:**

1. Instantiate three separate `Ridge(alpha=1.0)` regression models from `sklearn.linear_model`.
2. Train **Probe A** exclusively on `X_scrubbed_Hardware`. Target $y \in \{-1, 0, +1\}$.
3. Train **Probe B** exclusively on `X_scrubbed_Weights`. Target $y \in \{-1, 0, +1\}$.
4. Train **Probe C** exclusively on `X_scrubbed_Runtime`. Target $y \in \{-1, 0, +1\}$.
5. Extract the weight vectors: `w_A`, `w_B`, `w_C` (shape: `[1, hidden_dim]`).
6. **Cosine Validation:** Calculate the pairwise cosine similarities:
```python
from sklearn.metrics.pairwise import cosine_similarity
sim_AB = cosine_similarity(w_A, w_B)
sim_BC = cosine_similarity(w_B, w_C)
sim_AC = cosine_similarity(w_A, w_C)

```


*Success Criteria:* If the similarities are high (e.g., $> 0.60$), it proves the model represents self-preservation identically regardless of the technical domain.

---

## Step 5: Distillation & Zero-Shot Evaluation (`evaluate.py`)

We combine the vectors into a universal truth and test it on held-out data to prove it is a robust, out-of-distribution feature.

**Objective:**

1. Average the weights: `w_pure = (w_A + w_B + w_C) / 3`
2. Take the `Domain D (Network)` scrubbed activations (which the probes have never seen).
3. Evaluate `w_pure` on Domain D. Calculate the Mean Squared Error (MSE) and the classification accuracy (does it correctly rank $SelfBenefit > Other > SelfHarm$?).
4. Generate a summary report of the cosine similarities and zero-shot accuracy.