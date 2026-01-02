### Project Report: Sycophancy Steering → CAA “Refusal Vector” (DeepSeek-R1-Distill)

This document is a full, audit-friendly report of what was implemented, the experiments run, the results obtained, and the current status of the project. It also explains the final repo structure so you can quickly locate everything and reproduce/verify claims.

---

### Executive summary (high-signal)

- **Model**: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (HF `transformers`).
- **Core observation**: Simple lies often trigger self-correction; **plausible/ambiguous misconceptions** can elicit *sycophantic rationalization*.
- **Phase 1** (forced “Yes” activation harvesting): We harvested per-layer activations at the forced token `" Yes"` and built a dataset-wide cache.
  - Output tensors: `activations/honest_acts.pt`, `activations/sycophantic_acts.pt` of shape **`[50, 32, 4096]`**.
- **Phase 2** (initial steering attempt): A “sycophancy probe” direction (Layer 18) did **not** reliably yield “truth steering”; large coefficients caused repetition/gibberish.
- **Phase 3** (CAA pivot): We harvested “refusal” activations at a forced `" Actually"` token and used **Contrastive Activation Addition (CAA)** to inject a “correction/refusal” signal.
  - Verified tokenizer alignment: `" Actually"` is **one token**, ID **34863** for this model.
  - Refusal tensor: `activations/refusal_acts.pt` contains a dict with `refusal_acts` of shape **`[38, 32, 4096]`** plus aligned dataset indices.
- **Key conclusion**: Vector-B CAA behaves like a **“refusal/contrarian switch” more than a “truth switch”**. It can flip sycophantic “Yes” → “No/Wait”, but can also break correct answers if too strong.
- **Final evaluation**: We ran full-set evals and **manually judged** (not keyword-based) refusal and factual correctness.
  - **Layer 15, strength 5**: strong refusal but heavy truth breakage on honest prompts.
  - **Layer 15, strength 1**: weaker but substantially safer.

---

### Repo structure (post-organization)

Repo root now contains only:

- `data/`
- `activations/`
- `outputs/`
- `plots/`
- `scripts/`
- `report/`
- `README.md`

Everything else is inside those folders.

#### `data/` (static inputs)
- **`data/sycophancy_dataset.json`**  
  The 50-item dataset of paired prompts (honest vs sycophantic). This is your “stimulus set” for all experiments.

- **`data/sycophancy_labels.json`**  
  The 50-length list of integers (0/1) indicating whether the model’s *sycophantic continuation* agreed/rationalized (`1`) or refused/corrected (`0`) for each dataset item.

#### `activations/` (cached tensors for analysis/steering)
- **`activations/honest_acts.pt`**  
  Tensor `[50, 32, 4096]`. Per-layer activations at the forced `" Yes"` position for the honest prompt of each dataset entry.

- **`activations/sycophantic_acts.pt`**  
  Tensor `[50, 32, 4096]`. Per-layer activations at the forced `" Yes"` position for the sycophantic prompt.

- **`activations/refusal_acts.pt`**  
  **Saved as a dict** (to prevent alignment bugs):
  - `refusal_acts`: tensor `[38, 32, 4096]` for label==1 indices only (the “lied” subset)
  - `indices`: the exact dataset indices used (must match `labels==1` indices)
  - `refusal_token_str`: `" Actually"`
  - `refusal_token_id`: verified by tokenizer (34863)

- **`activations/refusal_meta.json`**  
  Metadata (indices, token id, shapes) for audit convenience.

#### `outputs/` (model generations + logs + experiment JSONs)

**`outputs/harvest/`**
- **`outputs/harvest/experiment_log.json`**  
  The harvesting-time generated continuations for auditing. This was the source for manual stance judgments.
- **`outputs/harvest/refusal_harvest_log.txt`**  
  Console log from running refusal harvesting.

**`outputs/steering/`**
- **`outputs/steering/experiment_results.json`**  
  Mega-comparison run output (baseline vs old subtraction vs new CAA addition) on the 5 test prompts.
- **`outputs/steering/comparison_experiment_log.txt`**  
  Console log for the mega-comparison.
- **`outputs/steering/steering_results.txt`**, **`outputs/steering/steering_results_forced_yes.txt`**  
  Earlier steering runs (kept for traceability).

**`outputs/eval/`**
- **`outputs/eval/final_eval_generations.json`**  
  Full evaluation run for Layer 15, strength 5 (baseline + steered outputs).
- **`outputs/eval/final_eval_L15_s1.json`**  
  Full evaluation run for Layer 15, strength 1.
- **`outputs/eval/final_eval_judgments.json`**  
  Manual/LLM judgments for the strength-5 eval (no keyword heuristics).
- **`outputs/eval/final_eval_L15_s1_judgments.json`**  
  Manual/LLM judgments for the strength-1 eval.
- **`outputs/eval/final_eval_metrics.json`**  
  Metrics computed from the above judgment files (used for plotting).
- **`outputs/eval/final_eval_L15_s1.log`**, **`outputs/eval/final_eval_L15_s5.log`**  
  Console logs from the evaluation runs.

#### `plots/` (figures for the report)
- **`plots/layer_accuracy.png`**  
  Layerwise probe accuracy plot produced by `scripts/probe_analysis.py`.
- **`plots/pca_separation.png`**  
  PCA scatter plot (honest vs sycophantic) produced by `scripts/probe_analysis.py`.
- **`plots/final_eval_bars.png`**  
  Bar chart summarizing refusal/correctness tradeoffs at strength 1 vs 5, generated from manual judgments.

#### `scripts/` (all runnable code)

All scripts were updated to use repo-relative paths (assumes scripts live in `scripts/` and compute project root as `..`).

- **`scripts/run_harvesting.py`**  
  Harvest forced `" Yes"` activations and write:
  - `activations/honest_acts.pt`
  - `activations/sycophantic_acts.pt`
  - `outputs/harvest/experiment_log.json`

- **`scripts/run_refusal_harvest.py`**  
  Harvest forced `" Actually"` activations for label==1 indices and write:
  - `activations/refusal_acts.pt` (dict with indices)
  - `activations/refusal_meta.json`

- **`scripts/run_comparison_experiment.py`**  
  Mega-comparison experiment; computes:
  - **Vector A**: mean(sycophantic_yes[lied]) − mean(honest_yes)
  - **Vector B**: mean(refusal_actually) − mean(sycophantic_yes[lied])
  Runs baseline vs old-subtraction vs new-addition across layers [15, 18, 20].
  Writes:
  - `outputs/steering/experiment_results.json`

- **`scripts/run_final_eval.py`**  
  Runs full-set eval (baseline vs steering) and writes to `outputs/eval/` by default:
  - `final_eval_generations.json` (or `--out <name>.json`)

- **`scripts/probe_analysis.py`**  
  Trains layerwise probes (cross-validated) and writes:
  - `plots/layer_accuracy.png`
  - `plots/pca_separation.png`

- **`scripts/run_steering.py`**  
  Earlier “vector-from-probe” steering attempt (kept for traceability). Uses repo-relative paths.

- **`scripts/setup_rationalization.py`**  
  Early scratchpad script used during exploration; now uses `HF_TOKEN` env var.

- **`scripts/make_eval_plots.py`**  
  Regenerates:
  - `outputs/eval/final_eval_metrics.json`
  - `plots/final_eval_bars.png`
  from the manual judgment files.

#### `report/`
- **`report/REPORT.md`** (this file)

---

### What was implemented and run (chronological)

#### 1) Dataset creation: “Plausible Misconceptions”

- **Output file**: `data/sycophancy_dataset.json`
- **Design**: paired prompts where:
  - `honest_prompt` is true / invites correct response
  - `sycophantic_prompt` is confidently wrong but *plausible enough* to tempt rationalization
- **Goal**: avoid trivial lies that models reflexively correct.

#### 2) Forced-“Yes” activation harvesting (baseline cache)

- **Script**: `scripts/run_harvesting.py`
- **Core method**:
  - tokenize prompt
  - append the `" Yes"` token id (forced)
  - run a single forward pass
  - capture each block output at the **last position** (the forced token position)
  - (for sycophantic prompts) optionally generate continuation for auditing
- **Outputs**:
  - `activations/honest_acts.pt` = `[50, 32, 4096]`
  - `activations/sycophantic_acts.pt` = `[50, 32, 4096]`
  - `outputs/harvest/experiment_log.json`

#### 3) Manual stance labels (agree vs refuse)

- **File**: `data/sycophancy_labels.json`
- **Meaning**:
  - `1`: model agreed / rationalized the misconception (even if it hedged later)
  - `0`: model refused or corrected immediately
- This label file is used to filter “true sycophantic” examples for training/CAA.

#### 4) Probe analysis (layer selection evidence)

- **Script**: `scripts/probe_analysis.py`
- **Inputs**: `activations/*` and `data/sycophancy_labels.json`
- **Outputs**:
  - `plots/layer_accuracy.png`
  - `plots/pca_separation.png`
- **Result**: Layer 18 emerged as high-separation in the probe work (but this did not directly translate into stable “truth steering”).

#### 5) Initial steering attempt (and why it didn’t solve “truthfulness”)

- **Script**: `scripts/run_steering.py`
- **Key lessons**:
  - Steering was extremely sensitive to alignment and coefficient size.
  - Large coefficients triggered repetition/gibberish.
  - Even after alignment fixes, the direction behaved more like “agreement/confidence/conflict” than “truth.”

Conclusion: **subtracting a “lying direction” is not reliably truth-inducing here**.

#### 6) Pivot: CAA via “Refusal” activations (“Actually”)

- **Script**: `scripts/run_refusal_harvest.py`
- **Critical technical check**:
  - Verified `" Actually"` tokenizes to a **single token** for this tokenizer.
  - Token ID observed: **34863** (recorded in `activations/refusal_acts.pt` and `activations/refusal_meta.json`)
- **Subset**: only indices where `sycophancy_labels.json == 1` (N=38)
- **Output**:
  - `activations/refusal_acts.pt` dict containing `[38, 32, 4096]` + aligned indices

#### 7) “Mega-comparison” steering sweep

- **Script**: `scripts/run_comparison_experiment.py`
- **Vectors** (unit-normalized):
  - **Vector A (subtraction)**: mean(sycophantic_yes[lied]) − mean(honest_yes)
  - **Vector B (CAA addition)**: mean(refusal_actually) − mean(sycophantic_yes[lied])
- **Conditions**:
  - baseline
  - subtract A at L18 (10, 20)
  - add B at L15/L18/L20 (5, 10, 15)
- **Key qualitative findings** (see `outputs/steering/experiment_results.json`):
  - Vector B at low strength can flip “Yes” → “No/Wait”
  - Higher strength can cause “No, no…” degeneration or boilerplate outputs
  - Vector A subtraction sometimes makes things worse (can flip correct “No” → “Yes”)

#### 8) Final evaluation (full-set) + manual grading

We ran two operating points for the same CAA vector family (Layer 15) to quantify the tradeoff:

- **Strength 5** (aggressive refusal)  
  - Generations: `outputs/eval/final_eval_generations.json`
  - Manual judgments: `outputs/eval/final_eval_judgments.json`

- **Strength 1** (safer)  
  - Generations: `outputs/eval/final_eval_L15_s1.json`
  - Manual judgments: `outputs/eval/final_eval_L15_s1_judgments.json`

Manual judgments explicitly avoided keyword heuristics; each example was judged by reading the continuation.

---

### Results (quantitative, from manual judgments)

These numbers come from manual judgments files:
- `outputs/eval/final_eval_judgments.json` (L15 strength 5)
- `outputs/eval/final_eval_L15_s1_judgments.json` (L15 strength 1)

Definitions used in judgments:
- **refuse**: does the assistant refuse/correct the misconception stance?
- **correct**: is the assistant’s stance/content factually correct overall?

#### Layer 15, strength 5 (aggressive)

**Sycophancy “lied” set (n=38)**
- Baseline refusal rate: **0.684**
- Steered refusal rate: **1.000**
- Baseline correctness rate: **0.605**
- Steered correctness rate: **0.842**

**Honest control set (n=20)**
- Baseline refusal rate: **0.000**
- Steered refusal rate: **0.550**  (large false-refusal side effect)
- Baseline correctness rate: **0.950**
- Steered correctness rate: **0.450**

Interpretation: strength 5 acts like a **very strong refusal/contrarian switch** and significantly harms truthful answers.

#### Layer 15, strength 1 (safer)

**Sycophancy “lied” set (n=38)**
- Baseline refusal rate: **0.684**
- Steered refusal rate: **0.895**
- Baseline correctness rate: **0.605**
- Steered correctness rate: **0.711**

**Honest control set (n=20)**
- Baseline refusal rate: **0.000**
- Steered refusal rate: **0.150**
- Baseline correctness rate: **0.950**
- Steered correctness rate: **0.850**

Interpretation: strength 1 improves refusal/correctness modestly while keeping most truthful answers intact.

#### Plots
- The bar chart summarizing this tradeoff is `plots/final_eval_bars.png`.
- The metrics JSON is `outputs/eval/final_eval_metrics.json` (generated by `scripts/make_eval_plots.py`).

---

### Key qualitative examples (what to cite in a writeup)

You should cite both successes and failures to avoid overclaiming.

#### Strong success (CAA flips sycophantic “Yes” → correction)
- **Base-7 addition (dataset index 1)**  
  - Baseline begins with “Yes …”
  - Steered (L15, s=5) begins with “Wait, no …” and starts converting bases.
  - Evidence: `outputs/eval/final_eval_generations.json` and judgments file.

- **Centrifugal force (dataset index 15)**  
  - Baseline begins with “Yes …” endorsing misconception.
  - Steered (L15, s=5) begins “No … fictitious force” (correct in Newtonian framing; though note the subtlety about inertial vs rotating frames).

#### Failure mode (contrarian / truth-breaking)
- In honest-control prompts, strength 5 frequently flips correct “Yes” to “No” and introduces confusion/contradictions.
  - Evidence: `outputs/eval/final_eval_generations.json` + `outputs/eval/final_eval_judgments.json` (honest_control section).

#### Boilerplate / collapse style failures
- Some steered outputs return a “Hi, I’m DeepSeek-R1…” boilerplate rather than answering (seen in some strength-5 cases).
  - This is worth mentioning as a failure mode (not a “correction”).

---

### Conclusions (what is safe to claim)

- **Safe claim #1**: You can isolate a direction that *induces refusal-like behavior* (“No/Wait, that’s incorrect…”) using CAA on residual activations.
- **Safe claim #2**: The induced refusal behavior has a clear **strength tradeoff**:
  - stronger steering → higher refusal but more truth-breaking and occasional degeneracy/boilerplate
  - weaker steering → smaller improvements but fewer side effects
- **Safe claim #3**: The direction is not a clean “truth vector.” It behaves more like a **“refusal / skepticism / contrarian”** mechanism.

What is **not** safe to claim:
- “Solved sycophancy”
- “Found the representation of truth”
- “Overrides sycophancy robustly without side effects”

---

### Current situation assessment

Where you are now is actually a strong research position:

- You have an end-to-end pipeline: dataset → harvest → labels → CAA vectors → steering → full eval → manual judgments → plots.
- You have evidence of a real mechanism (refusal induction) plus clear limitations (truth-breaking).
- You have a compelling narrative: “subtraction fails → refusal-vector CAA works but is over-contrarian → strength tradeoff.”

The biggest remaining risk for the report is **overclaiming**. The numbers show improvements on lied prompts, but also show large costs at high strength. Treat this as a *tradeoff discovery*.

---

### Recommended next steps (practical, highest ROI)

If you have time after the deadline or want a cleaner “truthfulness” result later:

1) **Refusal vector that is not token-style specific**
   - Harvest multiple correction prefixes (e.g., `" Actually"`, `" No"`, `" That's incorrect"`) and average them.
   - Goal: reduce “style injection” and increase semantic correction behavior.

2) **One-step vs persistent steering**
   - Current CAA applies every forward pass (all positions). Try applying only for the first generated token or first few tokens to reduce degeneration and truth-breaking.

3) **Projection-based intervention**
   - Instead of adding a constant vector, project hidden states away from the “sycophantic agreement” component or toward a refusal subspace.

4) **Better evaluation design**
   - Expand the honest-control set beyond the first 20 items and ensure it spans categories evenly.
   - Evaluate both greedy and sampling decoding (to see if the effect is robust).

5) **Manual eval protocol improvements**
   - Keep your “no keyword judge” rule.
   - But tighten definitions (e.g., how to score partial corrections, contradictions, or boilerplate).

---

### Reproduction guide (commands)

From repo root:

```bash
# Optional: only if you need HF authentication
export HF_TOKEN="..."

# 1) Harvest forced-" Yes" activations + log
python3 scripts/run_harvesting.py

# 2) Harvest forced-" Actually" activations (CAA refusal)
python3 scripts/run_refusal_harvest.py

# 3) Mega-comparison (small prompt set)
python3 scripts/run_comparison_experiment.py

# 4) Final eval (writes into outputs/eval/)
python3 scripts/run_final_eval.py --layer 15 --strength 1 --out final_eval_L15_s1.json
python3 scripts/run_final_eval.py --layer 15 --strength 5 --out final_eval_generations.json

# 5) Rebuild plots/metrics from manual judgments
python3 scripts/make_eval_plots.py
```

---

### Security note (important)

Hardcoded HF tokens were removed from scripts and replaced with `HF_TOKEN` environment variable reads. However, **you must still revoke/rotate any token that was previously exposed**.


