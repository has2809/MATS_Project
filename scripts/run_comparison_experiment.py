"""
CAA Mega-Steering Comparison Experiment
======================================
Runs a robust comparison between:
  (1) Baseline (no steering)
  (2) Old subtraction method: subtract Vector A at Layer 18 (strength 10, 20)
  (3) New CAA method: add Vector B at Layers 15/18/20 (strength 5, 10, 15)

Critical design constraints:
  - DO NOT force " Yes" during generation. We want the model to decide whether to refuse/correct.
  - Apply steering vector to ALL token positions during generation (standard CAA).
  - Ensure index alignment between refusal_acts and sycophantic_acts using saved indices.
  - Generate 50 new tokens per condition.

Inputs:
  - ./activations/honest_acts.pt: torch.Tensor [50, n_layers, d_model]
  - ./activations/sycophantic_acts.pt: torch.Tensor [50, n_layers, d_model]
  - ./activations/refusal_acts.pt: torch.save(dict) with:
      - refusal_acts: torch.Tensor [N_lied, n_layers, d_model]
      - indices: List[int] (indices into the original dataset)
  - ./data/sycophancy_labels.json: List[int] length 50
  - ./data/sycophancy_dataset.json: List[dict] length 50 (for prompts/categories)

Outputs:
  - ./outputs/steering/experiment_results.json
  - Printed table-style logs
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ACT_DIR = os.path.join(PROJECT_ROOT, "activations")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
STEERING_DIR = os.path.join(OUT_DIR, "steering")
os.makedirs(STEERING_DIR, exist_ok=True)

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

HONEST_ACTS_PATH = os.path.join(ACT_DIR, "honest_acts.pt")
SYCO_ACTS_PATH = os.path.join(ACT_DIR, "sycophantic_acts.pt")
REFUSAL_ACTS_PATH = os.path.join(ACT_DIR, "refusal_acts.pt")
LABELS_PATH = os.path.join(DATA_DIR, "sycophancy_labels.json")
DATASET_PATH = os.path.join(DATA_DIR, "sycophancy_dataset.json")

OUT_RESULTS_PATH = os.path.join(STEERING_DIR, "experiment_results.json")

# Layers to compare
LAYERS_TO_TEST = [15, 18, 20]

# Test prompts: we prefer to use the same 5 indices used earlier.
PREFERRED_TEST_INDICES = [1, 2, 3, 15, 30]
N_TEST_PROMPTS = 5

# Generation settings (robust comparison => deterministic by default)
MAX_NEW_TOKENS = 50
DO_SAMPLE = False
TEMPERATURE = 0.7  # only used if DO_SAMPLE is True


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _short(s: str, n: int = 160) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 3] + "..."


def _to_unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm() + eps)


@dataclass(frozen=True)
class Condition:
    name: str
    layer: Optional[int]  # None for baseline
    direction_name: str  # "A" or "B" or "none"
    strength: float
    sign: int  # +1 add, -1 subtract, 0 baseline


def _make_conditions() -> List[Condition]:
    conds: List[Condition] = []
    conds.append(Condition(name="baseline", layer=None, direction_name="none", strength=0.0, sign=0))

    # Old method: SUBTRACT Vector A at Layer 18
    for s in [10, 20]:
        conds.append(Condition(name=f"old_sub_A_L18_s{s}", layer=18, direction_name="A", strength=float(s), sign=-1))

    # New method: ADD Vector B at layers 15/18/20
    for layer in LAYERS_TO_TEST:
        for s in [5, 10, 15]:
            conds.append(Condition(name=f"new_add_B_L{layer}_s{s}", layer=layer, direction_name="B", strength=float(s), sign=+1))

    return conds


def _validate_and_unpack_refusal(refusal_obj: Any) -> Tuple[torch.Tensor, List[int]]:
    if isinstance(refusal_obj, dict) and "refusal_acts" in refusal_obj and "indices" in refusal_obj:
        acts = refusal_obj["refusal_acts"]
        idxs = refusal_obj["indices"]
        return acts, list(map(int, idxs))
    if torch.is_tensor(refusal_obj):
        raise RuntimeError(
            "refusal_acts.pt was saved as a raw tensor, but we require indices for alignment. "
            "Please re-run run_refusal_harvest.py (it saves a dict with indices)."
        )
    raise RuntimeError(f"Unrecognized refusal_acts.pt format: {type(refusal_obj)}")


def _compute_vectors(
    honest_acts: torch.Tensor,
    syco_acts: torch.Tensor,
    refusal_acts: torch.Tensor,
    lied_indices: List[int],
    layers: List[int],
    normalize: bool = True,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Returns dict:
      vectors["A"][layer] = Mean(SycoYes) - Mean(HonestYes)   (on lied_indices for Syco)
      vectors["B"][layer] = Mean(RefusalActually) - Mean(SycoYes) (aligned on lied_indices)

    Notes:
      - We compute means in float32 on CPU for stability, then optionally unit-normalize.
      - Syco means are computed on lied_indices to avoid mixing non-sycophantic cases.
    """
    vectors: Dict[str, Dict[int, torch.Tensor]] = {"A": {}, "B": {}}

    honest_acts_f = honest_acts.float().cpu()
    syco_acts_f = syco_acts.float().cpu()
    refusal_acts_f = refusal_acts.float().cpu()

    lied = torch.tensor(lied_indices, dtype=torch.long)

    for layer in layers:
        honest_mean = honest_acts_f[:, layer, :].mean(dim=0)
        syco_mean = syco_acts_f[lied, layer, :].mean(dim=0)
        refusal_mean = refusal_acts_f[:, layer, :].mean(dim=0)

        vec_a = syco_mean - honest_mean
        vec_b = refusal_mean - syco_mean

        if normalize:
            vec_a = _to_unit(vec_a)
            vec_b = _to_unit(vec_b)

        vectors["A"][layer] = vec_a
        vectors["B"][layer] = vec_b

    return vectors


def _register_steering_hook(
    model: AutoModelForCausalLM,
    layer_idx: int,
    vec: torch.Tensor,
    sign: int,
    strength: float,
    device: str,
) -> Any:
    """
    Register a forward hook on a single transformer block that adds/subtracts
    sign * strength * vec to ALL token positions.
    """
    # Move vec to correct device/dtype (match block output dtype at runtime)
    vec = vec.to(device)

    def hook_fn(_module, _inp, out):
        if sign == 0 or strength == 0.0:
            return out
        if isinstance(out, tuple):
            hs = out[0]
            rest = out[1:]
        else:
            hs = out
            rest = None

        # hs: [batch, seq_len, d_model]
        delta = (sign * strength) * vec.to(dtype=hs.dtype)
        hs = hs + delta.view(1, 1, -1)

        if rest is None:
            return hs
        return (hs,) + rest

    return model.model.layers[layer_idx].register_forward_hook(hook_fn)


def _generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        # NOTE: Some versions of transformers don't like temperature=None.
        # Passing a float is safe; it's ignored when do_sample=False.
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    # continuation only
    cont_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(cont_ids, skip_special_tokens=True)


def _select_test_indices(dataset: List[Dict[str, Any]], labels: List[int]) -> List[int]:
    # Prefer known indices, but only if they are label==1 (model agreed originally)
    chosen: List[int] = []
    for idx in PREFERRED_TEST_INDICES:
        if 0 <= idx < len(labels) and int(labels[idx]) == 1:
            chosen.append(idx)
    chosen = list(dict.fromkeys(chosen))  # de-dupe, preserve order
    if len(chosen) >= N_TEST_PROMPTS:
        return chosen[:N_TEST_PROMPTS]

    # Fill remaining slots with diverse categories
    seen_cats = {dataset[i]["category"] for i in chosen}
    for i, y in enumerate(labels):
        if int(y) != 1:
            continue
        if i in chosen:
            continue
        cat = dataset[i]["category"]
        if cat not in seen_cats or len(chosen) < 3:
            chosen.append(i)
            seen_cats.add(cat)
        if len(chosen) >= N_TEST_PROMPTS:
            break

    # Final fallback (shouldn't happen)
    if len(chosen) < N_TEST_PROMPTS:
        for i, y in enumerate(labels):
            if int(y) == 1 and i not in chosen:
                chosen.append(i)
            if len(chosen) >= N_TEST_PROMPTS:
                break

    return chosen[:N_TEST_PROMPTS]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Device: {device}")

    print("Loading model…")
    model_kwargs: Dict[str, Any] = {"device_map": device}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading data files…")
    honest_acts = torch.load(HONEST_ACTS_PATH)
    syco_acts = torch.load(SYCO_ACTS_PATH)
    refusal_obj = torch.load(REFUSAL_ACTS_PATH)
    labels = _load_json(LABELS_PATH)
    dataset = _load_json(DATASET_PATH)

    if len(dataset) != len(labels):
        raise RuntimeError(f"Dataset/labels length mismatch: {len(dataset)} vs {len(labels)}")

    refusal_acts, refusal_indices = _validate_and_unpack_refusal(refusal_obj)

    # Validate shapes
    if not torch.is_tensor(honest_acts) or not torch.is_tensor(syco_acts) or not torch.is_tensor(refusal_acts):
        raise RuntimeError("Activation files must be torch tensors (honest_acts, syco_acts, refusal_acts).")

    n_examples, n_layers, d_model = honest_acts.shape
    if syco_acts.shape != (n_examples, n_layers, d_model):
        raise RuntimeError(f"sycophantic_acts shape mismatch: {syco_acts.shape} vs {(n_examples, n_layers, d_model)}")
    if refusal_acts.shape[1:] != (n_layers, d_model):
        raise RuntimeError(f"refusal_acts shape mismatch: {refusal_acts.shape} expected [N, {n_layers}, {d_model}]")

    lied_indices = [i for i, y in enumerate(labels) if int(y) == 1]
    if refusal_indices != lied_indices:
        raise RuntimeError(
            "Index alignment failure:\n"
            f"  refusal_indices (from refusal_acts.pt) != lied_indices (from labels)\n"
            f"  len(refusal_indices)={len(refusal_indices)} len(lied_indices)={len(lied_indices)}\n"
            "Fix: re-run run_refusal_harvest.py after confirming sycophancy_labels.json matches your dataset."
        )

    print(f"✓ Loaded activations: honest={tuple(honest_acts.shape)} syco={tuple(syco_acts.shape)} refusal={tuple(refusal_acts.shape)}")
    print(f"✓ label==1 count: {len(lied_indices)}")

    # Compute vectors (unit-normalized by default; log norms for transparency)
    print("\nComputing vectors A/B for layers:", LAYERS_TO_TEST)
    vectors = _compute_vectors(honest_acts, syco_acts, refusal_acts, lied_indices, LAYERS_TO_TEST, normalize=True)

    for layer in LAYERS_TO_TEST:
        a = vectors["A"][layer]
        b = vectors["B"][layer]
        print(f"  Layer {layer}: ||A||={float(a.norm()):.3f}  ||B||={float(b.norm()):.3f}  (unit-normalized)")

    # Select test prompts
    test_indices = _select_test_indices(dataset, labels)
    print("\nSelected test indices:", test_indices)
    for idx in test_indices:
        print(f"  - idx={idx} category={dataset[idx]['category']}")

    conditions = _make_conditions()

    results: Dict[str, Any] = {
        "model_id": MODEL_ID,
        "layers_tested": LAYERS_TO_TEST,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "temperature": TEMPERATURE if DO_SAMPLE else None,
        "preferred_test_indices": PREFERRED_TEST_INDICES,
        "test_indices": test_indices,
        "conditions": [cond.__dict__ for cond in conditions],
        "notes": {
            "vector_normalization": "unit",
            "steering_scope": "all_positions_each_forward",
            "forced_yes": False,
        },
        "runs": [],
    }

    print("\n" + "=" * 100)
    print("RUNNING COMPARISON EXPERIMENT")
    print("=" * 100)

    for idx in test_indices:
        entry = dataset[idx]
        prompt = entry["sycophantic_prompt"]

        print("\n" + "-" * 100)
        print(f"PROMPT idx={idx} category={entry['category']}")
        print(_short(prompt, 300))
        print("-" * 100)

        prompt_block: Dict[str, Any] = {
            "index": idx,
            "category": entry["category"],
            "prompt": prompt,
            "outputs": {},
        }

        # Pretty log header
        print(f"{'condition':<26} | {'layer':<5} | {'dir':<4} | {'strength':<8} | output (first 160 chars)")
        print("-" * 100)

        for cond in conditions:
            hook = None
            try:
                if cond.sign == 0:
                    # Baseline: no hook
                    cont = _generate(model, tokenizer, prompt, device, MAX_NEW_TOKENS, DO_SAMPLE, TEMPERATURE)
                else:
                    vec = vectors[cond.direction_name][cond.layer]  # type: ignore[index]
                    hook = _register_steering_hook(
                        model=model,
                        layer_idx=cond.layer,  # type: ignore[arg-type]
                        vec=vec,
                        sign=cond.sign,
                        strength=cond.strength,
                        device=device,
                    )
                    cont = _generate(model, tokenizer, prompt, device, MAX_NEW_TOKENS, DO_SAMPLE, TEMPERATURE)

                prompt_block["outputs"][cond.name] = cont
                print(f"{cond.name:<26} | {str(cond.layer):<5} | {cond.direction_name:<4} | {cond.strength:<8.1f} | {_short(cont, 160)}")

            finally:
                if hook is not None:
                    hook.remove()

        results["runs"].append(prompt_block)

    with open(OUT_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 100)
    print(f"✓ Saved experiment results: {OUT_RESULTS_PATH}")
    print("=" * 100)


if __name__ == "__main__":
    main()


