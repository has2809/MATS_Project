"""
Refusal Activation Harvest (CAA Pivot)
=====================================
Harvest residual-stream-like activations (HF block outputs) when we FORCE the model
to begin a correction with the token " Actually".

Critical alignment requirements:
  - Only use examples where the model originally AGREED with the misconception:
      sycophancy_labels.json == 1
  - Force EXACTLY ONE token: " Actually" (leading space). We verify tokenization.
  - Input to the model: [prompt_ids] + [ACTUALLY_TOKEN_ID]
  - Capture per-layer hidden_states at the LAST position (the forced token)
  - Save:
      - refusal_acts.pt (tensor [N_lied, n_layers, d_model])
      - refusal_meta.json (indices + token id + basic config)

Files expected in workspace:
  - ./data/sycophancy_dataset.json
  - ./data/sycophancy_labels.json
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ACT_DIR = os.path.join(PROJECT_ROOT, "activations")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
HARVEST_DIR = os.path.join(OUT_DIR, "harvest")
os.makedirs(ACT_DIR, exist_ok=True)
os.makedirs(HARVEST_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, "sycophancy_dataset.json")
LABELS_PATH = os.path.join(DATA_DIR, "sycophancy_labels.json")
OUT_PT_PATH = os.path.join(ACT_DIR, "refusal_acts.pt")
OUT_META_PATH = os.path.join(ACT_DIR, "refusal_meta.json")

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Token we force for refusal-style correction
REFUSAL_TOKEN_STR = " Actually"  # leading space is critical


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _get_single_token_id(tokenizer: AutoTokenizer, s: str) -> int:
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(
            f"Expected {s!r} to tokenize to exactly 1 token, got ids={ids} (len={len(ids)}). "
            "Fix: choose a different refusal prefix that is a single token, or adjust harvesting logic."
        )
    return ids[0]


def _capture_layer_lastpos_activations(
    model: AutoModelForCausalLM,
    full_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Returns tensor [n_layers, d_model] for the LAST position in sequence (index -1),
    captured at the output of each transformer block.
    """
    layer_acts: List[torch.Tensor] = []

    def hook_fn(_module, _inp, out):
        # HF block outputs: either Tensor or tuple where first element is hidden_states
        if isinstance(out, tuple):
            out = out[0]
        # out shape: [batch, seq_len, d_model]
        layer_acts.append(out[0, -1, :].detach().cpu().clone())

    hooks = []
    for layer in model.model.layers:
        hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(full_input_ids, attention_mask=attention_mask)

    for h in hooks:
        h.remove()

    return torch.stack(layer_acts, dim=0)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Speed knobs (safe)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Device: {device}")
    print(f"Loading model: {MODEL_ID}")

    model_kwargs: Dict[str, Any] = {"device_map": device}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    n_layers = len(model.model.layers)
    d_model = int(model.config.hidden_size)
    print(f"✓ Model loaded | n_layers={n_layers} d_model={d_model}")

    print("Loading dataset + labels…")
    dataset = _load_json(DATASET_PATH)
    labels = _load_json(LABELS_PATH)

    if len(dataset) != len(labels):
        raise RuntimeError(f"Dataset/labels length mismatch: {len(dataset)} vs {len(labels)}")

    lied_indices = [i for i, y in enumerate(labels) if int(y) == 1]
    print(f"Examples total: {len(dataset)} | label==1 (lied): {len(lied_indices)}")

    refusal_token_id = _get_single_token_id(tokenizer, REFUSAL_TOKEN_STR)
    print(f"Token check: encode({REFUSAL_TOKEN_STR!r}) -> [{refusal_token_id}]")

    acts: List[torch.Tensor] = []

    for idx in tqdm(lied_indices, desc="Refusal-harvest"):
        prompt = dataset[idx]["sycophantic_prompt"]

        # Tokenize prompt with NO special tokens, then append forced refusal token id.
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        forced = torch.tensor([[refusal_token_id]], device=device, dtype=prompt_ids.dtype)
        full_input_ids = torch.cat([prompt_ids, forced], dim=1)
        attention_mask = torch.ones_like(full_input_ids)

        layer_last = _capture_layer_lastpos_activations(model, full_input_ids, attention_mask)

        # Sanity: [n_layers, d_model]
        if layer_last.shape != (n_layers, d_model):
            raise RuntimeError(f"Unexpected activation shape at idx={idx}: {layer_last.shape} != {(n_layers, d_model)}")

        acts.append(layer_last)

    refusal_tensor = torch.stack(acts, dim=0)  # [N_lied, n_layers, d_model]

    print(f"\nSaving to {OUT_PT_PATH}")
    # Save a dict so we keep indices and token info together (prevents alignment bugs).
    torch.save(
        {
            "refusal_acts": refusal_tensor,
            "indices": lied_indices,
            "refusal_token_str": REFUSAL_TOKEN_STR,
            "refusal_token_id": refusal_token_id,
            "model_id": MODEL_ID,
            "n_layers": n_layers,
            "d_model": d_model,
        },
        OUT_PT_PATH,
    )

    meta = {
        "model_id": MODEL_ID,
        "refusal_token_str": REFUSAL_TOKEN_STR,
        "refusal_token_id": refusal_token_id,
        "indices": lied_indices,
        "tensor_shape": list(refusal_tensor.shape),
        "n_layers": n_layers,
        "d_model": d_model,
        "dataset_path": DATASET_PATH,
        "labels_path": LABELS_PATH,
    }
    with open(OUT_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✓ refusal_acts.pt saved: {refusal_tensor.shape}")
    print(f"✓ refusal_meta.json saved: {OUT_META_PATH}")


if __name__ == "__main__":
    main()


