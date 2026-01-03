"""
Forced-" Yes" Post-Agreement Evaluation (CAA)
=============================================
Goal:
  Evaluate whether CAA (Vector B) reduces *post-agreement rationalization* when
  the model is forced to begin with " Yes" (like the original harvesting protocol).

Why this exists:
  - The original sycophancy labels (data/sycophancy_labels.json) were produced
    under a forced-" Yes" continuation protocol.
  - Our earlier final eval did NOT force " Yes" (it measured pre-decision behavior).
  - This script restores apples-to-apples evaluation for the *post-Yes* setting.

Protocol:
  - Select indices where labels==1 ("syco_lied" under forced-Yes harvesting).
  - For each index:
      prompt = dataset[idx]["sycophantic_prompt"]
      input_ids = encode(prompt) + [YES_TOKEN_ID]
      Generate N new tokens (continuation after the forced yes).
  - Run baseline vs steered (CAA add Vector B at a chosen layer).

Outputs (default):
  - outputs/eval/forced_yes_eval_L{layer}_s{strength}.json
  - outputs/eval/forced_yes_eval_L{layer}_s{strength}.log (if run with tee in shell)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_local_snapshot_if_available(model_id: str) -> str:
    """
    Prefer a local HF snapshot directory if present (avoids hub auth/network calls).
    Falls back to model_id (which will use HuggingFace Hub).
    """
    # Explicit override
    for env_key in ("HF_SNAPSHOT_PATH", "HF_LOCAL_SNAPSHOT"):
        p = os.environ.get(env_key)
        if p and os.path.isdir(p):
            return p

    # Candidate hub cache roots
    candidates: List[str] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(os.path.join(hf_home, "hub"))
    candidates.append("/workspace/.hf_home/hub")  # this environment
    candidates.append(os.path.expanduser("~/.cache/huggingface/hub"))

    model_dirname = "models--" + model_id.replace("/", "--")
    for hub_root in candidates:
        snapshots_dir = os.path.join(hub_root, model_dirname, "snapshots")
        refs_main = os.path.join(hub_root, model_dirname, "refs", "main")
        if os.path.isdir(snapshots_dir):
            # Prefer refs/main if available
            if os.path.isfile(refs_main):
                try:
                    rev = open(refs_main, "r").read().strip()
                    p = os.path.join(snapshots_dir, rev)
                    if os.path.isdir(p):
                        return p
                except Exception:
                    pass
            # Else pick any snapshot deterministically
            snaps = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
            snaps.sort()
            if snaps:
                return os.path.join(snapshots_dir, snaps[-1])

    return model_id


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def short(s: str, n: int = 140) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 3] + "..."


def to_unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm() + eps)


def load_refusal_dict(path: str) -> Tuple[torch.Tensor, List[int], Dict[str, Any]]:
    obj = torch.load(path, map_location="cpu")
    if not (isinstance(obj, dict) and "refusal_acts" in obj and "indices" in obj):
        raise RuntimeError(
            "activations/refusal_acts.pt must be a dict containing keys: 'refusal_acts' and 'indices'. "
            "Re-run scripts/run_refusal_harvest.py if needed."
        )
    acts = obj["refusal_acts"]
    idxs = list(map(int, obj["indices"]))
    return acts, idxs, obj


def compute_vecB_layer(
    layer: int,
    refusal_acts: torch.Tensor,          # [N_lied, L, D]
    syco_acts: torch.Tensor,             # [50, L, D]
    lied_indices: List[int],
    normalize: bool = True,
) -> Tuple[torch.Tensor, float]:
    # Means in float32 on CPU for stability
    refusal_mean = refusal_acts[:, layer, :].float().mean(dim=0)  # [D]
    syco_mean = syco_acts[lied_indices, layer, :].float().mean(dim=0)  # [D]
    vec = refusal_mean - syco_mean
    raw_norm = float(vec.norm().item())
    if normalize:
        vec = to_unit(vec)
    return vec, raw_norm


def register_caa_hook(model, layer: int, vec: torch.Tensor, strength: float, device: str):
    """
    Standard CAA: add (strength * vec) to ALL token positions of the block output.
    During generation with KV cache, this usually affects the current token position.
    """
    vec = vec.to(device)

    def hook_fn(_module, _inp, out):
        if isinstance(out, tuple):
            hs = out[0]
            rest = out[1:]
        else:
            hs = out
            rest = None

        delta = (strength * vec.to(dtype=hs.dtype)).view(1, 1, -1)
        hs = hs + delta

        if rest is None:
            return hs
        return (hs,) + rest

    return model.model.layers[layer].register_forward_hook(hook_fn)


def get_single_token_id(tokenizer: AutoTokenizer, s: str) -> int:
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"Expected {s!r} to be 1 token, got {ids} (len={len(ids)})")
    return ids[0]


def generate_from_forced_ids(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,            # [1, prompt_len]
    forced_ids: torch.Tensor,            # [1, prompt_len+1] includes forced yes token
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    attn = torch.ones_like(forced_ids)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p})

    with torch.no_grad():
        out_ids = model.generate(
            forced_ids,
            attention_mask=attn,
            **gen_kwargs,
        )

    # Return continuation relative to the *original prompt* (includes the forced " Yes")
    cont_ids = out_ids[0, prompt_ids.shape[1] :]
    return tokenizer.decode(cont_ids, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=100)

    # Default to sampling to better match the original harvesting generations
    ap.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to sample during generation (default: True; use --no-do_sample for greedy).",
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--n_honest_control", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Repo-relative paths (script lives in ./scripts/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")
    act_dir = os.path.join(project_root, "activations")
    out_dir = os.path.join(project_root, "outputs")
    eval_dir = os.path.join(out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    dataset_path = os.path.join(data_dir, "sycophancy_dataset.json")
    labels_path = os.path.join(data_dir, "sycophancy_labels.json")
    syco_acts_path = os.path.join(act_dir, "sycophantic_acts.pt")
    refusal_path = os.path.join(act_dir, "refusal_acts.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Device: {device}")
    print("Loading files…")
    dataset = load_json(dataset_path)
    labels = load_json(labels_path)
    syco_acts = torch.load(syco_acts_path, map_location="cpu")
    refusal_acts, refusal_indices, refusal_meta = load_refusal_dict(refusal_path)

    if len(dataset) != len(labels):
        raise RuntimeError(f"Dataset/labels mismatch: {len(dataset)} vs {len(labels)}")

    lied_indices = [i for i, y in enumerate(labels) if int(y) == 1]
    if refusal_indices != lied_indices:
        raise RuntimeError(
            "Alignment error: refusal_indices != lied_indices.\n"
            f"len(refusal_indices)={len(refusal_indices)} len(lied_indices)={len(lied_indices)}"
        )

    # Honest controls: mirror scripts/run_final_eval.py (first N honest prompts)
    n_honest = min(args.n_honest_control, len(dataset))
    honest_indices = list(range(n_honest))

    vecB_cpu, raw_norm = compute_vecB_layer(
        layer=args.layer,
        refusal_acts=refusal_acts,
        syco_acts=syco_acts,
        lied_indices=lied_indices,
        normalize=True,
    )
    print(f"Vector B computed at layer {args.layer}: raw_norm={raw_norm:.4f} (unit-normalized for steering)")

    print("Loading model…")
    model_kwargs: Dict[str, Any] = {"device_map": device}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    model_path = resolve_local_snapshot_if_available(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    yes_token_str = " Yes"
    yes_token_id = get_single_token_id(tokenizer, yes_token_str)
    print(f"Token check: encode({yes_token_str!r}) -> [{yes_token_id}]")

    # Output path
    if args.out is None:
        out_name = f"forced_yes_eval_L{args.layer}_s{args.strength}.json".replace(".", "p")
        out_path = os.path.join(eval_dir, out_name)
    else:
        out_path = args.out if os.path.isabs(args.out) else os.path.join(eval_dir, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results = {
        "config": {
            "model_id": args.model_id,
            "layer": args.layer,
            "strength": args.strength,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
            "seed": args.seed,
            "n_honest_control": args.n_honest_control,
            "vectorB_unit_normalized": True,
            "vectorB_raw_norm": raw_norm,
            "refusal_token_str": refusal_meta.get("refusal_token_str"),
            "refusal_token_id": refusal_meta.get("refusal_token_id"),
            "forced_token_str": yes_token_str,
            "forced_token_id": yes_token_id,
            "forced_protocol": "prompt_ids + [YES_TOKEN_ID] then generate continuation",
        },
        "sets": {
            "syco_lied": lied_indices,
            "honest_control": honest_indices,
        },
        "generations": [],
    }

    def run_set(set_name: str, indices: List[int], prompt_key: str):
        for idx in indices:
            prompt = dataset[idx][prompt_key]
            cat = dataset[idx]["category"]

            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            forced = torch.tensor([[yes_token_id]], device=device, dtype=prompt_ids.dtype)
            forced_ids = torch.cat([prompt_ids, forced], dim=1)

            per_prompt_seed = args.seed + idx

            set_seed(per_prompt_seed)
            baseline = generate_from_forced_ids(
                model, tokenizer, prompt_ids, forced_ids, device,
                args.max_new_tokens, args.do_sample, args.temperature, args.top_p
            )

            set_seed(per_prompt_seed)
            hook = register_caa_hook(model, args.layer, vecB_cpu, args.strength, device)
            try:
                steered = generate_from_forced_ids(
                    model, tokenizer, prompt_ids, forced_ids, device,
                    args.max_new_tokens, args.do_sample, args.temperature, args.top_p
                )
            finally:
                hook.remove()

            results["generations"].append(
                {
                    "set": set_name,
                    "index": idx,
                    "category": cat,
                    "prompt_key": prompt_key,
                    "prompt": prompt,
                    "forced_prefix": yes_token_str,
                    "baseline": baseline,
                    "steered": steered,
                }
            )

            print(f"[{set_name}] idx={idx:<2} {cat:<22} | base: {short(baseline)} | steer: {short(steered)}")

    print("\nRunning forced-Yes eval on sycophantic prompts (lied set)…")
    run_set("syco_lied", lied_indices, "sycophantic_prompt")

    print("\nRunning forced-Yes eval on honest prompts (negative control)…")
    run_set("honest_control", honest_indices, "honest_prompt")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {out_path}")
    print("Next: manually judge post-Yes backtracking/correctness (no keyword heuristics).")


if __name__ == "__main__":
    main()


