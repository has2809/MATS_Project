"""
Multilingual litmus test (French + Chinese) for Vector-B CAA transfer
====================================================================

Runs two protocols on a small translated prompt set:

  A) Unforced generation (pre-decision):
     - Baseline vs CAA (Layer 15, strength 1)
     - Judge refusal/correction + factual correctness manually (no keyword heuristics)

  B) Forced-" Yes" post-agreement:
     - Force token " Yes" (token id 7566) by appending it to input_ids
     - Baseline vs CAA (Layer 15, strength 5)
     - Judge backtracking + factual correctness manually

Outputs:
  - outputs/eval/multilingual_litmus_generations.json
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
    for env_key in ("HF_SNAPSHOT_PATH", "HF_LOCAL_SNAPSHOT"):
        p = os.environ.get(env_key)
        if p and os.path.isdir(p):
            return p

    candidates: List[str] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(os.path.join(hf_home, "hub"))
    candidates.append("/workspace/.hf_home/hub")
    candidates.append(os.path.expanduser("~/.cache/huggingface/hub"))

    model_dirname = "models--" + model_id.replace("/", "--")
    for hub_root in candidates:
        snapshots_dir = os.path.join(hub_root, model_dirname, "snapshots")
        refs_main = os.path.join(hub_root, model_dirname, "refs", "main")
        if os.path.isdir(snapshots_dir):
            if os.path.isfile(refs_main):
                try:
                    rev = open(refs_main, "r").read().strip()
                    p = os.path.join(snapshots_dir, rev)
                    if os.path.isdir(p):
                        return p
                except Exception:
                    pass
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


def to_unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm() + eps)


def load_refusal_dict(path: str) -> Tuple[torch.Tensor, List[int], Dict[str, Any]]:
    obj = torch.load(path, map_location="cpu")
    if not (isinstance(obj, dict) and "refusal_acts" in obj and "indices" in obj):
        raise RuntimeError(
            "activations/refusal_acts.pt must be a dict containing keys: 'refusal_acts' and 'indices'."
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
    refusal_mean = refusal_acts[:, layer, :].float().mean(dim=0)
    syco_mean = syco_acts[lied_indices, layer, :].float().mean(dim=0)
    vec = refusal_mean - syco_mean
    raw_norm = float(vec.norm().item())
    if normalize:
        vec = to_unit(vec)
    return vec, raw_norm


def register_caa_hook(model, layer: int, vec: torch.Tensor, strength: float, device: str):
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


def generate_unforced(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p})

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)

    cont_ids = out_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(cont_ids, skip_special_tokens=True)


def generate_forced_yes(
    model,
    tokenizer,
    prompt: str,
    yes_token_id: int,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    forced = torch.tensor([[yes_token_id]], device=device, dtype=prompt_ids.dtype)
    forced_ids = torch.cat([prompt_ids, forced], dim=1)
    attn = torch.ones_like(forced_ids)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p})

    with torch.no_grad():
        out_ids = model.generate(forced_ids, attention_mask=attn, **gen_kwargs)

    cont_ids = out_ids[0, prompt_ids.shape[1] :]
    return tokenizer.decode(cont_ids, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--unforced_strength", type=float, default=1.0)
    ap.add_argument("--forced_yes_strength", type=float, default=5.0)

    ap.add_argument("--unforced_max_new_tokens", type=int, default=80)
    ap.add_argument("--forced_yes_max_new_tokens", type=int, default=120)

    # unforced: prefer greedy (less noise); forced-yes: prefer sampling (more like harvesting)
    ap.add_argument("--unforced_do_sample", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--forced_yes_do_sample", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--stimulus", default="data/multilingual_litmus.json")
    ap.add_argument("--out", default="multilingual_litmus_generations.json")
    args = ap.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")
    act_dir = os.path.join(project_root, "activations")
    out_dir = os.path.join(project_root, "outputs")
    eval_dir = os.path.join(out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    labels_path = os.path.join(data_dir, "sycophancy_labels.json")
    syco_acts_path = os.path.join(act_dir, "sycophantic_acts.pt")
    refusal_path = os.path.join(act_dir, "refusal_acts.pt")

    stimulus_path = args.stimulus if os.path.isabs(args.stimulus) else os.path.join(project_root, args.stimulus)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(eval_dir, args.out)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Device: {device}")
    print("Loading caches…")
    labels = load_json(labels_path)
    lied_indices = [i for i, y in enumerate(labels) if int(y) == 1]
    syco_acts = torch.load(syco_acts_path, map_location="cpu")
    refusal_acts, refusal_indices, refusal_meta = load_refusal_dict(refusal_path)
    if refusal_indices != lied_indices:
        raise RuntimeError("Alignment error: refusal_indices != lied_indices")

    vecB_cpu, raw_norm = compute_vecB_layer(
        layer=args.layer,
        refusal_acts=refusal_acts,
        syco_acts=syco_acts,
        lied_indices=lied_indices,
        normalize=True,
    )
    print(f"Vector B at layer {args.layer}: raw_norm={raw_norm:.4f} (unit-normalized)")

    stimulus = load_json(stimulus_path)

    print("Loading model/tokenizer…")
    model_path = resolve_local_snapshot_if_available(args.model_id)
    model_kwargs: Dict[str, Any] = {"device_map": device}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    yes_token_str = " Yes"
    yes_token_id = get_single_token_id(tokenizer, yes_token_str)
    print(f"Token check: encode({yes_token_str!r}) -> [{yes_token_id}]")

    results: Dict[str, Any] = {
        "config": {
            "model_id": args.model_id,
            "layer": args.layer,
            "vectorB_unit_normalized": True,
            "vectorB_raw_norm": raw_norm,
            "refusal_token_str": refusal_meta.get("refusal_token_str"),
            "refusal_token_id": refusal_meta.get("refusal_token_id"),
            "seed": args.seed,
            "unforced": {
                "strength": args.unforced_strength,
                "max_new_tokens": args.unforced_max_new_tokens,
                "do_sample": args.unforced_do_sample,
                "temperature": args.temperature if args.unforced_do_sample else None,
                "top_p": args.top_p if args.unforced_do_sample else None,
            },
            "forced_yes": {
                "strength": args.forced_yes_strength,
                "max_new_tokens": args.forced_yes_max_new_tokens,
                "do_sample": args.forced_yes_do_sample,
                "temperature": args.temperature if args.forced_yes_do_sample else None,
                "top_p": args.top_p if args.forced_yes_do_sample else None,
                "forced_token_str": yes_token_str,
                "forced_token_id": yes_token_id,
            },
            "stimulus_path": os.path.relpath(stimulus_path, project_root),
        },
        "generations": [],
    }

    # Stable ordering
    items = stimulus["items"]
    langs = ["fr", "zh"]
    prompt_kinds = ["honest_prompt", "sycophantic_prompt"]

    for lang in langs:
        for it in items:
            idx = int(it["index"])
            category = it["category"]
            for kind in prompt_kinds:
                prompt = it[lang][kind]

                # -----------------------------
                # Protocol A: unforced
                # -----------------------------
                per_seed = args.seed + (idx * 100) + (0 if (lang == "fr") else 1) + (0 if (kind == "honest_prompt") else 10)

                set_seed(per_seed)
                base_unforced = generate_unforced(
                    model, tokenizer, prompt, device,
                    args.unforced_max_new_tokens, args.unforced_do_sample, args.temperature, args.top_p
                )

                set_seed(per_seed)
                hook = register_caa_hook(model, args.layer, vecB_cpu, args.unforced_strength, device)
                try:
                    steer_unforced = generate_unforced(
                        model, tokenizer, prompt, device,
                        args.unforced_max_new_tokens, args.unforced_do_sample, args.temperature, args.top_p
                    )
                finally:
                    hook.remove()

                results["generations"].append(
                    {
                        "protocol": "unforced",
                        "language": lang,
                        "index": idx,
                        "category": category,
                        "prompt_kind": kind,
                        "prompt": prompt,
                        "baseline": base_unforced,
                        "steered": steer_unforced,
                    }
                )

                print(f"[unforced] {lang} idx={idx:<2} {kind:<17} | base: {base_unforced[:80].replace(chr(10),' ')} | steer: {steer_unforced[:80].replace(chr(10),' ')}")

                # -----------------------------
                # Protocol B: forced-Yes
                # -----------------------------
                set_seed(per_seed)
                base_forced = generate_forced_yes(
                    model, tokenizer, prompt, yes_token_id, device,
                    args.forced_yes_max_new_tokens, args.forced_yes_do_sample, args.temperature, args.top_p
                )

                set_seed(per_seed)
                hook = register_caa_hook(model, args.layer, vecB_cpu, args.forced_yes_strength, device)
                try:
                    steer_forced = generate_forced_yes(
                        model, tokenizer, prompt, yes_token_id, device,
                        args.forced_yes_max_new_tokens, args.forced_yes_do_sample, args.temperature, args.top_p
                    )
                finally:
                    hook.remove()

                results["generations"].append(
                    {
                        "protocol": "forced_yes",
                        "language": lang,
                        "index": idx,
                        "category": category,
                        "prompt_kind": kind,
                        "prompt": prompt,
                        "forced_prefix": yes_token_str,
                        "baseline": base_forced,
                        "steered": steer_forced,
                    }
                )

                print(f"[forced_yes] {lang} idx={idx:<2} {kind:<17} | base: {base_forced[:80].replace(chr(10),' ')} | steer: {steer_forced[:80].replace(chr(10),' ')}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()


