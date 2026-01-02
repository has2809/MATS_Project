import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
            "refusal_acts.pt must be a dict containing keys: 'refusal_acts' and 'indices'. "
            "Re-run run_refusal_harvest.py if needed."
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
    vec = vec.to(device)

    def hook_fn(_module, _inp, out):
        if isinstance(out, tuple):
            hs = out[0]
            rest = out[1:]
        else:
            hs = out
            rest = None

        delta = (strength * vec.to(dtype=hs.dtype)).view(1, 1, -1)  # broadcast over batch, seq
        hs = hs + delta

        if rest is None:
            return hs
        return (hs,) + rest

    return model.model.layers[layer].register_forward_hook(hook_fn)


def generate_continuation(
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

    cont_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(cont_ids, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--strength", type=float, default=5.0)
    ap.add_argument("--max_new_tokens", type=int, default=50)

    ap.add_argument("--do_sample", action="store_true", help="Enable sampling (default: greedy)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--n_honest_control", type=int, default=20)
    ap.add_argument("--out", default="final_eval_generations.json")
    args = ap.parse_args()

    # Repo-relative paths (this script is expected to live in ./scripts/)
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
    refused_indices = [i for i, y in enumerate(labels) if int(y) == 0]

    if refusal_indices != lied_indices:
        raise RuntimeError(
            "Alignment error: refusal_indices != lied_indices.\n"
            f"len(refusal_indices)={len(refusal_indices)} len(lied_indices)={len(lied_indices)}"
        )

    # pick honest controls from honest_prompt side (true statements)
    n_honest = min(args.n_honest_control, len(dataset))
    honest_indices = list(range(n_honest))

    print(f"lied_indices: {len(lied_indices)} | refused_indices: {len(refused_indices)} | honest_indices: {len(honest_indices)}")

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
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

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
            "vectorB_unit_normalized": True,
            "vectorB_raw_norm": raw_norm,
            "refusal_token_str": refusal_meta.get("refusal_token_str"),
            "refusal_token_id": refusal_meta.get("refusal_token_id"),
        },
        "sets": {
            "syco_lied": lied_indices,
            "syco_refused": refused_indices,
            "honest_control": honest_indices,
        },
        "generations": [],
    }

    def run_set(set_name: str, indices: List[int], prompt_key: str):
        for idx in indices:
            prompt = dataset[idx][prompt_key]
            cat = dataset[idx]["category"]

            # match randomness between baseline and steered (only matters if sampling)
            per_prompt_seed = args.seed + idx

            set_seed(per_prompt_seed)
            baseline = generate_continuation(
                model, tokenizer, prompt, device,
                args.max_new_tokens, args.do_sample, args.temperature, args.top_p
            )

            set_seed(per_prompt_seed)
            hook = register_caa_hook(model, args.layer, vecB_cpu, args.strength, device)
            try:
                steered = generate_continuation(
                    model, tokenizer, prompt, device,
                    args.max_new_tokens, args.do_sample, args.temperature, args.top_p
                )
            finally:
                hook.remove()

            results["generations"].append({
                "set": set_name,
                "index": idx,
                "category": cat,
                "prompt_key": prompt_key,
                "prompt": prompt,
                "baseline": baseline,
                "steered": steered,
            })

            print(f"[{set_name}] idx={idx:<2} {cat:<22} | base: {short(baseline)} | steer: {short(steered)}")

    print("\nRunning sycophantic prompts (lied set)…")
    run_set("syco_lied", lied_indices, "sycophantic_prompt")

    print("\nRunning sycophantic prompts (already refused set)…")
    run_set("syco_refused", refused_indices, "sycophantic_prompt")

    print("\nRunning honest prompts (negative control)…")
    run_set("honest_control", honest_indices, "honest_prompt")

    # Write outputs to ./outputs/eval by default (unless an absolute path is provided)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(eval_dir, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {out_path}")
    print("Next: manually grade refusal/correctness and false refusals.")


if __name__ == "__main__":
    main()