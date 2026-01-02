"""
Activation Steering Experiment
==============================
Goal: Extract a "truthfulness" steering vector from Layer 18 and test its effect
      on reducing sycophantic behavior in DeepSeek-R1-Distill.

Method:
  1. Train LogisticRegression on Layer 18 activations (honest vs sycophantic)
  2. Extract and normalize the coefficient vector
  3. Apply steering by subtracting this vector from Layer 18's residual stream
  4. Test with different steering strengths
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
import os

# Optional HuggingFace login (DO NOT hardcode tokens in code)
from huggingface_hub import login  # noqa: E402

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("HF_TOKEN not set; assuming you are already logged in or the model is cached.")

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TARGET_LAYER = 18  # Layer identified from probe analysis
device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# PATHS (repo-relative; this script is expected to live in ./scripts/)
# =============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ACT_DIR = os.path.join(PROJECT_ROOT, "activations")

HONEST_ACTS_PATH = os.path.join(ACT_DIR, "honest_acts.pt")
SYCO_ACTS_PATH = os.path.join(ACT_DIR, "sycophantic_acts.pt")
LABELS_PATH = os.path.join(DATA_DIR, "sycophancy_labels.json")
DATASET_PATH = os.path.join(DATA_DIR, "sycophancy_dataset.json")

print("="*80)
print("ACTIVATION STEERING EXPERIMENT")
print("="*80)
print(f"Device: {device}")
print(f"Target Layer: {TARGET_LAYER}")
print()

# =============================================================================
# LOAD MODEL
# =============================================================================
print("Loading model...")
model_kwargs = {"device_map": device}
if device == "cuda":
    model_kwargs["torch_dtype"] = torch.float16

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print(f"✓ Model loaded: {MODEL_ID}\n")

# =============================================================================
# TOKEN ALIGNMENT (CRITICAL)
# =============================================================================
YES_TOKEN_STR = " Yes"  # leading space is important for Llama/SentencePiece
YES_TOKEN_IDS = tokenizer.encode(YES_TOKEN_STR, add_special_tokens=False)
if len(YES_TOKEN_IDS) != 1:
    raise RuntimeError(f"Expected '{YES_TOKEN_STR}' to tokenize to 1 token, got {YES_TOKEN_IDS}")
YES_TOKEN_ID = YES_TOKEN_IDS[0]
print(f"Token check: encode('{YES_TOKEN_STR}') -> {YES_TOKEN_IDS} (id={YES_TOKEN_ID})\n")

# =============================================================================
# LOAD ACTIVATIONS AND LABELS
# =============================================================================
print("Loading activations and labels...")
honest_acts = torch.load(HONEST_ACTS_PATH)
syco_acts = torch.load(SYCO_ACTS_PATH)

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

with open(DATASET_PATH, "r") as f:
    dataset = json.load(f)

print(f"✓ Loaded activations: {honest_acts.shape}")
print(f"✓ Loaded labels: {len(labels)} examples")
print()

# =============================================================================
# EXTRACT STEERING VECTOR FROM LAYER 18
# =============================================================================
print(f"Extracting steering vector from Layer {TARGET_LAYER}...")

# Get Layer 18 activations for all honest examples
X_honest = honest_acts[:, TARGET_LAYER, :].float().numpy()  # [N, D_model]

# Filter sycophantic examples: only use cases where label=1 (model agreed)
syco_indices = [i for i, label in enumerate(labels) if label == 1]
X_syco = syco_acts[syco_indices, TARGET_LAYER, :].float().numpy()  # [N_valid, D_model]

print(f"Honest examples: {X_honest.shape[0]}")
print(f"Valid sycophantic examples (label=1): {X_syco.shape[0]}")

# Combine into training data
X_train = np.vstack([X_honest, X_syco])
y_train = np.array([0] * len(X_honest) + [1] * len(X_syco))

# Train logistic regression classifier
print("Training LogisticRegression classifier...")
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

train_acc = clf.score(X_train, y_train)
print(f"✓ Training accuracy: {train_acc:.3f}")

# Extract and normalize steering vector
steering_vec = clf.coef_[0]  # Shape: [D_model]
steering_vec = steering_vec / np.linalg.norm(steering_vec)  # Normalize

# Convert to PyTorch tensor
steering_vec = torch.tensor(steering_vec, dtype=torch.float16 if device == "cuda" else torch.float32, device=device)
print(f"✓ Steering vector shape: {steering_vec.shape}")
print(f"✓ Steering vector norm: {torch.norm(steering_vec).item():.3f}")
print()

# =============================================================================
# DEFINE STEERING HOOK
# =============================================================================
def create_steering_hook(coefficient):
    """
    Creates a hook that subtracts (coefficient * steering_vec) from Layer 18.
    
    Args:
        coefficient: Steering strength (0 = no steering, higher = stronger)
    
    Returns:
        Hook function
    """
    def hook_fn(module, input, output):
        # output is typically a tuple (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # hidden_states shape: [batch, seq_len, hidden_size]
        # IMPORTANT: steer ONLY the last token position to match harvesting (activation at " Yes")
        if coefficient != 0:
            hidden_states = hidden_states.clone()
            hidden_states[:, -1, :] = hidden_states[:, -1, :] - (coefficient * steering_vec)
        
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states
    
    return hook_fn


def generate_with_steering(prompt, coefficient, max_new_tokens=200):
    """
    Generate text with activation steering applied to Layer 18.
    
    Args:
        prompt: Input text
        coefficient: Steering strength
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generated text
    """
    # Register hook on target layer
    hook = model.model.layers[TARGET_LAYER].register_forward_hook(
        create_steering_hook(coefficient)
    )
    
    try:
        # ---------------------------------------------------------------------
        # CRITICAL: Match harvesting context by forcing the next token to be " Yes"
        # We do this by appending the YES token ID to the prompt token IDs.
        # ---------------------------------------------------------------------
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        forced_ids = torch.cat(
            [prompt_ids, torch.tensor([[YES_TOKEN_ID]], device=device, dtype=prompt_ids.dtype)],
            dim=1,
        )
        attention_mask = torch.ones_like(forced_ids)

        with torch.no_grad():
            output_ids = model.generate(
                forced_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Return only the assistant continuation AFTER the prompt (includes the forced " Yes")
        # Token-safe slicing (avoid brittle string slicing).
        continuation_ids = output_ids[0, prompt_ids.shape[1]:]
        continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
        return continuation_text
    
    finally:
        # Always remove hook
        hook.remove()


# =============================================================================
# SELECT TEST PROMPTS
# =============================================================================
print("="*80)
print("SELECTING TEST PROMPTS")
print("="*80)

# Find examples where the model originally agreed with the misconception (label=1)
test_indices = []
test_categories = []

# Let's pick 5 diverse examples across different categories
for i, label in enumerate(labels):
    if label == 1:  # Model originally rationalized
        cat = dataset[i]["category"]
        # Get one from each category type
        if len(test_indices) < 5:
            if cat not in test_categories or len(test_indices) < 3:
                test_indices.append(i)
                test_categories.append(cat)

# If we need more variety, add specific interesting ones
priority_indices = [1, 5, 12, 30, 40]  # Base 7, Ramanujan sum, probability, quicksort, Great Wall
for idx in priority_indices:
    if idx < len(labels) and labels[idx] == 1 and idx not in test_indices:
        if len(test_indices) < 5:
            test_indices.append(idx)

test_indices = test_indices[:5]

print(f"Selected {len(test_indices)} test examples:")
for i, idx in enumerate(test_indices, 1):
    print(f"  {i}. [{dataset[idx]['category']}] Index {idx}")
print()

# =============================================================================
# RUN STEERING EXPERIMENT
# =============================================================================
print("="*80)
print("RUNNING STEERING EXPERIMENT")
print("="*80)
print()

steering_strengths = [0, 10, 20, 50]

for test_num, idx in enumerate(test_indices, 1):
    entry = dataset[idx]
    prompt = entry["sycophantic_prompt"]
    
    print("="*80)
    print(f"TEST {test_num}/5 - {entry['category']}")
    print("="*80)
    print(f"Prompt: {prompt}")
    print()
    
    for strength in steering_strengths:
        print(f"--- Steering Strength: {strength} ---")
        
        try:
            generated = generate_with_steering(prompt, strength, max_new_tokens=200)
            print(generated[:500])  # Show first 500 chars
            
            if len(generated) > 500:
                print("...")
            
        except Exception as e:
            print(f"ERROR: {e}")
        
        print()
    
    print("\n")

# =============================================================================
# SUMMARY
# =============================================================================
print("="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
print()
print("Key Observations to Look For:")
print("  - Baseline (0): After forced ' Yes', does the model rationalize the misconception?")
print("  - Steered (10/20/50): Does the model backtrack/correct vs double down?")
print("  - Failure mode: If outputs become repetitive/gibberish, the coefficient is too large for this intervention.")
print()
print("Note: This vector is trained on the *forced-' Yes' token state* at Layer 18,")
print("so the steering test must also force ' Yes' (as this script does).")

