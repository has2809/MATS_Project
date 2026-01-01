"""
Activation Harvesting for Sycophancy Detection
==============================================
Goal: Extract internal activations when model generates " Yes" for both:
  - Honest prompts (true statements)
  - Sycophantic prompts (false but plausible statements)

Method:
  1. Force model to generate " Yes" token
  2. Cache residual stream activations at that token position
  3. Save [N_examples, N_layers, D_model] tensors for PCA analysis
"""

from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import os

# Login
login(token="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")

# [OPTIMIZATION] Enable TF32 for speed on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =============================================================================
# MODEL LOADING
# =============================================================================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")
print(f"Loading model: {MODEL_ID}...")

# Load model and tokenizer
model_kwargs = {"device_map": device}
if device == "cuda":
    model_kwargs["torch_dtype"] = torch.float16

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"✓ Model loaded on {device}")
print(f"Model layers: {len(model.model.layers)}")
print(f"Hidden size: {model.config.hidden_size}\n")

# =============================================================================
# LOAD DATASET
# =============================================================================
print("Loading sycophancy dataset...")
with open('/workspace/MATS_Project/sycophancy_dataset.json', 'r') as f:
    dataset = json.load(f)

print(f"✓ Loaded {len(dataset)} examples\n")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_yes_token_id():
    """Find the token ID for ' Yes' (with leading space)"""
    token_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    print(f"' Yes' token ID: {token_id}\n")
    return token_id

YES_TOKEN_ID = get_yes_token_id()


def capture_activations_with_forced_token(prompt, forced_token_id, generate_continuation=True):
    """
    Generate text with a forced first token, capturing residual stream activations.
    
    Args:
        prompt: Input text
        forced_token_id: Token ID to force as first generated token (e.g., " Yes")
        generate_continuation: If True, generate more tokens after the forced one
        
    Returns:
        activations: List of tensors [hidden_size] for each layer at forced token position
        full_text: Complete generated text (for validation)
    """

    # 1. Prepare Inputs (Prompt + " Yes")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs["input_ids"]
    
    # Concatenate the " Yes" token manually
    forced_token_tensor = torch.tensor([[forced_token_id]], device=device)
    full_input_ids = torch.cat([prompt_ids, forced_token_tensor], dim=1)
    
    # Create attention mask)
    attention_mask = torch.ones_like(full_input_ids)
    
    # 2. CAPTURE ACTIVATIONS (Forward Pass)
    layer_activations = []
    
    def hook_fn(module, input, output):
        # output is usually a tuple (hidden_states, past_key_values)
        if isinstance(output, tuple):
            output = output[0]
        # output shape: [batch, seq_len, hidden_size]
        # We want the LAST token (the " Yes" token)
        activation = output[0, -1, :].detach().cpu().clone()
        layer_activations.append(activation)
    
    # Register hooks
    hooks = []
    for layer in model.model.layers:
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    # Run SINGLE forward pass
    with torch.no_grad():
        model(full_input_ids, attention_mask=attention_mask)
        
    # Remove hooks immediately so they don't capture generation tokens
    for hook in hooks:
        hook.remove()
        
    # 3. VALIDATION GENERATION (Separate Step)
    full_text = ""
    if generate_continuation:
        with torch.no_grad():
            # Pass the already-formed input (Prompt + Yes) to generate what comes next
            output_ids = model.generate(
                full_input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
    return layer_activations, full_text


# =============================================================================
# MAIN HARVESTING LOOP
# =============================================================================
print("="*70)
print("STARTING ACTIVATION HARVESTING")
print("="*70)
print(f"Processing {len(dataset)} examples...")
print(f"Capturing activations at ' Yes' token position")
print(f"Layers: {len(model.model.layers)} | Hidden size: {model.config.hidden_size}\n")

honest_acts = []
sycophantic_acts = []
metadata = []

for idx, entry in enumerate(tqdm(dataset, desc="Harvesting")):
    category = entry["category"]
    honest_prompt = entry["honest_prompt"]
    sycophantic_prompt = entry["sycophantic_prompt"]
    
    # Run A: Honest prompt
    # [OPTIMIZATION] Skip generation for honest prompt
    honest_activations, honest_text = capture_activations_with_forced_token(
        honest_prompt, 
        YES_TOKEN_ID,
        generate_continuation=False
    )
    
    # Run B: Sycophantic prompt
    syco_activations, syco_text = capture_activations_with_forced_token(
        sycophantic_prompt,
        YES_TOKEN_ID,
        generate_continuation=True
    )
    
    # Store activations (stack list into single tensor)
    honest_acts.append(torch.stack(honest_activations))  # [n_layers, hidden_size]
    sycophantic_acts.append(torch.stack(syco_activations))
    
    # Store metadata for validation
    metadata.append({
        "index": idx,
        "category": category,
        "honest_prompt": honest_prompt,
        "honest_continuation": honest_text[len(honest_prompt):],
        "sycophantic_prompt": sycophantic_prompt,
        "sycophantic_continuation": syco_text[len(sycophantic_prompt):]
    })

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Stack into final tensors [N_examples, N_layers, D_model]
honest_tensor = torch.stack(honest_acts)
syco_tensor = torch.stack(sycophantic_acts)

print(f"Honest activations shape: {honest_tensor.shape}")
print(f"Sycophantic activations shape: {syco_tensor.shape}")

# Save tensors
torch.save(honest_tensor, '/workspace/MATS_Project/honest_acts.pt')
torch.save(syco_tensor, '/workspace/MATS_Project/sycophantic_acts.pt')
print(f"✓ Saved activation tensors")

# Save metadata
with open('/workspace/MATS_Project/experiment_log.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Saved experiment log")

# =============================================================================
# VALIDATION: Show example continuations
# =============================================================================
print("\n" + "="*70)
print("VALIDATION: Sample Sycophantic Continuations")
print("="*70)
print("Checking if model rationalizes false statements...\n")

for i in [0, 1, 2]:  # Show first 3 examples
    meta = metadata[i]
    print(f"--- Example {i+1}: {meta['category']} ---")
    print(f"Sycophantic Prompt: {meta['sycophantic_prompt']}")
    print(f"Continuation: {meta['sycophantic_continuation'][:200]}...")
    print()

print("="*70)
print("HARVESTING COMPLETE")
print("="*70)
print(f"\nFiles created:")
print(f"  - honest_acts.pt ({honest_tensor.shape})")
print(f"  - sycophantic_acts.pt ({syco_tensor.shape})")
print(f"  - experiment_log.json ({len(metadata)} entries)")

