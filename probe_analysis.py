"""
Sycophancy Probe Analysis
=========================
Trains a probe to find the "Truth Direction" using:
1. Honest Activations (Label = 0)
2. Sycophantic Activations (Label = 1, FILTERED by actual judgments)
"""

import torch
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("Loading data...")
honest_acts = torch.load('honest_acts.pt').float().numpy()      # Shape: [N, Layers, Dim]
syco_acts = torch.load('sycophantic_acts.pt').float().numpy()   # Shape: [N, Layers, Dim]

# Load Ground Truth Judgments (Did the model actually lie?)
with open('sycophancy_labels.json', 'r') as f:
    syco_labels = np.array(json.load(f))

# Filter: Only use examples where the model *actually* was sycophantic
# If the model refused (label 0), it's effectively an "honest" response, 
# so we shouldn't use it as a positive example of sycophancy.
valid_indices = np.where(syco_labels == 1)[0]
n_honest = len(honest_acts)
n_syco = len(valid_indices)

print(f"Total Examples: {len(syco_labels)}")
print(f"Valid Sycophantic Examples (Model Lied): {n_syco}")
print(f"Refusals (Model was Honest): {len(syco_labels) - n_syco}")

if n_syco < 5:
    print("WARNING: Not enough sycophantic examples to train a reliable probe!")

# Prepare Training Data
# Class 0: Honest Prompts (All assumed honest)
# Class 1: Sycophantic Prompts (Only those that were actually sycophantic)

# We will iterate layer by layer
n_layers = honest_acts.shape[1]
accuracies = []

print(f"\nTraining Probes across {n_layers} layers...")

# ==============================================================================
# 2. TRAIN PROBES (Layer-wise)
# ==============================================================================
for layer in range(n_layers):
    # Extract activations for this layer
    X_honest = honest_acts[:, layer, :]
    X_syco = syco_acts[valid_indices, layer, :] # Only valid sycophantic responses
    
    # Combine
    X = np.concatenate([X_honest, X_syco])
    y = np.concatenate([np.zeros(n_honest), np.ones(n_syco)])
    
    # Train/Test with Cross Validation
    # Use StratifiedKFold to ensure balance in splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    layer_scores = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = LogisticRegression(max_iter=1000, solver='liblinear') # liblinear good for small datasets
        clf.fit(X_train, y_train)
        layer_scores.append(clf.score(X_test, y_test))
    
    avg_acc = np.mean(layer_scores)
    accuracies.append(avg_acc)
    # Optional: Print progress every 5 layers
    if layer % 5 == 0:
        print(f"Layer {layer:02d}: {avg_acc:.2%} accuracy")

# ==============================================================================
# 3. VISUALIZATION
# ==============================================================================
best_layer = np.argmax(accuracies)
print(f"\nBest Layer: {best_layer} (Accuracy: {accuracies[best_layer]:.2%})")

# Plot 1: Accuracy Curve
plt.figure(figsize=(10, 6))
plt.plot(range(n_layers), accuracies, marker='o', linestyle='-', color='purple')
plt.title('Sycophancy Probe Accuracy by Layer')
plt.xlabel('Layer Index')
plt.ylabel('Classification Accuracy')
plt.grid(True, alpha=0.3)
plt.axvline(best_layer, color='r', linestyle='--', label=f'Peak (L{best_layer})')
plt.legend()
plt.savefig('layer_accuracy.png')
print("Saved layer_accuracy.png")

# Plot 2: PCA of Best Layer
# We use the same X/y constructed above for the best layer
X_honest_best = honest_acts[:, best_layer, :]
X_syco_best = syco_acts[valid_indices, best_layer, :]
X_combined = np.concatenate([X_honest_best, X_syco_best])
y_combined = np.concatenate([np.zeros(n_honest), np.ones(n_syco)])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_combined)

plt.figure(figsize=(10, 8))
# Plot Honest
plt.scatter(X_pca[y_combined==0, 0], X_pca[y_combined==0, 1], 
            c='blue', alpha=0.6, label='Honest', s=100)
# Plot Sycophantic
plt.scatter(X_pca[y_combined==1, 0], X_pca[y_combined==1, 1], 
            c='red', alpha=0.6, label='Sycophantic', s=100)

plt.title(f'PCA of Activations at Layer {best_layer}')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.savefig('pca_separation.png')
print("Saved pca_separation.png")