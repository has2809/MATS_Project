### MATS_Project (Sycophancy Steering / CAA)

This repo contains the dataset, activation caches, scripts, and outputs for the DeepSeek-R1-Distill sycophancy/rationalization experiments and the Contrastive Activation Addition (CAA) “refusal vector” pivot.

### Quick start (paths)
- **Scripts** live in `scripts/`
- **Data** lives in `data/`
- **Activations** live in `activations/`
- **Outputs/logs** live in `outputs/`
- **Plots** live in `plots/`

> Note: Scripts use repo-relative paths assuming they live in `./scripts/` (they compute the project root as `..`).

### Running (examples)

```bash
# (Optional) login token for HF gated models
export HF_TOKEN="..."

# Harvest forced-" Yes" activations + experiment log
python3 scripts/run_harvesting.py

# Harvest forced-" Actually" activations (CAA refusal)
python3 scripts/run_refusal_harvest.py

# Compare subtraction (Vector A) vs CAA (Vector B) on a small test set
python3 scripts/run_comparison_experiment.py

# Final evaluation (writes into outputs/eval/ by default)
python3 scripts/run_final_eval.py --layer 15 --strength 1 --out final_eval_L15_s1.json
```


