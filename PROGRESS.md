# Rock Token Replication Progress

## Current Phase
Gradient geometry analysis (fixing bug). Core rock token identification complete and cached.

## Implementation Plan

### Key Methods to Implement:
1. [x] Read and understand paper thoroughly
2. [x] Write Rock Token identification pipeline (rock_token_identification.py)
3. [x] Run Rock Token identification pipeline (30 MATH-500 problems, results cached)
4. [ ] Fix and run Gradient geometry analysis (gradient_geometry.py - kl_data structure bug)
5. [ ] Implement Pillar Token knockout analysis (simplified)
6. [ ] Implement Selective Distillation simulation
7. [ ] Generate all visualizations and results
8. [ ] Create reproduce.sh and final report

### Completed Results:
- **Rock Token identification**: K=100 tokens identified
  - KL coverage at K=100: **66.5%** (paper: ~60%)
  - Mean density: **66.7%** (paper: ~18% — but this is fraction of positions containing rock tokens, not frequency-weighted)
  - Categories found: 12 LaTeX, 12 markdown, 21 discourse, 14 digits, 41 other
- **Cached data**: rollouts_and_logprobs.pkl (122MB), kl_data.pkl
- **Visualizations**: rock_token_analysis.png, rock_token_density.png, kl_vs_freq_scatter.png, jaccard_stability.png

### KL Data Structure (important for downstream code):
- `kl_data["token_kl_data"]` is dict: {int_token_id: list_of_kl_values}
- `kl_data["token_freq"]` is dict: {int_token_id: frequency_count}
- NOT nested dicts with "kl_values" key — it's directly a list

### Paper Setup:
- Teacher: Qwen3-30B-A3B-Instruct (MoE, ~3B active, 30.5B total)
- Student: Qwen3-4B-Instruct
- Both with thinking mode disabled (enable_thinking=False)
- Rock Score: R(v) = mean_KL(v) * Freq(v)
- Cutoff K=100 tokens
- Full pipeline needs pre/post OPD checkpoints (we don't have these)

### Key Results to Reproduce:
1. ✅ Rock Token identification: K=100 tokens, ~66% of KL (paper ~60%)
2. ✅ Rock Token categories: LaTeX delimiters, markdown structure, discourse markers, digits
3. ⬜ Pillar census: 3.5% on MATH-500, 1.5% on IFEval (7/200 and 3/200 Strong Pillars)
4. ⬜ Selective distillation: Rock-Freeze matches baseline, 1.4x speedup
5. ⬜ Gradient analysis: Rock tokens have low grad magnitude but high alignment

### Practical Constraints:
- Single H100 80GB GPU
- Can't train OPD from scratch (too expensive)
- Will use pre-trained models as-is for analysis
- 30B model needs 4-bit quantization
- Sequential model loading to manage memory

### Simplifications from Full Paper:
- No pre/post OPD persistent filtering (no training checkpoints)
- Use basic Rock Score R(v) without CCR context filtering
- 30 MATH-500 problems (paper uses 500)
- 512 max new tokens (paper uses 8000)
- Selective distillation: simulate rather than full training

## Key Decisions
- Use Qwen3-4B and Qwen3-30B-A3B (exact models from paper)
- Teacher loaded in 4-bit quantization for memory
- Sequential loading: student first, then teacher
- Focus on MATH-500 for all analyses
- K=100 cutoff as in paper

## Completed Work
- **rock_token_identification.py**: Full pipeline, tested and working
  - Generated 30 rollouts, computed KL divergences, identified top-100 rock tokens
  - Results in results/rock_token_results.json and 4 PNG visualizations
  - Cached data in cache/ directory
- **gradient_geometry.py**: Written but has bug (kl_data structure mismatch)
  - Bug: treats kl_data["token_kl_data"][tid] as dict with "kl_values" key, but it's just a list

## Failed Approaches
- Git push failed with 122MB file - fixed by git filter-branch to remove cache files from history
- Added cache/ to .gitignore

## Evaluation Coverage
### Addressed:
- ✅ Rock Token identification via Rock Score
- ✅ Rock Token categorization into 4 clusters
- ✅ Density statistics
- ✅ KL coverage (~66% at K=100)
- ✅ Jaccard stability analysis

### In Progress:
- ⬜ Gradient geometry (magnitude + alignment) — code written, needs bug fix
- ⬜ Pillar Token knockout — needs implementation
- ⬜ Selective distillation simulation — needs implementation

### Cannot Address (resource constraints):
- Full OPD training pipeline
- Pre/post OPD persistent filtering
- Context-consistent filtering with embeddings
- Full selective distillation training curves (AIME24, AIME25, HMMT25)
- IFEval benchmark evaluation

## Remaining Plan
1. Fix gradient_geometry.py kl_data bug → run → commit
2. Write pillar_knockout.py (simplified: measure accuracy change when masking rock tokens during generation)
3. Write selective_distillation.py (simulation showing speedup potential)
4. Create comprehensive final visualizations
5. Write reproduce.sh
6. Write REPORT.md
7. Final commit and push
