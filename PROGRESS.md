# Rock Token Replication Progress

## Current Phase
Running Rock Token identification pipeline. Need to execute code and generate results.

## Implementation Plan

### Key Methods to Implement:
1. [x] Read and understand paper thoroughly
2. [x] Write Rock Token identification pipeline (rock_token_identification.py)
3. [ ] Run Rock Token identification pipeline
4. [ ] Implement Pillar Token knockout analysis  
5. [ ] Implement Gradient geometry analysis
6. [ ] Implement Selective Distillation simulation
7. [ ] Generate all visualizations and results
8. [ ] Create reproduce.sh and final report

### Paper Setup:
- Teacher: Qwen3-30B-A3B-Instruct (MoE, ~3B active, 30.5B total params)
- Student: Qwen3-4B-Instruct
- Both with thinking mode disabled (enable_thinking=False)
- Rock Score: R(v) = mean_KL(v) * Freq(v)
- Context-consistent filtering: persistent high-loss + context window similarity
- Cutoff K=100 tokens
- Full pipeline needs pre/post OPD checkpoints (we don't have these)

### Key Results to Reproduce:
1. Rock Token identification: ~K=100 tokens, ~18% of output tokens, ~60% of KL
2. Rock Token categories: LaTeX delimiters, markdown structure, discourse markers, digits
3. Pillar census: 3.5% on MATH-500, 1.5% on IFEval (7/200 and 3/200 Strong Pillars)
4. Selective distillation: Rock-Freeze matches baseline, 1.4x speedup
5. Gradient analysis: Rock tokens have low grad magnitude but high alignment

### Practical Constraints:
- Single H100 80GB GPU
- Can't train OPD from scratch (too expensive)
- Will use pre-trained models as-is for analysis
- 30B model needs 4-bit quantization to fit alongside other operations
- Sequential model loading to manage memory

### Simplifications from Full Paper:
- No pre/post OPD persistent filtering (no training checkpoints)
- Use basic Rock Score R(v) without CCR context filtering
- Smaller subset of MATH-500 (100 problems instead of 500)
- Shorter generation length (1024 tokens)
- Selective distillation: simulate rather than full training

## Key Decisions
- Use Qwen3-4B and Qwen3-30B-A3B (exact models from paper)
- Teacher loaded in 4-bit quantization for memory
- Sequential loading: student first, then teacher
- Focus on MATH-500 for all analyses
- K=100 cutoff as in paper

## Completed Work
- rock_token_identification.py: Full pipeline with 5 steps + visualization
  - Step 1: Generate student rollouts
  - Step 2: Compute student log-probs (top-2000 per position)
  - Step 3: Load teacher, compute KL divergence
  - Step 4: Compute Rock Scores
  - Step 5: Categorize and analyze density

## Failed Approaches
(none yet)

## Evaluation Coverage
### Will Address:
- Rock Token identification via Rock Score
- Rock Token categorization into 4 clusters
- Density statistics (~18% of tokens)
- KL coverage (~60% at K=100)
- Pillar Token knockout analysis
- Gradient geometry (magnitude + alignment)

### Cannot Address (resource constraints):
- Full OPD training pipeline
- Pre/post OPD persistent filtering
- Context-consistent filtering with embeddings
- Full selective distillation training curves
- Multiple evaluation benchmarks (AIME24, AIME25, HMMT25)
