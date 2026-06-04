# Rock Token Replication Progress

## Current Phase
Reading paper and creating implementation plan.

## Implementation Plan

### Key Methods to Implement:
1. [x] Read and understand paper
2. [ ] Set up environment (install dependencies)
3. [ ] Implement Rock Token identification pipeline
   - Rock Score computation: R(v) = mean_KL(v) * Freq(v)
   - Context-consistent filtering with context windows
   - Cutoff selection at K=100
4. [ ] Implement Pillar Token knockout analysis  
   - Token logit masking during inference
   - Accuracy delta computation
   - Bootstrap significance testing
5. [ ] Implement Selective Distillation (gradient freeze)
   - Window-aware token reweighting
   - λ=0 for Rock Tokens (Rock-Freeze)
   - Frequency-matched random freeze baseline
6. [ ] Run experiments and generate results
7. [ ] Create reproduce.sh and final report

### Paper Setup:
- Teacher: Qwen3-30B-A3B-Instruct-2507 (MoE, ~3B active)
- Student: Qwen3-4B-Instruct-2507
- Both with thinking mode disabled
- Stage 1 (Off-Policy): 20k teacher solutions, forward KL, kd_ratio=0.5
- Stage 2 (On-Policy): 10k prompts, 4 rollouts each, reverse KL, kd_ratio=1.0
- Evaluation: AIME24, AIME25, HMMT25, MATH-500, IFEval

### Key Results to Reproduce:
1. Rock Token identification: ~K=100 tokens, ~18% of output tokens, ~60% of KL
2. Rock Token categories: LaTeX delimiters, markdown structure, discourse markers, digits
3. Pillar census: 3.5% on MATH-500, 1.5% on IFEval
4. Selective distillation: Rock-Freeze matches baseline, 1.4x speedup
5. Gradient analysis: Rock tokens have low grad magnitude but high alignment

### Hardware: Single H100 80GB
- Can fit Qwen3-4B (~8GB in bf16) easily
- Qwen3-30B-A3B might fit in bf16 (~60GB) since MoE with 3B active
- Both together might be tight, can use quantization or sequential loading

## Key Decisions
- Will use Qwen3-4B-Instruct and Qwen3-30B-A3B-Instruct
- Focus on MATH-500 for token-level analysis (500 problems, manageable)
- Will implement streamlined versions of expensive experiments

## Completed Work
- Paper reading complete

## Failed Approaches
(none yet)

## Evaluation Coverage
- Rock Token identification methodology
- Rock Token categorization  
- Pillar Token knockout analysis
- Selective distillation experiment
- Gradient geometry analysis
