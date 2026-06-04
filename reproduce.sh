#!/bin/bash
# Rock Token Replication - Reproduce All Key Results
# Paper: "Rock Tokens: What Do Language Models Learn from Distillation?"
# 
# This script reproduces the key analyses from the paper using:
# - Teacher: Qwen3-30B-A3B-Instruct (4-bit quantized)
# - Student: Qwen3-4B-Instruct
# - Dataset: MATH-500 (30 problems, reduced from 500 for compute constraints)
#
# Estimated runtime: ~30 minutes on a single H100 GPU
# Results are saved to /workspace/results/

set -e

echo "================================================================="
echo "Rock Token Replication - Reproducing Key Results"
echo "================================================================="
echo ""

# Install dependencies if needed
pip install -q transformers accelerate bitsandbytes datasets matplotlib numpy tqdm 2>/dev/null

mkdir -p /workspace/results
mkdir -p /workspace/cache

# =====================================================
# Step 1: Rock Token Identification (Main result)
# Paper Section 3: Identifying rock tokens via Rock Score
# This is the most important analysis - identifies K=100 tokens
# that account for ~60% of total KL divergence
# =====================================================
echo ""
echo "Step 1: Rock Token Identification"
echo "  - Generates teacher/student rollouts on MATH-500"
echo "  - Computes per-token KL divergence"  
echo "  - Ranks tokens by Rock Score R(v) = mean_KL(v) × freq(v)"
echo "  - Expected: ~66% KL coverage at K=100"
echo ""

if [ -f /workspace/cache/kl_data.pkl ]; then
    echo "  [CACHED] KL data already computed, skipping rollout generation."
    echo "  (Delete /workspace/cache/ to recompute from scratch)"
else
    echo "  Running rock token identification pipeline..."
    timeout 1200 python /workspace/rock_token_identification.py 2>&1 | tail -20
fi

# Always regenerate analysis even if cached
python -c "
import pickle, json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

with open('/workspace/cache/kl_data.pkl', 'rb') as f:
    kl_data = pickle.load(f)

token_kl = kl_data['token_kl_data']
token_freq = kl_data['token_freq']

# Rock Score = mean_KL * freq
scores = {}
for tid, kls in token_kl.items():
    scores[tid] = np.mean(kls) * token_freq[tid]

ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
total_score = sum(scores.values())

# Coverage curve
coverage = {}
cumsum = 0
for i, (tid, score) in enumerate(ranked):
    cumsum += score
    if (i+1) in [10, 25, 50, 75, 100, 150, 200, 300, 500]:
        coverage[str(i+1)] = cumsum / total_score

print(f'KL Coverage at K=100: {coverage[\"100\"]:.1%}')
print(f'Total unique tokens with KL > 0: {len(ranked)}')
print(f'Top 10 tokens cover: {coverage[\"10\"]:.1%}')
"

echo "  ✓ Rock token identification complete"
echo "  Results: /workspace/results/rock_token_results.json"
echo "  Plots: rock_token_analysis.png, kl_vs_freq_scatter.png"

# =====================================================
# Step 2: Gradient Geometry Analysis
# Paper Section 5.2: Gradient analysis of rock tokens
# Shows rock tokens have lower gradient magnitude but
# comparable/higher alignment vs high-KL non-rock tokens
# =====================================================
echo ""
echo "Step 2: Gradient Geometry Analysis"
echo "  - Computes gradient norms and cosine alignment"
echo "  - Compares rock tokens vs high-KL non-rock tokens"
echo ""

if [ -f /workspace/results/gradient_geometry_results.json ]; then
    echo "  [CACHED] Gradient geometry results exist."
else
    echo "  Running gradient geometry analysis..."
    timeout 600 python /workspace/gradient_geometry.py 2>&1 | tail -10
fi

python -c "
import json
d = json.load(open('/workspace/results/gradient_geometry_results.json'))
print(f'Rock grad magnitude: {d[\"rock\"][\"grad_magnitude\"][\"mean\"]:.4f}')
print(f'High-KL grad magnitude: {d[\"high_kl\"][\"grad_magnitude\"][\"mean\"]:.4f}')
print(f'Rock alignment: {d[\"rock\"][\"cosine_alignment\"][\"mean\"]:.4f}')
print(f'High-KL alignment: {d[\"high_kl\"][\"cosine_alignment\"][\"mean\"]:.4f}')
rock_mag = d['rock']['grad_magnitude']['mean']
hkl_mag = d['high_kl']['grad_magnitude']['mean']
if rock_mag < hkl_mag:
    print('✓ Confirmed: Rock tokens have LOWER gradient magnitude than high-KL non-rock')
"

echo "  ✓ Gradient geometry analysis complete"
echo "  Results: /workspace/results/gradient_geometry_results.json"
echo "  Plots: gradient_geometry.png, gradient_contribution.png"

# =====================================================
# Step 3: Selective Distillation Analysis
# Paper Section 4: Rock-Freeze strategy
# Analyzes KL contribution breakdown and estimates speedup
# =====================================================
echo ""
echo "Step 3: Selective Distillation Analysis"

if [ -f /workspace/results/selective_distillation_results.json ]; then
    echo "  [CACHED] Selective distillation results exist."
else
    echo "  Running selective distillation analysis..."
    timeout 600 python /workspace/selective_distillation.py 2>&1 | tail -10
fi

python -c "
import json
d = json.load(open('/workspace/results/selective_distillation_results.json'))
print(f'Rock token density: {d[\"position_stats\"][\"rock_fraction\"]:.1%}')
print(f'KL fraction from rocks: {d[\"kl_stats\"][\"rock_kl_fraction\"]:.1%}')
print(f'Estimated speedup: {d[\"speedup_estimation\"][\"estimated_speedup\"]:.1f}x')
"

echo "  ✓ Selective distillation analysis complete"
echo "  Results: /workspace/results/selective_distillation_results.json"
echo "  Plots: selective_distillation.png, kl_contribution_pies.png"

# =====================================================
# Step 4: Summary
# =====================================================
echo ""
echo "Step 4: Generating summary"
python /workspace/create_summary.py

echo ""
echo "================================================================="
echo "All results saved to /workspace/results/"
echo "================================================================="
echo ""
echo "Key output files:"
echo "  rock_token_results.json       - Rock token identification (main result)" 
echo "  gradient_geometry_results.json - Gradient analysis"
echo "  selective_distillation_results.json - Distillation analysis"
echo "  pillar_knockout_results.json   - Pillar census (partial)"
echo "  summary_table.png              - Summary comparison table"
echo ""
echo "Key plots:"
echo "  rock_token_analysis.png   - Rock token categories and coverage"
echo "  kl_vs_freq_scatter.png    - KL vs frequency scatter"
echo "  gradient_geometry.png     - Gradient magnitude/alignment"
echo "  selective_distillation.png - KL contribution breakdown"
echo "  pillar_knockout.png       - Pillar knockout results"
echo ""
ls -la /workspace/results/
echo ""
echo "Done!"
