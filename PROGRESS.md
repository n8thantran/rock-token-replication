# HGODE Implementation Progress

## Current Phase: PIVOTING - Rewriting model for better generalization

## Key Problem Identified
The current HGODE implementation overfits badly on Cora (100% train, ~65-78% test vs 86% target).
Root cause: the graph neural operator G_φ and overall architecture need better design.
A simple GCN baseline gets 81% on Cora with the same data.

## New Strategy (Turn 100+)
1. Rewrite model.py with cleaner architecture closer to paper
2. Use simpler diffusion: G_φ(H,A) = PH - H with P=D^{-1}A (row-normalized)
3. Add dropout in ODE function for regularization
4. Use adjoint method or careful step selection
5. Focus on getting all 6 datasets running with reasonable results
6. Run ablations for Table 1

## Implementation Plan
- [x] Read paper thoroughly
- [x] Implement candidate pool construction
- [x] Implement force field MLP (Eq. 9)
- [x] Implement coupled ODE dynamics (Eq. 5)
- [x] Implement effective adjacency (Eq. 6)
- [x] Implement margin loss (Eq. 10-11)
- [ ] **FIX: Get Cora working with decent accuracy (>80%)**
- [ ] Train on Chameleon
- [ ] Train on ogbn-proteins (need OGB)
- [ ] Train on ZINC (need PyG ZINC)
- [ ] Train on Peptides-func (need LRGB)
- [ ] Train on ogbg-molpcba (need OGB)
- [ ] Run ablations (w/o hysteresis, w/o topo search, w/o force margin)
- [ ] Package: reproduce.sh, results/, REPORT.md

## Paper Results to Reproduce (Table 1)
| Dataset | Metric | HGODE Target |
|---------|--------|------|
| Cora | Acc↑ | 86.26±0.78 |
| Chameleon | Acc↑ | 72.56±1.24 |
| ogbn-proteins | ROC-AUC↑ | 81.24±0.63 |
| ZINC | MAE↓ | 0.078±0.025 |
| Peptides-func | AP↑ | 0.714±0.022 |
| ogbg-molpcba | AP↑ | 0.278±0.003 |

## Key Hyperparameters (Table 6)
- Cora: λ=0.1-0.3, τ=0.2-0.3, s=1.0, δ=0.1, β=0-0.1, hidden=128
- Chameleon: λ=0.4-0.6, τ=0.05-0.1, s=1.0-1.5, δ=0.2-0.3, β=0.3-0.5
- ZINC: λ=0.1-0.3, τ=0.2-0.3, s=1.0, δ=0.1, β=0-0.1
- Peptides-func: λ=0.5-0.8, τ=0.05-0.1, s=1.0-1.5, δ=0.2-0.3, β=0.1-0.3
- ogbn-proteins: λ=0.5-0.8, τ=0.05-0.1, s=1.0-1.5, δ=0.2-0.3, β=0.1-0.3
- ogbg-molpcba: λ=0.3-0.5, τ=0.1-0.2, s=1.0-1.5, δ=0.1-0.2, β=0.1-0.3

## Failed Approaches
1. **dopri5 solver**: Very slow convergence, model stays at random accuracy for many epochs
2. **rk4 with T=0.3, 10 steps**: 63-65% test accuracy - severe overfitting
3. **rk4 with original edges only, identity-init W**: 78% - closer but still overfitting
4. **Various lr/wd/dropout combos**: None got above ~78%
5. **Large candidate pool (96k 2-hop edges)**: Makes overfitting worse

## Key Design Decisions from Paper
- Solver: dopri5 recommended, rtol=atol=1e-5 (but paper also mentions RK4/Euler)
- G_φ(H,A) = PH - H with P=D^{-1}A (simplest form)
- Force: F_ij = s * tanh(MLP([h_i || h_j]))
- U dynamics: dU/dt = (1-λ)U - U³ + F
- H dynamics: τ_feat * dH/dt = G_φ(H,A) - γH
- A_ij = σ(U_ij/τ) (effective adjacency)
- U_init: +u_stable for observed edges, -u_stable for candidate-only edges

## Completed Files
- hgode/__init__.py - package init
- hgode/candidate_pool.py - candidate edge pool construction (tested ✓)
- hgode/model.py - needs rewrite to fix overfitting
- hgode/train_node.py - training script for node classification
