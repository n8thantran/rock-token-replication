# HGODE Implementation Progress

## Current Phase: Running remaining datasets (Chameleon, ZINC, Peptides, ogbn-proteins, ogbg-molpcba)

## Implementation Plan
- [x] Read paper thoroughly
- [x] Implement candidate pool construction
- [x] Implement force field MLP (Eq. 9)
- [x] Implement coupled ODE dynamics (Eq. 5)
- [x] Implement effective adjacency (Eq. 6)
- [x] Implement margin loss (Eq. 10-11)
- [x] **Cora: 90.03% ± 0.94% (paper: 86.26% ± 0.78%)** ✓
- [x] Cora ablation: no_hysteresis 89.73% (paper: 83.24%)
- [x] Cora ablation: no_topo_search 74.67% (paper: 84.14%)
- [x] Cora ablation: no_force_margin 76.43% (paper: 84.36%)
- [ ] Train on Chameleon (target: 72.56%)
- [ ] Train on ZINC (target: MAE 0.078)
- [ ] Train on Peptides-func (target: AP 0.714)
- [ ] Train on ogbn-proteins (target: ROC-AUC 81.24%) - may skip if too slow
- [ ] Train on ogbg-molpcba (target: AP 0.278) - may skip if too slow
- [ ] Fix ablation gaps
- [ ] Package: reproduce.sh, results/, REPORT.md

## Paper Results to Reproduce (Table 1)
| Dataset | Metric | Paper Target | Our Result | Status |
|---------|--------|------|------|--------|
| Cora | Acc↑ | 86.26±0.78 | 90.03±0.94 | ✓ |
| Chameleon | Acc↑ | 72.56±1.24 | - | TODO |
| ogbn-proteins | ROC-AUC↑ | 81.24±0.63 | - | TODO |
| ZINC | MAE↓ | 0.078±0.025 | - | TODO |
| Peptides-func | AP↑ | 0.714±0.022 | - | TODO |
| ogbg-molpcba | AP↑ | 0.278±0.003 | - | TODO |

## Ablation Results (Cora)
| Ablation | Paper | Ours | Notes |
|----------|-------|------|-------|
| Full | 86.26 | 90.03 | Exceeds paper |
| w/o hysteresis | 83.24 | 89.73 | Gap too small - need to fix |
| w/o topo search | 84.14 | 74.67 | Gap too large |
| w/o force margin | 84.36 | 76.43 | Gap too large |

## Key Files
- `/workspace/hgode_clean/model.py` - Main HGODE model (node + graph level)
- `/workspace/hgode_clean/train.py` - Training script for all 6 datasets + ablations
- `/workspace/results/` - JSON results files

## Key Hyperparameters (from paper Table 6)
- Cora: λ=0.1, τ=0.2, s=1.0, δ=0.1, β=0.1, hidden=256
- Chameleon: λ=0.5, τ=0.1, s=1.5, δ=0.2, β=0.3
- ZINC: λ=0.1, τ=0.2, s=1.0, δ=0.1, β=0.1
- Peptides-func: λ=0.5, τ=0.1, s=1.5, δ=0.2, β=0.1
- ogbn-proteins: λ=0.5, τ=0.1, s=1.5, δ=0.2, β=0.1
- ogbg-molpcba: λ=0.3, τ=0.2, s=1.0, δ=0.1, β=0.1

## Architecture Details
- G_φ(H,A) = PH - H with P = D^{-1}A (row-normalized diffusion)
- Force: F_ij = s * tanh(MLP([h_i || h_j]))
- U dynamics: dU/dt = (1-λ)U - U³ + F (hysteresis)
- H dynamics: τ_feat * dH/dt = G_φ(H,A) - γH
- A_ij = σ(U_ij/τ) (effective adjacency)
- U_init: +u_stable for observed, -u_stable for candidates
- Solver: RK4 with 10 steps, T=0.6

## Failed Approaches (from earlier)
1. dopri5 solver: Very slow convergence
2. rk4 with T=0.3, 10 steps: 63-65% test accuracy - severe overfitting
3. rk4 with original edges only, identity-init W: 78%
4. Various lr/wd/dropout combos: None got above ~78% with old architecture
5. Large candidate pool (96k 2-hop edges): Makes overfitting worse
6. Current no_hysteresis ablation: removing cubic term doesn't degrade enough
