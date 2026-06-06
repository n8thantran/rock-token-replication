# HGODE Implementation Progress

## Current Phase: Planning

## Implementation Plan
- [ ] **Phase 1: Core HGODE model**
  - [ ] Candidate pool construction (2-hop, spectral, random)
  - [ ] Force field MLP (Eq. 9: F_ij = s * tanh(MLP([h_i || h_j])))
  - [ ] Coupled ODE dynamics (Eq. 5)
  - [ ] Effective adjacency from potentials (Eq. 6)
  - [ ] Graph neural operator G_φ (diffusion-style)
  - [ ] Margin loss (Eq. 10-11)
  - [ ] ODE integration via torchdiffeq (dopri5)

- [ ] **Phase 2: Synthetic experiments**
  - [ ] SBM graph generation
  - [ ] Soft-attention Graph ODE baseline
  - [ ] Monostability trap visualization (Figure 3)
  - [ ] Perturbation robustness experiment (Figure 4)

- [ ] **Phase 3: Real-world benchmarks**  
  - [ ] Cora (node classification, Accuracy)
  - [ ] Chameleon (node classification, Accuracy)
  - [ ] ogbn-proteins (node classification, ROC-AUC)
  - [ ] ZINC (graph regression, MAE)
  - [ ] Peptides-func (graph classification, AP)
  - [ ] ogbg-molpcba (graph classification, AP)

- [ ] **Phase 4: Ablation studies**
  - [ ] w/o hysteresis (remove cubic term)
  - [ ] w/o topology search (only observed edges)
  - [ ] w/o force margin (β=0)

## Key Design Decisions
- Solver: dopri5, rtol=atol=1e-5
- Optimizer: Adam
- Hyperparameter ranges from Table 5 in Appendix
- Dataset-specific starting configs from Table 6

## Key Equations Reference
- **Coupled ODE (Eq. 5)**: τ_feat * dH/dt = G_φ(H,A) - γH; τ_topo * dU/dt = (1-λ)U - U³ + F_θ(H)
- **Effective adj (Eq. 6)**: A_ij = σ(U_ij/τ) * μ(t) * 1[(i,j) ∈ E_cand]
- **Force (Eq. 9)**: F_ij = s * tanh(MLP([h_i || h_j]))
- **Margin loss (Eq. 10)**: softplus terms for positive/negative pairs
- **Total loss (Eq. 11)**: L = L_task + β * L_margin
- **F_crit = 2/(3√3) ≈ 0.3849**
- **Double-well equilibria at ±√(1-λ) when F=0**

## Completed Work
(none yet)

## Failed Approaches
(none yet)
