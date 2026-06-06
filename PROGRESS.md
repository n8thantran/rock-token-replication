# HGODE Implementation Progress

## Current Phase: Building training pipeline for real-world benchmarks

## Implementation Plan
- [x] **Phase 1: Core HGODE model** (TESTED ✓)
  - [x] Candidate pool construction (2-hop, random)
  - [x] Force field MLP (Eq. 9: F_ij = s * tanh(MLP([h_i || h_j])))
  - [x] Coupled ODE dynamics (Eq. 5)
  - [x] Effective adjacency from potentials (Eq. 6)
  - [x] Graph neural operator G_φ (diffusion-style)
  - [x] Margin loss (Eq. 10-11)
  - [x] ODE integration via torchdiffeq (dopri5)

- [ ] **Phase 2: Synthetic experiments**
  - [ ] SBM graph generation
  - [ ] Soft-attention Graph ODE baseline
  - [ ] Monostability trap visualization (Figure 3)
  - [ ] Perturbation robustness experiment (Figure 4)

- [ ] **Phase 3: Real-world benchmarks** (PRIORITY - main results table)
  - [ ] Cora (node classification, Accuracy) - target: 86.26±0.78
  - [ ] Chameleon (node classification, Accuracy) - target: 72.56±1.24
  - [ ] ogbn-proteins (node classification, ROC-AUC) - target: 81.24±0.63
  - [ ] ZINC (graph regression, MAE) - target: 0.078±0.025
  - [ ] Peptides-func (graph classification, AP) - target: 0.714±0.022
  - [ ] ogbg-molpcba (graph classification, AP) - target: 0.278±0.003

- [ ] **Phase 4: Ablation studies** (Table 1 bottom rows)
  - [ ] w/o hysteresis (remove cubic term)
  - [ ] w/o topology search (only observed edges)
  - [ ] w/o force margin (β=0)

## Key Design Decisions
- Solver: dopri5, rtol=atol=1e-5
- Optimizer: Adam
- Dataset-specific configs from Table 6:
  - Cora: λ∈[0.1,0.3], τ∈[0.2,0.3], s=1.0, δ=0.1, β∈{0,0.1}
  - Chameleon: λ∈[0.4,0.6], τ∈[0.05,0.1], s∈[1.0,1.5], δ∈[0.2,0.3], β∈[0.3,0.5]
  - ogbn-proteins: λ∈[0.5,0.8], τ∈[0.05,0.1], s∈[1.0,1.5], δ∈[0.2,0.3], β∈[0.1,0.3]
  - ZINC: λ∈[0.1,0.3], τ∈[0.2,0.3], s=1.0, δ=0.1, β∈{0,0.1}
  - Peptides-func: λ∈[0.5,0.8], τ∈[0.05,0.1], s∈[1.0,1.5], δ∈[0.2,0.3], β∈[0.1,0.3]
  - ogbg-molpcba: λ∈[0.3,0.5], τ∈[0.1,0.2], s∈[1.0,1.5], δ∈[0.1,0.2], β∈[0.1,0.3]

## Paper Results to Reproduce (Table 1)
| Dataset | Metric | HGODE | w/o hysteresis | w/o topo search | w/o force margin |
|---------|--------|-------|----------------|-----------------|------------------|
| Cora | Acc↑ | 86.26±0.78 | 83.24±0.32 | 84.14±0.46 | 84.36±0.19 |
| Chameleon | Acc↑ | 72.56±1.24 | 66.24±1.26 | 70.44±1.41 | 61.24±0.73 |
| ogbn-proteins | ROC-AUC↑ | 81.24±0.63 | 75.26±0.15 | 77.19±0.52 | 80.24±0.77 |
| ZINC | MAE↓ | 0.078±0.025 | 0.145±0.032 | 0.162±0.017 | 0.172±0.080 |
| Peptides-func | AP↑ | 0.714±0.022 | 0.671±0.013 | 0.653±0.041 | 0.689±0.034 |
| ogbg-molpcba | AP↑ | 0.278±0.003 | 0.254±0.005 | 0.262±0.002 | 0.260±0.003 |

## Completed Work
- hgode/__init__.py - package init
- hgode/candidate_pool.py - candidate edge pool construction (tested)
- hgode/model.py - core HGODE model with ForceFieldMLP, GraphNeuralOperator, HGODEFunc, HGODE, HGODEForGraphClassification (tested basic forward pass)

## Failed Approaches
(none yet)
