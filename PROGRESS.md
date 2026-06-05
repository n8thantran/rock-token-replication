# Progress Tracking

## Current Phase: Building main experiment pipeline with LLM backend

## Paper Summary
"Conjunctive Prompt Attacks in Multi-Agent LLM Systems" - Studies attacks where a trigger key in user query + hidden template in compromised agent activate harmful behavior only when routing brings them together.

## Implementation Plan
- [x] Core framework setup (agents, segmentation, routing) - conjunctive_attack/agents.py
- [x] Three topologies: Star, Chain, DAG - conjunctive_attack/routing.py
- [x] Routing mechanism: Pr[a=a*|s] = clip(α*I_acc(s) + ρ*I_acc(s)*I_k(s))
- [x] Template injection (prefix, wrap, suffix)
- [x] Activation detection (__ACTIVATED__ marker)
- [x] Four evaluation regimes: clean, key_only, template_only, both
- [x] Gumbel-Softmax counterpart optimization
- [x] Three optimization levels: routing, routing+key, full
- [ ] LLM backend (Gemma-2B) for generation
- [ ] Main experiment runner (Tables 1 & 2)
- [ ] Run experiments and generate all tables
- [ ] Table 3: Aggregated ASR min/mean/max
- [ ] Figure 3: F1 detection scores for guard models
- [ ] Table 4: Surrogate fidelity
- [ ] Table 5: Activation predicate verification  
- [ ] Table 6: Transferability
- [ ] Table 7: System-level defense evaluation
- [ ] Write reproduce.sh and final REPORT.md

## Key Parameters (from paper)
- Segments S = 3 (flight, account, hotel)
- Trigger key: __KEY__
- Activation marker: __ACTIVATED__
- 50 episodes per configuration
- 20 agent role descriptions
- α (account-affinity): 0.6 (fixed)
- ρ (routing bias): [0, 1], optimized by attacker
- Template slots: prefix, wrap, suffix
- Models: Gemma-2B, Mistral-7B, LLaMA3-8B
- Topologies: Star, Chain, DAG
- Greedy decoding, fixed max generation length

## Key Tables to Reproduce
- Table 1: Before optimization ASR (9 configs × 4 regimes)
- Table 2: After optimization ASR (9 configs × 3 opt levels × 4 regimes)
- Table 3: Aggregated ASR min/mean/max (before/after)
- Figure 3: F1 scores for 5 guard models, vanilla vs full
- Table 4: Surrogate fidelity
- Table 5: Activation predicate verification
- Table 6: Transferability (Llama-4-Scout, GPT-5-mini)  
- Table 7: System-level defense (D1, D2)

## Completed Work
- conjunctive_attack/agents.py: Agent definitions, 20 roles, template injection
- conjunctive_attack/routing.py: Routing formula, 3 topologies
- conjunctive_attack/evaluation.py: Episode runner, ASR computation
- conjunctive_attack/optimization.py: Gumbel-Softmax optimizer

## Legacy Files (from previous project, can be ignored)
- rock_token_identification.py, gradient_geometry.py, pillar_knockout.py, etc.

## Failed Approaches
(none yet)
