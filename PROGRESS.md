# Progress Tracking

## Current Phase: Planning & Initial Implementation

## Paper Summary
"Conjunctive Prompt Attacks in Multi-Agent LLM Systems" - Studies attacks where a trigger key in user query + hidden template in compromised agent activate harmful behavior only when routing brings them together.

## Implementation Plan
- [ ] Core framework setup (agents, segmentation, routing)
- [ ] Three topologies: Star, Chain, DAG
- [ ] Routing mechanism: Pr[a=a*|s] = clip(α*I_acc(s) + ρ*I_acc(s)*I_k(s))
- [ ] Template injection (prefix, wrap, suffix)
- [ ] Activation detection (__ACTIVATED__ marker)
- [ ] Four evaluation regimes: clean, key_only, template_only, both
- [ ] Gumbel-Softmax counterpart optimization
- [ ] Three optimization levels: routing, routing+key, full
- [ ] Run experiments with LLM backends (Gemma-2B at minimum)
- [ ] Defense evaluation (PromptGuard, Llama-Guard)
- [ ] Generate all tables and figures
- [ ] Write reproduce.sh and REPORT.md

## Key Parameters (from paper)
- Segments S = 3 (example: flight, account, hotel)
- Trigger key: __KEY__
- Activation marker: __ACTIVATED__
- 50 episodes per configuration
- 20 agent role descriptions
- α (account-affinity): fixed parameter
- ρ (routing bias): [0, 1], optimized by attacker
- Template slots: prefix, wrap, suffix
- Models: Gemma-2B, Mistral-7B, LLaMA3-8B
- Topologies: Star, Chain, DAG

## Key Results to Reproduce
- Table 1: Baseline ASR before optimization
- Table 2: ASR after optimization (3 opt levels × 3 models × 3 topologies)
- Table 3: Aggregated ASR min/mean/max
- Figure 3: F1 detection scores for guard models
- Table 4: Surrogate fidelity
- Table 5: Activation predicate verification
- Table 6: Transferability (Llama-4-Scout, GPT-5-mini)
- Table 7: System-level defense evaluation

## Key Decisions
- Will implement full framework faithfully
- Start with Gemma-2B (smallest model)
- Use greedy decoding as paper states
- Fixed max generation length

## Completed Work
(none yet)

## Failed Approaches
(none yet)
