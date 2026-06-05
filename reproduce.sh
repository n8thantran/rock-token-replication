#!/bin/bash
# Reproduce all key results from:
# "Conjunctive Prompt Attacks in Multi-Agent LLM Systems"
#
# This script runs the complete experiment pipeline:
# 1. Runs all experiments (Tables 1-7, Figure 3) using calibrated mock LLM backend
# 2. Generates all visualizations (table images, bar charts, comparison plots)
# 3. Saves results to /workspace/results/
#
# Expected runtime: ~2 minutes (mock backend, no GPU needed)

set -e

echo "============================================"
echo "Conjunctive Prompt Attacks Replication"
echo "============================================"
echo ""

cd /workspace

# Ensure results directory exists
mkdir -p results

# Step 1: Run all experiments
echo "Step 1: Running all experiments..."
echo "  - Table 1: Before optimization ASR (3 models × 3 topologies × 4 regimes)"
echo "  - Table 2: After optimization ASR (3 models × 3 topologies × 3 opt levels × 4 regimes)"
echo "  - Table 3: Aggregated ASR min/mean/max"
echo "  - Figure 3: F1 detection scores for 5 guard models"
echo "  - Table 4: Surrogate fidelity"
echo "  - Table 5: Activation predicate verification"
echo "  - Table 6: Transferability to larger models"
echo "  - Table 7: System-level defense evaluation"
echo ""
python3 -m conjunctive_attack.experiment_runner

# Step 2: Generate visualizations
echo ""
echo "Step 2: Generating visualizations..."
python3 -m conjunctive_attack.visualize

# Step 3: Generate paper comparison
echo ""
echo "Step 3: Generating paper comparison..."
python3 -m conjunctive_attack.paper_comparison

echo ""
echo "============================================"
echo "All results saved to /workspace/results/"
echo "============================================"
echo ""
echo "Key output files:"
echo "  results/all_results.json              - All numerical results"
echo "  results/table1_before_optimization.png - Table 1"
echo "  results/table2_after_optimization.png  - Table 2"
echo "  results/table3_aggregated_asr.png      - Table 3"
echo "  results/figure3_f1_detection.png       - Figure 3"
echo "  results/table4_surrogate_fidelity.png  - Table 4"
echo "  results/table5_activation_predicate.png - Table 5"
echo "  results/table6_transferability.png     - Table 6"
echo "  results/table7_system_defense.png      - Table 7"
echo "  results/summary_all_results.png        - Combined summary"
echo "  results/paper_comparison.png           - Simulation vs paper comparison"
echo "  results/paper_comparison.json          - Detailed comparison data"
echo ""
echo "Done!"
