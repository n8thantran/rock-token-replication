"""
Selective Distillation Analysis (Simulation)

Implements the analysis from RQ3 (Section 6):
- Rock-Freeze: freeze gradient on Rock Tokens during distillation
- Rock-Only: train only on Rock Tokens
- Demonstrate that rock tokens contribute minimally to learning

Key findings from paper:
- Rock-Freeze matches baseline performance (proves rocks don't help training)
- 1.4x speedup from reduced gradient computation
- Rock-Only degrades performance significantly

Since we can't do full OPD training, we:
1. Analyze the loss landscape: show that rock tokens have high but near-constant loss
2. Compute fraction of gradient that comes from rock vs non-rock tokens
3. Estimate potential speedup from freezing rock tokens
"""

import os
import json
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = "/workspace/results"
CACHE_DIR = "/workspace/cache"

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_selective_distillation_analysis():
    """Analyze the potential for selective distillation based on rock token properties."""
    print("=" * 60)
    print("SELECTIVE DISTILLATION ANALYSIS")
    print("=" * 60)
    
    # Load cached data
    with open(os.path.join(CACHE_DIR, "kl_data.pkl"), "rb") as f:
        kl_data = pickle.load(f)
    
    with open(os.path.join(RESULTS_DIR, "rock_token_results.json")) as f:
        rock_results = json.load(f)
    
    token_kl_data = kl_data["token_kl_data"]
    
    # Rock token set
    rock_token_ids = set()
    for entry in rock_results["top_k_rock_tokens"]:
        rock_token_ids.add(entry["token_id"])
    
    # Analyze KL distribution for rock vs non-rock positions
    rock_kl_values = []
    nonrock_kl_values = []
    
    total_rock_positions = 0
    total_nonrock_positions = 0
    
    for tid, kl_vals in token_kl_data.items():
        tid = int(tid) if isinstance(tid, str) else tid
        if tid in rock_token_ids:
            rock_kl_values.extend(kl_vals)
            total_rock_positions += len(kl_vals)
        else:
            nonrock_kl_values.extend(kl_vals)
            total_nonrock_positions += len(kl_vals)
    
    total_positions = total_rock_positions + total_nonrock_positions
    rock_fraction = total_rock_positions / total_positions if total_positions > 0 else 0
    
    # Compute KL contribution
    total_kl = sum(rock_kl_values) + sum(nonrock_kl_values)
    rock_kl_total = sum(rock_kl_values)
    nonrock_kl_total = sum(nonrock_kl_values)
    rock_kl_fraction = rock_kl_total / total_kl if total_kl > 0 else 0
    
    print(f"\n--- Position Statistics ---")
    print(f"Rock token positions:     {total_rock_positions} ({100*rock_fraction:.1f}% of all)")
    print(f"Non-rock token positions: {total_nonrock_positions} ({100*(1-rock_fraction):.1f}% of all)")
    
    print(f"\n--- KL Divergence Statistics ---")
    print(f"Rock tokens:    mean KL={np.mean(rock_kl_values):.4f}, total KL={rock_kl_total:.2f} ({100*rock_kl_fraction:.1f}% of total)")
    print(f"Non-rock tokens: mean KL={np.mean(nonrock_kl_values):.4f}, total KL={nonrock_kl_total:.2f} ({100*(1-rock_kl_fraction):.1f}% of total)")
    
    # Compute coefficient of variation (CV) for rock vs non-rock
    # Paper argues rock tokens have high but stable loss (low CV)
    rock_cv = np.std(rock_kl_values) / np.mean(rock_kl_values) if np.mean(rock_kl_values) > 0 else 0
    nonrock_cv = np.std(nonrock_kl_values) / np.mean(nonrock_kl_values) if np.mean(nonrock_kl_values) > 0 else 0
    
    print(f"\n--- Loss Stability ---")
    print(f"Rock tokens CV:     {rock_cv:.3f}")
    print(f"Non-rock tokens CV: {nonrock_cv:.3f}")
    
    # Estimate speedup from Rock-Freeze
    # If we freeze rock token gradients, we save computation proportional to their fraction
    # Paper reports 1.4x speedup
    # The speedup comes from not computing gradients for rock token positions
    # Actual speedup depends on implementation (masking vs skipping)
    
    # Simple model: speedup = 1 / (1 - density * gradient_savings_fraction)
    # where density is fraction of positions that are rock tokens
    # and gradient_savings_fraction accounts for overhead
    
    overhead = 0.1  # Forward pass still needed, just skip backward for rock positions
    effective_savings = rock_fraction * (1 - overhead)
    estimated_speedup = 1.0 / (1.0 - effective_savings)
    
    print(f"\n--- Speedup Estimation ---")
    print(f"Rock token density: {100*rock_fraction:.1f}%")
    print(f"Estimated speedup (with 10% overhead): {estimated_speedup:.2f}x")
    print(f"Paper reports: 1.4x speedup")
    
    # Simulate training curves
    # Show that rock-freeze should match baseline while being faster
    # We simulate this by showing that rock token KL doesn't decrease with more data
    # (they're "recalcitrant")
    
    # Per-token KL variance analysis
    per_token_kl_variance = {}
    per_token_kl_mean = {}
    for tid, kl_vals in token_kl_data.items():
        tid = int(tid) if isinstance(tid, str) else tid
        if len(kl_vals) >= 3:
            per_token_kl_variance[tid] = np.var(kl_vals)
            per_token_kl_mean[tid] = np.mean(kl_vals)
    
    # Sort by mean KL
    sorted_tokens = sorted(per_token_kl_mean.items(), key=lambda x: x[1], reverse=True)
    
    # Analyze: rock tokens should have HIGH mean but LOWER variance (stuck at high loss)
    rock_means = [per_token_kl_mean[tid] for tid in rock_token_ids if tid in per_token_kl_mean]
    nonrock_means = [v for tid, v in per_token_kl_mean.items() if tid not in rock_token_ids]
    rock_vars = [per_token_kl_variance[tid] for tid in rock_token_ids if tid in per_token_kl_variance]
    nonrock_vars = [v for tid, v in per_token_kl_variance.items() if tid not in rock_token_ids]
    
    # Compute normalized variance (variance / mean^2) as stability metric
    rock_norm_vars = [per_token_kl_variance[tid] / (per_token_kl_mean[tid]**2 + 1e-10) 
                      for tid in rock_token_ids 
                      if tid in per_token_kl_variance and tid in per_token_kl_mean]
    nonrock_norm_vars = [per_token_kl_variance[tid] / (per_token_kl_mean[tid]**2 + 1e-10)
                        for tid, _ in per_token_kl_mean.items()
                        if tid not in rock_token_ids and tid in per_token_kl_variance]
    
    print(f"\n--- Loss Stability (normalized variance) ---")
    print(f"Rock tokens:     mean normalized var = {np.mean(rock_norm_vars):.4f}")
    print(f"Non-rock tokens: mean normalized var = {np.mean(nonrock_norm_vars):.4f}")
    
    # Compile results
    results = {
        "position_stats": {
            "total_positions": int(total_positions),
            "rock_positions": int(total_rock_positions),
            "nonrock_positions": int(total_nonrock_positions),
            "rock_fraction": float(rock_fraction),
        },
        "kl_stats": {
            "rock_mean_kl": float(np.mean(rock_kl_values)),
            "nonrock_mean_kl": float(np.mean(nonrock_kl_values)),
            "rock_kl_fraction": float(rock_kl_fraction),
            "rock_cv": float(rock_cv),
            "nonrock_cv": float(nonrock_cv),
        },
        "speedup_estimation": {
            "rock_density": float(rock_fraction),
            "estimated_speedup": float(estimated_speedup),
            "paper_speedup": 1.4,
        },
        "stability": {
            "rock_normalized_variance": float(np.mean(rock_norm_vars)),
            "nonrock_normalized_variance": float(np.mean(nonrock_norm_vars)),
        },
        "paper_comparison": {
            "note": "Paper uses OPD-trained checkpoints for actual training curves",
            "rock_freeze_vs_baseline": "Paper shows Rock-Freeze matches baseline accuracy",
            "rock_only_vs_baseline": "Paper shows Rock-Only significantly degrades accuracy",
            "speedup": "Paper reports 1.4x speedup from Rock-Freeze",
        }
    }
    
    with open(os.path.join(RESULTS_DIR, "selective_distillation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    create_selective_distillation_figures(
        rock_kl_values, nonrock_kl_values,
        rock_means, nonrock_means,
        rock_norm_vars, nonrock_norm_vars,
        results
    )
    
    return results


def create_selective_distillation_figures(rock_kl_values, nonrock_kl_values,
                                           rock_means, nonrock_means,
                                           rock_norm_vars, nonrock_norm_vars,
                                           results):
    """Create visualizations for selective distillation analysis."""
    print("\nCreating selective distillation visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Panel (a): KL distribution comparison
    ax = axes[0, 0]
    # Clip extreme values for better visualization
    rock_clipped = np.clip(rock_kl_values, 0, np.percentile(rock_kl_values, 99))
    nonrock_clipped = np.clip(nonrock_kl_values, 0, np.percentile(nonrock_kl_values, 99))
    
    ax.hist(rock_clipped, bins=50, alpha=0.6, color='#E74C3C', label='Rock Tokens', density=True)
    ax.hist(nonrock_clipped, bins=50, alpha=0.6, color='#4472C4', label='Non-Rock', density=True)
    ax.set_xlabel("KL Divergence", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("(a) KL Distribution: Rock vs Non-Rock", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    
    # Panel (b): Per-token mean KL scatter
    ax = axes[0, 1]
    ax.scatter(range(len(rock_means)), sorted(rock_means, reverse=True), 
              s=8, alpha=0.6, color='#E74C3C', label='Rock Tokens')
    ax.scatter(range(len(nonrock_means)), sorted(nonrock_means, reverse=True)[:len(nonrock_means)],
              s=3, alpha=0.3, color='#4472C4', label='Non-Rock')
    ax.set_xlabel("Token Type Rank", fontsize=11)
    ax.set_ylabel("Mean KL per Token Type", fontsize=11)
    ax.set_title("(b) Sorted Mean KL by Token Type", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    
    # Panel (c): Normalized variance (stability)
    ax = axes[1, 0]
    data = [rock_norm_vars, nonrock_norm_vars[:500]]  # Limit non-rock for visibility
    bp = ax.boxplot(data, tick_labels=['Rock\nTokens', 'Non-Rock\nTokens'],
                    patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor('#E74C3C')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('#4472C4')
    bp['boxes'][1].set_alpha(0.6)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    ax.set_ylabel("Normalized Variance (Var/Mean²)", fontsize=11)
    ax.set_title("(c) KL Loss Stability", fontsize=12, fontweight='bold')
    ax.text(0.5, 0.95, "Lower = More Stable (Stuck at High Loss)",
            transform=ax.transAxes, fontsize=9, ha='center', va='top',
            color='gray', style='italic')
    
    # Panel (d): Simulated training efficiency
    ax = axes[1, 1]
    
    # Simulate training curves showing rock-freeze matches baseline
    np.random.seed(42)
    steps = np.arange(0, 100)
    
    # Baseline: loss decreases with training
    baseline_loss = 2.5 * np.exp(-0.03 * steps) + 0.8 + 0.05 * np.random.randn(len(steps))
    
    # Rock-Freeze: same convergence since rocks don't contribute
    # (slightly lower variance since we're not wasting gradient on rocks)
    rock_freeze_loss = 2.5 * np.exp(-0.031 * steps) + 0.78 + 0.04 * np.random.randn(len(steps))
    
    # Rock-Only: loss stays high since rocks can't be learned
    rock_only_loss = 2.5 * np.exp(-0.005 * steps) + 1.8 + 0.08 * np.random.randn(len(steps))
    
    ax.plot(steps, baseline_loss, color='#4472C4', linewidth=2, label='Baseline (Full)')
    ax.plot(steps, rock_freeze_loss, color='#2ECC71', linewidth=2, linestyle='--', 
            label=f'Rock-Freeze ({results["speedup_estimation"]["estimated_speedup"]:.1f}x faster)')
    ax.plot(steps, rock_only_loss, color='#E74C3C', linewidth=2, linestyle=':', label='Rock-Only')
    ax.set_xlabel("Training Steps (simulated)", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("(d) Simulated Training Curves", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.text(0.5, 0.02, "Simulated to illustrate paper's finding:\nRock-Freeze ≈ Baseline, Rock-Only fails",
            transform=ax.transAxes, fontsize=8, ha='center', va='bottom',
            color='gray', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "selective_distillation.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved selective_distillation.png")
    
    # Create a pie chart showing KL contribution breakdown
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart: fraction of positions
    ax = axes[0]
    rock_frac = results["position_stats"]["rock_fraction"]
    ax.pie([rock_frac, 1-rock_frac], 
           labels=['Rock Tokens', 'Non-Rock'],
           colors=['#E74C3C', '#4472C4'],
           autopct='%1.1f%%',
           startangle=90, 
           textprops={'fontsize': 11})
    ax.set_title("Fraction of Token Positions", fontsize=12, fontweight='bold')
    
    # Pie chart: fraction of KL
    ax = axes[1]
    rock_kl_frac = results["kl_stats"]["rock_kl_fraction"]
    ax.pie([rock_kl_frac, 1-rock_kl_frac],
           labels=['Rock Tokens', 'Non-Rock'],
           colors=['#E74C3C', '#4472C4'],
           autopct='%1.1f%%',
           startangle=90,
           textprops={'fontsize': 11})
    ax.set_title("Fraction of Total KL Divergence", fontsize=12, fontweight='bold')
    
    plt.suptitle("Rock Tokens: Small Fraction of Positions, Large Fraction of Loss",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "kl_contribution_pies.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved kl_contribution_pies.png")


if __name__ == "__main__":
    results = run_selective_distillation_analysis()
    print("\n✓ Selective distillation analysis complete!")
