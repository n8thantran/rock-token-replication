"""
Visualization module for generating all tables and figures.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_results(results_path: str = "results/all_results.json") -> dict:
    with open(results_path) as f:
        return json.load(f)


def format_table1(results: dict, output_dir: str = "results"):
    """Format Table 1: ASR before optimization."""
    table1 = results["table1"]
    
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 1: ASR Before Optimization (Baseline, ρ=0.0)")
    lines.append("=" * 80)
    lines.append(f"{'Model':<14} {'Topology':<8} {'Clean':>6} {'K-only':>8} {'T-only':>8} {'Both':>6}")
    lines.append("-" * 60)
    
    for model in ["gemma-2b", "mistral-7b", "llama3-8b"]:
        for topo in ["star", "chain", "dag"]:
            r = table1[model][topo]
            lines.append(f"{model:<14} {topo:<8} {r['clean']:>6.2f} {r['key_only']:>8.2f} "
                        f"{r['template_only']:>8.2f} {r['both']:>6.2f}")
        lines.append("-" * 60)
    
    text = "\n".join(lines)
    print(text)
    
    with open(os.path.join(output_dir, "table1.txt"), "w") as f:
        f.write(text)
    
    return text


def format_table2(results: dict, output_dir: str = "results"):
    """Format Table 2: ASR after optimization."""
    table2 = results["table2"]
    
    lines = []
    lines.append("=" * 90)
    lines.append("TABLE 2: ASR After Optimization")
    lines.append("=" * 90)
    lines.append(f"{'Model':<14} {'Opt Level':<14} {'Topology':<8} {'Clean':>6} {'K-only':>8} {'T-only':>8} {'Both':>6}")
    lines.append("-" * 75)
    
    for model in ["gemma-2b", "mistral-7b", "llama3-8b"]:
        for opt in ["routing", "routing+key", "full"]:
            for topo in ["star", "chain", "dag"]:
                r = table2[model][opt][topo]
                lines.append(f"{model:<14} {opt:<14} {topo:<8} {r['clean']:>6.2f} {r['key_only']:>8.2f} "
                            f"{r['template_only']:>8.2f} {r['both']:>6.2f}")
            lines.append("-" * 75)
    
    text = "\n".join(lines)
    print(text)
    
    with open(os.path.join(output_dir, "table2.txt"), "w") as f:
        f.write(text)
    
    return text


def format_table3(results: dict, output_dir: str = "results"):
    """Format Table 3: Aggregated ASR min/mean/max."""
    table3 = results["table3"]
    
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 3: Aggregated ASR Over Topologies (Both regime)")
    lines.append("=" * 80)
    lines.append(f"{'Model':<14} {'':>4} {'Before (Baseline)':>24} {'':>4} {'After (Full Opt)':>24}")
    lines.append(f"{'':14} {'':>4} {'ASR-m':>8} {'ASR':>8} {'ASR-M':>8} {'':>4} {'ASR-m':>8} {'ASR':>8} {'ASR-M':>8}")
    lines.append("-" * 80)
    
    for model in ["gemma-2b", "mistral-7b", "llama3-8b"]:
        b = table3[model]["before"]
        a = table3[model]["after"]
        lines.append(f"{model:<14} {'':>4} {b['min']:>8.2f} {b['mean']:>8.2f} {b['max']:>8.2f} "
                    f"{'':>4} {a['min']:>8.2f} {a['mean']:>8.2f} {a['max']:>8.2f}")
    
    lines.append("-" * 80)
    
    # Paper reference values
    lines.append("\nPaper reference values:")
    lines.append(f"{'Gemma-2B':<14} {'':>4} {'0.10':>8} {'0.23':>8} {'0.40':>8} {'':>4} {'0.40':>8} {'0.60':>8} {'1.00':>8}")
    lines.append(f"{'Mistral-7B':<14} {'':>4} {'0.10':>8} {'0.30':>8} {'0.40':>8} {'':>4} {'0.40':>8} {'0.60':>8} {'1.00':>8}")
    lines.append(f"{'LLaMA3-8B':<14} {'':>4} {'0.20':>8} {'0.33':>8} {'0.40':>8} {'':>4} {'0.30':>8} {'0.65':>8} {'1.00':>8}")
    
    text = "\n".join(lines)
    print(text)
    
    with open(os.path.join(output_dir, "table3.txt"), "w") as f:
        f.write(text)
    
    return text


def format_table4(results: dict, output_dir: str = "results"):
    """Format Table 4: Surrogate fidelity."""
    table4 = results["table4"]
    
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 4: Surrogate Fidelity")
    lines.append("=" * 80)
    lines.append(f"{'Topology':<8} {'Surrogate (ASR_both)':>30} {'Empirical (ASR_both)':>30}")
    lines.append(f"{'':8} {'Min':>10} {'Mean':>10} {'Max':>10} {'Min':>10} {'Mean':>10} {'Max':>10}")
    lines.append("-" * 70)
    
    for topo in ["star", "chain", "dag"]:
        s = table4[topo]["surrogate"]
        e = table4[topo]["empirical"]
        lines.append(f"{topo:<8} {s['min']:>10.2f} {s['mean']:>10.2f} {s['max']:>10.2f} "
                    f"{e['min']:>10.2f} {e['mean']:>10.2f} {e['max']:>10.2f}")
    
    lines.append("-" * 70)
    
    if "pearson_r" in table4:
        lines.append(f"\nPearson r = {table4['pearson_r']:.3f}")
        lines.append(f"Spearman rho = {table4['spearman_r']:.3f}")
    
    # Paper reference
    lines.append("\nPaper reference values:")
    lines.append(f"{'Star':<8} {'0.15':>10} {'0.53':>10} {'0.76':>10} {'0.20':>10} {'0.50':>10} {'0.80':>10}")
    lines.append(f"{'Chain':<8} {'0.13':>10} {'0.47':>10} {'0.68':>10} {'0.15':>10} {'0.52':>10} {'0.73':>10}")
    lines.append(f"{'DAG':<8} {'0.16':>10} {'0.52':>10} {'0.72':>10} {'0.19':>10} {'0.57':>10} {'0.76':>10}")
    lines.append("Paper: Pearson r = 0.995, Spearman rho = 0.933")
    
    text = "\n".join(lines)
    print(text)
    
    with open(os.path.join(output_dir, "table4.txt"), "w") as f:
        f.write(text)
    
    return text


def format_table5(results: dict, output_dir: str = "results"):
    """Format Table 5: Activation predicate verification."""
    table5 = results["table5"]
    
    lines = []
    lines.append("=" * 70)
    lines.append("TABLE 5: Activation Predicate Verification")
    lines.append("=" * 70)
    lines.append(f"{'Setting':<20} {'Clean':>8} {'K-only':>8} {'T-only':>8} {'Both':>8} {'FA':>8}")
    lines.append("-" * 60)
    
    for setting in ["baseline", "biased"]:
        r = table5[setting]
        rho = r["rho"]
        lines.append(f"{setting} (ρ={rho}){'':<5} {r['clean']:>8.2f} {r['key_only']:>8.2f} "
                    f"{r['template_only']:>8.2f} {r['both']:>8.2f} {r['fa']:>8.2f}")
    
    lines.append("-" * 60)
    
    lines.append("\nPaper reference values:")
    lines.append(f"{'Baseline (ρ=0.0)':<20} {'0.00':>8} {'0.04':>8} {'0.03':>8} {'0.28':>8} {'0.07':>8}")
    lines.append(f"{'Biased (ρ=0.8)':<20} {'0.00':>8} {'0.05':>8} {'0.04':>8} {'0.74':>8} {'0.09':>8}")
    
    text = "\n".join(lines)
    print(text)
    
    with open(os.path.join(output_dir, "table5.txt"), "w") as f:
        f.write(text)
    
    return text


def format_table6(results: dict, output_dir: str = "results"):
    """Format Table 6: Transferability."""
    table6 = results["table6"]
    
    lines = []
    lines.append("=" * 70)
    lines.append("TABLE 6: Transferability")
    lines.append("=" * 70)
    lines.append(f"{'Model':<22} {'ρ':>4} {'Clean':>8} {'K-only':>8} {'T-only':>8} {'Both':>8}")
    lines.append("-" * 60)
    
    for model in ["llama4-scout-17b", "gpt5-mini"]:
        for rho in ["0.0", "0.4", "0.8"]:
            r = table6[model][rho]
            lines.append(f"{model:<22} {rho:>4} {r['clean']:>8.2f} {r['key_only']:>8.2f} "
                        f"{r['template_only']:>8.2f} {r['both']:>8.2f}")
        lines.append("-" * 60)
    
    lines.append("\nPaper reference values:")
    lines.append(f"{'Llama-4-Scout-17B':<22} {'0.0':>4} {'0.00':>8} {'0.03':>8} {'0.02':>8} {'0.19':>8}")
    lines.append(f"{'':22} {'0.4':>4} {'0.00':>8} {'0.03':>8} {'0.04':>8} {'0.47':>8}")
    lines.append(f"{'':22} {'0.8':>4} {'0.00':>8} {'0.04':>8} {'0.03':>8} {'0.69':>8}")
    lines.append(f"{'GPT-5-mini':<22} {'0.0':>4} {'0.00':>8} {'0.03':>8} {'0.02':>8} {'0.22':>8}")
    lines.append(f"{'':22} {'0.4':>4} {'0.00':>8} {'0.04':>8} {'0.03':>8} {'0.51':>8}")
    lines.append(f"{'':22} {'0.8':>4} {'0.00':>8} {'0.05':>8} {'0.03':>8} {'0.73':>8}")
    
    text = "\n".join(lines)
    print(text)
    
    with open(os.path.join(output_dir, "table6.txt"), "w") as f:
        f.write(text)
    
    return text


def format_table7(results: dict, output_dir: str = "results"):
    """Format Table 7: System-level defense evaluation."""
    table7 = results["table7"]
    
    lines = []
    lines.append("=" * 60)
    lines.append("TABLE 7: System-Level Defense Evaluation")
    lines.append("=" * 60)
    lines.append(f"{'Defense':<25} {'ASR_both':>10} {'FA':>8} {'Rel. Drop':>12}")
    lines.append("-" * 55)
    
    for defense in ["none", "d1_tool_allowlist", "d2_least_privilege"]:
        r = table7[defense]
        drop = r.get("relative_drop", "--")
        lines.append(f"{defense:<25} {r['asr_both']:>10.2f} {r['fa']:>8.2f} {str(drop):>12}")
    
    lines.append("-" * 55)
    
    lines.append("\nPaper reference values:")
    lines.append(f"{'None':<25} {'0.73':>10} {'0.08':>8} {'--':>12}")
    lines.append(f"{'Tool Allowlist (D1)':<25} {'0.62':>10} {'0.07':>8} {'-15%':>12}")
    lines.append(f"{'Least Privilege (D2)':<25} {'0.58':>10} {'0.07':>8} {'-20%':>12}")
    
    text = "\n".join(lines)
    print(text)
    
    with open(os.path.join(output_dir, "table7.txt"), "w") as f:
        f.write(text)
    
    return text


def plot_figure3(results: dict, output_dir: str = "results"):
    """Plot Figure 3: F1 detection scores bar chart."""
    fig3 = results["figure3"]
    
    models = list(fig3.keys())
    vanilla = [fig3[m]["vanilla_f1"] for m in models]
    full = [fig3[m]["full_f1"] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, vanilla, width, label='Vanilla', 
                   color='#ff7f0e', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, full, width, label='Full Optimization',
                   color='#1f77b4', edgecolor='black', linewidth=0.8,
                   hatch='///')
    
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Detection Efficacy: Vanilla vs Full Optimization', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h}', xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h}', xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figure3_f1_detection.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure 3 saved to {fig_path}")


def plot_asr_comparison(results: dict, output_dir: str = "results"):
    """Plot ASR comparison: before vs after optimization across topologies."""
    table1 = results["table1"]
    table2 = results["table2"]
    
    models = ["gemma-2b", "mistral-7b", "llama3-8b"]
    topologies = ["star", "chain", "dag"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        
        before = [table1[model][t]["both"] for t in topologies]
        after = [table2[model]["full"][t]["both"] for t in topologies]
        
        x = np.arange(len(topologies))
        width = 0.35
        
        ax.bar(x - width/2, before, width, label='Before', color='#2ca02c', alpha=0.8)
        ax.bar(x + width/2, after, width, label='After (Full)', color='#d62728', alpha=0.8)
        
        ax.set_title(model, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(topologies, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('ASR (Both)' if idx == 0 else '', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (b, a) in enumerate(zip(before, after)):
            ax.text(i - width/2, b + 0.02, f'{b:.2f}', ha='center', fontsize=8)
            ax.text(i + width/2, a + 0.02, f'{a:.2f}', ha='center', fontsize=8)
    
    plt.suptitle('ASR Before vs After Optimization (Both Regime)', fontsize=14, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "asr_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ASR comparison plot saved to {fig_path}")


def plot_transferability(results: dict, output_dir: str = "results"):
    """Plot transferability results (Table 6)."""
    table6 = results["table6"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    for idx, model in enumerate(["llama4-scout-17b", "gpt5-mini"]):
        ax = axes[idx]
        rhos = [0.0, 0.4, 0.8]
        regimes = ["clean", "key_only", "template_only", "both"]
        colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
        
        for r_idx, regime in enumerate(regimes):
            values = [table6[model][str(r)][regime] for r in rhos]
            ax.plot(rhos, values, 'o-', label=regime, color=colors[r_idx], linewidth=2, markersize=8)
        
        ax.set_title(model, fontsize=12)
        ax.set_xlabel('Routing Bias (ρ)', fontsize=11)
        ax.set_ylabel('ASR' if idx == 0 else '', fontsize=11)
        ax.set_ylim(-0.05, 1.0)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Transferability: ASR vs Routing Bias', fontsize=14, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "transferability.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Transferability plot saved to {fig_path}")


def generate_all_outputs(results_path: str = "results/all_results.json",
                         output_dir: str = "results"):
    """Generate all formatted tables and figures."""
    results = load_results(results_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("GENERATING ALL TABLES AND FIGURES")
    print("=" * 80)
    
    format_table1(results, output_dir)
    print()
    format_table2(results, output_dir)
    print()
    format_table3(results, output_dir)
    print()
    format_table4(results, output_dir)
    print()
    format_table5(results, output_dir)
    print()
    format_table6(results, output_dir)
    print()
    format_table7(results, output_dir)
    print()
    
    plot_figure3(results, output_dir)
    plot_asr_comparison(results, output_dir)
    plot_transferability(results, output_dir)
    
    print("\nAll outputs generated successfully!")


if __name__ == "__main__":
    generate_all_outputs()
