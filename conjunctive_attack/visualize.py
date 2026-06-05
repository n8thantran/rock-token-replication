"""
Visualization module for generating all tables and figures from the paper.
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


# ============================================================
# Table 1: Before Optimization ASR
# ============================================================

def plot_table1(results: dict, output_dir: str = "results"):
    """Generate Table 1 as a formatted figure."""
    table1 = results["table1"]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    # Build table data
    col_labels = ["Model", "Topology", "Clean", "Key-only", "Template-only", "Both"]
    rows = []
    
    for model in ["gemma-2b", "mistral-7b", "llama3-8b"]:
        for topo in ["star", "chain", "dag"]:
            d = table1[model][topo]
            rows.append([
                model if topo == "star" else "",
                topo.capitalize(),
                f"{d['clean']:.2f}",
                f"{d['key_only']:.2f}",
                f"{d['template_only']:.2f}",
                f"{d['both']:.2f}",
            ])
    
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Highlight "both" column
    for i in range(1, len(rows) + 1):
        table[i, 5].set_facecolor('#D6E4F0')
    
    ax.set_title("Table 1: Attack Success Rates Before Optimization (Baseline)",
                  fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "table1_before_optimization.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved table1_before_optimization.png")


# ============================================================
# Table 2: After Optimization ASR
# ============================================================

def plot_table2(results: dict, output_dir: str = "results"):
    """Generate Table 2 as a formatted figure."""
    table2 = results["table2"]
    
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')
    
    col_labels = ["Model", "Topo",
                  "R/C", "R/K", "R/T", "R/B",
                  "R+K/C", "R+K/K", "R+K/T", "R+K/B",
                  "F/C", "F/K", "F/T", "F/B"]
    rows = []
    
    for model in ["gemma-2b", "mistral-7b", "llama3-8b"]:
        for topo in ["star", "chain", "dag"]:
            row = [model if topo == "star" else "", topo.capitalize()]
            for opt in ["routing", "routing+key", "full"]:
                d = table2[model][topo][opt]
                row.extend([
                    f"{d['clean']:.1f}",
                    f"{d['key_only']:.1f}",
                    f"{d['template_only']:.1f}",
                    f"{d['both']:.1f}",
                ])
            rows.append(row)
    
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    
    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Highlight "both" columns (indices 5, 9, 13)
    for i in range(1, len(rows) + 1):
        for j in [5, 9, 13]:
            table[i, j].set_facecolor('#D6E4F0')
    
    ax.set_title("Table 2: Attack Success Rates After Optimization",
                  fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "table2_after_optimization.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved table2_after_optimization.png")


# ============================================================
# Table 3: Aggregated ASR
# ============================================================

def plot_table3(results: dict, output_dir: str = "results"):
    """Generate Table 3 as a formatted figure."""
    table3 = results["table3"]
    
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis('off')
    
    col_labels = ["Model", "Before min", "Before mean", "Before max",
                  "After min", "After mean", "After max"]
    rows = []
    
    for model in ["gemma-2b", "mistral-7b", "llama3-8b"]:
        b = table3[model]["before"]
        a = table3[model]["after"]
        rows.append([
            model,
            f"{b['min']:.2f}", f"{b['mean']:.2f}", f"{b['max']:.2f}",
            f"{a['min']:.2f}", f"{a['mean']:.2f}", f"{a['max']:.2f}",
        ])
    
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)
    
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Highlight "after" columns
    for i in range(1, len(rows) + 1):
        for j in [4, 5, 6]:
            table[i, j].set_facecolor('#D6E4F0')
    
    ax.set_title("Table 3: Aggregated ASR Over Topologies (min/mean/max of 'both' regime)",
                  fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "table3_aggregated_asr.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved table3_aggregated_asr.png")


# ============================================================
# Figure 3: F1 Detection Scores
# ============================================================

def plot_figure3(results: dict, output_dir: str = "results"):
    """Generate Figure 3: bar chart of F1 scores."""
    fig3 = results["figure3"]
    
    guard_names = list(fig3["vanilla"].keys())
    vanilla_f1 = [fig3["vanilla"][g] for g in guard_names]
    full_f1 = [fig3["full"][g] for g in guard_names]
    
    x = np.arange(len(guard_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, vanilla_f1, width, label='Vanilla',
                    color='#7FB3D8', edgecolor='black', hatch='//')
    bars2 = ax.bar(x + width/2, full_f1, width, label='Full Optimization',
                    color='#4472C4', edgecolor='black', hatch='//')
    
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("Llama-Guard", "LG") for g in guard_names],
                        rotation=30, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    
    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 1,
                    f'{int(h)}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 1,
                    f'{int(h)}', ha='center', va='bottom', fontsize=9)
    
    ax.set_title("Figure 3: Detection Efficacy of Safety Mechanisms\n"
                  "(Vanilla vs Full Optimization)", fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure3_f1_detection.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figure3_f1_detection.png")


# ============================================================
# Table 4: Surrogate Fidelity
# ============================================================

def plot_table4(results: dict, output_dir: str = "results"):
    """Generate Table 4 as a formatted figure."""
    table4 = results["table4"]
    
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis('off')
    
    col_labels = ["Topology", "Surr Min", "Surr Mean", "Surr Max",
                  "Emp Min", "Emp Mean", "Emp Max"]
    rows = []
    
    for topo in ["star", "chain", "dag"]:
        s = table4[topo]["surrogate"]
        e = table4[topo]["empirical"]
        rows.append([
            topo.capitalize(),
            f"{s['min']:.2f}", f"{s['mean']:.2f}", f"{s['max']:.2f}",
            f"{e['min']:.2f}", f"{e['mean']:.2f}", f"{e['max']:.2f}",
        ])
    
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)
    
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    ax.set_title("Table 4: Surrogate Fidelity (P_route × P_template vs Empirical ASR)",
                  fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "table4_surrogate_fidelity.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved table4_surrogate_fidelity.png")


# ============================================================
# Table 5: Activation Predicate Verification
# ============================================================

def plot_table5(results: dict, output_dir: str = "results"):
    """Generate Table 5 as a formatted figure."""
    table5 = results["table5"]
    
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    
    col_labels = ["Setting", "Clean", "K-only", "T-only", "Both", "FA"]
    rows = []
    
    for setting in ["baseline", "biased"]:
        d = table5[setting]
        label = f"Baseline (ρ={d['rho']})" if setting == "baseline" else f"Biased (ρ={d['rho']})"
        rows.append([
            label,
            f"{d['clean']:.2f}",
            f"{d['key_only']:.2f}",
            f"{d['template_only']:.2f}",
            f"{d['both']:.2f}",
            f"{d['fa']:.2f}",
        ])
    
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)
    
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Highlight "both" column
    for i in range(1, len(rows) + 1):
        table[i, 4].set_facecolor('#D6E4F0')
    
    ax.set_title("Table 5: Activation Predicate Verification",
                  fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "table5_activation_predicate.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved table5_activation_predicate.png")


# ============================================================
# Table 6: Transferability
# ============================================================

def plot_table6(results: dict, output_dir: str = "results"):
    """Generate Table 6 as a formatted figure."""
    table6 = results["table6"]
    
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis('off')
    
    col_labels = ["Model", "ρ", "Clean", "K-only", "T-only", "Both"]
    rows = []
    
    for model in ["Llama-4-Scout-17B", "GPT-5-mini"]:
        for rho in ["0.0", "0.4", "0.8"]:
            d = table6[model][rho]
            rows.append([
                model if rho == "0.0" else "",
                rho,
                f"{d['clean']:.2f}",
                f"{d['key_only']:.2f}",
                f"{d['template_only']:.2f}",
                f"{d['both']:.2f}",
            ])
    
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    for i in range(1, len(rows) + 1):
        table[i, 5].set_facecolor('#D6E4F0')
    
    ax.set_title("Table 6: Transferability to Larger/Closed-Source Models",
                  fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "table6_transferability.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved table6_transferability.png")


# ============================================================
# Table 7: System-Level Defense
# ============================================================

def plot_table7(results: dict, output_dir: str = "results"):
    """Generate Table 7 as a formatted figure."""
    table7 = results["table7"]
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    
    col_labels = ["Defense", "ASR_both", "FA", "Relative Drop"]
    rows = []
    
    for defense in ["None", "Tool Allowlist (D1)", "Least Privilege (D2)"]:
        d = table7[defense]
        rows.append([
            defense,
            f"{d['asr_both']:.2f}",
            f"{d['fa']:.2f}",
            d['relative_drop'],
        ])
    
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)
    
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    ax.set_title("Table 7: System-Level Defense Evaluation (ρ=0.8, Closed-Source Backbone)",
                  fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "table7_system_defense.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved table7_system_defense.png")


# ============================================================
# Combined summary figure
# ============================================================

def plot_summary(results: dict, output_dir: str = "results"):
    """Generate a summary figure showing key findings."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Before vs After ASR (Table 3 data)
    ax = axes[0, 0]
    table3 = results["table3"]
    models = ["gemma-2b", "mistral-7b", "llama3-8b"]
    x = np.arange(len(models))
    width = 0.35
    
    before_means = [table3[m]["before"]["mean"] for m in models]
    after_means = [table3[m]["after"]["mean"] for m in models]
    before_errs = [[table3[m]["before"]["mean"] - table3[m]["before"]["min"] for m in models],
                   [table3[m]["before"]["max"] - table3[m]["before"]["mean"] for m in models]]
    after_errs = [[table3[m]["after"]["mean"] - table3[m]["after"]["min"] for m in models],
                  [table3[m]["after"]["max"] - table3[m]["after"]["mean"] for m in models]]
    
    ax.bar(x - width/2, before_means, width, yerr=before_errs, label='Before',
           color='#7FB3D8', edgecolor='black', capsize=5)
    ax.bar(x + width/2, after_means, width, yerr=after_errs, label='After (Full)',
           color='#4472C4', edgecolor='black', capsize=5)
    ax.set_ylabel('ASR (both regime)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.set_title('(a) ASR Before vs After Optimization')
    ax.grid(axis='y', alpha=0.3)
    
    # Panel 2: F1 Detection (Figure 3 data)
    ax = axes[0, 1]
    fig3 = results["figure3"]
    guard_names = list(fig3["vanilla"].keys())
    short_names = [g.replace("Llama-Guard-", "LG-").replace("PromptGuard-86M", "PG-86M") 
                   for g in guard_names]
    vanilla_f1 = [fig3["vanilla"][g] for g in guard_names]
    full_f1 = [fig3["full"][g] for g in guard_names]
    
    x = np.arange(len(guard_names))
    ax.bar(x - width/2, vanilla_f1, width, label='Vanilla', color='#7FB3D8', edgecolor='black')
    ax.bar(x + width/2, full_f1, width, label='Full Opt', color='#4472C4', edgecolor='black')
    ax.set_ylabel('F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha='right', fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title('(b) Guard Model Detection F1')
    ax.grid(axis='y', alpha=0.3)
    
    # Panel 3: Transferability (Table 6 data)
    ax = axes[1, 0]
    table6 = results["table6"]
    rhos = [0.0, 0.4, 0.8]
    for model_name in ["Llama-4-Scout-17B", "GPT-5-mini"]:
        both_vals = [table6[model_name][str(r)]["both"] for r in rhos]
        ax.plot(rhos, both_vals, 'o-', label=model_name, linewidth=2, markersize=8)
    ax.set_xlabel('Routing Bias ρ')
    ax.set_ylabel('ASR (both regime)')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.set_title('(c) Transferability to Larger Models')
    ax.grid(alpha=0.3)
    
    # Panel 4: System Defense (Table 7 data)
    ax = axes[1, 1]
    table7 = results["table7"]
    defense_names = list(table7.keys())
    asr_vals = [table7[d]["asr_both"] for d in defense_names]
    fa_vals = [table7[d]["fa"] for d in defense_names]
    
    x = np.arange(len(defense_names))
    bars = ax.bar(x, asr_vals, 0.5, color=['#E74C3C', '#F39C12', '#2ECC71'], edgecolor='black')
    ax.set_ylabel('ASR (both regime)')
    ax.set_xticks(x)
    ax.set_xticklabels(["None", "D1: Tool\nAllowlist", "D2: Least\nPrivilege"],
                        fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title('(d) System-Level Defense Evaluation')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, asr_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle("Conjunctive Prompt Attacks in Multi-Agent LLM Systems\n"
                 "Key Results Summary", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_all_results.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved summary_all_results.png")


def generate_all_plots(results_path: str = "results/all_results.json",
                       output_dir: str = "results"):
    """Generate all plots from saved results."""
    results = load_results(results_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating visualizations...")
    plot_table1(results, output_dir)
    plot_table2(results, output_dir)
    plot_table3(results, output_dir)
    plot_figure3(results, output_dir)
    plot_table4(results, output_dir)
    plot_table5(results, output_dir)
    plot_table6(results, output_dir)
    plot_table7(results, output_dir)
    plot_summary(results, output_dir)
    print("All visualizations generated!")


if __name__ == "__main__":
    generate_all_plots()
