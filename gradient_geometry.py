"""
Gradient Geometry Analysis for Rock Tokens

Implements the gradient analysis from the paper (Section RQ1, Figure 2):
- Per-token logit-gradient magnitude ||g_t|| by group (rock, high-KL, random)
- Cosine alignment with balanced global gradient G_balanced
- Contribution decomposition: contrib(t) = n_t * ||g_t|| * cos(g_t, G_balanced)

Key findings to reproduce:
- Rock tokens: low gradient magnitude (~0.016) but high alignment (~0.040)
- High-KL tokens: high gradient magnitude (~0.54) but lower alignment (~0.025)
- Random tokens: low alignment (~0.006)
"""

import os
import json
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STUDENT_MODEL = "Qwen/Qwen3-4B"
DEVICE = "cuda"
RESULTS_DIR = "/workspace/results"
CACHE_DIR = "/workspace/cache"

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_cached_data():
    """Load cached rollouts and rock token results."""
    rollout_file = os.path.join(CACHE_DIR, "rollouts_and_logprobs.pkl")
    results_file = os.path.join(RESULTS_DIR, "rock_token_results.json")
    kl_file = os.path.join(CACHE_DIR, "kl_data.pkl")
    
    with open(rollout_file, "rb") as f:
        rollout_data = pickle.load(f)
    
    with open(results_file) as f:
        rock_results = json.load(f)
    
    with open(kl_file, "rb") as f:
        kl_data = pickle.load(f)
    
    return rollout_data, rock_results, kl_data


def classify_tokens(rock_results, kl_data):
    """Classify tokens into rock, high-KL (rare), and random groups."""
    # token_kl_data: {int_token_id: [list of kl values]}
    token_kl_data = kl_data["token_kl_data"]
    
    # Rock token set
    rock_token_ids = set()
    for entry in rock_results["top_k_rock_tokens"]:
        rock_token_ids.add(entry["token_id"])
    
    # Compute mean KL and frequency per token
    token_mean_kl = {}
    token_freq = {}
    for tid, kl_vals in token_kl_data.items():
        tid = int(tid) if isinstance(tid, str) else tid
        token_mean_kl[tid] = float(np.mean(kl_vals))
        token_freq[tid] = len(kl_vals)
    
    # High-KL tokens: high mean KL, low frequency (rare), NOT in rock set
    # These are "learnable" tokens the paper contrasts with rocks
    non_rock_rare = {tid: mkl for tid, mkl in token_mean_kl.items()
                     if tid not in rock_token_ids and token_freq.get(tid, 0) <= 5}
    sorted_high_kl = sorted(non_rock_rare.items(), key=lambda x: x[1], reverse=True)
    high_kl_token_ids = set(tid for tid, _ in sorted_high_kl[:100])
    
    # Random tokens: not rock, not high-KL, moderate frequency
    random_candidates = [tid for tid in token_mean_kl
                        if tid not in rock_token_ids 
                        and tid not in high_kl_token_ids
                        and 3 <= token_freq.get(tid, 0) <= 50]
    np.random.seed(42)
    np.random.shuffle(random_candidates)
    random_token_ids = set(random_candidates[:100])
    
    return rock_token_ids, high_kl_token_ids, random_token_ids, token_mean_kl, token_freq


def compute_gradient_geometry():
    """
    Compute per-token gradient geometry for Rock vs non-Rock tokens.
    
    For each token type t with n_t occurrences, compute:
    - Mean per-occurrence gradient in logit space: g_bar_t = (1/n_t) sum_i g_i
    - ||g_bar_t||: gradient magnitude
    - cos(g_bar_t, G_balanced): alignment with balanced global gradient
    - contrib(t) = n_t * ||g_bar_t|| * cos(g_bar_t, G_balanced)
    
    The gradient of reverse KL w.r.t. logits z at position i is:
    dL/dz = p_student - p_teacher (approximately, for the relevant dimensions)
    
    Since we have student logprobs and approximate teacher logprobs from cached data,
    we compute these gradients efficiently.
    """
    print("=" * 60)
    print("GRADIENT GEOMETRY ANALYSIS")
    print("=" * 60)
    
    # Load cached data
    rollout_data, rock_results, kl_data = load_cached_data()
    rollouts = rollout_data["rollouts"]
    student_data = rollout_data["student_data"]
    
    # Classify tokens
    rock_ids, high_kl_ids, random_ids, token_mean_kl, token_freq = \
        classify_tokens(rock_results, kl_data)
    
    all_target_ids = rock_ids | high_kl_ids | random_ids
    
    print(f"Rock tokens: {len(rock_ids)}")
    print(f"High-KL rare tokens: {len(high_kl_ids)}")
    print(f"Random tokens: {len(random_ids)}")
    
    # Load student model
    print("\nLoading student model...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    vocab_size = model.config.vocab_size
    
    # We'll work with a reduced dimensionality for memory efficiency
    # Instead of full vocab_size gradients, we track a compressed representation
    # using the top-K dimensions that matter most
    
    # Strategy: For each token occurrence, the gradient of cross-entropy loss 
    # w.r.t. logits is: g = p_student - one_hot(target)
    # For KL divergence: g = p_student - p_teacher
    # We approximate p_teacher from the KL data we have
    # 
    # Since we only have KL values (not full teacher distributions), we use:
    # g ≈ p_student - one_hot(generated_token) scaled by KL
    # This captures the magnitude and direction of the distillation signal
    
    # For a more tractable analysis, we project gradients onto a 
    # random subspace of dimension D_proj
    D_proj = 2000  # Project to this dimension
    torch.manual_seed(42)
    proj_matrix = torch.randn(vocab_size, D_proj, dtype=torch.float32) / np.sqrt(D_proj)
    # We'll apply this projection lazily
    
    # Per-token gradient accumulators (in projected space)
    token_grad_sum = defaultdict(lambda: torch.zeros(D_proj, dtype=torch.float32))
    token_grad_count = defaultdict(int)
    
    num_rollouts = min(20, len(rollouts))
    print(f"\nComputing per-token gradients for {num_rollouts} rollouts...")
    
    for r_idx in tqdm(range(num_rollouts), desc="Computing gradients"):
        rollout = rollouts[r_idx]
        sdata = student_data[r_idx]
        
        if sdata is None:
            continue
        
        full_ids = torch.tensor(rollout["full_ids"]).unsqueeze(0).to(DEVICE)
        prompt_len = rollout["prompt_len"]
        gen_ids = rollout["generated_ids"]
        gen_len = len(gen_ids)
        
        if gen_len < 2:
            continue
        
        # Forward pass to get student probabilities
        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            start_pos = prompt_len - 1
            end_pos = prompt_len + gen_len - 1
            gen_logits = logits[start_pos:end_pos]  # [gen_len, vocab_size]
            student_probs = torch.softmax(gen_logits.float(), dim=-1).cpu()  # [gen_len, V]
        
        del outputs, logits, gen_logits
        torch.cuda.empty_cache()
        
        # For each position, compute gradient and project
        for pos in range(min(gen_len, 300)):  # Limit positions
            token_id = gen_ids[pos]
            
            if token_id not in all_target_ids:
                continue
            
            # Gradient of cross-entropy w.r.t. logits: p_student - one_hot(token)
            # Scaled by KL to approximate distillation gradient magnitude
            grad_full = student_probs[pos].clone()  # p_student
            grad_full[token_id] -= 1.0  # subtract one_hot
            
            # Scale by mean KL for this token type
            kl_scale = token_mean_kl.get(token_id, 0.1)
            grad_full *= kl_scale
            
            # Project to lower dimension
            grad_proj = grad_full @ proj_matrix  # [D_proj]
            
            token_grad_sum[token_id] += grad_proj
            token_grad_count[token_id] += 1
    
    # Free model
    del model, proj_matrix
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nComputing gradient statistics...")
    
    # Compute mean gradients per token type
    token_mean_grad = {}
    for tid in all_target_ids:
        if token_grad_count[tid] > 0:
            token_mean_grad[tid] = token_grad_sum[tid] / token_grad_count[tid]
    
    # Compute balanced global gradient: G_balanced = sum_t g_bar_t (equal weight per type)
    G_balanced = torch.zeros(D_proj, dtype=torch.float32)
    for tid, grad in token_mean_grad.items():
        G_balanced += grad
    
    G_balanced_norm = torch.norm(G_balanced).item()
    print(f"G_balanced norm: {G_balanced_norm:.6f}")
    
    # Compute per-token statistics
    results = {"rock": [], "high_kl": [], "random": []}
    
    for tid, grad in token_mean_grad.items():
        grad_norm = torch.norm(grad).item()
        if grad_norm < 1e-12:
            cos_sim = 0.0
        else:
            cos_sim = (torch.dot(grad, G_balanced) / (grad_norm * G_balanced_norm + 1e-10)).item()
        freq = token_grad_count[tid]
        contrib = freq * grad_norm * cos_sim
        
        entry = {
            "token_id": int(tid),
            "token_str": tokenizer.decode([tid]),
            "grad_norm": grad_norm,
            "cos_sim": cos_sim,
            "freq": freq,
            "contrib": contrib,
            "mean_kl": token_mean_kl.get(tid, 0),
        }
        
        if tid in rock_ids:
            results["rock"].append(entry)
        elif tid in high_kl_ids:
            results["high_kl"].append(entry)
        elif tid in random_ids:
            results["random"].append(entry)
    
    # Print summary statistics
    for group_name, entries in results.items():
        if not entries:
            print(f"\n{group_name.upper()}: No data")
            continue
        norms = [e["grad_norm"] for e in entries]
        cosines = [e["cos_sim"] for e in entries]
        contribs = [e["contrib"] for e in entries]
        
        print(f"\n{group_name.upper()} tokens ({len(entries)} types):")
        print(f"  Gradient magnitude: median={np.median(norms):.6f}, mean={np.mean(norms):.6f}")
        print(f"  Cosine alignment:   median={np.median(cosines):.6f}, mean={np.mean(cosines):.6f}")
        print(f"  Contribution:       median={np.median(contribs):.6f}, mean={np.mean(contribs):.6f}")
    
    return results


def create_gradient_visualizations(results):
    """Create gradient geometry visualizations matching paper Figure 2."""
    print("\nCreating gradient geometry visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    colors = {"rock": "#4472C4", "high_kl": "#ED7D31", "random": "#A5A5A5"}
    labels = {"rock": "Rock Tokens", "high_kl": "High-KL (Rare)", "random": "Random"}
    
    # Panel (a): Gradient magnitude box plot (matching paper style)
    ax = axes[0]
    data_mag = []
    group_labels_list = []
    color_list = []
    for group in ["rock", "high_kl", "random"]:
        if results[group]:
            norms = [e["grad_norm"] for e in results[group]]
            data_mag.append(norms)
            group_labels_list.append(labels[group])
            color_list.append(colors[group])
    
    bp = ax.boxplot(data_mag, labels=[l.replace(" ", "\n") for l in group_labels_list], 
                    patch_artist=True, showfliers=False, widths=0.6)
    for patch, c in zip(bp['boxes'], color_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    ax.set_ylabel("||ḡ_t||", fontsize=12)
    ax.set_title("(a) Gradient Magnitude", fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9)
    
    # Panel (b): Cosine alignment box plot
    ax = axes[1]
    data_cos = []
    for group in ["rock", "high_kl", "random"]:
        if results[group]:
            cosines = [e["cos_sim"] for e in results[group]]
            data_cos.append(cosines)
    
    bp = ax.boxplot(data_cos, labels=[l.replace(" ", "\n") for l in group_labels_list],
                    patch_artist=True, showfliers=False, widths=0.6)
    for patch, c in zip(bp['boxes'], color_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    ax.set_ylabel("cos(ḡ_t, G_balanced)", fontsize=12)
    ax.set_title("(b) Alignment with Global Gradient", fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.tick_params(axis='x', labelsize=9)
    
    # Panel (c): Scatter plot - magnitude vs alignment, size by frequency
    ax = axes[2]
    for group in ["rock", "high_kl", "random"]:
        if results[group]:
            norms = [e["grad_norm"] for e in results[group]]
            cosines = [e["cos_sim"] for e in results[group]]
            freqs = [e["freq"] for e in results[group]]
            ax.scatter(norms, cosines, 
                      s=[max(5, min(f*3, 200)) for f in freqs],
                      alpha=0.5, color=colors[group], label=labels[group], 
                      edgecolors='none')
    ax.set_xlabel("||ḡ_t||", fontsize=12)
    ax.set_ylabel("cos(ḡ_t, G_balanced)", fontsize=12)
    ax.set_title("(c) Magnitude vs Alignment", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "gradient_geometry.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved gradient_geometry.png")
    
    # Create a contribution decomposition figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for group in ["rock", "high_kl", "random"]:
        if results[group]:
            entries = sorted(results[group], key=lambda x: x["contrib"], reverse=True)
            contribs = [e["contrib"] for e in entries]
            ax.bar(range(len(contribs)), contribs, alpha=0.6, 
                  color=colors[group], label=labels[group])
    
    ax.set_xlabel("Token Type Rank", fontsize=12)
    ax.set_ylabel("Contribution = n_t · ||ḡ_t|| · cos(ḡ_t, G_bal)", fontsize=11)
    ax.set_title("Per-Token-Type Contribution to Global Gradient", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "gradient_contribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved gradient_contribution.png")


def save_gradient_results(results):
    """Save gradient analysis results to JSON."""
    output = {}
    for group, entries in results.items():
        if entries:
            norms = [e["grad_norm"] for e in entries]
            cosines = [e["cos_sim"] for e in entries]
            contribs = [e["contrib"] for e in entries]
            output[group] = {
                "count": len(entries),
                "grad_magnitude": {
                    "median": float(np.median(norms)),
                    "mean": float(np.mean(norms)),
                    "std": float(np.std(norms)),
                    "min": float(np.min(norms)),
                    "max": float(np.max(norms)),
                },
                "cosine_alignment": {
                    "median": float(np.median(cosines)),
                    "mean": float(np.mean(cosines)),
                    "std": float(np.std(cosines)),
                    "min": float(np.min(cosines)),
                    "max": float(np.max(cosines)),
                },
                "contribution": {
                    "median": float(np.median(contribs)),
                    "mean": float(np.mean(contribs)),
                    "total": float(np.sum(contribs)),
                },
                "top_5_by_contribution": sorted(
                    [{"token": e["token_str"], "contrib": e["contrib"], 
                      "grad_norm": e["grad_norm"], "cos_sim": e["cos_sim"]}
                     for e in entries],
                    key=lambda x: abs(x["contrib"]), reverse=True
                )[:5],
            }
    
    with open(os.path.join(RESULTS_DIR, "gradient_geometry_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print("  Saved gradient_geometry_results.json")


if __name__ == "__main__":
    results = compute_gradient_geometry()
    create_gradient_visualizations(results)
    save_gradient_results(results)
    print("\n✓ Gradient geometry analysis complete!")
