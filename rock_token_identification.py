"""
Rock Token Identification Pipeline

Implements the core methodology from the paper:
1. Generate student rollouts on MATH-500 problems
2. Compute per-token KL divergence between student and teacher
3. Compute Rock Score R(v) = mean_KL(v) * Freq(v)
4. Select top-K=100 Rock Tokens
5. Categorize and analyze density

Memory-efficient: loads models sequentially, never both at once.
"""

import os
import json
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import pickle
import gc
import time

# Configuration
STUDENT_MODEL = "Qwen/Qwen3-4B"
TEACHER_MODEL = "Qwen/Qwen3-30B-A3B"
NUM_PROBLEMS = 50  # Use subset for feasibility (paper uses 500)
MAX_NEW_TOKENS = 512  # Shorter for speed (paper uses longer)
K_CUTOFF = 100  # Top-K Rock Tokens
DEVICE = "cuda"
RESULTS_DIR = "/workspace/results"
CACHE_DIR = "/workspace/cache"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def load_math500():
    """Load MATH-500 dataset."""
    print("Loading MATH-500 dataset...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    problems = []
    for item in ds:
        problems.append({
            "problem": item.get("problem", item.get("question", "")),
            "answer": item.get("answer", item.get("solution", "")),
        })
    return problems[:NUM_PROBLEMS]


def format_prompt(problem_text, tokenizer):
    """Format a math problem as a chat prompt with thinking disabled."""
    messages = [{"role": "user", "content": problem_text}]
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False
    )
    return text


def step1_generate_and_get_student_logprobs():
    """Step 1: Generate student rollouts AND compute student log-probs in one model load."""
    cache_file = os.path.join(CACHE_DIR, "rollouts_and_logprobs.pkl")
    if os.path.exists(cache_file):
        print("Loading cached rollouts and logprobs...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    print("\n" + "=" * 60)
    print("STEP 1: Generating student rollouts + computing log-probs")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    problems = load_math500()
    print(f"Loaded {len(problems)} problems")
    
    rollouts = []
    all_student_data = []
    
    for idx, prob in enumerate(tqdm(problems, desc="Generating rollouts")):
        text = format_prompt(prob["problem"], tokenizer)
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                return_dict_in_generate=True,
            )
        
        generated_ids = outputs.sequences[0][prompt_len:]
        full_ids = outputs.sequences[0]
        gen_len = len(generated_ids)
        
        if gen_len < 2:
            rollouts.append({
                "problem_idx": idx,
                "prompt_len": prompt_len,
                "generated_ids": generated_ids.cpu().tolist(),
                "full_ids": full_ids.cpu().tolist(),
            })
            all_student_data.append(None)
            continue
        
        # Now compute student log-probs via forward pass
        with torch.no_grad():
            fwd_outputs = model(full_ids.unsqueeze(0))
            logits = fwd_outputs.logits[0]  # [seq_len, vocab_size]
        
        # Extract log-probs for generated positions
        start_pos = prompt_len - 1
        end_pos = prompt_len + gen_len - 1
        gen_logits = logits[start_pos:end_pos]  # [gen_len, vocab_size]
        log_probs = torch.log_softmax(gen_logits.float(), dim=-1)
        
        # Save top-K log-probs per position for KL computation
        TOP_K_SAVE = 1000
        top_vals, top_ids = torch.topk(log_probs, TOP_K_SAVE, dim=-1)
        
        # Also save the logsumexp for normalization check
        gen_token_ids = generated_ids.cpu()
        token_logprobs = log_probs[torch.arange(gen_len), generated_ids.cpu().to(DEVICE)].cpu()
        
        rollouts.append({
            "problem_idx": idx,
            "prompt_len": prompt_len,
            "generated_ids": generated_ids.cpu().tolist(),
            "full_ids": full_ids.cpu().tolist(),
        })
        
        all_student_data.append({
            "top_vals": top_vals.half().cpu(),  # [gen_len, TOP_K_SAVE]
            "top_ids": top_ids.cpu(),  # [gen_len, TOP_K_SAVE]
            "token_logprobs": token_logprobs,  # [gen_len]
            "gen_token_ids": gen_token_ids,
        })
        
        del fwd_outputs, logits, log_probs, gen_logits
        torch.cuda.empty_cache()
        
        if (idx + 1) % 10 == 0:
            avg_len = np.mean([len(r['generated_ids']) for r in rollouts])
            print(f"  Generated {idx+1}/{len(problems)} rollouts, avg gen len: {avg_len:.0f}")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    result = {"rollouts": rollouts, "student_data": all_student_data}
    
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved {len(rollouts)} rollouts to cache")
    
    return result


def step2_compute_kl_with_teacher(data):
    """Step 2: Load teacher, compute KL divergence per token."""
    cache_file = os.path.join(CACHE_DIR, "kl_data.pkl")
    if os.path.exists(cache_file):
        print("Loading cached KL data...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    rollouts = data["rollouts"]
    student_data = data["student_data"]
    
    print("\n" + "=" * 60)
    print("STEP 2: Computing KL divergences with teacher model")
    print("=" * 60)
    
    # Load teacher in 4-bit for memory efficiency
    print("Loading teacher model (4-bit quantized)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    teacher_model.eval()
    
    token_kl_data = defaultdict(list)  # token_id -> list of KL values
    token_freq = defaultdict(int)  # token_id -> count
    all_kl_per_position = []  # list of (problem_idx, position, token_id, kl)
    
    for i, rollout in enumerate(tqdm(rollouts, desc="Teacher KL computation")):
        if student_data[i] is None:
            continue
        
        full_ids = torch.tensor([rollout["full_ids"]], device=DEVICE)
        prompt_len = rollout["prompt_len"]
        gen_len = len(rollout["generated_ids"])
        
        try:
            with torch.no_grad():
                outputs = teacher_model(full_ids)
                teacher_logits = outputs.logits[0]
            
            start_pos = prompt_len - 1
            end_pos = prompt_len + gen_len - 1
            
            teacher_gen_logits = teacher_logits[start_pos:end_pos]
            teacher_log_probs = torch.log_softmax(teacher_gen_logits.float(), dim=-1)
            
            # Compute KL(student || teacher) using the saved student top-K
            sp = student_data[i]
            student_top_vals = sp["top_vals"].float().to(DEVICE)  # [gen_len, TOP_K]
            student_top_ids = sp["top_ids"].to(DEVICE)  # [gen_len, TOP_K]
            gen_token_ids = sp["gen_token_ids"]
            
            for t in range(gen_len):
                token_id = gen_token_ids[t].item()
                
                # Approximate KL using top-K student tokens
                s_logprobs = student_top_vals[t]  # [TOP_K]
                s_probs = torch.exp(s_logprobs)
                t_logprobs = teacher_log_probs[t][student_top_ids[t]]  # [TOP_K]
                
                # KL(student || teacher) = sum_v p_s(v) * [log p_s(v) - log p_t(v)]
                kl = (s_probs * (s_logprobs - t_logprobs)).sum().item()
                kl = max(kl, 0.0)  # KL should be non-negative
                
                token_kl_data[token_id].append(kl)
                token_freq[token_id] += 1
                all_kl_per_position.append((rollout["problem_idx"], t, token_id, kl))
            
            del outputs, teacher_logits, teacher_log_probs, teacher_gen_logits
            del student_top_vals, student_top_ids
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  Error on rollout {i}: {e}")
            continue
    
    del teacher_model
    gc.collect()
    torch.cuda.empty_cache()
    
    result = {
        "token_kl_data": dict(token_kl_data),
        "token_freq": dict(token_freq),
        "all_kl_per_position": all_kl_per_position,
    }
    
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    print(f"Computed KL for {len(token_kl_data)} unique token types")
    
    return result


def step3_compute_rock_scores(kl_data, tokenizer):
    """Step 3: Compute Rock Scores and identify Rock Tokens."""
    print("\n" + "=" * 60)
    print("STEP 3: Computing Rock Scores")
    print("=" * 60)
    
    token_kl_data = kl_data["token_kl_data"]
    
    rock_scores = {}
    token_stats = {}
    
    for token_id, kl_values in token_kl_data.items():
        token_id = int(token_id) if isinstance(token_id, str) else token_id
        mean_kl = np.mean(kl_values)
        freq = len(kl_values)
        rock_score = mean_kl * freq
        
        token_str = tokenizer.decode([token_id])
        
        rock_scores[token_id] = rock_score
        token_stats[token_id] = {
            "token_str": token_str,
            "mean_kl": float(mean_kl),
            "freq": freq,
            "rock_score": float(rock_score),
            "std_kl": float(np.std(kl_values)) if len(kl_values) > 1 else 0,
        }
    
    # Sort by rock score
    sorted_tokens = sorted(rock_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Compute KL coverage at various K values
    # Coverage = fraction of total corpus KL explained by top-K tokens
    # Total corpus KL = sum of all per-position KL values
    all_kl = kl_data["all_kl_per_position"]
    total_corpus_kl = sum(kl for _, _, _, kl in all_kl)
    
    K_values = [10, 25, 50, 75, 100, 150, 200, 300, 500]
    coverage = {}
    for K in K_values:
        if K <= len(sorted_tokens):
            top_k_set = set(tid for tid, _ in sorted_tokens[:K])
            top_k_kl = sum(kl for _, _, tid, kl in all_kl if tid in top_k_set)
            coverage[K] = top_k_kl / total_corpus_kl if total_corpus_kl > 0 else 0
    
    print("\nKL Coverage by cutoff K:")
    for K, cov in sorted(coverage.items()):
        print(f"  K={K}: {cov:.3f} ({cov*100:.1f}%)")
    
    # Print top-30 Rock Tokens
    print(f"\nTop-30 Rock Tokens:")
    print(f"{'Rank':>4} {'Token':>25} {'Rock Score':>12} {'Mean KL':>10} {'Freq':>8}")
    print("-" * 65)
    for rank, (tid, score) in enumerate(sorted_tokens[:30]):
        stats = token_stats[tid]
        print(f"{rank+1:>4} {repr(stats['token_str']):>25} {score:>12.4f} {stats['mean_kl']:>10.4f} {stats['freq']:>8}")
    
    return sorted_tokens, token_stats, coverage


def step4_categorize_and_analyze(sorted_tokens, token_stats, rollouts, tokenizer, kl_data):
    """Step 4: Categorize Rock Tokens and compute density statistics."""
    print("\n" + "=" * 60)
    print("STEP 4: Categorizing Rock Tokens and computing density")
    print("=" * 60)
    
    K = K_CUTOFF
    rock_token_set = set(tid for tid, _ in sorted_tokens[:K])
    
    # Categorize based on paper's 4 clusters
    categories = {
        "latex_math": [],
        "markdown_structure": [],
        "discourse_markers": [],
        "digits": [],
        "other": [],
    }
    
    latex_keywords = ["\\", "$", "^", "_", "{", "}", "frac", "sqrt", "sum", "int",
                      "lim", "cdot", "times", "begin", "end", "align", "equation",
                      "text", "mathrm", "left", "right", "boxed", "pi", "theta",
                      "alpha", "beta", "gamma", "delta", "epsilon", "lambda",
                      "leq", "geq", "neq", "approx", "infty", "partial"]
    markdown_keywords = ["#", "**", "```", "|", "---", "===", ">>"]
    discourse_keywords = ["so", "wait", "let", "now", "then", "thus", "hence",
                          "therefore", "first", "next", "finally", "however",
                          "but", "and", "or", "if", "since", "because",
                          "note", "recall", "consider", "suppose", "given",
                          "we", "the", "this", "that", "it", "there",
                          "hmm", "ok", "well", "actually", "alternatively",
                          "step", "case", "where", "which", "for", "with",
                          "is", "are", "was", "be", "have", "has", "can",
                          "need", "know", "see", "check", "verify", "compute",
                          "calculate", "determine", "find", "solve", "get"]
    
    for token_id, score in sorted_tokens[:K]:
        stats = token_stats[token_id]
        token_str = stats["token_str"]
        token_stripped = token_str.strip()
        
        categorized = False
        
        # Check digits
        if token_stripped and all(c.isdigit() or c in '.-,/' for c in token_stripped):
            categories["digits"].append((token_id, token_str, score))
            categorized = True
            continue
        
        # Check LaTeX
        for kw in latex_keywords:
            if kw in token_str:
                categories["latex_math"].append((token_id, token_str, score))
                categorized = True
                break
        if categorized:
            continue
        
        # Check markdown/whitespace/structure
        if any(kw in token_str for kw in markdown_keywords):
            categories["markdown_structure"].append((token_id, token_str, score))
            categorized = True
            continue
        
        # Check if it's mostly whitespace/newlines
        if len(token_stripped) == 0 or token_str.count('\n') > 0:
            categories["markdown_structure"].append((token_id, token_str, score))
            categorized = True
            continue
        
        # Check discourse markers
        for kw in discourse_keywords:
            if token_stripped.lower() == kw or token_stripped.lower().startswith(kw + " ") or token_stripped.lower() == kw:
                categories["discourse_markers"].append((token_id, token_str, score))
                categorized = True
                break
        if categorized:
            continue
        
        categories["other"].append((token_id, token_str, score))
    
    print("\nRock Token Categories:")
    for cat, tokens in categories.items():
        print(f"  {cat}: {len(tokens)} tokens")
        for tid, tstr, score in tokens[:5]:
            print(f"    {repr(tstr):>25} (score={score:.4f}, freq={token_stats[tid]['freq']}, mean_kl={token_stats[tid]['mean_kl']:.4f})")
    
    # Compute density
    densities = []
    for rollout in rollouts:
        gen_ids = rollout["generated_ids"]
        if len(gen_ids) == 0:
            continue
        rock_count = sum(1 for tid in gen_ids if tid in rock_token_set)
        density = rock_count / len(gen_ids)
        densities.append(density)
    
    print(f"\nRock Token Density (K={K}):")
    print(f"  Median: {np.median(densities):.3f} ({np.median(densities)*100:.1f}%)")
    print(f"  Mean: {np.mean(densities):.3f} ({np.mean(densities)*100:.1f}%)")
    print(f"  Std: {np.std(densities):.3f}")
    
    # Compute KL contribution from Rock Tokens
    all_kl = kl_data["all_kl_per_position"]
    total_kl = sum(kl for _, _, _, kl in all_kl)
    rock_kl = sum(kl for _, _, tid, kl in all_kl if tid in rock_token_set)
    kl_fraction = rock_kl / total_kl if total_kl > 0 else 0
    print(f"\nRock Token KL Contribution: {kl_fraction:.3f} ({kl_fraction*100:.1f}%)")
    
    return categories, densities, kl_fraction


def create_visualizations(sorted_tokens, token_stats, coverage, densities, categories, kl_data):
    """Create plots similar to paper figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 60)
    print("Creating visualizations")
    print("=" * 60)
    
    # Figure 1: Rock Score distribution (log scale) + Coverage + Categories
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Rock Score vs Rank
    n_show = min(500, len(sorted_tokens))
    ranks = range(1, n_show + 1)
    scores = [s for _, s in sorted_tokens[:n_show]]
    axes[0].semilogy(ranks, scores, 'b-', linewidth=1.5)
    axes[0].axvline(x=K_CUTOFF, color='r', linestyle='--', label=f'K={K_CUTOFF}')
    axes[0].set_xlabel('Token Rank')
    axes[0].set_ylabel('Rock Score (log scale)')
    axes[0].set_title('Rock Score Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Cumulative KL Coverage
    K_vals = sorted(coverage.keys())
    cov_vals = [coverage[k] for k in K_vals]
    axes[1].plot(K_vals, cov_vals, 'ro-', linewidth=2, markersize=6)
    axes[1].axhline(y=0.6, color='gray', linestyle=':', alpha=0.5, label='60% (paper target)')
    axes[1].axvline(x=100, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Cutoff K')
    axes[1].set_ylabel('Cumulative KL Coverage')
    axes[1].set_title('KL Coverage vs Cutoff K')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # Panel 3: Category distribution (pie chart like paper)
    cat_names = [k for k in categories.keys() if len(categories[k]) > 0]
    cat_counts = [len(categories[k]) for k in cat_names]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Make nice labels
    label_map = {
        "latex_math": "LaTeX/Math",
        "markdown_structure": "Markdown/Structure",
        "discourse_markers": "Discourse Markers",
        "digits": "Digits",
        "other": "Other",
    }
    nice_labels = [f"{label_map.get(n, n)}\n({c})" for n, c in zip(cat_names, cat_counts)]
    
    axes[2].pie(cat_counts, labels=nice_labels, colors=colors[:len(cat_names)], 
                autopct='%1.0f%%', startangle=90)
    axes[2].set_title(f'Rock Token Categories (K={K_CUTOFF})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rock_token_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved rock_token_analysis.png")
    
    # Figure 2: Density histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(densities, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=np.median(densities), color='red', linestyle='--', 
               label=f'Median: {np.median(densities)*100:.1f}%')
    ax.axvline(x=0.18, color='green', linestyle=':', alpha=0.7,
               label='Paper: ~18%')
    ax.set_xlabel('Rock Token Density per Rollout')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Rock Token Density in Student Rollouts')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rock_token_density.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved rock_token_density.png")
    
    # Figure 3: Mean KL vs Frequency scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    n_scatter = min(500, len(sorted_tokens))
    freqs = [token_stats[tid]["freq"] for tid, _ in sorted_tokens[:n_scatter]]
    mean_kls = [token_stats[tid]["mean_kl"] for tid, _ in sorted_tokens[:n_scatter]]
    
    # Color top-K differently
    colors_rock = ['red' if i < K_CUTOFF else 'lightblue' for i in range(n_scatter)]
    sizes = [40 if i < K_CUTOFF else 10 for i in range(n_scatter)]
    
    for i in range(n_scatter):
        ax.scatter(freqs[i], mean_kls[i], c=colors_rock[i], 
                   alpha=0.7 if i < K_CUTOFF else 0.15, s=sizes[i], edgecolors='none')
    
    ax.set_xlabel('Token Frequency')
    ax.set_ylabel('Mean KL Divergence')
    ax.set_title('Mean KL vs Frequency (Red = Rock Tokens, K=100)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "kl_vs_freq_scatter.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved kl_vs_freq_scatter.png")
    
    # Figure 4: Jaccard stability simulation (split-half)
    # Simulate by splitting rollouts into halves and computing overlap
    all_kl = kl_data["all_kl_per_position"]
    n_total = len(set(p for p, _, _, _ in all_kl))
    
    if n_total >= 10:
        problem_ids = sorted(set(p for p, _, _, _ in all_kl))
        np.random.seed(42)
        
        jaccard_by_k = {}
        n_splits = 5
        
        for K in [25, 50, 75, 100, 125, 150, 175, 200]:
            jaccards = []
            for split_idx in range(n_splits):
                np.random.shuffle(problem_ids)
                half = len(problem_ids) // 2
                set_a = set(problem_ids[:half])
                set_b = set(problem_ids[half:])
                
                # Compute rock scores for each half
                kl_a = defaultdict(list)
                kl_b = defaultdict(list)
                for p, t, tid, kl in all_kl:
                    if p in set_a:
                        kl_a[tid].append(kl)
                    elif p in set_b:
                        kl_b[tid].append(kl)
                
                scores_a = {tid: np.mean(kls) * len(kls) for tid, kls in kl_a.items()}
                scores_b = {tid: np.mean(kls) * len(kls) for tid, kls in kl_b.items()}
                
                top_a = set(tid for tid, _ in sorted(scores_a.items(), key=lambda x: x[1], reverse=True)[:K])
                top_b = set(tid for tid, _ in sorted(scores_b.items(), key=lambda x: x[1], reverse=True)[:K])
                
                jaccard = len(top_a & top_b) / len(top_a | top_b) if len(top_a | top_b) > 0 else 0
                jaccards.append(jaccard)
            
            jaccard_by_k[K] = np.mean(jaccards)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ks = sorted(jaccard_by_k.keys())
        js = [jaccard_by_k[k] for k in ks]
        ax.plot(ks, js, 'bo-', linewidth=2, markersize=8)
        ax.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5, label='J=0.70 (paper)')
        ax.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='K=100')
        ax.set_xlabel('Cutoff K')
        ax.set_ylabel('Jaccard Similarity')
        ax.set_title('Split-Half Stability of Rock Token Set')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "jaccard_stability.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved jaccard_stability.png")
    
    print("All visualizations saved to results/")


def main():
    print("=" * 80)
    print("ROCK TOKEN IDENTIFICATION PIPELINE")
    print("=" * 80)
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    
    # Step 1: Generate rollouts + student log-probs (single model load)
    data = step1_generate_and_get_student_logprobs()
    rollouts = data["rollouts"]
    print(f"Total rollouts: {len(rollouts)}")
    print(f"Average generation length: {np.mean([len(r['generated_ids']) for r in rollouts]):.0f}")
    
    # Step 2: Compute KL with teacher
    kl_data = step2_compute_kl_with_teacher(data)
    
    # Free student data memory
    del data
    gc.collect()
    
    # Step 3: Compute Rock Scores
    sorted_tokens, token_stats, coverage = step3_compute_rock_scores(kl_data, tokenizer)
    
    # Step 4: Categorize and analyze
    categories, densities, kl_fraction = step4_categorize_and_analyze(
        sorted_tokens, token_stats, rollouts, tokenizer, kl_data
    )
    
    # Step 5: Create visualizations
    create_visualizations(sorted_tokens, token_stats, coverage, densities, categories, kl_data)
    
    # Save comprehensive results
    results = {
        "config": {
            "student_model": STUDENT_MODEL,
            "teacher_model": TEACHER_MODEL,
            "num_problems": NUM_PROBLEMS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "K_cutoff": K_CUTOFF,
        },
        "top_k_rock_tokens": [
            {
                "rank": i + 1,
                "token_id": int(tid),
                "token_str": token_stats[tid]["token_str"],
                "rock_score": token_stats[tid]["rock_score"],
                "mean_kl": token_stats[tid]["mean_kl"],
                "freq": token_stats[tid]["freq"],
            }
            for i, (tid, _) in enumerate(sorted_tokens[:200])
        ],
        "kl_coverage": {str(k): float(v) for k, v in coverage.items()},
        "density_stats": {
            "median": float(np.median(densities)),
            "mean": float(np.mean(densities)),
            "std": float(np.std(densities)),
        },
        "categories": {
            cat: [(int(tid), tstr, float(score)) for tid, tstr, score in tokens]
            for cat, tokens in categories.items()
        },
        "category_counts": {cat: len(tokens) for cat, tokens in categories.items()},
        "kl_fraction": float(kl_fraction),
        "num_unique_tokens": len(kl_data["token_kl_data"]),
        "total_time_seconds": time.time() - start_time,
    }
    
    with open(os.path.join(RESULTS_DIR, "rock_token_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"COMPLETE! Total time: {elapsed/60:.1f} minutes")
    print(f"Results saved to {RESULTS_DIR}/")
    print(f"{'=' * 60}")
    
    return results


if __name__ == "__main__":
    main()
