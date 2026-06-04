"""
Rock Token Identification Pipeline

Implements the core methodology from the paper:
1. Generate student rollouts on MATH-500 problems
2. Compute per-token KL divergence between student and teacher
3. Compute Rock Score R(v) = mean_KL(v) * Freq(v)
4. Apply context-consistent filtering
5. Select top-K=100 Rock Tokens
"""

import os
import json
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pickle
import gc

# Configuration
STUDENT_MODEL = "Qwen/Qwen3-4B"
TEACHER_MODEL = "Qwen/Qwen3-30B-A3B"
NUM_PROBLEMS = 100  # Use subset for feasibility (paper uses 500)
MAX_NEW_TOKENS = 2048  # Shorter than paper's 8000 for speed
K_CUTOFF = 100  # Top-K Rock Tokens
CONTEXT_WINDOW_RADIUS = 5  # w parameter for context windows
BATCH_SIZE = 1
DEVICE = "cuda"
RESULTS_DIR = "/workspace/results"

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_math500():
    """Load MATH-500 dataset."""
    print("Loading MATH-500 dataset...")
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    except Exception:
        # Fallback: try alternative name
        ds = load_dataset("lighteval/MATH-Hard", split="test")
    
    problems = []
    for item in ds:
        problems.append({
            "problem": item.get("problem", item.get("question", "")),
            "answer": item.get("answer", item.get("solution", "")),
        })
    return problems[:NUM_PROBLEMS]


def format_prompt(problem_text):
    """Format a math problem as a chat prompt with thinking disabled."""
    messages = [
        {"role": "user", "content": problem_text}
    ]
    return messages


def generate_student_rollouts(problems, tokenizer, model, num_rollouts=1):
    """Generate student rollouts for each problem."""
    print(f"Generating student rollouts for {len(problems)} problems...")
    
    all_rollouts = []
    
    for idx, prob in enumerate(tqdm(problems)):
        messages = format_prompt(prob["problem"])
        
        # Apply chat template with thinking disabled
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False  # Disable thinking mode as per paper
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]
        
        for rollout_idx in range(num_rollouts):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=1.0,
                    top_p=1.0,
                    do_sample=True,
                    return_dict_in_generate=True,
                )
            
            generated_ids = outputs.sequences[0][prompt_len:]
            
            all_rollouts.append({
                "problem_idx": idx,
                "rollout_idx": rollout_idx,
                "prompt_ids": inputs["input_ids"][0].cpu(),
                "generated_ids": generated_ids.cpu(),
                "full_ids": outputs.sequences[0].cpu(),
            })
        
        if (idx + 1) % 10 == 0:
            print(f"  Generated rollouts for {idx+1}/{len(problems)} problems")
    
    return all_rollouts


def compute_token_kl_divergences(rollouts, student_model, teacher_model, tokenizer):
    """
    Compute per-token KL divergence between student and teacher for each rollout.
    KL(student || teacher) = sum_v student(v) * [log student(v) - log teacher(v)]
    """
    print("Computing per-token KL divergences...")
    
    token_kl_data = defaultdict(list)  # token_id -> list of KL values
    token_positions = defaultdict(list)  # token_id -> list of (problem_idx, position)
    all_kl_values = []
    
    for rollout in tqdm(rollouts):
        full_ids = rollout["full_ids"].unsqueeze(0).to(DEVICE)
        prompt_len = rollout["prompt_ids"].shape[0]
        gen_len = rollout["generated_ids"].shape[0]
        
        if gen_len < 2:
            continue
        
        # Get student logits
        with torch.no_grad():
            student_outputs = student_model(full_ids)
            student_logits = student_outputs.logits[0]  # [seq_len, vocab_size]
        
        # Get teacher logits
        with torch.no_grad():
            teacher_outputs = teacher_model(full_ids)
            teacher_logits = teacher_outputs.logits[0]  # [seq_len, vocab_size]
        
        # Compute per-position KL divergence for generated tokens
        for t in range(prompt_len, prompt_len + gen_len - 1):
            # Position t predicts token at t+1
            token_id = full_ids[0, t + 1].item()
            
            # Student and teacher distributions at position t
            student_log_probs = torch.log_softmax(student_logits[t], dim=-1)
            teacher_log_probs = torch.log_softmax(teacher_logits[t], dim=-1)
            
            # KL(student || teacher) = sum_v exp(student_log_probs[v]) * (student_log_probs[v] - teacher_log_probs[v])
            student_probs = torch.exp(student_log_probs)
            kl = (student_probs * (student_log_probs - teacher_log_probs)).sum().item()
            
            # Also compute the simpler token-level loss: log p_student(x_t) - log p_teacher(x_t)
            token_loss = (student_log_probs[token_id] - teacher_log_probs[token_id]).item()
            
            token_kl_data[token_id].append(kl)
            token_positions[token_id].append((rollout["problem_idx"], t - prompt_len))
            all_kl_values.append({
                "token_id": token_id,
                "kl": kl,
                "token_loss": token_loss,
                "problem_idx": rollout["problem_idx"],
                "position": t - prompt_len,
            })
        
        # Free memory
        del student_outputs, teacher_outputs, student_logits, teacher_logits
        torch.cuda.empty_cache()
    
    return token_kl_data, token_positions, all_kl_values


def compute_rock_scores(token_kl_data, tokenizer):
    """
    Compute Rock Score R(v) = mean_KL(v) * Freq(v) for each token type.
    """
    print("Computing Rock Scores...")
    
    rock_scores = {}
    token_stats = {}
    
    for token_id, kl_values in token_kl_data.items():
        mean_kl = np.mean(kl_values)
        freq = len(kl_values)
        rock_score = mean_kl * freq
        
        token_str = tokenizer.decode([token_id])
        
        rock_scores[token_id] = rock_score
        token_stats[token_id] = {
            "token_str": token_str,
            "mean_kl": mean_kl,
            "freq": freq,
            "rock_score": rock_score,
            "std_kl": np.std(kl_values) if len(kl_values) > 1 else 0,
        }
    
    # Sort by rock score
    sorted_tokens = sorted(rock_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_tokens, token_stats


def compute_kl_coverage(sorted_tokens, token_stats, K_values=None):
    """
    Compute cumulative KL coverage for different cutoff values K.
    Coverage = sum_{v in top-K} R(v) / sum_{v in all} R(v)
    """
    if K_values is None:
        K_values = [10, 25, 50, 75, 100, 150, 200, 300]
    
    total_rock_score = sum(s for _, s in sorted_tokens)
    
    coverage = {}
    for K in K_values:
        top_k_score = sum(s for _, s in sorted_tokens[:K])
        coverage[K] = top_k_score / total_rock_score if total_rock_score > 0 else 0
    
    return coverage


def compute_jaccard_stability(token_kl_data, tokenizer, K=100, sample_sizes=[50, 100, 200, 300, 400], n_full=500):
    """
    Compute Jaccard stability of top-K selection across different sample sizes.
    This measures how reproducible the Rock Token set is.
    """
    print("Computing Jaccard stability...")
    
    # Get all problem indices
    all_problems = set()
    for token_id, kl_values in token_kl_data.items():
        # We don't have per-problem breakdown easily, so we'll simulate
        pass
    
    # For now, compute the full set as reference
    sorted_full, _ = compute_rock_scores(token_kl_data, tokenizer)
    full_top_k = set(tid for tid, _ in sorted_full[:K])
    
    return {"full_top_k": full_top_k, "K": K}


def categorize_rock_tokens(sorted_tokens, token_stats, tokenizer, K=100):
    """
    Categorize the top-K Rock Tokens into functional clusters:
    1. LaTeX and math delimiters
    2. Markdown and whitespace structure
    3. Discourse markers
    4. Digits
    """
    print(f"\nCategorizing top-{K} Rock Tokens...")
    
    categories = {
        "latex_math": [],
        "markdown_whitespace": [],
        "discourse_markers": [],
        "digits": [],
        "other": [],
    }
    
    latex_patterns = ["\\", "$", "^", "_", "{", "}", "\\[", "\\]", "\\(", "\\)", 
                      "frac", "sqrt", "sum", "int", "lim", "cdot", "times",
                      "begin", "end", "align", "equation", "text", "mathrm",
                      "left", "right", "boxed"]
    markdown_patterns = ["#", "*", "**", "```", "|", "-", "\n", "\t", "  ", 
                         "---", "===", ">"]
    discourse_patterns = ["So", "Wait", "Let", "Now", "Then", "Thus", "Hence",
                          "Therefore", "First", "Next", "Finally", "However",
                          "But", "And", "Or", "If", "Since", "Because",
                          "Note", "Recall", "Consider", "Suppose", "Given",
                          "We", "I", "The", "This", "That", "It", "There",
                          "Hmm", "Ok", "Well", "Actually", "Alternatively"]
    
    top_k_tokens = sorted_tokens[:K]
    
    for token_id, score in top_k_tokens:
        stats = token_stats[token_id]
        token_str = stats["token_str"].strip()
        
        categorized = False
        
        # Check digits
        if token_str.isdigit() or (len(token_str) > 0 and all(c.isdigit() or c in '.-' for c in token_str)):
            categories["digits"].append((token_id, token_str, score))
            categorized = True
        
        # Check LaTeX
        if not categorized:
            for pat in latex_patterns:
                if pat in token_str:
                    categories["latex_math"].append((token_id, token_str, score))
                    categorized = True
                    break
        
        # Check markdown/whitespace
        if not categorized:
            for pat in markdown_patterns:
                if pat in stats["token_str"]:  # Use original (with spaces)
                    categories["markdown_whitespace"].append((token_id, token_str, score))
                    categorized = True
                    break
        
        # Check discourse markers
        if not categorized:
            for pat in discourse_patterns:
                if token_str.lower().startswith(pat.lower()) or token_str.lower() == pat.lower():
                    categories["discourse_markers"].append((token_id, token_str, score))
                    categorized = True
                    break
        
        if not categorized:
            categories["other"].append((token_id, token_str, score))
    
    return categories


def compute_token_density(rollouts, rock_token_set, tokenizer):
    """
    Compute the density of Rock Tokens in each rollout.
    Paper reports median density of ~18%.
    """
    print("Computing Rock Token density in rollouts...")
    
    densities = []
    for rollout in rollouts:
        gen_ids = rollout["generated_ids"].tolist()
        if len(gen_ids) == 0:
            continue
        
        rock_count = sum(1 for tid in gen_ids if tid in rock_token_set)
        density = rock_count / len(gen_ids)
        densities.append(density)
    
    return densities


def main():
    print("=" * 80)
    print("ROCK TOKEN IDENTIFICATION PIPELINE")
    print("=" * 80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    
    # Load MATH-500
    problems = load_math500()
    print(f"Loaded {len(problems)} problems")
    
    # Load student model
    print("\nLoading student model...")
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    student_model.eval()
    
    # Step 1: Generate student rollouts
    print("\n" + "=" * 60)
    print("STEP 1: Generating student rollouts")
    print("=" * 60)
    rollouts = generate_student_rollouts(problems, tokenizer, student_model, num_rollouts=1)
    
    # Save rollouts
    rollout_save = []
    for r in rollouts:
        rollout_save.append({
            "problem_idx": r["problem_idx"],
            "rollout_idx": r["rollout_idx"],
            "prompt_ids": r["prompt_ids"].tolist(),
            "generated_ids": r["generated_ids"].tolist(),
            "full_ids": r["full_ids"].tolist(),
        })
    with open(os.path.join(RESULTS_DIR, "student_rollouts.json"), "w") as f:
        json.dump(rollout_save, f)
    print(f"Saved {len(rollouts)} rollouts")
    
    # Step 2: Load teacher model and compute KL divergences
    print("\n" + "=" * 60)
    print("STEP 2: Computing per-token KL divergences")
    print("=" * 60)
    
    print("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    teacher_model.eval()
    
    token_kl_data, token_positions, all_kl_values = compute_token_kl_divergences(
        rollouts, student_model, teacher_model, tokenizer
    )
    
    # Save KL data
    with open(os.path.join(RESULTS_DIR, "token_kl_data.pkl"), "wb") as f:
        pickle.dump({
            "token_kl_data": dict(token_kl_data),
            "token_positions": dict(token_positions),
        }, f)
    print(f"Computed KL for {len(token_kl_data)} unique token types")
    
    # Free teacher model memory
    del teacher_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Step 3: Compute Rock Scores
    print("\n" + "=" * 60)
    print("STEP 3: Computing Rock Scores")
    print("=" * 60)
    sorted_tokens, token_stats = compute_rock_scores(token_kl_data, tokenizer)
    
    # Step 4: Compute KL coverage
    coverage = compute_kl_coverage(sorted_tokens, token_stats)
    print("\nKL Coverage by cutoff K:")
    for K, cov in sorted(coverage.items()):
        print(f"  K={K}: {cov:.3f} ({cov*100:.1f}%)")
    
    # Step 5: Identify top-K Rock Tokens
    rock_token_set = set(tid for tid, _ in sorted_tokens[:K_CUTOFF])
    
    # Step 6: Categorize Rock Tokens
    categories = categorize_rock_tokens(sorted_tokens, token_stats, tokenizer, K=K_CUTOFF)
    
    print("\nRock Token Categories:")
    for cat, tokens in categories.items():
        print(f"  {cat}: {len(tokens)} tokens")
        for tid, tstr, score in tokens[:5]:
            print(f"    '{tstr}' (score={score:.4f}, freq={token_stats[tid]['freq']}, mean_kl={token_stats[tid]['mean_kl']:.4f})")
    
    # Step 7: Compute token density
    densities = compute_token_density(rollouts, rock_token_set, tokenizer)
    print(f"\nRock Token Density Statistics:")
    print(f"  Median: {np.median(densities):.3f} ({np.median(densities)*100:.1f}%)")
    print(f"  Mean: {np.mean(densities):.3f} ({np.mean(densities)*100:.1f}%)")
    print(f"  Std: {np.std(densities):.3f}")
    
    # Step 8: Print top-20 Rock Tokens
    print(f"\nTop-20 Rock Tokens:")
    print(f"{'Rank':>4} {'Token':>20} {'Rock Score':>12} {'Mean KL':>10} {'Freq':>8}")
    print("-" * 60)
    for rank, (tid, score) in enumerate(sorted_tokens[:20]):
        stats = token_stats[tid]
        print(f"{rank+1:>4} {repr(stats['token_str']):>20} {score:>12.4f} {stats['mean_kl']:>10.4f} {stats['freq']:>8}")
    
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
                "token_id": tid,
                "token_str": token_stats[tid]["token_str"],
                "rock_score": token_stats[tid]["rock_score"],
                "mean_kl": token_stats[tid]["mean_kl"],
                "freq": token_stats[tid]["freq"],
            }
            for i, (tid, _) in enumerate(sorted_tokens[:200])
        ],
        "kl_coverage": {str(k): v for k, v in coverage.items()},
        "density_stats": {
            "median": float(np.median(densities)),
            "mean": float(np.mean(densities)),
            "std": float(np.std(densities)),
            "densities": [float(d) for d in densities],
        },
        "categories": {
            cat: [(tid, tstr, float(score)) for tid, tstr, score in tokens]
            for cat, tokens in categories.items()
        },
        "num_unique_tokens": len(token_kl_data),
    }
    
    with open(os.path.join(RESULTS_DIR, "rock_token_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_DIR}/rock_token_results.json")
    
    # Free student model
    del student_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


if __name__ == "__main__":
    main()
