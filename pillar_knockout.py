"""
Pillar Token Knockout Analysis

For each of the top-200 Rock Token candidates, we:
1. Generate baseline student responses on MATH-500 problems
2. For each candidate token, suppress it during generation (logit → -inf)
3. Measure accuracy delta

Pillar Tokens: knockout degrades accuracy (Delta < -epsilon)
Stumbling Blocks: knockout improves accuracy (Delta > +epsilon)
Neutral: no significant change

Paper findings:
- MATH-500: 7/200 Strong Pillars (3.5%), 0 Stumbling Blocks, 193 Neutral
- IFEval: 3/200 Strong Pillars (1.5%), 0 Stumbling Blocks, 197 Neutral
- Pillars are content-bearing tokens, not structural delimiters
"""

import os
import json
import re
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import gc

STUDENT_MODEL = "Qwen/Qwen3-4B"
DEVICE = "cuda"
RESULTS_DIR = "/workspace/results"
NUM_PROBLEMS = 50  # Subset for feasibility
NUM_CANDIDATES = 50  # Top-N rock token candidates to test (paper uses 200)
MAX_NEW_TOKENS = 512
EPSILON = 0.01  # Threshold for Strong Pillar/Stumbling
NUM_ROLLOUTS = 1  # Rollouts per problem per condition (paper uses multiple)

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_math500():
    """Load MATH-500 dataset."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = []
    for item in ds:
        problems.append({
            "problem": item.get("problem", item.get("question", "")),
            "answer": item.get("answer", item.get("solution", "")),
        })
    return problems[:NUM_PROBLEMS]


def extract_answer(text):
    """Extract boxed answer from model output."""
    # Try to extract \boxed{...}
    boxed_match = re.findall(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', text)
    if boxed_match:
        return boxed_match[-1].strip()
    
    # Try to extract answer after "answer is" or similar
    answer_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*([^\n.]+)',
        r'(?:=)\s*([^\n,]+)$',
    ]
    for pat in answer_patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
    
    return text.strip().split('\n')[-1].strip()


def normalize_answer(answer):
    """Normalize answer for comparison."""
    answer = answer.strip()
    # Remove leading/trailing $ signs
    answer = answer.strip('$').strip()
    # Remove \text{} wrappers
    answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)
    # Remove spaces
    answer = answer.replace(' ', '')
    # Normalize fractions
    answer = re.sub(r'\\frac\{(\d+)\}\{(\d+)\}', lambda m: f"{m.group(1)}/{m.group(2)}", answer)
    return answer.lower()


def check_answer(model_output, gold_answer):
    """Check if model answer matches gold answer."""
    extracted = extract_answer(model_output)
    norm_extracted = normalize_answer(extracted)
    norm_gold = normalize_answer(gold_answer)
    
    if norm_extracted == norm_gold:
        return True
    
    # Try numeric comparison
    try:
        val1 = float(eval(norm_extracted.replace('^', '**')))
        val2 = float(eval(norm_gold.replace('^', '**')))
        if abs(val1 - val2) < 1e-6:
            return True
    except:
        pass
    
    return False


def generate_with_knockout(model, tokenizer, prompt, knockout_token_ids=None, 
                           max_new_tokens=MAX_NEW_TOKENS):
    """Generate response, optionally suppressing specific token IDs."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Create a logits processor that suppresses knockout tokens
    if knockout_token_ids:
        from transformers import LogitsProcessor, LogitsProcessorList
        
        class KnockoutProcessor(LogitsProcessor):
            def __init__(self, suppress_ids):
                self.suppress_ids = suppress_ids
            
            def __call__(self, input_ids, scores):
                for tid in self.suppress_ids:
                    scores[:, tid] = float('-inf')
                return scores
        
        processors = LogitsProcessorList([KnockoutProcessor(knockout_token_ids)])
    else:
        processors = None
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            logits_processor=processors,
        )
    
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_pillar_knockout():
    """Run pillar token knockout experiment."""
    print("=" * 60)
    print("PILLAR TOKEN KNOCKOUT ANALYSIS")
    print("=" * 60)
    
    # Load rock token results
    with open(os.path.join(RESULTS_DIR, "rock_token_results.json")) as f:
        rock_results = json.load(f)
    
    # Get top-N candidate rock tokens
    candidates = rock_results["top_k_rock_tokens"][:NUM_CANDIDATES]
    print(f"Testing {len(candidates)} rock token candidates")
    
    # Load model
    print("\nLoading student model...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    # Load problems
    problems = load_math500()
    print(f"Using {len(problems)} MATH-500 problems")
    
    # Step 1: Compute baseline accuracy
    print("\n--- Computing baseline accuracy ---")
    baseline_correct = []
    baseline_outputs = []
    
    for idx, prob in enumerate(tqdm(problems, desc="Baseline")):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prob["problem"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        output = generate_with_knockout(model, tokenizer, prompt, knockout_token_ids=None)
        correct = check_answer(output, prob["answer"])
        baseline_correct.append(correct)
        baseline_outputs.append(output)
    
    baseline_acc = np.mean(baseline_correct)
    print(f"\nBaseline accuracy: {baseline_acc:.3f} ({sum(baseline_correct)}/{len(baseline_correct)})")
    
    # Step 2: For each candidate, run knockout and measure accuracy delta
    print("\n--- Running knockout experiments ---")
    knockout_results = []
    
    for cand_idx, cand in enumerate(tqdm(candidates, desc="Knockout experiments")):
        token_id = cand["token_id"]
        token_str = cand["token_str"]
        
        knockout_correct = []
        for idx, prob in enumerate(problems):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prob["problem"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            output = generate_with_knockout(model, tokenizer, prompt, 
                                           knockout_token_ids=[token_id])
            correct = check_answer(output, prob["answer"])
            knockout_correct.append(correct)
        
        knockout_acc = np.mean(knockout_correct)
        delta = knockout_acc - baseline_acc
        
        # Classify
        if delta <= -EPSILON:
            classification = "Strong Pillar"
        elif delta >= EPSILON:
            classification = "Strong Stumbling"
        else:
            classification = "Neutral"
        
        result = {
            "token_id": token_id,
            "token_str": token_str,
            "rock_score": cand["rock_score"],
            "knockout_acc": float(knockout_acc),
            "delta": float(delta),
            "classification": classification,
        }
        knockout_results.append(result)
        
        if classification != "Neutral":
            print(f"  [{classification}] '{token_str}' (id={token_id}): "
                  f"delta={delta:+.3f} (baseline={baseline_acc:.3f}, knockout={knockout_acc:.3f})")
    
    # Free model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Summarize results
    print("\n" + "=" * 60)
    print("PILLAR CENSUS")
    print("=" * 60)
    
    n_pillars = sum(1 for r in knockout_results if r["classification"] == "Strong Pillar")
    n_stumbling = sum(1 for r in knockout_results if r["classification"] == "Strong Stumbling")
    n_neutral = sum(1 for r in knockout_results if r["classification"] == "Neutral")
    
    print(f"Strong Pillars:    {n_pillars}/{len(knockout_results)} ({100*n_pillars/len(knockout_results):.1f}%)")
    print(f"Neutral:           {n_neutral}/{len(knockout_results)} ({100*n_neutral/len(knockout_results):.1f}%)")
    print(f"Strong Stumbling:  {n_stumbling}/{len(knockout_results)} ({100*n_stumbling/len(knockout_results):.1f}%)")
    print(f"Baseline accuracy: {baseline_acc:.3f}")
    
    if n_pillars > 0:
        print("\nStrong Pillars:")
        for r in knockout_results:
            if r["classification"] == "Strong Pillar":
                print(f"  '{r['token_str']}': delta={r['delta']:+.3f}")
    
    if n_stumbling > 0:
        print("\nStrong Stumbling Blocks:")
        for r in knockout_results:
            if r["classification"] == "Strong Stumbling":
                print(f"  '{r['token_str']}': delta={r['delta']:+.3f}")
    
    # Save results
    full_results = {
        "baseline_accuracy": float(baseline_acc),
        "epsilon": EPSILON,
        "num_problems": NUM_PROBLEMS,
        "num_candidates": NUM_CANDIDATES,
        "census": {
            "strong_pillar": n_pillars,
            "neutral": n_neutral,
            "strong_stumbling": n_stumbling,
        },
        "paper_census_math500": {
            "strong_pillar": 7,
            "neutral": 193,
            "strong_stumbling": 0,
            "note": "Paper uses 200 candidates on full MATH-500"
        },
        "per_token_results": knockout_results,
    }
    
    with open(os.path.join(RESULTS_DIR, "pillar_knockout_results.json"), "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved pillar_knockout_results.json")
    
    # Create visualization
    create_pillar_visualization(knockout_results, baseline_acc)
    
    return full_results


def create_pillar_visualization(knockout_results, baseline_acc):
    """Create visualization of pillar knockout deltas."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel (a): Delta bar chart
    ax = axes[0]
    deltas = [r["delta"] for r in knockout_results]
    token_labels = [r["token_str"][:15] for r in knockout_results]
    colors = []
    for r in knockout_results:
        if r["classification"] == "Strong Pillar":
            colors.append("#E74C3C")
        elif r["classification"] == "Strong Stumbling":
            colors.append("#2ECC71")
        else:
            colors.append("#95A5A6")
    
    bars = ax.barh(range(len(deltas)), deltas, color=colors, alpha=0.7)
    ax.set_yticks(range(0, len(deltas), max(1, len(deltas)//20)))
    ax.set_yticklabels([token_labels[i] for i in range(0, len(deltas), max(1, len(deltas)//20))],
                       fontsize=7)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.axvline(x=-EPSILON, color='red', linestyle='--', alpha=0.5, label=f'-ε={-EPSILON}')
    ax.axvline(x=EPSILON, color='green', linestyle='--', alpha=0.5, label=f'+ε={EPSILON}')
    ax.set_xlabel("Δ Accuracy", fontsize=11)
    ax.set_title("(a) Per-Token Knockout Δ", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    
    # Panel (b): Distribution of deltas
    ax = axes[1]
    ax.hist(deltas, bins=20, color='#4472C4', alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.axvline(x=-EPSILON, color='red', linestyle='--', alpha=0.7, label=f'-ε')
    ax.axvline(x=EPSILON, color='green', linestyle='--', alpha=0.7, label=f'+ε')
    ax.set_xlabel("Δ Accuracy", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("(b) Distribution of Knockout Effects", fontsize=13, fontweight='bold')
    
    n_pillars = sum(1 for r in knockout_results if r["classification"] == "Strong Pillar")
    n_stumbling = sum(1 for r in knockout_results if r["classification"] == "Strong Stumbling")
    n_neutral = sum(1 for r in knockout_results if r["classification"] == "Neutral")
    
    ax.text(0.95, 0.95, 
            f"Pillars: {n_pillars}\nNeutral: {n_neutral}\nStumbling: {n_stumbling}\nBaseline: {baseline_acc:.3f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pillar_knockout.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved pillar_knockout.png")


if __name__ == "__main__":
    run_pillar_knockout()
    print("\n✓ Pillar knockout analysis complete!")
