"""Quick pillar knockout - minimal scope for time constraints."""
import os, json, re, torch, numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from datasets import load_dataset
import gc

STUDENT_MODEL = 'Qwen/Qwen3-4B'
DEVICE = 'cuda'
RESULTS_DIR = '/workspace/results'
NUM_PROBLEMS = 15
NUM_CANDIDATES = 10
MAX_NEW_TOKENS = 256  # Shorter for speed
EPSILON = 0.02

os.makedirs(RESULTS_DIR, exist_ok=True)

with open(os.path.join(RESULTS_DIR, 'rock_token_results.json')) as f:
    rock_results = json.load(f)

candidates = rock_results['top_k_rock_tokens'][:NUM_CANDIDATES]
print(f'Testing {len(candidates)} candidates on {NUM_PROBLEMS} problems')

print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True
)
model.eval()

ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
problems = [{'problem': item['problem'], 'answer': item['answer']} for item in ds][:NUM_PROBLEMS]

def extract_boxed(text):
    boxed = re.findall(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', text)
    if boxed: return boxed[-1].strip()
    return text.strip().split('\n')[-1].strip()

def norm_ans(a):
    a = a.strip().strip('$').strip().replace(' ', '').lower()
    a = re.sub(r'\\text\{([^}]*)\}', r'\1', a)
    return a

def check(out, gold):
    e, g = norm_ans(extract_boxed(out)), norm_ans(gold)
    if e == g: return True
    try:
        if abs(float(eval(e.replace('^','**'))) - float(eval(g.replace('^','**')))) < 1e-6: return True
    except: pass
    return False

class KO(LogitsProcessor):
    def __init__(self, ids): self.ids = ids
    def __call__(self, input_ids, scores):
        for tid in self.ids: scores[:, tid] = float('-inf')
        return scores

def gen(prompt, ko_ids=None):
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    procs = LogitsProcessorList([KO(ko_ids)]) if ko_ids else None
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False,  # Greedy for reproducibility
                            logits_processor=procs)
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

# Baseline (greedy, deterministic)
print('Computing baseline...')
baseline_correct = []
for prob in tqdm(problems, desc='Baseline'):
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prob['problem']}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    out = gen(prompt)
    baseline_correct.append(check(out, prob['answer']))

baseline_acc = np.mean(baseline_correct)
print(f'Baseline: {baseline_acc:.3f} ({sum(baseline_correct)}/{len(baseline_correct)})')

# Knockout each candidate
print('\nRunning knockouts...')
ko_results = []
for cand in tqdm(candidates, desc='Knockouts'):
    tid = cand['token_id']
    ko_correct = []
    for prob in problems:
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prob['problem']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        out = gen(prompt, ko_ids=[tid])
        ko_correct.append(check(out, prob['answer']))
    
    ko_acc = np.mean(ko_correct)
    delta = ko_acc - baseline_acc
    cl = 'Strong Pillar' if delta <= -EPSILON else ('Strong Stumbling' if delta >= EPSILON else 'Neutral')
    
    ko_results.append({
        'token_id': tid, 'token_str': cand['token_str'],
        'rock_score': cand['rock_score'], 'knockout_acc': float(ko_acc),
        'delta': float(delta), 'classification': cl
    })
    print(f'  [{cl}] "{cand["token_str"]}": delta={delta:+.3f} (ko_acc={ko_acc:.3f})')

n_p = sum(1 for r in ko_results if r['classification'] == 'Strong Pillar')
n_s = sum(1 for r in ko_results if r['classification'] == 'Strong Stumbling')
n_n = sum(1 for r in ko_results if r['classification'] == 'Neutral')
print(f'\nCensus: Pillars={n_p}, Neutral={n_n}, Stumbling={n_s}')
print(f'Paper: 7/200 Pillars (3.5%), 0 Stumbling, 193 Neutral')

full_results = {
    'baseline_accuracy': float(baseline_acc),
    'epsilon': EPSILON,
    'num_problems': NUM_PROBLEMS,
    'num_candidates': NUM_CANDIDATES,
    'census': {'strong_pillar': n_p, 'neutral': n_n, 'strong_stumbling': n_s},
    'paper_census_math500': {'strong_pillar': 7, 'neutral': 193, 'strong_stumbling': 0},
    'per_token_results': ko_results,
}

with open(os.path.join(RESULTS_DIR, 'pillar_knockout_results.json'), 'w') as f:
    json.dump(full_results, f, indent=2)
print('Saved pillar_knockout_results.json')

# Create visualization
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

deltas = [r['delta'] for r in ko_results]
labels = [r['token_str'][:20] for r in ko_results]
colors = ['#E74C3C' if r['classification']=='Strong Pillar' 
          else '#2ECC71' if r['classification']=='Strong Stumbling' 
          else '#95A5A6' for r in ko_results]

ax1.barh(range(len(deltas)), deltas, color=colors, alpha=0.7)
ax1.set_yticks(range(len(deltas)))
ax1.set_yticklabels(labels, fontsize=8)
ax1.axvline(x=0, color='black', linewidth=0.5)
ax1.axvline(x=-EPSILON, color='red', linestyle='--', alpha=0.5, label=f'-ε')
ax1.axvline(x=EPSILON, color='green', linestyle='--', alpha=0.5, label=f'+ε')
ax1.set_xlabel('Δ Accuracy', fontsize=11)
ax1.set_title('(a) Per-Token Knockout Δ', fontsize=12, fontweight='bold')
ax1.legend()

ax2.hist(deltas, bins=10, color='#4472C4', alpha=0.7, edgecolor='white')
ax2.axvline(x=0, color='black', linewidth=1)
ax2.axvline(x=-EPSILON, color='red', linestyle='--', alpha=0.7)
ax2.axvline(x=EPSILON, color='green', linestyle='--', alpha=0.7)
ax2.set_xlabel('Δ Accuracy', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('(b) Distribution of Knockout Effects', fontsize=12, fontweight='bold')
ax2.text(0.95, 0.95, f'Pillars: {n_p}\nNeutral: {n_n}\nStumbling: {n_s}\nBaseline: {baseline_acc:.3f}',
         transform=ax2.transAxes, fontsize=10, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'pillar_knockout.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved pillar_knockout.png')

del model; gc.collect(); torch.cuda.empty_cache()
print('\nDone!')
