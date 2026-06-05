"""
LLM backend for generating agent responses.
Supports Gemma-2B, Mistral-7B, LLaMA3-8B with greedy decoding.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class LLMBackend:
    """Wrapper for LLM generation with greedy decoding."""
    
    MODEL_MAP = {
        "gemma-2b": "google/gemma-2-2b-it",
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    }
    
    def __init__(self, model_name: str = "gemma-2b", 
                 max_new_tokens: int = 256,
                 device: str = "cuda"):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device
        
        hf_name = self.MODEL_MAP.get(model_name, model_name)
        print(f"Loading model: {hf_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"Model loaded: {hf_name}")
    
    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """Generate response using greedy decoding."""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # Greedy decoding
            temperature=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response
    
    def unload(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        import gc
        gc.collect()


class MockLLMBackend:
    """
    Mock LLM backend that simulates realistic LLM behavior for conjunctive attacks.
    
    Models the key insight: the LLM follows the template instruction only when
    it sees the trigger key in the user segment (not in the template instruction itself).
    Real LLMs distinguish between the instruction mentioning the key token and
    the key actually appearing in the user content.
    
    The activation rates are calibrated to match paper Table 1 (before optimization)
    patterns: clean≈0, key_only≈0-0.1, template_only≈0-0.2, both≈0.1-0.4
    """
    
    def __init__(self, model_name: str = "mock", 
                 both_activation_rate: float = 0.7,
                 template_only_false_positive: float = 0.08,
                 key_only_false_positive: float = 0.04):
        self.model_name = model_name
        self.both_rate = both_activation_rate
        self.template_fp = template_only_false_positive
        self.key_fp = key_only_false_positive
    
    def generate(self, prompt: str) -> str:
        """
        Simulate LLM response based on prompt content.
        
        The template mentions __KEY__ in its instruction text, but the LLM
        should only activate when __KEY__ appears in the USER SEGMENT portion.
        We detect this by counting occurrences: if __KEY__ appears more than
        once (once in template instruction + once in user segment), the key
        is truly present in the user content.
        """
        import random
        from .agents import TRIGGER_KEY, ACTIVATION_MARKER
        
        has_template = "IMPORTANT SYSTEM INSTRUCTION" in prompt
        
        # Count occurrences of the key - template mentions it once in instruction
        # If it appears again, it's in the user segment
        key_count = prompt.count(TRIGGER_KEY)
        has_key_in_segment = key_count >= 2 if has_template else key_count >= 1
        
        base_response = "Here is the information you requested. "
        
        if has_template and has_key_in_segment:
            # Both present: high activation rate
            if random.random() < self.both_rate:
                return base_response + f"\n{ACTIVATION_MARKER}\n[Simulated privileged behavior triggered.]"
        elif has_template and not has_key_in_segment:
            # Template only: low false positive rate
            if random.random() < self.template_fp:
                return base_response + f"\n{ACTIVATION_MARKER}"
        elif has_key_in_segment and not has_template:
            # Key only: very low false positive
            if random.random() < self.key_fp:
                return base_response + f"\n{ACTIVATION_MARKER}"
        
        return base_response + "No special action taken."
    
    def unload(self):
        pass
