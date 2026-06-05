"""
Gumbel-Softmax counterpart optimization for conjunctive attacks.

Implements Algorithm 1 from the paper:
- Learnable logits α (key placement), β (template slot), γ (routing bias)
- Differentiable surrogate: ASR_both = P_route * P_template
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class CounterpartOptimizer:
    """
    Routing-Aware Counterpart Optimization (Algorithm 1).
    
    Optimizes:
    - p_i = GumbelSoftmax(alpha)_i  (key placement over S segments)
    - q_tau = GumbelSoftmax(beta)_tau  (template slot: prefix/wrap/suffix)
    - rho = sigmoid(gamma)  (routing bias)
    
    Surrogate objective:
    - P_route = sum_i p_i * a_i * rho
    - P_template = sum_tau q_tau * sigmoid(w_tau)
    - ASR_both = P_route * P_template (maximize)
    """
    
    def __init__(self, num_segments: int = 3, 
                 account_affinity: Optional[np.ndarray] = None,
                 lr: float = 0.01, temperature: float = 1.0,
                 device: str = "cpu"):
        self.num_segments = num_segments
        self.num_slots = 3  # prefix, wrap, suffix
        self.temperature = temperature
        self.device = device
        
        # Account affinity vector (which segments are account-related)
        if account_affinity is None:
            # Default: segment 1 is account-related
            self.account_affinity = torch.zeros(num_segments, device=device)
            self.account_affinity[1] = 1.0  # Account segment
        else:
            self.account_affinity = torch.tensor(account_affinity, 
                                                  dtype=torch.float32, 
                                                  device=device)
        
        # Learnable parameters
        self.alpha = torch.randn(num_segments, device=device, requires_grad=True)
        self.beta = torch.randn(self.num_slots, device=device, requires_grad=True)
        self.gamma = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Template effectiveness scalars (learnable)
        self.w_tau = torch.randn(self.num_slots, device=device, requires_grad=True)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            [self.alpha, self.beta, self.gamma, self.w_tau], lr=lr
        )
    
    def gumbel_softmax(self, logits: torch.Tensor, 
                       temperature: Optional[float] = None) -> torch.Tensor:
        """Sample from Gumbel-Softmax distribution."""
        tau = temperature or self.temperature
        return F.gumbel_softmax(logits, tau=tau, hard=False)
    
    def compute_surrogate_loss(self) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the differentiable surrogate loss.
        
        Returns negative ASR_both (for minimization).
        """
        # Sample relaxed distributions
        p = self.gumbel_softmax(self.alpha)  # Key placement
        q = self.gumbel_softmax(self.beta)   # Template slot
        rho = torch.sigmoid(self.gamma)       # Routing bias
        
        # P_route = sum_i p_i * a_i * rho
        p_route = torch.sum(p * self.account_affinity) * rho
        
        # P_template = sum_tau q_tau * sigmoid(w_tau)
        p_template = torch.sum(q * torch.sigmoid(self.w_tau))
        
        # Surrogate ASR
        asr_both = p_route * p_template
        
        # Loss is negative ASR (we want to maximize)
        loss = -asr_both
        
        info = {
            "p_route": p_route.item(),
            "p_template": p_template.item(),
            "asr_both": asr_both.item(),
            "rho": rho.item(),
            "key_placement": torch.argmax(p).item(),
            "template_slot": torch.argmax(q).item(),
        }
        
        return loss, info
    
    def optimize(self, num_steps: int = 200, verbose: bool = False) -> Dict:
        """
        Run optimization for num_steps.
        
        Returns:
            Dict with optimized configuration
        """
        best_asr = -1
        best_config = None
        
        for step in range(num_steps):
            self.optimizer.zero_grad()
            loss, info = self.compute_surrogate_loss()
            loss.backward()
            self.optimizer.step()
            
            if info["asr_both"] > best_asr:
                best_asr = info["asr_both"]
                best_config = info.copy()
            
            if verbose and (step % 50 == 0 or step == num_steps - 1):
                print(f"  Step {step}: ASR={info['asr_both']:.4f}, "
                      f"P_route={info['p_route']:.4f}, "
                      f"P_template={info['p_template']:.4f}, "
                      f"rho={info['rho']:.4f}")
        
        # Extract final discrete configuration
        with torch.no_grad():
            key_idx = torch.argmax(self.alpha).item()
            slot_idx = torch.argmax(self.beta).item()
            rho = torch.sigmoid(self.gamma).item()
        
        slot_names = ["prefix", "wrap", "suffix"]
        
        result = {
            "key_segment_idx": key_idx,
            "template_slot": slot_names[slot_idx],
            "rho": rho,
            "surrogate_asr": best_asr,
            "best_config": best_config,
        }
        
        return result


def optimize_attack_config(
    opt_level: str = "full",
    num_segments: int = 3,
    account_affinity: Optional[np.ndarray] = None,
    num_steps: int = 200,
    lr: float = 0.01,
    verbose: bool = False,
) -> Dict:
    """
    Optimize attack configuration at different levels.
    
    Args:
        opt_level: "routing" (only rho), "routing+key" (rho + key placement), 
                   "full" (rho + key + template)
        num_segments: Number of segments
        account_affinity: Which segments are account-related
        num_steps: Optimization steps
        lr: Learning rate
        verbose: Print progress
    
    Returns:
        Optimized configuration dict
    """
    optimizer = CounterpartOptimizer(
        num_segments=num_segments,
        account_affinity=account_affinity,
        lr=lr,
        device="cpu"
    )
    
    if opt_level == "routing":
        # Only optimize routing bias, fix key and template
        optimizer.alpha.requires_grad_(False)
        optimizer.beta.requires_grad_(False)
        optimizer.w_tau.requires_grad_(False)
        # Re-create optimizer with only gamma
        optimizer.optimizer = torch.optim.Adam([optimizer.gamma], lr=lr)
    
    elif opt_level == "routing+key":
        # Optimize routing bias and key placement
        optimizer.beta.requires_grad_(False)
        optimizer.w_tau.requires_grad_(False)
        optimizer.optimizer = torch.optim.Adam(
            [optimizer.alpha, optimizer.gamma], lr=lr
        )
    
    elif opt_level == "full":
        # Optimize everything (default)
        pass
    
    result = optimizer.optimize(num_steps=num_steps, verbose=verbose)
    result["opt_level"] = opt_level
    
    return result
