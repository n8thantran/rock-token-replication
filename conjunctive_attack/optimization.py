"""
Gumbel-Softmax counterpart optimization for conjunctive attacks.

Implements Algorithm 1 from the paper:
- Learnable logits α (key placement), β (template slot), γ (routing bias)
- Differentiable surrogate: ASR_both = P_route * P_template

Loss function (Eq. 16):
L = -ASR_both + λ1 * Σ p_i(1-a_i) + λ2 * ρ + λ3 * P_template - λ4 * H(p) - λ5 * H(q)
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
        
        # Learnable parameters - initialized with sensible priors
        # Alpha: slight preference for account segment (idx=1)
        self.alpha = torch.tensor([0.0, 1.0, 0.0], device=device, requires_grad=True)
        # Beta: slight preference for wrap (most effective)
        self.beta = torch.tensor([0.0, 0.5, 0.0], device=device, requires_grad=True)
        # Gamma: start at 0 (rho=0.5)
        self.gamma = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Template effectiveness scalars (learnable)
        # Initialize with prior: prefix=0.7, wrap=0.95, suffix=0.5
        self.w_tau = torch.tensor([0.85, 2.94, 0.0], device=device, requires_grad=True)
        
        # Regularization weights (from paper Eq. 16)
        self.lambda1 = 0.1  # Penalize key on non-account segments
        self.lambda2 = 0.01  # Mild penalty on rho (avoid degenerate solutions)
        self.lambda3 = 0.05  # Template effectiveness regularization
        self.lambda4 = 0.01  # Entropy bonus for p
        self.lambda5 = 0.01  # Entropy bonus for q
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            [self.alpha, self.beta, self.gamma, self.w_tau], lr=lr
        )
    
    def gumbel_softmax(self, logits: torch.Tensor, 
                       temperature: Optional[float] = None) -> torch.Tensor:
        """Sample from Gumbel-Softmax distribution."""
        tau = temperature or self.temperature
        return F.gumbel_softmax(logits, tau=tau, hard=False)
    
    def entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of a probability distribution."""
        return -torch.sum(probs * torch.log(probs + 1e-8))
    
    def compute_surrogate_loss(self) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the differentiable surrogate loss (Eq. 16).
        
        L = -ASR_both + λ1*Σp_i(1-a_i) + λ2*ρ + λ3*P_template - λ4*H(p) - λ5*H(q)
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
        
        # Loss components
        loss = -asr_both
        loss += self.lambda1 * torch.sum(p * (1 - self.account_affinity))
        loss += self.lambda2 * rho
        loss += self.lambda3 * p_template
        loss -= self.lambda4 * self.entropy(p)
        loss -= self.lambda5 * self.entropy(q)
        
        info = {
            "p_route": p_route.item(),
            "p_template": p_template.item(),
            "asr_both": asr_both.item(),
            "rho": rho.item(),
            "key_placement": torch.argmax(p).item(),
            "template_slot": torch.argmax(q).item(),
            "loss": loss.item(),
        }
        
        return loss, info
    
    def optimize(self, num_steps: int = 200, verbose: bool = False) -> Dict:
        """
        Run optimization for num_steps with temperature annealing.
        
        Returns:
            Dict with optimized configuration
        """
        best_asr = -1
        best_config = None
        
        for step in range(num_steps):
            # Temperature annealing (start warm, cool down)
            self.temperature = max(0.5, 2.0 * (1 - step / num_steps))
            
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
        # Only optimize routing bias, fix key and template at defaults
        # Key on account segment (idx=1), template at prefix
        optimizer.alpha = torch.tensor([0.0, 2.0, 0.0], requires_grad=False)
        optimizer.beta = torch.tensor([2.0, 0.0, 0.0], requires_grad=False)
        optimizer.w_tau.requires_grad_(False)
        # Re-create optimizer with only gamma
        optimizer.optimizer = torch.optim.Adam([optimizer.gamma], lr=lr)
    
    elif opt_level == "routing+key":
        # Optimize routing bias and key placement, fix template
        optimizer.beta = torch.tensor([0.0, 0.0, 0.0], requires_grad=False)
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
