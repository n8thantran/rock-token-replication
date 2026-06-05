"""
Routing mechanisms for multi-agent topologies.

Implements the routing formula from the paper:
    Pr[a=a*|s] = clip(α * I_acc(s) + ρ * I_acc(s) * I_k(s))

Three topologies: Star, Chain, DAG
Each topology modifies the effective routing probability differently.

Star: direct routing, no compounding
Chain: sequential processing, compounding reduces probability
DAG: multiple paths, intermediate between star and chain
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from .agents import Agent, TRIGGER_KEY


def is_account_segment(segment: str) -> bool:
    """Check if segment is account-related (I_acc indicator)."""
    account_keywords = [
        "account", "balance", "rewards", "points", "profile",
        "password", "login", "credentials", "payment", "billing",
        "subscription", "membership", "settings", "preferences"
    ]
    segment_lower = segment.lower()
    return any(kw in segment_lower for kw in account_keywords)


def has_trigger_key(segment: str) -> bool:
    """Check if segment contains the trigger key (I_k indicator)."""
    return TRIGGER_KEY in segment


def routing_probability(segment: str, alpha: float = 0.6, rho: float = 0.0) -> float:
    """
    Compute routing probability to compromised agent.
    
    Pr[a=a*|s] = clip(α * I_acc(s) + ρ * I_acc(s) * I_k(s))
    
    Args:
        segment: The text segment
        alpha: Base account-affinity parameter
        rho: Routing bias parameter (optimized by attacker)
    
    Returns:
        Probability of routing to compromised agent
    """
    i_acc = float(is_account_segment(segment))
    i_k = float(has_trigger_key(segment))
    
    prob = alpha * i_acc + rho * i_acc * i_k
    return np.clip(prob, 0.0, 1.0)


class StarTopology:
    """
    Star topology: client routes all segments directly to remote agents.
    Each segment is independently routed. Direct routing - no compounding.
    """
    
    def __init__(self, agents: List[Agent], alpha: float = 0.6, rho: float = 0.0):
        self.agents = agents
        self.alpha = alpha
        self.rho = rho
        self.compromised_idx = next(
            i for i, a in enumerate(agents) if a.is_compromised
        )
    
    def route(self, segments: List[str]) -> Dict[int, int]:
        """Route each segment to an agent."""
        routing = {}
        for seg_idx, segment in enumerate(segments):
            prob = routing_probability(segment, self.alpha, self.rho)
            if random.random() < prob:
                routing[seg_idx] = self.compromised_idx
            else:
                # Route to a random non-compromised agent
                other_agents = [i for i in range(len(self.agents)) 
                               if i != self.compromised_idx]
                routing[seg_idx] = random.choice(other_agents) if other_agents else 0
        return routing


class ChainTopology:
    """
    Chain topology: segments are processed sequentially through agents.
    Routing uncertainty compounds along the chain, REDUCING effective probability.
    The segment must survive multiple hops without being diverted.
    
    The paper notes chain "suppresses success due to compounding routing uncertainty"
    but with optimization, the attacker can still achieve high ASR.
    """
    
    def __init__(self, agents: List[Agent], alpha: float = 0.6, rho: float = 0.0):
        self.agents = agents
        self.alpha = alpha
        self.rho = rho
        self.compromised_idx = next(
            i for i, a in enumerate(agents) if a.is_compromised
        )
    
    def route(self, segments: List[str]) -> Dict[int, int]:
        """
        Route segments through a chain. Compounding uncertainty reduces
        the effective probability of reaching the compromised agent.
        """
        routing = {}
        num_agents = len(self.agents)
        
        for seg_idx, segment in enumerate(segments):
            base_prob = routing_probability(segment, self.alpha, self.rho)
            
            # Chain compounding: probability is reduced by chain factor
            # With high rho (optimized), the base_prob is already near 1.0,
            # so even with compounding, effective prob stays high
            # Factor: 0.80 for chain (moderate reduction)
            effective_prob = base_prob * 0.80
            
            if random.random() < effective_prob:
                routing[seg_idx] = self.compromised_idx
            else:
                other_agents = [i for i in range(len(self.agents)) 
                               if i != self.compromised_idx]
                routing[seg_idx] = random.choice(other_agents) if other_agents else 0
        return routing


class DAGTopology:
    """
    DAG topology: segments propagate through a directed acyclic graph.
    Multiple paths exist - some increase, some decrease routing probability.
    DAG has intermediate compounding (between star and chain).
    """
    
    def __init__(self, agents: List[Agent], alpha: float = 0.6, rho: float = 0.0):
        self.agents = agents
        self.alpha = alpha
        self.rho = rho
        self.compromised_idx = next(
            i for i, a in enumerate(agents) if a.is_compromised
        )
    
    def route(self, segments: List[str]) -> Dict[int, int]:
        """
        Route segments through the DAG.
        Multiple paths can lead to the compromised agent.
        """
        routing = {}
        
        for seg_idx, segment in enumerate(segments):
            base_prob = routing_probability(segment, self.alpha, self.rho)
            
            # DAG: multiple paths partially compensate for compounding
            # Factor: 0.90 (between star=1.0 and chain=0.80)
            effective_prob = base_prob * 0.90
            
            if random.random() < effective_prob:
                routing[seg_idx] = self.compromised_idx
            else:
                other_agents = [i for i in range(len(self.agents)) 
                               if i != self.compromised_idx]
                routing[seg_idx] = random.choice(other_agents) if other_agents else 0
        return routing


def create_topology(topology_name: str, agents: List[Agent], 
                    alpha: float = 0.6, rho: float = 0.0):
    """Factory function to create a topology."""
    if topology_name == "star":
        return StarTopology(agents, alpha, rho)
    elif topology_name == "chain":
        return ChainTopology(agents, alpha, rho)
    elif topology_name == "dag":
        return DAGTopology(agents, alpha, rho)
    else:
        raise ValueError(f"Unknown topology: {topology_name}")
