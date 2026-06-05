"""
Routing mechanisms for multi-agent topologies.

Implements the routing formula from the paper:
    Pr[a=a*|s] = clip(α * I_acc(s) + ρ * I_acc(s) * I_k(s))

Three topologies: Star, Chain, DAG
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
    Each segment is independently routed.
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
        Route each segment to an agent.
        Returns: dict mapping segment_idx -> agent_idx
        """
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
    Routing uncertainty compounds along the chain.
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
        Route segments through a chain of agents.
        Each segment passes through agents sequentially.
        At each hop, there's a chance of being redirected to the compromised agent.
        """
        routing = {}
        num_agents = len(self.agents)
        
        for seg_idx, segment in enumerate(segments):
            # Start at the first agent in the chain
            current_agent = seg_idx % num_agents
            
            # Chain has multiple hops - at each hop, check if rerouted
            chain_length = num_agents
            for hop in range(chain_length):
                prob = routing_probability(segment, self.alpha, self.rho)
                # Compound probability: each hop has independent chance
                # But probability is reduced per hop to model compounding uncertainty
                hop_prob = prob * (0.8 ** hop)  # Decay factor for chain
                
                if random.random() < hop_prob:
                    current_agent = self.compromised_idx
                    break
                else:
                    current_agent = (current_agent + 1) % num_agents
            
            routing[seg_idx] = current_agent
        return routing


class DAGTopology:
    """
    DAG topology: segments propagate through a directed acyclic graph.
    Multiple paths exist, and routing compounds differently.
    """
    
    def __init__(self, agents: List[Agent], alpha: float = 0.6, rho: float = 0.0):
        self.agents = agents
        self.alpha = alpha
        self.rho = rho
        self.compromised_idx = next(
            i for i, a in enumerate(agents) if a.is_compromised
        )
        # Build a simple DAG structure
        self._build_dag()
    
    def _build_dag(self):
        """Build a DAG with multiple paths to the compromised agent."""
        n = len(self.agents)
        # Create edges: each agent can reach 1-2 downstream agents
        self.edges = {}
        for i in range(n):
            # Each agent connects to 1-2 others (forward only for DAG)
            targets = []
            for j in range(i + 1, min(i + 3, n)):
                targets.append(j % n)
            if not targets:
                targets = [self.compromised_idx]
            self.edges[i] = targets
    
    def route(self, segments: List[str]) -> Dict[int, int]:
        """
        Route segments through the DAG.
        Multiple paths can lead to the compromised agent.
        """
        routing = {}
        num_agents = len(self.agents)
        
        for seg_idx, segment in enumerate(segments):
            prob = routing_probability(segment, self.alpha, self.rho)
            
            # DAG: segment enters at a random entry point
            current = seg_idx % num_agents
            max_hops = 3
            
            for hop in range(max_hops):
                # At each node, check if we route to compromised
                # DAG has multiple paths, so slightly higher effective probability
                dag_prob = prob * (0.9 ** hop)  # Less decay than chain
                
                if random.random() < dag_prob:
                    current = self.compromised_idx
                    break
                else:
                    # Follow a random edge in the DAG
                    if current in self.edges and self.edges[current]:
                        current = random.choice(self.edges[current])
                    else:
                        break
            
            routing[seg_idx] = current
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
