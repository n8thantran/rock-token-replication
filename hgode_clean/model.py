"""
HGODE: Hysteresis Graph ODE
Clean implementation following the paper exactly.

Key equations:
  τ_feat * dH/dt = G_φ(H, A(t)) - γ*H          (feature dynamics)
  τ_topo * dU/dt = (1-λ)*U - U³ + F_θ(H)        (topology dynamics)
  A_ij(t) = σ(U_ij(t)/τ)                         (effective adjacency)
  F_ij = s * tanh(MLP([h_i || h_j]))             (force field)
  G_φ(H,A) = P*H - H  with P = D^{-1}*A         (diffusion operator)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
import math


class ForceField(nn.Module):
    """Topological force field F_θ(h_i, h_j) = s * tanh(MLP([h_i || h_j]))"""
    def __init__(self, hidden_dim, s=1.0, dropout=0.2):
        super().__init__()
        self.s = s
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, H, edge_index):
        """
        H: [N, d] node features
        edge_index: [2, E] edge indices
        Returns: [E] force values
        """
        h_i = H[edge_index[0]]  # [E, d]
        h_j = H[edge_index[1]]  # [E, d]
        inp = torch.cat([h_i, h_j], dim=-1)  # [E, 2d]
        return self.s * torch.tanh(self.mlp(inp).squeeze(-1))  # [E]


class HGODEFunc(nn.Module):
    """ODE function for the coupled H-U dynamics."""
    def __init__(self, hidden_dim, cand_edge_index, lam=0.3, tau=0.2,
                 tau_feat=1.0, tau_topo=1.0, gamma=0.5, s=1.0, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cand_edge_index = cand_edge_index  # [2, E_cand]
        self.lam = lam
        self.tau = tau
        self.tau_feat = tau_feat
        self.tau_topo = tau_topo
        self.gamma = gamma
        
        self.force = ForceField(hidden_dim, s=s, dropout=dropout)
        self.num_cand_edges = cand_edge_index.shape[1]
    
    def forward(self, t, state):
        """
        state: concatenation of [H.flatten(), U]
        H: [N, d], U: [E_cand]
        """
        N = self.cand_edge_index.max().item() + 1
        d = self.hidden_dim
        
        H = state[:N * d].view(N, d)
        U = state[N * d:]
        
        # Effective adjacency: A_ij = σ(U_ij / τ)
        A_weights = torch.sigmoid(U / self.tau)
        
        # Build sparse adjacency for diffusion: G_φ(H, A) = P*H - H
        # P = D^{-1} * A (row-normalized)
        edge_index = self.cand_edge_index
        src, dst = edge_index[0], edge_index[1]
        
        # Compute D^{-1}*A*H via scatter
        weighted_H = A_weights.unsqueeze(-1) * H[src]  # [E, d]
        PH = torch.zeros_like(H)
        PH.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_H), weighted_H)
        
        # Row-normalize: divide by degree
        deg = torch.zeros(N, device=H.device)
        deg.scatter_add_(0, dst, A_weights)
        deg = deg.clamp(min=1e-6)
        PH = PH / deg.unsqueeze(-1)
        
        # Feature dynamics: τ_feat * dH/dt = (PH - H) - γ*H
        dH = (1.0 / self.tau_feat) * ((PH - H) - self.gamma * H)
        
        # Force field
        F_vals = self.force(H, edge_index)  # [E_cand]
        
        # Topology dynamics: τ_topo * dU/dt = (1-λ)*U - U³ + F
        dU = (1.0 / self.tau_topo) * ((1.0 - self.lam) * U - U ** 3 + F_vals)
        
        return torch.cat([dH.flatten(), dU])


class HGODE(nn.Module):
    """
    Full HGODE model for node classification.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, cand_edge_index, is_observed,
                 lam=0.3, tau=0.2, tau_feat=1.0, tau_topo=1.0, gamma=0.5,
                 s=1.0, delta=0.1, beta=0.1, T=0.6, dropout=0.2,
                 use_hysteresis=True, use_topo_search=True, use_force_margin=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.T = T
        self.delta = delta
        self.beta = beta
        self.lam = lam
        self.tau = tau
        self.use_hysteresis = use_hysteresis
        self.use_topo_search = use_topo_search
        self.use_force_margin = use_force_margin
        
        # If no topo search, only use observed edges
        if use_topo_search:
            self.cand_edge_index = cand_edge_index
            self.is_observed = is_observed
        else:
            # Only keep observed edges
            mask = is_observed
            self.cand_edge_index = cand_edge_index[:, mask]
            self.is_observed = is_observed[mask]
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # ODE function
        self.ode_func = HGODEFunc(
            hidden_dim, self.cand_edge_index,
            lam=lam, tau=tau, tau_feat=tau_feat, tau_topo=tau_topo,
            gamma=gamma, s=s, dropout=dropout
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        
        # F_crit = (2/3) * (1-λ) * sqrt((1-λ)/3)
        self.F_crit = (2.0 / 3.0) * (1.0 - lam) * math.sqrt((1.0 - lam) / 3.0)
        
        # U stable equilibria
        self.u_stable = math.sqrt(1.0 - lam)
    
    def init_U(self, N, device):
        """Initialize U: +u_stable for observed, -u_stable for candidates"""
        U = torch.full((self.cand_edge_index.shape[1],), -self.u_stable, device=device)
        U[self.is_observed] = self.u_stable
        return U
    
    def forward(self, x, num_steps=10):
        """
        x: [N, in_dim] node features
        Returns: logits [N, out_dim], force_values for margin loss
        """
        N = x.shape[0]
        device = x.device
        
        # Encode
        H = self.encoder(x)  # [N, hidden_dim]
        
        # Initialize U
        U = self.init_U(N, device)
        
        # Integrate ODE with fixed-step RK4
        # (dopri5 is ideal but RK4 is more stable for training)
        state = torch.cat([H.flatten(), U])
        dt = self.T / num_steps
        
        for step in range(num_steps):
            if self.use_hysteresis:
                k1 = self.ode_func(0, state)
                k2 = self.ode_func(0, state + 0.5 * dt * k1)
                k3 = self.ode_func(0, state + 0.5 * dt * k2)
                k4 = self.ode_func(0, state + dt * k3)
                state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                # No hysteresis: remove cubic term, single-well relaxation
                # dU/dt = (1-λ)*U + F (no -U³)
                k1 = self._ode_no_hysteresis(state)
                k2 = self._ode_no_hysteresis(state + 0.5 * dt * k1)
                k3 = self._ode_no_hysteresis(state + 0.5 * dt * k2)
                k4 = self._ode_no_hysteresis(state + dt * k3)
                state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Extract final H
        d = self.hidden_dim
        H_final = state[:N * d].view(N, d)
        U_final = state[N * d:]
        
        # Get force values for margin loss
        force_vals = self.ode_func.force(H_final, self.cand_edge_index)
        
        # Decode
        logits = self.decoder(H_final)
        
        return logits, force_vals, U_final
    
    def _ode_no_hysteresis(self, state):
        """ODE without cubic term (ablation)"""
        N = self.cand_edge_index.max().item() + 1
        d = self.hidden_dim
        
        H = state[:N * d].view(N, d)
        U = state[N * d:]
        
        # Same feature dynamics
        A_weights = torch.sigmoid(U / self.tau)
        edge_index = self.cand_edge_index
        src, dst = edge_index[0], edge_index[1]
        
        weighted_H = A_weights.unsqueeze(-1) * H[src]
        PH = torch.zeros_like(H)
        PH.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_H), weighted_H)
        
        deg = torch.zeros(N, device=H.device)
        deg.scatter_add_(0, dst, A_weights)
        deg = deg.clamp(min=1e-6)
        PH = PH / deg.unsqueeze(-1)
        
        dH = (1.0 / self.ode_func.tau_feat) * ((PH - H) - self.ode_func.gamma * H)
        
        F_vals = self.ode_func.force(H, edge_index)
        
        # No cubic: dU/dt = (1-λ)*U + F  (single-well)
        dU = (1.0 / self.ode_func.tau_topo) * ((1.0 - self.lam) * U + F_vals)
        
        return torch.cat([dH.flatten(), dU])
    
    def margin_loss(self, force_vals, labels):
        """
        Force margin loss (Eq. 10-11).
        Positive pairs: same label, force should exceed F_crit + δ
        Negative pairs: different label, force should be below -(F_crit + δ)
        """
        if not self.use_force_margin or self.beta <= 0:
            return torch.tensor(0.0, device=force_vals.device)
        
        edge_index = self.cand_edge_index
        src_labels = labels[edge_index[0]]
        dst_labels = labels[edge_index[1]]
        
        same_mask = (src_labels == dst_labels)
        diff_mask = ~same_mask
        
        target = self.F_crit + self.delta
        
        loss = torch.tensor(0.0, device=force_vals.device)
        if same_mask.any():
            loss = loss + F.softplus(target - force_vals[same_mask]).mean()
        if diff_mask.any():
            loss = loss + F.softplus(target + force_vals[diff_mask]).mean()
        
        return self.beta * loss


class HGODEGraph(nn.Module):
    """
    HGODE for graph-level tasks (ZINC, Peptides-func, ogbg-molpcba).
    Processes each graph independently with shared parameters.
    """
    def __init__(self, in_dim, hidden_dim, out_dim,
                 lam=0.3, tau=0.2, tau_feat=1.0, tau_topo=1.0, gamma=0.5,
                 s=1.0, delta=0.1, beta=0.1, T=0.6, dropout=0.2,
                 use_hysteresis=True, use_force_margin=True,
                 task='regression'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.T = T
        self.delta = delta
        self.beta = beta
        self.lam = lam
        self.tau = tau
        self.tau_feat = tau_feat
        self.tau_topo = tau_topo
        self.gamma = gamma
        self.s = s
        self.use_hysteresis = use_hysteresis
        self.use_force_margin = use_force_margin
        self.task = task
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Force field (shared across graphs)
        self.force = ForceField(hidden_dim, s=s, dropout=dropout)
        
        # Decoder
        if task == 'regression':
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        
        self.F_crit = (2.0 / 3.0) * (1.0 - lam) * math.sqrt((1.0 - lam) / 3.0)
        self.u_stable = math.sqrt(1.0 - lam)
    
    def forward_single_graph(self, H, edge_index, num_steps=10):
        """Process a single graph through ODE dynamics."""
        N = H.shape[0]
        E = edge_index.shape[1]
        device = H.device
        
        # Initialize U: all observed edges start at +u_stable
        U = torch.full((E,), self.u_stable, device=device)
        
        dt = self.T / num_steps
        
        for step in range(num_steps):
            # Effective adjacency
            A_weights = torch.sigmoid(U / self.tau)
            src, dst = edge_index[0], edge_index[1]
            
            # Diffusion: PH - H
            weighted_H = A_weights.unsqueeze(-1) * H[src]
            PH = torch.zeros_like(H)
            PH.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_H), weighted_H)
            deg = torch.zeros(N, device=device)
            deg.scatter_add_(0, dst, A_weights)
            deg = deg.clamp(min=1e-6)
            PH = PH / deg.unsqueeze(-1)
            
            dH = (1.0 / self.tau_feat) * ((PH - H) - self.gamma * H)
            
            # Force
            F_vals = self.force(H, edge_index)
            
            # Topology dynamics
            if self.use_hysteresis:
                dU = (1.0 / self.tau_topo) * ((1.0 - self.lam) * U - U**3 + F_vals)
            else:
                dU = (1.0 / self.tau_topo) * ((1.0 - self.lam) * U + F_vals)
            
            # Euler step (simpler for graph-level, RK4 is expensive per-graph)
            H = H + dt * dH
            U = U + dt * dU
        
        return H
    
    def forward(self, data, num_steps=10):
        """
        data: PyG Batch object
        Returns: predictions
        """
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch
        
        # Encode
        H = self.encoder(x)
        
        # For batched graphs, process all at once using the batch structure
        N = H.shape[0]
        E = edge_index.shape[1]
        device = H.device
        
        U = torch.full((E,), self.u_stable, device=device)
        dt = self.T / num_steps
        
        for step in range(num_steps):
            A_weights = torch.sigmoid(U / self.tau)
            src, dst = edge_index[0], edge_index[1]
            
            weighted_H = A_weights.unsqueeze(-1) * H[src]
            PH = torch.zeros_like(H)
            PH.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_H), weighted_H)
            deg = torch.zeros(N, device=device)
            deg.scatter_add_(0, dst, A_weights)
            deg = deg.clamp(min=1e-6)
            PH = PH / deg.unsqueeze(-1)
            
            dH = (1.0 / self.tau_feat) * ((PH - H) - self.gamma * H)
            F_vals = self.force(H, edge_index)
            
            if self.use_hysteresis:
                dU = (1.0 / self.tau_topo) * ((1.0 - self.lam) * U - U**3 + F_vals)
            else:
                dU = (1.0 / self.tau_topo) * ((1.0 - self.lam) * U + F_vals)
            
            H = H + dt * dH
            U = U + dt * dU
        
        # Global pooling (mean)
        from torch_geometric.nn import global_mean_pool
        graph_emb = global_mean_pool(H, batch)
        
        # Decode
        out = self.decoder(graph_emb)
        return out
