"""
Core HGODE model: Hysteresis Graph ODE.

Implements the coupled feature-topology ODE system:
  τ_feat * dH/dt = G_φ(H, A) - γ*H
  τ_topo * dU/dt = (1-λ)*U - U³ + F_θ(H)

where A_ij = σ(U_ij/τ) * μ(t) * 1[(i,j) ∈ E_cand]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean
from torchdiffeq import odeint
import math


class ForceFieldMLP(nn.Module):
    """
    Topological force field F_θ(h_i, h_j) = s * tanh(MLP([h_i || h_j]))
    """
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, scale=0.5):
        super().__init__()
        self.scale = scale
        
        layers = []
        layers.append(nn.Linear(2 * in_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, h_src, h_dst):
        h_cat = torch.cat([h_src, h_dst], dim=-1)
        return self.scale * torch.tanh(self.mlp(h_cat).squeeze(-1))


class GraphNeuralOperator(nn.Module):
    """
    Graph neural operator G_φ(H, A).
    Diffusion-style: G_φ(H, A) = P*H*W - H
    where P = D^{-1} * A_tilde (row-normalized adjacency with self-loops)
    """
    def __init__(self, in_dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or in_dim
        self.weight = nn.Linear(in_dim, out_dim, bias=True)
    
    def forward(self, H, edge_index, edge_weight, num_nodes):
        src, dst = edge_index[0], edge_index[1]
        
        # Add self-loops
        self_loop_idx = torch.arange(num_nodes, device=H.device)
        all_src = torch.cat([src, self_loop_idx])
        all_dst = torch.cat([dst, self_loop_idx])
        all_weight = torch.cat([edge_weight, torch.ones(num_nodes, device=H.device)])
        
        # Row-normalize (by destination)
        deg = scatter_add(all_weight, all_dst, dim=0, dim_size=num_nodes)
        deg_inv = 1.0 / (deg + 1e-10)
        norm_weight = all_weight * deg_inv[all_dst]
        
        # Message passing: P * H
        msg = H[all_src] * norm_weight.unsqueeze(-1)
        agg = scatter_add(msg, all_dst, dim=0, dim_size=num_nodes)
        
        # Apply weight and residual-like skip
        out = self.weight(agg) - H
        return out


class HGODEFunc(nn.Module):
    """
    ODE function for the coupled HGODE system.
    State: [H_flat, U] concatenated
    """
    def __init__(self, feat_dim, cand_edge_index, num_nodes,
                 tau_feat=1.0, tau_topo=1.0, lam=0.5, gamma=0.1,
                 gate_tau=0.1, force_scale=0.5, hidden_dim=64,
                 no_hysteresis=False, dropout=0.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_nodes = num_nodes
        self.tau_feat = tau_feat
        self.tau_topo = tau_topo
        self.lam = lam
        self.gamma = gamma
        self.gate_tau = gate_tau
        self.no_hysteresis = no_hysteresis
        self.dropout = dropout
        
        self.register_buffer('cand_edge_index', cand_edge_index)
        self.num_cand_edges = cand_edge_index.size(1)
        
        self.force_field = ForceFieldMLP(feat_dim, hidden_dim=hidden_dim, scale=force_scale)
        self.graph_op = GraphNeuralOperator(feat_dim, feat_dim)
    
    def forward(self, t, state):
        N, d = self.num_nodes, self.feat_dim
        E = self.num_cand_edges
        
        H = state[:N * d].reshape(N, d)
        U = state[N * d:]
        
        # Effective adjacency: A_ij = σ(U_ij / τ)
        A_eff = torch.sigmoid(U / self.gate_tau)
        
        # Feature dynamics: dH/dt = (1/τ_feat) * (G_φ(H, A) - γ*H)
        G_out = self.graph_op(H, self.cand_edge_index, A_eff, N)
        dH = (1.0 / self.tau_feat) * (G_out - self.gamma * H)
        
        # Topology dynamics
        src, dst = self.cand_edge_index[0], self.cand_edge_index[1]
        F_force = self.force_field(H[src], H[dst])
        
        if self.no_hysteresis:
            dU = (1.0 / self.tau_topo) * (-U + F_force)
        else:
            dU = (1.0 / self.tau_topo) * ((1.0 - self.lam) * U - U.pow(3) + F_force)
        
        dstate = torch.cat([dH.reshape(-1), dU])
        return dstate


class HGODE(nn.Module):
    """
    Full HGODE model for node classification.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, cand_edge_index, num_nodes,
                 orig_edge_index=None,
                 tau_feat=1.0, tau_topo=1.0, lam=0.5, gamma=0.1,
                 gate_tau=0.1, force_scale=0.5, T=1.0,
                 solver='dopri5', rtol=1e-5, atol=1e-5,
                 beta=0.1, delta=0.05, force_hidden=64,
                 use_encoder=True, no_hysteresis=False, dropout=0.0,
                 u_init_orig=2.0, u_init_cand=-2.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.T = T
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.beta = beta
        self.delta = delta
        self.u_init_orig = u_init_orig
        self.u_init_cand = u_init_cand
        
        # F_crit = 2/(3*sqrt(3))
        self.F_crit = 2.0 / (3.0 * math.sqrt(3.0))
        
        # Encoder
        self.use_encoder = use_encoder
        if use_encoder:
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            assert in_dim == hidden_dim
        
        # ODE function
        self.ode_func = HGODEFunc(
            feat_dim=hidden_dim,
            cand_edge_index=cand_edge_index,
            num_nodes=num_nodes,
            tau_feat=tau_feat,
            tau_topo=tau_topo,
            lam=lam,
            gamma=gamma,
            gate_tau=gate_tau,
            force_scale=force_scale,
            hidden_dim=force_hidden,
            no_hysteresis=no_hysteresis,
            dropout=dropout,
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        
        # Compute initial U values: mark original edges vs candidate-only edges
        self.register_buffer('cand_edge_index', cand_edge_index)
        
        # Create mask for original edges within candidate pool
        if orig_edge_index is not None:
            orig_set = set()
            oe = orig_edge_index.cpu()
            for i in range(oe.size(1)):
                orig_set.add((oe[0, i].item(), oe[1, i].item()))
            
            is_orig = torch.zeros(cand_edge_index.size(1), dtype=torch.bool)
            ce = cand_edge_index.cpu()
            for i in range(ce.size(1)):
                if (ce[0, i].item(), ce[1, i].item()) in orig_set:
                    is_orig[i] = True
            self.register_buffer('is_orig_edge', is_orig)
        else:
            # All edges are treated the same
            self.register_buffer('is_orig_edge', 
                               torch.ones(cand_edge_index.size(1), dtype=torch.bool))
    
    def get_U0(self, device):
        """Get initial U values with orig edges on and candidate edges off."""
        U0 = torch.full((self.cand_edge_index.size(1),), self.u_init_cand, device=device)
        U0[self.is_orig_edge] = self.u_init_orig
        return U0
    
    def compute_margin_loss(self, H, labels, mask=None):
        """
        Compute force margin loss (Eq. 10).
        """
        src, dst = self.cand_edge_index[0], self.cand_edge_index[1]
        h_src = H[src]
        h_dst = H[dst]
        
        F_ij = self.ode_func.force_field(h_src, h_dst)
        
        label_src = labels[src]
        label_dst = labels[dst]
        
        if mask is not None:
            valid = mask[src] & mask[dst]
        else:
            valid = torch.ones(src.size(0), dtype=torch.bool, device=src.device)
        
        same_label = (label_src == label_dst) & valid
        diff_label = (label_src != label_dst) & valid
        
        threshold = self.F_crit + self.delta
        
        loss = torch.tensor(0.0, device=H.device)
        if same_label.any():
            loss = loss + F.softplus(threshold - F_ij[same_label]).mean()
        if diff_label.any():
            loss = loss + F.softplus(threshold + F_ij[diff_label]).mean()
        
        return loss
    
    def forward(self, x, return_margin_data=False):
        if self.use_encoder:
            H0 = self.encoder(x)
        else:
            H0 = x
        
        N, d = H0.shape
        
        # Initialize U with original edges "on" and candidate edges "off"
        U0 = self.get_U0(x.device)
        
        state0 = torch.cat([H0.reshape(-1), U0])
        
        t_span = torch.tensor([0.0, self.T], device=x.device)
        
        state_T = odeint(
            self.ode_func, state0, t_span,
            method=self.solver,
            rtol=self.rtol, atol=self.atol
        )[-1]
        
        H_T = state_T[:N * d].reshape(N, d)
        
        logits = self.decoder(H_T)
        
        if return_margin_data:
            return logits, H_T
        return logits


class HGODEForGraphClassification(nn.Module):
    """
    HGODE variant for graph-level tasks.
    For graph-level tasks, candidate pool = original edges (no extra candidates 
    since graphs are small and topology search happens within each graph).
    """
    def __init__(self, in_dim, hidden_dim, out_dim,
                 tau_feat=1.0, tau_topo=1.0, lam=0.5, gamma=0.1,
                 gate_tau=0.1, force_scale=0.5, T=1.0,
                 solver='dopri5', rtol=1e-5, atol=1e-5,
                 beta=0.1, delta=0.05, force_hidden=64,
                 no_hysteresis=False, dropout=0.0,
                 num_hops=2, num_random=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.T = T
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.beta = beta
        self.delta = delta
        self.no_hysteresis = no_hysteresis
        self.num_hops = num_hops
        self.num_random = num_random
        
        self.F_crit = 2.0 / (3.0 * math.sqrt(3.0))
        
        # Shared modules
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.force_field = ForceFieldMLP(hidden_dim, hidden_dim=force_hidden, scale=force_scale)
        self.graph_op = GraphNeuralOperator(hidden_dim, hidden_dim)
        
        self.tau_feat = tau_feat
        self.tau_topo = tau_topo
        self.lam = lam
        self.gamma = gamma
        self.gate_tau = gate_tau
        
        # Readout + decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def ode_func(self, t, state, cand_edge_index, num_nodes):
        N, d = num_nodes, self.hidden_dim
        
        H = state[:N * d].reshape(N, d)
        U = state[N * d:]
        
        A_eff = torch.sigmoid(U / self.gate_tau)
        
        G_out = self.graph_op(H, cand_edge_index, A_eff, N)
        dH = (1.0 / self.tau_feat) * (G_out - self.gamma * H)
        
        src, dst = cand_edge_index[0], cand_edge_index[1]
        F_force = self.force_field(H[src], H[dst])
        
        if self.no_hysteresis:
            dU = (1.0 / self.tau_topo) * (-U + F_force)
        else:
            dU = (1.0 / self.tau_topo) * ((1.0 - self.lam) * U - U.pow(3) + F_force)
        
        return torch.cat([dH.reshape(-1), dU])
    
    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch
        
        H0 = self.encoder(x)
        
        num_nodes = x.size(0)
        E = edge_index.size(1)
        # For graph tasks, start with all edges "on" (using original edges as candidate pool)
        U0 = torch.ones(E, device=x.device) * 2.0  # positive init -> edges enabled
        state0 = torch.cat([H0.reshape(-1), U0])
        
        def func(t, state):
            return self.ode_func(t, state, edge_index, num_nodes)
        
        t_span = torch.tensor([0.0, self.T], device=x.device)
        state_T = odeint(func, state0, t_span, method=self.solver,
                        rtol=self.rtol, atol=self.atol)[-1]
        
        H_T = state_T[:num_nodes * self.hidden_dim].reshape(num_nodes, self.hidden_dim)
        
        graph_emb = scatter_mean(H_T, batch_idx, dim=0)
        
        logits = self.decoder(graph_emb)
        return logits
