"""
HGODE: Hysteresis-Gated Graph ODE
Final clean implementation covering node-level and graph-level tasks.

Paper equations:
  Feature: τ_feat * dH/dt = G_φ(H, A) - γ*H          (Eq. 5a)
  Topology: τ_topo * dU/dt = (1-λ)*U - U³ + F_θ(H)    (Eq. 5b)
  Adjacency: A_ij = σ(U_ij/τ) * 1[(i,j) ∈ E_cand]     (Eq. 6)
  G_φ(H,A) = PH - H, where P = D^{-1}A                (diffusion)
  Force: F_ij = s * tanh(MLP([h_i || h_j]))            (Eq. 9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torchdiffeq import odeint, odeint_adjoint
import math


class ForceFieldMLP(nn.Module):
    """Force field: F_ij = s * tanh(MLP([h_i || h_j]))"""
    def __init__(self, dim, hidden=64, scale=1.0):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, h_src, h_dst):
        h_cat = torch.cat([h_src, h_dst], dim=-1)
        return self.scale * torch.tanh(self.net(h_cat).squeeze(-1))


class CoupledODEFunc(nn.Module):
    """
    Coupled ODE dynamics for features H and topology potentials U.
    """
    def __init__(self, dim, edge_index, num_nodes, num_edges,
                 force_field, tau_feat=1.0, tau_topo=1.0, 
                 lam=0.3, gamma=0.5, gate_tau=0.2,
                 no_hysteresis=False, no_topo_search=False,
                 dropout=0.0, training=True):
        super().__init__()
        self.dim = dim
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.tau_feat = tau_feat
        self.tau_topo = tau_topo
        self.lam = lam
        self.gamma = gamma
        self.gate_tau = gate_tau
        self.no_hysteresis = no_hysteresis
        self.no_topo_search = no_topo_search
        self.dropout = dropout
        self.force_field = force_field
        self.nfe = 0
        self._training = training

        self.register_buffer('edge_index', edge_index)
        self.register_buffer('src', edge_index[0])
        self.register_buffer('dst', edge_index[1])
    
    def forward(self, t, state):
        self.nfe += 1
        N, d = self.num_nodes, self.dim
        
        H = state[:N * d].reshape(N, d)
        U = state[N * d:]
        
        # ---- Feature dynamics ----
        # Effective adjacency: A_ij = σ(U_ij / τ)
        A_eff = torch.sigmoid(U / self.gate_tau)
        
        # Apply dropout to adjacency during training
        if self._training and self.dropout > 0:
            drop_mask = torch.bernoulli(torch.full_like(A_eff, 1.0 - self.dropout))
            A_eff = A_eff * drop_mask / (1.0 - self.dropout + 1e-10)
        
        # Row-normalized diffusion: P = D^{-1}A
        # For each dst node, aggregate from src nodes
        deg = scatter_add(A_eff, self.dst, dim=0, dim_size=N) + 1e-10
        norm_weight = A_eff / deg[self.dst]
        
        # PH: aggregate neighbor features
        msg = H[self.src] * norm_weight.unsqueeze(-1)
        PH = scatter_add(msg, self.dst, dim=0, dim_size=N)
        
        # G(H, A) = PH - H (Laplacian diffusion)
        G = PH - H
        dH = (1.0 / self.tau_feat) * (G - self.gamma * H)
        
        # ---- Topology dynamics ----
        if self.no_topo_search:
            dU = torch.zeros_like(U)
        else:
            F_force = self.force_field(H[self.src], H[self.dst])
            
            if self.no_hysteresis:
                dU = (1.0 / self.tau_topo) * (-U + F_force)
            else:
                dU = (1.0 / self.tau_topo) * ((1.0 - self.lam) * U - U.pow(3) + F_force)
        
        dstate = torch.cat([dH.reshape(-1), dU])
        return dstate


class HGODEModel(nn.Module):
    """
    HGODE model for node classification and graph classification.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, 
                 cand_edge_index, num_nodes, orig_edge_mask,
                 task='node',  # 'node' or 'graph'
                 # Hysteresis params
                 lam=0.3, gate_tau=0.2, tau_feat=1.0, tau_topo=1.0, gamma=0.5,
                 # Force params  
                 force_scale=1.0, force_hidden=64,
                 # Margin loss params
                 beta=0.1, delta=0.1,
                 # ODE params
                 T=0.6, solver='dopri5', rtol=1e-5, atol=1e-5,
                 # Architecture params
                 dropout=0.5,
                 # Ablation flags
                 no_hysteresis=False, no_topo_search=False,
                 use_adjoint=False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.T = T
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.beta = beta
        self.delta = delta
        self.lam = lam
        self.task = task
        self.use_adjoint = use_adjoint
        
        self.F_crit = 2.0 / (3.0 * math.sqrt(3.0)) * (1.0 - lam) ** 1.5
        self.u_stable = math.sqrt(1.0 - lam) if lam < 1.0 else 0.1
        
        num_edges = cand_edge_index.size(1)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Force field
        self.force_field = ForceFieldMLP(hidden_dim, hidden=force_hidden, scale=force_scale)
        
        # ODE function
        self.ode_func = CoupledODEFunc(
            dim=hidden_dim,
            edge_index=cand_edge_index,
            num_nodes=num_nodes,
            num_edges=num_edges,
            force_field=self.force_field,
            tau_feat=tau_feat,
            tau_topo=tau_topo,
            lam=lam,
            gamma=gamma,
            gate_tau=gate_tau,
            no_hysteresis=no_hysteresis,
            no_topo_search=no_topo_search,
            dropout=dropout,
        )
        
        # Decoder
        if task == 'node':
            self.decoder = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            # Graph-level: readout + MLP
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        
        # Store edge info
        self.register_buffer('cand_edge_index', cand_edge_index)
        self.register_buffer('orig_edge_mask', orig_edge_mask)
    
    def get_initial_U(self, device):
        """Initialize U: +u_stable for original edges, -u_stable for candidate-only."""
        U0 = torch.full((self.cand_edge_index.size(1),), -self.u_stable, device=device)
        U0[self.orig_edge_mask] = self.u_stable
        return U0
    
    def compute_margin_loss(self, H, labels, mask=None):
        """Force margin loss (Eq. 10-11)."""
        src, dst = self.cand_edge_index[0], self.cand_edge_index[1]
        
        E = src.size(0)
        max_edges = 5000
        if E > max_edges:
            idx = torch.randperm(E, device=src.device)[:max_edges]
            src_s, dst_s = src[idx], dst[idx]
        else:
            src_s, dst_s = src, dst
        
        F_ij = self.force_field(H[src_s].detach(), H[dst_s].detach())
        
        if mask is not None:
            valid = mask[src_s] & mask[dst_s]
        else:
            valid = torch.ones_like(src_s, dtype=torch.bool)
        
        same = (labels[src_s] == labels[dst_s]) & valid
        diff = (labels[src_s] != labels[dst_s]) & valid
        
        threshold = self.F_crit + self.delta
        
        loss = torch.tensor(0.0, device=H.device)
        if same.any():
            loss = loss + F.softplus(threshold - F_ij[same]).mean()
        if diff.any():
            loss = loss + F.softplus(threshold + F_ij[diff]).mean()
        
        return loss
    
    def forward(self, x, batch=None, return_H=False):
        """
        Args:
            x: node features [N, d_in]
            batch: batch assignment for graph-level tasks [N]
            return_H: whether to return intermediate features
        """
        self.ode_func._training = self.training
        
        H0 = self.encoder(x)
        N, d = H0.shape
        
        U0 = self.get_initial_U(x.device)
        state0 = torch.cat([H0.reshape(-1), U0])
        
        t_span = torch.tensor([0.0, self.T], device=x.device, dtype=torch.float32)
        
        self.ode_func.nfe = 0
        
        solver_fn = odeint_adjoint if self.use_adjoint else odeint
        
        if self.solver == 'dopri5':
            state_T = solver_fn(
                self.ode_func, state0, t_span,
                method='dopri5',
                rtol=self.rtol, atol=self.atol,
            )[-1]
        else:
            step_size = self.T / 20
            state_T = solver_fn(
                self.ode_func, state0, t_span,
                method=self.solver,
                options={'step_size': step_size},
            )[-1]
        
        H_T = state_T[:N * d].reshape(N, d)
        
        if self.task == 'graph' and batch is not None:
            # Global mean pooling
            from torch_scatter import scatter_mean
            H_graph = scatter_mean(H_T, batch, dim=0)
            logits = self.decoder(H_graph)
        else:
            logits = self.decoder(H_T)
        
        if return_H:
            return logits, H_T
        return logits
