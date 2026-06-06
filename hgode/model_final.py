"""
HGODE: Hysteresis-Gated Graph ODE
Final clean implementation.

Paper equations:
  Feature: τ_feat * dH/dt = G_φ(H, A_eff) - γ*H          (Eq. 5a)
  Topology: τ_topo * dU/dt = (1-λ)*U - U³ + F_θ(H)        (Eq. 5b)
  Adjacency: A_ij = σ(U_ij/τ)                              (Eq. 6)
  G_φ(H,A) = P*W*H - H, P = D^{-1/2}(A+I)D^{-1/2}       (diffusion)
  Force: F_ij = s * tanh(MLP([h_i || h_j]))                (Eq. 9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_softmax
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
    Uses symmetric normalization with self-loops for the diffusion.
    """
    def __init__(self, dim, edge_index, num_nodes, num_edges,
                 force_field, tau_feat=1.0, tau_topo=1.0, 
                 lam=0.3, gamma=0.5, gate_tau=0.2,
                 no_hysteresis=False, no_topo_search=False,
                 dropout=0.0):
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
        self._training = True

        # Learnable weight for diffusion
        self.W = nn.Linear(dim, dim, bias=False)
        
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('src', edge_index[0])
        self.register_buffer('dst', edge_index[1])
        
        # Add self-loops
        self_loops = torch.arange(num_nodes, dtype=torch.long)
        self.register_buffer('self_src', self_loops)
        self.register_buffer('self_dst', self_loops)
    
    def forward(self, t, state):
        self.nfe += 1
        N, d = self.num_nodes, self.dim
        
        H = state[:N * d].reshape(N, d)
        U = state[N * d:]
        
        # ---- Effective adjacency ----
        A_eff = torch.sigmoid(U / self.gate_tau)
        
        # ---- Build full edge list with self-loops ----
        # Candidate edges with learned weights
        all_src = torch.cat([self.src, self.self_src])
        all_dst = torch.cat([self.dst, self.self_dst])
        # Self-loop weight = 1.0
        all_weights = torch.cat([A_eff, torch.ones(N, device=H.device)])
        
        # ---- Symmetric normalization: D^{-1/2} A D^{-1/2} ----
        deg = scatter_add(all_weights, all_dst, dim=0, dim_size=N) + 1e-10
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[all_src] * deg_inv_sqrt[all_dst] * all_weights
        
        # Apply dropout to edge weights during training
        if self._training and self.dropout > 0:
            drop_mask = torch.bernoulli(torch.full_like(norm, 1.0 - self.dropout))
            norm = norm * drop_mask / (1.0 - self.dropout + 1e-10)
        
        # ---- Diffusion: P * W * H ----
        WH = self.W(H)
        msg = WH[all_src] * norm.unsqueeze(-1)
        PH = scatter_add(msg, all_dst, dim=0, dim_size=N)
        
        # G(H, A) = PH - H
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
                 task='node',
                 lam=0.3, gate_tau=0.2, tau_feat=1.0, tau_topo=1.0, gamma=0.5,
                 force_scale=1.0, force_hidden=64,
                 beta=0.1, delta=0.1,
                 T=0.6, solver='dopri5', rtol=1e-5, atol=1e-5,
                 num_steps=20,
                 dropout=0.5,
                 no_hysteresis=False, no_topo_search=False,
                 use_adjoint=False,
                 skip_alpha=0.5):
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
        self.num_steps = num_steps
        self.skip_alpha = skip_alpha
        
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
            step_size = self.T / self.num_steps
            state_T = solver_fn(
                self.ode_func, state0, t_span,
                method=self.solver,
                options={'step_size': step_size},
            )[-1]
        
        H_T = state_T[:N * d].reshape(N, d)
        
        # Skip connection: blend initial and final features
        H_out = self.skip_alpha * H0 + (1.0 - self.skip_alpha) * H_T
        
        if self.task == 'graph' and batch is not None:
            from torch_scatter import scatter_mean
            H_graph = scatter_mean(H_out, batch, dim=0)
            logits = self.decoder(H_graph)
        else:
            logits = self.decoder(H_out)
        
        if return_H:
            return logits, H_T
        return logits


class GraphHGODE(nn.Module):
    """
    HGODE for graph-level tasks.
    Uses per-batch ODE integration with fixed-step solver.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, task_type='graph_regression',
                 lam=0.3, gate_tau=0.2, tau_feat=1.0, tau_topo=1.0, gamma=0.5,
                 force_scale=1.0, force_hidden=64,
                 T=0.6, solver='euler', num_steps=10, dropout=0.5,
                 no_hysteresis=False, no_topo_search=False,
                 skip_alpha=0.5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.T = T
        self.num_steps = num_steps
        self.lam = lam
        self.gate_tau = gate_tau
        self.tau_feat = tau_feat
        self.tau_topo = tau_topo
        self.gamma = gamma
        self.no_hysteresis = no_hysteresis
        self.no_topo_search = no_topo_search
        self.task_type = task_type
        self.skip_alpha = skip_alpha
        
        self.u_stable = math.sqrt(1.0 - lam) if lam < 1.0 else 0.1
        
        # Encoder - handle integer features (ZINC) vs float
        self.atom_encoder = nn.Embedding(28, hidden_dim)  # for ZINC
        self.feat_encoder = nn.Linear(in_dim, hidden_dim)  # for others
        self.in_dim = in_dim
        
        # Learnable diffusion weight
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Force field
        self.force_field = ForceFieldMLP(hidden_dim, hidden=force_hidden, scale=force_scale)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        
        self.dropout = dropout
    
    def forward(self, batch_data):
        x = batch_data.x
        edge_index = batch_data.edge_index
        batch = batch_data.batch if hasattr(batch_data, 'batch') else None
        
        N = x.size(0)
        
        # Encode
        if x.dim() == 1 or (x.dim() == 2 and x.size(1) == 1):
            if x.dim() == 2:
                x = x.squeeze(-1)
            H = self.atom_encoder(x.long())
        else:
            H = self.feat_encoder(x.float())
        
        H0 = H.clone()
        
        src, dst = edge_index[0], edge_index[1]
        E = src.size(0)
        
        # Add self-loops
        self_loops = torch.arange(N, device=H.device)
        all_src = torch.cat([src, self_loops])
        all_dst = torch.cat([dst, self_loops])
        
        # Initialize U
        U = torch.full((E,), self.u_stable, device=H.device)
        
        dt = self.T / self.num_steps
        
        for step in range(self.num_steps):
            A_eff = torch.sigmoid(U / self.gate_tau)
            all_weights = torch.cat([A_eff, torch.ones(N, device=H.device)])
            
            # Symmetric normalization
            deg = scatter_add(all_weights, all_dst, dim=0, dim_size=N) + 1e-10
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[all_src] * deg_inv_sqrt[all_dst] * all_weights
            
            if self.training and self.dropout > 0:
                drop_mask = torch.bernoulli(torch.full_like(norm, 1.0 - self.dropout))
                norm = norm * drop_mask / (1.0 - self.dropout + 1e-10)
            
            WH = self.W(H)
            msg = WH[all_src] * norm.unsqueeze(-1)
            PH = scatter_add(msg, all_dst, dim=0, dim_size=N)
            
            G = PH - H
            dH = (1.0 / self.tau_feat) * (G - self.gamma * H)
            
            if not self.no_topo_search:
                F_force = self.force_field(H[src], H[dst])
                if self.no_hysteresis:
                    dU = (1.0 / self.tau_topo) * (-U + F_force)
                else:
                    dU = (1.0 / self.tau_topo) * ((1.0 - self.lam) * U - U.pow(3) + F_force)
                U = U + dt * dU
            
            H = H + dt * dH
        
        # Skip connection
        H_out = self.skip_alpha * H0 + (1.0 - self.skip_alpha) * H
        
        # Readout
        if batch is not None:
            from torch_scatter import scatter_mean
            H_graph = scatter_mean(H_out, batch, dim=0)
        else:
            H_graph = H_out.mean(dim=0, keepdim=True)
        
        return self.decoder(H_graph)
