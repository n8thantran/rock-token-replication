"""
Core HGODE model: Hysteresis Graph ODE.

Implements the coupled feature-topology ODE system (Eq. 5):
  τ_feat * dH/dt = G_φ(H, A) - γ*H
  τ_topo * dU/dt = (1-λ)*U - U³ + F_θ(H)

where A_ij = σ(U_ij/τ) * 1[(i,j) ∈ E_cand]  (Eq. 6)

Key design: G_φ is a simple diffusion operator (PH - H) following GRAND-style.
The force field MLP is the main learnable component inside the ODE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch_scatter import scatter_add, scatter_mean
from torchdiffeq import odeint, odeint_adjoint
import math


class ForceFieldMLP(nn.Module):
    """
    Topological force field (Eq. 9):
    F_ij = s * tanh(MLP([h_i || h_j]))
    """
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, scale=1.0):
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


def graph_diffusion(H, edge_index, edge_weight, num_nodes):
    """
    Simple graph diffusion: G(H, A) = PH - H
    where P = D^{-1} * A (row-normalized weighted adjacency with self-loops)
    
    This is parameter-free - just message passing with normalization.
    """
    src, dst = edge_index[0], edge_index[1]
    
    # Add self-loops with weight 1
    self_loop_idx = torch.arange(num_nodes, device=H.device)
    all_src = torch.cat([src, self_loop_idx])
    all_dst = torch.cat([dst, self_loop_idx])
    all_weight = torch.cat([edge_weight, torch.ones(num_nodes, device=H.device)])
    
    # Row-normalize: P = D^{-1} A
    deg = scatter_add(all_weight, all_dst, dim=0, dim_size=num_nodes)
    deg_inv = 1.0 / (deg + 1e-10)
    norm_weight = all_weight * deg_inv[all_dst]
    
    # Message passing: P * H
    msg = H[all_src] * norm_weight.unsqueeze(-1)
    PH = scatter_add(msg, all_dst, dim=0, dim_size=num_nodes)
    
    # G(H, A) = PH - H
    return PH - H


class AttentionDiffusion(nn.Module):
    """
    Learnable attention-based diffusion operator.
    G_φ(H, A) = sum_j a_ij * W * h_j - h_i
    where a_ij = softmax_j(edge_weight_ij * score(h_i, h_j))
    """
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert dim % heads == 0
        
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
    
    def forward(self, H, edge_index, edge_weight, num_nodes):
        src, dst = edge_index[0], edge_index[1]
        
        Q = self.W_q(H).view(num_nodes, self.heads, self.head_dim)
        K = self.W_k(H).view(num_nodes, self.heads, self.head_dim)
        V = self.W_v(H).view(num_nodes, self.heads, self.head_dim)
        
        # Compute attention scores for edges
        q_src = Q[dst]  # queries from destination
        k_dst = K[src]  # keys from source
        attn = (q_src * k_dst).sum(dim=-1) / math.sqrt(self.head_dim)  # [E, heads]
        
        # Multiply by edge weight (from U potentials)
        attn = attn * edge_weight.unsqueeze(-1)
        
        # Add self-loops
        self_loop_idx = torch.arange(num_nodes, device=H.device)
        self_q = Q  # [N, heads, head_dim]
        self_k = K
        self_attn = (self_q * self_k).sum(dim=-1) / math.sqrt(self.head_dim)  # [N, heads]
        
        # Softmax normalization per destination node
        # Combine edge attention and self-loop attention
        all_src = torch.cat([src, self_loop_idx])
        all_dst = torch.cat([dst, self_loop_idx])
        all_attn = torch.cat([attn, self_attn], dim=0)  # [E+N, heads]
        
        # Softmax per destination
        attn_max = scatter_add(torch.zeros_like(all_attn), all_dst.unsqueeze(-1).expand_as(all_attn), 
                               dim=0, dim_size=num_nodes)
        # Use scatter for softmax
        from torch_scatter import scatter_softmax
        all_attn_soft = scatter_softmax(all_attn, all_dst.unsqueeze(-1).expand_as(all_attn), dim=0)
        
        if self.training and self.dropout > 0:
            all_attn_soft = Func.dropout(all_attn_soft, p=self.dropout, training=True)
        
        # Aggregate values
        all_V = torch.cat([V[src], V], dim=0)  # [E+N, heads, head_dim]
        weighted_V = all_V * all_attn_soft.unsqueeze(-1)
        agg = scatter_add(weighted_V, all_dst.unsqueeze(-1).unsqueeze(-1).expand_as(weighted_V),
                         dim=0, dim_size=num_nodes)
        
        out = agg.view(num_nodes, self.dim)
        return out - H


class HGODEFunc(nn.Module):
    """
    ODE function for the coupled HGODE system.
    State: [H_flat, U] concatenated
    """
    def __init__(self, feat_dim, cand_edge_index, num_nodes,
                 tau_feat=1.0, tau_topo=1.0, lam=0.3, gamma=0.5,
                 gate_tau=0.2, force_scale=1.0, force_hidden=64,
                 use_attention=False, attn_heads=4, dropout=0.0,
                 no_hysteresis=False, no_topo_search=False):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_nodes = num_nodes
        self.tau_feat = tau_feat
        self.tau_topo = tau_topo
        self.lam = lam
        self.gamma = gamma
        self.gate_tau = gate_tau
        self.no_hysteresis = no_hysteresis
        self.no_topo_search = no_topo_search
        self.use_attention = use_attention
        
        self.register_buffer('cand_edge_index', cand_edge_index)
        self.num_cand_edges = cand_edge_index.size(1)
        
        self.force_field = ForceFieldMLP(feat_dim, hidden_dim=force_hidden, scale=force_scale)
        
        if use_attention:
            self.graph_op = AttentionDiffusion(feat_dim, heads=attn_heads, dropout=dropout)
        
        # NFE counter for debugging
        self.nfe = 0
    
    def forward(self, t, state):
        self.nfe += 1
        N, d = self.num_nodes, self.feat_dim
        
        H = state[:N * d].reshape(N, d)
        U = state[N * d:]
        
        # Effective adjacency: A_ij = σ(U_ij / τ)  (Eq. 6)
        A_eff = torch.sigmoid(U / self.gate_tau)
        
        # Feature dynamics: dH/dt = (1/τ_feat) * (G_φ(H, A) - γ*H)
        if self.use_attention:
            G_out = self.graph_op(H, self.cand_edge_index, A_eff, N)
        else:
            G_out = graph_diffusion(H, self.cand_edge_index, A_eff, N)
        dH = (1.0 / self.tau_feat) * (G_out - self.gamma * H)
        
        # Topology dynamics: dU/dt = (1/τ_topo) * ((1-λ)*U - U³ + F)
        src, dst = self.cand_edge_index[0], self.cand_edge_index[1]
        F_force = self.force_field(H[src], H[dst])
        
        if self.no_topo_search:
            dU = torch.zeros_like(U)
        elif self.no_hysteresis:
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
                 tau_feat=1.0, tau_topo=1.0, lam=0.3, gamma=0.5,
                 gate_tau=0.2, force_scale=1.0, T=1.0,
                 solver='dopri5', rtol=1e-5, atol=1e-5,
                 beta=0.1, delta=0.1, force_hidden=64,
                 dropout=0.5, ode_dropout=0.0,
                 use_attention=False, attn_heads=4,
                 no_hysteresis=False, no_topo_search=False,
                 use_adjoint=False, num_steps=10):
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
        self.use_adjoint = use_adjoint
        self.num_steps = num_steps
        
        # F_crit = 2/(3*sqrt(3)) * (1-λ)^(3/2)
        self.F_crit = 2.0 / (3.0 * math.sqrt(3.0)) * (1.0 - lam) ** 1.5
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
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
            force_hidden=force_hidden,
            use_attention=use_attention,
            attn_heads=attn_heads,
            dropout=ode_dropout,
            no_hysteresis=no_hysteresis,
            no_topo_search=no_topo_search,
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        
        # Store candidate edge info
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
            self.register_buffer('is_orig_edge', 
                               torch.ones(cand_edge_index.size(1), dtype=torch.bool))
        
        # Compute stable fixed point for initialization
        self.u_stable = math.sqrt(1.0 - lam) if lam < 1.0 else 0.1
    
    def get_U0(self, device):
        """Get initial U values: orig edges at +stable, candidate edges at -stable."""
        U0 = torch.full((self.cand_edge_index.size(1),), -self.u_stable, device=device)
        U0[self.is_orig_edge] = self.u_stable
        return U0
    
    def compute_margin_loss(self, H, labels, mask=None):
        """
        Compute force margin loss (Eq. 10-11).
        """
        src, dst = self.cand_edge_index[0], self.cand_edge_index[1]
        
        E = src.size(0)
        max_edges = 5000
        if E > max_edges:
            idx = torch.randperm(E, device=src.device)[:max_edges]
            src_s, dst_s = src[idx], dst[idx]
        else:
            src_s, dst_s = src, dst
        
        F_ij = self.ode_func.force_field(H[src_s], H[dst_s])
        
        label_src = labels[src_s]
        label_dst = labels[dst_s]
        
        if mask is not None:
            valid = mask[src_s] & mask[dst_s]
        else:
            valid = torch.ones(src_s.size(0), dtype=torch.bool, device=src_s.device)
        
        same_label = (label_src == label_dst) & valid
        diff_label = (label_src != label_dst) & valid
        
        threshold = self.F_crit + self.delta
        
        loss = torch.tensor(0.0, device=H.device)
        if same_label.any():
            loss = loss + Func.softplus(threshold - F_ij[same_label]).mean()
        if diff_label.any():
            loss = loss + Func.softplus(threshold + F_ij[diff_label]).mean()
        
        return loss
    
    def forward(self, x, return_margin_data=False):
        H0 = self.encoder(x)
        
        N, d = H0.shape
        
        U0 = self.get_U0(x.device)
        
        state0 = torch.cat([H0.reshape(-1), U0])
        
        t_span = torch.tensor([0.0, self.T], device=x.device)
        
        self.ode_func.nfe = 0
        
        ode_solver = odeint_adjoint if self.use_adjoint else odeint
        
        if self.solver == 'dopri5':
            state_T = ode_solver(
                self.ode_func, state0, t_span,
                method=self.solver,
                rtol=self.rtol, atol=self.atol
            )[-1]
        elif self.solver in ('rk4', 'euler', 'midpoint'):
            state_T = ode_solver(
                self.ode_func, state0, t_span,
                method=self.solver,
                options={'step_size': self.T / self.num_steps}
            )[-1]
        else:
            state_T = ode_solver(
                self.ode_func, state0, t_span,
                method=self.solver,
            )[-1]
        
        H_T = state_T[:N * d].reshape(N, d)
        
        logits = self.decoder(H_T)
        
        if return_margin_data:
            return logits, H_T
        return logits


class HGODEForGraphClassification(nn.Module):
    """
    HGODE variant for graph-level tasks.
    """
    def __init__(self, in_dim, hidden_dim, out_dim,
                 tau_feat=1.0, tau_topo=1.0, lam=0.5, gamma=0.5,
                 gate_tau=0.2, force_scale=1.0, T=1.0,
                 solver='dopri5', rtol=1e-5, atol=1e-5,
                 beta=0.1, delta=0.1, force_hidden=64,
                 no_hysteresis=False, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.T = T
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.beta = beta
        self.delta = delta
        self.lam = lam
        self.no_hysteresis = no_hysteresis
        
        self.F_crit = 2.0 / (3.0 * math.sqrt(3.0)) * (1.0 - lam) ** 1.5
        
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.force_field = ForceFieldMLP(hidden_dim, hidden_dim=force_hidden, scale=force_scale)
        
        self.tau_feat = tau_feat
        self.tau_topo = tau_topo
        self.gamma = gamma
        self.gate_tau = gate_tau
        
        self.u_stable = math.sqrt(1.0 - lam) if lam < 1.0 else 0.1
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def ode_func(self, t, state, edge_index, num_nodes):
        N, d = num_nodes, self.hidden_dim
        
        H = state[:N * d].reshape(N, d)
        U = state[N * d:]
        
        A_eff = torch.sigmoid(U / self.gate_tau)
        
        G_out = graph_diffusion(H, edge_index, A_eff, N)
        dH = (1.0 / self.tau_feat) * (G_out - self.gamma * H)
        
        src, dst = edge_index[0], edge_index[1]
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
        U0 = torch.ones(E, device=x.device) * self.u_stable
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
