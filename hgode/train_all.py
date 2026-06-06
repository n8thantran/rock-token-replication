"""
Unified training script for HGODE across all 6 datasets.
Handles node classification (Cora, Chameleon, ogbn-proteins) and 
graph-level tasks (ZINC, Peptides-func, ogbg-molpcba).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import math
import argparse
import time
from torch_scatter import scatter_add

# ============================================================
# Dataset loading
# ============================================================

def load_cora(device):
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
    data = dataset[0].to(device)
    return data, dataset.num_features, dataset.num_classes, 'node'

def load_chameleon(device):
    from torch_geometric.datasets import WikipediaNetwork
    from torch_geometric.transforms import NormalizeFeatures
    dataset = WikipediaNetwork(root='./data', name='chameleon', transform=NormalizeFeatures())
    data = dataset[0].to(device)
    return data, dataset.num_features, dataset.num_classes, 'node'

def load_ogbn_proteins(device):
    """ogbn-proteins: node-level multi-label classification."""
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name='ogbn-proteins', root='./data')
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        
        # ogbn-proteins has edge features but no node features
        # Use node degree as features
        N = data.num_nodes
        edge_index = data.edge_index
        deg = scatter_add(torch.ones(edge_index.size(1)), edge_index[0], dim=0, dim_size=N)
        data.x = deg.unsqueeze(-1).float()
        
        # Also add edge_attr aggregation as node features
        if data.edge_attr is not None:
            edge_feat_agg = scatter_add(data.edge_attr.float(), edge_index[1], dim=0, dim_size=N)
            data.x = torch.cat([data.x, edge_feat_agg], dim=-1)
        
        data.train_mask = torch.zeros(N, dtype=torch.bool)
        data.val_mask = torch.zeros(N, dtype=torch.bool)
        data.test_mask = torch.zeros(N, dtype=torch.bool)
        data.train_mask[split_idx['train']] = True
        data.val_mask[split_idx['valid']] = True
        data.test_mask[split_idx['test']] = True
        
        # Multi-label: y is [N, 112]
        data.y = data.y.float()
        
        data = data.to(device)
        num_features = data.x.size(1)
        num_classes = data.y.size(1)  # 112 binary labels
        return data, num_features, num_classes, 'node_multilabel'
    except ImportError:
        print("OGB not installed. Install with: pip install ogb")
        return None, None, None, None

def load_zinc(device):
    """ZINC: graph regression (MAE)."""
    try:
        from torch_geometric.datasets import ZINC
        train_dataset = ZINC(root='./data/ZINC', subset=True, split='train')
        val_dataset = ZINC(root='./data/ZINC', subset=True, split='val')
        test_dataset = ZINC(root='./data/ZINC', subset=True, split='test')
        
        # ZINC has atom type as x (integer), need to embed
        num_features = 28  # number of atom types
        num_classes = 1  # regression target
        return (train_dataset, val_dataset, test_dataset), num_features, num_classes, 'graph_regression'
    except Exception as e:
        print(f"Error loading ZINC: {e}")
        return None, None, None, None

def load_peptides_func(device):
    """Peptides-func: graph multi-label classification (AP)."""
    try:
        from torch_geometric.datasets import LRGBDataset
        train_dataset = LRGBDataset(root='./data/LRGB', name='Peptides-func', split='train')
        val_dataset = LRGBDataset(root='./data/LRGB', name='Peptides-func', split='val')
        test_dataset = LRGBDataset(root='./data/LRGB', name='Peptides-func', split='test')
        
        num_features = train_dataset[0].x.size(1)
        num_classes = train_dataset[0].y.size(1) if train_dataset[0].y.dim() > 1 else 10
        return (train_dataset, val_dataset, test_dataset), num_features, num_classes, 'graph_multilabel'
    except Exception as e:
        print(f"Error loading Peptides-func: {e}")
        return None, None, None, None

def load_ogbg_molpcba(device):
    """ogbg-molpcba: graph multi-label classification (AP)."""
    try:
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(name='ogbg-molpcba', root='./data')
        split_idx = dataset.get_idx_split()
        
        train_dataset = dataset[split_idx['train']]
        val_dataset = dataset[split_idx['valid']]
        test_dataset = dataset[split_idx['test']]
        
        num_features = dataset[0].x.size(1) if dataset[0].x is not None else 9
        num_classes = dataset[0].y.size(1)  # 128 tasks
        return (train_dataset, val_dataset, test_dataset), num_features, num_classes, 'graph_multilabel'
    except ImportError:
        print("OGB not installed. Install with: pip install ogb")
        return None, None, None, None


# ============================================================
# Candidate pool construction
# ============================================================

def build_candidate_pool(edge_index, num_nodes, k_2hop=4, num_random=0):
    """Build candidate edge pool: original edges + k_2hop 2-hop neighbors + random."""
    src, dst = edge_index[0], edge_index[1]
    
    # Build adjacency list
    adj = [set() for _ in range(num_nodes)]
    for i in range(src.size(0)):
        s, d = src[i].item(), dst[i].item()
        adj[s].add(d)
    
    edge_set = set()
    # Add original edges
    for i in range(src.size(0)):
        edge_set.add((src[i].item(), dst[i].item()))
    
    # Add 2-hop neighbors (sample k_2hop per node)
    if k_2hop > 0:
        import random
        for node in range(num_nodes):
            two_hop = set()
            for nb in adj[node]:
                for nb2 in adj[nb]:
                    if nb2 != node and nb2 not in adj[node]:
                        two_hop.add(nb2)
            two_hop = list(two_hop)
            if len(two_hop) > k_2hop:
                two_hop = random.sample(two_hop, k_2hop)
            for t in two_hop:
                edge_set.add((node, t))
    
    # Add random edges
    if num_random > 0:
        import random
        for node in range(num_nodes):
            for _ in range(num_random):
                t = random.randint(0, num_nodes - 1)
                if t != node:
                    edge_set.add((node, t))
    
    edges = list(edge_set)
    if len(edges) == 0:
        return edge_index.clone()
    
    cand_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return cand_edge_index


def make_orig_edge_mask(cand_edge_index, orig_edge_index):
    """Create boolean mask: which candidate edges are in the original graph."""
    oe = orig_edge_index.cpu()
    orig_set = set()
    for i in range(oe.size(1)):
        orig_set.add((oe[0, i].item(), oe[1, i].item()))
    
    ce = cand_edge_index.cpu()
    mask = torch.zeros(ce.size(1), dtype=torch.bool)
    for i in range(ce.size(1)):
        if (ce[0, i].item(), ce[1, i].item()) in orig_set:
            mask[i] = True
    
    return mask


# ============================================================
# Training functions
# ============================================================

def train_node_classification(args, data, num_features, num_classes, device, seed, 
                               task_type='node'):
    """Train HGODE for node classification."""
    from model_final import HGODEModel
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Setup masks
    if not hasattr(data, 'train_mask') or data.train_mask is None:
        N = data.num_nodes
        perm = torch.randperm(N, device=device)
        n_train = int(0.6 * N)
        n_val = int(0.2 * N)
        data.train_mask = torch.zeros(N, dtype=torch.bool, device=device)
        data.val_mask = torch.zeros(N, dtype=torch.bool, device=device)
        data.test_mask = torch.zeros(N, dtype=torch.bool, device=device)
        data.train_mask[perm[:n_train]] = True
        data.val_mask[perm[n_train:n_train+n_val]] = True
        data.test_mask[perm[n_train+n_val:]] = True
    
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    
    # Build candidate pool
    cand_edge_index = build_candidate_pool(
        data.edge_index.cpu(), data.num_nodes,
        k_2hop=args.k_2hop, num_random=args.num_random
    ).to(device)
    
    orig_mask = make_orig_edge_mask(cand_edge_index, data.edge_index).to(device)
    
    num_orig = orig_mask.sum().item()
    num_cand = cand_edge_index.size(1)
    print(f"  Candidate edges: {num_cand} (orig: {num_orig}, new: {num_cand - num_orig})")
    
    model = HGODEModel(
        in_dim=num_features,
        hidden_dim=args.hidden_dim,
        out_dim=num_classes,
        cand_edge_index=cand_edge_index,
        num_nodes=data.num_nodes,
        orig_edge_mask=orig_mask,
        task='node',
        lam=args.lam,
        gate_tau=args.gate_tau,
        tau_feat=args.tau_feat,
        tau_topo=args.tau_topo,
        gamma=args.gamma,
        force_scale=args.force_scale,
        force_hidden=args.force_hidden,
        beta=args.beta,
        delta=args.delta,
        T=args.T,
        solver=args.solver,
        rtol=args.rtol,
        atol=args.atol,
        dropout=args.dropout,
        no_hysteresis=args.no_hysteresis,
        no_topo_search=args.no_topo_search,
        use_adjoint=args.use_adjoint,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    if args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    else:
        scheduler = None
    
    best_val = -1e9
    best_test = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        logits, H_T = model(data.x, return_H=True)
        
        if task_type == 'node_multilabel':
            # Multi-label BCE
            ce_loss = F.binary_cross_entropy_with_logits(
                logits[train_mask], data.y[train_mask])
        else:
            ce_loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        
        if args.beta > 0 and task_type == 'node':
            margin_loss = model.compute_margin_loss(H_T, data.y, mask=train_mask)
            loss = ce_loss + args.beta * margin_loss
        else:
            loss = ce_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        # Evaluate
        if epoch % args.eval_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                logits = model(data.x)
                
                if task_type == 'node_multilabel':
                    # ROC-AUC for ogbn-proteins
                    from sklearn.metrics import roc_auc_score
                    y_true = data.y[val_mask].cpu().numpy()
                    y_pred = torch.sigmoid(logits[val_mask]).cpu().numpy()
                    try:
                        val_score = roc_auc_score(y_true, y_pred, average='micro')
                    except:
                        val_score = 0.0
                    
                    y_true_test = data.y[test_mask].cpu().numpy()
                    y_pred_test = torch.sigmoid(logits[test_mask]).cpu().numpy()
                    try:
                        test_score = roc_auc_score(y_true_test, y_pred_test, average='micro')
                    except:
                        test_score = 0.0
                else:
                    pred = logits.argmax(dim=-1)
                    val_score = (pred[val_mask] == data.y[val_mask]).float().mean().item()
                    test_score = (pred[test_mask] == data.y[test_mask]).float().mean().item()
            
            if val_score > best_val:
                best_val = val_score
                best_test = test_score
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 50 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}: loss={loss.item():.4f} "
                      f"val={val_score:.4f} test={test_score:.4f} "
                      f"best_test={best_test:.4f}")
            
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break
    
    return best_val, best_test


def train_graph_task(args, datasets, num_features, num_classes, device, seed, task_type):
    """Train HGODE for graph-level tasks (ZINC, Peptides-func, ogbg-molpcba)."""
    from model_final import HGODEModel, ForceFieldMLP, CoupledODEFunc
    from torch_geometric.loader import DataLoader
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    train_dataset, val_dataset, test_dataset = datasets
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # For graph tasks, we need a per-batch model approach
    # Use a simpler architecture: encoder -> GNN-ODE -> readout -> decoder
    model = GraphHGODE(
        in_dim=num_features,
        hidden_dim=args.hidden_dim,
        out_dim=num_classes,
        task_type=task_type,
        lam=args.lam,
        gate_tau=args.gate_tau,
        tau_feat=args.tau_feat,
        tau_topo=args.tau_topo,
        gamma=args.gamma,
        force_scale=args.force_scale,
        force_hidden=args.force_hidden,
        T=args.T,
        solver=args.solver,
        dropout=args.dropout,
        no_hysteresis=args.no_hysteresis,
        no_topo_search=args.no_topo_search,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    if args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    else:
        scheduler = None
    
    best_val = -1e9 if task_type != 'graph_regression' else 1e9
    best_test = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            logits = model(batch_data)
            
            if task_type == 'graph_regression':
                loss = F.l1_loss(logits.squeeze(), batch_data.y.float())
            elif task_type == 'graph_multilabel':
                # Handle NaN in targets (ogbg-molpcba)
                target = batch_data.y.float()
                mask = ~torch.isnan(target)
                if mask.any():
                    loss = F.binary_cross_entropy_with_logits(
                        logits[mask], target[mask])
                else:
                    continue
            else:
                loss = F.cross_entropy(logits, batch_data.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if scheduler:
            scheduler.step()
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Evaluate
        if epoch % args.eval_every == 0 or epoch == 1:
            val_score = evaluate_graph(model, val_loader, device, task_type)
            test_score = evaluate_graph(model, test_loader, device, task_type)
            
            if task_type == 'graph_regression':
                improved = val_score < best_val
            else:
                improved = val_score > best_val
            
            if improved:
                best_val = val_score
                best_test = test_score
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 20 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f} "
                      f"val={val_score:.4f} test={test_score:.4f} "
                      f"best_test={best_test:.4f}")
            
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break
    
    return best_val, best_test


def evaluate_graph(model, loader, device, task_type):
    """Evaluate graph-level model."""
    model.eval()
    
    if task_type == 'graph_regression':
        total_error = 0
        total_count = 0
        with torch.no_grad():
            for batch_data in loader:
                batch_data = batch_data.to(device)
                pred = model(batch_data).squeeze()
                total_error += F.l1_loss(pred, batch_data.y.float(), reduction='sum').item()
                total_count += batch_data.y.size(0)
        return total_error / total_count  # MAE (lower is better)
    
    elif task_type == 'graph_multilabel':
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_data in loader:
                batch_data = batch_data.to(device)
                pred = torch.sigmoid(model(batch_data))
                all_preds.append(pred.cpu())
                all_targets.append(batch_data.y.float().cpu())
        
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        from sklearn.metrics import average_precision_score
        # Handle NaN
        mask = ~np.isnan(all_targets)
        if mask.any():
            try:
                # Per-task AP then average
                aps = []
                for i in range(all_targets.shape[1]):
                    task_mask = mask[:, i]
                    if task_mask.sum() > 0 and len(np.unique(all_targets[task_mask, i])) > 1:
                        ap = average_precision_score(all_targets[task_mask, i], all_preds[task_mask, i])
                        aps.append(ap)
                return np.mean(aps) if aps else 0.0
            except:
                return 0.0
        return 0.0
    
    else:
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data in loader:
                batch_data = batch_data.to(device)
                pred = model(batch_data).argmax(dim=-1)
                correct += (pred == batch_data.y).sum().item()
                total += batch_data.y.size(0)
        return correct / total


class GraphHGODE(nn.Module):
    """
    HGODE for graph-level tasks.
    Uses per-graph ODE integration with batch processing.
    For efficiency, uses fixed-step solver and processes batches together.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, task_type='graph_regression',
                 lam=0.3, gate_tau=0.2, tau_feat=1.0, tau_topo=1.0, gamma=0.5,
                 force_scale=1.0, force_hidden=64,
                 T=0.6, solver='euler', dropout=0.5,
                 no_hysteresis=False, no_topo_search=False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.T = T
        self.solver = solver
        self.lam = lam
        self.gate_tau = gate_tau
        self.tau_feat = tau_feat
        self.tau_topo = tau_topo
        self.gamma = gamma
        self.no_hysteresis = no_hysteresis
        self.no_topo_search = no_topo_search
        self.task_type = task_type
        
        self.u_stable = math.sqrt(1.0 - lam) if lam < 1.0 else 0.1
        
        # Encoder - handle integer features (ZINC) vs float
        self.atom_encoder = nn.Embedding(28, hidden_dim)  # for ZINC
        self.feat_encoder = nn.Linear(in_dim, hidden_dim)  # for others
        self.in_dim = in_dim
        
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
            # Integer features (ZINC)
            if x.dim() == 2:
                x = x.squeeze(-1)
            H = self.atom_encoder(x.long())
        else:
            H = self.feat_encoder(x.float())
        
        # Simple ODE integration using fixed steps
        src, dst = edge_index[0], edge_index[1]
        E = src.size(0)
        
        # Initialize U
        U = torch.full((E,), self.u_stable, device=H.device)
        
        # Fixed-step integration
        num_steps = 10
        dt = self.T / num_steps
        
        for step in range(num_steps):
            # Effective adjacency
            A_eff = torch.sigmoid(U / self.gate_tau)
            
            if self.training and self.dropout > 0:
                drop_mask = torch.bernoulli(torch.full_like(A_eff, 1.0 - self.dropout))
                A_eff = A_eff * drop_mask / (1.0 - self.dropout + 1e-10)
            
            # Row-normalized diffusion
            deg = scatter_add(A_eff, dst, dim=0, dim_size=N) + 1e-10
            norm_weight = A_eff / deg[dst]
            
            msg = H[src] * norm_weight.unsqueeze(-1)
            PH = scatter_add(msg, dst, dim=0, dim_size=N)
            
            G = PH - H
            dH = (1.0 / self.tau_feat) * (G - self.gamma * H)
            
            # Topology dynamics
            if not self.no_topo_search:
                F_force = self.force_field(H[src], H[dst])
                if self.no_hysteresis:
                    dU = (1.0 / self.tau_topo) * (-U + F_force)
                else:
                    dU = (1.0 / self.tau_topo) * ((1.0 - self.lam) * U - U.pow(3) + F_force)
                U = U + dt * dU
            
            H = H + dt * dH
        
        # Readout
        if batch is not None:
            from torch_scatter import scatter_mean
            H_graph = scatter_mean(H, batch, dim=0)
        else:
            H_graph = H.mean(dim=0, keepdim=True)
        
        return self.decoder(H_graph)


# ============================================================
# Main
# ============================================================

def get_default_args(dataset):
    """Get default hyperparameters for each dataset based on paper Table 6."""
    defaults = {
        'cora': dict(
            lam=0.3, gate_tau=0.2, tau_feat=1.0, tau_topo=1.0, gamma=0.5,
            force_scale=1.0, delta=0.1, beta=0.1,
            hidden_dim=128, dropout=0.5, lr=5e-4, wd=5e-4,
            T=0.6, solver='dopri5', k_2hop=4, num_random=0,
            epochs=500, patience=100, batch_size=0,
        ),
        'chameleon': dict(
            lam=0.5, gate_tau=0.1, tau_feat=1.0, tau_topo=1.0, gamma=0.5,
            force_scale=1.5, delta=0.2, beta=0.3,
            hidden_dim=128, dropout=0.5, lr=5e-4, wd=5e-4,
            T=0.6, solver='dopri5', k_2hop=4, num_random=0,
            epochs=500, patience=100, batch_size=0,
        ),
        'ogbn-proteins': dict(
            lam=0.5, gate_tau=0.1, tau_feat=1.0, tau_topo=0.5, gamma=0.5,
            force_scale=1.5, delta=0.2, beta=0.1,
            hidden_dim=128, dropout=0.2, lr=1e-3, wd=0,
            T=0.6, solver='euler', k_2hop=0, num_random=0,
            epochs=200, patience=50, batch_size=0,
        ),
        'zinc': dict(
            lam=0.3, gate_tau=0.2, tau_feat=1.0, tau_topo=1.0, gamma=0.5,
            force_scale=1.0, delta=0.1, beta=0.0,
            hidden_dim=128, dropout=0.2, lr=1e-3, wd=0,
            T=0.3, solver='euler', k_2hop=0, num_random=0,
            epochs=300, patience=50, batch_size=128,
        ),
        'peptides-func': dict(
            lam=0.5, gate_tau=0.1, tau_feat=1.0, tau_topo=1.0, gamma=0.5,
            force_scale=1.5, delta=0.2, beta=0.1,
            hidden_dim=128, dropout=0.2, lr=1e-3, wd=0,
            T=0.6, solver='euler', k_2hop=0, num_random=0,
            epochs=200, patience=50, batch_size=128,
        ),
        'ogbg-molpcba': dict(
            lam=0.3, gate_tau=0.2, tau_feat=1.0, tau_topo=1.0, gamma=0.5,
            force_scale=1.0, delta=0.1, beta=0.1,
            hidden_dim=128, dropout=0.2, lr=1e-3, wd=0,
            T=0.3, solver='euler', k_2hop=0, num_random=0,
            epochs=100, patience=30, batch_size=256,
        ),
    }
    return defaults.get(dataset, defaults['cora'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cora', 'chameleon', 'ogbn-proteins', 'zinc', 
                               'peptides-func', 'ogbg-molpcba'])
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=5)
    
    # Override defaults
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--force_hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--wd', type=float, default=None)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--lr_schedule', type=str, default='cosine')
    parser.add_argument('--lam', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--gate_tau', type=float, default=None)
    parser.add_argument('--tau_feat', type=float, default=None)
    parser.add_argument('--tau_topo', type=float, default=None)
    parser.add_argument('--force_scale', type=float, default=None)
    parser.add_argument('--T', type=float, default=None)
    parser.add_argument('--solver', type=str, default=None)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--delta', type=float, default=None)
    parser.add_argument('--use_adjoint', action='store_true')
    parser.add_argument('--k_2hop', type=int, default=None)
    parser.add_argument('--num_random', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    
    # Ablations
    parser.add_argument('--no_hysteresis', action='store_true')
    parser.add_argument('--no_topo_search', action='store_true')
    parser.add_argument('--no_force_margin', action='store_true')
    
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    # Apply defaults
    defaults = get_default_args(args.dataset)
    for key, val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)
    
    if args.no_force_margin:
        args.beta = 0.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Config: {vars(args)}")
    
    # Load dataset
    loaders = {
        'cora': load_cora,
        'chameleon': load_chameleon,
        'ogbn-proteins': load_ogbn_proteins,
        'zinc': load_zinc,
        'peptides-func': load_peptides_func,
        'ogbg-molpcba': load_ogbg_molpcba,
    }
    
    data, num_features, num_classes, task_type = loaders[args.dataset](device)
    
    if data is None:
        print(f"Failed to load {args.dataset}")
        return
    
    print(f"Task type: {task_type}, Features: {num_features}, Classes/Targets: {num_classes}")
    
    results = []
    for run in range(args.num_runs):
        print(f"\n=== Run {run+1}/{args.num_runs} ===")
        seed = run * 42 + 1
        
        if task_type in ('node', 'node_multilabel'):
            best_val, best_test = train_node_classification(
                args, data, num_features, num_classes, device, seed, task_type)
        else:
            best_val, best_test = train_graph_task(
                args, data, num_features, num_classes, device, seed, task_type)
        
        results.append(best_test)
        print(f"Run {run+1}: val={best_val:.4f}, test={best_test:.4f}")
    
    mean_score = np.mean(results)
    std_score = np.std(results)
    
    # Format based on task
    if task_type == 'node':
        print(f"\n{args.dataset.upper()} Final: {mean_score*100:.2f} ± {std_score*100:.2f} (Accuracy)")
    elif task_type == 'graph_regression':
        print(f"\n{args.dataset.upper()} Final: {mean_score:.4f} ± {std_score:.4f} (MAE)")
    else:
        print(f"\n{args.dataset.upper()} Final: {mean_score:.4f} ± {std_score:.4f}")
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        result_dict = {
            'dataset': args.dataset,
            'task_type': task_type,
            'results': [float(r) for r in results],
            'mean': float(mean_score),
            'std': float(std_score),
            'args': {k: v for k, v in vars(args).items() if not callable(v)},
        }
        with open(args.output, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
