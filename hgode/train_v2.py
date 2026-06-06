"""
Training script v2 for HGODE node classification.
Clean, focused approach.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import math
import argparse
import time
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from candidate_pool import build_candidate_pool_fast
from model_v2 import HGODENodeClassifier


def load_dataset(name, device):
    """Load and prepare a dataset."""
    if name == 'cora':
        dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
        data = dataset[0].to(device)
        return data, dataset.num_features, dataset.num_classes
    elif name == 'chameleon':
        dataset = WikipediaNetwork(root='./data', name='chameleon', transform=NormalizeFeatures())
        data = dataset[0].to(device)
        return data, dataset.num_features, dataset.num_classes
    else:
        raise ValueError(f"Unknown dataset: {name}")


def make_orig_edge_mask(cand_edge_index, orig_edge_index):
    """Create boolean mask: which candidate edges are in the original graph."""
    # Use set of (src, dst) tuples for fast lookup
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


def train_run(args, data, num_features, num_classes, device, seed):
    """Single training run."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Setup masks
    if args.dataset == 'cora':
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    elif args.dataset == 'chameleon':
        N = data.num_nodes
        perm = torch.randperm(N, device=device)
        n_train = int(0.6 * N)
        n_val = int(0.2 * N)
        train_mask = torch.zeros(N, dtype=torch.bool, device=device)
        val_mask = torch.zeros(N, dtype=torch.bool, device=device)
        test_mask = torch.zeros(N, dtype=torch.bool, device=device)
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:n_train+n_val]] = True
        test_mask[perm[n_train+n_val:]] = True
    
    # Build candidate pool
    cand_edge_index = build_candidate_pool_fast(
        data.edge_index, data.num_nodes,
        num_hops=args.k_2hop, num_random=args.num_random
    ).to(device)
    
    orig_mask = make_orig_edge_mask(cand_edge_index, data.edge_index).to(device)
    
    num_orig = orig_mask.sum().item()
    num_cand = cand_edge_index.size(1)
    print(f"  Candidate edges: {num_cand} (orig: {num_orig}, new: {num_cand - num_orig})")
    
    # Build model
    model = HGODENodeClassifier(
        in_dim=num_features,
        hidden_dim=args.hidden_dim,
        out_dim=num_classes,
        cand_edge_index=cand_edge_index,
        num_nodes=data.num_nodes,
        orig_edge_mask=orig_mask,
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    else:
        scheduler = None
    
    best_val = 0.0
    best_test = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        logits, H_T = model(data.x, return_H=True)
        
        ce_loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        
        if args.beta > 0:
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
                pred = logits.argmax(dim=-1)
                
                train_acc = (pred[train_mask] == data.y[train_mask]).float().mean().item()
                val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()
                test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()
            
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 50 == 0 or epoch == 1:
                nfe = model.ode_func.nfe
                print(f"  Epoch {epoch:3d}: loss={loss.item():.4f} "
                      f"train={train_acc:.4f} val={val_acc:.4f} test={test_acc:.4f} "
                      f"best_test={best_test:.4f} nfe_total={nfe}")
            
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break
    
    return best_val, best_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=5)
    
    # Architecture
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--force_hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # Optimization
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--lr_schedule', type=str, default='none')
    
    # HGODE params
    parser.add_argument('--lam', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--gate_tau', type=float, default=0.2)
    parser.add_argument('--tau_feat', type=float, default=1.0)
    parser.add_argument('--tau_topo', type=float, default=1.0)
    parser.add_argument('--force_scale', type=float, default=1.0)
    parser.add_argument('--T', type=float, default=0.6)
    parser.add_argument('--solver', type=str, default='dopri5')
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--use_adjoint', action='store_true')
    
    # Candidate pool
    parser.add_argument('--k_2hop', type=int, default=2)
    parser.add_argument('--num_random', type=int, default=0)
    
    # Ablations
    parser.add_argument('--no_hysteresis', action='store_true')
    parser.add_argument('--no_topo_search', action='store_true')
    parser.add_argument('--no_force_margin', action='store_true')
    
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.no_force_margin:
        args.beta = 0.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: {vars(args)}")
    
    data, num_features, num_classes = load_dataset(args.dataset, device)
    print(f"Dataset: {args.dataset}, Nodes: {data.num_nodes}, "
          f"Edges: {data.edge_index.size(1)}, Features: {num_features}, Classes: {num_classes}")
    
    results = []
    for run in range(args.num_runs):
        print(f"\n=== Run {run+1}/{args.num_runs} ===")
        seed = run * 42 + 1
        best_val, best_test = train_run(args, data, num_features, num_classes, device, seed)
        results.append(best_test)
        print(f"Run {run+1}: val={best_val:.4f}, test={best_test:.4f}")
    
    mean_acc = np.mean(results) * 100
    std_acc = np.std(results) * 100
    print(f"\n{args.dataset.upper()} Final: {mean_acc:.2f} ± {std_acc:.2f}")
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({
                'dataset': args.dataset,
                'results': [float(r) for r in results],
                'mean': float(mean_acc),
                'std': float(std_acc),
                'args': vars(args),
            }, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
