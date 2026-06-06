"""
Training script for HGODE on node classification datasets.
Supports: Cora, Chameleon
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
from model import HGODE


def train_node_classification(args):
    """Train HGODE on a node classification dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    if args.dataset == 'cora':
        dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
        data = dataset[0].to(device)
    elif args.dataset == 'chameleon':
        dataset = WikipediaNetwork(root='./data', name='chameleon', transform=NormalizeFeatures())
        data = dataset[0].to(device)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"Dataset: {args.dataset}, Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}, "
          f"Features: {dataset.num_features}, Classes: {dataset.num_classes}")
    
    results = []
    
    for run in range(args.num_runs):
        torch.manual_seed(run * 42 + 1)
        np.random.seed(run * 42 + 1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(run * 42 + 1)
        
        # Setup masks
        if args.dataset == 'cora':
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
        elif args.dataset == 'chameleon':
            # Random 60/20/20 split
            num_nodes = data.num_nodes
            perm = torch.randperm(num_nodes)
            n_train = int(0.6 * num_nodes)
            n_val = int(0.2 * num_nodes)
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            train_mask[perm[:n_train]] = True
            val_mask[perm[n_train:n_train+n_val]] = True
            test_mask[perm[n_train+n_val:]] = True
        
        # Build candidate pool
        cand_edge_index = build_candidate_pool_fast(
            data.edge_index, data.num_nodes, 
            num_hops=args.k_2hop, num_random=args.num_random
        ).to(device)
        
        print(f"Run {run+1}: Candidate edges: {cand_edge_index.size(1)}")
        
        model = HGODE(
            in_dim=dataset.num_features,
            hidden_dim=args.hidden_dim,
            out_dim=dataset.num_classes,
            cand_edge_index=cand_edge_index,
            num_nodes=data.num_nodes,
            orig_edge_index=data.edge_index,
            tau_feat=args.tau_feat,
            tau_topo=args.tau_topo,
            lam=args.lam,
            gamma=args.gamma,
            gate_tau=args.gate_tau,
            force_scale=args.force_scale,
            T=args.T,
            solver=args.solver,
            rtol=args.rtol,
            atol=args.atol,
            beta=args.beta,
            delta=args.delta,
            force_hidden=args.force_hidden,
            dropout=args.dropout,
            ode_dropout=args.ode_dropout,
            no_hysteresis=args.no_hysteresis,
            no_topo_search=args.no_topo_search,
            use_adjoint=args.use_adjoint,
        ).to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {num_params:,}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
        # LR scheduler
        if args.lr_schedule == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.lr_schedule == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        else:
            scheduler = None
        
        best_val, best_test = 0, 0
        patience_counter = 0
        
        t_start = time.time()
        
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            
            out, H_T = model(data.x, return_margin_data=True)
            loss = F.cross_entropy(out[train_mask], data.y[train_mask])
            
            if args.beta > 0:
                margin_loss = model.compute_margin_loss(H_T, data.y, mask=train_mask)
                total_loss = loss + args.beta * margin_loss
            else:
                total_loss = loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # Evaluate every eval_every epochs
            if (epoch + 1) % args.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    out = model(data.x)
                    pred = out.argmax(dim=-1)
                    
                    train_acc = (pred[train_mask] == data.y[train_mask]).float().mean().item()
                    val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()
                    test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()
                
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 50 == 0:
                    elapsed = time.time() - t_start
                    nfe = model.ode_func.nfe
                    print(f"  Epoch {epoch+1}: loss={total_loss.item():.4f} "
                          f"train={train_acc:.4f} val={val_acc:.4f} test={test_acc:.4f} "
                          f"best_val={best_val:.4f} best_test={best_test:.4f} "
                          f"nfe={nfe} time={elapsed:.1f}s")
                
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        results.append(best_test)
        print(f'Run {run+1}/{args.num_runs}: val={best_val:.4f} test={best_test:.4f}')
    
    mean_acc = np.mean(results) * 100
    std_acc = np.std(results) * 100
    print(f'\n{args.dataset.upper()} Results: {mean_acc:.2f} ± {std_acc:.2f}')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'chameleon'])
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--force_hidden', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--ode_dropout', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--lr_schedule', type=str, default='none', choices=['none', 'cosine', 'step'])
    
    # HGODE hyperparameters
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
    parser.add_argument('--k_2hop', type=int, default=2, help='0=no multi-hop, 2=2-hop')
    parser.add_argument('--num_random', type=int, default=0)
    
    # Ablations
    parser.add_argument('--no_hysteresis', action='store_true')
    parser.add_argument('--no_topo_search', action='store_true')
    parser.add_argument('--no_force_margin', action='store_true')
    
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.no_force_margin:
        args.beta = 0.0
    
    results = train_node_classification(args)
    
    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        result_dict = {
            'dataset': args.dataset,
            'results': [float(r) for r in results],
            'mean': float(np.mean(results) * 100),
            'std': float(np.std(results) * 100),
            'args': vars(args),
        }
        with open(args.output, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f'Results saved to {args.output}')
