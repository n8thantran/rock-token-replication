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
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from candidate_pool import build_candidate_pool_fast
from model import HGODE


def train_cora(args):
    """Train HGODE on Cora dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
    data = dataset[0].to(device)
    
    results = []
    
    for run in range(args.num_runs):
        torch.manual_seed(run * 42)
        np.random.seed(run * 42)
        
        # Build candidate pool
        cand_edge_index = build_candidate_pool_fast(
            data.edge_index, data.num_nodes, 
            num_hops=2, num_random=0
        ).to(device)
        
        model = HGODE(
            in_dim=dataset.num_features,
            hidden_dim=args.hidden_dim,
            out_dim=dataset.num_classes,
            cand_edge_index=cand_edge_index,
            num_nodes=data.num_nodes,
            orig_edge_index=data.edge_index,
            tau_feat=1.0,
            tau_topo=1.0,
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
            force_hidden=64,
            dropout=args.dropout,
            no_hysteresis=args.no_hysteresis,
            no_topo_search=args.no_topo_search,
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
        best_val, best_test = 0, 0
        patience_counter = 0
        
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            
            out, H_T = model(data.x, return_margin_data=True)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            
            if args.beta > 0:
                margin_loss = model.compute_margin_loss(H_T, data.y, mask=data.train_mask)
                total_loss = loss + args.beta * margin_loss
            else:
                total_loss = loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    out = model(data.x)
                    pred = out.argmax(dim=-1)
                    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
                
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= args.patience:
                    break
        
        results.append(best_test)
        print(f'Run {run+1}/{args.num_runs}: val={best_val:.4f} test={best_test:.4f}')
    
    mean_acc = np.mean(results) * 100
    std_acc = np.std(results) * 100
    print(f'\nCora Results: {mean_acc:.2f} ± {std_acc:.2f}')
    return results


def train_chameleon(args):
    """Train HGODE on Chameleon dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = WikipediaNetwork(root='./data', name='chameleon', transform=NormalizeFeatures())
    data = dataset[0].to(device)
    
    # Chameleon uses random splits
    num_nodes = data.num_nodes
    results = []
    
    for run in range(args.num_runs):
        torch.manual_seed(run * 42)
        np.random.seed(run * 42)
        
        # Random 60/20/20 split
        perm = torch.randperm(num_nodes)
        n_train = int(0.6 * num_nodes)
        n_val = int(0.2 * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:n_train+n_val]] = True
        test_mask[perm[n_train+n_val:]] = True
        
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
        
        # Build candidate pool
        cand_edge_index = build_candidate_pool_fast(
            data.edge_index, data.num_nodes,
            num_hops=2, num_random=0
        ).to(device)
        
        model = HGODE(
            in_dim=dataset.num_features,
            hidden_dim=args.hidden_dim,
            out_dim=dataset.num_classes,
            cand_edge_index=cand_edge_index,
            num_nodes=data.num_nodes,
            orig_edge_index=data.edge_index,
            tau_feat=1.0,
            tau_topo=1.0,
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
            force_hidden=64,
            dropout=args.dropout,
            no_hysteresis=args.no_hysteresis,
            no_topo_search=args.no_topo_search,
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
        best_val, best_test = 0, 0
        patience_counter = 0
        
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    out = model(data.x)
                    pred = out.argmax(dim=-1)
                    val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()
                    test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()
                
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= args.patience:
                    break
        
        results.append(best_test)
        print(f'Run {run+1}/{args.num_runs}: val={best_val:.4f} test={best_test:.4f}')
    
    mean_acc = np.mean(results) * 100
    std_acc = np.std(results) * 100
    print(f'\nChameleon Results: {mean_acc:.2f} ± {std_acc:.2f}')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'chameleon'])
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lam', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--gate_tau', type=float, default=0.2)
    parser.add_argument('--force_scale', type=float, default=1.0)
    parser.add_argument('--T', type=float, default=0.6)
    parser.add_argument('--solver', type=str, default='rk4')
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--atol', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--no_hysteresis', action='store_true')
    parser.add_argument('--no_topo_search', action='store_true')
    parser.add_argument('--no_force_margin', action='store_true')
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.no_force_margin:
        args.beta = 0.0
    
    if args.dataset == 'cora':
        results = train_cora(args)
    elif args.dataset == 'chameleon':
        results = train_chameleon(args)
    
    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        result_dict = {
            'dataset': args.dataset,
            'results': results,
            'mean': float(np.mean(results) * 100),
            'std': float(np.std(results) * 100),
            'args': vars(args),
        }
        with open(args.output, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f'Results saved to {args.output}')
