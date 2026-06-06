"""
Training script for HGODE on node classification tasks (Cora, Chameleon, ogbn-proteins).
"""

import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from torch_scatter import scatter_add
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from model import HGODE
from candidate_pool import build_candidate_pool
import math


def load_cora(data_dir='./data'):
    dataset = Planetoid(root=data_dir, name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    return data, dataset.num_features, dataset.num_classes


def load_chameleon(data_dir='./data'):
    dataset = WikipediaNetwork(root=data_dir, name='chameleon', transform=NormalizeFeatures())
    data = dataset[0]
    num_classes = data.y.max().item() + 1
    
    # Use random splits: 60/20/20
    N = data.num_nodes
    perm = torch.randperm(N)
    train_size = int(0.6 * N)
    val_size = int(0.2 * N)
    
    data.train_mask = torch.zeros(N, dtype=torch.bool)
    data.val_mask = torch.zeros(N, dtype=torch.bool)
    data.test_mask = torch.zeros(N, dtype=torch.bool)
    data.train_mask[perm[:train_size]] = True
    data.val_mask[perm[train_size:train_size + val_size]] = True
    data.test_mask[perm[train_size + val_size:]] = True
    
    return data, dataset.num_features, num_classes


def load_ogbn_proteins(data_dir='./data'):
    dataset = PygNodePropPredDataset(name='ogbn-proteins', root=data_dir)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    
    # ogbn-proteins: multi-label (112 tasks); features are edge features
    # Need to create node features from edge features
    # Standard approach: aggregate edge features to nodes
    row, col = data.edge_index
    edge_feat = data.edge_attr  # [E, 8]
    
    # Average edge features per destination node
    node_feat = scatter_add(edge_feat, col, dim=0, dim_size=data.num_nodes)
    deg = scatter_add(torch.ones(row.size(0), device=row.device), col, dim=0, dim_size=data.num_nodes).unsqueeze(-1)
    node_feat = node_feat / (deg + 1e-10)  # [N, 8]
    data.x = node_feat
    
    # Masks
    N = data.num_nodes
    data.train_mask = torch.zeros(N, dtype=torch.bool)
    data.val_mask = torch.zeros(N, dtype=torch.bool)
    data.test_mask = torch.zeros(N, dtype=torch.bool)
    data.train_mask[split_idx['train']] = True
    data.val_mask[split_idx['valid']] = True
    data.test_mask[split_idx['test']] = True
    
    num_classes = data.y.shape[1]  # 112 binary tasks
    
    return data, data.x.shape[1], num_classes


def train_epoch(model, data, optimizer, args, device):
    model.train()
    optimizer.zero_grad()
    
    logits, H_T = model(data.x.to(device), return_margin_data=True)
    
    # Task loss
    if args.dataset == 'ogbn-proteins':
        # Multi-label BCE
        loss_task = F.binary_cross_entropy_with_logits(
            logits[data.train_mask], data.y[data.train_mask].float().to(device)
        )
    else:
        loss_task = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask].to(device))
    
    # Margin loss
    loss_margin = torch.tensor(0.0, device=device)
    if args.beta > 0 and args.dataset != 'ogbn-proteins':
        loss_margin = model.compute_margin_loss(
            H_T, data.y.to(device), mask=data.train_mask.to(device)
        )
    
    loss = loss_task + args.beta * loss_margin
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
    optimizer.step()
    
    return loss.item(), loss_task.item(), loss_margin.item()


@torch.no_grad()
def evaluate(model, data, device, dataset_name='cora'):
    model.eval()
    logits = model(data.x.to(device))
    
    results = {}
    for split, mask in [('train', data.train_mask), ('val', data.val_mask), ('test', data.test_mask)]:
        if dataset_name == 'ogbn-proteins':
            y_pred = logits[mask].sigmoid()
            y_true = data.y[mask].float().to(device)
            # ROC-AUC
            from sklearn.metrics import roc_auc_score
            try:
                score = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
            except:
                score = 0.0
            results[split] = score
        else:
            pred = logits[mask].argmax(dim=-1)
            correct = (pred == data.y[mask].to(device)).sum().item()
            total = mask.sum().item()
            results[split] = correct / total
    
    return results


def run_experiment(args, seed):
    """Run single experiment with given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    if args.dataset == 'cora':
        data, in_dim, num_classes = load_cora(args.data_dir)
    elif args.dataset == 'chameleon':
        data, in_dim, num_classes = load_chameleon(args.data_dir)
    elif args.dataset == 'ogbn-proteins':
        data, in_dim, num_classes = load_ogbn_proteins(args.data_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    data = data.to(device)
    
    # Build candidate pool
    cand_edge_index = build_candidate_pool(
        data.edge_index, data.num_nodes,
        k_2hop=args.k_2hop,
        random_ratio=args.random_ratio
    ).to(device)
    
    print(f"[Seed {seed}] Nodes: {data.num_nodes}, "
          f"Original edges: {data.edge_index.size(1)}, "
          f"Candidate edges: {cand_edge_index.size(1)}")
    
    # Build model
    model = HGODE(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=num_classes,
        cand_edge_index=cand_edge_index,
        num_nodes=data.num_nodes,
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
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Seed {seed}] Model params: {num_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_val = -float('inf') if args.dataset != 'zinc' else float('inf')
    best_test = 0.0 
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, loss_task, loss_margin = train_epoch(model, data, optimizer, args, device)
        scheduler.step()
        
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            results = evaluate(model, data, device, args.dataset)
            
            val_metric = results['val']
            test_metric = results['test']
            
            improved = val_metric > best_val
            if improved:
                best_val = val_metric
                best_test = test_metric
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % (args.eval_every * 5) == 0 or epoch <= 5:
                elapsed = time.time() - t0
                print(f"[Seed {seed}] Epoch {epoch}: loss={loss:.4f} (task={loss_task:.4f}, margin={loss_margin:.4f}) "
                      f"| val={results['val']:.4f} test={results['test']:.4f} "
                      f"| best_val={best_val:.4f} best_test={best_test:.4f} "
                      f"| time={elapsed:.1f}s")
            
            if patience_counter >= args.patience and epoch >= args.min_epochs:
                print(f"[Seed {seed}] Early stopping at epoch {epoch}")
                break
    
    print(f"[Seed {seed}] Final: best_val={best_val:.4f}, best_test={best_test:.4f}")
    return best_test


def get_default_args(dataset):
    """Get default hyperparameters for each dataset based on paper Table 6."""
    defaults = {
        'cora': dict(
            hidden_dim=256, lr=5e-4, weight_decay=5e-4,
            lam=0.2, gate_tau=0.25, force_scale=1.0, delta=0.1, beta=0.1,
            tau_feat=1.0, tau_topo=1.0, gamma=0.5, T=0.6,
            k_2hop=4, random_ratio=0.001,
            epochs=300, patience=50, min_epochs=100, dropout=0.5,
        ),
        'chameleon': dict(
            hidden_dim=256, lr=5e-4, weight_decay=5e-4,
            lam=0.5, gate_tau=0.08, force_scale=1.5, delta=0.25, beta=0.4,
            tau_feat=1.0, tau_topo=1.0, gamma=0.5, T=0.6,
            k_2hop=8, random_ratio=0.005,
            epochs=300, patience=50, min_epochs=100, dropout=0.5,
        ),
        'ogbn-proteins': dict(
            hidden_dim=256, lr=1e-3, weight_decay=0,
            lam=0.6, gate_tau=0.08, force_scale=1.5, delta=0.25, beta=0.2,
            tau_feat=1.0, tau_topo=0.5, gamma=0.5, T=0.6,
            k_2hop=4, random_ratio=0.001,
            epochs=200, patience=30, min_epochs=50, dropout=0.2,
        ),
    }
    return defaults.get(dataset, defaults['cora'])


def main():
    parser = argparse.ArgumentParser(description='HGODE Node Classification')
    parser.add_argument('--dataset', type=str, default='cora', 
                       choices=['cora', 'chameleon', 'ogbn-proteins'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--solver', type=str, default='dopri5')
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--force_hidden', type=int, default=64)
    
    # Allow overriding defaults
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--lam', type=float, default=None)
    parser.add_argument('--gate_tau', type=float, default=None)
    parser.add_argument('--force_scale', type=float, default=None)
    parser.add_argument('--delta', type=float, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--tau_feat', type=float, default=None)
    parser.add_argument('--tau_topo', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--T', type=float, default=None)
    parser.add_argument('--k_2hop', type=int, default=None)
    parser.add_argument('--random_ratio', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--min_epochs', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    
    # Ablation flags
    parser.add_argument('--no_hysteresis', action='store_true',
                       help='Remove cubic term (ablation)')
    parser.add_argument('--no_topo_search', action='store_true', 
                       help='Only use observed edges (ablation)')
    parser.add_argument('--no_force_margin', action='store_true',
                       help='Set beta=0 (ablation)')
    
    parser.add_argument('--output_dir', type=str, default='./results')
    
    args = parser.parse_args()
    
    # Fill in defaults
    defaults = get_default_args(args.dataset)
    for key, val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)
    
    # Apply ablation flags
    if args.no_force_margin:
        args.beta = 0.0
    if args.no_topo_search:
        args.k_2hop = 0
        args.random_ratio = 0.0
    
    print(f"=== HGODE Node Classification on {args.dataset} ===")
    print(f"Config: {vars(args)}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    test_scores = []
    for seed in range(args.num_seeds):
        score = run_experiment(args, seed=seed + 42)
        test_scores.append(score)
    
    mean_score = np.mean(test_scores)
    std_score = np.std(test_scores)
    
    print(f"\n=== Final Results ({args.dataset}) ===")
    print(f"Test scores: {test_scores}")
    print(f"Mean ± Std: {mean_score:.4f} ± {std_score:.4f}")
    
    # Save results
    result = {
        'dataset': args.dataset,
        'test_scores': test_scores,
        'mean': float(mean_score),
        'std': float(std_score),
        'config': {k: v for k, v in vars(args).items() if not k.startswith('_')},
    }
    
    suffix = ''
    if args.no_hysteresis:
        suffix = '_no_hysteresis'
    elif args.no_topo_search:
        suffix = '_no_topo_search'  
    elif args.no_force_margin:
        suffix = '_no_force_margin'
    
    result_file = os.path.join(args.output_dir, f'{args.dataset}{suffix}.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_file}")


if __name__ == '__main__':
    main()
