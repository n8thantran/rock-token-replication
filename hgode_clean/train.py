"""
Training script for HGODE on all 6 datasets.
Supports ablations: --no_hysteresis, --no_topo_search, --no_force_margin
"""

import argparse
import json
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import HGODE, HGODEGraph


def build_candidate_pool(edge_index, num_nodes, k_2hop=4, random_ratio=0.001):
    """Build candidate edge pool: observed + 2-hop + random."""
    device = edge_index.device
    
    # Start with observed edges
    observed_set = set()
    adj = {}
    for i in range(edge_index.shape[1]):
        s, d = edge_index[0, i].item(), edge_index[1, i].item()
        observed_set.add((s, d))
        if s not in adj:
            adj[s] = []
        adj[s].append(d)
    
    # 2-hop neighbors (limited)
    twohop_edges = set()
    if k_2hop > 0:
        for node in range(num_nodes):
            if node not in adj:
                continue
            neighbors = adj[node]
            twohop = set()
            for n1 in neighbors:
                if n1 in adj:
                    for n2 in adj[n1]:
                        if n2 != node and (node, n2) not in observed_set:
                            twohop.add(n2)
            # Sample k_2hop
            twohop = list(twohop)
            if len(twohop) > k_2hop:
                indices = np.random.choice(len(twohop), k_2hop, replace=False)
                twohop = [twohop[i] for i in indices]
            for n2 in twohop:
                twohop_edges.add((node, n2))
    
    # Random edges
    num_random = int(random_ratio * num_nodes * num_nodes)
    random_edges = set()
    if num_random > 0:
        for _ in range(num_random):
            s = np.random.randint(0, num_nodes)
            d = np.random.randint(0, num_nodes)
            if s != d and (s, d) not in observed_set:
                random_edges.add((s, d))
    
    # Combine
    all_edges = list(observed_set) + list(twohop_edges) + list(random_edges)
    # Remove duplicates
    all_edges = list(set(all_edges))
    
    cand_edge_index = torch.tensor(all_edges, dtype=torch.long, device=device).t()
    
    # Mark which are observed
    is_observed = torch.zeros(len(all_edges), dtype=torch.bool, device=device)
    for i, (s, d) in enumerate(all_edges):
        if (s, d) in observed_set:
            is_observed[i] = True
    
    print(f"Candidate pool: {len(all_edges)} edges ({is_observed.sum().item()} observed, "
          f"{len(twohop_edges)} 2-hop, {len(random_edges)} random)")
    
    return cand_edge_index, is_observed


def train_cora(args):
    """Train on Cora dataset."""
    from torch_geometric.datasets import Planetoid
    
    dataset = Planetoid(root=args.data_dir, name='Cora')
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Build candidate pool
    cand_edge_index, is_observed = build_candidate_pool(
        data.edge_index, data.num_nodes,
        k_2hop=args.k_2hop, random_ratio=args.random_ratio
    )
    
    model = HGODE(
        in_dim=data.num_features,
        hidden_dim=args.hidden_dim,
        out_dim=dataset.num_classes,
        cand_edge_index=cand_edge_index,
        is_observed=is_observed,
        lam=args.lam, tau=args.tau,
        tau_feat=args.tau_feat, tau_topo=args.tau_topo,
        gamma=args.gamma, s=args.s,
        delta=args.delta, beta=args.beta,
        T=args.T, dropout=args.dropout,
        use_hysteresis=not args.no_hysteresis,
        use_topo_search=not args.no_topo_search,
        use_force_margin=not args.no_force_margin,
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        logits, force_vals, U_final = model(data.x, num_steps=args.num_steps)
        
        # Task loss
        loss_task = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        
        # Margin loss
        loss_margin = model.margin_loss(force_vals, data.y)
        
        loss = loss_task + loss_margin
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        
        # Evaluate
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                logits, _, _ = model(data.x, num_steps=args.num_steps)
                pred = logits.argmax(dim=-1)
                
                train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch}: loss={loss.item():.4f} train={train_acc:.4f} "
                  f"val={val_acc:.4f} test={test_acc:.4f} best_test={best_test_acc:.4f}")
            
            if patience_counter > args.patience // 10:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return best_test_acc


def train_chameleon(args):
    """Train on Chameleon dataset."""
    from torch_geometric.datasets import WikipediaNetwork
    
    dataset = WikipediaNetwork(root=args.data_dir, name='chameleon')
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Use random splits (60/20/20)
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size+val_size]] = True
    test_mask[perm[train_size+val_size:]] = True
    
    data.train_mask = train_mask.to(device)
    data.val_mask = val_mask.to(device)
    data.test_mask = test_mask.to(device)
    
    cand_edge_index, is_observed = build_candidate_pool(
        data.edge_index, data.num_nodes,
        k_2hop=args.k_2hop, random_ratio=args.random_ratio
    )
    
    num_classes = data.y.max().item() + 1
    
    model = HGODE(
        in_dim=data.num_features,
        hidden_dim=args.hidden_dim,
        out_dim=num_classes,
        cand_edge_index=cand_edge_index,
        is_observed=is_observed,
        lam=args.lam, tau=args.tau,
        tau_feat=args.tau_feat, tau_topo=args.tau_topo,
        gamma=args.gamma, s=args.s,
        delta=args.delta, beta=args.beta,
        T=args.T, dropout=args.dropout,
        use_hysteresis=not args.no_hysteresis,
        use_topo_search=not args.no_topo_search,
        use_force_margin=not args.no_force_margin,
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        logits, force_vals, U_final = model(data.x, num_steps=args.num_steps)
        loss_task = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss_margin = model.margin_loss(force_vals, data.y)
        loss = loss_task + loss_margin
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                logits, _, _ = model(data.x, num_steps=args.num_steps)
                pred = logits.argmax(dim=-1)
                train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch}: loss={loss.item():.4f} train={train_acc:.4f} "
                  f"val={val_acc:.4f} test={test_acc:.4f} best_test={best_test_acc:.4f}")
            
            if patience_counter > args.patience // 10:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return best_test_acc


def train_zinc(args):
    """Train on ZINC dataset (graph regression, MAE)."""
    from torch_geometric.datasets import ZINC
    from torch_geometric.loader import DataLoader
    
    train_dataset = ZINC(root=os.path.join(args.data_dir, 'ZINC'), subset=True, split='train')
    val_dataset = ZINC(root=os.path.join(args.data_dir, 'ZINC'), subset=True, split='val')
    test_dataset = ZINC(root=os.path.join(args.data_dir, 'ZINC'), subset=True, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ZINC has node features as integers (atom types), need embedding
    # x is 1D integer, we use embedding
    model = HGODEGraph(
        in_dim=28,  # ZINC has 28 atom types when using one-hot
        hidden_dim=args.hidden_dim,
        out_dim=1,
        lam=args.lam, tau=args.tau,
        tau_feat=args.tau_feat, tau_topo=args.tau_topo,
        gamma=args.gamma, s=args.s,
        delta=args.delta, beta=args.beta,
        T=args.T, dropout=args.dropout,
        use_hysteresis=not args.no_hysteresis,
        use_force_margin=not args.no_force_margin,
        task='regression',
    ).to(device)
    
    # Replace encoder with embedding + linear
    model.atom_encoder = nn.Embedding(28, args.hidden_dim).to(device)
    model.encoder = nn.Sequential(
        nn.Linear(args.hidden_dim, args.hidden_dim),
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(args.hidden_dim, args.hidden_dim),
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_mae = float('inf')
    best_test_mae = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        count = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Convert integer features to embeddings
            x_emb = model.atom_encoder(batch.x.long().squeeze(-1))
            batch_copy = batch.clone()
            batch_copy.x = x_emb
            
            pred = model(batch_copy, num_steps=args.num_steps)
            loss = F.l1_loss(pred.squeeze(-1), batch.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            count += batch.num_graphs
        
        scheduler.step()
        train_mae = total_loss / count
        
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_mae = evaluate_mae(model, val_loader, device)
            test_mae = evaluate_mae(model, test_loader, device)
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_test_mae = test_mae
            
            print(f"Epoch {epoch}: train_mae={train_mae:.4f} val_mae={val_mae:.4f} "
                  f"test_mae={test_mae:.4f} best_test={best_test_mae:.4f}")
    
    return best_test_mae


def evaluate_mae(model, loader, device):
    """Evaluate MAE on a data loader."""
    model.eval()
    total_error = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_emb = model.atom_encoder(batch.x.long().squeeze(-1))
            batch_copy = batch.clone()
            batch_copy.x = x_emb
            pred = model(batch_copy, num_steps=model.T)  # use num_steps from args
            total_error += F.l1_loss(pred.squeeze(-1), batch.y.float(), reduction='sum').item()
            count += batch.num_graphs
    return total_error / count


def train_peptides(args):
    """Train on Peptides-func (graph classification, AP)."""
    try:
        from torch_geometric.datasets import LRGBDataset
        train_dataset = LRGBDataset(root=os.path.join(args.data_dir, 'LRGB'), name='Peptides-func', split='train')
        val_dataset = LRGBDataset(root=os.path.join(args.data_dir, 'LRGB'), name='Peptides-func', split='val')
        test_dataset = LRGBDataset(root=os.path.join(args.data_dir, 'LRGB'), name='Peptides-func', split='test')
    except:
        print("LRGBDataset not available, skipping Peptides-func")
        return None
    
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Peptides-func: 9 node features, 10 classes (multi-label)
    in_dim = train_dataset[0].x.shape[1]
    out_dim = train_dataset[0].y.shape[1] if len(train_dataset[0].y.shape) > 1 else 10
    
    model = HGODEGraph(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=out_dim,
        lam=args.lam, tau=args.tau,
        tau_feat=args.tau_feat, tau_topo=args.tau_topo,
        gamma=args.gamma, s=args.s,
        delta=args.delta, beta=args.beta,
        T=args.T, dropout=args.dropout,
        use_hysteresis=not args.no_hysteresis,
        use_force_margin=not args.no_force_margin,
        task='classification',
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_ap = 0
    best_test_ap = 0
    
    from sklearn.metrics import average_precision_score
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        count = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred = model(batch, num_steps=args.num_steps)
            target = batch.y.float()
            loss = F.binary_cross_entropy_with_logits(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            count += batch.num_graphs
        
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == 1:
            val_ap = evaluate_ap(model, val_loader, device)
            test_ap = evaluate_ap(model, test_loader, device)
            
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_test_ap = test_ap
            
            print(f"Epoch {epoch}: loss={total_loss/count:.4f} val_ap={val_ap:.4f} "
                  f"test_ap={test_ap:.4f} best_test={best_test_ap:.4f}")
    
    return best_test_ap


def evaluate_ap(model, loader, device):
    """Evaluate Average Precision."""
    from sklearn.metrics import average_precision_score
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch, num_steps=10)
            all_preds.append(torch.sigmoid(pred).cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Per-class AP then average
    aps = []
    for i in range(all_targets.shape[1]):
        if all_targets[:, i].sum() > 0:
            aps.append(average_precision_score(all_targets[:, i], all_preds[:, i]))
    return np.mean(aps) if aps else 0.0


def train_ogbn_proteins(args):
    """Train on ogbn-proteins (node classification, ROC-AUC)."""
    try:
        from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
    except:
        print("OGB not available, skipping ogbn-proteins")
        return None
    
    dataset = PygNodePropPredDataset(name='ogbn-proteins', root=args.data_dir)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-proteins')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ogbn-proteins: node features from edge features
    # Aggregate edge features to node features
    from torch_geometric.utils import scatter
    edge_attr = data.edge_attr  # [E, 8]
    row, col = data.edge_index
    
    # Mean aggregation of edge features to get node features
    node_feat = torch.zeros(data.num_nodes, edge_attr.shape[1])
    deg = torch.zeros(data.num_nodes, dtype=torch.long)
    node_feat.scatter_add_(0, col.unsqueeze(-1).expand_as(edge_attr), edge_attr.float())
    deg.scatter_add_(0, col, torch.ones_like(col))
    deg = deg.clamp(min=1)
    node_feat = node_feat / deg.unsqueeze(-1).float()
    
    data.x = node_feat
    data = data.to(device)
    
    # Build candidate pool (just observed edges for proteins - it's already large)
    cand_edge_index = data.edge_index
    is_observed = torch.ones(cand_edge_index.shape[1], dtype=torch.bool, device=device)
    
    num_tasks = data.y.shape[1]  # 112 binary tasks
    
    model = HGODE(
        in_dim=data.x.shape[1],
        hidden_dim=args.hidden_dim,
        out_dim=num_tasks,
        cand_edge_index=cand_edge_index,
        is_observed=is_observed,
        lam=args.lam, tau=args.tau,
        tau_feat=args.tau_feat, tau_topo=args.tau_topo,
        gamma=args.gamma, s=args.s,
        delta=args.delta, beta=args.beta,
        T=args.T, dropout=args.dropout,
        use_hysteresis=not args.no_hysteresis,
        use_topo_search=not args.no_topo_search,
        use_force_margin=not args.no_force_margin,
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    train_idx = split_idx['train'].to(device)
    val_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)
    
    best_val_auc = 0
    best_test_auc = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        logits, force_vals, U_final = model(data.x, num_steps=args.num_steps)
        
        target = data.y[train_idx].float()
        loss = F.binary_cross_entropy_with_logits(logits[train_idx], target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits, _, _ = model(data.x, num_steps=args.num_steps)
            
            val_result = evaluator.eval({
                'y_true': data.y[val_idx].cpu(),
                'y_pred': logits[val_idx].cpu(),
            })['rocauc']
            
            test_result = evaluator.eval({
                'y_true': data.y[test_idx].cpu(),
                'y_pred': logits[test_idx].cpu(),
            })['rocauc']
            
            if val_result > best_val_auc:
                best_val_auc = val_result
                best_test_auc = test_result
            
            print(f"Epoch {epoch}: loss={loss.item():.4f} val_auc={val_result:.4f} "
                  f"test_auc={test_result:.4f} best_test={best_test_auc:.4f}")
    
    return best_test_auc


def train_ogbg_molpcba(args):
    """Train on ogbg-molpcba (graph classification, AP)."""
    try:
        from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
    except:
        print("OGB not available, skipping ogbg-molpcba")
        return None
    
    from torch_geometric.loader import DataLoader
    
    dataset = PygGraphPropPredDataset(name='ogbg-molpcba', root=args.data_dir)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbg-molpcba')
    
    train_loader = DataLoader(dataset[split_idx['train']], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset[split_idx['valid']], batch_size=args.batch_size)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=args.batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ogbg-molpcba: 9 node features, 128 tasks
    in_dim = dataset[0].x.shape[1]
    out_dim = dataset.num_tasks
    
    model = HGODEGraph(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=out_dim,
        lam=args.lam, tau=args.tau,
        tau_feat=args.tau_feat, tau_topo=args.tau_topo,
        gamma=args.gamma, s=args.s,
        delta=args.delta, beta=args.beta,
        T=args.T, dropout=args.dropout,
        use_hysteresis=not args.no_hysteresis,
        use_force_margin=not args.no_force_margin,
        task='classification',
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_ap = 0
    best_test_ap = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        count = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred = model(batch, num_steps=args.num_steps)
            
            # Handle NaN targets
            target = batch.y.float()
            is_valid = ~torch.isnan(target)
            loss = F.binary_cross_entropy_with_logits(
                pred[is_valid], target[is_valid]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            count += batch.num_graphs
        
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == 1:
            val_ap = evaluate_ogb_ap(model, val_loader, evaluator, device)
            test_ap = evaluate_ogb_ap(model, test_loader, evaluator, device)
            
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_test_ap = test_ap
            
            print(f"Epoch {epoch}: loss={total_loss/count:.4f} val_ap={val_ap:.4f} "
                  f"test_ap={test_ap:.4f} best_test={best_test_ap:.4f}")
    
    return best_test_ap


def evaluate_ogb_ap(model, loader, evaluator, device):
    """Evaluate AP using OGB evaluator."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch, num_steps=10)
            all_preds.append(pred.cpu())
            all_targets.append(batch.y.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    result = evaluator.eval({
        'y_true': all_targets,
        'y_pred': all_preds,
    })
    return result['ap']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cora', 'chameleon', 'zinc', 'peptides', 'ogbn-proteins', 'ogbg-molpcba'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_seeds', type=int, default=3)
    
    # Architecture
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # Hysteresis parameters
    parser.add_argument('--lam', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--tau_feat', type=float, default=1.0)
    parser.add_argument('--tau_topo', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.5)
    
    # Force parameters
    parser.add_argument('--s', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    
    # Candidate pool
    parser.add_argument('--k_2hop', type=int, default=4)
    parser.add_argument('--random_ratio', type=float, default=0.001)
    
    # Training
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--T', type=float, default=0.6)
    parser.add_argument('--patience', type=int, default=100)
    
    # Ablations
    parser.add_argument('--no_hysteresis', action='store_true')
    parser.add_argument('--no_topo_search', action='store_true')
    parser.add_argument('--no_force_margin', action='store_true')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dataset-specific hyperparameters
    if args.dataset == 'cora':
        defaults = dict(lam=0.1, tau=0.2, gamma=0.5, s=1.0, delta=0.1, beta=0.1,
                       lr=1e-3, epochs=300, hidden_dim=256, T=0.6, k_2hop=4,
                       random_ratio=0.001, num_steps=10)
    elif args.dataset == 'chameleon':
        defaults = dict(lam=0.5, tau=0.1, gamma=0.5, s=1.5, delta=0.2, beta=0.3,
                       lr=5e-4, epochs=300, hidden_dim=256, T=0.6, k_2hop=8,
                       random_ratio=0.005, num_steps=10)
    elif args.dataset == 'zinc':
        defaults = dict(lam=0.1, tau=0.2, gamma=0.5, s=1.0, delta=0.1, beta=0.1,
                       lr=1e-3, epochs=200, hidden_dim=256, T=0.6, batch_size=128,
                       num_steps=8)
    elif args.dataset == 'peptides':
        defaults = dict(lam=0.5, tau=0.1, gamma=0.5, s=1.5, delta=0.2, beta=0.1,
                       lr=5e-4, epochs=100, hidden_dim=256, T=0.6, batch_size=64,
                       num_steps=8)
    elif args.dataset == 'ogbn-proteins':
        defaults = dict(lam=0.5, tau=0.1, gamma=0.5, s=1.5, delta=0.2, beta=0.1,
                       lr=5e-4, epochs=200, hidden_dim=256, T=0.6, num_steps=8)
    elif args.dataset == 'ogbg-molpcba':
        defaults = dict(lam=0.3, tau=0.2, gamma=0.5, s=1.0, delta=0.1, beta=0.1,
                       lr=1e-3, epochs=100, hidden_dim=256, T=0.6, batch_size=128,
                       num_steps=8)
    else:
        defaults = {}
    
    # Apply defaults only if not explicitly set
    for k, v in defaults.items():
        if not any(f'--{k}' in arg for arg in sys.argv):
            setattr(args, k, v)
    
    print(f"Training HGODE on {args.dataset}")
    print(f"Hyperparameters: {vars(args)}")
    
    results = []
    for seed in range(args.num_seeds):
        actual_seed = args.seed + seed
        torch.manual_seed(actual_seed)
        np.random.seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(actual_seed)
        
        print(f"\n=== Seed {seed+1}/{args.num_seeds} (seed={actual_seed}) ===")
        
        if args.dataset == 'cora':
            result = train_cora(args)
        elif args.dataset == 'chameleon':
            result = train_chameleon(args)
        elif args.dataset == 'zinc':
            result = train_zinc(args)
        elif args.dataset == 'peptides':
            result = train_peptides(args)
        elif args.dataset == 'ogbn-proteins':
            result = train_ogbn_proteins(args)
        elif args.dataset == 'ogbg-molpcba':
            result = train_ogbg_molpcba(args)
        
        if result is not None:
            results.append(result)
            print(f"Seed {seed+1} result: {result:.4f}")
    
    if results:
        mean_result = np.mean(results)
        std_result = np.std(results)
        print(f"\n{'='*50}")
        print(f"Final: {mean_result:.4f} ± {std_result:.4f}")
        
        # Determine ablation suffix
        suffix = ""
        if args.no_hysteresis:
            suffix = "_no_hysteresis"
        elif args.no_topo_search:
            suffix = "_no_topo_search"
        elif args.no_force_margin:
            suffix = "_no_force_margin"
        
        # Save results
        result_dict = {
            'dataset': args.dataset,
            'mean': mean_result,
            'std': std_result,
            'results': results,
            'ablation': suffix.lstrip('_') if suffix else 'full',
            'hyperparameters': {k: v for k, v in vars(args).items() 
                              if k not in ['data_dir', 'output_dir']},
        }
        
        fname = f"{args.dataset}{suffix}.json"
        with open(os.path.join(args.output_dir, fname), 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"Results saved to {os.path.join(args.output_dir, fname)}")


if __name__ == '__main__':
    main()
