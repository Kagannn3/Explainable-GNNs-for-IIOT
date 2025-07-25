#!/usr/bin/env python3
"""
train_client.py

This script runs on each federated client to:
  1. Load its non-IID CMAPSS data split (sliding windows + graphs).
  2. Train three models: LSTM baseline, GCN regressor, GAT regressor.
  3. Evaluate their predictive performance (MAE, RMSE).
  4. Generate explanations:
       - Integrated Gradients for LSTM
       - GNNExplainer (new Explainer API) for GCN & GAT
  5. Save model checkpoints, metrics, and explanation masks.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json

from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.loader import DataLoader as GeoDataLoader
# ✅ Correct: use the new Explainer API
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.explain.algorithm import GNNExplainer

from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader as SeqLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from phase2_data_loader import load_cmapss, build_graph
from phase2_federated_split import non_iid_split_by_condition

# ----------------------
# Model Definitions
# ----------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)  # [B]

class GCNRegressor(nn.Module):
    def __init__(self, in_feats, hidden_size):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.fc    = nn.Linear(hidden_size, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze(-1)

class GATRegressor(nn.Module):
    def __init__(self, in_feats, hidden_size, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_feats, hidden_size, heads=heads)
        self.gat2 = GATConv(hidden_size*heads, hidden_size, heads=1)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze(-1)

# ----------------------
# Training & Evaluation
# ----------------------
def train(model, loader, device, epochs, lr, is_graph):
    model.to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for ep in range(1, epochs+1):
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            if is_graph:
                batch = batch.to(device)
                pred  = model(batch.x, batch.edge_index, batch.batch)
                truth = batch.y
            else:
                x_seq, truth = batch
                # ensure shape [B, T, F]
                x_seq = x_seq.to(device)
                if x_seq.dim()==2:
                    x_seq = x_seq.unsqueeze(0)
                pred  = model(x_seq)
                truth = truth.to(device).float()
            loss = criterion(pred, truth)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"[Epoch {ep}/{epochs}] Loss: {total/len(loader):.4f}")
    return model

def evaluate(model, dataset, device, is_graph, return_all=False):
    """
    Evaluate model on `dataset`.

    Args:
      model     : torch.nn.Module
      dataset   : list of Data objects (if is_graph) or list of (x_seq, y) tuples
      device    : torch.device
      is_graph  : bool — whether to use GeoDataLoader
      return_all: bool — if True, also returns (trues, preds) lists

    Returns:
      If return_all=False: (mae, rmse)
      If return_all=True : (mae, rmse, trues, preds)
    """
    model.to(device).eval()
    preds, trues = [], []

    with torch.no_grad():
        if is_graph:
            loader = GeoDataLoader(dataset, batch_size=32, shuffle=False)
            for batch in loader:
                batch = batch.to(device)
                out   = model(batch.x, batch.edge_index, batch.batch)
                preds.extend(out.cpu().tolist())
                trues.extend(batch.y.cpu().tolist())
        else:
            for x_seq, y in dataset:
                x_seq = x_seq.to(device)
                # ensure shape [B, T, F]
                if x_seq.dim() == 2:
                    x_seq = x_seq.unsqueeze(0)
                out = model(x_seq)
                # if out is tensor of shape [1], extract scalar
                preds.append(out.view(-1).item())
                trues.append(y.item())

    # compute metrics
    mae  = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))

    if return_all:
        return mae, rmse, trues, preds
    else:
        return mae, rmse

# ----------------------
# XAI Explanations
# ----------------------
def explain_lstm(model, sample, device):
    model.to(device).eval()
    ig = IntegratedGradients(model)
    x_seq, _ = sample
    x_seq = x_seq.unsqueeze(0).to(device)
    attr, _ = ig.attribute(x_seq, target=None, return_convergence_delta=True)
    return attr.squeeze(0).cpu().numpy()  # [T,F]

def explain_gnn(model, data, device):
    model.to(device).eval()

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=50),
        explanation_type='model',
        node_mask_type='attributes',  # instead of 'scalar'
        edge_mask_type='object',      # instead of 'scalar'
        model_config=ModelConfig(
            mode='regression',
            task_level='graph',
            return_type='raw'
        )
    )


    # Prepare inputs
    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    batch      = getattr(data, 'batch',
                         torch.zeros(x.size(0), dtype=torch.long, device=device))

    # Graph‐level explanation
    exp = explainer(
        x=x,
        edge_index=edge_index,
        batch=batch
    )

    node_mask = exp.node_mask.cpu().numpy()
    edge_mask = exp.edge_mask.cpu().numpy()
    return node_mask, edge_mask
# ----------------------
# Main
# ----------------------
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # prepare folders
    for d in ['checkpoints','explanations','results','results/plots']:
        os.makedirs(d, exist_ok=True)

    # load and split
    all_data = load_cmapss(args.data_dir, args.dataset,
                           window_size=args.window_size,
                           stride=args.stride)
    splits = non_iid_split_by_condition(all_data, num_clients=3)
    client_data = splits[args.client_id]
    print(f"Client {args.client_id}: {len(client_data)} windows")

    # ─── Fix: build sequence dataset with (batch, time, features) ───
    seq_dataset = []
    for d in client_data:
        # d.x is [num_sensors, window_size] → transpose to [window_size, num_sensors]
        x = d.x.T             # now (T, F)
        seq_dataset.append((x, d.y))
    seq_loader = SeqLoader(seq_dataset, batch_size=args.batch_size, shuffle=True)

    # build graphs
    for d in client_data:
        d.edge_index = build_graph(d.x, threshold=args.threshold)
    graph_loader = GeoDataLoader(client_data, batch_size=args.batch_size, shuffle=True)

    # ─── Fix: unpack dims correctly ───
    # Take the first sample to infer dims
    # Infer sequence length (T) and feature dimension (F) from a single sample
    sample_x, _ = seq_dataset[0]
    if sample_x.dim() == 2:
        T, F = sample_x.shape      # (time steps, features)
    else:
        # just in case you ever accidentally keep a batch dim
        _, T, F = sample_x.shape

    print(f"LSTM will receive input of shape [batch, time, features]: [*, {T}, {F}]")

    # instantiate models with correct in/out sizes
    lstm = LSTMRegressor(input_size=F, hidden_size=args.hidden_size)
    # For GNNs, nodes = sensors (F), features = time steps (T)
    gcn  = GCNRegressor(in_feats=T, hidden_size=args.hidden_size)
    gat  = GATRegressor(in_feats=T, hidden_size=args.hidden_size)

    
    # print model input shapes
    
    # train
    lstm = train(lstm, seq_loader, device, args.epochs, args.lr, is_graph=False)
    gcn  = train(gcn, graph_loader, device, args.epochs, args.lr, is_graph=True)
    gat  = train(gat, graph_loader, device, args.epochs, args.lr, is_graph=True)

    # eval
    # New calls, capturing raw predictions and truths
    mae_l, rmse_l, trues_l, preds_l = evaluate(
        lstm, seq_dataset, device, is_graph=False, return_all=True)

    mae_g, rmse_g, trues_g, preds_g = evaluate(
        gcn, client_data, device, is_graph=True, return_all=True)

    mae_t, rmse_t, trues_t, preds_t = evaluate(
        gat, client_data, device, is_graph=True, return_all=True)
    print(f"Results — LSTM:  MAE={mae_l:.3f}, RMSE={rmse_l:.3f}")
    print(f"       — GCN:   MAE={mae_g:.3f}, RMSE={rmse_g:.3f}")
    print(f"       — GAT:   MAE={mae_t:.3f}, RMSE={rmse_t:.3f}")

    # ensure results folder exists
    os.makedirs('results', exist_ok=True)

    # save each pair of arrays
    np.save(f'results/trues_client{args.client_id}_LSTM.npy', np.array(trues_l))
    np.save(f'results/preds_client{args.client_id}_LSTM.npy', np.array(preds_l))

    np.save(f'results/trues_client{args.client_id}_GCN.npy', np.array(trues_g))
    np.save(f'results/preds_client{args.client_id}_GCN.npy', np.array(preds_g))

    np.save(f'results/trues_client{args.client_id}_GAT.npy', np.array(trues_t))
    np.save(f'results/preds_client{args.client_id}_GAT.npy', np.array(preds_t))

    # XAI on first sample
    ig_mask   = explain_lstm(lstm,    seq_dataset[0], device)
    gcn_n, gcn_e = explain_gnn(gcn,  client_data[0], device)
    gat_n, gat_e = explain_gnn(gat,  client_data[0], device)

    # save metrics
    metrics = {
        'lstm': {'mae': mae_l, 'rmse': rmse_l},
        'gcn':  {'mae': mae_g, 'rmse': rmse_g},
        'gat':  {'mae': mae_t, 'rmse': rmse_t},
    }
    with open(f'results/metrics_client{args.client_id}.json','w') as f:
        json.dump(metrics, f, indent=2)

    # save checkpoints
    suffix = f"client{args.client_id}_h{args.hidden_size}_lr{args.lr}"
    torch.save(lstm.state_dict(), f"checkpoints/{suffix}_lstm.pt")
    torch.save(gcn.state_dict(),  f"checkpoints/{suffix}_gcn.pt")
    torch.save(gat.state_dict(),  f"checkpoints/{suffix}_gat.pt")

    # save explanations
    np.save(f"explanations/{suffix}_lstm_attr.npy", ig_mask)
    np.save(f"explanations/{suffix}_gcn_node.npy", gcn_n)
    np.save(f"explanations/{suffix}_gcn_edge.npy", gcn_e)
    np.save(f"explanations/{suffix}_gat_node.npy", gat_n)
    np.save(f"explanations/{suffix}_gat_edge.npy", gat_e)

    print("Client run complete. Artifacts saved.")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   required=True)
    p.add_argument('--dataset',    default='FD003')
    p.add_argument('--client_id',  type=int, choices=[0,1,2], required=True)
    p.add_argument('--batch_size', type=int,   default=32)
    p.add_argument('--epochs',     type=int,   default=10)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--hidden_size',type=int,   default=64)
    p.add_argument('--window_size',type=int,   default=30)
    p.add_argument('--stride',     type=int,   default=5)
    p.add_argument('--threshold',  type=float, default=0.8)
    return p.parse_args()

if __name__ == '__main__':
    main()
