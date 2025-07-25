#!/usr/bin/env python
"""
preprocess_secom.py

Utilities to load, clean, window and graphify the SECOM dataset
for predictive maintenance experiments.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

def load_secom(data_dir: str,
               window_size: int = 30,
               stride: int = 5,
               threshold: float = 0.8):
    """
    Load and preprocess SECOM data:
      1. Reads `secom.data` (features) and `secom_labels.data` (binary pass/fail).
      2. Imputes missing values (mean strategy).
      3. Standardizes features (z‐score).
      4. Creates sliding windows of length `window_size` with step `stride`.
      5. For each window, label = majority label in that window.
      6. Builds a graph per window via Pearson correlation > threshold.
    Returns:
      List[Data]  — each a torch_geometric.data.Data object with:
        x           (window_size × num_sensors)
        y           (window label)
        edge_index  (2 × num_edges) based on correlation graph
    """
    # 1. Read raw files
    feature_path = os.path.join(data_dir, "secom.data")
    label_path   = os.path.join(data_dir, "secom_labels.data")

    # SECOM: whitespace‐delimited; missing values encoded as NaN
    df_x = pd.read_csv(feature_path, sep='\s+', header=None, na_values='NaN')
    df_y = pd.read_csv(label_path,   sep='\s+', header=None, na_values='NaN')

    # 2. Impute missing (mean of column)
    df_x = df_x.fillna(df_x.mean())

    # 3. Standardize (z‐score)
    mu = df_x.mean()
    sigma = df_x.std().replace(0, 1.0)
    df_x = (df_x - mu) / sigma

    # 4. Windowing
    X = df_x.values  # shape: (num_samples, num_sensors)
    Y = df_y.values.ravel()  # binary labels: 1 = fail, -1 = pass
    N, F = X.shape

    data_list = []
    for start in range(0, N - window_size + 1, stride):
        end = start + window_size
        window_x = X[start:end, :]               # (window_size, F)
        window_y = Y[start:end]
        # majority‐vote label: 1 if any failure in window, else 0
        label = int((window_y == 1).any())

        # build correlation graph
        corr = np.corrcoef(window_x, rowvar=False)  # (F, F)
        # edges where |corr| > threshold, exclude self‐loops
        idx_u, idx_v = np.where((np.abs(corr) > threshold) & (np.eye(F) == 0))
        edge_index = torch.tensor([idx_u, idx_v], dtype=torch.long)

        # create PyG Data
        data = Data(
            x = torch.tensor(window_x, dtype=torch.float),
            y = torch.tensor([label], dtype=torch.long),
            edge_index = edge_index
        )
        data_list.append(data)

    return data_list


if __name__ == "__main__":
    # quick test/demo
    dl = load_secom(data_dir="../data/SECOM",
                    window_size=30,
                    stride=5,
                    threshold=0.8)
    print(f"Loaded {len(dl)} windows, each with {dl[0].x.shape[1]} sensors.")
    print(dl[0])  # print first window data
    print(" • x shape:", dl[0].x.shape)
    print(" • edge_index shape:", dl[0].edge_index.shape)
    print(" • y label:", dl[0].y)




