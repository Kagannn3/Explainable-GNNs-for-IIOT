# phase2_data_loader.py

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

def load_cmapss(data_dir: str,
                dataset: str = 'FD003',
                window_size: int = 30,
                stride: int = 5):
    """
    Load CMAPSS data, compute Remaining Useful Life (RUL), and
    segment into sliding windows.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing CMAPSS text files.
    dataset : str
        One of 'FD001', 'FD002', 'FD003', 'FD004'.
    window_size : int
        Number of time steps per window.
    stride : int
        Step size between the starts of consecutive windows.

    Returns
    -------
    List[Data]
        A list of torch_geometric.data.Data objects, one per window.
        - x: Tensor of shape [num_sensors, window_size]
        - y: Tensor of shape [1] (RUL at end of window)
    """
    # Build full file path
    file_path = os.path.join(data_dir, f'{dataset}.txt')
    # CMAPSS files are space-delimited without header
    # Columns: [unit, cycle, op_setting1, op_setting2, op_setting3, sensor1, ..., sensorN]
    col_names = ['unit', 'cycle'] + \
                [f'op_setting{i}' for i in range(1, 4)] + \
                [f'sensor{i}' for i in range(1, 22)]
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)

    # Compute RUL: for each engine unit, RUL = max_cycle - current_cycle
    df['max_cycle'] = df.groupby('unit')['cycle'].transform('max')
    df['RUL'] = df['max_cycle'] - df['cycle']
    # Drop auxiliary column
    df.drop(columns=['max_cycle'], inplace=True)

    windows = []
    # For each engine unit, segment its time series into sliding windows
    for unit_id, group in df.groupby('unit'):
        group = group.reset_index(drop=True)
        n_cycles = len(group)
        # Slide window over this unit's data
        # Ensure we don't go out of bounds
        # start from 0, end at n_cycles - window_size + 1 
        
        for start in range(0, n_cycles - window_size + 1, stride):
            end = start + window_size
            window_df = group.iloc[start:end]
            # Features: take sensor readings only (columns sensor1...sensor21)
            sensor_cols = [c for c in window_df.columns if c.startswith('sensor')]
            sensor_data = window_df[sensor_cols].values  # shape (window_size, num_sensors)
            # Transpose so that nodes = sensors, features = time steps
            x = torch.tensor(sensor_data.T, dtype=torch.float)  # shape (num_sensors, window_size)
            # Target: RUL at the last time step of this window
            y = torch.tensor([window_df['RUL'].iloc[-1]], dtype=torch.float)
            # Create a Data object (edge_index to be added later)
            data = Data(x=x, y=y)
            # inside load_cmapss, before appending each window
            window_x = df.values[start : start + window_size]   # <— ensure you’re using this
            #print(f"DEBUG: window #{len(window_x)} shape = {window_x.shape}")  
            # should print (20, F), not (21, F)

            windows.append(data)

    return windows


def build_graph(x: torch.Tensor,
                threshold: float = 0.8):
    """
    Build an undirected graph based on Pearson correlation between sensor nodes.

    Each node represents one sensor; an edge is created between two sensors
    if the absolute value of their Pearson correlation over the window exceeds
    the given threshold.

    Parameters
    ----------
    x : torch.Tensor
        Node features of shape [num_sensors, window_size].
    threshold : float
        Correlation threshold for creating edges (0 <= threshold <= 1).

    Returns
    -------
    torch.LongTensor
        edge_index tensor of shape [2, num_edges], suitable for PyG.
    """
    # Convert to numpy for correlation computation
    x_np = x.cpu().numpy()
    # Compute correlation matrix: shape (num_sensors, num_sensors)
    corr = np.corrcoef(x_np)
    # Find pairs (i, j) where |corr| >= threshold, excluding self-loops
    num_sensors = corr.shape[0]
    edges_src = []
    edges_dst = []
    for i in range(num_sensors):
        for j in range(i + 1, num_sensors):
            if abs(corr[i, j]) >= threshold:
                # Add both directions for undirected graph
                edges_src += [i, j]
                edges_dst += [j, i]

    # Stack into edge_index
    if len(edges_src) == 0:
        # No edges passed threshold; return empty edge_index
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    return edge_index
