
import os, sys
# Compute absolute path to project-root/src
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, SRC_DIR)

from phase2_data_loader import load_cmapss, build_graph

import os
import numpy as np
import torch

def test_load_and_build():
    # 1) Locate a small slice of CMAPSS data
    data_dir = "data/CMAPSS"
    dataset = "FD003"

    # 2) Load just 5 sliding windows (window_size=20, stride=50 to skip quickly)
    windows = load_cmapss(
        data_dir=data_dir,
        dataset=dataset,
        window_size=20,
        stride=50
    )[:5]

    assert len(windows) == 5, f"Expected 5 windows, got {len(windows)}"

    for i, data in enumerate(windows):
        # 3) Check that x is a FloatTensor of shape [window_size, num_sensors]
        assert isinstance(data.x, torch.FloatTensor), f"data.x not FloatTensor in window {i}"
        assert data.x.shape[0] == 20, f"Window {i} has wrong time dimension: {data.x.shape[0]}"
        assert data.x.ndim == 2, f"data.x should be 2D, got {data.x.ndim}D"

        # 4) Check that y is a FloatTensor or LongTensor with a single value
        assert isinstance(data.y, torch.Tensor), f"data.y not Tensor in window {i}"
        assert data.y.numel() == 1, f"Window {i} label should be scalar, got {data.y.numel()} elems"

        # 5) Build the graph at threshold=0.8
        edge_index = build_graph(data.x, threshold=0.8)
        #   a) It should be a LongTensor of shape [2, E]
        assert isinstance(edge_index, torch.LongTensor), f"edge_index not LongTensor for window {i}"
        assert edge_index.ndim == 2 and edge_index.shape[0] == 2, \
            f"edge_index should have shape [2, E], got {tuple(edge_index.shape)}"

        #   b) No self‐loops: u != v across all edges
        u, v = edge_index
        assert (u != v).all(), f"Self-loop detected in window {i}"

        #   c) For a random edge, verify abs(corr) >= 0.8
        if edge_index.shape[1] > 0:
            # pick one edge
            idx = np.random.randint(0, edge_index.shape[1])
            ui, vi = u[idx].item(), v[idx].item()
            sensor_series = data.x.numpy()
            corr = np.corrcoef(sensor_series[:, ui], sensor_series[:, vi])[0,1]
            assert abs(corr) >= 0.8, \
                f"Edge ({ui},{vi}) has corr={corr:.2f} < threshold in window {i}"

    print("✅ phase2_data_loader smoke test passed.")

if __name__ == "__main__":
    test_load_and_build()
