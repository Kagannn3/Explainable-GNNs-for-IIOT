import random
from typing import List, Dict
from torch_geometric.data import Data
import numpy as np

def non_iid_split_by_condition(
    data_list: List[Data],
    num_clients: int = 3,
    seed: int = 42,
    cm_pass_threshold: float = None
) -> Dict[int, List[Data]]:
    """
    Split data_list into `num_clients` subsets in a non-IID way.

    - For binary-labeled Data (d.y in {0,1}), treat 0 vs. 1 as classes.
    - For real-valued Data (regression RUL), bucket into:
        * class0 = windows with y >= threshold  (early-life)
        * class1 = windows with y <  threshold  (late-life)
      where threshold is e.g. the median RUL or a user-specified `cm_pass_threshold`.

    Splitting strategy (for both cases):
      - Client 0: 80% of class0 + 10% of class1
      - Client 1: 80% of class1 + 10% of class0
      - Client 2: remaining 10% of each class

    Args:
        data_list: list of PyG Data objects, each with scalar d.y
        num_clients: number of clients (3)
        seed: for reproducibility
        cm_pass_threshold: for regression, threshold to separate early vs late; 
            if None, set to median of all d.y

    Returns:
        dict mapping client_id -> list of Data
    """
    assert num_clients == 3, "This function supports exactly 3 clients."
    random.seed(seed)

    # Extract all labels into a numpy array
    ys = np.array([float(d.y.item()) for d in data_list])

    # Determine if this is a classification (binary labels 0/1) or regression (continuous)
    unique_labels = set(np.unique(ys))
    is_classification = unique_labels.issubset({0.0, 1.0})

    if is_classification:
        # Binary classification split
        class0 = [d for d, y in zip(data_list, ys) if y == 0.0]
        class1 = [d for d, y in zip(data_list, ys) if y == 1.0]
    else:
        # Regression: bucket by RUL threshold
        if cm_pass_threshold is None:
            cm_pass_threshold = float(np.median(ys))
        # early-life = large RUL; late-life = small RUL
        class0 = [d for d, y in zip(data_list, ys) if y >= cm_pass_threshold]
        class1 = [d for d, y in zip(data_list, ys) if y <  cm_pass_threshold]

    # Shuffle both lists
    random.shuffle(class0)
    random.shuffle(class1)

    # Compute split sizes
    n0_0 = int(0.8 * len(class0))
    n1_0 = int(0.1 * len(class1))
    client0 = class0[:n0_0] + class1[:n1_0]

    n1_1 = int(0.8 * len(class1))
    n0_1 = int(0.1 * len(class0))
    client1 = class1[:n1_1] + class0[:n0_1]

    # Client 2 gets the rest
    client2 = class0[n0_0:] + class1[n1_1:]

    # Final shuffle within each client
    for lst in (client0, client1, client2):
        random.shuffle(lst)

    return {0: client0, 1: client1, 2: client2}
