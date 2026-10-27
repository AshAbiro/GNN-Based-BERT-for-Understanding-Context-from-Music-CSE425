import random

import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(preferred: str = "cuda"):
    import torch

    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collate_graphs(graph_list):
    """List[torch_geometric.data.Data] -> a single batched Data via torch_geometric.data.Batch."""
    from torch_geometric.data import Batch

    return Batch.from_data_list(graph_list)
