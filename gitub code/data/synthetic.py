import torch
import numpy as np

class SyntheticANMGenerator:
    """
    Non-linear Additive Noise Model (ANM) generator for Erdős-Rényi (ER-2, ER-4) 
    and Scale-Free (SF-4) graph benchmarking[cite: 2].
    """
    def __init__(self, num_nodes=100, samples=600):
        self.num_nodes = num_nodes
        self.samples = samples

    def generate_erdos_renyi(self, p=0.05):
        adj = (np.random.rand(self.num_nodes, self.num_nodes) < p).astype(float)
        np.fill_diagonal(adj, 0)
        
        x = np.zeros((self.samples, self.num_nodes))
        for j in range(self.num_nodes):
            parents = np.where(adj[:, j] > 0)[0]
            noise = np.random.normal(0, 1, self.samples)
            if len(parents) > 0:
                parent_sum = np.sum(np.tanh(x[:, parents]), axis=1)
                x[:, j] = parent_sum + noise
            else:
                x[:, j] = noise
        return torch.tensor(x, dtype=torch.float32), torch.tensor(adj, dtype=torch.float32)
```[cite: 2]