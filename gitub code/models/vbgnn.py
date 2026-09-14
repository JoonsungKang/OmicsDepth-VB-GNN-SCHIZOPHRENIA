import torch
import torch.nn as nn
import torch.nn.functional as F

class VBGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(VBGNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, relaxed_adj):
        # Message passing over relaxed continuous edge probability manifold $\boldsymbol{\Pi}$
        support = self.linear(x)
        output = torch.matmul(relaxed_adj, support)
        return F.gelu(output)

class VBGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VBGNN, self).__init__()
        self.layer1 = VBGNNLayer(input_dim, hidden_dim)
        self.layer2 = VBGNNLayer(hidden_dim, output_dim)

    def forward(self, x, pi, temperature=0.5):
        # Gumbel-Softmax reparameterization for discrete DAG relaxation
        noise1 = -torch.log(-torch.log(torch.rand_like(pi) + 1e-9) + 1e-9)
        noise2 = -torch.log(-torch.log(torch.rand_like(pi) + 1e-9) + 1e-9)
        relaxed_adj = torch.sigmoid((torch.log(pi + 1e-9) - torch.log(1.0 - pi + 1e-9) + noise1 - noise2) / temperature)
        
        h = self.layer1(x, relaxed_adj)
        out = self.layer2(h, relaxed_adj)
        return out, relaxed_adj
```[cite: 2, 3]