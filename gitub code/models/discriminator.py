import torch
import torch.nn as nn

class AdversarialDiscriminator(nn.Module):
    """
    Adversarial surrogate variable module minimizing mutual information 
    I(Z_causal, Z_nuisance) to eliminate technical confounding artifacts[cite: 3].
    """
    def __init__(self, latent_dim):
        super(AdversarialDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z_causal, z_nuisance):
        # Concatenate causal and nuisance latent representations to estimate joint dependence
        joint = torch.cat([z_causal, z_nuisance], dim=1)
        return self.net(joint)
```[cite: 3]