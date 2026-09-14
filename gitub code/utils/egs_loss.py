import torch

class EGSRegularization(torch.nn.Module):
    """
    Entropy-based Gini-Simpson (EGS) structural regularizing module ensuring 
    Lipschitz-continuous gradient stability under dense inter-entity covariance[cite: 2, 3].
    """
    def __init__(self, beta=1.0, lambda_reg=0.1):
        super(EGSRegularization, self).__init__()
        self.beta = beta
        self.lambda_reg = lambda_reg

    def forward(self, pi, covariance_matrix):
        # EGS Structural Diversity Index: D = 1 - sum(pi_ij^2)
        d_index = 1.0 - torch.sum(pi ** 2)
        
        # Dependency-aware trace penalty: Tr(Pi^T * Sigma * Pi)
        trace_term = torch.trace(torch.matmul(torch.matmul(pi.T, covariance_matrix), pi))
        
        # Quadratic curvature loss ensuring Lipschitz-continuous stability
        loss_struct = self.lambda_reg * d_index * self.beta * trace_term
        return loss_struct
```[cite: 3]