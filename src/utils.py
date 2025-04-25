import torch
import torch.nn as nn
import numpy as np

class RunningMeanStd(nn.Module):
    """
    Calculates the running mean and standard deviation of a data stream.
    Uses Welford's online algorithm for numerical stability.
    """
    def __init__(self, shape, epsilon=1e-4):
        super().__init__()
        self.shape = shape
        self.epsilon = epsilon
        # Use register_buffer for state_dict compatibility without marking as parameters
        self.register_buffer('mean', torch.zeros(shape, dtype=torch.float64))
        self.register_buffer('var', torch.ones(shape, dtype=torch.float64))
        self.register_buffer('count', torch.tensor(epsilon, dtype=torch.float64)) # Start count > 0 to avoid division by zero

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """ Update running mean and variance using a batch of data x. """
        # Ensure x is float64 for stability
        x = x.to(torch.float64)
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False) # Use population variance for batch
        batch_count = x.shape[0]

        if batch_count == 0:
            return # Nothing to update

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        # Update mean
        new_mean = self.mean + delta * batch_count / tot_count
        # Update variance using Welford's method component
        M2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        # Update buffers
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(tot_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """ Normalize x using the running mean and variance. """
        # Ensure normalization happens with the same dtype as input or float32
        original_dtype = x.dtype
        mean = self.mean.to(original_dtype)
        std = torch.sqrt(self.var + self.epsilon).to(original_dtype)
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Alias for normalize for convenience in nn.Sequential. """
        return self.normalize(x)

    # Ensure state_dict includes mean, var, count
    # Note: Buffers are automatically included in state_dict