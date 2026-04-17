# losses/mmd.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MMDLoss(nn.Module):

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            source = F.normalize(source, dim=1)
            target = F.normalize(target, dim=1)
        mu_s = source.mean(dim=0)
        mu_t = target.mean(dim=0)
        delta = mu_s - mu_t
        return (delta * delta).sum()