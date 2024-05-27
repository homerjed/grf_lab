from math import floor
import torch
from torch import nn


class RealNVP(nn.Module):
    def __init__(self, net_s, net_t, masks, prior):
        super(RealNVP, self).__init__()
        self.prior = prior
        self.masks = nn.Parameter(masks, requires_grad=False) # Masks are not to be optimised!
        self.t = torch.nn.ModuleList(
            [net_t() for _ in range(len(masks))]
        )
        self.s = torch.nn.ModuleList(
            [net_s() for _ in range(len(masks))]
        )
        
    def reverse(self, z, y):
        log_det_J, x = z.new_zeros(z.shape[0]), z
        for i in range(len(self.t)):
            x_ = x * self.masks[i]
            s = self.s[i](torch.cat([x_, y], 1)) * (1 - self.masks[i])
            t = self.t[i](torch.cat([x_, y], 1)) * (1 - self.masks[i])
            x = x_ + (1 - self.masks[i]) * (x * torch.exp(s) + t)
            log_det_J += s.sum(dim=1)
        return x, log_det_J

    def forward(self, x, y):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.masks[i] * z
            s = self.s[i](torch.cat([z_, y], 1)) * (1 - self.masks[i])
            t = self.t[i](torch.cat([z_, y], 1)) * (1 - self.masks[i])
            z = (1 - self.masks[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self, x, y):
        z, logp = self.reverse(x, y)
        return self.prior.log_prob(z) + logp
        
    def sample(self, n, y): 
        z = self.prior.sample((n,))
        logp = self.prior.log_prob(z)
        x, _ = self.reverse(z, y)
        return x