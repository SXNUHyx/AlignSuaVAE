import torch
import torch.nn as nn

class Normal:
    def __init__(self, device="cpu"):
        self.device = device

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std, device=self.device)
        return mu + eps * std

    def log_pdf(self, x, mu, log_var, reduce=True):
        log_std = 0.5 * log_var
        log_p = -0.5 * ((x - mu) ** 2 / torch.exp(log_var) + 2 * log_std + torch.log(torch.tensor(2 * torch.pi, device=self.device)))
        if reduce:
            return torch.sum(log_p, dim=-1).mean()
        else:
            return torch.sum(log_p, dim=-1)