import torch

def ade(pred, gt):
    return torch.mean(torch.norm(pred - gt, dim=2))

def fde(pred, gt):
    return torch.mean(torch.norm(pred[:, -1] - gt[:, -1], dim=1))