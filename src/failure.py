import torch

def high_error_cases(pred, gt, threshold=1.0):
    errors = torch.norm(pred[:, -1] - gt[:, -1], dim=1)
    return (errors > threshold).sum().item()

def diversity_score(pred_samples):
    # variance across samples
    return torch.var(pred_samples, dim=0).mean().item()