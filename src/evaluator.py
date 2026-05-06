from src.metrics import ade, fde
from src.failure import high_error_cases

def evaluate(pred, gt):
    return {
        "ADE": ade(pred, gt).item(),
        "FDE": fde(pred, gt).item(),
        "Failures": high_error_cases(pred, gt)
    }