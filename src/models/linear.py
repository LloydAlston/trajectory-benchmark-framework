import torch

class LinearModel:
    def predict(self, obs, pred_len):
        velocity = obs[:, -1] - obs[:, -2]
        preds = []
        last = obs[:, -1]

        for _ in range(pred_len):
            last = last + velocity
            preds.append(last)

        return torch.stack(preds, dim=1)