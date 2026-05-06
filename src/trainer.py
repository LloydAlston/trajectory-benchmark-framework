import torch
from torch import nn

def train(model, X, Y, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()

    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()

        pred = model(X)
        loss = loss_fn(pred, Y)

        if torch.isnan(loss):
            print("Loss became NaN. Stopping.")
            break

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        print(f"Epoch {epoch} Loss {loss.item():.6f}", flush=True)