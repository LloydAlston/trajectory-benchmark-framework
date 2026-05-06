import pandas as pd
import torch
import numpy as np

def load_data(path, obs_len, pred_len):
    df = pd.read_csv(path)
    traj = df[['x','y']].values.astype(np.float32)

    mean = traj.mean(axis=0)
    std = traj.std(axis=0)

    std[std < 1e-3] = 1.0

    traj = (traj - mean) / std
    X, Y = [], []
    for i in range(len(traj) - obs_len - pred_len):
        X.append(traj[i:i+obs_len])
        Y.append(traj[i+obs_len:i+obs_len+pred_len])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    return torch.from_numpy(X), torch.from_numpy(Y)