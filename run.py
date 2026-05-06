from src.utils import load_config, set_seed
from src.data import load_data
from src.models.lstm import LSTMModel
from src.models.linear import LinearModel
from src.trainer import train
from src.evaluator import evaluate
import torch

config = load_config("config.yaml")
set_seed(config["seed"])

X, Y = load_data("data/sample.csv", config["obs_len"], config["pred_len"])
print("NaN check X:", torch.isnan(X).any().item())
print("NaN check Y:", torch.isnan(Y).any().item())
print("Data shape:", X.shape, Y.shape)

# LSTM
lstm = LSTMModel(config["pred_len"])
train(lstm, X, Y, config)

lstm.eval()
with torch.no_grad():
    pred_lstm = lstm(X)

# Linear baseline
linear = LinearModel()
pred_linear = linear.predict(X, config["pred_len"])

print("LSTM:", evaluate(pred_lstm, Y))
print("Linear:", evaluate(pred_linear, Y))