import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 2)
        self.pred_len = pred_len

    def forward(self, x):
        _, (h, c) = self.lstm(x)

        decoder_input = x[:, -1]  # last observed position
        outputs = []

        for _ in range(self.pred_len):
            decoder_input = decoder_input.unsqueeze(1)
            out, (h, c) = self.lstm(decoder_input, (h, c))
            pred = self.fc(out.squeeze(1))
            outputs.append(pred)
            decoder_input = pred

        return torch.stack(outputs, dim=1)