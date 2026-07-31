import torch
import torch.nn as nn

class LSTMModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(
            256,
            2
        )

    def forward(self, x):

        output, (hn, cn) = self.lstm(x)

        out = self.fc(
            hn[-1]
        )

        return out