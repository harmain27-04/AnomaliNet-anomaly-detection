import torch
import torch.nn as nn

from models.lstm_model.lstm_model import LSTMModel
from models.autoencoder.spatio_temporal_autoencoder import AutoEncoder


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


class AnomalyDetector:

    def __init__(self):

        self.lstm = LSTMModel().to(DEVICE)

        self.lstm.load_state_dict(
            torch.load(
                "models/lstm_model/lstm_model.pth",
                map_location=DEVICE
            )
        )

        self.lstm.eval() 

        self.autoencoder = AutoEncoder().to(DEVICE)

        self.autoencoder.load_state_dict(
            torch.load(
                "models/autoencoder/autoencoder.pth",
                map_location=DEVICE
            )
        )

        self.autoencoder.eval()

    def predict(self, sequence):

        with torch.no_grad():

            sequence = sequence.to(DEVICE)

            lstm_output = self.lstm(
                sequence.unsqueeze(0)
            )

            prediction = torch.argmax(
                lstm_output,
                dim=1
            ).item()

            last_feature = sequence[-1]

            reconstructed = self.autoencoder(
                last_feature.unsqueeze(0)
            )

            mse = torch.mean(
                (
                    reconstructed
                    -
                    last_feature.unsqueeze(0)
                ) ** 2
            ).item()

        return prediction, mse