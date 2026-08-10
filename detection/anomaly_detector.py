import torch
import torch.nn as nn

from models.lstm_model.lstm_model import LSTMModel
from models.autoencoder.spatio_temporal_autoencoder import SpatioTemporalAutoencoder


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

        self.autoencoder =SpatioTemporalAutoencoder().to(DEVICE)

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

            mean_feature = torch.mean(sequence, dim=0)

            reconstructed = self.autoencoder(
                mean_feature.unsqueeze(0)
            )

            mse = torch.mean(
                (
                    reconstructed
                    -
                    mean_feature.unsqueeze(0)
                ) ** 2
            ).item()

        return prediction, mse