from detection.config import *


class FeatureFusion:

    def __init__(self):

        print("✓ Fusion Module Ready")

    def calculate(

        self,

        lstm_probability,

        reconstruction_error

    ):

        reconstruction_score = reconstruction_error / AUTOENCODER_THRESHOLD

        reconstruction_score = max(
            0.0,
            min(
                reconstruction_score,
                1.0
            )
        )

        fusion_score = (

            (LSTM_WEIGHT * lstm_probability)

            +

            (

                AUTOENCODER_WEIGHT *

                reconstruction_score

            )

        )

        return fusion_score

    def classify(

        self,

        lstm_probability,

        reconstruction_error

    ):

        fusion_score = self.calculate(

            lstm_probability,

            reconstruction_error

        )

        anomaly = (

            fusion_score >= FUSION_THRESHOLD

        )

        return {

            "fusion_score": fusion_score,

            "is_anomaly": anomaly

        }
    