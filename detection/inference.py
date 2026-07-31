import torch
import numpy as np

from detection.anomaly_detector import (
    AnomalyDetector
)

detector = AnomalyDetector()


def run_inference(feature_sequence):

    sequence = torch.tensor(
        feature_sequence,
        dtype=torch.float32
    )

    prediction, score = detector.predict(
        sequence
    )

    if prediction == 1:
        label = "Fight"
    else:
        label = "NonFight"

    return {
        "label": label,
        "score": score
    }