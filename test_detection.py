import torch

from detection.score_calculator import (
    calculate_anomaly_score
)

from detection.threshold_decision import (
    is_anomaly
)

x = torch.randn(1,512)

y = x + torch.randn(1,512)*0.2

score = calculate_anomaly_score(
    x,
    y
)

print("Score:", score)

print(
    "Anomaly:",
    is_anomaly(score)
)