import torch

def calculate_anomaly_score(
    original_features,
    reconstructed_features
):

    mse = torch.mean(
        (original_features - reconstructed_features) ** 2
    )

    return mse.item()