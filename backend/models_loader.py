"""
=========================================================
AnomaliNet
AI Model Loader

Loads all Deep Learning models only ONCE.

Author : Harmain
=========================================================
"""

import os
import torch

from ultralytics import YOLO
from torchvision import models

from models.lstm_model.lstm_model import LSTMModel
from models.autoencoder.spatio_temporal_autoencoder import (
    SpatioTemporalAutoencoder
)
from models.fusion.feature_fusion import FeatureFusion

# =====================================================
# DEVICE
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print(f"Loading AI Models on : {DEVICE}")
print("=" * 60)

# =====================================================
# YOLO MODEL
# =====================================================

YOLO_MODEL = YOLO(
    "models/yolo/yolov8n.pt"
)

print("YOLO Loaded")

# =====================================================
# RESNET50
# =====================================================

RESNET_MODEL = models.resnet50(
    weights=None
)

RESNET_MODEL.fc = torch.nn.Identity()

RESNET_MODEL.load_state_dict(

    torch.load(

        "models/cnn/resnet50_weights.pth",

        map_location=DEVICE

    )

)

RESNET_MODEL.eval()

RESNET_MODEL.to(DEVICE)

print("ResNet Loaded")

# =====================================================
# LSTM
# =====================================================

LSTM_MODEL = LSTMModel(

    input_size=2048,

    hidden_size=512,

    num_layers=2,

    num_classes=2

)

LSTM_MODEL.load_state_dict(

    torch.load(

        "models/lstm_model/lstm_model.pth",

        map_location=DEVICE

    )

)

LSTM_MODEL.eval()

LSTM_MODEL.to(DEVICE)

print("LSTM Loaded")

# =====================================================
# AUTOENCODER
# =====================================================

AUTOENCODER_MODEL = SpatioTemporalAutoencoder(

    input_dim=2048

)

AUTOENCODER_MODEL.load_state_dict(

    torch.load(

        "models/autoencoder/autoencoder.pth",

        map_location=DEVICE

    )

)

AUTOENCODER_MODEL.eval()

AUTOENCODER_MODEL.to(DEVICE)

print("AutoEncoder Loaded")

# =====================================================
# FEATURE FUSION
# =====================================================

FEATURE_FUSION_MODEL = FeatureFusion()

print("Feature Fusion Loaded")

print("=" * 60)
print("All AI Models Loaded Successfully")
print("=" * 60)