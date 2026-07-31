"""
=========================================================
AnomaliNet
Spatio-Temporal Autoencoder

Used for anomaly detection by reconstructing
high-level spatial features extracted from ResNet50.

Input  : (Batch Size, 2048)
Output : (Batch Size, 2048)

Author : Harmain
=========================================================
"""

import torch
import torch.nn as nn


class SpatioTemporalAutoencoder(nn.Module):

    def __init__(self, input_dim=2048):
        super().__init__()

        self.input_dim = input_dim

        # ==========================================
        # Encoder
        # ==========================================

        self.encoder = nn.Sequential(

            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128)

        )

        # ==========================================
        # Decoder
        # ==========================================

        self.decoder = nn.Sequential(

            nn.Linear(128, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, input_dim)

        )

        self._initialize_weights()

    # =====================================================
    # Weight Initialization
    # =====================================================

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:

                    nn.init.zeros_(module.bias)

    # =====================================================
    # Forward Pass
    # =====================================================

    def forward(self, x):

        if x.dim() != 2:

            raise ValueError(
                f"Expected input shape (batch_size, {self.input_dim}), "
                f"but received {tuple(x.shape)}"
            )

        latent = self.encoder(x)

        reconstructed = self.decoder(latent)

        return reconstructed

    # =====================================================
    # Extract Latent Features
    # =====================================================

    def encode(self, x):

        return self.encoder(x)

    # =====================================================
    # Reconstruction
    # =====================================================

    def reconstruct(self, x):

        return self.forward(x)