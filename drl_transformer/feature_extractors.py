"""Transformer over frame-stacked observations for Stable-Baselines3 SAC."""

import torch
import torch.nn as nn
from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """Encode ``(B, n_frames * n_features)`` as a sequence with a Transformer encoder.

    Uses full-sequence attention (not causal); pools the last timestep, same idea
    as taking the final GRU state.
    """

    def __init__(
        self,
        observation_space: Box,
        n_frames: int = 8,
        n_features_per_frame: int = 31,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim=d_model)
        self.n_frames = n_frames
        self.n_features_per_frame = n_features_per_frame

        self.input_proj = nn.Linear(n_features_per_frame, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_frames, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        x = observations.view(batch, self.n_frames, self.n_features_per_frame)
        x = self.input_proj(x) + self.pos_embed
        x = self.encoder(x)
        return self.layer_norm(x[:, -1, :])
