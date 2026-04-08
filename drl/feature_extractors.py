"""Custom feature extractors for Stable-Baselines3.

Provides GRU, LSTM, and deep MLP extractors that plug into SAC/PPO via
``policy_kwargs["features_extractor_class"]``.

The GRU/LSTM extractors expect frame-stacked observations: the flat input
of shape ``(batch, n_frames * n_features)`` is reshaped to
``(batch, n_frames, n_features)`` and processed as a temporal sequence.
"""

import torch
import torch.nn as nn
from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GRUFeatureExtractor(BaseFeaturesExtractor):
    """Process frame-stacked observations through a multi-layer GRU.

    Reshapes ``(B, n_frames * n_features)`` → ``(B, n_frames, n_features)``,
    runs a GRU, and returns the final hidden state with LayerNorm.
    """

    def __init__(
        self,
        observation_space: Box,
        n_frames: int = 4,
        n_features_per_frame: int = 31,
        gru_hidden_size: int = 128,
        gru_layers: int = 2,
    ):
        super().__init__(observation_space, features_dim=gru_hidden_size)
        self.n_frames = n_frames
        self.n_features_per_frame = n_features_per_frame

        self.input_proj = nn.Linear(n_features_per_frame, gru_hidden_size)
        self.gru = nn.GRU(
            input_size=gru_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.1 if gru_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(gru_hidden_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        x = observations.view(batch, self.n_frames, self.n_features_per_frame)
        x = torch.relu(self.input_proj(x))
        _, hidden = self.gru(x)
        return self.layer_norm(hidden[-1])


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """Same idea as GRU but using LSTM cells."""

    def __init__(
        self,
        observation_space: Box,
        n_frames: int = 4,
        n_features_per_frame: int = 31,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
    ):
        super().__init__(observation_space, features_dim=lstm_hidden_size)
        self.n_frames = n_frames
        self.n_features_per_frame = n_features_per_frame

        self.input_proj = nn.Linear(n_features_per_frame, lstm_hidden_size)
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(lstm_hidden_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        x = observations.view(batch, self.n_frames, self.n_features_per_frame)
        x = torch.relu(self.input_proj(x))
        _, (hidden, _) = self.lstm(x)
        return self.layer_norm(hidden[-1])


class DeepMLPExtractor(BaseFeaturesExtractor):
    """Deeper MLP with LayerNorm and residual connection."""

    def __init__(self, observation_space: Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        input_dim = observation_space.shape[0]
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Linear(512, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )
        self.residual = nn.Linear(512, 512)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.block1(observations)
        x = self.block2(x) + self.residual(x)
        return self.block3(x)
