"""Custom Dict-aware feature extractor for the Pitwall PPO policy.

Larger NatureCNN-style network than SB3's default (channels 64→128→128 vs
the default 32→64→64), plus a small MLP for the state vector. Designed to
ingest stacked frames (n_stack=4 → 12 image channels after channel-axis
stacking) and a stacked state vector (3 → 12 with the same stacking).

By the time this extractor's __init__ and forward run, SB3 has wrapped
the vec env in `VecTransposeImage` so the image observation is already
in CHW layout. We rely on that — no permute in forward, and we read the
channel count from `image_space.shape[0]`.
"""

from __future__ import annotations

import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PitwallFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        image_features_dim: int = 512,
        state_features_dim: int = 64,
    ):
        # Total flat features the policy/value heads consume.
        super().__init__(
            observation_space,
            features_dim=image_features_dim + state_features_dim,
        )

        image_space = observation_space.spaces["image"]
        # Image obs is CHW (SB3 has applied VecTransposeImage upstream).
        n_input_channels = int(image_space.shape[0])
        self.image_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Run a dummy CHW sample to compute the flattened conv output dim.
        with th.no_grad():
            sample = th.as_tensor(image_space.sample()[None]).float()
            n_flatten = self.image_cnn(sample).shape[1]

        self.image_fc = nn.Sequential(
            nn.Linear(n_flatten, image_features_dim),
            nn.ReLU(),
        )

        state_space = observation_space.spaces["state"]
        state_dim = int(state_space.shape[0])
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, state_features_dim),
            nn.ReLU(),
            nn.Linear(state_features_dim, state_features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        # Image is already CHW. SB3 normalises uint8 [0, 255] -> float [0, 1]
        # itself when `normalize_images=True` (the default), but our CNN runs
        # before that hook, so we divide by 255 here for safety.
        image = observations["image"].float() / 255.0
        image_feat = self.image_fc(self.image_cnn(image))

        state_feat = self.state_mlp(observations["state"].float())
        return th.cat([image_feat, state_feat], dim=1)
