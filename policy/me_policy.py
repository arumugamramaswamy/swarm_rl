from torch import nn
from .layers.layers import (
    Mlp,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gym
import gym.spaces
import torch as th


class MeanEmbeddingExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, embedding_dim=16):
        super().__init__(observation_space, features_dim=1)

        self._mean_embedding = Mlp(2 + 2, embedding_dim)
        self._features_dim = embedding_dim + 2

    def forward(self, observations):
        other_pos = th.tensor(observations["other_pos"])
        other_pos_one_hot = th.zeros((*other_pos.shape[:-1], 2))
        other_pos_one_hot[:, :, -2] = 1
        other_pos = th.cat([other_pos, other_pos_one_hot], dim=-1)

        entity_pos = th.tensor(observations["entity_pos"])
        entity_pos_one_hot = th.zeros((*entity_pos.shape[:-1], 2))
        entity_pos_one_hot[:, :, -1] = 1
        entity_pos = th.cat([entity_pos, entity_pos_one_hot], dim=-1)

        all_pos = th.cat([other_pos, entity_pos], dim=-2)

        my_vel = th.tensor(observations["my_vel"])
        my_vel = observations["my_vel"]

        mean_embeddings = self._mean_embedding(all_pos).mean(axis=-2)
        output = th.cat([th.tensor(mean_embeddings), my_vel], dim=-1)

        assert output.shape[-1] == self._features_dim
        return output
