from policy.layers.layers import Mlp
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Categorical

import torch as th
import gym.spaces

class BootstrappingFE(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, num_agents_to_select=3, embedding_size=16, n_samples=3):
        super().__init__(observation_space, features_dim=1)

        self._n = num_agents_to_select
        self._n_samples = n_samples

        self._my_vel_extractor = Mlp(2, embedding_size)
        self._my_pos_extractor = Mlp(2, embedding_size)

        self._other_agents_pos_extractor = Mlp(2*self._n, self._n*embedding_size)
        self._target_pos_extractor = Mlp(2*self._n, self._n*embedding_size)

        self._features_dim = embedding_size * (2 + 2 * self._n)

    def forward(self, obs):
        batch_size = obs["my_vel"].shape[0]

        my_vel_embeddings = self._my_vel_extractor(obs["my_vel"])
        my_pos_embeddings = self._my_pos_extractor(obs["my_pos"])

        other_pos_embeddings = self.forward_other_pos(obs, batch_size)
        target_pos_embeddings = self.forward_target_pos(obs, batch_size)

        output = th.cat([my_vel_embeddings, my_pos_embeddings, other_pos_embeddings, target_pos_embeddings], dim=-1)
        assert output.shape[-1] == self._features_dim
        return output

    def forward_other_pos(self, obs, batch_size):
        other_pos = obs["other_pos"]
        other_pos_distances = th.linalg.norm(other_pos, dim=-1)
        other_pos_dist = Categorical(logits=-other_pos_distances)
        s = other_pos_dist.sample((self._n, self._n_samples))

        sampled_data_other_pos = other_pos[th.arange(batch_size), s].transpose(0,2)
        sampled_data_other_pos = sampled_data_other_pos.flatten(start_dim=-2)
        return self._other_agents_pos_extractor(sampled_data_other_pos).mean(dim=-2)

    def forward_target_pos(self, obs, batch_size):
        entity_pos = obs["entity_pos"]

        entity_pos_distances = th.linalg.norm(entity_pos, dim=-1)
        entity_pos_dist = Categorical(logits=-entity_pos_distances)
        s = entity_pos_dist.sample((self._n, self._n_samples))

        sampled_data_entity_pos = entity_pos[th.arange(batch_size), s].transpose(0,2)
        sampled_data_entity_pos = sampled_data_entity_pos.flatten(start_dim=-2)
        return self._target_pos_extractor(sampled_data_entity_pos).mean(dim=-2)
