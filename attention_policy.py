from torch import nn
from layers import ScaledDotProductAttention, ScaledDotProductAttentionWithExtractors, Mlp
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gym
import gym.spaces
import torch as th

class AttentionPolicyV1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, embedding_dim=16):
        super().__init__(observation_space, features_dim=1)

        self._my_vel_embedding_extractor = Mlp(2, embedding_dim)

        self._self_attn_other_agents = ScaledDotProductAttentionWithExtractors(embedding_dim, 2, 2, 2)
        self._my_vel_attn_other_agents = ScaledDotProductAttentionWithExtractors(embedding_dim, embedding_dim, embedding_dim, embedding_dim)

        self._self_attn_targets = ScaledDotProductAttentionWithExtractors(embedding_dim, 2, 2, 2)
        self._final_attn_targets = ScaledDotProductAttentionWithExtractors(embedding_dim, 2*embedding_dim, embedding_dim, embedding_dim)

        self._features_dim = 2 * embedding_dim

    def forward(self, observations):
        my_vel = observations["my_vel"]
        my_vel = th.reshape(my_vel, (my_vel.shape[0], 1, my_vel.shape[-1]))

        my_vel_embedding = self._my_vel_embedding_extractor(my_vel)

        self_attn_other_agents, _ = self._self_attn_other_agents(
            observations["other_pos"],
            observations["other_pos"],
            observations["other_pos"]
        )
        my_vel_attn_other_agents, _ = self._my_vel_attn_other_agents(
            my_vel_embedding,
            self_attn_other_agents,
            self_attn_other_agents
        )

        self_attn_targets, _ = self._self_attn_targets(
            observations["entity_pos"],
            observations["entity_pos"],
            observations["entity_pos"]
        )
        final_attn_targets, _ = self._final_attn_targets(
            th.cat([my_vel_embedding, my_vel_attn_other_agents], dim=-1),
            self_attn_targets,
            self_attn_targets
        )

        attn_output = th.cat([my_vel_attn_other_agents, final_attn_targets], dim=-1)
        attn_output = th.squeeze(attn_output, dim=-2)

        assert attn_output.shape[-1] == self._features_dim
        return attn_output

