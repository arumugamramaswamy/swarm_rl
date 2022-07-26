from policy.layers.layers import Mlp
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

import gym.spaces
import torch as th

class AttnSelectorPolicy(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gym.spaces.Dict, n_select, embedding_size=16, num_heads=4
    ):
        super().__init__(observation_space, features_dim=1)

        self._n_select = n_select

        self._pos_embedding_extractor = Mlp(2, embedding_size)
        self._vel_embedding_extractor = Mlp(2, embedding_size)

        self._other_pos_attn = nn.MultiheadAttention(embedding_size, num_heads)
        self._target_pos_attn = nn.MultiheadAttention(embedding_size, num_heads)

        self._other_pos_attn = nn.MultiheadAttention(embedding_size, num_heads)
        self._target_pos_attn = nn.MultiheadAttention(embedding_size, num_heads)

        self._other_pos_selector = Mlp(embedding_size, 1)
        self._target_pos_selector = Mlp(embedding_size, 1)

        self._other_pos_final_embed = Mlp(embedding_size*self._n_select, embedding_size*self._n_select)
        self._target_pos_final_embed = Mlp(embedding_size*self._n_select, embedding_size*self._n_select)

        self._features_dim = embedding_size * (2 + 2*n_select)

    def forward(self, obs) -> th.Tensor:

        batch_size = obs["my_vel"].shape[0]

        my_vel_embed = self._vel_embedding_extractor(obs["my_vel"])
        my_pos_embed = self._pos_embedding_extractor(obs["my_pos"])
        other_pos_final_embed = self._extract_other_pos(obs["other_pos"], batch_size)
        target_pos_final_embed = self._extract_target_pos(obs["entity_pos"], batch_size)

        output = th.cat([other_pos_final_embed, target_pos_final_embed, my_pos_embed, my_vel_embed], dim=-1)

        assert output.shape[-1] == self._features_dim
        return output

    def _extract_other_pos(self, other_pos, batch_size):
        other_pos_embed = self._pos_embedding_extractor(other_pos)
        other_pos_attn, _ = self._other_pos_attn(other_pos_embed, other_pos_embed, other_pos_embed)
        other_pos_scores = self._other_pos_selector(other_pos_attn).squeeze(dim=-1)

        ind = other_pos_scores.topk(self._n_select).indices
        top_n_other_pos = other_pos_attn[th.arange(batch_size), ind.transpose(0,1)].transpose(0,1)
        return self._other_pos_final_embed(top_n_other_pos.flatten(1))

    def _extract_target_pos(self, target_pos, batch_size):
        target_pos_embed = self._pos_embedding_extractor(target_pos)
        target_pos_attn, _ = self._target_pos_attn(target_pos_embed, target_pos_embed, target_pos_embed)
        target_pos_scores = self._target_pos_selector(target_pos_attn).squeeze(dim=-1)

        ind = target_pos_scores.topk(self._n_select).indices
        top_n_target_pos = target_pos_attn[th.arange(batch_size), ind.transpose(0,1)].transpose(0,1)
        return self._target_pos_final_embed(top_n_target_pos.flatten(1))
        
