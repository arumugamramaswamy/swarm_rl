from policy.layers.layers import Mlp
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
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
        

class AttnSelectorPretrained(BaseFeaturesExtractor):
    def __init__(
        self, observation_space, checkpoint_path, n_select, embedding_size=16, num_heads=4
    ):
        super().__init__(observation_space, 1)

        self._n_select = n_select

        self._pos_embedding = Mlp(2, embedding_size)

        self._other_pos_attn = nn.MultiheadAttention(embedding_size, num_heads)
        self._target_pos_attn = nn.MultiheadAttention(embedding_size, num_heads)

        self._other_pos_selector = Mlp(embedding_size, 1)
        self._target_pos_selector = Mlp(embedding_size, 1)

        ppo = PPO.load(checkpoint_path)
        self._features_extractor = ppo.policy.features_extractor
        for param in ppo.policy.features_extractor.parameters():
            param.req
        self._features_dim = self._features_extractor._features_dim

    def forward(self, obs):

        batch_size = obs["my_vel"].shape[0]

        other_pos_final = self._extract_other_pos(obs["other_pos"], batch_size)
        target_pos_final = self._extract_target_pos(obs["entity_pos"], batch_size)

        output = {x:y for x, y in obs.items()}
        output["other_pos"] = other_pos_final
        output["entity_pos"] = target_pos_final
        output["comm"] = output["comm"][:, :self._n_select-1]

        return self._features_extractor(output)

    def _extract_other_pos(self, other_pos, batch_size):
        other_pos_embed = self._pos_embedding(other_pos)
        other_pos_attn, _ = self._other_pos_attn(other_pos_embed, other_pos_embed, other_pos_embed)
        other_pos_scores = self._other_pos_selector(other_pos_attn).squeeze(dim=-1)

        ind = other_pos_scores.topk(self._n_select - 1).indices
        top_n_other_pos = other_pos[th.arange(batch_size), ind.transpose(0,1)].transpose(0,1)
        return top_n_other_pos 

    def _extract_target_pos(self, target_pos, batch_size):
        target_pos_embed = self._pos_embedding(target_pos)
        target_pos_attn, _ = self._target_pos_attn(target_pos_embed, target_pos_embed, target_pos_embed)
        target_pos_scores = self._target_pos_selector(target_pos_attn).squeeze(dim=-1)

        ind = target_pos_scores.topk(self._n_select).indices
        top_n_target_pos = target_pos[th.arange(batch_size), ind.transpose(0,1)].transpose(0,1)
        return top_n_target_pos
        
class Selector(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gym.spaces.Dict, n_select, embedding_size=16, num_heads=4
    ):
        super().__init__(observation_space, features_dim=1)

        self._n_select = n_select

        self._pos_embedding_extractor = Mlp(2, embedding_size)
        self._vel_embedding_extractor = Mlp(2, embedding_size)

        self._other_pos_attn = nn.MultiheadAttention(embedding_size, num_heads, batch_first=True)
        self._target_pos_attn = nn.MultiheadAttention(embedding_size, num_heads, batch_first=True)

        self._features_dim = 2 + 2 + 2 * n_select + 2 * (n_select - 1)

    def forward(self, obs) -> th.Tensor:

        batch_size = obs["my_vel"].shape[0]

        my_vel = obs["my_vel"]
        my_pos = obs["my_pos"]
        other_pos_selected = self._extract_other_pos(obs["other_pos"], batch_size).flatten(1)
        target_pos_selected = self._extract_target_pos(obs["entity_pos"], batch_size).flatten(1)

        output = th.cat([other_pos_selected, target_pos_selected, my_pos, my_vel], dim=-1)

        assert output.shape[-1] == self._features_dim
        return output

    def _extract_other_pos(self, other_pos, batch_size):
        other_pos_embed = self._pos_embedding_extractor(other_pos)
        attn, w = self._other_pos_attn(other_pos_embed, other_pos_embed, other_pos_embed)
        other_pos_scores = w.sum(-2)

        ind = other_pos_scores.topk(self._n_select - 1).indices
        top_n_other_pos = other_pos[th.arange(batch_size), ind.transpose(0,1)].transpose(0,1)
        return top_n_other_pos

    def _extract_target_pos(self, target_pos, batch_size):
        target_pos_embed = self._pos_embedding_extractor(target_pos)
        _, w = self._target_pos_attn(target_pos_embed, target_pos_embed, target_pos_embed)
        target_pos_scores = w.sum(-2)

        ind = target_pos_scores.topk(self._n_select).indices
        top_n_target_pos = target_pos[th.arange(batch_size), ind.transpose(0,1)].transpose(0,1)
        return top_n_target_pos
