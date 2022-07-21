from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .layers import Mlp, CustomAttention, SelfAttention

import torch as th

class CrossAttention(nn.Module):
    """Attention mechanism to pay attention to multiple sources of data.

    Eg:
    SimpleSpreadEnv
    Cross attention between other agent locations and target location
    """

    def __init__(self, input_size, embedding_size, query_size, num_heads) -> None:
        super().__init__()

        self._self_attn1 = SelfAttention(input_size, embedding_size, num_heads)
        self._self_attn2 = SelfAttention(input_size, embedding_size, num_heads)

        self._cross_attn1 = CustomAttention(embedding_size, embedding_size, embedding_size+query_size, num_heads)
        self._cross_attn2 = CustomAttention(embedding_size, embedding_size, embedding_size+query_size, num_heads)

    def forward(self, obs1, obs2, query):
        c1, _ = self._self_attn1(obs1)
        c2, _ = self._self_attn2(obs2)

        m1, _ = self._cross_attn1(c1, th.cat([th.sum(c2, dim=-2).unsqueeze(-2), query], dim=-1))
        m2, _ = self._cross_attn2(c2, th.cat([th.sum(c1, dim=-2).unsqueeze(-2), query], dim=-1))

        return th.cat([m1, m2], dim=-1)
