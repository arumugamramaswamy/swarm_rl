from torch import nn

import torch as th
import torch.nn.functional as F
import numpy as np
import typing as T

class Mlp(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(
                input_size,
                output_size,
            ),
            nn.Tanh(),
        )
    def forward(self, x):
        return self._mlp(x)

class CustomAttention(nn.Module):
    def __init__(self, input_size, embedding_size, query_size, num_heads) -> None:
        super().__init__()
        self._key_extractor = Mlp(input_size, embedding_size)
        self._value_extractor = Mlp(input_size, embedding_size)
        self._query_extractor = Mlp(query_size, embedding_size)
        self._attn = nn.MultiheadAttention(
            embedding_size,
            num_heads,
            batch_first=True,
        )

    def forward(self, x, query):
        k = self._key_extractor(x)
        v = self._value_extractor(x)
        q = self._query_extractor(query)
        return self._attn(q, k, v)

class SelfAttention(CustomAttention):
    def __init__(self, input_size, embedding_size, num_heads) -> None:
        super().__init__(input_size, embedding_size, input_size, num_heads)

    def forward(self, x):
        return super().forward(x, x)

class ScaledDotProductAttention(nn.Module):
    """
    Implementation from: https://github.com/sooftware/attentions/blob/master/attentions.py

    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim
        dim (int): dimention of attention
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: th.Tensor, key: th.Tensor, value: th.Tensor) -> T.Tuple[th.Tensor, th.Tensor]:
        score = th.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        attn = F.softmax(score, -1)
        context = th.bmm(attn, value)
        return context, attn

class ScaledDotProductAttentionWithExtractors(nn.Module):
    def __init__(self, d_model, q_dim, k_dim, v_dim) -> None:
        super().__init__()

        self._q_extractor = Mlp(q_dim, d_model)
        self._k_extractor = Mlp(k_dim, d_model)
        self._v_extractor = Mlp(v_dim, d_model)
        self._attn = ScaledDotProductAttention(d_model)

    def forward(self, q, k, v):
        return self._attn(
            self._q_extractor(q),
            self._k_extractor(k),
            self._v_extractor(v),
        )
