from torch import nn

class Mlp(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(
                input_size,
                output_size,
            ),
            nn.ReLU(),
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
