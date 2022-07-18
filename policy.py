from collections import OrderedDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

import gym.spaces
import torch as th
import typing as T

OTHER_POS = 0
ENTITY_POS = 1

class CustomMeanEmbeddingsExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, keys, mean_keys, embedding_size=16):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        self.mean_keys = mean_keys
        self.keys = keys

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            # The observation key is a vector, flatten it if needed
            extractors[key] = Mlp(subspace.shape[-1], embedding_size)
            total_concat_size += embedding_size

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        for key in self.keys:
            extractor = self.extractors[key]
            if key in self.mean_keys:
                encoded_tensor_list.append(extractor(observations[key]).sum(dim=-2))
            else:
                encoded_tensor_list.append(extractor(observations[key]))

        result = th.cat(encoded_tensor_list, dim=-1)
        assert result.shape[-1] == self._features_dim
        return result

class CustomMeanEmbeddingsExtractorV2(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, keys, mean_keys, embedding_size=16):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        self.mean_keys = mean_keys
        self.keys = keys

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            # The observation key is a vector, flatten it if needed
            extractors[key] = OrderedDict({
                "linear": Mlp(subspace.shape[-1], embedding_size),
                "tanh": Mlp(subspace.shape[-1], embedding_size),
            })
            extractors[key] = nn.ModuleDict(extractors[key])

            total_concat_size += 2*embedding_size

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        for key in self.keys:
            extractors = self.extractors[key]
            extracted = []
            for extractor in extractors.values():
                if key in self.mean_keys:
                    extracted.append(extractor(observations[key]).sum(dim=-2))
                else:
                    extracted.append(extractor(observations[key]))
            encoded_tensor_list.append(th.cat(extracted, dim=-1))

        result = th.cat(encoded_tensor_list, dim=-1)
        assert result.shape[-1] == self._features_dim
        return result

class CustomAttentionMeanEmbeddingsExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, keys, mean_keys, embedding_size=16, num_heads=4):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        self.mean_keys = mean_keys
        self.keys = keys

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            # The observation key is a vector, flatten it if needed
            extractors[key] = Mlp(subspace.shape[-1], embedding_size)
            total_concat_size += embedding_size

        self.extractors = nn.ModuleDict(extractors)

        attn_heads = {}
        for mean_key in mean_keys:
            attn_heads[mean_key] = nn.MultiheadAttention(embedding_size, num_heads, batch_first=True)

        self.attn_heads = nn.ModuleDict(attn_heads)
        # Update the features dim manually
        self._features_dim = total_concat_size
        

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_dict = {}

        for key in self.keys:
            extractor = self.extractors[key]
            encoded_tensor_dict[key] = extractor(observations[key])


        my_vel = encoded_tensor_dict["my_vel"]
        assert len(my_vel.shape) == 2

        weighted_dict = {}
        for mean_key in self.mean_keys:
            attn_head = self.attn_heads[mean_key]
            query = th.reshape(my_vel, (my_vel.shape[0], 1, my_vel.shape[-1]))
            weighted_dict[mean_key], _ = attn_head(query, encoded_tensor_dict[mean_key], encoded_tensor_dict[mean_key])
            weighted_dict[mean_key] = weighted_dict[mean_key].squeeze(dim=-2)

        encoded_tensor_dict.update(weighted_dict)

        encoded_tensor_list = []
        for key in self.keys:
            encoded_tensor_list.append(encoded_tensor_dict[key])
            
        result = th.cat(encoded_tensor_list, dim=-1)
        assert result.shape[-1] == self._features_dim
        return result

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
    
class SelfAttentionSimpleSpread(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, embedding_size=16, num_heads=4):
        super().__init__(observation_space, features_dim=1)

        assert observation_space["other_pos"].shape[-1] == observation_space["entity_pos"].shape[-1]
        self._attn_head = CustomAttention(observation_space["other_pos"].shape[-1] + 1, embedding_size, 2, num_heads)

        self._features_dim = embedding_size
    
    def forward(self, observations) -> th.Tensor:

        other_pos = th.cat([observations["other_pos"], OTHER_POS*th.ones(observations["other_pos"].shape[:-1] + (1,))], dim=-1)
        entity_pos = th.cat([observations["entity_pos"], ENTITY_POS*th.ones(observations["entity_pos"].shape[:-1] + (1,))], dim=-1)

        my_vel = observations["my_vel"]
        query = th.reshape(my_vel, (my_vel.shape[0], 1, my_vel.shape[-1]))
        input_data = th.cat([other_pos, entity_pos], dim=-2)
        print(input_data)
        attn_output, _ = self._attn_head(input_data, query)

        assert attn_output.shape[-1] == self._features_dim
        return attn_output

class CustomAttentionMeanEmbeddingsExtractorSimpleSpread(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, keys, embedding_size=16, num_heads=4):
        super().__init__(observation_space, features_dim=1)

        self.keys = keys

        self._embedding_extractors = nn.ModuleDict({
            "my_pos": Mlp(observation_space["my_pos"].shape[-1], embedding_size),
            "my_vel": Mlp(observation_space["my_vel"].shape[-1], embedding_size)
        })

        self._attn_heads = nn.ModuleDict({
            "comm": CustomAttention(observation_space["comm"].shape[-1], embedding_size, 2*embedding_size, num_heads),
            "other_pos": CustomAttention(observation_space["other_pos"].shape[-1], embedding_size, embedding_size, num_heads),
            # input to query network: embedding of my vel

            "entity_pos": CustomAttention(observation_space["entity_pos"].shape[-1], embedding_size, 2*embedding_size, num_heads),
            # input to query network: concat(embedding of my vel, attn of other_pos)
        })
        self._features_dim = embedding_size*5

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_dict = {}

        for key, extractor in self._embedding_extractors.items():
            encoded_tensor_dict[key] = extractor(observations[key])

        my_vel = encoded_tensor_dict["my_vel"]
        assert len(my_vel.shape) == 2

        weighted_dict = {}
        key = "other_pos"
        attn_head = self._attn_heads[key]
        query = th.reshape(my_vel, (my_vel.shape[0], 1, my_vel.shape[-1]))
        weighted_dict[key], _ = attn_head(observations[key], query)

        for key in ["entity_pos", "comm"]:
            attn_head = self._attn_heads[key]
            query_vel = th.reshape(my_vel, (my_vel.shape[0], 1, my_vel.shape[-1]))
            query_other_agent_attn = weighted_dict["other_pos"]
            query = th.cat([query_vel, query_other_agent_attn], dim=-1)
            weighted_dict[key], _ = self._attn_heads[key](observations[key], query)

        for key in weighted_dict:
            weighted_dict[key] = weighted_dict[key].squeeze(dim=-2)

        encoded_tensor_dict.update(weighted_dict)

        encoded_tensor_list = []
        for key in self.keys:
            encoded_tensor_list.append(encoded_tensor_dict[key])
            
        result = th.cat(encoded_tensor_list, dim=-1)
        assert result.shape[-1] == self._features_dim
        return result
