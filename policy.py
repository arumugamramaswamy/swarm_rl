from collections import OrderedDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

import gym.spaces
import torch as th
import typing as T


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
            extractors[key] = nn.Sequential(
                nn.Linear(
                    subspace.shape[-1],
                    embedding_size,
                ),
                nn.ReLU()
            )
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
                "linear":nn.Sequential(
                    nn.Linear(
                        subspace.shape[-1],
                        embedding_size,
                    ),
                    nn.ReLU()
                ),
                "tanh":nn.Sequential(
                    nn.Linear(
                        subspace.shape[-1],
                        embedding_size,
                    ),
                    nn.Tanh()
                )
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
            extractors[key] = nn.Sequential(
                nn.Linear(
                    subspace.shape[-1],
                    embedding_size,
                ),
                nn.ReLU()
            )
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

class CustomAttentionMeanEmbeddingsExtractorSimpleSpread(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, keys, embedding_size=16, num_heads=4):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        self.keys = keys
        total_size = 0

        self._embedding_extractor, size = self.create_embedding_extractor(observation_space, embedding_size)
        total_size += size

        self._attn_heads, size = self.create_attention_head(embedding_size, num_heads)
        total_size += size

        self._features_dim = total_size

    @staticmethod
    def create_embedding_extractor(observation_space, embedding_size) -> T.Tuple[nn.ModuleDict, int]:
        embedding_extractor = {}
        total_size = 0
        for k, subspace in observation_space.spaces.items():
            embedding_extractor[k] = nn.Sequential(
                    nn.Linear(
                        subspace.shape[-1],
                        embedding_size,
                    ),
                    nn.ReLU()
            )
            total_size += embedding_size
        return nn.ModuleDict(embedding_extractor), total_size

    @staticmethod
    def create_attention_head(embedding_size, num_heads) -> T.Tuple[nn.ModuleDict, int]:
        """Compute attention to agents and use this info to compute attention for targets and comm"""
        attn_heads = {}
        total_size = 0
        # compute attention to agents
        attn_heads["other_pos"] = nn.MultiheadAttention(embedding_size, num_heads, kdim=embedding_size, vdim=embedding_size, batch_first=True)

        # compute attention to targets and comm
        for key in ["entity_pos", "comm"]:
            attn_heads[key] = nn.MultiheadAttention(2*embedding_size, num_heads, kdim=embedding_size, vdim=embedding_size, batch_first=True)
            total_size += embedding_size

        return nn.ModuleDict(attn_heads), total_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_dict = {}

        for key, extractor in self._embedding_extractor.items():
            encoded_tensor_dict[key] = extractor(observations[key])

        my_vel = encoded_tensor_dict["my_vel"]
        assert len(my_vel.shape) == 2

        weighted_dict = {}

        # compute attention to other agents
        key = "other_pos"
        attn_head = self._attn_heads[key]
        query = th.reshape(my_vel, (my_vel.shape[0], 1, my_vel.shape[-1]))
        weighted_dict[key], _ = attn_head(query, encoded_tensor_dict[key], encoded_tensor_dict[key])

        for key in ["entity_pos", "comm"]:
            attn_head = self._attn_heads[key]
            my_vel = th.reshape(my_vel, (my_vel.shape[0], 1, my_vel.shape[-1]))
            other_agent_attn = weighted_dict["other_pos"]
            query = th.cat([my_vel, other_agent_attn], dim=-1)
            weighted_dict[key], _ = attn_head(query, encoded_tensor_dict[key], encoded_tensor_dict[key])

        for key in weighted_dict:
            weighted_dict[key] = weighted_dict[key].squeeze(dim=-2)

        encoded_tensor_dict.update(weighted_dict)

        encoded_tensor_list = []
        for key in self.keys:
            encoded_tensor_list.append(encoded_tensor_dict[key])
            
        result = th.cat(encoded_tensor_list, dim=-1)
        assert result.shape[-1] == self._features_dim
        return result
