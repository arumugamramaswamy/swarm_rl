from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

import gym.spaces
import torch as th


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
