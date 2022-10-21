from .attention_policy import AttentionPolicyV1, AttnV2
from .policy import (
    CustomAttentionMeanEmbeddingsExtractorSimpleSpread,
    CrossAttentionSimpleSpread,
    CrossAttentionSimpleSpreadV2,
    CrossAttentionV2SimpleSpreadV2,
    Flattener
)
from .bootstrapping_policy import BootstrappingFE
from .attn_selector_policy import AttnSelectorPolicy, Selector
from .me_policy import MeanEmbeddingExtractor

REGISTRY = dict(
    AttentionPolicyV1=AttentionPolicyV1,
    CustomAttentionMeanEmbeddingsExtractorSimpleSpread=CustomAttentionMeanEmbeddingsExtractorSimpleSpread,
    CrossAttentionSimpleSpread=CrossAttentionSimpleSpread,
    CrossAttentionSimpleSpreadV2=CrossAttentionSimpleSpreadV2,
    CrossAttentionV2SimpleSpreadV2=CrossAttentionV2SimpleSpreadV2,
    BootstrappingFE=BootstrappingFE,
    AttnSelectorPolicy=AttnSelectorPolicy,
    Flattener=Flattener,
    Selector=Selector,
    AttnV2=AttnV2,
    MeanEmbeddingExtractor=MeanEmbeddingExtractor,
)
