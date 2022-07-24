from .attention_policy import AttentionPolicyV1
from .policy import (
    CustomAttentionMeanEmbeddingsExtractorSimpleSpread,
    CrossAttentionSimpleSpread,
    CrossAttentionSimpleSpreadV2,
    CrossAttentionV2SimpleSpreadV2,
)

REGISTRY = dict(
    AttentionPolicyV1=AttentionPolicyV1,
    CustomAttentionMeanEmbeddingsExtractorSimpleSpread=CustomAttentionMeanEmbeddingsExtractorSimpleSpread,
    CrossAttentionSimpleSpread=CrossAttentionSimpleSpread,
    CrossAttentionSimpleSpreadV2=CrossAttentionSimpleSpreadV2,
    CrossAttentionV2SimpleSpreadV2=CrossAttentionV2SimpleSpreadV2,
)
