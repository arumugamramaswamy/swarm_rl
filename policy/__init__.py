from .attention_policy import AttentionPolicyV1
from .policy import (
    CustomAttentionMeanEmbeddingsExtractorSimpleSpread,
    CrossAttentionSimpleSpread
)

REGISTRY = dict(
    AttentionPolicyV1=AttentionPolicyV1,
    CustomAttentionMeanEmbeddingsExtractorSimpleSpread=CustomAttentionMeanEmbeddingsExtractorSimpleSpread,
    CrossAttentionSimpleSpread=CrossAttentionSimpleSpread,
)
