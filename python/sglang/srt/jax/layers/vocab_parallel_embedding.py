
from flax import linen as nn


class VocabParallelEmbedding(nn.Module):
    """Vocab parallel embedding."""

    num_embeddings: int
    embedding_dim: int

    @nn.compact
    def __call__(self):
        pass

class ParallelLMHead(VocabParallelEmbedding):
    """Parallel LM head."""

    @nn.compact
    def __call__(self):
        pass
