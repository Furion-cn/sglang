import jax
from flax import nnx


class VocabParallelEmbedding(nnx.Module):
    """Vocab parallel embedding."""

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 rngs: nnx.Rngs = None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def __call__(self, x: jax.Array):
        return x


class ParallelLMHead(VocabParallelEmbedding):
    """Parallel LM head."""

    def __call__(self):
        pass
