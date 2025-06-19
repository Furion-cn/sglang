from flax import nnx


class VocabParallelEmbedding(nnx.Module):
    """Vocab parallel embedding."""

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def __call__(self):
        pass


class ParallelLMHead(VocabParallelEmbedding):
    """Parallel LM head."""

    def __call__(self):
        pass
