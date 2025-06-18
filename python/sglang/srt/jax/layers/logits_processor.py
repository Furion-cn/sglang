import dataclasses
from typing import Union

import jax
from flax import linen as nn
from jax import numpy as jnp
from transformers import PretrainedConfig

from sglang.srt.jax.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch,LogitsMetadata
from jax import with_sharding_constraint, mesh_sharding, PartitionSpec


@dataclasses.dataclass
class LogitsProcessorOutput:
    logits: jax.Array = None


class LogitsProcessor(nn.Module):
    """Logits processor for the model."""
    config: PretrainedConfig
    num_embeddings: int
    embedding_dim: int

    def setup(self):
        self.lm_head=self.param(
          'lm_head',
          nn.with_partitioning(self.dense_init, (None, None)),
          (self.embedding_dim, self.num_embeddings))

    def __call__(self,
                 input_ids: jax.Array,
                 hidden_states: jax.Array,
                 lm_head: VocabParallelEmbedding,
                 logits_metadata: Union[LogitsMetadata, ForwardBatch],) -> LogitsProcessorOutput:
        hidden_states=with_sharding_constraint(hidden_states, mesh_sharding(PartitionSpec('data', None)))
        logits = jnp.dot(hidden_states, self.lm_head)
        return LogitsProcessorOutput(
            logits=logits
        )