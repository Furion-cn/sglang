import dataclasses
from typing import Union

import jax
from flax import nnx
from transformers import PretrainedConfig

from sglang.srt.jax.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch,LogitsMetadata
from jax import with_sharding_constraint, mesh_sharding, PartitionSpec


@dataclasses.dataclass
class LogitsProcessorOutput:
    logits: jax.Array = None


class LogitsProcessor(nnx.Module):
    """Logits processor for the model."""

    def __init__(self,
                 config: PretrainedConfig,
                 num_embeddings: int,
                 embedding_dim: int,
                 ):
        self.lm_head=self.param(
          'lm_head',
          nnx.with_partitioning(self.dense_init, (None, None)),
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