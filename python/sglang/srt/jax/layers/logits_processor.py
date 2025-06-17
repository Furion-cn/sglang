import dataclasses
from typing import Union

import jax
from flax import linen as nn
from jax import numpy as jnp
from transformers import PretrainedConfig

from sglang.srt.jax.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclasses.dataclass
class LogitsMetadata:
    pass


@dataclasses.dataclass
class LogitsProcessorOutput:
    pass


class LogitsProcessor(nn.Module):
    """Logits processor for the model."""

    config: PretrainedConfig

    @nn.compact
    def __call__(self,
                 input_ids: jax.Array,
                 hidden_states: jax.Array,
                 lm_head: VocabParallelEmbedding,
                 logits_metadata: Union[LogitsMetadata, ForwardBatch],) -> LogitsProcessorOutput:
        pass
