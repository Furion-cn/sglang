import dataclasses
from typing import Optional, Sequence

import jax
from flax import nnx
from transformers import PretrainedConfig


from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec
from jax import numpy as jnp
from flax.typing import Sharding
from sglang.srt.jax.layers.embeddings import Embed


@dataclasses.dataclass
class LogitsProcessorOutput:
    logits: jax.Array


class LogitsProcessor(nnx.Module):
    """Logits processor for the model."""

    def __init__(self):
        pass

    def __call__(
        self,
        hidden_states: jax.Array,
        lm_head: Embed,
    ) -> LogitsProcessorOutput:
        # hidden_states = with_sharding_constraint(
        #    hidden_states, PartitionSpec('data', None))
        # hidden_states.shape = [batch, sequence, hidden_size]
        logits = lm_head.attend(hidden_states[:, -1:, :])
        return LogitsProcessorOutput(logits=logits)
