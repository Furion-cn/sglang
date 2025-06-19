import dataclasses
from typing import Union

import jax
from flax import nnx
from transformers import PretrainedConfig


from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from jax.lax import with_sharding_constraint
from jax.sharding import NamedSharding, PartitionSpec
from jax import numpy as jnp
from flax.typing import Sharding


@dataclasses.dataclass
class LogitsProcessorOutput:
    logits: jax.Array


class LogitsProcessor(nnx.Module):
    """Logits processor for the model."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        kernel_init: nnx.Initializer = nnx.initializers.lecun_normal(),
        kernel_partition: Sharding = (None, None),
        *,  # Following arguments are keyword-only
        rngs: nnx.Rngs,
    ):
        vocab_size = ((vocab_size + 63) // 64) * 64

        self.lm_head = nnx.Linear(
            hidden_size,
            vocab_size,
            kernel_init=nnx.with_partitioning(kernel_init, kernel_partition),
            use_bias=False,
            rngs=rngs
        )

    def __call__(
        self,
        hidden_states: jax.Array,
    ) -> LogitsProcessorOutput:
        logits = self.lm_head(hidden_states)
        return LogitsProcessorOutput(logits=logits)
