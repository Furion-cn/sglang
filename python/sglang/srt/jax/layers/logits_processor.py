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


@dataclasses.dataclass
class LogitsProcessorOutput:
    logits: jax.Array


class LogitsProcessor(nnx.Module):
    """Logits processor for the model."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        kernel_axes: Optional[Sequence[str]] = None,
        rngs: nnx.Rngs = None,

    ):
        if kernel_axes is not None:
            kernel_axes = tuple(axis for axis in kernel_axes if axis is not None)
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(),kernel_axes)
        else:
            kernel_init=nnx.initializers.lecun_normal()

        self.lm_head = nnx.Linear(
            hidden_size,
            vocab_size,
            kernel_init=kernel_init,
            use_bias=False,
            rngs=rngs
        )

    def __call__(
        self,
        hidden_states: jax.Array,
    ) -> LogitsProcessorOutput:
        hidden_states = with_sharding_constraint(
            hidden_states, PartitionSpec('data', None))
        logits = self.lm_head(hidden_states)
        return LogitsProcessorOutput(logits=logits)
