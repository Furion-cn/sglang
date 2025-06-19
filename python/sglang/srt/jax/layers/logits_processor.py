import dataclasses
from typing import Optional, Sequence

import jax
from flax import nnx


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
        vocab_size = ((vocab_size + 63) // 64) * 64

        self.lm_head = nnx.Linear(
            hidden_size,
            vocab_size,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), kernel_axes),
            use_bias=False,
            rngs=rngs
        )

    def __call__(
        self,
        hidden_states: jax.Array,
    ) -> LogitsProcessorOutput:
        logits = self.lm_head(hidden_states)
        return LogitsProcessorOutput(logits=logits)
