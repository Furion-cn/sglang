import dataclasses

import jax
from flax import nnx

from sglang.srt.jax.layers.embeddings import Embed


@dataclasses.dataclass
class LogitsProcessorOutput:
    next_token_logits: jax.Array


class LogitsProcessor(nnx.Module):
    """Logits processor for the model."""
    _requires_weight_loading = False

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(
        self,
        hidden_states: jax.Array,
        lm_head: Embed,
    ) -> LogitsProcessorOutput:
        # hidden_states = with_sharding_constraint(
        #    hidden_states, PartitionSpec('data', None))
        # hidden_states.shape = [batch, sequence, hidden_size]
        logits = lm_head.attend(hidden_states[:, -1:, :])
        logits = logits[:, :, : self.vocab_size]
        return LogitsProcessorOutput(next_token_logits=logits)
