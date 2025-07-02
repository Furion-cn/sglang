import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx

from sglang.srt.jax.layers.embeddings import Embed
from sglang.srt.jax.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from functools import partial


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
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        # hidden_states = with_sharding_constraint(
        #    hidden_states, PartitionSpec('data', None))
        # hidden_states.shape = [total_tokens, hidden_size]

        # Extract the last token of each sequence based on seq_lens
        # forward_batch.extend_start_loc gives start position of each sequence
        # forward_batch.seq_lens gives length of each sequence
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            last_token_indices = forward_batch.extend_start_loc + forward_batch.seq_lens - 1
        else:
            last_token_indices = jnp.arange(forward_batch.batch_size)
        # Shape: [batch_size, hidden_size]
        last_hidden_states = hidden_states[last_token_indices]

        logits = lm_head.attend(last_hidden_states)
        logits = logits[:,
                        :self.vocab_size] if logits.ndim > 1 else logits[:self.vocab_size]
        return LogitsProcessorOutput(next_token_logits=logits)

@partial(jax.jit)
def __logits_processor_forward():
    pass