from dataclasses import dataclass
from enum import IntEnum, auto
from typing import List

import jax

from sglang.srt.jax.mem_cache.hash_kvcache import ReqToHashKVCachePool


class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    # It is also called "prefill" in common terminology.
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()


@dataclass
class ForwardBatch:
    """Store all inputs of a forward pass."""

    # The forward mode
    forward_mode: ForwardMode
    # The batch size
    batch_size: int
    # The input ids [total_tokens]
    input_ids: jax.Array
    # The sequence length for each request [batch_size]
    seq_lens: jax.Array
    # cache loc
    cache_loc: jax.Array
    # decode token position in kv cache
    out_cache_loc: jax.Array
    # Position information [total_tokens]
    positions: jax.Array = None
    # Start position for each sequence in extend mode [batch_size]
    extend_start_loc: jax.Array = None
    # token to kv cache pool
    token_to_kv_pool: ReqToHashKVCachePool = None

    def tree_flatten(self):
        children = (
            self.forward_mode,
            self.batch_size,
            self.input_ids,
            self.seq_lens,
            self.cache_loc,
            self.out_cache_loc,
            self.positions,
            self.extend_start_loc,
        )
        aux_data = (self.token_to_kv_pool,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (token_to_kv_pool,) = aux_data
        (
            forward_mode,
            batch_size,
            input_ids,
            seq_lens,
            cache_loc,
            out_cache_loc,
            positions,
            extend_start_loc,
        ) = children
        return cls(
            forward_mode=forward_mode,
            batch_size=batch_size,
            input_ids=input_ids,
            seq_lens=seq_lens,
            cache_loc=cache_loc,
            out_cache_loc=out_cache_loc,
            positions=positions,
            extend_start_loc=extend_start_loc,
            token_to_kv_pool=token_to_kv_pool,
        )

jax.tree_util.register_pytree_node_class(ForwardBatch)
