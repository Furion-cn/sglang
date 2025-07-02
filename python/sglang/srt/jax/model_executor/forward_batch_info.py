from dataclasses import dataclass
from enum import IntEnum, auto
from typing import List

import jax

from sglang.srt.jax.mem_cache.hash_kvcache import HashKVCache, ReqToHashKVCachePool


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
    # max seq len
    max_seq_len: int
    # The input ids [total_tokens]
    input_ids: jax.Array
    # The sequence length for each request [batch_size]
    seq_lens: jax.Array
    # decode token position in kv cache
    out_cache_loc: jax.Array
    # Position information [total_tokens]
    positions: jax.Array = None
    # Start position for each sequence in extend mode [batch_size]
    extend_start_loc: jax.Array = None
    # Total number of tokens across all sequences
    total_tokens: int = 0
    # sequences
    sequences: List[str] = None
    # prefix string
    prefix_str: List[str] = None
    # token to kv cache pool
    token_to_kv_pool: HashKVCache = None
    # current kv_cache
    current_kv_cache: List[ReqToHashKVCachePool] = None
