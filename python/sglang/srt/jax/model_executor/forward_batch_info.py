from enum import IntEnum, auto
from typing import List,Any
from flax import struct

import jax

from sglang.srt.jax.mem_cache.hash_kvcache import ReqToHashKVCachePool

@struct.dataclass
class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    # It is also called "prefill" in common terminology.
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()

FORWARD_MODE_EXTEND="extend"
FORWARD_MODE_DECODE="decode"

KCACHE=Any
VCACHE=Any

@struct.dataclass(frozen=False)
class ForwardBatch:
    """Store all inputs of a forward pass."""
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
    positions: jax.Array
    # Start position for each sequence in extend mode [batch_size]
    extend_start_loc: jax.Array
    # token to kv cache pool
    k_cache:jax.Array
    v_cache:jax.Array

