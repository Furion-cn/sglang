from dataclasses import dataclass
from enum import IntEnum, auto
from typing import List, Optional

import jax
import jax.numpy as jnp

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


@dataclass
class PaddedForwardBatch:
    """Store all inputs of a forward pass with static shapes for JIT compilation."""
    
    # The forward mode
    forward_mode: ForwardMode
    # Maximum batch size for padding
    max_batch_size: int
    # Maximum total tokens for padding  
    max_total_tokens: int
    
    # Padded arrays with static shapes
    # The input ids [max_total_tokens] - padded with 0
    input_ids: jax.Array
    # The sequence length for each request [max_batch_size] - padded with 0
    seq_lens: jax.Array
    # cache loc [max_total_tokens] - padded with 0
    cache_loc: jax.Array
    # decode token position in kv cache [max_total_tokens] - padded with 0
    out_cache_loc: jax.Array
    # Position information [max_total_tokens] - padded with 0
    positions: jax.Array
    # Start position for each sequence in extend mode [max_batch_size] - padded with 0
    extend_start_loc: jax.Array
    
    # Actual sizes (not padded)
    actual_batch_size: int
    actual_total_tokens: int
    
    # Validity masks for filtering padding
    # Boolean mask for valid tokens [max_total_tokens]
    token_mask: jax.Array
    # Boolean mask for valid sequences [max_batch_size]  
    seq_mask: jax.Array
    
    # token to kv cache pool
    token_to_kv_pool: ReqToHashKVCachePool = None

    @classmethod
    def from_forward_batch(
        cls, 
        forward_batch: ForwardBatch,
        max_batch_size: int,
        max_total_tokens: int
    ) -> 'PaddedForwardBatch':
        """Convert a dynamic ForwardBatch to a static PaddedForwardBatch."""
        
        actual_batch_size = forward_batch.batch_size
        actual_total_tokens = forward_batch.input_ids.shape[0]
        
        # Validate sizes
        if actual_batch_size > max_batch_size:
            raise ValueError(f"Actual batch size {actual_batch_size} exceeds max {max_batch_size}")
        if actual_total_tokens > max_total_tokens:
            raise ValueError(f"Actual total tokens {actual_total_tokens} exceeds max {max_total_tokens}")
        
        # Pad input_ids
        pad_tokens = max_total_tokens - actual_total_tokens
        input_ids_padded = jnp.concatenate([
            forward_batch.input_ids,
            jnp.zeros(pad_tokens, dtype=forward_batch.input_ids.dtype)
        ])
        
        # Pad seq_lens
        pad_seqs = max_batch_size - actual_batch_size
        seq_lens_padded = jnp.concatenate([
            forward_batch.seq_lens,
            jnp.zeros(pad_seqs, dtype=forward_batch.seq_lens.dtype)
        ])
        
        # Pad cache_loc
        cache_loc_padded = jnp.concatenate([
            forward_batch.cache_loc,
            jnp.zeros(pad_tokens, dtype=forward_batch.cache_loc.dtype)
        ])
        
        # Pad out_cache_loc  
        out_cache_loc_padded = jnp.concatenate([
            forward_batch.out_cache_loc,
            jnp.zeros(pad_tokens, dtype=forward_batch.out_cache_loc.dtype)
        ])
        
        # Pad positions
        positions_padded = jnp.concatenate([
            forward_batch.positions,
            jnp.zeros(pad_tokens, dtype=forward_batch.positions.dtype)
        ]) if forward_batch.positions is not None else jnp.zeros(max_total_tokens, dtype=jnp.int32)
        
        # Pad extend_start_loc
        extend_start_loc_padded = jnp.concatenate([
            forward_batch.extend_start_loc,
            jnp.zeros(pad_seqs, dtype=forward_batch.extend_start_loc.dtype)
        ]) if forward_batch.extend_start_loc is not None else jnp.zeros(max_batch_size, dtype=jnp.int32)
        
        # Create validity masks
        token_mask = jnp.concatenate([
            jnp.ones(actual_total_tokens, dtype=jnp.bool_),
            jnp.zeros(pad_tokens, dtype=jnp.bool_)
        ])
        
        seq_mask = jnp.concatenate([
            jnp.ones(actual_batch_size, dtype=jnp.bool_),
            jnp.zeros(pad_seqs, dtype=jnp.bool_)
        ])
        
        return cls(
            forward_mode=forward_batch.forward_mode,
            max_batch_size=max_batch_size,
            max_total_tokens=max_total_tokens,
            input_ids=input_ids_padded,
            seq_lens=seq_lens_padded,
            cache_loc=cache_loc_padded,
            out_cache_loc=out_cache_loc_padded,
            positions=positions_padded,
            extend_start_loc=extend_start_loc_padded,
            actual_batch_size=actual_batch_size,
            actual_total_tokens=actual_total_tokens,
            token_mask=token_mask,
            seq_mask=seq_mask,
            token_to_kv_pool=forward_batch.token_to_kv_pool
        )
    
    def to_forward_batch(self) -> ForwardBatch:
        """Convert back to dynamic ForwardBatch for compatibility."""
        return ForwardBatch(
            forward_mode=self.forward_mode,
            batch_size=self.actual_batch_size,
            input_ids=self.input_ids[:self.actual_total_tokens],
            seq_lens=self.seq_lens[:self.actual_batch_size],
            cache_loc=self.cache_loc[:self.actual_total_tokens],
            out_cache_loc=self.out_cache_loc[:self.actual_total_tokens],
            positions=self.positions[:self.actual_total_tokens] if self.positions is not None else None,
            extend_start_loc=self.extend_start_loc[:self.actual_batch_size] if self.extend_start_loc is not None else None,
            token_to_kv_pool=self.token_to_kv_pool
        )
