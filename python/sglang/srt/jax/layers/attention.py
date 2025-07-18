from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional

from sglang.srt.jax.model_executor.forward_batch_info import ForwardBatch, ForwardMode,FORWARD_MODE_DECODE,FORWARD_MODE_EXTEND
from sglang.srt.jax.mem_cache.hash_kvcache import get_kv_buffer,set_kv_buffer
from sglang.debug_tracer import global_tracer, trace_function


class Attention(nnx.Module):
    """Attention layer for variable-length sequences using ForwardBatch."""

    def __init__(self,
                 num_heads: int,
                 # add kv_heads for GQA attention and MQA attention
                 num_kv_heads: Optional[int] = None,
                 scale: float = None,
                 rngs: nnx.Rngs = None):
        self.scale = scale
        self.num_heads = num_heads
        if num_kv_heads is not None:
            self.num_kv_heads = num_kv_heads
        else:
            self.num_kv_heads = num_heads
        
    @trace_function(stage="INTERNAL_ATTENTION", include_args=False, include_output=True)
    def __call__(self,
                 q: jax.Array,
                 k: jax.Array,
                 v: jax.Array,
                 forward_batch: ForwardBatch,
                 layer_id: int,
                 attention_mask: jax.Array = None,
                 is_causal: bool = True,
                 forward_mode:str=None,
                 ):
        """
        Args:
            q, k, v: Input tensors of shape [total_tokens, hidden_size]
            forward_batch: ForwardBatch object containing seq_lens and batch_size
            attention_mask: Optional attention mask
            is_causal: Whether to apply causal masking
        Returns:
            Output tensor of shape [total_tokens, hidden_size]
        """

        k_buffer, v_buffer, forward_batch = self._get_and_set_kv_cache(
            q, k, v, forward_batch, layer_id,forward_mode)

        head_dim = q.shape[1] // self.num_heads

        if self.scale is None:
            scale = 1.0 / jnp.sqrt(head_dim)
        else:
            scale = self.scale

        if forward_mode == FORWARD_MODE_DECODE:
            is_causal = False

        return forward_attention(q, k_buffer, v_buffer, forward_batch.seq_lens, forward_batch.cache_loc, self.num_heads, self.num_kv_heads, scale, attention_mask, is_causal, forward_mode), forward_batch
    @trace_function(stage="INTERNAL_ATTENTION_GET_AND_SET_KV_CACHE", include_args=False, include_output=True)
    def _get_and_set_kv_cache(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        forward_batch: ForwardBatch,
        layer_id: int,
        forward_mode:str,
    ):
        """
        Get the kv cache from the forward batch.
        """
        #k_cache,v_cache=get_kv_buffer(forward_batch.k_cache,forward_batch.v_cache,layer_id)
        if forward_mode ==FORWARD_MODE_EXTEND:
            forward_batch.k_cache,forward_batch.v_cache=set_kv_buffer(layer_id, forward_batch.cache_loc, k, v,forward_batch.k_cache,forward_batch.v_cache)
        else:
            forward_batch.k_cache,forward_batch.v_cache=set_kv_buffer(layer_id, forward_batch.out_cache_loc, k, v,forward_batch.k_cache,forward_batch.v_cache) 
        k_buffer,v_buffer=get_kv_buffer(forward_batch.k_cache,forward_batch.v_cache,layer_id)
        return k_buffer,v_buffer, forward_batch

@trace_function(stage="INTERNAL_ATTENTION_FORWARD_ATTENTION", include_args=True, include_output=True)
def forward_attention(q: jax.Array,
                      k_cache: jax.Array,
                      v_cache: jax.Array,
                      seq_lengths: jax.Array,
                      loc: jax.Array,
                      num_heads, num_kv_heads,
                      scale=None, attention_mask=None, is_causal=True, mode=FORWARD_MODE_DECODE):
    """
    Forward pass using native JAX implementation with block-diagonal attention.
    This avoids padding while maintaining efficient matrix operations.

    Args:
        q: input token in decode mode, shape(batch_size, hidden_size), each batch has one token
        k_cache: prefix cache of key, shape(seq_len, hidden_size)
        v_cache: prefix cache of value, shape(seq_len, hidden_size)
        num_heads: number of query heads
        num_kv_heads: number of key/value heads
        attention_mask: Optional attention mask
        scale: scale for the attention weights
        attention_mask: Optional attention mask
        seq_mask: boolean mask of shape [batch_size, total_prefix_len]

    Returns:
        Output tensor of shape[batch_size, hidden_size]
    """
    k_cache = jnp.take(k_cache, loc, axis=0)
    v_cache = jnp.take(v_cache, loc, axis=0)

    num_tokens, hidden_size = q.shape
    head_dim = hidden_size // num_heads

    # Reshape to multi-head format
    # q: [num_tokens, num_heads, head_dim]
    # k, v: [total_prefix_len, num_heads, head_dim]
    q_heads = q.reshape(num_tokens, num_heads, head_dim)
    k_heads = k_cache.reshape(k_cache.shape[0], num_kv_heads, head_dim)
    v_heads = v_cache.reshape(v_cache.shape[0], num_kv_heads, head_dim)

    # Transpose for efficient matrix operations
    # q: shape of (num_heads, num_tokens, head_dim)
    # k, v: shape of (total_prefix_len, num_heads, head_dim)

    # For GQA attention, we need to copy k and v heads to match the number of query heads
    num_copies = num_heads // num_kv_heads
    # Use repeat to copy k and v heads
    # [total_prefix_len, num_kv_heads, head_dim] -> [total_prefix_len, num_heads, head_dim]
    k_heads = jnp.repeat(k_heads, num_copies, axis=1)
    v_heads = jnp.repeat(v_heads, num_copies, axis=1)

    q_t = jnp.transpose(q_heads, (1, 0, 2))
    k_t = jnp.transpose(k_heads, (1, 0, 2))
    v_t = jnp.transpose(v_heads, (1, 0, 2))

    # Compute full attention weights in one operation: [num_heads, num_tokens, head_dim]
    attn_weights = jnp.einsum("hqd,hkd->hqk", q_t, k_t) * scale

    # Apply sequence mask
    attn_weights = _apply_sequence_mask(
        attn_weights, seq_lengths, mode=mode)

    # Apply custom attention mask if provided
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[None, :, :]

    if is_causal:
        attn_weights = _apply_causal_mask(attn_weights, seq_lengths)

    # Softmax
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)

    attn_output = jnp.matmul(attn_weights, v_t)
    attn_output = jnp.transpose(attn_output, (1, 0, 2))
    return attn_output.reshape(num_tokens, hidden_size)


@trace_function(stage="INTERNAL_ATTENTION_APPLY_SEQUENCE_MASK", include_args=True, include_output=True)
def _apply_sequence_mask(attn_weights: jax.Array, seq_lengths: jax.Array, mode: str):
    """Create a sequence mask that ensures tokens only attend within their sequence."""
    batch_size = seq_lengths.shape[0]
    _, query_len, key_len = attn_weights.shape

    def create_extend_sequence_mask():
        q_positions = jnp.arange(query_len)
        k_positions = jnp.arange(key_len)

        def scan_fn(carry, i):
            start_pos = carry
            seq_len = seq_lengths[i]
            return start_pos + seq_len, (start_pos, start_pos + seq_len)

        _, boundaries = jax.lax.scan(scan_fn, 0, jnp.arange(batch_size))
        seq_starts, seq_ends = boundaries

        def assign_batch_id(pos):
            in_seq = (pos >= seq_starts) & (pos < seq_ends)
            batch_id = jnp.sum(jnp.arange(batch_size) * in_seq)
            valid = jnp.any(in_seq)
            return jnp.where(valid, batch_id, -1)

        q_batch_ids = jax.vmap(assign_batch_id)(q_positions)
        k_batch_ids = jax.vmap(assign_batch_id)(k_positions)

        seq_mask = (q_batch_ids[:, None] == k_batch_ids[None, :]) & (
            q_batch_ids[:, None] >= 0)
        return seq_mask

    def create_decode_sequence_mask():
        total_prefix_len = key_len
        seq_starts = jnp.cumsum(jnp.concatenate(
            [jnp.array([0]), seq_lengths[:-1]]))
        seq_ends = seq_starts + seq_lengths
        all_positions = jnp.arange(total_prefix_len)
        seq_mask = ((all_positions[None, :] >= seq_starts[:, None]) &
                    (all_positions[None, :] < seq_ends[:, None]))
        return seq_mask

    if mode == FORWARD_MODE_EXTEND:
        mask = create_extend_sequence_mask()
    else:
        mask = create_decode_sequence_mask()

    mask_value = jnp.finfo(attn_weights.dtype).min
    mask = mask[None, :, :]
    return jnp.where(mask, attn_weights, mask_value)

@trace_function(stage="INTERNAL_ATTENTION_APPLY_CAUSAL_MASK", include_args=False, include_output=True)
def _apply_causal_mask(attn_weights: jax.Array, seq_lengths: jax.Array):
    """Create a causal mask."""
    _, query_len, key_len = attn_weights.shape

    # Create position matrices
    q_positions = jnp.arange(query_len)
    k_positions = jnp.arange(key_len)

    # Simple causal mask: query position >= key position
    causal_mask = q_positions[:, None] >= k_positions[None, :]

    mask_value = jnp.finfo(attn_weights.dtype).min
    causal_mask = causal_mask[None, :, :]
    return jnp.where(causal_mask, attn_weights, mask_value)