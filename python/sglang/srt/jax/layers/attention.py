from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from sglang.srt.jax.model_executor.forward_batch_info import ForwardBatch, ForwardMode


class Attention(nnx.Module):
    """Attention layer for variable-length sequences using ForwardBatch."""

    def __init__(self,
                 num_heads: int,
                 # add kv_heads for GQA attention and MQA attention
                 num_kv_heads: Optional[int] = None,
                 scale: float = None,
                 use_dot_product_attention: bool = False,
                 rngs: nnx.Rngs = None):
        self.scale = scale
        self.num_heads = num_heads
        if num_kv_heads is not None:
            self.num_kv_heads = num_kv_heads
        else:
            self.num_kv_heads = num_heads
        self.use_dot_product_attention = use_dot_product_attention

    def __call__(self,
                 q: jax.Array,
                 k: jax.Array,
                 v: jax.Array,
                 forward_batch: ForwardBatch,
                 layer_id: int,
                 attention_mask: jax.Array = None,
                 is_causal: bool = True):
        """
        Args:
            q, k, v: Input tensors of shape [total_tokens, hidden_size]
            forward_batch: ForwardBatch object containing seq_lens and batch_size
            attention_mask: Optional attention mask
            is_causal: Whether to apply causal masking
        Returns:
            Output tensor of shape [total_tokens, hidden_size]
        """
        return self._attn(q, k, v, forward_batch, layer_id, attention_mask, is_causal)

    def _attn(self, q, k, v, forward_batch: ForwardBatch, layer_id: int, attention_mask=None, is_causal=True):
        """
        Enhanced attention implementation with simple backend switching.

        Args:
            q, k, v: Input tensors of shape [total_tokens, hidden_size]
            forward_batch: ForwardBatch object containing seq_lens and batch_size
            attention_mask: Optional attention mask
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor of shape [total_tokens, hidden_size]
        """
        if self.use_dot_product_attention:
            return self._forward_dot_product_attention(q, k, v, forward_batch, attention_mask, is_causal)
        else:
            if forward_batch.forward_mode == ForwardMode.DECODE:
                k_buffer, v_buffer = self._get_and_set_kv_cache(
                    q, k, v, forward_batch, layer_id)
                return self._forward_native_decode(q, k_buffer, v_buffer, forward_batch.seq_lens, attention_mask)
            else:
                # update kv cache
                self._get_and_set_kv_cache(q, k, v, forward_batch, layer_id)
                return self._forward_native_extend(q, k, v, forward_batch.seq_lens, attention_mask, is_causal)

    def _prepare_tensors(self, q, k, v, forward_batch: ForwardBatch):
        """
        Prepare tensors by padding from [total_tokens, hidden_size] to [batch_size, max_seq_len, hidden_size]
        and reshaping for multi-head attention.

        Returns:
            tuple: (q_reshaped, k_reshaped, v_reshaped, batch_size, max_seq_len, hidden_size, head_dim)
        """
        seq_lengths = forward_batch.seq_lens
        batch_size = len(seq_lengths)
        max_seq_len = int(jnp.max(seq_lengths))
        hidden_size = q.shape[-1]
        kv_size = k.shape[-1]
        head_dim = hidden_size // self.num_heads

        def pad_tensor(tensor, size):
            """Pad tensor from [total_tokens, hidden_size] to [batch_size, max_seq_len, hidden_size]"""
            padded = jnp.zeros(
                (batch_size, max_seq_len, size), dtype=tensor.dtype)

            start_idx = 0
            for i in range(batch_size):
                seq_len = seq_lengths[i]
                end_idx = start_idx + seq_len
                padded = padded.at[i, :seq_len].set(tensor[start_idx:end_idx])
                start_idx = end_idx

            return padded

        # Pad all tensors
        q_padded = pad_tensor(q, hidden_size)
        k_padded = pad_tensor(k, kv_size)
        v_padded = pad_tensor(v, kv_size)

        # Reshape for multi-head attention: [batch_size, max_seq_len, num_heads, head_dim]
        q_reshaped = q_padded.reshape(
            batch_size, max_seq_len, self.num_heads, head_dim)
        k_reshaped = k_padded.reshape(
            batch_size, max_seq_len, self.num_kv_heads, head_dim)
        v_reshaped = v_padded.reshape(
            batch_size, max_seq_len, self.num_kv_heads, head_dim)

        return q_reshaped, k_reshaped, v_reshaped, batch_size, max_seq_len, hidden_size, head_dim

    def _unpad_output(self, attn_output, forward_batch: ForwardBatch, batch_size, max_seq_len, hidden_size):
        """
        Unpad output from [batch_size, max_seq_len, hidden_size] back to [total_tokens, hidden_size]
        """
        seq_lengths = forward_batch.seq_lens

        # Reshape to [batch_size, max_seq_len, hidden_size]
        attn_flat = attn_output.reshape(batch_size, max_seq_len, hidden_size)

        # Unpad back to [total_tokens, hidden_size]
        result_tokens = []
        for i in range(batch_size):
            seq_len = seq_lengths[i]
            result_tokens.append(attn_flat[i, :seq_len])

        return jnp.concatenate(result_tokens, axis=0)

    def _forward_dot_product_attention(self, q, k, v, forward_batch: ForwardBatch, attention_mask=None, is_causal=True):
        """
        Forward pass using JAX's dot_product_attention.

        Args:
            q, k, v: Input tensors of shape [total_tokens, hidden_size]
            forward_batch: ForwardBatch object containing seq_lens and batch_size
            attention_mask: Optional attention mask
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor of shape [total_tokens, hidden_size]
        """
        seq_lengths = forward_batch.seq_lens

        # Set scale
        if self.scale is None:
            scale = 1.0 / jnp.sqrt(q.shape[-1] // self.num_heads)
        else:
            scale = self.scale

        # Prepare tensors
        q_reshaped, k_reshaped, v_reshaped, batch_size, max_seq_len, hidden_size, _ = self._prepare_tensors(
            q, k, v, forward_batch)

        # Apply JAX's dot_product_attention
        attn_output = jax.nn.dot_product_attention(
            q_reshaped, k_reshaped, v_reshaped,
            mask=attention_mask,
            is_causal=is_causal,
            scale=scale,
            query_seq_lengths=seq_lengths,
            key_value_seq_lengths=seq_lengths
        )

        return self._unpad_output(attn_output, forward_batch, batch_size, max_seq_len, hidden_size)

    def _forward_native_extend(self, q, k, v, seq_lengths: jax.Array, attention_mask=None, is_causal=True):
        """
        Forward pass using native JAX implementation with block-diagonal attention.
        This avoids padding while maintaining efficient matrix operations.

        Args:
            q, k, v: Input tensors of shape [total_tokens, hidden_size]
            seq_length: shape (batch_size,)
            attention_mask: Optional attention mask
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor of shape [total_tokens, hidden_size]
        """
        total_tokens, hidden_size = q.shape
        head_dim = hidden_size // self.num_heads

        # Set scale
        if self.scale is None:
            scale = 1.0 / jnp.sqrt(head_dim)
        else:
            scale = self.scale

        # Reshape to multi-head format: [total_tokens, num_heads, head_dim]
        # For GQA attention, num_heads is the number of query heads, num_kv_heads is the number of key/value heads
        q_heads = q.reshape(total_tokens, self.num_heads, head_dim)
        k_heads = k.reshape(total_tokens, self.num_kv_heads, head_dim)
        v_heads = v.reshape(total_tokens, self.num_kv_heads, head_dim)

        # For GQA attention, we need to copy k and v heads to match the number of query heads
        num_copies = self.num_heads // self.num_kv_heads
        # Use repeat to copy k and v heads
        # [total_tokens, num_kv_heads, head_dim] -> [total_tokens, num_heads, head_dim]
        k_heads = jnp.repeat(k_heads, num_copies, axis=1)
        v_heads = jnp.repeat(v_heads, num_copies, axis=1)

        # Transpose for efficient matrix operations
        # [num_heads, total_tokens, head_dim]
        q_t = jnp.transpose(q_heads, (1, 0, 2))
        # [num_heads, total_tokens, head_dim]
        k_t = jnp.transpose(k_heads, (1, 0, 2))
        # [num_heads, total_tokens, head_dim]
        v_t = jnp.transpose(v_heads, (1, 0, 2))

        # Compute full attention weights in one operation: [num_heads, total_tokens, total_tokens]
        attn_weights = jnp.einsum("hqd,hkd->hqk", q_t, k_t) * scale

        # Create block-diagonal mask for sequences
        # This ensures tokens only attend to tokens within their own sequence
        seq_mask = self._create_extend_sequence_mask(seq_lengths)
        seq_mask = seq_mask[None, :, :]  # [1, total_tokens, total_tokens]

        # Apply sequence mask (set inter-sequence attention to -inf)
        mask_value = jnp.finfo(attn_weights.dtype).min
        attn_weights = jnp.where(seq_mask, attn_weights, mask_value)

        # Apply causal mask if needed
        if is_causal:
            causal_mask = self._create_causal_mask(seq_lengths)
            # [1, total_tokens, total_tokens]
            causal_mask = causal_mask[None, :, :]
            attn_weights = jnp.where(causal_mask, attn_weights, mask_value)

        # Apply custom attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[None, :, :]

        # Softmax
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Compute output in one operation: [num_heads, total_tokens, head_dim]
        attn_output = jnp.matmul(attn_weights, v_t)

        # Transpose back: [total_tokens, num_heads, head_dim]
        attn_output = jnp.transpose(attn_output, (1, 0, 2))

        # Reshape to original format: [total_tokens, hidden_size]
        return attn_output.reshape(total_tokens, hidden_size)

    def _forward_native_decode(self, q, k_cache, v_cache, seq_lengths: jax.Array, attention_mask=None):
        """
        Forward pass using native JAX implementation with block-diagonal attention.
        This avoids padding while maintaining efficient matrix operations.

        Args:
            q: input token in decode mode, shape(batch_size, hidden_size), each batch has one token
            k_cache: prefix cache of key, shape(seq_len, hidden_size)
            v_cache: prefix cache of value, shape(seq_len, hidden_size)
            seq_lengths: shape(batch_size,)
            attention_mask: Optional attention mask
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor of shape[batch_size, hidden_size]
        """
        return forward_native_decode(q, k_cache, v_cache, seq_lengths, self.num_heads, self.num_kv_heads, self.scale, attention_mask)

    def _create_extend_sequence_mask(self, seq_lengths):
        """
        Create a block-diagonal mask that ensures tokens only attend within their sequence in extend mode.

        Returns:
            mask: [total_tokens, total_tokens] boolean mask (True for valid positions)
        """
        # Create position indices for each token
        token_seq_ids = []
        for seq_idx, seq_len in enumerate(seq_lengths):
            token_seq_ids.extend([seq_idx] * int(seq_len))

        token_seq_ids = jnp.array(token_seq_ids)

        # Create mask: tokens can only attend to tokens in the same sequence
        seq_mask = token_seq_ids[:, None] == token_seq_ids[None, :]

        return seq_mask

    def _create_causal_mask(self, seq_lengths):
        """
        Create a causal mask that respects sequence boundaries.

        Returns:
            mask: [total_tokens, total_tokens] boolean mask (True for valid positions)
        """
        # Create position indices within each sequence
        token_positions = []
        for seq_len in seq_lengths:
            token_positions.extend(list(range(int(seq_len))))

        token_positions = jnp.array(token_positions)

        # Create causal mask: tokens can only attend to previous tokens within the same sequence
        causal_mask = token_positions[:, None] >= token_positions[None, :]

        return causal_mask
    
    def _get_and_set_kv_cache(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        forward_batch: ForwardBatch,
        layer_id: int
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Get the kv cache from the forward batch.
        """
        k_buffer_list = []
        v_buffer_list = []
        for idx in range(forward_batch.batch_size):
            prefix_str = forward_batch.prefix_str[idx]
            k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_cache(
                prefix_str, layer_id)
            k_buffer_list.append(k_buffer)
            v_buffer_list.append(v_buffer)

        k_buffer = jnp.concatenate(k_buffer_list, axis=0)
        v_buffer = jnp.concatenate(v_buffer_list, axis=0)
        
        new_k_buffer, new_v_buffer = get_and_set_kv_cache(
            k, v, k_buffer, v_buffer, forward_batch.batch_size, forward_batch.max_seq_len, forward_batch.out_cache_loc, forward_batch.extend_start_loc, forward_batch.seq_lens, forward_batch.forward_mode)
        for idx in range(forward_batch.batch_size):
            prefix_str = forward_batch.prefix_str[idx]
            start_loc = forward_batch.max_seq_len * idx
            end_loc = start_loc + forward_batch.max_seq_len
            forward_batch.token_to_kv_pool.set_kv_cache(
                prefix_str, layer_id, new_k_buffer[start_loc:end_loc], new_v_buffer[start_loc:end_loc])
        if forward_batch.forward_mode == ForwardMode.DECODE:
            ranges = jnp.arange(forward_batch.batch_size, dtype=jnp.int32) * forward_batch.max_seq_len + forward_batch.out_cache_loc
            take_indices = jnp.concatenate(
                [jnp.arange(idx + 1) for idx in ranges])
            return jnp.take(take_indices, axis=0), jnp.take(take_indices, axis=0)

@partial(jax.jit, static_argnames=["batch_size", "max_seq_len", "forward_mode"])
def get_and_set_kv_cache(
    k: jax.Array,
    v: jax.Array,
    k_buffer: jax.Array,
    v_buffer: jax.Array,
    batch_size: int,
    max_seq_len: int,
    out_cache_loc: jax.Array,
    extend_start_loc: jax.Array,
    seq_lens: jax.Array,
    forward_mode: ForwardMode
):
    def decode_branch(carry):
        def loop_body(idx, carry):
            k, v, k_buffer, v_buffer, out_cache_loc, extend_start_loc, seq_lens = carry
            buffer_loc = max_seq_len * idx + out_cache_loc[idx]
            # 使用 dynamic_slice 提取单个元素，然后用 dynamic_update_slice 更新
            print(f'###### {k.shape[-1]=}')
            k_elem = jax.lax.dynamic_slice(k, (idx, 0), (1, k.shape[-1]))
            v_elem = jax.lax.dynamic_slice(v, (idx, 0), (1, v.shape[-1]))
            new_k_buffer = jax.lax.dynamic_update_slice(
                k_buffer, k_elem, (buffer_loc,))
            new_v_buffer = jax.lax.dynamic_update_slice(
                v_buffer, v_elem, (buffer_loc,))
            return new_k_buffer, new_v_buffer
        
        _, _, k_buffer, v_buffer, _, _, _ = jax.lax.fori_loop(
            0, batch_size, loop_body, carry)
        return k_buffer, v_buffer
    
    def extend_branch(carry):
        def loop_body(idx, carry):
            k, v, k_buffer, v_buffer, out_cache_loc, extend_start_loc, seq_lens = carry
            loc = extend_start_loc[idx]
            seq_len = seq_lens[idx]
            # 使用 dynamic_slice 替代动态索引
            key_ = jax.lax.dynamic_slice(k, (loc, 0), (seq_len, k.shape[-1]))
            value_ = jax.lax.dynamic_slice(v, (loc, 0), (seq_len, v.shape[-1]))
            # 使用 dynamic_update_slice 替代动态索引赋值
            start_pos = max_seq_len * idx
            new_k_buffer = jax.lax.dynamic_update_slice(
                k_buffer, key_, (start_pos,))
            new_v_buffer = jax.lax.dynamic_update_slice(
                v_buffer, value_, (start_pos,))
            return new_k_buffer, new_v_buffer
        
        _, _, k_buffer, v_buffer, _, _, _ = jax.lax.fori_loop(
            0, batch_size, loop_body, carry)
        return k_buffer, v_buffer
    
    init_carry = (k, v, k_buffer, v_buffer, out_cache_loc,
                  extend_start_loc, seq_lens)
    new_k_buffer, new_v_buffer = jax.lax.cond(
        jax.lax.eq(forward_mode == ForwardMode.DECODE,
                   jnp.
        decode_branch,
        extend_branch,
        init_carry
    )
    
    return new_k_buffer, new_v_buffer

@partial(jax.jit, static_argnames=["num_heads", "num_kv_heads"])
def forward_native_decode(q: jax.Array,
                          k_cache: jax.Array,
                          v_cache: jax.Array,
                          seq_lengths: jax.Array,
                          num_heads, num_kv_heads,
                          scale=None, attention_mask=None):
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
    batch_size, hidden_size = q.shape
    head_dim = hidden_size // num_heads

    # Set scale
    if scale is None:
        scale = 1.0 / jnp.sqrt(head_dim)

    # Reshape to multi-head format
    # q: [batch_size, num_heads, head_dim]
    # k, v: [total_prefix_len, num_heads, head_dim]
    q_heads = q.reshape(batch_size, num_heads, head_dim)
    k_heads = k_cache.reshape(
        *k_cache.shape[:1], num_kv_heads, head_dim)
    v_heads = v_cache.reshape(
        *v_cache.shape[:1], num_kv_heads, head_dim)

    # Transpose for efficient matrix operations
    # q: shape of (num_heads, batch_size, head_dim)
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

    # Compute full attention weights in one operation: [num_heads, batch_size, head_dim]
    attn_weights = jnp.einsum("hqd,hkd->hqk", q_t, k_t) * scale

    # Apply sequence mask (set inter-sequence attention to -inf)
    mask_value = jnp.finfo(attn_weights.dtype).min
    total_prefix_len = k_cache.shape[0]
    seq_starts = jnp.cumsum(jnp.concatenate(
        [jnp.array([0]), seq_lengths[:-1]]))
    seq_ends = seq_starts + seq_lengths
    all_positions = jnp.arange(total_prefix_len)
    seq_mask = ((all_positions[None, :] >= seq_starts[:, None]) &
                (all_positions[None, :] < seq_ends[:, None]))
    seq_mask = seq_mask[None, :, :]
    attn_weights = jnp.where(seq_mask, attn_weights, mask_value)

    # Apply custom attention mask if provided
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[None, :, :]

    # Softmax
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)

    # Compute output in one operation: [num_heads, batch_size, v_head_dim]
    attn_output = jnp.matmul(attn_weights, v_t)

    # Transpose back: [batch_size, num_heads, head_dim]
    attn_output = jnp.transpose(attn_output, (1, 0, 2))

    # Reshape to original format: [batch_size, hidden_size]
    return attn_output.reshape(batch_size, hidden_size)
