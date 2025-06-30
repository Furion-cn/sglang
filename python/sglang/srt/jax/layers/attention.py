import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, List

from sglang.srt.jax.model_executor.forward_batch_info import ForwardBatch, ForwardMode


class Attention(nnx.Module):
    """Attention layer for variable-length sequences using ForwardBatch."""

    def __init__(self,
                 num_heads: int,
                 scale: float = None,
                 use_dot_product_attention: bool = False,
                 rngs: nnx.Rngs = None):
        self.scale = scale
        self.num_heads = num_heads
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
                k_buffer, v_buffer = self._get_and_set_kv_cache(q, k, v, forward_batch, layer_id)
                return self._forward_native_decode(q, k_buffer, v_buffer, forward_batch.seq_lens, attention_mask)
            else:
                # update kv cache
                for idx, seq_len in enumerate(forward_batch.seq_lens):
                    extend_start_loc = forward_batch.extend_start_loc[idx]
                    forward_batch.current_kv_cache[idx].set_kv_buffer(
                        layer_id, k[extend_start_loc:extend_start_loc+seq_len], v[extend_start_loc:extend_start_loc+seq_len])
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
        head_dim = hidden_size // self.num_heads

        def pad_tensor(tensor):
            """Pad tensor from [total_tokens, hidden_size] to [batch_size, max_seq_len, hidden_size]"""
            padded = jnp.zeros(
                (batch_size, max_seq_len, hidden_size), dtype=tensor.dtype)

            start_idx = 0
            for i in range(batch_size):
                seq_len = seq_lengths[i]
                end_idx = start_idx + seq_len
                padded = padded.at[i, :seq_len].set(tensor[start_idx:end_idx])
                start_idx = end_idx

            return padded

        # Pad all tensors
        q_padded = pad_tensor(q)
        k_padded = pad_tensor(k)
        v_padded = pad_tensor(v)

        # Reshape for multi-head attention: [batch_size, max_seq_len, num_heads, head_dim]
        q_reshaped = q_padded.reshape(
            batch_size, max_seq_len, self.num_heads, head_dim)
        k_reshaped = k_padded.reshape(
            batch_size, max_seq_len, self.num_heads, head_dim)
        v_reshaped = v_padded.reshape(
            batch_size, max_seq_len, self.num_heads, head_dim)

        return q_reshaped, k_reshaped, v_reshaped, batch_size, max_seq_len, hidden_size

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
        q_reshaped, k_reshaped, v_reshaped, batch_size, max_seq_len, hidden_size = self._prepare_tensors(
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
            k_cache: cache of key, shape (seq_len, hidden_size)
            v_cache: cache of value, shape (seq_len, hidden_size)
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
        q_heads = q.reshape(total_tokens, self.num_heads, head_dim)
        k_heads = k.reshape(total_tokens, self.num_heads, head_dim)
        v_heads = v.reshape(total_tokens, self.num_heads, head_dim)

        # Transpose for efficient matrix operations: [num_heads, total_tokens, head_dim]
        q_t = jnp.transpose(q_heads, (1, 0, 2))
        k_t = jnp.transpose(k_heads, (1, 0, 2))
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
        batch_size, hidden_size = q.shape
        head_dim = hidden_size // self.num_heads

        # Set scale
        if self.scale is None:
            scale = 1.0 / jnp.sqrt(head_dim)
        else:
            scale = self.scale

        # Reshape to multi-head format
        # q: [batch_size, num_heads, head_dim]
        # k, v: [total_prefix_len, num_heads, head_dim]
        q_heads = q.reshape(batch_size, self.num_heads, head_dim)
        k_heads = k_cache.reshape(
            *k_cache.shape[:1], self.num_heads, head_dim)
        v_heads = v_cache.reshape(
            *v_cache.shape[:1], self.num_heads, head_dim)

        # Transpose for efficient matrix operations
        # q: shape of (num_heads, batch_size, head_dim)
        # k, v: shape of (total_prefix_len, num_heads, head_dim)
        
        q_t = jnp.transpose(q_heads, (1, 0, 2))
        k_t = jnp.transpose(k_heads, (1, 0, 2))
        v_t = jnp.transpose(v_heads, (1, 0, 2))

        # Compute full attention weights in one operation: [num_heads, batch_size, head_dim]
        attn_weights = jnp.einsum("hqd,hkd->hqk", q_t, k_t) * scale

        # Create block-diagonal mask for sequences
        # This ensures tokens only attend to tokens within their own sequence
        seq_mask = self._create_decode_sequence_mask(batch_size, seq_lengths)
        seq_mask = seq_mask[None, :, :]  # [1, batch_size, total_prefix_len]

        # Apply sequence mask (set inter-sequence attention to -inf)
        mask_value = jnp.finfo(attn_weights.dtype).min
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
    
    def _create_decode_sequence_mask(self, batch_size: int, seq_lengths):
        """
        Create a block-diagonal mask that ensures tokens only attend within their sequence in decode mode.

        Returns:
            mask: [total_tokens, total_tokens] boolean mask (True for valid positions)
        """
        # Create position indices for each token
        token_seq_ids = []
        for seq_idx, seq_len in enumerate(seq_lengths):
            token_seq_ids.extend([seq_idx] * int(seq_len))

        token_seq_ids = jnp.array(token_seq_ids)

        # Create mask: tokens can only attend to tokens in the same sequence
        seq_mask = jnp.arange(batch_size)[:, None] == token_seq_ids[None, :]

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
        for idx, prefix_str in enumerate(forward_batch.prefix_str):
            k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_cache(prefix_str, layer_id)
            new_k_buffer = jnp.concatenate([k_buffer, k[idx][jnp.newaxis, :]], axis=0)
            new_v_buffer = jnp.concatenate([v_buffer, v[idx][jnp.newaxis, :]], axis=0)
            k_buffer_list.append(new_k_buffer)
            v_buffer_list.append(new_v_buffer)
            forward_batch.current_kv_cache[idx].set_kv_buffer(layer_id, new_k_buffer, new_v_buffer)
        
        return jnp.concatenate(k_buffer_list, axis=0), jnp.concatenate(v_buffer_list, axis=0)
