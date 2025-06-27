import jax
import jax.numpy as jnp
from flax import nnx

from sglang.srt.jax.model_executor.forward_batch_info import ForwardBatch


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
        return self._attn(q, k, v, forward_batch, attention_mask, is_causal)

    def _attn(self, q, k, v, forward_batch: ForwardBatch, attention_mask=None, is_causal=True):
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
            return self._forward_native(q, k, v, forward_batch, attention_mask, is_causal)

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

    def _forward_native(self, q, k, v, forward_batch: ForwardBatch, attention_mask=None, is_causal=True):
        """
        Forward pass using native JAX implementation with block-diagonal attention.
        This avoids padding while maintaining efficient matrix operations.

        Args:
            q, k, v: Input tensors of shape [total_tokens, hidden_size]
            forward_batch: ForwardBatch object containing seq_lens and batch_size
            attention_mask: Optional attention mask
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor of shape [total_tokens, hidden_size]
        """
        seq_lengths = forward_batch.seq_lens
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
        seq_mask = self._create_sequence_mask(seq_lengths)
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

    def _create_sequence_mask(self, seq_lengths):
        """
        Create a block-diagonal mask that ensures tokens only attend within their sequence.

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
