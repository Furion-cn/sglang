import jax
import jax.numpy as jnp
from flax import nnx


class Attention(nnx.Module):
    """attention layer."""

    def __init__(self,
                 num_heads: int,    
                 scale: float = None,
                 rngs: nnx.Rngs = None):
        self.scale = scale
        self.num_heads = num_heads

    def __call__(self,
                 q: jax.Array,
                 k: jax.Array,
                 v: jax.Array,
                 attention_mask: jax.Array = None,
                 is_causal: bool = True):
        
        seq_len = q.shape[0]
        total_dim = q.shape[-1]
        head_dim = total_dim // self.num_heads

        # Reshape to [seq_len, num_heads, head_dim] for attention
        q_reshaped = q.reshape(seq_len, self.num_heads, head_dim)
        k_reshaped = k.reshape(seq_len, self.num_heads, head_dim)
        v_reshaped = v.reshape(seq_len, self.num_heads, head_dim)

        # Transpose to [1, num_heads, seq_len, head_dim] for scaled_dot_product_attention
        q_attn = q_reshaped.swapaxes(0, 1)[None, :, :, :]
        k_attn = k_reshaped.swapaxes(0, 1)[None, :, :, :]
        v_attn = v_reshaped.swapaxes(0, 1)[None, :, :, :]
        
        attn_output = jax.nn.dot_product_attention(q_attn, k_attn, v_attn, mask= attention_mask, is_causal=is_causal, scale=self.scale)
        return attn_output.reshape(seq_len, total_dim)

    def _attn(self, q, k, v, attention_mask, is_causal):
        if self.scale is None:
            scale = 1.0 / jnp.sqrt(q.shape[-1])
        else:
            scale = self.scale

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # query-key product
        attn_weights = jnp.einsum("bnqh,bnkh->bnqk", q, k)

        # scale
        attn_weights = attn_weights * scale

        # apply causal mask
        if is_causal:
            causal_mask = jnp.tril(
                jnp.ones((q.shape[2], q.shape[2]), dtype=bool))
            causal_mask = causal_mask[None, None, :, :]
            mask_value = jnp.finfo(attn_weights.dtype).min
            attn_weights = jnp.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # softmax
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Compute attention output: attn_weights @ V
        attn_output = jnp.matmul(attn_weights, v)
        attn_output = jnp.swapaxes(attn_output, 1, 2)

        return attn_output
