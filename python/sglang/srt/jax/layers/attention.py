import jax
import jax.numpy as jnp
from flax import nnx


class Attention(nnx.Module):
    """attention layer."""

    def __call__(self,
                 q: jax.Array,
                 k: jax.Array,
                 v: jax.Array,
                 attention_mask: jax.Array = None):

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # query-key product
        attn_weights = jnp.einsum("bnqh,bnkh->bnqk", q, k)

        # apply causal mask
        causal_mask = jnp.tril(jnp.ones((q.size(2), q.size(2)), dtype=bool))
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
