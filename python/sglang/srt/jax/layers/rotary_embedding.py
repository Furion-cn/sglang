import jax
from jax import nn


class RotaryEmbedding(nn.Module):
    """Rotary embedding layer."""

    @nn.compact
    def __call__(self, positions: jax.Array, q: jax.Array, k: jax.Array):
        return q, k
