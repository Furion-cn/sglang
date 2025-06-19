from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax


class RMSNorm(nnx.Module):
    """RMS normalization."""

    def __init__(self,
                 hidden_size: int,
                 epsilon: float = 1e-6,
                 kernel_axes: Optional[Sequence[str]] = None,
                 rngs: nnx.Rngs = None):
        self.variance_epsilon = epsilon
        self.weight = nnx.Param(
            nnx.with_partitioning(nnx.initializers.ones, kernel_axes)(
                rngs.params(), (hidden_size,))
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies layer normalization on the input."""
        x = jnp.asarray(x, jnp.float32)
        mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
        y = jnp.asarray(
            x * lax.rsqrt(mean2 + self.variance_epsilon), jnp.float32)
        scale = jnp.asarray(self.weight, jnp.float32)
        return y * scale
