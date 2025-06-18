from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax


class RMSNorm(nnx.Module):
    """RMS normalization."""

    def __init__(self,
                 epsilon: float = 1e-6,
                 dtype: Any = jnp.float32,
                 weight_dtype: Any = jnp.float32,
                 kernel_axes: Tuple[Optional[str], ...] = (),
                 scale_init: Callable[[jax.Array, Sequence[int],
                                       jnp.dtype], jax.Array] = nnx.initializers.ones,
                 parameter_memory_host_offload: bool = False):
        self.epsilon = epsilon
        self.dtype = dtype
        self.weight_dtype = weight_dtype
        self.kernel_axes = kernel_axes
        self.scale_init = scale_init
        self.parameter_memory_host_offload = parameter_memory_host_offload

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies layer normalization on the input."""
        x = jnp.asarray(x, jnp.float32)
        features = x.shape[-1]
        mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
        y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
        scale = self.param(
            "scale",
            nnx.with_logical_partitioning(self.scale_init, self.kernel_axes),
            (features,),
            self.weight_dtype,
        )
        scale = jnp.asarray(scale, self.dtype)
        return y * scale
