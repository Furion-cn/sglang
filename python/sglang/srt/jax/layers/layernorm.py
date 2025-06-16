
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax


class RMSNorm(nn.Module):
  """RMS normalization."""

  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  kernel_axes: Tuple[Optional[str], ...] = ()
  scale_init: Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray] = nn.initializers.ones
  parameter_memory_host_offload: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = self.param(
        "scale",
        nn.with_logical_partitioning(self.scale_init, self.kernel_axes),
        (features,),
        self.weight_dtype,
    )
    scale = jnp.asarray(scale, self.dtype)
    return y * scale