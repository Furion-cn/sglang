from abc import ABC, abstractmethod

import jax
from flax import linen as nn
from jax import numpy as jnp


class QuantizeMethodBase(ABC):
    """Base class for different quantized methods."""

    @abstractmethod
    def create_weights(
        self, *weight_args, **extra_weight_attrs
    ):
        """Create weights for a layer.

        The weights will be set as attributes of the layer."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, *args, **kwargs) -> jax.Array:
        """Apply the weights in layer to the input tensor.

        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Process the weight after loading.

        This can be used for example, to transpose weights for computation.
        """
        return

class QuantizationConfig(ABC):
    pass