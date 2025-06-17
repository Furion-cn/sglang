from abc import ABC, abstractmethod
from typing import List, Optional

import jax
from flax import linen as nn
from jax import numpy as jnp

from sglang.srt.jax.layers.quantization.base_config import QuantizeMethodBase


class LinearMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: jnp.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for a linear layer.
           The weights will be set as attributes of the layer.

        Args:
            layer: The layer that is using the LinearMethodBase factory.
            input_size_per_partition: Size of the weight input dim on rank X.
            output_partition_sizes: Sizes of the output dim of each logical
                weight on rank X. E.g., output_partition_sizes for QKVLinear
                is a list contains the width of Wq, Wk, Wv on rank X.
            input_size: Size of the input dim of the weight across all ranks.
            output_size: Size of the output dim of the weight across all ranks.
            params_dtype: Datatype of the parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: nn.Module,
        x: jax.Array,
        bias: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: jnp.dtype,
        **extra_weight_attrs,
    ):
        """Create weight parameters for the linear layer."""
        # Create weight parameter
        layer.weight = layer.param(
            'weight',
            nn.initializers.normal(stddev=0.01),
            (sum(output_partition_sizes), input_size_per_partition),
            params_dtype
        )
        
        # Store weight attributes for potential future use
        layer._weight_attrs = {
            "input_dim": 1, 
            "output_dim": 0,
            **extra_weight_attrs
        }

    def apply(
        self,
        layer: nn.Module,
        x: jax.Array,
        bias: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Apply linear transformation directly."""
        # Apply linear transformation: output = x @ weight.T
        output = jnp.dot(x, layer.weight.T)
        
        if bias is not None:
            output = output + bias
            
        return output


class LinearBase(nn.Module):
    """Base linear layer for Flax.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
    """
    
    input_size: int
    output_size: int
    bias: bool = True
    skip_bias_add: bool = False
    params_dtype: Optional[jnp.dtype] = None
    quant_config: Optional[object] = None  # QuantizationConfig type not available in JAX version
    prefix: str = ""

    def setup(self):
        """Initialize parameters and quantization method."""
        if self.params_dtype is None:
            self.params_dtype = jnp.float32
            
        # Create quantization method
        if self.quant_config is None:
            self.quant_method = UnquantizedLinearMethod()
        else:
            raise Exception("Quantization config is not supported")
        
        # Create weights through quantization method (like PyTorch version)
        assert self.quant_method is not None
        self.quant_method.create_weights(
            self,
            self.input_size,
            [self.output_size],
            self.input_size,
            self.output_size,
            self.params_dtype,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass of the linear layer."""
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict
    """
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass of the replicated linear layer."""
        bias = self.bias_param if (self.bias and not self.skip_bias_add) else None
        assert self.quant_method is not None
        output = self.quant_method.apply(self, x, bias)
        
        if self.skip_bias_add:
            return output, self.bias_param
        else:
            return output