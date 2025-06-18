from abc import ABC, abstractmethod
from typing import List, Optional

import jax
from flax import linen as nn
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec

from sglang.srt.distributed import divide
from sglang.srt.jax.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

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
        input_size: int,
        output_size: int,
        params_dtype: jnp.dtype,
        partition_spec: Optional[PartitionSpec] = None,
    ):
        """Create weight parameters for the linear layer."""
        layer.weight = layer.param(
            'weight',
            nn.with_partitioning(nn.initializers.normal(), partition_spec),
            (output_size, input_size),
            params_dtype
        )

    def apply(
        self,
        layer: nn.Module,
        x: jax.Array,
        bias: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Apply linear transformation directly."""
        output = jnp.dot(x, layer.weight.T)

        if bias is not None:
            output = output + bias

        return output


class LinearBase(nn.Module):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        partition_spec: Partition spec for the linear layer.
        prefix: Prefix for the linear layer.
    """

    input_size: int
    output_size: int
    bias: bool = True
    skip_bias_add: bool = False
    params_dtype: Optional[jnp.dtype] = None
    quant_config: Optional[QuantizationConfig] = None
    partition_spec: Optional[PartitionSpec] = None
    prefix: str = ""

    def setup(self):
        """Initialize parameters and quantization method."""
        if self.params_dtype is None:
            self.params_dtype = jnp.float32

        if self.quant_config is None:
            self.quant_method = UnquantizedLinearMethod()
        else:
            raise Exception("Quantization config is not supported")

        assert self.quant_method is not None
        self.quant_method.create_weights(
            self,
            self.input_size,
            self.output_size,
            self.params_dtype,
            self.partition_spec,
        )
        if self.bias:
            self.bias_param = self.param(
                "bias", 
                nn.with_partitioning(nn.initializers.zeros_init(), self.partition_spec),
                (self.output_size,))
        else:
            self.bias_param = None

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass of the linear layer."""

        bias = self.bias_param if not self.skip_bias_add else None
        assert self.quant_method is not None
        output = self.quant_method.apply(self, input, bias)
        output_bias = self.bias_param if self.skip_bias_add else None
        return output, output_bias
    

class QKVParallelLinear(LinearBase):
    """QKVParallelLinear layer."""

    hidden_size: int
    head_size: int
    num_heads: int
    num_kv_heads: int

    def setup(self):
        self.input_size = self.hidden_size
        self.output_size = (self.num_heads + 2 * self.num_kv_heads) * self.head_size
        super().setup()