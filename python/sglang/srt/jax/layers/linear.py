from abc import ABC, abstractmethod
from typing import List, Optional, Iterable, Tuple, Union

import jax
from flax import nnx
from flax.typing import Initializer
from flax.linen.linear import default_kernel_init
from jax import numpy as jnp
import jax.lax as lax
import numpy as np

from sglang.srt.jax.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.jax.layers.initializers import default_bias_init
from sglang.srt.jax.common_types import DType, Array


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _compute_dot_general(inputs, kernel, kernel_axes, axis, contract_ind, matmul_precision, quant):
    """Computes a dot_general operation that may be quantized."""
    dot_general = lax.dot_general
    matmul_precision = lax.Precision(matmul_precision)
    return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=matmul_precision)

class LinearMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def create_weights(
        self,
        in_features: Union[Iterable[int], int],
        out_features: Union[Iterable[int], int],
        axis: Union[Iterable[int], int] = -1,
        weight_dtype: DType = jnp.float32,
        dtype: DType = jnp.float32,
        kernel_init: Initializer = default_kernel_init,
        kernel_axes: Tuple[Optional[str], ...] = (),
        use_bias: bool = False,
        matmul_precision: str = "default",
        parameter_memory_host_offload: bool = False,
        *,  # Following arguments are keyword-only
        rngs: nnx.Rngs,
    ):
        """Initializes the DenseGeneral module.

        Args:
        in_features: tuple with numbers of input features for axes specified in 'axis'.
        out_features: tuple with numbers of output features.
        axis: tuple with axes to apply the transformation on.
        weight_dtype: the dtype of the weights (default: float32).
        dtype: the dtype of the computation (default: float32).
        kernel_init: initializer function for the weight matrix.
        kernel_axes: logical axes for partitioning the kernel.
        use_bias: whether to add bias in linear transformation.
        matmul_precision: Precision for matrix multiplication.
        parameter_memory_host_offload: Determines whether to offload params to host
        rngs: RNG state for initialization in nnx.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: nnx.Module,
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
        in_features: Union[Iterable[int], int],
        out_features: Union[Iterable[int], int],
        axis: Union[Iterable[int], int] = -1,
        weight_dtype: DType = jnp.float32,
        dtype: DType = jnp.float32,
        kernel_init: Initializer = default_kernel_init,
        kernel_axes: Tuple[Optional[str], ...] = (),
        use_bias: bool = False,
        matmul_precision: str = "default",
        *,  # Following arguments are keyword-only
        rngs: nnx.Rngs,
    ):
        """Initializes the DenseGeneral module.

        Args:
        in_features: tuple with numbers of input features for axes specified in 'axis'.
        out_features: tuple with numbers of output features.
        axis: tuple with axes to apply the transformation on.
        weight_dtype: the dtype of the weights (default: float32).
        dtype: the dtype of the computation (default: float32).
        kernel_init: initializer function for the weight matrix.
        kernel_axes: logical axes for partitioning the kernel.
        use_bias: whether to add bias in linear transformation.
        matmul_precision: Precision for matrix multiplication.
        rngs: RNG state for initialization in nnx.
        """
        # Parameter initialization
        self.in_features = _canonicalize_tuple(in_features)
        self.out_features = _canonicalize_tuple(out_features)
        self.axis = _canonicalize_tuple(axis)
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.kernel_init = kernel_init
        self.kernel_axes = kernel_axes
        self.use_bias = use_bias
        self.matmul_precision = matmul_precision

        kernel_shape = self.in_features + self.out_features
        kernel_in_axis = np.arange(len(self.axis))
        kernel_out_axis = np.arange(
            len(self.axis), len(self.axis) + len(self.out_features)
        )

        self.kernel = nnx.Param(
            self.kernel_init(
                rngs.params(),
                kernel_shape,
                self.weight_dtype,
                kernel_in_axis,
                kernel_out_axis,
            ),
            sharding=self.kernel_axes,
        )

        if self.use_bias:
            bias_axes = self.kernel_axes[-len(self.out_features) :]
            bias_shape = kernel_shape[-len(self.out_features) :]
            self.bias = nnx.Param(
                default_bias_init(rngs.params(), bias_shape, self.weight_dtype),
                sharding=bias_axes,
            )
        else:
            self.bias = None

    def apply(
        self,
        x: Array,
        skip_bias_add: bool = False,
    ) -> Array:
        """Applies a linear transformation to the inputs along multiple dimensions.

        Args:
        x: The nd-array to be transformed.
        bias: The bias to be added to the output if it is not None.

        Returns:
        The transformed input.
        """
        inputs = jnp.asarray(x, self.dtype)
        norm_axis = _normalize_axes(self.axis, inputs.ndim)

        for i, ax in enumerate(norm_axis):
            if inputs.shape[ax] != self.in_features[i]:
                raise ValueError(
                    f"Input dimension {inputs.shape[ax]} at axis {ax} "
                    f"does not match expected input feature size {self.in_features[i]}"
                )
        
        kernel = jnp.asarray(self.kernel[...], self.dtype)

        contract_ind = tuple(range(0, len(self.axis)))
        output = _compute_dot_general(
            inputs,
            kernel,
            self.kernel_axes,
            norm_axis,
            contract_ind,
            self.matmul_precision,
        )

        if self.bias is not None and not skip_bias_add:
            bias = jnp.asarray(self.bias[...], self.dtype)
            output += bias
        return output


def _canonicalize_tuple(x):
    if isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)


class LinearBase(nnx.Module):
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

    def __init__(
        self,
        in_features: Union[Iterable[int], int],
        out_features: Union[Iterable[int], int],
        axis: Union[Iterable[int], int] = -1,
        weight_dtype: DType = jnp.float32,
        dtype: DType = jnp.float32,
        kernel_init: Initializer = default_kernel_init,
        kernel_axes: Tuple[Optional[str], ...] = (),
        use_bias: bool = False,
        matmul_precision: str = "default",
        skip_bias_add: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        *,  # Following arguments are keyword-only
        rngs: nnx.Rngs,
    ):
        """Initializes the DenseGeneral module.

        Args:
        in_features: tuple with numbers of input features for axes specified in 'axis'.
        out_features: tuple with numbers of output features.
        axis: tuple with axes to apply the transformation on.
        weight_dtype: the dtype of the weights (default: float32).
        dtype: the dtype of the computation (default: float32).
        kernel_init: initializer function for the weight matrix.
        kernel_axes: logical axes for partitioning the kernel.
        use_bias: whether to add bias in linear transformation.
        matmul_precision: Precision for matrix multiplication.
        parameter_memory_host_offload: Determines whether to offload params to host
        rngs: RNG state for initialization in nnx.
        """
        super().__init__()
        self.skip_bias_add = skip_bias_add

        if quant_config is None:
            self.quant_method = UnquantizedLinearMethod()
        else:
            raise Exception("Quantization config is not supported")
        
        self.quant_method.create_weights(
            self,
            in_features,
            out_features,
            axis=axis,
            weight_dtype=weight_dtype,
            dtype=dtype,
            kernel_init=kernel_init,
            kernel_axes=kernel_axes,
            use_bias=use_bias,
            matmul_precision=matmul_precision,
            rngs=rngs,
        )


    def __call__(self, x: Array) -> Array:
        """Forward pass of the linear layer."""
        assert self.quant_method is not None
        output = self.quant_method.apply(self, x, self.skip_bias_add)
        output_bias = self.quant_method.bias if self.skip_bias_add else None
        return output, output_bias


class QKVParallelLinear(LinearBase):
    """QKVParallelLinear layer."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        num_heads: int,
        weight_dtype: DType = jnp.float32,
        dtype: DType = jnp.float32,
        kernel_init: Initializer = default_kernel_init,
        use_bias: bool = False,
        matmul_precision: str = "default",
        skip_bias_add: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        """Initializes the dense_general module.

        Args:
            inputs_shape: tuple with the shape of the inputs
            in_features: tuple with numbers of input features for axes specified in
            'axis'.
            features: tuple with numbers of output features.
            axis: tuple with axes to apply the transformation on.
            weight_dtype: the dtype of the weights (default: float32).
            dtype: the dtype of the computation (default: float32).
            kernel_init: initializer function for the weight matrix.
            kernel_axes: logical axes for partitioning the kernel.
            use_bias: whether to add bias in linear transformation.
            matmul_precision: Precision for matrix multiplication.
            parameter_memory_host_offload: Determines whether to offload params to host
            name: name passed to the ToLinen Module
        """

        super().__init__(
            in_features=_canonicalize_tuple(hidden_size),
            out_features=(3, num_heads, head_size),
            axis=_canonicalize_tuple(-1),
            weight_dtype=weight_dtype,
            dtype=dtype,
            kernel_init=kernel_init,
            kernel_axes=("embed", "qkv", "heads", "kv"),
            use_bias=use_bias,
            matmul_precision=matmul_precision,
            skip_bias_add=skip_bias_add,
            quant_config=quant_config,
        )
        assert self.quant_method is not None
