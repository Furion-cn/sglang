#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Embedding Layers."""

from typing import Optional

import jax
from jax import lax
import jax.numpy as jnp

from flax import nnx

from sglang.srt.jax import max_logging
from sglang.srt.jax.common_types import Config, DType, Array
from sglang.srt.jax.layers.initializers import Initializer, default_embed_init, default_bias_init


class Embed(nnx.Module):
    """A parameterized function from integers [0, n) to d-dimensional vectors.

    Attributes:
      num_embeddings: number of embeddings.
      features: number of feature dimensions for each embedding.
      dtype: the dtype of the embedding vectors (default: float32).
      embedding_init: embedding initializer.
    """

    def __init__(
        self,
        config: Config,
        num_embeddings: int,
        features: int,
        cast_input_dtype: Optional[DType] = None,
        dtype: DType = jnp.float32,
        attend_dtype: Optional[DType] = None,
        embedding_init: Initializer = default_embed_init,
        rngs: nnx.Rngs = None,
    ):
        """
        Sets up the embedding parameters for the model.

        This method initializes the embedding parameters with logical partitioning.
        The embedding is represented as a parameter with the specified shape and data type.

        Parameters:
        - embedding: The embedding parameter initialized using the specified method,
                     partitioned logically along the 'vocab' and 'embed' dimensions.

        Returns:
        None
        """
        super().__init__()
        self.config = config
        self.num_embeddings = num_embeddings
        self.features = features
        self.cast_input_dtype = cast_input_dtype
        self.dtype = dtype
        self.attend_dtype = attend_dtype
        self.embedding_init = embedding_init

        self.weight = nnx.Param(
            nnx.with_partitioning(self.embedding_init, ("vocab", "embed"))(
                rngs.params, (self.num_embeddings, self.features), self.config.weight_dtype
            )
        )

    def __call__(self, inputs: Array) -> Array:
        """Embeds the inputs along the last dimension.

        Args:
          inputs: input data, all dimensions are considered batch dimensions.

        Returns:
          Output which is embedded input data.  The output shape follows the input,
          with an additional `features` dimension appended.
        """
        cfg = self.config
        if self.cast_input_dtype:
            inputs = inputs.astype(self.cast_input_dtype)
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError(
                "Input type must be an integer or unsigned integer.")

        if cfg.use_iota_embed:
            iota = lax.iota(jnp.int32, self.num_embeddings)
            one_hot = jnp.array(
                inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
            output = jnp.dot(one_hot, jnp.asarray(self.weight, self.dtype))
        else:
            output = jnp.asarray(self.weight, self.dtype)[inputs]
        output = nnx.with_logical_constraint(
            output, ("activation_embed_and_logits_batch",
                     "activation_length", "activation_embed")
        )
        return output

    def attend(self, query: Array) -> Array:
        """Attend over the embedding using a query array.

        Args:
          query: array with last dimension equal the feature depth `features` of the
            embedding.

        Returns:
          An array with final dim `num_embeddings` corresponding to the batched
          inner-product of the array of query vectors against each embedding.
          Commonly used for weight-sharing between embeddings and logit transform
          in NLP models.
        """
        dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
        return jnp.dot(query, jnp.asarray(self.weight, jnp.bfloat16).T, preferred_element_type=dtype)


class ParallelLMHead(Embed):
    def __init__(
        self,
        config: Config,
        num_embeddings: int,
        features: int,
        cast_input_dtype: Optional[DType] = None,
        dtype: DType = jnp.float32,
        attend_dtype: Optional[DType] = None,
        embedding_init: Initializer = default_embed_init,
        rngs: nnx.Rngs = None,
        use_bias: bool = False,
        bias_init: Initializer = default_bias_init,
    ):
        super().__init__(
            config=config,
            num_embeddings=num_embeddings,
            features=features,
            cast_input_dtype=cast_input_dtype,
            dtype=dtype,
            attend_dtype=attend_dtype,
            embedding_init=embedding_init,
            rngs=rngs
        )
        if use_bias:
            self.bias = nnx.Param(
                nnx.with_partitioning(bias_init, ("vocab", "bias"))(
                    rngs.params, (self.num_embeddings,
                                  self.features), self.config.weight_dtype
                )
            )
        else:
            self.bias = None

    def tie_weights(self, embed_tokens: Embed):
        """Tie the weights with word embeddings."""
        self.weight = embed_tokens.weight
        return self

    def forward(self, input_):
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")


class RotaryEmbedding(nnx.Module):
    """Rotary Position Embedding.

    Attributes:
      min_timescale: Start of the geometric index. Determines the periodicity of
        the added signal.
      max_timescale: End of the geometric index. Determines the frequency of the
        added signal.
      embedding_dims: Dimension of the embedding to be generated.
    """

    def __init__(
        self,
        min_timescale: int,
        max_timescale: int,
        num_heads: int,
        embedding_dims: int = 0,
        cast_as_fprop_dtype: bool = True,
        fprop_dtype: DType = jnp.bfloat16
    ):
        super().__init__()
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.num_heads = num_heads
        self.embedding_dims = embedding_dims
        self.cast_as_fprop_dtype = cast_as_fprop_dtype
        self.fprop_dtype = fprop_dtype

        """init with timescale"""
        if self.embedding_dims % 2:
            raise ValueError(
                "Embedding dim for rotary position embedding must be a multiple of 2.")

        half_embedding_dim = self.embedding_dims // 2
        fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
        self.timescale = self.min_timescale * \
            (self.max_timescale / self.min_timescale) ** fraction

        half_embedding_dim = self.embedding_dims // 2
        fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
        self.timescale = self.min_timescale * \
            (self.max_timescale / self.min_timescale) ** fraction

    def __call__(
        self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
        inputs: jax.Array,
        position: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Generates a jax.Array of sinusoids with different frequencies.

        Args:
          inputs: The input sequence on which to apply the Rotary position
            embedding. Since rotary position embeddings are applied to query and
            keys after projection, it is assumed of shape [B, S, N, H].
          position: Optional position jax.Array which denotes the position of each
            token in the sequence. This only needs to be supplied when the sequence
            is packed. It is of shape [B, S].

        Returns:
          a jax.Array of shape [B, S, H] which includes the inputs together with
          the rotary position embedding incorporated in it.
        """
        assert position is not None
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        hidden_size = inputs.shape[2]
        head_dim = hidden_size // self.num_heads
        x = jnp.reshape(
            inputs, (batch_size, seq_len, self.num_heads, head_dim))
        if self.embedding_dims != head_dim:
            raise ValueError(
                "The embedding dims of the rotary position embedding" "must match the hidden dimension of the inputs."
            )

        position = position[:, :, jnp.newaxis, jnp.newaxis]
        sinusoid_inp = position / self.timescale
        sin = jnp.sin(sinusoid_inp).astype(x.dtype)
        cos = jnp.cos(sinusoid_inp).astype(x.dtype)
        first_half, second_half = jnp.split(x, 2, axis=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        if self.cast_as_fprop_dtype:
            first_part = first_part.astype(self.fprop_dtype)
            second_part = second_part.astype(self.fprop_dtype)
        x_out = jnp.concatenate((first_part, second_part), axis=-1)
        x_out = jnp.reshape(x_out, (batch_size, seq_len, hidden_size))
        return x_out
