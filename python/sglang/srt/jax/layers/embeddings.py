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
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import dtypes
from flax.nnx.nn.linear import default_embed_init
from flax.typing import PromoteDtypeFn


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
        num_embeddings: int,
        features: int,
        dtype: Optional[jnp.dtype] = None,
        param_dtype: jnp.dtype = jnp.bfloat16,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
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
        self.embedding = nnx.Param(
            nnx.with_partitioning(default_embed_init, (None, "tensor"))(
                rngs.params(), (num_embeddings, features), param_dtype
            )
        )

        self.num_embeddings = num_embeddings
        self.features = features
        self.dtype = dtype or self.embedding.value.dtype
        self.promote_dtype = promote_dtype

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Embeds the inputs along the last dimension.

        Args:
          inputs: input data, all dimensions are considered batch dimensions.

        Returns:
          Output which is embedded input data.  The output shape follows the input,
          with an additional `features` dimension appended.
        """
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError(
                'Input type must be an integer or unsigned integer.')
        # Use take because fancy indexing numpy arrays with JAX indices does not
        # work correctly.
        (embedding,) = self.promote_dtype(
            (self.embedding.value,), dtype=self.dtype, inexact=False
        )
        if self.num_embeddings == 1:
            return jnp.broadcast_to(embedding, inputs.shape + (self.features,))
        return jnp.take(embedding, inputs, axis=0)

    def attend(self, query: jax.Array) -> jax.Array:
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
        query, embedding = self.promote_dtype(
            (query, self.embedding.value), dtype=self.dtype
        )
        return jnp.dot(query, embedding.T)


class ParallelLMHead(Embed):
    def __init__(
        self,
        num_embeddings: int,
        features: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        use_bias: bool = False,
    ):
        super().__init__(
            num_embeddings=num_embeddings,
            features=features,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )
        if use_bias:
            self.bias = nnx.Param(
                nnx.with_partitioning(nnx.initializers.constant(0.0), (None, "tensor"))(
                    rngs.params(), (self.num_embeddings,
                                    self.features), dtype
                )
            )
        else:
            self.bias = None

    def tie_weights(self, embed_tokens: Embed):
        """Tie the weights with word embeddings."""
        self.embedding = embed_tokens.embedding
        return self

    def __call__(self, input_):
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")


def permute_to_match_maxtext_rope(inputs: jax.Array) -> jax.Array:
    """Permute the inputs to match the MaxText RoPE implementation."""
    evens = inputs[..., ::2]
    odds = inputs[..., 1::2]
    return jnp.concatenate((evens, odds), axis=inputs.ndim - 1)


def inverse_permute_from_maxtext_rope(arr):
    """
    permute_to_match_maxtext_rope 的逆操作
    
    输入格式：[evens, odds] = [a, c, e, g, b, d, f, h]
    输出格式：[a, b, c, d, e, f, g, h]
    """
    # 获取最后一维的大小
    last_dim = arr.shape[-1]

    # 确保最后一维是偶数
    assert last_dim % 2 == 0, f"最后一维必须是偶数，但得到 {last_dim}"

    # 计算中点
    mid = last_dim // 2

    # 分离偶数和奇数部分
    evens = arr[..., :mid]      # 前半部分：原来的偶数索引元素
    odds = arr[..., mid:]       # 后半部分：原来的奇数索引元素

    # 交替放置：使用 stack + reshape
    # stack 后形状变为 [..., mid, 2]，然后 reshape 为 [..., last_dim]
    interleaved = jnp.stack([evens, odds], axis=-1)
    restored = interleaved.reshape(arr.shape[:-1] + (last_dim,))

    return restored

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
        is_neox_style: bool = False,
    ):
        super().__init__()
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.num_heads = num_heads
        self.embedding_dims = embedding_dims
        self.is_neox_style = is_neox_style

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
        if self.is_neox_style:
            freqs_cis = self._precompute_freqs_cis(inputs.shape[1])
            x_out = self._apply_rotary_emb(inputs, freqs_cis)
            return x_out
        else: 
            assert position is not None
            batch_size = inputs.shape[0]
            seq_len = inputs.shape[1]
            hidden_size = inputs.shape[2]
            head_dim = hidden_size // self.num_heads
            x = jnp.reshape(
                inputs, (batch_size, seq_len, self.num_heads, head_dim))
            x = permute_to_match_maxtext_rope(x)
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
            x_out = jnp.concatenate((first_part, second_part), axis=-1)
            x_out = inverse_permute_from_maxtext_rope(x_out)
            x_out = jnp.reshape(x_out, (batch_size, seq_len, hidden_size))
            return x_out
        
    def _precompute_freqs_cis(self, seq_len: int, theta: float = 10000.0, dtype: jnp.dtype = jnp.bfloat16) -> jnp.ndarray:
        """Calculate the frequencies."""
        freqs = 1.0 / (theta ** (jnp.arange(0, self.embedding_dims, 2)
                                [: (self.embedding_dims // 2)].astype(dtype) / self.embedding_dims))
        print(f'RotaryEmbedding._precompute_freqs_cis {self.embedding_dims=} {seq_len=} {theta=} {dtype=}')
        print(f'RotaryEmbedding._precompute_freqs_cis {freqs=}')
        t = jnp.arange(seq_len*2)  # type: ignore
        freqs = jnp.outer(t, freqs).astype(dtype)  # type: ignore
        sin, cos = jnp.sin(freqs), jnp.cos(freqs)
        freqs_cis = jnp.complex64(cos + 1j * sin)
        freqs_cis = jnp.take(freqs_cis, jnp.arange(
            seq_len, dtype=jnp.int32)[None, :], axis=0)
        return jnp.asarray(freqs_cis)


    def _apply_rotary_emb(
        self,
        inputs: jnp.ndarray,
        freqs_cis: jnp.ndarray,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> jnp.ndarray:
        """Apply the computed Rotary Positional Embedding."""
        assert len(inputs.shape) == 3 

        x = inputs.reshape(inputs.shape[0], inputs.shape[1], self.num_heads, -1)

        reshape_x = x.astype(jnp.float32).reshape(
            *x.shape[:-1], -1, 2)

        x_ = jax.lax.complex(reshape_x[..., 0], reshape_x[..., 1])

        # add head dim
        freqs_cis = jnp.reshape(
            freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

        x_out = x_ * freqs_cis
        x_out = jnp.stack((jnp.real(x_out), jnp.imag(x_out)),
                        axis=-1).reshape(*x_out.shape[:-1], -1)

        return x_out.astype(dtype).reshape(x.shape[0], x.shape[1], -1)
