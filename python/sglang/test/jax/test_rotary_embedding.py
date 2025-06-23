"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

""" Tests for Llama. """

import unittest
from typing import Tuple
import numpy as np
import jax
from sglang.srt.jax.layers import embeddings
import jax.numpy as jnp


"""  
An example reference jax_llama RoPE implementation from https://github.com/Sea-Snell/ 
Users should feel free to change and optimize the RoPE implementation in MaxText defined in layers.py 
as long as it passes our tests. But they shouldn't change the "reference" implementation in 
llama_test.py which is only to be used for comparison purpose. 
"""


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype: jnp.dtype = jnp.bfloat16) -> jnp.ndarray:
    """Calculate the frequencies."""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)
                    [: (dim // 2)].astype(dtype) / dim))
    t = np.arange(end)  # type: ignore
    freqs = np.outer(t, freqs).astype(dtype)  # type: ignore
    sin, cos = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)


def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    num_heads: int,
    freqs_cis: jnp.ndarray,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the computed Rotary Positional Embedding."""
    assert len(xq.shape) == 3 and len(xk.shape) == 3

    inputs_q = xq.reshape(xq.shape[0], xq.shape[1], num_heads, -1)
    inputs_k = xk.reshape(xk.shape[0], xk.shape[1], num_heads, -1)

    reshape_xq = inputs_q.astype(jnp.float32).reshape(*inputs_q.shape[:-1], -1, 2)
    reshape_xk = inputs_k.astype(jnp.float32).reshape(*inputs_k.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    # add head dim
    freqs_cis = jnp.reshape(
        freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)),
                        axis=-1).reshape(*xq_out.shape[:-1], -1)

    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)),
                        axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(dtype).reshape(xq.shape[0], xq.shape[1], -1), xk_out.astype(dtype).reshape(xk.shape[0], xk.shape[1], -1)


def permute_to_match_maxtext_rope(arr):
    evens = arr[..., ::2]
    odds = arr[..., 1::2]
    return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)


class RoPETest(unittest.TestCase):
  """Test for the RoPE implementation."""

  def test_rope(self):
    batch_size = 2
    head_num = 4
    dim_per_head = 128
    seq_len = 8

    # Run the two implementations on some random query and key
    x_q = np.random.normal(1, 0.5, (batch_size, seq_len, head_num*dim_per_head))
    x_k = np.random.normal(1, 0.5, (batch_size, seq_len, head_num*dim_per_head))

    # Calculate RoPE embeddings from Sea-Snell implementation
    freqs_cis = precompute_freqs_cis(dim_per_head, seq_len * 2)
    freqs_cis = jnp.take(freqs_cis, jnp.arange(
        seq_len, dtype=np.int32)[None, :], axis=0)

    expected_output = apply_rotary_emb(
        jnp.asarray(x_q), jnp.asarray(x_k), num_heads=head_num, freqs_cis=freqs_cis)
    
    position = jnp.arange(seq_len, dtype=jnp.float32)[jnp.newaxis, :]

    for is_neox_style in [False, True]:
        if is_neox_style:
            rope = embeddings.RotaryEmbedding(
                min_timescale=1, max_timescale=10_000, embedding_dims=dim_per_head, num_heads=head_num, is_neox_style=is_neox_style)
            query_proj = rope(x_q, position)
            key_proj = rope(x_k, position)
            # Compare results
            diff = np.abs(query_proj - expected_output[0])
            max_diff = float(np.max(diff))
            mean_diff = float(np.mean(diff))
            print(f'qqq {max_diff=} {mean_diff}')
            print(f'{query_proj=}')
            print(f'{expected_output[0]=}')
            diff = np.abs(key_proj - expected_output[1])
            max_diff = float(np.max(diff))
            mean_diff = float(np.mean(diff))
            print(f'kkk {max_diff=} {mean_diff}')
            self.assertTrue(jnp.allclose(
                expected_output[0], query_proj, rtol=1e-01, atol=1e-04))
            self.assertTrue(jnp.allclose(
                expected_output[1], key_proj, rtol=1e-01, atol=1e-04))
        else:
            rope = embeddings.RotaryEmbedding(
                min_timescale=1, max_timescale=10_000, embedding_dims=dim_per_head, num_heads=head_num, is_neox_style=is_neox_style)
            query_proj = rope(x_q, position)
            key_proj = rope(x_k, position)
            # Compare results
            self.assertTrue(jnp.allclose(
                expected_output[0], query_proj, rtol=1e-01, atol=1e-04))
            self.assertTrue(jnp.allclose(
                expected_output[1], key_proj, rtol=1e-01, atol=1e-04))


if __name__ == "__main__":
  unittest.main()
