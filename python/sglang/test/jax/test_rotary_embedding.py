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

from typing import Tuple
import unittest
import numpy as np
import jax
from sglang.srt.jax.layers import embeddings
import jax.numpy as jnp
import torch
from sglang.srt.layers.rotary_embedding import RotaryEmbedding

"""
An example reference jax_llama RoPE implementation from https://github.com/Sea-Snell/
Users should feel free to change and optimize the RoPE implementation in MaxText defined in layers.py
as long as it passes our tests. But they shouldn't change the "reference" implementation in
llama_test.py which is only to be used for comparison purpose.
"""


def precompute_freqs_cis(dim: int, end: int, seq_len: int, theta: float = 10000.0, dtype: jnp.dtype = jnp.bfloat16) -> jnp.ndarray:
    """Calculate the frequencies."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)
                   [: (dim // 2)].astype(dtype) / dim))
    t = jnp.arange(end)  # type: ignore
    freqs = jnp.outer(t, freqs).astype(dtype)  # type: ignore
    sin, cos = jnp.sin(freqs), jnp.cos(freqs)
    freqs_cis = jnp.complex64(cos + 1j * sin)
    freqs_cis = jnp.take(freqs_cis, jnp.arange(
        seq_len, dtype=jnp.int32)[None, :], axis=0)
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

    reshape_xq = inputs_q.astype(jnp.float32).reshape(
        *inputs_q.shape[:-1], -1, 2)
    reshape_xk = inputs_k.astype(jnp.float32).reshape(
        *inputs_k.shape[:-1], -1, 2)

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
    head_num = 2
    dim_per_head = 128
    seq_len = 2

    # Run the two implementations on some random query and key
    x_q = np.random.normal(
        1, 0.5, (batch_size, seq_len, head_num*dim_per_head))
    x_k = np.random.normal(
        1, 0.5, (batch_size, seq_len, head_num*dim_per_head))
    x_q_torch = torch.tensor(
        x_q, dtype=torch.bfloat16).reshape(-1, head_num*dim_per_head)
    x_q_jax = jnp.array(x_q, dtype=jnp.bfloat16)
    x_k_torch = torch.tensor(
        x_k, dtype=torch.bfloat16).reshape(-1, head_num*dim_per_head)
    x_k_jax = jnp.array(x_k, dtype=jnp.bfloat16)

    freqs_cis = precompute_freqs_cis(dim_per_head, seq_len * 2, seq_len)
    expected_output = apply_rotary_emb(x_q_jax, x_k_jax, num_heads=head_num, freqs_cis=freqs_cis, dtype=jnp.bfloat16)
    q_jax_output_float32 = expected_output[0].astype(jnp.float32)
    k_jax_output_float32 = expected_output[1].astype(jnp.float32)

    torch_rope = RotaryEmbedding(
        head_size=dim_per_head,
        rotary_dim=dim_per_head,
        max_position_embeddings=seq_len,
        base=10000,
        is_neox_style=False,
        dtype=torch.bfloat16,
    )

    torch_position = torch.arange(seq_len, dtype=torch.int32)[
        None, :].repeat(batch_size, 1)
    torch_output = torch_rope.forward_native(
        query=x_q_torch,
        key=x_k_torch,
        positions=torch_position,
    )
    q_torch_output_float32 = torch_output[0].reshape(
        batch_size, seq_len, -1).to(torch.float32)
    k_torch_output_float32 = torch_output[1].reshape(
        batch_size, seq_len, -1).to(torch.float32)
    
    self.assertTrue(jnp.allclose(
        np.array(q_torch_output_float32), np.array(q_jax_output_float32), rtol=1e-01, atol=1e-04))
    self.assertTrue(jnp.allclose(
        np.array(k_torch_output_float32), np.array(k_jax_output_float32), rtol=1e-01, atol=1e-04))

    for is_neox_style in [False, ]:
        if is_neox_style:
            position = jnp.arange(seq_len, dtype=jnp.int32)[jnp.newaxis, :]
            rope = embeddings.RotaryEmbedding(
                min_timescale=1, max_timescale=10_000, embedding_dims=dim_per_head, num_heads=head_num, is_neox_style=is_neox_style)
            query_proj = rope(x_q_jax, position)
            key_proj = rope(x_k_jax, position)
            # Compare results
            q_jax_output_float32 = query_proj.astype(jnp.float32)
            k_jax_output_float32 = key_proj.astype(jnp.float32)
            self.assertTrue(jnp.allclose(
                np.array(q_torch_output_float32), np.array(q_jax_output_float32), rtol=1e-01, atol=1e-04))
            self.assertTrue(jnp.allclose(
                np.array(k_torch_output_float32), np.array(k_jax_output_float32), rtol=1e-01, atol=1e-04))
        else:
            position = jnp.arange(seq_len, dtype=jnp.int32)[jnp.newaxis, :]
            rope = embeddings.RotaryEmbedding(
                min_timescale=1, max_timescale=10_000, embedding_dims=dim_per_head, num_heads=head_num, is_neox_style=is_neox_style)
            query_proj = rope(x_q_jax, position)
            key_proj = rope(x_k_jax, position)
            # Compare results
            q_jax_output_float32 = query_proj.astype(jnp.float32)
            k_jax_output_float32 = key_proj.astype(jnp.float32)
            self.assertTrue(jnp.allclose(
                np.array(q_torch_output_float32), np.array(q_jax_output_float32), rtol=1e-01, atol=1e-04))
            self.assertTrue(jnp.allclose(
                np.array(k_torch_output_float32), np.array(k_jax_output_float32), rtol=1e-01, atol=1e-04))


if __name__ == "__main__":
    unittest.main()
