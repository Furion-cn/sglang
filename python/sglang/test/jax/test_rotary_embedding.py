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


class RoPETest(unittest.TestCase):
  """Test for the RoPE implementation."""

  def test_rope(self):
    batch_size = 3
    head_num = 3
    dim_per_head = 128
    seq_len = 10

    # Run the two implementations on some random query and key
    x_q = np.random.normal(
        1, 0.5, (batch_size*seq_len, head_num*dim_per_head))
    x_k = np.random.normal(
        1, 0.5, (batch_size*seq_len, head_num*dim_per_head))
    
    x_q_torch = torch.tensor(x_q, dtype=torch.bfloat16)
    x_q_jax = jnp.array(x_q, dtype=jnp.bfloat16)

    x_k_torch = torch.tensor(x_k, dtype=torch.bfloat16)
    x_k_jax = jnp.array(x_k, dtype=jnp.bfloat16)

    positions_jax = jnp.arange(seq_len, dtype=jnp.int32)[
        jnp.newaxis, :].repeat(batch_size, axis=0)
    positions_torch = torch.arange(seq_len, dtype=torch.int32)[
        None, :].repeat(batch_size, 1)

    # compare jax and torch
    for is_neox_style in [False, ]:
        # jax implementation
        rope_jax = embeddings.RotaryEmbedding(
            head_size=dim_per_head,
            rotary_dim=dim_per_head,
            max_position_embeddings=seq_len,
            base=10000,
            is_neox_style=is_neox_style,
            dtype=jnp.bfloat16,
        )
        
        output_jax = rope_jax(positions_jax, x_q_jax, x_k_jax)
        q_jax_output_float32 = output_jax[0].astype(jnp.float32)
        k_jax_output_float32 = output_jax[1].astype(jnp.float32)

        # torch implementation
        rope_torch = RotaryEmbedding(
            head_size=dim_per_head,
            rotary_dim=dim_per_head,
            max_position_embeddings=seq_len,
            base=10000,
            is_neox_style=is_neox_style,
            dtype=torch.bfloat16,
        )

        torch_output = rope_torch.forward_native(positions_torch, x_q_torch, x_k_torch)
        q_torch_output_float32 = torch_output[0].to(torch.float32)
        k_torch_output_float32 = torch_output[1].to(torch.float32)
        
        # compare results
        self.assertTrue(jnp.allclose(
            np.array(q_torch_output_float32), np.array(q_jax_output_float32), rtol=1e-04, atol=1e-05))
        self.assertTrue(jnp.allclose(
            np.array(k_torch_output_float32), np.array(k_jax_output_float32), rtol=1e-04, atol=1e-05))
        
    # TODO: compare is_neox_style=True and False
    rope_is_neox_style = embeddings.RotaryEmbedding(
        head_size=dim_per_head,
        rotary_dim=dim_per_head,
        max_position_embeddings=seq_len,
        base=10000,
        is_neox_style=True,
        dtype=jnp.bfloat16,
    )
    output_is_neox_style = rope_is_neox_style(positions_jax, x_q_jax, x_k_jax)
    q_is_neox_style_output_float32 = output_is_neox_style[0].astype(jnp.float32)
    k_is_neox_style_output_float32 = output_is_neox_style[1].astype(jnp.float32)
    
    rope_is_not_neox_style = embeddings.RotaryEmbedding(
        head_size=dim_per_head,
        rotary_dim=dim_per_head,
        max_position_embeddings=seq_len,
        base=10000,
        is_neox_style=False,
        dtype=jnp.bfloat16,
    )
    output_is_not_neox_style = rope_is_not_neox_style(positions_jax, x_q_jax, x_k_jax)
    q_is_not_neox_style_output_float32 = output_is_not_neox_style[0].astype(jnp.float32)
    k_is_not_neox_style_output_float32 = output_is_not_neox_style[1].astype(jnp.float32)
    
    # compare results
    self.assertTrue(jnp.allclose(
        np.array(q_is_neox_style_output_float32), np.array(q_is_not_neox_style_output_float32), rtol=1e-04, atol=1e-05))
    self.assertTrue(jnp.allclose(
        np.array(k_is_neox_style_output_float32), np.array(k_is_not_neox_style_output_float32), rtol=1e-04, atol=1e-05))
    

if __name__ == "__main__":
    unittest.main()
