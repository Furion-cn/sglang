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

""" Tests for QKVParallelLinear. """

from sglang.srt.jax.layers.linear import QKVParallelLinear
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec
from jax.lax import with_sharding_constraint
import flax.linen as nn
import jax.numpy as jnp
import jax
import unittest
import numpy as np


def reference_qkv_parallel_linear(x, weight, bias=None):
    """Reference implementation of QKV parallel linear using NumPy."""
    output = np.matmul(x, weight.T)
    if bias is not None:
        output = output + bias
    return output


class QKVParallelLinearTest(unittest.TestCase):
    """Test for the QKVParallelLinear implementation."""

    def setUp(self):
        """Set up device mesh for 4 devices."""
        # Create a 2x2 device mesh for 4 devices
        self.mesh = jax.make_mesh((2, 2), ('nodes', 'devices'))

        # Set up sharding specs
        self.data_sharding = jax.sharding.NamedSharding(self.mesh, PartitionSpec('nodes', None))
        self.model_sharding = jax.sharding.NamedSharding(self.mesh, PartitionSpec(None, 'devices'))
        self.data_model_sharding = jax.sharding.NamedSharding(self.mesh, PartitionSpec('nodes', 'devices'))

    def test_qkv_parallel_linear(self):
        batch_size = 4  # Must be divisible by data parallelism
        seq_len = 8
        hidden_size = 64
        head_size = 32
        num_heads = 4

        # Create random input with proper sharding
        x = np.random.normal(0, 0.5, (batch_size, seq_len,
                             hidden_size)).astype(np.float32)
        x = jnp.array(x)
        x = with_sharding_constraint(x, self.data_sharding)

        # Initialize the linear layer with sharding
        rng = jax.random.PRNGKey(42)
        linear = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_size,
            num_heads=num_heads,
            use_bias=True,
            # Use random normal initialization with sharding
            kernel_init=nn.with_partitioning(
                nn.initializers.normal(stddev=0.02),
                (None, 'devices')
            ),
        )

        # Extract weights and biases for reference implementation
        weight = np.array(linear.quant_method.kernel)
        bias = np.array(linear.quant_method.bias)

        # Run reference implementation
        ref_output = reference_qkv_parallel_linear(x, weight, bias)

        # Run JAX implementation with sharding
        jax_output = linear(x)

        # Compare results
        self.assertTrue(
            jnp.allclose(
                jnp.array(ref_output),
                jax_output,
                rtol=1e-4,
                atol=1e-4
            ),
            "QKV parallel linear outputs do not match"
        )

        # Test output shape
        expected_output_size = 3 * num_heads * head_size  # Q, K, V each with num_heads
        self.assertEqual(
            jax_output.shape,
            (batch_size, seq_len, expected_output_size),
            "Output shape mismatch"
        )

    def test_qkv_parallel_linear_no_bias(self):
        """Test QKVParallelLinear without bias."""
        batch_size = 2  # Must be divisible by data parallelism
        seq_len = 4
        hidden_size = 32
        head_size = 16
        num_heads = 2

        # Create random input with proper sharding
        x = np.random.normal(0, 0.5, (batch_size, seq_len,
                             hidden_size)).astype(np.float32)
        x = jnp.array(x)
        x = with_sharding_constraint(x, self.data_sharding)

        # Initialize the linear layer without bias
        rng = jax.random.PRNGKey(123)
        linear = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_size,
            num_heads=num_heads,
            use_bias=False,
            kernel_init=nn.with_partitioning(
                nn.initializers.normal(stddev=0.01),
                (None, 'devices')
            )
        )

        # Extract weights for reference implementation
        weight = np.array(linear.quant_method.kernel)

        # Run reference implementation (no bias)
        ref_output = reference_qkv_parallel_linear(x, weight, bias=None)

        # Run JAX implementation
        jax_output = linear(x)

        # Compare results
        self.assertTrue(
            jnp.allclose(
                jnp.array(ref_output),
                jax_output,
                rtol=1e-4,
                atol=1e-4
            ),
            "QKV parallel linear outputs do not match (no bias case)"
        )

    def test_sharding_consistency(self):
        """Test that sharding is consistent across different inputs."""
        batch_size = 4
        seq_len = 8
        hidden_size = 64
        head_size = 32
        num_heads = 4

        # Create two different inputs
        x1 = np.random.normal(
            0, 0.5, (batch_size, seq_len, hidden_size)).astype(np.float32)
        x2 = np.random.normal(
            0, 0.5, (batch_size, seq_len, hidden_size)).astype(np.float32)

        x1 = jnp.array(x1)
        x2 = jnp.array(x2)
        x1 = with_sharding_constraint(x1, self.data_sharding)
        x2 = with_sharding_constraint(x2, self.data_sharding)

        # Initialize the linear layer
        rng = jax.random.PRNGKey(42)
        linear = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_size,
            num_heads=num_heads,
            use_bias=True,
            kernel_init=nn.with_partitioning(
                nn.initializers.normal(stddev=0.02),
                (None, 'devices')
            ),
            bias_init=nn.with_partitioning(
                nn.initializers.zeros,
                ('devices',)
            )
        )

        # Run both inputs
        output1 = linear(x1)
        output2 = linear(x2)

        # Check that outputs have correct shapes
        expected_output_size = 3 * num_heads * head_size  # Q, K, V each with num_heads
        self.assertEqual(output1.shape, (batch_size,
                         seq_len, expected_output_size))
        self.assertEqual(output2.shape, (batch_size,
                         seq_len, expected_output_size))

        # Check that outputs are different (as inputs are different)
        self.assertFalse(jnp.allclose(output1, output2))


if __name__ == "__main__":
    unittest.main()
