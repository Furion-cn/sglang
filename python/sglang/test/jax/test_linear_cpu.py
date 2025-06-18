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

""" Tests for QKVParallelLinear on CPU. """

import unittest
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import sys
import os

# Add the parent directory to the path to import sglang modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from sglang.srt.jax.layers.linear import QKVParallelLinear
except ImportError:
    print("Warning: Could not import QKVParallelLinear. Running basic CPU tests only.")
    QKVParallelLinear = None


def reference_qkv_parallel_linear(x, weight, bias=None):
    """Reference implementation of QKV parallel linear using NumPy."""
    output = np.matmul(x, weight.T)
    if bias is not None:
        output = output + bias
    return output


class QKVParallelLinearCPUTest(unittest.TestCase):
    """Test for the QKVParallelLinear implementation on CPU."""

    def test_device_info(self):
        """Test to show CPU device information."""
        print(f"Available devices: {jax.devices()}")
        print(f"CPU devices: {jax.devices('cpu')}")
        print(f"Default device: {jax.devices()[0]}")
        
        # Create a simple array and check its device
        x = jnp.array([1, 2, 3])
        print(f"Array device: {x.device()}")
        
        # Verify we're using CPU
        self.assertTrue(str(x.device()).startswith('cpu'), "Not using CPU device")

    def test_basic_jax_operations(self):
        """Test basic JAX operations on CPU."""
        # Test matrix multiplication
        a = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
        b = jnp.array([[5, 6], [7, 8]], dtype=jnp.float32)
        
        result = jnp.matmul(a, b)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        
        self.assertTrue(jnp.allclose(result, expected))
        print(f"Matrix multiplication result:\n{result}")
        
        # Test random number generation
        rng = jax.random.PRNGKey(42)
        random_array = jax.random.normal(rng, (3, 3))
        print(f"Random array shape: {random_array.shape}")
        print(f"Random array device: {random_array.device()}")

    def test_qkv_parallel_linear(self):
        """Test QKVParallelLinear if available."""
        if QKVParallelLinear is None:
            self.skipTest("QKVParallelLinear not available")
            
        batch_size = 4
        seq_len = 8
        hidden_size = 64
        head_size = 32
        num_heads = 4

        # Create random input
        x = np.random.normal(0, 0.5, (batch_size, seq_len, hidden_size)).astype(np.float32)
        x = jnp.array(x)
        
        # Initialize the linear layer
        linear = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_size,
            num_heads=num_heads,
            use_bias=True,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros
        )
        
        # Initialize parameters with a fixed random seed for reproducibility
        rng = jax.random.PRNGKey(42)
        params = linear.init(rng, jnp.zeros((1, 1, hidden_size)))
        
        # Extract weights and biases for reference implementation
        weight = np.array(params['params']['kernel'])
        bias = np.array(params['params']['bias'])
        
        # Run reference implementation
        ref_output = reference_qkv_parallel_linear(x, weight, bias)
        
        # Run JAX implementation
        jax_output = linear.apply(params, x)
        
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


if __name__ == "__main__":
    unittest.main() 