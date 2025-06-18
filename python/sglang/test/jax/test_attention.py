import itertools
import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from sglang.srt.jax.layers.attention import Attention
from sglang.test.test_utils import CustomTestCase


class TestAttention(CustomTestCase):
    """Test cases for the Attention layer."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _create_test_inputs(self, batch_size=2, num_heads=4, seq_len=32, head_dim=64, dtype=jnp.float32, seed=0):
        """Create test inputs for attention computation."""
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 4)

        # Create Q, K, V with shape [batch, seq_len, num_heads, head_dim]
        q = jax.random.normal(
            keys[0], (batch_size, seq_len, num_heads, head_dim), dtype=dtype)
        k = jax.random.normal(
            keys[1], (batch_size, seq_len, num_heads, head_dim), dtype=dtype)
        v = jax.random.normal(
            keys[2], (batch_size, seq_len, num_heads, head_dim), dtype=dtype)

        # Optional attention mask
        attention_mask = jax.random.bernoulli(
            keys[3], 0.8, (batch_size, 1, seq_len, seq_len)).astype(dtype)

        return q, k, v, attention_mask

    def _get_cudnn_reference(self, q, k, v, attention_mask=None):
        """Get JAX cuDNN reference result."""
        try:
            return jax.nn.dot_product_attention(
                query=q, key=k, value=v,
                mask=attention_mask.astype(
                    bool) if attention_mask is not None else None,
                is_causal=attention_mask is None
            )
        except Exception as e:
            self.skipTest(f"JAX cuDNN attention not available: {e}")

    def _assert_attention_correctness(self, attention_layer, q, k, v, attention_mask=None):
        """Assert attention layer output matches cuDNN reference."""
        output = attention_layer(q, k, v, attention_mask)
        output_ref = self._get_cudnn_reference(q, k, v, attention_mask)

        # Check shape and finiteness
        expected_shape = (q.shape[0], q.shape[1], q.shape[2], q.shape[3])
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(jnp.all(jnp.isfinite(output)))

        # Check correctness
        self.assertTrue(jnp.allclose(output, output_ref, atol=1e-4, rtol=1e-3))

    def test_attention_basic(self):
        """Test basic attention functionality with various configurations."""
        # Create NNX module with random state
        rngs = nnx.Rngs(0)
        attention_layer = Attention(rngs=rngs)

        configs = [
            (1, 2, 16, 32),   # Small config
            (2, 4, 32, 64),   # Medium config
            (1, 8, 64, 128),  # Large config
        ]

        for batch_size, num_heads, seq_len, head_dim in configs:
            with self.subTest(batch_size=batch_size, num_heads=num_heads, seq_len=seq_len, head_dim=head_dim):
                q, k, v, _ = self._create_test_inputs(
                    batch_size, num_heads, seq_len, head_dim)
                self._assert_attention_correctness(attention_layer, q, k, v)

    def test_attention_with_mask(self):
        """Test attention with custom mask."""
        rngs = nnx.Rngs(0)
        attention_layer = Attention(rngs=rngs)

        q, k, v, attention_mask = self._create_test_inputs()

        # Test with mask
        output_with_mask = attention_layer(q, k, v, attention_mask)
        output_without_mask = attention_layer(q, k, v)

        # Should be different
        self.assertFalse(jnp.allclose(output_with_mask, output_without_mask))

        # Should match cuDNN reference
        self._assert_attention_correctness(
            attention_layer, q, k, v, attention_mask)

    def test_attention_dtypes(self):
        """Test attention with different data types."""
        for dtype in [jnp.float32, jnp.float16]:
            with self.subTest(dtype=dtype):
                rngs = nnx.Rngs(0)
                attention_layer = Attention(rngs=rngs)

                q, k, v, _ = self._create_test_inputs(dtype=dtype)
                output = attention_layer(q, k, v)

                self.assertEqual(output.dtype, dtype)
                self.assertTrue(jnp.all(jnp.isfinite(output)))

    def test_attention_zero_inputs(self):
        """Test attention with zero inputs."""
        rngs = nnx.Rngs(0)
        attention_layer = Attention(rngs=rngs)

        q = jnp.zeros((1, 8, 2, 16), dtype=jnp.float32)
        k = jnp.zeros((1, 8, 2, 16), dtype=jnp.float32)
        v = jnp.zeros((1, 8, 2, 16), dtype=jnp.float32)

        output = attention_layer(q, k, v)
        self.assertTrue(jnp.allclose(output, jnp.zeros_like(output)))

    def test_attention_gradients(self):
        """Test gradient flow through attention."""
        q, k, v, _ = self._create_test_inputs(1, 2, 8, 16)

        def loss_fn(q, k, v):
            return jnp.sum(self.attention_layer(q, k, v) ** 2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        for i, grad in enumerate(grads):
            self.assertEqual(grad.shape, [q, k, v][i].shape)
            self.assertTrue(jnp.all(jnp.isfinite(grad)))
            self.assertFalse(jnp.allclose(grad, jnp.zeros_like(grad)))

    def test_attention_deterministic(self):
        """Test deterministic behavior."""
        rngs = nnx.Rngs(0)
        attention_layer = Attention(rngs=rngs)

        q, k, v, _ = self._create_test_inputs()

        output1 = attention_layer(q, k, v)
        output2 = attention_layer(q, k, v)

        self.assertTrue(jnp.array_equal(output1, output2))


if __name__ == '__main__':
    unittest.main()
