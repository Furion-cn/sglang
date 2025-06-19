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
        print(f"\n=== Attention Correctness Check ===")
        print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
        if attention_mask is not None:
            print(f"Attention mask shape: {attention_mask.shape}")
        
        # 计算我们的实现
        output = attention_layer(q, k, v, attention_mask)
        print(f"Our implementation output shape: {output.shape}")
        print(f"Our implementation output dtype: {output.dtype}")
        print(f"Our implementation - mean: {jnp.mean(output):.6f}, std: {jnp.std(output):.6f}")
        print(f"Our implementation - min: {jnp.min(output):.6f}, max: {jnp.max(output):.6f}")
        
        # 计算参考实现
        output_ref = self._get_cudnn_reference(q, k, v, attention_mask)
        print(f"Reference output shape: {output_ref.shape}")
        print(f"Reference output dtype: {output_ref.dtype}")
        print(f"Reference - mean: {jnp.mean(output_ref):.6f}, std: {jnp.std(output_ref):.6f}")
        print(f"Reference - min: {jnp.min(output_ref):.6f}, max: {jnp.max(output_ref):.6f}")
        
        # 计算差异
        diff = output - output_ref
        abs_diff = jnp.abs(diff)
        print(f"Difference - mean: {jnp.mean(diff):.6f}, std: {jnp.std(diff):.6f}")
        print(f"Absolute difference - mean: {jnp.mean(abs_diff):.6f}, max: {jnp.max(abs_diff):.6f}")
        
        # 相对误差
        rel_error = abs_diff / (jnp.abs(output_ref) + 1e-8)
        print(f"Relative error - mean: {jnp.mean(rel_error):.6f}, max: {jnp.max(rel_error):.6f}")
        
        # 检查是否接近
        is_close_strict = jnp.allclose(output, output_ref, atol=1e-4, rtol=1e-3)
        is_close_loose = jnp.allclose(output, output_ref, atol=1e-3, rtol=1e-2)
        
        print(f"Close with atol=1e-4, rtol=1e-3: {is_close_strict}")
        print(f"Close with atol=1e-3, rtol=1e-2: {is_close_loose}")
        
        # 找出不相同的位置
        tolerance = 1e-4
        not_close_mask = abs_diff > tolerance
        num_different = jnp.sum(not_close_mask)
        total_elements = output.size
        
        print(f"Elements with abs diff > {tolerance}: {num_different}/{total_elements} ({100*num_different/total_elements:.2f}%)")
        
        if num_different > 0:
            # 找出差异最大的几个位置
            flat_abs_diff = abs_diff.flatten()
            flat_output = output.flatten()
            flat_ref = output_ref.flatten()
            flat_diff = diff.flatten()
            
            # 获取差异最大的前10个位置
            top_diff_indices = jnp.argsort(flat_abs_diff)[-min(10, num_different):]
            
            print(f"\nTop {len(top_diff_indices)} largest differences:")
            for i, idx in enumerate(reversed(top_diff_indices)):
                idx = int(idx)
                # 将平坦索引转换回多维索引
                multi_idx = jnp.unravel_index(idx, output.shape)
                print(f"  #{i+1} at {multi_idx}: our={flat_output[idx]:.6f}, ref={flat_ref[idx]:.6f}, diff={flat_diff[idx]:.6f}")
            
            # 如果差异不多，显示所有不同的位置
            if num_different <= 20:
                print(f"\nAll {num_different} different positions:")
                different_indices = jnp.where(not_close_mask)
                for i in range(num_different):
                    pos = tuple(int(different_indices[j][i]) for j in range(len(different_indices)))
                    our_val = output[pos]
                    ref_val = output_ref[pos]
                    diff_val = diff[pos]
                    print(f"  {pos}: our={our_val:.6f}, ref={ref_val:.6f}, diff={diff_val:.6f}")
            
            # 检查是否有模式
            print(f"\nDifference patterns:")
            print(f"  Positive differences: {jnp.sum(diff > tolerance)}")
            print(f"  Negative differences: {jnp.sum(diff < -tolerance)}")
            print(f"  Max positive diff: {jnp.max(diff):.6f}")
            print(f"  Max negative diff: {jnp.min(diff):.6f}")
            
            # 检查是否是特定维度的问题
            if len(output.shape) == 4:  # [batch, seq, heads, head_dim]
                for dim in range(4):
                    dim_diff = jnp.mean(abs_diff, axis=tuple(i for i in range(4) if i != dim))
                    max_dim_diff = jnp.max(dim_diff)
                    argmax_dim = jnp.argmax(dim_diff)
                    print(f"  Dim {dim} max avg diff: {max_dim_diff:.6f} at index {argmax_dim}")

            self.assertTrue(num_different == 0)
        else:
            print(f"✓ All elements are within tolerance!")
        
        # 检查是否完全相等
        exactly_equal = jnp.array_equal(output, output_ref)
        print(f"Exactly equal: {exactly_equal}")
        
        # 形状检查
        expected_shape = (q.shape[0], q.shape[1], q.shape[2], q.shape[3])
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(jnp.all(jnp.isfinite(output)))
        
        print(f"=== End Correctness Check ===\n")
        
        # 使用适当的容差进行断言
        if not is_close_loose:
            self.fail(f"Outputs differ significantly. {num_different}/{total_elements} elements differ by > {tolerance}")

    def test_attention_basic(self):
        """Test basic attention functionality with various configurations."""
        # Create Attention layer without any parameters
        attention_layer = Attention()

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
        attention_layer = Attention()

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
                attention_layer = Attention()

                q, k, v, _ = self._create_test_inputs(dtype=dtype)
                output = attention_layer(q, k, v)

                self.assertEqual(output.dtype, dtype)
                self.assertTrue(jnp.all(jnp.isfinite(output)))

    def test_attention_zero_inputs(self):
        """Test attention with zero inputs."""
        attention_layer = Attention()

        q = jnp.zeros((1, 8, 2, 16), dtype=jnp.float32)
        k = jnp.zeros((1, 8, 2, 16), dtype=jnp.float32)
        v = jnp.zeros((1, 8, 2, 16), dtype=jnp.float32)

        output = attention_layer(q, k, v)
        self.assertTrue(jnp.allclose(output, jnp.zeros_like(output)))

    def test_attention_gradients(self):
        """Test gradient flow through attention."""
        attention_layer = Attention()
        q, k, v, _ = self._create_test_inputs(1, 2, 8, 16)

        def loss_fn(q, k, v):
            return jnp.sum(attention_layer(q, k, v) ** 2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        for i, grad in enumerate(grads):
            self.assertEqual(grad.shape, [q, k, v][i].shape)
            self.assertTrue(jnp.all(jnp.isfinite(grad)))
            self.assertFalse(jnp.allclose(grad, jnp.zeros_like(grad)))

    def test_attention_deterministic(self):
        """Test deterministic behavior."""
        attention_layer = Attention()

        q, k, v, _ = self._create_test_inputs()

        output1 = attention_layer(q, k, v)
        output2 = attention_layer(q, k, v)

        self.assertTrue(jnp.array_equal(output1, output2))


if __name__ == '__main__':
    unittest.main()
