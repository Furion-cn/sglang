import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from flax import nnx

from sglang.srt.jax.layers.attention import Attention
from sglang.srt.jax.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.test_utils import CustomTestCase


def create_forward_batch(seq_lengths, input_ids=None):
    """Create a real ForwardBatch for testing."""
    batch_size = len(seq_lengths)
    total_tokens = sum(seq_lengths)

    # Create dummy input_ids if not provided
    if input_ids is None:
        input_ids = jnp.arange(total_tokens, dtype=jnp.int32)

    # Create sequence lengths array
    seq_lens = jnp.array(seq_lengths, dtype=jnp.int32)

    # Create positions (sequential positions for each token)
    positions = jnp.arange(total_tokens, dtype=jnp.int32)

    # Create extend_start_loc (start position of each sequence)
    extend_start_loc = jnp.array([sum(seq_lengths[:i])
                                 for i in range(batch_size)], dtype=jnp.int32)

    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=batch_size,
        input_ids=input_ids,
        seq_lens=seq_lens,
        positions=positions,
        extend_start_loc=extend_start_loc,
        total_tokens=total_tokens
    )


class TestAttention(CustomTestCase):
    """Test cases for the Attention layer."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

        # PyTorch comparison test parameters
        try:
            import torch
            self.pytorch_available = True
        except ImportError:
            self.pytorch_available = False

        self.seq_len = 16
        self.num_heads = 8
        self.head_dim = 64
        self.total_dim = self.num_heads * self.head_dim
        self.scale = (self.head_dim ** -0.5)

        # Initialize random seeds for reproducible results
        self.rng_key = jax.random.PRNGKey(42)
        if self.pytorch_available:
            torch.manual_seed(42)
        np.random.seed(42)

    def _generate_test_data_seq_format(self, seq_len=None, total_dim=None, dtype=jnp.bfloat16):
        """Generate test data with input shape [seq_len, total_dim]."""
        seq_len = seq_len or self.seq_len
        total_dim = total_dim or self.total_dim

        # Generate random q, k, v matrices
        key1, key2, key3 = jax.random.split(self.rng_key, 3)

        q_jax = jax.random.normal(key1, (seq_len, total_dim), dtype=dtype)
        k_jax = jax.random.normal(key2, (seq_len, total_dim), dtype=dtype)
        v_jax = jax.random.normal(key3, (seq_len, total_dim), dtype=dtype)

        return q_jax, k_jax, v_jax

    def _pytorch_reference_attention(self, q_jax, k_jax, v_jax, num_heads=None, scale=None):
        """Use PyTorch F.scaled_dot_product_attention as reference implementation."""
        if not self.pytorch_available:
            self.skipTest("PyTorch not available")

        num_heads = num_heads or self.num_heads
        scale = scale or self.scale

        seq_len = q_jax.shape[0]
        total_dim = q_jax.shape[-1]
        head_dim = total_dim // num_heads

        # Convert to PyTorch tensors with bf16 precision
        # First convert to float32 to avoid ml_dtypes.bfloat16 conversion issues
        q_pt = torch.from_numpy(np.asarray(
            q_jax.astype(jnp.float32))).to(torch.bfloat16)
        k_pt = torch.from_numpy(np.asarray(
            k_jax.astype(jnp.float32))).to(torch.bfloat16)
        v_pt = torch.from_numpy(np.asarray(
            v_jax.astype(jnp.float32))).to(torch.bfloat16)

        # Reshape to [seq_len, num_heads, head_dim]
        q_reshaped = q_pt.view(seq_len, num_heads, head_dim)
        k_reshaped = k_pt.view(seq_len, num_heads, head_dim)
        v_reshaped = v_pt.view(seq_len, num_heads, head_dim)

        # PyTorch expects [batch_size, num_heads, seq_len, head_dim]
        # So we transpose: [1, num_heads, seq_len, head_dim]
        q_attn = q_reshaped.transpose(0, 1).unsqueeze(0)
        k_attn = k_reshaped.transpose(0, 1).unsqueeze(0)
        v_attn = v_reshaped.transpose(0, 1).unsqueeze(0)

        # Apply attention mechanism
        with torch.no_grad():
            attn_out = F.scaled_dot_product_attention(
                q_attn, k_attn, v_attn,
                scale=scale,
                is_causal=True
            )

        # Convert back to [seq_len, total_dim]
        # attn_out shape: [1, num_heads, seq_len, head_dim]
        # We need to convert to [seq_len, num_heads, head_dim] to match JAX output
        attn_output_pt = attn_out.squeeze(0).transpose(
            0, 1).contiguous().view(seq_len, total_dim)

        # Convert back to JAX array with bf16 precision
        return jnp.asarray(attn_output_pt.to(torch.float32).numpy()).astype(jnp.bfloat16)

    def test_attention_basic_functionality(self):
        """Test basic functionality of attention with variable-length sequences"""
        # Parameters
        num_heads = 8
        head_dim = 64
        hidden_size = num_heads * head_dim
        scale = head_dim ** -0.5

        # Variable-length sequence setup
        batch_size = 3
        seq_lengths = [4, 6, 8]
        total_tokens = sum(seq_lengths)

        # Create mock forward_batch
        forward_batch = create_forward_batch(seq_lengths)

        # Create attention layer
        attention = Attention(num_heads=num_heads, scale=scale)

        # Generate test data in [total_tokens, hidden_size] format
        q = jax.random.normal(self.rng_key, (total_tokens, hidden_size))
        k = jax.random.normal(jax.random.split(self.rng_key)[
                              0], (total_tokens, hidden_size))
        v = jax.random.normal(jax.random.split(self.rng_key)[
                              1], (total_tokens, hidden_size))

        # Test attention
        output = attention(q, k, v, forward_batch, is_causal=True)

        # Check output shape and properties
        self.assertEqual(output.shape, (total_tokens, hidden_size))
        self.assertTrue(jnp.isfinite(output).all())
        self.assertEqual(output.dtype, q.dtype)

    def test_attention_accuracy(self):
        """Test JAX attention accuracy against PyTorch reference"""
        import torch
        import torch.nn.functional as F

        # Parameters
        num_heads = 8
        head_dim = 64
        hidden_size = num_heads * head_dim
        scale = head_dim ** -0.5
        batch_size = 2
        seq_lengths = [6, 8]
        total_tokens = sum(seq_lengths)
        max_seq_len = max(seq_lengths)

        # Create mock forward_batch
        forward_batch = create_forward_batch(seq_lengths)

        # Create test data
        key = jax.random.PRNGKey(42)
        q_jax = jax.random.normal(
            key, (total_tokens, hidden_size), dtype=jnp.bfloat16)
        k_jax = jax.random.normal(jax.random.split(
            key)[0], (total_tokens, hidden_size), dtype=jnp.bfloat16)
        v_jax = jax.random.normal(jax.random.split(
            key)[1], (total_tokens, hidden_size), dtype=jnp.bfloat16)

        # JAX attention
        jax_attention = Attention(num_heads=num_heads, scale=scale)
        jax_output = jax_attention(
            q_jax, k_jax, v_jax, forward_batch, is_causal=True)

        # Create PyTorch equivalent data
        def to_pytorch_batched(tensor, seq_lengths, max_seq_len):
            """Convert JAX tensor to PyTorch batched format."""
            batch_size = len(seq_lengths)
            hidden_size = tensor.shape[-1]

            # Convert to float32 first to avoid BFloat16 numpy conversion issues
            tensor = tensor.astype(jnp.float32)
            batched = torch.zeros(
                (batch_size, max_seq_len, hidden_size), dtype=torch.bfloat16)

            start_idx = 0
            for i, seq_len in enumerate(seq_lengths):
                end_idx = start_idx + seq_len
                tensor_slice = tensor[start_idx:end_idx]
                batched[i, :seq_len] = torch.from_numpy(
                    np.asarray(tensor_slice)).to(torch.bfloat16)
                start_idx = end_idx

            return batched

        q_torch = to_pytorch_batched(q_jax, seq_lengths, max_seq_len)
        k_torch = to_pytorch_batched(k_jax, seq_lengths, max_seq_len)
        v_torch = to_pytorch_batched(v_jax, seq_lengths, max_seq_len)

        # Create attention mask for variable lengths with proper shape for PyTorch
        # Shape: [batch_size, 1, max_seq_len, max_seq_len] - broadcasts across heads
        attention_mask = torch.zeros(
            (batch_size, 1, max_seq_len, max_seq_len), dtype=torch.bool)
        for i, seq_len in enumerate(seq_lengths):
            # Create causal mask for this sequence
            causal_mask = torch.tril(torch.ones(
                seq_len, seq_len, dtype=torch.bool))
            attention_mask[i, 0, :seq_len, :seq_len] = causal_mask

        # Reshape for PyTorch attention: [batch_size, num_heads, seq_len, head_dim]
        q_torch = q_torch.view(batch_size, max_seq_len,
                               num_heads, head_dim).transpose(1, 2)
        k_torch = k_torch.view(batch_size, max_seq_len,
                               num_heads, head_dim).transpose(1, 2)
        v_torch = v_torch.view(batch_size, max_seq_len,
                               num_heads, head_dim).transpose(1, 2)

        # PyTorch scaled dot product attention with proper padding mask
        with torch.no_grad():
            pytorch_output = F.scaled_dot_product_attention(
                q_torch, k_torch, v_torch,
                attn_mask=attention_mask,
                is_causal=False,  # We handle causality through the mask
                scale=scale
            )

        # Convert back to flat format
        pytorch_output = pytorch_output.transpose(
            1, 2).contiguous().view(batch_size, max_seq_len, hidden_size)
        pytorch_output_flat = []
        for i, seq_len in enumerate(seq_lengths):
            pytorch_output_flat.append(pytorch_output[i, :seq_len])
        pytorch_output_flat = torch.cat(pytorch_output_flat, dim=0)

        # Compare results - convert to float32 for proper comparison
        jax_np = np.array(jax_output.astype(jnp.float32))
        pytorch_np = pytorch_output_flat.to(torch.float32).numpy()

        abs_diff = np.abs(jax_np - pytorch_np)
        rel_diff = abs_diff / (np.abs(pytorch_np) + 1e-8)

        max_abs_error = np.max(abs_diff)
        mean_abs_error = np.mean(abs_diff)
        max_rel_error = np.max(rel_diff)
        mean_rel_error = np.mean(rel_diff)

        print(f"JAX output shape: {jax_output.shape}")
        print(f"PyTorch output shape: {pytorch_output_flat.shape}")
        print(f"Max absolute error: {max_abs_error:.8f}")
        print(f"Mean absolute error: {mean_abs_error:.8f}")
        print(f"Max relative error: {max_rel_error:.8f}")
        print(f"Mean relative error: {mean_rel_error:.8f}")

        # Test with reasonable thresholds for BFloat16 precision
        rtol = 2e-2  # Relative tolerance
        atol = 1e-2  # Absolute tolerance
        are_close = np.allclose(jax_np, pytorch_np, rtol=rtol, atol=atol)
        print(f"Are outputs close (rtol={rtol}, atol={atol})? {are_close}")

        assert are_close, f"JAX and PyTorch outputs differ significantly! Max abs error: {max_abs_error}, Max rel error: {max_rel_error}"


class TestGroupedQueryAttention(CustomTestCase):
    """Test cases for the GroupedQueryAttention layer."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _create_test_inputs(self, batch_size=2, num_q_heads=8, num_kv_heads=2, seq_len=32, head_dim=64, dtype=jnp.float32, seed=0):
        """Create test inputs for GQA computation."""
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 4)

        assert num_q_heads % num_kv_heads == 0

        q = jax.random.normal(
            keys[0], (batch_size, seq_len, num_q_heads, head_dim), dtype=dtype)
        k = jax.random.normal(
            keys[1], (batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype)
        v = jax.random.normal(
            keys[2], (batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype)

        attention_mask = jax.random.bernoulli(
            keys[3], 0.8, (batch_size, 1, seq_len, seq_len)).astype(dtype) * -1e9

        return q, k, v, attention_mask

    def _get_reference_output(self, q, k, v, attention_mask=None, is_causal=True):
        """Get JAX reference result."""
        try:
            return jax.nn.dot_product_attention(
                query=q, key=k, value=v,
                bias=attention_mask,
                is_causal=is_causal
            )
        except Exception as e:
            self.skipTest(f"JAX dot_product_attention not available: {e}")

    def _assert_attention_correctness(self, attention_layer, q, k, v, attention_mask=None, is_causal=True):
        """Assert attention layer output matches JAX reference."""
        output = attention_layer(q, k, v, attention_mask=attention_mask, is_causal=is_causal)
        output_ref = self._get_reference_output(q, k, v, attention_mask=attention_mask, is_causal=is_causal)

        self.assertEqual(output.shape, output_ref.shape)
        # Using a slightly looser tolerance for mixed precision
        atol = 1e-2 if output.dtype == jnp.float16 else 1e-5
        rtol = 1e-2 if output.dtype == jnp.float16 else 1e-5
        self.assertTrue(jnp.allclose(output, output_ref, atol=atol, rtol=rtol))

    def test_gqa_basic_causal(self):
        """Test basic GQA functionality with various configurations (causal)."""
        attention_layer = Attention()

        configs = [
            (1, 8, 2, 16, 32),
            (2, 16, 4, 32, 64),
            (1, 32, 8, 64, 128),
            (2, 8, 8, 16, 32),  # MHA case
        ]

        for batch_size, num_q_heads, num_kv_heads, seq_len, head_dim in configs:
            with self.subTest(b=batch_size, n_q=num_q_heads, n_kv=num_kv_heads, s=seq_len, h=head_dim):
                q, k, v, _ = self._create_test_inputs(
                    batch_size, num_q_heads, num_kv_heads, seq_len, head_dim)
                self._assert_attention_correctness(attention_layer, q, k, v, is_causal=True)

    def test_gqa_with_mask(self):
        """Test GQA with a custom mask (non-causal)."""
        attention_layer = Attention()
        q, k, v, attention_mask = self._create_test_inputs()

        # Test with mask and non-causal
        self._assert_attention_correctness(attention_layer, q, k, v, attention_mask=attention_mask, is_causal=False)

    def test_gqa_with_mask_and_causal(self):
        """Test GQA with both a custom mask and causal masking."""
        attention_layer = Attention()
        q, k, v, attention_mask = self._create_test_inputs()

        # Test with both mask and causal
        self._assert_attention_correctness(attention_layer, q, k, v, attention_mask=attention_mask, is_causal=True)

    def test_gqa_dtypes(self):
        """Test GQA with different data types."""
        for dtype in [jnp.float32, jnp.float16]:
            with self.subTest(dtype=dtype):
                attention_layer = Attention()
                q, k, v, _ = self._create_test_inputs(dtype=dtype)
                output = attention_layer(q, k, v)
                self.assertEqual(output.dtype, dtype)
                self.assertTrue(jnp.all(jnp.isfinite(output)))
                self._assert_attention_correctness(attention_layer, q, k, v)

    def test_gqa_gradients(self):
        """Test gradient flow through GQA."""
        attention_layer = Attention()
        q, k, v, _ = self._create_test_inputs(batch_size=1, num_q_heads=4, num_kv_heads=2, seq_len=8, head_dim=16)

        def loss_fn(q_in, k_in, v_in):
            # A simple loss function
            return jnp.sum(attention_layer(q_in, k_in, v_in) ** 2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        for i, grad in enumerate(grads):
            self.assertEqual(grad.shape, [q, k, v][i].shape)
            self.assertTrue(jnp.all(jnp.isfinite(grad)))
            # Ensure gradients are not all zero, which would indicate a problem
            self.assertFalse(jnp.allclose(grad, jnp.zeros_like(grad)))

if __name__ == '__main__':
    unittest.main()
