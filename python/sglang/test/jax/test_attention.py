import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from flax import nnx
from sglang.srt.jax.mem_cache.hash_kvcache import ReqToHashKVCachePool
from sglang.srt.jax.layers.attention import Attention
from sglang.srt.jax.model_executor.forward_batch_info import ForwardBatch, ForwardMode,FORWARD_MODE_EXTEND,FORWARD_MODE_DECODE
from sglang.test.test_utils import CustomTestCase


def create_forward_batch(seq_lengths, input_ids=None, model_config=None):
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

    current_kv_cache = [ReqToHashKVCachePool(
            seq_len=seq_len,
            head_num=model_config["num_kv_heads"],
            head_dim=model_config["head_dim"],
            layer_num=model_config["num_hidden_layers"],
            dtype=jnp.bfloat16 if model_config["bf16"] else jnp.float32
        ) for seq_len in seq_lens]
    # TODO: aolemila
    return ForwardBatch(
        batch_size=batch_size,
        input_ids=input_ids,
        seq_lens=seq_lens,
        positions=positions,
        extend_start_loc=extend_start_loc,
        current_kv_cache=current_kv_cache,
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
        forward_batch = create_forward_batch(seq_lengths, model_config={
            "num_kv_heads": num_heads,
            "head_dim": head_dim,
            "num_hidden_layers": 1,
            "bf16": True
        })

        # Create attention layer
        attention = Attention(num_heads=num_heads, scale=scale)

        # Generate test data in [total_tokens, hidden_size] format
        q = jax.random.normal(self.rng_key, (total_tokens, hidden_size))
        k = jax.random.normal(jax.random.split(self.rng_key)[
                              0], (total_tokens, hidden_size))
        v = jax.random.normal(jax.random.split(self.rng_key)[
                              1], (total_tokens, hidden_size))

        # Test attention
        output = attention(q, k, v, layer_id=0, forward_batch=forward_batch, is_causal=True,forward_mode=FORWARD_MODE_EXTEND)

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
        forward_batch = create_forward_batch(seq_lengths, model_config={
            "num_kv_heads": num_heads,
            "head_dim": head_dim,
            "num_hidden_layers": 1,
            "bf16": True
        })

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
            q_jax, k_jax, v_jax, layer_id=0, forward_batch=forward_batch, is_causal=True,forward_mode=FORWARD_MODE_EXTEND)

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

    def _create_test_inputs(self, total_tokens=17, num_q_heads=8, num_kv_heads=2, head_dim=64, dtype=jnp.float32, seed=0):
        """Create test inputs for GQA computation."""
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 4)

        assert num_q_heads % num_kv_heads == 0

        # 创建展平的输入格式 [total_tokens, hidden_size]
        q = jax.random.normal(
            keys[0], (total_tokens, num_q_heads * head_dim), dtype=dtype)
        k = jax.random.normal(
            keys[1], (total_tokens, num_kv_heads * head_dim), dtype=dtype)
        v = jax.random.normal(
            keys[2], (total_tokens, num_kv_heads * head_dim), dtype=dtype)

        # 创建序列长度信息
        seq_lengths = jnp.array([5, 7, 5], dtype=jnp.int32)  # 示例：3个序列，总长度17
        
        # 创建ForwardBatch对象
        forward_batch = create_forward_batch(seq_lengths, model_config={
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "num_hidden_layers": 1,
            "bf16": True
        })

        return q, k, v, forward_batch

    def test_gqa_attention_basic_functionality(self):
        """Test basic functionality of gqa attention with variable-length sequences"""
        # Parameters
        num_heads = 8
        num_kv_heads = 2
        head_dim = 64
        hidden_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        scale = head_dim ** -0.5

        # Variable-length sequence setup
        batch_size = 3
        seq_lengths = [4, 6, 8]
        total_tokens = sum(seq_lengths)

        # Create mock forward_batch
        forward_batch = create_forward_batch(seq_lengths, model_config={
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "num_hidden_layers": 1,
            "bf16": True
        })

        # Create attention layer
        attention = Attention(num_heads=num_heads, num_kv_heads=num_kv_heads, scale=scale)

        # Generate test data in [total_tokens, hidden_size] format
        q = jax.random.normal(self.rng_key, (total_tokens, hidden_size))
        k = jax.random.normal(jax.random.split(self.rng_key)[
                              0], (total_tokens, kv_size))
        v = jax.random.normal(jax.random.split(self.rng_key)[
                              1], (total_tokens, kv_size))

        # Test attention
        output = attention(q, k, v, layer_id=0, forward_batch=forward_batch, is_causal=True,forward_mode=FORWARD_MODE_EXTEND)

        # Check output shape and properties
        self.assertEqual(output.shape, (total_tokens, hidden_size))
        self.assertTrue(jnp.isfinite(output).all())
        self.assertEqual(output.dtype, q.dtype)

    def test_gqa_dtypes(self):
        """Test GQA with different data types."""
        attention_layer = Attention(
            num_heads=8,
            num_kv_heads=2,
            scale=1.0 / jnp.sqrt(64)
        )

        for dtype in [jnp.float32, jnp.float16, jnp.bfloat16]:
            with self.subTest(dtype=dtype):
                q, k, v, forward_batch = self._create_test_inputs(dtype=dtype)
                output = attention_layer(q, k, v, layer_id=0, forward_batch=forward_batch,forward_mode=FORWARD_MODE_EXTEND)
                
                self.assertEqual(output.dtype, dtype)
                self.assertTrue(jnp.all(jnp.isfinite(output)))
    
    def test_attention_accuracy(self):
        """Test JAX attention accuracy against PyTorch reference"""
        import torch
        import torch.nn.functional as F

        # Parameters
        num_heads = 8
        head_dim = 64
        num_kv_heads = 2
        hidden_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        scale = head_dim ** -0.5
        batch_size = 2
        seq_lengths = [6, 8]
        total_tokens = sum(seq_lengths)
        max_seq_len = max(seq_lengths)

        # Create mock forward_batch
        forward_batch = create_forward_batch(seq_lengths, model_config={
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "num_hidden_layers": 1,
            "bf16": True
        })

        # Create test data
        key = jax.random.PRNGKey(42)
        q_jax = jax.random.normal(
            key, (total_tokens, hidden_size), dtype=jnp.bfloat16)
        k_jax = jax.random.normal(jax.random.split(
            key)[0], (total_tokens, kv_size), dtype=jnp.bfloat16)
        v_jax = jax.random.normal(jax.random.split(
            key)[1], (total_tokens, kv_size), dtype=jnp.bfloat16)

        # JAX attention
        jax_attention = Attention(num_heads=num_heads, num_kv_heads=num_kv_heads, scale=scale)
        jax_output = jax_attention(
            q_jax, k_jax, v_jax, layer_id=0, forward_batch=forward_batch, is_causal=True,forward_mode=FORWARD_MODE_EXTEND)

        # Create PyTorch equivalent data
        def to_pytorch_batched(tensor, seq_lengths, max_seq_len):
            """Convert JAX tensor to PyTorch batched format."""
            batch_size = len(seq_lengths)
            size = tensor.shape[-1]

            # Convert to float32 first to avoid BFloat16 numpy conversion issues
            tensor = tensor.astype(jnp.float32)
            batched = torch.zeros(
                (batch_size, max_seq_len, size), dtype=torch.bfloat16)

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
                               num_kv_heads, head_dim).transpose(1, 2)
        v_torch = v_torch.view(batch_size, max_seq_len,
                               num_kv_heads, head_dim).transpose(1, 2)

        # 计算每个 KV head 需要重复的次数
        num_repeats = num_heads // num_kv_heads

        # 重复 k_torch 和 v_torch 以匹配 q_torch 的 head 数量
        # [batch_size, num_kv_heads, seq_len, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        k_torch = k_torch.repeat_interleave(num_repeats, dim=1)  # 在 head 维度上重复
        v_torch = v_torch.repeat_interleave(num_repeats, dim=1)  # 在 head 维度上重复

        # 验证维度匹配
        assert q_torch.shape == k_torch.shape == v_torch.shape, \
            f"Shape mismatch: q={q_torch.shape}, k={k_torch.shape}, v={v_torch.shape}"

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

if __name__ == '__main__':
    unittest.main()
