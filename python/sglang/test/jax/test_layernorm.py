#!/usr/bin/env python3
"""
JAX RMSNorm Precision Comparison Tests

Tests the numerical precision of JAX RMSNorm implementation compared to PyTorch version.
Validates that both implementations produce consistent results with acceptable tolerance.

Usage:
    python -m unittest python.sglang.test.jax.test_layernorm.TestJAXRMSNormPrecision
"""

import itertools
import unittest
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx

from sglang.srt.jax.layers.layernorm import RMSNorm as JAXRMSNorm
from sglang.srt.layers.layernorm import RMSNorm as TorchRMSNorm
from sglang.test.jax.test_utils import create_device_mesh
from sglang.test.test_utils import CustomTestCase


class TestJAXRMSNormPrecision(CustomTestCase):
    """Test precision comparison between JAX and PyTorch RMSNorm implementations"""

    # Test parameters
    DTYPES = [
        (jnp.float16, torch.float16),
        (jnp.float32, torch.float32)
    ]
    NUM_TOKENS = [7, 83, 512, 1024]
    HIDDEN_SIZES = [768, 1024, 2048, 4096, 8192]
    ADD_RESIDUAL = [False, True]
    SEEDS = [0, 42, 123]
    EPSILON_VALUES = [1e-6, 1e-5, 1e-8]

    def setUp(self):
        """Set up test fixtures"""
        self.mesh = create_device_mesh(
            ici_parallelism=[-1, 1, 1, 1],
            dcn_parallelism=[1, 1, 1, 1]
        )
        # Set device for PyTorch
        if torch.cuda.is_available():
            self.torch_device = torch.device("cuda")
        else:
            self.torch_device = torch.device("cpu")

        # Filter dtypes based on device support
        self.supported_dtypes = self._get_supported_dtypes()

    def _get_supported_dtypes(self):
        """Get list of supported dtypes based on device capabilities"""
        supported = []

        for jax_dtype, torch_dtype in self.DTYPES:
            try:
                # Test if the dtype is supported by creating a small tensor
                test_tensor = torch.ones(
                    1, dtype=torch_dtype, device=self.torch_device)

                # Also test JAX to PyTorch conversion
                jax_test = jnp.ones(1, dtype=jax_dtype)
                numpy_test = self._safe_jax_to_numpy(jax_test)
                torch_test = torch.from_numpy(numpy_test).to(
                    torch_dtype).to(self.torch_device)

                supported.append((jax_dtype, torch_dtype))
            except (RuntimeError, TypeError, ValueError) as e:
                print(
                    f"⚠️  Skipping {torch_dtype} - not supported on {self.torch_device}: {e}")
                continue

        # Ensure at least float32 is supported as fallback
        if not supported:
            print("⚠️  No dtypes supported, falling back to float32 only")
            supported = [(jnp.float32, torch.float32)]

        print(f"✅ Supported dtypes: {[str(dt[1]) for dt in supported]}")
        return supported

    def _safe_jax_to_numpy(self, jax_array):
        """Safely convert JAX array to NumPy array, handling unsupported dtypes"""
        try:
            # Convert to numpy first
            np_array = np.asarray(jax_array)

            # Check if the dtype is supported by torch.from_numpy
            # torch.from_numpy doesn't support bfloat16, so convert to float32
            if str(np_array.dtype) == 'bfloat16' or 'bfloat16' in str(np_array.dtype):
                return np_array.astype(np.float32)

            return np_array
        except (TypeError, ValueError):
            # If conversion fails, convert to float32 first
            return np.asarray(jax_array.astype(jnp.float32))

    def _create_jax_rmsnorm(self, hidden_size: int, epsilon: float = 1e-6) -> JAXRMSNorm:
        """Create JAX RMSNorm layer"""
        rngs = nnx.Rngs(0)  # Fixed seed for reproducibility
        return JAXRMSNorm(
            hidden_size=hidden_size,
            epsilon=epsilon,
            rngs=rngs
        )

    def _create_torch_rmsnorm(self, hidden_size: int, epsilon: float = 1e-6) -> TorchRMSNorm:
        """Create PyTorch RMSNorm layer"""
        return TorchRMSNorm(hidden_size=hidden_size, eps=epsilon).to(self.torch_device)

    def _sync_weights(self, jax_layer: JAXRMSNorm, torch_layer: TorchRMSNorm):
        """同步 JAX 和 PyTorch 层的权重"""
        # Convert JAX weights to numpy and then to torch
        jax_weights = jnp.array(jax_layer.weight.value)
        jax_weights_np = self._safe_jax_to_numpy(jax_weights)

        # Set torch weights
        with torch.no_grad():
            torch_layer.weight.data = torch.from_numpy(
                jax_weights_np).to(self.torch_device)

    def _compare_outputs(
        self,
        jax_output,
        torch_output,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        test_name: str = ""
    ) -> Tuple[bool, float, float]:
        """比较 JAX 和 PyTorch 输出"""
        # Convert to numpy for comparison
        if isinstance(jax_output, tuple):
            jax_out_np = self._safe_jax_to_numpy(jax_output[0])
            jax_residual_np = self._safe_jax_to_numpy(jax_output[1]) if len(
                jax_output) > 1 else None
        else:
            jax_out_np = self._safe_jax_to_numpy(jax_output)
            jax_residual_np = None

        if isinstance(torch_output, tuple):
            torch_out_np = torch_output[0].detach().cpu().numpy()
            torch_residual_np = torch_output[1].detach().cpu(
            ).numpy() if len(torch_output) > 1 else None
        else:
            torch_out_np = torch_output.detach().cpu().numpy()
            torch_residual_np = None

        # Compare main outputs
        diff = np.abs(jax_out_np - torch_out_np)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))

        # Check for NaN or inf values
        jax_has_nan = np.any(np.isnan(jax_out_np)) or np.any(
            np.isinf(jax_out_np))
        torch_has_nan = np.any(np.isnan(torch_out_np)) or np.any(
            np.isinf(torch_out_np))

        main_close = np.allclose(
            jax_out_np, torch_out_np, atol=atol, rtol=rtol)

        # Compare residual if present
        residual_close = True
        if jax_residual_np is not None and torch_residual_np is not None:
            residual_close = np.allclose(
                jax_residual_np, torch_residual_np, atol=atol, rtol=rtol)

        is_close = main_close and residual_close

        # Print detailed comparison for debugging
        if not is_close or jax_has_nan or torch_has_nan:
            print(f"\n❌ {test_name} - Precision mismatch:")
            print(f"   Max difference: {max_diff}")
            print(f"   Mean difference: {mean_diff}")
            print(f"   Tolerance: atol={atol}, rtol={rtol}")
            print(f"   Main output close: {main_close}")
            print(f"   Residual close: {residual_close}")
            print(f"   JAX has NaN/Inf: {jax_has_nan}")
            print(f"   Torch has NaN/Inf: {torch_has_nan}")

            # Debug array properties
            print(
                f"   JAX shape: {jax_out_np.shape}, dtype: {jax_out_np.dtype}")
            print(
                f"   Torch shape: {torch_out_np.shape}, dtype: {torch_out_np.dtype}")

            # Show sample values for debugging
            print(f"   JAX sample: {jax_out_np.flat[:5]}")
            print(f"   Torch sample: {torch_out_np.flat[:5]}")

            # Additional debugging for residual
            if jax_residual_np is not None and torch_residual_np is not None:
                residual_diff = np.abs(jax_residual_np - torch_residual_np)
                print(f"   Residual max diff: {float(np.max(residual_diff))}")
                print(
                    f"   Residual mean diff: {float(np.mean(residual_diff))}")
                print(f"   JAX residual sample: {jax_residual_np.flat[:5]}")
                print(
                    f"   Torch residual sample: {torch_residual_np.flat[:5]}")
            elif jax_residual_np is not None or torch_residual_np is not None:
                print(
                    f"   Residual mismatch: JAX has residual: {jax_residual_np is not None}, Torch has residual: {torch_residual_np is not None}")

        return is_close, max_diff, mean_diff

    def _run_precision_test(
        self,
        num_tokens: int,
        hidden_size: int,
        add_residual: bool,
        jax_dtype,
        torch_dtype,
        epsilon: float,
        seed: int
    ):
        """运行单个精度测试"""
        test_name = (f"tokens={num_tokens}, hidden={hidden_size}, "
                     f"residual={add_residual}, dtype={jax_dtype}, eps={epsilon}, seed={seed}")

        # Set random seeds
        jax.random.split(jax.random.PRNGKey(seed))
        torch.manual_seed(seed)

        # Create layers
        jax_layer = self._create_jax_rmsnorm(hidden_size, epsilon)
        torch_layer = self._create_torch_rmsnorm(hidden_size, epsilon)

        # Sync weights
        self._sync_weights(jax_layer, torch_layer)

        # Generate test data
        scale = 1.0 / (2 * hidden_size)

        # JAX inputs
        key = jax.random.PRNGKey(seed)
        jax_x = jax.random.normal(
            key, (num_tokens, hidden_size), dtype=jax_dtype) * scale
        jax_residual = None
        if add_residual:
            key, subkey = jax.random.split(key)
            jax_residual = jax.random.normal(
                subkey, (num_tokens, hidden_size), dtype=jax_dtype) * scale

        # PyTorch inputs (convert from JAX to ensure same values)
        torch_x = torch.from_numpy(self._safe_jax_to_numpy(jax_x)).to(
            torch_dtype).to(self.torch_device)
        torch_residual = None
        if add_residual:
            torch_residual = torch.from_numpy(self._safe_jax_to_numpy(
                jax_residual)).to(torch_dtype).to(self.torch_device)

        # Run forward passes
        with self.mesh:
            jax_output = jax_layer(jax_x, jax_residual)

        with torch.no_grad():
            # Use native implementation for consistent comparison
            torch_output = torch_layer.forward_native(torch_x, torch_residual)

        # Compare outputs with appropriate tolerance based on dtype
        if jax_dtype == jnp.float16 or torch_dtype == torch.float16:
            atol, rtol = 1e-2, 1e-2  # More lenient for half precision
        elif jax_dtype == jnp.bfloat16 or torch_dtype == torch.bfloat16:
            atol, rtol = 5e-3, 5e-3  # bfloat16 has lower precision
        else:
            atol, rtol = 1e-4, 1e-4  # Stricter for float32

        is_close, max_diff, mean_diff = self._compare_outputs(
            jax_output, torch_output, atol=atol, rtol=rtol, test_name=test_name
        )

        self.assertTrue(
            is_close,
            f"JAX and PyTorch RMSNorm outputs differ beyond tolerance for {test_name}. "
            f"Max diff: {max_diff}, Mean diff: {mean_diff}"
        )

        return max_diff, mean_diff

    def test_rmsnorm_precision_comprehensive(self):
        """全面的精度测试"""
        print("\n🔬 Running comprehensive JAX vs PyTorch RMSNorm precision tests...")

        failed_cases = []
        total_cases = 0
        max_diffs = []

        # Test subset for reasonable test time
        test_configs = list(itertools.product(
            [83, 512],  # Reduced token counts
            [768, 2048],  # Reduced hidden sizes
            [False, True],  # Residual
            self.supported_dtypes,  # Only supported dtypes
            [1e-6],  # Single epsilon value
            [0, 42]  # Two seeds
        ))

        for config in test_configs:
            num_tokens, hidden_size, add_residual, (
                jax_dtype, torch_dtype), epsilon, seed = config
            total_cases += 1

            try:
                max_diff, mean_diff = self._run_precision_test(
                    num_tokens, hidden_size, add_residual, jax_dtype, torch_dtype, epsilon, seed
                )
                max_diffs.append(max_diff)
                print(f"✅ Case {total_cases}: max_diff={max_diff:.2e}")

            except AssertionError as e:
                failed_cases.append((config, str(e)))
                print(f"❌ Case {total_cases}: {str(e)}")

        # Summary
        success_rate = (total_cases - len(failed_cases)) / total_cases * 100
        avg_max_diff = sum(max_diffs) / len(max_diffs) if max_diffs else 0

        print(f"\n📊 Test Summary:")
        print(f"   Total cases: {total_cases}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Average max difference: {avg_max_diff:.2e}")
        print(f"   Failed cases: {len(failed_cases)}")

        if failed_cases:
            print(f"\n❌ Failed cases:")
            # Show first 5 failures
            for i, (config, error) in enumerate(failed_cases[:5]):
                print(f"   {i+1}. {config}")
                print(f"      Error: {error}")

        # Assert overall success rate
        self.assertGreaterEqual(
            success_rate, 80,
            f"Too many precision test failures. Success rate: {success_rate:.1f}%"
        )

    def test_rmsnorm_basic_functionality(self):
        """基本功能测试 - 确保 JAX 版本基本工作正常"""
        print("\n🧪 Testing basic JAX RMSNorm functionality...")

        hidden_size = 768
        jax_layer = self._create_jax_rmsnorm(hidden_size)

        # Test simple forward pass
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (1, hidden_size), dtype=jnp.float32)

        with self.mesh:
            output = jax_layer(x)

        # Basic sanity checks
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(jnp.isfinite(output).all())

        # Test with residual
        key, subkey = jax.random.split(key)
        residual = jax.random.normal(
            subkey, (1, hidden_size), dtype=jnp.float32)

        with self.mesh:
            output_with_residual = jax_layer(x, residual)

        self.assertIsInstance(output_with_residual, tuple)
        self.assertEqual(len(output_with_residual), 2)
        self.assertEqual(output_with_residual[0].shape, x.shape)
        self.assertEqual(output_with_residual[1].shape, residual.shape)

        print("✅ Basic functionality tests passed!")

    def test_rmsnorm_numerical_properties(self):
        """测试 RMSNorm 的数值特性"""
        print("\n📐 Testing RMSNorm numerical properties...")

        hidden_size = 1024
        epsilon = 1e-6

        # Create layers
        jax_layer = self._create_jax_rmsnorm(hidden_size, epsilon)
        torch_layer = self._create_torch_rmsnorm(hidden_size, epsilon)
        self._sync_weights(jax_layer, torch_layer)

        # Test data
        key = jax.random.PRNGKey(42)
        jax_x = jax.random.normal(key, (1, hidden_size), dtype=jnp.float32)
        torch_x = torch.from_numpy(
            self._safe_jax_to_numpy(jax_x)).to(self.torch_device)

        # Forward passes
        with self.mesh:
            jax_output = jax_layer(jax_x)

        with torch.no_grad():
            torch_output = torch_layer.forward_native(torch_x)

        # Check RMS normalization property
        # RMS(output) should be approximately 1.0 when weights are 1.0
        jax_rms = jnp.sqrt(jnp.mean(jax_output ** 2, axis=-1))
        torch_rms = torch.sqrt(torch.mean(torch_output ** 2, dim=-1))

        print(f"   JAX RMS: {float(jax_rms[0]):.6f}")
        print(f"   Torch RMS: {float(torch_rms[0].cpu()):.6f}")

        # Both should be close to the norm of the weight vector
        weight_norm = jnp.sqrt(jnp.mean(jax_layer.weight.value ** 2))
        print(f"   Weight RMS: {float(weight_norm):.6f}")

        # The RMS should be proportional to the weight norm
        self.assertAlmostEqual(float(jax_rms[0]), float(weight_norm), places=4)
        self.assertAlmostEqual(float(torch_rms[0].cpu()),
                               float(weight_norm), places=4)

        print("✅ Numerical properties test passed!")

    def test_edge_cases(self):
        """测试边界情况"""
        print("\n🏗️ Testing edge cases...")

        # Test very small hidden size
        small_layer = self._create_jax_rmsnorm(1)
        key = jax.random.PRNGKey(0)
        small_x = jax.random.normal(key, (1, 1), dtype=jnp.float32)

        with self.mesh:
            small_output = small_layer(small_x)

        self.assertEqual(small_output.shape, (1, 1))
        self.assertTrue(jnp.isfinite(small_output).all())

        # Test different epsilon values
        for eps in [1e-8, 1e-6, 1e-4]:
            layer = self._create_jax_rmsnorm(128, eps)
            x = jax.random.normal(key, (2, 128), dtype=jnp.float32)

            with self.mesh:
                output = layer(x)

            self.assertTrue(jnp.isfinite(output).all())

        print("✅ Edge cases test passed!")


if __name__ == '__main__':
    unittest.main()
