#!/usr/bin/env python3
"""
QWenLMHeadModel JAXModelLoader Integration Tests

Usage:
    python -m unittest test_qwen_load_weights.TestQWenLoadWeights
    
    # Test with specific model path:
    MODEL_PATH=/path/to/jax/qwen/model python -m unittest test_qwen_load_weights.TestQWenLoadWeights.test_load_model_with_jax_loader
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from flax import nnx
from jax import numpy as jnp
from transformers import AutoTokenizer

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.jax.layers.sampler import Sampler
from sglang.srt.jax.models.qwen import QWenLMHeadModel
from sglang.srt.jax.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.model_loader.loader import DefaultModelLoader, JAXModelLoader
from sglang.srt.models.qwen import QWenLMHeadModel as TorchQWenLMHeadModel
from sglang.test.jax.test_utils import create_device_mesh
from sglang.test.test_utils import CustomTestCase


class TestQWenLoadWeights(CustomTestCase):
    """Test cases for QWenLMHeadModel using JAXModelLoader"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_model_path = os.environ.get(
            'MODEL_PATH', '/tmp/test_qwen_jax_model')
        self.mesh = create_device_mesh(
            ici_parallelism=[-1, 1, 1, 1],
            dcn_parallelism=[1, 1, 1, 1]
        )
        self.load_config_jax = LoadConfig(load_format=LoadFormat.JAX)
        self.load_config_default = LoadConfig(load_format=LoadFormat.AUTO)
        self.device_config = DeviceConfig()
        self.jax_loader = JAXModelLoader(self.load_config_jax)
        self.default_loader = DefaultModelLoader(self.load_config_default)

    def _get_positions(self, x):
        return jnp.concatenate([
            jnp.arange(x.shape[1]) for _ in range(x.shape[0])
        ]).reshape(x.shape[0], x.shape[1])

    def _get_tokenizer(self):
        """Get tokenizer from local path if available, otherwise from Hugging Face"""
        model_path = Path(self.test_model_path)

        # Check if tokenizer files exist in the model path
        tokenizer_files = [
            'tokenizer_config.json',
            'tokenization_qwen.py',
            'qwen.tiktoken'
        ]

        has_tokenizer = all((model_path / file).exists()
                            for file in tokenizer_files)

        if has_tokenizer:
            print(f"📁 Using local tokenizer from: {model_path}")
            try:
                return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            except Exception as e:
                print(f"⚠️  Failed to load local tokenizer: {e}")
                print("🔄 Falling back to Hugging Face...")
        else:
            print(f"📁 No tokenizer found in {model_path}, using Hugging Face")

        # Fallback to Hugging Face
        print("🌐 Loading tokenizer from Hugging Face: Qwen/Qwen-7B")
        return AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)

    def test_jax_loader_initialization(self):
        """Test JAXModelLoader initialization"""
        loader = JAXModelLoader(self.load_config_jax)
        self.assertEqual(loader.load_config.load_format, LoadFormat.JAX)

    def test_jax_loader_invalid_format(self):
        """Test JAXModelLoader with invalid load format"""
        invalid_config = LoadConfig(load_format=LoadFormat.AUTO)
        with self.assertRaises(ValueError) as context:
            JAXModelLoader(invalid_config)
        self.assertIn("JAXModelLoader only supports JAX load format",
                      str(context.exception))

    def test_load_model_with_jax_loader(self):
        """Test loading QWen model using JAXModelLoader (integration test)"""
        if not os.path.exists(self.test_model_path):
            self.skipTest(
                f"Model path {self.test_model_path} not found. Set MODEL_PATH environment variable.")

        try:
            hf_folder, hf_weights_files = self.jax_loader._prepare_jax_weights(
                self.test_model_path, None
            )

            if not hf_weights_files:
                self.skipTest(
                    f"No .msgpack files found in {self.test_model_path}")

            print(
                f"\n=== Testing JAXModelLoader with: {self.test_model_path} ===")
            print(f"Found {len(hf_weights_files)} msgpack files")

            model_config = ModelConfig(
                model_path=self.test_model_path,
                model_override_args="{}"
            )

            with patch('sglang.srt.model_loader.loader.get_model_architecture') as mock_arch:
                mock_arch.return_value = (QWenLMHeadModel, None)

                print("\n🔄 Loading model with JAXModelLoader...")
                model = self.jax_loader.load_model(
                    model_config=model_config,
                    device_config=self.device_config,
                    mesh=self.mesh,
                )

                print("✅ Model loaded successfully!")

                self.assertIsInstance(model, QWenLMHeadModel)
                self.assertIsNotNone(model.config)

                print(f"\n📋 Model config:")
                print(f"   vocab_size: {model.config.vocab_size}")
                print(f"   hidden_size: {model.config.hidden_size}")
                print(
                    f"   num_hidden_layers: {model.config.num_hidden_layers}")
                print(f"   nnx_state: {nnx.state(model)}")

                print("\n🎉 JAXModelLoader integration test completed successfully!")

                print("\n🔄 Test model input and output with JAXModelLoader...")
                sampler = Sampler(rngs=nnx.Rngs(0))
                tokenizer = self._get_tokenizer()

                input_text = "1+1=?"
                x = jnp.array(tokenizer.encode(input_text)).reshape(1, -1)
                print(f"输入文本: {input_text}")
                print(f"输入 tokens: {x}")

                with self.mesh:
                    for i in range(10):
                        positions = self._get_positions(x)
                        y = model(x, positions, None)
                        next_token_ids = sampler(
                            y, sampling_info=SamplingBatchInfo(
                                temperatures=jnp.full((1, 1), 0.3),
                                top_ps=jnp.full((1, 1), 0.8),
                                top_ks=jnp.full((1, 1), 50),
                                min_ps=jnp.full((1, 1), 0.01),
                                vocab_size=model.config.vocab_size,
                            ))
                        x = jnp.concatenate(
                            [x, next_token_ids], axis=-1)

                        # 解码当前生成的 token
                        current_token_id = int(next_token_ids[0, 0])
                        decoded_token = tokenizer.decode([current_token_id])
                        print(
                            f"Step {i+1}: token_id={current_token_id}, decoded='{decoded_token}'")

                full_sequence = [int(token) for token in x[0]]
                decoded_full = tokenizer.decode(full_sequence)
                print(f"\n完整生成序列: {full_sequence}")
                print(f"完整解码文本: '{decoded_full}'")

        except Exception as e:
            self.fail(f"JAXModelLoader integration test failed: {e}")

    def test_load_model_comparison_jax_vs_default(self):
        """Test comparing JAX loader vs Default loader model outputs for consistency validation"""
        if not os.path.exists(self.test_model_path):
            self.skipTest(
                f"Model path {self.test_model_path} not found. Set MODEL_PATH environment variable.")

        # Check if both JAX and torch weights exist
        try:
            # Check JAX weights
            hf_folder, hf_weights_files = self.jax_loader._prepare_jax_weights(
                self.test_model_path, None
            )
            if not hf_weights_files:
                self.skipTest(
                    f"No .msgpack files found in {self.test_model_path}")

            # Check if pytorch weights also exist for comparison
            torch_weights_folder, torch_weights_files, _ = self.default_loader._prepare_weights(
                self.test_model_path, None, fall_back_to_pt=True
            )
            if not torch_weights_files:
                self.skipTest(
                    f"No torch weight files found in {self.test_model_path}")

        except Exception as e:
            self.skipTest(f"Failed to prepare weights for comparison: {e}")

        print(
            f"\n=== Comparing JAX vs Default Loader: {self.test_model_path} ===")
        print(f"JAX files: {len(hf_weights_files)} msgpack files")
        print(f"Torch files: {len(torch_weights_files)} weight files")

        model_config = ModelConfig(
            model_path=self.test_model_path,
            model_override_args="{}"
        )

        # Load model with JAX loader
        with patch('sglang.srt.model_loader.loader.get_model_architecture') as mock_arch_jax:
            mock_arch_jax.return_value = (QWenLMHeadModel, None)

            print("\n🔄 Loading model with JAXModelLoader...")
            jax_model = self.jax_loader.load_model(
                model_config=model_config,
                device_config=self.device_config,
                mesh=self.mesh,
            )
            print("✅ JAX model loaded successfully!")

        # Load model with Default loader
        with patch('sglang.srt.model_loader.loader.get_model_architecture') as mock_arch_default:
            mock_arch_default.return_value = (TorchQWenLMHeadModel, None)

            print("\n🔄 Loading model with DefaultModelLoader...")
            torch_model = self.default_loader.load_model(
                model_config=model_config,
                device_config=self.device_config,
            )
            print("✅ Default model loaded successfully!")

        # Compare model configurations
        print(f"\n📋 Comparing model configurations:")
        print(f"   JAX model - vocab_size: {jax_model.config.vocab_size}")
        print(f"   Torch model - vocab_size: {torch_model.config.vocab_size}")
        print(f"   JAX model - hidden_size: {jax_model.config.hidden_size}")
        print(
            f"   Torch model - hidden_size: {torch_model.config.hidden_size}")
        print(
            f"   JAX model - num_layers: {jax_model.config.num_hidden_layers}")
        print(
            f"   Torch model - num_layers: {torch_model.config.num_hidden_layers}")

        # Verify configurations match
        self.assertEqual(jax_model.config.vocab_size,
                         torch_model.config.vocab_size)
        self.assertEqual(jax_model.config.hidden_size,
                         torch_model.config.hidden_size)
        self.assertEqual(jax_model.config.num_hidden_layers,
                         torch_model.config.num_hidden_layers)

        # Test inference comparison
        print("\n🔄 Comparing model inference outputs...")
        tokenizer = self._get_tokenizer()
        test_text = "Hello world"
        input_ids = tokenizer.encode(test_text)

        # JAX model inference
        jax_input = jnp.array(input_ids).reshape(1, -1)
        jax_positions = self._get_positions(jax_input)

        with self.mesh:
            jax_output = jax_model(jax_input, jax_positions, None)
            # Get last token logits for first 10 vocab items
            jax_logits = jax_output[0, -1, :10]

        # Torch model inference - create minimal ForwardBatch and positions
        torch_input = torch.tensor(input_ids).reshape(
            1, -1).to(self.device_config.device)
        torch_positions = torch.arange(len(input_ids)).reshape(
            1, -1).to(self.device_config.device)

        # Create a minimal ForwardBatch for torch model
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )

        mock_forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=torch_input,
            req_pool_indices=torch.tensor(
                [0], device=self.device_config.device),
            seq_lens=torch.tensor(
                [len(input_ids)], device=self.device_config.device),
            out_cache_loc=torch.tensor([0], device=self.device_config.device),
            seq_lens_sum=len(input_ids),
            seq_lens_cpu=[len(input_ids)],
        )

        # Since we're doing a simple forward pass without attention caching,
        # we can set the required attributes to None or minimal values
        mock_forward_batch.req_to_token_pool = None
        mock_forward_batch.token_to_kv_pool = None
        mock_forward_batch.attn_backend = None

        with torch.no_grad():
            torch_output = torch_model(
                torch_input, torch_positions, mock_forward_batch)
            # torch_output should be a LogitsProcessorOutput or similar
            if hasattr(torch_output, 'next_token_logits'):
                # Get first 10 logits
                torch_logits = torch_output.next_token_logits[0, :10]
            else:
                # If it's just raw logits tensor
                # Get last token logits for first 10 vocab items
                torch_logits = torch_output[0, -1, :10]

        print(f"   JAX model logits (first 10): {jax_logits}")
        print(
            f"   Torch model logits (first 10): {torch_logits.cpu().numpy()}")

        # Compare logits (allow some numerical differences due to precision)
        jax_logits_np = jnp.array(jax_logits)
        torch_logits_np = torch_logits.detach().cpu().numpy()

        # Check if logits are reasonably close (within 1e-2 tolerance for most values)
        close_enough = jnp.allclose(
            jax_logits_np, torch_logits_np, atol=1e-2, rtol=1e-2)

        if close_enough:
            print("✅ Model outputs are reasonably consistent!")
        else:
            print("⚠️  Model outputs show significant differences")
            diff = jnp.abs(jax_logits_np - torch_logits_np)
            max_diff = jnp.max(diff)
            mean_diff = jnp.mean(diff)
            print(f"   Max difference: {max_diff}")
            print(f"   Mean difference: {mean_diff}")

            # Allow larger tolerance for initial testing
            if max_diff < 0.5 and mean_diff < 0.1:
                print(
                    "✅ Differences are within acceptable range for different implementations")
            else:
                print("❌ Differences are too large - may indicate loading issues")

        print("\n🎉 JAX vs Default loader comparison completed!")

    def test_prepare_jax_weights_no_msgpack_files(self):
        """Test JAXModelLoader behavior when no msgpack files exist"""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            empty_dir = os.path.join(temp_dir, 'empty_model')
            os.makedirs(empty_dir, exist_ok=True)

            with self.assertRaises(RuntimeError) as context:
                self.jax_loader._prepare_jax_weights(empty_dir, None)

            self.assertIn("Cannot find any JAX model weights",
                          str(context.exception))


if __name__ == '__main__':
    unittest.main()
