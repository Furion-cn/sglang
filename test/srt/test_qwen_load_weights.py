#!/usr/bin/env python3
"""
QWenLMHeadJaxModel JAXModelLoader Integration Tests

Usage:
    python -m unittest test_qwen_load_weights.TestQWenLoadWeights
    
    # Test with specific model path:
    MODEL_PATH=/path/to/jax/qwen/model python -m unittest test_qwen_load_weights.TestQWenLoadWeights.test_load_model_with_jax_loader
"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from flax import nnx
from jax import numpy as jnp
from transformers import AutoTokenizer

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.jax.models.qwen import QWenLMHeadJaxModel
from sglang.srt.model_loader.loader import JAXModelLoader
from sglang.test.test_utils import CustomTestCase
from sglang.test.jax.test_utils import create_device_mesh
from sglang.srt.jax.layers.sampler import Sampler
from sglang.srt.jax.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.jax.models.qwen import QWenLMHeadJaxModel
from sglang.srt.jax.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.model_loader.loader import JAXModelLoader
from sglang.test.jax.test_utils import create_device_mesh
from sglang.test.test_utils import CustomTestCase


class TestQWenLoadWeights(CustomTestCase):
    """Test cases for QWenLMHeadJaxModel using JAXModelLoader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_model_path = os.environ.get(
            'MODEL_PATH', '/tmp/test_qwen_jax_model')
        self.mesh = create_device_mesh(
            ici_parallelism=[-1, 1, 1, 1],
            dcn_parallelism=[1, 1, 1, 1]
        )
        self.load_config = LoadConfig(load_format=LoadFormat.JAX)
        self.device_config = DeviceConfig()
        self.jax_loader = JAXModelLoader(self.load_config)

    def _get_positions(self, x):
        return jnp.concatenate([
            jnp.arange(x.shape[1]) for _ in range(x.shape[0])
        ]).reshape(x.shape[0], x.shape[1])

    def _create_batch(self, input_ids):
        """Convert input_ids [batch_size, seq_len] to ForwardBatch format"""
        batch_size, max_seq_len = input_ids.shape

        # For this example, assume all sequences have the same length
        seq_lens = jnp.full((batch_size,), max_seq_len, dtype=jnp.int32)

        # Flatten input_ids
        input_ids_flat = input_ids.reshape(-1)

        # Create positions for each token
        positions_flat = jnp.concatenate([
            jnp.arange(seq_len) for seq_len in seq_lens
        ])

        # Create start locations for each sequence
        extend_start_loc = jnp.cumsum(
            jnp.concatenate([jnp.array([0]), seq_lens[:-1]]))

        return ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=batch_size,
            input_ids=input_ids_flat,
            seq_lens=seq_lens,
            positions=positions_flat,
            extend_start_loc=extend_start_loc,
            total_tokens=len(input_ids_flat)
        )

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
        loader = JAXModelLoader(self.load_config)
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

        from sglang.debug_tracer import global_tracer

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
                mock_arch.return_value = (QWenLMHeadJaxModel, None)
                
                print("\n🔄 Loading model with JAXModelLoader...")
                model = self.jax_loader.load_model(
                    model_config=model_config,
                    device_config=self.device_config,
                    mesh=self.mesh,
                )

                print("✅ Model loaded successfully!")
                
                self.assertIsInstance(model, QWenLMHeadJaxModel)
                self.assertIsNotNone(model.config)

                print(f"\n📋 Model config:")
                print(f"   vocab_size: {model.config.vocab_size}")
                print(f"   hidden_size: {model.config.hidden_size}")
                print(
                    f"   num_hidden_layers: {model.config.num_hidden_layers}")
                print(f"   nnx_state: {nnx.state(model)}")

                print("\n🎉 JAXModelLoader integration test completed successfully!")

                print("\n🔄 Test model input and output with JAXModelLoader...")
                
                print("\n🟢 Starting debug tracer session...")
                global_tracer.start_session()
                
                sampler = Sampler(rngs=nnx.Rngs(0))
                tokenizer = self._get_tokenizer()

                input_text = "1+1=?"
                x = jnp.array(tokenizer.encode(input_text)).reshape(1, -1)
                print(f"输入文本: {input_text}")
                print(f"输入 tokens: {x}")

                with self.mesh:
                    for i in range(1):
                        # Create ForwardBatch for each iteration
                        forward_batch = self._create_batch(x)
                        y = model(forward_batch.input_ids,
                                  forward_batch.positions, forward_batch)

                        # The LogitsProcessor now automatically extracts the last token logits
                        # y.next_token_logits shape: [batch_size, vocab_size]

                        # Sample next token
                        next_token_ids = sampler(
                            y,  # Pass the LogitsProcessorOutput directly
                            sampling_info=SamplingBatchInfo(
                                temperatures=jnp.full((1, 1), 0.1),
                                top_ps=jnp.full((1, 1), 0.8),
                                top_ks=jnp.full((1, 1), 50),
                                min_ps=jnp.full((1, 1), 0.01),
                                vocab_size=model.config.vocab_size,
                            ))

                        # Update sequence with new token for next iteration
                        x = jnp.concatenate([x, next_token_ids], axis=-1)

                        # 解码当前生成的 token
                        current_token_id = int(next_token_ids[0, 0])
                        decoded_token = tokenizer.decode([current_token_id])
                        print(
                            f"Step {i+1}: token_id={current_token_id}, decoded='{decoded_token}'")

                full_sequence = [int(token) for token in x[0]]
                decoded_full = tokenizer.decode(full_sequence)
                print(f"\n完整生成序列: {full_sequence}")
                print(f"完整解码文本: '{decoded_full}'")

                print("\n🔴 Ending debug tracer session...")
                debug_file = global_tracer.end_session()
                if debug_file:
                    print(f"✅ Debug trace saved to: {debug_file}")
                else:
                    print("⚠️  Debug trace not saved")

        except Exception as e:
            if 'global_tracer' in locals():
                try:
                    global_tracer.end_session()
                    print("🔴 Debug tracer session ended due to exception")
                except:
                    pass
            self.fail(f"JAXModelLoader integration test failed: {e}")

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
