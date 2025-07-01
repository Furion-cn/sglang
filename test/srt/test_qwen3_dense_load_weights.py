#!/usr/bin/env python3
"""
Qwen3ForCausalLMJaxModel JAXModelLoader Integration Tests

Usage:
    python -m unittest test_qwen3_dense_load_weights.TestQwen3DenseLoadWeights
    
    # Test with specific model path:
    MODEL_PATH=/path/to/jax/qwen3/model python -m unittest test_qwen3_dense_load_weights.TestQwen3DenseLoadWeights.test_load_model_with_jax_loader
"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from flax import nnx
from jax import numpy as jnp
from transformers import AutoTokenizer
from typing import List

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.jax.layers.sampler import Sampler
from sglang.srt.jax.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.jax.models.qwen3 import Qwen3ForCausalLMJaxModel
from sglang.srt.jax.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.model_loader.loader import JAXModelLoader
from sglang.test.jax.test_utils import create_device_mesh
from sglang.test.test_utils import CustomTestCase
from sglang.srt.jax.mem_cache.hash_kvcache import ReqToHashKVCachePool, HashKVCache

class Sequence:
    def __init__(self, tokenizer, input_text: str):
        self.input_text = input_text

        self.input_ids = tokenizer.encode(input_text)
        self.seq_len = len(self.input_ids)

    def extend(self, next_token_ids: int):
        self.input_ids.append(next_token_ids)
        self.seq_len += 1

def sequence_extend(sequences: List[Sequence], next_token_ids: List[int]):
    for i, seq in enumerate(sequences):
        seq.extend(next_token_ids[i][0])
    return sequences

class TestQwen3DenseLoadWeights(CustomTestCase):
    """Test cases for Qwen3ForCausalLMJaxModel using JAXModelLoader"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_model_path = os.environ.get(
            'MODEL_PATH', '/tmp/test_qwen_jax_model')
        self.mesh = create_device_mesh(
            ici_parallelism=[-1, 1, 1, 1],
            dcn_parallelism=[1, 1, 1, 1]
        )
        self.load_config = LoadConfig(load_format=LoadFormat.JAX)
        self.device_config = DeviceConfig("cpu")
        self.jax_loader = JAXModelLoader(self.load_config)
        self.tokenizer = self._get_tokenizer()

    def _get_positions(self, x):
        return jnp.concatenate([
            jnp.arange(x.shape[1]) for _ in range(x.shape[0])
        ]).reshape(x.shape[0], x.shape[1])

    def _create_batch_from_texts(self, model_config, texts, tokenizer):
        """Create initial batch from texts with tokenization (no padding needed)

        Args:
            texts: List[str] input texts to process
            tokenizer: tokenizer to use for encoding

        Returns:
            tuple: (input_ids_array, actual_seq_lens, forward_batch)
        """
        # Tokenize each question
        tokenized_inputs = []
        actual_seq_lens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            tokenized_inputs.append(tokens)
            actual_seq_lens.append(len(tokens))

        # Concatenate all tokens directly without padding
        input_ids_flat = []
        positions_flat = []
        for tokens in tokenized_inputs:
            input_ids_flat.extend(tokens)
            # Create positions at the same time
            positions_flat.extend(range(len(tokens)))

        # Create required arrays
        input_ids_array = jnp.array(input_ids_flat, dtype=jnp.int32)
        positions_array = jnp.array(positions_flat, dtype=jnp.int32)
        seq_lens = jnp.array(actual_seq_lens, dtype=jnp.int32)

        # Create start locations
        extend_start_loc = jnp.cumsum(
            jnp.concatenate([jnp.array([0]), seq_lens[:-1]]))
        if model_config.torch_dtype == "bfloat16":
            dtype = jnp.bfloat16
        else:
            dtype = jnp.float32
        current_kv_cache = [ReqToHashKVCachePool(
            seq_len=seq_len,
            head_num=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            layer_num=model_config.num_hidden_layers,
            dtype=dtype
        ) for seq_len in seq_lens]
        # Create ForwardBatch
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=len(actual_seq_lens),
            input_ids=input_ids_array,
            seq_lens=seq_lens,
            positions=positions_array,
            extend_start_loc=extend_start_loc,
            total_tokens=len(input_ids_array),
            sequences=texts.copy(),
            current_kv_cache=current_kv_cache,
            prefix_str=texts.copy(),
            token_to_kv_pool=HashKVCache(),
        )

        return input_ids_array, actual_seq_lens, forward_batch

    def _get_tokenizer(self):
        """Get tokenizer from local path if available, otherwise from Hugging Face"""
        model_path = Path(self.test_model_path)

        # Check if tokenizer files exist in the model path
        tokenizer_files = [
            'tokenizer_config.json',
            'tokenizer.json',
            'merges.txt',
            'vocab.json',
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
        print("🌐 Loading tokenizer from Hugging Face: Qwen/Qwen3-8B")
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

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
        """Test loading Qwen3 model using JAXModelLoader (integration test)"""
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
                mock_arch.return_value = (Qwen3ForCausalLMJaxModel, None)

                print("\n🔄 Loading model with JAXModelLoader...")
                model = self.jax_loader.load_model(
                    model_config=model_config,
                    device_config=self.device_config,
                    mesh=self.mesh,
                )

                print("✅ Model loaded successfully!")

                self.assertIsInstance(model, Qwen3ForCausalLMJaxModel)
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
                #global_tracer.start_session()
                
                sampler = Sampler(rngs=nnx.Rngs(0))
                tokenizer = self._get_tokenizer()

                # Multiple questions to simulate batch > 1 scenario
                input_texts = [
                    "the capital of France is",
                    # "what is the largest planet in",
                    # "the founder of Apple company was",
                    # "the capital of China is",
                ]

                input_ids_array, actual_seq_lens, forward_batch = self._create_batch_from_texts(
                    model.config, input_texts, tokenizer)

                print(f"Input text batch: {input_texts}")
                print(f"Batch size: {len(input_texts)}")
                print(f"Actual sequence lengths: {actual_seq_lens}")
                print(f"Input tokens shape: {input_ids_array.shape}")
                print(f"Input tokens: {input_ids_array}")

                with self.mesh:
                    for i in range(5):
                        # Use existing forward_batch, no need to recreate
                        y = model(forward_batch.input_ids,
                                  forward_batch.positions, forward_batch)

                        # The LogitsProcessor now automatically extracts the last token logits
                        # y.next_token_logits shape: [batch_size, vocab_size]

                        # Sample next token for each sequence in the batch
                        next_token_ids = sampler(
                            y,  # Pass the LogitsProcessorOutput directly
                            sampling_info=SamplingBatchInfo(
                                temperatures=jnp.full(
                                    (len(input_texts), 1), 1.0),
                                top_ps=jnp.full((len(input_texts), 1), 1.0),
                                top_ks=jnp.ones((len(input_texts), 1)),
                                min_ps=jnp.full((len(input_texts), 1), 0.0),
                                vocab_size=model.config.vocab_size,
                            ))
                        self.update_forward_batch(forward_batch, next_token_ids, tokenizer)

                # Decode complete results for each sequence
                print(f"\n=== Complete Generation Results ===")
                start_idx = 0
                input_ids = forward_batch.input_ids.tolist()
                for batch_idx in range(len(input_texts)):
                    # Extract tokens for each sequence from flattened array
                    seq_len = actual_seq_lens[batch_idx]
                    end_idx = start_idx + seq_len
                    print(f"Decoded text {batch_idx}: '{forward_batch.sequences[batch_idx]}'")
                    print(
                        f"Original question {batch_idx}: '{input_texts[batch_idx]}'")
                    print(f"Actual length {batch_idx}: {seq_len}")
                    start_idx = end_idx
                    print()

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

    def update_forward_batch(self, forward_batch: ForwardBatch, next_token_ids, tokenizer):
        new_input_ids = []
        new_seq_lens = []
        for batch_idx, seq_len in enumerate(forward_batch.seq_lens):
            current_token_id = int(next_token_ids[batch_idx, 0])
            new_input_ids.append(current_token_id)
            new_seq_lens.append(seq_len + 1)
            decoded_token = tokenizer.decode(
                [current_token_id])

            # update prefix
            if forward_batch.forward_mode == ForwardMode.DECODE:
                forward_batch.prefix_str[batch_idx] = forward_batch.sequences[batch_idx]

            # update kv cache
            forward_batch.token_to_kv_pool.set_kv_cache(
                forward_batch.prefix_str[batch_idx],
                forward_batch.current_kv_cache[batch_idx]
            )
            # update sequences
            forward_batch.sequences[batch_idx] = forward_batch.prefix_str[batch_idx] + decoded_token
            print(
                f"Batch {batch_idx}: token_id={current_token_id}, decoded='{decoded_token} prefix={forward_batch.prefix_str[batch_idx]}")

        # update seq lens
        forward_batch.seq_lens = jnp.array(new_seq_lens, dtype=jnp.int32)
        # update extend start loc
        extend_start_loc = jnp.cumsum(
            jnp.concatenate([jnp.array([0]), forward_batch.seq_lens[:-1]]))
        # update positions
        forward_batch.positions = jnp.array(
            [seq_len - 1 for seq_len in new_seq_lens], dtype=jnp.int32)
        # update input ids
        forward_batch.input_ids = jnp.array(new_input_ids, dtype=jnp.int32)
        # update extend start loc
        forward_batch.extend_start_loc = extend_start_loc
        # update total tokens
        forward_batch.total_tokens = len(new_input_ids)
        # update forward mode
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            forward_batch.forward_mode = ForwardMode.DECODE

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
            
    def _batch_tokenize(self, input_text: List[str]) -> List[Sequence]:
        return [Sequence(self.tokenizer, text) for text in input_text]

if __name__ == '__main__':
    unittest.main()
