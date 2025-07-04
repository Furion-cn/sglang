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
from typing import List
from unittest.mock import patch

from flax import nnx
from jax import numpy as jnp
from transformers import AutoTokenizer

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.jax.layers.sampler import Sampler
from sglang.srt.jax.mem_cache.hash_kvcache import ReqToHashKVCachePool
from sglang.srt.jax.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.jax.models.qwen import QWenLMHeadJaxModel
from sglang.srt.jax.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.model_loader.loader import JAXModelLoader
from sglang.test.jax.test_utils import create_device_mesh, jax_trace_context
from sglang.test.test_utils import CustomTestCase


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


class TestQWenLoadWeights(CustomTestCase):
    """Test cases for QWenLMHeadJaxModel using JAXModelLoader"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_model_path = os.environ.get(
            'MODEL_PATH', '/tmp/test_qwen_jax_model')
        self.enable_debug_tracer = os.environ.get(
            'ENABLE_DEBUG_TRACER', False)
        self.mesh = create_device_mesh(
            ici_parallelism=[-1, 1, 1, 1],
            dcn_parallelism=[1, 1, 1, 1]
        )
        self.load_config = LoadConfig(load_format=LoadFormat.JAX)
        self.device_config = DeviceConfig()
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
        # new kv cache
        kv_cache = ReqToHashKVCachePool(
            head_num=model_config.num_attention_heads,
            head_dim=model_config.hidden_size // model_config.num_attention_heads,
            layer_num=model_config.num_hidden_layers,
            dtype=jnp.bfloat16 if model_config.bf16 else jnp.float32,
            max_seq_len=1024,
            max_batch_size=20
        )
        # batch size
        batch_size = len(actual_seq_lens)
        # cache loc
        cache_loc = jnp.arange(jnp.sum(seq_lens), dtype=jnp.int32)
        # Create ForwardBatch
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=batch_size,
            input_ids=input_ids_array,
            cache_loc=cache_loc, # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] if seq_lens = [3,4,3]
            out_cache_loc=None,
            seq_lens=seq_lens,
            positions=positions_array,
            extend_start_loc=extend_start_loc,
            total_tokens=len(input_ids_array),
            sequences=texts.copy(),
            prefix_str=texts.copy(),
            token_to_kv_pool=kv_cache,
        )

        return input_ids_array, actual_seq_lens, forward_batch

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
                # print(f"   nnx_state: {nnx.state(model)}")

                print("\n🎉 JAXModelLoader integration test completed successfully!")

                print("\n🔄 Test model input and output with JAXModelLoader...")

                print("\n🟢 Starting debug tracer session...")
                if self.enable_debug_tracer:
                    global_tracer.start_session()

                sampler = Sampler(rngs=nnx.Rngs(0))
                tokenizer = self._get_tokenizer()

                # Multiple questions to simulate batch > 1 scenario
                input_texts = [
                    "the capital of France is",
                    "what is the largest planet in",
                    "the founder of Apple company was",
                    "the capital of China is"
                ]

                input_ids_array, actual_seq_lens, forward_batch = self._create_batch_from_texts(
                    model.config, input_texts, tokenizer)

                print(f"Input text batch: {input_texts}")
                print(f"Batch size: {len(input_texts)}")
                print(f"Actual sequence lengths: {actual_seq_lens}")
                print(f"Input tokens shape: {input_ids_array.shape}")
                print(f"Input tokens: {input_ids_array}")

                jax_profiling_dir = os.environ.get(
                    "JAX_TRACE_PROFILING_DIR", "/tmp/jax_profiling")
                with self.mesh, jax_trace_context(jax_profiling_dir):
                    for _ in range(10):
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

                        self.update_forward_batch(
                            forward_batch, next_token_ids, tokenizer)

                # Decode complete results for each sequence
                print(f"\n=== Complete Generation Results ===")
                start_idx = 0
                input_ids = forward_batch.input_ids.tolist()
                for batch_idx in range(len(input_texts)):
                    # Extract tokens for each sequence from flattened array
                    seq_len = actual_seq_lens[batch_idx]
                    end_idx = start_idx + seq_len
                    print(
                        f"Decoded text {batch_idx}: '{forward_batch.sequences[batch_idx]}'")
                    print(
                        f"Original question {batch_idx}: '{input_texts[batch_idx]}'\n")
                    start_idx = end_idx

                if self.enable_debug_tracer:
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

    def _batch_tokenize(self, input_text: List[str]) -> List[Sequence]:
        return [Sequence(self.tokenizer, text) for text in input_text]

    def update_forward_batch(self, forward_batch: ForwardBatch, next_token_ids, tokenizer):
        # update out cache loc
        out_cache_start_loc = jnp.max(forward_batch.cache_loc) + 1
        forward_batch.out_cache_loc = jnp.arange(
            out_cache_start_loc, out_cache_start_loc + forward_batch.batch_size, dtype=jnp.int32)
            
        cache_start_loc = 0
        new_input_ids = []
        new_seq_lens = []
        new_cache_loc_list = []
        for batch_idx, seq_len in enumerate(forward_batch.seq_lens):
            new_seq_len = seq_len + 1
            current_token_id = int(next_token_ids[batch_idx, 0])
            new_input_ids.append(current_token_id)
            new_seq_lens.append(new_seq_len)
            decoded_token = tokenizer.decode(
                [current_token_id])
            
            # update cache loc
            old_cache_loc = forward_batch.cache_loc[
                cache_start_loc:cache_start_loc + seq_len]
            new_cache_loc_list.append(jnp.concatenate(
                [old_cache_loc, forward_batch.out_cache_loc[batch_idx:batch_idx+1]], axis=0))
            cache_start_loc += seq_len
            
            if forward_batch.forward_mode == ForwardMode.DECODE:
                # update prefix
                forward_batch.prefix_str[batch_idx] = forward_batch.sequences[batch_idx]

            # update sequences
            forward_batch.sequences[batch_idx] = forward_batch.prefix_str[batch_idx] + decoded_token
            print(
                f"Batch {batch_idx}: token_id={current_token_id}, decoded={decoded_token}")
        
        # update cache loc
        forward_batch.cache_loc = jnp.concatenate(new_cache_loc_list, axis=0)
        # update seq lens
        forward_batch.seq_lens = jnp.array(new_seq_lens, dtype=jnp.int32)
        # update extend start loc
        forward_batch.extend_start_loc = jnp.cumsum(
            jnp.concatenate([jnp.array([0]), forward_batch.seq_lens[:-1]]))
        # update positions
        forward_batch.positions = jnp.array(
            [seq_len - 1 for seq_len in new_seq_lens], dtype=jnp.int32)
        # update input ids
        forward_batch.input_ids = jnp.array(new_input_ids, dtype=jnp.int32)
        # update total tokens
        forward_batch.total_tokens = len(new_input_ids)
        # update forward mode
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            forward_batch.forward_mode = ForwardMode.DECODE


if __name__ == '__main__':
    unittest.main()
