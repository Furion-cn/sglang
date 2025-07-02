import os
import unittest
from pathlib import Path
from typing import List
from unittest.mock import patch

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.jax.layers.sampler import Sampler
from sglang.srt.jax.mem_cache.hash_kvcache import HashKVCache, ReqToHashKVCachePool
from sglang.srt.jax.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.jax.models.qwen import QWenLMHeadJaxModel
from sglang.srt.jax.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.model_loader.loader import JAXModelLoader
from sglang.test.jax.test_utils import create_device_mesh, jax_trace_context


class TestQwenModel(unittest.TestCase):
    """Test cases for the Qwen model."""

    def setUp(self):
        self.mesh = create_device_mesh(
            ici_parallelism=[-1, 1, 1, 1], dcn_parallelism=[1, 1, 1, 1])
        # Model path for local model and tokenizer
        self.test_model_path = os.environ.get(
            'MODEL_PATH', 'Qwen/Qwen-7B')  # Default to HuggingFace

        # JAX loader configuration
        self.load_config = LoadConfig(load_format=LoadFormat.JAX)
        self.device_config = DeviceConfig()
        self.jax_loader = JAXModelLoader(self.load_config)

    def _get_tokenizer(self):
        """Get tokenizer from local path if available, otherwise use HuggingFace"""
        model_path = Path(self.test_model_path)

        # Check if it's a local path and has tokenizer files
        if model_path.exists():
            tokenizer_files = ['tokenizer_config.json']
            has_tokenizer = any((model_path / file).exists()
                                for file in tokenizer_files)

            if has_tokenizer:
                print(f"📁 Using local tokenizer from: {model_path}")
                try:
                    return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
                except Exception as e:
                    print(f"⚠️  Failed to load local tokenizer: {e}")

        # Use HuggingFace model with network error handling
        try:
            print(
                f"🌐 Loading tokenizer from HuggingFace: {self.test_model_path}")
            return AutoTokenizer.from_pretrained(self.test_model_path, trust_remote_code=True)
        except Exception as e:
            print(f"⚠️  Failed to load tokenizer from HuggingFace: {e}")
            raise RuntimeError(
                f"Could not load tokenizer from local path or HuggingFace: {e}")

    def _setup_model(self):
        """Setup model using JAXModelLoader"""
        model_path = self.test_model_path

        # Check if it's a local path
        if os.path.exists(model_path):
            print(f"🔄 Loading model from local path: {model_path}")

            # Check for JAX model files (.msgpack)
            hf_folder, hf_weights_files = self.jax_loader._prepare_jax_weights(
                model_path, None
            )

            if not hf_weights_files:
                raise ValueError(f"No .msgpack files found in {model_path}")

            print(f"📦 Found {len(hf_weights_files)} msgpack files")
        else:
            print(f"🌐 Loading model from HuggingFace: {model_path}")

        # Create model config
        model_config = ModelConfig(
            model_path=model_path,
            model_override_args="{}"
        )

        # Load the model using JAXModelLoader
        with patch('sglang.srt.model_loader.loader.get_model_architecture') as mock_arch:
            mock_arch.return_value = (QWenLMHeadJaxModel, None)

            model = self.jax_loader.load_model(
                model_config=model_config,
                device_config=self.device_config,
                mesh=self.mesh,
            )

            print("✅ Model loaded successfully!")
            print(f"📋 Model config: vocab_size={model.config.vocab_size}, "
                  f"hidden_size={model.config.hidden_size}, "
                  f"num_layers={model.config.num_hidden_layers}")

            return model

    def _create_batch_from_texts(self, model_config, texts, tokenizer):
        """Create initial batch from texts with tokenization"""
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

        # Create real KV cache using existing JAX implementation
        current_kv_cache = []
        for seq_len in seq_lens:
            kv_cache = ReqToHashKVCachePool(
                seq_len=max(seq_len + 100, 128),  # Allow room for generation
                head_num=model_config.num_attention_heads,
                head_dim=model_config.hidden_size // model_config.num_attention_heads,
                layer_num=model_config.num_hidden_layers,
                dtype=jnp.float32
            )
            current_kv_cache.append(kv_cache)

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
            prefix_str=texts.copy(),
            token_to_kv_pool=HashKVCache(),
            current_kv_cache=current_kv_cache
        )

        return input_ids_array, actual_seq_lens, forward_batch

    def _is_finished(self, token_id, tokenizer):
        """Check if token is an end-of-sequence token using sglang's standard logic"""
        # Standard EOS token check
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            if token_id == tokenizer.eos_token_id:
                return True

        # Check additional stop token ids (this is where Qwen's special tokens would be)
        if hasattr(tokenizer, 'additional_stop_token_ids') and tokenizer.additional_stop_token_ids:
            if token_id in tokenizer.additional_stop_token_ids:
                return True

        # Fallback: Check for known Qwen stop tokens if additional_stop_token_ids is not set
        # This covers the case where the tokenizer doesn't configure additional_stop_token_ids properly
        qwen_stop_token_ids = [
            151643,  # <|endoftext|>
            151645,  # <|im_end|>
        ]
        if token_id in qwen_stop_token_ids:
            return True

        return False

    def _update_forward_batch_with_finished_handling(self, forward_batch, next_token_ids, tokenizer, finished_requests, original_indices):
        """Update forward batch while handling finished requests"""
        new_input_ids = []
        new_seq_lens = []
        batch_indices_to_keep = []
        new_original_indices = []

        for batch_idx, seq_len in enumerate(forward_batch.seq_lens):
            orig_idx = original_indices[batch_idx]

            if orig_idx in finished_requests:
                continue  # Skip finished requests

            current_token_id = int(next_token_ids[batch_idx, 0])
            decoded_token = tokenizer.decode(
                [current_token_id], skip_special_tokens=False)

            print(
                f"Request {orig_idx} (batch_idx {batch_idx}): token_id={current_token_id}, decoded='{decoded_token}'")

            # Check if this request should finish BEFORE updating sequences
            if self._is_finished(current_token_id, tokenizer):
                print(
                    f"🛑 Request {orig_idx} finished with EOS token: {current_token_id} ('{decoded_token}')")
                finished_requests.add(orig_idx)
                continue

            # Only update sequences for non-finished requests
            new_input_ids.append(current_token_id)
            new_seq_lens.append(seq_len + 1)
            batch_indices_to_keep.append(batch_idx)
            new_original_indices.append(orig_idx)

            # Update prefix
            if forward_batch.forward_mode == ForwardMode.DECODE:
                forward_batch.prefix_str[batch_idx] = forward_batch.sequences[batch_idx]

            # Update kv cache
            forward_batch.token_to_kv_pool.set_kv_cache(
                forward_batch.prefix_str[batch_idx],
                forward_batch.current_kv_cache[batch_idx]
            )

            # Update sequences and print ONLY for continuing requests
            forward_batch.sequences[batch_idx] = forward_batch.prefix_str[batch_idx] + decoded_token
            print(
                f"   → Updated sequence: '{forward_batch.sequences[batch_idx]}'")

        if not batch_indices_to_keep:
            # All requests are finished
            return None, None

        # Update batch with only unfinished requests
        forward_batch.batch_size = len(batch_indices_to_keep)
        forward_batch.seq_lens = jnp.array(new_seq_lens, dtype=jnp.int32)

        # Update positions for decode mode
        forward_batch.positions = jnp.array(
            [seq_len - 1 for seq_len in new_seq_lens], dtype=jnp.int32)

        # Update input ids
        forward_batch.input_ids = jnp.array(new_input_ids, dtype=jnp.int32)

        # Update extend start loc
        forward_batch.extend_start_loc = jnp.cumsum(
            jnp.concatenate([jnp.array([0]), forward_batch.seq_lens[:-1]]))

        # Update total tokens
        forward_batch.total_tokens = len(new_input_ids)

        # Update forward mode
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            forward_batch.forward_mode = ForwardMode.DECODE

        # Update other batch-related fields
        forward_batch.sequences = [forward_batch.sequences[i]
                                   for i in batch_indices_to_keep]
        forward_batch.prefix_str = [
            forward_batch.prefix_str[i] for i in batch_indices_to_keep]
        forward_batch.current_kv_cache = [
            forward_batch.current_kv_cache[i] for i in batch_indices_to_keep]

        return batch_indices_to_keep, new_original_indices

    def test_qwen_model_decode(self):
        """Test model with batch processing and EOS handling"""
        model = self._setup_model()
        jax_profiling_dir = os.environ.get(
            "JAX_TRACE_PROFILING_DIR", "/tmp/jax_profiling")
        with self.mesh, jax_trace_context(jax_profiling_dir):
            sampler = Sampler(rngs=nnx.Rngs(0))
            tokenizer = self._get_tokenizer()

            # Multiple input texts to test batch processing and EOS handling
            input_texts = [
                "The capital of France is",
                "What is 2+2?",
                "Hello, how are you",
                "道可道，非常道。请阐述这句话的哲学含义",
                "天地不仁，以万物为刍狗。请解释这句话的深层含义",
                "请解释道德经中无为而治的思想",
            ]

            print(f"\n🚀 Starting generation with {len(input_texts)} requests:")
            for i, text in enumerate(input_texts):
                print(f"   Request {i}: '{text}'")
                print(f"   Encoded: {tokenizer.encode(text)}")

            input_ids_array, actual_seq_lens, forward_batch = self._create_batch_from_texts(
                model.config, input_texts, tokenizer)

            print(f"\nBatch size: {len(input_texts)}")
            print(f"Actual sequence lengths: {actual_seq_lens}")
            print(f"Input tokens shape: {input_ids_array.shape}")
            print(f"Model vocab size: {model.config.vocab_size}")
            print(f"Tokenizer EOS token ID: {tokenizer.eos_token_id}")

            # Keep track of finished requests and their final results
            finished_requests = set()
            final_results = {}  # Store final results for all requests
            # Track original indices
            original_indices = list(range(len(input_texts)))

            # Store initial sequences for final results
            for i, text in enumerate(input_texts):
                final_results[i] = {
                    'input': text,
                    'output': text,  # Start with input text
                    'finished': False
                }

            max_iterations = 30  # Reduced for testing
            for iteration in range(max_iterations):
                if forward_batch is None:
                    print("\n🏁 All requests finished!")
                    break

                print(f"\n--- Iteration {iteration + 1} ---")
                print(f"Active requests: {forward_batch.batch_size}")
                print(f"Original indices: {original_indices}")

                # Forward pass
                y = model(forward_batch.input_ids,
                          forward_batch.positions, forward_batch)

                # Sample next token for each active sequence
                next_token_ids = sampler(
                    y,
                    sampling_info=SamplingBatchInfo(
                        temperatures=jnp.full(
                            (forward_batch.batch_size, 1), 1.0),
                        top_ps=jnp.full((forward_batch.batch_size, 1), 1.0),
                        top_ks=jnp.ones((forward_batch.batch_size, 1)),
                        min_ps=jnp.full((forward_batch.batch_size, 1), 0.0),
                        vocab_size=model.config.vocab_size,
                    ))

                print(f"Generated tokens: {next_token_ids.tolist()}")

                # Update batch and handle finished requests
                batch_indices_to_keep, new_original_indices = self._update_forward_batch_with_finished_handling(
                    forward_batch, next_token_ids, tokenizer, finished_requests, original_indices)

                # Update final results for active requests
                if batch_indices_to_keep is not None:
                    for new_idx, orig_idx in enumerate(new_original_indices):
                        final_results[orig_idx]['output'] = forward_batch.sequences[new_idx]
                    original_indices = new_original_indices

                # Handle newly finished requests
                for orig_idx in range(len(input_texts)):
                    if orig_idx in finished_requests and not final_results[orig_idx]['finished']:
                        final_results[orig_idx]['finished'] = True
                        print(f"✅ Request {orig_idx} completed!")

                # Update forward_batch to None if no active requests
                if batch_indices_to_keep is None:
                    forward_batch = None

            # Print final results
            print(f"\n🎉 === Final Generation Results ===")
            for i in range(len(input_texts)):
                result = final_results[i]
                status = "✅ Finished" if result['finished'] else "⏰ Max iterations reached"
                print(f"\nRequest {i} ({status}):")
                print(f"  Input:  '{result['input']}'")
                print(f"  Output: '{result['output']}'")

            # Verify shapes for the test
            self.assertEqual(len(final_results), len(input_texts))
            for i, result in final_results.items():
                self.assertIsNotNone(result['output'])
                self.assertTrue(len(result['output']) >= len(result['input']))

            print(f"\n✅ Test completed successfully!")

    def test_eos_detection(self):
        """Test EOS token detection logic specifically"""
        tokenizer = self._get_tokenizer()

        # Test normal token
        self.assertFalse(self._is_finished(100, tokenizer))

        # Test EOS token
        self.assertTrue(self._is_finished(tokenizer.eos_token_id, tokenizer))

        print("✅ EOS detection test passed!")


if __name__ == '__main__':
    unittest.main()
