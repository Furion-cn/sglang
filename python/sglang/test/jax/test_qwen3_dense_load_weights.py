#!/usr/bin/env python3
"""
Qwen3ForCausalLMJaxModel JAXModelLoader Integration Tests

Usage:
    python -m unittest test_qwen3_dense_load_weights.TestQwen3LoadWeights
    
    # Test with specific model path:
    MODEL_PATH=/path/to/jax/qwen3/model python -m unittest test_qwen3_dense_load_weights.TestQwen3LoadWeights.test_load_model_with_jax_loader
"""

import os
import jax
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
from sglang.test.jax.test_utils import create_device_mesh, jax_trace_context
from sglang.test.test_utils import CustomTestCase
from sglang.srt.jax.mem_cache.hash_kvcache import ReqToHashKVCachePool

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

class TestQwen3LoadWeights(CustomTestCase):
    """Test cases for Qwen3ForCausalLMJaxModel using JAXModelLoader"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_model_path = os.environ.get(
            'MODEL_PATH', 'Qwen/Qwen3-8B')
        import jax
        from jax.sharding import Mesh
        
        devices = jax.devices()
        # NOTE: Change mesh to use 2-way data parallelism.
        # This requires at least 2 * 4 = 8 devices.
        # The axes are (data, tensor, pipeline, experts).
        self.mesh = create_device_mesh(
            ici_parallelism=[2, 2, 1, 1],
            dcn_parallelism=[1, 1, 1, 1]
        )
        self.dp_degree = 2 # Data parallelism degree
        
        self.load_config = LoadConfig(load_format=LoadFormat.JAX)
        self.device_config = DeviceConfig("cpu")
        self.jax_loader = JAXModelLoader(self.load_config)
        self.tokenizer = self._get_tokenizer()
        self.enable_debug_tracer = os.environ.get("ENABLE_DEBUG_TRACER", "0") == "1"

    def _get_positions(self, x):
        return jnp.concatenate([
            jnp.arange(x.shape[1]) for _ in range(x.shape[0])
        ]).reshape(x.shape[0], x.shape[1])

    def _create_batch_from_texts(self, model_config, texts, tokenizer, cache_pool):
        """Create initial batch from texts with tokenization (no padding needed)

        Args:
            texts: List[str] input texts to process
            tokenizer: tokenizer to use for encoding
            cache_pool: The shared ReqToHashKVCachePool instance.

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

        # Create ForwardBatch using the provided shared cache_pool
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=len(actual_seq_lens),
            input_ids=input_ids_array,
            seq_lens=seq_lens,
            positions=positions_array,
            cache_loc=jnp.arange(jnp.sum(seq_lens), dtype=jnp.int32),
            out_cache_loc=None,
            extend_start_loc=extend_start_loc,
            token_to_kv_pool=cache_pool,
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

        # Fallback to Hugging Face - use Qwen3 model for better compatibility
        print("🌐 Loading tokenizer from Hugging Face: Qwen/Qwen3-30B-A3B")
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)

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
                f"\n=== Testing JAXModelLoader with Qwen3: {self.test_model_path} ===")
            print(f"Found {len(hf_weights_files)} msgpack files")

            model_config = ModelConfig(
                model_path=self.test_model_path,
                model_override_args="{}"
            )
            
            def custom_load_model_with_mesh(*args, **kwargs):
                mesh = kwargs.get('mesh')
                model_config = kwargs.get('model_config')
                
                with mesh:
                    model_config.hf_config.mesh = mesh
                    print(f"设置 mesh: {mesh}")
                    
                    model = self.jax_loader._initialize_jax_model(model_config)
                    pytree = self.jax_loader._get_jax_pytree(model_config)
                    self.jax_loader._load_pytree_weights(model, pytree)
                    
                    return model

            with patch('sglang.srt.model_loader.loader.get_model_architecture') as mock_arch:
                mock_arch.return_value = (Qwen3ForCausalLMJaxModel, None)

                print("\n🔄 Loading Qwen3 model with JAXModelLoader...")
                
                model = custom_load_model_with_mesh(
                    model_config=model_config,
                    device_config=self.device_config,
                    mesh=self.mesh,
                )

                print("✅ Qwen3 model loaded successfully!")
                self.assertIsInstance(model, Qwen3ForCausalLMJaxModel)
                self.assertIsNotNone(model.config)

                print(f"\n📋 Qwen3 Model config:")
                print(f"   vocab_size: {model.config.vocab_size}")
                print(f"   hidden_size: {model.config.hidden_size}")
                print(f"   num_hidden_layers: {model.config.num_hidden_layers}")

                print("\n🎉 JAXModelLoader integration test for Qwen3 completed successfully!")

                print("\n🔄 Test Qwen3 model input and output with JAXModelLoader...")
                
                print("\n🟢 Starting debug tracer session...")
                if self.enable_debug_tracer:
                    global_tracer.start_session()
                
                sampler = Sampler(rngs=nnx.Rngs(0))
                tokenizer = self._get_tokenizer()

                # Multiple questions to simulate batch > 1 scenario
                input_texts = [
                    "The capital of France is",
                    "What is 2+2?",
                    "Hello, my name is",
                    "1+1=?",
                ]

                # Split input_texts for data parallelism
                num_texts = len(input_texts)
                assert num_texts % self.dp_degree == 0, "Batch size must be divisible by data parallelism degree"
                group_size = num_texts // self.dp_degree
                input_texts_per_group = [input_texts[i:i+group_size] for i in range(0, num_texts, group_size)]
                
                print(f"Input texts split into {self.dp_degree} groups of size {group_size}")

                # Create a single shared KV cache pool
                cache_pool = ReqToHashKVCachePool(
                    head_num=model.config.num_key_value_heads,
                    head_dim=model.config.head_dim,
                    layer_num=model.config.num_hidden_layers,
                    dtype=jnp.bfloat16 if model.config.torch_dtype == "bfloat16" else jnp.float32,
                    max_seq_len=128,
                    max_batch_size=20,
                )

                # Create batches for each data parallel group
                batches = [self._create_batch_from_texts(model.config, texts, tokenizer, cache_pool) for texts in input_texts_per_group]
                
                # Unzip batches into separate lists
                input_ids_list, actual_seq_lens_list, forward_batch_list = zip(*batches)
                forward_batch_list = list(forward_batch_list)
                
                initial_complete_sequences_list = []
                for i in range(self.dp_degree):
                    group_sequences = []
                    for batch_idx in range(group_size):
                        start_idx = sum(actual_seq_lens_list[i][:batch_idx])
                        end_idx = start_idx + actual_seq_lens_list[i][batch_idx]
                        initial_tokens = [int(token) for token in input_ids_list[i][start_idx:end_idx]]
                        group_sequences.append(initial_tokens)
                    initial_complete_sequences_list.append(group_sequences)

                # Define the parallel computation step
                @jax.pmap
                def parallel_step(forward_batch, sampling_info):
                    y = model(forward_batch.input_ids, forward_batch.positions, forward_batch)
                    next_token_ids = sampler(y, sampling_info=sampling_info)
                    return next_token_ids
                
                jax_profiling_dir = os.environ.get("JAX_TRACE_PROFILING_DIR", "/tmp/jax_profiling")
                with self.mesh, jax_trace_context(jax_profiling_dir):
                    for i in range(10):
                        # Stack forward_batch objects for pmap. This requires ForwardBatch to be a PyTree.
                        # Assuming ForwardBatch is a dataclass or compatible PyTree.
                        stacked_forward_batch = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *forward_batch_list)
                        
                        # Create and stack sampling info
                        sampling_info_list = []
                        for j in range(self.dp_degree):
                            sampling_info_list.append(SamplingBatchInfo(
                                temperatures=jnp.full((group_size, 1), 1.0),
                                top_ps=jnp.full((group_size, 1), 1.0),
                                top_ks=jnp.ones((group_size, 1)),
                                min_ps=jnp.full((group_size, 1), 0.0),
                                vocab_size=model.config.vocab_size,
                            ))
                        stacked_sampling_info = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *sampling_info_list)
                        
                        # Execute in parallel
                        next_token_ids_sharded = parallel_step(
                            stacked_forward_batch, stacked_sampling_info
                        )

                        # Unstack results and update for next iteration
                        next_token_ids_list = [next_token_ids_sharded[j] for j in range(self.dp_degree)]
                        
                        # Manually update the host-side state for each group.
                        for j in range(self.dp_degree):
                            initial_complete_sequences_list[j] = self.update_forward_batch_with_sequences(
                                forward_batch_list[j], next_token_ids_list[j], tokenizer, initial_complete_sequences_list[j])

                # Decode complete results for each sequence
                print(f"\n=== Qwen3 Complete Generation Results ===")
                # Flatten the results from all groups
                all_complete_sequences = [seq for group in initial_complete_sequences_list for seq in group]
                all_actual_seq_lens = [l for group_lens in actual_seq_lens_list for l in group_lens]

                for batch_idx in range(len(input_texts)):
                    full_sequence = all_complete_sequences[batch_idx]
                    decoded_full = tokenizer.decode(full_sequence)
                    original_len = all_actual_seq_lens[batch_idx]
                    original_tokens = full_sequence[:original_len]
                    generated_tokens = full_sequence[original_len:]
                    
                    original_text = tokenizer.decode(original_tokens)
                    generated_text = tokenizer.decode(generated_tokens) if generated_tokens else ""
                    
                    print(f"Sequence {batch_idx}: {full_sequence}")
                    print(f"Original tokens {batch_idx}: {original_tokens}")
                    print(f"Generated tokens {batch_idx}: {generated_tokens}")
                    print(f"Complete decoded text {batch_idx}: '{decoded_full}'")
                    print(f"Original question {batch_idx}: '{original_text}'")
                    print(f"Generated answer {batch_idx}: '{generated_text}'")
                    print(f"Total length {batch_idx}: {len(full_sequence)} (original: {original_len}, generated: {len(generated_tokens)})")
                    print()

                print("\n🔴 Ending debug tracer session...")
                if self.enable_debug_tracer:
                    debug_file = global_tracer.end_session()
                    if debug_file:
                        print(f"✅ Debug trace saved to: {debug_file}")
                else:
                    print("⚠️  Debug trace not saved")

        except Exception as e:
            if 'global_tracer' in locals():
                try:
                    if self.enable_debug_tracer:
                        global_tracer.end_session()
                        print("🔴 Debug tracer session ended due to exception")
                except:
                    pass
            self.fail(f"JAXModelLoader integration test for Qwen3 failed: {e}")

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

    def update_forward_batch_with_sequences(self, forward_batch: ForwardBatch, next_token_ids, tokenizer, complete_sequences):
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
            decoded_token = tokenizer.decode([current_token_id])
            
            complete_sequences[batch_idx].append(current_token_id)
            
            # update cache loc
            old_cache_loc = forward_batch.cache_loc[
                cache_start_loc:cache_start_loc + seq_len]
            new_cache_loc_list.append(jnp.concatenate(
                [old_cache_loc, forward_batch.out_cache_loc[batch_idx:batch_idx+1]], axis=0))
            cache_start_loc += seq_len
            
            print(f"Batch {batch_idx}: token_id={current_token_id}, decoded='{decoded_token}'")
        
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

        # update forward mode
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            forward_batch.forward_mode = ForwardMode.DECODE
            
        return complete_sequences

if __name__ == '__main__':
    unittest.main()