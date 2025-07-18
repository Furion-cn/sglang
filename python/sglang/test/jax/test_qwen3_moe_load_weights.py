#!/usr/bin/env python3
"""
Qwen3MoeForCausalLMJaxModel JAXModelLoader Integration Tests

Usage:
    python -m unittest test_qwen3_moe_load_weights.TestQwen3MoeLoadWeights
    
    # Test with specific model path:
    MODEL_PATH=/path/to/jax/qwen3_moe/model python -m unittest test_qwen3_moe_load_weights.TestQwen3MoeLoadWeights.test_load_model_with_jax_loader
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
from sglang.srt.jax.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.jax.models.qwen3_moe import Qwen3MoeForCausalLMJaxModel
from sglang.srt.jax.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.model_loader.loader import JAXModelLoader
from sglang.test.jax.test_utils import create_device_mesh, jax_trace_context
from sglang.test.test_utils import CustomTestCase
from sglang.srt.jax.mem_cache.hash_kvcache import ReqToHashKVCachePool
import jax.tree_util

# Register ForwardBatch as a JAX PyTree to allow it to be passed through jax.pmap.
# This separates the object's fields into dynamic JAX arrays (children) and
# static metadata (aux_data). The `token_to_kv_pool` is treated as static
# because it's a complex object that should not be traced by JAX.
def _forward_batch_flatten(fb: ForwardBatch):
    """Flattens the ForwardBatch for JAX transformations."""
    children = (
        fb.input_ids,
        fb.seq_lens,
        fb.positions,
        fb.cache_loc,
        fb.out_cache_loc,
        fb.extend_start_loc,
        fb.batch_size,  # batch_size is traced as a JAX array
    )
    aux_data = (fb.forward_mode, fb.token_to_kv_pool)
    return children, aux_data

def _forward_batch_unflatten(aux_data, children):
    """Unflattens the ForwardBatch from JAX representations."""
    forward_mode, token_to_kv_pool = aux_data
    (
        input_ids,
        seq_lens,
        positions,
        cache_loc,
        out_cache_loc,
        extend_start_loc,
        batch_size,
    ) = children
    return ForwardBatch(
        forward_mode=forward_mode,
        batch_size=batch_size,
        input_ids=input_ids,
        seq_lens=seq_lens,
        positions=positions,
        cache_loc=cache_loc,
        out_cache_loc=out_cache_loc,
        extend_start_loc=extend_start_loc,
        token_to_kv_pool=token_to_kv_pool,
    )

jax.tree_util.register_pytree_node(
    ForwardBatch, _forward_batch_flatten, _forward_batch_unflatten
)

# Register LogitsProcessorOutput as a JAX PyTree. This allows it to be returned
# from JIT-compiled functions. The `next_token_logits` is the only dynamic
# JAX array (child), and there is no static metadata.
def _logits_processor_output_flatten(output: LogitsProcessorOutput):
    """Flattens the LogitsProcessorOutput for JAX transformations."""
    children = (output.next_token_logits,)
    aux_data = None
    return children, aux_data

def _logits_processor_output_unflatten(aux_data, children):
    """Unflattens the LogitsProcessorOutput from JAX representations."""
    (next_token_logits,) = children
    return LogitsProcessorOutput(next_token_logits=next_token_logits)

jax.tree_util.register_pytree_node(
    LogitsProcessorOutput,
    _logits_processor_output_flatten,
    _logits_processor_output_unflatten,
)


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

class TestQwen3MoeLoadWeights(CustomTestCase):
    """Test cases for Qwen3MoeForCausalLMJaxModel using JAXModelLoader"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_model_path = os.environ.get(
            'MODEL_PATH', '/tmp/test_qwen_moe_jax_model')
        import jax
        from jax.sharding import Mesh
        
        self.dp_size = int(os.environ.get('DP_SIZE', '1'))
        print(f"🔧 Setup: DP_SIZE={self.dp_size}")
        
        devices = jax.devices()        
        
        if self.dp_size > 1:
            print(f"🔧 Setting up mesh for data parallelism (DP_SIZE={self.dp_size})")
            self.mesh = create_device_mesh(
                ici_parallelism=[self.dp_size, len(devices) // self.dp_size, 1, 1],
                dcn_parallelism=[1, 1, 1, 1]
            )
        else:
            print(f"🔧 Setting up mesh for standard mode (DP_SIZE={self.dp_size})")
            self.mesh = create_device_mesh(
                ici_parallelism=[1, 4, 1, 1],
                dcn_parallelism=[1, 1, 1, 1]
            )
        
        print(f"🔧 Mesh configuration: {self.mesh.devices.shape}")
        print(f"🔧 Mesh axis names: {self.mesh.axis_names}")
        
        self.load_config = LoadConfig(load_format=LoadFormat.JAX)
        self.device_config = DeviceConfig("cpu")
        self.jax_loader = JAXModelLoader(self.load_config)
        self.tokenizer = self._get_tokenizer()
        self.enable_debug_tracer = os.environ.get("ENABLE_DEBUG_TRACER", "0") == "1"

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
        cache_pool = ReqToHashKVCachePool(
            head_num=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            layer_num=model_config.num_hidden_layers,
            dtype=jnp.bfloat16 if model_config.torch_dtype == "bfloat16" else jnp.float32,
            max_seq_len=128,
            max_batch_size=20,
        )

        # Create ForwardBatch
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

    def _create_batch_from_texts_dp(self, model_config, texts, tokenizer):
        """Create batch with data parallelism - simplified version to avoid JAX tracing issues

        Args:
            texts: List[str] input texts to process
            tokenizer: tokenizer to use for encoding

        Returns:
            tuple: (sharded_forward_batch, device_count)
        """
        print(f"\n🔄 开始创建真正的DP批次，输入文本数量: {len(texts)}")
        
        # 第一步：分词，保持每个序列分离
        tokenized_inputs = []
        actual_seq_lens = []
        max_seq_len = 0
        
        for i, text in enumerate(texts):
            tokens = tokenizer.encode(text)
            tokenized_inputs.append(tokens)
            actual_seq_lens.append(len(tokens))
            max_seq_len = max(max_seq_len, len(tokens))
            print(f"  序列 {i}: 长度={len(tokens)}, tokens={tokens[:5]}...")

        batch_size = len(texts)
        device_count = self.mesh.devices.shape[0]  # data维度的设备数
        
        # 确保batch_size能被device_count整除
        if batch_size % device_count != 0:
            raise ValueError(f"batch_size ({batch_size}) 必须能被 device_count ({device_count}) 整除")
            
        seqs_per_device = batch_size // device_count
        print(f"  批次大小: {batch_size}, 最大序列长度: {max_seq_len}")
        print(f"  设备数量: {device_count}, 每设备序列数: {seqs_per_device}")

        # 第二步：创建填充的2D数组，保持 [batch_size, max_seq_len] 形状
        pad_token_id = getattr(tokenizer, 'pad_token_id', 0) or 0
        
        # 创建2D数组: [batch_size, max_seq_len]
        input_ids_2d_list = []
        positions_2d_list = []
        
        for i, tokens in enumerate(tokenized_inputs):
            seq_len = len(tokens)
            # 填充input_ids
            row_ids = jnp.array(tokens + [pad_token_id] * (max_seq_len - seq_len), dtype=jnp.int32)
            input_ids_2d_list.append(row_ids)
            
            # 填充positions
            row_positions = jnp.array(list(range(seq_len)) + [0] * (max_seq_len - seq_len), dtype=jnp.int32)
            positions_2d_list.append(row_positions)

        input_ids_2d = jnp.stack(input_ids_2d_list)  # [batch_size, max_seq_len]
        positions_2d = jnp.stack(positions_2d_list)  # [batch_size, max_seq_len]
        seq_lens = jnp.array(actual_seq_lens, dtype=jnp.int32)  # [batch_size]
        
        print(f"  2D数组形状: input_ids={input_ids_2d.shape}, positions={positions_2d.shape}")
        print(f"  设备数量: {device_count}, 每设备序列数: {seqs_per_device}")
        
        # 第三步：创建JAX sharding策略
        from jax.sharding import NamedSharding, PartitionSpec as P
        
        # 数据并行sharding策略
        batch_sharding = NamedSharding(self.mesh, P('data'))          # 只分片batch维度: [batch_size]
        batch_seq_sharding = NamedSharding(self.mesh, P('data', None)) # 分片batch，复制seq: [batch_size, seq_len]
        replicated_sharding = NamedSharding(self.mesh, P(None))       # 完全复制
        
        print(f"  数据分片策略: batch_seq={batch_seq_sharding}")
        
        # 第四步：使用device_put分片所有数据
        print(f"  🔄 开始数据分片和设备放置...")
        
        sharded_input_ids_2d = jax.device_put(input_ids_2d, batch_seq_sharding)
        sharded_positions_2d = jax.device_put(positions_2d, batch_seq_sharding)
        sharded_seq_lens = jax.device_put(seq_lens, batch_sharding)
        
        print(f"  ✅ 2D数据分片完成!")
        print(f"  分片后形状: input_ids={sharded_input_ids_2d.shape}, seq_lens={sharded_seq_lens.shape}")
        
        # 第五步：在CPU上预计算所有需要的数据，避免pmap内部的复杂操作
        print(f"  🔄 预计算批次数据...")
        
        # 为每个设备预计算扁平化的数据
        device_input_ids = []
        device_positions = []
        device_seq_lens = []
        device_cache_locs = []
        device_extend_start_locs = []
        
        for device_id in range(device_count):
            # 每个设备处理的序列范围
            start_seq = device_id * seqs_per_device
            end_seq = start_seq + seqs_per_device
            
            device_seq_lens_list = actual_seq_lens[start_seq:end_seq]
            device_seq_lens_array = jnp.array(device_seq_lens_list, dtype=jnp.int32)
            
            # 扁平化这个设备的数据
            device_tokens = []
            device_pos = []
            for local_seq_idx, global_seq_idx in enumerate(range(start_seq, end_seq)):
                seq_len = actual_seq_lens[global_seq_idx]
                tokens = tokenized_inputs[global_seq_idx]
                device_tokens.extend(tokens)
                device_pos.extend(list(range(seq_len)))
            
            device_input_ids.append(jnp.array(device_tokens, dtype=jnp.int32))
            device_positions.append(jnp.array(device_pos, dtype=jnp.int32))
            device_seq_lens.append(device_seq_lens_array)
            
            # 创建cache_loc和extend_start_loc
            total_tokens = sum(device_seq_lens_list)
            device_cache_locs.append(jnp.arange(total_tokens, dtype=jnp.int32))
            
            extend_start_loc = jnp.cumsum(jnp.concatenate([jnp.array([0]), device_seq_lens_array[:-1]]))
            device_extend_start_locs.append(extend_start_loc)
            
            print(f"    设备 {device_id}: {len(device_seq_lens_list)} 序列, {total_tokens} tokens")
        
        # 第六步：分片预计算的数据
        print(f"  🔄 分片预计算的数据...")
        
        # 由于每个设备的token数量可能不同，我们需要pad到相同长度
        max_tokens_per_device = max(len(tokens) for tokens in device_input_ids)
        max_seqs_per_device = max(len(seq_lens) for seq_lens in device_seq_lens)
        
        # Pad所有设备的数据到相同长度
        padded_input_ids = []
        padded_positions = []
        padded_seq_lens = []
        padded_cache_locs = []
        padded_extend_start_locs = []
        
        for device_id in range(device_count):
            # Pad input_ids和positions
            tokens = device_input_ids[device_id]
            positions = device_positions[device_id]
            padded_tokens = jnp.concatenate([tokens, jnp.zeros(max_tokens_per_device - len(tokens), dtype=jnp.int32)])
            padded_pos = jnp.concatenate([positions, jnp.zeros(max_tokens_per_device - len(positions), dtype=jnp.int32)])
            
            # Pad seq_lens
            seq_lens = device_seq_lens[device_id]
            padded_seq = jnp.concatenate([seq_lens, jnp.zeros(max_seqs_per_device - len(seq_lens), dtype=jnp.int32)])
            
            # Pad cache_loc
            cache_loc = device_cache_locs[device_id]
            padded_cache = jnp.concatenate([cache_loc, jnp.zeros(max_tokens_per_device - len(cache_loc), dtype=jnp.int32)])
            
            # Pad extend_start_loc
            extend_start_loc = device_extend_start_locs[device_id]
            padded_extend = jnp.concatenate([extend_start_loc, jnp.zeros(max_seqs_per_device - len(extend_start_loc), dtype=jnp.int32)])
            
            padded_input_ids.append(padded_tokens)
            padded_positions.append(padded_pos)
            padded_seq_lens.append(padded_seq)
            padded_cache_locs.append(padded_cache)
            padded_extend_start_locs.append(padded_extend)
        
        # 堆叠为JAX数组
        final_input_ids = jnp.stack(padded_input_ids)      # [device_count, max_tokens_per_device]
        final_positions = jnp.stack(padded_positions)      # [device_count, max_tokens_per_device]
        final_seq_lens = jnp.stack(padded_seq_lens)        # [device_count, max_seqs_per_device]
        final_cache_locs = jnp.stack(padded_cache_locs)    # [device_count, max_tokens_per_device]
        final_extend_start_locs = jnp.stack(padded_extend_start_locs)  # [device_count, max_seqs_per_device]
        final_batch_sizes = jnp.array([seqs_per_device] * device_count, dtype=jnp.int32)  # [device_count]
        
        # 第七步：创建KV缓存池
        cache_pool = ReqToHashKVCachePool(
            head_num=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            layer_num=model_config.num_hidden_layers,
            dtype=jnp.bfloat16,
            max_seq_len=128,
            max_batch_size=20,
        )
        
        # 第八步：创建ForwardBatch对象 - 简化版本
        print(f"  🔄 创建ForwardBatch对象...")
        
        def create_simple_forward_batch(input_ids, positions, seq_lens, cache_loc, 
                                      extend_start_loc, batch_size):
            """简化的ForwardBatch创建"""
            return ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=batch_size,
                input_ids=input_ids,
                seq_lens=seq_lens,
                positions=positions,
                cache_loc=cache_loc,
                out_cache_loc=None,
                extend_start_loc=extend_start_loc,
                token_to_kv_pool=cache_pool,
            )
        
        # 使用pmap创建ForwardBatch
        sharded_forward_batch = jax.pmap(create_simple_forward_batch, axis_name='data')(
            final_input_ids,
            final_positions, 
            final_seq_lens,
            final_cache_locs,
            final_extend_start_locs,
            final_batch_sizes
        )
        
        print(f"  ✅ DP批次创建完成! ForwardBatch已在所有设备上准备就绪")
        
        return sharded_forward_batch, device_count

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

        # Fallback to Hugging Face - use Qwen3 MoE model for better compatibility
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
        """Test loading Qwen3 MoE model using JAXModelLoader (integration test)"""
        if self.dp_size > 1:
            self.skipTest("DP_SIZE must be > 1 for data parallelism test. Set DP_SIZE environment variable.")
        
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
                f"\n=== Testing JAXModelLoader with Qwen3 MoE: {self.test_model_path} ===")
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
                mock_arch.return_value = (Qwen3MoeForCausalLMJaxModel, None)

                print("\n🔄 Loading Qwen3 MoE model with JAXModelLoader...")
                
                model = custom_load_model_with_mesh(
                    model_config=model_config,
                    device_config=self.device_config,
                    mesh=self.mesh,
                )

                print("✅ Qwen3 MoE model loaded successfully!")
                self.assertIsInstance(model, Qwen3MoeForCausalLMJaxModel)
                self.assertIsNotNone(model.config)

                print(f"\n📋 Qwen3 MoE Model config:")
                print(f"   vocab_size: {model.config.vocab_size}")
                print(f"   hidden_size: {model.config.hidden_size}")
                print(f"   num_hidden_layers: {model.config.num_hidden_layers}")
                print(f"   num_experts: {getattr(model.config, 'num_experts', 'N/A')}")
                print(f"   num_experts_per_tok: {getattr(model.config, 'num_experts_per_tok', 'N/A')}")
                print(f"   moe_intermediate_size: {getattr(model.config, 'moe_intermediate_size', 'N/A')}")
                print(f"   mlp_only_layers: {getattr(model.config, 'mlp_only_layers', 'N/A')}")

                print("\n🎉 JAXModelLoader integration test for Qwen3 MoE completed successfully!")

                print("\n🔄 Test Qwen3 MoE model input and output with JAXModelLoader...")
                
                print("\n🟢 Starting debug tracer session...")
                if self.enable_debug_tracer:
                    global_tracer.start_session()
                
                sampler = Sampler(rngs=nnx.Rngs(0))
                tokenizer = self._get_tokenizer()

                # Multiple questions to simulate batch > 1 scenario
                # Use simpler prompts for MoE testing
                input_texts = [
                    "1+1=?",
                ]

                input_ids_array, actual_seq_lens, forward_batch = self._create_batch_from_texts(
                    model.config, input_texts, tokenizer)

                # 初始化完整序列历史，用于累积生成的token
                complete_sequences = []
                for batch_idx in range(len(input_texts)):
                    # 获取每个序列的初始token
                    start_idx = sum(actual_seq_lens[:batch_idx])
                    end_idx = start_idx + actual_seq_lens[batch_idx]
                    initial_tokens = [int(token) for token in input_ids_array[start_idx:end_idx]]
                    complete_sequences.append(initial_tokens)

                print(f"Input text batch: {input_texts}")
                print(f"Batch size: {len(input_texts)}")
                print(f"Actual sequence lengths: {actual_seq_lens}")
                print(f"Input tokens shape: {input_ids_array.shape}")
                print(f"Input tokens: {input_ids_array}")
                print(f"Initial complete sequences: {complete_sequences}")
                
                jax_profiling_dir = os.environ.get("JAX_TRACE_PROFILING_DIR", "/tmp/jax_profiling")
                with self.mesh:
                    for i in range(10):  # Reduced iterations for MoE testing
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

                        complete_sequences = self.update_forward_batch_with_sequences(
                            forward_batch, next_token_ids, tokenizer, complete_sequences)

                # Decode complete results for each sequence
                print(f"\n=== Qwen3 MoE Complete Generation Results ===")
                for batch_idx in range(len(input_texts)):
                    full_sequence = complete_sequences[batch_idx]
                    decoded_full = tokenizer.decode(full_sequence)
                    original_len = actual_seq_lens[batch_idx]
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
            self.fail(f"JAXModelLoader integration test for Qwen3 MoE failed: {e}")

    def test_load_model_with_jax_loader_dp(self):
        """Test loading Qwen3 MoE model using JAXModelLoader with Data Parallelism"""
        if self.dp_size <= 1:
            self.skipTest("DP_SIZE must be > 1 for data parallelism test. Set DP_SIZE environment variable.")
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

            print(f"\n=== 🚀 Testing JAX Data Parallelism with DP_SIZE={self.dp_size} ===")
            print(f"Model path: {self.test_model_path}")
            print(f"Device configuration: {self.mesh.devices.shape}")

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
                mock_arch.return_value = (Qwen3MoeForCausalLMJaxModel, None)

                print("\n🔄 Loading Qwen3 MoE model...")
                
                model = custom_load_model_with_mesh(
                    model_config=model_config,
                    device_config=self.device_config,
                    mesh=self.mesh,
                )

                print("✅ Model loaded successfully!")
                
                # 分离模型定义和状态，以避免JAX捕获巨大的常量
                model_state, model_def = nnx.split(model)
                
                # 准备DP测试数据，序列数量必须能被dp_size整除
                base_texts = ["1+1=?", "2+2=?", "3+3=?", "4+4=?"]
                # 确保有足够的序列用于dp_size个设备
                input_texts = base_texts[:self.dp_size] if len(base_texts) >= self.dp_size else base_texts * ((self.dp_size // len(base_texts)) + 1)
                input_texts = input_texts[:self.dp_size]  # 精确匹配dp_size
                
                tokenizer = self._get_tokenizer()
                
                print(f"\n📊 DP Configuration:")
                print(f"  Input sequences: {len(input_texts)}")
                print(f"  Device count: {self.mesh.devices.shape[0]}")
                print(f"  Sequences per device: {len(input_texts) // self.mesh.devices.shape[0]}")
                print(f"  Input texts: {input_texts}")
                
                # 使用DP批次创建方法
                sharded_forward_batch, device_count = self._create_batch_from_texts_dp(
                    model_config, input_texts, tokenizer)

                # 创建真正的并行前向传播+采样函数
                def dp_forward_and_sample(model_def, model_state, forward_batch, temps, top_ps, top_ks, min_ps, rng_key):
                    """数据并行：前向传播 + 采样 - 在每个设备上同时执行完整流程"""
                    # 1. 重新组合模型
                    model = nnx.merge(model_def, model_state)
                    
                    # 2. 模型前向传播
                    outputs = model(forward_batch.input_ids, forward_batch.positions, forward_batch)
                    
                    # 3. 直接在同一个pmap中采样
                    key, new_key = jax.random.split(rng_key)
                    next_token_ids = jax.random.categorical(key, outputs.next_token_logits, axis=-1)
                    next_token_ids = next_token_ids[..., None]
                    
                    return (outputs, next_token_ids), new_key

                # 使用pmap实现真正的数据并行 - 一个函数搞定前向+采样！
                print(f"\n🔄 Creating unified data parallel forward+sample function...")
                dp_forward_sample = jax.pmap(
                    dp_forward_and_sample,
                    axis_name='data',
                    in_axes=(None, None, 0, 0, 0, 0, 0, 0),  # model_def, model_state, forward_batch, ...
                    out_axes=((0, 0), 0),
                    static_broadcasted_argnums=(0,)
                )
                
                print(f"\n🚀 Starting real parallel execution...")
                print(f"  Device count: {device_count}")
                print(f"  Sequences per device: {len(input_texts) // device_count}")
                
                # 为每个设备创建分片的RNG keys
                rng_keys = jax.random.split(jax.random.PRNGKey(0), self.dp_size)
                
                # if self.enable_debug_tracer:
                #     global_tracer.start_session()
                
                jax_profiling_dir = os.environ.get("JAX_TRACE_PROFILING_DIR", "/tmp/jax_profiling")
                with self.mesh:
                    
                    # 初始化完整序列历史，用于累积生成的token
                    complete_sequences = []
                    for batch_idx in range(len(input_texts)):
                        # 从sharded_forward_batch中提取每个序列的初始token
                        device_id = batch_idx // (len(input_texts) // device_count)
                        local_batch_idx = batch_idx % (len(input_texts) // device_count)
                        
                        # 获取该序列在对应设备上的长度
                        seq_len = int(sharded_forward_batch.seq_lens[device_id][local_batch_idx])
                        
                        # 计算该序列在扁平化数组中的起始位置
                        if local_batch_idx == 0:
                            start_idx = 0
                        else:
                            start_idx = int(sharded_forward_batch.extend_start_loc[device_id][local_batch_idx])
                        
                        end_idx = start_idx + seq_len
                        initial_tokens = [int(token) for token in sharded_forward_batch.input_ids[device_id][start_idx:end_idx]]
                        complete_sequences.append(initial_tokens)
                    
                    print(f"\n📝 Initial sequences for decode:")
                    for i, seq in enumerate(complete_sequences):
                        decoded = tokenizer.decode(seq)
                        print(f"  Sequence {i}: {seq} -> '{decoded}'")
                    
                    # 准备分片的sampling参数
                    seqs_per_device = len(input_texts) // device_count
                    sampling_temps = jnp.full((device_count, seqs_per_device, 1), 1.0)
                    sampling_top_ps = jnp.full((device_count, seqs_per_device, 1), 1.0)
                    sampling_top_ks = jnp.ones((device_count, seqs_per_device, 1))
                    sampling_min_ps = jnp.full((device_count, seqs_per_device, 1), 0.0)
                    
                    # 执行生成循环 - 一个pmap搞定前向+采样！
                    for iteration in range(10):
                        print(f"\n  🔄 DP parallel iteration {iteration + 1}/10")
                        
                        # 一次调用完成：前向传播 + 采样！
                        (outputs, next_token_ids), rng_keys = dp_forward_sample(
                            model_def,
                            model_state,
                            sharded_forward_batch,
                            sampling_temps,
                            sampling_top_ps, 
                            sampling_top_ks,
                            sampling_min_ps,
                            rng_keys
                        )
                        
                        if iteration == 0:
                            # 第一次迭代显示详细统计
                            print(f"    ✅ All {device_count} devices executed forward+sample in parallel!")
                            print(f"    📊 Output statistics:")
                            print(f"      Output shape: {outputs.next_token_logits.shape}")
                            print(f"      Sequences per device: {outputs.next_token_logits.shape[1]}")
                            print(f"      Vocabulary size: {outputs.next_token_logits.shape[2]}")
                            print(f"      Next tokens shape: {next_token_ids.shape}")
                        
                        # 更新complete_sequences和sharded_forward_batch
                        complete_sequences = self.update_dp_forward_batch_with_sequences(
                            sharded_forward_batch, next_token_ids, tokenizer, complete_sequences, device_count)
                        
                        # 每几次迭代显示进度
                        if (iteration + 1) % 3 == 0:
                            print(f"    🔄 Generated {iteration + 1} tokens...")
                            for i, seq in enumerate(complete_sequences):
                                decoded = tokenizer.decode(seq)
                                print(f"      Sequence {i}: '{decoded}'")

                # 显示最终生成结果
                print(f"\n=== 🎉 JAX Data Parallelism Complete Generation Results ===")
                for batch_idx in range(len(input_texts)):
                    full_sequence = complete_sequences[batch_idx]
                    decoded_full = tokenizer.decode(full_sequence)
                    
                    # 计算原始长度
                    device_id = batch_idx // (len(input_texts) // device_count)
                    local_batch_idx = batch_idx % (len(input_texts) // device_count)
                    original_len = int(sharded_forward_batch.seq_lens[device_id][local_batch_idx])
                    
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

                # if self.enable_debug_tracer:
                #     debug_file = global_tracer.end_session()
                #     if debug_file:
                #         print(f"✅ Debug trace saved to: {debug_file}")

        except Exception as e:
            # if 'global_tracer' in locals():
            #     try:
            #         if self.enable_debug_tracer:
            #             global_tracer.end_session()
            #             print("🔴 Debug tracer session ended due to exception")
            #     except:
            #         pass
            self.fail(f"Data Parallelism test failed: {e}")

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

    def update_dp_forward_batch_with_sequences(self, sharded_forward_batch, next_token_ids, tokenizer, complete_sequences, device_count):
        """更新数据并行的ForwardBatch和complete_sequences"""
        # 更新complete_sequences
        for batch_idx in range(len(complete_sequences)):
            device_id = batch_idx // (len(complete_sequences) // device_count)
            local_batch_idx = batch_idx % (len(complete_sequences) // device_count)
            
            current_token_id = int(next_token_ids[device_id, local_batch_idx, 0])
            complete_sequences[batch_idx].append(current_token_id)
            
            decoded_token = tokenizer.decode([current_token_id])
            if batch_idx < 2:  # 只显示前两个序列的详细信息
                print(f"    Device {device_id}, Local batch {local_batch_idx}, Global batch {batch_idx}: token_id={current_token_id}, decoded='{decoded_token}'")
        
        # 更新sharded_forward_batch用于下一次迭代
        # 这里需要在每个设备上更新ForwardBatch
        def update_forward_batch_on_device(forward_batch, next_tokens):
            """在每个设备上更新ForwardBatch"""
            # 更新seq_lens
            new_seq_lens = forward_batch.seq_lens + 1
            
            # 更新positions (对于decode模式，position是当前序列长度-1)
            new_positions = new_seq_lens - 1
            
            # 更新input_ids (对于decode模式，input_ids是新生成的token)
            new_input_ids = next_tokens[:, 0]  # 取第一个维度
            
            # 更新extend_start_loc
            new_extend_start_loc = jnp.cumsum(jnp.concatenate([jnp.array([0]), new_seq_lens[:-1]]))
            
            # 更新cache_loc (添加新的cache位置)
            batch_size = len(new_seq_lens)
            if forward_batch.out_cache_loc is None:
                # 第一次更新，创建out_cache_loc
                max_cache_loc = jnp.max(forward_batch.cache_loc) if forward_batch.cache_loc.size > 0 else -1
                new_out_cache_loc = jnp.arange(max_cache_loc + 1, max_cache_loc + 1 + batch_size, dtype=jnp.int32)
            else:
                # 后续更新，扩展cache_loc
                new_out_cache_loc = jnp.max(forward_batch.cache_loc) + jnp.arange(1, batch_size + 1, dtype=jnp.int32)
            
            # 更新cache_loc，将out_cache_loc添加到cache_loc中
            # 这里简化处理，实际实现可能需要更复杂的缓存管理
            new_cache_loc = new_out_cache_loc
            
            return ForwardBatch(
                forward_mode=ForwardMode.DECODE,  # 切换到DECODE模式
                batch_size=forward_batch.batch_size,
                input_ids=new_input_ids,
                seq_lens=new_seq_lens,
                positions=new_positions,
                cache_loc=new_cache_loc,
                out_cache_loc=None,  # decode模式下重置
                extend_start_loc=new_extend_start_loc,
                token_to_kv_pool=forward_batch.token_to_kv_pool,
            )
        
        # 使用pmap在所有设备上更新ForwardBatch
        dp_update = jax.pmap(update_forward_batch_on_device, axis_name='data')
        
        # 更新sharded_forward_batch
        updated_batch = dp_update(sharded_forward_batch, next_token_ids)
        
        # 更新原始的sharded_forward_batch对象的属性
        sharded_forward_batch.forward_mode = updated_batch.forward_mode
        sharded_forward_batch.input_ids = updated_batch.input_ids
        sharded_forward_batch.seq_lens = updated_batch.seq_lens
        sharded_forward_batch.positions = updated_batch.positions
        sharded_forward_batch.cache_loc = updated_batch.cache_loc
        sharded_forward_batch.out_cache_loc = updated_batch.out_cache_loc
        sharded_forward_batch.extend_start_loc = updated_batch.extend_start_loc
        
        return complete_sequences

if __name__ == '__main__':
    unittest.main()