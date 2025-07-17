#!/usr/bin/env python3
"""
Qwen3MoeForCausalLMJaxModel JAXModelLoader Integration Tests

Usage:
    # 标准测试（纯TP）
    python -m unittest test_qwen3_moe_load_weights.TestQwen3MoeLoadWeights
    
    # 数据并行测试（外置DP）
    USE_DATA_PARALLEL=1 DP_SIZE=2 python -m unittest test_qwen3_moe_load_weights.TestQwen3MoeLoadWeights.test_load_model_with_jax_loader
    
    # DP + TP混合（2个设备DP，2个设备TP）
    USE_DATA_PARALLEL=1 DP_SIZE=2 python -m unittest test_qwen3_moe_load_weights.TestQwen3MoeLoadWeights.test_load_model_with_jax_loader
    
    # 纯DP模式（4个设备都做DP）
    USE_DATA_PARALLEL=1 DP_SIZE=4 python -m unittest test_qwen3_moe_load_weights.TestQwen3MoeLoadWeights.test_load_model_with_jax_loader
    
    # Test with specific model path:
    MODEL_PATH=/path/to/jax/qwen3_moe/model python -m unittest test_qwen3_moe_load_weights.TestQwen3MoeLoadWeights.test_load_model_with_jax_loader

Environment Variables:
    MODEL_PATH: 模型路径
    USE_DATA_PARALLEL: 是否启用数据并行 (0/1)
    DP_SIZE: 数据并行的设备数量
    ENABLE_DEBUG_TRACER: 是否启用debug追踪 (0/1)
    JAX_TRACE_PROFILING_DIR: JAX profiling输出目录
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
from sglang.srt.jax.models.qwen3_moe import Qwen3MoeForCausalLMJaxModel
from sglang.srt.jax.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.model_loader.loader import JAXModelLoader
from sglang.test.jax.test_utils import create_device_mesh, jax_trace_context
from sglang.test.test_utils import CustomTestCase
from sglang.srt.jax.mem_cache.hash_kvcache import ReqToHashKVCachePool
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

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
        os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'
        """Set up test fixtures"""
        self.test_model_path = os.environ.get(
            'MODEL_PATH', '/tmp/test_qwen_moe_jax_model')
        import jax
        from jax.sharding import Mesh
        
        devices = jax.devices()        
        
        # 支持DP的mesh配置
        # 选项1: 纯DP (4个设备做数据并行)
        # self.mesh = create_device_mesh(
        #     ici_parallelism=[4, 1, 1, 1],  # data=4, tensor=1
        #     dcn_parallelism=[1, 1, 1, 1]
        # )
        
        # 选项2: DP + TP混合 (2个设备DP, 2个设备TP)
        # self.mesh = create_device_mesh(
        #     ici_parallelism=[2, 2, 1, 1],  # data=2, tensor=2
        #     dcn_parallelism=[1, 1, 1, 1]
        # )
        
        # 当前配置: 纯TP
        self.mesh = create_device_mesh(
            ici_parallelism=[1, 4, 1, 1],
            dcn_parallelism=[1, 1, 1, 1]
        )
        
        # DP配置选择
        self.use_data_parallel = os.environ.get("USE_DATA_PARALLEL", "0") == "1"
        self.dp_size = int(os.environ.get("DP_SIZE", "1"))
        
        if self.use_data_parallel and self.dp_size > 1:
            print(f"🔄 启用数据并行，DP size: {self.dp_size}")
            # 重新配置mesh支持DP
            total_devices = len(devices)
            tp_size = total_devices // self.dp_size
            self.mesh = create_device_mesh(
                ici_parallelism=[self.dp_size, tp_size, 1, 1],
                dcn_parallelism=[1, 1, 1, 1]
            )
            print(f"📐 Mesh配置: data={self.dp_size}, tensor={tp_size}")
        
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

    def _create_dp_batch_from_texts(self, model_config, texts, tokenizer):
        """创建支持数据并行的batch（外置DP版本）
        
        Args:
            texts: List[str] 输入文本
            tokenizer: tokenizer
            
        Returns:
            tuple: (sharded_batches, global_batch_info)
        """
        if not self.use_data_parallel or self.dp_size <= 1:
            # 如果不使用DP，回退到常规方法
            result = self._create_batch_from_texts(model_config, texts, tokenizer)
            return result, {"total_texts": len(texts), "texts_per_device": len(texts)}
        
        print(f"🔄 创建DP batch，DP size: {self.dp_size}")
        
        # 确保batch size能被DP size整除
        total_texts = len(texts)
        if total_texts % self.dp_size != 0:
            # 填充到能整除的数量
            padding_needed = self.dp_size - (total_texts % self.dp_size)
            texts = texts + [""] * padding_needed
            print(f"📝 填充 {padding_needed} 个空文本，总数: {len(texts)}")
        
        # 分割文本到各个DP设备
        texts_per_device = len(texts) // self.dp_size
        device_text_groups = []
        for i in range(self.dp_size):
            start_idx = i * texts_per_device
            end_idx = start_idx + texts_per_device
            device_text_groups.append(texts[start_idx:end_idx])
        
        # 为每个设备创建独立的batch
        device_batches = []
        for device_texts in device_text_groups:
            input_ids, seq_lens, forward_batch = self._create_batch_from_texts(
                model_config, device_texts, tokenizer)
            device_batches.append((input_ids, seq_lens, forward_batch))
        
        print(f"📊 创建了 {len(device_batches)} 个设备batch")
        return device_batches, {"total_texts": total_texts, "texts_per_device": texts_per_device}

    def _dp_forward_wrapper(self, model, batches_info):
        """外置数据并行的forward包装器（模仿vLLM风格）
        
        Args:
            model: JAX模型
            batches_info: 各设备的batch信息
            
        Returns:
            各设备的输出结果
        """
        device_batches, global_info = batches_info
        
        if not self.use_data_parallel or self.dp_size <= 1:
            # 单设备模式
            input_ids, _, forward_batch = device_batches
            return model(input_ids, forward_batch.positions, forward_batch)
        
        print("🚀 执行数据并行forward...")
        
        # 使用JAX的pmap实现真正的数据并行
        from jax import pmap
        
        def single_device_forward(input_ids, positions, seq_lens, cache_loc, out_cache_loc, extend_start_loc, forward_mode, batch_size):
            """单个设备的forward函数，用于pmap"""
            # 在pmap内部重新组装ForwardBatch
            forward_batch = ForwardBatch(
                forward_mode=forward_mode,
                batch_size=batch_size,
                input_ids=input_ids,
                seq_lens=seq_lens,
                cache_loc=cache_loc,
                out_cache_loc=out_cache_loc,
                positions=positions,
                extend_start_loc=extend_start_loc,
                token_to_kv_pool=None  # 暂时设为None，或者用全局变量
            )
            return model(input_ids, positions, forward_batch)
        
        # 提取所有数据并padding到相同形状
        all_input_ids = []
        all_positions = []
        all_seq_lens = []
        all_cache_loc = []
        all_out_cache_loc = []
        all_extend_start_loc = []
        all_forward_mode = []
        all_batch_size = []
        
        max_tokens = max(batch[0].shape[0] for batch in device_batches)
        max_batch_size = max(len(batch[2].seq_lens) for batch in device_batches)
        
        for device_idx, (input_ids, seq_lens, forward_batch) in enumerate(device_batches):
            # Padding input_ids和positions
            current_tokens = input_ids.shape[0]
            padding_needed = max_tokens - current_tokens
            
            padded_input_ids = jnp.pad(input_ids, (0, padding_needed), mode='constant', constant_values=0)
            padded_positions = jnp.pad(forward_batch.positions, (0, padding_needed), mode='constant', constant_values=0)
            padded_cache_loc = jnp.pad(forward_batch.cache_loc, (0, padding_needed), mode='constant', constant_values=0)
            
            # Padding seq_lens和extend_start_loc
            current_batch_size = len(seq_lens)
            padding_needed_batch = max_batch_size - current_batch_size
            
            padded_seq_lens = jnp.pad(seq_lens, (0, padding_needed_batch), mode='constant', constant_values=0)
            padded_extend_start_loc = jnp.pad(forward_batch.extend_start_loc, (0, padding_needed_batch), mode='constant', constant_values=0)
            
            # 处理out_cache_loc（可能为None）
            if forward_batch.out_cache_loc is not None:
                padded_out_cache_loc = jnp.pad(forward_batch.out_cache_loc, (0, padding_needed), mode='constant', constant_values=0)
            else:
                padded_out_cache_loc = jnp.zeros(max_tokens, dtype=jnp.int32)
            
            all_input_ids.append(padded_input_ids)
            all_positions.append(padded_positions)
            all_seq_lens.append(padded_seq_lens)
            all_cache_loc.append(padded_cache_loc)
            all_out_cache_loc.append(padded_out_cache_loc)
            all_extend_start_loc.append(padded_extend_start_loc)
            all_forward_mode.append(forward_batch.forward_mode)
            all_batch_size.append(forward_batch.batch_size)
            
            print(f"  设备 {device_idx}: 处理 {input_ids.shape} tokens (padding到 {max_tokens})")
        
        # 堆叠成数组
        stacked_input_ids = jnp.stack(all_input_ids)
        stacked_positions = jnp.stack(all_positions)
        stacked_seq_lens = jnp.stack(all_seq_lens)
        stacked_cache_loc = jnp.stack(all_cache_loc)
        stacked_out_cache_loc = jnp.stack(all_out_cache_loc)
        stacked_extend_start_loc = jnp.stack(all_extend_start_loc)
        stacked_forward_mode = jnp.array(all_forward_mode)
        stacked_batch_size = jnp.array(all_batch_size)
        
        # 使用pmap并行执行 - 这是真正的并行，不使用for循环
        with self.mesh:
            results = pmap(single_device_forward, devices=self.mesh.local_devices[:self.dp_size])(
                stacked_input_ids,
                stacked_positions,
                stacked_seq_lens,
                stacked_cache_loc,
                stacked_out_cache_loc,
                stacked_extend_start_loc,
                stacked_forward_mode,
                stacked_batch_size
            )
        
        print("✅ 数据并行forward完成")
        print(f"   返回结果类型: {type(results)}, 长度: {len(results)}")
        return results

    def _dp_sampling_wrapper(self, sampler, model_outputs, sampling_info, global_info):
        """外置数据并行的采样包装器
        
        Args:
            sampler: 采样器
            model_outputs: 模型输出（分片的）
            sampling_info: 采样参数
            global_info: 全局batch信息
            
        Returns:
            采样结果（分片的）
        """
        if not self.use_data_parallel or self.dp_size <= 1:
            # 单设备采样
            return sampler(model_outputs, sampling_info)
        
        print("🎲 执行数据并行采样...")
        
        # 为每个设备创建采样参数
        texts_per_device = global_info["texts_per_device"]
        
        # 使用JAX的pmap实现真正的数据并行采样
        from jax import pmap
        
        def single_device_sampling(device_output, temperatures, top_ps, top_ks, min_ps, vocab_size):
            """单个设备的采样函数，用于pmap"""
            # 在pmap内部重新组装SamplingBatchInfo
            device_sampling_info = SamplingBatchInfo(
                temperatures=temperatures,
                top_ps=top_ps,
                top_ks=top_ks,
                min_ps=min_ps,
                vocab_size=vocab_size,
            )
            return sampler(device_output, device_sampling_info)
        
        # 提取所有数据并padding到相同形状
        all_device_outputs = []
        all_temperatures = []
        all_top_ps = []
        all_top_ks = []
        all_min_ps = []
        all_vocab_size = []
        
        # 找到最大的输出形状
        max_output_shape = max(output.shape for output in model_outputs)
        
        for device_idx, device_output in enumerate(model_outputs):
            # Padding device_output到相同形状
            current_shape = device_output.shape
            padding_needed = [max_output_shape[i] - current_shape[i] for i in range(len(current_shape))]
            
            if any(p > 0 for p in padding_needed):
                padded_output = jnp.pad(device_output, [(0, p) for p in padding_needed], mode='constant', constant_values=0)
            else:
                padded_output = device_output
            
            # 创建采样参数
            temperatures = jnp.full((texts_per_device, 1), 1.0)
            top_ps = jnp.full((texts_per_device, 1), 1.0)
            top_ks = jnp.ones((texts_per_device, 1))
            min_ps = jnp.full((texts_per_device, 1), 0.0)
            vocab_size = sampling_info.vocab_size
            
            all_device_outputs.append(padded_output)
            all_temperatures.append(temperatures)
            all_top_ps.append(top_ps)
            all_top_ks.append(top_ks)
            all_min_ps.append(min_ps)
            all_vocab_size.append(vocab_size)
            
            print(f"  设备 {device_idx}: 采样 {texts_per_device} 个序列")
        
        # 堆叠成数组
        stacked_device_outputs = jnp.stack(all_device_outputs)
        stacked_temperatures = jnp.stack(all_temperatures)
        stacked_top_ps = jnp.stack(all_top_ps)
        stacked_top_ks = jnp.stack(all_top_ks)
        stacked_min_ps = jnp.stack(all_min_ps)
        stacked_vocab_size = jnp.array(all_vocab_size)
        
        # 使用pmap并行执行 - 这是真正的并行，不使用for循环
        with self.mesh:
            sampling_results = pmap(single_device_sampling, devices=self.mesh.local_devices[:self.dp_size])(
                stacked_device_outputs,
                stacked_temperatures,
                stacked_top_ps,
                stacked_top_ks,
                stacked_min_ps,
                stacked_vocab_size
            )
        
        print("✅ 数据并行采样完成")
        print(f"   返回结果类型: {type(sampling_results)}, 长度: {len(sampling_results)}")
        return sampling_results

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
                    "2+2=?",
                    "Hello world",
                    "What is AI?",
                ]

                print(f"🎯 Input texts: {input_texts}")
                print(f"🔧 Use data parallel: {self.use_data_parallel}")
                print(f"📊 DP size: {self.dp_size}")

                # 使用DP版本的batch创建
                batches_info, global_info = self._create_dp_batch_from_texts(
                    model.config, input_texts, tokenizer)

                if self.use_data_parallel and self.dp_size > 1:
                    print(f"📋 DP模式 - 设备batch数量: {len(batches_info)}")
                    # 初始化每个设备的完整序列历史
                    all_complete_sequences = []
                    all_actual_seq_lens = []
                    
                    for device_idx, (input_ids_array, actual_seq_lens, forward_batch) in enumerate(batches_info):
                        print(f"设备 {device_idx}: batch_size={len(actual_seq_lens)}, tokens={input_ids_array.shape}")
                        
                        device_complete_sequences = []
                        for batch_idx in range(len(actual_seq_lens)):
                            start_idx = sum(actual_seq_lens[:batch_idx])
                            end_idx = start_idx + actual_seq_lens[batch_idx]
                            initial_tokens = [int(token) for token in input_ids_array[start_idx:end_idx]]
                            device_complete_sequences.append(initial_tokens)
                        
                        all_complete_sequences.append(device_complete_sequences)
                        all_actual_seq_lens.append(actual_seq_lens)
                else:
                    # 单设备模式
                    input_ids_array, actual_seq_lens, forward_batch = batches_info
                    complete_sequences = []
                    for batch_idx in range(len(actual_seq_lens)):
                        start_idx = sum(actual_seq_lens[:batch_idx])
                        end_idx = start_idx + actual_seq_lens[batch_idx]
                        initial_tokens = [int(token) for token in input_ids_array[start_idx:end_idx]]
                        complete_sequences.append(initial_tokens)

                jax_profiling_dir = os.environ.get("JAX_TRACE_PROFILING_DIR", "/tmp/jax_profiling")
                with self.mesh:
                    for i in range(5):  # 减少迭代次数
                        print(f"\n🔄 生成步骤 {i+1}")
                        
                        # 使用DP forward包装器
                        y = self._dp_forward_wrapper(model, (batches_info, global_info))

                        if self.use_data_parallel and self.dp_size > 1:
                            # DP模式下的采样
                            total_texts = global_info["total_texts"]
                            texts_per_device = global_info["texts_per_device"]
                            
                            # 创建采样参数（为所有设备，这里简化）
                            sampling_info = SamplingBatchInfo(
                                temperatures=jnp.full((texts_per_device, 1), 1.0),
                                top_ps=jnp.full((texts_per_device, 1), 1.0),
                                top_ks=jnp.ones((texts_per_device, 1)),
                                min_ps=jnp.full((texts_per_device, 1), 0.0),
                                vocab_size=model.config.vocab_size,
                            )
                            
                            # 执行DP采样
                            next_token_ids_list = self._dp_sampling_wrapper(
                                sampler, y, sampling_info, global_info)
                            
                            # 更新每个设备的序列
                            new_batches_info = []
                            for device_idx, (device_batch_info, device_next_tokens) in enumerate(
                                zip(batches_info, next_token_ids_list)):
                                
                                input_ids_array, actual_seq_lens, forward_batch = device_batch_info
                                device_complete_sequences = all_complete_sequences[device_idx]
                                
                                print(f"  设备 {device_idx}: 更新 {len(device_complete_sequences)} 个序列")
                                updated_sequences = self.update_forward_batch_with_sequences(
                                    forward_batch, device_next_tokens, tokenizer, device_complete_sequences)
                                all_complete_sequences[device_idx] = updated_sequences
                                
                                new_batches_info.append((
                                    forward_batch.input_ids, actual_seq_lens, forward_batch))
                            
                            batches_info = new_batches_info
                            
                        else:
                            # 单设备模式
                            sampling_info = SamplingBatchInfo(
                                temperatures=jnp.full((len(complete_sequences), 1), 1.0),
                                top_ps=jnp.full((len(complete_sequences), 1), 1.0),
                                top_ks=jnp.ones((len(complete_sequences), 1)),
                                min_ps=jnp.full((len(complete_sequences), 1), 0.0),
                                vocab_size=model.config.vocab_size,
                            )
                            
                            next_token_ids = sampler(y, sampling_info)
                            complete_sequences = self.update_forward_batch_with_sequences(
                                forward_batch, next_token_ids, tokenizer, complete_sequences)

                # 输出结果
                print(f"\n=== Qwen3 MoE DP Complete Generation Results ===")
                print(f"🔧 使用数据并行: {self.use_data_parallel}")
                
                if self.use_data_parallel and self.dp_size > 1:
                    # DP模式结果展示
                    for device_idx, device_sequences in enumerate(all_complete_sequences):
                        print(f"\n📱 设备 {device_idx} 结果:")
                        for batch_idx, full_sequence in enumerate(device_sequences):
                            global_batch_idx = device_idx * len(device_sequences) + batch_idx
                            if global_batch_idx < len(input_texts):  # 排除填充的空文本
                                original_text = input_texts[global_batch_idx]
                                decoded_full = tokenizer.decode(full_sequence)
                                print(f"  序列 {batch_idx} (全局 {global_batch_idx}): '{original_text}' -> '{decoded_full}'")
                else:
                    # 单设备模式结果展示
                    for batch_idx, full_sequence in enumerate(complete_sequences):
                        original_text = input_texts[batch_idx]
                        decoded_full = tokenizer.decode(full_sequence)
                        print(f"  序列 {batch_idx}: '{original_text}' -> '{decoded_full}'")

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

    def test_dp_functionality_basic(self):
        """基础的数据并行功能测试（不需要真实模型）"""
        print("\n🧪 测试数据并行基础功能...")
        
        # 模拟设置
        self.use_data_parallel = True
        self.dp_size = 2
        
        # 模拟数据
        test_texts = ["Hello", "World", "AI", "Test"]
        
        # 测试DP batch创建
        mock_model_config = type('MockConfig', (), {
            'vocab_size': 1000,
            'num_key_value_heads': 4,
            'head_dim': 64,
            'num_hidden_layers': 2,
            'torch_dtype': 'bfloat16'
        })()
        
        try:
            batches_info, global_info = self._create_dp_batch_from_texts(
                mock_model_config, test_texts, self.tokenizer)
            
            print(f"✅ DP batch创建成功")
            print(f"   设备数量: {len(batches_info)}")
            print(f"   全局信息: {global_info}")
            
            # 验证数据结构
            for device_idx, (input_ids, seq_lens, forward_batch) in enumerate(batches_info):
                print(f"   设备 {device_idx}: {input_ids.shape}, batch_size={len(seq_lens)}")
            
            print("🎉 数据并行基础功能测试通过！")
            
        except Exception as e:
            print(f"❌ DP功能测试失败: {e}")
            raise

if __name__ == '__main__':
    unittest.main()