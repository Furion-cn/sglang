import os
import unittest
from pathlib import Path
from typing import List,Any
from unittest.mock import patch

import jax.numpy as jnp
from flax import nnx

# from jax_smi import initialise_tracking
from transformers import AutoTokenizer

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.jax.layers.sampler import Sampler
from sglang.srt.jax.mem_cache.hash_kvcache import ReqToHashKVCachePool,create_kv_cache
from sglang.srt.jax.model_executor.forward_batch_info import ForwardBatch, ForwardMode,FORWARD_MODE_EXTEND,FORWARD_MODE_DECODE
from sglang.srt.jax.models.qwen import QWenLMHeadJaxModel
from sglang.srt.jax.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.model_loader.loader import JAXModelLoader
from sglang.test.jax.test_utils import create_device_mesh, jax_trace_context
import jax
from functools import partial
from sglang.debug_tracer import global_tracer

MAX_SEQ_LEN=128
MAX_BATCH_SIZE=20

class TestQwenModel(unittest.TestCase):
    """Test cases for the Qwen model."""

    def setUp(self):
        self.mesh = create_device_mesh(
            ici_parallelism=[-1, 1, 1, 1], dcn_parallelism=[1, 1, 1, 1])
        # note: please do not remove it because mesh will be gained in shard_map in Attention for pallas.
        jax.sharding.set_mesh(self.mesh)
        # Model path for local model and tokenizer
        self.test_model_path = os.environ.get(
            'MODEL_PATH', 'Qwen/Qwen-7B')  # Default to HuggingFace

        # JAX loader configuration
        self.load_config = LoadConfig(load_format=LoadFormat.JAX)
        self.device_config = DeviceConfig()
        self.jax_loader = JAXModelLoader(self.load_config)
        # initialise_tracking()

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

        self.enable_debug_tracer = os.environ.get(
            'ENABLE_DEBUG_TRACER', False)

        # Load the model using JAXModelLoader
        with patch('sglang.srt.model_loader.loader.get_model_architecture') as mock_arch:
            mock_arch.return_value = (QWenLMHeadJaxModel, None)

            model = self.jax_loader.load_model(
                model_config=model_config,
                device_config=self.device_config,
                mesh=self.mesh,
                max_seq_len=MAX_SEQ_LEN,
                max_batch_size=MAX_BATCH_SIZE,
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
        invalid_positions_flat=[]
        for tokens in tokenized_inputs:
            input_ids_flat.extend(tokens)
            # Create positions at the same time
            positions_flat.extend(range(len(tokens)))
            invalid_positions_flat.extend(range(len(tokens),MAX_SEQ_LEN,1))
        
        # Pad -inf at the end of inputs and positions, TODO

        # Create required arrays
        valid_input_ids_array = jnp.array(input_ids_flat, dtype=jnp.int32)
        invalid_input_ids_array = jnp.array([-1]*int(MAX_SEQ_LEN*len(texts)-valid_input_ids_array.size))
        input_ids_array = jnp.concat([valid_input_ids_array,invalid_input_ids_array],axis=0)
        valid_positions_array = jnp.array(positions_flat, dtype=jnp.int32)
        invalid_positions_array =  jnp.array(invalid_positions_flat,dtype=jnp.int32)
        positions_array = jnp.concat([valid_positions_array,invalid_positions_array],axis=0)
        seq_lens = jnp.array(actual_seq_lens, dtype=jnp.int32)

        # Create start locations
        extend_start_loc = jnp.cumsum(
            jnp.concatenate([jnp.array([0]), seq_lens[:-1]]))

        k_cache,v_cache=create_kv_cache(
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=MAX_BATCH_SIZE,
            head_num=model_config.num_attention_heads,
            head_dim=model_config.hidden_size // model_config.num_attention_heads,
            layer_num=model_config.num_hidden_layers,
            dtype=jnp.bfloat16 if model_config.bf16 else jnp.float32,
        )

        # Create ForwardBatch
        valid_cache_loc=jnp.arange(jnp.sum(seq_lens), dtype=jnp.int32)
        invalid_cache_loc=jnp.array([i for i in range(jnp.sum(seq_lens),MAX_SEQ_LEN*len(texts))])
        cache_loc =jnp.concat([valid_cache_loc,invalid_cache_loc],axis=0)
        forward_batch = ForwardBatch(
            batch_size=len(actual_seq_lens),
            input_ids=input_ids_array,
            seq_lens=seq_lens,
            positions=positions_array,
            cache_loc=cache_loc,
            out_cache_loc=None,
            extend_start_loc=extend_start_loc,
            k_cache=k_cache,
            v_cache=v_cache,
        )

        return input_ids_array, actual_seq_lens, forward_batch

    def _is_finished(self, token_id, tokenizer):
        """Check if a token indicates the end of generation"""
        return (token_id == tokenizer.eos_token_id or
                token_id == 151643 or  # Common stop token
                token_id == 151645)    # Another common stop token

    def _generate_random_questions(self, batch_size: int) -> list[str]:
        """
        生成指定数量的随机问题，用于批量性能测试

        Args:
            batch_size: 需要生成的问题数量

        Returns:
            包含随机问题的列表
        """
        import random

        # 预定义的问题模板
        question_templates = [
            # 数学问题
            "What is {} + {}?",
            "Calculate {} * {} =",
            "Solve {} - {} =",
            "What is {} divided by {}?",

            # 知识问答
            "The capital of {} is",
            "What is the population of {}?",
            "Tell me about the history of {}",
            "What language is spoken in {}?",

            # 科学问题
            "Explain the concept of {}",
            "What is the formula for {}?",
            "How does {} work?",
            "What are the properties of {}?",

            # 编程问题
            "Write a {} function in Python",
            "How to implement {} algorithm?",
            "What is {} in programming?",
            "Explain {} design pattern",

            # 哲学问题
            "What is the meaning of {}?",
            "Discuss the philosophy of {}",
            "What are the ethics of {}?",
            "How does {} affect society?",

            # 中文问题
            "请解释{}的含义",
            "{}有什么作用？",
            "如何理解{}这个概念？",
            "{}的历史背景是什么？",
            "描述一下{}的特点",

            # 简单对话
            "Hello, how are you",
            "What's your favorite {}?",
            "Can you help me with {}?",
            "I want to learn about {}",
            "Please tell me about {}",
        ]

        # 填充词汇
        fill_words = [
            # 国家/城市
            "France", "China", "Japan", "Germany", "Brazil", "India", "Australia", "Canada",
            "Beijing", "Tokyo", "London", "Paris", "New York", "Sydney", "Berlin", "Moscow",

            # 科学概念
            "gravity", "photosynthesis", "evolution", "quantum mechanics", "relativity",
            "DNA", "atoms", "molecules", "electricity", "magnetism", "thermodynamics",

            # 编程概念
            "recursion", "inheritance", "polymorphism", "encapsulation", "algorithm",
            "database", "machine learning", "artificial intelligence", "blockchain",

            # 抽象概念
            "happiness", "justice", "freedom", "love", "truth", "beauty", "wisdom",
            "technology", "progress", "innovation", "creativity", "sustainability",

            # 中文概念
            "道德", "智慧", "仁义", "礼仪", "文化", "传统", "和谐", "平衡",
            "自然", "科学", "技术", "教育", "艺术", "音乐", "文学", "历史",

            # 随机数字
            "5", "10", "25", "100", "1000", "2024", "42", "365", "7", "12",
        ]

        questions = []
        for i in range(batch_size):
            # 随机选择模板
            template = random.choice(question_templates)

            # 根据模板填充内容
            if "{}" in template:
                # 计算需要填充的参数数量
                param_count = template.count("{}")
                fill_params = random.sample(
                    fill_words, min(param_count, len(fill_words)))

                try:
                    question = template.format(*fill_params)
                except (IndexError, ValueError):
                    # 如果格式化失败，使用简单的模板
                    question = f"Question {i+1}: Tell me about {random.choice(fill_words)}"
            else:
                question = template

            questions.append(question)

        return questions

    def _update_forward_batch(self, forward_batch: ForwardBatch, next_token_ids, tokenizer, finished_requests, original_indices):
        """Update forward batch while handling finished requests"""
        new_input_ids = []
        new_seq_lens = []
        new_original_indices = []
        new_cache_loc = []

        cache_loc_start_loc = 0
        for batch_idx, seq_len in enumerate(forward_batch.seq_lens):
            orig_idx = original_indices[batch_idx]
            current_token_id = int(next_token_ids[batch_idx, 0])
            cache_loc = forward_batch.cache_loc[cache_loc_start_loc:
                                                cache_loc_start_loc + seq_len].tolist()
            cache_loc_start_loc += seq_len

            # Check if this request should finish BEFORE updating sequences
            if self._is_finished(current_token_id, tokenizer):
                print(
                    f"🛑 Request {orig_idx} will be removed from batch (token: {current_token_id})")
                finished_requests.add(orig_idx)
                continue

            # Only update sequences for non-finished requests
            new_input_ids.append(current_token_id)
            new_seq_lens.append(seq_len + 1)
            new_original_indices.append(orig_idx)
            new_cache_loc.append(cache_loc)

        if len(new_seq_lens) == 0:
            # All requests are finished
            return None

        # Update batch with only unfinished requests
        forward_batch.batch_size = len(new_seq_lens)
        forward_batch.seq_lens = jnp.array(new_seq_lens, dtype=jnp.int32)

        # update cache loc
        out_cache_start_loc = max(
            item for sublist in new_cache_loc for item in sublist) + 1
        forward_batch.out_cache_loc =  jnp.arange(out_cache_start_loc, out_cache_start_loc + forward_batch.batch_size, dtype=jnp.int32)
        valid_cache_loc=jnp.array([
            item for i, cache_loc in enumerate(new_cache_loc)
            for item in cache_loc + [int(forward_batch.out_cache_loc[i])]
        ], dtype=jnp.int32)
        invalid_cache_loc=jnp.array([i for i in range(valid_cache_loc.size,MAX_SEQ_LEN*forward_batch.batch_size)])
        forward_batch.cache_loc = jnp.concat([valid_cache_loc,invalid_cache_loc],axis=0)

        # Update positions for decode mode
        forward_batch.positions = jnp.array([seq_len - 1 for seq_len in new_seq_lens], dtype=jnp.int32)

        # Update input ids
        forward_batch.input_ids = jnp.array(new_input_ids, dtype=jnp.int32)

        # Update extend start loc
        forward_batch.extend_start_loc = jnp.cumsum(
            jnp.concatenate([jnp.array([0]), forward_batch.seq_lens[:-1]]))

        return new_original_indices

    def test_qwen_model_decode(self, batch_size: int = None):
        """
        Test Qwen model generation with configurable batch size

        Args:
            batch_size: Number of questions to generate for testing.
                       If None, uses the default small set.
        """
        print("🧪 Testing Qwen model generation...")
        model = self._setup_model()
        if self.enable_debug_tracer:
            global_tracer.start_session()
        jax_profiling_dir = os.environ.get(
            "JAX_TRACE_PROFILING_DIR", "/tmp/jax_profiling")
        batch_size = int(os.environ.get("BATCH_SIZE", 10))
        with self.mesh, jax_trace_context(jax_profiling_dir):
            sampler = Sampler(rngs=nnx.Rngs(0))
            tokenizer = self._get_tokenizer()

            # Generate input texts based on batch_size
            input_texts = self._generate_random_questions(batch_size)
            print(
                f"\n🚀 Generated {batch_size} random questions for batch testing")

            print(f"\n📊 Batch Configuration:")
            print(f"   Total requests: {len(input_texts)}")
            print(f"   Batch size: {len(input_texts)}")

            # 打印前几个问题作为示例
            print(f"\n📝 Sample questions:")
            for i, text in enumerate(input_texts[:min(5, len(input_texts))]):
                print(f"   {i+1}: '{text}'")
            if len(input_texts) > 5:
                print(f"   ... and {len(input_texts) - 5} more questions")

            # 开始计时
            import time
            start_time = time.time()


            input_ids_array, actual_seq_lens, forward_batch = self._create_batch_from_texts(
                model.config, input_texts, tokenizer)

            print(f"\n⚙️  Batch Processing Info:")
            print(f"   Input tokens shape: {input_ids_array.shape}")
            print(
                f"   Actual sequence lengths: {actual_seq_lens[:10]}{'...' if len(actual_seq_lens) > 10 else ''}")
            print(f"   Model vocab size: {model.config.vocab_size}")
            print(f"   Tokenizer EOS token ID: {tokenizer.eos_token_id}")

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

            max_iterations = 15 if batch_size and batch_size > 10 else 30
            print(
                f"\n🔄 Starting generation (max {max_iterations} iterations)...")

            # Forward pass
            # note: donate_argnums is necessary because 'jaxlib._jax.XlaRuntimeError: RESOURCE_EXHAUSTED' will meet without it.
            @nnx.jit(static_argnums=(2,),donate_argnums=(1,))
            def _forward_extend(model:QWenLMHeadJaxModel,forward_batch,batch_size):
                return model(forward_batch.input_ids,
                      forward_batch.positions, forward_batch,FORWARD_MODE_EXTEND,batch_size)

            @nnx.jit(static_argnums=(2,),donate_argnums=(1,))
            def _forward_decode(model:QWenLMHeadJaxModel,forward_batch,batch_size):
                return model(forward_batch.input_ids,
                     forward_batch.positions, forward_batch,FORWARD_MODE_DECODE,batch_size)

            for iteration in range(max_iterations):
                if forward_batch is None:
                    print(
                        f"\n🏁 All requests finished at iteration {iteration}!")
                    break

                if iteration % 5 == 0 or len(input_texts) <= 10:  # 减少大批量时的输出
                    print(f"--- Iteration {iteration + 1} ---")
                    print(f"Active requests: {forward_batch.batch_size}")

                if iteration==0:
                    y,forward_batch = _forward_extend(model,forward_batch,forward_batch.batch_size)
                else:
                    y,forward_batch = _forward_decode(model,forward_batch,forward_batch.batch_size)
                
                # Sample next token for each active sequence
                next_token_ids = sampler(
                    y,
                    sampling_info=SamplingBatchInfo(
                        is_all_greedy=True,
                        temperatures=jnp.full(
                            (forward_batch.batch_size, 1), 1.0),
                        top_ps=jnp.full((forward_batch.batch_size, 1), 1.0),
                        top_ks=jnp.ones((forward_batch.batch_size, 1)),
                        min_ps=jnp.full((forward_batch.batch_size, 1), 0.0),
                        vocab_size=model.config.vocab_size,
                    ))

                # 只为小批量打印详细信息
                if len(input_texts) <= 10 and (iteration % 5 == 0):
                    print(f"Generated tokens: {next_token_ids.tolist()}")

                for batch_idx, token_id in enumerate(next_token_ids):
                    decoded_token = tokenizer.decode(
                        int(token_id[0]), skip_special_tokens=False)
                    final_results[original_indices[batch_idx]
                                  ]['output'] += decoded_token

                    # 只为小批量打印详细信息
                    if len(input_texts) <= 10 and (iteration % 5 == 0):
                        print(
                            f"Request {original_indices[batch_idx]} (batch_idx {batch_idx}): token_id={token_id[0]}, decoded={decoded_token}")


                new_original_indices = self._update_forward_batch(
                    forward_batch, next_token_ids, tokenizer, finished_requests, original_indices)

                if new_original_indices is not None:
                    original_indices = new_original_indices
                else:
                    forward_batch = None  # All requests finished

                # Handle newly finished requests (只为小批量打印)
                if len(input_texts) <= 10:
                    for orig_idx in range(len(input_texts)):
                        if orig_idx in finished_requests and not final_results[orig_idx]['finished']:
                            final_results[orig_idx]['finished'] = True
                            print(f"✅ Request {orig_idx} completed!")
                else:
                    # 批量更新 finished 状态
                    for orig_idx in range(len(input_texts)):
                        if orig_idx in finished_requests:
                            final_results[orig_idx]['finished'] = True

            # 计算总时间
            end_time = time.time()
            total_time = end_time - start_time

            # 统计结果
            finished_count = sum(
                1 for r in final_results.values() if r['finished'])
            avg_output_length = sum(
                len(r['output']) for r in final_results.values()) / len(final_results)

            print(f"\n🎉 === Generation Results Summary ===")
            print(f"📊 Performance Metrics:")
            print(f"   Total time: {total_time:.2f} seconds")
            print(f"   Requests processed: {len(input_texts)}")
            print(f"   Requests finished: {finished_count}/{len(input_texts)}")
            print(
                f"   Average output length: {avg_output_length:.1f} characters")
            print(
                f"   Throughput: {len(input_texts)/total_time:.2f} requests/second")
            print(
                f"   Time per request: {total_time/len(input_texts)*1000:.2f} ms")

            # Print detailed results for small batches
            if len(input_texts) <= 10:
                print(f"\n📝 Detailed Results:")
                for i in range(len(input_texts)):
                    result = final_results[i]
                    status = "✅ Finished" if result['finished'] else "⏰ Max iterations reached"
                    print(f"\nRequest {i} ({status}):")
                    print(f"  Input:  '{result['input']}'")
                    print(f"  Output: '{result['output']}'")
            else:
                # 只显示几个示例
                print(f"\n📝 Sample Results (first 3):")
                for i in range(min(3, len(input_texts))):
                    result = final_results[i]
                    status = "✅ Finished" if result['finished'] else "⏰ Max iterations"
                    print(f"\nRequest {i} ({status}):")
                    print(
                        f"  Input:  '{result['input'][:50]}{'...' if len(result['input']) > 50 else ''}'")
                    print(
                        f"  Output: '{result['output'][:100]}{'...' if len(result['output']) > 100 else ''}'")

            # Verify shapes for the test
            self.assertEqual(len(final_results), len(input_texts))
            for i, result in final_results.items():
                self.assertIsNotNone(result['output'])
                self.assertTrue(len(result['output']) >= len(result['input']))

            print(f"\n✅ Batch test completed successfully!")


            if self.enable_debug_tracer:
                print("\n🔴 Ending debug tracer session...")
                debug_file = global_tracer.end_session()
                if debug_file:
                    print(f"✅ Debug trace saved to: {debug_file}")
                else:
                    print("⚠️  Debug trace not saved")
            return {
                'total_time': total_time,
                'throughput': len(input_texts)/total_time,
                'finished_count': finished_count,
                'total_requests': len(input_texts)
            }

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
