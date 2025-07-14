import unittest

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer, PretrainedConfig

from sglang.srt.jax.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.jax.layers.sampler import Sampler
from sglang.srt.jax.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.jax.models.qwen3 import Qwen3ForCausalLMJaxModel
from sglang.srt.jax.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.test.jax.test_utils import create_device_mesh


class TestQwen3Model(unittest.TestCase):
    """Test cases for the Qwen3 model."""

    def setUp(self):
        self.mesh = create_device_mesh(
            ici_parallelism=[-1, 1, 1, 1], dcn_parallelism=[1, 1, 1, 1])

    @staticmethod
    @nnx.jit
    def _setup_model():
        model = Qwen3ForCausalLMJaxModel(config=PretrainedConfig(
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151645,
            head_dim=128,
            hidden_act="silu",
            hidden_size=4096,
            initializer_range=0.02,
            intermediate_size=12288,
            max_position_embeddings=40960,
            max_window_layers=36,
            model_type="qwen3",
            num_attention_heads=32,
            num_hidden_layers=36,
            num_key_value_heads=8,
            rms_norm_eps=1e-06,
            rope_scaling=None,
            rope_theta=1000000,
            sliding_window=None,
            tie_word_embeddings=False,
            torch_dtype="bfloat16",
            vocab_size=151936
        ), rngs=nnx.Rngs(0))
        state = nnx.state(model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(model, sharded_state)
        return model

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

    def test_qwen3_model_prefill(self):
        with self.mesh:
            model = self._setup_model()
            x = jax.random.randint(jax.random.PRNGKey(0),
                                   (128, 2), 0, 10000)
            forward_batch = self._create_batch(x)
            y = model(forward_batch.input_ids,
                      forward_batch.positions, forward_batch)
            # Now y is LogitsProcessorOutput with next_token_logits for each sequence
            # Shape: [batch_size, vocab_size] = [128, 10000]
            self.assertEqual(y.next_token_logits.shape, (128, 10000))

    def test_qwen3_model_decode(self):
        with self.mesh:
            model = self._setup_model()
            sampler = Sampler(rngs=nnx.Rngs(0))
            
            # 使用Qwen3的tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen3-8B", trust_remote_code=True)
            except:
                # 如果无法加载真实的tokenizer，使用虚拟的token ids
                print("警告: 无法加载Qwen3 tokenizer，使用虚拟token ids")
                tokenizer = None

            if tokenizer:
                input_text = "1+1=?"
                x = jnp.array(tokenizer.encode(input_text)).reshape(1, -1)
                print(f"输入文本: {input_text}")
                print(f"输入 tokens: {x}")
            else:
                # 使用虚拟的token序列
                x = jnp.array([[1, 2, 3, 4]]) 
                print(f"使用虚拟输入 tokens: {x}")

            for i in range(10):
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
                        temperatures=jnp.full((1, 1), 0.6),
                        top_ps=jnp.full((1, 1), 0.9),
                        top_ks=jnp.ones((1, 1)),
                        min_ps=jnp.full((1, 1), 0.0),
                        vocab_size=10000,
                    ))

                # Update sequence with new token for next iteration
                x = jnp.concatenate([x, next_token_ids], axis=-1)

                current_token_id = int(next_token_ids[0, 0])
                if tokenizer:
                    decoded_token = tokenizer.decode([current_token_id])
                    print(
                        f"Step {i+1}: token_id={current_token_id}, decoded='{decoded_token}'")
                else:
                    print(f"Step {i+1}: token_id={current_token_id}")

            if tokenizer:
                full_sequence = [int(token) for token in x[0]]
                decoded_full = tokenizer.decode(full_sequence)
                print(f"\n完整生成序列: {full_sequence}")
                print(f"完整解码文本: '{decoded_full}'")
            else:
                print(f"\n完整生成序列: {[int(token) for token in x[0]]}")

            # Shape assertions: [batch_size, vocab_size] for next token logits
            self.assertEqual(y.next_token_logits.shape, (1, 10000))
            # 序列长度应该是初始长度 + 生成的10个token
            expected_length = 4 + 10 if not tokenizer else len(jnp.array(tokenizer.encode("1+1=?"))) + 10
            self.assertEqual(x.shape[1], expected_length)

    def test_qwen3_model_config_compatibility(self):
        """测试Qwen3模型配置的兼容性"""
        with self.mesh:
            # 测试不同的配置参数
            configs = [
                # 测试不同的head配置
                {
                    "vocab_size": 10000,
                    "hidden_size": 512,
                    "num_hidden_layers": 8,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 4,  # GQA
                    "intermediate_size": 2048,
                    "max_position_embeddings": 512,
                    "rope_theta": 1000000,
                    "rms_norm_eps": 1e-6,
                    "attention_bias": True,  # 测试带bias的注意力
                },
                # 测试更小的模型
                {
                    "vocab_size": 5000,
                    "hidden_size": 256,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "intermediate_size": 1024,
                    "max_position_embeddings": 256,
                    "rope_theta": 10000,
                    "rms_norm_eps": 1e-5,
                    "attention_bias": False,
                }
            ]
            
            for i, config_dict in enumerate(configs):
                with self.subTest(config=i):
                    model = Qwen3ForCausalLMJaxModel(
                        config=PretrainedConfig(**config_dict),
                        rngs=nnx.Rngs(i)
                    )
                    
                    # 测试前向传播
                    batch_size = 2
                    seq_len = 3
                    x = jax.random.randint(
                        jax.random.PRNGKey(i), 
                        (batch_size, seq_len), 
                        0, config_dict["vocab_size"]
                    )
                    forward_batch = self._create_batch(x)
                    y = model(forward_batch.input_ids,
                             forward_batch.positions, forward_batch)
                    
                    # 验证输出形状
                    expected_shape = (batch_size, config_dict["vocab_size"])
                    self.assertEqual(y.next_token_logits.shape, expected_shape)
                    print(f"✅ Config {i}: 输出形状 {y.next_token_logits.shape} 正确")


if __name__ == "__main__":
    unittest.main()
