from jax.experimental.compilation_cache import compilation_cache as cc
import unittest

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer, PretrainedConfig

from sglang.srt.jax.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.jax.layers.sampler import Sampler
from sglang.srt.jax.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.jax.models.qwen import QWenLMHeadJaxModel
from sglang.srt.jax.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.test.jax.test_utils import create_device_mesh


import os
cache_dir = '/mnt/data/users/aolemila/jax_compile_cache/'
os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
jax.config.update("jax_compilation_cache_dir", cache_dir)
cc.set_cache_dir(cache_dir)


class TestQwenModel(unittest.TestCase):
    """Test cases for the Qwen model."""

    def setUp(self):
        self.mesh = create_device_mesh(
            ici_parallelism=[1, -1, 1, 1], dcn_parallelism=[1, 1, 1, 1])

    @staticmethod
    @nnx.jit
    def _setup_model():
        model = QWenLMHeadJaxModel(config=PretrainedConfig(
            vocab_size=10000,
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=16,
            intermediate_size=4096,
            max_position_embeddings=1024,
            rope_theta=10000,
            layer_norm_epsilon=1e-5,
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

    def test_qwen_model_prefill(self):
        with self.mesh:
            model = self._setup_model()
            x = jax.random.randint(jax.random.PRNGKey(0),
                                   (128, 2), 0, 10000)
            forward_batch = self._create_batch(x)
            y = model(forward_batch.input_ids,
                      forward_batch.positions, forward_batch)
            # Now y is LogitsProcessorOutput with next_token_logits for each sequence
            # Shape: [batch_size, vocab_size] = [128, 10000]
            self.assertEqual(y.next_token_logits.shape, (128, 10048))

    def test_qwen_model_decode(self):
        with self.mesh, jax.profiler.trace("/root/users/aolemila/jax_profile_sglang_qwen/profile"):
            model = self._setup_model()
            sampler = Sampler(rngs=nnx.Rngs(0))
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen-7B", trust_remote_code=True)

            #input_text = "1+1=?"
            batch_size=1024
            input_text_list = ["1+1=?" for _ in range(batch_size)]
            #print(f"input_text_list: {input_text_list}")
            encoded_input=[[tokenizer.encode(input_text)] for input_text in input_text_list]
            #print(f"encoded_input: {encoded_input}")
            x = jnp.array(encoded_input).reshape(batch_size, -1)
            print(f"输入文本: {input_text_list}")
            print(f"输入 tokens: {x}")

            for i in range(3):
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
                decoded_token = tokenizer.decode([current_token_id])
                print(
                    f"Step {i+1}: token_id={current_token_id}, decoded='{decoded_token}'")
            x.block_until_ready()

            full_sequence = [int(token) for token in x[0]]
            decoded_full = tokenizer.decode(full_sequence)
            print(f"\n完整生成序列: {full_sequence}")
            print(f"完整解码文本: '{decoded_full}'")

            # Shape assertions: [batch_size, vocab_size] for next token logits
            # self.assertEqual(y.next_token_logits.shape, (1, 10048))
            # self.assertEqual(x.shape, (1, 14))
