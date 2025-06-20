import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer, PretrainedConfig

from sglang.srt.jax.layers.sampler import Sampler
from sglang.srt.jax.models.qwen import QWenLMHeadModel
from sglang.srt.jax.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.test.jax.test_utils import create_device_mesh
from sglang.test.test_utils import CustomTestCase


class TestQwenModel(CustomTestCase):
    """Test cases for the Qwen model."""

    def setUp(self):
        self.mesh = create_device_mesh(
            ici_parallelism=[-1, 1, 1, 1], dcn_parallelism=[1, 1, 1, 1])

    @staticmethod
    @nnx.jit
    def _setup_model():
        model = QWenLMHeadModel(config=PretrainedConfig(
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

    def _get_positions(self, x):
        return jnp.concatenate([
            jnp.arange(x.shape[1]) for _ in range(x.shape[0])
        ]).reshape(x.shape[0], x.shape[1])

    def test_qwen_model_prefill(self):
        with self.mesh:
            model = self._setup_model()
            x = jax.random.randint(jax.random.PRNGKey(0),
                                   (128, 2), 0, 10000)
            positions = self._get_positions(x)
            y = model(x, positions, None)
            self.assertEqual(y.logits.shape, (128, 10000))

    def test_qwen_model_decode(self):
        with self.mesh:
            model = self._setup_model()
            sampler = Sampler(rngs=nnx.Rngs(0))
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B")

            # 初始输入
            input_text = "1+1=?"
            x = jnp.array(tokenizer.encode(input_text)).reshape(1, -1)
            print(f"输入文本: {input_text}")
            print(f"输入 tokens: {x}")

            for i in range(10):
                positions = self._get_positions(x)
                y = model(x, positions, None)
                next_token_ids = sampler(
                    y, sampling_info=SamplingBatchInfo(
                        temperatures=jnp.full((1, 1), 0.6),
                        top_ps=jnp.full((1, 1), 0.9),
                        top_ks=jnp.ones((1, 1)),
                        min_ps=jnp.full((1, 1), 0.0),
                        vocab_size=10000,
                    ))
                x = jnp.concatenate(
                    [x, next_token_ids], axis=-1)

                # 解码当前生成的 token
                current_token_id = int(next_token_ids[0, 0])
                decoded_token = tokenizer.decode([current_token_id])
                print(
                    f"Step {i+1}: token_id={current_token_id}, decoded='{decoded_token}'")

            # 解码完整的生成序列
            full_sequence = [int(token) for token in x[0]]
            decoded_full = tokenizer.decode(full_sequence)
            print(f"\n完整生成序列: {full_sequence}")
            print(f"完整解码文本: '{decoded_full}'")

            self.assertEqual(y.next_token_logits.shape, (1, 1, 10048))
            self.assertEqual(x.shape, (1, 14))
