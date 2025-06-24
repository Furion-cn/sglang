import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer, PretrainedConfig

from sglang.srt.jax.layers.sampler import Sampler
from sglang.srt.jax.models.qwen import QWenLMHeadModel
from sglang.srt.jax.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.test.jax.test_utils import create_device_mesh
from sglang.test.test_utils import CustomTestCase


mesh = create_device_mesh(
    ici_parallelism=[-1, 1, 1, 1], dcn_parallelism=[1, 1, 1, 1])


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


def _get_positions(x):
    return jnp.concatenate([
        jnp.arange(x.shape[1]) for _ in range(x.shape[0])
    ]).reshape(x.shape[0], x.shape[1])


@nnx.jit
def inference(x):
    sampler = Sampler(rngs=nnx.Rngs(0))
    for i in range(5):
        positions = _get_positions(x)
        y = model(x, positions, None)
        # y.next_token_logits.block_until_ready()
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
        # current_token_id = int(next_token_ids[0, 0])
        # decoded_token = tokenizer.decode([current_token_id])
        # print(
        #     f"Step {i+1}: token_id={current_token_id}, decoded='{decoded_token}'")
    return x


with jax.profiler.trace("/root/users/aolemila/jax_profile_sglang_qwen/profile", create_perfetto_link=True), mesh:
    model = _setup_model()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen-7B", trust_remote_code=True)

    # 初始输入
    input_text = "1+1=?"
    x = jnp.array(tokenizer.encode(input_text)).reshape(1, -1)
    print(f"输入文本: {input_text}")
    print(f"输入 tokens: {x}")

    x = inference(x)

    # 解码完整的生成序列
    full_sequence = [int(token) for token in x[0]]
    decoded_full = tokenizer.decode(full_sequence)
    print(f"\n完整生成序列: {full_sequence}")
    print(f"完整解码文本: '{decoded_full}'")

    import time

    time.sleep(200)
