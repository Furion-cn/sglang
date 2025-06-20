import unittest

import jax
from flax import nnx
from jax import numpy as jnp
from sglang.srt.jax.layers.logits_processor import LogitsProcessor
from jax.sharding import Mesh
import numpy as np
import os
from sglang.srt.jax.layers.embeddings import Embed
from transformers import PretrainedConfig

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'


mesh = Mesh(devices=np.array(jax.devices()).reshape(2, -1, 1, 1),
            axis_names=('data', 'model', 'vocab', 'embed'))


hidden_size = 4
vocab_size = 4
batch_size = 2


class TestLogitsProcessor(unittest.TestCase):
    def test_logits_processor(self):
        def create_sharded_lm_head():
            # Unsharded at this moment.
            model = Embed(
                config=PretrainedConfig(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    num_hidden_layers=12,
                    num_attention_heads=16,
                    intermediate_size=4096,
                    max_position_embeddings=1024,
                    rope_theta=10000,
                    layer_norm_epsilon=1e-5,
                    weight_dtype=jnp.float32,
                ),
                num_embeddings=vocab_size,
                features=hidden_size,
                rngs=nnx.Rngs(0)
            )
            # The model's state, a pure pytree.
            state = nnx.state(model)
            # Strip out the annotations from state.
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            # The model is sharded now!
            nnx.update(model, sharded_state)
            return model

        def _ref_logits_processor(hidden_states, weight):
            return jnp.dot(hidden_states, weight, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

        with mesh:
            sharded_lm_head = create_sharded_lm_head()
            hidden_states = jnp.ones((batch_size, hidden_size))
            logits_processor = LogitsProcessor()
            output = logits_processor(hidden_states, sharded_lm_head)
            ref_output = _ref_logits_processor(
                hidden_states, sharded_lm_head.weight.value)

            # check shape
            assert output.logits.shape == (batch_size, vocab_size)
            # check correctness
            assert jnp.allclose(output.logits, ref_output)
            print(f"✓ Logits processor output is correct!")


if __name__ == '__main__':
    unittest.main()
