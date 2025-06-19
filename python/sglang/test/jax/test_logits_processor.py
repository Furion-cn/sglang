import unittest

import jax
from flax import nnx
from jax import numpy as jnp
from sglang.srt.jax.layers.logits_processor import LogitsProcessor
from jax.sharding import Mesh
import numpy as np
import os

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'


mesh = Mesh(devices=np.array(jax.devices()).reshape(2, -1),
            axis_names=('data', 'model'))


hidden_size = 4
vocab_size = 4
batch_size = 2


class TestLogitsProcessor(unittest.TestCase):
    def test_logits_processor(self):
        def create_sharded_model():
            # Unsharded at this moment.
            model = LogitsProcessor(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
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
            sharded_model = create_sharded_model()
            hidden_states = jnp.ones((batch_size, hidden_size))
            output = sharded_model(hidden_states)
            ref_output = _ref_logits_processor(
                hidden_states, sharded_model.lm_head.kernel.value)

            # check shape
            assert output.logits.shape == (batch_size, vocab_size)
            # check correctness
            assert jnp.array_equal(output.logits, ref_output)
            print(f"✓ Logits processor output is correct!")


if __name__ == '__main__':
    unittest.main()
