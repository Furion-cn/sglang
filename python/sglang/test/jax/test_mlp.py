import unittest

import jax
from flax import nnx
from jax import numpy as jnp

from sglang.srt.jax.models.qwen import QWenMLP
from jax.sharding import Mesh
import numpy as np
from jax.lax import with_sharding_constraint
import os
from jax.sharding import PartitionSpec

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'

mesh = Mesh(devices=np.array(jax.devices()).reshape(2, -1),
            axis_names=('data', 'tensor'))

hidden_size = 4
intermediate_size = 8
batch_size = 2


class TestMLP(unittest.TestCase):
    def test_mlp(self):
        @nnx.jit
        def create_sharded_model():
            # Unsharded at this moment.
            model = QWenMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
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

        def _ref_mlp(hidden_states, w1, w2, c_proj):
            a1 = w1(hidden_states)
            a2 = w2(hidden_states)
            intermediate_parallel = a1 * jax.nn.silu(a2)
            output = c_proj(intermediate_parallel)
            return output

        with mesh:
            sharded_model = create_sharded_model()
            hidden_states = jnp.ones((batch_size, hidden_size))
            hidden_states = with_sharding_constraint(
                hidden_states, PartitionSpec('data', None))

            # ref output
            ref_output = _ref_mlp(
                hidden_states, sharded_model.w1, sharded_model.w2, sharded_model.c_proj)
            # mlp output
            output = sharded_model(hidden_states)
            # check shape
            assert output.shape == (batch_size, hidden_size)
            # check value correctness
            exactly_equal = jnp.array_equal(output, ref_output)
            assert exactly_equal
            print(f"✓ MLP output is correct!")


if __name__ == '__main__':
    unittest.main()
