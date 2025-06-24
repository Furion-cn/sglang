import os
import flax
from flax import nnx
import numpy as np
import jax
from jax.sharding import Mesh, PartitionSpec
from sglang.srt.jax.models.qwen import QWenMLP
import jax.numpy as jnp


# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'

mesh = Mesh(devices=np.array(jax.devices()).reshape(1, -1),
            axis_names=('data', 'tensor'))

hidden_size = 4096
intermediate_size = 22016
batch_size = 256


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


with jax.profiler.trace("/root/users/aolemila/jax_profile_sglang_qwen/profile", create_perfetto_trace=True), mesh:
    sharded_model = create_sharded_model()
    hidden_states = jnp.ones(
        (batch_size, hidden_size), dtype=jnp.bfloat16)
    hidden_states = jax.lax.with_sharding_constraint(
        hidden_states, PartitionSpec('data', None))

    for i in range(3000):
        # print(f"interation: {i}")
        y = sharded_model(hidden_states=hidden_states)
    # y.block_until_ready()


# jax.profiler.start_server(8877)
# print("profiler server has started")

# with mesh:
#     sharded_model = create_sharded_model()
#     hidden_states = jnp.ones(
#         (batch_size, hidden_size), dtype=jnp.bfloat16)
#     hidden_states = jax.lax.with_sharding_constraint(
#         hidden_states, PartitionSpec('data', None))

#     for i in range(10):
#         y = sharded_model(hidden_states=hidden_states)

#     y.block_until_ready()

# jax.profiler.stop_server()
