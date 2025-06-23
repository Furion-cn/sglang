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
from sglang.test.jax.test_utils import convert_jax_array_to_torch_tensor

import torch

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'

mesh = Mesh(devices=np.array(jax.devices()).reshape(2, -1),
            axis_names=('data', 'tensor'))

hidden_size = 4
intermediate_size = 4
batch_size = 2


class TestPartitionedMLP(unittest.TestCase):
    def test_partitioned_mlp(self):
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
            hidden_states = jnp.ones(
                (batch_size, hidden_size), dtype=jnp.bfloat16)
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
            print(f"✓ Partitioned MLP output is correct!")


class TorchMLP(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        w1: jax.Array,
        w2: jax.Array,
        c_proj: jax.Array,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        # convert jax.Array to torch.Tensor

        self.w1 = torch.nn.Linear(
            hidden_size, intermediate_size//2, dtype=dtype, bias=False)
        self.w1.weight = torch.nn.Parameter(
            convert_jax_array_to_torch_tensor(w1).T.to(dtype))
        self.w2 = torch.nn.Linear(
            hidden_size, intermediate_size//2, dtype=dtype, bias=False)
        self.w2.weight = torch.nn.Parameter(
            convert_jax_array_to_torch_tensor(w2).T.to(dtype))
        self.c_proj = torch.nn.Linear(
            intermediate_size//2, hidden_size, dtype=dtype, bias=False)
        self.c_proj.weight = torch.nn.Parameter(
            convert_jax_array_to_torch_tensor(c_proj).T.to(dtype))
        self.act_func = torch.nn.SiLU()

    def forward(self, hidden_states: torch.Tensor):
        a1 = self.w1(hidden_states)
        a2 = self.w2(hidden_states)
        intermediate_parallel = a1 * self.act_func(a2)
        output = self.c_proj(intermediate_parallel)
        return output


class TestMLP(unittest.TestCase):
    """
    Test the MLP model with float16 precision.

    Result:
    jax_output: [[ 0.09094 -0.00801  0.1598  -0.1957 ], [ 0.09094 -0.00801  0.1598  -0.1957 ]]
    torch_output: [[ 0.0908, -0.0082,  0.1600, -0.1958], [ 0.0908, -0.0082,  0.1600, -0.1958]]
    note: use float16 precision rather than bfloat16 precision because numpy does not support bfloat16 and numpy is used to convert jax.Array to torch.Tensor.
    """

    def test_mlp(self):
        jax_model = QWenMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            rngs=nnx.Rngs(0),
            dtype=jnp.float16,
        )
        torch_model = TorchMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            w1=jax_model.w1.kernel.value,
            w2=jax_model.w2.kernel.value,
            c_proj=jax_model.c_proj.kernel.value,
            dtype=torch.float16,
        )

        # check output
        jax_hidden_states = jnp.ones(
            (batch_size, hidden_size), dtype=jnp.float16)
        torch_hidden_states = torch.ones(
            batch_size, hidden_size, dtype=torch.float16)
        with mesh:
            jax_output = jax_model(hidden_states=jax_hidden_states)
        torch_output = torch_model(hidden_states=torch_hidden_states)
        assert jax_output.shape == torch_output.shape
        assert torch.allclose(
            convert_jax_array_to_torch_tensor(jax_output), torch_output, atol=1e-3)
        print(f"✓ Torch Jax MLP output is correct!")


if __name__ == '__main__':
    unittest.main()
