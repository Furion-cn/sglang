import unittest
from sglang.srt.jax.models.qwen import QWenMLP
from jax import numpy as jnp
import jax

class TestQwen(unittest.TestCase):
    def test_qwen_mlp():
        MLP=QWenMLP(
            hidden_size=4096,
            intermediate_size=22016,
            hidden_act="silu",
            quant_config=None,
        )
        hidden_states=jnp.ones((1,4096))
        jax.sharding.with_sharding_constraint(hidden_states,jax.sharding.Mesh(('data',None)))
        output=MLP(hidden_states)
        assert output.shape==(1,4096)