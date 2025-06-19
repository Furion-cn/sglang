import unittest
from sglang.srt.jax.models.qwen import QWenMLP
from jax import numpy as jnp
import jax
from flax import nnx


class TestMLP(unittest.TestCase):
    def test_mlp(self):
        mlp = QWenMLP(
            hidden_size=4096,
            intermediate_size=22016,
            rngs=nnx.Rngs(0)
        )
        hidden_states = jnp.ones((1, 4096))
        output = mlp(hidden_states)
        assert output.shape == (1, 4096)


if __name__ == '__main__':
    unittest.main()
