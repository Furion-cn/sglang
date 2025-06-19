import unittest

import jax
from flax import nnx
from jax import numpy as jnp

from sglang.srt.jax.layers.logits_processor import LogitsProcessor


class TestLogitsProcessor(unittest.TestCase):
    def test_mlp(self):
        logits_processor = LogitsProcessor(
            hidden_size=4096,
            vocab_size=151936,
            rngs=nnx.Rngs(0)
        )
        hidden_states = jnp.ones((1, 4096))
        output = logits_processor(hidden_states)
        assert output.logits.shape == (1, 151936)


if __name__ == '__main__':
    unittest.main()
