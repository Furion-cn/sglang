
import unittest

import jax
from flax import nnx
from transformers import PretrainedConfig

from sglang.srt.jax.models.qwen import QWenLMHeadModel
from sglang.test.jax.test_utils import create_device_mesh
from sglang.test.test_utils import CustomTestCase


class TestQwenModel(CustomTestCase):
    """Test cases for the Qwen model."""

    def setUp(self):
        self.mesh = create_device_mesh(
            ici_parallelism=[1], dcn_parallelism=[1])

    @nnx.jit
    def _setup_model(self):
        model = QWenLMHeadModel(config=PretrainedConfig(
            vocab_size=10000,
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=16,
            intermediate_size=4096,
            max_position_embeddings=1024,
            rope_theta=10000,
        ))
        state = nnx.state(model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(model, sharded_state)
        return model

    def test_qwen_model(self):
        with self.mesh:
            model = self._setup_model()
            x = jax.random.normal(jax.random.PRNGKey(0), (1, 1024))
            y = model(x)
            self.assertEqual(y.shape, (1, 10000))
