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
import torch
from sglang.test.jax.test_utils import convert_jax_array_to_torch_tensor
from torch import nn

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'


mesh = Mesh(devices=np.array(jax.devices()).reshape(2, -1, 1, 1),
            axis_names=('data', 'tensor', 'vocab', 'embed'))


hidden_size = 4
vocab_size = 8
batch_size = 2


class TestPartitionedLogitsProcessor(unittest.TestCase):
    def test_partitioned_logits_processor(self):
        def create_sharded_lm_head():
            # Unsharded at this moment.
            model = Embed(
                num_embeddings=vocab_size,
                features=hidden_size,
                dtype=jnp.bfloat16,
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
            return jnp.dot(hidden_states[:, -1:, :], weight, precision=jax.lax.Precision.DEFAULT, preferred_element_type=jnp.bfloat16)

        with mesh:
            sharded_lm_head = create_sharded_lm_head()
            hidden_states = jax.random.randint(
                nnx.Rngs(0).params(), (batch_size, 3, hidden_size), 0, vocab_size)
            logits_processor = LogitsProcessor(vocab_size=vocab_size)
            output = logits_processor(hidden_states, sharded_lm_head)
            ref_output = _ref_logits_processor(
                hidden_states,  jnp.asarray(sharded_lm_head.embedding.value).T)

            # check shape
            assert output.next_token_logits.shape == (
                batch_size, 1, vocab_size)
            # check correctness
            assert jnp.allclose(output.next_token_logits,
                                ref_output, atol=1e-3)
            print(f"✓ Partitioned Logits processor output is correct!")


class TorchLogitsProcessor(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dtype, weight: jax.Array):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings, embedding_dim, dtype=dtype)
        self.embedding.weight = torch.nn.Parameter(
            convert_jax_array_to_torch_tensor(weight).T.to(dtype))

    def forward(self, hidden_states):
        return torch.matmul(hidden_states[:, -1:, :], self.embedding.weight)


class TestLogitsProcessor(unittest.TestCase):
    """
    Test the logits processor.

    return:
    torch_output: [[[ 0.8901,  1.5820, -0.5439, -0.6523, -1.2412,  1.0273,  1.4561,
          -0.7119]],

        [[ 0.8901,  1.5820, -0.5439, -0.6523, -1.2412,  1.0273,  1.4561,
          -0.7119]]]

    jax_output: [[[ 0.8906,  1.58  , -0.547 , -0.6504, -1.242 ,  1.027 ,  1.455 ,
         -0.711 ]],

       [[ 0.8906,  1.58  , -0.547 , -0.6504, -1.242 ,  1.027 ,  1.455 ,
         -0.711 ]]]

    note: use float16 precision rather than bfloat16 precision because numpy does not support bfloat16 and numpy is used to convert jax.Array to torch.Tensor.
    """

    def test_logits_processor(self):
        jax_embedding = Embed(
            num_embeddings=vocab_size,
            features=hidden_size,
            param_dtype=jnp.float16,
            dtype=jnp.float16,
            rngs=nnx.Rngs(0)
        )
        jax_hidden_states = jnp.ones(
            (batch_size, 3, hidden_size), dtype=jnp.float16)
        jax_logits_processor = LogitsProcessor(vocab_size=vocab_size)
        jax_output = jax_logits_processor(jax_hidden_states, jax_embedding)

        torch_hidden_states = torch.ones(
            batch_size, 3, hidden_size, dtype=torch.float16)
        torch_logits_processor = TorchLogitsProcessor(vocab_size, hidden_size, dtype=torch.float16,
                                                      weight=jax_embedding.embedding.value)
        torch_output = torch_logits_processor(torch_hidden_states)
        assert torch.allclose(torch_output, convert_jax_array_to_torch_tensor(
            jax_output.next_token_logits), atol=1e-2)
        print(f"✓ Torch and JaxLogits processor output is correct!")


if __name__ == '__main__':
    unittest.main()
