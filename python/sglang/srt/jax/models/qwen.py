from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from transformers import PretrainedConfig

from sglang.srt.jax.layers.attention import Attention
from sglang.srt.jax.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sglang.srt.jax.layers.layernorm import RMSNorm
from sglang.srt.jax.layers.linear import LinearBase
from sglang.srt.jax.layers.logits_processor import LogitsProcessor
from sglang.srt.jax.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.jax.utils import flatten_pytree_with_paths, get_expected_param_paths, update_state_recursive


class QWenMLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        rngs: nnx.Rngs = None,
    ):

        self.w1 = nnx.Linear(
            hidden_size,
            intermediate_size,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), (None, "tensor")),
            use_bias=False,
            rngs=rngs
        )

        self.w2 = nnx.Linear(
            hidden_size,
            intermediate_size,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), ("tensor", None)),
            use_bias=False,
            rngs=rngs
        )

        self.c_proj = nnx.Linear(
            intermediate_size,
            hidden_size,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), ("data", "tensor")),
            use_bias=False,
            rngs=rngs
        )

        self.act_func = jax.nn.silu

    def __call__(self, hidden_states: jnp.ndarray):
        gate = self.w1(hidden_states)
        up = self.w2(hidden_states)
        intermediate_parallel = up * self.act_func(gate)
        intermediate_parallel = jax.lax.with_sharding_constraint(
            intermediate_parallel, PartitionSpec('data', 'tensor'))
        output = self.c_proj(intermediate_parallel)
        return output


class QWenAttention(nnx.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 max_position_embeddings: int,
                 rope_theta: float,
                 rope_scaling: Optional[Dict[str, Any]],
                 rngs: nnx.Rngs = None):
        head_size = hidden_size // num_heads
        self.c_attn = LinearBase(
            input_size=hidden_size,
            output_size=(num_heads + 2 * num_heads) * head_size,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
        )
        self.c_proj = LinearBase(
            input_size=num_heads * head_size,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            rngs=rngs,
        )
        self.rotary_emb = RotaryEmbedding(
            min_timescale=1,
            max_timescale=10000,
            num_heads=num_heads,
            embedding_dims=head_size,
        )
        self.attn = Attention(
            scale=head_size**-0.5,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
    ) -> jax.Array:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = self.rotary_emb(q, positions)
        k = self.rotary_emb(k, positions)
        attn_output = self.attn(q, k, v, is_causal=True)
        output, _ = self.c_proj(attn_output)
        return output


class QWenBlock(nnx.Module):
    def __init__(self,
                 config: PretrainedConfig,
                 rngs: nnx.Rngs = None):
        self.ln_1 = RMSNorm(config.hidden_size,
                            epsilon=config.layer_norm_epsilon,
                            rngs=rngs)

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.attn = QWenAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rngs=rngs,
        )

        self.ln_2 = RMSNorm(config.hidden_size,
                            epsilon=config.layer_norm_epsilon,
                            rngs=rngs)

        self.mlp = QWenMLP(
            config.hidden_size,
            config.intermediate_size // 2,
            rngs=rngs,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
    ) -> jax.Array:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QWenModel(nnx.Module):
    """QWen model"""

    def __init__(self,
                 config: PretrainedConfig,
                 rngs: nnx.Rngs = None):
        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.embed_tokens = Embed(
            num_embeddings=vocab_size,
            features=config.hidden_size,
            rngs=rngs,
        )
        self.h = [
            QWenBlock(
                config,
                rngs=rngs,
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.ln_f = RMSNorm(config.hidden_size,
                            epsilon=config.layer_norm_epsilon,
                            rngs=rngs)

    def __call__(self,
                 input_ids: jax.Array,
                 positions: jax.Array,
                 forward_batch: ForwardBatch,
                 ):
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states = layer(
                positions,
                hidden_states,
                forward_batch,
            )
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class QWenLMHeadModel(nnx.Module):
    """QWen language head model"""

    def __init__(self,
                 config: PretrainedConfig,
                 rngs: nnx.Rngs = None):
        self.config = config
        self.transformer = QWenModel(config, rngs)
        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.lm_head = ParallelLMHead(vocab_size, config.hidden_size, rngs=rngs)
        self.logits_processor = LogitsProcessor(vocab_size)
    
    def load_pytree_weights(self, pytree):        
        flat_weights = flatten_pytree_with_paths(pytree)
        model_state = nnx.state(self)
        expected_paths = get_expected_param_paths(model_state)
        missing_paths = expected_paths - set(flat_weights.keys())
        if missing_paths:
            raise ValueError(f"Missing weights for parameters: {sorted(missing_paths)}")
        
        update_state_recursive(model_state, flat_weights)

        pspecs = nnx.get_partition_spec(model_state)
        pstate = jax.lax.with_sharding_constraint(model_state, pspecs)
        nnx.update(self, pstate)

    def __call__(self,
                 input_ids: jax.Array,
                 positions: jax.Array,
                 forward_batch: ForwardBatch,
                 ):
        hidden_states = self.transformer(input_ids, positions, forward_batch)
        return self.logits_processor(
            hidden_states, self.lm_head
        )

EntryClass = QWenLMHeadModel