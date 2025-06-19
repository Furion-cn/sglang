from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx
from jax.sharding import PartitionSpec, NamedSharding, Mesh
from jax import numpy as jnp
from jax.lax import with_sharding_constraint
from transformers import PretrainedConfig

from sglang.srt.jax.layers.attention import Attention
from sglang.srt.jax.layers.layernorm import RMSNorm
from sglang.srt.jax.layers.linear import LinearBase, QKVParallelLinear
from sglang.srt.jax.layers.logits_processor import LogitsProcessor
from sglang.srt.jax.layers.quantization.base_config import QuantizationConfig
from sglang.srt.jax.layers.embeddings import RotaryEmbedding
from sglang.srt.jax.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import add_prefix
import numpy as np
from jax.sharding import PartitionSpec
from flax.typing import Sharding


class QWenMLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        kernel_init: nnx.Initializer = nnx.initializers.lecun_normal(),
        kernel_1_partition: Sharding = (None, None),
        kernel_2_partition: Sharding = (None, None),
        kernel_3_partition: Sharding = (None, None),
        *,  # Following arguments are keyword-only
        rngs: nnx.Rngs,
    ):

        self.w1 = nnx.Linear(
            hidden_size,
            intermediate_size//2,
            kernel_init=nnx.with_partitioning(kernel_init, kernel_1_partition),
            use_bias=False,
            rngs=rngs
        )

        self.w2 = nnx.Linear(
            hidden_size,
            intermediate_size//2,
            kernel_init=nnx.with_partitioning(kernel_init, kernel_2_partition),
            use_bias=False,
            rngs=rngs
        )

        self.c_proj = nnx.Linear(
            intermediate_size//2,
            hidden_size,
            kernel_init=nnx.with_partitioning(kernel_init, kernel_3_partition),
            use_bias=False,
            rngs=rngs
        )

        self.act_func = nnx.silu

    def __call__(self, hidden_states: jnp.ndarray):
        a1 = self.w1(hidden_states)
        a2 = self.w2(hidden_states)
        intermediate_parallel = a1 * self.act_func(a2)
        output = self.c_proj(intermediate_parallel)
        return output


class QWenAttention(nnx.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 max_position_embeddings: int,
                 rope_theta: float,
                 rope_scaling: Optional[Dict[str, Any]],
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        head_size = hidden_size // num_heads
        self.c_attn = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_size,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("c_attn", prefix),
        )
        self.c_proj = LinearBase(
            input_size=num_heads * head_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("c_proj", prefix),
        )
        self.rotary_emb = RotaryEmbedding()
        self.attn = Attention()

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
    ) -> jax.Array:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        output, _ = self.c_proj(attn_output)
        return output


class QWenBlock(nnx.Module):
    def __init__(self,
                 config: PretrainedConfig,
                 layer_id: int,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        self.ln_1 = RMSNorm(config.hidden_size,
                            eps=config.layer_norm_epsilon)

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.attn = QWenAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        self.ln_2 = RMSNorm(config.hidden_size,
                            eps=config.layer_norm_epsilon)

        self.mlp = QWenMLP(
            config.hidden_size,
            config.intermediate_size // 2,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
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
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        self.wte = VocabParallelEmbedding(
            ((config.vocab_size + 63) // 64) * 64,
            config.hidden_size,
        )
        self.h = nnx.ModuleList(
            [
                QWenBlock(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"h.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = RMSNorm(epsilon=config.layer_norm_epsilon)

    def __call__(self,
                 input_ids: jax.Array,
                 positions: jax.Array,
                 forward_batch: ForwardBatch,
                 ):
        hidden_states = self.wte(input_ids)
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
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        self.transformer = QWenModel(config, quant_config, prefix)
        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.lm_head = ParallelLMHead(vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)

    def __call__(self,
                 input_ids: jax.Array,
                 positions: jax.Array,
                 forward_batch: ForwardBatch,
                 ):
        hidden_states = self.transformer(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head
        )
