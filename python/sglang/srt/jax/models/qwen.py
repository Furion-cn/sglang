from typing import Any, Callable, Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import PartitionSpec, mesh_sharding
from jax import numpy as jnp
from jax import with_sharding_constraint
from transformers import PretrainedConfig

from python.sglang.srt.jax.layers.attention import Attention
from sglang.srt.jax.layers.layernorm import RMSNorm
from sglang.srt.jax.layers.linear import LinearBase, QKVParallelLinear
from sglang.srt.jax.layers.logits_processor import LogitsProcessor
from sglang.srt.jax.layers.quantization.base_config import QuantizationConfig
from sglang.srt.jax.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.jax.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import add_prefix


class QWenMLP(nn.Module):
    hidden_size:int
    intermediate_size:int
    hidden_act:str="silu"
    quant_config: Optional[QuantizationConfig]
    dense_init: Callable = nn.initializers.xavier_normal()

    def setup(
        self,
    ):
        self.w1=nn.Dense(
          features=2*self.intermediate_size,
          use_bias=False,
          kernel_init=nn.with_partitioning(self.dense_init, (None, 'model')),
        )
        self.act_func=jax.nn.silu
        self.w2=self.param(
          'W2',
          nn.with_partitioning(self.dense_init, ('model', None)),
          (2*self.intermediate_size, self.hidden_size))


    def __call__(self,hidden_states:jnp.ndarray):
        y = self.w1(hidden_states)

        y=self.act_func(y)

        # Force a local sharding annotation.
        y = with_sharding_constraint(y, mesh_sharding(PartitionSpec('data', 'model')))

        z= jnp.dot(y,self.W2)
        # Force a local sharding annotation.
        z = with_sharding_constraint(z, mesh_sharding(PartitionSpec('data', None)))

        return z


class QWenAttention(nn.Module):
    hidden_size: int
    num_heads: int
    max_position_embeddings: int
    layer_id: int
    rope_theta: float
    rope_scaling: Optional[Dict[str, Any]]
    quant_config: Optional[QuantizationConfig] = None
    prefix: str = ""

    def setup(self):
        head_size = self.hidden_size // self.num_heads
        self.c_attn = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=head_size,
            num_heads=self.num_heads,
            bias=True,
            quant_config=self.quant_config,
            prefix=add_prefix("c_attn", self.prefix),
        )
        self.c_proj = LinearBase(
            input_size=self.num_heads * head_size,
            output_size=self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=add_prefix("c_proj", self.prefix),
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
        attn_output = self.attn(q, k, v)
        output, _ = self.c_proj(attn_output)
        return output


class QWenBlock(nn.Module):
    config: PretrainedConfig
    layer_id: int
    quant_config: Optional[QuantizationConfig] = None

    def setup(self):
        self.ln_1 = RMSNorm(self.config.hidden_size,
                            eps=self.config.layer_norm_epsilon)

        rope_theta = getattr(self.config, "rope_theta", 10000)
        rope_scaling = getattr(self.config, "rope_scaling", None)
        self.attn = QWenAttention(
            self.config.hidden_size,
            self.config.num_attention_heads,
            self.config.max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            layer_id=self.layer_id,
            quant_config=self.quant_config,
            prefix=add_prefix("attn", self.prefix),
        )

        self.ln_2 = RMSNorm(self.config.hidden_size,
                            eps=self.config.layer_norm_epsilon)

        self.mlp = QWenMLP(
            self.config.hidden_size,
            self.config.intermediate_size // 2,
            quant_config=self.quant_config,
            prefix=add_prefix("mlp", self.prefix),
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


class QWenModel(nn.Module):
    """QWen model"""

    config: PretrainedConfig
    quant_config: Optional[QuantizationConfig] = None

    def setup(self):
        self.vocab_size = self.config.vocab_size

        self.wte = VocabParallelEmbedding(
            ((self.config.vocab_size + 63) // 64) * 64,
            self.config.hidden_size,
        )
        self.h = nn.ModuleList(
            [
                QWenBlock(
                    self.config,
                    i,
                    quant_config=self.quant_config,
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )
        self.ln_f = RMSNorm(epsilon=self.config.layer_norm_epsilon)

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


class QWenLMHeadModel(nn.Module):
    """QWen language head model"""

    config: PretrainedConfig

    def setup(self, config: PretrainedConfig):
        self.transformer = QWenModel(config)
        vocab_size = ((self.config.vocab_size + 63) // 64) * 64
        self.lm_head = ParallelLMHead(vocab_size, self.config.hidden_size)
        self.logits_processor = LogitsProcessor(self.config)

    def __call__(self,
                 input_ids: jax.Array,
                 positions: jax.Array,
                 forward_batch: ForwardBatch,
                 ):
        hidden_states = self.transformer(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head
        )
