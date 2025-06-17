from typing import Any, Dict, Optional

import jax
from flax import linen as nn
from jax import numpy as jnp
from transformers import PretrainedConfig

from sglang.srt.jax.layers.layernorm import RMSNorm
from sglang.srt.jax.layers.logits_processor import LogitsProcessor
from sglang.srt.jax.layers.quantization.base_config import QuantizationConfig
from sglang.srt.jax.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
import flax.linen as nn
from typing import Callable
from jax import with_sharding_constraint, mesh_sharding, PartitionSpec
import jax
import jax.numpy as jnp
from typing import Optional,Any


class QWenMLP(nn.Module):
    def setup(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        quant_config: Optional[QuantizationConfig] = None,
        dense_init: Callable = nn.initializers.xavier_normal()
        
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.quant_config = quant_config
        self.dense_init = dense_init

    @nn.compact
    def __call__(self,hidden_states:jnp.ndarray):
        y = nn.Dense(
          features=2*self.intermediate_size,
          use_bias=False,
          kernel_init=nn.with_partitioning(self.dense_init, (None, 'model')),
        )(hidden_states)

        y=jax.nn.silu(y)

        # Force a local sharding annotation.
        y = with_sharding_constraint(y, mesh_sharding(PartitionSpec('data','model')))

        W2 = self.param(
          'W2',
          nn.with_partitioning(self.dense_init, ('model', None)),
          (self.hidden_size, y.shape[-1]))
        z= jnp.dot(y,W2)
        # Force a local sharding annotation.
        z = with_sharding_constraint(z, mesh_sharding(PartitionSpec('data', None)))

        return z 


class QWenAttention(nn.Module):
    def setup(
        self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        pass

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
    ) -> jax.Array:
        pass


class QWenBlock(nn.Module):
    config: PretrainedConfig
    layer_id: int
    quant_config: Optional[QuantizationConfig] = None

    def setup(self):
        pass

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
    ) -> jax.Array:
        pass


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
