# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/qwen.py#L1

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.debug_tracer import global_tracer, trace_function
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix


class QWenMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            2 * [intermediate_size],
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.c_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=add_prefix("c_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    @trace_function(stage="MLP", include_args=False, include_output=True)
    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.c_proj(x)
        return x


class QWenAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size
        self.head_dim = hidden_size // self.total_num_heads

        # pylint: disable=invalid-name
        self.c_attn = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("c_attn", prefix),
        )
        self.c_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=add_prefix("c_proj", prefix),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.scaling = self.head_dim**-0.5
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    @trace_function(stage="ATTENTION", include_args=False, include_output=True)
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        global_tracer.print(q, "q_proj", f"qkv_proj_layer_id_{self.layer_id}")
        global_tracer.print(k, "k_proj", f"qkv_proj_layer_id_{self.layer_id}")
        global_tracer.print(v, "v_proj", f"qkv_proj_layer_id_{self.layer_id}")
        q, k = self.rotary_emb(positions, q, k)
        global_tracer.print(q, "q_rotary", f"qkv_proj_layer_id_{self.layer_id}")
        global_tracer.print(k, "k_rotary", f"qkv_proj_layer_id_{self.layer_id}")
        attn_output = self.attn(q, k, v, forward_batch)
        global_tracer.print(attn_output, "attn_output", f"attn_layer_id_{self.layer_id}")
        output, _ = self.c_proj(attn_output)
        return output


class QWenBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

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

        self.ln_2 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = QWenMLP(
            config.hidden_size,
            config.intermediate_size // 2,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    @trace_function(stage="BLOCK", include_args=False, include_output=True)
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        
        global_tracer.print(hidden_states, f"RMSNorm_pre_attn_input", f"rmsnorm_layer_id_{self.layer_id}")
        hidden_states = self.ln_1(hidden_states)
        global_tracer.print(hidden_states, f"RMSNorm_pre_attn_output", f"rmsnorm_layer_id_{self.layer_id}")
        
        hidden_states = self.attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        
        global_tracer.print(hidden_states, f"RMSNorm_pre_mlp_input", f"rmsnorm_layer_id_{self.layer_id}")
        hidden_states = self.ln_2(hidden_states)
        global_tracer.print(hidden_states, f"RMSNorm_pre_mlp_output", f"rmsnorm_layer_id_{self.layer_id}")
        
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QWenModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.wte = VocabParallelEmbedding(
            vocab_size,
            config.hidden_size,
            prefix=add_prefix("wte", prefix),
        )
        self.h = nn.ModuleList(
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
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    @trace_function(stage="TRANSFORMER", include_args=False, include_output=True)
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        global_tracer.print(input_ids, "embedding_input", "embedding_all")
        hidden_states = self.wte(input_ids)
        global_tracer.print(hidden_states, "embedding_output", "embedding_all")
        
        for layer in self.h:
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        
        global_tracer.print(hidden_states, "RMSNorm_final_input", "rmsnorm_final")
        hidden_states = self.ln_f(hidden_states)
        global_tracer.print(hidden_states, "RMSNorm_final_output", "rmsnorm_final")
        
        return hidden_states


class QWenLMHeadModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.transformer = QWenModel(
            config, quant_config=quant_config, prefix=add_prefix("transformer", prefix)
        )
        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.lm_head = ParallelLMHead(
            vocab_size, config.hidden_size, prefix=add_prefix("lm_head", prefix)
        )
        self.logits_processor = LogitsProcessor(config)
        
        # Initialize tokenizer for debug tracer
        self._setup_debug_tracer()

    def _setup_debug_tracer(self):
        try:
            global_tracer.set_model(self)
        except Exception as e:
            print(f"Warning: Could not setup debug tracer: {str(e)}")

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        hidden_states = self.transformer(input_ids, positions, forward_batch)
        result = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
        
        if global_tracer.is_session_active():
            input_data = {
                "input_ids": input_ids,
                "input_shape": list(input_ids.shape)
            }
            
            output_data = {
                "output_type": str(type(result).__name__)
            }
            
            if hasattr(result, 'next_token_logits') and result.next_token_logits is not None:
                output_data.update({
                    "logits": result.next_token_logits,
                    "logits_shape": list(result.next_token_logits.shape)
                })
            
            global_tracer.print(result.next_token_logits, "next_token_logits", "logits_all")
            
            global_tracer.accumulate_step(input_data, output_data)
            
            if global_tracer.should_auto_save():
                global_tracer.end_session()
        
        return result

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w2", 0),
            ("gate_up_proj", "w1", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = QWenLMHeadModel
