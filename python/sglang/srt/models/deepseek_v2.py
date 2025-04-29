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

# Adapted from:
# https://github.com/vllm-project/vllm/blob/fb6af8bc086328ca6659e72d11ffd4309ce4de22/vllm/model_executor/models/deepseek_v2.py
"""Inference-only DeepseekV2 model."""

from sglang.srt.utils import DeepEPMode, add_prefix, is_cuda, is_hip
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.managers.expert_distribution import ExpertDistributionRecorder
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.layers.rotary_embedding import get_rope, get_rope_wrapper
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.quantization.int8_utils import (
    block_dequant as int8_block_dequant,
)
from sglang.srt.layers.quantization.fp8_utils import (
    input_to_float8,
    block_quant_to_tensor_quant,
    channel_quant_to_tensor_quant,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.fp8_kernel import per_tensor_quant_mla_fp8
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.ep_moe.token_dispatcher import (
    DeepEPDispatcher,
    DeepEPDispatcherImplLowLatency,
    DeepEPDispatcherImplNormal,
)
from sglang.srt.layers.moe.ep_moe.token_dispatcher import DeepEPDispatcher
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, EPMoE
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.layers.dp_attention import (
    dp_gather_partial,
    dp_scatter,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    tp_all_gather,
    tp_reduce_scatter,
    tp_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    parallel_state,
    tensor_model_parallel_all_reduce,
    get_tensor_model_parallel_rank
)
from transformers import PretrainedConfig
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import torch
from typing import Any, Dict, Iterable, Optional, Tuple
from enum import Enum
from copy import copy, deepcopy
from enum import IntEnum, auto
import logging
import os

_is_hip = is_hip()
_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import awq_dequantize, bmm_fp8, merge_state_v2
else:
    from vllm._custom_ops import awq_dequantize

if _is_hip:
    from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
        decode_attention_fwd_grouped_rope,
    )

expert_distribution_recorder = ExpertDistributionRecorder()

logger = logging.getLogger(__name__)


class AttnForwardMethod(IntEnum):
    # Use multi-head attention
    MHA = auto()

    # Use absorbed multi-latent attention
    MLA = auto()

    # Use multi-head attention, but with KV cache chunked.
    # This method can avoid OOM when prefix lengths are long.
    MHA_CHUNKED_KV = auto()


class MicroBatchOverlapStep(Enum):
    SHARED_EXPERTS = "shared_experts"
    MLP = "mlp"
    MOE_GATE = "moe_gate"
    # prefill attn
    PREFILL_ATTN = "prefill_attn"
    # decode attn
    DECODE_ATTN_0_STEP = "decode_attn_0"
    DECODE_ATTN_1_STEP = "decode_attn_1"
    # normal deepep
    WAIT_DISPATCH_NORMAL = "wait_dispatch_normal"
    WAIT_COMBINE_NORMAL = "wait_combine_normal"
    LAUNCH_DISPATCH_NORMAL = "launch_dispatch_normal"
    LAUNCH_COMBINE_NORMAL = "launch_combine_normal"
    # low latency deepep
    LAUNCH_DISPATCH_LL_STEP = "launch_dispatch_ll"
    WAIT_DISPATCH_LL_STEP = "wait_dispatch_ll"
    LAUNCH_COMBINE_LL_STEP = "launch_combine_ll"
    WAIT_COMBINE_LL_STEP = "wait_combine_ll"


class MicroBatchOverlapExtraArgs(Enum):
    # common extra_args keys
    EXTRA_ARGS_TOPK_IDX_KEY = "topk_idx"
    EXTRA_ARGS_TOPK_WEIGHTS_KEY = "topk_weights"
    EXTRA_ARGS_MICRO_BATCH_IDX_KEY = "micro_batch_idx"
    EXTRA_ARGS_MICRO_BATCH_EMPTY_MLP_OUTPUT_KEY = "empty_mlp_output"
    EXTRA_ARGS_SHARED_EXPERT_OUTPUT_KEY = "shared_experts_output"
    EXTRA_ARGS_BEFORE_DISPATCH_HIDDEN_STATES_KEY = "before_dispatch_hidden_states"

    EXTRA_ARGS_LAST_LAYER_SHARED_EXPERT_OUTPUT_KEY = "last_layer_shared_experts_output"
    EXTRA_ARGS_ATTN_0_Q_INPUT_KEY = "attn_0_q_input"
    EXTRA_ARGS_ATTN_0_K_INPUT_KEY = "attn_0_k_input"
    EXTRA_ARGS_ATTN_0_V_INPUT_KEY = "attn_0_v_input"
    EXTRA_ARGS_ATTN_1_TOPK_WEIGHTS_KEY = "attn_1_topk_weights"
    EXTRA_ARGS_ATTN_1_TOPK_IDX_KEY = "attn_1_topk_idx"

    # get reorder_topk_ids, seg_indptr, masked_m, expected_m from extra_args
    EXTRA_ARGS_REORDER_TOPK_IDS_KEY = "reorder_topk_ids"
    EXTRA_ARGS_SEG_INDPTR_KEY = "seg_indptr"
    EXTRA_ARGS_MASKED_M_KEY = "masked_m"
    EXTRA_ARGS_EXPECTED_M_KEY = "expected_m"
    EXTRA_ARGS_EVENT_KEY = "event"
    EXTRA_ARGS_HOOK_KEY = "hook"


class DeepseekV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MoEGate(nn.Module):
    def __init__(
        self,
        config,
        prefix: str = "",
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
        if config.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((config.n_routed_experts))
            )
        else:
            self.e_score_correction_bias = None

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight, None)
        return logits


class DeepseekV2MoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.n_share_experts_fusion = (
            global_server_args_dict["n_share_experts_fusion"]
            if global_server_args_dict["n_share_experts_fusion"] is not None
            else 0
        )

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = MoEGate(config=config, prefix=add_prefix("gate", prefix))

        MoEImpl = (
            DeepEPMoE
            if global_server_args_dict["enable_deepep_moe"]
            else (EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE)
        )

        self.experts = MoEImpl(
            num_experts=config.n_routed_experts + self.n_share_experts_fusion,
            top_k=config.num_experts_per_tok +
                  min(self.n_share_experts_fusion, 1),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            prefix=add_prefix("experts", prefix),
            **(
                dict(
                    deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]])
                if global_server_args_dict["enable_deepep_moe"]
                else {}
            ),
        )

        if config.n_shared_experts is not None and self.n_share_experts_fusion == 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            # disable tp for shared experts when enable deepep moe
            if not global_server_args_dict["enable_deepep_moe"]:
                self.shared_experts = DeepseekV2MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=add_prefix("shared_experts", prefix),
                )
            else:
                self.shared_experts = DeepseekV2MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=add_prefix("shared_experts", prefix),
                    tp_rank=0,
                    tp_size=1,
                )

        if global_server_args_dict["enable_deepep_moe"]:
            # TODO: we will support tp < ep in the future
            self.ep_size = get_tensor_model_parallel_world_size()
            self.num_experts = config.n_routed_experts
            self.top_k = config.num_experts_per_tok
            self.renormalize = config.norm_topk_prob
            self.topk_group = config.topk_group
            self.num_expert_group = config.n_group
            self.correction_bias = (
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

            if global_server_args_dict["enable_micro_batch_overlap"]:
                self.deepep_dispatchers = {
                    "prefill": {
                        0: DeepEPDispatcher(
                            group=parallel_state.get_tp_group().device_group,
                            router_topk=self.top_k,
                            permute_fusion=True,
                            num_experts=config.n_routed_experts,
                            num_local_experts=config.n_routed_experts // self.tp_size,
                            hidden_size=config.hidden_size,
                            params_dtype=config.torch_dtype,
                            deepep_mode=DeepEPMode.normal,
                            async_finish=True,  # TODO
                            return_recv_hook=True,
                        ),
                        1: DeepEPDispatcher(
                            group=parallel_state.get_tp_group().device_group,
                            router_topk=self.top_k,
                            permute_fusion=True,
                            num_experts=config.n_routed_experts,
                            num_local_experts=config.n_routed_experts // self.tp_size,
                            hidden_size=config.hidden_size,
                            params_dtype=config.torch_dtype,
                            deepep_mode=DeepEPMode.normal,
                            async_finish=True,  # TODO
                            return_recv_hook=True,
                        ),
                    },
                    "decode": {
                        0: DeepEPDispatcher(
                            group=parallel_state.get_tp_group().device_group,
                            router_topk=self.top_k,
                            permute_fusion=True,
                            num_experts=config.n_routed_experts,
                            num_local_experts=config.n_routed_experts // self.tp_size,
                            hidden_size=config.hidden_size,
                            params_dtype=config.torch_dtype,
                            deepep_mode=DeepEPMode.low_latency,
                            async_finish=True,  # TODO
                            return_recv_hook=True,
                        ),
                        1: DeepEPDispatcher(
                            group=parallel_state.get_tp_group().device_group,
                            router_topk=self.top_k,
                            permute_fusion=True,
                            num_experts=config.n_routed_experts,
                            num_local_experts=config.n_routed_experts // self.tp_size,
                            hidden_size=config.hidden_size,
                            params_dtype=config.torch_dtype,
                            deepep_mode=DeepEPMode.low_latency,
                            async_finish=True,  # TODO
                            return_recv_hook=True,
                        ),
                    },
                }
            self.deepep_dispatcher = DeepEPDispatcher(
                group=parallel_state.get_tp_group().device_group,
                router_topk=self.top_k,
                permute_fusion=True,
                num_experts=config.n_routed_experts,
                num_local_experts=config.n_routed_experts // self.tp_size,
                hidden_size=config.hidden_size,
                params_dtype=config.torch_dtype,
                deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]],
                async_finish=True,  # TODO
                return_recv_hook=True,
            )

    def forward(
        self, hidden_states: torch.Tensor, forward_mode: Optional[ForwardMode] = None
    ) -> torch.Tensor:
        if not global_server_args_dict["enable_deepep_moe"]:
            with torch.cuda.nvtx.range("deepseek v2 moe without deepep_moe"):
                return self.forward_normal(hidden_states)
        else:
            with torch.cuda.nvtx.range("deepseek v2 moe with deepep_moe"):
                return self.forward_deepep(hidden_states, forward_mode)

    def forward_normal(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.cuda.nvtx.range("forward_normal _forward_shared_experts"):
            shared_output = self._forward_shared_experts(hidden_states)
        # router_logits: (num_tokens, n_experts)
        with torch.cuda.nvtx.range("forward_normal gate"):
            router_logits = self.gate(hidden_states)
        with torch.cuda.nvtx.range("forward_normal self experts"):
            final_hidden_states = (
                self.experts(hidden_states=hidden_states,
                             router_logits=router_logits)
                * self.routed_scaling_factor
            )
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1:
            with torch.cuda.nvtx.range("tensor_model_parallel_all_reduce"):
                final_hidden_states = tensor_model_parallel_all_reduce(
                    final_hidden_states)
        return final_hidden_states

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_mode: ForwardMode
    ) -> torch.Tensor:
        shared_output = None
        topk_idx = torch.full(
            (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
        )
        topk_weights = torch.empty(
            (0, self.top_k), dtype=torch.float32, device=hidden_states.device
        )
        if (
            forward_mode is not None
            and not forward_mode.is_idle()
            and hidden_states.shape[0] > 0
        ):
            with torch.cuda.nvtx.range("forward_deepep gate"):
                # router_logits: (num_tokens, n_experts)
                router_logits = self.gate(hidden_states)
            with torch.cuda.nvtx.range("forward_deepep _forward_shared_experts"):
                shared_output = self._forward_shared_experts(hidden_states)
            topk_weights, topk_idx = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                use_grouped_topk=True,
                renormalize=self.renormalize,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                correction_bias=self.correction_bias,
            )
        # if get_attention_tp_rank() == 0:
        #     prev_cached_mem, prev_allocated_mem, prev_free_mem = get_gpu_memory_info_not_str(0)
        if self.ep_size > 1:
            with torch.cuda.nvtx.range("dispatch"):
                # TODO(ch-wan): allow users to set num_max_dispatch_tokens_per_rank value
                (
                    hidden_states,
                    topk_idx,
                    topk_weights,
                    reorder_topk_ids,
                    seg_indptr,
                    masked_m,
                    expected_m,
                ) = self.deepep_dispatcher.dispatch(
                    hidden_states,
                    topk_idx,
                    topk_weights,
                    forward_mode=forward_mode,
                )
        # if get_attention_tp_rank() == 0:
        #     logger.debug(
        #         f"original_moe,after dispatch,{get_gpu_memory_info_diff_str(0, prev_cached_mem, prev_allocated_mem, prev_free_mem)}")
        #     prev_cached_mem, prev_allocated_mem, prev_free_mem = get_gpu_memory_info_not_str(0)
        with torch.cuda.nvtx.range("experts_foward"):
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                reorder_topk_ids=reorder_topk_ids,
                seg_indptr=seg_indptr,
                masked_m=masked_m,
                expected_m=expected_m,
                forward_mode=forward_mode,
            )
        # if get_attention_tp_rank() == 0:
        #     logger.debug(
        #         f"original_moe,after experts,{get_gpu_memory_info_diff_str(0, prev_cached_mem, prev_allocated_mem, prev_free_mem)}")
        #     prev_cached_mem, prev_allocated_mem, prev_free_mem = get_gpu_memory_info_not_str(0)
        if self.ep_size > 1:
            with torch.cuda.nvtx.range("combine"):
                final_hidden_states = self.deepep_dispatcher.combine(
                    final_hidden_states,
                    topk_idx,
                    topk_weights,
                    forward_mode,
                )
        final_hidden_states *= self.routed_scaling_factor

        # if get_attention_tp_rank() == 0:
        #     logger.debug(
        #         f"original_moe,after combine,{get_gpu_memory_info_diff_str(0, prev_cached_mem, prev_allocated_mem, prev_free_mem)}")
        # prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states

    def _forward_shared_experts(self, hidden_states):
        if self.n_shared_experts is not None and self.n_share_experts_fusion == 0:
            return self.shared_experts(hidden_states)
        else:
            return None


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV2Attention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id=None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.dp_size = get_attention_dp_size()
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        self.num_local_heads = num_heads // attn_tp_size
        self.scaling = self.qk_head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_a_proj", prefix),
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_b_proj", prefix),
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_a_proj_with_mqa", prefix),
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
        )
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            reduce_results=reduce_results,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        rope_scaling["rope_type"] = "deepseek_yarn"
        self.rotary_emb = get_rope_wrapper(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
            device=global_server_args_dict["device"],
        )

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # TODO, support head_size 192
        self.attn = RadixAttention(
            self.num_local_heads,
            256,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            assert (
                not self.o_proj.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return hidden_states

        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        with torch.cuda.nvtx.range("deepseek v2 attention forward"):
            _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            latent_cache = latent_cache.unsqueeze(1)
            kv_a = self.kv_a_layernorm(kv_a.contiguous())
            kv = self.kv_b_proj(kv_a)[0]
            kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k_pe = latent_cache[:, :, self.kv_lora_rank:]
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            q[..., self.qk_nope_head_dim:] = q_pe
            k = torch.empty_like(q)
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim:] = k_pe
            q = torch.nn.functional.pad(q, [0, 256 - self.qk_head_dim], value=0).view(
                -1, self.num_local_heads * 256
            )
            k = torch.nn.functional.pad(k, [0, 256 - self.qk_head_dim], value=0).view(
                -1, self.num_local_heads * 256
            )
            v = torch.nn.functional.pad(v, [0, 256 - self.v_head_dim], value=0).view(
                -1, self.num_local_heads * 256
            )
            attn_output = self.attn(q, k, v, forward_batch)
            attn_output = attn_output.view(-1, self.num_local_heads, 256)[
                          ..., : self.v_head_dim
                          ].reshape(-1, self.num_local_heads * self.v_head_dim)
            output, _ = self.o_proj(attn_output)
            return output


class DeepseekV2AttentionMLA(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        layer_id: int = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.dp_size = get_attention_dp_size()
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        self.num_local_heads = num_heads // attn_tp_size
        self.scaling = self.qk_head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # For tensor parallel attention
        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_a_proj", prefix),
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_b_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_a_proj_with_mqa", prefix),
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale
        else:
            self.rotary_emb.forward = self.rotary_emb.forward_native

        self.attn_mqa = RadixAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            quant_config=quant_config,
            prefix=add_prefix("attn_mha", prefix),
        )

        self.w_kc = None
        self.w_vc = None
        self.w_scale = None

        self.flashinfer_mla_disable_ragged = global_server_args_dict[
            "flashinfer_mla_disable_ragged"
        ]
        self.disable_chunked_prefix_cache = global_server_args_dict[
            "disable_chunked_prefix_cache"
        ]
        self.attention_backend = global_server_args_dict["attention_backend"]
        self.rocm_fused_decode_mla = os.getenv("SGLANG_ROCM_FUSED_DECODE_MLA") == "1"

        # TODO: Design a finer way to determine the threshold
        self.chunked_prefix_cache_threshold = 8192

    def dispatch_attn_forward_method(
        self, forward_batch: ForwardBatch
    ) -> AttnForwardMethod:
        if self.attention_backend == "flashinfer":
            # Flashinfer MLA: Do not absorb when enabling ragged prefill
            if (
                not self.flashinfer_mla_disable_ragged
                and forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and sum(forward_batch.extend_prefix_lens_cpu) == 0
            ):
                return AttnForwardMethod.MHA
            else:
                return AttnForwardMethod.MLA
        elif self.attention_backend == "fa3":
            # Flash Attention: Use MHA with chunked KV cache when prefilling on long sequences.
            if (
                forward_batch.forward_mode.is_extend()
                and not self.disable_chunked_prefix_cache
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and sum(forward_batch.extend_prefix_lens_cpu)
                >= self.chunked_prefix_cache_threshold
            ):
                return AttnForwardMethod.MHA_CHUNKED_KV
            else:
                return AttnForwardMethod.MLA
        else:
            # Triton: Use normal computation for prefill and use weight absorption for extend/decode
            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and sum(forward_batch.extend_prefix_lens_cpu) == 0
            ):
                return AttnForwardMethod.MHA
            else:
                return AttnForwardMethod.MLA

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        with torch.cuda.nvtx.range("attention_mla_forward"):
            if hidden_states.shape[0] == 0:
                assert (
                    not self.o_proj.reduce_results
                ), "short-circuiting allreduce will lead to hangs"
                return hidden_states

            with torch.cuda.nvtx.range("dispatch_attn_method"):
                attn_forward_method = self.dispatch_attn_forward_method(forward_batch)

            if attn_forward_method == AttnForwardMethod.MHA:
                with torch.cuda.nvtx.range("forward_normal"):
                    return self.forward_normal(positions, hidden_states, forward_batch)
            elif attn_forward_method == AttnForwardMethod.MHA_CHUNKED_KV:
                with torch.cuda.nvtx.range("forward_normal_chunked_kv"):
                    return self.forward_normal_chunked_kv(
                        positions, hidden_states, forward_batch
                    )
            else:
                if _is_hip:
                    if (
                        self.rocm_fused_decode_mla
                        and forward_batch.forward_mode.is_decode()
                    ):
                        with torch.cuda.nvtx.range("forward_absorb_fused_mla_rope"):
                            return self.forward_absorb_fused_mla_rope(
                                positions, hidden_states, forward_batch
                            )
                    else:
                        with torch.cuda.nvtx.range("forward_absorb"):
                            return self.forward_absorb(positions, hidden_states, forward_batch)
                else:
                    with torch.cuda.nvtx.range("forward_absorb"):
                        return self.forward_absorb(positions, hidden_states, forward_batch)

    def forward_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim:]
        k_pe = latent_cache[:, :, self.kv_lora_rank:]
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim:] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe

        latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
        latent_cache[:, :, self.kv_lora_rank:] = k_pe

        # Save latent cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
        )
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_absorb(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        with torch.cuda.nvtx.range("absorb_forward"):
            q_len = hidden_states.shape[0]
            q_input = hidden_states.new_empty(
                q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
            )

            with torch.cuda.nvtx.range("q_projection"):
                if self.q_lora_rank is not None:
                    q = self.q_a_proj(hidden_states)[0]
                    q = self.q_a_layernorm(q)
                    q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
                else:
                    q = self.q_proj(hidden_states)[0].view(
                        -1, self.num_local_heads, self.qk_head_dim
                    )
                q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

            with torch.cuda.nvtx.range("w_kc_ops"):
                if self.w_kc.dtype == torch.float8_e4m3fnuz:
                    q_nope_out = torch.bmm(
                        q_nope.to(torch.bfloat16).transpose(0, 1),
                        self.w_kc.to(torch.bfloat16) * self.w_scale,
                    )
                elif self.w_kc.dtype == torch.float8_e4m3fn:
                    q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                        q_nope.transpose(0, 1),
                    )
                    q_nope_out = bmm_fp8(
                        q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
                    )
                else:
                    q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
                q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)

            with torch.cuda.nvtx.range("kv_projection"):
                latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
                v_input = latent_cache[..., : self.kv_lora_rank]
                v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
                k_input = latent_cache.unsqueeze(1)
                k_input[..., : self.kv_lora_rank] = v_input
                k_pe = k_input[..., self.kv_lora_rank:]

            with torch.cuda.nvtx.range("rope"):
                q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
                q_input[..., self.kv_lora_rank:] = q_pe
                k_input[..., self.kv_lora_rank:] = k_pe

            with torch.cuda.nvtx.range("attention"):
                attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
                attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

            with torch.cuda.nvtx.range("w_vc_ops"):
                if self.w_vc.dtype == torch.float8_e4m3fnuz:
                    attn_bmm_output = torch.bmm(
                        attn_output.to(torch.bfloat16).transpose(0, 1),
                        self.w_vc.to(torch.bfloat16) * self.w_scale,
                    )
                elif self.w_vc.dtype == torch.float8_e4m3fn:
                    attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                        attn_output.transpose(0, 1),
                    )
                    attn_bmm_output = bmm_fp8(
                        attn_output_val,
                        self.w_vc,
                        attn_output_scale,
                        self.w_scale,
                        torch.bfloat16,
                    )
                else:
                    attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
                attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

            with torch.cuda.nvtx.range("output_projection"):
                output, _ = self.o_proj(attn_output)
                return output

    def forward_absorb_fused_mla_rope(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        enable_rope_fusion = (
            os.getenv("SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION", "1") == "1"
        )
        q_len = hidden_states.shape[0]
        q_input = hidden_states.new_empty(
            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
        )
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if self.w_kc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                self.w_kc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1), dtype=torch.float8_e4m3fn
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
        q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        v_input = latent_cache[..., : self.kv_lora_rank]
        v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
        k_input = latent_cache.unsqueeze(1)
        k_input[..., : self.kv_lora_rank] = v_input

        if not enable_rope_fusion:
            k_pe = k_input[..., self.kv_lora_rank:]
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            q_input[..., self.kv_lora_rank:] = q_pe
            k_input[..., self.kv_lora_rank:] = k_pe
            k_pe_output = None
        else:
            k_pe_output = torch.empty_like(k_input[..., self.kv_lora_rank:])

        q_input[..., self.kv_lora_rank:] = q_pe

        # attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        # Use Fused ROPE with use_rope=OFF.
        attn_output = torch.empty(
            (q_len, self.num_local_heads, self.kv_lora_rank),
            dtype=q.dtype,
            device=q.device,
        )
        attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
            forward_batch.attn_backend.forward_metadata
        )
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        num_kv_split = forward_batch.attn_backend.num_kv_splits
        sm_scale = self.attn_mqa.scaling
        if attn_logits is None:
            attn_logits = torch.empty(
                (
                    forward_batch.batch_size,
                    self.num_local_heads,
                    num_kv_split,
                    self.kv_lora_rank + 1,
                ),
                dtype=torch.float32,
                device=q.device,
            )

        # save current latent cache.
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mqa, forward_batch.out_cache_loc, k_input, None
        )
        key_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
            self.attn_mqa.layer_id
        )
        val_cache_buf = key_cache_buf[..., : self.kv_lora_rank]

        decode_attention_fwd_grouped_rope(
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            self.kv_lora_rank,
            self.rotary_emb.rotary_dim,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            logit_cap=self.attn_mqa.logit_cap,
            use_rope=enable_rope_fusion,
            is_neox_style=self.rotary_emb.is_neox_style,
        )

        if enable_rope_fusion:
            k_input[..., self.kv_lora_rank:] = k_pe_output
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn_mqa, forward_batch.out_cache_loc, k_input, None
            )

        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.w_vc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                self.w_vc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1), dtype=torch.float8_e4m3fn
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)

        return output

    def _chunked_prefix_attn_mha(
        self,
        q: torch.Tensor,
        accum_output: torch.Tensor,
        accum_lse: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        assert forward_batch.num_prefix_chunks is not None
        for i in range(forward_batch.num_prefix_chunks):
            forward_batch.set_prefix_chunk_idx(i)

            # Fetch latent cache from memory pool with precomputed chunked kv indices
            latent_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
                self.attn_mha.layer_id
            )
            latent_cache = latent_cache_buf[
                forward_batch.prefix_chunk_kv_indices[i]
            ].contiguous()

            kv_a_normed, k_pe = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a_normed = kv_a_normed.squeeze(1).contiguous()
            kv = self.kv_b_proj(kv_a_normed)[0]
            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            v = kv[..., self.qk_nope_head_dim:]
            k_nope = kv[..., : self.qk_nope_head_dim]

            k = torch.empty(
                (
                    k_nope.shape[0],
                    self.num_local_heads,
                    self.qk_nope_head_dim + self.qk_rope_head_dim,
                ),
                dtype=v.dtype,
                device=v.device,
            )
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim:] = k_pe

            output, lse = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
            lse = torch.transpose(lse, 0, 1).contiguous()
            tmp_output = torch.empty_like(accum_output)
            tmp_lse = torch.empty_like(accum_lse)
            merge_state_v2(output, lse, accum_output, accum_lse, tmp_output, tmp_lse)
            accum_output, accum_lse = tmp_output, tmp_lse

        return accum_output

    def forward_normal_chunked_kv(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # In normal mha, the k and v tensors will become overly large when the prefix length is long.
        # To avoid this, we split the kv cache into chunks and process them one after another.
        # Since mha is compute friendly, the for loop induced here will not introduce significant overhead.
        # The top comments in https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py
        # will be helpful for understanding the purpose of this function.

        # First do normal mha forward to get output for extended part
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim:]
        k_pe = latent_cache[:, :, self.kv_lora_rank:]

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim:] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe

        latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
        latent_cache[:, :, self.kv_lora_rank:] = k_pe

        # Save latent cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
        )

        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        attn_output, lse = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
        lse = torch.transpose(lse, 0, 1).contiguous()

        # Do mha attention with chunked prefix cache if there are any sequence with prefix
        if any(forward_batch.extend_prefix_lens_cpu):
            # Only initialize the info once
            if forward_batch.num_prefix_chunks is None:
                forward_batch.prepare_chunked_prefix_cache_info(q.device)

            forward_batch.set_attn_attend_prefix_cache(True)
            attn_output = self._chunked_prefix_attn_mha(
                q=q,
                accum_output=attn_output,
                accum_lse=lse,
                forward_batch=forward_batch,
            )

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
    ) -> None:

        def is_sparse_layer(l: int):
            return (
                config.n_routed_experts is not None
                and l >= config.first_k_dense_replace
                and l % config.moe_layer_freq == 0
            )

        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.enable_dp_attention = global_server_args_dict["enable_dp_attention"]
        self.layer_id = layer_id
        self.dp_size = get_attention_dp_size()
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.num_local_heads = config.num_attention_heads

        if not global_server_args_dict["disable_mla"]:
            self.self_attn = DeepseekV2AttentionMLA(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=(
                    config.q_lora_rank if hasattr(config, "q_lora_rank") else None
                ),
                kv_lora_rank=config.kv_lora_rank,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                layer_id=layer_id,
                reduce_results=False,
                prefix=add_prefix("self_attn", prefix),
            )
        else:
            self.self_attn = DeepseekV2Attention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=(
                    config.q_lora_rank if hasattr(config, "q_lora_rank") else None
                ),
                kv_lora_rank=config.kv_lora_rank,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                layer_id=layer_id,
                reduce_results=False,
                prefix=add_prefix("self_attn", prefix),
            )

        if is_nextn or is_sparse_layer(layer_id):
            # note: discussion, old one DeepseekV2MoE
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
            self.is_sparse = True
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=0,
                tp_size=1,
            )
            self.is_sparse = False

        self.input_is_scattered = (
            is_sparse_layer(layer_id - 1)
            and global_server_args_dict["enable_deepep_moe"]
        )
        self.is_last_layer = self.layer_id == config.num_hidden_layers - 1

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward_shared_experts(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        shared_output = None
        if (
            forward_batch.forward_mode is not None
            and not forward_batch.forward_mode.is_idle()
            and hidden_states.shape[0] > 0
        ):
            if self.mlp.n_shared_experts is not None:
                shared_output = self.mlp.shared_experts(hidden_states)
        extra_args.update(
            {
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_SHARED_EXPERT_OUTPUT_KEY: shared_output,
            }
        )
        return hidden_states, residual, extra_args

    def forward_moe_gate(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        topk_idx = torch.full(
            (0, self.mlp.top_k), -1, dtype=torch.int, device=hidden_states.device
        )
        topk_weights = torch.empty(
            (0, self.mlp.top_k), dtype=torch.float32, device=hidden_states.device
        )
        if (
            forward_batch.forward_mode is not None
            and not forward_batch.forward_mode.is_idle()
            and hidden_states.shape[0] > 0
        ):
            # router_logits: (num_tokens, n_experts)
            router_logits = self.mlp.gate(hidden_states)
            topk_weights, topk_idx = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.mlp.top_k,
                use_grouped_topk=True,
                renormalize=self.mlp.renormalize,
                topk_group=self.mlp.topk_group,
                num_expert_group=self.mlp.num_expert_group,
                correction_bias=self.mlp.correction_bias,
            )
        extra_args.update(
            {
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_IDX_KEY: topk_idx,
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_WEIGHTS_KEY: topk_weights,
            }
        )
        return hidden_states, residual, extra_args

    def forward_decode_attn_0(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        """
        note:
        1. attn_0: add last layer's shared_experts_output
        2. attn_0: MLA down/up projection
        3. attn_0: only support absort implementation
        """
        with torch.cuda.nvtx.range("decode_attn_0"):
            # calculate residual
            if hidden_states.shape[0] == 0:
                residual = hidden_states
            else:
                if residual is None:
                    residual = hidden_states
                    hidden_states = self.input_layernorm(hidden_states)
                else:
                    hidden_states, residual = self.input_layernorm(hidden_states, residual)

            if self.attn_tp_size != 1 and self.input_is_scattered:
                hidden_states, local_hidden_states = (
                    forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                    hidden_states,
                )
                # logger.debug(
                # f"layer[{self.layer_id}], before tp_all_gather in forward_decode_attn_0, hidden_states: {hidden_states.shape}")
                tp_all_gather(
                    list(hidden_states.tensor_split(self.attn_tp_size)), local_hidden_states
                )
                # logger.debug(
                # f"layer[{self.layer_id}], after tp_all_gather in forward_decode_attn_0, hidden_states: {hidden_states.shape}")

            if hidden_states.shape[0] == 0:
                assert (
                    not self.self_attn.o_proj.reduce_results
                ), "short-circuiting allreduce will lead to hangs"
                return hidden_states, residual, extra_args if extra_args else {}

            # cuda(not hip) && absorb
            q_len = hidden_states.shape[0]
            q_input = hidden_states.new_empty(
                q_len, self.self_attn.num_local_heads, self.self_attn.kv_lora_rank + self.self_attn.qk_rope_head_dim
            )

            with torch.cuda.nvtx.range("q_projection"):
                if self.self_attn.q_lora_rank is not None:
                    q = self.self_attn.q_a_proj(hidden_states)[0]
                    q = self.self_attn.q_a_layernorm(q)
                    q = self.self_attn.q_b_proj(q)[0].view(-1, self.self_attn.num_local_heads,
                                                           self.self_attn.qk_head_dim)
                else:
                    q = self.self_attn.q_proj(hidden_states)[0].view(
                        -1, self.self_attn.num_local_heads, self.self_attn.qk_head_dim
                    )
                q_nope, q_pe = q.split([self.self_attn.qk_nope_head_dim, self.self_attn.qk_rope_head_dim], dim=-1)

            with torch.cuda.nvtx.range("w_kc_ops"):
                if self.self_attn.w_kc.dtype == torch.float8_e4m3fnuz:
                    q_nope_out = torch.bmm(
                        q_nope.to(torch.bfloat16).transpose(0, 1),
                        self.self_attn.w_kc.to(torch.bfloat16) * self.self_attn.w_scale,
                    )
                elif self.self_attn.w_kc.dtype == torch.float8_e4m3fn:
                    q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                        q_nope.transpose(0, 1),
                    )
                    q_nope_out = bmm_fp8(
                        q_nope_val, self.self_attn.w_kc, q_nope_scale, self.self_attn.w_scale, torch.bfloat16
                    )
                else:
                    q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.self_attn.w_kc)
                q_input[..., : self.self_attn.kv_lora_rank] = q_nope_out.transpose(0, 1)

            with torch.cuda.nvtx.range("kv_projection"):
                latent_cache = self.self_attn.kv_a_proj_with_mqa(hidden_states)[0]
                v_input = latent_cache[..., : self.self_attn.kv_lora_rank]
                v_input = self.self_attn.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
                k_input = latent_cache.unsqueeze(1)
                k_input[..., : self.self_attn.kv_lora_rank] = v_input
                k_pe = k_input[..., self.self_attn.kv_lora_rank:]

            with torch.cuda.nvtx.range("rope"):
                q_pe, k_pe = self.self_attn.rotary_emb(positions, q_pe, k_pe)
                q_input[..., self.self_attn.kv_lora_rank:] = q_pe
                k_input[..., self.self_attn.kv_lora_rank:] = k_pe

            # update extra_args
            extra_args.update(
                {
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_0_Q_INPUT_KEY: q_input,
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_0_K_INPUT_KEY: k_input,
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_0_V_INPUT_KEY: v_input,
                }
            )

            return hidden_states, residual, extra_args

    def forward_decode_attn_1(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        """
        note:
        1. core attention
        2. output projection
        3. MoE Gate routing
        """
        with torch.cuda.nvtx.range("decode_attn_1"):
            if hidden_states.shape[0] == 0:
                assert (
                    not self.self_attn.o_proj.reduce_results
                ), "short-circuiting allreduce will lead to hangs"
                return hidden_states, residual, extra_args if extra_args else {}

            # logger.debug(
            # f"layer[{self.layer_id}] in forward_decode_attn_1, hidden_states.shape:{hidden_states.shape}, residual: {residual.shape}, extra_args: {extra_args.keys()}")
            # get q_input, k_input and v_input from extra_args
            assert (
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_0_Q_INPUT_KEY in extra_args
            ), f"layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_0_Q_INPUT_KEY} not in extra_args:{extra_args}, hidden_states: {hidden_states}"
            assert (
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_0_K_INPUT_KEY in extra_args
            ), f"layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_0_K_INPUT_KEY} not in extra_args:{extra_args}, hidden_states: {hidden_states}"
            assert (
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_0_V_INPUT_KEY in extra_args
            ), f"layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_0_V_INPUT_KEY} not in extra_args:{extra_args}, hidden_states: {hidden_states}"
            q_input = extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_0_Q_INPUT_KEY]
            k_input = extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_0_K_INPUT_KEY]
            v_input = extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_0_V_INPUT_KEY]

            # def print_gpu_mem():
            #     # 检查 sglang 是否使用了 GPU
            #     if torch.cuda.is_available():
            #         device = torch.cuda.current_device()
            #         free_memory = torch.cuda.get_device_properties(
            #             device
            #         ).total_memory - torch.cuda.memory_allocated(device)
            #         # logger.debug(f"layer[{self.layer_id}], left gpu memory: {free_memory / 1024 ** 2:.2f} MB")

            # print_gpu_mem()

            # logger.debug(
            # f"layer[{self.layer_id}], before self.self_attn.attn_mqa, q_input: {q_input.shape}, k_input: {k_input.shape}, v_input: {v_input.shape}")
            # core attention calculation
            with torch.cuda.nvtx.range("attention"):
                attn_output = self.self_attn.attn_mqa(q_input, k_input, v_input, forward_batch)
                attn_output = attn_output.view(-1, self.self_attn.num_local_heads, self.self_attn.kv_lora_rank)

            with torch.cuda.nvtx.range("w_vc_ops"):
                if self.self_attn.w_vc.dtype == torch.float8_e4m3fnuz:
                    attn_bmm_output = torch.bmm(
                        attn_output.to(torch.bfloat16).transpose(0, 1),
                        self.self_attn.w_vc.to(torch.bfloat16) * self.self_attn.w_scale,
                    )
                elif self.self_attn.w_vc.dtype == torch.float8_e4m3fn:
                    attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                        attn_output.transpose(0, 1),
                    )
                    attn_bmm_output = bmm_fp8(
                        attn_output_val,
                        self.self_attn.w_vc,
                        attn_output_scale,
                        self.self_attn.w_scale,
                        torch.bfloat16,
                    )
                else:
                    attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.self_attn.w_vc)
                attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

            with torch.cuda.nvtx.range("output_projection"):
                hidden_states, _ = self.self_attn.o_proj(attn_output)

            with torch.cuda.nvtx.range("post_attention_layernorm"):
                # post attention layernorm
                if self.attn_tp_size != 1:
                    if self.input_is_scattered:
                        tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                        # #logger.debug(
                        #    f"layer[{self.layer_id}] after tensor_split & input_is_scatter, hidden_states.shape:{hidden_states.shape}")
                        hidden_states = tensor_list[self.attn_tp_rank]
                        # #logger.debug(
                        #    f"layer[{self.layer_id}] after tensor_list & input_is_scatter, hidden_states.shape:{hidden_states.shape}, tensor_list: {len(tensor_list)}, tensor_list: {[t.shape for t in tensor_list]}")
                        tp_reduce_scatter(hidden_states, tensor_list)
                        # #logger.debug(
                        #    f"layer[{self.layer_id}] after tp_reduce_scatter & input_is_scatter, hidden_states.shape:{hidden_states.shape}, tensor_list: {len(tensor_list)}")
                        if hidden_states.shape[0] != 0:
                            hidden_states, residual = self.post_attention_layernorm(
                                hidden_states, residual
                            )
                    else:
                        if self.attn_tp_rank == 0:
                            hidden_states += residual
                        tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                        # #logger.debug(
                        #    f"layer[{self.layer_id}] after tensor_split & not input_is_scatter, hidden_states.shape:{hidden_states.shape}, tensor_list: {len(tensor_list)}")
                        hidden_states = tensor_list[self.attn_tp_rank]
                        # #logger.debug(
                        #    f"layer[{self.layer_id}] after tensor_list & not input_is_scatter, hidden_states.shape:{hidden_states.shape}, tensor_list: {len(tensor_list)}")
                        tp_reduce_scatter(hidden_states, tensor_list)
                        # #logger.debug(
                        #    f"layer[{self.layer_id}] after tp_reduce_scatter & not input_is_scatter, hidden_states.shape:{hidden_states.shape}, tensor_list: {len(tensor_list)}")
                        residual = hidden_states
                        if hidden_states.shape[0] != 0:
                            hidden_states = self.post_attention_layernorm(hidden_states)
                else:
                    if hidden_states.shape[0] != 0:
                        hidden_states, residual = self.post_attention_layernorm(
                            hidden_states, residual
                        )

            with torch.cuda.nvtx.range("moe_gate_routing"):
                # MoE Gate routing
                topk_idx = torch.full(
                    (0, self.mlp.top_k), -1, dtype=torch.int, device=hidden_states.device
                )
                topk_weights = torch.empty(
                    (0, self.mlp.top_k), dtype=torch.float32, device=hidden_states.device
                )
                if hidden_states.shape[0] > 0:
                    # router_logits: (num_tokens, n_experts)
                    router_logits = self.mlp.gate(hidden_states)
                    topk_weights, topk_idx = select_experts(
                        hidden_states=hidden_states,
                        router_logits=router_logits,
                        top_k=self.mlp.top_k,
                        use_grouped_topk=True,
                        renormalize=self.mlp.renormalize,
                        topk_group=self.mlp.topk_group,
                        num_expert_group=self.mlp.num_expert_group,
                        correction_bias=self.mlp.correction_bias,
                    )

            del extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_0_Q_INPUT_KEY]
            del extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_0_K_INPUT_KEY]
            del extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_0_V_INPUT_KEY]

            extra_args.update(
                {
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_WEIGHTS_KEY: topk_weights,
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_IDX_KEY: topk_idx,
                }
            )

            return hidden_states, residual, extra_args

    def forward_mlp(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        # if get_attention_tp_rank() == 0:
        #     prev_cached_mem, prev_allocated_mem, prev_free_mem = get_gpu_memory_info_not_str(0)
        #     # get reorder_topk_ids, seg_indptr, masked_m, expected_m from extra_args
        assert (
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_REORDER_TOPK_IDS_KEY in extra_args
        ), f"layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_REORDER_TOPK_IDS_KEY} not in extra_args:{extra_args} "
        assert (
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_SEG_INDPTR_KEY in extra_args
        ), f"layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_SEG_INDPTR_KEY} not in extra_args:{extra_args}"
        assert (
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_MASKED_M_KEY in extra_args
        ), f"layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_MASKED_M_KEY} not in extra_args:{extra_args}"
        assert (
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_EXPECTED_M_KEY in extra_args
        ), f"layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_EXPECTED_M_KEY} not in extra_args:{extra_args}"

        reorder_topk_ids = extra_args[
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_REORDER_TOPK_IDS_KEY
        ]
        seg_indptr = extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_SEG_INDPTR_KEY]
        masked_m = extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_MASKED_M_KEY]
        expected_m = extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_EXPECTED_M_KEY]
        micro_batch_idx = extra_args[
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY
        ]
        mlp_output_tensor = extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_EMPTY_MLP_OUTPUT_KEY]

        # if get_attention_tp_rank() == 0:
        #     logger.debug(
        #         f"layer[{self.layer_id}]],forward_mlp,after get keys from dict,{get_gpu_memory_info_diff_str(0, prev_cached_mem, prev_allocated_mem, prev_free_mem)}")
        #     prev_cached_mem, prev_allocated_mem, prev_free_mem = get_gpu_memory_info_not_str(0)

        # self.mlp.experts.deepep_mode = DeepEPMode.normal if forward_batch.forward_mode.is_extend() else DeepEPMode.low_latency
        final_hidden_states = self.mlp.experts(
            hidden_states=hidden_states,
            reorder_topk_ids=reorder_topk_ids,
            seg_indptr=seg_indptr,
            masked_m=masked_m,
            expected_m=expected_m,
            forward_mode=forward_batch.forward_mode,
            output_tensor=mlp_output_tensor,
        )

        # if get_attention_tp_rank() == 0:
        #     logger.debug(
        #         f"layer[{self.layer_id}]],forward_mlp,after experts,after clear_tmp_data,{get_gpu_memory_info_diff_str(0, prev_cached_mem, prev_allocated_mem, prev_free_mem)}")
        #     prev_cached_mem, prev_allocated_mem, prev_free_mem = get_gpu_memory_info_not_str(0)

        # remove used args
        del extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_REORDER_TOPK_IDS_KEY]
        del extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_SEG_INDPTR_KEY]
        del extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_MASKED_M_KEY]
        del extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_EXPECTED_M_KEY]

        # if get_attention_tp_rank() == 0:
        #     logger.debug(
        #         f"layer[{self.layer_id}]],forward_mlp,after del,{get_gpu_memory_info_diff_str(0, prev_cached_mem, prev_allocated_mem, prev_free_mem)}")
        #     # prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)

        return final_hidden_states, residual, extra_args

    def forward_decode_launch_dispatch_ll(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        # get topk_ids and topk_weights
        assert (
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_IDX_KEY in extra_args
        ), f"layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_IDX_KEY} not in extra_args:{extra_args} "
        assert (
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_WEIGHTS_KEY in extra_args
        ), f"layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_WEIGHTS_KEY} not in extra_args:{extra_args} "

        topk_idx = extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_IDX_KEY]
        # logger.debug(f"forward_decode_launch_dispatch_ll topk_idx.dtype: {topk_idx.dtype}")
        topk_weights = extra_args[
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_WEIGHTS_KEY
        ]
        micro_batch_idx = extra_args[
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY
        ]
        # if get_attention_tp_rank() == 0 and self.layer_id > 4:
        #     acc_gpu_mem = 0
        #     for j in range(4, self.layer_id):
        #         t = self.mlp.deepep_dispatchers[get_role(forward_batch)][micro_batch_idx].get_hidden_states(
        #             forward_mode=forward_batch.forward_mode)
        #         acc_gpu_mem += t.numel() * t.element_size() if t is not None else 0
        #     logger.debug(f"layer[{self.layer_id}], micro_idx: {micro_batch_idx}, acc_gpu_mem: {acc_gpu_mem}")

        dispatched_hidden_states = self.mlp.deepep_dispatchers[get_role(forward_batch)][
            micro_batch_idx
        ].launch_dispatch(
            hidden_states,
            topk_idx,
            topk_weights,
            self.mlp.num_experts,
            forward_mode=forward_batch.forward_mode,
        )
        # logger.debug(f"forward_decode_launch_dispatch_ll after dispatch topk_idx.dtype: {topk_idx.dtype}")

        # update extra_args
        extra_args.update(
            {
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_WEIGHTS_KEY: topk_weights,
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_IDX_KEY: topk_idx,
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_BEFORE_DISPATCH_HIDDEN_STATES_KEY: hidden_states,
            }
        )

        return dispatched_hidden_states, residual, extra_args

    def forward_decode_wait_dispatch_ll(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        micro_batch_idx = extra_args[
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY
        ]
        (
            hidden_states,
            topk_idx,
            topk_weights,
            reorder_topk_ids,
            seg_indptr,
            masked_m,
            expected_m,
        ) = self.mlp.deepep_dispatchers[get_role(forward_batch)][
            micro_batch_idx
        ].wait_dispatch(
            forward_batch.forward_mode
        )

        # update extra_args
        extra_args.update(
            {
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_REORDER_TOPK_IDS_KEY: reorder_topk_ids,
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_SEG_INDPTR_KEY: seg_indptr,
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_MASKED_M_KEY: masked_m,
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_EXPECTED_M_KEY: expected_m,
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_IDX_KEY: topk_idx,
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_WEIGHTS_KEY: topk_weights,
            }
        )

        # if get_attention_tp_rank() == 0:
        #     logger.debug(f"layer[{self.layer_id}]],forward_decode,after clear_tmp_data,{get_gpu_memory_info(0)}")

        return hidden_states, residual, extra_args

    def forward_decode_launch_combine_ll(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        assert (
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_IDX_KEY in extra_args
        ), f"layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_IDX_KEY} not in extra_args:{extra_args} "
        assert (
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_WEIGHTS_KEY in extra_args
        ), f"layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_WEIGHTS_KEY} not in extra_args:{extra_args} "

        topk_idx = extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_IDX_KEY]
        # logger.debug(f"forward_decode_launch_combine_ll before launch_combine topk_idx.dtype: {topk_idx.dtype}")
        topk_weights = extra_args[
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_ATTN_1_TOPK_WEIGHTS_KEY
        ]
        micro_batch_idx = extra_args[
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY
        ]

        self.mlp.deepep_dispatchers[get_role(forward_batch)][
            micro_batch_idx
        ].launch_combine(
            hidden_states,
            topk_idx,
            topk_weights,
            forward_mode=forward_batch.forward_mode,
        )

        return hidden_states, residual, extra_args

    def forward_decode_wait_combine_ll(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        if self.mlp.ep_size > 1:
            micro_batch_idx = extra_args[
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY
            ]
            hidden_states = self.mlp.deepep_dispatchers[get_role(forward_batch)][
                micro_batch_idx
            ].wait_combine(forward_batch.forward_mode)
            hidden_states *= self.mlp.routed_scaling_factor
        return hidden_states, residual, extra_args

    def forward_prefill_self_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        if hidden_states.shape[0] == 0:
            residual = hidden_states
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.attn_tp_size != 1 and self.input_is_scattered:
            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            tp_all_gather(
                list(hidden_states.tensor_split(self.attn_tp_size)), local_hidden_states
            )
        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        # add residual
        if self.attn_tp_size != 1:
            if self.input_is_scattered:
                tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                hidden_states = tensor_list[self.attn_tp_rank]
                # #logger.debug(f"layer[{self.layer_id}] before tp_reduce_scatter & input_is_scatter & prefill, hidden_states.shape:{hidden_states.shape}, tensor_list: {len(tensor_list)}")
                tp_reduce_scatter(hidden_states, tensor_list)
                # #logger.debug(f"layer[{self.layer_id}] after tp_reduce_scatter & input_is_scatter & prefill, hidden_states.shape:{hidden_states.shape}, tensor_list: {len(tensor_list)}")
                if hidden_states.shape[0] != 0:
                    hidden_states, residual = self.post_attention_layernorm(
                        hidden_states, residual
                    )
            else:
                if self.attn_tp_rank == 0:
                    hidden_states += residual
                tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                hidden_states = tensor_list[self.attn_tp_rank]
                # #logger.debug(f"layer[{self.layer_id}] before tp_reduce_scatter & not input_is_scatter & prefill, hidden_states.shape:{hidden_states.shape}, tensor_list: {len(tensor_list)}")
                tp_reduce_scatter(hidden_states, tensor_list)
                # #logger.debug(f"layer[{self.layer_id}] after tp_reduce_scatter & not input_is_scatter & prefill, hidden_states.shape:{hidden_states.shape}, tensor_list: {len(tensor_list)}")
                residual = hidden_states
                if hidden_states.shape[0] != 0:
                    hidden_states = self.post_attention_layernorm(hidden_states)
        else:
            if hidden_states.shape[0] != 0:
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )
        return hidden_states, residual, extra_args

    def forward_launch_dispatch_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        if self.mlp.ep_size > 1:
            assert (
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_IDX_KEY in extra_args
            ), f"[forward_launch_dispatch_normal] layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_IDX_KEY} not in extra_args:{extra_args} "

            assert (
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_WEIGHTS_KEY in extra_args
            ), f"[forward_launch_dispatch_normal] layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_WEIGHTS_KEY} not in extra_args:{extra_args} "

            assert (
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY in extra_args
            ), f"[forward_launch_dispatch_normal] layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY} not in extra_args:{extra_args} "

            # TODO(ch-wan): allow users to set num_max_dispatch_tokens_per_rank value
            micro_batch_idx = extra_args[
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY
            ]
            self.mlp.deepep_dispatchers[get_role(forward_batch)][
                micro_batch_idx
            ].launch_dispatch(
                hidden_states,
                extra_args.get(MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_IDX_KEY),
                extra_args.get(MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_WEIGHTS_KEY),
                self.mlp.num_experts,
                forward_mode=forward_batch.forward_mode,
            )

        return hidden_states, residual, extra_args

    def forward_wait_dispatch_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        if self.mlp.ep_size > 1:
            assert (
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY in extra_args
            ), f"[forward_launch_dispatch_normal] layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY} not in extra_args:{extra_args} "
            micro_batch_idx = extra_args[
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY
            ]
            (
                after_dispatch_hidden_states,
                topk_idx,
                topk_weights,
                reorder_topk_ids,
                seg_indptr,
                masked_m,
                expected_m,
            ) = self.mlp.deepep_dispatchers[get_role(forward_batch)][
                micro_batch_idx
            ].wait_dispatch(
                forward_batch.forward_mode
            )

            # update extra_args
            extra_args.update(
                {
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_IDX_KEY: topk_idx,
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_WEIGHTS_KEY: topk_weights,
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_REORDER_TOPK_IDS_KEY: reorder_topk_ids,
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_SEG_INDPTR_KEY: seg_indptr,
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_MASKED_M_KEY: masked_m,
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_EXPECTED_M_KEY: expected_m,
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_BEFORE_DISPATCH_HIDDEN_STATES_KEY: hidden_states,
                }
            )
        return after_dispatch_hidden_states, residual, extra_args

    def forward_launch_combine_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        if self.mlp.ep_size > 1:
            assert (
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_IDX_KEY in extra_args
            ), f"[forward_launch_combine_normal] layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_IDX_KEY} not in extra_args:{extra_args} "

            assert (
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_WEIGHTS_KEY in extra_args
            ), f"[forward_launch_combine_normal] layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_WEIGHTS_KEY} not in extra_args:{extra_args} "

            assert (
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY in extra_args
            ), f"[forward_launch_dispatch_normal] layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY} not in extra_args:{extra_args} "

            micro_batch_idx = extra_args[
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY
            ]
            self.mlp.deepep_dispatchers[get_role(forward_batch)][
                micro_batch_idx
            ].launch_combine(
                hidden_states,
                extra_args.get(MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_IDX_KEY),
                extra_args.get(MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_WEIGHTS_KEY),
                forward_mode=forward_batch.forward_mode,
            )
        # every layer should remove topk_idx and topk_weights after combine
        del extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_IDX_KEY]
        del extra_args[MicroBatchOverlapExtraArgs.EXTRA_ARGS_TOPK_WEIGHTS_KEY]
        return hidden_states, residual, extra_args

    def forward_wait_combine_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        if self.mlp.ep_size > 1:
            assert (
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY in extra_args
            ), f"[forward_launch_dispatch_normal] layer_id: {self.layer_id}, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY} not in extra_args:{extra_args} "

            micro_batch_idx = extra_args[
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY
            ]
            hidden_states = self.mlp.deepep_dispatchers[get_role(forward_batch)][
                micro_batch_idx
            ].wait_combine(forward_batch.forward_mode)
            hidden_states *= self.mlp.routed_scaling_factor
        return hidden_states, residual, extra_args

    def forward_step(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        step: Optional[MicroBatchOverlapStep] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        step_to_func = {
            MicroBatchOverlapStep.MOE_GATE: self.forward_moe_gate,
            MicroBatchOverlapStep.SHARED_EXPERTS: self.forward_shared_experts,
            MicroBatchOverlapStep.MLP: self.forward_mlp,
            # prefill attn
            MicroBatchOverlapStep.PREFILL_ATTN: self.forward_prefill_self_attn,
            # decode attn
            MicroBatchOverlapStep.DECODE_ATTN_0_STEP: self.forward_decode_attn_0,
            MicroBatchOverlapStep.DECODE_ATTN_1_STEP: self.forward_decode_attn_1,
            # normal deepep
            MicroBatchOverlapStep.LAUNCH_DISPATCH_NORMAL: self.forward_launch_dispatch_normal,
            MicroBatchOverlapStep.WAIT_DISPATCH_NORMAL: self.forward_wait_dispatch_normal,
            MicroBatchOverlapStep.LAUNCH_COMBINE_NORMAL: self.forward_launch_combine_normal,
            MicroBatchOverlapStep.WAIT_COMBINE_NORMAL: self.forward_wait_combine_normal,
            # low latency deepep
            MicroBatchOverlapStep.LAUNCH_DISPATCH_LL_STEP: self.forward_decode_launch_dispatch_ll,
            MicroBatchOverlapStep.WAIT_DISPATCH_LL_STEP: self.forward_decode_wait_dispatch_ll,
            MicroBatchOverlapStep.LAUNCH_COMBINE_LL_STEP: self.forward_decode_launch_combine_ll,
            MicroBatchOverlapStep.WAIT_COMBINE_LL_STEP: self.forward_decode_wait_combine_ll,
        }
        if step not in step_to_func:
            raise f"{step} func does not implemented"
        return step_to_func[step](
            positions, hidden_states, forward_batch, residual, extra_args
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        step: Optional[str] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        if global_server_args_dict["enable_deepep_moe"] and self.is_sparse:
            with torch.cuda.nvtx.range("decode_layer_forward_deepep"):
                return self.forward_deepep(
                    positions, hidden_states, forward_batch, residual, step, extra_args
                )
        else:
            with torch.cuda.nvtx.range("decode_layer_forward_normal"):
                return self.forward_normal(
                    positions, hidden_states, forward_batch, residual
                )

    def forward_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:

        # if get_attention_tp_rank()==0:
        #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)

        if hidden_states.shape[0] == 0:
            residual = hidden_states
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

            assert not (
                self.attn_tp_size != 1 and self.input_is_scattered
            ), "moe_layer_freq > 1 is not supported when attn_tp_size > 1"

            # Self Attention
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

            # if get_attention_tp_rank()==0:
            #     logger.debug(f"layer[{self.layer_id}]],original_normal,after self_attention,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)

        # Gather
        if get_tensor_model_parallel_world_size() > 1:
            # all gather and all reduce
            if self.dp_size != 1:
                # if not global_server_args_dict["enable_deepep_moe"]:
                #     if self.attn_tp_rank == 0:
                #         hidden_states += residual
                #     hidden_states, local_hidden_states = (
                #         forward_batch.gathered_buffer,
                #         hidden_states,
                #     )
                #     dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
                #     dp_scatter(residual, hidden_states, forward_batch)
                #     hidden_states = self.post_attention_layernorm(hidden_states)
                # else:
                #     if self.attn_tp_size != 0:
                #         hidden_states = tp_all_reduce(hidden_states)
                #     if hidden_states.shape[0] != 0:
                #         hidden_states, residual = self.post_attention_layernorm(
                #             hidden_states, residual
                #         )
                if self.attn_tp_rank == 0:
                    hidden_states += residual
                hidden_states, local_hidden_states = (
                    forward_batch.gathered_buffer,
                    hidden_states,
                )
                dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
                dp_scatter(residual, hidden_states, forward_batch)
                hidden_states = self.post_attention_layernorm(hidden_states)
            else:
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        # if get_attention_tp_rank()==0:
        #     logger.debug(f"layer[{self.layer_id}]],original_normal,after post_attention,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
        #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        # TODO(ch-wan): ues reduce-scatter in MLP to avoid this scatter
        # Scatter
        if self.dp_size != 1:
            # important: forward batch.gathered_buffer is used both after scatter and after gather.
            # be careful about this!
            hidden_states, global_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            dp_scatter(hidden_states, global_hidden_states, forward_batch)

        # if get_attention_tp_rank()==0:
        #     logger.debug(f"layer[{self.layer_id}]],original_normal,after total,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
        #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)

        return hidden_states, residual, None

    def forward_deepep(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        step: Optional[str] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        if step:
            return self.forward_step(
                positions, hidden_states, forward_batch, residual, step, extra_args
            )

        # if get_attention_tp_rank()==0:
        #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)

        if hidden_states.shape[0] == 0:
            residual = hidden_states
        else:
            with torch.cuda.nvtx.range("layer_norm"):
                if residual is None:
                    residual = hidden_states
                    hidden_states = self.input_layernorm(hidden_states)
                else:
                    hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.attn_tp_size != 1 and self.input_is_scattered:
            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            with torch.cuda.nvtx.range("all_gather"):
                tp_all_gather(
                    list(hidden_states.tensor_split(self.attn_tp_size)), local_hidden_states
                )
        with torch.cuda.nvtx.range("self_attention"):
            # Self Attention
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        # if get_attention_tp_rank()==0:
        #     logger.debug(f"layer[{self.layer_id}]],original,after self_attention,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
        #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)

        if self.attn_tp_size != 1:
            if self.input_is_scattered:
                with torch.cuda.nvtx.range("reduce_scatter"):
                    tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                    hidden_states = tensor_list[self.attn_tp_rank]
                    tp_reduce_scatter(hidden_states, tensor_list)
                if hidden_states.shape[0] != 0:
                    with torch.cuda.nvtx.range("post_attn_norm"):
                        hidden_states, residual = self.post_attention_layernorm(
                            hidden_states, residual
                        )
            else:
                if self.attn_tp_rank == 0:
                    with torch.cuda.nvtx.range("add_residual"):
                        hidden_states += residual
                with torch.cuda.nvtx.range("reduce_scatter"):
                    tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                    hidden_states = tensor_list[self.attn_tp_rank]
                    tp_reduce_scatter(hidden_states, tensor_list)
                residual = hidden_states
                if hidden_states.shape[0] != 0:
                    with torch.cuda.nvtx.range("post_attn_norm"):
                        hidden_states = self.post_attention_layernorm(hidden_states)
        else:
            if hidden_states.shape[0] != 0:
                with torch.cuda.nvtx.range("post_attn_norm"):
                    hidden_states, residual = self.post_attention_layernorm(
                        hidden_states, residual
                    )
        # if get_attention_tp_rank()==0:
        #     logger.debug(f"layer[{self.layer_id}]],original,after post attention,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
        #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)

        with torch.cuda.nvtx.range("mlp"):
            hidden_states = self.mlp(hidden_states, forward_batch.forward_mode)

        # if get_attention_tp_rank()==0:
        #     logger.debug(f"layer[{self.layer_id}]],original,after mlp,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
        #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)

        if self.is_last_layer and self.attn_tp_size != 1:
            with torch.cuda.nvtx.range("last_layer_ops"):
                hidden_states += residual
                residual = None
                hidden_states, local_hidden_states = (
                    forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                    hidden_states,
                )
                with torch.cuda.nvtx.range("final_all_gather"):
                    tp_all_gather(
                        list(hidden_states.tensor_split(self.attn_tp_size)), local_hidden_states
                    )
        # if get_attention_tp_rank()==0:
        #     logger.debug(f"layer[{self.layer_id}]],original,after total,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
        #     #prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)

        return hidden_states, residual, None


def not_match_mode(fwd_mode: ForwardMode, role: str) -> bool:
    if fwd_mode.is_extend() and role == "decode":
        return True
    if fwd_mode.is_decode() and role == "prefill":
        return True


class DeepseekV2Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.dp_size = get_attention_dp_size()
        self.config = config

        # initialize mb0 and mb1 tensor
        ep_size = get_tensor_model_parallel_world_size()
        num_experts = config.n_routed_experts
        assert num_experts % ep_size == 0, f"num_experts[{config.n_routed_experts}]%ep_size[{ep_size}] must be 0"
        self.top_k = config.num_experts_per_tok
        # constraints: low_latency && decode
        # shape=[local_num_experts, num_max_dispatch_tokens_per_rank*num_ranks, hidden_size]
        self.mb0_mlp_output = torch.empty(
            (num_experts // ep_size, 128 * parallel_state.get_tp_group().world_size, config.hidden_size),
            device=torch.cuda.current_device(), dtype=torch.bfloat16
        )
        self.mb1_mlp_output = torch.empty(
            (num_experts // ep_size, 128 * parallel_state.get_tp_group().world_size, config.hidden_size),
            device=torch.cuda.current_device(), dtype=torch.bfloat16
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            with torch.cuda.nvtx.range("embedding_tokens"):
                hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        if (
            forward_batch.batch_size == 1
            or not global_server_args_dict["enable_micro_batch_overlap"]
            or forward_batch.forward_mode.is_extend()
            # or not_match_mode(forward_batch.forward_mode,global_server_args_dict["role"])
        ):
            for i in range(len(self.layers)):
                # logger.debug(
                # f"layer[{i}], bs: {forward_batch.batch_size}, hiddens_states.shape: {hidden_states.shape}")
                expert_distribution_recorder.set_current_layer(i)
                layer = self.layers[i]
                hidden_states, residual, _ = layer(
                    positions, hidden_states, forward_batch, residual
                )
        else:
            # first k dense layers
            for i in range(self.config.first_k_dense_replace):
                expert_distribution_recorder.set_current_layer(i)
                layer = self.layers[i]
                hidden_states, residual, _ = layer(
                    positions, hidden_states, forward_batch, residual
                )

            if forward_batch.forward_mode == ForwardMode.EXTEND:
                hidden_states, residual = self.forward_prefill(
                    hidden_states,
                    positions,
                    forward_batch,
                    residual,
                )
            elif forward_batch.forward_mode == ForwardMode.DECODE:
                hidden_states, residual = self.forward_decode(
                    hidden_states,
                    positions,
                    forward_batch,
                    residual,
                )
            else:
                raise ValueError(
                    f"Unsupported forward mode: {forward_batch.forward_mode}"
                )

        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """
        # prefill primary step
        attn_step = "attn"
        moe_gate_step = "moe_gate"
        shared_experts_step = "shared_experts"
        mlp_step = "mlp"
        wait_normal_dispatch_step = "wait_normal_dispatch"
        wait_normal_combine_step = "wait_normal_combine"
        launch_normal_dispatch_step = "launch_normal_dispatch"
        launch_normal_combine_step = "launch_normal_combine"
        """

        # split forward_batch into two micro-batches
        bs_joint_batch_boundary, fwd_batch0, fwd_batch1 = token_balanced_batch_split(
            forward_batch
        )
        hidden_states_0 = hidden_states[0:bs_joint_batch_boundary]
        hidden_states_1 = hidden_states[bs_joint_batch_boundary:]
        positions_0 = positions[0:bs_joint_batch_boundary]
        positions_1 = positions[bs_joint_batch_boundary:]
        residual_0 = residual[0:bs_joint_batch_boundary]
        residual_1 = residual[bs_joint_batch_boundary:]
        extra_args_0, extra_args_1 = {
                                         MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY: 0
                                     }, {MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY: 1}
        l0, l1 = (
            self.config.first_k_dense_replace - 1,
            self.config.first_k_dense_replace,
        )
        # init attn_backend metadata again
        fwd_batch0.attn_backend.init_forward_metadata(fwd_batch0)
        fwd_batch1.attn_backend.init_forward_metadata(fwd_batch1)

        # if get_tensor_model_parallel_rank() == 0:
        # logger.debug(f'********* {forward_batch=}')
        # logger.debug(f'********* {fwd_batch0=}')
        # logger.debug(f'********* {fwd_batch1=}')
        # last moe layer
        for i in range(self.config.first_k_dense_replace, len(self.layers) + 1):
            # logger.debug(f'$$$$$$ {hidden_states_0.shape=}, {hidden_states_1.shape=}')
            # overlap b1 attn and b0 shared_experts with b0 combine
            hidden_states_0, residual_0, extra_args_0 = self.forward_layer(
                l0,
                positions_0,
                hidden_states_0,
                fwd_batch0,
                residual_0,
                MicroBatchOverlapStep.LAUNCH_COMBINE_NORMAL,
                extra_args_0,
            )
            hidden_states_1, residual_1, extra_args_1 = self.forward_layer(
                l1,
                positions_1,
                hidden_states_1,
                fwd_batch1,
                residual_1,
                MicroBatchOverlapStep.PREFILL_ATTN,
                extra_args_1,
            )
            hidden_states_1, residual_1, extra_args_1 = self.forward_layer(
                l1,
                positions_1,
                hidden_states_1,
                fwd_batch1,
                residual_1,
                MicroBatchOverlapStep.MOE_GATE,
                extra_args_1,
            )

            after_dispatch_hidden_states_0 = extra_args_0.get(
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_BEFORE_DISPATCH_HIDDEN_STATES_KEY
            )

            _, residual_0, extra_args_0 = self.forward_layer(
                l0,
                positions_0,
                after_dispatch_hidden_states_0,
                fwd_batch0,
                residual_0,
                MicroBatchOverlapStep.SHARED_EXPERTS,
                extra_args_0,
            )
            hidden_states_0, residual_0, _ = self.forward_layer(
                l0,
                positions_0,
                hidden_states_0,
                fwd_batch0,
                residual_0,
                MicroBatchOverlapStep.WAIT_COMBINE_NORMAL,
                extra_args_0,
            )
            # add shared experts
            if l0 >= self.config.first_k_dense_replace:
                shared_experts_output_0 = extra_args_0.get(
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_SHARED_EXPERT_OUTPUT_KEY
                )
                if shared_experts_output_0 is not None:
                    hidden_states_0 += shared_experts_output_0
                    del extra_args_0[
                        MicroBatchOverlapExtraArgs.EXTRA_ARGS_SHARED_EXPERT_OUTPUT_KEY
                    ]

            l0 += 1
            # overlap b0 dispatch with b1 attn
            hidden_states_1, residual_1, extra_args_1 = self.forward_layer(
                l1,
                positions_1,
                hidden_states_1,
                fwd_batch1,
                residual_1,
                MicroBatchOverlapStep.LAUNCH_DISPATCH_NORMAL,
                extra_args_1,
            )
            hidden_states_0, residual_0, extra_args_0 = self.forward_layer(
                l0,
                positions_0,
                hidden_states_0,
                fwd_batch0,
                residual_0,
                MicroBatchOverlapStep.PREFILL_ATTN,
                extra_args_0,
            )

            hidden_states_0, residual_0, extra_args_0 = self.forward_layer(
                l0,
                positions_0,
                hidden_states_0,
                fwd_batch0,
                residual_0,
                MicroBatchOverlapStep.MOE_GATE,
                extra_args_0,
            )
            hidden_states_0, residual_0, extra_args_0 = self.forward_layer(
                l0,
                positions_0,
                hidden_states_0,
                fwd_batch0,
                residual_0,
                MicroBatchOverlapStep.LAUNCH_DISPATCH_NORMAL,
                extra_args_0,
            )
            hidden_states_1, residual_1, extra_args_1 = self.forward_layer(
                l1,
                positions_1,
                hidden_states_1,
                fwd_batch1,
                residual_1,
                MicroBatchOverlapStep.WAIT_DISPATCH_NORMAL,
                extra_args_1,
            )
            hidden_states_1, residual_1, extra_args_1 = self.forward_layer(
                l1,
                positions_1,
                hidden_states_1,
                fwd_batch1,
                residual_1,
                MicroBatchOverlapStep.MLP,
                extra_args_1,
            )

            hidden_states_1, residual_1, extra_args_1 = self.forward_layer(
                l1,
                positions_1,
                hidden_states_1,
                fwd_batch1,
                residual_1,
                MicroBatchOverlapStep.LAUNCH_COMBINE_NORMAL,
                extra_args_1,
            )
            hidden_states_0, residual_0, extra_args_0 = self.forward_layer(
                l0,
                positions_0,
                hidden_states_0,
                fwd_batch0,
                residual_0,
                MicroBatchOverlapStep.WAIT_DISPATCH_NORMAL,
                extra_args_0,
            )
            hidden_states_0, residual_0, extra_args_0 = self.forward_layer(
                l0,
                positions_0,
                hidden_states_0,
                fwd_batch0,
                residual_0,
                MicroBatchOverlapStep.MLP,
                extra_args_0,
            )

            after_dispatch_hidden_states_1 = extra_args_1.get(
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_BEFORE_DISPATCH_HIDDEN_STATES_KEY
            )

            _, residual_1, extra_args_1 = self.forward_layer(
                l1,
                positions_1,
                after_dispatch_hidden_states_1,
                fwd_batch1,
                residual_1,
                MicroBatchOverlapStep.SHARED_EXPERTS,
                extra_args_1,
            )

            hidden_states_1, residual_1, extra_args_1 = self.forward_layer(
                l1,
                positions_1,
                hidden_states_1,
                fwd_batch1,
                residual_1,
                MicroBatchOverlapStep.WAIT_COMBINE_NORMAL,
                extra_args_1,
            )

            # add shared experts
            if l1 < len(self.layers):
                shared_experts_output_1 = extra_args_1.get(
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_SHARED_EXPERT_OUTPUT_KEY
                )
                if shared_experts_output_1 is not None:
                    hidden_states_1 += shared_experts_output_1
                    del extra_args_1[
                        MicroBatchOverlapExtraArgs.EXTRA_ARGS_SHARED_EXPERT_OUTPUT_KEY
                    ]
            l1 += 1

        # check last_layer
        # #logger.debug(f"~~~~~~~~~~ after self.forward_step, step: {step},batch_size: {forward_batch.batch_size},layer_id:{self.layer_id}")
        # #logger.debug(f'~~~~~~~~~~ after self.forward_step, hidden_states.shape: {hidden_states.shape}, \
        #                  residual.shape:{residual.shape}')
        # #logger.debug(f'~~~~~~~~~~ after self.forward_step, hidden_states.shape: {hidden_states.shape}, \
        #                  residual_after.shape:{residual_after.shape}')
        attn_tp_size = get_attention_tp_size()
        if attn_tp_size != 1:
            hidden_states_0 += residual_0
            # residual_after = None # note: never set None to residual, this is different to official
            hidden_states_0, local_hidden_states_0 = (
                fwd_batch0.gathered_buffer[: fwd_batch0.input_ids.shape[0]],
                hidden_states_0,
            )
            tp_all_gather(
                list(hidden_states_0.tensor_split(attn_tp_size)), local_hidden_states_0
            )
            hidden_states_1 += residual_1
            # residual_after = None # note: never set None to residual, this is different to official
            hidden_states_1, local_hidden_states_1 = (
                fwd_batch1.gathered_buffer[: fwd_batch1.input_ids.shape[0]],
                hidden_states_1,
            )
            tp_all_gather(
                list(hidden_states_1.tensor_split(attn_tp_size)), local_hidden_states_1
            )
        # logger.debug(
        # f'~~~~~~~~~~ before layers,{hidden_states_0.shape=},{hidden_states_1.shape=},{residual_0.shape=},{residual_1.shape=}')
        hidden_states = torch.cat([hidden_states_0, hidden_states_1], dim=0)
        # note: keep same with python/sglang/srt/models/deepseek_v2.py:1247(official v0.4.5 implementation)
        # residual = torch.cat([residual_0, residual_1], dim=0)
        residual = None
        # logger.debug(f'~~~~~~~~~~ after layers, {hidden_states.shape=}')

        return hidden_states, residual

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """
        # decode primary step
        attn_0_step = "attn_0"
        attn_1_step = "attn_1"
        moe_gate_step = "moe_gate"
        shared_experts_step = "shared_experts"
        mlp_step = "mlp"
        wait_ll_dispatch_step = "wait_ll_dispatch"
        wait_ll_combine_step = "wait_ll_combine"
        launch_ll_dispatch_step = "launch_ll_dispatch"
        launch_ll_combine_step = "launch_ll_combine"
        """
        # logger.debug(f"forward_decode in two micro batch")
        # if get_attention_tp_rank()==0:
        # logger.debug(f"forward_decode start: forward_batch, attn_backend: {forward_batch.attn_backend}, attn_backend1: {forward_batch.attn_backend1}")
        bs_joint_batch_boundary, fwd_batch0, fwd_batch1 = token_balanced_batch_split(
            forward_batch
        )
        # if get_attention_tp_rank()==0:
        # logger.debug(f"forward_decode start: fwd_batch0, attn_backend: {fwd_batch0.attn_backend}, attn_backend1: {fwd_batch0.attn_backend1}")
        # logger.debug(f"forward_decode start: fwd_batch1, attn_backend: {fwd_batch1.attn_backend}, attn_backend1: {fwd_batch1.attn_backend1}")
        mb0_hidden_states = hidden_states[0:bs_joint_batch_boundary]
        mb1_hidden_states = hidden_states[bs_joint_batch_boundary:]
        mb0_positions = positions[0:bs_joint_batch_boundary]
        mb1_positions = positions[bs_joint_batch_boundary:]
        mb0_residual = residual[0:bs_joint_batch_boundary]
        mb1_residual = residual[bs_joint_batch_boundary:]
        mb0_extra_args = {
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY: 0,
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_EMPTY_MLP_OUTPUT_KEY: self.mb0_mlp_output,
        }
        mb1_extra_args = {
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_IDX_KEY: 1,
            MicroBatchOverlapExtraArgs.EXTRA_ARGS_MICRO_BATCH_EMPTY_MLP_OUTPUT_KEY: self.mb0_mlp_output,
        }
        l0, l1 = (
            self.config.first_k_dense_replace - 1,
            self.config.first_k_dense_replace,
        )
        # init attn_backend metadata again
        # if get_attention_tp_rank()==0:
        # logger.debug(f"[token_balanced_batch_split] fwd_batch0: {fwd_batch0}")
        # logger.debug(f"[token_balanced_batch_split] fwd_batch1: {fwd_batch1}")
        fwd_batch0.attn_backend.init_forward_metadata(fwd_batch0)
        # if get_attention_tp_rank()==0:
        # logger.debug(f"fwd_batch0.attn_backend.metdata: {fwd_batch0.attn_backend.forward_metadata}")
        fwd_batch1.attn_backend = forward_batch.attn_backend1
        fwd_batch1.attn_backend.init_forward_metadata(fwd_batch1)
        # if get_attention_tp_rank()==0:
        # logger.debug(f"fwd_batch1.attn_backend.metdata: {fwd_batch1.attn_backend.forward_metadata}")

        for i in range(self.config.first_k_dense_replace, len(self.layers) + 1):
            # debug
            # if i<=6:
            #     torch.cuda.memory._record_memory_history()
            # else:
            #     torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
            #     raise "collect enough"

            expert_distribution_recorder.set_current_layer(i)
            # pipeline two micro batches according to https://github.com/deepseek-ai/profile-data
            # logger.debug(f'~~~~~~~~~~ begin to run decode layer[{i}], {hidden_states=}, {residual=}')

            # mb0: launch_dispatch
            mb0_hidden_states, _, mb0_extra_args = self.forward_layer(
                l0,
                mb0_positions,
                mb0_hidden_states,
                fwd_batch0,
                mb0_residual,
                MicroBatchOverlapStep.LAUNCH_DISPATCH_LL_STEP,
                mb0_extra_args,
            )

            # if get_attention_tp_rank()==0:
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)

            # if get_attention_tp_rank()==0:
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 LAUNCH_DISPATCH_LL_STEP,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)
            # logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 LAUNCH_DISPATCH_LL_STEP,{get_gpu_memory_info(0)}")
            # if True or get_attention_tp_rank() in [0, 3, 4]:
            # logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb0 LAUNCH_DISPATCH_LL_STEP: \
            # mb0_hidden_states.shape: {mb0_hidden_states.shape}, \
            # mb0_residual: {mb0_residual.shape} \
            # mb0_extra_args: {mb0_extra_args.keys()} \
            # mb1_hidden_states.shape: {mb1_hidden_states.shape}, \
            # mb1_residual: {mb1_residual.shape} \
            # mb1_extra_args: {mb1_extra_args.keys()}")
            # mb0: shared_expert_output
            mb0_before_dispatch_hidden_states = mb0_extra_args.get(
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_BEFORE_DISPATCH_HIDDEN_STATES_KEY,
                None,
            )
            # note: use _ rather than mb0_before_dispatch_hidden_states due to shared_export layer will return mb0_before_dispatch_hidden_states
            _, mb0_residual, mb0_extra_args = self.forward_layer(
                l0,
                mb0_positions,
                mb0_before_dispatch_hidden_states,
                fwd_batch0,
                mb0_residual,
                MicroBatchOverlapStep.SHARED_EXPERTS,
                mb0_extra_args,
            )
            # if get_attention_tp_rank()==0:
            #     #logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 SHARED_EXPERTS,{get_gpu_memory_info(0)}")
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 SHARED_EXPERTS,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)
            # if True or get_attention_tp_rank() in [0, 3, 4]:
            # logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb0 SHARED_EXPERTS: \
            # mb0_hidden_states.shape: {mb0_hidden_states.shape}, \
            # mb0_residual: {mb0_residual.shape} \
            # mb0_extra_args: {mb0_extra_args.keys()} \
            # mb1_hidden_states.shape: {mb1_hidden_states.shape}, \
            # mb1_residual: {mb1_residual.shape} \
            # mb1_extra_args: {mb1_extra_args.keys()}")

            # mb1: attn_0
            mb1_hidden_states, mb1_residual, mb1_extra_args = self.forward_layer(
                l1,
                mb1_positions,
                mb1_hidden_states,
                fwd_batch1,
                mb1_residual,
                MicroBatchOverlapStep.DECODE_ATTN_0_STEP,
                mb1_extra_args,
            )
            # if get_attention_tp_rank()==0:
            #     #logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 DECODE_ATTN_0_STEP,{get_gpu_memory_info(0)}")
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 DECODE_ATTN_0_STEP,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)
            # if True or get_attention_tp_rank() in [0, 3, 4]:
            # logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb1 DECODE_ATTN_0_STEP: \
            # mb0_hidden_states.shape: {mb0_hidden_states.shape}, \
            # mb0_residual: {mb0_residual.shape} \
            # mb0_extra_args: {mb0_extra_args.keys()} \
            # mb1_hidden_states.shape: {mb1_hidden_states.shape}, \
            # mb1_residual: {mb1_residual.shape} \
            # mb1_extra_args: {mb1_extra_args.keys()}")
            # mb0: wait_dispatch
            mb0_hidden_states, mb0_residual, mb0_extra_args = self.forward_layer(
                l0,
                mb0_positions,
                mb0_hidden_states,
                fwd_batch0,
                mb0_residual,
                MicroBatchOverlapStep.WAIT_DISPATCH_LL_STEP,
                mb0_extra_args,
            )
            # if get_attention_tp_rank()==0:
            #     #logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 WAIT_DISPATCH_LL_STEP,{get_gpu_memory_info(0)}")
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 WAIT_DISPATCH_LL_STEP,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)
            # if True or get_attention_tp_rank() in [0]:
            # logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb0 WAIT_DISPATCH_LL_STEP: \
            # mb0_hidden_states.shape: {mb0_hidden_states[0].shape if len(mb0_hidden_states) > 1 else mb0_hidden_states.shape}, \
            # mb0_residual: {mb0_residual.shape} \
            # mb0_extra_args: {mb0_extra_args.keys()} \
            # mb1_hidden_states.shape: {mb1_hidden_states.shape}, \
            # mb1_residual: {mb1_residual.shape} \
            # mb1_extra_args: {mb1_extra_args.keys()}")
            # mb0: mlp
            mb0_hidden_states, mb0_residual, mb0_extra_args = self.forward_layer(
                l0,
                mb0_positions,
                mb0_hidden_states,
                fwd_batch0,
                mb0_residual,
                MicroBatchOverlapStep.MLP,
                mb0_extra_args,
            )
            # if get_attention_tp_rank()==0:
            #     #logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 MLP,{get_gpu_memory_info(0)}")
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 MLP,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)
            # if True or get_attention_tp_rank() in [0]:
            # logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb0 MLP: \
            # mb0_hidden_states.shape: {mb0_hidden_states.shape}, \
            # mb0_residual: {mb0_residual.shape} \
            # mb0_extra_args: {mb0_extra_args.keys()} \
            # mb1_hidden_states.shape: {mb1_hidden_states.shape}, \
            # mb1_residual: {mb1_residual.shape} \
            # mb1_extra_args: {mb1_extra_args.keys()}")
            # mb0: launch_combine
            mb0_hidden_states, _, mb0_extra_args = self.forward_layer(
                l0,
                mb0_positions,
                mb0_hidden_states,
                fwd_batch0,
                mb0_residual,
                MicroBatchOverlapStep.LAUNCH_COMBINE_LL_STEP,
                mb0_extra_args,
            )
            # note: mb0_hidden_states does not need because new mb0_hidden_states will be set after WAIT_COMBINE_LL_STEP
            # del mb0_hidden_states
            # mb0_hidden_states=None
            # if get_attention_tp_rank()==0:
            #     #logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 LAUNCH_COMBINE_LL_STEP,{get_gpu_memory_info(0)}")
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 LAUNCH_COMBINE_LL_STEP,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)
            # if True or get_attention_tp_rank() in [0]:
            #     #logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb0 LAUNCH_COMBINE_LL_STEP: \
            #                  mb0_hidden_states.shape: {mb0_hidden_states.shape}, \
            #                  mb0_residual: {mb0_residual.shape} \
            #                  mb0_extra_args: {mb0_extra_args.keys()} \
            #                  mb1_hidden_states.shape: {mb1_hidden_states.shape}, \
            #                  mb1_residual: {mb1_residual.shape} \
            #                  mb1_extra_args: {mb1_extra_args.keys()}")
            # mb1: attn_1
            mb1_hidden_states, mb1_residual, mb1_extra_args = self.forward_layer(
                l1,
                mb1_positions,
                mb1_hidden_states,
                fwd_batch1,
                mb1_residual,
                MicroBatchOverlapStep.DECODE_ATTN_1_STEP,
                mb1_extra_args,
            )
            # if get_attention_tp_rank()==0:
            #     #logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 DECODE_ATTN_1_STEP,{get_gpu_memory_info(0)}")
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 DECODE_ATTN_1_STEP,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)
            # if True or get_attention_tp_rank() in [0]:
            #     #logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb1 DECODE_ATTN_1_STEP: \
            #                  mb0_hidden_states.shape: {mb0_hidden_states.shape}, \
            #                  mb0_residual: {mb0_residual.shape} \
            #                  mb0_extra_args: {mb0_extra_args.keys()} \
            #                  mb1_hidden_states.shape: {mb1_hidden_states.shape}, \
            #                  mb1_residual: {mb1_residual.shape} \
            #                  mb1_extra_args: {mb1_extra_args.keys()}")
            # mb0: wait_combine
            mb0_hidden_states, mb0_residual, mb0_extra_args = self.forward_layer(
                l0,
                mb0_positions,
                mb0_hidden_states,
                fwd_batch0,
                mb0_residual,
                MicroBatchOverlapStep.WAIT_COMBINE_LL_STEP,
                mb0_extra_args,
            )
            # if get_attention_tp_rank()==0:
            #     #logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 WAIT_COMBINE_LL_STEP,{get_gpu_memory_info(0)}")
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 WAIT_COMBINE_LL_STEP,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)
            # if True or get_attention_tp_rank() in [0]:
            #     #logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb0 WAIT_COMBINE_LL_STEP: \
            #                  mb0_hidden_states.shape: {mb0_hidden_states.shape}, \
            #                  mb0_residual: {mb0_residual.shape} \
            #                  mb0_extra_args: {mb0_extra_args.keys()} \
            #                  mb1_hidden_states.shape: {mb1_hidden_states.shape}, \
            #                  mb1_residual: {mb1_residual.shape} \
            #                  mb1_extra_args: {mb1_extra_args.keys()}")

            # mb0: add shared_expert_output
            if (
                l0 < len(self.layers)
                and MicroBatchOverlapExtraArgs.EXTRA_ARGS_LAST_LAYER_SHARED_EXPERT_OUTPUT_KEY
                in mb0_extra_args
            ):
                if mb0_extra_args[
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_LAST_LAYER_SHARED_EXPERT_OUTPUT_KEY
                ]:
                    mb0_hidden_states = (
                        mb0_hidden_states
                        + mb0_extra_args[
                            MicroBatchOverlapExtraArgs.EXTRA_ARGS_LAST_LAYER_SHARED_EXPERT_OUTPUT_KEY
                        ]
                    )
                del mb0_extra_args[
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_LAST_LAYER_SHARED_EXPERT_OUTPUT_KEY
                ]
            # if True or get_attention_tp_rank() in [0]:
            #     #logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb0 add shared_expert_output: \
            #                  mb0_hidden_states.shape: {mb0_hidden_states.shape}, \
            #                  mb0_residual: {mb0_residual.shape} \
            #                  mb0_extra_args: {mb0_extra_args.keys()} \
            #                  mb1_hidden_states.shape: {mb1_hidden_states.shape}, \
            #                  mb1_residual: {mb1_residual.shape} \
            #                  mb1_extra_args: {mb1_extra_args.keys()}")
            l0 += 1
            # if get_attention_tp_rank()==0:
            #     #logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 add shared_expert_output,{get_gpu_memory_info(0)}")
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 add shared_expert_output,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)

            # mb1: launch_dispatch
            mb1_hidden_states, _, mb1_extra_args = self.forward_layer(
                l1,
                mb1_positions,
                mb1_hidden_states,
                fwd_batch1,
                mb1_residual,
                MicroBatchOverlapStep.LAUNCH_DISPATCH_LL_STEP,
                mb1_extra_args,
            )
            # if get_attention_tp_rank()==0:
            #     #logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 LAUNCH_DISPATCH_LL_STEP,{get_gpu_memory_info(0)}")
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 LAUNCH_DISPATCH_LL_STEP,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)
            # if True or get_attention_tp_rank() in [0]:
            #     #logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb1 LAUNCH_DISPATCH_LL_STEP: \
            #                  mb0_hidden_states.shape: {mb0_hidden_states.shape}, \
            #                  mb0_residual: {mb0_residual.shape} \
            #                  mb0_extra_args: {mb0_extra_args.keys()} \
            #                  mb1_hidden_states.shape: {mb1_hidden_states.shape}, \
            #                  mb1_residual: {mb1_residual.shape} \
            #                  mb1_extra_args: {mb1_extra_args.keys()}")
            # mb1: shared_expert
            mb1_before_dispatch_hidden_states = mb1_extra_args.get(
                MicroBatchOverlapExtraArgs.EXTRA_ARGS_BEFORE_DISPATCH_HIDDEN_STATES_KEY,
                None,
            )
            _, mb1_residual, mb1_extra_args = self.forward_layer(
                l1,
                mb1_positions,
                mb1_before_dispatch_hidden_states,
                fwd_batch1,
                mb1_residual,
                MicroBatchOverlapStep.SHARED_EXPERTS,
                mb1_extra_args,
            )
            # if get_attention_tp_rank()==0:
            #     #logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 SHARED_EXPERTS,{get_gpu_memory_info(0)}")
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 SHARED_EXPERTS,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)
            # if True or get_attention_tp_rank() in [0]:
            #     #logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb1 SHARED_EXPERTS: \
            #                  mb0_hidden_states.shape: {mb0_hidden_states.shape}, \
            #                  mb0_residual: {mb0_residual.shape} \
            #                  mb0_extra_args: {mb0_extra_args.keys()} \
            #                  mb1_hidden_states.shape: {mb1_hidden_states.shape}, \
            #                  mb1_residual: {mb1_residual.shape} \
            #                  mb1_extra_args: {mb1_extra_args.keys()}")

            # mb0: attn_0
            mb0_hidden_states, mb0_residual, mb0_extra_args = self.forward_layer(
                l0,
                mb0_positions,
                mb0_hidden_states,
                fwd_batch0,
                mb0_residual,
                MicroBatchOverlapStep.DECODE_ATTN_0_STEP,
                mb0_extra_args,
            )
            # if get_attention_tp_rank()==0:
            #     #logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 DECODE_ATTN_0_STEP,{get_gpu_memory_info(0)}")
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 DECODE_ATTN_0_STEP,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)
            # if True or get_attention_tp_rank() in [0]:
            #     #logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb0 DECODE_ATTN_0_STEP: \
            #                  mb0_hidden_states.shape: {mb0_hidden_states.shape}, \
            #                  mb0_residual: {mb0_residual.shape} \
            #                  mb0_extra_args: {mb0_extra_args.keys()} \
            #                  mb1_hidden_states.shape: {mb1_hidden_states.shape}, \
            #                  mb1_residual: {mb1_residual.shape} \
            #                  mb1_extra_args: {mb1_extra_args.keys()}")
            # mb1: wait_launch
            mb1_hidden_states, mb1_residual, mb1_extra_args = self.forward_layer(
                l1,
                mb1_positions,
                mb1_hidden_states,
                fwd_batch1,
                mb1_residual,
                MicroBatchOverlapStep.WAIT_DISPATCH_LL_STEP,
                mb1_extra_args,
            )
            # if get_attention_tp_rank()==0:
            #     #logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 WAIT_DISPATCH_LL_STEP,{get_gpu_memory_info(0)}")
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 WAIT_DISPATCH_LL_STEP,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)
            # if True or get_attention_tp_rank() in [0]:
            #     #logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb1 WAIT_DISPATCH_LL_STEP: \
            #                  mb0_hidden_states.shape: {mb0_hidden_states.shape}, \
            #                  mb0_residual: {mb0_residual.shape} \
            #                  mb0_extra_args: {mb0_extra_args.keys()} \
            #                  mb1_hidden_states.shape: {mb1_hidden_states[0].shape if len(mb1_hidden_states) > 1 else mb1_hidden_states.shape}, \
            #                  mb1_residual: {mb1_residual.shape} \
            #                  mb1_extra_args: {mb1_extra_args.keys()}")
            # mb1: mlp
            mb1_hidden_states, mb1_residual, mb1_extra_args = self.forward_layer(
                l1,
                mb1_positions,
                mb1_hidden_states,
                fwd_batch1,
                mb1_residual,
                MicroBatchOverlapStep.MLP,
                mb1_extra_args,
            )
            # if get_attention_tp_rank()==0:
            #     #logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 MLP,{get_gpu_memory_info(0)}")
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 MLP,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)
            # if True or get_attention_tp_rank() in [0]:
            #     #logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb1 MLP: \
            #                  mb0_hidden_states.shape: {mb0_hidden_states.shape}, \
            #                  mb0_residual: {mb0_residual.shape} \
            #                  mb0_extra_args: {mb0_extra_args.keys()} \
            #                  mb1_hidden_states.shape: {mb1_hidden_states.shape}, \
            #                  mb1_residual: {mb1_residual.shape} \
            #                  mb1_extra_args: {mb1_extra_args.keys()}")
            # mb1: launch combine
            mb1_hidden_states, _, mb1_extra_args = self.forward_layer(
                l1,
                mb1_positions,
                mb1_hidden_states,
                fwd_batch1,
                mb1_residual,
                MicroBatchOverlapStep.LAUNCH_COMBINE_LL_STEP,
                mb1_extra_args,
            )
            # note: mb0_hidden_states does not need because new mb1_hidden_states will be set after WAIT_COMBINE_LL_STEP
            # del mb1_hidden_states
            # mb1_hidden_states=None
            # if get_attention_tp_rank()==0:
            #     #logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 LAUNCH_COMBINE_LL_STEP,{get_gpu_memory_info(0)}")
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 LAUNCH_COMBINE_LL_STEP,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)
            # if True or get_attention_tp_rank() in [0]:
            #     #logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb1 LAUNCH_COMBINE_LL_STEP: \
            #                  mb0_hidden_states.shape: {mb0_hidden_states.shape}, \
            #                  mb0_residual: {mb0_residual.shape} \
            #                  mb0_extra_args: {mb0_extra_args.keys()} \
            #                  mb1_hidden_states.shape: {mb1_hidden_states.shape}, \
            #                  mb1_residual: {mb1_residual.shape} \
            #                  mb1_extra_args: {mb1_extra_args.keys()}")
            # mb0: attn_1
            mb0_hidden_states, mb0_residual, mb0_extra_args = self.forward_layer(
                l0,
                mb0_positions,
                mb0_hidden_states,
                fwd_batch0,
                mb0_residual,
                MicroBatchOverlapStep.DECODE_ATTN_1_STEP,
                mb0_extra_args,
            )
            # if get_attention_tp_rank()==0:
            #     #logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 DECODE_ATTN_1_STEP,{get_gpu_memory_info(0)}")
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb0 DECODE_ATTN_1_STEP,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)
            # if True or get_attention_tp_rank() in [0]:
            #     #logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb0 DECODE_ATTN_1_STEP: \
            #                  mb0_hidden_states.shape: {mb0_hidden_states.shape}, \
            #                  mb0_residual: {mb0_residual.shape} \
            #                  mb0_extra_args: {mb0_extra_args.keys()} \
            #                  mb1_hidden_states.shape: {mb1_hidden_states.shape}, \
            #                  mb1_residual: {mb1_residual.shape} \
            #                  mb1_extra_args: {mb1_extra_args.keys()}")
            # mb1: wait_combine
            mb1_hidden_states, mb1_residual, mb1_extra_args = self.forward_layer(
                l1,
                mb1_positions,
                mb1_hidden_states,
                fwd_batch1,
                mb1_residual,
                MicroBatchOverlapStep.WAIT_COMBINE_LL_STEP,
                mb1_extra_args,
            )
            # if get_attention_tp_rank()==0:
            #     #logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 WAIT_COMBINE_LL_STEP,{get_gpu_memory_info(0)}")
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 WAIT_COMBINE_LL_STEP,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)
            # if True or get_attention_tp_rank() in [0]:
            #     #logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb1 WAIT_COMBINE_LL_STEP: \
            #                  mb0_hidden_states.shape: {mb0_hidden_states.shape}, \
            #                  mb0_residual: {mb0_residual.shape} \
            #                  mb0_extra_args: {mb0_extra_args.keys()} \
            #                  mb1_hidden_states.shape: {mb1_hidden_states.shape}, \
            #                  mb1_residual: {mb1_residual.shape} \
            #                  mb1_extra_args: {mb1_extra_args.keys()}")
            # mb1: add shared_expert_output
            if (
                l1 < len(self.layers)
                and MicroBatchOverlapExtraArgs.EXTRA_ARGS_LAST_LAYER_SHARED_EXPERT_OUTPUT_KEY
                in mb1_extra_args
            ):
                if mb1_extra_args[
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_LAST_LAYER_SHARED_EXPERT_OUTPUT_KEY
                ]:
                    mb1_hidden_states = (
                        mb1_hidden_states
                        + mb1_extra_args[
                            MicroBatchOverlapExtraArgs.EXTRA_ARGS_LAST_LAYER_SHARED_EXPERT_OUTPUT_KEY
                        ]
                    )
                del mb1_extra_args[
                    MicroBatchOverlapExtraArgs.EXTRA_ARGS_LAST_LAYER_SHARED_EXPERT_OUTPUT_KEY
                ]
            # if True or get_attention_tp_rank() in [0]:
            #     #logger.debug(f"layer[{i}], l0[{l0}], l1[{l1}], after mb1 add shared_expert_output: \
            #                  mb0_hidden_states.shape: {mb0_hidden_states.shape}, \
            #                  mb0_residual: {mb0_residual.shape} \
            #                  mb0_extra_args: {mb0_extra_args.keys()} \
            #                  mb1_hidden_states.shape: {mb1_hidden_states.shape}, \
            #                  mb1_residual: {mb1_residual.shape} \
            #                  mb1_extra_args: {mb1_extra_args.keys()}")
            l1 += 1
            # if get_attention_tp_rank()==0:
            #     #logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 add shared_expert_output,{get_gpu_memory_info(0)}")
            #     logger.debug(f"layer[{i}],l0[{l0}],l1[{l1}],after mb1 add shared_expert_output,{get_gpu_memory_info_diff_str(0,prev_cached_mem,prev_allocated_mem,prev_free_mem)}")
            #     prev_cached_mem,prev_allocated_mem,prev_free_mem=get_gpu_memory_info_not_str(0)

        attn_tp_size = get_attention_tp_size()
        if attn_tp_size != 1:
            mb0_hidden_states += mb0_residual
            # residual_after = None # note: never set None to residual, this is different to official
            mb0_hidden_states, local_mb0_hidden_states = (
                fwd_batch0.gathered_buffer[: fwd_batch0.input_ids.shape[0]],
                mb0_hidden_states,
            )
            tp_all_gather(
                list(mb0_hidden_states.tensor_split(attn_tp_size)),
                local_mb0_hidden_states,
            )
            mb1_hidden_states += mb1_residual
            # residual_after = None # note: never set None to residual, this is different to official
            mb1_hidden_states, local_mb1_hidden_states = (
                fwd_batch1.gathered_buffer[: fwd_batch1.input_ids.shape[0]],
                mb1_hidden_states,
            )
            tp_all_gather(
                list(mb1_hidden_states.tensor_split(attn_tp_size)),
                local_mb1_hidden_states,
            )

        hidden_states = torch.cat([mb0_hidden_states, mb1_hidden_states], dim=0)
        # note: keep same with python/sglang/srt/models/deepseek_v2.py:1247(official v0.4.5 implementation)
        # residual = torch.cat([mb0_residual, mb1_residual], dim=0)
        residual = None
        # logger.debug(
        # f'~~~~~~~~~~ decode finished,{hidden_states.shape=}, hidden_states: {hidden_states}')

        return hidden_states, residual

    def forward_layer(
        self,
        layer_id: int,  # start from 0
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        step: Optional[MicroBatchOverlapStep] = None,
        extra_args: Optional[Dict[str, Any]] = {},
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        if layer_id >= len(self.layers) or layer_id < self.config.first_k_dense_replace:
            return hidden_states, residual, extra_args
        return self.layers[layer_id](
            positions, hidden_states, forward_batch, residual, step, extra_args
        )


class DeepseekV2ForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.n_share_experts_fusion = global_server_args_dict["n_share_experts_fusion"]
        # Only Deepseek V3/R1 can use shared experts fusion optimization now.
        if (
            global_server_args_dict.get("disable_shared_experts_fusion", False)
            or self.config.architectures[0] != "DeepseekV3ForCausalLM"
            or self.config.n_routed_experts != 256
            or self.config.routed_scaling_factor != 2.5
        ):
            self.n_share_experts_fusion = None
            global_server_args_dict["n_share_experts_fusion"] = None
            logger.info(
                "Only Deepseek V3/R1 can use shared experts fusion optimization. Shared experts fusion optimization is disabled."
            )
        elif self.n_share_experts_fusion is None:
            global_server_args_dict["n_share_experts_fusion"] = self.tp_size
            self.n_share_experts_fusion = self.tp_size
            logger.info(
                f"Shared experts fusion optimization is default enabled in DeepSeek V3/R1, and n_share_experts_fusion is set to {self.tp_size}. You can tune it by setting --n_share_experts_fusion or disable it by setting --disable_shared_experts_fusion."
            )

        self.model = DeepseekV2Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)
        self.dp_size = get_attention_dp_size()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:

        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        logits_output = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
        # logger.debug(
        # f'~~~~~~~~~~ after self.model, hidden_states.shape: {hidden_states.shape}, batch_size: {forward_batch.batch_size}')
        # logger.debug(
        # f'~~~~~~~~~~ after self.model, batch_size: {forward_batch.batch_size}, logits_output: {logits_output}')
        return logits_output

    def post_load_weights(self):

        # Perform post-processing after loading weights

        if not global_server_args_dict["disable_mla"]:
            for layer_id in range(self.config.num_hidden_layers):
                self_attn = self.model.layers[layer_id].self_attn
                if hasattr(self_attn.kv_b_proj, "qweight"):
                    # AWQ compatible
                    if _is_cuda:
                        w = awq_dequantize(
                            self_attn.kv_b_proj.qweight,
                            self_attn.kv_b_proj.scales,
                            self_attn.kv_b_proj.qzeros,
                        ).T
                    else:
                        w = awq_dequantize(
                            self_attn.kv_b_proj.qweight,
                            self_attn.kv_b_proj.scales,
                            self_attn.kv_b_proj.qzeros,
                            0,
                            0,
                            0,
                        ).T
                else:
                    w = self_attn.kv_b_proj.weight
                # NOTE(HandH1998): Since `bmm_fp8` only supports per-tensor scale, we have to requantize `self_attn.kv_b_proj`.
                # This may affect the accuracy of fp8 model.
                if w.dtype in (
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fnuz,
                ):
                    if hasattr(self.quant_config, "weight_block_size"):
                        weight_block_size = self.quant_config.weight_block_size
                        if weight_block_size is not None:
                            assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                            if _is_hip:
                                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                                    weight=w,
                                    weight_scale=self_attn.kv_b_proj.weight_scale_inv,
                                    input_scale=None,
                                )
                            else:
                                weight = w
                                weight_scale = self_attn.kv_b_proj.weight_scale_inv

                            w, scale = block_quant_to_tensor_quant(
                                weight, weight_scale, weight_block_size
                            )
                            self_attn.w_scale = scale
                    else:
                        weight = w
                        weight_scale = self_attn.kv_b_proj.weight_scale
                        w, scale = channel_quant_to_tensor_quant(weight, weight_scale)
                        self_attn.w_scale = scale

                if w.dtype == torch.int8:
                    if hasattr(self.quant_config, "weight_block_size"):
                        # block-wise int8 need it
                        weight_block_size = self.quant_config.weight_block_size
                        if weight_block_size is not None:
                            assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                            weight = w
                            weight_scale = self_attn.kv_b_proj.weight_scale_inv
                            w = int8_block_dequant(
                                weight, weight_scale, weight_block_size
                            ).to(torch.bfloat16)
                    else:
                        # channel-wise int8 need it
                        w = w.to(torch.bfloat16) * self_attn.kv_b_proj.weight_scale.to(
                            torch.bfloat16
                        )
                w_kc, w_vc = w.unflatten(
                    0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
                ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
                self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
                if (
                    hasattr(self_attn.kv_b_proj, "weight_scale")
                    and self_attn.w_scale is None
                ):
                    self_attn.w_scale = self_attn.kv_b_proj.weight_scale
                    if _is_hip:
                        self_attn.w_scale *= 2.0

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        if self.n_share_experts_fusion is not None and self.n_share_experts_fusion > 0:
            weights_list = list(weights)
            weights_dict = dict(weights_list)
            if self.quant_config.get_name() == "w8a8_int8":
                suffix_list = [
                    "down_proj.weight",
                    "down_proj.weight_scale",
                    "gate_proj.weight",
                    "gate_proj.weight_scale",
                    "up_proj.weight",
                    "up_proj.weight_scale",
                ]
            else:
                suffix_list = [
                    "down_proj.weight",
                    "down_proj.weight_scale_inv",
                    "gate_proj.weight",
                    "gate_proj.weight_scale_inv",
                    "up_proj.weight",
                    "up_proj.weight_scale_inv",
                ]
            names_to_remove = []
            for moe_layer in tqdm(
                range(
                    self.config.first_k_dense_replace,
                    self.config.num_hidden_layers,
                    self.config.moe_layer_freq,
                ),
                desc=f"Cloning {self.n_share_experts_fusion} "
                     "replicas of the shared expert into MoE",
            ):
                for num_repeat in range(self.n_share_experts_fusion):
                    for suffix in suffix_list:
                        shared_expert_weight_name = (
                            f"model.layers.{moe_layer}.mlp.shared_experts.{suffix}"
                        )
                        weights_list.append(
                            (
                                f"model.layers.{moe_layer}."
                                f"mlp.experts."
                                f"{self.config.n_routed_experts + num_repeat}"
                                f".{suffix}",
                                weights_dict[shared_expert_weight_name].clone(),
                            )
                        )
                        names_to_remove += [shared_expert_weight_name]
            weights = [w for w in weights_list if w[0] not in names_to_remove]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        MoEImpl = (
            DeepEPMoE
            if global_server_args_dict["enable_deepep_moe"]
            else (EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE)
        )
        expert_params_mapping = MoEImpl.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
                        + (
                            self.n_share_experts_fusion
                            if self.n_share_experts_fusion is not None
                            else 0
                        ),
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # TODO(HandH1998): Modify it when nextn is supported.
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                if num_nextn_layers > 0 and name.startswith("model.layers"):
                    name_list = name.split(".")
                    if (
                        len(name_list) >= 3
                        and int(name_list[2]) >= self.config.num_hidden_layers
                    ):
                        continue
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
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
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

        self.post_load_weights()

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def token_balanced_batch_split(fwd_batch: Optional[ForwardBatch]):
    if fwd_batch.forward_mode.is_extend():
        return split_extend_batch(fwd_batch)
    elif fwd_batch.forward_mode.is_decode():
        return split_decode_batch(fwd_batch)
    else:
        assert False, f"{fwd_batch.forward_mode} is not supported to split"


def split_extend_batch(fwd_batch: Optional[ForwardBatch]):
    sub_fwd_batch0 = copy(fwd_batch)
    sub_fwd_batch1 = copy(fwd_batch)
    bs_joint_batch_boundary = 0
    batch_boundary = 0

    all_tokens = sum(fwd_batch.extend_seq_lens)
    batch_boundary = 0
    for batch_tokens in fwd_batch.extend_seq_lens[:-1]:
        bs_joint_batch_boundary += batch_tokens
        batch_boundary += 1
        if bs_joint_batch_boundary >= all_tokens // 2:
            break

    for key in [
        "req_pool_indices",
        "seq_lens",
        "seq_lens_cpu",  # Optional[torch.Tensor]
        "extend_seq_lens",
        "extend_prefix_lens",
        "extend_start_loc",
        "extend_prefix_lens_cpu",
        "extend_seq_lens_cpu",  # Optional[List[int]]
        "extend_logprob_start_lens_cpu",
    ]:
        if getattr(fwd_batch, key) is not None:
            setattr(sub_fwd_batch0, key, getattr(fwd_batch, key)[:batch_boundary])
            setattr(sub_fwd_batch1, key, getattr(fwd_batch, key)[batch_boundary:])

    if getattr(fwd_batch, "extend_num_tokens") is not None:
        sub_fwd_batch0.extend_num_tokens = bs_joint_batch_boundary.item()
        sub_fwd_batch1.extend_num_tokens = (
            fwd_batch.extend_num_tokens - sub_fwd_batch0.extend_num_tokens
        )

    for key in [
        "input_ids",
        "positions",
        "out_cache_loc",
    ]:
        setattr(sub_fwd_batch0, key, getattr(fwd_batch, key)[:bs_joint_batch_boundary])
        setattr(sub_fwd_batch1, key, getattr(fwd_batch, key)[bs_joint_batch_boundary:])

    if hasattr(fwd_batch, "extend_seq_lens_cpu") and hasattr(
        fwd_batch, "global_num_tokens_cpu"
    ):
        sub_fwd_batch0.global_num_tokens_cpu = [sum(sub_fwd_batch0.extend_seq_lens_cpu)]
        sub_fwd_batch1.global_num_tokens_cpu = [sum(sub_fwd_batch1.extend_seq_lens_cpu)]

    if hasattr(fwd_batch, "extend_seq_lens") and hasattr(
        fwd_batch, "global_num_tokens_gpu"
    ):
        sub_fwd_batch0.global_num_tokens_gpu = sub_fwd_batch0.extend_seq_lens.sum(
            0, keepdim=True
        )
        sub_fwd_batch1.global_num_tokens_gpu = sub_fwd_batch1.extend_seq_lens.sum(
            0, keepdim=True
        )

    sub_fwd_batch0.seq_lens_sum = int(sub_fwd_batch0.seq_lens_cpu.sum().item())
    sub_fwd_batch1.seq_lens_sum = int(sub_fwd_batch1.seq_lens_cpu.sum().item())

    sub_fwd_batch0.batch_size = batch_boundary
    sub_fwd_batch1.batch_size = fwd_batch.batch_size - sub_fwd_batch0.batch_size

    return bs_joint_batch_boundary, sub_fwd_batch0, sub_fwd_batch1


def split_decode_batch(fwd_batch: Optional[ForwardBatch]):
    ## if get_attention_tp_rank()==0:
    # logger.debug(f"[token_balanced_batch_split] original forward batch: {fwd_batch}")
    sub_fwd_batch0 = copy(fwd_batch)
    sub_fwd_batch1 = copy(fwd_batch)
    batch_boundary = fwd_batch.batch_size // 2

    # set attributes
    for key in [
        "input_ids",
        "req_pool_indices",
        "seq_lens",
        "out_cache_loc",
        "seq_lens_cpu",
        "positions",
    ]:
        setattr(sub_fwd_batch0, key, getattr(fwd_batch, key)[:batch_boundary])
        setattr(sub_fwd_batch1, key, getattr(fwd_batch, key)[batch_boundary:])

    # batch_size, global_num_tokens_cpu, global_num_tokens_gpu, global_num_tokens_for_logprob_cpu, global_num_tokens_for_logprob_gpu
    sub_fwd_batch0.batch_size = batch_boundary
    sub_fwd_batch1.batch_size = fwd_batch.batch_size - batch_boundary
    sub_fwd_batch0.global_num_tokens_cpu = [batch_boundary]
    sub_fwd_batch1.global_num_tokens_cpu = [fwd_batch.batch_size - batch_boundary]
    sub_fwd_batch0.global_num_tokens_gpu = torch.tensor(
        [batch_boundary],
        device=f"cuda:{str(sub_fwd_batch0.global_num_tokens_gpu.get_device())}",
    )
    sub_fwd_batch1.global_num_tokens_gpu = torch.tensor(
        [fwd_batch.batch_size - batch_boundary],
        device=f"cuda:{str(sub_fwd_batch1.global_num_tokens_gpu.get_device())}",
    )
    sub_fwd_batch0.global_num_tokens_for_logprob_cpu = [batch_boundary]
    sub_fwd_batch1.global_num_tokens_for_logprob_cpu = [
        fwd_batch.batch_size - batch_boundary
    ]
    sub_fwd_batch0.global_num_tokens_for_logprob_gpu = torch.tensor(
        [batch_boundary],
        device=f"cuda:{str(sub_fwd_batch0.global_num_tokens_gpu.get_device())}",
    )
    sub_fwd_batch1.global_num_tokens_for_logprob_gpu = torch.tensor(
        [fwd_batch.batch_size - batch_boundary],
        device=f"cuda:{str(sub_fwd_batch1.global_num_tokens_gpu.get_device())}",
    )

    # seq_lens_sum
    sub_fwd_batch0.seq_lens_sum = int(sub_fwd_batch0.seq_lens_cpu.sum().item())
    sub_fwd_batch1.seq_lens_sum = int(sub_fwd_batch1.seq_lens_cpu.sum().item())

    # sampling_info
    # for key in [
    #     "temperatures",
    #     "top_ps",
    #     "top_ks",
    #     "min_ps"
    # ]:
    #     setattr(sub_fwd_batch0.sampling_info, key, getattr(fwd_batch.sampling_info, key)[:batch_boundary])
    #     setattr(sub_fwd_batch1.sampling_info, key, getattr(fwd_batch.sampling_info, key)[batch_boundary:])

    # dp_local_start_pos, dp_local_num_tokens, gathered_buffer, to support when enable-dp-size

    return batch_boundary, sub_fwd_batch0, sub_fwd_batch1


def get_role(forward_batch: ForwardBatch) -> str:
    role = None
    if forward_batch.forward_mode.is_extend():
        role = "prefill"
    elif forward_batch.forward_mode.is_decode():
        role = "decode"
    assert role is not None, f"{forward_batch.forward_mode} is not supported for role"
    return role


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM]


def get_gpu_memory_info_not_str(device) -> tuple:
    return torch.cuda.memory_reserved(device=device), torch.cuda.memory_allocated(
        device=device), torch.cuda.memory_reserved(device=device) - torch.cuda.memory_allocated(device=device)


def get_gpu_memory_info_diff_str(device, prev_cached, prev_allocated, prev_free) -> str:
    return f"device[{device}],diff,cached:{torch.cuda.memory_reserved(device=device) - prev_cached},[[[allocated:{torch.cuda.memory_allocated(device=device) - prev_allocated},]]]free:{torch.cuda.memory_reserved(device=device) - torch.cuda.memory_allocated(device=device) - prev_free}"


def get_gpu_memory_info(device) -> str:
    return f"device[{device}],cached:{torch.cuda.memory_reserved(device=device)},[[[allocated:{torch.cuda.memory_allocated(device=device)},]]]free:{torch.cuda.memory_reserved(device=device) - torch.cuda.memory_allocated(device=device)}"
