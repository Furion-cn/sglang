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
"""Fused operators for normalization layers."""

import logging
from typing import Optional, Tuple, Union

import nvtx
import torch
import torch.nn as nn

from sglang.srt.custom_op import CustomOp
from sglang.srt.utils import is_cuda_available

_is_cuda = is_cuda_available()

if _is_cuda:
    from sgl_kernel import (
        fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )


logger = logging.getLogger(__name__)


class RMSNorm(CustomOp):

    @nvtx.annotate(color="slateblue", category="rms_norm")
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, *args, **kwargs):
        if torch.compiler.is_compiling():
            return self.forward_native(*args, **kwargs)
        if _is_cuda:
            return self.forward_cuda(*args, **kwargs)
        elif _is_hip:
            return self.forward_hip(*args, **kwargs)
        else:
            return self.forward_native(*args, **kwargs)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        with nvtx.annotate(message="forward_cuda", color="slateblue", category="rms_norm"):
            if x.shape[0] == 0:
                if residual is not None:
                    return x, residual
                return x
            if not x.is_contiguous():
                x = x.contiguous()
            # logger.info(f"x contiguous {x.is_contiguous()}")
            if residual is not None:
                if not residual.is_contiguous():
                    residual = residual.contiguous()
                fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
                return x, residual
            out = rmsnorm(x, self.weight.data, self.variance_epsilon)
            return out

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            # NOTE: Romove this if aiter kernel supports discontinuous input
            x = x.contiguous()
        if residual is not None:
            fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = torch.empty_like(x)
        rms_norm(out, x, self.weight.data, self.variance_epsilon)
        return out

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        with nvtx.annotate(message="forward_native", color="slateblue", category="rms_norm"):
            if not x.is_contiguous():
                x = x.contiguous()
            orig_dtype = x.dtype
            x = x.to(torch.float32)
            if residual is not None:
                x = x + residual.to(torch.float32)
                residual = x.to(orig_dtype)

            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
            x = (x * self.weight).to(orig_dtype)
            if residual is None:
                return x
            else:
                return x, residual


class GemmaRMSNorm(CustomOp):

    @nvtx.annotate(color="darkslateblue", category="gemma_rms_norm")
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        with nvtx.annotate(message="forward_native", color="darkslateblue", category="gemma_rms_norm"):
            orig_dtype = x.dtype
            if residual is not None:
                x = x + residual
                residual = x

            x = x.float()
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
            x = x * (1.0 + self.weight.float())
            x = x.to(orig_dtype)
            return x if residual is None else (x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        with nvtx.annotate(message="forward_cuda", color="darkslateblue", category="gemma_rms_norm"):
            if residual is not None:
                gemma_fused_add_rmsnorm(
                    x, residual, self.weight.data, self.variance_epsilon
                )
                return x, residual
            out = gemma_rmsnorm(x, self.weight.data, self.variance_epsilon)
            return out


class Gemma3RMSNorm(nn.Module):

    @nvtx.annotate(color="darkslateblue", category="gemma3_rms_norm")
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        with nvtx.annotate(message="_norm", color="darkslateblue", category="gemma3_rms_norm"):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        with nvtx.annotate(message="forward", color="darkslateblue", category="gemma3_rms_norm"):
            output = self._norm(x.float())
            # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
            # See https://github.com/huggingface/transformers/pull/29402
            output = output * (1.0 + self.weight.float())
            return output.type_as(x)

    def extra_repr(self):
        with nvtx.annotate(message="extra_repr", color="darkslateblue", category="gemma3_rms_norm"):
            return f"{tuple(self.weight.shape)}, eps={self.eps}"


if not _is_cuda:
    logger.info(
        "sgl-kernel is not available on Non-NV platforms. Fallback to other kernel libraries."
    )
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
