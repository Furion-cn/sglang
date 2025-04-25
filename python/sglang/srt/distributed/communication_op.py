# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py

from typing import Any, Dict, Optional, Union

import torch
import nvtx
import torch.distributed

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    with nvtx.annotate(message="tensor_model_parallel_all_reduce", color="seagreen", category="communication"):
        """All-reduce the input tensor across model parallel group."""
        return get_tp_group().all_reduce(input_)

def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    with nvtx.annotate(message="tensor_model_parallel_all_gather", color="seagreen", category="communication"):
        """All-gather the input tensor across model parallel group."""
        return get_tp_group().all_gather(input_, dim)

def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    with nvtx.annotate(message="tensor_model_parallel_gather", color="seagreen", category="communication"):
        """Gather the input tensor across model parallel group."""
        return get_tp_group().gather(input_, dst, dim)

def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    with nvtx.annotate(message="broadcast_tensor_dicta", color="seagreen", category="communication"):
        if not torch.distributed.is_initialized():
            return tensor_dict
        return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
