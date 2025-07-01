"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter

"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""

import abc
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import debug_timing, is_cuda, next_power_of_2

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()

# TPU/JAX相关导入
try:
    import jax
    import jax.numpy as jnp
    from jax import device_get, device_put

    _has_jax = True
except ImportError:
    _has_jax = False


def is_tpu_device(device: str) -> bool:
    """检测是否为TPU设备"""
    return device.startswith("tpu") or device.startswith("xla")


def get_jax_device(device: str):
    """获取对应的JAX设备"""
    if not _has_jax:
        return None

    if device.startswith("tpu"):
        # 提取TPU设备ID
        try:
            device_id = int(device.split(":")[-1]) if ":" in device else 0
            return jax.devices("tpu")[device_id]
        except (IndexError, ValueError):
            return jax.devices("tpu")[0] if jax.devices("tpu") else None
    elif device.startswith("xla"):
        # XLA设备
        return jax.devices("xla")[0] if jax.devices("xla") else None
    else:
        return None


def torch_to_jax_dtype(torch_dtype):
    """将PyTorch数据类型转换为JAX数据类型"""
    dtype_mapping = {
        torch.float32: jnp.float32,
        torch.float16: jnp.float16,
        torch.bfloat16: jnp.bfloat16,
        torch.int32: jnp.int32,
        torch.int64: jnp.int64,
        torch.uint8: jnp.uint8,
        torch.bool: jnp.bool_,
    }
    return dtype_mapping.get(torch_dtype, jnp.float32)


def jax_to_torch_dtype(jax_dtype):
    """将JAX数据类型转换为PyTorch数据类型"""
    dtype_mapping = {
        jnp.float32: torch.float32,
        jnp.float16: torch.float16,
        jnp.bfloat16: torch.bfloat16,
        jnp.int32: torch.int32,
        jnp.int64: torch.int64,
        jnp.uint8: torch.uint8,
        jnp.bool_: torch.bool,
    }
    return dtype_mapping.get(jax_dtype, torch.float32)


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region():
            self.req_to_token = torch.zeros(
                (size, max_context_len), dtype=torch.int32, device=device
            )
        self.free_slots = list(range(size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        self.free_slots = list(range(self.size))


class KVCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        backend: str = "torch",
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device

        self.backend = backend
        self.is_jax_backend = backend == "jax"
        self.jax_device = get_jax_device(device) if self.is_jax_backend else None

        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype

        if self.is_jax_backend and _has_jax:
            self.jax_dtype = torch_to_jax_dtype(self.store_dtype)
        else:
            self.jax_dtype = None

        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

    def _tensor_to_jax(self, tensor: torch.Tensor):
        """将PyTorch tensor转换为JAX array"""
        if not self.is_jax_backend or not _has_jax:
            return tensor

        # 转换为numpy，然后转换为JAX array
        numpy_array = tensor.detach().cpu().numpy()
        jax_array = jnp.array(numpy_array, dtype=self.jax_dtype)
        if self.jax_device:
            jax_array = device_put(jax_array, self.jax_device)
        return jax_array

    def _jax_to_tensor(self, jax_array, original_tensor: torch.Tensor = None):
        """将JAX array转换为PyTorch tensor"""
        if not self.is_jax_backend or not _has_jax:
            return jax_array

        # 从JAX设备获取数据
        if self.jax_device:
            jax_array = device_get(jax_array)

        # 转换为numpy，然后转换为tensor
        numpy_array = np.array(jax_array)
        tensor = torch.from_numpy(numpy_array)

        # 保持原始tensor的设备信息
        if original_tensor is not None:
            tensor = tensor.to(
                device=original_tensor.device, dtype=original_tensor.dtype
            )
        else:
            tensor = tensor.to(device=self.device, dtype=self.dtype)

        return tensor

    def _create_jax_buffer(self, shape, dtype=None):
        """创建JAX buffer"""
        if not self.is_jax_backend or not _has_jax:
            return None

        jax_dtype = dtype or self.jax_dtype
        jax_array = jnp.zeros(shape, dtype=jax_dtype)
        if self.jax_device:
            jax_array = device_put(jax_array, self.jax_device)
        return jax_array

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    def get_flat_data(self, indices):
        raise NotImplementedError()

    def transfer(self, indices, flat_data):
        raise NotImplementedError()

    def transfer_per_layer(self, indices, flat_data, layer_id):
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter):
        self.layer_transfer_counter = layer_transfer_counter


class TokenToKVPoolAllocator:
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.page_size = 1

        self.free_slots = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()

        self._kvcache = kvcache

    def available_size(self):
        return len(self.free_slots)

    def debug_print(self) -> str:
        return ""

    def get_kvcache(self):
        return self._kvcache

    def alloc(self, need_size: int):
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            self.free_slots = torch.cat((self.free_slots, free_index))
        else:
            self.free_group.append(free_index)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def backup_state(self):
        return self.free_slots

    def restore_state(self, free_slots):
        self.free_slots = free_slots

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)


class MHATokenToKVPool(KVCache):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        backend: str = "torch",
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            backend,
        )

        self.head_num = head_num
        self.head_dim = head_dim
        self._create_buffers()

        # used for chunked cpu-offloading
        self.chunk_size = 8192
        self.layer_transfer_counter = None
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream() if _is_cuda else None

        k_size, v_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB"
            f"{' (TPU/JAX backend)' if self.is_jax_backend else ''}"
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region():
            if self.is_jax_backend and _has_jax:
                # 使用JAX创建缓冲区
                self.k_buffer = [
                    self._create_jax_buffer(
                        (self.size + self.page_size, self.head_num, self.head_dim)
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    self._create_jax_buffer(
                        (self.size + self.page_size, self.head_num, self.head_dim)
                    )
                    for _ in range(self.layer_num)
                ]
            else:
                # 原始PyTorch实现
                self.k_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            if self.is_jax_backend and _has_jax:
                k_size_bytes += np.prod(k_cache.shape) * k_cache.dtype.itemsize
            else:
                k_size_bytes += np.prod(k_cache.shape) * k_cache.dtype.itemsize
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            if self.is_jax_backend and _has_jax:
                v_size_bytes += np.prod(v_cache.shape) * v_cache.dtype.itemsize
            else:
                v_size_bytes += np.prod(v_cache.shape) * v_cache.dtype.itemsize
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        if self.is_jax_backend and _has_jax:
            # TPU backend不支持直接获取指针信息
            logger.warning("get_contiguous_buf_infos not supported for TPU backend")
            return [], [], []

        kv_data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self.get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_cpu_copy(self, indices):
        if self.is_jax_backend and _has_jax:
            # TPU backend的CPU拷贝
            kv_cache_cpu = []
            for layer_id in range(self.layer_num):
                kv_cache_cpu.append([])
                for i in range(0, len(indices), self.chunk_size):
                    chunk_indices = indices[i : i + self.chunk_size]
                    # 从JAX获取数据并转换为CPU tensor
                    k_jax = self.k_buffer[layer_id][chunk_indices]
                    v_jax = self.v_buffer[layer_id][chunk_indices]
                    k_cpu = self._jax_to_tensor(k_jax).cpu()
                    v_cpu = self._jax_to_tensor(v_jax).cpu()
                    kv_cache_cpu[-1].append([k_cpu, v_cpu])
            return kv_cache_cpu
        else:
            # 原始CUDA实现
            torch.cuda.synchronize()
            kv_cache_cpu = []
            for layer_id in range(self.layer_num):
                kv_cache_cpu.append([])
                for i in range(0, len(indices), self.chunk_size):
                    chunk_indices = indices[i : i + self.chunk_size]
                    k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                        "cpu", non_blocking=True
                    )
                    v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                        "cpu", non_blocking=True
                    )
                    kv_cache_cpu[-1].append([k_cpu, v_cpu])
            torch.cuda.synchronize()
            return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        if self.is_jax_backend and _has_jax:
            # TPU backend的CPU到设备拷贝
            for layer_id in range(self.layer_num):
                for i in range(0, len(indices), self.chunk_size):
                    chunk_indices = indices[i : i + self.chunk_size]
                    k_cpu, v_cpu = (
                        kv_cache_cpu[layer_id][i // self.chunk_size][0],
                        kv_cache_cpu[layer_id][i // self.chunk_size][1],
                    )
                    assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                    # 转换为JAX array
                    k_jax = self._tensor_to_jax(k_cpu)
                    v_jax = self._tensor_to_jax(v_cpu)
                    # 使用JAX的索引更新
                    self.k_buffer[layer_id] = (
                        self.k_buffer[layer_id].at[chunk_indices].set(k_jax)
                    )
                    self.v_buffer[layer_id] = (
                        self.v_buffer[layer_id].at[chunk_indices].set(v_jax)
                    )
        else:
            # 原始CUDA实现
            torch.cuda.synchronize()
            for layer_id in range(self.layer_num):
                for i in range(0, len(indices), self.chunk_size):
                    chunk_indices = indices[i : i + self.chunk_size]
                    k_cpu, v_cpu = (
                        kv_cache_cpu[layer_id][i // self.chunk_size][0],
                        kv_cache_cpu[layer_id][i // self.chunk_size][1],
                    )
                    assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                    k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                    v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                    self.k_buffer[layer_id][chunk_indices] = k_chunk
                    self.v_buffer[layer_id][chunk_indices] = v_chunk
            torch.cuda.synchronize()

    # Todo: different memory layout
    def get_flat_data(self, indices):
        # prepare a large chunk of contiguous data for efficient transfer
        if self.is_jax_backend and _has_jax:
            # TPU backend的扁平化数据
            k_data = [self.k_buffer[i][indices] for i in range(self.layer_num)]
            v_data = [self.v_buffer[i][indices] for i in range(self.layer_num)]
            # 转换为tensor返回
            k_tensors = [self._jax_to_tensor(k) for k in k_data]
            v_tensors = [self._jax_to_tensor(v) for v in v_data]
            flatten = torch.stack(
                [
                    torch.stack(k_tensors),
                    torch.stack(v_tensors),
                ]
            )
            return flatten
        else:
            # 原始实现
            flatten = torch.stack(
                [
                    torch.stack(
                        [self.k_buffer[i][indices] for i in range(self.layer_num)]
                    ),
                    torch.stack(
                        [self.v_buffer[i][indices] for i in range(self.layer_num)]
                    ),
                ]
            )
            return flatten

    @debug_timing
    def transfer(self, indices, flat_data):
        # transfer prepared data from host to device
        if self.is_jax_backend and _has_jax:
            # TPU backend的数据传输
            flat_data = flat_data.to(device=self.device, non_blocking=False)
            k_data, v_data = flat_data[0], flat_data[1]
            for i in range(self.layer_num):
                k_jax = self._tensor_to_jax(k_data[i])
                v_jax = self._tensor_to_jax(v_data[i])
                self.k_buffer[i] = self.k_buffer[i].at[indices].set(k_jax)
                self.v_buffer[i] = self.v_buffer[i].at[indices].set(v_jax)
        else:
            # 原始实现
            flat_data = flat_data.to(device=self.device, non_blocking=False)
            k_data, v_data = flat_data[0], flat_data[1]
            for i in range(self.layer_num):
                self.k_buffer[i][indices] = k_data[i]
                self.v_buffer[i][indices] = v_data[i]

    def transfer_per_layer(self, indices, flat_data, layer_id):
        # transfer prepared data from host to device
        if self.is_jax_backend and _has_jax:
            # TPU backend的逐层传输
            flat_data = flat_data.to(device=self.device, non_blocking=False)
            k_data, v_data = flat_data[0], flat_data[1]
            k_jax = self._tensor_to_jax(k_data)
            v_jax = self._tensor_to_jax(v_data)
            self.k_buffer[layer_id - self.start_layer] = (
                self.k_buffer[layer_id - self.start_layer].at[indices].set(k_jax)
            )
            self.v_buffer[layer_id - self.start_layer] = (
                self.v_buffer[layer_id - self.start_layer].at[indices].set(v_jax)
            )
        else:
            # 原始实现
            flat_data = flat_data.to(device=self.device, non_blocking=False)
            k_data, v_data = flat_data[0], flat_data[1]
            self.k_buffer[layer_id - self.start_layer][indices] = k_data
            self.v_buffer[layer_id - self.start_layer][indices] = v_data

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.is_jax_backend and _has_jax:
            # TPU backend: 返回JAX array转换为tensor
            jax_buffer = self.k_buffer[layer_id - self.start_layer]
            if self.store_dtype != self.dtype:
                # 需要类型转换
                jax_buffer = jax_buffer.astype(torch_to_jax_dtype(self.dtype))
            return self._jax_to_tensor(jax_buffer)
        else:
            # 原始实现
            if self.store_dtype != self.dtype:
                return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
            return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.is_jax_backend and _has_jax:
            # TPU backend: 返回JAX array转换为tensor
            jax_buffer = self.v_buffer[layer_id - self.start_layer]
            if self.store_dtype != self.dtype:
                # 需要类型转换
                jax_buffer = jax_buffer.astype(torch_to_jax_dtype(self.dtype))
            return self._jax_to_tensor(jax_buffer)
        else:
            # 原始实现
            if self.store_dtype != self.dtype:
                return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
            return self.v_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        layer_id = layer.layer_id

        if self.is_jax_backend and _has_jax:
            # TPU backend的KV缓存设置
            if cache_k.dtype != self.dtype:
                if k_scale is not None:
                    cache_k = cache_k / k_scale
                if v_scale is not None:
                    cache_v = cache_v / v_scale
                cache_k = cache_k.to(self.dtype)
                cache_v = cache_v.to(self.dtype)

            if self.store_dtype != self.dtype:
                cache_k = cache_k.view(self.store_dtype)
                cache_v = cache_v.view(self.store_dtype)

            # 转换为JAX array
            k_jax = self._tensor_to_jax(cache_k)
            v_jax = self._tensor_to_jax(cache_v)

            # 使用JAX的索引更新
            self.k_buffer[layer_id - self.start_layer] = (
                self.k_buffer[layer_id - self.start_layer].at[loc].set(k_jax)
            )
            self.v_buffer[layer_id - self.start_layer] = (
                self.v_buffer[layer_id - self.start_layer].at[loc].set(v_jax)
            )
        else:
            # 原始CUDA实现
            if cache_k.dtype != self.dtype:
                if k_scale is not None:
                    cache_k.div_(k_scale)
                if v_scale is not None:
                    cache_v.div_(v_scale)
                cache_k = cache_k.to(self.dtype)
                cache_v = cache_v.to(self.dtype)

            if self.store_dtype != self.dtype:
                cache_k = cache_k.view(self.store_dtype)
                cache_v = cache_v.view(self.store_dtype)

            if get_is_capture_mode() and self.alt_stream is not None:
                # Overlap the copy of K and V cache for small batch size
                current_stream = self.device_module.current_stream()
                self.alt_stream.wait_stream(current_stream)
                self.k_buffer[layer_id - self.start_layer][loc] = cache_k
                with self.device_module.stream(self.alt_stream):
                    self.v_buffer[layer_id - self.start_layer][loc] = cache_v
                current_stream.wait_stream(self.alt_stream)
            else:
                self.k_buffer[layer_id - self.start_layer][loc] = cache_k
                self.v_buffer[layer_id - self.start_layer][loc] = cache_v


@triton.jit
def set_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim

    loc = tl.load(loc_ptr + pid_loc)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    if base + BLOCK <= nope_dim:
        src = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask,
        )
    else:
        offs_rope = offs - nope_dim
        src = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + offs_rope,
            mask=mask,
        )

    tl.store(dst_ptr, src, mask=mask)


def set_mla_kv_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128
    n_loc = loc.numel()
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))

    set_mla_kv_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
    )


class MLATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        backend: str = "torch",
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            backend,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        with self.memory_saver_adapter.region():
            if self.is_jax_backend and _has_jax:
                # 使用JAX创建MLA缓冲区
                self.kv_buffer = [
                    self._create_jax_buffer(
                        (size + page_size, 1, kv_lora_rank + qk_rope_head_dim)
                    )
                    for _ in range(layer_num)
                ]
            else:
                # 原始PyTorch实现
                self.kv_buffer = [
                    torch.zeros(
                        (size + page_size, 1, kv_lora_rank + qk_rope_head_dim),
                        dtype=self.store_dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]

        self.layer_transfer_counter = None

        kv_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, KV size: {kv_size / GB:.2f} GB"
            f"{' (TPU/JAX backend)' if self.is_jax_backend else ''}"
        )

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            if self.is_jax_backend and _has_jax:
                kv_size_bytes += np.prod(kv_cache.shape) * kv_cache.dtype.itemsize
            else:
                kv_size_bytes += np.prod(kv_cache.shape) * kv_cache.dtype.itemsize
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        if self.is_jax_backend and _has_jax:
            # TPU backend不支持直接获取指针信息
            logger.warning("get_contiguous_buf_infos not supported for TPU backend")
            return [], [], []

        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.is_jax_backend and _has_jax:
            # TPU backend: 返回JAX array转换为tensor
            jax_buffer = self.kv_buffer[layer_id - self.start_layer]
            if self.store_dtype != self.dtype:
                jax_buffer = jax_buffer.astype(torch_to_jax_dtype(self.dtype))
            return self._jax_to_tensor(jax_buffer)
        else:
            # 原始实现
            if self.store_dtype != self.dtype:
                return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)
            return self.kv_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.is_jax_backend and _has_jax:
            # TPU backend: 返回JAX array转换为tensor
            jax_buffer = self.kv_buffer[layer_id - self.start_layer][
                ..., : self.kv_lora_rank
            ]
            if self.store_dtype != self.dtype:
                jax_buffer = jax_buffer.astype(torch_to_jax_dtype(self.dtype))
            return self._jax_to_tensor(jax_buffer)
        else:
            # 原始实现
            if self.store_dtype != self.dtype:
                return self.kv_buffer[layer_id - self.start_layer][
                    ..., : self.kv_lora_rank
                ].view(self.dtype)
            return self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id

        if self.is_jax_backend and _has_jax:
            # TPU backend的KV缓存设置
            if cache_k.dtype != self.dtype:
                cache_k = cache_k.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k = cache_k.view(self.store_dtype)

            # 转换为JAX array
            k_jax = self._tensor_to_jax(cache_k)

            # 使用JAX的索引更新
            self.kv_buffer[layer_id - self.start_layer] = (
                self.kv_buffer[layer_id - self.start_layer].at[loc].set(k_jax)
            )
        else:
            # 原始实现
            if cache_k.dtype != self.dtype:
                cache_k = cache_k.to(self.dtype)
            if self.store_dtype != self.dtype:
                self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view(
                    self.store_dtype
                )
            else:
                self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id

        if self.is_jax_backend and _has_jax:
            # TPU backend的MLA KV缓存设置
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            # 转换为JAX array并合并
            k_nope_jax = self._tensor_to_jax(cache_k_nope)
            k_rope_jax = self._tensor_to_jax(cache_k_rope)
            combined_k = jnp.concatenate([k_nope_jax, k_rope_jax], axis=-1)

            # 使用JAX的索引更新
            self.kv_buffer[layer_id - self.start_layer] = (
                self.kv_buffer[layer_id - self.start_layer].at[loc].set(combined_k)
            )
        else:
            # 原始Triton实现
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id], loc, cache_k_nope, cache_k_rope
            )

    def get_flat_data(self, indices):
        # prepare a large chunk of contiguous data for efficient transfer
        if self.is_jax_backend and _has_jax:
            # TPU backend的扁平化数据
            jax_data = [self.kv_buffer[i][indices] for i in range(self.layer_num)]
            # 转换为tensor返回
            tensor_data = [self._jax_to_tensor(j) for j in jax_data]
            return torch.stack(tensor_data)
        else:
            # 原始实现
            return torch.stack(
                [self.kv_buffer[i][indices] for i in range(self.layer_num)]
            )

    @debug_timing
    def transfer(self, indices, flat_data):
        # transfer prepared data from host to device
        if self.is_jax_backend and _has_jax:
            # TPU backend的数据传输
            flat_data = flat_data.to(device=self.device, non_blocking=False)
            for i in range(self.layer_num):
                jax_data = self._tensor_to_jax(flat_data[i])
                self.kv_buffer[i] = self.kv_buffer[i].at[indices].set(jax_data)
        else:
            # 原始实现
            flat_data = flat_data.to(device=self.device, non_blocking=False)
            for i in range(self.layer_num):
                self.kv_buffer[i][indices] = flat_data[i]

    def transfer_per_layer(self, indices, flat_data, layer_id):
        # transfer prepared data from host to device
        if self.is_jax_backend and _has_jax:
            # TPU backend的逐层传输
            flat_data = flat_data.to(device=self.device, non_blocking=False)
            jax_data = self._tensor_to_jax(flat_data)
            self.kv_buffer[layer_id - self.start_layer] = (
                self.kv_buffer[layer_id - self.start_layer].at[indices].set(jax_data)
            )
        else:
            # 原始实现
            flat_data = flat_data.to(device=self.device, non_blocking=False)
            self.kv_buffer[layer_id - self.start_layer][indices] = flat_data


class DoubleSparseTokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        heavy_channel_num: int,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        backend: str = "torch",
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            backend,
        )

        with self.memory_saver_adapter.region():
            # [size, head_num, head_dim] for each layer
            self.k_buffer = [
                torch.zeros(
                    (size + page_size, head_num, head_dim), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (size + page_size, head_num, head_dim), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]

            # [size, head_num, heavy_channel_num] for each layer
            self.label_buffer = [
                torch.zeros(
                    (size + 1, head_num, heavy_channel_num), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]

    def get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_label_buffer(self, layer_id: int):
        return self.label_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_label: torch.Tensor,
    ):
        # NOTE(Andy): ignore the dtype check
        layer_id = layer.layer_id
        self.k_buffer[layer_id - self.start_layer][loc] = cache_k
        self.v_buffer[layer_id - self.start_layer][loc] = cache_v
        self.label_buffer[layer_id - self.start_layer][loc] = cache_label

    def get_flat_data(self, indices):
        pass

    def transfer(self, indices, flat_data):
        pass

    def transfer_per_layer(self, indices, flat_data, layer_id):
        pass
