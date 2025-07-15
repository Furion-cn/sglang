from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P, Mesh, NamedSharding
from sglang.srt.jax.mem_cache.memory_pool import KVCache


class ReqToHashKVCachePool(KVCache):
    def __init__(
        self,
        head_num: int,
        head_dim: int,
        layer_num: int,
        max_seq_len: int,
        max_batch_size: int,
        dtype: jnp.dtype,
    ):
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self._create_cache()

    def _create_cache(self):
        max_tokens = self.max_seq_len * self.max_batch_size
        hidden_dim = self.head_num * self.head_dim

        self.k_cache = jnp.zeros(
            (self.layer_num, max_tokens, hidden_dim),
            dtype=self.dtype
        )
        self.v_cache = jnp.zeros(
            (self.layer_num, max_tokens, hidden_dim),
            dtype=self.dtype
        )
        
        print(f"[KV Cache] Checking DP sharding for cache: {self.k_cache.shape}")
        
        # 获取当前mesh环境
        current_mesh = jax.experimental.maps.thread_resources.env.physical_mesh
        
        if current_mesh is not None and 'data' in current_mesh.axis_names:
            try:
                # 使用当前设备创建对齐的mesh
                current_devices = list(jax.devices())
                aligned_mesh = Mesh(current_devices, axis_names=('data',))
                cache_pspec = P(None, 'data', None)
                cache_sharding = NamedSharding(aligned_mesh, cache_pspec)
                
                print(f"  - Applying DP sharding with aligned mesh: {aligned_mesh}")
                
                self.k_cache = jax.device_put(self.k_cache, cache_sharding)
                self.v_cache = jax.device_put(self.v_cache, cache_sharding)
                
                print(f"[KV Cache] DP sharding applied successfully")
                print(f"  - K cache sharding: {self.k_cache.sharding}")
                print(f"  - V cache sharding: {self.v_cache.sharding}")
            except Exception as e:
                print(f"[KV Cache] Failed to apply DP sharding: {e}, continuing without constraint")
        else:
            print(f"[KV Cache] No DP mesh detected, skipping sharding")

    def get_kv_buffer(self, layer_id: int) -> Tuple[jax.Array, jax.Array]:
        return get_kv_buffer(layer_id, self.k_cache, self.v_cache)

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: jax.Array,
        cache_k: jax.Array,
        cache_v: jax.Array
    ):
        self.k_cache, self.v_cache = set_kv_cache(
            layer_id, loc, cache_k, cache_v,
            self.k_cache, self.v_cache
        )


@partial(jax.jit, static_argnames=["layer_id"])
def get_kv_buffer(layer_id: int, k_cache: jax.Array, v_cache: jax.Array) -> Tuple[jax.Array, jax.Array]:
    k_buffer = k_cache[layer_id]
    v_buffer = v_cache[layer_id]
    
    # 获取当前mesh并尝试应用DP分片，如果失败则继续
    current_mesh = jax.experimental.maps.thread_resources.env.physical_mesh
    if current_mesh is not None and 'data' in current_mesh.axis_names:
        try:
            # 使用实际设备创建对齐的mesh
            devices = list(k_buffer.sharding.device_set)
            if len(devices) > 1:  # 只在多设备时应用分片
                aligned_mesh = Mesh(devices, axis_names=('data',))
                pspec = P('data', None)
                buffer_sharding = NamedSharding(aligned_mesh, pspec)
                
                k_buffer = jax.device_put(k_buffer, buffer_sharding)
                v_buffer = jax.device_put(v_buffer, buffer_sharding)
        except Exception:
            # 如果分片失败，继续执行不应用约束
            pass
    
    return k_buffer, v_buffer


@partial(jax.jit, static_argnames=["layer_id"])
def set_kv_cache(
    layer_id: int,
    loc: jax.Array,
    k: jax.Array,
    v: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    assert loc.shape[0] == k.shape[0] == v.shape[0], "Batch size mismatch"
    
    # 尝试应用DP分片，如果失败则继续
    current_mesh = jax.experimental.maps.thread_resources.env.physical_mesh
    if current_mesh is not None and 'data' in current_mesh.axis_names:
        try:
            # 为k, v应用DP分片
            kv_devices = list(k.sharding.device_set)
            if len(kv_devices) > 1:
                aligned_mesh = Mesh(kv_devices, axis_names=('data',))
                pspec = P('data', None)
                kv_sharding = NamedSharding(aligned_mesh, pspec)
                
                k = jax.device_put(k, kv_sharding)
                v = jax.device_put(v, kv_sharding)
            
            # 为cache应用DP分片
            cache_devices = list(k_cache.sharding.device_set)
            if len(cache_devices) > 1:
                aligned_mesh = Mesh(cache_devices, axis_names=('data',))
                cache_pspec = P(None, 'data', None) if k_cache.ndim == 3 else P('data', None)
                cache_sharding = NamedSharding(aligned_mesh, cache_pspec)
                
                k_cache = jax.device_put(k_cache, cache_sharding)
                v_cache = jax.device_put(v_cache, cache_sharding)
        except Exception:
            # 如果分片失败，继续执行不应用约束
            pass

    k_cache = k_cache.at[layer_id, loc].set(k)
    v_cache = v_cache.at[layer_id, loc].set(v)

    return k_cache, v_cache
