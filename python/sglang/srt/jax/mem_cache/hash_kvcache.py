import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Dict, List
from sglang.srt.jax.mem_cache.memory_pool import KVCache

class ReqToHashKVCachePool(KVCache):
    def __init__(
        self,
        head_num: int,
        head_dim: int,
        layer_num: int,
        max_seq_len: int,
        dtype: jnp.dtype,
    ):
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self._create_cache()

    def _create_cache(self):
        self.k_cache = [
            jnp.zeros(
                (self.max_seq_len, self.head_num*self.head_dim), dtype=self.dtype)
            for _ in range(self.layer_num)
        ]
        self.v_cache = [
            jnp.zeros(
                (self.max_seq_len, self.head_num*self.head_dim), dtype=self.dtype)
            for _ in range(self.layer_num)
        ]
    
    def get_kv_buffer(self, layer_id: int) -> Tuple[jax.Array, jax.Array]:
       return self.k_cache[layer_id], self.v_cache[layer_id]

    def set_kv_buffer(
        self,
        layer_id: int,
        cache_k: jax.Array,
        cache_v: jax.Array
    ):
        self.k_cache[layer_id] = cache_k
        self.v_cache[layer_id] = cache_v


class HashKVCache:
    def __init__(
        self,
        head_num: int,
        head_dim: int,
        layer_num: int,
        max_seq_len: int,
        dtype: jnp.dtype,
    ):
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.req_to_kv_cache_pool_map: Dict[str, ReqToHashKVCachePool] = {}

    def add(self, req: str):
        self.req_to_kv_cache_pool_map[req] = ReqToHashKVCachePool(
            self.head_num,
            self.head_dim,
            self.layer_num,
            self.max_seq_len,
            self.dtype)
        
    def remove(self, req: str):
        del self.req_to_kv_cache_pool_map[req]

    def get_kv_cache(self, req: str, layer_id: int) -> Tuple[jax.Array, jax.Array]:
        if req not in self.req_to_kv_cache_pool_map:
            raise ValueError(f"Request '{req}' not found in kv cache")
        return self.req_to_kv_cache_pool_map[req].get_kv_buffer(layer_id)

    def set_kv_cache(
        self,
        req: str,
        layer_id: int,
        cache_k: jax.Array,
        cache_v: jax.Array
    ):
        self.req_to_kv_cache_pool_map[req].set_kv_buffer(layer_id, cache_k, cache_v)
