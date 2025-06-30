import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Dict, List
from sglang.srt.jax.mem_cache.memory_pool import KVCache

class ReqToHashKVCachePool(KVCache):
    def __init__(
        self,
        seq_len: int,
        head_num: int,
        head_dim: int,
        layer_num: int,
        dtype: jnp.dtype,
    ):
        self.seq_len = seq_len
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.dtype = dtype
        self._create_cache()

    def _create_cache(self):
        self.k_cache = [
            jnp.zeros(
                (self.seq_len, self.head_num*self.head_dim), dtype=self.dtype)
            for _ in range(self.layer_num)
        ]
        self.v_cache = [
            jnp.zeros(
                (self.seq_len, self.head_num*self.head_dim), dtype=self.dtype)
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
    ):
        self.req_to_kv_cache_pool_map: Dict[str, ReqToHashKVCachePool] = {}

    def get_kv_cache(self, req: str, layer_id: int) -> Tuple[jax.Array, jax.Array]:
        if req not in self.req_to_kv_cache_pool_map:
            raise ValueError(f"Request {req} not found in kv cache")
        return self.req_to_kv_cache_pool_map[req].get_kv_buffer(layer_id)

    def set_kv_cache(
        self,
        req: str,
        req_to_kv_cache_pool: ReqToHashKVCachePool
    ):
        self.req_to_kv_cache_pool_map[req] = req_to_kv_cache_pool
