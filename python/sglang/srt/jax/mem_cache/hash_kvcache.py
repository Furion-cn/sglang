from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

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
    return k_cache[layer_id], v_cache[layer_id]


@partial(jax.jit, static_argnames=["layer_id"])
def set_kv_cache(
    layer_id: int,
    loc: jax.Array,
    k: jax.Array,
    v: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    k_cache = k_cache.at[layer_id, loc].set(k)
    v_cache = v_cache.at[layer_id, loc].set(v)

    return k_cache, v_cache
