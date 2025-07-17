import unittest

import jax
import jax.numpy as jnp
from sglang.test.test_utils import CustomTestCase
import random
from sglang.srt.jax.mem_cache.hash_kvcache import update_kv_cache
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P


class TestKVCache(CustomTestCase):
    """Test cases for the Attention layer."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

        # PyTorch comparison test parameters
        try:
            import torch
            self.pytorch_available = True
        except ImportError:
            self.pytorch_available = False

        self.max_seq_len = 16
        self.num_heads = 2
        self.head_dim = 128
        self.batch_size = 2
        self.layer_num = 2

    def generate_test_data(self, layer_idx: int, is_prefill: bool):
        max_total_len_of_one_layer = self.batch_size*self.max_seq_len
        kv_cache_of_one_layer = jnp.zeros(
            (max_total_len_of_one_layer, self.num_heads, self.head_dim), dtype=jnp.bfloat16)
        self.k_cache = jnp.concatenate(
            [kv_cache_of_one_layer] * self.layer_num, axis=0)
        self.v_cache = jnp.concatenate(
            [kv_cache_of_one_layer] * self.layer_num, axis=0)

        if is_prefill:
            self.k = jax.random.uniform(
                jax.random.PRNGKey(42), (self.batch_size*self.max_seq_len, self.num_heads, self.head_dim), dtype=jnp.bfloat16)
            self.v = jax.random.uniform(
                jax.random.PRNGKey(42), (self.batch_size*self.max_seq_len, self.num_heads, self.head_dim), dtype=jnp.bfloat16)
            self.seq_lens = jax.random.randint(
                jax.random.PRNGKey(42), (self.batch_size,), 1, self.max_seq_len+1, dtype=jnp.int32)
            self.kv_start_loc = jnp.arange(
                self.batch_size, dtype=jnp.int32) * self.max_seq_len
        else:
            self.k = jax.random.uniform(
                jax.random.PRNGKey(42), (self.batch_size, self.num_heads, self.head_dim), dtype=jnp.bfloat16)
            self.v = jax.random.uniform(
                jax.random.PRNGKey(42), (self.batch_size, self.num_heads, self.head_dim), dtype=jnp.bfloat16)
            self.seq_lens = jnp.ones((self.batch_size,), dtype=jnp.int32)
            self.kv_start_loc = jnp.arange(
                self.batch_size, dtype=jnp.int32)

        max_real_seq_len = jnp.max(self.seq_lens)
        max_offset = self.max_seq_len - max_real_seq_len
        self.kv_cache_start_loc = max_total_len_of_one_layer*layer_idx + (jnp.arange(
            self.batch_size, dtype=jnp.int32) * self.max_seq_len + jax.random.randint(jax.random.PRNGKey(42), (self.batch_size,), 0, max_offset, dtype=jnp.int32))

        return self.k, self.v, self.k_cache, self.v_cache, self.kv_cache_start_loc, self.seq_lens, self.kv_start_loc

    def expected_at_set_update_kv_cache(self, k, v, k_cache, v_cache, kv_cache_start_loc, k_seq_lens, k_start_loc):
        batch_size = k_seq_lens.shape[0]
        for i in range(batch_size):
            seq_len = k_seq_lens[i]
            cache_loc = jnp.arange(
                seq_len, dtype=jnp.int32) + kv_cache_start_loc[i]
            k_cache = k_cache.at[cache_loc].set(k[k_start_loc[i]:k_start_loc[i]+seq_len, :, :])
            v_cache = v_cache.at[cache_loc].set(v[k_start_loc[i]:k_start_loc[i]+seq_len, :, :])
        return k_cache, v_cache

    def test_kv_cache_update_prefill_without_mesh(self):
        test_loop = random.randint(1, self.layer_num)
        for layer_idx in range(test_loop):
            k, v, k_cache, v_cache, kv_cache_start_loc, k_seq_lens, k_start_loc = self.generate_test_data(
                layer_idx, is_prefill=True)
            k_cache_dump, v_cache_dump = k_cache.copy(), v_cache.copy()
            # test_output
            k_cache, v_cache = update_kv_cache(k, v, k_cache, v_cache, k_seq_lens, k_start_loc, kv_cache_start_loc)
            # expected data
            expected_k_cache, expected_v_cache = self.expected_at_set_update_kv_cache(
                k, v, k_cache_dump, v_cache_dump, kv_cache_start_loc, k_seq_lens, k_start_loc)
            self.assertTrue(jnp.allclose(k_cache, expected_k_cache))
            self.assertTrue(jnp.allclose(v_cache, expected_v_cache))

    def test_kv_cache_update_decode_with_mesh(self):
        mesh = jax.make_mesh((2, 1, 1), ('x', 'y', 'z'))
        sharding= jax.sharding.NamedSharding(mesh, P('x',))
        jax.sharding.set_mesh(mesh)
        test_loop = random.randint(1, self.layer_num)
        for layer_idx in range(test_loop):
            k, v, k_cache, v_cache, kv_cache_start_loc, k_seq_lens, k_start_loc = self.generate_test_data(
                layer_idx, is_prefill=False)
            k_cache_dump, v_cache_dump = k_cache.copy(), v_cache.copy()
            k, v = jax.device_put(k, sharding), jax.device_put(v, sharding)
            # test_output
            k_cache, v_cache = update_kv_cache(k, v, k_cache, v_cache, k_seq_lens, k_start_loc, kv_cache_start_loc)
            # expected data
            expected_k_cache, expected_v_cache = self.expected_at_set_update_kv_cache(
                k, v, k_cache_dump, v_cache_dump, kv_cache_start_loc, k_seq_lens, k_start_loc)
            self.assertTrue(jnp.allclose(k_cache, expected_k_cache))
            self.assertTrue(jnp.allclose(v_cache, expected_v_cache))


if __name__ == "__main__":
    unittest.main()