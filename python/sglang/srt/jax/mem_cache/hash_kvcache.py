from functools import partial
from typing import Dict, Tuple
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
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
        return _get_kv_buffer(layer_id, self.k_cache, self.v_cache)

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: jax.Array,
        cache_k: jax.Array,
        cache_v: jax.Array
    ):
        self.k_cache, self.v_cache = _set_kv_cache(
            layer_id, loc, cache_k, cache_v,
            self.k_cache, self.v_cache
        )
    

def create_kv_cache(
    max_seq_len:int,
    max_batch_size:int,
    head_num:int,
    head_dim:int, 
    layer_num:int,
    dtype:jnp.dtype,
):
    max_tokens = max_seq_len * max_batch_size
    hidden_dim = head_num * head_dim

    k_cache = jnp.zeros(
        (layer_num, max_tokens, hidden_dim),
        dtype=dtype
    )
    v_cache = jnp.zeros(
        (layer_num, max_tokens, hidden_dim),
        dtype=dtype
    )
    return k_cache,v_cache

def get_kv_buffer(k_cache,v_cache, layer_id: int) -> Tuple[jax.Array, jax.Array]:
    return k_cache[layer_id], v_cache[layer_id]

def set_kv_buffer(
    layer_id: int,
    loc: jax.Array,
    cache_k: jax.Array,
    cache_v: jax.Array,
    k_cache:jax.Array,
    v_cache:jax.Array,
):
    # k_cache, v_cache = _set_kv_cache(
    #     layer_id, loc, cache_k, cache_v,
    #     k_cache, v_cache
    # )
    assert loc.shape[0] == cache_k.shape[0] == cache_v.shape[0], "Batch size mismatch"
    # print(f"layer_id: {layer_id}, loc.shape: {loc.shape}, loc: {loc}")
    # print(f"k_cache: {k_cache.shape}, k: {cache_k.shape}")
    # print(f"v_cache: {v_cache.shape}, v: {cache_k.shape}")

    k_cache = k_cache.at[layer_id, loc].set(cache_k)
    v_cache = v_cache.at[layer_id, loc].set(cache_v)

    return k_cache, v_cache
    # return k_cache,v_cache


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def _kv_cache_update_kernel(
    # Prefetch
    slices_ref,  # [3, padded_num_slices], list of (kv_cache_start,
    # new_kv_start, slice_len)
    # Input
    new_kv_hbm_ref,  # [num_tokens, num_combined_kv_heads, head_dim]
    kv_cache_hbm_ref,  # [total_num_pages * page_size, num_combined_kv_heads,
    # head_dim]
    # Output
    _,  # [total_num_pages * page_size, num_combined_kv_heads, head_dim]
    # Scratch
    scratch,  # [num_slices_per_block, page_size, num_combined_kv_heads,
    # head_dim]
    sem,
):
    async_copies = []
    block_idx = pl.program_id(0)
    num_slices_per_block = scratch.shape[0]
    # Copy from new_kv_hbm_ref to scratch
    for i in range(num_slices_per_block):
        offset_i = i + block_idx * num_slices_per_block
        new_kv_start = slices_ref[1, offset_i]
        length = slices_ref[2, offset_i]
        async_copy = pltpu.make_async_copy(
            new_kv_hbm_ref.at[pl.ds(new_kv_start, length), ...],
            scratch.at[i, pl.ds(0, length), ...],
            sem,
        )
        async_copy.start()
        async_copies.append(async_copy)

    for async_copy in async_copies:
        async_copy.wait()

    # Copy from scratch to kv_cache_hbm_ref
    async_copies.clear()
    for i in range(num_slices_per_block):
        offset_i = i + block_idx * num_slices_per_block
        kv_cache_start = slices_ref[0, offset_i]
        length = slices_ref[2, offset_i]
        async_copy = pltpu.make_async_copy(
            scratch.at[i, pl.ds(0, length), ...],
            kv_cache_hbm_ref.at[pl.ds(kv_cache_start, length), ...],
            sem,
        )
        async_copy.start()
        async_copies.append(async_copy)
    for async_copy in async_copies:
        async_copy.wait()


@partial(
    jax.jit,
    static_argnames=["page_size", "num_slices_per_block"],
)
def kv_cache_update(
    new_kv: jax.Array,  # [total_num_token, num_combined_kv_heads, head_dim]
    # [3, slices], list of (kv_cache_start, new_kv_start, slice_len)
    slices: jax.Array,
    # [total_num_pages * page_size, num_combined_kv_heads, head_dim]
    kv_cache: jax.Array,
    num_kv_update_slices: jax.Array,  # [1]
    *,
    page_size: int = 1024,
    num_slices_per_block: int = 8,
):
    assert slices.shape[1] % num_slices_per_block == 0, f"slices.shape[1]={slices.shape[1]} is not divisible by num_slices_per_block={num_slices_per_block}"
    _, num_combined_kv_heads, head_dim = new_kv.shape
    assert kv_cache.shape[1] == num_combined_kv_heads, f"kv_cache.shape[1]={kv_cache.shape[1]} is not equal to num_combined_kv_heads={num_combined_kv_heads}"
    assert kv_cache.shape[2] == head_dim, f"kv_cache.shape[2]={kv_cache.shape[2]} is not equal to head_dim={head_dim}"
    assert head_dim % 128 == 0, f"head_dim={head_dim} is not divisible by 128"
    # TODO: Add dynamic check to make sure that the all the slice lengths are
    # smaller or equal to page_size

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
    ]

    out_specs = [pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY)]
    out_shape = [jax.ShapeDtypeStruct(kv_cache.shape, dtype=kv_cache.dtype)]

    scalar_prefetches = [slices]
    scratch = pltpu.VMEM(
        (num_slices_per_block, page_size, num_combined_kv_heads, head_dim),
        new_kv.dtype,
    )

    scratch_shapes = [
        scratch,
        pltpu.SemaphoreType.DMA,
    ]

    kernel = pl.pallas_call(
        _kv_cache_update_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            in_specs=in_specs,
            out_specs=out_specs,
            grid=(cdiv(num_kv_update_slices[0], num_slices_per_block), ),
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shape,
        input_output_aliases={len(scalar_prefetches) + 1: 0},
    )
    
    cur_mesh = jax.sharding.get_abstract_mesh()
    if not cur_mesh.empty:
        kernel = shard_map(kernel, mesh=jax.sharding.get_abstract_mesh(
        ), in_specs=P(None,), out_specs=P(None,), check_rep=False)

    return kernel(*scalar_prefetches, new_kv, kv_cache)[0]



def _get_slot_mapping(
    num_slices_per_block: int,
    kv_cache_start_loc: jax.Array,
    new_kv_start_loc: jax.Array,
    slice_lens: jax.Array,
):
    slot_mapping = jnp.stack(
        [kv_cache_start_loc, new_kv_start_loc, slice_lens], axis=1)
    padded_size = (slot_mapping.shape[0] + num_slices_per_block -
                   1) // num_slices_per_block * num_slices_per_block
    slot_mapping = jnp.pad(slot_mapping,
                           [[0, padded_size - slot_mapping.shape[0]], [0, 0]],
                           constant_values=0)
    slot_mapping = jnp.transpose(slot_mapping)
    return slot_mapping


VME_SIZE = 32 * 1024 * 1024 # 32MB
NUM_SLICES_PER_BLOCK = 4
PAGE_SIZE = 1024


@jax.jit
def update_kv_cache(
    k: jax.Array,          # padding key (batch_size)
    v: jax.Array,          # padding value
    k_cache: jax.Array,
    v_cache: jax.Array,
    seq_lens: jax.Array,   # (batch_size, )
    kv_start_loc: jax.Array,
    kv_cache_start_loc: jax.Array,  # (batch_size, )
):
    batch_size = seq_lens.shape[0]
    num_kv_update_slices = jnp.array([batch_size], dtype=jnp.int32)
    get_slot_mapping = partial(
        _get_slot_mapping, kv_cache_start_loc=kv_cache_start_loc, new_kv_start_loc=kv_start_loc, slice_lens=seq_lens)
    slot_mapping = get_slot_mapping(NUM_SLICES_PER_BLOCK)

    k_cache = kv_cache_update(k, slot_mapping, k_cache, num_kv_update_slices,
                    num_slices_per_block=NUM_SLICES_PER_BLOCK, page_size=PAGE_SIZE)
    v_cache = kv_cache_update(v, slot_mapping, v_cache, num_kv_update_slices,
                    num_slices_per_block=NUM_SLICES_PER_BLOCK, page_size=PAGE_SIZE)
    return k_cache, v_cache



@partial(jax.jit, static_argnames=["layer_id"])
def _get_kv_buffer(layer_id: int, k_cache: jax.Array, v_cache: jax.Array) -> Tuple[jax.Array, jax.Array]:
    return k_cache[layer_id], v_cache[layer_id]


@partial(jax.jit, static_argnames=["layer_id"])
def _set_kv_cache(
    layer_id: int,
    loc: jax.Array,
    k: jax.Array,
    v: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    assert loc.shape[0] == k.shape[0] == v.shape[0], "Batch size mismatch"
    # print(f"layer_id: {layer_id}, loc.shape: {loc.shape}, loc: {loc}")
    # print(f"k_cache: {k_cache.shape}, k: {k.shape}")
    # print(f"v_cache: {v_cache.shape}, v: {v.shape}")

    k_cache = v_cache.at[layer_id, loc].set(k)
    v_cache = v_cache.at[layer_id, loc].set(v)

    return k_cache, v_cache
