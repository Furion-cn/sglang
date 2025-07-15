import os
from contextlib import nullcontext
from typing import Sequence, Tuple

import jax
from jax import numpy as jnp
import numpy as np
import torch
from jax._src import mesh_utils
from sglang.srt.jax.model_executor.forward_batch_info import ForwardBatch, ForwardMode

mesh_axes = [
    "data",  # data parallelism
    "tensor",  # tensor parallelism
    "pipeline",  # pipeline parallelism
    "expert",  # expert parallelism
]


def create_device_mesh(ici_parallelism: Sequence[int],
                       dcn_parallelism: Sequence[int],
                       devices=None,
                       num_slices: int = 1,
                       allow_split_physical_axes: bool = True) -> jax.sharding.Mesh:
    """Create a device mesh"""
    if devices is None:
        devices = jax.devices()

    ici_parallelism = fill_unspecified_parallelism(
        ici_parallelism, len(devices))
    if num_slices > 1:
        dcn_parallelism = fill_unspecified_parallelism(
            dcn_parallelism, num_slices)
        devices_array = mesh_utils.create_hybrid_device_mesh(
            ici_parallelism,
            dcn_parallelism,
            devices=devices,
            allow_split_physical_axes=allow_split_physical_axes,
        )
    else:
        devices_array = mesh_utils.create_device_mesh(
            ici_parallelism,
            devices=devices,
            contiguous_submeshes=False,
            allow_split_physical_axes=allow_split_physical_axes,
        )
    mesh = jax.sharding.Mesh(devices_array, mesh_axes)
    return mesh


def fill_unspecified_parallelism(parallelism: Sequence[int], num_devices: int) -> Sequence[int]:
    if -1 not in parallelism:
        return parallelism

    assert parallelism.count(-1) == 1, "At most one axis can be unspecified."
    unspecified_axis_idx = parallelism.index(-1)
    determined_val = num_devices / np.prod(parallelism) * -1
    assert determined_val >= 1 and determined_val.is_integer, "Unspecified value unable to be determined with the given parallelism values"
    parallelism[unspecified_axis_idx] = int(determined_val)
    return parallelism


def convert_jax_array_to_torch_tensor(jax_array: jax.Array) -> torch.Tensor:
    numpy_array = np.array(jax_array)
    return torch.from_numpy(numpy_array)


def update_forward_batch(forward_batch: ForwardBatch, next_token_ids, tokenizer, finished_requests, original_indices):
    """Update forward batch while handling finished requests"""
    new_input_ids = []
    new_seq_lens = []
    new_original_indices = []
    new_cache_loc = []

    cache_loc_start_loc = 0
    for batch_idx, seq_len in enumerate(forward_batch.seq_lens):
        orig_idx = original_indices[batch_idx]
        current_token_id = int(next_token_ids[batch_idx, 0])
        cache_loc = forward_batch.cache_loc[cache_loc_start_loc:
                                            cache_loc_start_loc + seq_len].tolist()
        cache_loc_start_loc += seq_len

        # Check if this request should finish BEFORE updating sequences
        if is_finished(current_token_id, tokenizer):
            print(
                f"🛑 Request {orig_idx} will be removed from batch (token: {current_token_id})")
            finished_requests.add(orig_idx)
            continue

        # Only update sequences for non-finished requests
        new_input_ids.append(current_token_id)
        new_seq_lens.append(seq_len + 1)
        new_original_indices.append(orig_idx)
        new_cache_loc.append(cache_loc)

    if len(new_seq_lens) == 0:
        # All requests are finished
        return None

    # Update batch with only unfinished requests
    forward_batch.batch_size = len(new_seq_lens)
    forward_batch.seq_lens = jnp.array(new_seq_lens, dtype=jnp.int32)

    # update cache loc
    out_cache_start_loc = max(
        item for sublist in new_cache_loc for item in sublist) + 1
    forward_batch.out_cache_loc = jnp.arange(
        out_cache_start_loc, out_cache_start_loc + forward_batch.batch_size, dtype=jnp.int32)
    forward_batch.cache_loc = jnp.array([
        item for i, cache_loc in enumerate(new_cache_loc)
        for item in cache_loc + [int(forward_batch.out_cache_loc[i])]
    ], dtype=jnp.int32)

    # Update positions for decode mode
    forward_batch.positions = jnp.array(
        [seq_len - 1 for seq_len in new_seq_lens], dtype=jnp.int32)

    # Update input ids
    forward_batch.input_ids = jnp.array(new_input_ids, dtype=jnp.int32)

    # Update extend start loc
    forward_batch.extend_start_loc = jnp.cumsum(
        jnp.concatenate([jnp.array([0]), forward_batch.seq_lens[:-1]]))

    # Update forward mode
    if forward_batch.forward_mode == ForwardMode.EXTEND:
        forward_batch.forward_mode = ForwardMode.DECODE

    return new_original_indices


def is_finished(token_id, tokenizer):
    """Check if a token indicates the end of generation"""
    return (token_id == tokenizer.eos_token_id or
            token_id == 151643 or  # Common stop token
            token_id == 151645)


def jax_trace_context(log_dir: str):
    """Return a JAX trace context manager with options configured via env vars.

    The following environment variables are honored (all optional):

    1. ``JAX_TRACE_CREATE_PERFETTO_LINK`` – Boolean-like string (``1``, ``0``). Controls ``create_perfetto_link``.

    Example::

        os.environ["JAX_TRACE_HOST_TRACER_LEVEL"] = "2"
        with jax_trace_context("/tmp/trace"):
            ...  # code to profile
    """

    jax_trace_enabled = os.getenv("ENABLE_JAX_TRACE", "1")
    if jax_trace_enabled == "0":
        return nullcontext()

    create_perfetto_link = os.getenv(
        "JAX_TRACE_CREATE_PERFETTO_LINK", "1") == "1"

    return jax.profiler.trace(log_dir, create_perfetto_trace=True,
                              create_perfetto_link=create_perfetto_link)
