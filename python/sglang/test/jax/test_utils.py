import os
from contextlib import nullcontext
from typing import Sequence, Tuple

import jax
import numpy as np
import torch
from jax._src import mesh_utils

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

    return jax.profiler.trace(log_dir,
                              create_perfetto_link=create_perfetto_link)
