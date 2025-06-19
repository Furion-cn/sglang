
from typing import Sequence, Tuple

import jax
from jax._src.mesh_utils import create_hybrid_device_mesh

mesh_axes = [
    "data",  # data parallelism
    "tensor",  # tensor parallelism
    "pipeline",  # pipeline parallelism
    "expert",  # expert parallelism
]


def create_device_mesh(ici_parallelism: Sequence[int],
                       dcn_parallelism: Sequence[int],
                       devices=None,
                       allow_split_physical_axes: bool = True) -> jax.sharding.Mesh:
    """Create a device mesh"""
    if devices is None:
        devices = jax.devices()

    devices_array = create_hybrid_device_mesh(
        ici_parallelism,
        dcn_parallelism,
        devices=devices,
        allow_split_physical_axes=allow_split_physical_axes,
    )
    mesh = jax.sharding.Mesh(devices_array, mesh_axes)
    return mesh
