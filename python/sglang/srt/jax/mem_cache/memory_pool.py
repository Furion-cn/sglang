from abc import ABC, abstractmethod
from typing import Tuple

import jax


class KVCache(ABC):
    @abstractmethod
    def __init__(
        self,
    ):
        raise NotImplementedError()

    @abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[jax.Array, jax.Array]:
        raise NotImplementedError()

    @abstractmethod
    def set_kv_buffer(
        self,
        layer_id: int,
        cache_k: jax.Array,
        cache_v: jax.Array,
    ) -> None:
        raise NotImplementedError()
