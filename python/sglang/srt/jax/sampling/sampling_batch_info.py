from dataclasses import dataclass

import jax


@dataclass
class SamplingBatchInfo:
    # Basic batched sampling params
    temperatures: jax.Array
    top_ps: jax.Array
    top_ks: jax.Array
    min_ps: jax.Array

    # Masking tensors for grammar-guided structured outputs
    vocab_size: int

    # Whether all requests use greedy sampling
    is_all_greedy: bool = False

    # Whether any requests use top_p sampling
    need_top_p_sampling: bool = False

    # Whether any requests use top_k sampling
    need_top_k_sampling: bool = False

    # Whether any request needs min_p sampling
    need_min_p_sampling: bool = False

    def tree_flatten(self):
        children = (
            self.temperatures,
            self.top_ps,
            self.top_ks,
            self.min_ps,
        )
        aux_data = (
            self.vocab_size,
            self.is_all_greedy,
            self.need_top_p_sampling,
            self.need_top_k_sampling,
            self.need_min_p_sampling,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            vocab_size,
            is_all_greedy,
            need_top_p_sampling,
            need_top_k_sampling,
            need_min_p_sampling,
        ) = aux_data
        (
            temperatures,
            top_ps,
            top_ks,
            min_ps,
        ) = children
        return cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            vocab_size=vocab_size,
            is_all_greedy=is_all_greedy,
            need_top_p_sampling=need_top_p_sampling,
            need_top_k_sampling=need_top_k_sampling,
            need_min_p_sampling=need_min_p_sampling,
        )


jax.tree_util.register_pytree_node_class(SamplingBatchInfo)

