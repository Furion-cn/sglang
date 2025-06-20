import jax

class SamplingBatchInfo:
    # Basic batched sampling params
    temperatures: jax.Array
    top_ps: jax.Array
    top_ks: jax.Array
    min_ps: jax.Array

    # Whether all requests use greedy sampling
    is_all_greedy: bool

    # Whether any requests use top_p sampling
    need_top_p_sampling: bool

    # Whether any requests use top_k sampling
    need_top_k_sampling: bool

    # Whether any request needs min_p sampling
    need_min_p_sampling: bool

    # Masking tensors for grammar-guided structured outputs
    vocab_size: int
