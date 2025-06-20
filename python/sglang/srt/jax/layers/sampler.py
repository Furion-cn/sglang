from typing import List

import jax
from flax import nnx
from jax import numpy as jnp
from jax import random

from sglang.srt.jax.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.jax.sampling.sampling_batch_info import SamplingBatchInfo


class Sampler(nnx.Module):
    def __init__(self, rngs: nnx.Rngs = None):
        self.rngs = rngs

    def __call__(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
            sampling_info: Metadata for sampling
            return_logprob: If set, store the output logprob information to
                logits_output
            top_logprobs_nums: Number of top lobprobs per sequence in a batch
            batch_next_token_ids: next token IDs. If set, skip sampling and only
                compute output logprobs It is used for speculative decoding which
                performs sampling in draft workers.
        """
        logits = jnp.reshape(logits_output.next_token_logits,
                             (-1, logits_output.next_token_logits.shape[-1]))

        if sampling_info.is_all_greedy:
            # Use torch.argmax if all requests use greedy sampling
            batch_next_token_ids = jnp.argmax(logits, -1).reshape(-1, 1)
        else:
            # Post process logits
            probs = jnp.divide(logits, sampling_info.temperatures)
            _, new_rng = jax.random.split(self.rngs.params())
            # A slower fallback implementation with torch native operations.
            batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                probs,
                sampling_info.top_ks,
                sampling_info.top_ps,
                sampling_info.min_ps,
                sampling_info.need_min_p_sampling,
                new_rng
            )
        return batch_next_token_ids


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: jax.Array,
    top_ks: jax.Array,
    top_ps: jax.Array,
    min_ps: jax.Array,
    need_min_p_sampling: bool,
    rng: nnx.Rngs,
):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    probs_sort = jnp.sort(
        probs, axis=-1)[:, ::-1]  # Sort and reverse for descending order
    # Get indices and reverse for descending order
    probs_idx = jnp.argsort(probs, axis=-1)[:, ::-1]
    probs_sum = jnp.cumsum(probs_sort, axis=-1)

    # Apply top-k filtering using jnp.where instead of in-place assignment
    top_k_mask = jnp.arange(
        0, probs.shape[-1]).reshape(1, -1) >= top_ks.reshape(-1, 1)
    probs_sort = jnp.where(top_k_mask, 0.0, probs_sort)

    # Apply top-p filtering using jnp.where instead of in-place assignment
    top_p_mask = (probs_sum - probs_sort) > top_ps.reshape(-1, 1)
    probs_sort = jnp.where(top_p_mask, 0.0, probs_sort)

    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        # Apply min-p filtering using jnp.where instead of in-place assignment
        min_p_mask = probs_sort < min_p_thresholds.reshape(-1, 1)
        probs_sort = jnp.where(min_p_mask, 0.0, probs_sort)

    sampled_index = random.categorical(rng, probs_sort)
    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.astype(jnp.int32)
    # sampled_index has shape (batch_size,), need to reshape to (batch_size, 1) for take_along_axis
    sampled_index = sampled_index.reshape(-1, 1)
    batch_next_token_ids = jnp.take_along_axis(
        probs_idx, axis=1, indices=sampled_index).reshape(-1, 1)
    return batch_next_token_ids
