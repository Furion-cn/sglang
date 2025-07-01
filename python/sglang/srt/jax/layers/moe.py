from typing import Optional, Sequence, Tuple, Iterable, Union, DType

import jax
from jax.sharding import Mesh
from flax import nnx
from jax import numpy as jnp
from sglang.srt.jax.layers import linear

class GateLogit(nnx.Module):
    """A layer used to compute gate logits, allowing to return the pre bias values for DeepSeek routing.

    Attributes:
        input_size: input dimension of the layer.
        features: tuple with numbers of output features.
        model_name: which model to run.
        axis: tuple with axes to apply the transformation on.
        weight_dtype: the dtype of the weights (default: float32).
        dtype: the dtype of the computation (default: float32).
        kernel_axes: tuple with axes to apply kernel function.
        use_bias: whether to add learnable bias in gate logit scores.
          When enabled, this bias aids expert load balancing (like in DeepSeek V3),
          and is not part of the loss calculation.
        score_func: scoring function for output normalization before applying bias.
        matmul_precision: precision for JAX functions.
    """

    def __init__(
            self,
            input_size: int,
            features: Union[Iterable[int], int],
            model_name: str,
            axis: Union[Iterable[int], int] = -1,
            weight_dtype: jnp.dtype = jnp.float32,
            dtype: jnp.dtype = jnp.float32,
            kernel_axes: Optional[Sequence[str]] = None,
            use_bias: bool = False,
            score_func: str = "",
            matmul_precision: str = "default",
            rngs: nnx.Rngs = None):
        
        self.features = linear._canonicalize_tuple(features)
        self.axis = linear._canonicalize_tuple(axis)
        self.model_name = model_name
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.use_bias = use_bias
        self.score_func = score_func
        self.matmul_precision = matmul_precision
        
        self.kernel_axes = kernel_axes or ()
        
        kernel_shape = (input_size,) + self.features
        
        self.kernel = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), self.kernel_axes)(
                rngs.params(), kernel_shape, self.weight_dtype
            )
        )
        
        if self.use_bias:
            bias_shape = self.features
            bias_axes = self.kernel_axes[-len(self.features):] if self.kernel_axes else ()
            self.bias = nnx.Param(
                nnx.with_partitioning(nnx.initializers.zeros_init(), bias_axes)(
                    rngs.params(), bias_shape, self.weight_dtype
                )
            )
        else:
            self.bias = None

    def __call__(self, inputs: jax.Array) -> Tuple[jax.Array, Optional[jax.Array]]:
        inputs = jnp.asarray(inputs, self.dtype)
        
        kernel = jnp.asarray(self.kernel.value, self.dtype)
        output = jnp.dot(inputs, kernel)
                
        if self.score_func:
            if self.score_func == "softmax":
                output = jax.nn.softmax(output)
            elif self.score_func == "sigmoid": 
                output = jax.nn.sigmoid(output)
            elif self.score_func == "tanh":
                output = jax.nn.tanh(output)
        
        if self.use_bias and self.bias is not None:
            bias = jnp.asarray(self.bias.value, self.dtype)
            output += bias
            
        return output

class Qwen3MoE(nnx.Module):
    def __init__(self,
                 config,
                 num_experts: int,
                 num_experts_per_tok: int,
                 intermediate_dim: int = 2048,
                 weight_dtype: jnp.dtype = jnp.bfloat16,
                 dtype: jnp.dtype = jnp.bfloat16,
                 expert_axis_name: str = 'expert',
                 rngs: nnx.Rngs = None):
        
        self.config = config
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.expert_axis_name = expert_axis_name
        
        self.mesh = getattr(config, 'expert_mesh', None)
        if self.mesh is None:
            raise ValueError("need to provide mesh with expert axis")
        
        self.has_expert_parallelism = expert_axis_name in self.mesh.axis_names
        if not self.has_expert_parallelism:
            raise ValueError(f"{expert_axis_name} axis is not in mesh")
            
        self.expert_parallel_size = self.mesh.shape[expert_axis_name]
        
        if num_experts % self.expert_parallel_size != 0:
            raise ValueError(f"num_experts({num_experts}) must be divisible by expert_parallel_size({self.expert_parallel_size})")
        
        self.experts_per_device = num_experts // self.expert_parallel_size
        
        expert_kernel_axes = (expert_axis_name, None, None)
        
        self.wi_0 = nnx.Param(
            nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02),
                expert_kernel_axes
            )(
                rngs.params(), 
                (self.experts_per_device, config.hidden_size, intermediate_dim), 
                weight_dtype
            )
        )
        
        self.wi_1 = nnx.Param(
            nnx.with_partitioning(
                nnx.initializers.normal(s),
                expert_kernel_axes
            )(
                rngs.params(), 
                (self.experts_per_device, config.hidden_size, intermediate_dim), 
                weight_dtype
            )
        )
        
        self.wo = nnx.Param(
            nnx.with_partitioning(
                nnx.initializers.normal(),
                expert_kernel_axes
            )(
                rngs.params(), 
                (self.experts_per_device, intermediate_dim, config.hidden_size), 
                weight_dtype
            )
        )

    def __call__(self, inputs, router_logits=None):
        if router_logits is None:
            raise ValueError("router_logits is required for Qwen3MoE")
            
        inputs = inputs.astype(self.dtype)
        batch_size, seq_len, hidden_dim = inputs.shape
        
        top_k_logits, top_k_indices = jax.lax.top_k(router_logits, self.num_experts_per_tok)
        top_k_weights = jax.nn.softmax(top_k_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        
        num_tokens = batch_size * seq_len
        tokens = inputs.reshape(num_tokens, hidden_dim)  # (num_tokens, hidden_dim)
        top_k_indices_flat = top_k_indices.reshape(num_tokens, self.num_experts_per_tok)
        top_k_weights_flat = top_k_weights.reshape(num_tokens, self.num_experts_per_tok)
        
        output = self._expert_parallel_forward(
            tokens, top_k_indices_flat, top_k_weights_flat
        )
        
        output = output.reshape(batch_size, seq_len, hidden_dim)
        
        return output
        
    def _get_all_to_all_params(self, device_group_sizes, shard_id):
        input_offsets = jnp.zeros(self.expert_parallel_size, dtype=jnp.int32)
        
        send_sizes = jnp.repeat(device_group_sizes[shard_id], self.expert_parallel_size)

        output_offset = jnp.concatenate([jnp.array([0]), jnp.cumsum(device_group_sizes[:-1])])[shard_id]
        output_offsets = jnp.repeat(output_offset, self.expert_parallel_size)
        
        recv_sizes = device_group_sizes
        
        return input_offsets, send_sizes, output_offsets, recv_sizes

    def _dispatch_tokens(self, tokens, top_k_indices, top_k_weights, device_mask):
        num_tokens, hidden_dim = tokens.shape
        
        expanded_tokens = jnp.repeat(tokens[:, None, :], self.num_experts_per_tok, axis=1)
        expanded_tokens = expanded_tokens.reshape(-1, hidden_dim)
        
        expanded_indices = top_k_indices.reshape(-1)
        expanded_weights = top_k_weights.reshape(-1)
        
        token_positions = jnp.repeat(jnp.arange(num_tokens)[:, None], self.num_experts_per_tok, axis=1)
        token_positions = token_positions.reshape(-1)
        
        group_sizes = jnp.bincount(expanded_indices, length=self.num_experts)
        
        local_expert_size = self.experts_per_device
        device_group_sizes = jnp.zeros(self.expert_parallel_size, dtype=jnp.int32)
        for device in range(self.expert_parallel_size):
            start_expert = device * local_expert_size
            end_expert = (device + 1) * local_expert_size
            device_group_sizes = device_group_sizes.at[device].set(
                jnp.sum(group_sizes[start_expert:end_expert])
            )
        
        sorted_indices = jnp.argsort(expanded_indices)
        sorted_tokens = expanded_tokens[sorted_indices]
        sorted_expert_ids = expanded_indices[sorted_indices]
        sorted_weights = expanded_weights[sorted_indices]
        sorted_positions = token_positions[sorted_indices]
        
        sorted_weights_float = sorted_weights.astype(self.dtype)
        sorted_positions_float = sorted_positions.astype(self.dtype)
        
        combined_data = jnp.concatenate([
            sorted_tokens,                        # (N, hidden_dim)
            sorted_weights_float[..., None],      # (N, 1)
            sorted_positions_float[..., None]     # (N, 1)
        ], axis=-1)  # (N, hidden_dim + 2)
        
        expert_shard_id = jax.lax.axis_index(self.expert_axis_name)
        input_offsets, send_sizes, output_offsets, recv_sizes = self._get_all_to_all_params(
            device_group_sizes, expert_shard_id
        )
        
        total_recv_tokens = jnp.sum(recv_sizes)
        output_shape = jnp.zeros((total_recv_tokens, hidden_dim + 2), dtype=self.dtype)
        
        dispatched_combined = jax.lax.ragged_all_to_all(
            combined_data,
            output_shape,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name=self.expert_axis_name
        )
        
        dispatched_tokens = dispatched_combined[..., :hidden_dim]
        dispatched_weights = dispatched_combined[..., hidden_dim].astype(self.dtype)
        dispatched_positions = dispatched_combined[..., hidden_dim + 1].astype(jnp.int32)
        
        local_expert_start = expert_shard_id * self.experts_per_device
        local_expert_end = (expert_shard_id + 1) * self.experts_per_device
        local_group_sizes = group_sizes[local_expert_start:local_expert_end]
        
        return dispatched_tokens, dispatched_weights, dispatched_positions, local_group_sizes

    def _compute_expert_outputs(self, dispatched_tokens, dispatched_weights, dispatched_positions, local_group_sizes):
        if dispatched_tokens.shape[0] == 0:
            return jnp.zeros((0, dispatched_tokens.shape[-1] + 1), dtype=self.dtype)
        
        expert_outputs = []
        token_start = 0
        
        for local_expert_id in range(self.experts_per_device):
            expert_token_count = local_group_sizes[local_expert_id]
            
            if expert_token_count == 0:
                continue
            
            token_end = token_start + expert_token_count
            expert_tokens = dispatched_tokens[token_start:token_end]  # (expert_token_count, hidden_dim)
            expert_weights = dispatched_weights[token_start:token_end]  # (expert_token_count,)
            expert_positions = dispatched_positions[token_start:token_end]  # (expert_token_count,)
            
            wi_0 = self.wi_0.value[local_expert_id]  # (hidden_dim, intermediate_dim)
            wi_1 = self.wi_1.value[local_expert_id]  # (hidden_dim, intermediate_dim)
            wo = self.wo.value[local_expert_id]      # (intermediate_dim, hidden_dim)
            
            gate_output = jnp.dot(expert_tokens, wi_0)  # (expert_token_count, intermediate_dim)
            up_output = jnp.dot(expert_tokens, wi_1)    # (expert_token_count, intermediate_dim)
            activated = jax.nn.silu(gate_output) * up_output  # (expert_token_count, intermediate_dim)
            expert_result = jnp.dot(activated, wo)  # (expert_token_count, hidden_dim)
            
            weighted_result = expert_result * expert_weights[:, None]  # (expert_token_count, hidden_dim)
            
            position_info = expert_positions.astype(self.dtype)[:, None]
            output_with_position = jnp.concatenate([weighted_result, position_info], axis=-1)
            
            expert_outputs.append(output_with_position)
            token_start = token_end
        
        if expert_outputs:
            return jnp.concatenate(expert_outputs, axis=0)
        else:
            return jnp.zeros((0, dispatched_tokens.shape[-1] + 1), dtype=self.dtype)

    def _combine_outputs(self, expert_outputs, num_tokens, hidden_dim, original_group_sizes):
        expert_shard_id = jax.lax.axis_index(self.expert_axis_name)
        
        local_expert_size = self.experts_per_device
        device_group_sizes = jnp.zeros(self.expert_parallel_size, dtype=jnp.int32)
        for device in range(self.expert_parallel_size):
            start_expert = device * local_expert_size
            end_expert = (device + 1) * local_expert_size
            device_group_sizes = device_group_sizes.at[device].set(
                jnp.sum(original_group_sizes[start_expert:end_expert])
            )
        
        input_offsets, send_sizes, output_offsets, recv_sizes = self._get_all_to_all_params(
            device_group_sizes, expert_shard_id
        )
        
        total_recv_tokens = jnp.sum(recv_sizes)
        output_shape = jnp.zeros((total_recv_tokens, hidden_dim + 1), dtype=self.dtype)
        
        combined_outputs_with_positions = jax.lax.ragged_all_to_all(
            expert_outputs,
            output_shape,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name=self.expert_axis_name
        )
        
        final_output = jnp.zeros((num_tokens, hidden_dim), dtype=self.dtype)
        
        if combined_outputs_with_positions.shape[0] > 0:
            combined_outputs = combined_outputs_with_positions[..., :hidden_dim]
            token_positions = combined_outputs_with_positions[..., hidden_dim].astype(jnp.int32)
            
            final_output = final_output.at[token_positions].add(combined_outputs)
        
        return final_output
    
    def _expert_parallel_forward(self, tokens, top_k_indices, top_k_weights):
        num_tokens, hidden_dim = tokens.shape
        
        if self.expert_parallel_size == 1:
            return self._local_forward(tokens, top_k_indices, top_k_weights)
        
        device_id = jax.lax.axis_index(self.expert_axis_name)
        expert_start = device_id * self.experts_per_device
        expert_end = (device_id + 1) * self.experts_per_device
        
        device_mask = (top_k_indices >= expert_start) & (top_k_indices < expert_end)
        
        expanded_indices = top_k_indices.reshape(-1)
        original_group_sizes = jnp.bincount(expanded_indices, length=self.num_experts)
        
        dispatched_tokens, dispatched_weights, dispatched_positions, local_group_sizes = self._dispatch_tokens(
            tokens, top_k_indices, top_k_weights, device_mask
        )
        
        expert_outputs = self._compute_expert_outputs(
            dispatched_tokens, dispatched_weights, dispatched_positions, local_group_sizes
        )
        
        final_output = self._combine_outputs(expert_outputs, num_tokens, hidden_dim, original_group_sizes)
        
        return final_output

    def _local_forward(self, tokens, top_k_indices, top_k_weights):
        num_tokens, hidden_dim = tokens.shape
        
        expert_weights = jnp.zeros((num_tokens, self.num_experts), dtype=self.dtype)
        
        token_indices = jnp.arange(num_tokens)[:, None]  # (num_tokens, 1)
        expert_weights = expert_weights.at[token_indices, top_k_indices].set(top_k_weights)
        
        all_wi_0 = self.wi_0.value  # (experts_per_device, hidden_dim, intermediate_dim)
        all_wi_1 = self.wi_1.value  # (experts_per_device, hidden_dim, intermediate_dim)
        all_wo = self.wo.value      # (experts_per_device, intermediate_dim, hidden_dim)
        
        layer_w0 = jnp.einsum('th,ehd->ted', tokens, all_wi_0)  # (num_tokens, experts_per_device, intermediate_dim)
        layer_w1 = jnp.einsum('th,ehd->ted', tokens, all_wi_1)  # (num_tokens, experts_per_device, intermediate_dim)
        
        activated = jax.nn.silu(layer_w0) * layer_w1  # (num_tokens, experts_per_device, intermediate_dim)
        
        expert_outputs = jnp.einsum('ted,edh->teh', activated, all_wo)  # (num_tokens, experts_per_device, hidden_dim)
        
        final_output = jnp.einsum('te,teh->th', expert_weights, expert_outputs)  # (num_tokens, hidden_dim)
        
        return final_output
