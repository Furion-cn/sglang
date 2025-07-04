from typing import Optional, Sequence, Tuple, Iterable, Union

import jax
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
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
                nnx.initializers.normal(),
                expert_kernel_axes
            )(
                rngs.params(), 
                (self.experts_per_device, config.hidden_size, intermediate_dim), 
                weight_dtype
            )
        )
        
        self.wi_1 = nnx.Param(
            nnx.with_partitioning(
                nnx.initializers.normal(),
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
        
        state = nnx.state(self)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(self, sharded_state)

    def __call__(self, inputs, router_logits=None):
        if router_logits is None:
            raise ValueError("router_logits is required for Qwen3MoE")
            
        inputs = inputs.astype(self.dtype)
        total_tokens, hidden_dim = inputs.shape
        
        if router_logits.shape[0] != total_tokens:
            raise ValueError(f"router_logits shape {router_logits.shape} doesn't match inputs shape {inputs.shape}")
        
        print(f"MoE processing {total_tokens} tokens with {self.num_experts} experts")
        
        # 选择每个token的top-k专家
        top_k_logits, top_k_indices = jax.lax.top_k(router_logits, self.num_experts_per_tok)
        top_k_weights = jax.nn.softmax(top_k_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        
        # 使用简单但优化的方案
        if self.expert_parallel_size == 1:
            print("Using local forward mode")
            output = self._local_forward(inputs, top_k_indices, top_k_weights)
        else:
            print(f"Using expert parallel mode with {self.expert_parallel_size} devices")
            output = self._expert_parallel_forward(inputs, top_k_indices, top_k_weights)
        
        return output
    
    def _expert_parallel_forward(self, tokens, top_k_indices, top_k_weights):        
        print("Step 1: Permuting tokens by expert assignment")
        # Step 1: permute - global sorting and grouping
        sorted_inputs, sorted_selected_experts, weights, global_group_sizes, sorted_experts = self._permute_exact(
            tokens, top_k_indices, top_k_weights
        )
        
        print("Step 2: Dispatching tokens to expert devices")
        # Step 2: Expert parallel dispatch 
        expert_shard_id = jax.lax.axis_index(self.expert_axis_name)
        local_expert_size = self.experts_per_device
        
        if self.expert_parallel_size > 1:
            x, local_sorted_indices, local_group_sizes, selected_experts = self._expert_dispatch(
                sorted_inputs, global_group_sizes, sorted_experts, expert_shard_id, local_expert_size
            )
        else:
            # Single device mode
            x = sorted_inputs
            local_group_sizes = global_group_sizes
            selected_experts = sorted_experts
            local_sorted_indices = jnp.arange(len(sorted_inputs))
        
        print("Step 3: Computing expert outputs")
        # Step 3: GMM computation
        intermediate_output = self._gmm_compute_exact(x, local_group_sizes, selected_experts)
        
        print("Step 4: Collecting results from expert devices")
        # Step 4: Result collection
        if self.expert_parallel_size > 1:
            intermediate_output = self._result_collection(
                intermediate_output, local_sorted_indices, global_group_sizes, 
                expert_shard_id, local_expert_size, tokens.shape[0] * self.num_experts_per_tok
            )
        
        print("Step 5: Restoring original token order")
        # Step 5: Unpermute - restore original order
        final_output = self._unpermute_exact(
            intermediate_output, sorted_selected_experts, weights, 
            tokens.shape[0], tokens.shape[1]
        )
        
        print("MoE forward completed")
        return final_output
    
    def _permute_exact(self, inputs, top_k_indices, top_k_weights):
        inputs_shape = inputs.shape
        
        # Fix reshape logic: input is already (seq_len, hidden_dim)
        if len(inputs_shape) == 2:
            # Input is already 2D: (seq_len, hidden_dim)
            inputs_2d = inputs
            bsz_times_seq_len = inputs_shape[0]
        else:
            # Input is 3D: (batch, seq_len, hidden_dim) -> (batch*seq_len, hidden_dim)
            bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
            inputs_2d = jnp.reshape(inputs, (bsz_times_seq_len, inputs_shape[-1]))
        
        flatten_selected_experts = jnp.ravel(top_k_indices)
        sorted_selected_experts = jnp.argsort(flatten_selected_experts)
        sorted_indices = sorted_selected_experts // self.num_experts_per_tok
        
        # Sort inputs by expert
        sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0).astype(self.dtype)
        
        # Compute global group_sizes (number of tokens per expert)
        group_sizes = jnp.bincount(flatten_selected_experts, length=self.num_experts)
        
        # Generate sorted_experts
        expert_indices = jnp.arange(self.num_experts)
        sorted_experts = jnp.repeat(expert_indices, repeats=group_sizes, total_repeat_length=flatten_selected_experts.shape[0])
        
        return sorted_inputs, sorted_selected_experts, top_k_weights, group_sizes, sorted_experts
    
    def _expert_dispatch(self, sorted_inputs, global_group_sizes, sorted_experts, expert_shard_id, local_expert_size):
        # global_group_sizes: (num_experts,) -> reshaped: (num_expert_parallelism,)
        reshaped_group_sizes = jnp.sum(global_group_sizes.reshape(self.expert_parallel_size, local_expert_size), axis=1)
        
        # Unified communication abstraction
        x, local_sorted_indices, local_group_sizes, selected_experts = self._unified_expert_communication(
            sorted_inputs, global_group_sizes, sorted_experts, expert_shard_id, 
            local_expert_size, reshaped_group_sizes, is_dispatch=True
        )
        
        return x, local_sorted_indices, local_group_sizes, selected_experts
    
    def _unified_expert_communication(self, data, global_group_sizes, sorted_experts, expert_shard_id, 
                                     local_expert_size, reshaped_group_sizes, is_dispatch=True):
        try:
            devices = jax.devices()
            is_cpu_only = all(device.platform == 'cpu' for device in devices)
            can_use_ragged = not is_cpu_only and hasattr(jax.lax, 'ragged_all_to_all')
        except:
            is_cpu_only = True
            can_use_ragged = False
        
        mode = "ragged_all_to_all" if can_use_ragged else "regular_all_to_all"
        action = "dispatching" if is_dispatch else "collecting"
        print(f"Communication: {action} using {mode}")
        
        if can_use_ragged:
            # GPU/TPU: use ragged_all_to_all
            return self._ragged_communication(
                data, global_group_sizes, sorted_experts, expert_shard_id, 
                local_expert_size, reshaped_group_sizes, is_dispatch
            )
        else:
            # CPU: use regular all_to_all
            return self._regular_communication(
                data, global_group_sizes, sorted_experts, expert_shard_id, 
                local_expert_size, reshaped_group_sizes, is_dispatch
            )
    
    def _ragged_communication(self, data, global_group_sizes, sorted_experts, expert_shard_id, 
                             local_expert_size, reshaped_group_sizes, is_dispatch):        
        input_offsets, send_sizes, output_offsets, recv_sizes = self._get_all_to_all_params(
            reshaped_group_sizes[None, :], expert_shard_id, self.expert_parallel_size, is_batch_sharded=False
        )
        
        if is_dispatch:
            buffer_size = int(self.expert_parallel_size * data.shape[0])
            output_shape = jnp.zeros((buffer_size, data.shape[1]), dtype=data.dtype)
            
            x = jax.lax.ragged_all_to_all(
                data, output_shape, input_offsets, send_sizes,
                output_offsets, recv_sizes, axis_name=self.expert_axis_name,
            )
            
            x, local_sorted_indices, local_group_sizes, selected_experts = self._local_permute_exact(
                x, global_group_sizes[None, :], local_expert_size, expert_shard_id
            )
            
            return x, local_sorted_indices, local_group_sizes, selected_experts
        else:
            original_inputs_first_dim = recv_sizes.shape[0] if len(recv_sizes.shape) > 0 else data.shape[0]
            output_shape = jnp.zeros((original_inputs_first_dim, data.shape[1]), dtype=data.dtype)
            
            result = jax.lax.ragged_all_to_all(
                data, output_shape, input_offsets, send_sizes,
                output_offsets, recv_sizes, axis_name=self.expert_axis_name,
            )
            
            return result
    
    def _regular_communication(self, data, global_group_sizes, sorted_experts, expert_shard_id, 
                              local_expert_size, reshaped_group_sizes, is_dispatch):
        total_data = data.shape[0]
        remainder = total_data % self.expert_parallel_size
        padding_needed = (self.expert_parallel_size - remainder) % self.expert_parallel_size
        target_size = total_data + padding_needed
        
        if padding_needed > 0:
            padding_shape = (padding_needed, data.shape[1])
            padding_data = jnp.zeros(padding_shape, dtype=data.dtype)
            padded_data = jnp.concatenate([data, padding_data], axis=0)
        else:
            padded_data = data
        
        tokens_per_device = target_size // self.expert_parallel_size
        reshaped_data = padded_data.reshape(
            self.expert_parallel_size, tokens_per_device, data.shape[1]
        )
        
        communicated_data = jax.lax.all_to_all(
            reshaped_data,
            axis_name=self.expert_axis_name,
            split_axis=0,
            concat_axis=1
        )
        
        flattened_data = communicated_data.reshape(-1, data.shape[1])
        
        if is_dispatch:
            all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
                global_group_sizes[None, :], expert_shard_id * local_expert_size, local_expert_size, axis=1
            )
            local_group_sizes = jnp.sum(all_shard_local_sizes, axis=0)  # tokens per local expert

            num_valid_tokens = jnp.sum(local_group_sizes)
            
            expert_ids = jnp.arange(local_expert_size)
            sorted_experts_ids = jnp.repeat(expert_ids, local_group_sizes, total_repeat_length=num_valid_tokens)
            
            valid_data = flattened_data[:num_valid_tokens]
            local_sorted_indices = jnp.arange(num_valid_tokens)
            
            return valid_data, local_sorted_indices, local_group_sizes, sorted_experts_ids
        else:
            return flattened_data
    
    def _local_permute_exact(self, inputs, global_group_sizes, local_expert_size, shard_index, is_offset=False, global_sorted_experts=None):
        all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
            global_group_sizes, shard_index * local_expert_size, local_expert_size, axis=1
        )
        local_sizes = all_shard_local_sizes.reshape(-1)
        
        local_group_size = jnp.sum(all_shard_local_sizes, axis=0)
        
        if is_offset:
            divided_assignments = jnp.floor_divide(global_sorted_experts, local_expert_size)
            expert_indices = jnp.where(
                divided_assignments == shard_index, 
                jnp.mod(global_sorted_experts, local_expert_size), 
                local_expert_size
            )
        else:
            base_indices = jnp.mod(jnp.arange(local_sizes.shape[0]), local_expert_size)
            expert_indices = jnp.repeat(base_indices, local_sizes, total_repeat_length=inputs.shape[0])
        
        # Sort by local expert ID
        sorted_indices = jnp.argsort(expert_indices)
        sorted_inputs = jnp.take(inputs, indices=sorted_indices, axis=0)
        sorted_experts_ids = expert_indices[sorted_indices]
        
        return sorted_inputs, sorted_indices, local_group_size, sorted_experts_ids
    
    def _gmm_compute_exact(self, x, local_group_sizes, selected_experts):
        def gmm_layer(inputs, kernel, group_sizes, expert_assignments):
            return jax.lax.ragged_dot(
                lhs=inputs,
                rhs=kernel,
                group_sizes=group_sizes,
                preferred_element_type=self.dtype
            )
        
        w0_kernel = self.wi_0.value
        w1_kernel = self.wi_1.value
        wo_kernel = self.wo.value

        print(f"w0 sharding: {w0_kernel.sharding}")
        print(f"w1 sharding: {w1_kernel.sharding}")
        print(f"wo sharding: {wo_kernel.sharding}")
        
        # Key understanding: JAX sharding keeps weights in global shape (128) in code, but local_group_sizes is local size (16)
        # Need to expand local_group_sizes to global expert count to match weight shape
        expected_global_experts = w0_kernel.shape[0]  # global expert count (128)
        local_expert_count = len(local_group_sizes)   # local expert count (16)
        
        if local_expert_count != expected_global_experts:
            print(f"Expanding group_sizes from {local_expert_count} to {expected_global_experts}")
            expert_shard_id = jax.lax.axis_index(self.expert_axis_name)
            experts_per_device = self.experts_per_device
            local_expert_start = expert_shard_id * experts_per_device
            local_expert_end = (expert_shard_id + 1) * experts_per_device
            
            expanded_group_sizes = jnp.zeros(expected_global_experts, dtype=local_group_sizes.dtype)
            
            expanded_group_sizes = expanded_group_sizes.at[local_expert_start:local_expert_end].set(local_group_sizes)
            
            final_group_sizes = expanded_group_sizes
        else:
            final_group_sizes = local_group_sizes
        
        print(f"GMM computing with {jnp.sum(final_group_sizes)} tokens across {len(final_group_sizes)} experts")
        
        layer_w0 = gmm_layer(x, w0_kernel, final_group_sizes, selected_experts)
        layer_w1 = gmm_layer(x, w1_kernel, final_group_sizes, selected_experts)
        
        layer_act = jax.nn.silu(layer_w0)
        intermediate_layer = jnp.multiply(layer_act, layer_w1)
        
        intermediate_output = gmm_layer(intermediate_layer, wo_kernel, final_group_sizes, selected_experts)
        
        return intermediate_output
    
    def _result_collection(self, intermediate_output, local_sorted_indices, global_group_sizes, 
                                   expert_shard_id, local_expert_size, original_inputs_first_dim):        
        if len(intermediate_output) == 0:
            return jnp.zeros((original_inputs_first_dim, intermediate_output.shape[-1]), dtype=self.dtype)
        
        local_output = jnp.take(intermediate_output, indices=jnp.argsort(local_sorted_indices), axis=0)
        
        reshaped_group_sizes = jnp.sum(global_group_sizes.reshape(self.expert_parallel_size, local_expert_size), axis=1)
        
        result = self._unified_expert_communication(
            local_output, global_group_sizes, None, expert_shard_id,
            local_expert_size, reshaped_group_sizes, is_dispatch=False
        )
        
        if result.shape[0] > original_inputs_first_dim:
            result = result[:original_inputs_first_dim]
        elif result.shape[0] < original_inputs_first_dim:
            padding_size = original_inputs_first_dim - result.shape[0]
            padding = jnp.zeros((padding_size, result.shape[1]), dtype=result.dtype)
            result = jnp.concatenate([result, padding], axis=0)
        
        return result

    def _unpermute_exact(self, intermediate, sorted_selected_experts, weights, batch_size, sequence_length):        
        unsort_intermediate = jnp.take(intermediate, indices=jnp.argsort(sorted_selected_experts), axis=0)
        
        reshaped_weights = jnp.reshape(weights, (-1, self.num_experts_per_tok))
        
        reshaped_intermediate = jnp.reshape(
            unsort_intermediate,
            (reshaped_weights.shape[0], self.num_experts_per_tok, -1),
        )
        
        output = jnp.einsum(
            "BKE,BK -> BE",
            reshaped_intermediate.astype(jnp.float32),
            reshaped_weights.astype(jnp.float32),
            precision=jax.lax.Precision.DEFAULT,
        )
        final_output = output.astype(self.dtype)
        
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

    def _get_all_to_all_params(self, all_shards_group_sizes, shard_id, num_expert_parallelism, is_batch_sharded=True):        
        def transform_array(input_array, shard_id, strategy, is_batch_sharded):
            if is_batch_sharded:
                if strategy == "INPUT_OFFSET":
                    local_array = input_array[shard_id]
                    return jnp.concatenate((jnp.array([0]), jnp.cumsum(local_array)[:-1]))
                elif strategy == "SEND_SIZE":
                    return input_array[shard_id]
                elif strategy == "OUTPUT_OFFSET":
                    zero_row = jnp.zeros((1,) + input_array.shape[1:], dtype=input_array.dtype)
                    array_with_zeros = jnp.concatenate((zero_row, input_array), axis=0)
                    cumulated_array = jnp.cumsum(array_with_zeros, axis=0, dtype=input_array.dtype)
                    return cumulated_array[shard_id]
                elif strategy == "RECV_SIZE":
                    return input_array[:, shard_id]
            else:
                if strategy == "INPUT_OFFSET":
                    return jnp.zeros(num_expert_parallelism, dtype=input_array.dtype)
                elif strategy == "SEND_SIZE":
                    return jnp.repeat(input_array[shard_id], num_expert_parallelism)
                elif strategy == "OUTPUT_OFFSET":
                    output_offset = jnp.concatenate((jnp.array([0]), jnp.cumsum(input_array[:-1])))[shard_id]
                    return jnp.repeat(output_offset, num_expert_parallelism)
                elif strategy == "RECV_SIZE":
                    return input_array
            raise ValueError(f"Unknown transform strategy: {strategy}")
        
        input_offsets = transform_array(all_shards_group_sizes, shard_id, "INPUT_OFFSET", is_batch_sharded)
        send_sizes = transform_array(all_shards_group_sizes, shard_id, "SEND_SIZE", is_batch_sharded)
        output_offsets = transform_array(all_shards_group_sizes, shard_id, "OUTPUT_OFFSET", is_batch_sharded)
        recv_sizes = transform_array(all_shards_group_sizes, shard_id, "RECV_SIZE", is_batch_sharded)
        
        return input_offsets, send_sizes, output_offsets, recv_sizes

    def _get_device_expert_range(self):
        expert_ids = jnp.arange(self.num_experts)
        expert_ids_sharded = jax.lax.with_sharding_constraint(
            expert_ids,
            jax.sharding.PartitionSpec(self.expert_axis_name)
        )
        
        local_expert_start = expert_ids_sharded[0]  # 第一个专家ID
        local_expert_end = expert_ids_sharded[-1] + 1  # 最后一个专家ID + 1
        
        return local_expert_start, local_expert_end