from typing import Optional, Sequence, Tuple, Iterable, Union

import jax
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from flax import nnx
from jax import numpy as jnp
from sglang.srt.jax.layers import linear
from sglang.debug_tracer import global_tracer, trace_function

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
            layer_id: int = 0,
            rngs: nnx.Rngs = None):
        
        self.features = linear._canonicalize_tuple(features)
        self.axis = linear._canonicalize_tuple(axis)
        self.model_name = model_name
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.use_bias = use_bias
        self.score_func = score_func
        self.matmul_precision = matmul_precision
        self.layer_id = layer_id
        
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

    @trace_function(stage="MOE_GATE_FORWARD", include_args=False, include_output=True)
    def __call__(self, inputs: jax.Array) -> Tuple[jax.Array, Optional[jax.Array]]:
        inputs = jnp.asarray(inputs, self.dtype)
        
        global_tracer.print(inputs, f"gate_input", f"moe_gate_layer_id_{self.layer_id}")
        
        kernel = jnp.asarray(self.kernel.value, self.dtype)
        output = jnp.dot(inputs, kernel)
        
        global_tracer.print(output, f"gate_raw_output", f"moe_gate_layer_id_{self.layer_id}")
                
        if self.score_func:
            if self.score_func == "softmax":
                output = jax.nn.softmax(output)
            elif self.score_func == "sigmoid": 
                output = jax.nn.sigmoid(output)
            elif self.score_func == "tanh":
                output = jax.nn.tanh(output)
            
            global_tracer.print(output, f"gate_after_score_func", f"moe_gate_layer_id_{self.layer_id}")
        
        if self.use_bias and self.bias is not None:
            bias = jnp.asarray(self.bias.value, self.dtype)
            output += bias
            global_tracer.print(output, f"gate_after_bias", f"moe_gate_layer_id_{self.layer_id}")
        
        global_tracer.print(output, f"gate_final_output", f"moe_gate_layer_id_{self.layer_id}")
            
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
                 layer_id: int = 0,
                 rngs: nnx.Rngs = None):
        
        self.config = config
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.expert_axis_name = expert_axis_name
        self.layer_id = layer_id
        
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

    @trace_function(stage="MOE_SPARSE_FORWARD", include_args=False, include_output=True)
    def __call__(self, inputs, router_logits=None):
        if router_logits is None:
            raise ValueError("router_logits is required for Qwen3MoE")
            
        inputs = inputs.astype(self.dtype)
        total_tokens, hidden_dim = inputs.shape
        
        global_tracer.print(inputs, f"moe_input", f"moe_sparse_layer_id_{self.layer_id}")
        global_tracer.print(router_logits, f"router_logits", f"moe_sparse_layer_id_{self.layer_id}")
        
        if router_logits.shape[0] != total_tokens:
            raise ValueError(f"router_logits shape {router_logits.shape} doesn't match inputs shape {inputs.shape}")
        
        print(f"MoE processing {total_tokens} tokens with {self.num_experts} experts")
        
        top_k_logits, top_k_indices = jax.lax.top_k(router_logits, self.num_experts_per_tok)
        top_k_weights = jax.nn.softmax(top_k_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        
        global_tracer.print(top_k_logits, f"top_k_logits", f"moe_sparse_layer_id_{self.layer_id}")
        global_tracer.print(top_k_indices, f"top_k_indices", f"moe_sparse_layer_id_{self.layer_id}")
        global_tracer.print(top_k_weights, f"top_k_weights", f"moe_sparse_layer_id_{self.layer_id}")
        
        if self.expert_parallel_size == 1:
            print("Using local forward mode")
            output = self._local_forward(inputs, top_k_indices, top_k_weights)
        else:
            print(f"Using expert parallel mode with {self.expert_parallel_size} devices")
            output = self._expert_parallel_forward(inputs, top_k_indices, top_k_weights)
        
        global_tracer.print(output, f"moe_final_output", f"moe_sparse_layer_id_{self.layer_id}")
        
        return output
    
    def _expert_parallel_forward(self, tokens, top_k_indices, top_k_weights):        
        print(f"[DEBUG] _expert_parallel_forward: Starting MoE forward")
        print(f"[DEBUG] _expert_parallel_forward: tokens.shape={tokens.shape}")
        print(f"[DEBUG] _expert_parallel_forward: top_k_indices.shape={top_k_indices.shape}")
        print(f"[DEBUG] _expert_parallel_forward: top_k_weights.shape={top_k_weights.shape}")
        print(f"[DEBUG] _expert_parallel_forward: num_experts={self.num_experts}, num_experts_per_tok={self.num_experts_per_tok}")
        print(f"[DEBUG] _expert_parallel_forward: expert_parallel_size={self.expert_parallel_size}, experts_per_device={self.experts_per_device}")
        
        print("Step 1: Permuting tokens by expert assignment")
        # Step 1: permute - global sorting and grouping
        sorted_inputs, sorted_selected_experts, weights, global_group_sizes, sorted_experts = self._permute_exact(
            tokens, top_k_indices, top_k_weights
        )
        
        global_tracer.print(sorted_inputs, f"moe_permute_output", f"moe_dispatch_layer_id_{self.layer_id}")
        global_tracer.print(global_group_sizes, f"global_group_sizes", f"moe_dispatch_layer_id_{self.layer_id}")
        
        print("Step 2: Dispatching tokens to expert devices")
        # Step 2: Expert parallel dispatch 
        expert_shard_id = jax.lax.axis_index(self.expert_axis_name)
        local_expert_size = self.experts_per_device
        
        print(f"[DEBUG] _expert_parallel_forward: current expert_shard_id={expert_shard_id}, local_expert_size={local_expert_size}")
        
        if self.expert_parallel_size > 1:
            x, local_sorted_indices, local_group_sizes, selected_experts = self._expert_dispatch(
                sorted_inputs, global_group_sizes, sorted_experts, expert_shard_id, local_expert_size
            )
        else:
            # Single device mode
            print(f"[DEBUG] _expert_parallel_forward: using single device mode")
            x = sorted_inputs
            local_group_sizes = global_group_sizes
            selected_experts = sorted_experts
            local_sorted_indices = jnp.arange(len(sorted_inputs))
        
        print(f"[DEBUG] _expert_parallel_forward: after dispatch - x.shape={x.shape}")
        print(f"[DEBUG] _expert_parallel_forward: local_group_sizes={local_group_sizes}")
        
        global_tracer.print(x, f"moe_dispatch_output", f"moe_dispatch_layer_id_{self.layer_id}")
        global_tracer.print(local_group_sizes, f"local_group_sizes", f"moe_dispatch_layer_id_{self.layer_id}")
        
        print("Step 3: Computing expert outputs")
        # Step 3: GMM computation
        intermediate_output = self._gmm_compute_exact(x, local_group_sizes, selected_experts)
        
        print(f"[DEBUG] _expert_parallel_forward: after GMM - intermediate_output.shape={intermediate_output.shape}")
        global_tracer.print(intermediate_output, f"moe_compute_output", f"moe_compute_layer_id_{self.layer_id}")
        
        print("Step 4: Collecting results from expert devices")
        # Step 4: Result collection
        if self.expert_parallel_size > 1:
            intermediate_output = self._result_collection(
                intermediate_output, local_sorted_indices, global_group_sizes, 
                expert_shard_id, local_expert_size, tokens.shape[0] * self.num_experts_per_tok
            )
        else:
            print(f"[DEBUG] _expert_parallel_forward: skipping collection in single device mode")
        
        print(f"[DEBUG] _expert_parallel_forward: after collection - intermediate_output.shape={intermediate_output.shape}")
        global_tracer.print(intermediate_output, f"moe_collection_output", f"moe_combine_layer_id_{self.layer_id}")
        
        print("Step 5: Restoring original token order")
        # Step 5: Unpermute - restore original order
        final_output = self._unpermute_exact(
            intermediate_output, sorted_selected_experts, weights, 
            tokens.shape[0], tokens.shape[1]
        )
        
        print(f"[DEBUG] _expert_parallel_forward: final_output.shape={final_output.shape}")
        global_tracer.print(final_output, f"moe_unpermute_output", f"moe_combine_layer_id_{self.layer_id}")
        
        print("MoE forward completed")
        return final_output
    
    def _permute_exact(self, inputs, top_k_indices, top_k_weights):
        inputs_shape = inputs.shape
        
        print(f"[DEBUG] _permute_exact: inputs.shape={inputs_shape}")
        print(f"[DEBUG] _permute_exact: top_k_indices.shape={top_k_indices.shape}")
        print(f"[DEBUG] _permute_exact: top_k_weights.shape={top_k_weights.shape}")
        
        # Fix reshape logic: input is already (seq_len, hidden_dim)
        if len(inputs_shape) == 2:
            # Input is already 2D: (seq_len, hidden_dim)
            inputs_2d = inputs
            bsz_times_seq_len = inputs_shape[0]
        else:
            # Input is 3D: (batch, seq_len, hidden_dim) -> (batch*seq_len, hidden_dim)
            bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
            inputs_2d = jnp.reshape(inputs, (bsz_times_seq_len, inputs_shape[-1]))
        
        print(f"[DEBUG] _permute_exact: inputs_2d.shape={inputs_2d.shape}, bsz_times_seq_len={bsz_times_seq_len}")
        
        flatten_selected_experts = jnp.ravel(top_k_indices)
        print(f"[DEBUG] _permute_exact: flatten_selected_experts.shape={flatten_selected_experts.shape}")
        
        sorted_selected_experts = jnp.argsort(flatten_selected_experts)
        print(f"[DEBUG] _permute_exact: sorted_selected_experts.shape={sorted_selected_experts.shape}")
        
        sorted_indices = sorted_selected_experts // self.num_experts_per_tok
        print(f"[DEBUG] _permute_exact: sorted_indices.shape={sorted_indices.shape}")
        
        # Sort inputs by expert
        sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0).astype(self.dtype)
        print(f"[DEBUG] _permute_exact: sorted_inputs.shape={sorted_inputs.shape}")
        
        # Compute global group_sizes (number of tokens per expert)
        group_sizes = jnp.bincount(flatten_selected_experts, length=self.num_experts)
        print(f"[DEBUG] _permute_exact: group_sizes.shape={group_sizes.shape}")
        
        # 🛠️ FIX: 使用JAX兼容的方式显示非零专家信息
        non_zero_mask = group_sizes > 0
        non_zero_count = jnp.sum(non_zero_mask)
        print(f"[DEBUG] _permute_exact: number of non-zero experts: {non_zero_count}")
        print(f"[DEBUG] _permute_exact: total tokens distributed: {jnp.sum(group_sizes)}")
        
        # Generate sorted_experts
        expert_indices = jnp.arange(self.num_experts)
        sorted_experts = jnp.repeat(expert_indices, repeats=group_sizes, total_repeat_length=flatten_selected_experts.shape[0])
        print(f"[DEBUG] _permute_exact: sorted_experts.shape={sorted_experts.shape}")
        
        print(f"[DEBUG] _permute_exact: returning - sorted_inputs.shape={sorted_inputs.shape}, group_sizes.shape={group_sizes.shape}")
        
        return sorted_inputs, sorted_selected_experts, top_k_weights, group_sizes, sorted_experts
    
    def _expert_dispatch(self, sorted_inputs, global_group_sizes, sorted_experts, expert_shard_id, local_expert_size):
        # Add detailed tracer for dispatch input
        global_tracer.print(sorted_inputs, f"dispatch_input_sorted", f"moe_dispatch_layer_id_{self.layer_id}")
        global_tracer.print(global_group_sizes, f"dispatch_global_group_sizes", f"moe_dispatch_layer_id_{self.layer_id}")
        
        # global_group_sizes: (num_experts,) -> reshaped: (num_expert_parallelism,)
        reshaped_group_sizes = jnp.sum(global_group_sizes.reshape(self.expert_parallel_size, local_expert_size), axis=1)
        
        global_tracer.print(reshaped_group_sizes, f"dispatch_reshaped_group_sizes", f"moe_dispatch_layer_id_{self.layer_id}")
        
        # Unified communication abstraction
        x, local_sorted_indices, local_group_sizes, selected_experts = self._unified_expert_communication(
            sorted_inputs, global_group_sizes, sorted_experts, expert_shard_id, 
            local_expert_size, reshaped_group_sizes, is_dispatch=True
        )
        
        # Add detailed tracer for dispatch output
        global_tracer.print(x, f"dispatch_communicated_x", f"moe_dispatch_layer_id_{self.layer_id}")
        global_tracer.print(local_group_sizes, f"dispatch_local_group_sizes", f"moe_dispatch_layer_id_{self.layer_id}")
        global_tracer.print(selected_experts, f"dispatch_selected_experts", f"moe_dispatch_layer_id_{self.layer_id}")
        
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
        
        stage_name = "dispatch" if is_dispatch else "collection"
        global_tracer.print(data, f"comm_{stage_name}_input", f"moe_{stage_name}_layer_id_{self.layer_id}")
        
        if can_use_ragged:
            # GPU/TPU: use ragged_all_to_all
            result = self._ragged_communication(
                data, global_group_sizes, sorted_experts, expert_shard_id, 
                local_expert_size, reshaped_group_sizes, is_dispatch
            )
        else:
            # CPU: use regular all_to_all
            result = self._regular_communication(
                data, global_group_sizes, sorted_experts, expert_shard_id, 
                local_expert_size, reshaped_group_sizes, is_dispatch
            )
        
        if is_dispatch:
            global_tracer.print(result[0], f"comm_dispatch_output", f"moe_dispatch_layer_id_{self.layer_id}")
        else:
            global_tracer.print(result, f"comm_collection_output", f"moe_combine_layer_id_{self.layer_id}")
        
        return result
    
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
        print(f"[DEBUG] _regular_communication: {'dispatch' if is_dispatch else 'collection'} mode")
        print(f"[DEBUG] _regular_communication: data.shape={data.shape}, expert_shard_id={expert_shard_id}")
        print(f"[DEBUG] _regular_communication: reshaped_group_sizes.shape={reshaped_group_sizes.shape}")
        
        total_data = data.shape[0]
        remainder = total_data % self.expert_parallel_size
        padding_needed = (self.expert_parallel_size - remainder) % self.expert_parallel_size
        target_size = total_data + padding_needed
        
        print(f"[DEBUG] _regular_communication: total_data={total_data}, remainder={remainder}, padding_needed={padding_needed}")
        
        if padding_needed > 0:
            padding_shape = (padding_needed, data.shape[1])
            padding_data = jnp.zeros(padding_shape, dtype=data.dtype)
            padded_data = jnp.concatenate([data, padding_data], axis=0)
            print(f"[DEBUG] _regular_communication: added padding, padded_data.shape={padded_data.shape}")
        else:
            padded_data = data
            print(f"[DEBUG] _regular_communication: no padding needed")
        
        stage_name = "dispatch" if is_dispatch else "collection"
        global_tracer.print(padded_data, f"regular_comm_{stage_name}_padded", f"moe_{stage_name}_layer_id_{self.layer_id}")
        
        tokens_per_device = target_size // self.expert_parallel_size
        reshaped_data = padded_data.reshape(
            self.expert_parallel_size, tokens_per_device, data.shape[1]
        )
        
        print(f"[DEBUG] _regular_communication: tokens_per_device={tokens_per_device}, reshaped_data.shape={reshaped_data.shape}")
        
        global_tracer.print(reshaped_data, f"regular_comm_{stage_name}_reshaped", f"moe_{stage_name}_layer_id_{self.layer_id}")
        
        communicated_data = jax.lax.all_to_all(
            reshaped_data,
            axis_name=self.expert_axis_name,
            split_axis=0,
            concat_axis=1
        )
        
        print(f"[DEBUG] _regular_communication: after all_to_all, communicated_data.shape={communicated_data.shape}")
        
        global_tracer.print(communicated_data, f"regular_comm_{stage_name}_all_to_all", f"moe_{stage_name}_layer_id_{self.layer_id}")
        
        flattened_data = communicated_data.reshape(-1, data.shape[1])
        
        print(f"[DEBUG] _regular_communication: flattened_data.shape={flattened_data.shape}")
        global_tracer.print(flattened_data, f"regular_comm_{stage_name}_flattened", f"moe_{stage_name}_layer_id_{self.layer_id}")
        
        if is_dispatch:
            # 🛠️ FIX: 更精确的本地数据提取逻辑
            all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
                global_group_sizes[None, :], expert_shard_id * local_expert_size, local_expert_size, axis=1
            )
            local_group_sizes = jnp.sum(all_shard_local_sizes, axis=0)  # tokens per local expert

            num_valid_tokens = jnp.sum(local_group_sizes)
            
            print(f"[DEBUG] _regular_communication: local_group_sizes.shape={local_group_sizes.shape}")
            print(f"[DEBUG] _regular_communication: num_valid_tokens={num_valid_tokens}")
            
            # 🛠️ FIX: 更安全的数据提取 - 考虑all_to_all的具体语义
            # all_to_all with split_axis=0, concat_axis=1 意味着：
            # - 每个设备的数据被分到其他设备
            # - 结果在第1维度拼接
            # 我们需要提取属于当前设备的那部分数据
            
            # 计算当前设备应该接收的数据范围
            my_start_idx = expert_shard_id * tokens_per_device
            my_end_idx = (expert_shard_id + 1) * tokens_per_device
            
            print(f"[DEBUG] _regular_communication: extracting data range [{my_start_idx}:{my_end_idx}] from flattened_data")
            
            # 从all_to_all结果中提取当前设备的数据
            if flattened_data.shape[0] >= my_end_idx:
                device_data = flattened_data[my_start_idx:my_end_idx]
            else:
                print(f"[WARNING] _regular_communication: flattened_data too small, using all available data")
                device_data = flattened_data
            
            print(f"[DEBUG] _regular_communication: device_data.shape={device_data.shape}")
            
            # 进一步截取到实际需要的token数量
            if device_data.shape[0] >= num_valid_tokens:
                valid_data = device_data[:num_valid_tokens]
            else:
                print(f"[WARNING] _regular_communication: device_data too small for num_valid_tokens, padding with zeros")
                padding_size = int(num_valid_tokens) - device_data.shape[0]
                padding = jnp.zeros((padding_size, device_data.shape[1]), dtype=device_data.dtype)
                valid_data = jnp.concatenate([device_data, padding], axis=0)
            
            expert_ids = jnp.arange(local_expert_size)
            sorted_experts_ids = jnp.repeat(expert_ids, local_group_sizes, total_repeat_length=num_valid_tokens)
            
            local_sorted_indices = jnp.arange(num_valid_tokens)
            
            print(f"[DEBUG] _regular_communication: final valid_data.shape={valid_data.shape}")
            print(f"[DEBUG] _regular_communication: sorted_experts_ids.shape={sorted_experts_ids.shape}")
            
            global_tracer.print(valid_data, f"regular_dispatch_valid_data", f"moe_dispatch_layer_id_{self.layer_id}")
            global_tracer.print(local_group_sizes, f"regular_dispatch_local_sizes", f"moe_dispatch_layer_id_{self.layer_id}")
            
            return valid_data, local_sorted_indices, local_group_sizes, sorted_experts_ids
        else:
            print(f"[DEBUG] _regular_communication: collection mode, returning flattened_data.shape={flattened_data.shape}")
            return flattened_data
    
    def _local_permute_exact(self, inputs, global_group_sizes, local_expert_size, shard_index, is_offset=False, global_sorted_experts=None):
        print(f"[DEBUG] _local_permute_exact: inputs.shape={inputs.shape}, local_expert_size={local_expert_size}, shard_index={shard_index}")
        
        all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
            global_group_sizes, shard_index * local_expert_size, local_expert_size, axis=1
        )
        local_sizes = all_shard_local_sizes.reshape(-1)
        
        print(f"[DEBUG] _local_permute_exact: all_shard_local_sizes.shape={all_shard_local_sizes.shape}")
        print(f"[DEBUG] _local_permute_exact: local_sizes.shape={local_sizes.shape}")
        
        local_group_size = jnp.sum(all_shard_local_sizes, axis=0)
        print(f"[DEBUG] _local_permute_exact: local_group_size.shape={local_group_size.shape}")
        
        if is_offset:
            divided_assignments = jnp.floor_divide(global_sorted_experts, local_expert_size)
            expert_indices = jnp.where(
                divided_assignments == shard_index, 
                jnp.mod(global_sorted_experts, local_expert_size), 
                local_expert_size
            )
            print(f"[DEBUG] _local_permute_exact: using is_offset=True branch")
        else:
            # 🛠️ FIX: 正确的expert索引分配逻辑 - 需要在host上执行
            print(f"[DEBUG] _local_permute_exact: generating expert_indices for {inputs.shape[0]} tokens")
            
            # 将local_sizes转移到host进行处理
            local_sizes_host = jax.device_get(local_sizes)
            expert_indices_list = []
            for i, size in enumerate(local_sizes_host):
                expert_indices_list.extend([i] * int(size))
                print(f"[DEBUG] _local_permute_exact: expert {i} gets {int(size)} tokens")
            
            if len(expert_indices_list) != inputs.shape[0]:
                print(f"[ERROR] _local_permute_exact: expert_indices length {len(expert_indices_list)} != inputs.shape[0] {inputs.shape[0]}")
                # 如果长度不匹配，截取或填充
                if len(expert_indices_list) > inputs.shape[0]:
                    expert_indices_list = expert_indices_list[:inputs.shape[0]]
                else:
                    # 如果不够，用最后一个expert ID填充
                    last_expert_id = len(local_sizes_host) - 1
                    expert_indices_list.extend([last_expert_id] * (inputs.shape[0] - len(expert_indices_list)))
            
            expert_indices = jnp.array(expert_indices_list)
            print(f"[DEBUG] _local_permute_exact: expert_indices.shape={expert_indices.shape}")
        
        # Sort by local expert ID
        sorted_indices = jnp.argsort(expert_indices)
        sorted_inputs = jnp.take(inputs, indices=sorted_indices, axis=0)
        sorted_experts_ids = expert_indices[sorted_indices]
        
        print(f"[DEBUG] _local_permute_exact: sorted_indices.shape={sorted_indices.shape}")
        print(f"[DEBUG] _local_permute_exact: sorted_experts_ids.shape={sorted_experts_ids.shape}")
        print(f"[DEBUG] _local_permute_exact: output shapes - sorted_inputs={sorted_inputs.shape}, local_group_size={local_group_size}")
        
        return sorted_inputs, sorted_indices, local_group_size, sorted_experts_ids
    
    def _gmm_compute_exact(self, x, local_group_sizes, selected_experts):
        # Add detailed input tracers
        global_tracer.print(x, f"gmm_input_x", f"moe_compute_layer_id_{self.layer_id}")
        global_tracer.print(local_group_sizes, f"gmm_local_group_sizes", f"moe_compute_layer_id_{self.layer_id}")
        global_tracer.print(selected_experts, f"gmm_selected_experts", f"moe_compute_layer_id_{self.layer_id}")
        
        print(f"[DEBUG] _gmm_compute_exact: x.shape={x.shape}, local_group_sizes.shape={local_group_sizes.shape}")
        print(f"[DEBUG] _gmm_compute_exact: selected_experts.shape={selected_experts.shape}")
        
        def gmm_layer(inputs, kernel, group_sizes, expert_assignments, layer_name):
            print(f"[DEBUG] gmm_layer {layer_name}: inputs.shape={inputs.shape}, kernel.shape={kernel.shape}")
            print(f"[DEBUG] gmm_layer {layer_name}: group_sizes.shape={group_sizes.shape}")
            
            global_tracer.print(inputs, f"gmm_{layer_name}_input", f"moe_compute_layer_id_{self.layer_id}")
            global_tracer.print(kernel, f"gmm_{layer_name}_kernel", f"moe_compute_layer_id_{self.layer_id}")
            global_tracer.print(group_sizes, f"gmm_{layer_name}_group_sizes", f"moe_compute_layer_id_{self.layer_id}")
            
            result = jax.lax.ragged_dot(
                lhs=inputs,
                rhs=kernel,
                group_sizes=group_sizes,
                preferred_element_type=self.dtype
            )
            
            print(f"[DEBUG] gmm_layer {layer_name}: result.shape={result.shape}")
            global_tracer.print(result, f"gmm_{layer_name}_output", f"moe_compute_layer_id_{self.layer_id}")
            
            return result
        
        w0_kernel = self.wi_0.value
        w1_kernel = self.wi_1.value
        wo_kernel = self.wo.value
        
        # Add weight tracers
        global_tracer.print(w0_kernel, f"gmm_w0_kernel", f"moe_compute_layer_id_{self.layer_id}")
        global_tracer.print(w1_kernel, f"gmm_w1_kernel", f"moe_compute_layer_id_{self.layer_id}")
        global_tracer.print(wo_kernel, f"gmm_wo_kernel", f"moe_compute_layer_id_{self.layer_id}")
        
        print(f"[DEBUG] _gmm_compute_exact: weight shapes - w0={w0_kernel.shape}, w1={w1_kernel.shape}, wo={wo_kernel.shape}")
        
        # 🛠️ FIX: 更安全的group_sizes处理逻辑
        expected_global_experts = w0_kernel.shape[0]
        local_expert_count = len(local_group_sizes)
        
        print(f"[DEBUG] _gmm_compute_exact: expected_global_experts={expected_global_experts}, local_expert_count={local_expert_count}")
        
        if local_expert_count != expected_global_experts:
            print(f"[DEBUG] _gmm_compute_exact: expanding group_sizes from {local_expert_count} to {expected_global_experts}")
            expert_shard_id = jax.lax.axis_index(self.expert_axis_name)
            experts_per_device = self.experts_per_device
            local_expert_start = expert_shard_id * experts_per_device
            local_expert_end = (expert_shard_id + 1) * experts_per_device
            
            print(f"[DEBUG] _gmm_compute_exact: expert_shard_id={expert_shard_id}, experts_per_device={experts_per_device}")
            print(f"[DEBUG] _gmm_compute_exact: local expert range [{local_expert_start}:{local_expert_end}]")
            
            # 检查权重是否真的是全局的还是本地的
            if expected_global_experts == self.num_experts:
                # 权重是全局的，创建mask
                expanded_group_sizes = jnp.zeros(expected_global_experts, dtype=local_group_sizes.dtype)
                expanded_group_sizes = expanded_group_sizes.at[local_expert_start:local_expert_end].set(local_group_sizes)
                final_group_sizes = expanded_group_sizes
                print(f"[DEBUG] _gmm_compute_exact: using global weight logic, final_group_sizes.shape={final_group_sizes.shape}")
            else:
                # 权重是本地的，直接使用local_group_sizes
                final_group_sizes = local_group_sizes
                print(f"[DEBUG] _gmm_compute_exact: using local weight logic, final_group_sizes.shape={final_group_sizes.shape}")
        else:
            final_group_sizes = local_group_sizes
            print(f"[DEBUG] _gmm_compute_exact: no expansion needed, final_group_sizes.shape={final_group_sizes.shape}")
        
        print(f"[DEBUG] _gmm_compute_exact: GMM computing with {jnp.sum(final_group_sizes)} tokens across {len(final_group_sizes)} experts")
        
        global_tracer.print(final_group_sizes, f"gmm_final_group_sizes", f"moe_compute_layer_id_{self.layer_id}")
        
        layer_w0 = gmm_layer(x, w0_kernel, final_group_sizes, selected_experts, "wi_0")
        
        layer_w1 = gmm_layer(x, w1_kernel, final_group_sizes, selected_experts, "wi_1")
        
        
        layer_act = jax.nn.silu(layer_w0)
        
        global_tracer.print(layer_act, f"gmm_silu_activation", f"moe_compute_layer_id_{self.layer_id}")
        
        intermediate_layer = jnp.multiply(layer_act, layer_w1)
        
        global_tracer.print(intermediate_layer, f"gmm_intermediate_layer", f"moe_compute_layer_id_{self.layer_id}")
        
        
        intermediate_output = gmm_layer(intermediate_layer, wo_kernel, final_group_sizes, selected_experts, "wo")
        
        
        global_tracer.print(intermediate_output, f"gmm_final_output", f"moe_compute_layer_id_{self.layer_id}")
        
        print(f"[DEBUG] _gmm_compute_exact: final intermediate_output.shape={intermediate_output.shape}")
        
        return intermediate_output
    
    def _result_collection(self, intermediate_output, local_sorted_indices, global_group_sizes, 
                                   expert_shard_id, local_expert_size, original_inputs_first_dim):        
        # Add collection input tracers
        print(f"[DEBUG] _result_collection: intermediate_output.shape={intermediate_output.shape}")
        print(f"[DEBUG] _result_collection: local_sorted_indices.shape={local_sorted_indices.shape}")
        print(f"[DEBUG] _result_collection: expert_shard_id={expert_shard_id}, local_expert_size={local_expert_size}")
        print(f"[DEBUG] _result_collection: original_inputs_first_dim={original_inputs_first_dim}")
        
        global_tracer.print(intermediate_output, f"collection_input", f"moe_combine_layer_id_{self.layer_id}")
        global_tracer.print(local_sorted_indices, f"collection_local_indices", f"moe_combine_layer_id_{self.layer_id}")
        
        if len(intermediate_output) == 0:
            print(f"[DEBUG] _result_collection: empty intermediate_output, creating zeros")
            empty_result = jnp.zeros((original_inputs_first_dim, intermediate_output.shape[-1]), dtype=self.dtype)
            global_tracer.print(empty_result, f"collection_empty_result", f"moe_combine_layer_id_{self.layer_id}")
            return empty_result
        
        # 🛠️ FIX: 更安全的本地排序恢复
        if len(local_sorted_indices) > 0 and len(intermediate_output) > 0:
            # 确保索引范围有效
            max_idx = len(intermediate_output) - 1
            valid_indices = jnp.clip(jnp.argsort(local_sorted_indices), 0, max_idx)
            local_output = jnp.take(intermediate_output, indices=valid_indices, axis=0)
            print(f"[DEBUG] _result_collection: recovered local_output.shape={local_output.shape}")
        else:
            local_output = intermediate_output
            print(f"[DEBUG] _result_collection: no sorting needed, using intermediate_output directly")
        
        global_tracer.print(local_output, f"collection_local_output", f"moe_combine_layer_id_{self.layer_id}")
        
        reshaped_group_sizes = jnp.sum(global_group_sizes.reshape(self.expert_parallel_size, local_expert_size), axis=1)
        
        print(f"[DEBUG] _result_collection: reshaped_group_sizes={reshaped_group_sizes}")
        global_tracer.print(reshaped_group_sizes, f"collection_reshaped_sizes", f"moe_combine_layer_id_{self.layer_id}")
        
        result = self._unified_expert_communication(
            local_output, global_group_sizes, None, expert_shard_id,
            local_expert_size, reshaped_group_sizes, is_dispatch=False
        )
        
        print(f"[DEBUG] _result_collection: after communication, result.shape={result.shape}")
        global_tracer.print(result, f"collection_after_comm", f"moe_combine_layer_id_{self.layer_id}")
        
        # 🛠️ FIX: 更安全的尺寸对齐逻辑
        expected_size = original_inputs_first_dim
        actual_size = result.shape[0]
        
        print(f"[DEBUG] _result_collection: expected_size={expected_size}, actual_size={actual_size}")
        
        if actual_size > expected_size:
            print(f"[DEBUG] _result_collection: trimming from {actual_size} to {expected_size}")
            result = result[:expected_size]
            global_tracer.print(result, f"collection_trimmed", f"moe_combine_layer_id_{self.layer_id}")
        elif actual_size < expected_size:
            print(f"[DEBUG] _result_collection: padding from {actual_size} to {expected_size}")
            padding_size = expected_size - actual_size
            padding = jnp.zeros((padding_size, result.shape[1]), dtype=result.dtype)
            result = jnp.concatenate([result, padding], axis=0)
            global_tracer.print(result, f"collection_padded", f"moe_combine_layer_id_{self.layer_id}")
        else:
            print(f"[DEBUG] _result_collection: size matches, no adjustment needed")
        
        print(f"[DEBUG] _result_collection: final result.shape={result.shape}")
        global_tracer.print(result, f"collection_final_result", f"moe_combine_layer_id_{self.layer_id}")
        
        return result

    def _unpermute_exact(self, intermediate, sorted_selected_experts, weights, batch_size, sequence_length):        
        # Add unpermute input tracers
        global_tracer.print(intermediate, f"unpermute_input", f"moe_combine_layer_id_{self.layer_id}")
        global_tracer.print(sorted_selected_experts, f"unpermute_sorted_experts", f"moe_combine_layer_id_{self.layer_id}")
        global_tracer.print(weights, f"unpermute_weights", f"moe_combine_layer_id_{self.layer_id}")
        
        unsort_intermediate = jnp.take(intermediate, indices=jnp.argsort(sorted_selected_experts), axis=0)
        
        global_tracer.print(unsort_intermediate, f"unpermute_unsorted", f"moe_combine_layer_id_{self.layer_id}")
        
        reshaped_weights = jnp.reshape(weights, (-1, self.num_experts_per_tok))
        
        global_tracer.print(reshaped_weights, f"unpermute_reshaped_weights", f"moe_combine_layer_id_{self.layer_id}")
        
        reshaped_intermediate = jnp.reshape(
            unsort_intermediate,
            (reshaped_weights.shape[0], self.num_experts_per_tok, -1),
        )
        
        global_tracer.print(reshaped_intermediate, f"unpermute_reshaped_intermediate", f"moe_combine_layer_id_{self.layer_id}")
        
        # Add einsum input tracers
        intermediate_f32 = reshaped_intermediate.astype(jnp.float32)
        weights_f32 = reshaped_weights.astype(jnp.float32)
        global_tracer.print(intermediate_f32, f"unpermute_intermediate_f32", f"moe_combine_layer_id_{self.layer_id}")
        global_tracer.print(weights_f32, f"unpermute_weights_f32", f"moe_combine_layer_id_{self.layer_id}")
        
        output = jnp.einsum(
            "BKE,BK -> BE",
            intermediate_f32,
            weights_f32,
            precision=jax.lax.Precision.DEFAULT,
        )
        
        global_tracer.print(output, f"unpermute_einsum_output", f"moe_combine_layer_id_{self.layer_id}")
        
        final_output = output.astype(self.dtype)
        
        
        global_tracer.print(final_output, f"unpermute_final_output", f"moe_combine_layer_id_{self.layer_id}")
        
        return final_output

    def _local_forward(self, tokens, top_k_indices, top_k_weights):
        num_tokens, hidden_dim = tokens.shape
        
        global_tracer.print(tokens, f"moe_local_input", f"moe_compute_layer_id_{self.layer_id}")
        
        expert_weights = jnp.zeros((num_tokens, self.num_experts), dtype=self.dtype)
        
        token_indices = jnp.arange(num_tokens)[:, None]  # (num_tokens, 1)
        expert_weights = expert_weights.at[token_indices, top_k_indices].set(top_k_weights)
        
        global_tracer.print(expert_weights, f"expert_weights_matrix", f"moe_compute_layer_id_{self.layer_id}")
        
        all_wi_0 = self.wi_0.value  # (experts_per_device, hidden_dim, intermediate_dim)
        all_wi_1 = self.wi_1.value  # (experts_per_device, hidden_dim, intermediate_dim)
        all_wo = self.wo.value      # (experts_per_device, intermediate_dim, hidden_dim)
        
        layer_w0 = jnp.einsum('th,ehd->ted', tokens, all_wi_0)  # (num_tokens, experts_per_device, intermediate_dim)
        layer_w1 = jnp.einsum('th,ehd->ted', tokens, all_wi_1)  # (num_tokens, experts_per_device, intermediate_dim)
        
        global_tracer.print(layer_w0, f"layer_w0_output", f"moe_compute_layer_id_{self.layer_id}")
        global_tracer.print(layer_w1, f"layer_w1_output", f"moe_compute_layer_id_{self.layer_id}")
        
        activated = jax.nn.silu(layer_w0) * layer_w1  # (num_tokens, experts_per_device, intermediate_dim)
        
        global_tracer.print(activated, f"activated_intermediate", f"moe_compute_layer_id_{self.layer_id}")
        
        expert_outputs = jnp.einsum('ted,edh->teh', activated, all_wo)  # (num_tokens, experts_per_device, hidden_dim)
        
        global_tracer.print(expert_outputs, f"expert_outputs", f"moe_compute_layer_id_{self.layer_id}")
        
        final_output = jnp.einsum('te,teh->th', expert_weights, expert_outputs)  # (num_tokens, hidden_dim)
        
        global_tracer.print(final_output, f"moe_local_final_output", f"moe_compute_layer_id_{self.layer_id}")
        
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