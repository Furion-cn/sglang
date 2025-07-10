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
    """简化版MoE，完全模仿sparse_matmul逻辑"""
    
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
        
        # Mesh setup
        self.mesh = getattr(config, 'expert_mesh', None)
        if self.mesh is None:
            raise ValueError("Need expert_mesh in config")
        
        self.expert_parallelism = self.mesh.shape.get(expert_axis_name, 1)
        if num_experts % self.expert_parallelism != 0:
            raise ValueError(f"num_experts({num_experts}) must be divisible by expert_parallelism({self.expert_parallelism})")
        
        self.experts_per_device = num_experts // self.expert_parallelism
        
        # Expert权重：每个设备只存储负责的专家
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
        
        # 应用分片约束
        state = nnx.state(self)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(self, sharded_state)

    def _detect_device_capabilities(self):
        """检测设备类型和ragged_all_to_all支持情况"""
        try:
            devices = jax.devices()
            is_cpu_only = all(device.platform == 'cpu' for device in devices)
            can_use_ragged = not is_cpu_only and hasattr(jax.lax, 'ragged_all_to_all')
            
            device_types = [device.platform for device in devices]
            primary_device = device_types[0] if device_types else 'unknown'
            
            global_tracer.print(
                jnp.array([is_cpu_only, can_use_ragged]), 
                f"device_capabilities_cpu_ragged", 
                f"moe_device_layer_id_{self.layer_id}"
            )
            
            return can_use_ragged, primary_device
        except Exception as e:
            # 回退到CPU模式
            return False, 'cpu'

    @trace_function(stage="MOE_SPARSE_FORWARD", include_args=False, include_output=True)
    def __call__(self, inputs, router_logits=None):
        if router_logits is None:
            raise ValueError("router_logits is required for Qwen3MoE")
            
        inputs = inputs.astype(self.dtype)
        total_tokens, hidden_dim = inputs.shape
        
        global_tracer.print(inputs, f"moe_input", f"moe_sparse_layer_id_{self.layer_id}")
        global_tracer.print(router_logits, f"router_logits", f"moe_sparse_layer_id_{self.layer_id}")
        
        # ✅ 添加输入统计检查
        jax.debug.print("🔍 JAX MoE Input Stats: min={min}, max={max}, mean={mean}, std={std}", 
                        min=inputs.min(), max=inputs.max(), mean=inputs.mean(), std=inputs.std())
        
        # ✅ 添加权重统计检查
        w0_stats = f"w0: min={self.wi_0.value.min()}, max={self.wi_0.value.max()}, mean={self.wi_0.value.mean()}"
        w1_stats = f"w1: min={self.wi_1.value.min()}, max={self.wi_1.value.max()}, mean={self.wi_1.value.mean()}"  
        wo_stats = f"wo: min={self.wo.value.min()}, max={self.wo.value.max()}, mean={self.wo.value.mean()}"
        jax.debug.print("🔍 JAX MoE Weight Stats: {w0} | {w1} | {wo}", w0=w0_stats, w1=w1_stats, wo=wo_stats)
        
        if router_logits.shape[0] != total_tokens:
            raise ValueError(f"router_logits shape {router_logits.shape} doesn't match inputs shape {inputs.shape}")
        
        if self.expert_parallelism == 1:
            # 单设备模式：直接计算，无需shard_map
            output = self._single_device_forward(inputs, router_logits)
        else:
            # ✅ 多设备模式：在MoE内部使用shard_map，权重作为参数传入
            output = self._expert_parallel_forward_with_shard_map(inputs, router_logits)
        
        # ✅ 添加输出统计检查
        jax.debug.print("🔍 JAX MoE Output Stats: min={min}, max={max}, mean={mean}, std={std}", 
                        min=output.min(), max=output.max(), mean=output.mean(), std=output.std())
        
        jax.debug.print("layer_id={layer_id}, jax_moe_final_output={output}, min={min}, max={max}, mean={mean}, std={std}", layer_id=self.layer_id, output=output, min=output.min(), max=output.max(), mean=output.mean(), std=output.std())
        global_tracer.print(output, f"moe_final_output", f"moe_sparse_layer_id_{self.layer_id}")
        return output
    
    def _expert_parallel_forward_with_shard_map(self, inputs, router_logits):        
        def _internal_moe_computation(hidden_states, router_logits, w0_weights, w1_weights, wo_weights):
            """
            ✅ 内部计算函数：权重作为参数传入，已经是分片的
            在shard_map内部，w0_weights.shape = (16, 2048, 4096)
            """
            # ✅ 添加设备特定的日志验证
            expert_shard_id = jax.lax.axis_index(self.expert_axis_name)
            
            # 获取top-k专家
            top_k_logits, top_k_indices = jax.lax.top_k(router_logits, self.num_experts_per_tok)
            top_k_weights = jax.nn.softmax(top_k_logits.astype(jnp.bfloat16), axis=-1).astype(self.dtype)
            
            # ✅ 添加权重归一化，匹配PyTorch版本的renormalize=True行为
            top_k_weights = top_k_weights / jnp.sum(top_k_weights, axis=-1, keepdims=True)
            
            # ✅ 修复：正确处理输入维度
            if hidden_states.ndim == 2:
                # 2D输入：(total_tokens, hidden_dim)
                total_tokens = hidden_states.shape[0]
                batch_size, seq_len = 1, total_tokens  # 假设batch_size=1
            else:
                # 3D输入：(batch_size, seq_len, hidden_dim)
                batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
                total_tokens = batch_size * seq_len
            
            x, sorted_selected_experts, weights, group_sizes, selected_experts = self._permute(
                hidden_states, top_k_indices, top_k_weights
            )
            
            # ✅ Step 2: Expert Parallelism Dispatch
            expert_shard_id = jax.lax.axis_index(self.expert_axis_name)
            if self.expert_parallelism > 1:
                x, local_group_sizes, selected_experts = self._expert_all_to_all_dispatch(
                    x, group_sizes, selected_experts, expert_shard_id
                )
            else:
                local_group_sizes = group_sizes
                        
            # ✅ Step 3: GMM计算 - 现在权重已经是分片的！
            intermediate_output = self._gmm_compute_with_sharded_weights(
                x, local_group_sizes, selected_experts, w0_weights, w1_weights, wo_weights
            )
            
            # ✅ Step 4: Expert Parallelism Collection
            if self.expert_parallelism > 1:
                # ✅ 修复：正确计算original_size
                original_size = total_tokens * self.num_experts_per_tok
                intermediate_output = self._expert_all_to_all_collect(
                    intermediate_output, group_sizes, expert_shard_id, original_size
                )
            
            # ✅ Step 5: Unpermute - 恢复原始顺序
            output = self._unpermute(
                intermediate_output, sorted_selected_experts, weights, batch_size, seq_len
            )

            jax.debug.print("layer_id={layer_id}, jax_moe_final_output={output}, min={min}, max={max}, mean={mean}, std={std}", layer_id=self.layer_id, output=output, min=output.min(), max=output.max(), mean=output.mean(), std=output.std())
            
            return output
        
        # ✅ 使用shard_map，权重作为参数传入
        return shard_map(
            _internal_moe_computation,
            mesh=self.mesh,
            in_specs=(
                P(None),                     # hidden_states  
                P(None),                     # router_logits
                P(self.expert_axis_name, None, None),  # w0_weights - 在expert维度分片
                P(self.expert_axis_name, None, None),  # w1_weights  
                P(self.expert_axis_name, None, None),  # wo_weights
            ),
            out_specs=P(None),
            check_rep=False,
        )(inputs, router_logits, self.wi_0.value, self.wi_1.value, self.wo.value)
    
    def _gmm_compute_with_sharded_weights(self, x, local_group_sizes, selected_experts, w0_kernel, w1_kernel, wo_kernel):
        """✅ 新版GMM计算：权重已经通过shard_map正确分片，处理空输入情况"""
        global_tracer.print(x, f"gmm_sharded_input_x", f"moe_compute_layer_id_{self.layer_id}")
        global_tracer.print(w0_kernel, f"gmm_sharded_w0_kernel_shape", f"moe_compute_layer_id_{self.layer_id}")
        global_tracer.print(local_group_sizes, f"gmm_sharded_local_group_sizes", f"moe_compute_layer_id_{self.layer_id}")
        
        # ✅ 处理空输入：当前设备没有分配到任何token
        if x.shape[0] == 0:
            # 返回空的输出，shape需要匹配wo的输出维度
            # wo_kernel.shape = (16, 768, 2048)，输出维度应该是2048
            empty_output = jnp.zeros((0, wo_kernel.shape[-1]), dtype=x.dtype)  # (0, hidden_dim)
            global_tracer.print(empty_output, f"gmm_sharded_empty_output", f"moe_compute_layer_id_{self.layer_id}")
            return empty_output
        
        # ✅ 正常情况：进行ragged_dot计算
        # gate
        layer_w0 = jax.lax.ragged_dot(
            lhs=x,
            rhs=w0_kernel,
            group_sizes=local_group_sizes,
            preferred_element_type=self.dtype
        )
        # up
        layer_w1 = jax.lax.ragged_dot(
            lhs=x,
            rhs=w1_kernel,
            group_sizes=local_group_sizes,
            preferred_element_type=self.dtype
        )
        
        # 激活函数和合并
        # 
        layer_act = jax.nn.silu(layer_w0)
        intermediate_layer = jnp.multiply(layer_act, layer_w1)
        
        # 输出层
        intermediate_output = jax.lax.ragged_dot(
            lhs=intermediate_layer,
            rhs=wo_kernel,
            group_sizes=local_group_sizes,
            preferred_element_type=self.dtype
        )
        
        # ✅ 关键：GMM计算完成后的统计信息
        expert_shard_id = jax.lax.axis_index(self.expert_axis_name)
        jax.debug.print("🔍 [Layer {layer_id}] GMM output dev{dev_id}: min={min}, max={max}, mean={mean}, std={std}", 
                       layer_id=self.layer_id, dev_id=expert_shard_id, 
                       min=intermediate_output.min(), max=intermediate_output.max(), 
                       mean=intermediate_output.mean(), std=intermediate_output.std())
        
        global_tracer.print(intermediate_output, f"gmm_sharded_final_output", f"moe_compute_layer_id_{self.layer_id}")
        return intermediate_output
    
    def _single_device_forward(self, inputs, router_logits):
        """单设备模式：简化处理"""
        # 获取top-k专家
        top_k_logits, top_k_indices = jax.lax.top_k(router_logits, self.num_experts_per_tok)
        top_k_weights = jax.nn.softmax(top_k_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        
        # ✅ 添加权重归一化，匹配PyTorch版本的renormalize=True行为
        top_k_weights = top_k_weights / jnp.sum(top_k_weights, axis=-1, keepdims=True)
        
        return self._single_device_forward_impl(inputs, top_k_indices, top_k_weights)
    
    def _single_device_forward_impl(self, inputs, top_k_indices, top_k_weights):
        """单设备模式的具体实现"""
        global_tracer.print(inputs, f"moe_local_input", f"moe_compute_layer_id_{self.layer_id}")
        
        num_tokens = inputs.shape[0] * (inputs.shape[1] if inputs.ndim > 1 else 1)
        inputs_flat = inputs.reshape(num_tokens, -1)
        
        # 创建专家权重mask
        expert_weights = jnp.zeros((num_tokens, self.num_experts), dtype=self.dtype)
        token_indices = jnp.arange(num_tokens)[:, None]
        
        top_k_indices_flat = top_k_indices.reshape(num_tokens, -1)
        top_k_weights_flat = top_k_weights.reshape(num_tokens, -1)
        
        expert_weights = expert_weights.at[token_indices, top_k_indices_flat].set(top_k_weights_flat)
        
        global_tracer.print(expert_weights, f"expert_weights_matrix", f"moe_compute_layer_id_{self.layer_id}")
        
        # 直接计算
        all_wi_0 = self.wi_0.value
        all_wi_1 = self.wi_1.value
        all_wo = self.wo.value
        
        layer_w0 = jnp.einsum('th,ehd->ted', inputs_flat, all_wi_0)
        layer_w1 = jnp.einsum('th,ehd->ted', inputs_flat, all_wi_1)
        
        global_tracer.print(layer_w0, f"layer_w0_output", f"moe_compute_layer_id_{self.layer_id}")
        global_tracer.print(layer_w1, f"layer_w1_output", f"moe_compute_layer_id_{self.layer_id}")
        
        activated = jax.nn.silu(layer_w0) * layer_w1
        expert_outputs = jnp.einsum('ted,edh->teh', activated, all_wo)
        final_output = jnp.einsum('te,teh->th', expert_weights, expert_outputs)
        
        global_tracer.print(final_output, f"moe_local_final_output", f"moe_compute_layer_id_{self.layer_id}")
        return final_output.reshape(inputs.shape).astype(self.dtype)
    
    def _permute(self, inputs, top_k_indices, top_k_weights):
        inputs_shape = inputs.shape
        
        if len(inputs_shape) == 2:
            inputs_2d = inputs
            bsz_times_seq_len = inputs_shape[0]
        else:
            bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
            inputs_2d = jnp.reshape(inputs, (bsz_times_seq_len, inputs_shape[-1]))
        
        # ✅ 添加permute调试
        jax.debug.print("🔍 Permute Input Stats: min={min}, max={max}, mean={mean}", 
                        min=inputs_2d.min(), max=inputs_2d.max(), mean=inputs_2d.mean())
        jax.debug.print("🔍 Top-k indices shape: {shape}, weights shape: {w_shape}", 
                        shape=top_k_indices.shape, w_shape=top_k_weights.shape)
        
        flatten_selected_experts = jnp.ravel(top_k_indices)
        sorted_selected_experts = jnp.argsort(flatten_selected_experts)
        sorted_indices = sorted_selected_experts // self.num_experts_per_tok
        
        sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0).astype(self.dtype)
        
        # ✅ 检查sorted_inputs
        jax.debug.print("🔍 Sorted Inputs Stats: min={min}, max={max}, mean={mean}", 
                        min=sorted_inputs.min(), max=sorted_inputs.max(), mean=sorted_inputs.mean())
        
        group_sizes = jnp.bincount(flatten_selected_experts, length=self.num_experts)
        
        # ✅ 检查group_sizes
        jax.debug.print("🔍 Group sizes: {sizes}, total_tokens: {total}", 
                        sizes=group_sizes, total=jnp.sum(group_sizes))
        
        expert_indices = jnp.arange(self.num_experts)
        sorted_experts = jnp.repeat(
            expert_indices, repeats=group_sizes, total_repeat_length=flatten_selected_experts.shape[0]
        )
        
        return sorted_inputs, sorted_selected_experts, top_k_weights, group_sizes, sorted_experts
    
    def _expert_all_to_all_dispatch(self, data, global_group_sizes, sorted_experts, expert_shard_id):
        can_use_ragged, device_type = self._detect_device_capabilities()
        
        global_tracer.print(
            jnp.array([can_use_ragged]), 
            f"dispatch_device_ragged_support", 
            f"moe_dispatch_layer_id_{self.layer_id}"
        )
        
        if can_use_ragged:
            return self._ragged_all_to_all_dispatch(data, global_group_sizes, sorted_experts, expert_shard_id)
        else:
            # ✅ CPU模式：直接使用local_permute逻辑，简化处理
            return self._cpu_simple_dispatch(data, global_group_sizes, sorted_experts, expert_shard_id)
    
    def _cpu_simple_dispatch(self, data, global_group_sizes, sorted_experts, expert_shard_id):
        """
        ✅ JIT安全的dispatch：使用固定size方法
        """
        local_expert_size = self.experts_per_device
        
        jax.debug.print("dispatch_debug: expert_shard_id={expert_shard_id} local_expert_size={local_expert_size}", 
                       expert_shard_id=expert_shard_id, local_expert_size=local_expert_size)
        jax.debug.print("dispatch_debug: data.shape={data_shape} sorted_experts={sorted_experts}", 
                       data_shape=data.shape, sorted_experts=sorted_experts)
        
        # ✅ MaxText is_offset=True 逻辑：计算每个token属于哪个expert shard
        divided_assignments = jnp.floor_divide(sorted_experts, local_expert_size)
        
        # ✅ 创建mask：只有属于当前shard的tokens才保留
        belongs_to_this_shard = (divided_assignments == expert_shard_id)
        
        # ✅ 计算局部expert indices（保持原始shape）
        local_experts = jnp.where(
            belongs_to_this_shard,
            jnp.mod(sorted_experts, local_expert_size),
            local_expert_size  # 无效tokens标记为local_expert_size（超出范围）
        )
        
        # ✅ JIT安全：使用fixed size的nonzero
        valid_indices = jnp.nonzero(belongs_to_this_shard, size=data.shape[0])[0]
        num_valid_tokens = jnp.sum(belongs_to_this_shard)
        
        # ✅ 提取有效数据（使用fixed-size索引）
        local_data = data[valid_indices]
        local_experts_extracted = local_experts[valid_indices]
        
        # ✅ 计算local_group_sizes：只统计有效范围内的experts
        # 创建一个mask来只统计valid tokens
        valid_expert_mask = jnp.arange(data.shape[0]) < num_valid_tokens
        valid_experts_for_bincount = jnp.where(
            valid_expert_mask,
            local_experts_extracted,
            local_expert_size  # 无效位置设为超出范围的值
        )
        local_group_sizes = jnp.bincount(valid_experts_for_bincount, length=local_expert_size)
        
        jax.debug.print("dispatch_debug: device_{dev_id} has {num_tokens} valid tokens", 
                       dev_id=expert_shard_id, num_tokens=num_valid_tokens)
        jax.debug.print("dispatch_debug: local_group_sizes={sizes}", sizes=local_group_sizes)
        
        global_tracer.print(local_data, f"cpu_dispatch_output", f"moe_dispatch_layer_id_{self.layer_id}")
        global_tracer.print(local_group_sizes, f"cpu_dispatch_group_sizes", f"moe_dispatch_layer_id_{self.layer_id}")
        
        return local_data, local_group_sizes, local_experts_extracted
    
    def _ragged_all_to_all_dispatch(self, data, global_group_sizes, sorted_experts, expert_shard_id):
        local_expert_size = self.experts_per_device
        reshaped_group_sizes = jnp.sum(
            global_group_sizes.reshape(self.expert_parallelism, local_expert_size), axis=1
        )
        
        # 计算ragged_all_to_all的参数
        input_offsets, send_sizes, output_offsets, recv_sizes = self._get_ragged_all_to_all_params(
            reshaped_group_sizes, expert_shard_id
        )
        
        buffer_size = int(self.expert_parallelism * data.shape[0])
        output_shape = jnp.zeros((buffer_size, data.shape[1]), dtype=data.dtype)
        
        # 执行ragged_all_to_all
        communicated_data = jax.lax.ragged_all_to_all(
            data, output_shape, input_offsets, send_sizes,
            output_offsets, recv_sizes, axis_name=self.expert_axis_name,
        )
        
        x, local_group_sizes, selected_experts = self._local_permute_for_ragged(
            communicated_data, global_group_sizes, local_expert_size, expert_shard_id
        )
        
        global_tracer.print(x, f"ragged_dispatch_output", f"moe_dispatch_layer_id_{self.layer_id}")
        return x, local_group_sizes, selected_experts
    
    def _expert_all_to_all_collect(self, data, global_group_sizes, expert_shard_id, target_size):
        can_use_ragged, device_type = self._detect_device_capabilities()
        
        global_tracer.print(
            jnp.array([can_use_ragged]), 
            f"collect_device_ragged_support", 
            f"moe_collect_layer_id_{self.layer_id}"
        )
        
        if can_use_ragged:
            return self._ragged_all_to_all_collect(data, global_group_sizes, expert_shard_id, target_size)
        else:
            return self._cpu_simple_collect(data, global_group_sizes, expert_shard_id, target_size)
    
    def _cpu_simple_collect(self, data, global_group_sizes, expert_shard_id, target_size):  
        # ✅ All-reduce前的统计
        jax.debug.print("🔍 [Layer {layer_id}] Before all-reduce dev{dev_id}: min={min:.6f}, max={max:.6f}, mean={mean:.8f}, std={std:.6f}", 
                       layer_id=self.layer_id, dev_id=expert_shard_id, 
                       min=data.min(), max=data.max(), mean=data.mean(), std=data.std())
        
        # ✅ 关键修复：使用psum进行all-reduce求和，而不是all_gather
        # 这匹配PyTorch版本的tensor_model_parallel_all_reduce行为
        summed_data = jax.lax.psum(data, axis_name=self.expert_axis_name)
        
        # ✅ All-reduce后的统计
        jax.debug.print("🔍 [Layer {layer_id}] After all-reduce dev{dev_id}: min={min:.6f}, max={max:.6f}, mean={mean:.8f}, std={std:.6f}", 
                       layer_id=self.layer_id, dev_id=expert_shard_id, 
                       min=summed_data.min(), max=summed_data.max(), mean=summed_data.mean(), std=summed_data.std())
        
        global_tracer.print(summed_data, f"cpu_collect_psum_data", f"moe_combine_layer_id_{self.layer_id}")
        
        result = summed_data
        
        global_tracer.print(result, f"cpu_collect_final_simple", f"moe_combine_layer_id_{self.layer_id}")
        
        actual_size = result.shape[0] 
        if actual_size != target_size:
            raise ValueError(
                f"Collection output size mismatch: actual={actual_size}, expected={target_size}. "
                f"This suggests an issue with the dispatch/collect logic."
            )
        
        global_tracer.print(
            jnp.array([result.shape[0], target_size]), 
            f"cpu_collect_size_verified", 
            f"moe_combine_layer_id_{self.layer_id}"
        )
        
        return result
    
    def _ragged_all_to_all_collect(self, data, global_group_sizes, expert_shard_id, target_size):
        """TPU/GPU: 使用ragged_all_to_all进行collection"""
        local_expert_size = self.experts_per_device
        reshaped_group_sizes = jnp.sum(
            global_group_sizes.reshape(self.expert_parallelism, local_expert_size), axis=1
        )
        
        # 计算ragged_all_to_all的参数（transpose版本用于collection）
        input_offsets, send_sizes, output_offsets, recv_sizes = self._get_ragged_all_to_all_params(
            reshaped_group_sizes.T, expert_shard_id  # 注意这里需要转置
        )
        
        # 创建输出buffer
        output_shape = jnp.zeros((target_size, data.shape[1]), dtype=data.dtype)
        
        # 执行ragged_all_to_all
        result = jax.lax.ragged_all_to_all(
            data, output_shape, input_offsets, send_sizes,
            output_offsets, recv_sizes, axis_name=self.expert_axis_name,
        )
        
        global_tracer.print(result, f"ragged_collect_output", f"moe_combine_layer_id_{self.layer_id}")
        return result
    
    def _get_ragged_all_to_all_params(self, group_sizes, shard_id):
        input_offsets = jnp.zeros(self.expert_parallelism, dtype=jnp.int32)
        send_sizes = jnp.repeat(group_sizes[shard_id], self.expert_parallelism)
        
        output_offset = jnp.concatenate((jnp.array([0]), jnp.cumsum(group_sizes[:-1])))[shard_id]
        output_offsets = jnp.repeat(output_offset, self.expert_parallelism)
        
        recv_sizes = group_sizes
        
        return input_offsets, send_sizes, output_offsets, recv_sizes
    
    def _local_permute_for_ragged(self, inputs, global_group_sizes, local_expert_size, shard_index):
        local_group_sizes = global_group_sizes[
            shard_index * local_expert_size:(shard_index + 1) * local_expert_size
        ]
        
        expert_indices = jnp.repeat(
            jnp.arange(local_expert_size),
            local_group_sizes,
            total_repeat_length=jnp.sum(local_group_sizes)
        )
        
        sorted_indices = jnp.argsort(expert_indices)
        sorted_inputs = jnp.take(inputs, indices=sorted_indices, axis=0)
        sorted_experts_ids = expert_indices[sorted_indices]
        
        return sorted_inputs, local_group_sizes, sorted_experts_ids
    
    def _unpermute(self, intermediate, sorted_selected_experts, weights, batch_size, seq_len):
        # ✅ Unpermute输入统计
        jax.debug.print("🔍 [Layer {layer_id}] Unpermute input: min={min:.6f}, max={max:.6f}, mean={mean:.8f}, std={std:.6f}", 
                        layer_id=self.layer_id, min=intermediate.min(), max=intermediate.max(), 
                        mean=intermediate.mean(), std=intermediate.std())
        
        global_tracer.print(intermediate, f"unpermute_input", f"moe_combine_layer_id_{self.layer_id}")
        global_tracer.print(sorted_selected_experts, f"unpermute_sorted_experts", f"moe_combine_layer_id_{self.layer_id}")
        global_tracer.print(weights, f"unpermute_weights", f"moe_combine_layer_id_{self.layer_id}")
        
        expected_tokens = sorted_selected_experts.shape[0]
        actual_tokens = intermediate.shape[0]
        
        global_tracer.print(
            jnp.array([actual_tokens, expected_tokens]), 
            f"unpermute_token_count_check", 
            f"moe_combine_layer_id_{self.layer_id}"
        )
        
        if actual_tokens != expected_tokens:
            if actual_tokens > expected_tokens:
                intermediate = intermediate[:expected_tokens]
                global_tracer.print(
                    jnp.array([1, actual_tokens, expected_tokens]), 
                    f"unpermute_truncated", 
                    f"moe_combine_layer_id_{self.layer_id}"
                )
            else:
                padding_size = expected_tokens - actual_tokens
                padding = jnp.zeros((padding_size, intermediate.shape[1]), dtype=intermediate.dtype)
                intermediate = jnp.concatenate([intermediate, padding], axis=0)
                global_tracer.print(
                    jnp.array([2, actual_tokens, expected_tokens, padding_size]), 
                    f"unpermute_padded", 
                    f"moe_combine_layer_id_{self.layer_id}"
                )
        
        argsort_indices = jnp.argsort(sorted_selected_experts)
        unsort_intermediate = jnp.take(intermediate, indices=argsort_indices, axis=0)
        
        total_tokens = weights.shape[0] * weights.shape[1] // self.num_experts_per_tok
        
        reshaped_weights = jnp.reshape(weights, (total_tokens, self.num_experts_per_tok))
        reshaped_intermediate = jnp.reshape(
            unsort_intermediate,
            (total_tokens, self.num_experts_per_tok, -1),
        )
        
        global_tracer.print(reshaped_weights, f"unpermute_reshaped_weights", f"moe_combine_layer_id_{self.layer_id}")
        global_tracer.print(reshaped_intermediate, f"unpermute_reshaped_intermediate", f"moe_combine_layer_id_{self.layer_id}")
        
        # ✅ Einsum前的统计
        intermediate_fp32 = reshaped_intermediate.astype(jnp.float32)
        weights_fp32 = reshaped_weights.astype(jnp.float32)
        
        jax.debug.print("🔍 [Layer {layer_id}] Before einsum: intermediate min={i_min:.6f}, max={i_max:.6f}, weights min={w_min:.6f}, max={w_max:.6f}", 
                        layer_id=self.layer_id, i_min=intermediate_fp32.min(), i_max=intermediate_fp32.max(), 
                        w_min=weights_fp32.min(), w_max=weights_fp32.max())
        
        output = jnp.einsum(
            "BKE,BK -> BE",
            intermediate_fp32,
            weights_fp32,
        )
        
        # ✅ Einsum后的统计
        jax.debug.print("🔍 [Layer {layer_id}] After einsum: min={min:.6f}, max={max:.6f}, mean={mean:.8f}, std={std:.6f}", 
                        layer_id=self.layer_id, min=output.min(), max=output.max(), 
                        mean=output.mean(), std=output.std())
        
        global_tracer.print(output, f"unpermute_einsum_output", f"moe_combine_layer_id_{self.layer_id}")
        
        if len(weights.shape) == 2:
            final_output = output.astype(self.dtype)
        else:
            final_output = output.reshape(batch_size, seq_len, -1).astype(self.dtype)
        
        # ✅ 最终输出统计
        jax.debug.print("🔍 [Layer {layer_id}] Final output: min={min:.6f}, max={max:.6f}, mean={mean:.8f}, std={std:.6f}", 
                        layer_id=self.layer_id, min=final_output.min(), max=final_output.max(), 
                        mean=final_output.mean(), std=final_output.std())
        
        global_tracer.print(final_output, f"unpermute_final_output", f"moe_combine_layer_id_{self.layer_id}")
        return final_output