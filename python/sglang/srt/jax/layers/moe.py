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
        
        pre_bias_logits = None
        
        if self.score_func:
            if self.score_func == "softmax":
                output = jax.nn.softmax(output)
            elif self.score_func == "sigmoid": 
                output = jax.nn.sigmoid(output)
            elif self.score_func == "tanh":
                output = jax.nn.tanh(output)
            
            if self.model_name.startswith("deepseek3"):
                pre_bias_logits = output
        
        if self.use_bias and self.bias is not None:
            bias = jnp.asarray(self.bias.value, self.dtype)
            output += bias
            
        return output, pre_bias_logits

def random_routing(rng_key, gate_logits, num_experts_per_tok):
  """
  Performs random routing of tokens to experts.

  Args:
    rng_key: A JAX PRNGKey for randomness.
    gate_logits: A JAX array of shape (batch_size, sequence_length, num_experts)
                 representing the logits for each expert.
    num_experts_per_tok: The number of experts to select for each token.

  Returns:
    A tuple containing:
      - top_k_indices: JAX array of shape (batch_size, sequence_length, num_experts_per_tok)
                       representing the indices of the selected experts for each token.
      - top_k_weights: JAX array of shape (batch_size, sequence_length, num_experts_per_tok)
                       representing the weights for the selected experts.
  """
  bs, seq_len, num_experts = gate_logits.shape
  indices = jnp.arange(num_experts).repeat(bs * seq_len)
  selected_num = bs * seq_len * num_experts_per_tok
  top_k_indices = jax.random.choice(rng_key, indices, shape=(selected_num,)).reshape(bs, seq_len, num_experts_per_tok)
  top_k_weights = jnp.take_along_axis(gate_logits, top_k_indices, axis=-1)
  return top_k_weights, top_k_indices

DISPATCH = "dispatch"
COMBINE = "combine"

class RoutedMoE(nnx.Module):
    """Implements a routed MoE block.

    Attributes:
        config: Configuration object.
        num_experts: Number of experts.
        num_experts_per_tok: Number of experts for each token.
        mesh: Mesh, device mesh.
        kernel_init: Kernel function, passed to the dense layers.
        kernel_axes: Tuple with axes to apply kernel function.
        intermediate_dim: Intermediate dimension of MoE.
        weight_dtype: Type for the weights.
        dtype: Type for the dense layer.
        quant: Optional quantization config, no quantization if None.
    """

    def __init__(self,
                 config,
                 num_experts: int,
                 num_experts_per_tok: int,
                 mesh: Mesh,
                 kernel_init,
                 kernel_axes,
                 intermediate_dim: int = 2048,
                 weight_dtype: jnp.dtype = jnp.float32,
                 dtype: jnp.dtype = jnp.float32,
                 quant = None,
                 rngs: nnx.Rngs = None):
        
        self.config = config
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.mesh = mesh
        self.kernel_init = kernel_init
        self.kernel_axes = kernel_axes
        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.quant = quant
        
        self.wi_kernel_axes = ("exp", "embed_no_exp", "mlp")
        self.wo_kernel_axes = ("exp", "mlp", "embed_no_exp")
        
        self.gate = GateLogit(
            input_size=config.emb_dim,
            features=self.num_experts,
            model_name=config.model_name,
            weight_dtype=self.weight_dtype,
            dtype=self.dtype,
            kernel_axes=self.kernel_axes,
            use_bias=getattr(config, 'routed_bias', False),
            score_func=getattr(config, 'routed_score_func', ""),
            matmul_precision=getattr(config, 'matmul_precision', 'default'),
            rngs=rngs
        )
        
        self.wi_0 = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), self.wi_kernel_axes)(
                rngs.params(), (num_experts, config.emb_dim, intermediate_dim), weight_dtype
            )
        )
        
        self.wi_1 = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), self.wi_kernel_axes)(
                rngs.params(), (num_experts, config.emb_dim, intermediate_dim), weight_dtype
            )
        )
        
        self.wo = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), self.wo_kernel_axes)(
                rngs.params(), (num_experts, intermediate_dim, config.emb_dim), weight_dtype
            )
        )

    def get_expert_parallelism_size(self):
        return self.mesh.shape["expert"]

    def get_tensor_parallelism_size(self):
        return self.mesh.shape["tensor"]

    def get_context_autoregressive_parallelism_size(self):
        return self.mesh.shape["context_autoregressive"]

    def get_topk(self, gate_logits, pre_bias_logits):
        if getattr(self.config, 'use_random_routing', False):
            rng = nnx.Rngs(42).random()  # 简化版本
            top_k_weights, top_k_indices = random_routing(rng, gate_logits, self.num_experts_per_tok)
            return top_k_weights, top_k_indices

        if self.config.model_name.startswith("deepseek3"):
            top_k_weights, top_k_indices = self.deepseek_routing(gate_logits, pre_bias_logits)
        else:
            top_k_weights, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)

        decoder_block = getattr(self.config, 'decoder_block', None)
        if hasattr(decoder_block, 'name'):
            decoder_block = decoder_block.name
        elif isinstance(decoder_block, str):
            pass
        else:
            decoder_block = str(decoder_block) if decoder_block else None
            
        if decoder_block == "DEEPSEEK":
            top_k_weights = self.deepseek_scale_weights(top_k_weights)
        elif decoder_block != "LLAMA4":
            top_k_weights = jax.nn.softmax(top_k_weights.astype(jnp.float32), axis=-1).astype(self.dtype)
        return top_k_weights, top_k_indices

    def deepseek_scale_weights(self, weights):
        routed_score_func = getattr(self.config, 'routed_score_func', '')
        if routed_score_func == "sigmoid":
            weights /= weights.sum(-1, keepdims=True)
        routed_scaling_factor = getattr(self.config, 'routed_scaling_factor', 1.0)
        weights *= routed_scaling_factor
        return weights

    def deepseek_routing(self, gate_logits, pre_bias_logits):
        batch_size, seq_len = gate_logits.shape[0], gate_logits.shape[1]
        n = batch_size * seq_len
        gate_logits_flat = jnp.reshape(gate_logits, (n, self.num_experts))
        pre_bias_logits_flat = jnp.reshape(pre_bias_logits, (n, self.num_experts))

        n_routing_groups = getattr(self.config, 'n_routing_groups', -1)
        if n_routing_groups != -1:
            experts_per_group = self.num_experts // n_routing_groups
            scores_grouped = jnp.reshape(gate_logits_flat, (n, n_routing_groups, experts_per_group))

            top2_in_group_vals, _ = jax.lax.top_k(scores_grouped, k=2)
            group_scores = jnp.sum(top2_in_group_vals.astype(jnp.float32), axis=-1)
            topk_routing_group = getattr(self.config, 'topk_routing_group', 1)
            group_idx = jax.lax.top_k(group_scores, k=topk_routing_group)[1]

            group_mask = jax.nn.one_hot(group_idx, num_classes=n_routing_groups, dtype=jnp.float32)
            group_mask = jnp.sum(group_mask, axis=1)

            score_mask_grouped = jnp.expand_dims(group_mask, axis=-1)
            score_mask_expanded = jnp.broadcast_to(score_mask_grouped, (n, n_routing_groups, experts_per_group))
            score_mask = jnp.reshape(score_mask_expanded, (n, self.num_experts))
            negative_infinity = -jnp.inf
            masked_scores = jnp.where(score_mask > 0, gate_logits_flat, negative_infinity)
            top_k_indices = jax.lax.top_k(masked_scores, k=self.num_experts_per_tok)[1]
        else:
            top_k_indices = jax.lax.top_k(gate_logits_flat, k=self.num_experts_per_tok)[1]

        top_k_weights = jnp.take_along_axis(pre_bias_logits_flat, top_k_indices, axis=-1)
        top_k_indices = jnp.reshape(top_k_indices, (batch_size, seq_len, self.num_experts_per_tok))
        top_k_weights = jnp.reshape(top_k_weights, (batch_size, seq_len, self.num_experts_per_tok))
        return top_k_weights, top_k_indices

    def permute(self, inputs, gate_logits, pre_bias_logits):
        inputs_shape = inputs.shape
        bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
        inputs_2d = jnp.reshape(inputs, (bsz_times_seq_len, inputs_shape[2]))
        weights, selected_experts = self.get_topk(gate_logits, pre_bias_logits)

        decoder_block = getattr(self.config, 'decoder_block', None)
        if hasattr(decoder_block, 'name'):
            decoder_block = decoder_block.name
        elif isinstance(decoder_block, str):
            pass
        else:
            decoder_block = str(decoder_block) if decoder_block else None
            
        if decoder_block == "LLAMA4":
            router_scores = jax.nn.sigmoid(weights.astype(jnp.float32))
            inputs_2d = inputs_2d * router_scores.reshape(bsz_times_seq_len, -1)

        flatten_selected_experts = jnp.ravel(selected_experts)
        sorted_selected_experts = jnp.argsort(flatten_selected_experts)
        sorted_indices = sorted_selected_experts // self.num_experts_per_tok
        sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0).astype(self.dtype)
        group_size = jnp.bincount(flatten_selected_experts, length=self.num_experts)
        expert_indices = jnp.arange(self.num_experts)
        sorted_experts = jnp.repeat(expert_indices, repeats=group_size, total_repeat_length=flatten_selected_experts.shape[0])
        return sorted_inputs, sorted_selected_experts, weights, group_size, sorted_experts

    def unpermute(self, intermediate, sorted_selected_experts, weights, batch_size, sequence_length):
        unsort_intermediate = jnp.take(intermediate, indices=jnp.argsort(sorted_selected_experts), axis=0)
        reshaped_weights = jnp.reshape(weights, (-1, self.num_experts_per_tok))
        reshaped_intermediate = jnp.reshape(
            unsort_intermediate,
            (reshaped_weights.shape[0], self.num_experts_per_tok, -1),
        )
        with jax.named_scope("weight_sum"):
            matmul_precision = getattr(self.config, 'matmul_precision', 'default')
            precision = jax.lax.Precision(matmul_precision) if matmul_precision != 'default' else None
            
            decoder_block = getattr(self.config, 'decoder_block', None)
            if hasattr(decoder_block, 'name'):
                decoder_block = decoder_block.name
            elif isinstance(decoder_block, str):
                pass
            else:
                decoder_block = str(decoder_block) if decoder_block else None
                
            if decoder_block == "LLAMA4":
                reshaped_weights = jnp.ones_like(reshaped_weights)
                
            if precision:
                output = jnp.einsum(
                    "BKE,BK -> BE",
                    reshaped_intermediate.astype(jnp.float32),
                    reshaped_weights.astype(jnp.float32),
                    precision=precision,
                )
            else:
                output = jnp.einsum(
                    "BKE,BK -> BE",
                    reshaped_intermediate.astype(jnp.float32),
                    reshaped_weights.astype(jnp.float32),
                )
        return output.reshape(batch_size, sequence_length, -1).astype(self.dtype)

    @staticmethod
    def local_permute(inputs, global_group_sizes, local_expert_size, shard_index, is_offset=False, global_sorted_experts=None):
        all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
            global_group_sizes, shard_index * local_expert_size, local_expert_size, axis=1
        )
        local_sizes = all_shard_local_sizes.reshape(-1)
        local_group_size = jnp.sum(all_shard_local_sizes, axis=0)

        if is_offset:
            divided_assignments = jnp.floor_divide(global_sorted_experts, local_expert_size)
            expert_indices = jnp.where(
                divided_assignments == shard_index, jnp.mod(global_sorted_experts, local_expert_size), local_expert_size
            )
        else:
            base_indices = jnp.mod(jnp.arange(local_sizes.shape[0]), local_expert_size)
            expert_indices = jnp.repeat(base_indices, local_sizes, total_repeat_length=inputs.shape[0])

        sorted_indices = jnp.argsort(expert_indices)
        sorted_inputs = jnp.take(inputs, indices=sorted_indices, axis=0)
        sorted_experts_ids = expert_indices[sorted_indices]
        return (sorted_inputs, sorted_indices, local_group_size, sorted_experts_ids)

    @staticmethod
    def get_all_to_all_params(all_shards_group_sizes, shard_id, num_expert_parallelism, is_batch_sharded=True):
        from enum import Enum, auto
        
        class TransformStrategy(Enum):
            INPUT_OFFSET = auto()
            SEND_SIZE = auto()
            OUTPUT_OFFSET = auto()
            RECV_SIZE = auto()

        def transform_array(input_array, shard_id, strategy, is_batch_sharded):
            if is_batch_sharded:
                if strategy == TransformStrategy.INPUT_OFFSET:
                    local_array = input_array[shard_id]
                    return jnp.concatenate((jnp.array([0]), jnp.cumsum(local_array)[:-1]))
                elif strategy == TransformStrategy.SEND_SIZE:
                    return input_array[shard_id]
                elif strategy == TransformStrategy.OUTPUT_OFFSET:
                    zero_row = jnp.zeros((1,) + input_array.shape[1:], dtype=input_array.dtype)
                    array_with_zeros = jnp.concatenate((zero_row, input_array), axis=0)
                    cumulated_array = jnp.cumsum(array_with_zeros, axis=0, dtype=input_array.dtype)
                    return cumulated_array[shard_id]
                elif strategy == TransformStrategy.RECV_SIZE:
                    return input_array[:, shard_id]
                else:
                    raise ValueError(f"Unknown tranform array strategy: {strategy}")
            else:
                if strategy == TransformStrategy.INPUT_OFFSET:
                    return jnp.zeros(num_expert_parallelism, dtype=input_array.dtype)
                elif strategy == TransformStrategy.SEND_SIZE:
                    return jnp.repeat(input_array[shard_id], num_expert_parallelism)
                elif strategy == TransformStrategy.OUTPUT_OFFSET:
                    output_offset = jnp.concatenate((jnp.array([0]), jnp.cumsum(input_array[:-1])))[shard_id]
                    return jnp.repeat(output_offset, num_expert_parallelism)
                elif strategy == TransformStrategy.RECV_SIZE:
                    return input_array
                else:
                    raise ValueError(f"Unknown tranform array strategy: {strategy}")

        input_offsets = transform_array(all_shards_group_sizes, shard_id, TransformStrategy.INPUT_OFFSET, is_batch_sharded)
        send_sizes = transform_array(all_shards_group_sizes, shard_id, TransformStrategy.SEND_SIZE, is_batch_sharded)
        output_offsets = transform_array(all_shards_group_sizes, shard_id, TransformStrategy.OUTPUT_OFFSET, is_batch_sharded)
        recv_sizes = transform_array(all_shards_group_sizes, shard_id, TransformStrategy.RECV_SIZE, is_batch_sharded)
        return input_offsets, send_sizes, output_offsets, recv_sizes

    def sparse_matmul(self, inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel):        
        def gmm(inputs, kernel, group_sizes, expert_assignments):
            tile_batch_seq = getattr(self.config, 'tile_batch_seq', 512)
            PAD_LENGTH = tile_batch_seq
            hs_shape = inputs.shape
            
            if inputs.shape[0] != expert_assignments.shape[0]:
                raise ValueError("The number of input tokens must match the number of expert assignments!")
            
            pad_length = PAD_LENGTH
            if hs_shape[0] % PAD_LENGTH:
                pad_length = PAD_LENGTH - hs_shape[0] % PAD_LENGTH
                inputs = jax.lax.pad(inputs.astype(jnp.float32), 0.0, [(0, pad_length, 0), (0, 0, 0)])

            inputs = inputs.astype(self.dtype)
            kernel = kernel.astype(self.dtype)

            output = jax.lax.ragged_dot(
                lhs=inputs,
                rhs=kernel,
                group_sizes=group_sizes,
                preferred_element_type=jnp.bfloat16,
            )
                
            if hs_shape[0] % PAD_LENGTH:
                output = output[: hs_shape[0]]
            return output

        try:
            logical_axis_rules = getattr(self.config, 'logical_axis_rules', [])
            activation_batch_rules = [rule for rule in logical_axis_rules if rule[0] == "activation_batch"]
            is_batch_sharded_by_expert = (
                activation_batch_rules and "expert" in activation_batch_rules[0][1]
            )
        except:
            is_batch_sharded_by_expert = False
            
        if is_batch_sharded_by_expert and inputs.shape[0] > 1:
            batch_logical_axis = "activation_batch"
        else:
            batch_logical_axis = "activation_batch_no_exp"

        def wrapper(x, logits, pre_bias_logits, w0, w1, wo):
            batch_size, sequence_length, _ = x.shape
            x, sorted_selected_experts, weights, group_sizes, selected_experts = self.permute(x, logits, pre_bias_logits)
            expert_axis_name = "expert"
            expert_shard_id = jax.lax.axis_index(expert_axis_name) if self.get_expert_parallelism_size() > 1 else 0
            num_expert_parallelism = self.get_expert_parallelism_size()
            
            if num_expert_parallelism > 1:
                batch_axis = "expert" if is_batch_sharded_by_expert else "data"
                # get group sizes for all shards
                local_expert_size = getattr(self.config, 'num_experts', self.num_experts) // num_expert_parallelism
                reshaped_group_sizes = jnp.sum(group_sizes.reshape(-1, local_expert_size), axis=1)
                global_group_sizes = group_sizes
                
                if is_batch_sharded_by_expert:
                    all_shards_group_sizes = jax.lax.all_gather(reshaped_group_sizes, axis_name=batch_axis)
                    input_offsets, send_sizes, output_offsets, recv_sizes = RoutedMoE.get_all_to_all_params(
                        all_shards_group_sizes, expert_shard_id, num_expert_parallelism
                    )
                    # Calculate buffer size
                    per_device_batch_size = getattr(self.config, 'per_device_batch_size', batch_size)
                    max_target_length = getattr(self.config, 'max_target_length', sequence_length)
                    num_experts_per_tok = getattr(self.config, 'num_experts_per_tok', self.num_experts_per_tok)
                    emb_dim = getattr(self.config, 'emb_dim', x.shape[-1])
                    
                    buffer_size = int(
                        num_expert_parallelism
                        * per_device_batch_size
                        * max_target_length
                        * num_experts_per_tok
                    )
                    output_shape = jnp.zeros((buffer_size, emb_dim), dtype=x.dtype)

                    x = jax.lax.ragged_all_to_all(
                        x,
                        output_shape,
                        input_offsets,
                        send_sizes,
                        output_offsets,
                        recv_sizes,
                        axis_name=expert_axis_name,
                    )
                    global_group_sizes = jax.lax.all_gather(group_sizes, axis_name=expert_axis_name)
                    x, local_sorted_indices, group_sizes, selected_experts = RoutedMoE.local_permute(
                        x, global_group_sizes, local_expert_size, shard_index=expert_shard_id
                    )
                else:
                    x, local_sorted_indices, group_sizes, selected_experts = RoutedMoE.local_permute(
                        x,
                        global_group_sizes[None, :],
                        local_expert_size,
                        shard_index=expert_shard_id,
                        is_offset=True,
                        global_sorted_experts=selected_experts,
                    )

            layer_w0 = gmm(x, w0, group_sizes, selected_experts)
            layer_w1 = gmm(x, w1, group_sizes, selected_experts)
            
            mlp_activations = getattr(self.config, 'mlp_activations', ['silu'])
            if mlp_activations[0] == 'silu':
                layer_w0_act = jax.nn.silu(layer_w0)
            elif mlp_activations[0] == 'relu':
                layer_w0_act = jax.nn.relu(layer_w0)
            elif mlp_activations[0] == 'gelu':
                layer_w0_act = jax.nn.gelu(layer_w0)
            else:
                layer_w0_act = jax.nn.silu(layer_w0)
                
            intermediate_layer = jnp.multiply(layer_w0_act, layer_w1)
            intermediate_output = gmm(intermediate_layer, wo, group_sizes, selected_experts)

            if self.get_tensor_parallelism_size() > 1:
                intermediate_output = jax.lax.psum_scatter(intermediate_output, "tensor", scatter_dimension=1, tiled=True)

            if num_expert_parallelism > 1:
                original_inputs_first_dim = batch_size * sequence_length * self.num_experts_per_tok
                if sorted_selected_experts.shape[0] != original_inputs_first_dim:
                    raise ValueError("original_inputs_first_dim does not match the original tensor shape!")
                
                tensor_parallelism_size = self.get_tensor_parallelism_size()
                emb_dim_per_shard = getattr(self.config, 'emb_dim', x.shape[-1]) // tensor_parallelism_size
                output_shape = jnp.zeros(
                    (original_inputs_first_dim, emb_dim_per_shard),
                    dtype=intermediate_output.dtype,
                )
                
                if is_batch_sharded_by_expert:
                    # locally unpermute back to the original order
                    local_output = jnp.take(intermediate_output, indices=jnp.argsort(local_sorted_indices), axis=0)
                    input_offsets, send_sizes, output_offsets, recv_sizes = RoutedMoE.get_all_to_all_params(
                        jnp.transpose(all_shards_group_sizes), expert_shard_id, num_expert_parallelism
                    )
                    intermediate_output = jax.lax.ragged_all_to_all(
                        local_output,
                        output_shape,
                        input_offsets,
                        send_sizes,
                        output_offsets,
                        recv_sizes,
                        axis_name=expert_axis_name,
                    )
                else:
                    # If batch is replicated across EP shards then each shard should send
                    # 0..local_shard_size data to the other shards and receive the local_shard data from
                    # all of the other shards using ragged_all_to_all.
                    input_offsets, send_sizes, output_offsets, recv_sizes = RoutedMoE.get_all_to_all_params(
                        reshaped_group_sizes, expert_shard_id, num_expert_parallelism, is_batch_sharded=False
                    )
                    intermediate_output = jax.lax.ragged_all_to_all(
                        intermediate_output,
                        output_shape,
                        input_offsets,
                        send_sizes,
                        output_offsets,
                        recv_sizes,
                        axis_name=expert_axis_name,
                    )

            output = self.unpermute(
                intermediate_output, sorted_selected_experts, weights, batch_size=batch_size, sequence_length=sequence_length
            )
            return output, None

        return wrapper(inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel)

    def reshape_and_update_weights(self, weights, indices):
        update_weights = jnp.zeros((weights.shape[0], weights.shape[1], self.num_experts), dtype=self.dtype)
        index_update = (
            jnp.arange(weights.shape[0])[:, None, None],
            jnp.arange(weights.shape[1])[:, None],
            indices,
        )
        update_weights = update_weights.at[index_update].set(weights)
        return update_weights

    def get_context_partition_and_sub_seq(self, seq_len):
        cp = self.get_context_autoregressive_parallelism_size()
        if seq_len % cp != 0:
            cp = 1
        sub_seq = seq_len // cp
        return cp, sub_seq

    def generate_masks_subgroup(self, top_k_indices, softmax_probs):
        import math
        batch_size, seq_len, _ = top_k_indices.shape
        cp, sub_seq = self.get_context_partition_and_sub_seq(seq_len)

        top_k_indices = jnp.reshape(top_k_indices, (batch_size, cp, sub_seq, top_k_indices.shape[2]))

        tokens_per_batch = sub_seq * self.num_experts_per_tok
        capacity_factor = getattr(self.config, 'capacity_factor', 1.0)
        expert_capacity_per_batch = int(
            max(
                math.ceil(tokens_per_batch / self.num_experts) * capacity_factor,
                capacity_factor,
            )
        )

        expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
        expert_mask_fused = jnp.reshape(expert_mask, (batch_size, cp, sub_seq * self.num_experts_per_tok, self.num_experts))
        expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=2)
        expert_token_count = jnp.reshape(
            expert_token_count_fused,
            ((batch_size, cp, sub_seq, self.num_experts_per_tok, self.num_experts)),
        )
        trunc_expert_mask = expert_mask * jnp.less_equal(expert_token_count, expert_capacity_per_batch)
        combined_expert_mask = jnp.sum(trunc_expert_mask, axis=3)

        softmax_probs = jnp.reshape(softmax_probs, ((batch_size, cp, sub_seq, self.num_experts)))
        softmax_probs *= combined_expert_mask

        expert_token_position_fused = expert_mask_fused * expert_token_count_fused
        expert_token_position = jnp.reshape(
            expert_token_position_fused,
            (batch_size, cp, sub_seq, self.num_experts_per_tok, self.num_experts),
        )
        combined_expert_token_position = jnp.sum(expert_token_position, axis=3) * combined_expert_mask
        expert_token_position_in_capacity = jax.nn.one_hot(
            combined_expert_token_position,
            num_classes=expert_capacity_per_batch + 1,
            dtype=jnp.int32,
        )

        combine_mask = softmax_probs[..., None] * expert_token_position_in_capacity
        combine_mask = combine_mask[..., 1:]
        dispatch_mask = combine_mask.astype(bool)

        dispatch_mask = jnp.reshape(dispatch_mask, (batch_size, cp, sub_seq, self.num_experts, expert_capacity_per_batch))
        combine_mask = jnp.reshape(combine_mask, (batch_size, cp, sub_seq, self.num_experts, expert_capacity_per_batch))

        return dispatch_mask, combine_mask

    def generate_masks(self, top_k_indices, softmax_probs):
        import math
        batch_size, seq_len, _ = top_k_indices.shape

        tokens_per_batch = seq_len * self.num_experts_per_tok
        capacity_factor = getattr(self.config, 'capacity_factor', 1.0)
        expert_capacity_per_batch = int(
            max(
                math.ceil(tokens_per_batch / self.num_experts) * capacity_factor,
                capacity_factor,
            )
        )

        expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
        expert_mask_fused = jnp.reshape(expert_mask, (batch_size, seq_len * self.num_experts_per_tok, self.num_experts))
        expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=1)
        expert_token_count = jnp.reshape(
            expert_token_count_fused,
            ((batch_size, seq_len, self.num_experts_per_tok, self.num_experts)),
        )
        trunc_expert_mask = expert_mask * jnp.less_equal(expert_token_count, expert_capacity_per_batch)
        combined_expert_mask = jnp.sum(trunc_expert_mask, axis=2)

        softmax_probs *= combined_expert_mask

        expert_token_position_fused = expert_mask_fused * expert_token_count_fused
        expert_token_position = jnp.reshape(
            expert_token_position_fused,
            (batch_size, seq_len, self.num_experts_per_tok, self.num_experts),
        )
        combined_expert_token_position = jnp.sum(expert_token_position, axis=2) * combined_expert_mask
        expert_token_position_in_capacity = jax.nn.one_hot(
            combined_expert_token_position,
            num_classes=expert_capacity_per_batch + 1,
            dtype=jnp.int32,
        )

        combine_mask = softmax_probs[..., None] * expert_token_position_in_capacity
        combine_mask = combine_mask[..., 1:]
        dispatch_mask = combine_mask.astype(bool)

        return dispatch_mask, combine_mask

    def load_balance_loss(self, top_k_indices, logits):
        expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
        summed_expert_mask = jnp.sum(expert_mask, axis=2)
        density = jnp.mean(summed_expert_mask, axis=1)
        density_prob = jnp.mean(logits, axis=1)
        load_balance_loss_weight = getattr(self.config, 'load_balance_loss_weight', 0.01)
        loss = jnp.mean(density * density_prob) * (self.num_experts**2) * load_balance_loss_weight
        return loss

    def get_einsum(self, rhs_mesh_axes = (), einsum_name=None):
        model_call_mode = getattr(self.config, 'model_call_mode', 'training')
        if model_call_mode == "inference" and einsum_name in (DISPATCH, COMBINE):
            return jnp.einsum
        return jnp.einsum

    def maybe_all_gather_kernel_weight_in_expert_parallelism(self, kernel, kernel_axes):
        if self.get_expert_parallelism_size() <= 1:
            return kernel
            
        need_all_gather = (
            kernel_axes and 
            len(kernel_axes) > 0 and 
            kernel_axes[0] == "exp" and
            hasattr(kernel, 'sharding')
        )
        
        if not need_all_gather:
            return kernel
            
        try:
            if hasattr(jax, '_src') and hasattr(jax._src, 'core'):
                current_mesh = getattr(jax._src.core, 'thread_local_state', None)
                if current_mesh and hasattr(current_mesh, 'get_mesh'):
                    mesh = current_mesh.get_mesh()
                    if mesh and "expert" in mesh.axis_names:
                        gathered_kernel = jax.lax.all_gather(
                            kernel, 
                            axis_name="expert",
                            axis=0,
                            tiled=True
                        )
                        return gathered_kernel
        except Exception as e:
            import warnings
            warnings.warn(
                f"Expert parallelism all-gather failed: {e}. "
                "Using local weights which may affect model accuracy.",
                RuntimeWarning
            )
        return kernel

    def dense_matmul(self, inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel):
        top_k_weights, top_k_indices = self.get_topk(gate_logits, pre_bias_logits)
        
        decoder_block = getattr(self.config, 'decoder_block', None)
        if hasattr(decoder_block, 'name'):
            decoder_block = decoder_block.name
        elif isinstance(decoder_block, str):
            pass
        else:
            decoder_block = str(decoder_block) if decoder_block else None
            
        is_llama4_decoder_layer = decoder_block == "LLAMA4"
        if is_llama4_decoder_layer:
            router_scores = jax.nn.sigmoid(top_k_weights.astype(jnp.float32)).astype(jnp.bfloat16)
            inputs = inputs * router_scores
        else:
            weights = self.reshape_and_update_weights(top_k_weights, top_k_indices)
            
        matmul_precision = getattr(self.config, 'matmul_precision', 'default')
        precision = jax.lax.Precision(matmul_precision) if matmul_precision != 'default' else None

        model_call_mode = getattr(self.config, 'model_call_mode', 'training')
        if model_call_mode != "inference":
            softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
            loss = self.load_balance_loss(top_k_indices, softmax_probs)
        else:
            loss = None
            
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        cp, sub_seq = self.get_context_partition_and_sub_seq(seq_len)
        capacity_factor = getattr(self.config, 'capacity_factor', 0)

        if capacity_factor > 0:
            # token dropping if needed
            if model_call_mode != "inference":
                dispatch_mask, combine_mask = self.generate_masks(top_k_indices, weights)
                mask_axes = ("activation_batch", "activation_length", None, None)
                input_axis = ("activation_batch", "activation_length", "activation_embed")
                dispatch_axis = ("activation_exp", "activation_batch_no_exp", None, "activation_embed")
                mlp_axis = ("activation_exp", "activation_batch_no_exp", None, "activation_mlp")
                dispatch_einsum = "BSM,BSEC -> EBCM"
                mlp_up_einsum = "EBCM,EMH -> EBCH"
                mlp_down_einsum = "EBCH,EHM -> EBCM"
                output_einsum = "EBCM,BSEC -> BSM"
            else:
                softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
                dispatch_mask, combine_mask = self.generate_masks_subgroup(top_k_indices, softmax_probs)
                if self.get_context_autoregressive_parallelism_size() > 0 and cp == 1:
                    mask_axes = ("activation_length", "activation_batch", None, None, None)
                    input_axis = ("activation_length", "activation_batch", None, "activation_embed")
                    dispatch_axis = ("activation_exp", "activation_batch_no_exp", None, None, "activation_embed")
                    mlp_axis = ("activation_exp", "activation_batch_no_exp", None, None, "activation_mlp")
                else:
                    mask_axes = ("activation_batch", "activation_length", None, None, None)
                    input_axis = ("activation_batch", "activation_length", None, "activation_embed")
                    dispatch_axis = ("activation_exp", "activation_batch_no_exp", None, None, "activation_embed")
                    mlp_axis = ("activation_exp", "activation_batch_no_exp", None, None, "activation_mlp")
                dispatch_einsum = "BNSM,BNSEC -> EBNCM"
                mlp_up_einsum = "EBNCM,EMH -> EBNCH"
                mlp_down_einsum = "EBNCH,EHM -> EBNCM"
                output_einsum = "EBNCM,BNSEC -> BNSM"

                inputs = jnp.reshape(inputs, (batch_size, cp, sub_seq, inputs.shape[2]))

            with jax.named_scope("dispatch"):
                dispatch = self.get_einsum(rhs_mesh_axes=mask_axes, einsum_name=DISPATCH)(
                    dispatch_einsum, inputs, dispatch_mask, precision=precision
                )

            with jax.named_scope("wi_0"):
                w0_kernel_axes = ("exp", None, "mlp")
                w0_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(w0_kernel, w0_kernel_axes)
                layer_w0 = self.get_einsum(rhs_mesh_axes=w0_kernel_axes)(
                    mlp_up_einsum, dispatch, w0_kernel, precision=precision
                )

                activations_in_float32 = getattr(self.config, 'activations_in_float32', False)
                if activations_in_float32:
                    layer_w0 = layer_w0.astype(jnp.float32)
                    
            with jax.named_scope("wi_1"):
                w1_kernel_axes = ("exp", None, "mlp")
                w1_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(w1_kernel, w1_kernel_axes)
                layer_w1 = self.get_einsum(rhs_mesh_axes=w1_kernel_axes)(
                    mlp_up_einsum, dispatch, w1_kernel, precision=precision
                )
                if activations_in_float32:
                    layer_w1 = layer_w1.astype(jnp.float32)

            mlp_activations = getattr(self.config, 'mlp_activations', ['silu'])
            if mlp_activations[0] == 'silu':
                layer_w0_act = jax.nn.silu(layer_w0)
            elif mlp_activations[0] == 'relu':
                layer_w0_act = jax.nn.relu(layer_w0)
            elif mlp_activations[0] == 'gelu':
                layer_w0_act = jax.nn.gelu(layer_w0)
            else:
                layer_w0_act = jax.nn.silu(layer_w0)
                
            layer_multiply = jnp.multiply(layer_w0_act, layer_w1).astype(self.dtype)
            
            with jax.named_scope("wo"):
                wo_kernel_axes = ("exp", "mlp", None)
                wo_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(wo_kernel, wo_kernel_axes)
                intermediate_layer = self.get_einsum(rhs_mesh_axes=wo_kernel_axes)(
                    mlp_down_einsum, layer_multiply, wo_kernel, precision=precision
                )
                if activations_in_float32:
                    intermediate_layer = intermediate_layer.astype(jnp.float32)

            with jax.named_scope("combine"):
                output = self.get_einsum(rhs_mesh_axes=mask_axes, einsum_name=COMBINE)(
                    output_einsum,
                    intermediate_layer,
                    combine_mask,
                    precision=precision,
                )
                if output.ndim == 4:
                    output = jnp.reshape(output, (output.shape[0], output.shape[1] * output.shape[2], output.shape[3]))
            return output, loss
        else:
            with jax.named_scope("wi_0"):
                layer_w0 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
                    "BSM,EMH -> BSEH", inputs, w0_kernel, precision=precision
                )
                activations_in_float32 = getattr(self.config, 'activations_in_float32', False)
                if activations_in_float32:
                    layer_w0 = layer_w0.astype(jnp.float32)

            with jax.named_scope("wi_1"):
                layer_w1 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
                    "BSM,EMH -> BSEH", inputs, w1_kernel, precision=precision
                )
                if activations_in_float32:
                    layer_w1 = layer_w1.astype(jnp.float32)

            mlp_activations = getattr(self.config, 'mlp_activations', ['silu'])
            if mlp_activations[0] == 'silu':
                layer_w0_act = jax.nn.silu(layer_w0)
            elif mlp_activations[0] == 'relu':
                layer_w0_act = jax.nn.relu(layer_w0)
            elif mlp_activations[0] == 'gelu':
                layer_w0_act = jax.nn.gelu(layer_w0)
            else:
                layer_w0_act = jax.nn.silu(layer_w0)
                
            layer_multiply = jnp.multiply(layer_w0_act, layer_w1).astype(self.dtype)
            
            with jax.named_scope("wo"):
                intermediate_layer = self.get_einsum(rhs_mesh_axes=self.wo_kernel_axes)(
                    "BSEH,EHM -> BSEM", layer_multiply, wo_kernel, precision=precision
                )
                if activations_in_float32:
                    intermediate_layer = intermediate_layer.astype(jnp.float32)

            with jax.named_scope("w_sum"):
                if is_llama4_decoder_layer:
                    weights = self.reshape_and_update_weights(jnp.ones_like(top_k_weights), top_k_indices)
                output = jnp.einsum("BSEM,BSE -> BSM", intermediate_layer, weights).astype(self.dtype)
            return output, None

    def __call__(self, inputs):
        cfg = self.config
        inputs = inputs.astype(cfg.dtype)
        gate_logits, pre_bias_logits = self.gate(inputs)

        w0_kernel, w1_kernel, wo_kernel = self.wi_0.value, self.wi_1.value, self.wo.value
        sparse_matmul = getattr(cfg, 'sparse_matmul', False)
        if sparse_matmul:
            return self.sparse_matmul(inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel)
        else:
            return self.dense_matmul(inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel)