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

class RoutedMoE(nnx.Module):
    """Implements a routed MoE block.

    Attributes:
        config: Configuration object.
        num_experts: Number of experts.
        num_experts_per_tok: Number of experts for each token.
        mesh: Mesh, device mesh.
        intermediate_dim: Intermediate dimension of MoE.
        weight_dtype: Type for the weights.
        dtype: Type for the dense layer.
    """

    def __init__(self,
                 config,
                 num_experts: int,
                 num_experts_per_tok: int,
                 mesh: Mesh,
                 intermediate_dim: int = 2048,
                 weight_dtype: jnp.dtype = jnp.float32,
                 dtype: jnp.dtype = jnp.float32,
                 rngs: nnx.Rngs = None):
        
        self.config = config
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.mesh = mesh
        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        
        # Kernel axes
        self.wi_kernel_axes = ("exp", "embed_no_exp", "mlp")
        self.wo_kernel_axes = ("exp", "mlp", "embed_no_exp")
        
        # 初始化 gate 层
        self.gate = GateLogit(
            input_size=config.emb_dim,
            features=self.num_experts,
            model_name=config.model_name,
            weight_dtype=self.weight_dtype,
            dtype=self.dtype,
            kernel_axes=getattr(config, 'kernel_axes', None),
            use_bias=getattr(config, 'routed_bias', False),
            score_func=getattr(config, 'routed_score_func', ""),
            rngs=rngs
        )
        
        # 初始化专家权重
        self.w0_kernel = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), self.wi_kernel_axes)(
                rngs.params(), 
                (self.num_experts, config.emb_dim, self.intermediate_dim), 
                self.weight_dtype
            )
        )
        
        self.w1_kernel = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), self.wi_kernel_axes)(
                rngs.params(), 
                (self.num_experts, config.emb_dim, self.intermediate_dim), 
                self.weight_dtype
            )
        )
        
        self.wo_kernel = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), self.wo_kernel_axes)(
                rngs.params(), 
                (self.num_experts, self.intermediate_dim, config.emb_dim), 
                self.weight_dtype
            )
        )

    def get_expert_parallelism_size(self):
        return self.mesh.shape["expert"]

    def get_tensor_parallelism_size(self):
        return self.mesh.shape["tensor"]

    def get_context_autoregressive_parallelism_size(self):
        return self.mesh.shape["context_autoregressive"]

    def generate_kernels(self):
        w0_kernel = jnp.asarray(self.w0_kernel.value, self.dtype)
        w1_kernel = jnp.asarray(self.w1_kernel.value, self.dtype)
        wo_kernel = jnp.asarray(self.wo_kernel.value, self.dtype)
        return w0_kernel, w1_kernel, wo_kernel

    def get_topk(self, gate_logits, pre_bias_logits):
        """get topk. shape of top_k_weights & top_k_indices: (batch, sequence, num_experts_per_tok)"""
        if getattr(self.config, 'use_random_routing', False):
            top_k_weights, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
            return top_k_weights, top_k_indices

        if self.config.model_name.startswith("deepseek3"):
            top_k_weights, top_k_indices = self.deepseek_routing(gate_logits, pre_bias_logits)
        else:
            top_k_weights, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)

        decoder_block = getattr(self.config, 'decoder_block', None)
        if decoder_block == "DEEPSEEK":
            top_k_weights = self.deepseek_scale_weights(top_k_weights)
        elif decoder_block != "LLAMA4":
            top_k_weights = jax.nn.softmax(top_k_weights.astype(jnp.float32), axis=-1).astype(self.dtype)
        return top_k_weights, top_k_indices

    def deepseek_scale_weights(self, weights):
        """Scales weights according to DeepSeek's v3 reference implementation."""
        routed_score_func = getattr(self.config, 'routed_score_func', '')
        if routed_score_func == "sigmoid":
            weights /= weights.sum(-1, keepdims=True)
        routed_scaling_factor = getattr(self.config, 'routed_scaling_factor', 1.0)
        weights *= routed_scaling_factor
        return weights

    def deepseek_routing(self, gate_logits, pre_bias_logits):
        """DeepSeek routing logic."""
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

    def reshape_and_update_weights(self, weights, indices):
        """reshape and update weights."""
        update_weights = jnp.zeros((weights.shape[0], weights.shape[1], self.num_experts), dtype=self.dtype)
        index_update = (
            jnp.arange(weights.shape[0])[:, None, None],
            jnp.arange(weights.shape[1])[:, None],
            indices,
        )
        update_weights = update_weights.at[index_update].set(weights)
        return update_weights

    def load_balance_loss(self, top_k_indices, logits):
        """See Switch Transformer for more details."""
        expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
        summed_expert_mask = jnp.sum(expert_mask, axis=2)
        density = jnp.mean(summed_expert_mask, axis=1)
        density_prob = jnp.mean(logits, axis=1)
        load_balance_loss_weight = getattr(self.config, 'load_balance_loss_weight', 0.01)
        loss = jnp.mean(density * density_prob) * (self.num_experts**2) * load_balance_loss_weight
        return loss

    def dense_matmul(self, inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel):
        """dense matrix multiplication"""
        top_k_weights, top_k_indices = self.get_topk(gate_logits, pre_bias_logits)
        
        decoder_block = getattr(self.config, 'decoder_block', None)
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

        with jax.named_scope("wi_0"):
            if precision:
                layer_w0 = jnp.einsum("BSM,EMH -> BSEH", inputs, w0_kernel, precision=precision)
            else:
                layer_w0 = jnp.einsum("BSM,EMH -> BSEH", inputs, w0_kernel)
                
        with jax.named_scope("wi_1"):
            if precision:
                layer_w1 = jnp.einsum("BSM,EMH -> BSEH", inputs, w1_kernel, precision=precision)
            else:
                layer_w1 = jnp.einsum("BSM,EMH -> BSEH", inputs, w1_kernel)

        mlp_activations = getattr(self.config, 'mlp_activations', ['silu'])
        if mlp_activations[0] == 'silu':
            layer_w0_act = jax.nn.silu(layer_w0)
        elif mlp_activations[0] == 'relu':
            layer_w0_act = jax.nn.relu(layer_w0)
        else:
            layer_w0_act = jax.nn.silu(layer_w0)
            
        layer_multiply = jnp.multiply(layer_w0_act, layer_w1).astype(self.dtype)
        
        with jax.named_scope("wo"):
            if precision:
                intermediate_layer = jnp.einsum("BSEH,EHM -> BSEM", layer_multiply, wo_kernel, precision=precision)
            else:
                intermediate_layer = jnp.einsum("BSEH,EHM -> BSEM", layer_multiply, wo_kernel)

        with jax.named_scope("w_sum"):
            if is_llama4_decoder_layer:
                weights = self.reshape_and_update_weights(jnp.ones_like(top_k_weights), top_k_indices)
            output = jnp.einsum("BSEM,BSE -> BSM", intermediate_layer, weights).astype(self.dtype)
            
        return output, loss

    def __call__(self, inputs):
        inputs = inputs.astype(self.config.dtype)
        
        gate_logits, pre_bias_logits = self.gate(inputs)
        w0_kernel, w1_kernel, wo_kernel = self.generate_kernels()
        
        sparse_matmul = getattr(self.config, 'sparse_matmul', False)
        if sparse_matmul:
            return self.dense_matmul(inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel)
        else:
            return self.dense_matmul(inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel)