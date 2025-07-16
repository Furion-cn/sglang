from typing import Any, Dict, Optional, Tuple
from sglang.srt.jax.layers.logits_processor import LogitsProcessor
from flax import nnx
from jax import numpy as jnp
from jax import jax

from transformers import PretrainedConfig
from sglang.srt.jax.layers.layernorm import RMSNorm
from sglang.srt.jax.layers.linear import LinearBase
from sglang.srt.jax.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sglang.srt.jax.layers.attention import Attention
from sglang.debug_tracer import global_tracer, trace_function
from sglang.srt.jax.utils import (
    flatten_pytree_with_paths,
    get_expected_param_paths,
    update_state_recursive,
)
from sglang.srt.jax.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.jax.models.qwen3 import Qwen3MLP
from sglang.srt.jax.layers.moe import GateLogit, Qwen3MoE
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

class QWen3MoeAttention(nnx.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position_embeddings: int,
                 rope_theta: float = 10000,
                 rope_scaling: Optional[Dict[str, Any]] = None,
                 head_dim: Optional[int] = None,
                 rms_norm_eps: float = None,
                 layer_id: int = 0,
                 attention_bias: bool = False,
                 rngs: nnx.Rngs = None):
        self.layer_id = layer_id
        assert num_heads % num_kv_heads == 0
        self.head_dim = head_dim or hidden_size // num_heads

        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        
        self.q_norm = RMSNorm(self.head_dim, epsilon=rms_norm_eps, rngs=rngs)
        self.k_norm = RMSNorm(self.head_dim, epsilon=rms_norm_eps, rngs=rngs)
        self.c_attn = LinearBase(
            input_size=hidden_size,
            output_size=(num_heads + 2 * num_kv_heads) * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
        )
        jax.debug.print("{c_attn_type}", c_attn_type=type(self.c_attn.weight))
        self.c_proj = LinearBase(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            rngs=rngs,
        )
        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            dtype=jnp.bfloat16,
        )
        self.attn = Attention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            scale=self.scaling,
        )

    @trace_function(stage="MOE_ATTENTION_FORWARD", include_args=False, include_output=True)
    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
    ) -> jax.Array:
        q, k, v = self._proj_qkv(positions, hidden_states)
        attn_output = self.attn(q, k, v, forward_batch, self.layer_id, is_causal=True)
        output, _ = self.c_proj(attn_output)
        return output
    
    def _proj_qkv(self, positions, hidden_states, q_size, kv_size):
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = jnp.split(qkv, [q_size, q_size + self.kv_size], axis=-1)

        q_by_head = q.reshape(-1, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.reshape(q.shape)

        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.reshape(k.shape)

        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

class QWen3MoeDecoderLayer(nnx.Module):
    def __init__(self,
                 config: PretrainedConfig,
                 layer_id: int = 0,
                 rngs: nnx.Rngs = None):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 40960)
        head_dim = getattr(config, "head_dim", None)
        
        self.self_attn = QWen3MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            rms_norm_eps=config.rms_norm_eps,
            layer_id=layer_id,
            attention_bias=getattr(config, 'attention_bias', False),
            rngs=rngs,
        )

        mlp_only_layers = getattr(config, 'mlp_only_layers', [])
        
        if layer_id in mlp_only_layers:
            self.mlp = Qwen3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_id=layer_id,
                rngs=rngs,
            )
            self.is_moe_layer = False
            self.moe_gate = None
        else:
            self.mesh = getattr(config, 'mesh', None)
            if self.mesh is None:
                raise ValueError("Need mesh in config")      
                  
            num_experts = getattr(config, 'num_experts', 128)
            num_experts_per_tok = getattr(config, 'num_experts_per_tok', 8)
            moe_intermediate_size = getattr(config, 'moe_intermediate_size', 768)
            expert_parallel_size = self.mesh.shape.get('data', 1) * self.mesh.shape.get('tensor', 1)
            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                features=num_experts,
                model_name=getattr(config, 'model_name', 'qwen3_moe'),
                use_bias=False,
                kernel_axes=(None, 'expert'), 
                dtype=jnp.bfloat16,
                layer_id=layer_id,
                rngs=rngs
            )
            self.mlp = Qwen3MoE(
                config=config,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                intermediate_dim=moe_intermediate_size,
                mesh=self.mesh,
                expert_parallel_size=expert_parallel_size,
                weight_dtype=jnp.bfloat16,
                dtype=jnp.bfloat16,
                layer_id=layer_id,
                rngs=rngs,
            )
            self.is_moe_layer = True

        self.input_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs)

    @trace_function(stage="MOE_DECODER_LAYER_FORWARD", include_args=False, include_output=True)
    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        residual: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        global_tracer.print(hidden_states, f"decoder_layer_input", f"moe_decoder_layer_id_{self.layer_id}")
        
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        global_tracer.print(hidden_states, f"input_layernorm_output", f"moe_decoder_layer_id_{self.layer_id}")
        global_tracer.print(residual, f"residual_after_input_norm", f"moe_decoder_layer_id_{self.layer_id}")
        
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        
        global_tracer.print(hidden_states, f"self_attn_output", f"moe_decoder_layer_id_{self.layer_id}")
        
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        global_tracer.print(hidden_states, f"post_attention_layernorm_output", f"moe_decoder_layer_id_{self.layer_id}")
        global_tracer.print(residual, f"residual_after_post_attn_norm", f"moe_decoder_layer_id_{self.layer_id}")
        
        if self.is_moe_layer:
            router_logits = self.moe_gate(hidden_states)            
            global_tracer.print(router_logits, f"gate_final_output", f"moe_gate_layer_id_{self.layer_id}")
            
            mlp_output = self.mlp(hidden_states, router_logits=router_logits)
            global_tracer.print(mlp_output, f"moe_output", f"moe_decoder_layer_id_{self.layer_id}")
                        
            hidden_states = mlp_output
        else:
            hidden_states = self.mlp(hidden_states)
            
        return hidden_states, residual

class QWen3MoeModel(nnx.Module):
    def __init__(self,
                 config: PretrainedConfig,
                 rngs: nnx.Rngs = None):
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            rngs=rngs,
        )

        self.layers = [
            QWen3MoeDecoderLayer(
                config=config,
                layer_id=i,
                rngs=rngs,
            )
            for i in range(config.num_hidden_layers)
        ]

        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs)

    @trace_function(stage="MOE_TRANSFORMER_FORWARD", include_args=False, include_output=True)
    def __call__(self,
                 input_ids: jax.Array,
                 positions: jax.Array,
                 forward_batch: ForwardBatch,
                 ) -> jax.Array:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        
        if residual is not None:
            hidden_states, residual = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        
        return hidden_states


class Qwen3MoeForCausalLMJaxModel(nnx.Module):
    def __init__(self,
                 config: PretrainedConfig,
                 rngs: nnx.Rngs = None):
        self.config = config
        self.model = QWen3MoeModel(config, rngs)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, rngs=rngs)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self._setup_debug_tracer()

    def _setup_debug_tracer(self):
        try:
            global_tracer.set_model(self)
        except Exception as e:
            print(f"Warning: Could not setup debug tracer: {str(e)}")

    def load_pytree_weights(self, pytree):
        flat_weights = flatten_pytree_with_paths(pytree)
        model_state = nnx.state(self)
        expected_paths = get_expected_param_paths(model_state)
        missing_paths = expected_paths - set(flat_weights.keys())
        if missing_paths:
            raise ValueError(
                f"Missing weights for parameters: {sorted(missing_paths)}")

        update_state_recursive(model_state, flat_weights)
        pspecs = nnx.get_partition_spec(model_state)
        pstate = jax.lax.with_sharding_constraint(model_state, pspecs)
        nnx.update(self, pstate)
        # self._apply_sharding_constraints_with_mixed_meshes(model_state)

    def _apply_sharding_constraints_with_mixed_meshes(self, model_state):
        import jax
        from jax.sharding import PartitionSpec as P
        
        expert_mesh = getattr(self.config, 'expert_mesh', None)
        
        if expert_mesh is None:
            pspecs = nnx.get_partition_spec(model_state)
            pstate = jax.lax.with_sharding_constraint(model_state, pspecs)
            nnx.update(self, pstate)
            return
        
        print(f"apply mix mesh constraint...")
        print(f"Expert mesh: {expert_mesh}")
        
        pspecs = nnx.get_partition_spec(model_state)
        
        moe_layer_ids = set()
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'is_moe_layer') and layer.is_moe_layer:
                moe_layer_ids.add(i)
        
        print(f"MoE layer ids: {moe_layer_ids}")
        
        def is_moe_parameter(path):
            for layer_id in moe_layer_ids:
                layer_path = f"model/layers/{layer_id}"
                if layer_path in path:
                    if any(component in path for component in ['mlp', 'moe_gate']):
                        return True
            return False
        
        def get_moe_partition_spec(path, original_pspec):
            if 'moe_gate' in path:
                return P(None, 'expert')
            elif 'mlp' in path:
                if len(original_pspec.axis_names) >= 2:
                    return P('expert', *(original_pspec.axis_names[1:]))
                else:
                    return P('expert', None)
            else:
                return original_pspec
        
        def deep_override(specs, state, path=""):
            if isinstance(specs, dict) and isinstance(state, dict):
                result = {}
                for key in specs:
                    if key in state:
                        current_path = f"{path}/{key}" if path else key
                        result[key] = deep_override(specs[key], state[key], current_path)
                    else:
                        result[key] = specs[key]
                return result
            elif isinstance(specs, P) and hasattr(state, 'shape'):
                if is_moe_parameter(path):
                    new_pspec = get_moe_partition_spec(path, specs)
                    print(f"rewrite: {path} -> {new_pspec}")
                    return new_pspec
                else:
                    return specs
            else:
                return specs
        
        modified_pspecs = deep_override(pspecs, model_state)

        
        try:
            def apply_mixed_constraints(state, specs, path=""):
                if isinstance(state, dict) and isinstance(specs, dict):
                    result = {}
                    for key in state:
                        if key in specs:
                            current_path = f"{path}/{key}" if path else key
                            result[key] = apply_mixed_constraints(state[key], specs[key], current_path)
                        else:
                            result[key] = state[key]
                    return result
                elif hasattr(state, 'shape') and isinstance(specs, P):
                    if is_moe_parameter(path):
                        with expert_mesh:
                            return jax.lax.with_sharding_constraint(state, specs)
                    else:
                        return jax.lax.with_sharding_constraint(state, specs)
                else:
                    return state
            
            constrained_state = apply_mixed_constraints(model_state, modified_pspecs)
            nnx.update(self, constrained_state)
            print("mix mesh constraint applied")
            
        except Exception as e:
            print(f"mix mesh constraint failed: {e}")
            try:
                pstate = jax.lax.with_sharding_constraint(model_state, pspecs)
                nnx.update(self, pstate)
                print("fallback to standard constraint")
            except Exception as fallback_e:
                print(f"standard constraint failed: {fallback_e}")
                nnx.update(self, model_state)
                print("use unconstrainted model state")

    @trace_function(stage="MOE_CAUSAL_LM_FORWARD", include_args=False, include_output=True)
    def __call__(self,
                 input_ids: jax.Array,
                 positions: jax.Array,
                 forward_batch: ForwardBatch,
                 ) -> Any:
        hidden_states = self.model(input_ids, positions, forward_batch)
        result = self.logits_processor(hidden_states, self.lm_head, forward_batch)
        return result

EntryClass = Qwen3MoeForCausalLMJaxModel