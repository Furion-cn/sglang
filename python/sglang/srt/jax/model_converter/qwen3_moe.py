# pylint: disable=g-line-too-long
import argparse
import pathlib
import os
import gc
import logging
import shutil
import json

from safetensors import safe_open

import ml_dtypes

import psutil

from tqdm import tqdm

import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"

import torch

import jax
from jax import tree

from flax.training import train_state
from flax import serialization

from sglang.srt.jax.model_converter import check_pointing
from sglang.srt.jax.model_converter import converter_logging
from sglang.srt.jax.model_converter.utils import str2bool, sed_model_config
from sglang.srt.jax.model_converter import save_checkpoint

MODEL_PARAMS_DICT = {

    "qwen3-30b-a3b": {
        "attention_bias": False,
        "num_layers": 48,
        "num_heads": 32,
        "num_kv_heads": 4,
        "dims_per_head": 128,
        "vocab": 151936,
        "base_emb_dim": 2048,
        "base_mlp_dim": 6144,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "mlp_only_layers": [],
    },
}

SIMULATED_CPU_DEVICES_COUNT = 16

# NOTE: numpy doesn't have native support for bfloat16, so
# we'll use ml_dtypes instead (which is quasi native)
# NOTE: it's incredibly silly but you can't directly cast from
# a torch tensor of type bfloat16 to a numpy array of type bfloat16
# so we have to cast to float32 first
CAST_DTYPE = ml_dtypes.bfloat16


def list_safetensor_keys(model_path: str):
    """列出 safetensors 文件中的所有键名以分析 Qwen3 MoE 结构"""
    ckpt_paths = sorted(pathlib.Path(model_path).glob("*.safetensors"))
    
    all_keys = []
    converter_logging.log(f"Found {len(ckpt_paths)} safetensors files")
    
    for i, ckpt_path in enumerate(ckpt_paths):
        converter_logging.log(f"=== File {i+1}: {ckpt_path.name} ===")
        
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            converter_logging.log(f"Total {len(keys)} weights:")
            
            for key in sorted(keys):
                tensor = f.get_tensor(key)
                converter_logging.log(f"  {key:<80} {str(tensor.shape):<20} {tensor.dtype}")
                all_keys.append(key)
    
    return all_keys


def _infer_model_params_from_config(config_path: str) -> dict:
    """从 config.json 文件推断模型参数"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return {
            "attention_bias": config.get("attention_bias", False),
            "num_layers": config["num_hidden_layers"],
            "num_heads": config["num_attention_heads"],
            "num_kv_heads": config["num_key_value_heads"],
            "dims_per_head": config.get("head_dim", config["hidden_size"] // config["num_attention_heads"]),
            "vocab": config["vocab_size"],
            "base_emb_dim": config["hidden_size"],
            "base_mlp_dim": config["intermediate_size"],
            "num_experts": config.get("num_experts", 128),
            "num_experts_per_tok": config.get("num_experts_per_tok", 8),
            "moe_intermediate_size": config.get("moe_intermediate_size", 768),
            "mlp_only_layers": config.get("mlp_only_layers", []),
        }
    except Exception as e:
        converter_logging.log(f"❌ Failed to read config file {config_path}: {str(e)}")
        return None


def _get_hf_to_jax_key_mapping(layer_idx: int = -1, has_attention_bias: bool = False, is_moe_layer: bool = True, num_experts: int = 128) -> dict:
    """
    Maps from Qwen3 MoE checkpoint weights to JAX model weights.
    
    Args:
        layer_idx: The layer index of the model.
        has_attention_bias: Whether the model has attention bias.
        is_moe_layer: Whether this layer is a MoE layer or regular MLP layer.
        num_experts: Number of experts for MoE layers.
        
    Returns:
        A dictionary mapping from Qwen3 MoE checkpoint to JAX model weights.
    """
    mapping = {
        # Embeddings and output
        "model.embed_tokens.weight": "model.embed_tokens.embedding",
        "model.norm.weight": "model.norm.weight", 
        "lm_head.weight": "lm_head.embedding",
    }
    
    if layer_idx >= 0:
        # Layer-specific mappings for Qwen3 MoE
        layer_mapping = {
            f"model.layers.{layer_idx}.input_layernorm.weight": f"model.layers.{layer_idx}.input_layernorm.weight",
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": f"model.layers.{layer_idx}.post_attention_layernorm.weight",
            
            # Attention weights - Qwen3 使用分离的 Q、K、V 投影，需要特殊处理合并
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"model.layers.{layer_idx}.self_attn.k_proj.weight": f"model.layers.{layer_idx}.self_attn.k_proj.weight", 
            f"model.layers.{layer_idx}.self_attn.v_proj.weight": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
            
            # Q/K normalization (Qwen3 specific)
            f"model.layers.{layer_idx}.self_attn.q_norm.weight": f"model.layers.{layer_idx}.self_attn.q_norm.weight",
            f"model.layers.{layer_idx}.self_attn.k_norm.weight": f"model.layers.{layer_idx}.self_attn.k_norm.weight",
        }
        
        # 只有当模型有 bias 时才添加 bias 映射
        if has_attention_bias:
            layer_mapping.update({
                f"model.layers.{layer_idx}.self_attn.q_proj.bias": f"model.layers.{layer_idx}.self_attn.q_proj.bias",
                f"model.layers.{layer_idx}.self_attn.k_proj.bias": f"model.layers.{layer_idx}.self_attn.k_proj.bias",
                f"model.layers.{layer_idx}.self_attn.v_proj.bias": f"model.layers.{layer_idx}.self_attn.v_proj.bias",
            })
        
        # MoE 层和普通 MLP 层的权重映射
        if is_moe_layer:
            # MoE 层包含路由器和专家权重
            layer_mapping.update({
                # 路由器权重
                f"model.layers.{layer_idx}.moe_gate.weight": f"model.layers.{layer_idx}.moe_gate.kernel",
            })
            
            # 专家权重 - 动态添加所有专家的权重映射
            for expert_idx in range(num_experts):
                layer_mapping.update({
                    f"model.layers.{layer_idx}.moe.experts.{expert_idx}.gate_proj.weight": f"model.layers.{layer_idx}.moe.experts.{expert_idx}.gate_proj.weight",
                    f"model.layers.{layer_idx}.moe.experts.{expert_idx}.up_proj.weight": f"model.layers.{layer_idx}.moe.experts.{expert_idx}.up_proj.weight",
                    f"model.layers.{layer_idx}.moe.experts.{expert_idx}.down_proj.weight": f"model.layers.{layer_idx}.moe.experts.{expert_idx}.down_proj.weight",
                })
        else:
            # 普通 MLP 层
            layer_mapping.update({
                f"model.layers.{layer_idx}.mlp.gate_proj.weight": f"model.layers.{layer_idx}.mlp.gate_proj.weight",
                f"model.layers.{layer_idx}.mlp.up_proj.weight": f"model.layers.{layer_idx}.mlp.up_proj.weight", 
                f"model.layers.{layer_idx}.mlp.down_proj.weight": f"model.layers.{layer_idx}.mlp.down_proj.weight",
            })
        
        mapping.update(layer_mapping)
    
    return mapping


def _convert_huggingface_to_jax_weights(base_model_path: str, model_size: str, model_params: dict, mem_info: psutil.Process):
    """Convert a Huggingface Qwen3 MoE Checkpoint to a dictionary of Numpy arrays representing the weights.

    Args:
        base_model_path (str): Path to the base model checkpoint.
        model_size (str): Size of the base model.
        model_params (dict): Dictionary containing model parameters.
        mem_info (psutil.Process): Process object to track memory usage.

    Returns:
        jax_weights (dict): Dictionary containing the converted weights.
    """
    base_num_decoder_layers = model_params["num_layers"]
    has_attention_bias = model_params["attention_bias"]
    num_experts = model_params["num_experts"]
    mlp_only_layers = model_params["mlp_only_layers"]

    converter_logging.log(f"Loading the Qwen3 MoE model from {base_model_path}")
    converter_logging.log(f"Model has {num_experts} experts, MLP-only layers: {mlp_only_layers}")
    
    # 首先列出所有权重键名以便调试
    converter_logging.log("Analyzing Qwen3 MoE model structure...")
    all_keys = list_safetensor_keys(base_model_path)
    
    ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors"))
    chkpt_vars = {}
    
    for i, ckpt_path in enumerate(ckpt_paths):
        converter_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")

        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                # 跳过以 .bias 结尾的权重，如果模型没有 attention bias
                if key.endswith(".bias") and not has_attention_bias:
                    converter_logging.log(f"Skipping bias weight: {key}")
                    continue
                    
                chkpt_vars[key] = f.get_tensor(key)

    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

    # Initialize the data structure for storing jax_weights to match Qwen3MoeForCausalLM structure
    jax_weights = {
        "model": {
            "embed_tokens": {"embedding": None},
            "layers": {},
            "norm": {"weight": None},
        },
        "lm_head": {"embedding": None},
        "logits_processor": {},
    }

    # Final layer norm scale
    converter_logging.log("Processing final layer norm scale")
    if "model.norm.weight" in chkpt_vars:
        ln_f_scale = chkpt_vars["model.norm.weight"].to(torch.float32).numpy().astype(CAST_DTYPE)
        jax_weights["model"]["norm"]["weight"] = ln_f_scale
    else:
        converter_logging.log("❌ Final layer norm weight not found")

    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

    # Logits dense
    converter_logging.log("Processing logits dense")
    if "lm_head.weight" in chkpt_vars:
        lm_head_weight = chkpt_vars["lm_head.weight"].to(torch.float32).numpy().astype(CAST_DTYPE)
        jax_weights["lm_head"]["embedding"] = lm_head_weight
    else:
        converter_logging.log("❌ LM head weight not found")

    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

    # Token embedding
    converter_logging.log("Processing token embeddings")
    if "model.embed_tokens.weight" in chkpt_vars:
        jax_weights["model"]["embed_tokens"]["embedding"] = (
            chkpt_vars["model.embed_tokens.weight"].to(torch.float32).numpy().astype(CAST_DTYPE)
        )
    else:
        converter_logging.log("❌ Embedding tokens weight not found")

    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

    # Initialize layer structure
    for layer_idx in range(base_num_decoder_layers):
        is_moe_layer = layer_idx not in mlp_only_layers
        
        # 动态构建 attention 权重结构
        attn_structure = {
            "c_attn": {"weight": None},
            "c_proj": {"weight": None},
            "q_norm": {"weight": None},
            "k_norm": {"weight": None},
        }
        
        # 只有当模型有 attention bias 时才添加 bias 字段
        if has_attention_bias:
            attn_structure["c_attn"]["bias"] = None
            attn_structure["c_proj"]["bias"] = None
            
        layer_structure = {
            "self_attn": attn_structure,
            "input_layernorm": {"weight": None},
            "post_attention_layernorm": {"weight": None},
        }
        
        if is_moe_layer:
            # MoE 层结构
            layer_structure.update({
                "moe_gate": {"kernel": None},
                "mlp": {
                    "wi_0": {"value": None},  # Expert gate weights
                    "wi_1": {"value": None},  # Expert up weights  
                    "wo": {"value": None},    # Expert down weights
                },
            })
        else:
            # 普通 MLP 层结构
            layer_structure.update({
                "mlp": {
                    "gate_proj": {"weight": None},
                    "up_proj": {"weight": None},
                    "down_proj": {"weight": None},
                },
            })
            
        jax_weights["model"]["layers"][layer_idx] = layer_structure

    # Self attention - 处理分离的 Q、K、V 权重
    converter_logging.log("Processing self attention")
    for layer_idx in tqdm(range(base_num_decoder_layers), desc="attention layers", leave=False):
        is_moe_layer = layer_idx not in mlp_only_layers
        
        # 获取当前层的键名映射
        layer_mapping = _get_hf_to_jax_key_mapping(
            layer_idx=layer_idx, 
            has_attention_bias=has_attention_bias, 
            is_moe_layer=is_moe_layer, 
            num_experts=num_experts
        )
        
        # 查找 Q、K、V 权重键
        q_proj_key = None
        k_proj_key = None
        v_proj_key = None
        o_proj_key = None
        q_norm_key = None
        k_norm_key = None
        
        for hf_key, jax_key in layer_mapping.items():
            if hf_key.endswith(f"layers.{layer_idx}.self_attn.q_proj.weight"):
                q_proj_key = hf_key
            elif hf_key.endswith(f"layers.{layer_idx}.self_attn.k_proj.weight"):
                k_proj_key = hf_key
            elif hf_key.endswith(f"layers.{layer_idx}.self_attn.v_proj.weight"):
                v_proj_key = hf_key
            elif hf_key.endswith(f"layers.{layer_idx}.self_attn.o_proj.weight"):
                o_proj_key = hf_key
            elif hf_key.endswith(f"layers.{layer_idx}.self_attn.q_norm.weight"):
                q_norm_key = hf_key
            elif hf_key.endswith(f"layers.{layer_idx}.self_attn.k_norm.weight"):
                k_norm_key = hf_key
        
        # 检查并合并 Q、K、V 权重
        if q_proj_key in chkpt_vars and k_proj_key in chkpt_vars and v_proj_key in chkpt_vars:
            # 获取原始权重 [output_dim, input_dim]
            q_weight = chkpt_vars[q_proj_key].to(torch.float32).numpy().astype(CAST_DTYPE)
            k_weight = chkpt_vars[k_proj_key].to(torch.float32).numpy().astype(CAST_DTYPE)
            v_weight = chkpt_vars[v_proj_key].to(torch.float32).numpy().astype(CAST_DTYPE)
            
            converter_logging.log(f"Layer {layer_idx}: Q shape {q_weight.shape}, K shape {k_weight.shape}, V shape {v_weight.shape}")
            
            # 按照 [Q, K, V] 的顺序合并权重
            qkv_weight = np.concatenate([q_weight, k_weight, v_weight], axis=0)
            
            # 转置以匹配 JAX 模型的期望格式 [hidden_size, total_proj_dim]
            qkv_weight = qkv_weight.transpose()
            
            jax_weights["model"]["layers"][layer_idx]["self_attn"]["c_attn"]["weight"] = qkv_weight
            converter_logging.log(f"✅ Layer {layer_idx}: Combined QKV weight shape {qkv_weight.shape}")
        else:
            missing_keys = []
            if q_proj_key not in chkpt_vars:
                missing_keys.append("q_proj")
            if k_proj_key not in chkpt_vars:
                missing_keys.append("k_proj")
            if v_proj_key not in chkpt_vars:
                missing_keys.append("v_proj")
            converter_logging.log(f"❌ Missing weights for layer {layer_idx}: {missing_keys}")

        # 处理 bias (如果存在)
        if has_attention_bias:
            q_bias_key = None
            k_bias_key = None
            v_bias_key = None
            
            for hf_key, jax_key in layer_mapping.items():
                if hf_key.endswith(f"layers.{layer_idx}.self_attn.q_proj.bias"):
                    q_bias_key = hf_key
                elif hf_key.endswith(f"layers.{layer_idx}.self_attn.k_proj.bias"):
                    k_bias_key = hf_key
                elif hf_key.endswith(f"layers.{layer_idx}.self_attn.v_proj.bias"):
                    v_bias_key = hf_key
            
            if q_bias_key in chkpt_vars and k_bias_key in chkpt_vars and v_bias_key in chkpt_vars:
                q_bias = chkpt_vars[q_bias_key].to(torch.float32).numpy().astype(CAST_DTYPE)
                k_bias = chkpt_vars[k_bias_key].to(torch.float32).numpy().astype(CAST_DTYPE)
                v_bias = chkpt_vars[v_bias_key].to(torch.float32).numpy().astype(CAST_DTYPE)
                
                # 合并 bias
                qkv_bias = np.concatenate([q_bias, k_bias, v_bias], axis=0)
                jax_weights["model"]["layers"][layer_idx]["self_attn"]["c_attn"]["bias"] = qkv_bias
                converter_logging.log(f"✅ Layer {layer_idx}: Combined QKV bias shape {qkv_bias.shape}")
            elif has_attention_bias:
                converter_logging.log(f"❌ QKV bias not found for layer {layer_idx}")
            
        if o_proj_key and o_proj_key in chkpt_vars:
            o_weight = chkpt_vars[o_proj_key].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()
            jax_weights["model"]["layers"][layer_idx]["self_attn"]["c_proj"]["weight"] = o_weight
        else:
            converter_logging.log(f"❌ O proj weight not found for layer {layer_idx}")
            
        # Q/K normalization weights (Qwen3 specific)
        if q_norm_key and q_norm_key in chkpt_vars:
            q_norm = chkpt_vars[q_norm_key].to(torch.float32).numpy().astype(CAST_DTYPE)
            jax_weights["model"]["layers"][layer_idx]["self_attn"]["q_norm"]["weight"] = q_norm
        else:
            converter_logging.log(f"❌ Q norm weight not found for layer {layer_idx}")
            
        if k_norm_key and k_norm_key in chkpt_vars:
            k_norm = chkpt_vars[k_norm_key].to(torch.float32).numpy().astype(CAST_DTYPE)
            jax_weights["model"]["layers"][layer_idx]["self_attn"]["k_norm"]["weight"] = k_norm
        else:
            converter_logging.log(f"❌ K norm weight not found for layer {layer_idx}")

    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

    # Layer norms
    converter_logging.log("Processing layer norms")
    for layer_idx in tqdm(range(base_num_decoder_layers), desc="layer norms", leave=False):
        is_moe_layer = layer_idx not in mlp_only_layers
        
        # 获取当前层的键名映射
        layer_mapping = _get_hf_to_jax_key_mapping(
            layer_idx=layer_idx, 
            has_attention_bias=has_attention_bias, 
            is_moe_layer=is_moe_layer, 
            num_experts=num_experts
        )
        
        # 查找 layer norm 键
        input_ln_key = None
        post_ln_key = None
        
        for hf_key, jax_key in layer_mapping.items():
            if hf_key.endswith(f"layers.{layer_idx}.input_layernorm.weight"):
                input_ln_key = hf_key
            elif hf_key.endswith(f"layers.{layer_idx}.post_attention_layernorm.weight"):
                post_ln_key = hf_key
        
        if input_ln_key and input_ln_key in chkpt_vars:
            input_layernorm = chkpt_vars[input_ln_key].to(torch.float32).numpy().astype(CAST_DTYPE)
            jax_weights["model"]["layers"][layer_idx]["input_layernorm"]["weight"] = input_layernorm
        else:
            converter_logging.log(f"❌ Input layernorm weight not found for layer {layer_idx}")
            
        if post_ln_key and post_ln_key in chkpt_vars:
            post_attention_layernorm = chkpt_vars[post_ln_key].to(torch.float32).numpy().astype(CAST_DTYPE)
            jax_weights["model"]["layers"][layer_idx]["post_attention_layernorm"]["weight"] = post_attention_layernorm
        else:
            converter_logging.log(f"❌ Post attention layernorm weight not found for layer {layer_idx}")

    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

    # MLP/MoE 层权重处理
    converter_logging.log("Processing MLP and MoE layer weights")
    for layer_idx in tqdm(range(base_num_decoder_layers), desc="MLP/MoE layers", leave=False):
        is_moe_layer = layer_idx not in mlp_only_layers
        
        # 获取当前层的键名映射
        layer_mapping = _get_hf_to_jax_key_mapping(
            layer_idx=layer_idx, 
            has_attention_bias=has_attention_bias, 
            is_moe_layer=is_moe_layer, 
            num_experts=num_experts
        )
        
        if is_moe_layer:
            converter_logging.log(f"Processing MoE layer {layer_idx}")
            
            # 处理路由器权重
            moe_gate_key = None
            for hf_key, jax_key in layer_mapping.items():
                if hf_key.endswith(f"layers.{layer_idx}.moe_gate.weight"):
                    moe_gate_key = hf_key
                    break
            
            if moe_gate_key and moe_gate_key in chkpt_vars:
                moe_gate_weight = chkpt_vars[moe_gate_key].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()
                jax_weights["model"]["layers"][layer_idx]["moe_gate"]["kernel"] = moe_gate_weight
                converter_logging.log(f"✅ Layer {layer_idx}: MoE gate weight shape {moe_gate_weight.shape}")
            else:
                converter_logging.log(f"❌ MoE gate weight not found for layer {layer_idx}")
            
            # 处理专家权重 - 收集所有专家的权重
            expert_gate_weights = []
            expert_up_weights = []
            expert_down_weights = []
            
            for expert_idx in range(num_experts):
                # 从映射中查找专家权重键
                gate_key = None
                up_key = None
                down_key = None
                
                for hf_key, jax_key in layer_mapping.items():
                    if hf_key.endswith(f"layers.{layer_idx}.moe.experts.{expert_idx}.gate_proj.weight"):
                        gate_key = hf_key
                    elif hf_key.endswith(f"layers.{layer_idx}.moe.experts.{expert_idx}.up_proj.weight"):
                        up_key = hf_key
                    elif hf_key.endswith(f"layers.{layer_idx}.moe.experts.{expert_idx}.down_proj.weight"):
                        down_key = hf_key
                
                if gate_key in chkpt_vars and up_key in chkpt_vars and down_key in chkpt_vars:
                    gate_weight = chkpt_vars[gate_key].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()
                    up_weight = chkpt_vars[up_key].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()
                    down_weight = chkpt_vars[down_key].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()
                    
                    expert_gate_weights.append(gate_weight)
                    expert_up_weights.append(up_weight)
                    expert_down_weights.append(down_weight)
                else:
                    converter_logging.log(f"❌ Expert {expert_idx} weights not found for layer {layer_idx}")
            
            if expert_gate_weights and expert_up_weights and expert_down_weights:
                # 堆叠所有专家权重：(num_experts, input_dim, output_dim)
                all_gate_weights = np.stack(expert_gate_weights, axis=0)
                all_up_weights = np.stack(expert_up_weights, axis=0)
                all_down_weights = np.stack(expert_down_weights, axis=0)
                
                jax_weights["model"]["layers"][layer_idx]["mlp"]["wi_0"]["value"] = all_gate_weights
                jax_weights["model"]["layers"][layer_idx]["mlp"]["wi_1"]["value"] = all_up_weights
                jax_weights["model"]["layers"][layer_idx]["mlp"]["wo"]["value"] = all_down_weights
                
                converter_logging.log(f"✅ Layer {layer_idx}: Expert weights - gate: {all_gate_weights.shape}, up: {all_up_weights.shape}, down: {all_down_weights.shape}")
            else:
                converter_logging.log(f"❌ Failed to collect expert weights for layer {layer_idx}")
                
        else:
            # 普通 MLP 层
            converter_logging.log(f"Processing regular MLP layer {layer_idx}")
            
            # 从映射中查找 MLP 权重键
            gate_proj_key = None
            up_proj_key = None
            down_proj_key = None
            
            for hf_key, jax_key in layer_mapping.items():
                if hf_key.endswith(f"layers.{layer_idx}.mlp.gate_proj.weight"):
                    gate_proj_key = hf_key
                elif hf_key.endswith(f"layers.{layer_idx}.mlp.up_proj.weight"):
                    up_proj_key = hf_key
                elif hf_key.endswith(f"layers.{layer_idx}.mlp.down_proj.weight"):
                    down_proj_key = hf_key
            
            if gate_proj_key and gate_proj_key in chkpt_vars:
                gate_proj = chkpt_vars[gate_proj_key].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()
                jax_weights["model"]["layers"][layer_idx]["mlp"]["gate_proj"]["weight"] = gate_proj
            else:
                converter_logging.log(f"❌ Gate proj weight not found for layer {layer_idx}")
                
            if up_proj_key and up_proj_key in chkpt_vars:
                up_proj = chkpt_vars[up_proj_key].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()
                jax_weights["model"]["layers"][layer_idx]["mlp"]["up_proj"]["weight"] = up_proj
            else:
                converter_logging.log(f"❌ Up proj weight not found for layer {layer_idx}")
                
            if down_proj_key and down_proj_key in chkpt_vars:
                down_proj = chkpt_vars[down_proj_key].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()
                jax_weights["model"]["layers"][layer_idx]["mlp"]["down_proj"]["weight"] = down_proj
            else:
                converter_logging.log(f"❌ Down proj weight not found for layer {layer_idx}")

    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

    del chkpt_vars
    gc.collect()
    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
    return jax_weights


def convert_to_jax_weights(base_model_path: str, model_size: str, huggingface_ckpt: bool):
    # 首先尝试从预定义字典获取参数
    if model_size in MODEL_PARAMS_DICT:
        model_params = MODEL_PARAMS_DICT[model_size]
        converter_logging.log(f"Using predefined parameters for {model_size}")
    else:
        # 尝试从配置文件推断参数
        config_path = os.path.join(base_model_path, "config.json")
        if os.path.exists(config_path):
            converter_logging.log(f"Model size {model_size} not found in predefined dict, inferring from config.json")
            model_params = _infer_model_params_from_config(config_path)
            if model_params is None:
                raise ValueError(f"Failed to infer model parameters from {config_path}")
        else:
            raise ValueError(f"Model size {model_size} not supported and no config.json found at {config_path}")
    
    mem_info = psutil.Process()
    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

    converter_logging.log(f"Loading the Qwen3 MoE base model from {base_model_path}")
    converter_logging.log(f"Model parameters: {model_params}")

    return _convert_huggingface_to_jax_weights(base_model_path, model_size, model_params, mem_info)


def save_flax_msgpack(maxtext_model_path: str, jax_weights: dict):
    converter_logging.log(f"Converting jax weights to flax msgpack")
    serialized_weights = serialization.to_bytes(jax_weights)
    
    msgpack_path = os.path.join(maxtext_model_path, "flax_model.msgpack")
    with open(msgpack_path, "wb") as f:
        f.write(serialized_weights)
    
    converter_logging.log(f"Saved Flax msgpack to {msgpack_path}")


def copy_model_config_files(base_model_path: str, maxtext_model_path: str):
    from pathlib import Path
    
    # List of files to copy
    files_to_copy = [
      'config.json',
      'tokenizer_config.json',
      'README.md',
      'tokenizer.json',
      'tokenizer_config.json',
      'vocab.json',
      'merges.txt'
    ]
    
    source_dir = Path(base_model_path)
    dest_dir = Path(maxtext_model_path)
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    missing_files = []
    
    for filename in files_to_copy:
        source_file = source_dir / filename
        dest_file = dest_dir / filename
        
        if source_file.exists():
            try:
                shutil.copy2(source_file, dest_file)
                copied_files.append(filename)
                # If it is config.json, modify it
                if filename == "config.json":
                    sed_model_config(str(dest_file))
                    continue
                converter_logging.log(f"✅ Copied {filename}")
            except Exception as e:
                converter_logging.log(f"❌ Failed to copy {filename}: {str(e)}")
        else:
            missing_files.append(filename)
            converter_logging.log(f"⚠️  Source file not found: {filename}")
    
    converter_logging.log(f"📁 File copy summary:")
    converter_logging.log(f"   Copied {len(copied_files)} files: {copied_files}")
    if missing_files:
        converter_logging.log(f"   Missing {len(missing_files)} files: {missing_files}")


def save_weights_to_checkpoint(
    base_model_path: str, maxtext_model_path: str, jax_weights: dict, device_count: int, use_ocdbt: bool, use_zarr3: bool, sv: bool
):
    """
    Function to save jax_weights ready for JAX model to a parameters checkpoint.

    Args:
        base_model_path: Path to the source model directory (for copying config files).
        maxtext_model_path: Path to save the JAX checkpoint.
        jax_weights: The JAX model weights to be saved.
        device_count: The number of simulated devices.
        use_ocdbt: Whether to use Optimized Checkpoint Database with Transactions.
        use_zarr3: Whether to use Zarr3 or not.
        save_checkpoint: Whether to save checkpoint or not.
    """
    converter_logging.log("📂 Copying model configuration files...")
    copy_model_config_files(base_model_path, maxtext_model_path)
    
    mem_info = psutil.Process()
    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
    gc.collect()
    mesh = jax.sharding.Mesh(jax.devices(), "checkpoint_sharding_axis")
    s1 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("checkpoint_sharding_axis"))  # shards first axis
    s2 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "checkpoint_sharding_axis"))  # shards second axis
    s3 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))  # no sharding

    def checkpoint_device_put(arr):
        if arr.shape[0] % device_count == 0:
            converter_logging.log("sharding first axis")
            return jax.device_put(arr, device=s1)
        elif len(arr.shape) > 1 and arr.shape[1] % device_count == 0:
            converter_logging.log("sharding second axis")
            return jax.device_put(arr, device=s2)
        else:
            converter_logging.log("no sharding was possible, replicating")
            return jax.device_put(arr, device=s3)

    # Convert all weights to jax.numpy with sharding if applicable
    jax_weights_flat, jax_weights_struct = tree.flatten(jax_weights)
    jax_weights_new = []
    while len(jax_weights_flat) > 0:
        jax_weight = jax_weights_flat.pop(0)
        jax_weights_new.append(checkpoint_device_put(jax_weight))
        del jax_weight
        gc.collect()
        logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

    jax_weights = tree.unflatten(jax_weights_struct, jax_weights_new)

    # Dummy configs for the checkpoint_manager
    step_number_to_save_new_ckpt = 0
    enable_checkpointing = True
    async_checkpointing = False
    save_interval_steps = 1

    state_new = train_state.TrainState(
        step=0, apply_fn=None, params={"params": jax_weights}, tx=None, opt_state={}  # type: ignore
    )

    # Save flax msgpack
    save_flax_msgpack(maxtext_model_path, jax_weights)

    if sv:
        checkpoint_manager = check_pointing.create_orbax_checkpoint_manager(
            maxtext_model_path,
            enable_checkpointing,
            async_checkpointing,
            save_interval_steps,
            use_ocdbt=use_ocdbt,
            use_zarr3=use_zarr3,
        )
        # Save checkpoint
        logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
        if checkpoint_manager is not None:
            if save_checkpoint.save_checkpoint(checkpoint_manager, step_number_to_save_new_ckpt, state_new):
                converter_logging.log(f"saved a checkpoint at step {step_number_to_save_new_ckpt}")
            # Upon preemption, exit when and only when all ongoing saves are complete.
            checkpoint_manager.wait_until_finished()


def analyze_model_structure(model_path: str):
    """检查 Qwen3 MoE 模型结构的辅助函数"""
    converter_logging.log("Analyzing Qwen3 MoE model structure...")
    list_safetensor_keys(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--maxtext-model-path", type=str, required=False, 
                       help="Path to save the JAX checkpoint (not required for --analyze mode)")
    parser.add_argument("--model-size", type=str, required=False, 
                       help="Model size (e.g., qwen3-30b-a3b) or any identifier if config.json is available (not required for --analyze mode)")
    parser.add_argument("--huggingface-checkpoint", type=str2bool, required=False, default=True)
    parser.add_argument("--save-checkpoint", type=str2bool, required=False, default=False)
    parser.add_argument("--use-ocdbt", type=str2bool, required=False, default=True)
    parser.add_argument("--use-zarr3", type=str2bool, required=False, default=True)
    parser.add_argument("--analyze", action="store_true", help="Analyze model structure only")
    args = parser.parse_args()

    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={SIMULATED_CPU_DEVICES_COUNT}"

    if args.analyze:
        converter_logging.log("Running in analysis mode...")
        analyze_model_structure(args.base_model_path)
    else:
        # 检查转换模式下的必需参数
        if not args.maxtext_model_path:
            parser.error("--maxtext-model-path is required when not using --analyze mode")
        if not args.model_size:
            parser.error("--model-size is required when not using --analyze mode")
            
        try:
            save_weights_to_checkpoint(
                args.base_model_path,
                args.maxtext_model_path,
                convert_to_jax_weights(args.base_model_path, args.model_size, args.huggingface_checkpoint),
                SIMULATED_CPU_DEVICES_COUNT,
                args.use_ocdbt,
                args.use_zarr3,
                args.save_checkpoint,
            )
            converter_logging.log(f"Successfully saved Qwen3 MoE weights to {args.maxtext_model_path}.")
        except Exception as e:
            converter_logging.log(f"❌ Conversion failed: {str(e)}")
            exit(1) 