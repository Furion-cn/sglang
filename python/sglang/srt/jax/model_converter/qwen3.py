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
    "qwen3-0.6b": {
        "attention_bias": False,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 151936,
        "base_emb_dim": 1024,
        "base_mlp_dim": 3072,
    },
    "qwen3-1.7b": {
        "attention_bias": False,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 151936,
        "base_emb_dim": 2048,
        "base_mlp_dim": 6144,
    },
    "qwen3-4b": {
        "attention_bias": False,
        "num_layers": 36,
        "num_heads": 32,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 151936,
        "base_emb_dim": 2560,
        "base_mlp_dim": 9728,
    },
    "qwen3-8b": {
        "attention_bias": False,
        "num_layers": 36,
        "num_heads": 32,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 151936,
        "base_emb_dim": 4096,
        "base_mlp_dim": 12288,
    },
    "qwen3-14b": {
        "attention_bias": False,
        "num_layers": 40,
        "num_heads": 40,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 151936,
        "base_emb_dim": 5120,
        "base_mlp_dim": 17408,
    },
    "qwen3-32b": {
        "attention_bias": False,
        "num_layers": 64,
        "num_heads": 64,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 151936,
        "base_emb_dim": 5120,
        "base_mlp_dim": 25600,
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
    """列出 safetensors 文件中的所有键名以分析 Qwen3 结构"""
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
                converter_logging.log(f"  {key:<60} {str(tensor.shape):<20} {tensor.dtype}")
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
        }
    except Exception as e:
        converter_logging.log(f"❌ Failed to read config file {config_path}: {str(e)}")
        return None


def _qwen3_hf_to_jax_mapping(layer_idx: int = -1, has_attention_bias: bool = False) -> dict:
    """
    Maps from Qwen3 checkpoint weights to JAX model weights.
    
    Args:
        layer_idx: The layer index of the model.
        has_attention_bias: Whether the model has attention bias.
        
    Returns:
        A dictionary mapping from Qwen3 checkpoint to JAX model weights.
    """
    mapping = {
        # Embeddings and output
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight", 
        "lm_head.weight": "lm_head.weight",
        
        # Layer-specific mappings for Qwen3
        f"model.layers.{layer_idx}.input_layernorm.weight": f"layers.{layer_idx}.input_layernorm.weight",
        f"model.layers.{layer_idx}.post_attention_layernorm.weight": f"layers.{layer_idx}.post_attention_layernorm.weight",
        
        # Qwen3 使用分离的 Q、K、V 投影，需要特殊处理合并
        f"model.layers.{layer_idx}.self_attn.q_proj.weight": f"layers.{layer_idx}.self_attn.q_proj.weight",
        f"model.layers.{layer_idx}.self_attn.k_proj.weight": f"layers.{layer_idx}.self_attn.k_proj.weight", 
        f"model.layers.{layer_idx}.self_attn.v_proj.weight": f"layers.{layer_idx}.self_attn.v_proj.weight",
        f"model.layers.{layer_idx}.self_attn.o_proj.weight": f"layers.{layer_idx}.self_attn.o_proj.weight",
        
        # Q/K normalization (Qwen3 specific)
        f"model.layers.{layer_idx}.self_attn.q_norm.weight": f"layers.{layer_idx}.self_attn.q_norm.weight",
        f"model.layers.{layer_idx}.self_attn.k_norm.weight": f"layers.{layer_idx}.self_attn.k_norm.weight",
        
        # MLP weights - Qwen3 uses gate_proj, up_proj, down_proj
        f"model.layers.{layer_idx}.mlp.gate_proj.weight": f"layers.{layer_idx}.mlp.gate_proj.weight",
        f"model.layers.{layer_idx}.mlp.up_proj.weight": f"layers.{layer_idx}.mlp.up_proj.weight", 
        f"model.layers.{layer_idx}.mlp.down_proj.weight": f"layers.{layer_idx}.mlp.down_proj.weight",
    }
    
    # 只有当模型有 bias 时才添加 bias 映射
    if has_attention_bias:
        mapping.update({
            f"model.layers.{layer_idx}.self_attn.q_proj.bias": f"layers.{layer_idx}.self_attn.q_proj.bias",
            f"model.layers.{layer_idx}.self_attn.k_proj.bias": f"layers.{layer_idx}.self_attn.k_proj.bias",
            f"model.layers.{layer_idx}.self_attn.v_proj.bias": f"layers.{layer_idx}.self_attn.v_proj.bias",
        })
    
    return mapping


def _convert_huggingface_to_jax_weights(base_model_path: str, model_size: str, model_params: dict, mem_info: psutil.Process):
    """Convert a Huggingface Qwen3 Checkpoint to a dictionary of Numpy arrays representing the weights.

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

    converter_logging.log(f"Loading the Qwen3 model from {base_model_path}")
    
    # 首先列出所有权重键名以便调试
    converter_logging.log("Analyzing Qwen3 model structure...")
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
                    
                # 提取层索引
                parts = key.split(".")
                layer = 0
                if "model.layers." in key and len(parts) >= 3:
                    try:
                        layer = int(parts[2])  # model.layers.{layer_idx}.*
                    except (IndexError, ValueError):
                        converter_logging.log(f"Failed to extract layer index from {key}")
                        continue
                        
                try:
                    mapping = _qwen3_hf_to_jax_mapping(layer, has_attention_bias)
                    if key in mapping:
                        mapped_key = mapping[key]
                        chkpt_vars[mapped_key] = f.get_tensor(key)
                    else:
                        converter_logging.log(f"⚠️  No mapping found for key: {key}")
                except Exception as e:
                    converter_logging.log(f"❌ Error processing key {key}: {str(e)}")

    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

    # Initialize the data structure for storing jax_weights to match Qwen3ForCausalLM structure
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
        # 动态构建 attention 权重结构
        attn_structure = {
            "q_proj": {"weight": None},
            "k_proj": {"weight": None},
            "v_proj": {"weight": None},
            "o_proj": {"weight": None},
            "q_norm": {"weight": None},
            "k_norm": {"weight": None},
        }
        
        # 只有当模型有 attention bias 时才添加 bias 字段
        if has_attention_bias:
            attn_structure["qkv_proj"]["bias"] = None
            attn_structure["o_proj"]["bias"] = None
            
        jax_weights["model"]["layers"][layer_idx] = {
            "self_attn": attn_structure,
            "input_layernorm": {"weight": None},
            "post_attention_layernorm": {"weight": None},
            "mlp": {
                "gate_proj": {"weight": None},
                "up_proj": {"weight": None},
                "down_proj": {"weight": None},
            },
        }

    # Self attention - 处理分离的 Q、K、V 权重
    converter_logging.log("Processing self attention")
    for layer_idx in tqdm(range(base_num_decoder_layers), desc="attention layers", leave=False):
        # Qwen3 使用分离的 Q、K、V 投影
        q_proj_key = f"layers.{layer_idx}.self_attn.q_proj.weight"
        k_proj_key = f"layers.{layer_idx}.self_attn.k_proj.weight"
        v_proj_key = f"layers.{layer_idx}.self_attn.v_proj.weight"
        o_proj_key = f"layers.{layer_idx}.self_attn.o_proj.weight"
        q_norm_key = f"layers.{layer_idx}.self_attn.q_norm.weight"
        k_norm_key = f"layers.{layer_idx}.self_attn.k_norm.weight"
        
        # 检查并合并 Q、K、V 权重
        if q_proj_key in chkpt_vars and k_proj_key in chkpt_vars and v_proj_key in chkpt_vars:
            # 获取原始权重 [output_dim, input_dim]
            q_weight = chkpt_vars[q_proj_key].to(torch.float32).numpy().astype(CAST_DTYPE)  # [num_heads*head_dim, hidden_size]
            k_weight = chkpt_vars[k_proj_key].to(torch.float32).numpy().astype(CAST_DTYPE)  # [num_kv_heads*head_dim, hidden_size]
            v_weight = chkpt_vars[v_proj_key].to(torch.float32).numpy().astype(CAST_DTYPE)  # [num_kv_heads*head_dim, hidden_size]
            
            jax_q_weight = q_weight.transpose()
            jax_k_weight = k_weight.transpose()
            jax_v_weight = v_weight.transpose()
            
            jax_weights["model"]["layers"][layer_idx]["self_attn"]["q_proj"]["weight"] = jax_q_weight
            jax_weights["model"]["layers"][layer_idx]["self_attn"]["k_proj"]["weight"] = jax_k_weight
            jax_weights["model"]["layers"][layer_idx]["self_attn"]["v_proj"]["weight"] = jax_v_weight
            converter_logging.log(f"✅ Layer {layer_idx}: Combined QKV weight shape {jax_q_weight.shape}")
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
            q_bias_key = f"layers.{layer_idx}.self_attn.q_proj.bias"
            k_bias_key = f"layers.{layer_idx}.self_attn.k_proj.bias"
            v_bias_key = f"layers.{layer_idx}.self_attn.v_proj.bias"
            o_bias_key = f"layers.{layer_idx}.self_attn.o_proj.bias"
            
            if q_bias_key in chkpt_vars and k_bias_key in chkpt_vars and v_bias_key in chkpt_vars:
                q_bias = chkpt_vars[q_bias_key].to(torch.float32).numpy().astype(CAST_DTYPE)
                k_bias = chkpt_vars[k_bias_key].to(torch.float32).numpy().astype(CAST_DTYPE)
                v_bias = chkpt_vars[v_bias_key].to(torch.float32).numpy().astype(CAST_DTYPE)
                
                # 合并 bias
                qkv_bias = np.concatenate([q_bias, k_bias, v_bias], axis=0)
                jax_weights["model"]["layers"][layer_idx]["self_attn"]["qkv_proj"]["bias"] = qkv_bias
                converter_logging.log(f"✅ Layer {layer_idx}: Combined QKV bias shape {qkv_bias.shape}")
            elif has_attention_bias:
                converter_logging.log(f"❌ QKV bias not found for layer {layer_idx}")
            
        if o_proj_key in chkpt_vars:
            o_weight = chkpt_vars[o_proj_key].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()
            jax_weights["model"]["layers"][layer_idx]["self_attn"]["o_proj"]["weight"] = o_weight
        else:
            converter_logging.log(f"❌ O proj weight not found for layer {layer_idx}")
            
        # Q/K normalization weights (Qwen3 specific)
        if q_norm_key in chkpt_vars:
            q_norm = chkpt_vars[q_norm_key].to(torch.float32).numpy().astype(CAST_DTYPE)
            jax_weights["model"]["layers"][layer_idx]["self_attn"]["q_norm"]["weight"] = q_norm
        else:
            converter_logging.log(f"❌ Q norm weight not found for layer {layer_idx}")
            
        if k_norm_key in chkpt_vars:
            k_norm = chkpt_vars[k_norm_key].to(torch.float32).numpy().astype(CAST_DTYPE)
            jax_weights["model"]["layers"][layer_idx]["self_attn"]["k_norm"]["weight"] = k_norm
        else:
            converter_logging.log(f"❌ K norm weight not found for layer {layer_idx}")

    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

    # Layer norms
    converter_logging.log("Processing layer norms")
    for layer_idx in tqdm(range(base_num_decoder_layers), desc="layer norms", leave=False):
        input_ln_key = f"layers.{layer_idx}.input_layernorm.weight"
        post_ln_key = f"layers.{layer_idx}.post_attention_layernorm.weight"
        
        if input_ln_key in chkpt_vars:
            input_layernorm = chkpt_vars[input_ln_key].to(torch.float32).numpy().astype(CAST_DTYPE)
            jax_weights["model"]["layers"][layer_idx]["input_layernorm"]["weight"] = input_layernorm
        else:
            converter_logging.log(f"❌ Input layernorm weight not found for layer {layer_idx}")
            
        if post_ln_key in chkpt_vars:
            post_attention_layernorm = chkpt_vars[post_ln_key].to(torch.float32).numpy().astype(CAST_DTYPE)
            jax_weights["model"]["layers"][layer_idx]["post_attention_layernorm"]["weight"] = post_attention_layernorm
        else:
            converter_logging.log(f"❌ Post attention layernorm weight not found for layer {layer_idx}")

    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

    # MLP weights
    converter_logging.log("Processing MLP layer weights")
    for layer_idx in tqdm(range(base_num_decoder_layers), desc="MLP layers", leave=False):
        gate_proj_key = f"layers.{layer_idx}.mlp.gate_proj.weight"
        up_proj_key = f"layers.{layer_idx}.mlp.up_proj.weight"
        down_proj_key = f"layers.{layer_idx}.mlp.down_proj.weight"
        
        if gate_proj_key in chkpt_vars:
            gate_proj = chkpt_vars[gate_proj_key].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()
            jax_weights["model"]["layers"][layer_idx]["mlp"]["gate_proj"]["weight"] = gate_proj
        else:
            converter_logging.log(f"❌ Gate proj weight not found for layer {layer_idx}")
            
        if up_proj_key in chkpt_vars:
            up_proj = chkpt_vars[up_proj_key].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()
            jax_weights["model"]["layers"][layer_idx]["mlp"]["up_proj"]["weight"] = up_proj
        else:
            converter_logging.log(f"❌ Up proj weight not found for layer {layer_idx}")
            
        if down_proj_key in chkpt_vars:
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

    converter_logging.log(f"Loading the Qwen3 base model from {base_model_path}")
    converter_logging.log(f"Model parameters: {model_params}")

    return _convert_huggingface_to_jax_weights(base_model_path, model_size, model_params, mem_info)


def save_flax_msgpack(maxtext_model_path: str, jax_weights: dict):
    converter_logging.log(f"Converting jax weights to flax msgpack")
    serialized_weights = serialization.to_bytes(jax_weights)
    
    msgpack_path = os.path.join(maxtext_model_path, "flax_model.msgpack")
    with open(msgpack_path, "wb") as f:
        f.write(serialized_weights)
    
    converter_logging.log(f"Saved Flax msgpack to {msgpack_path}")


def load_flax_msgpack(msgpack_path: str) -> dict:
    converter_logging.log(f"Loading weights from {msgpack_path}")
    with open(msgpack_path, "rb") as f:
        serialized_weights = f.read()
    
    jax_weights = serialization.from_bytes(None, serialized_weights)
    converter_logging.log(f"Successfully loaded weights from {msgpack_path}")
    return jax_weights


def load_pytorch_weights(base_model_path: str) -> dict:
    converter_logging.log(f"Loading original PyTorch weights from {base_model_path}")
    ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors"))
    pytorch_weights = {}
    
    for i, ckpt_path in enumerate(ckpt_paths):
        converter_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
        
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                pytorch_weights[key] = f.get_tensor(key).to(torch.float32).numpy()
    
    converter_logging.log(f"Successfully loaded {len(pytorch_weights)} PyTorch weights")
    return pytorch_weights


def compare_weights(original_weights: dict, converted_weights: dict, model_params: dict, tolerance: float = 1e-5) -> bool:
    converter_logging.log("Starting PyTorch vs JAX weight comparison...")
    
    def compare_arrays(arr1, arr2, path: str, tolerance: float) -> bool:
        if arr1.shape != arr2.shape:
            converter_logging.log(f"❌ Shape mismatch at {path}: {arr1.shape} vs {arr2.shape}")
            return False
        
        if hasattr(arr1, 'device'):
            arr1 = np.array(arr1)
        if hasattr(arr2, 'device'):
            arr2 = np.array(arr2)
        
        if arr1.dtype != arr2.dtype:
            if arr1.dtype == np.float32 and arr2.dtype == ml_dtypes.bfloat16:
                arr2 = arr2.astype(np.float32)
                converter_logging.log(f"Converting {path} from bfloat16 to float32 for comparison")
            elif arr1.dtype == ml_dtypes.bfloat16 and arr2.dtype == np.float32:
                arr1 = arr1.astype(np.float32)
                converter_logging.log(f"Converting {path} from bfloat16 to float32 for comparison")
            else:
                converter_logging.log(f"❌ Dtype mismatch at {path}: {arr1.dtype} vs {arr2.dtype}")
                return False
        
        max_diff = np.max(np.abs(arr1 - arr2))
        if hasattr(max_diff, 'item'):
            max_diff = max_diff.item()
        
        if max_diff > tolerance:
            converter_logging.log(f"❌ Value mismatch at {path}: max difference = {max_diff} (tolerance = {tolerance})")
            return False
        
        converter_logging.log(f"✅ {path}: shapes {arr1.shape}, max_diff = {max_diff:.2e}")
        return True
    
    def get_jax_weight_by_pytorch_key(pytorch_key: str, jax_weights: dict):
        layer = 0
        if "model.layers." in pytorch_key:
            parts = pytorch_key.split(".")
            try:
                layer = int(parts[2])  # model.layers.{layer_idx}.*
            except (IndexError, ValueError):
                converter_logging.log(f"Failed to extract layer index from {pytorch_key}")
                return None
        
        try:
            mapping = _qwen3_hf_to_jax_mapping(layer, model_params["attention_bias"])
            if pytorch_key not in mapping:
                converter_logging.log(f"PyTorch key {pytorch_key} not found in mapping for layer {layer}")
                return None
                
            jax_key = mapping[pytorch_key]
            converter_logging.log(f"Mapping {pytorch_key} -> {jax_key}")
            
            if jax_key == "model.embed_tokens.weight":
                return jax_weights.get("model", {}).get("embed_tokens", {}).get("embedding")
            elif jax_key == "model.norm.weight":
                return jax_weights.get("model", {}).get("norm", {}).get("weight")
            elif jax_key == "lm_head.weight":
                return jax_weights.get("lm_head", {}).get("embedding")
            elif jax_key.startswith(f"layers.{layer}."):
                layers_dict = jax_weights.get("model", {}).get("layers", {})
                layer_str = str(layer)
                if layer_str not in layers_dict:
                    converter_logging.log(f"Layer {layer_str} not found in JAX weights. Available layers: {list(layers_dict.keys())}")
                    return None
                layer_weights = layers_dict[layer_str]
                
                if "input_layernorm.weight" in jax_key:
                    return layer_weights.get("input_layernorm", {}).get("weight")
                elif "post_attention_layernorm.weight" in jax_key:
                    return layer_weights.get("post_attention_layernorm", {}).get("weight")
                    
                # 处理分离的 Q、K、V 权重 - 从 qkv_proj 中拆分
                elif jax_key.endswith("self_attn.q_proj.weight"):
                    return _extract_q_weight_from_qkv(layer_weights, model_params)
                elif jax_key.endswith("self_attn.k_proj.weight"):
                    return _extract_k_weight_from_qkv(layer_weights, model_params)
                elif jax_key.endswith("self_attn.v_proj.weight"):
                    return _extract_v_weight_from_qkv(layer_weights, model_params)
                elif jax_key.endswith("self_attn.q_proj.bias"):
                    return _extract_q_bias_from_qkv(layer_weights, model_params)
                elif jax_key.endswith("self_attn.k_proj.bias"):
                    return _extract_k_bias_from_qkv(layer_weights, model_params)
                elif jax_key.endswith("self_attn.v_proj.bias"):
                    return _extract_v_bias_from_qkv(layer_weights, model_params)
                    
                elif "self_attn.o_proj.weight" in jax_key:
                    jax_weight = layer_weights.get("self_attn", {}).get("o_proj", {}).get("weight")
                    if jax_weight is not None:
                        converter_logging.log(f"Applying transpose for layer {layer} o_proj weight comparison")
                        return jax_weight.transpose()
                    return jax_weight
                elif "self_attn.q_norm.weight" in jax_key:
                    return layer_weights.get("self_attn", {}).get("q_norm", {}).get("weight")
                elif "self_attn.k_norm.weight" in jax_key:
                    return layer_weights.get("self_attn", {}).get("k_norm", {}).get("weight")
                elif "mlp.gate_proj.weight" in jax_key:
                    jax_weight = layer_weights.get("mlp", {}).get("gate_proj", {}).get("weight")
                    if jax_weight is not None:
                        converter_logging.log(f"Applying transpose for layer {layer} gate_proj weight comparison")
                        return jax_weight.transpose()
                    return jax_weight
                elif "mlp.up_proj.weight" in jax_key:
                    jax_weight = layer_weights.get("mlp", {}).get("up_proj", {}).get("weight")
                    if jax_weight is not None:
                        converter_logging.log(f"Applying transpose for layer {layer} up_proj weight comparison")
                        return jax_weight.transpose()
                    return jax_weight
                elif "mlp.down_proj.weight" in jax_key:
                    jax_weight = layer_weights.get("mlp", {}).get("down_proj", {}).get("weight")
                    if jax_weight is not None:
                        converter_logging.log(f"Applying transpose for layer {layer} down_proj weight comparison")
                        return jax_weight.transpose()
                    return jax_weight
            
            converter_logging.log(f"No mapping found for JAX key: {jax_key}")
            return None
        except Exception as e:
            converter_logging.log(f"Error mapping {pytorch_key}: {str(e)}")
            return None
    
    matched_count = 0
    total_count = len(original_weights)
    
    for pytorch_key, pytorch_weight in original_weights.items():
        jax_weight = get_jax_weight_by_pytorch_key(pytorch_key, converted_weights)
        
        if jax_weight is None:
            converter_logging.log(f"❌ No corresponding JAX weight found for PyTorch key: {pytorch_key}")
            continue
        
        if compare_arrays(pytorch_weight, jax_weight, pytorch_key, tolerance):
            matched_count += 1
    
    converter_logging.log(f"Matched {matched_count}/{total_count} weights")
    
    if matched_count == total_count:
        converter_logging.log("✅ Weight comparison PASSED: All weights match within tolerance")
        return True
    else:
        converter_logging.log("❌ Weight comparison FAILED: Some weights could not be matched")
        return False


def verify_conversion(base_model_path: str, model_size: str, maxtext_model_path: str, huggingface_ckpt: bool = True, tolerance: float = 1e-2) -> bool:
    converter_logging.log("Starting conversion verification...")
    
    converter_logging.log("Loading original PyTorch weights...")
    original_weights = load_pytorch_weights(base_model_path)
    
    msgpack_path = os.path.join(maxtext_model_path, "flax_model.msgpack")
    if not os.path.exists(msgpack_path):
        converter_logging.log(f"❌ Msgpack file not found: {msgpack_path}")
        return False
    
    converter_logging.log("Loading converted weights from msgpack...")
    converted_weights = load_flax_msgpack(msgpack_path)
    
    return compare_weights(original_weights, converted_weights, MODEL_PARAMS_DICT[model_size], tolerance)

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
    """检查 Qwen3 模型结构的辅助函数"""
    converter_logging.log("Analyzing Qwen3 model structure...")
    list_safetensor_keys(model_path)


def _extract_q_weight_from_qkv(layer_weights: dict, model_params: dict):
    """从合并的 qkv_proj 中提取 Q 权重"""
    qkv_weight = layer_weights.get("self_attn", {}).get("qkv_proj", {}).get("weight")
    if qkv_weight is None:
        return None
    
    num_heads = model_params["num_heads"]
    head_dim = model_params["dims_per_head"]
    q_size = num_heads * head_dim
    
    # qkv_weight 形状: [hidden_size, total_proj_dim]
    # 需要转置回 [total_proj_dim, hidden_size] 然后拆分
    qkv_transposed = qkv_weight.transpose()  # [total_proj_dim, hidden_size]
    
    # Q 权重是前 q_size 行
    q_weight = qkv_transposed[:q_size, :]  # [q_size, hidden_size]
    
    converter_logging.log(f"Extracted Q weight shape: {q_weight.shape}")
    return q_weight


def _extract_k_weight_from_qkv(layer_weights: dict, model_params: dict):
    """从合并的 qkv_proj 中提取 K 权重"""
    qkv_weight = layer_weights.get("self_attn", {}).get("qkv_proj", {}).get("weight")
    if qkv_weight is None:
        return None
    
    num_heads = model_params["num_heads"]
    num_kv_heads = model_params["num_kv_heads"]
    head_dim = model_params["dims_per_head"]
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    
    # qkv_weight 形状: [hidden_size, total_proj_dim]
    qkv_transposed = qkv_weight.transpose()  # [total_proj_dim, hidden_size]
    
    # K 权重是从 q_size 开始的 kv_size 行
    k_weight = qkv_transposed[q_size:q_size+kv_size, :]  # [kv_size, hidden_size]
    
    converter_logging.log(f"Extracted K weight shape: {k_weight.shape}")
    return k_weight


def _extract_v_weight_from_qkv(layer_weights: dict, model_params: dict):
    """从合并的 qkv_proj 中提取 V 权重"""
    qkv_weight = layer_weights.get("self_attn", {}).get("qkv_proj", {}).get("weight")
    if qkv_weight is None:
        return None
    
    num_heads = model_params["num_heads"]
    num_kv_heads = model_params["num_kv_heads"]
    head_dim = model_params["dims_per_head"]
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    
    # qkv_weight 形状: [hidden_size, total_proj_dim]
    qkv_transposed = qkv_weight.transpose()  # [total_proj_dim, hidden_size]
    
    # V 权重是从 q_size + kv_size 开始的 kv_size 行
    v_weight = qkv_transposed[q_size+kv_size:q_size+2*kv_size, :]  # [kv_size, hidden_size]
    
    converter_logging.log(f"Extracted V weight shape: {v_weight.shape}")
    return v_weight


def _extract_q_bias_from_qkv(layer_weights: dict, model_params: dict):
    """从合并的 qkv_proj 中提取 Q bias"""
    qkv_bias = layer_weights.get("self_attn", {}).get("qkv_proj", {}).get("bias")
    if qkv_bias is None:
        return None
    
    num_heads = model_params["num_heads"]
    head_dim = model_params["dims_per_head"]
    q_size = num_heads * head_dim
    
    # Q bias 是前 q_size 个元素
    q_bias = qkv_bias[:q_size]
    
    converter_logging.log(f"Extracted Q bias shape: {q_bias.shape}")
    return q_bias


def _extract_k_bias_from_qkv(layer_weights: dict, model_params: dict):
    """从合并的 qkv_proj 中提取 K bias"""
    qkv_bias = layer_weights.get("self_attn", {}).get("qkv_proj", {}).get("bias")
    if qkv_bias is None:
        return None
    
    num_heads = model_params["num_heads"]
    num_kv_heads = model_params["num_kv_heads"]
    head_dim = model_params["dims_per_head"]
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    
    # K bias 是从 q_size 开始的 kv_size 个元素
    k_bias = qkv_bias[q_size:q_size+kv_size]
    
    converter_logging.log(f"Extracted K bias shape: {k_bias.shape}")
    return k_bias


def _extract_v_bias_from_qkv(layer_weights: dict, model_params: dict):
    """从合并的 qkv_proj 中提取 V bias"""
    qkv_bias = layer_weights.get("self_attn", {}).get("qkv_proj", {}).get("bias")
    if qkv_bias is None:
        return None
    
    num_heads = model_params["num_heads"]
    num_kv_heads = model_params["num_kv_heads"]
    head_dim = model_params["dims_per_head"]
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    
    # V bias 是从 q_size + kv_size 开始的 kv_size 个元素
    v_bias = qkv_bias[q_size+kv_size:q_size+2*kv_size]
    
    converter_logging.log(f"Extracted V bias shape: {v_bias.shape}")
    return v_bias


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--maxtext-model-path", type=str, required=True)
    parser.add_argument("--model-size", type=str, required=True, 
                       help="Model size (e.g., qwen3-0.6b, qwen3-8b) or any identifier if config.json is available")
    parser.add_argument("--huggingface-checkpoint", type=str2bool, required=False, default=True)
    parser.add_argument("--save-checkpoint", type=str2bool, required=False, default=False)
    parser.add_argument("--use-ocdbt", type=str2bool, required=False, default=True)
    parser.add_argument("--use-zarr3", type=str2bool, required=False, default=True)
    parser.add_argument("--check", action="store_true", help="Verify conversion by comparing original and converted weights")
    parser.add_argument("--analyze", action="store_true", help="Analyze model structure only")
    args = parser.parse_args()

    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={SIMULATED_CPU_DEVICES_COUNT}"

    if args.analyze:
        converter_logging.log("Running in analysis mode...")
        analyze_model_structure(args.base_model_path)
    elif args.check:
        converter_logging.log("Running in verification mode...")
        success = verify_conversion(
            args.base_model_path,
            args.model_size,
            args.maxtext_model_path,
            args.huggingface_checkpoint
        )
        if success:
            converter_logging.log("✅ Conversion verification PASSED")
            exit(0)
        else:
            converter_logging.log("❌ Conversion verification FAILED")
            exit(1)
    else:
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
            converter_logging.log(f"Successfully saved Qwen3 weights to {args.maxtext_model_path}.")
        except Exception as e:
            converter_logging.log(f"❌ Conversion failed: {str(e)}")
            exit(1)
