# pylint: disable=g-line-too-long
import argparse
import pathlib
import os
import gc
import logging
import shutil

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
from sglang.srt.jax.model_converter.utils import str2bool
from sglang.srt.jax.model_converter import save_checkpoint

MODEL_PARAMS_DICT = {
    "qwen-7b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,
        "dims_per_head": 128,
        "vocab": 151936,
        "base_emb_dim": 4096,
        "base_mlp_dim": 22016,
    },
}

SIMULATED_CPU_DEVICES_COUNT = 16

# NOTE: numpy doesn't have native support for bfloat16, so
# we'll use ml_dtypes instead (which is quasi native)
# NOTE: it's incredibly silly but you can't directly cast from
# a torch tensor of type bfloat16 to a numpy array of type bfloat16
# so we have to cast to float32 first
CAST_DTYPE = ml_dtypes.bfloat16


def _qwen_hf_to_jax_mapping(layer_idx: int = -1) -> dict:
  """
  Maps from Qwen checkpoint weights to MaxText model weights.
  
  Args:
    layer_idx: The layer index of the model.
    
  Returns:
    A dictionary mapping from Qwen checkpoint to MaxText model weights.
  """
  # pylint: disable=line-too-long
  return {
      # Embeddings and output
      "transformer.wte.weight": "model.embed_tokens.weight",
      "transformer.ln_f.weight": "model.norm.weight",
      "lm_head.weight": "lm_head.weight",
      
      # Layer-specific mappings
      f"transformer.h.{layer_idx}.ln_1.weight": f"layers.{layer_idx}.attention_norm.weight",
      f"transformer.h.{layer_idx}.ln_2.weight": f"layers.{layer_idx}.ffn_norm.weight",
      
      # Attention weights - Qwen uses c_attn for combined QKV
      f"transformer.h.{layer_idx}.attn.c_attn.weight": f"layers.{layer_idx}.attention.wqkv.weight",
      f"transformer.h.{layer_idx}.attn.c_attn.bias": f"layers.{layer_idx}.attention.wqkv.bias",
      f"transformer.h.{layer_idx}.attn.c_proj.weight": f"layers.{layer_idx}.attention.wo.weight",
      
      # MLP weights
      f"transformer.h.{layer_idx}.mlp.w1.weight": f"layers.{layer_idx}.feed_forward.w1.weight",
      f"transformer.h.{layer_idx}.mlp.w2.weight": f"layers.{layer_idx}.feed_forward.w2.weight",
      f"transformer.h.{layer_idx}.mlp.c_proj.weight": f"layers.{layer_idx}.feed_forward.c_proj.weight",
  }

def _convert_huggingface_to_jax_weights(base_model_path: str, model_size: str, model_params: dict, mem_info: psutil.Process):
  """Convert a Huggingface Checkpoint to a dictionary of Numpy arrays representing the weights.

  Args:
    base_model_path (str): Path to the base model checkpoint.
    model_size (str): Size of the base model.
    model_params (dict): Dictionary containing model parameters.
    mem_info (psutil.Process): Process object to track memory usage.

  Returns:
    jax_weights (dict): Dictionary containing the converted weights.
  """
  base_num_decoder_layers = model_params["num_layers"]

  converter_logging.log(f"Loading the base model from {base_model_path}")
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors"))
  chkpt_vars = {}
  for i, ckpt_path in enumerate(ckpt_paths):
    converter_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")

    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        parts = key.split(".")
        if "transformer.h." in key and len(parts) >= 3:
          layer = int(parts[2])
        else:
          layer = 0
        mapped_key = _qwen_hf_to_jax_mapping(layer)[key]
        chkpt_vars[mapped_key] = f.get_tensor(key)


  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # initialize the data structure for storing jax_weights to match QWenLMHeadJaxModel structure
  jax_weights = {
      "transformer": {
          "embed_tokens": {"embedding": None},
          "h": {},
          "ln_f": {"weight": None},
      },
      "lm_head": {"embedding": None},
      "logits_processor": {},
  }

  # final layer norm scale ###########################################
  converter_logging.log("Processing final layer norm scale")
  ln_f_scale = chkpt_vars["model.norm.weight"].to(torch.float32).numpy().astype(CAST_DTYPE)
  jax_weights["transformer"]["ln_f"]["weight"] = ln_f_scale

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # logits dense #################################################
  converter_logging.log("Processing logits dense")
  
  lm_head_weight = chkpt_vars["lm_head.weight"].to(torch.float32).numpy().astype(CAST_DTYPE)
  
  jax_weights["lm_head"]["embedding"] = lm_head_weight

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # token embedding ##############################################
  converter_logging.log("Processing token embeddings")

  jax_weights["transformer"]["embed_tokens"]["embedding"] = (
      chkpt_vars["model.embed_tokens.weight"].to(torch.float32).numpy().astype(CAST_DTYPE)
  )

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # Initialize layer structure
  for layer_idx in range(base_num_decoder_layers):
    jax_weights["transformer"]["h"][layer_idx] = {
        "attn": {
            "c_attn": {"weight": None, "bias": None},
            "c_proj": {"weight": None},
        },
        "ln_1": {"weight": None},
        "ln_2": {"weight": None},
        "mlp": {
            "w1": {"kernel": None},
            "w2": {"kernel": None},
            "c_proj": {"kernel": None},
        },
    }

  # self attention ###############################################
  converter_logging.log("Processing self attention")
  for layer_idx in tqdm(range(base_num_decoder_layers), desc="layers", leave=False):
    wqkv = chkpt_vars[f"layers.{layer_idx}.attention.wqkv.weight"].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()
    bqkv = chkpt_vars[f"layers.{layer_idx}.attention.wqkv.bias"].to(torch.float32).numpy().astype(CAST_DTYPE)
    w_post = chkpt_vars[f"layers.{layer_idx}.attention.wo.weight"].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()
    
    jax_weights["transformer"]["h"][layer_idx]["attn"]["c_attn"]["weight"] = wqkv
    jax_weights["transformer"]["h"][layer_idx]["attn"]["c_attn"]["bias"] = bqkv
    jax_weights["transformer"]["h"][layer_idx]["attn"]["c_proj"]["weight"] = w_post

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  converter_logging.log("Processing pre and post self attention norms")
  
  for layer_idx in tqdm(range(base_num_decoder_layers), desc="layer norms", leave=False):
    pre_self_attention_layernorm = (
        chkpt_vars[f"layers.{layer_idx}.attention_norm.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
    )
    post_self_attention_layernorm = (
        chkpt_vars[f"layers.{layer_idx}.ffn_norm.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
    )
    
    jax_weights["transformer"]["h"][layer_idx]["ln_1"]["weight"] = pre_self_attention_layernorm
    jax_weights["transformer"]["h"][layer_idx]["ln_2"]["weight"] = post_self_attention_layernorm
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # layer weights ################################################
  converter_logging.log("Processing MLP layer weights")
  
  for layer_idx in tqdm(range(base_num_decoder_layers), desc="MLP layers", leave=False):
    w1 = (
        chkpt_vars[f"layers.{layer_idx}.feed_forward.w1.weight"].type(torch.float32).numpy().astype(CAST_DTYPE).transpose()
    )
    w2 = (
        chkpt_vars[f"layers.{layer_idx}.feed_forward.w2.weight"].type(torch.float32).numpy().astype(CAST_DTYPE).transpose()
    )
    c_proj = (
        chkpt_vars[f"layers.{layer_idx}.feed_forward.c_proj.weight"].type(torch.float32).numpy().astype(CAST_DTYPE).transpose()
    )
    
    jax_weights["transformer"]["h"][layer_idx]["mlp"]["w1"]["weight"] = w1
    jax_weights["transformer"]["h"][layer_idx]["mlp"]["w2"]["weight"] = w2
    jax_weights["transformer"]["h"][layer_idx]["mlp"]["c_proj"]["weight"] = c_proj
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  del chkpt_vars
  gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  return jax_weights

def convert_to_jax_weights(base_model_path: str, model_size: str, huggingface_ckpt: bool):
  model_params = MODEL_PARAMS_DICT[model_size]
  mem_info = psutil.Process()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  converter_logging.log(f"Loading the base model from {base_model_path}")

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

def compare_weights(original_weights: dict, converted_weights: dict, tolerance: float = 1e-5) -> bool:
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
    if "transformer.h." in pytorch_key:
      parts = pytorch_key.split(".")
      try:
        layer = int(parts[2])  # transformer.h.{layer_idx}.*
      except (IndexError, ValueError):
        converter_logging.log(f"Failed to extract layer index from {pytorch_key}")
        return None
    
    try:
      mapping = _qwen_hf_to_jax_mapping(layer)
      if pytorch_key not in mapping:
        converter_logging.log(f"PyTorch key {pytorch_key} not found in mapping for layer {layer}")
        return None
        
      jax_key = mapping[pytorch_key]
      converter_logging.log(f"Mapping {pytorch_key} -> {jax_key}")
      
      if jax_key == "model.embed_tokens.weight":
        return jax_weights.get("transformer", {}).get("embed_tokens", {}).get("embedding")
      elif jax_key == "model.norm.weight":
        return jax_weights.get("transformer", {}).get("ln_f", {}).get("weight")
      elif jax_key == "lm_head.weight":
        return jax_weights.get("lm_head", {}).get("embedding")
      elif jax_key.startswith(f"layers.{layer}."):
        h_dict = jax_weights.get("transformer", {}).get("h", {})
        layer_str = str(layer)
        if layer_str not in h_dict:
          converter_logging.log(f"Layer {layer_str} not found in JAX weights. Available layers: {list(h_dict.keys())}")
          return None
        layer_weights = h_dict[layer_str]
          
        if "attention_norm.weight" in jax_key:
          return layer_weights.get("ln_1", {}).get("weight")
        elif "ffn_norm.weight" in jax_key:
          return layer_weights.get("ln_2", {}).get("weight")
        elif "attention.wqkv.weight" in jax_key:
          jax_weight = layer_weights.get("attn", {}).get("c_attn", {}).get("weight")
          if jax_weight is not None:
            converter_logging.log(f"Applying inverse RoPE permutation and transpose for layer {layer} c_attn weight comparison")
            
            return jax_weight.transpose()
          return jax_weight
        elif "attention.wqkv.bias" in jax_key:
          return layer_weights.get("attn", {}).get("c_attn", {}).get("bias")
        elif "attention.wo.weight" in jax_key:
          jax_weight = layer_weights.get("attn", {}).get("c_proj", {}).get("weight")
          if jax_weight is not None:
            converter_logging.log(f"Applying transpose for layer {layer} c_proj weight comparison")
            return jax_weight.transpose()
          return jax_weight
        elif "feed_forward.w1.weight" in jax_key:
          jax_weight = layer_weights.get("mlp", {}).get("w1", {}).get("kernel")
          if jax_weight is not None:
            converter_logging.log(f"Applying transpose for layer {layer} w1 weight comparison")
            return jax_weight.transpose()
          return jax_weight
        elif "feed_forward.w2.weight" in jax_key:
          jax_weight = layer_weights.get("mlp", {}).get("w2", {}).get("kernel")
          if jax_weight is not None:
            converter_logging.log(f"Applying transpose for layer {layer} w2 weight comparison")
            return jax_weight.transpose()
          return jax_weight
        elif "feed_forward.c_proj.weight" in jax_key:
          jax_weight = layer_weights.get("mlp", {}).get("c_proj", {}).get("kernel")
          if jax_weight is not None:
            converter_logging.log(f"Applying transpose for layer {layer} c_proj weight comparison")
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
  
  return compare_weights(original_weights, converted_weights, tolerance)

def copy_model_config_files(base_model_path: str, maxtext_model_path: str):
  from pathlib import Path
  
  # List of files to copy
  files_to_copy = [
      'config.json',
      'configuration_qwen.py', 
      'qwen.tiktoken',
      'tokenization_qwen.py',
      'tokenizer_config.json'
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
    base_model_path: str, maxtext_model_path: str, jax_weights: dict, device_count: int, use_ocdbt: bool, use_zarr3: bool, save_checkpoint: bool
):
  """
  Function to save jax_weights ready for MaxText to a parameters checkpoint.

  Args:
      base_model_path: Path to the source model directory (for copying config files).
      maxtext_model_path: Path to save the MaxText checkpoint.
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

  # convert all weights to jax.numpy with sharding if applicable
  jax_weights_flat, jax_weights_struct = tree.flatten(jax_weights)
  jax_weights_new = []
  while len(jax_weights_flat) > 0:
    jax_weight = jax_weights_flat.pop(0)
    jax_weights_new.append(checkpoint_device_put(jax_weight))
    del jax_weight
    gc.collect()
    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  jax_weights = tree.unflatten(jax_weights_struct, jax_weights_new)

  # dummy configs for the checkpoint_manager
  step_number_to_save_new_ckpt = 0
  enable_checkpointing = True
  async_checkpointing = False
  save_interval_steps = 1

  state_new = train_state.TrainState(
      step=0, apply_fn=None, params={"params": jax_weights}, tx=None, opt_state={}  # type: ignore
  )

  # save flax msgpack
  save_flax_msgpack(maxtext_model_path, jax_weights)

  if save_checkpoint:
    checkpoint_manager = check_pointing.create_orbax_checkpoint_manager(
        maxtext_model_path,
        enable_checkpointing,
        async_checkpointing,
        save_interval_steps,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
    )
    # save maxtext checkpoint
    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
    if checkpoint_manager is not None:
      if save_checkpoint.save_checkpoint(checkpoint_manager, step_number_to_save_new_ckpt, state_new):
        converter_logging.log(f"saved a maxtext checkpoint at step {step_number_to_save_new_ckpt}")
      # Upon preemption, exit when and only when all ongoing saves are complete.
      checkpoint_manager.wait_until_finished()


def list_folders_pathlib(directory: str):
  """Lists folders in a directory using pathlib module.

  Args:
    directory: The path to the directory

  Returns:
    A list of strings, where each string is the name of a folder.
    Returns an empty list if the directory doesn't exist or is not a directory.
  """
  dir_path = pathlib.Path(directory)

  if not dir_path.is_dir():
    return []

  folders = []
  for item in dir_path.iterdir():
    if item.is_dir():
      folders.append(item.name)  # Append only the name

  return folders


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--base-model-path", type=str, required=True)
  parser.add_argument("--maxtext-model-path", type=str, required=True)
  parser.add_argument("--model-size", type=str, required=True)
  parser.add_argument("--huggingface-checkpoint", type=str2bool, required=False, default=False)
  parser.add_argument("--save-checkpoint", type=str2bool, required=False, default=False)
  parser.add_argument("--use-ocdbt", type=str2bool, required=False, default=True)
  parser.add_argument("--use-zarr3", type=str2bool, required=False, default=True)
  parser.add_argument("--check", action="store_true", help="Verify conversion by comparing original and converted weights")
  args = parser.parse_args()

  if args.model_size not in MODEL_PARAMS_DICT:
    raise NotImplementedError

  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={SIMULATED_CPU_DEVICES_COUNT}"
  base_weights_path = args.maxtext_model_path

  if args.check:
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
    save_weights_to_checkpoint(
        args.base_model_path,
        args.maxtext_model_path,
        convert_to_jax_weights(args.base_model_path, args.model_size, args.huggingface_checkpoint),
        SIMULATED_CPU_DEVICES_COUNT,
        args.use_ocdbt,
        args.use_zarr3,
        args.save_checkpoint,
    )
    converter_logging.log(f"Successfully saved base_weights to {base_weights_path}.")
