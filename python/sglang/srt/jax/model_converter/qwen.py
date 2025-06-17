# pylint: disable=g-line-too-long
import argparse
import pathlib
import os
import gc
import logging

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
        "vocab": 151851,
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

def permute_to_match_maxtext_rope(arr):
  """
  Permutes the Qwen model's rotary position embedding weights to match MaxText's RoPE implementation.
  
  Qwen uses a different RoPE format where frequencies are concatenated as (freqs, freqs),
  while MaxText expects an interleaved format. This function converts from Qwen's format
  to MaxText's expected format.
  
  Qwen RoPE implementation:
  - freqs = torch.outer(seq, inv_freq)  # [seq_len, dim//2]
  - emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim] - concatenated format
  
  MaxText expects:
  - Interleaved format where cos/sin values alternate

  Args:
    arr (np.ndarray): Qwen model's RoPE weight array to permute.

  Returns:
    np.ndarray: Permutated array compatible with MaxText's RoPE implementation.
  """
  assert arr.shape[-1] % 2 == 0, "The last dimension for rope has to be even."
  
  # Qwen format: [cos_0, cos_1, ..., cos_n/2-1, sin_0, sin_1, ..., sin_n/2-1]
  # Jax format: [cos_0, sin_0, cos_1, sin_1, ..., cos_n/2-1, sin_n/2-1]
  
  half_dim = arr.shape[-1] // 2
  cos_part = arr[..., :half_dim]
  sin_part = arr[..., half_dim:]
  
  result = np.empty_like(arr)
  result[..., ::2] = cos_part
  result[..., 1::2] = sin_part
  
  return result


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
  base_num_query_heads = model_params["num_heads"]
  head_dim = model_params["dims_per_head"]
  base_num_kv_heads = model_params["num_kv_heads"]
  vocab_size = model_params["vocab"]

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

  # initialize the data structure for storing jax_weights to match QWenLMHeadModel structure
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
  
  converter_logging.log(f"Using original vocab_size: {vocab_size}")

  lm_head_weight = chkpt_vars["lm_head.weight"].to(torch.float32).numpy().astype(CAST_DTYPE)
  
  if lm_head_weight.shape[0] >= vocab_size:
    processed_weight = lm_head_weight[:vocab_size, :]
  else:
    processed_weight = np.zeros((vocab_size, lm_head_weight.shape[1]), dtype=lm_head_weight.dtype)
    processed_weight[:lm_head_weight.shape[0], :] = lm_head_weight
  
  jax_weights["lm_head"]["embedding"] = processed_weight

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # token embedding ##############################################
  converter_logging.log("Processing token embeddings")

  jax_weights["transformer"]["embed_tokens"]["embedding"] = (
      chkpt_vars["model.embed_tokens.weight"].to(torch.float32).numpy().astype(CAST_DTYPE)[:vocab_size, :]
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
    
    if model_size.startswith("qwen"):
      converter_logging.log(f"Applying Qwen RoPE permutation for layer {layer_idx}")
      q_dim = base_num_query_heads * head_dim
      k_dim = base_num_kv_heads * head_dim
      v_dim = base_num_kv_heads * head_dim
      
      wq = wqkv[:, :q_dim].reshape([base_num_query_heads * head_dim, base_num_query_heads, head_dim])
      wk = wqkv[:, q_dim:q_dim + k_dim].reshape([base_num_query_heads * head_dim, base_num_kv_heads, head_dim])
      wv = wqkv[:, q_dim + k_dim:q_dim + k_dim + v_dim].reshape([base_num_query_heads * head_dim, base_num_kv_heads, head_dim])
      
      wq = permute_to_match_maxtext_rope(wq)
      wk = permute_to_match_maxtext_rope(wk)
      
      wqkv = np.concatenate([
          wq.reshape([base_num_query_heads * head_dim, q_dim]),
          wk.reshape([base_num_query_heads * head_dim, k_dim]),
          wv.reshape([base_num_query_heads * head_dim, v_dim])
      ], axis=1)
    
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
    
    jax_weights["transformer"]["h"][layer_idx]["mlp"]["w1"]["kernel"] = w1
    jax_weights["transformer"]["h"][layer_idx]["mlp"]["w2"]["kernel"] = w2
    jax_weights["transformer"]["h"][layer_idx]["mlp"]["c_proj"]["kernel"] = c_proj
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

def save_weights_to_checkpoint(
    maxtext_model_path: str, jax_weights: dict, device_count: int, use_ocdbt: bool, use_zarr3: bool
):
  """
  Function to save jax_weights ready for MaxText to a parameters checkpoint.

  Args:
      maxtext_model_path: Path to save the MaxText checkpoint.
      jax_weights: The JAX model weights to be saved.
      device_count: The number of simulated devices.
      use_ocdbt: Whether to use Optimized Checkpoint Database with Transactions.
      use_zarr3: Whether to use Zarr3 or not.
  """
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

  checkpoint_manager = check_pointing.create_orbax_checkpoint_manager(
      maxtext_model_path,
      enable_checkpointing,
      async_checkpointing,
      save_interval_steps,
      use_ocdbt=use_ocdbt,
      use_zarr3=use_zarr3,
  )

  state_new = train_state.TrainState(
      step=0, apply_fn=None, params={"params": jax_weights}, tx=None, opt_state={}  # type: ignore
  )
  # save maxtext checkpoint
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  if checkpoint_manager is not None:
    if save_checkpoint.save_checkpoint(checkpoint_manager, step_number_to_save_new_ckpt, state_new):
      converter_logging.log(f"saved a maxtext checkpoint at step {step_number_to_save_new_ckpt}")
    # Upon preemption, exit when and only when all ongoing saves are complete.
    checkpoint_manager.wait_until_finished()
  # save flax msgpack
  save_flax_msgpack(maxtext_model_path, jax_weights)

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
  parser.add_argument("--use-ocdbt", type=str2bool, required=False, default=True)
  parser.add_argument("--use-zarr3", type=str2bool, required=False, default=True)
  args = parser.parse_args()

  if args.model_size not in MODEL_PARAMS_DICT:
    raise NotImplementedError

  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={SIMULATED_CPU_DEVICES_COUNT}"
  base_weights_path = args.maxtext_model_path

  save_weights_to_checkpoint(
      args.maxtext_model_path,
      convert_to_jax_weights(args.base_model_path, args.model_size, args.huggingface_checkpoint),
      SIMULATED_CPU_DEVICES_COUNT,
      args.use_ocdbt,
      args.use_zarr3,
  )
  converter_logging.log(f"Successfully saved base_weights to {base_weights_path}.")
