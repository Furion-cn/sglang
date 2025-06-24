#!/usr/bin/env python3
"""
QWen JAX vs PyTorch Forward Pass Comparison Test

This test loads both JAX and PyTorch versions of QWen models from local files
and compares their forward pass outputs to diagnose differences.

Usage:
    # Set model path and run test
    MODEL_PATH=/path/to/qwen/model python -m unittest test_qwen_jax_pytorch_forward_comparison.TestQWenForwardComparison
    
    # Run specific test
    MODEL_PATH=/path/to/qwen/model python -m unittest test_qwen_jax_pytorch_forward_comparison.TestQWenForwardComparison.test_forward_pass_comparison
"""

import os
import sys
import pytest
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['VLLM_USE_MODELSCOPE'] = 'false'

# Add the parent directory to the path to import sglang modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import jax.numpy as jnp
from sglang.srt.jax.models.qwen import QWenLMHeadJaxModel as JAXQWenLMHeadModel
from sglang.srt.model_loader.loader import JAXModelLoader, get_model_loader
from sglang.test.jax.test_utils import create_device_mesh

import torch

from transformers import AutoTokenizer, AutoConfig

from sglang.srt.debug_tracer import global_tracer
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend


class MockModelRunner:
    """Mock ModelRunner for TorchNativeAttnBackend"""
    def __init__(self):
        self.device = torch.device("cpu")


class MockReqToTokenPool:
    """Mock implementation of ReqToTokenPool for testing"""
    
    def __init__(self):
        self.size = 100
        self.max_context_len = 512
        self.req_to_token = torch.zeros(
            (self.size, self.max_context_len), dtype=torch.int32
        )
        self.free_slots = list(range(self.size))
    
    def write(self, indices, values):
        self.req_to_token[indices] = values
    
    def available_size(self):
        return len(self.free_slots)
    
    def alloc(self, need_size: int):
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index
    
    def free(self, free_index):
        if isinstance(free_index, int):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)
    
    def clear(self):
        self.free_slots = list(range(self.size))


class MockTokenToKVPool:
    """Mock implementation of TokenToKVPool for testing"""
    
    def __init__(self):
        # Create actual memory buffers for more realistic testing
        self.size = 1000
        self.page_size = 16
        self.num_layers = 32
        self.num_heads = 32
        self.head_dim = 128
        
        # Initialize key and value buffers
        self.key_buffer = torch.zeros(
            (self.size, self.num_layers, self.page_size, self.num_heads, self.head_dim),
            dtype=torch.float16
        )
        self.value_buffer = torch.zeros(
            (self.size, self.num_layers, self.page_size, self.num_heads, self.head_dim),
            dtype=torch.float16
        )
        
        # Track free pages
        self.free_pages = list(range(self.size))
    
    def get_key_buffer(self, layer_id):
        """Return the key buffer for a specific layer"""
        return self.key_buffer[:, layer_id]
    
    def get_value_buffer(self, layer_id):
        """Return the value buffer for a specific layer"""
        return self.value_buffer[:, layer_id]
    
    def get_kv_buffer(self, layer_id):
        """Return both key and value buffers for a specific layer"""
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)
    
    def set_kv_buffer(self, layer, loc, cache_k, cache_v):
        """Set KV buffer - mock implementation for testing"""
        # Mock implementation - just store the data without actual processing
        # In a real implementation, this would store cache_k and cache_v at the specified locations
        pass


class MockForwardBatch:
    """Mock implementation of ForwardBatch for testing"""
    
    def __init__(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        
        # Basic attributes
        self.forward_mode = ForwardMode.DECODE
        self.batch_size = 1
        self.seq_len = 10
        self.max_seq_len = 512
        
        # Memory pools
        self.req_to_token_pool = MockReqToTokenPool()
        self.token_to_kv_pool = MockTokenToKVPool()
        
        # Required tensor attributes
        self.input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.int64)
        self.req_pool_indices = torch.zeros(1, dtype=torch.int32)
        self.seq_lens = torch.ones(1, dtype=torch.int32) * 10
        self.out_cache_loc = torch.zeros(10, dtype=torch.int32)
        self.positions = torch.arange(10, dtype=torch.int64)
        
        # Sequence length info
        self.seq_lens_sum = 10
        self.seq_lens_cpu = None
        
        # For extend mode
        self.extend_num_tokens = 0
        self.extend_seq_lens = torch.ones(1, dtype=torch.int32) * 10
        self.extend_prefix_lens = torch.zeros(1, dtype=torch.int32)
        self.extend_start_loc = torch.zeros(1, dtype=torch.int32)
        self.extend_prefix_lens_cpu = [0]
        self.extend_seq_lens_cpu = [10]
        self.extend_logprob_start_lens_cpu = None
        self.extend_input_logprob_token_ids_gpu = None
        
        # For logprob
        self.return_logprob = False
        self.top_logprobs_nums = None
        self.token_ids_logprobs = None
        
        # Temperature and sampling
        self.temp_scaled_logprobs = False
        self.temperature = None
        self.top_p_normalized_logprobs = False
        self.top_p = None
        
        # Multimodal
        self.mm_inputs = None
        
        # Encoder-decoder
        self.encoder_cached = None
        self.encoder_lens = None
        self.encoder_lens_cpu = None
        self.encoder_out_cache_loc = None
        
        # LoRA
        self.lora_paths = None
        
        # Input embeddings
        self.input_embeds = None
        
        # Sampling info
        self.sampling_info = None
        
        # Use TorchNative backend for CPU compatibility
        mock_model_runner = MockModelRunner()
        self.attn_backend = TorchNativeAttnBackend(mock_model_runner)
        print("Using TorchNative attention backend for CPU")
        
        # DP attention
        self.global_num_tokens_cpu = None
        self.global_num_tokens_gpu = None
        self.global_num_tokens_for_logprob_cpu = None
        self.global_num_tokens_for_logprob_gpu = None
        self.dp_local_start_pos = None
        self.dp_local_num_tokens = None
        self.gathered_buffer = None
        self.can_run_dp_cuda_graph = False
        self.global_forward_mode = None
        
        # Speculative decoding
        self.spec_info = None
        self.spec_algorithm = None
        self.capture_hidden_mode = None
        
        # Padding
        self.padded_static_len = -1
        self.num_token_non_padded = None
        
        # Qwen2-VL
        self.mrope_positions = None
        
        # Two-batch overlap
        self.tbo_split_seq_index = None
        self.tbo_parent_token_range = None
        self.tbo_children = None
        self.can_run_tbo = False
        
        # MLA chunked prefix cache
        self.attn_attend_prefix_cache = None
        self.num_prefix_chunks = None
        self.prefix_chunk_idx = None
        self.prefix_chunk_len = None
        self.prefix_chunk_starts = None
        self.prefix_chunk_seq_lens = None
        self.prefix_chunk_cu_seq_lens = None
        self.prefix_chunk_max_seq_lens = None
        self.prefix_chunk_num_tokens = None
        self.prefix_chunk_kv_indices = None
    
    def contains_mm_inputs(self):
        """Check if batch contains multimodal inputs"""
        return self.mm_inputs is not None and any(mm is not None for mm in self.mm_inputs)
    
    def contains_image_inputs(self):
        """Check if batch contains image inputs"""
        return False  # Mock implementation


class TestQWenForwardComparison(unittest.TestCase):
    """Test cases for comparing JAX and PyTorch QWen model forward passes"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            init_distributed_environment(
                backend="gloo",  # Use gloo backend for CPU
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method="tcp://127.0.0.1:2646",
            )
            initialize_model_parallel(tensor_model_parallel_size=1)
        except AssertionError:
            # ignore this error: tensor model parallel group is already initialized
            pass
        
        # Force CPU mode for all operations
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['JAX_PLATFORMS'] = 'cpu'
        
        # Print attention backend availability
        self._print_backend_info()
        
        # Configuration
        # Support both old MODEL_PATH and new separate paths
        self.jax_model_path = os.environ.get('JAX_MODEL_PATH', os.environ.get('MODEL_PATH', '/tmp/test_qwen_model'))
        self.pytorch_model_path = os.environ.get('PYTORCH_MODEL_PATH', os.environ.get('MODEL_PATH', '/tmp/test_qwen_model'))
        # For backward compatibility, use pytorch_model_path as default test_model_path
        self.test_model_path = self.pytorch_model_path
        self.test_text = "Hello, how are you today?"
        self.max_new_tokens = 5
        self.temperature = 0.0  # Use deterministic generation for comparison
        
        # JAX setup
        self.mesh = create_device_mesh(
            ici_parallelism=[-1, 1, 1, 1], 
            dcn_parallelism=[1, 1, 1, 1]
        )
        self.load_config = LoadConfig(load_format=LoadFormat.JAX)
        self.device_config = DeviceConfig()
        self.jax_loader = JAXModelLoader(self.load_config)
        
        # Enable debug tracing for PyTorch
        global_tracer.enable()
        global_tracer.clear_records()
    
    def tearDown(self):
        """Clean up after tests"""
        global_tracer.disable()
        global_tracer.clear_records()
    
    def _print_backend_info(self):
        """Print information about attention backend"""
        print("\n=== Attention Backend Information ===")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print("Using TorchNative backend for CPU compatibility")
        print("====================================\n")
    
    def _get_positions_jax(self, x):
        """Get position embeddings for JAX model"""
        return jnp.concatenate([
            jnp.arange(x.shape[1]) for _ in range(x.shape[0])
        ]).reshape(x.shape[0], x.shape[1])
    
    def _get_positions_pytorch(self, x):
        """Get position embeddings for PyTorch model"""
        return torch.concatenate([
            torch.arange(x.shape[1]) for _ in range(x.shape[0])
        ]).reshape(x.shape[0], x.shape[1])
    
    def _get_tokenizer(self):
        """Get tokenizer from local path if available, otherwise from Hugging Face"""
        # Try PyTorch model path first, then JAX model path
        for model_path_str in [self.pytorch_model_path, self.jax_model_path]:
            model_path = Path(model_path_str)
            
            # Check if tokenizer files exist in the model path
            tokenizer_files = [
                'tokenizer_config.json',
                'tokenization_qwen.py', 
                'qwen.tiktoken'
            ]
            
            has_tokenizer = all((model_path / file).exists() for file in tokenizer_files)
            
            if has_tokenizer:
                print(f"📁 Using local tokenizer from: {model_path}")
                try:
                    return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
                except Exception as e:
                    print(f"⚠️  Failed to load local tokenizer from {model_path}: {e}")
                    continue
        
        print(f"📁 No tokenizer found in model paths, using Hugging Face")
        
        # Fallback to Hugging Face
        print("🌐 Loading tokenizer from Hugging Face: Qwen/Qwen-7B")
        return AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
    
    def _load_jax_model(self):
        """Load JAX model from local path"""
        if not os.path.exists(self.jax_model_path):
            self.skipTest(f"JAX model path {self.jax_model_path} not found. Set JAX_MODEL_PATH environment variable.")
        
        try:
            hf_folder, hf_weights_files = self.jax_loader._prepare_jax_weights(
                self.jax_model_path, None
            )
            
            if not hf_weights_files:
                self.skipTest(f"No .msgpack files found in {self.jax_model_path}")
            
            print(f"\n=== Loading JAX Model from: {self.jax_model_path} ===")
            print(f"Found {len(hf_weights_files)} msgpack files")
            
            model_config = ModelConfig(
                model_path=self.jax_model_path,
                model_override_args="{}"
            )
            
            with patch('sglang.srt.model_loader.loader.get_model_architecture') as mock_arch:
                mock_arch.return_value = (JAXQWenLMHeadModel, None)
                
                jax_model = self.jax_loader.load_model(
                    model_config=model_config,
                    device_config=self.device_config,
                    mesh=self.mesh,
                )
                
                print("✅ JAX Model loaded successfully!")
                return jax_model
                
        except Exception as e:
            self.fail(f"Failed to load JAX model: {e}")
    
    def _load_pytorch_model(self):
        """Load PyTorch model from local path using DefaultModelLoader"""
        if not os.path.exists(self.pytorch_model_path):
            self.skipTest(f"PyTorch model path {self.pytorch_model_path} not found. Set PYTORCH_MODEL_PATH environment variable.")
        
        try:
            print(f"\n=== Loading PyTorch Model from: {self.pytorch_model_path} ===")
            
            # Create load config for PyTorch (default format)
            pytorch_load_config = LoadConfig(load_format=LoadFormat.AUTO)
            
            # Create model config
            model_config = ModelConfig(
                model_path=self.pytorch_model_path,
                model_override_args="{}"
            )
            
            # Create device config for CPU
            device_config = DeviceConfig(device="cpu")
            
            # Get PyTorch model loader
            pytorch_loader = get_model_loader(pytorch_load_config)
            
            # Load model using standard loader
            pytorch_model = pytorch_loader.load_model(
                model_config=model_config,
                device_config=device_config,
            )
            
            print("✅ PyTorch Model loaded successfully using DefaultModelLoader!")
            return pytorch_model
            
        except Exception as e:
            self.fail(f"Failed to load PyTorch model: {e}")
    
    def test_model_loading(self):
        """Test that both JAX and PyTorch models can be loaded successfully"""
        print("\n=== Testing Model Loading ===")
        
        # Test JAX model loading
        try:
            jax_model = self._load_jax_model()
            print("✓ JAX model loaded successfully")
            print(f"JAX model type: {type(jax_model)}")
        except Exception as e:
            print(f"✗ JAX model loading failed: {e}")
            self.fail(f"JAX model loading failed: {e}")
        
        # Test PyTorch model loading
        try:
            pytorch_model = self._load_pytorch_model()
            print("✓ PyTorch model loaded successfully")
            print(f"PyTorch model type: {type(pytorch_model)}")
        except Exception as e:
            print(f"✗ PyTorch model loading failed: {e}")
            self.fail(f"PyTorch model loading failed: {e}")
    
    def test_forward_pass_comparison(self):
        """Test forward pass comparison between JAX and PyTorch models"""
        print("\n=== Testing Forward Pass Comparison ===")
        
        # All dependencies are required - no skip logic
        
        # Load models
        jax_model = self._load_jax_model()
        pytorch_model = self._load_pytorch_model()
        
        # Load tokenizer
        tokenizer = self._get_tokenizer()
        
        # Prepare input
        input_text = self.test_text
        input_ids = tokenizer.encode(input_text)
        
        print(f"Input text: {input_text}")
        print(f"Input IDs shape: {len(input_ids)}")
        
        # Clear debug tracer
        global_tracer.clear_records()
        
        # JAX forward pass
        print("\n--- JAX Forward Pass ---")
        jax_input_ids = jnp.array(input_ids).reshape(1, -1)
        jax_positions = self._get_positions_jax(jax_input_ids)
        
        with self.mesh:
            jax_output = jax_model(jax_input_ids, jax_positions, None)
        jax_records = global_tracer.get_records()
        
        # Clear tracer for PyTorch
        global_tracer.clear_records()
        
        # PyTorch forward pass
        print("\n--- PyTorch Forward Pass ---")
        torch_input_ids = torch.tensor(input_ids, dtype=torch.long).reshape(1, -1)
        torch_positions = self._get_positions_pytorch(torch_input_ids)
        
        # Create mock forward batch with TorchNative backend
        mock_batch = MockForwardBatch()
        
        pytorch_output = pytorch_model(torch_input_ids, torch_positions, mock_batch)
        pytorch_records = global_tracer.get_records()
        
        # Compare outputs
        print("\n--- Output Comparison ---")
        print(f"JAX output shape: {jax_output.shape}")
        print(f"PyTorch output shape: {pytorch_output.shape}")
        
        # Convert to numpy for comparison
        jax_output_np = np.array(jax_output)
        pytorch_output_np = pytorch_output.detach().cpu().numpy()
        
        # Calculate differences
        abs_diff = np.abs(jax_output_np - pytorch_output_np)
        rel_diff = abs_diff / (np.abs(jax_output_np) + 1e-8)
        
        print(f"Max absolute difference: {np.max(abs_diff):.6f}")
        print(f"Mean absolute difference: {np.mean(abs_diff):.6f}")
        print(f"Max relative difference: {np.max(rel_diff):.6f}")
        print(f"Mean relative difference: {np.mean(rel_diff):.6f}")
        
        # Check for NaN or Inf
        jax_has_nan = np.any(np.isnan(jax_output_np))
        jax_has_inf = np.any(np.isinf(jax_output_np))
        pytorch_has_nan = np.any(np.isnan(pytorch_output_np))
        pytorch_has_inf = np.any(np.isinf(pytorch_output_np))
        
        print(f"JAX output has NaN: {jax_has_nan}, Inf: {jax_has_inf}")
        print(f"PyTorch output has NaN: {pytorch_has_nan}, Inf: {pytorch_has_inf}")
        
        # Compare debug traces
        print("\n--- Debug Trace Comparison ---")
        print(f"JAX recorded {len(jax_records)} steps")
        print(f"PyTorch recorded {len(pytorch_records)} steps")
        
        # Assert reasonable differences
        self.assertFalse(jax_has_nan, "JAX output contains NaN")
        self.assertFalse(jax_has_inf, "JAX output contains Inf")
        self.assertFalse(pytorch_has_nan, "PyTorch output contains NaN")
        self.assertFalse(pytorch_has_inf, "PyTorch output contains Inf")
        
        # Allow for some numerical differences due to different implementations
        self.assertLess(np.max(abs_diff), 1e-3, "Outputs differ too much")
        
        print("✓ Forward pass comparison completed successfully")
    
    def test_generation_comparison(self):
        """Test generation comparison between JAX and PyTorch models"""
        print("\n=== Testing Generation Comparison ===")
        
        # All dependencies are required - no skip logic
        
        # Load models
        jax_model = self._load_jax_model()
        pytorch_model = self._load_pytorch_model()
        
        # Load tokenizer
        tokenizer = self._get_tokenizer()
        
        # Prepare input
        input_text = self.test_text
        input_ids = tokenizer.encode(input_text)
        
        print(f"Input text: {input_text}")
        print(f"Input IDs: {input_ids}")
        
        # JAX generation (simplified)
        print("\n--- JAX Generation ---")
        jax_input_ids = jnp.array(input_ids).reshape(1, -1)
        
        # Simple greedy generation for JAX
        generated_jax = []
        current_ids = jax_input_ids
        
        with self.mesh:
            for step in range(self.max_new_tokens):
                positions = self._get_positions_jax(current_ids)
                
                logits_output = jax_model(current_ids, positions, None)
                
                # Extract logits from LogitsProcessorOutput
                logits = logits_output.next_token_logits
                
                # Get next token (greedy)
                next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                generated_jax.append(int(next_token[0, 0]))
                
                # Append to current sequence
                current_ids = jnp.concatenate([current_ids, next_token], axis=1)
                
                print(f"Step {step + 1}: Generated token {int(next_token[0, 0])}")
        
        # Decode JAX generation
        jax_generated_text = tokenizer.decode(generated_jax)
        print(f"JAX generated text: {jax_generated_text}")
        
        # PyTorch generation (simplified - note: full generation not implemented)
        print("\n--- PyTorch Generation (Forward Pass Only) ---")
        torch_input_ids = torch.tensor(input_ids, dtype=torch.long).reshape(1, -1)
        torch_positions = self._get_positions_pytorch(torch_input_ids)
        
        # Create mock forward batch with TorchNative backend
        mock_batch = MockForwardBatch()
        
        with torch.no_grad():
            pytorch_logits = pytorch_model(torch_input_ids, torch_positions, mock_batch)
            
            # Get next token (greedy)
            next_token_pytorch = torch.argmax(pytorch_logits[:, -1, :], dim=-1)
            
        print(f"PyTorch next token: {int(next_token_pytorch[0])}")
        
        # Compare first token predictions
        print("\n--- Generation Comparison ---")
        jax_first_token = generated_jax[0] if generated_jax else None
        pytorch_first_token = int(next_token_pytorch[0])
        
        print(f"JAX first generated token: {jax_first_token}")
        print(f"PyTorch first predicted token: {pytorch_first_token}")
        
        if jax_first_token is not None:
            tokens_match = jax_first_token == pytorch_first_token
            print(f"First tokens match: {tokens_match}")
            
            if not tokens_match:
                print("⚠️ First token predictions differ - this may indicate model differences")
        
        print("✓ Generation comparison completed")


if __name__ == '__main__':
    unittest.main()