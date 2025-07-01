"""Benchmark for RotaryEmbedding implementation in JAX."""

import argparse
import time
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

# Import the RotaryEmbedding class
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from python.sglang.srt.jax.layers.embeddings import RotaryEmbedding


class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int = 10000,
        is_neox_style: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
    
    def __str__(self):
        return (
            f"head_size={self.head_size}, rotary_dim={self.rotary_dim}, "
            f"max_pos={self.max_position_embeddings}, base={self.base}, "
            f"neox_style={self.is_neox_style}, dtype={self.dtype}"
        )


def create_test_inputs(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    dtype: jnp.dtype,
    key: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Create test inputs for RotaryEmbedding."""
    subkeys = jax.random.split(key, 3)
    
    # Positions: [batch_size, seq_len]
    positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    
    # Query and Key: [batch_size * seq_len, num_heads, head_size]
    total_tokens = batch_size * seq_len
    query = jax.random.normal(subkeys[0], (total_tokens, num_heads, head_size), dtype=dtype)
    key_tensor = jax.random.normal(subkeys[1], (total_tokens, num_heads, head_size), dtype=dtype)
    
    return positions, query, key_tensor


def benchmark_rotary_embedding(
    config: BenchmarkConfig,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_warmup: int = 10,
    num_trials: int = 100,
    key: jax.Array = None
) -> Dict[str, float]:
    """Benchmark RotaryEmbedding with given configuration."""
    
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # Initialize RotaryEmbedding
    rngs = nnx.Rngs(0)
    rotary_emb = RotaryEmbedding(
        head_size=config.head_size,
        rotary_dim=config.rotary_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.base,
        is_neox_style=config.is_neox_style,
        dtype=config.dtype,
    )
    
    # Create test inputs
    positions, query, key_tensor = create_test_inputs(
        batch_size, seq_len, num_heads, config.head_size, config.dtype, key
    )
    
    # Compile the function by running it once
    _ = rotary_emb(positions, query, key_tensor)
    
    # Warmup runs
    for _ in range(num_warmup):
        _ = rotary_emb(positions, query, key_tensor)
    
    # Actual benchmark runs
    times = []
    for _ in range(num_trials):
        start_time = time.perf_counter()
        output_query, output_key = rotary_emb(positions, query, key_tensor)
        
        # Ensure computation is complete
        jax.block_until_ready(output_query)
        jax.block_until_ready(output_key)
        
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    # Calculate statistics
    times = jnp.array(times)
    stats = {
        "mean_ms": float(jnp.mean(times)),
        "std_ms": float(jnp.std(times)),
        "min_ms": float(jnp.min(times)),
        "max_ms": float(jnp.max(times)),
        "median_ms": float(jnp.median(times)),
        "p95_ms": float(jnp.percentile(times, 95)),
        "p99_ms": float(jnp.percentile(times, 99)),
        "throughput_tokens_per_sec": (batch_size * seq_len) / (float(jnp.mean(times)) / 1000),
    }
    
    return stats


def benchmark_memory_usage(
    config: BenchmarkConfig,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    key: jax.Array = None
) -> Dict[str, int]:
    """Measure memory usage of RotaryEmbedding."""
    
    if key is None:
        key = jax.random.PRNGKey(42)
    
    try:
        # Get initial memory usage
        initial_memory_stats = jax.devices()[0].memory_stats()
        if initial_memory_stats is None:
            # Fallback for devices that don't support memory stats
            return {
                "model_memory_bytes": 0,
                "input_memory_bytes": 0,
                "output_memory_bytes": 0,
                "total_memory_bytes": 0,
                "model_memory_mb": 0.0,
                "input_memory_mb": 0.0,
                "output_memory_mb": 0.0,
                "total_memory_mb": 0.0,
            }
        
        initial_memory = initial_memory_stats['bytes_in_use']
        
        # Initialize RotaryEmbedding
        rngs = nnx.Rngs(0)
        rotary_emb = RotaryEmbedding(
            head_size=config.head_size,
            rotary_dim=config.rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.base,
            is_neox_style=config.is_neox_style,
            dtype=config.dtype,
        )
        
        model_memory_stats = jax.devices()[0].memory_stats()
        model_memory = model_memory_stats['bytes_in_use'] if model_memory_stats else initial_memory
        
        # Create test inputs
        positions, query, key_tensor = create_test_inputs(
            batch_size, seq_len, num_heads, config.head_size, config.dtype, key
        )
        
        input_memory_stats = jax.devices()[0].memory_stats()
        input_memory = input_memory_stats['bytes_in_use'] if input_memory_stats else model_memory
        
        # Run forward pass
        output_query, output_key = rotary_emb(positions, query, key_tensor)
        jax.block_until_ready(output_query)
        jax.block_until_ready(output_key)
        
        output_memory_stats = jax.devices()[0].memory_stats()
        output_memory = output_memory_stats['bytes_in_use'] if output_memory_stats else input_memory
        
        return {
            "model_memory_bytes": model_memory - initial_memory,
            "input_memory_bytes": input_memory - model_memory,
            "output_memory_bytes": output_memory - input_memory,
            "total_memory_bytes": output_memory - initial_memory,
            "model_memory_mb": (model_memory - initial_memory) / (1024 * 1024),
            "input_memory_mb": (input_memory - model_memory) / (1024 * 1024),
            "output_memory_mb": (output_memory - input_memory) / (1024 * 1024),
            "total_memory_mb": (output_memory - initial_memory) / (1024 * 1024),
        }
        
    except Exception as e:
        print(f"Warning: Memory measurement failed ({e}), using fallback values")
        return {
            "model_memory_bytes": 0,
            "input_memory_bytes": 0,
            "output_memory_bytes": 0,
            "total_memory_bytes": 0,
            "model_memory_mb": 0.0,
            "input_memory_mb": 0.0,
            "output_memory_mb": 0.0,
            "total_memory_mb": 0.0,
        }
    
max_token_length: int = 4096

def run_comprehensive_benchmark():
    """Run a comprehensive benchmark across different configurations."""
    
    print("=" * 80)
    print("RotaryEmbedding Benchmark Results")
    print("=" * 80)
    
    # Test configurations
    configs = [
        # for qwen-7B
        BenchmarkConfig(head_size=32, rotary_dim=32,
                        max_position_embeddings=max_token_length),
        BenchmarkConfig(head_size=32, rotary_dim=32,
                        max_position_embeddings=max_token_length, is_neox_style=False),
        
    ]
    
    # Input size configurations
    input_configs = [
        (1, 1024, 32),    # Small: batch=1, seq_len=128, num_heads=32
        (4, 1024, 32),    # Medium: batch=4, seq_len=512, num_heads=32
        (8, 1024, 32),   # Large: batch=8, seq_len=1024, num_heads=32
        (16, 1024, 32),  # Very Large: batch=16, seq_len=2048, num_heads=32
    ]
    
    key = jax.random.PRNGKey(42)
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        print("-" * 60)
        
        for batch_size, seq_len, num_heads in input_configs:
            input_desc = f"batch={batch_size}, seq_len={seq_len}, heads={num_heads}"
            print(f"\nInput: {input_desc}")
            
            try:
                # Performance benchmark
                perf_stats = benchmark_rotary_embedding(
                    config, batch_size, seq_len, num_heads, key=key
                )
                
                # Memory benchmark
                memory_stats = benchmark_memory_usage(
                    config, batch_size, seq_len, num_heads, key=key
                )
                
                print(f"  Performance:")
                print(f"    Mean time: {perf_stats['mean_ms']:.3f} ± {perf_stats['std_ms']:.3f} ms")
                print(f"    Min/Max: {perf_stats['min_ms']:.3f} / {perf_stats['max_ms']:.3f} ms")
                print(f"    P95/P99: {perf_stats['p95_ms']:.3f} / {perf_stats['p99_ms']:.3f} ms")
                print(f"    Throughput: {perf_stats['throughput_tokens_per_sec']:.0f} tokens/sec")
                
                print(f"  Memory:")
                print(f"    Model: {memory_stats['model_memory_mb']:.2f} MB")
                print(f"    Input: {memory_stats['input_memory_mb']:.2f} MB")
                print(f"    Output: {memory_stats['output_memory_mb']:.2f} MB")
                print(f"    Total: {memory_stats['total_memory_mb']:.2f} MB")
                
            except Exception as e:
                print(f"  Error: {e}")
    
    print("\n" + "=" * 80)


def run_scalability_test():
    """Test scalability with increasing input sizes."""
    
    print("\nScalability Test: RotaryEmbedding Performance vs Input Size")
    print("=" * 60)
    
    config = BenchmarkConfig(head_size=128, rotary_dim=128, max_position_embeddings=8192)
    num_heads = 32
    key = jax.random.PRNGKey(42)
    
    # Test different sequence lengths
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    batch_size = 4
    
    print(f"Configuration: {config}")
    print(f"Fixed: batch_size={batch_size}, num_heads={num_heads}")
    print(f"Variable: seq_len\n")
    
    print(f"{'Seq Len':<8} {'Time (ms)':<12} {'Throughput':<15} {'Memory (MB)':<12}")
    print("-" * 50)
    
    for seq_len in seq_lengths:
        try:
            perf_stats = benchmark_rotary_embedding(
                config, batch_size, seq_len, num_heads, num_trials=50, key=key
            )
            memory_stats = benchmark_memory_usage(
                config, batch_size, seq_len, num_heads, key=key
            )
            
            print(f"{seq_len:<8} {perf_stats['mean_ms']:<12.3f} "
                  f"{perf_stats['throughput_tokens_per_sec']:<15.0f} "
                  f"{memory_stats['total_memory_mb']:<12.2f}")
                  
        except Exception as e:
            print(f"{seq_len:<8} ERROR: {e}")


def run_dtype_comparison():
    """Compare performance across different data types."""
    
    print("\nData Type Comparison")
    print("=" * 40)
    
    dtypes = [jnp.float16, jnp.bfloat16, jnp.float32]
    batch_size, seq_len, num_heads = 4, 1024, 32
    key = jax.random.PRNGKey(42)
    
    print(f"Input: batch={batch_size}, seq_len={seq_len}, heads={num_heads}\n")
    print(f"{'Dtype':<12} {'Time (ms)':<12} {'Memory (MB)':<12}")
    print("-" * 40)
    
    for dtype in dtypes:
        config = BenchmarkConfig(
            head_size=128, rotary_dim=128, max_position_embeddings=2048, dtype=dtype
        )
        
        try:
            perf_stats = benchmark_rotary_embedding(
                config, batch_size, seq_len, num_heads, num_trials=50, key=key
            )
            memory_stats = benchmark_memory_usage(
                config, batch_size, seq_len, num_heads, key=key
            )
            
            print(f"{str(dtype):<12} {perf_stats['mean_ms']:<12.3f} "
                  f"{memory_stats['total_memory_mb']:<12.2f}")
                  
        except Exception as e:
            print(f"{str(dtype):<12} ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark RotaryEmbedding implementation")
    parser.add_argument(
        "--test-type",
        choices=["comprehensive", "scalability", "dtype", "all"],
        default="all",
        help="Type of benchmark to run"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run benchmark on (auto, gpu, cpu)"
    )
    
    args = parser.parse_args()
    
    # Set up JAX
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default device: {jax.devices()[0]}")
    
    # Check for GPU availability
    try:
        gpu_devices = jax.devices("gpu")
        has_gpu = len(gpu_devices) > 0
    except:
        has_gpu = False
    
    if args.device == "gpu":
        if not has_gpu:
            print("Warning: No GPU found, falling back to CPU")
        else:
            print(f"Running on GPU: {gpu_devices[0]}")
    elif args.device == "auto":
        if has_gpu:
            print(f"Auto-detected GPU: {gpu_devices[0]}")
        else:
            print("Auto-detected CPU")
    else:
        print("Running on CPU")
    
    if args.test_type in ["comprehensive", "all"]:
        run_comprehensive_benchmark()
    
    if args.test_type in ["scalability", "all"]:
        run_scalability_test()
    
    if args.test_type in ["dtype", "all"]:
        run_dtype_comparison()


if __name__ == "__main__":
    main()
