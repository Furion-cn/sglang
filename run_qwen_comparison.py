#!/usr/bin/env python3
"""
Wrapper script for running QWen JAX vs PyTorch comparison tests.

Usage:
    python run_qwen_comparison.py --jax-model /path/to/jax/model --pytorch-model /path/to/pytorch/model
    python run_qwen_comparison.py --model /path/to/shared/model  # Use same path for both
    python run_qwen_comparison.py --jax-model /path/to/jax/model --pytorch-model /path/to/pytorch/model --test forward
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run QWen JAX vs PyTorch comparison tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use separate model paths
  python run_qwen_comparison.py --jax-model /path/to/jax/qwen --pytorch-model /path/to/pytorch/qwen
  
  # Use same model path for both frameworks
  python run_qwen_comparison.py --model /path/to/qwen/model
  
  # Run specific test
  python run_qwen_comparison.py --model /path/to/qwen --test loading
  
  # Run with verbose output
  python run_qwen_comparison.py --model /path/to/qwen --verbose
"""
    )
    
    # Model path arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model", 
        help="Model path to use for both JAX and PyTorch (sets both JAX_MODEL_PATH and PYTORCH_MODEL_PATH)"
    )
    
    parser.add_argument(
        "--jax-model", 
        help="Path to JAX model (sets JAX_MODEL_PATH)"
    )
    parser.add_argument(
        "--pytorch-model", 
        help="Path to PyTorch model (sets PYTORCH_MODEL_PATH)"
    )
    
    # Test selection
    parser.add_argument(
        "--test", 
        choices=["all", "loading", "forward", "generation"],
        default="all",
        help="Which test to run (default: all)"
    )
    
    # Output options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--save-results",
        help="Save test results to specified file"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model and not (args.jax_model and args.pytorch_model):
        parser.error("Must specify either --model or both --jax-model and --pytorch-model")
    
    # Set up environment variables
    env = os.environ.copy()
    
    if args.model:
        # Use same path for both
        env["JAX_MODEL_PATH"] = args.model
        env["PYTORCH_MODEL_PATH"] = args.model
        print(f"Using model path: {args.model}")
    else:
        # Use separate paths
        if args.jax_model:
            env["JAX_MODEL_PATH"] = args.jax_model
            print(f"JAX model path: {args.jax_model}")
        if args.pytorch_model:
            env["PYTORCH_MODEL_PATH"] = args.pytorch_model
            print(f"PyTorch model path: {args.pytorch_model}")
    
    # Validate model paths exist
    for env_var in ["JAX_MODEL_PATH", "PYTORCH_MODEL_PATH"]:
        if env_var in env:
            model_path = Path(env[env_var])
            if not model_path.exists():
                print(f"Error: Model path does not exist: {model_path}")
                sys.exit(1)
            if not (model_path / "config.json").exists():
                print(f"Warning: config.json not found in {model_path}")
    
    # Build pytest command
    test_file = Path(__file__).parent / "test" / "srt" / "test_qwen_jax_pytorch_forward_comparison.py"
    
    cmd = ["python", "-m", "pytest", str(test_file)]
    
    # Add test selection
    if args.test != "all":
        test_method_map = {
            "loading": "test_model_loading",
            "forward": "test_forward_pass_comparison", 
            "generation": "test_generation_comparison"
        }
        cmd.extend(["-k", test_method_map[args.test]])
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
        cmd.append("-s")  # Don't capture output
    
    # Add result saving
    if args.save_results:
        cmd.extend(["--tb=short", f"--resultlog={args.save_results}"])
    
    print(f"\nRunning command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run the test
    try:
        result = subprocess.run(cmd, env=env, check=False)
        
        print("=" * 60)
        if result.returncode == 0:
            print("✓ All tests passed successfully!")
        else:
            print(f"✗ Tests failed with exit code {result.returncode}")
            
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return 130
    except Exception as e:
        print(f"✗ Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())