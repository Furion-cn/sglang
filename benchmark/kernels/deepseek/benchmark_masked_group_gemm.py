import random

import torch
import triton

from sglang.srt.layers.moe.ep_moe.layer import grouped_gemm_masked_triton


def masked_grouped_gemm_triton(x_grouped, y_grouped, out, m_indices):
    grouped_gemm_masked_triton(
        x_grouped, y_grouped, out, m_indices, c_dtype=torch.bfloat16
    )
    return out


def create_benchmark_configs():
    configs = [
        (1024, 4096, 7168, 1),
        (1024, 7168, 2048, 1),
        (512, 4096, 7168, 2),
        (512, 7168, 2048, 2),
        (256, 4096, 7168, 4),
        (256, 7168, 2048, 4),
        (128, 2048, 7168, 8),
        (128, 4096, 7168, 8),
    ]
    return configs


def get_benchmark():
    all_configs = create_benchmark_configs()

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n", "k", "num_groups"],
            x_vals=[config for config in all_configs],
            line_arg="provider",
            line_vals=["triton"],
            line_names=["Triton"],
            styles=[("red", "-")],
            ylabel="us",
            plot_name=f"masked-grouped-gemm-performance",
            args={},
        )
    )
    def benchmark(m, n, k, num_groups, provider):
        print(f"Shape (m={m}, n={n}, k={k}, num_groups={num_groups}")
        x = torch.randn((num_groups, m, k), device="cuda", dtype=torch.bfloat16)
        y = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((num_groups, m, n), device="cuda", dtype=torch.bfloat16)

        masked_m_candidates = list(
            filter(lambda candidate: candidate <= m, (64, 128, 192, 256, 320, 384))
        )
        masked_m = torch.empty((num_groups,), device="cuda", dtype=torch.int)
        for i in range(num_groups):
            masked_m[i] = random.choice(masked_m_candidates)

        quantiles = [0.5, 0.2, 0.8]

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: masked_grouped_gemm_triton(
                x,
                y,
                out,
                masked_m,
            ),
            quantiles=quantiles,
        )

        # Calculate TFLOPS
        flops = 2 * num_groups * m * n * k  # multiply-adds
        tflops = flops / (ms * 1e-3) / 1e12

        print(f"Time: {ms * 1000:.2f} us, TFLOPS: {tflops:.2f}")
        return ms * 1000, max_ms * 1000, min_ms * 1000  # convert to us

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./configs/benchmark_ops/masked_group_gemm/",
        help="Path to save masked group gemm benchmark results",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Enable TF32, adapted from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_core.py#L148
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    benchmark = get_benchmark()

    print(f"Running performance benchmark...")
    benchmark.run(print_data=True, save_path=args.save_path)
