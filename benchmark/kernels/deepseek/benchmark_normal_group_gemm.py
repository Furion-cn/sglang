import torch
import triton

from sglang.srt.layers.moe.ep_moe.layer import grouped_gemm_triton


def normal_grouped_gemm_triton(
    x,
    y,
    out,
    num_groups,
    weight_column_major=True,
    seg_indptr=None,
    weight_indices=None,
):
    grouped_gemm_triton(
        x,
        y,
        out,
        num_groups,
        weight_column_major=weight_column_major,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
    )
    return out


def create_benchmark_configs():
    configs = [
        (16 * 1024, 2048, 7168, 8),
        (16 * 1024, 4096, 7168, 8),
        (12 * 1024, 2048, 7168, 8),
        (12 * 1024, 4096, 7168, 8),
        (8 * 1024, 2048, 7168, 8),
        (8 * 1024, 4096, 7168, 8),
        (4 * 1024, 2048, 7168, 8),
        (4 * 1024, 4096, 7168, 8),
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
        x = torch.randn((num_groups * m, k), device="cuda", dtype=torch.bfloat16)
        y = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((num_groups * m, n), device="cuda", dtype=torch.bfloat16)

        seg_indptr = torch.zeros((num_groups + 1,), device="cuda", dtype=torch.int)
        for i in range(num_groups):
            seg_indptr[i] = i * m
        seg_indptr[num_groups] = num_groups * m
        weight_indices = torch.arange(num_groups, device="cuda", dtype=torch.int)

        quantiles = [0.5, 0.2, 0.8]

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: normal_grouped_gemm_triton(
                x,
                y,
                out,
                num_groups,
                weight_column_major=True,
                seg_indptr=seg_indptr,
                weight_indices=weight_indices,
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
        default="./configs/benchmark_ops/normal_group_gemm/",
        help="Path to save normal group gemm benchmark results",
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
