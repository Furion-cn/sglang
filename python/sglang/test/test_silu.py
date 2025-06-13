import unittest
import torch
import triton
from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_masked_fwd

class TestSiluAndMulMaskedFwd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试类初始化，设置设备"""
        cls.device = 'cuda'
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    def setUp(self):
        """每个测试用例前的设置"""
        # 设置随机种子以确保可重复性
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def test_basic_functionality(self):
        """测试基本功能：检查输出形状和数值正确性"""
        # 设置测试参数
        expert_num = 4
        token_num = 32
        hidden_dim = 256

        # 创建输入张量
        input = torch.randn(expert_num, token_num, hidden_dim, dtype=torch.bfloat16, device=self.device)
        # 创建输出张量
        output = torch.empty(expert_num, token_num, hidden_dim // 2, dtype=torch.bfloat16, device=self.device)
        # 创建 masked_m 张量
        masked_m = torch.tensor([token_num] * expert_num, device=self.device)

        # 运行kernel
        silu_and_mul_masked_fwd(input, output, masked_m)

        # 验证输出形状
        self.assertEqual(output.shape, (expert_num, token_num, hidden_dim // 2))
        self.assertEqual(output.dtype, torch.bfloat16)

        # 验证数值正确性（使用CPU进行验证）
        input_cpu = input.cpu().float()
        expected_output = torch.empty_like(output.cpu().float())

        # 手动计算期望结果
        for e in range(expert_num):
            for t in range(token_num):
                gate = input_cpu[e, t, :hidden_dim//2]
                up = input_cpu[e, t, hidden_dim//2:]
                gate = gate * torch.sigmoid(gate)
                expected_output[e, t] = gate * up

        # 比较结果（考虑BF16的精度）
        output_cpu = output.cpu().float()
        self.assertTrue(torch.allclose(output_cpu, expected_output, rtol=1e-2, atol=1e-2))

    def test_different_expert_tokens(self):
        """测试不同专家处理不同数量的token的情况"""
        expert_num = 4
        max_token_num = 32
        hidden_dim = 256

        # 为每个专家设置不同的token数量
        token_nums = [16, 24, 32, 8]
        max_tokens = max(token_nums)

        # 创建输入张量（使用最大token数进行填充）
        input = torch.randn(expert_num, max_tokens, hidden_dim, dtype=torch.bfloat16, device=self.device)
        output = torch.empty(expert_num, max_tokens, hidden_dim // 2, dtype=torch.bfloat16, device=self.device)
        masked_m = torch.tensor(token_nums, device=self.device)

        # 运行kernel
        silu_and_mul_masked_fwd(input, output, masked_m)

        # 验证每个专家的输出
        input_cpu = input.cpu().float()
        output_cpu = output.cpu().float()

        for e, token_num in enumerate(token_nums):
            # 验证有效token的输出
            for t in range(token_num):
                gate = input_cpu[e, t, :hidden_dim//2]
                up = input_cpu[e, t, hidden_dim//2:]
                gate = gate * torch.sigmoid(gate)
                expected = gate * up
                self.assertTrue(torch.allclose(output_cpu[e, t], expected, rtol=1e-2, atol=1e-2))

            # 验证填充token的输出（应该为0）
            for t in range(token_num, max_tokens):
                self.assertTrue(torch.allclose(output_cpu[e, t], torch.zeros_like(output_cpu[e, t])))

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试最小专家数
        expert_num = 1
        token_num = 32
        hidden_dim = 256
        input = torch.randn(expert_num, token_num, hidden_dim, dtype=torch.bfloat16, device=self.device)
        output = torch.empty(expert_num, token_num, hidden_dim // 2, dtype=torch.bfloat16, device=self.device)
        masked_m = torch.tensor([token_num], device=self.device)

        silu_and_mul_masked_fwd(input, output, masked_m)
        self.assertEqual(output.shape, (expert_num, token_num, hidden_dim // 2))

        # 测试最小token数
        expert_num = 4
        token_num = 1
        input = torch.randn(expert_num, token_num, hidden_dim, dtype=torch.bfloat16, device=self.device)
        output = torch.empty(expert_num, token_num, hidden_dim // 2, dtype=torch.bfloat16, device=self.device)
        masked_m = torch.tensor([token_num] * expert_num, device=self.device)

        silu_and_mul_masked_fwd(input, output, masked_m)
        self.assertEqual(output.shape, (expert_num, token_num, hidden_dim // 2))

    def test_input_validation(self):
        """测试输入验证"""
        expert_num = 4
        token_num = 32
        hidden_dim = 256

        # 测试非连续输入
        input = torch.randn(expert_num, token_num, hidden_dim, dtype=torch.bfloat16, device=self.device)
        input = input.transpose(0, 1)  # 使输入不连续
        output = torch.empty(expert_num, token_num, hidden_dim // 2, dtype=torch.bfloat16, device=self.device)
        masked_m = torch.tensor([token_num] * expert_num, device=self.device)

        with self.assertRaises(AssertionError):
            silu_and_mul_masked_fwd(input, output, masked_m)

        # 测试错误的输出类型
        input = torch.randn(expert_num, token_num, hidden_dim, dtype=torch.bfloat16, device=self.device)
        output = torch.empty(expert_num, token_num, hidden_dim // 2, dtype=torch.float32, device=self.device)

        with self.assertRaises(AssertionError):
            silu_and_mul_masked_fwd(input, output, masked_m)

        # 测试错误的hidden_dim
        input = torch.randn(expert_num, token_num, 255, dtype=torch.bfloat16, device=self.device)
        output = torch.empty(expert_num, token_num, 127, dtype=torch.bfloat16, device=self.device)

        with self.assertRaises(AssertionError):
            silu_and_mul_masked_fwd(input, output, masked_m)

    def test_performance(self):
        """测试性能（可选）"""
        expert_num = 8
        token_num = 1024
        hidden_dim = 2048
        num_runs = 100

        # 创建输入张量
        input = torch.randn(expert_num, token_num, hidden_dim, dtype=torch.bfloat16, device=self.device)
        output = torch.empty(expert_num, token_num, hidden_dim // 2, dtype=torch.bfloat16, device=self.device)
        masked_m = torch.tensor([token_num] * expert_num, device=self.device)

        # 预热
        for _ in range(10):
            silu_and_mul_masked_fwd(input, output, masked_m)

        # 同步GPU
        torch.cuda.synchronize()

        # 计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_runs):
            silu_and_mul_masked_fwd(input, output, masked_m)
        end_event.record()

        # 同步GPU
        torch.cuda.synchronize()

        # 计算平均时间
        avg_time = start_event.elapsed_time(end_event) / num_runs
        print(f"\nAverage execution time: {avg_time:.3f} ms")

if __name__ == '__main__':
    unittest.main(verbosity=2)