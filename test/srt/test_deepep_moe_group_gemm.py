import unittest
import torch
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
from sglang.srt.utils import DeepEPMode
from sglang.srt.model_executor.forward_batch_info import ForwardMode

class TestDeepEPMoE(unittest.TestCase):
    def setUp(self):
        # 设置测试环境
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.seq_len = 16
        self.hidden_size = 2
        self.intermediate_size = 4
        self.num_experts = 2

        # 创建 DeepEPMoE 实例
        self.model = DeepEPMoE(
            layer_id=0,
            num_experts=self.num_experts,
            top_k=2,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            tp_size=1,
            tp_rank=0,
            deepep_mode=DeepEPMode.low_latency,
        )

        # 将模型移动到设备上
        self.model.to(self.device)

        # 初始化权重
        with torch.no_grad():
            w13_weight = torch.zeros(self.num_experts, 2 * self.intermediate_size, self.hidden_size, device=self.device)
            for i in range(self.num_experts):
                for j in range(2 * self.intermediate_size):
                    for k in range(self.hidden_size):
                        w13_weight[i, j, k] = 0.1 * (i + 1) * (j + 1) * (k + 1)
            self.model.w13_weight.data = w13_weight.to(torch.bfloat16)

            w2_weight = torch.zeros(self.num_experts, self.hidden_size, self.intermediate_size, device=self.device)
            for i in range(self.num_experts):
                for j in range(self.hidden_size):
                    for k in range(self.intermediate_size):
                        w2_weight[i, j, k] = 0.1 * (i + 1) * (j + 1) * (k + 1)
            self.model.w2_weight.data = w2_weight.to(torch.bfloat16)

    def test_forward_masked_with_runner(self):
        # 创建输入数据
        hidden_states = torch.randn(
            self.num_experts, self.seq_len, self.hidden_size,
            device=self.device,
            dtype=torch.bfloat16
        )

        # 创建 masked_m 张量，表示每个专家处理的序列长度
        # 这里我们假设每个专家处理不同长度的序列
        masked_m = torch.tensor([20, 42, 30, 36], dtype=torch.int32, device=self.device)

        # 计算 expected_m (平均长度 + 1，但不超过最大长度)
        expected_m = min(int(masked_m.float().mean()) + 1, self.seq_len)

        # 调用被测试的方法
        with torch.no_grad():
            output = self.model.forward_masked_with_runner(
                hidden_states, masked_m, expected_m
            )

        print(output, masked_m, expected_m)

        # 验证输出形状
        self.assertEqual(output.shape, (self.num_experts, self.seq_len, self.hidden_size))

        # 验证输出类型
        self.assertEqual(output.dtype, torch.bfloat16)

        # 验证输出不包含 NaN 或 Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

        # 验证 masked 区域外的值是否为零
        # 对于每个专家，检查其处理长度之外的输出是否为零
        for i, length in enumerate(masked_m):
            # 获取当前专家超出处理长度的部分
            beyond_mask = output[i, length:, :]
            # 验证这些值是否全为零
            # 注意：由于浮点精度问题，我们检查它们是否接近零
            self.assertTrue(torch.allclose(beyond_mask, torch.zeros_like(beyond_mask), atol=1e-5))

    def test_forward_integration(self):
        """测试 forward_masked_with_runner 是否能正确集成到 forward 方法中"""
        # 创建输入数据
        hidden_states = torch.randn(
            self.num_experts, self.seq_len, self.hidden_size,
            device=self.device
        )

        # 创建其他必要的输入参数
        topk_idx = torch.randint(0, self.seq_len, (self.batch_size, 2), device=self.device)
        topk_weights = torch.rand(self.batch_size, 2, device=self.device)
        reorder_topk_ids = torch.randint(0, self.num_experts, (self.seq_len,), device=self.device)
        seg_indptr = torch.zeros(self.num_experts + 1, dtype=torch.int32, device=self.device)
        seg_indptr[1:] = torch.cumsum(torch.tensor([20, 42, 30, 36], dtype=torch.int32, device=self.device), dim=0)

        masked_m = torch.tensor([20, 42, 30, 36], dtype=torch.int32, device=self.device)
        expected_m = min(int(masked_m.float().mean()) + 1, self.seq_len)
        num_recv_tokens_per_expert = [20, 42, 30, 36]

        # 调用 forward 方法
        with torch.no_grad():
            output = self.model.forward(
                hidden_states,
                topk_idx,
                topk_weights,
                reorder_topk_ids,
                seg_indptr,
                masked_m,
                expected_m,
                num_recv_tokens_per_expert,
                ForwardMode.DECODE  # 使用 DECODE 模式，这样会解析为 low_latency 模式
            )

        # 验证输出形状
        self.assertEqual(output.shape, (self.num_experts, self.seq_len, self.hidden_size))

        # 验证输出类型
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_grouped_gemm_correctness(self):
        """验证 GroupedGemmRunner 计算结果的正确性"""
        # 创建输入数据
        hidden_states = torch.zeros(self.num_experts, self.seq_len, self.hidden_size, device=self.device, dtype=torch.bfloat16)
        for i in range(self.num_experts):
            for j in range(self.seq_len):
                for k in range(self.hidden_size):
                    hidden_states[i, j, k] = 0.1 * (i + 1) * (j + 1) * (k + 1)

        # 创建 masked_m 张量
        masked_m = torch.tensor([4, 12], dtype=torch.int32, device=self.device)
        expected_m = min(int(masked_m.float().mean()) + 1, self.seq_len)

        # 使用 GroupedGemmRunner 计算
        with torch.no_grad():
            grouped_output = self.model.forward_masked_with_runner(
                hidden_states.clone(), masked_m, expected_m
            )

        # 使用标准矩阵乘法计算参考结果
        reference_output = torch.zeros_like(grouped_output)

        with torch.no_grad():
            # 第一次矩阵乘法
            for i in range(self.num_experts):
                # 只使用有效长度的输入
                valid_input = hidden_states[i, :masked_m[i], :]
                # 第一次矩阵乘法
                gate_up = torch.matmul(valid_input, self.model.w13_weight[i].t())

                # 应用 SiLU 激活函数并分割
                gate, up = gate_up.chunk(2, dim=-1)
                act_output = gate * torch.nn.functional.silu(up)

                # 第二次矩阵乘法
                out = torch.matmul(act_output, self.model.w2_weight[i].t())

                # 将结果放回对应位置
                reference_output[i, :masked_m[i], :] = out

        # 验证两种计算方法的结果是否接近
        # 注意：由于浮点精度和不同算法可能导致的差异，我们使用相对误差
        max_diff = torch.max(torch.abs(grouped_output - reference_output))
        rel_diff = max_diff / (torch.max(torch.abs(reference_output)) + 1e-10)

        print(f"Maximum absolute difference: {max_diff}")
        print(f"Relative difference: {rel_diff}")

        # 验证相对误差在可接受范围内
        # 对于 bfloat16，我们可能需要更宽松的阈值
        self.assertTrue(rel_diff < 1e-2, f"Relative difference too large: {rel_diff}")

        # 验证掩码区域外的值是否为零
        for i, length in enumerate(masked_m):
            beyond_mask = grouped_output[i, length:, :]
            self.assertTrue(torch.allclose(beyond_mask, torch.zeros_like(beyond_mask), atol=1e-5))

if __name__ == "__main__":
    unittest.main()
