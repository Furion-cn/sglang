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
        self.seq_len = 12
        self.hidden_size = 4
        self.intermediate_size = 5
        self.num_experts = 2
        
        # 创建 DeepEPMoE 实例
        self.model = DeepEPMoE(
            layer_id=0,
            num_experts=self.num_experts,
            top_k=1,
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
            self.model.w13_weight.data = torch.zeros(self.num_experts, 2 * self.intermediate_size, self.hidden_size, device=self.device, dtype=torch.bfloat16)
            for i in range(self.num_experts):
                for j in range(2 * self.intermediate_size):
                    for k in range(self.hidden_size):
                        self.model.w13_weight.data[i, j, k] = 1 * (i + 1) * (j + 1) * (k + 1)
            
            self.model.w2_weight.data = torch.zeros(self.num_experts, self.hidden_size, self.intermediate_size, device=self.device, dtype=torch.bfloat16)
            for i in range(self.num_experts):
                for j in range(self.hidden_size):
                    for k in range(self.intermediate_size):
                        self.model.w2_weight.data[i, j, k] = 1 * (i + 1) * (j + 1) * (k + 1)
    
    def test_forward_masked_with_runner(self):
        # 创建输入数据
        masked_hidden_states = torch.empty(self.num_experts, self.seq_len, self.hidden_size, device=self.device, dtype=torch.bfloat16)
        
        # 创建 masked_m 张量，表示每个专家处理的序列长度
        masked_m = torch.tensor([4, 6], dtype=torch.int32, device=self.device)
        
        # 计算总有效 token 数
        total_tokens = sum(masked_m.tolist())
        
        # 为 normal 模式准备输入，只包含有效的 token
        normal_hidden_states = torch.empty(total_tokens, self.hidden_size, device=self.device, dtype=torch.bfloat16)
        
        # 准备 reorder_topk_ids 和 seg_indptr
        reorder_topk_ids = torch.zeros(total_tokens, dtype=torch.int64, device=self.device)
        seg_indptr = torch.zeros(self.num_experts + 1, dtype=torch.int64, device=self.device)
        
        # 填充数据
        token_idx = 0
        for i in range(self.num_experts):
            for j in range(self.seq_len):
                if j < masked_m[i]:
                    for k in range(self.hidden_size):
                        masked_hidden_states[i, j, k] = 1.0 * (i + 1) * (j + 1) * (k + 1)
                        normal_hidden_states[token_idx, k] = 1.0 * (i + 1) * (j + 1) * (k + 1)
                    reorder_topk_ids[token_idx] = i
                    token_idx += 1
            seg_indptr[i + 1] = token_idx
        
        # 打印调试信息
        print(f"masked_hidden_states shape: {masked_hidden_states.shape}, {masked_hidden_states=}")
        print(f"normal_hidden_states shape: {normal_hidden_states.shape}, {normal_hidden_states=}")
        print(f"masked_m: {masked_m}")
        print(f"reorder_topk_ids: {reorder_topk_ids}")
        print(f"seg_indptr: {seg_indptr}")
        
        # 调用被测试的方法
        with torch.no_grad():
            # masked 版本
            output_masked = self.model.forward_masked_with_runner(
                masked_hidden_states, masked_m, None
            )
            
            # normal 版本
            output_normal_flat = self.model.forward_normal(
                normal_hidden_states, reorder_topk_ids, seg_indptr
            )

        print(f"output_normal_flat: {output_normal_flat}")
        
        # 将 normal 输出重组为与 masked 输出相同的形状
        output_normal = torch.zeros_like(output_masked)
        token_idx = 0
        for i in range(self.num_experts):
            for j in range(masked_m[i]):
                output_normal[i, j] = output_normal_flat[token_idx]
                token_idx += 1
        
        # 创建有效区域的掩码
        valid_mask = torch.zeros((self.num_experts, self.seq_len), dtype=torch.bool, device=self.device)
        for i in range(self.num_experts):
            valid_mask[i, :masked_m[i]] = True
        
        # 只比较有效区域的输出
        masked_output_valid = output_masked[valid_mask]
        normal_output_valid = output_normal[valid_mask]
        
        # 计算和打印差异
        abs_diff = torch.abs(masked_output_valid - normal_output_valid)
        max_diff = torch.max(abs_diff)
        mean_diff = torch.mean(abs_diff)
        rel_diff = max_diff / (torch.max(torch.abs(normal_output_valid)) + 1e-8)
        
        print(f"Maximum absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")
        print(f"Relative difference: {rel_diff}")
        
        # # 验证两种方法的输出在有效区域内是否足够接近
        # # 对于 bfloat16，使用较宽松的容差
        # self.assertTrue(rel_diff < 1e-2, f"Relative difference too large: {rel_diff}")
        
        # # 验证 masked 区域外的值是否为零
        # for i, length in enumerate(masked_m):
        #     beyond_mask = output_masked[i, length:, :]
        #     self.assertTrue(torch.allclose(beyond_mask, torch.zeros_like(beyond_mask), atol=1e-5))
        
        # 可选：打印部分结果进行比较
        print("\nMasked output (first few tokens):")
        print(output_masked[0, :, :])  # 第一个专家的前3个token的前3个值
        print(output_masked[1, :, :])  # 第二个专家的前3个token的前3个值
        
        print("\nNormal output (first few tokens):")
        print(output_normal[0, :, :])  # 第一个专家的前3个token的前3个值
        print(output_normal[1, :, :])  # 第二个专家的前3个token的前3个值
    
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