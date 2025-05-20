import unittest
import torch
import numpy as np
from sglang.srt.layers.moe.ep_moe.kernels import grouped_gemm_triton
from sglang.srt.layers.moe.ep_moe.kernels import grouped_gemm_masked_triton

class TestGroupedGemmTriton(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 小型测试用例，方便调试
        self.batch_size = 2  # 专家数量
        self.seq_len = 16    # 序列长度
        self.hidden_size = 4 # 隐藏层大小
        self.out_size = 8    # 输出大小
        
        # 创建可控的输入数据
        self._create_deterministic_inputs()
    
    def _create_deterministic_inputs(self):
        """创建确定性输入，使结果可预测和可重现"""
        # 输入张量 a: (batch_size * seq_per_batch, hidden_size)
        # 每个专家处理不同数量的序列
        self.seq_per_batch = [6, 10]  # 每个专家处理的序列数
        total_seqs = sum(self.seq_per_batch)
        
        # 初始化输入数据
        self.a = torch.zeros(total_seqs, self.hidden_size, device=self.device)
        for i in range(total_seqs):
            for j in range(self.hidden_size):
                self.a[i, j] = 0.1 * (i + 1) * (j + 1)
        
        # 权重张量 b: (batch_size, out_size, hidden_size)
        self.b = torch.zeros(self.batch_size, self.out_size, self.hidden_size, device=self.device)
        for i in range(self.batch_size):
            for j in range(self.out_size):
                for k in range(self.hidden_size):
                    self.b[i, j, k] = 0.01 * (i + 1) * (j + 1) * (k + 1)
        
        # 创建 seg_indptr，指示每个专家处理的序列范围
        self.seg_indptr = torch.zeros(self.batch_size + 1, device=self.device, dtype=torch.int64)
        for i in range(self.batch_size):
            self.seg_indptr[i + 1] = self.seg_indptr[i] + self.seq_per_batch[i]
        
        # 创建 weight_indices，指示每个序列对应的权重索引
        self.weight_indices = torch.arange(self.batch_size, device=self.device, dtype=torch.int64)
        
        # 转换为 bfloat16 以匹配实际使用场景
        self.a = self.a.to(torch.bfloat16)
        self.b = self.b.to(torch.bfloat16)
        
        # 预分配输出张量
        self.c = torch.zeros(total_seqs, self.out_size, device=self.device, dtype=torch.bfloat16)
        
        # 打印输入信息供调试
        print(f"a shape: {self.a.shape}, dtype: {self.a.dtype}")
        print(f"b shape: {self.b.shape}, dtype: {self.b.dtype}")
        print(f"c shape: {self.c.shape}, dtype: {self.c.dtype}")
        print(f"seg_indptr: {self.seg_indptr}")
        print(f"weight_indices: {self.weight_indices}")
    
    def test_grouped_gemm_basic(self):
        """基本功能测试 - 验证 grouped_gemm_triton 的基本计算正确性"""
        # 使用 grouped_gemm_triton 计算
        result = grouped_gemm_triton(
            a=self.a,
            b=self.b,
            c=self.c.clone(),
            batch_size=self.batch_size,
            weight_column_major=True,
            seg_indptr=self.seg_indptr,
            weight_indices=self.weight_indices,
            use_fp8_w8a8=False,
            scale_a=None,
            scale_b=None,
        )
        
        # 计算参考结果
        reference = torch.zeros_like(result)
        start_idx = 0
        for i in range(self.batch_size):
            # 每个专家处理的序列范围
            start = self.seg_indptr[i].item()
            end = self.seg_indptr[i+1].item()
            # 获取当前专家的输入和权重
            expert_input = self.a[start:end]
            expert_weight = self.b[i]
            # 计算输出
            expert_output = torch.matmul(expert_input, expert_weight.t())
            # 存储到参考结果
            reference[start:end] = expert_output
        
        # 打印部分结果进行比较
        print("First few values of grouped_gemm_triton output:")
        print(result[0, 0:2])
        print("First few values of reference output:")
        print(reference[0, 0:2])
        
        # 计算误差
        abs_diff = torch.abs(result - reference)
        max_diff = torch.max(abs_diff)
        mean_diff = torch.mean(abs_diff)
        rel_diff = max_diff / (torch.max(torch.abs(reference)) + 1e-8)
        
        print(f"Maximum absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")
        print(f"Relative difference: {rel_diff}")
        
        # 对于 bfloat16，使用适当的容差
        self.assertTrue(rel_diff < 1e-2, f"Relative difference too large: {rel_diff}")
    
    def test_grouped_gemm_with_mask_m(self):
        """测试使用 mask_m 参数而不是 seg_indptr"""
        # 从 seg_indptr 计算 mask_m
        masked_m = self.seg_indptr[1:] - self.seg_indptr[:-1]
        
        # 使用 mask_m 计算
        result_with_mask = grouped_gemm_triton(
            a=self.a,
            b=self.b,
            c=None,  # 测试自动创建输出
            batch_size=self.batch_size,
            weight_column_major=True,
            seg_indptr=None,  # 不提供 seg_indptr
            weight_indices=self.weight_indices,
            masked_m=masked_m,  # 提供 mask_m
            use_fp8_w8a8=False,
            scale_a=None,
            scale_b=None,
            c_dtype=torch.bfloat16
        )
        
        # 使用 seg_indptr 计算
        result_with_seg = grouped_gemm_triton(
            a=self.a,
            b=self.b,
            c=None,
            batch_size=self.batch_size,
            weight_column_major=True,
            seg_indptr=self.seg_indptr,
            weight_indices=self.weight_indices,
            masked_m=None,
            use_fp8_w8a8=False,
            scale_a=None,
            scale_b=None,
            c_dtype=torch.bfloat16
        )
        
        # 验证两种方法的结果是否一致
        self.assertTrue(torch.allclose(result_with_mask, result_with_seg, atol=1e-4))
    
    def test_different_sequence_lengths(self):
        """测试不同序列长度的场景"""
        # 创建不同长度的序列
        seq_lengths = [4, 8]  # 两个专家处理不同长度的序列
        total_seqs = sum(seq_lengths)
        
        # 创建输入
        a = torch.ones(total_seqs, self.hidden_size, device=self.device, dtype=torch.bfloat16)
        
        # 创建 seg_indptr
        seg_indptr = torch.zeros(self.batch_size + 1, device=self.device, dtype=torch.int64)
        for i in range(self.batch_size):
            seg_indptr[i + 1] = seg_indptr[i] + seq_lengths[i]
        
        # 计算结果
        result = grouped_gemm_triton(
            a=a,
            b=self.b,
            c=None,
            batch_size=self.batch_size,
            weight_column_major=True,
            seg_indptr=seg_indptr,
            weight_indices=self.weight_indices,
            use_fp8_w8a8=False,
            scale_a=None,
            scale_b=None,
            c_dtype=torch.bfloat16
        )
        
        # 验证输出形状
        self.assertEqual(result.shape, (total_seqs, self.out_size))
        
        # 计算参考结果
        reference = torch.zeros_like(result)
        for i in range(self.batch_size):
            start = seg_indptr[i].item()
            end = seg_indptr[i+1].item()
            expert_input = a[start:end]
            expert_weight = self.b[i]
            expert_output = torch.matmul(expert_input, expert_weight.t())
            reference[start:end] = expert_output
        
        # 验证结果
        rel_diff = torch.max(torch.abs(result - reference)) / torch.max(torch.abs(reference))
        self.assertTrue(rel_diff < 1e-2, f"Relative difference too large: {rel_diff}")

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试某个专家没有输入的情况
        seq_lengths = [6, 0]  # 第二个专家没有输入
        total_seqs = sum(seq_lengths)
        
        # 创建输入
        a = torch.ones(total_seqs, self.hidden_size, device=self.device, dtype=torch.bfloat16)
        
        # 创建 seg_indptr
        seg_indptr = torch.zeros(self.batch_size + 1, device=self.device, dtype=torch.int64)
        for i in range(self.batch_size):
            seg_indptr[i + 1] = seg_indptr[i] + seq_lengths[i]
        
        # 计算结果 - 应该不会出错
        result = grouped_gemm_triton(
            a=a,
            b=self.b,
            c=None,
            batch_size=self.batch_size,
            weight_column_major=True,
            seg_indptr=seg_indptr,
            weight_indices=self.weight_indices,
            use_fp8_w8a8=False,
            scale_a=None,
            scale_b=None,
            c_dtype=torch.bfloat16
        )
        
        # 验证输出形状
        self.assertEqual(result.shape, (total_seqs, self.out_size))
        
        # 第一个专家的输出应该有值，第二个专家没有输出
        self.assertTrue(torch.all(result[:seq_lengths[0]] != 0))
        
        # 测试单专家情况
        single_batch_result = grouped_gemm_triton(
            a=self.a[:self.seq_per_batch[0]],
            b=self.b[:1],
            c=None,
            batch_size=1,
            weight_column_major=True,
            seg_indptr=self.seg_indptr[:2],
            weight_indices=self.weight_indices[:1],
            use_fp8_w8a8=False,
            scale_a=None,
            scale_b=None,
            c_dtype=torch.bfloat16
        )
        
        # 验证单专家输出
        self.assertEqual(single_batch_result.shape, (self.seq_per_batch[0], self.out_size))

class TestGroupedGemmMaskedTriton(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            self.skipTest("需要CUDA设备来运行此测试")
            
        # 设置随机种子以获得可重复的结果
        torch.manual_seed(42)
        
        # 小型测试用例，方便调试
        self.num_experts = 2      # 专家数量
        self.seq_len = 3         # 每个专家的最大序列长度
        self.hidden_size = 4     # 隐藏层大小
        self.intermediate_size = 5  # 输出大小
        
        # 创建可控的输入数据
        self._create_deterministic_inputs()
    
    def _create_deterministic_inputs(self):
        """创建确定性输入，使结果可预测和可重现"""
        # 创建输入张量 a: [num_experts, seq_len, hidden_size]
        # 注意：现在使用原始三维形状，不需要重塑
        self.a = torch.empty(self.num_experts, self.seq_len, self.hidden_size, device=self.device)
        for i in range(self.num_experts):
            for j in range(self.seq_len):
                for k in range(self.hidden_size):
                    self.a[i, j, k] = 1 * (i + 1) * (j + 1) * (k + 1)
        
        # 创建权重张量 b: [num_experts, intermediate_size, hidden_size]
        self.b = torch.empty(self.num_experts, self.intermediate_size, self.hidden_size, device=self.device)
        for i in range(self.num_experts):
            for j in range(self.intermediate_size):
                for k in range(self.hidden_size):
                    self.b[i, j, k] = 1 * (i + 1) * (j + 1) * (k + 1)
        
        # 创建每个专家处理的有效token数
        self.masked_m = torch.tensor([1, 1], device=self.device)  # 每个专家处理的有效token数
        
        # 转换为 bfloat16 以匹配实际使用场景
        self.a = self.a.to(torch.bfloat16)
        self.b = self.b.to(torch.bfloat16)
        
        # 预分配输出张量 - 现在维度为[num_experts, seq_len, intermediate_size]
        self.c = torch.empty(self.num_experts, self.seq_len, self.intermediate_size, 
                            device=self.device, dtype=torch.bfloat16)
        
        # 打印输入信息供调试
        print(f"a shape: {self.a.shape}, dtype: {self.a.dtype}")
        print(f"b shape: {self.b.shape}, dtype: {self.b.dtype}")
        print(f"c shape: {self.c.shape}, dtype: {self.c.dtype}")
        print(f"masked_m: {self.masked_m}")
    
    def test_basic_functionality(self):
        """基本功能测试 - 验证 grouped_gemm_masked_triton 的基本计算正确性"""
        # 使用 grouped_gemm_masked_triton 计算，直接使用3D张量
        result = grouped_gemm_masked_triton(
            a=self.a,  # 直接使用3D张量
            b=self.b,
            c=self.c.clone(),
            masked_m=self.masked_m,
            c_dtype=torch.bfloat16
        )
        
        # 计算参考结果
        reference = torch.zeros_like(result)
        
        for i in range(self.num_experts):
            # 只处理有效的token
            for j in range(self.masked_m[i].item()):
                # 对每个有效token进行矩阵乘法
                expert_output = torch.matmul(
                    self.a[i, j].to(torch.float32), 
                    self.b[i].transpose(0, 1).to(torch.float32)  # 转置为[hidden_size, intermediate_size]
                ).to(torch.bfloat16)
                
                reference[i, j] = expert_output
        
        # 打印部分结果进行比较
        print("First few values of grouped_gemm_masked_triton output:")
        print(result[0, 0, 0:2])
        print("First few values of reference output:")
        print(reference[0, 0, 0:2])

        print(f"{result=}")
        print(f"{reference=}")
        
        # 创建掩码，只比较有效的token
        valid_mask = torch.zeros((self.num_experts, self.seq_len), dtype=torch.bool, device=self.device)
        for i in range(self.num_experts):
            valid_mask[i, :self.masked_m[i]] = True
        
        # 只比较有效的token
        result_valid = result[valid_mask]
        reference_valid = reference[valid_mask]
        # 计算误差
        abs_diff = torch.abs(result_valid - reference_valid)
        max_diff = torch.max(abs_diff)
        mean_diff = torch.mean(abs_diff)
        rel_diff = max_diff / (torch.max(torch.abs(reference_valid)) + 1e-8)
        
        print(f"Maximum absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")
        print(f"Relative difference: {rel_diff}")
        
        # 对于 bfloat16，使用适当的容差
        self.assertTrue(rel_diff < 1e-2, f"Relative difference too large: {rel_diff}")
    
    def test_without_weight_indices(self):
        """测试不提供 weight_indices 的情况"""
        # 使用 grouped_gemm_masked_triton 计算，不提供 weight_indices
        result = grouped_gemm_masked_triton(
            a=self.a,
            b=self.b,
            c=None,
            masked_m=self.masked_m,
            c_dtype=torch.bfloat16
        )
        
        # 计算参考结果
        reference = torch.zeros_like(result)
        
        for i in range(self.num_experts):
            # 只处理有效的token
            for j in range(self.masked_m[i].item()):
                # 对每个有效token进行矩阵乘法
                expert_output = torch.matmul(
                    self.a[i, j].to(torch.float32), 
                    self.b[i].transpose(0, 1).to(torch.float32)  # 转置为[hidden_size, intermediate_size]
                ).to(torch.bfloat16)
                
                reference[i, j] = expert_output
        
        # 创建掩码，只比较有效的token
        valid_mask = torch.zeros((self.num_experts, self.seq_len), dtype=torch.bool, device=self.device)
        for i in range(self.num_experts):
            valid_mask[i, :self.masked_m[i]] = True
        
        # 只比较有效的token
        result_valid = result[valid_mask]
        reference_valid = reference[valid_mask]
        
        # 计算误差
        rel_diff = torch.max(torch.abs(result_valid - reference_valid)) / torch.max(torch.abs(reference_valid))
        
        print(f"Without weight_indices - Relative difference: {rel_diff}")
        self.assertTrue(rel_diff < 1e-2, f"Relative difference too large: {rel_diff}")
    
    def test_with_padding(self):
        """测试带有padding的情况"""
        # 创建带有显式padding的输入
        a_padded = self.a.clone()
        
        # 为每个专家添加padding（将超出masked_m的部分设为0）
        for i in range(self.num_experts):
            a_padded[i, self.masked_m[i]:] = 0.0
        
        # 使用 grouped_gemm_masked_triton 计算
        result = grouped_gemm_masked_triton(
            a=a_padded,
            b=self.b,
            c=None,
            masked_m=self.masked_m,
            c_dtype=torch.bfloat16
        )
        
        # 计算参考结果
        reference = torch.zeros_like(result)
        
        for i in range(self.num_experts):
            # 只处理有效的token
            for j in range(self.masked_m[i].item()):
                # 对每个有效token进行矩阵乘法
                expert_output = torch.matmul(
                    a_padded[i, j].to(torch.float32), 
                    self.b[i].transpose(0, 1).to(torch.float32)  # 转置为[hidden_size, intermediate_size]
                ).to(torch.bfloat16)
                
                reference[i, j] = expert_output
        
        # 创建掩码，只比较有效的token
        valid_mask = torch.zeros((self.num_experts, self.seq_len), dtype=torch.bool, device=self.device)
        for i in range(self.num_experts):
            valid_mask[i, :self.masked_m[i]] = True
        
        # 只比较有效的token
        result_valid = result[valid_mask]
        reference_valid = reference[valid_mask]
        
        # 计算误差
        rel_diff = torch.max(torch.abs(result_valid - reference_valid)) / torch.max(torch.abs(reference_valid))
        
        print(f"With padding - Relative difference: {rel_diff}")
        self.assertTrue(rel_diff < 1e-2, f"Relative difference too large: {rel_diff}")
    
    def test_extreme_padding(self):
        """测试极端padding情况，有些专家没有有效token"""
        # 创建极端的masked_m值，包括0（表示该专家没有有效token）
        extreme_masked_m = torch.tensor([2, 0], device=self.device)  # 修改为与专家数量匹配
        
        # 创建带有显式padding的输入
        a_padded = self.a.clone()
        
        # 为每个专家添加padding（将超出masked_m的部分设为0）
        for i in range(self.num_experts):
            if extreme_masked_m[i] > 0:
                a_padded[i, extreme_masked_m[i]:] = 0.0
            else:
                a_padded[i, :] = 0.0
        
        # 使用 grouped_gemm_masked_triton 计算
        result = grouped_gemm_masked_triton(
            a=a_padded,
            b=self.b,
            c=None,
            masked_m=extreme_masked_m,
            c_dtype=torch.bfloat16
        )
        
        # 计算参考结果
        reference = torch.zeros_like(result)
        
        for i in range(self.num_experts):
            # 只处理有效的token
            for j in range(extreme_masked_m[i].item()):
                # 对每个有效token进行矩阵乘法
                expert_output = torch.matmul(
                    a_padded[i, j].to(torch.float32), 
                    self.b[i].transpose(0, 1).to(torch.float32)  # 转置为[hidden_size, intermediate_size]
                ).to(torch.bfloat16)
                
                reference[i, j] = expert_output
        
        # 创建掩码，只比较有效的token
        valid_mask = torch.zeros((self.num_experts, self.seq_len), dtype=torch.bool, device=self.device)
        for i in range(self.num_experts):
            valid_mask[i, :extreme_masked_m[i]] = True
        
        # 只比较有效的token
        if valid_mask.any():
            result_valid = result[valid_mask]
            reference_valid = reference[valid_mask]
            
            # 计算误差
            rel_diff = torch.max(torch.abs(result_valid - reference_valid)) / torch.max(torch.abs(reference_valid))
            
            print(f"Extreme padding - Relative difference: {rel_diff}")
            self.assertTrue(rel_diff < 1e-2, f"Relative difference too large: {rel_diff}")
        else:
            print("Warning: No valid tokens to compare")
    
    def test_custom_weight_indices(self):
        """测试自定义的weight_indices"""
        # 创建更多的权重矩阵
        num_weights = 5  # 权重数量大于专家数量
        custom_b = torch.zeros(num_weights, self.intermediate_size, self.hidden_size, 
                              device=self.device, dtype=torch.bfloat16)
        
        # 初始化权重 - 注意维度顺序已修正
        for i in range(num_weights):
            for j in range(self.intermediate_size):
                for k in range(self.hidden_size):
                    custom_b[i, j, k] = 0.01 * (i + 1) * (j + 1) * (k + 1)
        
        # 创建自定义weight_indices，将每个专家映射到不同的权重
        custom_weight_indices = torch.tensor([2, 0], device=self.device)  # 修改为与专家数量匹配
        
        # 使用 grouped_gemm_masked_triton 计算
        result = grouped_gemm_masked_triton(
            a=self.a,
            b=custom_b,
            c=None,
            masked_m=self.masked_m,
            c_dtype=torch.bfloat16
        )
        
        # 计算参考结果
        reference = torch.zeros_like(result)
        
        for i in range(self.num_experts):
            # 使用映射的权重ID
            weight_idx = custom_weight_indices[i].item()
            
            # 只处理有效的token
            for j in range(self.masked_m[i].item()):
                # 对每个有效token进行矩阵乘法
                expert_output = torch.matmul(
                    self.a[i, j].to(torch.float32), 
                    custom_b[weight_idx].transpose(0, 1).to(torch.float32)  # 转置为[hidden_size, intermediate_size]
                ).to(torch.bfloat16)
                
                reference[i, j] = expert_output
        
        # 创建掩码，只比较有效的token
        valid_mask = torch.zeros((self.num_experts, self.seq_len), dtype=torch.bool, device=self.device)
        for i in range(self.num_experts):
            valid_mask[i, :self.masked_m[i]] = True
        
        # 只比较有效的token
        result_valid = result[valid_mask]
        reference_valid = reference[valid_mask]
        
        # 计算误差
        rel_diff = torch.max(torch.abs(result_valid - reference_valid)) / torch.max(torch.abs(reference_valid))
        
        print(f"Custom weight_indices - Relative difference: {rel_diff}")
        self.assertTrue(rel_diff < 1e-2, f"Relative difference too large: {rel_diff}")
    
    def test_performance(self):
        """测试性能"""
        # 创建更大的输入用于性能测试
        batch_size = 8
        seq_len = 128
        hidden_size = 1024
        intermediate_size = 4096
        
        # 创建输入张量 - 注意维度顺序
        a_large = torch.randn(batch_size, seq_len, hidden_size, device=self.device, dtype=torch.bfloat16)
        b_large = torch.randn(batch_size, intermediate_size, hidden_size, device=self.device, dtype=torch.bfloat16)
        masked_m_large = torch.full((batch_size,), seq_len, device=self.device)
        
        # 预热
        for _ in range(5):
            _ = grouped_gemm_masked_triton(
                a=a_large,
                b=b_large,
                c=None,
                masked_m=masked_m_large,
                c_dtype=torch.bfloat16
            )
        
        # 测量triton版本的性能
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        iterations = 10
        
        start.record()
        for _ in range(iterations):
            _ = grouped_gemm_masked_triton(
                a=a_large,
                b=b_large,
                c=None,
                masked_m=masked_m_large,
                c_dtype=torch.bfloat16
            )
        end.record()
        
        torch.cuda.synchronize()
        triton_time = start.elapsed_time(end) / iterations
        
        # 测量PyTorch版本的性能
        start.record()
        for _ in range(iterations):
            c_ref = torch.zeros(batch_size, seq_len, intermediate_size, device=self.device, dtype=torch.bfloat16)
            
            for i in range(batch_size):
                for j in range(seq_len):
                    c_ref[i, j] = torch.matmul(
                        a_large[i, j].to(torch.float32), 
                        b_large[i].transpose(0, 1).to(torch.float32)
                    ).to(torch.bfloat16)
        end.record()
        
        torch.cuda.synchronize()
        pytorch_time = start.elapsed_time(end) / iterations
        
        speedup = pytorch_time / triton_time
        print(f"Performance test - Triton: {triton_time:.3f} ms, PyTorch: {pytorch_time:.3f} ms, Speedup: {speedup:.2f}x")
        
        # 不强制要求加速，但打印结果以供参考
        self.assertTrue(True, "Performance test completed")

if __name__ == "__main__":
    unittest.main()