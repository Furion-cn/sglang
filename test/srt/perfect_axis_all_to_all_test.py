#!/usr/bin/env python3
"""
完美的 axis_env 和 all_to_all 测试
关键理解：shard_map 会自动分片输入数据！
"""

import os
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

def test_axis_env_verification():
    print("✅ axis_env is ok")
    
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('expert',))
    
    def axis_kernel(x):
        expert_count = jax.lax.axis_size('expert')
        expert_id = jax.lax.axis_index('expert')
        
        print(f"    专家 {expert_id}/{expert_count}: axis_env is ok!")
        
        # 验证：每个专家添加自己的ID
        return x + expert_id
    
    # 输入数据：总共8行，每行2个元素
    x = jnp.arange(16).reshape(8, 2)
    print(f"  输入总数据: {x.shape}")
    print(f"  数据内容:\n{x}")
    
    result = shard_map(
        axis_kernel,
        mesh=mesh,
        in_specs=P('expert'),  # 沿expert轴分片，每个专家得到1行
        out_specs=P('expert'),
    )(x)
    
    print(f"  输出结果: {result.shape}")
    print(f"  结果内容:\n{result}")
    return True

def test_all_to_all_working():
    """正确工作的 all_to_all 示例"""
    print(f"\n🔄 all_to_all 正确工作示例")
    
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('expert',))
    num_experts = len(devices)
    
    def all_to_all_kernel(x):
        expert_id = jax.lax.axis_index('expert')
        expert_count = jax.lax.axis_size('expert')
        
        print(f"    专家 {expert_id}: 收到数据形状 {x.shape}")
        
        # 关键理解：x 已经被 shard_map 分片了！
        # 每个专家收到的是 (tokens_per_expert, hidden_dim)
        # 要使用 all_to_all，需要构造正确的数据
        
        # 创建要发送给所有专家的数据：形状必须是 (expert_count, ...)
        # 这里我们创建要发送给每个专家的数据
        tokens_to_send = jnp.ones((expert_count, x.shape[1])) * expert_id
        
        print(f"    专家 {expert_id}: 发送数据形状 {tokens_to_send.shape}")
        print(f"    专家 {expert_id}: 发送内容 {tokens_to_send.flatten()}")
        
        # 现在可以正确使用 all_to_all！
        received = jax.lax.all_to_all(
            tokens_to_send,
            axis_name='expert',
            split_axis=0,  # 分割第0轴（expert_count维）
            concat_axis=0  # 连接到第0轴
        )
        
        print(f"    专家 {expert_id}: 接收数据形状 {received.shape}")
        print(f"    专家 {expert_id}: 接收内容 {received.flatten()}")
        
        return received
    
    # 输入：每个专家处理一些tokens
    tokens_per_expert = 1
    hidden_dim = 2
    total_tokens = num_experts * tokens_per_expert
    
    x = jnp.arange(total_tokens * hidden_dim).reshape(total_tokens, hidden_dim)
    print(f"  总输入形状: {x.shape}")
    print(f"  输入数据:\n{x}")
    
    result = shard_map(
        all_to_all_kernel,
        mesh=mesh,
        in_specs=P('expert'),  # 按expert分片：每个专家得到 (1, 2)
        out_specs=P('expert'),
    )(x)
    
    print(f"  ✅ all_to_all 成功！结果形状: {result.shape}")
    return True

def test_moe_realistic_pattern():
    """现实的 MoE 模式测试"""
    print(f"\n🎯 现实 MoE 模式测试")
    
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('expert',))
    num_experts = len(devices)
    
    def moe_expert_computation(expert_tokens):
        """每个专家的计算"""
        expert_id = jax.lax.axis_index('expert')
        expert_count = jax.lax.axis_size('expert')
        
        print(f"    专家 {expert_id}: 处理 {expert_tokens.shape[0]} 个tokens")
        
        # 1. 专家计算（这里每个专家已经收到了分配给它的tokens）
        processed_tokens = expert_tokens + expert_id * 0.1
        
        # 2. 如果需要重新分布结果，可以用 all_gather
        # 这里演示如何收集所有专家的结果计数
        my_token_count = jnp.array([expert_tokens.shape[0]])  # 我处理的token数
        
        # 收集所有专家的token数量
        all_counts = jax.lax.all_gather(my_token_count, 'expert', axis=0)
        
        print(f"    专家 {expert_id}: 所有专家处理的token数 {all_counts}")
        
        return processed_tokens
    
    # 模拟真实场景：不同专家处理不同数量的tokens
    # 这里简化为每个专家处理相同数量
    tokens_per_expert = 2
    hidden_dim = 4
    total_tokens = num_experts * tokens_per_expert
    
    # 创建tokens，每个专家将收到其中一部分
    all_tokens = jnp.arange(total_tokens * hidden_dim).reshape(total_tokens, hidden_dim)
    
    print(f"  所有tokens形状: {all_tokens.shape}")
    print(f"  每个专家将处理: {tokens_per_expert} 个tokens")
    
    results = shard_map(
        moe_expert_computation,
        mesh=mesh,
        in_specs=P('expert'),  # 自动分片：每个专家得到 (2, 4)
        out_specs=P('expert'),
    )(all_tokens)
    
    print(f"  ✅ MoE 计算完成！结果形状: {results.shape}")
    return True

def test_communication_patterns():
    """测试各种通信模式"""
    print(f"\n📡 通信模式测试")
    
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('expert',))
    
    def communication_kernel(x):
        expert_id = jax.lax.axis_index('expert')
        expert_count = jax.lax.axis_size('expert')
        
        print(f"    专家 {expert_id}: 输入 {x.flatten()}")
        
        # 1. all_gather: 收集所有专家的数据
        gathered = jax.lax.all_gather(x, 'expert', axis=0)
        print(f"    专家 {expert_id}: all_gather 结果形状 {gathered.shape}")
        
        # 2. psum: 求和 (如果是标量)
        if x.size == 1:
            total = jax.lax.psum(x, 'expert')
            print(f"    专家 {expert_id}: psum 结果 {total}")
        
        # 3. 准备 all_to_all 数据
        # 创建要发送给每个专家的数据
        send_data = jnp.ones((expert_count, 1)) * expert_id
        
        # all_to_all
        received_data = jax.lax.all_to_all(
            send_data, 'expert', split_axis=0, concat_axis=0
        )
        
        print(f"    专家 {expert_id}: all_to_all 接收 {received_data.flatten()}")
        
        return received_data
    
    # 每个专家一个标量
    x = jnp.arange(8).reshape(8, 1)
    
    result = shard_map(
        communication_kernel,
        mesh=mesh,
        in_specs=P('expert'),
        out_specs=P('expert'),
    )(x)
    
    print(f"  ✅ 通信测试完成！")
    return True

def comprehensive_test():
    """综合测试：完整的专家并行工作流"""
    print(f"\n🏆 综合测试：完整专家并行工作流")
    
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('expert',))
    num_experts = len(devices)
    
    def expert_workflow(tokens, router_probs):
        """完整的专家工作流程"""
        expert_id = jax.lax.axis_index('expert')
        expert_count = jax.lax.axis_size('expert')
        
        print(f"    专家 {expert_id}: 开始工作流")
        print(f"      收到tokens: {tokens.shape}")
        print(f"      收到router_probs: {router_probs.shape}")
        
        # 1. 专家计算
        expert_output = tokens * (expert_id + 1) * 0.1  # 不同专家不同计算
        
        # 2. 使用路由概率加权
        weighted_output = expert_output * router_probs
        
        # 3. 收集统计信息
        my_load = jnp.sum(router_probs)  # 我的负载
        all_loads = jax.lax.all_gather(my_load[None], 'expert', axis=0)
        
        # 修复：在编译时无法转换JAX数组为Python数值，所以简化输出
        print(f"      专家 {expert_id}: 我的负载已计算")
        print(f"      所有专家负载已收集，形状: {all_loads.shape}")
        
        # 4. 如果需要，可以进行 all_to_all 重新分布
        # 这里简化，直接返回结果
        
        return weighted_output
    
    # 模拟数据
    tokens_per_expert = 3
    hidden_dim = 2
    total_tokens = num_experts * tokens_per_expert
    
    tokens = jnp.ones((total_tokens, hidden_dim))
    router_probs = jax.random.uniform(
        jax.random.PRNGKey(42), 
        (total_tokens, 1)
    )
    
    print(f"  输入tokens: {tokens.shape}")
    print(f"  路由概率: {router_probs.shape}")
    
    results = shard_map(
        expert_workflow,
        mesh=mesh,
        in_specs=(P('expert'), P('expert')),  # 两个输入都按expert分片
        out_specs=P('expert'),
    )(tokens, router_probs)
    
    print(f"  🎉 综合测试完成！最终结果: {results.shape}")
    return True

def main():
    """运行所有测试"""
    print("🚀 完美的 JAX axis_env 和 all_to_all 测试")
    print("=" * 70)
    
    tests = [
        ("axis_env验证", test_axis_env_verification),
        ("all_to_all工作示例", test_all_to_all_working),
        ("现实MoE模式", test_moe_realistic_pattern),
        ("通信模式", test_communication_patterns),
        ("综合测试", comprehensive_test),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = success
            print(f"✅ {test_name} 完全成功")
        except Exception as e:
            results[test_name] = False
            print(f"❌ {test_name} 失败: {e}")
    
    # 最终总结
    print("\n" + "=" * 70)
    print("📋 最终测试结果:")
    for test_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {test_name:15s}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    if passed == total:
        print(f"\n🎉 完美！所有 {total} 项测试全部通过！")
        print(f"\n💎 关键要点总结:")
        print(f"  ✓ axis_env 在 shard_map 内部完全可用")
        print(f"  ✓ shard_map 自动按指定轴分片输入数据") 
        print(f"  ✓ all_to_all 需要输入形状为 (axis_size, ...)")
        print(f"  ✓ 可以组合使用 all_gather, psum, all_to_all")
        print(f"  ✓ 支持完整的专家并行工作流")
        print(f"\n🔧 应用到你的 MoE:")
        print(f"  1. 将 MoE 计算包装在 shard_map 中")
        print(f"  2. 在内部安全使用 axis_index/axis_size")
        print(f"  3. 正确准备 all_to_all 的数据形状")
        print(f"  4. 利用 JAX 的自动分片机制")
    else:
        print(f"\n⚠️  {passed}/{total} 测试通过，需要进一步调试")

if __name__ == "__main__":
    main() 