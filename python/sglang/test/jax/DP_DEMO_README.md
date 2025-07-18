# 真正的JAX数据并行(Data Parallelism) Demo

这个demo展示了如何在JAX中实现**真正的数据并行**，保持batch和sequence维度分离，然后使用JAX sharding和device_put来分布数据到不同设备，最重要的是使用**pmap实现真正的并行执行**而不是串行for循环。

## 🎯 Demo目标

实现以下**真正的**数据并行流程：
1. **保持batch×sequence二维结构** - 不立即扁平化
2. **JAX sharding** - 使用NamedSharding定义数据分片策略
3. **device_put** - 将所有数据（包括ForwardBatch）分布到各个设备
4. **pmap并行执行** - 在所有设备上同时执行模型前向传播
5. **结果收集** - 汇总各设备的计算结果

## 🚀 快速开始

### 控制参数

使用 `DP_SIZE` 环境变量控制是否启用数据并行：

- **DP_SIZE=1** (默认): 运行标准单设备测试 (`test_load_model_with_jax_loader`)
- **DP_SIZE>1**: 运行数据并行测试 (`test_load_model_with_jax_loader_dp`)

### 🔧 Mesh自动配置

系统会在 `setUp()` 中根据 `DP_SIZE` 自动配置mesh：

```python
# DP_SIZE=1 (标准模式)
mesh = create_device_mesh(ici_parallelism=[1, 4, 1, 1])  # tensor并行

# DP_SIZE=4 (数据并行模式) 
mesh = create_device_mesh(ici_parallelism=[4, 1, 1, 1])  # data并行

# DP_SIZE=2 (混合模式)
mesh = create_device_mesh(ici_parallelism=[2, 2, 1, 1])  # data + tensor并行
```

**axis配置说明**:
- `data`: 数据并行轴，分片batch维度
- `tensor`: 张量并行轴，分片模型参数  
- `pipeline`: 流水线并行轴（暂未使用）
- `expert`: 专家并行轴（MoE专用）

### 运行方式

```bash
# 1. 标准单设备测试 (tensor并行)
python python/sglang/test/jax/run_dp_demo.py
# 或者
DP_SIZE=1 python python/sglang/test/jax/run_dp_demo.py

# 2. 纯数据并行测试 (4个设备)
DP_SIZE=4 python python/sglang/test/jax/run_dp_demo.py

# 3. 混合并行测试 (2x2配置)  
DP_SIZE=2 python python/sglang/test/jax/run_dp_demo.py

# 4. 指定模型路径
DP_SIZE=4 MODEL_PATH=/path/to/jax/qwen3_moe/model python python/sglang/test/jax/run_dp_demo.py
```

### 直接运行unittest

```bash
# 标准测试
python -m unittest python.sglang.test.jax.test_qwen3_moe_load_weights.TestQwen3MoeLoadWeights.test_load_model_with_jax_loader

# 数据并行测试
DP_SIZE=4 python -m unittest python.sglang.test.jax.test_qwen3_moe_load_weights.TestQwen3MoeLoadWeights.test_load_model_with_jax_loader_dp
```

## 🔧 完整功能特性

### 数据并行测试包含完整decode流程

1. **统一的并行函数**: 合并前向传播和采样为一个pmap，减少设备间通信
2. **并行前向传播**: 所有设备同时执行模型推理
3. **并行采样**: 在同一pmap中直接采样下一个token
4. **状态更新**: 并行更新ForwardBatch和序列状态
5. **完整生成**: 多轮迭代生成完整的回答
6. **结果解码**: 显示每个序列的完整生成结果

```python
# 🚀 关键优化：统一的pmap函数
def dp_forward_and_sample(forward_batch, temps, top_ps, top_ks, min_ps):
    """数据并行：前向传播 + 采样 - 在每个设备上同时执行完整流程"""
    # 1. 模型前向传播
    outputs = model(forward_batch.input_ids, forward_batch.positions, forward_batch)
    
    # 2. 直接在同一个pmap中采样 - 无需额外通信！
    local_sampling_info = SamplingBatchInfo(...)
    next_token_ids = sampler(outputs, sampling_info=local_sampling_info)
    
    return outputs, next_token_ids

# 一个pmap搞定所有事情！
dp_forward_sample = jax.pmap(dp_forward_and_sample, axis_name='data')

# 一次调用完成：前向传播 + 采样！
outputs, next_token_ids = dp_forward_sample(
    sharded_forward_batch, sampling_temps, sampling_top_ps, ...
)
```

**性能优化**：
- ✅ **减少pmap调用**: 从2个pmap减少到1个
- ✅ **减少设备通信**: 避免中间结果的设备间传输
- ✅ **提高缓存效率**: 在同一kernel中完成前向+采样
- ✅ **降低延迟**: 消除两次pmap之间的同步开销

## ⚡ 关键改进：真正的并行 vs 假并行

### ❌ 之前的假并行（串行for循环）
```python
# 错误的做法：用for循环串行执行
for device_id, device_batch in enumerate(device_forward_batches):
    with jax.default_device(devices[device_id]):
        output = model(device_batch.input_ids, device_batch.positions, device_batch)
```

### ✅ 现在的真并行（JAX pmap）
```python
# 正确的做法：JAX pmap真正并行执行
@jax.pmap
def dp_model_forward(forward_batch):
    return model(forward_batch.input_ids, forward_batch.positions, forward_batch)

# 所有设备同时执行！
outputs = dp_model_forward(sharded_forward_batch)
```

## 🔧 测试方法架构

### `test_load_model_with_jax_loader` (DP_SIZE=1)
- **用途**: 标准单设备测试
- **特点**: 使用原有的 `_create_batch_from_texts` 方法
- **适用场景**: 功能验证、调试、单设备环境

### `test_load_model_with_jax_loader_dp` (DP_SIZE>1)  
- **用途**: 数据并行测试
- **特点**: 使用新的 `_create_batch_from_texts_dp` 方法
- **适用场景**: 多设备并行、性能测试、扩展性验证

## 🔧 关键实现

### 1. 数据预处理 (`_create_batch_from_texts_dp`)

```python
# 一步到位：直接创建分片的ForwardBatch
sharded_forward_batch, device_count = _create_batch_from_texts_dp(model_config, texts, tokenizer)

@jax.pmap  # 关键：pmap装饰器
def dp_model_forward(forward_batch):
    # 直接使用已经创建好的ForwardBatch
    return model(forward_batch.input_ids, forward_batch.positions, forward_batch)

# 真正的并行调用 - 所有设备同时执行！超级简洁！
outputs = dp_model_forward(sharded_forward_batch)
```

### 2. 完整的数据流

1. **保持2D结构** → 分词但不立即扁平化
2. **JAX sharding** → 自动分片到设备  
3. **设备上扁平化** → 移除padding
4. **创建ForwardBatch** → 在每个设备上并行创建
5. **pmap并行执行** → 一行代码调用
6. **结果收集** → 自动同步

## 📊 演示场景

### 输入数据 (DP_SIZE=4)
- 4个序列: `["1+1=?", "2+2=?", "3+3=?", "4+4=?"]`
- 4个设备: 每个设备处理1个序列

### 数据分布
```
设备0: 序列0 ("1+1=?")
设备1: 序列1 ("2+2=?") 
设备2: 序列2 ("3+3=?")
设备3: 序列3 ("4+4=?")
```

### 真正并行执行的输出示例
```
🚀 启动JAX模型测试
DP_SIZE: 4

=== 🚀 Testing JAX Data Parallelism with DP_SIZE=4 ===
📊 DP Configuration:
  Input sequences: 4
  Device count: 4
  Sequences per device: 1
  Input texts: ['1+1=?', '2+2=?', '3+3=?', '4+4=?']

🔄 开始创建真正的DP批次，输入文本数量: 4
  ✅ DP批次创建完成! ForwardBatch已在所有设备上准备就绪

🚀 Starting real parallel execution...
  🔄 DP parallel iteration 1/3
    ✅ All 4 devices executed in parallel!
    📊 Output statistics:
      Output shape: (4, 1, 151936)  # [device_count, seqs_per_device, vocab_size]
      Sequences per device: 1
      Device 0: (1, 151936), max_logits=[15.2]
      Device 1: (1, 151936), max_logits=[14.8]
```

## 🔑 核心优势

1. **真正并行**: 使用JAX pmap，所有设备同时执行
2. **代码简洁**: 一步创建ForwardBatch，一行调用并行执行
3. **自动分片**: JAX自动处理数据分布，无需手动分片
4. **灵活切换**: 通过DP_SIZE参数轻松在单设备和多设备间切换
5. **内存效率**: 避免全局padding，只在设备本地padding
6. **零拷贝**: device_put直接在设备间传输数据
7. **类型安全**: 所有数据通过JAX类型系统验证

## 🔍 使用场景

| 场景 | DP_SIZE设置 | 测试方法 | 用途 |
|------|-------------|----------|------|
| 开发调试 | `DP_SIZE=1` | `test_load_model_with_jax_loader` | 快速验证功能 |
| 性能测试 | `DP_SIZE=4` | `test_load_model_with_jax_loader_dp` | 验证并行性能 |
| 扩展性测试 | `DP_SIZE=8` | `test_load_model_with_jax_loader_dp` | 大规模并行 |
| 单机多卡 | `DP_SIZE=GPU数量` | `test_load_model_with_jax_loader_dp` | 充分利用硬件 |

## 📝 注意事项

1. **DP_SIZE限制**: 确保输入序列数量能被DP_SIZE整除
2. **内存对齐**: JAX要求所有设备上的数据形状一致
3. **pmap限制**: pmap函数内不能有条件分支
4. **环境变量**: 确保正确设置 `DP_SIZE` 和 `MODEL_PATH`

## 🔮 扩展方向

1. **完整生成循环**: 在pmap内实现多步token生成
2. **梯度并行**: 扩展到训练场景的allreduce
3. **混合并行**: 结合张量并行(shard_map)和流水线并行
4. **动态形状**: 支持变长序列的真正动态batching
5. **自动调优**: 根据硬件配置自动选择最优DP_SIZE 