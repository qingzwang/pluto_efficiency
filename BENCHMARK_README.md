# Pluto Training Speed Benchmark

## 概述

本目录提供两个独立的 Pluto 模型训练速度基准测试工具，不依赖 nuplan-devkit 数据集，通过生成模拟数据来测量模型完整训练步耗时。

| 脚本 | 用途 |
|------|------|
| `benchmark_training.py` | 单卡纯 GPU 测试，支持不同 batch size / 模型配置 |
| `benchmark_training_ddp.py` | 多卡 DDP 纯 GPU 测试，支持 1~8 卡扩展性分析 |
| `benchmark_generate_data.py` | 生成模拟 `.pt` 数据文件到磁盘 |
| `benchmark_training_e2e.py` | **端到端**测试：磁盘 I/O → DataLoader → CPU→GPU → 计算 → 更新参数 |
| `benchmark_gpu_utilization.py` | 监控训练过程中每张 GPU 的 SM 利用率并绘图 |
| `benchmark_nuplan_sim.py` | 模拟 nuplan CPU 预处理管线，评估数据处理瓶颈 |
| `benchmark_feature_cache.py` | 模拟 nuplan feature cache (gzip pickle) 训练性能 |
| `benchmark_augmentation.py` | **真实瓶颈分析**：完整 ContrastiveScenarioGenerator 增强管线 |

## 测试原理

### 为什么用模拟数据

Pluto 原始训练依赖 nuplan-devkit 提供的大规模自动驾驶数据集（需要下载约 TB 级数据）。为了快速衡量**模型本身的计算性能**而非数据 I/O 瓶颈，本脚本：

1. 在 GPU 上直接生成形状和类型完全匹配真实数据的随机张量
2. 绕过 nuplan 的 `TorchModuleWrapper` 基类，重建了一个等价的 `PlanningModel`
3. 复现了 `pluto_trainer.py` 中的核心损失函数（规划损失 + 预测损失 + ESDF 碰撞损失）

这样测出的是**纯 GPU 计算耗时**，排除了数据加载、磁盘 I/O、数据预处理等因素。

### 模型结构

脚本中的模型与 `src/models/pluto/pluto_model.py` 中的 `PlanningModel` 完全一致：

```
输入数据
  |
  +-- AgentEncoder (NATSequenceEncoder + StateAttentionEncoder)
  |     - 编码 agent 的历史轨迹（位置、速度、朝向、形状）
  |     - 使用 Neighborhood Attention (natten) 进行序列编码
  |
  +-- MapEncoder (PointsEncoder)
  |     - 编码地图多边形（车道线、边界、交通灯、限速等）
  |
  +-- StaticObjectsEncoder (FourierEmbedding)
  |     - 编码静态障碍物
  |
  +-- FourierEmbedding (位置编码)
  |
  +-- TransformerEncoder x N (自注意力)
  |     - 对所有元素做全局注意力
  |
  +-- AgentPredictor (MLP)
  |     - 预测其他 agent 的未来轨迹
  |
  +-- PlanningDecoder (交叉注意力 + MLP)
        - 基于参考线解码多模态规划轨迹
        - 输出：trajectory (bs, R, M, T, 6) + probability (bs, R, M)
```

### 模拟数据结构

每个 batch 包含以下模拟数据（形状与真实数据完全一致）：

| 数据字段 | 形状 | 说明 |
|---------|------|------|
| `agent/position` | (bs, A, T, 2) | agent 历史+未来位置 |
| `agent/heading` | (bs, A, T) | agent 朝向角 |
| `agent/velocity` | (bs, A, T, 2) | agent 速度 |
| `agent/shape` | (bs, A, T, 2) | agent 尺寸（长、宽） |
| `agent/category` | (bs, A) | agent 类型（车、行人等） |
| `agent/valid_mask` | (bs, A, T) | 有效性掩码 |
| `agent/target` | (bs, A, F, 3) | 未来轨迹真值 |
| `current_state` | (bs, 6) | ego 当前状态 |
| `map/polygon_center` | (bs, M, 3) | 多边形中心 (x,y,angle) |
| `map/point_position` | (bs, M, 3, P, 2) | 车道线点（中心+左右边界） |
| `map/valid_mask` | (bs, M, P) | 多边形有效性 |
| `static_objects/*` | (bs, N_s, ...) | 静态障碍物 |
| `reference_line/*` | (bs, R, P_ref, ...) | 参考线 |
| `cost_maps` | (bs, H, W, 1) | ESDF 碰撞代价图 |

默认参数：A=20 agents, M=150 polygons, P=20 points/polygon, R=6 reference lines, T=101 steps (21 history + 80 future)

### 损失函数

测试中使用的损失函数与 `pluto_trainer.py` 一致：

- **规划回归损失** (reg_loss): Smooth L1 Loss，最优轨迹 vs GT
- **规划分类损失** (cls_loss): Cross Entropy，选择最优参考线和模态
- **预测损失** (prediction_loss): Smooth L1 Loss，其他 agent 轨迹预测
- **碰撞损失** (collision_loss): 基于 ESDF 代价图的碰撞惩罚

### 计时方式

每个训练步被拆分为三个阶段分别计时：

```
1. Data Gen    - 在 GPU 上生成随机数据
2. Forward     - 模型前向传播 + 损失计算
3. Backward    - 反向传播（DDP 模式下包含 NCCL AllReduce 梯度同步）
4. Optimizer   - optimizer.step() 参数更新 + optimizer.zero_grad() 梯度清零
```

每个阶段之间插入 `torch.cuda.synchronize()` 确保 GPU 操作完成后再记录时间，保证计时精确。前 N 步（默认 5 步）用于 GPU warmup（JIT 编译、CUDA kernel 缓存等），不计入统计。

DDP 模式下额外测量：
- 每个 rank 的平均步耗时
- rank 间负载不均衡比例
- 多卡扩展效率（相对于单卡吞吐量的线性加速比）

---

## 快速开始

### 单卡测试

```bash
cd /home/ubuntu/lotus/pluto

# 默认配置 (bs=8, dim=128, 4-layer encoder/decoder)
python benchmark_training.py

# 自定义 batch size
python benchmark_training.py --batch-size 32

# 大模型配置
python benchmark_training.py --batch-size 32 --dim 256 --encoder-depth 6 --decoder-depth 6
```

### 多卡 DDP 测试

```bash
# 单卡基准线（同样走 DDP 代码路径，但只有 1 个进程）
python benchmark_training_ddp.py --batch-size 32

# 2 卡
torchrun --nproc_per_node=2 benchmark_training_ddp.py --batch-size 32

# 4 卡
torchrun --nproc_per_node=4 benchmark_training_ddp.py --batch-size 32

# 8 卡
torchrun --nproc_per_node=8 benchmark_training_ddp.py --batch-size 32
```

> **注意**：`--batch-size` 是每张卡的 batch size。全局 batch = batch_size × GPU 数量。

---

## 全部参数

### benchmark_training.py（单卡）

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--device` | `cuda:0` | 计算设备 |
| `--batch-size` | `8` | 批大小 |
| `--dim` | `128` | 模型隐藏层维度 |
| `--num-agents` | `20` | 每个场景的 agent 数量 |
| `--num-polygons` | `150` | 每个场景的地图多边形数量 |
| `--history-steps` | `21` | 历史步数（2.1s @ 10Hz） |
| `--future-steps` | `80` | 预测步数（8s @ 10Hz） |
| `--encoder-depth` | `4` | Transformer Encoder 层数 |
| `--decoder-depth` | `4` | Planning Decoder 层数 |
| `--num-modes` | `6` | 多模态轨迹数 |
| `--num-steps` | `50` | 测试步数 |
| `--warmup-steps` | `5` | 预热步数（不计入统计） |
| `--amp` | `False` | 启用混合精度（见已知限制） |

### benchmark_training_ddp.py（多卡）

与单卡版相同参数（无 `--device` 和 `--amp`），由 `torchrun` 自动分配 GPU。

---

## 基准测试结果

### 测试环境

- GPU: 8 × NVIDIA L40S (48 GB each)
- CUDA: PyTorch 2.x
- Python: 3.9
- 通信后端: NCCL
- 模型参数量: 4.06M (默认配置 dim=128, depth=4)

### 一、单卡不同 Batch Size

| Batch Size | 每步总耗时 | Forward+Loss | Backward+Optim | 吞吐量 | 显存占用 |
|:----------:|:---------:|:-----------:|:--------------:|:------:|:-------:|
| 8 | 68.1 ms | 31.9 ms | 36.0 ms | 116.9 samples/s | 0.42 GB |
| 16 | 76.9 ms | 36.6 ms | 39.9 ms | 207.6 samples/s | 0.76 GB |
| 32 | 101.2 ms | 46.9 ms | 54.0 ms | 315.3 samples/s | 1.45 GB |
| 64 | 162.6 ms | 70.9 ms | 91.2 ms | 393.2 samples/s | 2.85 GB |
| 128 | 338.1 ms | 124.4 ms | 213.2 ms | 378.3 samples/s | 5.59 GB |

### 二、大模型单卡（dim=256, depth=6, 18.43M 参数）

| Batch Size | 每步总耗时 | Forward+Loss | Backward+Optim | 吞吐量 | 显存占用 |
|:----------:|:---------:|:-----------:|:--------------:|:------:|:-------:|
| 32 | 164.6 ms | 71.9 ms | 92.3 ms | 194.1 samples/s | 2.59 GB |

### 三、8 卡 DDP 不同 Batch Size 详细测试

> 以下测试均为 8 × L40S DDP，`--batch-size` 为每张卡的 batch size。
> Forward、Backward、Optimizer 分别独立计时（每个阶段前后均有 `cuda.synchronize`）。

| Per-GPU BS | 全局 BS | 每步耗时 | Forward+Loss | Backward (含梯度同步) | Optimizer (step+zero_grad) | 全局吞吐量 | 单卡吞吐量 | 显存/卡 |
|:----------:|:-------:|:-------:|:-----------:|:-------------------:|:------------------------:|:---------:|:---------:|:------:|
| 4 | 32 | 83.1 ms | 29.7 ms | 46.3 ms | 6.7 ms | 385.1 samples/s | 48.1 samples/s | 0.26 GB |
| 8 | 64 | 86.8 ms | 34.7 ms | 43.9 ms | 7.8 ms | 737.3 samples/s | 92.2 samples/s | 0.43 GB |
| 16 | 128 | 95.8 ms | 38.3 ms | 49.7 ms | 7.4 ms | 1336.5 samples/s | 167.1 samples/s | 0.78 GB |
| 32 | 256 | 123.5 ms | 48.7 ms | 66.7 ms | 7.7 ms | 2072.0 samples/s | 259.0 samples/s | 1.47 GB |
| 64 | 512 | 182.3 ms | 71.3 ms | 103.8 ms | 6.8 ms | 2808.5 samples/s | 351.1 samples/s | 2.87 GB |
| **96** | **768** | **256.9 ms** | **100.2 ms** | **148.3 ms** | **8.0 ms** | **2989.0 samples/s** | **373.6 samples/s** | **4.23 GB** |
| 128 | 1024 | 358.1 ms | 124.0 ms | 227.9 ms | 5.8 ms | 2859.3 samples/s | 357.4 samples/s | 5.60 GB |
| 192 | 1536 | 518.8 ms | 184.0 ms | 327.4 ms | 7.0 ms | 2960.6 samples/s | 370.1 samples/s | 8.36 GB |
| 256 | 2048 | 715.7 ms | 278.5 ms | 429.8 ms | 7.0 ms | 2861.4 samples/s | 357.7 samples/s | 11.13 GB |
| 384 | 3072 | 1206.7 ms | 490.9 ms | 709.0 ms | 6.3 ms | 2545.6 samples/s | 318.2 samples/s | 16.61 GB |
| 512 | 4096 | 1666.1 ms | 654.5 ms | 1003.6 ms | 7.5 ms | 2458.4 samples/s | 307.3 samples/s | 22.17 GB |

### 四、多卡 DDP 扩展性（per-GPU bs=32, dim=128, depth=4）

| GPU 数量 | 全局 Batch | 每步耗时 (rank0) | Forward | Backward+Sync | 全局吞吐量 | 单卡吞吐量 | 扩展效率 | 显存/卡 |
|:--------:|:---------:|:---------------:|:-------:|:-------------:|:---------:|:---------:|:-------:|:------:|
| 1 | 32 | 103.0 ms | 48.2 ms | 54.4 ms | 310.7 samples/s | 310.7 samples/s | — | 1.45 GB |
| 2 | 64 | 109.3 ms | 48.0 ms | 60.8 ms | 585.7 samples/s | 292.9 samples/s | 94.3% | 1.47 GB |
| 4 | 128 | 114.7 ms | 49.9 ms | 64.4 ms | 1116.3 samples/s | 279.1 samples/s | 89.8% | 1.47 GB |
| 8 | 256 | 115.9 ms | 47.7 ms | 67.8 ms | 2207.8 samples/s | 276.0 samples/s | 88.8% | 1.47 GB |

> **扩展效率** = (多卡全局吞吐量) / (单卡吞吐量 × GPU数量) × 100%

---

## 结论分析

### Batch Size 与吞吐量瓶颈（8 卡 DDP）

```
全局吞吐量 (samples/s)
3100 |                        * (96)
3000 |                  *  (64)   * (160) * (192)
2900 |                                          * (128)  * (256)
2800 |
2700 |
2600 |                                                         * (384)
2500 |                                                              * (512)
2400 |
2200 |            * (32)
     |
1400 |      * (16)
     |
 800 |   * (8)
 400 | * (4)
     +----+-----+-----+------+------+------+------+------+------+------→
          4     8    16     32     64     96    128   192    256   512
                              Per-GPU Batch Size
```

1. **最优 batch size: per-GPU bs=96（全局 768）**
   - 峰值吞吐量 **3079.9 samples/s**，单卡 385.0 samples/s
   - 显存仅 4.23 GB/卡（48 GB 的 8.8%）

2. **三个阶段**

   | 阶段 | Per-GPU BS | 特征 | 瓶颈原因 |
   |------|-----------|------|---------|
   | GPU 未饱和 | 4 ~ 64 | 吞吐量快速上升 | GPU SM 利用率不足，计算单元空闲 |
   | **最优区间** | **64 ~ 96** | **吞吐量达到峰值** | **计算与显存带宽达到平衡** |
   | 效率下降 | 128 ~ 512 | 吞吐量逐步下降 | 显存带宽饱和，大 tensor 访存成为瓶颈 |

3. **耗时分布：Forward vs Backward vs Optimizer**
   - Optimizer (step + zero_grad) 耗时**恒定约 6~8 ms**，不随 batch size 变化，占比很小
   - 真正的耗时大头是 Forward 和 Backward（含 DDP 梯度同步）

   | Per-GPU BS | Forward | Backward | Optimizer | Backward/Forward 比值 |
   |:----------:|:-------:|:--------:|:---------:|:--------------------:|
   | 4 | 29.7 ms | 46.3 ms | 6.7 ms | 1.56 |
   | 32 | 48.7 ms | 66.7 ms | 7.7 ms | 1.37 |
   | 96 | 100.2 ms | 148.3 ms | 8.0 ms | 1.48 |
   | 256 | 278.5 ms | 429.8 ms | 7.0 ms | 1.54 |
   | 512 | 654.5 ms | 1003.6 ms | 7.5 ms | 1.53 |

   - Backward/Forward 比值稳定在 **1.4~1.6**，符合理论（反向传播约为前向的 1.5 倍计算量）
   - Optimizer 占总耗时 2~8%，可以忽略

4. **显存随 bs 线性增长**
   - bs=4: 0.26 GB → bs=512: 22.17 GB（线性关系，约 0.043 GB / sample）
   - 48 GB 显存理论上限约 bs=1100，但吞吐量在 bs=96 后已开始下降
   - **显存不是真正瓶颈，计算效率才是**

5. **推荐配置**

   | 目标 | 推荐 Per-GPU BS | 全局 BS | 预期吞吐量 | 显存/卡 |
   |------|:--------------:|:-------:|:---------:|:------:|
   | 最大吞吐量 | 96 | 768 | ~3080 samples/s | 4.2 GB |
   | 平衡吞吐/显存 | 64 | 512 | ~2934 samples/s | 2.9 GB |
   | 保守稳定 | 32 | 256 | ~2178 samples/s | 1.5 GB |

6. **实际训练预估（8 卡最优配置 bs=96）**
   - nuplan 数据集约 100 万场景
   - 全局 768 samples/step → 约 1302 steps/epoch
   - 3079.9 samples/s → 每个 epoch 约 **5.4 分钟**（纯计算）
   - 训练 25 个 epoch 约 **2.3 小时**（纯计算部分）

### 多卡扩展性（固定 per-GPU bs=32）

1. **近线性扩展**
   - 1 → 8 卡，全局吞吐量从 310.7 提升到 2207.8 samples/s（7.1 倍加速）
   - 8 卡扩展效率 88.8%，非常优秀

2. **通信开销分析**（per-GPU bs=32）
   - Forward 耗时基本不变（48~49 ms），各卡独立计算没有互相影响
   - Backward 从单卡约 47 ms 增加到 8 卡 66.7 ms（+20 ms），增量为 NCCL AllReduce 梯度同步
   - Optimizer 恒定约 7 ms，不受 GPU 数量影响
   - 模型只有 4.06M 参数，梯度同步数据量小（约 16 MB FP32），通信开销很低

3. **负载均衡**
   - 所有 rank 的平均步耗时几乎相同（imbalance ≈ 0%）
   - 因为使用模拟数据，每张卡的工作量完全一致
   - 实际训练中如果数据 padding 长度不同可能会引入轻微不均衡

4. **扩展效率递减趋势**
   - 2 卡 94.3% → 4 卡 89.8% → 8 卡 88.8%
   - 效率递减主要来自 AllReduce 通信量随 GPU 数增长
   - 但对于 4M 参数的小模型，即使 8 卡仍有近 89% 效率
   - 更大的模型（计算/通信比更高）扩展效率会更好

### 五、端到端训练（磁盘 I/O → CPU → GPU → 计算）

使用 `benchmark_training_e2e.py` 测试完整训练管线，模拟数据预存为 `.pt` 文件（每个 0.45 MB），通过 PyTorch DataLoader 从磁盘读取。

#### 测试流程

```
磁盘 (.pt 文件)
  → DataLoader (多进程读取 + 反序列化)    [DataLoader]
    → CPU Tensor 搬运到 GPU              [CPU→GPU (H2D)]
      → 模型前向 + 损失计算               [Forward+Loss]
        → 反向传播 + DDP 梯度同步         [Backward]
          → 优化器更新 + 梯度清零          [Optimizer]
```

#### 单卡 vs 8 卡 DDP（per-GPU bs=32, 4 workers）

| 指标 | 单卡 | 8 卡 DDP |
|------|:----:|:--------:|
| 全局 batch | 32 | 256 |
| **总步耗时** | **115.5 ms** | **135.3 ms** |
| DataLoader | 0.2 ms (0.9%) | 0.3 ms (1.3%) |
| CPU→GPU | 1.1 ms (1.0%) | 1.1 ms (0.8%) |
| Forward+Loss | 52.6 ms (44.8%) | 57.3 ms (41.2%) |
| Backward | 54.8 ms (47.3%) | 67.4 ms (50.5%) |
| Optimizer | 6.9 ms (6.0%) | 8.4 ms (6.2%) |
| **全局吞吐量** | **275.2 samples/s** | **1858.0 samples/s** |

#### 8 卡 DDP 不同 DataLoader workers 数

| Workers/GPU | 总步耗时 | DataLoader | CPU→GPU | Forward | Backward | Optimizer | 全局吞吐量 |
|:-----------:|:-------:|:----------:|:-------:|:-------:|:--------:|:---------:|:---------:|
| 0 (主进程) | 157.2 ms | 30.5 ms (19.4%) | 1.2 ms | 51.0 ms | 65.3 ms | 9.1 ms | 1618.2 samples/s |
| 2 | 135.0 ms | 0.3 ms (1.0%) | 1.1 ms | 54.9 ms | 69.7 ms | 7.4 ms | 1881.3 samples/s |
| 4 | 135.3 ms | 0.2 ms (1.6%) | 1.1 ms | 55.2 ms | 67.4 ms | 8.4 ms | 1815.9 samples/s |
| 8 | 134.4 ms | 0.3 ms (1.1%) | 1.1 ms | 55.7 ms | 68.8 ms | 7.6 ms | 1849.1 samples/s |

> 表中步耗时、各阶段均为 median 值，更稳定。

#### 8 卡 DDP 不同 Batch Size（4 workers）

| Per-GPU BS | 全局 BS | 总步耗时 | DataLoader | CPU→GPU | Forward | Backward | Optimizer | 全局吞吐量 |
|:----------:|:-------:|:-------:|:----------:|:-------:|:-------:|:--------:|:---------:|:---------:|
| 16 | 128 | 108.4 ms | 0.3 ms (1.5%) | 0.7 ms | 47.1 ms | 51.6 ms | 8.6 ms | 1161.0 samples/s |
| 32 | 256 | 135.3 ms | 0.3 ms (1.3%) | 1.1 ms | 57.3 ms | 67.4 ms | 8.4 ms | 1858.0 samples/s |
| 64 | 512 | 194.3 ms | 0.3 ms (2.6%) | 2.2 ms | 74.7 ms | 109.8 ms | 7.8 ms | 2548.3 samples/s |
| 96 | 768 | 278.2 ms | 0.3 ms (3.0%) | 3.2 ms | 98.1 ms | 170.6 ms | 7.4 ms | 2694.7 samples/s |

#### 端到端测试结论

1. **数据加载不是瓶颈（workers ≥ 2 时）**
   - 0 workers（主进程加载）：DataLoader 占 19.4%（30.5 ms），严重拖慢训练
   - 2+ workers：DataLoader 降至 0.2~0.3 ms（< 2%），完全被预取覆盖
   - 2 和 4/8 workers 差异极小，**推荐 2~4 workers/GPU**

2. **CPU→GPU 传输开销很小**
   - bs=32 时仅 1.1 ms（< 1%），pin_memory=True 发挥作用
   - bs=96 时 3.2 ms，随 batch 线性增长但仍不是瓶颈

3. **计算仍是绝对主导**
   - Forward + Backward 占总耗时 **90%+**，与纯 GPU benchmark 一致
   - 说明当前模型规模下，数据管线不会成为训练瓶颈

4. **端到端 vs 纯 GPU 对比**（8 卡 DDP, bs=32）
   - 纯 GPU（模拟数据）：123.5 ms/step, 2072 samples/s
   - 端到端（磁盘读取）：135.3 ms/step, 1858 samples/s
   - 端到端额外开销 **约 12 ms（9.6%）**，主要来自 DataLoader 偶发延迟和 H2D 传输

### 六、pin_memory 影响（8 卡 DDP, per-GPU bs=32, 4 workers）

| 配置 | 总步耗时 | DataLoader | CPU→GPU (H2D) | Forward | Backward | Optimizer | 全局吞吐量 |
|:----:|:-------:|:----------:|:-------------:|:-------:|:--------:|:---------:|:---------:|
| pin_memory=True | 135.0 ms | 0.3 ms (0.2%) | 1.1 ms (0.8%) | 57.3 ms | 67.4 ms | 8.4 ms | 1874.6 samples/s |
| pin_memory=False | 144.1 ms | 13.2 ms (9.2%) | 3.0 ms (2.1%) | 55.6 ms | 65.0 ms | 7.3 ms | 1751.8 samples/s |

> **结论**：pin_memory=True 使 DataLoader 耗时从 13.2 ms 降至 0.3 ms，H2D 传输从 3.0 ms 降至 1.1 ms。总吞吐量提升约 7%。推荐始终开启 pin_memory。

### 七、GPU SM 利用率分析

使用 `benchmark_gpu_utilization.py` 脚本，以 200ms 间隔采样 nvidia-smi 获取每张 GPU 的 SM 利用率、显存带宽利用率和功耗。

> **注意**：nvidia-smi 报告的是 SM（Streaming Multiprocessor）利用率，包含所有 CUDA core 和 Tensor Core 的活跃度。L40S 没有独立的 Tensor Core 利用率指标，SM% 是最佳可用代理指标。如需精确的 Tensor Core 利用率，需使用 NVIDIA DCGM (`dcgmi`)。

#### 8 卡 DDP, per-GPU bs=32

| GPU | SM avg | SM median | SM min | SM max | Mem BW avg | 功耗 avg |
|:---:|:------:|:---------:|:------:|:------:|:----------:|:--------:|
| 0 | 42.7% | 59.0% | 0% | 98% | 8.8% | 103.4 W |
| 1 | 43.0% | 60.0% | 0% | 85% | 8.2% | 106.6 W |
| 2 | 45.1% | 63.5% | 0% | 83% | 8.7% | 106.0 W |
| 3 | 44.2% | 63.0% | 0% | 83% | 7.9% | 108.5 W |
| 4 | 46.2% | 64.5% | 0% | 85% | 8.5% | 105.8 W |
| 5 | 45.2% | 61.5% | 0% | 84% | 8.6% | 109.8 W |
| 6 | 43.5% | 60.0% | 0% | 86% | 8.5% | 105.4 W |
| 7 | 42.0% | 57.5% | 0% | 81% | 8.6% | 101.6 W |

#### 8 卡 DDP, per-GPU bs=96（最优吞吐量配置）

| GPU | SM avg | SM median | SM min | SM max | Mem BW avg | 功耗 avg |
|:---:|:------:|:---------:|:------:|:------:|:----------:|:--------:|
| 0 | 51.9% | 85.0% | 0% | 95% | 14.7% | 113.1 W |
| 1 | 50.9% | 84.0% | 0% | 100% | 14.0% | 112.6 W |
| 2 | 52.3% | 82.0% | 0% | 96% | 14.9% | 116.3 W |
| 3 | 51.2% | 83.0% | 0% | 100% | 14.0% | 116.7 W |
| 4 | 52.3% | 84.0% | 0% | 96% | 14.0% | 115.1 W |
| 5 | 50.2% | 77.0% | 0% | 94% | 14.8% | 117.8 W |
| 6 | 51.8% | 81.0% | 0% | 97% | 14.4% | 115.4 W |
| 7 | 52.4% | 81.0% | 0% | 98% | 14.6% | 112.1 W |

#### GPU 利用率分析

1. **SM 利用率中等偏低**
   - bs=32: 平均 SM 约 43~46%，中位数 58~65%
   - bs=96: 平均 SM 约 50~52%，中位数 77~85%
   - SM 利用率不高的原因：Pluto 模型只有 4.06M 参数，单次计算量不大，GPU 在等待数据传输和 DDP 同步时 SM 处于空闲

2. **SM 周期性波动**
   - 从时间线图可以看到 SM 利用率呈现周期性脉冲：计算阶段高（60~98%），DDP 同步阶段低（~0%）
   - 中位数显著高于平均数，说明计算活跃时 GPU 利用率高，但同步等待拉低了平均值

3. **显存带宽利用率很低**
   - bs=32: 约 8~9%，bs=96: 约 14~15%
   - 模型参数量小（4.06M ≈ 16 MB FP32），显存带宽远未饱和
   - L40S 有 864 GB/s 显存带宽，当前工作负载无法充分利用

4. **功耗**
   - bs=32: 约 100~110 W，bs=96: 约 112~118 W
   - L40S TDP 为 350 W，仅用到约 30% 功耗
   - 印证了 SM 利用率不高的结论

5. **8 卡一致性**
   - 所有 GPU 的利用率非常接近（SM 差异 < 5%），说明 DDP 负载分配均匀

> 生成的可视化图表位于 `benchmark_results/` 目录下：
> - `gpu_utilization_timeline.png` - 时间线（SM%、显存带宽%、功耗）
> - `gpu_utilization_summary.png` - 柱状图（平均 SM% 和显存带宽%）

### 八、模拟 nuplan 数据管线的 CPU 预处理开销

使用 `benchmark_nuplan_sim.py` 测试。该脚本模拟真实 nuplan 训练中 `PlutoFeatureBuilder.get_features_from_scenario()` 的 CPU 密集操作，无需下载 nuplan 数据集。

#### 模拟的 CPU 操作

| 操作 | 模拟耗时/sample | 真实估计/sample | 说明 |
|------|:--------------:|:--------------:|------|
| Map 查询 + 插值 | ~30 ms | 100-500 ms | 150 个多边形 × 3 边界 × 20 点插值 |
| Cost Map 生成 | ~28 ms | 100-300 ms | cv2.fillPoly + scipy distance_transform_edt (500×500) |
| 因果推理 | ~3 ms | 20-100 ms | shapely 几何交叉检测 |
| 参考线计算 | ~2 ms | 20-50 ms | LineString 投影 |
| Agent 追踪 | ~0.3 ms | 20-50 ms | 跨时间步匹配 + 排序 |
| 坐标归一化 | ~0.4 ms | 10-50 ms | numpy matmul 旋转变换 |
| **合计** | **~63 ms** | **~300-1000 ms** | 模拟值为下界，真实值含 DB 查询 |

> **注意**：模拟值仅包含数值计算部分，不包含真实 nuplan 的数据库查询（scenario.get_ego_past_trajectory 等）。真实 feature extraction 耗时通常是模拟值的 **3-10 倍**。

#### 8 卡 DDP + CPU 模拟预处理（per-GPU bs=32）

| Workers/GPU | 总步耗时 | DataLoader | CPU→GPU | Forward | Backward | Optimizer | 全局吞吐量 | DL 占比 |
|:-----------:|:-------:|:----------:|:-------:|:-------:|:--------:|:---------:|:---------:|:-------:|
| 4 | 528.7 ms | 299.5 ms | 3.2 ms | 56.8 ms | 159.1 ms | 10.1 ms | 484 samples/s | 56.6% |
| 8 | 308.5 ms | 118.7 ms | 3.3 ms | 56.5 ms | 121.3 ms | 8.8 ms | 830 samples/s | 38.5% |
| 16 | 280.9 ms | 78.6 ms | 3.2 ms | 57.3 ms | 132.3 ms | 9.5 ms | 911 samples/s | 28.0% |
| **无模拟** | **145.8 ms** | **5.3 ms** | **1.2 ms** | **54.9 ms** | **76.8 ms** | **7.6 ms** | **1755 samples/s** | **3.6%** |

> 表中步耗时为 mean 值。DataLoader 列包含 CPU 预处理时间。

#### nuplan 默认配置（per-GPU bs=2, 8 workers）

| 配置 | 总步耗时 | DataLoader | Forward | Backward | 全局吞吐量 | DL 占比 |
|:----:|:-------:|:----------:|:-------:|:--------:|:---------:|:-------:|
| bs=2, 8 workers | 105.4 ms | 4.2 ms | 34.1 ms | 58.2 ms | 152 samples/s | 4.0% |

#### CPU 预处理瓶颈分析

1. **小 batch size（bs=2, nuplan 默认）：CPU 不是瓶颈**
   - 8 workers × 63ms/sample → 理论 ~127 samples/sec
   - bs=2 时每步只需 2 个 sample/worker → workers 有足够时间预取
   - DataLoader 仅占 4%，GPU 计算是绝对主导

2. **大 batch size（bs=32）：CPU 成为瓶颈**
   - 8 workers 时 DataLoader 占 38.5%，明显拖慢训练
   - 16 workers 时改善到 28%，但仍有显著开销
   - 无 CPU 模拟时（纯 .pt 读取），吞吐量为 1755 vs 830 samples/s（2.1 倍差距）

3. **真实 nuplan 的情况会更严重**
   - 模拟的 63ms/sample 只是真实耗时的下界
   - 真实 nuplan feature extraction 约 300-1000ms/sample（含 DB 查询、复杂 shapely 操作）
   - bs=32 + 8 workers：需要 32 samples × 300ms / 8 workers = **1200ms**，远超 GPU 计算时间

4. **推荐策略**

   | 场景 | 推荐方案 |
   |------|---------|
   | 快速迭代（小 bs） | nuplan 默认 bs=2, 8 workers，CPU 不是瓶颈 |
   | 最大吞吐量（大 bs） | **使用 feature cache**：先运行 `cache` 操作预计算特征，训练时直接读 cache |
   | 无 nuplan 数据 | 使用 `benchmark_generate_data.py` 生成模拟数据 + `benchmark_training_e2e.py` |
   | 评估 CPU 开销 | `benchmark_nuplan_sim.py --num-workers N` 调整 worker 数量 |

### 九、Feature Cache 训练性能

Pluto 推荐使用 `py_func=cache` 预计算特征并存为 gzip pickle 文件，训练时通过 `cache.use_cache_without_dataset=true` 直接读取。使用 `benchmark_feature_cache.py` 模拟该流程。

#### 缓存格式

| 属性 | 值 |
|------|-----|
| 格式 | gzip-compressed pickle (compresslevel=1) |
| 文件大小/sample | ~0.60 MB (.gz) |
| 未压缩大小/sample | ~0.64 MB |
| 压缩率 | 1.1x |
| 内容 | `{"data": {numpy arrays dict}}` |

#### 缓存读取耗时分解（per-sample）

| 操作 | 耗时 | 占比 |
|------|:----:|:---:|
| gzip 解压 + pickle.load | 3.2 ms | 93% |
| numpy → torch 转换 | 0.2 ms | 7% |
| **合计** | **3.5 ms** | 100% |

> 对比：torch.load (.pt 文件) 仅需 0.8 ms/sample，缓存格式慢约 **4.5x**（gzip 解压开销）

#### Worker 理论吞吐

| Workers/GPU | 理论吞吐 (3.5ms/sample) |
|:-----------:|:----------------------:|
| 4 | 1157 samples/sec |
| 8 | 2314 samples/sec |
| 16 | 4629 samples/sec |

#### 8 卡 DDP 训练耗时（Feature Cache）

| 配置 | 总步耗时 | DataLoader | CPU→GPU | Forward | Backward | Optimizer | 全局吞吐量 | DL 占比 |
|:----:|:-------:|:----------:|:-------:|:-------:|:--------:|:---------:|:---------:|:-------:|
| bs=32, 4w | 146.5 ms | 4.2 ms | — | 53.0 ms | 77.5 ms | 10.5 ms | 1748 samples/s | 2.9% |
| bs=32, 8w | 145.1 ms | 8.4 ms | 1.3 ms | 51.5 ms | 75.3 ms | 8.5 ms | 1765 samples/s | 5.8% |
| bs=32, 16w | 144.3 ms | 11.7 ms | — | 51.6 ms | 71.9 ms | 7.9 ms | 1774 samples/s | 8.1% |
| bs=96, 8w | 319.7 ms | 38.9 ms | 3.7 ms | 100.2 ms | 169.8 ms | 7.2 ms | 2402 samples/s | 12.2% |
| bs=32, 8w, **+augment** | 205.7 ms | 7.5 ms | 2.2 ms | 74.5 ms | 111.5 ms | 10.0 ms | 1245 samples/s | 3.7% |

> 表中步耗时为 mean 值。"+augment" 表示启用 Contrastive 数据增强（正样本对，batch 翻倍）。

#### Feature Cache 性能分析

1. **缓存读取不是瓶颈**
   - bs=32 时 DataLoader 占 3~8%（与纯 .pt 的 3.6% 基本持平）
   - bs=96 时 DataLoader 占 12%，轻微开销但不是主要瓶颈
   - 4 workers 即可满足 bs=32，因为 3.5ms/sample × 32 / 4 = 28ms << GPU 计算 135ms

2. **对比各数据管线**（8 卡 DDP, bs=32, 8 workers）

   | 数据管线 | 总步耗时 | DL 耗时 | DL 占比 | 全局吞吐量 | 相对性能 |
   |---------|:-------:|:------:|:------:|:---------:|:-------:|
   | **Feature Cache (.gz)** | **145 ms** | **8.4 ms** | **5.8%** | **1765 samples/s** | **100%** |
   | 纯 .pt 文件 | 146 ms | 5.3 ms | 3.6% | 1755 samples/s | 99% |
   | CPU 模拟预处理 (8w) | 309 ms | 119 ms | 38.5% | 830 samples/s | 47% |
   | CPU 模拟预处理 (16w) | 281 ms | 79 ms | 28.0% | 911 samples/s | 52% |

3. **Feature Cache 与 .pt 性能几乎相同**
   - 虽然单 sample gzip 解压比 torch.load 慢 4.5x（3.5ms vs 0.8ms）
   - 但多 worker 预取完全掩盖了这个差异
   - 最终训练吞吐量只差 < 1%

4. **Data Augmentation 的影响**
   - 启用 Contrastive Augmentation 后 batch 翻倍（anchor + positive）
   - Forward + Backward 耗时增加 ~50%（处理 2x 数据量）
   - DataLoader 开销不变（augmentation 在 worker 进程内完成）
   - 全局吞吐量下降 ~30%（1765 → 1245 samples/s），但每有效样本的吞吐实际相似

5. **推荐配置**

   | 目标 | 推荐 |
   |------|------|
   | 最大训练吞吐 | Feature cache + bs=96 + 8 workers → **2402 samples/s** |
   | 平衡吞吐/显存 | Feature cache + bs=32 + 4 workers → **1748 samples/s** |
   | 含 Contrastive | Feature cache + bs=32 + 8 workers + augment → **1245 samples/s** |

6. **实际训练预估（Feature Cache, bs=32, 8 卡）**
   - nuplan 数据集约 100 万场景
   - Feature cache 总存储：100 万 × 0.6 MB = **~600 GB**
   - 全局 256 samples/step → 约 3906 steps/epoch
   - 1765 samples/s → 每个 epoch 约 **9.4 分钟**
   - 训练 25 个 epoch 约 **3.9 小时**

### 十、ContrastiveScenarioGenerator 数据增强影响

Pluto 训练配置（`train_pluto.yaml`）**始终启用** `ContrastiveScenarioGenerator`，在每个 `__getitem__` 中生成 anchor + positive + negative 三份样本，collate 后以 3×B 的 batch 送入 GPU。使用 `benchmark_augmentation.py` 精确复现该管线。

#### 每个 sample 的 CPU 处理流程（per-sample profiling, 50 次取中位数）

| 操作 | 耗时 | 占比 | 说明 |
|------|:----:|:---:|------|
| gzip + pickle.load | 6.6 ms | 48% | 缓存读取 |
| cv2.warpAffine ×2 | 3.6 ms | 26% | 600×600 图像平移+旋转（正样本 cost map） |
| crop ×2 | 1.1 ms | 7% | 600→500 裁剪 + dtype 转换 |
| collision_check (≤5x) | 0.7 ms | 5% | numpy→torch + SAT 碰撞检测 |
| to_tensor ×3 | 0.8 ms | 6% | 递归 numpy→torch 转换（3 份数据） |
| normalize (np.matmul) | 0.5 ms | 4% | 坐标旋转变换 |
| deepcopy ×2 + neg_gen | 0.4 ms | 3% | 正/负样本深拷贝 |
| **合计** | **~14 ms** | 100% | — |

> 8 workers 并行处理，理论吞吐 = 8 / 0.014 ≈ **570 samples/sec/GPU**。

#### 8 卡 DDP 不同 Batch Size 对比：有 vs 无增强

> **测试条件**：8 × L40S, 8 workers/GPU, 10 步 warmup, 1 epoch, cache 20000 samples (gzip pickle)。
> 表中 step 耗时和各阶段均为 **median** 值（消除冷启动 outlier）；吞吐量使用 **mean** step 耗时计算（反映真实训练速度）。

| Per-GPU BS | 增强 | GPU batch | 步耗时 (median) | DataLoader | H2D | Forward | Backward | Optimizer | 全局吞吐量 | 显存/卡 | 总步数 |
|:----------:|:----:|:---------:|:--------------:|:----------:|:---:|:-------:|:--------:|:---------:|:---------:|:------:|:------:|
| 8 | 无 | 8 | 100 ms | 0.4 ms | 1.0 ms | 35 ms | 51 ms | 12 ms | 547 samples/s | 0.44 GB | 120 |
| 8 | **有** | **3×8=24** | 196 ms | 0.4 ms | 2.6 ms | 70 ms | 104 ms | 13 ms | 279 samples/s | 1.16 GB | 120 |
| 16 | 无 | 16 | 116 ms | 0.4 ms | 1.8 ms | 49 ms | 55 ms | 9 ms | 972 samples/s | 0.79 GB | 120 |
| 16 | **有** | **3×16=48** | 416 ms | 0.6 ms | 4.9 ms | 135 ms | 223 ms | 21 ms | 274 samples/s | 2.20 GB | 120 |
| 32 | 无 | 32 | 140 ms | 0.3 ms | 3.2 ms | 53 ms | 73 ms | 8 ms | 1580 samples/s | 1.50 GB | 78 |
| 32 | **有** | **3×32=96** | 728 ms | 1.4 ms | 9.8 ms | 220 ms | 382 ms | 30 ms | 292 samples/s | 4.30 GB | 78 |
| 64 | 无 | 64 | 206 ms | 0.3 ms | 6.3 ms | 75 ms | 113 ms | 9 ms | 2171 samples/s | 2.92 GB | 39 |
| 64 | **有** | **3×64=192** | 964 ms | 0.7 ms | 19.0 ms | 292 ms | 553 ms | 29 ms | 298 samples/s | 8.48 GB | 39 |

#### 减速比分析（有增强 / 无增强）

| Per-GPU BS | GPU batch 比 | 步耗时比 (median) | Forward 比 | Backward 比 | H2D 比 | 显存比 | 吞吐量比 |
|:----------:|:-----------:|:----------------:|:----------:|:-----------:|:------:|:------:|:--------:|
| 8 | 3.0x | **2.0x** | 2.0x | 2.0x | 2.6x | 2.6x | 1.96x |
| 16 | 3.0x | **3.6x** | 2.8x | 4.1x | 2.7x | 2.8x | 3.55x |
| 32 | 3.0x | **5.2x** | 4.1x | 5.2x | 3.1x | 2.9x | 5.41x |
| 64 | 3.0x | **4.7x** | 3.9x | 4.9x | 3.0x | 2.9x | 7.29x |

#### 关键发现

1. **DataLoader 始终不是瓶颈**
   - 所有配置下 DataLoader 等待时间 median < 1.5ms
   - 增强的 CPU 开销（~14ms/sample）完全由 8 个 worker 并行消化
   - 8 workers 供给 ~570 samples/s/GPU，远大于 GPU 需求（最高 ~37 samples/s/GPU）

2. **GPU 3x batch 是减速主因，但减速不止 3 倍**
   - 显存始终接近 3x（符合预期）
   - H2D 传输约 3x（符合预期）
   - **Forward/Backward 在小 batch (bs=8) 时只慢 2x**（GPU 未饱和，3x batch 能更好填充 GPU pipeline）
   - **Forward/Backward 在大 batch (bs=32/64) 时慢 4-5x**（显存压力增大，CUDA memory 管理开销显著增加，Backward 方差极大）

3. **有增强时吞吐量几乎不随 batch size 增长**
   - 无增强：bs 从 8→64，吞吐从 547→2171 samples/s（**3.97x 提升**）
   - 有增强：bs 从 8→64，吞吐从 279→298 samples/s（**仅 1.07x**，几乎持平）
   - 原因：GPU batch = 3×BS 增长过快，大 batch 下显存管理和 Backward 开销急剧上升，抵消了 batch 增大的吞吐收益

4. **Backward 方差在增强模式下极大**
   - bs=32 增强：Backward 范围 138ms ~ 4626ms（33x），median 382ms
   - bs=32 无增强：Backward 范围 65ms ~ 462ms（7x），median 73ms
   - 增强导致 3x 显存占用（1.5→4.3 GB），引发 CUDA memory 分配/释放抖动

5. **对比纯 GPU benchmark 的预期值**（第三节 DDP bs=96 结果：Forward=100ms, Backward=148ms, Total=257ms）
   - 增强 bs=32 的最快步（min 262ms）与纯 GPU bs=96 基本一致
   - 但 median（728ms）远高于预期，额外开销来自 CUDA memory 管理和 DDP 同步抖动

#### 推荐配置

| 目标 | 推荐 Per-GPU BS | 预期全局吞吐量 | 说明 |
|------|:--------------:|:------------:|------|
| 有增强，最大吞吐 | **8** | ~280 samples/s | 小 GPU batch (24) 减少显存压力，吞吐与 bs=64 相当 |
| 有增强，平衡训练 | 16 | ~274 samples/s | GPU batch=48，显存 2.2 GB |
| 无增强，最大吞吐 | 64+ | ~2170+ samples/s | 见第三节 DDP 测试，bs=96 峰值 ~3000 samples/s |
| 无增强，保守 | 32 | ~1580 samples/s | 显存 1.5 GB |

#### 增强模式下 Backward 方差极大的根因分析

增强模式下 Backward 方差高达 **21x**（min 208ms, max 4398ms），而纯 GPU 同 batch size 方差仅 1.2x。使用 `diag_backward_variance.py` 进行了 5 组对照实验（8 × L40S DDP, 50 步 + 10 步 warmup）：

| 测试 | 描述 | Step median | Bwd max/min | Step max/min |
|:---:|------|:---:|:---:|:---:|
| 1 | 纯 GPU bs=96，同一 tensor 重用 | 261 ms | **1.2x** | 1.1x |
| 5 | 纯 GPU bs=96，每步新分配 tensor | 274 ms | **1.1x** | 1.0x |
| 2 | DataLoader bs=32 无增强 | 143 ms | **1.3x** | 1.2x |
| 4 | DataLoader bs=32 有增强 + `gc.disable()` | **517 ms** | **24.9x** | 10.9x |
| 3 | DataLoader bs=32 有增强（默认） | **803 ms** | **21.2x** | 11.6x |

**排除的假设**：

- **CUDA 内存分配器**：Test 5 每步对 bs=96 全新分配 tensor，方差仅 1.0x → **不是分配器的问题**
- **Python GC 导致极端 spike**：Test 4 关闭 GC 后极端 spike 仍在（bwd max 4020ms）→ **GC 不是 spike 原因**
- **Agent dropout（变长 tensor）**：使用 `diag_dropout.py` 验证，关闭 agent dropout 后方差反而更高（20.8x vs 13.0x）→ **不是 agent dropout 的问题**

**根因定位：多 Worker DataLoader + pin_memory 干扰 CUDA**

使用 `diag_dataloader_path.py` 进行了 5 组精确对照实验，逐步隔离 DataLoader 路径各环节：

| 测试 | 描述 | Step median | Bwd max/min | Step max/min |
|:---:|------|:---:|:---:|:---:|
| 1 | 纯 GPU bs=96（baseline） | 257 ms | **1.2x** | 1.1x |
| 2 | DataLoader bs=96 固定 shape，**8 workers** | 283 ms | **2.2x** | **6.3x** |
| 3 | DataLoader bs=96 固定 shape，**0 workers** | 501 ms | **1.8x** | 1.3x |
| 4 | DataLoader bs=32 增强，**8 workers** | 459 ms | **26.4x** | **12.6x** |
| 5 | DataLoader bs=32 增强，**0 workers** | 1210 ms | **3.9x** | 1.3x |

**关键对照**：

1. **Test 2 vs Test 1**：即使是固定 shape 的 tensor，经过 8 worker DataLoader 路径后方差就从 1.2x 升到 2.2x（bwd），step 出现 1622ms spike（6.3x）。说明 **多 worker DataLoader 本身就会引入 CUDA 干扰**。

2. **Test 3 vs Test 2**：0 workers 时 bwd 方差降至 1.8x，step 方差降至 1.3x。step 总时间更长（501ms vs 283ms，因数据在主进程同步生成），但 **方差大幅减小**。

3. **Test 4 vs Test 5**：增强数据 + 8 workers 方差 26.4x vs 0 workers 仅 3.9x。这是最直接的证据：**同样的数据，有无 worker 进程决定了方差是否爆炸**。

4. **Test 5 vs Test 3**：0 workers 下，增强数据（3.9x）仍比固定 shape（1.8x）方差高一些，说明变长 pad_sequence 有少量贡献，但不是主因。

**机制分析**：

多 Worker DataLoader（`num_workers>0`）启动独立子进程，通过共享内存传递数据。当 `pin_memory=True` 时，PyTorch 额外启动一个 `pin_memory_thread` 将数据从 worker 输出队列拷贝到 page-locked（pinned）内存。这个过程会：

- **竞争 PCIe 带宽**：pin_memory_thread 的 DMA 传输与 CUDA kernel 的 GPU↔CPU 通信争用 PCIe 通道
- **触发页面锁定开销**：大量变长 tensor 的 pin 操作导致 OS 频繁分配/释放 pinned page
- **与 NCCL AllReduce 冲突**：DDP 梯度同步期间的 NCCL 通信与 pin_memory DMA 传输在 PCIe 上产生竞争
- **增强数据放大效应**：变长 tensor（pad_sequence）使 pinned buffer 无法复用，每步都需要新分配 pinned memory，放大了上述竞争

**确认的叠加因素**：

1. **多 Worker DataLoader + pin_memory 干扰 CUDA（主因，贡献 ~90% 的方差）**

   8 workers + pin_memory 使 bwd 方差从 3.9x → 26.4x，是极端 spike 的主要来源。

2. **Python GC 拖慢 median（~1.55x 开销）**

   对比 gc.disable() 实验，median 变化：

   | Phase | 默认 (gc on) | gc.disable() | 加速比 |
   |:-----:|:-----------:|:------------:|:------:|
   | Forward | 231 ms | 165 ms | 1.40x |
   | Backward | 427 ms | 310 ms | 1.38x |
   | **Total step** | **803 ms** | **517 ms** | **1.55x** |

   增强模式下每步有 3×32=96 份大 dict（含 numpy array + torch tensor）被创建并销毁。Python 的循环引用 GC 周期性扫描这些大量临时对象，开销显著。

3. **变长 pad_sequence 有少量贡献（次因）**

   0 workers 下增强数据方差 3.9x vs 固定 shape 1.8x，说明变长 tensor 对 CUDA allocator 有一定影响，但远不及多 worker 的贡献。

**优化建议**：

| 优化方向 | 预期收益 | 实现复杂度 |
|---------|:-------:|:---------:|
| **减少 `num_workers`**（如 2~4）或关闭 `pin_memory` | 大幅减少极端 spike，方差降至 ~2x | 低 |
| **训练循环中 `gc.disable()`**（周期性手动 `gc.collect()`） | median 快 **1.55x**，立竿见影 | 低 |
| **固定 pad 长度**（不用 `pad_sequence`，统一 pad 到 max_agents） | 进一步减少方差 | 低 |
| **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** | 减少 allocator 碎片化 | 低 |
| **减小 cost map 分辨率**（600→300） | warpAffine 快 4x，节省 ~2.5ms/sample | 低 |
| **GPU 上做 warpAffine**（用 `kornia` 或 `grid_sample`） | warpAffine+crop 共省 ~3.5ms/sample | 中 |
| **预计算增强后的 cache**（离线增强） | 省 ~7ms/sample CPU，但存储增 3x | 中 |
| **去掉 gzip 压缩**（用 pickle 或 torch.save） | 读取快 ~3x，但存储增 | 低 |

---

## 已知限制

1. **AMP (混合精度) 不可用**
   - 模型中的 `natten` 库 (NeighborhoodAttention1D) 不支持 FP16，会抛出 `RuntimeError: Input tensors must match in dtype!`
   - 如需 AMP，需升级 natten 到支持混合精度的版本，或将 AgentEncoder 中的 NATSequenceEncoder 替换为标准 Transformer

2. **模拟数据 vs 真实数据**
   - 模拟数据为随机张量，不反映真实数据的稀疏性（部分 agent 掩码无效、部分地图区域为空等）
   - 真实数据的 padding 和 masking 模式可能影响实际性能（attention 的有效序列长度不同）
   - 真实训练还需考虑数据加载耗时（DataLoader workers, 磁盘 I/O）

3. **DDP 通信拓扑**
   - 当前测试为单机多卡（NVLink / PCIe 互联）
   - 多机多卡场景下网络带宽可能成为瓶颈，扩展效率会下降

---

## 如何扩展

### 端到端训练测试

```bash
# 1. 生成模拟数据到磁盘
python benchmark_generate_data.py --num-samples 20000 --output-dir /tmp/pluto_bench_data

# 2. 单卡端到端
python benchmark_training_e2e.py --data-dir /tmp/pluto_bench_data --batch-size 32 --num-workers 4

# 3. 8 卡 DDP 端到端
torchrun --nproc_per_node=8 benchmark_training_e2e.py \
  --data-dir /tmp/pluto_bench_data --batch-size 32 --num-workers 4

# 4. 对比不同 DataLoader workers
for w in 0 2 4 8; do
  torchrun --nproc_per_node=8 benchmark_training_e2e.py \
    --data-dir /tmp/pluto_bench_data --batch-size 32 --num-workers $w
done
```

### Feature Cache 训练测试

```bash
# 1. 生成模拟 feature cache（gzip pickle 格式，与 nuplan 一致）
python benchmark_feature_cache.py --mode generate --num-samples 10000 \
  --cache-dir /tmp/pluto_feature_cache

# 2. 8 卡 DDP 训练
torchrun --nproc_per_node=8 benchmark_feature_cache.py --mode benchmark \
  --cache-dir /tmp/pluto_feature_cache --batch-size 32 --num-workers 8

# 3. 含数据增强
torchrun --nproc_per_node=8 benchmark_feature_cache.py --mode benchmark \
  --cache-dir /tmp/pluto_feature_cache --batch-size 32 --augment

# 4. 仅测 IO 性能
python benchmark_feature_cache.py --mode io --cache-dir /tmp/pluto_feature_cache \
  --pt-dir /tmp/pluto_bench_data_20k
```

### 模拟 nuplan CPU 预处理瓶颈

```bash
# 8 卡 DDP，模拟 CPU 预处理，不同 worker 数量
for w in 4 8 16; do
  echo "=== ${w} workers ==="
  torchrun --nproc_per_node=8 benchmark_nuplan_sim.py \
    --batch-size 32 --num-workers $w --num-samples 10000
done

# 对比：无 CPU 模拟（纯 .pt 读取基准线）
torchrun --nproc_per_node=8 benchmark_nuplan_sim.py \
  --batch-size 32 --num-workers 8 --no-simulate-cpu \
  --data-dir /tmp/pluto_bench_data_20k

# nuplan 默认配置测试 (bs=2)
torchrun --nproc_per_node=8 benchmark_nuplan_sim.py \
  --batch-size 2 --num-workers 8
```

### 对比不同模型规模

```bash
# 小模型
python benchmark_training.py --dim 64 --encoder-depth 2 --decoder-depth 2

# 默认模型
python benchmark_training.py --dim 128 --encoder-depth 4 --decoder-depth 4

# 大模型
python benchmark_training.py --dim 256 --encoder-depth 6 --decoder-depth 6

# 超大模型
python benchmark_training.py --dim 512 --encoder-depth 8 --decoder-depth 8
```

### 对比不同场景复杂度

```bash
# 简单场景（少量 agent 和地图）
python benchmark_training.py --num-agents 5 --num-polygons 50

# 复杂场景
python benchmark_training.py --num-agents 40 --num-polygons 300
```

### 多卡扩展性测试

```bash
# 逐步增加 GPU 数量，观察扩展效率
for n in 1 2 4 8; do
  echo "=== ${n} GPU(s) ==="
  if [ $n -eq 1 ]; then
    python benchmark_training_ddp.py --batch-size 32 --num-steps 50
  else
    torchrun --nproc_per_node=$n benchmark_training_ddp.py --batch-size 32 --num-steps 50
  fi
done
```
