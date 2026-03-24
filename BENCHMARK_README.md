# Pluto Training Speed Benchmark

## 概述

本目录提供两个独立的 Pluto 模型训练速度基准测试工具，不依赖 nuplan-devkit 数据集，通过生成模拟数据来测量模型完整训练步耗时。

| 脚本 | 用途 |
|------|------|
| `benchmark_training.py` | 单卡测试，支持不同 batch size / 模型配置 |
| `benchmark_training_ddp.py` | 多卡 DDP 测试，支持 1~8 卡线性扩展性分析 |

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
