# Pluto 训练性能分析报告

> 环境：8 × NVIDIA L40S (48GB)，模型 PlanningModel 4.06M 参数，nuplan feature cache (gzip pickle)

---

## 核心结论

**Pluto 模型太小，GPU 不是瓶颈。训练速度受限于 CPU↔GPU 系统级交互开销。**

- GPU SM 利用率仅 43~52%，显存带宽利用率 8~15%，功耗仅 TDP 的 30%
- 纯 GPU 计算（无数据加载）8 卡可达 **~3000 samples/s**，但实际训练仅 **~280 samples/s**（开启增强时）

---

## 性能瓶颈定位

| 环节 | 是否瓶颈 | 说明 |
|------|:--------:|------|
| 磁盘 I/O / 缓存读取 | 否 | gzip pickle 3.5ms/sample，多 worker 预取后 <2% 开销 |
| CPU 预处理 | 否 | Feature cache 已预计算，跳过 CPU 密集操作 |
| DataLoader 数据供给 | 否 | 8 workers 供给 ~570 samples/s/GPU，GPU 仅需 ~37 |
| **ContrastiveScenarioGenerator** | **是（主因）** | 生成 3×batch（anchor+positive+negative），GPU 计算量 3 倍 |
| **pin_memory + NCCL PCIe 竞争** | **是（加剧因素）** | Backward 方差高达 33x，极端 spike 4000ms+ |
| **Python GC** | **是（次因）** | 每步创建/销毁 96 份大 dict，GC 拖慢 median 1.55x |

---

## 关键数据

### 增强模式下不同配置吞吐量（8 卡全局，samples/s）

| Per-GPU BS | 无增强 | 有增强 | 减速比 |
|:----------:|:------:|:------:|:------:|
| 8 | 547 | 279 | 2.0x |
| 16 | 972 | 274 | 3.5x |
| 32 | 1,580 | 292 | 5.4x |
| 64 | 2,171 | 298 | 7.3x |

增强模式下吞吐量锁死在 ~280 samples/s，不随 batch size 增长。

### DataLoader 配置优化（增强模式，bs=32）

| 配置 | 吞吐量 | Backward 方差 (max/min) |
|------|:------:|:----------------------:|
| 8 workers, pin_memory=True（默认） | 292 s/s | **33x** |
| 4 workers, pin_memory=True | 268 s/s | **34x** |
| 4 workers, pin_memory=False | 260 s/s | **33x** |
| **2 workers, pin_memory=False** | **289 s/s** | **7.2x** |

### DDP 多卡扩展性（无增强，bs=32）

| GPU 数量 | 全局吞吐量 | 扩展效率 |
|:--------:|:---------:|:-------:|
| 1 | 311 s/s | — |
| 2 | 586 s/s | 94.3% |
| 4 | 1,116 s/s | 89.8% |
| 8 | 2,208 s/s | 88.8% |

---

## 推荐配置

| 训练模式 | num_workers | pin_memory | 预期吞吐量 |
|---------|:-----------:|:----------:|:---------:|
| **有增强（CIL）** | 2 | False | ~289 s/s |
| **无增强** | 4~8 | True | ~1,580~2,171 s/s |

## 训练时间预估（100 万场景，25 epochs）

| 模式 | 每 epoch | 总训练时间 |
|------|:-------:|:---------:|
| 无增强 (bs=32, 8卡) | ~11 min | ~4.5 h |
| 有增强 (bs=32, 8卡) | ~57 min | ~24 h |

---

## 可行优化方向

| 优化 | 预期收益 | 复杂度 |
|------|:-------:|:-----:|
| 训练循环 `gc.disable()` | median 快 1.55x | 低 |
| `num_workers=2, pin_memory=False`（增强时） | 方差降 4~5 倍 | 低 |
| 固定 pad 长度（替代 `pad_sequence`） | 减少 pinned memory 重分配 | 低 |
| 减小 cost map 分辨率 600→300 | CPU 增强省 ~2.5ms/sample | 低 |
| 预计算增强 cache（离线） | 省 ~7ms/sample CPU，消除 3x batch | 中 |
| 换更大模型（提高 GPU 利用率） | GPU 利用率从 50%→80%+ | 高 |
