# Trajectron++ Car-Road 数据集实验记录

> 实验时间: 2026-03-04 ~ 2026-03-07
> 实验环境: Python 3.9 + PyTorch + CUDA
> 数据集: car_road_data_TianJin (路侧 LiDAR 3D 检测 + 追踪)

---

## 1. 实验目标

将 Trajectron++ (Stanford ASL) 模型从 nuScenes 自动驾驶数据集迁移到自建的路侧感知数据集 (car-road)，实现交通参与者轨迹预测。

核心差异:
- **视角**: nuScenes 为车载视角 (ego-centric)，car-road 为路侧固定视角 (infrastructure)
- **坐标系**: nuScenes 位置相对自车，范围较小；car-road 为全局坐标，位置跨度数百米
- **传感器**: nuScenes 为多传感器融合；car-road 为路侧 LiDAR (原始频率 ~7-8Hz)
- **标注**: car-road 提供 `ori_labels` (原始) 和 `interpolation_labels` (插值平滑)

---

## 2. 数据处理流程

### 2.1 原始数据结构

```
/mnt/car_road_data_TianJin/
├── 001_car0325_road0327_t1/
│   └── road_labels/
│       ├── ori_labels/*.json
│       └── interpolation_labels/*.json
├── 002_car0325_road0327_t2/
│   └── ...
└── ... (共约 90 个场景)
```

每个 JSON 帧包含:
```json
{
  "timestamp": 1648174256000,
  "object": [
    {"id": 42, "label": "Car", "x": 105.3, "y": -42.1, "z": 1.2,
     "yaw": 1.57, "length": 4.5, "width": 1.8, "height": 1.5}
  ]
}
```

### 2.2 预处理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `label_type` | `interpolation` | 使用插值后的标注 |
| `resample_dt` | `0.5s` | 与 nuScenes 对齐 (原始 ~7-8Hz 降采样到 2Hz) |
| `min_track_length` | `4` | 最少 4 帧 (2 秒) |
| `val_split` | `0.15` | 15% 场景作为验证集 |
| `test_split` | `0.15` | 15% 场景作为测试集 |
| `augment` | `0°~360°, 步长 15°` | 旋转增强 (24 个角度) |

### 2.3 类别映射

| 原始标签 | Trajectron++ 类型 |
|---------|------------------|
| Car, Suv, Bus, Truck, Van, Trailer, Engineering_vehicle, Fire_engine, Vehicle_else, Huge_vehicle | `VEHICLE` |
| Pedestrian, Pedestrian_else, Person | `PEDESTRIAN` |
| Cyclist, Bicycle, Motorcycle, Tricycle, Non_motor_rider, Motor_rider, Other_rider | `PEDESTRIAN` (合并处理) |

### 2.4 场景中心化

每个场景的所有坐标减去场景均值:
```python
x_mean = data['x'].mean()
y_mean = data['y'].mean()
data['x'] = data['x'] - x_mean
data['y'] = data['y'] - y_mean
```

---

## 3. 数据验证结果 (validate_data.py)

### 3.1 第一轮验证 (Epoch 20 训练前)

| 指标 | train | val | test |
|------|-------|-----|------|
| 场景数 | 61 | 14 | 14 |
| PEDESTRIAN 数 | 1490 | 291 | 277 |
| VEHICLE 数 | 3792 | 842 | 876 |
| dt | 0.5s | 0.5s | 0.5s |
| track 长度 (中位数) | 31 | 28 | 28 |
| track 长度 (最大) | 88 | 46 | 46 |
| position x std | 82.4 | 80.8 | 84.7 |
| position y std | 40.2 | 39.7 | 39.8 |
| velocity norm (mean) | 3.81 m/s | 3.97 m/s | 4.07 m/s |
| velocity norm (p95) | 14.63 m/s | 14.80 m/s | 15.02 m/s |
| velocity norm (max) | 42.43 m/s | 30.03 m/s | 31.84 m/s |
| acceleration norm (mean) | 0.35 m/s² | 0.35 m/s² | 0.35 m/s² |
| acceleration norm (max) | 11.31 m/s² | 11.31 m/s² | 11.31 m/s² |
| \|pos_x\| > 80 占比 | 28.7% | 26.6% | 29.2% |
| \|pos_y\| > 80 占比 | 7.6% | 7.4% | 6.8% |

**关键发现**:
- 速度 max=42.43 m/s 和加速度 max=11.31 m/s² 恰好等于 `30×√2` 和 `8×√2`，说明分量级 clipping 在对角方向存在 √2 倍放大
- 位置范围很大 (x: -372~307m, y: -265~236m)，但中心化后分布合理

---

## 4. 模型配置

### 4.1 网络架构 (config/car_road.json)

| 参数 | 值 |
|------|-----|
| `prediction_horizon` | 6 步 (3 秒) |
| `maximum_history_length` | 8 步 (4 秒) |
| `minimum_history_length` | 1 步 (0.5 秒) |
| `enc_rnn_dim_history` | 32 |
| `enc_rnn_dim_future` | 32 |
| `enc_rnn_dim_edge` | 32 |
| `dec_rnn_dim` | 128 |
| `GMM_components` | 1 |
| `learning_rate` | 0.003 |
| `learning_decay_rate` | 0.9999 (exp decay) |
| `kl_weight` | 100.0 |
| `kl_min` | 0.07 |
| `grad_clip` | 1.0 |
| `dropout` (RNN) | 0.25 (keep=0.75) |
| `dropout` (MLP) | 0.1 (keep=0.9) |

### 4.2 动力学模型

| 类型 | 动力学模型 | 状态变量 | 预测输出 |
|------|-----------|---------|---------|
| VEHICLE | Unicycle | position(x,y), velocity(x,y), acceleration(x,y), heading(°,d°) | position(x,y) |
| PEDESTRIAN | SingleIntegrator | position(x,y), velocity(x,y), acceleration(x,y) | position(x,y) |

### 4.3 训练参数

| 参数 | 值 |
|------|-----|
| `batch_size` | 4096 |
| `train_epochs` | 20 (初始) → 100 (修复后) |
| `preprocess_workers` | 16 |
| `eval_every` | 10 |
| `save_every` | 10 |
| `augment` | 是 |
| `node_freq_mult_train` | 是 (高曲率轨迹过采样) |
| `dynamic_edges` | 是 |
| `edge_influence_combine_method` | attention |
| `offline_scene_graph` | 是 |
| `device` | cuda:0 |

---

## 5. 第一轮训练结果 (Epoch 20, 修复前)

### 5.1 训练 Loss

从训练日志中提取的 Epoch 20 loss:

| 节点类型 | Loss 范围 | 最终值 | 状态 |
|---------|----------|--------|------|
| **VEHICLE** | -0.35 ~ 0.55 | 0.08 | 正常收敛 |
| **PEDESTRIAN** | 9.93 ~ 10.56 | ~10.3 | **未收敛** |

### 5.2 评估指标 (VEHICLE, test set, ph=6)

在 checkpoint 20 上对 VEHICLE 的 Most Likely Z 评估结果 (5827 个预测样本):

| 指标 | mean | median | std | min | max | p25 | p75 | p95 |
|------|------|--------|-----|-----|-----|-----|-----|-----|
| **ADE** (m) | 1.084 | 0.769 | 1.331 | 0.047 | 37.119 | 0.238 | 1.419 | 3.068 |
| **FDE** (m) | 2.020 | 1.466 | 2.221 | 0.016 | 37.169 | 0.449 | 2.696 | 5.905 |
| **KDE NLL** | 1.685 | 1.973 | 1.956 | -1.745 | 20.000 | -0.129 | 2.720 | 4.235 |

### 5.3 误差分布

**ADE 阈值分析**:

| 阈值 | 超过比例 |
|------|---------|
| > 1m | 39.6% |
| > 2m | 13.0% |
| > 3m | 5.3% |
| > 5m | 1.5% |
| > 10m | 0.1% |

**FDE 阈值分析**:

| 阈值 | 超过比例 |
|------|---------|
| > 1m | 60.4% |
| > 2m | 37.0% |
| > 5m | 7.4% |
| > 10m | 1.1% |
| > 20m | 0.1% |

### 5.4 初步评估

- VEHICLE ADE 中位数 0.77m、FDE 中位数 1.47m，对于 3 秒预测来说基本合理
- 约 5% 的样本 ADE > 3m，这些可能是转弯场景或追踪 ID 切换
- 最大误差 37m 说明存在极端异常值
- **PEDESTRIAN 模型完全不可用** (loss 未收敛)

---

## 6. 问题诊断

### 6.1 问题 1: PEDESTRIAN Loss 不收敛 (loss ≈ 10)

**根因**: `process_data.py` 中 PEDESTRIAN 的 standardization 参数直接复制自 nuScenes，position std = 1。

```python
# 错误配置 (从 nuScenes 复制)
'PEDESTRIAN': {
    'position': {
        'x': {'mean': 0, 'std': 1},   # 实际数据 std ≈ 82
        'y': {'mean': 0, 'std': 1}    # 实际数据 std ≈ 40
    }
}
```

**为什么 nuScenes 上 std=1 可以工作**: nuScenes 的行人坐标是相对自车的，自车附近的行人位置范围很小 (通常 < 20m)。而 car-road 数据集使用路侧全局坐标，经场景中心化后位置 std 仍然高达 80m。

**影响**: 模型输入层接收到归一化后的位置值 ~80 (实际值/std)，远超 RNN 的有效工作范围，导致梯度异常、loss 无法下降。

### 6.2 问题 2: 速度/加速度 Clipping 有 √2 放大

**根因**: 原始实现按分量独立 clip:
```python
vx = np.clip(vx, -30, 30)
vy = np.clip(vy, -30, 30)
```
当 vx=30, vy=30 时，norm = 30√2 = 42.4 m/s，超过预期的 30 m/s 限制。

**验证**: validate_data 中 velocity max = 42.43 m/s = 30×√2，acceleration max = 11.31 m/s² = 8×√2，完美吻合。

### 6.3 问题 3: 训练 Epoch 不足

20 epoch 对于 Trajectron++ 的 CVAE 架构偏少。KL annealing (`kl_crossover=400`, `kl_decay_rate=0.99995`) 设计的是在数百个 iteration 后逐渐增大 KL 权重，20 epoch × 222 batch = 4440 iterations，KL 可能尚未完全 anneal。

---

## 7. 修复方案

### 7.1 修复 PEDESTRIAN Standardization (commit 7e88fde)

```python
# 修复后
'PEDESTRIAN': {
    'position': {
        'x': {'mean': 0, 'std': 80},   # 匹配实际数据分布
        'y': {'mean': 0, 'std': 40}    # 匹配实际数据分布
    },
    'velocity': {
        'x': {'mean': 0, 'std': 2},    # 行人速度约 1-3 m/s, 合理
        'y': {'mean': 0, 'std': 2}
    },
    'acceleration': {
        'x': {'mean': 0, 'std': 1},    # 行人加速度约 0-1 m/s², 合理
        'y': {'mean': 0, 'std': 1}
    }
}
```

### 7.2 修复 Norm-based Clipping (commit 7e88fde)

```python
# 修复后: 按 norm clip, 保持方向
v_norm = np.sqrt(vx**2 + vy**2)
v_scale = np.where(v_norm > max_vel, max_vel / np.maximum(v_norm, 1e-8), 1.0)
vx = vx * v_scale
vy = vy * v_scale
```

修复后速度 max 将严格 ≤ 30 m/s，加速度 max ≤ 8 m/s²。

### 7.3 增加训练 Epoch (commit 7e88fde)

训练 epoch 从 20 增加到 100，确保:
- KL annealing 完成 (100 × 222 = 22200 iterations)
- 学习率充分衰减 (exp decay)
- 模型有足够迭代学习复杂交互模式

---

## 8. 第二轮训练结果 (Epoch 80, 修复后)

### 8.1 修复内容回顾

本轮训练基于以下三项修复 (commit 7e88fde):
1. PEDESTRIAN standardization 参数修正 (position std: 1 → 80/40)
2. 速度/加速度 clipping 改为 norm-based (消除 √2 放大)
3. 训练 epoch 从 20 增加到 100

### 8.2 数据预处理验证

修复后重新预处理的数据统计:

| 指标 | train | val | test |
|------|-------|-----|------|
| 场景数 | 61 | 14 | 14 |
| VEHICLE 数 | 3792 | 842 | 876 |
| 曲率 > 0.1 | 50 | 13 | 15 |
| 曲率 > 0.2 | 481 | 80 | 69 |

数据规模与第一轮一致，确认预处理流程稳定。

### 8.3 评估指标 (VEHICLE, test set, ph=6, checkpoint=80)

| 模式 | ADE (m) | FDE (m) | KDE NLL |
|------|---------|---------|---------|
| **GMM Z Mode (Most Likely)** | **0.619** | **1.300** | **-1.429** |
| **Full (2000 samples)** | 0.903 | 1.829 | -1.429 |

### 8.4 与第一轮对比 (GMM Z Mode)

| 指标 | Epoch 20 (修复前) | Epoch 80 (修复后) | 改善幅度 |
|------|-------------------|-------------------|----------|
| **ADE** | 1.084 m | 0.619 m | **-42.9%** |
| **FDE** | 2.020 m | 1.300 m | **-35.6%** |
| **KDE NLL** | 1.685 | -1.429 | **-3.114** |

### 8.5 结果分析

**显著改善**:
- ADE 从 1.08m 降至 0.62m (降幅 43%)，意味着平均轨迹偏差从约 1 米缩小到 0.6 米
- FDE 从 2.02m 降至 1.30m (降幅 36%)，3 秒后的终点误差改善明显
- KDE NLL 从 1.685 降至 -1.429，表明概率密度估计质量大幅提升，模型对轨迹分布的建模更准确

**改善来源分析**:
1. **Norm-based clipping**: 消除了 √2 放大效应，极端速度/加速度值更合理，减少了异常轨迹对模型的干扰
2. **更长训练 (80 vs 20 epoch)**: KL annealing 充分完成，CVAE 后验坍缩问题缓解，模型学到更丰富的多模态分布
3. **两者协同**: 更干净的数据 + 更充分的训练 = 更好的泛化

**GMM Z Mode vs Full 对比**:
- Full 模式 (2000 samples) ADE=0.903 > ML ADE=0.619，这是正常现象
- Full 模式采样了完整后验分布 (包括低概率模态)，而 ML 只取最高概率的预测
- 两者差距不大，说明模型的概率分布集中度良好

### 8.4 PEDESTRIAN 评估结果 (test set, ph=6, checkpoint=80)

**样本数**: 5827 (包含行人、骑行者、电动车等非机动车目标)

> **注**: PEDESTRIAN 类别包含三类原始标注:
> - **纯行人**: Pedestrian, Person, Pedestrian_else
> - **骑行者/非机动车**: Cyclist, Bicycle, Motorcycle, Tricycle, Non_motor_rider, Motor_rider, Other_rider
>
> 骑行者和非机动车 (如电瓶车) 被合并到 PEDESTRIAN，因为其运动模式 (无明确航向角) 更适合 SingleIntegrator 动力学模型。这也解释了速度段中 5-15 m/s 区间有大量样本——这些主要是电瓶车和摩托车。

| 模式 | ADE (m) | FDE (m) | KDE NLL |
|------|---------|---------|---------|
| **GMM Z Mode (Most Likely)** | **0.649** | **1.237** | - |
| **Full (2000 samples)** | 0.810 | 1.541 | **1.308** |

**按速度段统计 (Most Likely Z)**:

| 速度段 | 样本数 | ADE mean | ADE median | FDE mean | FDE median |
|--------|--------|----------|------------|----------|------------|
| 静止 (<0.5 m/s) | 1588 | 0.185 | 0.029 | 0.316 | 0.041 |
| 低速 (0.5-2 m/s) | 554 | 0.571 | 0.327 | 1.155 | 0.638 |
| 中速 (2-5 m/s) | 1103 | 0.731 | 0.509 | 1.484 | 0.980 |
| 正常 (5-15 m/s) | 2582 | 0.917 | 0.658 | 1.715 | 1.191 |
| 运动 (>=1.0) 合计 | 4041 | 0.830 | 0.588 | 1.591 | 1.095 |

### 8.5 PEDESTRIAN 结果分析

**修复效果验证**:
- 第一轮训练 PEDESTRIAN loss ≈ 10，模型完全不可用
- 修复 standardization 后，PEDESTRIAN 模型成功收敛，**ML ADE = 0.649m，ML FDE = 1.237m**

**PEDESTRIAN vs VEHICLE 对比**:

| 指标 | VEHICLE | PEDESTRIAN | 说明 |
|------|---------|------------|------|
| ML ADE | 0.619 m | 0.649 m | 接近，PEDESTRIAN 略高 |
| ML FDE | 1.300 m | 1.237 m | PEDESTRIAN 反而略优 |
| KDE NLL | -1.429 | 1.308 | PEDESTRIAN 概率建模质量较低 |

- PEDESTRIAN 的 KDE NLL (1.308) 明显高于 VEHICLE (-1.429)，说明 PEDESTRIAN 的概率分布建模不如 VEHICLE 准确。这可能是因为 PEDESTRIAN 类别混合了步行者和非机动车两种截然不同的运动模式，单一的 SingleIntegrator 动力学模型难以同时精确建模两者的概率分布

### 8.6 综合评估

| 指标 | VEHICLE (Epoch 20) | VEHICLE (Epoch 80) | PEDESTRIAN (Epoch 80) |
|------|--------------------|--------------------|----------------------|
| ML ADE | 1.084 m | **0.619 m** | **0.649 m** |
| ML FDE | 2.020 m | **1.300 m** | **1.237 m** |
| KDE NLL | 1.685 | **-1.429** | 1.308 |
| 状态 | 基本可用 | 显著改善 | **从不可用到可用** |

**结论**:
1. 三项修复 (standardization + norm clipping + 训练 epoch) 效果显著
2. VEHICLE ADE 降低 43%，PEDESTRIAN 从完全不收敛到 ADE=0.649m
3. 对于路侧感知场景下 3 秒 (6 步) 轨迹预测，两类目标的 ADE 均在 0.6-0.7m 范围，属于可用水平
4. 训练尚未完成 (80/100 epoch)，最终 checkpoint 100 可能进一步改善

---

## 9. 开发时间线 (Git 提交历史)

| 时间 | Commit | 说明 |
|------|--------|------|
| 03-04 06:37 | 400ac0f | 初始适配: 添加 car-road 数据集支持 |
| 03-04 06:40 | 43c67b0 | 添加远程服务器部署脚本 |
| 03-04 06:44 | e1ee998 | 创建 trajectronpp conda 环境 |
| 03-04 07:49 | 1bc5219 | 更新数据路径到 TianJin 数据集 |
| 03-04 07:54 | 7d6cc86 | 修复 numpy 弃用警告 (np.float → np.float64) |
| 03-04 07:57 | 43402b8 | 增大 batch_size 到 4096 |
| 03-04 08:12 | 56b5777 | 尝试 AMP 混合精度训练 |
| 03-04 08:15 | 6e61543 | AMP 从 fp16 改为 bfloat16 (修 GMM NaN) |
| 03-04 08:18 | 7a75fba | 移除 AMP (OneHotCategorical 不兼容) |
| 03-04 08:38 | bf792bc | 补充更多标签类型映射 |
| 03-04 08:49 | bce3057 | 添加 validate_data.py 数据验证脚本 |
| 03-04 08:51 | 50bc559 | 修复 validate_data.py 索引方式 |
| 03-04 08:53 | de25ec0 | 第一次 clipping 修复 (分量级) + 调整 VEHICLE position std |
| 03-05 05:41 | 2d1b8c4 | 添加可视化脚本 |
| 03-05 05:44 | 5b0ba9c | 修复 pickle 兼容性 |
| 03-05 05:47 | 72217b3 | 修复 dill/pickle 兼容 |
| 03-05 06:07 | 04bfba9 | 重写可视化脚本 |
| 03-05 06:25 | 456027a | 整理结果文件到 results/ 目录 |
| **03-05 08:08** | **7e88fde** | **修复 PEDESTRIAN 不收敛: standardization + norm clipping + epoch 100** |
| **03-07** | - | **第二轮评估: VEHICLE ADE 0.619m (-43%), PEDESTRIAN ADE 0.649m (从不可用到可用)** |

---

## 10. 文件清单

| 文件 | 说明 |
|------|------|
| `process_data.py` | 数据预处理: JSON → pickle |
| `run_preprocess.sh` | 预处理启动脚本 |
| `run_train.sh` | 训练启动脚本 |
| `run_eval.sh` | 评估启动脚本 |
| `evaluate.py` | 评估逻辑: ADE/FDE/KDE 计算 |
| `validate_data.py` | 数据质量验证 |
| `visualize.py` | 轨迹可视化 |
| `deploy.sh` | 远程服务器环境部署 |
| `run_eval_pedestrian.sh` | PEDESTRIAN 评估启动脚本 |
| `diagnose_testset.py` | 测试集分析诊断 |
| `check_velocity.py` | 速度分布分析 |
| `config/car_road.json` | 模型超参数配置 |
| `results/*.csv` | 评估结果 |

---

## 附录 A: nuScenes vs car-road 参数对比

| 参数 | nuScenes | car-road (修复前) | car-road (修复后) |
|------|----------|-----------------|-----------------|
| PEDESTRIAN pos std | 1 | 1 | **80/40** |
| VEHICLE pos std | 80/80 | 120/60 | 120/60 |
| VEHICLE vel std | 15 | 15 | 15 |
| PEDESTRIAN vel std | 2 | 2 | 2 |
| vel clipping | 无 | 分量级 30 m/s | **norm级 30 m/s** |
| acc clipping | 无 | 分量级 8 m/s² | **norm级 8 m/s²** |
| dt | 0.5s | 0.5s | 0.5s |
| 视角 | 车载 ego-centric | 路侧 infrastructure | 路侧 infrastructure |

## 附录 B: 评估结果汇总

| 轮次 | 节点类型 | Checkpoint | 修复项 | ML ADE | ML FDE | KDE NLL | 状态 |
|------|---------|-----------|--------|--------|--------|---------|------|
| 第一轮 | VEHICLE | Epoch 20 | 无 | 1.084 | 2.020 | 1.685 | 基本可用 |
| 第一轮 | PEDESTRIAN | Epoch 20 | 无 | - | - | - | **不收敛 (loss≈10)** |
| 第二轮 | VEHICLE | Epoch 80 | std + clipping + epoch | **0.619** | **1.300** | **-1.429** | 显著改善 |
| 第二轮 | PEDESTRIAN | Epoch 80 | std + clipping + epoch | **0.649** | **1.237** | **1.308** | **从不可用到可用** |
