# MIPCandy vs nnUNet - Pipeline 对比分析

本文档总结了 nnUNet 使用但 MIPCandy 当前未实现的关键技巧。

## 关键差异概览

| 技巧 | nnUNet | MIPCandy | 影响 |
|------|--------|----------|------|
| 深度监督 | ✅ | ❌ | 高 |
| Poly LR | ✅ | ❌ (线性) | 中 |
| SGD+Nesterov | ✅ | ❌ (AdamW) | 中 |
| 丰富数据增强 | ✅ 10+ | ❌ 依赖MONAI | 高 |
| 数据指纹分析 | ✅ | ❌ | 中 |
| 自动实验规划 | ✅ | ❌ | 中 |
| TTA镜像 | ✅ | ❌ | 高 |
| 模型集成 | ✅ | ❌ | 高 |
| 混合精度 | ✅ | ❌ | 中 |
| 梯度裁剪 | ✅ | ❌ | 中 |
| 前景过采样 | ✅ | ❌ | 高 |
| torch.compile | ✅ | ❌ | 低 |
| 分布式训练 | ✅ | ❌ | 低 |
| 级联模型 | ✅ | ❌ | 中 |

---

## 1. 深度监督 (Deep Supervision)

### nnUNet 实现
```python
# nnUNet: training/loss/deep_supervision.py
class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        # 使用指数衰减权重: 1, 0.5, 0.25, ...
        self.weight_factors = tuple(weight_factors)
        self.loss = loss

    def forward(self, *args):
        return sum([weights[i] * self.loss(*inputs)
                    for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])
```

**权重计算**: `weights = [1 / (2 ** i) for i in range(num_scales)]`

### MIPCandy 当前状态
❌ 未实现，只在最终输出上计算损失

### 建议实现
在 `mipcandy/common/optim/loss.py` 中添加 `DeepSupervisionWrapper`

---

## 2. 学习率调度器

### nnUNet: Poly LR
```python
# nnUNet: training/lr_scheduler/polylr.py
new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** 0.9
```

### MIPCandy: 线性衰减
```python
# mipcandy/common/optim/lr_scheduler.py
r = self._k * step + self._b  # 线性衰减
```

### 建议实现
添加 `PolyLRScheduler` 到 `mipcandy/common/optim/lr_scheduler.py`

---

## 3. 优化器配置

### nnUNet
```python
optimizer = torch.optim.SGD(
    params, lr=1e-2, weight_decay=3e-5,
    momentum=0.99, nesterov=True
)
```

### MIPCandy
```python
return optim.AdamW(params)  # 默认参数
```

### 说明
nnUNet 论文表明 SGD + Nesterov + 高动量 在医学图像分割上表现更好

---

## 4. 数据增强管线

### nnUNet 完整增强链
位于 `nnUNetTrainer.get_training_transforms()`:

1. **空间变换** (`SpatialTransform`) - p=0.2
   - 旋转: ±30° (3D) / ±180° (2D)
   - 缩放: 0.7-1.4

2. **高斯噪声** (`GaussianNoiseTransform`) - p=0.1
   - variance: 0-0.1

3. **高斯模糊** (`GaussianBlurTransform`) - p=0.2
   - sigma: 0.5-1.0

4. **亮度变换** (`MultiplicativeBrightnessTransform`) - p=0.15
   - multiplier: 0.75-1.25

5. **对比度变换** (`ContrastTransform`) - p=0.15
   - range: 0.75-1.25

6. **模拟低分辨率** (`SimulateLowResolutionTransform`) - p=0.25
   - scale: 0.5-1.0

7. **Gamma变换** (`GammaTransform`)
   - 反向 p=0.1, 正向 p=0.3
   - gamma: 0.7-1.5

8. **镜像翻转** (`MirrorTransform`)
   - 所有允许的轴

### MIPCandy 当前状态
主要依赖 MONAI 库，自身只实现了 `JointTransform` 包装器

---

## 5. 测试时增强 (TTA)

### nnUNet 实现
```python
# predict_from_raw_data.py
def _internal_maybe_mirror_and_predict(self, x):
    prediction = self.network(x)
    if mirror_axes:
        axes_combinations = [
            c for i in range(len(mirror_axes))
            for c in itertools.combinations(mirror_axes, i + 1)
        ]
        for axes in axes_combinations:
            prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
        prediction /= (len(axes_combinations) + 1)
    return prediction
```

### MIPCandy 当前状态
❌ 推理时没有镜像增强

### 建议实现
在 `mipcandy/inference.py` 的 `Predictor` 类中添加 TTA 支持

---

## 6. 模型集成

### nnUNet 实现
```python
# 加载多个fold的权重
for params in self.list_of_parameters:
    self.network.load_state_dict(params)
    prediction += self.predict_sliding_window_return_logits(data)
prediction /= len(self.list_of_parameters)
```

### MIPCandy 当前状态
❌ 单模型推理

---

## 7. 混合精度训练

### nnUNet 实现
```python
self.grad_scaler = GradScaler("cuda")

with autocast(self.device.type, enabled=True):
    output = self.network(data)
    l = self.loss(output, target)

self.grad_scaler.scale(l).backward()
self.grad_scaler.unscale_(self.optimizer)
torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
self.grad_scaler.step(self.optimizer)
self.grad_scaler.update()
```

### MIPCandy 当前状态
❌ 全精度训练

---

## 8. 前景过采样

### nnUNet 实现
```python
self.oversample_foreground_percent = 0.33
# 33% 的样本强制包含前景像素
```

### MIPCandy 当前状态
❌ 随机均匀采样

---

## 9. 数据集指纹提取

### nnUNet 自动分析内容
- spacing 分布统计
- crop 后的 shape 分布
- 前景强度统计 (mean, median, std, min, max, percentiles)
- 各向异性检测

### MIPCandy 当前状态
❌ 需要手动分析和配置

---

## 10. 自动实验规划

### nnUNet 自动确定
- 目标 spacing (基于中位数)
- patch size (基于 GPU 内存估算)
- batch size (基于 VRAM 和 patch size)
- 网络深度和特征数
- 是否需要 3d_lowres 配置

### MIPCandy 当前状态
❌ 所有参数需手动配置

---

## 优先实现建议

### 高优先级 (预期提升 2-5 Dice 点)
1. **深度监督** - 多尺度损失监督
2. **TTA 镜像** - 推理时8倍增强 (3D)
3. **前景过采样** - 处理类别不平衡
4. **丰富数据增强** - 特别是 Gamma 和 SimulateLowRes

### 中优先级 (预期提升 1-2 Dice 点)
5. **Poly LR 调度器**
6. **混合精度训练**
7. **梯度裁剪**
8. **模型集成**

### 低优先级 (工程优化)
9. torch.compile 支持
10. 分布式训练
11. 自动实验规划

---

## 参考文献

- [nnU-Net Paper](https://www.nature.com/articles/s41592-020-01008-z)
- [nnUNet GitHub](https://github.com/MIC-DKFZ/nnUNet)
