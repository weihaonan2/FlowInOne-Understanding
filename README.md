# FlowInOne-Understanding

FlowInOne teacher 的 perception-as-generation 理解能力微调。

## 三个数据集

| 数据集 | 任务 | 指标 | 结果 |
|--------|------|------|:---:|
| RefCOCOg | referring segmentation | cIoU | **0.475** |
| ReasonSeg | reasoning segmentation | cIoU | **0.481** |
| Cityscapes | semantic segmentation (19-class) | mIoU | **0.276** |
| Cityscapes | semantic segmentation (coarse-7) | mIoU | **0.567** |

## 核心发现

**Inverted Mask**: binary mask 存在"全白崩溃"——flow matching latent loss 对稀疏二值输出有系统性偏置。翻转 mask（白底黑目标）即可修复，RefCOCOg cIoU 从 0.085 提升到 0.475（5.6x），ReasonSeg 从 0.138 到 0.481（3.5x）。

**Aligned Prompt**: 输入图像与输出 mask 的几何对齐是 dense prediction 的前提。Cityscapes square→aligned: mIoU 0.152→0.242。

## 文件结构

```
├── code/           # 训练 + 评估代码
│   ├── scripts/    # 数据打包 & 评估脚本
│   ├── configs/    # FlowInOne 训练配置
│   └── launch/     # 启动脚本
├── data/           # WDS 数据集 (Git LFS)
├── weights/        # 最终模型权重 (Git LFS)
└── docs/           # 详细实验文档
```

## 复现

详见 `docs/交接_三个数据集的理解能力微调.md`

## 引用

```
FlowInOne: Unifying Multimodal Generation as Image-in, Image-out Flow Matching
https://arxiv.org/abs/2604.06757

Image Generators are Generalist Vision Learners (Vision Banana)
https://arxiv.org/abs/2604.20329
```
