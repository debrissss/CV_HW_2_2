# 卷积神经网络结构与正则化技巧对比实验

本项目是计算机视觉（CV）课程的作业，旨在通过 CIFAR-10 图像分类任务，对比不同 CNN 设计选择对模型性能的影响。

## 实验内容
本实验涵盖以下 7 组对比：
1. **网络深度 vs 宽度**：在参数量相近的情况下，对比深而窄与浅而宽的网络。
2. **激活函数**：对比 tanh, ReLU, GELU, Swish 对收敛速度和准确率的影响。
3. **归一化层**：对比 No Norm, BatchNorm, LayerNorm, GroupNorm。
4. **正则化方法**：对比 Dropout, Weight Decay, BatchNorm 及其组合。
5. **残差连接**：在 10 层深度下对比有无残差连接。
6. **卷积核大小与感受野**：对比 3x3 堆叠、5x5 堆叠与 7x7 单层。
7. **下采样方式**：对比 Max Pooling, Average Pooling, Strided Convolution。

## 环境要求
- MacOS (Apple Silicon M1/M2/M3/M4)
- miniforge3 (Conda)
- Python 3.10
- PyTorch (支持 MPS 加速)
- 依赖项见 `requirements.txt`

## 快速上手
### 1. 环境配置
```bash
conda create -y -n cv_hw_2_2 python=3.10
conda activate cv_hw_2_2
pip install -r requirements.txt
```

### 2. 运行单次实验
```bash
python main.py --config configs/exp1_shallow_wide.yaml
```

### 3. 运行重复实验 (获取均值与标准差)
```bash
python main.py --config configs/exp1_shallow_wide.yaml --repeat 3
```

## 项目结构
- `configs/`: 存放 7 组实验的 YAML 配置文件。
- `models/`: 模型定义，支持动态构建。
- `utils/`: 数据加载、指标记录与配置解析。
- `results/`: 实验结果输出（CSV 与 PNG 曲线图）。
- `train.py`: 核心训练引擎。
- `main.py`: 程序入口。
