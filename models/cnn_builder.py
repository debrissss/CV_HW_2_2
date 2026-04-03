import torch
import torch.nn as nn
from .layers import ConvBlock, ResidualBlock, DownsampleBlock

class ConfigurableCNN(nn.Module):
    """
    根据配置字典动态构建的卷积神经网络类。
    
    该类支持多维度的对照实验，包括深度、宽度、激活函数、归一化、下采样方式、残差连接及卷积核大小。
    """
    def __init__(self, config):
        """
        初始化 ConfigurableCNN。

        Args:
            config (dict): 包含模型超参数的配置字典。
                - depth (int): 卷积层的总数。
                - width (int): 基础特征通道数。
                - use_residual (bool): 是否在层间应用残差连接。
                - act_type (str): 激活函数类型。
                - norm_type (str): 归一化层类型。
                - pool_type (str): 下采样池化类型。
                - use_dropout (bool): 是否启用 Dropout。
                - kernel_size (int): 卷积核尺寸。
                - stack_type (str): 实验六的堆叠策略标识。
        """
        super().__init__()
        
        # 从配置中提取参数，若缺失则使用默认值
        depth = config.get('depth', 3)
        width = config.get('width', 64)
        use_residual = config.get('use_residual', False)
        act_type = config.get('act_type', 'relu')
        norm_type = config.get('norm_type', 'bn')
        pool_type = config.get('pool_type', 'max')
        use_dropout = config.get('use_dropout', False)
        dropout_p = config.get('dropout_p', 0.5)
        kernel_size = config.get('kernel_size', 3) 
        stack_type = config.get('stack_type', 'baseline') 

        layers = []
        in_channels = 3 # CIFAR-10 输入为 RGB 3 通道
        curr_width = width
        
        # 为了保证不同深度下特征图尺寸变化的一致性，我们将下采样点固定在总深度的 1/3 和 2/3 处。
        # 这样可以确保输出到分类器前的特征图尺寸（对于 32x32 输入，经过两次 1/2 下采样变为 8x8）。
        downsample_indices = [depth // 3, (2 * depth) // 3]

        for i in range(depth):
            # 根据 stack_type（实验六）动态调整卷积块配置
            # baseline 对应 3x3 堆叠，medium/large 对应 5x5/7x7 对齐感受野
            if stack_type == 'baseline':
                layers.append(ConvBlock(in_channels, curr_width, kernel_size=3, padding=1, 
                                        norm_type=norm_type, act_type=act_type, 
                                        use_dropout=use_dropout, dropout_p=dropout_p))
            else:
                # 使用配置中指定的 kernel_size，并自动计算 padding 以保持特征图尺寸（除非下采样）
                layers.append(ConvBlock(in_channels, curr_width, kernel_size=kernel_size, padding=kernel_size//2,
                                        norm_type=norm_type, act_type=act_type))

            # 残差连接逻辑：仅在启用 residual 且通道数对齐（in == out）且不是第一层时应用。
            # 这是因为 skip connection 要求输入输出维度完全一致。
            if use_residual and in_channels == curr_width and i > 0:
                # 替换当前最后一个加入的普通卷积块为残差块
                layers[-1] = ResidualBlock(curr_width, norm_type=norm_type, act_type=act_type)

            in_channels = curr_width
            
            # 在预设的索引处插入下采样层
            if i in downsample_indices and i < depth - 1:
                layers.append(DownsampleBlock(pool_type=pool_type, in_channels=in_channels, out_channels=in_channels))

        self.features = nn.Sequential(*layers)
        
        # 全局平均池化：将 (B, C, H, W) 转换为 (B, C, 1, 1)，增强模型对空间位置的鲁棒性
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类逻辑：映射到 CIFAR-10 的 10 个类别
        self.classifier = nn.Linear(curr_width, 10)

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入图像张量，shape: (B, 3, 32, 32)。

        Returns:
            torch.Tensor: 各类别的未归一化得分（Logits），shape: (B, 10)。
        """
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1) # 展平为 (B, C)
        x = self.classifier(x)
        return x

def get_model(config):
    """
    模型工厂函数：根据配置实例化 ConfigurableCNN。

    Args:
        config (dict): 配置字典。

    Returns:
        ConfigurableCNN: 实例化的 PyTorch 模型。
    """
    return ConfigurableCNN(config)
