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
                - arch_type (str): 架构类型 ('baseline' 或 'vgg')。
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
        arch_type = config.get('arch_type', 'baseline')

        layers = []
        in_channels = 3 # CIFAR-10 输入为 RGB 3 通道

        if arch_type == 'vgg':
            # 实验二要求的 VGG-style 结构：3 个 Block，每个 Block 由 2 个卷积层组成，通道数翻倍
            block_channels = [64, 128, 256]
            curr_width = block_channels[-1] # 最终输出通道数
            
            for out_channels in block_channels:
                # 每个 Block 包含 2 个 Conv 组合
                for _ in range(2):
                    layers.append(ConvBlock(in_channels, out_channels, kernel_size=kernel_size, 
                                            padding=kernel_size//2, norm_type=norm_type, 
                                            act_type=act_type, use_dropout=use_dropout, 
                                            dropout_p=dropout_p))
                    in_channels = out_channels
                
                # 每个 Block 结尾由 MaxPool2d (stride=2) 进行下采样
                layers.append(DownsampleBlock(pool_type=pool_type, in_channels=in_channels, out_channels=in_channels))
        else:
            # 默认的 baseline 结构逻辑
            curr_width = width
            downsample_indices = [depth // 3, (2 * depth) // 3]

            for i in range(depth):
                if stack_type == 'baseline':
                    layers.append(ConvBlock(in_channels, curr_width, kernel_size=3, padding=1, 
                                            norm_type=norm_type, act_type=act_type, 
                                            use_dropout=use_dropout, dropout_p=dropout_p))
                else:
                    layers.append(ConvBlock(in_channels, curr_width, kernel_size=kernel_size, padding=kernel_size//2,
                                            norm_type=norm_type, act_type=act_type))

                if use_residual and in_channels == curr_width and i > 0:
                    layers[-1] = ResidualBlock(curr_width, norm_type=norm_type, act_type=act_type)

                in_channels = curr_width
                
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
