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
            self.features = nn.Sequential(*layers)
        else:
            # 引入现代 CV 的 Stage-Based (分阶段) 堆叠机制
            curr_width = width
            # CIFAR-10 标准分配策略：[n, n, n]，其中 depth = 1(stem) + 3 * n * 2 + 1(fc)
            n = max(1, (depth - 2) // 6) 
            
            # 1. 组装入口特征提取层 (Stem)
            self.stem = ConvBlock(in_channels, curr_width, kernel_size=3, padding=1, 
                                  norm_type=norm_type, act_type=act_type, 
                                  use_dropout=use_dropout, dropout_p=dropout_p)
            
            # 2. 分阶段构建主体网络，遵循“尺寸减半、通道加倍”准则
            # Stage 1: stride=1，维持原特征图尺寸
            stage1 = self._make_stage(curr_width, curr_width, n, stride=1, 
                                      use_residual=use_residual, pool_type=pool_type, 
                                      norm_type=norm_type, act_type=act_type, 
                                      kernel_size=kernel_size, use_dropout=use_dropout, dropout_p=dropout_p)
            
            # Stage 2: stride=2，下采样并增加通道
            stage2 = self._make_stage(curr_width, curr_width * 2, n, stride=2, 
                                      use_residual=use_residual, pool_type=pool_type, 
                                      norm_type=norm_type, act_type=act_type, 
                                      kernel_size=kernel_size, use_dropout=use_dropout, dropout_p=dropout_p)
            
            # Stage 3: stride=2，进一步获取高级抽象特征
            stage3 = self._make_stage(curr_width * 2, curr_width * 4, n, stride=2, 
                                      use_residual=use_residual, pool_type=pool_type, 
                                      norm_type=norm_type, act_type=act_type, 
                                      kernel_size=kernel_size, use_dropout=use_dropout, dropout_p=dropout_p)
            
            curr_width = curr_width * 4
            self.features = nn.Sequential(self.stem, stage1, stage2, stage3)
        
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

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, use_residual, pool_type, norm_type, act_type, kernel_size, use_dropout, dropout_p):
        """
        阶段构建辅助方法。为 Baseline（无残差）和 ResNet（有残差）执行统一的维度、步长分配。
        """
        layers = []
        if use_residual:
            # 对于残差网络，Stage的第一个Block兼具过渡功能，通过传入 stride 实现空间、通道对齐
            layers.append(ResidualBlock(in_channels, out_channels, stride=stride, norm_type=norm_type, act_type=act_type, kernel_size=kernel_size))
            # 剩余的Block都是等维度的特征提取
            for _ in range(1, num_blocks):
                layers.append(ResidualBlock(out_channels, out_channels, stride=1, norm_type=norm_type, act_type=act_type, kernel_size=kernel_size))
        else:
            # 无残差的普通网络：为了与 ResNet 对齐参数量与预算层数，每个“Block”由两层 Conv 构成
            if stride == 2:
                # 遇到阶段间换乘，首先进行通道与空间维度的下探
                if pool_type == 'stride_conv':
                    # 步长卷积自带一层卷积预算并处理了换通道
                    layers.append(DownsampleBlock(pool_type=pool_type, in_channels=in_channels, out_channels=out_channels, norm_type=norm_type, act_type=act_type, kernel_size=kernel_size))
                    # 补充一层维持通道的 Conv 以补齐一个 Block 内 2 层的预算
                    layers.append(ConvBlock(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, norm_type=norm_type, act_type=act_type, use_dropout=use_dropout, dropout_p=dropout_p))
                else:
                    # 池化操作不消耗卷积预算，但完成了空间降采样 (Kernel Size 传入通常被安全地视为兼容参数不做改动，或由积木内部判断跳过)
                    layers.append(DownsampleBlock(pool_type=pool_type, norm_type=norm_type, act_type=act_type, kernel_size=kernel_size))
                    # 连接首层卷积，完成通道数推展 (in -> out)
                    layers.append(ConvBlock(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, norm_type=norm_type, act_type=act_type, use_dropout=use_dropout, dropout_p=dropout_p))
                    # 补充第二层卷积 (out -> out) 补齐预算
                    layers.append(ConvBlock(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, norm_type=norm_type, act_type=act_type, use_dropout=use_dropout, dropout_p=dropout_p))
            else:
                # 不具有步长下采样的情况 (例如 Stage 1)，直联 2个普通卷积
                layers.append(ConvBlock(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, norm_type=norm_type, act_type=act_type, use_dropout=use_dropout, dropout_p=dropout_p))
                layers.append(ConvBlock(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, norm_type=norm_type, act_type=act_type, use_dropout=use_dropout, dropout_p=dropout_p))

            # 剩余的特征提取（每个 Block 也是 2层 Conv），维持 out_channels
            for _ in range(1, num_blocks):
                layers.append(ConvBlock(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, norm_type=norm_type, act_type=act_type, use_dropout=use_dropout, dropout_p=dropout_p))
                layers.append(ConvBlock(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, norm_type=norm_type, act_type=act_type, use_dropout=use_dropout, dropout_p=dropout_p))

        return nn.Sequential(*layers)

def get_model(config):
    """
    模型工厂函数：根据配置实例化 ConfigurableCNN。

    Args:
        config (dict): 配置字典。

    Returns:
        ConfigurableCNN: 实例化的 PyTorch 模型。
    """
    return ConfigurableCNN(config)
