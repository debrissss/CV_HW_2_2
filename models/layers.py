import torch
import torch.nn as nn

class LayerNorm4d(nn.Module):
    """
    针对 4D 特征图 (B, C, H, W) 的 Layer Normalization 包装器。
    
    由于原始 nn.LayerNorm 通常作用于最后一个维度，本类通过轴置换实现对通道维度的归一化。
    """
    def __init__(self, num_features, eps=1e-5):
        """
        初始化 LayerNorm4d。

        Args:
            num_features (int): 特征通道数。
            eps (float): 用于数值稳定性的极小值。
        """
        super().__init__()
        self.layernorm = nn.LayerNorm(num_features, eps=eps)

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入特征图，shape: (B, C, H, W)。

        Returns:
            torch.Tensor: 归一化后的特征图，shape: (B, C, H, W)。
        """
        # 将通道维度置换到最后，以符合 nn.LayerNorm 的输入要求
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.layernorm(x)
        # 恢复原始维度顺序
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class ConvBlock(nn.Module):
    """
    一个通用的卷积块，集成了 卷积层、归一化层、激活函数 和 Dropout。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 norm_type=None, act_type='relu', use_dropout=False, dropout_p=0.5):
        """
        初始化 ConvBlock。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size (int): 卷积核大小，默认为 3。
            stride (int): 卷积步长，默认为 1。
            padding (int): 填充大小，默认为 1。
            norm_type (str, optional): 归一化类型 ('bn', 'ln', 'gn' 或 None)。
            act_type (str): 激活函数类型 ('relu', 'tanh', 'gelu', 'swish')。
            use_dropout (bool): 是否使用 Dropout。
            dropout_p (float): Dropout 概率。
        """
        super().__init__()
        
        layers = []
        # 如果使用了 Norm 层，通常不需要卷积层的 bias，因为会被 Norm 的 offset 抵消
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=(norm_type is None)))
        
        # 动态添加归一化层
        if norm_type == 'bn':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm_type == 'ln':
            layers.append(LayerNorm4d(out_channels))
        elif norm_type == 'gn':
            # 使用固定分组数 8，假设 out_channels 能被 8 整除
            layers.append(nn.GroupNorm(8, out_channels))
            
        # 动态添加激活函数
        if act_type == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif act_type == 'tanh':
            layers.append(nn.Tanh())
        elif act_type == 'gelu':
            layers.append(nn.GELU())
        elif act_type == 'swish':
            layers.append(nn.SiLU())
            
        # 动态添加 Dropout
        if use_dropout:
            layers.append(nn.Dropout2d(p=dropout_p))
            
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入特征图，shape: (B, in_channels, H, W)。

        Returns:
            torch.Tensor: 输出特征图，shape: (B, out_channels, H', W')。
        """
        return self.block(x)

class ResidualBlock(nn.Module):
    """
    标准的残差块，包含两个卷积层和一条跳跃连接。
    """
    def __init__(self, channels, norm_type='bn', act_type='relu'):
        """
        初始化 ResidualBlock。

        Args:
            channels (int): 输入与输出通道数（残差由于加法要求，通道数需保持一致）。
            norm_type (str): 归一化类型。
            act_type (str): 激活函数类型。
        """
        super().__init__()
        # 第一层卷积
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1, norm_type=norm_type, act_type=act_type)
        # 第二层卷积，最后的激活函数由于要放在残差相加之后，因此此处设为 None
        self.conv2 = ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1, norm_type=norm_type, act_type=None)
        
        # 最终相加后的激活函数
        if act_type == 'relu':
            self.final_act = nn.ReLU(inplace=True)
        elif act_type == 'tanh':
            self.final_act = nn.Tanh()
        elif act_type == 'gelu':
            self.final_act = nn.GELU()
        elif act_type == 'swish':
            self.final_act = nn.SiLU()
        else:
            self.final_act = nn.Identity()

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入特征图，shape: (B, channels, H, W)。

        Returns:
            torch.Tensor: 输出特征图，shape: (B, channels, H, W)。
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        # 残差相加：out = F(x) + x
        out += residual
        return self.final_act(out)

class DownsampleBlock(nn.Module):
    """
    用于缩小特征图尺寸（下采样）的模块。
    """
    def __init__(self, pool_type='max', in_channels=None, out_channels=None):
        """
        初始化 DownsampleBlock。

        Args:
            pool_type (str): 下采样方式 ('max', 'avg', 'stride_conv')。
            in_channels (int, optional): 输入通道数（主要供步长卷积使用）。
            out_channels (int, optional): 输出通道数（主要供步长卷积使用）。
        """
        super().__init__()
        if pool_type == 'max':
            # 保留局部最大响应，常用于提取显著纹理
            self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool_type == 'avg':
            # 保留局部平滑特征，保留更多背景信息
            self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pool_type == 'stride_conv':
            # 使用可学习参数的卷积进行下采样，相比池化更具灵活性
            self.down = nn.Conv2d(in_channels, out_channels if out_channels else in_channels, 
                                  kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入特征图，shape: (B, C, H, W)。

        Returns:
            torch.Tensor: 下采样后的特征图，shape: (B, C, H/2, W/2)。
        """
        return self.down(x)
