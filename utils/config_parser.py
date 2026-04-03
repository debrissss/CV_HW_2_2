import yaml
import os

def parse_config(config_path):
    """
    解析 YAML 配置文件，并与默认配置合并。
    
    该函数确保即使配置文件中缺少某些字段，模型也能以预设的实验基准参数运行。

    Args:
        config_path (str): 配置文件 (.yaml) 的绝对或相对路径。
        
    Returns:
        dict: 完整的配置字典，包含实验所需的所有超参数。
    """
    with open(config_path, 'r') as f:
        # 使用 safe_load 避免执行任意代码，确保安全性
        config = yaml.safe_load(f)
        
    # 定义默认实验配置。这些值作为对照实验的基准(Baseline)。
    default_config = {
        'batch_size': 128,
        'lr': 0.001,
        'epochs': 200,
        'weight_decay': 1e-4,
        'depth': 3,
        'width': 64,
        'act_type': 'relu',
        'norm_type': 'bn',
        'use_dropout': False,
        'dropout_p': 0.5,
        'pool_type': 'max',
        'use_residual': False,
        'kernel_size': 3,
        'stack_type': 'baseline' # 对实验六中卷积核堆叠方式的标识
    }
    
    # 将用户定义的配置合并到默认配置中，用户定义的配置具有更高优先级
    final_config = {**default_config, **(config if config else {})}
    return final_config
