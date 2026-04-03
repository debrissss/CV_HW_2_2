import argparse
import torch
import random
import numpy as np
from utils.config_parser import parse_config
from utils.data_loader import get_dataloaders
from models.cnn_builder import get_model
from train import Trainer

def set_seed(seed):
    """
    设置全局随机种子以确保实验结果的可复现性。

    Args:
        seed (int): 随机种子数值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CUDA 设备的种子设置，如果是 MacOS MPS 则主要依靠 torch.manual_seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 禁用 CuDNN 的非确定性优化，牺牲少量性能以换取严谨的实验一致性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_experiment(config_path, exp_name_override=None):
    """
    执行单次完整的实验流程。

    Args:
        config_path (str): 配置文件路径。
        exp_name_override (str, optional): 覆盖默认实验名称。

    Returns:
        float: 本次实验在测试集上的准确率 (%)。
    """
    # 1. 解析 YAML 配置并与默认值合并
    config = parse_config(config_path)
    # 若未指定 override，则以文件名作为实验 ID
    exp_name = exp_name_override if exp_name_override else config_path.split('/')[-1].split('.')[0]
    
    print(f"\n--- Starting Experiment: {exp_name} ---")
    
    # 2. 加载数据，确保 45k/5k/10k 的标准划分
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="./data", 
        batch_size=config['batch_size']
    )
    
    # 3. 动态构建模型
    model = get_model(config)
    
    # 统计模型参数量，这是对比实验中“模型规模”的重要衡量指标
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params:,}")
    
    # 4. 初始化训练器并启动训练任务
    trainer = Trainer(model, train_loader, val_loader, test_loader, config, exp_name)
    trainer.run()
    
    return trainer.tracker.test_acc

def run_repeated_experiment(config_path, times=3):
    """
    重复执行多次实验，计算平均值和标准差，以评估模型的稳定性。

    Args:
        config_path (str): 配置文件路径。
        times (int): 重复次数，默认为 3 次。
    """
    accs = []
    base_name = config_path.split('/')[-1].split('.')[0]
    
    for i in range(times):
        print(f"\n=== Run {i+1}/{times} for {base_name} ===")
        # 每次运行使用不同的种子，模拟不同的权重初始化和数据打乱顺序
        set_seed(42 + i) 
        exp_name = f"{base_name}_run{i+1}"
        acc = run_experiment(config_path, exp_name_override=exp_name)
        accs.append(acc)
    
    # 计算统计指标：均值反映了性能水平，标准差反映了鲁棒性
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    
    print(f"\nResults for {base_name} after {times} runs:")
    print(f"Mean Test Accuracy: {mean_acc:.2f}%")
    print(f"Std Test Accuracy: {std_acc:.2f}%")
    
    # 将汇总后的统计结果持久化到文件，方便撰写报告
    summary_path = f"results/{base_name}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Mean: {mean_acc:.2f}\nStd: {std_acc:.2f}\nAll Runs: {accs}\n")
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    # 使用命令行参数解析，增强实验的可操作性
    parser = argparse.ArgumentParser(description="CNN Comparative Experiments on CIFAR-10")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--repeat", type=int, default=1, help="重复实验次数（用于统计均值与方差）")
    
    args = parser.parse_args()
    
    # 设置一个基准随机种子
    set_seed(42) 
    
    if args.repeat > 1:
        # 进入统计对比模式
        run_repeated_experiment(args.config, times=args.repeat)
    else:
        # 进入单次训练模式
        run_experiment(args.config)
