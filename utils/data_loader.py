import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

def get_dataloaders(data_dir, batch_size=128, split_ratio=(45000, 5000)):
    """
    加载 CIFAR-10 数据集并进行训练集、验证集和测试集的划分 (45k:5k:10k)。
    
    Args:
        data_dir (str): 数据集存放的根目录路径。
        batch_size (int): 每个 batch 的样本数量。
        split_ratio (tuple): 训练集与验证集的划分数量，默认为 (45000, 5000)。
        
    Returns:
        tuple: 包含 (train_loader, val_loader, test_loader) 的元组。
            - train_loader (DataLoader): 训练数据加载器，带数据增强。
            - val_loader (DataLoader): 验证数据加载器，无数据增强。
            - test_loader (DataLoader): 测试数据加载器，无数据增强。
    """
    # 训练集数据增强：RandomCrop 和 Flip 旨在提高模型的泛化能力，减少过拟合
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # 使用 CIFAR-10 官方统计的均值和标准差进行归一化，加速模型收敛
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 测试集与验证集仅进行归一化，保持评估环境的一致性
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 下载并加载完整训练集 (50,000 张图片)
    full_train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    # 下载并加载测试集 (10,000 张图片)
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # 按照指定比例划分训练集和验证集
    train_size, val_size = split_ratio
    # 固定 Generator 种子 (42) 以确保每次运行的数据划分完全一致，增强实验可复现性
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 针对验证集，我们需要移除训练时的 transform (如数据增强)，
    # 因此重新创建一个不带增强的数据集，并通过索引 (indices) 进行切片
    full_val_dataset_raw = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_test
    )
    val_dataset = Subset(full_val_dataset_raw, val_dataset.indices)

    # 配置 DataLoader。num_workers 设为 2 是为了在 MacOS M4 系统上平衡 CPU 负载与 IO 效率
    num_workers = 2 
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # 执行简易功能验证测试
    tl, vl, tsl = get_dataloaders("../data", batch_size=64)
    print(f"Train batches: {len(tl)}, Val batches: {len(vl)}, Test batches: {len(tsl)}")
