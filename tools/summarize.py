import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
import shutil

def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="CNN 实验结果汇总工具")
    parser.add_argument("--group", type=str, required=True, help="实验组前缀，如 exp1")
    return parser.parse_args()

def extract_test_acc(file_path):
    """
    从测试准确率文本文件中提取数值。
    
    Args:
        file_path (str): .txt 文件路径。
        
    Returns:
        float: 提取出的准确率百分比。
    """
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        content = f.read()
        # 使用正则匹配百分比数字
        match = re.search(r"(\d+\.\d+)%", content)
        if match:
            return float(match.group(1))
    return None

def summarize_group(group_prefix):
    """
    执行指定组的实验汇总。
    
    Args:
        group_prefix (str): 文件夹前缀 (如 'exp1')。
    """
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} 目录不存在。")
        return

    # 1. 寻找匹配的文件夹
    exp_folders = [f for f in os.listdir(results_dir) 
                   if os.path.isdir(os.path.join(results_dir, f)) and f.startswith(group_prefix)]
    
    if not exp_folders:
        print(f"未找到前缀为 {group_prefix} 的实验结果。")
        return

    print(f"--> 发现 {len(exp_folders)} 个符合条件的实验目录: {exp_folders}")

    summary_data = [] # 用于保存表格数据
    
    # 2. 创建专属的汇总目录
    group_summary_dir = os.path.join(results_dir, f"{group_prefix}_summarize")
    os.makedirs(group_summary_dir, exist_ok=True)

    # 创建 1x3 的子图布局
    fig = plt.figure(figsize=(18, 5))
    ax_train = fig.add_subplot(1, 3, 1)
    ax_val = fig.add_subplot(1, 3, 2)
    ax_val_acc = fig.add_subplot(1, 3, 3)

    # 创建局部放大图 (Inset axes)
    # 训练/验证 Loss 放大图稍微向下移动，避开右上角的图例
    axins_train = ax_train.inset_axes([0.55, 0.45, 0.4, 0.4])
    axins_val = ax_val.inset_axes([0.55, 0.45, 0.4, 0.4])
    # 验证准确率向上移动，避开底部边缘和可能的重叠
    axins_acc = ax_val_acc.inset_axes([0.55, 0.25, 0.4, 0.4])

    # 收集最后 50 epoch 内所有曲线的最大/小值以便确定 y 轴范围
    last50_train_loss = []
    last50_val_loss = []
    last50_val_acc = []
    max_epoch_overall = 0

    for folder in sorted(exp_folders):
        folder_path = os.path.join(results_dir, folder)
        csv_path = os.path.join(folder_path, f"{folder}.csv")
        test_acc_path = os.path.join(folder_path, f"{folder}_test_acc.txt")

        # 读取指标 CSV 并复制文件
        if os.path.exists(csv_path):
            shutil.copy(csv_path, os.path.join(group_summary_dir, f"{folder}.csv"))
            df = pd.read_csv(csv_path)
            
            ax_train.plot(df['epoch'], df['train_loss'], label=folder)
            ax_val.plot(df['epoch'], df['val_loss'], label=folder)
            ax_val_acc.plot(df['epoch'], df['val_acc'], label=folder)

            axins_train.plot(df['epoch'], df['train_loss'])
            axins_val.plot(df['epoch'], df['val_loss'])
            axins_acc.plot(df['epoch'], df['val_acc'])

            if len(df) > 0:
                max_epoch_overall = max(max_epoch_overall, df['epoch'].max())
                last_50_df = df.tail(50)
                last50_train_loss.extend(last_50_df['train_loss'].tolist())
                last50_val_loss.extend(last_50_df['val_loss'].tolist())
                last50_val_acc.extend(last_50_df['val_acc'].tolist())
        
        # 提取测试准确率
        test_acc = extract_test_acc(test_acc_path)
        summary_data.append({
            "Experiment": folder,
            "Test Accuracy (%)": test_acc if test_acc is not None else "N/A"
        })

    # 计算局部放大图的极值限制
    def set_inset_limits(axins, data, max_epoch):
        if data:
            y_min, y_max = min(data), max(data)
            y_pad = (y_max - y_min) * 0.1
            if y_pad == 0: y_pad = 0.1
            axins.set_xlim(max(0, max_epoch - 50), max_epoch)
            axins.set_ylim(y_min - y_pad, y_max + y_pad)
            axins.grid(True, linestyle='--', alpha=0.6)

    # 应用放大图坐标轴限值
    set_inset_limits(axins_train, last50_train_loss, max_epoch_overall)
    set_inset_limits(axins_val, last50_val_loss, max_epoch_overall)
    set_inset_limits(axins_acc, last50_val_acc, max_epoch_overall)

    # 2. 润色主图
    ax_train.set_title(f"Comparison of Training Loss ({group_prefix})")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Loss")
    ax_train.legend()
    ax_train.grid(True)

    ax_val.set_title(f"Comparison of Validation Loss ({group_prefix})")
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Loss")
    ax_val.legend()
    ax_val.grid(True)

    ax_val_acc.set_title(f"Comparison of Validation Accuracy ({group_prefix})")
    ax_val_acc.set_xlabel("Epoch")
    ax_val_acc.set_ylabel("Accuracy (%)")
    ax_val_acc.legend()
    ax_val_acc.grid(True)

    fig.tight_layout()
    plot_path = os.path.join(group_summary_dir, f"{group_prefix}_metrics_comparison.png")
    fig.savefig(plot_path)
    print(f"--> 已生成全局指标对比图: {plot_path}")

    # 4. 生成汇总表格
    summary_df = pd.DataFrame(summary_data)
    # 打印到终端
    print("\n" + "="*40)
    print(f" Summary of {group_prefix} Results")
    print("="*40)
    print(summary_df.to_string(index=False))
    print("="*40)
    
    # 保存为 CSV
    table_path = os.path.join(group_summary_dir, f"{group_prefix}_test_acc_summary.csv")
    summary_df.to_csv(table_path, index=False)
    print(f"--> 汇总表已保存至: {table_path}")

    plt.close('all')

if __name__ == "__main__":
    args = parse_args()
    summarize_group(args.group)
