import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re

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

    # 设置绘图风格
    plt.style.use('seaborn-v0_8')
    fig_train, ax_train = plt.subplots(figsize=(10, 6))
    fig_val, ax_val = plt.subplots(figsize=(10, 6))

    for folder in sorted(exp_folders):
        folder_path = os.path.join(results_dir, folder)
        csv_path = os.path.join(folder_path, f"{folder}.csv")
        test_acc_path = os.path.join(folder_path, f"{folder}_test_acc.txt")

        # 读取指标 CSV
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # 绘制训练 Loss
            ax_train.plot(df['epoch'], df['train_loss'], label=folder)
            # 绘制验证 Loss
            ax_val.plot(df['epoch'], df['val_loss'], label=folder)
        
        # 提取测试准确率
        test_acc = extract_test_acc(test_acc_path)
        summary_data.append({
            "Experiment": folder,
            "Test Accuracy (%)": test_acc if test_acc is not None else "N/A"
        })

    # 2. 润色并保存训练 Loss 对比图
    ax_train.set_title(f"Comparison of Training Loss ({group_prefix})")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Loss")
    ax_train.legend()
    train_plot_path = os.path.join(group_summary_dir, f"{group_prefix}_train_loss_comparison.png")
    fig_train.savefig(train_plot_path)
    print(f"--> 已生成训练 Loss 对比图: {train_plot_path}")

    # 3. 润色并保存验证 Loss 对比图
    ax_val.set_title(f"Comparison of Validation Loss ({group_prefix})")
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Loss")
    ax_val.legend()
    val_plot_path = os.path.join(group_summary_dir, f"{group_prefix}_val_loss_comparison.png")
    fig_val.savefig(val_plot_path)
    print(f"--> 已生成验证 Loss 对比图: {val_plot_path}")

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
