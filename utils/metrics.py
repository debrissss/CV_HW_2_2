import os
import pandas as pd
import matplotlib.pyplot as plt

class ExperimentTracker:
    """
    用于跟踪、持久化和可视化实验过程中的性能指标（Loss 和 Accuracy）。
    
    Attributes:
        exp_name (str): 实验名称，用于命名的文件和文件夹。
        save_dir (str): 结果保存的根目录，默认为 "results"。
        history (dict): 存储训练过程中每轮数据的字典。
            结构: {
                'epoch': List[int],
                'train_loss': List[float],
                'val_loss': List[float],
                'val_acc': List[float]
            }
        test_acc (float): 最终测试集的准确率。
    """
    def __init__(self, exp_name, save_dir="results"):
        """
        初始化指标跟踪器。

        Args:
            exp_name (str): 实验的唯一标识符。
            save_dir (str): 存储结果的目录路径。
        """
        self.exp_name = exp_name
        # 为每个实验创建独立的子文件夹，避免结果混淆
        self.save_dir = os.path.join(save_dir, exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        self.test_acc = 0.0

    def update(self, epoch, train_loss, val_loss, val_acc):
        """
        更新每轮产生的实验数据。

        Args:
            epoch (int): 当前轮次编号。
            train_loss (float): 当前轮次的训练损失。
            val_loss (float): 当前轮次的验证损失。
            val_acc (float): 当前轮次的验证准确率 (%)。
        """
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)

    def resume(self):
        """
        尝试从现有的 CSV 文件中恢复实验历史记录。
        
        Returns:
            int: 恢复的起始轮次 (Epoch)。若无历史记录，返回 0。
        """
        csv_path = os.path.join(self.save_dir, f"{self.exp_name}.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                self.history['epoch'] = df['epoch'].tolist()
                self.history['train_loss'] = df['train_loss'].tolist()
                self.history['val_loss'] = df['val_loss'].tolist()
                self.history['val_acc'] = df['val_acc'].tolist()
                print(f"--> Resumed experiment history from {csv_path} (Last Epoch: {self.history['epoch'][-1]})")
                return self.history['epoch'][-1]
            except Exception as e:
                print(f"--> Failed to resume history: {e}")
        return 0

    def save_results(self):
        """
        将历史指标持久化为 CSV 文件，并生成可视化曲线图 (PNG)。
        """
        # 1. 保存为 CSV，方便后续进行精确的数值分析或对比
        df = pd.DataFrame(self.history)
        csv_path = os.path.join(self.save_dir, f"{self.exp_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        # 2. 绘制 PNG 曲线。左图展示 Loss 变化以观察收敛情况；
        # 右图展示验证集准确率，用于评估过拟合程度。
        plt.figure(figsize=(12, 5))
        
        # Loss 曲线绘制
        plt.subplot(1, 2, 1)
        plt.plot(self.history['epoch'], self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['epoch'], self.history['val_loss'], label='Val Loss')
        plt.title(f'Loss - {self.exp_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy 曲线绘制
        plt.subplot(1, 2, 2)
        plt.plot(self.history['epoch'], self.history['val_acc'], label='Val Acc')
        plt.title(f'Accuracy - {self.exp_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        png_path = os.path.join(self.save_dir, f"{self.exp_name}.png")
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()
        print(f"Curves saved to {png_path}")

    def log_model_architecture(self, model):
        """
        打印并保存模型的层次结构。

        Args:
            model (nn.Module): 待记录的 PyTorch 模型。
        """
        model_str = str(model)
        # 1. 打印到控制台，方便实验开始时即时查看
        print("\n" + "="*50)
        print(f" Model Architecture: {self.exp_name}")
        print("="*50)
        print(model_str)
        print("="*50 + "\n")
        
        # 2. 保存到文本文件，方便后期作为实验报告的附录
        arch_path = os.path.join(self.save_dir, "model_arch.txt")
        with open(arch_path, "w") as f:
            f.write(f"Experiment: {self.exp_name}\n")
            f.write("="*30 + "\n")
            f.write(model_str)
            f.write("\n" + "="*30)
        print(f"Model architecture saved to {arch_path}")

    def log_test_acc(self, test_acc):
        """
        记录最终的测试集准确率到文本文件中。

        Args:
            test_acc (float): 最终在测试集上的准确率 (%)。
        """
        self.test_acc = test_acc
        with open(os.path.join(self.save_dir, f"{self.exp_name}_test_acc.txt"), "w") as f:
            f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        print(f"Final Test Accuracy: {test_acc:.2f}%")
