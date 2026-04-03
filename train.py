import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.metrics import ExperimentTracker

class Trainer:
    """
    负责模型训练、验证和测试的核心类。
    
    该类集成了损失函数定义、优化器配置、设备(CPU/MPS/CUDA)管理以及基于进度条的训练循环。
    """
    def __init__(self, model, train_loader, val_loader, test_loader, config, exp_name):
        """
        初始化训练器。

        Args:
            model (nn.Module): 待训练的 PyTorch 模型。
            train_loader (DataLoader): 训练数据加载器。
            val_loader (DataLoader): 验证数据加载器。
            test_loader (DataLoader): 测试数据加载器。
            config (dict): 包含 lr, weight_decay, epochs 等超参数的配置。
            exp_name (str): 实验名称，用于指标跟踪。
        """
        self.config = config
        self.exp_name = exp_name
        
        # 设备自适应选择：MacOS M4 优先使用 mps，Linux/Windows 若有英伟达显卡则用 cuda，否则 fallback 到 cpu
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        if self.device.type == "cpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            
        print(f"Using device: {self.device}")
        
        # 将模型迁移到指定计算设备
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 对于分类任务，统一使用交叉熵损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 使用 AdamW 优化器，它在处理权重衰减 (Weight Decay) 时比标准 Adam 更有效，有助于正则化
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config['lr'], 
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # 初始化指标跟踪器
        self.tracker = ExperimentTracker(exp_name)
        self.epochs = config.get('epochs', 200)
        
        # 在实验开始前，记录并保存模型的详细架构
        self.tracker.log_model_architecture(self.model)

    def train_epoch(self, epoch):
        """
        执行单轮训练。

        Args:
            epoch (int): 当前轮次索引。

        Returns:
            float: 本轮训练的平均损失 (Average Loss)。
        """
        self.model.train()
        running_loss = 0.0
        # 添加 tqdm 进度条以实时监控训练速率和当前的 Batch Loss
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]", leave=False)
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad() # 清除之前的梯度，防止累加
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward() # 反向传播计算梯度
            self.optimizer.step() # 根据梯度更新模型权重
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        return running_loss / len(self.train_loader)

    def validate(self, epoch):
        """
        在验证集上评估当前模型的性能。

        Args:
            epoch (int): 当前轮次索引。

        Returns:
            tuple: (平均验证损失, 验证准确率 %)。
        """
        self.model.eval() # 切换到评估模式，关闭 Dropout 和 BatchNorm 的运行统计更新
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]", leave=False)
        
        with torch.no_grad(): # 评估阶段禁用法梯度计算，以节省内存并提速
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                # 统计分类正确的数量
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        return running_loss / len(self.val_loader), val_acc

    def test(self):
        """
        在测试集上执行最终评估。

        Returns:
            float: 测试集准确率 (%)。
        """
        self.model.eval()
        correct = 0
        total = 0
        pbar = tqdm(self.test_loader, desc="Testing", leave=True)
        
        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        test_acc = 100 * correct / total
        # 记录最终结果到文件
        self.tracker.log_test_acc(test_acc)
        return test_acc

    def save_checkpoint(self, epoch, best_val_acc):
        """
        保存断点续训所需的完整状态 (Checkpoint)。

        Args:
            epoch (int): 当前完成的轮次。
            best_val_acc (float): 截至目前为止的最佳验证集准确率。
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'config': self.config
        }
        path = os.path.join(self.tracker.save_dir, "checkpoint.pth")
        torch.save(checkpoint, path)
        # 同时也更新各轮产生的 CSV，确保指标同步
        self.tracker.save_results()

    def load_checkpoint(self):
        """
        尝试从本地加载 checkpoint.pth。

        Returns:
            tuple: (起始轮次, 历史最佳准确率)。若无 checkpoint，返回 (0, 0.0)。
        """
        path = os.path.join(self.tracker.save_dir, "checkpoint.pth")
        if os.path.exists(path):
            try:
                # 针对 MacOS MPS 加速，加载时需要映射到对应设备
                checkpoint = torch.load(path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # 恢复指标历史
                _ = self.tracker.resume()
                
                print(f"--> Successfully resumed from checkpoint: {path} (Epoch {checkpoint['epoch']})")
                return checkpoint['epoch'], checkpoint['best_val_acc']
            except Exception as e:
                print(f"--> Error loading checkpoint: {e}. Starting from scratch.")
        return 0, 0.0

    def run(self):
        """
        启动完整的训练流程循环（Train -> Validate -> Test）。
        
        支持断点续训：启动时会自动检查并加载 `checkpoint.pth`。
        """
        # 尝试恢复进度
        start_epoch, best_val_acc = self.load_checkpoint()
        
        # 预设模型保存路径
        best_model_path = os.path.join(self.tracker.save_dir, "best_model.pth")
        latest_model_path = os.path.join(self.tracker.save_dir, "latest_model.pth")

        for epoch in range(start_epoch, self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            # 更新历史记录
            self.tracker.update(epoch + 1, train_loss, val_loss, val_acc)
            
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 记录并保存最佳模型表现
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), best_model_path)
                print(f"--> Best model saved with Acc: {val_acc:.2f}%")
                
            # 每轮结束保存 Checkpoint，用于断点恢复
            self.save_checkpoint(epoch + 1, best_val_acc)
        
        # 训练结束后保存最新权重
        torch.save(self.model.state_dict(), latest_model_path)
        print(f"--> Latest model saved.")

        # 最终保存所有持久化结果并进行测试
        self.tracker.save_results()
        self.test()
