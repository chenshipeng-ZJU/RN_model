import os
# 必须先设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import torch.nn as nn
import torch.optim as optim
from base_model import RN_Model, EEGFeatureExtractor, FNIRSFeatureExtractor
from finetune_model import Model
import lmdb
import pickle
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------- 1. 配置日志与训练参数 --------------------------
# 固定随机种子（复现实验专用）
import random
import numpy as np
def set_seed(seed=42):
    random.seed(seed)          # Python原生随机数
    np.random.seed(seed)       # NumPy随机数
    torch.manual_seed(seed)    # PyTorch CPU随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)        # 单卡GPU随机数
        torch.cuda.manual_seed_all(seed)    # 多卡GPU随机数
        # 禁用CUDA非确定性算法（确保GPU计算可复现）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # 自定义日志格式，只保留日志消息本身
    handlers=[logging.FileHandler('/dev/yowant2_cfs/tianyi/csp/RN_Project/RN_reward/model_weights/rn_model_train.log'), logging.StreamHandler()]
)

# 训练参数
class TrainConfig:
    def __init__(self):
        # 预训练标志 or 微调标志
        self.pretrain = True
        # self.pretrain = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 15
        self.val_batch_size = 5
        self.test_batch_size = 1
        self.epochs = 100
        self.lr = 1e-4
        self.weight_decay = 1e-2
        self.betas = (0.9, 0.999)
        self.data_dir = "/dev/yowant2_cfs/tianyi/csp/RN_Project/data/clean_data/S_1"
        self.use_pretrained_weights = True
        self.dir_pretrained_weights = "/dev/yowant2_cfs/tianyi/csp/RN_Project/RN_reward/model_weights/checkpoints/pretrain_model.pth"
        self.frozen = False     # 冻结预训练模型
        self.pretrain_mask_ratio = 0.5
        
        self.dropout = 0.1
        self.d_model = 256
        self.nhead = 8
        self.num_layers = 4
        self.dim_feedforward = 1024
        self.patience = 150      # 早停耐心值
        self.save_dir = f"/dev/yowant2_cfs/tianyi/csp/RN_Project/RN_reward/model_weights/checkpoints"

        self.eeg_sfreq = 250
        self.fnirs_sfreq = 25
        self.eeg_channels = 2
        self.fnirs_channels = 16
        self.patch_num = 10
        self.win_length=100 
        self.hop_length=50
        self.age = 30
        self.num_of_classes = 4  # 回归任务数（4个输出）

# -------------------------- 2. 数据集定义 --------------------------
class CustomDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        super().__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get(b'__keys__'))[mode]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        # 确保数据形状为 (C, S, T)，适配模型输入
        eeg = torch.from_numpy(pair['eeg']).float().squeeze()  # 去冗余维度
        fnirs = torch.from_numpy(pair['fnirs']).float().squeeze()
        label = torch.from_numpy(pair['label']).float()  # 回归标签（batch_size, 4）
        return eeg, fnirs, label

    def collate(self, batch):
        # 批量堆叠，自动处理维度对齐
        eeg = torch.stack([x[0] for x in batch], dim=0)
        fnirs = torch.stack([x[1] for x in batch], dim=0)
        label = torch.stack([x[2] for x in batch], dim=0)
        # 确保标签维度为 (batch_size, 4)
        if label.ndim == 1:
            label = label.unsqueeze(1).repeat(1, 4)
        return eeg, fnirs, label

class LoadDataset:
    def __init__(self, params):
        self.params = params

    def get_data_loader(self):
        # 精简数据集初始化，复用collate函数
        def get_loader(mode):
            dataset = CustomDataset(self.params.data_dir, mode)
            return DataLoader(
                dataset,
                batch_size=self.params.batch_size if mode == 'train' else self.params.val_batch_size,
                collate_fn=dataset.collate,
                shuffle=mode == 'train',
                pin_memory=True,  # 加速GPU数据传输
                num_workers=16
            )
        return {
            'train': get_loader('train'),
            'val': get_loader('val'),
            'test': get_loader('test')
        }

# -------------------------- 3. 损失函数设计 --------------------------
class MultiTaskRegressionLoss(nn.Module):
    def __init__(self, task_weights=None):
        super().__init__()
        # 每个任务的权重
        self.task_weights = task_weights or [1.0, 1.0, 1.0, 1.0]
        # 每个任务的损失函数
        self.loss_fns = [nn.MSELoss(reduction="mean") for _ in range(len(self.task_weights))]

    def forward(self, pred, target):
        total_loss = 0.0
        for i, (loss_fn, weight) in enumerate(zip(self.loss_fns, self.task_weights)):
            # 每个任务单独计算损失，加权求和
            total_loss += weight * loss_fn(pred[:, i:i+1], target[:, i:i+1])
        return total_loss
class PretrainLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fns = nn.MSELoss(reduction='mean')

    def forward(self, pred, target):
        total_loss = self.loss_fns(pred, target)
        return total_loss

def generate_mask(bz, ch_num, patch_num, mask_ratio, device):
    mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long, device=device)
    mask = mask.bernoulli_(mask_ratio)
    return mask
def normalize_data(x):
    """通道级Z-score归一化,对每个通道的时间维度统计均值/方差"""
    mean = x.mean(dim=-1, keepdim=True)  # (bz, ch, seg, 1)
    std = x.std(dim=-1, keepdim=True)    # (bz, ch, seg, 1)
    std = torch.clamp(std, min=1e-6)     # 避免除零
    return (x - mean) / std

# -------------------------- 4. 训练与验证核心函数 --------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, eeg_extractor, fnirs_extractor, config, epoch):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for eeg, fnirs, label in tqdm(dataloader, desc=f"Train Epoch {epoch+1}/{config.epochs}", leave=False):
        batch_size = eeg.size(0)  # 关键：用当前批次实际样本数（避免最后一批不足 config.batch_size 的情况）
        total_samples += batch_size  # 累加总样本数
        # 数据移至设备
        eeg, fnirs, label = eeg.to(config.device, non_blocking=True), fnirs.to(config.device, non_blocking=True), label.to(config.device, non_blocking=True)        
        eeg = normalize_data(eeg)
        fnirs = normalize_data(fnirs)

        # 特征提取
        eeg_feat, eeg_low, eeg_high = eeg_extractor(eeg)
        fnirs_feat = fnirs_extractor(fnirs)

        # 输入数据维度线性对齐
        fnirs = torch.nn.functional.interpolate(
            fnirs.reshape(config.batch_size * config.fnirs_channels, config.patch_num, config.fnirs_sfreq),
            size=config.eeg_sfreq,
            mode='linear',
            align_corners=False
        ).reshape(config.batch_size, config.fnirs_channels, config.patch_num, config.eeg_sfreq)

        # 前向传播
        optimizer.zero_grad()
        if config.pretrain:
            eeg_mask = generate_mask(config.batch_size, config.eeg_channels, config.patch_num, mask_ratio=config.pretrain_mask_ratio, device=config.device)
            fnirs_mask = generate_mask(config.batch_size, config.fnirs_channels, config.patch_num, mask_ratio=config.pretrain_mask_ratio, device=config.device)
            pred = model(eeg, fnirs, eeg_mask, fnirs_mask, gate_feat=(eeg_low, eeg_high))
            label = torch.cat([eeg, fnirs], dim=1)              # 预训练任务以EEG/Fnirs重建为目标
            label = label.flatten(0, 2)                         # 形状：(batch_size*(ch_eeg + ch_fnirs)*patch, eeg_sfreq)
            loss = criterion(pred, label)
        else:
            pred = model(eeg, fnirs, gate_feat=(eeg_low, eeg_high))    # 无掩码
            loss = criterion(pred, label)         # 微调任务以预测情绪为目标
        # 反向传播与优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪防溢出
        optimizer.step()

        # 统计指标（只累加总损失，不计算平均）
        total_loss += loss.item() * config.batch_size  # 累加批次损失（batch_size * 批次平均损失）

    # 计算平均损失
    avg_loss = total_loss / total_samples
    # 记录epoch级总损失
    logging.info(
        f"Train Epoch {epoch+1} | Train_Loss: {avg_loss:.4f}"
    )
    return avg_loss

def validate(model, dataloader, criterion, eeg_extractor, fnirs_extractor, config, epoch):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for eeg, fnirs, label in tqdm(dataloader, desc=f"Val Epoch {epoch+1}/{config.epochs}", leave=False):
            batch_size = eeg.size(0)  # 用当前批次实际样本数
            total_samples += batch_size  # 累加总样本数
            eeg, fnirs, label = eeg.to(config.device, non_blocking=True), fnirs.to(config.device, non_blocking=True), label.to(config.device, non_blocking=True)
            eeg = normalize_data(eeg)
            fnirs = normalize_data(fnirs)
            # 特征提取
            eeg_feat, eeg_low, eeg_high = eeg_extractor(eeg)
            fnirs_feat = fnirs_extractor(fnirs) 
            # 输入数据维度对齐
            fnirs = torch.nn.functional.interpolate(
                fnirs.reshape(config.val_batch_size * config.fnirs_channels, config.patch_num, config.fnirs_sfreq),
                size=config.eeg_sfreq,
                mode='linear',
                align_corners=False
            ).reshape(config.val_batch_size, config.fnirs_channels, config.patch_num, config.eeg_sfreq)       

            # 前向传播
            if config.pretrain:
                pred = model(eeg, fnirs, gate_feat=(eeg_low, eeg_high))
                label = torch.cat([eeg, fnirs], dim=1)              # 预训练任务以EEG/Fnirs重建为目标
                label = label.flatten(0, 2)
                loss = criterion(pred, label)
            else:
                pred = model(eeg, fnirs, gate_feat=(eeg_low, eeg_high))    
                loss = criterion(pred, label)
            # 统计指标（只累加总损失，不计算平均）
            total_loss += loss.item() * config.val_batch_size  # 累加批次损失（batch_size * 批次平均损失）
    
    # 计算平均损失
    avg_loss = total_loss / total_samples
    logging.info(
        f"Val Epoch {epoch+1} | Val_Loss: {avg_loss:.4f}"
    )
    return avg_loss

# -------------------------- 5. 主训练流程 --------------------------
def main():
    # 0. 初始化参数
    config = TrainConfig()
    
    # 1. 构建数据集和数据加载器（B, C, S, T）
    data_loader = LoadDataset(config).get_data_loader()
    train_loader = data_loader['train']
    val_loader = data_loader['val']
    test_loader = data_loader['test']
    logging.info(f"训练集样本数: {len(train_loader.dataset)}, 验证集样本数: {len(val_loader.dataset)}, 测试集样本数: {len(test_loader.dataset)}")

    # 2. ===============================特征提取器(待优化，添加专家模型组作为门控)！！！===============================
    eeg_extractor = EEGFeatureExtractor(config.eeg_sfreq, config.win_length, config.hop_length)
    fnirs_extractor = FNIRSFeatureExtractor(config.age)

    # 3. 初始化模型
    if config.pretrain:
        model = RN_Model(num_layers=config.num_layers).to(config.device)
        logging.info(f"预训练模型--参数: {sum(p.numel() for p in model.parameters()):,}")   
        # 4. 初始化损失函数、优化器和学习率调度器     
        criterion = PretrainLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=config.betas, eps=1e-8)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    else:
        model = Model(config).to(config.device)
        logging.info(f"微调模型--参数: {sum(p.numel() for p in model.parameters()):,}")
        # 4. 初始化损失函数、优化器和学习率调度器
        criterion = MultiTaskRegressionLoss(task_weights=[1, 1, 1, 1])
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=config.betas, eps=1e-8)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 5. 早停设置（适配回归：监控验证损失）
    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs(config.save_dir, exist_ok=True)
    pretrain_model_path = os.path.join(config.save_dir, "pretrain_model.pth")
    finetune_model_path = os.path.join(config.save_dir, "finetune_model.pth")

    train_losses = []  # 记录每个epoch的训练损失
    val_losses = []    # 记录每个epoch的验证损失
    # 6. 开始训练
    logging.info("============================ 开始训练(pretrain/finetune) ================================")
    for epoch in range(config.epochs):
        # 训练
        epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, eeg_extractor, fnirs_extractor, config, epoch)
        # 验证
        val_loss = validate(model, val_loader, criterion, eeg_extractor, fnirs_extractor, config, epoch)
        # 绘图
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        plt.clf()  # 清空画布
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', linewidth=2)
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', linewidth=2, linestyle='--')
        plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.title('Training & Validation Loss (Real-Time)'), plt.legend()
        if config.pretrain:
            plt.savefig(os.path.join(config.save_dir, 'pretrain_loss_curve_real_time.png'), dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(config.save_dir, 'finetune_loss_curve_real_time.png'), dpi=150, bbox_inches='tight')
        plt.close()  # 关闭画布，避免内存泄漏
        
        # 学习率调度（监控验证损失）
        scheduler.step(val_loss)
        logging.info(f"学习率更新：{optimizer.param_groups[0]['lr']}")

        # 保存最佳模型（损失下降则更新）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if config.pretrain:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }, pretrain_model_path)
                logging.info(f"保存最佳模型（验证损失: {best_val_loss:.4f}）到: {pretrain_model_path}")
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }, finetune_model_path)
                logging.info(f"保存最佳模型（验证损失: {best_val_loss:.4f}）到: {finetune_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(f"早停计数器: {patience_counter}/{config.patience}")

        # 早停触发
        if patience_counter >= config.patience:
            logging.info(f"早停触发！最佳验证损失: {best_val_loss:.4f}")
            break

    # 6. 训练结束
    logging.info("============================ 结束训练 ================================")
    logging.info(f"最终最佳验证损失: {best_val_loss:.4f}")

    # 测试集评估
    test_loss = validate(model, val_loader, criterion, eeg_extractor, fnirs_extractor, config, epoch=-1)
    logging.info(f"测试集平均损失: {test_loss:.4f}")

if __name__ == '__main__':
    set_seed(seed=42)
    main()