# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Config
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from dataset import ASVDataset
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderLayer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
import math
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variables (some libraries may rely on this)
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[info] Seed set to {seed}")

set_seed(42)
class ProjectedEncoderLayer(Wav2Vec2EncoderLayer):
    def __init__(self, config, output_dim=768):
        super().__init__(config)
        # self.layer = Wav2Vec2EncoderLayer(config)
        self.projector = nn.Linear(config.hidden_size, output_dim)
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        out = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        # last_hidden_state, optional attn weights
        hidden = out[0]
        projected = self.projector(hidden)
        return (projected,)

def expand_and_load_encoder_layer(origin_layer, new_layer):
    old_sd = origin_layer.state_dict()
    new_sd = new_layer.state_dict()
    for k in new_sd.keys():
        if k not in old_sd:
            print(f"[skip] {k} not found in old layer")
            continue

        old_param = old_sd[k]
        new_param = new_sd[k]

        # 判断维度对不上的情况
        if old_param.shape == new_param.shape:
            new_sd[k] = old_param
        elif len(old_param.shape) == 2:
            # Linear weights: expand top-left corner
            new_sd[k][:old_param.shape[0], :old_param.shape[1]] = old_param
        elif len(old_param.shape) == 1:
            # Bias / LayerNorm
            new_sd[k][:old_param.shape[0]] = old_param
        else:
            print(f"[warn] Shape mismatch for {k}: old {old_param.shape}, new {new_param.shape}")

    new_layer.load_state_dict(new_sd, strict=False)

def truncate_and_load_encoder_layer(origin_layer, new_layer):
    old_sd = origin_layer.state_dict()
    new_sd = new_layer.state_dict()

    new_trunc_sd = {}

    for k in new_sd.keys():
        if k not in old_sd:
            print(f"[skip] {k} not found in origin_layer")
            continue

        old_param = old_sd[k]
        new_param = new_sd[k]

        if old_param.shape == new_param.shape:
            new_trunc_sd[k] = old_param
        elif len(old_param.shape) == 2:
            # Linear weights
            new_trunc_sd[k] = old_param[:new_param.shape[0], :new_param.shape[1]]
        elif len(old_param.shape) == 1:
            # Biases / LayerNorm
            new_trunc_sd[k] = old_param[:new_param.shape[0]]
        else:
            print(f"[warn] {k} shape mismatch: old {old_param.shape}, new {new_param.shape}")
            continue

    new_layer.load_state_dict(new_trunc_sd, strict=False)


class Wav2Vec2WithCustomLayer(nn.Module):
    def __init__(self, model_state_dict, pretrained_model_config="/data/yl7622/dropin/wav2vec/wav2vec2_model", num_labels=2):
        super().__init__()
        
        config = Wav2Vec2Config(
            hidden_size=256,        # 改成你想要的输出维度
            num_hidden_layers=12,   # 其他参数也可以调整
            intermediate_size=512,
            num_attention_heads=4
        )

        # 初始化模型（不会加载预训练参数）
        config.layerdrop=0.0
        base_model = Wav2Vec2Model(config)

        # 加载预训练参数
        state_dict = model_state_dict
        
        self.wav2vec2 = base_model
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.wav2vec2.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )
        self.load_state_dict(state_dict)
        # 替换第6层（第7层）为自定义层，第5层feedforward
        # config_dropin_before = Wav2Vec2Config.from_dict(config.to_dict())
        # config_dropin_before.layerdrop = 0.0

        layer2_dropin = ProjectedEncoderLayer(config, output_dim=768)
        layer5_dropin = ProjectedEncoderLayer(config, output_dim=768)
        layer10_dropin = ProjectedEncoderLayer(config, output_dim=768)
        
        config_dropin=Wav2Vec2Config(
                hidden_size=768,
                num_attention_heads=12,
                intermediate_size=2048,
                hidden_dropout=0.1,
                attention_dropout=0.1,
            )
        
        layer3_dropin = ProjectedEncoderLayer(config_dropin, output_dim=256)

        layer6_dropin = ProjectedEncoderLayer(config_dropin, output_dim=256)
        layer11_dropin = ProjectedEncoderLayer(config_dropin, output_dim=256)
        
        origin_layer2 = base_model.encoder.layers[0]
        expand_and_load_encoder_layer(origin_layer2, layer2_dropin)
        self.wav2vec2.encoder.layers[0] = layer2_dropin  # 注意是索引5

        origin_layer3 = base_model.encoder.layers[1]
        expand_and_load_encoder_layer(origin_layer3, layer3_dropin)
        self.wav2vec2.encoder.layers[1] = layer3_dropin  # 注意是索引6

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        return self.classifier(pooled)

class dropinWav2Vec2ASVClassifier(nn.Module):
    """
    基于Wav2Vec2的ASV欺骗检测分类器
    """
    def __init__(self, model_name="facebook/wav2vec2-base", num_classes=2, freeze_feature_extractor=True):
        super().__init__()
        
        # 加载预训练的Wav2Vec2模型
        config = Wav2Vec2Config(
            hidden_size=768,        # 改成你想要的输出维度
            num_hidden_layers=12,   # 其他参数也可以调整
            intermediate_size=2048,
            num_attention_heads=12
        )

        # 初始化模型（不会加载预训练参数）
        self.wav2vec2 = Wav2Vec2Model(config)
        
        # 是否冻结特征提取器
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.wav2vec2.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_values):
        # 通过Wav2Vec2获取特征
        outputs = self.wav2vec2(input_values)
        
        # 获取最后一层隐藏状态
        hidden_states = outputs.last_hidden_state
        
        # 全局平均池化
        pooled_output = torch.mean(hidden_states, dim=1)
        
        # 分类
        logits = self.classifier(pooled_output)

        return logits

class Wav2Vec2ASVClassifier(nn.Module):
    """
    基于Wav2Vec2的ASV欺骗检测分类器
    """
    def __init__(self, model_name="facebook/wav2vec2-base", num_classes=2, freeze_feature_extractor=True):
        super().__init__()
        
        # 加载预训练的Wav2Vec2模型
        config = Wav2Vec2Config(
            hidden_size=256,        # 改成你想要的输出维度
            num_hidden_layers=12,   # 其他参数也可以调整
            intermediate_size=512,
            num_attention_heads=4
        )

        # 初始化模型（不会加载预训练参数）
        self.wav2vec2 = Wav2Vec2Model(config)
        
        # 是否冻结特征提取器
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.wav2vec2.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_values):
        # 通过Wav2Vec2获取特征
        outputs = self.wav2vec2(input_values)
        
        # 获取最后一层隐藏状态
        hidden_states = outputs.last_hidden_state
        
        # 全局平均池化
        pooled_output = torch.mean(hidden_states, dim=1)
        
        # 分类
        logits = self.classifier(pooled_output)
        
        return logits

def compute_eer(y_true, y_scores):
    """计算等错误率(EER)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.argmin(np.abs(fnr - fpr))]
    eer = fpr[np.argmin(np.abs(fnr - fpr))]
    return eer, eer_threshold

def evaluate_model(model, dataloader, criterion, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Evaluating")):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data.squeeze(1))  # 移除channel维度
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            # 获取预测和分数
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_scores.extend(probabilities[:, 1].cpu().numpy())  # 正类概率
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    eer, eer_threshold = compute_eer(all_labels, all_scores)
    
    return avg_loss, accuracy, eer, eer_threshold, all_labels, all_predictions

def train_model(ori_model, train_loader, val_loader, criterion, device, num_epochs=30, save_dir="checkpoints"):
    """训练模型"""
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    dropin_model = Wav2Vec2WithCustomLayer(ori_model.state_dict(), pretrained_model_config = '/data/yl7622/dropin/wav2vec/wav2vec2_model/wav2vec-base').to(device)
 

    # 训练历史
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_eers = []
    
    best_eer = float('inf')
    best_epoch = 0
    model = ori_model
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 只训练 model1
    for p in ori_model.parameters():
            p.requires_grad = True
    for p in dropin_model.parameters():
        p.requires_grad = False

    save_epoch = False # make sure save at the final stage
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        if epoch == (num_epochs / 3):
            dropin_model = Wav2Vec2WithCustomLayer(ori_model.state_dict(), pretrained_model_config = '/data/yl7622/dropin/wav2vec/wav2vec2_model/wav2vec-base').to(device)
            model = dropin_model
            optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

            # 只训练 model2
            for p in ori_model.parameters():
                p.requires_grad = False
            for p in dropin_model.parameters():
                p.requires_grad = True

            print("Model has been droped in.")
        elif epoch == (num_epochs / 3 * 2):
            save_epoch = True
            truncate_and_load_encoder_layer(dropin_model, ori_model)

            model = ori_model
            optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

            # 只训练 model1
            for p in ori_model.parameters():
                p.requires_grad = True
            for p in dropin_model.parameters():
                p.requires_grad = False
            print("Model has been droped out back.")

        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(device), target.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            output = model(data.squeeze(1))  # 移除channel维度
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 计算训练准确率
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
            
        # 平均训练损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # 验证阶段
        val_loss, val_accuracy, val_eer, eer_threshold, val_labels, val_predictions = evaluate_model(
            model, val_loader, criterion, device
        )
        
        # 记录历史
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_eers.append(val_eer)
        
        logger.info(f'Epoch {epoch+1}:')
        logger.info(f'  Train Loss: {avg_train_loss:.6f}, Train Acc: {train_accuracy:.4f}')
        logger.info(f'  Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy:.4f}')
        logger.info(f'  Val EER: {val_eer:.4f}, EER Threshold: {eer_threshold:.4f}')
        
        # 保存最佳模型
        if val_eer <= best_eer and not math.isinf(eer_threshold) and save_epoch:
            best_eer = val_eer
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_eer': val_eer,
                'val_accuracy': val_accuracy,
            }, os.path.join(save_dir, 'best_model.pth'))
            logger.info(f'New best model saved with EER: {val_eer:.4f}')
        
        # 保存当前epoch的模型
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_eer': val_eer,
            'val_accuracy': val_accuracy,
        }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # 打印分类报告
        if epoch == num_epochs - 1:  # 最后一个epoch打印详细报告
            logger.info('\nClassification Report:')
            logger.info(classification_report(val_labels, val_predictions, 
                                           target_names=['Spoof', 'Bonafide']))
    
    logger.info(f'Training completed. Best EER: {best_eer:.4f} at epoch {best_epoch}')
    
    return train_losses, val_losses, val_accuracies, val_eers

def plot_training_history(train_losses, val_losses, val_accuracies, val_eers, save_dir="checkpoints"):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 准确率曲线
    axes[0, 1].plot(val_accuracies, label='Val Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # EER曲线
    axes[1, 0].plot(val_eers, label='Val EER')
    axes[1, 0].set_title('Validation EER')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('EER')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 移除空的子图
    axes[1, 1].remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.show()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 数据路径
    data_root = "/data/yl7622/dropin/dataset/LA"
    train_protocol = os.path.join(data_root, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
    dev_protocol = os.path.join(data_root, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
    eval_protocol = os.path.join(data_root, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
    
    # 创建数据集
    logger.info("Loading datasets...")
    train_dataset = ASVDataset(data_root, train_protocol)
    val_dataset = ASVDataset(data_root, dev_protocol)
    test_dataset = ASVDataset(data_root, eval_protocol)
    
    # 创建数据加载器
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # 创建模型
    model = Wav2Vec2ASVClassifier(
        model_name="/data/yl7622/dropin/wav2vec/wav2vec2_model/wav2vec-base",
        # model_name="facebook/wav2vec2-base",
        num_classes=2,
        freeze_feature_extractor=False
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    
    # 训练模型
    logger.info("Starting training...")
    train_losses, val_losses, val_accuracies, val_eers = train_model(
        model, train_loader, val_loader, criterion, device, num_epochs=45, save_dir = '/data/yl7622/dropin/wav2vec-small/checkpoints_dropin_index1'
    )
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, val_accuracies, val_eers, save_dir = '/data/yl7622/dropin/wav2vec-small/checkpoints_dropin_index1')
    
    # 加载最佳模型并在测试集上评估
    logger.info("Loading best model for final evaluation...")
    checkpoint = torch.load('checkpoints_dropin_index1/best_model.pth', weights_only=False)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        model = Wav2Vec2WithCustomLayer(model.state_dict(), pretrained_model_config = '/data/yl7622/dropin/wav2vec/wav2vec2_model/wav2vec-base').to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_accuracy, test_eer, test_eer_threshold, test_labels, test_predictions = evaluate_model(
        model, test_loader, criterion, device
    )
    
    logger.info(f"Final Test Results index1:")
    logger.info(f"  Test Loss: {test_loss:.6f}")
    logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"  Test EER: {test_eer:.4f}")
    logger.info(f"  EER Threshold: {test_eer_threshold:.4f}")
    
    logger.info('\nFinal Test Classification Report:')
    logger.info(classification_report(test_labels, test_predictions, 
                                   target_names=['Spoof', 'Bonafide']))

if __name__ == "__main__":
    main()