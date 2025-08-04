import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
from sklearn.metrics import accuracy_score, roc_curve
import numpy as np
from tqdm import tqdm
import json
import os
import time
import csv
import random
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from dropin import DropInWav2Vec2

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

def compute_eer(y_true, y_scores):
    """
    计算等错误率 (Equal Error Rate)
    Args:
        y_true: 真实标签 (0或1)
        y_scores: 预测分数 (概率值)
    Returns:
        eer: 等错误率
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

from transformers import Wav2Vec2FeatureExtractor
import torch
import numpy as np

from transformers import Wav2Vec2FeatureExtractor
import torch
import numpy as np

class Wav2Vec2Collator:
    """数据整理器，用于批处理"""
    def __init__(self, processor, max_length=16000*4):
        # 创建一个不进行归一化的feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=False,  # 禁用归一化
            return_attention_mask=False  # 不返回attention_mask，让模型自己处理
        )
        self.max_length = max_length
    
    def __call__(self, batch):
        # 分离音频和标签
        waveforms = []
        labels = []
        
        for item in batch:
            # item[0] 是音频tensor，shape: [1, length] 或 [length]
            # item[1] 是标签
            audio = item[0]
            label = item[1]
            
            # 确保音频是1D的
            if audio.dim() > 1:
                audio = audio.squeeze()  # 移除所有大小为1的维度
            
            # 转换为numpy数组（feature_extractor期望numpy数组）
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            
            # 确保长度不超过最大长度
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            
            waveforms.append(audio)
            labels.append(label)
        
        # 使用feature_extractor处理批次数据（不进行归一化）
        try:
            inputs = self.feature_extractor(
                waveforms,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True
            )
            
            # 检查feature_extractor输出中的NaN
            if torch.isnan(inputs.input_values).any():
                print("Warning: NaN found in feature_extractor output, replacing with zeros")
                inputs.input_values = torch.nan_to_num(inputs.input_values, nan=0.0)
            
            return {
                'input_values': inputs.input_values,
                'attention_mask': None,  # 不提供attention_mask
                'labels': torch.tensor(labels, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error in collator: {e}")
            # 返回安全的默认值
            return {
                'input_values': torch.zeros(len(batch), self.max_length),
                'attention_mask': None,  # 不提供attention_mask
                'labels': torch.tensor(labels, dtype=torch.long)
            }

def train_model(model, train_loader, dev_loader, num_epochs=10, lr=1e-4):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=3
    )
    
    best_eer = float('inf')
    best_model_state = None
    train_losses = []
    dev_eers = []
    forward_times = []
    backward_times = []
    learning_rates = []  # 记录学习率变化
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        epoch_forward_time = 0
        epoch_backward_time = 0
        train_bar = tqdm(train_loader, desc="Evaluating")
        
        for batch_idx, (data, target) in enumerate(train_bar):
            try:
                input_values = data.to(device)
                attention_mask = None
                labels = target.to(device)

                optimizer.zero_grad()
                
                # 测量前向传播时间
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                forward_start = time.time()
                
                outputs = model(input_values.squeeze(1))
                
                
                loss = criterion(outputs, labels)

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                forward_end = time.time()
                epoch_forward_time += (forward_end - forward_start)
                
                # 测量反向传播时间
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                backward_start = time.time()
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                backward_end = time.time()
                epoch_backward_time += (backward_end - backward_start)
                
                total_loss += loss.item()
                train_bar.set_postfix({'loss': loss.item()})
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / len(train_loader)
        avg_forward_time = epoch_forward_time / len(train_loader)
        avg_backward_time = epoch_backward_time / len(train_loader)
        
        train_losses.append(avg_loss)
        forward_times.append(avg_forward_time)
        backward_times.append(avg_backward_time)
        
        # 验证阶段
        dev_eer, dev_loss = evaluate_model_eer(model, dev_loader, device)
        dev_eers.append(dev_eer)
        # 更新学习率调度器
        scheduler.step(dev_loss)
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, Dev EER = {dev_eer:.4f}, '
              f'LR = {current_lr:.2e}, '
              f'Forward Time = {avg_forward_time:.4f}s, Backward Time = {avg_backward_time:.4f}s')
        
        # 保存最佳模型（EER越小越好）
        if dev_eer < best_eer:
            best_eer = dev_eer
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, 'best_model.pth')
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'train_losses': train_losses,
        'dev_eers': dev_eers,
        'best_eer': best_eer,
        'forward_times': forward_times,
        'backward_times': backward_times,
        'learning_rates': learning_rates,
        'avg_forward_time': np.mean(forward_times),
        'avg_backward_time': np.mean(backward_times)
    }

def evaluate_model_eer(model, data_loader, device):
    """使用EER评估模型"""
    model.eval()
    all_predictions = []
    all_scores = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        
        criterion = nn.CrossEntropyLoss()

        for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc="Evaluating")):
            input_values = data.to(device)
            attention_mask = None
            labels = target.to(device)
            
            outputs = model(input_values.squeeze())
            
            # 获取softmax概率
            probs = torch.softmax(outputs, dim=-1)
            predictions = torch.argmax(outputs, dim=-1)
            
            # 取正类的概率作为分数
            scores = probs[:, 1]  # 假设标签1是正类
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
                
    
    if len(all_predictions) > 0:
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else float('inf')
        eer = compute_eer(all_labels, all_scores)
        return eer, avg_loss
    else:
        return float('inf')

def test_model_eer(model, test_loader, device):
    """测试模型并返回详细指标"""
    model.eval()
    all_predictions = []
    all_scores = []
    all_labels = []
    total_time = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing")):
            input_values = data.to(device)
            attention_mask = None
            labels = target.to(device)

            # 测量推理时间
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            outputs = model(input_values.squeeze(1))
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            total_time += (end_time - start_time)
            
            probs = torch.softmax(outputs, dim=-1)
            predictions = torch.argmax(outputs, dim=-1)
            scores = probs[:, 1]
            
            all_predictions.extend(predictions.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if len(all_predictions) > 0:
        eer = compute_eer(all_labels, all_scores)
        accuracy = accuracy_score(all_labels, all_predictions)
        avg_inference_time = total_time / len(test_loader)
        
        return {
            'eer': eer,
            'accuracy': accuracy,
            'inference_time': avg_inference_time
        }
    else:
        return {
            'eer': float('inf'),
            'accuracy': 0.0,
            'inference_time': 0.0
        }

def save_results_to_csv(all_results, filename='results.csv'):
    """保存所有结果到CSV文件"""
    fieldnames = [
        'experiment_name', 'mode', 'dropin_config',
        'total_params', 'trainable_params', 'frozen_params',
        'best_dev_eer', 'test_eer', 'test_accuracy',
        'avg_forward_time', 'avg_backward_time', 'avg_inference_time'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            writer.writerow(result)
    
    print(f"Results saved to {filename}")

def main():
    # 数据路径
    data_root = "/data/yl7622/dropin/dataset/LA"
    train_protocol = os.path.join(data_root, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
    dev_protocol = os.path.join(data_root, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
    eval_protocol = os.path.join(data_root, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
    
    # 加载数据集
    from dataset import ASVDataset
    
    train_dataset = ASVDataset(data_root, train_protocol)
    dev_dataset = ASVDataset(data_root, dev_protocol)
    eval_dataset = ASVDataset(data_root, eval_protocol)
    
    # 创建数据加载器
    processor = Wav2Vec2Processor.from_pretrained("/data/yl7622/dropin/wav2vec/wav2vec2_model/wav2vec-base")
    collator = Wav2Vec2Collator(processor, max_length=16000*4)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=4)
    

    # 实验配置
    experiments = [
        {
            'name': 'dropin_frozen',
            'dropin_config': {5: 768 * 2},  # 第5层扩展到1024维
            'mode': 'frozen'
        },
        {
            'name': 'dropin_unfrozen',
            'dropin_config': {5: 768 * 2},
            'mode': 'unfrozen'
        }
    ]
    
    all_results = []
    
    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"Running experiment: {exp['name']}")
        print(f"{'='*50}")
        
        # 创建模型

        model = DropInWav2Vec2(
            pretrained_model="/data/yl7622/dropin/wav2vec/wav2vec2_model/wav2vec-base",
            num_labels=2,
            dropin_config=exp['dropin_config'],
            mode=exp['mode']
        )
            
        # 计算参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
        
        # 训练模型
        print("Starting training...")
        train_results = train_model(
            model, train_loader, dev_loader, 
            num_epochs=30,
            lr=1e-4
        )
        
        # 测试模型
        print("Testing model...")
        test_results = test_model_eer(model, eval_loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 整合结果
        result = {
            'experiment_name': exp['name'],
            'mode': exp['mode'],
            'dropin_config': str(exp['dropin_config']),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'best_dev_eer': train_results['best_eer'],
            'test_eer': test_results['eer'],
            'test_accuracy': test_results['accuracy'],
            'avg_forward_time': train_results['avg_forward_time'],
            'avg_backward_time': train_results['avg_backward_time'],
            'avg_inference_time': test_results['inference_time']
        }
        
        all_results.append(result)
        
        # 保存详细训练结果
        detailed_results = {
            'experiment_config': exp,
            'model_stats': {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'frozen_params': frozen_params
            },
            'training_results': train_results,
            'test_results': test_results
        }
        
        with open(f"detailed_results_{exp['name']}.json", 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"Results for {exp['name']}:")
        print(f"  Best Dev EER: {train_results['best_eer']:.4f}")
        print(f"  Test EER: {test_results['eer']:.4f}")
        print(f"  Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"  Avg Forward Time: {train_results['avg_forward_time']:.4f}s")
        print(f"  Avg Backward Time: {train_results['avg_backward_time']:.4f}s")
        print(f"  Avg Inference Time: {test_results['inference_time']:.4f}s")

        if exp['name'] == 'baseline':
            save_path = '/data/yl7622/dropin/wav2vec2_dropin/saved_model/baseline_wav2vec2.pt'
            torch.save({
                'base_model_state_dict': model.base_model.state_dict(),
                'classifier_state_dict': model.classifier.state_dict()
            }, save_path)
            print(f"Baseline model saved at {save_path}")
    
    # 保存汇总结果到CSV
    save_results_to_csv(all_results, 'experiment_results.csv')
    
    # 打印比较结果
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*60}")
    print(f"{'Experiment':<20} {'Test EER':<10} {'Accuracy':<10} {'Trainable Params':<15} {'Inference Time':<15}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['experiment_name']:<20} "
              f"{result['test_eer']:<10.4f} "
              f"{result['test_accuracy']:<10.4f} "
              f"{result['trainable_params']:<15,} "
              f"{result['avg_inference_time']:<15.4f}")
    
    print(f"\nDetailed results saved to experiment_results.csv")
    print("Individual experiment details saved to detailed_results_*.json")

if __name__ == "__main__":
    main()