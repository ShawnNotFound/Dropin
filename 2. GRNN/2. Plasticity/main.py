# main.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import gc
import traceback
import random
import torchaudio
from tqdm import tqdm

# Import our modules
from lc_grnn import LCGRNN, SVMLoss
from dataset import ASVDataset, seed_worker, get_gpu_memory_usage
from training import (
    train, validate, plot_metrics, plot_confusion_matrix, run_experiment
)

# Set all random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Random seed set to: {SEED} for full reproducibility")


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

def main():
    # Hyperparameters
    num_epochs = 15
    batch_size = 32
    learning_rate = 0.0001
    
    # Dataset paths
    data_root = "/data/yl7622/dropin/dataset/LA"
    train_protocol = os.path.join(data_root, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
    dev_protocol = os.path.join(data_root, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
    eval_protocol = os.path.join(data_root, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    try:
        # Create datasets
        print("Loading datasets...")
        train_dataset = ASVDataset(data_root, train_protocol)
        dev_dataset = ASVDataset(data_root, dev_protocol)
        eval_dataset = ASVDataset(data_root, eval_protocol)
        
        # Create data loaders with fixed seeds for workers
        g = torch.Generator()
        g.manual_seed(SEED)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g
        )
        
        val_loader = DataLoader(
            dev_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g
        )
        
        test_loader = DataLoader(
            eval_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g
        )
        # ========== Experiment 1: Baseline Model ==========
        print("\n\n" + "="*50)
        print("EXPERIMENT 1: Training Baseline Model")
        print("="*50)
        
        
        input_size = 20  # LC feature dimension
        hidden_sizes = [64, 64, 32, 32]  # GRNN hidden layer sizes
        
        # Create baseline model (without dropin)
        baseline_model = LCGRNN(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropin=False,
            dropin_sizes=[0, 0, 0, 0],
            num_classes=2
        ).to(device)
        
        # Run experiment
        _, baseline_model = run_experiment(
            'baseline', 
            baseline_model, 
            train_loader, 
            val_loader, 
            test_loader, 
            num_epochs // 3, 
            learning_rate=1e-4,
            start_epoch=1,
            return_model=True
        )

        # Define dropin sizes for each layer
        dropin_sizes = [0, 64, 0, 0]  # Additional neurons to add to each layer
        
        # Create dropin model with frozen layers
        dropin_model = LCGRNN(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropin=True,
            dropin_sizes=dropin_sizes,
            num_classes=2
        ).to(device)
        expand_and_load_encoder_layer(baseline_model, dropin_model)
        _, dropin_model = run_experiment(
            'dropin', 
            dropin_model, 
            train_loader, 
            val_loader, 
            test_loader, 
            num_epochs // 3, 
            learning_rate,
            start_epoch=1,
            return_model=True
        )
        
        truncate_and_load_encoder_layer(dropin_model, baseline_model)
        baseline_metrics, baseline_model_again = run_experiment(
            'baseline_again', 
            baseline_model, 
            train_loader, 
            val_loader, 
            test_loader, 
            num_epochs // 3, 
            learning_rate,
            start_epoch=1,
            return_model=True
        )
        # Additional analysis
        print("\nKey Observations:")

        print(f"\nPerformance Metrics:")

        baseline_acc = baseline_metrics['test_acc']
        
        baseline_eer = baseline_metrics['test_eer']
        
        print(f"   - Accuracy:")
        print(f"Baseline: {baseline_acc:.2f}% test accuracy")
        
        print(f"   - Equal Error Rate (EER):")
        print(f"Baseline: {baseline_eer:.4f} EER difference")
 
    except Exception as e:
        print(f"Error during experiment: {str(e)}")
        traceback.print_exc()
        
        # Try to debug audio loading issue if that's the problem
        try:
            print("\nTrying to debug audio loading issue...")
            # Try to locate one audio file
            for subset in ["ASVspoof2019_LA_train", "ASVspoof2019_LA_dev", "ASVspoof2019_LA_eval"]:
                flac_dir = os.path.join(data_root, subset, "flac")
                if os.path.exists(flac_dir):
                    files = os.listdir(flac_dir)
                    if files:
                        sample_audio_path = os.path.join(flac_dir, files[0])
                        print(f"Sample audio path: {sample_audio_path}")
                        
                        # Check file
                        if not os.path.exists(sample_audio_path):
                            print(f"File does not exist!")
                        else:
                            print(f"File size: {os.path.getsize(sample_audio_path)} bytes")
                            
                            # Try loading with torchaudio
                            try:
                                waveform, sample_rate = torchaudio.load(sample_audio_path)
                                print(f"Successfully loaded with torchaudio. Shape: {waveform.shape}, Sample rate: {sample_rate}")
                                break
                            except Exception as e:
                                print(f"Failed to load with torchaudio: {str(e)}")
                                
                                # Try loading with librosa
                                try:
                                    import librosa
                                    waveform, sample_rate = librosa.load(sample_audio_path, sr=None)
                                    print(f"Successfully loaded with librosa. Shape: {waveform.shape}, Sample rate: {sample_rate}")
                                    break
                                except Exception as e:
                                    print(f"Failed to load with librosa too: {str(e)}")
                else:
                    print(f"Directory {flac_dir} does not exist")
        except Exception as debug_e:
            print(f"Debug attempt failed: {str(debug_e)}")

if __name__ == "__main__":
    main()