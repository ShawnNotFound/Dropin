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
from models.lc_grnn import LCGRNN, SVMLoss
from utils.dataset import ASVDataset, seed_worker, get_gpu_memory_usage
from utils.training import (
    train, validate, plot_metrics, plot_timing_comparison, 
    plot_memory_usage, plot_confusion_matrix, 
    load_parameters_from_baseline, run_experiment
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

def main():
    # Hyperparameters
    num_epochs = 10
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
        print("EXPERIMENT 1: Training Baseline LC-GRNN+SVM Model")
        print("="*50)
        
        # Define model hyperparameters
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
        baseline_metrics, baseline_model = run_experiment(
            'baseline', 
            baseline_model, 
            train_loader, 
            val_loader, 
            test_loader, 
            num_epochs, 
            learning_rate=1e-4,
            start_epoch=1,
            return_model=True
        )
        
        # ========== Experiment 2: Dropin Model with Frozen Layers ==========
        print("\n\n" + "="*50)
        print("EXPERIMENT 2: Training Dropin LC-GRNN+SVM with Frozen Layers (using baseline parameters)")
        print("="*50)
        
        # Define dropin sizes for each layer
        dropin_sizes = [0, 64, 0, 0]  # Additional neurons to add to each layer
        
        # Create dropin model with frozen layers
        dropin_frozen_model = LCGRNN(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropin=True,
            dropin_sizes=dropin_sizes,
            num_classes=2
        ).to(device)
        
        # Load baseline parameters to dropin model, handling shape mismatches properly
        print("Loading parameters from baseline model to dropin_frozen model...")
        params_loaded, params_skipped = load_parameters_from_baseline(baseline_model, dropin_frozen_model)
        
        # Freeze original layers after loading parameters
        dropin_frozen_model.freeze_original_layers()
        
        # Train only the dropin layers
        dropin_frozen_metrics = run_experiment(
            'dropin_frozen', 
            dropin_frozen_model, 
            train_loader, 
            val_loader, 
            test_loader, 
            num_epochs, 
            learning_rate=1e-4,
            start_epoch=1
        )
        
        # ========== Experiment 3: Dropin Model with All Layers Trainable (from scratch) ==========
        print("\n\n" + "="*50)
        print("EXPERIMENT 3: Training Dropin LC-GRNN+SVM with All Layers Trainable (from scratch)")
        print("="*50)
        
        # Create a new model with dropin but with all layers trainable
        dropin_unfrozen_model = LCGRNN(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropin=True,
            dropin_sizes=dropin_sizes,
            num_classes=2
        ).to(device)
        
        # Make sure all layers are trainable
        dropin_unfrozen_model.unfreeze_all_layers()
        
        # Train the model from scratch
        dropin_unfrozen_metrics = run_experiment(
            'dropin_unfrozen', 
            dropin_unfrozen_model, 
            train_loader, 
            val_loader, 
            test_loader, 
            num_epochs, 
            learning_rate=1e-4,
            start_epoch=1
        )
        
        # ========== Compare Results ==========
        # Combine all metrics
        all_metrics = {
            'baseline': baseline_metrics,
            'dropin_frozen': dropin_frozen_metrics,
            'dropin_unfrozen': dropin_unfrozen_metrics
        }
        
        # Create comparison plots
        plot_metrics(all_metrics, save_path='model_comparison.png')
        plot_memory_usage(all_metrics, save_path='memory_comparison.png')
        plot_timing_comparison(all_metrics, save_path='timing_comparison.png')
        
        # Plot EER comparison
        plt.figure(figsize=(10, 6))
        for model_name, metrics in all_metrics.items():
            if 'val_eer' in metrics:
                epochs = range(metrics['start_epoch'], metrics['start_epoch'] + len(metrics['val_eer']))
                plt.plot(epochs, metrics['val_eer'], label=model_name)
        plt.title('Validation Equal Error Rate (EER)')
        plt.xlabel('Epochs')
        plt.ylabel('EER')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('eer_comparison.png')
        plt.close()
        
        # Create summary table
        summary = pd.DataFrame([
            {
                'Model': 'Baseline',
                'Parameters': baseline_metrics['param_count'],
                'Trainable Params': baseline_metrics['trainable_param_count'],
                'Best Val Acc': max(baseline_metrics['val_acc']),
                'Best Val EER': min(baseline_metrics['val_eer']),
                'Test Acc': baseline_metrics['test_acc'],
                'Test EER': baseline_metrics['test_eer'],
                'Training Time (s)': baseline_metrics['training_time'],
                'Time/Epoch (s)': baseline_metrics['time_per_epoch'],
                'Forward Time (ms)': baseline_metrics['timing']['forward_avg'] * 1000,
                'Backward Time (ms)': baseline_metrics['timing']['backward_avg'] * 1000,
                'Max Memory (MB)': baseline_metrics['memory']['max'],
                'Avg Memory (MB)': baseline_metrics['memory']['avg']
            },
            {
                'Model': 'Dropin Frozen',
                'Parameters': dropin_frozen_metrics['param_count'],
                'Trainable Params': dropin_frozen_metrics['trainable_param_count'],
                'Best Val Acc': max(dropin_frozen_metrics['val_acc']),
                'Best Val EER': min(dropin_frozen_metrics['val_eer']),
                'Test Acc': dropin_frozen_metrics['test_acc'],
                'Test EER': dropin_frozen_metrics['test_eer'],
                'Training Time (s)': dropin_frozen_metrics['training_time'],
                'Time/Epoch (s)': dropin_frozen_metrics['time_per_epoch'],
                'Forward Time (ms)': dropin_frozen_metrics['timing']['forward_avg'] * 1000,
                'Backward Time (ms)': dropin_frozen_metrics['timing']['backward_avg'] * 1000,
                'Max Memory (MB)': dropin_frozen_metrics['memory']['max'],
                'Avg Memory (MB)': dropin_frozen_metrics['memory']['avg']
            },
            {
                'Model': 'Dropin Unfrozen',
                'Parameters': dropin_unfrozen_metrics['param_count'],
                'Trainable Params': dropin_unfrozen_metrics['trainable_param_count'],
                'Best Val Acc': max(dropin_unfrozen_metrics['val_acc']),
                'Best Val EER': min(dropin_unfrozen_metrics['val_eer']),
                'Test Acc': dropin_unfrozen_metrics['test_acc'],
                'Test EER': dropin_unfrozen_metrics['test_eer'],
                'Training Time (s)': dropin_unfrozen_metrics['training_time'],
                'Time/Epoch (s)': dropin_unfrozen_metrics['time_per_epoch'],
                'Forward Time (ms)': dropin_unfrozen_metrics['timing']['forward_avg'] * 1000,
                'Backward Time (ms)': dropin_unfrozen_metrics['timing']['backward_avg'] * 1000,
                'Max Memory (MB)': dropin_unfrozen_metrics['memory']['max'],
                'Avg Memory (MB)': dropin_unfrozen_metrics['memory']['avg']
            }
        ])
        
        summary.to_csv('experiment_summary.csv', index=False)
        print("\nExperiment Summary:")
        print(summary)
        
        # Additional analysis
        print("\nKey Observations:")
        
        print(f"1. Parameter Efficiency:")
        dropin_frozen_trainable = dropin_frozen_metrics['trainable_param_count']
        dropin_unfrozen_trainable = dropin_unfrozen_metrics['trainable_param_count']
        baseline_trainable = baseline_metrics['trainable_param_count']
        
        print(f"   - Dropin (Frozen) uses {dropin_frozen_trainable / dropin_unfrozen_trainable:.2%} of the trainable parameters compared to Dropin (Unfrozen)")
        print(f"   - Dropin (Frozen) uses {dropin_frozen_trainable / baseline_trainable:.2%} of the trainable parameters compared to Baseline")
        
        print(f"\n2. Training Speed:")
        print(f"   - Dropin (Frozen) is {dropin_unfrozen_metrics['training_time'] / dropin_frozen_metrics['training_time']:.2f}x faster than Dropin (Unfrozen)")
        print(f"   - Dropin (Frozen) is {baseline_metrics['training_time'] / dropin_frozen_metrics['training_time']:.2f}x faster than Baseline")
        
        print(f"\n3. Forward/Backward Pass Analysis:")
        print(f"   - Forward Pass: Dropin (Frozen) takes {dropin_frozen_metrics['timing']['forward_avg'] / baseline_metrics['timing']['forward_avg']:.2f}x the time of Baseline")
        print(f"   - Backward Pass: Dropin (Frozen) takes {dropin_frozen_metrics['timing']['backward_avg'] / baseline_metrics['timing']['backward_avg']:.2f}x the time of Baseline")
        print(f"   - Forward Pass: Dropin (Unfrozen) takes {dropin_unfrozen_metrics['timing']['forward_avg'] / baseline_metrics['timing']['forward_avg']:.2f}x the time of Baseline")
        print(f"   - Backward Pass: Dropin (Unfrozen) takes {dropin_unfrozen_metrics['timing']['backward_avg'] / baseline_metrics['timing']['backward_avg']:.2f}x the time of Baseline")
        
        print(f"\n4. Memory Usage:")
        print(f"   - Dropin (Frozen) uses {dropin_frozen_metrics['memory']['max'] / dropin_unfrozen_metrics['memory']['max']:.2%} of the memory compared to Dropin (Unfrozen)")
        print(f"   - Dropin (Frozen) uses {dropin_frozen_metrics['memory']['max'] / baseline_metrics['memory']['max']:.2%} of the memory compared to Baseline")
        
        print(f"\n5. Performance Metrics:")
        dropin_frozen_acc = dropin_frozen_metrics['test_acc']
        dropin_unfrozen_acc = dropin_unfrozen_metrics['test_acc']
        baseline_acc = baseline_metrics['test_acc']
        
        dropin_frozen_eer = dropin_frozen_metrics['test_eer']
        dropin_unfrozen_eer = dropin_unfrozen_metrics['test_eer']
        baseline_eer = baseline_metrics['test_eer']
        
        print(f"   - Accuracy:")
        print(f"     * Dropin (Frozen) vs Dropin (Unfrozen): {dropin_frozen_acc - dropin_unfrozen_acc:.2f}% test accuracy difference")
        print(f"     * Dropin (Frozen) vs Baseline: {dropin_frozen_acc - baseline_acc:.2f}% test accuracy difference")
        
        print(f"   - Equal Error Rate (EER):")
        print(f"     * Dropin (Frozen) vs Dropin (Unfrozen): {dropin_frozen_eer - dropin_unfrozen_eer:.4f} EER difference")
        print(f"     * Dropin (Frozen) vs Baseline: {dropin_frozen_eer - baseline_eer:.4f} EER difference")
        print(f"     * Note: Lower EER is better")
        
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