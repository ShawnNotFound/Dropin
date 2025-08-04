import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
import seaborn as sns
import pandas as pd
import copy
import gc
import traceback
import random

# Set all random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)

# Import the ResNet model from resnet.py
from resnet import ResNet18

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Random seed set to: {SEED} for full reproducibility")

# For tracking GPU memory


def calculate_eer(y_true, y_score):
    """Calculate the Equal Error Rate (EER)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    
    # Find the threshold where FPR and FNR are closest
    eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
    
    return eer, thresholds[eer_threshold_idx]

def seed_worker(worker_id):
    """Function to ensure DataLoader workers use different seeds derived from the base seed"""
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class ASVDataset(Dataset):
    def __init__(self, root_dir, protocol_file, transform=None):
        """
        Args:
            root_dir (string): Directory with all the audio files.
            protocol_file (string): Path to the protocol file that maps audio files to labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.bad_files = []
        
        # Read the protocol file
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                    
                file_id = parts[1]  # File ID (e.g., LA_D_9841321)
                label = parts[-1]   # Label (bonafide or spoof)
                
                # Determine the file path based on the protocol file
                if "train" in protocol_file:
                    subset_dir = "ASVspoof2019_LA_train/flac"
                elif "dev" in protocol_file:
                    subset_dir = "ASVspoof2019_LA_dev/flac"
                else:  # eval
                    subset_dir = "ASVspoof2019_LA_eval/flac"
                
                audio_path = os.path.join(root_dir, subset_dir, f"{file_id}.flac")
                
                # Check if file exists
                if os.path.exists(audio_path):
                    self.samples.append((audio_path, 1 if label == "bonafide" else 0))
                else:
                    self.bad_files.append((audio_path, "File not found"))
        
        print(f"Found {len(self.samples)} samples in protocol file")
        if self.bad_files:
            print(f"Warning: {len(self.bad_files)} files could not be found")
            if len(self.bad_files) <= 5:
                for path, reason in self.bad_files:
                    print(f" - {path}: {reason}")
        
        # Create feature extractors
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,  # ASVspoof typically uses 16kHz
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Fix length to 4 seconds (64000 samples at 16kHz)
            target_length = 4 * sample_rate
            if waveform.shape[1] < target_length:
                # Pad if too short
                padding = target_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            elif waveform.shape[1] > target_length:
                # Crop if too long
                waveform = waveform[:, :target_length]
            
            # Generate mel spectrogram
            mel_spectrogram = self.mel_spectrogram(waveform)
            mel_spectrogram = self.amplitude_to_db(mel_spectrogram)
            
            # Ensure fixed size (224 is a common input size for ResNet)
            target_width = 224
            current_width = mel_spectrogram.shape[2]
            
            if current_width < target_width:
                # Pad if too short
                padding = target_width - current_width
                mel_spectrogram = F.pad(mel_spectrogram, (0, padding))
            elif current_width > target_width:
                # Crop if too long
                mel_spectrogram = mel_spectrogram[:, :, :target_width]
            
            # Normalize the spectrogram
            mean = mel_spectrogram.mean()
            std = mel_spectrogram.std()
            mel_spectrogram = (mel_spectrogram - mean) / (std + 1e-9)
            
            if self.transform:
                mel_spectrogram = self.transform(mel_spectrogram)
            
            return mel_spectrogram, label
            
        except Exception as e:
            print(f"Error processing file {audio_path}: {str(e)}")
            # Return a zero tensor of the expected shape and the correct label
            return torch.zeros(1, 64, 224), label

def train(model, train_loader, criterion, optimizer, epoch, device, memory_tracking=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        
        loss.backward()
        optimizer.step()
        
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        

        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (progress_bar.n + 1),
            'acc': 100. * correct / total,

        })
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    

    
    return train_loss, train_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_scores = []  # For EER calculation
    all_labels = []
    
    
    # Track forward time
    total_forward_time = 0.0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Time forward pass
            forward_start = time.time()
            outputs = model(inputs)
            forward_end = time.time()
            total_forward_time += (forward_end - forward_start)
            
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            
            # Get prediction scores
            scores = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probability for the 'bonafide' class
            
            # Get predicted classes
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_scores.extend(scores)
            all_labels.extend(labels.cpu().numpy())
            
            
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Compute EER
    eer, eer_threshold = calculate_eer(all_labels, all_scores)
    

    return val_loss, val_acc, conf_matrix, eer, eer_threshold

def plot_metrics(metrics_dict, save_path='metrics_comparison.png'):
    """Plot comparison of metrics across all models"""
    # Define colors for each model
    colors = {
        'baseline': 'blue',
        'dropin_frozen': 'red',
        'dropin_unfrozen': 'green'
    }
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training loss
    ax = axs[0, 0]
    for model_name, metrics in metrics_dict.items():
        epochs = range(metrics['start_epoch'], metrics['start_epoch'] + len(metrics['train_loss']))
        ax.plot(epochs, metrics['train_loss'], color=colors.get(model_name, 'gray'), label=model_name)
    ax.set_title('Training Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    
    # Validation loss
    ax = axs[0, 1]
    for model_name, metrics in metrics_dict.items():
        epochs = range(metrics['start_epoch'], metrics['start_epoch'] + len(metrics['val_loss']))
        ax.plot(epochs, metrics['val_loss'], color=colors.get(model_name, 'gray'), label=model_name)
    ax.set_title('Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    
    # Training accuracy
    ax = axs[1, 0]
    for model_name, metrics in metrics_dict.items():
        epochs = range(metrics['start_epoch'], metrics['start_epoch'] + len(metrics['train_acc']))
        ax.plot(epochs, metrics['train_acc'], color=colors.get(model_name, 'gray'), label=model_name)
    ax.set_title('Training Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    
    # Validation accuracy
    ax = axs[1, 1]
    for model_name, metrics in metrics_dict.items():
        epochs = range(metrics['start_epoch'], metrics['start_epoch'] + len(metrics['val_acc']))
        ax.plot(epochs, metrics['val_acc'], color=colors.get(model_name, 'gray'), label=model_name)
    ax.set_title('Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(conf_matrix, save_path='confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Spoof', 'Bonafide'], yticklabels=['Spoof', 'Bonafide'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_parameters_from_baseline(baseline_model, target_model):
    """
    Load parameters from baseline model to target model, only copying parameters with matching shapes.
    Returns the number of parameters loaded and skipped.
    """
    baseline_state = baseline_model.state_dict()
    target_state = target_model.state_dict()
    
    # Create a dictionary only for shape-matching parameters
    compatible_state = {}
    skipped_params = []
    
    for key in target_state:
        if key in baseline_state and target_state[key].shape == baseline_state[key].shape:
            compatible_state[key] = baseline_state[key]
        elif key in baseline_state:
            skipped_params.append((key, baseline_state[key].shape, target_state[key].shape))
    
    # Load the compatible parameters
    target_model.load_state_dict(compatible_state, strict=False)
    
    # Print parameter loading summary
    print(f"Loaded {len(compatible_state)} parameters from baseline model")
    print(f"Skipped {len(skipped_params)} parameters due to shape mismatch")
    
    # Print details of a few skipped parameters
    if skipped_params:
        print("Examples of skipped parameters:")
        for i, (name, baseline_shape, target_shape) in enumerate(skipped_params[:5]):
            print(f"  {name}: baseline shape {baseline_shape}, target shape {target_shape}")
    
    return len(compatible_state), len(skipped_params)



def run_experiment(experiment_name, model, train_loader, val_loader, test_loader, num_epochs, 
                   learning_rate=0.001, start_epoch=1, return_model=False):
    """Run a complete training experiment and return metrics"""
    print(f"\n{'-'*20} Running {experiment_name} experiment {'-'*20}")
    
    # Display model parameters
    param_stats = model.get_param_stats() if hasattr(model, 'get_param_stats') else {
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    print(f"Model parameters: {param_stats['total_params']:,}")
    print(f"Trainable parameters: {param_stats.get('trainable_params', param_stats['total_params']):,}")
    
    # Loss function and optimizer - only optimize trainable parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # Track metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_eers = []

    # Clear GPU cache before starting
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    best_eer = float('inf')
    best_model_state = None

    # Training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Train
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
      
        
        # Validate
        val_loss, val_acc, conf_matrix, eer, eer_threshold = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_eers.append(eer)

        if eer < best_eer:
            best_eer = eer
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, f'{experiment_name}_best_model.pth')
            print(f'>> Saved new best model at epoch {epoch} with EER={eer:.4f}')
        
       
        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, '
              f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, '
              f'EER={eer:.4f}, '
              )
        
        # Save latest confusion matrix
        plot_confusion_matrix(conf_matrix, save_path=f'{experiment_name}_confusion_matrix_epoch_{epoch}.png')
  
    # Plot and save metrics
    plot_metrics(
        {experiment_name: {
            'train_loss': train_losses,
            'train_acc': train_accs,
            'val_loss': val_losses,
            'val_acc': val_accs,
            'start_epoch': start_epoch
        }}, 
        save_path=f'{experiment_name}_metrics.png'
    )

    # Load best model for test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model based on dev EER for final testing.")

    test_loss, test_acc, test_conf_matrix, test_eer, test_eer_threshold = validate(model, test_loader, criterion, device)

    print(f'Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%, Test EER={test_eer:.4f}')
    plot_confusion_matrix(test_conf_matrix, save_path=f'{experiment_name}_confusion_matrix_test.png')
    
    # Save model
    torch.save(model.state_dict(), f'{experiment_name}_model.pth')
    
    # Save metrics
    metrics = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs,
        'val_eer': val_eers,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'test_eer': test_eer,
        
    }
    
    # Create a metrics DataFrame
    metrics_df = pd.DataFrame({
        'epoch': range(start_epoch, start_epoch + num_epochs),
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs,
        'val_eer': val_eers,
        
    })
    metrics_df.to_csv(f'{experiment_name}_metrics.csv', index=False)
    
    # Print summary
    print(f"\n{experiment_name} Summary:")
    print(f"Best validation accuracy: {max(val_accs):.2f}%")
    print(f"Best validation EER: {min(val_eers):.4f}")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Final test EER: {test_eer:.4f}")

    
    if return_model:
        return metrics, model
    else:
        return metrics




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
    learning_rate = 0.001
    
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
        
        baseline_model = ResNet18(input_c=1, dropin=False, num_classes=2).to(device)
        _, baseline_model = run_experiment(
            'baseline', 
            baseline_model, 
            train_loader, 
            val_loader, 
            test_loader, 
            num_epochs // 3, 
            learning_rate,
            start_epoch=1,
            return_model=True
        )


        
        dropin_model = ResNet18(input_c=1, dropin=True, num_classes=2).to(device)
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