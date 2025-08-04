# utils/dataset.py
import os
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import roc_curve
import random

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
            
            if self.transform:
                waveform = self.transform(waveform)
            
            return waveform, label
            
        except Exception as e:
            print(f"Error processing file {audio_path}: {str(e)}")
            # Return a zero tensor of the expected shape and the correct label
            return torch.zeros(1, 4 * 16000), label


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
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_gpu_memory_usage():
    """Get GPU memory usage in MB"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        return 0
    
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024 / 1024  # Convert to MB
    except:
        # If pynvml is not available, use torch's memory stats
        return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB