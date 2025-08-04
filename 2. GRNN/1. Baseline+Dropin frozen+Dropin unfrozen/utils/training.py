# utils/training.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import gc

from utils.dataset import get_gpu_memory_usage, calculate_eer
from models.lc_grnn import SVMLoss, MaskedOptimizer

def train(model, train_loader, criterion, optimizer, epoch, device, memory_tracking=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    # Track max memory usage
    max_memory = 0
    memory_per_batch = []
    
    # Timing variables
    total_forward_time = 0.0
    total_backward_time = 0.0
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Time forward pass
        forward_start = time.time()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        forward_end = time.time()
        forward_time = forward_end - forward_start
        total_forward_time += forward_time
        
        # Time backward pass
        backward_start = time.time()
        loss.backward()
        optimizer.step()
        backward_end = time.time()
        backward_time = backward_end - backward_start
        total_backward_time += backward_time
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Track memory
        if memory_tracking and device.type == 'cuda':
            memory = get_gpu_memory_usage()
            memory_per_batch.append(memory)
            max_memory = max(max_memory, memory)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (progress_bar.n + 1),
            'acc': 100. * correct / total,
            'memory': f'{get_gpu_memory_usage():.1f}MB' if device.type == 'cuda' else 'N/A',
            'fwd': f'{forward_time*1000:.1f}ms',
            'bwd': f'{backward_time*1000:.1f}ms'
        })
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    # Average timing per batch
    avg_forward_time = total_forward_time / len(train_loader)
    avg_backward_time = total_backward_time / len(train_loader)
    
    memory_stats = {
        'max': max_memory,
        'avg': np.mean(memory_per_batch) if memory_per_batch else 0,
        'per_batch': memory_per_batch
    } if memory_tracking else None
    
    timing_stats = {
        'forward_total': total_forward_time,
        'backward_total': total_backward_time,
        'forward_avg': avg_forward_time,
        'backward_avg': avg_backward_time
    }
    
    return train_loss, train_acc, memory_stats, timing_stats


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_scores = []  # For EER calculation
    all_labels = []
    
    # Track memory
    max_memory = 0
    
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
            
            # Track memory
            if device.type == 'cuda':
                memory = get_gpu_memory_usage()
                max_memory = max(max_memory, memory)
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Compute EER
    eer, eer_threshold = calculate_eer(all_labels, all_scores)
    
    # Average forward time per batch
    avg_forward_time = total_forward_time / len(val_loader)
    
    timing_stats = {
        'forward_total': total_forward_time,
        'forward_avg': avg_forward_time
    }
    
    return val_loss, val_acc, conf_matrix, eer, eer_threshold, max_memory, timing_stats


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


def plot_timing_comparison(metrics_dict, save_path='timing_comparison.png'):
    """Plot forward and backward pass timing comparison across all models"""
    # Setup
    models = list(metrics_dict.keys())
    
    # Check if we have timing data
    if 'timing' not in metrics_dict[models[0]]:
        print("No timing data available for plotting")
        return
    
    # Create figure with subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract timing data
    forward_times = []
    backward_times = []
    total_times = []
    
    for model_name in models:
        if 'timing' in metrics_dict[model_name]:
            forward_times.append(metrics_dict[model_name]['timing']['forward_avg'] * 1000)  # Convert to ms
            backward_times.append(metrics_dict[model_name]['timing'].get('backward_avg', 0) * 1000)  # Convert to ms
            total_times.append((metrics_dict[model_name]['timing']['forward_avg'] + 
                              metrics_dict[model_name]['timing'].get('backward_avg', 0)) * 1000)  # Convert to ms
    
    # Bar chart for forward and backward times
    x = np.arange(len(models))
    width = 0.35
    
    # Forward vs backward time comparison
    ax = axs[0]
    ax.bar(x - width/4, forward_times, width/2, label='Forward')
    ax.bar(x + width/4, backward_times, width/2, label='Backward')
    ax.set_title('Forward vs Backward Pass Time')
    ax.set_xlabel('Model')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # Total time comparison
    ax = axs[1]
    ax.bar(x, total_times, width)
    ax.set_title('Total Pass Time (Forward + Backward)')
    ax.set_xlabel('Model')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_memory_usage(metrics_dict, save_path='memory_comparison.png'):
    """Plot memory usage across all models"""
    # Define colors for each model
    colors = {
        'baseline': 'blue',
        'dropin_frozen': 'red',
        'dropin_unfrozen': 'green'
    }
    
    plt.figure(figsize=(12, 6))
    
    # Bar chart for max memory usage
    models = []
    max_memory = []
    avg_memory = []
    
    for model_name, metrics in metrics_dict.items():
        if 'memory' in metrics:
            models.append(model_name)
            max_memory.append(metrics['memory']['max'])
            avg_memory.append(metrics['memory']['avg'])
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, max_memory, width, label='Max Memory', color=[colors.get(m, 'gray') for m in models])
    plt.bar(x + width/2, avg_memory, width, label='Avg Memory', color=[colors.get(m, 'gray') for m in models], alpha=0.7)
    
    plt.xlabel('Model')
    plt.ylabel('Memory Usage (MB)')
    plt.title('GPU Memory Usage Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(conf_matrix, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
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
    Load parameters from baseline model to target model, properly handling shape differences
    for dropin architecture by copying partial tensors.
    """
    baseline_state = baseline_model.state_dict()
    target_state = target_model.state_dict()
    
    # Track statistics
    params_loaded_fully = 0
    params_loaded_partially = 0
    params_skipped = 0
    
    # For each parameter in the target model
    for key in target_state:
        if key in baseline_state:
            baseline_param = baseline_state[key]
            target_param = target_state[key]
            
            if baseline_param.shape == target_param.shape:
                # If shapes match exactly, copy the whole parameter
                target_state[key] = baseline_param.clone()
                params_loaded_fully += 1
            else:
                # For parameters with different shapes, try partial copying
                try:
                    if len(baseline_param.shape) == 2:  # For weight matrices (2D)
                        # Copy the portion that fits (e.g., for layer growing)
                        rows = min(baseline_param.shape[0], target_param.shape[0])
                        cols = min(baseline_param.shape[1], target_param.shape[1])
                        target_param[:rows, :cols] = baseline_param[:rows, :cols]
                        target_state[key] = target_param
                        params_loaded_partially += 1
                    elif len(baseline_param.shape) == 1:  # For bias vectors (1D)
                        # Copy the portion that fits
                        size = min(baseline_param.shape[0], target_param.shape[0])
                        target_param[:size] = baseline_param[:size]
                        target_state[key] = target_param
                        params_loaded_partially += 1
                    else:
                        # Skip parameters with other dimensionality
                        params_skipped += 1
                        print(f"  Skipped {key}: unsupported tensor dimensions {baseline_param.shape}")
                except Exception as e:
                    params_skipped += 1
                    print(f"  Error copying {key}: {str(e)}")
        else:
            params_skipped += 1
    
    # Load the modified state dictionary
    target_model.load_state_dict(target_state)
    
    # Print parameter loading summary
    print(f"Parameter loading summary:")
    print(f"  Fully loaded parameters: {params_loaded_fully}")
    print(f"  Partially loaded parameters: {params_loaded_partially}")
    print(f"  Skipped parameters: {params_skipped}")
    
    # Return stats for reporting - make it compatible with the original function
    total_loaded = params_loaded_fully + params_loaded_partially
    return total_loaded, params_skipped

def run_experiment(experiment_name, model, train_loader, val_loader, test_loader, num_epochs, 
                   learning_rate=0.001, start_epoch=1, return_model=False):
    """Run a complete training experiment and return metrics"""
    print(f"\n{'-'*20} Running {experiment_name} experiment {'-'*20}")
    
    # Display model parameters
    if hasattr(model, 'get_param_stats'):
        print("using first way to get parameter stats")
    else:
        print("using second way to get parameter stats")

    param_stats = model.get_param_stats() if hasattr(model, 'get_param_stats') else {
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'frozen_params': sum(p.numel() for p in model.parameters() if not p.requires_grad)
    }

    # param_stats = {
    #     'total_params': sum(p.numel() for p in model.parameters()),
    #     'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
    #     'frozen_params': sum(p.numel() for p in model.parameters() if not p.requires_grad)
    # }

    print(f"Model parameters: {param_stats['total_params']:,}")
    print(f"Trainable parameters: {param_stats.get('trainable_params', param_stats['total_params']):,}")
    print(f"Frozen parameters: {param_stats.get('frozen_params', 0):,}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loss function - use SVM loss instead of CrossEntropyLoss
    criterion = SVMLoss(margin=1.0)
    
    # Optimizer with gradient mask support
    base_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = MaskedOptimizer(base_optimizer, model)
    
    # Track metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_eers = []
    memory_usage = {'max': 0, 'avg': 0, 'per_epoch': []}
    
    # Track timing for forward and backward passes
    timing_stats = {
        'forward_total': [],
        'backward_total': [],
        'forward_avg': [],
        'backward_avg': [],
        'val_forward_avg': []
    }
    
    # Start timing
    start_time = time.time()
    
    # Clear GPU cache before starting
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    best_val_eer = float('inf')

    # Training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Train
        train_loss, train_acc, memory_stats, train_timing = train(model, train_loader, criterion, optimizer, epoch, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Update timing stats
        timing_stats['forward_total'].append(train_timing['forward_total'])
        timing_stats['backward_total'].append(train_timing['backward_total'])
        timing_stats['forward_avg'].append(train_timing['forward_avg'])
        timing_stats['backward_avg'].append(train_timing['backward_avg'])
        
        # Update memory stats
        if memory_stats:
            memory_usage['max'] = max(memory_usage['max'], memory_stats['max'])
            memory_usage['per_epoch'].append(memory_stats['max'])
        
        # Validate
        val_loss, val_acc, conf_matrix, eer, eer_threshold, val_memory, val_timing = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_eers.append(eer)
        
        # Update validation timing stats
        timing_stats['val_forward_avg'].append(val_timing['forward_avg'])
        
        # Update memory stats
        memory_usage['max'] = max(memory_usage['max'], val_memory)
        
        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, '
              f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, '
              f'EER={eer:.4f}, Memory={memory_usage["max"]:.1f}MB, '
              f'Fwd={train_timing["forward_avg"]*1000:.1f}ms, Bwd={train_timing["backward_avg"]*1000:.1f}ms')

        if eer < best_val_eer:
            best_val_eer = eer
            torch.save(model.state_dict(), f'{experiment_name}_best.pth')
            print(f"  => Saved best model at epoch {epoch} with EER={eer:.4f}")
        
        # Save latest confusion matrix
        # plot_confusion_matrix(conf_matrix, save_path=f'{experiment_name}_confusion_matrix_epoch_{epoch}.png')
    
    # End timing
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training finished in {training_time:.2f} seconds")
    
    # Calculate average memory usage
    memory_usage['avg'] = np.mean(memory_usage['per_epoch']) if memory_usage['per_epoch'] else 0
    
    # Calculate average timing stats
    timing_avg = {
        'forward_avg': np.mean(timing_stats['forward_avg']),
        'backward_avg': np.mean(timing_stats['backward_avg']),
        'val_forward_avg': np.mean(timing_stats['val_forward_avg']),
        'total_avg': np.mean(timing_stats['forward_avg']) + np.mean(timing_stats['backward_avg'])
    }
    
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
    
    # Load best validation model before testing
    best_model_path = f'{experiment_name}_best.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path} for final testing")
    else:
        print(f"Best model not found at {best_model_path}, using last epoch model")

    # Test the model
    test_loss, test_acc, test_conf_matrix, test_eer, test_eer_threshold, test_memory, test_timing = validate(model, test_loader, criterion, device)

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
        'training_time': training_time,
        'time_per_epoch': training_time / num_epochs,
        'param_count': param_stats['total_params'],
        'trainable_param_count': param_stats.get('trainable_params', param_stats['total_params']),
        'memory': memory_usage,
        'timing': timing_avg,
        'timing_data': timing_stats,
        'start_epoch': start_epoch
    }
    
    # Create a metrics DataFrame
    metrics_df = pd.DataFrame({
        'epoch': range(start_epoch, start_epoch + num_epochs),
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs,
        'val_eer': val_eers,
        'forward_time_ms': [t * 1000 for t in timing_stats['forward_avg']],
        'backward_time_ms': [t * 1000 for t in timing_stats['backward_avg']],
        'val_forward_time_ms': [t * 1000 for t in timing_stats['val_forward_avg']],
        'memory': memory_usage['per_epoch'] if memory_usage['per_epoch'] else [0] * num_epochs
    })
    metrics_df.to_csv(f'{experiment_name}_metrics.csv', index=False)
    
    # Print summary
    print(f"\n{experiment_name} Summary:")
    print(f"Best validation accuracy: {max(val_accs):.2f}%")
    print(f"Best validation EER: {min(val_eers):.4f}")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Final test EER: {test_eer:.4f}")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Average time per epoch: {training_time/num_epochs:.2f} seconds")
    print(f"Average forward pass time: {timing_avg['forward_avg']*1000:.2f} ms")
    print(f"Average backward pass time: {timing_avg['backward_avg']*1000:.2f} ms")
    print(f"Average validation forward pass time: {timing_avg['val_forward_avg']*1000:.2f} ms")
    print(f"Parameter count: {param_stats['total_params']:,}")
    print(f"Trainable parameter count: {param_stats.get('trainable_params', param_stats['total_params']):,}")
    print(f"Max GPU memory usage: {memory_usage['max']:.1f}MB")
    print(f"Avg GPU memory usage: {memory_usage['avg']:.1f}MB")
    
    if return_model:
        return metrics, model
    else:
        return metrics