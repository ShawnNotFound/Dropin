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

from dataset import get_gpu_memory_usage, calculate_eer
from lc_grnn import SVMLoss, MaskedOptimizer

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
        
        # Time forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Time backward pass
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

    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Time forward pass
            outputs = model(inputs)
            
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


def run_experiment(experiment_name, model, train_loader, val_loader, test_loader, num_epochs, 
                   learning_rate=0.001, start_epoch=1, return_model=False):
    """Run a complete training experiment and return metrics"""
    print(f"\n{'-'*20} Running {experiment_name} experiment {'-'*20}")


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

    # Clear GPU cache before starting
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    best_val_eer = float('inf')

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
       
        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, '
              f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, '
              f'EER={eer:.4f}')

        if eer < best_val_eer:
            best_val_eer = eer
            torch.save(model.state_dict(), f'{experiment_name}_best.pth')
            print(f"  => Saved best model at epoch {epoch} with EER={eer:.4f}")
        
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