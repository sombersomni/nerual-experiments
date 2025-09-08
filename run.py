import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import os
import yaml
import random
from datetime import datetime
from dataset import get_dataset_loaders, visualize_triplet_samples
from models import (
    ImageTransformer, TripletLoss, 
    ImageTransformerLightning, MultiTaskImageTransformerLightning
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"CUDNN Version: {torch.backends.cudnn.version()}")
    torch.set_float32_matmul_precision('medium')

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_random_seeds(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make cudnn deterministic (slower but more reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_scheduler(optimizer, config):
    """Create learning rate scheduler based on config"""
    scheduler_config = config['training']['scheduler']
    scheduler_type = scheduler_config['type']
    
    if scheduler_type == 'StepLR':
        return optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=scheduler_config['step_size'], 
            gamma=scheduler_config['gamma']
        )
    elif scheduler_type == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config['T_max']
        )
    elif scheduler_type == 'ExponentialLR':
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_config['gamma']
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")



def create_lightning_trainer(config, dataset_name, num_epochs):
    """Create PyTorch Lightning trainer with callbacks and logger"""
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=f"img_tasks/{dataset_name}",
        name="experiment",
        version=None
    )
    
    # Create callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"lightning_checkpoints/{dataset_name}",
        filename="{epoch:02d}-{val/accuracy:.2f}",
        monitor="val/accuracy",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val/accuracy",
        min_delta=0.001,
        patience=5,
        mode="max",
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config['logging']['print_frequency'],
        deterministic=False,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    return trainer, checkpoint_callback


def visualize_embeddings(lightning_model, test_loader, num_samples=1000, is_multitask=False):
    lightning_model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for i, (data, targets) in enumerate(test_loader):
            if len(embeddings) >= num_samples:
                break
            
            data = data.to(lightning_model.device)
            data = data.view(data.size(0), -1)
            
            if is_multitask:
                outputs = lightning_model(data, reconstruct=False)
                emb = outputs['embeddings']
            else:
                emb, _ = lightning_model(data)
            
            embeddings.extend(emb.cpu().numpy())
            labels.extend(targets.numpy())
    
    embeddings = np.array(embeddings[:num_samples])
    labels = np.array(labels[:num_samples])
    
    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    
    for i in range(10):
        mask = labels == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[colors[i]], label=f'Class {i}', alpha=0.7, s=20)
    
    plt.legend()
    plt.title('t-SNE Visualization of Learned Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    plt.savefig('embeddings_tsne.png')
    plt.close()

def load_lightning_model(checkpoint_path, model_class=None):
    """
    Load a PyTorch Lightning model from checkpoint.
    
    Args:
        checkpoint_path: Path to the Lightning checkpoint file
        model_class: Lightning model class (ImageTransformerLightning or MultiTaskImageTransformerLightning)
        
    Returns:
        Lightning model instance
    """
    if model_class is None:
        # Try to determine model class from checkpoint
        try:
            model = MultiTaskImageTransformerLightning.load_from_checkpoint(checkpoint_path)
            print("Loaded MultiTaskImageTransformerLightning from checkpoint")
        except:
            model = ImageTransformerLightning.load_from_checkpoint(checkpoint_path)
            print("Loaded ImageTransformerLightning from checkpoint")
    else:
        model = model_class.load_from_checkpoint(checkpoint_path)
        print(f"Loaded {model_class.__name__} from checkpoint")
    
    model.eval()
    return model


def load_model(model_path, device='cpu'):
    """
    Load a saved model from a .pth file (backward compatibility).
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        tuple: (model, checkpoint_info)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration
    model_config = checkpoint['model_config']
    
    # Create model instance
    model = ImageTransformer(
        input_size=model_config['input_size'],
        num_channels=model_config['num_channels'],
        patch_size=model_config['patch_size'],
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        num_classes=model_config['num_classes']
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded model trained on {checkpoint['dataset']} dataset")
    print(f"Training epochs: {checkpoint['epochs']}")
    print(f"Final accuracy: {checkpoint['final_accuracy']:.2f}%")
    print(f"Timestamp: {checkpoint['timestamp']}")
    
    return model, checkpoint

# Main training script
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Image Transformer with Triplet Loss using YAML configs')
    parser.add_argument('--epochs', type=int, default=1, 
                        help='Number of training epochs (default: 1)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed override (overrides config file)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    dataset_name = config['dataset']['name']
    
    # Set random seed for reproducibility
    seed = args.seed if args.seed is not None else config['random_seed']
    set_random_seeds(seed)
    print(f"Random seed set to: {seed}")
    
    # Load dataset with config parameters
    train_loader, test_loader, val_loader, dataset_config = get_dataset_loaders(
        dataset_name=dataset_name,
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers']
    )
    
    # Get triplet dataset from train_loader for visualization
    triplet_dataset = train_loader.dataset
    
    # Visualize triplet samples before training (if enabled)
    if config['logging']['save_triplet_samples']:
        print("Visualizing triplet samples...")
        visualize_triplet_samples(
            triplet_dataset, 
            dataset_config, 
            num_samples=config['logging']['num_visualization_samples']
        )
    
    # Create model with config parameters
    model_config = config['model']
    training_config = config['training']
    
    # Determine which model to use
    use_multitask = training_config.get('use_multitask_loss', False)
    
    if use_multitask:
        print("Using MultiTaskImageTransformerLightning...")
        lightning_model = MultiTaskImageTransformerLightning(config, dataset_config)
    else:
        print("Using ImageTransformerLightning...")
        lightning_model = ImageTransformerLightning(config, dataset_config)
    
    print(f"Model created for {dataset_config.name} dataset!")
    print(f"Configuration: {args.config}")
    print(f"Number of parameters: {sum(p.numel() for p in lightning_model.parameters())}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Scheduler: {config['training']['scheduler']['type']}")
    print(f"Multi-task learning: {use_multitask}")
    
    # Create Lightning trainer
    trainer, checkpoint_callback = create_lightning_trainer(config, dataset_name, args.epochs)
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    trainer.fit(lightning_model, train_loader, val_loader)
    
    # Test the model
    print("Running final test...")
    test_results = trainer.test(lightning_model, test_loader, verbose=True)
    
    # Get the best model
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    
    # Load best model for final evaluation and visualization
    if best_model_path:
        if use_multitask:
            best_model = MultiTaskImageTransformerLightning.load_from_checkpoint(best_model_path)
        else:
            best_model = ImageTransformerLightning.load_from_checkpoint(best_model_path)
        best_model.eval()
    else:
        best_model = lightning_model

    # Visualize learned embeddings (if enabled)
    if config['logging']['save_embeddings_visualization']:
        print("Visualizing embeddings...")
        visualize_embeddings(best_model, test_loader, is_multitask=use_multitask)
    
    # Extract final test accuracy from results
    if test_results:
        final_accuracy = test_results[0]['test/accuracy'] * 100  # Convert to percentage
        print(f'Final Test Accuracy: {final_accuracy:.2f}%')
    else:
        print("No test results available")
        final_accuracy = 0.0
    
    # Lightning automatically handles model checkpointing
    print(f'Best model checkpoint saved at: {best_model_path}')
    print(f'All Lightning logs saved in: img_tasks/{dataset_name}')
    print(f'TensorBoard command: tensorboard --logdir=img_tasks/{dataset_name}')
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Dataset: {dataset_config.name}")
    print(f"Model type: {'MultiTask' if use_multitask else 'Simple'}")
    print(f"Final test accuracy: {final_accuracy:.2f}%")
    print(f"Best model: {best_model_path}")
    print(f"Logs: img_tasks/{dataset_name}")