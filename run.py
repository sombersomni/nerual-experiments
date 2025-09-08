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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ImageTransformer(nn.Module):
    def __init__(self, input_size=28, num_channels=1, patch_size=4, embed_dim=128, num_heads=8, num_layers=6, num_classes=10):
        super(ImageTransformer, self).__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (input_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_size * patch_size * num_channels, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, self.num_patches + 1)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Embedding head (for triplet loss)
        self.embedding_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 128),  # Final embedding dimension
            nn.LayerNorm(128)
        )
        
        # Classification head (separate MLP)
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Convert image to patches
        x = x.view(batch_size, self.num_channels, self.input_size, self.input_size)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_size * self.patch_size * self.num_channels)
        
        # Patch embedding
        x = self.patch_embed(patches)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer
        x = self.transformer(x)
        
        # Use class token for embedding
        cls_output = x[:, 0]  # First token is class token
        
        # Get embeddings
        embeddings = self.embedding_head(cls_output)
        
        # Classification
        logits = self.classifier(embeddings)
        
        return embeddings, logits

class TripletLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


def train_model(model, train_loader, val_loader, config, num_epochs=1):
    model = model.to(device)
    
    # Get training parameters from config
    training_config = config['training']
    triplet_loss = TripletLoss(margin=training_config['triplet_margin'])
    classification_loss = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=training_config['learning_rate'], 
        weight_decay=training_config['weight_decay']
    )
    scheduler = get_scheduler(optimizer, config)
    
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_triplet_loss = 0
        epoch_class_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (anchor, positive, negative, labels, _) in enumerate(train_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Get embeddings and predictions for triplet
            anchor_emb, anchor_logits = model(anchor)
            positive_emb, _ = model(positive)
            negative_emb, _ = model(negative)
            
            # Compute losses
            triplet_l = triplet_loss(anchor_emb, positive_emb, negative_emb)
            class_l = classification_loss(anchor_logits, labels)
            
            # Combined loss
            total_loss = triplet_l + training_config['classifier_weight'] * class_l
            
            total_loss.backward()
            optimizer.step()
            
            epoch_triplet_loss += triplet_l.item()
            epoch_class_loss += class_l.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(anchor_logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % config['logging']['print_frequency'] == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Triplet Loss: {triplet_l.item():.4f}, Class Loss: {class_l.item():.4f}')
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                data = data.view(data.size(0), -1)
                _, outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_acc = 100 * correct / total
        train_acc = 100 * train_correct / train_total
        
        val_accuracies.append(val_acc)
        train_accuracies.append(train_acc)
        train_losses.append(epoch_triplet_loss / len(train_loader))
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Triplet Loss: {epoch_triplet_loss/len(train_loader):.4f}')
        print(f'  Classification Loss: {epoch_class_loss/len(train_loader):.4f}')
        print(f'  Training Accuracy: {train_acc:.2f}%')
        print(f'  Validation Accuracy: {val_acc:.2f}%')
        if train_acc > val_acc + 5:  # Simple overfitting check
            print(f'  ⚠️  Potential overfitting detected (train acc {train_acc:.1f}% > val acc {val_acc:.1f}%)')
        print('-' * 50)
    
    return train_losses, train_accuracies, val_accuracies


def visualize_embeddings(model, test_loader, num_samples=1000):
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for i, (data, targets) in enumerate(test_loader):
            if len(embeddings) >= num_samples:
                break
            
            data = data.to(device)
            data = data.view(data.size(0), -1)
            emb, _ = model(data)
            
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
                   c=[colors[i]], label=f'Digit {i}', alpha=0.7, s=20)
    
    plt.legend()
    plt.title('t-SNE Visualization of Learned Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    plt.savefig('embeddings_tsne.png')

def load_model(model_path, device='cpu'):
    """
    Load a saved model from a .pth file.
    
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
    model = ImageTransformer(
        input_size=dataset_config.input_size,
        num_channels=dataset_config.num_channels,
        patch_size=model_config['patch_size'],
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        num_classes=dataset_config.num_classes
    )
    
    print(f"Model created for {dataset_config.name} dataset!")
    print(f"Configuration: {args.config}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Scheduler: {config['training']['scheduler']['type']}")
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    train_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, config, num_epochs=args.epochs
    )
    
    # Plot training progress
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Triplet Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(val_accuracies, label='Validation Accuracy', color='red')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    accuracy_gap = [train - val for train, val in zip(train_accuracies, val_accuracies)]
    plt.plot(accuracy_gap, color='orange')
    plt.title('Overfitting Gap (Train - Val Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Difference (%)')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Overfitting threshold')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_progress_{dataset_config.name.lower()}.png')

    # Visualize learned embeddings (if enabled)
    if config['logging']['save_embeddings_visualization']:
        print("Visualizing embeddings...")
        visualize_embeddings(model, test_loader)
    
    # Final test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            data = data.view(data.size(0), -1)
            _, outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    final_accuracy = 100 * correct / total
    print(f'Final Test Accuracy: {final_accuracy:.2f}%')
    
    # Save model weights
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'models/image_transformer_{dataset_config.name.lower()}_{args.epochs}epochs_{final_accuracy:.1f}acc_{timestamp}.pth'
    
    # Save model state dict and training info
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': dataset_config.input_size,
            'num_channels': dataset_config.num_channels,
            'patch_size': model_config['patch_size'],
            'embed_dim': model_config['embed_dim'],
            'num_heads': model_config['num_heads'],
            'num_layers': model_config['num_layers'],
            'num_classes': dataset_config.num_classes
        },
        'dataset': dataset_config.name,
        'epochs': args.epochs,
        'final_accuracy': final_accuracy,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'timestamp': timestamp
    }, model_filename)
    
    print(f'Model saved as: {model_filename}')
    
    # Also save the best model (if this is the first or better than existing)
    best_model_filename = f'models/best_{dataset_config.name.lower()}_model.pth'
    save_as_best = True
    
    if os.path.exists(best_model_filename):
        best_model = torch.load(best_model_filename, map_location=device)
        if best_model['final_accuracy'] >= final_accuracy:
            save_as_best = False
    
    if save_as_best:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': dataset_config.input_size,
                'num_channels': dataset_config.num_channels,
                'patch_size': model_config['patch_size'],
                'embed_dim': model_config['embed_dim'],
                'num_heads': model_config['num_heads'],
                'num_layers': model_config['num_layers'],
                'num_classes': dataset_config.num_classes
            },
            'dataset': dataset_config.name,
            'epochs': args.epochs,
            'final_accuracy': final_accuracy,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'timestamp': timestamp
        }, best_model_filename)
        print(f'New best model saved as: {best_model_filename} (accuracy: {final_accuracy:.2f}%)')
    else:
        print(f'Current model accuracy ({final_accuracy:.2f}%) did not exceed best model accuracy ({best_model["final_accuracy"]:.2f}%)')