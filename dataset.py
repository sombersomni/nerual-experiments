import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import random
from collections import defaultdict


class TripletDataset(Dataset):
    """
    Generic triplet dataset that works with any image dataset.
    Creates triplets of (anchor, positive, negative) samples for triplet loss training.
    """
    def __init__(self, dataset, num_classes, augment=True):
        print(f"Initializing TripletDataset with {len(dataset)} samples and {num_classes} classes...")
        self.dataset = dataset
        self.num_classes = num_classes
        self.augment = augment
        
        # Create label to indices mapping
        self.label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            self.label_to_indices[label].append(idx)
        
        # Augmentation transforms - these work for any grayscale/RGB image
        self.augment_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        anchor_img, anchor_label = self.dataset[idx]
        
        # Get positive sample (same class)
        positive_idx = random.choice(self.label_to_indices[anchor_label])
        positive_img, _ = self.dataset[positive_idx]
        
        # Get negative sample (different class)
        available_labels = [l for l in range(self.num_classes) if l != anchor_label]
        negative_label = random.choice(available_labels)
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_img, _ = self.dataset[negative_idx]
        
        # Apply augmentation
        if self.augment:
            if random.random() > 0.5:
                anchor_img = self.augment_transform(anchor_img)
            if random.random() > 0.5:
                positive_img = self.augment_transform(positive_img)
        
        # Convert to tensor and flatten
        anchor_img = anchor_img.flatten()
        positive_img = positive_img.flatten()
        negative_img = negative_img.flatten()
        
        return anchor_img, positive_img, negative_img, anchor_label, negative_label


class DatasetConfig:
    """Configuration class for different datasets"""
    def __init__(self, name, num_classes, input_size, num_channels=1, download_kwargs=None):
        self.name = name
        self.num_classes = num_classes
        self.input_size = input_size
        self.num_channels = num_channels
        self.download_kwargs = download_kwargs or {}


# Predefined dataset configurations
DATASET_CONFIGS = {
    'mnist': DatasetConfig(
        name='MNIST',
        num_classes=10,
        input_size=28,
        num_channels=1,
        download_kwargs={'root': './data', 'download': True}
    ),
    'cifar10': DatasetConfig(
        name='CIFAR10', 
        num_classes=10,
        input_size=32,
        num_channels=3,
        download_kwargs={'root': './data', 'download': True}
    ),
    'fashionmnist': DatasetConfig(
        name='FashionMNIST',
        num_classes=10,
        input_size=28,
        num_channels=1,
        download_kwargs={'root': './data', 'download': True}
    ),
}


def get_dataset_loaders(dataset_name='mnist', batch_size=64, num_workers=2, val_split=0.1):
    """
    Generic function to load any supported dataset and create train/val/test loaders.
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'cifar10', 'fashionmnist')
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        val_split: Fraction of training data to use for validation (default: 0.1)
        
    Returns:
        tuple: (train_loader, test_loader, val_loader, config)
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    
    # Define transforms based on dataset
    if config.num_channels == 1:  # Grayscale
        transform = transforms.Compose([transforms.ToTensor()])
    else:  # RGB
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # Load the appropriate dataset
    dataset_class = getattr(torchvision.datasets, config.name)
    
    full_train_dataset = dataset_class(
        train=True,
        transform=transform,
        **config.download_kwargs
    )
    
    test_dataset = dataset_class(
        train=False,
        transform=transform,
        **config.download_kwargs
    )
    
    # Split training data into train/validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducible splits
    )
    
    # Create triplet dataset from training split only
    triplet_train_dataset = TripletDataset(
        train_dataset, 
        num_classes=config.num_classes,
        augment=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        triplet_train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    # Validation loader - separate from training data
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers
    )
    
    print(f"Dataset: {config.name}")
    print(f"Training samples: {len(train_dataset)} ({100*(1-val_split):.0f}%)")
    print(f"Validation samples: {len(val_dataset)} ({100*val_split:.0f}%)")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Input size: {config.input_size}x{config.input_size}")
    print(f"Channels: {config.num_channels}")
    print(f"Classes: {config.num_classes}")
    
    return train_loader, test_loader, val_loader, config


def visualize_triplet_samples(triplet_dataset, config, num_samples=5):
    """
    Visualize anchor, positive, and negative samples to verify triplet matching.
    Works with any dataset configuration.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(6, 2*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        anchor, positive, negative, anchor_label, negative_label = triplet_dataset[i]
        
        # Reshape back to original image size for visualization
        img_size = config.input_size
        if config.num_channels == 1:
            anchor_img = anchor.reshape(img_size, img_size)
            positive_img = positive.reshape(img_size, img_size)
            negative_img = negative.reshape(img_size, img_size)
            cmap = 'gray'
        else:
            anchor_img = anchor.reshape(config.num_channels, img_size, img_size).permute(1, 2, 0)
            positive_img = positive.reshape(config.num_channels, img_size, img_size).permute(1, 2, 0)
            negative_img = negative.reshape(config.num_channels, img_size, img_size).permute(1, 2, 0)
            cmap = None
        
        # Display anchor
        axes[i, 0].imshow(anchor_img, cmap=cmap)
        axes[i, 0].set_title(f'Anchor\nLabel: {anchor_label}')
        axes[i, 0].axis('off')
        
        # Display positive (should be same class as anchor)
        axes[i, 1].imshow(positive_img, cmap=cmap)
        axes[i, 1].set_title(f'Positive\nLabel: {anchor_label}')
        axes[i, 1].axis('off')
        
        # Display negative (should be different class)
        axes[i, 2].imshow(negative_img, cmap=cmap)
        axes[i, 2].set_title(f'Negative\nLabel: {negative_label}')
        axes[i, 2].axis('off')
    
    plt.suptitle(f'Triplet Samples from {config.name}: Anchor - Positive - Negative', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'triplet_samples_{config.name.lower()}.png')
    print(f"Triplet samples visualization saved as triplet_samples_{config.name.lower()}.png")